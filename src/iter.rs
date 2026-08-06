//! Iterators over the contents of a [`Cord`].

use core::marker::PhantomData;

use crate::cord::Cord;
use crate::inline_data::InlineData;
use crate::rep::reader::CordRepBtreeReader;
use crate::rep::{CordRepSubstring, MAX_BYTES_TO_COPY, MAX_INLINE, OwnedRep, RepRef, RepView};

/// Iterator over the contiguous chunks of a [`Cord`], created by
/// [`Cord::chunks`].
///
/// Every yielded chunk is non-empty. The iterator holds a navigation stack
/// proportional to the tree height (~150 bytes); prefer passing it by
/// reference.
#[derive(Clone)]
pub struct Chunks<'a> {
    /// A view of the bytes of the current chunk (possibly a suffix of the
    /// current data edge when used by a [`Cursor`]). Empty at the end.
    current_chunk: &'a [u8],
    /// The current leaf if the cord is a single (non-btree) data edge, `None`
    /// otherwise. Used to share memory when reading sub-cords.
    current_leaf: Option<RepRef<'a>>,
    /// Bytes of `current_leaf`'s own data consumed so far, i.e. `current_leaf`'s
    /// data (unadjusted for any `Substring` indirection) sliced from this
    /// offset equals `current_chunk`. Meaningless (and never read) while
    /// `current_leaf` is `None`. A carried counter rather than recovering the
    /// same value later via pointer subtraction between `current_chunk` and
    /// the leaf's base address (provenance-fragile: the two pointers don't
    /// obviously derive from one another at the point of subtraction).
    leaf_offset: usize,
    /// Number of bytes left, counting from the start of `current_chunk`.
    bytes_remaining: usize,
    /// Reader for btree cords; empty otherwise.
    btree_reader: CordRepBtreeReader<'a>,
    _marker: PhantomData<&'a Cord>,
}

impl<'a> Chunks<'a> {
    /// Creates an iterator positioned at the first chunk of `cord`.
    pub(crate) fn new(cord: &'a Cord) -> Self {
        let mut it = Self::empty();
        if let Some(tree) = cord.tree_ref() {
            it.bytes_remaining = tree.len();
            if it.bytes_remaining != 0 {
                it.init_tree(tree);
            }
        } else {
            it.current_chunk = cord.inline_slice();
            it.bytes_remaining = it.current_chunk.len();
        }
        it
    }

    /// An iterator over a single slice.
    pub(crate) fn single(slice: &'a [u8]) -> Self {
        let mut it = Self::empty();
        it.current_chunk = slice;
        it.bytes_remaining = slice.len();
        it
    }

    /// An exhausted iterator.
    pub(crate) fn empty() -> Self {
        Self {
            current_chunk: &[],
            current_leaf: None,
            leaf_offset: 0,
            bytes_remaining: 0,
            btree_reader: CordRepBtreeReader::new(),
            _marker: PhantomData,
        }
    }

    /// Positions `self` at `tree`'s first chunk. Requires `tree`'s length to
    /// be non-zero (checked by the sole caller, `new`, before calling this).
    fn init_tree(&mut self, tree: RepRef<'a>) {
        if let RepView::Btree(btree) = tree.view() {
            self.current_chunk = self.btree_reader.init(btree);
        } else {
            // SUBSTRING, EXTERNAL or FLAT: all are data edges.
            debug_assert!(tree.is_data_edge());
            self.current_leaf = Some(tree);
            self.current_chunk = tree.data();
        }
    }

    /// Bytes remaining, counting from the start of the current chunk.
    #[inline]
    #[allow(dead_code)]
    pub(crate) fn bytes_remaining(&self) -> usize {
        self.bytes_remaining
    }

    /// Moves to the next chunk. Requires `bytes_remaining > 0`.
    #[inline]
    pub(crate) fn step(&mut self) {
        debug_assert!(self.bytes_remaining > 0, "iterating past the end of a Cord");
        debug_assert!(self.bytes_remaining >= self.current_chunk.len());
        self.bytes_remaining -= self.current_chunk.len();
        if self.bytes_remaining > 0 {
            if self.btree_reader.is_some() {
                self.current_chunk = self.btree_reader.next();
                return;
            }
            debug_assert!(!self.current_chunk.is_empty(), "step() on an invalid iterator");
        }
        self.current_chunk = &[];
    }

    /// Drops `n` bytes from the front of the current chunk. Requires
    /// `n < current_chunk.len()`.
    #[inline]
    pub(crate) fn remove_chunk_prefix(&mut self, n: usize) {
        debug_assert!(n < self.current_chunk.len());
        self.current_chunk = &self.current_chunk[n..];
        self.bytes_remaining -= n;
        self.leaf_offset += n;
    }

    /// Skips `n` bytes. Requires `n <= bytes_remaining`.
    pub(crate) fn advance_bytes(&mut self, n: usize) {
        debug_assert!(self.bytes_remaining >= n);
        if n < self.current_chunk.len() {
            self.remove_chunk_prefix(n);
        } else if n != 0 {
            if self.btree_reader.is_some() {
                self.advance_bytes_btree(n);
            } else {
                self.bytes_remaining = 0;
                self.current_chunk = &[];
            }
        }
    }

    fn advance_bytes_btree(&mut self, n: usize) {
        debug_assert!(n >= self.current_chunk.len());
        self.bytes_remaining -= n;
        if self.bytes_remaining != 0 {
            if n == self.current_chunk.len() {
                self.current_chunk = self.btree_reader.next();
            } else {
                // SAFETY: this branch runs only while `bytes_remaining != 0`
                // on a btree-backed iterator, so the reader was initialized
                // (`is_some()`) — `length`/`seek`'s contract.
                unsafe {
                    let offset = self.btree_reader.length() - self.bytes_remaining;
                    self.current_chunk = self.btree_reader.seek(offset);
                }
            }
        } else {
            self.current_chunk = &[];
        }
    }

    /// Reads the next `n` bytes into a new cord, sharing memory with the
    /// iterated cord where possible, and advances past them. Requires
    /// `n <= bytes_remaining`.
    pub(crate) fn read_bytes(&mut self, n: usize) -> Cord {
        debug_assert!(self.bytes_remaining >= n);

        if n <= MAX_INLINE {
            // The range fits inline: flatten it. Gather from a cheap clone
            // (cloning never bumps refcounts or mutates `self`; see
            // `Chunks`' doc) so `fill_inline_from` can consume whole chunks
            // via the ordinary `Iterator` impl, then reposition the real
            // `self` in one step via `advance_bytes`, which already knows
            // how to land inside a chunk without walking it byte by byte.
            let data = InlineData::fill_inline_from(self.clone(), n);
            self.advance_bytes(n);
            return Cord { data };
        }

        if self.btree_reader.is_some() {
            let chunk_size = self.current_chunk.len();
            let subcord = if n <= chunk_size && n <= MAX_BYTES_TO_COPY {
                let subcord = Cord::copy_from_slice(&self.current_chunk[..n]);
                if n < chunk_size {
                    self.current_chunk = &self.current_chunk[n..];
                } else {
                    self.current_chunk = self.btree_reader.next();
                }
                subcord
            } else {
                // SAFETY: this branch is guarded by `btree_reader.is_some()`
                // above — `read`'s non-empty-reader contract.
                let (chunk, tree) = unsafe { self.btree_reader.read(n, chunk_size) };
                self.current_chunk = chunk;
                debug_assert!(tree.is_some(), "read_bytes: n <= bytes_remaining rules out an exceeded read");
                // SAFETY: `n <= self.bytes_remaining` (this fn's precondition,
                // checked above) guarantees this read does not exceed the
                // reader's remaining data, so `CordRepBtreeReader::read`
                // always returns `Some` here (see its doc for when it
                // returns `None`).
                Cord::from_owned_rep(unsafe { tree.unwrap_unchecked() })
            };
            self.bytes_remaining -= n;
            return subcord;
        }

        // A single data edge. `current_leaf` is always set here for a
        // `Chunks` built from a `Cord` (`Chunks::new` -> `init_tree` sets it
        // whenever `btree_reader` stays empty and the cord holds a tree, and
        // nothing clears it afterwards) — the only path that can reach
        // `read_bytes` at all, since `Cursor` (this fn's sole caller via
        // `Cursor::read`) is only ever built over a `Cord` by `Cursor::new`.
        // `Chunks::single` (used by `CordLike::chunks`, e.g. `&[u8]`) leaves
        // `current_leaf` unset, but a `Cursor` is never built over one, so
        // that state can't reach here; `expect` (not the unchecked cousin)
        // keeps that a debug-and-release-checked invariant rather than one
        // provable only by tracing every call site.
        let leaf = self.current_leaf.expect("read_bytes: single data edge with no current_leaf");
        if n == leaf.len() {
            // Reading the entire edge: share it.
            self.bytes_remaining = 0;
            self.current_chunk = &[];
            return Cord::from_owned_rep(leaf.to_owned());
        }
        // A partial substring node: compute the offset into the flat or
        // external payload.
        let (payload, base_offset) = match leaf.view() {
            RepView::Substring { start, child } => (child, start),
            _ => (leaf, 0),
        };
        let offset = base_offset + self.leaf_offset;
        // SAFETY: `payload` is a live flat or external data edge (`leaf`'s
        // own invariant, plus `RepRef::view`'s dispatch, guarantees this:
        // the `Substring` arm's `child` and the fallback arm's `leaf` are
        // never anything else, since `current_leaf` is only ever set to a
        // substring, flat, or external node). `offset`/`n` form a valid,
        // non-empty sub-range of it: `n < leaf.len()` was just established
        // above, and the single-data-edge invariant `bytes_remaining ==
        // current_chunk.len() == leaf.len() - offset` (maintained by
        // `remove_chunk_prefix`/`init_tree`) together with this fn's
        // precondition `n <= bytes_remaining` bounds `offset + n <=
        // leaf.len()`.
        let tree = unsafe { CordRepSubstring::substring(payload.as_ptr(), offset, n) };
        // SAFETY: `substring` (called without `ADOPT`) returns a fresh,
        // independently owned reference, which `OwnedRep::from_raw` adopts.
        let subcord = Cord::from_owned_rep(unsafe { OwnedRep::from_raw(tree) });
        self.bytes_remaining -= n;
        self.current_chunk = &self.current_chunk[n..];
        self.leaf_offset += n;
        subcord
    }
}

impl<'a> Iterator for Chunks<'a> {
    type Item = &'a [u8];

    #[inline]
    fn next(&mut self) -> Option<&'a [u8]> {
        if self.bytes_remaining == 0 {
            return None;
        }
        let chunk = self.current_chunk;
        self.step();
        Some(chunk)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.bytes_remaining == 0 { (0, Some(0)) } else { (1, Some(self.bytes_remaining)) }
    }
}

impl core::iter::FusedIterator for Chunks<'_> {}

impl core::fmt::Debug for Chunks<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Chunks").field("bytes_remaining", &self.bytes_remaining).finish_non_exhaustive()
    }
}

/// Iterator over the bytes of a [`Cord`], created by [`Cord::bytes`].
#[derive(Clone, Debug)]
pub struct Bytes<'a> {
    chunks: Chunks<'a>,
}

impl<'a> Bytes<'a> {
    #[inline]
    pub(crate) fn new(cord: &'a Cord) -> Self {
        Self { chunks: Chunks::new(cord) }
    }
}

impl Iterator for Bytes<'_> {
    type Item = u8;

    #[inline]
    fn next(&mut self) -> Option<u8> {
        if self.chunks.bytes_remaining == 0 {
            return None;
        }
        let byte = self.chunks.current_chunk[0];
        if self.chunks.current_chunk.len() > 1 {
            self.chunks.remove_chunk_prefix(1);
        } else {
            self.chunks.step();
        }
        Some(byte)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.chunks.bytes_remaining, Some(self.chunks.bytes_remaining))
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<u8> {
        if n >= self.chunks.bytes_remaining {
            self.chunks.advance_bytes(self.chunks.bytes_remaining);
            return None;
        }
        self.chunks.advance_bytes(n);
        self.next()
    }

    #[inline]
    fn count(self) -> usize {
        self.chunks.bytes_remaining
    }
}

impl ExactSizeIterator for Bytes<'_> {}
impl core::iter::FusedIterator for Bytes<'_> {}

/// A position inside a [`Cord`] supporting byte-wise and chunk-wise reads,
/// skipping, and sub-cord extraction. Created by [`Cord::cursor`].
///
/// A cursor is cheap to clone. It also implements [`std::io::Read`] and
/// [`std::io::BufRead`] (and `bytes::Buf` with the `bytes` feature).
///
/// ```
/// use cord_rs::Cord;
/// let cord = Cord::from("header:payload");
/// let mut cursor = cord.cursor();
/// let header = cursor.read(7);
/// assert_eq!(header, "header:");
/// assert_eq!(cursor.position(), 7);
/// assert_eq!(cursor.chunk(), b"payload");
/// ```
#[derive(Clone, Debug)]
pub struct Cursor<'a> {
    chunks: Chunks<'a>,
    len: usize,
}

impl<'a> Cursor<'a> {
    #[inline]
    pub(crate) fn new(cord: &'a Cord) -> Self {
        Self { chunks: Chunks::new(cord), len: cord.len() }
    }

    /// Number of bytes remaining after the cursor.
    #[inline]
    #[must_use]
    pub fn remaining(&self) -> usize {
        self.chunks.bytes_remaining
    }

    /// Offset of the cursor from the start of the cord.
    #[inline]
    #[must_use]
    pub fn position(&self) -> usize {
        self.len - self.chunks.bytes_remaining
    }

    /// Returns `true` if the cursor is at the end.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.chunks.bytes_remaining == 0
    }

    /// The longest contiguous run of bytes starting at the cursor (empty at
    /// the end).
    #[inline]
    #[must_use]
    pub fn chunk(&self) -> &'a [u8] {
        self.chunks.current_chunk
    }

    /// The byte at the cursor, if any, without advancing.
    #[inline]
    #[must_use]
    pub fn peek(&self) -> Option<u8> {
        self.chunks.current_chunk.first().copied()
    }

    /// Advances the cursor by `n` bytes.
    ///
    /// # Panics
    ///
    /// Panics if `n > remaining()`.
    #[track_caller]
    #[inline]
    pub fn advance(&mut self, n: usize) {
        assert!(
            n <= self.chunks.bytes_remaining,
            "cannot advance past the end of a Cord: n = {n}, remaining = {}",
            self.chunks.bytes_remaining
        );
        self.chunks.advance_bytes(n);
    }

    /// Reads the next `n` bytes into a new cord and advances past them.
    /// The returned cord shares memory with the source where possible.
    ///
    /// # Panics
    ///
    /// Panics if `n > remaining()`.
    #[track_caller]
    pub fn read(&mut self, n: usize) -> Cord {
        assert!(
            n <= self.chunks.bytes_remaining,
            "cannot read past the end of a Cord: n = {n}, remaining = {}",
            self.chunks.bytes_remaining
        );
        self.chunks.read_bytes(n)
    }

    /// Returns the next byte and advances, or `None` at the end.
    #[inline]
    pub fn next_byte(&mut self) -> Option<u8> {
        let byte = self.peek()?;
        if self.chunks.current_chunk.len() > 1 {
            self.chunks.remove_chunk_prefix(1);
        } else {
            self.chunks.step();
        }
        Some(byte)
    }

    /// Returns an iterator over the remaining chunks, starting at the cursor.
    #[inline]
    #[must_use]
    pub fn chunks(&self) -> Chunks<'a> {
        self.chunks.clone()
    }
}

impl Iterator for Cursor<'_> {
    type Item = u8;

    #[inline]
    fn next(&mut self) -> Option<u8> {
        self.next_byte()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining(), Some(self.remaining()))
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<u8> {
        if n >= self.remaining() {
            self.chunks.advance_bytes(self.remaining());
            return None;
        }
        self.chunks.advance_bytes(n);
        self.next_byte()
    }
}

impl ExactSizeIterator for Cursor<'_> {}
impl core::iter::FusedIterator for Cursor<'_> {}
