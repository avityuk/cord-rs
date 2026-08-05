//! Iterators over the contents of a [`Cord`].

use core::marker::PhantomData;

use crate::cord::Cord;
use crate::rep::btree::as_btree;
use crate::rep::reader::CordRepBtreeReader;
use crate::rep::{CordRep, CordRepSubstring, MAX_BYTES_TO_COPY, MAX_INLINE, RepPtr, edge_data, ref_rep};

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
    /// The current leaf if the cord is a single (non-btree) data edge, null
    /// otherwise. Used to share memory when reading sub-cords.
    current_leaf: *mut CordRep,
    /// Number of bytes left, counting from the start of `current_chunk`.
    bytes_remaining: usize,
    /// Reader for btree cords; empty otherwise.
    btree_reader: CordRepBtreeReader,
    _marker: PhantomData<&'a Cord>,
}

// SAFETY: the raw pointers reference immutable, reference counted nodes kept
// alive by the borrowed cord.
unsafe impl Send for Chunks<'_> {}
unsafe impl Sync for Chunks<'_> {}

impl<'a> Chunks<'a> {
    /// Creates an iterator positioned at the first chunk of `cord`.
    pub(crate) fn new(cord: &'a Cord) -> Self {
        let mut it = Self::empty();
        if let Some(tree) = cord.tree() {
            // SAFETY: the tree is kept alive by `cord` for `'a`.
            unsafe {
                it.bytes_remaining = tree.length();
                if it.bytes_remaining != 0 {
                    it.init_tree(tree);
                }
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
            current_leaf: core::ptr::null_mut(),
            bytes_remaining: 0,
            btree_reader: CordRepBtreeReader::new(),
            _marker: PhantomData,
        }
    }

    /// # Safety
    ///
    /// `tree` must be a non-null pointer to a live rep tree that outlives
    /// `self` for `'a` (borrowed, not adopted: this call does not affect
    /// `tree`'s refcount, matching the borrowed-`Cord` lifetime tracked by
    /// `self._marker`).
    unsafe fn init_tree(&mut self, tree: *mut CordRep) {
        // SAFETY: `tree` is live per the caller contract above, which is all
        // `is_btree`, `as_btree`, `btree_reader.init` and `edge_data` require.
        unsafe {
            if tree.is_btree() {
                self.current_chunk = self.btree_reader.init(as_btree(tree));
            } else {
                self.current_leaf = tree;
                self.current_chunk = edge_data(tree);
            }
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
                // SAFETY: the reader references a live tree.
                self.current_chunk = unsafe { self.btree_reader.next() };
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
        // SAFETY: the reader references a live tree.
        unsafe {
            if self.bytes_remaining != 0 {
                if n == self.current_chunk.len() {
                    self.current_chunk = self.btree_reader.next();
                } else {
                    let offset = self.btree_reader.length() - self.bytes_remaining;
                    self.current_chunk = self.btree_reader.seek(offset);
                }
            } else {
                self.current_chunk = &[];
            }
        }
    }

    /// Reads the next `n` bytes into a new cord, sharing memory with the
    /// iterated cord where possible, and advances past them. Requires
    /// `n <= bytes_remaining`.
    pub(crate) fn read_bytes(&mut self, mut n: usize) -> Cord {
        debug_assert!(self.bytes_remaining >= n);
        let mut subcord = Cord::new();

        if n <= MAX_INLINE {
            // The range fits inline: flatten it.
            subcord.data.set_inline_size(n);
            // SAFETY: we copy exactly `n <= 15` bytes into the inline buffer.
            unsafe {
                let mut data = subcord.data.as_chars_mut();
                while n > self.current_chunk.len() {
                    core::ptr::copy_nonoverlapping(
                        self.current_chunk.as_ptr(),
                        data,
                        self.current_chunk.len(),
                    );
                    data = data.add(self.current_chunk.len());
                    n -= self.current_chunk.len();
                    self.step();
                }
                core::ptr::copy_nonoverlapping(self.current_chunk.as_ptr(), data, n);
            }
            if n < self.current_chunk.len() {
                self.remove_chunk_prefix(n);
            } else if n > 0 {
                self.step();
            }
            return subcord;
        }

        if self.btree_reader.is_some() {
            let chunk_size = self.current_chunk.len();
            // SAFETY: the reader references a live tree; `read` returns a
            // new reference.
            unsafe {
                if n <= chunk_size && n <= MAX_BYTES_TO_COPY {
                    subcord = Cord::copy_from_slice(&self.current_chunk[..n]);
                    if n < chunk_size {
                        self.current_chunk = &self.current_chunk[n..];
                    } else {
                        self.current_chunk = self.btree_reader.next();
                    }
                } else {
                    let (chunk, rep) = self.btree_reader.read(n, chunk_size);
                    self.current_chunk = chunk;
                    subcord = Cord::from_rep(rep);
                }
            }
            self.bytes_remaining -= n;
            return subcord;
        }

        // A single data edge.
        debug_assert!(!self.current_leaf.is_null());
        // SAFETY: `current_leaf` is a live data edge.
        unsafe {
            if n == self.current_leaf.length() {
                // Reading the entire edge: share it.
                self.bytes_remaining = 0;
                self.current_chunk = &[];
                return Cord::from_rep(ref_rep(self.current_leaf));
            }
            // A partial substring node: compute the offset into the flat or
            // external payload.
            let payload = if self.current_leaf.is_substring() {
                (*self.current_leaf.cast::<CordRepSubstring>()).child
            } else {
                self.current_leaf
            };
            let base = edge_data(payload).as_ptr();
            let offset = self.current_chunk.as_ptr().addr() - base.addr();
            let tree = CordRepSubstring::substring(payload, offset, n);
            subcord = Cord::from_rep(tree);
        }
        self.bytes_remaining -= n;
        self.current_chunk = &self.current_chunk[n..];
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
