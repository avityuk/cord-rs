//! Iterators over the contents of a [`Cord`]: [`Chunks`] yields contiguous
//! byte slices, [`Bytes`] yields individual bytes, and [`Cursor`] is a
//! positioned reader supporting skipping and seeking, `std::io::Read`/
//! `Seek`, and cheap sub-cord extraction. Not re-exported at the crate root
//! — `Bytes` and `Cursor` would collide with `bytes::Bytes` and
//! `std::io::Cursor`, so (like `str::Bytes`, `slice::Chunks` and
//! `io::Cursor` in `std`) they stay reachable through this module.

use core::marker::PhantomData;

use alloc::boxed::Box;

use crate::cord::Cord;
use crate::inline_data::InlineData;
use crate::rep::reader::CordRepBtreeReader;
use crate::rep::{CordRepSubstring, MAX_BYTES_TO_COPY, MAX_INLINE, OwnedRep, RepRef, RepView};

/// A forward-only chunk position: every field the public [`Chunks`] had
/// before lazy reverse iteration gave it a lazily
/// allocated `back` field. All fields here are `Copy`, so cloning one is a
/// memcpy and dropping it is a no-op — properties `Chunks` itself lost once
/// `back` gave it drop glue and a branching `Clone`.
///
/// Every hot internal path that never needs reverse iteration — [`Cursor`],
/// `find`/`ends_with`/comparisons and sub-cord extraction in `cord.rs` — is
/// built directly on this type instead of the public `Chunks`, so
/// constructing, cloning or dropping one of these positions (which `Cursor`
/// does per search candidate in `find_impl`) stays as cheap as it was before
/// double-ended iteration existed. Kept crate-private: the public iterator
/// remains `Chunks`.
#[derive(Clone)]
pub(crate) struct ForwardChunks<'a> {
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

impl core::fmt::Debug for ForwardChunks<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ForwardChunks")
            .field("bytes_remaining", &self.bytes_remaining)
            .finish_non_exhaustive()
    }
}

// The size `Chunks` itself had before lazy reverse iteration
// added the `back` field. Pointer-width scaled (the navigation stack is
// `MAX_DEPTH` pointers), so only checked on 64-bit; smaller on 32-bit
// targets.
#[cfg(target_pointer_width = "64")]
const _: () = assert!(core::mem::size_of::<ForwardChunks<'static>>() == 176);

// Every field above is `Copy`, so this type has no `Drop` impl and needs
// none — cloning it is a memcpy and dropping it is a no-op. This is exactly
// the property that keeps `Cursor` (which embeds one of these) cheap to
// clone per search candidate in `find_impl`.
const _: () = assert!(!core::mem::needs_drop::<ForwardChunks<'static>>());

#[derive(Clone)]
struct BackChunks<'a> {
    current_chunk: &'a [u8],
    btree_reader: CordRepBtreeReader<'a>,
}

/// Iterator over the contiguous chunks of a [`Cord`], created by
/// [`Cord::chunks`].
///
/// Every yielded chunk is non-empty. The iterator holds a navigation stack
/// proportional to the tree height (184 bytes on 64-bit platforms, less on
/// 32-bit); prefer passing it by reference. Reverse iteration allocates a
/// second navigation stack lazily on the first call to `next_back`.
#[derive(Clone)]
pub struct Chunks<'a> {
    /// The forward position: everything `Chunks` needs until reverse
    /// iteration is actually used.
    front: ForwardChunks<'a>,
    /// Backward position, allocated only when reverse iteration is used.
    back: Option<Box<BackChunks<'a>>>,
}

// Keeps the size claim in the doc comment above honest. Pointer-width
// scaled (the navigation stack is `MAX_DEPTH` pointers), so only checked on
// 64-bit; smaller on 32-bit targets.
#[cfg(target_pointer_width = "64")]
const _: () = assert!(core::mem::size_of::<Chunks<'static>>() == 184);

impl<'a> ForwardChunks<'a> {
    /// Creates a position at the first chunk of `cord`.
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

    /// A position over a single slice.
    pub(crate) fn single(slice: &'a [u8]) -> Self {
        let mut it = Self::empty();
        it.current_chunk = slice;
        it.bytes_remaining = slice.len();
        it
    }

    /// An exhausted position.
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

    /// The longest contiguous run starting at the current position.
    #[inline]
    pub(crate) fn chunk(&self) -> &'a [u8] {
        self.current_chunk
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
            // SAFETY: not `Btree` (matched above) means SUBSTRING, EXTERNAL
            // or FLAT, all data edges (see the comment above).
            self.current_chunk = unsafe { tree.data() };
        }
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
            // A non-btree iterator's single chunk always covers the whole
            // remaining length, so `bytes_remaining` must have reached zero
            // above; reaching this point means the iterator is corrupted.
            // Panic outright (not just in debug) rather than falling through
            // to `current_chunk = &[]`, which would leave `bytes_remaining >
            // 0` forever and make callers loop yielding empty chunks.
            unreachable!("step() on an invalid iterator");
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

    /// Reads the next `n <= MAX_INLINE` bytes into an [`InlineData`],
    /// copying via the branchless [`small_memmove`](crate::rep::small_memmove)
    /// and stepping the underlying reader only past chunks it fully
    /// consumes. `MAINTAIN` selects what happens to the final,
    /// partially-consumed chunk:
    ///
    /// - `true`: the iterator's position is updated to just past the read
    ///   bytes (the [`Cursor`] contract) — `remove_chunk_prefix`/`step` run
    ///   for the final chunk as needed.
    /// - `false`: the final positioning update is skipped entirely; the
    ///   iterator is left mid-chunk with stale bookkeeping and MUST be
    ///   discarded (the `subcord` gather, where the iterator dies
    ///   immediately, measured ~8% faster this way).
    pub(crate) fn gather_inline<const MAINTAIN: bool>(&mut self, n: usize) -> InlineData {
        debug_assert!(n <= crate::rep::MAX_INLINE);
        debug_assert!(n <= self.bytes_remaining);
        let mut out = InlineData::new();
        let mut remaining = n;
        // SAFETY: every copy below writes at most `remaining <= MAX_INLINE`
        // bytes in total into `out`'s zero-initialized 15-byte tail, with
        // `dst` advanced by exactly the bytes already written; each
        // `small_memmove` length is <= 15 (bounded by `remaining`), within
        // its contract. Chunks are live input data for `'a`.
        unsafe {
            let mut dst = out.tail_mut().as_mut_ptr();
            while remaining > self.current_chunk.len() {
                let chunk = self.current_chunk;
                crate::rep::small_memmove::<false>(dst, chunk.as_ptr(), chunk.len());
                dst = dst.add(chunk.len());
                remaining -= chunk.len();
                self.step();
            }
            if remaining != 0 {
                crate::rep::small_memmove::<false>(dst, self.current_chunk.as_ptr(), remaining);
                if MAINTAIN {
                    if remaining < self.current_chunk.len() {
                        self.remove_chunk_prefix(remaining);
                    } else {
                        self.step();
                    }
                }
            }
        }
        out.set_inline_size(n);
        out
    }

    /// Reads the next `n` bytes into a new cord, sharing memory with the
    /// iterated cord where possible, and advances past them. Requires
    /// `n <= bytes_remaining`.
    pub(crate) fn read_bytes(&mut self, n: usize) -> Cord {
        debug_assert!(self.bytes_remaining >= n);

        if n <= MAX_INLINE {
            // The range fits inline: flatten it in one pass.
            return Cord { data: self.gather_inline::<true>(n) };
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
        // `ForwardChunks` built from a `Cord` (`ForwardChunks::new` ->
        // `init_tree` sets it whenever `btree_reader` stays empty and the
        // cord holds a tree, and nothing clears it afterwards) — the only
        // path that can reach `read_bytes` at all, since `Cursor` (this fn's
        // sole caller via `Cursor::read_cord`) is only ever built over a
        // `Cord` by `Cursor::new`. `ForwardChunks::single` (used by
        // `CordLike::chunks`, e.g. `&[u8]`) leaves `current_leaf` unset, but
        // a `Cursor` is never built over one, so that state can't reach
        // here; `expect` (not the unchecked cousin) keeps that a
        // debug-and-release-checked invariant rather than one provable only
        // by tracing every call site.
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

    /// Returns the current chunk and advances past it, or `None` at the
    /// end. The fast forward-only path `Chunks::next` had before lazy
    /// reverse iteration — no `back` to check.
    #[inline]
    pub(crate) fn next_chunk(&mut self) -> Option<&'a [u8]> {
        if self.bytes_remaining == 0 {
            return None;
        }
        let chunk = self.current_chunk;
        self.step();
        Some(chunk)
    }
}

impl<'a> Chunks<'a> {
    /// Creates an iterator positioned at the first chunk of `cord`.
    pub(crate) fn new(cord: &'a Cord) -> Self {
        Self { front: ForwardChunks::new(cord), back: None }
    }

    /// An iterator over a single slice.
    pub(crate) fn single(slice: &'a [u8]) -> Self {
        Self { front: ForwardChunks::single(slice), back: None }
    }

    /// Skips `n` bytes. Requires `n <= bytes_remaining`.
    pub(crate) fn advance_bytes(&mut self, n: usize) {
        if self.back.is_some() {
            self.advance_bytes_between_ends(n);
        } else {
            self.front.advance_bytes(n);
        }
    }

    fn advance_bytes_between_ends(&mut self, mut n: usize) {
        while n != 0 {
            let available = self.front.current_chunk.len().min(self.front.bytes_remaining);
            if n < available {
                self.front.current_chunk = &self.front.current_chunk[n..];
                self.front.bytes_remaining -= n;
                self.front.leaf_offset += n;
                return;
            }
            n -= available;
            self.front.bytes_remaining -= available;
            if self.front.bytes_remaining == 0 {
                self.front.current_chunk = &[];
                return;
            }
            debug_assert_eq!(available, self.front.current_chunk.len());
            self.front.current_chunk = self.front.btree_reader.next();
        }
    }
}

impl<'a> Iterator for Chunks<'a> {
    type Item = &'a [u8];

    #[inline]
    fn next(&mut self) -> Option<&'a [u8]> {
        if self.back.is_none() {
            // The fast path: no second position to meet in the middle with.
            return self.front.next_chunk();
        }
        if self.front.bytes_remaining == 0 {
            return None;
        }
        let n = self.front.current_chunk.len().min(self.front.bytes_remaining);
        let chunk = &self.front.current_chunk[..n];
        self.front.bytes_remaining -= n;
        if self.front.bytes_remaining == 0 {
            self.front.current_chunk = &[];
        } else {
            debug_assert_eq!(n, self.front.current_chunk.len());
            // Bytes remain past a fully consumed chunk, which only a btree
            // can supply: a single-chunk iterator's chunk always covers its
            // whole remaining length. An uninitialized reader would hand
            // back an empty chunk here and leave `bytes_remaining` stuck
            // above zero, so a corrupted iterator panics (as `step` does)
            // rather than yielding empty chunks forever.
            if !self.front.btree_reader.is_some() {
                unreachable!("next() on an invalid iterator");
            }
            self.front.current_chunk = self.front.btree_reader.next();
        }
        Some(chunk)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.front.bytes_remaining == 0 { (0, Some(0)) } else { (1, Some(self.front.bytes_remaining)) }
    }
}

impl<'a> DoubleEndedIterator for Chunks<'a> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a [u8]> {
        if self.front.bytes_remaining == 0 {
            return None;
        }

        let back = self.back.get_or_insert_with(|| {
            let mut reader = CordRepBtreeReader::new();
            let current_chunk = if self.front.btree_reader.is_some() {
                reader.init_last_from(&self.front.btree_reader)
            } else {
                self.front.current_chunk
            };
            Box::new(BackChunks { current_chunk, btree_reader: reader })
        });
        let n = back.current_chunk.len().min(self.front.bytes_remaining);
        let split = back.current_chunk.len() - n;
        let chunk = &back.current_chunk[split..];
        self.front.bytes_remaining -= n;
        if self.front.bytes_remaining == 0 {
            self.front.current_chunk = &[];
            back.current_chunk = &[];
        } else if split == 0 {
            back.current_chunk = back.btree_reader.previous();
        } else {
            back.current_chunk = &back.current_chunk[..split];
        }
        Some(chunk)
    }
}

impl core::iter::FusedIterator for Chunks<'_> {}

impl core::fmt::Debug for Chunks<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Chunks").field("bytes_remaining", &self.front.bytes_remaining).finish_non_exhaustive()
    }
}

/// Iterator over the bytes of a [`Cord`], created by [`Cord::bytes`].
///
/// Per-byte iteration walks `current` — a plain slice iterator over the
/// unyielded tail of the most recently pulled chunk — and touches `chunks`
/// only at chunk boundaries. `Chunks`' per-byte bookkeeping
/// (`bytes_remaining`, `leaf_offset`) exists for `Cursor`, and paying it
/// once per byte measurably slowed whole-cord byte sums.
#[derive(Clone, Debug)]
pub struct Bytes<'a> {
    chunks: Chunks<'a>,
    current: core::slice::Iter<'a, u8>,
    back_current: core::slice::Iter<'a, u8>,
}

impl<'a> Bytes<'a> {
    #[inline]
    pub(crate) fn new(cord: &'a Cord) -> Self {
        let mut chunks = Chunks::new(cord);
        let current = chunks.next().unwrap_or(&[]).iter();
        Self { chunks, current, back_current: [].iter() }
    }

    /// Bytes not yet yielded.
    #[inline]
    fn remaining(&self) -> usize {
        self.current.as_slice().len() + self.chunks.front.bytes_remaining + self.back_current.as_slice().len()
    }
}

impl Iterator for Bytes<'_> {
    type Item = u8;

    #[inline]
    fn next(&mut self) -> Option<u8> {
        if let Some(&byte) = self.current.next() {
            return Some(byte);
        }
        if let Some(chunk) = self.chunks.next() {
            self.current = chunk.iter();
            // Chunks yields non-empty chunks, so this is always `Some`.
            return self.current.next().copied();
        }
        self.back_current.next().copied()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.remaining();
        (remaining, Some(remaining))
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<u8> {
        let cur = self.current.as_slice();
        if n < cur.len() {
            self.current = cur[n..].iter();
            return self.current.next().copied();
        }
        let skip = n - cur.len();
        self.current = [].iter();
        if skip < self.chunks.front.bytes_remaining {
            self.chunks.advance_bytes(skip);
            let chunk = self.chunks.next()?;
            self.current = chunk.iter();
            return self.current.next().copied();
        }
        let skip = skip - self.chunks.front.bytes_remaining;
        if self.chunks.front.bytes_remaining != 0 {
            self.chunks.advance_bytes(self.chunks.front.bytes_remaining);
        }
        let back = self.back_current.as_slice();
        if skip >= back.len() {
            self.back_current = [].iter();
            return None;
        }
        self.back_current = back[skip..].iter();
        self.back_current.next().copied()
    }

    #[inline]
    fn count(self) -> usize {
        self.remaining()
    }
}

impl DoubleEndedIterator for Bytes<'_> {
    #[inline]
    fn next_back(&mut self) -> Option<u8> {
        if let Some(&byte) = self.back_current.next_back() {
            return Some(byte);
        }
        if let Some(chunk) = self.chunks.next_back() {
            self.back_current = chunk.iter();
            // Chunks yields non-empty chunks, so this is always `Some`.
            return self.back_current.next_back().copied();
        }
        self.current.next_back().copied()
    }
}

impl ExactSizeIterator for Bytes<'_> {}
impl core::iter::FusedIterator for Bytes<'_> {}

/// A position inside a [`Cord`] supporting byte-wise and chunk-wise reads,
/// skipping, seeking, and sub-cord extraction. Created by [`Cord::cursor`].
///
/// Reach for [`chunks`](Cord::chunks) to process every byte of a cord and
/// [`bytes`](Cord::bytes) to iterate byte by byte; use a `Cursor` when the
/// position itself matters — parsing a wire format, reading a header and
/// then the payload it describes, or handing a cord to code that wants
/// `std::io::Read`. A cursor is cheap to clone; advancing, seeking and
/// [`read_cord`](Self::read_cord) are all O(log n) in the number of chunks,
/// and `read_cord` shares memory with the source cord rather than copying it
/// (see its own docs for the exact rule).
///
/// With the `std` feature it also implements `std::io::Read`,
/// `std::io::BufRead` and `std::io::Seek` (and `bytes::Buf` with the `bytes`
/// feature) — seeking forward advances in place, seeking backward rebuilds
/// the cursor from the start of the cord. `Cursor` does not implement
/// [`Iterator`]:
/// `take`/`by_ref` would be ambiguous with the `bytes::Buf` and
/// `std::io::Read` methods of the same name, and per-byte iteration belongs
/// to [`Cord::bytes`] instead. Use [`next_byte`](Self::next_byte),
/// [`read_cord`](Self::read_cord), [`advance`](Self::advance) or
/// [`peek`](Self::peek) here.
///
/// A cursor borrows its cord, so unlike abseil's `CharIterator` — whose docs
/// have to warn that it is invalidated by any mutation of the cord while
/// it's alive — the borrow checker rules that out at compile time: a cord
/// simply cannot be mutated while a cursor over it exists.
///
/// ```
/// use cord_rs::Cord;
/// let cord = Cord::from("header:payload");
/// let mut cursor = cord.cursor();
/// let header = cursor.read_cord(7);
/// assert_eq!(header, "header:");
/// assert_eq!(cursor.position(), 7);
/// assert_eq!(cursor.chunk(), b"payload");
/// ```
#[derive(Clone, Debug)]
pub struct Cursor<'a> {
    chunks: ForwardChunks<'a>,
    len: usize,
    /// The cord the cursor was built over. Unused by any forward-only
    /// operation; kept so a backward `std::io::Seek` can rebuild `chunks`
    /// from the start. Only `io::Seek` (`src/io.rs`) reads it, so it (and
    /// its accessor below) only exist with the `std` feature.
    #[cfg(feature = "std")]
    cord: &'a Cord,
}

// `ForwardChunks` has no drop glue (asserted above) and `Cursor`'s other
// fields are a `usize` and (with `std`) a shared reference, neither of which
// ever does; this is exactly the property that keeps a `Cursor` cheap to
// clone per search candidate in `find_impl`.
const _: () = assert!(!core::mem::needs_drop::<Cursor<'static>>());

impl<'a> Cursor<'a> {
    #[inline]
    pub(crate) fn new(cord: &'a Cord) -> Self {
        Self {
            chunks: ForwardChunks::new(cord),
            len: cord.len(),
            #[cfg(feature = "std")]
            cord,
        }
    }

    /// The cord this cursor was built over, for `io::Seek`'s backward-seek
    /// rebuild (`src/io.rs`).
    #[cfg(feature = "std")]
    #[inline]
    pub(crate) fn cord(&self) -> &'a Cord {
        self.cord
    }

    /// The underlying forward-only chunk position. Callers may only advance
    /// the returned position (`advance_bytes`, or reading through it) —
    /// never replace, reset or rebuild it: `Cursor` caches the cord's length
    /// in `len` and derives `position()` from the position's
    /// `bytes_remaining`, which stays consistent with `len` only under
    /// forward movement.
    #[inline]
    pub(crate) fn chunks_mut(&mut self) -> &mut ForwardChunks<'a> {
        &mut self.chunks
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

    /// Returns `true` if at least one byte remains after the cursor.
    ///
    /// With the `bytes` feature enabled, this shadows the
    /// identically-behaved `bytes::Buf::has_remaining` also implemented for
    /// `Cursor`.
    #[inline]
    #[must_use]
    pub fn has_remaining(&self) -> bool {
        self.chunks.bytes_remaining != 0
    }

    /// The longest contiguous run of bytes starting at the cursor (empty at
    /// the end).
    ///
    /// Empty exactly when [`has_remaining`](Self::has_remaining) is
    /// `false` — this is `std::io::BufRead::fill_buf` without the `Result`.
    #[inline]
    #[must_use]
    pub fn chunk(&self) -> &'a [u8] {
        self.chunks.current_chunk
    }

    /// The byte at the cursor, if any, without advancing.
    ///
    /// Amortized O(1): it reads from the current chunk and only touches the
    /// tree when crossing into the next one.
    #[inline]
    #[must_use]
    pub fn peek(&self) -> Option<u8> {
        self.chunks.current_chunk.first().copied()
    }

    /// Advances the cursor by `n` bytes.
    ///
    /// O(log n) in the number of chunks — it re-seeks the underlying btree
    /// rather than walking bytes one at a time.
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
    ///
    /// The result shares memory with the source cord rather than copying
    /// it, except when copying is cheaper: results of 15 bytes or fewer are
    /// copied into the returned cord's inline storage, and — when this
    /// cursor is over a multi-chunk cord — reads of up to 511 bytes that
    /// land entirely inside the cursor's current chunk are copied too, so a
    /// small read out of a large cord does not keep the whole source buffer
    /// alive. Everything else references the source's buffers, which stay
    /// alive for as long as the returned cord does.
    ///
    /// Named `read_cord` rather than `read` because, with the `std` feature,
    /// `Cursor` also implements `std::io::Read`, whose `read` fills a
    /// caller-provided buffer.
    ///
    /// ```
    /// use cord_rs::Cord;
    /// let cord = Cord::from("header:payload");
    /// let mut cursor = cord.cursor();
    /// let header = cursor.read_cord(7);
    /// assert_eq!(header, "header:");
    /// assert_eq!(cursor.position(), 7);
    /// assert_eq!(cursor.chunk(), b"payload");
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `n > remaining()`.
    #[track_caller]
    pub fn read_cord(&mut self, n: usize) -> Cord {
        assert!(
            n <= self.chunks.bytes_remaining,
            "cannot read past the end of a Cord: n = {n}, remaining = {}",
            self.chunks.bytes_remaining
        );
        self.chunks.read_bytes(n)
    }

    /// Returns the next byte and advances, or `None` at the end.
    ///
    /// Amortized O(1), for the same reason as [`peek`](Self::peek).
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
        Chunks { front: self.chunks.clone(), back: None }
    }
}
