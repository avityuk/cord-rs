//! The [`Cord`] type.

use core::cmp::Ordering;
use core::fmt;
use core::hash::{Hash, Hasher};
use core::ops::{Bound, Index, RangeBounds};

use crate::buffer::{ConsumedBuffer, CordBuffer};
use crate::inline_data::InlineData;
use crate::iter::{Bytes, Chunks, Cursor};
use crate::rep::btree::{BtreePtr, CordRepBtree, as_btree};
use crate::rep::external::{CordRepExternal, StableBytes};
use crate::rep::flat::{self, MAX_FLAT_LENGTH};
use crate::rep::{
    self, CordRep, CordRepSubstring, MAX_BYTES_TO_COPY, MAX_INLINE, RepPtr, edge_data, ref_rep,
    small_memmove, unref,
};
use crate::source::{CordLike, CordSource};

/// Memory accounting modes for [`Cord::estimated_memory_usage`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum MemoryAccounting {
    /// Counts the *approximate* number of bytes held in full or in part by
    /// this cord. Cords that share memory are each "charged" independently
    /// for the same shared memory, and memory referenced more than once by
    /// the same cord is counted more than once.
    #[default]
    Total,
    /// Like [`Total`](Self::Total), except that memory referenced more than
    /// once by this cord is only counted once. More expensive to compute as
    /// it requires deduplicating all memory references.
    TotalMorePrecise,
    /// Counts the *approximate* number of bytes held by this cord weighted by
    /// the sharing ratio of that data: memory shared by four cords is charged
    /// one quarter to each.
    FairShare,
}

/// A rope-like sequence of bytes with cheap append, prepend, slicing and
/// cloning.
///
/// A `Cord` stores its bytes either inline (up to 15 bytes, no allocation) or
/// in a reference counted B-tree of immutable buffers. This makes it well
/// suited for large byte sequences that are built incrementally, sliced, or
/// shared across API boundaries, e.g. wire-format messages that need a header
/// prepended or a payload appended.
///
/// `Cord` is a port of abseil's [`absl::Cord`], preserving its representation
/// and its optimizations:
///
/// * `size_of::<Cord>() == 16` with a 15 byte small-value optimization.
/// * O(log n) append, prepend and slicing; O(1) clone (a reference count
///   bump).
/// * Amortized in-place appends into spare capacity of privately owned
///   buffers; small values are copied instead of shared to keep memory
///   overhead low.
/// * Zero-copy adoption of large `Vec<u8>` / `String` / `Arc<[u8]>` /
///   `&'static [u8]` values.
///
/// Cords should not be used for general string data: they have more overhead
/// than `Vec<u8>` and random access is O(log n).
///
/// # Thread safety
///
/// `Cord` is `Send + Sync`. Buffers shared between cords are immutable; a
/// buffer is only mutated in place while a single cord references it.
///
/// # Bounds checking
///
/// Like `Vec` and `bytes::Bytes`, methods taking indices or ranges panic when
/// out of bounds; non-panicking variants are provided where useful
/// ([`get`](Self::get), [`try_slice`](Self::try_slice)).
///
/// [`absl::Cord`]: https://github.com/abseil/abseil-cpp/blob/master/absl/strings/cord.h
#[repr(transparent)]
pub struct Cord {
    pub(crate) data: InlineData,
}

// SAFETY: a cord is either inline data or a pointer to reference counted
// nodes. Nodes shared between cords are immutable, reference counts are
// atomic, and external owners are required to be `Send + Sync`.
unsafe impl Send for Cord {}
unsafe impl Sync for Cord {}

const _: () = assert!(core::mem::size_of::<Cord>() == 16);

// --- Free helpers on reps ---------------------------------------------------

/// Creates a new flat or btree out of `data`. Requires non-empty data.
unsafe fn new_tree(data: &[u8], alloc_hint: usize) -> *mut CordRep {
    debug_assert!(!data.is_empty());
    if data.len() <= MAX_FLAT_LENGTH {
        return flat::create(data, alloc_hint);
    }
    let first = flat::create(&data[..MAX_FLAT_LENGTH], 0);
    let root = CordRepBtree::create(first);
    CordRepBtree::append_data(root, &data[MAX_FLAT_LENGTH..], alloc_hint).as_rep()
}

/// Returns `rep` converted into a btree (a no-op if it already is one).
#[inline]
unsafe fn force_btree(rep: *mut CordRep) -> *mut CordRepBtree {
    if rep.is_btree() { rep.cast() } else { CordRepBtree::create(rep) }
}

/// Creates a rep from a large owned buffer. Copies the data if the buffer is
/// small or wasteful, adopts it otherwise. Requires `len > MAX_INLINE`.
fn rep_from_owned<O: StableBytes>(owner: O, capacity: usize) -> *mut CordRep {
    let bytes = owner.as_bytes();
    debug_assert!(bytes.len() > MAX_INLINE);
    if bytes.len() <= MAX_BYTES_TO_COPY || bytes.len() < capacity / 2 {
        // Short: copy to avoid the external node overhead.
        // Wasteful: copy to avoid pinning too much unused memory.
        return unsafe { new_tree(bytes, 0) };
    }
    CordRepExternal::create(owner)
}

/// Searches for a non-full flat at the right-most leaf of `root`. On success
/// the lengths of all nodes on the path are increased and the append region
/// is returned; the caller must immediately fill it.
#[inline]
unsafe fn prepare_append_region(root: *mut CordRep, max_length: usize) -> Option<(*mut u8, usize)> {
    if root.is_btree()
        && root.refcount().is_one()
        && let Some(span) = CordRepBtree::get_append_buffer(as_btree(root), max_length)
    {
        return Some(span);
    }
    if !root.is_flat() || !root.refcount().is_one() {
        return None;
    }
    let in_use = root.length();
    let capacity = flat::capacity(root);
    if in_use == capacity {
        return None;
    }
    let size_increase = (capacity - in_use).min(max_length);
    root.set_length(in_use + size_increase);
    Some((flat::data(root).add(in_use), size_increase))
}

/// Returns the flat data of `rep` if it is a single contiguous buffer.
unsafe fn get_flat_aux<'a>(rep: *mut CordRep) -> Option<&'a [u8]> {
    if rep.is_btree() {
        return CordRepBtree::as_flat(as_btree(rep));
    }
    // FLAT, EXTERNAL or SUBSTRING of those: all are data edges.
    debug_assert!(rep::is_data_edge(rep));
    Some(edge_data(rep))
}

/// `extract_append_buffer` for any rep type.
unsafe fn extract_append_buffer(rep: *mut CordRep, min_capacity: usize) -> rep::btree::ExtractResult {
    use rep::btree::ExtractResult;
    if rep.is_btree() {
        return CordRepBtree::extract_append_buffer(as_btree(rep), min_capacity);
    }
    if rep.is_flat() && rep.refcount().is_one() && flat::capacity(rep) - rep.length() >= min_capacity {
        return ExtractResult { tree: core::ptr::null_mut(), extracted: rep };
    }
    ExtractResult { tree: rep, extracted: core::ptr::null_mut() }
}

#[cold]
#[inline(never)]
#[track_caller]
fn length_overflow() -> ! {
    panic!("Cord length overflow")
}

// --- Internal representation helpers --------------------------------------

impl Cord {
    /// The tree, if this cord is not inline.
    #[inline]
    pub(crate) fn tree(&self) -> Option<*mut CordRep> {
        self.data.tree()
    }

    #[inline]
    pub(crate) fn is_tree(&self) -> bool {
        self.data.is_tree()
    }

    /// The inline data. Requires `!is_tree()`.
    #[inline]
    pub(crate) fn inline_slice(&self) -> &[u8] {
        self.data.inline_slice()
    }

    /// Creates a cord holding `rep` (which must be non-null and non-empty).
    #[inline]
    pub(crate) unsafe fn from_rep(rep: *mut CordRep) -> Self {
        debug_assert!(!rep.is_null());
        debug_assert!(rep.length() != 0);
        Self { data: InlineData::from_tree(rep) }
    }

    /// Creates an inline cord. Requires `data.len() <= MAX_INLINE`.
    #[inline]
    pub(crate) fn from_inline(data: &[u8]) -> Self {
        let mut cord = Self::new();
        cord.data.set_inline_data(data);
        cord
    }

    /// Steals the tree out of `self` without decrementing its refcount.
    /// Requires `is_tree()`.
    #[inline]
    unsafe fn into_rep(self) -> *mut CordRep {
        let rep = self.data.as_tree();
        core::mem::forget(self);
        rep
    }

    /// Returns a new reference to the tree. Requires `is_tree()`.
    #[inline]
    unsafe fn take_rep_ref(&self) -> *mut CordRep {
        ref_rep(self.data.as_tree())
    }

    #[inline]
    unsafe fn emplace_tree(&mut self, rep: *mut CordRep) {
        self.data.make_tree(rep);
    }

    #[inline]
    unsafe fn set_tree(&mut self, rep: *mut CordRep) {
        self.data.set_tree(rep);
    }

    #[inline]
    unsafe fn set_tree_or_empty(&mut self, rep: *mut CordRep) {
        debug_assert!(self.is_tree());
        if rep.is_null() {
            self.data = InlineData::new();
        } else {
            self.data.set_tree(rep);
        }
    }

    /// Commits a new or updated root `rep`; `had_tree` tells whether the cord
    /// previously held a tree.
    #[inline]
    unsafe fn commit_tree(&mut self, had_tree: bool, rep: *mut CordRep) {
        if had_tree {
            self.set_tree(rep);
        } else {
            self.emplace_tree(rep);
        }
    }

    /// Creates a flat from the inline data with `extra` bytes of capacity.
    unsafe fn make_flat_with_extra_capacity(&self, extra: usize) -> *mut CordRep {
        let len = self.data.inline_size();
        let result = flat::new(len + extra);
        result.set_length(len);
        self.data.copy_max_inline_to(flat::data(result));
        result
    }

    unsafe fn append_tree(&mut self, tree: *mut CordRep) {
        debug_assert!(!tree.is_null());
        debug_assert!(tree.length() != 0);
        if self.is_tree() {
            let tree = CordRepBtree::append(force_btree(self.data.as_tree()), tree);
            self.set_tree(tree.as_rep());
        } else {
            let mut tree = tree;
            if !self.data.is_empty() {
                let flat = self.make_flat_with_extra_capacity(0);
                tree = CordRepBtree::append(CordRepBtree::create(flat), tree).as_rep();
            }
            self.emplace_tree(tree);
        }
    }

    unsafe fn prepend_tree(&mut self, tree: *mut CordRep) {
        debug_assert!(!tree.is_null());
        debug_assert!(tree.length() != 0);
        if self.is_tree() {
            let tree = CordRepBtree::prepend(force_btree(self.data.as_tree()), tree);
            self.set_tree(tree.as_rep());
        } else {
            let mut tree = tree;
            if !self.data.is_empty() {
                let flat = self.make_flat_with_extra_capacity(0);
                tree = CordRepBtree::prepend(CordRepBtree::create(flat), tree).as_rep();
            }
            self.emplace_tree(tree);
        }
    }

    /// Appends `src`, using spare capacity in the last flat where possible.
    pub(crate) fn append_slice(&mut self, mut src: &[u8]) {
        if src.is_empty() {
            return;
        }
        // SAFETY: standard rep manipulation; see abseil's `AppendArray`.
        unsafe {
            let mut appended = 0;
            let root = self.tree();
            let rep: *mut CordRep;
            if let Some(root) = root {
                rep = root;
                if let Some((region, n)) = prepare_append_region(rep, src.len()) {
                    core::ptr::copy_nonoverlapping(src.as_ptr(), region, n);
                    appended = n;
                }
            } else {
                // Try to fit in the inline buffer if possible.
                let inline_length = self.data.inline_size();
                if src.len() <= MAX_INLINE - inline_length {
                    self.data.set_inline_size(inline_length + src.len());
                    core::ptr::copy_nonoverlapping(
                        src.as_ptr(),
                        self.data.as_chars_mut().add(inline_length),
                        src.len(),
                    );
                    return;
                }
                // Allocate a flat that is a perfect fit on the first append
                // exceeding the inline size. Subsequent growth is amortized
                // until we reach the maximum flat size.
                rep = flat::new(inline_length + src.len());
                appended = src.len().min(flat::capacity(rep) - inline_length);
                core::ptr::copy_nonoverlapping(self.data.as_chars(), flat::data(rep), inline_length);
                core::ptr::copy_nonoverlapping(src.as_ptr(), flat::data(rep).add(inline_length), appended);
                rep.set_length(inline_length + appended);
            }

            src = &src[appended..];
            if src.is_empty() {
                self.commit_tree(root.is_some(), rep);
                return;
            }

            // Keep abseil's 10% growth rate.
            let tree = force_btree(rep);
            let min_growth = (tree.length() / 10).max(src.len());
            let tree = CordRepBtree::append_data(tree, src, min_growth - src.len());
            self.commit_tree(root.is_some(), tree.as_rep());
        }
    }

    /// Prepends `src`.
    pub(crate) fn prepend_slice(&mut self, src: &[u8]) {
        if src.is_empty() {
            return;
        }
        if !self.is_tree() {
            let cur_size = self.data.inline_size();
            if cur_size + src.len() <= MAX_INLINE {
                let mut data = InlineData::new();
                data.set_inline_size(cur_size + src.len());
                // SAFETY: both copies stay within the 15 inline bytes.
                unsafe {
                    core::ptr::copy_nonoverlapping(src.as_ptr(), data.as_chars_mut(), src.len());
                    core::ptr::copy_nonoverlapping(
                        self.data.as_chars(),
                        data.as_chars_mut().add(src.len()),
                        cur_size,
                    );
                }
                self.data = data;
                return;
            }
        }
        // SAFETY: `src` is non-empty.
        unsafe {
            let rep = new_tree(src, 0);
            self.prepend_tree(rep);
        }
    }

    /// Appends `src` with precise sizing (no spare capacity is used or
    /// allocated). Requires `0 < src.len() <= MAX_FLAT_LENGTH`.
    unsafe fn append_precise(&mut self, src: &[u8]) {
        debug_assert!(!src.is_empty());
        debug_assert!(src.len() <= MAX_FLAT_LENGTH);
        if self.remaining_inline_capacity() >= src.len() {
            let inline_length = self.data.inline_size();
            self.data.set_inline_size(inline_length + src.len());
            core::ptr::copy_nonoverlapping(
                src.as_ptr(),
                self.data.as_chars_mut().add(inline_length),
                src.len(),
            );
        } else {
            self.append_tree(flat::create(src, 0));
        }
    }

    /// Prepends `src` with precise sizing. Requires `0 < src.len() <= MAX_FLAT_LENGTH`.
    unsafe fn prepend_precise(&mut self, src: &[u8]) {
        debug_assert!(!src.is_empty());
        debug_assert!(src.len() <= MAX_FLAT_LENGTH);
        if self.remaining_inline_capacity() >= src.len() {
            let cur_size = self.data.inline_size();
            let mut data = InlineData::new();
            data.set_inline_size(cur_size + src.len());
            core::ptr::copy_nonoverlapping(src.as_ptr(), data.as_chars_mut(), src.len());
            core::ptr::copy_nonoverlapping(
                self.data.as_chars(),
                data.as_chars_mut().add(src.len()),
                cur_size,
            );
            self.data = data;
        } else {
            self.prepend_tree(flat::create(src, 0));
        }
    }

    #[inline]
    fn remaining_inline_capacity(&self) -> usize {
        if self.is_tree() { 0 } else { MAX_INLINE - self.data.inline_size() }
    }

    /// Appends another cord (borrowed).
    pub(crate) fn append_cord(&mut self, src: &Cord) {
        // SAFETY: see abseil's `AppendImpl`.
        unsafe {
            if src.is_empty() {
                return;
            }
            if src.len() > usize::MAX - self.len() {
                length_overflow();
            }
            if self.is_empty() {
                // The destination is empty: take the tree or copy inline data.
                if let Some(tree) = src.tree() {
                    self.emplace_tree(ref_rep(tree));
                } else {
                    self.data = src.data;
                }
                return;
            }
            // For short cords it is faster to copy data if there is room.
            let src_size = src.len();
            if src_size <= MAX_BYTES_TO_COPY {
                match src.tree() {
                    None => self.append_slice(src.inline_slice()),
                    Some(tree) if tree.is_flat() => self.append_slice(edge_data(tree)),
                    Some(_) => {
                        for chunk in src.chunks() {
                            self.append_slice(chunk);
                        }
                    }
                }
                return;
            }
            // Guaranteed to be a tree (MAX_BYTES_TO_COPY > MAX_INLINE).
            self.append_tree(src.take_rep_ref());
        }
    }

    /// Appends another cord (owned), stealing its tree.
    pub(crate) fn append_owned_cord(&mut self, src: Cord) {
        // SAFETY: see abseil's `AppendImpl`.
        unsafe {
            if src.is_empty() {
                return;
            }
            if src.len() > usize::MAX - self.len() {
                length_overflow();
            }
            if self.is_empty() {
                if src.is_tree() {
                    let rep = src.into_rep();
                    self.emplace_tree(rep);
                } else {
                    self.data = src.data;
                }
                return;
            }
            let src_size = src.len();
            if src_size <= MAX_BYTES_TO_COPY {
                match src.tree() {
                    None => self.append_slice(src.inline_slice()),
                    Some(tree) if tree.is_flat() => self.append_slice(edge_data(tree)),
                    Some(_) => {
                        for chunk in src.chunks() {
                            self.append_slice(chunk);
                        }
                    }
                }
                return;
            }
            let rep = src.into_rep();
            self.append_tree(rep);
        }
    }

    /// Prepends another cord (borrowed).
    pub(crate) fn prepend_cord(&mut self, src: &Cord) {
        if src.is_empty() {
            return;
        }
        if src.len() > usize::MAX - self.len() {
            length_overflow();
        }
        if let Some(tree) = src.tree() {
            // SAFETY: `src` holds a reference; we add one for ourselves.
            unsafe {
                ref_rep(tree);
                self.prepend_tree(tree);
            }
            return;
        }
        self.prepend_slice(src.inline_slice());
    }

    /// Prepends another cord (owned), stealing its tree.
    pub(crate) fn prepend_owned_cord(&mut self, src: Cord) {
        if src.is_empty() {
            return;
        }
        if src.len() > usize::MAX - self.len() {
            length_overflow();
        }
        if src.is_tree() {
            // SAFETY: `into_rep` transfers src's reference to us.
            unsafe {
                let rep = src.into_rep();
                self.prepend_tree(rep);
            }
            return;
        }
        self.prepend_slice(src.inline_slice());
    }

    /// Appends a large owned buffer, adopting it if worthwhile.
    pub(crate) fn append_owned<O: StableBytes>(&mut self, owner: O, capacity: usize) {
        let len = owner.as_bytes().len();
        if len <= MAX_BYTES_TO_COPY {
            self.append_slice(owner.as_bytes());
        } else {
            // SAFETY: `rep_from_owned` returns a fresh non-empty rep.
            unsafe { self.append_tree(rep_from_owned(owner, capacity)) }
        }
    }

    /// Prepends a large owned buffer, adopting it if worthwhile.
    pub(crate) fn prepend_owned<O: StableBytes>(&mut self, owner: O, capacity: usize) {
        let len = owner.as_bytes().len();
        if len <= MAX_BYTES_TO_COPY {
            self.prepend_slice(owner.as_bytes());
        } else {
            // SAFETY: `rep_from_owned` returns a fresh non-empty rep.
            unsafe { self.prepend_tree(rep_from_owned(owner, capacity)) }
        }
    }

    /// Creates a cord from an owned buffer, adopting it if worthwhile.
    pub(crate) fn from_owned<O: StableBytes>(owner: O, capacity: usize) -> Self {
        let bytes = owner.as_bytes();
        if bytes.len() <= MAX_INLINE {
            return Self::from_inline(bytes);
        }
        // SAFETY: `rep_from_owned` returns a fresh non-empty rep.
        unsafe { Self::from_rep(rep_from_owned(owner, capacity)) }
    }

    /// Appends the contents of a [`CordBuffer`].
    pub(crate) fn append_buffer(&mut self, buffer: CordBuffer) {
        if buffer.is_empty() {
            return;
        }
        // SAFETY: a consumed buffer's rep is a fresh flat with a refcount of 1.
        unsafe {
            match buffer.consume() {
                ConsumedBuffer::Rep(rep) => self.append_tree(rep),
                ConsumedBuffer::Short(short) => self.append_precise(short.as_slice()),
            }
        }
    }

    /// Prepends the contents of a [`CordBuffer`].
    pub(crate) fn prepend_buffer(&mut self, buffer: CordBuffer) {
        if buffer.is_empty() {
            return;
        }
        // SAFETY: as in `append_buffer`.
        unsafe {
            match buffer.consume() {
                ConsumedBuffer::Rep(rep) => self.prepend_tree(rep),
                ConsumedBuffer::Short(short) => self.prepend_precise(short.as_slice()),
            }
        }
    }

    /// Locates the first flat or external chunk without initializing an
    /// iterator. Returns empty for an empty cord.
    pub(crate) fn first_chunk(&self) -> &[u8] {
        let Some(mut node) = self.tree() else {
            return self.data.inline_slice();
        };
        // SAFETY: `node` is a live tree held by `self`.
        unsafe {
            if node.is_btree() {
                let mut tree = as_btree(node);
                let mut height = tree.height();
                while height > 0 {
                    height -= 1;
                    tree = as_btree(tree.edge_at::<{ rep::btree::FRONT }>());
                }
                return tree.data(tree.begin());
            }
            // FLAT, EXTERNAL or a SUBSTRING thereof.
            let mut offset = 0;
            let length = node.length();
            debug_assert!(length != 0);
            if node.is_substring() {
                let sub: *mut CordRepSubstring = node.cast();
                offset = (*sub).start;
                node = (*sub).child;
            }
            &edge_data(node)[offset..offset + length]
        }
    }

    /// Slow path of [`flatten`](Self::flatten).
    fn flatten_slow_path(&mut self) {
        let total_size = self.len();
        // SAFETY: `self` holds a non-flat tree which we replace by a flat one.
        unsafe {
            let new_rep = if total_size <= MAX_FLAT_LENGTH {
                let rep = flat::new(total_size);
                rep.set_length(total_size);
                self.copy_to_ptr(flat::data(rep));
                rep
            } else {
                let mut buffer: Vec<u8> = Vec::with_capacity(total_size);
                self.copy_to_ptr(buffer.as_mut_ptr());
                buffer.set_len(total_size);
                CordRepExternal::create(buffer)
            };
            unref(self.data.as_tree());
            self.set_tree(new_rep);
        }
    }

    /// Copies all bytes to `dst`, which must have room for `len()` bytes.
    unsafe fn copy_to_ptr(&self, mut dst: *mut u8) {
        if let Some(chunk) = self.as_flat() {
            core::ptr::copy_nonoverlapping(chunk.as_ptr(), dst, chunk.len());
            return;
        }
        for chunk in self.chunks() {
            core::ptr::copy_nonoverlapping(chunk.as_ptr(), dst, chunk.len());
            dst = dst.add(chunk.len());
        }
    }
}

// --- Public API ---------------------------------------------------------------

impl Cord {
    /// Creates an empty cord. Does not allocate.
    ///
    /// ```
    /// use cord_rs::Cord;
    /// let cord = Cord::new();
    /// assert!(cord.is_empty());
    /// ```
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self { data: InlineData::new() }
    }

    /// Creates a cord referencing static data without copying it.
    ///
    /// Values of 15 bytes or less are stored inline instead.
    ///
    /// ```
    /// use cord_rs::Cord;
    /// static PAYLOAD: [u8; 32] = [0xAB; 32];
    /// let cord = Cord::from_static(&PAYLOAD);
    /// assert_eq!(cord.as_flat(), Some(&PAYLOAD[..]));
    /// ```
    pub fn from_static<T: AsRef<[u8]> + ?Sized>(data: &'static T) -> Self {
        let bytes = data.as_ref();
        if bytes.len() <= MAX_INLINE {
            return Self::from_inline(bytes);
        }
        // SAFETY: `CordRepExternal::new` returns a fresh non-empty rep.
        unsafe { Self::from_rep(CordRepExternal::create(bytes)) }
    }

    /// Creates a cord by copying `data`.
    ///
    /// Equivalent to `Cord::from(data)`.
    #[must_use]
    pub fn copy_from_slice(data: &[u8]) -> Self {
        if data.len() <= MAX_INLINE {
            return Self::from_inline(data);
        }
        // SAFETY: `data` is non-empty.
        unsafe { Self::from_rep(new_tree(data, 0)) }
    }

    /// Returns the number of bytes in the cord.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        match self.tree() {
            // SAFETY: the tree is live.
            Some(tree) => unsafe { tree.length() },
            None => self.data.inline_size(),
        }
    }

    /// Returns `true` if the cord holds no bytes.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Releases the cord's data, making it empty. Any buffers shared with
    /// other cords have their reference counts decremented.
    #[inline]
    pub fn clear(&mut self) {
        if let Some(tree) = self.tree() {
            self.data = InlineData::new();
            // SAFETY: we held a reference on `tree` and no longer point to it.
            unsafe { unref(tree) };
        } else {
            self.data = InlineData::new();
        }
    }

    /// Appends `src` to the cord.
    ///
    /// Accepts byte slices, strings, other cords (borrowed or owned), owned
    /// buffers (`Vec<u8>`, `String`, `Box<[u8]>`, `Arc<[u8]>`, ...) and
    /// [`CordBuffer`]s; see [`CordSource`]. Owned buffers larger than a few
    /// hundred bytes are adopted without copying; appending an owned `Cord`
    /// is a pointer move.
    ///
    /// ```
    /// use cord_rs::Cord;
    /// let mut cord = Cord::from("hello");
    /// cord.append(", ");
    /// cord.append(String::from("world"));
    /// cord.append(&Cord::from("!"));
    /// assert_eq!(cord, "hello, world!");
    /// ```
    #[inline]
    pub fn append<S: CordSource>(&mut self, src: S) {
        src.append_to(self);
    }

    /// Prepends `src` to the cord. Accepts the same inputs as
    /// [`append`](Self::append).
    ///
    /// ```
    /// use cord_rs::Cord;
    /// let mut cord = Cord::from("world");
    /// cord.prepend("hello ");
    /// assert_eq!(cord, "hello world");
    /// ```
    #[inline]
    pub fn prepend<S: CordSource>(&mut self, src: S) {
        src.prepend_to(self);
    }

    /// Returns a [`CordBuffer`] for appending, reusing spare capacity in the
    /// cord's last buffer if possible.
    ///
    /// If the cord's last buffer is privately owned and has at least 16 bytes
    /// of spare capacity, that buffer is *removed* from the cord and returned
    /// together with its existing contents (so the returned buffer has a
    /// non-zero length and a capacity of at least `len + 16`). Otherwise a
    /// new buffer of the requested `capacity` (capped at
    /// [`CordBuffer::DEFAULT_LIMIT`]) is returned. Either way the caller must
    /// [`append`](Self::append) the buffer back to restore the data.
    ///
    /// ```
    /// use cord_rs::Cord;
    /// fn append_random(cord: &mut Cord, mut n: usize) {
    ///     let mut first = true;
    ///     while n > 0 {
    ///         let mut buffer = if first {
    ///             cord.take_append_buffer(n)
    ///         } else {
    ///             cord_rs::CordBuffer::with_default_limit(n)
    ///         };
    ///         let count = buffer.available().min(n);
    ///         let data = vec![42u8; count]; // e.g. fill from a random source
    ///         buffer.put_slice(&data);
    ///         cord.append(buffer);
    ///         n -= count;
    ///         first = false;
    ///     }
    /// }
    /// let mut cord = Cord::new();
    /// append_random(&mut cord, 10_000);
    /// assert_eq!(cord.len(), 10_000);
    /// ```
    #[inline]
    pub fn take_append_buffer(&mut self, capacity: usize) -> CordBuffer {
        if self.is_empty() {
            return CordBuffer::with_default_limit(capacity);
        }
        self.take_append_buffer_slow_path(0, capacity, 16)
    }

    /// Like [`take_append_buffer`](Self::take_append_buffer) with explicit
    /// parameters: a newly allocated buffer uses
    /// [`CordBuffer::with_custom_limit`]`(block_size, capacity)` (or the
    /// default limit if `block_size` is 0), and an existing buffer is only
    /// reused if it has at least `min_capacity` bytes available.
    pub fn take_append_buffer_with(
        &mut self,
        block_size: usize,
        capacity: usize,
        min_capacity: usize,
    ) -> CordBuffer {
        if self.is_empty() {
            return if block_size != 0 {
                CordBuffer::with_custom_limit(block_size, capacity)
            } else {
                CordBuffer::with_default_limit(capacity)
            };
        }
        self.take_append_buffer_slow_path(block_size, capacity, min_capacity)
    }

    fn take_append_buffer_slow_path(
        &mut self,
        block_size: usize,
        capacity: usize,
        min_capacity: usize,
    ) -> CordBuffer {
        if let Some(tree) = self.tree() {
            // SAFETY: `tree` is our live tree; on success the extracted flat
            // has a refcount of one and is no longer referenced by the tree.
            unsafe {
                let result = extract_append_buffer(tree, min_capacity);
                if !result.extracted.is_null() {
                    self.set_tree_or_empty(result.tree);
                    return CordBuffer::from_flat(result.extracted);
                }
            }
            return if block_size != 0 {
                CordBuffer::with_custom_limit(block_size, capacity)
            } else {
                CordBuffer::with_default_limit(capacity)
            };
        }
        // Inline: move the inline data into a new buffer.
        let size = self.data.inline_size();
        let max_capacity = usize::MAX - size;
        let capacity = capacity.min(max_capacity) + size;
        let mut buffer = if block_size != 0 {
            CordBuffer::with_custom_limit(block_size, capacity)
        } else {
            CordBuffer::with_default_limit(capacity)
        };
        buffer.put_slice(self.data.inline_slice());
        self.data = InlineData::new();
        buffer
    }

    /// Removes the first `n` bytes.
    ///
    /// # Panics
    ///
    /// Panics if `n > self.len()`.
    ///
    /// ```
    /// use cord_rs::Cord;
    /// let mut cord = Cord::from("hello world");
    /// cord.advance(6);
    /// assert_eq!(cord, "world");
    /// ```
    #[track_caller]
    pub fn advance(&mut self, n: usize) {
        let len = self.len();
        assert!(n <= len, "cannot advance past end of Cord: n = {n}, len = {len}");
        let Some(tree) = self.tree() else {
            // SAFETY: inline data, `n <= inline size`.
            unsafe {
                let size = self.data.inline_size() - n;
                small_memmove::<false>(self.data.as_chars_mut(), self.data.as_chars().add(n), size);
                self.reduce_inline_size(n);
            }
            return;
        };
        // SAFETY: standard rep manipulation; see abseil's `RemovePrefix`.
        unsafe {
            let new_tree: *mut CordRep = if n >= tree.length() {
                unref(tree);
                core::ptr::null_mut()
            } else if tree.is_btree() {
                let sub = CordRepBtree::sub_tree(as_btree(tree), n, tree.length() - n);
                unref(tree);
                sub
            } else if tree.is_substring() && tree.refcount().is_one() {
                let sub: *mut CordRepSubstring = tree.cast();
                (*sub).start += n;
                tree.set_length(tree.length() - n);
                tree
            } else {
                let rep = CordRepSubstring::substring(tree, n, tree.length() - n);
                unref(tree);
                rep
            };
            self.set_tree_or_empty(new_tree);
        }
    }

    /// Shortens the cord to `len` bytes, keeping the first `len`. Has no
    /// effect if `len >= self.len()`.
    ///
    /// ```
    /// use cord_rs::Cord;
    /// let mut cord = Cord::from("hello world");
    /// cord.truncate(5);
    /// assert_eq!(cord, "hello");
    /// ```
    pub fn truncate(&mut self, len: usize) {
        let current = self.len();
        if len >= current {
            return;
        }
        let n = current - len;
        let Some(tree) = self.tree() else {
            // SAFETY: inline data, `n <= inline size`.
            unsafe { self.reduce_inline_size(n) };
            return;
        };
        // SAFETY: standard rep manipulation; see abseil's `RemoveSuffix`.
        unsafe {
            let new_tree: *mut CordRep = if n >= tree.length() {
                unref(tree);
                core::ptr::null_mut()
            } else if tree.is_btree() {
                CordRepBtree::remove_suffix(as_btree(tree), n)
            } else if !tree.is_external() && tree.refcount().is_one() {
                debug_assert!(tree.is_flat() || tree.is_substring());
                tree.set_length(tree.length() - n);
                tree
            } else {
                let rep = CordRepSubstring::substring(tree, 0, tree.length() - n);
                unref(tree);
                rep
            };
            self.set_tree_or_empty(new_tree);
        }
    }

    /// Reduces the inline size by `n`, zeroing the tail.
    unsafe fn reduce_inline_size(&mut self, n: usize) {
        let size = self.data.inline_size();
        debug_assert!(size >= n);
        let new_size = size - n;
        core::ptr::write_bytes(self.data.as_chars_mut().add(new_size), 0, n);
        self.data.set_inline_size(new_size);
    }

    /// Returns a new cord holding the bytes in `range`, sharing memory with
    /// this cord where possible (small results are copied).
    ///
    /// # Panics
    ///
    /// Panics if the range is out of bounds or if `start > end`.
    ///
    /// ```
    /// use cord_rs::Cord;
    /// let cord = Cord::from("hello world");
    /// assert_eq!(cord.slice(6..), "world");
    /// assert_eq!(cord.slice(..5), "hello");
    /// assert_eq!(cord.slice(2..=3), "ll");
    /// ```
    #[track_caller]
    #[must_use]
    pub fn slice(&self, range: impl RangeBounds<usize>) -> Cord {
        let (pos, new_size) = resolve_range(range, self.len());
        self.subcord(pos, new_size)
    }

    /// Non-panicking version of [`slice`](Self::slice): returns `None` if the
    /// range is out of bounds.
    pub fn try_slice(&self, range: impl RangeBounds<usize>) -> Option<Cord> {
        let (pos, new_size) = try_resolve_range(range, self.len())?;
        Some(self.subcord(pos, new_size))
    }

    /// Returns the `new_size` bytes starting at `pos`. Requires bounds to be
    /// checked.
    fn subcord(&self, pos: usize, new_size: usize) -> Cord {
        debug_assert!(pos <= self.len() && new_size <= self.len() - pos);
        if new_size == 0 {
            return Cord::new();
        }
        let Some(tree) = self.tree() else {
            return Cord::from_inline(&self.data.inline_slice()[pos..pos + new_size]);
        };
        if new_size <= MAX_INLINE {
            let mut sub = Cord::new();
            sub.data.set_inline_size(new_size);
            // SAFETY: we copy exactly `new_size <= 15` bytes into the inline
            // buffer, which is zero initialized.
            unsafe {
                let mut dest = sub.data.as_chars_mut();
                let mut it = self.chunks();
                it.advance_bytes(pos);
                let mut remaining = new_size;
                loop {
                    let chunk = it.current_chunk();
                    if remaining > chunk.len() {
                        small_memmove::<false>(dest, chunk.as_ptr(), chunk.len());
                        remaining -= chunk.len();
                        dest = dest.add(chunk.len());
                        it.next();
                    } else {
                        small_memmove::<false>(dest, chunk.as_ptr(), remaining);
                        break;
                    }
                }
            }
            return sub;
        }
        // SAFETY: `tree` is live; `sub_tree` / `substring` return a new
        // reference.
        unsafe {
            let rep = if tree.is_btree() {
                CordRepBtree::sub_tree(as_btree(tree), pos, new_size)
            } else {
                CordRepSubstring::substring(tree, pos, new_size)
            };
            Cord::from_rep(rep)
        }
    }

    /// Splits the cord into two at `at`: `self` keeps `[0, at)` and the
    /// returned cord holds `[at, len)`.
    ///
    /// # Panics
    ///
    /// Panics if `at > self.len()`.
    #[track_caller]
    #[must_use = "the split off tail is dropped if unused; use `truncate` to just shorten"]
    pub fn split_off(&mut self, at: usize) -> Cord {
        let len = self.len();
        assert!(at <= len, "split_off index out of bounds: at = {at}, len = {len}");
        if at == len {
            return Cord::new();
        }
        if at == 0 {
            return core::mem::take(self);
        }
        let tail = self.subcord(at, len - at);
        self.truncate(at);
        tail
    }

    /// Splits the cord into two at `at`: the returned cord holds `[0, at)`
    /// and `self` keeps `[at, len)`.
    ///
    /// # Panics
    ///
    /// Panics if `at > self.len()`.
    #[track_caller]
    #[must_use = "the split off head is dropped if unused; use `advance` to just skip"]
    pub fn split_to(&mut self, at: usize) -> Cord {
        let len = self.len();
        assert!(at <= len, "split_to index out of bounds: at = {at}, len = {len}");
        if at == 0 {
            return Cord::new();
        }
        if at == len {
            return core::mem::take(self);
        }
        let head = self.subcord(0, at);
        self.advance(at);
        head
    }

    /// Returns the cord's bytes as a single slice if they are stored
    /// contiguously, `None` otherwise.
    ///
    /// ```
    /// use cord_rs::Cord;
    /// let cord = Cord::from("contiguous");
    /// assert_eq!(cord.as_flat(), Some(&b"contiguous"[..]));
    /// ```
    #[inline]
    #[must_use]
    pub fn as_flat(&self) -> Option<&[u8]> {
        match self.tree() {
            None => Some(self.data.inline_slice()),
            // SAFETY: the tree is live and immutable while `&self` is held.
            Some(rep) => unsafe { get_flat_aux(rep) },
        }
    }

    /// Flattens the cord into a single contiguous buffer and returns it.
    /// If the cord is already flat its contents are not modified.
    ///
    /// ```
    /// use cord_rs::Cord;
    /// let mut cord = Cord::from(vec![b'a'; 5000]);
    /// cord.append(vec![b'b'; 5000]);
    /// assert!(cord.as_flat().is_none());
    /// assert_eq!(cord.flatten().len(), 10_000);
    /// assert!(cord.as_flat().is_some());
    /// ```
    pub fn flatten(&mut self) -> &[u8] {
        if self.as_flat().is_none() {
            self.flatten_slow_path();
        }
        self.as_flat().unwrap_or_default()
    }

    /// Copies the cord's bytes into a new `Vec<u8>`.
    #[must_use]
    pub fn to_vec(&self) -> Vec<u8> {
        let mut vec = Vec::with_capacity(self.len());
        // SAFETY: the vector has capacity for `len()` bytes which we fully
        // initialize.
        unsafe {
            self.copy_to_ptr(vec.as_mut_ptr());
            vec.set_len(self.len());
        }
        vec
    }

    /// Copies up to `dst.len()` bytes from the start of the cord into `dst`
    /// and returns the number of bytes copied. The cord is not modified.
    ///
    /// (With the `bytes` feature, `bytes::Buf::copy_to_slice` is the
    /// *consuming* variant.)
    pub fn copy_prefix_to(&self, dst: &mut [u8]) -> usize {
        if self.len() <= dst.len() {
            // SAFETY: `dst` has room for `len()` bytes.
            unsafe { self.copy_to_ptr(dst.as_mut_ptr()) };
            return self.len();
        }
        let mut dst = dst;
        let result = dst.len();
        for chunk in self.chunks() {
            let n = chunk.len().min(dst.len());
            if n == 0 {
                break;
            }
            dst[..n].copy_from_slice(&chunk[..n]);
            dst = &mut dst[n..];
        }
        result
    }

    /// Returns an iterator over the contiguous chunks of the cord.
    ///
    /// Every yielded chunk is non-empty. Iterating chunks is the most
    /// efficient way to process a cord's data.
    ///
    /// ```
    /// use cord_rs::Cord;
    /// let cord = Cord::from("hello");
    /// let joined: Vec<u8> = cord.chunks().flatten().copied().collect();
    /// assert_eq!(joined, b"hello");
    /// ```
    #[inline]
    #[must_use]
    pub fn chunks(&self) -> Chunks<'_> {
        Chunks::new(self)
    }

    /// Returns an iterator over the contiguous chunks of the cord; the same
    /// as [`chunks`](Self::chunks) (provided so `&Cord` iteration has the
    /// conventional `iter` spelling).
    #[inline]
    #[must_use]
    pub fn iter(&self) -> Chunks<'_> {
        self.chunks()
    }

    /// Returns an iterator over the bytes of the cord.
    ///
    /// Prefer [`chunks`](Self::chunks) for bulk processing.
    #[inline]
    #[must_use]
    pub fn bytes(&self) -> Bytes<'_> {
        Bytes::new(self)
    }

    /// Returns a [`Cursor`] positioned at the start of the cord, supporting
    /// byte-wise and chunk-wise reading, skipping and sub-cord extraction.
    #[inline]
    #[must_use]
    pub fn cursor(&self) -> Cursor<'_> {
        Cursor::new(self)
    }

    /// Returns the byte at `index`, or `None` if out of bounds.
    ///
    /// Random access is O(log n) in the number of chunks; use iteration for
    /// sequential access.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<u8> {
        if index >= self.len() {
            return None;
        }
        Some(*self.byte_ref(index))
    }

    /// Returns a reference to the byte at `index`. Requires `index < len()`.
    fn byte_ref(&self, index: usize) -> &u8 {
        debug_assert!(index < self.len());
        let Some(mut rep) = self.tree() else {
            return &self.data.inline_slice()[index];
        };
        let mut offset = index;
        // SAFETY: the tree is live and immutable while `&self` is held.
        unsafe {
            loop {
                debug_assert!(offset < rep.length());
                if rep.is_btree() {
                    let mut node = as_btree(rep);
                    let mut height = node.height();
                    loop {
                        let front = node.index_of(offset);
                        if height == 0 {
                            return &node.data(front.index)[front.n];
                        }
                        height -= 1;
                        offset = front.n;
                        node = as_btree(node.edge(front.index));
                    }
                } else if rep.is_substring() {
                    let sub: *mut CordRepSubstring = rep.cast();
                    offset += (*sub).start;
                    rep = (*sub).child;
                } else {
                    return &edge_data(rep)[offset];
                }
            }
        }
    }

    /// Returns the position of the first occurrence of `needle`, or `None`.
    /// An empty needle is found at position 0.
    ///
    /// ```
    /// use cord_rs::Cord;
    /// let cord = Cord::from("hello world");
    /// assert_eq!(cord.find("world"), Some(6));
    /// assert_eq!(cord.find(&Cord::from("o w")), Some(4));
    /// assert_eq!(cord.find("xyz"), None);
    /// ```
    pub fn find<N: CordLike + ?Sized>(&self, needle: &N) -> Option<usize> {
        let needle_size = needle.len();
        if needle_size == 0 {
            return Some(0);
        }
        if needle_size > self.len() {
            return None;
        }
        if needle_size == self.len() {
            return if self.equals(needle) { Some(0) } else { None };
        }
        let needle_chunk = needle.first_chunk();
        let mut haystack = self.cursor();
        if needle_chunk.len() == needle_size {
            // Single chunk needle.
            return find_impl(&mut haystack, needle_chunk).then(|| haystack.position());
        }
        loop {
            if !find_impl(&mut haystack, needle_chunk) || haystack.remaining() < needle_size {
                return None;
            }
            // We found the first chunk of `needle`; check the remainder.
            let mut haystack_advanced = haystack.clone();
            let mut needle_it = needle.chunks();
            needle_it.next();
            haystack_advanced.advance(needle_chunk.len());
            if is_subcord_at(&mut haystack_advanced, needle_it) {
                return Some(haystack.position());
            }
            haystack.advance(1);
            if haystack.remaining() < needle_size {
                return None;
            }
            if haystack.remaining() == needle_size {
                // Exactly `needle_size` bytes remain: the needle is either
                // here or not at all.
                let mut it = haystack.clone();
                return is_subcord_at(&mut it, needle.chunks()).then(|| haystack.position());
            }
        }
    }

    /// Returns `true` if the cord contains `needle`.
    pub fn contains<N: CordLike + ?Sized>(&self, needle: &N) -> bool {
        needle.is_empty() || self.find(needle).is_some()
    }

    /// Returns `true` if the cord starts with `prefix`.
    ///
    /// ```
    /// use cord_rs::Cord;
    /// let cord = Cord::from("hello world");
    /// assert!(cord.starts_with("hello"));
    /// assert!(cord.starts_with(&Cord::from("hello w")));
    /// assert!(!cord.starts_with("world"));
    /// ```
    pub fn starts_with<P: CordLike + ?Sized>(&self, prefix: &P) -> bool {
        let prefix_size = prefix.len();
        if self.len() < prefix_size {
            return false;
        }
        self.compare_prefix(prefix, prefix_size) == Ordering::Equal
    }

    /// Returns `true` if the cord ends with `suffix`.
    pub fn ends_with<S: CordLike + ?Sized>(&self, suffix: &S) -> bool {
        let my_size = self.len();
        let suffix_size = suffix.len();
        if my_size < suffix_size {
            return false;
        }
        let mut tmp = self.clone();
        tmp.advance(my_size - suffix_size);
        tmp.compare_prefix(suffix, suffix_size) == Ordering::Equal
    }

    /// Compares the cord with `rhs` lexicographically as sequences of
    /// unsigned bytes.
    ///
    /// ```
    /// use core::cmp::Ordering;
    /// use cord_rs::Cord;
    /// let cord = Cord::from("abc");
    /// assert_eq!(cord.compare("abd"), Ordering::Less);
    /// assert_eq!(cord.compare(&Cord::from("abc")), Ordering::Equal);
    /// assert_eq!(cord.compare(b"ab"), Ordering::Greater);
    /// ```
    pub fn compare<R: CordLike + ?Sized>(&self, rhs: &R) -> Ordering {
        if let Some(rhs_cord) = rhs.as_cord()
            && !self.is_tree()
            && !rhs_cord.is_tree()
        {
            return self.data.compare(&rhs_cord.data);
        }
        let lhs_size = self.len();
        let rhs_size = rhs.len();
        if lhs_size == rhs_size {
            return self.compare_prefix(rhs, lhs_size);
        }
        if lhs_size < rhs_size {
            let result = self.compare_prefix(rhs, lhs_size);
            return if result == Ordering::Equal { Ordering::Less } else { result };
        }
        let result = self.compare_prefix(rhs, rhs_size);
        if result == Ordering::Equal { Ordering::Greater } else { result }
    }

    /// Returns `true` if the cord's bytes equal those of `rhs`.
    pub fn equals<R: CordLike + ?Sized>(&self, rhs: &R) -> bool {
        let rhs_size = rhs.len();
        if self.len() != rhs_size {
            return false;
        }
        if let Some(rhs_cord) = rhs.as_cord()
            && self.data.is_same(&rhs_cord.data)
        {
            return true;
        }
        self.compare_prefix(rhs, rhs_size) == Ordering::Equal
    }

    /// Compares the first `size_to_compare` bytes of `self` and `rhs`. Both
    /// must hold at least that many bytes.
    fn compare_prefix<R: CordLike + ?Sized>(&self, rhs: &R, size_to_compare: usize) -> Ordering {
        let lhs_chunk = self.first_chunk();
        let rhs_chunk = rhs.first_chunk();
        let compared_size = lhs_chunk.len().min(rhs_chunk.len());
        debug_assert!(size_to_compare >= compared_size);
        let result = lhs_chunk[..compared_size].cmp(&rhs_chunk[..compared_size]);
        if compared_size == size_to_compare || result != Ordering::Equal {
            return result;
        }
        self.compare_slow_path(rhs, compared_size, size_to_compare)
    }

    #[inline(never)]
    fn compare_slow_path<R: CordLike + ?Sized>(
        &self,
        rhs: &R,
        compared_size: usize,
        mut size_to_compare: usize,
    ) -> Ordering {
        fn advance<'a>(it: &mut Chunks<'a>, chunk: &mut &'a [u8]) -> bool {
            if !chunk.is_empty() {
                return true;
            }
            match it.next() {
                Some(next) => {
                    *chunk = next;
                    true
                }
                None => false,
            }
        }
        let mut lhs_it = self.chunks();
        let mut rhs_it = rhs.chunks();
        // `compared_size` is inside both first chunks.
        let mut lhs_chunk: &[u8] = lhs_it.next().unwrap_or(&[]);
        let mut rhs_chunk: &[u8] = rhs_it.next().unwrap_or(&[]);
        debug_assert!(compared_size <= lhs_chunk.len());
        debug_assert!(compared_size <= rhs_chunk.len());
        lhs_chunk = &lhs_chunk[compared_size..];
        rhs_chunk = &rhs_chunk[compared_size..];
        size_to_compare -= compared_size;

        while advance(&mut lhs_it, &mut lhs_chunk) && advance(&mut rhs_it, &mut rhs_chunk) {
            let n = lhs_chunk.len().min(rhs_chunk.len());
            debug_assert!(size_to_compare >= n);
            size_to_compare -= n;
            let result = lhs_chunk[..n].cmp(&rhs_chunk[..n]);
            if result != Ordering::Equal {
                return result;
            }
            lhs_chunk = &lhs_chunk[n..];
            rhs_chunk = &rhs_chunk[n..];
            if size_to_compare == 0 {
                return Ordering::Equal;
            }
        }
        match (rhs_chunk.is_empty(), lhs_chunk.is_empty()) {
            (true, false) => Ordering::Greater,
            (false, true) => Ordering::Less,
            _ => Ordering::Equal,
        }
    }

    /// Returns the *approximate* number of bytes held by this cord, including
    /// the cord itself. See [`MemoryAccounting`].
    #[must_use]
    pub fn estimated_memory_usage(&self, accounting: MemoryAccounting) -> usize {
        let mut result = core::mem::size_of::<Cord>();
        if let Some(rep) = self.tree() {
            // SAFETY: the tree is live.
            result += unsafe {
                match accounting {
                    MemoryAccounting::FairShare => rep::analysis::estimated_fair_share_memory_usage(rep),
                    MemoryAccounting::TotalMorePrecise => rep::analysis::more_precise_memory_usage(rep),
                    MemoryAccounting::Total => rep::analysis::estimated_memory_usage(rep),
                }
            };
        }
        result
    }
}

/// Searches `needle` (a single chunk) starting at `it`; on success `it` is
/// positioned at the match. Requires a non-empty needle no longer than the
/// remaining haystack.
fn find_impl(it: &mut Cursor<'_>, needle: &[u8]) -> bool {
    debug_assert!(!needle.is_empty());
    debug_assert!(it.remaining() >= needle.len());
    // Go chunk by chunk looking for the first byte of `needle`; on a hit
    // check whether the needle is there, else advance one byte and retry.
    while it.remaining() >= needle.len() {
        let haystack_chunk = it.chunk();
        debug_assert!(!haystack_chunk.is_empty());
        let Some(idx) = haystack_chunk.iter().position(|&b| b == needle[0]) else {
            it.advance(haystack_chunk.len());
            continue;
        };
        it.advance(idx);
        if it.remaining() < needle.len() {
            break;
        }
        if is_slice_at(it.clone(), needle) {
            return true;
        }
        it.advance(1);
    }
    false
}

/// Whether the bytes at `position` start with `needle`. Requires the
/// remaining bytes to be at least `needle.len()`.
fn is_slice_at(mut position: Cursor<'_>, mut needle: &[u8]) -> bool {
    loop {
        let haystack_chunk = position.chunk();
        debug_assert!(!haystack_chunk.is_empty());
        let min_length = haystack_chunk.len().min(needle.len());
        if haystack_chunk[..min_length] != needle[..min_length] {
            return false;
        }
        needle = &needle[min_length..];
        if needle.is_empty() {
            return true;
        }
        position.advance(min_length);
    }
}

/// Whether the bytes at `haystack` start with the chunks of `needle`.
fn is_subcord_at(haystack: &mut Cursor<'_>, needle: Chunks<'_>) -> bool {
    for needle_chunk in needle {
        if !is_slice_at(haystack.clone(), needle_chunk) {
            return false;
        }
        haystack.advance(needle_chunk.len());
    }
    true
}

#[track_caller]
fn resolve_range(range: impl RangeBounds<usize>, len: usize) -> (usize, usize) {
    let start = match range.start_bound() {
        Bound::Included(&s) => s,
        Bound::Excluded(&s) => s.checked_add(1).expect("range start overflow"),
        Bound::Unbounded => 0,
    };
    let end = match range.end_bound() {
        Bound::Included(&e) => e.checked_add(1).expect("range end overflow"),
        Bound::Excluded(&e) => e,
        Bound::Unbounded => len,
    };
    assert!(start <= end, "range start must not be greater than end: {start} <= {end}");
    assert!(end <= len, "range end out of bounds: {end} <= {len}");
    (start, end - start)
}

fn try_resolve_range(range: impl RangeBounds<usize>, len: usize) -> Option<(usize, usize)> {
    let start = match range.start_bound() {
        Bound::Included(&s) => s,
        Bound::Excluded(&s) => s.checked_add(1)?,
        Bound::Unbounded => 0,
    };
    let end = match range.end_bound() {
        Bound::Included(&e) => e.checked_add(1)?,
        Bound::Excluded(&e) => e,
        Bound::Unbounded => len,
    };
    (start <= end && end <= len).then(|| (start, end - start))
}

// --- Trait impls --------------------------------------------------------------

impl Default for Cord {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for Cord {
    /// Clones the cord in O(1) by sharing its buffers.
    #[inline]
    fn clone(&self) -> Self {
        if let Some(tree) = self.tree() {
            // SAFETY: we hold a reference on `tree`; add one for the clone.
            unsafe { ref_rep(tree) };
        }
        Self { data: self.data }
    }

    fn clone_from(&mut self, source: &Self) {
        if core::ptr::eq(self, source) {
            return;
        }
        if !self.is_tree() && !source.is_tree() {
            self.data = source.data;
            return;
        }
        // SAFETY: reference counting as in abseil's `AssignSlow`.
        unsafe {
            let old = self.tree();
            if let Some(src_tree) = source.tree() {
                ref_rep(src_tree);
                self.data.make_tree(src_tree);
            } else {
                self.data = source.data;
            }
            if let Some(old) = old {
                unref(old);
            }
        }
    }
}

impl Drop for Cord {
    #[inline]
    fn drop(&mut self) {
        if let Some(tree) = self.tree() {
            // SAFETY: we hold exactly one reference on `tree`.
            unsafe { unref(tree) };
        }
    }
}

impl Index<usize> for Cord {
    type Output = u8;

    /// Returns the byte at `index`. O(log n); prefer iteration for
    /// sequential access.
    ///
    /// # Panics
    ///
    /// Panics if `index >= len()`.
    #[track_caller]
    #[inline]
    fn index(&self, index: usize) -> &u8 {
        let len = self.len();
        assert!(index < len, "index out of bounds: the len is {len} but the index is {index}");
        self.byte_ref(index)
    }
}

impl Hash for Cord {
    /// Hashes the bytes of the cord independently of how they are chunked.
    ///
    /// The bytes are fed to the hasher as a length prefix followed by fixed
    /// size blocks, so equal cords hash equally regardless of their internal
    /// structure. The hash is *not* guaranteed to equal that of the
    /// equivalent `[u8]`.
    fn hash<H: Hasher>(&self, state: &mut H) {
        const BLOCK: usize = 1024;
        state.write_usize(self.len());
        if let Some(flat) = self.as_flat() {
            for block in flat.chunks(BLOCK) {
                state.write(block);
            }
            return;
        }
        // Re-block the chunks so the write sequence only depends on the data.
        let mut buffer = [0u8; BLOCK];
        let mut filled = 0;
        for chunk in self.chunks() {
            let mut chunk = chunk;
            while !chunk.is_empty() {
                let n = chunk.len().min(BLOCK - filled);
                buffer[filled..filled + n].copy_from_slice(&chunk[..n]);
                filled += n;
                chunk = &chunk[n..];
                if filled == BLOCK {
                    state.write(&buffer);
                    filled = 0;
                }
            }
        }
        if filled > 0 {
            state.write(&buffer[..filled]);
        }
    }
}

impl<T: CordLike + ?Sized> PartialEq<T> for Cord {
    #[inline]
    fn eq(&self, other: &T) -> bool {
        self.equals(other)
    }
}

impl Eq for Cord {}

impl<T: CordLike + ?Sized> PartialOrd<T> for Cord {
    #[inline]
    fn partial_cmp(&self, other: &T) -> Option<Ordering> {
        Some(self.compare(other))
    }
}

impl Ord for Cord {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.compare(other)
    }
}

macro_rules! impl_reverse_eq {
    ($($t:ty),* $(,)?) => {$(
        impl PartialEq<Cord> for $t {
            #[inline]
            fn eq(&self, other: &Cord) -> bool {
                other.equals(self)
            }
        }
        impl PartialOrd<Cord> for $t {
            #[inline]
            fn partial_cmp(&self, other: &Cord) -> Option<Ordering> {
                Some(other.compare(self).reverse())
            }
        }
    )*};
}
impl_reverse_eq!([u8], &[u8], str, &str, Vec<u8>, String);

impl<const N: usize> PartialEq<Cord> for [u8; N] {
    #[inline]
    fn eq(&self, other: &Cord) -> bool {
        other.equals(&self[..])
    }
}

impl<const N: usize> PartialEq<Cord> for &[u8; N] {
    #[inline]
    fn eq(&self, other: &Cord) -> bool {
        other.equals(&self[..])
    }
}

impl fmt::Debug for Cord {
    /// Formats the bytes as a byte string literal, e.g. `b"hello\n"`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("b\"")?;
        for chunk in self.chunks() {
            write!(f, "{}", chunk.escape_ascii())?;
        }
        f.write_str("\"")
    }
}

impl fmt::Display for Cord {
    /// Formats the bytes as UTF-8, replacing invalid sequences with
    /// `U+FFFD` (like `String::from_utf8_lossy`), without allocating.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        crate::io::fmt_lossy(self.chunks(), f)
    }
}

// --- Conversions -----------------------------------------------------------------

impl From<&[u8]> for Cord {
    #[inline]
    fn from(data: &[u8]) -> Self {
        Self::copy_from_slice(data)
    }
}

impl From<&str> for Cord {
    #[inline]
    fn from(data: &str) -> Self {
        Self::copy_from_slice(data.as_bytes())
    }
}

impl From<&Vec<u8>> for Cord {
    #[inline]
    fn from(data: &Vec<u8>) -> Self {
        Self::copy_from_slice(data)
    }
}

impl From<&String> for Cord {
    #[inline]
    fn from(data: &String) -> Self {
        Self::copy_from_slice(data.as_bytes())
    }
}

impl<const N: usize> From<&[u8; N]> for Cord {
    #[inline]
    fn from(data: &[u8; N]) -> Self {
        Self::copy_from_slice(data)
    }
}

impl<const N: usize> From<[u8; N]> for Cord {
    #[inline]
    fn from(data: [u8; N]) -> Self {
        Self::copy_from_slice(&data)
    }
}

impl From<Vec<u8>> for Cord {
    /// Adopts the vector without copying if it is large (more than 511
    /// bytes) and not wasteful (at least half of its capacity is used);
    /// copies it otherwise.
    #[inline]
    fn from(data: Vec<u8>) -> Self {
        let capacity = data.capacity();
        Self::from_owned(data, capacity)
    }
}

impl From<String> for Cord {
    /// See `From<Vec<u8>>`.
    #[inline]
    fn from(data: String) -> Self {
        let capacity = data.capacity();
        Self::from_owned(data, capacity)
    }
}

impl From<Box<[u8]>> for Cord {
    /// Adopts the box without copying if it is more than 511 bytes.
    #[inline]
    fn from(data: Box<[u8]>) -> Self {
        let capacity = data.len();
        Self::from_owned(data, capacity)
    }
}

impl From<Box<str>> for Cord {
    /// See `From<Box<[u8]>>`.
    #[inline]
    fn from(data: Box<str>) -> Self {
        Self::from(data.into_boxed_bytes())
    }
}

impl From<std::sync::Arc<[u8]>> for Cord {
    /// Shares the `Arc` without copying if it is more than 511 bytes.
    #[inline]
    fn from(data: std::sync::Arc<[u8]>) -> Self {
        let capacity = data.len();
        Self::from_owned(data, capacity)
    }
}

impl From<std::sync::Arc<str>> for Cord {
    /// See `From<Arc<[u8]>>`.
    #[inline]
    fn from(data: std::sync::Arc<str>) -> Self {
        let capacity = data.len();
        Self::from_owned(data, capacity)
    }
}

impl From<std::sync::Arc<Vec<u8>>> for Cord {
    /// See `From<Arc<[u8]>>`.
    #[inline]
    fn from(data: std::sync::Arc<Vec<u8>>) -> Self {
        let capacity = data.len();
        Self::from_owned(data, capacity)
    }
}

impl From<std::sync::Arc<String>> for Cord {
    /// See `From<Arc<[u8]>>`.
    #[inline]
    fn from(data: std::sync::Arc<String>) -> Self {
        let capacity = data.len();
        Self::from_owned(data, capacity)
    }
}

impl From<CordBuffer> for Cord {
    #[inline]
    fn from(buffer: CordBuffer) -> Self {
        let mut cord = Cord::new();
        cord.append_buffer(buffer);
        cord
    }
}

impl From<Cord> for Vec<u8> {
    #[inline]
    fn from(cord: Cord) -> Self {
        cord.to_vec()
    }
}

impl TryFrom<Cord> for String {
    type Error = std::string::FromUtf8Error;

    /// Copies the bytes into a `String`, failing if they are not valid UTF-8.
    #[inline]
    fn try_from(cord: Cord) -> Result<Self, Self::Error> {
        String::from_utf8(cord.to_vec())
    }
}

impl<S: CordSource> Extend<S> for Cord {
    fn extend<I: IntoIterator<Item = S>>(&mut self, iter: I) {
        for item in iter {
            self.append(item);
        }
    }
}

impl Extend<u8> for Cord {
    fn extend<I: IntoIterator<Item = u8>>(&mut self, iter: I) {
        let mut buffer = [0u8; 256];
        let mut filled = 0;
        for byte in iter {
            buffer[filled] = byte;
            filled += 1;
            if filled == buffer.len() {
                self.append_slice(&buffer);
                filled = 0;
            }
        }
        self.append_slice(&buffer[..filled]);
    }
}

impl<S: CordSource> FromIterator<S> for Cord {
    fn from_iter<I: IntoIterator<Item = S>>(iter: I) -> Self {
        let mut cord = Cord::new();
        cord.extend(iter);
        cord
    }
}

impl FromIterator<u8> for Cord {
    fn from_iter<I: IntoIterator<Item = u8>>(iter: I) -> Self {
        let mut cord = Cord::new();
        cord.extend(iter);
        cord
    }
}

impl<'a> IntoIterator for &'a Cord {
    type Item = &'a [u8];
    type IntoIter = Chunks<'a>;

    /// Iterates over the chunks of the cord.
    #[inline]
    fn into_iter(self) -> Chunks<'a> {
        self.chunks()
    }
}
