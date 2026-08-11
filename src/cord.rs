//! The [`Cord`] type.

use core::cmp::Ordering;
use core::fmt;
use core::hash::{Hash, Hasher};
use core::mem::MaybeUninit;
use core::ops::{Bound, Index, RangeBounds};
use core::ptr::NonNull;

use crate::buffer::{ConsumedBuffer, CordBuffer};
use crate::inline_data::{InlineData, Repr};
use crate::iter::{Bytes, Chunks, Cursor};
use crate::rep::btree::{BtreePtr, CordRepBtree, as_btree};
use crate::rep::external::{CordRepExternal, StableBytes};
use crate::rep::flat::{self, MAX_FLAT_LENGTH};
use crate::rep::{
    self, CordRep, CordRepSubstring, MAX_BYTES_TO_COPY, MAX_INLINE, OwnedRep, RepPtr, RepRef, RepView, unref,
};
use crate::source::{CordLike, CordSource};

/// Memory accounting modes for [`Cord::estimated_memory_usage`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
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
fn new_tree(data: &[u8], alloc_hint: usize) -> OwnedRep {
    debug_assert!(!data.is_empty());
    // SAFETY: `data` is non-empty (checked above), so both `flat::create`
    // calls (whole-`data` or the `MAX_FLAT_LENGTH`-sized first chunk) get a
    // non-empty slice, and `append_data` gets the (possibly empty)
    // remainder of an already non-empty `data`. Each constructor returns a
    // fresh rep with a refcount of one, adopted into the `OwnedRep`.
    unsafe {
        if data.len() <= MAX_FLAT_LENGTH {
            return OwnedRep::from_raw(flat::create(data, alloc_hint));
        }
        let first = flat::create(&data[..MAX_FLAT_LENGTH], 0);
        let root = CordRepBtree::create(first);
        let tree = CordRepBtree::append_data(root, &data[MAX_FLAT_LENGTH..], alloc_hint);
        OwnedRep::from_raw(tree.as_rep())
    }
}

/// Returns `rep` converted into a btree (a no-op if it already is one),
/// adopting `rep`'s reference.
#[inline]
fn force_btree(rep: OwnedRep) -> *mut CordRepBtree {
    let raw = rep.into_raw();
    // SAFETY: `raw` is `rep`'s just-transferred-out reference: if it's
    // already a btree it's returned unchanged (same reference), otherwise
    // `CordRepBtree::create` adopts it into a freshly created btree. Either
    // way the original reference is preserved, now represented by the
    // returned pointer.
    unsafe { if raw.is_btree() { raw.cast() } else { CordRepBtree::create(raw) } }
}

/// Creates a rep from a large owned buffer. Copies the data if the buffer is
/// small or wasteful, adopts it otherwise. Requires `len > MAX_INLINE`.
fn rep_from_owned<O: StableBytes>(owner: O, capacity: usize) -> OwnedRep {
    let bytes = owner.as_bytes();
    debug_assert!(bytes.len() > MAX_INLINE);
    if bytes.len() <= MAX_BYTES_TO_COPY || bytes.len() < capacity / 2 {
        // Short: copy to avoid the external node overhead.
        // Wasteful: copy to avoid pinning too much unused memory.
        return new_tree(bytes, 0);
    }
    // SAFETY: `CordRepExternal::create` returns a fresh non-empty rep.
    unsafe { OwnedRep::from_raw(CordRepExternal::create(owner)) }
}

/// Attempts to append (a prefix of) `src` in place into the writable end of
/// `root`: `root`'s own spare flat capacity, or a non-full flat found at the
/// tree's right-most leaf. Returns the number of bytes copied.
fn prepare_append_region(root: &mut rep::UniqueRep<'_>, src: &[u8]) -> usize {
    if root.as_ref().is_btree() {
        // SAFETY: `root`'s uniqueness proves `ref_is_one()` at the top of
        // the tree, which is `get_append_buffer`'s precondition; it
        // establishes uniqueness of descendants dynamically as it walks
        // down, and (on success) returns a region already accounted for in
        // every node's length on the path, which this call fills
        // immediately.
        return unsafe {
            match CordRepBtree::get_append_buffer(root.as_ptr().cast(), src.len()) {
                Some((ptr, n)) => {
                    core::ptr::copy_nonoverlapping(src.as_ptr(), ptr, n);
                    n
                }
                None => 0,
            }
        };
    }
    if !root.as_ref().is_flat() {
        return 0;
    }
    let in_use = root.as_ref().len();
    // SAFETY: `root.as_ref().is_flat()` was just confirmed above.
    let spare = unsafe { root.flat_spare_capacity_mut() };
    let n = spare.len().min(src.len());
    if n == 0 {
        return 0;
    }
    // Vec-like order: fill the spare capacity, then commit the new length.
    spare[..n].write_copy_of_slice(&src[..n]);
    root.set_len(in_use + n);
    n
}

/// Returns the flat data of `rep` if it is a single contiguous buffer.
fn get_flat_aux(rep: RepRef<'_>) -> Option<&[u8]> {
    if rep.is_btree() {
        // SAFETY: tag checked on the line above.
        unsafe { rep.btree_unchecked() }.as_flat()
    } else {
        // SUBSTRING, EXTERNAL or FLAT: all are data edges.
        debug_assert!(rep.is_data_edge());
        // SAFETY: not BTREE (checked above) means SUBSTRING, EXTERNAL or
        // FLAT, all of which are data edges (see the comment above).
        Some(unsafe { rep.data() })
    }
}

/// `extract_append_buffer` for any rep type. Consumes `rep`'s reference,
/// which comes back split between the returned `ExtractResult`'s `tree` and
/// `extracted` fields: exactly one is non-`None` if extraction was possible
/// (repurposed as the extracted buffer, and whatever remains of the tree, if
/// any); otherwise `tree` alone carries the reference back unchanged.
fn extract_append_buffer(rep: OwnedRep, min_capacity: usize) -> rep::btree::ExtractResult {
    use rep::btree::ExtractResult;
    let raw = rep.into_raw();
    // SAFETY: `raw` is `rep`'s just-transferred reference (live per
    // `OwnedRep`'s invariant); reading its tag is a read-only query, and if
    // it is a btree, adopting it into `CordRepBtree::extract_append_buffer`
    // matches that fn's own contract.
    unsafe {
        if raw.is_btree() {
            return CordRepBtree::extract_append_buffer(as_btree(raw), min_capacity);
        }
    }
    // SAFETY: `raw` is a live rep (transferred from `rep` above); reading
    // its tag, refcount and (if flat) capacity are all read-only queries.
    unsafe {
        if raw.is_flat() && raw.ref_is_one() && flat::capacity(raw) - raw.length() >= min_capacity {
            return ExtractResult { tree: None, extracted: NonNull::new(raw) };
        }
    }
    ExtractResult { tree: NonNull::new(raw), extracted: None }
}

#[cold]
#[inline(never)]
#[track_caller]
fn length_overflow() -> ! {
    panic!("Cord length overflow")
}

// --- Internal representation helpers --------------------------------------

impl Cord {
    /// The tree, if this cord is not inline. Kept as a raw-pointer accessor
    /// for `lib.rs` call sites (`validate`/`dump`/`make_substring`) not yet
    /// converted to the handle types; new code should prefer
    /// [`tree_ref`](Self::tree_ref).
    #[inline]
    pub(crate) fn tree(&self) -> Option<*mut CordRep> {
        self.data.tree()
    }

    /// The tree, if this cord is not inline, as a borrowed [`RepRef`].
    #[inline]
    pub(crate) fn tree_ref(&self) -> Option<RepRef<'_>> {
        match self.data.view() {
            Repr::Tree(tree) => Some(tree),
            Repr::Inline(_) => None,
        }
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

    /// Creates a cord holding `rep`, adopting its reference. Requires
    /// `rep.len() != 0`.
    #[inline]
    pub(crate) fn from_owned_rep(rep: OwnedRep) -> Self {
        // SAFETY: `rep` is a live rep per `OwnedRep`'s own invariant.
        unsafe { rep::debug_assert_nonempty_rep(rep.as_ref().as_ptr()) };
        Self { data: InlineData::from_tree(rep) }
    }

    /// Creates an inline cord. Requires `data.len() <= MAX_INLINE`.
    #[inline]
    pub(crate) fn from_inline(data: &[u8]) -> Self {
        Self { data: InlineData::inline_from(data) }
    }

    /// Commits `rep` as `self`'s new tree, or resets `self` to empty inline
    /// data if `rep` is `None`. Does *not* touch whatever tree `self` may
    /// have held before — the caller must have already accounted for its
    /// reference (used after `extract_append_buffer`, which repurposes it
    /// into `rep`'s fields itself).
    #[inline]
    fn set_tree_or_empty(&mut self, rep: Option<OwnedRep>) {
        match rep {
            Some(rep) => self.data.set_tree(rep),
            None => self.data = InlineData::new(),
        }
    }

    /// Creates a flat from the inline data with `extra` bytes of capacity.
    ///
    /// # Safety
    ///
    /// `self` must currently be inline (`!self.is_tree()`).
    unsafe fn make_flat_with_extra_capacity(&self, extra: usize) -> *mut CordRep {
        let len = self.data.inline_size();
        // SAFETY: `flat::new` returns a fresh flat with a refcount of one
        // and capacity `>= MIN_FLAT_LENGTH > MAX_INLINE` (see flat.rs's
        // const assertion), so the 15-byte inline tail always fits
        // regardless of `len`/`extra`; `set_length` and the payload pointer
        // below are sound because `result` is exclusively owned by this
        // call.
        unsafe {
            let result = flat::new(len + extra);
            result.set_length(len);
            let capacity = flat::capacity(result);
            let dst = core::slice::from_raw_parts_mut(flat::data(result).cast::<MaybeUninit<u8>>(), capacity);
            self.data.copy_max_inline_to(dst);
            result
        }
    }

    /// Appends `tree` (a non-empty rep) to `self`, folding it into `self`'s
    /// existing representation, adopting `tree`'s reference.
    fn append_tree(&mut self, tree: OwnedRep) {
        debug_assert!(tree.len() != 0);
        if self.is_tree() {
            let cur = self.data.take_tree().expect("is_tree() checked above");
            // SAFETY: `append`'s raw result carries the reference that
            // `force_btree`'s consumed input transferred in; adopted here
            // and installed below.
            let owned = unsafe {
                let btree = force_btree(cur);
                let appended = CordRepBtree::append(btree, tree.into_raw());
                OwnedRep::from_raw(appended.as_rep())
            };
            self.data.set_tree(owned);
        } else {
            let mut tree = tree;
            if !self.data.is_empty() {
                // SAFETY: creates a fresh single-leaf btree from the inline
                // data and appends `tree` into it.
                tree = unsafe {
                    let flat = self.make_flat_with_extra_capacity(0);
                    let btree = CordRepBtree::create(flat);
                    let appended = CordRepBtree::append(btree, tree.into_raw());
                    OwnedRep::from_raw(appended.as_rep())
                };
            }
            self.data.set_tree(tree);
        }
    }

    /// Prepends `tree` (a non-empty rep) to `self`, folding it into `self`'s
    /// existing representation, adopting `tree`'s reference.
    fn prepend_tree(&mut self, tree: OwnedRep) {
        debug_assert!(tree.len() != 0);
        if self.is_tree() {
            let cur = self.data.take_tree().expect("is_tree() checked above");
            // SAFETY: see `append_tree`.
            let owned = unsafe {
                let btree = force_btree(cur);
                let prepended = CordRepBtree::prepend(btree, tree.into_raw());
                OwnedRep::from_raw(prepended.as_rep())
            };
            self.data.set_tree(owned);
        } else {
            let mut tree = tree;
            if !self.data.is_empty() {
                // SAFETY: see `append_tree`.
                tree = unsafe {
                    let flat = self.make_flat_with_extra_capacity(0);
                    let btree = CordRepBtree::create(flat);
                    let prepended = CordRepBtree::prepend(btree, tree.into_raw());
                    OwnedRep::from_raw(prepended.as_rep())
                };
            }
            self.data.set_tree(tree);
        }
    }

    /// Appends `src`, using spare capacity in the last flat where possible.
    pub(crate) fn append_slice(&mut self, mut src: &[u8]) {
        if src.is_empty() {
            return;
        }
        let owned_cur: OwnedRep;
        if let Some(mut unique) = self.data.tree_unique() {
            let appended = prepare_append_region(&mut unique, src);
            src = &src[appended..];
            if src.is_empty() {
                // In-place growth above already updated the existing tree;
                // `self.data`'s reference is untouched.
                return;
            }
            owned_cur = self.data.take_tree().expect("tree_unique() proved this is a tree");
        } else if self.is_tree() {
            owned_cur = self.data.take_tree().expect("is_tree() checked above");
        } else {
            // Try to fit in the inline buffer if possible.
            let inline_length = self.data.inline_size();
            if src.len() <= MAX_INLINE - inline_length {
                self.data.push_back_inline(src);
                return;
            }
            // Allocate a flat that is a perfect fit on the first append
            // exceeding the inline size. Subsequent growth is amortized
            // until we reach the maximum flat size.
            // SAFETY: creates a fresh flat and copies the existing inline
            // bytes plus as much of `src` as fits into it; the fresh flat's
            // single reference is adopted into `owned_cur` (sole owner).
            owned_cur = unsafe {
                let r = flat::new(inline_length + src.len());
                let appended = src.len().min(flat::capacity(r) - inline_length);
                core::ptr::copy_nonoverlapping(self.data.as_chars(), flat::data(r), inline_length);
                core::ptr::copy_nonoverlapping(src.as_ptr(), flat::data(r).add(inline_length), appended);
                r.set_length(inline_length + appended);
                src = &src[appended..];
                OwnedRep::from_raw(r)
            };
            if src.is_empty() {
                self.data.set_tree(owned_cur);
                return;
            }
        }

        // Keep abseil's 10% growth rate.
        // SAFETY: `append_data`'s raw result carries the reference that
        // `force_btree`'s consumed input transferred in; adopted here and
        // installed below.
        let owned = unsafe {
            let tree = force_btree(owned_cur);
            let min_growth = (tree.length() / 10).max(src.len());
            let tree = CordRepBtree::append_data(tree, src, min_growth - src.len());
            OwnedRep::from_raw(tree.as_rep())
        };
        self.data.set_tree(owned);
    }

    /// Prepends `src`.
    pub(crate) fn prepend_slice(&mut self, src: &[u8]) {
        if src.is_empty() {
            return;
        }
        if !self.is_tree() {
            let cur_size = self.data.inline_size();
            if cur_size + src.len() <= MAX_INLINE {
                self.data.push_front_inline(src);
                return;
            }
            self.prepend_tree(new_tree(src, 0));
            return;
        }

        let owned_cur = self.data.take_tree().expect("is_tree() checked above");
        // Unlike append capacity, spare bytes at the end of a flat cannot
        // absorb a later prepend, so request no unusable extra capacity.
        // SAFETY: `prepend_data`'s raw result carries the reference that
        // `force_btree`'s consumed input transferred in; adopted here and
        // installed below.
        let owned = unsafe {
            let tree = force_btree(owned_cur);
            let tree = CordRepBtree::prepend_data(tree, src, 0);
            OwnedRep::from_raw(tree.as_rep())
        };
        self.data.set_tree(owned);
    }

    /// Appends `src` with precise sizing (no spare capacity is used or
    /// allocated). Requires `0 < src.len() <= MAX_FLAT_LENGTH`.
    ///
    /// # Safety
    ///
    /// `0 < src.len() <= MAX_FLAT_LENGTH` must hold (`flat::create`'s own
    /// precondition on the else branch).
    unsafe fn append_precise(&mut self, src: &[u8]) {
        debug_assert!(!src.is_empty());
        debug_assert!(src.len() <= MAX_FLAT_LENGTH);
        if self.remaining_inline_capacity() >= src.len() {
            self.data.push_back_inline(src);
        } else {
            // SAFETY: `src.len() <= MAX_FLAT_LENGTH` per this fn's own
            // contract.
            self.append_tree(unsafe { OwnedRep::from_raw(flat::create(src, 0)) });
        }
    }

    /// Prepends `src` with precise sizing. Requires `0 < src.len() <= MAX_FLAT_LENGTH`.
    ///
    /// # Safety
    ///
    /// Same contract as [`append_precise`](Self::append_precise).
    unsafe fn prepend_precise(&mut self, src: &[u8]) {
        debug_assert!(!src.is_empty());
        debug_assert!(src.len() <= MAX_FLAT_LENGTH);
        if self.remaining_inline_capacity() >= src.len() {
            self.data.push_front_inline(src);
        } else {
            // SAFETY: `src.len() <= MAX_FLAT_LENGTH` per this fn's own
            // contract.
            self.prepend_tree(unsafe { OwnedRep::from_raw(flat::create(src, 0)) });
        }
    }

    #[inline]
    fn remaining_inline_capacity(&self) -> usize {
        if self.is_tree() { 0 } else { MAX_INLINE - self.data.inline_size() }
    }

    /// Appends another cord (borrowed).
    pub(crate) fn append_cord(&mut self, src: &Cord) {
        if src.is_empty() {
            return;
        }
        if src.len() > usize::MAX - self.len() {
            length_overflow();
        }
        if self.is_empty() {
            // The destination is empty: take the tree or copy inline data.
            match src.tree_ref() {
                Some(tree) => self.data.set_tree(tree.to_owned()),
                None => self.data = src.data,
            }
            return;
        }
        // For short cords it is faster to copy data if there is room.
        let src_size = src.len();
        if src_size <= MAX_BYTES_TO_COPY {
            match src.tree_ref() {
                None => self.append_slice(src.inline_slice()),
                Some(tree) if tree.is_flat() => self.append_slice(
                    // SAFETY: `tree.is_flat()` (match guard) implies `is_data_edge()`.
                    unsafe { tree.data() },
                ),
                Some(_) => {
                    for chunk in src.chunks() {
                        self.append_slice(chunk);
                    }
                }
            }
            return;
        }
        // Guaranteed to be a tree (MAX_BYTES_TO_COPY > MAX_INLINE).
        self.append_tree(
            src.tree_ref().expect("cord larger than MAX_BYTES_TO_COPY must be a tree").to_owned(),
        );
    }

    /// Appends another cord (owned), stealing its tree.
    pub(crate) fn append_owned_cord(&mut self, mut src: Cord) {
        if src.is_empty() {
            return;
        }
        if src.len() > usize::MAX - self.len() {
            length_overflow();
        }
        if self.is_empty() {
            match src.data.take_tree() {
                Some(tree) => self.data.set_tree(tree),
                None => self.data = src.data,
            }
            return;
        }
        let src_size = src.len();
        if src_size <= MAX_BYTES_TO_COPY {
            match src.tree_ref() {
                None => self.append_slice(src.inline_slice()),
                Some(tree) if tree.is_flat() => self.append_slice(
                    // SAFETY: `tree.is_flat()` (match guard) implies `is_data_edge()`.
                    unsafe { tree.data() },
                ),
                Some(_) => {
                    for chunk in src.chunks() {
                        self.append_slice(chunk);
                    }
                }
            }
            return;
        }
        self.append_tree(src.data.take_tree().expect("cord larger than MAX_BYTES_TO_COPY must be a tree"));
    }

    /// Prepends another cord (borrowed).
    pub(crate) fn prepend_cord(&mut self, src: &Cord) {
        if src.is_empty() {
            return;
        }
        if src.len() > usize::MAX - self.len() {
            length_overflow();
        }
        if let Some(tree) = src.tree_ref() {
            self.prepend_tree(tree.to_owned());
            return;
        }
        self.prepend_slice(src.inline_slice());
    }

    /// Prepends another cord (owned), stealing its tree.
    pub(crate) fn prepend_owned_cord(&mut self, mut src: Cord) {
        if src.is_empty() {
            return;
        }
        if src.len() > usize::MAX - self.len() {
            length_overflow();
        }
        if let Some(tree) = src.data.take_tree() {
            self.prepend_tree(tree);
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
            self.append_tree(rep_from_owned(owner, capacity));
        }
    }

    /// Prepends a large owned buffer, adopting it if worthwhile.
    pub(crate) fn prepend_owned<O: StableBytes>(&mut self, owner: O, capacity: usize) {
        let len = owner.as_bytes().len();
        if len <= MAX_BYTES_TO_COPY {
            self.prepend_slice(owner.as_bytes());
        } else {
            self.prepend_tree(rep_from_owned(owner, capacity));
        }
    }

    /// Creates a cord from an owned buffer, adopting it if worthwhile.
    pub(crate) fn from_owned<O: StableBytes>(owner: O, capacity: usize) -> Self {
        let bytes = owner.as_bytes();
        if bytes.len() <= MAX_INLINE {
            return Self::from_inline(bytes);
        }
        Self::from_owned_rep(rep_from_owned(owner, capacity))
    }

    /// Appends the contents of a [`CordBuffer`].
    pub(crate) fn append_buffer(&mut self, buffer: CordBuffer) {
        if buffer.is_empty() {
            return;
        }
        match buffer.consume() {
            // SAFETY: a consumed buffer's rep is a fresh flat with a
            // refcount of one.
            ConsumedBuffer::Rep(rep) => self.append_tree(unsafe { OwnedRep::from_raw(rep) }),
            // SAFETY: a consumed short buffer's slice is at most
            // `MAX_FLAT_LENGTH`.
            ConsumedBuffer::Short(short) => unsafe { self.append_precise(short.as_slice()) },
        }
    }

    /// Prepends the contents of a [`CordBuffer`].
    pub(crate) fn prepend_buffer(&mut self, buffer: CordBuffer) {
        if buffer.is_empty() {
            return;
        }
        match buffer.consume() {
            // SAFETY: as in `append_buffer`.
            ConsumedBuffer::Rep(rep) => self.prepend_tree(unsafe { OwnedRep::from_raw(rep) }),
            // SAFETY: as in `append_buffer`.
            ConsumedBuffer::Short(short) => unsafe { self.prepend_precise(short.as_slice()) },
        }
    }

    /// Locates the first flat or external chunk without initializing an
    /// iterator. Returns empty for an empty cord.
    pub(crate) fn first_chunk(&self) -> &[u8] {
        let Some(node) = self.tree_ref() else {
            return self.data.inline_slice();
        };
        if !node.is_btree() {
            // FLAT, EXTERNAL or a SUBSTRING thereof: all are data edges.
            // SAFETY: see the comment above.
            return unsafe { node.data() };
        }
        // SAFETY: tag checked on the line above.
        let mut tree = unsafe { node.btree_unchecked() };
        let mut height = tree.height();
        while height > 0 {
            height -= 1;
            // SAFETY: this loop only reaches non-leaf nodes (`height > 0`
            // here); a well-formed btree node is never empty, so
            // `edge_at`'s non-empty requirement holds, and the returned
            // edge is itself a btree node of one lesser height, satisfying
            // `btree_unchecked`'s requirement.
            tree = unsafe { tree.edge_at::<{ rep::btree::FRONT }>().btree_unchecked() };
        }
        // SAFETY: the loop above runs until `height == 0`, so `tree` is a
        // leaf; a well-formed btree node is never empty, so `tree.begin()`
        // is a valid index in `[begin(), end())`.
        unsafe { tree.data(tree.begin()) }
    }

    /// Slow path of [`flatten`](Self::flatten).
    fn flatten_slow_path(&mut self) {
        let total_size = self.len();
        let new_tree = if total_size <= MAX_FLAT_LENGTH {
            // SAFETY: `flat::new` returns a fresh flat with a refcount of
            // one.
            let mut owned = unsafe { OwnedRep::from_raw(flat::new(total_size)) };
            let mut unique = owned.try_unique().expect("freshly allocated flat has refcount one");
            // SAFETY: `unique` wraps the flat just allocated by `flat::new`
            // above.
            let spare = unsafe { unique.flat_spare_capacity_mut() };
            self.copy_to_uninit(&mut spare[..total_size]);
            unique.set_len(total_size);
            owned
        } else {
            let mut buffer: Vec<u8> = Vec::with_capacity(total_size);
            self.copy_to_uninit(buffer.spare_capacity_mut());
            // SAFETY: `copy_to_uninit` just initialized the first
            // `total_size` bytes (`buffer`'s full capacity) of `buffer`'s
            // spare capacity.
            unsafe { buffer.set_len(total_size) };
            // SAFETY: `CordRepExternal::create` returns a fresh non-empty
            // rep.
            unsafe { OwnedRep::from_raw(CordRepExternal::create(buffer)) }
        };
        self.data.take_tree(); // Drops (unrefs) the old tree.
        self.data.set_tree(new_tree);
    }

    /// Copies all bytes to `dst`, which must have room for `self.len()`
    /// bytes.
    fn copy_to_uninit(&self, dst: &mut [MaybeUninit<u8>]) {
        if let Some(chunk) = self.as_flat() {
            dst[..chunk.len()].write_copy_of_slice(chunk);
            return;
        }
        let mut dst = dst;
        for chunk in self.chunks() {
            let (head, tail) = dst.split_at_mut(chunk.len());
            head.write_copy_of_slice(chunk);
            dst = tail;
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
        // SAFETY: `CordRepExternal::create` returns a fresh non-empty rep.
        Self::from_owned_rep(unsafe { OwnedRep::from_raw(CordRepExternal::create(bytes)) })
    }

    /// Creates a cord by copying `data`.
    ///
    /// Equivalent to `Cord::from(data)`.
    #[must_use]
    pub fn copy_from_slice(data: &[u8]) -> Self {
        if data.len() <= MAX_INLINE {
            return Self::from_inline(data);
        }
        Self::from_owned_rep(new_tree(data, 0))
    }

    /// Returns the number of bytes in the cord.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
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
        // `take_tree` already resets `self.data` to empty inline data when
        // there was a tree to drop (unref); only the no-tree case (already
        // inline, possibly non-empty) still needs the explicit reset below,
        // so this never stores the empty value twice.
        if self.data.take_tree().is_none() {
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
    /// Unlike `append`, `prepend` does not currently amortize into spare
    /// front capacity across calls: each call allocates its own buffer for
    /// `src`, so repeated prepending is not O(1) amortized the way repeated
    /// appending is.
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
        if self.is_tree() {
            let owned = self.data.take_tree().expect("is_tree() checked above");
            let result = extract_append_buffer(owned, min_capacity);
            // On failure `result.tree` carries the taken reference back
            // unchanged; on success it is whatever remains. Reinstall it
            // either way before returning.
            // SAFETY: `result.tree`, when `Some`, carries exactly one
            // reference (`extract_append_buffer`'s contract), adopted here.
            self.set_tree_or_empty(result.tree.map(|t| unsafe { OwnedRep::from_raw(t.as_ptr()) }));
            if let Some(extracted) = result.extracted {
                // SAFETY: `extracted` is a uniquely owned flat with
                // refcount one, no longer referenced by the tree.
                return unsafe { CordBuffer::from_flat(extracted.as_ptr()) };
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
        let Some(tree) = self.tree_ref() else {
            self.data.drop_front_inline(n);
            return;
        };
        if n >= tree.len() {
            self.data.take_tree(); // Drops (unrefs) the old tree.
            return;
        }
        if tree.is_btree() {
            // SAFETY: standard rep manipulation; see abseil's
            // `RemovePrefix`. `sub_tree` returns a fresh reference;
            // `unref` releases `self`'s original one.
            let owned = unsafe {
                let raw = tree.as_ptr();
                let sub = CordRepBtree::sub_tree(as_btree(raw), n, tree.len() - n);
                unref(raw);
                OwnedRep::from_raw(sub)
            };
            self.data.set_tree(owned);
            return;
        }
        let tree_len = tree.len();
        let is_substring = tree.is_substring();
        let raw = tree.as_ptr();
        if is_substring && let Some(mut unique) = self.data.tree_unique() {
            // In-place mutation: shift the substring's start forward.
            // SAFETY: `unique` wraps `self.data`'s tree, the same rep `tree`
            // borrows, and `is_substring` (`tree.is_substring()` above)
            // confirms its tag is SUBSTRING.
            *unsafe { unique.substring_start_mut() } += n;
            unique.set_len(tree_len - n);
            return;
        }
        // SAFETY: standard rep manipulation; `substring` returns a fresh
        // reference, `unref` releases `self`'s original one.
        let owned = unsafe {
            let rep = CordRepSubstring::substring(raw, n, tree_len - n);
            unref(raw);
            OwnedRep::from_raw(rep)
        };
        self.data.set_tree(owned);
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
        let Some(tree) = self.tree_ref() else {
            self.data.truncate_inline(len);
            return;
        };
        if n >= tree.len() {
            self.data.take_tree(); // Drops (unrefs) the old tree.
            return;
        }
        if tree.is_btree() {
            // SAFETY: standard rep manipulation; see abseil's
            // `RemoveSuffix`. `remove_suffix` adopts `tree`'s reference
            // (no separate `unref` needed).
            let owned =
                unsafe { OwnedRep::from_raw(CordRepBtree::remove_suffix(as_btree(tree.as_ptr()), n)) };
            self.data.set_tree(owned);
            return;
        }
        let tree_len = tree.len();
        let is_external = tree.is_external();
        let raw = tree.as_ptr();
        if !is_external && let Some(mut unique) = self.data.tree_unique() {
            debug_assert!(unique.as_ref().is_flat() || unique.as_ref().is_substring());
            // In-place mutation: shrink the length.
            unique.set_len(tree_len - n);
            return;
        }
        // SAFETY: standard rep manipulation; `substring` returns a fresh
        // reference, `unref` releases `self`'s original one.
        let owned = unsafe {
            let rep = CordRepSubstring::substring(raw, 0, tree_len - n);
            unref(raw);
            OwnedRep::from_raw(rep)
        };
        self.data.set_tree(owned);
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
    #[must_use]
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
        let Some(tree) = self.tree_ref() else {
            return Cord::from_inline(&self.data.inline_slice()[pos..pos + new_size]);
        };
        if new_size <= MAX_INLINE {
            let mut it = self.chunks();
            it.advance_bytes(pos);
            // `it` is discarded right here, so the final positioning
            // update would be dead work.
            return Cord { data: it.gather_inline::<false>(new_size) };
        }
        // SAFETY: `tree` is live; `sub_tree` / `substring` return a new
        // reference.
        let owned = unsafe {
            let raw = tree.as_ptr();
            let rep = if tree.is_btree() {
                CordRepBtree::sub_tree(as_btree(raw), pos, new_size)
            } else {
                CordRepSubstring::substring(raw, pos, new_size)
            };
            OwnedRep::from_raw(rep)
        };
        Cord::from_owned_rep(owned)
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
        match self.data.view() {
            Repr::Inline(bytes) => Some(bytes),
            Repr::Tree(rep) => get_flat_aux(rep),
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
        self.copy_to_uninit(vec.spare_capacity_mut());
        // SAFETY: `copy_to_uninit` just initialized exactly `self.len()`
        // bytes (`vec`'s full capacity) of `vec`'s spare capacity.
        unsafe { vec.set_len(self.len()) };
        vec
    }

    /// Copies up to `dst.len()` bytes from the start of the cord into `dst`
    /// and returns the number of bytes copied. The cord is not modified.
    ///
    /// (With the `bytes` feature, `bytes::Buf::copy_to_slice` is the
    /// *consuming* variant.)
    pub fn copy_prefix_to(&self, dst: &mut [u8]) -> usize {
        if let Some(flat) = self.as_flat() {
            // Contiguous (inline or single flat): one memcpy.
            let n = flat.len().min(dst.len());
            dst[..n].copy_from_slice(&flat[..n]);
            return n;
        }
        let mut dst = dst;
        let result = self.len().min(dst.len());
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
        let Some(mut rep) = self.tree_ref() else {
            return &self.data.inline_slice()[index];
        };
        let mut offset = index;
        loop {
            debug_assert!(offset < rep.len());
            match rep.view() {
                RepView::Btree(mut node) => {
                    let mut height = node.height();
                    loop {
                        // SAFETY: on the first iteration, `offset <
                        // rep.len() == node.len()` (the outer loop's
                        // `debug_assert` above, `node` being `rep` viewed as
                        // a btree); on later iterations, `offset` was just
                        // set to `front.n`, which `index_of`'s own
                        // postcondition keeps `< node.len()` for the
                        // reassigned `node` below.
                        let front = unsafe { node.index_of(offset) };
                        if height == 0 {
                            // SAFETY: `height == 0` means `node` is a leaf;
                            // `index_of`'s postcondition keeps `front.index`
                            // in `[begin(), end())` for the in-range
                            // `offset` established above.
                            return &unsafe { node.data(front.index) }[front.n];
                        }
                        height -= 1;
                        offset = front.n;
                        // SAFETY: `front.index` is in bounds (see above),
                        // and every non-leaf edge of a well-formed btree is
                        // a btree node (`height > 0` here).
                        node = unsafe { node.edge(front.index).btree_unchecked() };
                    }
                }
                RepView::Substring { start, child } => {
                    offset += start;
                    rep = child;
                }
                // SAFETY: `Btree` and `Substring` are matched above, so this
                // arm only sees `Flat`/`External`, both data edges.
                _ => return &unsafe { rep.data() }[offset],
            }
        }
    }

    /// Returns the position of the first occurrence of `needle`, or `None`.
    /// An empty needle is found at position 0.
    ///
    /// Uses a naive search, not Boyer-Moore/KMP: worst case is O(n·m) for a
    /// haystack of length n and needle of length m.
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
    ///
    /// Currently implemented by cloning the cord and calling
    /// [`advance`](Self::advance) on the clone, which for a tree cord forces
    /// copy-on-write down the tree spine (the clone shares the tree, so it
    /// is never privately owned) — this allocates even though the cord
    /// itself is not modified.
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
        if let Some(rhs_cord) = rhs.as_cord() {
            if self.data.is_same(&rhs_cord.data) {
                return Ordering::Equal;
            }
            if !self.is_tree() && !rhs_cord.is_tree() {
                return self.data.compare(&rhs_cord.data);
            }
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
        if let Some(rep) = self.tree_ref() {
            // SAFETY: `rep` is live for as long as this borrow of `self`,
            // which outlives this call.
            result += unsafe {
                match accounting {
                    MemoryAccounting::FairShare => {
                        rep::analysis::estimated_fair_share_memory_usage(rep.as_ptr())
                    }
                    MemoryAccounting::TotalMorePrecise => {
                        rep::analysis::more_precise_memory_usage(rep.as_ptr())
                    }
                    MemoryAccounting::Total => rep::analysis::estimated_memory_usage(rep.as_ptr()),
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
    // Wording matches `[T]`'s range-indexing panics (`slice_index_order_fail`
    // / `slice_end_index_len_fail` in core::slice::index).
    assert!(start <= end, "slice index starts at {start} but ends at {end}");
    assert!(end <= len, "range end index {end} out of range for slice of length {len}");
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
        Self { data: self.data.clone_with_ref() }
    }

    fn clone_from(&mut self, source: &Self) {
        if core::ptr::eq(self, source) {
            return;
        }
        if !self.is_tree() && !source.is_tree() {
            self.data = source.data;
            return;
        }
        // Reference counting as in abseil's `AssignSlow`: increment
        // `source`'s tree (if any) before releasing `self`'s old one, so a
        // tree shared between `self` and `source` never drops to zero
        // mid-reassignment.
        let old = self.data.take_tree();
        match source.tree_ref() {
            Some(src_tree) => self.data.set_tree(src_tree.to_owned()),
            None => self.data = source.data,
        }
        drop(old);
    }
}

impl Drop for Cord {
    #[inline]
    fn drop(&mut self) {
        self.data.release_for_drop();
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
    /// `U+FFFD` (like `String::from_utf8_lossy`). Honors width, fill,
    /// alignment and precision the same way `str`'s `Display` does; the
    /// common case (none of those set) streams the decoded text straight to
    /// the sink without allocating.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if f.width().is_none() && f.precision().is_none() && f.align().is_none() && !f.sign_aware_zero_pad() {
            return crate::io::fmt_lossy(self.chunks(), f);
        }
        // A flag that affects layout is set: materialize the lossy string so
        // `f.pad` can apply width/fill/align/precision the way it would for
        // a plain `&str`.
        let mut decoded = String::with_capacity(self.len());
        crate::io::fmt_lossy(self.chunks(), &mut decoded)?;
        f.pad(&decoded)
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
        // Buffers up to 256 bytes at a time before appending, so a
        // panicking `iter` (e.g. a user `Iterator::next` that panics
        // partway through) must not lose the up-to-255 already-consumed
        // bytes sitting in the buffer. `Guard::drop` flushes them on
        // unwinding; the happy path flushes explicitly and defuses the
        // guard (by zeroing `filled`) so `drop` is then a no-op.
        struct Guard<'a> {
            cord: &'a mut Cord,
            buffer: [u8; 256],
            filled: usize,
        }
        impl Drop for Guard<'_> {
            fn drop(&mut self) {
                if self.filled > 0 {
                    self.cord.append_slice(&self.buffer[..self.filled]);
                }
            }
        }

        let mut guard = Guard { cord: self, buffer: [0u8; 256], filled: 0 };
        for byte in iter {
            guard.buffer[guard.filled] = byte;
            guard.filled += 1;
            if guard.filled == guard.buffer.len() {
                guard.cord.append_slice(&guard.buffer);
                guard.filled = 0;
            }
        }
        guard.cord.append_slice(&guard.buffer[..guard.filled]);
        guard.filled = 0;
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

#[expect(
    clippy::into_iter_without_iter,
    reason = "the conventional `Cord::iter` alias was deliberately removed (untested, and ambiguous \
              with std's element-iterator convention); `chunks` is the discoverable inherent method"
)]
impl<'a> IntoIterator for &'a Cord {
    /// A contiguous chunk of the cord's bytes (the same as
    /// [`chunks`](Cord::chunks) yields), not a single byte.
    type Item = &'a [u8];
    type IntoIter = Chunks<'a>;

    /// Same as [`chunks`](Cord::chunks): iterates over the cord's
    /// contiguous byte chunks, not its individual bytes.
    #[inline]
    fn into_iter(self) -> Chunks<'a> {
        self.chunks()
    }
}
