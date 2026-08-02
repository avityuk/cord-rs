//! B-tree reps: the tree structure of a non-trivial cord.
//!
//! Data is stored at the leaf level only; non-leaf nodes contain down pointers
//! only. Allowed data edges are FLAT, EXTERNAL and SUBSTRINGs thereof. Data can
//! be added at either end of the tree only (no inserts), which yields good fill
//! ratios: all nodes except the outer "legs" are 100% full for trees built by
//! appending / prepending, and merged trees are typically well above 50%. All
//! operations are O(log n) or better and the tree never needs balancing.
//!
//! Reference counting follows the [module convention](super): functions taking
//! a tree or rep adopt a reference on it; functions returning one transfer a
//! reference back to the caller.
//!
//! Port of abseil's `cord_rep_btree.{h,cc}`.

use core::fmt;
use core::sync::atomic::{AtomicBool, Ordering};

use super::{
    BTREE, CordRep, CordRepSubstring, EXTERNAL, FLAT, RepPtr, SUBSTRING, edge_data, external, flat,
    is_data_edge, ref_rep, small_u8, unref,
};

/// Converts a node height to the signed form used by the copy / sub-tree
/// algorithms, where `-1` denotes a data edge below the leaf level.
#[inline]
fn height_to_isize(height: usize) -> isize {
    debug_assert!(height <= MAX_HEIGHT);
    #[expect(clippy::cast_possible_wrap, reason = "heights are bounded by MAX_HEIGHT")]
    let signed = height as isize;
    signed
}

/// Inverse of [`height_to_isize`]. Requires `height >= 0`.
#[inline]
fn height_from_isize(height: isize) -> usize {
    debug_assert!(height >= 0);
    #[expect(clippy::cast_sign_loss, reason = "callers only convert non-negative heights")]
    let unsigned = height as usize;
    unsigned
}

/// Maximum number of edges per node. Chosen so a node is exactly 64 bytes on
/// 64-bit platforms (16 byte header + 6 pointers).
pub(crate) const MAX_CAPACITY: usize = 6;

/// Reasonable maximum depth of a btree. With a fill ratio of at least ~50% and
/// 4 edges per node this allows for ~16 million leaf nodes; navigation stacks
/// of `MAX_DEPTH` pointers fit comfortably on the stack.
pub(crate) const MAX_DEPTH: usize = 12;

/// Maximum `height` of a node (leaf nodes have height 0).
pub(crate) const MAX_HEIGHT: usize = MAX_DEPTH - 1;

/// `true` selects the back (append) edge, `false` the front (prepend) edge.
pub(crate) const BACK: bool = true;
/// See [`BACK`].
pub(crate) const FRONT: bool = false;

/// Effect of an operation on a node that must be propagated to its parents.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Action {
    /// The operation was performed directly on the node (it was privately
    /// owned); parents only need their `length` updated.
    InPlace,
    /// The operation was performed on a copy of the (shared) node; the parent
    /// must replace its down pointer with the copy.
    Copied,
    /// The node had no capacity; a new "leg" was created that must be added
    /// to the parent (cascading up, possibly growing the tree by one level).
    Popped,
}

/// Result of an operation on a node.
#[derive(Clone, Copy)]
pub(crate) struct OpResult {
    pub(crate) tree: *mut CordRepBtree,
    pub(crate) action: Action,
}

/// Result of `copy_prefix` / `copy_suffix`: an edge at some height, where
/// `-1` identifies a plain data edge.
#[derive(Clone, Copy)]
pub(crate) struct CopyResult {
    pub(crate) edge: *mut CordRep,
    pub(crate) height: isize,
}

/// Logical position inside a node: an edge index and a size or offset.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Position {
    pub(crate) index: usize,
    pub(crate) n: usize,
}

/// Result of `extract_append_buffer`: the remaining tree (possibly null) and
/// the extracted flat (null on failure).
#[derive(Clone, Copy)]
pub(crate) struct ExtractResult {
    pub(crate) tree: *mut CordRep,
    pub(crate) extracted: *mut CordRep,
}

/// A btree node. `storage[0..3]` of the header hold `height`, `begin`, `end`.
#[repr(C)]
pub(crate) struct CordRepBtree {
    pub(crate) rep: CordRep,
    pub(crate) edges: [*mut CordRep; MAX_CAPACITY],
}

const _: () = assert!(
    core::mem::size_of::<CordRepBtree>()
        == core::mem::size_of::<CordRep>() + MAX_CAPACITY * core::mem::size_of::<usize>()
);

static EXHAUSTIVE_VALIDATION: AtomicBool = AtomicBool::new(false);

/// Forces exhaustive (recursive) validation in `assert_valid` / `is_valid`.
pub(crate) fn set_exhaustive_validation(enabled: bool) {
    EXHAUSTIVE_VALIDATION.store(enabled, Ordering::Relaxed);
}

/// Returns whether exhaustive validation is enabled.
pub(crate) fn is_exhaustive_validation_enabled() -> bool {
    EXHAUSTIVE_VALIDATION.load(Ordering::Relaxed)
}

// --- Substring helpers (consuming variants) -------------------------------

/// Creates a substring of `rep` **adopting** a reference on `rep`.
/// Requires `n != 0 && offset + n <= rep.length && (offset != 0 || n != length)`.
///
/// # Safety
///
/// `rep` must be a non-null pointer to a live flat, external, or substring
/// node, satisfying `n != 0 && offset + n <= rep.length() && (offset != 0 ||
/// n != rep.length())`. The caller donates its reference on `rep` (it is
/// consumed: unwrapped and re-referenced onto its child if `rep` is itself a
/// substring, otherwise incorporated directly as the new substring's
/// child); the returned pointer carries the freshly allocated node's own
/// reference back to the caller.
unsafe fn create_substring(mut rep: *mut CordRep, mut offset: usize, n: usize) -> *mut CordRep {
    unsafe {
        debug_assert!(n != 0);
        debug_assert!(offset + n <= rep.length());
        debug_assert!(offset != 0 || n != rep.length());
        if rep.tag() == SUBSTRING {
            let substring: *mut CordRepSubstring = rep.cast();
            offset += (*substring).start;
            rep = ref_rep((*substring).child);
            unref(substring.cast());
        }
        debug_assert!(rep.is_external() || rep.is_flat());
        Box::into_raw(Box::new(CordRepSubstring {
            rep: CordRep::new(n, SUBSTRING),
            start: offset,
            child: rep,
        }))
        .cast()
    }
}

/// Returns `rep` if `n == rep.length`, null (unreffing `rep`) if `n == 0`,
/// else a substring. Adopts a reference on `rep`.
///
/// # Safety
///
/// `rep` must be a non-null pointer to a live flat, external, or substring
/// node with `offset + n <= rep.length()`. The caller donates its reference
/// on `rep`; the returned pointer (possibly null, possibly `rep` itself)
/// carries the resulting reference back to the caller.
#[inline]
unsafe fn make_substring(rep: *mut CordRep, offset: usize, n: usize) -> *mut CordRep {
    unsafe {
        if n == rep.length() {
            return rep;
        }
        if n == 0 {
            unref(rep);
            return core::ptr::null_mut();
        }
        create_substring(rep, offset, n)
    }
}

/// `make_substring(rep, offset, rep.length - offset)`.
///
/// # Safety
///
/// Same as [`make_substring`]: `rep` must be a non-null pointer to a live
/// flat, external, or substring node with `offset <= rep.length()`. Adopts
/// and transfers a reference on `rep` exactly as `make_substring` does.
#[inline]
unsafe fn make_substring_from(rep: *mut CordRep, offset: usize) -> *mut CordRep {
    unsafe {
        if offset == 0 {
            return rep;
        }
        create_substring(rep, offset, rep.length() - offset)
    }
}

/// Resizes `edge` to `length`, adopting a reference on `edge`. If
/// `is_mutable`, flats and substrings are resized in place; otherwise a new
/// substring is returned. Requires `0 < length <= edge.length`.
///
/// # Safety
///
/// `edge` must be a non-null pointer to a live data edge (`is_data_edge`)
/// with `0 < length <= edge.length()`. `is_mutable` must be `true` only if
/// the caller has verified `edge` is not shared with any other tree (e.g.
/// `edge.refcount().is_one()`): a `true` value licenses this function to
/// mutate `edge`'s length in place, which would corrupt any other live
/// reference to `edge` if it were in fact shared. The caller donates its
/// reference on `edge`; the returned pointer (either `edge` itself or a new
/// substring) carries the resulting reference back to the caller.
unsafe fn resize_edge(edge: *mut CordRep, length: usize, is_mutable: bool) -> *mut CordRep {
    unsafe {
        debug_assert!(length > 0);
        debug_assert!(length <= edge.length());
        debug_assert!(is_data_edge(edge));
        if length >= edge.length() {
            return edge;
        }
        if is_mutable && (edge.tag() >= FLAT || edge.tag() == SUBSTRING) {
            edge.set_length(length);
            return edge;
        }
        create_substring(edge, 0, length)
    }
}

/// Removes `n` bytes from the consumed end of `s`.
#[inline]
fn consume<const IS_BACK: bool>(s: &[u8], n: usize) -> &[u8] {
    if IS_BACK { &s[n..] } else { &s[..s.len() - n] }
}

/// Copies `n` bytes from the consumed end of `s` to `dst` and returns the rest.
///
/// # Safety
///
/// `dst` must be valid for writes of `n` bytes and must not overlap `s`'s
/// backing storage (the copy uses `copy_nonoverlapping`). Requires
/// `n <= s.len()`.
#[inline]
unsafe fn consume_copy<const IS_BACK: bool>(dst: *mut u8, s: &[u8], n: usize) -> &[u8] {
    unsafe {
        if IS_BACK {
            core::ptr::copy_nonoverlapping(s.as_ptr(), dst, n);
            &s[n..]
        } else {
            let offset = s.len() - n;
            core::ptr::copy_nonoverlapping(s.as_ptr().add(offset), dst, n);
            &s[..offset]
        }
    }
}

/// Frees `substring` and, if this was its last reference, its child edge.
///
/// # Safety
///
/// `substring` must be a non-null pointer to a live `CordRepSubstring` whose
/// reference count has just reached zero: the caller is relinquishing the
/// last reference to it (not merely borrowing), so `substring` must not be
/// read through any other pointer afterwards. Its `child` must be a live
/// flat or external node.
unsafe fn delete_substring(substring: *mut CordRepSubstring) {
    unsafe {
        let rep = (*substring).child;
        if !rep.refcount().decrement() {
            if rep.tag() >= FLAT {
                flat::delete(rep);
            } else {
                debug_assert_eq!(rep.tag(), EXTERNAL);
                external::CordRepExternal::delete(rep);
            }
        }
        CordRepSubstring::delete(substring);
    }
}

/// Deletes a leaf node data edge whose reference count has reached zero.
/// Requires `is_data_edge(rep)`.
///
/// # Safety
///
/// `rep` must be a non-null pointer to a live data edge with no outstanding
/// references: the caller is relinquishing the last reference to it, exactly
/// as [`unref`](super::unref) does when its decrement reaches zero on a
/// flat, external, or substring node.
unsafe fn delete_leaf_edge(rep: *mut CordRep) {
    unsafe {
        debug_assert!(is_data_edge(rep));
        if rep.tag() >= FLAT {
            flat::delete(rep);
        } else if rep.tag() == EXTERNAL {
            external::CordRepExternal::delete(rep);
        } else {
            delete_substring(rep.cast());
        }
    }
}

// --- Node accessors ---------------------------------------------------------

/// Accessors on raw btree node pointers.
///
/// # Safety
///
/// For every method in this trait, `self` must be a non-null pointer to a
/// live, well-formed `CordRepBtree`: `begin() <= end() <= capacity()`, and
/// for every `i` in `[begin(), end())`, `edges[i]` is a non-null pointer to
/// a live rep (a data edge if `height() == 0`, else a `CordRepBtree` of
/// height `self.height() - 1`). Read-only accessors (`as_rep`, `height`,
/// `begin`, `end`, `edge`, `length`, `refcount`, and everything built purely
/// on those) require only this well-formedness and a live borrow of `self`
/// for the call's duration — they never require exclusive access, and never
/// change `self`'s own reference count (it is always borrowed, never
/// adopted or transferred). Mutating accessors (`set_begin`, `set_end`,
/// `set_edge_ptr`, `set_length`, and everything built on them, e.g.
/// `add_length`, `sub_fetch_begin`, `fetch_add_end`) additionally require
/// the caller to hold the *only* outstanding reference to `self` (or
/// otherwise have exclusive access): mutating a node that is shared with
/// another tree corrupts every other reference to it.
pub(crate) trait BtreePtr: Copy {
    /// Reinterprets `self` as a `*mut CordRep`.
    fn as_rep(self) -> *mut CordRep;
    /// Reads `self`'s `height`.
    unsafe fn height(self) -> usize;
    /// Reads `self`'s `begin` cursor.
    unsafe fn begin(self) -> usize;
    /// Reads `self`'s `end` cursor.
    unsafe fn end(self) -> usize;
    /// Sets `self`'s `begin` cursor. Requires exclusive access (see the
    /// trait-level `# Safety`) and `begin <= end()`, so every other accessor
    /// relying on the trait's well-formedness invariant remains sound.
    unsafe fn set_begin(self, begin: usize);
    /// Sets `self`'s `end` cursor. Requires exclusive access (see the
    /// trait-level `# Safety`) and `begin() <= end <= capacity()`, so every
    /// other accessor relying on the trait's well-formedness invariant
    /// remains sound.
    unsafe fn set_end(self, end: usize);
    /// Reads the edge pointer at `index`. In addition to the trait-level
    /// `# Safety`, requires `begin() <= index < end()`.
    unsafe fn edge(self, index: usize) -> *mut CordRep;
    /// Overwrites the edge slot at `index` with `edge`, *without* unreffing
    /// the previous occupant or adopting a reference on `edge` — the caller
    /// is fully responsible for both reference counts. Requires exclusive
    /// access (see the trait-level `# Safety`) and `index < capacity()`.
    unsafe fn set_edge_ptr(self, index: usize, edge: *mut CordRep);

    /// Reads `self`'s `length`.
    #[inline]
    unsafe fn length(self) -> usize {
        unsafe { self.as_rep().length() }
    }
    /// Sets `self`'s `length`. Requires exclusive access (see the
    /// trait-level `# Safety`).
    #[inline]
    unsafe fn set_length(self, length: usize) {
        unsafe {
            self.as_rep().set_length(length);
        }
    }
    /// Increases `self`'s `length` by `delta`. Requires exclusive access
    /// (see the trait-level `# Safety`).
    #[inline]
    unsafe fn add_length(self, delta: usize) {
        unsafe {
            self.set_length(self.length() + delta);
        }
    }
    /// Decreases `self`'s `length` by `delta`. Requires exclusive access
    /// (see the trait-level `# Safety`) and `delta <= length()`.
    #[inline]
    unsafe fn sub_length(self, delta: usize) {
        unsafe {
            self.set_length(self.length() - delta);
        }
    }
    /// Reads `self`'s refcount.
    #[inline]
    unsafe fn refcount<'a>(self) -> &'a super::Refcount {
        unsafe { self.as_rep().refcount() }
    }
    /// Index of the back edge.
    ///
    /// # Safety
    ///
    /// In addition to the trait-level `# Safety`, requires the node to be
    /// non-empty (`size() >= 1`), since this computes `end() - 1` with no
    /// underflow check.
    #[inline]
    unsafe fn back(self) -> usize {
        unsafe { self.end() - 1 }
    }
    /// `end() - begin()`.
    #[inline]
    unsafe fn size(self) -> usize {
        unsafe { self.end() - self.begin() }
    }
    /// Returns [`MAX_CAPACITY`]. Performs no pointer access, so this is a
    /// safe method despite the rest of the trait being unsafe.
    #[inline]
    fn capacity(self) -> usize {
        MAX_CAPACITY
    }
    /// Index of the front or back edge.
    ///
    /// # Safety
    ///
    /// In addition to the trait-level `# Safety`, requires the node to be
    /// non-empty when `IS_BACK` (see [`back`](Self::back)).
    #[inline]
    unsafe fn index<const IS_BACK: bool>(self) -> usize {
        unsafe { if IS_BACK { self.back() } else { self.begin() } }
    }
    /// The front or back edge.
    ///
    /// # Safety
    ///
    /// In addition to the trait-level `# Safety`, requires the node to be
    /// non-empty (`size() >= 1`).
    #[inline]
    unsafe fn edge_at<const IS_BACK: bool>(self) -> *mut CordRep {
        unsafe { self.edge(self.index::<IS_BACK>()) }
    }
    /// Decreases `begin` by `n` and returns the new value. Requires exclusive
    /// access (see the trait-level `# Safety`) and `n <= begin()`, so the new
    /// `begin` does not underflow and stays `<= end()`.
    #[inline]
    unsafe fn sub_fetch_begin(self, n: usize) -> usize {
        unsafe {
            let new_begin = self.begin() - n;
            self.set_begin(new_begin);
            new_begin
        }
    }
    /// Increases `end` by `n` and returns the previous value. Requires
    /// exclusive access (see the trait-level `# Safety`) and
    /// `end() + n <= capacity()`.
    #[inline]
    unsafe fn fetch_add_end(self, n: usize) -> usize {
        unsafe {
            let current = self.end();
            self.set_end(current + n);
            current
        }
    }
    /// Iterates the edges in `[begin, end)`, reading lazily.
    ///
    /// # Safety
    ///
    /// In addition to the trait-level `# Safety`, requires `self.begin() <=
    /// begin <= end <= self.end()`; the returned iterator borrows `self` and
    /// must not outlive it or be used after `self` is mutated.
    #[inline]
    unsafe fn edges_range(self, begin: usize, end: usize) -> impl Iterator<Item = *mut CordRep> {
        debug_assert!(begin <= end);
        // SAFETY: reading `self.begin()`/`self.end()` only requires `self` to
        // point at a live, well-formed node (trait-level `# Safety`), which
        // this method's own contract requires of its caller.
        debug_assert!(begin >= unsafe { self.begin() });
        debug_assert!(end <= unsafe { self.end() });
        // SAFETY: the closure is invoked only for `i` in `[begin, end)`,
        // which by this method's contract is within `[self.begin(),
        // self.end())`, so `self.edge(i)` meets `edge`'s bounds requirement.
        (begin..end).map(move |i| unsafe { self.edge(i) })
    }
    /// Iterates all edges.
    #[inline]
    unsafe fn edges(self) -> impl Iterator<Item = *mut CordRep> {
        unsafe { self.edges_range(self.begin(), self.end()) }
    }
    /// The data of the edge at `index`.
    ///
    /// # Safety
    ///
    /// In addition to the trait-level `# Safety`, requires a leaf node
    /// (`height() == 0`) and `begin() <= index < end()` (see
    /// [`edge`](Self::edge)). The returned slice borrows the edge's storage
    /// and must not outlive it or be read while the edge is being mutated in
    /// place.
    #[inline]
    unsafe fn data<'a>(self, index: usize) -> &'a [u8] {
        unsafe {
            debug_assert_eq!(self.height(), 0);
            edge_data(self.edge(index))
        }
    }

    /// Returns the index of the last edge starting on or before `offset` and
    /// the relative offset inside that edge.
    ///
    /// # Safety
    ///
    /// In addition to the trait-level `# Safety`, requires `offset <
    /// length()`: an out-of-range `offset` walks `index` past `end()`,
    /// violating `edge`'s bounds requirement.
    #[inline]
    unsafe fn index_of(self, mut offset: usize) -> Position {
        unsafe {
            debug_assert!(offset < self.length());
            let mut index = self.begin();
            while offset >= self.edge(index).length() {
                offset -= self.edge(index).length();
                index += 1;
            }
            Position { index, n: offset }
        }
    }

    /// Returns the index of the last edge starting *before* `offset` and the
    /// relative offset inside that edge.
    ///
    /// # Safety
    ///
    /// In addition to the trait-level `# Safety`, requires `0 < offset <=
    /// length()`: an out-of-range `offset` walks `index` past `end()`,
    /// violating `edge`'s bounds requirement.
    #[inline]
    unsafe fn index_before(self, mut offset: usize) -> Position {
        unsafe {
            debug_assert!(offset > 0);
            debug_assert!(offset <= self.length());
            let mut index = self.begin();
            while offset > self.edge(index).length() {
                offset -= self.edge(index).length();
                index += 1;
            }
            Position { index, n: offset }
        }
    }

    /// `index_before(front.n + offset)` optimized to start at `front.index`.
    ///
    /// # Safety
    ///
    /// In addition to the trait-level `# Safety`, requires `front.index` to
    /// be a valid edge index (`begin() <= front.index < end()`), typically
    /// obtained from a prior `index_of`/`index_before`/`index_beyond` call on
    /// this same node, and `front.n + offset` to land within the edges from
    /// `front.index` up to `end()` (equivalently, `0 < front.n + offset <=`
    /// the total length from the start of `front.index` to `length()`).
    /// Otherwise `index` walks past `end()`, violating `edge`'s bounds
    /// requirement.
    #[inline]
    unsafe fn index_before_from(self, front: Position, offset: usize) -> Position {
        unsafe {
            let mut index = front.index;
            let mut offset = offset + front.n;
            while offset > self.edge(index).length() {
                offset -= self.edge(index).length();
                index += 1;
            }
            Position { index, n: offset }
        }
    }

    /// Returns the index of the edge ending at (or on) length `n` and the
    /// number of bytes inside that edge up to `n`.
    ///
    /// # Safety
    ///
    /// In addition to the trait-level `# Safety`, requires `n <= length()`
    /// and the node to be non-empty (see [`back`](Self::back)): an
    /// out-of-range `n` walks `index` below `begin()`, underflowing.
    #[inline]
    unsafe fn index_of_length(self, n: usize) -> Position {
        unsafe {
            debug_assert!(n <= self.length());
            let mut index = self.back();
            let mut strip = self.length() - n;
            while strip >= self.edge(index).length() {
                strip -= self.edge(index).length();
                index -= 1;
            }
            Position { index, n: self.edge(index).length() - strip }
        }
    }

    /// Returns the index of the edge directly beyond the edge containing
    /// `offset` and the distance of that edge from `offset`.
    ///
    /// # Safety
    ///
    /// In addition to the trait-level `# Safety`, requires `offset <=
    /// length()`; an `offset` greater than `length()` walks `index` past
    /// `end()`, violating `edge`'s bounds requirement. (`offset == length()`
    /// is allowed and yields `index == end()`, one past the last edge.)
    #[inline]
    unsafe fn index_beyond(self, offset: usize) -> Position {
        unsafe {
            let mut off = 0;
            let mut index = self.begin();
            while offset > off {
                off += self.edge(index).length();
                index += 1;
            }
            Position { index, n: off - offset }
        }
    }
}

impl BtreePtr for *mut CordRepBtree {
    #[inline]
    fn as_rep(self) -> *mut CordRep {
        // A `CordRepBtree` starts with its `rep: CordRep` field (`#[repr(C)]`),
        // so this cast is a plain pointer reinterpretation with no deref.
        self.cast()
    }
    #[inline]
    unsafe fn height(self) -> usize {
        unsafe {
            // SAFETY: per the trait's `# Safety`, `self` is a live, well-formed
            // node, so dereferencing it and reading its `storage` header is sound.
            (*self).rep.storage[0] as usize
        }
    }
    #[inline]
    unsafe fn begin(self) -> usize {
        unsafe {
            // SAFETY: see `height` above.
            (*self).rep.storage[1] as usize
        }
    }
    #[inline]
    unsafe fn end(self) -> usize {
        unsafe {
            // SAFETY: see `height` above.
            (*self).rep.storage[2] as usize
        }
    }
    #[inline]
    unsafe fn set_begin(self, begin: usize) {
        unsafe {
            // SAFETY: `set_begin`'s contract requires `self` to be exclusively
            // owned and `begin <= end()`, so overwriting the `begin` cursor keeps
            // the node well-formed for subsequent accessors.
            (*self).rep.storage[1] = small_u8(begin);
        }
    }
    #[inline]
    unsafe fn set_end(self, end: usize) {
        unsafe {
            // SAFETY: `set_end`'s contract requires `self` to be exclusively
            // owned and `begin() <= end <= capacity()`, so overwriting the `end`
            // cursor keeps the node well-formed for subsequent accessors.
            (*self).rep.storage[2] = small_u8(end);
        }
    }
    #[inline]
    unsafe fn edge(self, index: usize) -> *mut CordRep {
        unsafe {
            debug_assert!(index >= self.begin());
            debug_assert!(index < self.end());
            // SAFETY: `self` is live and well-formed (trait contract), and
            // `edge`'s own contract requires `index` in `[begin(), end())`
            // (checked above in debug builds), which is within the fixed-size
            // `edges` array's bounds (`end() <= capacity() == MAX_CAPACITY ==
            // edges.len()`).
            (*self).edges[index]
        }
    }
    #[inline]
    unsafe fn set_edge_ptr(self, index: usize, edge: *mut CordRep) {
        unsafe {
            // Unlike `edge`'s `[begin(), end())` window, this is intentionally
            // just the array bound: many callers write a slot before bumping
            // `begin`/`end` to include it (or into a freshly `alloc`ed node
            // whose cursors are still `0..0`), which is well-formed usage that
            // a tighter `[begin(), end())` assert would wrongly reject.
            debug_assert!(index < self.capacity());
            // SAFETY: `set_edge_ptr`'s contract requires exclusive access and
            // `index < capacity() == edges.len()` (checked above in debug
            // builds); this overwrites the slot without touching either
            // pointer's reference count, as documented.
            (*self).edges[index] = edge;
        }
    }
}

/// Casts a rep known to be a btree node.
///
/// # Safety
///
/// `rep` must be a non-null pointer to a live rep with tag `BTREE`; it is
/// borrowed, not consumed.
#[inline]
pub(crate) unsafe fn as_btree(rep: *mut CordRep) -> *mut CordRepBtree {
    unsafe {
        debug_assert!(rep.is_btree());
        rep.cast()
    }
}

// --- Stack operations -------------------------------------------------------

/// Builds a left-most or right-most "leg" from the root to the leaf level and
/// propagates node changes back up.
struct StackOperations<const IS_BACK: bool> {
    /// Depth at which nodes become shared: 0 if the root is shared, 1 if the
    /// second node is shared, ..., `> depth` if no node is shared.
    share_depth: usize,
    stack: [*mut CordRepBtree; MAX_DEPTH],
}

impl<const IS_BACK: bool> StackOperations<IS_BACK> {
    #[inline]
    fn new() -> Self {
        Self { share_depth: 0, stack: [core::ptr::null_mut(); MAX_DEPTH] }
    }

    /// True if the node at `depth` and all of its parents are privately owned.
    #[inline]
    fn owned(&self, depth: usize) -> bool {
        depth < self.share_depth
    }

    /// Builds a `depth` levels deep stack starting at `tree`, recording where
    /// nodes become shared. Returns the node at `depth`.
    ///
    /// # Safety
    ///
    /// `tree` must be a non-null pointer to a live, well-formed btree node
    /// with `depth <= tree.height()`. `tree` is borrowed, not consumed: this
    /// only walks `IS_BACK`-edge down-pointers and records them in
    /// `self.stack`, it does not affect any refcount.
    #[inline]
    unsafe fn build_stack(&mut self, mut tree: *mut CordRepBtree, depth: usize) -> *mut CordRepBtree {
        unsafe {
            // SAFETY: `current_depth < depth <= tree.height()` is maintained by
            // both loops below, so `edge_at` always reads a down-pointer of a
            // non-leaf node, itself a live, well-formed btree node.
            debug_assert!(depth <= tree.height());
            let mut current_depth = 0;
            while current_depth < depth && tree.refcount().is_one() {
                self.stack[current_depth] = tree;
                current_depth += 1;
                tree = tree.edge_at::<IS_BACK>().cast();
            }
            self.share_depth = current_depth + usize::from(tree.refcount().is_one());
            while current_depth < depth {
                self.stack[current_depth] = tree;
                current_depth += 1;
                tree = tree.edge_at::<IS_BACK>().cast();
            }
            tree
        }
    }

    /// Builds a stack with the invariant that all nodes are privately owned.
    ///
    /// # Safety
    ///
    /// `tree` must be a non-null pointer to a live, well-formed, uniquely
    /// owned btree node (refcount one, and so is every node on its
    /// `IS_BACK`-edge path down to `height` levels) with `height <=
    /// MAX_HEIGHT`. `tree` is borrowed, not consumed.
    #[inline]
    unsafe fn build_owned_stack(&mut self, mut tree: *mut CordRepBtree, height: usize) {
        unsafe {
            // SAFETY: `depth < height`, so `edge_at` reads a down-pointer of a
            // non-leaf node at each step, itself a live, well-formed, uniquely
            // owned btree node per the loop's own `debug_assert` below.
            debug_assert!(height <= MAX_HEIGHT);
            let mut depth = 0;
            while depth < height {
                debug_assert!(tree.refcount().is_one());
                self.stack[depth] = tree;
                depth += 1;
                tree = tree.edge_at::<IS_BACK>().cast();
            }
            debug_assert!(tree.refcount().is_one());
            self.share_depth = depth + 1;
        }
    }

    /// Processes the final top level result action for the tree.
    ///
    /// # Safety
    ///
    /// `tree` and `result.tree` must be non-null pointers to live,
    /// well-formed btree nodes, consistent with `result.action` as produced
    /// by the same stack's earlier operations: `Copied` means `tree`'s old
    /// reference is being replaced by `result.tree` (so `tree` is unreffed
    /// here), `Popped` means `result.tree` is a new sibling leg to pair with
    /// `tree`, `InPlace` means `result.tree` already *is* `tree`.
    #[inline]
    unsafe fn finalize(tree: *mut CordRepBtree, result: OpResult) -> *mut CordRepBtree {
        unsafe {
            match result.action {
                Action::Popped => {
                    let tree = if IS_BACK {
                        CordRepBtree::new_pair(tree, result.tree)
                    } else {
                        CordRepBtree::new_pair(result.tree, tree)
                    };
                    if tree.height() > MAX_HEIGHT {
                        core::hint::cold_path();
                        let tree = CordRepBtree::rebuild(tree);
                        assert!(tree.height() <= MAX_HEIGHT, "cord-rs: max btree height exceeded");
                        return tree;
                    }
                    tree
                }
                Action::Copied => {
                    unref(tree.as_rep());
                    result.tree
                }
                Action::InPlace => result.tree,
            }
        }
    }

    /// Propagates `result` up into all nodes of the stack starting at `depth`.
    /// `length` is the extra length added at the lowest level. If `PROPAGATE`
    /// is set, copied node values are updated into the stack for iterative
    /// processing on the same stack.
    ///
    /// # Safety
    ///
    /// `self.stack[0..depth]` must hold live, well-formed btree nodes as
    /// built by `build_stack`/`build_owned_stack`, with `self.share_depth`
    /// accurately marking which are privately owned. `tree` and `result`
    /// must be consistent with the deepest of those nodes, per `finalize`'s
    /// contract (this function is `finalize` generalized to unwind through
    /// the whole stack instead of a single level).
    #[inline]
    unsafe fn unwind<const PROPAGATE: bool>(
        &mut self,
        tree: *mut CordRepBtree,
        mut depth: usize,
        length: usize,
        mut result: OpResult,
    ) -> *mut CordRepBtree {
        unsafe {
            if depth != 0 {
                loop {
                    depth -= 1;
                    let node = self.stack[depth];
                    let owned = depth < self.share_depth;
                    match result.action {
                        Action::Popped => {
                            debug_assert!(!PROPAGATE);
                            result =
                                CordRepBtree::add_edge::<IS_BACK>(node, owned, result.tree.as_rep(), length);
                        }
                        Action::Copied => {
                            result =
                                CordRepBtree::set_edge::<IS_BACK>(node, owned, result.tree.as_rep(), length);
                            if PROPAGATE {
                                self.stack[depth] = result.tree;
                            }
                        }
                        Action::InPlace => {
                            let mut node = node;
                            node.add_length(length);
                            while depth > 0 {
                                depth -= 1;
                                node = self.stack[depth];
                                node.add_length(length);
                            }
                            return node;
                        }
                    }
                    if depth == 0 {
                        break;
                    }
                }
            }
            Self::finalize(tree, result)
        }
    }

    /// `unwind::<true>`: propagates `result`, updating `self.stack` in place.
    ///
    /// # Safety
    ///
    /// Same contract as [`unwind`](Self::unwind).
    #[inline]
    unsafe fn propagate(
        &mut self,
        tree: *mut CordRepBtree,
        depth: usize,
        length: usize,
        result: OpResult,
    ) -> *mut CordRepBtree {
        unsafe { self.unwind::<true>(tree, depth, length, result) }
    }
}

// --- Construction / destruction --------------------------------------------

impl CordRepBtree {
    /// Allocates a fresh, uninitialized-edges node.
    ///
    /// Ownership obligation on the result (not a precondition of calling):
    /// the caller should eventually fill `edges[begin..end]` with live rep
    /// pointers before the node is treated as a well-formed tree; not doing
    /// so only leaks memory.
    ///
    /// # Panics
    ///
    /// Panics if `height > MAX_DEPTH` (it would otherwise be silently
    /// truncated by `small_u8`, corrupting the node's declared height).
    /// `MAX_DEPTH`, not `MAX_HEIGHT`, is the bound here: `finalize`'s
    /// `new_pair` of two `MAX_HEIGHT` children is a legitimate, if
    /// momentary, `MAX_DEPTH`-high tree that gets folded back down to
    /// `MAX_HEIGHT` by [`rebuild`](Self::rebuild) right after.
    #[inline]
    fn alloc(length: usize, height: usize, begin: usize, end: usize) -> *mut CordRepBtree {
        assert!(height <= MAX_DEPTH, "cord-rs: height {height} exceeds MAX_DEPTH");
        let mut rep = CordRep::new(length, BTREE);
        rep.storage = [small_u8(height), small_u8(begin), small_u8(end)];
        Box::into_raw(Box::new(CordRepBtree { rep, edges: [core::ptr::null_mut(); MAX_CAPACITY] }))
    }

    /// Creates a new empty node at `height`.
    ///
    /// # Panics
    ///
    /// Panics if `height` is too large for [`alloc`](Self::alloc) to accept
    /// (see there); callers are expected to keep `height <= MAX_HEIGHT` for
    /// the result to be a well-formed tree.
    #[inline]
    pub(crate) fn new_node(height: usize) -> *mut CordRepBtree {
        Self::alloc(0, height, 0, 0)
    }

    /// Creates a new node containing `rep`, at height `rep.height + 1` for a
    /// btree `rep` and 0 otherwise.
    ///
    /// # Safety
    ///
    /// `rep` must be a non-null pointer to a live rep; the caller donates its
    /// reference, which becomes the new node's sole edge.
    #[inline]
    pub(crate) unsafe fn new_with(rep: *mut CordRep) -> *mut CordRepBtree {
        unsafe {
            let height = if rep.is_btree() { as_btree(rep).height() + 1 } else { 0 };
            let tree = Self::alloc(rep.length(), height, 0, 1);
            tree.set_edge_ptr(0, rep);
            tree
        }
    }

    /// Creates a new node containing `front` and `back`, which must be of
    /// equal height.
    ///
    /// # Safety
    ///
    /// `front` and `back` must be non-null pointers to live btree nodes of
    /// equal height (`<= MAX_HEIGHT - 1`, so the new parent's height stays
    /// `<= MAX_HEIGHT`); the caller donates both references, which become
    /// the new node's two edges.
    #[inline]
    pub(crate) unsafe fn new_pair(front: *mut CordRepBtree, back: *mut CordRepBtree) -> *mut CordRepBtree {
        unsafe {
            debug_assert_eq!(front.height(), back.height());
            let tree = Self::alloc(front.length() + back.length(), front.height() + 1, 0, 2);
            tree.set_edge_ptr(0, front.as_rep());
            tree.set_edge_ptr(1, back.as_rep());
            tree
        }
    }

    /// Creates a btree from `rep`, adopting a reference. Returns `rep` if it
    /// already is a btree, else a new leaf containing the data edge `rep`.
    ///
    /// # Safety
    ///
    /// `rep` must be a non-null pointer to a live rep that is either a data
    /// edge or a btree node; the caller donates its reference, which
    /// transfers to the returned btree (as `rep` itself, or as its sole
    /// edge).
    #[inline]
    pub(crate) unsafe fn create(rep: *mut CordRep) -> *mut CordRepBtree {
        unsafe {
            if is_data_edge(rep) {
                return Self::new_with(rep);
            }
            assert!(rep.is_btree(), "cord-rs: unexpected node type {} in CordRepBtree::create", rep.tag());
            rep.cast()
        }
    }

    /// Frees the node itself (its `edges` are not touched: the caller must
    /// have already released or relocated every live edge).
    ///
    /// # Safety
    ///
    /// `tree` must be a non-null pointer previously returned by `alloc` (via
    /// `new_node`/`new_with`/`new_pair`/a copy helper), not read through any
    /// other pointer afterwards, and must not have any edge in
    /// `[begin, end)` that still needs unreffing.
    #[inline]
    pub(crate) unsafe fn delete(tree: *mut CordRepBtree) {
        unsafe {
            drop(Box::from_raw(tree));
        }
    }

    /// Unrefs all edges in `[begin, end)` of `tree` (assumed likely one).
    ///
    /// # Safety
    ///
    /// `tree` must be a non-null pointer to a live, well-formed btree node,
    /// with `[begin, end)` a sub-range of `tree`'s currently populated edges
    /// (`tree.begin() <= begin <= end <= tree.end()`); each unreffed edge
    /// must not be read again afterwards.
    #[inline]
    pub(crate) unsafe fn unref_edges(tree: *mut CordRepBtree, begin: usize, end: usize) {
        unsafe {
            for edge in tree.edges_range(begin, end) {
                if !edge.refcount().decrement() {
                    core::hint::cold_path();
                    super::destroy(edge);
                }
            }
        }
    }

    /// Destroys `tree`, whose reference count reached zero.
    ///
    /// # Safety
    ///
    /// `tree` must be a non-null pointer to a live, well-formed btree node
    /// whose reference count has just reached zero: the caller is
    /// relinquishing the last reference (not merely borrowing), so `tree`
    /// (and everything under it that isn't independently referenced
    /// elsewhere) must not be read through any other pointer afterwards.
    pub(crate) unsafe fn destroy(tree: *mut CordRepBtree) {
        unsafe {
            match tree.height() {
                0 => {
                    for edge in tree.edges() {
                        if !edge.refcount().decrement() {
                            delete_leaf_edge(edge);
                        }
                    }
                    Self::delete(tree);
                }
                1 => Self::destroy_tree::<1>(tree),
                _ => Self::destroy_tree::<2>(tree),
            }
        }
    }

    /// `destroy` for `tree.height() >= 1`. `SIZE == 1` selects leaf
    /// grandchildren (data edges), `SIZE != 1` selects btree grandchildren
    /// (recursing via `destroy`); this only walks one level further than
    /// `destroy` itself and relies on recursion (through `destroy`) or the
    /// data-edge case for anything deeper, so it does not need to know the
    /// exact height beyond "at least 1".
    ///
    /// # Safety
    ///
    /// Same contract as [`destroy`](Self::destroy): `tree` must be live,
    /// well-formed, with its reference count just reached zero, and
    /// `SIZE == 1` iff `tree.height() == 1`.
    unsafe fn destroy_tree<const SIZE: usize>(tree: *mut CordRepBtree) {
        unsafe {
            for node in tree.edges() {
                if node.refcount().decrement() {
                    continue;
                }
                let node = as_btree(node);
                for edge in node.edges() {
                    if edge.refcount().decrement() {
                        continue;
                    }
                    if SIZE == 1 {
                        delete_leaf_edge(edge);
                    } else {
                        Self::destroy(as_btree(edge));
                    }
                }
                Self::delete(node);
            }
            Self::delete(tree);
        }
    }

    // --- Copies -------------------------------------------------------------

    /// Raw copy of this node with `new_length`, copying all properties but
    /// without adding references to the edges.
    ///
    /// # Safety
    ///
    /// `this` must be a non-null pointer to a live, well-formed btree node;
    /// it is borrowed, not consumed. The result is a new, uniquely owned
    /// node whose `edges` are a bitwise copy of `this`'s (i.e. it does not
    /// yet own its own references on them — callers must `ref_rep` the
    /// edges they keep before the copy and `this` can be treated as
    /// independent).
    #[inline]
    unsafe fn copy_raw(this: *mut CordRepBtree, new_length: usize) -> *mut CordRepBtree {
        unsafe {
            let tree = Self::alloc(new_length, 0, 0, 0);
            // `tag`, `storage` (the height/begin/end triple) and `edges` are
            // plain data: copy them field by field. This is equivalent to
            // the previous single `copy_nonoverlapping` from `tag` onwards
            // but does not depend on there being no padding between fields.
            (*tree).rep.tag = (*this).rep.tag;
            (*tree).rep.storage = (*this).rep.storage;
            (*tree).edges = (*this).edges;
            tree
        }
    }

    /// Full copy of this node, adding a reference on all edges.
    ///
    /// # Safety
    ///
    /// `this` must be a non-null pointer to a live, well-formed btree node;
    /// it is borrowed, not consumed.
    #[inline]
    unsafe fn copy(this: *mut CordRepBtree) -> *mut CordRepBtree {
        unsafe {
            let tree = Self::copy_raw(this, this.length());
            for rep in this.edges() {
                ref_rep(rep);
            }
            tree
        }
    }

    /// Copy of the edges starting at `begin`, with `new_length`.
    ///
    /// # Safety
    ///
    /// `this` must be a non-null pointer to a live, well-formed btree node
    /// with `this.begin() <= begin <= this.end()`; it is borrowed, not
    /// consumed.
    #[inline]
    unsafe fn copy_to_end_from(
        this: *mut CordRepBtree,
        begin: usize,
        new_length: usize,
    ) -> *mut CordRepBtree {
        unsafe {
            debug_assert!(begin >= this.begin());
            debug_assert!(begin <= this.end());
            let tree = Self::copy_raw(this, new_length);
            tree.set_begin(begin);
            for edge in tree.edges() {
                ref_rep(edge);
            }
            tree
        }
    }

    /// Copy of the edges up to `end`, with `new_length`.
    ///
    /// # Safety
    ///
    /// `this` must be a non-null pointer to a live, well-formed btree node
    /// with `this.begin() <= end <= this.capacity()`; it is borrowed, not
    /// consumed.
    #[inline]
    unsafe fn copy_begin_to(this: *mut CordRepBtree, end: usize, new_length: usize) -> *mut CordRepBtree {
        unsafe {
            debug_assert!(end <= this.capacity());
            debug_assert!(end >= this.begin());
            let tree = Self::copy_raw(this, new_length);
            tree.set_end(end);
            for edge in tree.edges() {
                ref_rep(edge);
            }
            tree
        }
    }

    /// Returns a tree containing edges `[begin, end)` with `new_length`,
    /// consuming `tree` (in place if privately owned).
    ///
    /// # Safety
    ///
    /// `tree` must be a non-null pointer to a live, well-formed btree node
    /// with `end <= tree.end()`; the caller donates its reference on `tree`,
    /// which is consumed by this call (either reused in place or unreffed
    /// after being copied).
    unsafe fn consume_begin_to(tree: *mut CordRepBtree, end: usize, new_length: usize) -> *mut CordRepBtree {
        unsafe {
            debug_assert!(end <= tree.end());
            if tree.refcount().is_one() {
                Self::unref_edges(tree, end, tree.end());
                tree.set_end(end);
                tree.set_length(new_length);
                tree
            } else {
                let old = tree;
                let tree = Self::copy_begin_to(tree, end, new_length);
                unref(old.as_rep());
                tree
            }
        }
    }

    /// Extracts the front edge from `tree`, consuming `tree`.
    ///
    /// # Safety
    ///
    /// `tree` must be a non-null pointer to a live, well-formed, non-empty
    /// btree node; the caller donates its reference on `tree`, which is
    /// consumed by this call. The returned edge carries its own, independent
    /// reference (either `tree`'s former reference to it, reused in place,
    /// or a freshly acquired one).
    unsafe fn extract_front(tree: *mut CordRepBtree) -> *mut CordRep {
        unsafe {
            let front = tree.edge(tree.begin());
            if tree.refcount().is_one() {
                Self::unref_edges(tree, tree.begin() + 1, tree.end());
                Self::delete(tree);
            } else {
                ref_rep(front);
                unref(tree.as_rep());
            }
            front
        }
    }

    // --- Edge manipulation --------------------------------------------------

    /// Aligns existing edges to start at index 0.
    ///
    /// # Safety
    ///
    /// `this` must be a non-null pointer to a live, well-formed, uniquely
    /// owned btree node (edges are shifted in place).
    #[inline]
    unsafe fn align_begin(this: *mut CordRepBtree) {
        unsafe {
            let delta = this.begin();
            if delta != 0 {
                core::hint::cold_path();
                let new_end = this.end() - delta;
                this.set_begin(0);
                this.set_end(new_end);
                for i in 0..new_end {
                    // SAFETY: `i + delta` ranges over the *old* `[delta,
                    // old_end)` window (this loop's own `[0, new_end)` shifted
                    // by `delta`), which was `this`'s live `[begin(), end())`
                    // before the cursor update just above; now that `begin`/
                    // `end` have moved, that range sits above the *new*
                    // `end()`, so `edge`'s tighter `[begin(), end())` bound
                    // would wrongly reject it even though the slot is still
                    // live and in bounds (`< MAX_CAPACITY`).
                    this.set_edge_ptr(i, (*this).edges[i + delta]);
                }
            }
        }
    }

    /// Aligns existing edges to end at `capacity`.
    ///
    /// # Safety
    ///
    /// Same contract as [`align_begin`](Self::align_begin): `this` must be a
    /// non-null pointer to a live, well-formed, uniquely owned btree node.
    #[inline]
    unsafe fn align_end(this: *mut CordRepBtree) {
        unsafe {
            let delta = this.capacity() - this.end();
            if delta != 0 {
                let new_begin = this.begin() + delta;
                let new_end = this.end() + delta;
                this.set_begin(new_begin);
                this.set_end(new_end);
                let mut i = new_end;
                while i > new_begin {
                    i -= 1;
                    // SAFETY: `i - delta` ranges over the *old* `[old_begin,
                    // old_end)` window (this loop's `[new_begin, new_end)`
                    // shifted down by `delta`), which was `this`'s live
                    // `[begin(), end())` before the cursor update just above;
                    // now that `begin`/`end` have moved, that range sits below
                    // the *new* `begin()`, so `edge`'s tighter `[begin(),
                    // end())` bound would wrongly reject it even though the
                    // slot is still live and in bounds (`< MAX_CAPACITY`).
                    this.set_edge_ptr(i, (*this).edges[i - delta]);
                }
            }
        }
    }

    /// Adds `rep` at the back or front. Requires spare capacity.
    ///
    /// # Safety
    ///
    /// `this` must be a non-null pointer to a live, well-formed, uniquely
    /// owned btree node with `this.size() < MAX_CAPACITY`. `rep` must be a
    /// non-null pointer to a live rep; the caller donates its reference,
    /// which becomes one of `this`'s edges.
    #[inline]
    unsafe fn add<const IS_BACK: bool>(this: *mut CordRepBtree, rep: *mut CordRep) {
        unsafe {
            if IS_BACK {
                Self::align_begin(this);
                let idx = this.fetch_add_end(1);
                this.set_edge_ptr(idx, rep);
            } else {
                Self::align_end(this);
                let idx = this.sub_fetch_begin(1);
                this.set_edge_ptr(idx, rep);
            }
        }
    }

    /// Adds all edges of `src` at the back or front. Requires spare capacity.
    ///
    /// # Safety
    ///
    /// `this` must be a non-null pointer to a live, well-formed, uniquely
    /// owned btree node with `this.size() + src.size() <= MAX_CAPACITY`.
    /// `src` must be a non-null pointer to a live, well-formed btree node,
    /// borrowed (not consumed): the copied edge pointers become additional
    /// references shared with `src`, so the caller remains responsible for
    /// `src`'s own reference on each (typically by not separately unreffing
    /// them, e.g. when `src` itself is about to be `delete`d without
    /// unreffing its edges).
    #[inline]
    unsafe fn add_edges_from<const IS_BACK: bool>(this: *mut CordRepBtree, src: *mut CordRepBtree) {
        unsafe {
            let (sb, se) = (src.begin(), src.end());
            if IS_BACK {
                Self::align_begin(this);
                let mut new_end = this.end();
                for i in sb..se {
                    this.set_edge_ptr(new_end, src.edge(i));
                    new_end += 1;
                }
                this.set_end(new_end);
            } else {
                Self::align_end(this);
                let new_begin = this.begin() - (se - sb);
                this.set_begin(new_begin);
                for (dst, i) in (new_begin..).zip(sb..se) {
                    this.set_edge_ptr(dst, src.edge(i));
                }
            }
        }
    }

    /// Adds `edge` to `this` if possible: in place if `owned`, on a copy if
    /// shared, or as a new popped leg if at capacity.
    ///
    /// # Safety
    ///
    /// `this` must be a non-null pointer to a live, well-formed btree node;
    /// `owned` must accurately state whether `this` is privately owned
    /// (refcount one) and therefore safe to mutate in place — `add` is only
    /// reached (via `to_op_result`) when `owned` is true. `edge` must be a
    /// non-null pointer to a live rep; the caller donates its reference,
    /// which is incorporated into the result (as a new edge of `this` or of
    /// a fresh popped node).
    #[inline]
    unsafe fn add_edge<const IS_BACK: bool>(
        this: *mut CordRepBtree,
        owned: bool,
        edge: *mut CordRep,
        delta: usize,
    ) -> OpResult {
        unsafe {
            if this.size() >= MAX_CAPACITY {
                return OpResult { tree: Self::new_with(edge), action: Action::Popped };
            }
            let result = Self::to_op_result(this, owned);
            Self::add::<IS_BACK>(result.tree, edge);
            result.tree.add_length(delta);
            result
        }
    }

    /// Replaces the front or back edge with `edge`, in place if `owned` or on
    /// a copy otherwise. Adopts a reference on `edge`.
    ///
    /// # Safety
    ///
    /// `this` must be a non-null pointer to a live, well-formed, non-empty
    /// btree node; `owned` must accurately state whether `this` is privately
    /// owned (refcount one). `edge` must be a non-null pointer to a live
    /// rep; the caller donates its reference, which replaces the old front
    /// or back edge (whose reference is released via `unref`, in the
    /// `owned` case, or simply not copied into the new node otherwise).
    unsafe fn set_edge<const IS_BACK: bool>(
        this: *mut CordRepBtree,
        owned: bool,
        edge: *mut CordRep,
        delta: usize,
    ) -> OpResult {
        unsafe {
            let idx = this.index::<IS_BACK>();
            let result = if owned {
                unref(this.edge(idx));
                OpResult { tree: this, action: Action::InPlace }
            } else {
                // Copy containing all unchanged edges: [begin, back) or
                // [begin + 1, end) depending on the edge type.
                let result = OpResult { tree: Self::copy_raw(this, this.length()), action: Action::Copied };
                let shift = usize::from(!IS_BACK);
                for r in this.edges_range(this.begin() + shift, this.back() + shift) {
                    ref_rep(r);
                }
                result
            };
            result.tree.set_edge_ptr(idx, edge);
            result.tree.add_length(delta);
            result
        }
    }

    /// Wraps `this` as an `OpResult` ready for in-place mutation: `InPlace`
    /// (reusing `this`) if `owned`, `Copied` (a fresh, uniquely owned copy)
    /// otherwise.
    ///
    /// # Safety
    ///
    /// `this` must be a non-null pointer to a live, well-formed btree node,
    /// borrowed (not consumed): the `InPlace` case reuses it directly, the
    /// `Copied` case leaves `this` untouched and returns an independent
    /// copy. `owned` must accurately state whether `this` is privately
    /// owned (refcount one).
    #[inline]
    unsafe fn to_op_result(this: *mut CordRepBtree, owned: bool) -> OpResult {
        unsafe {
            if owned {
                OpResult { tree: this, action: Action::InPlace }
            } else {
                OpResult { tree: Self::copy(this), action: Action::Copied }
            }
        }
    }

    // --- Append / prepend ---------------------------------------------------

    /// Adds `rep` as a new edge (not merged into existing data), at the back
    /// or front of `tree`.
    ///
    /// # Safety
    ///
    /// `tree` must be a non-null pointer to a live, well-formed btree node;
    /// the caller donates its reference on `tree`, consumed by this call.
    /// `rep` must be a non-null pointer to a live rep; the caller donates
    /// its reference, incorporated into the returned tree.
    unsafe fn add_cord_rep<const IS_BACK: bool>(
        tree: *mut CordRepBtree,
        rep: *mut CordRep,
    ) -> *mut CordRepBtree {
        unsafe {
            let depth = tree.height();
            let length = rep.length();
            let mut ops = StackOperations::<IS_BACK>::new();
            let leaf = ops.build_stack(tree, depth);
            let result = Self::add_edge::<IS_BACK>(leaf, ops.owned(depth), rep, length);
            ops.unwind::<false>(tree, depth, length, result)
        }
    }

    /// Creates a new leaf containing as much of `data` as possible.
    ///
    /// # Safety
    ///
    /// None beyond `height <= MAX_HEIGHT` for the `new_node(0)` call this
    /// makes internally (always satisfied: a fresh leaf has height 0).
    unsafe fn new_leaf<const IS_BACK: bool>(mut data: &[u8], extra: usize) -> *mut CordRepBtree {
        unsafe {
            let leaf = Self::new_node(0);
            let mut length = 0;
            let cap = leaf.capacity();
            if IS_BACK {
                let mut end = 0;
                while !data.is_empty() && end != cap {
                    let f = flat::new(data.len() + extra);
                    let n = data.len().min(flat::capacity(f));
                    f.set_length(n);
                    length += n;
                    leaf.set_edge_ptr(end, f);
                    end += 1;
                    data = consume_copy::<IS_BACK>(flat::data(f), data, n);
                }
                leaf.set_length(length);
                leaf.set_end(end);
            } else {
                let mut begin = cap;
                leaf.set_end(cap);
                while !data.is_empty() && begin != 0 {
                    let f = flat::new(data.len() + extra);
                    let n = data.len().min(flat::capacity(f));
                    f.set_length(n);
                    length += n;
                    begin -= 1;
                    leaf.set_edge_ptr(begin, f);
                    data = consume_copy::<IS_BACK>(flat::data(f), data, n);
                }
                leaf.set_length(length);
                leaf.set_begin(begin);
            }
            leaf
        }
    }

    /// Adds data to this leaf until all data is consumed or the node is full.
    /// Returns the remaining data. Requires a non-full leaf and non-empty data.
    ///
    /// # Safety
    ///
    /// `this` must be a non-null pointer to a live, well-formed, uniquely
    /// owned leaf (height 0) btree node with `this.size() < this.capacity()`
    /// and `!data.is_empty()`.
    unsafe fn add_data_to_leaf<'a, const IS_BACK: bool>(
        this: *mut CordRepBtree,
        mut data: &'a [u8],
        extra: usize,
    ) -> &'a [u8] {
        unsafe {
            debug_assert!(!data.is_empty());
            debug_assert!(this.size() < this.capacity());
            if IS_BACK {
                Self::align_begin(this);
                let cap = this.capacity();
                loop {
                    let f = flat::new(data.len() + extra);
                    let n = data.len().min(flat::capacity(f));
                    f.set_length(n);
                    let idx = this.fetch_add_end(1);
                    this.set_edge_ptr(idx, f);
                    data = consume_copy::<IS_BACK>(flat::data(f), data, n);
                    if data.is_empty() || this.end() == cap {
                        break;
                    }
                }
            } else {
                Self::align_end(this);
                loop {
                    let f = flat::new(data.len() + extra);
                    let n = data.len().min(flat::capacity(f));
                    f.set_length(n);
                    let idx = this.sub_fetch_begin(1);
                    this.set_edge_ptr(idx, f);
                    data = consume_copy::<IS_BACK>(flat::data(f), data, n);
                    if data.is_empty() || this.begin() == 0 {
                        break;
                    }
                }
            }
            data
        }
    }

    /// Adds `data` to `tree`, appending/prepending into spare leaf capacity
    /// where possible and creating new leaves for the rest.
    ///
    /// # Safety
    ///
    /// `tree` must be a non-null pointer to a live, well-formed btree node;
    /// the caller donates its reference, consumed by this call and
    /// transferred to the returned tree.
    unsafe fn add_data<const IS_BACK: bool>(
        mut tree: *mut CordRepBtree,
        mut data: &[u8],
        extra: usize,
    ) -> *mut CordRepBtree {
        unsafe {
            if data.is_empty() {
                return tree;
            }
            let original_data_size = data.len();
            let mut depth = tree.height();
            let mut ops = StackOperations::<IS_BACK>::new();
            let leaf = ops.build_stack(tree, depth);

            // If there is capacity in the last leaf, append as much as possible
            // into it.
            if leaf.size() < leaf.capacity() {
                let result = Self::to_op_result(leaf, ops.owned(depth));
                data = Self::add_data_to_leaf::<IS_BACK>(result.tree, data, extra);
                if data.is_empty() {
                    result.tree.add_length(original_data_size);
                    return ops.unwind::<false>(tree, depth, original_data_size, result);
                }
                // We added some data but not all. Propagate the added length to
                // the root and rebuild the stack with any copied nodes. From here
                // on the leg towards the leaf is privately owned.
                let delta = original_data_size - data.len();
                debug_assert!(delta > 0);
                result.tree.add_length(delta);
                tree = ops.propagate(tree, depth, delta, result);
                ops.share_depth = depth + 1;
            }

            // Put the remaining data into new leaf node(s) and merge each into the
            // first level towards the root that has capacity.
            loop {
                let result =
                    OpResult { tree: Self::new_leaf::<IS_BACK>(data, extra), action: Action::Popped };
                let added = result.tree.length();
                if added == data.len() {
                    return ops.unwind::<false>(tree, depth, added, result);
                }
                data = consume::<IS_BACK>(data, added);
                tree = ops.unwind::<false>(tree, depth, added, result);
                depth = tree.height();
                ops.build_owned_stack(tree, depth);
            }
        }
    }

    /// Merges `src` into `dst` before (`FRONT`) or after (`BACK`) it.
    /// Requires `dst.height >= src.height`.
    ///
    /// # Safety
    ///
    /// `dst` and `src` must be non-null pointers to live, well-formed btree
    /// nodes with `dst.height() >= src.height()`; the caller donates its
    /// references on both, consumed by this call and transferred to the
    /// returned tree.
    unsafe fn merge<const IS_BACK: bool>(
        dst: *mut CordRepBtree,
        src: *mut CordRepBtree,
    ) -> *mut CordRepBtree {
        unsafe {
            debug_assert!(dst.height() >= src.height());
            // Capture the source length as we may consume / destroy `src`.
            let length = src.length();
            // Merge `src` at its corresponding height in `dst`.
            let depth = dst.height() - src.height();
            let mut ops = StackOperations::<IS_BACK>::new();
            let merge_node = ops.build_stack(dst, depth);

            // If there is enough space in `merge_node` for all edges from `src`,
            // add them (copying the node if shared). Otherwise `unwind` /
            // `finalize` merge `src` into the first level with capacity, or create
            // a new top level node.
            let result = if merge_node.size() + src.size() <= MAX_CAPACITY {
                let result = Self::to_op_result(merge_node, ops.owned(depth));
                Self::add_edges_from::<IS_BACK>(result.tree, src);
                result.tree.add_length(src.length());
                if src.refcount().is_one() {
                    Self::delete(src);
                } else {
                    for edge in src.edges() {
                        ref_rep(edge);
                    }
                    unref(src.as_rep());
                }
                result
            } else {
                OpResult { tree: src, action: Action::Popped }
            };

            if depth != 0 {
                return ops.unwind::<false>(dst, depth, length, result);
            }
            StackOperations::<IS_BACK>::finalize(dst, result)
        }
    }

    /// Returns a tree containing `left` followed by `right`.
    ///
    /// # Safety
    ///
    /// `left` and `right` must be non-null pointers to live, well-formed
    /// btree nodes; the caller donates its references on both, consumed by
    /// this call.
    pub(crate) unsafe fn merge_trees(left: *mut CordRepBtree, right: *mut CordRepBtree) -> *mut CordRepBtree {
        unsafe {
            if left.height() >= right.height() {
                Self::merge::<BACK>(left, right)
            } else {
                Self::merge::<FRONT>(right, left)
            }
        }
    }

    /// Appends `rep` (a data edge or a btree) to `tree`.
    ///
    /// # Safety
    ///
    /// `tree` must be a non-null pointer to a live, well-formed btree node.
    /// `rep` must be a non-null pointer to a live rep that is either a data
    /// edge or a btree node. The caller donates its references on both,
    /// consumed by this call and transferred to the returned tree.
    #[inline]
    pub(crate) unsafe fn append(tree: *mut CordRepBtree, rep: *mut CordRep) -> *mut CordRepBtree {
        unsafe {
            if is_data_edge(rep) {
                return Self::add_cord_rep::<BACK>(tree, rep);
            }
            assert!(rep.is_btree(), "cord-rs: unexpected node type {} in CordRepBtree::append", rep.tag());
            Self::merge_trees(tree, rep.cast())
        }
    }

    /// Prepends `rep` (a data edge or a btree) to `tree`.
    ///
    /// # Safety
    ///
    /// Same contract as [`append`](Self::append).
    #[inline]
    pub(crate) unsafe fn prepend(tree: *mut CordRepBtree, rep: *mut CordRep) -> *mut CordRepBtree {
        unsafe {
            if is_data_edge(rep) {
                return Self::add_cord_rep::<FRONT>(tree, rep);
            }
            assert!(rep.is_btree(), "cord-rs: unexpected node type {} in CordRepBtree::prepend", rep.tag());
            Self::merge_trees(rep.cast(), tree)
        }
    }

    /// Appends `data` to `tree`. `extra` is a hint for additional capacity to
    /// allocate in any newly created flat. Data of any size is supported.
    ///
    /// # Safety
    ///
    /// `tree` must be a non-null pointer to a live, well-formed btree node;
    /// the caller donates its reference, consumed by this call and
    /// transferred to the returned tree.
    #[inline]
    pub(crate) unsafe fn append_data(
        tree: *mut CordRepBtree,
        data: &[u8],
        extra: usize,
    ) -> *mut CordRepBtree {
        unsafe { Self::add_data::<BACK>(tree, data, extra) }
    }

    /// Prepends `data` to `tree`. See [`append_data`](Self::append_data).
    ///
    /// # Safety
    ///
    /// Same contract as [`append_data`](Self::append_data).
    #[inline]
    pub(crate) unsafe fn prepend_data(
        tree: *mut CordRepBtree,
        data: &[u8],
        extra: usize,
    ) -> *mut CordRepBtree {
        unsafe { Self::add_data::<FRONT>(tree, data, extra) }
    }

    // --- Sub trees ----------------------------------------------------------

    /// Partial copy of the tree containing all data starting at `offset`.
    /// Requires `offset < length`. Does not consume `this`.
    ///
    /// # Safety
    ///
    /// `this` must be a non-null pointer to a live, well-formed btree node
    /// with `offset < this.length()`; it is borrowed, not consumed. The
    /// result's `edge` carries a fresh, independent reference (built from
    /// `ref_rep`/`copy_to_end_from`/`make_substring_from` along the path).
    unsafe fn copy_suffix(this: *mut CordRepBtree, mut offset: usize) -> CopyResult {
        unsafe {
            debug_assert!(offset < this.length());

            // As long as `offset` starts inside the last edge we can drop the
            // current depth: if it references the last data edge there is only a
            // single path from the root to that edge.
            let mut height = height_to_isize(this.height());
            let mut node = this;
            let mut len = node.length() - offset;
            let mut back = node.edge_at::<BACK>();
            while back.length() >= len {
                offset = back.length() - len;
                height -= 1;
                if height < 0 {
                    return CopyResult { edge: make_substring_from(ref_rep(back), offset), height };
                }
                node = as_btree(back);
                back = node.edge_at::<BACK>();
            }
            if offset == 0 {
                return CopyResult { edge: ref_rep(node.as_rep()), height };
            }

            // Offset does not point into the last edge, so we span at least two
            // edges. `index_beyond` gives the edge beyond the offset if the offset
            // is not a clean start of an edge.
            let mut pos = node.index_beyond(offset);
            let mut sub = Self::copy_to_end_from(node, pos.index, len);
            let result = CopyResult { edge: sub.as_rep(), height };

            // `pos.n` is non-zero if the offset is not an exact start of an edge:
            // it holds the trailing bytes of the preceding edge. Iteratively adjust
            // that edge until we have a perfect start.
            while pos.n != 0 {
                debug_assert!(pos.index >= 1);
                let begin = pos.index - 1;
                sub.set_begin(begin);
                let edge = node.edge(begin);

                len = pos.n;
                offset = edge.length() - len;

                height -= 1;
                if height < 0 {
                    sub.set_edge_ptr(begin, make_substring(ref_rep(edge), offset, len));
                    return result;
                }

                node = as_btree(edge);
                pos = node.index_beyond(offset);

                let nsub = Self::copy_to_end_from(node, pos.index, len);
                sub.set_edge_ptr(begin, nsub.as_rep());
                sub = nsub;
            }
            sub.set_begin(pos.index);
            result
        }
    }

    /// Partial copy of the tree containing the first `n` bytes. The result
    /// may be less high than the tree, or a single data edge (height -1) if
    /// `allow_folding`. Requires `0 < n <= length`. Does not consume `this`.
    ///
    /// # Safety
    ///
    /// `this` must be a non-null pointer to a live, well-formed btree node
    /// with `0 < n <= this.length()`; it is borrowed, not consumed. The
    /// result's `edge` carries a fresh, independent reference.
    unsafe fn copy_prefix(this: *mut CordRepBtree, mut n: usize, allow_folding: bool) -> CopyResult {
        unsafe {
            debug_assert!(n > 0);
            debug_assert!(n <= this.length());

            // As long as `n` does not exceed the length of the first edge we can
            // drop the current depth.
            let mut height = height_to_isize(this.height());
            let mut node = this;
            let mut front = node.edge_at::<FRONT>();
            if allow_folding {
                while front.length() >= n {
                    height -= 1;
                    if height < 0 {
                        return CopyResult { edge: make_substring(ref_rep(front), 0, n), height: -1 };
                    }
                    node = as_btree(front);
                    front = node.edge_at::<FRONT>();
                }
            }
            if node.length() == n {
                return CopyResult { edge: ref_rep(node.as_rep()), height };
            }

            // `n` spans at least two nodes: find the end point of the span.
            let mut pos = node.index_of(n);

            // Partial copy up to `pos.index` with a defined length of `n`; any
            // partial last edge is added below.
            let mut sub = Self::copy_begin_to(node, pos.index, n);
            let result = CopyResult { edge: sub.as_rep(), height };

            // `pos.n` is the offset inside the edge at `index_of(n)`. While it is
            // not zero we don't have a clean cut and need a partial copy of that
            // last edge.
            while pos.n != 0 {
                let mut end = pos.index;
                n = pos.n;

                let edge = node.edge(pos.index);
                height -= 1;
                if height < 0 {
                    sub.set_edge_ptr(end, make_substring(ref_rep(edge), 0, n));
                    end += 1;
                    sub.set_end(end);
                    Self::assert_valid(as_btree(result.edge), true);
                    return result;
                }

                node = as_btree(edge);
                pos = node.index_of(n);
                let nsub = Self::copy_begin_to(node, pos.index, n);
                sub.set_edge_ptr(end, nsub.as_rep());
                end += 1;
                sub.set_end(end);
                sub = nsub;
            }
            sub.set_end(pos.index);
            Self::assert_valid(as_btree(result.edge), true);
            result
        }
    }

    /// Returns a new tree containing `n` bytes starting at `offset`, sharing
    /// nodes and edges with this tree where possible. Requires
    /// `offset + n <= length`. Returns null if `n == 0`. Does not consume
    /// `this`.
    ///
    /// # Safety
    ///
    /// `this` must be a non-null pointer to a live, well-formed btree node
    /// with `n <= this.length()` and `offset <= this.length() - n`; it is
    /// borrowed, not consumed. The result carries a fresh, independent
    /// reference.
    pub(crate) unsafe fn sub_tree(this: *mut CordRepBtree, offset: usize, n: usize) -> *mut CordRep {
        unsafe {
            debug_assert!(n <= this.length());
            debug_assert!(offset <= this.length() - n);
            if n == 0 {
                core::hint::cold_path();
                return core::ptr::null_mut();
            }

            let mut node = this;
            let mut height = height_to_isize(node.height());
            let mut front = node.index_of(offset);
            let mut left = node.edge(front.index);
            while front.n + n <= left.length() {
                height -= 1;
                if height < 0 {
                    return make_substring(ref_rep(left), front.n, n);
                }
                node = as_btree(left);
                front = node.index_of(front.n);
                left = node.edge(front.index);
            }

            let back = node.index_before_from(front, n);
            let right = node.edge(back.index);
            debug_assert!(back.index > front.index);

            // Get partial suffix and prefix entries.
            let (mut prefix, mut suffix);
            if height > 0 {
                // Copy prefix and suffix of the boundary nodes.
                prefix = Self::copy_suffix(as_btree(left), front.n);
                suffix = Self::copy_prefix(as_btree(right), back.n, true);

                // If there is an edge between the prefix and suffix edges the tree
                // must remain at its previous (full) height. Otherwise the tree
                // must be as high as the highest of the (collapsed) prefix /
                // suffix edges.
                if front.index + 1 == back.index {
                    height = prefix.height.max(suffix.height) + 1;
                }

                // Raise prefix and suffix to the new tree height.
                for _ in (prefix.height + 1)..height {
                    prefix.edge = Self::new_with(prefix.edge).as_rep();
                }
                for _ in (suffix.height + 1)..height {
                    suffix.edge = Self::new_with(suffix.edge).as_rep();
                }
            } else {
                // Leaf node: simply take substrings for prefix and suffix.
                prefix = CopyResult { edge: make_substring_from(ref_rep(left), front.n), height: -1 };
                suffix = CopyResult { edge: make_substring(ref_rep(right), 0, back.n), height: -1 };
            }

            // Compose the resulting tree.
            let sub = Self::new_node(height_from_isize(height));
            let mut end = 0;
            sub.set_edge_ptr(end, prefix.edge);
            end += 1;
            for r in node.edges_range(front.index + 1, back.index) {
                sub.set_edge_ptr(end, ref_rep(r));
                end += 1;
            }
            sub.set_edge_ptr(end, suffix.edge);
            end += 1;
            sub.set_end(end);
            sub.set_length(n);
            Self::assert_valid(sub, true).as_rep()
        }
    }

    /// Removes `n` trailing bytes from `tree`, consuming it, and returns the
    /// resulting tree or data edge (in place where possible). Returns `tree`
    /// if `n == 0` and null if `n == length`.
    ///
    /// # Safety
    ///
    /// `tree` must be a non-null pointer to a live, well-formed btree node
    /// with `n <= tree.length()`; the caller donates its reference, consumed
    /// by this call and transferred to the returned rep (reused in place
    /// where possible).
    pub(crate) unsafe fn remove_suffix(mut tree: *mut CordRepBtree, n: usize) -> *mut CordRep {
        unsafe {
            debug_assert!(n <= tree.length());
            let len = tree.length();
            if n == 0 {
                core::hint::cold_path();
                return tree.as_rep();
            }
            if n >= len {
                core::hint::cold_path();
                unref(tree.as_rep());
                return core::ptr::null_mut();
            }

            let mut length = len - n;
            let mut height = height_to_isize(tree.height());
            let mut is_mutable = tree.refcount().is_one();

            // Extract all top nodes which are reduced to size = 1.
            let mut pos = tree.index_of_length(length);
            while pos.index == tree.begin() {
                let edge = Self::extract_front(tree);
                is_mutable &= edge.refcount().is_one();
                if height == 0 {
                    return resize_edge(edge, length, is_mutable);
                }
                height -= 1;
                tree = as_btree(edge);
                pos = tree.index_of_length(length);
            }

            // Traverse down the tree:
            // - Crop the top node to the last remaining edge, adjusting length.
            // - Set the length of down edges to the partial length in that edge.
            // - Repeat until the last edge is included in full.
            // - At the data edge level, resize and return the last data edge.
            tree = Self::consume_begin_to(tree, pos.index + 1, length);
            let top = tree;
            let mut edge = tree.edge(pos.index);
            length = pos.n;
            while length != edge.length() {
                // `consume_begin_to` guarantees `tree` is a privately owned copy.
                debug_assert!(tree.refcount().is_one());
                let edge_is_mutable = edge.refcount().is_one();

                if height == 0 {
                    tree.set_edge_ptr(pos.index, resize_edge(edge, length, edge_is_mutable));
                    return Self::assert_valid(top, true).as_rep();
                }
                height -= 1;

                if !edge_is_mutable {
                    // We can't remove suffixes in place down this edge: replace it
                    // with a prefix copy instead.
                    tree.set_edge_ptr(pos.index, Self::copy_prefix(as_btree(edge), length, false).edge);
                    unref(edge);
                    return Self::assert_valid(top, true).as_rep();
                }

                // Move down one level, rinse, repeat.
                tree = as_btree(edge);
                pos = tree.index_of_length(length);
                tree = Self::consume_begin_to(as_btree(edge), pos.index + 1, length);
                edge = tree.edge(pos.index);
                length = pos.n;
            }

            Self::assert_valid(top, true).as_rep()
        }
    }

    // --- Queries ------------------------------------------------------------

    /// Returns the data if this tree holds a single data edge.
    ///
    /// # Safety
    ///
    /// `this` must be a non-null pointer to a live, well-formed btree node,
    /// borrowed for the returned slice's lifetime `'a` (the caller is
    /// responsible for `this` actually outliving `'a`, and for not mutating
    /// it in place while the slice is alive).
    #[inline]
    pub(crate) unsafe fn as_flat<'a>(this: *mut CordRepBtree) -> Option<&'a [u8]> {
        unsafe { if this.height() == 0 && this.size() == 1 { Some(this.data(this.begin())) } else { None } }
    }

    /// Returns the `n` bytes starting at `offset` if they are contained in a
    /// single data edge. Requires `offset + n <= length`.
    ///
    /// # Safety
    ///
    /// `this` must be a non-null pointer to a live, well-formed btree node
    /// with `n <= this.length()` and `offset <= this.length() - n`,
    /// borrowed for the returned slice's lifetime `'a` exactly as
    /// [`as_flat`](Self::as_flat).
    pub(crate) unsafe fn as_flat_range<'a>(
        this: *mut CordRepBtree,
        mut offset: usize,
        n: usize,
    ) -> Option<&'a [u8]> {
        unsafe {
            debug_assert!(n <= this.length());
            debug_assert!(offset <= this.length() - n);
            if n == 0 {
                return None;
            }
            let mut height = this.height();
            let mut node = this;
            loop {
                let front = node.index_of(offset);
                let edge = node.edge(front.index);
                if edge.length() < front.n + n {
                    return None;
                }
                if height == 0 {
                    return Some(&edge_data(edge)[front.n..front.n + n]);
                }
                height -= 1;
                offset = front.n;
                node = as_btree(edge);
            }
        }
    }

    /// Returns the byte at `offset`. Requires `offset < length`.
    ///
    /// # Safety
    ///
    /// `this` must be a non-null pointer to a live, well-formed btree node
    /// with `offset < this.length()`; it is borrowed.
    pub(crate) unsafe fn get_byte(this: *mut CordRepBtree, mut offset: usize) -> u8 {
        unsafe {
            debug_assert!(offset < this.length());
            let mut node = this;
            let mut height = node.height();
            loop {
                let front = node.index_of(offset);
                if height == 0 {
                    return node.data(front.index)[front.n];
                }
                height -= 1;
                offset = front.n;
                node = as_btree(node.edge(front.index));
            }
        }
    }

    /// Returns a pointer to and length of up to `size` bytes of spare capacity
    /// in the last flat of this tree, increasing the lengths of the flat and
    /// all nodes on the path by that amount, iff:
    /// - none of the nodes down to the flat are shared,
    /// - the last data edge is a non-shared flat with available capacity.
    ///
    /// The caller must immediately initialize the returned bytes. Requires
    /// `this.refcount().is_one()`.
    ///
    /// # Safety
    ///
    /// `this` must be a non-null pointer to a live, well-formed btree node
    /// with `this.refcount().is_one()`; it is borrowed, not consumed (any
    /// node whose length is grown remains owned by `this`'s tree). On
    /// success the caller must fill the returned region (already accounted
    /// for in the tree's length) before any other access to `this`.
    pub(crate) unsafe fn get_append_buffer(this: *mut CordRepBtree, size: usize) -> Option<(*mut u8, usize)> {
        unsafe {
            debug_assert!(this.refcount().is_one());
            let depth = this.height();
            let mut node = this;
            let mut stack = [core::ptr::null_mut::<CordRepBtree>(); MAX_DEPTH];
            for slot in stack.iter_mut().take(depth) {
                node = as_btree(node.edge_at::<BACK>());
                if !node.refcount().is_one() {
                    return None;
                }
                *slot = node;
            }
            // Must be a privately owned, mutable flat with capacity.
            let edge = node.edge_at::<BACK>();
            if !edge.refcount().is_one() || edge.tag() < FLAT {
                return None;
            }
            let avail = flat::capacity(edge) - edge.length();
            if avail == 0 {
                return None;
            }
            let delta = size.min(avail);
            let span = (flat::data(edge).add(edge.length()), delta);
            edge.set_length(edge.length() + delta);
            this.add_length(delta);
            for &n in stack.iter().take(depth) {
                n.add_length(delta);
            }
            Some(span)
        }
    }

    /// Extracts the right-most flat from `tree` iff the tree and all nodes
    /// down to it are unshared, it is an unshared flat, and it has at least
    /// `extra_capacity` bytes available. Returns `{tree, null}` otherwise.
    /// On success the flat is removed from the tree, which may collapse to a
    /// single data edge or to null.
    ///
    /// # Safety
    ///
    /// `tree` must be a non-null pointer to a live, well-formed btree node;
    /// the caller donates its reference, consumed by this call. On failure
    /// the result's `tree` field carries that same reference back unchanged
    /// (`extracted` is null); on success `extracted` carries a reference to
    /// the removed flat and `tree` carries a reference to whatever remains
    /// (possibly null, if the whole tree was consumed).
    pub(crate) unsafe fn extract_append_buffer(
        mut tree: *mut CordRepBtree,
        extra_capacity: usize,
    ) -> ExtractResult {
        unsafe {
            let mut depth = 0;
            let mut stack = [core::ptr::null_mut::<CordRepBtree>(); MAX_DEPTH];
            let mut result = ExtractResult { tree: tree.as_rep(), extracted: core::ptr::null_mut() };

            // Dive down the right side of the tree, making sure no edges are shared.
            while tree.height() > 0 {
                if !tree.refcount().is_one() {
                    return result;
                }
                stack[depth] = tree;
                depth += 1;
                tree = as_btree(tree.edge_at::<BACK>());
            }
            if !tree.refcount().is_one() {
                return result;
            }

            // Validate we ended on a non shared flat with enough capacity.
            let mut rep = tree.edge_at::<BACK>();
            if !(rep.is_flat() && rep.refcount().is_one()) {
                return result;
            }
            let length = rep.length();
            let avail = flat::capacity(rep) - length;
            if extra_capacity > avail {
                return result;
            }
            result.extracted = rep;

            // Cascading delete all nodes that become empty.
            while tree.size() == 1 {
                Self::delete(tree);
                if depth == 0 {
                    // We consumed the entire tree.
                    result.tree = core::ptr::null_mut();
                    return result;
                }
                depth -= 1;
                tree = stack[depth];
            }

            // Remove the edge or cascaded up parent node and adjust lengths.
            tree.set_end(tree.end() - 1);
            tree.sub_length(length);
            while depth > 0 {
                depth -= 1;
                tree = stack[depth];
                tree.sub_length(length);
            }

            // Remove unnecessary top nodes with size = 1, possibly all the way
            // down to the leaf, in which case the remaining last edge is returned.
            while tree.size() == 1 {
                let height = tree.height();
                rep = tree.edge_at::<BACK>();
                Self::delete(tree);
                if height == 0 {
                    result.tree = rep;
                    return result;
                }
                tree = as_btree(rep);
            }

            result.tree = tree.as_rep();
            result
        }
    }

    // --- Rebuild ------------------------------------------------------------

    /// Recursively re-inserts every data edge of `tree` into the (already
    /// partially built) rebalanced tree(s) in `stack`. `consume` requests
    /// that `tree`'s own reference be released once its edges are dealt
    /// with (`unref`d, or `delete`d in place if uniquely owned).
    ///
    /// # Safety
    ///
    /// `tree` must be a non-null pointer to a live, well-formed btree node;
    /// if `consume` the caller donates its reference, which is released by
    /// this call, otherwise `tree` is only borrowed. `stack` must be an
    /// in-progress rebuild stack as constructed by `rebuild`, indexed by
    /// height, with each populated entry a live, uniquely owned btree node.
    unsafe fn rebuild_into(
        stack: &mut [*mut CordRepBtree; MAX_DEPTH + 1],
        tree: *mut CordRepBtree,
        consume: bool,
    ) {
        unsafe {
            let owned = consume && tree.refcount().is_one();
            if tree.height() == 0 {
                for mut edge in tree.edges() {
                    if !owned {
                        edge = ref_rep(edge);
                    }
                    let mut height = 0;
                    let length = edge.length();
                    let mut node = stack[0];
                    let mut result = Self::add_edge::<BACK>(node, true, edge, length);
                    while result.action == Action::Popped {
                        stack[height] = result.tree;
                        height += 1;
                        assert!(height < MAX_DEPTH, "cord-rs: CordRepBtree::rebuild exceeded max depth");
                        if stack[height].is_null() {
                            result.action = Action::InPlace;
                            stack[height] = Self::new_pair(node, result.tree);
                        } else {
                            node = stack[height];
                            result = Self::add_edge::<BACK>(node, true, result.tree.as_rep(), length);
                        }
                    }
                    height += 1;
                    while height < MAX_DEPTH && !stack[height].is_null() {
                        stack[height].add_length(length);
                        height += 1;
                    }
                }
            } else {
                for rep in tree.edges() {
                    Self::rebuild_into(stack, as_btree(rep), owned);
                }
            }
            if consume {
                if owned {
                    Self::delete(tree);
                } else {
                    unref(tree.as_rep());
                }
            }
        }
    }

    /// Creates a fully balanced tree from all data edges of `tree`, consuming
    /// it. Invoked automatically when a tree exceeds the maximum height.
    ///
    /// # Safety
    ///
    /// `tree` must be a non-null pointer to a live, well-formed btree node;
    /// the caller donates its reference, consumed by this call and
    /// transferred to the returned tree.
    pub(crate) unsafe fn rebuild(tree: *mut CordRepBtree) -> *mut CordRepBtree {
        unsafe {
            let mut node = Self::new_node(0);
            let mut stack = [core::ptr::null_mut::<CordRepBtree>(); MAX_DEPTH + 1];
            stack[0] = node;
            Self::rebuild_into(&mut stack, tree, true);
            for &parent in &stack {
                if parent.is_null() {
                    return node;
                }
                node = parent;
            }
            unreachable!("cord-rs: rebuild stack not null terminated")
        }
    }

    // --- Diagnostics --------------------------------------------------------

    /// Checks that `tree` is valid and internally consistent. If `shallow`,
    /// only the node and the cumulative length / types / heights of its
    /// direct children are checked (unless exhaustive validation is enabled).
    ///
    /// # Safety
    ///
    /// `tree` must be non-null and point to a live `CordRepBtree` — that is,
    /// the very thing this function otherwise checks; unlike the rest of the
    /// module it must tolerate a *structurally* invalid tree (that's the
    /// point of validation) but the pointer and its immediate header must
    /// still be dereferenceable.
    pub(crate) unsafe fn check_valid(tree: *const CordRepBtree, shallow: bool) -> Result<(), String> {
        unsafe {
            macro_rules! check {
                ($cond:expr) => {
                    if !$cond {
                        return Err(format!("CordRepBtree::check_valid() FAILED: {}", stringify!($cond)));
                    }
                };
            }
            let tree = tree.cast_mut();
            check!(!tree.is_null());
            check!(tree.as_rep().is_btree());
            check!(tree.height() <= MAX_HEIGHT);
            check!(tree.begin() < tree.capacity());
            check!(tree.end() <= tree.capacity());
            check!(tree.begin() <= tree.end());
            let mut child_length = 0usize;
            for edge in tree.edges() {
                check!(!edge.is_null());
                if tree.height() > 0 {
                    check!(edge.is_btree());
                    check!(as_btree(edge).height() == tree.height() - 1);
                } else {
                    check!(is_data_edge(edge));
                }
                child_length += edge.length();
            }
            if child_length != tree.length() {
                return Err(format!(
                    "CordRepBtree::check_valid() FAILED: child_length != tree.length ({} vs {})",
                    child_length,
                    tree.length()
                ));
            }
            if (!shallow || is_exhaustive_validation_enabled()) && tree.height() > 0 {
                for edge in tree.edges() {
                    Self::check_valid(as_btree(edge), shallow)?;
                }
            }
            Ok(())
        }
    }

    /// Returns `true` if `tree` is valid. See [`check_valid`](Self::check_valid).
    ///
    /// # Safety
    ///
    /// Same contract as [`check_valid`](Self::check_valid).
    pub(crate) unsafe fn is_valid(tree: *const CordRepBtree, shallow: bool) -> bool {
        unsafe { Self::check_valid(tree, shallow).is_ok() }
    }

    /// Asserts (in debug builds) that `tree` is valid and returns it.
    ///
    /// # Safety
    ///
    /// Same contract as [`check_valid`](Self::check_valid); `tree` is
    /// borrowed and returned unchanged.
    #[inline]
    pub(crate) unsafe fn assert_valid(tree: *mut CordRepBtree, shallow: bool) -> *mut CordRepBtree {
        unsafe {
            if cfg!(debug_assertions)
                && let Err(msg) = Self::check_valid(tree, shallow)
            {
                let mut dump = String::new();
                let _ = Self::dump(tree.as_rep(), "CordRepBtree validation failed:", false, &mut dump);
                panic!("{msg}\n{dump}");
            }
            tree
        }
    }

    /// Dumps the structure of `rep` (a btree, substring, flat or external) to
    /// `out`. Intended for debugging and testing only.
    ///
    /// # Safety
    ///
    /// `rep` must be null or a pointer to a live, well-formed rep tree; it
    /// is borrowed, not consumed.
    pub(crate) unsafe fn dump(
        rep: *const CordRep,
        label: &str,
        include_contents: bool,
        out: &mut dyn fmt::Write,
    ) -> fmt::Result {
        unsafe {
            writeln!(out, "===================================")?;
            if !label.is_empty() {
                writeln!(out, "{label}")?;
                writeln!(out, "-----------------------------------")?;
            }
            if rep.is_null() { writeln!(out, "NULL") } else { Self::dump_all(rep, include_contents, out, 0) }
        }
    }

    /// `dump`'s recursive worker: `rep` here is always non-null (the null
    /// case is handled by `dump` itself before recursing).
    ///
    /// # Safety
    ///
    /// `rep` must be a non-null pointer to a live, well-formed rep tree; it
    /// is borrowed, not consumed.
    unsafe fn dump_all(
        rep: *const CordRep,
        include_contents: bool,
        out: &mut dyn fmt::Write,
        depth: usize,
    ) -> fmt::Result {
        unsafe {
            debug_assert!(depth <= MAX_DEPTH + 2);
            let rep = rep.cast_mut();
            let sharing = if rep.refcount().is_one() {
                "Private".to_string()
            } else {
                format!("Shared({})", rep.refcount().get())
            };
            let maybe_dump_data = |out: &mut dyn fmt::Write, r: *mut CordRep| -> fmt::Result {
                if include_contents {
                    const MAX_DATA_LENGTH: usize = 60;
                    let data = edge_data(r);
                    let shown = &data[..data.len().min(MAX_DATA_LENGTH)];
                    write!(
                        out,
                        ", data = \"{}\"{}",
                        shown.escape_ascii(),
                        if data.len() > MAX_DATA_LENGTH { "..." } else { "" }
                    )?;
                }
                writeln!(out)
            };
            write!(out, "{:indent$}{sharing} ({rep:p}) ", "", indent = depth * 2)?;
            if rep.is_btree() {
                let node = as_btree(rep);
                let label =
                    if node.height() != 0 { format!("Node({})", node.height()) } else { "Leaf".to_string() };
                writeln!(
                    out,
                    "{label}, len = {}, begin = {}, end = {}",
                    node.length(),
                    node.begin(),
                    node.end()
                )?;
                for edge in node.edges() {
                    Self::dump_all(edge, include_contents, out, depth + 1)?;
                }
            } else if rep.tag() == SUBSTRING {
                let substring: *mut CordRepSubstring = rep.cast();
                write!(out, "Substring, len = {}, start = {}", rep.length(), (*substring).start)?;
                maybe_dump_data(out, rep)?;
                Self::dump_all((*substring).child, include_contents, out, depth + 1)?;
            } else if rep.tag() >= FLAT {
                write!(out, "Flat, len = {}, cap = {}", rep.length(), flat::capacity(rep))?;
                maybe_dump_data(out, rep)?;
            } else if rep.tag() == EXTERNAL {
                write!(out, "Extn, len = {}", rep.length())?;
                maybe_dump_data(out, rep)?;
            }
            Ok(())
        }
    }
}
