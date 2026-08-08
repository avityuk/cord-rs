//! Navigation over the data edges of a btree.
//!
//! A navigator keeps a stack of `(node, index)` pairs from the root to the
//! current leaf edge and can move forward / backward, seek to an offset, skip
//! bytes, and read a sub range into a new tree. Port of abseil's
//! `cord_rep_btree_navigator.{h,cc}`.

use core::ptr::NonNull;

use super::btree::{BACK, BtreePtr, CordRepBtree, FRONT, MAX_DEPTH, MAX_HEIGHT, as_btree};
use super::{CordRep, RepPtr, is_data_edge, ref_rep, small_u8, substring_impl, unref};

// `Option<NonNull<T>>` is niche-optimized to the same size as `*mut T`
// (`None` is the all-zero / null bit pattern), so wrapping the genuine null
// sentinels below in `Option<NonNull<_>>` costs nothing over the raw
// pointers they replace.
const _: () =
    assert!(core::mem::size_of::<Option<NonNull<CordRep>>>() == core::mem::size_of::<*mut CordRep>());

/// A data edge and an offset inside it, as returned by `seek` / `skip`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct NavPosition {
    /// The edge, or `None` if the requested position is beyond the tree.
    pub(crate) edge: Option<NonNull<CordRep>>,
    pub(crate) offset: usize,
}

/// Result of [`CordRepBtreeNavigator::read`].
#[derive(Clone, Copy, Debug)]
pub(crate) struct ReadResult {
    /// The tree holding the read data (identical to `sub_tree(...)` on the
    /// navigated tree), or `None` if the read exceeded the tree.
    pub(crate) tree: Option<NonNull<CordRep>>,
    /// Bytes used from the last navigated-to edge.
    pub(crate) n: usize,
}

/// Returns a substring of the data edge `rep` (adopting no reference on
/// `rep`; the result holds its own). Null if `n == 0`; `ref(rep)` if `n ==
/// rep.length`.
///
/// # Safety
///
/// `rep` must be non-null, point to a live data edge (`is_data_edge(rep)`),
/// and not be concurrently mutated for the duration of the call. Callers
/// must uphold `n <= rep.length()`, `offset < rep.length()` and `offset + n
/// <= rep.length()`. This function does not adopt `rep`'s reference — the
/// caller keeps its own and remains responsible for eventually `unref`ing
/// it; the returned pointer (if non-null) carries a fresh reference that
/// the caller now owns.
unsafe fn substring(rep: *mut CordRep, offset: usize, n: usize) -> *mut CordRep {
    unsafe {
        debug_assert!(n <= rep.length());
        debug_assert!(offset < rep.length());
        debug_assert!(offset <= rep.length() - n);
        debug_assert!(is_data_edge(rep));
        if n == 0 {
            return core::ptr::null_mut();
        }
        if n == rep.length() {
            return ref_rep(rep);
        }
        substring_impl::<false>(rep, offset, n)
    }
}

/// Like [`substring`], but from `offset` to the end of `rep`.
///
/// # Safety
///
/// Same contract as [`substring`], with `n = rep.length() - offset`: `rep`
/// must be non-null, a live data edge, and `offset <= rep.length()` (so the
/// subtraction does not wrap).
#[inline]
unsafe fn substring_from(rep: *mut CordRep, offset: usize) -> *mut CordRep {
    unsafe { substring(rep, offset, rep.length() - offset) }
}

/// See the [module documentation](self).
#[derive(Clone, Copy)]
pub(crate) struct CordRepBtreeNavigator {
    /// Height of the current tree, or `None` if empty.
    height: Option<usize>,
    /// Path to the current data edge: `node[0].edge(index[0])`. Undefined
    /// until initialized (`height >= 0`).
    index: [u8; MAX_DEPTH],
    /// Root-to-leaf path of the current tree. Only `node[0..=height]` are
    /// meaningful (guaranteed live and well-formed while the navigator is
    /// non-empty); entries above that watermark are left dangling
    /// (`NonNull::dangling()`) and must never be read. Plain `NonNull`
    /// (rather than `Option`) so reading a path entry within the watermark
    /// never carries an extra branch in the hot navigation loops below.
    node: [NonNull<CordRepBtree>; MAX_DEPTH],
}

impl Default for CordRepBtreeNavigator {
    fn default() -> Self {
        Self::new()
    }
}

impl CordRepBtreeNavigator {
    /// An empty navigator.
    pub(crate) const fn new() -> Self {
        Self { height: None, index: [0; MAX_DEPTH], node: [NonNull::dangling(); MAX_DEPTH] }
    }

    /// Returns `true` if not empty.
    #[inline]
    pub(crate) fn is_some(&self) -> bool {
        self.height.is_some()
    }

    /// Height of the current tree. Requires a non-empty navigator.
    #[inline]
    fn tree_height(&self) -> usize {
        debug_assert!(self.height.is_some());
        self.height.unwrap_or(0)
    }

    /// The tree being navigated, or null if empty.
    #[inline]
    pub(crate) fn btree(&self) -> *mut CordRepBtree {
        self.height.map_or(core::ptr::null_mut(), |height| self.node[height].as_ptr())
    }

    /// The data edge at the current position. Requires a non-empty navigator.
    ///
    /// # Safety
    ///
    /// The navigator must be non-empty (`self.is_some()`, i.e. previously
    /// positioned by `init_first`/`init_last`/`init_offset`/`seek`), and
    /// every node on its root-to-leaf path — `self.node[0..=height]` — must
    /// still be a live, unmutated `CordRepBtree` (the navigator holds raw
    /// pointers into the tree without owning a reference; the caller is
    /// responsible for keeping the tree alive for as long as the navigator
    /// is used).
    #[inline]
    pub(crate) unsafe fn current(&self) -> *mut CordRep {
        unsafe {
            // SAFETY: per the contract above, `self.node[0]` is a live btree
            // leaf node and `self.index[0]` is within its `[begin, end)` range
            // (that invariant is maintained by every method that mutates them).
            debug_assert!(self.height.is_some());
            self.node[0].as_ptr().edge(self.index[0] as usize)
        }
    }

    /// Resets to the empty state.
    #[inline]
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "backward navigation kept for API completeness, no production caller yet"
        )
    )]
    pub(crate) fn reset(&mut self) {
        self.height = None;
    }

    /// Resets to `tree`, returning its first data edge.
    ///
    /// # Safety
    ///
    /// `tree` must be non-null, point to a live `CordRepBtree` with
    /// `size() > 0` and `height() <= MAX_HEIGHT`. This does not adopt a
    /// reference on `tree`: the navigator only stores raw pointers into it
    /// and its descendants, so the caller must keep `tree` alive and
    /// unmutated for as long as the navigator (and any edge pointer it
    /// returns) is used afterward.
    #[inline]
    pub(crate) unsafe fn init_first(&mut self, tree: *mut CordRepBtree) -> *mut CordRep {
        unsafe { self.init::<FRONT>(tree) }
    }

    /// Resets to `tree`, returning its last data edge.
    ///
    /// # Safety
    ///
    /// Same contract as [`init_first`](Self::init_first).
    #[inline]
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "backward navigation kept for API completeness, no production caller yet"
        )
    )]
    pub(crate) unsafe fn init_last(&mut self, tree: *mut CordRepBtree) -> *mut CordRep {
        unsafe { self.init::<BACK>(tree) }
    }

    /// # Safety
    ///
    /// `tree` must be non-null, point to a live `CordRepBtree` with
    /// `size() > 0` and `height() <= MAX_HEIGHT`. Does not adopt a
    /// reference on `tree`; see [`init_first`](Self::init_first).
    unsafe fn init<const IS_BACK: bool>(&mut self, mut tree: *mut CordRepBtree) -> *mut CordRep {
        unsafe {
            // SAFETY: the caller's contract guarantees `tree` is a live btree
            // node with a valid front/back index and `height() <= MAX_HEIGHT`
            // (so the loop below never indexes `self.node`/`self.index` out of
            // bounds); descending through `edge(index)` at each level stays
            // within a live child node because every non-leaf edge of a valid
            // btree points to a live child of one lesser height. `tree` (and
            // every node reached while descending) is non-null per the same
            // contract, so wrapping it as `NonNull` is sound.
            debug_assert!(!tree.is_null());
            debug_assert!(tree.size() > 0);
            debug_assert!(tree.height() <= MAX_HEIGHT);
            let mut height = tree.height();
            self.height = Some(height);
            let mut index = tree.index::<IS_BACK>();
            self.node[height] = NonNull::new_unchecked(tree);
            self.index[height] = small_u8(index);
            while height > 0 {
                height -= 1;
                tree = as_btree(tree.edge(index));
                self.node[height] = NonNull::new_unchecked(tree);
                index = tree.index::<IS_BACK>();
                self.index[height] = small_u8(index);
            }
            self.node[0].as_ptr().edge(index)
        }
    }

    /// Resets to `tree` at `offset`, returning the data edge containing it
    /// and the relative offset. Returns `NavPosition { edge: None, .. }`
    /// (leaving the navigator unchanged) if `offset >= tree.length`.
    ///
    /// # Safety
    ///
    /// `tree` must be non-null, point to a live `CordRepBtree` with
    /// `height() <= MAX_HEIGHT`. Does not adopt a reference on `tree`; see
    /// [`init_first`](Self::init_first).
    #[inline]
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "backward navigation kept for API completeness, no production caller yet"
        )
    )]
    pub(crate) unsafe fn init_offset(&mut self, tree: *mut CordRepBtree, offset: usize) -> NavPosition {
        unsafe {
            // SAFETY: the caller's contract guarantees `tree` is a live btree
            // node with `height() <= MAX_HEIGHT`, so reading `.length()` and
            // `.height()` and storing `tree` at `self.node[height]` (within
            // `MAX_DEPTH`) is sound; `seek` below is called only after `self`
            // has been positioned at a valid root, satisfying its own contract.
            debug_assert!(!tree.is_null());
            debug_assert!(tree.height() <= MAX_HEIGHT);
            if offset >= tree.length() {
                core::hint::cold_path();
                return NavPosition { edge: None, offset: 0 };
            }
            let height = tree.height();
            self.height = Some(height);
            self.node[height] = NonNull::new_unchecked(tree);
            self.seek(offset)
        }
    }

    /// Navigates to the data edge containing `offset`. Returns
    /// `NavPosition { edge: None, .. }` if `offset >= length`.
    ///
    /// # Safety
    ///
    /// The navigator must be non-empty (`self.is_some()`) with a live,
    /// unmutated tree reachable from `self.node[self.tree_height()]`; see
    /// [`current`](Self::current) for the general navigator-validity
    /// contract that every positioning method here relies on.
    #[inline]
    pub(crate) unsafe fn seek(&mut self, offset: usize) -> NavPosition {
        unsafe {
            // SAFETY: the caller's contract guarantees the root node at
            // `self.tree_height()` is live; each loop iteration descends via
            // `edge(index)` into a child of one lesser height, which is live by
            // the same well-formedness guarantee a valid btree provides (every
            // non-leaf edge points to a live child, hence non-null).
            debug_assert!(!self.btree().is_null());
            let mut height = self.tree_height();
            let mut edge = self.node[height].as_ptr();
            if offset >= edge.length() {
                core::hint::cold_path();
                return NavPosition { edge: None, offset: 0 };
            }
            let mut index = edge.index_of(offset);
            self.index[height] = small_u8(index.index);
            while height > 0 {
                height -= 1;
                edge = as_btree(edge.edge(index.index));
                self.node[height] = NonNull::new_unchecked(edge);
                index = edge.index_of(index.n);
                self.index[height] = small_u8(index.index);
            }
            let data_edge = edge.edge(index.index);
            NavPosition { edge: Some(NonNull::new_unchecked(data_edge)), offset: index.n }
        }
    }

    /// Navigates to the next data edge, or returns null (leaving the position
    /// unchanged) at the end.
    ///
    /// # Safety
    ///
    /// Same navigator-validity contract as [`current`](Self::current): must
    /// be non-empty, with every node on `self.node[0..=height]` live.
    #[inline]
    pub(crate) unsafe fn next(&mut self) -> *mut CordRep {
        unsafe {
            // SAFETY: per the contract above, `self.node[0]` is a live leaf and
            // `self.index[0]` is within its bounds; when it's already at the
            // last edge (`back()`), `next_up` is used instead of indexing past
            // `end()`.
            let edge = self.node[0].as_ptr();
            if self.index[0] as usize == edge.back() {
                self.next_up()
            } else {
                self.index[0] += 1;
                edge.edge(self.index[0] as usize)
            }
        }
    }

    /// Navigates to the previous data edge, or returns null (leaving the
    /// position unchanged) at the beginning.
    ///
    /// # Safety
    ///
    /// Same navigator-validity contract as [`current`](Self::current).
    #[inline]
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "backward navigation kept for API completeness, no production caller yet"
        )
    )]
    pub(crate) unsafe fn previous(&mut self) -> *mut CordRep {
        unsafe {
            // SAFETY: per the contract above, `self.node[0]` is a live leaf and
            // `self.index[0]` is within its bounds; when it's already at the
            // first edge (`begin()`), `previous_up` is used instead of
            // indexing before `begin()`.
            let edge = self.node[0].as_ptr();
            if self.index[0] as usize == edge.begin() {
                self.previous_up()
            } else {
                self.index[0] -= 1;
                edge.edge(self.index[0] as usize)
            }
        }
    }

    /// # Safety
    ///
    /// Same navigator-validity contract as [`current`](Self::current).
    /// Callers must additionally ensure `self.index[0] == self.node[0].back()`
    /// (i.e. the leaf is exhausted forward).
    unsafe fn next_up(&mut self) -> *mut CordRep {
        unsafe {
            // SAFETY: the contract above guarantees every node on the current
            // path is live; the search loop walks up parents (each already
            // live) and, once an unexhausted parent is found, descends back
            // down through `edge(index)`, which stays within live children by
            // the well-formedness of a valid btree.
            debug_assert_eq!(self.index[0] as usize, self.node[0].as_ptr().back());
            let mut height = 0usize;
            let mut edge;
            let mut index;
            loop {
                height += 1;
                if height > self.tree_height() {
                    return core::ptr::null_mut();
                }
                edge = self.node[height].as_ptr();
                index = self.index[height] as usize + 1;
                if index != edge.end() {
                    break;
                }
            }
            self.index[height] = small_u8(index);
            loop {
                height -= 1;
                edge = as_btree(edge.edge(index));
                self.node[height] = NonNull::new_unchecked(edge);
                index = edge.begin();
                self.index[height] = small_u8(index);
                if height == 0 {
                    break;
                }
            }
            edge.edge(index)
        }
    }

    /// # Safety
    ///
    /// Same navigator-validity contract as [`current`](Self::current).
    /// Callers must additionally ensure `self.index[0] ==
    /// self.node[0].begin()` (i.e. the leaf is exhausted backward).
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "backward navigation kept for API completeness, no production caller yet"
        )
    )]
    unsafe fn previous_up(&mut self) -> *mut CordRep {
        unsafe {
            // SAFETY: the contract above guarantees every node on the current
            // path is live; the search loop walks up parents (each already
            // live) and, once an unexhausted parent is found, descends back
            // down through `edge(index)`, which stays within live children by
            // the well-formedness of a valid btree.
            debug_assert_eq!(self.index[0] as usize, self.node[0].as_ptr().begin());
            let mut height = 0usize;
            let mut edge;
            let mut index;
            loop {
                height += 1;
                if height > self.tree_height() {
                    return core::ptr::null_mut();
                }
                edge = self.node[height].as_ptr();
                index = self.index[height] as usize;
                if index != edge.begin() {
                    break;
                }
            }
            index -= 1;
            self.index[height] = small_u8(index);
            loop {
                height -= 1;
                edge = as_btree(edge.edge(index));
                self.node[height] = NonNull::new_unchecked(edge);
                index = edge.back();
                self.index[height] = small_u8(index);
                if height == 0 {
                    break;
                }
            }
            edge.edge(index)
        }
    }

    /// Skips `n` bytes forward from the current data edge, returning the new
    /// edge and offset inside it. The state is unchanged if `n` is smaller
    /// than the current edge's length. Returns `NavPosition { edge: None,
    /// .. }` if the skip exceeds the tree.
    ///
    /// # Safety
    ///
    /// Same navigator-validity contract as [`current`](Self::current).
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "kept for API completeness alongside CordRepBtreeReader::skip, its only \
                      caller, which is itself dead outside tests"
        )
    )]
    pub(crate) unsafe fn skip(&mut self, mut n: usize) -> NavPosition {
        unsafe {
            // SAFETY: the contract above guarantees every node reachable from
            // `self.node[0..=tree_height()]` is live. The first loop only walks
            // forward within nodes already on the path (or one level up, which
            // is checked against `tree_height()` before indexing `self.node`),
            // so every `node.edge(index)` stays in bounds. The second loop
            // descends via `as_btree(edge)` into children reached the same way
            // `init`/`seek` do, which are live by btree well-formedness (hence
            // non-null).
            let mut height = 0usize;
            let mut index = self.index[0] as usize;
            let mut node = self.node[0].as_ptr();
            let mut edge = node.edge(index);

            // Find an edge of at least the length we need to skip, consuming all
            // smaller edges. Move up a level when the current level is exhausted;
            // hitting the top means the skip exceeds the tree length.
            while n >= edge.length() {
                n -= edge.length();
                index += 1;
                while index == node.end() {
                    height += 1;
                    if height > self.tree_height() {
                        return NavPosition { edge: None, offset: n };
                    }
                    node = self.node[height].as_ptr();
                    index = self.index[height] as usize;
                    index += 1;
                }
                edge = node.edge(index);
            }

            // If we moved up, descend to the leaf level consuming skipped edges.
            while height > 0 {
                node = as_btree(edge);
                self.index[height] = small_u8(index);
                height -= 1;
                self.node[height] = NonNull::new_unchecked(node);
                index = node.begin();
                edge = node.edge(index);
                while n >= edge.length() {
                    n -= edge.length();
                    index += 1;
                    debug_assert_ne!(index, node.end());
                    edge = node.edge(index);
                }
            }
            self.index[0] = small_u8(index);
            NavPosition { edge: Some(NonNull::new_unchecked(edge)), offset: n }
        }
    }

    /// Reads `n` bytes starting at `edge_offset` of the current data edge into
    /// a new tree. `ReadResult::n` is the number of bytes used from the last
    /// navigated-to edge. Returns `ReadResult { tree: None, .. }` if `n`
    /// exceeds the remaining data.
    ///
    /// # Safety
    ///
    /// Same navigator-validity contract as [`current`](Self::current), with
    /// `edge_offset` less than the current edge's length (checked below).
    /// The returned `ReadResult::tree`, if non-null, carries a reference the
    /// caller now owns.
    pub(crate) unsafe fn read(&mut self, edge_offset: usize, n: usize) -> ReadResult {
        unsafe {
            // SAFETY: the contract above guarantees every node reachable from
            // `self.node[0..=tree_height()]` is live, so walking forward /
            // ascending / descending through `edge(index)` and `as_btree(edge)`
            // (as in `skip`, above) stays in bounds and points at live nodes.
            // `CordRepBtree::new_with`/`new_node` return freshly allocated,
            // exclusively owned nodes of capacity `MAX_CAPACITY`; `subtree_end`
            // only increments once per edge consumed by this read, and the
            // read never spans more than `MAX_CAPACITY` edges below a node it
            // itself just allocated (mirroring the same accounting abseil's
            // `cord_rep_btree_navigator.cc` `Read()` uses), so every
            // `subtree.set_edge_ptr(subtree_end, ...)` call below stays within
            // `set_edge_ptr`'s `index < capacity()` bound, even though
            // `subtree_end` runs ahead of `subtree`'s `end` cursor (bumped
            // only afterwards, via `set_end`) throughout this function.
            // `substring`/`substring_from`/`ref_rep` are given live data edges
            // reached the same way, satisfying their own contracts. `subtree`
            // and `tree` are always freshly allocated (hence non-null) where
            // wrapped as `NonNull` below.
            let mut height = 0usize;
            let mut length = edge_offset + n;
            let mut index = self.index[0] as usize;
            let mut node = self.node[0].as_ptr();
            let mut edge = node.edge(index);
            debug_assert!(edge_offset < edge.length());

            if length < edge.length() {
                return ReadResult { tree: NonNull::new(substring(edge, edge_offset, n)), n: length };
            }

            // Consume all edges inside `length`, moving up a level when a level
            // is exhausted, until we hit the final edge to be (partially) read.
            let mut subtree = CordRepBtree::new_with(substring_from(edge, edge_offset));
            let mut subtree_end = 1usize;
            loop {
                length -= edge.length();
                index += 1;
                while index == node.end() {
                    self.index[height] = small_u8(index);
                    height += 1;
                    if height > self.tree_height() {
                        subtree.set_end(subtree_end);
                        if length == 0 {
                            return ReadResult { tree: Some(NonNull::new_unchecked(subtree.as_rep())), n: 0 };
                        }
                        unref(subtree.as_rep());
                        return ReadResult { tree: None, n: length };
                    }
                    if length != 0 {
                        subtree.set_end(subtree_end);
                        subtree = CordRepBtree::new_with(subtree.as_rep());
                        subtree_end = 1;
                    }
                    node = self.node[height].as_ptr();
                    index = self.index[height] as usize;
                    index += 1;
                }
                edge = node.edge(index);
                if length >= edge.length() {
                    subtree.add_length(edge.length());
                    subtree.set_edge_ptr(subtree_end, ref_rep(edge));
                    subtree_end += 1;
                }
                if length < edge.length() {
                    break;
                }
            }
            let tree = subtree;
            subtree.add_length(length);

            // If we moved up, descend to the leaf level consuming all edges to be
            // read, adding "down" nodes to `subtree`.
            while height > 0 {
                node = as_btree(edge);
                self.index[height] = small_u8(index);
                height -= 1;
                self.node[height] = NonNull::new_unchecked(node);
                index = node.begin();
                edge = node.edge(index);

                if length != 0 {
                    let right = CordRepBtree::new_node(height);
                    right.set_length(length);
                    subtree.set_edge_ptr(subtree_end, right.as_rep());
                    subtree_end += 1;
                    subtree.set_end(subtree_end);
                    subtree = right;
                    subtree_end = 0;
                    while length >= edge.length() {
                        subtree.set_edge_ptr(subtree_end, ref_rep(edge));
                        subtree_end += 1;
                        length -= edge.length();
                        index += 1;
                        edge = node.edge(index);
                    }
                }
            }
            // Add any partial edge still remaining at the leaf level.
            if length != 0 {
                subtree.set_edge_ptr(subtree_end, substring(edge, 0, length));
                subtree_end += 1;
            }
            subtree.set_end(subtree_end);
            self.index[0] = small_u8(index);
            ReadResult { tree: Some(NonNull::new_unchecked(tree.as_rep())), n: length }
        }
    }
}
