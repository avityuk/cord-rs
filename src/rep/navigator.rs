//! Navigation over the data edges of a btree.
//!
//! A navigator keeps a stack of `(node, index)` pairs from the root to the
//! current leaf edge and can move forward / backward, seek to an offset, skip
//! bytes, and read a sub range into a new tree. Port of abseil's
//! `cord_rep_btree_navigator.{h,cc}`.

use super::btree::{BACK, BtreePtr, CordRepBtree, FRONT, MAX_DEPTH, MAX_HEIGHT, as_btree};
use super::{CordRep, CordRepSubstring, RepPtr, SUBSTRING, is_data_edge, ref_rep, small_u8, unref};

/// A data edge and an offset inside it, as returned by `seek` / `skip`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct NavPosition {
    /// The edge, or null if the requested position is beyond the tree.
    pub(crate) edge: *mut CordRep,
    pub(crate) offset: usize,
}

/// Result of [`CordRepBtreeNavigator::read`].
#[derive(Clone, Copy, Debug)]
pub(crate) struct ReadResult {
    /// The tree holding the read data (identical to `sub_tree(...)` on the
    /// navigated tree), or null if the read exceeded the tree.
    pub(crate) tree: *mut CordRep,
    /// Bytes used from the last navigated-to edge.
    pub(crate) n: usize,
}

/// Returns a substring of the data edge `rep` (adopting no reference on
/// `rep`; the result holds its own). Null if `n == 0`; `ref(rep)` if `n ==
/// rep.length`.
unsafe fn substring(mut rep: *mut CordRep, mut offset: usize, n: usize) -> *mut CordRep {
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
    if rep.tag() == SUBSTRING {
        let sub: *mut CordRepSubstring = rep.cast();
        offset += (*sub).start;
        rep = (*sub).child;
    }
    debug_assert!(rep.is_external() || rep.is_flat());
    Box::into_raw(Box::new(CordRepSubstring {
        rep: CordRep::new(n, SUBSTRING),
        start: offset,
        child: ref_rep(rep),
    }))
    .cast()
}

#[inline]
unsafe fn substring_from(rep: *mut CordRep, offset: usize) -> *mut CordRep {
    substring(rep, offset, rep.length() - offset)
}

/// See the [module documentation](self).
#[derive(Clone, Copy)]
pub(crate) struct CordRepBtreeNavigator {
    /// Height of the current tree, or `None` if empty.
    height: Option<usize>,
    /// Path to the current data edge: `node[0].edge(index[0])`. Undefined
    /// until initialized (`height >= 0`).
    index: [u8; MAX_DEPTH],
    node: [*mut CordRepBtree; MAX_DEPTH],
}

impl Default for CordRepBtreeNavigator {
    fn default() -> Self {
        Self::new()
    }
}

impl CordRepBtreeNavigator {
    /// An empty navigator.
    pub(crate) const fn new() -> Self {
        Self { height: None, index: [0; MAX_DEPTH], node: [core::ptr::null_mut(); MAX_DEPTH] }
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
        self.height.map_or(core::ptr::null_mut(), |height| self.node[height])
    }

    /// The data edge at the current position. Requires a non-empty navigator.
    #[inline]
    pub(crate) unsafe fn current(&self) -> *mut CordRep {
        debug_assert!(self.height.is_some());
        self.node[0].edge(self.index[0] as usize)
    }

    /// Resets to the empty state.
    #[inline]
    pub(crate) fn reset(&mut self) {
        self.height = None;
    }

    /// Resets to `tree`, returning its first data edge.
    #[inline]
    pub(crate) unsafe fn init_first(&mut self, tree: *mut CordRepBtree) -> *mut CordRep {
        self.init::<FRONT>(tree)
    }

    /// Resets to `tree`, returning its last data edge.
    #[inline]
    pub(crate) unsafe fn init_last(&mut self, tree: *mut CordRepBtree) -> *mut CordRep {
        self.init::<BACK>(tree)
    }

    unsafe fn init<const IS_BACK: bool>(&mut self, mut tree: *mut CordRepBtree) -> *mut CordRep {
        debug_assert!(!tree.is_null());
        debug_assert!(tree.size() > 0);
        debug_assert!(tree.height() <= MAX_HEIGHT);
        let mut height = tree.height();
        self.height = Some(height);
        let mut index = tree.index::<IS_BACK>();
        self.node[height] = tree;
        self.index[height] = small_u8(index);
        while height > 0 {
            height -= 1;
            tree = as_btree(tree.edge(index));
            self.node[height] = tree;
            index = tree.index::<IS_BACK>();
            self.index[height] = small_u8(index);
        }
        self.node[0].edge(index)
    }

    /// Resets to `tree` at `offset`, returning the data edge containing it
    /// and the relative offset. Returns a null edge (leaving the navigator
    /// unchanged) if `offset >= tree.length`.
    #[inline]
    pub(crate) unsafe fn init_offset(&mut self, tree: *mut CordRepBtree, offset: usize) -> NavPosition {
        debug_assert!(!tree.is_null());
        debug_assert!(tree.height() <= MAX_HEIGHT);
        if offset >= tree.length() {
            core::hint::cold_path();
            return NavPosition { edge: core::ptr::null_mut(), offset: 0 };
        }
        let height = tree.height();
        self.height = Some(height);
        self.node[height] = tree;
        self.seek(offset)
    }

    /// Navigates to the data edge containing `offset`. Returns a null edge if
    /// `offset >= length`.
    #[inline]
    pub(crate) unsafe fn seek(&mut self, offset: usize) -> NavPosition {
        debug_assert!(!self.btree().is_null());
        let mut height = self.tree_height();
        let mut edge = self.node[height];
        if offset >= edge.length() {
            core::hint::cold_path();
            return NavPosition { edge: core::ptr::null_mut(), offset: 0 };
        }
        let mut index = edge.index_of(offset);
        self.index[height] = small_u8(index.index);
        while height > 0 {
            height -= 1;
            edge = as_btree(edge.edge(index.index));
            self.node[height] = edge;
            index = edge.index_of(index.n);
            self.index[height] = small_u8(index.index);
        }
        NavPosition { edge: edge.edge(index.index), offset: index.n }
    }

    /// Navigates to the next data edge, or returns null (leaving the position
    /// unchanged) at the end.
    #[inline]
    pub(crate) unsafe fn next(&mut self) -> *mut CordRep {
        let edge = self.node[0];
        if self.index[0] as usize == edge.back() {
            self.next_up()
        } else {
            self.index[0] += 1;
            edge.edge(self.index[0] as usize)
        }
    }

    /// Navigates to the previous data edge, or returns null (leaving the
    /// position unchanged) at the beginning.
    #[inline]
    pub(crate) unsafe fn previous(&mut self) -> *mut CordRep {
        let edge = self.node[0];
        if self.index[0] as usize == edge.begin() {
            self.previous_up()
        } else {
            self.index[0] -= 1;
            edge.edge(self.index[0] as usize)
        }
    }

    unsafe fn next_up(&mut self) -> *mut CordRep {
        debug_assert_eq!(self.index[0] as usize, self.node[0].back());
        let mut height = 0usize;
        let mut edge;
        let mut index;
        loop {
            height += 1;
            if height > self.tree_height() {
                return core::ptr::null_mut();
            }
            edge = self.node[height];
            index = self.index[height] as usize + 1;
            if index != edge.end() {
                break;
            }
        }
        self.index[height] = small_u8(index);
        loop {
            height -= 1;
            edge = as_btree(edge.edge(index));
            self.node[height] = edge;
            index = edge.begin();
            self.index[height] = small_u8(index);
            if height == 0 {
                break;
            }
        }
        edge.edge(index)
    }

    unsafe fn previous_up(&mut self) -> *mut CordRep {
        debug_assert_eq!(self.index[0] as usize, self.node[0].begin());
        let mut height = 0usize;
        let mut edge;
        let mut index;
        loop {
            height += 1;
            if height > self.tree_height() {
                return core::ptr::null_mut();
            }
            edge = self.node[height];
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
            self.node[height] = edge;
            index = edge.back();
            self.index[height] = small_u8(index);
            if height == 0 {
                break;
            }
        }
        edge.edge(index)
    }

    /// Skips `n` bytes forward from the current data edge, returning the new
    /// edge and offset inside it. The state is unchanged if `n` is smaller
    /// than the current edge's length. Returns a null edge if the skip
    /// exceeds the tree.
    pub(crate) unsafe fn skip(&mut self, mut n: usize) -> NavPosition {
        let mut height = 0usize;
        let mut index = self.index[0] as usize;
        let mut node = self.node[0];
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
                    return NavPosition { edge: core::ptr::null_mut(), offset: n };
                }
                node = self.node[height];
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
            self.node[height] = node;
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
        NavPosition { edge, offset: n }
    }

    /// Reads `n` bytes starting at `edge_offset` of the current data edge into
    /// a new tree. `ReadResult::n` is the number of bytes used from the last
    /// navigated-to edge. Returns a null tree if `n` exceeds the remaining
    /// data.
    pub(crate) unsafe fn read(&mut self, edge_offset: usize, n: usize) -> ReadResult {
        let mut height = 0usize;
        let mut length = edge_offset + n;
        let mut index = self.index[0] as usize;
        let mut node = self.node[0];
        let mut edge = node.edge(index);
        debug_assert!(edge_offset < edge.length());

        if length < edge.length() {
            return ReadResult { tree: substring(edge, edge_offset, n), n: length };
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
                        return ReadResult { tree: subtree.as_rep(), n: 0 };
                    }
                    unref(subtree.as_rep());
                    return ReadResult { tree: core::ptr::null_mut(), n: length };
                }
                if length != 0 {
                    subtree.set_end(subtree_end);
                    subtree = CordRepBtree::new_with(subtree.as_rep());
                    subtree_end = 1;
                }
                node = self.node[height];
                index = self.index[height] as usize;
                index += 1;
            }
            edge = node.edge(index);
            if length >= edge.length() {
                subtree.add_length(edge.length());
                (*subtree).edges[subtree_end] = ref_rep(edge);
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
            self.node[height] = node;
            index = node.begin();
            edge = node.edge(index);

            if length != 0 {
                let right = CordRepBtree::new_node(height);
                right.set_length(length);
                (*subtree).edges[subtree_end] = right.as_rep();
                subtree_end += 1;
                subtree.set_end(subtree_end);
                subtree = right;
                subtree_end = 0;
                while length >= edge.length() {
                    (*subtree).edges[subtree_end] = ref_rep(edge);
                    subtree_end += 1;
                    length -= edge.length();
                    index += 1;
                    edge = node.edge(index);
                }
            }
        }
        // Add any partial edge still remaining at the leaf level.
        if length != 0 {
            (*subtree).edges[subtree_end] = substring(edge, 0, length);
            subtree_end += 1;
        }
        subtree.set_end(subtree_end);
        self.index[0] = small_u8(index);
        ReadResult { tree: tree.as_rep(), n: length }
    }
}
