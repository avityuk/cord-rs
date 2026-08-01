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
use core::mem::offset_of;
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
unsafe fn create_substring(mut rep: *mut CordRep, mut offset: usize, n: usize) -> *mut CordRep {
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
    Box::into_raw(Box::new(CordRepSubstring { rep: CordRep::new(n, SUBSTRING), start: offset, child: rep }))
        .cast()
}

/// Returns `rep` if `n == rep.length`, null (unreffing `rep`) if `n == 0`,
/// else a substring. Adopts a reference on `rep`.
#[inline]
unsafe fn make_substring(rep: *mut CordRep, offset: usize, n: usize) -> *mut CordRep {
    if n == rep.length() {
        return rep;
    }
    if n == 0 {
        unref(rep);
        return core::ptr::null_mut();
    }
    create_substring(rep, offset, n)
}

/// `make_substring(rep, offset, rep.length - offset)`.
#[inline]
unsafe fn make_substring_from(rep: *mut CordRep, offset: usize) -> *mut CordRep {
    if offset == 0 {
        return rep;
    }
    create_substring(rep, offset, rep.length() - offset)
}

/// Resizes `edge` to `length`, adopting a reference on `edge`. If
/// `is_mutable`, flats and substrings are resized in place; otherwise a new
/// substring is returned. Requires `0 < length <= edge.length`.
unsafe fn resize_edge(edge: *mut CordRep, length: usize, is_mutable: bool) -> *mut CordRep {
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

/// Removes `n` bytes from the consumed end of `s`.
#[inline]
fn consume<const IS_BACK: bool>(s: &[u8], n: usize) -> &[u8] {
    if IS_BACK { &s[n..] } else { &s[..s.len() - n] }
}

/// Copies `n` bytes from the consumed end of `s` to `dst` and returns the rest.
#[inline]
unsafe fn consume_copy<const IS_BACK: bool>(dst: *mut u8, s: &[u8], n: usize) -> &[u8] {
    if IS_BACK {
        core::ptr::copy_nonoverlapping(s.as_ptr(), dst, n);
        &s[n..]
    } else {
        let offset = s.len() - n;
        core::ptr::copy_nonoverlapping(s.as_ptr().add(offset), dst, n);
        &s[..offset]
    }
}

unsafe fn delete_substring(substring: *mut CordRepSubstring) {
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

/// Deletes a leaf node data edge. Requires `is_data_edge(rep)`.
unsafe fn delete_leaf_edge(rep: *mut CordRep) {
    debug_assert!(is_data_edge(rep));
    if rep.tag() >= FLAT {
        flat::delete(rep);
    } else if rep.tag() == EXTERNAL {
        external::CordRepExternal::delete(rep);
    } else {
        delete_substring(rep.cast());
    }
}

// --- Node accessors ---------------------------------------------------------

/// Accessors on raw btree node pointers. All methods require `self` to point
/// to a live btree node.
pub(crate) trait BtreePtr: Copy {
    unsafe fn as_rep(self) -> *mut CordRep;
    unsafe fn height(self) -> usize;
    unsafe fn begin(self) -> usize;
    unsafe fn end(self) -> usize;
    unsafe fn set_begin(self, begin: usize);
    unsafe fn set_end(self, end: usize);
    unsafe fn edge(self, index: usize) -> *mut CordRep;
    unsafe fn set_edge_ptr(self, index: usize, edge: *mut CordRep);

    #[inline]
    unsafe fn length(self) -> usize {
        self.as_rep().length()
    }
    #[inline]
    unsafe fn set_length(self, length: usize) {
        self.as_rep().set_length(length);
    }
    #[inline]
    unsafe fn add_length(self, delta: usize) {
        self.set_length(self.length() + delta);
    }
    #[inline]
    unsafe fn sub_length(self, delta: usize) {
        self.set_length(self.length() - delta);
    }
    #[inline]
    unsafe fn refcount<'a>(self) -> &'a super::Refcount {
        self.as_rep().refcount()
    }
    #[inline]
    unsafe fn back(self) -> usize {
        self.end() - 1
    }
    #[inline]
    unsafe fn size(self) -> usize {
        self.end() - self.begin()
    }
    #[inline]
    unsafe fn capacity(self) -> usize {
        MAX_CAPACITY
    }
    /// Index of the front or back edge.
    #[inline]
    unsafe fn index<const IS_BACK: bool>(self) -> usize {
        if IS_BACK { self.back() } else { self.begin() }
    }
    /// The front or back edge.
    #[inline]
    unsafe fn edge_at<const IS_BACK: bool>(self) -> *mut CordRep {
        self.edge(self.index::<IS_BACK>())
    }
    /// Decreases `begin` by `n` and returns the new value.
    #[inline]
    unsafe fn sub_fetch_begin(self, n: usize) -> usize {
        let new_begin = self.begin() - n;
        self.set_begin(new_begin);
        new_begin
    }
    /// Increases `end` by `n` and returns the previous value.
    #[inline]
    unsafe fn fetch_add_end(self, n: usize) -> usize {
        let current = self.end();
        self.set_end(current + n);
        current
    }
    /// Iterates the edges in `[begin, end)`, reading lazily.
    #[inline]
    unsafe fn edges_range(self, begin: usize, end: usize) -> impl Iterator<Item = *mut CordRep> {
        debug_assert!(begin <= end);
        debug_assert!(begin >= self.begin());
        debug_assert!(end <= self.end());
        (begin..end).map(move |i| unsafe { self.edge(i) })
    }
    /// Iterates all edges.
    #[inline]
    unsafe fn edges(self) -> impl Iterator<Item = *mut CordRep> {
        self.edges_range(self.begin(), self.end())
    }
    /// The data of the edge at `index`. Requires a leaf node.
    #[inline]
    unsafe fn data<'a>(self, index: usize) -> &'a [u8] {
        debug_assert_eq!(self.height(), 0);
        edge_data(self.edge(index))
    }

    /// Returns the index of the last edge starting on or before `offset` and
    /// the relative offset inside that edge. Requires `offset < length`.
    #[inline]
    unsafe fn index_of(self, mut offset: usize) -> Position {
        debug_assert!(offset < self.length());
        let mut index = self.begin();
        while offset >= self.edge(index).length() {
            offset -= self.edge(index).length();
            index += 1;
        }
        Position { index, n: offset }
    }

    /// Returns the index of the last edge starting *before* `offset` and the
    /// relative offset inside that edge. Requires `0 < offset <= length`.
    #[inline]
    unsafe fn index_before(self, mut offset: usize) -> Position {
        debug_assert!(offset > 0);
        debug_assert!(offset <= self.length());
        let mut index = self.begin();
        while offset > self.edge(index).length() {
            offset -= self.edge(index).length();
            index += 1;
        }
        Position { index, n: offset }
    }

    /// `index_before(front.n + offset)` optimized to start at `front.index`.
    #[inline]
    unsafe fn index_before_from(self, front: Position, offset: usize) -> Position {
        let mut index = front.index;
        let mut offset = offset + front.n;
        while offset > self.edge(index).length() {
            offset -= self.edge(index).length();
            index += 1;
        }
        Position { index, n: offset }
    }

    /// Returns the index of the edge ending at (or on) length `n` and the
    /// number of bytes inside that edge up to `n`. Requires `n <= length`.
    #[inline]
    unsafe fn index_of_length(self, n: usize) -> Position {
        debug_assert!(n <= self.length());
        let mut index = self.back();
        let mut strip = self.length() - n;
        while strip >= self.edge(index).length() {
            strip -= self.edge(index).length();
            index -= 1;
        }
        Position { index, n: self.edge(index).length() - strip }
    }

    /// Returns the index of the edge directly beyond the edge containing
    /// `offset` and the distance of that edge from `offset`.
    #[inline]
    unsafe fn index_beyond(self, offset: usize) -> Position {
        let mut off = 0;
        let mut index = self.begin();
        while offset > off {
            off += self.edge(index).length();
            index += 1;
        }
        Position { index, n: off - offset }
    }
}

impl BtreePtr for *mut CordRepBtree {
    #[inline]
    unsafe fn as_rep(self) -> *mut CordRep {
        self.cast()
    }
    #[inline]
    unsafe fn height(self) -> usize {
        (*self).rep.storage[0] as usize
    }
    #[inline]
    unsafe fn begin(self) -> usize {
        (*self).rep.storage[1] as usize
    }
    #[inline]
    unsafe fn end(self) -> usize {
        (*self).rep.storage[2] as usize
    }
    #[inline]
    unsafe fn set_begin(self, begin: usize) {
        (*self).rep.storage[1] = small_u8(begin);
    }
    #[inline]
    unsafe fn set_end(self, end: usize) {
        (*self).rep.storage[2] = small_u8(end);
    }
    #[inline]
    unsafe fn edge(self, index: usize) -> *mut CordRep {
        debug_assert!(index >= self.begin());
        debug_assert!(index < self.end());
        (*self).edges[index]
    }
    #[inline]
    unsafe fn set_edge_ptr(self, index: usize, edge: *mut CordRep) {
        (*self).edges[index] = edge;
    }
}

/// Casts a rep known to be a btree node.
#[inline]
pub(crate) unsafe fn as_btree(rep: *mut CordRep) -> *mut CordRepBtree {
    debug_assert!(rep.is_btree());
    rep.cast()
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
    #[inline]
    unsafe fn build_stack(&mut self, mut tree: *mut CordRepBtree, depth: usize) -> *mut CordRepBtree {
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

    /// Builds a stack with the invariant that all nodes are privately owned.
    #[inline]
    unsafe fn build_owned_stack(&mut self, mut tree: *mut CordRepBtree, height: usize) {
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

    /// Processes the final top level result action for the tree.
    #[inline]
    unsafe fn finalize(tree: *mut CordRepBtree, result: OpResult) -> *mut CordRepBtree {
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

    /// Propagates `result` up into all nodes of the stack starting at `depth`.
    /// `length` is the extra length added at the lowest level. If `PROPAGATE`
    /// is set, copied node values are updated into the stack for iterative
    /// processing on the same stack.
    #[inline]
    unsafe fn unwind<const PROPAGATE: bool>(
        &mut self,
        tree: *mut CordRepBtree,
        mut depth: usize,
        length: usize,
        mut result: OpResult,
    ) -> *mut CordRepBtree {
        if depth != 0 {
            loop {
                depth -= 1;
                let node = self.stack[depth];
                let owned = depth < self.share_depth;
                match result.action {
                    Action::Popped => {
                        debug_assert!(!PROPAGATE);
                        result = CordRepBtree::add_edge::<IS_BACK>(node, owned, result.tree.as_rep(), length);
                    }
                    Action::Copied => {
                        result = CordRepBtree::set_edge::<IS_BACK>(node, owned, result.tree.as_rep(), length);
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

    #[inline]
    unsafe fn propagate(
        &mut self,
        tree: *mut CordRepBtree,
        depth: usize,
        length: usize,
        result: OpResult,
    ) -> *mut CordRepBtree {
        self.unwind::<true>(tree, depth, length, result)
    }
}

// --- Construction / destruction --------------------------------------------

impl CordRepBtree {
    #[inline]
    unsafe fn alloc(length: usize, height: usize, begin: usize, end: usize) -> *mut CordRepBtree {
        let mut rep = CordRep::new(length, BTREE);
        rep.storage = [small_u8(height), small_u8(begin), small_u8(end)];
        Box::into_raw(Box::new(CordRepBtree { rep, edges: [core::ptr::null_mut(); MAX_CAPACITY] }))
    }

    /// Creates a new empty node at `height`.
    #[inline]
    pub(crate) unsafe fn new_node(height: usize) -> *mut CordRepBtree {
        Self::alloc(0, height, 0, 0)
    }

    /// Creates a new node containing `rep`, at height `rep.height + 1` for a
    /// btree `rep` and 0 otherwise.
    #[inline]
    pub(crate) unsafe fn new_with(rep: *mut CordRep) -> *mut CordRepBtree {
        let height = if rep.is_btree() { as_btree(rep).height() + 1 } else { 0 };
        let tree = Self::alloc(rep.length(), height, 0, 1);
        (*tree).edges[0] = rep;
        tree
    }

    /// Creates a new node containing `front` and `back`, which must be of
    /// equal height.
    #[inline]
    pub(crate) unsafe fn new_pair(front: *mut CordRepBtree, back: *mut CordRepBtree) -> *mut CordRepBtree {
        debug_assert_eq!(front.height(), back.height());
        let tree = Self::alloc(front.length() + back.length(), front.height() + 1, 0, 2);
        (*tree).edges[0] = front.as_rep();
        (*tree).edges[1] = back.as_rep();
        tree
    }

    /// Creates a btree from `rep`, adopting a reference. Returns `rep` if it
    /// already is a btree, else a new leaf containing the data edge `rep`.
    #[inline]
    pub(crate) unsafe fn create(rep: *mut CordRep) -> *mut CordRepBtree {
        if is_data_edge(rep) {
            return Self::new_with(rep);
        }
        assert!(rep.is_btree(), "cord-rs: unexpected node type {} in CordRepBtree::create", rep.tag());
        rep.cast()
    }

    /// Frees the node itself.
    #[inline]
    pub(crate) unsafe fn delete(tree: *mut CordRepBtree) {
        drop(Box::from_raw(tree));
    }

    /// Unrefs all edges in `[begin, end)` of `tree` (assumed likely one).
    #[inline]
    pub(crate) unsafe fn unref_edges(tree: *mut CordRepBtree, begin: usize, end: usize) {
        for edge in tree.edges_range(begin, end) {
            if !edge.refcount().decrement() {
                core::hint::cold_path();
                super::destroy(edge);
            }
        }
    }

    /// Destroys `tree`, whose reference count reached zero.
    pub(crate) unsafe fn destroy(tree: *mut CordRepBtree) {
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

    unsafe fn destroy_tree<const SIZE: usize>(tree: *mut CordRepBtree) {
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

    // --- Copies -------------------------------------------------------------

    /// Raw copy of this node with `new_length`, copying all properties but
    /// without adding references to the edges.
    #[inline]
    unsafe fn copy_raw(this: *mut CordRepBtree, new_length: usize) -> *mut CordRepBtree {
        // Everything from `tag` onwards is plain data: copy it in one go.
        const OFFSET: usize = offset_of!(CordRep, tag);
        let tree = Self::alloc(new_length, 0, 0, 0);
        core::ptr::copy_nonoverlapping(
            this.cast::<u8>().add(OFFSET),
            tree.cast::<u8>().add(OFFSET),
            core::mem::size_of::<CordRepBtree>() - OFFSET,
        );
        tree
    }

    /// Full copy of this node, adding a reference on all edges.
    #[inline]
    unsafe fn copy(this: *mut CordRepBtree) -> *mut CordRepBtree {
        let tree = Self::copy_raw(this, this.length());
        for rep in this.edges() {
            ref_rep(rep);
        }
        tree
    }

    /// Copy of the edges starting at `begin`, with `new_length`.
    #[inline]
    unsafe fn copy_to_end_from(
        this: *mut CordRepBtree,
        begin: usize,
        new_length: usize,
    ) -> *mut CordRepBtree {
        debug_assert!(begin >= this.begin());
        debug_assert!(begin <= this.end());
        let tree = Self::copy_raw(this, new_length);
        tree.set_begin(begin);
        for edge in tree.edges() {
            ref_rep(edge);
        }
        tree
    }

    /// Copy of the edges up to `end`, with `new_length`.
    #[inline]
    unsafe fn copy_begin_to(this: *mut CordRepBtree, end: usize, new_length: usize) -> *mut CordRepBtree {
        debug_assert!(end <= this.capacity());
        debug_assert!(end >= this.begin());
        let tree = Self::copy_raw(this, new_length);
        tree.set_end(end);
        for edge in tree.edges() {
            ref_rep(edge);
        }
        tree
    }

    /// Returns a tree containing edges `[begin, end)` with `new_length`,
    /// consuming `tree` (in place if privately owned).
    unsafe fn consume_begin_to(tree: *mut CordRepBtree, end: usize, new_length: usize) -> *mut CordRepBtree {
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

    /// Extracts the front edge from `tree`, consuming `tree`.
    unsafe fn extract_front(tree: *mut CordRepBtree) -> *mut CordRep {
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

    // --- Edge manipulation --------------------------------------------------

    /// Aligns existing edges to start at index 0.
    #[inline]
    unsafe fn align_begin(this: *mut CordRepBtree) {
        let delta = this.begin();
        if delta != 0 {
            core::hint::cold_path();
            let new_end = this.end() - delta;
            this.set_begin(0);
            this.set_end(new_end);
            for i in 0..new_end {
                (*this).edges[i] = (*this).edges[i + delta];
            }
        }
    }

    /// Aligns existing edges to end at `capacity`.
    #[inline]
    unsafe fn align_end(this: *mut CordRepBtree) {
        let delta = this.capacity() - this.end();
        if delta != 0 {
            let new_begin = this.begin() + delta;
            let new_end = this.end() + delta;
            this.set_begin(new_begin);
            this.set_end(new_end);
            let mut i = new_end;
            while i > new_begin {
                i -= 1;
                (*this).edges[i] = (*this).edges[i - delta];
            }
        }
    }

    /// Adds `rep` at the back or front. Requires spare capacity.
    #[inline]
    unsafe fn add<const IS_BACK: bool>(this: *mut CordRepBtree, rep: *mut CordRep) {
        if IS_BACK {
            Self::align_begin(this);
            let idx = this.fetch_add_end(1);
            (*this).edges[idx] = rep;
        } else {
            Self::align_end(this);
            let idx = this.sub_fetch_begin(1);
            (*this).edges[idx] = rep;
        }
    }

    /// Adds all edges of `src` at the back or front. Requires spare capacity.
    #[inline]
    unsafe fn add_edges_from<const IS_BACK: bool>(this: *mut CordRepBtree, src: *mut CordRepBtree) {
        let (sb, se) = (src.begin(), src.end());
        if IS_BACK {
            Self::align_begin(this);
            let mut new_end = this.end();
            for i in sb..se {
                (*this).edges[new_end] = (*src).edges[i];
                new_end += 1;
            }
            this.set_end(new_end);
        } else {
            Self::align_end(this);
            let new_begin = this.begin() - (se - sb);
            this.set_begin(new_begin);
            for (dst, i) in (new_begin..).zip(sb..se) {
                (*this).edges[dst] = (*src).edges[i];
            }
        }
    }

    /// Adds `edge` to `this` if possible: in place if `owned`, on a copy if
    /// shared, or as a new popped leg if at capacity.
    #[inline]
    unsafe fn add_edge<const IS_BACK: bool>(
        this: *mut CordRepBtree,
        owned: bool,
        edge: *mut CordRep,
        delta: usize,
    ) -> OpResult {
        if this.size() >= MAX_CAPACITY {
            return OpResult { tree: Self::new_with(edge), action: Action::Popped };
        }
        let result = Self::to_op_result(this, owned);
        Self::add::<IS_BACK>(result.tree, edge);
        result.tree.add_length(delta);
        result
    }

    /// Replaces the front or back edge with `edge`, in place if `owned` or on
    /// a copy otherwise. Adopts a reference on `edge`.
    unsafe fn set_edge<const IS_BACK: bool>(
        this: *mut CordRepBtree,
        owned: bool,
        edge: *mut CordRep,
        delta: usize,
    ) -> OpResult {
        let idx = this.index::<IS_BACK>();
        let result = if owned {
            unref((*this).edges[idx]);
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
        (*result.tree).edges[idx] = edge;
        result.tree.add_length(delta);
        result
    }

    #[inline]
    unsafe fn to_op_result(this: *mut CordRepBtree, owned: bool) -> OpResult {
        if owned {
            OpResult { tree: this, action: Action::InPlace }
        } else {
            OpResult { tree: Self::copy(this), action: Action::Copied }
        }
    }

    // --- Append / prepend ---------------------------------------------------

    unsafe fn add_cord_rep<const IS_BACK: bool>(
        tree: *mut CordRepBtree,
        rep: *mut CordRep,
    ) -> *mut CordRepBtree {
        let depth = tree.height();
        let length = rep.length();
        let mut ops = StackOperations::<IS_BACK>::new();
        let leaf = ops.build_stack(tree, depth);
        let result = Self::add_edge::<IS_BACK>(leaf, ops.owned(depth), rep, length);
        ops.unwind::<false>(tree, depth, length, result)
    }

    /// Creates a new leaf containing as much of `data` as possible.
    unsafe fn new_leaf<const IS_BACK: bool>(mut data: &[u8], extra: usize) -> *mut CordRepBtree {
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
                (*leaf).edges[end] = f;
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
                (*leaf).edges[begin] = f;
                data = consume_copy::<IS_BACK>(flat::data(f), data, n);
            }
            leaf.set_length(length);
            leaf.set_begin(begin);
        }
        leaf
    }

    /// Adds data to this leaf until all data is consumed or the node is full.
    /// Returns the remaining data. Requires a non-full leaf and non-empty data.
    unsafe fn add_data_to_leaf<'a, const IS_BACK: bool>(
        this: *mut CordRepBtree,
        mut data: &'a [u8],
        extra: usize,
    ) -> &'a [u8] {
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
                (*this).edges[idx] = f;
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
                (*this).edges[idx] = f;
                data = consume_copy::<IS_BACK>(flat::data(f), data, n);
                if data.is_empty() || this.begin() == 0 {
                    break;
                }
            }
        }
        data
    }

    unsafe fn add_data<const IS_BACK: bool>(
        mut tree: *mut CordRepBtree,
        mut data: &[u8],
        extra: usize,
    ) -> *mut CordRepBtree {
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
            let result = OpResult { tree: Self::new_leaf::<IS_BACK>(data, extra), action: Action::Popped };
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

    /// Merges `src` into `dst` before (`FRONT`) or after (`BACK`) it.
    /// Requires `dst.height >= src.height`.
    unsafe fn merge<const IS_BACK: bool>(
        dst: *mut CordRepBtree,
        src: *mut CordRepBtree,
    ) -> *mut CordRepBtree {
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

    /// Returns a tree containing `left` followed by `right`.
    pub(crate) unsafe fn merge_trees(left: *mut CordRepBtree, right: *mut CordRepBtree) -> *mut CordRepBtree {
        if left.height() >= right.height() {
            Self::merge::<BACK>(left, right)
        } else {
            Self::merge::<FRONT>(right, left)
        }
    }

    /// Appends `rep` (a data edge or a btree) to `tree`.
    #[inline]
    pub(crate) unsafe fn append(tree: *mut CordRepBtree, rep: *mut CordRep) -> *mut CordRepBtree {
        if is_data_edge(rep) {
            return Self::add_cord_rep::<BACK>(tree, rep);
        }
        assert!(rep.is_btree(), "cord-rs: unexpected node type {} in CordRepBtree::append", rep.tag());
        Self::merge_trees(tree, rep.cast())
    }

    /// Prepends `rep` (a data edge or a btree) to `tree`.
    #[inline]
    pub(crate) unsafe fn prepend(tree: *mut CordRepBtree, rep: *mut CordRep) -> *mut CordRepBtree {
        if is_data_edge(rep) {
            return Self::add_cord_rep::<FRONT>(tree, rep);
        }
        assert!(rep.is_btree(), "cord-rs: unexpected node type {} in CordRepBtree::prepend", rep.tag());
        Self::merge_trees(rep.cast(), tree)
    }

    /// Appends `data` to `tree`. `extra` is a hint for additional capacity to
    /// allocate in any newly created flat. Data of any size is supported.
    #[inline]
    pub(crate) unsafe fn append_data(
        tree: *mut CordRepBtree,
        data: &[u8],
        extra: usize,
    ) -> *mut CordRepBtree {
        Self::add_data::<BACK>(tree, data, extra)
    }

    /// Prepends `data` to `tree`. See [`append_data`](Self::append_data).
    #[inline]
    pub(crate) unsafe fn prepend_data(
        tree: *mut CordRepBtree,
        data: &[u8],
        extra: usize,
    ) -> *mut CordRepBtree {
        Self::add_data::<FRONT>(tree, data, extra)
    }

    // --- Sub trees ----------------------------------------------------------

    /// Partial copy of the tree containing all data starting at `offset`.
    /// Requires `offset < length`. Does not consume `this`.
    unsafe fn copy_suffix(this: *mut CordRepBtree, mut offset: usize) -> CopyResult {
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
                (*sub).edges[begin] = make_substring(ref_rep(edge), offset, len);
                return result;
            }

            node = as_btree(edge);
            pos = node.index_beyond(offset);

            let nsub = Self::copy_to_end_from(node, pos.index, len);
            (*sub).edges[begin] = nsub.as_rep();
            sub = nsub;
        }
        sub.set_begin(pos.index);
        result
    }

    /// Partial copy of the tree containing the first `n` bytes. The result
    /// may be less high than the tree, or a single data edge (height -1) if
    /// `allow_folding`. Requires `0 < n <= length`. Does not consume `this`.
    unsafe fn copy_prefix(this: *mut CordRepBtree, mut n: usize, allow_folding: bool) -> CopyResult {
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
                (*sub).edges[end] = make_substring(ref_rep(edge), 0, n);
                end += 1;
                sub.set_end(end);
                Self::assert_valid(as_btree(result.edge), true);
                return result;
            }

            node = as_btree(edge);
            pos = node.index_of(n);
            let nsub = Self::copy_begin_to(node, pos.index, n);
            (*sub).edges[end] = nsub.as_rep();
            end += 1;
            sub.set_end(end);
            sub = nsub;
        }
        sub.set_end(pos.index);
        Self::assert_valid(as_btree(result.edge), true);
        result
    }

    /// Returns a new tree containing `n` bytes starting at `offset`, sharing
    /// nodes and edges with this tree where possible. Requires
    /// `offset + n <= length`. Returns null if `n == 0`. Does not consume
    /// `this`.
    pub(crate) unsafe fn sub_tree(this: *mut CordRepBtree, offset: usize, n: usize) -> *mut CordRep {
        debug_assert!(n <= this.length());
        debug_assert!(offset <= this.length() - n);
        if n == 0 {
            core::hint::cold_path();
            return core::ptr::null_mut();
        }

        let mut node = this;
        let mut height = height_to_isize(node.height());
        let mut front = node.index_of(offset);
        let mut left = (*node).edges[front.index];
        while front.n + n <= left.length() {
            height -= 1;
            if height < 0 {
                return make_substring(ref_rep(left), front.n, n);
            }
            node = as_btree(left);
            front = node.index_of(front.n);
            left = (*node).edges[front.index];
        }

        let back = node.index_before_from(front, n);
        let right = (*node).edges[back.index];
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
        (*sub).edges[end] = prefix.edge;
        end += 1;
        for r in node.edges_range(front.index + 1, back.index) {
            (*sub).edges[end] = ref_rep(r);
            end += 1;
        }
        (*sub).edges[end] = suffix.edge;
        end += 1;
        sub.set_end(end);
        sub.set_length(n);
        Self::assert_valid(sub, true).as_rep()
    }

    /// Removes `n` trailing bytes from `tree`, consuming it, and returns the
    /// resulting tree or data edge (in place where possible). Returns `tree`
    /// if `n == 0` and null if `n == length`.
    pub(crate) unsafe fn remove_suffix(mut tree: *mut CordRepBtree, n: usize) -> *mut CordRep {
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
                (*tree).edges[pos.index] = resize_edge(edge, length, edge_is_mutable);
                return Self::assert_valid(top, true).as_rep();
            }
            height -= 1;

            if !edge_is_mutable {
                // We can't remove suffixes in place down this edge: replace it
                // with a prefix copy instead.
                (*tree).edges[pos.index] = Self::copy_prefix(as_btree(edge), length, false).edge;
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

    // --- Queries ------------------------------------------------------------

    /// Returns the data if this tree holds a single data edge.
    #[inline]
    pub(crate) unsafe fn as_flat<'a>(this: *mut CordRepBtree) -> Option<&'a [u8]> {
        if this.height() == 0 && this.size() == 1 { Some(this.data(this.begin())) } else { None }
    }

    /// Returns the `n` bytes starting at `offset` if they are contained in a
    /// single data edge. Requires `offset + n <= length`.
    pub(crate) unsafe fn as_flat_range<'a>(
        this: *mut CordRepBtree,
        mut offset: usize,
        n: usize,
    ) -> Option<&'a [u8]> {
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

    /// Returns the byte at `offset`. Requires `offset < length`.
    pub(crate) unsafe fn get_byte(this: *mut CordRepBtree, mut offset: usize) -> u8 {
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

    /// Returns a pointer to and length of up to `size` bytes of spare capacity
    /// in the last flat of this tree, increasing the lengths of the flat and
    /// all nodes on the path by that amount, iff:
    /// - none of the nodes down to the flat are shared,
    /// - the last data edge is a non-shared flat with available capacity.
    ///
    /// The caller must immediately initialize the returned bytes. Requires
    /// `this.refcount().is_one()`.
    pub(crate) unsafe fn get_append_buffer(this: *mut CordRepBtree, size: usize) -> Option<(*mut u8, usize)> {
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

    /// Extracts the right-most flat from `tree` iff the tree and all nodes
    /// down to it are unshared, it is an unshared flat, and it has at least
    /// `extra_capacity` bytes available. Returns `{tree, null}` otherwise.
    /// On success the flat is removed from the tree, which may collapse to a
    /// single data edge or to null.
    pub(crate) unsafe fn extract_append_buffer(
        mut tree: *mut CordRepBtree,
        extra_capacity: usize,
    ) -> ExtractResult {
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

    // --- Rebuild ------------------------------------------------------------

    unsafe fn rebuild_into(
        stack: &mut [*mut CordRepBtree; MAX_DEPTH + 1],
        tree: *mut CordRepBtree,
        consume: bool,
    ) {
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

    /// Creates a fully balanced tree from all data edges of `tree`, consuming
    /// it. Invoked automatically when a tree exceeds the maximum height.
    pub(crate) unsafe fn rebuild(tree: *mut CordRepBtree) -> *mut CordRepBtree {
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

    // --- Diagnostics --------------------------------------------------------

    /// Checks that `tree` is valid and internally consistent. If `shallow`,
    /// only the node and the cumulative length / types / heights of its
    /// direct children are checked (unless exhaustive validation is enabled).
    pub(crate) unsafe fn check_valid(tree: *const CordRepBtree, shallow: bool) -> Result<(), String> {
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

    /// Returns `true` if `tree` is valid. See [`check_valid`](Self::check_valid).
    pub(crate) unsafe fn is_valid(tree: *const CordRepBtree, shallow: bool) -> bool {
        Self::check_valid(tree, shallow).is_ok()
    }

    /// Asserts (in debug builds) that `tree` is valid and returns it.
    #[inline]
    pub(crate) unsafe fn assert_valid(tree: *mut CordRepBtree, shallow: bool) -> *mut CordRepBtree {
        if cfg!(debug_assertions)
            && let Err(msg) = Self::check_valid(tree, shallow)
        {
            let mut dump = String::new();
            let _ = Self::dump(tree.as_rep(), "CordRepBtree validation failed:", false, &mut dump);
            panic!("{msg}\n{dump}");
        }
        tree
    }

    /// Dumps the structure of `rep` (a btree, substring, flat or external) to
    /// `out`. Intended for debugging and testing only.
    pub(crate) unsafe fn dump(
        rep: *const CordRep,
        label: &str,
        include_contents: bool,
        out: &mut dyn fmt::Write,
    ) -> fmt::Result {
        writeln!(out, "===================================")?;
        if !label.is_empty() {
            writeln!(out, "{label}")?;
            writeln!(out, "-----------------------------------")?;
        }
        if rep.is_null() { writeln!(out, "NULL") } else { Self::dump_all(rep, include_contents, out, 0) }
    }

    unsafe fn dump_all(
        rep: *const CordRep,
        include_contents: bool,
        out: &mut dyn fmt::Write,
        depth: usize,
    ) -> fmt::Result {
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
