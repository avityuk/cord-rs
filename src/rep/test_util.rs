//! Test helpers shared by the rep test suites.
//!
//! Port of abseil's `cord_rep_test_util.h` plus the tree builders from
//! `cord_rep_btree_test.cc` (`MakeHexFlat`, `MakeLeaf`, `MakeTree`,
//! `CreateTree`, `CreateTreeReverse`, `GetLeafEdges`).
//!
//! All builders follow the rep reference counting conventions: they adopt
//! the reps they are given and return a new reference to the caller.

use super::btree::{BtreePtr, CordRepBtree, MAX_CAPACITY, as_btree};
use super::external::CordRepExternal;
use super::{BTREE, CordRep, CordRepSubstring, EXTERNAL, FLAT, RepPtr, SUBSTRING, flat, ref_rep, unref};

/// Anything that can be viewed as a `*mut CordRep`.
pub(crate) trait AsRepPtr: Copy {
    fn as_rep_ptr(self) -> *mut CordRep;
}

impl AsRepPtr for *mut CordRep {
    fn as_rep_ptr(self) -> *mut CordRep {
        self
    }
}

impl AsRepPtr for *mut CordRepBtree {
    fn as_rep_ptr(self) -> *mut CordRep {
        self.cast()
    }
}

impl AsRepPtr for *mut CordRepSubstring {
    fn as_rep_ptr(self) -> *mut CordRep {
        self.cast()
    }
}

/// `MakeSubstring(start, len, rep)`: a substring node over `rep`, adopting
/// `rep`. `len == 0` means "all of `rep` starting at `start`". Performs no
/// validation so that tests can build deliberately invalid nodes (e.g. a
/// substring of a substring).
///
/// # Safety
///
/// `rep` must be a non-null pointer to a live rep; the caller donates its
/// reference, which becomes the new substring's `child`.
pub(crate) unsafe fn make_substring(start: usize, len: usize, rep: *mut CordRep) -> *mut CordRep {
    unsafe {
        let length = if len == 0 { rep.length() - start } else { len };
        Box::into_raw(Box::new(CordRepSubstring { rep: CordRep::new(length, SUBSTRING), start, child: rep }))
            .cast()
    }
}

/// `MakeFlat(value)`: a flat holding exactly `value`.
///
/// # Safety
///
/// None beyond `value.len() <= flat::MAX_FLAT_LENGTH` (asserted below).
pub(crate) unsafe fn make_flat(value: &[u8]) -> *mut CordRep {
    unsafe {
        assert!(value.len() <= flat::MAX_FLAT_LENGTH);
        flat::create(value, 0)
    }
}

/// `MakeExternal(s)`: an external node owning a private copy of `s`.
pub(crate) fn make_external(s: &[u8]) -> *mut CordRep {
    assert!(!s.is_empty());
    // SAFETY: asserted non-empty above.
    unsafe { CordRepExternal::create_global(s.to_vec()) }
}

/// A minimal `std::minstd_rand` (Park-Miller LCG, default seed 1). Only used
/// to produce deterministic pseudo random test data / choices.
pub(crate) struct MinstdRand(u32);

impl MinstdRand {
    pub(crate) fn new() -> Self {
        Self(1)
    }

    pub(crate) fn next_u32(&mut self) -> u32 {
        self.0 = ((u64::from(self.0) * 48271) % 2_147_483_647) as u32;
        self.0
    }

    /// Uniform value in the inclusive range `[lo, hi]`, like
    /// `std::uniform_int_distribution`.
    pub(crate) fn uniform(&mut self, lo: usize, hi: usize) -> usize {
        debug_assert!(lo <= hi);
        lo + (self.next_u32() as usize) % (hi - lo + 1)
    }
}

/// `CreateRandomString(n)`: `n` deterministic pseudo random ASCII bytes.
pub(crate) fn create_random_string(n: usize) -> Vec<u8> {
    const DATA: &[u8] = b"abcdefghijklmnopqrstuvwxyz\
        ABCDEFGHIJKLMNOPQRSTUVWXYZ\
        0123456789~!@#$%^&*()_+=-<>?:\"{}[]|";
    let mut rnd = MinstdRand::new();
    (0..n).map(|_| DATA[rnd.uniform(0, DATA.len() - 1)]).collect()
}

/// `CreateFlatsFromString(data, chunk_size)`: chops `data` into flats of
/// `chunk_size` bytes (the last one possibly shorter).
///
/// # Safety
///
/// None beyond `chunk_size > 0` (asserted below) and each chunk's length
/// satisfying [`make_flat`]'s contract (always true here, chunks are
/// `<= chunk_size` and `chunk_size` is caller-controlled test data).
pub(crate) unsafe fn create_flats_from_string(data: &[u8], chunk_size: usize) -> Vec<*mut CordRep> {
    unsafe {
        assert!(chunk_size > 0);
        data.chunks(chunk_size).map(|s| make_flat(s)).collect()
    }
}

/// `CordRepBtreeFromFlats(flats)`: a tree built by appending all `flats` in
/// order. Adopts the flats.
///
/// # Safety
///
/// `flats` must be non-empty, and each entry a non-null pointer to a live
/// rep; the caller donates its references on all of them.
pub(crate) unsafe fn cord_rep_btree_from_flats(flats: &[*mut CordRep]) -> *mut CordRepBtree {
    unsafe {
        assert!(!flats.is_empty());
        let mut node = CordRepBtree::create(flats[0]);
        for &f in &flats[1..] {
            node = CordRepBtree::append(node, f);
        }
        node
    }
}

/// `CordVisitReps(rep, fn)`: visits `rep`, the children of any substring
/// chain, and (recursively) all edges of btree nodes.
///
/// # Safety
///
/// `rep` must be a non-null pointer to a live, well-formed rep tree; it is
/// borrowed, not consumed.
pub(crate) unsafe fn cord_visit_reps<F: FnMut(*mut CordRep)>(mut rep: *mut CordRep, f: &mut F) {
    unsafe {
        f(rep);
        while rep.tag() == SUBSTRING {
            rep = (*rep.cast::<CordRepSubstring>()).child;
            f(rep);
        }
        if rep.tag() == BTREE {
            for edge in as_btree(rep).edges() {
                cord_visit_reps(edge, f);
            }
        }
    }
}

/// `CordCollectRepsIf(predicate, rep)`: all reps visited by
/// [`cord_visit_reps`] for which `predicate` holds, in visiting order.
///
/// # Safety
///
/// Same contract as [`cord_visit_reps`]: `rep` must be a non-null pointer to
/// a live, well-formed rep tree, borrowed.
pub(crate) unsafe fn cord_collect_reps_if<P: FnMut(*mut CordRep) -> bool>(
    mut predicate: P,
    rep: *mut CordRep,
) -> Vec<*mut CordRep> {
    unsafe {
        let mut reps = Vec::new();
        cord_visit_reps(rep, &mut |r| {
            if predicate(r) {
                reps.push(r);
            }
        });
        reps
    }
}

/// Appends the bytes of `rep` to `s`.
///
/// # Safety
///
/// `rep` must be a non-null pointer to a live, well-formed rep tree; it is
/// borrowed, not consumed.
unsafe fn cord_to_string_into(mut rep: *mut CordRep, s: &mut Vec<u8>) {
    unsafe {
        let mut offset = 0;
        let length = rep.length();
        while rep.tag() == SUBSTRING {
            let sub = rep.cast::<CordRepSubstring>();
            offset += (*sub).start;
            rep = (*sub).child;
        }
        if rep.tag() == BTREE {
            for edge in as_btree(rep).edges() {
                cord_to_string_into(edge, s);
            }
        } else if rep.tag() >= FLAT {
            s.extend_from_slice(core::slice::from_raw_parts(flat::data(rep).add(offset), length));
        } else if rep.tag() == EXTERNAL {
            s.extend_from_slice(core::slice::from_raw_parts(
                (*rep.cast::<CordRepExternal>()).base.add(offset),
                length,
            ));
        } else {
            panic!("Unsupported tag {}", rep.tag());
        }
    }
}

/// `CordToString(rep)`: the bytes represented by `rep` (a btree, flat,
/// external or substring thereof).
///
/// # Safety
///
/// Same contract as [`cord_visit_reps`]: `rep` must be a non-null pointer to
/// a live, well-formed rep tree, borrowed.
pub(crate) unsafe fn cord_to_string(rep: *mut CordRep) -> Vec<u8> {
    unsafe {
        let mut s = Vec::with_capacity(rep.length());
        cord_to_string_into(rep, &mut s);
        s
    }
}

/// RAII helper to automatically unref reps on destruction (also during
/// unwinding, which keeps `#[should_panic]` tests leak free).
#[derive(Default)]
pub(crate) struct AutoUnref {
    unrefs: Vec<*mut CordRep>,
}

impl AutoUnref {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Adds `rep` to the list of reps to be unreffed at destruction.
    pub(crate) fn add<T: AsRepPtr>(&mut self, rep: T) -> T {
        self.unrefs.push(rep.as_rep_ptr());
        rep
    }

    /// Increments the reference count of `rep` by one, and adds it to the
    /// list of reps to be unreffed at destruction.
    ///
    /// # Safety
    ///
    /// `rep.as_rep_ptr()` must be a non-null pointer to a live rep; `rep` is
    /// borrowed (a new, independent reference is taken via `ref_rep`).
    pub(crate) unsafe fn add_ref<T: AsRepPtr>(&mut self, rep: T) -> T {
        unsafe {
            self.unrefs.push(ref_rep(rep.as_rep_ptr()));
            rep
        }
    }

    /// Increments the reference count of `rep` by one if `condition` is true,
    /// and adds it to the list of reps to be unreffed at destruction.
    ///
    /// # Safety
    ///
    /// Same contract as [`add_ref`](Self::add_ref).
    pub(crate) unsafe fn ref_if<T: AsRepPtr>(&mut self, condition: bool, rep: T) -> T {
        unsafe {
            if condition {
                self.unrefs.push(ref_rep(rep.as_rep_ptr()));
            }
            rep
        }
    }
}

impl Drop for AutoUnref {
    fn drop(&mut self) {
        for rep in self.unrefs.drain(..) {
            if !rep.is_null() {
                unsafe { unref(rep) };
            }
        }
    }
}

// --- Tree builders from cord_rep_btree_test.cc ------------------------------

/// `MakeHexFlat(i)`: a flat containing the hexadecimal value of `i` zero
/// padded to at least 4 digits and prefixed with "0x", e.g. "0x04ac".
///
/// # Safety
///
/// None: the formatted string is always well under `MAX_FLAT_LENGTH`.
pub(crate) unsafe fn make_hex_flat(i: usize) -> *mut CordRep {
    unsafe { make_flat(format!("0x{i:04x}").as_bytes()) }
}

/// `MakeLeaf(size)`: a leaf holding `size` hex flats (`size <= MAX_CAPACITY`).
///
/// # Safety
///
/// `size <= MAX_CAPACITY` (asserted below).
pub(crate) unsafe fn make_leaf(size: usize) -> *mut CordRepBtree {
    unsafe {
        assert!(size <= MAX_CAPACITY);
        let mut leaf = CordRepBtree::create(make_hex_flat(0));
        for i in 1..size {
            leaf = CordRepBtree::append(leaf, make_hex_flat(i));
        }
        leaf
    }
}

/// `MakeTree(size, append)`: a tree holding `max(size, 1)` hex flats, built by
/// appending (or prepending) them one at a time.
pub(crate) fn make_tree(size: usize, append: bool) -> *mut CordRepBtree {
    unsafe {
        let mut tree = CordRepBtree::create(make_hex_flat(0));
        for i in 1..size {
            tree = if append {
                CordRepBtree::append(tree, make_hex_flat(i))
            } else {
                CordRepBtree::prepend(tree, make_hex_flat(i))
            };
        }
        tree
    }
}

/// `CreateTree(reps)`: a tree built by appending all `reps` in order.
///
/// # Safety
///
/// Same contract as [`cord_rep_btree_from_flats`].
pub(crate) unsafe fn create_tree(reps: &[*mut CordRep]) -> *mut CordRepBtree {
    unsafe { cord_rep_btree_from_flats(reps) }
}

/// `CreateTree(data, chunk_size)`: a tree of flats of `chunk_size` bytes.
pub(crate) fn create_tree_from_string(data: &[u8], chunk_size: usize) -> *mut CordRepBtree {
    unsafe { create_tree(&create_flats_from_string(data, chunk_size)) }
}

/// `CreateTreeReverse(data, chunk_size)`: like [`create_tree_from_string`]
/// but built by prepending the flats in reverse order.
pub(crate) fn create_tree_reverse(data: &[u8], chunk_size: usize) -> *mut CordRepBtree {
    unsafe {
        let flats = create_flats_from_string(data, chunk_size);
        let mut rit = flats.iter().rev().copied();
        let mut tree = CordRepBtree::create(rit.next().unwrap());
        for f in rit {
            tree = CordRepBtree::prepend(tree, f);
        }
        tree
    }
}

/// `GetLeafEdges(tree, edges)`: recursively collects all leaf (data) edges of
/// `tree` in order, appending them to `edges`.
///
/// # Safety
///
/// `tree` must be a non-null pointer to a live, well-formed btree node; it
/// is borrowed, not consumed.
pub(crate) unsafe fn get_leaf_edges_into(tree: *mut CordRepBtree, edges: &mut Vec<*mut CordRep>) {
    unsafe {
        if tree.height() == 0 {
            edges.extend(tree.edges());
        } else {
            for edge in tree.edges() {
                get_leaf_edges_into(as_btree(edge), edges);
            }
        }
    }
}

/// `GetLeafEdges(tree)`: all leaf (data) edges of `tree` in order.
///
/// # Safety
///
/// Same contract as [`get_leaf_edges_into`].
pub(crate) unsafe fn get_leaf_edges(tree: *mut CordRepBtree) -> Vec<*mut CordRep> {
    unsafe {
        let mut edges = Vec::new();
        get_leaf_edges_into(tree, &mut edges);
        edges
    }
}
