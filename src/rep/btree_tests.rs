//! Port of abseil's `cord_rep_btree_test.cc`.
//!
//! Test names mirror the C++ tests in `snake_case`. Parametrized C++ fixtures
//! (`shared` / `first_shared, second_shared` / `height`) are ported as loops
//! over all parameter values inside one test.
#![allow(clippy::cast_possible_truncation, reason = "tests pack small known values into bytes")]

use core::ptr::NonNull;

use alloc::collections::VecDeque;
use alloc::format;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

use super::btree::{BtreePtr, CordRepBtree, ExtractResult, MAX_CAPACITY, MAX_HEIGHT, as_btree};
use super::test_util::*;
use super::{
    BTREE, CordRep, CordRepSubstring, FLAT, RepPtr, SUBSTRING, edge_data, flat, is_data_edge, ref_rep, unref,
};

// --- Matchers ---------------------------------------------------------------

/// `IsNode(height)`: a valid btree node of the given height.
///
/// # Safety
///
/// `rep` must be null or a pointer to a live rep; borrowed, not consumed.
unsafe fn is_node(rep: *mut CordRep, height: usize) -> bool {
    unsafe {
        if rep.is_null() || rep.tag() != BTREE {
            return false;
        }
        let tree = as_btree(rep);
        if let Err(e) = CordRepBtree::check_valid(tree, false) {
            // `eprintln!` is diagnostic-only test output with no `alloc`
            // equivalent, so it's only available with the `std` feature;
            // the failure itself is still reported via the `false` return.
            #[cfg(feature = "std")]
            {
                let mut dump = String::new();
                let _ = CordRepBtree::dump(NonNull::new(rep), "Expected valid NODE, got:", false, &mut dump);
                std::eprintln!("{e}\n{dump}");
            }
            #[cfg(not(feature = "std"))]
            let _ = e;
            return false;
        }
        tree.height() == height
    }
}

/// `IsSubstring(start, length)`.
///
/// # Safety
///
/// `rep` must be null or a pointer to a live rep; borrowed, not consumed.
unsafe fn is_substring(rep: *mut CordRep, start: usize, length: usize) -> bool {
    unsafe {
        if rep.is_null() || rep.tag() != SUBSTRING {
            return false;
        }
        let sub: *mut CordRepSubstring = rep.cast();
        (*sub).start == start && rep.length() == length
    }
}

fn eq_extract_result(result: ExtractResult, tree: *mut CordRep, rep: *mut CordRep) -> bool {
    result.tree == NonNull::new(tree) && result.extracted == NonNull::new(rep)
}

/// # Safety
///
/// `tree` must be a non-null pointer to a live, well-formed btree node;
/// borrowed, not consumed.
unsafe fn edges_of(tree: *mut CordRepBtree) -> Vec<*mut CordRep> {
    unsafe { tree.edges().collect() }
}

/// `DataConsumer`: consumes string fragments forwards or backwards.
struct DataConsumer<'a> {
    data: &'a [u8],
    consumed: usize,
    forward: bool,
}

impl<'a> DataConsumer<'a> {
    fn new(data: &'a [u8], forward: bool) -> Self {
        Self { data, consumed: 0, forward }
    }

    fn next(&mut self, n: usize) -> &'a [u8] {
        assert!(n <= self.data.len() - self.consumed);
        self.consumed += n;
        let start = if self.forward { self.consumed - n } else { self.data.len() - self.consumed };
        &self.data[start..start + n]
    }

    fn consumed(&self) -> &'a [u8] {
        if self.forward { &self.data[..self.consumed] } else { &self.data[self.data.len() - self.consumed..] }
    }
}

/// # Safety
///
/// `node` must be a non-null pointer to a live, well-formed btree node; the
/// caller donates its reference, consumed by this call and transferred to
/// the returned tree.
unsafe fn btree_add(node: *mut CordRepBtree, append: bool, data: &[u8]) -> *mut CordRepBtree {
    unsafe {
        if append {
            CordRepBtree::append_data(node, data, 0)
        } else {
            CordRepBtree::prepend_data(node, data, 0)
        }
    }
}

const SHARED: [bool; 2] = [false, true];
const DUAL: [(bool, bool); 4] = [(false, false), (false, true), (true, false), (true, true)];

// --- Tests ------------------------------------------------------------------

#[test]
fn size_is_multiple_of_64() {
    if core::mem::size_of::<usize>() == 8 {
        assert_eq!(core::mem::size_of::<CordRepBtree>() % 64, 0, "Should be multiple of 64");
    }
}

#[test]
fn new_destroy_empty_tree() {
    unsafe {
        let tree = CordRepBtree::new_node(0);
        assert_eq!(tree.size(), 0);
        assert_eq!(tree.height(), 0);
        assert!(edges_of(tree).is_empty());
        CordRepBtree::destroy(tree);
    }
}

#[test]
fn new_destroy_empty_tree_at_height() {
    unsafe {
        let tree = CordRepBtree::new_node(3);
        assert_eq!(tree.size(), 0);
        assert_eq!(tree.height(), 3);
        assert!(edges_of(tree).is_empty());
        CordRepBtree::destroy(tree);
    }
}

#[test]
fn btree() {
    unsafe {
        let rep = CordRepBtree::new_node(0).as_rep();
        assert_eq!(as_btree(rep).as_rep(), rep);
        unref(rep);
    }
}

#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "assertion failed: rep.is_btree()")]
fn btree_death_on_flat() {
    unsafe {
        let mut refs = AutoUnref::new();
        let rep = refs.add(make_flat(b"Hello world"));
        let _ = as_btree(rep);
    }
}

#[test]
fn edge_data_test() {
    unsafe {
        let f = make_flat(b"Hello world");
        let external = make_external(b"Hello external");
        let substr1 = make_substring(1, 6, ref_rep(f));
        let substr2 = make_substring(1, 6, ref_rep(external));
        let bad_substr = make_substring(1, 2, ref_rep(substr1));

        assert!(is_data_edge(f));
        assert_eq!(edge_data(f).as_ptr(), flat::data(f).cast_const());
        assert_eq!(edge_data(f), b"Hello world");

        assert!(is_data_edge(external));
        assert_eq!(edge_data(external).as_ptr(), (*external.cast::<super::external::CordRepExternal>()).base);
        assert_eq!(edge_data(external), b"Hello external");

        assert!(is_data_edge(substr1));
        assert_eq!(edge_data(substr1).as_ptr(), flat::data(f).add(1).cast_const());
        assert_eq!(edge_data(substr1), b"ello w");

        assert!(is_data_edge(substr2));
        assert_eq!(
            edge_data(substr2).as_ptr(),
            (*external.cast::<super::external::CordRepExternal>()).base.add(1)
        );
        assert_eq!(edge_data(substr2), b"ello e");

        assert!(!is_data_edge(bad_substr));

        unref(bad_substr);
        unref(substr2);
        unref(substr1);
        unref(external);
        unref(f);
    }
}

#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "assertion failed: is_data_edge(edge)")]
fn edge_data_death_on_bad_substr() {
    unsafe {
        let mut refs = AutoUnref::new();
        let f = refs.add(make_flat(b"Hello world"));
        let substr1 = refs.add(make_substring(1, 6, ref_rep(f)));
        let bad_substr = refs.add(make_substring(1, 2, ref_rep(substr1)));
        let _ = edge_data(bad_substr);
    }
}

#[test]
fn create_unref_leaf() {
    unsafe {
        let f = make_flat(b"a");
        let leaf = CordRepBtree::create(f);
        assert_eq!(leaf.size(), 1);
        assert_eq!(leaf.height(), 0);
        assert_eq!(edges_of(leaf), vec![f]);
        unref(leaf.as_rep());
    }
}

#[test]
fn new_unref_node() {
    unsafe {
        let leaf = CordRepBtree::create(make_flat(b"a"));
        let tree = CordRepBtree::new_with(leaf.as_rep());
        assert_eq!(tree.size(), 1);
        assert_eq!(tree.height(), 1);
        assert_eq!(edges_of(tree), vec![leaf.as_rep()]);
        unref(tree.as_rep());
    }
}

#[test]
fn append_to_leaf_to_capacity() {
    for shared in SHARED {
        unsafe {
            let mut refs = AutoUnref::new();
            let mut flats = vec![make_hex_flat(0)];
            let mut leaf = CordRepBtree::create(flats[0]);
            for i in 1..MAX_CAPACITY {
                refs.ref_if(shared, leaf);
                flats.push(make_hex_flat(i));
                let result = CordRepBtree::append(leaf, flats[i]);
                assert_eq!(result.height(), 0);
                assert_eq!(result != leaf, shared, "shared = {shared}");
                assert_eq!(edges_of(result), flats);
                leaf = result;
            }
            unref(leaf.as_rep());
        }
    }
}

#[test]
fn prepend_to_leaf_to_capacity() {
    for shared in SHARED {
        unsafe {
            let mut refs = AutoUnref::new();
            let mut flats = VecDeque::new();
            flats.push_front(make_hex_flat(0));
            let mut leaf = CordRepBtree::create(flats[0]);
            for i in 1..MAX_CAPACITY {
                refs.ref_if(shared, leaf);
                flats.push_front(make_hex_flat(i));
                let result = CordRepBtree::prepend(leaf, flats[0]);
                assert_eq!(result.height(), 0);
                assert_eq!(result != leaf, shared, "shared = {shared}");
                assert_eq!(edges_of(result), Vec::from(flats.clone()));
                leaf = result;
            }
            unref(leaf.as_rep());
        }
    }
}

/// Exercises the code aligning data at the front or back of `edges`:
/// alternating append / prepend moves `begin()` / `end()` as needed.
#[test]
fn append_prepend_to_leaf_to_capacity() {
    for shared in SHARED {
        unsafe {
            let mut refs = AutoUnref::new();
            let mut flats = VecDeque::new();
            flats.push_front(make_hex_flat(0));
            let mut leaf = CordRepBtree::create(flats[0]);
            for i in 1..MAX_CAPACITY {
                refs.ref_if(shared, leaf);
                let result = if i % 2 != 0 {
                    flats.push_front(make_hex_flat(i));
                    CordRepBtree::prepend(leaf, flats[0])
                } else {
                    flats.push_back(make_hex_flat(i));
                    CordRepBtree::append(leaf, *flats.back().unwrap())
                };
                assert_eq!(result.height(), 0);
                assert_eq!(result != leaf, shared, "shared = {shared}");
                assert_eq!(edges_of(result), Vec::from(flats.clone()));
                leaf = result;
            }
            unref(leaf.as_rep());
        }
    }
}

#[test]
fn append_to_leaf_beyond_capacity() {
    for shared in SHARED {
        unsafe {
            let mut refs = AutoUnref::new();
            let leaf = make_leaf(MAX_CAPACITY);
            refs.ref_if(shared, leaf);
            let f = make_flat(b"abc");
            let result = CordRepBtree::append(leaf, f);
            assert!(is_node(result.as_rep(), 1));
            assert_ne!(result, leaf);
            let edges = edges_of(result);
            assert_eq!(edges.len(), 2);
            assert_eq!(edges[0], leaf.as_rep());
            assert!(is_node(edges[1], 0));
            assert_eq!(edges_of(as_btree(edges[1])), vec![f]);
            unref(result.as_rep());
        }
    }
}

#[test]
fn prepend_to_leaf_beyond_capacity() {
    for shared in SHARED {
        unsafe {
            let mut refs = AutoUnref::new();
            let leaf = make_leaf(MAX_CAPACITY);
            refs.ref_if(shared, leaf);
            let f = make_flat(b"abc");
            let result = CordRepBtree::prepend(leaf, f);
            assert!(is_node(result.as_rep(), 1));
            assert_ne!(result, leaf);
            let edges = edges_of(result);
            assert_eq!(edges.len(), 2);
            assert!(is_node(edges[0], 0));
            assert_eq!(edges[1], leaf.as_rep());
            assert_eq!(edges_of(as_btree(edges[0])), vec![f]);
            unref(result.as_rep());
        }
    }
}

#[test]
fn append_to_tree_one_deep() {
    let max_cap = MAX_CAPACITY;
    for shared in SHARED {
        unsafe {
            let mut refs = AutoUnref::new();
            let mut flats = vec![make_hex_flat(0)];
            let mut tree = CordRepBtree::create(flats[0]);
            for i in 1..=max_cap {
                flats.push(make_hex_flat(i));
                tree = CordRepBtree::append(tree, flats[i]);
            }
            assert!(is_node(tree.as_rep(), 1));

            for i in (max_cap + 1)..(max_cap * max_cap) {
                // Ref the top level tree based on the param, and the leaf
                // node once every 4 iterations (only effect: leaf copied).
                refs.ref_if(shared, tree);
                refs.ref_if(i % 4 == 0, *edges_of(tree).last().unwrap());

                flats.push(make_hex_flat(i));
                let result = CordRepBtree::append(tree, flats[i]);
                assert!(is_node(result.as_rep(), 1));
                assert_eq!(result != tree, shared, "shared = {shared}");
                assert_eq!(get_leaf_edges(result), flats);
                tree = result;
            }
            unref(tree.as_rep());
        }
    }
}

#[test]
fn append_to_tree_two_deep() {
    let max_cap = MAX_CAPACITY;
    for shared in SHARED {
        unsafe {
            let mut refs = AutoUnref::new();
            let mut flats = vec![make_hex_flat(0)];
            let mut tree = CordRepBtree::create(flats[0]);
            for i in 1..=(max_cap * max_cap) {
                flats.push(make_hex_flat(i));
                tree = CordRepBtree::append(tree, flats[i]);
            }
            assert!(is_node(tree.as_rep(), 2));
            for i in (max_cap * max_cap + 1)..(max_cap * max_cap * max_cap) {
                refs.ref_if(shared, tree);
                let back = *edges_of(tree).last().unwrap();
                refs.ref_if(i % 16 == 0, back);
                refs.ref_if(i % 4 == 0, *edges_of(as_btree(back)).last().unwrap());

                flats.push(make_hex_flat(i));
                let result = CordRepBtree::append(tree, flats[i]);
                assert!(is_node(result.as_rep(), 2));
                assert_eq!(result != tree, shared, "shared = {shared}");
                assert_eq!(get_leaf_edges(result), flats);
                tree = result;
            }
            unref(tree.as_rep());
        }
    }
}

#[test]
fn prepend_to_tree_one_deep() {
    let max_cap = MAX_CAPACITY;
    for shared in SHARED {
        unsafe {
            let mut refs = AutoUnref::new();
            let mut flats = VecDeque::new();
            flats.push_back(make_hex_flat(0));
            let mut tree = CordRepBtree::create(flats[0]);
            for i in 1..=max_cap {
                flats.push_front(make_hex_flat(i));
                tree = CordRepBtree::prepend(tree, flats[0]);
            }
            assert!(is_node(tree.as_rep(), 1));

            for i in (max_cap + 1)..(max_cap * max_cap) {
                refs.ref_if(shared, tree);
                refs.ref_if(i % 4 == 0, *edges_of(tree).last().unwrap());

                flats.push_front(make_hex_flat(i));
                let result = CordRepBtree::prepend(tree, flats[0]);
                assert!(is_node(result.as_rep(), 1));
                assert_eq!(result != tree, shared, "shared = {shared}");
                assert_eq!(get_leaf_edges(result), Vec::from(flats.clone()));
                tree = result;
            }
            unref(tree.as_rep());
        }
    }
}

#[test]
fn prepend_to_tree_two_deep() {
    let max_cap = MAX_CAPACITY;
    for shared in SHARED {
        unsafe {
            let mut refs = AutoUnref::new();
            let mut flats = VecDeque::new();
            flats.push_back(make_hex_flat(0));
            let mut tree = CordRepBtree::create(flats[0]);
            for i in 1..=(max_cap * max_cap) {
                flats.push_front(make_hex_flat(i));
                tree = CordRepBtree::prepend(tree, flats[0]);
            }
            assert!(is_node(tree.as_rep(), 2));
            for i in (max_cap * max_cap + 1)..(max_cap * max_cap * max_cap) {
                refs.ref_if(shared, tree);
                let back = *edges_of(tree).last().unwrap();
                refs.ref_if(i % 16 == 0, back);
                refs.ref_if(i % 4 == 0, *edges_of(as_btree(back)).last().unwrap());

                flats.push_front(make_hex_flat(i));
                let result = CordRepBtree::prepend(tree, flats[0]);
                assert!(is_node(result.as_rep(), 2));
                assert_eq!(result != tree, shared, "shared = {shared}");
                assert_eq!(get_leaf_edges(result), Vec::from(flats.clone()));
                tree = result;
            }
            unref(tree.as_rep());
        }
    }
}

#[test]
fn merge_leafs_not_exceeding_capacity() {
    for (first_shared, second_shared) in DUAL {
        for use_append in [false, true] {
            unsafe {
                let mut refs = AutoUnref::new();
                let mut flats = Vec::new();
                let left = make_leaf(3);
                get_leaf_edges_into(left, &mut flats);
                refs.ref_if(first_shared, left);
                let right = make_leaf(2);
                get_leaf_edges_into(right, &mut flats);
                refs.ref_if(second_shared, right);

                let tree = if use_append {
                    CordRepBtree::append(left, right.as_rep())
                } else {
                    CordRepBtree::prepend(right, left.as_rep())
                };
                assert!(is_node(tree.as_rep(), 0));
                assert_eq!(
                    edges_of(tree),
                    flats,
                    "append = {use_append}, shared = {first_shared}/{second_shared}"
                );
                unref(tree.as_rep());
            }
        }
    }
}

#[test]
fn merge_leafs_exceeding_capacity() {
    for (first_shared, second_shared) in DUAL {
        for use_append in [false, true] {
            unsafe {
                let mut refs = AutoUnref::new();
                let left = make_leaf(MAX_CAPACITY - 2);
                refs.ref_if(first_shared, left);
                let right = make_leaf(MAX_CAPACITY - 1);
                refs.ref_if(second_shared, right);

                let tree = if use_append {
                    CordRepBtree::append(left, right.as_rep())
                } else {
                    CordRepBtree::prepend(right, left.as_rep())
                };
                assert!(is_node(tree.as_rep(), 1));
                assert_eq!(edges_of(tree), vec![left.as_rep(), right.as_rep()]);
                unref(tree.as_rep());
            }
        }
    }
}

#[test]
fn merge_equal_height_trees() {
    for (first_shared, second_shared) in DUAL {
        for use_append in [false, true] {
            unsafe {
                let mut refs = AutoUnref::new();
                let mut flats = Vec::new();
                let left = make_tree(MAX_CAPACITY * 3, true);
                get_leaf_edges_into(left, &mut flats);
                refs.ref_if(first_shared, left);
                let right = make_tree(MAX_CAPACITY * 2, true);
                get_leaf_edges_into(right, &mut flats);
                refs.ref_if(second_shared, right);

                let tree = if use_append {
                    CordRepBtree::append(left, right.as_rep())
                } else {
                    CordRepBtree::prepend(right, left.as_rep())
                };
                assert!(is_node(tree.as_rep(), 1));
                assert_eq!(tree.size(), 5);
                assert_eq!(get_leaf_edges(tree), flats);
                unref(tree.as_rep());
            }
        }
    }
}

#[test]
fn merge_leaf_with_tree_not_exceeding_leaf_capacity() {
    for (first_shared, second_shared) in DUAL {
        for use_append in [false, true] {
            unsafe {
                let mut refs = AutoUnref::new();
                let mut flats = Vec::new();
                let left = make_tree(MAX_CAPACITY * 2 + 2, true);
                get_leaf_edges_into(left, &mut flats);
                refs.ref_if(first_shared, left);
                let right = make_tree(3, true);
                get_leaf_edges_into(right, &mut flats);
                refs.ref_if(second_shared, right);

                let tree = if use_append {
                    CordRepBtree::append(left, right.as_rep())
                } else {
                    CordRepBtree::prepend(right, left.as_rep())
                };
                assert!(is_node(tree.as_rep(), 1));
                assert_eq!(tree.size(), 3);
                assert_eq!(get_leaf_edges(tree), flats);
                unref(tree.as_rep());
            }
        }
    }
}

#[test]
fn merge_leaf_with_tree_exceeding_leaf_capacity() {
    for (first_shared, second_shared) in DUAL {
        for use_append in [false, true] {
            unsafe {
                let mut refs = AutoUnref::new();
                let mut flats = Vec::new();
                let left = make_tree(MAX_CAPACITY * 3 - 2, true);
                get_leaf_edges_into(left, &mut flats);
                refs.ref_if(first_shared, left);
                let right = make_tree(3, true);
                get_leaf_edges_into(right, &mut flats);
                refs.ref_if(second_shared, right);

                let tree = if use_append {
                    CordRepBtree::append(left, right.as_rep())
                } else {
                    CordRepBtree::prepend(right, left.as_rep())
                };
                assert!(is_node(tree.as_rep(), 1));
                assert_eq!(tree.size(), 4);
                assert_eq!(get_leaf_edges(tree), flats);
                unref(tree.as_rep());
            }
        }
    }
}

/// # Safety
///
/// `tree` must be a non-null pointer to a live, well-formed btree node with
/// at least `depth` levels below it; borrowed, not consumed.
unsafe fn ref_edges_at(depth: usize, refs: &mut AutoUnref, tree: *mut CordRepBtree) {
    unsafe {
        let edges = edges_of(tree);
        if depth == 0 {
            refs.add_ref(edges[0]);
            refs.add_ref(*edges.last().unwrap());
        } else {
            assert!(tree.height() > 0);
            ref_edges_at(depth - 1, refs, as_btree(edges[0]));
            ref_edges_at(depth - 1, refs, as_btree(*edges.last().unwrap()));
        }
    }
}

#[test]
fn merge_fuzz_test() {
    let max_cap = MAX_CAPACITY;
    let mut rnd = MinstdRand::new();
    let iterations = if cfg!(miri) {
        40
    } else if cfg!(debug_assertions) {
        3000
    } else {
        10000
    };
    for _ in 0..iterations {
        unsafe {
            let random_leaf_count = |rnd: &mut MinstdRand| {
                let height = rnd.uniform(0, 3);
                let leaf = rnd.uniform(0, max_cap - 1);
                (if height > 0 { max_cap.pow(height as u32) } else { 0 }) + leaf
            };
            let mut refs = AutoUnref::new();
            let mut flats = Vec::new();

            let count = random_leaf_count(&mut rnd);
            let append = rnd.uniform(0, 1) == 1;
            let left = make_tree(count, append);
            get_leaf_edges_into(left, &mut flats);
            if rnd.uniform(1, 6) == 1 {
                let depth = rnd.uniform(0, left.height());
                ref_edges_at(depth, &mut refs, left);
            }

            let count = random_leaf_count(&mut rnd);
            let append = rnd.uniform(0, 1) == 1;
            let right = make_tree(count, append);
            get_leaf_edges_into(right, &mut flats);
            if rnd.uniform(1, 6) == 1 {
                let depth = rnd.uniform(0, right.height());
                ref_edges_at(depth, &mut refs, right);
            }

            let tree = CordRepBtree::append(left, right.as_rep());
            assert_eq!(get_leaf_edges(tree), flats);
            unref(tree.as_rep());
        }
    }
}

#[test]
fn remove_suffix() {
    let max_cap = MAX_CAPACITY;
    for shared in SHARED {
        for cap in [max_cap - 1, max_cap * 2, max_cap * max_cap * 2] {
            let data = create_random_string(cap * 512);
            unsafe {
                {
                    // RemoveSuffix(<all>)
                    let mut refs = AutoUnref::new();
                    let node = refs.ref_if(shared, create_tree_from_string(&data, 512));
                    assert!(CordRepBtree::remove_suffix(node, data.len()).is_null());

                    // RemoveSuffix(<none>)
                    let node = refs.ref_if(shared, create_tree_from_string(&data, 512));
                    assert_eq!(CordRepBtree::remove_suffix(node, 0), node.as_rep());
                    unref(node.as_rep());
                }

                let step = if cfg!(miri) {
                    397
                } else if cfg!(debug_assertions) && cap > max_cap * 2 {
                    7
                } else {
                    1
                };
                for n in (1..data.len()).step_by(step) {
                    let mut refs = AutoUnref::new();
                    let flats = create_flats_from_string(&data, 512);
                    let node = refs.ref_if(shared, create_tree(&flats));
                    let rep = refs.add(CordRepBtree::remove_suffix(node, n));
                    assert_eq!(cord_to_string(rep), &data[..data.len() - n]);

                    // Collect all flats.
                    let mut edges = cord_collect_reps_if(|r| r.tag() >= FLAT, rep);
                    assert!(edges.len() <= flats.len());

                    // Isolate the last edge.
                    let last_edge = edges.pop().unwrap();
                    let last_length = rep.length() - edges.len() * 512;

                    // All flats except the last edge must be kept or copied as is.
                    let mut index = 0;
                    for edge in edges {
                        assert_eq!(edge, flats[index]);
                        index += 1;
                        assert_eq!(edge.length(), 512);
                    }

                    // Small substrings may be optimized to avoid waste, so
                    // only check sharing where the code always does this.
                    if last_length >= 500 {
                        assert_eq!(last_edge, flats[index]);
                        if shared {
                            assert_eq!(last_edge.length(), 512);
                        } else {
                            assert!(last_edge.ref_is_one());
                            assert_eq!(last_edge.length(), last_length);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn sub_tree() {
    let max_cap = MAX_CAPACITY;
    let n = max_cap * max_cap * 2;
    let data = create_random_string(n * 3);
    unsafe {
        let flats: Vec<*mut CordRep> = data.chunks(3).map(|s| make_flat(s)).collect();
        let mut node = CordRepBtree::create(ref_rep(flats[0]));
        for &f in &flats[1..] {
            node = CordRepBtree::append(node, ref_rep(f));
        }
        let step = if cfg!(miri) { 23 } else { 1 };
        for offset in (0..data.len()).step_by(step) {
            for length in (1..=(data.len() - offset)).step_by(step) {
                let rep = CordRepBtree::sub_tree(node, offset, length);
                assert_eq!(
                    cord_to_string(rep),
                    &data[offset..offset + length],
                    "offset {offset} length {length}"
                );
                unref(rep);
            }
        }
        unref(node.as_rep());
        for f in flats {
            unref(f);
        }
    }
}

/// A `sub_tree` call on a pre-existing (large) substring adjusts the
/// existing substring if not shared, and else rewrites it.
#[test]
fn sub_tree_on_existing_substring() {
    unsafe {
        let data = create_random_string(1000);
        let mut leaf = CordRepBtree::create(make_flat(b"abc"));
        let f = make_flat(&data);
        leaf = CordRepBtree::append(leaf, f);

        // Setup tree containing substring.
        let result = CordRepBtree::sub_tree(leaf, 0, 3 + 990);
        assert_eq!(result.tag(), BTREE);
        unref(leaf.as_rep());
        leaf = as_btree(result);
        let edges = edges_of(leaf);
        assert_eq!(edges.len(), 2);
        assert!(is_substring(edges[1], 0, 990));
        assert_eq!((*edges[1].cast::<CordRepSubstring>()).child, f);

        // Verify substring of substring.
        let result = CordRepBtree::sub_tree(leaf, 3 + 5, 970);
        assert!(is_substring(result, 5, 970));
        assert_eq!((*result.cast::<CordRepSubstring>()).child, f);
        unref(result);

        unref(leaf.as_rep());
    }
}

#[test]
fn add_data_to_leaf() {
    let n = MAX_CAPACITY;
    let data = create_random_string(n * 3);
    for shared in SHARED {
        for append in [true, false] {
            unsafe {
                let mut refs = AutoUnref::new();
                let mut consumer = DataConsumer::new(&data, append);
                let mut leaf = CordRepBtree::create(make_flat(consumer.next(3)));
                for _ in 1..n {
                    refs.ref_if(shared, leaf);
                    let result = btree_add(leaf, append, consumer.next(3));
                    assert_eq!(result != leaf, shared, "append = {append}, shared = {shared}");
                    assert_eq!(cord_to_string(result.as_rep()), consumer.consumed());
                    leaf = result;
                }
                unref(leaf.as_rep());
            }
        }
    }
}

#[test]
fn append_data_to_tree() {
    for shared in SHARED {
        unsafe {
            let mut refs = AutoUnref::new();
            let n = MAX_CAPACITY + MAX_CAPACITY / 2;
            let data = create_random_string(n * 3);
            let tree = refs.ref_if(shared, create_tree_from_string(&data, 3));
            let edges = edges_of(tree);
            let (leaf0, leaf1) = (edges[0], edges[1]);
            let result = CordRepBtree::append_data(tree, b"123456789", 0);
            assert_eq!(result != tree, shared, "shared = {shared}");
            let result_edges = edges_of(result);
            assert_eq!(result_edges.len(), 2);
            assert_eq!(result_edges[0], leaf0);
            assert_eq!(result_edges[1] != leaf1, shared);
            let mut expected = data.clone();
            expected.extend_from_slice(b"123456789");
            assert_eq!(cord_to_string(result.as_rep()), expected);
            unref(result.as_rep());
        }
    }
}

#[test]
fn prepend_data_to_tree() {
    for shared in SHARED {
        unsafe {
            let mut refs = AutoUnref::new();
            let n = MAX_CAPACITY + MAX_CAPACITY / 2;
            let data = create_random_string(n * 3);
            let tree = refs.ref_if(shared, create_tree_reverse(&data, 3));
            let edges = edges_of(tree);
            let (leaf0, leaf1) = (edges[0], edges[1]);
            let result = CordRepBtree::prepend_data(tree, b"123456789", 0);
            assert_eq!(result != tree, shared, "shared = {shared}");
            let result_edges = edges_of(result);
            assert_eq!(result_edges.len(), 2);
            assert_eq!(result_edges[0] != leaf0, shared);
            assert_eq!(result_edges[1], leaf1);
            let mut expected = b"123456789".to_vec();
            expected.extend_from_slice(&data);
            assert_eq!(cord_to_string(result.as_rep()), expected);
            unref(result.as_rep());
        }
    }
}

#[test]
fn add_data_to_tree_three_levels_deep() {
    let max_cap = MAX_CAPACITY;
    let two_deep = max_cap * max_cap;
    // Miri interprets every operation; the full climb to `max_cap^3` items
    // (~650 tree operations across the parameter matrix) is minutes, not a
    // check. Keep the same shape — leaf fill, climb to height 1, fill height
    // 1 to its max, climb to height 2 — but stop shortly after reaching
    // height 2 instead of also filling it all the way to its cubic capacity,
    // so Miri still sees the deepest-surgery code path.
    let n = if cfg!(miri) { two_deep + max_cap } else { max_cap * max_cap * max_cap };
    let data = create_random_string(n * 3);
    for shared in SHARED {
        for append in [true, false] {
            unsafe {
                let mut refs = AutoUnref::new();
                let mut consumer = DataConsumer::new(&data, append);

                // Fill leaf.
                let mut tree = CordRepBtree::create(make_flat(consumer.next(3)));
                for _ in 1..max_cap {
                    tree = btree_add(tree, append, consumer.next(3));
                }
                assert_eq!(cord_to_string(tree.as_rep()), consumer.consumed());

                // Fill to maximum at one deep.
                refs.ref_if(shared, tree);
                let mut result = btree_add(tree, append, consumer.next(3));
                assert!(is_node(result.as_rep(), 1));
                assert_ne!(result, tree);
                assert_eq!(cord_to_string(result.as_rep()), consumer.consumed());
                tree = result;
                for _ in (max_cap + 1)..two_deep {
                    refs.ref_if(shared, tree);
                    result = btree_add(tree, append, consumer.next(3));
                    assert_eq!(result != tree, shared);
                    assert_eq!(cord_to_string(result.as_rep()), consumer.consumed());
                    tree = result;
                }

                // Fill to maximum at two deep.
                refs.ref_if(shared, tree);
                result = btree_add(tree, append, consumer.next(3));
                assert!(is_node(result.as_rep(), 2));
                assert_ne!(result, tree);
                assert_eq!(cord_to_string(result.as_rep()), consumer.consumed());
                tree = result;
                for _ in (two_deep + 1)..n {
                    refs.ref_if(shared, tree);
                    result = btree_add(tree, append, consumer.next(3));
                    assert_eq!(result != tree, shared);
                    assert_eq!(cord_to_string(result.as_rep()), consumer.consumed());
                    tree = result;
                }
                unref(tree.as_rep());
            }
        }
    }
}

#[test]
fn add_large_data_to_leaf() {
    let max_cap = MAX_CAPACITY;
    let n = if cfg!(miri) { max_cap + 2 } else { max_cap * max_cap * max_cap * 3 + 2 };
    let data = create_random_string(n * flat::MAX_FLAT_LENGTH);
    for shared in SHARED {
        for append in [true, false] {
            unsafe {
                let mut refs = AutoUnref::new();
                let leaf = CordRepBtree::create(make_flat(b"abc"));
                refs.ref_if(shared, leaf);
                let result = btree_add(leaf, append, &data);
                let expected =
                    if append { [b"abc".as_slice(), &data].concat() } else { [&data[..], b"abc"].concat() };
                assert_eq!(cord_to_string(result.as_rep()), expected);
                unref(result.as_rep());
            }
        }
    }
}

#[test]
fn create_from_tree_returns_tree() {
    for shared in SHARED {
        unsafe {
            let mut refs = AutoUnref::new();
            let leaf = CordRepBtree::create(make_flat(b"Hello world"));
            refs.ref_if(shared, leaf);
            let result = CordRepBtree::create(leaf.as_rep());
            assert_eq!(result, leaf);
            unref(result.as_rep());
        }
    }
}

#[test]
fn get_character() {
    let n = MAX_CAPACITY * MAX_CAPACITY + 2;
    let mut data = create_random_string(n * 3);
    unsafe {
        let mut tree = create_tree_from_string(&data, 3);
        // Add a substring node for good measure.
        tree = CordRepBtree::append(tree, make_substring(4, 5, make_flat(b"abcdefghijklm")));
        data.extend_from_slice(b"efghi");
        for (i, &b) in data.iter().enumerate() {
            assert_eq!(CordRepBtree::get_byte(tree, i), b, "index {i}");
        }
        unref(tree.as_rep());
    }
}

#[test]
fn is_flat_single_flat() {
    unsafe {
        let leaf = CordRepBtree::create(make_flat(b"Hello world"));
        assert_eq!(CordRepBtree::as_flat(leaf), Some(&b"Hello world"[..]));
        assert_eq!(CordRepBtree::as_flat_range(leaf, 0, 11), Some(&b"Hello world"[..]));
        // Arbitrary ranges must check true as well.
        assert_eq!(CordRepBtree::as_flat_range(leaf, 1, 4), Some(&b"ello"[..]));
        assert_eq!(CordRepBtree::as_flat_range(leaf, 6, 5), Some(&b"world"[..]));
        unref(leaf.as_rep());
    }
}

#[test]
fn is_flat_multi_flat() {
    let n = MAX_CAPACITY * MAX_CAPACITY + 2;
    let mut data = create_random_string(n * 3);
    unsafe {
        let mut tree = create_tree_from_string(&data, 3);
        // Add substring nodes for good measure.
        tree = CordRepBtree::append(tree, make_substring(4, 3, make_flat(b"abcdefghijklm")));
        tree = CordRepBtree::append(tree, make_substring(8, 3, make_flat(b"abcdefghijklm")));
        data.extend_from_slice(b"efgijk");

        assert!(CordRepBtree::as_flat(tree).is_none());
        for offset in (0..data.len()).step_by(3) {
            assert_eq!(CordRepBtree::as_flat_range(tree, offset, 3), Some(&data[offset..offset + 3]));
            if offset > 0 {
                assert!(CordRepBtree::as_flat_range(tree, offset - 1, 4).is_none());
            }
            if offset < data.len() - 4 {
                assert!(CordRepBtree::as_flat_range(tree, offset, 4).is_none());
            }
        }
        unref(tree.as_rep());
    }
}

#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "assertion failed: this.as_rep().ref_is_one()")]
fn get_append_buffer_not_private() {
    unsafe {
        let mut refs = AutoUnref::new();
        let tree = refs.add(CordRepBtree::create(make_external(b"Foo")));
        refs.add_ref(tree);
        let _ = CordRepBtree::get_append_buffer(tree, 1);
    }
}

#[test]
fn get_append_buffer_not_flat() {
    for height in 0..MAX_HEIGHT {
        unsafe {
            let mut tree = CordRepBtree::create(make_external(b"Foo"));
            for _ in 1..=height {
                tree = CordRepBtree::new_with(tree.as_rep());
            }
            assert!(CordRepBtree::get_append_buffer(tree, 1).is_none(), "height {height}");
            unref(tree.as_rep());
        }
    }
}

#[test]
fn get_append_buffer_flat_not_private() {
    for height in 0..MAX_HEIGHT {
        unsafe {
            let f = make_flat(b"abc");
            let mut tree = CordRepBtree::create(ref_rep(f));
            for _ in 1..=height {
                tree = CordRepBtree::new_with(tree.as_rep());
            }
            assert!(CordRepBtree::get_append_buffer(tree, 1).is_none(), "height {height}");
            unref(tree.as_rep());
            unref(f);
        }
    }
}

#[test]
fn get_append_buffer_tree_not_private() {
    for height in 1..MAX_HEIGHT {
        unsafe {
            let mut refs = AutoUnref::new();
            let f = make_flat(b"abc");
            let mut tree = CordRepBtree::create(ref_rep(f));
            for i in 1..=height {
                if i == height.div_ceil(2) {
                    refs.add_ref(tree);
                }
                tree = CordRepBtree::new_with(tree.as_rep());
            }
            assert!(CordRepBtree::get_append_buffer(tree, 1).is_none(), "height {height}");
            unref(tree.as_rep());
            unref(f);
        }
    }
}

#[test]
fn get_append_buffer_flat_no_capacity() {
    for height in 0..MAX_HEIGHT {
        unsafe {
            let f = make_flat(b"abc");
            f.set_length(flat::capacity(f));
            let mut tree = CordRepBtree::create(f);
            for _ in 1..=height {
                tree = CordRepBtree::new_with(tree.as_rep());
            }
            assert!(CordRepBtree::get_append_buffer(tree, 1).is_none(), "height {height}");
            unref(tree.as_rep());
        }
    }
}

#[test]
fn get_append_buffer_flat_with_capacity() {
    for height in 0..MAX_HEIGHT {
        unsafe {
            let f = make_flat(b"abc");
            let mut tree = CordRepBtree::create(f);
            for _ in 1..=height {
                tree = CordRepBtree::new_with(tree.as_rep());
            }
            let (ptr, len) = CordRepBtree::get_append_buffer(tree, 2).expect("span");
            assert_eq!(len, 2, "height {height}");
            assert_eq!(ptr, flat::data(f).add(3));
            assert_eq!(tree.length(), 5);

            let avail = flat::capacity(f) - 5;
            let (ptr, len) = CordRepBtree::get_append_buffer(tree, avail + 100).expect("span");
            assert_eq!(len, avail);
            assert_eq!(ptr, flat::data(f).add(5));
            assert_eq!(tree.length(), 5 + avail);
            unref(tree.as_rep());
        }
    }
}

#[test]
fn dump() {
    unsafe {
        // Handles null.
        let mut s = String::new();
        CordRepBtree::dump(None, "", false, &mut s).unwrap();
        CordRepBtree::dump(None, "Once upon a label", false, &mut s).unwrap();
        CordRepBtree::dump(None, "Once upon a label", true, &mut s).unwrap();
        assert!(s.contains("NULL"));

        // Cover legal edges.
        let f = make_flat(b"Hello world");
        let external = make_external(b"Hello external");
        let substr_flat = make_substring(1, 6, ref_rep(f));
        let substr_external = make_substring(2, 7, ref_rep(external));

        // Build tree.
        let mut tree = CordRepBtree::create(f);
        tree = CordRepBtree::append(tree, external);
        tree = CordRepBtree::append(tree, substr_flat);
        tree = CordRepBtree::append(tree, substr_external);

        // Repeat until we have a tree.
        while tree.height() == 0 {
            tree = CordRepBtree::append(tree, ref_rep(f));
            tree = CordRepBtree::append(tree, ref_rep(external));
            tree = CordRepBtree::append(tree, ref_rep(substr_flat));
            tree = CordRepBtree::append(tree, ref_rep(substr_external));
        }

        for api in 0..=2 {
            let mut s = String::new();
            match api {
                0 => CordRepBtree::dump(NonNull::new(tree.as_rep()), "", false, &mut s).unwrap(),
                1 => CordRepBtree::dump(NonNull::new(tree.as_rep()), "Once upon a label", false, &mut s)
                    .unwrap(),
                _ => CordRepBtree::dump(NonNull::new(tree.as_rep()), "Once upon a label", true, &mut s)
                    .unwrap(),
            }
            // Contains Node(depth) / Leaf and private / shared indicators.
            for needle in ["Node(1)", "Leaf", "Private", "Shared"] {
                assert!(s.contains(needle), "api {api}: missing {needle}:\n{s}");
            }
            // Contains length and start offset of all data edges.
            for needle in ["len = 11", "len = 14", "len = 6", "len = 7", "start = 1", "start = 2"] {
                assert!(s.contains(needle), "api {api}: missing {needle}:\n{s}");
            }
            // Contains address of all data edges.
            for rep in [f, external, substr_flat, substr_external] {
                assert!(s.contains(&format!("{rep:p}")), "api {api}: missing address of {rep:p}:\n{s}");
            }
            if api != 0 {
                assert!(s.contains("Once upon a label"));
            }
            let contents = [
                "data = \"Hello world\"",
                "data = \"Hello external\"",
                "data = \"ello w\"",
                "data = \"llo ext\"",
            ];
            if api == 2 {
                for needle in contents {
                    assert!(s.contains(needle), "api {api}: missing {needle}:\n{s}");
                }
            } else {
                for needle in contents {
                    assert!(!s.contains(needle), "api {api}: unexpected {needle}");
                }
            }
        }
        unref(tree.as_rep());
    }
}

#[test]
fn is_valid() {
    unsafe {
        assert!(!CordRepBtree::is_valid(core::ptr::null(), false));

        let empty = CordRepBtree::new_node(0);
        assert!(CordRepBtree::is_valid(empty, false));
        unref(empty.as_rep());

        for as_tree in [false, true] {
            let leaf = CordRepBtree::create(make_flat(b"abc"));
            let tree = if as_tree { CordRepBtree::new_with(leaf.as_rep()) } else { core::ptr::null_mut() };
            let check = if as_tree { tree } else { leaf };

            assert!(CordRepBtree::is_valid(check, false));
            (*leaf).rep.length -= 1;
            assert!(!CordRepBtree::is_valid(check, false));
            (*leaf).rep.length += 1;

            assert!(CordRepBtree::is_valid(check, false));
            (*leaf).rep.tag -= 1;
            assert!(!CordRepBtree::is_valid(check, false));
            (*leaf).rep.tag += 1;

            // Height.
            assert!(CordRepBtree::is_valid(check, false));
            (*leaf).rep.storage[0] = (MAX_HEIGHT + 1) as u8;
            assert!(!CordRepBtree::is_valid(check, false));
            (*leaf).rep.storage[0] = 1;
            assert!(!CordRepBtree::is_valid(check, false));
            (*leaf).rep.storage[0] = 0;

            // Begin.
            assert!(CordRepBtree::is_valid(check, false));
            let begin = (*leaf).rep.storage[1];
            (*leaf).rep.storage[1] = MAX_CAPACITY as u8;
            assert!(!CordRepBtree::is_valid(check, false));
            (*leaf).rep.storage[1] = 2;
            assert!(!CordRepBtree::is_valid(check, false));
            (*leaf).rep.storage[1] = begin;

            // End.
            assert!(CordRepBtree::is_valid(check, false));
            let end = (*leaf).rep.storage[2];
            (*leaf).rep.storage[2] = (MAX_CAPACITY + 1) as u8;
            assert!(!CordRepBtree::is_valid(check, false));
            (*leaf).rep.storage[2] = end;

            // Data edge tag and value.
            assert!(CordRepBtree::is_valid(check, false));
            let edge = edges_of(leaf)[0];
            let tag = edge.tag();
            (*leaf).edges[begin as usize] = core::ptr::null_mut();
            assert!(!CordRepBtree::is_valid(check, false));
            (*leaf).edges[begin as usize] = edge;
            (*edge).tag = BTREE;
            assert!(!CordRepBtree::is_valid(check, false));
            (*edge).tag = tag;

            if as_tree {
                assert!(CordRepBtree::is_valid(check, false));
                (*leaf).rep.length -= 1;
                assert!(!CordRepBtree::is_valid(check, false));
                (*leaf).rep.length += 1;

                // Height.
                assert!(CordRepBtree::is_valid(check, false));
                (*tree).rep.storage[0] = 2;
                assert!(!CordRepBtree::is_valid(check, false));
                (*tree).rep.storage[0] = 1;

                // Btree edge.
                assert!(CordRepBtree::is_valid(check, false));
                let edge = edges_of(tree)[0];
                let tag = edge.tag();
                (*edge).tag = FLAT;
                assert!(!CordRepBtree::is_valid(check, false));
                (*edge).tag = tag;
            }

            assert!(CordRepBtree::is_valid(check, false));
            unref(check.as_rep());
        }
    }
}

#[test]
fn assert_valid() {
    unsafe {
        let tree = CordRepBtree::create(make_flat(b"abc"));
        assert_eq!(CordRepBtree::assert_valid(tree, true), tree);
        assert_eq!(CordRepBtree::assert_valid(tree, false), tree);
        unref(tree.as_rep());
    }
}

#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "check_valid() FAILED")]
fn assert_valid_death_on_null() {
    unsafe {
        let _ = CordRepBtree::assert_valid(core::ptr::null_mut(), true);
    }
}

#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "check_valid() FAILED")]
fn assert_valid_death_on_bad_length() {
    struct Restore(*mut CordRepBtree);
    impl Drop for Restore {
        fn drop(&mut self) {
            unsafe { (*self.0).rep.length += 1 };
        }
    }
    unsafe {
        let mut refs = AutoUnref::new();
        let tree = refs.add(CordRepBtree::create(make_flat(b"abc")));
        (*tree).rep.length -= 1;
        let _restore = Restore(tree);
        let _ = CordRepBtree::assert_valid(tree, true);
    }
}

#[test]
fn check_assert_valid_shallow_vs_deep() {
    // This used to also flip the process-wide `EXHAUSTIVE_VALIDATION` flag
    // (via `set_exhaustive_validation`) to check that it forces even a
    // `shallow = true` check to go deep, guarded by a lock serializing it
    // against other tests that do the same. But that flag is a raw
    // `AtomicBool` with no scoping of its own, and nothing else that reads
    // it (in particular the library's own internal, debug-only
    // `assert_valid(_, true)` calls made while *any* other test mutates a
    // btree) takes that lock — so toggling it, even briefly, could make an
    // unrelated concurrently-running test's shallow validation go deep
    // mid-mutation and spuriously fail. `check_valid(_, shallow = false)`
    // already *is* the deep path unconditionally (see its doc comment), so
    // the shallow-vs-deep distinction below is fully covered by the
    // `shallow` parameter alone, without ever touching the global flag.
    unsafe {
        // Create a tree of at least 2 levels, and mess with the original flat,
        // which should go undetected in shallow mode as the flat is too far
        // away, but must be detected by a non-shallow check.
        let f = make_flat(b"abc");
        let mut tree = CordRepBtree::create(f);
        let n = MAX_CAPACITY * MAX_CAPACITY * 2;
        for _ in 0..n {
            tree = CordRepBtree::append(tree, make_flat(b"Hello world"));
        }
        f.set_length(100);

        assert!(!CordRepBtree::is_valid(tree, false));
        assert!(CordRepBtree::is_valid(tree, true));
        CordRepBtree::assert_valid(tree, true);
        assert!(CordRepBtree::check_valid(tree, false).is_err());

        f.set_length(3);
        unref(tree.as_rep());
    }
}

#[test]
fn rebuild() {
    let sizes: &[usize] = if cfg!(miri) {
        &[3, 8, 100, 1500]
    } else if cfg!(debug_assertions) {
        &[3, 8, 100, 10000, 200_000]
    } else {
        &[3, 8, 100, 10000, 1_000_000]
    };
    for shared in SHARED {
        for &size in sizes {
            unsafe {
                let mut flats = Vec::with_capacity(size);
                for _ in 0..size {
                    let f = flat::new(2);
                    *flat::data(f) = b'x';
                    f.set_length(1);
                    flats.push(f);
                }

                // Build the tree into `right`, and every so many `split_limit`
                // edges combine `left` + `right` into a new `left` and start a
                // new `right`. This guarantees a reasonable amount of chaos.
                let mut split_count = 0;
                let mut split_limit = 3;
                let mut it = flats.iter().copied();
                let mut left: *mut CordRepBtree = core::ptr::null_mut();
                let mut right = CordRepBtree::new_with(it.next().unwrap());
                for f in it {
                    split_count += 1;
                    if split_count >= split_limit {
                        split_limit += split_limit / 16;
                        left =
                            if left.is_null() { right } else { CordRepBtree::append(left, right.as_rep()) };
                        right = CordRepBtree::new_with(f);
                    } else {
                        right = CordRepBtree::append(right, f);
                    }
                }
                // Finalize tree.
                left = if left.is_null() { right } else { CordRepBtree::append(left, right.as_rep()) };

                // Rebuild.
                let mut refs = AutoUnref::new();
                let input = refs.ref_if(shared, left);
                let left = refs.add(CordRepBtree::rebuild(input));
                assert!(CordRepBtree::is_valid(left, false));

                // Verify we have the exact same edges in the exact same order.
                let mut ok = true;
                let mut index = 0;
                cord_visit_reps(left.as_rep(), &mut |edge| {
                    if edge.tag() < FLAT {
                        return;
                    }
                    ok = ok && index < flats.len() && flats[index] == edge;
                    index += 1;
                });
                assert!(ok && index == flats.len(), "Rebuild edges mismatch (size {size}, shared {shared})");
            }
        }
    }
}

/// # Safety
///
/// Same contract as [`CordRepBtree::extract_append_buffer`]: `input` must be
/// a non-null pointer to a live, well-formed btree node; the caller donates
/// its reference, consumed by this call.
unsafe fn extract_last(input: *mut CordRepBtree, cap: usize) -> ExtractResult {
    unsafe { CordRepBtree::extract_append_buffer(input, cap) }
}

#[test]
fn extract_append_buffer_leaf_single_flat() {
    unsafe {
        let f = make_flat(b"Abc");
        let leaf = CordRepBtree::create(f);
        assert!(eq_extract_result(extract_last(leaf, 1), core::ptr::null_mut(), f));
        unref(f);
    }
}

#[test]
fn extract_append_buffer_node_single_flat() {
    unsafe {
        let f = make_flat(b"Abc");
        let leaf = CordRepBtree::create(f);
        let node = CordRepBtree::new_with(leaf.as_rep());
        assert!(eq_extract_result(extract_last(node, 1), core::ptr::null_mut(), f));
        unref(f);
    }
}

#[test]
fn extract_append_buffer_leaf_two_flats() {
    unsafe {
        let flats = create_flats_from_string(b"abcdef", 3);
        let leaf = create_tree(&flats);
        assert!(eq_extract_result(extract_last(leaf, 1), flats[0], flats[1]));
        unref(flats[0]);
        unref(flats[1]);
    }
}

#[test]
fn extract_append_buffer_node_two_flats() {
    unsafe {
        let flats = create_flats_from_string(b"abcdef", 3);
        let leaf = create_tree(&flats);
        let node = CordRepBtree::new_with(leaf.as_rep());
        assert!(eq_extract_result(extract_last(node, 1), flats[0], flats[1]));
        unref(flats[0]);
        unref(flats[1]);
    }
}

#[test]
fn extract_append_buffer_node_two_flats_in_two_leafs() {
    unsafe {
        let flats = create_flats_from_string(b"abcdef", 3);
        let leaf1 = CordRepBtree::create(flats[0]);
        let leaf2 = CordRepBtree::create(flats[1]);
        let node = CordRepBtree::new_pair(leaf1, leaf2);
        assert!(eq_extract_result(extract_last(node, 1), flats[0], flats[1]));
        unref(flats[0]);
        unref(flats[1]);
    }
}

#[test]
fn extract_append_buffer_leaf_three_flats() {
    unsafe {
        let flats = create_flats_from_string(b"abcdefghi", 3);
        let leaf = create_tree(&flats);
        assert!(eq_extract_result(extract_last(leaf, 1), leaf.as_rep(), flats[2]));
        unref(flats[2]);
        unref(leaf.as_rep());
    }
}

#[test]
fn extract_append_buffer_node_three_flats_right_no_folding() {
    unsafe {
        let f = make_flat(b"Abc");
        let flats = create_flats_from_string(b"defghi", 3);
        let leaf1 = CordRepBtree::create(f);
        let leaf2 = create_tree(&flats);
        let node = CordRepBtree::new_pair(leaf1, leaf2);
        assert!(eq_extract_result(extract_last(node, 1), node.as_rep(), flats[1]));
        assert_eq!(edges_of(node), vec![leaf1.as_rep(), leaf2.as_rep()]);
        assert_eq!(edges_of(leaf1), vec![f]);
        assert_eq!(edges_of(leaf2), vec![flats[0]]);
        unref(node.as_rep());
        unref(flats[1]);
    }
}

#[test]
fn extract_append_buffer_node_three_flats_right_leaf_folding() {
    unsafe {
        let f = make_flat(b"Abc");
        let flats = create_flats_from_string(b"defghi", 3);
        let leaf1 = create_tree(&flats);
        let leaf2 = CordRepBtree::create(f);
        let node = CordRepBtree::new_pair(leaf1, leaf2);
        assert!(eq_extract_result(extract_last(node, 1), leaf1.as_rep(), f));
        assert_eq!(edges_of(leaf1), flats);
        unref(leaf1.as_rep());
        unref(f);
    }
}

#[test]
fn extract_append_buffer_no_capacity() {
    unsafe {
        let flats = create_flats_from_string(b"abcdef", 3);
        let leaf = create_tree(&flats);
        let avail = flat::capacity(flats[1]) - flats[1].length();
        assert!(eq_extract_result(extract_last(leaf, avail + 1), leaf.as_rep(), core::ptr::null_mut()));
        assert!(eq_extract_result(extract_last(leaf, avail), flats[0], flats[1]));
        unref(flats[0]);
        unref(flats[1]);
    }
}

#[test]
fn extract_append_buffer_not_flat() {
    unsafe {
        let flats = create_flats_from_string(b"abcdef", 3);
        let substr = make_substring(1, 2, flats[1]);
        let leaf = create_tree(&[flats[0], substr]);
        assert!(eq_extract_result(extract_last(leaf, 1), leaf.as_rep(), core::ptr::null_mut()));
        unref(leaf.as_rep());
    }
}

#[test]
fn extract_append_buffer_shared() {
    unsafe {
        let flats = create_flats_from_string(b"abcdef", 3);
        let leaf = create_tree(&flats);

        ref_rep(flats[1]);
        assert!(eq_extract_result(extract_last(leaf, 1), leaf.as_rep(), core::ptr::null_mut()));
        unref(flats[1]);

        ref_rep(leaf.as_rep());
        assert!(eq_extract_result(extract_last(leaf, 1), leaf.as_rep(), core::ptr::null_mut()));
        unref(leaf.as_rep());

        let node = CordRepBtree::new_with(leaf.as_rep());
        ref_rep(node.as_rep());
        assert!(eq_extract_result(extract_last(node, 1), node.as_rep(), core::ptr::null_mut()));
        unref(node.as_rep());

        unref(node.as_rep());
    }
}
