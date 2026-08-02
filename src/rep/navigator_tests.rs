//! Port of abseil's `cord_rep_btree_navigator_test.cc`.

use super::btree::{BtreePtr, CordRepBtree, MAX_CAPACITY, MAX_HEIGHT, as_btree};
use super::navigator::CordRepBtreeNavigator;
use super::test_util::*;
use super::{BTREE, CordRep, RepPtr, ref_rep, unref};

const CHARS_PER_FLAT: usize = 3;

/// Parameter values of the C++ fixture: number of data edges in the tree.
const ALL_COUNTS: [usize; 7] = [
    1,
    MAX_CAPACITY - 1,
    MAX_CAPACITY,
    MAX_CAPACITY * MAX_CAPACITY - 1,
    MAX_CAPACITY * MAX_CAPACITY,
    MAX_CAPACITY * MAX_CAPACITY + 1,
    MAX_CAPACITY * MAX_CAPACITY * 2 + 17,
];

/// Under Miri only the smaller trees are exercised.
const COUNTS: &[usize] =
    if cfg!(miri) { &[1, MAX_CAPACITY, MAX_CAPACITY * MAX_CAPACITY + 1] } else { &ALL_COUNTS };

/// The C++ fixture: a tree of `count` flats of 3 chars, where flat 0 or 1 is
/// replaced by a substring to cover partial reads on substrings.
struct Fixture {
    data: Vec<u8>,
    flats: Vec<*mut CordRep>,
    tree: *mut CordRepBtree,
}

impl Fixture {
    fn new(count: usize) -> Self {
        unsafe {
            let data = create_random_string(count * CHARS_PER_FLAT);
            let mut flats = create_flats_from_string(&data, CHARS_PER_FLAT);
            if count > 1 {
                unref(flats[1]);
                flats[1] = make_substring(CHARS_PER_FLAT, CHARS_PER_FLAT, make_flat(&data));
            } else {
                unref(flats[0]);
                flats[0] = make_substring(0, CHARS_PER_FLAT, make_flat(&data));
            }
            let tree = cord_rep_btree_from_flats(&flats);
            Self { data, flats, tree }
        }
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        unsafe { unref(self.tree.as_rep()) };
    }
}

#[test]
fn uninitialized() {
    let nav = CordRepBtreeNavigator::new();
    assert!(!nav.is_some());
    assert!(nav.btree().is_null());
}

#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "assertion failed: self.height.is_some()")]
fn uninitialized_current_death() {
    let nav = CordRepBtreeNavigator::new();
    unsafe {
        let _ = nav.current();
    }
}

#[test]
fn init_first() {
    for &count in COUNTS {
        unsafe {
            let f = Fixture::new(count);
            let mut nav = CordRepBtreeNavigator::new();
            let edge = nav.init_first(f.tree);
            assert!(nav.is_some());
            assert_eq!(nav.btree(), f.tree);
            assert_eq!(nav.current(), f.flats[0]);
            assert_eq!(edge, f.flats[0]);
        }
    }
}

#[test]
fn init_last() {
    for &count in COUNTS {
        unsafe {
            let f = Fixture::new(count);
            let mut nav = CordRepBtreeNavigator::new();
            let edge = nav.init_last(f.tree);
            assert!(nav.is_some());
            assert_eq!(nav.btree(), f.tree);
            assert_eq!(nav.current(), *f.flats.last().unwrap());
            assert_eq!(edge, *f.flats.last().unwrap());
        }
    }
}

#[test]
fn next_prev() {
    for &count in COUNTS {
        unsafe {
            let f = Fixture::new(count);
            let flats = &f.flats;
            let mut nav = CordRepBtreeNavigator::new();
            nav.init_first(f.tree);

            assert!(nav.previous().is_null());
            assert_eq!(nav.current(), flats[0]);
            for &flat in &flats[1..] {
                assert_eq!(nav.next(), flat);
                assert_eq!(nav.current(), flat);
            }
            assert!(nav.next().is_null());
            assert_eq!(nav.current(), *flats.last().unwrap());
            for i in (1..flats.len()).rev() {
                assert_eq!(nav.previous(), flats[i - 1]);
                assert_eq!(nav.current(), flats[i - 1]);
            }
            assert!(nav.previous().is_null());
            assert_eq!(nav.current(), flats[0]);
        }
    }
}

#[test]
fn prev_next() {
    for &count in COUNTS {
        unsafe {
            let f = Fixture::new(count);
            let flats = &f.flats;
            let mut nav = CordRepBtreeNavigator::new();
            nav.init_last(f.tree);

            assert!(nav.next().is_null());
            assert_eq!(nav.current(), *flats.last().unwrap());
            for i in (1..flats.len()).rev() {
                assert_eq!(nav.previous(), flats[i - 1]);
                assert_eq!(nav.current(), flats[i - 1]);
            }
            assert!(nav.previous().is_null());
            assert_eq!(nav.current(), flats[0]);
            for &flat in &flats[1..] {
                assert_eq!(nav.next(), flat);
                assert_eq!(nav.current(), flat);
            }
            assert!(nav.next().is_null());
            assert_eq!(nav.current(), *flats.last().unwrap());
        }
    }
}

#[test]
fn reset() {
    unsafe {
        let tree = CordRepBtree::create(make_flat(b"abc"));
        let mut nav = CordRepBtreeNavigator::new();
        nav.init_first(tree);
        nav.reset();
        assert!(!nav.is_some());
        assert!(nav.btree().is_null());
        unref(tree.as_rep());
    }
}

#[test]
fn skip() {
    for &count in COUNTS {
        unsafe {
            let f = Fixture::new(count);
            let flats = &f.flats;
            let mut nav = CordRepBtreeNavigator::new();
            nav.init_first(f.tree);

            for char_offset in 0..CHARS_PER_FLAT {
                let pos = nav.skip(char_offset);
                assert_eq!(pos.edge, nav.current());
                assert_eq!(pos.edge, flats[0]);
                assert_eq!(pos.offset, char_offset);
            }

            for index1 in 0..count {
                for index2 in index1..count {
                    for char_offset in 0..CHARS_PER_FLAT {
                        let mut nav = CordRepBtreeNavigator::new();
                        nav.init_first(f.tree);

                        let length1 = index1 * CHARS_PER_FLAT;
                        let pos1 = nav.skip(length1 + char_offset);
                        assert_eq!(pos1.edge, flats[index1]);
                        assert_eq!(pos1.edge, nav.current());
                        assert_eq!(pos1.offset, char_offset);

                        let length2 = index2 * CHARS_PER_FLAT;
                        let pos2 = nav.skip(length2 - length1 + char_offset);
                        assert_eq!(pos2.edge, flats[index2]);
                        assert_eq!(pos2.edge, nav.current());
                        assert_eq!(pos2.offset, char_offset);
                    }
                }
            }
        }
    }
}

#[test]
fn seek() {
    for &count in COUNTS {
        unsafe {
            let f = Fixture::new(count);
            let flats = &f.flats;
            let mut nav = CordRepBtreeNavigator::new();
            nav.init_first(f.tree);

            for char_offset in 0..CHARS_PER_FLAT {
                let pos = nav.seek(char_offset);
                assert_eq!(pos.edge, nav.current());
                assert_eq!(pos.edge, flats[0]);
                assert_eq!(pos.offset, char_offset);
            }

            for (index, &flat) in flats.iter().enumerate().take(count) {
                for char_offset in 0..CHARS_PER_FLAT {
                    let offset = index * CHARS_PER_FLAT + char_offset;
                    let pos1 = nav.seek(offset);
                    assert_eq!(pos1.edge, flat);
                    assert_eq!(pos1.edge, nav.current());
                    assert_eq!(pos1.offset, char_offset);
                }
            }
        }
    }
}

#[test]
fn init_offset() {
    // Whitebox: init_offset is implemented in terms of seek, which is
    // exhaustively tested. Only test that it initializes / forwards.
    unsafe {
        let mut tree = CordRepBtree::create(make_flat(b"abc"));
        tree = CordRepBtree::append(tree, make_flat(b"def"));
        let mut nav = CordRepBtreeNavigator::new();
        let pos = nav.init_offset(tree, 5);
        assert!(nav.is_some());
        assert_eq!(nav.btree(), tree);
        assert_eq!(pos.edge, tree.edge(1));
        assert_eq!(pos.edge, nav.current());
        assert_eq!(pos.offset, 2);
        unref(tree.as_rep());
    }
}

#[test]
fn init_offset_and_seek_beyond_length() {
    unsafe {
        let tree1 = CordRepBtree::create(make_flat(b"abc"));
        let tree2 = CordRepBtree::create(make_flat(b"def"));

        let mut nav = CordRepBtreeNavigator::new();
        nav.init_first(tree1);
        assert!(nav.seek(3).edge.is_null());
        assert!(nav.seek(100).edge.is_null());
        assert_eq!(nav.btree(), tree1);
        assert_eq!(nav.current(), tree1.edge(0));

        assert!(nav.init_offset(tree2, 3).edge.is_null());
        assert!(nav.init_offset(tree2, 100).edge.is_null());
        assert_eq!(nav.btree(), tree1);
        assert_eq!(nav.current(), tree1.edge(0));

        unref(tree1.as_rep());
        unref(tree2.as_rep());
    }
}

#[test]
fn read() {
    for &count in COUNTS {
        unsafe {
            let f = Fixture::new(count);
            let (flats, data) = (&f.flats, &f.data);
            for offset in 0..data.len() {
                for length in 1..=(data.len() - offset) {
                    let mut nav = CordRepBtreeNavigator::new();
                    nav.init_first(f.tree);

                    // Skip towards the edge holding offset.
                    let edge_offset = nav.skip(offset).offset;

                    // Read node.
                    let result = nav.read(edge_offset, length);
                    assert!(!result.tree.is_null());
                    assert_eq!(result.tree.length(), length);
                    if result.tree.tag() == BTREE {
                        assert!(CordRepBtree::is_valid(as_btree(result.tree), false));
                    }

                    // Verify contents.
                    assert_eq!(cord_to_string(result.tree), &data[offset..offset + length]);

                    // Verify 'partial last edge' reads.
                    let partial = (offset + length) % CHARS_PER_FLAT;
                    assert_eq!(result.n, partial);

                    // Verify ending position if not EOF.
                    if offset + length < data.len() {
                        let index = (offset + length) / CHARS_PER_FLAT;
                        assert_eq!(nav.current(), flats[index]);
                    }

                    unref(result.tree);
                }
            }
        }
    }
}

#[test]
fn read_beyond_length_of_tree() {
    for &count in COUNTS {
        unsafe {
            let f = Fixture::new(count);
            let mut nav = CordRepBtreeNavigator::new();
            nav.init_first(f.tree);
            let result = nav.read(2, f.tree.length());
            assert!(result.tree.is_null());
        }
    }
}

#[test]
fn navigate_maximum_tree_depth() {
    unsafe {
        let flat1 = make_flat(b"Hello world");
        let flat2 = make_flat(b"World Hello");

        let mut node = CordRepBtree::create(flat1);
        node = CordRepBtree::append(node, flat2);
        while node.height() < MAX_HEIGHT {
            node = CordRepBtree::new_with(node.as_rep());
        }

        let mut nav = CordRepBtreeNavigator::new();
        let edge = nav.init_first(node);
        assert_eq!(edge, flat1);
        assert_eq!(nav.next(), flat2);
        assert!(nav.next().is_null());
        assert_eq!(nav.previous(), flat1);
        assert!(nav.previous().is_null());

        unref(node.as_rep());
    }
}

#[test]
fn navigator_is_copy() {
    unsafe {
        let tree = CordRepBtree::create(make_flat(b"abc"));
        let mut nav = CordRepBtreeNavigator::new();
        nav.init_first(tree);
        let copy = nav;
        assert_eq!(copy.current(), nav.current());
        let _ = ref_rep(tree.as_rep());
        unref(tree.as_rep());
        unref(tree.as_rep());
    }
}
