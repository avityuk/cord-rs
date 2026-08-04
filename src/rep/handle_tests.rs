//! Tests for the typed handles in `rep.rs`/`rep/flat.rs`/`rep/external.rs`/
//! `rep/btree.rs`: `RepRef`, `OwnedRep`, `RepView`, `FlatRef`, `ExternalRef`
//! and `BtreeRef`.

use super::btree::{BtreePtr, CordRepBtree};
use super::external::EXTERNAL_REP_SIZE;
use super::test_util::*;
use super::{CordRep, OwnedRep, RepPtr, RepRef, RepView, flat, unref};

const VALUE: &[u8] = b"Lorem ipsum dolor sit amet, consectetur adipiscing elit.";

#[test]
fn rep_ref_basic_accessors_on_flat() {
    unsafe {
        let raw = make_flat(VALUE);
        let rep = RepRef::from_raw(raw);
        assert_eq!(rep.len(), VALUE.len());
        assert_eq!(rep.tag(), raw.tag());
        assert!(rep.is_flat());
        assert!(!rep.is_btree());
        assert!(!rep.is_external());
        assert!(!rep.is_substring());
        assert!(rep.is_data_edge());
        assert!(rep.ref_is_one());
        assert_eq!(rep.ref_get(), 1);
        assert_eq!(rep.as_ptr(), raw);
        assert_eq!(rep.data(), VALUE);
        unref(raw);
    }
}

#[test]
fn rep_ref_basic_accessors_on_external() {
    unsafe {
        let raw = make_external(VALUE);
        let rep = RepRef::from_raw(raw);
        assert!(rep.is_external());
        assert!(!rep.is_flat());
        assert!(rep.is_data_edge());
        assert_eq!(rep.data(), VALUE);
        unref(raw);
    }
}

#[test]
fn rep_ref_basic_accessors_on_substring() {
    unsafe {
        let raw = make_flat(VALUE);
        let substr = make_substring(1, 20, raw);
        let rep = RepRef::from_raw(substr);
        assert!(rep.is_substring());
        assert!(rep.is_data_edge());
        assert_eq!(rep.data(), &VALUE[1..21]);
        unref(substr);
    }
}

#[test]
fn rep_ref_basic_accessors_on_btree() {
    unsafe {
        let raw = make_flat(VALUE);
        let tree = CordRepBtree::new_with(raw);
        let rep = RepRef::from_raw(tree.as_rep());
        assert!(rep.is_btree());
        assert!(!rep.is_data_edge());
        unref(tree.as_rep());
    }
}

#[test]
fn rep_ref_ref_is_one_reflects_sharing() {
    unsafe {
        let mut refs = AutoUnref::new();
        let raw = refs.add(make_flat(VALUE));
        let rep = RepRef::from_raw(raw);
        assert!(rep.ref_is_one());
        refs.add_ref(raw);
        assert!(!rep.ref_is_one());
        assert_eq!(rep.ref_get(), 2);
    }
}

#[test]
fn view_returns_flat() {
    unsafe {
        let raw = make_flat(VALUE);
        match RepRef::from_raw(raw).view() {
            RepView::Flat(f) => {
                assert_eq!(f.data(), VALUE);
                assert_eq!(f.len(), VALUE.len());
            }
            _ => panic!("expected RepView::Flat"),
        }
        unref(raw);
    }
}

#[test]
fn view_returns_external() {
    unsafe {
        let raw = make_external(VALUE);
        match RepRef::from_raw(raw).view() {
            RepView::External(e) => assert_eq!(e.data(), VALUE),
            _ => panic!("expected RepView::External"),
        }
        unref(raw);
    }
}

#[test]
fn view_returns_substring() {
    unsafe {
        let raw = make_flat(VALUE);
        let substr = make_substring(1, 20, raw);
        match RepRef::from_raw(substr).view() {
            RepView::Substring { start, child } => {
                assert_eq!(start, 1);
                assert_eq!(child.data(), VALUE);
            }
            _ => panic!("expected RepView::Substring"),
        }
        unref(substr);
    }
}

#[test]
fn view_returns_btree() {
    unsafe {
        let raw = make_flat(VALUE);
        let tree = CordRepBtree::new_with(raw);
        match RepRef::from_raw(tree.as_rep()).view() {
            RepView::Btree(b) => assert_eq!(b.height(), 0),
            _ => panic!("expected RepView::Btree"),
        }
        unref(tree.as_rep());
    }
}

#[test]
fn owned_rep_drop_decrements_and_clone_increments() {
    unsafe {
        let raw = make_flat(VALUE);
        let owned = OwnedRep::from_raw(raw);
        assert_eq!(owned.as_ref().ref_get(), 1);

        let cloned = owned.clone();
        assert_eq!(owned.as_ref().ref_get(), 2);
        assert_eq!(cloned.as_ref().ref_get(), 2);
        assert_eq!(owned.len(), VALUE.len());

        drop(cloned);
        assert_eq!(owned.as_ref().ref_get(), 1);
        // `owned` drops at the end of this scope, releasing the last
        // reference; Miri verifies there's no leak or double free.
    }
}

#[test]
fn owned_rep_into_raw_transfers_the_reference() {
    unsafe {
        let raw = make_flat(VALUE);
        let owned = OwnedRep::from_raw(raw);
        let back = owned.into_raw();
        assert_eq!(back, raw);
        // `into_raw` must not have unreffed: the reference is still live.
        assert_eq!(RepRef::from_raw(back).ref_get(), 1);
        unref(back);
    }
}

#[test]
fn rep_ref_to_owned_takes_a_fresh_reference() {
    unsafe {
        let mut refs = AutoUnref::new();
        let raw = refs.add(make_flat(VALUE));
        let rep = RepRef::from_raw(raw);
        assert!(rep.ref_is_one());

        let owned = rep.to_owned();
        assert_eq!(rep.ref_get(), 2);
        assert_eq!(owned.len(), VALUE.len());

        drop(owned);
        assert!(rep.ref_is_one());
    }
}

#[test]
fn flat_ref_accessors() {
    unsafe {
        let raw = make_flat(VALUE);
        let RepView::Flat(f) = RepRef::from_raw(raw).view() else { unreachable!() };
        assert_eq!(f.len(), VALUE.len());
        assert_eq!(f.capacity(), flat::capacity(raw));
        assert_eq!(f.allocated_size(), flat::allocated_size(raw));
        assert_eq!(f.data(), VALUE);
        assert_eq!(f.as_ptr(), raw);
        unref(raw);
    }
}

#[test]
fn external_ref_accessors() {
    unsafe {
        let raw = make_external(VALUE);
        let RepView::External(e) = RepRef::from_raw(raw).view() else { unreachable!() };
        assert_eq!(e.len(), VALUE.len());
        assert_eq!(e.data(), VALUE);
        assert_eq!(e.allocated_size(), VALUE.len() + EXTERNAL_REP_SIZE);
        assert_eq!(e.as_ptr().cast::<CordRep>(), raw);
        unref(raw);
    }
}

#[test]
fn btree_ref_accessors_and_edges() {
    unsafe {
        let flats: Vec<_> = (0..3).map(|i| make_hex_flat(i)).collect();
        let tree = cord_rep_btree_from_flats(&flats);
        let RepView::Btree(b) = RepRef::from_raw(tree.as_rep()).view() else { unreachable!() };
        assert_eq!(b.height(), 0);
        assert_eq!(b.len(), tree.length());
        assert_eq!(b.as_ptr(), tree);
        assert!(b.as_rep_ref().is_btree());

        let collected: Vec<Vec<u8>> = b.edges().map(|e| e.data().to_vec()).collect();
        let expected: Vec<Vec<u8>> = (0..3).map(|i| format!("0x{i:04x}").into_bytes()).collect();
        assert_eq!(collected, expected);

        unref(tree.as_rep());
    }
}
