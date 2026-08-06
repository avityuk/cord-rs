//! Tests for the typed handles in `rep.rs`/`rep/flat.rs`/`rep/external.rs`/
//! `rep/btree.rs`: `RepRef`, `OwnedRep`, `RepView`, `FlatRef`, `ExternalRef`
//! and `BtreeRef`.

use crate::inline_data::InlineData;

use super::btree::{BtreePtr, CordRepBtree};
use super::external::EXTERNAL_REP_SIZE;
use super::test_util::*;
use super::{CordRep, OwnedRep, RepPtr, RepRef, RepView, flat, ref_rep, unref};

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

// --- UniqueRep -------------------------------------------------------------
//
// `UniqueRep` is the crate's central mutation witness (see its soundness
// note in `rep.rs`), so two of its three legitimate `&mut`-borrow
// constructors (`OwnedRep::try_unique`, `InlineData::tree_unique`; the
// third, `CordBuffer`'s internal `Rep::view_mut`, is exercised by
// `buffer.rs`'s own tests) and every method it exposes get direct coverage
// here.

#[test]
fn owned_rep_try_unique_none_when_shared_some_when_unique() {
    unsafe {
        let raw = make_flat(VALUE);
        let mut owned = OwnedRep::from_raw(raw);
        let mut cloned = owned.clone();

        // Shared (refcount 2): neither handle may claim exclusivity.
        assert!(owned.try_unique().is_none());
        assert!(cloned.try_unique().is_none());

        drop(cloned);

        // Back to refcount 1: the sole remaining handle is now unique.
        assert!(owned.try_unique().is_some());
    }
}

#[test]
fn inline_data_tree_unique_none_when_shared_some_when_unique() {
    unsafe {
        let raw = make_flat(VALUE);
        let extra = ref_rep(raw);
        let mut data = InlineData::from_tree(OwnedRep::from_raw(raw));

        // `extra` is a second, independent reference on the same rep.
        assert!(data.tree_unique().is_none());
        unref(extra);
        assert!(data.tree_unique().is_some());

        // Clean up the tree reference `data` still holds.
        drop(data.take_tree());
    }
}

#[test]
fn unique_rep_flat_spare_capacity_mut_bounds() {
    unsafe {
        let raw = flat::new(64);
        let capacity = flat::capacity(raw);
        let mut owned = OwnedRep::from_raw(raw);

        // len == 0: the whole capacity is spare.
        {
            let mut unique = owned.try_unique().expect("sole ref");
            assert_eq!(unique.flat_spare_capacity_mut().len(), capacity);
            unique.set_len(capacity);
        }

        // len == capacity: nothing is spare.
        {
            let mut unique = owned.try_unique().expect("sole ref");
            assert_eq!(unique.flat_spare_capacity_mut().len(), 0);
            // Restore length 0 so nothing downstream reads the
            // never-initialized payload as data.
            unique.set_len(0);
        }
    }
}

#[test]
fn unique_rep_set_len_visible_via_as_ref_len() {
    unsafe {
        let raw = flat::new(32);
        let mut owned = OwnedRep::from_raw(raw);
        let mut unique = owned.try_unique().expect("sole ref");
        assert_eq!(unique.as_ref().len(), 0);
        unique.set_len(10);
        assert_eq!(unique.as_ref().len(), 10);
    }
}

#[test]
fn unique_rep_substring_start_mut() {
    unsafe {
        let raw = make_flat(VALUE);
        let substr = make_substring(1, 20, raw);
        let mut owned = OwnedRep::from_raw(substr);

        {
            let mut unique = owned.try_unique().expect("sole ref");
            assert_eq!(*unique.substring_start_mut(), 1);
            *unique.substring_start_mut() = 5;
        }

        assert_eq!(*owned.try_unique().expect("sole ref").substring_start_mut(), 5);
    }
}

#[test]
fn unique_rep_flat_data_mut_and_into_flat_spare_capacity_mut() {
    unsafe {
        let raw = flat::new(32);
        let capacity = flat::capacity(raw);
        let mut owned = OwnedRep::from_raw(raw);

        // Fill 5 bytes through the `&mut self`-borrowed spare capacity.
        {
            let mut unique = owned.try_unique().expect("sole ref");
            unique.flat_spare_capacity_mut()[..5].write_copy_of_slice(b"hello");
            unique.set_len(5);
        }

        // `flat_data_mut` consumes the witness and hands back a slice tied
        // to `owned`'s borrow instead of the call's own: it outlives this
        // block's `unique` binding, which is exactly the point.
        {
            let data = owned.try_unique().expect("sole ref").flat_data_mut();
            assert_eq!(data, b"hello");
            data[0] = b'H';
        }
        assert_eq!(owned.as_ref().data(), b"Hello");

        // `into_flat_spare_capacity_mut` is the same by-value handoff for
        // the uninitialized tail.
        {
            let unique = owned.try_unique().expect("sole ref");
            let spare = unique.into_flat_spare_capacity_mut();
            assert_eq!(spare.len(), capacity - 5);
        }
    }
}
