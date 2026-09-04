use super::btree::{BtreePtr, CordRepBtree};
use super::test_util::*;
use super::{edge_data, is_data_edge, unref};

const VALUE: &[u8] = b"Lorem ipsum dolor sit amet, consectetur ...";

#[test]
fn is_data_edge_on_flat() {
    unsafe {
        let rep = make_flat(VALUE);
        assert!(is_data_edge(rep));
        unref(rep);
    }
}

#[test]
fn is_data_edge_on_external() {
    unsafe {
        let rep = make_external(VALUE);
        assert!(is_data_edge(rep));
        unref(rep);
    }
}

#[test]
fn is_data_edge_on_substring_of_flat() {
    unsafe {
        let rep = make_flat(VALUE);
        let substr = make_substring(1, 20, rep);
        assert!(is_data_edge(substr));
        unref(substr);
    }
}

#[test]
fn is_data_edge_on_substring_of_external() {
    unsafe {
        let rep = make_external(VALUE);
        let substr = make_substring(1, 20, rep);
        assert!(is_data_edge(substr));
        unref(substr);
    }
}

#[test]
fn is_data_edge_on_btree() {
    unsafe {
        let rep = make_flat(VALUE);
        let tree = CordRepBtree::new_with(rep);
        assert!(!is_data_edge(tree.as_rep()));
        unref(tree.as_rep());
    }
}

#[test]
fn is_data_edge_on_bad_substr() {
    unsafe {
        let rep = make_flat(VALUE);
        let substr = make_substring(1, 18, make_substring(1, 20, rep));
        assert!(!is_data_edge(substr));
        unref(substr);
    }
}

#[test]
fn edge_data_on_flat() {
    unsafe {
        let rep = make_flat(VALUE);
        assert_eq!(edge_data(rep), VALUE);
        unref(rep);
    }
}

#[test]
fn edge_data_on_external() {
    unsafe {
        let rep = make_external(VALUE);
        assert_eq!(edge_data(rep), VALUE);
        unref(rep);
    }
}

#[test]
fn edge_data_on_substring_of_flat() {
    unsafe {
        let rep = make_flat(VALUE);
        let substr = make_substring(1, 20, rep);
        assert_eq!(edge_data(substr), &VALUE[1..21]);
        unref(substr);
    }
}

#[test]
fn edge_data_on_substring_of_external() {
    unsafe {
        let rep = make_external(VALUE);
        let substr = make_substring(1, 20, rep);
        assert_eq!(edge_data(substr), &VALUE[1..21]);
        unref(substr);
    }
}

#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "assertion failed: is_data_edge(edge)")]
fn edge_data_on_btree_death() {
    unsafe {
        let mut refs = AutoUnref::new();
        let rep = make_flat(VALUE);
        let tree = refs.add(CordRepBtree::new_with(rep));
        let _ = edge_data(tree.as_rep());
    }
}

#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "assertion failed: is_data_edge(edge)")]
fn edge_data_on_bad_substr_death() {
    unsafe {
        let mut refs = AutoUnref::new();
        let rep = make_flat(VALUE);
        let substr = refs.add(make_substring(1, 18, make_substring(1, 20, rep)));
        let _ = edge_data(substr);
    }
}
