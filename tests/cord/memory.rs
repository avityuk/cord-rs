//! `estimated_memory_usage` accounting.

use std::sync::Arc;

use crate::common::internal;
use cord_rs::{Cord, CordBuffer, MemoryAccounting};

const SIZEOF_CORD: usize = core::mem::size_of::<Cord>();
const FAIR_SHARE: MemoryAccounting = MemoryAccounting::FairShare;
const TOTAL: MemoryAccounting = MemoryAccounting::Total;
const TOTAL_MORE_PRECISE: MemoryAccounting = MemoryAccounting::TotalMorePrecise;

/// Creates a cord of `n` bytes `c` without adopting the source buffer.
fn make_cord(n: usize, c: u8) -> Cord {
    Cord::from(&vec![c; n][..])
}

#[test]
fn appending_a_small_buffer_costs_little_extra_memory() {
    // Allow a 32 byte flat and 128 bytes for glue nodes.
    const MAX_DELTA: usize = 128 + 32;
    const MAX_FLAT_LENGTH: usize = internal::MAX_FLAT_LENGTH;
    // Create a cord large enough to force 40KB of flats.
    let test_data = vec![b'x'; MAX_FLAT_LENGTH * 10];
    let mut cord1 = Cord::from(&test_data[..]);
    let mut cord2 = Cord::from(&test_data[..]);
    let size1 = cord1.estimated_memory_usage(MemoryAccounting::Total);
    let size2 = cord2.estimated_memory_usage(MemoryAccounting::Total);

    let mut buffer = CordBuffer::with_capacity(3);
    buffer.put_slice(b"Abc");
    cord1.append(buffer);

    let mut buffer = CordBuffer::with_capacity(3);
    buffer.put_slice(b"Abc");
    cord2.prepend(buffer);

    assert!(cord1.estimated_memory_usage(MemoryAccounting::Total) - size1 <= MAX_DELTA);
    assert!(cord2.estimated_memory_usage(MemoryAccounting::Total) - size2 <= MAX_DELTA);

    assert_eq!(cord1, [&test_data[..], b"Abc"].concat());
    assert_eq!(cord2, [b"Abc", &test_data[..]].concat());
}

#[test]
fn empty_cord_accounts_for_the_handle_only() {
    let cord = Cord::new();
    assert_eq!(cord.estimated_memory_usage(TOTAL), SIZEOF_CORD);
    assert_eq!(cord.estimated_memory_usage(FAIR_SHARE), SIZEOF_CORD);
    assert_eq!(cord.estimated_memory_usage(TOTAL_MORE_PRECISE), SIZEOF_CORD);
}

#[test]
fn inline_cord_accounts_for_the_handle_only() {
    let a = Cord::from("hello");
    assert_eq!(a.estimated_memory_usage(TOTAL), SIZEOF_CORD);
    assert_eq!(a.estimated_memory_usage(FAIR_SHARE), SIZEOF_CORD);
    assert_eq!(a.estimated_memory_usage(TOTAL_MORE_PRECISE), SIZEOF_CORD);
}

#[test]
fn external_node_accounts_for_data_plus_node() {
    let mut cord = Cord::new();
    cord.append(internal::make_external(&[b'x'; 1000]));
    let expected = SIZEOF_CORD + 1000 + internal::EXTERNAL_NODE_SIZE;
    assert_eq!(cord.estimated_memory_usage(TOTAL), expected);
    assert_eq!(cord.estimated_memory_usage(FAIR_SHARE), expected);
    assert_eq!(cord.estimated_memory_usage(TOTAL_MORE_PRECISE), expected);
}

#[test]
fn adopted_owners_account_for_their_allocation_not_their_length() {
    fn assert_usage(cord: &Cord, allocation_size: usize) {
        let expected = SIZEOF_CORD + allocation_size + internal::EXTERNAL_NODE_SIZE;
        assert_eq!(cord.estimated_memory_usage(TOTAL), expected);
        assert_eq!(cord.estimated_memory_usage(FAIR_SHARE), expected);
        assert_eq!(cord.estimated_memory_usage(TOTAL_MORE_PRECISE), expected);
    }

    let mut bytes = Vec::with_capacity(1536);
    bytes.resize(1024, b'x');
    let allocation_size = bytes.capacity();
    assert_usage(&Cord::from(bytes), allocation_size);

    let mut string = String::with_capacity(1536);
    string.push_str(&"x".repeat(1024));
    let allocation_size = string.capacity();
    assert_usage(&Cord::from(string), allocation_size);

    let boxed = vec![b'x'; 1024].into_boxed_slice();
    assert_usage(&Cord::from(boxed), 1024);
}

#[test]
fn arc_owner_accounts_for_its_length() {
    let owner: Arc<[u8]> = Arc::from(vec![b'x'; 1024]);
    let cord = Cord::from(owner);
    let expected = SIZEOF_CORD + cord.len() + internal::EXTERNAL_NODE_SIZE;
    assert_eq!(cord.estimated_memory_usage(TOTAL), expected);
    assert_eq!(cord.estimated_memory_usage(FAIR_SHARE), expected);
    assert_eq!(cord.estimated_memory_usage(TOTAL_MORE_PRECISE), expected);
}

#[test]
fn flat_accounts_for_its_allocated_size() {
    let cord = make_cord(1000, b'a');
    let flat_size = internal::flat_allocated_size(&cord).unwrap();
    assert_eq!(cord.estimated_memory_usage(TOTAL), SIZEOF_CORD + flat_size);
    assert_eq!(cord.estimated_memory_usage(FAIR_SHARE), SIZEOF_CORD + flat_size);
    assert_eq!(cord.estimated_memory_usage(TOTAL_MORE_PRECISE), SIZEOF_CORD + flat_size);
}

#[test]
fn substring_over_a_shared_flat_halves_the_fair_share() {
    let flat = make_cord(2000, b'a');
    let flat_size = internal::flat_allocated_size(&flat).unwrap();
    let cord = flat.slice(500..1500);
    assert_eq!(cord.estimated_memory_usage(TOTAL), SIZEOF_CORD + internal::SUBSTRING_NODE_SIZE + flat_size);
    assert_eq!(
        cord.estimated_memory_usage(TOTAL_MORE_PRECISE),
        SIZEOF_CORD + internal::SUBSTRING_NODE_SIZE + flat_size
    );
    assert_eq!(
        cord.estimated_memory_usage(FAIR_SHARE),
        SIZEOF_CORD + internal::SUBSTRING_NODE_SIZE + flat_size / 2
    );
}

#[test]
fn a_shared_flat_halves_the_fair_share() {
    let shared = make_cord(1000, b'a');
    let cord = shared.clone();
    let flat_size = internal::flat_allocated_size(&cord).unwrap();
    assert_eq!(cord.estimated_memory_usage(TOTAL), SIZEOF_CORD + flat_size);
    assert_eq!(cord.estimated_memory_usage(TOTAL_MORE_PRECISE), SIZEOF_CORD + flat_size);
    assert_eq!(cord.estimated_memory_usage(FAIR_SHARE), SIZEOF_CORD + flat_size / 2);
}

#[test]
fn btree_accounting_adds_up_across_nodes_and_sharing() {
    let mut cord1 = Cord::new();
    let mut flats1_size = 0;
    let flats1 = [make_cord(1000, b'a'), make_cord(1100, b'a'), make_cord(1200, b'a'), make_cord(1300, b'a')];
    for flat in flats1.iter().cloned() {
        flats1_size += internal::flat_allocated_size(&flat).unwrap();
        cord1.append(flat);
    }
    assert!(internal::is_btree(&cord1));

    let rep1_size = internal::BTREE_NODE_SIZE + flats1_size;
    let rep1_shared_size = internal::BTREE_NODE_SIZE + flats1_size / 2;

    assert_eq!(cord1.estimated_memory_usage(TOTAL), SIZEOF_CORD + rep1_size);
    assert_eq!(cord1.estimated_memory_usage(TOTAL_MORE_PRECISE), SIZEOF_CORD + rep1_size);
    assert_eq!(cord1.estimated_memory_usage(FAIR_SHARE), SIZEOF_CORD + rep1_shared_size);

    let mut cord2 = Cord::new();
    let mut flats2_size = 0;
    let flats2 = [make_cord(600, b'a'), make_cord(700, b'a'), make_cord(800, b'a'), make_cord(900, b'a')];
    for flat in flats2 {
        flats2_size += internal::flat_allocated_size(&flat).unwrap();
        cord2.append(flat);
    }
    let rep2_size = internal::BTREE_NODE_SIZE + flats2_size;

    assert_eq!(cord2.estimated_memory_usage(TOTAL), SIZEOF_CORD + rep2_size);
    assert_eq!(cord2.estimated_memory_usage(TOTAL_MORE_PRECISE), SIZEOF_CORD + rep2_size);
    assert_eq!(cord2.estimated_memory_usage(FAIR_SHARE), SIZEOF_CORD + rep2_size);

    let mut cord = cord1.clone();
    cord.append(cord2);

    assert_eq!(
        cord.estimated_memory_usage(TOTAL),
        SIZEOF_CORD + internal::BTREE_NODE_SIZE + rep1_size + rep2_size
    );
    assert_eq!(
        cord.estimated_memory_usage(TOTAL_MORE_PRECISE),
        SIZEOF_CORD + internal::BTREE_NODE_SIZE + rep1_size + rep2_size
    );
    assert_eq!(
        cord.estimated_memory_usage(FAIR_SHARE),
        SIZEOF_CORD + internal::BTREE_NODE_SIZE + rep1_shared_size / 2 + rep2_size
    );
}

/// Regression test: going from the inline rep to a tree too soon was
/// observable through memory usage.
#[test]
fn a_full_inline_cord_never_allocates() {
    let small_string = [b'x'; internal::MAX_INLINE];
    let c1 = Cord::from(&small_string[..]);
    let mut c2 = Cord::new();
    c2.append(&small_string[..]);
    assert_eq!(c1, c2);
    assert_eq!(c1.estimated_memory_usage(TOTAL), c2.estimated_memory_usage(TOTAL));
}

#[test]
fn total_more_precise_deduplicates_repeated_references() {
    const CHUNK_SIZE: usize = 2000;
    let flat = Cord::from(vec![b'x'; CHUNK_SIZE]);

    // `fragmented` has two references into the same buffer shared with `flat`.
    let mut fragmented = flat.clone();
    fragmented.append(&flat);

    let flat_internal_usage = flat.estimated_memory_usage(TOTAL) - SIZEOF_CORD;

    // `fragmented` holds a Cord and a btree node pointing to two copies of
    // flat's internals, which are expected to dedup.
    assert_eq!(
        fragmented.estimated_memory_usage(TOTAL_MORE_PRECISE),
        SIZEOF_CORD + internal::BTREE_NODE_SIZE + flat_internal_usage
    );
    // `Total` overestimates.
    assert_eq!(
        fragmented.estimated_memory_usage(TOTAL),
        SIZEOF_CORD + internal::BTREE_NODE_SIZE + 2 * flat_internal_usage
    );
}

#[test]
fn total_more_precise_deduplicates_through_substrings() {
    const CHUNK_SIZE: usize = 2000;
    let flat = Cord::from(vec![b'x'; CHUNK_SIZE]);

    // Each reference is through a slice this time.
    let mut fragmented = Cord::new();
    fragmented.append(flat.slice(1..CHUNK_SIZE - 1));
    fragmented.append(flat.slice(1..CHUNK_SIZE - 1));

    let flat_internal_usage = flat.estimated_memory_usage(TOTAL) - SIZEOF_CORD;

    assert_eq!(
        fragmented.estimated_memory_usage(TOTAL_MORE_PRECISE),
        SIZEOF_CORD + internal::BTREE_NODE_SIZE + 2 * internal::SUBSTRING_NODE_SIZE + flat_internal_usage
    );
    assert_eq!(
        fragmented.estimated_memory_usage(TOTAL),
        SIZEOF_CORD + internal::BTREE_NODE_SIZE + 2 * internal::SUBSTRING_NODE_SIZE + 2 * flat_internal_usage
    );
}

#[test]
fn accounting_modes_diverge_under_sharing() {
    let mut cord = Cord::new();
    for i in 0..100u8 {
        cord.append(vec![i; 100]);
    }
    let expected: Vec<u8> = (0..100u8).flat_map(|i| vec![i; 100]).collect();
    let total = cord.estimated_memory_usage(MemoryAccounting::Total);
    let precise = cord.estimated_memory_usage(MemoryAccounting::TotalMorePrecise);
    let fair = cord.estimated_memory_usage(MemoryAccounting::FairShare);
    assert!(total >= expected.len() + 16);
    assert_eq!(total, precise);
    assert_eq!(total, fair);
    let clone = cord.clone();
    assert_eq!(clone.estimated_memory_usage(MemoryAccounting::Total), total);
    assert!(clone.estimated_memory_usage(MemoryAccounting::FairShare) < total);
    let mut doubled = cord.clone();
    doubled.append(&cord);
    assert!(
        doubled.estimated_memory_usage(MemoryAccounting::Total)
            > doubled.estimated_memory_usage(MemoryAccounting::TotalMorePrecise)
    );
    drop(clone);
    drop(doubled);
}
