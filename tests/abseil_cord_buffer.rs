//! Port of abseil's `cord_buffer_test.cc`.
#![allow(unused_assignments)]

use cord_rs::{__internal as internal, Cord, CordBuffer};

const INLINED_SIZE: usize = core::mem::size_of::<CordBuffer>() - 1;
const DEFAULT_MAX_CAPACITY: usize = CordBuffer::DEFAULT_MAX_CAPACITY;
const MAX_BLOCK_SIZE: usize = CordBuffer::MAX_BLOCK_SIZE;
const MAX_FLAT_SIZE: usize = 4096;
const MAX_FLAT_LENGTH: usize = internal::MAX_FLAT_LENGTH;
const FLAT_OVERHEAD: usize = internal::FLAT_OVERHEAD;
const K8KIB: usize = 8 << 10;
const K16KIB: usize = 16 << 10;
const K64KIB: usize = 64 << 10;
const K1MB: usize = 1 << 20;

/// The C++ fixture parameters: "medium" requested sizes.
const PARAMS: [usize; 6] =
    [1, INLINED_SIZE - 1, INLINED_SIZE, INLINED_SIZE + 1, DEFAULT_MAX_CAPACITY - 1, DEFAULT_MAX_CAPACITY];

/// Maximum capacity for a given block size (always `block_size - overhead`).
fn max_capacity_for(block_size: usize, _requested: usize) -> usize {
    block_size - FLAT_OVERHEAD
}

fn fill(buffer: &mut CordBuffer, byte: u8) {
    for slot in buffer.spare_capacity_mut() {
        slot.write(byte);
    }
}

#[test]
fn maximum_payload() {
    assert_eq!(CordBuffer::DEFAULT_MAX_CAPACITY, MAX_FLAT_LENGTH);
    assert_eq!(CordBuffer::max_capacity_for(512), 512 - FLAT_OVERHEAD);
    assert_eq!(CordBuffer::max_capacity_for(K64KIB), K64KIB - FLAT_OVERHEAD);
    assert_eq!(CordBuffer::max_capacity_for(K1MB), K64KIB - FLAT_OVERHEAD);
}

#[test]
fn construct_default() {
    let mut buffer = CordBuffer::new();
    assert_eq!(buffer.capacity(), core::mem::size_of::<CordBuffer>() - 1);
    assert_eq!(buffer.len(), 0);
    assert_eq!(buffer.spare_capacity_mut().len(), buffer.capacity());
    fill(&mut buffer, 0xCD);
    assert_eq!(buffer.available(), buffer.capacity());
}

#[test]
fn create_sso_with_default_limit() {
    let mut buffer = CordBuffer::with_capacity(3);
    assert!(buffer.capacity() >= 3);
    assert!(buffer.capacity() <= core::mem::size_of::<CordBuffer>());
    assert_eq!(buffer.len(), 0);
    fill(&mut buffer, 0xCD);

    buffer.put_slice(b"Abc");
    assert_eq!(buffer.len(), 3);
    assert_eq!(&*buffer, b"Abc");
    // A consumed SSO buffer becomes an inline cord.
    let cord = Cord::from(buffer);
    assert_eq!(cord, "Abc");
    assert!(!internal::is_tree(&cord));
}

#[test]
fn available() {
    for requested in PARAMS {
        let mut buffer = CordBuffer::with_capacity(requested);
        let base = buffer.as_slice().as_ptr();
        assert_eq!(buffer.spare_capacity_mut().as_ptr().cast::<u8>(), base);
        assert_eq!(buffer.spare_capacity_mut().len(), buffer.capacity());

        // SAFETY: the two bytes are initialized right below.
        buffer.spare_capacity_mut()[0].write(b'a');
        buffer.spare_capacity_mut()[1].write(b'b');
        unsafe { buffer.set_len(2) };
        // SAFETY: pointer arithmetic within the buffer.
        assert_eq!(buffer.spare_capacity_mut().as_ptr().cast::<u8>(), unsafe { base.add(2) });
        assert_eq!(buffer.spare_capacity_mut().len(), buffer.capacity() - 2);
        assert_eq!(buffer.available(), buffer.capacity() - 2);
    }
}

#[test]
fn increase_length_by() {
    for requested in PARAMS {
        let mut buffer = CordBuffer::with_capacity(requested);
        fill(&mut buffer, 0);
        // SAFETY: the whole capacity was initialized above.
        unsafe {
            buffer.set_len(2);
            assert_eq!(buffer.len(), 2);
            buffer.set_len(7);
            assert_eq!(buffer.len(), 7);
        }
    }
}

#[test]
fn available_up_to() {
    for requested in PARAMS {
        let mut buffer = CordBuffer::with_capacity(requested);
        let expected_up_to = 3.min(buffer.capacity());
        assert_eq!(buffer.spare_capacity_mut()[..expected_up_to].len(), expected_up_to);
        let n = buffer.put_slice_partial(b"xyz");
        assert_eq!(n, expected_up_to);
    }
}

#[test]
fn create_with_default_limit() {
    for requested in PARAMS {
        let mut buffer = CordBuffer::with_capacity(requested);
        assert!(buffer.capacity() >= requested);
        assert!(buffer.capacity() <= max_capacity_for(MAX_FLAT_SIZE, requested));
        assert_eq!(buffer.len(), 0);

        fill(&mut buffer, 0xCD);

        let data = vec![b'x'; requested - 1];
        buffer.put_slice(&data);
        buffer.put_slice(&[0]);
        assert_eq!(buffer.len(), requested);
        assert_eq!(&buffer[..requested - 1], &data[..]);
    }
}

#[test]
fn create_with_default_limit_asking_for_2gb() {
    let k2gib: usize = 1 << 31;
    let mut buffer = CordBuffer::with_capacity(k2gib);
    // Never awarded more than a reasonable memory size.
    assert!(buffer.capacity() <= 2 * CordBuffer::DEFAULT_MAX_CAPACITY);
    assert_eq!(buffer.len(), 0);
    fill(&mut buffer, 0xCD);
}

#[test]
fn move_construct() {
    for requested in PARAMS {
        let mut from = CordBuffer::with_capacity(requested);
        let capacity = from.capacity();
        from.put_slice(b"Abc\0");
        let to = from;
        assert_eq!(to.capacity(), capacity);
        assert_eq!(to.len(), 4);
        assert_eq!(&to[..3], b"Abc");
    }
}

#[test]
fn move_assign() {
    for requested in PARAMS {
        let mut from = CordBuffer::with_capacity(requested);
        let capacity = from.capacity();
        from.put_slice(b"Abc\0");
        let mut to = CordBuffer::new();
        to = core::mem::take(&mut from);
        assert_eq!(to.capacity(), capacity);
        assert_eq!(to.len(), 4);
        assert_eq!(&to[..3], b"Abc");
        assert_eq!(from.len(), 0);
    }
}

#[test]
fn consume_value() {
    for requested in PARAMS {
        let mut buffer = CordBuffer::with_capacity(requested);
        buffer.put_slice(b"Abc");
        let heap = buffer.capacity() > INLINED_SIZE;
        let cord = Cord::from(buffer);
        assert_eq!(cord, "Abc");
        // A heap buffer is moved into the cord as is; SSO data is inlined.
        assert_eq!(internal::is_flat(&cord), heap);
    }
}

#[test]
fn create_with_custom_limit_within_default_limit() {
    for requested in PARAMS {
        let mut buffer = CordBuffer::with_capacity_and_block_size(requested, MAX_FLAT_SIZE);
        assert!(buffer.capacity() >= requested);
        assert!(buffer.capacity() <= max_capacity_for(MAX_FLAT_SIZE, requested));
        assert_eq!(buffer.len(), 0);

        fill(&mut buffer, 0xCD);

        let data = vec![b'x'; requested - 1];
        buffer.put_slice(&data);
        buffer.put_slice(&[0]);
        assert_eq!(buffer.len(), requested);
        assert_eq!(&buffer[..requested - 1], &data[..]);
    }
}

#[test]
fn create_at_or_below_default_limit() {
    let buffer = CordBuffer::with_capacity_and_block_size(DEFAULT_MAX_CAPACITY, K64KIB);
    assert!(buffer.capacity() >= DEFAULT_MAX_CAPACITY);
    assert!(buffer.capacity() <= max_capacity_for(MAX_FLAT_SIZE, DEFAULT_MAX_CAPACITY));

    let buffer = CordBuffer::with_capacity_and_block_size(3178, K64KIB);
    assert!(buffer.capacity() >= 3178);
}

#[test]
fn create_with_custom_limit() {
    assert!(MAX_FLAT_SIZE.is_power_of_two());
    let mut size = MAX_FLAT_SIZE;
    while size <= MAX_BLOCK_SIZE {
        let buffer = CordBuffer::with_capacity_and_block_size(size, size);
        let expected = size - FLAT_OVERHEAD;
        assert!(buffer.capacity() >= expected);
        assert!(buffer.capacity() <= max_capacity_for(size, expected));
        size *= 2;
    }
}

#[test]
fn create_with_too_large_limit() {
    let buffer = CordBuffer::with_capacity_and_block_size(K1MB, K64KIB);
    assert!(buffer.capacity() >= K64KIB - FLAT_OVERHEAD);
    assert!(buffer.capacity() <= max_capacity_for(K64KIB, K1MB));
}

#[test]
fn create_with_huge_value_for_overflow_hardening() {
    for dist_from_max in 0..=32usize {
        let capacity = usize::MAX - dist_from_max;

        let buffer = CordBuffer::with_capacity(capacity);
        assert!(buffer.capacity() >= DEFAULT_MAX_CAPACITY);
        assert!(buffer.capacity() <= max_capacity_for(MAX_FLAT_SIZE, capacity));

        let mut limit = MAX_FLAT_SIZE;
        while limit <= MAX_BLOCK_SIZE {
            let buffer = CordBuffer::with_capacity_and_block_size(capacity, limit);
            assert!(buffer.capacity() >= limit - FLAT_OVERHEAD);
            assert!(buffer.capacity() <= max_capacity_for(limit, capacity));
            limit *= 2;
        }
    }
}

#[test]
fn create_with_small_limit() {
    let buffer = CordBuffer::with_capacity_and_block_size(1024, 512);
    assert!(buffer.capacity() >= 512 - FLAT_OVERHEAD);
    assert!(buffer.capacity() <= max_capacity_for(512, 1024));

    // Precise block size returns size - overhead.
    let buffer = CordBuffer::with_capacity_and_block_size(512, 512);
    assert!(buffer.capacity() >= 512 - FLAT_OVERHEAD);
    assert!(buffer.capacity() <= max_capacity_for(512, 512));

    // Corner case: 511 < block_size, but 511 + overhead is above.
    let buffer = CordBuffer::with_capacity_and_block_size(511, 512);
    assert!(buffer.capacity() >= 512 - FLAT_OVERHEAD);
    assert!(buffer.capacity() <= max_capacity_for(512, 511));

    // Corner case: 498 + overhead < block_size.
    let buffer = CordBuffer::with_capacity_and_block_size(498, 512);
    assert!(buffer.capacity() >= 512 - FLAT_OVERHEAD);
    assert!(buffer.capacity() <= max_capacity_for(512, 498));
}

#[test]
fn create_waste_full() {
    // 15 KiB gets rounded down to the next power of 2.
    let requested = 15 << 10;
    let buffer = CordBuffer::with_capacity_and_block_size(requested, K16KIB);
    assert!(buffer.capacity() >= K8KIB - FLAT_OVERHEAD);
    assert!(buffer.capacity() <= max_capacity_for(K8KIB, requested));
}

#[test]
fn create_small_slop() {
    let requested = K16KIB - 2 * FLAT_OVERHEAD;
    let buffer = CordBuffer::with_capacity_and_block_size(requested, K16KIB);
    assert!(buffer.capacity() >= K16KIB - FLAT_OVERHEAD);
    assert!(buffer.capacity() <= max_capacity_for(K16KIB, requested));
}
