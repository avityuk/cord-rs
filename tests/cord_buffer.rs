//! `CordBuffer`'s own API: capacity, limits, `set_len`, clone, ordering,
//! hashing, `io::Write`, and conversion into a `Cord`.
#![expect(clippy::cast_possible_truncation, reason = "tests juggle small integers freely")]

#[path = "common/mod.rs"]
mod common;

use std::cmp::Ordering;
use std::collections::HashSet;
#[cfg(feature = "std")]
use std::io::Write;

use common::internal;
use cord_rs::{Cord, CordBuffer};

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
const REQUESTED_SIZES: [usize; 6] =
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
fn capacity_constants_match_the_flat_layout() {
    const { assert!(internal::FLAT_OVERHEAD < 32) };
    assert_eq!(CordBuffer::DEFAULT_MAX_CAPACITY, MAX_FLAT_LENGTH);
    assert_eq!(MAX_FLAT_LENGTH, 4096 - internal::FLAT_OVERHEAD);
    assert_eq!(CordBuffer::max_capacity_for(512), 512 - FLAT_OVERHEAD);
    assert_eq!(CordBuffer::max_capacity_for(K64KIB), K64KIB - FLAT_OVERHEAD);
    assert_eq!(CordBuffer::max_capacity_for(K1MB), K64KIB - FLAT_OVERHEAD);
}

#[test]
fn a_default_buffer_is_empty_and_inline() {
    let mut buffer = CordBuffer::new();
    assert_eq!(buffer.capacity(), core::mem::size_of::<CordBuffer>() - 1);
    assert_eq!(buffer.len(), 0);
    assert_eq!(buffer.spare_capacity_mut().len(), buffer.capacity());
    fill(&mut buffer, 0xCD);
    assert_eq!(buffer.available(), buffer.capacity());

    let default_buffer = CordBuffer::default();
    assert!(default_buffer.is_empty());
    assert_eq!(default_buffer.capacity(), CordBuffer::new().capacity());
}

#[test]
fn a_small_request_stays_inline_and_yields_an_inline_cord() {
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
fn spare_capacity_follows_the_written_length() {
    for requested in REQUESTED_SIZES {
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
fn set_len_extends_the_written_region() {
    for requested in REQUESTED_SIZES {
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
fn put_slice_partial_writes_only_what_fits() {
    for requested in REQUESTED_SIZES {
        let mut buffer = CordBuffer::with_capacity(requested);
        let expected_up_to = 3.min(buffer.capacity());
        assert_eq!(buffer.spare_capacity_mut()[..expected_up_to].len(), expected_up_to);
        let n = buffer.put_slice_partial(b"xyz");
        assert_eq!(n, expected_up_to);
    }
}

/// The two ways to request a buffer up to the default limit: the plain
/// constructor, and the block-size constructor called with the default
/// block size explicitly.
type BufferCtor = fn(usize) -> CordBuffer;
const CTORS: [(&str, BufferCtor); 2] = [
    ("with_capacity", CordBuffer::with_capacity),
    ("with_capacity_and_block_size(4 KiB)", |n| CordBuffer::with_capacity_and_block_size(n, MAX_FLAT_SIZE)),
];

#[test]
fn with_capacity_honours_requests_up_to_the_default_limit() {
    for (label, ctor) in CTORS {
        for requested in REQUESTED_SIZES {
            let mut buffer = ctor(requested);
            assert!(buffer.capacity() >= requested, "{label}: requested={requested}");
            assert!(
                buffer.capacity() <= max_capacity_for(MAX_FLAT_SIZE, requested),
                "{label}: requested={requested}"
            );
            assert_eq!(buffer.len(), 0, "{label}: requested={requested}");

            fill(&mut buffer, 0xCD);

            let data = vec![b'x'; requested - 1];
            buffer.put_slice(&data);
            buffer.put_slice(&[0]);
            assert_eq!(buffer.len(), requested, "{label}: requested={requested}");
            assert_eq!(&buffer[..requested - 1], &data[..], "{label}: requested={requested}");
        }
    }
}

#[test]
fn an_absurd_request_is_clamped_to_a_sane_size() {
    let k2gib: usize = 1 << 31;
    let mut buffer = CordBuffer::with_capacity(k2gib);
    // Never awarded more than a reasonable memory size.
    assert!(buffer.capacity() <= 2 * CordBuffer::DEFAULT_MAX_CAPACITY);
    assert_eq!(buffer.len(), 0);
    fill(&mut buffer, 0xCD);
}

#[test]
fn converting_a_buffer_into_a_cord_preserves_its_representation() {
    for requested in REQUESTED_SIZES {
        let mut buffer = CordBuffer::with_capacity(requested);
        buffer.put_slice(b"Abc");
        let heap = buffer.capacity() > INLINED_SIZE;
        let cord = Cord::from(buffer);
        assert_eq!(cord, "Abc");
        // A heap buffer is moved into the cord as is; SSO data is inlined.
        assert_eq!(internal::is_flat(&cord), heap);
    }

    // A small buffer's data is copied inline.
    let from_buffer: Cord = {
        let mut b = CordBuffer::with_capacity(100);
        b.put_slice(b"hello");
        b.into()
    };
    common::check(&from_buffer, b"hello");
}

// (requested, block_size, min_capacity, upper_bound_block, note)
const CASES: &[(usize, usize, usize, usize, &str)] = &[
    (DEFAULT_MAX_CAPACITY, K64KIB, DEFAULT_MAX_CAPACITY, MAX_FLAT_SIZE, "at the default limit"),
    (3178, K64KIB, 3178, MAX_FLAT_SIZE, "below the default limit"),
    (MAX_FLAT_SIZE, MAX_FLAT_SIZE, MAX_FLAT_SIZE - FLAT_OVERHEAD, MAX_FLAT_SIZE, "exact block"),
    (K8KIB, K8KIB, K8KIB - FLAT_OVERHEAD, K8KIB, "exact block"),
    (K16KIB, K16KIB, K16KIB - FLAT_OVERHEAD, K16KIB, "exact block"),
    (32 << 10, 32 << 10, (32 << 10) - FLAT_OVERHEAD, 32 << 10, "exact block"),
    (K64KIB, K64KIB, K64KIB - FLAT_OVERHEAD, K64KIB, "exact block"),
    (K1MB, K64KIB, K64KIB - FLAT_OVERHEAD, K64KIB, "request above the max block"),
    (1024, 512, 512 - FLAT_OVERHEAD, 512, "block smaller than the request"),
    (512, 512, 512 - FLAT_OVERHEAD, 512, "request equals the block"),
    (511, 512, 512 - FLAT_OVERHEAD, 512, "request + overhead exceeds the block"),
    (498, 512, 512 - FLAT_OVERHEAD, 512, "request + overhead fits the block"),
    (15 << 10, K16KIB, K8KIB - FLAT_OVERHEAD, K8KIB, "rounds down to the next power of two"),
    (K16KIB - 2 * FLAT_OVERHEAD, K16KIB, K16KIB - FLAT_OVERHEAD, K16KIB, "small slop keeps the full block"),
];

#[test]
fn capacity_for_request_and_block_size() {
    for &(requested, block_size, min_capacity, upper_bound_block, note) in CASES {
        let buffer = CordBuffer::with_capacity_and_block_size(requested, block_size);
        assert!(buffer.capacity() >= min_capacity, "{note}: requested={requested} block_size={block_size}");
        assert!(
            buffer.capacity() <= upper_bound_block - FLAT_OVERHEAD,
            "{note}: requested={requested} block_size={block_size}"
        );
    }
}

#[test]
fn capacity_near_usize_max_does_not_overflow() {
    // Matches the pattern in `edit.rs`'s
    // `append_buffer_capacity_near_usize_max_does_not_overflow`.
    let dist_from_max_max: usize = if cfg!(miri) { 4 } else { 32 };
    for dist_from_max in 0..=dist_from_max_max {
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
fn cord_buffer_clone_preserves_contents_and_capacity() {
    let buffers = vec![
        CordBuffer::new(),
        CordBuffer::with_capacity(100),
        CordBuffer::with_capacity(4000),
        CordBuffer::with_capacity_and_block_size(20_000, 64 << 10),
        CordBuffer::with_capacity_and_block_size(2000, 512),
    ];
    for mut original in buffers {
        original.put_slice(b"hello");
        let mut clone = original.clone();
        assert_eq!(clone.capacity(), original.capacity(), "clone must preserve capacity");
        assert_eq!(clone.as_slice(), b"hello");
        assert_eq!(clone, original);

        // Independence: mutating the clone must not affect the original.
        clone.put_slice(b"!");
        assert_eq!(clone.as_slice(), b"hello!");
        assert_eq!(original.as_slice(), b"hello");
    }
}

#[test]
fn extend_fills_partial_then_exact_capacity() {
    // Both the inline (`Short`) and heap (`Flat`) representations.
    for mut buffer in [CordBuffer::new(), CordBuffer::with_capacity(100)] {
        let cap = buffer.capacity();
        let half = cap / 2;

        // Partial fill via `Extend<u8>`.
        let first: Vec<u8> = (0..half as u32).map(|i| (i % 256) as u8).collect();
        buffer.extend(first.iter().copied());
        assert_eq!(buffer.as_slice(), first.as_slice());
        assert_eq!(buffer.available(), cap - half);

        // Exactly fill the rest via `Extend<&u8>`.
        let rest: Vec<u8> = (0..(cap - half) as u32).map(|i| ((i + 100) % 256) as u8).collect();
        buffer.extend(rest.iter());
        assert_eq!(buffer.available(), 0);
        let mut expected = first;
        expected.extend_from_slice(&rest);
        assert_eq!(buffer.as_slice(), expected.as_slice());
    }
}

#[test]
fn buffer_ordering_hashing_and_cross_type_equality() {
    let mut low = CordBuffer::with_capacity(16);
    low.put_slice(b"abc");
    let mut mid = CordBuffer::with_capacity(16);
    mid.put_slice(b"abd");
    let mut high = CordBuffer::with_capacity(16);
    high.put_slice(b"abe");

    // `==`, `cmp` and `partial_cmp` all agree with the equivalent slice
    // comparison for a Less/Equal/Greater triple.
    for (a, b) in [(&low, &mid), (&mid, &mid), (&high, &mid)] {
        assert_eq!(a.cmp(b), a.as_slice().cmp(b.as_slice()));
        assert_eq!(a.partial_cmp(b), Some(a.cmp(b)));
        assert_eq!(*a == *b, a.as_slice() == b.as_slice());
    }
    assert_eq!(low.cmp(&mid), Ordering::Less);
    assert_eq!(mid.cmp(&mid), Ordering::Equal);
    assert_eq!(high.cmp(&mid), Ordering::Greater);

    // `Borrow<[u8]>` + `Hash` agreement with `<[u8]>::hash`, exercised
    // through a `HashSet<CordBuffer>` looked up by `&[u8]`.
    let mut set = HashSet::new();
    set.insert(low.clone());
    set.insert(mid.clone());
    assert!(set.contains(b"abc".as_slice()));
    assert!(set.contains(b"abd".as_slice()));
    assert!(!set.contains(b"xyz".as_slice()));
    assert_eq!(common::default_hash(&low), common::default_hash(low.as_slice()));
    assert_eq!(common::default_hash(&mid), common::default_hash(mid.as_slice()));

    // Cross-type equality, both directions.
    let v: Vec<u8> = b"abc".to_vec();
    assert_eq!(low, v);
    assert_eq!(v, low);
    assert_eq!(low, b"abc".as_slice());
    assert_eq!(b"abc".as_slice(), low);
    assert_eq!(low, *b"abc");
    assert_eq!(*b"abc", low);
    let arr: [u8; 3] = *b"abc";
    assert_eq!(low, &arr);
    assert_eq!(&arr, low);

    let mut text = CordBuffer::with_capacity(16);
    text.put_slice(b"hi");
    assert_eq!(text, "hi");
    assert_eq!("hi", text);
    assert_eq!(text, "hi".to_string());
    assert_eq!("hi".to_string(), text);
}

#[cfg(feature = "std")]
#[test]
fn io_write_returns_zero_when_the_buffer_is_full() {
    let mut buffer = CordBuffer::with_capacity(4);
    let cap = buffer.capacity();
    let n = buffer.write(&vec![b'x'; cap]).unwrap();
    assert_eq!(n, cap);
    // Documented contract: `write` returns `Ok(0)` once full, so `write_all`
    // fails with `WriteZero` rather than looping or panicking.
    let err = buffer.write_all(b"more").unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::WriteZero);
    buffer.flush().unwrap();
}

#[test]
fn capacity_sweep_over_block_sizes_and_requests() {
    let min_flat = internal::MIN_FLAT_LENGTH;
    let capacities: &[usize] = &[
        0,
        1,
        8,
        internal::FLAT_OVERHEAD,
        internal::FLAT_OVERHEAD + 1,
        CordBuffer::DEFAULT_MAX_CAPACITY - 1,
        CordBuffer::DEFAULT_MAX_CAPACITY,
        CordBuffer::DEFAULT_MAX_CAPACITY + 1,
        1000,
        19_586,
        CordBuffer::MAX_BLOCK_SIZE - 1,
        CordBuffer::MAX_BLOCK_SIZE,
        CordBuffer::MAX_BLOCK_SIZE + 1,
        1 << 20,
    ];
    let mut block_size = 16usize;
    while block_size <= CordBuffer::MAX_BLOCK_SIZE {
        let expected_max = CordBuffer::max_capacity_for(block_size);
        for &capacity in capacities {
            let buffer = CordBuffer::with_capacity_and_block_size(capacity, block_size);
            // Documented floor: never less than `min(requested, MIN_FLAT_LENGTH)`
            // (`with_capacity_and_block_size` allocates via `flat::new_large`,
            // which floors the payload at `MIN_FLAT_LENGTH` regardless of how
            // small `capacity`/`block_size` are).
            assert!(
                buffer.capacity() >= capacity.min(min_flat),
                "block_size={block_size} capacity={capacity}: got {}",
                buffer.capacity()
            );
            // Saturating case: requesting at least the full block size must
            // agree exactly with `max_capacity_for`.
            if capacity >= block_size {
                assert_eq!(
                    buffer.capacity(),
                    expected_max,
                    "block_size={block_size} capacity={capacity}: disagrees with max_capacity_for"
                );
            }
        }
        block_size *= 2;
    }

    // Illegal block sizes: powers of two at or below FLAT_OVERHEAD must panic.
    let mut illegal = 1usize;
    while illegal <= internal::FLAT_OVERHEAD {
        let result = std::panic::catch_unwind(|| CordBuffer::with_capacity_and_block_size(10, illegal));
        assert!(result.is_err(), "block_size={illegal} should have panicked");
        let result = std::panic::catch_unwind(|| CordBuffer::max_capacity_for(illegal));
        assert!(result.is_err(), "max_capacity_for({illegal}) should have panicked");
        illegal *= 2;
    }
}
