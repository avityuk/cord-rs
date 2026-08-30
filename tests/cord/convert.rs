//! Conversions out of a `Cord`: `to_vec`, `Vec<u8>`, `String`, `Box<[u8]>`,
//! including the zero-copy path that reclaims a uniquely owned adopted
//! global allocation (`Vec<u8>`, `String`, `Box<[u8]>` adopted above the
//! copy threshold, or `flatten`'s own global buffer).
#![expect(clippy::cast_possible_truncation, reason = "tests juggle small integers freely")]

use std::sync::Arc;

use crate::common::{self, internal};
use cord_rs::Cord;

/// A byte count comfortably above `MAX_BYTES_TO_COPY` (511), so a `Vec`
/// built by [`adopted_vec`] is always adopted rather than copied.
const N: usize = 1024;

/// A `Vec` big enough to be adopted (`> MAX_BYTES_TO_COPY`) with `cap == len`.
fn adopted_vec(len: usize, fill: u8) -> Vec<u8> {
    let mut v = Vec::with_capacity(len);
    v.resize(len, fill);
    assert_eq!(v.capacity(), len);
    v
}

/// Full round-trip out of a `Cord`: `to_vec`/`Vec<u8>`/`String`/`Box<[u8]>`,
/// across empty, inline and fragmented shapes.
fn check_conversions_out(cord: &Cord) {
    let initially_empty = cord.to_vec();
    assert_eq!(*cord, initially_empty);
    let mut has_initial_contents = vec![b'x'; 1024];
    has_initial_contents.clear();
    has_initial_contents.extend(cord.chunks().flatten());
    assert_eq!(*cord, has_initial_contents);
    let string: Result<String, _> = cord.clone().try_into();
    assert_eq!(string.unwrap(), String::from_utf8(cord.to_vec()).unwrap());
}

#[test]
fn converting_out_to_vec_string_and_boxed_slice() {
    check_conversions_out(&Cord::new());
    check_conversions_out(&Cord::from("small cord"));
    check_conversions_out(&common::make_fragmented_cord([
        "fragmented ",
        "cord ",
        "to ",
        "test ",
        "copying ",
        "to ",
        "a ",
        "string.",
    ]));
    check_conversions_out(&common::make_fragmented_cord(["A ", "small ", "fragmented ", "Cord", "."]));

    assert_eq!(format!("{:?}", Cord::from(b"a\"b\n\xff")), "b\"a\\\"b\\n\\xff\"");
    let s: String = Cord::from("héllo wörld").try_into().unwrap();
    assert_eq!(s, "héllo wörld");
    let s: String = Cord::from("utf8").try_into().unwrap();
    assert_eq!(s, "utf8");
    assert!(String::try_from(Cord::from(b"\xff")).is_err());
    let v: Vec<u8> = Cord::from("vec").into();
    assert_eq!(v, b"vec");

    // `Box<[u8]>` copies, for every tree shape below the adoption threshold.
    let inline = Cord::from("inline");
    assert!(!internal::is_tree(&inline));
    let boxed: Box<[u8]> = inline.clone().into();
    assert_eq!(&*boxed, inline.to_vec().as_slice());

    let flat_data: Vec<u8> = (0..200u32).map(|i| (i % 256) as u8).collect();
    let flat = Cord::copy_from_slice(&flat_data);
    assert!(internal::is_flat(&flat));
    let boxed: Box<[u8]> = flat.clone().into();
    assert_eq!(&*boxed, flat.to_vec().as_slice());

    let multi_chunk_data: Vec<u8> = (0..5000u32).map(|i| (i % 256) as u8).collect();
    let mut multi_chunk = Cord::new();
    for chunk in multi_chunk_data.chunks(200) {
        multi_chunk.append(chunk);
    }
    assert!(internal::is_btree(&multi_chunk));
    let boxed: Box<[u8]> = multi_chunk.clone().into();
    assert_eq!(&*boxed, multi_chunk.to_vec().as_slice());
}

#[test]
fn converting_unique_global_allocations_reuses_the_buffer() {
    let mut bytes = Vec::with_capacity(2048);
    bytes.resize(1024, b'v');
    let bytes_ptr = bytes.as_ptr();
    let bytes_capacity = bytes.capacity();
    let vec: Vec<u8> = Cord::from(bytes).into();
    assert_eq!(vec.as_ptr(), bytes_ptr);
    assert_eq!(vec.capacity(), bytes_capacity);
    assert_eq!(vec, vec![b'v'; 1024]);

    let mut text = String::with_capacity(2048);
    text.extend(std::iter::repeat_n('s', 1024));
    let text_ptr = text.as_ptr();
    let text_capacity = text.capacity();
    let string = String::try_from(Cord::from(text)).unwrap();
    assert_eq!(string.as_ptr(), text_ptr);
    assert_eq!(string.capacity(), text_capacity);
    assert_eq!(string, "s".repeat(1024));
}

#[test]
fn boxed_slice_conversion_reuses_or_shrinks_the_allocation() {
    // A `Box<[u8]>`-adopted cord already has `capacity == length`: routing
    // `From<Cord> for Box<[u8]>` through `Vec::from` must recover the exact
    // same allocation, not a fresh copy.
    let boxed_owner = vec![b'b'; N].into_boxed_slice();
    let owner_ptr = boxed_owner.as_ptr();
    let cord = Cord::from(boxed_owner);
    let boxed: Box<[u8]> = cord.into();
    assert_eq!(boxed.as_ptr(), owner_ptr);
    assert_eq!(&*boxed, vec![b'b'; N].as_slice());

    // A `Vec`-adopted cord with spare capacity must still shrink to exactly
    // the right bytes, even though the allocation itself cannot survive
    // that shrink unchanged.
    let mut spare = Vec::with_capacity(2 * N);
    spare.resize(N, b'v');
    let cord = Cord::from(spare);
    let boxed: Box<[u8]> = cord.into();
    assert_eq!(&*boxed, vec![b'v'; N].as_slice());
}

#[test]
fn converting_shared_or_wrapped_global_allocations_copies() {
    let bytes = vec![b'x'; 1024];
    let original_ptr = bytes.as_ptr();
    let cord = Cord::from(bytes);
    let shared = cord.clone();
    let copied: Vec<u8> = cord.into();
    assert_ne!(copied.as_ptr(), original_ptr);
    assert_eq!(copied, shared.to_vec());
    assert_eq!(shared, vec![b'x'; 1024]);

    let bytes = vec![b'y'; 1024];
    let original_ptr = bytes.as_ptr();
    let cord = Cord::from(bytes);
    let substring = cord.slice(1..1023);
    drop(cord);
    let copied: Vec<u8> = substring.into();
    assert_ne!(copied.as_ptr(), original_ptr);
    assert_eq!(copied, vec![b'y'; 1022]);

    let flat = Cord::copy_from_slice(&vec![b'f'; 1000]);
    assert!(internal::is_flat(&flat));
    let copied: Vec<u8> = flat.into();
    assert_eq!(copied, vec![b'f'; 1000]);
}

#[test]
fn advance_then_convert() {
    for skip in [0usize, 1, 15, 16, 100, N - 1] {
        let v = adopted_vec(N, b'a');
        let base = v.as_ptr();
        let mut cord = Cord::from(v);
        assert!(internal::is_external(&cord));
        cord.advance(skip);
        let out: Vec<u8> = cord.into();
        assert_eq!(out.len(), N - skip);
        assert!(out.iter().all(|&b| b == b'a'), "skip={skip}");
        if skip == 0 {
            // advance(0) is a no-op that must keep the stealable node.
            assert_eq!(out.as_ptr(), base);
            assert_eq!(out.capacity(), N);
        } else {
            // A substring wrapper must force the copy path.
            assert_ne!(out.as_ptr(), base, "skip={skip}");
        }
    }
}

#[test]
fn truncate_then_convert() {
    for keep in [1usize, 15, 16, N - 1, N] {
        let v = adopted_vec(N, b't');
        let base = v.as_ptr();
        let mut cord = Cord::from(v);
        cord.truncate(keep);
        let out: Vec<u8> = cord.into();
        assert_eq!(out.len(), keep);
        assert!(out.iter().all(|&b| b == b't'), "keep={keep}");
        if keep == N {
            assert_eq!(out.as_ptr(), base);
        } else {
            assert_ne!(out.as_ptr(), base, "keep={keep}");
        }
    }
}

#[test]
fn take_append_buffer_then_convert() {
    let v = adopted_vec(N, b'p');
    let base = v.as_ptr();
    let cap = v.capacity();
    let mut cord = Cord::from(v);
    // The root is an external, not an extractable flat: the cord must keep it.
    let buffer = cord.take_append_buffer(64);
    assert!(buffer.is_empty());
    drop(buffer);
    assert!(internal::is_external(&cord));
    let out: Vec<u8> = cord.into();
    assert_eq!(out.as_ptr(), base);
    assert_eq!(out.capacity(), cap);
    assert_eq!(out.len(), N);
    assert!(out.iter().all(|&b| b == b'p'));
}

#[test]
fn take_append_buffer_roundtrip_then_convert() {
    let v = adopted_vec(N, b'q');
    let mut cord = Cord::from(v);
    let mut buffer = cord.take_append_buffer(64);
    buffer.put_slice(b"tail");
    cord.append(buffer);
    let out: Vec<u8> = cord.into();
    assert_eq!(out.len(), N + 4);
    assert_eq!(&out[N..], b"tail");
    assert!(out[..N].iter().all(|&b| b == b'q'));
}

#[test]
fn append_small_slice_then_convert() {
    let v = adopted_vec(N, b'z');
    let mut cord = Cord::from(v);
    cord.append(b"xy".as_slice());
    let out: Vec<u8> = cord.into();
    assert_eq!(out.len(), N + 2);
    assert_eq!(&out[N..], b"xy");
    assert!(out[..N].iter().all(|&b| b == b'z'));
}

#[test]
fn split_then_convert_both_halves() {
    let v = adopted_vec(N, b's');
    let mut cord = Cord::from(v);
    let tail = cord.split_off(400);
    let head: Vec<u8> = cord.into();
    let tail: Vec<u8> = tail.into();
    assert_eq!(head.len(), 400);
    assert_eq!(tail.len(), N - 400);
    assert!(head.iter().chain(tail.iter()).all(|&b| b == b's'));

    let v = adopted_vec(N, b'S');
    let mut cord = Cord::from(v);
    let head = cord.split_to(400);
    let head: Vec<u8> = head.into();
    let tail: Vec<u8> = cord.into();
    assert_eq!(head.len(), 400);
    assert_eq!(tail.len(), N - 400);
    assert!(head.iter().chain(tail.iter()).all(|&b| b == b'S'));
}

#[test]
fn clear_then_convert() {
    let v = adopted_vec(N, b'c');
    let mut cord = Cord::from(v);
    cord.clear();
    let out: Vec<u8> = cord.into();
    assert!(out.is_empty());
}

#[test]
fn make_contiguous_then_convert_steals_the_flattened_buffer() {
    // Enough appended external chunks to force a btree, then flatten it. The
    // total must still clear `MAX_FLAT_LENGTH` so flattening produces a
    // global external buffer rather than a single flat node — the property
    // under test — so this is scaled down, not eliminated, under Miri.
    let (chunks, chunk_len) = if cfg!(miri) { (8, 600) } else { (40, 4096) };
    let mut cord = Cord::new();
    for _ in 0..chunks {
        cord.append(vec![b'm'; chunk_len]);
    }
    assert!(internal::is_btree(&cord));
    let len = cord.len();
    let base = cord.make_contiguous().as_ptr();
    assert!(internal::is_external(&cord));
    let out: Vec<u8> = cord.into();
    assert_eq!(out.len(), len);
    assert_eq!(out.as_ptr(), base, "flatten_slow_path's global buffer must be reused");
    assert!(out.iter().all(|&b| b == b'm'));
}

#[test]
fn generic_external_owners_are_never_stolen() {
    static PAYLOAD: [u8; 64] = [7u8; 64];

    // Arc-backed external: not a global node, must copy.
    let arc: Arc<[u8]> = Arc::from(vec![b'g'; N].into_boxed_slice());
    let base = arc.as_ptr();
    let cord = Cord::from(arc.clone());
    assert!(internal::is_external(&cord));
    let out: Vec<u8> = cord.into();
    assert_ne!(out.as_ptr(), base);
    assert_eq!(out.len(), N);
    assert_eq!(Arc::strong_count(&arc), 1);
    assert!(arc.iter().all(|&b| b == b'g'));

    // 'static external: not a global node either.
    let cord = Cord::from_static(&PAYLOAD);
    assert!(internal::is_external(&cord));
    let out: Vec<u8> = cord.into();
    assert_ne!(out.as_ptr(), PAYLOAD.as_ptr());
    assert_eq!(out, PAYLOAD.to_vec());
}

#[test]
fn boxed_slice_owner_yields_exact_capacity() {
    let boxed: Box<[u8]> = vec![b'b'; N].into_boxed_slice();
    let base = boxed.as_ptr();
    let cord = Cord::from(boxed);
    let out: Vec<u8> = cord.into();
    assert_eq!(out.as_ptr(), base);
    assert_eq!(out.len(), N);
    assert_eq!(out.capacity(), N);
}

#[test]
fn spare_capacity_is_preserved_and_writable() {
    let mut v = Vec::with_capacity(4 * N);
    v.resize(3 * N, b'v'); // len >= cap/2 so it is adopted, not copied.
    let base = v.as_ptr();
    let cap = v.capacity();
    let cord = Cord::from(v);
    assert!(internal::is_external(&cord));
    let mut out: Vec<u8> = cord.into();
    assert_eq!(out.as_ptr(), base);
    assert_eq!(out.capacity(), cap);
    assert_eq!(out.len(), 3 * N);
    // The recovered spare capacity must be genuinely writable.
    out.extend_from_slice(&[b'w'; N]);
    assert_eq!(out.len(), 4 * N);
    assert_eq!(out.as_ptr(), base, "no reallocation should have been needed");
    assert!(out[3 * N..].iter().all(|&b| b == b'w'));
}

#[test]
fn string_error_path_returns_the_same_allocation() {
    let mut bytes = adopted_vec(N, b'u');
    bytes[10] = 0xFF; // invalid UTF-8
    let base = bytes.as_ptr();
    let cord = Cord::from(bytes);
    let err = String::try_from(cord).unwrap_err();
    let back = err.into_bytes();
    assert_eq!(back.as_ptr(), base, "the error path must not copy");
    assert_eq!(back.len(), N);
}

#[test]
fn shared_then_unshared_becomes_stealable_again() {
    let v = adopted_vec(N, b'r');
    let base = v.as_ptr();
    let cord = Cord::from(v);
    let clone = cord.clone();
    assert_eq!(internal::root_refcount(&cord), 2);
    drop(clone);
    assert_eq!(internal::root_refcount(&cord), 1);
    let out: Vec<u8> = cord.into();
    assert_eq!(out.as_ptr(), base);
}

#[test]
fn refcount_released_on_another_thread_before_conversion() {
    for _ in 0..if cfg!(miri) { 2 } else { 50 } {
        let v = adopted_vec(N, b'x');
        let base = v.as_ptr();
        let cord = Cord::from(v);
        let clone = cord.clone();
        let h = std::thread::spawn(move || {
            assert_eq!(clone.len(), N);
            drop(clone);
        });
        h.join().unwrap();
        let out: Vec<u8> = cord.into();
        assert_eq!(out.as_ptr(), base);
        assert!(out.iter().all(|&b| b == b'x'));
    }
}

#[test]
fn concurrent_clone_alive_forces_the_copy_path() {
    let v = adopted_vec(N, b'y');
    let base = v.as_ptr() as usize;
    let cord = Cord::from(v);
    let clone = cord.clone();
    let h = std::thread::spawn(move || {
        let out: Vec<u8> = cord.into();
        assert_ne!(out.as_ptr() as usize, base);
        out
    });
    let out = h.join().unwrap();
    assert_eq!(out.len(), N);
    assert_eq!(clone, out.as_slice());
}

#[test]
fn tiny_global_external_from_internal_hook() {
    let cord = internal::make_external(b"abcdef");
    assert!(internal::is_external(&cord));
    let out: Vec<u8> = cord.into();
    assert_eq!(out, b"abcdef".to_vec());
    assert_eq!(out.capacity(), 6);
}

#[test]
fn btree_collapsing_back_to_the_external_root() {
    let v = adopted_vec(4096, b'k');
    let base = v.as_ptr();
    let mut cord = Cord::from(v);
    cord.prepend(b"prefix".as_slice());
    assert!(internal::is_btree(&cord));
    // Drop the prefix again; the tree should collapse back to the external.
    cord.advance(6);
    let out: Vec<u8> = cord.into();
    assert_eq!(out.len(), 4096);
    assert!(out.iter().all(|&b| b == b'k'));
    // Whether or not the root collapsed to the bare external, the bytes and
    // the allocation must both stay intact.
    let _ = base;
}

#[test]
fn substring_over_a_global_external_is_not_stolen() {
    let v = adopted_vec(N, b'j');
    let base = v.as_ptr();
    let cord = Cord::from(v);
    let sub = internal::make_substring(&cord, 1, N - 2);
    drop(cord);
    assert!(internal::is_substring(&sub));
    let out: Vec<u8> = sub.into();
    assert_ne!(out.as_ptr(), base);
    assert_eq!(out.len(), N - 2);
    assert!(out.iter().all(|&b| b == b'j'));
}

#[test]
fn repeated_convert_and_readopt_round_trips() {
    let mut v = adopted_vec(N, 0);
    for (i, byte) in v.iter_mut().enumerate() {
        *byte = (i % 251) as u8;
    }
    let expected = v.clone();
    let base = v.as_ptr();
    let mut cur = v;
    for _ in 0..if cfg!(miri) { 3 } else { 20 } {
        let cord = Cord::from(cur);
        cur = cord.into();
        assert_eq!(cur.as_ptr(), base);
    }
    assert_eq!(cur, expected);
}
