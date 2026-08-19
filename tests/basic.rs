//! Deterministic end-to-end tests of the public API.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    reason = "tests juggle small integers freely"
)]

use std::borrow::Cow;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::io::{BufRead, Read, Write};
use std::sync::Arc;

use cord_rs::{__internal as internal, Cord, CordBuffer, MemoryAccounting};

fn check(cord: &Cord, expected: &[u8]) {
    internal::validate(cord).unwrap_or_else(|e| panic!("{e}\n{}", internal::dump(cord, true)));
    assert_eq!(cord.len(), expected.len());
    assert_eq!(cord.is_empty(), expected.is_empty());
    assert_eq!(cord.to_vec(), expected);
    assert!(cord == expected, "eq via CordLike");
    let joined: Vec<u8> = cord.chunks().inspect(|c| assert!(!c.is_empty())).flatten().copied().collect();
    assert_eq!(joined, expected);
    let bytes: Vec<u8> = cord.bytes().collect();
    assert_eq!(bytes, expected);
    if let Some(flat) = cord.as_contiguous() {
        assert_eq!(flat, expected);
    }
}

#[test]
fn empty_and_inline() {
    let cord = Cord::new();
    check(&cord, b"");
    assert_eq!(cord.as_contiguous(), Some(&b""[..]));
    assert!(!internal::is_tree(&cord));
    let cord = Cord::from("hello");
    check(&cord, b"hello");
    assert!(!internal::is_tree(&cord));
    let cord = Cord::from("123456789012345");
    check(&cord, b"123456789012345");
    assert!(!internal::is_tree(&cord));
    let cord = Cord::from("1234567890123456");
    check(&cord, b"1234567890123456");
    assert!(internal::is_tree(&cord));
    assert!(internal::is_flat(&cord));
    assert_eq!(core::mem::size_of::<Cord>(), 16);
    assert_eq!(core::mem::size_of::<Option<Cord>>(), 16 + core::mem::size_of::<usize>());
}

#[test]
fn construction_from_various_sources() {
    static STATIC: [u8; 100] = [9; 100];
    let big: Vec<u8> = (0..10_000u32).map(|i| (i % 251) as u8).collect();
    check(&Cord::from(&big[..]), &big);
    check(&Cord::from(big.clone()), &big);
    check(&Cord::from(big.clone().into_boxed_slice()), &big);
    check(&Cord::from(Arc::<[u8]>::from(&big[..])), &big);
    check(&Cord::from(String::from_utf8_lossy(&big).into_owned()), String::from_utf8_lossy(&big).as_bytes());
    check(&Cord::copy_from_slice(&big), &big);
    check(&big.iter().copied().collect::<Cord>(), &big);
    check(&big.chunks(7).collect::<Cord>(), &big);
    let owned = Cord::from(big.clone());
    assert!(internal::is_external(&owned), "large vec should be adopted");
    let string = "x".repeat(10_000);
    let string_ptr = string.as_ptr();
    let owned = Cord::from(string);
    assert!(internal::is_external(&owned), "large string should be adopted");
    assert_eq!(owned.as_contiguous().unwrap().as_ptr(), string_ptr);
    let boxed = vec![7u8; 10_000].into_boxed_slice();
    let boxed_ptr = boxed.as_ptr();
    let owned = Cord::from(boxed);
    assert!(internal::is_external(&owned), "large boxed slice should be adopted");
    assert_eq!(owned.as_contiguous().unwrap().as_ptr(), boxed_ptr);
    let small = Cord::from(vec![1u8; 100]);
    assert!(internal::is_flat(&small), "small vec should be copied into a flat");
    assert!(internal::is_flat(&Cord::from(vec![1u8; 511])));
    assert!(internal::is_external(&Cord::from(vec![1u8; 512])));
    assert!(internal::is_flat(&Cord::from("x".repeat(511))));
    assert!(internal::is_external(&Cord::from("x".repeat(512))));
    assert!(internal::is_flat(&Cord::from(vec![1u8; 511].into_boxed_slice())));
    assert!(internal::is_external(&Cord::from(vec![1u8; 512].into_boxed_slice())));
    let wasteful = {
        let mut v = Vec::with_capacity(100_000);
        v.extend_from_slice(&big);
        v
    };
    assert!(!internal::is_external(&Cord::from(wasteful)), "wasteful vec is copied, not adopted");
    let s = Cord::from_static(&STATIC);
    check(&s, &STATIC);
    assert!(internal::is_external(&s));
    assert_eq!(s.as_contiguous().unwrap().as_ptr(), STATIC.as_ptr(), "from_static must not copy");
    check(&Cord::from_static("static str"), b"static str");
    let arc: Arc<str> = Arc::from("x".repeat(1000).as_str());
    let c = Cord::from(arc.clone());
    assert_eq!(Arc::strong_count(&arc), 2);
    drop(c);
    assert_eq!(Arc::strong_count(&arc), 1);

    // `Cow<[u8]>`: `Borrowed` always copies (`From<&[u8]>`), `Owned` follows
    // the `Vec<u8>` adopt-or-copy threshold (`From<Vec<u8>>`).
    let small_bytes = b"cow borrowed".to_vec();
    check(&Cord::from(Cow::Borrowed(&small_bytes[..])), &small_bytes);
    let borrowed_big: Cow<'_, [u8]> = Cow::Borrowed(&big[..]);
    let from_borrowed_big = Cord::from(borrowed_big);
    check(&from_borrowed_big, &big);
    assert!(!internal::is_external(&from_borrowed_big), "borrowed slice is always copied");
    let owned_vec = vec![4u8; 10_000];
    let owned_vec_ptr = owned_vec.as_ptr();
    let owned_bytes: Cow<'_, [u8]> = Cow::Owned(owned_vec);
    let from_owned_vec = Cord::from(owned_bytes);
    assert!(internal::is_external(&from_owned_vec), "owned vec via Cow should be adopted");
    assert_eq!(from_owned_vec.as_contiguous().unwrap().as_ptr(), owned_vec_ptr);

    // `Cow<str>`: same, via `From<&str>` / `From<String>`.
    let small_str = "cow borrowed str".to_string();
    check(&Cord::from(Cow::Borrowed(small_str.as_str())), small_str.as_bytes());
    let big_str = "z".repeat(10_000);
    let borrowed_big_str: Cow<'_, str> = Cow::Borrowed(&big_str);
    let from_borrowed_big_str = Cord::from(borrowed_big_str);
    check(&from_borrowed_big_str, big_str.as_bytes());
    assert!(!internal::is_external(&from_borrowed_big_str), "borrowed str is always copied");
    let owned_string = "w".repeat(10_000);
    let owned_string_ptr = owned_string.as_ptr();
    let owned_str: Cow<'_, str> = Cow::Owned(owned_string);
    let from_owned_string = Cord::from(owned_str);
    assert!(internal::is_external(&from_owned_string), "owned string via Cow should be adopted");
    assert_eq!(from_owned_string.as_contiguous().unwrap().as_ptr(), owned_string_ptr);

    // `FromStr` never fails; it's equivalent to `Cord::from`.
    assert_eq!("parsed text".parse::<Cord>().unwrap(), Cord::from("parsed text"));
}

#[test]
fn append_and_prepend_growth() {
    let mut cord = Cord::new();
    let mut expected = Vec::new();
    let iters: usize = if cfg!(miri) { 200 } else { 2000 };
    for i in 0..iters {
        let piece = vec![(i % 256) as u8; i % 37 + 1];
        if i % 3 == 0 {
            cord.prepend(&piece[..]);
            let mut e = piece.clone();
            e.extend_from_slice(&expected);
            expected = e;
        } else {
            cord.append(&piece[..]);
            expected.extend_from_slice(&piece);
        }
        if i % 100 == 0 {
            check(&cord, &expected);
        }
    }
    check(&cord, &expected);
    assert!(internal::is_btree(&cord));

    // Append a cord to itself via clone; append owned.
    let clone = cord.clone();
    cord.append(&clone);
    expected.extend_from_within(..);
    check(&cord, &expected);
    cord.append(clone);
    expected.extend_from_within(..expected.len() / 2);
    check(&cord, &expected);
    cord.prepend(Cord::from("prefix"));
    let mut e = b"prefix".to_vec();
    e.extend_from_slice(&expected);
    expected = e;
    check(&cord, &expected);
}

#[test]
fn large_appends_cross_flat_boundaries() {
    let mut cord = Cord::from("start");
    let mut expected = b"start".to_vec();
    let max_flat = internal::MAX_FLAT_LENGTH;
    let sizes: &[usize] = if cfg!(miri) {
        &[1, 15, 16, max_flat - 1, max_flat, max_flat + 1, 10_000]
    } else {
        &[1, 15, 16, max_flat - 1, max_flat, max_flat + 1, 10_000, 100_000, 300_000]
    };
    for &size in sizes {
        let piece: Vec<u8> = (0..size).map(|i| (i * 7 % 256) as u8).collect();
        cord.append(&piece[..]);
        expected.extend_from_slice(&piece);
        check(&cord, &expected);
        cord.prepend(&piece[..]);
        let mut e = piece.clone();
        e.extend_from_slice(&expected);
        expected = e;
        check(&cord, &expected);
    }
}

#[test]
fn advance_truncate_slice_split() {
    let len: u32 = if cfg!(miri) { 6_000 } else { 50_000 };
    let data: Vec<u8> = (0..len).map(|i| (i % 253) as u8).collect();
    let mut cord = Cord::new();
    for chunk in data.chunks(1000) {
        cord.append(chunk);
    }
    check(&cord, &data);

    let n = data.len();
    for (start, end) in [
        (0, 0),
        (0, 5),
        (0, 15),
        (0, 16),
        (10, 20),
        (999, 1001),
        (n / 10, n - n / 10),
        (0, n),
        (n - 10, n),
        (n / 4, n / 4),
    ] {
        let sub = cord.slice(start..end);
        check(&sub, &data[start..end]);
        assert_eq!(cord.get(start..end).unwrap(), sub);
    }
    assert!(cord.get(10..=n).is_none());
    #[allow(clippy::reversed_empty_ranges)]
    let reversed = 11..10;
    assert!(cord.get(reversed).is_none());
    check(&cord.slice(..), &data);
    check(&cord.slice(100..=200), &data[100..=200]);

    let mut c = cord.clone();
    c.advance(0);
    check(&c, &data);
    c.advance(1);
    check(&c, &data[1..]);
    let adv = n * 2 / 5;
    c.advance(adv);
    check(&c, &data[adv + 1..]);
    c.truncate(n * 2);
    check(&c, &data[adv + 1..]);
    c.truncate(n / 5);
    check(&c, &data[adv + 1..adv + 1 + n / 5]);
    c.truncate(10);
    check(&c, &data[adv + 1..adv + 11]);
    c.advance(10);
    check(&c, b"");
    assert!(!internal::is_tree(&c));

    let mut c = cord.clone();
    let sp = n * 3 / 5;
    let tail = c.split_off(sp);
    check(&c, &data[..sp]);
    check(&tail, &data[sp..]);
    let head = c.split_to(1000);
    check(&head, &data[..1000]);
    check(&c, &data[1000..sp]);
    let all = c.split_to(c.len());
    check(&c, b"");
    check(&all, &data[1000..sp]);
    let none = c.split_off(0);
    check(&none, b"");

    // Inline variants.
    let mut small = Cord::from("hello world");
    small.advance(6);
    check(&small, b"world");
    small.truncate(3);
    check(&small, b"wor");
    let mut small = Cord::from("hello world");
    let w = small.split_off(6);
    check(&w, b"world");
    check(&small, b"hello ");
    check(&Cord::from("hello").slice(1..4), b"ell");
}

#[test]
#[should_panic(expected = "cannot advance past end")]
fn advance_out_of_bounds_panics() {
    let mut c = Cord::from("abc");
    c.advance(4);
}

#[test]
#[should_panic(expected = "range end index 4 out of range for slice of length 3")]
fn slice_out_of_bounds_panics() {
    let c = Cord::from("abc");
    let _ = c.slice(2..4);
}

#[test]
#[should_panic(expected = "index out of bounds")]
fn index_out_of_bounds_panics() {
    let c = Cord::from("abc");
    let _ = c[3];
}

#[test]
fn indexing_and_get() {
    let len: u32 = if cfg!(miri) { 4_000 } else { 20_000 };
    let data: Vec<u8> = (0..len).map(|i| (i * 13 % 256) as u8).collect();
    let mut cord = Cord::new();
    for chunk in data.chunks(300) {
        cord.append(chunk);
    }
    for i in (0..data.len()).step_by(97) {
        assert_eq!(cord[i], data[i]);
        assert_eq!(cord.get(i), Some(data[i]));
    }
    assert_eq!(cord.get(data.len()), None);
    let inline = Cord::from("abc");
    assert_eq!(inline[1], b'b');
    assert_eq!(inline.get(3), None);
    let sub = cord.slice(1000..1100);
    for i in 0..100 {
        assert_eq!(sub[i], data[1000 + i]);
    }
}

/// Runs the same `get` battery against `cord`/`data`, covering every
/// implemented [`cord_rs::CordIndex`] type: `usize` and each range type, in
/// and out of bounds, checked against `[u8]::get` (or, for the `(Bound,
/// Bound)` tuple form, against the equivalent `Range`).
fn check_get_index_types(cord: &Cord, data: &[u8]) {
    use core::ops::Bound;

    let len = data.len();
    assert_eq!(cord.len(), len);

    // `usize`: in bounds (also cross-checked against sequential iteration)
    // and out of bounds.
    for &i in &[0, 1, len / 2, len - 1] {
        assert_eq!(cord.get(i), Some(data[i]));
        assert_eq!(cord.get(i), cord.bytes().nth(i));
    }
    assert_eq!(cord.get(len), None);
    assert_eq!(cord.get(len + 1), None);
    assert_eq!(cord.get(usize::MAX), None);

    // `Range<usize>`: empty, whole, interior, `start == end == len`, `start
    // > end`, `end > len`.
    for (start, end) in [(0, 0), (0, len), (1, len - 1), (len, len), (2, 1), (0, len + 1)] {
        assert_eq!(
            cord.get(start..end).as_ref().map(Cord::to_vec),
            data.get(start..end).map(<[u8]>::to_vec),
            "Range {start}..{end}"
        );
    }

    // `RangeInclusive<usize>`: valid up to the last index, one past the end
    // (invalid), and `end == usize::MAX` (must not overflow, matching
    // `[u8]::get`'s own overflow guard).
    assert_eq!(cord.get(0..=len - 1).as_ref().map(Cord::to_vec), data.get(0..=len - 1).map(<[u8]>::to_vec));
    assert_eq!(cord.get(0..=len).as_ref().map(Cord::to_vec), data.get(0..=len).map(<[u8]>::to_vec));
    assert_eq!(
        cord.get(0..=usize::MAX).as_ref().map(Cord::to_vec),
        data.get(0..=usize::MAX).map(<[u8]>::to_vec)
    );

    // `RangeFrom<usize>`: `start == len` yields an empty result, `start >
    // len` yields `None`.
    assert_eq!(cord.get(0..).as_ref().map(Cord::to_vec), Some(data.to_vec()));
    assert_eq!(cord.get(len..).as_ref().map(Cord::to_vec), Some(Vec::new()));
    assert!(cord.get(len + 1..).is_none());

    // `RangeTo<usize>`.
    assert_eq!(cord.get(..len).as_ref().map(Cord::to_vec), Some(data.to_vec()));
    assert_eq!(cord.get(..0).as_ref().map(Cord::to_vec), Some(Vec::new()));
    assert!(cord.get(..len + 1).is_none());

    // `RangeToInclusive<usize>`, at the boundary.
    assert_eq!(cord.get(..=len - 1).as_ref().map(Cord::to_vec), Some(data.to_vec()));
    assert!(cord.get(..=len).is_none());

    // `RangeFull`.
    assert_eq!(cord.get(..).as_ref().map(Cord::to_vec), Some(data.to_vec()));

    // `(Bound<usize>, Bound<usize>)`, including an excluded start, and
    // overflow safety at `usize::MAX`.
    assert_eq!(
        cord.get((Bound::Included(0), Bound::Excluded(len))).as_ref().map(Cord::to_vec),
        Some(data.to_vec())
    );
    assert_eq!(
        cord.get((Bound::Excluded(0), Bound::Included(len - 1))).as_ref().map(Cord::to_vec),
        data.get(1..len).map(<[u8]>::to_vec)
    );
    assert_eq!(
        cord.get((Bound::<usize>::Unbounded, Bound::<usize>::Unbounded)).as_ref().map(Cord::to_vec),
        Some(data.to_vec())
    );
    assert!(cord.get((Bound::Excluded(usize::MAX), Bound::Unbounded)).is_none());
}

#[test]
fn get_over_index_types() {
    // Inline.
    let inline_data = b"0123456789".to_vec();
    let inline = Cord::from(inline_data.as_slice());
    assert!(!internal::is_tree(&inline));
    check_get_index_types(&inline, &inline_data);

    // Single flat.
    let flat_data: Vec<u8> = (0..200u32).map(|i| (i % 256) as u8).collect();
    let flat = Cord::copy_from_slice(&flat_data);
    assert!(internal::is_flat(&flat));
    check_get_index_types(&flat, &flat_data);

    // Multi-chunk btree.
    let btree_len: u32 = if cfg!(miri) { 4_000 } else { 20 * 1024 };
    let btree_data: Vec<u8> = (0..btree_len).map(|i| (i % 251) as u8).collect();
    let mut btree = Cord::new();
    for chunk in btree_data.chunks(1000) {
        btree.append(chunk);
    }
    assert!(internal::is_btree(&btree));
    check_get_index_types(&btree, &btree_data);
}

#[test]
fn comparison_and_search() {
    let mut a = Cord::new();
    for chunk in b"the quick brown fox jumps over the lazy dog".chunks(5) {
        a.append(chunk);
    }
    assert!(internal::is_btree(&a));
    let b = Cord::from("the quick brown fox jumps over the lazy dog");
    assert_eq!(a, b);
    assert_eq!(a, "the quick brown fox jumps over the lazy dog");
    assert_eq!(a, b"the quick brown fox jumps over the lazy dog");
    assert_eq!(a, b"the quick brown fox jumps over the lazy dog".to_vec());
    assert!(a != "the quick brown fox jumps over the lazy dog!");
    assert_eq!("the quick brown fox jumps over the lazy dog", a);
    assert!(a < "the quick brown fox jumps over the lazy dog!");
    assert!(a < Cord::from("the quick brown fox jumps over the lazy dog!").slice(..));
    assert!(a > "the quick brown fox");
    assert!(a < "the quick brown fox jumps over the lazy dog!");
    assert_eq!(a.cmp(&b), std::cmp::Ordering::Equal);
    assert_eq!(a.compare("the quick"), std::cmp::Ordering::Greater);
    assert_eq!(Cord::from("abc").compare("abd"), std::cmp::Ordering::Less);
    assert_eq!(Cord::from("").compare(""), std::cmp::Ordering::Equal);
    assert_eq!(Cord::from("").compare("a"), std::cmp::Ordering::Less);
    assert_eq!(Cord::from(b"\xff").compare("a"), std::cmp::Ordering::Greater);

    assert!(a.starts_with("the quick"));
    assert!(a.starts_with(&a.slice(..10)));
    assert!(!a.starts_with("quick"));
    assert!(a.ends_with("lazy dog"));
    assert!(a.ends_with(&b.slice(20..)));
    assert!(!a.ends_with("lazy cat"));
    assert!(a.contains("brown fox"));
    assert!(a.contains(""));
    assert!(!a.contains("brown cat"));
    assert_eq!(a.find("fox"), Some(16));
    assert_eq!(a.find(&Cord::from("jumps over")), Some(20));
    assert_eq!(a.find(&a.slice(20..30)), Some(20));
    assert_eq!(a.find("dog"), Some(40));
    assert_eq!(a.find("dogs"), None);
    assert_eq!(a.find("the lazy dog"), Some(31));
    assert_eq!(a.find(""), Some(0));
    assert_eq!(a.find(&b), Some(0));
    let mut needle = Cord::new();
    for c in b"over the lazy".chunks(2) {
        needle.append(c);
    }
    assert_eq!(a.find(&needle), Some(26));
}

#[test]
fn hash_is_structure_independent() {
    fn hash<H: Hash>(h: &H) -> u64 {
        let mut s = DefaultHasher::new();
        h.hash(&mut s);
        s.finish()
    }
    let len: u32 = if cfg!(miri) { 4_000 } else { 30_000 };
    let data: Vec<u8> = (0..len).map(|i| (i % 7) as u8).collect();
    let flat = Cord::from(&data[..]);
    let mut chunked = Cord::new();
    for chunk in data.chunks(333) {
        chunked.append(chunk);
    }
    let mut prepended = Cord::new();
    for chunk in data.rchunks(1001) {
        prepended.prepend(chunk);
    }
    assert_eq!(flat, chunked);
    assert_eq!(hash(&flat), hash(&chunked));
    assert_eq!(hash(&flat), hash(&prepended));
    assert_eq!(hash(&flat.slice(100..2000)), hash(&chunked.slice(100..2000)));
    assert_ne!(hash(&flat), hash(&flat.slice(..data.len() - 1)));
    assert_eq!(hash(&Cord::from("abc")), hash(&Cord::from(b"abc".repeat(100)).slice(..3)));
}

#[test]
fn iteration_and_cursor() {
    let len: u32 = if cfg!(miri) { 5_000 } else { 10_000 };
    let data: Vec<u8> = (0..len).map(|i| (i % 255) as u8).collect();
    let mut cord = Cord::new();
    for chunk in data.chunks(123) {
        cord.append(chunk);
    }
    let mut cursor = cord.cursor();
    assert_eq!(cursor.remaining(), data.len());
    assert_eq!(cursor.position(), 0);
    let first = cursor.read_cord(10);
    check(&first, &data[..10]);
    assert_eq!(cursor.position(), 10);
    cursor.advance(500);
    assert_eq!(cursor.position(), 510);
    let mid = cursor.read_cord(3000);
    check(&mid, &data[510..3510]);
    assert_eq!(cursor.peek(), Some(data[3510]));
    assert_eq!(cursor.next_byte(), Some(data[3510]));
    let rest: Vec<u8> = cursor.chunks().flatten().copied().collect();
    assert_eq!(rest, &data[3511..]);
    let last = cursor.read_cord(cursor.remaining());
    check(&last, &data[3511..]);
    assert!(!cursor.has_remaining());
    assert_eq!(cursor.read_cord(0), Cord::new());
    assert_eq!(cursor.next_byte(), None);

    // io::Read / BufRead.
    let mut cursor = cord.cursor();
    let mut buf = [0u8; 1000];
    cursor.read_exact(&mut buf).unwrap();
    assert_eq!(&buf[..], &data[..1000]);
    let first_chunk = cursor.fill_buf().unwrap().to_vec();
    assert!(!first_chunk.is_empty());
    cursor.consume(first_chunk.len());
    let mut rest = Vec::new();
    cursor.read_to_end(&mut rest).unwrap();
    assert_eq!(rest, &data[1000 + first_chunk.len()..]);

    // Iterator helpers.
    assert_eq!(cord.bytes().nth(4567), Some(data[4567]));
    assert_eq!(cord.bytes().count(), data.len());
    assert_eq!(cord.bytes().len(), data.len());
    // `Cursor` doesn't implement `Iterator` (see its doc comment); use
    // `advance`/`next_byte` for the same "skip then read one, then confirm
    // exhaustion" check `nth`/`next` performed before the removal.
    let mut c = cord.cursor();
    c.advance(data.len() - 1);
    assert_eq!(c.next_byte(), Some(data[data.len() - 1]));
    assert_eq!(c.next_byte(), None);
}

#[test]
fn make_contiguous_and_memory_usage() {
    let mut cord = Cord::new();
    for i in 0..100u8 {
        cord.append(vec![i; 100]);
    }
    let expected: Vec<u8> = (0..100u8).flat_map(|i| vec![i; 100]).collect();
    assert!(cord.as_contiguous().is_none());
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

    assert_eq!(cord.make_contiguous(), &expected[..]);
    assert!(cord.as_contiguous().is_some());
    check(&cord, &expected);
    assert!(internal::is_external(&cord), "10000 bytes > max flat length -> external");
    let mut small = Cord::from("a");
    small.append(vec![b'b'; 20]);
    small.append("c");
    assert_eq!(small.make_contiguous(), [b"a".as_slice(), &[b'b'; 20], b"c"].concat());
    assert!(internal::is_flat(&small));
    let mut inline = Cord::from("xyz");
    assert_eq!(inline.make_contiguous(), b"xyz");
    assert_eq!(Cord::new().estimated_memory_usage(MemoryAccounting::Total), 16);
}

#[test]
fn cord_buffer_roundtrip() {
    let mut cord = Cord::new();
    let mut expected = Vec::new();
    let mut n = 25_000usize;
    let mut first = true;
    while n > 0 {
        let mut buffer = if first { cord.take_append_buffer(n) } else { CordBuffer::with_capacity(n) };
        first = false;
        let count = buffer.available().min(n);
        let piece = vec![(n % 256) as u8; count];
        let before = buffer.len();
        buffer.put_slice(&piece);
        assert_eq!(buffer.len(), before + count);
        expected.extend_from_slice(&piece);
        cord.append(buffer);
        n -= count;
    }
    check(&cord, &expected);

    // Reusing spare capacity: the last flat is handed back with its data.
    let mut cord = Cord::from("0123456789abcdefghij");
    assert!(internal::is_flat(&cord));
    // The flat was sized to fit exactly (plus size class rounding), so the
    // default minimum of 16 spare bytes is not met and a fresh buffer is
    // returned; with a minimum of 1 the flat itself is handed out.
    let buffer = cord.take_append_buffer(10);
    assert!(buffer.is_empty() && buffer.capacity() >= 10);
    assert_eq!(cord.len(), 20);
    drop(buffer);
    let mut buffer = cord.take_append_buffer_with(0, 10, 1);
    assert!(cord.is_empty(), "the sole flat was extracted");
    assert_eq!(buffer.as_slice(), b"0123456789abcdefghij");
    assert!(buffer.available() >= 1);
    assert!(buffer.available() < 16);
    buffer.put_slice(b"KLM");
    cord.append(buffer);
    check(&cord, b"0123456789abcdefghijKLM");
    let mut buffer = cord.take_append_buffer(4000);
    assert!(buffer.is_empty(), "no spare capacity: fresh buffer");
    buffer.put_slice(b"...");
    cord.append(buffer);
    check(&cord, b"0123456789abcdefghijKLM...");
    // Appending past the first flat allocates the next one with amortized
    // (10%) extra capacity, which a later take_append_buffer can hand out.
    let mut cord = Cord::from("abc");
    cord.append(&[b'x'; 4100][..]);
    cord.append("y");
    let buffer = cord.take_append_buffer(10);
    assert!(!buffer.is_empty(), "amortized growth left spare capacity: {}", internal::dump(&cord, false));
    assert!(buffer.available() >= 16);
    let mut expected = b"abc".to_vec();
    expected.extend_from_slice(&[b'x'; 4100]);
    expected.push(b'y');
    let taken = buffer.len();
    cord.append(buffer);
    assert!(taken > 0);
    check(&cord, &expected);
    let mut cord = Cord::from("0123456789abcdefghij");
    let mut buffer = cord.take_append_buffer_with(0, 10, 1);
    buffer.put_slice(b"KLM");
    cord.append(buffer);
    check(&cord, b"0123456789abcdefghijKLM");
    assert!(internal::is_flat(&cord));

    // Inline data moves into the buffer.
    let mut cord = Cord::from("abc");
    let mut buffer = cord.take_append_buffer(100);
    assert!(cord.is_empty());
    assert_eq!(&*buffer, b"abc");
    buffer.put_slice(b"def");
    cord.prepend(buffer);
    check(&cord, b"abcdef");

    // Custom limits.
    let mut cord = Cord::from(vec![1u8; 5000]);
    let buffer = cord.take_append_buffer_with(64 << 10, 100_000, 1);
    assert_eq!(buffer.capacity(), (64 << 10) - internal::FLAT_OVERHEAD);
    cord.append(buffer);
    check(&cord, &[1u8; 5000]);

    // Empty buffers are no-ops; small buffers are copied inline.
    let mut cord = Cord::from("ab");
    cord.append(CordBuffer::new());
    cord.prepend(CordBuffer::with_capacity(1000));
    check(&cord, b"ab");
    let mut b = CordBuffer::new();
    b.put_slice(b"cd");
    cord.append(b);
    check(&cord, b"abcd");
    assert!(!internal::is_tree(&cord));
    let from_buffer: Cord = {
        let mut b = CordBuffer::with_capacity(100);
        b.put_slice(b"hello");
        b.into()
    };
    check(&from_buffer, b"hello");
}

#[test]
fn write_extend_and_formatting() {
    let mut cord = Cord::new();
    std::fmt::Write::write_fmt(&mut cord, format_args!("{}-{}", 1, "two")).unwrap();
    cord.write_all(b"|bytes").unwrap();
    cord.extend(*b"!?");
    cord.extend(["a", "b"]);
    cord.extend(vec![vec![b'c'; 1000]]);
    let mut expected = b"1-two|bytes!?ab".to_vec();
    expected.extend_from_slice(&[b'c'; 1000]);
    check(&cord, &expected);
    assert_eq!(format!("{:?}", Cord::from(b"a\"b\n\xff")), "b\"a\\\"b\\n\\xff\"");
    let s: String = Cord::from("héllo wörld").try_into().unwrap();
    assert_eq!(s, "héllo wörld");
    let s: String = Cord::from("utf8").try_into().unwrap();
    assert_eq!(s, "utf8");
    assert!(String::try_from(Cord::from(b"\xff")).is_err());
    let v: Vec<u8> = Cord::from("vec").into();
    assert_eq!(v, b"vec");

    // `Box<[u8]>` copies, for every tree shape.
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
fn extend_u8_preserves_consumed_bytes_on_panic() {
    // An iterator that panics partway through, after having already yielded
    // (been "consumed" by) `limit` bytes.
    struct PanicAfter {
        i: u32,
        limit: u32,
    }
    impl Iterator for PanicAfter {
        type Item = u8;
        fn next(&mut self) -> Option<u8> {
            assert!(self.i < self.limit, "PanicAfter: intentional panic for the test");
            let b = (self.i % 256) as u8;
            self.i += 1;
            Some(b)
        }
    }

    // `Extend<u8>` batches into a 256-byte block before appending; 300 is
    // past one full block, so the panic lands with an un-flushed partial
    // block still in the buffer, exactly the case the drop guard covers.
    let limit = 300u32;
    let mut cord = Cord::new();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        cord.extend(PanicAfter { i: 0, limit });
    }));
    assert!(result.is_err(), "extend should propagate the iterator's panic");
    let expected: Vec<u8> = (0..limit).map(|i| (i % 256) as u8).collect();
    assert_eq!(cord.to_vec(), expected, "bytes consumed before the panic must not be lost");
}

#[test]
fn clone_and_sharing_semantics() {
    let mut a = Cord::from(vec![1u8; 10_000]);
    let b = a.clone();
    assert_eq!(internal::root_refcount(&a), 2);
    a.append(vec![2u8; 100]);
    check(&b, &[1u8; 10_000]);
    let mut expected = vec![1u8; 10_000];
    expected.extend_from_slice(&[2u8; 100]);
    check(&a, &expected);
    let mut c = b.clone();
    c.clone_from(&a);
    check(&c, &expected);
    c.clone_from(&Cord::from("x"));
    check(&c, b"x");
    c.clone_from(&a);
    check(&c, &expected);
    let d = a.clone();
    drop(c);
    a.clear();
    check(&a, b"");
    check(&d, &expected);
    assert_eq!(internal::root_refcount(&d), 1);
    a = d.clone();
    a.truncate(5);
    check(&d, &expected);
    check(&a, &expected[..5]);
    let sub = d.slice(9990..10_050);
    drop(d);
    check(&sub, &expected[9990..10_050]);
}

#[test]
fn send_sync_across_threads() {
    let len: u32 = if cfg!(miri) { 2_000 } else { 100_000 };
    let num_threads: u32 = if cfg!(miri) { 3 } else { 8 };
    let data: Vec<u8> = (0..len).map(|i| (i % 200) as u8).collect();
    let cord = Cord::from(data.clone());
    let handles: Vec<_> = (0..num_threads)
        .map(|t| {
            let mut c = cord.clone();
            let data = data.clone();
            std::thread::spawn(move || {
                c.append(vec![t as u8; 1000]);
                c.prepend("x");
                let sub = c.slice(1..=data.len());
                assert_eq!(sub, data);
                c.len()
            })
        })
        .collect();
    for h in handles {
        assert_eq!(h.join().unwrap(), data.len() + 1001);
    }
    check(&cord, &data);
}

#[test]
fn default_impls() {
    let cord = Cord::default();
    check(&cord, b"");
    let default_buffer = CordBuffer::default();
    assert!(default_buffer.is_empty());
    assert_eq!(default_buffer.capacity(), CordBuffer::new().capacity());
}

#[test]
fn cordlike_for_cord_buffer() {
    let mut buffer = CordBuffer::with_capacity(32);
    buffer.put_slice(b"needle");
    let exact = Cord::from("needle");
    assert!(exact == buffer, "Cord should compare equal to a CordBuffer with the same bytes");
    assert_ne!(exact, CordBuffer::new());
    let haystack = Cord::from("a needle in a haystack");
    assert_eq!(haystack.find(&buffer), Some(2));
    assert!(haystack.contains(&buffer));
}

#[test]
fn from_boxed_str_arc_string_and_refs() {
    let boxed: Box<str> = "boxed str".into();
    check(&Cord::from(boxed), b"boxed str");

    let arc_string = Arc::new("arc string".to_string());
    check(&Cord::from(arc_string.clone()), b"arc string");

    // Large enough to be adopted (shared) rather than copied.
    let big_string = Arc::new("y".repeat(10_000));
    let before = Arc::strong_count(&big_string);
    let shared = Cord::from(big_string.clone());
    assert_eq!(Arc::strong_count(&big_string), before + 1);
    check(&shared, big_string.as_bytes());
    drop(shared);
    assert_eq!(Arc::strong_count(&big_string), before);

    let v: Vec<u8> = b"by-ref vec".to_vec();
    check(&Cord::from(&v), &v);
    let s: String = "by-ref string".to_string();
    check(&Cord::from(&s), s.as_bytes());
}

#[test]
fn chunks_and_bytes_iterator_exhaustion() {
    let data: Vec<u8> = (0..5000u32).map(|i| (i % 256) as u8).collect();
    let mut cord = Cord::new();
    for chunk in data.chunks(300) {
        cord.append(chunk);
    }
    assert!(internal::is_btree(&cord));

    let mut chunks = cord.chunks();
    assert_eq!(chunks.size_hint(), (1, Some(cord.len())));
    let mut total = 0;
    for chunk in chunks.by_ref() {
        total += chunk.len();
    }
    assert_eq!(total, cord.len());
    // Exhausted: FusedIterator guarantees repeated `None`, not a resumed walk.
    assert_eq!(chunks.next(), None);
    assert_eq!(chunks.next(), None);
    assert_eq!(chunks.size_hint(), (0, Some(0)));

    let mut bytes = cord.bytes();
    assert_eq!(bytes.len(), cord.len());
    assert_eq!(bytes.size_hint(), (cord.len(), Some(cord.len())));
    let mut count = 0;
    for _ in bytes.by_ref() {
        count += 1;
    }
    assert_eq!(count, cord.len());
    assert_eq!(bytes.next(), None);
    assert_eq!(bytes.next(), None);
    assert_eq!(bytes.len(), 0);
    assert_eq!(bytes.size_hint(), (0, Some(0)));
}

#[test]
fn io_write_for_cord() {
    let mut cord = Cord::from("a");
    let n = cord.write(b"bc").unwrap();
    assert_eq!(n, 2);
    cord.flush().unwrap();
    check(&cord, b"abc");
}

#[test]
fn io_write_for_cord_buffer_write_zero_on_full() {
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
fn btree_height_sanity() {
    assert_eq!(internal::btree_height(&Cord::new()), None);
    assert_eq!(internal::btree_height(&Cord::from("inline")), None);

    let mut cord = Cord::new();
    for i in 0..500u32 {
        cord.append(vec![(i % 256) as u8; 20]);
    }
    assert!(internal::is_btree(&cord));
    let height = internal::btree_height(&cord).expect("a btree cord must report a height");
    assert!((1..=6).contains(&height), "unexpected height {height} for a modest 500-leaf tree");
}

#[test]
fn constructor_space_with_capacity_and_block_size() {
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
