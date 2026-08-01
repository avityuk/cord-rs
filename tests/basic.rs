//! Deterministic end-to-end tests of the public API.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    reason = "tests juggle small integers freely"
)]

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::io::{BufRead, Read, Write};
use std::sync::Arc;

use cord_rs::{Cord, CordBuffer, MemoryAccounting, internal};

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
    if let Some(flat) = cord.as_flat() {
        assert_eq!(flat, expected);
    }
}

#[test]
fn empty_and_inline() {
    let cord = Cord::new();
    check(&cord, b"");
    assert_eq!(cord.as_flat(), Some(&b""[..]));
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
    let small = Cord::from(vec![1u8; 100]);
    assert!(internal::is_flat(&small), "small vec should be copied into a flat");
    let wasteful = {
        let mut v = Vec::with_capacity(100_000);
        v.extend_from_slice(&big);
        v
    };
    assert!(!internal::is_external(&Cord::from(wasteful)), "wasteful vec is copied, not adopted");
    let s = Cord::from_static(&STATIC);
    check(&s, &STATIC);
    assert!(internal::is_external(&s));
    assert_eq!(s.as_flat().unwrap().as_ptr(), STATIC.as_ptr(), "from_static must not copy");
    check(&Cord::from_static("static str"), b"static str");
    let arc: Arc<str> = Arc::from("x".repeat(1000).as_str());
    let c = Cord::from(arc.clone());
    assert_eq!(Arc::strong_count(&arc), 2);
    drop(c);
    assert_eq!(Arc::strong_count(&arc), 1);
}

#[test]
fn append_and_prepend_growth() {
    let mut cord = Cord::new();
    let mut expected = Vec::new();
    for i in 0..2000usize {
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
    for size in [1usize, 15, 16, 4082, 4083, 4084, 10_000, 100_000, 300_000] {
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
    let data: Vec<u8> = (0..50_000u32).map(|i| (i % 253) as u8).collect();
    let mut cord = Cord::new();
    for chunk in data.chunks(1000) {
        cord.append(chunk);
    }
    check(&cord, &data);

    for (start, end) in [
        (0, 0),
        (0, 5),
        (0, 15),
        (0, 16),
        (10, 20),
        (999, 1001),
        (5000, 45_000),
        (0, 50_000),
        (49_990, 50_000),
        (12_345, 12_345),
    ] {
        let sub = cord.slice(start..end);
        check(&sub, &data[start..end]);
        assert_eq!(cord.try_slice(start..end).unwrap(), sub);
    }
    assert!(cord.try_slice(10..50_001).is_none());
    #[allow(clippy::reversed_empty_ranges)]
    let reversed = 11..10;
    assert!(cord.try_slice(reversed).is_none());
    check(&cord.slice(..), &data);
    check(&cord.slice(100..=200), &data[100..=200]);

    let mut c = cord.clone();
    c.advance(0);
    check(&c, &data);
    c.advance(1);
    check(&c, &data[1..]);
    c.advance(20_000);
    check(&c, &data[20_001..]);
    c.truncate(100_000);
    check(&c, &data[20_001..]);
    c.truncate(10_000);
    check(&c, &data[20_001..30_001]);
    c.truncate(10);
    check(&c, &data[20_001..20_011]);
    c.advance(10);
    check(&c, b"");
    assert!(!internal::is_tree(&c));

    let mut c = cord.clone();
    let tail = c.split_off(30_000);
    check(&c, &data[..30_000]);
    check(&tail, &data[30_000..]);
    let head = c.split_to(1000);
    check(&head, &data[..1000]);
    check(&c, &data[1000..30_000]);
    let all = c.split_to(c.len());
    check(&c, b"");
    check(&all, &data[1000..30_000]);
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
#[should_panic(expected = "out of bounds")]
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
    let data: Vec<u8> = (0..20_000u32).map(|i| (i * 13 % 256) as u8).collect();
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
    let data: Vec<u8> = (0..30_000u32).map(|i| (i % 7) as u8).collect();
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
    let data: Vec<u8> = (0..10_000u32).map(|i| (i % 255) as u8).collect();
    let mut cord = Cord::new();
    for chunk in data.chunks(123) {
        cord.append(chunk);
    }
    let mut cursor = cord.cursor();
    assert_eq!(cursor.remaining(), data.len());
    assert_eq!(cursor.position(), 0);
    let first = cursor.read(10);
    check(&first, &data[..10]);
    assert_eq!(cursor.position(), 10);
    cursor.advance(500);
    assert_eq!(cursor.position(), 510);
    let mid = cursor.read(3000);
    check(&mid, &data[510..3510]);
    assert_eq!(cursor.peek(), Some(data[3510]));
    assert_eq!(cursor.next_byte(), Some(data[3510]));
    let rest: Vec<u8> = cursor.clone().collect();
    assert_eq!(rest, &data[3511..]);
    let last = cursor.read(cursor.remaining());
    check(&last, &data[3511..]);
    assert!(cursor.is_empty());
    assert_eq!(cursor.read(0), Cord::new());
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
    assert_eq!((&cord).into_iter().count(), cord.chunks().count());
    let mut c = cord.cursor();
    assert_eq!(c.nth(9999), Some(data[9999]));
    assert_eq!(c.next(), None);
}

#[test]
fn flatten_and_memory_usage() {
    let mut cord = Cord::new();
    for i in 0..100u8 {
        cord.append(vec![i; 100]);
    }
    let expected: Vec<u8> = (0..100u8).flat_map(|i| vec![i; 100]).collect();
    assert!(cord.as_flat().is_none());
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

    assert_eq!(cord.flatten(), &expected[..]);
    assert!(cord.as_flat().is_some());
    check(&cord, &expected);
    assert!(internal::is_external(&cord), "10000 bytes > max flat length -> external");
    let mut small = Cord::from("a");
    small.append(vec![b'b'; 20]);
    small.append("c");
    assert_eq!(small.flatten(), [b"a".as_slice(), &[b'b'; 20], b"c"].concat());
    assert!(internal::is_flat(&small));
    let mut inline = Cord::from("xyz");
    assert_eq!(inline.flatten(), b"xyz");
    assert_eq!(Cord::new().estimated_memory_usage(MemoryAccounting::Total), 16);
}

#[test]
fn cord_buffer_roundtrip() {
    let mut cord = Cord::new();
    let mut expected = Vec::new();
    let mut n = 25_000usize;
    let mut first = true;
    while n > 0 {
        let mut buffer = if first { cord.take_append_buffer(n) } else { CordBuffer::with_default_limit(n) };
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
    cord.prepend(CordBuffer::with_default_limit(1000));
    check(&cord, b"ab");
    let mut b = CordBuffer::new();
    b.put_slice(b"cd");
    cord.append(b);
    check(&cord, b"abcd");
    assert!(!internal::is_tree(&cord));
    let from_buffer: Cord = {
        let mut b = CordBuffer::with_default_limit(100);
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
    assert_eq!(Cord::from("héllo wörld").to_string(), "héllo wörld");
    assert_eq!(Cord::from(b"a\xffb").to_string(), "a\u{FFFD}b");
    // Multi-byte characters split across chunks decode correctly.
    let text = "🦀 ünïcödé 🦀 ".repeat(200);
    let bytes = text.as_bytes();
    for chunk_size in [1usize, 2, 3, 5, 7, 16, 100] {
        let mut c = Cord::new();
        for chunk in bytes.chunks(chunk_size) {
            c.append(chunk);
        }
        assert_eq!(c.to_string(), text, "chunk size {chunk_size}");
    }
    let mut invalid = Cord::new();
    for chunk in b"ok\xf0\x9f\xa6\x80x\xe2\x82y\xf0\x9fz\xff".chunks(3) {
        invalid.append(chunk);
    }
    assert_eq!(invalid.to_string(), String::from_utf8_lossy(&invalid.to_vec()));
    let s: String = Cord::from("utf8").try_into().unwrap();
    assert_eq!(s, "utf8");
    assert!(String::try_from(Cord::from(b"\xff")).is_err());
    let v: Vec<u8> = Cord::from("vec").into();
    assert_eq!(v, b"vec");
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
    let data: Vec<u8> = (0..100_000u32).map(|i| (i % 200) as u8).collect();
    let cord = Cord::from(data.clone());
    let handles: Vec<_> = (0..8)
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
