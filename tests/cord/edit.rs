//! Append/prepend (slices, cords, buffers), `take_append_buffer*`,
//! advance/truncate/clear/split, `Write`/`Extend`, copy-on-write.
#![expect(clippy::cast_possible_truncation, reason = "tests juggle small integers freely")]

use std::fmt::Write as _;

use crate::common::{self, internal};
use cord_rs::{Cord, CordBuffer};

#[test]
fn mem_swap_exchanges_two_cords() {
    let mut x = Cord::from("Dexter");
    let mut y = Cord::from("Mandark");
    core::mem::swap(&mut x, &mut y);
    assert_eq!(x, Cord::from("Mandark"));
    assert_eq!(y, Cord::from("Dexter"));
    core::mem::swap(&mut x, &mut y);
    assert_eq!(x, Cord::from("Dexter"));
    assert_eq!(y, Cord::from("Mandark"));
}

#[test]
fn appending_empty_buffers_is_a_no_op() {
    let mut cord = Cord::new();
    cord.append(CordBuffer::new());
    cord.append(CordBuffer::with_capacity(2000));
    assert!(cord.is_empty());

    let mut cord = Cord::from(vec![b'x'; 2000]);
    cord.append(CordBuffer::new());
    cord.append(CordBuffer::with_capacity(2000));
    assert_eq!(cord.len(), 2000);

    let mut cord = Cord::from(vec![b'x'; 2000]);
    cord.append(vec![b'y'; 2000]);
    cord.append(CordBuffer::new());
    cord.append(CordBuffer::with_capacity(2000));
    assert_eq!(cord.len(), 4000);

    // Empty buffers are no-ops on the inline representation too, and
    // `prepend` behaves the same as `append`; small buffers are copied
    // inline.
    let mut cord = Cord::from("ab");
    cord.append(CordBuffer::new());
    cord.prepend(CordBuffer::with_capacity(1000));
    common::check(&cord, b"ab");
    let mut b = CordBuffer::new();
    b.put_slice(b"cd");
    cord.append(b);
    common::check(&cord, b"abcd");
    assert!(!internal::is_tree(&cord));
}

#[test]
fn small_buffers_coalesce_into_one_chunk() {
    let mut cord = Cord::new();
    let mut buffer = CordBuffer::with_capacity(3);
    assert!(buffer.capacity() <= 15);
    buffer.put_slice(b"Abc");
    cord.append(buffer);

    let mut buffer = CordBuffer::with_capacity(3);
    buffer.put_slice(b"defgh");
    cord.append(buffer);

    assert_eq!(cord.chunks().collect::<Vec<_>>(), vec![&b"Abcdefgh"[..]]);

    // Mirror image: prepend instead of append.
    let mut cord = Cord::new();
    let mut buffer = CordBuffer::with_capacity(3);
    assert!(buffer.capacity() <= 15);
    buffer.put_slice(b"Abc");
    cord.prepend(buffer);

    let mut buffer = CordBuffer::with_capacity(3);
    buffer.put_slice(b"defgh");
    cord.prepend(buffer);

    assert_eq!(cord.chunks().collect::<Vec<_>>(), vec![&b"defghAbc"[..]]);
}

#[test]
fn large_buffers_stay_separate_chunks() {
    let mut cord = Cord::new();
    let s1 = vec![b'1'; 700];
    let mut buffer = CordBuffer::with_capacity(s1.len());
    buffer.put_slice(&s1);
    cord.append(buffer);

    let s2 = vec![b'2'; 1000];
    let mut buffer = CordBuffer::with_capacity(s2.len());
    buffer.put_slice(&s2);
    cord.append(buffer);

    assert_eq!(cord.chunks().collect::<Vec<_>>(), vec![&s1[..], &s2[..]]);

    // Mirror image: prepend instead of append.
    let mut cord = Cord::new();
    let s1 = vec![b'1'; 700];
    let mut buffer = CordBuffer::with_capacity(s1.len());
    buffer.put_slice(&s1);
    cord.prepend(buffer);

    let s2 = vec![b'2'; 1000];
    let mut buffer = CordBuffer::with_capacity(s2.len());
    buffer.put_slice(&s2);
    cord.prepend(buffer);

    assert_eq!(cord.chunks().collect::<Vec<_>>(), vec![&s2[..], &s1[..]]);
}

/// The two ways to ask a cord for an append buffer: with the default limit,
/// and with an explicit maximum block size.
struct AppendBufferLimit {
    custom: bool,
}

impl AppendBufferLimit {
    const ALL: [Self; 2] = [Self { custom: false }, Self { custom: true }];

    fn block_size(&self) -> usize {
        if self.custom { CordBuffer::MAX_BLOCK_SIZE } else { 0 }
    }

    fn max_capacity(&self) -> usize {
        if self.custom {
            CordBuffer::max_capacity_for(self.block_size())
        } else {
            CordBuffer::DEFAULT_MAX_CAPACITY
        }
    }

    fn take(&self, cord: &mut Cord, capacity: usize, min_capacity: usize) -> CordBuffer {
        cord.take_append_buffer_with(self.block_size(), capacity, min_capacity)
    }
}

#[test]
fn append_buffer_from_an_empty_cord() {
    for p in AppendBufferLimit::ALL {
        let mut cord = Cord::new();
        let buffer = p.take(&mut cord, 1000, 16);
        assert!(buffer.capacity() >= 1000);
        assert_eq!(buffer.len(), 0);
    }
}

#[test]
fn append_buffer_takes_over_inline_data() {
    let inlined_size = core::mem::size_of::<CordBuffer>() - 1;
    for p in AppendBufferLimit::ALL {
        for size in [6, inlined_size - 3, inlined_size - 2, 1000] {
            let mut cord = Cord::from("Abc");
            let buffer = p.take(&mut cord, size, 1);
            assert!(buffer.capacity() >= 3 + size);
            assert_eq!(buffer.len(), 3);
            assert_eq!(&*buffer, b"Abc");
            assert!(cord.is_empty());
        }
    }
}

#[test]
fn append_buffer_capacity_near_usize_max_does_not_overflow() {
    // Asking for something like `usize::MAX - k` must not overflow on
    // `usize::MAX - k + size` and must return the maximum allowed size.
    for p in AppendBufferLimit::ALL {
        for dist_from_max in 0..=4usize {
            let mut cord = Cord::from("Abc");
            let size = usize::MAX - dist_from_max;
            let buffer = p.take(&mut cord, size, 1);
            assert!(buffer.capacity() >= p.max_capacity());
            assert_eq!(buffer.len(), 3);
            assert_eq!(&*buffer, b"Abc");
            assert!(cord.is_empty());
        }
    }
}

#[test]
fn append_buffer_reuses_a_unique_flats_spare_capacity() {
    for p in AppendBufferLimit::ALL {
        // Create a cord with a single flat and extra capacity.
        let mut cord = Cord::new();
        let mut buffer = CordBuffer::with_capacity(500);
        let expected_capacity = buffer.capacity();
        buffer.put_slice(b"Abc");
        cord.append(buffer);

        let buffer = p.take(&mut cord, 6, 16);
        assert_eq!(buffer.capacity(), expected_capacity);
        assert_eq!(buffer.len(), 3);
        assert_eq!(&*buffer, b"Abc");
        assert!(cord.is_empty());
    }
}

#[test]
fn append_buffer_allocates_when_the_flat_misses_the_minimum() {
    for p in AppendBufferLimit::ALL {
        let mut cord = Cord::new();
        let mut buffer = CordBuffer::with_capacity(500);
        buffer.put_slice(&[b'x'; 30]);
        cord.append(buffer);

        let buffer = p.take(&mut cord, 1000, 900);
        assert!(buffer.capacity() >= 1000);
        assert_eq!(buffer.len(), 0);
        assert_eq!(cord, vec![b'x'; 30]);
    }
}

#[test]
fn append_buffer_takes_the_trees_last_flat() {
    let mut rng = common::Rng::new(7);
    for p in AppendBufferLimit::ALL {
        for num_flats in [2, 3, 100] {
            // Create a cord with `num_flats` flats and extra capacity.
            let mut cord = Cord::new();
            let mut prefix = Vec::new();
            let mut last = Vec::new();
            for _ in 0..num_flats - 1 {
                prefix.extend_from_slice(&last);
                last = rng.lowercase(10);
                let mut buffer = CordBuffer::with_capacity(500);
                buffer.put_slice(&last);
                cord.append(buffer);
            }
            let buffer = p.take(&mut cord, 6, 16);
            assert!(buffer.capacity() >= 500);
            assert_eq!(buffer.len(), 10);
            assert_eq!(&*buffer, &last[..]);
            assert_eq!(cord, prefix);
        }
    }
}

#[test]
fn append_buffer_on_a_tree_allocates_when_the_tail_misses_the_minimum() {
    for p in AppendBufferLimit::ALL {
        let mut cord = Cord::new();
        for i in 0..2 {
            let mut buffer = CordBuffer::with_capacity(500);
            buffer.put_slice(if i != 0 { b"def" } else { b"Abc" });
            cord.append(buffer);
        }
        let buffer = p.take(&mut cord, 1000, 900);
        assert!(buffer.capacity() >= 1000);
        assert_eq!(buffer.len(), 0);
        assert_eq!(cord, "Abcdef");
    }
}

#[test]
fn append_buffer_is_denied_on_a_substring() {
    for p in AppendBufferLimit::ALL {
        // A large cord with a single flat and some extra capacity.
        let mut cord = Cord::new();
        let mut buffer = CordBuffer::with_capacity(500);
        buffer.put_slice(&[b'x'; 450]);
        cord.append(buffer);
        cord.advance(1);

        // Denied on a substring.
        let buffer = p.take(&mut cord, 6, 16);
        assert_eq!(buffer.len(), 0);
        assert_eq!(cord, vec![b'x'; 449]);
    }
}

#[test]
fn append_buffer_is_denied_when_the_tail_is_shared() {
    for p in AppendBufferLimit::ALL {
        // A shared cord with a single flat and extra capacity.
        let mut cord = Cord::new();
        let mut buffer = CordBuffer::with_capacity(500);
        buffer.put_slice(b"Abc");
        cord.append(buffer);
        let _shared_cord = cord.clone();

        // Denied on a flat.
        let buffer = p.take(&mut cord, 6, 16);
        assert_eq!(buffer.len(), 0);
        assert_eq!(cord, "Abc");

        let mut buffer = CordBuffer::with_capacity(500);
        buffer.put_slice(b"def");
        cord.append(buffer);
        let _shared_cord = cord.clone();

        // Denied on a tree.
        let buffer = p.take(&mut cord, 6, 16);
        assert_eq!(buffer.len(), 0);
        assert_eq!(cord, "Abcdef");
    }
}

#[test]
fn truncate_wraps_an_external_then_adjusts_the_substring_in_place() {
    let mut cord = internal::make_external(b"foo bar baz");
    assert_eq!(cord.to_vec(), b"foo bar baz");
    // This truncate wraps the EXTERNAL node in a SUBSTRING node.
    cord.truncate(cord.len() - 4);
    assert_eq!(cord.to_vec(), b"foo bar");
    assert!(internal::is_substring(&cord));
    // This truncate adjusts the SUBSTRING node in place.
    cord.truncate(cord.len() - 4);
    assert_eq!(cord.to_vec(), b"foo");
}

#[test]
fn truncating_a_shared_append_restores_the_original() {
    let mut c = Cord::new();
    c.append(Cord::from(vec![b'x'; 100]));
    let other_ref = c.clone(); // Prevent in place appends.
    assert_eq!(other_ref, c);
    c.append(Cord::from(vec![b'y'; 200]));
    c.truncate(c.len() - 200);
    assert_eq!(c.to_vec(), vec![b'x'; 100]);
}

fn cord_with_zed_block(size: usize) -> Cord {
    internal::make_external(&vec![b'z'; size])
}

#[test]
fn advancing_within_and_past_an_external_block() {
    let blob = cord_with_zed_block(10);
    assert_eq!(blob.len(), 10);
    assert_eq!(blob.to_vec(), b"zzzzzzzzzz");

    let blob = cord_with_zed_block(0);
    assert_eq!(blob.len(), 0);
    assert_eq!(blob.to_vec(), b"");

    let blob = cord_with_zed_block(10);
    assert_eq!(blob.len(), 10);
    let mut suffix = blob.clone();
    suffix.advance(9);
    assert_eq!(suffix.len(), 1);
    assert_eq!(suffix.to_vec(), b"z");

    let blob = cord_with_zed_block(10);
    let mut suffix = blob.clone();
    suffix.advance(10);
    assert_eq!(suffix.len(), 0);
    assert_eq!(suffix.to_vec(), b"");
}

fn big_cord(len: usize, v: u8) -> Cord {
    Cord::from(&vec![v; len][..])
}

/// Splices `block` into `blob` at `offset`.
fn splice_cord(blob: &Cord, offset: usize, block: &Cord) -> Cord {
    assert!(offset + block.len() <= blob.len());
    let mut result = blob.clone();
    result.truncate(offset);
    result.append(block);
    let mut suffix = blob.clone();
    suffix.advance(offset + block.len());
    result.append(suffix);
    assert_eq!(blob.len(), result.len());
    result
}

#[test]
fn splicing_a_block_over_the_whole_cord() {
    let zero = cord_with_zed_block(10);
    let mut suffix = zero.clone();
    suffix.advance(10);
    let mut result = Cord::new();
    result.append(suffix);
    assert!(result.is_empty());

    let zero = cord_with_zed_block(10);
    let mut prefix = zero.clone();
    prefix.truncate(0);
    let mut suffix = zero.clone();
    suffix.advance(10);
    let mut result = prefix.clone();
    result.append(suffix);
    assert!(result.is_empty());

    let blob = cord_with_zed_block(10);
    let block = big_cord(10, b'b');
    let blob = splice_cord(&blob, 0, &block);
    assert_eq!(blob, "bbbbbbbbbb");
}

#[test]
fn advance_and_truncate_reach_every_substring() {
    // Exhaustively try all sub-strings.
    let cord = common::make_composite_cord();
    let s = cord.to_vec();
    for offset in 0..=s.len() {
        for length in 0..=(s.len() - offset) {
            let mut result = cord.clone();
            result.advance(offset);
            result.truncate(length);
            assert_eq!(result.to_vec(), &s[offset..offset + length], "{offset} {length}");
        }
    }
}

/// `append` works when handed a reference to (a clone of) itself.
#[test]
fn appending_a_clone_of_itself_doubles_the_cord() {
    let mut empty = Cord::new();
    let copy = empty.clone();
    empty.append(copy);
    assert_eq!(empty, "");

    // Run until the data is ~16K, covering small, medium and large data.
    let mut control_data = b"Abc".to_vec();
    let mut data = Cord::from(&control_data[..]);
    while control_data.len() < 0x4000 {
        let copy = data.clone();
        data.append(copy);
        control_data.extend_from_within(..);
        assert_eq!(data, control_data);
    }
}

#[test]
fn truncate_past_the_end_is_a_no_op() {
    // Unlike the C++ original's RemoveSuffix, truncate follows `Vec::truncate`.
    let mut cord = Cord::from("hello");
    cord.truncate(6);
    assert_eq!(cord, "hello");
}

#[test]
fn interleaved_appends_and_prepends_build_a_btree() {
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
            common::check(&cord, &expected);
        }
    }
    common::check(&cord, &expected);
    assert!(internal::is_btree(&cord));

    // Append a cord to itself via clone; append owned.
    let clone = cord.clone();
    cord.append(&clone);
    expected.extend_from_within(..);
    common::check(&cord, &expected);
    cord.append(clone);
    expected.extend_from_within(..expected.len() / 2);
    common::check(&cord, &expected);
    cord.prepend(Cord::from("prefix"));
    let mut e = b"prefix".to_vec();
    e.extend_from_slice(&expected);
    expected = e;
    common::check(&cord, &expected);
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
        common::check(&cord, &expected);
        cord.prepend(&piece[..]);
        let mut e = piece.clone();
        e.extend_from_slice(&expected);
        expected = e;
        common::check(&cord, &expected);
    }
}

#[test]
fn advance_truncate_and_split_a_multi_chunk_cord() {
    let len: u32 = if cfg!(miri) { 6_000 } else { 50_000 };
    let data: Vec<u8> = (0..len).map(|i| (i % 253) as u8).collect();
    let mut cord = Cord::new();
    for chunk in data.chunks(1000) {
        cord.append(chunk);
    }
    common::check(&cord, &data);

    let n = data.len();

    let mut c = cord.clone();
    c.advance(0);
    common::check(&c, &data);
    c.advance(1);
    common::check(&c, &data[1..]);
    let adv = n * 2 / 5;
    c.advance(adv);
    common::check(&c, &data[adv + 1..]);
    c.truncate(n * 2);
    common::check(&c, &data[adv + 1..]);
    c.truncate(n / 5);
    common::check(&c, &data[adv + 1..adv + 1 + n / 5]);
    c.truncate(10);
    common::check(&c, &data[adv + 1..adv + 11]);
    c.advance(10);
    common::check(&c, b"");
    assert!(!internal::is_tree(&c));

    let mut c = cord.clone();
    let sp = n * 3 / 5;
    let tail = c.split_off(sp);
    common::check(&c, &data[..sp]);
    common::check(&tail, &data[sp..]);
    let head = c.split_to(1000);
    common::check(&head, &data[..1000]);
    common::check(&c, &data[1000..sp]);
    let all = c.split_to(c.len());
    common::check(&c, b"");
    common::check(&all, &data[1000..sp]);
    let none = c.split_off(0);
    common::check(&none, b"");

    // Inline variants.
    let mut small = Cord::from("hello world");
    small.advance(6);
    common::check(&small, b"world");
    small.truncate(3);
    common::check(&small, b"wor");
    let mut small = Cord::from("hello world");
    let w = small.split_off(6);
    common::check(&w, b"world");
    common::check(&small, b"hello ");
}

#[test]
fn filling_a_cord_through_append_buffers() {
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
    common::check(&cord, &expected);
}

#[test]
fn take_append_buffer_reuses_spare_capacity_when_it_can() {
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
    common::check(&cord, b"0123456789abcdefghijKLM");
    let mut buffer = cord.take_append_buffer(4000);
    assert!(buffer.is_empty(), "no spare capacity: fresh buffer");
    buffer.put_slice(b"...");
    cord.append(buffer);
    common::check(&cord, b"0123456789abcdefghijKLM...");
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
    common::check(&cord, &expected);
    let mut cord = Cord::from("0123456789abcdefghij");
    let mut buffer = cord.take_append_buffer_with(0, 10, 1);
    buffer.put_slice(b"KLM");
    cord.append(buffer);
    common::check(&cord, b"0123456789abcdefghijKLM");
    assert!(internal::is_flat(&cord));
}

#[test]
fn take_append_buffer_moves_inline_data_and_honours_a_custom_limit() {
    // Inline data moves into the buffer.
    let mut cord = Cord::from("abc");
    let mut buffer = cord.take_append_buffer(100);
    assert!(cord.is_empty());
    assert_eq!(&*buffer, b"abc");
    buffer.put_slice(b"def");
    cord.prepend(buffer);
    common::check(&cord, b"abcdef");

    // Custom limits.
    let mut cord = Cord::from(vec![1u8; 5000]);
    let buffer = cord.take_append_buffer_with(64 << 10, 100_000, 1);
    assert_eq!(buffer.capacity(), (64 << 10) - internal::FLAT_OVERHEAD);
    cord.append(buffer);
    common::check(&cord, &[1u8; 5000]);
}

#[test]
fn fmt_and_io_writes_append_to_the_cord() {
    let mut cord = Cord::new();
    std::fmt::Write::write_fmt(&mut cord, format_args!("{}-{}", 1, "two")).unwrap();
    #[cfg(feature = "std")]
    std::io::Write::write_all(&mut cord, b"|bytes").unwrap();
    #[cfg(not(feature = "std"))]
    cord.append(&b"|bytes"[..]);
    cord.extend(*b"!?");
    cord.extend(["a", "b"]);
    cord.extend(vec![vec![b'c'; 1000]]);
    let mut expected = b"1-two|bytes!?ab".to_vec();
    expected.extend_from_slice(&[b'c'; 1000]);
    common::check(&cord, &expected);

    // `write!`/`fmt::Write` with format specs.
    let mut c = Cord::new();
    write!(c, "There were {:04} little pigs.", 3).unwrap();
    assert_eq!(c, "There were 0003 little pigs.");
    write!(c, "And {:<3x} bad wolf!", 1).unwrap();
    assert_eq!(c, "There were 0003 little pigs.And 1   bad wolf!");

    // `io::Write::write`'s return value and `flush`.
    #[cfg(feature = "std")]
    {
        let mut cord = Cord::from("a");
        let n = std::io::Write::write(&mut cord, b"bc").unwrap();
        assert_eq!(n, 2);
        std::io::Write::flush(&mut cord).unwrap();
        common::check(&cord, b"abc");
    }
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
fn mutating_one_clone_leaves_the_others_unchanged() {
    let mut a = Cord::from(vec![1u8; 10_000]);
    let b = a.clone();
    assert_eq!(internal::root_refcount(&a), 2);
    a.append(vec![2u8; 100]);
    common::check(&b, &[1u8; 10_000]);
    let mut expected = vec![1u8; 10_000];
    expected.extend_from_slice(&[2u8; 100]);
    common::check(&a, &expected);
    let mut c = b.clone();
    c.clone_from(&a);
    common::check(&c, &expected);
    c.clone_from(&Cord::from("x"));
    common::check(&c, b"x");
    c.clone_from(&a);
    common::check(&c, &expected);
    let d = a.clone();
    drop(c);
    a.clear();
    common::check(&a, b"");
    common::check(&d, &expected);
    assert_eq!(internal::root_refcount(&d), 1);
    a = d.clone();
    a.truncate(5);
    common::check(&d, &expected);
    common::check(&a, &expected[..5]);
    let sub = d.slice(9990..10_050);
    drop(d);
    common::check(&sub, &expected[9990..10_050]);
}
