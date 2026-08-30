//! `chunks`/`bytes`/`cursor`, `read_cord`, `io::Read`/`BufRead`/`Seek`, bulk
//! copy-out.
#![expect(clippy::cast_possible_truncation, reason = "tests juggle small integers freely")]

#[cfg(feature = "std")]
use std::io::{BufRead, Read, Seek, SeekFrom};

use crate::common::{self, internal};
use cord_rs::Cord;

fn verify_append_cord_to_string(cord: &Cord) {
    let initial = b"initial contents.";
    let mut expected = initial.to_vec();
    expected.extend_from_slice(&cord.to_vec());
    let mut no_reserve = initial.to_vec();
    for chunk in cord.chunks() {
        no_reserve.extend_from_slice(chunk);
    }
    assert_eq!(no_reserve, expected);
    let mut has_reserved = Vec::with_capacity(initial.len() + cord.len());
    has_reserved.extend_from_slice(initial);
    let address_before = has_reserved.as_ptr();
    has_reserved.extend(cord.bytes());
    assert_eq!(has_reserved, expected);
    assert_eq!(has_reserved.as_ptr(), address_before);
}

#[test]
fn extending_a_vec_from_chunks_and_bytes() {
    verify_append_cord_to_string(&Cord::new());
    verify_append_cord_to_string(&Cord::from("small cord"));
    verify_append_cord_to_string(&common::make_fragmented_cord([
        "fragmented ",
        "cord ",
        "to ",
        "test ",
        "appending ",
        "to ",
        "a ",
        "string.",
    ]));
}

fn verify_copy_to_span(cord: &Cord) {
    // Span exactly the same size as the cord.
    {
        let mut dst = vec![0u8; cord.len()];
        let copied = cord.copy_prefix_to(&mut dst);
        assert_eq!(copied, cord.len());
        assert_eq!(dst, cord.to_vec());
    }
    // Span larger than the cord.
    {
        let mut dst = vec![b'x'; cord.len() + 10];
        let copied = cord.copy_prefix_to(&mut dst);
        assert_eq!(copied, cord.len());
        assert_eq!(&dst[..copied], &cord.to_vec()[..]);
        assert!(dst[copied..].iter().all(|&b| b == b'x'));
    }
    // Span smaller than the cord.
    {
        let target_size = cord.len() / 2;
        let mut dst = vec![0u8; target_size];
        let copied = cord.copy_prefix_to(&mut dst);
        assert_eq!(copied, target_size);
        assert_eq!(dst, &cord.to_vec()[..target_size]);
    }
    // Empty span.
    {
        let mut dst: [u8; 0] = [];
        assert_eq!(cord.copy_prefix_to(&mut dst), 0);
    }
}

#[test]
fn copy_prefix_to_fills_spans_of_any_size() {
    verify_copy_to_span(&Cord::new());
    verify_copy_to_span(&Cord::from("small cord"));
    verify_copy_to_span(&common::make_fragmented_cord([
        "fragmented ",
        "cord ",
        "to ",
        "test ",
        "copying ",
        "to ",
        "a ",
        "span.",
    ]));
}

#[test]
fn the_fragmented_helper_gives_each_fragment_its_own_chunk() {
    let fragmented = common::make_fragmented_cord(["A ", "fragmented ", "Cord"]);
    assert_eq!(fragmented, "A fragmented Cord");
    let chunks: Vec<&[u8]> = fragmented.chunks().collect();
    assert_eq!(chunks, vec![&b"A "[..], b"fragmented ", b"Cord"]);
}

fn verify_chunk_iterator(cord: &Cord, expected_chunks: usize) {
    assert_eq!(cord.chunks().next().is_none(), cord.is_empty());
    let content = cord.to_vec();
    let mut pos = 0;
    let mut n_chunks = 0;
    let pre_iter = cord.chunks();
    let mut post_iter = cord.chunks();
    for chunk in pre_iter {
        let other = post_iter.next().unwrap();
        assert_eq!(chunk, other);
        assert_eq!(chunk.as_ptr(), other.as_ptr());
        assert!(!chunk.is_empty());
        assert!(pos + chunk.len() <= content.len());
        assert_eq!(&content[pos..pos + chunk.len()], chunk);
        pos += chunk.len();
        n_chunks += 1;
    }
    assert_eq!(expected_chunks, n_chunks);
    assert_eq!(pos, content.len());
    assert!(post_iter.next().is_none());
    assert_eq!(cord.chunks().count(), expected_chunks);
}

#[test]
fn chunks_walk_every_shape_in_order() {
    verify_chunk_iterator(&Cord::new(), 0);
    verify_chunk_iterator(&Cord::from("small cord"), 1);
    verify_chunk_iterator(&Cord::from("larger than small buffer optimization"), 1);
    verify_chunk_iterator(
        &common::make_fragmented_cord([
            "a ",
            "small ",
            "fragmented ",
            "cord ",
            "for ",
            "testing ",
            "chunk ",
            "iterations.",
        ]),
        8,
    );

    let mut reused_nodes_cord = Cord::from(vec![b'c'; 40]);
    reused_nodes_cord.prepend(Cord::from(vec![b'b'; 40]));
    reused_nodes_cord.prepend(Cord::from(vec![b'a'; 40]));
    let mut expected_chunks = 3;
    let doublings: u32 = if cfg!(miri) { 5 } else { 8 };
    for _ in 0..doublings {
        let copy = reused_nodes_cord.clone();
        reused_nodes_cord.prepend(copy);
        expected_chunks *= 2;
        verify_chunk_iterator(&reused_nodes_cord, expected_chunks);
    }

    let mut rng = common::Rng::new(13);
    let flat_cord = Cord::from(rng.lowercase(256));
    let mut subcords = Cord::new();
    let n_subcords: usize = if cfg!(miri) { 32 } else { 128 };
    for i in 0..n_subcords {
        subcords.prepend(flat_cord.slice(i..i + 128));
    }
    verify_chunk_iterator(&subcords, n_subcords);
}

fn verify_double_ended_iterators(cord: &Cord) {
    let expected_chunks: Vec<&[u8]> = cord.chunks().collect();
    let reversed_chunks: Vec<&[u8]> = cord.chunks().rev().collect();
    assert_eq!(reversed_chunks, expected_chunks.iter().rev().copied().collect::<Vec<_>>());

    let mut chunks = cord.chunks();
    let (mut front, mut back) = (0, expected_chunks.len());
    let mut take_front = true;
    while front != back {
        if take_front {
            assert_eq!(chunks.next(), Some(expected_chunks[front]));
            front += 1;
        } else {
            back -= 1;
            assert_eq!(chunks.next_back(), Some(expected_chunks[back]));
        }
        take_front = !take_front;
    }
    assert_eq!(chunks.next(), None);
    assert_eq!(chunks.next_back(), None);

    // Cloning after initializing the lazy back position must give each
    // iterator an independent navigation stack.
    let mut chunks = cord.chunks();
    let first_back = chunks.next_back();
    let mut cloned = chunks.clone();
    assert_eq!(cloned.next_back(), chunks.next_back());
    assert_eq!(first_back, expected_chunks.last().copied());

    let expected = cord.to_vec();
    assert_eq!(cord.bytes().rev().collect::<Vec<_>>(), expected.iter().rev().copied().collect::<Vec<_>>());

    let mut bytes = cord.bytes();
    let (mut front, mut back) = (0, expected.len());
    let mut take_front = true;
    while front != back {
        assert_eq!(bytes.len(), back - front);
        if take_front {
            assert_eq!(bytes.next(), Some(expected[front]));
            front += 1;
        } else {
            back -= 1;
            assert_eq!(bytes.next_back(), Some(expected[back]));
        }
        take_front = !take_front;
    }
    assert_eq!(bytes.next(), None);
    assert_eq!(bytes.next_back(), None);

    if expected.len() >= 4 {
        let mut bytes = cord.bytes();
        assert_eq!(bytes.next_back(), expected.last().copied());
        assert_eq!(bytes.nth(expected.len() - 3), Some(expected[expected.len() - 3]));
        assert_eq!(bytes.next(), Some(expected[expected.len() - 2]));
        assert_eq!(bytes.next(), None);
    }
}

#[test]
fn chunks_and_bytes_are_double_ended_over_every_shape() {
    verify_double_ended_iterators(&Cord::new());
    verify_double_ended_iterators(&Cord::from("inline cord"));

    let data: Vec<u8> = (0..5000u32).map(|i| (i % 251) as u8).collect();
    let flat = Cord::copy_from_slice(&data[..1000]);
    assert!(internal::is_flat(&flat));
    verify_double_ended_iterators(&flat);

    let external = internal::make_external(&data);
    let substring = external.slice(123..4321);
    assert!(internal::is_substring(&substring));
    verify_double_ended_iterators(&substring);

    let tree = common::make_fragmented_cord(data.chunks(137));
    assert!(internal::is_btree(&tree));
    verify_double_ended_iterators(&tree);
    verify_double_ended_iterators(&tree.slice(73..tree.len() - 91));
}

#[test]
fn cursor_read_cord_on_a_single_data_edge() {
    let mut rng = common::Rng::new(17);
    let data = rng.lowercase(2000);
    for as_flat in [true, false] {
        let cord = if as_flat { Cord::from(&data[..]) } else { internal::make_external(&data) };

        let mut it = cord.cursor();
        let frag = it.read_cord(2000);
        assert_eq!(frag, data);
        assert!(!it.has_remaining());

        let mut it = cord.cursor();
        let frag = it.read_cord(200);
        assert_eq!(frag, &data[..200]);
        assert!(it.has_remaining());

        let frag = it.read_cord(1500);
        assert_eq!(frag, &data[200..1700]);
        assert!(it.has_remaining());

        let frag = it.read_cord(300);
        assert_eq!(frag, &data[1700..2000]);
        assert!(!it.has_remaining());
    }
}

#[test]
fn cursor_read_cord_on_a_substring_data_edge() {
    let mut rng = common::Rng::new(19);
    let data = rng.lowercase(2500);
    for as_flat in [true, false] {
        let cord = if as_flat { Cord::from(&data[..]) } else { internal::make_external(&data) };
        let cord = cord.slice(200..2200);
        let substr = &data[200..2200];

        let mut it = cord.cursor();
        let frag = it.read_cord(2000);
        assert_eq!(frag, substr);
        assert!(!it.has_remaining());

        let mut it = cord.cursor();
        let frag = it.read_cord(200);
        assert_eq!(frag, &substr[..200]);
        assert!(it.has_remaining());

        let frag = it.read_cord(1500);
        assert_eq!(frag, &substr[200..1700]);
        assert!(it.has_remaining());

        let frag = it.read_cord(300);
        assert_eq!(frag, &substr[1700..2000]);
        assert!(!it.has_remaining());
    }
}

fn verify_char_iterator(cord: &Cord) {
    assert_eq!(!cord.cursor().has_remaining(), cord.is_empty());
    assert_eq!(cord.cursor().remaining(), cord.len());
    assert_eq!(cord.bytes().len(), cord.len());

    let content = cord.to_vec();
    let mut i = 0;
    let mut pre_iter = cord.cursor();
    let mut post_iter = cord.bytes();
    while pre_iter.has_remaining() {
        assert!(i < cord.len());
        assert_eq!(content[i], pre_iter.peek().unwrap());
        assert_eq!(pre_iter.position(), i);

        let character_address = pre_iter.chunk().as_ptr();
        let mut copy = pre_iter.clone();
        copy.next_byte();
        assert_eq!(character_address, pre_iter.chunk().as_ptr());

        let mut advance_iter = cord.cursor();
        advance_iter.advance(i);
        assert_eq!(advance_iter.position(), pre_iter.position());
        assert_eq!(advance_iter.chunk(), pre_iter.chunk());

        let mut advance_iter = cord.cursor();
        assert_eq!(advance_iter.read_cord(i), cord.slice(..i));
        assert_eq!(advance_iter.position(), i);

        let mut advance_iter = pre_iter.clone();
        advance_iter.advance(cord.len() - i);
        assert!(!advance_iter.has_remaining());
        assert_eq!(advance_iter.position(), cord.len());
        assert_eq!(advance_iter.remaining(), 0);

        let mut advance_iter = pre_iter.clone();
        assert_eq!(advance_iter.read_cord(cord.len() - i), cord.slice(i..));
        assert!(!advance_iter.has_remaining());

        i += 1;
        assert_eq!(pre_iter.next_byte(), Some(content[i - 1]));
        assert_eq!(post_iter.next(), Some(content[i - 1]));
    }
    assert_eq!(i, cord.len());
    assert!(post_iter.next().is_none());

    let mut zero_advanced_end = cord.cursor();
    zero_advanced_end.advance(cord.len());
    zero_advanced_end.advance(0);
    assert!(!zero_advanced_end.has_remaining());

    let mut it = cord.cursor();
    for chunk in cord.chunks() {
        let mut chunk = chunk;
        while !chunk.is_empty() {
            assert_eq!(it.chunk(), chunk);
            chunk = &chunk[1..];
            it.next_byte();
        }
    }
}

#[test]
fn cursor_and_bytes_agree_at_every_position() {
    verify_char_iterator(&Cord::new());
    verify_char_iterator(&Cord::from("small cord"));
    verify_char_iterator(&Cord::from("larger than small buffer optimization"));
    verify_char_iterator(&common::make_fragmented_cord([
        "a ",
        "small ",
        "fragmented ",
        "cord ",
        "for ",
        "testing ",
        "character ",
        "iteration.",
    ]));

    let mut reused_nodes_cord = Cord::from("ghi");
    reused_nodes_cord.prepend(Cord::from("def"));
    reused_nodes_cord.prepend(Cord::from("abc"));
    for _ in 0..4 {
        let copy = reused_nodes_cord.clone();
        reused_nodes_cord.prepend(copy);
        verify_char_iterator(&reused_nodes_cord);
    }

    let mut rng = common::Rng::new(23);
    let flat_cord = Cord::from(rng.lowercase(256));
    let mut subcords = Cord::new();
    let n_subcords: usize = if cfg!(miri) { 2 } else { 4 };
    for i in 0..n_subcords {
        subcords.prepend(flat_cord.slice(16 * i..16 * i + 128));
    }
    verify_char_iterator(&subcords);
}

/// Six flats of 2500 bytes read in chunks of 150, 1500, 2500 and 3000 bytes,
/// covering partial, full and straddled reads including reads below the copy
/// threshold. b/197776822 surfaced a bug for a specific small read at the end.
#[test]
fn cursor_read_cord_in_fixed_size_steps() {
    const BLOCKS: usize = 6;
    const BLOCK_SIZE: usize = 2500;
    let mut rng = common::Rng::new(29);
    let data = rng.lowercase(BLOCKS * BLOCK_SIZE);
    let mut cord = Cord::new();
    for i in 0..BLOCKS {
        cord.append(Cord::from(&data[i * BLOCK_SIZE..(i + 1) * BLOCK_SIZE]));
    }

    for chunk_size in [1500usize, 2500, 3000, 150] {
        let mut it = cord.cursor();
        let mut it_remaining = cord.len();
        let mut it_advanced = 0;
        let mut offset = 0;
        while offset < data.len() {
            assert_eq!(it.remaining(), it_remaining);
            assert_eq!(it.position(), it_advanced);
            let n = (data.len() - offset).min(chunk_size);
            let chunk = it.read_cord(n);
            assert_eq!(chunk.len(), n);
            assert_eq!(chunk.compare(&data[offset..offset + n]), std::cmp::Ordering::Equal);
            offset += n;
            it_remaining -= n;
            it_advanced += n;
            assert_eq!(it.remaining(), it_remaining);
            assert_eq!(it.position(), it_advanced);
        }
    }
}

#[test]
fn chunks_yield_every_fragment() {
    for num_elements in [1, 10, 200] {
        let cord_chunks: Vec<String> = (0..num_elements).map(|i| format!("[{i}]")).collect();
        let c = common::make_fragmented_cord(&cord_chunks);
        let iterated: Vec<String> = c.chunks().map(|c| String::from_utf8(c.to_vec()).unwrap()).collect();
        assert_eq!(iterated, cord_chunks);
    }
}

#[test]
fn cursor_reads_peeks_skips_and_exhausts() {
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
    common::check(&first, &data[..10]);
    assert_eq!(cursor.position(), 10);
    cursor.advance(500);
    assert_eq!(cursor.position(), 510);
    let mid = cursor.read_cord(3000);
    common::check(&mid, &data[510..3510]);
    assert_eq!(cursor.peek(), Some(data[3510]));
    assert_eq!(cursor.next_byte(), Some(data[3510]));
    let rest: Vec<u8> = cursor.chunks().flatten().copied().collect();
    assert_eq!(rest, &data[3511..]);
    let last = cursor.read_cord(cursor.remaining());
    common::check(&last, &data[3511..]);
    assert!(!cursor.has_remaining());
    assert_eq!(cursor.read_cord(0), Cord::new());
    assert_eq!(cursor.next_byte(), None);

    // io::Read / BufRead.
    #[cfg(feature = "std")]
    {
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
    }

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

#[cfg(feature = "std")]
#[test]
fn cursor_seek_over_every_shape() {
    // Inline (<= 15 bytes).
    let inline_data = b"inline-cord-15!".to_vec();
    let inline = Cord::from(inline_data.as_slice());
    assert!(!internal::is_tree(&inline));
    check_cursor_seek(&inline, &inline_data);

    // Single flat (~1000 bytes).
    let flat_data: Vec<u8> = (0..1000u32).map(|i| (i % 256) as u8).collect();
    let flat = Cord::copy_from_slice(&flat_data);
    assert!(internal::is_flat(&flat));
    check_cursor_seek(&flat, &flat_data);

    // Multi-chunk btree (20 x 1000-byte appends).
    let btree_data: Vec<u8> = (0..20_000u32).map(|i| (i % 251) as u8).collect();
    let mut btree = Cord::new();
    for chunk in btree_data.chunks(1000) {
        btree.append(chunk);
    }
    assert!(internal::is_btree(&btree));
    check_cursor_seek(&btree, &btree_data);
}

/// Exercises the `io::Seek` contract of a cursor over `cord`, checking every
/// resulting position and read against `expected`.
#[cfg(feature = "std")]
#[expect(
    clippy::seek_from_current,
    reason = "deliberately exercises seek(Current(0)) as a no-op, not just stream_position"
)]
#[expect(clippy::cast_possible_wrap, reason = "tests juggle small integers freely")]
fn check_cursor_seek(cord: &Cord, expected: &[u8]) {
    let len = expected.len();
    let mut cursor = cord.cursor();

    // `Start`: forward seek, verified with `read_exact`.
    let pos = len / 3;
    assert_eq!(cursor.seek(SeekFrom::Start(pos as u64)).unwrap(), pos as u64);
    assert_eq!(cursor.position(), pos);
    assert_eq!(cursor.stream_position().unwrap(), pos as u64);
    let mut buf = vec![0u8; len - pos];
    cursor.read_exact(&mut buf).unwrap();
    assert_eq!(buf, expected[pos..]);
    assert_eq!(cursor.position(), len);

    // `Start`: backward seek from the end just reached, verified with
    // `read_to_end`.
    let pos = len / 5;
    assert_eq!(cursor.seek(SeekFrom::Start(pos as u64)).unwrap(), pos as u64);
    assert_eq!(cursor.position(), pos);
    let mut rest = Vec::new();
    cursor.read_to_end(&mut rest).unwrap();
    assert_eq!(rest, expected[pos..]);
    assert_eq!(cursor.position(), len);

    // `End` with a negative delta.
    let back = len / 4;
    let pos = len - back;
    assert_eq!(cursor.seek(SeekFrom::End(-(back as i64))).unwrap(), pos as u64);
    assert_eq!(cursor.position(), pos);
    assert_eq!(cursor.stream_position().unwrap(), pos as u64);
    let mut rest = Vec::new();
    cursor.read_to_end(&mut rest).unwrap();
    assert_eq!(rest, expected[pos..]);
    assert_eq!(cursor.position(), len);

    // `Current` with a negative delta (jump back to the middle from the end).
    let pos = len / 2;
    let delta = pos as i64 - len as i64;
    assert_eq!(cursor.seek(SeekFrom::Current(delta)).unwrap(), pos as u64);
    assert_eq!(cursor.position(), pos);
    let mut rest = Vec::new();
    cursor.read_to_end(&mut rest).unwrap();
    assert_eq!(rest, expected[pos..]);
    assert_eq!(cursor.position(), len);

    // `Current` with a positive delta.
    let base = len / 6;
    assert_eq!(cursor.seek(SeekFrom::Start(base as u64)).unwrap(), base as u64);
    let step = len / 8;
    let pos = base + step;
    assert_eq!(cursor.seek(SeekFrom::Current(step as i64)).unwrap(), pos as u64);
    assert_eq!(cursor.position(), pos);
    let mut rest = Vec::new();
    cursor.read_to_end(&mut rest).unwrap();
    assert_eq!(rest, expected[pos..]);
    assert_eq!(cursor.position(), len);

    // `seek(Current(0))` is a no-op, both from a mid-cord position and from
    // the end.
    let mid = len / 6;
    cursor.seek(SeekFrom::Start(mid as u64)).unwrap();
    let chunk_before = cursor.chunk().to_vec();
    let remaining_before = cursor.remaining();
    assert_eq!(cursor.seek(SeekFrom::Current(0)).unwrap(), mid as u64);
    assert_eq!(cursor.position(), mid);
    assert_eq!(cursor.remaining(), remaining_before);
    assert_eq!(cursor.chunk(), &chunk_before[..]);

    cursor.seek(SeekFrom::Start(len as u64)).unwrap();
    assert_eq!(cursor.seek(SeekFrom::Current(0)).unwrap(), len as u64);
    assert_eq!(cursor.position(), len);
    assert_eq!(cursor.remaining(), 0);

    // `rewind`.
    cursor.rewind().unwrap();
    assert_eq!(cursor.position(), 0);
    assert_eq!(cursor.stream_position().unwrap(), 0);
    let mut all = Vec::new();
    cursor.read_to_end(&mut all).unwrap();
    assert_eq!(all, expected);

    // `Start(len)` succeeds and leaves nothing remaining.
    assert_eq!(cursor.seek(SeekFrom::Start(len as u64)).unwrap(), len as u64);
    assert_eq!(cursor.position(), len);
    assert_eq!(cursor.remaining(), 0);
    assert!(!cursor.has_remaining());

    // Errors: each leaves `position()` unchanged.
    cursor.seek(SeekFrom::Start(mid as u64)).unwrap();
    let before = cursor.position();

    let err = cursor.seek(SeekFrom::Start(len as u64 + 1)).unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
    assert_eq!(cursor.position(), before);

    let err = cursor.seek(SeekFrom::End(1)).unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
    assert_eq!(cursor.position(), before);

    let err = cursor.seek(SeekFrom::End(-(len as i64 + 1))).unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
    assert_eq!(cursor.position(), before);

    let err = cursor.seek(SeekFrom::Current(-(before as i64 + 1))).unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
    assert_eq!(cursor.position(), before);

    // `Seek` composes with `io::Read::read_exact` and
    // `BufRead::fill_buf`/`consume` after a backward seek: force the cursor
    // to the end first so the next seek is a genuine backward rebuild.
    cursor.seek(SeekFrom::Start(len as u64)).unwrap();
    let target = len / 3;
    cursor.seek(SeekFrom::Start(target as u64)).unwrap();
    let head_len = (len - target).min(7);
    let mut head = vec![0u8; head_len];
    cursor.read_exact(&mut head).unwrap();
    assert_eq!(head, expected[target..target + head_len]);
    let filled = cursor.fill_buf().unwrap().to_vec();
    assert_eq!(&filled[..], &expected[target + head_len..target + head_len + filled.len()]);
    cursor.consume(filled.len());
    let mut tail = Vec::new();
    cursor.read_to_end(&mut tail).unwrap();
    assert_eq!(tail, expected[target + head_len + filled.len()..]);
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
