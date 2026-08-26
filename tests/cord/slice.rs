//! `slice`/`get`/`Index`, subcord sharing, `as_contiguous`/`make_contiguous`.
#![expect(clippy::cast_possible_truncation, reason = "tests juggle small integers freely")]

use crate::common::{self, internal};
use cord_rs::Cord;

/// `AppendWithFragments`: appends `s` in ~10 random fragments, half of them
/// external. Returns the size of the smallest fragment.
fn append_with_fragments(s: &[u8], rng: &mut common::Rng, cord: &mut Cord) -> usize {
    let mut j = 0;
    let max_size = s.len() / 5;
    let mut min_size = max_size;
    while j < s.len() {
        let mut n = 1 + rng.up_to(max_size);
        if n > s.len() - j {
            n = s.len() - j;
        }
        min_size = min_size.min(n);
        if rng.coin_flip() {
            common::add_external_memory(&s[j..j + n], cord);
        } else {
            cord.append(&s[j..j + n]);
        }
        j += n;
    }
    min_size
}

#[test]
fn subcords_at_interesting_offsets_match_the_source() {
    let mut rng = common::Rng::new(1);
    let source_len: usize = if cfg!(miri) { 256 } else { 1024 };
    let s = rng.lowercase(source_len);

    let mut a = Cord::new();
    append_with_fragments(&s, &mut rng, &mut a);
    assert_eq!(a.to_vec(), s);

    // Check subcords of a, from a variety of interesting points.
    let mut positions = std::collections::BTreeSet::new();
    for i in 0..=32usize {
        positions.insert(i);
        positions.insert((i * 32).wrapping_sub(1));
        positions.insert(i * 32);
        positions.insert(i * 32 + 1);
        positions.insert(a.len() - i);
    }
    positions.insert(237);
    positions.insert(732);
    let positions: Vec<usize> = if cfg!(miri) {
        positions.iter().step_by(3).copied().collect()
    } else {
        positions.into_iter().collect()
    };
    for &pos in &positions {
        if pos > a.len() {
            continue;
        }
        for &end_pos in &positions {
            if end_pos < pos || end_pos > a.len() {
                continue;
            }
            let sa = a.slice(pos..end_pos);
            assert_eq!(sa.to_vec(), &s[pos..end_pos], "{pos}..{end_pos}");
            common::assert_valid(&sa);
        }
    }

    // Do the same thing for an inline cord.
    let sh = b"short";
    let c = Cord::from(&sh[..]);
    for pos in 0..=sh.len() {
        for n in 0..=(sh.len() - pos) {
            let sc = c.slice(pos..pos + n);
            assert_eq!(sc.to_vec(), &sh[pos..pos + n]);
        }
    }

    // Check subcords of subcords.
    let mut sa = a.slice(..);
    let mut ss = &s[..];
    while sa.len() > 1 {
        sa = sa.slice(1..sa.len() - 1);
        ss = &ss[1..ss.len() - 1];
        assert_eq!(sa.to_vec(), ss);
    }

    // Asking for too much is an error in Rust (the C++ original clamps).
    assert!(a.get(0..=a.len()).is_none());
    assert!(a.get((a.len() + 1)..=a.len()).is_none());
    assert!(a.get(a.len()..a.len()).unwrap().is_empty());
}

#[test]
fn as_contiguous_across_representations() {
    // empty
    assert_eq!(Cord::new().as_contiguous(), Some(&b""[..]));

    // flat
    assert_eq!(Cord::from("hello").as_contiguous(), Some(&b"hello"[..]));

    // substr inlined
    let mut c = Cord::from("hello");
    c.advance(1);
    assert_eq!(c.as_contiguous(), Some(&b"ello"[..]));

    // substr flat
    let c = Cord::from("longer than 15 bytes");
    let sub = internal::make_substring(&c, 1, c.len() - 1);
    assert_eq!(sub.as_contiguous(), Some(&b"onger than 15 bytes"[..]));

    // concat
    let c = common::make_fragmented_cord(["hel", "lo"]);
    assert_eq!(c.as_contiguous(), None);

    // external
    let c = internal::make_external(b"hell");
    assert_eq!(c.as_contiguous(), Some(&b"hell"[..]));

    // substr external
    let c = internal::make_external(b"hell");
    let sub = internal::make_substring(&c, 1, c.len() - 1);
    assert_eq!(sub.as_contiguous(), Some(&b"ell"[..]));
}

/// Not part of the API contract, but intended to be true of the current
/// implementation: sub-cords of whole chunks are flat.
#[test]
fn subcords_of_whole_chunks_are_contiguous() {
    let fragments: [&str; 8] = [
        "A fragmented test",
        " cord",
        " to test subcords",
        " of ",
        "a",
        " cord for",
        " each chunk returned by the ",
        "iterator",
    ];
    let c = common::make_fragmented_cord(fragments);
    let mut offset = 0;
    let mut cursor = c.cursor();
    for (fragment, sv) in c.chunks().enumerate() {
        let expected = fragments[fragment].as_bytes();
        let subcord1 = c.slice(offset..offset + sv.len());
        let subcord2 = cursor.read_cord(sv.len());
        assert_eq!(subcord1.as_contiguous(), Some(expected));
        assert_eq!(subcord2.as_contiguous(), Some(expected));
        offset += sv.len();
    }
}

fn has_one_chunk(c: &Cord) -> bool {
    c.chunks().count() <= 1
}

fn verify_flatten(mut c: Cord) {
    let old_contents = c.to_vec();
    let already_flat_and_non_empty = has_one_chunk(&c) && !c.is_empty();
    let old_flat_ptr =
        if already_flat_and_non_empty { Some(c.chunks().next().unwrap().as_ptr()) } else { None };
    let new_flat = c.make_contiguous();
    assert_eq!(new_flat, &old_contents[..]);
    if let Some(old_ptr) = old_flat_ptr {
        assert_eq!(old_ptr, new_flat.as_ptr(), "Allocated new memory even though the Cord was already flat.");
    }
    assert_eq!(c.to_vec(), old_contents);
    assert!(has_one_chunk(&c));
}

#[test]
fn make_contiguous_never_reallocates_an_already_flat_cord() {
    verify_flatten(Cord::new());
    verify_flatten(Cord::from("small cord"));
    verify_flatten(Cord::from("larger than small buffer optimization"));
    verify_flatten(common::make_fragmented_cord(["small ", "fragmented ", "cord"]));
    // Longer than the largest flat buffer.
    let mut rng = common::Rng::new(3);
    verify_flatten(Cord::from(rng.lowercase(8192)));
}

#[test]
fn indexing_walks_external_and_inline_chunks() {
    let mut cord = Cord::from("hello");
    common::add_external_memory(b" world!", &mut cord);
    common::add_external_memory(b" how are ", &mut cord);
    cord.append(" you?");
    let s = cord.to_vec();
    for (i, &b) in s.iter().enumerate() {
        assert_eq!(b, cord[i]);
    }
}

#[test]
fn slice_and_get_over_a_multi_chunk_cord() {
    let len: u32 = if cfg!(miri) { 6_000 } else { 50_000 };
    let data: Vec<u8> = (0..len).map(|i| (i % 253) as u8).collect();
    let mut cord = Cord::new();
    for chunk in data.chunks(1000) {
        cord.append(chunk);
    }
    common::check(&cord, &data);

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
        common::check(&sub, &data[start..end]);
        assert_eq!(cord.get(start..end).unwrap(), sub);
    }
    assert!(cord.get(10..=n).is_none());
    #[expect(
        clippy::reversed_empty_ranges,
        reason = "exercising the out-of-range result of a deliberately reversed range"
    )]
    let reversed = 11..10;
    assert!(cord.get(reversed).is_none());
    common::check(&cord.slice(..), &data);
    common::check(&cord.slice(100..=200), &data[100..=200]);

    common::check(&Cord::from("hello").slice(1..4), b"ell");
}

#[test]
fn indexing_and_get_over_a_multi_chunk_cord() {
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
fn get_accepts_every_index_type() {
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
fn make_contiguous_flattens_every_shape() {
    let mut cord = Cord::new();
    for i in 0..100u8 {
        cord.append(vec![i; 100]);
    }
    let expected: Vec<u8> = (0..100u8).flat_map(|i| vec![i; 100]).collect();
    assert!(cord.as_contiguous().is_none());

    assert_eq!(cord.make_contiguous(), &expected[..]);
    assert!(cord.as_contiguous().is_some());
    common::check(&cord, &expected);
    assert!(internal::is_external(&cord), "10000 bytes > max flat length -> external");
    let mut small = Cord::from("a");
    small.append(vec![b'b'; 20]);
    small.append("c");
    assert_eq!(small.make_contiguous(), [b"a".as_slice(), &[b'b'; 20], b"c"].concat());
    assert!(internal::is_flat(&small));
    let mut inline = Cord::from("xyz");
    assert_eq!(inline.make_contiguous(), b"xyz");
}
