//! `compare`/`Ord`/`Eq`/`Hash`, `find`/`contains`/`starts_with`/`ends_with`.
#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "tests juggle small integers freely"
)]

use crate::common::{self, internal};
use cord_rs::{Cord, CordBuffer};

#[test]
fn starts_with_and_ends_with_accept_cords_and_strings() {
    let x = Cord::from("abcde");
    let empty = Cord::from("");

    assert!(x.starts_with(&Cord::from("abcde")));
    assert!(x.starts_with(&Cord::from("abc")));
    assert!(x.starts_with(&Cord::from("")));
    assert!(empty.starts_with(&Cord::from("")));
    assert!(x.ends_with(&Cord::from("abcde")));
    assert!(x.ends_with(&Cord::from("cde")));
    assert!(x.ends_with(&Cord::from("")));
    assert!(empty.ends_with(&Cord::from("")));

    assert!(!x.starts_with(&Cord::from("xyz")));
    assert!(!empty.starts_with(&Cord::from("xyz")));
    assert!(!x.ends_with(&Cord::from("xyz")));
    assert!(!empty.ends_with(&Cord::from("xyz")));

    assert!(x.starts_with("abcde"));
    assert!(x.starts_with("abc"));
    assert!(x.starts_with(""));
    assert!(empty.starts_with(""));
    assert!(x.ends_with("abcde"));
    assert!(x.ends_with("cde"));
    assert!(x.ends_with(""));
    assert!(empty.ends_with(""));

    assert!(!x.starts_with("xyz"));
    assert!(!empty.starts_with("xyz"));
    assert!(!x.ends_with("xyz"));
    assert!(!empty.ends_with("xyz"));
}

#[test]
fn contains_over_flat_and_fragmented_haystacks() {
    let flat_haystack = Cord::from("this is a flat cord");
    let fragmented_haystack =
        common::make_fragmented_cord(["this", " ", "is", " ", "a", " ", "fragmented", " ", "cord"]);

    assert!(flat_haystack.contains(""));
    assert!(fragmented_haystack.contains(""));
    assert!(flat_haystack.contains(&Cord::from("")));
    assert!(fragmented_haystack.contains(&Cord::from("")));
    assert!(Cord::from("").contains(""));
    assert!(Cord::from("").contains(&Cord::from("")));
    assert!(!Cord::from("").contains(&flat_haystack));
    assert!(!Cord::from("").contains(&fragmented_haystack));

    assert!(!flat_haystack.contains("z"));
    assert!(!fragmented_haystack.contains("z"));
    assert!(!flat_haystack.contains(&Cord::from("z")));
    assert!(!fragmented_haystack.contains(&Cord::from("z")));

    assert!(!flat_haystack.contains("is an"));
    assert!(!fragmented_haystack.contains("is an"));
    assert!(!flat_haystack.contains(&Cord::from("is an")));
    assert!(!fragmented_haystack.contains(&Cord::from("is an")));
    assert!(!flat_haystack.contains(&common::make_fragmented_cord(["is", " ", "an"])));
    assert!(!fragmented_haystack.contains(&common::make_fragmented_cord(["is", " ", "an"])));

    assert!(flat_haystack.contains("is a"));
    assert!(fragmented_haystack.contains("is a"));
    assert!(flat_haystack.contains(&Cord::from("is a")));
    assert!(fragmented_haystack.contains(&Cord::from("is a")));
    assert!(flat_haystack.contains(&common::make_fragmented_cord(["is", " ", "a"])));
    assert!(fragmented_haystack.contains(&common::make_fragmented_cord(["is", " ", "a"])));
}

#[test]
fn find_over_flat_and_fragmented_haystacks() {
    let flat_haystack = Cord::from("this is a flat cord");
    let fragmented_haystack =
        common::make_fragmented_cord(["this", " ", "is", " ", "a", " ", "fragmented", " ", "cord"]);
    let empty_haystack = Cord::from("");

    assert_eq!(flat_haystack.find(""), Some(0));
    assert_eq!(fragmented_haystack.find(""), Some(0));
    assert_eq!(flat_haystack.find(&Cord::from("")), Some(0));
    assert_eq!(fragmented_haystack.find(&Cord::from("")), Some(0));
    assert_eq!(empty_haystack.find(""), Some(0));
    assert_eq!(empty_haystack.find(&Cord::from("")), Some(0));
    assert_eq!(empty_haystack.find(&flat_haystack), None);
    assert_eq!(empty_haystack.find(&fragmented_haystack), None);

    assert_eq!(flat_haystack.find("z"), None);
    assert_eq!(fragmented_haystack.find("z"), None);
    assert_eq!(flat_haystack.find(&Cord::from("z")), None);
    assert_eq!(fragmented_haystack.find(&Cord::from("z")), None);

    assert_eq!(flat_haystack.find("is an"), None);
    assert_eq!(fragmented_haystack.find("is an"), None);
    assert_eq!(flat_haystack.find(&Cord::from("is an")), None);
    assert_eq!(fragmented_haystack.find(&Cord::from("is an")), None);
    assert_eq!(flat_haystack.find(&common::make_fragmented_cord(["is", " ", "an"])), None);
    assert_eq!(fragmented_haystack.find(&common::make_fragmented_cord(["is", " ", "an"])), None);

    assert_eq!(flat_haystack.find("is a"), Some(5));
    assert_eq!(fragmented_haystack.find("is a"), Some(5));
    assert_eq!(flat_haystack.find(&Cord::from("is a")), Some(5));
    assert_eq!(fragmented_haystack.find(&Cord::from("is a")), Some(5));
    assert_eq!(flat_haystack.find(&common::make_fragmented_cord(["is", " ", "a"])), Some(5));
    assert_eq!(fragmented_haystack.find(&common::make_fragmented_cord(["is", " ", "a"])), Some(5));
}

#[test]
fn find_across_fragment_boundaries_matches_slice_windows() {
    let n: i32 = if cfg!(miri) { 32 } else { 64 };
    let bytes: Vec<u8> = (0..n).map(|i| (i * 17 % 251) as u8).collect();
    let chunk_sizes: Vec<usize> = if cfg!(miri) { vec![1, 5] } else { (1..=12).collect() };
    for chunk_size in chunk_sizes {
        let haystack = common::make_fragmented_cord(bytes.chunks(chunk_size));
        for start in 0..bytes.len() {
            for end in (start + 1)..=(start + 12).min(bytes.len()) {
                let needle = &bytes[start..end];
                let expected = bytes.windows(needle.len()).position(|window| window == needle);
                assert_eq!(haystack.find(needle), expected, "chunk_size={chunk_size}, range={start}..{end}");
            }
        }
    }

    let repeated = vec![b'a'; 4096];
    let haystack = common::make_fragmented_cord(repeated.chunks(31));
    assert_eq!(haystack.find(&b"aaaaab"[..]), None);
    assert_eq!(haystack.find(&b"aaaaaa"[..]), Some(0));
}

/// `ends_with` against a genuinely fragmented suffix. `make_fragmented_cord`
/// gives every fragment its own external node (unlike plain `append`, which
/// coalesces small pieces back together), so this checks the fragmentation
/// held rather than assuming it.
#[test]
fn ends_with_fragmented_suffix() {
    let haystack =
        common::make_fragmented_cord(["this", " ", "is", " ", "a", " ", "fragmented", " ", "cord"]);
    assert!(haystack.chunks().count() > 1, "haystack must stay fragmented");

    // Suffix straddling multiple chunk boundaries.
    let suffix = common::make_fragmented_cord(["a", " ", "fragmented", " ", "cord"]);
    assert!(suffix.chunks().count() > 1, "suffix must stay fragmented");
    assert!(haystack.ends_with(&suffix));
    assert!(haystack.ends_with("a fragmented cord"));

    // Suffix == whole cord.
    let whole = common::make_fragmented_cord(["this", " ", "is", " ", "a", " ", "fragmented", " ", "cord"]);
    assert!(whole.chunks().count() > 1, "whole-cord suffix must stay fragmented");
    assert!(haystack.ends_with(&whole));

    // Mismatch in the last byte.
    let mismatch_last = common::make_fragmented_cord(["a", " ", "fragmented", " ", "corD"]);
    assert!(!haystack.ends_with(&mismatch_last));

    // Mismatch in the first suffix chunk.
    let mismatch_first = common::make_fragmented_cord(["X", " ", "fragmented", " ", "cord"]);
    assert!(!haystack.ends_with(&mismatch_first));
}

/// A fragmented `find` match that lands flush against the very end of a
/// fragmented haystack: the needle's last byte and the haystack's last byte
/// coincide, so `is_subcord_at` consumes both cursors down to nothing in
/// lockstep across many single-byte chunk boundaries. Regression coverage
/// for the hardening in `is_subcord_at`/`is_slice_at`, which now return
/// `false` instead of relying on a length precondition that, if violated,
/// used to infinite-loop.
#[test]
fn find_fragmented_needle_flush_with_haystack_end() {
    let bytes: Vec<u8> = (0..40).map(|i| (i * 17 % 251) as u8).collect();
    let haystack = common::make_fragmented_cord(bytes.chunks(1));
    assert!(haystack.chunks().count() > 1, "haystack must stay fragmented");

    // The needle is the tail of the haystack, itself fragmented into
    // single-byte chunks: matching it exhausts both cursors on the same
    // step.
    let needle = common::make_fragmented_cord(bytes[30..].chunks(1));
    assert!(needle.chunks().count() > 1, "needle must stay fragmented");
    assert_eq!(needle.chunks().count(), bytes.len() - 30);
    assert_eq!(haystack.find(&needle), Some(30));

    // Same needle, but the very last byte differs: the mismatch is only
    // discovered on the final chunk pair, after the cursors have advanced
    // in lockstep through every preceding one.
    let mut mismatched = bytes[30..].to_vec();
    *mismatched.last_mut().unwrap() = mismatched.last().unwrap().wrapping_add(1);
    let needle_mismatched = common::make_fragmented_cord(mismatched.chunks(1));
    assert_eq!(haystack.find(&needle_mismatched), None);
}

fn check_comparison(lhs: &Cord, rhs: &Cord) {
    let lhs_bytes = lhs.to_vec();
    let rhs_bytes = rhs.to_vec();
    let expected = lhs_bytes.cmp(&rhs_bytes);
    assert_eq!(lhs.compare(rhs), expected, "LHS={lhs:?}; RHS={rhs:?}");
    assert_eq!(lhs.compare(&rhs_bytes[..]), expected);
    assert_eq!(rhs.compare(lhs), expected.reverse());
    assert_eq!(rhs.compare(&lhs_bytes[..]), expected.reverse());
    assert_eq!(lhs.cmp(rhs), expected);
    assert_eq!(lhs.partial_cmp(&rhs_bytes), Some(expected));
}

#[test]
fn compare_matches_byte_ordering() {
    let subcord = Cord::from("aaaaaBBBBBcccccDDDDD").slice(3..13);

    let mut tmp = Cord::from("aaaaaaaaaaaaaaaa");
    tmp.append("BBBBBBBBBBBBBBBB");
    let mut concat = Cord::from("cccccccccccccccc");
    concat.append("DDDDDDDDDDDDDDDD");
    concat.prepend(tmp);

    let mut concat2 = Cord::from("aaaaaaaaaaaaa");
    concat2.append("aaaBBBBBBBBBBBBBBBBccccc");
    concat2.append("cccccccccccDDDDDDDDDDDDDD");
    concat2.append("DD");

    let cases: Vec<(Cord, Cord)> = vec![
        // Inline cords.
        (Cord::from("abcdef"), Cord::from("abcdef")),
        (Cord::from("abcdef"), Cord::from("abcdee")),
        (Cord::from("abcdef"), Cord::from("abcdeg")),
        (Cord::from("bbcdef"), Cord::from("abcdef")),
        (Cord::from("bbcdef"), Cord::from("abcdeg")),
        (Cord::from("abcdefa"), Cord::from("abcdef")),
        (Cord::from("abcdef"), Cord::from("abcdefa")),
        // Small flat cords.
        (Cord::from("aaaaaBBBBBcccccDDDDD"), Cord::from("aaaaaBBBBBcccccDDDDD")),
        (Cord::from("aaaaaBBBBBcccccDDDDD"), Cord::from("aaaaaBBBBBxccccDDDDD")),
        (Cord::from("aaaaaBBBBBcxcccDDDDD"), Cord::from("aaaaaBBBBBcccccDDDDD")),
        (Cord::from("aaaaaBBBBBxccccDDDDD"), Cord::from("aaaaaBBBBBcccccDDDDX")),
        (Cord::from("aaaaaBBBBBcccccDDDDDa"), Cord::from("aaaaaBBBBBcccccDDDDD")),
        (Cord::from("aaaaaBBBBBcccccDDDDD"), Cord::from("aaaaaBBBBBcccccDDDDDa")),
        // Subcords.
        (subcord.clone(), subcord.clone()),
        (subcord.clone(), Cord::from("aaBBBBBccc")),
        (subcord.clone(), Cord::from("aaBBBBBccd")),
        (subcord.clone(), Cord::from("aaBBBBBccb")),
        (subcord.clone(), Cord::from("aaBBBBBxcb")),
        (subcord.clone(), Cord::from("aaBBBBBccca")),
        (subcord.clone(), Cord::from("aaBBBBBcc")),
        // Concats.
        (concat.clone(), concat.clone()),
        (concat.clone(), Cord::from("aaaaaaaaaaaaaaaaBBBBBBBBBBBBBBBBccccccccccccccccDDDDDDDDDDDDDDDD")),
        (concat.clone(), Cord::from("aaaaaaaaaaaaaaaaBBBBBBBBBBBBBBBBcccccccccccccccxDDDDDDDDDDDDDDDD")),
        (concat.clone(), Cord::from("aaaaaaaaaaaaaaaaBBBBBBBBBBBBBBBBacccccccccccccccDDDDDDDDDDDDDDDD")),
        (concat.clone(), Cord::from("aaaaaaaaaaaaaaaaBBBBBBBBBBBBBBBBccccccccccccccccDDDDDDDDDDDDDDD")),
        (concat.clone(), Cord::from("aaaaaaaaaaaaaaaaBBBBBBBBBBBBBBBBccccccccccccccccDDDDDDDDDDDDDDDDe")),
        (concat.clone(), concat2.clone()),
        // Empty-vs-empty and equal-vs-equal, the two cases that still carry
        // meaning in Rust from a C++ "operator= leaves no stale state" test
        // (it otherwise exercised `operator=` itself, which has no Rust
        // analogue).
        (Cord::new(), Cord::new()),
        (Cord::from("cccccc"), Cord::from("cccccc")),
    ];
    for (lhs, rhs) in &cases {
        check_comparison(lhs, rhs);
    }
}

#[test]
fn comparison_treats_bytes_as_unsigned() {
    let mut rng = common::Rng::new(11);
    let x = rng.up_to(256) as u8;
    let n1 = rng.up_to(100);
    let n2 = rng.up_to(100);
    check_comparison(&Cord::from(vec![x; n1]), &Cord::from(vec![x ^ 0x80; n2]));
    assert_eq!(Cord::from(b"\x80").compare(b"\x7f"), std::cmp::Ordering::Greater);
}

#[test]
fn random_cords_compare_like_their_bytes() {
    let iters: i32 = if cfg!(miri) { 200 } else { 5000 };
    let mut rng = common::Rng::new(42);
    let n = rng.up_to(5000);
    let a = [
        internal::make_external(&vec![b'x'; n]),
        Cord::from("ant"),
        Cord::from("elephant"),
        Cord::from("giraffe"),
        Cord::from(vec![rng.up_to(100) as u8; rng.up_to(100)]),
        Cord::from(""),
        Cord::from("x"),
        Cord::from("A"),
        Cord::from("B"),
        Cord::from("C"),
    ];
    for i in 0..iters {
        let mut c = Cord::new();
        let mut d = Cord::new();
        for _ in 0..=(i % 7) {
            c.append(&a[rng.up_to(a.len())]);
            d.append(&a[rng.up_to(a.len())]);
        }
        let c = if rng.coin_flip() { c } else { Cord::from(c.to_vec()) };
        let d = if rng.coin_flip() { d } else { Cord::from(d.to_vec()) };
        check_comparison(&c, &d);
    }
}

#[test]
#[expect(
    clippy::neg_cmp_op_on_partial_ord,
    reason = "deliberately exercises the negated forms of each comparison operator"
)]
fn comparison_operators_against_slices_strings_and_cords() {
    fn check<L, R>(a: &L, b: &R, a2: &L)
    where
        L: PartialEq<R> + PartialOrd<R> + PartialEq<L> + PartialOrd<L>,
        R: PartialEq<L> + PartialOrd<L>,
    {
        assert!(*a == *a2);
        assert!(!(*a == *b));
        assert!(*a != *b);
        assert!(!(*a != *a2));
        assert!(*a < *b);
        assert!(!(*b < *a));
        assert!(*b > *a);
        assert!(!(*a > *b));
        assert!(*a >= *a2);
        assert!(*b >= *a);
        assert!(!(*a >= *b));
        assert!(*a <= *a2);
        assert!(*a <= *b);
        assert!(!(*b <= *a));
    }
    check(&Cord::from("a"), &Cord::from("b"), &Cord::from("a"));
    check(&Cord::from("a"), &"b", &Cord::from("a"));
    check(&"a", &Cord::from("b"), &"a");
    check(&Cord::from("a"), &String::from("b"), &Cord::from("a"));
    check(&String::from("a"), &Cord::from("b"), &String::from("a"));
    check(&Cord::from("a"), &&b"b"[..], &Cord::from("a"));
    check(&&b"a"[..], &Cord::from("b"), &&b"a"[..]);
    check(&Cord::from("a"), &b"b".to_vec(), &Cord::from("a"));
    check(&b"a".to_vec(), &Cord::from("b"), &b"a".to_vec());
}

#[test]
fn hash_agrees_with_equality_across_block_boundaries() {
    // Hits the 1024 byte hashing block boundaries precisely.
    let cords = [
        Cord::new(),
        common::make_fragmented_cord([vec![b'a'; 600], vec![b'a'; 600]]),
        common::make_fragmented_cord([vec![b'a'; 1200]]),
        common::make_fragmented_cord([vec![b'b'; 900], vec![b'b'; 900]]),
        common::make_fragmented_cord([vec![b'b'; 1800]]),
        common::make_fragmented_cord([vec![b'c'; 2000], vec![b'c'; 2000]]),
        common::make_fragmented_cord([vec![b'c'; 4000]]),
        common::make_fragmented_cord([vec![b'd'; 1024]]),
        common::make_fragmented_cord([vec![b'd'; 1023], b"d".to_vec()]),
        common::make_fragmented_cord([vec![b'e'; 1025]]),
        common::make_fragmented_cord([vec![b'e'; 1024], b"e".to_vec()]),
        common::make_fragmented_cord([vec![b'e'; 1023], b"e".to_vec(), b"e".to_vec()]),
    ];
    // Hoisted: 12 hashes per hasher instead of recomputing inside the O(n^2)
    // double loop below.
    let default_hashes: Vec<u64> = cords.iter().map(common::default_hash).collect();
    let boundary_hashes: Vec<u64> = cords.iter().map(common::boundary_hash).collect();
    for (i, a) in cords.iter().enumerate() {
        for (j, b) in cords.iter().enumerate() {
            let equal = a == b;
            assert_eq!(default_hashes[i] == default_hashes[j], equal, "{a:?} vs {b:?}");
            if equal {
                assert_eq!(boundary_hashes[i], boundary_hashes[j], "{a:?} vs {b:?}");
            }
        }
    }
}

#[test]
fn equality_and_ordering_against_slices_and_strings() {
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
}

#[test]
fn search_over_a_fragmented_cord() {
    let mut a = Cord::new();
    for chunk in b"the quick brown fox jumps over the lazy dog".chunks(5) {
        a.append(chunk);
    }
    assert!(internal::is_btree(&a));
    let b = Cord::from("the quick brown fox jumps over the lazy dog");

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
fn comparison_is_symmetric_for_arrays_and_buffers() {
    let cord = Cord::from("bbb");
    let cases = [
        (*b"aaa", std::cmp::Ordering::Less),
        (*b"bbb", std::cmp::Ordering::Equal),
        (*b"ccc", std::cmp::Ordering::Greater),
    ];
    for (arr, expected) in cases {
        let slice_cmp = arr[..].partial_cmp(&cord.to_vec()[..]);
        assert_eq!(slice_cmp, Some(expected));

        // [u8; N], both directions, both == and partial_cmp.
        assert_eq!(arr == cord, expected == std::cmp::Ordering::Equal);
        assert_eq!(cord == arr, arr == cord);
        assert_eq!(arr.partial_cmp(&cord), slice_cmp);
        assert_eq!(cord.partial_cmp(&arr), slice_cmp.map(std::cmp::Ordering::reverse));

        // &[u8; N], both directions, both == and partial_cmp.
        let r = &arr;
        assert_eq!(r == cord, arr == cord);
        assert_eq!(cord == r, r == cord);
        assert_eq!(r.partial_cmp(&cord), arr.partial_cmp(&cord));
    }

    for (bytes, expected) in [
        (*b"aaa", std::cmp::Ordering::Less),
        (*b"bbb", std::cmp::Ordering::Equal),
        (*b"ccc", std::cmp::Ordering::Greater),
    ] {
        let mut buffer = CordBuffer::with_capacity(bytes.len());
        buffer.put_slice(&bytes);
        let slice_cmp = buffer.as_slice().partial_cmp(&cord.to_vec()[..]);
        assert_eq!(slice_cmp, Some(expected));

        assert_eq!(buffer == cord, expected == std::cmp::Ordering::Equal);
        assert_eq!(cord == buffer, buffer == cord);
        assert_eq!(buffer.partial_cmp(&cord), slice_cmp);
        assert_eq!(cord.partial_cmp(&buffer), slice_cmp.map(std::cmp::Ordering::reverse));
    }

    // A `CordBuffer` as a `find`/`contains` needle.
    let mut buffer = CordBuffer::with_capacity(32);
    buffer.put_slice(b"needle");
    let haystack = Cord::from("a needle in a haystack");
    assert_eq!(haystack.find(&buffer), Some(2));
    assert!(haystack.contains(&buffer));
}

#[test]
fn hash_is_independent_of_chunk_layout() {
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
    assert_eq!(common::default_hash(&flat), common::default_hash(&chunked));
    assert_eq!(common::boundary_hash(&flat), common::boundary_hash(&chunked));
    assert_eq!(common::default_hash(&flat), common::default_hash(&prepended));
    assert_eq!(common::boundary_hash(&flat), common::boundary_hash(&prepended));
    assert_eq!(common::default_hash(&flat.slice(100..2000)), common::default_hash(&chunked.slice(100..2000)));
    assert_eq!(
        common::boundary_hash(&flat.slice(100..2000)),
        common::boundary_hash(&chunked.slice(100..2000))
    );
    assert_ne!(common::default_hash(&flat), common::default_hash(&flat.slice(..data.len() - 1)));
    assert_eq!(
        common::default_hash(&Cord::from("abc")),
        common::default_hash(&Cord::from(b"abc".repeat(100)).slice(..3))
    );
    assert_eq!(
        common::boundary_hash(&Cord::from("abc")),
        common::boundary_hash(&Cord::from(b"abc".repeat(100)).slice(..3))
    );
}
