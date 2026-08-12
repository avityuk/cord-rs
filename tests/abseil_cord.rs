//! Port of abseil's `cord_test.cc` to the `cord_rs` API.
//!
//! The C++ suite is parametrized over "hardened" (CRC carrying) cords; the
//! CRC node was not ported so every test runs once. Tests that exist only to
//! exercise C++ releaser plumbing, `absl::Format`, Cordz or CRC semantics are
//! omitted or reduced to their observable Rust equivalents (noted inline).
//! Names mirror the C++ tests in `snake_case`.
#![allow(unused_assignments)]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    reason = "tests juggle small integers freely"
)]

use std::collections::hash_map::DefaultHasher;
use std::fmt::Write as _;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, LazyLock};

use cord_rs::{Cord, CordBuffer, MemoryAccounting, internal};

const SIZEOF_CORD: usize = core::mem::size_of::<Cord>();
const MAX_FLAT_LENGTH: usize = internal::MAX_FLAT_LENGTH;

// --- Helpers ----------------------------------------------------------------

/// A small deterministic PRNG (`SplitMix64`) standing in for `std::mt19937_64`.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// `GetUniformRandomUpTo`: a value in `[0, upper_bound)` (0 if empty).
    fn up_to(&mut self, upper_bound: usize) -> usize {
        if upper_bound > 0 { (self.next_u64() % upper_bound as u64) as usize } else { 0 }
    }

    fn coin_flip(&mut self) -> bool {
        self.next_u64() & 1 == 1
    }

    /// `GenerateSkewedRandom`.
    fn skewed(&mut self, max_log: u32) -> usize {
        let base = (self.next_u64() % u64::from(max_log + 1)) as u32;
        let mask = if base < 32 { (1u64 << base) - 1 } else { 0 };
        (self.next_u64() & mask) as usize
    }

    /// `RandomLowercaseString(rng, length)`.
    fn lowercase(&mut self, length: usize) -> Vec<u8> {
        (0..length).map(|_| b'a' + self.up_to(26) as u8).collect()
    }

    /// `RandomLowercaseString(rng)`: skewed length, rarely large.
    fn lowercase_skewed(&mut self) -> Vec<u8> {
        let roll = self.next_u64() % 10_000;
        let length = if roll == 0 {
            self.up_to(1_048_576)
        } else if roll < 10 {
            self.up_to(10_000)
        } else {
            self.skewed(10)
        };
        self.lowercase(length)
    }
}

/// `absl::MakeFragmentedCord`: every fragment becomes its own external node.
fn make_fragmented_cord<I, S>(fragments: I) -> Cord
where
    I: IntoIterator<Item = S>,
    S: AsRef<[u8]>,
{
    let mut result = Cord::new();
    for fragment in fragments {
        let mut tmp = internal::make_external(fragment.as_ref());
        tmp.prepend(&result);
        result = tmp;
    }
    result
}

/// `AddExternalMemory`: appends `s` as an external node.
fn add_external_memory(s: &[u8], dst: &mut Cord) {
    dst.append(internal::make_external(s));
}

/// `AppendWithFragments`: appends `s` in ~10 random fragments, half of them
/// external. Returns the size of the smallest fragment.
fn append_with_fragments(s: &[u8], rng: &mut Rng, cord: &mut Cord) -> usize {
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
            add_external_memory(&s[j..j + n], cord);
        } else {
            cord.append(&s[j..j + n]);
        }
        j += n;
    }
    min_size
}

/// `MakeComposite`: a cord out of many different node types.
fn make_composite() -> Cord {
    let mut cord = Cord::from("the");
    add_external_memory(b" quick brown", &mut cord);
    add_external_memory(b" fox jumped", &mut cord);

    let mut full = Cord::from(" over");
    add_external_memory(b" the lazy", &mut full);
    add_external_memory(b" dog slept the whole day away", &mut full);
    let mut substring = full.slice(0..18);

    // Make substring long enough to defeat the copying fast path in append.
    substring.append(vec![b'.'; 1000]);
    cord.append(substring);
    let len = cord.len();
    cord = cord.slice(0..len - 998); // Remove most of the extra junk.
    cord
}

fn hash<H: Hash>(h: &H) -> u64 {
    let mut s = DefaultHasher::new();
    h.hash(&mut s);
    s.finish()
}

fn check_valid(cord: &Cord) {
    internal::validate(cord).unwrap_or_else(|e| panic!("{e}\n{}", internal::dump(cord, false)));
}

// --- CordRepFlat ---------------------------------------------------------------

#[test]
fn flat_constants() {
    const { assert!(internal::FLAT_OVERHEAD < 32) };
    assert_eq!(CordBuffer::DEFAULT_LIMIT, MAX_FLAT_LENGTH);
    assert_eq!(MAX_FLAT_LENGTH, 4096 - internal::FLAT_OVERHEAD);
}

#[test]
fn all_flat_sizes() {
    for s in 0..MAX_FLAT_LENGTH {
        let src: Vec<u8> = (0..s).map(|i| b'a' + (i % 26) as u8).collect();
        let dst = Cord::from(&src[..]);
        assert_eq!(dst.to_vec(), src, "{s}");
    }
}

// --- CordTest -----------------------------------------------------------------

/// Creates a cord of at least 128 GB (2 GB on 32-bit) using reference counting.
#[test]
fn gigabyte_cord_from_external() {
    let one_gig: usize = 1024 * 1024 * 1024;
    // 128 GiB on 64-bit targets, 2 GiB on 32-bit ones (`checked_mul` keeps the
    // 64-bit constant from being rejected by the overflow lint on 32-bit).
    let max_size = one_gig.checked_mul(128).unwrap_or(2 * one_gig);
    let length = 128 * 1024;
    let from = internal::make_external(&vec![b'x'; length]);

    // Grow incrementally and exponentially so the tree needs rebalancing.
    let mut c = Cord::new();
    c.append(&from);
    while c.len() < max_size {
        let copy = c.clone();
        c.append(copy);
        c.append(&from);
        c.append(&from);
        c.append(&from);
        c.append(&from);
    }
    for _ in 0..1024 {
        c.append(&from);
    }
    assert!(c.len() >= max_size);
    check_valid(&c);
    assert_eq!(c[c.len() - 1], b'x');
}

fn make_external_cord(size: usize) -> Cord {
    internal::make_external(&vec![b'x'; size])
}

#[test]
fn assignment() {
    let mut x = Cord::from("hi there");
    let y = x.clone();
    assert_eq!(x.to_vec(), b"hi there");
    assert_eq!(y.to_vec(), b"hi there");
    assert_eq!(x, y);
    assert!(x <= y);
    assert!(y <= x);

    x = Cord::from("foo");
    assert_eq!(x.to_vec(), b"foo");
    assert_eq!(y.to_vec(), b"hi there");
    assert!(x < y);
    assert!(y > x);
    assert_ne!(x, y);
    assert!(x <= y);
    assert!(y >= x);

    x = Cord::from("foo");
    assert_eq!(x, "foo");

    // Going from inline rep to tree and back must not leak or break.
    let pairs = [
        ("hi there", "foo"),
        ("loooooong coooooord", "short cord"),
        ("short cord", "loooooong coooooord"),
        ("loooooong coooooord1", "loooooong coooooord2"),
    ];
    for (first, second) in pairs {
        let tmp = Cord::from(first);
        let mut z = tmp;
        assert_eq!(z.to_vec(), first.as_bytes());
        let tmp = Cord::from(second);
        z = tmp;
        assert_eq!(z.to_vec(), second.as_bytes());
        z.clone_from(&Cord::from(first));
        assert_eq!(z, first);
        let mut w = Cord::from(second);
        w.clone_from(&z);
        assert_eq!(w, first);
    }
    {
        // Self clone_from must not crash or leak.
        let mut small = Cord::from("foo");
        let mut big = Cord::from("loooooong coooooord");
        let small_alias = &raw mut small;
        let big_alias = &raw mut big;
        // SAFETY: aliasing the same cord for a self assignment test.
        unsafe {
            (*small_alias).clone_from(&*small_alias);
            (*big_alias).clone_from(&*big_alias);
        }
        assert_eq!(small, "foo");
        assert_eq!(big, "loooooong coooooord");
    }
}

#[test]
fn starts_ends_with() {
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
fn contains() {
    let flat_haystack = Cord::from("this is a flat cord");
    let fragmented_haystack =
        make_fragmented_cord(["this", " ", "is", " ", "a", " ", "fragmented", " ", "cord"]);

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
    assert!(!flat_haystack.contains(&make_fragmented_cord(["is", " ", "an"])));
    assert!(!fragmented_haystack.contains(&make_fragmented_cord(["is", " ", "an"])));

    assert!(flat_haystack.contains("is a"));
    assert!(fragmented_haystack.contains("is a"));
    assert!(flat_haystack.contains(&Cord::from("is a")));
    assert!(fragmented_haystack.contains(&Cord::from("is a")));
    assert!(flat_haystack.contains(&make_fragmented_cord(["is", " ", "a"])));
    assert!(fragmented_haystack.contains(&make_fragmented_cord(["is", " ", "a"])));
}

#[test]
fn find() {
    let flat_haystack = Cord::from("this is a flat cord");
    let fragmented_haystack =
        make_fragmented_cord(["this", " ", "is", " ", "a", " ", "fragmented", " ", "cord"]);
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
    assert_eq!(flat_haystack.find(&make_fragmented_cord(["is", " ", "an"])), None);
    assert_eq!(fragmented_haystack.find(&make_fragmented_cord(["is", " ", "an"])), None);

    assert_eq!(flat_haystack.find("is a"), Some(5));
    assert_eq!(fragmented_haystack.find("is a"), Some(5));
    assert_eq!(flat_haystack.find(&Cord::from("is a")), Some(5));
    assert_eq!(fragmented_haystack.find(&Cord::from("is a")), Some(5));
    assert_eq!(flat_haystack.find(&make_fragmented_cord(["is", " ", "a"])), Some(5));
    assert_eq!(fragmented_haystack.find(&make_fragmented_cord(["is", " ", "a"])), Some(5));
}

#[test]
fn find_across_fragment_boundaries_matches_slice_windows() {
    let bytes: Vec<u8> = (0..64).map(|i| (i * 17 % 251) as u8).collect();
    for chunk_size in 1..=12 {
        let haystack = make_fragmented_cord(bytes.chunks(chunk_size));
        for start in 0..bytes.len() {
            for end in (start + 1)..=(start + 12).min(bytes.len()) {
                let needle = &bytes[start..end];
                let expected = bytes.windows(needle.len()).position(|window| window == needle);
                assert_eq!(haystack.find(needle), expected, "chunk_size={chunk_size}, range={start}..{end}");
            }
        }
    }

    let repeated = vec![b'a'; 4096];
    let haystack = make_fragmented_cord(repeated.chunks(31));
    assert_eq!(haystack.find(&b"aaaaab"[..]), None);
    assert_eq!(haystack.find(&b"aaaaaa"[..]), Some(0));
}

/// `ends_with` against a genuinely fragmented suffix. `make_fragmented_cord`
/// gives every fragment its own external node (unlike plain `append`, which
/// coalesces small pieces back together), so this checks the fragmentation
/// held rather than assuming it.
#[test]
fn ends_with_fragmented_suffix() {
    let haystack = make_fragmented_cord(["this", " ", "is", " ", "a", " ", "fragmented", " ", "cord"]);
    assert!(haystack.chunks().count() > 1, "haystack must stay fragmented");

    // Suffix straddling multiple chunk boundaries.
    let suffix = make_fragmented_cord(["a", " ", "fragmented", " ", "cord"]);
    assert!(suffix.chunks().count() > 1, "suffix must stay fragmented");
    assert!(haystack.ends_with(&suffix));
    assert!(haystack.ends_with("a fragmented cord"));

    // Suffix == whole cord.
    let whole = make_fragmented_cord(["this", " ", "is", " ", "a", " ", "fragmented", " ", "cord"]);
    assert!(whole.chunks().count() > 1, "whole-cord suffix must stay fragmented");
    assert!(haystack.ends_with(&whole));

    // Mismatch in the last byte.
    let mismatch_last = make_fragmented_cord(["a", " ", "fragmented", " ", "corD"]);
    assert!(!haystack.ends_with(&mismatch_last));

    // Mismatch in the first suffix chunk.
    let mismatch_first = make_fragmented_cord(["X", " ", "fragmented", " ", "cord"]);
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
    let haystack = make_fragmented_cord(bytes.chunks(1));
    assert!(haystack.chunks().count() > 1, "haystack must stay fragmented");

    // The needle is the tail of the haystack, itself fragmented into
    // single-byte chunks: matching it exhausts both cursors on the same
    // step.
    let needle = make_fragmented_cord(bytes[30..].chunks(1));
    assert!(needle.chunks().count() > 1, "needle must stay fragmented");
    assert_eq!(needle.chunks().count(), bytes.len() - 30);
    assert_eq!(haystack.find(&needle), Some(30));

    // Same needle, but the very last byte differs: the mismatch is only
    // discovered on the final chunk pair, after the cursors have advanced
    // in lockstep through every preceding one.
    let mut mismatched = bytes[30..].to_vec();
    *mismatched.last_mut().unwrap() = mismatched.last().unwrap().wrapping_add(1);
    let needle_mismatched = make_fragmented_cord(mismatched.chunks(1));
    assert_eq!(haystack.find(&needle_mismatched), None);
}

#[test]
fn subcord() {
    let mut rng = Rng::new(1);
    let s = rng.lowercase(1024);

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
            check_valid(&sa);
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

    // Asking for too much is an error in Rust (abseil clamps).
    assert!(a.try_slice(0..=a.len()).is_none());
    assert!(a.try_slice((a.len() + 1)..=a.len()).is_none());
    assert!(a.try_slice(a.len()..a.len()).unwrap().is_empty());
}

#[test]
fn swap() {
    let mut x = Cord::from("Dexter");
    let mut y = Cord::from("Mandark");
    core::mem::swap(&mut x, &mut y);
    assert_eq!(x, Cord::from("Mandark"));
    assert_eq!(y, Cord::from("Dexter"));
    core::mem::swap(&mut x, &mut y);
    assert_eq!(x, Cord::from("Dexter"));
    assert_eq!(y, Cord::from("Mandark"));
}

fn verify_copy_to_string(cord: &Cord) {
    let initially_empty = cord.to_vec();
    assert_eq!(*cord, initially_empty);
    let mut has_initial_contents = vec![b'x'; 1024];
    has_initial_contents.clear();
    has_initial_contents.extend(cord.chunks().flatten());
    assert_eq!(*cord, has_initial_contents);
    let string: Result<String, _> = cord.clone().try_into();
    assert_eq!(string.unwrap(), cord.to_string());
}

#[test]
fn copy_to_string() {
    verify_copy_to_string(&Cord::new());
    verify_copy_to_string(&Cord::from("small cord"));
    verify_copy_to_string(&make_fragmented_cord([
        "fragmented ",
        "cord ",
        "to ",
        "test ",
        "copying ",
        "to ",
        "a ",
        "string.",
    ]));
}

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
fn append_to_string() {
    verify_append_cord_to_string(&Cord::new());
    verify_append_cord_to_string(&Cord::from("small cord"));
    verify_append_cord_to_string(&make_fragmented_cord([
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
fn copy_to_span() {
    verify_copy_to_span(&Cord::new());
    verify_copy_to_span(&Cord::from("small cord"));
    verify_copy_to_span(&make_fragmented_cord([
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
fn append_empty_buffer() {
    let mut cord = Cord::new();
    cord.append(CordBuffer::new());
    cord.append(CordBuffer::with_default_limit(2000));
    assert!(cord.is_empty());
}

#[test]
fn append_empty_buffer_to_flat() {
    let mut cord = Cord::from(vec![b'x'; 2000]);
    cord.append(CordBuffer::new());
    cord.append(CordBuffer::with_default_limit(2000));
    assert_eq!(cord.len(), 2000);
}

#[test]
fn append_empty_buffer_to_tree() {
    let mut cord = Cord::from(vec![b'x'; 2000]);
    cord.append(vec![b'y'; 2000]);
    cord.append(CordBuffer::new());
    cord.append(CordBuffer::with_default_limit(2000));
    assert_eq!(cord.len(), 4000);
}

#[test]
fn append_small_buffer() {
    let mut cord = Cord::new();
    let mut buffer = CordBuffer::with_default_limit(3);
    assert!(buffer.capacity() <= 15);
    buffer.put_slice(b"Abc");
    cord.append(buffer);

    let mut buffer = CordBuffer::with_default_limit(3);
    buffer.put_slice(b"defgh");
    cord.append(buffer);

    assert_eq!(cord.chunks().collect::<Vec<_>>(), vec![&b"Abcdefgh"[..]]);
}

#[test]
fn append_and_prepend_buffer_are_precise() {
    // Allow a 32 byte flat and 128 bytes for glue nodes.
    const MAX_DELTA: usize = 128 + 32;
    // Create a cord large enough to force 40KB of flats.
    let test_data = vec![b'x'; MAX_FLAT_LENGTH * 10];
    let mut cord1 = Cord::from(&test_data[..]);
    let mut cord2 = Cord::from(&test_data[..]);
    let size1 = cord1.estimated_memory_usage(MemoryAccounting::Total);
    let size2 = cord2.estimated_memory_usage(MemoryAccounting::Total);

    let mut buffer = CordBuffer::with_default_limit(3);
    buffer.put_slice(b"Abc");
    cord1.append(buffer);

    let mut buffer = CordBuffer::with_default_limit(3);
    buffer.put_slice(b"Abc");
    cord2.prepend(buffer);

    assert!(cord1.estimated_memory_usage(MemoryAccounting::Total) - size1 <= MAX_DELTA);
    assert!(cord2.estimated_memory_usage(MemoryAccounting::Total) - size2 <= MAX_DELTA);

    assert_eq!(cord1, [&test_data[..], b"Abc"].concat());
    assert_eq!(cord2, [b"Abc", &test_data[..]].concat());
}

#[test]
fn prepend_small_buffer() {
    let mut cord = Cord::new();
    let mut buffer = CordBuffer::with_default_limit(3);
    assert!(buffer.capacity() <= 15);
    buffer.put_slice(b"Abc");
    cord.prepend(buffer);

    let mut buffer = CordBuffer::with_default_limit(3);
    buffer.put_slice(b"defgh");
    cord.prepend(buffer);

    assert_eq!(cord.chunks().collect::<Vec<_>>(), vec![&b"defghAbc"[..]]);
}

#[test]
fn append_large_buffer() {
    let mut cord = Cord::new();
    let s1 = vec![b'1'; 700];
    let mut buffer = CordBuffer::with_default_limit(s1.len());
    buffer.put_slice(&s1);
    cord.append(buffer);

    let s2 = vec![b'2'; 1000];
    let mut buffer = CordBuffer::with_default_limit(s2.len());
    buffer.put_slice(&s2);
    cord.append(buffer);

    assert_eq!(cord.chunks().collect::<Vec<_>>(), vec![&s1[..], &s2[..]]);
}

#[test]
fn prepend_large_buffer() {
    let mut cord = Cord::new();
    let s1 = vec![b'1'; 700];
    let mut buffer = CordBuffer::with_default_limit(s1.len());
    buffer.put_slice(&s1);
    cord.prepend(buffer);

    let s2 = vec![b'2'; 1000];
    let mut buffer = CordBuffer::with_default_limit(s2.len());
    buffer.put_slice(&s2);
    cord.prepend(buffer);

    assert_eq!(cord.chunks().collect::<Vec<_>>(), vec![&s2[..], &s1[..]]);
}

// --- CordAppendBufferTest (parametrized over default / custom limit) ----------

struct AppendBufferParam {
    is_default: bool,
}

impl AppendBufferParam {
    const ALL: [AppendBufferParam; 2] =
        [AppendBufferParam { is_default: true }, AppendBufferParam { is_default: false }];

    fn limit(&self) -> usize {
        if self.is_default { CordBuffer::DEFAULT_LIMIT } else { CordBuffer::CUSTOM_LIMIT }
    }

    fn maximum_payload(&self) -> usize {
        if self.is_default {
            CordBuffer::maximum_payload()
        } else {
            CordBuffer::maximum_payload_for(self.limit())
        }
    }

    fn get_append_buffer(&self, cord: &mut Cord, capacity: usize, min_capacity: usize) -> CordBuffer {
        if self.is_default {
            cord.take_append_buffer_with(0, capacity, min_capacity)
        } else {
            cord.take_append_buffer_with(self.limit(), capacity, min_capacity)
        }
    }
}

#[test]
fn get_append_buffer_on_empty_cord() {
    for p in AppendBufferParam::ALL {
        let mut cord = Cord::new();
        let buffer = p.get_append_buffer(&mut cord, 1000, 16);
        assert!(buffer.capacity() >= 1000);
        assert_eq!(buffer.len(), 0);
    }
}

#[test]
fn get_append_buffer_on_inlined_cord() {
    let inlined_size = core::mem::size_of::<CordBuffer>() - 1;
    for p in AppendBufferParam::ALL {
        for size in [6, inlined_size - 3, inlined_size - 2, 1000] {
            let mut cord = Cord::from("Abc");
            let buffer = p.get_append_buffer(&mut cord, size, 1);
            assert!(buffer.capacity() >= 3 + size);
            assert_eq!(buffer.len(), 3);
            assert_eq!(&*buffer, b"Abc");
            assert!(cord.is_empty());
        }
    }
}

#[test]
fn get_append_buffer_on_inlined_cord_capacity_close_to_max() {
    // Asking for something like `usize::MAX - k` must not overflow on
    // `usize::MAX - k + size` and must return the maximum allowed size.
    for p in AppendBufferParam::ALL {
        for dist_from_max in 0..=4usize {
            let mut cord = Cord::from("Abc");
            let size = usize::MAX - dist_from_max;
            let buffer = p.get_append_buffer(&mut cord, size, 1);
            assert!(buffer.capacity() >= p.maximum_payload());
            assert_eq!(buffer.len(), 3);
            assert_eq!(&*buffer, b"Abc");
            assert!(cord.is_empty());
        }
    }
}

#[test]
fn get_append_buffer_on_flat() {
    for p in AppendBufferParam::ALL {
        // Create a cord with a single flat and extra capacity.
        let mut cord = Cord::new();
        let mut buffer = CordBuffer::with_default_limit(500);
        let expected_capacity = buffer.capacity();
        buffer.put_slice(b"Abc");
        cord.append(buffer);

        let buffer = p.get_append_buffer(&mut cord, 6, 16);
        assert_eq!(buffer.capacity(), expected_capacity);
        assert_eq!(buffer.len(), 3);
        assert_eq!(&*buffer, b"Abc");
        assert!(cord.is_empty());
    }
}

#[test]
fn get_append_buffer_on_flat_without_min_capacity() {
    for p in AppendBufferParam::ALL {
        let mut cord = Cord::new();
        let mut buffer = CordBuffer::with_default_limit(500);
        buffer.put_slice(&[b'x'; 30]);
        cord.append(buffer);

        let buffer = p.get_append_buffer(&mut cord, 1000, 900);
        assert!(buffer.capacity() >= 1000);
        assert_eq!(buffer.len(), 0);
        assert_eq!(cord, vec![b'x'; 30]);
    }
}

#[test]
fn get_append_buffer_on_tree() {
    let mut rng = Rng::new(7);
    for p in AppendBufferParam::ALL {
        for num_flats in [2, 3, 100] {
            // Create a cord with `num_flats` flats and extra capacity.
            let mut cord = Cord::new();
            let mut prefix = Vec::new();
            let mut last = Vec::new();
            for _ in 0..num_flats - 1 {
                prefix.extend_from_slice(&last);
                last = rng.lowercase(10);
                let mut buffer = CordBuffer::with_default_limit(500);
                buffer.put_slice(&last);
                cord.append(buffer);
            }
            let buffer = p.get_append_buffer(&mut cord, 6, 16);
            assert!(buffer.capacity() >= 500);
            assert_eq!(buffer.len(), 10);
            assert_eq!(&*buffer, &last[..]);
            assert_eq!(cord, prefix);
        }
    }
}

#[test]
fn get_append_buffer_on_tree_without_min_capacity() {
    for p in AppendBufferParam::ALL {
        let mut cord = Cord::new();
        for i in 0..2 {
            let mut buffer = CordBuffer::with_default_limit(500);
            buffer.put_slice(if i != 0 { b"def" } else { b"Abc" });
            cord.append(buffer);
        }
        let buffer = p.get_append_buffer(&mut cord, 1000, 900);
        assert!(buffer.capacity() >= 1000);
        assert_eq!(buffer.len(), 0);
        assert_eq!(cord, "Abcdef");
    }
}

#[test]
fn get_append_buffer_on_substring() {
    for p in AppendBufferParam::ALL {
        // A large cord with a single flat and some extra capacity.
        let mut cord = Cord::new();
        let mut buffer = CordBuffer::with_default_limit(500);
        buffer.put_slice(&[b'x'; 450]);
        cord.append(buffer);
        cord.advance(1);

        // Denied on a substring.
        let buffer = p.get_append_buffer(&mut cord, 6, 16);
        assert_eq!(buffer.len(), 0);
        assert_eq!(cord, vec![b'x'; 449]);
    }
}

#[test]
fn get_append_buffer_on_shared_cord() {
    for p in AppendBufferParam::ALL {
        // A shared cord with a single flat and extra capacity.
        let mut cord = Cord::new();
        let mut buffer = CordBuffer::with_default_limit(500);
        buffer.put_slice(b"Abc");
        cord.append(buffer);
        let _shared_cord = cord.clone();

        // Denied on a flat.
        let buffer = p.get_append_buffer(&mut cord, 6, 16);
        assert_eq!(buffer.len(), 0);
        assert_eq!(cord, "Abc");

        let mut buffer = CordBuffer::with_default_limit(500);
        buffer.put_slice(b"def");
        cord.append(buffer);
        let _shared_cord = cord.clone();

        // Denied on a tree.
        let buffer = p.get_append_buffer(&mut cord, 6, 16);
        assert_eq!(buffer.len(), 0);
        assert_eq!(cord, "Abcdef");
    }
}

// --- TryFlat / Flatten --------------------------------------------------------------

#[test]
fn try_flat_empty() {
    assert_eq!(Cord::new().as_flat(), Some(&b""[..]));
}

#[test]
fn try_flat_flat() {
    assert_eq!(Cord::from("hello").as_flat(), Some(&b"hello"[..]));
}

#[test]
fn try_flat_substr_inlined() {
    let mut c = Cord::from("hello");
    c.advance(1);
    assert_eq!(c.as_flat(), Some(&b"ello"[..]));
}

#[test]
fn try_flat_substr_flat() {
    let c = Cord::from("longer than 15 bytes");
    let sub = internal::make_substring(&c, 1, c.len() - 1);
    assert_eq!(sub.as_flat(), Some(&b"onger than 15 bytes"[..]));
}

#[test]
fn try_flat_concat() {
    let c = make_fragmented_cord(["hel", "lo"]);
    assert_eq!(c.as_flat(), None);
}

#[test]
fn try_flat_external() {
    let c = internal::make_external(b"hell");
    assert_eq!(c.as_flat(), Some(&b"hell"[..]));
}

#[test]
fn try_flat_substr_external() {
    let c = internal::make_external(b"hell");
    let sub = internal::make_substring(&c, 1, c.len() - 1);
    assert_eq!(sub.as_flat(), Some(&b"ell"[..]));
}

/// Not part of the API contract, but intended to be true of the current
/// implementation: sub-cords of whole chunks are flat.
#[test]
fn try_flat_commonly_assumed_invariants() {
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
    let c = make_fragmented_cord(fragments);
    let mut offset = 0;
    let mut cursor = c.cursor();
    for (fragment, sv) in c.chunks().enumerate() {
        let expected = fragments[fragment].as_bytes();
        let subcord1 = c.slice(offset..offset + sv.len());
        let subcord2 = cursor.read(sv.len());
        assert_eq!(subcord1.as_flat(), Some(expected));
        assert_eq!(subcord2.as_flat(), Some(expected));
        offset += sv.len();
    }
}

fn is_flat(c: &Cord) -> bool {
    c.chunks().count() <= 1
}

fn verify_flatten(mut c: Cord) {
    let old_contents = c.to_vec();
    let already_flat_and_non_empty = is_flat(&c) && !c.is_empty();
    let old_flat_ptr =
        if already_flat_and_non_empty { Some(c.chunks().next().unwrap().as_ptr()) } else { None };
    let new_flat = c.flatten();
    assert_eq!(new_flat, &old_contents[..]);
    if let Some(old_ptr) = old_flat_ptr {
        assert_eq!(old_ptr, new_flat.as_ptr(), "Allocated new memory even though the Cord was already flat.");
    }
    assert_eq!(c.to_vec(), old_contents);
    assert!(is_flat(&c));
}

#[test]
fn flatten() {
    verify_flatten(Cord::new());
    verify_flatten(Cord::from("small cord"));
    verify_flatten(Cord::from("larger than small buffer optimization"));
    verify_flatten(make_fragmented_cord(["small ", "fragmented ", "cord"]));
    // Longer than the largest flat buffer.
    let mut rng = Rng::new(3);
    verify_flatten(Cord::from(rng.lowercase(8192)));
}

// --- MultipleLengths -------------------------------------------------------------------

struct TestData {
    data: Vec<Vec<u8>>,
}

impl TestData {
    fn make_string(length: usize) -> Vec<u8> {
        let buf = format!("({length})");
        let mut result = Vec::new();
        while result.len() < length {
            result.extend_from_slice(buf.as_bytes());
        }
        result.truncate(length);
        result
    }

    fn new() -> Self {
        // Strings around half of the maximum flat length (32-bit-correct:
        // derived from `MAX_FLAT_LENGTH`, not a hardcoded 64-bit value).
        const HALF: usize = MAX_FLAT_LENGTH / 2;
        let mut data = Vec::new();
        // Short strings increasing in length by one.
        for i in 0..30 {
            data.push(Self::make_string(i));
        }
        for i in -10i64..=10 {
            data.push(Self::make_string((HALF as i64 + i) as usize));
        }
        for i in -10i64..=10 {
            data.push(Self::make_string((MAX_FLAT_LENGTH as i64 + i) as usize));
        }
        Self { data }
    }
}

#[test]
fn multiple_lengths() {
    let d = TestData::new();
    for a in &d.data {
        {
            // Construct from Cord.
            let tmp = Cord::from(&a[..]);
            let x = tmp.clone();
            assert_eq!(x.to_vec(), *a);
        }
        {
            // Construct from slice.
            let x = Cord::from(&a[..]);
            assert_eq!(x.to_vec(), *a);
        }
        {
            // Append cord to self.
            let mut this = Cord::from(&a[..]);
            let copy = this.clone();
            this.append(copy);
            assert_eq!(this.to_vec(), [&a[..], &a[..]].concat());
        }
        {
            // Prepend cord to self.
            let mut this = Cord::from(&a[..]);
            let copy = this.clone();
            this.prepend(copy);
            assert_eq!(this.to_vec(), [&a[..], &a[..]].concat());
        }
        // Try to append / prepend others.
        for b in &d.data {
            {
                // clone_from Cord.
                let mut x = Cord::from(&a[..]);
                let y = Cord::from(&b[..]);
                x.clone_from(&y);
                assert_eq!(x.to_vec(), *b);
            }
            {
                // Assign from slice.
                let mut x = Cord::from(&a[..]);
                x = Cord::from(&b[..]);
                assert_eq!(x.to_vec(), *b);
            }
            {
                // append(&Cord)
                let mut x = Cord::from(&a[..]);
                let y = Cord::from(&b[..]);
                x.append(&y);
                assert_eq!(x.to_vec(), [&a[..], &b[..]].concat());
            }
            {
                // append(&[u8])
                let mut x = Cord::from(&a[..]);
                x.append(&b[..]);
                assert_eq!(x.to_vec(), [&a[..], &b[..]].concat());
            }
            {
                // prepend(&Cord)
                let mut x = Cord::from(&a[..]);
                let y = Cord::from(&b[..]);
                x.prepend(&y);
                assert_eq!(x.to_vec(), [&b[..], &a[..]].concat());
            }
            {
                // prepend(&[u8])
                let mut x = Cord::from(&a[..]);
                x.prepend(&b[..]);
                assert_eq!(x.to_vec(), [&b[..], &a[..]].concat());
            }
        }
    }
}

#[test]
fn remove_suffix_with_external_or_substring() {
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
fn remove_suffix_makes_zero_length_node() {
    let mut c = Cord::new();
    c.append(Cord::from(vec![b'x'; 100]));
    let other_ref = c.clone(); // Prevent in place appends.
    assert_eq!(other_ref, c);
    c.append(Cord::from(vec![b'y'; 200]));
    c.truncate(c.len() - 200);
    assert_eq!(c.to_vec(), vec![b'x'; 100]);
}

// --- CordSpliceTest -----------------------------------------------------------------

fn cord_with_zed_block(size: usize) -> Cord {
    internal::make_external(&vec![b'z'; size])
}

#[test]
fn cord_splice_test_zed_block() {
    let blob = cord_with_zed_block(10);
    assert_eq!(blob.len(), 10);
    assert_eq!(blob.to_vec(), b"zzzzzzzzzz");
}

#[test]
fn cord_splice_test_zed_block0() {
    let blob = cord_with_zed_block(0);
    assert_eq!(blob.len(), 0);
    assert_eq!(blob.to_vec(), b"");
}

#[test]
fn cord_splice_test_zed_block_suffix1() {
    let blob = cord_with_zed_block(10);
    assert_eq!(blob.len(), 10);
    let mut suffix = blob.clone();
    suffix.advance(9);
    assert_eq!(suffix.len(), 1);
    assert_eq!(suffix.to_vec(), b"z");
}

#[test]
fn cord_splice_test_zed_block_suffix0() {
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
fn cord_splice_test_remove_entire_block1() {
    let zero = cord_with_zed_block(10);
    let mut suffix = zero.clone();
    suffix.advance(10);
    let mut result = Cord::new();
    result.append(suffix);
    assert!(result.is_empty());
}

#[test]
fn cord_splice_test_remove_entire_block2() {
    let zero = cord_with_zed_block(10);
    let mut prefix = zero.clone();
    prefix.truncate(0);
    let mut suffix = zero.clone();
    suffix.advance(10);
    let mut result = prefix.clone();
    result.append(suffix);
    assert!(result.is_empty());
}

#[test]
fn cord_splice_test_remove_entire_block3() {
    let blob = cord_with_zed_block(10);
    let block = big_cord(10, b'b');
    let blob = splice_cord(&blob, 0, &block);
    assert_eq!(blob, "bbbbbbbbbb");
}

// --- Compare ---------------------------------------------------------------------

fn verify_comparison(lhs: &Cord, rhs: &Cord) {
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
fn compare() {
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
    ];
    for (lhs, rhs) in &cases {
        verify_comparison(lhs, rhs);
    }
}

#[test]
fn compare_after_assign() {
    let mut a = Cord::from("aaaaaa1111111");
    let mut b = Cord::from("aaaaaa2222222");
    a = Cord::from("cccccc");
    b = Cord::from("cccccc");
    assert_eq!(a, b);
    assert!(a >= b);

    a = Cord::from("aaaa");
    b = Cord::from("bbbbb");
    a = Cord::from("");
    b = Cord::from("");
    assert_eq!(a, b);
    assert!(a >= b);
}

fn test_compare(c: &Cord, d: &Cord) {
    let expected = c.to_vec().cmp(&d.to_vec());
    assert_eq!(c.compare(d), expected, "{c:?}, {d:?}");
}

#[test]
fn compare_comparison_is_unsigned() {
    let mut rng = Rng::new(11);
    let x = rng.up_to(256) as u8;
    let n1 = rng.up_to(100);
    let n2 = rng.up_to(100);
    test_compare(&Cord::from(vec![x; n1]), &Cord::from(vec![x ^ 0x80; n2]));
    assert_eq!(Cord::from(b"\x80").compare(b"\x7f"), std::cmp::Ordering::Greater);
}

#[test]
fn compare_random_comparisons() {
    let iters = 5000;
    let mut rng = Rng::new(42);
    let n = rng.up_to(5000);
    let a = [
        make_external_cord(n),
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
        test_compare(&c, &d);
    }
}

#[test]
#[allow(clippy::neg_cmp_op_on_partial_ord)]
fn comparison_operators() {
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

// --- External memory ---------------------------------------------------------------
//
// abseil's releaser variants (function pointers, move-only / non-const /
// no-arg callables, reference qualifier overloads) are C++ specific. The
// observable behavior -- the owner is released exactly once, when the last
// reference goes away -- is tested through `Arc` strong counts.

#[test]
fn construct_from_external_releaser_invoked() {
    // Empty external memory: nothing is retained.
    {
        let owner: Arc<Vec<u8>> = Arc::new(Vec::new());
        let c = Cord::from(owner.clone());
        assert_eq!(c, "");
        assert_eq!(Arc::strong_count(&owner), 1);
    }
    // Large data is shared, not copied, until the last cord goes away.
    let large_dummy: Arc<Vec<u8>> = Arc::new(vec![b'c'; 2048]);
    {
        let c = Cord::from(large_dummy.clone());
        assert_eq!(c, *large_dummy);
        assert_eq!(Arc::strong_count(&large_dummy), 2);
    }
    assert_eq!(Arc::strong_count(&large_dummy), 1);
    {
        let copy;
        {
            let c = Cord::from(large_dummy.clone());
            copy = c.clone();
            assert_eq!(Arc::strong_count(&large_dummy), 2);
        }
        assert_eq!(Arc::strong_count(&large_dummy), 2);
        drop(copy);
    }
    assert_eq!(Arc::strong_count(&large_dummy), 1);
}

#[test]
fn construct_from_external_compare_contents() {
    let mut rng = Rng::new(5);
    let mut length = 1;
    while length <= 2048 {
        let data = rng.lowercase(length);
        let cord = internal::make_external(&data);
        assert_eq!(cord, data);
        let shared: Arc<[u8]> = Arc::from(&data[..]);
        let cord = Cord::from(shared.clone());
        assert_eq!(cord, data);
        if length > 511 {
            assert_eq!(cord.as_flat().unwrap().as_ptr(), shared.as_ptr(), "large Arc data is shared");
        }
        length *= 2;
    }
}

#[test]
fn external_memory_basic_usage() {
    for s in [&b""[..], b"hello", b"there"] {
        let mut dst = Cord::from("(prefix)");
        add_external_memory(s, &mut dst);
        dst.append("(suffix)");
        assert_eq!(dst.to_vec(), [b"(prefix)", s, b"(suffix)"].concat());
    }
}

#[test]
fn external_memory_remove_prefix_suffix() {
    // Exhaustively try all sub-strings.
    let cord = make_composite();
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

#[test]
fn external_memory_get() {
    let mut cord = Cord::from("hello");
    add_external_memory(b" world!", &mut cord);
    add_external_memory(b" how are ", &mut cord);
    cord.append(" you?");
    let s = cord.to_vec();
    for (i, &b) in s.iter().enumerate() {
        assert_eq!(b, cord[i]);
    }
}

// --- Memory usage --------------------------------------------------------------------

const FAIR_SHARE: MemoryAccounting = MemoryAccounting::FairShare;
const TOTAL: MemoryAccounting = MemoryAccounting::Total;
const TOTAL_MORE_PRECISE: MemoryAccounting = MemoryAccounting::TotalMorePrecise;

/// Creates a cord of `n` bytes `c` without adopting the source buffer.
fn make_cord(n: usize, c: u8) -> Cord {
    Cord::from(&vec![c; n][..])
}

#[test]
fn cord_memory_usage_empty() {
    let cord = Cord::new();
    assert_eq!(cord.estimated_memory_usage(TOTAL), SIZEOF_CORD);
    assert_eq!(cord.estimated_memory_usage(FAIR_SHARE), SIZEOF_CORD);
    assert_eq!(cord.estimated_memory_usage(TOTAL_MORE_PRECISE), SIZEOF_CORD);
}

#[test]
fn cord_memory_usage_inlined() {
    let a = Cord::from("hello");
    assert_eq!(a.estimated_memory_usage(TOTAL), SIZEOF_CORD);
    assert_eq!(a.estimated_memory_usage(FAIR_SHARE), SIZEOF_CORD);
    assert_eq!(a.estimated_memory_usage(TOTAL_MORE_PRECISE), SIZEOF_CORD);
}

#[test]
fn cord_memory_usage_external_memory() {
    let mut cord = Cord::new();
    add_external_memory(&[b'x'; 1000], &mut cord);
    let expected = SIZEOF_CORD + 1000 + internal::EXTERNAL_NODE_SIZE;
    assert_eq!(cord.estimated_memory_usage(TOTAL), expected);
    assert_eq!(cord.estimated_memory_usage(FAIR_SHARE), expected);
    assert_eq!(cord.estimated_memory_usage(TOTAL_MORE_PRECISE), expected);
}

#[test]
fn cord_memory_usage_flat() {
    let cord = make_cord(1000, b'a');
    let flat_size = internal::flat_allocated_size(&cord).unwrap();
    assert_eq!(cord.estimated_memory_usage(TOTAL), SIZEOF_CORD + flat_size);
    assert_eq!(cord.estimated_memory_usage(FAIR_SHARE), SIZEOF_CORD + flat_size);
    assert_eq!(cord.estimated_memory_usage(TOTAL_MORE_PRECISE), SIZEOF_CORD + flat_size);
}

#[test]
fn cord_memory_usage_sub_string_shared_flat() {
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
fn cord_memory_usage_flat_shared() {
    let shared = make_cord(1000, b'a');
    let cord = shared.clone();
    let flat_size = internal::flat_allocated_size(&cord).unwrap();
    assert_eq!(cord.estimated_memory_usage(TOTAL), SIZEOF_CORD + flat_size);
    assert_eq!(cord.estimated_memory_usage(TOTAL_MORE_PRECISE), SIZEOF_CORD + flat_size);
    assert_eq!(cord.estimated_memory_usage(FAIR_SHARE), SIZEOF_CORD + flat_size / 2);
}

#[test]
fn cord_memory_usage_btree() {
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

#[test]
fn test_hash_fragmentation() {
    // Hits the 1024 byte hashing block boundaries precisely.
    let cords = [
        Cord::new(),
        make_fragmented_cord([vec![b'a'; 600], vec![b'a'; 600]]),
        make_fragmented_cord([vec![b'a'; 1200]]),
        make_fragmented_cord([vec![b'b'; 900], vec![b'b'; 900]]),
        make_fragmented_cord([vec![b'b'; 1800]]),
        make_fragmented_cord([vec![b'c'; 2000], vec![b'c'; 2000]]),
        make_fragmented_cord([vec![b'c'; 4000]]),
        make_fragmented_cord([vec![b'd'; 1024]]),
        make_fragmented_cord([vec![b'd'; 1023], b"d".to_vec()]),
        make_fragmented_cord([vec![b'e'; 1025]]),
        make_fragmented_cord([vec![b'e'; 1024], b"e".to_vec()]),
        make_fragmented_cord([vec![b'e'; 1023], b"e".to_vec(), b"e".to_vec()]),
    ];
    for a in &cords {
        for b in &cords {
            assert_eq!(hash(a) == hash(b), a == b, "{a:?} vs {b:?}");
        }
    }
}

/// Regression test: going from the inline rep to a tree too soon was
/// observable through memory usage.
#[test]
fn cord_memory_usage_inline_rep() {
    let small_string = [b'x'; internal::MAX_INLINE];
    let c1 = Cord::from(&small_string[..]);
    let mut c2 = Cord::new();
    c2.append(&small_string[..]);
    assert_eq!(c1, c2);
    assert_eq!(c1.estimated_memory_usage(TOTAL), c2.estimated_memory_usage(TOTAL));
}

#[test]
fn cord_memory_usage_total_more_precise_mode() {
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
fn cord_memory_usage_total_more_precise_mode_with_substring() {
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

// --- Growth -----------------------------------------------------------------------------

/// Regression test: appending to a copy must not modify the original.
#[test]
fn concat_append() {
    let mut s1 = Cord::from("foobarbarbarbarbar");
    s1.append("abcdefgabcdefgabcdefgabcdefgabcdefgabcdefgabcdefg");
    let size = s1.len();

    let mut s2 = s1.clone();
    s2.append("x");

    assert_eq!(s1.len(), size);
    assert_eq!(s2.len(), size + 1);
}

/// A diabolical `append(<one byte>)` loop where the cord is shared before
/// every append, producing a terribly fragmented cord.
#[test]
fn diabolical_growth() {
    let mut rng = Rng::new(9);
    let expected = rng.lowercase(5000);
    let mut cord = Cord::new();
    for &c in &expected {
        let shared = cord.clone();
        assert_eq!(cord, shared);
        cord.append(&[c][..]);
    }
    assert_eq!(cord.to_vec(), expected);
    check_valid(&cord);
}

// `HugeCord` (a >4 GB cord built from an external node with a faked length)
// cannot be expressed with real owners; the length accounting it exercises is
// covered by `gigabyte_cord_from_external`.

/// `append` works when handed a reference to (a clone of) itself.
#[test]
fn append_self() {
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
fn make_fragmented_cord_from_list() {
    let fragmented = make_fragmented_cord(["A ", "fragmented ", "Cord"]);
    assert_eq!(fragmented, "A fragmented Cord");
    let chunks: Vec<&[u8]> = fragmented.chunks().collect();
    assert_eq!(chunks, vec![&b"A "[..], b"fragmented ", b"Cord"]);
}

#[test]
fn make_fragmented_cord_from_vector() {
    let chunks = vec!["A ", "fragmented ", "Cord"];
    let fragmented = make_fragmented_cord(&chunks);
    assert_eq!(fragmented, "A fragmented Cord");
    let chunks: Vec<&[u8]> = fragmented.chunks().collect();
    assert_eq!(chunks, vec![&b"A "[..], b"fragmented ", b"Cord"]);
}

// --- Chunk iteration ------------------------------------------------------------------

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
    assert_eq!(cord.into_iter().count(), expected_chunks);
}

#[test]
fn cord_chunk_iterator_operations() {
    verify_chunk_iterator(&Cord::new(), 0);
    verify_chunk_iterator(&Cord::from("small cord"), 1);
    verify_chunk_iterator(&Cord::from("larger than small buffer optimization"), 1);
    verify_chunk_iterator(
        &make_fragmented_cord([
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
    for _ in 0..8 {
        let copy = reused_nodes_cord.clone();
        reused_nodes_cord.prepend(copy);
        expected_chunks *= 2;
        verify_chunk_iterator(&reused_nodes_cord, expected_chunks);
    }

    let mut rng = Rng::new(13);
    let flat_cord = Cord::from(rng.lowercase(256));
    let mut subcords = Cord::new();
    for i in 0..128 {
        subcords.prepend(flat_cord.slice(i..i + 128));
    }
    verify_chunk_iterator(&subcords, 128);
}

#[test]
fn advance_and_read_on_data_edge() {
    let mut rng = Rng::new(17);
    let data = rng.lowercase(2000);
    for as_flat in [true, false] {
        let cord = if as_flat { Cord::from(&data[..]) } else { internal::make_external(&data) };

        let mut it = cord.cursor();
        let frag = it.read(2000);
        assert_eq!(frag, data);
        assert!(it.is_empty());

        let mut it = cord.cursor();
        let frag = it.read(200);
        assert_eq!(frag, &data[..200]);
        assert!(!it.is_empty());

        let frag = it.read(1500);
        assert_eq!(frag, &data[200..1700]);
        assert!(!it.is_empty());

        let frag = it.read(300);
        assert_eq!(frag, &data[1700..2000]);
        assert!(it.is_empty());
    }
}

#[test]
#[should_panic(expected = "cannot read past the end")]
fn advance_and_read_beyond_end_panics() {
    let cord = Cord::from(vec![b'x'; 2000]);
    let mut it = cord.cursor();
    let _ = it.read(2001);
}

#[test]
fn advance_and_read_on_substring_data_edge() {
    let mut rng = Rng::new(19);
    let data = rng.lowercase(2500);
    for as_flat in [true, false] {
        let cord = if as_flat { Cord::from(&data[..]) } else { internal::make_external(&data) };
        let cord = cord.slice(200..2200);
        let substr = &data[200..2200];

        let mut it = cord.cursor();
        let frag = it.read(2000);
        assert_eq!(frag, substr);
        assert!(it.is_empty());

        let mut it = cord.cursor();
        let frag = it.read(200);
        assert_eq!(frag, &substr[..200]);
        assert!(!it.is_empty());

        let frag = it.read(1500);
        assert_eq!(frag, &substr[200..1700]);
        assert!(!it.is_empty());

        let frag = it.read(300);
        assert_eq!(frag, &substr[1700..2000]);
        assert!(it.is_empty());
    }
}

// --- Char iteration --------------------------------------------------------------------

fn verify_char_iterator(cord: &Cord) {
    assert_eq!(cord.cursor().is_empty(), cord.is_empty());
    assert_eq!(cord.cursor().remaining(), cord.len());
    assert_eq!(cord.bytes().len(), cord.len());

    let content = cord.to_vec();
    let mut i = 0;
    let mut pre_iter = cord.cursor();
    let mut post_iter = cord.bytes();
    while !pre_iter.is_empty() {
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
        assert_eq!(advance_iter.read(i), cord.slice(..i));
        assert_eq!(advance_iter.position(), i);

        let mut advance_iter = pre_iter.clone();
        advance_iter.advance(cord.len() - i);
        assert!(advance_iter.is_empty());
        assert_eq!(advance_iter.position(), cord.len());
        assert_eq!(advance_iter.remaining(), 0);

        let mut advance_iter = pre_iter.clone();
        assert_eq!(advance_iter.read(cord.len() - i), cord.slice(i..));
        assert!(advance_iter.is_empty());

        i += 1;
        assert_eq!(pre_iter.next_byte(), Some(content[i - 1]));
        assert_eq!(post_iter.next(), Some(content[i - 1]));
    }
    assert_eq!(i, cord.len());
    assert!(post_iter.next().is_none());

    let mut zero_advanced_end = cord.cursor();
    zero_advanced_end.advance(cord.len());
    zero_advanced_end.advance(0);
    assert!(zero_advanced_end.is_empty());

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
fn char_iterator_operations() {
    verify_char_iterator(&Cord::new());
    verify_char_iterator(&Cord::from("small cord"));
    verify_char_iterator(&Cord::from("larger than small buffer optimization"));
    verify_char_iterator(&make_fragmented_cord([
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

    let mut rng = Rng::new(23);
    let flat_cord = Cord::from(rng.lowercase(256));
    let mut subcords = Cord::new();
    for i in 0..4 {
        subcords.prepend(flat_cord.slice(16 * i..16 * i + 128));
    }
    verify_char_iterator(&subcords);
}

/// Six flats of 2500 bytes read in chunks of 150, 1500, 2500 and 3000 bytes,
/// covering partial, full and straddled reads including reads below the copy
/// threshold. b/197776822 surfaced a bug for a specific small read at the end.
#[test]
fn char_iterator_advance_and_read() {
    const BLOCKS: usize = 6;
    const BLOCK_SIZE: usize = 2500;
    let mut rng = Rng::new(29);
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
            let chunk = it.read(n);
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
fn streaming_output() {
    let c = make_fragmented_cord(["A ", "small ", "fragmented ", "Cord", "."]);
    assert_eq!(c.to_string(), "A small fragmented Cord.");
    assert_eq!(format!("{c}"), "A small fragmented Cord.");
}

#[test]
fn for_each_chunk() {
    for num_elements in [1, 10, 200] {
        let cord_chunks: Vec<String> = (0..num_elements).map(|i| format!("[{i}]")).collect();
        let c = make_fragmented_cord(&cord_chunks);
        let iterated: Vec<String> = c.chunks().map(|c| String::from_utf8(c.to_vec()).unwrap()).collect();
        assert_eq!(iterated, cord_chunks);
    }
}

#[test]
fn small_buffer_assign_from_own_data() {
    let contents = b"small buff cord";
    assert_eq!(contents.len(), internal::MAX_INLINE);
    for pos in 0..contents.len() {
        for count in (1..=contents.len() - pos).rev() {
            let mut c = Cord::from(&contents[..]);
            let sub = c.flatten()[pos..pos + count].to_vec();
            c = Cord::from(&sub[..]);
            assert_eq!(c, &contents[pos..pos + count], "pos = {pos}; count = {count}");
        }
    }
}

#[test]
fn format() {
    let mut c = Cord::new();
    write!(c, "There were {:04} little pigs.", 3).unwrap();
    assert_eq!(c, "There were 0003 little pigs.");
    write!(c, "And {:<3x} bad wolf!", 1).unwrap();
    assert_eq!(c, "There were 0003 little pigs.And 1   bad wolf!");
}

#[test]
fn stringify() {
    let c = make_fragmented_cord(["A ", "small ", "fragmented ", "Cord", "."]);
    assert_eq!(c.to_string(), "A small fragmented Cord.");
}

#[test]
#[should_panic(expected = "cannot advance past end")]
fn hardening_remove_prefix() {
    let mut cord = Cord::from("hello");
    cord.advance(6);
}

#[test]
#[should_panic(expected = "index out of bounds")]
fn hardening_index() {
    let cord = Cord::from("hello");
    let _ = cord[5];
}

#[test]
fn hardening_truncate_is_lenient() {
    // Unlike abseil's RemoveSuffix, truncate follows `Vec::truncate`.
    let mut cord = Cord::from("hello");
    cord.truncate(6);
    assert_eq!(cord, "hello");
}

/// Mimics an application repeatedly splitting a cord, overwriting a value and
/// recomposing the pieces. This is hostile towards a btree: splits share the
/// boundary nodes and recomposition injects edges, quickly growing the tree
/// to its maximum height.
#[test]
fn btree_hostile_split_insert_join() {
    let mut rng = Rng::new(31);
    let data = vec![b'x'; 1 << 10];
    let buffer = Cord::from(&data[..]);
    let mut cord = Cord::new();
    let appends = if cfg!(debug_assertions) { 100_000 } else { 1_000_000 };
    for _ in 0..appends {
        cord.append(&buffer);
    }

    for _ in 0..1000 {
        let offset = rng.up_to(cord.len());
        let length = 100 + rng.up_to(data.len() - 100);
        if cord.len() == offset {
            cord.append(&data[..length]);
        } else {
            let mut suffix = Cord::new();
            if offset + length < cord.len() {
                suffix = cord.clone();
                suffix.advance(offset + length);
            }
            if cord.len() > offset {
                cord.truncate(offset);
            }
            cord.append(&data[..length]);
            if !suffix.is_empty() {
                cord.append(suffix);
            }
        }
    }
    check_valid(&cord);
    assert!(cord.len() >= appends * data.len() - 1000 * data.len());

    // Even under this adversarial split/join pattern, height must stay
    // within a conservative O(log n) bound instead of degenerating toward a
    // linear chain. The crate doesn't guarantee tight (max-fanout) packing —
    // this workload is deliberately hostile to packing — so assume nodes
    // may be as sparse as half of `BTREE_MAX_CAPACITY`, plus a couple of
    // levels of slack; that's loose enough to tolerate the adversarial
    // workload while still catching gross (e.g. near-linear) degeneracy.
    let chunks = cord.chunks().count();
    let height = internal::btree_height(&cord).expect("a cord this large must be a btree");
    let min_fanout = internal::BTREE_MAX_CAPACITY / 2;
    let mut bound = 0usize;
    let mut reachable = 1usize;
    while reachable < chunks {
        reachable *= min_fanout;
        bound += 1;
    }
    bound += 2; // slack
    assert!(
        height <= bound,
        "btree height {height} exceeds conservative O(log_{min_fanout} n) bound {bound} for {chunks} chunks"
    );
}

// --- Static / after exit --------------------------------------------------------------

static SHORT_CORD: LazyLock<Cord> = LazyLock::new(|| Cord::from_static("SSO string"));
static LONG_CORD: LazyLock<Cord> = LazyLock::new(|| Cord::from_static("String that does not fit SSO."));

fn test_after_exit(cord: &Cord, expected: &'static str) {
    assert_eq!(*cord, expected);
    {
        let copy = cord.clone();
        assert_eq!(copy, expected);
    }
    assert_eq!(*cord, expected);
    {
        let mut copy = cord.clone();
        let mut expected_copy = expected.to_string();
        for _ in 0..10 {
            copy.append(cord);
            expected_copy.push_str(expected);
            assert_eq!(copy, expected_copy);
        }
    }
    assert_eq!(internal::is_tree(cord), cord.len() >= 16);
    for _ in 0..10 {
        assert_eq!(Cord::from_static(expected), *cord);
    }
}

#[test]
fn after_exit() {
    static DATA: [u8; 64] = [b'q'; 64];
    test_after_exit(&SHORT_CORD, "SSO string");
    test_after_exit(&LONG_CORD, "String that does not fit SSO.");
    // Static data is referenced, not copied.
    let c = Cord::from_static(&DATA);
    assert_eq!(c.as_flat().unwrap().as_ptr(), DATA.as_ptr());
}

// --- Populated cord factories and mutators (from the checksum tests) ------------------
//
// The checksum expectations are out of scope, but the factory / mutator matrix
// is a good structural test: every mutation must leave a valid cord and every
// undoable mutation must restore the original value.

struct PopulatedCordFactory {
    name: &'static str,
    generator: fn() -> Cord,
}

const CORD_FACTORIES: &[PopulatedCordFactory] = &[
    PopulatedCordFactory { name: "sso", generator: || Cord::from("abcde") },
    PopulatedCordFactory {
        name: "flat",
        generator: || {
            // Too large for SSO, small enough to be a single flat.
            let mut flat = Cord::from([b"abcde".as_slice(), &[b'x'; 1000]].concat());
            flat.flatten();
            flat
        },
    },
    PopulatedCordFactory { name: "external", generator: || internal::make_external(b"abcde External!") },
    PopulatedCordFactory {
        name: "external substring",
        generator: || {
            let ext = internal::make_external(b"-abcde External!");
            internal::make_substring(&ext, 1, ext.len() - 1)
        },
    },
    PopulatedCordFactory {
        name: "substring",
        generator: || {
            let mut flat = Cord::from([b"-abcde".as_slice(), &[b'x'; 1000]].concat());
            flat.flatten();
            flat.slice(1..999)
        },
    },
    PopulatedCordFactory {
        name: "fragmented",
        generator: || {
            let fragment = [b"abcde".as_slice(), &[b'x'; 195]].concat();
            let cord = make_fragmented_cord(std::iter::repeat_n(fragment, 200));
            assert_eq!(cord.len(), 40000);
            cord
        },
    },
];

struct CordMutator {
    name: &'static str,
    mutate: fn(&mut Cord),
    undo: Option<fn(&mut Cord)>,
}

const CORD_MUTATORS: &[CordMutator] = &[
    CordMutator { name: "clear", mutate: |c| c.clear(), undo: None },
    CordMutator { name: "overwrite", mutate: |c| *c = Cord::from("overwritten"), undo: None },
    CordMutator {
        name: "append string",
        mutate: |c| c.append("0123456789"),
        undo: Some(|c| c.truncate(c.len() - 10)),
    },
    CordMutator {
        name: "append cord",
        mutate: |c| c.append(make_fragmented_cord(["12345", "67890"])),
        undo: Some(|c| c.truncate(c.len() - 10)),
    },
    CordMutator {
        name: "append self",
        mutate: |c| {
            let copy = c.clone();
            c.append(copy);
        },
        undo: Some(|c| c.truncate(c.len() / 2)),
    },
    CordMutator { name: "append empty string", mutate: |c| c.append(""), undo: Some(|_| {}) },
    CordMutator { name: "append empty cord", mutate: |c| c.append(Cord::new()), undo: Some(|_| {}) },
    CordMutator {
        name: "prepend string",
        mutate: |c| c.prepend("9876543210"),
        undo: Some(|c| c.advance(10)),
    },
    CordMutator {
        name: "prepend cord",
        mutate: |c| c.prepend(make_fragmented_cord(["98765", "43210"])),
        undo: Some(|c| c.advance(10)),
    },
    CordMutator { name: "prepend empty string", mutate: |c| c.prepend(""), undo: Some(|_| {}) },
    CordMutator { name: "prepend empty cord", mutate: |c| c.prepend(Cord::new()), undo: Some(|_| {}) },
    CordMutator {
        name: "prepend self",
        mutate: |c| {
            let copy = c.clone();
            c.prepend(copy);
        },
        undo: Some(|c| c.advance(c.len() / 2)),
    },
    CordMutator { name: "remove prefix", mutate: |c| c.advance(c.len() / 2), undo: None },
    CordMutator { name: "remove suffix", mutate: |c| c.truncate(c.len() / 2), undo: None },
    CordMutator { name: "remove 0-prefix", mutate: |c| c.advance(0), undo: None },
    CordMutator { name: "remove 0-suffix", mutate: |c| c.truncate(c.len()), undo: None },
    CordMutator { name: "subcord", mutate: |c| *c = c.slice(1..c.len() - 1), undo: None },
    CordMutator {
        name: "swap inline",
        mutate: |c| {
            let mut other = Cord::from("swap");
            core::mem::swap(c, &mut other);
        },
        undo: None,
    },
    CordMutator {
        name: "swap tree",
        mutate: |c| {
            let mut other = Cord::from(vec![b'x'; 10000]);
            core::mem::swap(c, &mut other);
        },
        undo: None,
    },
    CordMutator { name: "split off", mutate: |c| drop(c.split_off(c.len() / 2)), undo: None },
    CordMutator { name: "split to", mutate: |c| drop(c.split_to(c.len() / 2)), undo: None },
    CordMutator {
        name: "flatten",
        mutate: |c| {
            let _ = c.flatten();
        },
        undo: Some(|_| {}),
    },
];

#[test]
fn factories_and_mutators() {
    for factory in CORD_FACTORIES {
        for shared in [false, true] {
            let shared_cord_source = (factory.generator)();
            let make_instance = || if shared { shared_cord_source.clone() } else { (factory.generator)() };

            let base_value = (factory.generator)();
            let base_value_as_bytes = (factory.generator)().to_vec();
            assert!(base_value.starts_with("abcde"), "{}", factory.name);
            check_valid(&base_value);

            let c1 = make_instance();
            assert_eq!(c1, base_value);

            for mutator in CORD_MUTATORS {
                let mut c2 = make_instance();
                (mutator.mutate)(&mut c2);
                check_valid(&c2);
                if let Some(undo) = mutator.undo {
                    undo(&mut c2);
                    check_valid(&c2);
                    assert_eq!(c2, base_value, "{} / {} / shared {shared}", factory.name, mutator.name);
                }
            }

            // All reading operations function on any representation.
            let cc3 = make_instance();
            assert_eq!(cc3.len(), base_value_as_bytes.len());
            assert!(!cc3.is_empty());
            assert_eq!(cc3.compare(&base_value), std::cmp::Ordering::Equal);
            assert_eq!(cc3.compare(&base_value_as_bytes[..]), std::cmp::Ordering::Equal);
            assert_eq!(cc3.compare("wxyz"), std::cmp::Ordering::Less);
            assert_eq!(cc3.compare(&Cord::from("wxyz")), std::cmp::Ordering::Less);
            assert_eq!(cc3.compare("aaaa"), std::cmp::Ordering::Greater);
            assert_eq!(cc3.compare(&Cord::from("aaaa")), std::cmp::Ordering::Greater);
            assert_eq!(Cord::from("wxyz").compare(&cc3), std::cmp::Ordering::Greater);
            assert_eq!(Cord::from("aaaa").compare(&cc3), std::cmp::Ordering::Less);
            assert!(cc3.starts_with("abcd"));
            assert_eq!(cc3.to_vec(), base_value_as_bytes);
            assert!(cc3.chunks().next().unwrap().starts_with(b"abcde"));
            assert_eq!(cc3.bytes().next(), Some(b'a'));

            let mut char_it = cc3.cursor();
            char_it.advance(2);
            assert_eq!(char_it.read(2), "cd");
            assert_eq!(char_it.peek(), Some(b'e'));
            let mut char_it = cc3.cursor();
            char_it.advance(2);
            assert!(char_it.chunk().starts_with(b"cde"));

            assert_eq!(cc3[0], b'a');
            assert_eq!(cc3[4], b'e');
            assert_eq!(hash(&cc3), hash(&base_value));
        }
    }
}

// `CordSanitizerTest` (a false positive report under sanitizers), the CRC
// tests and the C++20 `<=>` tests have no Rust equivalent; ordering is
// covered by `comparison_operators` and `Ord`:
#[test]
fn three_way_comparison() {
    use std::cmp::Ordering;
    assert_eq!(Cord::from("a").cmp(&Cord::from("a")), Ordering::Equal);
    assert_eq!(Cord::from("aaaa").cmp(&Cord::from("aaab")), Ordering::Less);
    assert_eq!(Cord::from("baaa").cmp(&Cord::from("a")), Ordering::Greater);
    assert_eq!("a".partial_cmp(&Cord::from("a")), Some(Ordering::Equal));
    assert_eq!(Cord::from("a").partial_cmp("b"), Some(Ordering::Less));
    assert_eq!("b".partial_cmp(&Cord::from("a")), Some(Ordering::Greater));
}

#[test]
fn many_appends_stay_valid() {
    // `DumpGrowth` from the C++ file: 1000 single byte appends.
    let mut s = Cord::new();
    for i in 0..1000usize {
        s.append(&[b'a' + (i % 26) as u8][..]);
    }
    assert_eq!(s.len(), 1000);
    check_valid(&s);
    let mut rng = Rng::new(37);
    for _ in 0..200 {
        s.append(rng.lowercase_skewed());
    }
    check_valid(&s);
}
