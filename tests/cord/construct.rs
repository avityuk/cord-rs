//! Constructors, conversions in/out, adoption thresholds, external
//! ownership, statics.
#![expect(clippy::cast_possible_truncation, reason = "tests juggle small integers freely")]

use std::borrow::Cow;
use std::sync::{Arc, LazyLock};

use crate::common::{self, internal};
use cord_rs::Cord;

#[test]
fn flat_lengths_round_trip() {
    const MAX_FLAT_LENGTH: usize = internal::MAX_FLAT_LENGTH;
    let lengths: Vec<usize> = if cfg!(miri) {
        let mut v: Vec<usize> = (0..MAX_FLAT_LENGTH).step_by(97).collect();
        v.extend([0, 1, 15, 16, MAX_FLAT_LENGTH - 1]);
        v
    } else {
        (0..MAX_FLAT_LENGTH).collect()
    };
    for s in lengths {
        let src: Vec<u8> = (0..s).map(|i| b'a' + (i % 26) as u8).collect();
        let dst = Cord::from(&src[..]);
        assert_eq!(dst.to_vec(), src, "{s}");
    }
}

#[test]
fn clone_from_switches_between_inline_and_tree() {
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
}

#[test]
fn external_owner_is_released_when_the_last_cord_drops() {
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
fn external_data_is_shared_at_every_length() {
    let mut rng = common::Rng::new(5);
    let mut length = 1;
    while length <= 2048 {
        let data = rng.lowercase(length);
        let cord = internal::make_external(&data);
        assert_eq!(cord, data);
        let shared: Arc<[u8]> = Arc::from(&data[..]);
        let cord = Cord::from(shared.clone());
        assert_eq!(cord, data);
        if length > 511 {
            assert_eq!(cord.as_contiguous().unwrap().as_ptr(), shared.as_ptr(), "large Arc data is shared");
        }
        length *= 2;
    }
}

#[test]
fn external_nodes_join_inline_neighbours() {
    for s in [&b""[..], b"hello", b"there"] {
        let mut dst = Cord::from("(prefix)");
        common::add_external_memory(s, &mut dst);
        dst.append("(suffix)");
        assert_eq!(dst.to_vec(), [b"(prefix)", s, b"(suffix)"].concat());
    }
}

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
fn static_cords_stay_usable_and_share_their_bytes() {
    test_after_exit(&SHORT_CORD, "SSO string");
    test_after_exit(&LONG_CORD, "String that does not fit SSO.");
}

#[test]
fn inline_and_tree_thresholds() {
    let cord = Cord::default();
    common::check(&cord, b"");

    let cord = Cord::new();
    common::check(&cord, b"");
    assert_eq!(cord.as_contiguous(), Some(&b""[..]));
    assert!(!internal::is_tree(&cord));
    let cord = Cord::from("hello");
    common::check(&cord, b"hello");
    assert!(!internal::is_tree(&cord));
    let cord = Cord::from("123456789012345");
    common::check(&cord, b"123456789012345");
    assert!(!internal::is_tree(&cord));
    let cord = Cord::from("1234567890123456");
    common::check(&cord, b"1234567890123456");
    assert!(internal::is_tree(&cord));
    assert!(internal::is_flat(&cord));
    assert_eq!(core::mem::size_of::<Cord>(), 16);
    assert_eq!(core::mem::size_of::<Option<Cord>>(), 16 + core::mem::size_of::<usize>());

    // `Cord::from` for every sub-range of a 15-byte (`MAX_INLINE`) string.
    let contents = b"small buff cord";
    assert_eq!(contents.len(), internal::MAX_INLINE);
    for pos in 0..contents.len() {
        for count in (1..=contents.len() - pos).rev() {
            let mut c = Cord::from(&contents[..]);
            let sub = c.make_contiguous()[pos..pos + count].to_vec();
            c = Cord::from(&sub[..]);
            assert_eq!(c, &contents[pos..pos + count], "pos = {pos}; count = {count}");
        }
    }
}

#[test]
fn construction_from_slices_strings_and_iterators() {
    let big: Vec<u8> = (0..10_000u32).map(|i| (i % 251) as u8).collect();
    common::check(&Cord::from(&big[..]), &big);
    common::check(&Cord::from(big.clone()), &big);
    common::check(&Cord::from(big.clone().into_boxed_slice()), &big);
    common::check(&Cord::from(Arc::<[u8]>::from(&big[..])), &big);
    common::check(
        &Cord::from(String::from_utf8_lossy(&big).into_owned()),
        String::from_utf8_lossy(&big).as_bytes(),
    );
    common::check(&Cord::copy_from_slice(&big), &big);
    common::check(&big.iter().copied().collect::<Cord>(), &big);
    common::check(&big.chunks(7).collect::<Cord>(), &big);

    // `FromStr` never fails; it's equivalent to `Cord::from`.
    assert_eq!("parsed text".parse::<Cord>().unwrap(), Cord::from("parsed text"));

    // `Box<str>`, `&Vec<u8>`, `&String`.
    let boxed: Box<str> = "boxed str".into();
    common::check(&Cord::from(boxed), b"boxed str");
    let v: Vec<u8> = b"by-ref vec".to_vec();
    common::check(&Cord::from(&v), &v);
    let s: String = "by-ref string".to_string();
    common::check(&Cord::from(&s), s.as_bytes());
}

#[test]
fn owned_buffers_are_adopted_above_the_copy_threshold() {
    let big: Vec<u8> = (0..10_000u32).map(|i| (i % 251) as u8).collect();

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

    // `Cow<[u8]>`: `Borrowed` always copies (`From<&[u8]>`), `Owned` follows
    // the `Vec<u8>` adopt-or-copy threshold (`From<Vec<u8>>`).
    let small_bytes = b"cow borrowed".to_vec();
    common::check(&Cord::from(Cow::Borrowed(&small_bytes[..])), &small_bytes);
    let borrowed_big: Cow<'_, [u8]> = Cow::Borrowed(&big[..]);
    let from_borrowed_big = Cord::from(borrowed_big);
    common::check(&from_borrowed_big, &big);
    assert!(!internal::is_external(&from_borrowed_big), "borrowed slice is always copied");
    let owned_vec = vec![4u8; 10_000];
    let owned_vec_ptr = owned_vec.as_ptr();
    let owned_bytes: Cow<'_, [u8]> = Cow::Owned(owned_vec);
    let from_owned_vec = Cord::from(owned_bytes);
    assert!(internal::is_external(&from_owned_vec), "owned vec via Cow should be adopted");
    assert_eq!(from_owned_vec.as_contiguous().unwrap().as_ptr(), owned_vec_ptr);

    // `Cow<str>`: same, via `From<&str>` / `From<String>`.
    let small_str = "cow borrowed str".to_string();
    common::check(&Cord::from(Cow::Borrowed(small_str.as_str())), small_str.as_bytes());
    let big_str = "z".repeat(10_000);
    let borrowed_big_str: Cow<'_, str> = Cow::Borrowed(&big_str);
    let from_borrowed_big_str = Cord::from(borrowed_big_str);
    common::check(&from_borrowed_big_str, big_str.as_bytes());
    assert!(!internal::is_external(&from_borrowed_big_str), "borrowed str is always copied");
    let owned_string = "w".repeat(10_000);
    let owned_string_ptr = owned_string.as_ptr();
    let owned_str: Cow<'_, str> = Cow::Owned(owned_string);
    let from_owned_string = Cord::from(owned_str);
    assert!(internal::is_external(&from_owned_string), "owned string via Cow should be adopted");
    assert_eq!(from_owned_string.as_contiguous().unwrap().as_ptr(), owned_string_ptr);
}

#[test]
fn static_and_arc_owners_share_their_bytes() {
    static STATIC: [u8; 100] = [9; 100];

    let s = Cord::from_static(&STATIC);
    common::check(&s, &STATIC);
    assert!(internal::is_external(&s));
    assert_eq!(s.as_contiguous().unwrap().as_ptr(), STATIC.as_ptr(), "from_static must not copy");
    common::check(&Cord::from_static("static str"), b"static str");
    let arc: Arc<str> = Arc::from("x".repeat(1000).as_str());
    let c = Cord::from(arc.clone());
    assert_eq!(Arc::strong_count(&arc), 2);
    drop(c);
    assert_eq!(Arc::strong_count(&arc), 1);

    let arc_string = Arc::new("arc string".to_string());
    common::check(&Cord::from(arc_string.clone()), b"arc string");

    // Large enough to be adopted (shared) rather than copied.
    let big_string = Arc::new("y".repeat(10_000));
    let before = Arc::strong_count(&big_string);
    let shared = Cord::from(big_string.clone());
    assert_eq!(Arc::strong_count(&big_string), before + 1);
    common::check(&shared, big_string.as_bytes());
    drop(shared);
    assert_eq!(Arc::strong_count(&big_string), before);
}
