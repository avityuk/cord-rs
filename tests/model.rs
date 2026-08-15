//! Property based model test: random operation sequences on `Cord`s are
//! checked against `Vec<u8>` oracles, with tree validation after every step.
// proptest is not available on wasm targets (see Cargo.toml).
#![cfg(not(target_family = "wasm"))]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    reason = "tests juggle small integers freely"
)]

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use cord_rs::{Cord, CordBuffer, internal};
use proptest::prelude::*;

const SLOTS: usize = 4;

#[derive(Debug, Clone)]
enum Op {
    AppendSlice { t: usize, data: Vec<u8> },
    AppendOwned { t: usize, data: Vec<u8> },
    AppendCord { t: usize, s: usize },
    AppendOwnedClone { t: usize, s: usize },
    AppendBuffer { t: usize, data: Vec<u8>, reuse: bool },
    PrependSlice { t: usize, data: Vec<u8> },
    PrependOwned { t: usize, data: Vec<u8> },
    PrependCord { t: usize, s: usize },
    PrependOwnedClone { t: usize, s: usize },
    Advance { t: usize, frac: f64 },
    Truncate { t: usize, frac: f64 },
    Slice { t: usize, s: usize, a: f64, b: f64 },
    SplitOff { t: usize, s: usize, frac: f64 },
    SplitTo { t: usize, s: usize, frac: f64 },
    Clone { t: usize, s: usize },
    Clear { t: usize },
    Flatten { t: usize },
    CursorRead { t: usize, s: usize, a: f64, b: f64 },
    ExtendBytes { t: usize, data: Vec<u8> },
}

fn data() -> impl Strategy<Value = Vec<u8>> {
    prop_oneof![
        4 => prop::collection::vec(any::<u8>(), 0..40),
        3 => prop::collection::vec(any::<u8>(), 0..600),
        2 => prop::collection::vec(any::<u8>(), 0..5000),
        1 => prop::collection::vec(any::<u8>(), 4000..20_000),
    ]
}

fn slot() -> impl Strategy<Value = usize> {
    0..SLOTS
}

fn frac() -> impl Strategy<Value = f64> {
    0.0..=1.0f64
}

fn op() -> impl Strategy<Value = Op> {
    prop_oneof![
        (slot(), data()).prop_map(|(t, data)| Op::AppendSlice { t, data }),
        (slot(), data()).prop_map(|(t, data)| Op::AppendOwned { t, data }),
        (slot(), slot()).prop_map(|(t, s)| Op::AppendCord { t, s }),
        (slot(), slot()).prop_map(|(t, s)| Op::AppendOwnedClone { t, s }),
        (slot(), data(), any::<bool>()).prop_map(|(t, data, reuse)| Op::AppendBuffer { t, data, reuse }),
        (slot(), data()).prop_map(|(t, data)| Op::PrependSlice { t, data }),
        (slot(), data()).prop_map(|(t, data)| Op::PrependOwned { t, data }),
        (slot(), slot()).prop_map(|(t, s)| Op::PrependCord { t, s }),
        (slot(), slot()).prop_map(|(t, s)| Op::PrependOwnedClone { t, s }),
        (slot(), frac()).prop_map(|(t, frac)| Op::Advance { t, frac }),
        (slot(), frac()).prop_map(|(t, frac)| Op::Truncate { t, frac }),
        (slot(), slot(), frac(), frac()).prop_map(|(t, s, a, b)| Op::Slice { t, s, a, b }),
        (slot(), slot(), frac()).prop_map(|(t, s, frac)| Op::SplitOff { t, s, frac }),
        (slot(), slot(), frac()).prop_map(|(t, s, frac)| Op::SplitTo { t, s, frac }),
        (slot(), slot()).prop_map(|(t, s)| Op::Clone { t, s }),
        slot().prop_map(|t| Op::Clear { t }),
        slot().prop_map(|t| Op::Flatten { t }),
        (slot(), slot(), frac(), frac()).prop_map(|(t, s, a, b)| Op::CursorRead { t, s, a, b }),
        (slot(), prop::collection::vec(any::<u8>(), 0..700))
            .prop_map(|(t, data)| Op::ExtendBytes { t, data }),
    ]
}

fn index(frac: f64, len: usize) -> usize {
    ((frac * len as f64) as usize).min(len)
}

fn range(a: f64, b: f64, len: usize) -> (usize, usize) {
    let (x, y) = (index(a, len), index(b, len));
    if x <= y { (x, y) } else { (y, x) }
}

fn hash<H: Hash>(h: &H) -> u64 {
    let mut s = DefaultHasher::new();
    h.hash(&mut s);
    s.finish()
}

fn check(cord: &Cord, expected: &[u8], step: usize) {
    if let Err(e) = internal::validate(cord) {
        panic!("step {step}: invalid tree: {e}\n{}", internal::dump(cord, false));
    }
    assert_eq!(cord.len(), expected.len(), "step {step}: len");
    assert!(cord == expected, "step {step}: content differs");
    let joined: Vec<u8> = cord
        .chunks()
        .inspect(|c| assert!(!c.is_empty(), "step {step}: empty chunk"))
        .flatten()
        .copied()
        .collect();
    assert_eq!(joined, expected, "step {step}: chunks");
    if let Some(flat) = cord.as_flat() {
        assert_eq!(flat, expected, "step {step}: as_flat");
    }
}

fn deep_check(cord: &Cord, expected: &[u8], step: usize, a: f64, b: f64) {
    let len = expected.len();
    assert_eq!(cord.to_vec(), expected, "step {step}: to_vec");
    let bytes: Vec<u8> = cord.bytes().collect();
    assert_eq!(bytes, expected, "step {step}: bytes()");
    if len > 0 {
        let i = index(a, len - 1);
        assert_eq!(cord[i], expected[i], "step {step}: index {i}");
        assert_eq!(cord.get(i), Some(expected[i]));
    }
    assert_eq!(cord.get(len), None);
    let (x, y) = range(a, b, len);
    let sub = cord.slice(x..y);
    assert!(sub == expected[x..y], "step {step}: slice {x}..{y}");
    let needle = &expected[x..y];
    let found = cord.find(needle);
    let oracle =
        if needle.is_empty() { Some(0) } else { expected.windows(needle.len()).position(|w| w == needle) };
    assert_eq!(found, oracle, "step {step}: find {x}..{y}");
    assert_eq!(cord.find(&sub), oracle, "step {step}: find cord {x}..{y}");
    assert!(cord.starts_with(&expected[..x]), "step {step}: starts_with");
    assert!(cord.ends_with(&expected[y..]), "step {step}: ends_with");
    assert_eq!(cord.contains(needle), oracle.is_some());

    // The checks above only ever probe genuine prefixes/suffixes/substrings
    // of `expected`, so an implementation that just returned `true`
    // unconditionally would pass them. Corrupt each one (flip a byte, or
    // extend an empty needle past the boundary) and compare against the
    // `Vec` oracle's real (often negative) answer instead of a fixed
    // expectation.
    let mutate = |mut bytes: Vec<u8>| -> Vec<u8> {
        if bytes.is_empty() {
            bytes.push(0xA5); // extend an empty needle past the boundary
        } else {
            let mid = bytes.len() / 2;
            bytes[mid] ^= 0xFF; // flip a byte partway through
        }
        bytes
    };
    let bad_prefix = mutate(expected[..x].to_vec());
    assert_eq!(
        cord.starts_with(&bad_prefix),
        expected.starts_with(&bad_prefix[..]),
        "step {step}: starts_with mutated"
    );
    let bad_suffix = mutate(expected[y..].to_vec());
    assert_eq!(
        cord.ends_with(&bad_suffix),
        expected.ends_with(&bad_suffix[..]),
        "step {step}: ends_with mutated"
    );
    let bad_needle = mutate(needle.to_vec());
    let bad_oracle = expected.windows(bad_needle.len()).position(|w| w == bad_needle);
    assert_eq!(cord.find(&bad_needle), bad_oracle, "step {step}: find mutated {x}..{y}");
    assert_eq!(cord.contains(&bad_needle), bad_oracle.is_some(), "step {step}: contains mutated");

    let rebuilt = Cord::from(expected);
    assert_eq!(hash(cord), hash(&rebuilt), "step {step}: hash");
    assert_eq!(cord.cmp(&rebuilt), std::cmp::Ordering::Equal);
    let mut copy = vec![0u8; len / 2];
    assert_eq!(cord.copy_prefix_to(&mut copy), len / 2);
    assert_eq!(&copy[..], &expected[..len / 2]);
    assert_eq!(cord.to_string(), String::from_utf8_lossy(expected), "step {step}: display");
}

proptest! {
    // Miri interprets every case; 200 of them is an hour, not a check. And
    // proptest's default file-backed failure persistence calls getcwd at
    // startup, which Miri's isolation forbids — turn it off under Miri
    // rather than loosening isolation for the whole test.
    #![proptest_config(ProptestConfig {
        cases: if cfg!(miri) { 8 } else { 200 },
        failure_persistence: if cfg!(miri) {
            Some(Box::new(proptest::test_runner::FileFailurePersistence::Off))
        } else {
            ProptestConfig::default().failure_persistence
        },
        ..ProptestConfig::default()
    })]

    #[test]
    fn cord_matches_vec_oracle(
        ops in prop::collection::vec(op(), 1..if cfg!(miri) { 24 } else { 120 }),
        checks in prop::collection::vec((frac(), frac()), 1..if cfg!(miri) { 24 } else { 120 }),
    ) {
        let mut cords: Vec<Cord> = (0..SLOTS).map(|_| Cord::new()).collect();
        let mut oracles: Vec<Vec<u8>> = vec![Vec::new(); SLOTS];
        for (step, op) in ops.iter().enumerate() {
            let t = match op {
                Op::AppendSlice { t, data } => { cords[*t].append(&data[..]); oracles[*t].extend_from_slice(data); *t }
                Op::AppendOwned { t, data } => { cords[*t].append(data.clone()); oracles[*t].extend_from_slice(data); *t }
                Op::AppendCord { t, s } => { let src = cords[*s].clone(); cords[*t].append(&src); let o = oracles[*s].clone(); oracles[*t].extend_from_slice(&o); *t }
                Op::AppendOwnedClone { t, s } => { let src = cords[*s].clone(); cords[*t].append(src); let o = oracles[*s].clone(); oracles[*t].extend_from_slice(&o); *t }
                Op::AppendBuffer { t, data, reuse } => {
                    let mut buffer = if *reuse { cords[*t].take_append_buffer(data.len()) } else { CordBuffer::with_default_limit(data.len()) };
                    let n = buffer.put_slice_partial(data);
                    cords[*t].append(buffer);
                    oracles[*t].extend_from_slice(&data[..n]);
                    *t
                }
                Op::PrependSlice { t, data } => { cords[*t].prepend(&data[..]); let mut o = data.clone(); o.extend_from_slice(&oracles[*t]); oracles[*t] = o; *t }
                Op::PrependOwned { t, data } => { cords[*t].prepend(data.clone()); let mut o = data.clone(); o.extend_from_slice(&oracles[*t]); oracles[*t] = o; *t }
                Op::PrependCord { t, s } => { let src = cords[*s].clone(); cords[*t].prepend(&src); let mut o = oracles[*s].clone(); o.extend_from_slice(&oracles[*t]); oracles[*t] = o; *t }
                Op::PrependOwnedClone { t, s } => { let src = cords[*s].clone(); cords[*t].prepend(src); let mut o = oracles[*s].clone(); o.extend_from_slice(&oracles[*t]); oracles[*t] = o; *t }
                Op::Advance { t, frac } => { let n = index(*frac, oracles[*t].len()); cords[*t].advance(n); oracles[*t].drain(..n); *t }
                Op::Truncate { t, frac } => { let n = index(*frac, oracles[*t].len()); cords[*t].truncate(n); oracles[*t].truncate(n); *t }
                Op::Slice { t, s, a, b } => { let (x, y) = range(*a, *b, oracles[*s].len()); cords[*t] = cords[*s].slice(x..y); oracles[*t] = oracles[*s][x..y].to_vec(); *t }
                Op::SplitOff { t, s, frac } => {
                    let n = index(*frac, oracles[*s].len());
                    let tail = cords[*s].split_off(n);
                    let otail = oracles[*s].split_off(n);
                    check(&cords[*s], &oracles[*s], step);
                    cords[*t] = tail; oracles[*t] = otail; *t
                }
                Op::SplitTo { t, s, frac } => {
                    let n = index(*frac, oracles[*s].len());
                    let head = cords[*s].split_to(n);
                    let ohead: Vec<u8> = oracles[*s].drain(..n).collect();
                    check(&cords[*s], &oracles[*s], step);
                    cords[*t] = head; oracles[*t] = ohead; *t
                }
                Op::Clone { t, s } => { cords[*t] = cords[*s].clone(); oracles[*t] = oracles[*s].clone(); *t }
                Op::Clear { t } => { cords[*t].clear(); oracles[*t].clear(); *t }
                Op::Flatten { t } => { let flat = cords[*t].flatten().to_vec(); assert_eq!(flat, oracles[*t], "step {step}: flatten"); *t }
                Op::CursorRead { t, s, a, b } => {
                    let (x, y) = range(*a, *b, oracles[*s].len());
                    let mut cursor = cords[*s].cursor();
                    cursor.advance(x);
                    let read = cursor.read_cord(y - x);
                    assert_eq!(cursor.position(), y);
                    assert_eq!(cursor.remaining(), oracles[*s].len() - y);
                    let rest: Vec<u8> = cursor.chunks().flatten().copied().collect();
                    assert_eq!(rest, &oracles[*s][y..], "step {step}: cursor rest");
                    cords[*t] = read; oracles[*t] = oracles[*s][x..y].to_vec(); *t
                }
                Op::ExtendBytes { t, data } => { cords[*t].extend(data.iter().copied()); oracles[*t].extend_from_slice(data); *t }
            };
            check(&cords[t], &oracles[t], step);
            if let Some((a, b)) = checks.get(step) {
                deep_check(&cords[t], &oracles[t], step, *a, *b);
            }
        }
        for i in 0..SLOTS {
            check(&cords[i], &oracles[i], usize::MAX);
            for j in 0..SLOTS {
                assert_eq!(cords[i].cmp(&cords[j]), oracles[i].cmp(&oracles[j]), "final cmp {i} {j}");
                assert_eq!(cords[i] == cords[j], oracles[i] == oracles[j]);
            }
        }
    }
}
