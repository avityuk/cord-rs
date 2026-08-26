//! Helpers shared by the public-API test binaries.
#![allow(dead_code, reason = "compiled into every test binary; not every binary uses every helper")]
#![expect(
    clippy::cast_possible_truncation,
    reason = "the test PRNG and data generators juggle small integers freely"
)]

use core::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;

pub(crate) use cord_rs::__internal as internal;
use cord_rs::Cord;

/// Full oracle check: validates the tree and cross-checks every read path
/// against `expected`.
pub(crate) fn check(cord: &Cord, expected: &[u8]) {
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

/// Validates `cord`'s tree structure only, with no contents check; panics
/// with a dump on failure.
pub(crate) fn assert_valid(cord: &Cord) {
    internal::validate(cord).unwrap_or_else(|e| panic!("{e}\n{}", internal::dump(cord, false)));
}

/// A small deterministic PRNG (`SplitMix64`) standing in for `std::mt19937_64`.
pub(crate) struct Rng(u64);

impl Rng {
    pub(crate) fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub(crate) fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// `GetUniformRandomUpTo`: a value in `[0, upper_bound)` (0 if empty).
    pub(crate) fn up_to(&mut self, upper_bound: usize) -> usize {
        if upper_bound > 0 { (self.next_u64() % upper_bound as u64) as usize } else { 0 }
    }

    pub(crate) fn coin_flip(&mut self) -> bool {
        self.next_u64() & 1 == 1
    }

    /// `GenerateSkewedRandom`.
    pub(crate) fn skewed(&mut self, max_log: u32) -> usize {
        let base = (self.next_u64() % u64::from(max_log + 1)) as u32;
        let mask = if base < 32 { (1u64 << base) - 1 } else { 0 };
        (self.next_u64() & mask) as usize
    }

    /// `RandomLowercaseString(rng, length)`.
    pub(crate) fn lowercase(&mut self, length: usize) -> Vec<u8> {
        (0..length).map(|_| b'a' + self.up_to(26) as u8).collect()
    }

    /// `RandomLowercaseString(rng)`: skewed length, rarely large.
    pub(crate) fn lowercase_skewed(&mut self) -> Vec<u8> {
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
pub(crate) fn make_fragmented_cord<I, S>(fragments: I) -> Cord
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
pub(crate) fn add_external_memory(s: &[u8], dst: &mut Cord) {
    dst.append(internal::make_external(s));
}

/// `MakeComposite`: a cord out of many different node types.
pub(crate) fn make_composite_cord() -> Cord {
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

/// The default (`SipHash`) hash of `value`, via `std`'s `DefaultHasher`.
pub(crate) fn default_hash<T: Hash + ?Sized>(value: &T) -> u64 {
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}

/// A `Hasher` that mixes once per `write` call, so the *boundaries* between
/// calls are part of the result.
///
/// `DefaultHasher` absorbs bytes into an internal buffer and is insensitive
/// to how a value splits them across `write` calls, which is precisely what
/// `Hash for Cord` has to get right: it re-blocks the chunks into fixed size
/// writes so the call sequence depends only on the bytes. Hashing with this
/// instead turns "the same bytes, chunked differently" into a real
/// assertion. (FxHash-style mixing: cheap, deterministic, and not a
/// cryptographic claim.)
#[derive(Default)]
pub(crate) struct ChunkSensitiveHasher {
    state: u64,
}

impl Hasher for ChunkSensitiveHasher {
    fn write(&mut self, bytes: &[u8]) {
        const SEED: u64 = 0x517C_C1B7_2722_0A95;
        for &b in bytes {
            self.state = (self.state.rotate_left(5) ^ u64::from(b)).wrapping_mul(SEED);
        }
        // Mix in the call boundary itself: a differently batched but
        // byte-identical sequence lands on a different state.
        self.state = (self.state.rotate_left(5) ^ bytes.len() as u64).wrapping_mul(SEED);
    }

    fn finish(&self) -> u64 {
        self.state
    }
}

/// `default_hash`'s boundary-sensitive counterpart.
pub(crate) fn boundary_hash<T: Hash + ?Sized>(value: &T) -> u64 {
    let mut hasher = ChunkSensitiveHasher::default();
    value.hash(&mut hasher);
    hasher.finish()
}
