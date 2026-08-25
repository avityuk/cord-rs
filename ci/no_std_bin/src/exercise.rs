//! `no_std` + `alloc` exercise of the `cord-rs` public surface. Compiled and
//! linked for `aarch64-unknown-none` (see `main.rs`): a real allocator, a
//! real entry point, no `std`.
//!
//! Every check funnels through [`Checker::check`], which mixes the boolean
//! result into a running checksum and records the id (with its call-site
//! line) of the first failure. `run` returns that checksum plus the failure
//! count; `main.rs` stores both in a `static` the linker cannot strip, so
//! the optimizer cannot delete the work and a bad build shows up as a
//! nonzero failure count rather than silently vanishing.

use alloc::string::String;
use alloc::vec::Vec;
use core::hash::{Hash, Hasher};

use cord_rs::{Cord, CordBuffer, MemoryAccounting};

/// FNV-1a: exercises `Hash for Cord`/`CordBuffer` against a real
/// `core::hash::Hasher` without pulling in `std`'s `DefaultHasher`.
struct Fnv(u64);

impl Fnv {
    const fn new() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }
}

impl Hasher for Fnv {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 ^= u64::from(b);
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
}

/// Accumulates a checksum over every check's outcome and remembers the
/// first failure, so a regression is visible in the returned `(sum,
/// first_failure, failures)` even though nothing here can print or unwind.
struct Checker {
    sum: u64,
    first_failure: u32,
    failures: u32,
    next_id: u32,
}

impl Checker {
    const fn new() -> Self {
        Self { sum: 0x9e37_79b9_7f4a_7c15, first_failure: 0, failures: 0, next_id: 1 }
    }

    fn mix(&mut self, v: u64) {
        let mut h = Fnv(self.sum);
        h.write(&v.to_le_bytes());
        self.sum = h.finish();
    }

    fn check(&mut self, ok: bool) {
        let id = self.next_id;
        self.next_id += 1;
        if !ok {
            self.failures += 1;
            if self.first_failure == 0 {
                self.first_failure = id;
            }
        }
        self.mix(u64::from(ok));
    }
}

// A few KiB, so appends below cross `MAX_FLAT_LENGTH` and force first a
// flat-to-tree transition and then real btree growth, not just the inline
// (<= 15/23-byte) representation.
const CHUNK: &[u8] = b"0123456789abcdef";
const KIB_MULTIPLIER: usize = 400; // 400 * 16 B = 6400 B, several btree leaves.

/// Exercises construction, append/prepend, slicing, iteration, comparison,
/// search, `CordBuffer`, and memory accounting. Returns `(checksum,
/// first_failure_id, failure_count)` -- all zero-failure on success.
#[must_use]
pub fn run() -> (u64, u32, u32) {
    let mut c = Checker::new();

    // ---- construction: from a slice, and from an owned Vec -------------
    let from_slice = Cord::from(b"hello, no_std world" as &[u8]);
    c.check(from_slice.len() == 19);
    c.check(from_slice == "hello, no_std world");

    let owned: Vec<u8> = (0u32..4096).map(|i| (i % 251) as u8).collect();
    let from_vec = Cord::from(owned.clone());
    c.check(from_vec.len() == 4096);
    c.check(from_vec.to_vec() == owned);

    // ---- append / prepend: inline -> flat -> btree ----------------------
    // Starts inline (a handful of bytes), then each append below grows past
    // the inline and single-flat capacities into a multi-node btree.
    let mut cord = Cord::from("start:");
    c.check(cord.len() == 6); // still inline at this point

    for _ in 0..KIB_MULTIPLIER {
        cord.append(CHUNK);
    }
    let after_append_len = 6 + KIB_MULTIPLIER * CHUNK.len();
    c.check(cord.len() == after_append_len);

    for _ in 0..KIB_MULTIPLIER {
        cord.prepend(CHUNK);
    }
    let total_len = after_append_len + KIB_MULTIPLIER * CHUNK.len();
    c.check(cord.len() == total_len);

    // An independent oracle built with plain `alloc`, never touching a
    // `Cord`, so every comparison below is meaningful.
    let mut oracle: Vec<u8> = Vec::new();
    for _ in 0..KIB_MULTIPLIER {
        oracle.extend_from_slice(CHUNK);
    }
    oracle.extend_from_slice(b"start:");
    for _ in 0..KIB_MULTIPLIER {
        oracle.extend_from_slice(CHUNK);
    }
    c.check(oracle.len() == total_len);
    c.check(cord.to_vec() == oracle);

    // ---- slice / get -----------------------------------------------------
    let mid = total_len / 2;
    let sliced = cord.slice(mid..mid + 64);
    c.check(sliced.len() == 64);
    c.check(sliced.to_vec() == oracle[mid..mid + 64]);

    c.check(cord.get(0) == Some(oracle[0]));
    c.check(cord.get(total_len).is_none());
    c.check(cord.get(mid..mid + 8).map(|s| s.to_vec()) == Some(oracle[mid..mid + 8].to_vec()));
    c.check(cord.get(total_len..total_len + 1).is_none());

    // ---- chunks() / bytes() sums ------------------------------------------
    let mut chunk_count = 0usize;
    let mut chunk_total = 0usize;
    for chunk in cord.chunks() {
        chunk_count += 1;
        chunk_total += chunk.len();
    }
    c.check(chunk_total == cord.len());
    c.check(chunk_count > 1); // a real multi-chunk tree, not one flat.
    c.mix(chunk_count as u64);

    let byte_sum: u64 = cord.bytes().map(u64::from).sum();
    let oracle_sum: u64 = oracle.iter().map(|&b| u64::from(b)).sum();
    c.check(byte_sum == oracle_sum);
    c.check(cord.bytes().len() == cord.len());
    c.mix(byte_sum);

    // ---- ==, cmp -----------------------------------------------------------
    let clone = cord.clone();
    c.check(clone == cord);
    c.check(clone.cmp(&cord) == core::cmp::Ordering::Equal);
    let (a, b, ab) = (Cord::from("a"), Cord::from("b"), Cord::from("ab"));
    c.check(a < b);
    c.check(a.cmp(&ab) == core::cmp::Ordering::Less);
    c.check(cord == oracle);

    // ---- find ---------------------------------------------------------------
    // A needle placed squarely inside the appended region (past the inline
    // prefix and the first few flats), so `find` has to cross chunk
    // boundaries to succeed.
    let needle_at = after_append_len - CHUNK.len();
    c.check(cord.find(CHUNK) == Some(0)); // prepend put a CHUNK-aligned copy at 0.
    c.check(&oracle[needle_at..needle_at + CHUNK.len()] == CHUNK);
    c.check(cord.find("not-present-anywhere").is_none());
    c.check(cord.contains("start:"));

    // ---- CordBuffer: put / append -------------------------------------------
    let mut buf = CordBuffer::with_capacity(1024);
    c.check(buf.is_empty());
    buf.put_slice(b"buffered-");
    buf.put_slice(&owned[..500]);
    c.check(buf.len() == 9 + 500);
    c.check(&buf.as_slice()[..9] == b"buffered-");

    let buf_len = buf.len();
    let mut buffered_cord = Cord::from("prefix:");
    buffered_cord.append(buf);
    c.check(buffered_cord.len() == 7 + buf_len);
    c.check(buffered_cord.starts_with("prefix:buffered-"));

    // ---- estimated_memory_usage --------------------------------------------
    let total = cord.estimated_memory_usage(MemoryAccounting::Total);
    let precise = cord.estimated_memory_usage(MemoryAccounting::TotalMorePrecise);
    let fair = cord.estimated_memory_usage(MemoryAccounting::FairShare);
    c.check(total >= cord.len());
    c.check(precise >= cord.len());
    c.check(precise <= total);
    c.check(fair > 0);
    c.mix(total as u64);

    // ---- Hash: same bytes via different chunk layouts hash equal ------------
    let mut contiguous = cord.clone();
    contiguous.make_contiguous();
    let mut h1 = Fnv::new();
    cord.hash(&mut h1);
    let mut h2 = Fnv::new();
    contiguous.hash(&mut h2);
    c.check(h1.finish() == h2.finish());
    c.mix(h1.finish());

    // ---- String round-trip (valid UTF-8) -------------------------------------
    let text = Cord::from("no_std round-trip \u{2713}");
    let s: Result<String, _> = String::try_from(text.clone());
    c.check(s.as_deref() == Ok("no_std round-trip \u{2713}"));

    #[cfg(feature = "bytes")]
    run_bytes(&mut c, &cord, &owned);

    #[cfg(feature = "serde")]
    run_serde(&mut c, &cord);

    (c.sum, c.first_failure, c.failures)
}

// ---------------------------------------------------------------------------
// `bytes` feature: `Buf`/`BufMut` and zero-copy `Bytes` conversions.
// ---------------------------------------------------------------------------

#[cfg(feature = "bytes")]
fn run_bytes(c: &mut Checker, cord: &Cord, owned: &[u8]) {
    use bytes::{Buf, Bytes};

    let mut b = cord.clone();
    let remaining_before = Buf::remaining(&b);
    c.check(remaining_before == cord.len());
    Buf::advance(&mut b, 6);
    c.check(Buf::remaining(&b) == cord.len() - 6);

    let bytes_val = Bytes::from(owned.to_vec());
    let from_bytes = Cord::from(bytes_val.clone());
    c.check(from_bytes.len() == owned.len());
    c.check(from_bytes == owned);
    let back: Bytes = Cord::from(owned.to_vec()).into();
    c.check(&back[..] == owned);
}

// ---------------------------------------------------------------------------
// `serde` feature: a minimal `no_std` byte-sequence serializer/deserializer,
// so the crate's `Serialize`/`Deserialize` impls are monomorphized *and*
// exercised without needing a `no_std`-hostile format crate.
// ---------------------------------------------------------------------------

#[cfg(feature = "serde")]
mod tiny_serde {
    use alloc::vec::Vec;
    use core::fmt;

    use serde::{Deserializer, Serializer, de, ser};

    #[derive(Debug)]
    pub struct SerdeErr;

    impl fmt::Display for SerdeErr {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("serde error")
        }
    }
    impl core::error::Error for SerdeErr {}
    impl ser::Error for SerdeErr {
        fn custom<T: fmt::Display>(_: T) -> Self {
            Self
        }
    }
    impl de::Error for SerdeErr {
        fn custom<T: fmt::Display>(_: T) -> Self {
            Self
        }
    }

    /// Only supports `serialize_bytes`, which is all `Cord` uses.
    pub struct BytesSerializer;

    macro_rules! unsupported {
        ($($m:ident($($a:ty),*)),* $(,)?) => {$(
            fn $m(self $(, _: $a)*) -> Result<Vec<u8>, SerdeErr> { Err(SerdeErr) }
        )*};
    }

    impl Serializer for BytesSerializer {
        type Ok = Vec<u8>;
        type Error = SerdeErr;
        type SerializeSeq = ser::Impossible<Vec<u8>, SerdeErr>;
        type SerializeTuple = ser::Impossible<Vec<u8>, SerdeErr>;
        type SerializeTupleStruct = ser::Impossible<Vec<u8>, SerdeErr>;
        type SerializeTupleVariant = ser::Impossible<Vec<u8>, SerdeErr>;
        type SerializeMap = ser::Impossible<Vec<u8>, SerdeErr>;
        type SerializeStruct = ser::Impossible<Vec<u8>, SerdeErr>;
        type SerializeStructVariant = ser::Impossible<Vec<u8>, SerdeErr>;

        fn serialize_bytes(self, v: &[u8]) -> Result<Vec<u8>, SerdeErr> {
            Ok(v.to_vec())
        }

        unsupported! {
            serialize_bool(bool), serialize_i8(i8), serialize_i16(i16),
            serialize_i32(i32), serialize_i64(i64), serialize_u8(u8),
            serialize_u16(u16), serialize_u32(u32), serialize_u64(u64),
            serialize_f32(f32), serialize_f64(f64), serialize_char(char),
            serialize_str(&str), serialize_none(), serialize_unit(),
            serialize_unit_struct(&'static str),
        }

        fn serialize_some<T: ?Sized + serde::Serialize>(self, _: &T) -> Result<Vec<u8>, SerdeErr> {
            Err(SerdeErr)
        }
        fn serialize_unit_variant(self, _: &str, _: u32, _: &str) -> Result<Vec<u8>, SerdeErr> {
            Err(SerdeErr)
        }
        fn serialize_newtype_struct<T: ?Sized + serde::Serialize>(
            self,
            _: &'static str,
            _: &T,
        ) -> Result<Vec<u8>, SerdeErr> {
            Err(SerdeErr)
        }
        fn serialize_newtype_variant<T: ?Sized + serde::Serialize>(
            self,
            _: &'static str,
            _: u32,
            _: &'static str,
            _: &T,
        ) -> Result<Vec<u8>, SerdeErr> {
            Err(SerdeErr)
        }
        fn serialize_seq(self, _: Option<usize>) -> Result<Self::SerializeSeq, SerdeErr> {
            Err(SerdeErr)
        }
        fn serialize_tuple(self, _: usize) -> Result<Self::SerializeTuple, SerdeErr> {
            Err(SerdeErr)
        }
        fn serialize_tuple_struct(
            self,
            _: &'static str,
            _: usize,
        ) -> Result<Self::SerializeTupleStruct, SerdeErr> {
            Err(SerdeErr)
        }
        fn serialize_tuple_variant(
            self,
            _: &'static str,
            _: u32,
            _: &'static str,
            _: usize,
        ) -> Result<Self::SerializeTupleVariant, SerdeErr> {
            Err(SerdeErr)
        }
        fn serialize_map(self, _: Option<usize>) -> Result<Self::SerializeMap, SerdeErr> {
            Err(SerdeErr)
        }
        fn serialize_struct(self, _: &'static str, _: usize) -> Result<Self::SerializeStruct, SerdeErr> {
            Err(SerdeErr)
        }
        fn serialize_struct_variant(
            self,
            _: &'static str,
            _: u32,
            _: &'static str,
            _: usize,
        ) -> Result<Self::SerializeStructVariant, SerdeErr> {
            Err(SerdeErr)
        }
    }

    /// Feeds a byte buffer straight into `visit_byte_buf`.
    pub struct BytesDeserializer(pub Vec<u8>);

    impl<'de> Deserializer<'de> for BytesDeserializer {
        type Error = SerdeErr;

        fn deserialize_any<V: de::Visitor<'de>>(self, v: V) -> Result<V::Value, SerdeErr> {
            v.visit_byte_buf(self.0)
        }

        serde::forward_to_deserialize_any! {
            bool i8 i16 i32 i64 i128 u8 u16 u32 u64 u128 f32 f64 char str string
            bytes byte_buf option unit unit_struct newtype_struct seq tuple
            tuple_struct map struct enum identifier ignored_any
        }
    }
}

#[cfg(feature = "serde")]
fn run_serde(c: &mut Checker, cord: &Cord) {
    use serde::{Deserialize as _, Serialize as _};

    let bytes = cord.serialize(tiny_serde::BytesSerializer).unwrap_or_default();
    c.check(bytes == cord.to_vec());

    let round = Cord::deserialize(tiny_serde::BytesDeserializer(bytes)).unwrap_or_default();
    c.check(round == *cord);
    c.check(round.len() == cord.len());
}
