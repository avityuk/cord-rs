//! `serde` support: a [`Cord`] serializes as a byte sequence.

use core::fmt;

use alloc::string::String;
use alloc::vec::Vec;

use serde::de::{Deserialize, Deserializer, Error, SeqAccess, Visitor};
use serde::ser::{Serialize, Serializer};

use crate::cord::Cord;

#[cfg_attr(docsrs, doc(cfg(feature = "serde")))]
impl Serialize for Cord {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self.as_contiguous() {
            Some(flat) => serializer.serialize_bytes(flat),
            None => serializer.serialize_bytes(&self.to_vec()),
        }
    }
}

/// Upper bound on how much a `size_hint` is trusted to pre-allocate. In
/// self-describing formats the hint comes straight from untrusted input, so
/// (mirroring serde's own `Vec<u8>` deserializer) it is capped rather than
/// passed to `Vec::with_capacity` as-is — otherwise a few bytes of encoded
/// input claiming a huge sequence length would abort the process on
/// allocation.
const MAX_PREALLOCATED: usize = 1 << 20;

struct CordVisitor;

impl<'de> Visitor<'de> for CordVisitor {
    type Value = Cord;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("a byte sequence")
    }

    fn visit_bytes<E: Error>(self, v: &[u8]) -> Result<Cord, E> {
        Ok(Cord::from(v))
    }

    fn visit_byte_buf<E: Error>(self, v: Vec<u8>) -> Result<Cord, E> {
        Ok(Cord::from(v))
    }

    fn visit_str<E: Error>(self, v: &str) -> Result<Cord, E> {
        Ok(Cord::from(v))
    }

    fn visit_string<E: Error>(self, v: String) -> Result<Cord, E> {
        Ok(Cord::from(v))
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Cord, A::Error> {
        let mut bytes = Vec::with_capacity(seq.size_hint().unwrap_or(0).min(MAX_PREALLOCATED));
        while let Some(byte) = seq.next_element::<u8>()? {
            bytes.push(byte);
        }
        Ok(Cord::from(bytes))
    }
}

#[cfg_attr(docsrs, doc(cfg(feature = "serde")))]
impl<'de> Deserialize<'de> for Cord {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Cord, D::Error> {
        deserializer.deserialize_byte_buf(CordVisitor)
    }
}
