//! `serde` support: a [`Cord`] serializes as a byte sequence.

use core::fmt;

use alloc::string::String;
use alloc::vec::Vec;

use serde::de::{Deserialize, Deserializer, Error, SeqAccess, Visitor};
use serde::ser::{Serialize, Serializer};

use crate::buffer::CordBuffer;
use crate::cord::Cord;
use crate::rep::{MAX_INLINE, flat};

#[cfg_attr(docsrs, doc(cfg(feature = "serde")))]
impl Serialize for Cord {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self.as_contiguous() {
            Some(flat) => serializer.serialize_bytes(flat),
            None => serializer.serialize_bytes(&self.to_vec()),
        }
    }
}

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

    /// Builds the cord straight into [`CordBuffer`]s: every byte is written
    /// once, into memory the cord adopts, and no intermediate `Vec` is
    /// staged.
    ///
    /// While nothing has been committed to the cord yet, an undersized head
    /// buffer is grown by copying rather than frozen into the tree as a
    /// short chunk — but the growth is geometric (quadrupling the buffer's
    /// capacity each time it fills, capped at
    /// [`CordBuffer::DEFAULT_MAX_CAPACITY`]), not a single jump straight to
    /// full size. That keeps a small value's transient buffer small too: a
    /// 60-byte value never allocates a 4 KiB buffer just because an absent
    /// hint started it out inline. The cost is at most a handful of extra
    /// allocate-and-copy steps on the way to full size, all of which are
    /// freed again once the final, right-sized chunk is committed — every
    /// commit, not just the head's, copies a buffer out instead of adopting
    /// it whenever the buffer's size class exceeds what its actual contents
    /// need (see [`commit`]), so a large payload's last, partially filled
    /// chunk never pins a full-size buffer either.
    ///
    /// A sequence's `size_hint` is only a hint — self-describing formats read
    /// it from an unvalidated length prefix (a nine-byte CBOR array header can
    /// claim `u64::MAX` elements), while streaming ones such as JSON have
    /// nothing to report at all. It is therefore used only to size buffers,
    /// never to decide how much to trust: [`CordBuffer::with_capacity`] caps
    /// every request at [`CordBuffer::DEFAULT_MAX_CAPACITY`], so the worst a
    /// lie can cost is one 4 KiB buffer that is freed again on the way out.
    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Cord, A::Error> {
        // Read once. A hint re-read per buffer would let a sequence that
        // keeps under-reporting hold every buffer down to the minimum size,
        // turning a large payload into thousands of tiny chunks.
        let hint = seq.size_hint();
        let mut cord = Cord::new();
        // `with_capacity` returns the allocation-free inline buffer for 15
        // bytes or less, so a small (or absent) hint costs nothing.
        let mut buffer = CordBuffer::with_capacity(hint.unwrap_or(0));
        let mut consumed = 0usize;
        loop {
            let before = buffer.len();
            let next = fill(&mut seq, &mut buffer)?;
            consumed += buffer.len() - before;
            let Some(byte) = next else { break };
            if cord.is_empty() && buffer.capacity() < CordBuffer::DEFAULT_MAX_CAPACITY {
                // Nothing has been committed to the tree yet, so an
                // under-sized first buffer (an absent hint, or one that
                // under-reports) can still be replaced rather than frozen
                // into the tree as a short chunk. Quadruple the capacity
                // instead of jumping straight to full size, so a value that
                // turns out to be small never pins a large buffer even
                // transiently.
                let grown_capacity = (buffer.capacity() * 4).min(CordBuffer::DEFAULT_MAX_CAPACITY);
                let mut grown = CordBuffer::with_capacity(grown_capacity);
                grown.put_slice(&buffer);
                buffer = grown;
            } else {
                // The hint is trusted only for as long as it lasts: while it
                // still predicts more bytes the next buffer is sized to fit
                // exactly what is left, and once it is used up (or was never
                // there) every further buffer is a full-size chunk. This
                // branch is also where a head buffer that has finished
                // growing (capacity == DEFAULT_MAX_CAPACITY) starts being
                // committed instead of grown further.
                let want = match hint.map(|total| total.saturating_sub(consumed)) {
                    Some(remaining) if remaining != 0 => remaining,
                    _ => CordBuffer::DEFAULT_MAX_CAPACITY,
                };
                commit(&mut cord, buffer);
                buffer = CordBuffer::with_capacity(want);
            }
            buffer.put_slice(&[byte]);
            consumed += 1;
        }
        commit(&mut cord, buffer);
        Ok(cord)
    }
}

/// Fills `buffer`'s spare capacity from `seq`. Returns the first byte that did
/// not fit (the buffer is full), or `None` if the sequence ended first.
fn fill<'de, A: SeqAccess<'de>>(seq: &mut A, buffer: &mut CordBuffer) -> Result<Option<u8>, A::Error> {
    let start = buffer.len();
    let mut written = 0;
    let mut ended = false;
    let mut error = None;
    for slot in buffer.spare_capacity_mut() {
        match seq.next_element::<u8>() {
            Ok(Some(byte)) => {
                slot.write(byte);
                written += 1;
            }
            Ok(None) => {
                ended = true;
                break;
            }
            Err(e) => {
                error = Some(e);
                break;
            }
        }
    }
    // SAFETY: the loop above initialized exactly the first `written` bytes of
    // the spare capacity, and nothing beyond them.
    unsafe { buffer.set_len(start + written) };
    match error {
        Some(e) => Err(e),
        None if ended => Ok(None),
        // The buffer is full; one more element has to be read to find out
        // whether the sequence continues. Reading it here rather than after
        // the next buffer exists is what keeps a sequence whose length is an
        // exact multiple of the buffer size from allocating a buffer it never
        // uses.
        None => seq.next_element::<u8>(),
    }
}

/// Adds `buffer` to `cord`, copying rather than adopting whenever the
/// buffer's own allocation is a larger size class than its contents need —
/// e.g. a head buffer grown for a bigger payload than the sequence turned
/// out to have, or a 4083-byte tail chunk holding only a few hundred real
/// bytes. Adopting such a buffer as-is would permanently pin the size class
/// it happened to grow to rather than the one its data needs, so every
/// commit is checked against [`flat::capacity_for`], the size class a fresh
/// [`Cord::from`]`(&[u8])` flat would land on for the same length — the
/// buffer is copied out whenever that is smaller than what it already has,
/// and adopted unchanged when it is already exactly that size (the common
/// case: an honest hint, or a buffer that filled precisely). The copy this
/// can trigger is bounded by one buffer (at most `DEFAULT_MAX_CAPACITY`
/// bytes) per commit.
fn commit(cord: &mut Cord, buffer: CordBuffer) {
    if buffer.is_empty() {
        return;
    }
    // A result short enough for the inline handle is always copied in: a
    // `CordBuffer`'s own inline capacity is pointer-width dependent (seven
    // bytes on 32-bit targets), so a hint of up to `MAX_INLINE` bytes may
    // still have produced a heap buffer, and adopting it would make a flat
    // out of a value `Cord::from(&[u8])` keeps inline.
    if buffer.len() <= MAX_INLINE || flat::capacity_for(buffer.len()) < buffer.capacity() {
        // SAFETY: the buffer is non-empty (checked above) and its length is at
        // most `DEFAULT_MAX_CAPACITY`, which is `MAX_FLAT_LENGTH`.
        unsafe { cord.append_precise(&buffer) };
    } else {
        cord.append(buffer);
    }
}

#[cfg_attr(docsrs, doc(cfg(feature = "serde")))]
impl<'de> Deserialize<'de> for Cord {
    /// Deserializes a byte sequence, however the format encodes it. Whatever
    /// length hint a `visit_seq` array reports — absent, honest, or wrong —
    /// the result has the same chunking as [`Cord::from`]`(&[u8])` of the
    /// same bytes; the hint only ever changes how the bytes are staged on
    /// the way in, never the shape of the cord that comes out.
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Cord, D::Error> {
        deserializer.deserialize_byte_buf(CordVisitor)
    }
}
