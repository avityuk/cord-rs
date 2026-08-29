//! Tests for the optional `bytes` and `serde` integrations.
#![cfg_attr(
    any(feature = "bytes", feature = "serde"),
    expect(clippy::cast_possible_truncation, reason = "tests juggle small integers freely")
)]

#[cfg(any(feature = "bytes", feature = "serde"))]
use cord_rs::Cord;

#[cfg(any(feature = "bytes", feature = "serde"))]
fn fragmented(n: usize, chunk: usize) -> Cord {
    let bytes: Vec<u8> = (0..n).map(|i| (i * 7 % 251) as u8).collect();
    let mut cord = Cord::new();
    for piece in bytes.chunks(chunk) {
        cord.append(piece);
    }
    cord
}

#[cfg(feature = "bytes")]
mod bytes_feature {
    use super::*;
    use bytes::{Buf, BufMut, Bytes};
    use cord_rs::__internal as internal;
    use cord_rs::{CordBuffer, CordWriter};

    #[test]
    fn buf_for_cord() {
        let mut cord = fragmented(10_000, 333);
        let expected = cord.to_vec();
        assert_eq!(cord.remaining(), 10_000);
        // Chunk boundaries follow the buffers, not the appended pieces.
        let first = cord.chunks().next().unwrap().len();
        assert_eq!(cord.chunk(), &expected[..first]);
        cord.advance(100);
        assert_eq!(cord.chunk(), &expected[100..first]);
        assert_eq!(cord.get_u8(), expected[100]);
        assert_eq!(cord.get_u16(), u16::from_be_bytes([expected[101], expected[102]]));
        let bytes = cord.copy_to_bytes(5000);
        assert_eq!(&bytes[..], &expected[103..5103]);
        assert_eq!(cord.remaining(), 10_000 - 5103);
        let mut out = vec![0u8; cord.remaining()];
        cord.copy_to_slice(&mut out);
        assert_eq!(out, &expected[5103..]);
        assert!(!cord.has_remaining());
        assert!(cord.is_empty());
    }

    #[test]
    fn buf_for_cursor() {
        let cord = fragmented(5000, 100);
        let expected = cord.to_vec();
        let mut cursor = cord.cursor();
        assert_eq!(Buf::remaining(&cursor), 5000);
        let first = cord.chunks().next().unwrap().len();
        assert_eq!(Buf::chunk(&cursor), &expected[..first]);
        Buf::advance(&mut cursor, 250);
        assert_eq!(cursor.get_u32(), u32::from_be_bytes(expected[250..254].try_into().unwrap()));
        let bytes = cursor.copy_to_bytes(1000);
        assert_eq!(&bytes[..], &expected[254..1254]);
        assert_eq!(cursor.position(), 1254);
        // The cord is untouched.
        assert_eq!(cord.len(), 5000);
    }

    #[test]
    fn bytes_conversions() {
        // Small: copied (inline).
        let small = Bytes::from_static(b"small");
        let cord = Cord::from(small.clone());
        assert_eq!(cord, "small");
        assert!(!internal::is_tree(&cord));
        // Large: shared without copying, released with the cord.
        let large = Bytes::from(vec![7u8; 10_000]);
        let cord = Cord::from(large.clone());
        assert!(internal::is_external(&cord));
        assert_eq!(cord.as_contiguous().unwrap().as_ptr(), large.as_ptr());
        // Back to Bytes without copying when flat.
        let back = Bytes::from(cord);
        assert_eq!(back.as_ptr(), large.as_ptr());
        assert_eq!(back, large);
        // Fragmented cords are copied.
        let tree = fragmented(10_000, 100);
        let expected = tree.to_vec();
        let bytes = Bytes::from(tree);
        assert_eq!(&bytes[..], &expected[..]);
        // Inline cords are copied.
        let bytes = Bytes::from(Cord::from("abc"));
        assert_eq!(&bytes[..], b"abc");
        // A flat (non external) cord hands out its buffer.
        let flat = Cord::from(&expected[..1000]);
        let ptr = flat.as_contiguous().unwrap().as_ptr();
        let bytes = Bytes::from(flat);
        assert_eq!(bytes.as_ptr(), ptr);
        assert_eq!(&bytes[..], &expected[..1000]);
        // Cords compare with Bytes.
        assert_eq!(Cord::from("xyz"), Bytes::from_static(b"xyz"));
        assert_eq!(Cord::from("xyz").find(&Bytes::from_static(b"z")), Some(2));
    }

    #[test]
    fn bytes_comparison_symmetry() {
        let cord = Cord::from("bbb");
        for (raw, expected) in [
            (*b"aaa", std::cmp::Ordering::Less),
            (*b"bbb", std::cmp::Ordering::Equal),
            (*b"ccc", std::cmp::Ordering::Greater),
        ] {
            let bytes = Bytes::copy_from_slice(&raw);
            let slice_cmp = raw[..].partial_cmp(&cord.to_vec()[..]);
            assert_eq!(slice_cmp, Some(expected));

            assert_eq!(bytes == cord, expected == std::cmp::Ordering::Equal);
            assert_eq!(cord == bytes, bytes == cord);
            assert_eq!(bytes.partial_cmp(&cord), slice_cmp);
            assert_eq!(cord.partial_cmp(&bytes), slice_cmp.map(std::cmp::Ordering::reverse));
        }
    }

    #[test]
    fn append_bytes_source() {
        let mut cord = Cord::from("head-");
        cord.append(Bytes::from_static(b"static"));
        let big = Bytes::from(vec![b'b'; 4000]);
        cord.append(big.clone());
        cord.prepend(big.clone());
        let mut expected = vec![b'b'; 4000];
        expected.extend_from_slice(b"head-static");
        expected.extend_from_slice(&[b'b'; 4000]);
        assert_eq!(cord, expected);
        internal::validate(&cord).unwrap();
    }

    #[test]
    fn cord_writer_buf_mut() {
        // Miri interprets every byte; 10_000 + 50_000 bytes is minutes, not
        // a check.
        let iters: u32 = if cfg!(miri) { 200 } else { 10_000 };
        let slice_len: usize = if cfg!(miri) { 500 } else { 50_000 };
        let mut cord = Cord::from("head");
        {
            let mut writer = CordWriter::new(&mut cord);
            assert!(writer.remaining_mut() > usize::MAX / 2);
            writer.put_u32(0xDEAD_BEEF);
            writer.put_slice(b" middle ");
            for i in 0..iters {
                writer.put_u8((i % 256) as u8);
            }
            writer.put_slice(&vec![b'z'; slice_len]);
            writer.put_u16_le(0x1234);
            writer.flush();
            writer.put_slice(b"after flush");
        }
        let mut expected = b"head".to_vec();
        expected.extend_from_slice(&0xDEAD_BEEFu32.to_be_bytes());
        expected.extend_from_slice(b" middle ");
        expected.extend((0..iters).map(|i| (i % 256) as u8));
        expected.extend(std::iter::repeat_n(b'z', slice_len));
        expected.extend_from_slice(&0x1234u16.to_le_bytes());
        expected.extend_from_slice(b"after flush");
        assert_eq!(cord, expected);
        internal::validate(&cord).unwrap();

        // io::Write through the writer and into_inner.
        let mut cord = Cord::new();
        let mut writer = CordWriter::new(&mut cord);
        #[cfg(feature = "std")]
        std::io::Write::write_all(&mut writer, b"io write").unwrap();
        #[cfg(not(feature = "std"))]
        writer.put_slice(b"io write");
        let inner = writer.into_inner();
        assert_eq!(*inner, "io write");
        inner.append("!");
        assert_eq!(cord, "io write!");
    }

    /// Direct coverage of the raw `chunk_mut` + `advance_mut` protocol
    /// (`BufMut`'s documented low-level contract), not routed through the
    /// safe `put_slice` override: `chunk_mut`/`advance_mut` are the crate's
    /// only unsafe impl of a foreign trait and had no direct test coverage.
    #[test]
    fn cord_writer_raw_chunk_mut_advance_mut_round_trip() {
        let mut cord = Cord::from("head-");
        {
            let mut writer = CordWriter::new(&mut cord);
            let chunk = writer.chunk_mut();
            assert!(chunk.len() >= 5);
            for (i, b) in b"HELLO".iter().enumerate() {
                chunk.write_byte(i, *b);
            }
            // SAFETY: the 5 bytes just written above via `write_byte` are
            // now initialized.
            unsafe { writer.advance_mut(5) };

            // A second round through the same buffer's remaining capacity.
            let chunk = writer.chunk_mut();
            chunk.write_byte(0, b'!');
            // SAFETY: see above.
            unsafe { writer.advance_mut(1) };
        }
        assert_eq!(cord, "head-HELLO!");
        internal::validate(&cord).unwrap();
    }

    /// `advance_mut`'s documented contract: the caller must not claim more
    /// bytes than the chunk `chunk_mut` last returned.
    #[test]
    #[should_panic(expected = "exceed the chunk capacity")]
    fn cord_writer_advance_mut_past_chunk_panics() {
        let mut cord = Cord::new();
        let mut writer = CordWriter::new(&mut cord);
        let chunk_len = writer.chunk_mut().len();
        // SAFETY: none of these "initialized" bytes are ever read — the
        // assertion below panics before that could happen.
        unsafe { writer.advance_mut(chunk_len + 1) };
    }

    /// `chunk_mut`'s auto-flush-on-full path: filling a buffer exactly (via
    /// `put_slice`, which does not itself flush) must not prevent the
    /// *next* `chunk_mut` call from handing out a fresh buffer — it must
    /// flush the full one first rather than returning an empty chunk.
    ///
    /// (`CordWriter` holds `&mut Cord` and has a `Drop` impl that flushes,
    /// so `cord` can't be peeked at directly while `writer` is still alive;
    /// this instead checks the flush indirectly, through the size of the
    /// chunk `chunk_mut` hands back.)
    #[test]
    fn cord_writer_chunk_mut_flushes_full_buffer() {
        let mut cord = Cord::new();
        let cap;
        {
            let mut writer = CordWriter::new(&mut cord);
            cap = writer.chunk_mut().len();
            writer.put_slice(&vec![b'x'; cap]);
            // The buffer is now exactly full. If the next `chunk_mut` call
            // failed to flush it first, `spare_capacity_mut` on the still-full
            // buffer would return an empty slice; a fresh buffer instead
            // gives back the same default capacity.
            let next_chunk_len = writer.chunk_mut().len();
            assert!(next_chunk_len > 0, "chunk_mut must flush the full buffer before reallocating");
            assert_eq!(next_chunk_len, cap, "the fresh buffer should have the same default capacity");
        }
        assert_eq!(cord.len(), cap, "the full buffer's bytes must have reached the cord by the end");
        internal::validate(&cord).unwrap();
    }

    #[test]
    fn cord_writer_reuses_spare_capacity() {
        let mut cord = Cord::new();
        cord.append(&[b'a'; 4100][..]);
        cord.append("b");
        let usage_before = cord.estimated_memory_usage(cord_rs::MemoryAccounting::Total);
        {
            let mut writer = CordWriter::new(&mut cord);
            writer.put_slice(b"cc");
        }
        assert_eq!(cord.len(), 4103);
        assert_eq!(cord.estimated_memory_usage(cord_rs::MemoryAccounting::Total), usage_before);
    }

    #[test]
    fn buf_mut_for_cord_buffer() {
        let mut buffer = CordBuffer::with_capacity(64);
        assert_eq!(buffer.remaining_mut(), buffer.available());

        // `put_u32`/`put_slice` round-trip against `as_slice()`.
        buffer.put_u32(0xDEAD_BEEF);
        BufMut::put_slice(&mut buffer, b" tail");
        let mut expected = 0xDEAD_BEEFu32.to_be_bytes().to_vec();
        expected.extend_from_slice(b" tail");
        assert_eq!(buffer.as_slice(), expected.as_slice());
        assert_eq!(buffer.remaining_mut(), buffer.available());

        // `chunk_mut`'s length always matches the remaining spare capacity.
        assert_eq!(buffer.chunk_mut().len(), buffer.available());

        // A manual `chunk_mut` write followed by `advance_mut`.
        let available_before = buffer.available();
        let chunk = buffer.chunk_mut();
        chunk.write_byte(0, b'!');
        // SAFETY: the byte just written above via `write_byte` is now
        // initialized.
        unsafe { buffer.advance_mut(1) };
        expected.push(b'!');
        assert_eq!(buffer.as_slice(), expected.as_slice());
        assert_eq!(buffer.available(), available_before - 1);

        // The written bytes flow into a `Cord` intact.
        let cord = Cord::from(buffer);
        assert_eq!(cord, expected);
        internal::validate(&cord).unwrap();
    }

    #[test]
    #[should_panic(expected = "exceed the available capacity")]
    fn buf_mut_put_slice_for_cord_buffer_past_capacity_panics() {
        let mut buffer = CordBuffer::new();
        let overflow = vec![0u8; buffer.capacity() + 1];
        BufMut::put_slice(&mut buffer, &overflow);
    }
}

#[cfg(feature = "serde")]
mod serde_feature {
    use super::*;
    use cord_rs::__internal as internal;
    use cord_rs::MemoryAccounting;

    #[test]
    fn json_roundtrip() {
        for cord in [Cord::new(), Cord::from("inline"), fragmented(5000, 123), Cord::from(vec![0xFFu8; 700])]
        {
            let json = serde_json::to_string(&cord).unwrap();
            let expected_json = serde_json::to_string(&cord.to_vec()).unwrap();
            assert_eq!(json, expected_json);
            let back: Cord = serde_json::from_str(&json).unwrap();
            assert_eq!(back, cord);
        }
        // Strings deserialize too.
        let from_str: Cord = serde_json::from_str("\"text\"").unwrap();
        assert_eq!(from_str, "text");
    }

    #[test]
    fn struct_field() {
        #[derive(serde::Serialize, serde::Deserialize, PartialEq, Debug)]
        struct Message {
            id: u32,
            payload: Cord,
        }
        let msg = Message { id: 7, payload: fragmented(1000, 10) };
        let json = serde_json::to_string(&msg).unwrap();
        let back: Message = serde_json::from_str(&json).unwrap();
        assert_eq!(back, msg);
    }

    /// An iterator that reports whatever `size_hint` it is constructed with,
    /// regardless of how many elements are actually left. Drives `visit_seq`
    /// through `serde::de::value::SeqDeserializer`, whose `size_hint` reports
    /// the wrapped iterator's bound verbatim whenever its lower and upper
    /// bounds agree — exactly the shape of a self-describing format's
    /// attacker-controlled or honest-but-absent sequence-length prefix.
    struct Hinted<I> {
        inner: I,
        hint: Option<usize>,
    }

    impl<I: Iterator> Iterator for Hinted<I> {
        type Item = I::Item;

        fn next(&mut self) -> Option<Self::Item> {
            self.inner.next()
        }

        fn size_hint(&self) -> (usize, Option<usize>) {
            match self.hint {
                Some(n) => (n, Some(n)),
                None => (0, None),
            }
        }
    }

    /// Deserializes `bytes` as a sequence whose `size_hint` is `hint`.
    fn from_seq(bytes: &[u8], hint: Option<usize>) -> Cord {
        let iter = Hinted { inner: bytes.iter().copied(), hint };
        let de = serde::de::value::SeqDeserializer::<_, serde::de::value::Error>::new(iter);
        let cord: Cord = serde::Deserialize::deserialize(de).unwrap();
        assert!(internal::validate(&cord).is_ok(), "{}", internal::dump(&cord, false));
        cord
    }

    fn payload(n: usize) -> Vec<u8> {
        (0..n).map(|i| (i % 251) as u8).collect()
    }

    // Miri interprets every byte pushed through `visit_seq`'s one-element-at-
    // a-time `SeqAccess`; the full 100_000-byte case takes minutes under it,
    // so it is scaled down. 9_000 still spans three chunks and a btree, so no
    // coverage is lost.
    const BIG: usize = if cfg!(miri) { 9_000 } else { 100_000 };

    const SIZES: [usize; 9] = [0, 1, 15, 16, 511, 512, 4083, 4084, BIG];

    // Scaled like `BIG`: enough full-size chunks ahead of a short tail to
    // prove the tail's buffer is copied out to its own size class rather
    // than adopted at the full `DEFAULT_MAX_CAPACITY` it grew into.
    const TAIL_CHUNKS: usize = if cfg!(miri) { 2 } else { 24 };

    /// The representation `visit_seq` is expected to produce for `len` bytes:
    /// inline up to 15, one flat up to a full chunk, a btree beyond that —
    /// the same shape `Cord::from(&[u8])` produces.
    fn assert_native_shape(cord: &Cord, len: usize) {
        if len <= internal::MAX_INLINE {
            assert!(!internal::is_tree(cord), "{len} bytes should stay inline");
        } else if len <= internal::MAX_FLAT_LENGTH {
            assert!(internal::is_flat(cord), "{len} bytes should be one flat");
        } else {
            assert!(internal::is_btree(cord), "{len} bytes should be a btree");
        }
    }

    #[test]
    fn visit_seq_exact_hint() {
        for len in SIZES {
            let data = payload(len);
            let cord = from_seq(&data, Some(len));
            assert_eq!(cord, data, "len {len}");
            assert_native_shape(&cord, len);
            // An honest hint is the ideal case: the result costs no more than
            // copying the same bytes in one go.
            assert!(
                cord.estimated_memory_usage(MemoryAccounting::Total)
                    <= Cord::from(&data[..]).estimated_memory_usage(MemoryAccounting::Total),
                "len {len}"
            );
        }
    }

    #[test]
    fn visit_seq_absent_hint() {
        for len in SIZES {
            let data = payload(len);
            let cord = from_seq(&data, None);
            assert_eq!(cord, data, "len {len}");
            assert_native_shape(&cord, len);
            assert_eq!(
                cord.estimated_memory_usage(MemoryAccounting::Total),
                Cord::from(&data[..]).estimated_memory_usage(MemoryAccounting::Total),
                "len {len}: an absent hint should still land on the native shape"
            );
        }
        // Lengths that land partway through (or just past) the geometric head
        // ramp — neither finishing inside it nor filling a grown buffer
        // exactly — depend on `commit`'s size-class rule, not the ramp
        // itself, to still reach the ideal shape.
        for len in [2200usize, 3000, 4000] {
            let data = payload(len);
            let cord = from_seq(&data, None);
            assert_eq!(cord, data, "len {len}");
            assert_eq!(
                cord.estimated_memory_usage(MemoryAccounting::Total),
                Cord::from(&data[..]).estimated_memory_usage(MemoryAccounting::Total),
                "len {len}"
            );
        }
        // A large payload's last, partially filled chunk must not pin the
        // full-size buffer it grew into: once the head is committed, every
        // further buffer is requested at `DEFAULT_MAX_CAPACITY` (the hint is
        // absent), so a short tail relies on `commit`'s size-class rule to
        // land on a right-sized flat instead of adopting a 4 KiB one.
        let data = payload(TAIL_CHUNKS * internal::MAX_FLAT_LENGTH + 3000);
        let cord = from_seq(&data, None);
        assert_eq!(cord, data);
        assert_eq!(
            cord.estimated_memory_usage(MemoryAccounting::Total),
            Cord::from(&data[..]).estimated_memory_usage(MemoryAccounting::Total),
            "a large payload's short tail chunk must match Cord::from's shape"
        );
    }

    #[test]
    fn visit_seq_under_reporting_hint() {
        // Hints that under-report (a truncated length prefix, an iterator
        // adapter that only knows a lower bound) are trusted for the first
        // buffer and then abandoned.
        for len in SIZES {
            let data = payload(len);
            for hint in [len / 4, len.saturating_sub(1), 1] {
                let cord = from_seq(&data, Some(hint));
                assert_eq!(cord, data, "len {len} hint {hint}");
                assert!(internal::validate(&cord).is_ok());
            }
        }
        // Nothing is left over-allocated: a badly under-reported 4 KiB value
        // is still a single chunk.
        let data = payload(4083);
        let cord = from_seq(&data, Some(1));
        assert!(internal::is_flat(&cord));
    }

    /// A hint just above `MAX_INLINE` (so `with_capacity` allocates a flat,
    /// not an inline buffer) that still lands in `flat`'s minimum size class
    /// (`MIN_FLAT_LENGTH`) must not cause an at-most-`MAX_INLINE`-byte result
    /// to be adopted as that flat: `flat::capacity_for` floors at
    /// `MIN_FLAT_LENGTH` for every length up to `MAX_INLINE` too, so the
    /// size-class rule alone would see the flat as "already right-sized" and
    /// adopt it, even though the value belongs inline. Every hint in
    /// `(MAX_INLINE, MIN_FLAT_LENGTH]`, paired with every actual length up to
    /// `MAX_INLINE`, must still land inline.
    #[test]
    fn visit_seq_hint_in_min_flat_window_still_lands_inline() {
        for hint in (internal::MAX_INLINE + 1)..=internal::MIN_FLAT_LENGTH {
            for len in 0..=internal::MAX_INLINE {
                let data = payload(len);
                let cord = from_seq(&data, Some(hint));
                assert_eq!(cord, data, "hint {hint} len {len}");
                assert!(!internal::is_tree(&cord), "hint {hint} len {len} should stay inline");
                assert_eq!(
                    cord.estimated_memory_usage(MemoryAccounting::Total),
                    Cord::from(&data[..]).estimated_memory_usage(MemoryAccounting::Total),
                    "hint {hint} len {len}"
                );
            }
        }
    }

    #[test]
    fn visit_seq_lying_huge_hint() {
        // A self-describing format's length prefix is attacker-controlled: a
        // nine-byte CBOR array header can claim `u64::MAX` elements, and
        // `rmp-serde` reports an `Array32` count of 0xFFFF_FFFF verbatim.
        // `CordBuffer::with_capacity` caps every request at
        // `DEFAULT_MAX_CAPACITY`, so the lie costs at most one 4 KiB buffer,
        // and nothing of it survives into the result — the geometric ramp
        // never even starts, since a lying hint sizes the *first* buffer
        // directly (`with_capacity(hint)`, itself capped).
        let cord = from_seq(&[1, 2, 3], Some(usize::MAX / 2));
        assert_eq!(cord, [1u8, 2, 3][..]);
        assert!(!internal::is_tree(&cord), "3 bytes must still be inline");
        assert_eq!(
            cord.estimated_memory_usage(MemoryAccounting::Total),
            core::mem::size_of::<Cord>(),
            "a lying hint must not leave an oversized buffer behind"
        );
        // Same for a value large enough to need a tree.
        let data = payload(5000);
        let cord = from_seq(&data, Some(usize::MAX));
        assert_eq!(cord, data);
        assert_eq!(
            cord.estimated_memory_usage(MemoryAccounting::Total),
            Cord::from(&data[..]).estimated_memory_usage(MemoryAccounting::Total)
        );
    }

    #[test]
    fn visit_seq_over_reporting_hint() {
        for len in [0usize, 16, 600] {
            let data = payload(len);
            let cord = from_seq(&data, Some(4 * len + 1000));
            assert_eq!(cord, data, "len {len}");
            assert_native_shape(&cord, len);
            assert_eq!(
                cord.estimated_memory_usage(MemoryAccounting::Total),
                Cord::from(&data[..]).estimated_memory_usage(MemoryAccounting::Total),
                "len {len}: an over-sized buffer must not be adopted as-is"
            );
        }
    }

    #[test]
    fn visit_seq_chunks_are_full_size() {
        // Large payloads come out as a chain of full-size chunks, not a ramp
        // of growing ones: the geometric head ramp only ever runs before
        // anything is committed, so it cannot leave a short chunk behind
        // once the cord is no longer empty.
        let data = payload(BIG);
        for hint in [Some(data.len()), None, Some(usize::MAX / 2)] {
            let cord = from_seq(&data, hint);
            assert_eq!(cord, data);
            let chunks: Vec<usize> = cord.chunks().map(<[u8]>::len).collect();
            assert_eq!(chunks.len(), data.len().div_ceil(internal::MAX_FLAT_LENGTH), "hint {hint:?}");
            for (i, n) in chunks.iter().enumerate() {
                if i + 1 < chunks.len() {
                    assert_eq!(*n, internal::MAX_FLAT_LENGTH, "hint {hint:?} chunk {i}");
                }
            }
        }
    }

    #[test]
    fn visit_seq_json_array() {
        // serde_json's `SeqAccess` has no `size_hint` at all, so a JSON array
        // exercises the absent-hint path end to end.
        for len in [0usize, 15, 16, 4083, 5000] {
            let data = payload(len);
            let json = serde_json::to_string(&data).unwrap();
            let cord: Cord = serde_json::from_str(&json).unwrap();
            assert_eq!(cord, data, "len {len}");
            assert_native_shape(&cord, len);
        }
    }
}
