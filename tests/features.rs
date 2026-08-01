//! Tests for the optional `bytes` and `serde` integrations.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    reason = "tests juggle small integers freely"
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
    use cord_rs::CordWriter;
    use cord_rs::internal;

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
        assert_eq!(cord.as_flat().unwrap().as_ptr(), large.as_ptr());
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
        let ptr = flat.as_flat().unwrap().as_ptr();
        let bytes = Bytes::from(flat);
        assert_eq!(bytes.as_ptr(), ptr);
        assert_eq!(&bytes[..], &expected[..1000]);
        // Cords compare with Bytes.
        assert!(Cord::from("xyz").equals(&Bytes::from_static(b"xyz")));
        assert_eq!(Cord::from("xyz").find(&Bytes::from_static(b"z")), Some(2));
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
        let mut cord = Cord::from("head");
        {
            let mut writer = CordWriter::new(&mut cord);
            assert!(writer.remaining_mut() > usize::MAX / 2);
            writer.put_u32(0xDEAD_BEEF);
            writer.put_slice(b" middle ");
            for i in 0..10_000u32 {
                writer.put_u8((i % 256) as u8);
            }
            writer.put_slice(&vec![b'z'; 50_000]);
            writer.put_u16_le(0x1234);
            writer.flush();
            writer.put_slice(b"after flush");
        }
        let mut expected = b"head".to_vec();
        expected.extend_from_slice(&0xDEAD_BEEFu32.to_be_bytes());
        expected.extend_from_slice(b" middle ");
        expected.extend((0..10_000u32).map(|i| (i % 256) as u8));
        expected.extend(std::iter::repeat_n(b'z', 50_000));
        expected.extend_from_slice(&0x1234u16.to_le_bytes());
        expected.extend_from_slice(b"after flush");
        assert_eq!(cord, expected);
        internal::validate(&cord).unwrap();

        // io::Write through the writer and into_inner.
        let mut cord = Cord::new();
        let mut writer = CordWriter::new(&mut cord);
        std::io::Write::write_all(&mut writer, b"io write").unwrap();
        let inner = writer.into_inner();
        assert_eq!(*inner, "io write");
        inner.append("!");
        assert_eq!(cord, "io write!");
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
}

#[cfg(feature = "serde")]
mod serde_feature {
    use super::*;

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
}
