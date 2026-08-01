//! Port of abseil's `cord_rep_btree_reader_test.cc`.

use super::btree::{BtreePtr, CordRepBtree, MAX_CAPACITY};
use super::reader::CordRepBtreeReader;
use super::test_util::*;
use super::unref;

const CHARS: usize = 3;

fn counts() -> Vec<usize> {
    let cap = MAX_CAPACITY;
    if cfg!(miri) {
        vec![1, 2, cap, cap * cap + 1]
    } else {
        vec![1, 2, cap, cap * cap, cap * cap + 1, cap * cap * 2 + 17]
    }
}

#[test]
fn next() {
    for count in counts() {
        unsafe {
            let data = create_random_string(count * CHARS);
            let flats = create_flats_from_string(&data, CHARS);
            let node = cord_rep_btree_from_flats(&flats);

            let mut reader = CordRepBtreeReader::new();
            let mut remaining = data.len();
            let mut chunk = reader.init(node);
            assert_eq!(chunk, &data[..chunk.len()]);

            remaining -= chunk.len();
            assert_eq!(reader.remaining(), remaining);

            while remaining > 0 {
                let offset = data.len() - remaining;
                chunk = reader.next();
                assert_eq!(chunk, &data[offset..offset + chunk.len()]);
                remaining -= chunk.len();
                assert_eq!(reader.remaining(), remaining);
            }
            assert_eq!(reader.remaining(), 0);

            // Reading beyond EOF returns empty.
            assert!(reader.next().is_empty());

            unref(node.as_rep());
        }
    }
}

#[test]
fn skip() {
    for count in counts() {
        unsafe {
            let data = create_random_string(count * CHARS);
            let flats = create_flats_from_string(&data, CHARS);
            let node = cord_rep_btree_from_flats(&flats);

            let step = if cfg!(miri) { 7 } else { 1 };
            for skip1 in (0..(data.len() - CHARS)).step_by(step) {
                for skip2 in (0..(data.len() - CHARS)).step_by(step) {
                    let mut reader = CordRepBtreeReader::new();
                    let mut remaining = data.len();
                    let mut chunk = reader.init(node);
                    remaining -= chunk.len();

                    chunk = reader.skip(skip1);
                    let offset = data.len() - remaining;
                    assert_eq!(chunk, &data[offset + skip1..offset + skip1 + chunk.len()]);
                    remaining -= chunk.len() + skip1;
                    assert_eq!(reader.remaining(), remaining);

                    if remaining == 0 {
                        continue;
                    }

                    let skip = (remaining - 1).min(skip2);
                    chunk = reader.skip(skip);
                    let offset = data.len() - remaining;
                    assert_eq!(chunk, &data[offset + skip..offset + skip + chunk.len()]);
                }
            }
            unref(node.as_rep());
        }
    }
}

#[test]
fn skip_beyond_length() {
    unsafe {
        let mut tree = CordRepBtree::create(make_flat(b"abc"));
        tree = CordRepBtree::append(tree, make_flat(b"def"));
        let mut reader = CordRepBtreeReader::new();
        reader.init(tree);
        assert!(reader.skip(100).is_empty());
        assert_eq!(reader.remaining(), 0);
        unref(tree.as_rep());
    }
}

#[test]
fn seek() {
    for count in counts() {
        unsafe {
            let data = create_random_string(count * CHARS);
            let flats = create_flats_from_string(&data, CHARS);
            let node = cord_rep_btree_from_flats(&flats);

            for seek in 0..(data.len() - 1) {
                let mut reader = CordRepBtreeReader::new();
                reader.init(node);
                let chunk = reader.seek(seek);
                assert!(!chunk.is_empty());
                assert_eq!(chunk, &data[seek..seek + chunk.len()]);
                assert_eq!(reader.remaining(), data.len() - seek - chunk.len());
            }
            unref(node.as_rep());
        }
    }
}

#[test]
fn seek_beyond_length() {
    unsafe {
        let mut tree = CordRepBtree::create(make_flat(b"abc"));
        tree = CordRepBtree::append(tree, make_flat(b"def"));
        let mut reader = CordRepBtreeReader::new();
        reader.init(tree);
        assert!(reader.seek(6).is_empty());
        assert_eq!(reader.remaining(), 0);
        assert!(reader.seek(100).is_empty());
        assert_eq!(reader.remaining(), 0);
        unref(tree.as_rep());
    }
}

#[test]
fn read() {
    unsafe {
        let data = b"abcdefghijklmno";
        let flats = create_flats_from_string(data, 5);
        let node = cord_rep_btree_from_flats(&flats);
        let mut reader = CordRepBtreeReader::new();

        // Read zero bytes.
        let chunk = reader.init(node);
        let (chunk, tree) = reader.read(0, chunk.len());
        assert!(tree.is_null());
        assert_eq!(chunk, b"abcde");
        assert_eq!(reader.remaining(), 10);
        assert_eq!(reader.next(), b"fghij");

        // Read in full.
        let chunk = reader.init(node);
        let (chunk, tree) = reader.read(15, chunk.len());
        assert!(!tree.is_null());
        assert_eq!(cord_to_string(tree), b"abcdefghijklmno");
        assert_eq!(chunk, b"");
        assert_eq!(reader.remaining(), 0);
        unref(tree);

        // Read < chunk bytes.
        let chunk = reader.init(node);
        let (chunk, tree) = reader.read(3, chunk.len());
        assert!(!tree.is_null());
        assert_eq!(cord_to_string(tree), b"abc");
        assert_eq!(chunk, b"de");
        assert_eq!(reader.remaining(), 10);
        assert_eq!(reader.next(), b"fghij");
        unref(tree);

        // Read < chunk bytes at offset.
        let chunk = reader.init(node);
        let (chunk, tree) = reader.read(2, chunk.len() - 2);
        assert!(!tree.is_null());
        assert_eq!(cord_to_string(tree), b"cd");
        assert_eq!(chunk, b"e");
        assert_eq!(reader.remaining(), 10);
        assert_eq!(reader.next(), b"fghij");
        unref(tree);

        // Read from consumed chunk.
        reader.init(node);
        let (chunk, tree) = reader.read(3, 0);
        assert!(!tree.is_null());
        assert_eq!(cord_to_string(tree), b"fgh");
        assert_eq!(chunk, b"ij");
        assert_eq!(reader.remaining(), 5);
        assert_eq!(reader.next(), b"klmno");
        unref(tree);

        // Read across chunks.
        let chunk = reader.init(node);
        let (chunk, tree) = reader.read(12, chunk.len() - 2);
        assert!(!tree.is_null());
        assert_eq!(cord_to_string(tree), b"cdefghijklmn");
        assert_eq!(chunk, b"o");
        assert_eq!(reader.remaining(), 0);
        unref(tree);

        // Read across chunks landing on an exact edge boundary.
        let chunk = reader.init(node);
        let (chunk, tree) = reader.read(10 - 2, chunk.len() - 2);
        assert!(!tree.is_null());
        assert_eq!(cord_to_string(tree), b"cdefghij");
        assert_eq!(chunk, b"klmno");
        assert_eq!(reader.remaining(), 0);
        unref(tree);

        unref(node.as_rep());
    }
}

#[test]
fn read_exhaustive() {
    let cap = MAX_CAPACITY;
    let counts: &[usize] = if cfg!(miri) {
        &[1, 2, cap, cap * cap + 1]
    } else {
        &[1, 2, cap, cap * cap + 1, cap * cap * cap * 2 + 17]
    };
    for &count in counts {
        unsafe {
            let data = create_random_string(count * CHARS);
            let flats = create_flats_from_string(&data, CHARS);
            let node = cord_rep_btree_from_flats(&flats);

            for read_size in [CHARS - 1, CHARS, CHARS + 7, cap * cap] {
                let mut reader = CordRepBtreeReader::new();
                let mut chunk = reader.init(node);

                // `consumed` tracks the end of the last consumed chunk, which
                // is the start of the next: we always read with
                // `chunk_size = chunk.len()`.
                let mut consumed = 0;
                let mut remaining = data.len();
                while remaining > 0 {
                    let n = remaining.min(read_size);
                    let (next_chunk, tree) = reader.read(n, chunk.len());
                    chunk = next_chunk;
                    assert!(!tree.is_null());
                    assert_eq!(cord_to_string(tree), &data[consumed..consumed + n]);
                    unref(tree);

                    consumed += n;
                    remaining -= n;
                    assert_eq!(reader.remaining(), remaining - chunk.len());

                    if remaining > 0 {
                        assert!(!chunk.is_empty());
                        assert_eq!(chunk, &data[consumed..consumed + chunk.len()]);
                    } else {
                        assert!(chunk.is_empty());
                    }
                }
            }
            unref(node.as_rep());
        }
    }
}
