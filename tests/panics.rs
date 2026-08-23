//! `should_panic` coverage for out-of-range / capacity-exceeded misuse that
//! isn't already covered by the panic tests in `tests/basic.rs`.

use cord_rs::{__internal as internal, Cord, CordBuffer};

#[test]
#[should_panic(expected = "split_off index out of bounds")]
fn split_off_out_of_range_panics() {
    let mut c = Cord::from("abc");
    let _ = c.split_off(4);
}

#[test]
#[should_panic(expected = "split_to index out of bounds")]
fn split_to_out_of_range_panics() {
    let mut c = Cord::from("abc");
    let _ = c.split_to(4);
}

#[test]
#[should_panic(expected = "slice index starts at 5 but ends at 3")]
#[expect(clippy::reversed_empty_ranges, reason = "exercising the panic on a deliberately reversed range")]
fn slice_reversed_range_panics() {
    let c = Cord::from("hello world");
    let _ = c.slice(5..3);
}

#[test]
#[should_panic(expected = "cannot advance past the end of a Cord")]
fn cursor_advance_past_end_panics() {
    let c = Cord::from("abc");
    let mut cursor = c.cursor();
    cursor.advance(4);
}

#[test]
#[should_panic(expected = "exceeds capacity")]
fn cord_buffer_set_len_over_capacity_panics() {
    let mut buffer = CordBuffer::new();
    let cap = buffer.capacity();
    // SAFETY: the assertion panics before any bytes would be treated as
    // initialized; nothing here is ever read.
    unsafe { buffer.set_len(cap + 1) };
}

#[test]
#[should_panic(expected = "exceed the available capacity")]
fn put_slice_overflow_panics_on_heap_buffer() {
    // `with_capacity_and_block_size` always allocates through
    // `flat::new_large` (unlike `with_capacity`, which stays inline for
    // small requests), so this exercises `put_slice`'s heap (`Flat`) branch
    // specifically, not the inline `Short` branch the crate's own internal
    // unit test already covers.
    let mut buffer = CordBuffer::with_capacity_and_block_size(64, 64);
    let overflow = vec![0u8; buffer.capacity() + 1];
    buffer.put_slice(&overflow);
}

#[test]
#[should_panic(expected = "exceed the available capacity")]
fn cord_buffer_extend_overflow_panics_on_inline_buffer() {
    let mut buffer = CordBuffer::new();
    let overflow = vec![0u8; buffer.capacity() + 1];
    buffer.extend(overflow);
}

#[test]
fn cord_buffer_extend_overflow_keeps_bytes_written_before_it() {
    // Heap (`Flat`) buffer, to exercise the other representation than the
    // panicking test above.
    let mut buffer = CordBuffer::with_capacity(64);
    let cap = buffer.capacity();
    let overflow: Vec<u8> = std::iter::repeat(0..=u8::MAX).flatten().take(cap + 5).collect();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        buffer.extend(overflow.iter().copied());
    }));
    assert!(result.is_err(), "extend should propagate the capacity overflow as a panic");
    assert_eq!(buffer.as_slice(), &overflow[..cap], "bytes written before the overflow must be kept");
    assert_eq!(buffer.available(), 0);
}

#[cfg(feature = "bytes")]
#[test]
#[should_panic(expected = "cannot advance past the end of a Cord")]
fn copy_to_bytes_over_read_panics() {
    use bytes::Buf;
    let mut cord = Cord::from("short");
    let len = cord.len() + 1;
    let _ = cord.copy_to_bytes(len);
}

#[test]
#[should_panic(expected = "make_substring: len 21 out of range for src of length 20 at offset 0")]
fn make_substring_out_of_range_len_panics() {
    // Long enough to force a real tree node (`make_substring` requires
    // `src` to hold a flat/external node, not inline data); a release build
    // must reject this the same as debug, since the callee's own range
    // checks are `debug_assert!`s only.
    let c = Cord::from("longer than 15 bytes");
    let _ = internal::make_substring(&c, 0, c.len() + 1);
}

#[test]
fn make_substring_in_range_sanity() {
    let c = Cord::from("longer than 15 bytes");
    let sub = internal::make_substring(&c, 1, c.len() - 1);
    assert_eq!(sub, "onger than 15 bytes");
}
