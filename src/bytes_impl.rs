//! Integration with the [`bytes`] crate.

use std::cmp::Ordering;
use std::io;

use bytes::buf::UninitSlice;
use bytes::{Buf, BufMut, Bytes};

use crate::buffer::CordBuffer;
use crate::cord::Cord;
use crate::iter::Cursor;
use crate::rep::MAX_INLINE;

impl Buf for Cord {
    #[inline]
    fn remaining(&self) -> usize {
        self.len()
    }

    /// The first contiguous chunk of the cord.
    #[inline]
    fn chunk(&self) -> &[u8] {
        self.first_chunk()
    }

    /// Removes the first `cnt` bytes. O(log n).
    #[inline]
    fn advance(&mut self, cnt: usize) {
        Cord::advance(self, cnt);
    }

    /// Splits off the first `len` bytes as a `Bytes`, without copying when
    /// they are contiguous.
    fn copy_to_bytes(&mut self, len: usize) -> Bytes {
        assert!(
            len <= self.len(),
            "cannot advance past the end of a Cord: len = {len}, remaining = {}",
            self.len()
        );
        Bytes::from(self.split_to(len))
    }
}

impl Buf for Cursor<'_> {
    #[inline]
    fn remaining(&self) -> usize {
        Cursor::remaining(self)
    }

    #[inline]
    fn chunk(&self) -> &[u8] {
        Cursor::chunk(self)
    }

    #[inline]
    fn advance(&mut self, cnt: usize) {
        Cursor::advance(self, cnt);
    }

    fn copy_to_bytes(&mut self, len: usize) -> Bytes {
        Bytes::from(self.read_cord(len))
    }
}

impl From<Bytes> for Cord {
    /// Shares the `Bytes` without copying if it is more than 511 bytes;
    /// copies it otherwise.
    #[inline]
    fn from(bytes: Bytes) -> Self {
        let capacity = bytes.len();
        Cord::from_owned(bytes, capacity)
    }
}

/// Owner type handing a flat cord's buffer to `Bytes::from_owner`.
struct FlatCord(Cord);

impl AsRef<[u8]> for FlatCord {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        self.0.as_contiguous().expect("FlatCord holds a flat cord")
    }
}

impl From<Cord> for Bytes {
    /// Converts without copying if the cord's bytes are contiguous (and not
    /// inline); copies otherwise.
    fn from(cord: Cord) -> Self {
        match cord.as_contiguous() {
            Some(flat) if flat.len() <= MAX_INLINE || !cord.is_tree() => Bytes::copy_from_slice(flat),
            Some(_) => Bytes::from_owner(FlatCord(cord)),
            None => Bytes::from(cord.to_vec()),
        }
    }
}

impl PartialEq<Cord> for Bytes {
    #[inline]
    fn eq(&self, other: &Cord) -> bool {
        other.equals(self)
    }
}

impl PartialOrd<Cord> for Bytes {
    #[inline]
    fn partial_cmp(&self, other: &Cord) -> Option<Ordering> {
        Some(other.compare(self).reverse())
    }
}

/// A [`BufMut`] / [`io::Write`] adapter that appends to a [`Cord`] through
/// [`CordBuffer`]s, so that data can be written into uninitialized memory
/// that becomes part of the cord without copying.
///
/// Buffered data is appended to the cord when the writer is dropped, when
/// [`flush`](Self::flush) is called, or whenever a buffer fills up. Large
/// [`put_slice`](BufMut::put_slice) calls bypass the buffer.
///
/// ```
/// use bytes::BufMut;
/// use cord_rs::{Cord, CordWriter};
///
/// let mut cord = Cord::from("head");
/// {
///     let mut writer = CordWriter::new(&mut cord);
///     writer.put_u32(0xDEADBEEF);
///     writer.put_slice(b" tail");
/// }
/// assert_eq!(cord.len(), 4 + 4 + 5);
/// assert!(cord.ends_with(" tail"));
/// ```
pub struct CordWriter<'a> {
    /// `None` only after [`into_inner`](Self::into_inner) has taken it.
    cord: Option<&'a mut Cord>,
    /// The buffer being filled, taken from the cord's spare capacity on first
    /// use so that small writes land in existing buffers.
    buffer: Option<CordBuffer>,
}

impl<'a> CordWriter<'a> {
    /// Creates a writer appending to `cord`.
    pub fn new(cord: &'a mut Cord) -> Self {
        Self { cord: Some(cord), buffer: None }
    }

    /// Appends any buffered data to the cord.
    pub fn flush(&mut self) {
        if let Some(buffer) = self.buffer.take()
            && !buffer.is_empty()
            && let Some(cord) = self.cord.as_deref_mut()
        {
            cord.append(buffer);
        }
    }

    /// Flushes and returns the underlying cord.
    ///
    /// # Panics
    ///
    /// Never panics: `into_inner` is the only place that ever takes `cord`
    /// out, so it is always present when this runs.
    #[must_use]
    pub fn into_inner(mut self) -> &'a mut Cord {
        self.flush();
        self.cord.take().expect("CordWriter used after into_inner")
    }

    /// Returns a buffer with spare capacity, reusing the cord's own spare
    /// capacity where possible.
    fn buffer_mut(&mut self) -> &mut CordBuffer {
        if self.buffer.as_ref().is_none_or(|b| b.available() == 0) {
            self.flush();
            self.buffer = Some(
                self.cord
                    .as_deref_mut()
                    .expect("CordWriter used after into_inner")
                    .take_append_buffer(CordBuffer::DEFAULT_MAX_CAPACITY),
            );
        }
        self.buffer.as_mut().expect("buffer was just set")
    }
}

impl Drop for CordWriter<'_> {
    fn drop(&mut self) {
        self.flush();
    }
}

// SAFETY: `chunk_mut` returns the buffer's spare capacity and `advance_mut`
// only marks bytes within that capacity as initialized.
unsafe impl BufMut for CordWriter<'_> {
    #[inline]
    fn remaining_mut(&self) -> usize {
        usize::MAX
            - self.cord.as_deref().map_or(0, Cord::len)
            - self.buffer.as_ref().map_or(0, CordBuffer::len)
    }

    /// # Safety
    ///
    /// Per [`BufMut::advance_mut`]'s contract, the caller must have
    /// initialized the first `cnt` bytes of the spare capacity most recently
    /// returned by [`chunk_mut`](bytes::BufMut::chunk_mut).
    #[inline]
    unsafe fn advance_mut(&mut self, cnt: usize) {
        let buffer = self.buffer.as_mut().expect("advance_mut without a preceding chunk_mut");
        let new_len = buffer.len() + cnt;
        assert!(
            new_len <= buffer.capacity(),
            "CordWriter::advance_mut: {cnt} bytes exceed the chunk capacity"
        );
        // SAFETY: the caller guarantees the bytes were initialized.
        unsafe { buffer.set_len(new_len) };
        if buffer.available() == 0 {
            self.flush();
        }
    }

    #[inline]
    fn chunk_mut(&mut self) -> &mut UninitSlice {
        UninitSlice::uninit(self.buffer_mut().spare_capacity_mut())
    }

    fn put_slice(&mut self, src: &[u8]) {
        let buffer = self.buffer_mut();
        if src.len() <= buffer.available() {
            buffer.put_slice(src);
        } else {
            self.flush();
            self.cord.as_deref_mut().expect("CordWriter used after into_inner").append(src);
        }
    }
}

impl io::Write for CordWriter<'_> {
    #[inline]
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        BufMut::put_slice(self, buf);
        Ok(buf.len())
    }

    #[inline]
    fn flush(&mut self) -> io::Result<()> {
        CordWriter::flush(self);
        Ok(())
    }
}
