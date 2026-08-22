//! `std::io` integration. Gated behind the `std` feature as a whole by
//! `lib.rs`'s `#[cfg(feature = "std")] mod io;`; `core::fmt::Write for Cord`
//! is not `std`-only, so it lives in `cord.rs` instead, next to `Debug`.

use std::io;

use alloc::vec::Vec;

use crate::buffer::CordBuffer;
use crate::cord::Cord;
use crate::iter::Cursor;

#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
impl io::Write for Cord {
    /// Appends `buf` to the cord. Never fails and never writes partially.
    #[inline]
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.append_slice(buf);
        Ok(buf.len())
    }

    #[inline]
    fn write_all(&mut self, buf: &[u8]) -> io::Result<()> {
        self.append_slice(buf);
        Ok(())
    }

    #[inline]
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
impl io::Write for CordBuffer {
    /// Writes as many bytes as fit in the available capacity. Returns
    /// `Ok(0)` when the buffer is full (so `write_all` fails with
    /// `WriteZero`).
    #[inline]
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        Ok(self.put_slice_partial(buf))
    }

    #[inline]
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
impl io::Read for Cursor<'_> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let mut written = 0;
        while written < buf.len() {
            let chunk = self.chunk();
            if chunk.is_empty() {
                break;
            }
            let n = chunk.len().min(buf.len() - written);
            buf[written..written + n].copy_from_slice(&chunk[..n]);
            self.advance(n);
            written += n;
        }
        Ok(written)
    }

    fn read_to_end(&mut self, buf: &mut Vec<u8>) -> io::Result<usize> {
        let n = self.remaining();
        buf.reserve(n);
        for chunk in self.chunks() {
            buf.extend_from_slice(chunk);
        }
        self.advance(n);
        Ok(n)
    }
}

#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
impl io::BufRead for Cursor<'_> {
    #[inline]
    fn fill_buf(&mut self) -> io::Result<&[u8]> {
        Ok(self.chunk())
    }

    #[inline]
    fn consume(&mut self, amt: usize) {
        self.advance(amt);
    }
}

#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
impl io::Seek for Cursor<'_> {
    /// Seeks to a byte offset within the cord.
    ///
    /// Unlike [`std::io::Cursor`], whose position may run past the end of
    /// its data, a `Cursor` maintains `position() + remaining() == len` at
    /// all times: a target beyond the cord's length is
    /// [`io::ErrorKind::InvalidInput`], same as a target that would resolve
    /// to a negative position (an out-of-range `SeekFrom::End`/`Current`
    /// delta). A failed seek leaves the cursor's position unchanged.
    ///
    /// Seeking to or past the current position advances in place; seeking
    /// backward rebuilds the cursor from the start of the cord and advances
    /// from there. Either way the cost is O(log n) in the number of chunks.
    fn seek(&mut self, pos: io::SeekFrom) -> io::Result<u64> {
        let current = self.position();
        let len = current + self.remaining();

        let target = match pos {
            io::SeekFrom::Start(n) => Some(n),
            io::SeekFrom::End(delta) => {
                u64::try_from(len).ok().and_then(|len_u64| len_u64.checked_add_signed(delta))
            }
            io::SeekFrom::Current(delta) => {
                u64::try_from(current).ok().and_then(|current_u64| current_u64.checked_add_signed(delta))
            }
        };
        let invalid = || {
            io::Error::new(io::ErrorKind::InvalidInput, "invalid seek to a negative or out-of-range position")
        };
        let target = target.ok_or_else(invalid)?;
        let target_len = usize::try_from(target).ok().filter(|&t| t <= len).ok_or_else(invalid)?;

        if target_len >= current {
            self.advance(target_len - current);
        } else {
            *self = Cursor::new(self.cord());
            self.advance(target_len);
        }
        Ok(target)
    }

    /// Cheaper than the default (`seek(SeekFrom::Current(0))`): returns the
    /// current position directly.
    #[inline]
    fn stream_position(&mut self) -> io::Result<u64> {
        Ok(u64::try_from(self.position()).unwrap_or(u64::MAX))
    }
}
