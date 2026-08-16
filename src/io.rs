//! `std::io` / `core::fmt` integration.

use core::fmt;
use std::io;

use crate::buffer::CordBuffer;
use crate::cord::Cord;
use crate::iter::Cursor;

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

impl fmt::Write for Cord {
    /// Appends `s` to the cord, so `write!(cord, ...)` works.
    #[inline]
    fn write_str(&mut self, s: &str) -> fmt::Result {
        self.append_slice(s.as_bytes());
        Ok(())
    }
}

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
