//! `std::io` / `core::fmt` integration.

use core::fmt;
use core::fmt::Write as _;
use std::io;

use crate::buffer::CordBuffer;
use crate::cord::Cord;
use crate::iter::{Chunks, Cursor};

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

/// Writes the chunks as UTF-8, replacing invalid sequences with `U+FFFD`
/// (like `String::from_utf8_lossy`) and correctly decoding characters that
/// span chunk boundaries, without allocating.
pub(crate) fn fmt_lossy(chunks: Chunks<'_>, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    /// Decodes `buf`, storing an incomplete trailing sequence in `carry`.
    fn decode(
        mut buf: &[u8],
        carry: &mut [u8; 4],
        carry_len: &mut usize,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        loop {
            match core::str::from_utf8(buf) {
                Ok(s) => return f.write_str(s),
                Err(e) => {
                    let valid = e.valid_up_to();
                    // SAFETY: `from_utf8` validated this prefix.
                    f.write_str(unsafe { core::str::from_utf8_unchecked(&buf[..valid]) })?;
                    match e.error_len() {
                        Some(n) => {
                            f.write_char(char::REPLACEMENT_CHARACTER)?;
                            buf = &buf[valid + n..];
                        }
                        None => {
                            // Incomplete sequence at the end: keep it.
                            let tail = &buf[valid..];
                            carry[..tail.len()].copy_from_slice(tail);
                            *carry_len = tail.len();
                            return Ok(());
                        }
                    }
                }
            }
        }
    }

    let mut carry = [0u8; 4];
    let mut carry_len = 0;
    for chunk in chunks {
        let mut rest = chunk;
        // Complete (or reject) a pending sequence using at most the bytes
        // needed for a maximal UTF-8 character. Rejecting it may leave a new
        // partial sequence from the taken bytes, hence the loop; every
        // iteration consumes at least one byte of `rest`.
        while carry_len > 0 && !rest.is_empty() {
            let take = (4 - carry_len).min(rest.len());
            let mut tmp = [0u8; 8];
            tmp[..carry_len].copy_from_slice(&carry[..carry_len]);
            tmp[carry_len..carry_len + take].copy_from_slice(&rest[..take]);
            let tmp_len = carry_len + take;
            rest = &rest[take..];
            carry_len = 0;
            decode(&tmp[..tmp_len], &mut carry, &mut carry_len, f)?;
        }
        if !rest.is_empty() {
            debug_assert!(carry_len == 0);
            decode(rest, &mut carry, &mut carry_len, f)?;
        }
    }
    if carry_len > 0 {
        f.write_char(char::REPLACEMENT_CHARACTER)?;
    }
    Ok(())
}
