//! The 16 byte in-place storage of a [`Cord`](crate::Cord).
//!
//! `InlineData` holds either up to 15 bytes of inline data or a pointer to a
//! tree ([`CordRep`]). Byte zero is the control byte: for inline data it holds
//! `size << 1` (bit zero clear); for a tree it is `1` (bit zero set), followed
//! by 7 padding bytes and the rep pointer. Port of abseil's
//! `cord_internal::InlineData` with the Cordz sampling pointer removed.
//!
//! Invariant: bytes beyond the inline size are always zero. Several fast paths
//! (`compare`, `is_same`) rely on it.

use core::cmp::Ordering;

use crate::rep::{CordRep, MAX_INLINE, small_memmove, small_u8};

#[repr(C)]
#[derive(Clone, Copy)]
struct AsTree {
    tag: u8,
    _pad: [u8; 7],
    rep: *mut CordRep,
}

/// See the [module documentation](self).
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) union InlineData {
    bytes: [u8; MAX_INLINE + 1],
    tree: AsTree,
}

const _: () = assert!(core::mem::size_of::<InlineData>() == MAX_INLINE + 1);
const _: () = assert!(core::mem::size_of::<AsTree>() <= MAX_INLINE + 1);

impl InlineData {
    /// The empty value.
    #[inline]
    pub(crate) const fn new() -> Self {
        Self { bytes: [0; MAX_INLINE + 1] }
    }

    /// A value holding the tree `rep`.
    #[inline]
    pub(crate) fn from_tree(rep: *mut CordRep) -> Self {
        let mut data = Self::new();
        data.make_tree(rep);
        data
    }

    #[inline]
    fn tag(&self) -> u8 {
        // SAFETY: all 16 bytes are always initialized.
        unsafe { self.bytes[0] }
    }

    #[inline]
    fn set_tag(&mut self, tag: u8) {
        // SAFETY: writing a u8 into an always-initialized byte array.
        unsafe { self.bytes[0] = tag }
    }

    /// Returns `true` if this holds an inline value of zero length.
    #[inline]
    pub(crate) fn is_empty(&self) -> bool {
        self.tag() == 0
    }

    /// Returns `true` if this holds a tree.
    #[inline]
    pub(crate) fn is_tree(&self) -> bool {
        self.tag() & 1 != 0
    }

    /// Returns the tree. Requires `is_tree()`.
    #[inline]
    pub(crate) fn as_tree(&self) -> *mut CordRep {
        debug_assert!(self.is_tree());
        // SAFETY: `is_tree()` implies the `tree` variant was last written.
        unsafe { self.tree.rep }
    }

    /// Returns the tree if any.
    #[inline]
    pub(crate) fn tree(&self) -> Option<*mut CordRep> {
        if self.is_tree() { Some(self.as_tree()) } else { None }
    }

    /// Initializes this instance to hold the tree `rep`.
    #[inline]
    pub(crate) fn make_tree(&mut self, rep: *mut CordRep) {
        debug_assert!(!rep.is_null());
        self.tree = AsTree { tag: 1, _pad: [0; 7], rep };
    }

    /// Replaces the tree. Requires `is_tree()`.
    #[inline]
    pub(crate) fn set_tree(&mut self, rep: *mut CordRep) {
        debug_assert!(self.is_tree());
        debug_assert!(!rep.is_null());
        self.tree.rep = rep;
    }

    /// Returns the inline size. Requires `!is_tree()`.
    #[inline]
    pub(crate) fn inline_size(&self) -> usize {
        debug_assert!(!self.is_tree());
        (self.tag() >> 1) as usize
    }

    /// Sets the inline size. Requires `size <= MAX_INLINE`. Callers must keep
    /// the zero-tail invariant.
    #[inline]
    pub(crate) fn set_inline_size(&mut self, size: usize) {
        debug_assert!(size <= MAX_INLINE);
        self.set_tag(small_u8(size << 1));
    }

    /// Read-only pointer to the inline character data. Requires `!is_tree()`.
    #[inline]
    pub(crate) fn as_chars(&self) -> *const u8 {
        debug_assert!(!self.is_tree());
        // SAFETY: pointer into the always-initialized byte array.
        unsafe { self.bytes.as_ptr().add(1) }
    }

    /// Mutable pointer to the inline character data (15 bytes).
    ///
    /// Intended for write-only use when setting an inline value; the size may
    /// be set before or after writing the data.
    #[inline]
    pub(crate) fn as_chars_mut(&mut self) -> *mut u8 {
        // SAFETY: pointer into the always-initialized byte array.
        unsafe { self.bytes.as_mut_ptr().add(1) }
    }

    /// The inline data as a slice. Requires `!is_tree()`.
    #[inline]
    pub(crate) fn inline_slice(&self) -> &[u8] {
        let n = self.inline_size();
        // SAFETY: `n <= 15` and the bytes are initialized.
        unsafe { &self.bytes[1..=n] }
    }

    /// Sets the inline data and size, zero padding the tail.
    #[inline]
    pub(crate) fn set_inline_data(&mut self, data: &[u8]) {
        debug_assert!(data.len() <= MAX_INLINE);
        self.set_tag(small_u8(data.len() << 1));
        // SAFETY: destination has room for 15 bytes; source has `data.len()`.
        unsafe { small_memmove::<true>(self.as_chars_mut(), data.as_ptr(), data.len()) }
    }

    /// Copies all 15 inline bytes to `dst` (which must have room for 15
    /// bytes). Requires `!is_tree()`.
    #[inline]
    pub(crate) unsafe fn copy_max_inline_to(&self, dst: *mut u8) {
        debug_assert!(!self.is_tree());
        core::ptr::copy_nonoverlapping(self.as_chars(), dst, MAX_INLINE);
    }

    /// Byte-wise equality of the whole 16 bytes (same inline value, or same
    /// tree pointer).
    #[inline]
    pub(crate) fn is_same(&self, other: &Self) -> bool {
        // SAFETY: reading always-initialized bytes (pointer provenance is
        // irrelevant for a comparison).
        unsafe { self.bytes == other.bytes }
    }

    /// Lexicographic comparison of two inline values. Requires both to be
    /// inline.
    #[inline]
    pub(crate) fn compare(&self, rhs: &Self) -> Ordering {
        debug_assert!(!self.is_tree() && !rhs.is_tree());
        // SAFETY: always-initialized bytes.
        let (l, r) = unsafe { (&self.bytes, &rhs.bytes) };
        // Bytes 1..9 then 8..16 cover the 15 data bytes with two wide loads.
        // Thanks to the zero tail, equal prefixes with different sizes are
        // decided by the size comparison below.
        let mut x = u64::from_be_bytes(l[1..9].try_into().unwrap());
        let mut y = u64::from_be_bytes(r[1..9].try_into().unwrap());
        if x == y {
            x = u64::from_be_bytes(l[8..16].try_into().unwrap());
            y = u64::from_be_bytes(r[8..16].try_into().unwrap());
            if x == y {
                return self.inline_size().cmp(&rhs.inline_size());
            }
        }
        x.cmp(&y)
    }
}

impl Default for InlineData {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inline_roundtrip() {
        let mut d = InlineData::new();
        assert!(d.is_empty());
        assert!(!d.is_tree());
        assert_eq!(d.inline_size(), 0);
        d.set_inline_data(b"hello");
        assert_eq!(d.inline_slice(), b"hello");
        assert!(!d.is_empty());
        d.set_inline_data(b"123456789012345");
        assert_eq!(d.inline_slice(), b"123456789012345");
        d.set_inline_data(b"ab");
        assert_eq!(d.inline_slice(), b"ab");
        // Zero tail invariant.
        assert!(unsafe { &d.bytes[3..] }.iter().all(|&b| b == 0));
    }

    #[test]
    fn tree_roundtrip() {
        let mut d = InlineData::new();
        let fake: *mut CordRep = core::ptr::without_provenance_mut(0x1000);
        d.make_tree(fake);
        assert!(d.is_tree());
        assert!(!d.is_empty());
        assert_eq!(d.as_tree(), fake);
        let fake2: *mut CordRep = core::ptr::without_provenance_mut(0x2000);
        d.set_tree(fake2);
        assert_eq!(d.tree(), Some(fake2));
        let copy = d;
        assert!(copy.is_same(&d));
        assert_eq!(InlineData::from_tree(fake2).as_tree(), fake2);
        assert!(InlineData::from_tree(fake2).is_same(&d));
    }

    #[test]
    fn compare_matches_slice_ordering() {
        let samples: &[&[u8]] = &[
            b"",
            b"a",
            b"b",
            b"aa",
            b"ab",
            b"abc",
            b"abcdefgh",
            b"abcdefghi",
            b"abcdefghijklmno",
            b"abcdefghijklmnp",
            b"\xff",
            b"\x00",
            b"\x00\x01",
            b"zzzzzzzzzzzzzzz",
            b"abcdefg\xff",
            b"abcdefg\x00",
        ];
        for &a in samples {
            for &b in samples {
                let mut da = InlineData::new();
                let mut db = InlineData::new();
                da.set_inline_data(a);
                db.set_inline_data(b);
                assert_eq!(da.compare(&db), a.cmp(b), "{a:?} vs {b:?}");
                assert_eq!(da.is_same(&db), a == b);
            }
        }
    }
}
