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
use core::mem::MaybeUninit;

use crate::rep::{CordRep, MAX_INLINE, OwnedRep, RepRef, UniqueRep, small_u8};

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

/// The checked, safe view of an [`InlineData`]'s union: the result of
/// [`InlineData::view`].
pub(crate) enum Repr<'a> {
    /// Inline data (0 to 15 bytes).
    Inline(&'a [u8]),
    /// A tree.
    Tree(RepRef<'a>),
}

impl InlineData {
    /// The empty value.
    #[inline]
    pub(crate) const fn new() -> Self {
        Self { bytes: [0; MAX_INLINE + 1] }
    }

    /// A value holding the tree `rep`, adopting its reference.
    #[inline]
    pub(crate) fn from_tree(rep: OwnedRep) -> Self {
        let mut data = Self::new();
        data.set_tree(rep);
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
    fn make_tree(&mut self, rep: *mut CordRep) {
        debug_assert!(!rep.is_null());
        self.tree = AsTree { tag: 1, _pad: [0; 7], rep };
    }

    /// Adopts `rep` as this value's tree, overwriting whatever was here
    /// before (inline data, or a different tree) without touching any
    /// previous tree's reference count — the caller must have already
    /// accounted for it (typically by having just extracted it via
    /// [`take_tree`](Self::take_tree), or because there was none).
    #[inline]
    pub(crate) fn set_tree(&mut self, rep: OwnedRep) {
        self.make_tree(rep.into_raw());
    }

    /// Steals the tree out of this value, if any, resetting it to empty
    /// inline data and transferring the tree's reference to the returned
    /// [`OwnedRep`].
    #[inline]
    pub(crate) fn take_tree(&mut self) -> Option<OwnedRep> {
        if !self.is_tree() {
            return None;
        }
        let rep = self.as_tree();
        *self = Self::new();
        // SAFETY: `rep` was this value's own tree reference; resetting
        // `self` to empty inline data above transfers that single owned
        // reference out without touching its refcount, matching
        // `OwnedRep::from_raw`'s adopt contract.
        Some(unsafe { OwnedRep::from_raw(rep) })
    }

    /// Builds a fresh inline value holding `data`. Requires
    /// `data.len() <= MAX_INLINE`. One zero store plus one fused
    /// copy-and-zero.
    #[inline]
    pub(crate) fn inline_from(data: &[u8]) -> Self {
        debug_assert!(data.len() <= MAX_INLINE);
        let mut this = Self::new();
        // SAFETY: `data.len() <= MAX_INLINE` (asserted above) and the
        // destination is the 15-byte tail; `NULLIFY_TAIL` zero-fills
        // `tail[len..]`, so all 15 bytes end up written — abseil's fused
        // copy-and-zero, one branchless dance instead of copy + fill.
        unsafe {
            crate::rep::small_memmove::<true>(this.tail_mut().as_mut_ptr(), data.as_ptr(), data.len());
        }
        this.set_inline_size(data.len());
        this
    }

    /// Releases any held tree reference and resets only the tag byte, so
    /// the value is logically empty but its tail bytes may be stale —
    /// violating the zero-tail invariant that `is_same`/`compare` rely on.
    /// ONLY for [`Drop`]: the value must not be read, compared, or reused
    /// after this call. (Skipping the full 16-byte reset is what keeps
    /// clone+drop pairs on the old fast path.)
    #[inline]
    pub(crate) fn release_for_drop(&mut self) {
        if self.is_tree() {
            // SAFETY: the tree is live per `self`'s invariant; this drops
            // the reference the value held.
            unsafe { crate::rep::unref(self.as_tree()) };
        }
        self.set_inline_size(0);
    }

    /// O(1) copy of this value: a bitwise copy which, when a tree is held,
    /// first takes one additional reference on it. The clone fast path: one
    /// tag test, at most one atomic increment, one 16-byte copy — no
    /// [`view`](Self::view) dispatch (which materializes the inline slice).
    #[inline]
    pub(crate) fn clone_with_ref(&self) -> Self {
        if self.is_tree() {
            // SAFETY: the tree is live per `self`'s invariant; incrementing
            // its refcount shares it with the returned copy.
            unsafe { crate::rep::RepPtr::ref_inc(self.as_tree()) };
        }
        *self
    }

    /// Number of bytes held (tree length or inline size): one tag test,
    /// no slice materialization — the `len()` fast path.
    #[inline]
    pub(crate) fn len(&self) -> usize {
        if self.is_tree() {
            // SAFETY: the tree is live per `self`'s invariant.
            unsafe { crate::rep::RepPtr::length(self.as_tree()) }
        } else {
            self.inline_size()
        }
    }

    /// The checked, safe view of this value: the single consumer-facing
    /// read of the inline/tree union.
    #[inline]
    pub(crate) fn view(&self) -> Repr<'_> {
        if self.is_tree() {
            // SAFETY: `is_tree()` guarantees the `tree` variant was last
            // written; `&self`'s borrow proves the rep stays live and
            // unmutated (other than through its interior-mutable refcount)
            // for the returned `Repr`'s lifetime.
            Repr::Tree(unsafe { RepRef::from_raw(self.as_tree()) })
        } else {
            Repr::Inline(self.inline_slice())
        }
    }

    /// Attempts to obtain a mutation witness for this value's tree: `Some`
    /// iff it holds a tree whose reference count is exactly one. See
    /// [`UniqueRep`]'s soundness note for why this (a `&mut self` path) is
    /// the only sound way to construct one.
    #[inline]
    pub(crate) fn tree_unique(&mut self) -> Option<UniqueRep<'_>> {
        if !self.is_tree() {
            return None;
        }
        let ptr = self.as_tree();
        // SAFETY: `ptr` is live per `is_tree()`; this `RepRef` is scoped to
        // just the `ref_is_one()` read below, not retained.
        if !unsafe { RepRef::from_raw(ptr) }.ref_is_one() {
            return None;
        }
        // SAFETY: this `&mut self` borrow proves no other handle to
        // `self`'s slot exists, and `ref_is_one()` just confirmed no
        // reference outside this slot exists either — together, `ptr` has
        // exactly one live handle anywhere: this call. See `UniqueRep`'s
        // soundness note.
        Some(unsafe { UniqueRep::from_raw(ptr) })
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
    ///
    /// A permanent, intentional escape hatch to the raw pointer: `cord.rs`'s
    /// one call site needs it for a `copy_nonoverlapping` bulk copy, which
    /// the safe editing API below (`push_back_inline` etc.) has no slice-based
    /// equivalent for by design. New code that isn't doing that kind of raw
    /// bulk copy should prefer those, or [`tail`](Self::tail).
    #[inline]
    pub(crate) fn as_chars(&self) -> *const u8 {
        debug_assert!(!self.is_tree());
        // SAFETY: pointer into the always-initialized byte array.
        unsafe { self.bytes.as_ptr().add(1) }
    }

    /// The inline data as a slice. Requires `!is_tree()`.
    #[inline]
    pub(crate) fn inline_slice(&self) -> &[u8] {
        let n = self.inline_size();
        // SAFETY: `n <= 15` and the bytes are initialized.
        unsafe { &self.bytes[1..=n] }
    }

    /// The 15 data bytes, read-only. The tag byte (`bytes[0]`) is not
    /// reachable through this accessor.
    #[inline]
    fn tail(&self) -> &[u8; MAX_INLINE] {
        // SAFETY: all 16 bytes of the `bytes` union variant are always
        // initialized; slicing off the tag byte leaves exactly the
        // remaining 15, matching the array size below.
        (unsafe { &self.bytes })[1..].try_into().unwrap()
    }

    /// The 15 data bytes, mutable. The tag byte (`bytes[0]`) is not
    /// reachable through this accessor, so inline-editing code built on it
    /// can never scribble the tag/tree discriminant. Callers outside this
    /// module inherit the invariants the editing API otherwise maintains:
    /// bytes beyond the inline size must stay zero, and the size must be
    /// committed via [`set_inline_size`](Self::set_inline_size) before the
    /// value is observed.
    #[inline]
    pub(crate) fn tail_mut(&mut self) -> &mut [u8; MAX_INLINE] {
        // SAFETY: see `tail`.
        let bytes: &mut [u8; MAX_INLINE + 1] = unsafe { &mut self.bytes };
        (&mut bytes[1..]).try_into().unwrap()
    }

    /// Appends `src` to this inline value in place. Requires
    /// `self.inline_size() + src.len() <= MAX_INLINE`.
    #[inline]
    pub(crate) fn push_back_inline(&mut self, src: &[u8]) {
        let cur = self.inline_size();
        debug_assert!(cur + src.len() <= MAX_INLINE);
        self.tail_mut()[cur..cur + src.len()].copy_from_slice(src);
        self.set_inline_size(cur + src.len());
    }

    /// Prepends `src` to this inline value in place. Requires
    /// `self.inline_size() + src.len() <= MAX_INLINE`.
    #[inline]
    pub(crate) fn push_front_inline(&mut self, src: &[u8]) {
        *self = Self::concat_inline(src, self.inline_slice());
    }

    /// Returns a fresh inline value holding `a` followed by `b`. Requires
    /// `a.len() + b.len() <= MAX_INLINE`.
    #[inline]
    pub(crate) fn concat_inline(a: &[u8], b: &[u8]) -> Self {
        debug_assert!(a.len() + b.len() <= MAX_INLINE);
        let mut out = Self::new();
        let dst = out.tail_mut();
        dst[..a.len()].copy_from_slice(a);
        dst[a.len()..a.len() + b.len()].copy_from_slice(b);
        out.set_inline_size(a.len() + b.len());
        out
    }

    /// Truncates to `new_len` inline bytes, zero-filling the freed tail.
    /// Requires `new_len <= self.inline_size()`.
    #[inline]
    pub(crate) fn truncate_inline(&mut self, new_len: usize) {
        let cur = self.inline_size();
        debug_assert!(new_len <= cur);
        self.tail_mut()[new_len..cur].fill(0);
        self.set_inline_size(new_len);
    }

    /// Removes the first `n` bytes, shifting the rest to the front and
    /// zero-filling the freed tail. Requires `n <= self.inline_size()`.
    #[inline]
    pub(crate) fn drop_front_inline(&mut self, n: usize) {
        let cur = self.inline_size();
        debug_assert!(n <= cur);
        let new_len = cur - n;
        let tail = self.tail_mut();
        tail.copy_within(n..cur, 0);
        tail[new_len..cur].fill(0);
        self.set_inline_size(new_len);
    }

    /// Copies this inline value's full 15-byte storage (including any zero
    /// tail) to the front of `dst`. Requires `!is_tree()` and `dst.len() >=
    /// MAX_INLINE`.
    #[inline]
    pub(crate) fn copy_max_inline_to(&self, dst: &mut [MaybeUninit<u8>]) {
        debug_assert!(!self.is_tree());
        dst[..MAX_INLINE].write_copy_of_slice(self.tail());
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
        d = InlineData::inline_from(b"hello");
        assert_eq!(d.inline_slice(), b"hello");
        assert!(!d.is_empty());
        d = InlineData::inline_from(b"123456789012345");
        assert_eq!(d.inline_slice(), b"123456789012345");
        d = InlineData::inline_from(b"ab");
        assert_eq!(d.inline_slice(), b"ab");
        // Zero tail invariant.
        // SAFETY: reading always-initialized bytes of the `bytes` union
        // variant (last written by `inline_from` above).
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
        // SAFETY: `fake`/`fake2` are synthetic addresses used only to
        // exercise `InlineData`'s bit patterns, never dereferenced. The
        // `OwnedRep` values wrapping them are always consumed via
        // `into_raw()` (inside `set_tree`/`from_tree`), never dropped, so
        // `unref` never actually runs on them.
        d.set_tree(unsafe { OwnedRep::from_raw(fake2) });
        assert_eq!(d.tree(), Some(fake2));
        let copy = d;
        assert!(copy.is_same(&d));
        assert_eq!(InlineData::from_tree(unsafe { OwnedRep::from_raw(fake2) }).as_tree(), fake2);
        assert!(InlineData::from_tree(unsafe { OwnedRep::from_raw(fake2) }).is_same(&d));
    }

    #[test]
    fn view_and_take_tree() {
        let mut d = InlineData::inline_from(b"hi");
        assert!(matches!(d.view(), Repr::Inline(b) if b == b"hi"));
        assert!(d.take_tree().is_none());

        let fake: *mut CordRep = core::ptr::without_provenance_mut(0x4000);
        // SAFETY: see `tree_roundtrip`; `fake` is never dereferenced.
        d.set_tree(unsafe { OwnedRep::from_raw(fake) });
        assert!(matches!(d.view(), Repr::Tree(r) if r.as_ptr() == fake));
        let taken = d.take_tree().expect("was a tree");
        assert!(d.is_empty());
        assert!(!d.is_tree());
        // Don't drop `taken`: `fake` is not a real rep.
        core::mem::forget(taken);
    }

    #[test]
    fn inline_editing_api() {
        let mut d = InlineData::new();
        d.push_back_inline(b"abc");
        assert_eq!(d.inline_slice(), b"abc");
        d.push_back_inline(b"de");
        assert_eq!(d.inline_slice(), b"abcde");
        d.push_front_inline(b"XY");
        assert_eq!(d.inline_slice(), b"XYabcde");
        d.drop_front_inline(2);
        assert_eq!(d.inline_slice(), b"abcde");
        d.truncate_inline(3);
        assert_eq!(d.inline_slice(), b"abc");
        // SAFETY: reading always-initialized bytes of the `bytes` union
        // variant (last written by `truncate_inline` above).
        assert!(unsafe { &d.bytes[4..] }.iter().all(|&b| b == 0));

        let cat = InlineData::concat_inline(b"foo", b"bar");
        assert_eq!(cat.inline_slice(), b"foobar");

        let mut buf = [MaybeUninit::new(0xAAu8); MAX_INLINE];
        let e = InlineData::inline_from(b"hello world!!!!");
        e.copy_max_inline_to(&mut buf);
        // SAFETY: `copy_max_inline_to` initializes all `MAX_INLINE` bytes.
        let init: [u8; MAX_INLINE] = unsafe { core::mem::transmute(buf) };
        assert_eq!(&init, b"hello world!!!!");
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
                let da = InlineData::inline_from(a);
                let db = InlineData::inline_from(b);
                assert_eq!(da.compare(&db), a.cmp(b), "{a:?} vs {b:?}");
                assert_eq!(da.is_same(&db), a == b);
            }
        }
    }
}
