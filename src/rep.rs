//! Internal representation nodes ("reps") that back a non-inline [`Cord`].
//!
//! This module and its submodules are the `unsafe` core of the crate and are a
//! close port of abseil's `absl/strings/internal/cord_internal.h`,
//! `cord_rep_flat.h`, `cord_data_edge.h` and friends.
//!
//! # Layout
//!
//! Every rep starts with a [`CordRep`] header: `length`, an atomic
//! [`Refcount`], a one byte `tag` and three bytes of `storage` for use by
//! derived reps. Instead of a vtable, the `tag` byte identifies the node type:
//!
//! * `SUBSTRING` (1): a [`CordRepSubstring`] referencing a sub-range of a flat
//!   or external node.
//! * `BTREE` (3): a [`btree::CordRepBtree`] node.
//! * `EXTERNAL` (5): a [`external::CordRepExternal`] node referencing memory
//!   owned by a user provided value (`Vec<u8>`, `Arc<[u8]>`, `&'static [u8]`,
//!   ...).
//! * `FLAT` (6) ..= `MAX_FLAT_TAG` (248): a flat node whose payload directly
//!   follows the header. The tag value encodes the allocated size of the node
//!   (see [`flat`]).
//!
//! # Reference counting conventions
//!
//! Unless documented otherwise, functions that take a `*mut CordRep` argument
//! *adopt* ("consume") a reference on that argument and functions returning a
//! `*mut CordRep` *transfer* a reference back to the caller. A node with a
//! refcount of one is exclusively owned by the caller and may be mutated in
//! place; a shared node is immutable.
//!
//! [`Cord`]: crate::Cord

// Parts of the rep API are only exercised by the test suites and benchmarks.
#![allow(dead_code)]

use core::sync::atomic::{AtomicI32, Ordering};

pub(crate) mod analysis;
pub(crate) mod btree;
#[cfg(test)]
mod btree_tests;
#[cfg(test)]
mod data_edge_tests;
pub(crate) mod external;
pub(crate) mod flat;
pub(crate) mod navigator;
#[cfg(test)]
mod navigator_tests;
pub(crate) mod reader;
#[cfg(test)]
mod reader_tests;
#[cfg(test)]
pub(crate) mod test_util;

/// Prefer copying blocks of at most this size, otherwise reference count.
pub(crate) const MAX_BYTES_TO_COPY: usize = 511;

/// The maximum number of bytes a `Cord` stores inline without allocating.
pub(crate) const MAX_INLINE: usize = 15;

// --- Node tags -------------------------------------------------------------
//
// Numeric values mirror abseil's `CordRepKind`; `2` (CRC) and `4` (RING) are
// intentionally unused. `FLAT == EXTERNAL + 1` lets `is_data_edge` test both
// with a single `tag >= EXTERNAL` comparison.

/// Tag of a [`CordRepSubstring`].
pub(crate) const SUBSTRING: u8 = 1;
/// Tag of a [`btree::CordRepBtree`].
pub(crate) const BTREE: u8 = 3;
/// Tag of a [`external::CordRepExternal`].
pub(crate) const EXTERNAL: u8 = 5;
/// Smallest tag of a flat node (encodes the minimum flat allocation size).
pub(crate) const FLAT: u8 = 6;
/// Largest tag of a flat node (encodes the maximum flat allocation size).
pub(crate) const MAX_FLAT_TAG: u8 = 248;

const _: () = assert!(FLAT == EXTERNAL + 1, "EXTERNAL and FLAT not consecutive");

// --- Refcount --------------------------------------------------------------

const NUM_FLAGS: i32 = 1;
const IMMORTAL_FLAG: i32 = 0x1;
const REF_INCREMENT: i32 = 1 << NUM_FLAGS;

/// Compact atomic reference count with an "immortal" flag in the low bit.
///
/// Mirrors abseil's `RefcountAndFlags`. The count is stored shifted left by
/// one; bit zero marks reps that must never be destroyed.
#[repr(transparent)]
pub(crate) struct Refcount(AtomicI32);

impl Refcount {
    /// A refcount of one.
    #[inline]
    pub(crate) const fn new() -> Self {
        Self(AtomicI32::new(REF_INCREMENT))
    }

    /// An immortal refcount: `decrement` never reports zero.
    #[inline]
    #[allow(dead_code)]
    pub(crate) const fn immortal() -> Self {
        Self(AtomicI32::new(IMMORTAL_FLAG))
    }

    /// Increments the reference count. Imposes no memory ordering.
    #[inline]
    pub(crate) fn increment(&self) {
        let prev = self.0.fetch_add(REF_INCREMENT, Ordering::Relaxed);
        if prev >= (i32::MAX / 3) * 2 {
            increment_overflow();
        }
    }

    /// Decrements the reference count if it is greater than one.
    ///
    /// Returns `false` if there are no references outstanding (the caller
    /// must destroy the rep), `true` otherwise. Always returns `true` for
    /// immortal reps. Inserts the barriers needed for the thread observing
    /// `false` to see all writes made before the last other decrement.
    #[inline]
    pub(crate) fn decrement(&self) -> bool {
        let refcount = self.0.load(Ordering::Acquire);
        debug_assert!(refcount > 0 || refcount & IMMORTAL_FLAG != 0);
        refcount != REF_INCREMENT && self.0.fetch_sub(REF_INCREMENT, Ordering::AcqRel) != REF_INCREMENT
    }

    /// Same as [`decrement`](Self::decrement) but expects the count to be
    /// greater than one, skipping the initial load.
    #[inline]
    pub(crate) fn decrement_expect_high_refcount(&self) -> bool {
        let refcount = self.0.fetch_sub(REF_INCREMENT, Ordering::AcqRel);
        debug_assert!(refcount > 0 || refcount & IMMORTAL_FLAG != 0);
        refcount != REF_INCREMENT
    }

    /// Returns the current reference count (acquire semantics).
    #[inline]
    pub(crate) fn get(&self) -> usize {
        (self.0.load(Ordering::Acquire) >> NUM_FLAGS) as usize
    }

    /// Returns `true` if the count is exactly one, i.e. the caller owns the
    /// only reference and may mutate the rep. Performs the acquire needed to
    /// act on that knowledge. Always `false` for immortal reps.
    #[inline]
    pub(crate) fn is_one(&self) -> bool {
        self.0.load(Ordering::Acquire) == REF_INCREMENT
    }

    /// Returns `true` if the immortal flag is set.
    #[inline]
    pub(crate) fn is_immortal(&self) -> bool {
        self.0.load(Ordering::Relaxed) & IMMORTAL_FLAG != 0
    }
}

#[cold]
#[inline(never)]
fn increment_overflow() -> ! {
    // Mirrors `Arc`: a refcount this large can only be the result of a leak
    // loop and continuing risks a use-after-free on wrap around.
    std::process::abort()
}

// --- CordRep ---------------------------------------------------------------

/// Common header of every rep. See the [module documentation](self).
///
/// The layout is `#[repr(C)]` and must stay in sync with
/// [`flat::FLAT_OVERHEAD`] (the payload of a flat node starts at `storage`).
#[repr(C)]
pub(crate) struct CordRep {
    /// Number of bytes of data represented by this rep.
    pub(crate) length: usize,
    /// Reference count.
    pub(crate) refcount: Refcount,
    /// Node type (see the tag constants) or, for flats, the encoded size.
    pub(crate) tag: u8,
    /// Start of the flat payload, or three bytes of derived-rep storage
    /// (`height`, `begin`, `end` for btree nodes).
    pub(crate) storage: [u8; 3],
}

impl CordRep {
    /// Creates a header with a refcount of one.
    #[inline]
    pub(crate) const fn new(length: usize, tag: u8) -> Self {
        Self { length, refcount: Refcount::new(), tag, storage: [0; 3] }
    }
}

/// Convenience accessors on raw rep pointers.
///
/// All methods are `unsafe`: `self` must point to a live rep. They read and
/// write through the raw pointer without creating references to the header,
/// so they are safe to interleave with other raw pointers to the same node.
pub(crate) trait RepPtr: Copy {
    unsafe fn length(self) -> usize;
    unsafe fn set_length(self, length: usize);
    unsafe fn tag(self) -> u8;
    unsafe fn refcount<'a>(self) -> &'a Refcount;

    #[inline]
    unsafe fn is_substring(self) -> bool {
        self.tag() == SUBSTRING
    }
    #[inline]
    unsafe fn is_btree(self) -> bool {
        self.tag() == BTREE
    }
    #[inline]
    unsafe fn is_external(self) -> bool {
        self.tag() == EXTERNAL
    }
    #[inline]
    unsafe fn is_flat(self) -> bool {
        self.tag() >= FLAT
    }
}

impl RepPtr for *mut CordRep {
    #[inline]
    unsafe fn length(self) -> usize {
        (*self).length
    }
    #[inline]
    unsafe fn set_length(self, length: usize) {
        (*self).length = length;
    }
    #[inline]
    unsafe fn tag(self) -> u8 {
        (*self).tag
    }
    #[inline]
    unsafe fn refcount<'a>(self) -> &'a Refcount {
        &*core::ptr::addr_of!((*self).refcount)
    }
}

/// Increments the reference count of `rep` and returns it.
#[inline]
pub(crate) unsafe fn ref_rep(rep: *mut CordRep) -> *mut CordRep {
    debug_assert!(!rep.is_null());
    rep.refcount().increment();
    rep
}

/// Decrements the reference count of `rep`, destroying it on zero.
#[inline]
pub(crate) unsafe fn unref(rep: *mut CordRep) {
    debug_assert!(!rep.is_null());
    if !rep.refcount().decrement_expect_high_refcount() {
        destroy(rep);
    }
}

/// Destroys `rep`, whose reference count has reached zero.
pub(crate) unsafe fn destroy(mut rep: *mut CordRep) {
    loop {
        debug_assert!(!rep.refcount().is_immortal());
        let tag = rep.tag();
        if tag == BTREE {
            btree::CordRepBtree::destroy(rep.cast());
            return;
        } else if tag == EXTERNAL {
            external::CordRepExternal::delete(rep);
            return;
        } else if tag == SUBSTRING {
            let substring: *mut CordRepSubstring = rep.cast();
            rep = (*substring).child;
            CordRepSubstring::delete(substring);
            if rep.refcount().decrement() {
                return;
            }
        } else {
            debug_assert!(tag >= FLAT);
            flat::delete(rep);
            return;
        }
    }
}

// --- Substring -------------------------------------------------------------

/// A rep referencing `length` bytes starting at `start` of a flat or external
/// `child` node.
#[repr(C)]
pub(crate) struct CordRepSubstring {
    pub(crate) rep: CordRep,
    /// Starting offset of the substring inside `child`.
    pub(crate) start: usize,
    /// The referenced flat or external node.
    pub(crate) child: *mut CordRep,
}

impl CordRepSubstring {
    /// Creates a substring on `child`, adopting a reference on `child`.
    ///
    /// Requires `child` to be a flat or external node and `pos`/`n` to form a
    /// non-empty partial sub range of `child`: `n > 0 && n < child.length &&
    /// pos + n <= child.length`.
    #[inline]
    pub(crate) unsafe fn create(child: *mut CordRep, pos: usize, n: usize) -> *mut CordRepSubstring {
        debug_assert!(!child.is_null());
        debug_assert!(n > 0);
        debug_assert!(n < child.length());
        debug_assert!(pos < child.length());
        debug_assert!(n <= child.length() - pos);
        assert!(
            child.is_external() || child.is_flat(),
            "cord-rs: unexpected node type {} for substring child",
            child.tag()
        );
        Box::into_raw(Box::new(CordRepSubstring { rep: CordRep::new(n, SUBSTRING), start: pos, child }))
    }

    /// Creates a substring of `rep` **without** adopting a reference on `rep`.
    ///
    /// Requires `is_data_edge(rep) && n > 0 && pos + n <= rep.length`. If
    /// `n == rep.length` this returns `ref_rep(rep)`. If `rep` is itself a
    /// substring, the returned substring references its child with `pos`
    /// adjusted by the original `start`.
    #[inline]
    pub(crate) unsafe fn substring(mut rep: *mut CordRep, mut pos: usize, n: usize) -> *mut CordRep {
        debug_assert!(!rep.is_null());
        debug_assert!(n != 0);
        debug_assert!(pos < rep.length());
        debug_assert!(n <= rep.length() - pos);
        if n == rep.length() {
            return ref_rep(rep);
        }
        if rep.is_substring() {
            let sub: *mut CordRepSubstring = rep.cast();
            pos += (*sub).start;
            rep = (*sub).child;
        }
        let substring =
            Box::new(CordRepSubstring { rep: CordRep::new(n, SUBSTRING), start: pos, child: ref_rep(rep) });
        Box::into_raw(substring).cast()
    }

    /// Frees the substring node itself (not its child).
    #[inline]
    pub(crate) unsafe fn delete(substring: *mut CordRepSubstring) {
        drop(Box::from_raw(substring));
    }
}

// --- Data edges ------------------------------------------------------------

/// Returns `true` if `edge` is a FLAT, EXTERNAL or a SUBSTRING of a FLAT or
/// EXTERNAL node.
#[inline]
pub(crate) unsafe fn is_data_edge(mut edge: *const CordRep) -> bool {
    debug_assert!(!edge.is_null());
    // Fast path: EXTERNAL or FLAT is a single well predicted branch.
    let tag = (*edge).tag;
    if tag == EXTERNAL || tag >= FLAT {
        return true;
    }
    if tag == SUBSTRING {
        edge = (*edge.cast::<CordRepSubstring>()).child;
    }
    let tag = (*edge).tag;
    tag == EXTERNAL || tag >= FLAT
}

/// Returns the bytes referenced by the data edge `edge`.
///
/// Requires `is_data_edge(edge)`. The returned slice borrows the node's
/// memory: the caller must ensure the edge outlives `'a` and is not mutated
/// (i.e. it is held by a live cord and not being appended to in place).
#[inline]
pub(crate) unsafe fn edge_data<'a>(mut edge: *const CordRep) -> &'a [u8] {
    debug_assert!(is_data_edge(edge));
    let mut offset = 0;
    let length = (*edge).length;
    if (*edge).tag == SUBSTRING {
        let sub = edge.cast::<CordRepSubstring>();
        offset = (*sub).start;
        edge = (*sub).child;
    }
    let base = if (*edge).tag >= FLAT {
        flat::data(edge as *mut CordRep) as *const u8
    } else {
        (*edge.cast::<external::CordRepExternal>()).base
    };
    core::slice::from_raw_parts(base.add(offset), length)
}

// --- Small memmove ---------------------------------------------------------

/// Fast `memmove` for up to 15 bytes, safe for overlapping regions.
///
/// If `NULLIFY_TAIL` is true the destination is zero padded up to 15 bytes
/// (so `dst` must have room for 15 bytes regardless of `n`).
#[inline]
pub(crate) unsafe fn small_memmove<const NULLIFY_TAIL: bool>(dst: *mut u8, src: *const u8, n: usize) {
    use core::ptr::{read_unaligned, write_bytes, write_unaligned};
    if n >= 8 {
        debug_assert!(n <= 15);
        let buf1 = read_unaligned(src.cast::<u64>());
        let buf2 = read_unaligned(src.add(n - 8).cast::<u64>());
        if NULLIFY_TAIL {
            write_bytes(dst.add(7), 0, 8);
        }
        write_unaligned(dst.cast::<u64>(), buf1);
        write_unaligned(dst.add(n - 8).cast::<u64>(), buf2);
    } else if n >= 4 {
        let buf1 = read_unaligned(src.cast::<u32>());
        let buf2 = read_unaligned(src.add(n - 4).cast::<u32>());
        if NULLIFY_TAIL {
            write_bytes(dst.add(4), 0, 4);
            write_bytes(dst.add(7), 0, 8);
        }
        write_unaligned(dst.cast::<u32>(), buf1);
        write_unaligned(dst.add(n - 4).cast::<u32>(), buf2);
    } else {
        if n != 0 {
            let b0 = *src;
            let bm = *src.add(n / 2);
            let bl = *src.add(n - 1);
            *dst = b0;
            *dst.add(n / 2) = bm;
            *dst.add(n - 1) = bl;
        }
        if NULLIFY_TAIL {
            write_bytes(dst.add(7), 0, 8);
            write_bytes(dst.add(n), 0, 8);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_layout() {
        // These are relied upon by the flat tag <-> size mapping.
        assert_eq!(core::mem::offset_of!(CordRep, length), 0);
        assert_eq!(core::mem::offset_of!(CordRep, storage), flat::FLAT_OVERHEAD);
        assert!(core::mem::size_of::<CordRep>() < flat::MIN_FLAT_SIZE);
        const { assert!(flat::MIN_FLAT_LENGTH > MAX_INLINE) };
    }

    #[test]
    fn refcount_basics() {
        let rc = Refcount::new();
        assert!(rc.is_one());
        assert_eq!(rc.get(), 1);
        rc.increment();
        assert!(!rc.is_one());
        assert_eq!(rc.get(), 2);
        assert!(rc.decrement());
        assert!(rc.is_one());
        assert!(!rc.decrement());
        let im = Refcount::immortal();
        assert!(im.is_immortal());
        assert!(!im.is_one());
        assert!(im.decrement());
        assert!(im.decrement_expect_high_refcount());
        assert!(im.is_immortal());
    }

    #[test]
    fn small_memmove_all_sizes() {
        let src: Vec<u8> = (1..=15).collect();
        for n in 0..=15 {
            let mut dst = [0xAAu8; 16];
            unsafe { small_memmove::<true>(dst.as_mut_ptr(), src.as_ptr(), n) };
            assert_eq!(&dst[..n], &src[..n], "n={n}");
            assert!(dst[n..15].iter().all(|&b| b == 0), "n={n} tail={:?}", &dst[..]);
            let mut dst2 = [0xAAu8; 16];
            unsafe { small_memmove::<false>(dst2.as_mut_ptr(), src.as_ptr(), n) };
            assert_eq!(&dst2[..n], &src[..n], "n={n}");
        }
        // Overlapping move.
        let mut buf: Vec<u8> = (0..16).collect();
        unsafe { small_memmove::<false>(buf.as_mut_ptr(), buf.as_ptr().add(3), 12) };
        assert_eq!(&buf[..12], &(3..15).collect::<Vec<u8>>()[..]);
    }
}
