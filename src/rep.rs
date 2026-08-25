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

use core::marker::PhantomData;
use core::mem::MaybeUninit;
use core::ptr::NonNull;
use core::sync::atomic::{AtomicI32, Ordering};

use alloc::boxed::Box;

pub(crate) mod analysis;
pub(crate) mod btree;
#[cfg(test)]
mod btree_tests;
#[cfg(test)]
mod data_edge_tests;
pub(crate) mod external;
pub(crate) mod flat;
#[cfg(test)]
mod handle_tests;
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
        usize::try_from(self.0.load(Ordering::Acquire) >> NUM_FLAGS).unwrap_or(0)
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
    // This abort is the guard that prevents a use-after-free on overflow:
    // without it, wrapping the count back through zero would let a live
    // reference outlive the node. `Arc` aborts the same way at the same kind
    // of threshold. Unlike `Arc`, though, a cord can drive the count here
    // quickly through self-sharing alone — structural sharing doubles a
    // node's refcount per operation, so `loop { let d = c.clone();
    // c.append(d); }` reaches this threshold after ~31 iterations of
    // O(log n) work each, not billions of individual clones. The length
    // would need to reach roughly 2^42 bytes of shared structure first, so
    // this is not reachable by accident with real data — but it is
    // reachable, unlike a comparable `Arc` leak loop.
    abort()
}

/// Aborts the process immediately, with no chance of the caller observing an
/// unwind.
///
/// With the `std` feature this is exactly `std::process::abort`. Without
/// it — `no_std` has no portable abort — this uses the double-panic idiom
/// instead: a local guard whose `Drop` panics, then a `panic!`. On targets
/// built with `panic = "abort"` that first `panic!` already aborts and the
/// guard never runs. Otherwise the first `panic!` unwinds and invokes the
/// application's own `#[panic_handler]` — there is no runtime to turn it
/// into an abort in `no_std`, and that handler is expected not to return (it
/// typically halts or resets the device). The guard only matters if that
/// handler instead lets the unwind continue: the panic already in flight
/// while it unwinds then triggers the guard's own `panic!`, a double panic,
/// which is what actually aborts.
#[cold]
#[inline(never)]
fn abort() -> ! {
    #[cfg(feature = "std")]
    {
        std::process::abort()
    }
    #[cfg(not(feature = "std"))]
    {
        struct AbortOnDrop;
        impl Drop for AbortOnDrop {
            fn drop(&mut self) {
                panic!("cord-rs: aborting (double panic)");
            }
        }
        let _guard = AbortOnDrop;
        panic!("cord-rs: aborting");
    }
}

/// Narrows a small value to a `u8`, as stored in the `storage` bytes of reps
/// (values bounded by `MAX_FLAT_TAG`, `btree::MAX_CAPACITY` or `MAX_INLINE`).
#[inline]
pub(crate) const fn small_u8(value: usize) -> u8 {
    debug_assert!(value <= u8::MAX as usize);
    #[expect(clippy::cast_possible_truncation, reason = "bounded by the assertion above")]
    let narrowed = value as u8;
    narrowed
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
/// write through the raw pointer without creating a reference to the whole
/// header, so they are safe to interleave with other raw pointers to the
/// same node. The refcount operations (`ref_inc` and its siblings) are the
/// one partial exception: they do borrow, but the borrow is scoped to just
/// the `refcount` field — never the header as a whole — which is what keeps
/// them safe to interleave too; see `ref_inc`'s own comment for why that
/// field-scoped borrow is sound.
///
/// # Safety
///
/// Every method on this trait requires `self` to be a non-null pointer to a
/// live [`CordRep`] header (directly, or as the header of a derived rep such
/// as a flat, external, substring or btree node) for the duration of the
/// call. Methods that mutate a field (`set_length`) additionally require the
/// caller to hold the rep exclusively (refcount of one) per the [module's
/// reference-counting convention](self) — mutating a shared node is a data
/// race with any other reader.
pub(crate) trait RepPtr: Copy {
    /// Reads `self`'s `length`.
    unsafe fn length(self) -> usize;
    /// Sets `self`'s `length`. Requires exclusive access, see the
    /// [trait-level safety section](RepPtr).
    unsafe fn set_length(self, length: usize);
    /// Reads `self`'s `tag`.
    unsafe fn tag(self) -> u8;
    /// Increments `self`'s refcount. See the [trait-level safety
    /// section](RepPtr).
    unsafe fn ref_inc(self);
    /// Decrements `self`'s refcount; see [`Refcount::decrement`]. See the
    /// [trait-level safety section](RepPtr).
    unsafe fn ref_dec(self) -> bool;
    /// Decrements `self`'s refcount; see
    /// [`Refcount::decrement_expect_high_refcount`]. See the [trait-level
    /// safety section](RepPtr).
    unsafe fn ref_dec_expect_high_refcount(self) -> bool;
    /// Returns `true` if `self`'s refcount is exactly one. See the
    /// [trait-level safety section](RepPtr).
    unsafe fn ref_is_one(self) -> bool;
    /// Returns `true` if `self`'s refcount is immortal. See the
    /// [trait-level safety section](RepPtr).
    unsafe fn ref_is_immortal(self) -> bool;
    /// Reads `self`'s current refcount; see [`Refcount::get`]. See the
    /// [trait-level safety section](RepPtr).
    unsafe fn ref_get(self) -> usize;

    #[inline]
    unsafe fn is_btree(self) -> bool {
        unsafe { self.tag() == BTREE }
    }
    #[inline]
    unsafe fn is_external(self) -> bool {
        unsafe { self.tag() == EXTERNAL }
    }
    #[inline]
    unsafe fn is_flat(self) -> bool {
        unsafe { self.tag() >= FLAT }
    }
}

impl RepPtr for *mut CordRep {
    #[inline]
    unsafe fn length(self) -> usize {
        unsafe { (*self).length }
    }
    #[inline]
    unsafe fn set_length(self, length: usize) {
        unsafe { (*self).length = length }
    }
    #[inline]
    unsafe fn tag(self) -> u8 {
        unsafe { (*self).tag }
    }
    #[inline]
    unsafe fn ref_inc(self) {
        // SAFETY: the atomic's interior mutability makes a shared reference
        // to just the `refcount` field sound to hand to `Refcount::increment`
        // even though other raw pointers may alias the rest of the header;
        // the reference is scoped to this call and never escapes.
        unsafe { &(*self).refcount }.increment();
    }
    #[inline]
    unsafe fn ref_dec(self) -> bool {
        // SAFETY: see `ref_inc`.
        unsafe { &(*self).refcount }.decrement()
    }
    #[inline]
    unsafe fn ref_dec_expect_high_refcount(self) -> bool {
        // SAFETY: see `ref_inc`.
        unsafe { &(*self).refcount }.decrement_expect_high_refcount()
    }
    #[inline]
    unsafe fn ref_is_one(self) -> bool {
        // SAFETY: see `ref_inc`.
        unsafe { &(*self).refcount }.is_one()
    }
    #[inline]
    unsafe fn ref_is_immortal(self) -> bool {
        // SAFETY: see `ref_inc`.
        unsafe { &(*self).refcount }.is_immortal()
    }
    #[inline]
    unsafe fn ref_get(self) -> usize {
        // SAFETY: see `ref_inc`.
        unsafe { &(*self).refcount }.get()
    }
}

/// Debug-only check that `rep` is non-null and non-empty — the adoption
/// contract of [`Cord::from_owned_rep`]. Compiles away in release builds.
/// (`OwnedRep::from_raw` itself adopts any live, well-formed rep and does
/// not require non-emptiness; only `Cord::from_owned_rep`'s narrower
/// contract — a tree is never empty, an empty `Cord` is always inline —
/// does.)
///
/// [`Cord::from_owned_rep`]: crate::cord::Cord::from_owned_rep
///
/// # Safety
///
/// `rep` must point to a live rep (its header is read in debug builds).
#[inline]
pub(crate) unsafe fn debug_assert_nonempty_rep(rep: *mut CordRep) {
    unsafe {
        debug_assert!(!rep.is_null());
        debug_assert!(rep.length() != 0);
    }
}

/// Debug-only check that `rep` is a uniquely-owned flat — the adoption
/// contract of `CordBuffer::from_flat`. Compiles away in release builds.
///
/// # Safety
///
/// Same contract as [`debug_assert_nonempty_rep`].
#[inline]
pub(crate) unsafe fn debug_assert_unique_flat(rep: *mut CordRep) {
    unsafe {
        debug_assert!(!rep.is_null());
        debug_assert!(rep.is_flat() && rep.ref_is_one());
    }
}

/// Increments the reference count of `rep` and returns it.
///
/// # Safety
///
/// `rep` must be a non-null pointer to a live rep. This does not consume an
/// existing reference: the returned pointer is a *new*, additional reference
/// on `rep` (mirrors abseil's `CordRep::Ref`), on top of whatever reference
/// the caller already held.
#[inline]
pub(crate) unsafe fn ref_rep(rep: *mut CordRep) -> *mut CordRep {
    debug_assert!(!rep.is_null());
    unsafe { rep.ref_inc() };
    rep
}

/// Decrements the reference count of `rep`, destroying it on zero.
///
/// # Safety
///
/// `rep` must be a non-null pointer to a live rep, and the caller must be
/// relinquishing exactly one reference it owns on `rep` (this fn *adopts* a
/// reference per the [module convention](self)). The caller must not use
/// `rep` again afterwards unless it independently holds another reference.
#[inline]
pub(crate) unsafe fn unref(rep: *mut CordRep) {
    debug_assert!(!rep.is_null());
    // SAFETY: `rep` is a live rep per this fn's contract. `destroy` is only
    // called once `decrement_expect_high_refcount` reports the count
    // reached zero, at which point this call's adopted reference was the
    // last one outstanding, so `rep` may be freed.
    unsafe {
        if !rep.ref_dec_expect_high_refcount() {
            destroy(rep);
        }
    }
}

/// Destroys `rep`, whose reference count has reached zero.
///
/// # Safety
///
/// `rep` must be a non-null pointer to a live rep whose reference count has
/// just reached zero, transferring final ownership to this function (the
/// caller must not use `rep`, or any substring child reference it may
/// release along the way, again afterwards).
pub(crate) unsafe fn destroy(mut rep: *mut CordRep) {
    // SAFETY: `rep` is a live rep with a refcount of zero per this fn's
    // contract. Each branch below reads `rep.tag()` and then casts/dispatches
    // on it, which is sound because a rep's tag byte always correctly
    // identifies its concrete type by construction (see the module's
    // "Layout" doc) — a BTREE-tagged rep really is a `CordRepBtree`, etc.
    // The `SUBSTRING` branch does not recurse into the child: it decrements
    // the child's refcount (the reference the substring held) and, if that
    // also reaches zero, reassigns `rep` to the child and loops rather than
    // calling `destroy` recursively, so long substring chains can't blow the
    // stack. Every branch either `return`s directly or loops with a `rep`
    // that again satisfies this fn's contract (a live, now-zero-refcount
    // rep owned by this call).
    unsafe {
        loop {
            debug_assert!(!rep.ref_is_immortal());
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
                if rep.ref_dec() {
                    return;
                }
            } else {
                debug_assert!(tag >= FLAT);
                flat::delete(rep);
                return;
            }
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

/// Core substring-node constructor, shared by every substring-creating
/// function in the rep layer ([`CordRepSubstring::create`],
/// [`CordRepSubstring::substring`], [`btree`]'s `make_substring`/
/// `make_substring_from`/`resize_edge`, and [`navigator`]'s `substring`/
/// `substring_from`): builds a `CordRepSubstring` covering `n` bytes starting
/// `offset` bytes into `rep`, following (at most) one level of
/// substring-of-substring flattening exactly like abseil's
/// `CordRepBtree::MakeSubstring` does. This function never applies the
/// `n == 0` / `n == rep.length()` / `offset == 0` shortcuts its callers rely
/// on — it always allocates — so those load-bearing early returns stay in
/// each thin wrapper, at the call site that knows whether they apply.
///
/// If `ADOPT`, the caller donates its reference on `rep`: it is consumed
/// (and, if flattening walks to a child, immediately re-established there
/// via `ref_rep` before the substring wrapper's own reference is released,
/// so the child never dips to zero refs in between) and installed as the new
/// node's `child` with no further `ref_rep`. If `ADOPT` is `false`, the
/// caller keeps its own reference on `rep`; the new node's `child` carries
/// an independently acquired reference (`ref_rep`), taken after flattening.
///
/// # Safety
///
/// `rep` must be a non-null pointer to a live flat, external, or substring
/// node with `n != 0 && offset + n <= rep.length()`. If `ADOPT`, the caller
/// must be transferring exactly one reference it owns on `rep`.
#[expect(
    clippy::inline_always,
    reason = "six formerly straight-line callers; interleaved benching showed a \
              measurable partial-inlining cost with plain #[inline]"
)]
#[inline(always)]
unsafe fn substring_impl<const ADOPT: bool>(
    mut rep: *mut CordRep,
    mut offset: usize,
    n: usize,
) -> *mut CordRep {
    // SAFETY: `rep` is a live rep per this fn's contract. If it is a
    // SUBSTRING, its tag guarantees the cast, and its `child` field is live
    // because the substring holds a reference on it; in the `ADOPT` case we
    // take a fresh reference on `child` *before* releasing the substring
    // wrapper's own reference via `unref`, so `child` stays referenced
    // throughout even if that `unref` frees the wrapper. The final
    // `Box::new` allocation and initialization is ordinary safe code —
    // storing `child`'s pointer value doesn't dereference it.
    unsafe {
        debug_assert!(!rep.is_null());
        debug_assert!(n != 0);
        debug_assert!(offset + n <= rep.length());
        debug_assert!(offset != 0 || n != rep.length());
        if rep.tag() == SUBSTRING {
            let sub: *mut CordRepSubstring = rep.cast();
            offset += (*sub).start;
            if ADOPT {
                let child = ref_rep((*sub).child);
                unref(rep);
                rep = child;
            } else {
                rep = (*sub).child;
            }
        }
        debug_assert!(rep.is_external() || rep.is_flat());
        let child = if ADOPT { rep } else { ref_rep(rep) };
        Box::into_raw(Box::new(CordRepSubstring { rep: CordRep::new(n, SUBSTRING), start: offset, child }))
            .cast()
    }
}

impl CordRepSubstring {
    /// Creates a substring on `child`, adopting a reference on `child`.
    ///
    /// Requires `child` to be a flat or external node and `pos`/`n` to form a
    /// non-empty partial sub range of `child`: `n > 0 && n < child.length &&
    /// pos + n <= child.length`.
    ///
    /// # Safety
    ///
    /// `child` must be a non-null, live flat or external rep, and the caller
    /// must be transferring (adopting away) one reference it owns on `child`
    /// to the newly created substring. `pos`/`n` must satisfy the range
    /// requirement above.
    #[inline]
    pub(crate) unsafe fn create(child: *mut CordRep, pos: usize, n: usize) -> *mut CordRepSubstring {
        unsafe {
            debug_assert!(!child.is_null());
            debug_assert!(n > 0);
            debug_assert!(n < child.length());
            debug_assert!(pos < child.length());
            debug_assert!(n <= child.length() - pos);
            // Entry boundary: `child` has not been spliced into anything
            // yet (the substring node itself is only built by
            // `substring_impl` below), and this fn's sole caller
            // (`__internal::make_substring`) is a top-level entry point, not
            // mid-surgery — unwinding here only leaks the donated
            // reference, so it is safe to leave unwinding.
            assert!(
                child.is_external() || child.is_flat(),
                "cord-rs: unexpected node type {} for substring child",
                child.tag()
            );
            // `child`'s contract already rules out SUBSTRING, so
            // `substring_impl`'s flatten branch is dead code here and this
            // is exactly `create`'s original unconditional
            // `Box::into_raw(Box::new(CordRepSubstring { .. }))`.
            substring_impl::<true>(child, pos, n).cast()
        }
    }

    /// Creates a substring of `rep` **without** adopting a reference on `rep`.
    ///
    /// Requires `is_data_edge(rep) && n > 0 && pos + n <= rep.length`. If
    /// `n == rep.length` this returns `ref_rep(rep)`. If `rep` is itself a
    /// substring, the returned substring references its child with `pos`
    /// adjusted by the original `start`.
    ///
    /// # Safety
    ///
    /// `rep` must be a non-null pointer to a live rep for which
    /// `is_data_edge(rep)` holds, and `pos`/`n` must satisfy the range
    /// requirement above. Unlike [`create`](Self::create), this does *not*
    /// consume a reference on `rep`: the caller keeps whatever reference it
    /// already held, and the new substring (or the `ref_rep(rep)` result)
    /// carries its own, independently acquired reference.
    #[inline]
    pub(crate) unsafe fn substring(rep: *mut CordRep, pos: usize, n: usize) -> *mut CordRep {
        unsafe {
            debug_assert!(!rep.is_null());
            debug_assert!(n != 0);
            debug_assert!(pos < rep.length());
            debug_assert!(n <= rep.length() - pos);
            if n == rep.length() {
                return ref_rep(rep);
            }
            substring_impl::<false>(rep, pos, n)
        }
    }

    /// Frees the substring node itself (not its child).
    ///
    /// # Safety
    ///
    /// `substring` must be a pointer originally produced by
    /// [`create`](Self::create) or [`substring`](Self::substring) (i.e. a
    /// live, exclusively-owned `CordRepSubstring` obtained from
    /// `Box::into_raw`), and this call takes ownership of it: `substring`
    /// must not be used again afterwards. The referenced `child` is left
    /// untouched — its reference must be released separately by the caller.
    #[inline]
    pub(crate) unsafe fn delete(substring: *mut CordRepSubstring) {
        // SAFETY: `substring` is a live, uniquely-owned box pointer per this
        // fn's contract, so reconstructing and dropping the `Box` is sound
        // and frees exactly that allocation.
        unsafe { drop(Box::from_raw(substring)) };
    }
}

// --- Data edges ------------------------------------------------------------

/// Returns `true` if `edge` is a FLAT, EXTERNAL or a SUBSTRING of a FLAT or
/// EXTERNAL node.
///
/// # Safety
///
/// `edge` must be a non-null pointer to a live rep.
#[inline]
pub(crate) unsafe fn is_data_edge(mut edge: *const CordRep) -> bool {
    // SAFETY: `edge` is a live rep per this fn's contract, so its header may
    // be read directly. If its tag is SUBSTRING, the tag itself guarantees
    // the pointer really is a `CordRepSubstring`, whose `child` field is in
    // turn always a live rep (the substring holds a reference on it) — so
    // following it and reading its tag is likewise sound.
    unsafe {
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
}

/// Returns the bytes referenced by the data edge `edge`.
///
/// Requires `is_data_edge(edge)`. The returned slice borrows the node's
/// memory: the caller must ensure the edge outlives `'a` and is not mutated
/// (i.e. it is held by a live cord and not being appended to in place).
///
/// # Safety
///
/// `edge` must be a non-null pointer to a live rep for which
/// `is_data_edge(edge)` holds, and the referenced bytes must remain valid
/// and unmutated for the lifetime `'a` of the returned slice.
#[inline]
pub(crate) unsafe fn edge_data<'a>(mut edge: *const CordRep) -> &'a [u8] {
    // SAFETY: `edge` is a live data edge per this fn's contract. Following a
    // SUBSTRING to its `child` is sound for the same reason as in
    // `is_data_edge` (the tag guarantees the cast, the substring holds a
    // reference on `child`). Dispatching on the (possibly substring-derefed)
    // tag to read either a flat's payload or an external's `base` is sound
    // because the tag correctly identifies the concrete type. The resulting
    // pointer, `offset` and `length` together describe exactly the `length`
    // bytes this data edge represents, which the caller's contract keeps
    // valid for `'a`.
    unsafe {
        debug_assert!(is_data_edge(edge));
        let mut offset = 0;
        let length = (*edge).length;
        if (*edge).tag == SUBSTRING {
            let sub = edge.cast::<CordRepSubstring>();
            offset = (*sub).start;
            edge = (*sub).child;
        }
        let base = if (*edge).tag >= FLAT {
            flat::data(edge.cast_mut()).cast_const()
        } else {
            (*edge.cast::<external::CordRepExternal>()).base
        };
        core::slice::from_raw_parts(base.add(offset), length)
    }
}

// --- Typed handles -----------------------------------------------------

/// Copy handle borrowing a live rep for `'a`.
///
/// Replaces ad hoc `*mut CordRep` + [`RepPtr`] call sites with a type that
/// carries its liveness invariant once, at construction, instead of at every
/// call site. `self`'s pointer is never turned into a `&CordRep` (a
/// whole-header reference); every method below reads through the raw
/// pointer or scopes a borrow to just the (interior-mutable) refcount field,
/// exactly like [`RepPtr`]'s impl for `*mut CordRep` already does.
///
/// # Invariant
///
/// The wrapped pointer is non-null and points to a live, well-formed rep
/// (directly, or as the header of a derived flat, external, substring or
/// btree node; see the [module doc](self)) that is not mutated — other than
/// through its interior-mutable refcount — for the duration of `'a` (this
/// is what lets [`data`](Self::data) hand out a `&'a [u8]`).
/// Established once, at the sole constructor [`from_raw`](Self::from_raw);
/// every other method on this type is safe because it needs nothing more
/// than this invariant.
#[derive(Clone, Copy)]
pub(crate) struct RepRef<'a> {
    ptr: NonNull<CordRep>,
    _marker: PhantomData<&'a CordRep>,
}

impl<'a> RepRef<'a> {
    /// Wraps `ptr` as a handle borrowed for `'a`.
    ///
    /// # Safety
    ///
    /// `ptr` must be non-null and point to a live, well-formed rep (directly,
    /// or as the header of a derived flat, external, substring or btree
    /// node). The caller must guarantee — by holding, or borrowing, a
    /// reference on it — that it stays live, and is not mutated other than
    /// through its interior-mutable refcount, for `'a`.
    #[inline]
    pub(crate) unsafe fn from_raw(ptr: *mut CordRep) -> Self {
        debug_assert!(!ptr.is_null());
        // SAFETY: non-null per the debug_assert above (and per this fn's own
        // precondition on `ptr` in release builds).
        Self { ptr: unsafe { NonNull::new_unchecked(ptr) }, _marker: PhantomData }
    }

    /// This handle's `length`.
    #[inline]
    pub(crate) fn len(self) -> usize {
        // SAFETY: `self`'s invariant (struct doc) guarantees `self.ptr` is a
        // live rep for the call's duration.
        unsafe { self.ptr.as_ptr().length() }
    }

    /// This handle's tag byte.
    #[inline]
    pub(crate) fn tag(self) -> u8 {
        // SAFETY: see `len`.
        unsafe { self.ptr.as_ptr().tag() }
    }

    /// `true` if this handle is a [`CordRepSubstring`].
    #[inline]
    pub(crate) fn is_substring(self) -> bool {
        self.tag() == SUBSTRING
    }

    /// `true` if this handle is a [`btree::CordRepBtree`].
    #[inline]
    pub(crate) fn is_btree(self) -> bool {
        self.tag() == BTREE
    }

    /// `true` if this handle is a [`external::CordRepExternal`].
    #[inline]
    pub(crate) fn is_external(self) -> bool {
        self.tag() == EXTERNAL
    }

    /// `true` if this handle is a flat node.
    #[inline]
    pub(crate) fn is_flat(self) -> bool {
        self.tag() >= FLAT
    }

    /// `true` if this handle is a data edge: a flat, external, or substring
    /// of one. See [`is_data_edge`].
    #[inline]
    pub(crate) fn is_data_edge(self) -> bool {
        // SAFETY: see `len`.
        unsafe { is_data_edge(self.ptr.as_ptr()) }
    }

    /// `true` if this handle's reference count is exactly one.
    #[inline]
    pub(crate) fn ref_is_one(self) -> bool {
        // SAFETY: see `len`.
        unsafe { self.ptr.as_ptr().ref_is_one() }
    }

    /// This handle's current reference count.
    #[inline]
    pub(crate) fn ref_get(self) -> usize {
        // SAFETY: see `len`.
        unsafe { self.ptr.as_ptr().ref_get() }
    }

    /// The bytes referenced by this data edge.
    ///
    /// The bounded-lifetime replacement for the free function [`edge_data`],
    /// whose returned `&'a [u8]` has an inferred, unconstrained lifetime at
    /// its call sites.
    ///
    /// # Safety
    ///
    /// [`self.is_data_edge()`](Self::is_data_edge) must hold: on a BTREE
    /// handle, the underlying [`edge_data`] would reinterpret `edges[0]` as
    /// an external node's `base` pointer, which is real UB, not just a wrong
    /// answer.
    #[inline]
    pub(crate) unsafe fn data(self) -> &'a [u8] {
        debug_assert!(self.is_data_edge());
        // SAFETY: `self`'s invariant makes `self.ptr` a live rep for `'a`;
        // `self.is_data_edge()` holds per this fn's precondition, and the
        // same invariant keeps the referenced bytes valid and unmutated for
        // `'a`.
        unsafe { edge_data(self.ptr.as_ptr()) }
    }

    /// This handle as a [`btree::BtreeRef`], without a checked dispatch.
    /// The debug-asserted unchecked counterpart of matching
    /// [`view`](Self::view) for `RepView::Btree` — for hot descents where
    /// btree well-formedness (every non-leaf edge is a btree node) already
    /// guarantees the kind and a checked match would add a dead branch.
    ///
    /// # Safety
    ///
    /// `self.is_btree()` must hold.
    #[inline]
    pub(crate) unsafe fn btree_unchecked(self) -> btree::BtreeRef<'a> {
        debug_assert!(self.is_btree());
        // SAFETY: tag == BTREE per this fn's precondition, so the cast is
        // sound; liveness for `'a` carries over from `self`'s invariant.
        unsafe { btree::BtreeRef::from_raw(self.ptr.as_ptr().cast()) }
    }

    /// This handle as a substring without checked dispatch.
    ///
    /// # Safety
    ///
    /// `self.is_substring()` must hold.
    #[inline]
    pub(crate) unsafe fn substring_unchecked(self) -> (usize, RepRef<'a>) {
        debug_assert!(self.is_substring());
        let sub = self.ptr.as_ptr().cast::<CordRepSubstring>();
        // SAFETY: tag == SUBSTRING per this fn's precondition, so the cast is
        // sound; the substring's reference keeps `child` live for `'a`.
        let (start, child) = unsafe { ((*sub).start, (*sub).child) };
        // SAFETY: a well-formed substring owns a non-null, live child.
        (start, unsafe { RepRef::from_raw(child) })
    }

    /// The checked, typed view of this handle: one tag read, then a
    /// dispatch to the concrete node kind (mirrors [`destroy`]'s dispatch).
    #[inline]
    pub(crate) fn view(self) -> RepView<'a> {
        let ptr = self.ptr.as_ptr();
        match self.tag() {
            // SAFETY: tag == BTREE guarantees this cast is sound (the
            // module's tag invariant); `self`'s own invariant keeps `ptr`
            // live and unmutated for exactly `'a`.
            BTREE => RepView::Btree(unsafe { btree::BtreeRef::from_raw(ptr.cast()) }),
            SUBSTRING => {
                // SAFETY: tag == SUBSTRING in this match arm.
                let (start, child) = unsafe { self.substring_unchecked() };
                RepView::Substring { start, child }
            }
            // SAFETY: tag == EXTERNAL guarantees this cast is sound (the
            // module's tag invariant); see the BTREE arm above for liveness.
            EXTERNAL => RepView::External(unsafe { external::ExternalRef::from_raw(ptr.cast()) }),
            // SAFETY: tag >= FLAT is exactly `FlatRef::from_raw`'s
            // precondition (the module's tag invariant); see the BTREE arm
            // above for liveness.
            tag if tag >= FLAT => RepView::Flat(unsafe { flat::FlatRef::from_raw(ptr) }),
            tag => unreachable!("cord-rs: unexpected rep tag {tag}"),
        }
    }

    /// Takes a fresh, owned reference on this handle's rep.
    ///
    /// Safe because `self`'s invariant already guarantees liveness, which is
    /// exactly [`ref_inc`](RepPtr::ref_inc)'s precondition.
    #[inline]
    pub(crate) fn to_owned(self) -> OwnedRep {
        let ptr = self.ptr.as_ptr();
        // SAFETY: `self`'s invariant makes `ptr` live; incrementing its
        // count and adopting the new reference into `OwnedRep` is sound.
        unsafe {
            ptr.ref_inc();
            OwnedRep::from_raw(ptr)
        }
    }

    /// Escape hatch to the raw pointer: a permanent, intentional interop
    /// point with the raw surgery layer (deep btree operations, `lib.rs`'s
    /// inspection hooks, and other code that works directly on `*mut
    /// CordRep`/[`RepPtr`] by design), not a stopgap pending conversion to
    /// the handle types.
    #[inline]
    pub(crate) fn as_ptr(self) -> *mut CordRep {
        self.ptr.as_ptr()
    }
}

// SAFETY: mirrors `OwnedRep`'s own `unsafe impl Send`/`Sync` below: a
// `RepRef` only ever reads through the node it borrows (or scopes a borrow
// to the interior-mutable refcount, exactly like `RepPtr`'s impl), and its
// invariant (struct doc) already requires the node to be live and
// unmutated (other than through that refcount) for `'a` — precisely the
// condition under which sharing a read-only view across threads is sound,
// the same way `&CordRep` would be if `CordRep` itself were `Sync` (its
// fields are all plain, `Sync` data). Needed so types that hold a `RepRef`
// field (e.g. `Chunks` in iter.rs) can derive `Send`/`Sync` instead of
// asserting them manually.
unsafe impl Send for RepRef<'_> {}
// SAFETY: see above.
unsafe impl Sync for RepRef<'_> {}

/// RAII owner of exactly one reference on a rep.
///
/// [`Drop`] unrefs it; [`into_raw`](Self::into_raw) transfers the owned
/// reference back out to the crate's raw-pointer adopt/transfer convention
/// (see the [module doc](self)) for code not yet converted to this type.
pub(crate) struct OwnedRep {
    ptr: NonNull<CordRep>,
}

impl OwnedRep {
    /// Adopts one reference on `ptr`.
    ///
    /// # Safety
    ///
    /// `ptr` must be non-null and point to a live, well-formed rep, and the
    /// caller must be transferring (adopting away) exactly one reference it
    /// owns on `ptr` to the returned `OwnedRep`, per the [module's
    /// reference-counting convention](self).
    #[inline]
    pub(crate) unsafe fn from_raw(ptr: *mut CordRep) -> Self {
        debug_assert!(!ptr.is_null());
        // SAFETY: non-null per the debug_assert above.
        Self { ptr: unsafe { NonNull::new_unchecked(ptr) } }
    }

    /// Transfers the owned reference back out as a raw pointer, per the
    /// crate's adopt/transfer convention: the caller becomes responsible for
    /// eventually releasing it (via [`unref`] or another `OwnedRep`).
    #[inline]
    pub(crate) fn into_raw(self) -> *mut CordRep {
        // `ManuallyDrop` suppresses `Drop::drop` (which would unref the very
        // reference this fn is transferring out) while still moving out of
        // `self`.
        let this = core::mem::ManuallyDrop::new(self);
        this.ptr.as_ptr()
    }

    /// Borrows this owned reference as a [`RepRef`] tied to the borrow.
    #[inline]
    pub(crate) fn as_ref(&self) -> RepRef<'_> {
        // SAFETY: `self` owns a live reference on `self.ptr`, which outlives
        // the `'_` borrow of `self` taken here.
        unsafe { RepRef::from_raw(self.ptr.as_ptr()) }
    }

    /// This rep's `length`.
    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.as_ref().len()
    }

    /// Attempts to obtain a mutation witness for this owned rep: `Some` iff
    /// its reference count is exactly one, i.e. `self` is the only
    /// outstanding reference anywhere. See [`UniqueRep`]'s soundness note
    /// for why a `&mut self` path like this one is the only sound way to
    /// construct one.
    #[inline]
    pub(crate) fn try_unique(&mut self) -> Option<UniqueRep<'_>> {
        if self.as_ref().ref_is_one() {
            // SAFETY: `self` owns the sole reference (struct invariant) and
            // `ref_is_one()` just confirmed no other reference exists
            // either, so this `&mut self` borrow is the only live handle to
            // the node anywhere, for as long as the borrow lasts.
            Some(unsafe { UniqueRep::from_raw(self.ptr.as_ptr()) })
        } else {
            None
        }
    }
}

impl Drop for OwnedRep {
    #[inline]
    fn drop(&mut self) {
        // SAFETY: `self` owns exactly one reference on `self.ptr` (struct
        // invariant), which this call relinquishes exactly once.
        unsafe { unref(self.ptr.as_ptr()) };
    }
}

impl Clone for OwnedRep {
    #[inline]
    fn clone(&self) -> Self {
        let ptr = self.ptr.as_ptr();
        // SAFETY: `self` proves `ptr` is live, so incrementing its count and
        // adopting the fresh reference into a new `OwnedRep` is sound.
        unsafe {
            ptr.ref_inc();
            Self::from_raw(ptr)
        }
    }
}

// SAFETY: mirrors `Cord`'s own `unsafe impl Send`/`Sync` (see cord.rs):
// nodes shared between cords/handles are immutable, reference counts are
// atomic, and external owners are required to be `Send + Sync`.
unsafe impl Send for OwnedRep {}
// SAFETY: see `Send` above.
unsafe impl Sync for OwnedRep {}

/// Checked, typed view of a rep: the result of [`RepRef::view`].
pub(crate) enum RepView<'a> {
    /// A [`btree::CordRepBtree`] node.
    Btree(btree::BtreeRef<'a>),
    /// A [`CordRepSubstring`].
    Substring {
        /// Starting offset of the substring inside `child`.
        start: usize,
        /// The referenced flat or external node.
        child: RepRef<'a>,
    },
    /// A [`external::CordRepExternal`] node.
    External(external::ExternalRef<'a>),
    /// A flat node.
    Flat(flat::FlatRef<'a>),
}

/// Refcount-one mutation witness: proof that, for `'a`, this call is the
/// *only* live handle to the wrapped rep, so it may be mutated in place.
///
/// # Soundness
///
/// The sole constructor, [`from_raw`](Self::from_raw), is a crate-private
/// `unsafe fn` documented to be called *only* from
/// [`OwnedRep::try_unique`], [`crate::inline_data::InlineData::tree_unique`]
/// and [`crate::buffer::CordBuffer`]'s internal `Rep::view_mut` (see that
/// type's own doc) — the first two take `&mut self` and check
/// [`ref_is_one`](RepRef::ref_is_one) themselves right before constructing;
/// that `&mut` borrow is what makes the combination sound: it proves no
/// *other* copy of the owning slot (the `OwnedRep` value, or the
/// `Cord`/`InlineData` it lives in) can be read or written for as long as
/// the resulting `UniqueRep` exists, so together with `ref_is_one()` this
/// really is the only handle to the node anywhere. Minting a `UniqueRep`
/// from a `Copy` [`RepRef`] instead would break this: two independent
/// copies of the same `RepRef` could each separately observe
/// `ref_is_one()` and each construct a `UniqueRep`, yielding two
/// "exclusive" mutable views of the same node at once. `CordBuffer`'s call
/// site proves the same thing a different way: it never checks
/// `ref_is_one()` dynamically, because its flat rep, whenever present, is
/// *unconditionally* exclusively owned for the buffer's entire lifetime (by
/// construction: only [`CordBuffer::from_flat`] creates one, which requires
/// this, and `CordBuffer` is not `Clone` and never exposes the pointer
/// while retaining ownership) — so its own `&mut self` borrow is, if
/// anything, a strictly stronger proof than the dynamic check. No other
/// code in the crate may call `from_raw`.
pub(crate) struct UniqueRep<'a> {
    ptr: NonNull<CordRep>,
    _marker: PhantomData<&'a mut CordRep>,
}

impl<'a> UniqueRep<'a> {
    /// # Safety
    ///
    /// `ptr` must be non-null and point to a live, well-formed rep with
    /// `ref_is_one()` true, and the caller must additionally guarantee,
    /// via a `&mut` borrow on the sole owner of this reference, that no
    /// other handle to it exists or is created for `'a`. See the
    /// [type-level soundness note](Self) for why only three call sites in
    /// the crate may use this.
    #[inline]
    pub(crate) unsafe fn from_raw(ptr: *mut CordRep) -> Self {
        debug_assert!(!ptr.is_null());
        // SAFETY: non-null per the debug_assert above.
        Self { ptr: unsafe { NonNull::new_unchecked(ptr) }, _marker: PhantomData }
    }

    /// This handle reinterpreted as a read-only [`RepRef`], borrowed from
    /// `self` rather than tied to `'a` (so it cannot outlive further
    /// mutation through `self`).
    #[inline]
    pub(crate) fn as_ref(&self) -> RepRef<'_> {
        // SAFETY: `self`'s invariant guarantees `self.ptr` is live and
        // touched only through this borrow for the returned handle's
        // lifetime.
        unsafe { RepRef::from_raw(self.ptr.as_ptr()) }
    }

    /// Escape hatch to the raw pointer: a permanent, intentional interop
    /// point with the raw surgery layer (e.g. deep btree surgery, which
    /// stays raw by design), not a stopgap pending conversion to the handle
    /// types.
    #[inline]
    pub(crate) fn as_ptr(&self) -> *mut CordRep {
        self.ptr.as_ptr()
    }

    /// Sets this rep's `length`. Sound because `self`'s invariant is
    /// exclusive access.
    #[inline]
    pub(crate) fn set_len(&mut self, len: usize) {
        // SAFETY: exclusive access per `self`'s invariant.
        unsafe { self.ptr.as_ptr().set_length(len) };
    }

    /// The writable region of a flat's payload, from the current `length`
    /// up to `capacity`, as `MaybeUninit` (bytes past `length` were never
    /// written as `u8`).
    ///
    /// # Safety
    ///
    /// This handle's rep must be a flat node (tag in `FLAT..=MAX_FLAT_TAG`):
    /// on a SUBSTRING, `flat::capacity` would decode the substring's `start`
    /// field as a capacity byte, which can underflow `capacity - len` into a
    /// huge slice length.
    #[inline]
    pub(crate) unsafe fn flat_spare_capacity_mut(&mut self) -> &mut [MaybeUninit<u8>] {
        debug_assert!(self.as_ref().is_flat());
        let ptr = self.ptr.as_ptr();
        // SAFETY: exclusive access per `self`'s invariant makes it sound to
        // hand out a mutable view of the region beyond `length` (never read
        // back as initialized `u8`) up to `capacity`; `self` is a flat per
        // this fn's precondition, so the pointer derived from its own
        // allocation via `flat::data` is valid for reads/writes over the
        // whole capacity per its own contract.
        unsafe {
            let len = ptr.length();
            let capacity = flat::capacity(ptr);
            let spare = flat::data(ptr).add(len).cast::<MaybeUninit<u8>>();
            core::slice::from_raw_parts_mut(spare, capacity - len)
        }
    }

    /// The initialized payload of a flat, mutably (`length()` bytes), tied
    /// to `'a` rather than to this call's own borrow — unlike
    /// [`flat_spare_capacity_mut`](Self::flat_spare_capacity_mut), this
    /// *consumes* `self` (it is not `Copy`) precisely so the returned slice
    /// can outlive the call: giving up the witness in exchange for the
    /// slice means there is no way to mint a second live view through it
    /// afterward.
    ///
    /// # Safety
    ///
    /// This handle's rep must be a flat node (tag in `FLAT..=MAX_FLAT_TAG`);
    /// see [`flat_spare_capacity_mut`](Self::flat_spare_capacity_mut) for
    /// why a non-flat tag is real UB, not just a wrong answer.
    #[inline]
    pub(crate) unsafe fn flat_data_mut(self) -> &'a mut [u8] {
        debug_assert!(self.as_ref().is_flat());
        let ptr = self.ptr.as_ptr();
        // SAFETY: exclusive access per `self`'s invariant, consumed by this
        // call, makes it sound to hand out a mutable view of the
        // initialized payload for the full `'a`; `self` is a flat per this
        // fn's precondition, so the pointer derived from its own allocation
        // via `flat::data` stays within the bounds `flat::data`'s own
        // contract allows writes over (a flat's `length` never exceeds its
        // capacity).
        unsafe {
            let len = ptr.length();
            core::slice::from_raw_parts_mut(flat::data(ptr), len)
        }
    }

    /// The [`flat_spare_capacity_mut`](Self::flat_spare_capacity_mut)
    /// region, tied to `'a` rather than to this call's own borrow —
    /// consumes `self`, for the same reason and with the same soundness
    /// argument as [`flat_data_mut`](Self::flat_data_mut).
    ///
    /// # Safety
    ///
    /// Same as [`flat_spare_capacity_mut`](Self::flat_spare_capacity_mut):
    /// this handle's rep must be a flat node.
    #[inline]
    pub(crate) unsafe fn into_flat_spare_capacity_mut(self) -> &'a mut [MaybeUninit<u8>] {
        debug_assert!(self.as_ref().is_flat());
        let ptr = self.ptr.as_ptr();
        // SAFETY: see `flat_spare_capacity_mut`, with exclusivity carried
        // for the full `'a` because this call consumes `self`, the same
        // way `flat_data_mut` above does.
        unsafe {
            let len = ptr.length();
            let capacity = flat::capacity(ptr);
            let spare = flat::data(ptr).add(len).cast::<MaybeUninit<u8>>();
            core::slice::from_raw_parts_mut(spare, capacity - len)
        }
    }

    /// Mutable access to a [`CordRepSubstring`]'s `start` field.
    ///
    /// # Safety
    ///
    /// This handle's rep must be a substring (tag == SUBSTRING): the cast
    /// below reinterprets it as a `CordRepSubstring`, which is real UB for
    /// any other tag.
    #[inline]
    pub(crate) unsafe fn substring_start_mut(&mut self) -> &mut usize {
        debug_assert!(self.as_ref().is_substring());
        let sub: *mut CordRepSubstring = self.ptr.as_ptr().cast();
        // SAFETY: tag == SUBSTRING per this fn's precondition guarantees the
        // cast is sound (the module's tag invariant); exclusive access per
        // `self`'s invariant makes the `&mut` sound.
        unsafe { &mut (*sub).start }
    }
}

// --- Small memmove ---------------------------------------------------------

/// Fast `memmove` for up to 15 bytes, safe for overlapping regions.
///
/// If `NULLIFY_TAIL` is true the destination is zero padded up to 15 bytes
/// (so `dst` must have room for 15 bytes regardless of `n`).
///
/// # Safety
///
/// - `n` must be at most 15.
/// - `src` must be valid for reads of `n` bytes.
/// - `dst` must be valid for writes of `n` bytes, or of 15 bytes if
///   `NULLIFY_TAIL` is true (regardless of `n`).
/// - `src` and `dst` may overlap arbitrarily: each branch reads both ends of
///   its `n`-byte range into locals before writing anything, so it behaves
///   like `memmove`, not `memcpy`.
#[inline]
pub(crate) unsafe fn small_memmove<const NULLIFY_TAIL: bool>(dst: *mut u8, src: *const u8, n: usize) {
    use core::ptr::{read_unaligned, write_bytes, write_unaligned};
    // SAFETY: `n <= 15` and `src`/`dst` are valid for `n` (or, with
    // `NULLIFY_TAIL`, up to 15) bytes, per this fn's contract. Each branch
    // below only touches offsets within that range (e.g. `src.add(n - 8)` /
    // `dst.add(n - 8)` in the `n >= 8` branch stay `<= src + 14` / `dst +
    // 14`), and every read of `src` is captured into a local (`buf1`,
    // `buf2`, or `b0`/`bm`/`bl`) before any write to `dst`, so the copy is
    // correct even when `src` and `dst` overlap.
    unsafe {
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
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

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
        // SAFETY: `n = 12 <= 15`; `src = buf.as_ptr().add(3)` is valid for 12
        // reads (`buf` has 16 bytes, offset 3 leaves 13); `dst = buf.as_mut_ptr()`
        // is valid for 12 writes into the same 16-byte `buf`.
        unsafe { small_memmove::<false>(buf.as_mut_ptr(), buf.as_ptr().add(3), 12) };
        assert_eq!(&buf[..12], &(3..15).collect::<Vec<u8>>()[..]);
    }
}
