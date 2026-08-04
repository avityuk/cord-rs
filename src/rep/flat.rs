//! Flat reps: a [`CordRep`] header immediately followed by its payload.
//!
//! Flats are allocated at one of a fixed set of sizes so that the size can be
//! stored in the one byte `tag`. The granularity is:
//!
//! * 8 byte steps for allocation sizes in `[32, 512]`
//! * 64 byte steps for sizes in `(512, 8 KiB]`
//! * 4 KiB steps for sizes in `(8 KiB, 256 KiB]`
//!
//! Port of abseil's `cord_rep_flat.h`.

use core::alloc::Layout;
use core::marker::PhantomData;
use core::mem::{align_of, offset_of};
use core::ptr::NonNull;

use super::{CordRep, FLAT, MAX_FLAT_TAG, RepPtr, small_u8};

/// Size of the header preceding the payload of a flat.
pub(crate) const FLAT_OVERHEAD: usize = offset_of!(CordRep, storage);
/// Smallest flat allocation size.
pub(crate) const MIN_FLAT_SIZE: usize = 32;
/// Largest flat allocation size for "default" flats.
pub(crate) const MAX_FLAT_SIZE: usize = 4096;
/// Largest payload of a default flat.
pub(crate) const MAX_FLAT_LENGTH: usize = MAX_FLAT_SIZE - FLAT_OVERHEAD;
/// Smallest payload of a flat.
pub(crate) const MIN_FLAT_LENGTH: usize = MIN_FLAT_SIZE - FLAT_OVERHEAD;
/// Largest flat allocation size for explicitly "large" flats.
pub(crate) const MAX_LARGE_FLAT_SIZE: usize = 256 * 1024;
/// Largest payload of a large flat.
#[allow(dead_code)]
pub(crate) const MAX_LARGE_FLAT_LENGTH: usize = MAX_LARGE_FLAT_SIZE - FLAT_OVERHEAD;

/// Makes the size <-> tag mapping resilient against changes to `FLAT`.
const TAG_BASE: usize = (FLAT - 4) as usize;

/// Converts an exactly representable allocation size to its tag.
pub(crate) const fn allocated_size_to_tag_unchecked(size: usize) -> u8 {
    let tag = if size <= 512 {
        TAG_BASE + size / 8
    } else if size <= 8192 {
        TAG_BASE + 512 / 8 + size / 64 - 512 / 64
    } else {
        TAG_BASE + 512 / 8 + ((8192 - 512) / 64) + size / 4096 - 8192 / 4096
    };
    small_u8(tag)
}

/// Converts a tag to the corresponding allocated size.
pub(crate) const fn tag_to_allocated_size(tag: u8) -> usize {
    let tag = tag as usize;
    if tag <= TAG_BASE + 512 / 8 {
        tag * 8 - TAG_BASE * 8
    } else if tag <= TAG_BASE + (512 / 8) + ((8192 - 512) / 64) {
        512 + tag * 64 - TAG_BASE * 64 - 512 / 8 * 64
    } else {
        8192 + tag * 4096 - TAG_BASE * 4096 - ((512 / 8) + ((8192 - 512) / 64)) * 4096
    }
}

const _: () = assert!(allocated_size_to_tag_unchecked(MIN_FLAT_SIZE) == FLAT);
const _: () = assert!(allocated_size_to_tag_unchecked(MAX_LARGE_FLAT_SIZE) == MAX_FLAT_TAG);
const _: () = assert!(tag_to_allocated_size(MAX_FLAT_TAG) == MAX_LARGE_FLAT_SIZE, "Bad tag logic");

/// Rounds `n` up to the nearest multiple of the power of two `m`.
#[inline]
pub(crate) const fn round_up(n: usize, m: usize) -> usize {
    (n + m - 1) & (0usize.wrapping_sub(m))
}

/// Rounds `size` up to the nearest size that can be expressed exactly as a
/// tag value.
#[inline]
pub(crate) fn round_up_for_tag(size: usize) -> usize {
    round_up(
        size,
        if size <= 512 {
            8
        } else if size <= 8192 {
            64
        } else {
            4096
        },
    )
}

/// Converts an allocated size to a tag, rounding down if the size is not
/// exactly representable. Requires `size <= MAX_LARGE_FLAT_SIZE`.
#[inline]
pub(crate) fn allocated_size_to_tag(size: usize) -> u8 {
    let tag = allocated_size_to_tag_unchecked(size);
    debug_assert!(tag <= MAX_FLAT_TAG);
    tag
}

/// Converts a tag to the payload capacity it encodes.
#[inline]
pub(crate) const fn tag_to_length(tag: u8) -> usize {
    tag_to_allocated_size(tag) - FLAT_OVERHEAD
}

#[inline]
fn layout_for(size: usize) -> Layout {
    // `align_of::<CordRep>()` is a compile-time power of two and `size` is
    // bounded by `MAX_LARGE_FLAT_SIZE`, so this can never actually fail; the
    // formatting-free cold arm keeps the caller's inlining unaffected.
    match Layout::from_size_align(size, align_of::<CordRep>()) {
        Ok(layout) => layout,
        Err(_) => unreachable!(),
    }
}

/// Allocates a flat with a payload capacity of at least
/// `min(len, MAX_SIZE - FLAT_OVERHEAD)` bytes (and at least
/// `MIN_FLAT_LENGTH`), with the header initialized to length `0` and
/// refcount one. `MAX_SIZE` is only ever instantiated as [`MAX_FLAT_SIZE`]
/// or [`MAX_LARGE_FLAT_SIZE`] (by [`new`] / [`new_large`] below).
///
/// Ownership obligation on the result (not a precondition of calling): the
/// returned `CordRep` is newly allocated with a refcount of one, and the
/// caller becomes responsible for eventually releasing it via [`delete`] or
/// [`super::unref`] exactly once — not doing so merely leaks memory.
#[inline]
#[expect(clippy::cast_ptr_alignment, reason = "the layout requests align_of::<CordRep>()")]
fn new_impl<const MAX_SIZE: usize>(mut len: usize) -> *mut CordRep {
    if len <= MIN_FLAT_LENGTH {
        len = MIN_FLAT_LENGTH;
    } else if len > MAX_SIZE - FLAT_OVERHEAD {
        len = MAX_SIZE - FLAT_OVERHEAD;
    }
    // Round size up so it matches a size we can exactly express in a tag.
    let size = round_up_for_tag(len + FLAT_OVERHEAD);
    let layout = layout_for(size);
    // SAFETY: `layout` was produced by `layout_for`, which guarantees a
    // non-zero size (>= MIN_FLAT_SIZE) and an alignment matching
    // `align_of::<CordRep>()`, so `alloc` may be called with it; the
    // returned block (once checked non-null) is immediately initialized
    // with a full `CordRep` header before any other code can observe it.
    unsafe {
        let raw = std::alloc::alloc(layout);
        let raw = NonNull::new(raw).unwrap_or_else(|| std::alloc::handle_alloc_error(layout));
        let rep = raw.as_ptr().cast::<CordRep>();
        rep.write(CordRep::new(0, allocated_size_to_tag(size)));
        rep
    }
}

/// Allocates a new flat with a capacity of at least `min(len, MAX_FLAT_LENGTH)`
/// bytes (and at least `MIN_FLAT_LENGTH`). The returned flat has `length == 0`.
///
/// Carries the same ownership obligation as [`new_impl`].
#[inline]
pub(crate) fn new(len: usize) -> *mut CordRep {
    new_impl::<MAX_FLAT_SIZE>(len)
}

/// Like [`new`] but allows capacities up to `MAX_LARGE_FLAT_LENGTH`.
///
/// Carries the same ownership obligation as [`new_impl`].
#[inline]
pub(crate) fn new_large(len: usize) -> *mut CordRep {
    new_impl::<MAX_LARGE_FLAT_SIZE>(len)
}

/// Deallocates a flat created by [`new`] / [`new_large`].
///
/// # Safety
///
/// `rep` must be a non-null pointer to a live flat rep (tag in
/// `FLAT..=MAX_FLAT_TAG`, i.e. originally returned by [`new`], [`new_large`]
/// or [`create`]) whose reference count has just reached zero, transferring
/// final ownership to this call; `rep` must not be used again afterwards.
/// The memory is freed using the [`Layout`] implied by `rep`'s tag, which
/// must therefore still match the tag-derived layout it was allocated with.
#[inline]
pub(crate) unsafe fn delete(rep: *mut CordRep) {
    // SAFETY: `rep` is a live flat rep with a refcount of zero per this
    // fn's contract, so its `tag` may be read and, since the tag correctly
    // identifies the allocation's size class by construction, the matching
    // layout may be reconstructed and passed to `dealloc` to free exactly
    // the allocation `rep` was carved from.
    unsafe {
        let tag = rep.tag();
        debug_assert!((FLAT..=MAX_FLAT_TAG).contains(&tag));
        std::alloc::dealloc(rep.cast(), layout_for(tag_to_allocated_size(tag)));
    }
}

/// Creates a flat containing `data` with up to `extra` bytes of additional
/// capacity. Requires `data.len() <= MAX_FLAT_LENGTH`.
///
/// # Safety
///
/// `data.len()` must not exceed [`MAX_FLAT_LENGTH`]: the allocated flat's
/// capacity is `min(data.len() + extra, MAX_FLAT_LENGTH)`, so a longer
/// `data` would make the `copy_nonoverlapping` below write past the end of
/// the allocation. The returned rep is newly allocated and uniquely owned;
/// see [`new_impl`]'s ownership obligation note.
#[inline]
pub(crate) unsafe fn create(data: &[u8], extra: usize) -> *mut CordRep {
    debug_assert!(data.len() <= MAX_FLAT_LENGTH);
    let flat = new(data.len() + extra.min(MAX_FLAT_LENGTH));
    // SAFETY: `flat` was just allocated by `new` with capacity at least
    // `data.len()` (per this fn's contract on `data.len()` above), so
    // `self::data(flat)` is valid for `data.len()` bytes of write and does
    // not overlap `data` (a distinct, already-existing allocation).
    // `set_length` is sound because `flat` is exclusively owned by this
    // call.
    unsafe {
        core::ptr::copy_nonoverlapping(data.as_ptr(), self::data(flat), data.len());
        flat.set_length(data.len());
    }
    flat
}

/// Returns a pointer to the payload of `rep`.
///
/// The pointer is derived from the allocation pointer (not from a reference
/// to the header), so it is valid for the whole capacity.
///
/// # Safety
///
/// `rep` must be a non-null pointer to a live flat rep (tag in
/// `FLAT..=MAX_FLAT_TAG`), i.e. one allocated by [`new`], [`new_large`] or
/// [`create`]. The returned pointer is valid for reads/writes over
/// `capacity(rep)` bytes; writing through it additionally requires the
/// caller to hold `rep` exclusively (refcount of one), per the module's
/// reference-counting convention.
#[inline]
pub(crate) unsafe fn data(rep: *mut CordRep) -> *mut u8 {
    // SAFETY: `rep` is a live flat rep per this fn's contract, so offsetting
    // past its `FLAT_OVERHEAD`-byte header stays within the allocation and
    // yields the start of the payload.
    unsafe { rep.cast::<u8>().add(FLAT_OVERHEAD) }
}

/// Returns the payload capacity of `rep`.
///
/// # Safety
///
/// `rep` must be a non-null pointer to a live flat rep (tag in
/// `FLAT..=MAX_FLAT_TAG`).
#[inline]
pub(crate) unsafe fn capacity(rep: *mut CordRep) -> usize {
    unsafe { tag_to_length(rep.tag()) }
}

/// Returns the allocated size (payload + overhead) of `rep`.
///
/// # Safety
///
/// `rep` must be a non-null pointer to a live flat rep (tag in
/// `FLAT..=MAX_FLAT_TAG`).
#[inline]
pub(crate) unsafe fn allocated_size(rep: *mut CordRep) -> usize {
    unsafe { tag_to_allocated_size(rep.tag()) }
}

/// Copy handle borrowing a live flat rep for `'a`.
///
/// # Invariant
///
/// The wrapped pointer is non-null and points to a live flat rep (tag in
/// `FLAT..=MAX_FLAT_TAG`) for the duration of `'a`, established once at the
/// sole constructor [`from_raw`](Self::from_raw).
#[derive(Clone, Copy)]
pub(crate) struct FlatRef<'a> {
    ptr: NonNull<CordRep>,
    _marker: PhantomData<&'a CordRep>,
}

impl<'a> FlatRef<'a> {
    /// Wraps `ptr` as a flat handle borrowed for `'a`.
    ///
    /// # Safety
    ///
    /// `ptr` must be non-null and point to a live flat rep (tag in
    /// `FLAT..=MAX_FLAT_TAG`) that the caller guarantees stays live, and unmutated other than
    /// through its interior-mutable refcount, for `'a`.
    #[inline]
    pub(crate) unsafe fn from_raw(ptr: *mut CordRep) -> Self {
        debug_assert!(!ptr.is_null());
        // SAFETY: non-null per the debug_assert above.
        Self { ptr: unsafe { NonNull::new_unchecked(ptr) }, _marker: PhantomData }
    }

    /// This flat's `length`.
    #[inline]
    pub(crate) fn len(self) -> usize {
        // SAFETY: `self`'s invariant (struct doc) guarantees `self.ptr` is a
        // live flat rep for the call's duration.
        unsafe { self.ptr.as_ptr().length() }
    }

    /// This flat's payload capacity.
    #[inline]
    pub(crate) fn capacity(self) -> usize {
        // SAFETY: see `len`.
        unsafe { capacity(self.ptr.as_ptr()) }
    }

    /// This flat's allocated size (payload + overhead).
    #[inline]
    pub(crate) fn allocated_size(self) -> usize {
        // SAFETY: see `len`.
        unsafe { allocated_size(self.ptr.as_ptr()) }
    }

    /// The initialized prefix of this flat's payload (`len()` bytes).
    #[inline]
    pub(crate) fn data(self) -> &'a [u8] {
        let len = self.len();
        // SAFETY: `self`'s invariant makes `self.ptr` a live flat rep for
        // `'a`, so the pointer `data` derives from the allocation is valid
        // for reads over the whole capacity, which is at least `len` (a
        // flat's `length` never exceeds its capacity).
        unsafe { core::slice::from_raw_parts(data(self.ptr.as_ptr()), len) }
    }

    /// Escape hatch to the raw pointer, for code not yet converted to the
    /// handle types.
    #[inline]
    pub(crate) fn as_ptr(self) -> *mut CordRep {
        self.ptr.as_ptr()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tag_roundtrip() {
        let mut size = MIN_FLAT_SIZE;
        let mut expected_tag = FLAT;
        while size <= MAX_LARGE_FLAT_SIZE {
            assert_eq!(allocated_size_to_tag(size), expected_tag, "size {size}");
            assert_eq!(tag_to_allocated_size(expected_tag), size, "tag {expected_tag}");
            assert_eq!(round_up_for_tag(size), size);
            assert_eq!(tag_to_length(expected_tag), size - FLAT_OVERHEAD);
            let step = if size < 512 {
                8
            } else if size < 8192 {
                64
            } else {
                4096
            };
            size += step;
            expected_tag += 1;
        }
        assert_eq!(expected_tag, MAX_FLAT_TAG + 1);
    }

    #[test]
    fn round_up_for_tag_rounds_to_next_step() {
        assert_eq!(round_up_for_tag(33), 40);
        assert_eq!(round_up_for_tag(512), 512);
        assert_eq!(round_up_for_tag(513), 576);
        assert_eq!(round_up_for_tag(8192), 8192);
        assert_eq!(round_up_for_tag(8193), 12288);
    }

    #[test]
    fn new_and_delete_sizes() {
        unsafe {
            for &len in &[
                0usize,
                1,
                MIN_FLAT_LENGTH,
                MIN_FLAT_LENGTH + 1,
                100,
                1000,
                MAX_FLAT_LENGTH,
                MAX_FLAT_LENGTH + 1,
                1 << 20,
            ] {
                let rep = new(len);
                assert!(rep.ref_is_one());
                assert_eq!(rep.length(), 0);
                assert!(rep.is_flat());
                let cap = capacity(rep);
                assert!(cap >= len.min(MAX_FLAT_LENGTH));
                assert!(cap <= MAX_FLAT_LENGTH);
                assert!(cap >= MIN_FLAT_LENGTH);
                // Write the whole capacity: must be within the allocation.
                core::ptr::write_bytes(data(rep), 0xAB, cap);
                delete(rep);

                let large = new_large(len);
                let cap = capacity(large);
                assert!(cap >= len.min(MAX_LARGE_FLAT_LENGTH));
                assert!(cap <= MAX_LARGE_FLAT_LENGTH);
                core::ptr::write_bytes(data(large), 0xAB, cap);
                delete(large);
            }
        }
    }

    #[test]
    fn create_copies_data() {
        unsafe {
            let rep = create(b"hello world", 100);
            assert_eq!(rep.length(), 11);
            assert!(capacity(rep) >= 111);
            assert_eq!(super::super::edge_data(rep), b"hello world");
            delete(rep);
        }
    }
}
