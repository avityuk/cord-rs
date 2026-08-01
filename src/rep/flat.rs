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
use core::mem::{align_of, offset_of};

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
    // SAFETY: size is a multiple of 8 >= 32 and the alignment is a power of 2.
    unsafe { Layout::from_size_align_unchecked(size, align_of::<CordRep>()) }
}

#[inline]
#[expect(clippy::cast_ptr_alignment, reason = "the layout requests align_of::<CordRep>()")]
unsafe fn new_impl<const MAX_SIZE: usize>(mut len: usize) -> *mut CordRep {
    if len <= MIN_FLAT_LENGTH {
        len = MIN_FLAT_LENGTH;
    } else if len > MAX_SIZE - FLAT_OVERHEAD {
        len = MAX_SIZE - FLAT_OVERHEAD;
    }
    // Round size up so it matches a size we can exactly express in a tag.
    let size = round_up_for_tag(len + FLAT_OVERHEAD);
    let layout = layout_for(size);
    let raw = std::alloc::alloc(layout);
    if raw.is_null() {
        std::alloc::handle_alloc_error(layout);
    }
    let rep = raw.cast::<CordRep>();
    rep.write(CordRep::new(0, allocated_size_to_tag(size)));
    rep
}

/// Allocates a new flat with a capacity of at least `min(len, MAX_FLAT_LENGTH)`
/// bytes (and at least `MIN_FLAT_LENGTH`). The returned flat has `length == 0`.
#[inline]
pub(crate) unsafe fn new(len: usize) -> *mut CordRep {
    new_impl::<MAX_FLAT_SIZE>(len)
}

/// Like [`new`] but allows capacities up to `MAX_LARGE_FLAT_LENGTH`.
#[inline]
pub(crate) unsafe fn new_large(len: usize) -> *mut CordRep {
    new_impl::<MAX_LARGE_FLAT_SIZE>(len)
}

/// Deallocates a flat created by [`new`] / [`new_large`].
#[inline]
pub(crate) unsafe fn delete(rep: *mut CordRep) {
    let tag = rep.tag();
    debug_assert!((FLAT..=MAX_FLAT_TAG).contains(&tag));
    std::alloc::dealloc(rep.cast(), layout_for(tag_to_allocated_size(tag)));
}

/// Creates a flat containing `data` with up to `extra` bytes of additional
/// capacity. Requires `data.len() <= MAX_FLAT_LENGTH`.
#[inline]
pub(crate) unsafe fn create(data: &[u8], extra: usize) -> *mut CordRep {
    debug_assert!(data.len() <= MAX_FLAT_LENGTH);
    let flat = new(data.len() + extra.min(MAX_FLAT_LENGTH));
    core::ptr::copy_nonoverlapping(data.as_ptr(), self::data(flat), data.len());
    flat.set_length(data.len());
    flat
}

/// Returns a pointer to the payload of `rep`.
///
/// The pointer is derived from the allocation pointer (not from a reference
/// to the header), so it is valid for the whole capacity.
#[inline]
pub(crate) unsafe fn data(rep: *mut CordRep) -> *mut u8 {
    rep.cast::<u8>().add(FLAT_OVERHEAD)
}

/// Returns the payload capacity of `rep`.
#[inline]
pub(crate) unsafe fn capacity(rep: *mut CordRep) -> usize {
    tag_to_length(rep.tag())
}

/// Returns the allocated size (payload + overhead) of `rep`.
#[inline]
pub(crate) unsafe fn allocated_size(rep: *mut CordRep) -> usize {
    tag_to_allocated_size(rep.tag())
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
                assert!(rep.refcount().is_one());
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
