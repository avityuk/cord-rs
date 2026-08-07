//! [`CordBuffer`]: a writable buffer that can be appended to a [`Cord`]
//! without copying.

use core::fmt;
use core::mem::MaybeUninit;
use core::ops::{Deref, DerefMut};

use crate::rep::flat::{self, FLAT_OVERHEAD, FlatRef, MAX_FLAT_LENGTH, MAX_LARGE_FLAT_SIZE};
use crate::rep::{CordRep, UniqueRep, small_u8};

/// Inline (small buffer) capacity of a `CordBuffer`.
const INLINE_CAPACITY: usize = core::mem::size_of::<usize>() * 2 - 1;

// Assume the cost of an "up-rounded" allocation to `ceil_pow2(size)` versus
// the cost of allocating at least one extra flat <= 4KB: flat overhead (13)
// + amortized btree cost per node (~13) + 64 byte tcmalloc granularity at
// 4K (~32 average). Splitting must save something; as a poor man's measure
// the slop needs to be at least double the cost offset: ~128 bytes.
const MAX_PAGE_SLOP: usize = 128;

#[cfg(target_endian = "little")]
#[repr(C)]
#[derive(Clone, Copy)]
struct Long {
    rep: *mut CordRep,
    _padding: usize,
}

#[cfg(target_endian = "little")]
#[repr(C)]
#[derive(Clone, Copy)]
struct Short {
    raw_size: u8,
    data: [u8; INLINE_CAPACITY],
}

#[cfg(target_endian = "big")]
#[repr(C)]
#[derive(Clone, Copy)]
struct Long {
    _padding: usize,
    rep: *mut CordRep,
}

#[cfg(target_endian = "big")]
#[repr(C)]
#[derive(Clone, Copy)]
struct Short {
    data: [u8; INLINE_CAPACITY],
    raw_size: u8,
}

/// The internal representation: either an inline buffer or a flat rep. The
/// least significant byte of the (always even) rep pointer overlaps
/// `raw_size`, whose low bit is set for the inline form.
#[repr(C)]
#[derive(Clone, Copy)]
union Rep {
    long: Long,
    short: Short,
}

const _: () = assert!(core::mem::size_of::<Rep>() == 2 * core::mem::size_of::<usize>());
const _: () = assert!(core::mem::size_of::<Short>() == core::mem::size_of::<Long>());

impl Short {
    #[inline]
    fn len(&self) -> usize {
        (self.raw_size >> 1) as usize
    }

    #[inline]
    fn set_len(&mut self, length: usize) {
        debug_assert!(length <= INLINE_CAPACITY);
        self.raw_size = small_u8((length << 1) + 1);
    }
}

/// The safe, read-only view of a [`Rep`]'s union: the single site (besides
/// [`Rep::view_mut`]) that reads its discriminant.
#[derive(Clone, Copy)]
enum BufRepr<'a> {
    /// Inline (small buffer) form.
    Short(&'a Short),
    /// Heap allocated flat form.
    Flat(FlatRef<'a>),
}

impl<'a> BufRepr<'a> {
    #[inline]
    fn len(self) -> usize {
        match self {
            Self::Short(s) => s.len(),
            Self::Flat(f) => f.len(),
        }
    }

    #[inline]
    fn capacity(self) -> usize {
        match self {
            Self::Short(_) => INLINE_CAPACITY,
            Self::Flat(f) => f.capacity(),
        }
    }

    #[inline]
    fn data(self) -> &'a [u8] {
        match self {
            Self::Short(s) => &s.data[..s.len()],
            Self::Flat(f) => f.data(),
        }
    }
}

/// The mutable counterpart of [`BufRepr`]: the result of [`Rep::view_mut`].
enum BufReprMut<'a> {
    /// Inline (small buffer) form.
    Short(&'a mut Short),
    /// Heap allocated flat form. A [`UniqueRep`] rather than a raw pointer:
    /// see [`Rep::view_mut`]'s doc for why constructing one here is sound.
    Flat(UniqueRep<'a>),
}

impl Rep {
    #[inline]
    const fn new_short() -> Self {
        Self { short: Short { raw_size: 1, data: [0; INLINE_CAPACITY] } }
    }

    /// `true` if this rep holds the inline (short) form. The one low-level
    /// union-tag read that [`view`](Self::view) and
    /// [`view_mut`](Self::view_mut) build on; every other accessor in this
    /// file goes through one of those two instead of reading the union
    /// directly.
    #[inline]
    fn is_short(&self) -> bool {
        // SAFETY: `raw_size` overlaps the low byte of an even pointer in the
        // long form, so the low bit identifies the form either way. Reading
        // a byte of a pointer as an integer is fine.
        unsafe { self.short.raw_size & 1 != 0 }
    }

    /// The safe, read-only view of this rep's union.
    #[inline]
    fn view(&self) -> BufRepr<'_> {
        if self.is_short() {
            // SAFETY: `is_short()` (checked above) confirms the `short`
            // variant was last written.
            BufRepr::Short(unsafe { &self.short })
        } else {
            // SAFETY: long form is active (checked above); `CordBuffer`
            // maintains a live flat rep whenever the long form is active
            // (established by `from_flat`'s contract, and never changed
            // to anything else afterward), kept live for `self`'s borrow.
            BufRepr::Flat(unsafe { FlatRef::from_raw(self.long.rep) })
        }
    }

    /// The mutable counterpart of [`view`](Self::view).
    ///
    /// Sound because `CordBuffer`'s flat, whenever present, is
    /// *unconditionally* exclusively owned (refcount one) for the buffer's
    /// entire lifetime: it is only ever created by
    /// [`CordBuffer::from_flat`] (which requires this), and `CordBuffer` is
    /// not `Clone` and never exposes the pointer while retaining ownership.
    /// So this `&mut self` borrow is sufficient — the same way
    /// `OwnedRep::try_unique`'s is — to prove no other handle to the node
    /// exists for as long as it lasts; see [`UniqueRep`]'s own soundness
    /// note for the general pattern this mirrors (and for why it names this
    /// fn as one of its three permitted call sites).
    #[inline]
    fn view_mut(&mut self) -> BufReprMut<'_> {
        if self.is_short() {
            // SAFETY: see `view`.
            BufReprMut::Short(unsafe { &mut self.short })
        } else {
            // SAFETY: see this fn's doc above and `view`.
            BufReprMut::Flat(unsafe { UniqueRep::from_raw(self.long.rep) })
        }
    }
}

/// A buffer of bytes that can be appended to (or prepended to) a [`Cord`](crate::Cord)
/// without copying.
///
/// `CordBuffer` is useful for zero-copy APIs (e.g. reading from a socket
/// directly into memory that becomes part of a cord) and for building large
/// cords with control over allocation sizes. A buffer has a `capacity` and a
/// `len`; the first `len` bytes are initialized data.
///
/// ```
/// use cord_rs::{Cord, CordBuffer};
///
/// fn read_all(mut src: &[u8]) -> Cord {
///     let mut cord = Cord::new();
///     while !src.is_empty() {
///         let mut buffer = CordBuffer::with_default_limit(src.len());
///         let n = buffer.available().min(src.len());
///         // Zero-copy fill: write into the uninitialized spare capacity.
///         buffer.put_slice(&src[..n]);
///         src = &src[n..];
///         cord.append(buffer);
///     }
///     cord
/// }
/// assert_eq!(read_all(&[7u8; 10_000]).len(), 10_000);
/// ```
///
/// Buffers of up to [`DEFAULT_LIMIT`](Self::DEFAULT_LIMIT) bytes (just under
/// 4 KiB) are created with [`with_default_limit`](Self::with_default_limit);
/// larger buffers need [`with_custom_limit`](Self::with_custom_limit). The
/// default limit balances CPU efficiency (larger buffers) against memory
/// overhead and fragmentation (smaller buffers). A buffer's capacity may
/// exceed the requested one due to allocation size rounding; use
/// [`capacity`](Self::capacity) / [`available`](Self::available).
///
/// The uninitialized part of a buffer is exposed as
/// [`spare_capacity_mut`](Self::spare_capacity_mut) plus the `unsafe`
/// [`set_len`](Self::set_len), mirroring `Vec`; the safe
/// [`put_slice`](Self::put_slice) and `std::io::Write` cover most uses.
pub struct CordBuffer {
    rep: Rep,
}

// SAFETY: the buffer exclusively owns its (possibly heap allocated) memory.
// `Rep`'s long form stores a raw `*mut CordRep` (the tagged-pointer trick
// overlapping its low bit with `Short::raw_size` needs a real pointer, not a
// `NonNull`-wrapping handle), so unlike `Chunks`/`CordRepBtreeReader` this
// type can never auto-derive `Send`/`Sync` regardless of how the rest of
// this file is organized; kept manual, mirroring `OwnedRep`'s own impls
// (rep.rs) with the same justification.
unsafe impl Send for CordBuffer {}
unsafe impl Sync for CordBuffer {}

/// The result of consuming a `CordBuffer`.
pub(crate) enum ConsumedBuffer {
    /// A heap allocated flat holding the data (refcount 1).
    Rep(*mut CordRep),
    /// Inline data.
    Short(ShortValue),
}

/// Copied out inline data of a consumed buffer.
pub(crate) struct ShortValue {
    data: [u8; INLINE_CAPACITY],
    len: usize,
}

impl ShortValue {
    #[inline]
    pub(crate) fn as_slice(&self) -> &[u8] {
        &self.data[..self.len]
    }
}

impl CordBuffer {
    /// Default capacity limit of buffers created with
    /// [`with_default_limit`](Self::with_default_limit): just under 4 KiB.
    pub const DEFAULT_LIMIT: usize = MAX_FLAT_LENGTH;

    /// Maximum size of buffers created with
    /// [`with_custom_limit`](Self::with_custom_limit) (64 KiB). The effective
    /// capacity is slightly less because of internal overhead.
    pub const CUSTOM_LIMIT: usize = 64 << 10;

    const _CHECK: () =
        assert!(Self::CUSTOM_LIMIT <= MAX_LARGE_FLAT_SIZE, "custom limit exceeds max flat size");

    /// Creates an empty buffer with a small inline capacity. Does not
    /// allocate.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self { rep: Rep::new_short() }
    }

    /// The guaranteed maximum payload of a buffer created with
    /// [`with_default_limit`](Self::with_default_limit). Useful to estimate
    /// the number of buffers needed for a given size.
    #[inline]
    #[must_use]
    pub const fn maximum_payload() -> usize {
        MAX_FLAT_LENGTH
    }

    /// The maximum payload of a buffer created with
    /// [`with_custom_limit`](Self::with_custom_limit) for `block_size`.
    #[inline]
    #[must_use]
    pub const fn maximum_payload_for(block_size: usize) -> usize {
        let limit = if block_size < Self::CUSTOM_LIMIT { block_size } else { Self::CUSTOM_LIMIT };
        limit - FLAT_OVERHEAD
    }

    /// Creates a buffer of the desired `capacity`, capped at
    /// [`DEFAULT_LIMIT`](Self::DEFAULT_LIMIT). The returned buffer has a
    /// capacity of at least `min(DEFAULT_LIMIT, capacity)`.
    #[must_use]
    pub fn with_default_limit(capacity: usize) -> Self {
        if capacity > INLINE_CAPACITY {
            // SAFETY: a fresh flat with length 0.
            unsafe {
                let rep = flat::new(capacity);
                return Self::from_flat(rep);
            }
        }
        Self::new()
    }

    /// Creates a buffer of the desired `capacity` rounded to an appropriate
    /// power of two size less than or equal to `block_size`.
    ///
    /// If `capacity <= DEFAULT_LIMIT` this behaves like
    /// [`with_default_limit`](Self::with_default_limit). If `capacity >=
    /// block_size` the buffer has an *allocated size* of `block_size` bytes
    /// (capacity `block_size - 13` on 64-bit). Otherwise a suitable smaller
    /// power of two block is chosen: typically the highest or second highest
    /// power of two <= `capacity`, favoring low memory slop over precise
    /// sizing to reduce fragmentation. Both arguments are capped at
    /// [`CUSTOM_LIMIT`](Self::CUSTOM_LIMIT).
    ///
    /// Only use custom limits when the data is expected to be many times the
    /// chosen block size, based on measurements.
    ///
    /// # Panics
    ///
    /// Panics if `block_size` is not a power of two.
    #[must_use]
    pub fn with_custom_limit(block_size: usize, capacity: usize) -> Self {
        assert!(block_size.is_power_of_two(), "block_size must be a power of two, got {block_size}");
        let mut capacity = capacity.min(Self::CUSTOM_LIMIT);
        let block_size = block_size.min(Self::CUSTOM_LIMIT);
        if capacity + FLAT_OVERHEAD >= block_size {
            capacity = block_size;
        } else if capacity <= Self::DEFAULT_LIMIT {
            capacity += FLAT_OVERHEAD;
        } else if !capacity.is_power_of_two() {
            // Check if rounding up to the next power of 2 is a good enough
            // fit with limited waste.
            let rounded_up = capacity.next_power_of_two();
            let slop = rounded_up - capacity;
            capacity = if (FLAT_OVERHEAD..=MAX_PAGE_SLOP + FLAT_OVERHEAD).contains(&slop) {
                rounded_up
            } else {
                // Round down to the highest power of 2 <= capacity.
                1 << capacity.ilog2()
            };
        }
        let length = capacity - FLAT_OVERHEAD;
        // SAFETY: a fresh flat with length 0.
        unsafe {
            let rep = flat::new_large(length);
            Self::from_flat(rep)
        }
    }

    /// Wraps a flat rep with a refcount of one.
    ///
    /// # Safety
    ///
    /// `rep` must be a non-null pointer to a live, uniquely-owned (refcount
    /// one) flat node; the caller transfers that sole reference to the
    /// returned `CordBuffer`, which will free it on drop.
    #[inline]
    pub(crate) unsafe fn from_flat(rep: *mut CordRep) -> Self {
        // SAFETY: `rep` is a live flat node per the caller contract above,
        // which is all `debug_assert_unique_flat` requires.
        unsafe {
            crate::rep::debug_assert_unique_flat(rep);
        }
        Self { rep: Rep { long: Long { rep, _padding: 0 } } }
    }

    /// Consumes the buffer, returning its flat rep (with the current length)
    /// or a copy of its inline data.
    pub(crate) fn consume(self) -> ConsumedBuffer {
        let this = core::mem::ManuallyDrop::new(self);
        match this.rep.view() {
            BufRepr::Short(s) => ConsumedBuffer::Short(ShortValue { data: s.data, len: s.len() }),
            BufRepr::Flat(f) => ConsumedBuffer::Rep(f.as_ptr()),
        }
    }

    /// Number of initialized bytes.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.rep.view().len()
    }

    /// Returns `true` if the buffer holds no initialized bytes.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Total capacity. Always non-zero: even default buffers have a small
    /// inline capacity.
    #[inline]
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.rep.view().capacity()
    }

    /// Number of bytes that can still be written: `capacity() - len()`.
    #[inline]
    #[must_use]
    pub fn available(&self) -> usize {
        self.capacity() - self.len()
    }

    /// The initialized bytes.
    #[inline]
    #[must_use]
    pub fn as_slice(&self) -> &[u8] {
        self.rep.view().data()
    }

    /// The initialized bytes, mutably.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        match self.rep.view_mut() {
            BufReprMut::Short(s) => {
                let len = s.len();
                &mut s.data[..len]
            }
            BufReprMut::Flat(f) => f.flat_data_mut(),
        }
    }

    /// The uninitialized spare capacity (`capacity() - len()` bytes).
    ///
    /// Write data into it, then call [`set_len`](Self::set_len) to mark it
    /// initialized.
    ///
    /// ```
    /// use cord_rs::CordBuffer;
    /// let mut buffer = CordBuffer::with_default_limit(64);
    /// let spare = buffer.spare_capacity_mut();
    /// spare[0].write(b'h');
    /// spare[1].write(b'i');
    /// // SAFETY: the first two bytes were initialized above.
    /// unsafe { buffer.set_len(2) };
    /// assert_eq!(buffer.as_slice(), b"hi");
    /// ```
    #[inline]
    pub fn spare_capacity_mut(&mut self) -> &mut [MaybeUninit<u8>] {
        match self.rep.view_mut() {
            BufReprMut::Short(s) => {
                let len = s.len();
                let ptr = s.data.as_mut_ptr().cast::<MaybeUninit<u8>>();
                // SAFETY: `s.data` spans `INLINE_CAPACITY` bytes and `len <=
                // INLINE_CAPACITY` (`Short`'s own invariant), so `[len,
                // INLINE_CAPACITY)` stays in bounds; `MaybeUninit<u8>` has
                // the same layout as `u8`, and viewing already-initialized
                // bytes through it only weakens what's known about them.
                unsafe { core::slice::from_raw_parts_mut(ptr.add(len), INLINE_CAPACITY - len) }
            }
            BufReprMut::Flat(f) => f.into_flat_spare_capacity_mut(),
        }
    }

    /// Sets the number of initialized bytes.
    ///
    /// Setting a smaller length does not release memory.
    ///
    /// # Safety
    ///
    /// The first `len` bytes must be initialized.
    ///
    /// # Panics
    ///
    /// Panics if `len > capacity()`.
    #[track_caller]
    #[inline]
    pub unsafe fn set_len(&mut self, len: usize) {
        assert!(
            len <= self.capacity(),
            "CordBuffer::set_len: len {len} exceeds capacity {}",
            self.capacity()
        );
        // `len` was just checked against `capacity()` above; the first `len`
        // bytes being initialized is this fn's own `# Safety` contract.
        match self.rep.view_mut() {
            BufReprMut::Short(s) => s.set_len(len),
            BufReprMut::Flat(mut f) => f.set_len(len),
        }
    }

    /// Appends `src` to the initialized data.
    ///
    /// # Panics
    ///
    /// Panics if `src.len() > available()`.
    #[track_caller]
    pub fn put_slice(&mut self, src: &[u8]) {
        // One union dispatch for the whole operation (length read, capacity
        // check, copy, length update) — the accessor-per-step form paid
        // this dispatch five times.
        match self.rep.view_mut() {
            BufReprMut::Short(short) => {
                let len = short.len();
                let available = INLINE_CAPACITY - len;
                assert!(
                    src.len() <= available,
                    "CordBuffer::put_slice: {} bytes exceed the available capacity of {}",
                    src.len(),
                    available
                );
                short.data[len..len + src.len()].copy_from_slice(src);
                short.set_len(len + src.len());
            }
            BufReprMut::Flat(mut unique) => {
                let len = unique.as_ref().len();
                let spare = unique.flat_spare_capacity_mut();
                assert!(
                    src.len() <= spare.len(),
                    "CordBuffer::put_slice: {} bytes exceed the available capacity of {}",
                    src.len(),
                    spare.len()
                );
                spare[..src.len()].write_copy_of_slice(src);
                unique.set_len(len + src.len());
            }
        }
    }

    /// Appends as many bytes of `src` as fit and returns how many were
    /// written.
    pub fn put_slice_partial(&mut self, src: &[u8]) -> usize {
        let n = src.len().min(self.available());
        self.put_slice(&src[..n]);
        n
    }

    /// Shortens the initialized data to `len` bytes. No effect if `len >=
    /// self.len()`. Does not release memory. Clamping means this can only
    /// shrink (or no-op), which can never expose uninitialized bytes, so it
    /// is sound as a safe fn (unlike `set_len`, which can also grow).
    #[inline]
    pub fn truncate(&mut self, len: usize) {
        let len = len.min(self.len());
        // SAFETY: `len <= self.len()`, so every byte up to it is already
        // initialized.
        unsafe { self.set_len(len) };
    }

    /// Sets the length to zero. Does not release memory.
    #[inline]
    pub fn clear(&mut self) {
        self.truncate(0);
    }
}

impl Default for CordBuffer {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for CordBuffer {
    #[inline]
    fn drop(&mut self) {
        if let BufRepr::Flat(f) = self.rep.view() {
            // SAFETY: we exclusively own the flat (`CordBuffer`'s
            // invariant; see `Rep::view_mut`'s doc).
            unsafe { flat::delete(f.as_ptr()) };
        }
    }
}

impl Deref for CordBuffer {
    type Target = [u8];
    #[inline]
    fn deref(&self) -> &[u8] {
        self.as_slice()
    }
}

impl DerefMut for CordBuffer {
    #[inline]
    fn deref_mut(&mut self) -> &mut [u8] {
        self.as_mut_slice()
    }
}

impl AsRef<[u8]> for CordBuffer {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        self.as_slice()
    }
}

impl AsMut<[u8]> for CordBuffer {
    #[inline]
    fn as_mut(&mut self) -> &mut [u8] {
        self.as_mut_slice()
    }
}

impl fmt::Debug for CordBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CordBuffer")
            .field("len", &self.len())
            .field("capacity", &self.capacity())
            .field("data", &format_args!("b\"{}\"", self.as_slice().escape_ascii()))
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout() {
        assert_eq!(core::mem::size_of::<CordBuffer>(), 2 * core::mem::size_of::<usize>());
        let b = CordBuffer::new();
        assert!(b.rep.is_short());
        assert_eq!(b.capacity(), INLINE_CAPACITY);
        assert_eq!(b.len(), 0);
    }

    #[test]
    fn inline_and_heap() {
        let mut b = CordBuffer::with_default_limit(INLINE_CAPACITY);
        assert!(b.rep.is_short());
        b.put_slice(b"abc");
        assert_eq!(&*b, b"abc");
        assert_eq!(b.available(), INLINE_CAPACITY - 3);

        let mut b = CordBuffer::with_default_limit(INLINE_CAPACITY + 1);
        assert!(!b.rep.is_short());
        assert!(b.capacity() > INLINE_CAPACITY);
        assert!(b.capacity() <= CordBuffer::DEFAULT_LIMIT);
        let n = b.put_slice_partial(&[1u8; 10_000]);
        assert_eq!(n, b.capacity());
        assert_eq!(b.available(), 0);
        b.truncate(3);
        assert_eq!(b.len(), 3);
        b.clear();
        assert!(b.is_empty());
    }

    #[test]
    fn custom_limit_sizes() {
        let b = CordBuffer::with_custom_limit(64 << 10, 64 << 10);
        assert_eq!(b.capacity(), (64 << 10) - FLAT_OVERHEAD);
        let b = CordBuffer::with_custom_limit(64 << 10, 100);
        assert!(b.capacity() >= 100);
        let b = CordBuffer::with_custom_limit(64 << 10, 19_586);
        assert_eq!(b.capacity(), (16 << 10) - FLAT_OVERHEAD);
        let b = CordBuffer::with_custom_limit(64 << 10, 1 << 20);
        assert_eq!(b.capacity(), (64 << 10) - FLAT_OVERHEAD);
        assert_eq!(CordBuffer::maximum_payload_for(8 << 10), (8 << 10) - FLAT_OVERHEAD);
        assert_eq!(CordBuffer::maximum_payload_for(1 << 20), CordBuffer::CUSTOM_LIMIT - FLAT_OVERHEAD);
    }

    #[test]
    #[should_panic(expected = "power of two")]
    fn custom_limit_rejects_non_pow2() {
        let _ = CordBuffer::with_custom_limit(1000, 10);
    }

    #[test]
    #[should_panic(expected = "exceed the available capacity")]
    fn put_slice_overflow_panics() {
        let mut b = CordBuffer::new();
        b.put_slice(&[0u8; 100]);
    }
}
