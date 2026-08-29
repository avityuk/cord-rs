//! [`CordBuffer`]: a writable buffer that can be appended to a [`Cord`]
//! without copying.

use core::borrow::{Borrow, BorrowMut};
use core::cmp::Ordering;
use core::fmt;
use core::hash::{Hash, Hasher};
use core::mem::MaybeUninit;
use core::ops::{Deref, DerefMut};

use alloc::string::String;
use alloc::vec::Vec;

use crate::rep::flat::{self, FLAT_OVERHEAD, FlatRef, MAX_FLAT_LENGTH, MAX_LARGE_FLAT_SIZE, MIN_FLAT_LENGTH};
use crate::rep::{CordRep, MAX_INLINE, UniqueRep, small_u8};

/// Inline (small buffer) capacity of a `CordBuffer`: 15 bytes on every
/// target. `CordBuffer` is a fixed 16-byte type regardless of pointer
/// width — the two-word `Long`/`Short` layout below only needs `Long` to be
/// exactly 16 bytes, not two pointer-widths, so the inline capacity does not
/// shrink on 32-bit targets. Kept equal to [`MAX_INLINE`] (the inline
/// capacity of `Cord` itself) so a buffer's contents never need special-
/// casing to land inline once handed to a `Cord`; see the `INLINE_CAPACITY
/// == MAX_INLINE` assertion below.
const INLINE_CAPACITY: usize = MAX_INLINE;

// Assume the cost of an "up-rounded" allocation to `ceil_pow2(size)` versus
// the cost of allocating at least one extra flat <= 4KB: flat overhead (13)
// + amortized btree cost per node (~13) + 64 byte tcmalloc granularity at
// 4K (~32 average). Splitting must save something; as a poor man's measure
// the slop needs to be at least double the cost offset: ~128 bytes.
const MAX_PAGE_SLOP: usize = 128;

/// `Long`'s trailing padding, sized so `Long` is exactly 16 bytes on any
/// pointer width (8 on 64-bit, matching the old all-`usize` layout; 12 on
/// 32-bit, wider than before so the struct — and with it `Rep`/`CordBuffer`
/// — stays a fixed 16 bytes rather than shrinking to two pointer-widths).
const LONG_PADDING: usize = 16 - core::mem::size_of::<*mut CordRep>();

#[cfg(target_endian = "little")]
#[repr(C)]
#[derive(Clone, Copy)]
struct Long {
    rep: *mut CordRep,
    _padding: [u8; LONG_PADDING],
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
    _padding: [u8; LONG_PADDING],
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
/// `raw_size`, whose low bit is set for the inline form. This holds on every
/// pointer width and both endiannesses: little-endian, `rep` starts at
/// offset 0, so its least significant byte *is* byte 0, same as `raw_size`;
/// big-endian, `rep` occupies the last `size_of::<*mut CordRep>()` bytes (12
/// through 15 on 32-bit, 8 through 15 on 64-bit), and a big-endian pointer's
/// least significant byte is its last one — byte 15 — which is exactly where
/// `Short`'s 15-byte `data` places `raw_size`.
#[repr(C)]
#[derive(Clone, Copy)]
union Rep {
    long: Long,
    short: Short,
}

const _: () = assert!(core::mem::size_of::<Rep>() == 16);
const _: () = assert!(core::mem::size_of::<Short>() == 16);
const _: () = assert!(core::mem::size_of::<Long>() == 16);
const _: () = assert!(INLINE_CAPACITY == MAX_INLINE);
const _: () = assert!(core::mem::align_of::<Long>() == core::mem::align_of::<*mut CordRep>());

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
/// `CordBuffer` exists for zero-copy ingestion — reading from a socket, a
/// file or a decoder directly into memory that becomes part of a cord,
/// instead of filling a `Vec` and copying it in — and for control over
/// allocation size when building large cords (see the README's "Zero-copy
/// ingestion" section for the fuller picture). A buffer has a `capacity` and
/// a `len`; the first `len` bytes are initialized data. Appending a buffer
/// to a cord adds exactly one chunk and one tree edge, whatever the
/// buffer's size.
///
/// ```
/// use cord_rs::{Cord, CordBuffer};
///
/// fn read_all(mut src: &[u8]) -> Cord {
///     let mut cord = Cord::new();
///     while !src.is_empty() {
///         let mut buffer = CordBuffer::with_capacity(src.len());
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
/// # Sizing
///
/// [`with_capacity`](Self::with_capacity) caps the buffer at
/// [`DEFAULT_MAX_CAPACITY`](Self::DEFAULT_MAX_CAPACITY) (just under 4 KiB) —
/// silently: a request for more is not an error, it is capped, so check
/// [`capacity`](Self::capacity) rather than assuming the request was
/// honored in full. That default balances CPU efficiency (larger buffers)
/// against memory overhead and fragmentation (smaller buffers), and is the
/// right choice unless measurements say otherwise;
/// [`with_capacity_and_block_size`](Self::with_capacity_and_block_size) goes
/// up to [`MAX_BLOCK_SIZE`](Self::MAX_BLOCK_SIZE) for data known to be many
/// times that size. Either way, a buffer's capacity may exceed the
/// requested one because allocations are rounded to a size class; always
/// drive loops off [`capacity`](Self::capacity) / [`available`](Self::available)
/// rather than the requested number.
///
/// The uninitialized part of a buffer is exposed as
/// [`spare_capacity_mut`](Self::spare_capacity_mut) plus the `unsafe`
/// [`set_len`](Self::set_len), mirroring `Vec`; the safe
/// [`put_slice`](Self::put_slice), [`Extend`] and `std::io::Write` (and
/// `bytes::BufMut` with the `bytes` feature) cover most uses.
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
    /// [`with_capacity`](Self::with_capacity): just under 4 KiB. Also the
    /// guaranteed maximum payload of such a buffer — useful to estimate the
    /// number of buffers needed for a given size.
    pub const DEFAULT_MAX_CAPACITY: usize = MAX_FLAT_LENGTH;

    /// Maximum `block_size` accepted by
    /// [`with_capacity_and_block_size`](Self::with_capacity_and_block_size)
    /// (64 KiB). The buffer's capacity is slightly less because of the
    /// internal header.
    ///
    /// The underlying allocation-size tag can address blocks up to 256 KiB;
    /// this limit is simply where the crate stops asking the allocator for
    /// more, not a hard ceiling of the format.
    pub const MAX_BLOCK_SIZE: usize = 64 << 10;

    const _CHECK: () =
        assert!(Self::MAX_BLOCK_SIZE <= MAX_LARGE_FLAT_SIZE, "max block size exceeds max flat size");

    /// Creates an empty buffer with a small inline capacity. Does not
    /// allocate.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self { rep: Rep::new_short() }
    }

    /// The maximum payload of a buffer created with
    /// [`with_capacity_and_block_size`](Self::with_capacity_and_block_size)
    /// for `block_size`.
    ///
    /// # Panics
    ///
    /// Panics if `block_size` is not a power of two, or is not greater than
    /// the flat header overhead (13 bytes on 64-bit platforms, 9 on
    /// 32-bit) — the smallest legal `block_size` is 16.
    #[inline]
    #[must_use]
    pub const fn max_capacity_for(block_size: usize) -> usize {
        assert!(
            block_size.is_power_of_two() && block_size > FLAT_OVERHEAD,
            "block_size must be a power of two greater than FLAT_OVERHEAD"
        );
        let limit = if block_size < Self::MAX_BLOCK_SIZE { block_size } else { Self::MAX_BLOCK_SIZE };
        let payload = limit - FLAT_OVERHEAD;
        // `with_capacity_and_block_size` allocates via `flat::new_large`,
        // which floors the payload at `MIN_FLAT_LENGTH`; mirror that here so
        // this agrees with what `with_capacity_and_block_size` actually
        // produces for small blocks.
        if payload < MIN_FLAT_LENGTH { MIN_FLAT_LENGTH } else { payload }
    }

    /// Creates a buffer of the desired `capacity`, capped at
    /// [`DEFAULT_MAX_CAPACITY`](Self::DEFAULT_MAX_CAPACITY). The returned
    /// buffer has a capacity of at least `min(DEFAULT_MAX_CAPACITY,
    /// capacity)`.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
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
    /// If `capacity <= DEFAULT_MAX_CAPACITY` this behaves like
    /// [`with_capacity`](Self::with_capacity). If `capacity >= block_size`
    /// the buffer has an *allocated size* of `block_size` bytes (capacity
    /// `block_size - 13` on 64-bit). Otherwise a suitable smaller power of
    /// two block is chosen: typically the highest or second highest power of
    /// two <= `capacity`, favoring low memory slop over precise sizing to
    /// reduce fragmentation. Both arguments are capped at
    /// [`MAX_BLOCK_SIZE`](Self::MAX_BLOCK_SIZE).
    ///
    /// Only use a custom block size when the data is expected to be many
    /// times the chosen block size, and base the choice on measurements: a
    /// larger block means fewer chunks and less per-chunk overhead, but
    /// pins more memory per chunk and raises the risk of fragmentation.
    /// Rounding down (rather than up to the next block) keeps the
    /// distribution of allocation sizes narrow, which is what makes this a
    /// good trade for the allocator: a stream of requests produces a
    /// handful of distinct sizes instead of a spread of block-rounded ones
    /// with unused tails. For example, on 64-bit, a 1 MiB request against a
    /// 64 KiB block rounds up to the full block (capacity 65,523, the block
    /// minus its 13-byte header), while a 19,586-byte request against the
    /// same block size rounds *down* to a 16 KiB block (capacity 16,371)
    /// rather than up to 32 KiB, because rounding up there would waste more
    /// than it saves.
    ///
    /// ```
    /// use cord_rs::CordBuffer;
    /// // A request at or above the block size gets the full block.
    /// let big = CordBuffer::with_capacity_and_block_size(1 << 20, 64 << 10);
    /// assert_eq!(big.capacity(), CordBuffer::max_capacity_for(64 << 10));
    /// // A request that falls well short of the block size rounds *down*
    /// // to a smaller power-of-two block rather than up to the full one.
    /// let odd = CordBuffer::with_capacity_and_block_size(19_586, 64 << 10);
    /// assert!(odd.capacity() < CordBuffer::max_capacity_for(64 << 10));
    /// // Small requests are still satisfied precisely.
    /// let small = CordBuffer::with_capacity_and_block_size(3_215, 64 << 10);
    /// assert!(small.capacity() >= 3_215);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `block_size` is not a power of two, or is not greater than
    /// the flat header overhead (13 bytes on 64-bit platforms, 9 on
    /// 32-bit) — the smallest legal `block_size` is 16.
    #[must_use]
    pub fn with_capacity_and_block_size(capacity: usize, block_size: usize) -> Self {
        assert!(
            block_size.is_power_of_two() && block_size > FLAT_OVERHEAD,
            "block_size must be a power of two greater than FLAT_OVERHEAD ({FLAT_OVERHEAD}), got {block_size}"
        );
        let mut capacity = capacity.min(Self::MAX_BLOCK_SIZE);
        let block_size = block_size.min(Self::MAX_BLOCK_SIZE);
        if capacity + FLAT_OVERHEAD >= block_size {
            capacity = block_size;
        } else if capacity <= Self::DEFAULT_MAX_CAPACITY {
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
        Self { rep: Rep { long: Long { rep, _padding: [0; LONG_PADDING] } } }
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
            // SAFETY: `BufReprMut::Flat` is only ever constructed by
            // `Rep::view_mut` wrapping the buffer's flat rep (see its doc),
            // never any other tag.
            BufReprMut::Flat(f) => unsafe { f.flat_data_mut() },
        }
    }

    /// The uninitialized spare capacity (`capacity() - len()` bytes).
    ///
    /// Write data into it, then call [`set_len`](Self::set_len) to mark it
    /// initialized. `std::io::Read::read` needs an already-initialized `&mut
    /// [u8]`, so to fill a buffer from a `Read` either zero this region
    /// first (e.g. `extend(core::iter::repeat_n(0, n))`) or write through
    /// this method and commit with `set_len`, as below.
    ///
    /// ```
    /// use cord_rs::CordBuffer;
    /// let mut buffer = CordBuffer::with_capacity(64);
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
            // SAFETY: see `as_mut_slice`: `BufReprMut::Flat` always wraps a
            // genuine flat rep.
            BufReprMut::Flat(f) => unsafe { f.into_flat_spare_capacity_mut() },
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
                // SAFETY: see `as_mut_slice`: `BufReprMut::Flat` always
                // wraps a genuine flat rep.
                let spare = unsafe { unique.flat_spare_capacity_mut() };
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

impl Clone for CordBuffer {
    /// Returns an independent buffer with the same contents and the same
    /// [`capacity`](Self::capacity) as `self`.
    fn clone(&self) -> Self {
        match self.rep.view() {
            BufRepr::Short(s) => Self { rep: Rep { short: *s } },
            BufRepr::Flat(f) => {
                let rep = flat::new_large(f.capacity());
                // SAFETY: `rep` is a freshly allocated flat rep with a
                // refcount of one (see `flat::new_impl`'s ownership
                // obligation), matching `from_flat`'s contract; that sole
                // reference transfers to the new `CordBuffer`.
                let mut buf = unsafe { Self::from_flat(rep) };
                buf.put_slice(f.data());
                buf
            }
        }
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

impl PartialEq for CordBuffer {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl Eq for CordBuffer {}

impl PartialOrd for CordBuffer {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for CordBuffer {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.as_slice().cmp(other.as_slice())
    }
}

impl Hash for CordBuffer {
    /// Hashes exactly like `<[u8]>::hash`, so `Borrow<[u8]>` is sound for
    /// `HashMap`/`HashSet` lookups keyed by `&[u8]`.
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        <[u8] as Hash>::hash(self.as_slice(), state);
    }
}

impl Borrow<[u8]> for CordBuffer {
    #[inline]
    fn borrow(&self) -> &[u8] {
        self.as_slice()
    }
}

impl BorrowMut<[u8]> for CordBuffer {
    #[inline]
    fn borrow_mut(&mut self) -> &mut [u8] {
        self.as_mut_slice()
    }
}

impl Extend<u8> for CordBuffer {
    /// Writes the yielded bytes into the buffer's spare capacity, like
    /// `Vec`'s `Extend` — but bounded, like `arrayvec::ArrayVec`'s: the
    /// buffer never grows to make room.
    ///
    /// # Panics
    ///
    /// Panics if `iter` yields more bytes than [`available`](Self::available).
    /// Bytes yielded before the overflow are kept (`len()` becomes
    /// `capacity()`).
    #[track_caller]
    fn extend<I: IntoIterator<Item = u8>>(&mut self, iter: I) {
        let len = self.len();
        let available = self.available();
        let mut iter = iter.into_iter();
        let mut written = 0;
        for slot in self.spare_capacity_mut() {
            let Some(byte) = iter.next() else { break };
            slot.write(byte);
            written += 1;
        }
        let overflowed = written == available && iter.next().is_some();
        // SAFETY: the loop above wrote exactly the first `written` bytes of
        // the spare capacity, and nothing else, before this call.
        unsafe { self.set_len(len + written) };
        assert!(
            !overflowed,
            "CordBuffer::extend: iterator yields bytes that exceed the available capacity of {available}"
        );
    }
}

/// Delegates to [`Extend<u8>`](Extend), like `Vec`'s impl.
impl<'a> Extend<&'a u8> for CordBuffer {
    fn extend<I: IntoIterator<Item = &'a u8>>(&mut self, iter: I) {
        self.extend(iter.into_iter().copied());
    }
}

macro_rules! impl_partial_eq_cord_buffer {
    ($($t:ty => |$v:ident| $slice:expr),* $(,)?) => {$(
        impl PartialEq<$t> for CordBuffer {
            #[inline]
            fn eq(&self, other: &$t) -> bool {
                let $v = other;
                self.as_slice() == $slice
            }
        }
        impl PartialEq<CordBuffer> for $t {
            #[inline]
            fn eq(&self, other: &CordBuffer) -> bool {
                let $v = self;
                $slice == other.as_slice()
            }
        }
    )*};
}

impl_partial_eq_cord_buffer! {
    [u8] => |v| v,
    &[u8] => |v| *v,
    Vec<u8> => |v| v.as_slice(),
    str => |v| v.as_bytes(),
    &str => |v| v.as_bytes(),
    String => |v| v.as_bytes(),
}

impl<const N: usize> PartialEq<[u8; N]> for CordBuffer {
    #[inline]
    fn eq(&self, other: &[u8; N]) -> bool {
        self.as_slice() == &other[..]
    }
}

impl<const N: usize> PartialEq<CordBuffer> for [u8; N] {
    #[inline]
    fn eq(&self, other: &CordBuffer) -> bool {
        &self[..] == other.as_slice()
    }
}

impl<const N: usize> PartialEq<&[u8; N]> for CordBuffer {
    #[inline]
    fn eq(&self, other: &&[u8; N]) -> bool {
        self.as_slice() == &other[..]
    }
}

impl<const N: usize> PartialEq<CordBuffer> for &[u8; N] {
    #[inline]
    fn eq(&self, other: &CordBuffer) -> bool {
        &self[..] == other.as_slice()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout() {
        assert_eq!(core::mem::size_of::<CordBuffer>(), 16);
        assert_eq!(INLINE_CAPACITY, 15);
        let b = CordBuffer::new();
        assert!(b.rep.is_short());
        assert_eq!(b.capacity(), INLINE_CAPACITY);
        assert_eq!(b.len(), 0);
    }

    #[test]
    fn inline_and_heap() {
        let mut b = CordBuffer::with_capacity(INLINE_CAPACITY);
        assert!(b.rep.is_short());
        b.put_slice(b"abc");
        assert_eq!(&*b, b"abc");
        assert_eq!(b.available(), INLINE_CAPACITY - 3);

        let mut b = CordBuffer::with_capacity(INLINE_CAPACITY + 1);
        assert!(!b.rep.is_short());
        assert!(b.capacity() > INLINE_CAPACITY);
        assert!(b.capacity() <= CordBuffer::DEFAULT_MAX_CAPACITY);
        let n = b.put_slice_partial(&[1u8; 10_000]);
        assert_eq!(n, b.capacity());
        assert_eq!(b.available(), 0);
        b.truncate(3);
        assert_eq!(b.len(), 3);
        b.clear();
        assert!(b.is_empty());
    }

    #[test]
    fn block_size_sizes() {
        let b = CordBuffer::with_capacity_and_block_size(64 << 10, 64 << 10);
        assert_eq!(b.capacity(), (64 << 10) - FLAT_OVERHEAD);
        let b = CordBuffer::with_capacity_and_block_size(100, 64 << 10);
        assert!(b.capacity() >= 100);
        let b = CordBuffer::with_capacity_and_block_size(19_586, 64 << 10);
        assert_eq!(b.capacity(), (16 << 10) - FLAT_OVERHEAD);
        let b = CordBuffer::with_capacity_and_block_size(1 << 20, 64 << 10);
        assert_eq!(b.capacity(), (64 << 10) - FLAT_OVERHEAD);
        assert_eq!(CordBuffer::max_capacity_for(8 << 10), (8 << 10) - FLAT_OVERHEAD);
        assert_eq!(CordBuffer::max_capacity_for(1 << 20), CordBuffer::MAX_BLOCK_SIZE - FLAT_OVERHEAD);
    }

    #[test]
    #[should_panic(expected = "power of two")]
    fn block_size_rejects_non_pow2() {
        let _ = CordBuffer::with_capacity_and_block_size(10, 1000);
    }

    #[test]
    #[should_panic(expected = "power of two")]
    fn block_size_rejects_below_overhead() {
        // The largest power of two `<= FLAT_OVERHEAD` is always illegal
        // (this used to underflow `capacity - FLAT_OVERHEAD` instead of
        // panicking cleanly).
        let below = (FLAT_OVERHEAD + 1).next_power_of_two() / 2;
        let _ = CordBuffer::with_capacity_and_block_size(10, below);
    }

    #[test]
    #[should_panic(expected = "power of two")]
    fn max_capacity_for_rejects_block_size_below_overhead() {
        let below = (FLAT_OVERHEAD + 1).next_power_of_two() / 2;
        let _ = CordBuffer::max_capacity_for(below);
    }

    #[test]
    fn smallest_block_size_is_sane() {
        // 16 is the smallest power of two greater than FLAT_OVERHEAD on
        // both 64-bit (13) and 32-bit (9) platforms, so it is the smallest
        // legal `block_size`.
        let smallest = (FLAT_OVERHEAD + 1).next_power_of_two();
        assert_eq!(smallest, 16);
        let b = CordBuffer::with_capacity_and_block_size(smallest, smallest);
        // `flat::new_large` floors the payload at `MIN_FLAT_LENGTH`, so the
        // resulting capacity is sane (no underflow, no giant allocation).
        assert!(b.capacity() >= MIN_FLAT_LENGTH);
        assert!(b.capacity() < smallest + MIN_FLAT_LENGTH);
    }

    #[test]
    fn max_capacity_for_matches_actual_capacity() {
        for block_size in [16, 32, 64, 8 << 10, 64 << 10, 1 << 20] {
            let b = CordBuffer::with_capacity_and_block_size(block_size, block_size);
            assert_eq!(
                CordBuffer::max_capacity_for(block_size),
                b.capacity(),
                "max_capacity_for({block_size}) disagrees with an actual allocation"
            );
        }
    }

    #[test]
    #[should_panic(expected = "exceed the available capacity")]
    fn put_slice_overflow_panics() {
        let mut b = CordBuffer::new();
        b.put_slice(&[0u8; 100]);
    }
}
