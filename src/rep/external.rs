//! External reps: nodes referencing separately owned byte storage.
//!
//! General owners (`Arc`, `bytes::Bytes`, static data, ...) are stored beside
//! the external header, mirroring abseil's `CordRepExternalImpl<Releaser>`.
//! Standard global byte allocations (`Vec<u8>`, `String`, `Box<[u8]>`) use a
//! compact node containing only the allocation size and share one releaser.
//! When the reference count drops to zero, both storage and node are released.

use core::alloc::Layout;
use core::marker::PhantomData;
use core::mem::ManuallyDrop;
use core::ptr::NonNull;

use super::{CordRep, EXTERNAL, RepPtr};

/// Function that releases the referenced storage and frees its external node.
type ReleaserInvoker = unsafe fn(*mut CordRepExternal);

/// Header shared by both generic-owner and global-allocation external nodes.
#[repr(C)]
pub(crate) struct CordRepExternal {
    pub(crate) rep: CordRep,
    /// Start of the referenced bytes.
    pub(crate) base: *const u8,
    /// Knows how to release the storage and deallocate the node.
    releaser_invoker: ReleaserInvoker,
}

#[repr(C)]
struct CordRepExternalImpl<O> {
    ext: CordRepExternal,
    owner: O,
}

/// External node owning a raw allocation made by Rust's global allocator.
#[repr(C)]
struct CordRepExternalGlobal {
    ext: CordRepExternal,
    allocation_size: usize,
}

/// Size used for memory accounting of an external node (abseil similarly
/// accounts `sizeof(CordRepExternalImpl<intptr_t>)`). This is exact for the
/// compact global node and remains approximate for arbitrary generic owners.
pub(crate) const EXTERNAL_REP_SIZE: usize = core::mem::size_of::<CordRepExternalGlobal>();

/// Trait implemented by values that own a stable byte buffer a cord may
/// reference without copying.
///
/// # Safety
///
/// `as_bytes` must return the same pointer and length for the whole lifetime
/// of the value, including after the value is moved (i.e. the bytes must live
/// on the heap or in static memory, not inline in the value).
pub(crate) unsafe trait StableBytes: Send + Sync + 'static {
    fn as_bytes(&self) -> &[u8];
}

unsafe impl StableBytes for std::sync::Arc<[u8]> {
    #[inline]
    fn as_bytes(&self) -> &[u8] {
        self
    }
}
unsafe impl StableBytes for std::sync::Arc<Vec<u8>> {
    #[inline]
    fn as_bytes(&self) -> &[u8] {
        self
    }
}
unsafe impl StableBytes for std::sync::Arc<str> {
    #[inline]
    fn as_bytes(&self) -> &[u8] {
        str::as_bytes(self)
    }
}
unsafe impl StableBytes for std::sync::Arc<String> {
    #[inline]
    fn as_bytes(&self) -> &[u8] {
        String::as_bytes(self)
    }
}
unsafe impl StableBytes for &'static [u8] {
    #[inline]
    fn as_bytes(&self) -> &[u8] {
        self
    }
}
unsafe impl StableBytes for &'static str {
    #[inline]
    fn as_bytes(&self) -> &[u8] {
        str::as_bytes(self)
    }
}
#[cfg(feature = "bytes")]
unsafe impl StableBytes for bytes::Bytes {
    #[inline]
    fn as_bytes(&self) -> &[u8] {
        self
    }
}

/// Raw parts of a byte buffer allocated by Rust's global allocator.
pub(crate) struct RawGlobalParts {
    base: NonNull<u8>,
    len: usize,
    allocation_size: usize,
}

/// An owned byte buffer whose allocation can be released with
/// `dealloc(base, Layout::from_size_align(allocation_size, 1))`.
///
/// # Safety
///
/// `allocation_size` must return the exact non-zero layout size used for the
/// allocation whenever the buffer is non-empty, and that allocation must have
/// one-byte alignment. `into_raw_parts` must
/// transfer sole ownership of that live global allocation, preserve its
/// pointer provenance, return the same length and allocation size, and not
/// unwind after disarming the owner. The returned length must not exceed the
/// allocation size. No element destruction may be required.
pub(crate) unsafe trait GlobalBytes: StableBytes {
    fn allocation_size(&self) -> usize;

    /// Transfers the allocation out of `self`.
    ///
    /// # Safety
    ///
    /// The caller must eventually deallocate the returned allocation using
    /// the layout described by this trait's safety contract.
    unsafe fn into_raw_parts(self) -> RawGlobalParts;
}

unsafe impl StableBytes for Vec<u8> {
    #[inline]
    fn as_bytes(&self) -> &[u8] {
        self
    }
}

unsafe impl GlobalBytes for Vec<u8> {
    #[inline]
    fn allocation_size(&self) -> usize {
        self.capacity()
    }

    unsafe fn into_raw_parts(self) -> RawGlobalParts {
        let mut owner = ManuallyDrop::new(self);
        RawGlobalParts {
            // SAFETY: a Vec pointer is always non-null, including for an
            // empty vector; ownership remains disarmed in `owner`.
            base: unsafe { NonNull::new_unchecked(owner.as_mut_ptr()) },
            len: owner.len(),
            allocation_size: owner.capacity(),
        }
    }
}

unsafe impl StableBytes for String {
    #[inline]
    fn as_bytes(&self) -> &[u8] {
        self.as_bytes()
    }
}

unsafe impl GlobalBytes for String {
    #[inline]
    fn allocation_size(&self) -> usize {
        self.capacity()
    }

    #[inline]
    unsafe fn into_raw_parts(self) -> RawGlobalParts {
        // `String::into_bytes` preserves the allocation and cannot unwind.
        unsafe { GlobalBytes::into_raw_parts(self.into_bytes()) }
    }
}

unsafe impl StableBytes for Box<[u8]> {
    #[inline]
    fn as_bytes(&self) -> &[u8] {
        self
    }
}

unsafe impl GlobalBytes for Box<[u8]> {
    #[inline]
    fn allocation_size(&self) -> usize {
        self.len()
    }

    unsafe fn into_raw_parts(self) -> RawGlobalParts {
        let len = self.len();
        let base = Box::into_raw(self).cast::<u8>();
        RawGlobalParts {
            // SAFETY: Box allocations are non-null, including empty slices;
            // ownership was transferred by `Box::into_raw` above.
            base: unsafe { NonNull::new_unchecked(base) },
            len,
            allocation_size: len,
        }
    }
}

impl CordRepExternal {
    /// Creates an external rep referencing the bytes of `owner`.
    ///
    /// Requires `!owner.as_bytes().is_empty()`. The owner is dropped when
    /// the rep is destroyed.
    pub(crate) fn create<O: StableBytes>(owner: O) -> *mut CordRep {
        let length = owner.as_bytes().len();
        debug_assert!(length > 0);
        let node = Box::into_raw(Box::new(CordRepExternalImpl {
            ext: CordRepExternal {
                rep: CordRep::new(length, EXTERNAL),
                base: core::ptr::null(),
                releaser_invoker: release::<O>,
            },
            owner,
        }));
        // Derive `base` only now that the owner is at its final address: a
        // pointer taken from a `Box<[u8]>` owner *before* the box was moved
        // into the node would be invalidated by the move (Stacked Borrows
        // treats a moved `Box` like a fresh `&mut`).
        // SAFETY: `node` is a valid, exclusively owned allocation.
        unsafe {
            let base = (*node).owner.as_bytes().as_ptr();
            (*node).ext.base = base;
        }
        node.cast()
    }

    /// Creates an external rep owning a standard global byte allocation.
    ///
    /// The payload and node are deallocated when the rep is destroyed.
    ///
    /// # Safety
    ///
    /// `owner.as_bytes()` must be non-empty, so it represents a live
    /// allocation rather than a dangling zero-capacity sentinel.
    pub(crate) unsafe fn create_global<O: GlobalBytes>(owner: O) -> *mut CordRep {
        let length = owner.as_bytes().len();
        let allocation_size = owner.allocation_size();
        debug_assert!(length > 0);
        debug_assert!(length <= allocation_size);

        // Allocate the metadata while `owner` still provides panic-safe
        // ownership of the payload. Everything after `into_raw_parts` is
        // infallible field assignment and ownership transfer.
        let mut node = Box::new(CordRepExternalGlobal {
            ext: CordRepExternal {
                rep: CordRep::new(length, EXTERNAL),
                base: core::ptr::null(),
                releaser_invoker: release_global,
            },
            allocation_size,
        });
        // SAFETY: this function assumes responsibility for releasing the
        // allocation through `release_global` below.
        let raw = unsafe { owner.into_raw_parts() };
        node.ext.rep.length = raw.len;
        node.ext.base = raw.base.as_ptr();
        node.allocation_size = raw.allocation_size;
        Box::into_raw(node).cast()
    }

    /// Drops the owner and deallocates the rep. Requires `rep.is_external()`.
    ///
    /// # Safety
    ///
    /// `rep` must be a non-null pointer to a live external rep (tag ==
    /// `EXTERNAL`) originally produced by [`create`](Self::create) or
    /// [`create_global`](Self::create_global), whose reference count has just
    /// reached zero, transferring final ownership to this call; `rep` must
    /// not be used again afterwards.
    #[inline]
    pub(crate) unsafe fn delete(rep: *mut CordRep) {
        debug_assert!(unsafe { rep.is_external() });
        let ext: *mut CordRepExternal = rep.cast();
        // SAFETY: `ext` is `rep` reinterpreted as its actual concrete type
        // (sound because `rep`'s EXTERNAL tag guarantees it really is a
        // `CordRepExternal` header, per this fn's contract), so its
        // `releaser_invoker` field may be read. The matching constructor set
        // it to a function whose concrete-node contract is satisfied by this
        // fn's final-ownership contract on `rep`.
        unsafe { ((*ext).releaser_invoker)(ext) }
    }
}

/// Type-erased `ReleaserInvoker` used as `CordRepExternalImpl<O>::ext`'s
/// `releaser_invoker`: reconstructs the owning `Box` and drops it, running
/// `O`'s destructor and freeing the node.
///
/// # Safety
///
/// `ext` must be a non-null pointer to the `ext` field of a live, uniquely
/// owned `CordRepExternalImpl<O>` (i.e. the same `O` this fn was
/// monomorphized for by [`CordRepExternal::create`]) originally obtained
/// from `Box::into_raw`, whose reference count has just reached zero; `ext`
/// must not be used again afterwards.
unsafe fn release<O>(ext: *mut CordRepExternal) {
    // SAFETY: `ext` is the live, uniquely owned `ext` field of a
    // `CordRepExternalImpl<O>` box per this fn's contract, so casting back
    // to the enclosing `CordRepExternalImpl<O>` (repr(C), `ext` is the first
    // field) and reconstructing the `Box` recovers exactly the allocation
    // `Box::into_raw` produced in `create`; dropping it runs `O`'s
    // destructor and frees the node.
    unsafe { drop(Box::from_raw(ext.cast::<CordRepExternalImpl<O>>())) }
}

/// Releases a payload produced by [`GlobalBytes`] and its metadata node.
///
/// # Safety
///
/// `ext` must point to a uniquely owned live `CordRepExternalGlobal` whose
/// reference count has reached zero.
unsafe fn release_global(ext: *mut CordRepExternal) {
    let node = ext.cast::<CordRepExternalGlobal>();
    // Copy everything needed before either allocation is released.
    let base = unsafe { (*ext).base.cast_mut() };
    let allocation_size = unsafe { (*node).allocation_size };
    // SAFETY: `GlobalBytes` guarantees this is the exact non-zero layout of
    // the live payload allocation and final refcount ownership guarantees no
    // payload references remain. Deallocate the payload before the metadata,
    // matching normal owner-field drop order.
    unsafe {
        std::alloc::dealloc(base, Layout::from_size_align_unchecked(allocation_size, 1));
        drop(Box::from_raw(node));
    }
}

/// Copy handle borrowing a live external rep for `'a`.
///
/// # Invariant
///
/// The wrapped pointer is non-null and points to a live external rep (tag
/// == `EXTERNAL`) that is not mutated — other than through its
/// interior-mutable refcount — for the duration of `'a` (this is what lets
/// [`data`](Self::data) hand out a `&'a [u8]`). Established once, at the
/// sole constructor [`from_raw`](Self::from_raw).
#[derive(Clone, Copy)]
pub(crate) struct ExternalRef<'a> {
    ptr: NonNull<CordRepExternal>,
    _marker: PhantomData<&'a CordRepExternal>,
}

impl<'a> ExternalRef<'a> {
    /// Wraps `ptr` as an external handle borrowed for `'a`.
    ///
    /// # Safety
    ///
    /// `ptr` must be non-null and point to a live external rep that the
    /// caller guarantees stays live, and unmutated other than
    /// through its interior-mutable refcount, for `'a`.
    #[inline]
    pub(crate) unsafe fn from_raw(ptr: *mut CordRepExternal) -> Self {
        debug_assert!(!ptr.is_null());
        // SAFETY: non-null per the debug_assert above.
        Self { ptr: unsafe { NonNull::new_unchecked(ptr) }, _marker: PhantomData }
    }

    /// This external node's `length`.
    #[inline]
    pub(crate) fn len(self) -> usize {
        // SAFETY: `self`'s invariant (struct doc) guarantees `self.ptr` is a
        // live external rep for the call's duration; this only borrows the
        // embedded `CordRep.length` field, not the whole header.
        unsafe { (*self.ptr.as_ptr()).rep.length }
    }

    /// The referenced bytes.
    #[inline]
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "API completeness for ExternalRef (RepRef::data is the production path for \
                      external data edges), exercised only by this module's tests"
        )
    )]
    pub(crate) fn data(self) -> &'a [u8] {
        let len = self.len();
        // SAFETY: `self`'s invariant makes `self.ptr` a live external rep
        // for `'a`. Generic nodes derive `base` from a final-position owner
        // whose `StableBytes` contract keeps it valid; global nodes retain
        // sole ownership of the live raw allocation. Both keep `len` bytes
        // readable for as long as the node itself, i.e. `'a`.
        unsafe { core::slice::from_raw_parts((*self.ptr.as_ptr()).base, len) }
    }

    /// The size this external node contributes to memory-usage accounting:
    /// the fixed per-node overhead ([`EXTERNAL_REP_SIZE`]) plus the
    /// referenced length.
    #[inline]
    pub(crate) fn allocated_size(self) -> usize {
        self.len() + EXTERNAL_REP_SIZE
    }

    /// Escape hatch to the raw pointer: a permanent, intentional interop
    /// point with the raw surgery layer, not a stopgap pending conversion to
    /// the handle types.
    #[inline]
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "API completeness for ExternalRef, exercised only by this module's tests"
        )
    )]
    pub(crate) fn as_ptr(self) -> *mut CordRepExternal {
        self.ptr.as_ptr()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct DropCounter(Vec<u8>, Arc<AtomicUsize>);
    unsafe impl StableBytes for DropCounter {
        fn as_bytes(&self) -> &[u8] {
            &self.0
        }
    }
    impl Drop for DropCounter {
        fn drop(&mut self) {
            self.1.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn external_rep_drops_owner_once() {
        let drops = Arc::new(AtomicUsize::new(0));
        let owner = DropCounter(b"external data".to_vec(), drops.clone());
        unsafe {
            let rep = CordRepExternal::create(owner);
            assert!(rep.is_external());
            assert_eq!(rep.length(), 13);
            assert_eq!(super::super::edge_data(rep), b"external data");
            super::super::ref_rep(rep);
            super::super::unref(rep);
            assert_eq!(drops.load(Ordering::SeqCst), 0);
            super::super::unref(rep);
            assert_eq!(drops.load(Ordering::SeqCst), 1);
        }
    }

    #[test]
    fn generic_and_global_owners() {
        unsafe {
            let rep = CordRepExternal::create(b"static bytes" as &'static [u8]);
            assert_eq!(super::super::edge_data(rep), b"static bytes");
            super::super::unref(rep);

            let boxed: Box<[u8]> = vec![7u8; 10_000].into_boxed_slice();
            let rep = CordRepExternal::create_global(boxed);
            assert_eq!(super::super::edge_data(rep).len(), 10_000);
            assert!(super::super::edge_data(rep).iter().all(|&b| b == 7));
            super::super::unref(rep);

            let string = String::from("owned string data");
            let rep = CordRepExternal::create_global(string);
            assert_eq!(super::super::edge_data(rep), b"owned string data");
            super::super::unref(rep);

            let arc: Arc<[u8]> = Arc::from(&b"arc bytes"[..]);
            let rep = CordRepExternal::create(arc.clone());
            assert_eq!(Arc::strong_count(&arc), 2);
            super::super::unref(rep);
            assert_eq!(Arc::strong_count(&arc), 1);
        }
    }

    #[test]
    fn global_owners_preserve_payload_pointer() {
        unsafe {
            let vec = vec![1u8; 4096];
            let expected = vec.as_ptr();
            let rep = CordRepExternal::create_global(vec);
            assert_eq!(super::super::edge_data(rep).as_ptr(), expected);
            super::super::unref(rep);

            let string = "x".repeat(4096);
            let expected = string.as_ptr();
            let rep = CordRepExternal::create_global(string);
            assert_eq!(super::super::edge_data(rep).as_ptr(), expected);
            super::super::unref(rep);

            let boxed = vec![2u8; 4096].into_boxed_slice();
            let expected = boxed.as_ptr();
            let rep = CordRepExternal::create_global(boxed);
            assert_eq!(super::super::edge_data(rep).as_ptr(), expected);
            super::super::unref(rep);
        }
    }

    #[test]
    fn global_node_matches_accounted_size() {
        assert_eq!(EXTERNAL_REP_SIZE, core::mem::size_of::<CordRepExternalGlobal>());
        #[cfg(target_pointer_width = "64")]
        assert_eq!(EXTERNAL_REP_SIZE, 40);
    }
}
