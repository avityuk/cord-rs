//! External reps: nodes referencing memory owned by a user supplied value.
//!
//! An external rep is allocated together with the value that owns the bytes
//! (a `Vec<u8>`, `Arc<[u8]>`, `bytes::Bytes`, `()` for `&'static` data, ...)
//! in a single allocation, mirroring abseil's `CordRepExternalImpl<Releaser>`.
//! When the reference count drops to zero the owner is dropped, releasing the
//! memory.

use super::{CordRep, EXTERNAL, RepPtr};

/// Function that drops the owner and frees the `CordRepExternalImpl`.
type ReleaserInvoker = unsafe fn(*mut CordRepExternal);

/// Header of an external rep. The owner value follows in memory (see
/// [`CordRepExternalImpl`]).
#[repr(C)]
pub(crate) struct CordRepExternal {
    pub(crate) rep: CordRep,
    /// Start of the referenced bytes.
    pub(crate) base: *const u8,
    /// Knows how to drop the owner and deallocate the node.
    releaser_invoker: ReleaserInvoker,
}

#[repr(C)]
struct CordRepExternalImpl<O> {
    ext: CordRepExternal,
    owner: O,
}

/// Size used for memory accounting of an external node (abseil uses
/// `sizeof(CordRepExternalImpl<intptr_t>)`).
pub(crate) const EXTERNAL_REP_SIZE: usize = core::mem::size_of::<CordRepExternalImpl<usize>>();

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

unsafe impl StableBytes for Vec<u8> {
    #[inline]
    fn as_bytes(&self) -> &[u8] {
        self
    }
}
unsafe impl StableBytes for Box<[u8]> {
    #[inline]
    fn as_bytes(&self) -> &[u8] {
        self
    }
}
unsafe impl StableBytes for String {
    #[inline]
    fn as_bytes(&self) -> &[u8] {
        self.as_bytes()
    }
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

    /// Drops the owner and deallocates the rep. Requires `rep.is_external()`.
    ///
    /// # Safety
    ///
    /// `rep` must be a non-null pointer to a live external rep (tag ==
    /// `EXTERNAL`) originally produced by [`create`](Self::create), whose
    /// reference count has just reached zero, transferring final ownership
    /// to this call; `rep` must not be used again afterwards.
    #[inline]
    pub(crate) unsafe fn delete(rep: *mut CordRep) {
        debug_assert!(unsafe { rep.is_external() });
        let ext: *mut CordRepExternal = rep.cast();
        // SAFETY: `ext` is `rep` reinterpreted as its actual concrete type
        // (sound because `rep`'s EXTERNAL tag guarantees it really is a
        // `CordRepExternal` header, per this fn's contract), so its
        // `releaser_invoker` field may be read. `releaser_invoker` was set
        // by `create::<O>` to `release::<O>`, whose own contract (that `ext`
        // points at a live `CordRepExternalImpl<O>` with a refcount of zero)
        // is satisfied by this fn's contract on `rep`.
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
    fn static_and_arc_owners() {
        unsafe {
            let rep = CordRepExternal::create(b"static bytes" as &'static [u8]);
            assert_eq!(super::super::edge_data(rep), b"static bytes");
            super::super::unref(rep);

            let boxed: Box<[u8]> = vec![7u8; 10_000].into_boxed_slice();
            let rep = CordRepExternal::create(boxed);
            assert_eq!(super::super::edge_data(rep).len(), 10_000);
            assert!(super::super::edge_data(rep).iter().all(|&b| b == 7));
            super::super::unref(rep);

            let string = String::from("owned string data");
            let rep = CordRepExternal::create(string);
            assert_eq!(super::super::edge_data(rep), b"owned string data");
            super::super::unref(rep);

            let arc: Arc<[u8]> = Arc::from(&b"arc bytes"[..]);
            let rep = CordRepExternal::create(arc.clone());
            assert_eq!(Arc::strong_count(&arc), 2);
            super::super::unref(rep);
            assert_eq!(Arc::strong_count(&arc), 1);
        }
    }
}
