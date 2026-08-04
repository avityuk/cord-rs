//! Forward reader over the data edges of a btree, tracking remaining bytes.
//!
//! Port of abseil's `cord_rep_btree_reader.{h,cc}`.

use core::ptr::NonNull;

use super::btree::{BtreePtr, CordRepBtree};
use super::navigator::CordRepBtreeNavigator;
use super::{CordRep, RepPtr, edge_data};

/// See the [module documentation](self).
#[derive(Clone, Copy, Default)]
pub(crate) struct CordRepBtreeReader {
    /// Bytes remaining after the end of the last returned chunk.
    remaining: usize,
    navigator: CordRepBtreeNavigator,
}

impl CordRepBtreeReader {
    /// An empty reader.
    pub(crate) const fn new() -> Self {
        Self { remaining: 0, navigator: CordRepBtreeNavigator::new() }
    }

    /// Returns `true` if not empty.
    #[inline]
    pub(crate) fn is_some(&self) -> bool {
        self.navigator.is_some()
    }

    /// The tree referenced, or null if empty.
    #[inline]
    pub(crate) fn btree(&self) -> *mut CordRepBtree {
        self.navigator.btree()
    }

    /// The current data edge. Requires a non-empty reader.
    ///
    /// # Safety
    ///
    /// The reader must be non-empty (`self.is_some()`, i.e. previously
    /// positioned by [`init`](Self::init)) and the tree it was initialized
    /// with must still be live and unmutated: this type stores raw pointers
    /// into the tree without owning a reference on it, so the caller is
    /// responsible for keeping it alive for as long as the reader is used.
    #[inline]
    pub(crate) unsafe fn node(&self) -> *mut CordRep {
        unsafe { self.navigator.current() }
    }

    /// Length of the referenced tree. Requires a non-empty reader.
    ///
    /// # Safety
    ///
    /// Same contract as [`node`](Self::node).
    #[inline]
    pub(crate) unsafe fn length(&self) -> usize {
        unsafe {
            // SAFETY: the contract above guarantees `self.btree()` is a live
            // btree node, so reading its `.length()` is sound.
            debug_assert!(!self.btree().is_null());
            self.btree().length()
        }
    }

    /// Bytes remaining after the last returned chunk. Zero after the last
    /// edge was returned (further `next` / `skip` calls return empty).
    #[inline]
    pub(crate) fn remaining(&self) -> usize {
        self.remaining
    }

    /// Resets to an empty value.
    #[inline]
    pub(crate) fn reset(&mut self) {
        self.navigator.reset();
    }

    /// Initializes with `tree`, returning its first data edge.
    ///
    /// # Safety
    ///
    /// `tree` must be non-null, point to a live `CordRepBtree` with
    /// `size() > 0` and `height() <= MAX_HEIGHT` (see
    /// [`CordRepBtreeNavigator::init_first`]). Does not adopt a reference on
    /// `tree`. The returned slice's lifetime `'a` is not tied to any actual
    /// borrow: the caller must not use it beyond the scope in which `tree`
    /// stays live and unmutated.
    #[inline]
    pub(crate) unsafe fn init<'a>(&mut self, tree: *mut CordRepBtree) -> &'a [u8] {
        unsafe {
            // SAFETY: `tree`'s validity is the caller's contract above;
            // `init_first` and `edge_data` each require exactly that (a live
            // tree / a live data edge respectively), which the navigator's own
            // invariants (see `navigator.rs`) guarantee the returned edge to be.
            debug_assert!(!tree.is_null());
            let edge = self.navigator.init_first(tree);
            self.remaining = tree.length() - edge.length();
            edge_data(edge)
        }
    }

    /// Navigates to and returns the next data edge, or an empty slice at EOF.
    ///
    /// # Safety
    ///
    /// Same contract as [`node`](Self::node); the returned slice's lifetime
    /// is unbound in the same way as [`init`](Self::init)'s.
    #[inline]
    pub(crate) unsafe fn next<'a>(&mut self) -> &'a [u8] {
        unsafe {
            // SAFETY: per the contract above the reader is positioned on a live
            // tree, so `self.navigator.next()` and `edge_data` on its
            // non-null result are sound.
            if self.remaining == 0 {
                return &[];
            }
            let edge = self.navigator.next();
            debug_assert!(!edge.is_null());
            self.remaining -= edge.length();
            edge_data(edge)
        }
    }

    /// Skips `skip` bytes past the end of the current chunk and returns the
    /// data directly following them.
    ///
    /// # Safety
    ///
    /// Same contract as [`node`](Self::node); the returned slice's lifetime
    /// is unbound in the same way as [`init`](Self::init)'s.
    #[inline]
    pub(crate) unsafe fn skip<'a>(&mut self, skip: usize) -> &'a [u8] {
        unsafe {
            // SAFETY: per the contract above the reader is positioned on a live
            // tree, so `self.navigator.current()`/`.skip()` and `edge_data` on
            // a non-null result are sound.
            // We are positioned on the last consumed edge, so skip it too.
            let edge_length = self.navigator.current().length();
            let pos = self.navigator.skip(skip + edge_length);
            let Some(edge) = pos.edge else {
                core::hint::cold_path();
                self.remaining = 0;
                return &[];
            };
            let edge = edge.as_ptr();
            // All edges skipped before `pos.edge` (`skip - pos.offset` bytes) are
            // consumed, as is the current edge.
            self.remaining -= skip - pos.offset + edge.length();
            &edge_data(edge)[pos.offset..]
        }
    }

    /// Navigates to the chunk containing `offset` and returns the data from
    /// `offset` to the end of that chunk, or empty if `offset >= length`.
    ///
    /// # Safety
    ///
    /// Same contract as [`node`](Self::node); the returned slice's lifetime
    /// is unbound in the same way as [`init`](Self::init)'s.
    #[inline]
    pub(crate) unsafe fn seek<'a>(&mut self, offset: usize) -> &'a [u8] {
        unsafe {
            // SAFETY: per the contract above the reader is positioned on a live
            // tree, so `self.navigator.seek()`, `edge_data`, and `self.length()`
            // are all sound to call.
            let pos = self.navigator.seek(offset);
            let Some(edge) = pos.edge else {
                core::hint::cold_path();
                self.remaining = 0;
                return &[];
            };
            let chunk = &edge_data(edge.as_ptr())[pos.offset..];
            self.remaining = self.length() - offset - chunk.len();
            chunk
        }
    }

    /// Reads `n` bytes into a new tree. If `chunk_size` is zero the read
    /// starts at the next data edge, else at the last `chunk_size` bytes of
    /// the last returned edge. Returns the remaining data of the edge the
    /// read ended in (empty if all data was read) and the tree (null if `n`
    /// exceeded the remaining data).
    ///
    /// # Safety
    ///
    /// Same contract as [`node`](Self::node), plus `chunk_size <=
    /// self.navigator.current().length()` (checked below). The returned
    /// tree, if non-null, carries a reference the caller now owns; the
    /// returned slice's lifetime is unbound in the same way as
    /// [`init`](Self::init)'s.
    pub(crate) unsafe fn read<'a>(&mut self, n: usize, chunk_size: usize) -> (&'a [u8], *mut CordRep) {
        unsafe {
            // SAFETY: per the contract above the reader is positioned on a live
            // tree, so `self.navigator.current()`/`.next()`/`.read()` and
            // `edge_data` on their results are sound.
            debug_assert!(chunk_size <= self.navigator.current().length());

            // Start inside the last returned edge, or at the next edge.
            let mut edge = if chunk_size != 0 { self.navigator.current() } else { self.navigator.next() };
            let offset = if chunk_size != 0 { edge.length() - chunk_size } else { 0 };

            let result = self.navigator.read(offset, n);
            // `read`'s own return type stays a raw pointer this phase (its
            // sole caller, `iter.rs`, is converted in the next phase); thread
            // `Option<NonNull>` internally and drop to a raw pointer only at
            // this boundary.
            let tree = result.tree.map_or(core::ptr::null_mut(), NonNull::as_ptr);

            // If the data was covered entirely by `chunk_size` we did not consume
            // any additional data and directly return the rest of the edge.
            if n < chunk_size {
                return (&edge_data(edge)[result.n..], tree);
            }

            // `chunk_size` bytes were taken from the last edge and `result.n` is
            // the offset into the current edge trailing the read data. The read
            // may have consumed all remaining data, so check before calling
            // `current()`.
            let consumed_by_read = n - chunk_size - result.n;
            if consumed_by_read >= self.remaining {
                self.remaining = 0;
                return (&[], tree);
            }

            edge = self.navigator.current();
            self.remaining -= consumed_by_read + edge.length();
            (&edge_data(edge)[result.n..], tree)
        }
    }
}
