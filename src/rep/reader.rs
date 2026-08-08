//! Forward reader over the data edges of a btree, tracking remaining bytes.
//!
//! Port of abseil's `cord_rep_btree_reader.{h,cc}`.

use core::marker::PhantomData;

use super::btree::{BtreePtr, BtreeRef, CordRepBtree};
use super::navigator::CordRepBtreeNavigator;
use super::{OwnedRep, RepPtr, edge_data};

/// See the [module documentation](self). Borrows the `CordRepBtree` it is
/// [`init`](Self::init)ialized with for `'a`: that borrow is this type's
/// liveness invariant, established at `init`. Methods that navigate from
/// the current position additionally require a non-empty reader (their
/// `# Safety` sections) — a default reader's internal path is dangling,
/// which is why they are `unsafe`.
#[derive(Clone, Copy, Default)]
pub(crate) struct CordRepBtreeReader<'a> {
    /// Bytes remaining after the end of the last returned chunk.
    remaining: usize,
    navigator: CordRepBtreeNavigator,
    _marker: PhantomData<&'a CordRepBtree>,
}

// SAFETY: a reader only ever reads through the raw pointers it stores
// internally (never exposing interior mutability beyond a live rep's own
// atomic refcount, same as `RepRef`/`BtreeRef`), and its invariant (struct
// doc) requires the tree they reference to be live and unmutated for `'a` —
// exactly the condition under which sharing a read-only view across threads
// is sound. Needed so `Chunks` (iter.rs), which holds a reader field, can
// derive `Send`/`Sync` instead of asserting them manually.
unsafe impl Send for CordRepBtreeReader<'_> {}
// SAFETY: see above.
unsafe impl Sync for CordRepBtreeReader<'_> {}

impl<'a> CordRepBtreeReader<'a> {
    /// An empty reader.
    pub(crate) const fn new() -> Self {
        Self { remaining: 0, navigator: CordRepBtreeNavigator::new(), _marker: PhantomData }
    }

    /// Returns `true` if not empty.
    #[inline]
    pub(crate) fn is_some(&self) -> bool {
        self.navigator.is_some()
    }

    /// Length of the referenced tree.
    ///
    /// # Safety
    ///
    /// The reader must be non-empty (`is_some()`): on a default reader
    /// `navigator.btree()` is null.
    #[inline]
    pub(crate) unsafe fn length(&self) -> usize {
        debug_assert!(self.is_some());
        // SAFETY: non-empty per this fn's precondition, so the navigator's
        // stored tree pointer refers to a live, well-formed `CordRepBtree`
        // for `'a` per `init`'s contract.
        unsafe { BtreeRef::from_raw(self.navigator.btree()).len() }
    }

    /// Bytes remaining after the last returned chunk. Zero after the last
    /// edge was returned (further `next` / `skip` calls return empty).
    #[inline]
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "API completeness for CordRepBtreeReader, exercised only by tests")
    )]
    pub(crate) fn remaining(&self) -> usize {
        self.remaining
    }

    /// Initializes with `tree`, returning its first data edge. This is the
    /// sole place `self`'s `'a` liveness invariant (struct doc) is
    /// established: `tree`'s own invariant (a live, well-formed
    /// `CordRepBtree` for `'a`) is exactly what every other method here
    /// relies on afterward.
    #[inline]
    pub(crate) fn init(&mut self, tree: BtreeRef<'a>) -> &'a [u8] {
        let ptr = tree.as_ptr();
        // SAFETY: `tree`'s invariant guarantees `ptr` is a live,
        // well-formed `CordRepBtree` with `size() > 0` (a `BtreeRef` can
        // only wrap a well-formed node) for `'a`, matching
        // `init_first`/`edge_data`'s contract.
        unsafe {
            let edge = self.navigator.init_first(ptr);
            self.remaining = ptr.length() - edge.length();
            edge_data(edge)
        }
    }

    /// Navigates to and returns the next data edge, or an empty slice at EOF.
    #[inline]
    pub(crate) fn next(&mut self) -> &'a [u8] {
        if self.remaining == 0 {
            return &[];
        }
        // SAFETY: per this reader's invariant (established at `init`), the
        // tree is live for `'a`, so `self.navigator.next()` and `edge_data`
        // on its non-null result are sound.
        unsafe {
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
    /// The reader must be non-empty (`is_some()`): on a default reader the
    /// navigator's path is dangling.
    #[inline]
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "API completeness for CordRepBtreeReader, exercised only by tests")
    )]
    pub(crate) unsafe fn skip(&mut self, skip: usize) -> &'a [u8] {
        // SAFETY: per this reader's invariant the tree is live for `'a`, so
        // `self.navigator.current()`/`.skip()` and `edge_data` on a
        // non-null result are sound.
        unsafe {
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
    /// The reader must be non-empty (`is_some()`): on a default reader the
    /// navigator's path is dangling.
    #[inline]
    pub(crate) unsafe fn seek(&mut self, offset: usize) -> &'a [u8] {
        // SAFETY: per this reader's invariant the tree is live for `'a`, so
        // `self.navigator.seek()`, `edge_data`, and `self.length()` are all
        // sound to call.
        unsafe {
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
    /// read ended in (empty if all data was read) and the tree (`None` if
    /// `n` exceeded the remaining data).
    ///
    /// Requires `chunk_size <= self.navigator.current().length()` (checked
    /// below).
    ///
    /// # Safety
    ///
    /// The reader must be non-empty (`is_some()`): on a default reader the
    /// navigator's path is dangling.
    pub(crate) unsafe fn read(&mut self, n: usize, chunk_size: usize) -> (&'a [u8], Option<OwnedRep>) {
        // SAFETY: per this reader's invariant the tree is live for `'a`, so
        // `self.navigator.current()`/`.next()`/`.read()` and `edge_data` on
        // their results are sound; `result.tree`, when `Some`, carries a
        // fresh reference (`CordRepBtreeNavigator::read`'s own contract),
        // adopted here into the returned `OwnedRep`.
        unsafe {
            debug_assert!(chunk_size <= self.navigator.current().length());

            // Start inside the last returned edge, or at the next edge.
            let mut edge = if chunk_size != 0 { self.navigator.current() } else { self.navigator.next() };
            let offset = if chunk_size != 0 { edge.length() - chunk_size } else { 0 };

            let result = self.navigator.read(offset, n);
            let tree = result.tree.map(|t| OwnedRep::from_raw(t.as_ptr()));

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
