//! Forward reader over the data edges of a btree, tracking remaining bytes.
//!
//! Port of abseil's `cord_rep_btree_reader.{h,cc}`.

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
    #[inline]
    pub(crate) unsafe fn node(&self) -> *mut CordRep {
        self.navigator.current()
    }

    /// Length of the referenced tree. Requires a non-empty reader.
    #[inline]
    pub(crate) unsafe fn length(&self) -> usize {
        debug_assert!(!self.btree().is_null());
        self.btree().length()
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
    #[inline]
    pub(crate) unsafe fn init<'a>(&mut self, tree: *mut CordRepBtree) -> &'a [u8] {
        debug_assert!(!tree.is_null());
        let edge = self.navigator.init_first(tree);
        self.remaining = tree.length() - edge.length();
        edge_data(edge)
    }

    /// Navigates to and returns the next data edge, or an empty slice at EOF.
    #[inline]
    pub(crate) unsafe fn next<'a>(&mut self) -> &'a [u8] {
        if self.remaining == 0 {
            return &[];
        }
        let edge = self.navigator.next();
        debug_assert!(!edge.is_null());
        self.remaining -= edge.length();
        edge_data(edge)
    }

    /// Skips `skip` bytes past the end of the current chunk and returns the
    /// data directly following them.
    #[inline]
    pub(crate) unsafe fn skip<'a>(&mut self, skip: usize) -> &'a [u8] {
        // We are positioned on the last consumed edge, so skip it too.
        let edge_length = self.navigator.current().length();
        let pos = self.navigator.skip(skip + edge_length);
        if pos.edge.is_null() {
            core::hint::cold_path();
            self.remaining = 0;
            return &[];
        }
        // All edges skipped before `pos.edge` (`skip - pos.offset` bytes) are
        // consumed, as is the current edge.
        self.remaining -= skip - pos.offset + pos.edge.length();
        &edge_data(pos.edge)[pos.offset..]
    }

    /// Navigates to the chunk containing `offset` and returns the data from
    /// `offset` to the end of that chunk, or empty if `offset >= length`.
    #[inline]
    pub(crate) unsafe fn seek<'a>(&mut self, offset: usize) -> &'a [u8] {
        let pos = self.navigator.seek(offset);
        if pos.edge.is_null() {
            core::hint::cold_path();
            self.remaining = 0;
            return &[];
        }
        let chunk = &edge_data(pos.edge)[pos.offset..];
        self.remaining = self.length() - offset - chunk.len();
        chunk
    }

    /// Reads `n` bytes into a new tree. If `chunk_size` is zero the read
    /// starts at the next data edge, else at the last `chunk_size` bytes of
    /// the last returned edge. Returns the remaining data of the edge the
    /// read ended in (empty if all data was read) and the tree (null if `n`
    /// exceeded the remaining data).
    pub(crate) unsafe fn read<'a>(&mut self, n: usize, chunk_size: usize) -> (&'a [u8], *mut CordRep) {
        debug_assert!(chunk_size <= self.navigator.current().length());

        // Start inside the last returned edge, or at the next edge.
        let mut edge = if chunk_size != 0 { self.navigator.current() } else { self.navigator.next() };
        let offset = if chunk_size != 0 { edge.length() - chunk_size } else { 0 };

        let result = self.navigator.read(offset, n);
        let tree = result.tree;

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
