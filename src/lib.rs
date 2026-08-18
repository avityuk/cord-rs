//! A rope-like byte sequence with O(log n) append, prepend and slicing, O(1)
//! cloning, and a 16 byte footprint with 15 bytes of inline storage — a port
//! of abseil's [`absl::Cord`], with some changes along the way.
//!
//! # When to use a `Cord`
//!
//! A [`Cord`] is designed for large byte sequences that change over their
//! lifetime or are shared across API boundaries: wire-format messages that
//! get headers prepended or payloads appended, buffers assembled from many
//! pieces, or data that is sliced and passed around without copying. For
//! small, contiguous, rarely modified data prefer `Vec<u8>` or
//! `bytes::Bytes`.
//!
//! # Example
//!
//! ```
//! use cord_rs::{Cord, CordBuffer};
//!
//! // Small values live inline; no allocation.
//! let mut cord = Cord::from("hello");
//! assert_eq!(cord.len(), 5);
//!
//! // Appending, prepending and slicing share memory and stay O(log n).
//! cord.append(" world");
//! cord.prepend(">> ");
//! let world = cord.slice(9..);
//! assert_eq!(world, "world");
//!
//! // Large owned buffers are adopted rather than copied.
//! cord.append(vec![b'!'; 4096]);
//! assert_eq!(cord.len(), 3 + 11 + 4096);
//!
//! // Zero-copy building through `CordBuffer`.
//! let mut buffer = CordBuffer::with_capacity(1024);
//! buffer.put_slice(b" tail");
//! cord.append(buffer);
//!
//! // Iterate chunks for efficient processing, or bytes for convenience.
//! let total: usize = cord.chunks().map(<[u8]>::len).sum();
//! assert_eq!(total, cord.len());
//! assert!(cord.ends_with(" tail"));
//! ```
//!
//! # Representation
//!
//! A `Cord` is 16 bytes: either up to 15 bytes of inline data, or a pointer
//! to a reference counted tree. Trees are B-trees of up to 6 edges per node
//! whose leaves reference immutable *flat* buffers (allocated in size classes
//! from 32 bytes to 4 KiB by default, or up to 64 KiB when built through a
//! [`CordBuffer`] with a custom limit — the one-byte tag that encodes a
//! flat's allocation size can address up to 256 KiB, but nothing in the
//! crate allocates that large today), *external* buffers owned by user
//! values (`Vec<u8>`, `Arc<[u8]>`, `&'static [u8]`, ...), or *substrings* of
//! those. Buffers referenced by a single cord are grown in place; buffers
//! shared between cords are never modified.
//!
//! The implementation started from abseil's design — the same heuristics
//! (copy-vs-share thresholds, amortized growth, buffer size classes) — and
//! evolves independently where that keeps the crate simpler or the trees
//! smaller. The Cordz sampling layer and the CRC checksum node were not
//! ported.
//!
//! # Features
//!
//! * `bytes` — `bytes::Buf` for [`Cord`] and [`Cursor`](crate::iter::Cursor),
//!   `bytes::BufMut` for [`CordWriter`], and zero-copy conversions with
//!   `bytes::Bytes`.
//! * `serde` — `Serialize` / `Deserialize` for [`Cord`] as a byte sequence.
//!
//! [`absl::Cord`]: https://github.com/abseil/abseil-cpp/blob/master/absl/strings/cord.h
#![cfg_attr(docsrs, feature(doc_cfg))]
#![warn(missing_docs, rust_2018_idioms, unreachable_pub)]

mod buffer;
mod cord;
mod inline_data;
mod io;
pub mod iter;
mod rep;
mod source;

#[cfg(feature = "bytes")]
#[cfg_attr(docsrs, doc(cfg(feature = "bytes")))]
mod bytes_impl;
#[cfg(feature = "serde")]
#[cfg_attr(docsrs, doc(cfg(feature = "serde")))]
mod serde_impl;

pub use buffer::CordBuffer;
pub use cord::{Cord, MemoryAccounting};
pub use source::{CordLike, CordSource};

#[cfg(feature = "bytes")]
#[cfg_attr(docsrs, doc(cfg(feature = "bytes")))]
pub use bytes_impl::CordWriter;

/// Compile-time guard against silently losing `Send`/`Sync` on the crate's
/// public types. `Cord` and `CordBuffer` assert both explicitly (see their
/// `unsafe impl` blocks); `Chunks`/`Cursor`/`Bytes` (and, with the `bytes`
/// feature, `CordWriter`) get them for free by auto-derivation from their
/// fields — in particular `rep::RepRef`'s own `unsafe impl` — so a future
/// field change that broke that derivation would otherwise compile clean
/// with no other signal anywhere. Calling `assert_send_sync::<T>()` from a
/// `const` item forces the compiler to resolve `T: Send + Sync` for that
/// exact `T` right here, at definition time — no value of the type, or any
/// other call site, is needed. The same silent-loss argument applies to the
/// unwind-safety auto traits: `Vec<u8>` and `bytes::Bytes` are both
/// `UnwindSafe + RefUnwindSafe`, so a raw-pointer field change here could
/// just as quietly drop `Cord` out of them with no other signal.
const fn assert_send_sync<T: ?Sized + Send + Sync>() {}

const _: () = {
    assert_send_sync::<Cord>();
    assert_send_sync::<CordBuffer>();
    assert_send_sync::<crate::iter::Chunks<'_>>();
    assert_send_sync::<crate::iter::Cursor<'_>>();
    assert_send_sync::<crate::iter::Bytes<'_>>();
};

#[cfg(feature = "bytes")]
const _: () = assert_send_sync::<CordWriter<'_>>();

/// Compile-time guard against silently losing unwind-safety on the crate's
/// public types (see [`assert_send_sync`] for why this needs a `const` call
/// rather than relying on the auto trait alone). `CordWriter` is deliberately
/// excluded: it holds a `&mut Cord`, so it is correctly `!UnwindSafe`, the
/// same way `std::io::BufWriter<&mut W>` is.
const fn assert_unwind_safe<T: ?Sized + std::panic::UnwindSafe + std::panic::RefUnwindSafe>() {}

const _: () = {
    assert_unwind_safe::<Cord>();
    assert_unwind_safe::<CordBuffer>();
    assert_unwind_safe::<crate::iter::Chunks<'_>>();
    assert_unwind_safe::<crate::iter::Cursor<'_>>();
    assert_unwind_safe::<crate::iter::Bytes<'_>>();
};

/// Internal inspection hooks for tests and benchmarks. Exempt from semver
/// guarantees: this module exists solely for the crate's own tests and
/// benchmarks and may change or disappear without notice, regardless of
/// crate version.
#[doc(hidden)]
pub mod __internal {
    use core::ptr::NonNull;

    use crate::Cord;
    use crate::rep::btree::{CordRepBtree, as_btree};
    use crate::rep::{self, OwnedRep, RepPtr, RepRef, RepView};

    /// Maximum inline size of a `Cord`.
    pub const MAX_INLINE: usize = rep::MAX_INLINE;
    /// Maximum payload of a default flat.
    pub const MAX_FLAT_LENGTH: usize = rep::flat::MAX_FLAT_LENGTH;
    /// Minimum payload of a flat.
    pub const MIN_FLAT_LENGTH: usize = rep::flat::MIN_FLAT_LENGTH;
    /// Overhead of a flat allocation.
    pub const FLAT_OVERHEAD: usize = rep::flat::FLAT_OVERHEAD;
    /// Cords at most this size are copied rather than shared.
    pub const MAX_BYTES_TO_COPY: usize = rep::MAX_BYTES_TO_COPY;
    /// Maximum btree node fan-out.
    pub const BTREE_MAX_CAPACITY: usize = rep::btree::MAX_CAPACITY;

    /// Whether `cord` holds a tree (as opposed to inline data).
    #[must_use]
    pub fn is_tree(cord: &Cord) -> bool {
        cord.is_tree()
    }

    /// Wraps `cord`'s root tree pointer, if any, as a borrowed [`RepRef`].
    fn root(cord: &Cord) -> Option<RepRef<'_>> {
        cord.tree_ref()
    }

    /// Whether `cord` holds a btree.
    #[must_use]
    pub fn is_btree(cord: &Cord) -> bool {
        root(cord).is_some_and(RepRef::is_btree)
    }

    /// Whether `cord` holds a single flat node.
    #[must_use]
    pub fn is_flat(cord: &Cord) -> bool {
        root(cord).is_some_and(RepRef::is_flat)
    }

    /// Whether `cord` holds a single external node.
    #[must_use]
    pub fn is_external(cord: &Cord) -> bool {
        root(cord).is_some_and(RepRef::is_external)
    }

    /// Whether `cord` holds a single substring node.
    #[must_use]
    pub fn is_substring(cord: &Cord) -> bool {
        root(cord).is_some_and(RepRef::is_substring)
    }

    /// Height of the btree, if any.
    #[must_use]
    pub fn btree_height(cord: &Cord) -> Option<usize> {
        root(cord).and_then(|rep| match rep.view() {
            RepView::Btree(tree) => Some(tree.height()),
            _ => None,
        })
    }

    /// Reference count of the root node (0 if inline).
    #[must_use]
    pub fn root_refcount(cord: &Cord) -> usize {
        root(cord).map_or(0, RepRef::ref_get)
    }

    /// Validates the tree structure, returning an error message on failure.
    pub fn validate(cord: &Cord) -> Result<(), String> {
        let Some(tree) = cord.tree() else { return Ok(()) };
        // SAFETY: the tree is live.
        unsafe {
            if tree.is_btree() {
                CordRepBtree::check_valid(as_btree(tree), false)
            } else if rep::is_data_edge(tree) {
                Ok(())
            } else {
                Err(format!("unexpected root node type {}", tree.tag()))
            }
        }
    }

    /// Dumps the tree structure.
    #[must_use]
    pub fn dump(cord: &Cord, include_contents: bool) -> String {
        let mut out = String::new();
        match cord.tree() {
            None => out.push_str("(inline)\n"),
            // SAFETY: the tree is live.
            Some(tree) => unsafe {
                let _ = CordRepBtree::dump(NonNull::new(tree), "", include_contents, &mut out);
            },
        }
        out
    }

    /// Forces exhaustive validation in debug assertions.
    pub fn set_exhaustive_validation(enabled: bool) {
        rep::btree::set_exhaustive_validation(enabled);
    }

    /// Size of a btree node.
    pub const BTREE_NODE_SIZE: usize = core::mem::size_of::<CordRepBtree>();
    /// Size of a substring node.
    pub const SUBSTRING_NODE_SIZE: usize = core::mem::size_of::<rep::CordRepSubstring>();
    /// Size accounted for an external node (excluding the referenced data).
    pub const EXTERNAL_NODE_SIZE: usize = rep::external::EXTERNAL_REP_SIZE;

    /// Creates a cord holding `data` in a single external node regardless of
    /// its size (mirrors abseil's `MakeCordFromExternal` in tests). An empty
    /// `data` yields an empty cord.
    #[must_use]
    pub fn make_external(data: &[u8]) -> Cord {
        if data.is_empty() {
            return Cord::new();
        }
        // SAFETY: a fresh, non-empty external rep.
        let owned =
            unsafe { OwnedRep::from_raw(rep::external::CordRepExternal::create_global(data.to_vec())) };
        Cord::from_owned_rep(owned)
    }

    /// Creates a cord holding a substring node over the flat or external node
    /// of `src` (mirrors abseil's `CordTestPeer::MakeSubstring`). Requires
    /// `src` to hold a single flat or external node and `0 < len < src.len()`.
    #[must_use]
    pub fn make_substring(src: &Cord, offset: usize, len: usize) -> Cord {
        let tree = src.tree().expect("make_substring: src must not be inline");
        // SAFETY: `create` adopts the added reference; preconditions checked
        // by `create`.
        let owned = unsafe {
            assert!(
                tree.is_flat() || tree.is_external(),
                "make_substring: src must be a flat or external node"
            );
            let rep = rep::CordRepSubstring::create(rep::ref_rep(tree), offset, len);
            OwnedRep::from_raw(rep.cast())
        };
        Cord::from_owned_rep(owned)
    }

    /// The allocated size of the flat node held by `cord`, if it holds one.
    #[must_use]
    pub fn flat_allocated_size(cord: &Cord) -> Option<usize> {
        root(cord).and_then(|rep| match rep.view() {
            RepView::Flat(flat) => Some(flat.allocated_size()),
            _ => None,
        })
    }
}
