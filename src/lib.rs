//! A Rust port of abseil's [`absl::Cord`]: a rope-like byte sequence with
//! O(log n) append, prepend and slicing, O(1) cloning, and a 16 byte
//! footprint with 15 bytes of inline storage.
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
//! let mut buffer = CordBuffer::with_default_limit(1024);
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
//! from 32 bytes to 256 KiB), *external* buffers owned by user values
//! (`Vec<u8>`, `Arc<[u8]>`, `&'static [u8]`, ...), or *substrings* of those.
//! Buffers referenced by a single cord are grown in place; buffers shared
//! between cords are never modified.
//!
//! The implementation mirrors abseil's `absl::Cord` including its
//! heuristics (copy-vs-share thresholds, amortized growth, buffer size
//! classes), so performance characteristics carry over. The Cordz sampling
//! layer and the CRC checksum node were not ported.
//!
//! # Features
//!
//! * `bytes` — `bytes::Buf` for [`Cord`] and [`Cursor`], `bytes::BufMut`
//!   for [`CordWriter`], and zero-copy conversions with `bytes::Bytes`.
//! * `serde` — `Serialize` / `Deserialize` for [`Cord`] as a byte sequence.
//!
//! [`absl::Cord`]: https://github.com/abseil/abseil-cpp/blob/master/absl/strings/cord.h
#![cfg_attr(docsrs, feature(doc_cfg))]
#![warn(missing_docs, rust_2018_idioms, unreachable_pub)]
#![allow(clippy::needless_lifetimes)]

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
pub use iter::{Bytes, Chunks, Cursor};
pub use source::{CordLike, CordSource};

#[cfg(feature = "bytes")]
#[cfg_attr(docsrs, doc(cfg(feature = "bytes")))]
pub use bytes_impl::CordWriter;

/// Internal inspection hooks for tests and benchmarks. Not part of the
/// public API; may change without notice.
#[doc(hidden)]
pub mod internal {
    use crate::Cord;
    use crate::rep::btree::{BtreePtr, CordRepBtree, as_btree};
    use crate::rep::{self, RepPtr};

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

    /// Whether `cord` holds a btree.
    #[must_use]
    pub fn is_btree(cord: &Cord) -> bool {
        // SAFETY: the tree is live.
        cord.tree().is_some_and(|t| unsafe { t.is_btree() })
    }

    /// Whether `cord` holds a single flat node.
    #[must_use]
    pub fn is_flat(cord: &Cord) -> bool {
        // SAFETY: the tree is live.
        cord.tree().is_some_and(|t| unsafe { t.is_flat() })
    }

    /// Whether `cord` holds a single external node.
    #[must_use]
    pub fn is_external(cord: &Cord) -> bool {
        // SAFETY: the tree is live.
        cord.tree().is_some_and(|t| unsafe { t.is_external() })
    }

    /// Whether `cord` holds a single substring node.
    #[must_use]
    pub fn is_substring(cord: &Cord) -> bool {
        // SAFETY: the tree is live.
        cord.tree().is_some_and(|t| unsafe { t.is_substring() })
    }

    /// Height of the btree, if any.
    #[must_use]
    pub fn btree_height(cord: &Cord) -> Option<usize> {
        // SAFETY: the tree is live.
        cord.tree().and_then(|t| unsafe { t.is_btree().then(|| as_btree(t).height()) })
    }

    /// Reference count of the root node (0 if inline).
    #[must_use]
    pub fn root_refcount(cord: &Cord) -> usize {
        // SAFETY: the tree is live.
        cord.tree().map_or(0, |t| unsafe { t.refcount().get() })
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
                let _ = CordRepBtree::dump(tree, "", include_contents, &mut out);
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
        unsafe { Cord::from_rep(rep::external::CordRepExternal::create(data.to_vec())) }
    }

    /// Creates a cord holding a substring node over the flat or external node
    /// of `src` (mirrors abseil's `CordTestPeer::MakeSubstring`). Requires
    /// `src` to hold a single flat or external node and `0 < len < src.len()`.
    #[must_use]
    pub fn make_substring(src: &Cord, offset: usize, len: usize) -> Cord {
        let tree = src.tree().expect("make_substring: src must not be inline");
        // SAFETY: `create` adopts the added reference; preconditions checked
        // by `create`.
        unsafe {
            assert!(
                tree.is_flat() || tree.is_external(),
                "make_substring: src must be a flat or external node"
            );
            let rep = rep::CordRepSubstring::create(rep::ref_rep(tree), offset, len);
            Cord::from_rep(rep.cast())
        }
    }

    /// The allocated size of the flat node held by `cord`, if it holds one.
    #[must_use]
    pub fn flat_allocated_size(cord: &Cord) -> Option<usize> {
        // SAFETY: the tree is live.
        cord.tree().and_then(|t| unsafe { t.is_flat().then(|| rep::flat::allocated_size(t)) })
    }
}
