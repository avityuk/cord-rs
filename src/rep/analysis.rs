//! Memory usage estimation for cord trees.

use alloc::collections::BTreeSet;

use super::{CordRep, CordRepSubstring, RepRef, RepView};

/// Accounting mode, see [`crate::MemoryAccounting`].
#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    FairShare,
    Total,
    TotalMorePrecise,
}

/// Computes the estimated memory used by the tree `rep` (excluding the
/// `Cord` itself) for the given accounting mode.
struct Analysis {
    mode: Mode,
    total: f64,
    // Keyed by pointer address rather than `*const CordRep` itself: this is
    // a cold memory-accounting path, so a `BTreeSet<usize>` (no `HashSet` is
    // available under `alloc`) is a fine trade for the simplicity of an
    // address key over a raw-pointer `Ord` impl.
    counted: BTreeSet<usize>,
}

/// A rep handle carrying the cumulative inverse refcount weight ("fair
/// share" fraction) of the path from the root.
#[derive(Clone, Copy)]
struct Node<'a> {
    rep: RepRef<'a>,
    fraction: f64,
}

impl<'a> Node<'a> {
    #[inline]
    #[expect(clippy::cast_precision_loss, reason = "fair share accounting is an approximation by design")]
    fn new(mode: Mode, rep: RepRef<'a>, frac: f64) -> Self {
        let fraction = if mode == Mode::FairShare {
            let refcount = rep.ref_get();
            if refcount == 1 { frac } else { frac / refcount as f64 }
        } else {
            1.0
        };
        Self { rep, fraction }
    }

    #[inline]
    fn child(self, mode: Mode, child: RepRef<'a>) -> Self {
        Self::new(mode, child, self.fraction)
    }
}

impl Analysis {
    #[expect(clippy::cast_precision_loss, reason = "fair share accounting is an approximation by design")]
    fn add(&mut self, size: usize, node: Node<'_>) {
        match self.mode {
            Mode::Total => self.total += size as f64,
            Mode::FairShare => self.total += size as f64 * node.fraction,
            Mode::TotalMorePrecise => {
                if self.counted.insert(node.rep.as_ptr().addr()) {
                    self.total += size as f64;
                }
            }
        }
    }

    /// External reps are assumed heap allocated at their exact size.
    ///
    /// Requires `node.rep.is_data_edge()` (a FLAT, EXTERNAL, or SUBSTRING of
    /// one).
    fn analyze_data_edge(&mut self, mut node: Node<'_>) {
        debug_assert!(node.rep.is_data_edge());
        if let RepView::Substring { child, .. } = node.rep.view() {
            self.add(core::mem::size_of::<CordRepSubstring>(), node);
            node = node.child(self.mode, child);
        }
        let size = match node.rep.view() {
            RepView::Flat(flat) => flat.allocated_size(),
            RepView::External(ext) => ext.allocated_size(),
            _ => unreachable!("cord-rs: data edge must resolve to a flat or external node"),
        };
        self.add(size, node);
    }

    /// Requires `node.rep.is_btree()`.
    fn analyze_btree(&mut self, node: Node<'_>) {
        self.add(core::mem::size_of::<super::btree::CordRepBtree>(), node);
        let RepView::Btree(tree) = node.rep.view() else {
            unreachable!("cord-rs: analyze_btree requires a btree node")
        };
        // The recursive calls' contracts (`analyze_btree`'s and
        // `analyze_data_edge`'s own) are satisfied by btree well-formedness:
        // `tree.edges()` yields either live btree children of one lesser
        // height, or live data edges.
        if tree.height() > 0 {
            for edge in tree.edges() {
                self.analyze_btree(node.child(self.mode, edge));
            }
        } else {
            for edge in tree.edges() {
                self.analyze_data_edge(node.child(self.mode, edge));
            }
        }
    }

    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "the total is a non-negative approximation of a byte count"
    )]
    fn run(mode: Mode, rep: RepRef<'_>) -> usize {
        let mut analysis = Analysis { mode, total: 0.0, counted: BTreeSet::new() };
        let node = Node::new(mode, rep, 1.0);
        if rep.is_data_edge() {
            analysis.analyze_data_edge(node);
        } else {
            debug_assert!(rep.is_btree());
            analysis.analyze_btree(node);
        }
        analysis.total as usize
    }
}

/// Approximate bytes held by `rep`, counting shared memory fully for each
/// reference.
///
/// # Safety
///
/// `rep` must be non-null and point to a live [`CordRep`] tree (a data edge
/// or a `CordRepBtree`).
pub(crate) unsafe fn estimated_memory_usage(rep: *const CordRep) -> usize {
    // SAFETY: `rep` is a live rep tree per this fn's contract, which is
    // exactly `RepRef::from_raw`'s precondition; `Analysis::run` neither
    // adopts nor transfers a reference, so `rep`'s cast to `*mut` here is
    // just a pointer reinterpretation (the resulting `RepRef` is read-only).
    let rep = unsafe { RepRef::from_raw(rep.cast_mut()) };
    Analysis::run(Mode::Total, rep)
}

/// Like [`estimated_memory_usage`] but counting each distinct node once.
///
/// # Safety
///
/// Same contract as [`estimated_memory_usage`].
pub(crate) unsafe fn more_precise_memory_usage(rep: *const CordRep) -> usize {
    // SAFETY: see `estimated_memory_usage`.
    let rep = unsafe { RepRef::from_raw(rep.cast_mut()) };
    Analysis::run(Mode::TotalMorePrecise, rep)
}

/// Approximate bytes held by `rep` weighted by the sharing ratio of each node.
///
/// # Safety
///
/// Same contract as [`estimated_memory_usage`].
pub(crate) unsafe fn estimated_fair_share_memory_usage(rep: *const CordRep) -> usize {
    // SAFETY: see `estimated_memory_usage`.
    let rep = unsafe { RepRef::from_raw(rep.cast_mut()) };
    Analysis::run(Mode::FairShare, rep)
}
