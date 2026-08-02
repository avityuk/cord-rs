//! Memory usage estimation for cord trees. Port of abseil's
//! `cord_analysis.{h,cc}`.

use std::collections::HashSet;

use super::btree::{BtreePtr, as_btree};
use super::external::EXTERNAL_REP_SIZE;
use super::{BTREE, CordRep, CordRepSubstring, FLAT, RepPtr, SUBSTRING, flat, is_data_edge};

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
    counted: HashSet<*const CordRep>,
}

/// A rep reference carrying the cumulative inverse refcount weight ("fair
/// share" fraction) of the path from the root.
#[derive(Clone, Copy)]
struct RepRef {
    rep: *const CordRep,
    fraction: f64,
}

impl RepRef {
    /// # Safety
    ///
    /// `rep` must be non-null and point to a live [`CordRep`] (its refcount
    /// is read when `mode == Mode::FairShare`).
    #[inline]
    #[expect(clippy::cast_precision_loss, reason = "fair share accounting is an approximation by design")]
    unsafe fn new(mode: Mode, rep: *const CordRep, frac: f64) -> Self {
        unsafe {
            // SAFETY: `cast_mut` only changes pointer mutability, not what it
            // points to; the count is only ever read here, never mutated.
            let fraction = if mode == Mode::FairShare {
                let refcount = rep.cast_mut().refcount().get();
                if refcount == 1 { frac } else { frac / refcount as f64 }
            } else {
                1.0
            };
            Self { rep, fraction }
        }
    }

    /// # Safety
    ///
    /// `child` must be non-null and point to a live [`CordRep`] (same
    /// contract as [`new`](Self::new)).
    #[inline]
    unsafe fn child(self, mode: Mode, child: *const CordRep) -> Self {
        unsafe { Self::new(mode, child, self.fraction) }
    }
}

impl Analysis {
    #[expect(clippy::cast_precision_loss, reason = "fair share accounting is an approximation by design")]
    fn add(&mut self, size: usize, rep: RepRef) {
        match self.mode {
            Mode::Total => self.total += size as f64,
            Mode::FairShare => self.total += size as f64 * rep.fraction,
            Mode::TotalMorePrecise => {
                if self.counted.insert(rep.rep) {
                    self.total += size as f64;
                }
            }
        }
    }

    /// External reps are assumed heap allocated at their exact size.
    ///
    /// # Safety
    ///
    /// `rep.rep` must be non-null and point to a live data edge
    /// (`is_data_edge(rep.rep)`: a FLAT, EXTERNAL, or SUBSTRING of one).
    unsafe fn analyze_data_edge(&mut self, mut rep: RepRef) {
        unsafe {
            // SAFETY: `flat::allocated_size` requires a live flat node, which
            // the `tag >= FLAT` check below establishes.
            debug_assert!(is_data_edge(rep.rep));
            if (*rep.rep).tag == SUBSTRING {
                self.add(core::mem::size_of::<CordRepSubstring>(), rep);
                rep = rep.child(self.mode, (*rep.rep.cast::<CordRepSubstring>()).child);
            }
            let size = if (*rep.rep).tag >= FLAT {
                flat::allocated_size(rep.rep.cast_mut())
            } else {
                (*rep.rep).length + EXTERNAL_REP_SIZE
            };
            self.add(size, rep);
        }
    }

    /// # Safety
    ///
    /// `rep.rep` must be non-null and point to a live `CordRepBtree` (tag
    /// `BTREE`).
    unsafe fn analyze_btree(&mut self, rep: RepRef) {
        unsafe {
            // SAFETY: `tree.edges()` yields either live btree children of one
            // lesser height, or live data edges, by btree well-formedness —
            // satisfying `analyze_btree`'s and `analyze_data_edge`'s own
            // contracts for the recursive calls below.
            self.add(core::mem::size_of::<super::btree::CordRepBtree>(), rep);
            let tree = as_btree(rep.rep.cast_mut());
            if tree.height() > 0 {
                for edge in tree.edges() {
                    self.analyze_btree(rep.child(self.mode, edge));
                }
            } else {
                for edge in tree.edges() {
                    self.analyze_data_edge(rep.child(self.mode, edge));
                }
            }
        }
    }

    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "the total is a non-negative approximation of a byte count"
    )]
    /// # Safety
    ///
    /// `rep` must be non-null and point to a live [`CordRep`] tree (a data
    /// edge or a `CordRepBtree`).
    unsafe fn run(mode: Mode, rep: *const CordRep) -> usize {
        unsafe {
            let mut analysis = Analysis { mode, total: 0.0, counted: HashSet::new() };
            let repref = RepRef::new(mode, rep, 1.0);
            if is_data_edge(repref.rep) {
                analysis.analyze_data_edge(repref);
            } else {
                debug_assert_eq!((*repref.rep).tag, BTREE);
                analysis.analyze_btree(repref);
            }
            analysis.total as usize
        }
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
    unsafe { Analysis::run(Mode::Total, rep) }
}

/// Like [`estimated_memory_usage`] but counting each distinct node once.
///
/// # Safety
///
/// Same contract as [`estimated_memory_usage`].
pub(crate) unsafe fn more_precise_memory_usage(rep: *const CordRep) -> usize {
    unsafe { Analysis::run(Mode::TotalMorePrecise, rep) }
}

/// Approximate bytes held by `rep` weighted by the sharing ratio of each node.
///
/// # Safety
///
/// Same contract as [`estimated_memory_usage`].
pub(crate) unsafe fn estimated_fair_share_memory_usage(rep: *const CordRep) -> usize {
    unsafe { Analysis::run(Mode::FairShare, rep) }
}
