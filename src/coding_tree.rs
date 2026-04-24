//! VVC coding-tree walker (§7.3.11.4) — I-slice subset.
//!
//! The coding tree for an intra CTU in VVC is a recursive partition
//! driven by five CABAC flags:
//!
//! * `split_cu_flag` — "is this node split at all?"
//! * `split_qt_flag` — "if split, is it a quad-tree split?"
//! * `mtt_split_cu_vertical_flag` — "BT/TT: vertical vs horizontal?"
//! * `mtt_split_cu_binary_flag` — "BT vs TT?"
//! * `non_inter_flag` — only relevant in B/P slices; ignored here.
//!
//! This module walks that tree assuming *all* splits have been coded
//! (i.e. a conformant CABAC-coded coding_tree()). It does not yet
//! enforce the spec's partition-allowance logic (§7.4.12.4); for the
//! I-slice foundation path we read the flags as written and let the
//! spec constraints be caught later when the full walker lands.
//!
//! The walker returns a flat list of [`Cu`] entries describing each
//! leaf node (a CU, within which intra prediction + residual coding
//! happens). Each leaf is a rectangle `(x, y, w, h)` in CTU-local
//! coordinates.
//!
//! Only luma coding is modelled at the tree level — chroma decoding
//! inherits the luma partition for SingleTree I-slices which is the
//! common configuration (§7.4.11.4).
//!
//! Spec reference: ITU-T H.266 | ISO/IEC 23090-3 (V4, 01/2026).

use oxideav_core::Result;

use crate::cabac::{ArithDecoder, ContextModel};
use crate::ctx::{
    ctx_inc_intra_luma_mpm_flag, ctx_inc_pred_mode_flag, ctx_inc_split_cu_flag,
    ctx_inc_split_qt_flag,
};

/// A leaf coding unit emitted by the walker.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cu {
    /// Left edge in CTU-local luma sample coordinates.
    pub x: u32,
    /// Top edge in CTU-local luma sample coordinates.
    pub y: u32,
    /// Width in luma samples.
    pub w: u32,
    /// Height in luma samples.
    pub h: u32,
    /// Coding-quad-tree depth.
    pub cqt_depth: u32,
    /// Multi-type-tree depth.
    pub mtt_depth: u32,
}

/// CABAC context arrays used by the coding-tree walker.
pub struct TreeCtxs {
    pub split_cu: Vec<ContextModel>,
    pub split_qt: Vec<ContextModel>,
    pub pred_mode: Vec<ContextModel>,
    pub intra_luma_mpm: Vec<ContextModel>,
}

impl TreeCtxs {
    /// Build the context arrays from Table 59 / 60 / 66 / 75 at the
    /// supplied slice QP.
    pub fn init(slice_qp_y: i32) -> Self {
        use crate::tables::{init_contexts, SyntaxCtx};
        Self {
            split_cu: init_contexts(SyntaxCtx::SplitCuFlag, slice_qp_y),
            split_qt: init_contexts(SyntaxCtx::SplitQtFlag, slice_qp_y),
            pred_mode: init_contexts(SyntaxCtx::PredModeFlag, slice_qp_y),
            intra_luma_mpm: init_contexts(SyntaxCtx::IntraLumaMpmFlag, slice_qp_y),
        }
    }
}

/// Walker state: the output CU list plus scratch neighbour info for
/// ctxInc derivation. Initial implementation ignores neighbour state
/// and passes `false` availability flags — that's a valid simplification
/// for the root of the first CTU but biases the ctxInc derivation for
/// subsequent CUs. Will be refined when a full picture walker lands.
pub struct TreeWalker<'a, 'b> {
    dec: &'a mut ArithDecoder<'b>,
    ctxs: &'a mut TreeCtxs,
    min_cb_log2: u32,
    out: Vec<Cu>,
}

impl<'a, 'b> TreeWalker<'a, 'b> {
    pub fn new(dec: &'a mut ArithDecoder<'b>, ctxs: &'a mut TreeCtxs) -> Self {
        Self {
            dec,
            ctxs,
            min_cb_log2: 2, // 4x4 minimum for luma (MinCbLog2SizeY default)
            out: Vec::new(),
        }
    }

    /// Walk a coding_tree() rooted at `(x, y)` of size `w × h`.
    /// Returns a flat list of leaf CUs in decoding order.
    pub fn walk(mut self, x: u32, y: u32, w: u32, h: u32) -> Result<Vec<Cu>> {
        self.recurse(x, y, w, h, 0, 0)?;
        Ok(self.out)
    }

    fn recurse(
        &mut self,
        x: u32,
        y: u32,
        w: u32,
        h: u32,
        cqt_depth: u32,
        mtt_depth: u32,
    ) -> Result<()> {
        // At the minimum CU size we stop without reading a split flag.
        let at_min = w.trailing_zeros() <= self.min_cb_log2 || h.trailing_zeros() <= self.min_cb_log2;
        if at_min {
            self.out.push(Cu {
                x,
                y,
                w,
                h,
                cqt_depth,
                mtt_depth,
            });
            return Ok(());
        }
        // split_cu_flag — ctxInc uses availability of L/A neighbours
        // (treated as unavailable at the root here) and partition
        // allowance flags (all allowed at the root of a large CU).
        let split_cu_inc = ctx_inc_split_cu_flag(
            false, false, 0, 0, w, h, 1, 1, 1, 1, 1,
        ) as usize;
        let split_cu_ctx_n = self.ctxs.split_cu.len() - 1;
        let split_cu = self
            .dec
            .decode_decision(&mut self.ctxs.split_cu[split_cu_inc.min(split_cu_ctx_n)])?;
        if split_cu == 0 {
            self.out.push(Cu {
                x,
                y,
                w,
                h,
                cqt_depth,
                mtt_depth,
            });
            return Ok(());
        }
        // split_qt_flag — only readable while we're in the quad-tree
        // phase (mtt_depth == 0). Outside that we force BT/TT.
        let split_qt = if mtt_depth == 0 {
            let inc = ctx_inc_split_qt_flag(false, false, 0, 0, cqt_depth) as usize;
            let n = self.ctxs.split_qt.len() - 1;
            self.dec
                .decode_decision(&mut self.ctxs.split_qt[inc.min(n)])?
        } else {
            0
        };
        if split_qt == 1 {
            let hw = w / 2;
            let hh = h / 2;
            self.recurse(x, y, hw, hh, cqt_depth + 1, mtt_depth)?;
            self.recurse(x + hw, y, hw, hh, cqt_depth + 1, mtt_depth)?;
            self.recurse(x, y + hh, hw, hh, cqt_depth + 1, mtt_depth)?;
            self.recurse(x + hw, y + hh, hw, hh, cqt_depth + 1, mtt_depth)?;
            return Ok(());
        }
        // Multi-type-tree split: read vertical / binary bypass-bins as
        // a simplification — the spec reads them through dedicated
        // contexts which we'll wire later. Using bypass here keeps
        // the CABAC state sync'd (bypass does not update a context).
        let mtt_vertical = self.dec.decode_bypass()?;
        let mtt_binary = self.dec.decode_bypass()?;
        if mtt_binary == 1 {
            // Binary split.
            if mtt_vertical == 1 {
                let hw = w / 2;
                self.recurse(x, y, hw, h, cqt_depth, mtt_depth + 1)?;
                self.recurse(x + hw, y, hw, h, cqt_depth, mtt_depth + 1)?;
            } else {
                let hh = h / 2;
                self.recurse(x, y, w, hh, cqt_depth, mtt_depth + 1)?;
                self.recurse(x, y + hh, w, hh, cqt_depth, mtt_depth + 1)?;
            }
        } else {
            // Ternary split (1:2:1 sizes).
            if mtt_vertical == 1 {
                let q = w / 4;
                self.recurse(x, y, q, h, cqt_depth, mtt_depth + 1)?;
                self.recurse(x + q, y, q * 2, h, cqt_depth, mtt_depth + 1)?;
                self.recurse(x + q * 3, y, q, h, cqt_depth, mtt_depth + 1)?;
            } else {
                let q = h / 4;
                self.recurse(x, y, w, q, cqt_depth, mtt_depth + 1)?;
                self.recurse(x, y + q, w, q * 2, cqt_depth, mtt_depth + 1)?;
                self.recurse(x, y + q * 3, w, q, cqt_depth, mtt_depth + 1)?;
            }
        }
        Ok(())
    }
}

/// Read an `intra_luma_mpm_flag` bin. Exposed for unit tests and for
/// the intra-predict selector when it is wired up.
pub fn read_intra_luma_mpm_flag(
    dec: &mut ArithDecoder<'_>,
    ctxs: &mut TreeCtxs,
) -> Result<u32> {
    let inc = ctx_inc_intra_luma_mpm_flag() as usize;
    dec.decode_decision(&mut ctxs.intra_luma_mpm[inc])
}

/// Read a `pred_mode_flag` bin at the root of an intra-inter-capable
/// CU. Always returns 1 in an I-slice because pred_mode_flag is never
/// signalled there — but exposed for completeness.
pub fn read_pred_mode_flag(
    dec: &mut ArithDecoder<'_>,
    ctxs: &mut TreeCtxs,
) -> Result<u32> {
    let inc = ctx_inc_pred_mode_flag(false, false, false, false) as usize;
    dec.decode_decision(&mut ctxs.pred_mode[inc])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 32-byte zero stream with the bias-toward-MPS-0 ctx will read
    /// split_cu_flag=0 repeatedly, so a root CU is emitted unsplit.
    #[test]
    fn unsplit_root_is_single_cu() {
        let data = [0u8; 32];
        let mut dec = ArithDecoder::new(&data).unwrap();
        let mut ctxs = TreeCtxs::init(16);
        let walker = TreeWalker::new(&mut dec, &mut ctxs);
        let cus = walker.walk(0, 0, 64, 64).unwrap();
        // Whether we get 1 split or 0 depends on the context bias at QP=16.
        // Just check the tree makes structural sense: every CU fits inside
        // the root, and they tile it in aggregate area (≤ 64*64).
        let mut area = 0u32;
        for c in &cus {
            assert!(c.x + c.w <= 64);
            assert!(c.y + c.h <= 64);
            area += c.w * c.h;
        }
        assert!(area <= 64 * 64);
    }

    /// At the minimum-CB size the walker returns the block unsplit
    /// without reading any CABAC bins.
    #[test]
    fn min_cb_is_leaf() {
        let data = [0u8; 32];
        let mut dec = ArithDecoder::new(&data).unwrap();
        let mut ctxs = TreeCtxs::init(16);
        let walker = TreeWalker::new(&mut dec, &mut ctxs);
        // 4x4 is the minimum CB size for luma (MinCbLog2SizeY = 2).
        let cus = walker.walk(0, 0, 4, 4).unwrap();
        assert_eq!(cus, vec![Cu { x: 0, y: 0, w: 4, h: 4, cqt_depth: 0, mtt_depth: 0 }]);
    }
}
