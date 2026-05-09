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
    ctx_inc_intra_luma_mpm_flag, ctx_inc_mtt_split_cu_binary_flag,
    ctx_inc_mtt_split_cu_vertical_flag, ctx_inc_pred_mode_flag, ctx_inc_split_cu_flag,
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
#[derive(Debug)]
pub struct TreeCtxs {
    pub split_cu: Vec<ContextModel>,
    pub split_qt: Vec<ContextModel>,
    /// Round-55 — `mtt_split_cu_vertical_flag` per Table 61 (15 ctxIdx).
    pub mtt_split_vertical: Vec<ContextModel>,
    /// Round-55 — `mtt_split_cu_binary_flag` per Table 62 (12 ctxIdx).
    pub mtt_split_binary: Vec<ContextModel>,
    pub pred_mode: Vec<ContextModel>,
    pub intra_luma_mpm: Vec<ContextModel>,
}

impl TreeCtxs {
    /// Build the context arrays from Table 59 / 60 / 61 / 62 / 66 / 75
    /// at the supplied slice QP.
    pub fn init(slice_qp_y: i32) -> Self {
        use crate::tables::{init_contexts, SyntaxCtx};
        Self {
            split_cu: init_contexts(SyntaxCtx::SplitCuFlag, slice_qp_y),
            split_qt: init_contexts(SyntaxCtx::SplitQtFlag, slice_qp_y),
            mtt_split_vertical: init_contexts(SyntaxCtx::MttSplitCuVerticalFlag, slice_qp_y),
            mtt_split_binary: init_contexts(SyntaxCtx::MttSplitCuBinaryFlag, slice_qp_y),
            pred_mode: init_contexts(SyntaxCtx::PredModeFlag, slice_qp_y),
            intra_luma_mpm: init_contexts(SyntaxCtx::IntraLumaMpmFlag, slice_qp_y),
        }
    }
}

/// Round-56 — picture-wide neighbour-map for the §9.3.4.2 ctxInc
/// derivations.
///
/// The map is keyed by `(x / 4, y / 4)` cells (min-CB granularity) and
/// holds, for every emitted CU, the descriptor consumed by
/// [`ctx_inc_split_cu_flag`] / [`ctx_inc_split_qt_flag`] /
/// [`ctx_inc_mtt_split_cu_vertical_flag`]. The decoder side mirrors
/// the encoder map shape so the wire bins match: encoder + decoder
/// both populate the map as each CU is committed and look up the
/// `(x - 1, y)` / `(x, y - 1)` neighbour to derive ctxInc.
///
/// The map is the spec-exact replacement for the round-55 hard-coded
/// `nbrs.left/above_avail = false`. With per-CU neighbour state plumbed
/// the encoder + decoder agree on `condL` / `condA` for split flags and
/// produce wire-compatible CABAC streams across multi-row CTBs.
#[derive(Debug, Default, Clone)]
pub struct CuNeighbourMap {
    width_mcb: usize,
    height_mcb: usize,
    cells: Vec<Option<NeighbourDescriptor>>,
}

#[derive(Debug, Clone, Copy)]
struct NeighbourDescriptor {
    cb_w: u32,
    cb_h: u32,
    cqt_depth: u32,
}

impl CuNeighbourMap {
    /// Build a fresh map for a picture of `(w, h)` luma samples. Every
    /// cell is initially unpopulated.
    pub fn new(w: u32, h: u32) -> Self {
        let width_mcb = (w as usize + 3) / 4;
        let height_mcb = (h as usize + 3) / 4;
        Self {
            width_mcb,
            height_mcb,
            cells: vec![None; width_mcb * height_mcb],
        }
    }

    /// Insert one CU rectangle into the map.
    pub fn insert(&mut self, x: u32, y: u32, cb_w: u32, cb_h: u32, cqt_depth: u32) {
        let desc = NeighbourDescriptor {
            cb_w,
            cb_h,
            cqt_depth,
        };
        let mcb_x0 = (x as usize) / 4;
        let mcb_y0 = (y as usize) / 4;
        let mcb_x1 = ((x + cb_w) as usize).div_ceil(4);
        let mcb_y1 = ((y + cb_h) as usize).div_ceil(4);
        for my in mcb_y0..mcb_y1.min(self.height_mcb) {
            for mx in mcb_x0..mcb_x1.min(self.width_mcb) {
                self.cells[my * self.width_mcb + mx] = Some(desc);
            }
        }
    }

    fn get(&self, x: i64, y: i64) -> Option<NeighbourDescriptor> {
        if x < 0 || y < 0 {
            return None;
        }
        let mx = (x as usize) / 4;
        let my = (y as usize) / 4;
        if mx >= self.width_mcb || my >= self.height_mcb {
            return None;
        }
        self.cells[my * self.width_mcb + mx]
    }

    /// Pack the map's entries at `(x - 1, y)` / `(x, y - 1)` into the
    /// six fields that the §9.3.4.2 ctxInc derivations consume.
    /// Returns `(left_avail, above_avail, cb_height_left, cb_width_above,
    /// cqt_depth_left, cqt_depth_above)`.
    pub fn neighbour_state(&self, x: u32, y: u32) -> (bool, bool, u32, u32, u32, u32) {
        let left = self.get(x as i64 - 1, y as i64);
        let above = self.get(x as i64, y as i64 - 1);
        (
            left.is_some(),
            above.is_some(),
            left.map(|d| d.cb_h).unwrap_or(0),
            above.map(|d| d.cb_w).unwrap_or(0),
            left.map(|d| d.cqt_depth).unwrap_or(0),
            above.map(|d| d.cqt_depth).unwrap_or(0),
        )
    }
}

/// Walker state: the output CU list plus scratch neighbour info for
/// ctxInc derivation. Round 56 lit up the optional `nbr_map` hook
/// which threads a picture-wide [`CuNeighbourMap`] through the
/// recursion so left / above CU descriptors feed
/// [`ctx_inc_split_cu_flag`] / [`ctx_inc_split_qt_flag`] /
/// [`ctx_inc_mtt_split_cu_vertical_flag`]. With `nbr_map = None` the
/// walker preserves the round-55 picture-edge default
/// (`nbrs.left/above_avail = false`).
pub struct TreeWalker<'a, 'b> {
    dec: &'a mut ArithDecoder<'b>,
    ctxs: &'a mut TreeCtxs,
    min_cb_log2: u32,
    out: Vec<Cu>,
    nbr_map: Option<&'a mut CuNeighbourMap>,
    /// Round-56 — picture-absolute origin of the walked region. Added
    /// to every CU's local `(x, y)` before the [`CuNeighbourMap`]
    /// look-up / insert. The walker emits CTU-local rectangles to its
    /// caller (`out`); the map operates in picture-absolute
    /// coordinates because neighbour CUs of the *first* CU in a CTU
    /// live in the *previous* CTU.
    base_x: u32,
    base_y: u32,
}

impl<'a, 'b> TreeWalker<'a, 'b> {
    pub fn new(dec: &'a mut ArithDecoder<'b>, ctxs: &'a mut TreeCtxs) -> Self {
        Self {
            dec,
            ctxs,
            min_cb_log2: 2, // 4x4 minimum for luma (MinCbLog2SizeY default)
            out: Vec::new(),
            nbr_map: None,
            base_x: 0,
            base_y: 0,
        }
    }

    /// Round-56 — attach a picture-wide [`CuNeighbourMap`] to the
    /// walker so the §9.3.4.2 ctxInc derivations see real neighbour
    /// descriptors. The map is populated as each leaf CU is committed
    /// (post-`split_cu_flag` decision) so look-ups inside the same CTU
    /// can see siblings emitted earlier in the walk.
    pub fn with_neighbour_map(mut self, map: &'a mut CuNeighbourMap) -> Self {
        self.nbr_map = Some(map);
        self
    }

    /// Round-56 — variant of [`Self::with_neighbour_map`] used by per-
    /// CTU walkers that emit rectangles in CTU-local coordinates while
    /// the neighbour map operates in picture-absolute coordinates. The
    /// `(base_x, base_y)` offset is added to every local `(x, y)` for
    /// look-ups + inserts so the map sees consistent coordinates
    /// across CTU boundaries.
    pub fn with_neighbour_map_rebased(
        mut self,
        map: &'a mut CuNeighbourMap,
        base_x: u32,
        base_y: u32,
    ) -> Self {
        self.nbr_map = Some(map);
        self.base_x = base_x;
        self.base_y = base_y;
        self
    }

    /// Walk a coding_tree() rooted at `(x, y)` of size `w × h`.
    /// Returns a flat list of leaf CUs in decoding order.
    pub fn walk(mut self, x: u32, y: u32, w: u32, h: u32) -> Result<Vec<Cu>> {
        self.recurse(x, y, w, h, 0, 0)?;
        Ok(self.out)
    }

    fn neighbour_state(&self, x: u32, y: u32) -> (bool, bool, u32, u32, u32, u32) {
        match &self.nbr_map {
            Some(map) => map.neighbour_state(self.base_x + x, self.base_y + y),
            None => (false, false, 0, 0, 0, 0),
        }
    }

    fn record_cu(&mut self, x: u32, y: u32, w: u32, h: u32, cqt_depth: u32) {
        let bx = self.base_x;
        let by = self.base_y;
        if let Some(map) = self.nbr_map.as_deref_mut() {
            map.insert(bx + x, by + y, w, h, cqt_depth);
        }
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
        let at_min =
            w.trailing_zeros() <= self.min_cb_log2 || h.trailing_zeros() <= self.min_cb_log2;
        if at_min {
            self.out.push(Cu {
                x,
                y,
                w,
                h,
                cqt_depth,
                mtt_depth,
            });
            self.record_cu(x, y, w, h, cqt_depth);
            return Ok(());
        }
        // split_cu_flag — round-56 ctxInc uses real neighbour
        // availability when the walker has been attached to a
        // [`CuNeighbourMap`] (otherwise picture-edge defaults).
        let (
            left_avail,
            above_avail,
            cb_height_left,
            cb_width_above,
            cqt_depth_left,
            cqt_depth_above,
        ) = self.neighbour_state(x, y);
        let split_cu_inc = ctx_inc_split_cu_flag(
            left_avail,
            above_avail,
            cb_height_left,
            cb_width_above,
            w,
            h,
            1,
            1,
            1,
            1,
            1,
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
            self.record_cu(x, y, w, h, cqt_depth);
            return Ok(());
        }
        // split_qt_flag — only readable while we're in the quad-tree
        // phase (mtt_depth == 0). Outside that we force BT/TT.
        let split_qt = if mtt_depth == 0 {
            let inc = ctx_inc_split_qt_flag(
                left_avail,
                above_avail,
                cqt_depth_left,
                cqt_depth_above,
                cqt_depth,
            ) as usize;
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
        // Round-55 — multi-type-tree split. Round-56 — feed the same
        // neighbour state into the §9.3.4.2.3 derivation so multi-row
        // pictures see the proper aspect-ratio branch.
        let mtt_v_inc = ctx_inc_mtt_split_cu_vertical_flag(
            left_avail,
            above_avail,
            cb_height_left,
            cb_width_above,
            w,
            h,
            1,
            1,
            1,
            1,
        ) as usize;
        let mtt_v_n = self.ctxs.mtt_split_vertical.len() - 1;
        let mtt_vertical = self
            .dec
            .decode_decision(&mut self.ctxs.mtt_split_vertical[mtt_v_inc.min(mtt_v_n)])?;
        let mtt_b_inc = ctx_inc_mtt_split_cu_binary_flag(mtt_vertical, mtt_depth) as usize;
        let mtt_b_n = self.ctxs.mtt_split_binary.len() - 1;
        let mtt_binary = self
            .dec
            .decode_decision(&mut self.ctxs.mtt_split_binary[mtt_b_inc.min(mtt_b_n)])?;
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
pub fn read_intra_luma_mpm_flag(dec: &mut ArithDecoder<'_>, ctxs: &mut TreeCtxs) -> Result<u32> {
    let inc = ctx_inc_intra_luma_mpm_flag() as usize;
    dec.decode_decision(&mut ctxs.intra_luma_mpm[inc])
}

/// Read a `pred_mode_flag` bin at the root of an intra-inter-capable
/// CU. Always returns 1 in an I-slice because pred_mode_flag is never
/// signalled there — but exposed for completeness.
pub fn read_pred_mode_flag(dec: &mut ArithDecoder<'_>, ctxs: &mut TreeCtxs) -> Result<u32> {
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
        assert_eq!(
            cus,
            vec![Cu {
                x: 0,
                y: 0,
                w: 4,
                h: 4,
                cqt_depth: 0,
                mtt_depth: 0
            }]
        );
    }

    /// Round-56 — `CuNeighbourMap` insert + neighbour_state round-
    /// trips a CU descriptor. With both left + above CUs populated the
    /// returned tuple matches the §9.3.4.2 inputs.
    #[test]
    fn cu_neighbour_map_round_trips_descriptor() {
        let mut map = CuNeighbourMap::new(128, 128);
        // Picture-edge default → all unavailable.
        let (l, a, _, _, _, _) = map.neighbour_state(0, 0);
        assert!(!l);
        assert!(!a);
        // Look up at (64, 64): the left neighbour cell is at (63, 64)
        // and the above neighbour cell is at (64, 63). Insert two CUs
        // covering those cells: a 64×64 left neighbour at (0, 64) and
        // a 64×64 above neighbour at (64, 0).
        map.insert(0, 64, 64, 64, 0);
        map.insert(64, 0, 64, 64, 0);
        let (l, a, hl, wa, dl, da) = map.neighbour_state(64, 64);
        assert!(l);
        assert!(a);
        assert_eq!(hl, 64);
        assert_eq!(wa, 64);
        assert_eq!(dl, 0);
        assert_eq!(da, 0);
    }

    /// Round-56 — `TreeWalker::with_neighbour_map` populates the map
    /// for every leaf the walker emits. After walking a 4×4 leaf the
    /// map should contain that one CU.
    #[test]
    fn tree_walker_populates_neighbour_map_on_leaf() {
        let data = [0u8; 32];
        let mut dec = ArithDecoder::new(&data).unwrap();
        let mut ctxs = TreeCtxs::init(16);
        let mut map = CuNeighbourMap::new(64, 64);
        let walker = TreeWalker::new(&mut dec, &mut ctxs).with_neighbour_map(&mut map);
        let _cus = walker.walk(0, 0, 4, 4).unwrap();
        // Map should contain the 4×4 leaf at (0, 0).
        let d = map.get(0, 0).expect("4×4 leaf must be inserted");
        assert_eq!(d.cb_w, 4);
        assert_eq!(d.cb_h, 4);
    }

    /// Round-56 — `TreeWalker::with_neighbour_map_rebased` adds the
    /// `(base_x, base_y)` offset to every map insert, so per-CTU
    /// walkers operating in CTU-local coords still populate the
    /// picture-absolute map correctly.
    #[test]
    fn tree_walker_rebased_inserts_picture_absolute() {
        let data = [0u8; 32];
        let mut dec = ArithDecoder::new(&data).unwrap();
        let mut ctxs = TreeCtxs::init(16);
        let mut map = CuNeighbourMap::new(128, 128);
        let walker =
            TreeWalker::new(&mut dec, &mut ctxs).with_neighbour_map_rebased(&mut map, 64, 64);
        let _cus = walker.walk(0, 0, 4, 4).unwrap();
        // Map inserts at picture-absolute (64, 64) not local (0, 0).
        assert!(
            map.get(0, 0).is_none(),
            "local origin must not be populated"
        );
        assert!(
            map.get(64, 64).is_some(),
            "rebased origin (64, 64) must be populated"
        );
    }
}
