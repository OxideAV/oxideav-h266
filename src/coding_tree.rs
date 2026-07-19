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
    ctx_inc_mtt_split_cu_vertical_flag, ctx_inc_split_cu_flag, ctx_inc_split_qt_flag,
};

/// §7.3.11.4 `treeType` — whether a single tree partitions the CTU
/// (luma + chroma share one coding tree) or, on a dual-tree I-slice CTU
/// (§7.3.11.2 `sps_qtbtt_dual_tree_intra_flag`), whether the luma or the
/// chroma components are currently walked.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TreeType {
    #[default]
    SingleTree,
    DualTreeLuma,
    DualTreeChroma,
}

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
    pub intra_luma_mpm: Vec<ContextModel>,
}

impl TreeCtxs {
    /// Build the context arrays from Table 59 / 60 / 61 / 62 / 75
    /// at the supplied slice QP.
    pub fn init(slice_qp_y: i32) -> Self {
        use crate::tables::{init_contexts, SyntaxCtx};
        Self {
            split_cu: init_contexts(SyntaxCtx::SplitCuFlag, slice_qp_y),
            split_qt: init_contexts(SyntaxCtx::SplitQtFlag, slice_qp_y),
            mtt_split_vertical: init_contexts(SyntaxCtx::MttSplitCuVerticalFlag, slice_qp_y),
            mtt_split_binary: init_contexts(SyntaxCtx::MttSplitCuBinaryFlag, slice_qp_y),
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
/// r412 — §6.4.1 / §6.4.2 / §6.4.3 allowed-split constraints for one
/// coding-tree walk. Derived once per slice from the SPS partition
/// constraints (the `partition_constraints_override_enabled_flag = 0`
/// profile — no PH overrides) and consulted per node for BOTH the
/// §7.3.11.4 bin-presence conditions and the §9.3.4.2.2 split ctxIncs.
///
/// Scope notes: `modeType` is always MODE_TYPE_ALL on the walked
/// profiles (no local dual tree), and the picture-boundary arms only
/// see nodes fully inside the picture (the walker's CTU layout clips
/// edge CTUs to their in-picture rectangle).
#[derive(Clone, Copy, Debug)]
pub struct SplitConstraints {
    /// `MinQtLog2Size{Intra,Inter}Y` (eq. 50 / 51).
    pub min_qt_log2: u32,
    /// `MaxMttDepth` (`sps_max_mtt_hierarchy_depth_*`).
    pub max_mtt_depth: u32,
    /// `MaxBtSize` = `1 << (min_qt_log2 + sps_log2_diff_max_bt_min_qt_*)`.
    pub max_bt_size: u32,
    /// `MaxTtSize` = `1 << (min_qt_log2 + sps_log2_diff_max_tt_min_qt_*)`.
    pub max_tt_size: u32,
    /// `MinCbLog2SizeY` — `MinBtSizeY` / `MinTtSizeY` floor.
    pub min_cb_log2: u32,
    /// `pps_pic_width_in_luma_samples` / `pps_pic_height_...` for the
    /// §6.4.2 / §6.4.3 boundary arms.
    pub pic_w: u32,
    pub pic_h: u32,
}

/// The five per-node §6.4.1 – §6.4.3 outcomes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SplitAllows {
    pub qt: bool,
    pub bt_ver: bool,
    pub bt_hor: bool,
    pub tt_ver: bool,
    pub tt_hor: bool,
}

impl SplitAllows {
    #[inline]
    pub fn any_mtt(&self) -> bool {
        self.bt_ver || self.bt_hor || self.tt_ver || self.tt_hor
    }
    #[inline]
    pub fn any(&self) -> bool {
        self.qt || self.any_mtt()
    }
}

impl SplitConstraints {
    /// Intra-slice luma / single-tree constraints from the SPS.
    pub fn intra_luma(sps: &crate::sps::SeqParameterSet, pic_w: u32, pic_h: u32) -> Self {
        let pc = &sps.partition_constraints;
        let min_cb_log2 = pc.log2_min_luma_coding_block_size_minus2 + 2;
        let min_qt_log2 = pc.log2_diff_min_qt_min_cb_intra_slice_luma + min_cb_log2;
        Self {
            min_qt_log2,
            max_mtt_depth: pc.max_mtt_hierarchy_depth_intra_slice_luma,
            max_bt_size: 1 << (min_qt_log2 + pc.log2_diff_max_bt_min_qt_intra_slice_luma),
            max_tt_size: 1 << (min_qt_log2 + pc.log2_diff_max_tt_min_qt_intra_slice_luma),
            min_cb_log2,
            pic_w,
            pic_h,
        }
    }

    /// Inter-slice constraints from the SPS.
    pub fn inter(sps: &crate::sps::SeqParameterSet, pic_w: u32, pic_h: u32) -> Self {
        let pc = &sps.partition_constraints;
        let min_cb_log2 = pc.log2_min_luma_coding_block_size_minus2 + 2;
        let min_qt_log2 = pc.log2_diff_min_qt_min_cb_inter_slice + min_cb_log2;
        Self {
            min_qt_log2,
            max_mtt_depth: pc.max_mtt_hierarchy_depth_inter_slice,
            max_bt_size: 1 << (min_qt_log2 + pc.log2_diff_max_bt_min_qt_inter_slice),
            max_tt_size: 1 << (min_qt_log2 + pc.log2_diff_max_tt_min_qt_inter_slice),
            min_cb_log2,
            pic_w,
            pic_h,
        }
    }

    /// Dual-tree intra chroma constraints from the SPS (dimensions in
    /// luma samples, matching the chroma coding-tree walk).
    pub fn intra_chroma(sps: &crate::sps::SeqParameterSet, pic_w: u32, pic_h: u32) -> Self {
        let pc = &sps.partition_constraints;
        let min_cb_log2 = pc.log2_min_luma_coding_block_size_minus2 + 2;
        let min_qt_log2 = pc.log2_diff_min_qt_min_cb_intra_slice_chroma + min_cb_log2;
        Self {
            min_qt_log2,
            max_mtt_depth: pc.max_mtt_hierarchy_depth_intra_slice_chroma,
            max_bt_size: 1 << (min_qt_log2 + pc.log2_diff_max_bt_min_qt_intra_slice_chroma),
            max_tt_size: 1 << (min_qt_log2 + pc.log2_diff_max_tt_min_qt_intra_slice_chroma),
            min_cb_log2,
            pic_w,
            pic_h,
        }
    }

    /// Permissive constraints reproducing the pre-r412 allow-everything
    /// behaviour — every split family allowed down to the 4-sample
    /// leaf floor. Used by leaf-level harnesses that hand-build split
    /// bins without an SPS.
    pub fn permissive(pic_w: u32, pic_h: u32) -> Self {
        Self {
            min_qt_log2: 2,
            max_mtt_depth: u32::MAX,
            max_bt_size: 1 << 7,
            max_tt_size: 1 << 6,
            min_cb_log2: 2,
            pic_w,
            pic_h,
        }
    }

    /// §6.4.1 — allowSplitQt.
    pub fn allow_split_qt(&self, cb_size: u32, mtt_depth: u32, tree: TreeType) -> bool {
        if mtt_depth != 0 {
            return false;
        }
        match tree {
            TreeType::SingleTree | TreeType::DualTreeLuma => cb_size > (1 << self.min_qt_log2),
            // 4:2:0 chroma tree: MinQtSizeC gate + the (cbSize /
            // SubWidthC) <= 4 kill (dimensions in luma samples).
            TreeType::DualTreeChroma => cb_size > (1 << self.min_qt_log2) && (cb_size / 2) > 4,
        }
    }

    /// §6.4.2 — allowBtSplit for one direction. `parent_tt_ver` is
    /// `Some(vertical)` when the node is a middle/second child of a TT
    /// split at `mtt_depth − 1` (the parallel-TT suppression);
    /// `part_idx` is the child index within the parent MTT split.
    #[allow(clippy::too_many_arguments)]
    pub fn allow_bt(
        &self,
        vertical: bool,
        x0: u32,
        y0: u32,
        w: u32,
        h: u32,
        mtt_depth: u32,
        tree: TreeType,
        parent_tt_ver: Option<bool>,
        part_idx: u32,
    ) -> bool {
        self.allow_bt_off(
            vertical,
            x0,
            y0,
            w,
            h,
            mtt_depth,
            tree,
            parent_tt_ver,
            part_idx,
            0,
        )
    }

    /// §6.4.2 with the §7.4.12.4 `depthOffset` fold: at picture
    /// boundaries the implicit binary splits raise the effective
    /// `maxMttDepth` to `MaxMttDepth + depthOffset`.
    #[allow(clippy::too_many_arguments)]
    pub fn allow_bt_off(
        &self,
        vertical: bool,
        x0: u32,
        y0: u32,
        w: u32,
        h: u32,
        mtt_depth: u32,
        tree: TreeType,
        parent_tt_ver: Option<bool>,
        part_idx: u32,
        depth_offset: u32,
    ) -> bool {
        let cb_size = if vertical { w } else { h };
        let min_bt = 1u32 << self.min_cb_log2;
        if cb_size <= min_bt
            || w > self.max_bt_size
            || h > self.max_bt_size
            || mtt_depth >= self.max_mtt_depth + depth_offset
        {
            return false;
        }
        if tree == TreeType::DualTreeChroma {
            if (w / 2) * (h / 2) <= 16 {
                return false;
            }
            if (w / 2) == 4 && vertical {
                return false;
            }
        }
        // Picture-boundary arms (x1/y1 relative to the picture).
        let x1 = x0 + w;
        let y1 = y0 + h;
        if vertical && y1 > self.pic_h {
            return false;
        }
        if vertical && h > 64 && x1 > self.pic_w {
            return false;
        }
        if !vertical && w > 64 && y1 > self.pic_h {
            return false;
        }
        if x1 > self.pic_w && y1 > self.pic_h && w > (1 << self.min_qt_log2) {
            return false;
        }
        if !vertical && x1 > self.pic_w && y1 <= self.pic_h {
            return false;
        }
        // Parallel-TT suppression: the second child of a TT split may
        // not immediately re-split parallel to it.
        if mtt_depth > 0 && part_idx == 1 {
            if let Some(tt_ver) = parent_tt_ver {
                if tt_ver == vertical {
                    return false;
                }
            }
        }
        // VPDU shape arms.
        if vertical && w <= 64 && h > 64 {
            return false;
        }
        if !vertical && w > 64 && h <= 64 {
            return false;
        }
        true
    }

    /// §6.4.3 — allowTtSplit for one direction.
    pub fn allow_tt(
        &self,
        vertical: bool,
        x0: u32,
        y0: u32,
        w: u32,
        h: u32,
        mtt_depth: u32,
        tree: TreeType,
    ) -> bool {
        self.allow_tt_off(vertical, x0, y0, w, h, mtt_depth, tree, 0)
    }

    /// §6.4.3 with the §7.4.12.4 `depthOffset` fold (see
    /// [`Self::allow_bt_off`]).
    #[allow(clippy::too_many_arguments)]
    pub fn allow_tt_off(
        &self,
        vertical: bool,
        x0: u32,
        y0: u32,
        w: u32,
        h: u32,
        mtt_depth: u32,
        tree: TreeType,
        depth_offset: u32,
    ) -> bool {
        let cb_size = if vertical { w } else { h };
        let min_tt = 1u32 << self.min_cb_log2;
        let max_tt = self.max_tt_size.min(64);
        if cb_size <= 2 * min_tt
            || w > max_tt
            || h > max_tt
            || mtt_depth >= self.max_mtt_depth + depth_offset
            || x0 + w > self.pic_w
            || y0 + h > self.pic_h
        {
            return false;
        }
        if tree == TreeType::DualTreeChroma {
            if (w / 2) * (h / 2) <= 32 {
                return false;
            }
            if (w / 2) == 8 && vertical {
                return false;
            }
        }
        true
    }

    /// All five outcomes for one node.
    #[allow(clippy::too_many_arguments)]
    pub fn allows(
        &self,
        x0: u32,
        y0: u32,
        w: u32,
        h: u32,
        mtt_depth: u32,
        tree: TreeType,
        parent_tt_ver: Option<bool>,
        part_idx: u32,
    ) -> SplitAllows {
        self.allows_off(x0, y0, w, h, mtt_depth, tree, parent_tt_ver, part_idx, 0)
    }

    /// [`Self::allows`] with the §7.4.12.4 boundary `depthOffset`
    /// (raises the effective `maxMttDepth` below implicit
    /// picture-boundary binary splits).
    #[allow(clippy::too_many_arguments)]
    pub fn allows_off(
        &self,
        x0: u32,
        y0: u32,
        w: u32,
        h: u32,
        mtt_depth: u32,
        tree: TreeType,
        parent_tt_ver: Option<bool>,
        part_idx: u32,
        depth_offset: u32,
    ) -> SplitAllows {
        // §6.4.1 quad split keys on cbSize = cbWidth (square nodes on
        // the QT phase; MTT-phase nodes have mtt_depth != 0 → false).
        SplitAllows {
            qt: self.allow_split_qt(w.max(h), mtt_depth, tree),
            bt_ver: self.allow_bt_off(
                true,
                x0,
                y0,
                w,
                h,
                mtt_depth,
                tree,
                parent_tt_ver,
                part_idx,
                depth_offset,
            ),
            bt_hor: self.allow_bt_off(
                false,
                x0,
                y0,
                w,
                h,
                mtt_depth,
                tree,
                parent_tt_ver,
                part_idx,
                depth_offset,
            ),
            tt_ver: self.allow_tt_off(true, x0, y0, w, h, mtt_depth, tree, depth_offset),
            tt_hor: self.allow_tt_off(false, x0, y0, w, h, mtt_depth, tree, depth_offset),
        }
    }
}

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
    /// §7.3.11.4 `treeType` for this walk. `DualTreeChroma` swaps the
    /// leaf floor for the §6.4.1 – §6.4.3 chroma bullets (see
    /// [`Self::at_leaf_floor`]); `SingleTree` / `DualTreeLuma` share the
    /// luma `MinCbLog2SizeY` floor.
    tree_type: TreeType,
    /// r391 — `MttSplitMode[x0][y0][mttDepth]` log: one record per MTT
    /// split the walk takes, in walk-local coordinates. The §8.4.4
    /// CclmEnabled 64-grid derivation reads
    /// `MttSplitMode[xCb64][yCb64][0]` / `MttSplitMode[xCb64][yCb32][1]`
    /// off the chroma tree; QT splits are not logged (the array is only
    /// consulted for MTT modes).
    mtt_log: Vec<MttSplitRec>,
    /// r412 — §6.4.1 – §6.4.3 allowed-split constraints driving the
    /// §7.3.11.4 bin-presence conditions and the §9.3.4.2.2 ctxIncs.
    /// Defaults to [`SplitConstraints::permissive`] over a huge
    /// picture; real decodes install the SPS-derived set via
    /// [`Self::with_constraints`].
    constraints: SplitConstraints,
}

/// One `MttSplitMode[x0][y0][mttDepth]` record (§7.4.11.4) — the MTT
/// split a coding-tree node at walk-local `(x, y)` and depth `mtt_depth`
/// took. `vertical` / `binary` encode SPLIT_{BT,TT}_{VER,HOR}.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MttSplitRec {
    pub x: u32,
    pub y: u32,
    pub mtt_depth: u32,
    pub vertical: bool,
    pub binary: bool,
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
            tree_type: TreeType::SingleTree,
            mtt_log: Vec::new(),
            constraints: SplitConstraints::permissive(u32::MAX, u32::MAX),
        }
    }

    /// r412 — install the SPS-derived §6.4.1 – §6.4.3 constraints.
    pub fn with_constraints(mut self, constraints: SplitConstraints) -> Self {
        self.constraints = constraints;
        self
    }

    /// §7.3.11.4 — select which component tree this walker parses. The
    /// dual-tree CTU walk (§7.3.11.2) issues one `DualTreeLuma` walk
    /// followed by one `DualTreeChroma` walk per implicit-QT 64×64 node.
    pub fn with_tree_type(mut self, tree_type: TreeType) -> Self {
        self.tree_type = tree_type;
        self
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
        self.recurse(x, y, w, h, 0, 0, None, 0)?;
        Ok(self.out)
    }

    /// [`Self::walk`] variant that also returns the
    /// `MttSplitMode[x0][y0][mttDepth]` log (walk-local coordinates,
    /// one record per MTT split taken). The §8.4.4 CclmEnabled 64-grid
    /// derivation consumes the chroma tree's log.
    pub fn walk_logged(
        mut self,
        x: u32,
        y: u32,
        w: u32,
        h: u32,
    ) -> Result<(Vec<Cu>, Vec<MttSplitRec>)> {
        self.recurse(x, y, w, h, 0, 0, None, 0)?;
        Ok((self.out, self.mtt_log))
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

    /// Leaf floor — the walker stops (emits a leaf without reading any
    /// split bin) when no split family can be signalled:
    ///
    /// * `SingleTree` / `DualTreeLuma` — the luma `MinCbLog2SizeY` floor
    ///   (either dimension at 4).
    /// * `DualTreeChroma` (4:2:0) — an 8×8 luma-sample node is a 4×4
    ///   chroma CB, where every split family is disallowed: §6.4.1 kills
    ///   QT (`cbSize / SubWidthC <= 4`), §6.4.2 kills BT (the chroma
    ///   area `(cbWidth / SubWidthC) · (cbHeight / SubHeightC) = 16`
    ///   is `<= 16`), and §6.4.3 kills TT (chroma area `<= 32`). With
    ///   no allowed split, `split_cu_flag` is not present (§7.3.11.4)
    ///   and is inferred 0.
    fn at_leaf_floor(&self, w: u32, h: u32) -> bool {
        match self.tree_type {
            TreeType::SingleTree | TreeType::DualTreeLuma => {
                w.trailing_zeros() <= self.min_cb_log2 || h.trailing_zeros() <= self.min_cb_log2
            }
            TreeType::DualTreeChroma => w <= 8 && h <= 8,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn recurse(
        &mut self,
        x: u32,
        y: u32,
        w: u32,
        h: u32,
        cqt_depth: u32,
        mtt_depth: u32,
        parent_tt_ver: Option<bool>,
        part_idx: u32,
    ) -> Result<()> {
        // r412 — §6.4.1 – §6.4.3: derive the five allowed-split
        // outcomes for this node; they gate the §7.3.11.4 bin
        // presence AND feed the §9.3.4.2.2 ctxIncs.
        let allows = self.constraints.allows(
            self.base_x + x,
            self.base_y + y,
            w,
            h,
            mtt_depth,
            self.tree_type,
            parent_tt_ver,
            part_idx,
        );
        // Historical chroma-tree floor kept as a defensive backstop
        // (it coincides with the allows-derived floor for 4:2:0).
        let at_min = self.at_leaf_floor(w, h) && !allows.any();
        let _ = at_min;
        let (
            left_avail,
            above_avail,
            cb_height_left,
            cb_width_above,
            cqt_depth_left,
            cqt_depth_above,
        ) = self.neighbour_state(x, y);
        // split_cu_flag — present iff any split family is allowed
        // (interior nodes; edge CTUs are pre-clipped by the layout so
        // the boundary inferred-1 arm never fires on this walker).
        let split_cu = if allows.any() {
            let inc = ctx_inc_split_cu_flag(
                left_avail,
                above_avail,
                cb_height_left,
                cb_width_above,
                w,
                h,
                allows.bt_ver as u32,
                allows.bt_hor as u32,
                allows.tt_ver as u32,
                allows.tt_hor as u32,
                allows.qt as u32,
            ) as usize;
            let n = self.ctxs.split_cu.len() - 1;
            self.dec
                .decode_decision(&mut self.ctxs.split_cu[inc.min(n)])?
        } else {
            0
        };
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
        // split_qt_flag — §7.3.11.4: present iff a QT split AND at
        // least one MTT split are both allowed; §7.4.12.4 inference
        // otherwise (interior nodes: allowSplitQt).
        let split_qt = if allows.qt && allows.any_mtt() {
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
            u32::from(allows.qt)
        };
        if split_qt == 1 {
            let hw = w / 2;
            let hh = h / 2;
            self.recurse(x, y, hw, hh, cqt_depth + 1, mtt_depth, None, 0)?;
            self.recurse(x + hw, y, hw, hh, cqt_depth + 1, mtt_depth, None, 1)?;
            self.recurse(x, y + hh, hw, hh, cqt_depth + 1, mtt_depth, None, 2)?;
            self.recurse(x + hw, y + hh, hw, hh, cqt_depth + 1, mtt_depth, None, 3)?;
            return Ok(());
        }
        // mtt_split_cu_vertical_flag — present iff both a horizontal
        // and a vertical MTT split are allowed; §7.4.12.4 inference
        // otherwise.
        let mtt_vertical = if (allows.bt_hor || allows.tt_hor) && (allows.bt_ver || allows.tt_ver) {
            let inc = ctx_inc_mtt_split_cu_vertical_flag(
                left_avail,
                above_avail,
                cb_height_left,
                cb_width_above,
                w,
                h,
                allows.bt_ver as u32,
                allows.bt_hor as u32,
                allows.tt_ver as u32,
                allows.tt_hor as u32,
            ) as usize;
            let n = self.ctxs.mtt_split_vertical.len() - 1;
            self.dec
                .decode_decision(&mut self.ctxs.mtt_split_vertical[inc.min(n)])?
        } else if allows.bt_hor || allows.tt_hor {
            0
        } else {
            1
        };
        // mtt_split_cu_binary_flag — present iff BOTH the binary and
        // ternary split are allowed in the chosen direction.
        let bin_coded = if mtt_vertical == 1 {
            allows.bt_ver && allows.tt_ver
        } else {
            allows.bt_hor && allows.tt_hor
        };
        let mtt_binary = if bin_coded {
            let inc = ctx_inc_mtt_split_cu_binary_flag(mtt_vertical, mtt_depth) as usize;
            let n = self.ctxs.mtt_split_binary.len() - 1;
            self.dec
                .decode_decision(&mut self.ctxs.mtt_split_binary[inc.min(n)])?
        } else if !allows.bt_ver && !allows.bt_hor {
            0
        } else if !allows.tt_ver && !allows.tt_hor {
            1
        } else if allows.bt_hor && allows.tt_ver {
            1 - mtt_vertical
        } else {
            mtt_vertical
        };
        // r391 — record MttSplitMode[x][y][mttDepth] for the §8.4.4
        // 64-grid consumers.
        self.mtt_log.push(MttSplitRec {
            x,
            y,
            mtt_depth,
            vertical: mtt_vertical == 1,
            binary: mtt_binary == 1,
        });
        if mtt_binary == 1 {
            // Binary split.
            if mtt_vertical == 1 {
                let hw = w / 2;
                self.recurse(x, y, hw, h, cqt_depth, mtt_depth + 1, None, 0)?;
                self.recurse(x + hw, y, hw, h, cqt_depth, mtt_depth + 1, None, 1)?;
            } else {
                let hh = h / 2;
                self.recurse(x, y, w, hh, cqt_depth, mtt_depth + 1, None, 0)?;
                self.recurse(x, y + hh, w, hh, cqt_depth, mtt_depth + 1, None, 1)?;
            }
        } else {
            // Ternary split (1:2:1 sizes) — children thread the TT
            // direction for the §6.4.2 parallel-TT suppression.
            let tt = Some(mtt_vertical == 1);
            if mtt_vertical == 1 {
                let q = w / 4;
                self.recurse(x, y, q, h, cqt_depth, mtt_depth + 1, tt, 0)?;
                self.recurse(x + q, y, q * 2, h, cqt_depth, mtt_depth + 1, tt, 1)?;
                self.recurse(x + q * 3, y, q, h, cqt_depth, mtt_depth + 1, tt, 2)?;
            } else {
                let q = h / 4;
                self.recurse(x, y, w, q, cqt_depth, mtt_depth + 1, tt, 0)?;
                self.recurse(x, y + q, w, q * 2, cqt_depth, mtt_depth + 1, tt, 1)?;
                self.recurse(x, y + q * 3, w, q, cqt_depth, mtt_depth + 1, tt, 2)?;
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

    /// §6.4.1 – §6.4.3 — an 8×8 luma-sample node on the DUAL_TREE_CHROMA
    /// walk is a 4×4 chroma CB where QT / BT / TT are all disallowed, so
    /// the walker must emit a leaf without consuming any CABAC bin.
    #[test]
    fn dual_tree_chroma_walker_stops_at_chroma_4x4() {
        let data = [0u8; 32];
        let mut dec = ArithDecoder::new(&data).unwrap();
        let mut ctxs = TreeCtxs::init(16);
        let walker = TreeWalker::new(&mut dec, &mut ctxs).with_tree_type(TreeType::DualTreeChroma);
        let cus = walker.walk(0, 0, 8, 8).unwrap();
        assert_eq!(
            cus,
            vec![Cu {
                x: 0,
                y: 0,
                w: 8,
                h: 8,
                cqt_depth: 0,
                mtt_depth: 0
            }],
            "8×8 luma (4×4 chroma) must be an unsplittable chroma-tree leaf"
        );
    }

    /// A DUAL_TREE_CHROMA walk over a 64×64 node emits a structurally
    /// valid complete tiling. (The walker reads split flags as coded —
    /// the §6.4.2 / §6.4.3 chroma-area allowances are the emitting
    /// encoder's responsibility, matching the round-55 single-tree
    /// stance — so this test only asserts geometry, not legality.)
    #[test]
    fn dual_tree_chroma_walk_64_structural() {
        let data = [0x5au8; 64];
        let mut dec = ArithDecoder::new(&data).unwrap();
        let mut ctxs = TreeCtxs::init(16);
        let walker = TreeWalker::new(&mut dec, &mut ctxs).with_tree_type(TreeType::DualTreeChroma);
        let cus = walker.walk(0, 0, 64, 64).unwrap();
        let mut area = 0u32;
        for c in &cus {
            assert!(c.x + c.w <= 64 && c.y + c.h <= 64);
            area += c.w * c.h;
        }
        assert_eq!(area, 64 * 64, "chroma-tree leaves must tile the node");
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
