//! VVC §8.5.5.7 affine AMVP — luma affine control-point motion-vector
//! predictor candidate list derivation.
//!
//! Round 120 lands the §8.5.5.7 driver, the §8.5.5.8 constructed
//! affine CPMV-predictor inner derivation, and a `select_affine_mvp`
//! helper that consumes `mvp_lX_flag` (round-108) and feeds the
//! AMVR-shifted per-CP mvd (round-103) through the eq. 664 – 667
//! `mvLX[cpIdx] = mvpLX[cpIdx] + mvdLX[cpIdx]` MODE+ wrap (with
//! `Clip3(−2^17, 2^17 − 1)`) to produce the final per-CB CPMVs.
//!
//! ## What this module ships
//!
//! 1. [`derive_inherited_affine_mvp_candidate`] — the per-neighbour
//!    inherited CPMVP wrapper around the existing round-91
//!    [`crate::affine_merge::derive_inherited_affine_cpmvs`]. Adds the
//!    §8.5.5.7 step-4 / step-5 per-list AMVP gate (list X first then
//!    list Y = 1 − X; `DiffPicOrderCnt == 0`), and AMVR-rounds every
//!    emitted CP component per the §8.5.5.7 "rounding process for
//!    motion vectors" inner bullet.
//! 2. [`derive_constructed_affine_mvp_candidate`] — the §8.5.5.8
//!    constructed CPMV-predictor inner derivation. Reads per-corner
//!    neighbour MVs at TL (`B2 / B3 / A2` cascade), TR (`B1 / B0`),
//!    BL (`A1 / A0`); each corner picks the first effectively-
//!    available neighbour with `PredFlagLX == 1 ∧ DiffPicOrderCnt == 0`
//!    (cross-list fallback for Y = 1 − X), assigns the neighbour MV
//!    directly (eqs. 841 – 846), and AMVR-rounds. Emits
//!    `availableConsFlagLX = 1` only when all `numCpMv` corners are
//!    available (the spec's `availableFlagLX[0] && availableFlagLX[1]
//!    && availableFlagLX[2] == 1`, or the 4-param trailing-bullet
//!    `availableFlagLX[0] && availableFlagLX[1] && MotionModelIdc ==
//!    1` shortcut).
//! 3. [`build_affine_mvp_cand_list`] — §8.5.5.7 steps 1 – 9: drives
//!    the inherited A-scan + B-scan, the constructed CPMV append, the
//!    per-corner standalone-candidate fallback (step 7's
//!    `for nbCpIdx = 2..0`), the §8.5.2.11 temporal candidate (with
//!    its `numCpMv`-copy per-CP expansion), and the eqs. 838 / 839
//!    zero-MV pad to exactly `MAX_AFFINE_MVP_CAND == 2` entries.
//! 4. [`select_affine_mvp`] — eq. 840: pick
//!    `cpMvpListLX[mvp_lX_flag][cpIdx]`.
//! 5. [`derive_final_affine_cpmvs`] — eqs. 664 – 667: fold per-CP
//!    AMVR-shifted `mvdCpLX` into the chosen predictor.
//!
//! ## Affine AMVP differs from regular AMVP in five ways
//!
//! * **Per-CP, not per-CU.** The list is `cpMvpListLX[mvpIdx][cpIdx]`
//!   rather than `mvpListLX[mvpIdx]`. Each list slot carries 2 or 3
//!   CPMVs.
//! * **Inherited from neighbour CPMVs.** The A / B scans pull
//!   neighbour CPMVs via §8.5.5.5 (not the neighbour's stored MV at the
//!   sample-aligned corner), so a neighbour's affine slope reaches the
//!   current CB. The §8.5.5.5 derivation maps the neighbour's CPMVs to
//!   the current CB's origin / top-right / bottom-left positions.
//! * **Step-4 cross-list fallback is identical.** Like §8.5.2.10 the
//!   scan picks list X first, then list Y = 1 − X — using the same
//!   `DiffPicOrderCnt == 0` gate.
//! * **Constructed CPMVP is per-corner, not per-triple.** §8.5.5.8
//!   reads neighbour MVs at each of the three (or two) corners
//!   directly; it does **not** synthesise CPMVs from triples like
//!   §8.5.5.6 (which is the *merge* path's constructed candidate).
//! * **Step-7 standalone-corner fallback.** When the constructed list
//!   came up short, the spec emits *single-corner* candidates: it
//!   takes one available constructed corner's MV and replicates it for
//!   every cpIdx position. There are up to three such entries
//!   (`nbCpIdx = 2, 1, 0`).
//!
//! ## Out of scope (deferred to later rounds)
//!
//! * The CTU-walker fuse — populating [`NeighbourAffineQuery`] from the
//!   live per-CB CPMV grid + the §6.4.4 neighbour-availability table
//!   wired into the actual coding-tree walk. The driver here takes
//!   pure-data inputs so callers can stage them as the walker comes
//!   online. The wire-up is a CTU-walker change, not an §8.5.5.7
//!   change.
//! * The §8.5.5.7 step 8 temporal collocated invocation: this module
//!   takes an `Option<MotionVector>` for `mvLXCol` so the caller
//!   resolves `ColPic` once and runs the (per-CB, not per-CP) §8.5.2.11
//!   walk via [`crate::amvp::derive_temporal_amvp_candidate`]; the
//!   §8.5.5.7 driver then replicates that single MV across every CP
//!   position. Same wiring deferred as the regular-AMVP path.
//!
//! ## Spec reference
//!
//! ITU-T H.266 | ISO/IEC 23090-3 (V4, 01/2026):
//! * §8.5.5.7 — "Derivation process for luma affine control point
//!   motion vector predictors" (eqs. 819 – 840 + the per-step
//!   conditional pseudo-code).
//! * §8.5.5.8 — "Derivation process for constructed affine control
//!   point motion vector prediction candidates" (eqs. 841 – 846 + the
//!   per-corner cascades).
//! * §8.5.5.5 — "Derivation process for luma affine control point
//!   motion vectors from a neighbouring block" — invoked from §8.5.5.7
//!   step 4 / 5 and re-used here via
//!   [`crate::affine_merge::derive_inherited_affine_cpmvs`].
//! * §8.5.2.14 — "Rounding process for motion vectors" with
//!   `rightShift = leftShift = AmvrShift` (eqs. 608 – 610).
//! * §8.5.2.1 / §7.3.10.10 — the eq. 664 – 667 final MV folding
//!   wrapping `(mvpCpLX[i] + mvdCpLX[i]) & (2^18 − 1)` with the eq.
//!   665 / 667 `>= 2^17 ? ... − 2^18 : ...` two's-complement unwrap.
//!
//! No third-party VVC decoder source was consulted; the
//! implementation is spec-only.

use crate::affine::{AffineCpmvs, MotionModel};
use crate::affine_merge::{
    derive_inherited_affine_cpmvs, InheritedAffineGeom, NeighbourCpmvSource,
};
use crate::amvr::AmvrShift;
use crate::inter::MotionVector;

/// Maximum length of the affine MVP candidate list (§8.5.5.7 step 9 —
/// the list is padded to exactly 2 entries).
pub const MAX_AFFINE_MVP_CAND: usize = 2;

/// Maximum CPMV count per candidate (numCpMv ∈ {2, 3}). Buffers are
/// sized to 3.
pub const MAX_NUM_CP_MV: usize = 3;

/// Reference-list selector — the list the current CU predicts on.
/// Mirrors [`crate::amvp::RefList`] in shape so the §8.5.5.7 driver
/// (which is per-list) feels analogous to the round-111 regular-AMVP
/// `build_mvp_cand_list`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RefList {
    L0,
    L1,
}

impl RefList {
    /// `1 − X` — the opposite list consulted as the §8.5.5.7
    /// cross-list fallback.
    pub fn other(self) -> RefList {
        match self {
            RefList::L0 => RefList::L1,
            RefList::L1 => RefList::L0,
        }
    }
}

/// One affine-MVP candidate — `numCpMv` CPMVs packed into a fixed-size
/// array. Slot `i` carries `cpMvpListLX[mvpIdx][i]`. Components are in
/// canonical 1/16-luma units, AMVR-rounded per §8.5.5.7's step-4 /
/// step-5 / step-7 / step-8 rounding bullets.
///
/// Unused slots (e.g. slot 2 for `numCpMv == 2`) carry
/// [`MotionVector::ZERO`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AffineMvpCandidate {
    /// Number of valid CPMVs (2 or 3, mirroring `numCpMv`).
    pub num_cp_mv: u32,
    /// CPMVs, slot `i` ≡ `cpMvpListLX[*][i]`.
    pub cp_mvs: [MotionVector; MAX_NUM_CP_MV],
}

impl AffineMvpCandidate {
    /// Wrap an [`AffineCpmvs`] record into an [`AffineMvpCandidate`].
    pub fn from_cpmvs(cpmvs: AffineCpmvs) -> Self {
        let num_cp_mv = cpmvs.model.num_cp_mv() as u32;
        let mut cp_mvs = [MotionVector::ZERO; MAX_NUM_CP_MV];
        for i in 0..num_cp_mv as usize {
            cp_mvs[i] = cpmvs.cpmvs[i];
        }
        Self { num_cp_mv, cp_mvs }
    }

    /// Build an all-CPMVs-equal candidate. Used for §8.5.5.7 step 7
    /// (single-corner replication) and step 8 (temporal MV replicated
    /// across every CP) and step 9 (zero-MV pad).
    pub fn from_single_mv(mv: MotionVector, num_cp_mv: u32) -> Self {
        debug_assert!((2..=3).contains(&num_cp_mv));
        let mut cp_mvs = [MotionVector::ZERO; MAX_NUM_CP_MV];
        for slot in cp_mvs.iter_mut().take(num_cp_mv as usize) {
            *slot = mv;
        }
        Self { num_cp_mv, cp_mvs }
    }
}

/// One neighbour CB carrying its affine state — input to the §8.5.5.7
/// inherited A / B scan and to the §8.5.5.8 constructed CPMVP derivation.
///
/// The §6.4.4 neighbour-availability table is folded into
/// [`Self::available`] by the caller (the CTU walker), AND-ed with the
/// parallel-merge-level suppression (`xCb >> Log2ParMrgLevel ==
/// xNbN >> Log2ParMrgLevel && yCb >> Log2ParMrgLevel == yNbN >>
/// Log2ParMrgLevel ⇒ availableN = FALSE`, eq. 60). For inherited-scan
/// purposes a translational neighbour (`motion_model.is_translational()`)
/// is rejected by §8.5.5.7's "MotionModelIdc > 0" gate — but the
/// constructed-CPMVP derivation in §8.5.5.8 accepts translational
/// neighbours (it reads the neighbour's per-corner sample MV directly,
/// not the CPMV record).
#[derive(Clone, Copy, Debug)]
pub struct NeighbourAffineQuery {
    /// `availableN` per the §6.4.4 derivation — AND'd with the
    /// parallel-merge-level suppression (eq. 60).
    pub available: bool,
    /// `predFlagL0[xNb][yNb]` — true iff the neighbour predicts on L0.
    pub pred_flag_l0: bool,
    /// `predFlagL1[xNb][yNb]`.
    pub pred_flag_l1: bool,
    /// `RefIdxL0[xNb][yNb]` — the neighbour's L0 reference index.
    /// `-1` when `!pred_flag_l0`.
    pub ref_idx_l0: i32,
    /// `RefIdxL1[xNb][yNb]`.
    pub ref_idx_l1: i32,
    /// Neighbour CB top-left luma sample (picture-absolute).
    /// `(CbPosX[0][xNb][yNb], CbPosY[0][xNb][yNb])`.
    pub xnb: i32,
    pub ynb: i32,
    /// Neighbour CB dimensions in luma samples.
    pub nb_w: u32,
    pub nb_h: u32,
    /// `MotionModelIdc[xNb][yNb]` — the neighbour's affine model.
    /// Translational (idc == 0) suppresses the §8.5.5.5 invocation
    /// per the "MotionModelIdc[xNbN][yNbN] is greater than 0" gate in
    /// step 4 / step 5; constructed-CPMVP (§8.5.5.8) accepts it.
    pub motion_model: MotionModel,
    /// Neighbour's stored CPMV record on L0 — read by §8.5.5.5 when
    /// `pred_flag_l0 && motion_model.is_affine()`. `cpmvs[0]` carries
    /// the sample-anchor MV when the neighbour was translational (used
    /// by §8.5.5.8's per-corner read).
    pub cpmvs_l0: AffineCpmvs,
    /// Neighbour's L1 CPMV record (B-slice neighbours).
    pub cpmvs_l1: AffineCpmvs,
}

impl NeighbourAffineQuery {
    /// Spec's all-unavailable corner — every later invocation reads
    /// `available == false` and short-circuits.
    pub const UNAVAILABLE: NeighbourAffineQuery = NeighbourAffineQuery {
        available: false,
        pred_flag_l0: false,
        pred_flag_l1: false,
        ref_idx_l0: -1,
        ref_idx_l1: -1,
        xnb: 0,
        ynb: 0,
        nb_w: 0,
        nb_h: 0,
        motion_model: MotionModel::Translational,
        cpmvs_l0: AffineCpmvs {
            model: MotionModel::Translational,
            cpmvs: [MotionVector::ZERO; 3],
        },
        cpmvs_l1: AffineCpmvs {
            model: MotionModel::Translational,
            cpmvs: [MotionVector::ZERO; 3],
        },
    };
}

/// Per-CU AMVP reference context — POC-matching gate for the
/// `DiffPicOrderCnt == 0` test used by every §8.5.5.7 / §8.5.5.8 scan.
///
/// Same shape as [`crate::amvp::AmvpRefContext`] but lives here so the
/// affine module stands on its own. The closures resolve a neighbour's
/// per-list reference index to its reference picture's POC; they
/// return `None` for the `-1` "no reference" sentinel.
pub struct AffineMvpRefContext<'a> {
    /// `X` — the list the current CU predicts on.
    pub list: RefList,
    /// POC of `RefPicList[X][refIdxLX]` for the current CU.
    pub current_ref_poc: i32,
    /// Resolve a neighbour's L0 reference index → that reference's POC.
    pub poc_of_l0_ref: &'a dyn Fn(i32) -> Option<i32>,
    /// Resolve a neighbour's L1 reference index → that reference's POC.
    pub poc_of_l1_ref: &'a dyn Fn(i32) -> Option<i32>,
}

impl AffineMvpRefContext<'_> {
    /// Try the neighbour's `which`-list prediction: return its POC iff
    /// `predFlag(which) && resolver(refIdx)` succeed.
    fn neighbour_list_poc(&self, nb: &NeighbourAffineQuery, which: RefList) -> Option<i32> {
        match which {
            RefList::L0 if nb.pred_flag_l0 => (self.poc_of_l0_ref)(nb.ref_idx_l0),
            RefList::L1 if nb.pred_flag_l1 => (self.poc_of_l1_ref)(nb.ref_idx_l1),
            _ => None,
        }
    }

    /// §8.5.5.7 step-4 / step-5 cross-list scan: try list X first, then
    /// list Y = 1 − X. Returns the picked list (so the caller knows
    /// which CPMV record / which sample-anchor MV to read).
    fn pick_list(&self, nb: &NeighbourAffineQuery) -> Option<RefList> {
        if let Some(poc) = self.neighbour_list_poc(nb, self.list) {
            if poc == self.current_ref_poc {
                return Some(self.list);
            }
        }
        if let Some(poc) = self.neighbour_list_poc(nb, self.list.other()) {
            if poc == self.current_ref_poc {
                return Some(self.list.other());
            }
        }
        None
    }
}

// =====================================================================
// §8.5.5.5 wrapper for the AMVP path: derive the inherited CPMVP from
// one neighbour CB, with the §8.5.5.7 list-X-then-list-Y gate, then
// AMVR-round every CP.
// =====================================================================

/// §8.5.5.7 step 4 / step 5 — derive the inherited affine CPMVP from
/// **one** neighbour CB.
///
/// Returns `None` when:
/// * The neighbour is unavailable, OR
/// * The neighbour is translational (`MotionModelIdc == 0` — the spec's
///   `MotionModelIdc[xNbN][yNbN] > 0` gate fails), OR
/// * Neither list X nor list Y matches the current CU's reference picture
///   (no `DiffPicOrderCnt == 0` hit), OR
/// * The §8.5.5.5 derivation fails its preconditions.
///
/// When the neighbour spans the CTU-boundary above the current CU,
/// `source_for_neighbour` builds the [`NeighbourCpmvSource::AboveCtuBoundary`]
/// variant; otherwise it builds the [`NeighbourCpmvSource::SameOrLeftCtu`]
/// variant. This is the same data the §8.5.5.5 inherited-merge path
/// consumes.
pub fn derive_inherited_affine_mvp_candidate(
    xcb: i32,
    ycb: i32,
    cb_w: u32,
    cb_h: u32,
    nb: &NeighbourAffineQuery,
    ctx: &AffineMvpRefContext<'_>,
    amvr: AmvrShift,
    num_cp_mv: u32,
    is_ctu_boundary_above: bool,
) -> Option<AffineMvpCandidate> {
    if !nb.available {
        return None;
    }
    // §8.5.5.7 step-4 inner gate: "MotionModelIdc[xNbAk][yNbAk] > 0".
    if matches!(nb.motion_model, MotionModel::Translational) {
        return None;
    }
    // §8.5.5.7's "PredFlagLX[xNb][yNb] == 1 AND
    // DiffPicOrderCnt(RefPicList[X][RefIdxLX[xNb][yNb]],
    // RefPicList[X][refIdxLX]) == 0" gate. Picks list X first, then
    // list Y = 1 − X.
    let picked_list = ctx.pick_list(nb)?;

    let source = if is_ctu_boundary_above {
        let cpmvs = match picked_list {
            RefList::L0 => nb.cpmvs_l0,
            RefList::L1 => nb.cpmvs_l1,
        };
        // The §8.5.5.5 `isCTUboundary` path needs the neighbour's two
        // bottom-row sub-block MVs (eqs. 736 – 739). We pre-compute them
        // from the neighbour CPMV record via the §8.5.5.9 driver, which
        // mirrors what the spec would read from the neighbour's
        // already-derived MvLX[ ][ ] sub-block grid.
        let (mv_bl, mv_br) = derive_bottom_row_subblock_mvs(&cpmvs, nb.nb_w, nb.nb_h);
        NeighbourCpmvSource::AboveCtuBoundary {
            mv_bottom_left: mv_bl,
            mv_bottom_right: mv_br,
        }
    } else {
        NeighbourCpmvSource::SameOrLeftCtu {
            cpmvs: match picked_list {
                RefList::L0 => nb.cpmvs_l0,
                RefList::L1 => nb.cpmvs_l1,
            },
        }
    };

    let geom = InheritedAffineGeom {
        xcb,
        ycb,
        cb_width: cb_w,
        cb_height: cb_h,
        xnb: nb.xnb,
        ynb: nb.ynb,
        nb_w: nb.nb_w,
        nb_h: nb.nb_h,
    };
    let inherited = derive_inherited_affine_cpmvs(geom, source, num_cp_mv).ok()?;

    // §8.5.5.7 inner bullet: §8.5.2.14 round with
    // `rightShift = leftShift = AmvrShift`, applied per-component.
    let mut cp_mvs = [MotionVector::ZERO; MAX_NUM_CP_MV];
    for i in 0..num_cp_mv as usize {
        cp_mvs[i] = round_mv_amvr(inherited.cpmvs[i], amvr);
    }
    Some(AffineMvpCandidate { num_cp_mv, cp_mvs })
}

/// Compute the neighbour's two bottom-row sub-block MVs from its CPMV
/// record. Mirrors what the §8.5.5.9 sub-block MV derivation would write
/// at `(sbIdxX = 0, sbIdxY = numSbY − 1)` (eq. 870 / 871 give
/// `xPosCb = 2 + 0 = 2`, `yPosCb = 2 + 4*(numSbY − 1)`) and
/// `(sbIdxX = numSbX − 1, sbIdxY = numSbY − 1)`. We sample at the same
/// centre offsets so this stays the "MvLX[x][y]" array the spec eqs.
/// 736 – 739 read.
fn derive_bottom_row_subblock_mvs(
    cpmvs: &AffineCpmvs,
    nb_w: u32,
    nb_h: u32,
) -> (MotionVector, MotionVector) {
    debug_assert!(nb_w.is_power_of_two() && nb_h.is_power_of_two());
    let log2_w = nb_w.trailing_zeros() as i32;
    let log2_h = nb_h.trailing_zeros() as i32;

    // §8.5.5.9 eqs. 850 – 857.
    let mv_scale_hor = (cpmvs.cpmvs[0].x as i64) << 7;
    let mv_scale_ver = (cpmvs.cpmvs[0].y as i64) << 7;
    let d_hor_x = ((cpmvs.cpmvs[1].x as i64) - (cpmvs.cpmvs[0].x as i64)) << (7 - log2_w).max(0);
    let d_ver_x = ((cpmvs.cpmvs[1].y as i64) - (cpmvs.cpmvs[0].y as i64)) << (7 - log2_w).max(0);
    let (d_hor_y, d_ver_y) = match cpmvs.model {
        MotionModel::Affine6Param => {
            let s = (7 - log2_h).max(0);
            (
                ((cpmvs.cpmvs[2].x as i64) - (cpmvs.cpmvs[0].x as i64)) << s,
                ((cpmvs.cpmvs[2].y as i64) - (cpmvs.cpmvs[0].y as i64)) << s,
            )
        }
        _ => (-d_ver_x, d_hor_x),
    };

    let num_sb_x = (nb_w >> 2) as i64; // §8.5.5.9 eq. 668.
    let num_sb_y = (nb_h >> 2) as i64;

    // Centre offsets within the bottom-row sub-blocks.
    let y_bottom = 2 + ((num_sb_y - 1) << 2);
    let x_left = 2i64;
    let x_right = 2 + ((num_sb_x - 1) << 2);

    let mv_bl = compute_subblock_mv(
        mv_scale_hor,
        mv_scale_ver,
        d_hor_x,
        d_ver_x,
        d_hor_y,
        d_ver_y,
        x_left,
        y_bottom,
    );
    let mv_br = compute_subblock_mv(
        mv_scale_hor,
        mv_scale_ver,
        d_hor_x,
        d_ver_x,
        d_hor_y,
        d_ver_y,
        x_right,
        y_bottom,
    );
    (mv_bl, mv_br)
}

#[allow(clippy::too_many_arguments)]
fn compute_subblock_mv(
    mv_scale_hor: i64,
    mv_scale_ver: i64,
    d_hor_x: i64,
    d_ver_x: i64,
    d_hor_y: i64,
    d_ver_y: i64,
    x_pos: i64,
    y_pos: i64,
) -> MotionVector {
    // §8.5.5.9 eqs. 872 / 873.
    let raw_x = mv_scale_hor + d_hor_x * x_pos + d_hor_y * y_pos;
    let raw_y = mv_scale_ver + d_ver_x * x_pos + d_ver_y * y_pos;
    // §8.5.2.14 round with rightShift = 7, leftShift = 0 + eqs. 874 /
    // 875 clip to [−2^17, 2^17 − 1].
    let x = clip_mv17(round_component_i64(raw_x, 7));
    let y = clip_mv17(round_component_i64(raw_y, 7));
    MotionVector { x, y }
}

// =====================================================================
// §8.5.5.8 constructed affine CPMVP — per-corner neighbour-MV sampling.
// =====================================================================

/// One per-corner input to [`derive_constructed_affine_mvp_candidate`].
/// Carries the neighbour MV the §8.5.5.8 corner cascade picked at that
/// corner — or `None` when none of the cascade positions was effectively
/// available (no `availableN`, no matching POC).
///
/// The cascade is run by the caller (or by [`pick_constructed_corner`]):
/// * Corner 0 (top-left) walks `B2 → B3 → A2` (eqs. 841 – 842 use the
///   first matching `TL` position's `MvLX[xNbTL][yNbTL]`).
/// * Corner 1 (top-right) walks `B1 → B0` (eqs. 843 – 844 use TR).
/// * Corner 2 (bottom-left) walks `A1 → A0` (eqs. 845 – 846 use BL).
///
/// Per-position cross-list fallback (`Y = 1 − X`) is folded into the
/// caller's pick.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ConstructedAffineMvpCorner {
    /// `availableFlagLX[cpIdx]` per §8.5.5.8.
    pub available: bool,
    /// The corner CPMV the cascade picked, AMVR-rounded by the caller
    /// (or by [`derive_constructed_affine_mvp_candidate`]).
    pub mv: MotionVector,
}

/// §8.5.5.8 per-corner neighbour cascade — picks the first
/// effectively-available neighbour position whose prediction matches the
/// current CU's RefPicList[X][refIdxLX].
///
/// `positions` carries the cascade in order. The first element with
/// `available == true` and a list-X-or-list-Y POC match contributes its
/// MV; subsequent positions are short-circuited.
pub fn pick_constructed_corner(
    positions: &[NeighbourAffineQuery],
    ctx: &AffineMvpRefContext<'_>,
) -> ConstructedAffineMvpCorner {
    for nb in positions {
        if !nb.available {
            continue;
        }
        // §8.5.5.8 reads MvLX (or MvLY) at the neighbour sample
        // anchor: that's the neighbour's stored `cpmvs_l*[0]` slot —
        // the sample-aligned MV the §8.5.5.9 driver wrote (for affine)
        // or the regular per-sample MV (for translational). Either way,
        // it is the `MvLX[xNbN][yNbN]` cell the spec reads.
        if let Some(picked_list) = ctx.pick_list(nb) {
            let mv = match picked_list {
                RefList::L0 => nb.cpmvs_l0.cpmvs[0],
                RefList::L1 => nb.cpmvs_l1.cpmvs[0],
            };
            return ConstructedAffineMvpCorner {
                available: true,
                mv,
            };
        }
    }
    ConstructedAffineMvpCorner::default()
}

/// §8.5.5.8 — assemble the constructed affine CPMV-predictor candidate.
///
/// Inputs are the three (or two — `numCpMv == 2` ignores `bottom_left`)
/// per-corner picks from [`pick_constructed_corner`]. The function
/// AMVR-rounds each corner per the §8.5.5.8 "rounding process for motion
/// vectors with mvX set equal to cpMvLX[*]" inner bullet, then emits:
///
/// * `availableConsFlagLX = TRUE` and the full per-CP candidate when
///   all `numCpMv` corners are available (the spec's
///   `availableFlagLX[0] && availableFlagLX[1] && availableFlagLX[2] ==
///   1` for 6-param, or `availableFlagLX[0] && availableFlagLX[1]` for
///   4-param).
/// * `None` otherwise.
///
/// The result fits one slot in `cpMvpListLX` (eqs. 832 – 833).
pub fn derive_constructed_affine_mvp_candidate(
    top_left: ConstructedAffineMvpCorner,
    top_right: ConstructedAffineMvpCorner,
    bottom_left: ConstructedAffineMvpCorner,
    num_cp_mv: u32,
    amvr: AmvrShift,
) -> Option<AffineMvpCandidate> {
    debug_assert!((2..=3).contains(&num_cp_mv));
    if !top_left.available || !top_right.available {
        return None;
    }
    if num_cp_mv == 3 && !bottom_left.available {
        return None;
    }
    let mut cp_mvs = [MotionVector::ZERO; MAX_NUM_CP_MV];
    cp_mvs[0] = round_mv_amvr(top_left.mv, amvr);
    cp_mvs[1] = round_mv_amvr(top_right.mv, amvr);
    if num_cp_mv == 3 {
        cp_mvs[2] = round_mv_amvr(bottom_left.mv, amvr);
    }
    Some(AffineMvpCandidate { num_cp_mv, cp_mvs })
}

// =====================================================================
// §8.5.5.7 driver — assemble the per-list cpMvpListLX with exactly 2
// entries.
// =====================================================================

/// Inputs to [`build_affine_mvp_cand_list`] — already-derived per-list
/// pieces. Keeps the driver pure-data so the CTU walker can stage them
/// per-CU once per-CB CPMV storage is online.
///
/// * `inherited_a` / `inherited_b` — outputs of
///   [`derive_inherited_affine_mvp_candidate`] (already AMVR-rounded),
///   `None` when neither the A0 / A1 nor the B0 / B1 / B2 cascade
///   yielded an effectively-available affine neighbour.
/// * `constructed_full` — output of
///   [`derive_constructed_affine_mvp_candidate`] for the requested
///   `numCpMv` (None when fewer than `numCpMv` corners are available).
/// * `constructed_corners` — the three per-corner picks (top-left,
///   top-right, bottom-left), AMVR-rounded by the caller. Used for
///   the §8.5.5.7 step-7 standalone-corner fallback (`nbCpIdx = 2, 1,
///   0`).
/// * `temporal_col` — the §8.5.5.7 step-8 temporal MV
///   (`Option<MotionVector>`), already AMVR-rounded by the caller via
///   [`crate::amvp::derive_temporal_amvp_candidate`].
pub struct AffineMvpListInputs<'a> {
    /// `numCpMv` for the current CU — 2 (`MotionModelIdc == 1`) or 3
    /// (`MotionModelIdc == 2`).
    pub num_cp_mv: u32,
    /// Inherited A candidate (`None` when no A-side affine neighbour
    /// hit).
    pub inherited_a: Option<AffineMvpCandidate>,
    /// Inherited B candidate.
    pub inherited_b: Option<AffineMvpCandidate>,
    /// Full constructed candidate (`None` when fewer than `num_cp_mv`
    /// corners available).
    pub constructed_full: Option<AffineMvpCandidate>,
    /// Per-corner picks for step-7 single-corner replication. Slot 0 =
    /// top-left, slot 1 = top-right, slot 2 = bottom-left. The
    /// `available` flag is the §8.5.5.8 `availableFlagLX[cpIdx]`.
    pub constructed_corners: [ConstructedAffineMvpCorner; 3],
    /// §8.5.5.7 step 8 — temporal collocated MV (single MV, replicated
    /// across every CP).
    pub temporal_col: Option<MotionVector>,
    /// Phantom lifetime — kept for symmetry with future
    /// list-of-references inputs (e.g. multiple inherited candidates).
    pub _phantom: core::marker::PhantomData<&'a ()>,
}

impl Default for AffineMvpListInputs<'_> {
    fn default() -> Self {
        Self {
            num_cp_mv: 2,
            inherited_a: None,
            inherited_b: None,
            constructed_full: None,
            constructed_corners: [ConstructedAffineMvpCorner::default(); 3],
            temporal_col: None,
            _phantom: core::marker::PhantomData,
        }
    }
}

/// §8.5.5.7 steps 1 – 9 — assemble `cpMvpListLX` with exactly
/// [`MAX_AFFINE_MVP_CAND`] entries.
///
/// Insertion order:
/// 1. inherited A (when available),
/// 2. inherited B (when available),
/// 3. constructed full (when `numCpMvpCandLX < 2`),
/// 4. constructed standalone corners for `nbCpIdx = 2, 1, 0` (when
///    `numCpMvpCandLX < 2`),
/// 5. temporal MV (when `numCpMvpCandLX < 2`),
/// 6. zero-MV pad to `numCpMvpCandLX == 2`.
///
/// Eqs. 824 / 825 / 828 / 829 / 832 / 833 / 834 / 835 / 836 / 837 / 838 /
/// 839 each correspond to a step's increment of `numCpMvpCandLX` and the
/// matching `cpMvpListLX[numCpMvpCandLX][cpIdx]` assignment; the loop
/// here mirrors them in order. No pruning step is specified.
pub fn build_affine_mvp_cand_list(
    inputs: AffineMvpListInputs<'_>,
) -> [AffineMvpCandidate; MAX_AFFINE_MVP_CAND] {
    let num_cp_mv = inputs.num_cp_mv;
    debug_assert!((2..=3).contains(&num_cp_mv));
    let zero_cand = AffineMvpCandidate::from_single_mv(MotionVector::ZERO, num_cp_mv);
    let mut list = [zero_cand; MAX_AFFINE_MVP_CAND];
    let mut n = 0usize;

    let mut push = |cand: AffineMvpCandidate, n: &mut usize| {
        if *n < MAX_AFFINE_MVP_CAND {
            list[*n] = cand;
            *n += 1;
        }
    };

    // Step 4 — inherited A.
    if let Some(cand) = inputs.inherited_a {
        push(cand, &mut n);
    }
    // Step 5 — inherited B.
    if n < MAX_AFFINE_MVP_CAND {
        if let Some(cand) = inputs.inherited_b {
            push(cand, &mut n);
        }
    }
    // Step 6 — constructed full candidate.
    if n < MAX_AFFINE_MVP_CAND {
        if let Some(cand) = inputs.constructed_full {
            // Truncate to numCpMv slots in case the caller passed a
            // wider record.
            let mut adj = cand;
            adj.num_cp_mv = num_cp_mv;
            push(adj, &mut n);
        }
    }
    // Step 7 — for nbCpIdx = 2, 1, 0: single-corner replication when the
    // corner is available. Each emitted candidate replicates the
    // available corner MV across every CP slot.
    if n < MAX_AFFINE_MVP_CAND {
        for nb_cp_idx in (0..3).rev() {
            if n >= MAX_AFFINE_MVP_CAND {
                break;
            }
            let corner = inputs.constructed_corners[nb_cp_idx];
            if corner.available {
                push(
                    AffineMvpCandidate::from_single_mv(corner.mv, num_cp_mv),
                    &mut n,
                );
            }
        }
    }
    // Step 8 — temporal MV, replicated across every CP.
    if n < MAX_AFFINE_MVP_CAND {
        if let Some(mv) = inputs.temporal_col {
            push(AffineMvpCandidate::from_single_mv(mv, num_cp_mv), &mut n);
        }
    }
    // Step 9 — zero-MV pad.
    while n < MAX_AFFINE_MVP_CAND {
        push(zero_cand, &mut n);
    }
    list
}

// =====================================================================
// Eq. 840 — pick cpMvpListLX[mvp_lX_flag][cpIdx], and eqs. 664 – 667
// final MV fold.
// =====================================================================

/// §8.5.5.7 eq. 840 — `cpMvpLX[cpIdx] = cpMvpListLX[mvp_lX_flag][cpIdx]`.
/// `mvp_lx_flag` is `0` or `1` (1-bit FL bin per Table 88).
pub fn select_affine_mvp(
    list: &[AffineMvpCandidate; MAX_AFFINE_MVP_CAND],
    mvp_lx_flag: u32,
) -> AffineMvpCandidate {
    list[(mvp_lx_flag as usize) & (MAX_AFFINE_MVP_CAND - 1)]
}

/// §8.5.5.1 eqs. 664 – 667 — fold the per-CP AMVR-shifted `mvdCpLX` into
/// the chosen predictor:
///
/// ```text
/// uLX[cpIdx][i] = (mvpCpLX[cpIdx][i] + mvdCpLX[cpIdx][i]) & (2^18 − 1)
/// cpMvLX[cpIdx][i] = (uLX[cpIdx][i] >= 2^17) ? uLX[cpIdx][i] − 2^18
///                                            : uLX[cpIdx][i]
/// ```
///
/// `mvd_cp` carries the AMVR-shifted per-CP differences (eqs. 660 – 663
/// gave `mvdCpLX[cpIdx] = MvdCpLX[xCb][yCb][cpIdx][·]`, with cpIdx ≥ 1
/// pre-added to cpIdx 0 per eq. 662 / 663 to form the cumulative-delta
/// stream — that's the caller's responsibility).
///
/// Returns the per-CP final CPMVs (already in the modular 18-bit
/// wrapped + two's-complement-unwrapped form the §8.5.5.9 derivation
/// expects).
pub fn derive_final_affine_cpmvs(
    mvp: &AffineMvpCandidate,
    mvd_cp: &[MotionVector; MAX_NUM_CP_MV],
) -> AffineCpmvs {
    let mut cps = [MotionVector::ZERO; MAX_NUM_CP_MV];
    let num = mvp.num_cp_mv as usize;
    let mask: u32 = (1u32 << 18) - 1;
    let half: u32 = 1u32 << 17;
    for i in 0..num {
        let ux = (mvp.cp_mvs[i].x as u32).wrapping_add(mvd_cp[i].x as u32) & mask;
        let uy = (mvp.cp_mvs[i].y as u32).wrapping_add(mvd_cp[i].y as u32) & mask;
        let x = if ux >= half {
            (ux as i64) - (1i64 << 18)
        } else {
            ux as i64
        };
        let y = if uy >= half {
            (uy as i64) - (1i64 << 18)
        } else {
            uy as i64
        };
        cps[i] = MotionVector {
            x: x as i32,
            y: y as i32,
        };
    }
    let model = if num == 3 {
        MotionModel::Affine6Param
    } else {
        MotionModel::Affine4Param
    };
    AffineCpmvs { model, cpmvs: cps }
}

/// §8.5.5.5 eqs. 660 – 663 — the cumulative per-CP MVD fold.
///
/// The parser stores the on-wire per-CP differences `MvdCpLX[cpIdx]`
/// (post-AMVR, eqs. 165 – 176 already applied) directly. The §8.5.5.5
/// derivation then forms the *cumulative* `mvdCpLX[cpIdx]` the eq.
/// 664 – 667 final-MV fold consumes:
///
/// ```text
/// mvdCpLX[0]      = MvdCpLX[0]                          (eqs. 660 / 661)
/// mvdCpLX[cpIdx]  = MvdCpLX[cpIdx] + mvdCpLX[0]   cpIdx ≥ 1  (eqs. 662 / 663)
/// ```
///
/// i.e. every higher control point's parsed difference is *relative to*
/// CP0's, so the stored stream is differential. This helper produces the
/// absolute cumulative array, which is then ready for
/// [`derive_final_affine_cpmvs`].
///
/// `mvd_cp_stored[i]` is `MvdCpLX[xCb][yCb][i]` as the parser captured it
/// (`mvd_cp_l0` / `mvd_cp_l1` on
/// [`crate::non_merge_inter_pre_residual_enc::NonMergeInterPreResidualAffineDecision`]).
/// Slots `i >= num_cp_mv` are ignored.
pub fn cumulate_affine_mvd_cp(
    mvd_cp_stored: &[MotionVector; MAX_NUM_CP_MV],
    num_cp_mv: u32,
) -> [MotionVector; MAX_NUM_CP_MV] {
    debug_assert!((2..=3).contains(&num_cp_mv));
    let num = num_cp_mv as usize;
    let mut out = [MotionVector::ZERO; MAX_NUM_CP_MV];
    // eqs. 660 / 661.
    out[0] = mvd_cp_stored[0];
    // eqs. 662 / 663 — add CP0's MVD to every higher control point.
    for i in 1..num {
        out[i] = MotionVector {
            x: mvd_cp_stored[i].x.wrapping_add(out[0].x),
            y: mvd_cp_stored[i].y.wrapping_add(out[0].y),
        };
    }
    out
}

// =====================================================================
// §8.5.2.14 helpers — copied here so the module stands on its own.
// =====================================================================

/// §8.5.2.14 rounding (signed-magnitude) with
/// `rightShift = leftShift = AmvrShift`.
fn round_mv_amvr(mv: MotionVector, amvr: AmvrShift) -> MotionVector {
    let s = amvr.value();
    if s == 0 {
        return mv;
    }
    MotionVector {
        x: round_component_i32(mv.x, s),
        y: round_component_i32(mv.y, s),
    }
}

fn round_component_i32(v: i32, shift: u32) -> i32 {
    if shift == 0 {
        return v;
    }
    let offset = (1i32 << (shift - 1)) - 1;
    let sign = v.signum();
    sign * (((v.abs() + offset) >> shift) << shift)
}

fn round_component_i64(v: i64, right_shift: i32) -> i64 {
    if right_shift == 0 {
        return v;
    }
    let offset = (1i64 << (right_shift - 1)) - 1;
    let sign = if v < 0 { -1 } else { 1 };
    sign * ((v.abs() + offset) >> right_shift)
}

fn clip_mv17(v: i64) -> i32 {
    v.clamp(-(1i64 << 17), (1i64 << 17) - 1) as i32
}

// =====================================================================
// Tests
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // POC resolvers returning the same POC for every refIdx — a simple
    // single-reference scenario that satisfies the DiffPicOrderCnt == 0
    // gate without needing a full RPL.
    fn pocs_l0(_: i32) -> Option<i32> {
        Some(8)
    }
    fn pocs_l1(_: i32) -> Option<i32> {
        Some(8)
    }

    fn ctx_l0() -> AffineMvpRefContext<'static> {
        AffineMvpRefContext {
            list: RefList::L0,
            current_ref_poc: 8,
            poc_of_l0_ref: &pocs_l0,
            poc_of_l1_ref: &pocs_l1,
        }
    }

    fn mismatch_ctx_l0() -> AffineMvpRefContext<'static> {
        // current_ref_poc 9 vs neighbour's resolver returning 8 → no
        // DiffPicOrderCnt == 0 hit on either list ⇒ scan picks nothing.
        AffineMvpRefContext {
            list: RefList::L0,
            current_ref_poc: 9,
            poc_of_l0_ref: &pocs_l0,
            poc_of_l1_ref: &pocs_l1,
        }
    }

    fn make_affine_neighbour(model: MotionModel, mv_l0: MotionVector) -> NeighbourAffineQuery {
        let cpmvs = match model {
            MotionModel::Affine6Param => AffineCpmvs::new_6param(mv_l0, mv_l0, mv_l0),
            _ => AffineCpmvs::new_4param(mv_l0, mv_l0),
        };
        NeighbourAffineQuery {
            available: true,
            pred_flag_l0: true,
            pred_flag_l1: false,
            ref_idx_l0: 0,
            ref_idx_l1: -1,
            xnb: 0,
            ynb: 0,
            nb_w: 16,
            nb_h: 16,
            motion_model: model,
            cpmvs_l0: cpmvs,
            cpmvs_l1: AffineCpmvs::new_4param(MotionVector::ZERO, MotionVector::ZERO),
        }
    }

    #[test]
    fn inherited_translational_neighbour_rejected() {
        // §8.5.5.7 step-4 inner gate "MotionModelIdc[xNbAk][yNbAk] > 0"
        // rejects translational neighbours.
        let nb = make_affine_neighbour(MotionModel::Translational, MotionVector { x: 32, y: -16 });
        let cand = derive_inherited_affine_mvp_candidate(
            16,
            16,
            16,
            16,
            &nb,
            &ctx_l0(),
            AmvrShift(0),
            2,
            false,
        );
        assert!(cand.is_none());
    }

    #[test]
    fn inherited_unavailable_neighbour_rejected() {
        let mut nb = make_affine_neighbour(MotionModel::Affine4Param, MotionVector { x: 16, y: 0 });
        nb.available = false;
        let cand = derive_inherited_affine_mvp_candidate(
            16,
            16,
            16,
            16,
            &nb,
            &ctx_l0(),
            AmvrShift(0),
            2,
            false,
        );
        assert!(cand.is_none());
    }

    #[test]
    fn inherited_poc_mismatch_rejected() {
        let nb = make_affine_neighbour(MotionModel::Affine4Param, MotionVector { x: 16, y: 0 });
        let cand = derive_inherited_affine_mvp_candidate(
            16,
            16,
            16,
            16,
            &nb,
            &mismatch_ctx_l0(),
            AmvrShift(0),
            2,
            false,
        );
        assert!(cand.is_none());
    }

    #[test]
    fn inherited_affine_neighbour_emits_2cp_candidate() {
        // A 4-param affine neighbour with all CPMVs equal to (16, 0) at
        // origin (0, 0) and the current CU at (16, 16) with a 16x16 CB
        // should inherit a translation-only (CP0 == CP1 == (16, 0))
        // candidate.
        let nb = make_affine_neighbour(MotionModel::Affine4Param, MotionVector { x: 16, y: 0 });
        let cand = derive_inherited_affine_mvp_candidate(
            16,
            16,
            16,
            16,
            &nb,
            &ctx_l0(),
            AmvrShift(0),
            2,
            false,
        )
        .expect("inherited candidate");
        assert_eq!(cand.num_cp_mv, 2);
        // Constant CPMVs ⇒ translational ⇒ same MV at every projected
        // corner.
        assert_eq!(cand.cp_mvs[0], MotionVector { x: 16, y: 0 });
        assert_eq!(cand.cp_mvs[1], MotionVector { x: 16, y: 0 });
    }

    #[test]
    fn inherited_affine_neighbour_emits_3cp_candidate_when_requested() {
        let nb = make_affine_neighbour(MotionModel::Affine6Param, MotionVector { x: 32, y: 32 });
        let cand = derive_inherited_affine_mvp_candidate(
            16,
            16,
            16,
            16,
            &nb,
            &ctx_l0(),
            AmvrShift(0),
            3,
            false,
        )
        .expect("3-cp inherited");
        assert_eq!(cand.num_cp_mv, 3);
        assert_eq!(cand.cp_mvs[0], MotionVector { x: 32, y: 32 });
        assert_eq!(cand.cp_mvs[1], MotionVector { x: 32, y: 32 });
        assert_eq!(cand.cp_mvs[2], MotionVector { x: 32, y: 32 });
    }

    #[test]
    fn inherited_cross_list_fallback() {
        // Neighbour predicts only on L1 with matching POC; current CU
        // is asking for list L0 candidates ⇒ §8.5.5.7's "Otherwise when
        // PredFlagLY == 1 ∧ DiffPicOrderCnt(Y) == 0" branch should
        // pick the L1 CPMVs.
        let mut nb = make_affine_neighbour(MotionModel::Affine4Param, MotionVector { x: 64, y: 0 });
        nb.pred_flag_l0 = false;
        nb.pred_flag_l1 = true;
        nb.ref_idx_l0 = -1;
        nb.ref_idx_l1 = 0;
        nb.cpmvs_l1 =
            AffineCpmvs::new_4param(MotionVector { x: 64, y: 0 }, MotionVector { x: 64, y: 0 });
        let cand = derive_inherited_affine_mvp_candidate(
            16,
            16,
            16,
            16,
            &nb,
            &ctx_l0(),
            AmvrShift(0),
            2,
            false,
        )
        .expect("cross-list inherited");
        assert_eq!(cand.cp_mvs[0], MotionVector { x: 64, y: 0 });
    }

    #[test]
    fn inherited_amvr_rounds_components() {
        // AmvrShift(4) = 1-luma resolution → low 4 bits zeroed, signed-
        // magnitude rounded (offset = 7).
        let nb = make_affine_neighbour(MotionModel::Affine4Param, MotionVector { x: 23, y: -23 });
        let cand = derive_inherited_affine_mvp_candidate(
            16,
            16,
            16,
            16,
            &nb,
            &ctx_l0(),
            AmvrShift(4),
            2,
            false,
        )
        .expect("amvr-rounded");
        // |23| + 7 = 30; 30 >> 4 = 1; 1 << 4 = 16.
        assert_eq!(cand.cp_mvs[0].x, 16);
        assert_eq!(cand.cp_mvs[0].y, -16);
    }

    #[test]
    fn constructed_corner_pick_translational_neighbour() {
        // §8.5.5.8's per-corner read accepts a translational neighbour
        // (unlike §8.5.5.7 step-4 inherited scan).
        let nb = make_affine_neighbour(MotionModel::Translational, MotionVector { x: 48, y: -16 });
        let corner = pick_constructed_corner(&[nb], &ctx_l0());
        assert!(corner.available);
        assert_eq!(corner.mv, MotionVector { x: 48, y: -16 });
    }

    #[test]
    fn constructed_corner_cascade_picks_first_available() {
        let mut nb_unavail =
            make_affine_neighbour(MotionModel::Affine4Param, MotionVector { x: 0, y: 0 });
        nb_unavail.available = false;
        let nb_hit =
            make_affine_neighbour(MotionModel::Translational, MotionVector { x: 12, y: 6 });
        let corner = pick_constructed_corner(&[nb_unavail, nb_hit], &ctx_l0());
        assert!(corner.available);
        assert_eq!(corner.mv, MotionVector { x: 12, y: 6 });
    }

    #[test]
    fn constructed_corner_no_hit_yields_default() {
        // Cascade with no available neighbour ⇒ availableFlag == false.
        let mut nb1 =
            make_affine_neighbour(MotionModel::Translational, MotionVector { x: 1, y: 1 });
        nb1.available = false;
        let mut nb2 = nb1;
        nb2.pred_flag_l0 = false;
        nb2.pred_flag_l1 = false;
        nb2.available = true;
        let corner = pick_constructed_corner(&[nb1, nb2], &ctx_l0());
        assert!(!corner.available);
    }

    #[test]
    fn constructed_mvp_requires_all_corners_for_6param() {
        let tl = ConstructedAffineMvpCorner {
            available: true,
            mv: MotionVector { x: 1, y: 2 },
        };
        let tr = ConstructedAffineMvpCorner {
            available: true,
            mv: MotionVector { x: 3, y: 4 },
        };
        // bottom_left unavailable ⇒ no 6-param candidate.
        let bl = ConstructedAffineMvpCorner::default();
        assert!(derive_constructed_affine_mvp_candidate(tl, tr, bl, 3, AmvrShift(0)).is_none());
        // ... but the 4-param variant only needs TL + TR.
        let cand = derive_constructed_affine_mvp_candidate(tl, tr, bl, 2, AmvrShift(0))
            .expect("4-param constructed");
        assert_eq!(cand.cp_mvs[0], MotionVector { x: 1, y: 2 });
        assert_eq!(cand.cp_mvs[1], MotionVector { x: 3, y: 4 });
        assert_eq!(cand.cp_mvs[2], MotionVector::ZERO);
    }

    #[test]
    fn constructed_mvp_full_6param() {
        let tl = ConstructedAffineMvpCorner {
            available: true,
            mv: MotionVector { x: 1, y: 2 },
        };
        let tr = ConstructedAffineMvpCorner {
            available: true,
            mv: MotionVector { x: 3, y: 4 },
        };
        let bl = ConstructedAffineMvpCorner {
            available: true,
            mv: MotionVector { x: 5, y: 6 },
        };
        let cand = derive_constructed_affine_mvp_candidate(tl, tr, bl, 3, AmvrShift(0))
            .expect("6-param constructed");
        assert_eq!(cand.num_cp_mv, 3);
        assert_eq!(cand.cp_mvs[0], MotionVector { x: 1, y: 2 });
        assert_eq!(cand.cp_mvs[1], MotionVector { x: 3, y: 4 });
        assert_eq!(cand.cp_mvs[2], MotionVector { x: 5, y: 6 });
    }

    #[test]
    fn list_assembly_inherited_only() {
        // Inherited A + inherited B fill the list; constructed /
        // temporal / zero-pad all suppressed.
        let inh_a = AffineMvpCandidate {
            num_cp_mv: 2,
            cp_mvs: [
                MotionVector { x: 16, y: 0 },
                MotionVector { x: 16, y: 0 },
                MotionVector::ZERO,
            ],
        };
        let inh_b = AffineMvpCandidate {
            num_cp_mv: 2,
            cp_mvs: [
                MotionVector { x: -16, y: 0 },
                MotionVector { x: -16, y: 0 },
                MotionVector::ZERO,
            ],
        };
        let list = build_affine_mvp_cand_list(AffineMvpListInputs {
            num_cp_mv: 2,
            inherited_a: Some(inh_a),
            inherited_b: Some(inh_b),
            ..Default::default()
        });
        assert_eq!(list[0], inh_a);
        assert_eq!(list[1], inh_b);
    }

    #[test]
    fn list_assembly_constructed_full_after_one_inherited() {
        let inh_a = AffineMvpCandidate {
            num_cp_mv: 2,
            cp_mvs: [
                MotionVector { x: 16, y: 0 },
                MotionVector { x: 16, y: 0 },
                MotionVector::ZERO,
            ],
        };
        let constructed = AffineMvpCandidate {
            num_cp_mv: 2,
            cp_mvs: [
                MotionVector { x: 8, y: 8 },
                MotionVector { x: 8, y: 8 },
                MotionVector::ZERO,
            ],
        };
        let list = build_affine_mvp_cand_list(AffineMvpListInputs {
            num_cp_mv: 2,
            inherited_a: Some(inh_a),
            constructed_full: Some(constructed),
            ..Default::default()
        });
        assert_eq!(list[0], inh_a);
        assert_eq!(list[1], constructed);
    }

    #[test]
    fn list_assembly_step7_single_corner_order_2_1_0() {
        // §8.5.5.7 step 7 walks nbCpIdx = 2, 1, 0. All three corners
        // available + no inherited / constructed-full → list takes
        // corner[2] then corner[1].
        let c0 = ConstructedAffineMvpCorner {
            available: true,
            mv: MotionVector { x: 1, y: 1 },
        };
        let c1 = ConstructedAffineMvpCorner {
            available: true,
            mv: MotionVector { x: 2, y: 2 },
        };
        let c2 = ConstructedAffineMvpCorner {
            available: true,
            mv: MotionVector { x: 3, y: 3 },
        };
        let list = build_affine_mvp_cand_list(AffineMvpListInputs {
            num_cp_mv: 2,
            constructed_corners: [c0, c1, c2],
            ..Default::default()
        });
        // First entry: corner[2] replicated.
        assert_eq!(list[0].cp_mvs[0], MotionVector { x: 3, y: 3 });
        assert_eq!(list[0].cp_mvs[1], MotionVector { x: 3, y: 3 });
        // Second entry: corner[1] replicated.
        assert_eq!(list[1].cp_mvs[0], MotionVector { x: 2, y: 2 });
        assert_eq!(list[1].cp_mvs[1], MotionVector { x: 2, y: 2 });
    }

    #[test]
    fn list_assembly_step8_temporal_when_short() {
        let list = build_affine_mvp_cand_list(AffineMvpListInputs {
            num_cp_mv: 2,
            temporal_col: Some(MotionVector { x: 64, y: -64 }),
            ..Default::default()
        });
        // Temporal MV becomes the first entry, replicated across both
        // CPs.
        assert_eq!(list[0].cp_mvs[0], MotionVector { x: 64, y: -64 });
        assert_eq!(list[0].cp_mvs[1], MotionVector { x: 64, y: -64 });
        // Second entry zero-padded.
        assert_eq!(list[1].cp_mvs[0], MotionVector::ZERO);
        assert_eq!(list[1].cp_mvs[1], MotionVector::ZERO);
    }

    #[test]
    fn list_assembly_zero_pad_when_nothing_available() {
        let list = build_affine_mvp_cand_list(AffineMvpListInputs {
            num_cp_mv: 3,
            ..Default::default()
        });
        for cand in &list {
            assert_eq!(cand.num_cp_mv, 3);
            assert_eq!(cand.cp_mvs[0], MotionVector::ZERO);
            assert_eq!(cand.cp_mvs[1], MotionVector::ZERO);
            assert_eq!(cand.cp_mvs[2], MotionVector::ZERO);
        }
    }

    #[test]
    fn list_assembly_step_order_dominates() {
        // Even when constructed_full + corners + temporal are all
        // available, the inherited A + inherited B come first and the
        // others never reach the list.
        let inh_a = AffineMvpCandidate {
            num_cp_mv: 2,
            cp_mvs: [
                MotionVector { x: 1, y: 0 },
                MotionVector { x: 1, y: 0 },
                MotionVector::ZERO,
            ],
        };
        let inh_b = AffineMvpCandidate {
            num_cp_mv: 2,
            cp_mvs: [
                MotionVector { x: 2, y: 0 },
                MotionVector { x: 2, y: 0 },
                MotionVector::ZERO,
            ],
        };
        let constructed = AffineMvpCandidate {
            num_cp_mv: 2,
            cp_mvs: [
                MotionVector { x: 3, y: 0 },
                MotionVector { x: 3, y: 0 },
                MotionVector::ZERO,
            ],
        };
        let c0 = ConstructedAffineMvpCorner {
            available: true,
            mv: MotionVector { x: 4, y: 0 },
        };
        let list = build_affine_mvp_cand_list(AffineMvpListInputs {
            num_cp_mv: 2,
            inherited_a: Some(inh_a),
            inherited_b: Some(inh_b),
            constructed_full: Some(constructed),
            constructed_corners: [c0, c0, c0],
            temporal_col: Some(MotionVector { x: 5, y: 0 }),
            ..Default::default()
        });
        assert_eq!(list[0], inh_a);
        assert_eq!(list[1], inh_b);
    }

    #[test]
    fn select_affine_mvp_picks_index() {
        let a = AffineMvpCandidate {
            num_cp_mv: 2,
            cp_mvs: [
                MotionVector { x: 1, y: 1 },
                MotionVector { x: 1, y: 1 },
                MotionVector::ZERO,
            ],
        };
        let b = AffineMvpCandidate {
            num_cp_mv: 2,
            cp_mvs: [
                MotionVector { x: 2, y: 2 },
                MotionVector { x: 2, y: 2 },
                MotionVector::ZERO,
            ],
        };
        let list = [a, b];
        assert_eq!(select_affine_mvp(&list, 0), a);
        assert_eq!(select_affine_mvp(&list, 1), b);
    }

    #[test]
    fn derive_final_affine_cpmvs_adds_mvd_per_cp() {
        let mvp = AffineMvpCandidate {
            num_cp_mv: 2,
            cp_mvs: [
                MotionVector { x: 10, y: -5 },
                MotionVector { x: 12, y: -3 },
                MotionVector::ZERO,
            ],
        };
        let mvd = [
            MotionVector { x: 2, y: -1 },
            MotionVector { x: -4, y: 8 },
            MotionVector::ZERO,
        ];
        let out = derive_final_affine_cpmvs(&mvp, &mvd);
        assert_eq!(out.cpmvs[0], MotionVector { x: 12, y: -6 });
        assert_eq!(out.cpmvs[1], MotionVector { x: 8, y: 5 });
        assert!(matches!(out.model, MotionModel::Affine4Param));
    }

    #[test]
    fn cumulate_affine_mvd_cp_adds_cp0_to_higher_cps() {
        // §8.5.5.5 eqs. 660 – 663 — CP0 is identity; CP1/CP2 are
        // differential relative to CP0.
        let stored = [
            MotionVector { x: 5, y: -2 },
            MotionVector { x: 3, y: 7 },
            MotionVector { x: -1, y: 4 },
        ];
        let out6 = cumulate_affine_mvd_cp(&stored, 3);
        assert_eq!(out6[0], MotionVector { x: 5, y: -2 });
        assert_eq!(out6[1], MotionVector { x: 8, y: 5 });
        assert_eq!(out6[2], MotionVector { x: 4, y: 2 });

        // 4-param: only CP0 / CP1 are folded; CP2 stays zero.
        let out4 = cumulate_affine_mvd_cp(&stored, 2);
        assert_eq!(out4[0], MotionVector { x: 5, y: -2 });
        assert_eq!(out4[1], MotionVector { x: 8, y: 5 });
        assert_eq!(out4[2], MotionVector::ZERO);
    }

    #[test]
    fn cumulate_then_final_fold_round_trips_translational_cpmvs() {
        // When every stored MvdCp is zero the final CPMVs equal the
        // predictor exactly (the affine model degenerates to the
        // predictor's CPMVs).
        let mvp = AffineMvpCandidate {
            num_cp_mv: 2,
            cp_mvs: [
                MotionVector { x: 16, y: -16 },
                MotionVector { x: 32, y: 0 },
                MotionVector::ZERO,
            ],
        };
        let stored = [MotionVector::ZERO; MAX_NUM_CP_MV];
        let cumulative = cumulate_affine_mvd_cp(&stored, 2);
        let out = derive_final_affine_cpmvs(&mvp, &cumulative);
        assert_eq!(out.cpmvs[0], mvp.cp_mvs[0]);
        assert_eq!(out.cpmvs[1], mvp.cp_mvs[1]);
    }

    #[test]
    fn derive_final_affine_cpmvs_two_complement_wrap() {
        // mvp + mvd just above 2^17 should wrap to a large negative
        // per eq. 665 / 667.
        let mvp = AffineMvpCandidate {
            num_cp_mv: 2,
            cp_mvs: [
                MotionVector {
                    x: (1 << 17) - 1,
                    y: 0,
                },
                MotionVector::ZERO,
                MotionVector::ZERO,
            ],
        };
        let mvd = [
            MotionVector { x: 1, y: 0 },
            MotionVector::ZERO,
            MotionVector::ZERO,
        ];
        let out = derive_final_affine_cpmvs(&mvp, &mvd);
        // (131071 + 1) & (2^18 − 1) = 131072 = 2^17 → ≥ 2^17 → −131072.
        assert_eq!(out.cpmvs[0].x, -(1 << 17));
    }

    #[test]
    fn derive_final_affine_cpmvs_6param() {
        let mvp = AffineMvpCandidate {
            num_cp_mv: 3,
            cp_mvs: [
                MotionVector { x: 1, y: 1 },
                MotionVector { x: 2, y: 2 },
                MotionVector { x: 3, y: 3 },
            ],
        };
        let mvd = [
            MotionVector { x: 0, y: 0 },
            MotionVector { x: 0, y: 0 },
            MotionVector { x: 0, y: 0 },
        ];
        let out = derive_final_affine_cpmvs(&mvp, &mvd);
        assert!(matches!(out.model, MotionModel::Affine6Param));
        assert_eq!(out.cpmvs[0], MotionVector { x: 1, y: 1 });
        assert_eq!(out.cpmvs[1], MotionVector { x: 2, y: 2 });
        assert_eq!(out.cpmvs[2], MotionVector { x: 3, y: 3 });
    }

    #[test]
    fn ctu_boundary_path_uses_bottom_row_subblocks() {
        // CTU boundary case: the spec reads two sub-block MVs from the
        // neighbour's bottom row instead of the stored CPMVs. With a
        // constant-MV neighbour both bottom-row sub-blocks are the same
        // MV, so the inherited candidate should still be translational.
        let nb = make_affine_neighbour(MotionModel::Affine4Param, MotionVector { x: 32, y: 16 });
        let cand = derive_inherited_affine_mvp_candidate(
            16,
            16,
            16,
            16,
            &nb,
            &ctx_l0(),
            AmvrShift(0),
            2,
            true,
        )
        .expect("ctu-boundary inherited");
        assert_eq!(cand.cp_mvs[0], MotionVector { x: 32, y: 16 });
        assert_eq!(cand.cp_mvs[1], MotionVector { x: 32, y: 16 });
    }

    #[test]
    fn round_trip_full_path_inherited_plus_mvd() {
        // End-to-end: derive an inherited candidate, select it,
        // fold a per-CP MVD in, verify the final CPMV.
        let nb = make_affine_neighbour(MotionModel::Affine4Param, MotionVector { x: 16, y: 0 });
        let cand = derive_inherited_affine_mvp_candidate(
            16,
            16,
            16,
            16,
            &nb,
            &ctx_l0(),
            AmvrShift(0),
            2,
            false,
        )
        .unwrap();
        let list = build_affine_mvp_cand_list(AffineMvpListInputs {
            num_cp_mv: 2,
            inherited_a: Some(cand),
            ..Default::default()
        });
        let picked = select_affine_mvp(&list, 0);
        let mvd = [
            MotionVector { x: 4, y: -2 },
            MotionVector { x: -2, y: 6 },
            MotionVector::ZERO,
        ];
        let out = derive_final_affine_cpmvs(&picked, &mvd);
        assert_eq!(out.cpmvs[0], MotionVector { x: 20, y: -2 });
        assert_eq!(out.cpmvs[1], MotionVector { x: 14, y: 6 });
    }
}
