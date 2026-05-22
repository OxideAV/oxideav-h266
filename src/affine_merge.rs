//! Affine merge candidate list derivation — round-91 §8.5.5.5 (inherited
//! from neighbour) + §8.5.5.6 (constructed from per-corner triples).
//!
//! Round 65 landed the §8.5.5.9 sub-block MV derivation that turns a CU's
//! `AffineCpmvs` into a per-4×4 grid of luma MVs; round 78 layered the
//! §8.5.6.4 PROF refinement on top of that. Both of those modules
//! consume a CPMV record but had no way to produce one from neighbour
//! state — affine-merge CUs need a candidate list before sub-block MC
//! can run. This module fills that gap.
//!
//! ## What this module ships
//!
//! 1. [`AffineCpRecord`] — the per-corner record consumed by the
//!    constructed-candidate derivation: `(predFlagL0/L1, refIdxL0/L1,
//!    mvL0/L1, bcwIdx, available)`. Mirrors the spec's
//!    `cpMvLXCorner[k]` / `predFlagLXCorner[k]` / `refIdxLXCorner[k]` /
//!    `bcwIdxCorner[k]` fan-out across the four corner indices
//!    `k ∈ {0, 1, 2, 3}` (top-left, top-right, bottom-left, temporal
//!    bottom-right).
//! 2. [`AffineMergeCandidate`] — the per-candidate record emitted by
//!    both [`derive_inherited_affine_cpmvs`] and
//!    [`derive_constructed_affine_merge_candidates`]: per-list CPMVs
//!    (eq. 748 – 755 inherited path, eq. 783 – 818 constructed path),
//!    per-list (predFlag, refIdx), `MotionModelIdc`, `bcwIdx`.
//! 3. [`derive_inherited_affine_cpmvs`] — §8.5.5.5 verbatim:
//!    - `isCTUboundary` bullet (eqs. 736 – 739 use the sub-block MV
//!      array `MvLX[xNb + nNbW − 1][yNb + nNbH − 1]`, eq. 746 / 747
//!      apply the 4-parameter `dHorY = −dVerX, dVerY = dHorX`
//!      similarity even when `MotionModelIdc == 2`).
//!    - Non-boundary bullet (eqs. 740 – 745 use the neighbour CU's
//!      stored CPMVs `CpMvLX[xNb][yNb][cpIdx]`; the dHorY / dVerY
//!      branch only consults `cpMvLX[2]` when
//!      `MotionModelIdc == 2`).
//!    - The CPMV computation (eqs. 748 – 753), §8.5.2.14 rounding with
//!      `rightShift = 7, leftShift = 0`, and the eqs. 754 / 755
//!      `Clip3(−2^17, 2^17 − 1, ·)` clip.
//! 4. [`derive_constructed_affine_merge_candidates`] — §8.5.5.6
//!    verbatim:
//!    - Corner selection (B2 / B3 / A2 cascade for top-left,
//!      B1 / B0 for top-right, A1 / A0 for bottom-left, temporal
//!      bottom-right via §8.5.5.6 step "fourth corner").
//!    - Six combination triples (3 corners ⇒ Const1..Const4 for the
//!      6-parameter set, 2 corners ⇒ Const5..Const6 for the
//!      4-parameter set).
//!    - For each combination: per-list predFlag derivation gated by
//!      "all three predFlags ⇒ 1 ∧ all three refIdx match", the
//!      cpMvLXConstK assignments (eq. 783 – 818), and the
//!      `motionModelIdcConstK = 2` (6-param triples) or `= 1`
//!      (4-param pairs).
//!    - The bcwIdx tie-breaker: inherited from corner 0 (Const1, 2,
//!      3, 5, 6) or corner 1 (Const4) when both L0 + L1 sides are
//!      derived, else 0.
//!
//! ## Out of scope (deferred to later rounds)
//!
//! * The §8.5.5.2 driver that fuses inherited A / inherited B / Const1..6
//!   into the subblockMergeCandList and indexes it via
//!   `merge_subblock_idx`. The driver needs the per-CB CPMV storage
//!   wired into the CTU walker — a separate cross-module change.
//! * SbTMVP (§8.5.5.3 / §8.5.5.4) — the SbCol candidate. Lands in a
//!   future round once `MotionField` carries the per-CB collocated
//!   8×8 motion buffer.
//! * The §8.5.5.7 affine AMVP candidate list (explicit `mvp_lx_flag`
//!   + per-CPMV mvd_coding). Same neighbour-availability rules but
//!   different output shape; deferred until the AMVP path comes
//!   online.
//!
//! ## Spec reference
//!
//! ITU-T H.266 | ISO/IEC 23090-3 (V4, 01/2026):
//! * §8.5.5.2 — "Derivation process for motion vectors and reference
//!   indices in subblock merge mode" (the driver that consumes this
//!   module's outputs).
//! * §8.5.5.5 — "Derivation process for luma affine control point
//!   motion vectors from a neighbouring block" (eqs. 734 – 755).
//! * §8.5.5.6 — "Derivation process for constructed affine control
//!   point motion vector merging candidates" (eqs. 756 – 818).
//! * §8.5.2.14 — "Rounding process for motion vectors" (eqs. 608 –
//!   610).
//!
//! No third-party VVC decoder source was consulted; the
//! implementation is spec-only.

use oxideav_core::Result;

use crate::affine::{AffineCpmvs, MotionModel};
use crate::inter::MotionVector;

/// Per-corner record consumed by the §8.5.5.6 constructed-candidate
/// derivation. Mirrors the spec's `cpMvLXCorner[k]` /
/// `predFlagLXCorner[k]` / `refIdxLXCorner[k]` / `bcwIdxCorner[k]`
/// arrays at corner index `k ∈ {0, 1, 2, 3}`.
///
/// Corner indexing:
/// * `k == 0` — top-left (filled by B2 / B3 / A2 cascade per the spec).
/// * `k == 1` — top-right (B1 / B0 cascade).
/// * `k == 2` — bottom-left (A1 / A0 cascade).
/// * `k == 3` — temporal bottom-right (collocated picture).
///
/// `available == false` corresponds to the spec's
/// `availableFlagCorner[k] == FALSE` short-circuit; in that case all
/// other fields are unread.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AffineCpRecord {
    /// `availableFlagCorner[k]` per §8.5.5.6.
    pub available: bool,
    /// `predFlagL0Corner[k]` — eq. 757 / 762 / 767 / 775.
    pub pred_flag_l0: bool,
    /// `predFlagL1Corner[k]` — eq. 757 / 762 / 767 / 779 (B-slice only
    /// for k == 3).
    pub pred_flag_l1: bool,
    /// `refIdxL0Corner[k]` — eq. 756 / 761 / 766 (corner 3 fixed to 0
    /// per §8.5.5.6 fourth-corner bullet).
    pub ref_idx_l0: i32,
    /// `refIdxL1Corner[k]` — same as L0.
    pub ref_idx_l1: i32,
    /// `cpMvL0Corner[k]` — eq. 758 / 763 / 768 / 776.
    pub mv_l0: MotionVector,
    /// `cpMvL1Corner[k]` — eq. 758 / 763 / 768 / 780 (B-slice only).
    pub mv_l1: MotionVector,
    /// `bcwIdxCorner[k]` — eq. 759 / 764 (only corners 0 + 1 carry it;
    /// corners 2 + 3 surface 0).
    pub bcw_idx: u8,
}

impl AffineCpRecord {
    /// Spec's "FALSE / 0 / 0 / −1 / 0" all-clear corner — every later
    /// invocation reads `available == false` and short-circuits.
    pub const UNAVAILABLE: AffineCpRecord = AffineCpRecord {
        available: false,
        pred_flag_l0: false,
        pred_flag_l1: false,
        ref_idx_l0: -1,
        ref_idx_l1: -1,
        mv_l0: MotionVector::ZERO,
        mv_l1: MotionVector::ZERO,
        bcw_idx: 0,
    };
}

/// One affine-merge candidate — what either §8.5.5.5 (inherited path) or
/// §8.5.5.6 (constructed path) emits.
///
/// Carries both list halves; uni-pred candidates set
/// `pred_flag_l1 == false` and leave the L1 CPMV record zero (matching
/// the spec's `mvLXA` / `cpMvLXConstK` defaults for the inactive list).
///
/// `AffineCpmvs` is `Clone + Copy + Debug` but not `PartialEq` — tests
/// compare per-field rather than struct-as-whole.
#[derive(Clone, Copy, Debug)]
pub struct AffineMergeCandidate {
    /// `predFlagL0` for this candidate.
    pub pred_flag_l0: bool,
    /// `predFlagL1` for this candidate.
    pub pred_flag_l1: bool,
    /// `refIdxL0` — `-1` when `pred_flag_l0 == false`.
    pub ref_idx_l0: i32,
    /// `refIdxL1` — `-1` when `pred_flag_l1 == false`.
    pub ref_idx_l1: i32,
    /// L0 CPMV record (model + up-to-three control points in 1/16-pel).
    pub cpmvs_l0: AffineCpmvs,
    /// L1 CPMV record (B-slice only — `Translational` + zero CPs when
    /// `pred_flag_l1 == false`).
    pub cpmvs_l1: AffineCpmvs,
    /// `motionModelIdc` for this candidate. Always
    /// [`MotionModel::Affine4Param`] or [`MotionModel::Affine6Param`]
    /// (the spec emits 4-param for Const5/6 and inherited-from-4-param
    /// neighbours; 6-param for Const1..4 and inherited-from-6-param
    /// neighbours).
    pub motion_model: MotionModel,
    /// `bcwIdx` for this candidate.
    pub bcw_idx: u8,
}

impl Default for AffineMergeCandidate {
    fn default() -> Self {
        Self {
            pred_flag_l0: false,
            pred_flag_l1: false,
            ref_idx_l0: -1,
            ref_idx_l1: -1,
            cpmvs_l0: AffineCpmvs::new_4param(MotionVector::ZERO, MotionVector::ZERO),
            cpmvs_l1: AffineCpmvs::new_4param(MotionVector::ZERO, MotionVector::ZERO),
            motion_model: MotionModel::Affine4Param,
            bcw_idx: 0,
        }
    }
}

// =====================================================================
// §8.5.5.5 — Derivation process for luma affine control point motion
// vectors from a neighbouring block
// =====================================================================

/// Per-list input to [`derive_inherited_affine_cpmvs`]. The spec runs
/// §8.5.5.5 once per active list (L0 + L1 when the neighbour is bi-pred);
/// callers pass `None` for any list the neighbour didn't carry
/// (`PredFlagLX[xNb][yNb] == 0`).
///
/// The two branches reflect §8.5.5.5's `isCTUboundary` split:
/// * [`NeighbourCpmvSource::AboveCtuBoundary`] — the neighbour CB lives
///   in the CTU row above the current one (`(yNb + nNbH) % CtbSizeY ==
///   0 && yNb + nNbH == yCb`). Eqs. 736 – 739 read the neighbour's
///   per-4×4 sub-block MV array at `(xNb, yNb + nNbH − 1)` (top-left
///   bottom-row anchor) and `(xNb + nNbW − 1, yNb + nNbH − 1)`
///   (top-right bottom-row anchor); the affine slope is reconstructed
///   from the difference between these two adjacent bottom-row sub-block
///   MVs. The 4-parameter similarity (`dHorY = −dVerX, dVerY = dHorX`)
///   per eq. 746 / 747 is forced even when the neighbour itself was
///   6-parameter.
/// * [`NeighbourCpmvSource::SameOrLeftCtu`] — the regular path: eqs.
///   740 – 745 read the neighbour CB's three CPMVs directly
///   (`CpMvLX[xNb][yNb][cpIdx]`), and the 6-parameter `dHorY / dVerY`
///   branch (eqs. 744 / 745) is taken iff `MotionModelIdc[xNb][yNb] ==
///   2`.
#[derive(Clone, Copy, Debug)]
pub enum NeighbourCpmvSource {
    /// Eqs. 736 – 739: neighbour spans a CTU boundary above the current
    /// CU. Carries the two sub-block MVs the spec reads from the
    /// neighbour's bottom row.
    AboveCtuBoundary {
        /// `MvLX[xNb][yNb + nNbH − 1]` — bottom-left sub-block of the
        /// neighbour (the sub-block immediately above the current CU's
        /// top-left).
        mv_bottom_left: MotionVector,
        /// `MvLX[xNb + nNbW − 1][yNb + nNbH − 1]` — bottom-right
        /// sub-block of the neighbour.
        mv_bottom_right: MotionVector,
    },
    /// Eqs. 740 – 745: neighbour is in the same CTU (or to the left).
    /// Carries the neighbour CB's CPMV record straight from
    /// `CpMvLX[xNb][yNb]`.
    SameOrLeftCtu {
        /// Neighbour CB's stored CPMV record (1 to 3 control points per
        /// `numCpMv = motionModelIdc + 1`).
        cpmvs: AffineCpmvs,
    },
}

/// Geometry inputs to [`derive_inherited_affine_cpmvs`]. Tracks the spec's
/// `(xCb, yCb, cbWidth, cbHeight)` for the current CU and
/// `(xNb, yNb, nNbW, nNbH)` for the neighbour CB.
#[derive(Clone, Copy, Debug)]
pub struct InheritedAffineGeom {
    /// Current CU top-left luma sample (picture-absolute).
    pub xcb: i32,
    pub ycb: i32,
    /// Current CU dimensions in luma samples (must be a power of two).
    pub cb_width: u32,
    pub cb_height: u32,
    /// Neighbour CB top-left luma sample (picture-absolute).
    pub xnb: i32,
    pub ynb: i32,
    /// Neighbour CB dimensions in luma samples (must be a power of two).
    pub nb_w: u32,
    pub nb_h: u32,
}

/// §8.5.5.5 — derive the inherited affine control-point MVs from one
/// neighbour CB.
///
/// `num_cp_mv` is the spec's `numCpMv` — 2 for `MotionModelIdc == 1` or
/// when the inherit caller wants a 4-parameter output; 3 for
/// `MotionModelIdc == 2`. The output [`AffineCpmvs`] carries the
/// corresponding [`MotionModel`].
///
/// Both eqs. 754 / 755 `Clip3(−2^17, 2^17 − 1, ·)` and the §8.5.2.14
/// signed-magnitude rounding with `rightShift = 7` are applied to
/// every emitted CP component.
pub fn derive_inherited_affine_cpmvs(
    geom: InheritedAffineGeom,
    source: NeighbourCpmvSource,
    num_cp_mv: u32,
) -> Result<AffineCpmvs> {
    assert!(
        num_cp_mv == 2 || num_cp_mv == 3,
        "inherited affine: numCpMv must be 2 or 3, got {num_cp_mv}"
    );
    assert!(
        geom.nb_w.is_power_of_two() && geom.nb_h.is_power_of_two(),
        "inherited affine: neighbour {}x{} not power-of-two",
        geom.nb_w,
        geom.nb_h,
    );

    let log2_nb_w = geom.nb_w.trailing_zeros() as i32; // eq. 734
    let log2_nb_h = geom.nb_h.trailing_zeros() as i32; // eq. 735

    // Derive mvScaleHor / mvScaleVer / dHorX / dVerX / dHorY / dVerY in
    // the spec's 1/128-pel-of-1/16-pel internal precision (i.e. the
    // value space the eqs. 748 – 753 evaluation expects). All arithmetic
    // is done in i64 to keep headroom for the `<< 7` and the partial-MV
    // multiplications.
    let (mv_scale_hor, mv_scale_ver, d_hor_x, d_ver_x, d_hor_y, d_ver_y) = match source {
        // Eqs. 736 – 739 + 746 / 747.
        NeighbourCpmvSource::AboveCtuBoundary {
            mv_bottom_left,
            mv_bottom_right,
        } => {
            let scale_hor = (mv_bottom_left.x as i64) << 7;
            let scale_ver = (mv_bottom_left.y as i64) << 7;
            // 7 - log2_nb_w can be negative for 256-wide neighbours
            // (none exist in VVC; max CB width is 128 ⇒ log2 == 7 ⇒
            // shift == 0 ⇒ identity). We still represent it as a signed
            // shift in i64 to keep the algebra straight.
            let shift_w = 7 - log2_nb_w;
            let d_hor_x = shift_signed(
                (mv_bottom_right.x as i64) - (mv_bottom_left.x as i64),
                shift_w,
            );
            let d_ver_x = shift_signed(
                (mv_bottom_right.y as i64) - (mv_bottom_left.y as i64),
                shift_w,
            );
            // CTU-boundary forces the 4-parameter similarity (eq. 746 /
            // 747), even when the neighbour was 6-parameter.
            (scale_hor, scale_ver, d_hor_x, d_ver_x, -d_ver_x, d_hor_x)
        }
        // Eqs. 740 – 745.
        NeighbourCpmvSource::SameOrLeftCtu { cpmvs } => {
            let cp0 = cpmvs.cpmvs[0];
            let cp1 = cpmvs.cpmvs[1];
            let cp2 = cpmvs.cpmvs[2];
            let scale_hor = (cp0.x as i64) << 7;
            let scale_ver = (cp0.y as i64) << 7;
            let shift_w = 7 - log2_nb_w;
            let d_hor_x = shift_signed((cp1.x as i64) - (cp0.x as i64), shift_w);
            let d_ver_x = shift_signed((cp1.y as i64) - (cp0.y as i64), shift_w);

            let (d_hor_y, d_ver_y) = match cpmvs.model {
                // Eqs. 744 / 745 — only when the neighbour is
                // 6-parameter.
                MotionModel::Affine6Param => {
                    let shift_h = 7 - log2_nb_h;
                    (
                        shift_signed((cp2.x as i64) - (cp0.x as i64), shift_h),
                        shift_signed((cp2.y as i64) - (cp0.y as i64), shift_h),
                    )
                }
                // Eq. 746 / 747 — 4-parameter similarity.
                _ => (-d_ver_x, d_hor_x),
            };
            (scale_hor, scale_ver, d_hor_x, d_ver_x, d_hor_y, d_ver_y)
        }
    };

    // For the CTU-boundary case the spec explicitly sets `yNb = yCb`
    // before applying eqs. 748 – 753 (the bullet right above eq. 748).
    let y_nb_eff = match source {
        NeighbourCpmvSource::AboveCtuBoundary { .. } => geom.ycb as i64,
        NeighbourCpmvSource::SameOrLeftCtu { .. } => geom.ynb as i64,
    };
    let x_nb = geom.xnb as i64;
    let x_cb = geom.xcb as i64;
    let y_cb = geom.ycb as i64;

    // Eq. 748 – 751: cpMvLX[0] / cpMvLX[1].
    let cp0_x_raw = mv_scale_hor + d_hor_x * (x_cb - x_nb) + d_hor_y * (y_cb - y_nb_eff);
    let cp0_y_raw = mv_scale_ver + d_ver_x * (x_cb - x_nb) + d_ver_y * (y_cb - y_nb_eff);
    let cp1_x_raw =
        mv_scale_hor + d_hor_x * (x_cb + geom.cb_width as i64 - x_nb) + d_hor_y * (y_cb - y_nb_eff);
    let cp1_y_raw =
        mv_scale_ver + d_ver_x * (x_cb + geom.cb_width as i64 - x_nb) + d_ver_y * (y_cb - y_nb_eff);

    let mut cps_raw: [(i64, i64); 3] = [(cp0_x_raw, cp0_y_raw), (cp1_x_raw, cp1_y_raw), (0, 0)];

    if num_cp_mv == 3 {
        // Eq. 752 / 753 — only when the *output* numCpMv is 3.
        let cp2_x_raw = mv_scale_hor
            + d_hor_x * (x_cb - x_nb)
            + d_hor_y * (y_cb + geom.cb_height as i64 - y_nb_eff);
        let cp2_y_raw = mv_scale_ver
            + d_ver_x * (x_cb - x_nb)
            + d_ver_y * (y_cb + geom.cb_height as i64 - y_nb_eff);
        cps_raw[2] = (cp2_x_raw, cp2_y_raw);
    }

    // §8.5.2.14 rounding with rightShift = 7, leftShift = 0 (eqs. 608 –
    // 610) and eqs. 754 / 755 Clip3(−2^17, 2^17 − 1, ·).
    let mut cps_out: [MotionVector; 3] = [MotionVector::ZERO; 3];
    let active = num_cp_mv as usize;
    for i in 0..active {
        let x_round = round_mv_component(cps_raw[i].0, 7);
        let y_round = round_mv_component(cps_raw[i].1, 7);
        cps_out[i] = MotionVector {
            x: clip_mv17(x_round),
            y: clip_mv17(y_round),
        };
    }
    let model = if num_cp_mv == 3 {
        MotionModel::Affine6Param
    } else {
        MotionModel::Affine4Param
    };
    Ok(AffineCpmvs {
        model,
        cpmvs: cps_out,
    })
}

// =====================================================================
// §8.5.5.6 — Derivation process for constructed affine control point
// motion vector merging candidates
// =====================================================================

/// `sps_6param_affine_enabled_flag` gates Const1..4 (the three
/// 6-parameter triples). When `false`, only Const5 / Const6 are derived
/// — matching the spec's "When sps_6param_affine_enabled_flag is equal
/// to 1, the first four constructed affine control point motion vector
/// merging candidates ConstK with K = 1..4 ... are derived" bullet.
///
/// `slice_type_b` controls the L1 half of Const4 / corner-3: per
/// §8.5.5.6 fourth-corner bullet, `availableFlagCorner[3]` only OR-folds
/// L1Col into the corner-3 availability for B-slices; the same bullet's
/// later "When sh_slice_type is equal to B" branch is the only place
/// `predFlagL1Corner[3]` ever becomes 1.
#[derive(Clone, Copy, Debug)]
pub struct ConstructedAffineFlags {
    pub sps_6param_affine_enabled_flag: bool,
    pub slice_type_b: bool,
}

/// Output buffer for the six constructed affine merge candidates. Slot
/// `k - 1` carries the spec's `ConstK` payload. Slots whose
/// `availableFlagConstK == FALSE` carry the default
/// [`AffineMergeCandidate`] but the `[bool; 6]` flag array reports
/// availability directly.
#[derive(Clone, Copy, Debug, Default)]
pub struct ConstructedAffineCandidates {
    /// `availableFlagConstK` for K = 1..6 (zero-indexed: `[0] ==
    /// Const1`, `[5] == Const6`).
    pub available: [bool; 6],
    /// `ConstK` payloads, parallel to [`Self::available`].
    pub cands: [AffineMergeCandidate; 6],
}

impl ConstructedAffineCandidates {
    /// Number of `availableFlagConstK == TRUE` entries.
    pub fn count(&self) -> usize {
        self.available.iter().filter(|b| **b).count()
    }
}

/// §8.5.5.6 — derive the six constructed affine merge candidates from
/// per-corner inputs.
///
/// The four corner inputs follow the spec's k ∈ {0, 1, 2, 3} layout
/// described on [`AffineCpRecord`]. Inputs are typically the output of
/// the §8.5.5.2 step 6 corner-selection cascade: corner 0 picked from
/// the B2 / B3 / A2 sequence; corner 1 from B1 / B0; corner 2 from
/// A1 / A0; corner 3 from the §8.5.2.11 collocated MV at the
/// bottom-right.
pub fn derive_constructed_affine_merge_candidates(
    cb_width: u32,
    cb_height: u32,
    corners: &[AffineCpRecord; 4],
    flags: ConstructedAffineFlags,
) -> ConstructedAffineCandidates {
    let mut out = ConstructedAffineCandidates::default();

    // -- Const1..4: 6-parameter triples (gated by
    //    sps_6param_affine_enabled_flag) ------------------------------
    if flags.sps_6param_affine_enabled_flag {
        // Const1: corners {0, 1, 2} ⇒ cpMvLXConst1 = (CP0, CP1, CP2).
        if corners[0].available && corners[1].available && corners[2].available {
            if let Some(cand) =
                build_const_triple_6param(corners, 0, 1, 2, ConstTripleAssembly::Const1)
            {
                out.cands[0] = cand;
                out.available[0] = true;
            }
        }
        // Const2: corners {0, 1, 3} ⇒ cpMvLXConst2[2] = CP3 + CP0 − CP1
        //                                              (clipped).
        if corners[0].available && corners[1].available && corners[3].available {
            if let Some(cand) =
                build_const_triple_6param(corners, 0, 1, 3, ConstTripleAssembly::Const2)
            {
                out.cands[1] = cand;
                out.available[1] = true;
            }
        }
        // Const3: corners {0, 2, 3} ⇒ cpMvLXConst3[1] = CP3 + CP0 − CP2
        //                                              (clipped).
        if corners[0].available && corners[2].available && corners[3].available {
            if let Some(cand) =
                build_const_triple_6param(corners, 0, 2, 3, ConstTripleAssembly::Const3)
            {
                out.cands[2] = cand;
                out.available[2] = true;
            }
        }
        // Const4: corners {1, 2, 3} ⇒ cpMvLXConst4[0] = CP1 + CP2 − CP3
        //                                              (clipped).
        if corners[1].available && corners[2].available && corners[3].available {
            if let Some(cand) =
                build_const_triple_6param(corners, 1, 2, 3, ConstTripleAssembly::Const4)
            {
                out.cands[3] = cand;
                out.available[3] = true;
            }
        }
    }

    // -- Const5: 4-parameter pair {0, 1} -------------------------------
    if corners[0].available && corners[1].available {
        if let Some(cand) = build_const_pair_4param_top(corners) {
            out.cands[4] = cand;
            out.available[4] = true;
        }
    }
    // -- Const6: 4-parameter pair {0, 2} with the eq. 811 / 812
    //    diagonal-projected top-right derivation -----------------------
    if corners[0].available && corners[2].available {
        if let Some(cand) = build_const_pair_4param_diag(corners, cb_width, cb_height) {
            out.cands[5] = cand;
            out.available[5] = true;
        }
    }

    let _ = flags.slice_type_b; // §8.5.5.6 fourth-corner bullet already
                                // baked into corners[3] by the caller
                                // (B-slice handling). Suppress warning.

    out
}

/// Which 6-parameter triple to assemble — selects between the eq. 783 –
/// 806 cpMvLXConstK[cpIdx] permutation tables.
#[derive(Clone, Copy, Debug)]
enum ConstTripleAssembly {
    /// Const1: cpMvLXConst1 = (CP0, CP1, CP2). Eqs. 783 – 785.
    Const1,
    /// Const2: cpMvLXConst2 = (CP0, CP1, CP3 + CP0 − CP1). Eqs. 788 –
    /// 792 (CP[2] derived + clipped).
    Const2,
    /// Const3: cpMvLXConst3 = (CP0, CP3 + CP0 − CP2, CP2). Eqs. 795 –
    /// 799 (CP[1] derived + clipped).
    Const3,
    /// Const4: cpMvLXConst4 = (CP1 + CP2 − CP3, CP1, CP2). Eqs. 802 –
    /// 806 (CP[0] derived + clipped).
    Const4,
}

fn build_const_triple_6param(
    corners: &[AffineCpRecord; 4],
    ka: usize,
    kb: usize,
    kc: usize,
    assembly: ConstTripleAssembly,
) -> Option<AffineMergeCandidate> {
    // Per-list availability gate per spec: predFlagLX must be 1 at all
    // three corners AND all three refIdx must match. Run for X = 0 and
    // X = 1 independently.
    let a = &corners[ka];
    let b = &corners[kb];
    let c = &corners[kc];

    let l0_ok = a.pred_flag_l0
        && b.pred_flag_l0
        && c.pred_flag_l0
        && a.ref_idx_l0 == b.ref_idx_l0
        && a.ref_idx_l0 == c.ref_idx_l0;
    let l1_ok = a.pred_flag_l1
        && b.pred_flag_l1
        && c.pred_flag_l1
        && a.ref_idx_l1 == b.ref_idx_l1
        && a.ref_idx_l1 == c.ref_idx_l1;

    // "If availableFlagL0 or availableFlagL1 is equal to 1,
    //  availableFlagConstK is set equal to TRUE".
    if !l0_ok && !l1_ok {
        return None;
    }

    let (cp_l0_a, cp_l0_b, cp_l0_c) = if l0_ok {
        (a.mv_l0, b.mv_l0, c.mv_l0)
    } else {
        (MotionVector::ZERO, MotionVector::ZERO, MotionVector::ZERO)
    };
    let (cp_l1_a, cp_l1_b, cp_l1_c) = if l1_ok {
        (a.mv_l1, b.mv_l1, c.mv_l1)
    } else {
        (MotionVector::ZERO, MotionVector::ZERO, MotionVector::ZERO)
    };

    let (cpmvs_l0, cpmvs_l1) = match assembly {
        ConstTripleAssembly::Const1 => (
            AffineCpmvs::new_6param(cp_l0_a, cp_l0_b, cp_l0_c),
            AffineCpmvs::new_6param(cp_l1_a, cp_l1_b, cp_l1_c),
        ),
        ConstTripleAssembly::Const2 => {
            // CP[2] = CP3 + CP0 − CP1, then Clip3(−2^17, 2^17 − 1, ·).
            // Inputs CP0 / CP1 / CP3 are the ka / kb / kc corners.
            let cp2_l0 = combine_clipped(cp_l0_c, cp_l0_a, cp_l0_b);
            let cp2_l1 = combine_clipped(cp_l1_c, cp_l1_a, cp_l1_b);
            (
                AffineCpmvs::new_6param(cp_l0_a, cp_l0_b, cp2_l0),
                AffineCpmvs::new_6param(cp_l1_a, cp_l1_b, cp2_l1),
            )
        }
        ConstTripleAssembly::Const3 => {
            // CP[1] = CP3 + CP0 − CP2 (corners {0,2,3} ⇒ ka=0, kb=2, kc=3).
            let cp1_l0 = combine_clipped(cp_l0_c, cp_l0_a, cp_l0_b);
            let cp1_l1 = combine_clipped(cp_l1_c, cp_l1_a, cp_l1_b);
            (
                AffineCpmvs::new_6param(cp_l0_a, cp1_l0, cp_l0_b),
                AffineCpmvs::new_6param(cp_l1_a, cp1_l1, cp_l1_b),
            )
        }
        ConstTripleAssembly::Const4 => {
            // CP[0] = CP1 + CP2 − CP3 (corners {1,2,3} ⇒ ka=1, kb=2, kc=3).
            let cp0_l0 = combine_clipped(cp_l0_a, cp_l0_b, cp_l0_c);
            let cp0_l1 = combine_clipped(cp_l1_a, cp_l1_b, cp_l1_c);
            (
                AffineCpmvs::new_6param(cp0_l0, cp_l0_a, cp_l0_b),
                AffineCpmvs::new_6param(cp0_l1, cp_l1_a, cp_l1_b),
            )
        }
    };

    // bcwIdx: derived from corner 0 for Const1/2/3, corner 1 for Const4
    // (per the corresponding spec bullets). Only emitted as non-zero
    // when both L0 + L1 sides materialised.
    let bcw_idx = if l0_ok && l1_ok {
        match assembly {
            ConstTripleAssembly::Const4 => corners[1].bcw_idx,
            _ => corners[0].bcw_idx,
        }
    } else {
        0
    };

    Some(AffineMergeCandidate {
        pred_flag_l0: l0_ok,
        pred_flag_l1: l1_ok,
        ref_idx_l0: if l0_ok { a.ref_idx_l0 } else { -1 },
        ref_idx_l1: if l1_ok { a.ref_idx_l1 } else { -1 },
        cpmvs_l0,
        cpmvs_l1,
        motion_model: MotionModel::Affine6Param,
        bcw_idx,
    })
}

/// `out = a + b − c`, each component independently clipped to
/// [−2^17, 2^17 − 1] per eqs. 791 / 792 / 797 / 798 / 803 / 804. The
/// clip happens after the signed difference, mirroring the spec's
/// `Clip3(−2^17, 2^17 − 1, cpMvLXConstK[cpIdx][·])` placement (which is
/// itself the *only* clip; intermediate sums of two i32 1/16-pel values
/// are well within i32 range so no overflow guard is needed before the
/// clip).
fn combine_clipped(a: MotionVector, b: MotionVector, c: MotionVector) -> MotionVector {
    let raw_x = (a.x as i64) + (b.x as i64) - (c.x as i64);
    let raw_y = (a.y as i64) + (b.y as i64) - (c.y as i64);
    MotionVector {
        x: clip_mv17(raw_x),
        y: clip_mv17(raw_y),
    }
}

/// Const5 — eq. 807 – 810: cpMvLXConst5 = (CP0, CP1). The 4-parameter
/// pair from corners {0, 1}.
fn build_const_pair_4param_top(corners: &[AffineCpRecord; 4]) -> Option<AffineMergeCandidate> {
    let a = &corners[0];
    let b = &corners[1];

    let l0_ok = a.pred_flag_l0 && b.pred_flag_l0 && a.ref_idx_l0 == b.ref_idx_l0;
    let l1_ok = a.pred_flag_l1 && b.pred_flag_l1 && a.ref_idx_l1 == b.ref_idx_l1;
    if !l0_ok && !l1_ok {
        return None;
    }

    let cpmvs_l0 = if l0_ok {
        AffineCpmvs::new_4param(a.mv_l0, b.mv_l0)
    } else {
        AffineCpmvs::new_4param(MotionVector::ZERO, MotionVector::ZERO)
    };
    let cpmvs_l1 = if l1_ok {
        AffineCpmvs::new_4param(a.mv_l1, b.mv_l1)
    } else {
        AffineCpmvs::new_4param(MotionVector::ZERO, MotionVector::ZERO)
    };
    let bcw_idx = if l0_ok && l1_ok {
        corners[0].bcw_idx
    } else {
        0
    };
    Some(AffineMergeCandidate {
        pred_flag_l0: l0_ok,
        pred_flag_l1: l1_ok,
        ref_idx_l0: if l0_ok { a.ref_idx_l0 } else { -1 },
        ref_idx_l1: if l1_ok { a.ref_idx_l1 } else { -1 },
        cpmvs_l0,
        cpmvs_l1,
        motion_model: MotionModel::Affine4Param,
        bcw_idx,
    })
}

/// Const6 — eqs. 811 – 818: 4-parameter pair from corners {0, 2}. The
/// second CPMV (top-right) is *derived* from the bottom-left CPMV via
/// the eq. 811 / 812 diagonal projection, then §8.5.2.14 rounded with
/// `rightShift = 7` and clipped per eq. 817 / 818.
///
/// The derivation, in shorthand (`s = 7 + log2(W) − log2(H)`):
///
/// ```text
/// CP1.x = (CP0.x << 7) + ((CP2.y − CP0.y) << s)
/// CP1.y = (CP0.y << 7) − ((CP2.x − CP0.x) << s)
/// ```
///
/// The `<< 7` left-shift puts the result back into the precision space
/// the rounding step expects (the `>> 7` undoes the `<< 7`, modulo the
/// added 2^6 offset).
fn build_const_pair_4param_diag(
    corners: &[AffineCpRecord; 4],
    cb_width: u32,
    cb_height: u32,
) -> Option<AffineMergeCandidate> {
    assert!(cb_width.is_power_of_two() && cb_height.is_power_of_two());
    let a = &corners[0];
    let c = &corners[2];

    let l0_ok = a.pred_flag_l0 && c.pred_flag_l0 && a.ref_idx_l0 == c.ref_idx_l0;
    let l1_ok = a.pred_flag_l1 && c.pred_flag_l1 && a.ref_idx_l1 == c.ref_idx_l1;
    if !l0_ok && !l1_ok {
        return None;
    }

    let log2_w = cb_width.trailing_zeros() as i32;
    let log2_h = cb_height.trailing_zeros() as i32;
    // 7 + log2(W) − log2(H). May be negative when log2_h > log2_w + 7
    // (impossible for any CU size VVC allows — max log2 is 7), so we
    // simply represent it as a signed shift through [`shift_signed`].
    let shift_wh = 7 + log2_w - log2_h;

    let derive_cp1 = |cp0: MotionVector, cp2: MotionVector| -> MotionVector {
        // Eq. 811 / 812 — in 1/(128 * 16)-pel units.
        let cp1_x_pre =
            ((cp0.x as i64) << 7) + shift_signed((cp2.y as i64) - (cp0.y as i64), shift_wh);
        let cp1_y_pre =
            ((cp0.y as i64) << 7) - shift_signed((cp2.x as i64) - (cp0.x as i64), shift_wh);
        // §8.5.2.14 rounding with rightShift = 7, leftShift = 0.
        let cp1_x = round_mv_component(cp1_x_pre, 7);
        let cp1_y = round_mv_component(cp1_y_pre, 7);
        MotionVector {
            x: clip_mv17(cp1_x),
            y: clip_mv17(cp1_y),
        }
    };

    let cpmvs_l0 = if l0_ok {
        AffineCpmvs::new_4param(a.mv_l0, derive_cp1(a.mv_l0, c.mv_l0))
    } else {
        AffineCpmvs::new_4param(MotionVector::ZERO, MotionVector::ZERO)
    };
    let cpmvs_l1 = if l1_ok {
        AffineCpmvs::new_4param(a.mv_l1, derive_cp1(a.mv_l1, c.mv_l1))
    } else {
        AffineCpmvs::new_4param(MotionVector::ZERO, MotionVector::ZERO)
    };
    let bcw_idx = if l0_ok && l1_ok {
        corners[0].bcw_idx
    } else {
        0
    };
    Some(AffineMergeCandidate {
        pred_flag_l0: l0_ok,
        pred_flag_l1: l1_ok,
        ref_idx_l0: if l0_ok { a.ref_idx_l0 } else { -1 },
        ref_idx_l1: if l1_ok { a.ref_idx_l1 } else { -1 },
        cpmvs_l0,
        cpmvs_l1,
        motion_model: MotionModel::Affine4Param,
        bcw_idx,
    })
}

// =====================================================================
// Local helpers
// =====================================================================

/// §8.5.2.14 rounding (signed-magnitude) for one i64 value. Mirrors
/// `affine.rs::round_mv_component` (kept private there) — defined here
/// to avoid a cross-module helper export, and to allow this module to
/// stand on its own once the inherited / constructed derivation paths
/// land in a fully wired CTU walker.
///
/// Formula: `offset = (rightShift == 0) ? 0 : ((1 << (rightShift − 1)) − 1)`,
/// `out = sign(in) * ((|in| + offset) >> rightShift)` (eq. 608 – 610
/// with leftShift = 0).
fn round_mv_component(v: i64, right_shift: i32) -> i64 {
    if right_shift == 0 {
        return v;
    }
    let offset = (1i64 << (right_shift - 1)) - 1;
    let sign = if v < 0 { -1 } else { 1 };
    sign * (((v.abs()) + offset) >> right_shift)
}

/// `Clip3(−2^17, 2^17 − 1, ·)` per eqs. 754 / 755 / 791 / 792 / 797 /
/// 798 / 803 / 804 / 817 / 818. Folds an i64 to the spec's stored MV
/// range.
fn clip_mv17(v: i64) -> i32 {
    v.clamp(-(1i64 << 17), (1i64 << 17) - 1) as i32
}

/// Signed-shift helper — `v << n` for `n >= 0`, `v >> (-n)` (arithmetic
/// right shift, signed-magnitude rounding NOT applied) for `n < 0`.
/// The shift amounts we see (`7 − log2_nb_w` etc.) are non-negative for
/// every legal VVC CB size, so the right-shift branch is dead code in
/// production but kept for defensive correctness in tests that
/// hypothetically pass an oversized CB.
fn shift_signed(v: i64, n: i32) -> i64 {
    if n >= 0 {
        v << n
    } else {
        v >> (-n)
    }
}

// =====================================================================
// Tests — §8.5.5.5 inherited + §8.5.5.6 constructed derivation
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn mv(x: i32, y: i32) -> MotionVector {
        MotionVector { x, y }
    }

    // ----- §8.5.2.14 rounding -----------------------------------------

    #[test]
    fn round_mv_component_rightshift_zero_is_identity() {
        for v in [-1024, -1, 0, 1, 1024] {
            assert_eq!(round_mv_component(v, 0), v);
        }
    }

    #[test]
    fn round_mv_component_matches_spec_eq608_offset() {
        // rightShift = 7 → offset = 63 → 128 ⇒ 1, 64 ⇒ 0 (sign-magnitude
        // round-toward-zero), −128 ⇒ −1, −64 ⇒ 0.
        assert_eq!(round_mv_component(128, 7), 1);
        assert_eq!(round_mv_component(64, 7), 0);
        assert_eq!(round_mv_component(-128, 7), -1);
        assert_eq!(round_mv_component(-64, 7), 0);
        // 191 / 128 ≈ 1.49 → 1; 192 / 128 = 1.5 → 1 (offset 63 ⇒
        // (192 + 63) >> 7 = 255 >> 7 = 1, which is round-half-down for
        // ties — matches the spec's exact formula).
        assert_eq!(round_mv_component(191, 7), 1);
        assert_eq!(round_mv_component(192, 7), 1);
        assert_eq!(round_mv_component(255, 7), 2);
    }

    // ----- Clip3 ------------------------------------------------------

    #[test]
    fn clip_mv17_clamps_to_spec_bounds() {
        assert_eq!(clip_mv17(0), 0);
        assert_eq!(clip_mv17(1 << 17), (1 << 17) - 1); // upper saturation
        assert_eq!(clip_mv17(-(1 << 17) - 1), -(1 << 17)); // lower saturation
        assert_eq!(clip_mv17(1 << 30), (1 << 17) - 1);
    }

    // ----- §8.5.5.5 inherited from neighbour --------------------------

    /// A 16×16 neighbour CB at (0, 0) with translational-degenerate
    /// CPMVs `(8, 0)` for both CP0 and CP1 → eq. 740 – 743 collapse:
    /// mvScaleHor = 8 << 7 = 1024, dHorX = dVerX = 0, dHorY = dVerY = 0.
    /// CPMV emit: cp0 = (8, 0), cp1 = (8, 0), cp2 = (8, 0) regardless
    /// of the current CU geometry — every inherited CP is identical to
    /// the neighbour's translational MV.
    #[test]
    fn inherited_translational_neighbour_collapses_to_uniform_mv() {
        let geom = InheritedAffineGeom {
            xcb: 16,
            ycb: 0,
            cb_width: 16,
            cb_height: 16,
            xnb: 0,
            ynb: 0,
            nb_w: 16,
            nb_h: 16,
        };
        let cpmvs = AffineCpmvs::new_4param(mv(8, 0), mv(8, 0));
        let out =
            derive_inherited_affine_cpmvs(geom, NeighbourCpmvSource::SameOrLeftCtu { cpmvs }, 2)
                .expect("derive ok");
        assert_eq!(out.model, MotionModel::Affine4Param);
        assert_eq!(out.cpmvs[0], mv(8, 0));
        assert_eq!(out.cpmvs[1], mv(8, 0));
    }

    /// A 16×16 4-parameter affine neighbour with CP0 = (0, 0), CP1 =
    /// (256, 0) → uniform horizontal shear: every column gets a +1-pel
    /// (16/16 of 1/16-pel) MV per column step. Current CU is at (16, 0)
    /// — eq. 748 / 750 should produce CP0_new = (256, 0), CP1_new =
    /// (512, 0).
    #[test]
    fn inherited_4param_neighbour_uniform_horizontal_shear() {
        let geom = InheritedAffineGeom {
            xcb: 16,
            ycb: 0,
            cb_width: 16,
            cb_height: 16,
            xnb: 0,
            ynb: 0,
            nb_w: 16,
            nb_h: 16,
        };
        let cpmvs = AffineCpmvs::new_4param(mv(0, 0), mv(256, 0));
        let out =
            derive_inherited_affine_cpmvs(geom, NeighbourCpmvSource::SameOrLeftCtu { cpmvs }, 2)
                .expect("derive ok");
        // mvScaleHor = 0 << 7 = 0
        // dHorX = (256 - 0) << (7 - 4) = 256 << 3 = 2048
        // dVerX = 0 << 3 = 0
        // dHorY = -dVerX = 0
        // dVerY = dHorX = 2048
        // CP0_new: 0 + 2048 * (16 - 0) + 0 * (0 - 0) = 32768
        //   → round_mv_component(32768, 7) = (32768 + 63) >> 7 = 256
        // CP1_new: 0 + 2048 * (16 + 16 - 0) + 0 * 0 = 65536
        //   → (65536 + 63) >> 7 = 512
        assert_eq!(out.cpmvs[0].x, 256, "cp0.x");
        assert_eq!(out.cpmvs[0].y, 0, "cp0.y");
        assert_eq!(out.cpmvs[1].x, 512, "cp1.x");
        assert_eq!(out.cpmvs[1].y, 0, "cp1.y");
    }

    /// Asking for 3 CPMVs from a 4-param neighbour still works (the
    /// derivation uses the 4-param similarity for the dHorY / dVerY
    /// pair, then evaluates eq. 752 / 753 for the bottom-left CP).
    /// With the uniform horizontal shear above, the bottom-left CPMV
    /// (CP2_new) ends up at `(CP0_new.x, CP0_new.y + cbHeight * dVerY /
    /// 128)` — i.e. (256, 0 + 16 * 2048 / 128) = (256, 256).
    #[test]
    fn inherited_4param_neighbour_third_cpmv_uses_similarity() {
        let geom = InheritedAffineGeom {
            xcb: 16,
            ycb: 0,
            cb_width: 16,
            cb_height: 16,
            xnb: 0,
            ynb: 0,
            nb_w: 16,
            nb_h: 16,
        };
        let cpmvs = AffineCpmvs::new_4param(mv(0, 0), mv(256, 0));
        let out =
            derive_inherited_affine_cpmvs(geom, NeighbourCpmvSource::SameOrLeftCtu { cpmvs }, 3)
                .expect("derive ok");
        assert_eq!(out.model, MotionModel::Affine6Param);
        // CP2_new.x: 0 + 2048 * (16 - 0) + 0 * 16 = 32768 → 256
        // CP2_new.y: 0 + 0 * 16 + 2048 * (16 - 0) = 32768 → 256
        assert_eq!(out.cpmvs[2].x, 256, "cp2.x");
        assert_eq!(out.cpmvs[2].y, 256, "cp2.y");
    }

    /// CTU-boundary path: neighbour is in the row above the current CU.
    /// The bottom-row sub-block MVs `mv_bl = (0, 0)` and `mv_br =
    /// (16, 0)` give a horizontal slope; the inherited CPMVs of the
    /// current CU at (0, 16) get derived from those.
    #[test]
    fn inherited_above_ctu_boundary_horizontal_shear() {
        let geom = InheritedAffineGeom {
            xcb: 0,
            ycb: 16,
            cb_width: 16,
            cb_height: 16,
            xnb: 0,
            ynb: 0,
            nb_w: 16,
            nb_h: 16,
        };
        let src = NeighbourCpmvSource::AboveCtuBoundary {
            mv_bottom_left: mv(0, 0),
            mv_bottom_right: mv(16, 0),
        };
        let out = derive_inherited_affine_cpmvs(geom, src, 2).expect("derive ok");
        // mvScaleHor = 0 << 7 = 0
        // dHorX = (16 - 0) << (7 - 4) = 16 << 3 = 128
        // dVerX = 0
        // dHorY = -dVerX = 0; dVerY = dHorX = 128.
        // yNb_eff = yCb = 16 ⇒ (yCb - yNb_eff) = 0.
        // CP0_new: 0 + 128 * (0 - 0) + 0 * 0 = 0 → 0.
        // CP1_new: 0 + 128 * (0 + 16 - 0) + 0 * 0 = 2048 → 16.
        assert_eq!(out.cpmvs[0], mv(0, 0));
        assert_eq!(out.cpmvs[1].x, 16);
        assert_eq!(out.cpmvs[1].y, 0);
    }

    /// CTU-boundary path always emits 4-parameter similarity even when
    /// the spec's `MotionModelIdc[xNb][yNb] == 2`. The
    /// AboveCtuBoundary source carries only the two bottom-row MVs;
    /// the dHorY / dVerY pair is forced to `-dVerX / dHorX` per eqs.
    /// 746 / 747.
    #[test]
    fn inherited_above_ctu_boundary_forces_4param_similarity() {
        let geom = InheritedAffineGeom {
            xcb: 0,
            ycb: 16,
            cb_width: 16,
            cb_height: 16,
            xnb: 0,
            ynb: 0,
            nb_w: 16,
            nb_h: 16,
        };
        let src = NeighbourCpmvSource::AboveCtuBoundary {
            mv_bottom_left: mv(0, 0),
            mv_bottom_right: mv(0, 16), // dVerX != 0 ⇒ true vertical slope on bottom edge
        };
        // Even when we ask for 3 CPMVs, dHorY = -dVerX = -16<<3 = -128,
        // dVerY = dHorX = 0 — the third CPMV reflects the similarity
        // imposed by eqs. 746 / 747, NOT an independent 6-param slope.
        let out = derive_inherited_affine_cpmvs(geom, src, 3).expect("derive ok");
        // CP2_new.x: 0 + 0 * (0 - 0) + (-128) * (16 + 16 - 16) = -2048 → -16
        // CP2_new.y: 0 + 128 * (0 - 0) + 0 * 16 = 0
        assert_eq!(out.cpmvs[2].x, -16);
        assert_eq!(out.cpmvs[2].y, 0);
    }

    // ----- §8.5.5.6 constructed candidates ----------------------------

    /// Helper: a synthetic 4-corner setup where every corner is L0-only,
    /// refIdx 0, with caller-provided MVs.
    fn l0_corners(
        cp0: MotionVector,
        cp1: MotionVector,
        cp2: MotionVector,
        cp3: MotionVector,
    ) -> [AffineCpRecord; 4] {
        let mk = |m: MotionVector| AffineCpRecord {
            available: true,
            pred_flag_l0: true,
            pred_flag_l1: false,
            ref_idx_l0: 0,
            ref_idx_l1: -1,
            mv_l0: m,
            mv_l1: MotionVector::ZERO,
            bcw_idx: 0,
        };
        [mk(cp0), mk(cp1), mk(cp2), mk(cp3)]
    }

    #[test]
    fn const1_triple_cps_0_1_2_assembles_6param_when_all_three_available() {
        let corners = l0_corners(mv(10, 0), mv(40, 0), mv(0, 40), mv(50, 50));
        let out = derive_constructed_affine_merge_candidates(
            32,
            32,
            &corners,
            ConstructedAffineFlags {
                sps_6param_affine_enabled_flag: true,
                slice_type_b: false,
            },
        );
        assert!(out.available[0], "Const1 should be available");
        let c = &out.cands[0];
        assert_eq!(c.motion_model, MotionModel::Affine6Param);
        assert_eq!(c.cpmvs_l0.cpmvs[0], mv(10, 0));
        assert_eq!(c.cpmvs_l0.cpmvs[1], mv(40, 0));
        assert_eq!(c.cpmvs_l0.cpmvs[2], mv(0, 40));
        assert!(c.pred_flag_l0);
        assert!(!c.pred_flag_l1);
    }

    #[test]
    fn const2_cp2_derived_as_cp3_plus_cp0_minus_cp1() {
        let corners = l0_corners(mv(10, 0), mv(40, 0), mv(0, 40), mv(50, 50));
        let out = derive_constructed_affine_merge_candidates(
            32,
            32,
            &corners,
            ConstructedAffineFlags {
                sps_6param_affine_enabled_flag: true,
                slice_type_b: false,
            },
        );
        assert!(out.available[1], "Const2 should be available");
        let c = &out.cands[1];
        // CP[2] = CP3 + CP0 − CP1 = (50, 50) + (10, 0) − (40, 0) = (20, 50)
        assert_eq!(c.cpmvs_l0.cpmvs[2], mv(20, 50));
        assert_eq!(c.cpmvs_l0.cpmvs[0], mv(10, 0));
        assert_eq!(c.cpmvs_l0.cpmvs[1], mv(40, 0));
    }

    #[test]
    fn const3_cp1_derived_as_cp3_plus_cp0_minus_cp2() {
        let corners = l0_corners(mv(10, 0), mv(40, 0), mv(0, 40), mv(50, 50));
        let out = derive_constructed_affine_merge_candidates(
            32,
            32,
            &corners,
            ConstructedAffineFlags {
                sps_6param_affine_enabled_flag: true,
                slice_type_b: false,
            },
        );
        assert!(out.available[2], "Const3 should be available");
        let c = &out.cands[2];
        // CP[1] = CP3 + CP0 − CP2 = (50, 50) + (10, 0) − (0, 40) = (60, 10)
        assert_eq!(c.cpmvs_l0.cpmvs[1], mv(60, 10));
        assert_eq!(c.cpmvs_l0.cpmvs[0], mv(10, 0));
        assert_eq!(c.cpmvs_l0.cpmvs[2], mv(0, 40));
    }

    #[test]
    fn const4_cp0_derived_as_cp1_plus_cp2_minus_cp3() {
        let corners = l0_corners(mv(10, 0), mv(40, 0), mv(0, 40), mv(50, 50));
        let out = derive_constructed_affine_merge_candidates(
            32,
            32,
            &corners,
            ConstructedAffineFlags {
                sps_6param_affine_enabled_flag: true,
                slice_type_b: false,
            },
        );
        assert!(out.available[3], "Const4 should be available");
        let c = &out.cands[3];
        // CP[0] = CP1 + CP2 − CP3 = (40, 0) + (0, 40) − (50, 50) = (-10, -10)
        assert_eq!(c.cpmvs_l0.cpmvs[0], mv(-10, -10));
        assert_eq!(c.cpmvs_l0.cpmvs[1], mv(40, 0));
        assert_eq!(c.cpmvs_l0.cpmvs[2], mv(0, 40));
    }

    #[test]
    fn const5_pair_cps_0_1_emits_4param() {
        let corners = l0_corners(mv(10, 0), mv(40, 0), mv(0, 40), mv(50, 50));
        let out = derive_constructed_affine_merge_candidates(
            32,
            32,
            &corners,
            ConstructedAffineFlags {
                sps_6param_affine_enabled_flag: true,
                slice_type_b: false,
            },
        );
        assert!(out.available[4], "Const5 should be available");
        let c = &out.cands[4];
        assert_eq!(c.motion_model, MotionModel::Affine4Param);
        assert_eq!(c.cpmvs_l0.cpmvs[0], mv(10, 0));
        assert_eq!(c.cpmvs_l0.cpmvs[1], mv(40, 0));
    }

    #[test]
    fn const6_pair_derives_top_right_from_diagonal_projection() {
        // Use a square CU so log2(W) - log2(H) = 0 ⇒ shift_wh = 7.
        // CP0 = (16, 0), CP2 = (0, 16) ⇒ horizontal-vertical orthogonal
        // rotation by 90°: CP1 should be derived as (16, 16).
        let mut corners = [AffineCpRecord::UNAVAILABLE; 4];
        corners[0] = AffineCpRecord {
            available: true,
            pred_flag_l0: true,
            pred_flag_l1: false,
            ref_idx_l0: 0,
            ref_idx_l1: -1,
            mv_l0: mv(16, 0),
            mv_l1: MotionVector::ZERO,
            bcw_idx: 0,
        };
        corners[2] = AffineCpRecord {
            available: true,
            pred_flag_l0: true,
            pred_flag_l1: false,
            ref_idx_l0: 0,
            ref_idx_l1: -1,
            mv_l0: mv(0, 16),
            mv_l1: MotionVector::ZERO,
            bcw_idx: 0,
        };
        let out = derive_constructed_affine_merge_candidates(
            16,
            16,
            &corners,
            ConstructedAffineFlags {
                sps_6param_affine_enabled_flag: true,
                slice_type_b: false,
            },
        );
        assert!(out.available[5], "Const6 should be available");
        let c = &out.cands[5];
        assert_eq!(c.motion_model, MotionModel::Affine4Param);
        // Eq. 811: CP1.x_pre = (16 << 7) + ((16 - 0) << 7) = 2048 + 2048 = 4096
        //          rounded: (4096 + 63) >> 7 = 4159 >> 7 = 32 → clip → 32.
        // Eq. 812: CP1.y_pre = (0 << 7) - ((0 - 16) << 7) = 0 - (-16 << 7) = 2048
        //          rounded: (2048 + 63) >> 7 = 16.
        assert_eq!(c.cpmvs_l0.cpmvs[0], mv(16, 0));
        assert_eq!(c.cpmvs_l0.cpmvs[1].x, 32);
        assert_eq!(c.cpmvs_l0.cpmvs[1].y, 16);
    }

    /// When sps_6param_affine_enabled_flag == 0, Const1..4 must not be
    /// emitted; Const5 / Const6 still appear.
    #[test]
    fn sps_6param_disabled_suppresses_const1_through_const4() {
        let corners = l0_corners(mv(10, 0), mv(40, 0), mv(0, 40), mv(50, 50));
        let out = derive_constructed_affine_merge_candidates(
            32,
            32,
            &corners,
            ConstructedAffineFlags {
                sps_6param_affine_enabled_flag: false,
                slice_type_b: false,
            },
        );
        assert!(!out.available[0]);
        assert!(!out.available[1]);
        assert!(!out.available[2]);
        assert!(!out.available[3]);
        assert!(out.available[4]);
        assert!(out.available[5]);
        assert_eq!(out.count(), 2);
    }

    /// A corner with `pred_flag_l0 == 0` must short-circuit both Const1
    /// (corners 0/1/2) and Const5 (corners 0/1) since their L0 gate
    /// requires every contributing corner to have `predFlagL0 == 1`.
    #[test]
    fn unavailable_corner_predflag_blocks_dependent_candidates() {
        let mut corners = l0_corners(mv(10, 0), mv(40, 0), mv(0, 40), mv(50, 50));
        corners[2].pred_flag_l0 = false; // corner 2 unavailable for L0.
        let out = derive_constructed_affine_merge_candidates(
            32,
            32,
            &corners,
            ConstructedAffineFlags {
                sps_6param_affine_enabled_flag: true,
                slice_type_b: false,
            },
        );
        assert!(!out.available[0], "Const1 needs corner 2");
        assert!(!out.available[2], "Const3 needs corner 2");
        assert!(!out.available[3], "Const4 needs corner 2");
        assert!(!out.available[5], "Const6 needs corner 2");
        assert!(out.available[1], "Const2 only needs corners 0/1/3");
        assert!(out.available[4], "Const5 only needs corners 0/1");
    }

    /// All corners available BUT with different refIdx — the spec's
    /// "refIdxLXCorner[0] == refIdxLXCorner[1]" gate fires and every
    /// triple gets dropped.
    #[test]
    fn mismatched_refidx_blocks_every_triple() {
        let mut corners = l0_corners(mv(10, 0), mv(40, 0), mv(0, 40), mv(50, 50));
        corners[1].ref_idx_l0 = 1; // corner 1 references a different list-0 picture.
        let out = derive_constructed_affine_merge_candidates(
            32,
            32,
            &corners,
            ConstructedAffineFlags {
                sps_6param_affine_enabled_flag: true,
                slice_type_b: false,
            },
        );
        assert!(!out.available[0], "Const1: refIdx mismatch (0/1/2)");
        assert!(!out.available[1], "Const2: refIdx mismatch (0/1/3)");
        assert!(!out.available[3], "Const4: refIdx mismatch (1/2/3)");
        assert!(!out.available[4], "Const5: refIdx mismatch (0/1)");
        // Const3 uses corners {0, 2, 3} which all stay at refIdx 0 — it
        // should still materialise.
        assert!(out.available[2], "Const3 only touches corners 0/2/3");
        // Const6 uses corners {0, 2} which still match.
        assert!(out.available[5], "Const6 only touches corners 0/2");
    }

    /// Bi-pred corners (both L0 + L1 present, matching refIdx on each
    /// list) — Const1 should emit BOTH L0 + L1 CPMVs and carry the
    /// corner-0 bcwIdx.
    #[test]
    fn bipred_const1_emits_both_lists_with_corner0_bcw_idx() {
        let mk = |x_l0: i32, y_l0: i32, x_l1: i32, y_l1: i32, bcw: u8| AffineCpRecord {
            available: true,
            pred_flag_l0: true,
            pred_flag_l1: true,
            ref_idx_l0: 0,
            ref_idx_l1: 0,
            mv_l0: mv(x_l0, y_l0),
            mv_l1: mv(x_l1, y_l1),
            bcw_idx: bcw,
        };
        let corners = [
            mk(10, 0, 5, 0, 3), // corner 0 — bcwIdx 3
            mk(40, 0, 35, 0, 99),
            mk(0, 40, -5, 40, 99),
            mk(50, 50, 45, 50, 99),
        ];
        let out = derive_constructed_affine_merge_candidates(
            32,
            32,
            &corners,
            ConstructedAffineFlags {
                sps_6param_affine_enabled_flag: true,
                slice_type_b: true,
            },
        );
        assert!(out.available[0]);
        let c = &out.cands[0];
        assert!(c.pred_flag_l0 && c.pred_flag_l1);
        assert_eq!(c.cpmvs_l0.cpmvs[0], mv(10, 0));
        assert_eq!(c.cpmvs_l1.cpmvs[0], mv(5, 0));
        assert_eq!(c.bcw_idx, 3, "bcwIdx inherits from corner 0 in bipred");
    }

    /// Uni-pred (only L0 active) clears bcwIdx to 0 per the spec's
    /// "Otherwise, bcwIdxConstK is set equal to 0" bullet.
    #[test]
    fn unipred_const1_zeroes_bcw_idx() {
        let mut corners = l0_corners(mv(10, 0), mv(40, 0), mv(0, 40), mv(50, 50));
        corners[0].bcw_idx = 4;
        let out = derive_constructed_affine_merge_candidates(
            32,
            32,
            &corners,
            ConstructedAffineFlags {
                sps_6param_affine_enabled_flag: true,
                slice_type_b: false,
            },
        );
        assert!(out.available[0]);
        assert_eq!(out.cands[0].bcw_idx, 0, "uni-pred forces bcwIdx = 0");
    }

    /// Const4 inherits bcwIdx from corner 1 (NOT corner 0) when bipred.
    #[test]
    fn bipred_const4_inherits_bcw_from_corner1() {
        let mk = |bcw: u8| AffineCpRecord {
            available: true,
            pred_flag_l0: true,
            pred_flag_l1: true,
            ref_idx_l0: 0,
            ref_idx_l1: 0,
            mv_l0: mv(10, 0),
            mv_l1: mv(5, 0),
            bcw_idx: bcw,
        };
        let corners = [mk(99), mk(2), mk(99), mk(99)];
        let out = derive_constructed_affine_merge_candidates(
            32,
            32,
            &corners,
            ConstructedAffineFlags {
                sps_6param_affine_enabled_flag: true,
                slice_type_b: true,
            },
        );
        assert!(out.available[3]);
        assert_eq!(out.cands[3].bcw_idx, 2);
    }

    /// `count()` reflects the total number of `availableFlagConstK ==
    /// TRUE` slots. With every corner available + 6-param SPS flag + a
    /// 6-param-permitted CU geometry, all six candidates should appear.
    #[test]
    fn all_six_candidates_when_every_corner_available_and_consistent() {
        let corners = l0_corners(mv(10, 0), mv(40, 0), mv(0, 40), mv(50, 50));
        let out = derive_constructed_affine_merge_candidates(
            32,
            32,
            &corners,
            ConstructedAffineFlags {
                sps_6param_affine_enabled_flag: true,
                slice_type_b: false,
            },
        );
        assert_eq!(out.count(), 6);
    }

    /// Integration: a derived inherited CPMV record fed into the
    /// existing `derive_subblock_mvs` produces a valid sub-block MV
    /// grid. Confirms the new module's output is shape-compatible with
    /// the round-65 pipeline.
    #[test]
    fn inherited_cpmvs_drive_existing_subblock_mv_derivation() {
        let geom = InheritedAffineGeom {
            xcb: 16,
            ycb: 0,
            cb_width: 16,
            cb_height: 16,
            xnb: 0,
            ynb: 0,
            nb_w: 16,
            nb_h: 16,
        };
        let neighbour_cpmvs = AffineCpmvs::new_4param(mv(0, 0), mv(256, 0));
        let inherited = derive_inherited_affine_cpmvs(
            geom,
            NeighbourCpmvSource::SameOrLeftCtu {
                cpmvs: neighbour_cpmvs,
            },
            2,
        )
        .expect("derive ok");
        let grid =
            crate::affine::derive_subblock_mvs(geom.cb_width, geom.cb_height, &inherited, false)
                .expect("sub-block grid");
        assert_eq!(grid.num_sb_x, 4);
        assert_eq!(grid.num_sb_y, 4);
        assert_eq!(grid.mvs.len(), 16);
        // Top-left sub-block centre is at (xPosCb = 2, yPosCb = 2);
        // dHorX = (512 - 256) << 3 = 2048; first sub-block MV should be
        // close to CP0_new = (256, 0).
        let mv00 = grid.mv_at(0, 0);
        assert!(
            mv00.x >= 256 && mv00.x < 350,
            "first sub-block x = {}",
            mv00.x
        );
    }
}
