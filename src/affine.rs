//! Affine sub-block motion-compensation scaffold — round-65 §8.5.5.9
//! implementation.
//!
//! VVC supports an affine inter-prediction mode in which a coding unit
//! is described not by a single translational motion vector but by a
//! 4-parameter (2 control points) or 6-parameter (3 control points)
//! affine model. The decoder splits the CU into 4×4 luma sub-blocks
//! and derives one MV per sub-block from the control point motion
//! vectors (CPMVs) using the §8.5.5.9 sub-block MV array derivation;
//! each sub-block then runs through the existing §8.5.6.3 fractional
//! sample interpolation, using the affine-mode luma filter family of
//! Tables 30 / 31 / 32 in place of the regular Table 27 family.
//!
//! ## What this module ships in round 65
//!
//! 1. [`MotionModel`] — the spec's `MotionModelIdc` enum (§7.4.10.5 Table
//!    15): translational / 4-parameter affine / 6-parameter affine.
//! 2. [`AffineCpmvs`] — packed CPMV record carrying 2 or 3 motion
//!    vectors in 1/16-pel units, mirroring the spec's `cpMvLX[ cpIdx ]`
//!    array (cpIdx ∈ {0, 1} for 4-param, {0, 1, 2} for 6-param).
//! 3. [`derive_subblock_mvs`] — §8.5.5.9 eqs. 847 – 875 sub-block MV
//!    derivation. Given `(cbWidth, cbHeight)`, the CPMV set, and the
//!    spec-derived `(numSbX, numSbY) = (cbWidth >> 2, cbHeight >> 2)`,
//!    produces a per-sub-block MV grid clipped to `Clip3(-2^17,
//!    2^17 - 1, ·)` (eqs. 874 / 875).
//! 4. [`fallback_mode_triggered`] — §8.5.5.9 eqs. 858 – 867 + the
//!    bxWX/bxHX threshold check (255 / 225 / 165). Returns the spec's
//!    `fallbackModeTriggered` boolean; when `true` the per-sub-block
//!    MV becomes a single CU-centre MV (xPosCb = cbWidth >> 1,
//!    yPosCb = cbHeight >> 1) and the entire CU is MC'd with that one
//!    MV.
//! 5. [`AFFINE_LUMA_FILTER_SET_0`] / `_1` / `_2` — Tables 30 / 31 / 32
//!    luma 1/16-pel interpolation filter coefficients for affine motion
//!    mode. 16 fractional positions each, 8 taps wide, coefficients sum
//!    to 64 (table 30 has identity at p=0; table 31 / 32 use non-trivial
//!    p=0 rows per the spec listing).
//! 6. [`predict_luma_subblock_affine`] — separable 8-tap MC helper that
//!    consumes one sub-block from the reference plane and writes its
//!    prediction into the destination plane using a caller-selected
//!    affine luma filter table. The §8.5.6.3.2 separable
//!    horizontal-then-vertical pipeline with shift1 = 0 (BitDepth = 8)
//!    + shift2 = 6 + §8.5.6.6.2 uni-pred clamp `(v + 32) >> 6 → u8`,
//!    matching the existing `inter::predict_luma_block` behaviour for
//!    Table 27.
//! 7. [`predict_luma_block_affine`] — full-CU driver. Combines
//!    [`derive_subblock_mvs`] / [`fallback_mode_triggered`] /
//!    [`predict_luma_subblock_affine`] into a single `(cpMvs, cbWidth,
//!    cbHeight, src, dst, dst_x, dst_y) → ()` entry point that the
//!    CTU walker will call once affine-flagged CUs come online in
//!    later rounds.
//!
//! ## Out of scope (deferred to later rounds)
//!
//! * Affine merge candidate list derivation (§8.5.5.6 / §8.5.5.7 — the
//!   `availableConsFlagLX` constructed candidates).
//! * Affine AMVP candidate list (§8.5.5.8) — the explicit non-merge
//!   `mvp_l0_flag` + `mvd_coding × numCpMv` path.
//! * Affine sub-block chroma MC. The §8.5.5.9 `mvAvgLX` derivation
//!   (eqs. 876 – 879) lands here at the same time the CTU walker
//!   needs chroma sub-block MC; this round keeps luma-only.
//! * PROF — Prediction Refinement with Optical Flow (§8.5.5.10).
//!   `cbProfFlagLX` is parsed in §8.5.5.9 but its application requires
//!   gradient-based refinement out of scope until a later round.
//! * The §8.5.5.9 spec-prescribed sub-block boundary deblocking
//!   adjustments — those go in §8.8.3.4 and require the affine flag to
//!   propagate into the deblocker, which is a separate round.
//!
//! ## Spec reference
//!
//! ITU-T H.266 | ISO/IEC 23090-3 (V4, 01/2026):
//! * §7.4.10.5 — `MotionModelIdc[ x0 ][ y0 ]` semantics (Table 15).
//! * §8.5.5.9 — "Derivation process for motion vector arrays from
//!   affine control point motion vectors" (eqs. 847 – 887).
//! * §8.5.6.3 — luma sample interpolation (the separable filter
//!   pipeline; Tables 30 / 31 / 32 replace Table 27 when the affine
//!   filter family is selected).
//!
//! No third-party VVC decoder source was consulted; the
//! implementation is spec-only.

use oxideav_core::{Error, Result};

use crate::inter::MotionVector;
use crate::reconstruct::PicturePlane;

// ---------------------------------------------------------------------
// 1. MotionModelIdc + AffineCpmvs
// ---------------------------------------------------------------------

/// `MotionModelIdc[ x0 ][ y0 ]` per §7.4.10.5 Table 15 — names the
/// inter-prediction motion model for one CU.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MotionModel {
    /// `MotionModelIdc == 0` — translational motion (single MV per CU,
    /// the non-affine path that rounds 21 – 64 already cover).
    Translational,
    /// `MotionModelIdc == 1` — 4-parameter affine motion. CPMVs at
    /// `(cpIdx = 0, 1)` ⇒ `(top-left, top-right)`.
    Affine4Param,
    /// `MotionModelIdc == 2` — 6-parameter affine motion. CPMVs at
    /// `(cpIdx = 0, 1, 2)` ⇒ `(top-left, top-right, bottom-left)`.
    Affine6Param,
}

impl MotionModel {
    /// Number of control point motion vectors per the §8.5.5.5
    /// `numCpMv = motionModelIdc + 1` derivation.
    pub fn num_cp_mv(self) -> usize {
        match self {
            MotionModel::Translational => 1,
            MotionModel::Affine4Param => 2,
            MotionModel::Affine6Param => 3,
        }
    }

    /// Numeric `MotionModelIdc` value as carried by the spec arrays.
    pub fn idc(self) -> u8 {
        match self {
            MotionModel::Translational => 0,
            MotionModel::Affine4Param => 1,
            MotionModel::Affine6Param => 2,
        }
    }
}

/// Per-CU CPMV record for the affine modes. `cpmvs[0]` is `cpMvLX[0]`
/// (top-left), `cpmvs[1]` is `cpMvLX[1]` (top-right), and `cpmvs[2]`
/// — present only when [`Self::model`] is [`MotionModel::Affine6Param`]
/// — is `cpMvLX[2]` (bottom-left). All components are in 1/16-pel
/// units (the spec's stored CPMV precision after §8.5.2.14 rounding).
#[derive(Clone, Copy, Debug)]
pub struct AffineCpmvs {
    pub model: MotionModel,
    /// Up to three CPMVs. `cpmvs[2]` is read only for `Affine6Param`.
    pub cpmvs: [MotionVector; 3],
}

impl AffineCpmvs {
    /// 4-parameter constructor — `(top_left, top_right)`.
    pub fn new_4param(cp0: MotionVector, cp1: MotionVector) -> Self {
        Self {
            model: MotionModel::Affine4Param,
            cpmvs: [cp0, cp1, MotionVector::ZERO],
        }
    }

    /// 6-parameter constructor — `(top_left, top_right, bottom_left)`.
    pub fn new_6param(cp0: MotionVector, cp1: MotionVector, cp2: MotionVector) -> Self {
        Self {
            model: MotionModel::Affine6Param,
            cpmvs: [cp0, cp1, cp2],
        }
    }

    /// True iff every CPMV equals the same translational MV — i.e. the
    /// affine model degenerates to translational. Used by the §8.5.5.9
    /// `cbProfFlagLX` derivation (one of its `cbProfFlagLX = false`
    /// triggers).
    pub fn is_translational(&self) -> bool {
        let cp0 = self.cpmvs[0];
        match self.model {
            MotionModel::Translational => true,
            MotionModel::Affine4Param => self.cpmvs[1] == cp0,
            MotionModel::Affine6Param => self.cpmvs[1] == cp0 && self.cpmvs[2] == cp0,
        }
    }
}

// ---------------------------------------------------------------------
// 2. §8.5.5.9 sub-block MV derivation
// ---------------------------------------------------------------------

/// Number of luma sub-blocks per the §8.5.5.9 eqs. 668 / 669 (in
/// §8.5.5.1 — the spec aliases them as `numSbX = cbWidth >> 2`,
/// `numSbY = cbHeight >> 2`). Sub-blocks are always 4×4 luma samples
/// (the spec's minimum CU sub-block size).
pub const AFFINE_SB_SIZE: u32 = 4;

/// Per-sub-block MV grid produced by [`derive_subblock_mvs`].
#[derive(Clone, Debug)]
pub struct SubblockMvGrid {
    /// Number of sub-blocks horizontally (§8.5.5.1 eq. 668 / §8.5.5.9
    /// eq. 697 / eq. 890 alias `sbWidth = cbWidth / numSbX`).
    pub num_sb_x: u32,
    /// Number of sub-blocks vertically (eq. 669 / eq. 891).
    pub num_sb_y: u32,
    /// True when [`fallback_mode_triggered`] returned `true` — the
    /// per-sub-block MV is then a single CU-centre MV, but the grid
    /// still carries `num_sb_x × num_sb_y` copies so consumers can
    /// dispatch sub-block-by-sub-block uniformly.
    pub fallback: bool,
    /// Row-major `num_sb_x × num_sb_y` array of per-sub-block MVs in
    /// 1/16-pel units, clipped to `[-2^17, 2^17 - 1]` per eqs. 874 /
    /// 875.
    pub mvs: Vec<MotionVector>,
}

impl SubblockMvGrid {
    /// Read the MV for sub-block `(sb_idx_x, sb_idx_y)`.
    pub fn mv_at(&self, sb_idx_x: u32, sb_idx_y: u32) -> MotionVector {
        assert!(sb_idx_x < self.num_sb_x && sb_idx_y < self.num_sb_y);
        self.mvs[(sb_idx_y * self.num_sb_x + sb_idx_x) as usize]
    }
}

/// §8.5.5.9 eqs. 858 – 867 fallback-mode threshold check. Returns
/// `true` when the spec's bxWX / bxHX thresholds (225 for the
/// `4×4`-aligned all-axis bound under bi-pred, 165 for uni-pred axis
/// bounds) are exceeded for the given CPMV set + block size — the
/// per-sub-block MV derivation then collapses to a single CU-centre
/// MV per eqs. 868 / 869.
///
/// Inputs are in spec internal units:
/// * `cb_width` / `cb_height` — CU dimensions in luma samples (must be
///   multiples of 4).
/// * `cpmvs` — the §8.5.5.9 CPMV set.
/// * `bipred` — `true` iff both `predFlagL0 == 1` and `predFlagL1 == 1`
///   for the current CU.
pub fn fallback_mode_triggered(
    cb_width: u32,
    cb_height: u32,
    cpmvs: &AffineCpmvs,
    bipred: bool,
) -> bool {
    let (d_hor_x, d_ver_x, d_hor_y, d_ver_y) = compute_affine_partials(cb_width, cb_height, cpmvs);

    // eqs. 858 – 861
    let max_w4: i64 = (0i64)
        .max(4 * (2048 + d_hor_y as i64))
        .max(4 * d_hor_y as i64)
        .max(4 * (2048 + d_hor_y as i64) + 4 * d_hor_y as i64);
    let min_w4: i64 = (0i64)
        .min(4 * (2048 + d_hor_y as i64))
        .min(4 * d_hor_y as i64)
        .min(4 * (2048 + d_hor_y as i64) + 4 * d_hor_y as i64);
    let max_h4: i64 = (0i64)
        .max(4 * d_ver_y as i64)
        .max(4 * (2048 + d_ver_y as i64))
        .max(4 * d_ver_y as i64 + 4 * (2048 + d_ver_y as i64));
    let min_h4: i64 = (0i64)
        .min(4 * d_ver_y as i64)
        .min(4 * (2048 + d_ver_y as i64))
        .min(4 * d_ver_y as i64 + 4 * (2048 + d_ver_y as i64));

    // eqs. 862 / 863
    let bx_w_x4: i64 = ((max_w4 - min_w4) >> 11) + 9;
    let bx_h_x4: i64 = ((max_h4 - min_h4) >> 11) + 9;
    // eqs. 864 – 867
    let bx_w_h: i64 =
        (((0i64).max(4 * (2048 + d_hor_x as i64)) - (0i64).min(4 * (2048 + d_hor_x as i64))) >> 11)
            + 9;
    let bx_h_h: i64 = (((0i64).max(4 * d_ver_x as i64) - (0i64).min(4 * d_ver_x as i64)) >> 11) + 9;
    let bx_w_v: i64 = (((0i64).max(4 * d_hor_y as i64) - (0i64).min(4 * d_hor_y as i64)) >> 11) + 9;
    let bx_h_v: i64 =
        (((0i64).max(4 * (2048 + d_ver_y as i64)) - (0i64).min(4 * (2048 + d_ver_y as i64))) >> 11)
            + 9;

    // bi-pred branch: `bxWX4 * bxHX4 <= 225` clears fallback. Note: the
    // spec phrases the cleared-fallback case (`fallbackModeTriggered =
    // 0`); we return the negated value.
    if bipred {
        if bx_w_x4 * bx_h_x4 <= 225 {
            return false;
        }
    } else {
        // uni-pred branch — also gated by the per-axis 165 ceiling.
        if bx_w_h * bx_h_h <= 165 && bx_w_v * bx_h_v <= 165 {
            return false;
        }
    }
    true
}

/// §8.5.5.9 eqs. 850 / 851 / 852 / 853 / 854 / 855 / 856 / 857 derive
/// the four affine motion partials `(dHorX, dVerX, dHorY, dVerY)` from
/// the CPMV set. Returns them in spec-internal units (the same
/// 1/16-pel-scaled-by-`(1 << (7 - log2cbW))` precision the per-
/// sub-block MV formula consumes).
fn compute_affine_partials(
    cb_width: u32,
    cb_height: u32,
    cpmvs: &AffineCpmvs,
) -> (i32, i32, i32, i32) {
    let log2_cb_w = log2_u32(cb_width);
    let log2_cb_h = log2_u32(cb_height);
    let shift_w = 7 - log2_cb_w as i32;
    let shift_h = 7 - log2_cb_h as i32;
    let cp0_x = cpmvs.cpmvs[0].x;
    let cp0_y = cpmvs.cpmvs[0].y;
    let cp1_x = cpmvs.cpmvs[1].x;
    let cp1_y = cpmvs.cpmvs[1].y;

    let d_hor_x = (cp1_x - cp0_x) << shift_w; // eq. 852
    let d_ver_x = (cp1_y - cp0_y) << shift_w; // eq. 853

    let (d_hor_y, d_ver_y) = match cpmvs.model {
        MotionModel::Affine6Param => {
            let cp2_x = cpmvs.cpmvs[2].x;
            let cp2_y = cpmvs.cpmvs[2].y;
            (
                (cp2_x - cp0_x) << shift_h, // eq. 854
                (cp2_y - cp0_y) << shift_h, // eq. 855
            )
        }
        MotionModel::Affine4Param | MotionModel::Translational => {
            // eqs. 856 / 857 — the 4-parameter row constrains the model
            // to be a similarity transform: dHorY = -dVerX, dVerY = dHorX.
            (-d_ver_x, d_hor_x)
        }
    };
    (d_hor_x, d_ver_x, d_hor_y, d_ver_y)
}

/// §8.5.5.9 sub-block MV array derivation — produces a `numSbX × numSbY`
/// grid of per-4×4-sub-block MVs from a CU's CPMVs.
///
/// Inputs:
/// * `cb_width` / `cb_height` — CU dimensions in luma samples. Must be
///   multiples of 4 (the spec only invokes affine on CUs with `cbW *
///   cbH >= 256` and both `cbW >= 8`, `cbH >= 8`).
/// * `cpmvs` — CPMV record per [`AffineCpmvs`]. `Translational`
///   collapses to a single MV (degenerate path — included for
///   completeness; production paths should call regular non-affine
///   MC).
/// * `bipred` — feeds [`fallback_mode_triggered`].
///
/// Output: a [`SubblockMvGrid`] with the per-sub-block 1/16-pel MVs.
pub fn derive_subblock_mvs(
    cb_width: u32,
    cb_height: u32,
    cpmvs: &AffineCpmvs,
    bipred: bool,
) -> Result<SubblockMvGrid> {
    if cb_width < 8 || cb_height < 8 {
        return Err(Error::invalid(format!(
            "h266 affine: CU {cb_width}x{cb_height} below the 8×8 affine floor"
        )));
    }
    if cb_width % AFFINE_SB_SIZE != 0 || cb_height % AFFINE_SB_SIZE != 0 {
        return Err(Error::invalid(format!(
            "h266 affine: CU {cb_width}x{cb_height} not aligned to {AFFINE_SB_SIZE}x{AFFINE_SB_SIZE} sub-block grid"
        )));
    }
    let num_sb_x = cb_width >> 2; // eq. 668
    let num_sb_y = cb_height >> 2; // eq. 669
    let fallback = match cpmvs.model {
        // Translational degenerates: no fallback ever needed; the
        // single CPMV is the MV for every sub-block.
        MotionModel::Translational => false,
        _ => fallback_mode_triggered(cb_width, cb_height, cpmvs, bipred),
    };

    // Translational shortcut: every sub-block carries cpmvs[0].
    if matches!(cpmvs.model, MotionModel::Translational) {
        let mv = cpmvs.cpmvs[0];
        return Ok(SubblockMvGrid {
            num_sb_x,
            num_sb_y,
            fallback: false,
            mvs: vec![mv; (num_sb_x * num_sb_y) as usize],
        });
    }

    // Affine partials (eqs. 850 – 857).
    let (d_hor_x, d_ver_x, d_hor_y, d_ver_y) = compute_affine_partials(cb_width, cb_height, cpmvs);
    let mv_scale_hor = cpmvs.cpmvs[0].x << 7; // eq. 850
    let mv_scale_ver = cpmvs.cpmvs[0].y << 7; // eq. 851

    let mut mvs = Vec::with_capacity((num_sb_x * num_sb_y) as usize);
    for sb_idx_y in 0..num_sb_y {
        for sb_idx_x in 0..num_sb_x {
            let (x_pos_cb, y_pos_cb) = if fallback {
                // eqs. 868 / 869 — CU-centre.
                ((cb_width >> 1) as i32, (cb_height >> 1) as i32)
            } else {
                // eqs. 870 / 871 — sub-block centre, expressed in luma
                // sample units relative to the CU origin.
                (2 + ((sb_idx_x as i32) << 2), 2 + ((sb_idx_y as i32) << 2))
            };
            // eqs. 872 / 873 — luma sub-block MV in spec internal
            // precision (before the `>> 7` rounding pass).
            let raw_x = (mv_scale_hor as i64)
                + (d_hor_x as i64) * (x_pos_cb as i64)
                + (d_hor_y as i64) * (y_pos_cb as i64);
            let raw_y = (mv_scale_ver as i64)
                + (d_ver_x as i64) * (x_pos_cb as i64)
                + (d_ver_y as i64) * (y_pos_cb as i64);
            // §8.5.2.14 rounding with rightShift = 7, leftShift = 0.
            let rounded_x = round_mv_component(raw_x, 7);
            let rounded_y = round_mv_component(raw_y, 7);
            // eqs. 874 / 875 — clip to `[-2^17, 2^17 - 1]`.
            let clipped_x = rounded_x.clamp(-(1i64 << 17), (1i64 << 17) - 1) as i32;
            let clipped_y = rounded_y.clamp(-(1i64 << 17), (1i64 << 17) - 1) as i32;
            mvs.push(MotionVector {
                x: clipped_x,
                y: clipped_y,
            });
        }
    }
    Ok(SubblockMvGrid {
        num_sb_x,
        num_sb_y,
        fallback,
        mvs,
    })
}

/// §8.5.2.14 rounding helper (one component): the spec's "rightShift
/// signed magnitude" semantic — for `mvX` and `rightShift > 0` returns
/// `(mvX + offset - (mvX >= 0 ? 1 : 0)) >> rightShift` with
/// `offset = 1 << (rightShift - 1)`. For `rightShift == 0` it's a
/// no-op.
fn round_mv_component(mv: i64, right_shift: i32) -> i64 {
    if right_shift == 0 {
        return mv;
    }
    let offset = 1i64 << (right_shift - 1);
    let plus_one_when_nonneg = if mv >= 0 { 1 } else { 0 };
    (mv + offset - plus_one_when_nonneg) >> right_shift
}

/// Pure log2 helper for power-of-two CU dimensions (the only sizes
/// VVC permits — 4, 8, 16, 32, 64, 128). Returns `0` for unsupported
/// non-power-of-two inputs, which the caller surfaces as
/// `Error::invalid` before reaching here.
fn log2_u32(v: u32) -> u32 {
    assert!(v > 0 && v.is_power_of_two());
    v.trailing_zeros()
}

// ---------------------------------------------------------------------
// 3. Tables 30 / 31 / 32 — affine luma interpolation filters
// ---------------------------------------------------------------------

/// Table 30 — luma 1/16-pel interpolation filter for affine motion
/// mode (the spec's "default" affine filter set; numCpMv-driven
/// fallback uses Tables 31 / 32 for high-frequency CUs).
///
/// Row p ∈ [0, 15] is the filter for fractional position p. The spec
/// lists rows 1 – 15 explicitly and leaves row 0 implicit as the
/// integer-pel filter `{0, 1, -3, 63, 4, -2, 1, 0}` — same as Table
/// 27's row 1. We populate the actual row-0 with the identity
/// `{0, 0, 0, 64, 0, 0, 0, 0}` so the §8.5.6.3 fast path (`xFrac == 0
/// && yFrac == 0`) collapses to a pure-integer copy by skipping the
/// filter; consumers must therefore special-case the integer-pel
/// position the way `predict_luma_block` does.
pub const AFFINE_LUMA_FILTER_SET_0: [[i32; 8]; 16] = [
    [0, 0, 0, 64, 0, 0, 0, 0],   // p == 0 sentinel
    [0, 1, -3, 63, 4, -2, 1, 0], // p == 1
    [0, 1, -5, 62, 8, -3, 1, 0],
    [0, 2, -8, 60, 13, -4, 1, 0],
    [0, 3, -10, 58, 17, -5, 1, 0],
    [0, 3, -11, 52, 26, -8, 2, 0],
    [0, 2, -9, 47, 31, -10, 3, 0],
    [0, 3, -11, 45, 34, -10, 3, 0],
    [0, 3, -11, 40, 40, -11, 3, 0], // p == 8 (half-pel)
    [0, 3, -10, 34, 45, -11, 3, 0],
    [0, 3, -10, 31, 47, -9, 2, 0],
    [0, 2, -8, 26, 52, -11, 3, 0],
    [0, 1, -5, 17, 58, -10, 3, 0],
    [0, 1, -4, 13, 60, -8, 2, 0],
    [0, 1, -3, 8, 62, -5, 1, 0],
    [0, 1, -2, 4, 63, -3, 1, 0],
];

/// Table 31 — luma 1/16-pel interpolation filter for affine motion
/// mode (set 1; one of the affine alternative filter families). The
/// spec defines all 16 rows (row 0 is **not** the identity — affine
/// filter sets 31 / 32 have non-trivial p == 0 rows used by the
/// §8.5.6.3 affine integer-position path).
pub const AFFINE_LUMA_FILTER_SET_1: [[i32; 8]; 16] = [
    [0, -6, 17, 42, 17, -5, -1, 0], // p == 0
    [0, -5, 15, 41, 19, -5, -1, 0],
    [0, -5, 13, 40, 21, -4, -1, 0],
    [0, -5, 11, 39, 24, -4, -1, 0],
    [0, -5, 9, 38, 26, -3, -1, 0],
    [0, -5, 7, 38, 28, -2, -2, 0],
    [0, -4, 5, 36, 30, -1, -2, 0],
    [0, -3, 3, 35, 32, 0, -3, 0],
    [0, -3, 2, 33, 33, 2, -3, 0],
    [0, -3, 0, 32, 35, 3, -3, 0],
    [0, -2, -1, 30, 36, 5, -4, 0],
    [0, -2, -2, 28, 38, 7, -5, 0],
    [0, -1, -3, 26, 38, 9, -5, 0],
    [0, -1, -4, 24, 39, 11, -5, 0],
    [0, -1, -4, 21, 40, 13, -5, 0],
    [0, -1, -5, 19, 41, 15, -5, 0],
];

/// Table 32 — luma 1/16-pel interpolation filter for affine motion
/// mode (set 2; second affine alternative filter family).
pub const AFFINE_LUMA_FILTER_SET_2: [[i32; 8]; 16] = [
    [0, -2, 20, 28, 20, 2, -4, 0], // p == 0
    [0, -4, 19, 29, 21, 5, -6, 0],
    [0, -5, 18, 29, 22, 6, -6, 0],
    [0, -5, 16, 29, 23, 7, -6, 0],
    [0, -5, 16, 28, 24, 7, -6, 0],
    [0, -5, 14, 28, 25, 8, -6, 0],
    [0, -6, 14, 27, 26, 9, -6, 0],
    [0, -4, 12, 28, 25, 10, -7, 0],
    [0, -6, 11, 27, 27, 11, -6, 0], // p == 8
    [0, -7, 10, 25, 28, 12, -4, 0],
    [0, -6, 9, 26, 27, 14, -6, 0],
    [0, -6, 8, 25, 28, 14, -5, 0],
    [0, -6, 7, 24, 28, 16, -5, 0],
    [0, -6, 7, 23, 29, 16, -5, 0],
    [0, -6, 6, 22, 29, 18, -5, 0],
    [0, -6, 5, 21, 29, 19, -4, 0],
];

/// Affine-mode luma filter sets per the spec's Tables 30 / 31 / 32.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AffineLumaFilterSet {
    /// Table 30 (the default affine filter — used unless RPR / `1.5×`
    /// / `2×` scaling triggers Table 31 or 32 per §8.5.6.3.2's
    /// `scalingRatio` cascade).
    Set0,
    /// Table 31 (selected by §8.5.6.3.2 when `scalingRatio` >
    /// 20 480 for affine mode).
    Set1,
    /// Table 32 (selected by §8.5.6.3.2 when `scalingRatio` >
    /// 28 672 for affine mode).
    Set2,
}

impl AffineLumaFilterSet {
    /// Return the 16-row coefficient table for this filter set.
    pub fn table(self) -> &'static [[i32; 8]; 16] {
        match self {
            AffineLumaFilterSet::Set0 => &AFFINE_LUMA_FILTER_SET_0,
            AffineLumaFilterSet::Set1 => &AFFINE_LUMA_FILTER_SET_1,
            AffineLumaFilterSet::Set2 => &AFFINE_LUMA_FILTER_SET_2,
        }
    }
}

// ---------------------------------------------------------------------
// 4. Sub-block luma MC using a selected affine filter set
// ---------------------------------------------------------------------

/// §8.5.6.3.2 separable 8-tap luma interpolation for one affine
/// sub-block. Reads `sb_w × sb_h` samples (typically 4×4) from `src`
/// at `(dst_x + mv.x >> 4, dst_y + mv.y >> 4)` and writes the
/// 8-bit-clipped prediction into `dst`.
///
/// `filter_set` selects between Tables 30 / 31 / 32. The pipeline is
/// the same as `inter::predict_luma_block` (horizontal-then-vertical
/// separable, picture-edge clamp, `shift1 = 0` for BitDepth = 8,
/// `shift2 = 6`, §8.5.6.6.2 `(v + 32) >> 6 → u8` uni-pred clamp).
#[allow(clippy::too_many_arguments)]
pub fn predict_luma_subblock_affine(
    dst: &mut PicturePlane,
    dst_x: u32,
    dst_y: u32,
    sb_w: u32,
    sb_h: u32,
    src: &PicturePlane,
    mv: MotionVector,
    filter_set: AffineLumaFilterSet,
) -> Result<()> {
    if dst_x as usize + sb_w as usize > dst.width || dst_y as usize + sb_h as usize > dst.height {
        return Err(Error::invalid(format!(
            "h266 affine luma MC: destination sub-block ({dst_x},{dst_y}) {sb_w}x{sb_h} \
             out of plane bounds {}x{}",
            dst.width, dst.height
        )));
    }
    let table = filter_set.table();
    let x_int_base = dst_x as i32 + (mv.x >> 4);
    let y_int_base = dst_y as i32 + (mv.y >> 4);
    let x_frac = (mv.x & 15) as usize;
    let y_frac = (mv.y & 15) as usize;

    let pic_w = src.width as i32;
    let pic_h = src.height as i32;
    let d_stride = dst.stride;

    // Integer-pel fast path mirrors the §8.5.6.3 eq. 932 shortcut.
    if x_frac == 0 && y_frac == 0 && filter_set == AffineLumaFilterSet::Set0 {
        for r in 0..sb_h as i32 {
            let yi = (y_int_base + r).clamp(0, pic_h - 1) as usize;
            for c in 0..sb_w as i32 {
                let xi = (x_int_base + c).clamp(0, pic_w - 1) as usize;
                dst.samples
                    [(dst_y as usize + r as usize) * d_stride + dst_x as usize + c as usize] =
                    src.samples[yi * src.stride + xi];
            }
        }
        return Ok(());
    }

    if y_frac == 0 {
        // Horizontal-only filter.
        for r in 0..sb_h as i32 {
            let yi = (y_int_base + r).clamp(0, pic_h - 1) as usize;
            for c in 0..sb_w as i32 {
                let intermediate = h_tap(table, src, x_int_base + c, yi, x_frac);
                dst.samples
                    [(dst_y as usize + r as usize) * d_stride + dst_x as usize + c as usize] =
                    pb_clip_8bit(intermediate);
            }
        }
        return Ok(());
    }

    if x_frac == 0 {
        // Vertical-only filter.
        for c in 0..sb_w as i32 {
            let xi = (x_int_base + c).clamp(0, pic_w - 1) as usize;
            for r in 0..sb_h as i32 {
                let intermediate = v_only_tap(table, src, xi, y_int_base + r, y_frac);
                dst.samples
                    [(dst_y as usize + r as usize) * d_stride + dst_x as usize + c as usize] =
                    pb_clip_8bit(intermediate);
            }
        }
        return Ok(());
    }

    // 2D case — `(sb_h + 7) × sb_w` intermediate.
    let inter_h = sb_h as usize + 7;
    let mut intermediate = vec![0i32; inter_h * sb_w as usize];
    for r in 0..inter_h as i32 {
        let yi = (y_int_base - 3 + r).clamp(0, pic_h - 1) as usize;
        for c in 0..sb_w as i32 {
            intermediate[r as usize * sb_w as usize + c as usize] =
                h_tap(table, src, x_int_base + c, yi, x_frac);
        }
    }
    let mut col = [0i32; 8];
    for r in 0..sb_h as i32 {
        for c in 0..sb_w as i32 {
            for i in 0..8 {
                col[i] = intermediate[(r as usize + i) * sb_w as usize + c as usize];
            }
            let v = v_tap(table, &col, y_frac);
            dst.samples[(dst_y as usize + r as usize) * d_stride + dst_x as usize + c as usize] =
                pb_clip_8bit(v);
        }
    }
    Ok(())
}

/// Horizontal 8-tap pass against an arbitrary 16-row coefficient
/// table (Tables 27 / 30 / 31 / 32 all share the 8-tap geometry).
fn h_tap(
    table: &[[i32; 8]; 16],
    plane: &PicturePlane,
    x_int: i32,
    y_clamped: usize,
    x_frac: usize,
) -> i32 {
    let coeffs = &table[x_frac];
    let pic_w = plane.width as i32;
    let mut acc = 0i32;
    let row_base = y_clamped * plane.stride;
    for (i, c) in coeffs.iter().enumerate() {
        let xi = (x_int + (i as i32) - 3).clamp(0, pic_w - 1) as usize;
        acc += c * (plane.samples[row_base + xi] as i32);
    }
    acc // shift1 = 0 for BitDepth = 8
}

/// Vertical 8-tap pass over an intermediate column.
fn v_tap(table: &[[i32; 8]; 16], temp: &[i32; 8], y_frac: usize) -> i32 {
    let coeffs = &table[y_frac];
    let mut acc = 0i32;
    for i in 0..8 {
        acc += coeffs[i] * temp[i];
    }
    acc >> 6 // shift2 = 6
}

/// Vertical-only pass (used when `xFracL == 0`).
fn v_only_tap(
    table: &[[i32; 8]; 16],
    plane: &PicturePlane,
    x_clamped: usize,
    y_int: i32,
    y_frac: usize,
) -> i32 {
    let coeffs = &table[y_frac];
    let pic_h = plane.height as i32;
    let mut acc = 0i32;
    for i in 0..8 {
        let yi = (y_int + (i as i32) - 3).clamp(0, pic_h - 1) as usize;
        acc += coeffs[i] * (plane.samples[yi * plane.stride + x_clamped] as i32);
    }
    acc
}

/// §8.5.6.6.2 default uni-pred clamp at BitDepth = 8.
#[inline]
fn pb_clip_8bit(intermediate: i32) -> u8 {
    let v = (intermediate + 32) >> 6;
    v.clamp(0, 255) as u8
}

// ---------------------------------------------------------------------
// 5. Full-CU affine luma MC driver
// ---------------------------------------------------------------------

/// Drive [`derive_subblock_mvs`] + [`predict_luma_subblock_affine`]
/// across one CU. Walks the `numSbX × numSbY` 4×4 sub-block grid and
/// dispatches each through the chosen affine luma filter.
///
/// This is the entry point a future CTU walker will call for affine
/// inter CUs. Today it is exercised end-to-end by the round-65
/// integration tests against a synthetic affine-transformed fixture.
#[allow(clippy::too_many_arguments)]
pub fn predict_luma_block_affine(
    dst: &mut PicturePlane,
    dst_x: u32,
    dst_y: u32,
    cb_width: u32,
    cb_height: u32,
    src: &PicturePlane,
    cpmvs: &AffineCpmvs,
    filter_set: AffineLumaFilterSet,
) -> Result<()> {
    if dst_x as usize + cb_width as usize > dst.width
        || dst_y as usize + cb_height as usize > dst.height
    {
        return Err(Error::invalid(format!(
            "h266 affine luma MC: CU ({dst_x},{dst_y}) {cb_width}x{cb_height} \
             out of plane bounds {}x{}",
            dst.width, dst.height
        )));
    }
    let grid = derive_subblock_mvs(cb_width, cb_height, cpmvs, /*bipred*/ false)?;
    let sb_w = cb_width / grid.num_sb_x; // typically 4
    let sb_h = cb_height / grid.num_sb_y;
    for sb_iy in 0..grid.num_sb_y {
        for sb_ix in 0..grid.num_sb_x {
            let mv = grid.mv_at(sb_ix, sb_iy);
            let sb_x = dst_x + sb_ix * sb_w;
            let sb_y = dst_y + sb_iy * sb_h;
            predict_luma_subblock_affine(dst, sb_x, sb_y, sb_w, sb_h, src, mv, filter_set)?;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// All three affine filter tables must have coefficients summing
    /// to 64 per the §8.5.6.3 separable-filter normalisation (the
    /// vertical-pass shift is `>> 6 = >> log2(64)`, so the implicit
    /// DC gain must be 1.0).
    #[test]
    fn affine_filter_tables_sum_to_64() {
        for (name, table) in [
            ("Table 30", &AFFINE_LUMA_FILTER_SET_0),
            ("Table 31", &AFFINE_LUMA_FILTER_SET_1),
            ("Table 32", &AFFINE_LUMA_FILTER_SET_2),
        ] {
            for (p, row) in table.iter().enumerate() {
                let s: i32 = row.iter().sum();
                // Set 0 row 0 is our integer-pel identity sentinel
                // (sum 64). Sets 1 / 2 row 0 are real spec filter rows
                // (sum 64).
                assert_eq!(s, 64, "{name} row {p} {row:?} should sum to 64, got {s}",);
            }
        }
    }

    #[test]
    fn motion_model_num_cp_mv_matches_spec() {
        assert_eq!(MotionModel::Translational.num_cp_mv(), 1);
        assert_eq!(MotionModel::Affine4Param.num_cp_mv(), 2);
        assert_eq!(MotionModel::Affine6Param.num_cp_mv(), 3);
    }

    #[test]
    fn motion_model_idc_matches_table_15() {
        assert_eq!(MotionModel::Translational.idc(), 0);
        assert_eq!(MotionModel::Affine4Param.idc(), 1);
        assert_eq!(MotionModel::Affine6Param.idc(), 2);
    }

    /// Identity CPMV set (all CPMVs equal one translation) must
    /// collapse to a uniform per-sub-block MV grid — every entry equal
    /// to the CPMV. This is the basic sanity check that §8.5.5.9 eqs.
    /// 850 / 851 / 852 / 853 produce zero partials when CPMVs match.
    #[test]
    fn derive_subblock_mvs_identity_collapses_to_translational() {
        let cp = MotionVector { x: 16, y: -32 };
        let cpmvs = AffineCpmvs::new_4param(cp, cp);
        let grid = derive_subblock_mvs(16, 16, &cpmvs, false).expect("4-param identity");
        assert_eq!(grid.num_sb_x, 4);
        assert_eq!(grid.num_sb_y, 4);
        assert!(!grid.fallback);
        for mv in &grid.mvs {
            assert_eq!(*mv, cp, "every sub-block MV must equal the CPMV");
        }
    }

    #[test]
    fn derive_subblock_mvs_6param_identity_collapses_to_translational() {
        let cp = MotionVector { x: 8, y: 8 };
        let cpmvs = AffineCpmvs::new_6param(cp, cp, cp);
        let grid = derive_subblock_mvs(16, 16, &cpmvs, false).expect("6-param identity");
        for mv in &grid.mvs {
            assert_eq!(*mv, cp);
        }
    }

    /// 4-parameter affine: a pure horizontal shear `(cp1 - cp0).x ≠ 0`
    /// must produce a monotonically increasing per-sub-block MV.x
    /// across columns and a constant value down each column. With
    /// cp0 = (0, 0), cp1 = (W * 16, 0) (one full sample horizontal
    /// shift per CU width — i.e. the right edge moves +1 luma sample
    /// relative to the left edge), the per-sub-block MV.x at sub-block
    /// column `ix` should be roughly `((ix*4 + 2) * 16 / W)` 1/16-pel
    /// units (the spec's `posOffsetX = 0` simple linear interpolation).
    #[test]
    fn derive_subblock_mvs_4param_horizontal_shear_monotone() {
        let w = 16u32;
        let h = 16u32;
        // CPMVs in 1/16-pel units. cp0 = (0, 0); cp1 = (W, 0) i.e.
        // 1 luma sample horizontal shift across the CU width. (1 luma
        // sample = 16 in 1/16-pel units.)
        let cp0 = MotionVector { x: 0, y: 0 };
        let cp1 = MotionVector { x: 16, y: 0 };
        let cpmvs = AffineCpmvs::new_4param(cp0, cp1);
        let grid = derive_subblock_mvs(w, h, &cpmvs, false).expect("shear");
        assert!(!grid.fallback);
        // For each row, MV.x must be non-decreasing across columns.
        for iy in 0..grid.num_sb_y {
            let mut prev_x = i32::MIN;
            for ix in 0..grid.num_sb_x {
                let mv = grid.mv_at(ix, iy);
                assert!(
                    mv.x >= prev_x,
                    "MV.x should be non-decreasing across columns at row {iy}: got {} after {prev_x}",
                    mv.x,
                );
                prev_x = mv.x;
            }
        }
        // Also MV.x should be constant down each column.
        for ix in 0..grid.num_sb_x {
            let col_mv = grid.mv_at(ix, 0).x;
            for iy in 0..grid.num_sb_y {
                assert_eq!(
                    grid.mv_at(ix, iy).x,
                    col_mv,
                    "column {ix}: MV.x should be constant down rows for a pure horizontal shear"
                );
            }
        }
    }

    /// 6-parameter affine: a pure vertical shear `(cp2 - cp0).y ≠ 0`
    /// must produce a monotonically increasing per-sub-block MV.y
    /// down columns and a constant value across each row.
    #[test]
    fn derive_subblock_mvs_6param_vertical_shear_monotone() {
        let w = 16u32;
        let h = 16u32;
        let cp0 = MotionVector { x: 0, y: 0 };
        let cp1 = MotionVector { x: 0, y: 0 };
        let cp2 = MotionVector { x: 0, y: 16 };
        let cpmvs = AffineCpmvs::new_6param(cp0, cp1, cp2);
        let grid = derive_subblock_mvs(w, h, &cpmvs, false).expect("v-shear");
        assert!(!grid.fallback);
        // MV.y constant across each row.
        for iy in 0..grid.num_sb_y {
            let row_mv = grid.mv_at(0, iy).y;
            for ix in 0..grid.num_sb_x {
                assert_eq!(
                    grid.mv_at(ix, iy).y,
                    row_mv,
                    "row {iy}: MV.y should be constant across columns for a pure vertical shear"
                );
            }
        }
        // MV.y non-decreasing down rows.
        for ix in 0..grid.num_sb_x {
            let mut prev_y = i32::MIN;
            for iy in 0..grid.num_sb_y {
                let mv = grid.mv_at(ix, iy);
                assert!(mv.y >= prev_y);
                prev_y = mv.y;
            }
        }
    }

    /// Sub-block size below the §8.5.5.9 8x8 floor must error.
    #[test]
    fn derive_subblock_mvs_rejects_below_8x8() {
        let cp = MotionVector { x: 0, y: 0 };
        let cpmvs = AffineCpmvs::new_4param(cp, cp);
        let r = derive_subblock_mvs(4, 16, &cpmvs, false);
        assert!(r.is_err());
        let r = derive_subblock_mvs(16, 4, &cpmvs, false);
        assert!(r.is_err());
    }

    /// 4-parameter affine with a small CPMV difference (within the
    /// fallback threshold ≤ 225) must NOT trigger fallback under
    /// bi-pred.
    #[test]
    fn fallback_mode_not_triggered_for_small_cpmv_delta() {
        let cp0 = MotionVector { x: 0, y: 0 };
        let cp1 = MotionVector { x: 8, y: 0 }; // half-pel shift
        let cpmvs = AffineCpmvs::new_4param(cp0, cp1);
        assert!(!fallback_mode_triggered(32, 32, &cpmvs, true));
    }

    /// Filter set selector returns the right table reference.
    #[test]
    fn affine_filter_set_table_dispatch() {
        let p_eq_8: [i32; 8] = AffineLumaFilterSet::Set0.table()[8];
        assert_eq!(p_eq_8, [0, 3, -11, 40, 40, -11, 3, 0]);
        let p_eq_8: [i32; 8] = AffineLumaFilterSet::Set1.table()[8];
        assert_eq!(p_eq_8, [0, -3, 2, 33, 33, 2, -3, 0]);
        let p_eq_8: [i32; 8] = AffineLumaFilterSet::Set2.table()[8];
        assert_eq!(p_eq_8, [0, -6, 11, 27, 27, 11, -6, 0]);
    }

    /// Integer-pel sub-block MC with Set0 + (0, 0) MV must copy the
    /// reference block exactly. This is the §8.5.6.3 fast path
    /// degenerate case.
    #[test]
    fn predict_luma_subblock_affine_integer_pel_set0_copies() {
        let mut src = PicturePlane::filled(16, 16, 0);
        for y in 0..16 {
            for x in 0..16 {
                src.samples[y * 16 + x] = ((y * 17 + x * 3) % 251) as u8;
            }
        }
        let mut dst = PicturePlane::filled(16, 16, 99);
        predict_luma_subblock_affine(
            &mut dst,
            4,
            4,
            4,
            4,
            &src,
            MotionVector::from_int_pel(0, 0),
            AffineLumaFilterSet::Set0,
        )
        .expect("int-pel MC");
        for r in 0..4usize {
            for c in 0..4usize {
                assert_eq!(
                    dst.samples[(4 + r) * 16 + (4 + c)],
                    src.samples[(4 + r) * 16 + (4 + c)],
                    "int-pel copy mismatch at ({r},{c})",
                );
            }
        }
    }

    /// Translational `predict_luma_block_affine` (CPMV0 == CPMV1) at
    /// integer-pel MV must copy the reference CU exactly.
    #[test]
    fn predict_luma_block_affine_translational_copies() {
        let mut src = PicturePlane::filled(32, 32, 0);
        for y in 0..32 {
            for x in 0..32 {
                src.samples[y * 32 + x] = ((y * 5 + x * 7) % 251) as u8;
            }
        }
        let mut dst = PicturePlane::filled(32, 32, 0);
        let cp = MotionVector::from_int_pel(0, 0);
        let cpmvs = AffineCpmvs::new_4param(cp, cp);
        predict_luma_block_affine(
            &mut dst,
            8,
            8,
            16,
            16,
            &src,
            &cpmvs,
            AffineLumaFilterSet::Set0,
        )
        .expect("translational MC");
        for r in 0..16usize {
            for c in 0..16usize {
                assert_eq!(
                    dst.samples[(8 + r) * 32 + (8 + c)],
                    src.samples[(8 + r) * 32 + (8 + c)],
                );
            }
        }
    }

    /// Per §8.5.5.9 the sub-block MV is sampled at the sub-block
    /// CENTRE `(xPosCb, yPosCb) = (2 + 4*sbIdxX, 2 + 4*sbIdxY)`
    /// (eqs. 870 / 871) and a 4-parameter affine with cp1.x != cp0.x
    /// inherits the similarity constraint `dVerY = dHorX` per eqs.
    /// 852 / 857. So a "pure horizontal" CPMV delta still produces a
    /// VERTICAL drift component down rows — that drift is exactly
    /// `dVerY * (2 + 4*sbIdxY) >> 7` per eq. 873.
    ///
    /// With cp0 = (0, 0), cp1 = (16 = 1 luma sample, 0), W = 16,
    /// log2(W) = 4, shift = 3: dHorX = (16) << 3 = 128. dVerY = 128.
    /// At sub-block centre yPosCb = 2 + 4*sbIdxY: MV.y =
    /// (0 + 0 + 128 * yPosCb) >> 7 = yPosCb. So row 0 gets MV.y = 2,
    /// row 1 gets MV.y = 6, etc. Check the spec-prescribed values
    /// match exactly.
    #[test]
    fn derive_subblock_mvs_4param_sample_at_subblock_centre() {
        let w = 16u32;
        let h = 16u32;
        let cp0 = MotionVector { x: 0, y: 0 };
        let cp1 = MotionVector { x: 16, y: 0 };
        let cpmvs = AffineCpmvs::new_4param(cp0, cp1);
        let grid = derive_subblock_mvs(w, h, &cpmvs, false).expect("centre-sampled");
        // dHorX = 128, dVerY = 128 (4-param similarity), so
        // expected: MV.x[ix,iy] ≈ dHorX * xPosCb >> 7 = xPosCb
        //           MV.y[ix,iy] ≈ dVerY * yPosCb >> 7 = yPosCb
        for iy in 0..grid.num_sb_y {
            for ix in 0..grid.num_sb_x {
                let mv = grid.mv_at(ix, iy);
                let x_pos = 2 + 4 * ix as i32;
                let y_pos = 2 + 4 * iy as i32;
                assert_eq!(
                    mv.x, x_pos,
                    "MV.x at ({ix},{iy}): expected {x_pos}, got {}",
                    mv.x,
                );
                assert_eq!(
                    mv.y, y_pos,
                    "MV.y at ({ix},{iy}): expected {y_pos}, got {}",
                    mv.y,
                );
            }
        }
    }

    /// `is_translational` test — degenerate CPMVs report true.
    #[test]
    fn is_translational_detects_degenerate_cpmvs() {
        let cp = MotionVector { x: 32, y: -16 };
        assert!(AffineCpmvs::new_4param(cp, cp).is_translational());
        assert!(AffineCpmvs::new_6param(cp, cp, cp).is_translational());
        let other = MotionVector { x: 33, y: -16 };
        assert!(!AffineCpmvs::new_4param(cp, other).is_translational());
        assert!(!AffineCpmvs::new_6param(cp, cp, other).is_translational());
    }
}
