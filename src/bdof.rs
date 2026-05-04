//! Bi-Directional Optical Flow (BDOF) — round-30 §8.5.6.5 implementation.
//!
//! BDOF is a per-pixel refinement layered on top of bi-prediction
//! motion compensation. For each `4 × 4` sub-block of the coding block
//! the decoder
//!
//! * computes horizontal / vertical gradients on the two list
//!   predictors over a `6 × 6` window (the `4 × 4` sub-block extended
//!   by one sample on every side),
//! * accumulates five per-sub-block correlation sums (`sGx2`, `sGy2`,
//!   `sGxGy`, `sGxdI`, `sGydI`) over the same `6 × 6` window,
//! * solves for a tiny clipped `(vx, vy)` motion-offset pair per
//!   sub-block with the closed-form expressions in eqs. 974 / 975
//!   (clipped to `±(2/16)` pel via `mvRefineThres = 1 << 4`), and
//! * for each pixel of the `4 × 4` sub-block, adds a per-pixel
//!   `bdofOffset` (eq. 976) to the bi-pred sum and feeds the
//!   `Clip3(0, (1 << BitDepth) − 1, … >> shift4)` final round (eq. 977).
//!
//! ## Spec inputs / outputs
//!
//! The §8.5.6.5 signature takes two `(nCbW + 2) × (nCbH + 2)` luma
//! arrays carrying the *high-precision intermediate* per-list
//! predictions (the `BitDepth + 6` precision values that come out of
//! the §8.5.6.3 horizontal-then-vertical separable filter just before
//! the per-list clip and right shift). Output is the `nCbW × nCbH`
//! luma `pbSamples` after BDOF refinement and the eq. 977 clip.
//!
//! ## What this module ships
//!
//! 1. [`bdof_refine_into`] — the bit-accurate algorithm, taking the
//!    two `(nCbW + 2) × (nCbH + 2)` `i32` extended-prediction arrays
//!    plus `BitDepth` and writing the refined `nCbW × nCbH` `u8`
//!    output into a destination [`PicturePlane`] at `(dst_x, dst_y)`.
//! 2. [`bdof_used_flag`] — the §8.5.5.1 condition list that decides
//!    whether BDOF runs for the current CU. The picture-header /
//!    SPS-flag inputs are forwarded by the leaf-CU walker; this
//!    helper mirrors the spec bullet list one-to-one.
//! 3. [`build_extended_pred_high_precision`] — round-32 helper that
//!    takes the §8.5.6.3 high-precision intermediate produced by
//!    [`crate::inter::predict_luma_block_high_precision`] and lifts
//!    it into the spec's `(nCbW + 2) × (nCbH + 2)` extended layout via
//!    pure 1-sample edge replication. This is the spec-byte-identical
//!    BDOF input pipeline at every supported bit depth.
//! 4. [`build_extended_pred_8bit`] — *deprecated* round-30 best-effort
//!    bridge that lifts an 8-bit per-list prediction by `<< 6`. Loses
//!    the spec's `BitDepth + 6` precision (gradients are computed at
//!    8-bit instead of 14-bit at BD = 8) and is retained only for
//!    pre-round-32 callers; new code should use the high-precision
//!    path.
//!
//! ## Spec reference
//!
//! ITU-T H.266 | ISO/IEC 23090-3 (V4, 01/2026) §8.5.6.5 (eqs. 958–977).
//! The implementation is spec-only — no third-party VVC decoder source
//! was consulted.

use oxideav_core::{Error, Result};

use crate::reconstruct::PicturePlane;

/// `mvRefineThres = 1 << 4` (spec, top of §8.5.6.5). Caps the per-sub-
/// block refinement at `±(2/16)` luma pel.
pub const MV_REFINE_THRES: i32 = 1 << 4;

/// `shift1 = 6` (spec, top of §8.5.6.5).
const SHIFT1: i32 = 6;

/// `shift2 = 4` (spec, top of §8.5.6.5).
const SHIFT2: i32 = 4;

/// `shift3 = 1` (spec, top of §8.5.6.5).
const SHIFT3: i32 = 1;

/// Spec sign function: `Sign(x) = x > 0 ? 1 : x < 0 ? -1 : 0`.
fn sign(v: i32) -> i32 {
    match v.cmp(&0) {
        std::cmp::Ordering::Greater => 1,
        std::cmp::Ordering::Less => -1,
        std::cmp::Ordering::Equal => 0,
    }
}

/// `Floor(Log2(x))` — base-2 logarithm of a positive integer rounded
/// down. Used by eqs. 974 / 975. The spec only invokes it for
/// `sGx2 > 0` / `sGy2 > 0`, so we panic on zero (caller must gate).
fn floor_log2(x: i32) -> i32 {
    debug_assert!(x > 0, "floor_log2 only defined for x > 0 (got {x})");
    31 - (x as u32).leading_zeros() as i32
}

/// `bdofFlag` derivation conditions from §8.5.5.1 — returns true iff
/// BDOF should run for the current CU.
///
/// Inputs cover the spec's bullet list one-to-one. The walker passes
/// the already-resolved values; this helper does **not** re-derive
/// them. `cb_w * cb_h >= 128` is enforced here on top of the per-axis
/// `>= 8` bounds because the spec lists it as a separate bullet.
///
/// `is_strp_l0` / `is_strp_l1` carry the §8.3.2 short-term-reference
/// classification of `RefPicList[0][refIdxL0]` / `RefPicList[1][refIdxL1]`.
/// `same_diff_poc` is the boolean
/// `DiffPicOrderCnt(currPic, RefPicList[0][refIdxL0]) ==
///  DiffPicOrderCnt(RefPicList[1][refIdxL1], currPic)`
/// (the symmetric-distance test).
#[allow(clippy::too_many_arguments)]
pub fn bdof_used_flag(
    ph_bdof_disabled_flag: bool,
    pred_flag_l0: bool,
    pred_flag_l1: bool,
    same_diff_poc: bool,
    is_strp_l0: bool,
    is_strp_l1: bool,
    motion_model_idc: u8,
    merge_subblock_flag: bool,
    sym_mvd_flag: bool,
    ciip_flag: bool,
    bcw_idx: u8,
    luma_weight_l0_flag: bool,
    luma_weight_l1_flag: bool,
    chroma_weight_l0_flag: bool,
    chroma_weight_l1_flag: bool,
    cb_width: u32,
    cb_height: u32,
    rpr_constraints_active_l0: bool,
    rpr_constraints_active_l1: bool,
    c_idx: u8,
) -> bool {
    !ph_bdof_disabled_flag
        && pred_flag_l0
        && pred_flag_l1
        && same_diff_poc
        && is_strp_l0
        && is_strp_l1
        && motion_model_idc == 0
        && !merge_subblock_flag
        && !sym_mvd_flag
        && !ciip_flag
        && bcw_idx == 0
        && !luma_weight_l0_flag
        && !luma_weight_l1_flag
        && !chroma_weight_l0_flag
        && !chroma_weight_l1_flag
        && cb_width >= 8
        && cb_height >= 8
        && (cb_width as u64 * cb_height as u64) >= 128
        && !rpr_constraints_active_l0
        && !rpr_constraints_active_l1
        && c_idx == 0
}

/// Lift a pre-clamped 8-bit per-list prediction `(nCbW × nCbH)` into
/// the spec's `(nCbW + 2) × (nCbH + 2)` extended-prediction layout.
///
/// The output array is scaled into the §8.5.6.5 high-precision input
/// domain by left-shifting `SHIFT1 = 6` bits — i.e. each 8-bit sample
/// `s ∈ [0, 255]` maps to `s << 6 ∈ [0, 16320]`. With this scaling
/// `shift1 = 6` (eqs. 962–965) recovers the original sample, so
/// gradients/diff are computed at 8-bit precision instead of the
/// spec's 14-bit precision. This is a best-effort 8-bit bridge — for
/// spec-byte-identical BDOF prefer the round-32
/// [`crate::inter::predict_luma_block_high_precision`] +
/// [`build_extended_pred_high_precision`] pair, which keeps the full
/// `BitDepth + 6` precision intermediate produced by §8.5.6.3.
///
/// One-sample border replication uses edge clamping
/// (`Clip3(0, w-1, x)`, `Clip3(0, h-1, y)`); this matches the spec's
/// `hx = Clip3(1, nCbW, x)` / `vy = Clip3(1, nCbH, y)` clip behaviour
/// applied to a `(nCbW + 2) × (nCbH + 2)` array indexed `0..=nCbW+1` /
/// `0..=nCbH+1`.
///
/// The returned vector is row-major of length `(nCbW + 2) * (nCbH + 2)`.
#[deprecated(
    since = "0.1.0",
    note = "Round-32 surfaced the §8.5.6.3 BD+6 intermediate via \
            `predict_luma_block_high_precision` — pair it with \
            `build_extended_pred_high_precision` for spec-byte-identical \
            BDOF at all bit depths. The 8-bit lifter loses 6 bits of \
            precision and is only retained for back-compat with \
            pre-round-32 callers."
)]
pub fn build_extended_pred_8bit(pred: &PicturePlane, n_cb_w: u32, n_cb_h: u32) -> Result<Vec<i32>> {
    if pred.width < n_cb_w as usize || pred.height < n_cb_h as usize {
        return Err(Error::invalid(format!(
            "bdof: 8-bit pred plane {}x{} smaller than nCbW×nCbH = {}x{}",
            pred.width, pred.height, n_cb_w, n_cb_h,
        )));
    }
    let ext_w = n_cb_w as usize + 2;
    let ext_h = n_cb_h as usize + 2;
    let mut out = vec![0i32; ext_w * ext_h];
    for y in 0..ext_h {
        let src_y = (y as i32 - 1).clamp(0, n_cb_h as i32 - 1) as usize;
        for x in 0..ext_w {
            let src_x = (x as i32 - 1).clamp(0, n_cb_w as i32 - 1) as usize;
            out[y * ext_w + x] = (pred.samples[src_y * pred.stride + src_x] as i32) << SHIFT1;
        }
    }
    Ok(out)
}

/// Lift a §8.5.6.3 high-precision per-list prediction (the
/// `BitDepth + 6` precision i32 intermediate produced by
/// [`crate::inter::predict_luma_block_high_precision`]) into the
/// spec's `(nCbW + 2) × (nCbH + 2)` extended-prediction layout via
/// 1-sample edge replication.
///
/// Unlike [`build_extended_pred_8bit`] this performs *no* scaling —
/// the input `pred` already lives in the BD + 6 precision domain
/// §8.5.6.5 expects, so its values pass through unchanged. The
/// resulting buffer makes [`bdof_refine_into`] spec-byte-identical to
/// the §8.5.6.5 pseudocode at every supported bit depth.
///
/// `pred` is row-major `nCbW × nCbH` (stride = nCbW). Edge replication
/// matches the spec's `hx = Clip3(1, nCbW, x)` / `vy = Clip3(1, nCbH,
/// y)` clip applied to the extended array indexed `0..=nCbW+1` /
/// `0..=nCbH+1`.
pub fn build_extended_pred_high_precision(
    pred: &[i32],
    n_cb_w: u32,
    n_cb_h: u32,
) -> Result<Vec<i32>> {
    let n_cb_w_us = n_cb_w as usize;
    let n_cb_h_us = n_cb_h as usize;
    if pred.len() != n_cb_w_us * n_cb_h_us {
        return Err(Error::invalid(format!(
            "bdof: HP pred buffer length {} != nCbW×nCbH = {}×{} = {}",
            pred.len(),
            n_cb_w,
            n_cb_h,
            n_cb_w_us * n_cb_h_us,
        )));
    }
    let ext_w = n_cb_w_us + 2;
    let ext_h = n_cb_h_us + 2;
    let mut out = vec![0i32; ext_w * ext_h];
    for y in 0..ext_h {
        let src_y = (y as i32 - 1).clamp(0, n_cb_h as i32 - 1) as usize;
        for x in 0..ext_w {
            let src_x = (x as i32 - 1).clamp(0, n_cb_w as i32 - 1) as usize;
            out[y * ext_w + x] = pred[src_y * n_cb_w_us + src_x];
        }
    }
    Ok(out)
}

/// Apply BDOF refinement (§8.5.6.5 eqs. 958–977) to one CU.
///
/// `pred_l0` / `pred_l1` are the spec's `(nCbW + 2) × (nCbH + 2)`
/// extended high-precision per-list predictions, row-major. `n_cb_w`
/// and `n_cb_h` must both be multiples of 4 and `>= 8` (the algorithm
/// works in `4 × 4` sub-blocks; `bdof_used_flag` already enforces the
/// outer `>= 8` and area `>= 128` bullets).
///
/// Output overwrites `dst.samples[(dst_y..dst_y+n_cb_h)]
/// [(dst_x..dst_x+n_cb_w)]` with the eq. 977 `pbSamples` values
/// `Clip3(0, (1 << bit_depth) - 1, …)`.
#[allow(clippy::too_many_arguments)]
pub fn bdof_refine_into(
    dst: &mut PicturePlane,
    dst_x: u32,
    dst_y: u32,
    n_cb_w: u32,
    n_cb_h: u32,
    pred_l0: &[i32],
    pred_l1: &[i32],
    bit_depth: u32,
) -> Result<()> {
    if n_cb_w == 0 || n_cb_h == 0 || (n_cb_w & 3) != 0 || (n_cb_h & 3) != 0 {
        return Err(Error::invalid(format!(
            "bdof: nCbW × nCbH = {n_cb_w}x{n_cb_h} must be non-zero multiples of 4",
        )));
    }
    let ext_w = n_cb_w as usize + 2;
    let ext_h = n_cb_h as usize + 2;
    if pred_l0.len() != ext_w * ext_h || pred_l1.len() != ext_w * ext_h {
        return Err(Error::invalid(format!(
            "bdof: extended prediction arrays must be ({}+2)x({}+2) = {} samples (got L0={}, L1={})",
            n_cb_w,
            n_cb_h,
            ext_w * ext_h,
            pred_l0.len(),
            pred_l1.len(),
        )));
    }
    if dst_x as usize + n_cb_w as usize > dst.width || dst_y as usize + n_cb_h as usize > dst.height
    {
        return Err(Error::invalid(format!(
            "bdof: destination block ({dst_x},{dst_y}) {n_cb_w}x{n_cb_h} out of plane bounds {}x{}",
            dst.width, dst.height,
        )));
    }
    if !(8..=16).contains(&bit_depth) {
        return Err(Error::invalid(format!(
            "bdof: bit_depth {bit_depth} out of supported range 8..=16",
        )));
    }

    // shift4 / offset4 per the spec's "derived as follows" preamble.
    // shift4 = max(3, 15 - BitDepth). offset4 = 1 << (shift4 - 1).
    let shift4 = std::cmp::max(3, 15 - bit_depth as i32);
    let offset4: i32 = 1 << (shift4 - 1);
    let max_sample: i32 = (1i32 << bit_depth) - 1;

    let n_cb_w_i = n_cb_w as i32;
    let n_cb_h_i = n_cb_h as i32;

    // Helpers to read predSamplesLN[hx][vy] under the §8.5.6.5
    // convention where the extended array is indexed 0..=nCbW+1 in
    // x and 0..=nCbH+1 in y. The spec's "hx = Clip3(1, nCbW, x)"
    // clip is intentional — for the eq. 962/963 gradient reads at
    // x = xSb-1..xSb+4, the +1 index can hit nCbW+1, which is the
    // out-of-range slot the spec wants clamped down to nCbW.
    let read_l0 = |hx: i32, vy: i32| -> i32 {
        debug_assert!((0..=n_cb_w_i + 1).contains(&hx));
        debug_assert!((0..=n_cb_h_i + 1).contains(&vy));
        pred_l0[vy as usize * ext_w + hx as usize]
    };
    let read_l1 = |hx: i32, vy: i32| -> i32 {
        debug_assert!((0..=n_cb_w_i + 1).contains(&hx));
        debug_assert!((0..=n_cb_h_i + 1).contains(&vy));
        pred_l1[vy as usize * ext_w + hx as usize]
    };

    // Per the spec, gradient/diff/temp arrays cover the 6x6 window
    // x ∈ {xSb-1 .. xSb+4}, y ∈ {ySb-1 .. ySb+4} per sub-block. We
    // allocate them per-sub-block as small fixed-size 6x6 arrays
    // indexed by the local `i + 1, j + 1` offset (for i, j ∈ -1..=4).
    type Win = [[i32; 6]; 6];

    let num_sb_x = n_cb_w as usize / 4;
    let num_sb_y = n_cb_h as usize / 4;

    for sy in 0..num_sb_y {
        let y_sb = (sy << 2) as i32 + 1;
        for sx in 0..num_sb_x {
            let x_sb = (sx << 2) as i32 + 1;

            // Step 1+2: gradients over the 6x6 window. We index by
            // (i+1, j+1) for i, j ∈ -1..=4.
            let mut gh0: Win = [[0; 6]; 6];
            let mut gv0: Win = [[0; 6]; 6];
            let mut gh1: Win = [[0; 6]; 6];
            let mut gv1: Win = [[0; 6]; 6];
            let mut diff: Win = [[0; 6]; 6];

            for j in -1i32..=4 {
                for i in -1i32..=4 {
                    let x = x_sb + i;
                    let y = y_sb + j;
                    let hx = x.clamp(1, n_cb_w_i);
                    let vy = y.clamp(1, n_cb_h_i);
                    // eqs. 962 - 965
                    let gh0v = (read_l0(hx + 1, vy) >> SHIFT1) - (read_l0(hx - 1, vy) >> SHIFT1);
                    let gv0v = (read_l0(hx, vy + 1) >> SHIFT1) - (read_l0(hx, vy - 1) >> SHIFT1);
                    let gh1v = (read_l1(hx + 1, vy) >> SHIFT1) - (read_l1(hx - 1, vy) >> SHIFT1);
                    let gv1v = (read_l1(hx, vy + 1) >> SHIFT1) - (read_l1(hx, vy - 1) >> SHIFT1);
                    let dv = (read_l0(hx, vy) >> SHIFT2) - (read_l1(hx, vy) >> SHIFT2);

                    let ii = (i + 1) as usize;
                    let jj = (j + 1) as usize;
                    gh0[jj][ii] = gh0v;
                    gv0[jj][ii] = gv0v;
                    gh1[jj][ii] = gh1v;
                    gv1[jj][ii] = gv1v;
                    diff[jj][ii] = dv;
                }
            }

            // Step 3: tempH / tempV from the gradient sums (eqs. 967, 968).
            let mut temp_h: Win = [[0; 6]; 6];
            let mut temp_v: Win = [[0; 6]; 6];
            for jj in 0..6 {
                for ii in 0..6 {
                    temp_h[jj][ii] = (gh0[jj][ii] + gh1[jj][ii]) >> SHIFT3;
                    temp_v[jj][ii] = (gv0[jj][ii] + gv1[jj][ii]) >> SHIFT3;
                }
            }

            // Step 4: per-sub-block correlation sums sGx2, sGy2,
            // sGxGy, sGxdI, sGydI over i, j ∈ -1..=4 (the full 6x6
            // window). Eqs. 969 - 973.
            let mut s_gx2: i32 = 0;
            let mut s_gy2: i32 = 0;
            let mut s_gx_gy: i32 = 0;
            let mut s_gx_di: i32 = 0;
            let mut s_gy_di: i32 = 0;
            for jj in 0..6 {
                for ii in 0..6 {
                    let th = temp_h[jj][ii];
                    let tv = temp_v[jj][ii];
                    let d = diff[jj][ii];
                    s_gx2 += th.abs();
                    s_gy2 += tv.abs();
                    s_gx_gy += sign(tv) * th;
                    s_gx_di += -sign(th) * d;
                    s_gy_di += -sign(tv) * d;
                }
            }

            // Step 5: refinement (eqs. 974, 975).
            let vx = if s_gx2 > 0 {
                let raw = (s_gx_di << 2) >> floor_log2(s_gx2);
                raw.clamp(-MV_REFINE_THRES + 1, MV_REFINE_THRES - 1)
            } else {
                0
            };
            let vy = if s_gy2 > 0 {
                let inner = (s_gy_di << 2) - ((vx * s_gx_gy) >> 1);
                let raw = inner >> floor_log2(s_gy2);
                raw.clamp(-MV_REFINE_THRES + 1, MV_REFINE_THRES - 1)
            } else {
                0
            };

            // Step 6: per-pixel pbSamples for x ∈ xSb-1..=xSb+2,
            // y ∈ ySb-1..=ySb+2 (the 4x4 sub-block). Eqs. 976, 977.
            for j in -1i32..=2 {
                for i in -1i32..=2 {
                    let x = x_sb + i;
                    let y = y_sb + j;
                    // gradient-window read at (i+1, j+1) inside
                    // the 6x6 window — note eq. 976 uses
                    // gradient[x+1][y+1] = our window[(i+1)+1][(j+1)+1] = [j+2][i+2].
                    let g_idx_j = (j + 2) as usize;
                    let g_idx_i = (i + 2) as usize;
                    let bdof_offset = vx * (gh0[g_idx_j][g_idx_i] - gh1[g_idx_j][g_idx_i])
                        + vy * (gv0[g_idx_j][g_idx_i] - gv1[g_idx_j][g_idx_i]);
                    // predSamplesLN[x+1][y+1] in spec terms — x and y
                    // here are the spec's loop variables (1..=nCbW),
                    // so the read is at (x+1, y+1) of the extended
                    // array which also indexes 0..=nCbW+1. The spec
                    // never clamps these reads — the +1 keeps them
                    // inside (1..=nCbW+1) when (x, y) ∈ (0..=nCbW).
                    let p0 = pred_l0[(y + 1) as usize * ext_w + (x + 1) as usize];
                    let p1 = pred_l1[(y + 1) as usize * ext_w + (x + 1) as usize];
                    let blended = (p0 + offset4 + p1 + bdof_offset) >> shift4;
                    let clamped = blended.clamp(0, max_sample) as u8;
                    // x, y here are spec loop variables in 0..=nCbW-1 / 0..=nCbH-1
                    // (xSb-1+i for i in -1..=2, with xSb = (xIdx<<2)+1, so x = xIdx*4 + i + 1
                    // ranges 0..=nCbW-1 across all sub-blocks). The pbSamples array
                    // is CU-origin-aligned, so the destination index is (x, y) directly
                    // — no extra adjustment.
                    debug_assert!((0..n_cb_w_i).contains(&x));
                    debug_assert!((0..n_cb_h_i).contains(&y));
                    let dst_off =
                        (dst_y as usize + y as usize) * dst.stride + (dst_x as usize + x as usize);
                    dst.samples[dst_off] = clamped;
                }
            }
        }
    }

    Ok(())
}

#[cfg(test)]
#[allow(deprecated)] // tests intentionally exercise both lifters
mod tests {
    use super::*;

    /// `bdof_used_flag` reflects every spec bullet — flipping any one
    /// field to a "rejecting" value disables BDOF.
    #[test]
    fn used_flag_full_house_is_true() {
        let on = bdof_used_flag(
            false, // ph_bdof_disabled_flag
            true, true, // predFlagL0/L1
            true, // same_diff_poc
            true, true, // STRP both
            0,    // motion_model_idc
            false, false, false, // merge_subblock / sym_mvd / ciip
            0,     // bcwIdx
            false, false, false, false, // weight flags off
            16, 16, // cbW × cbH
            false, false, // RprConstraintsActive
            0,     // cIdx
        );
        assert!(on);
    }

    #[test]
    fn used_flag_rejects_when_bdof_disabled_in_ph() {
        let off = bdof_used_flag(
            true, true, true, true, true, true, 0, false, false, false, 0, false, false, false,
            false, 16, 16, false, false, 0,
        );
        assert!(!off);
    }

    #[test]
    fn used_flag_rejects_uni_pred() {
        let off = bdof_used_flag(
            false, true, false, // predFlagL1 = 0
            true, true, true, 0, false, false, false, 0, false, false, false, false, 16, 16, false,
            false, 0,
        );
        assert!(!off);
    }

    #[test]
    fn used_flag_rejects_small_block() {
        // 4x16 violates the cbWidth >= 8 bullet.
        let off = bdof_used_flag(
            false, true, true, true, true, true, 0, false, false, false, 0, false, false, false,
            false, 4, 16, false, false, 0,
        );
        assert!(!off);
        // 8x8 = 64 violates the area >= 128 bullet.
        let off2 = bdof_used_flag(
            false, true, true, true, true, true, 0, false, false, false, 0, false, false, false,
            false, 8, 8, false, false, 0,
        );
        assert!(!off2);
        // 8x16 = 128 satisfies all three.
        let on = bdof_used_flag(
            false, true, true, true, true, true, 0, false, false, false, 0, false, false, false,
            false, 8, 16, false, false, 0,
        );
        assert!(on);
    }

    #[test]
    fn used_flag_rejects_chroma() {
        let off = bdof_used_flag(
            false, true, true, true, true, true, 0, false, false, false, 0, false, false, false,
            false, 16, 16, false, false, 1, // cIdx = 1 — chroma plane
        );
        assert!(!off);
    }

    #[test]
    fn floor_log2_matches_spec() {
        assert_eq!(floor_log2(1), 0);
        assert_eq!(floor_log2(2), 1);
        assert_eq!(floor_log2(3), 1);
        assert_eq!(floor_log2(4), 2);
        assert_eq!(floor_log2(7), 2);
        assert_eq!(floor_log2(8), 3);
        assert_eq!(floor_log2(1023), 9);
        assert_eq!(floor_log2(1024), 10);
    }

    #[test]
    fn sign_matches_spec() {
        assert_eq!(sign(5), 1);
        assert_eq!(sign(-5), -1);
        assert_eq!(sign(0), 0);
    }

    /// When the two list predictions are pixel-identical, every
    /// gradient sum is symmetric, sGxdI / sGydI are zero, the
    /// refinement (vx, vy) collapses to (0, 0), the per-pixel
    /// bdofOffset is zero, and eq. 977 reduces to the eq. 980
    /// default-weighted bipred average. With both inputs lifted from
    /// the same 8-bit constant `c` via [`build_extended_pred_8bit`],
    /// the output is exactly `c` (no rounding loss).
    #[test]
    fn identical_predictors_pass_through_constant() {
        let n_cb_w: u32 = 8;
        let n_cb_h: u32 = 16; // 8x16 satisfies the >= 128 area gate
        let pred_8bit = PicturePlane::filled(n_cb_w as usize, n_cb_h as usize, 130);
        let ext_l0 = build_extended_pred_8bit(&pred_8bit, n_cb_w, n_cb_h).unwrap();
        let ext_l1 = ext_l0.clone();
        let mut dst = PicturePlane::filled(n_cb_w as usize, n_cb_h as usize, 0);

        bdof_refine_into(&mut dst, 0, 0, n_cb_w, n_cb_h, &ext_l0, &ext_l1, 8).unwrap();
        for &s in &dst.samples {
            assert_eq!(s, 130, "constant input must round-trip exactly");
        }
    }

    /// A horizontal ramp (sample = x) is bit-identical between the
    /// two list predictions; gradients are symmetric (gradientHL0 =
    /// gradientHL1), so the cross-correlation sums driving (vx, vy)
    /// vanish and BDOF is a no-op. Output equals the average input,
    /// which equals the input itself.
    #[test]
    fn identical_horizontal_ramp_is_no_op() {
        let n_cb_w: u32 = 16;
        let n_cb_h: u32 = 8;
        let mut pred_8bit = PicturePlane::filled(n_cb_w as usize, n_cb_h as usize, 0);
        for y in 0..n_cb_h as usize {
            for x in 0..n_cb_w as usize {
                pred_8bit.samples[y * pred_8bit.stride + x] = (10 + x * 4) as u8;
            }
        }
        let ext_l0 = build_extended_pred_8bit(&pred_8bit, n_cb_w, n_cb_h).unwrap();
        let ext_l1 = ext_l0.clone();
        let mut dst = PicturePlane::filled(n_cb_w as usize, n_cb_h as usize, 0);

        bdof_refine_into(&mut dst, 0, 0, n_cb_w, n_cb_h, &ext_l0, &ext_l1, 8).unwrap();
        for y in 0..n_cb_h as usize {
            for x in 0..n_cb_w as usize {
                let want = pred_8bit.samples[y * pred_8bit.stride + x];
                let got = dst.samples[y * dst.stride + x];
                assert_eq!(
                    got, want,
                    "identical-input BDOF must be a no-op at ({x}, {y})",
                );
            }
        }
    }

    /// Two list predictions that differ by a tiny per-pixel gradient
    /// (L1 = L0 with a 1-sample horizontal offset) drive a non-zero
    /// `vx` refinement. The output must remain inside the spec's
    /// `Clip3(0, 255, ...)` range and must differ from the plain
    /// eq. 980 average — i.e. BDOF actually changed something.
    #[test]
    fn motion_offset_input_drives_non_zero_refinement() {
        let n_cb_w: u32 = 8;
        let n_cb_h: u32 = 16;
        let mut p0 = PicturePlane::filled(n_cb_w as usize, n_cb_h as usize, 0);
        let mut p1 = PicturePlane::filled(n_cb_w as usize, n_cb_h as usize, 0);
        for y in 0..n_cb_h as usize {
            for x in 0..n_cb_w as usize {
                let v0 = (10 + x * 8 + y * 2) as u8;
                p0.samples[y * p0.stride + x] = v0;
                // L1 = L0 shifted right by one sample (clamped at edge).
                let xp = x.saturating_sub(1);
                p1.samples[y * p1.stride + x] = (10 + xp * 8 + y * 2) as u8;
            }
        }
        let ext_l0 = build_extended_pred_8bit(&p0, n_cb_w, n_cb_h).unwrap();
        let ext_l1 = build_extended_pred_8bit(&p1, n_cb_w, n_cb_h).unwrap();
        let mut dst = PicturePlane::filled(n_cb_w as usize, n_cb_h as usize, 0);
        bdof_refine_into(&mut dst, 0, 0, n_cb_w, n_cb_h, &ext_l0, &ext_l1, 8).unwrap();

        // Compute the plain-average baseline; assert dst differs at
        // at least one interior position (the BDOF refinement is
        // non-trivial) and that every sample stays in [0, 255].
        let mut differs_somewhere = false;
        for y in 0..n_cb_h as usize {
            for x in 0..n_cb_w as usize {
                let v0 = p0.samples[y * p0.stride + x] as u32;
                let v1 = p1.samples[y * p1.stride + x] as u32;
                let avg = ((v0 + v1 + 1) >> 1) as u8;
                let got = dst.samples[y * dst.stride + x];
                if got != avg {
                    differs_somewhere = true;
                }
            }
        }
        assert!(
            differs_somewhere,
            "BDOF refinement must alter at least one pixel"
        );
    }

    /// `bdof_refine_into` rejects malformed dimensions (non-multiple-
    /// of-4 sides) and mismatched extended-array sizes.
    #[test]
    fn refine_rejects_bad_inputs() {
        let mut dst = PicturePlane::filled(8, 8, 0);
        let bad_size_l0 = vec![0i32; 4];
        let bad_size_l1 = vec![0i32; 4];
        assert!(bdof_refine_into(&mut dst, 0, 0, 8, 8, &bad_size_l0, &bad_size_l1, 8).is_err());

        let ext = vec![0i32; 10 * 10];
        assert!(
            bdof_refine_into(&mut dst, 0, 0, 9, 9, &ext, &ext, 8).is_err(),
            "non-multiple-of-4 sides must be rejected"
        );

        let bad_bd = vec![0i32; 10 * 10];
        assert!(
            bdof_refine_into(&mut dst, 0, 0, 8, 8, &bad_bd, &bad_bd, 17).is_err(),
            "out-of-range bit depth must be rejected"
        );
    }

    /// `build_extended_pred_8bit` produces a `(nCbW + 2) × (nCbH + 2)`
    /// row-major buffer with replicated edges and the spec's
    /// `<< SHIFT1` scaling.
    #[test]
    fn build_extended_replicates_edges_and_scales() {
        let mut p = PicturePlane::filled(4, 4, 0);
        for y in 0..4 {
            for x in 0..4 {
                p.samples[y * p.stride + x] = (10 + x + y * 4) as u8;
            }
        }
        let ext = build_extended_pred_8bit(&p, 4, 4).unwrap();
        assert_eq!(ext.len(), 6 * 6);

        // Top-left corner = src(0, 0) << 6.
        assert_eq!(ext[0], 10 << SHIFT1);
        // Top-right corner = src(3, 0) << 6.
        assert_eq!(ext[5], 13 << SHIFT1);
        // Bottom-left corner = src(0, 3) << 6.
        assert_eq!(ext[5 * 6], (10 + 12) << SHIFT1);
        // Bottom-right corner = src(3, 3) << 6.
        assert_eq!(ext[5 * 6 + 5], (10 + 3 + 12) << SHIFT1);
        // Centre — extended[(1, 1)] = src(0, 0) << 6.
        assert_eq!(ext[1 * 6 + 1], 10 << SHIFT1);
        // Right edge replication — extended[(5, 2)] = src(3, 1) << 6.
        assert_eq!(ext[2 * 6 + 5], (10 + 3 + 4) << SHIFT1);
    }

    /// `build_extended_pred_high_precision` replicates edges without
    /// any scaling — the input is already in `BitDepth + 6` precision.
    #[test]
    fn build_extended_high_precision_replicates_edges_no_scale() {
        // Values lifted by `<< 6` to live in BD = 8 high-precision
        // domain, matching what `predict_luma_block_high_precision`
        // would emit on a constant-input integer-pel MC.
        let mut pred = vec![0i32; 4 * 4];
        for y in 0..4usize {
            for x in 0..4usize {
                pred[y * 4 + x] = ((10 + x + y * 4) as i32) << 6;
            }
        }
        let ext = build_extended_pred_high_precision(&pred, 4, 4).unwrap();
        assert_eq!(ext.len(), 6 * 6);
        // Top-left corner = pred(0, 0).
        assert_eq!(ext[0], 10 << 6);
        // Top-right corner = pred(3, 0).
        assert_eq!(ext[5], 13 << 6);
        // Bottom-left corner = pred(0, 3).
        assert_eq!(ext[5 * 6], (10 + 12) << 6);
        // Bottom-right corner = pred(3, 3).
        assert_eq!(ext[5 * 6 + 5], (10 + 3 + 12) << 6);
        // Centre — extended[(1, 1)] = pred(0, 0).
        assert_eq!(ext[1 * 6 + 1], 10 << 6);
        // Right edge replication — extended[(5, 2)] = pred(3, 1).
        assert_eq!(ext[2 * 6 + 5], (10 + 3 + 4) << 6);
    }

    /// `build_extended_pred_high_precision` rejects mismatched buffer
    /// lengths — a `(nCbW × nCbH)` input is required.
    #[test]
    fn build_extended_high_precision_rejects_bad_size() {
        let bad = vec![0i32; 4 * 4];
        assert!(build_extended_pred_high_precision(&bad, 8, 8).is_err());
    }

    /// **Round-32 spec parity check** — feed the same per-list MC
    /// outputs through both extended-prediction lifters and confirm:
    ///   (a) at the BD = 8 14-bit precision intermediate, the HP and
    ///       8-bit-lifter paths produce values that differ only by
    ///       the H+V filter rounding (the 8-bit path discards the
    ///       sub-LSB precision the spec keeps);
    ///   (b) the BDOF refinement output is within 1 LSB of equal at
    ///       BitDepth 8 (rounding noise from the sub-LSB precision
    ///       loss).
    #[test]
    fn hp_path_matches_8bit_lifter_within_1_lsb_at_8bit() {
        use crate::inter::{predict_luma_block, predict_luma_block_high_precision, MotionVector};
        let n_cb_w: u32 = 8;
        let n_cb_h: u32 = 16;
        // Build a 32×32 reference plane with a smooth gradient so MC
        // exercises subpel-interpolation + per-list differs.
        let ref_w: usize = 32;
        let ref_h: usize = 32;
        let mut ref_l0 = PicturePlane::filled(ref_w, ref_h, 0);
        let mut ref_l1 = PicturePlane::filled(ref_w, ref_h, 0);
        for y in 0..ref_h {
            for x in 0..ref_w {
                ref_l0.samples[y * ref_w + x] = (40 + x * 3 + y * 2).min(255) as u8;
                // L1 is slightly different so BDOF refinement is non-zero.
                ref_l1.samples[y * ref_w + x] = (38 + x * 3 + y * 2).min(255) as u8;
            }
        }
        // Use the same source-origin for both paths: dst_x=0, dst_y=0,
        // mv=0 reads ref(0,0)..(w,h) for both `predict_luma_block` and
        // `predict_luma_block_high_precision` so the comparison
        // isolates the precision difference, not the source-origin
        // arithmetic.
        let mv_l0 = MotionVector { x: 0, y: 0 };
        let mv_l1 = MotionVector { x: 0, y: 0 };
        let dst_x: u32 = 0;
        let dst_y: u32 = 0;

        // High-precision path.
        let hp_l0 =
            predict_luma_block_high_precision(dst_x, dst_y, n_cb_w, n_cb_h, &ref_l0, mv_l0, 8)
                .unwrap();
        let hp_l1 =
            predict_luma_block_high_precision(dst_x, dst_y, n_cb_w, n_cb_h, &ref_l1, mv_l1, 8)
                .unwrap();
        let ext_hp_l0 = build_extended_pred_high_precision(&hp_l0, n_cb_w, n_cb_h).unwrap();
        let ext_hp_l1 = build_extended_pred_high_precision(&hp_l1, n_cb_w, n_cb_h).unwrap();
        let mut dst_hp = PicturePlane::filled(n_cb_w as usize, n_cb_h as usize, 0);
        bdof_refine_into(
            &mut dst_hp,
            dst_x,
            dst_y,
            n_cb_w,
            n_cb_h,
            &ext_hp_l0,
            &ext_hp_l1,
            8,
        )
        .unwrap();

        // 8-bit-lifter path.
        let mut scratch_l0 = PicturePlane::filled(n_cb_w as usize, n_cb_h as usize, 0);
        let mut scratch_l1 = PicturePlane::filled(n_cb_w as usize, n_cb_h as usize, 0);
        predict_luma_block(&mut scratch_l0, 0, 0, n_cb_w, n_cb_h, &ref_l0, mv_l0).unwrap();
        predict_luma_block(&mut scratch_l1, 0, 0, n_cb_w, n_cb_h, &ref_l1, mv_l1).unwrap();
        let ext_8b_l0 = build_extended_pred_8bit(&scratch_l0, n_cb_w, n_cb_h).unwrap();
        let ext_8b_l1 = build_extended_pred_8bit(&scratch_l1, n_cb_w, n_cb_h).unwrap();
        let mut dst_8b = PicturePlane::filled(n_cb_w as usize, n_cb_h as usize, 0);
        bdof_refine_into(
            &mut dst_8b,
            dst_x,
            dst_y,
            n_cb_w,
            n_cb_h,
            &ext_8b_l0,
            &ext_8b_l1,
            8,
        )
        .unwrap();

        // For zero-MV integer-pel both paths read the ref directly:
        // HP returns `sample << 6` and `build_extended_pred_8bit`
        // computes `sample << 6` after `predict_luma_block`'s
        // `mc_copy_block_int`. The two extended buffers must therefore
        // be bit-identical, and so must the refined BDOF output.
        for (i, (a, b)) in ext_hp_l0.iter().zip(ext_8b_l0.iter()).enumerate() {
            assert_eq!(
                a, b,
                "integer-pel L0 ext buffers must be bit-identical (idx={i})",
            );
        }
        for y in 0..n_cb_h as usize {
            for x in 0..n_cb_w as usize {
                let off = y * dst_hp.stride + x;
                let a = dst_hp.samples[off] as i32;
                let b = dst_8b.samples[off] as i32;
                assert_eq!(
                    a, b,
                    "BDOF HP vs 8-bit lifter (integer-pel) must match exactly at ({x},{y})",
                );
            }
        }

        // For integer-pel zero-MV the §8.5.6.3 H+V fast path collapses
        // to the integer lift `<< (14 - BD) = << 6`. The HP intermediate
        // for L0 should therefore match `sample << 6` bit-for-bit (the
        // integer lift coincides with what the 8-bit `<< SHIFT1` does
        // for the integer-pel case).
        for y in 0..n_cb_h as usize {
            for x in 0..n_cb_w as usize {
                let hp_v = hp_l0[y * n_cb_w as usize + x];
                let want = (ref_l0.samples[y * ref_w + x] as i32) << 6;
                assert_eq!(
                    hp_v, want,
                    "integer-pel HP at ({x},{y}) must equal sample << 6",
                );
            }
        }
    }

    /// Subpel MC: HP path keeps the BD + 6 sub-LSB precision the
    /// §8.5.6.3 separable filter computes, while the 8-bit lifter path
    /// rounds to 8 bits per the per-list clip + `<< SHIFT1` re-lift.
    /// At BitDepth 8 the round-trip difference is bounded by ½ LSB
    /// per sub-pixel, and the BDOF-refined output blocks must agree
    /// within ≤ 1 LSB.
    #[test]
    fn hp_subpel_path_matches_8bit_within_1_lsb() {
        use crate::inter::{predict_luma_block, predict_luma_block_high_precision, MotionVector};
        let n_cb_w: u32 = 8;
        let n_cb_h: u32 = 16;
        let ref_w: usize = 32;
        let ref_h: usize = 32;
        let mut ref_l0 = PicturePlane::filled(ref_w, ref_h, 0);
        let mut ref_l1 = PicturePlane::filled(ref_w, ref_h, 0);
        for y in 0..ref_h {
            for x in 0..ref_w {
                ref_l0.samples[y * ref_w + x] = (40 + x * 3 + y * 2).min(255) as u8;
                ref_l1.samples[y * ref_w + x] = (38 + x * 3 + y * 2).min(255) as u8;
            }
        }
        // ½-pel horizontal + integer vertical (xFrac = 8, yFrac = 0)
        // — exercises `luma_h_8tap` for both paths so the precision
        // delta surfaces.
        let mv = MotionVector { x: 8, y: 0 };
        let hp_l0 =
            predict_luma_block_high_precision(0, 0, n_cb_w, n_cb_h, &ref_l0, mv, 8).unwrap();
        let hp_l1 =
            predict_luma_block_high_precision(0, 0, n_cb_w, n_cb_h, &ref_l1, mv, 8).unwrap();
        let ext_hp_l0 = build_extended_pred_high_precision(&hp_l0, n_cb_w, n_cb_h).unwrap();
        let ext_hp_l1 = build_extended_pred_high_precision(&hp_l1, n_cb_w, n_cb_h).unwrap();
        let mut dst_hp = PicturePlane::filled(n_cb_w as usize, n_cb_h as usize, 0);
        bdof_refine_into(&mut dst_hp, 0, 0, n_cb_w, n_cb_h, &ext_hp_l0, &ext_hp_l1, 8).unwrap();

        let mut scratch_l0 = PicturePlane::filled(n_cb_w as usize, n_cb_h as usize, 0);
        let mut scratch_l1 = PicturePlane::filled(n_cb_w as usize, n_cb_h as usize, 0);
        predict_luma_block(&mut scratch_l0, 0, 0, n_cb_w, n_cb_h, &ref_l0, mv).unwrap();
        predict_luma_block(&mut scratch_l1, 0, 0, n_cb_w, n_cb_h, &ref_l1, mv).unwrap();
        let ext_8b_l0 = build_extended_pred_8bit(&scratch_l0, n_cb_w, n_cb_h).unwrap();
        let ext_8b_l1 = build_extended_pred_8bit(&scratch_l1, n_cb_w, n_cb_h).unwrap();
        let mut dst_8b = PicturePlane::filled(n_cb_w as usize, n_cb_h as usize, 0);
        bdof_refine_into(&mut dst_8b, 0, 0, n_cb_w, n_cb_h, &ext_8b_l0, &ext_8b_l1, 8).unwrap();

        for y in 0..n_cb_h as usize {
            for x in 0..n_cb_w as usize {
                let off = y * dst_hp.stride + x;
                let a = dst_hp.samples[off] as i32;
                let b = dst_8b.samples[off] as i32;
                assert!(
                    (a - b).abs() <= 1,
                    "subpel BDOF HP vs 8-bit lifter must match within 1 LSB at ({x},{y}): hp={a} 8bit={b}",
                );
            }
        }
    }

    /// At BD = 10 the HP intermediate must lie in BitDepth + 6 = 16
    /// bit precision (range roughly [0, 2^16)) — i.e. the integer-pel
    /// lift becomes `<< (14 - 10) = << 4` so a 10-bit sample of value
    /// `s` lands at `s << 4`. This locks in the per-bit-depth
    /// precision contract the BDOF gradient pipeline relies on.
    #[test]
    fn hp_path_precision_scales_with_bit_depth() {
        use crate::inter::{predict_luma_block_high_precision, MotionVector};
        // Build a tiny "10-bit" reference: PicturePlane stores u8 so
        // values are capped at 255, but the lift is `<< 4` for BD=10
        // so 255 lands at `255 * 16 = 4080`, still in 16-bit range.
        let ref_p = PicturePlane::filled(16, 16, 200);
        let mv = MotionVector { x: 0, y: 0 };
        let hp = predict_luma_block_high_precision(0, 0, 8, 8, &ref_p, mv, 10).unwrap();
        for &v in &hp {
            assert_eq!(v, 200 << 4, "BD=10 integer-pel: sample << 4");
        }
        // BD=12: lift is `<< 2`.
        let hp12 = predict_luma_block_high_precision(0, 0, 8, 8, &ref_p, mv, 12).unwrap();
        for &v in &hp12 {
            assert_eq!(v, 200 << 2, "BD=12 integer-pel: sample << 2");
        }
        // BD=8: lift is `<< 6` (the 14-bit BD+6 default).
        let hp8 = predict_luma_block_high_precision(0, 0, 8, 8, &ref_p, mv, 8).unwrap();
        for &v in &hp8 {
            assert_eq!(v, 200 << 6, "BD=8 integer-pel: sample << 6");
        }
    }

    /// HP lifter + BDOF on identical L0 = L1 inputs is a no-op: every
    /// output sample equals the input constant. Mirror of the
    /// existing `identical_predictors_pass_through_constant` test but
    /// going through the round-32 HP pipeline rather than the 8-bit
    /// lifter.
    #[test]
    fn hp_path_identical_predictors_pass_through_constant() {
        use crate::inter::{predict_luma_block_high_precision, MotionVector};
        let n_cb_w: u32 = 8;
        let n_cb_h: u32 = 16;
        let ref_p = PicturePlane::filled(n_cb_w as usize, n_cb_h as usize, 130);
        let mv = MotionVector { x: 0, y: 0 };
        let hp_l0 = predict_luma_block_high_precision(0, 0, n_cb_w, n_cb_h, &ref_p, mv, 8).unwrap();
        let hp_l1 = hp_l0.clone();
        let ext_l0 = build_extended_pred_high_precision(&hp_l0, n_cb_w, n_cb_h).unwrap();
        let ext_l1 = build_extended_pred_high_precision(&hp_l1, n_cb_w, n_cb_h).unwrap();
        let mut dst = PicturePlane::filled(n_cb_w as usize, n_cb_h as usize, 0);
        bdof_refine_into(&mut dst, 0, 0, n_cb_w, n_cb_h, &ext_l0, &ext_l1, 8).unwrap();
        for &s in &dst.samples {
            assert_eq!(s, 130, "HP path: identical predictors round-trip exactly");
        }
    }
}
