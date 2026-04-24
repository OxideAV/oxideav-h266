//! VVC per-syntax-element CABAC context-increment derivations
//! (§9.3.4.2.2 – 9.3.4.2.10).
//!
//! Each syntax element gets its own `ctxInc` function. They operate on
//! spec-level inputs (availability flags, neighbouring-CU sizes / prediction
//! modes, coefficient scan position, etc.) and return the `ctxInc` value
//! that the CABAC engine adds to `ctxIdxOffset` from Table 51 to pick the
//! active context model.
//!
//! Functions implemented in this module:
//!
//! * [`ctx_inc_split_cu_flag`] — §9.3.4.2.2 / eq. 1551 with Table 133.
//! * [`ctx_inc_split_qt_flag`] — same clause, Table 133.
//! * [`ctx_inc_pred_mode_flag`] — §9.3.4.2.2 / eq. 1552.
//! * [`ctx_inc_intra_luma_mpm_flag`] — fixed context 0 (Table 132).
//! * [`ctx_inc_sig_coeff_flag`] — §9.3.4.2.8 / eqs. 1572 – 1574.
//! * [`ctx_inc_abs_level_gt_1_flag`] — §9.3.4.2.9 / eqs. 1582 – 1584
//!   (the `j = 0` case, a.k.a. `abs_level_gtx_flag[n][0]`, with the
//!   regular-residual-coding branch).
//! * [`ctx_inc_abs_level_gt_3_flag`] — same clause with the `+= 32`
//!   adjustment of eq. 1585 (the `j = 1` case,
//!   `abs_level_gtx_flag[n][1]`).
//! * [`ctx_inc_coeff_sign_flag_ts`] — §9.3.4.2.10 / eqs. 1586 – 1590.
//!   (In regular residual coding `coeff_sign_flag` is bypass-coded, so
//!   the contextual derivation is only invoked in transform-skip mode.)
//!
//! Spec reference: ITU-T H.266 | ISO/IEC 23090-3 (V4, 01/2026). The
//! implementation is spec-only; no third-party VVC decoder source was
//! consulted.

/// ctxInc for `split_cu_flag` per §9.3.4.2.2 eq. 1551 + Table 133.
///
/// Inputs:
/// * `available_l` / `available_a` — neighbour availability (§6.4.4).
/// * `cb_height_left` — `CbHeight[chType][xNbL][yNbL]` (ignored when
///   `available_l` is false).
/// * `cb_width_above` — `CbWidth[chType][xNbA][yNbA]` (ignored when
///   `available_a` is false).
/// * `cb_width` / `cb_height` — current block size.
/// * `allow_split_bt_ver` / `allow_split_bt_hor` / `allow_split_tt_ver`
///   / `allow_split_tt_hor` / `allow_split_qt` — partition-allowance
///   flags from §7.4.12.4.
pub fn ctx_inc_split_cu_flag(
    available_l: bool,
    available_a: bool,
    cb_height_left: u32,
    cb_width_above: u32,
    cb_width: u32,
    cb_height: u32,
    allow_split_bt_ver: u32,
    allow_split_bt_hor: u32,
    allow_split_tt_ver: u32,
    allow_split_tt_hor: u32,
    allow_split_qt: u32,
) -> u32 {
    let cond_l = available_l && cb_height_left < cb_height;
    let cond_a = available_a && cb_width_above < cb_width;
    let ctx_set_idx = (allow_split_bt_ver
        + allow_split_bt_hor
        + allow_split_tt_ver
        + allow_split_tt_hor
        + 2 * allow_split_qt
        - 1)
        / 2;
    (cond_l as u32) + (cond_a as u32) + ctx_set_idx * 3
}

/// ctxInc for `split_qt_flag` per §9.3.4.2.2 eq. 1551 + Table 133.
///
/// * `cqt_depth_left` / `cqt_depth_above` — `CqtDepth[chType][..][..]`
///   at the neighbouring luma locations (ignored if the corresponding
///   availability flag is false).
/// * `cqt_depth` — current coding quadtree depth.
pub fn ctx_inc_split_qt_flag(
    available_l: bool,
    available_a: bool,
    cqt_depth_left: u32,
    cqt_depth_above: u32,
    cqt_depth: u32,
) -> u32 {
    let cond_l = available_l && cqt_depth_left > cqt_depth;
    let cond_a = available_a && cqt_depth_above > cqt_depth;
    let ctx_set_idx = if cqt_depth >= 2 { 1 } else { 0 };
    (cond_l as u32) + (cond_a as u32) + ctx_set_idx * 3
}

/// ctxInc for `pred_mode_flag` per §9.3.4.2.2 eq. 1552 + Table 133
/// (output is 0 or 1).
///
/// * `left_is_intra` / `above_is_intra` — `CuPredMode[chType][..][..]
///   == MODE_INTRA` at the neighbouring blocks (ignored when the
///   corresponding availability flag is false).
pub fn ctx_inc_pred_mode_flag(
    available_l: bool,
    available_a: bool,
    left_is_intra: bool,
    above_is_intra: bool,
) -> u32 {
    let cond_l = available_l && left_is_intra;
    let cond_a = available_a && above_is_intra;
    (cond_l || cond_a) as u32
}

/// ctxInc for `intra_luma_mpm_flag[x0][y0]` — Table 132 assigns the
/// single context 0 at binIdx 0. No neighbour-based adjustment.
pub fn ctx_inc_intra_luma_mpm_flag() -> u32 {
    0
}

/// ctxInc for `sig_coeff_flag` in regular-residual-coding mode
/// (transform_skip_flag = 0), per §9.3.4.2.8 eqs. 1573 / 1574.
///
/// * `c_idx` — colour component (0 = luma, 1/2 = chroma).
/// * `x_c`, `y_c` — coefficient scan position within the TB.
/// * `q_state` — dependent-quantisation state ∈ 0..3 (§7.4.11.11,
///   driven by eq. 1551-equivalent in the residual walker).
/// * `loc_sum_abs_pass1` — `locSumAbsPass1` from §9.3.4.2.7, the sum
///   of abs-level values over the scan-order neighbourhood already
///   decoded in pass 1.
pub fn ctx_inc_sig_coeff_flag(
    c_idx: u32,
    x_c: u32,
    y_c: u32,
    q_state: i32,
    loc_sum_abs_pass1: u32,
) -> u32 {
    let d = x_c + y_c;
    let q_term = 0i32.max(q_state - 1) as u32;
    let sum_term = ((loc_sum_abs_pass1 + 1) >> 1).min(3);
    if c_idx == 0 {
        // eq. 1573
        let d_term = if d < 2 {
            8
        } else if d < 5 {
            4
        } else {
            0
        };
        12 * q_term + sum_term + d_term
    } else {
        // eq. 1574
        let d_term = if d < 2 { 4 } else { 0 };
        36 + 8 * q_term + sum_term + d_term
    }
}

/// ctxInc for `sig_coeff_flag` in transform-skip residual-coding mode
/// (§9.3.4.2.8 eq. 1572).
pub fn ctx_inc_sig_coeff_flag_ts(loc_num_sig: u32) -> u32 {
    60 + loc_num_sig
}

/// ctxInc for `abs_level_gtx_flag[n][0]` (a.k.a. `abs_level_gt_1_flag`)
/// in regular residual coding (§9.3.4.2.9 eqs. 1582 / 1583 / 1584).
///
/// * `c_idx` — colour component.
/// * `x_c`, `y_c` — scan-position within TB.
/// * `last_sig_x`, `last_sig_y` — `LastSignificantCoeffX/Y` from the
///   residual syntax.
/// * `loc_num_sig`, `loc_sum_abs_pass1` — from §9.3.4.2.7.
pub fn ctx_inc_abs_level_gt_1_flag(
    c_idx: u32,
    x_c: u32,
    y_c: u32,
    last_sig_x: u32,
    last_sig_y: u32,
    loc_num_sig: u32,
    loc_sum_abs_pass1: u32,
) -> u32 {
    let d = x_c + y_c;
    let ctx_offset = (loc_sum_abs_pass1.saturating_sub(loc_num_sig)).min(4);
    if x_c == last_sig_x && y_c == last_sig_y {
        // eq. 1582
        return if c_idx == 0 { 0 } else { 21 };
    }
    if c_idx == 0 {
        // eq. 1583
        let d_term = if d == 0 {
            15
        } else if d < 3 {
            10
        } else if d < 10 {
            5
        } else {
            0
        };
        1 + ctx_offset + d_term
    } else {
        // eq. 1584
        let d_term = if d == 0 { 5 } else { 0 };
        22 + ctx_offset + d_term
    }
}

/// ctxInc for `abs_level_gtx_flag[n][1]` (a.k.a. `abs_level_gt_3_flag`)
/// — same as `gt_1` plus the `+= 32` of eq. 1585.
pub fn ctx_inc_abs_level_gt_3_flag(
    c_idx: u32,
    x_c: u32,
    y_c: u32,
    last_sig_x: u32,
    last_sig_y: u32,
    loc_num_sig: u32,
    loc_sum_abs_pass1: u32,
) -> u32 {
    ctx_inc_abs_level_gt_1_flag(
        c_idx,
        x_c,
        y_c,
        last_sig_x,
        last_sig_y,
        loc_num_sig,
        loc_sum_abs_pass1,
    ) + 32
}

/// ctxInc for `coeff_sign_flag` in transform-skip mode (§9.3.4.2.10
/// eqs. 1588 / 1589 / 1590). In regular residual coding
/// `coeff_sign_flag` is bypass-coded, so this context-variable path is
/// never taken.
///
/// * `left_sign` / `above_sign` — spec's `CoeffSignLevel` values at the
///   left / above neighbours (use 0 if off-edge, see eqs. 1586 / 1587).
/// * `bdpcm` — `BdpcmFlag[x0][y0][cIdx]`.
pub fn ctx_inc_coeff_sign_flag_ts(left_sign: i32, above_sign: i32, bdpcm: bool) -> u32 {
    let bdpcm_shift = if bdpcm { 3 } else { 0 };
    if (left_sign == 0 && above_sign == 0) || left_sign == -above_sign {
        bdpcm_shift
    } else if left_sign >= 0 && above_sign >= 0 {
        1 + bdpcm_shift
    } else {
        2 + bdpcm_shift
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_cu_flag_basic() {
        // No neighbours → cond_l = cond_a = 0; allow_split_qt = 1 gives
        // ctxSetIdx = (0 + 0 + 0 + 0 + 2 - 1) / 2 = 0.
        let inc = ctx_inc_split_cu_flag(false, false, 0, 0, 128, 128, 0, 0, 0, 0, 1);
        assert_eq!(inc, 0);
        // Both neighbours available and smaller than current → cond_l = cond_a = 1.
        let inc = ctx_inc_split_cu_flag(true, true, 64, 64, 128, 128, 1, 1, 1, 1, 1);
        // ctxSetIdx = (1+1+1+1+2-1)/2 = 5/2 = 2. inc = 1 + 1 + 2*3 = 8.
        assert_eq!(inc, 8);
    }

    #[test]
    fn split_qt_flag_basic() {
        // Both neighbours deeper → both conds true; cqt_depth < 2.
        let inc = ctx_inc_split_qt_flag(true, true, 3, 3, 1);
        assert_eq!(inc, 2);
        // Shallow depth, both conds true, cqt_depth >= 2 → set_idx=1.
        let inc = ctx_inc_split_qt_flag(true, true, 3, 3, 2);
        assert_eq!(inc, 5);
    }

    #[test]
    fn pred_mode_flag_is_or_of_intra_neighbours() {
        assert_eq!(ctx_inc_pred_mode_flag(false, false, true, true), 0);
        assert_eq!(ctx_inc_pred_mode_flag(true, false, true, true), 1);
        assert_eq!(ctx_inc_pred_mode_flag(false, true, true, false), 0);
        assert_eq!(ctx_inc_pred_mode_flag(false, true, false, true), 1);
    }

    #[test]
    fn intra_luma_mpm_flag_has_single_context() {
        assert_eq!(ctx_inc_intra_luma_mpm_flag(), 0);
    }

    /// sig_coeff_flag for luma DC position (xC=yC=0): d=0, so d_term=8,
    /// q_state=0 → q_term=0, loc_sum=0 → sum_term=0. Expect inc = 8.
    #[test]
    fn sig_coeff_flag_luma_dc() {
        let inc = ctx_inc_sig_coeff_flag(0, 0, 0, 0, 0);
        assert_eq!(inc, 8);
    }

    /// sig_coeff_flag for chroma with d≥2: no d_term bonus.
    /// c_idx=1, x=1, y=2 → d=3. q_state=2 → q_term=1. sum=2 → (2+1)/2=1.
    /// Expect 36 + 8*1 + 1 + 0 = 45.
    #[test]
    fn sig_coeff_flag_chroma_mid() {
        let inc = ctx_inc_sig_coeff_flag(1, 1, 2, 2, 2);
        assert_eq!(inc, 45);
    }

    /// abs_level_gt_1_flag at the last-significant position: fixed 0
    /// for luma, 21 for chroma.
    #[test]
    fn abs_level_gt_1_at_last_significant() {
        assert_eq!(ctx_inc_abs_level_gt_1_flag(0, 5, 7, 5, 7, 0, 0), 0);
        assert_eq!(ctx_inc_abs_level_gt_1_flag(1, 5, 7, 5, 7, 0, 0), 21);
    }

    /// abs_level_gt_1_flag away from last_sig: luma, d=0, offset=0
    /// → 1 + 0 + 15 = 16.
    #[test]
    fn abs_level_gt_1_luma_dc() {
        let inc = ctx_inc_abs_level_gt_1_flag(0, 0, 0, 3, 3, 0, 0);
        assert_eq!(inc, 16);
    }

    /// abs_level_gt_3_flag = gt_1_flag + 32.
    #[test]
    fn abs_level_gt_3_shifts_by_32() {
        let gt1 = ctx_inc_abs_level_gt_1_flag(0, 0, 0, 3, 3, 0, 0);
        let gt3 = ctx_inc_abs_level_gt_3_flag(0, 0, 0, 3, 3, 0, 0);
        assert_eq!(gt3, gt1 + 32);
    }

    /// coeff_sign_flag TS dispatch:
    /// * both zero or opposite-and-cancelling → 0 (not bdpcm) / 3 (bdpcm)
    /// * both ≥ 0 (and not both zero) → 1 / 4
    /// * otherwise (both negative or non-cancelling mix) → 2 / 5
    #[test]
    fn coeff_sign_flag_ts_paths() {
        // Branch 1 (eq. 1588).
        assert_eq!(ctx_inc_coeff_sign_flag_ts(0, 0, false), 0);
        assert_eq!(ctx_inc_coeff_sign_flag_ts(1, -1, false), 0);
        assert_eq!(ctx_inc_coeff_sign_flag_ts(0, 0, true), 3);
        // Branch 2 (eq. 1589).
        assert_eq!(ctx_inc_coeff_sign_flag_ts(1, 1, false), 1);
        assert_eq!(ctx_inc_coeff_sign_flag_ts(1, 1, true), 4);
        // Branch 3 (eq. 1590): both negative, or non-cancelling mix.
        assert_eq!(ctx_inc_coeff_sign_flag_ts(-1, -1, false), 2);
        assert_eq!(ctx_inc_coeff_sign_flag_ts(-1, -1, true), 5);
        assert_eq!(ctx_inc_coeff_sign_flag_ts(1, -2, false), 2);
    }

    /// sig_coeff_flag_ts = 60 + locNumSig.
    #[test]
    fn sig_coeff_flag_ts_is_offset_60() {
        assert_eq!(ctx_inc_sig_coeff_flag_ts(0), 60);
        assert_eq!(ctx_inc_sig_coeff_flag_ts(4), 64);
    }
}
