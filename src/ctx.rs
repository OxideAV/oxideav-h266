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

/// ctxInc for `intra_bdpcm_luma_flag` — Table 132 column gives a single
/// context (0) at binIdx 0; no neighbour adjustment is used.
pub fn ctx_inc_intra_bdpcm_luma_flag() -> u32 {
    0
}

/// ctxInc for `intra_bdpcm_luma_dir_flag` — Table 132 single context.
pub fn ctx_inc_intra_bdpcm_luma_dir_flag() -> u32 {
    0
}

/// ctxInc for `intra_bdpcm_chroma_flag` — Table 132 single context.
pub fn ctx_inc_intra_bdpcm_chroma_flag() -> u32 {
    0
}

/// ctxInc for `intra_bdpcm_chroma_dir_flag` — Table 132 single context.
pub fn ctx_inc_intra_bdpcm_chroma_dir_flag() -> u32 {
    0
}

/// ctxInc for `intra_luma_not_planar_flag[x0][y0]` — Table 132 gives
/// `!intra_subpartitions_mode_flag` at binIdx 0 (so ctx 0 when ISP is
/// enabled for the current CU, ctx 1 when ISP is disabled).
pub fn ctx_inc_intra_luma_not_planar_flag(intra_subpartitions_mode_flag: bool) -> u32 {
    if intra_subpartitions_mode_flag {
        0
    } else {
        1
    }
}

/// ctxInc for `intra_mip_flag` per §9.3.4.2.1 Table 132 + §9.3.4.2.2
/// eq. 1551 + Table 133.
///
/// If `|log2(cbWidth) - log2(cbHeight)| > 1` the spec forces `ctxInc = 3`
/// (a dedicated context for highly non-square blocks). Otherwise the
/// `condL` / `condA` conditions take the neighbouring `IntraMipFlag`
/// values.
pub fn ctx_inc_intra_mip_flag(
    cb_width: u32,
    cb_height: u32,
    available_l: bool,
    available_a: bool,
    left_mip: bool,
    above_mip: bool,
) -> u32 {
    let log2_w = cb_width.trailing_zeros() as i32;
    let log2_h = cb_height.trailing_zeros() as i32;
    if (log2_w - log2_h).abs() > 1 {
        return 3;
    }
    let cond_l = available_l && left_mip;
    let cond_a = available_a && above_mip;
    (cond_l as u32) + (cond_a as u32)
}

/// ctxInc for `intra_chroma_pred_mode` — Table 132 gives ctxInc 0 at
/// binIdx 0; binIdx 1 and 2 are bypass-coded.
pub fn ctx_inc_intra_chroma_pred_mode() -> u32 {
    0
}

/// ctxInc for `intra_luma_ref_idx` — Table 132 entries are (0, 1) for
/// binIdx 0 and 1 respectively.
pub fn ctx_inc_intra_luma_ref_idx(bin_idx: u32) -> u32 {
    bin_idx.min(1)
}

/// ctxInc for `intra_subpartitions_mode_flag` — fixed 0 (Table 132).
pub fn ctx_inc_intra_subpartitions_mode_flag() -> u32 {
    0
}

/// ctxInc for `intra_subpartitions_split_flag` — fixed 0 (Table 132).
pub fn ctx_inc_intra_subpartitions_split_flag() -> u32 {
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

/// ctxInc for `tu_y_coded_flag` per §9.3.4.2.5.
///
/// * `bdpcm_y` — `BdpcmFlag[x0][y0][0]`.
/// * `isp_split` — `IntraSubPartitionsSplitType != ISP_NO_SPLIT`.
/// * `prev_tu_cbf_y` — `tu_y_coded_flag` of the previous luma TU in the
///   same CU (inferred 0 when this is the first TU, or when ISP is off).
pub fn ctx_inc_tu_y_coded_flag(bdpcm_y: bool, isp_split: bool, prev_tu_cbf_y: bool) -> u32 {
    if bdpcm_y {
        1
    } else if !isp_split {
        0
    } else {
        2 + (prev_tu_cbf_y as u32)
    }
}

/// ctxInc for `tu_cb_coded_flag` per Table 127 — simply the
/// `intra_bdpcm_chroma_flag` bit (1 if BDPCM-chroma, else 0).
pub fn ctx_inc_tu_cb_coded_flag(bdpcm_chroma: bool) -> u32 {
    if bdpcm_chroma {
        1
    } else {
        0
    }
}

/// ctxInc for `tu_cr_coded_flag` per Table 127:
/// `bdpcm_chroma ? 2 : tu_cb_coded_flag`.
pub fn ctx_inc_tu_cr_coded_flag(bdpcm_chroma: bool, tu_cb_coded_flag: bool) -> u32 {
    if bdpcm_chroma {
        2
    } else {
        tu_cb_coded_flag as u32
    }
}

/// ctxInc for `cu_qp_delta_abs` per Table 127: bin 0 ctx 0, bins 1-4 ctx 1.
pub fn ctx_inc_cu_qp_delta_abs(bin_idx: u32) -> u32 {
    if bin_idx == 0 {
        0
    } else {
        1
    }
}

/// ctxInc for `cu_chroma_qp_offset_flag` — fixed 0 (Table 127).
pub fn ctx_inc_cu_chroma_qp_offset_flag() -> u32 {
    0
}

/// ctxInc for `cu_chroma_qp_offset_idx` — fixed 0 (Table 127).
pub fn ctx_inc_cu_chroma_qp_offset_idx() -> u32 {
    0
}

/// ctxInc for `last_sig_coeff_x_prefix` / `last_sig_coeff_y_prefix`
/// per §9.3.4.2.4 eqs. 1555 / 1556.
///
/// * `bin_idx` — current bin index within the TR prefix.
/// * `c_idx` — colour component (0 = luma, 1/2 = chroma).
/// * `log2_tb_size` — `Log2FullTbWidth` for the X prefix,
///   `Log2FullTbHeight` for the Y prefix.
pub fn ctx_inc_last_sig_coeff_prefix(bin_idx: u32, c_idx: u32, log2_tb_size: u32) -> u32 {
    // offsetY[] = {0, 0, 3, 6, 10, 15} indexed by log2TbSize - 1.
    const OFFSET_Y: [u32; 6] = [0, 0, 3, 6, 10, 15];
    let (ctx_offset, ctx_shift) = if c_idx == 0 {
        let idx = log2_tb_size.saturating_sub(1) as usize;
        let off = if idx < OFFSET_Y.len() {
            OFFSET_Y[idx]
        } else {
            15
        };
        let shift = (log2_tb_size + 1) >> 2;
        (off, shift)
    } else {
        // Chroma: ctxOffset = 20, ctxShift = Clip3(0, 2, 2 * log2TbSize >> 3).
        let shift = ((2 * log2_tb_size) >> 3).min(2);
        (20, shift)
    };
    (bin_idx >> ctx_shift) + ctx_offset
}

/// csbfCtx derivation fragment for `sb_coded_flag` — §9.3.4.2.6 eqs.
/// 1564 / 1565 / 1566 / 1567. Regular residual coding (no transform-skip)
/// path: the "right/below" neighbour form is used, capped at 1.
///
/// * `right_sb_coded` — `sb_coded_flag[xS+1][yS]` (0 if xS is at the
///   right edge of the sub-block grid).
/// * `below_sb_coded` — `sb_coded_flag[xS][yS+1]`.
pub fn csbf_ctx_regular(right_sb_coded: bool, below_sb_coded: bool) -> u32 {
    right_sb_coded as u32 + below_sb_coded as u32
}

/// ctxInc for `sb_coded_flag` in regular residual coding — §9.3.4.2.6
/// eqs. 1569 / 1570.
pub fn ctx_inc_sb_coded_flag_regular(c_idx: u32, csbf_ctx: u32) -> u32 {
    if c_idx == 0 {
        csbf_ctx.min(1)
    } else {
        2 + csbf_ctx.min(1)
    }
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

/// ctxInc for `par_level_flag[n]` in regular residual coding — shares
/// the §9.3.4.2.9 derivation with `abs_level_gtx_flag[n][0]` (same
/// inputs, same output). This wrapper simply calls
/// [`ctx_inc_abs_level_gt_1_flag`] so call sites document intent.
pub fn ctx_inc_par_level_flag(
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
    )
}

/// §9.3.4.2.7 derivation of `(locNumSig, locSumAbsPass1)` for regular
/// (non-transform-skip) residual coding, given a `AbsLevelPass1[]` and
/// `sig_coeff_flag[]` array in row-major `(n_tb_w * n_tb_h)` layout.
///
/// Returns `(locNumSig, locSumAbsPass1)`. Reads the 5-sample
/// forward-diagonal neighbourhood: (xC+1,yC), (xC+2,yC), (xC+1,yC+1),
/// (xC,yC+1), (xC,yC+2), each guarded against the `Log2ZoTbWidth /
/// Log2ZoTbHeight` edge per the pseudo-code.
pub fn loc_num_sig_and_sum_abs_pass1(
    x_c: u32,
    y_c: u32,
    abs_level_pass1: &[u32],
    sig_coeff_flag: &[bool],
    log2_zo_tb_w: u32,
    log2_zo_tb_h: u32,
    n_tb_w: u32,
) -> (u32, u32) {
    let mut loc_num_sig = 0u32;
    let mut loc_sum = 0u32;
    let w_max = 1u32 << log2_zo_tb_w;
    let h_max = 1u32 << log2_zo_tb_h;
    let at = |xc: u32, yc: u32| -> usize { (yc as usize) * (n_tb_w as usize) + (xc as usize) };
    if x_c < w_max - 1 {
        loc_num_sig += sig_coeff_flag[at(x_c + 1, y_c)] as u32;
        loc_sum += abs_level_pass1[at(x_c + 1, y_c)];
        if x_c < w_max - 2 {
            loc_num_sig += sig_coeff_flag[at(x_c + 2, y_c)] as u32;
            loc_sum += abs_level_pass1[at(x_c + 2, y_c)];
        }
        if y_c < h_max - 1 {
            loc_num_sig += sig_coeff_flag[at(x_c + 1, y_c + 1)] as u32;
            loc_sum += abs_level_pass1[at(x_c + 1, y_c + 1)];
        }
    }
    if y_c < h_max - 1 {
        loc_num_sig += sig_coeff_flag[at(x_c, y_c + 1)] as u32;
        loc_sum += abs_level_pass1[at(x_c, y_c + 1)];
        if y_c < h_max - 2 {
            loc_num_sig += sig_coeff_flag[at(x_c, y_c + 2)] as u32;
            loc_sum += abs_level_pass1[at(x_c, y_c + 2)];
        }
    }
    (loc_num_sig, loc_sum)
}

/// §9.3.3.2 derivation of `locSumAbs` for Rice-parameter lookup. This
/// is the neighbourhood for `abs_remainder[]` / `dec_abs_level[]`
/// binarisation. Same shape as [`loc_num_sig_and_sum_abs_pass1`] but
/// reads the full `AbsLevel[]` (post-pass-2) array and falls back to
/// `hist_value` at the zo-edge (not `Log2ZoTb*` but `Log2FullTb*`, per
/// the spec — at this level they are identical when zero-out is off).
pub fn loc_sum_abs_rice(
    x_c: u32,
    y_c: u32,
    abs_level: &[u32],
    log2_full_tb_w: u32,
    log2_full_tb_h: u32,
    n_tb_w: u32,
    hist_value: u32,
) -> u32 {
    let mut loc_sum = 0u32;
    let w_max = 1u32 << log2_full_tb_w;
    let h_max = 1u32 << log2_full_tb_h;
    let at = |xc: u32, yc: u32| -> usize { (yc as usize) * (n_tb_w as usize) + (xc as usize) };
    if x_c < w_max - 1 {
        loc_sum += abs_level[at(x_c + 1, y_c)];
        if x_c < w_max - 2 {
            loc_sum += abs_level[at(x_c + 2, y_c)];
        } else {
            loc_sum += hist_value;
        }
        if y_c < h_max - 1 {
            loc_sum += abs_level[at(x_c + 1, y_c + 1)];
        } else {
            loc_sum += hist_value;
        }
    } else {
        loc_sum += 2 * hist_value;
    }
    if y_c < h_max - 1 {
        loc_sum += abs_level[at(x_c, y_c + 1)];
        if y_c < h_max - 2 {
            loc_sum += abs_level[at(x_c, y_c + 2)];
        } else {
            loc_sum += hist_value;
        }
    } else {
        loc_sum += hist_value;
    }
    loc_sum
}

/// Table 128 lookup of `cRiceParam` from `locSumAbs`, after the
/// §9.3.3.2 Clip3(0, 31, (locSumAbs >> shiftVal) − baseLevel * 5)
/// normalisation has been applied by the caller.
///
/// The spec tabulates the full 0..31 range and maps it to 0..3
/// piecewise: 0..6 → 0, 7..13 → 1, 14..27 → 2, 28..31 → 3.
pub fn rice_param_from_loc_sum_abs(loc_sum_abs_clipped: u32) -> u32 {
    match loc_sum_abs_clipped {
        0..=6 => 0,
        7..=13 => 1,
        14..=27 => 2,
        _ => 3,
    }
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

    #[test]
    fn intra_luma_not_planar_flag_inverts_isp() {
        assert_eq!(ctx_inc_intra_luma_not_planar_flag(false), 1);
        assert_eq!(ctx_inc_intra_luma_not_planar_flag(true), 0);
    }

    #[test]
    fn intra_mip_flag_non_square_forces_ctx3() {
        // 16x4 → log2W=4, log2H=2 → |diff|=2 > 1 → ctx=3.
        assert_eq!(ctx_inc_intra_mip_flag(16, 4, true, true, true, true), 3);
        // 4x16 → symmetric.
        assert_eq!(ctx_inc_intra_mip_flag(4, 16, false, false, false, false), 3);
    }

    #[test]
    fn intra_mip_flag_square_counts_neighbours() {
        // No neighbours → 0.
        assert_eq!(ctx_inc_intra_mip_flag(8, 8, false, false, true, true), 0);
        // Left available with MIP → 1.
        assert_eq!(ctx_inc_intra_mip_flag(8, 8, true, false, true, false), 1);
        // Both available with MIP → 2.
        assert_eq!(ctx_inc_intra_mip_flag(8, 8, true, true, true, true), 2);
    }

    #[test]
    fn intra_luma_ref_idx_ctx_inc_by_bin() {
        assert_eq!(ctx_inc_intra_luma_ref_idx(0), 0);
        assert_eq!(ctx_inc_intra_luma_ref_idx(1), 1);
        assert_eq!(ctx_inc_intra_luma_ref_idx(2), 1);
    }

    #[test]
    fn intra_chroma_pred_mode_has_single_context() {
        assert_eq!(ctx_inc_intra_chroma_pred_mode(), 0);
    }

    #[test]
    fn intra_subpartitions_flags_have_fixed_ctx() {
        assert_eq!(ctx_inc_intra_subpartitions_mode_flag(), 0);
        assert_eq!(ctx_inc_intra_subpartitions_split_flag(), 0);
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

    /// tu_y_coded_flag: bdpcm forces 1, no-ISP gives 0, ISP propagates
    /// the previous-TU CBF.
    #[test]
    fn tu_y_coded_flag_ctx_inc_branches() {
        assert_eq!(ctx_inc_tu_y_coded_flag(true, false, false), 1);
        assert_eq!(ctx_inc_tu_y_coded_flag(false, false, true), 0);
        assert_eq!(ctx_inc_tu_y_coded_flag(false, true, false), 2);
        assert_eq!(ctx_inc_tu_y_coded_flag(false, true, true), 3);
    }

    /// tu_cb_coded_flag: ctxInc = intra_bdpcm_chroma_flag.
    #[test]
    fn tu_cb_coded_flag_ctx_inc_branches() {
        assert_eq!(ctx_inc_tu_cb_coded_flag(false), 0);
        assert_eq!(ctx_inc_tu_cb_coded_flag(true), 1);
    }

    /// tu_cr_coded_flag: bdpcm_chroma → 2; else → tu_cb_coded_flag.
    #[test]
    fn tu_cr_coded_flag_ctx_inc_branches() {
        assert_eq!(ctx_inc_tu_cr_coded_flag(false, false), 0);
        assert_eq!(ctx_inc_tu_cr_coded_flag(false, true), 1);
        assert_eq!(ctx_inc_tu_cr_coded_flag(true, false), 2);
        assert_eq!(ctx_inc_tu_cr_coded_flag(true, true), 2);
    }

    /// cu_qp_delta_abs: bin 0 ctx 0, rest ctx 1.
    #[test]
    fn cu_qp_delta_abs_ctx_inc_branches() {
        assert_eq!(ctx_inc_cu_qp_delta_abs(0), 0);
        assert_eq!(ctx_inc_cu_qp_delta_abs(1), 1);
        assert_eq!(ctx_inc_cu_qp_delta_abs(4), 1);
    }

    /// last_sig_coeff_prefix: luma 4×4 TB at binIdx 0 has log2TbSize=2,
    /// so ctxOffset = offsetY[1] = 0, ctxShift = (2+1)>>2 = 0,
    /// ctxInc = 0 + 0 = 0.
    #[test]
    fn last_sig_coeff_prefix_luma_4x4_bin_0() {
        assert_eq!(ctx_inc_last_sig_coeff_prefix(0, 0, 2), 0);
    }

    /// last_sig_coeff_prefix: luma 32-wide TB at binIdx 4 has
    /// log2TbSize=5, ctxOffset = offsetY[4] = 10, ctxShift = 6>>2 = 1,
    /// ctxInc = (4 >> 1) + 10 = 12.
    #[test]
    fn last_sig_coeff_prefix_luma_32_bin_4() {
        assert_eq!(ctx_inc_last_sig_coeff_prefix(4, 0, 5), 12);
    }

    /// last_sig_coeff_prefix chroma: offset 20, shift = Clip3(0, 2, 2 * log2 >> 3).
    /// At log2 = 4 → shift = 8 >> 3 = 1. binIdx=3 → (3>>1)+20 = 21.
    #[test]
    fn last_sig_coeff_prefix_chroma_16_bin_3() {
        assert_eq!(ctx_inc_last_sig_coeff_prefix(3, 1, 4), 21);
    }

    /// csbfCtx regular form: counts right/below coded flags.
    #[test]
    fn csbf_ctx_regular_counts_coded_neighbours() {
        assert_eq!(csbf_ctx_regular(false, false), 0);
        assert_eq!(csbf_ctx_regular(true, false), 1);
        assert_eq!(csbf_ctx_regular(false, true), 1);
        assert_eq!(csbf_ctx_regular(true, true), 2);
    }

    /// sb_coded_flag regular: luma caps at 1; chroma gets +2 offset.
    #[test]
    fn sb_coded_flag_regular_ctx_inc_branches() {
        assert_eq!(ctx_inc_sb_coded_flag_regular(0, 0), 0);
        assert_eq!(ctx_inc_sb_coded_flag_regular(0, 2), 1);
        assert_eq!(ctx_inc_sb_coded_flag_regular(1, 0), 2);
        assert_eq!(ctx_inc_sb_coded_flag_regular(1, 2), 3);
    }

    /// par_level_flag shares §9.3.4.2.9 with abs_level_gt_1_flag — same
    /// inputs must give identical ctxInc.
    #[test]
    fn par_level_flag_matches_gt1() {
        let a = ctx_inc_par_level_flag(0, 1, 2, 3, 3, 1, 2);
        let b = ctx_inc_abs_level_gt_1_flag(0, 1, 2, 3, 3, 1, 2);
        assert_eq!(a, b);
    }

    /// locNumSig / locSumAbsPass1 on a 4×4 TB with a single sig at
    /// (1, 0) (abs=1) and otherwise zero: at position (0, 0) we look at
    /// (1,0),(2,0),(1,1),(0,1),(0,2) — only (1,0) is sig, sum = 1.
    #[test]
    fn loc_num_sig_and_sum_small_tb() {
        let mut sig = vec![false; 16];
        let mut a1 = vec![0u32; 16];
        sig[1] = true; // (1,0)
        a1[1] = 1;
        let (n, s) = loc_num_sig_and_sum_abs_pass1(0, 0, &a1, &sig, 2, 2, 4);
        assert_eq!(n, 1);
        assert_eq!(s, 1);
    }

    /// locNumSig / locSumAbsPass1 at the right-edge column (xC = w-1)
    /// must not include (xC+1, *) lookups.
    #[test]
    fn loc_num_sig_right_edge_skips_forward_neighbour() {
        let sig = vec![true; 16];
        let a1 = vec![2u32; 16];
        // (3, 0) on a 4×4 TB: no xC+1 branch; y-branches only read
        // (3, 1), (3, 2) → sum = 4, count = 2.
        let (n, s) = loc_num_sig_and_sum_abs_pass1(3, 0, &a1, &sig, 2, 2, 4);
        assert_eq!(n, 2);
        assert_eq!(s, 4);
    }

    /// Table 128 Rice-parameter lookup.
    #[test]
    fn rice_param_table_128() {
        assert_eq!(rice_param_from_loc_sum_abs(0), 0);
        assert_eq!(rice_param_from_loc_sum_abs(6), 0);
        assert_eq!(rice_param_from_loc_sum_abs(7), 1);
        assert_eq!(rice_param_from_loc_sum_abs(13), 1);
        assert_eq!(rice_param_from_loc_sum_abs(14), 2);
        assert_eq!(rice_param_from_loc_sum_abs(27), 2);
        assert_eq!(rice_param_from_loc_sum_abs(28), 3);
        assert_eq!(rice_param_from_loc_sum_abs(31), 3);
    }

    /// locSumAbsRice on an all-zero neighbourhood with hist_value = 0
    /// returns 0 even at the zo-edges (all fallback values are 0).
    #[test]
    fn loc_sum_abs_rice_zero_neighbourhood() {
        let a = vec![0u32; 16];
        assert_eq!(loc_sum_abs_rice(3, 3, &a, 2, 2, 4, 0), 0);
        assert_eq!(loc_sum_abs_rice(0, 0, &a, 2, 2, 4, 0), 0);
    }
}
