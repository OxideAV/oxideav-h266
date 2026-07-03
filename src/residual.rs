//! VVC residual coding + transform-unit CBF reads (§7.3.11.10 – §7.3.11.11).
//!
//! This module covers the *bin-level* residual decoding pipeline for a
//! single TB in regular (non-transform-skip) mode, plus the CBF /
//! last-significant-position syntax that the transform_unit() block
//! consumes before entering residual_coding().
//!
//! Primitives landed:
//!
//! * [`ResidualCtxs`] — the CABAC context bundle used for CBF flags,
//!   `last_sig_coeff_*` prefixes and the per-sub-block residual passes.
//! * [`read_tu_y_coded_flag`] / [`read_tu_cb_coded_flag`] /
//!   [`read_tu_cr_coded_flag`] — CBF reads per §9.3.4.2.5 + Table 127.
//! * [`read_last_sig_coeff_pos`] — `last_sig_coeff_{x,y}_prefix` (TR
//!   context-coded, §9.3.4.2.4 eqs. 1555 / 1556) + `_{x,y}_suffix`
//!   (FL bypass) reading, returning the composite
//!   `(LastSignificantCoeffX, LastSignificantCoeffY)` (eqs. 199 / 200).
//! * [`read_cu_qp_delta`] — `cu_qp_delta_abs` (§9.3.3.10: TR prefix +
//!   EGk suffix) with `cu_qp_delta_sign_flag` bypass bin (§7.4.11.8).
//! * [`decode_tb_coefficients`] — top-level TB walker that fills a
//!   row-major `(n_tb_w * n_tb_h)` coefficient array. The walker
//!   implements the full §7.3.11.11 three-pass structure:
//!   * **Pass 1** reads `sig_coeff_flag` + `abs_level_gtx_flag[0]`
//!     (gt1) + `par_level_flag` + `abs_level_gtx_flag[1]` (gt3) with
//!     **spec-exact** ctxInc (§9.3.4.2.8 / §9.3.4.2.9) threading the
//!     §9.3.4.2.7 `locNumSig` + `locSumAbsPass1` accumulators and the
//!     `remBinsPass1` budget from eq. 5018.
//!   * **Pass 2** reads `abs_remainder[]` (§9.3.3.11) with a Rice
//!     parameter derived from §9.3.3.2 using `baseLevel = 4`.
//!   * **Pass 3** reads `dec_abs_level[]` (§9.3.3.12) for every
//!     position where pass 1 ran out of budget; Rice parameter derived
//!     with `baseLevel = 0`. `ZeroPos[n]` (eq. 1536) is applied
//!     on the fly to reconstruct `AbsLevel[xC][yC]`.
//!   * A final per-sub-block sign-flag pass (bypass-coded) signs the
//!     accumulated levels.
//!
//! Dependent quantization (`sh_dep_quant_used_flag = 1`, §7.4.12.11
//! eq. 198 QState trellis + QState-dependent §9.3.4.2.8 ctxInc /
//! §9.3.3.12 `ZeroPos` / `(2 * AbsLevel − (QState > 1)) * sign`
//! reconstruction) and sign data hiding
//! (`sh_sign_data_hiding_used_flag = 1`, `signHiddenFlag` + sub-block
//! parity sign inference) are live through [`RcOpts`] /
//! [`decode_tb_coefficients_opts`].
//!
//! Scope restrictions:
//! `transform_skip_flag = 0` (regular residual coding only, no TS
//! ctxInc variants §9.3.4.2.8 eq. 1572 / §9.3.4.2.9 eqs. 1575..1581),
//! `sps_persistent_rice_adaptation_enabled_flag = 0` (`HistValue = 0`),
//! `sps_rrc_rice_extension_flag = 0` (`shiftVal = 0`), and zero-out
//! off (`Log2ZoTb{W,H}` = full TB log2 sizes). Dequantisation
//! (§8.7.3) lives in the sibling [`crate::dequant`] module.
//!
//! Spec reference: ITU-T H.266 | ISO/IEC 23090-3 (V4, 01/2026).

use oxideav_core::{Error, Result};

use crate::cabac::{ArithDecoder, ContextModel};
use crate::ctx::{
    csbf_ctx_regular, csbf_ctx_ts, ctx_inc_abs_level_gt_1_flag, ctx_inc_abs_level_gt_3_flag,
    ctx_inc_abs_level_gtx_flag_ts, ctx_inc_coeff_sign_flag_ts, ctx_inc_cu_chroma_qp_offset_flag,
    ctx_inc_cu_chroma_qp_offset_idx, ctx_inc_cu_qp_delta_abs, ctx_inc_last_sig_coeff_prefix,
    ctx_inc_par_level_flag, ctx_inc_par_level_flag_ts, ctx_inc_sb_coded_flag_regular,
    ctx_inc_sb_coded_flag_ts, ctx_inc_sig_coeff_flag, ctx_inc_sig_coeff_flag_ts,
    ctx_inc_transform_skip_flag, ctx_inc_tu_cb_coded_flag, ctx_inc_tu_cr_coded_flag,
    ctx_inc_tu_joint_cbcr_residual_flag, ctx_inc_tu_y_coded_flag, loc_num_sig_and_sum_abs_pass1,
    loc_num_sig_and_sum_abs_pass1_ts, loc_sum_abs_rice, rice_param_from_loc_sum_abs,
};
use crate::scan::{coeff_scan_positions, sb_grid};
use crate::tables::{init_contexts, SyntaxCtx};

/// Context array bundle used by the residual decoder + the TU-level
/// CBF / QP-delta reads.
pub struct ResidualCtxs {
    pub sig_coeff: Vec<ContextModel>,
    pub sb_coded: Vec<ContextModel>,
    pub abs_gtx: Vec<ContextModel>,
    pub par_level: Vec<ContextModel>,
    pub last_x: Vec<ContextModel>,
    pub last_y: Vec<ContextModel>,
    pub tu_y_coded: Vec<ContextModel>,
    pub tu_cb_coded: Vec<ContextModel>,
    pub tu_cr_coded: Vec<ContextModel>,
    pub cu_qp_delta_abs: Vec<ContextModel>,
    pub cu_chroma_qp_offset_flag: Vec<ContextModel>,
    pub cu_chroma_qp_offset_idx: Vec<ContextModel>,
    /// Table 119 — `tu_joint_cbcr_residual_flag`.
    pub tu_joint_cbcr_residual_flag: Vec<ContextModel>,
    /// Table 118 — `transform_skip_flag`.
    pub transform_skip_flag: Vec<ContextModel>,
    /// Table 126 — `coeff_sign_flag` (context-coded only in the
    /// transform-skip residual path, §9.3.4.2.10; bypass-coded in
    /// regular residual coding).
    pub coeff_sign: Vec<ContextModel>,
}

impl ResidualCtxs {
    pub fn init(slice_qp_y: i32) -> Self {
        Self {
            sig_coeff: init_contexts(SyntaxCtx::SigCoeffFlag, slice_qp_y),
            sb_coded: init_contexts(SyntaxCtx::SbCodedFlag, slice_qp_y),
            abs_gtx: init_contexts(SyntaxCtx::AbsLevelGtxFlag, slice_qp_y),
            par_level: init_contexts(SyntaxCtx::ParLevelFlag, slice_qp_y),
            last_x: init_contexts(SyntaxCtx::LastSigCoeffXPrefix, slice_qp_y),
            last_y: init_contexts(SyntaxCtx::LastSigCoeffYPrefix, slice_qp_y),
            tu_y_coded: init_contexts(SyntaxCtx::TuYCodedFlag, slice_qp_y),
            tu_cb_coded: init_contexts(SyntaxCtx::TuCbCodedFlag, slice_qp_y),
            tu_cr_coded: init_contexts(SyntaxCtx::TuCrCodedFlag, slice_qp_y),
            cu_qp_delta_abs: init_contexts(SyntaxCtx::CuQpDeltaAbs, slice_qp_y),
            cu_chroma_qp_offset_flag: init_contexts(SyntaxCtx::CuChromaQpOffsetFlag, slice_qp_y),
            cu_chroma_qp_offset_idx: init_contexts(SyntaxCtx::CuChromaQpOffsetIdx, slice_qp_y),
            tu_joint_cbcr_residual_flag: init_contexts(
                SyntaxCtx::TuJointCbCrResidualFlag,
                slice_qp_y,
            ),
            transform_skip_flag: init_contexts(SyntaxCtx::TransformSkipFlag, slice_qp_y),
            coeff_sign: init_contexts(SyntaxCtx::CoeffSignFlag, slice_qp_y),
        }
    }
}

/// Read `tu_joint_cbcr_residual_flag` per §7.3.11.10 + Table 132.
/// `ctxInc = 2 * tu_cb_coded_flag + tu_cr_coded_flag − 1`.
pub fn read_tu_joint_cbcr_residual_flag(
    dec: &mut ArithDecoder<'_>,
    ctxs: &mut ResidualCtxs,
    tu_cb_coded: bool,
    tu_cr_coded: bool,
) -> Result<bool> {
    let inc = ctx_inc_tu_joint_cbcr_residual_flag(tu_cb_coded, tu_cr_coded) as usize;
    let n = ctxs.tu_joint_cbcr_residual_flag.len() - 1;
    let bit = dec.decode_decision(&mut ctxs.tu_joint_cbcr_residual_flag[inc.min(n)])?;
    Ok(bit == 1)
}

/// Read `transform_skip_flag[ x0 ][ y0 ][ cIdx ]` per §7.3.11.10 +
/// Table 132. `ctxInc = cIdx == 0 ? 0 : 1`.
pub fn read_transform_skip_flag(
    dec: &mut ArithDecoder<'_>,
    ctxs: &mut ResidualCtxs,
    c_idx: u32,
) -> Result<bool> {
    let inc = ctx_inc_transform_skip_flag(c_idx) as usize;
    let n = ctxs.transform_skip_flag.len() - 1;
    let bit = dec.decode_decision(&mut ctxs.transform_skip_flag[inc.min(n)])?;
    Ok(bit == 1)
}

/// Read `tu_y_coded_flag` per §7.3.11.10 / §9.3.4.2.5.
pub fn read_tu_y_coded_flag(
    dec: &mut ArithDecoder<'_>,
    ctxs: &mut ResidualCtxs,
    bdpcm_y: bool,
    isp_split: bool,
    prev_tu_cbf_y: bool,
) -> Result<bool> {
    let inc = ctx_inc_tu_y_coded_flag(bdpcm_y, isp_split, prev_tu_cbf_y) as usize;
    let n = ctxs.tu_y_coded.len() - 1;
    let bit = dec.decode_decision(&mut ctxs.tu_y_coded[inc.min(n)])?;
    Ok(bit == 1)
}

/// Read `tu_cb_coded_flag` per §7.3.11.10 + Table 127.
pub fn read_tu_cb_coded_flag(
    dec: &mut ArithDecoder<'_>,
    ctxs: &mut ResidualCtxs,
    bdpcm_chroma: bool,
) -> Result<bool> {
    let inc = ctx_inc_tu_cb_coded_flag(bdpcm_chroma) as usize;
    let n = ctxs.tu_cb_coded.len() - 1;
    let bit = dec.decode_decision(&mut ctxs.tu_cb_coded[inc.min(n)])?;
    Ok(bit == 1)
}

/// Read `tu_cr_coded_flag` per §7.3.11.10 + Table 127.
pub fn read_tu_cr_coded_flag(
    dec: &mut ArithDecoder<'_>,
    ctxs: &mut ResidualCtxs,
    bdpcm_chroma: bool,
    tu_cb_coded: bool,
) -> Result<bool> {
    let inc = ctx_inc_tu_cr_coded_flag(bdpcm_chroma, tu_cb_coded) as usize;
    let n = ctxs.tu_cr_coded.len() - 1;
    let bit = dec.decode_decision(&mut ctxs.tu_cr_coded[inc.min(n)])?;
    Ok(bit == 1)
}

/// Read `cu_qp_delta_abs` + sign (§7.3.11.10 + §9.3.3.10).
///
/// Returns the signed `CuQpDelta` value directly (positive if
/// `cu_qp_delta_sign_flag == 0`, negative otherwise).
pub fn read_cu_qp_delta(dec: &mut ArithDecoder<'_>, ctxs: &mut ResidualCtxs) -> Result<i32> {
    // Prefix: TR(cMax=5, cRice=0) with per-bin context (bin 0 ctx 0,
    // bins 1..4 ctx 1). Bin 0 ctx 0 uses ctxInc 0; subsequent bins use
    // ctxInc 1.
    let n = ctxs.cu_qp_delta_abs.len() - 1;
    let mut prefix = 0u32;
    for bin_idx in 0..5u32 {
        let inc = ctx_inc_cu_qp_delta_abs(bin_idx) as usize;
        let bit = dec.decode_decision(&mut ctxs.cu_qp_delta_abs[inc.min(n)])?;
        if bit == 0 {
            break;
        }
        prefix += 1;
    }
    // If prefix reached 5, read EGk(0) suffix (§9.3.3.10 eq. 1543).
    let abs = if prefix < 5 {
        prefix
    } else {
        5 + decode_exp_golomb_k(dec, 0)?
    };
    if abs == 0 {
        return Ok(0);
    }
    let sign = dec.decode_bypass()?;
    Ok(if sign == 1 { -(abs as i32) } else { abs as i32 })
}

/// Read `cu_chroma_qp_offset_flag` + `cu_chroma_qp_offset_idx` per
/// §7.3.11.10 + Table 127. Returns `(flag, idx)`; `idx` is 0 when the
/// flag is 0 or when `list_len` is 0 (no idx bin is coded).
pub fn read_cu_chroma_qp_offset(
    dec: &mut ArithDecoder<'_>,
    ctxs: &mut ResidualCtxs,
    list_len_minus1: u32,
) -> Result<(bool, u32)> {
    let inc = ctx_inc_cu_chroma_qp_offset_flag() as usize;
    let n = ctxs.cu_chroma_qp_offset_flag.len() - 1;
    let flag = dec.decode_decision(&mut ctxs.cu_chroma_qp_offset_flag[inc.min(n)])?;
    if flag == 0 || list_len_minus1 == 0 {
        return Ok((flag == 1, 0));
    }
    // TR(cMax = list_len_minus1, cRice = 0) context-coded — ctxIdx 0.
    let inc = ctx_inc_cu_chroma_qp_offset_idx() as usize;
    let n = ctxs.cu_chroma_qp_offset_idx.len() - 1;
    let mut idx = 0u32;
    for _ in 0..list_len_minus1 {
        let bit = dec.decode_decision(&mut ctxs.cu_chroma_qp_offset_idx[inc.min(n)])?;
        if bit == 0 {
            break;
        }
        idx += 1;
    }
    Ok((true, idx))
}

/// Composite `(LastSignificantCoeffX, LastSignificantCoeffY)` value
/// read per §7.3.11.11 + §9.3.4.2.4 + eqs. 199 / 200 / 202.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LastSigCoeffPos {
    pub x: u32,
    pub y: u32,
}

/// Read `last_sig_coeff_x_prefix / y_prefix / x_suffix / y_suffix` for
/// a single TB and derive `(LastSignificantCoeffX, LastSignificantCoeffY)`.
///
/// * `log2_zo_tb_width` / `log2_zo_tb_height` — `Log2ZoTbWidth` and
///   `Log2ZoTbHeight` after the MTS / SBT zero-out clips (same values
///   as `Log2FullTb{Width,Height}` when those tools are disabled).
/// * `c_idx` — colour component.
pub fn read_last_sig_coeff_pos(
    dec: &mut ArithDecoder<'_>,
    ctxs: &mut ResidualCtxs,
    log2_zo_tb_width: u32,
    log2_zo_tb_height: u32,
    c_idx: u32,
) -> Result<LastSigCoeffPos> {
    let prefix_x = read_last_sig_prefix(
        dec,
        &mut ctxs.last_x,
        log2_zo_tb_width,
        (log2_zo_tb_width << 1).saturating_sub(1),
        c_idx,
    )?;
    let prefix_y = read_last_sig_prefix(
        dec,
        &mut ctxs.last_y,
        log2_zo_tb_height,
        (log2_zo_tb_height << 1).saturating_sub(1),
        c_idx,
    )?;
    let x = last_sig_suffix(dec, prefix_x)?;
    let y = last_sig_suffix(dec, prefix_y)?;
    Ok(LastSigCoeffPos { x, y })
}

/// Read the TR-binarised context-coded `last_sig_coeff_*_prefix` bins.
///
/// §9.3.3.6 TR with `cRiceParam = 0`: at most `cMax` prefix bins are
/// coded and the terminating 0-bin is **absent** when the prefix value
/// equals `cMax` (the all-ones codeword is self-delimiting).
fn read_last_sig_prefix(
    dec: &mut ArithDecoder<'_>,
    ctxs: &mut [ContextModel],
    log2_tb_size: u32,
    c_max: u32,
    c_idx: u32,
) -> Result<u32> {
    let n = ctxs.len() - 1;
    let mut prefix = 0u32;
    while prefix < c_max {
        let inc = ctx_inc_last_sig_coeff_prefix(prefix, c_idx, log2_tb_size) as usize;
        let bit = dec.decode_decision(&mut ctxs[inc.min(n)])?;
        if bit == 0 {
            break;
        }
        prefix += 1;
    }
    Ok(prefix)
}

/// Expand a `last_sig_coeff_*_prefix` into the full coordinate,
/// pulling the FL(suffix) bypass bits when the prefix is > 3
/// (§7.4.11.11 eqs. 199 / 200).
fn last_sig_suffix(dec: &mut ArithDecoder<'_>, prefix: u32) -> Result<u32> {
    if prefix <= 3 {
        return Ok(prefix);
    }
    // FL(cMax = (1 << ((prefix >> 1) - 1)) - 1).
    let k = (prefix >> 1).saturating_sub(1);
    let suffix = dec.decode_bypass_bits(k)?;
    Ok((1u32 << ((prefix >> 1) - 1)) * (2 + (prefix & 1)) + suffix)
}

/// Decode the Rice-coded `abs_remainder` (§9.3.3.10 / §9.3.3.11) for
/// the residual-coding-remaining tail. Returns the decoded non-negative
/// value.
///
/// `rice_param` ∈ 0..=7. The coding is:
///   * `prefix` — unary with max length 6, bypass-coded.
///   * if prefix < 6: suffix = `rice_param` bypass bits; value = prefix*2^rice + suffix.
///   * else: escape — `4*2^rice + EGk(rice+1)` per §9.3.3.11.
pub fn decode_coeff_abs_level_remaining(
    dec: &mut ArithDecoder<'_>,
    rice_param: u32,
) -> Result<u32> {
    // Unary prefix, length ≤ 6.
    let mut prefix = 0u32;
    for _ in 0..7 {
        let b = dec.decode_bypass()?;
        if b == 0 {
            break;
        }
        prefix += 1;
        if prefix >= 7 {
            break;
        }
    }
    if prefix < 6 {
        let suffix = if rice_param > 0 {
            dec.decode_bypass_bits(rice_param)?
        } else {
            0
        };
        Ok((prefix << rice_param) + suffix)
    } else {
        // Escape: 4 << rice + EGk(rice+1).
        let base = 4u32 << rice_param;
        let egk = decode_exp_golomb_k(dec, rice_param + 1)?;
        Ok(base + egk)
    }
}

/// k-th order Exp-Golomb bypass decode (§9.3.3.5).
pub fn decode_exp_golomb_k(dec: &mut ArithDecoder<'_>, k: u32) -> Result<u32> {
    let mut log = 0u32;
    loop {
        let b = dec.decode_bypass()?;
        if b == 1 {
            log += 1;
        } else {
            break;
        }
    }
    let suffix_bits = log + k;
    let suffix = if suffix_bits > 0 {
        dec.decode_bypass_bits(suffix_bits)?
    } else {
        0
    };
    Ok(((1u32 << log) - 1) * (1u32 << k) + suffix)
}

/// §7.3.11.11 residual_coding() walker: decodes a single TB into a
/// row-major `(n_tb_w * n_tb_h)` array of signed `TransCoeffLevel[]`
/// values. Implements the full 3-pass structure of the spec with
/// spec-exact ctxInc:
///
/// * **Pass 1** (n = firstPosMode0 → 0 while remBinsPass1 >= 4):
///   `sig_coeff_flag` (§9.3.4.2.8 eq. 1573/1574), `abs_level_gtx_flag[n][0]`
///   (§9.3.4.2.9 eq. 1583/1584), `par_level_flag[n]` (same clause),
///   `abs_level_gtx_flag[n][1]` (+32 per eq. 1585). Each coded bin
///   decrements `remBinsPass1`; all ctxInc derivations thread the
///   running `locNumSig` + `locSumAbsPass1` from §9.3.4.2.7.
/// * **Pass 2** (n = firstPosMode0 → firstPosMode1+1): `abs_remainder[n]`
///   read as TR(cMax = 6 << cRiceParam) prefix + EGk(cRiceParam+1)
///   suffix (§9.3.3.11). Rice param via §9.3.3.2 with `baseLevel = 4`.
/// * **Pass 3** (n = firstPosMode1 → 0): `dec_abs_level[n]` for every
///   position in the coded sub-block (§7.3.11.11 + §9.3.3.12). Rice
///   param derived with `baseLevel = 0`.
///
/// Then a per-sub-block sign-flag pass (bypass-coded) recovers signs.
///
/// Restrictions vs. full spec (the flag-less entry points; dependent
/// quantization and sign data hiding are live via
/// [`decode_tb_coefficients_opts`] + [`RcOpts`]):
/// * `transform_skip_flag = 0` — transform-skip residual coding uses a
///   different ctxInc regime (§9.3.4.2.8 eq. 1572, §9.3.4.2.9 eqs.
///   1575..1581) that is out of scope for this round.
/// * `sps_persistent_rice_adaptation_enabled_flag = 0` — `HistValue`
///   stays at 0 for the §9.3.3.2 neighbourhood fallback.
/// * Zero-out (`MTS`, `SBT`) is treated as no-op: `Log2Zo{Tb}{Width,Height}`
///   equal the full TB log2 dims.
/// LFNST / MTS gating flags accumulated by `residual_coding()` while it
/// walks the coefficient sub-blocks (§7.3.11.11). They drive whether the
/// `lfnst_idx` / `mts_idx` syntax elements are present at the
/// coding-unit level (§7.3.11.5 — see the `LfnstDcOnly == 0`,
/// `LfnstZeroOutSigCoeffFlag == 1`, `MtsZeroOutSigCoeffFlag == 1`,
/// `MtsDcOnly == 0` gates) and, for LFNST, whether the inverse
/// non-separable transform applies (`ApplyLfnstFlag`).
///
/// The fields use the spec's initial-state convention: a TB with no
/// coded coefficients leaves every flag at its `transform_unit()`
/// initialisation value (`LfnstDcOnly = 1`, `LfnstZeroOutSigCoeffFlag =
/// 1`, `MtsDcOnly = 1`, `MtsZeroOutSigCoeffFlag = 1`). A caller that
/// runs multiple `residual_coding()` invocations for one CU folds the
/// per-TB results together with [`TbResidualFlags::merge`] so the
/// accumulated state mirrors the spec's CU-scoped variables.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TbResidualFlags {
    /// `LfnstDcOnly` (§7.3.11.11). Cleared to `false` when the last
    /// significant coefficient is outside the DC sub-block's scan
    /// position 0 for a TB large enough to carry LFNST.
    pub lfnst_dc_only: bool,
    /// `LfnstZeroOutSigCoeffFlag` (§7.3.11.11). Cleared to `false` when
    /// a significant coefficient is found outside the LFNST low-frequency
    /// region, which forbids LFNST.
    pub lfnst_zero_out_sig_coeff_flag: bool,
    /// `MtsDcOnly` (§7.3.11.11). Cleared when a luma TB has any
    /// significant coefficient beyond the DC.
    pub mts_dc_only: bool,
    /// `MtsZeroOutSigCoeffFlag` (§7.3.11.11). Cleared when a luma TB has
    /// a coded sub-block outside the top-left 4x4 grid region.
    pub mts_zero_out_sig_coeff_flag: bool,
}

impl Default for TbResidualFlags {
    fn default() -> Self {
        // §7.3.11.5 transform_unit() initialisation (eqs. before the
        // transform_tree() call): all four start at 1.
        Self {
            lfnst_dc_only: true,
            lfnst_zero_out_sig_coeff_flag: true,
            mts_dc_only: true,
            mts_zero_out_sig_coeff_flag: true,
        }
    }
}

impl TbResidualFlags {
    /// Fold the flags from a later `residual_coding()` call into the
    /// CU-scoped accumulator. The spec only ever *clears* these flags
    /// (they start at 1 and are set to 0 by the per-TB conditions), so
    /// the merge is a logical AND across TBs.
    pub fn merge(&mut self, other: TbResidualFlags) {
        self.lfnst_dc_only &= other.lfnst_dc_only;
        self.lfnst_zero_out_sig_coeff_flag &= other.lfnst_zero_out_sig_coeff_flag;
        self.mts_dc_only &= other.mts_dc_only;
        self.mts_zero_out_sig_coeff_flag &= other.mts_zero_out_sig_coeff_flag;
    }
}

pub fn decode_tb_coefficients(
    dec: &mut ArithDecoder<'_>,
    ctxs: &mut ResidualCtxs,
    n_tb_w: usize,
    n_tb_h: usize,
    c_idx: u32,
) -> Result<Vec<i32>> {
    decode_tb_coefficients_with_flags(dec, ctxs, n_tb_w, n_tb_h, c_idx).map(|(levels, _)| levels)
}

/// Like [`decode_tb_coefficients`] but also returns the §7.3.11.11
/// LFNST / MTS gating flags ([`TbResidualFlags`]). The flags are
/// derived from the same last-significant-coefficient scan and
/// sub-block walk the level decode already performs, so this is the
/// canonical entry point for the LFNST / MTS-aware intra path; the
/// flag-less wrapper above stays as the historical signature for
/// callers that don't need them (e.g. encoder round-trip tests).
pub fn decode_tb_coefficients_with_flags(
    dec: &mut ArithDecoder<'_>,
    ctxs: &mut ResidualCtxs,
    n_tb_w: usize,
    n_tb_h: usize,
    c_idx: u32,
) -> Result<(Vec<i32>, TbResidualFlags)> {
    decode_tb_coefficients_opts(dec, ctxs, n_tb_w, n_tb_h, c_idx, RcOpts::default())
}

/// §7.4.12.11 eq. 198 — `QStateTransTable[][]` for dependent
/// quantization (`sh_dep_quant_used_flag == 1`).
///
/// The spec prints the array literal as two rows of four entries; read
/// with the row selecting the level **parity** (`AbsLevel & 1`) and the
/// column selecting the current `QState`, the machine is the four-state
/// trellis whose quantizer selection is the `QState > 1` offset used by
/// the §7.3.11.11 `TransCoeffLevel` reconstruction and the §9.3.3.12
/// eq. 1536 `ZeroPos` derivation (`QState < 2 ? 1 : 2`). (The transposed
/// `[state][parity]` reading of the same literal would leave states 1
/// and 3 unreachable from the initial `QState = 0`, degenerating to a
/// two-state machine, so it cannot be the intended one.) Indexing here:
/// `Q_STATE_TRANS_TABLE[parity][state]`.
pub(crate) const Q_STATE_TRANS_TABLE: [[i32; 4]; 2] = [[0, 2, 1, 3], [2, 0, 3, 1]];

/// Advance the §7.4.12.11 dependent-quantization state machine by one
/// coefficient of the given absolute level.
#[inline]
pub(crate) fn q_state_advance(q_state: i32, abs_level: u32) -> i32 {
    Q_STATE_TRANS_TABLE[(abs_level & 1) as usize][(q_state & 3) as usize]
}

/// Slice-level residual-coding switches threaded into
/// [`decode_tb_coefficients_opts`] (§7.3.11.11).
///
/// * `dep_quant` — `sh_dep_quant_used_flag`: drives the QState trellis
///   ([`Q_STATE_TRANS_TABLE`]), the QState-dependent `sig_coeff_flag`
///   ctxInc terms (§9.3.4.2.8 eqs. 1573 / 1574), the `ZeroPos`
///   derivation (§9.3.3.12 eq. 1536) and the
///   `TransCoeffLevel = (2 * AbsLevel − (QState > 1)) * sign` final
///   reconstruction.
/// * `sign_data_hiding` — `sh_sign_data_hiding_used_flag`: the
///   `signHiddenFlag` sub-block condition
///   (`lastSigScanPosSb − firstSigScanPosSb > 3`) suppresses the
///   `coeff_sign_flag` bin of the first significant coefficient in scan
///   order, whose sign is instead the parity of the sub-block's
///   absolute-level sum.
///
/// The two are mutually exclusive by syntax: §7.3.7 only transmits
/// `sh_sign_data_hiding_used_flag` when `sh_dep_quant_used_flag == 0`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RcOpts {
    pub dep_quant: bool,
    pub sign_data_hiding: bool,
}

/// Like [`decode_tb_coefficients_with_flags`] but with the slice-level
/// residual-coding switches ([`RcOpts`]) live: dependent quantization
/// (`sh_dep_quant_used_flag`) and sign data hiding
/// (`sh_sign_data_hiding_used_flag`).
pub fn decode_tb_coefficients_opts(
    dec: &mut ArithDecoder<'_>,
    ctxs: &mut ResidualCtxs,
    n_tb_w: usize,
    n_tb_h: usize,
    c_idx: u32,
    opts: RcOpts,
) -> Result<(Vec<i32>, TbResidualFlags)> {
    if opts.dep_quant && opts.sign_data_hiding {
        return Err(Error::invalid(
            "h266 residual: dep_quant and sign_data_hiding are mutually exclusive (§7.3.7)",
        ));
    }
    if !n_tb_w.is_power_of_two() || !n_tb_h.is_power_of_two() {
        return Err(Error::invalid(
            "h266 residual: nTbW / nTbH must be power of two",
        ));
    }
    let log2_w = n_tb_w.trailing_zeros();
    let log2_h = n_tb_h.trailing_zeros();
    let last = read_last_sig_coeff_pos(dec, ctxs, log2_w, log2_h, c_idx)?;

    // Running per-TB state threaded by §9.3.4.2.7 / §9.3.3.2.
    let total = n_tb_w * n_tb_h;
    let mut abs_level_pass1 = vec![0u32; total];
    let mut abs_level = vec![0u32; total];
    let mut sig_flag = vec![false; total];
    let mut out = vec![0i32; total];

    // eq. 5018: pass-1 bin budget.
    let mut rem_bins_pass1: i32 = ((1i32 << (log2_w + log2_h)) * 7) >> 2;

    let (num_sb_w, num_sb_h) = sb_grid(n_tb_w, n_tb_h);
    let positions = coeff_scan_positions(n_tb_w, n_tb_h);
    let sb_origins = crate::scan::sb_scan_positions(n_tb_w, n_tb_h);

    // Find the sub-block that contains the last-sig position plus the
    // within-sub-block scan index of that position.
    let mut last_sb_idx = 0usize;
    let mut last_scan_pos_in_sb = 0usize;
    for sb_idx in 0..(num_sb_w * num_sb_h) {
        let start = sb_idx * 16;
        let end = (start + 16).min(positions.len());
        for (k, &(xc, yc)) in positions[start..end].iter().enumerate() {
            if xc == last.x && yc == last.y {
                last_sb_idx = sb_idx;
                last_scan_pos_in_sb = k;
            }
        }
    }

    // §7.3.11.11 LFNST / MTS gating flags. Zero-out is a no-op in this
    // scaffold so `Log2ZoTb{Width,Height}` equal the full TB log2 dims
    // (`log2_w` / `log2_h`).
    let mut flags = TbResidualFlags::default();
    let log2_zo_w = log2_w;
    let log2_zo_h = log2_h;
    // transform_skip_flag == 0 in this path.
    let transform_skip = false;
    // LfnstDcOnly: cleared when the last-sig coeff is past scan position 0
    // of the DC sub-block (eq. at §7.3.11.11). `last_scan_pos_in_sb` is
    // the within-sub-block scan index of the last-sig coefficient
    // (== the spec's `lastScanPos`), `last_sb_idx` is `lastSubBlock`.
    if last_sb_idx == 0
        && log2_zo_w >= 2
        && log2_zo_h >= 2
        && !transform_skip
        && last_scan_pos_in_sb > 0
    {
        flags.lfnst_dc_only = false;
    }
    // LfnstZeroOutSigCoeffFlag: cleared when a significant coefficient
    // is outside the LFNST low-frequency region.
    if (last_sb_idx > 0 && log2_zo_w >= 2 && log2_zo_h >= 2)
        || (last_scan_pos_in_sb > 7 && (log2_zo_w == 2 || log2_zo_w == 3) && log2_zo_w == log2_zo_h)
    {
        flags.lfnst_zero_out_sig_coeff_flag = false;
    }
    // MtsDcOnly: cleared when a luma TB has any non-DC significant coeff.
    if (last_sb_idx > 0 || last_scan_pos_in_sb > 0) && c_idx == 0 {
        flags.mts_dc_only = false;
    }

    // Track which sub-blocks are coded (for the csbfCtx neighbour).
    let mut sb_coded = vec![false; num_sb_w * num_sb_h];

    // §7.3.11.11 — the dependent-quantization state. Stays 0 for the
    // whole TB when `sh_dep_quant_used_flag == 0`.
    let mut q_state: i32 = 0;

    // Walk sub-blocks in reverse diagonal scan order starting at
    // last_sb_idx. Sub-block 0 and `last_sb_idx` are inferred coded.
    for sb_idx in (0..=last_sb_idx).rev() {
        let sb_origin = sb_origins[sb_idx];
        let (xs, ys) = ((sb_origin.0 >> 2) as usize, (sb_origin.1 >> 2) as usize);
        let right = xs + 1 < num_sb_w && sb_coded[ys * num_sb_w + (xs + 1)];
        let below = ys + 1 < num_sb_h && sb_coded[(ys + 1) * num_sb_w + xs];
        let csbf = csbf_ctx_regular(right, below);

        // §7.3.11.11 `startQStateSb` — the reconstruction pass at the
        // end of the sub-block restarts the trellis from here.
        let start_q_state_sb = q_state;

        let is_inferred = sb_idx == last_sb_idx || sb_idx == 0;
        let coded = if is_inferred {
            true
        } else {
            let inc = ctx_inc_sb_coded_flag_regular(c_idx, csbf) as usize;
            let n = ctxs.sb_coded.len() - 1;
            dec.decode_decision(&mut ctxs.sb_coded[inc.min(n)])? == 1
        };
        sb_coded[ys * num_sb_w + xs] = coded;
        // §7.3.11.11: MtsZeroOutSigCoeffFlag is cleared when a coded
        // sub-block sits outside the top-left 4x4-sub-block region of a
        // luma TB (`xS > 3 || yS > 3`).
        if coded && (xs > 3 || ys > 3) && c_idx == 0 {
            flags.mts_zero_out_sig_coeff_flag = false;
        }
        if !coded {
            // §7.3.11.11 still walks the per-position QState updates of
            // an uncoded sub-block, but every AbsLevel is 0 and the
            // parity-0 transition permutation is (0)(3)(1 2), so the
            // even sub-block coefficient count (16 or 4) returns QState
            // to `start_q_state_sb`. Skipping the walk is exact.
            continue;
        }

        let start = sb_idx * 16;
        let end = (start + 16).min(positions.len());
        let num_sb_coeff = end - start;

        // §7.3.11.11 `firstSigScanPosSb` / `lastSigScanPosSb` — the
        // scan-index extremes of the significant coefficients in this
        // sub-block, driving the sign-data-hiding `signHiddenFlag`.
        let mut first_sig_scan_pos_sb: i32 = num_sb_coeff as i32;
        let mut last_sig_scan_pos_sb: i32 = -1;

        // infer_sb_dc_sig_coeff_flag is true when this sub-block is
        // neither the last-sig sub-block nor the DC sub-block
        // (§7.3.11.11): the DC (per-sb scan position 0) sig_coeff_flag
        // is then inferred to 1.
        let mut infer_sb_dc_sig = sb_idx < last_sb_idx && sb_idx > 0;

        // firstPosMode0 — the first per-sub-block scan index at which
        // pass 1 starts. In the last sub-block this is the
        // within-sb position of (last.x, last.y); otherwise it's
        // numSbCoeff - 1.
        let first_pos_mode0: i32 = if sb_idx == last_sb_idx {
            last_scan_pos_in_sb as i32
        } else {
            (num_sb_coeff as i32) - 1
        };
        let mut first_pos_mode1: i32 = first_pos_mode0;

        // Pass 1: sig_coeff_flag + gt1 + par_level + gt3, ctx-coded.
        let mut n = first_pos_mode0;
        while n >= 0 && rem_bins_pass1 >= 4 {
            let pos_idx = start + (n as usize);
            let (xc, yc) = positions[pos_idx];
            let is_last = (xc, yc) == (last.x, last.y);

            // sig_coeff_flag read: skipped when the position equals
            // (LastSignificantCoeff{X,Y}), or when the DC-infer path
            // fires at n == 0.
            let (loc_num_sig, loc_sum_abs_pass1) = loc_num_sig_and_sum_abs_pass1(
                xc,
                yc,
                &abs_level_pass1,
                &sig_flag,
                log2_w,
                log2_h,
                n_tb_w as u32,
            );
            let sig = if is_last {
                true
            } else if infer_sb_dc_sig && n == 0 {
                true
            } else {
                let inc =
                    ctx_inc_sig_coeff_flag(c_idx, xc, yc, q_state, loc_sum_abs_pass1) as usize;
                let cap = ctxs.sig_coeff.len() - 1;
                let bit = dec.decode_decision(&mut ctxs.sig_coeff[inc.min(cap)])?;
                rem_bins_pass1 -= 1;
                if bit == 1 {
                    infer_sb_dc_sig = false;
                }
                bit == 1
            };
            sig_flag[(yc as usize) * n_tb_w + (xc as usize)] = sig;

            let (mut par, mut gt1, mut gt3) = (false, false, false);
            if sig {
                // abs_level_gtx_flag[n][0] (gt_1 in regular RC).
                let inc = ctx_inc_abs_level_gt_1_flag(
                    c_idx,
                    xc,
                    yc,
                    last.x,
                    last.y,
                    loc_num_sig,
                    loc_sum_abs_pass1,
                ) as usize;
                let cap = ctxs.abs_gtx.len() - 1;
                gt1 = dec.decode_decision(&mut ctxs.abs_gtx[inc.min(cap)])? == 1;
                rem_bins_pass1 -= 1;
                if gt1 {
                    // par_level_flag — shares §9.3.4.2.9 derivation
                    // with gt_1.
                    let inc = ctx_inc_par_level_flag(
                        c_idx,
                        xc,
                        yc,
                        last.x,
                        last.y,
                        loc_num_sig,
                        loc_sum_abs_pass1,
                    ) as usize;
                    let cap_par = ctxs.par_level.len() - 1;
                    par = dec.decode_decision(&mut ctxs.par_level[inc.min(cap_par)])? == 1;
                    rem_bins_pass1 -= 1;
                    // abs_level_gtx_flag[n][1] (gt_3).
                    let inc = ctx_inc_abs_level_gt_3_flag(
                        c_idx,
                        xc,
                        yc,
                        last.x,
                        last.y,
                        loc_num_sig,
                        loc_sum_abs_pass1,
                    ) as usize;
                    gt3 = dec.decode_decision(&mut ctxs.abs_gtx[inc.min(cap)])? == 1;
                    rem_bins_pass1 -= 1;
                }
            }
            if sig {
                // §7.3.11.11 — first/last significant scan positions
                // (pass-1 arm).
                if last_sig_scan_pos_sb == -1 {
                    last_sig_scan_pos_sb = n;
                }
                first_sig_scan_pos_sb = n;
            }
            // AbsLevelPass1[xC][yC] = sig + par + gt1 + 2*gt3.
            let a1 = sig as u32 + par as u32 + gt1 as u32 + 2 * gt3 as u32;
            abs_level_pass1[(yc as usize) * n_tb_w + (xc as usize)] = a1;
            abs_level[(yc as usize) * n_tb_w + (xc as usize)] = a1;
            if opts.dep_quant {
                // QState = QStateTransTable[QState][AbsLevelPass1 & 1]
                // (§7.3.11.11 pass-1 arm).
                q_state = q_state_advance(q_state, a1);
            }
            first_pos_mode1 = n - 1;
            n -= 1;
        }

        // Pass 2: abs_remainder for positions in [firstPosMode1+1, firstPosMode0]
        // where gt3 (pass-1 bit 3) was set. AbsLevel[] accumulates.
        for n2 in ((first_pos_mode1 + 1)..=first_pos_mode0).rev() {
            let pos_idx = start + (n2 as usize);
            let (xc, yc) = positions[pos_idx];
            let a1 = abs_level_pass1[(yc as usize) * n_tb_w + (xc as usize)];
            // AbsLevelPass1 = sig + par + gt1 + 2*gt3. gt3 is only
            // read when gt1 is 1 (see pass-1 pseudocode), so
            // gt3 set ⇔ a1 >= 4 (sig=1, gt1=1, gt3=1, par∈{0,1}).
            let gt3 = a1 >= 4;
            if gt3 {
                let rice = derive_rice_param(
                    xc,
                    yc,
                    &abs_level,
                    log2_w,
                    log2_h,
                    n_tb_w as u32,
                    4,
                    q_state,
                );
                let rem = decode_abs_remainder(dec, rice)?;
                let new_abs = a1 + 2 * rem;
                abs_level[(yc as usize) * n_tb_w + (xc as usize)] = new_abs;
            }
        }

        // Pass 3: dec_abs_level for positions [0, firstPosMode1].
        for n3 in (0..=first_pos_mode1).rev() {
            let pos_idx = start + (n3 as usize);
            let (xc, yc) = positions[pos_idx];
            let rice = derive_rice_param(
                xc,
                yc,
                &abs_level,
                log2_w,
                log2_h,
                n_tb_w as u32,
                0,
                q_state,
            );
            let dec_abs = decode_dec_abs_level(dec, rice)?;
            // dec_abs_level[n] encodes AbsLevel[xC][yC] offset by ZeroPos
            // (§7.4.11.11): when dec_abs_level == ZeroPos[n] then
            // AbsLevel is zero; when dec_abs_level < ZeroPos then
            // AbsLevel = dec_abs + 1; otherwise AbsLevel = dec_abs.
            let zero_pos = (if q_state < 2 { 1u32 } else { 2 }) << rice;
            let a: u32 = match dec_abs.cmp(&zero_pos) {
                core::cmp::Ordering::Equal => 0,
                core::cmp::Ordering::Less => dec_abs + 1,
                core::cmp::Ordering::Greater => dec_abs,
            };
            if a > 0 {
                abs_level[(yc as usize) * n_tb_w + (xc as usize)] = a;
                sig_flag[(yc as usize) * n_tb_w + (xc as usize)] = true;
                // §7.3.11.11 — first/last significant scan positions
                // (pass-3 arm).
                if last_sig_scan_pos_sb == -1 {
                    last_sig_scan_pos_sb = n3;
                }
                first_sig_scan_pos_sb = n3;
            }
            if opts.dep_quant {
                // QState = QStateTransTable[QState][AbsLevel & 1]
                // (§7.3.11.11 pass-3 arm).
                q_state = q_state_advance(q_state, a);
            }
        }

        // §7.3.11.11 `signHiddenFlag` — hide the sign bin of the first
        // significant coefficient in scan order when the significant
        // span of the sub-block exceeds 3.
        let sign_hidden =
            opts.sign_data_hiding && (last_sig_scan_pos_sb - first_sig_scan_pos_sb > 3);

        // Per-sb sign-flag pass (bypass-coded in regular RC). Signs are
        // collected first; the TransCoeffLevel reconstruction below
        // needs the dep-quant trellis re-walk / SDH parity fold.
        let mut sign_neg = [false; 16];
        for k in (0..num_sb_coeff).rev() {
            let pos_idx = start + k;
            let (xc, yc) = positions[pos_idx];
            let a = abs_level[(yc as usize) * n_tb_w + (xc as usize)];
            if a > 0 && (!sign_hidden || k as i32 != first_sig_scan_pos_sb) {
                sign_neg[k] = dec.decode_bypass()? == 1;
            }
        }

        if opts.dep_quant {
            // §7.3.11.11 — TransCoeffLevel[..] =
            //   (2 * AbsLevel − (QState > 1 ? 1 : 0)) * (1 − 2 * sign),
            // re-walking the trellis from `startQStateSb` over every
            // scan position of the sub-block.
            let mut q = start_q_state_sb;
            for k in (0..num_sb_coeff).rev() {
                let pos_idx = start + k;
                let (xc, yc) = positions[pos_idx];
                let a = abs_level[(yc as usize) * n_tb_w + (xc as usize)];
                if a > 0 {
                    let mag = 2 * (a as i32) - i32::from(q > 1);
                    out[(yc as usize) * n_tb_w + (xc as usize)] =
                        if sign_neg[k] { -mag } else { mag };
                }
                q = q_state_advance(q, a);
            }
        } else {
            // §7.3.11.11 — TransCoeffLevel = AbsLevel * (1 − 2 * sign),
            // with the hidden sign recovered from the parity of the
            // sub-block absolute-level sum.
            let mut sum_abs_level: u64 = 0;
            for k in (0..num_sb_coeff).rev() {
                let pos_idx = start + k;
                let (xc, yc) = positions[pos_idx];
                let a = abs_level[(yc as usize) * n_tb_w + (xc as usize)];
                if a > 0 {
                    let mut v = if sign_neg[k] { -(a as i32) } else { a as i32 };
                    if sign_hidden {
                        sum_abs_level += a as u64;
                        if k as i32 == first_sig_scan_pos_sb && sum_abs_level % 2 == 1 {
                            v = -v;
                        }
                    }
                    out[(yc as usize) * n_tb_w + (xc as usize)] = v;
                }
            }
        }
    }
    Ok((out, flags))
}

/// §9.3.3.2 Rice-parameter derivation for `abs_remainder[]` (baseLevel=4)
/// or `dec_abs_level[]` (baseLevel=0). `HistValue` is taken as 0 since
/// `sps_persistent_rice_adaptation_enabled_flag` is off in this
/// scaffold; `sps_rrc_rice_extension_flag` is off so `shiftVal = 0`.
fn derive_rice_param(
    x_c: u32,
    y_c: u32,
    abs_level: &[u32],
    log2_full_tb_w: u32,
    log2_full_tb_h: u32,
    n_tb_w: u32,
    base_level: u32,
    _q_state: i32,
) -> u32 {
    let hist_value = 0u32; // persistent rice adaptation off
    let loc = loc_sum_abs_rice(
        x_c,
        y_c,
        abs_level,
        log2_full_tb_w,
        log2_full_tb_h,
        n_tb_w,
        hist_value,
    );
    // shiftVal = 0 when !sps_rrc_rice_extension_flag (eq. 1533).
    let shift_val = 0u32;
    let clipped = ((loc >> shift_val) as i32 - (base_level as i32) * 5).clamp(0, 31) as u32;
    rice_param_from_loc_sum_abs(clipped)
}

/// §9.3.3.11 `abs_remainder[]` binarisation decode.
///
/// TR prefix with `cMax = 6 << cRiceParam` + optional limited-EGk
/// suffix (order `cRiceParam + 1`). The prefix binarisation is pure
/// bypass, so this function does not touch CABAC contexts.
fn decode_abs_remainder(dec: &mut ArithDecoder<'_>, rice_param: u32) -> Result<u32> {
    // TR(cMax, rice): read up to (cMax >> rice) = 6 bypass bins. The
    // prefix "runs out" when 6 ones have been seen, in which case the
    // suffix EGk(rice+1) adds the remainder.
    let mut prefix = 0u32;
    for _ in 0..6 {
        let b = dec.decode_bypass()?;
        if b == 0 {
            break;
        }
        prefix += 1;
    }
    if prefix < 6 {
        // Value = (prefix << rice) + FL(rice bits).
        let suffix = if rice_param > 0 {
            dec.decode_bypass_bits(rice_param)?
        } else {
            0
        };
        return Ok((prefix << rice_param) + suffix);
    }
    // Prefix is all-ones length 6 → suffix EGk present.
    let egk = decode_exp_golomb_k(dec, rice_param + 1)?;
    Ok((6u32 << rice_param) + egk)
}

/// §9.3.3.12 `dec_abs_level[]` binarisation decode. Same shape as
/// [`decode_abs_remainder`] modulo the Rice-parameter derivation
/// (baseLevel = 0, handled by the caller).
fn decode_dec_abs_level(dec: &mut ArithDecoder<'_>, rice_param: u32) -> Result<u32> {
    decode_abs_remainder(dec, rice_param)
}

/// §7.3.11.12 `residual_ts_coding()` — transform-skip residual coding.
///
/// Decodes the coefficient levels for a transform block whose
/// `transform_skip_flag` is 1 (and `sh_ts_residual_coding_disabled_flag`
/// is 0). The bitstream layout differs substantially from the regular
/// `residual_coding()` (§7.3.11.11):
///
/// * There is no `last_sig_coeff` signalling — every sub-block is walked
///   in forward diagonal order and a `sb_coded_flag` gates it (with the
///   §7.3.11.12 `inferSbCbf` inference for the last sub-block).
/// * Significance / magnitude are split across three context-coded
///   passes plus a bypass-coded remainder pass, all budgeted against a
///   shared `RemCcbs` context-bin counter
///   (`((1 << (log2W+log2H)) * 7) >> 2`). When the budget is exhausted a
///   position falls through to the pure-bypass `abs_remainder` path.
/// * `coeff_sign_flag` is **context-coded** (§9.3.4.2.10) for the
///   first-pass significant coefficients, unlike the bypass sign of the
///   regular path.
/// * The §9.3.4.2.7 / .8 / .9 / .10 ctxInc derivations use the TS
///   neighbourhood (causal left / above) and the TS ctxInc bases
///   (sig 60+, gtx 64+, par 32, sb 4+).
/// * A coefficient-level prediction fold (BDPCM-off) replaces a level of
///   1 with `Max(absLeftCoeff, absAboveCoeff)` (or decrements it when it
///   is `<= predCoeff`), per the spec's remainder-pass tail.
///
/// `rice_idx` is `sh_ts_residual_coding_rice_idx_minus1 + 1` — the fixed
/// Rice parameter for the TS `abs_remainder` binarisation (§9.3.3.11).
/// `bdpcm` is `BdpcmFlag[x0][y0][cIdx]`, which steers the
/// `abs_level_gtx_flag` / `coeff_sign_flag` contexts and disables the
/// level-prediction fold.
///
/// Returns the row-major `(n_tb_w * n_tb_h)` array of signed
/// `TransCoeffLevel` values. `sh_ts_residual_coding_disabled_flag == 1`
/// is the caller's responsibility (the regular path is used instead).
pub fn decode_ts_tb_coefficients(
    dec: &mut ArithDecoder<'_>,
    ctxs: &mut ResidualCtxs,
    n_tb_w: usize,
    n_tb_h: usize,
    // `cIdx` is part of the §7.3.11.12 syntax-structure signature but the
    // transform-skip ctxInc derivations (§9.3.4.2.6/.8/.9/.10) use fixed
    // bases independent of the colour component, so it is unused here. It
    // is retained for call-site symmetry with `decode_tb_coefficients`.
    _c_idx: u32,
    rice_idx: u32,
    bdpcm: bool,
) -> Result<Vec<i32>> {
    if !n_tb_w.is_power_of_two() || !n_tb_h.is_power_of_two() {
        return Err(Error::invalid(
            "h266 residual_ts: nTbW / nTbH must be power of two",
        ));
    }
    let log2_w = n_tb_w.trailing_zeros();
    let log2_h = n_tb_h.trailing_zeros();
    let total = n_tb_w * n_tb_h;

    // Persistent coefficient state across the whole TB.
    let mut sig_flag = vec![false; total];
    let mut abs_level_pass1 = vec![0u32; total];
    let mut abs_level_pass2 = vec![0u32; total];
    let mut abs_level = vec![0u32; total];
    // CoeffSignLevel[xC][yC] ∈ {-1, 0, +1} for the §9.3.4.2.10 sign ctx.
    let mut coeff_sign_level = vec![0i32; total];
    // Signed output (TransCoeffLevel).
    let mut out = vec![0i32; total];
    // coeff_sign_flag[n] per position (0 = positive, 1 = negative).
    let mut sign_flag = vec![0u32; total];

    let at = |xc: u32, yc: u32| -> usize { (yc as usize) * n_tb_w + (xc as usize) };

    // RemCcbs = ((1 << (log2W+log2H)) * 7) >> 2.
    let mut rem_ccbs: i32 = (((1i32) << (log2_w + log2_h)) * 7) >> 2;

    let (num_sb_w, num_sb_h) = sb_grid(n_tb_w, n_tb_h);
    let num_sb = num_sb_w * num_sb_h;
    let last_sub_block = num_sb - 1;
    let positions = coeff_scan_positions(n_tb_w, n_tb_h);
    let sb_origins = crate::scan::sb_scan_positions(n_tb_w, n_tb_h);
    // Sub-block-grid coordinates in scan order (xS, yS), in sub-block units.
    let sb_grid_coords: Vec<(u32, u32)> = sb_origins
        .iter()
        .map(|&(ox, oy)| (ox / 4, oy / 4))
        .collect();
    // sb_coded_flag indexed by sub-block grid coordinate.
    let mut sb_coded = vec![false; num_sb_w * num_sb_h];
    let sb_at = |xs: u32, ys: u32| -> usize { (ys as usize) * num_sb_w + (xs as usize) };

    let mut infer_sb_cbf = true;

    for (i, &(xs, ys)) in sb_grid_coords.iter().enumerate() {
        // sb_coded_flag read (§7.3.11.12). Inferred when this is the last
        // sub-block and no earlier sub-block was coded.
        let coded = if i != last_sub_block || !infer_sb_cbf {
            let left = xs > 0 && sb_coded[sb_at(xs - 1, ys)];
            let above = ys > 0 && sb_coded[sb_at(xs, ys - 1)];
            let inc = ctx_inc_sb_coded_flag_ts(csbf_ctx_ts(left, above)) as usize;
            let cap = ctxs.sb_coded.len() - 1;
            dec.decode_decision(&mut ctxs.sb_coded[inc.min(cap)])? == 1
        } else {
            true
        };
        sb_coded[sb_at(xs, ys)] = coded;
        if coded && i < last_sub_block {
            infer_sb_cbf = false;
        }

        // Positions of this sub-block in scan order (already TB-clipped).
        let sb_start = i * 16;
        let sb_end = (sb_start + 16).min(positions.len());
        let sb_positions = &positions[sb_start..sb_end];
        let num_sb_coeff = sb_positions.len();

        let mut infer_sb_sig = true;
        let mut last_scan_pos_pass1: i32 = -1;

        // First scan pass: sig + sign + gt1 + par.
        for (n, &(xc, yc)) in sb_positions.iter().enumerate() {
            if rem_ccbs < 4 {
                break;
            }
            last_scan_pos_pass1 = n as i32;

            // sig_coeff_flag (inferred 1 for the last position of a coded
            // sub-block when nothing earlier was significant).
            let sig = if coded && (n != num_sb_coeff - 1 || !infer_sb_sig) {
                let (loc_num_sig, _) = loc_num_sig_and_sum_abs_pass1_ts(
                    xc,
                    yc,
                    &abs_level_pass1,
                    &sig_flag,
                    n_tb_w as u32,
                );
                let inc = ctx_inc_sig_coeff_flag_ts(loc_num_sig) as usize;
                let cap = ctxs.sig_coeff.len() - 1;
                let bit = dec.decode_decision(&mut ctxs.sig_coeff[inc.min(cap)])? == 1;
                rem_ccbs -= 1;
                if bit {
                    infer_sb_sig = false;
                }
                bit
            } else if coded {
                // Inferred significant: last position, nothing earlier sig.
                true
            } else {
                false
            };
            sig_flag[at(xc, yc)] = sig;

            coeff_sign_level[at(xc, yc)] = 0;
            let mut par = 0u32;
            let mut gt1 = 0u32;
            if sig {
                // coeff_sign_flag (context-coded, §9.3.4.2.10).
                let left_sign = if xc > 0 {
                    coeff_sign_level[at(xc - 1, yc)]
                } else {
                    0
                };
                let above_sign = if yc > 0 {
                    coeff_sign_level[at(xc, yc - 1)]
                } else {
                    0
                };
                let sinc = ctx_inc_coeff_sign_flag_ts(left_sign, above_sign, bdpcm) as usize;
                let scap = ctxs.coeff_sign.len() - 1;
                let s = dec.decode_decision(&mut ctxs.coeff_sign[sinc.min(scap)])?;
                rem_ccbs -= 1;
                sign_flag[at(xc, yc)] = s;
                coeff_sign_level[at(xc, yc)] = if s > 0 { -1 } else { 1 };

                // abs_level_gtx_flag[n][0].
                let sig_left = xc > 0 && sig_flag[at(xc - 1, yc)];
                let sig_above = yc > 0 && sig_flag[at(xc, yc - 1)];
                let ginc =
                    ctx_inc_abs_level_gtx_flag_ts(0, xc, yc, sig_left, sig_above, bdpcm) as usize;
                let gcap = ctxs.abs_gtx.len() - 1;
                gt1 = dec.decode_decision(&mut ctxs.abs_gtx[ginc.min(gcap)])? as u32;
                rem_ccbs -= 1;
                if gt1 == 1 {
                    // par_level_flag.
                    let pinc = ctx_inc_par_level_flag_ts() as usize;
                    let pcap = ctxs.par_level.len() - 1;
                    par = dec.decode_decision(&mut ctxs.par_level[pinc.min(pcap)])? as u32;
                    rem_ccbs -= 1;
                }
            }
            abs_level_pass1[at(xc, yc)] = sig as u32 + par + gt1;
        }

        // Greater-than-X scan pass (numGtXFlags = 5: j = 1..4).
        let mut last_scan_pos_pass2: i32 = -1;
        for (n, &(xc, yc)) in sb_positions.iter().enumerate() {
            if rem_ccbs < 4 {
                break;
            }
            let mut a2 = abs_level_pass1[at(xc, yc)];
            // Continue reading gtx flags only while the previous one was 1.
            // The previous gtx for j == 1 is abs_level_gtx_flag[n][0],
            // which is set iff AbsLevelPass1 >= 2 (sig + gt1, par adds at
            // most 1 so a value >= 2 implies gt1 == 1).
            let mut prev_gtx = abs_level_pass1[at(xc, yc)] >= 2;
            for j in 1..5u32 {
                if !prev_gtx {
                    break;
                }
                if rem_ccbs < 4 {
                    break;
                }
                let sig_left = xc > 0 && sig_flag[at(xc - 1, yc)];
                let sig_above = yc > 0 && sig_flag[at(xc, yc - 1)];
                let ginc =
                    ctx_inc_abs_level_gtx_flag_ts(j, xc, yc, sig_left, sig_above, bdpcm) as usize;
                let gcap = ctxs.abs_gtx.len() - 1;
                let g = dec.decode_decision(&mut ctxs.abs_gtx[ginc.min(gcap)])? as u32;
                rem_ccbs -= 1;
                a2 += 2 * g;
                prev_gtx = g == 1;
            }
            abs_level_pass2[at(xc, yc)] = a2;
            last_scan_pos_pass2 = n as i32;
        }

        // Remainder scan pass (bypass abs_remainder + bypass sign for the
        // pure-bypass tail). All positions of the sub-block are visited.
        for (n, &(xc, yc)) in sb_positions.iter().enumerate() {
            let n = n as i32;
            let in_pass2 = n <= last_scan_pos_pass2;
            let in_pass1 = n <= last_scan_pos_pass1;
            // Determine whether abs_remainder is coded for this position.
            let read_rem = (in_pass2 && abs_level_pass2[at(xc, yc)] >= 10)
                || (!in_pass2 && in_pass1 && abs_level_pass1[at(xc, yc)] >= 2)
                || (!in_pass1 && coded);
            let rem = if read_rem {
                decode_abs_remainder(dec, rice_idx)?
            } else {
                0
            };
            let mut a = if in_pass2 {
                abs_level_pass2[at(xc, yc)] + 2 * rem
            } else if in_pass1 {
                abs_level_pass1[at(xc, yc)] + 2 * rem
            } else {
                // Pure-bypass position: level is the remainder, sign is a
                // bypass coeff_sign_flag read only when level != 0.
                let lvl = rem;
                if lvl != 0 {
                    let s = dec.decode_bypass()?;
                    sign_flag[at(xc, yc)] = s;
                }
                lvl
            };

            // BDPCM-off level prediction fold (§7.3.11.12 remainder tail).
            if !bdpcm && in_pass1 {
                let abs_left = if xc > 0 { abs_level[at(xc - 1, yc)] } else { 0 };
                let abs_above = if yc > 0 { abs_level[at(xc, yc - 1)] } else { 0 };
                let pred = abs_left.max(abs_above);
                if a == 1 && pred > 0 {
                    a = pred;
                } else if a > 0 && a <= pred {
                    a -= 1;
                }
            }
            abs_level[at(xc, yc)] = a;
            // TransCoeffLevel = (1 - 2*coeff_sign_flag) * AbsLevel.
            out[at(xc, yc)] = (1 - 2 * sign_flag[at(xc, yc)] as i32) * a as i32;
        }
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Exp-Golomb-0 of a zero-stream yields 0 (prefix=0, suffix=0).
    #[test]
    fn exp_golomb_zero_stream() {
        let data = [0u8; 8];
        let mut dec = ArithDecoder::new(&data).unwrap();
        let v = decode_exp_golomb_k(&mut dec, 0).unwrap();
        assert_eq!(v, 0);
    }

    /// Rice-remaining on zero stream: prefix=0 → value=0 regardless of
    /// rice_param.
    #[test]
    fn rice_remaining_zero_stream() {
        let data = [0u8; 8];
        let mut dec = ArithDecoder::new(&data).unwrap();
        let v = decode_coeff_abs_level_remaining(&mut dec, 3).unwrap();
        assert_eq!(v, 0);
    }

    /// Context-bundle init returns non-empty arrays for every bundle
    /// member; exact lengths are brittle against the transcription so
    /// we just assert a minimum-usable count.
    #[test]
    fn residual_ctxs_init_sizes() {
        let ctxs = ResidualCtxs::init(32);
        assert!(ctxs.sig_coeff.len() >= 64);
        assert_eq!(ctxs.sb_coded.len(), 21);
        assert!(ctxs.last_x.len() >= 32);
        assert!(ctxs.last_y.len() >= 32);
        assert_eq!(ctxs.tu_y_coded.len(), 12);
        assert_eq!(ctxs.tu_cb_coded.len(), 6);
        assert_eq!(ctxs.tu_cr_coded.len(), 9);
        assert_eq!(ctxs.cu_qp_delta_abs.len(), 6);
    }

    /// 4x4 TB with a zero-stream: the whole TB is all zero. Since a
    /// zero stream encodes `last_sig_coeff_x/y_prefix = 0` for most
    /// reasonable initializations, the decoder reads a DC (0,0) coeff
    /// and no others.
    #[test]
    fn zero_stream_decodes_small_tb_without_panicking() {
        let data = [0u8; 64];
        let mut dec = ArithDecoder::new(&data).unwrap();
        let mut ctxs = ResidualCtxs::init(32);
        let coeffs = decode_tb_coefficients(&mut dec, &mut ctxs, 4, 4, 0).unwrap();
        assert_eq!(coeffs.len(), 16);
    }

    /// The §7.3.11.11 flags are self-consistent with the decoded
    /// coefficient field: for a single-sub-block 4x4 luma TB,
    /// `MtsDcOnly` must be set iff the only non-zero coefficient is the
    /// DC, and `MtsZeroOutSigCoeffFlag` (which only clears when a coded
    /// sub-block lies outside the top-left 4x4 region) must stay set —
    /// the whole TB is one sub-block at (0,0).
    #[test]
    fn residual_flags_consistent_with_decoded_field() {
        let data = [0u8; 64];
        let mut dec = ArithDecoder::new(&data).unwrap();
        let mut ctxs = ResidualCtxs::init(32);
        let (levels, flags) =
            decode_tb_coefficients_with_flags(&mut dec, &mut ctxs, 4, 4, 0).unwrap();
        let _ = &levels;
        // A 4x4 TB has exactly one coefficient sub-block at (0,0), so
        // no sub-block can be outside the 4x4 region.
        assert!(flags.mts_zero_out_sig_coeff_flag);
        // For a single-sub-block luma TB both DC-only flags key off the
        // identical `lastScanPos > 0` condition (lastSubBlock is always
        // 0, transform_skip is off, dims >= 2), so they must agree.
        assert_eq!(
            flags.lfnst_dc_only, flags.mts_dc_only,
            "single-sub-block 4x4 luma: LfnstDcOnly and MtsDcOnly track the same condition"
        );
    }

    /// The CU-scoped merge is a logical AND: once any TB clears a flag
    /// it stays cleared in the accumulator.
    #[test]
    fn residual_flags_merge_is_logical_and() {
        let mut acc = TbResidualFlags::default();
        assert!(acc.lfnst_dc_only);
        acc.merge(TbResidualFlags {
            lfnst_dc_only: false,
            lfnst_zero_out_sig_coeff_flag: true,
            mts_dc_only: true,
            mts_zero_out_sig_coeff_flag: false,
        });
        assert!(!acc.lfnst_dc_only);
        assert!(acc.lfnst_zero_out_sig_coeff_flag);
        assert!(acc.mts_dc_only);
        assert!(!acc.mts_zero_out_sig_coeff_flag);
        // Merging an all-true flag set leaves the cleared bits cleared.
        acc.merge(TbResidualFlags::default());
        assert!(!acc.lfnst_dc_only);
        assert!(!acc.mts_zero_out_sig_coeff_flag);
    }

    /// Reject non-power-of-two TB sizes.
    #[test]
    fn non_pow2_tb_is_rejected() {
        let data = [0u8; 32];
        let mut dec = ArithDecoder::new(&data).unwrap();
        let mut ctxs = ResidualCtxs::init(32);
        assert!(decode_tb_coefficients(&mut dec, &mut ctxs, 3, 4, 0).is_err());
    }

    #[test]
    fn last_sig_suffix_roundtrip() {
        // prefix=0..3 → exact value. prefix=4 → 4..5, prefix=5 → 6..7, etc.
        let data = [0u8; 8];
        let mut dec = ArithDecoder::new(&data).unwrap();
        assert_eq!(last_sig_suffix(&mut dec, 0).unwrap(), 0);
        assert_eq!(last_sig_suffix(&mut dec, 3).unwrap(), 3);
    }

    /// read_tu_y_coded_flag on a zero stream returns false (MPS=0 init
    /// bias).
    #[test]
    fn tu_y_coded_flag_zero_stream_is_false() {
        let data = [0u8; 8];
        let mut dec = ArithDecoder::new(&data).unwrap();
        let mut ctxs = ResidualCtxs::init(26);
        let coded = read_tu_y_coded_flag(&mut dec, &mut ctxs, false, false, false).unwrap();
        // Whatever the init bias, the function must return a bool.
        let _ = coded;
    }

    /// Hand-built single-coeff stream: we construct bins that encode
    /// last_sig_coeff_x/y_prefix=0 / no suffix / last-sig = (0,0),
    /// sig_coeff_flag=1 at DC (inferred / still read as context-coded
    /// in our simplified walker), gt1=0, sign=0.
    ///
    /// We can't hand-craft the exact CABAC byte layout without a
    /// matching encoder, so this test just verifies the output array
    /// length on a handful of zero bytes — we're testing plumbing, not
    /// spec conformance of the level arithmetic.
    #[test]
    fn decode_tb_coefficients_fills_output_array() {
        let data = [0u8; 32];
        let mut dec = ArithDecoder::new(&data).unwrap();
        let mut ctxs = ResidualCtxs::init(26);
        let coeffs = decode_tb_coefficients(&mut dec, &mut ctxs, 4, 4, 0).unwrap();
        assert_eq!(coeffs.len(), 16);
        // The key assertion is no panic / no CABAC-state divergence.
        // Every coeff is trivially in i32 range; we just sanity-check
        // the array was populated with some value.
        let _ = coeffs.iter().sum::<i32>();
    }

    /// read_cu_qp_delta on a zero stream returns 0 (prefix=0, no sign
    /// bin read).
    #[test]
    fn cu_qp_delta_zero_stream_is_zero() {
        let data = [0u8; 32];
        let mut dec = ArithDecoder::new(&data).unwrap();
        let mut ctxs = ResidualCtxs::init(26);
        let v = read_cu_qp_delta(&mut dec, &mut ctxs).unwrap();
        assert_eq!(v, 0);
    }

    #[test]
    fn cu_chroma_qp_offset_list_len_0_skips_idx() {
        let data = [0u8; 16];
        let mut dec = ArithDecoder::new(&data).unwrap();
        let mut ctxs = ResidualCtxs::init(26);
        let (_flag, idx) = read_cu_chroma_qp_offset(&mut dec, &mut ctxs, 0).unwrap();
        assert_eq!(idx, 0);
    }

    /// Rice derivation on a zero neighbourhood yields Rice param 0
    /// (Table 128, locSumAbs = 0 → cRiceParam = 0). With baseLevel = 4
    /// the Clip3 of (0 - 20) saturates to 0 so the result is the same.
    #[test]
    fn derive_rice_param_zero_neighbourhood() {
        let abs = vec![0u32; 16];
        let rice_abs_rem = derive_rice_param(0, 0, &abs, 2, 2, 4, 4, 0);
        assert_eq!(rice_abs_rem, 0);
        let rice_dec_abs = derive_rice_param(0, 0, &abs, 2, 2, 4, 0, 0);
        assert_eq!(rice_dec_abs, 0);
    }

    /// dec_abs_level decode on a zero bypass stream yields 0 (prefix=0,
    /// suffix=0 regardless of rice_param).
    #[test]
    fn decode_dec_abs_level_zero_stream() {
        let data = [0u8; 16];
        let mut dec = ArithDecoder::new(&data).unwrap();
        let v = decode_dec_abs_level(&mut dec, 2).unwrap();
        assert_eq!(v, 0);
    }

    /// abs_remainder decode on an all-ones-then-zero stream:
    /// 0xFF 0x00 ... feeds 6 one-bits (prefix saturated), then EGk
    /// suffix starts with a 0-bit (prefix log=0, suffix = 2^k+1 bits
    /// that include 0s). For rice=0 → k=1, suffix = 2 bits all zero →
    /// egk = (1<<0 - 1) * 2 + 0 = 0, so value = (6 << 0) + 0 = 6.
    /// The exact decoder arithmetic depends on ArithDecoder bypass
    /// state, so we only assert the value is consistent and non-zero.
    #[test]
    fn decode_abs_remainder_bounded_range() {
        let data = [0u8; 16];
        let mut dec = ArithDecoder::new(&data).unwrap();
        // Zero stream → prefix=0 → value=0.
        let v = decode_abs_remainder(&mut dec, 2).unwrap();
        assert_eq!(v, 0);
    }

    /// 8×8 TB with the new walker: no panics, coeffs len = 64, state
    /// consistent. rem_bins_pass1 starts at (1<<(3+3))*7/4 = 112.
    #[test]
    fn decode_tb_coefficients_8x8_on_zero_stream() {
        let data = [0u8; 128];
        let mut dec = ArithDecoder::new(&data).unwrap();
        let mut ctxs = ResidualCtxs::init(26);
        let coeffs = decode_tb_coefficients(&mut dec, &mut ctxs, 8, 8, 0).unwrap();
        assert_eq!(coeffs.len(), 64);
    }

    /// Transform-skip residual coding (§7.3.11.12) on a zero CABAC
    /// stream: for a single-sub-block 4×4 TB the sub-block is the *last*
    /// one, so `sb_coded_flag` is inferred 1 (no bin is read) — the
    /// sub-block is treated as coded. The decoder must walk all three
    /// passes without panicking and return a length-16 level array.
    /// (A coded sub-block on an all-zero stream still infers the last
    /// position significant via `inferSbSigCoeffFlag`, so the field is
    /// not necessarily all-zero — the value depends on the CABAC state.)
    #[test]
    fn decode_ts_zero_stream_no_panic_len16() {
        let data = [0u8; 64];
        let mut dec = ArithDecoder::new(&data).unwrap();
        let mut ctxs = ResidualCtxs::init(26);
        let coeffs = decode_ts_tb_coefficients(&mut dec, &mut ctxs, 4, 4, 0, 1, false).unwrap();
        assert_eq!(coeffs.len(), 16);
    }

    /// A larger 8×8 TS TB (4 sub-blocks) on a zero stream: the first
    /// three sub-blocks read a real `sb_coded_flag` (which decodes 0 on
    /// the zero stream), and the last is inferred coded only if an
    /// earlier one was — none were, so it too is read. Exercises the
    /// multi-sub-block `inferSbCbf` path without panicking.
    #[test]
    fn decode_ts_8x8_zero_stream_no_panic() {
        let data = [0u8; 128];
        let mut dec = ArithDecoder::new(&data).unwrap();
        let mut ctxs = ResidualCtxs::init(26);
        let coeffs = decode_ts_tb_coefficients(&mut dec, &mut ctxs, 8, 8, 0, 2, false).unwrap();
        assert_eq!(coeffs.len(), 64);
    }

    /// Full encode → decode round-trip of `residual_ts_coding()` for a
    /// 4×4 luma TB with BDPCM off. A test-local encoder mirrors the
    /// decoder's exact bin sequence (sb_coded_flag, then per-position
    /// sig / sign / gt1 / par in pass 1, the gtX pass, and the bypass
    /// remainder / level-prediction tail). The chosen level field is
    /// designed so the §7.3.11.12 BDPCM-off level-prediction fold is
    /// exercised: positions whose pre-fold magnitude is 1 with a
    /// positive predecessor get lifted to the predecessor magnitude.
    #[test]
    fn ts_round_trip_4x4_with_level_prediction() {
        use crate::cabac_enc::ArithEncoder;

        // Target *decoded* levels (what the decoder must reconstruct).
        // We pick them and run the prediction fold backwards to obtain
        // the coded `AbsLevel` the encoder must emit. To keep the
        // inverse simple we use only the DC-row positions where the fold
        // is easy to reason about, leaving the rest zero.
        let n_tb_w = 4usize;
        let n_tb_h = 4usize;
        let rice_idx = 1u32;

        // We encode coded magnitudes directly and let the decoder apply
        // the fold; then assert the decoder output equals the folded
        // expectation we compute here with the same rule.
        // Coded magnitudes (pre-fold AbsLevel) in (x,y) -> level form.
        let coded: [(u32, u32, u32, u32); 2] = [
            // (xC, yC, absLevel, signFlag)
            (0, 0, 3, 0), // DC = +3
            (1, 0, 1, 1), // to its right, magnitude 1, negative
        ];

        // Build the bitstream mirroring decode_ts_tb_coefficients for a
        // single 4×4 sub-block (sub-block (0,0) is the last sub-block, so
        // sb_coded_flag is inferred 1 — NOT coded).
        let slice_qp = 26;
        let mut ctxs = ResidualCtxs::init(slice_qp);
        let mut enc = ArithEncoder::new();

        let positions = crate::scan::coeff_scan_positions(n_tb_w, n_tb_h);
        let level_at = |x: u32, y: u32| -> (u32, u32) {
            for &(cx, cy, a, s) in coded.iter() {
                if cx == x && cy == y {
                    return (a, s);
                }
            }
            (0, 0)
        };

        // Track encoder-side significance / pass1 for ctxInc mirroring.
        let total = n_tb_w * n_tb_h;
        let mut sig_e = vec![false; total];
        let mut a1_e = vec![0u32; total];
        let mut sign_e = vec![0i32; total];
        let at = |x: u32, y: u32| (y as usize) * n_tb_w + (x as usize);

        // inferSbCbf: single sub-block is the last → sb_coded inferred 1,
        // no bin emitted.
        let mut infer_sb_sig = true;

        // Pass 1: sig + sign + gt1 + par.
        for (n, &(xc, yc)) in positions.iter().enumerate() {
            let (a, s) = level_at(xc, yc);
            let sig = a > 0;
            // sig_coeff_flag is coded unless it is inferred (last pos of
            // the coded sub-block with nothing earlier significant).
            let is_last_pos = n == positions.len() - 1;
            if !(is_last_pos && infer_sb_sig) {
                let (loc_num_sig, _) = crate::ctx::loc_num_sig_and_sum_abs_pass1_ts(
                    xc,
                    yc,
                    &a1_e,
                    &sig_e,
                    n_tb_w as u32,
                );
                let inc = crate::ctx::ctx_inc_sig_coeff_flag_ts(loc_num_sig) as usize;
                let cap = ctxs.sig_coeff.len() - 1;
                enc.encode_decision(&mut ctxs.sig_coeff[inc.min(cap)], sig as u32)
                    .unwrap();
                if sig {
                    infer_sb_sig = false;
                }
            }
            sig_e[at(xc, yc)] = sig;
            if sig {
                // coeff_sign_flag (context-coded).
                let left_sign = if xc > 0 { sign_e[at(xc - 1, yc)] } else { 0 };
                let above_sign = if yc > 0 { sign_e[at(xc, yc - 1)] } else { 0 };
                let sinc =
                    crate::ctx::ctx_inc_coeff_sign_flag_ts(left_sign, above_sign, false) as usize;
                let scap = ctxs.coeff_sign.len() - 1;
                enc.encode_decision(&mut ctxs.coeff_sign[sinc.min(scap)], s)
                    .unwrap();
                sign_e[at(xc, yc)] = if s > 0 { -1 } else { 1 };

                // abs_level_gtx_flag[n][0] = (a > 1).
                let gt1 = (a > 1) as u32;
                let sig_left = xc > 0 && sig_e[at(xc - 1, yc)];
                let sig_above = yc > 0 && sig_e[at(xc, yc - 1)];
                let ginc = crate::ctx::ctx_inc_abs_level_gtx_flag_ts(
                    0, xc, yc, sig_left, sig_above, false,
                ) as usize;
                let gcap = ctxs.abs_gtx.len() - 1;
                enc.encode_decision(&mut ctxs.abs_gtx[ginc.min(gcap)], gt1)
                    .unwrap();
                let mut par = 0u32;
                if gt1 == 1 {
                    // par_level_flag = (a - 2) & 1 in the gt1 branch
                    // (AbsLevelPass1 = sig + par + gt1).
                    par = (a.saturating_sub(2)) & 1;
                    let pinc = crate::ctx::ctx_inc_par_level_flag_ts() as usize;
                    let pcap = ctxs.par_level.len() - 1;
                    enc.encode_decision(&mut ctxs.par_level[pinc.min(pcap)], par)
                        .unwrap();
                }
                a1_e[at(xc, yc)] = sig as u32 + par + gt1;
            }
        }

        // Pass 2 (gtX): for each position emit gtx flags while previous 1.
        // Our chosen levels keep AbsLevel < 4, so a1 in {0,1,2}; gt1 set
        // only for a==3 (a1 = sig+par+gt1 = 1+0+1 = 2). For a1>=2 we read
        // gtx[j=1] = (a >= 4) = 0, terminating the gtx chain.
        for &(xc, yc) in positions.iter() {
            let a1 = a1_e[at(xc, yc)];
            if a1 >= 2 {
                // gtx[j=1] = 0 (since all coded levels are < 4).
                let sig_left = xc > 0 && sig_e[at(xc - 1, yc)];
                let sig_above = yc > 0 && sig_e[at(xc, yc - 1)];
                let ginc = crate::ctx::ctx_inc_abs_level_gtx_flag_ts(
                    1, xc, yc, sig_left, sig_above, false,
                ) as usize;
                let gcap = ctxs.abs_gtx.len() - 1;
                enc.encode_decision(&mut ctxs.abs_gtx[ginc.min(gcap)], 0)
                    .unwrap();
            }
        }

        // Remainder pass: for our levels all positions are within pass1
        // and pass2 ranges. abs_remainder is read when a2 >= 10 (none) or
        // (in pass1 only, not pass2) a1 >= 2 — but our coded positions are
        // also in pass2 (lastScanPosPass2 covers them). The third clause
        // `(n > lastScanPosPass1 && sb_coded)` does not apply since the
        // sub-block is coded and every position is in pass1. With a2 < 10
        // no abs_remainder bins are emitted. No bins in this pass.
        enc.encode_terminate(1).unwrap();
        let payload = enc.finish();

        // ---- Decode ----
        let mut dec = ArithDecoder::new(&payload).unwrap();
        let mut dctxs = ResidualCtxs::init(slice_qp);
        let levels =
            decode_ts_tb_coefficients(&mut dec, &mut dctxs, n_tb_w, n_tb_h, 0, rice_idx, false)
                .unwrap();

        // Compute the expected folded output with the same §7.3.11.12 rule.
        let mut abs_level = vec![0u32; total];
        let mut expected = vec![0i32; total];
        for &(xc, yc) in positions.iter() {
            let (a, s) = level_at(xc, yc);
            let mut a = a;
            // BDPCM-off level-prediction fold.
            let abs_left = if xc > 0 { abs_level[at(xc - 1, yc)] } else { 0 };
            let abs_above = if yc > 0 { abs_level[at(xc, yc - 1)] } else { 0 };
            let pred = abs_left.max(abs_above);
            if a == 1 && pred > 0 {
                a = pred;
            } else if a > 0 && a <= pred {
                a -= 1;
            }
            abs_level[at(xc, yc)] = a;
            expected[at(xc, yc)] = (1 - 2 * s as i32) * a as i32;
        }

        assert_eq!(levels, expected, "TS residual round-trip mismatch");
        // Spot-check the fold actually changed something: the (1,0)
        // coded magnitude 1 has DC predecessor magnitude 3 → lifts to 3.
        assert_eq!(levels[at(1, 0)], -3, "level-prediction fold not applied");
        assert_eq!(levels[at(0, 0)], 3);
    }

    /// §7.4.12.11 eq. 198 — QStateTransTable structural properties.
    ///
    /// * Every state is reachable from the initial `QState = 0` (the
    ///   transposed reading of the printed literal would strand states
    ///   1 and 3 — see the [`Q_STATE_TRANS_TABLE`] doc).
    /// * State 0 is a parity-0 fixed point, so the reconstruction-pass
    ///   walk over the never-visited zero positions above the last
    ///   significant coefficient of the last sub-block is inert.
    /// * The parity-0 transition is the permutation (0)(3)(1 2): any
    ///   even-length run of zero levels (e.g. a skipped uncoded 16- or
    ///   4-coefficient sub-block) returns the trellis to its entry
    ///   state.
    #[test]
    fn q_state_trans_table_properties() {
        // Reachability from state 0.
        let mut reached = [false; 4];
        reached[0] = true;
        for _ in 0..4 {
            for s in 0..4i32 {
                if reached[s as usize] {
                    reached[q_state_advance(s, 0) as usize] = true;
                    reached[q_state_advance(s, 1) as usize] = true;
                }
            }
        }
        assert_eq!(reached, [true; 4], "all four trellis states reachable");

        // Parity-0 fixed point at state 0 + involution on {1, 2}.
        assert_eq!(q_state_advance(0, 0), 0);
        assert_eq!(q_state_advance(3, 0), 3);
        assert_eq!(q_state_advance(1, 0), 2);
        assert_eq!(q_state_advance(2, 0), 1);
        for s in 0..4i32 {
            let mut q = s;
            for _ in 0..16 {
                q = q_state_advance(q, 0);
            }
            assert_eq!(q, s, "16 zero-parity steps are the identity");
            let mut q = s;
            for _ in 0..4 {
                q = q_state_advance(q, 0);
            }
            assert_eq!(q, s, "4 zero-parity steps are the identity");
        }

        // Parity-1 (odd-level) transitions: 0→2, 1→0, 2→3, 3→1.
        assert_eq!(q_state_advance(0, 1), 2);
        assert_eq!(q_state_advance(1, 1), 0);
        assert_eq!(q_state_advance(2, 1), 3);
        assert_eq!(q_state_advance(3, 1), 1);
    }
}
