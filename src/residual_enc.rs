//! VVC residual CABAC **encoder** — forward side of §7.3.11.11.
//!
//! This module is the encoder dual of [`crate::residual`]: it takes a
//! row-major array of signed `TransCoeffLevel[]` values (output of
//! [`crate::transform_fwd::forward_dct_ii_2d`] followed by
//! [`crate::transform_fwd::quantize_tb_flat`]) and writes the
//! corresponding CABAC-coded bit sequence into an
//! [`crate::cabac_enc::ArithEncoder`].
//!
//! ## Scope
//!
//! * [`encode_last_sig_coeff_pos`] — TR context-coded `last_sig_coeff_*`
//!   prefix + FL bypass suffix (dual of [`crate::residual::read_last_sig_coeff_pos`]).
//! * [`encode_tb_coefficients`] — full §7.3.11.11 three-pass residual emit
//!   mirror of [`crate::residual::decode_tb_coefficients`]:
//!   - Pass 1: `sig_coeff_flag`, `abs_level_gtx_flag[0]`, `par_level_flag`,
//!     `abs_level_gtx_flag[1]` with spec-exact ctxInc; `remBinsPass1`
//!     budget per eq. 5018.
//!   - Pass 2: `abs_remainder[]` Rice-coded bypass (§9.3.3.11).
//!   - Pass 3: `dec_abs_level[]` Rice-coded bypass (§9.3.3.12).
//!   - Sign-flag bypass pass.
//! * [`write_tu_y_coded_flag`] / [`write_tu_cb_coded_flag`] /
//!   [`write_tu_cr_coded_flag`] — CBF flag emit (dual of the decoder
//!   `read_tu_*_coded_flag` functions).
//!
//! ## Restrictions (same as the decoder module)
//!
//! * `sh_dep_quant_used_flag = 0` (QState stays 0).
//! * `transform_skip_flag = 0` (regular residual coding only).
//! * `sh_sign_data_hiding_used_flag = 0`.
//! * `sps_persistent_rice_adaptation_enabled_flag = 0` (`HistValue = 0`).
//! * `sps_rrc_rice_extension_flag = 0` (`shiftVal = 0`).
//! * Zero-out off (`Log2ZoTb{W,H}` = full TB log2 sizes).
//!
//! Spec reference: ITU-T H.266 | ISO/IEC 23090-3 (V4, 01/2026).

use oxideav_core::Result;

use crate::cabac_enc::ArithEncoder;
use crate::ctx::{
    csbf_ctx_regular, ctx_inc_abs_level_gt_1_flag, ctx_inc_abs_level_gt_3_flag,
    ctx_inc_cu_chroma_qp_offset_flag, ctx_inc_cu_chroma_qp_offset_idx, ctx_inc_cu_qp_delta_abs,
    ctx_inc_last_sig_coeff_prefix, ctx_inc_par_level_flag, ctx_inc_sb_coded_flag_regular,
    ctx_inc_sig_coeff_flag, ctx_inc_tu_cb_coded_flag, ctx_inc_tu_cr_coded_flag,
    ctx_inc_tu_y_coded_flag, loc_num_sig_and_sum_abs_pass1, loc_sum_abs_rice,
    rice_param_from_loc_sum_abs,
};
use crate::residual::ResidualCtxs;
use crate::scan::{coeff_scan_positions, sb_grid, sb_scan_positions};

/// Write `tu_y_coded_flag` per §7.3.11.10 / §9.3.4.2.5.
pub fn write_tu_y_coded_flag(
    enc: &mut ArithEncoder,
    ctxs: &mut ResidualCtxs,
    coded: bool,
    bdpcm_y: bool,
    isp_split: bool,
    prev_tu_cbf_y: bool,
) -> Result<()> {
    let inc = ctx_inc_tu_y_coded_flag(bdpcm_y, isp_split, prev_tu_cbf_y) as usize;
    let n = ctxs.tu_y_coded.len() - 1;
    enc.encode_decision(&mut ctxs.tu_y_coded[inc.min(n)], coded as u32)
}

/// Write `tu_cb_coded_flag` per §7.3.11.10 + Table 127.
pub fn write_tu_cb_coded_flag(
    enc: &mut ArithEncoder,
    ctxs: &mut ResidualCtxs,
    coded: bool,
    bdpcm_chroma: bool,
) -> Result<()> {
    let inc = ctx_inc_tu_cb_coded_flag(bdpcm_chroma) as usize;
    let n = ctxs.tu_cb_coded.len() - 1;
    enc.encode_decision(&mut ctxs.tu_cb_coded[inc.min(n)], coded as u32)
}

/// Write `tu_cr_coded_flag` per §7.3.11.10 + Table 127.
pub fn write_tu_cr_coded_flag(
    enc: &mut ArithEncoder,
    ctxs: &mut ResidualCtxs,
    coded: bool,
    bdpcm_chroma: bool,
    tu_cb_coded: bool,
) -> Result<()> {
    let inc = ctx_inc_tu_cr_coded_flag(bdpcm_chroma, tu_cb_coded) as usize;
    let n = ctxs.tu_cr_coded.len() - 1;
    enc.encode_decision(&mut ctxs.tu_cr_coded[inc.min(n)], coded as u32)
}

/// Write `cu_qp_delta_abs` + sign (§7.3.11.10 + §9.3.3.10).
pub fn write_cu_qp_delta(
    enc: &mut ArithEncoder,
    ctxs: &mut ResidualCtxs,
    delta: i32,
) -> Result<()> {
    let abs = delta.unsigned_abs();
    // Prefix TR(cMax=5, cRice=0): ctx 0 for bin 0, ctx 1 for bins 1-4.
    let n = ctxs.cu_qp_delta_abs.len() - 1;
    let prefix = abs.min(5);
    for bin_idx in 0..prefix {
        let inc = ctx_inc_cu_qp_delta_abs(bin_idx) as usize;
        enc.encode_decision(&mut ctxs.cu_qp_delta_abs[inc.min(n)], 1)?;
    }
    if prefix < 5 {
        let inc = ctx_inc_cu_qp_delta_abs(prefix) as usize;
        enc.encode_decision(&mut ctxs.cu_qp_delta_abs[inc.min(n)], 0)?;
    } else {
        // Suffix EGk(0).
        encode_exp_golomb_k(enc, abs - 5, 0)?;
    }
    if abs > 0 {
        let sign = if delta < 0 { 1u32 } else { 0u32 };
        enc.encode_bypass(sign)?;
    }
    Ok(())
}

/// Write `cu_chroma_qp_offset_flag` + `cu_chroma_qp_offset_idx`.
pub fn write_cu_chroma_qp_offset(
    enc: &mut ArithEncoder,
    ctxs: &mut ResidualCtxs,
    flag: bool,
    idx: u32,
    list_len_minus1: u32,
) -> Result<()> {
    let inc = ctx_inc_cu_chroma_qp_offset_flag() as usize;
    let n = ctxs.cu_chroma_qp_offset_flag.len() - 1;
    enc.encode_decision(&mut ctxs.cu_chroma_qp_offset_flag[inc.min(n)], flag as u32)?;
    if !flag || list_len_minus1 == 0 {
        return Ok(());
    }
    let inc = ctx_inc_cu_chroma_qp_offset_idx() as usize;
    let n = ctxs.cu_chroma_qp_offset_idx.len() - 1;
    for i in 0..list_len_minus1 {
        let bin = if i < idx { 1 } else { 0 };
        enc.encode_decision(&mut ctxs.cu_chroma_qp_offset_idx[inc.min(n)], bin)?;
        if bin == 0 {
            break;
        }
    }
    Ok(())
}

// ------------------------------------------------------------------
// Last-significant-coefficient position emit
// ------------------------------------------------------------------

/// Write `last_sig_coeff_x_prefix/suffix` + `last_sig_coeff_y_prefix/suffix`
/// for a single TB.
pub fn encode_last_sig_coeff_pos(
    enc: &mut ArithEncoder,
    ctxs: &mut ResidualCtxs,
    log2_zo_tb_width: u32,
    log2_zo_tb_height: u32,
    c_idx: u32,
    last_x: u32,
    last_y: u32,
) -> Result<()> {
    encode_last_sig_prefix(
        enc,
        &mut ctxs.last_x,
        log2_zo_tb_width,
        (log2_zo_tb_width << 1).saturating_sub(1),
        c_idx,
        last_x,
    )?;
    encode_last_sig_prefix(
        enc,
        &mut ctxs.last_y,
        log2_zo_tb_height,
        (log2_zo_tb_height << 1).saturating_sub(1),
        c_idx,
        last_y,
    )?;
    encode_last_sig_suffix(enc, last_x)?;
    encode_last_sig_suffix(enc, last_y)?;
    Ok(())
}

fn encode_last_sig_prefix(
    enc: &mut ArithEncoder,
    ctxs: &mut [crate::cabac::ContextModel],
    log2_tb_size: u32,
    c_max: u32,
    c_idx: u32,
    value: u32,
) -> Result<()> {
    // Derive the prefix: how many leading 1-bins before the terminating 0.
    // Prefix encodes the "number of leading 1s" = value for value <= c_max.
    // (Same structure as a unary code with prefix = min(value, c_max) 1-bins
    // terminated by a 0 unless we reached c_max.)
    let prefix = value.min(c_max + 1); // prefix == number of 1-bins
    let n = ctxs.len() - 1;
    for bin_idx in 0..prefix {
        let inc = ctx_inc_last_sig_coeff_prefix(bin_idx, c_idx, log2_tb_size) as usize;
        enc.encode_decision(&mut ctxs[inc.min(n)], 1)?;
    }
    if prefix <= c_max {
        let inc = ctx_inc_last_sig_coeff_prefix(prefix, c_idx, log2_tb_size) as usize;
        enc.encode_decision(&mut ctxs[inc.min(n)], 0)?;
    }
    Ok(())
}

fn encode_last_sig_suffix(enc: &mut ArithEncoder, value: u32) -> Result<()> {
    // If value <= 3: no suffix bits needed (prefix was exact).
    if value <= 3 {
        return Ok(());
    }
    // Suffix: FL(k) where k = (prefix >> 1) - 1. Prefix = value as computed
    // by the decoder's formula: for value >= 4,
    //   prefix = 2*(k+1) + (value >= 2*(k+1) + 2^k)
    // We need to recover the suffix that reconstructs value.
    // Decode prefix from value directly.
    // Decoder: value = (1<<((prefix>>1)-1))*(2+(prefix&1)) + suffix.
    // For encoding, we need to write FL((prefix>>1)-1) bypass bits where
    // prefix for value: smallest p s.t. (1<<(p>>1>>0))*(2+(p&1)) > value.
    let mut prefix = 0u32;
    while prefix <= 3
        || (1u32 << ((prefix >> 1) - 1)) * (2 + (prefix & 1)) + (1u32 << ((prefix >> 1) - 1)) - 1
            < value
    {
        prefix += 1;
        if prefix > 63 {
            break;
        }
    }
    // k = (prefix >> 1) - 1, suffix bits.
    let k_bits = (prefix >> 1).saturating_sub(1);
    if k_bits > 0 {
        let base = (1u32 << ((prefix >> 1) - 1)) * (2 + (prefix & 1));
        let suffix = value - base;
        // Emit k_bits bits MSB-first.
        for i in (0..k_bits).rev() {
            enc.encode_bypass((suffix >> i) & 1)?;
        }
    }
    Ok(())
}

// ------------------------------------------------------------------
// Main TB coefficient encoder
// ------------------------------------------------------------------

/// §7.3.11.11 residual_coding() **encoder**: writes a single TB's
/// `TransCoeffLevel[]` values into the CABAC stream via [`ArithEncoder`].
///
/// This is the exact inverse of [`crate::residual::decode_tb_coefficients`]:
/// calling `encode_tb_coefficients` then `decode_tb_coefficients` on the
/// resulting bytes should recover the original `levels` array.
///
/// `levels` is row-major `(n_tb_w * n_tb_h)`, with the sign included.
/// Zero-level coefficients must be present in the array (as 0).
pub fn encode_tb_coefficients(
    enc: &mut ArithEncoder,
    ctxs: &mut ResidualCtxs,
    n_tb_w: usize,
    n_tb_h: usize,
    c_idx: u32,
    levels: &[i32],
) -> Result<()> {
    debug_assert_eq!(levels.len(), n_tb_w * n_tb_h);
    let log2_w = n_tb_w.trailing_zeros();
    let log2_h = n_tb_h.trailing_zeros();

    // Find the last non-zero coefficient in diagonal scan order.
    let positions = coeff_scan_positions(n_tb_w, n_tb_h);
    let mut last_x = 0u32;
    let mut last_y = 0u32;
    let mut found = false;
    for &(xc, yc) in &positions {
        if levels[(yc as usize) * n_tb_w + (xc as usize)] != 0 {
            last_x = xc;
            last_y = yc;
            found = true;
        }
    }
    if !found {
        // All-zero block: encode as if last sig = (0,0) with level 0.
        // This should not happen in practice (CBF would be 0), but
        // emit a minimal valid stream rather than panicking.
        last_x = 0;
        last_y = 0;
    }

    // Emit last_sig_coeff_x/y_prefix / suffix.
    encode_last_sig_coeff_pos(enc, ctxs, log2_w, log2_h, c_idx, last_x, last_y)?;

    // Build state arrays mirroring the decoder.
    let total = n_tb_w * n_tb_h;
    let mut abs_level = vec![0u32; total];
    let mut abs_level_pass1 = vec![0u32; total];
    let mut sig_flag = vec![false; total];

    // Pre-fill abs_level from the input (unsigned).
    for (i, &l) in levels.iter().enumerate() {
        abs_level[i] = l.unsigned_abs();
        sig_flag[i] = l != 0;
    }

    // eq. 5018: pass-1 bin budget.
    let mut rem_bins_pass1: i32 = ((1i32 << (log2_w + log2_h)) * 7) >> 2;

    let (num_sb_w, num_sb_h) = sb_grid(n_tb_w, n_tb_h);
    let sb_origins = sb_scan_positions(n_tb_w, n_tb_h);

    // Find the sub-block that contains the last-sig position.
    let mut last_sb_idx = 0usize;
    let mut last_scan_pos_in_sb = 0usize;
    for sb_idx in 0..(num_sb_w * num_sb_h) {
        let start = sb_idx * 16;
        let end = (start + 16).min(positions.len());
        for (k, &(xc, yc)) in positions[start..end].iter().enumerate() {
            if xc == last_x && yc == last_y {
                last_sb_idx = sb_idx;
                last_scan_pos_in_sb = k;
            }
        }
    }

    let mut sb_coded = vec![false; num_sb_w * num_sb_h];
    let q_state: i32 = 0;

    // Walk sub-blocks in reverse diagonal scan order starting at last_sb_idx.
    for sb_idx in (0..=last_sb_idx).rev() {
        let sb_origin = sb_origins[sb_idx];
        let (xs, ys) = ((sb_origin.0 >> 2) as usize, (sb_origin.1 >> 2) as usize);

        let right = xs + 1 < num_sb_w && sb_coded[ys * num_sb_w + (xs + 1)];
        let below = ys + 1 < num_sb_h && sb_coded[(ys + 1) * num_sb_w + xs];
        let csbf = csbf_ctx_regular(right, below);

        // Determine if this sub-block is coded (has any non-zero coeff).
        let start = sb_idx * 16;
        let end = (start + 16).min(positions.len());
        let sb_has_nonzero = positions[start..end]
            .iter()
            .any(|&(xc, yc)| abs_level[(yc as usize) * n_tb_w + (xc as usize)] != 0);

        let is_inferred = sb_idx == last_sb_idx || sb_idx == 0;
        let coded = sb_has_nonzero || is_inferred;

        if !is_inferred {
            let inc = ctx_inc_sb_coded_flag_regular(c_idx, csbf) as usize;
            let n = ctxs.sb_coded.len() - 1;
            enc.encode_decision(&mut ctxs.sb_coded[inc.min(n)], coded as u32)?;
        }
        sb_coded[ys * num_sb_w + xs] = coded;

        if !coded {
            continue;
        }

        let num_sb_coeff = end - start;
        let mut infer_sb_dc_sig = sb_idx < last_sb_idx && sb_idx > 0;

        let first_pos_mode0: i32 = if sb_idx == last_sb_idx {
            last_scan_pos_in_sb as i32
        } else {
            (num_sb_coeff as i32) - 1
        };
        let mut first_pos_mode1: i32 = first_pos_mode0;

        // ---- Pass 1: sig_coeff_flag + gt1 + par + gt3 (ctx-coded) ----
        let mut n = first_pos_mode0;
        while n >= 0 && rem_bins_pass1 >= 4 {
            let pos_idx = start + (n as usize);
            let (xc, yc) = positions[pos_idx];
            let is_last = (xc, yc) == (last_x, last_y);
            let lv = abs_level[(yc as usize) * n_tb_w + (xc as usize)];
            let sig = lv != 0;

            let (loc_num_sig, loc_sum_abs_pass1) = loc_num_sig_and_sum_abs_pass1(
                xc,
                yc,
                &abs_level_pass1,
                &sig_flag,
                log2_w,
                log2_h,
                n_tb_w as u32,
            );

            // sig_coeff_flag: emitted unless it's the last-sig position or
            // DC-inferred.
            if is_last {
                // Inferred to 1; don't emit.
            } else if infer_sb_dc_sig && n == 0 {
                // Inferred to 1; don't emit.
            } else {
                let inc =
                    ctx_inc_sig_coeff_flag(c_idx, xc, yc, q_state, loc_sum_abs_pass1) as usize;
                let cap = ctxs.sig_coeff.len() - 1;
                enc.encode_decision(&mut ctxs.sig_coeff[inc.min(cap)], sig as u32)?;
                rem_bins_pass1 -= 1;
                if sig {
                    infer_sb_dc_sig = false;
                }
            }
            sig_flag[(yc as usize) * n_tb_w + (xc as usize)] = sig;

            // Pre-compute all pass-1 flag values from the known level.
            // Emission ORDER (must match decoder's read order): gt1 → par → gt3.
            // AbsLevelPass1 = sig + par + gt1 + 2*gt3.
            // Derivation of par: AbsLevelPass1 must have the same parity as lv
            // so that pass-2 `rem = (lv - a1) / 2` is integral.
            //   a1 = 2 + par + 2*gt3  (with sig=gt1=1 when lv >= 2).
            //   lv = a1 + 2*rem  → lv - a1 must be even → (lv - 2 - 2*gt3) must be ≡ par mod 2.
            //   par = (lv - 2 - 2*gt3) & 1  (only relevant when gt1=true).
            let gt1 = lv >= 2;
            let gt3 = gt1 && lv >= 4;
            let par = if gt1 {
                (lv.wrapping_sub(2).wrapping_sub(2 * (gt3 as u32))) & 1 != 0
            } else {
                false
            };

            let (mut gt1_e, mut par_e, mut gt3_e) = (false, false, false);
            if sig {
                // abs_level_gtx_flag[n][0] (gt1).
                gt1_e = gt1;
                let inc = ctx_inc_abs_level_gt_1_flag(
                    c_idx,
                    xc,
                    yc,
                    last_x,
                    last_y,
                    loc_num_sig,
                    loc_sum_abs_pass1,
                ) as usize;
                let cap = ctxs.abs_gtx.len() - 1;
                enc.encode_decision(&mut ctxs.abs_gtx[inc.min(cap)], gt1 as u32)?;
                rem_bins_pass1 -= 1;

                if gt1 {
                    // par_level_flag.
                    par_e = par;
                    let inc = ctx_inc_par_level_flag(
                        c_idx,
                        xc,
                        yc,
                        last_x,
                        last_y,
                        loc_num_sig,
                        loc_sum_abs_pass1,
                    ) as usize;
                    let cap_par = ctxs.par_level.len() - 1;
                    enc.encode_decision(&mut ctxs.par_level[inc.min(cap_par)], par as u32)?;
                    rem_bins_pass1 -= 1;

                    // abs_level_gtx_flag[n][1] (gt3).
                    gt3_e = gt3;
                    let inc = ctx_inc_abs_level_gt_3_flag(
                        c_idx,
                        xc,
                        yc,
                        last_x,
                        last_y,
                        loc_num_sig,
                        loc_sum_abs_pass1,
                    ) as usize;
                    enc.encode_decision(&mut ctxs.abs_gtx[inc.min(cap)], gt3 as u32)?;
                    rem_bins_pass1 -= 1;
                }
            }
            let (gt1, par, gt3) = (gt1_e, par_e, gt3_e);

            // AbsLevelPass1 = sig + par + gt1 + 2*gt3.
            let a1 = sig as u32 + par as u32 + gt1 as u32 + 2 * gt3 as u32;
            abs_level_pass1[(yc as usize) * n_tb_w + (xc as usize)] = a1;

            first_pos_mode1 = n - 1;
            n -= 1;
        }

        // ---- Pass 2: abs_remainder (bypass, Rice-coded) ----
        for n2 in ((first_pos_mode1 + 1)..=first_pos_mode0).rev() {
            let pos_idx = start + (n2 as usize);
            let (xc, yc) = positions[pos_idx];
            let a1 = abs_level_pass1[(yc as usize) * n_tb_w + (xc as usize)];
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
                let lv = abs_level[(yc as usize) * n_tb_w + (xc as usize)];
                // abs_remainder = (lv - a1) / 2 when a1 is from pass 1.
                // Decoder: new_abs = a1 + 2 * rem → rem = (lv - a1) / 2.
                let rem = (lv.saturating_sub(a1)) / 2;
                encode_abs_remainder(enc, rice, rem)?;
            }
        }

        // ---- Pass 3: dec_abs_level (bypass, Rice-coded) ----
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
            let lv = abs_level[(yc as usize) * n_tb_w + (xc as usize)];
            // Encoder: emit dec_abs_level such that the decoder recovers lv.
            // ZeroPos = (1 if q_state < 2 else 2) << rice.
            let zero_pos = (if q_state < 2 { 1u32 } else { 2 }) << rice;
            // Decoder: dec == zero_pos → lv = 0;
            //           dec < zero_pos → lv = dec + 1;
            //           dec > zero_pos → lv = dec.
            // Encoder (inverse): if lv == 0: emit zero_pos.
            //                    if lv <= zero_pos: emit lv - 1.
            //                    else: emit lv.
            let dec_val = if lv == 0 {
                zero_pos
            } else if lv <= zero_pos {
                lv - 1
            } else {
                lv
            };
            encode_abs_remainder(enc, rice, dec_val)?;
        }

        // ---- Sign-flag bypass pass ----
        for k in (0..num_sb_coeff).rev() {
            let pos_idx = start + k;
            let (xc, yc) = positions[pos_idx];
            let lv = levels[(yc as usize) * n_tb_w + (xc as usize)];
            if lv != 0 {
                let sign = if lv < 0 { 1u32 } else { 0u32 };
                enc.encode_bypass(sign)?;
            }
        }
    }
    Ok(())
}

// ------------------------------------------------------------------
// Internal helpers
// ------------------------------------------------------------------

/// Rice-parameter derivation — mirrors [`crate::residual`]'s private
/// `derive_rice_param`.
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
    let hist_value = 0u32;
    let loc = loc_sum_abs_rice(
        x_c,
        y_c,
        abs_level,
        log2_full_tb_w,
        log2_full_tb_h,
        n_tb_w,
        hist_value,
    );
    let shift_val = 0u32;
    let clipped = ((loc >> shift_val) as i32 - (base_level as i32) * 5).clamp(0, 31) as u32;
    rice_param_from_loc_sum_abs(clipped)
}

/// §9.3.3.11 / §9.3.3.12 `abs_remainder[]` / `dec_abs_level[]` bypass
/// encode. TR(cMax = 6 << rice) prefix + EGk(rice+1) suffix.
fn encode_abs_remainder(enc: &mut ArithEncoder, rice_param: u32, value: u32) -> Result<()> {
    // Prefix: how many 1-bins before the terminating 0. Max prefix = 6.
    let threshold = 6u32 << rice_param;
    if value < threshold {
        // value = (prefix << rice) + suffix, prefix < 6.
        let prefix = value >> rice_param;
        let suffix = value & ((1u32 << rice_param) - 1);
        // Emit `prefix` 1-bins then a 0-bin.
        for _ in 0..prefix {
            enc.encode_bypass(1)?;
        }
        enc.encode_bypass(0)?;
        // Emit `rice_param` suffix bits MSB-first.
        for i in (0..rice_param).rev() {
            enc.encode_bypass((suffix >> i) & 1)?;
        }
    } else {
        // Prefix is all-6 ones; suffix is EGk(rice+1).
        for _ in 0..6 {
            enc.encode_bypass(1)?;
        }
        let remainder = value - threshold;
        encode_exp_golomb_k(enc, remainder, rice_param + 1)?;
    }
    Ok(())
}

/// k-th order Exp-Golomb bypass encode (§9.3.3.5, inverse of decode).
///
/// Decoder: `log` leading 1-bits, one terminating 0-bit, then `log+k` suffix
/// bits (MSB first). `value = ((1<<log) - 1) * (1<<k) + suffix`.
fn encode_exp_golomb_k(enc: &mut ArithEncoder, value: u32, k: u32) -> Result<()> {
    // Find log: the largest value such that ((1<<log)-1)*(1<<k) <= value.
    // i.e. (1<<log) <= value/(1<<k) + 1
    // i.e. log = floor_log2(value/(1<<k) + 1) but only if that gives
    // ((1<<log)-1)*(1<<k) <= value.
    //
    // Simple search: start from log=0.
    let mut log = 0u32;
    while ((1u32 << (log + 1)) - 1) * (1u32 << k) <= value {
        log += 1;
    }
    // Emit `log` 1-bits (unary prefix).
    for _ in 0..log {
        enc.encode_bypass(1)?;
    }
    // Emit the terminating 0-bit.
    enc.encode_bypass(0)?;
    // Suffix = value - ((1<<log) - 1) * (1<<k). Emit `log + k` bits MSB-first.
    let base = ((1u32 << log) - 1) * (1u32 << k);
    let suffix = value - base;
    let suffix_bits = log + k;
    for i in (0..suffix_bits).rev() {
        enc.encode_bypass((suffix >> i) & 1)?;
    }
    Ok(())
}

// ------------------------------------------------------------------
// Tests
// ------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::residual::{decode_tb_coefficients, ResidualCtxs};

    fn pad(bytes: Vec<u8>) -> Vec<u8> {
        let mut out = bytes;
        out.extend_from_slice(&[0u8; 64]);
        out
    }

    /// Encode then decode a 4×4 TB with a single DC coefficient = 1.
    /// (encode_tb_coefficients is only called when CBF = 1, i.e. there
    /// is at least one non-zero level.)
    #[test]
    fn encode_decode_4x4_level_one() {
        let mut levels = vec![0i32; 16];
        levels[0] = 1;
        let mut enc = ArithEncoder::new();
        let mut enc_ctxs = ResidualCtxs::init(26);
        encode_tb_coefficients(&mut enc, &mut enc_ctxs, 4, 4, 0, &levels).unwrap();
        let bytes = pad(enc.finish());
        let mut dec = crate::cabac::ArithDecoder::new(&bytes).unwrap();
        let mut dec_ctxs = ResidualCtxs::init(26);
        let recovered = decode_tb_coefficients(&mut dec, &mut dec_ctxs, 4, 4, 0).unwrap();
        assert_eq!(recovered, levels);
    }

    /// Encode then decode a 4×4 TB with a single DC coefficient.
    #[test]
    fn encode_decode_4x4_single_dc() {
        let mut levels = vec![0i32; 16];
        levels[0] = 7; // DC at (0,0)
        let mut enc = ArithEncoder::new();
        let mut enc_ctxs = ResidualCtxs::init(26);
        encode_tb_coefficients(&mut enc, &mut enc_ctxs, 4, 4, 0, &levels).unwrap();
        let bytes = pad(enc.finish());
        let mut dec = crate::cabac::ArithDecoder::new(&bytes).unwrap();
        let mut dec_ctxs = ResidualCtxs::init(26);
        let recovered = decode_tb_coefficients(&mut dec, &mut dec_ctxs, 4, 4, 0).unwrap();
        assert_eq!(recovered, levels);
    }

    /// Encode then decode a 4×4 TB with a negative coefficient.
    #[test]
    fn encode_decode_4x4_negative_coeff() {
        let mut levels = vec![0i32; 16];
        levels[0] = -5;
        let mut enc = ArithEncoder::new();
        let mut enc_ctxs = ResidualCtxs::init(26);
        encode_tb_coefficients(&mut enc, &mut enc_ctxs, 4, 4, 0, &levels).unwrap();
        let bytes = pad(enc.finish());
        let mut dec = crate::cabac::ArithDecoder::new(&bytes).unwrap();
        let mut dec_ctxs = ResidualCtxs::init(26);
        let recovered = decode_tb_coefficients(&mut dec, &mut dec_ctxs, 4, 4, 0).unwrap();
        assert_eq!(recovered, levels);
    }

    /// Encode then decode a 4×4 TB with multiple scattered coefficients.
    #[test]
    fn encode_decode_4x4_scattered_coeffs() {
        let mut levels = vec![0i32; 16];
        levels[0] = 3;
        levels[1] = -2;
        levels[4] = 1;
        levels[5] = -4;
        levels[10] = 6;
        let mut enc = ArithEncoder::new();
        let mut enc_ctxs = ResidualCtxs::init(26);
        encode_tb_coefficients(&mut enc, &mut enc_ctxs, 4, 4, 0, &levels).unwrap();
        let bytes = pad(enc.finish());
        let mut dec = crate::cabac::ArithDecoder::new(&bytes).unwrap();
        let mut dec_ctxs = ResidualCtxs::init(26);
        let recovered = decode_tb_coefficients(&mut dec, &mut dec_ctxs, 4, 4, 0).unwrap();
        assert_eq!(recovered, levels);
    }

    /// Encode then decode an 8×8 TB with large coefficients.
    #[test]
    fn encode_decode_8x8_large_coeffs() {
        let mut levels = vec![0i32; 64];
        levels[0] = 100;
        levels[1] = -50;
        levels[8] = 30;
        levels[9] = -20;
        levels[16] = 10;
        let mut enc = ArithEncoder::new();
        let mut enc_ctxs = ResidualCtxs::init(26);
        encode_tb_coefficients(&mut enc, &mut enc_ctxs, 8, 8, 0, &levels).unwrap();
        let bytes = pad(enc.finish());
        let mut dec = crate::cabac::ArithDecoder::new(&bytes).unwrap();
        let mut dec_ctxs = ResidualCtxs::init(26);
        let recovered = decode_tb_coefficients(&mut dec, &mut dec_ctxs, 8, 8, 0).unwrap();
        assert_eq!(recovered, levels);
    }

    /// Chroma (c_idx = 1) round-trip.
    #[test]
    fn encode_decode_4x4_chroma() {
        let mut levels = vec![0i32; 16];
        levels[0] = 4;
        levels[2] = -3;
        let mut enc = ArithEncoder::new();
        let mut enc_ctxs = ResidualCtxs::init(26);
        encode_tb_coefficients(&mut enc, &mut enc_ctxs, 4, 4, 1, &levels).unwrap();
        let bytes = pad(enc.finish());
        let mut dec = crate::cabac::ArithDecoder::new(&bytes).unwrap();
        let mut dec_ctxs = ResidualCtxs::init(26);
        let recovered = decode_tb_coefficients(&mut dec, &mut dec_ctxs, 4, 4, 1).unwrap();
        assert_eq!(recovered, levels);
    }

    /// exp_golomb_k round-trip: encode then decode several values.
    #[test]
    fn exp_golomb_k_round_trip() {
        use crate::residual::decode_exp_golomb_k;
        for k in 0u32..=3 {
            for val in [0u32, 1, 2, 5, 10, 31, 127] {
                let mut enc = ArithEncoder::new();
                encode_exp_golomb_k(&mut enc, val, k).unwrap();
                let bytes = pad(enc.finish());
                let mut dec = crate::cabac::ArithDecoder::new(&bytes).unwrap();
                let got = decode_exp_golomb_k(&mut dec, k).unwrap();
                assert_eq!(got, val, "EGk({k}) round-trip failure: val={val} got={got}");
            }
        }
    }

    /// encode_tb_coefficients round-trip with large coefficient values
    /// (exercises the abs_remainder / EGk suffix path).
    #[test]
    fn encode_decode_4x4_large_dc_exercises_abs_remainder() {
        // Level 50 will trigger pass-2 (abs_remainder) via gt3=true.
        let mut levels = vec![0i32; 16];
        levels[0] = 50;
        let mut enc = ArithEncoder::new();
        let mut enc_ctxs = ResidualCtxs::init(26);
        encode_tb_coefficients(&mut enc, &mut enc_ctxs, 4, 4, 0, &levels).unwrap();
        let bytes = pad(enc.finish());
        let mut dec = crate::cabac::ArithDecoder::new(&bytes).unwrap();
        let mut dec_ctxs = ResidualCtxs::init(26);
        let recovered = decode_tb_coefficients(&mut dec, &mut dec_ctxs, 4, 4, 0).unwrap();
        assert_eq!(recovered, levels);
    }
}
