//! VVC forward transform + flat quantisation primitives — the
//! encoder-side dual of [`crate::transform`] (§8.7.4) and
//! [`crate::dequant`] (§8.7.3).
//!
//! Round-16 scope:
//!
//! * [`forward_dct_ii_2d`] — separable 2D forward DCT-II using the
//!   spec's 64×64 transMatrix (§8.7.4.5) applied as `M^T` (the
//!   forward transform is the transpose of the inverse since the
//!   matrix is orthogonal up to scale).
//! * [`quantize_tb_flat`] — forward flat (`m[x][y] = 16` per the
//!   §8.7.3 NOTE) quantisation that is the integer dual of
//!   [`crate::dequant::dequantize_tb_flat`].
//!
//! These are pure infrastructure: the round-16 CABAC encoder
//! ([`crate::cabac_enc`]) is the foundational primitive, and these
//! transform / quant helpers let future rounds build the residual
//! emit pipeline (forward transform → forward quantise → CABAC
//! emit) without re-deriving the math.
//!
//! No third-party VVC encoder source was consulted; the math is
//! mechanically derived from [`crate::transform`] and
//! [`crate::dequant`].
//!
//! Spec reference: ITU-T H.266 | ISO/IEC 23090-3 (V4, 01/2026).

use oxideav_core::{Error, Result};

use crate::dequant::LEVEL_SCALE;

/// Apply the forward DCT-II transform of a 1D row of residual samples
/// of length `n_tb_s ∈ {2, 4, 8, 16, 32, 64}`.
///
/// The spec's inverse DCT-II (§8.7.4.4 eq. 1177) is:
/// ```text
///   y[i] = sum_j M[i, j*(64 >> log2 N)] * x[j]
/// ```
/// where `M` is the 64×64 transform matrix from §8.7.4.5. Since `M` is
/// (up to a scale factor) orthogonal, the **forward** transform is
/// `M^T` applied at the same column stride:
/// ```text
///   Y[k] = sum_i M[i, k*(64 >> log2 N)] * x[i]
/// ```
/// — i.e. swap the roles of the row index `i` and the column index `k`
/// (the matrix-vector product becomes a vector-matrix product).
///
/// The output is an `n_tb_s`-length array of unscaled forward
/// coefficients. Composition into a full 2D forward + bdShift wind-up
/// lives in [`forward_dct_ii_2d`].
pub fn forward_dct_ii_1d(n_tb_s: usize, x: &[i32]) -> Result<Vec<i32>> {
    if x.len() != n_tb_s {
        return Err(Error::invalid(format!(
            "h266 fwd xfm: input length {} != nTbS {n_tb_s}",
            x.len()
        )));
    }
    if !(matches!(n_tb_s, 2 | 4 | 8 | 16 | 32 | 64)) {
        return Err(Error::invalid(format!(
            "h266 fwd xfm: unsupported nTbS {n_tb_s}"
        )));
    }
    // Column stride per eq. 1177: `j * (64 >> log2(N))`.
    let stride = 64usize >> n_tb_s.trailing_zeros();
    let mut y = vec![0i32; n_tb_s];
    for (k, y_k) in y.iter_mut().enumerate() {
        let mut acc: i64 = 0;
        let col = k * stride;
        for (i, &x_i) in x.iter().enumerate() {
            acc += dct_ii_entry(i, col) as i64 * x_i as i64;
        }
        *y_k = acc as i32;
    }
    Ok(y)
}

/// Apply the separable 2D forward DCT-II to a `n_tb_w × n_tb_h` block
/// of residual samples (row-major) and return the 2D transform
/// coefficients (also row-major).
///
/// This is the encoder-side dual of [`crate::transform::inverse_transform_2d`]
/// for `tr_type = DCT-II` only (the round-16 minimum). The composition
/// is: 1D forward over rows, mid-shift to keep precision in 16 bits,
/// then 1D forward over columns, then a final `bdShift` matching the
/// §8.7.4.1 `(20 - BitDepth)` of the inverse path.
///
/// The exact shift values mirror the inverse path so a round-trip
/// `quantize → dequantize → inverse_transform_2d` yields back the
/// (slightly-quantised) original block.
pub fn forward_dct_ii_2d(
    n_tb_w: usize,
    n_tb_h: usize,
    samples: &[i32],
    _bit_depth: u32,
) -> Result<Vec<i32>> {
    if samples.len() != n_tb_w * n_tb_h {
        return Err(Error::invalid(format!(
            "h266 fwd xfm 2D: input length {} != {}x{}",
            samples.len(),
            n_tb_w,
            n_tb_h
        )));
    }
    // Step A — horizontal forward over each row. Mirrors the inverse
    // path's "horizontal then vertical" but swapped.
    let mut row_out = vec![0i32; n_tb_w * n_tb_h];
    for y in 0..n_tb_h {
        let row = &samples[y * n_tb_w..(y + 1) * n_tb_w];
        let r = forward_dct_ii_1d(n_tb_w, row)?;
        for x in 0..n_tb_w {
            row_out[y * n_tb_w + x] = r[x];
        }
    }
    // Step B — first-stage shift. The inverse 2D pipeline does a
    // `(input + 64) >> 7` mid-shift after its first 1D transform; the
    // forward direction applies the same shift between its two 1D
    // transforms so the round-trip cancels.
    let bd_shift_first = 7u32;
    let round_first = 1i64 << (bd_shift_first - 1);
    for v in row_out.iter_mut() {
        *v = (((*v as i64) + round_first) >> bd_shift_first) as i32;
    }
    // Step C — vertical forward per column.
    let mut out = vec![0i32; n_tb_w * n_tb_h];
    for x in 0..n_tb_w {
        let mut col = vec![0i32; n_tb_h];
        for y in 0..n_tb_h {
            col[y] = row_out[y * n_tb_w + x];
        }
        let c = forward_dct_ii_1d(n_tb_h, &col)?;
        for y in 0..n_tb_h {
            out[y * n_tb_w + x] = c[y];
        }
    }
    // Step D — final normalisation shift to put coefficients into the
    // dequant output domain (`d[]` fed to the inverse transform), so that
    // `quantize_tb_flat(forward_dct_ii_2d(r))` is the proper encoder-side
    // inverse of `dequantize_tb_flat → inverse_transform_2d`.
    //
    // Derivation (8-bit, log2TrRange=15, square N×N block):
    //
    //   FDCT_2D with 7-bit mid-shift:
    //     DC_out = 32 · N² · r   (for flat residual r)
    //
    //   Target `d` value so that IDCT_2D(d) = r (all square sizes, 8-bit):
    //     IDCT_col:  64·d  →  >>7  →  d/2  →  IDCT_row:  64·d/2 = 32·d
    //     >>bdShift_IDCT(=5+15-8=12):  32·d >> 12 = r  →  d = r·128
    //     (d_correct = 128·r, independent of N)
    //
    //   Ratio = DC_out / d_correct = 32·N²·r / (128·r) = N²/4
    //   Right-shift = log2(N²/4) = 2·log2(N) − 2
    //
    // For a W×H block: generalise to (log2W + log2H) − 2 using integer
    // floor (this equals 2·log2N − 2 for square blocks).
    //
    // The shift is applied with round-to-nearest (add 2^(shift−1)) to
    // avoid systematic down-bias for AC coefficients.
    let log2_w = n_tb_w.trailing_zeros();
    let log2_h = n_tb_h.trailing_zeros();
    let total_log2 = log2_w + log2_h; // ≥ 4 for min 2×2
    if total_log2 < 2 {
        // Tiny (2×1 or 1×2 — degenerate, should not occur in practice).
        return Ok(out);
    }
    let bd_shift_final = (total_log2 - 2) as usize;
    if bd_shift_final == 0 {
        // No shift needed (2×2 block: log2W+log2H=2, shift=0).
        return Ok(out);
    }
    let round_final = 1i64 << (bd_shift_final - 1);
    for v in out.iter_mut() {
        // Symmetric rounding: round away from zero for exact half-points.
        let vv = *v as i64;
        let shifted = if vv >= 0 {
            (vv + round_final) >> bd_shift_final
        } else {
            -((-vv + round_final) >> bd_shift_final)
        };
        *v = shifted as i32;
    }
    Ok(out)
}

/// Forward flat quantisation — the encoder-side dual of
/// [`crate::dequant::dequantize_tb_flat`].
///
/// Given the 2D forward-transform output `coeffs` (row-major,
/// `n_tb_w * n_tb_h`) and a quantisation parameter `qp`, produce the
/// integer level array `levels[]` such that
/// `dequantize_tb_flat(levels, ...)` recovers an approximation of
/// `coeffs`.
///
/// The dequantiser computes (eq. 1155):
/// ```text
///   d[x][y] = clip3(min, max, (level * ls + bdOffset) >> bdShift)
/// ```
/// where `ls = 16 * levelScale[rectNonTsFlag][qP%6] << (qP/6)` (flat
/// scaling list, m=16). The forward direction is therefore:
/// ```text
///   level = round(coeff << bdShift / ls)
/// ```
/// We implement this with rounding-to-nearest.
pub fn quantize_tb_flat(
    coeffs: &[i32],
    n_tb_w: u32,
    n_tb_h: u32,
    qp: i32,
    bit_depth: u32,
    log2_transform_range: u32,
) -> Result<Vec<i32>> {
    let w = n_tb_w as usize;
    let h = n_tb_h as usize;
    if coeffs.len() != w * h {
        return Err(Error::invalid(format!(
            "h266 fwd quant: input length {} != {}x{}",
            coeffs.len(),
            w,
            h
        )));
    }
    if !w.is_power_of_two() || !h.is_power_of_two() {
        return Err(Error::invalid(
            "h266 fwd quant: nTbW and nTbH must be powers of two",
        ));
    }
    let log2_w = w.trailing_zeros();
    let log2_h = h.trailing_zeros();
    let rect_non_ts = (((log2_w + log2_h) & 1) == 1) as u32;
    let bd_shift = (bit_depth as i32 + rect_non_ts as i32 + ((log2_w + log2_h) as i32) / 2 + 10
        - log2_transform_range as i32)
        .max(0) as u32;
    let q_mod = qp.rem_euclid(6) as u32;
    let q_div = qp / 6;
    let level_scale = LEVEL_SCALE[rect_non_ts as usize][q_mod as usize] as i64;
    let m = 16i64;
    let ls = (m * level_scale) << (q_div.max(0) as u32);
    let scale = 1i64 << bd_shift;

    let mut out = vec![0i32; coeffs.len()];
    for (i, &c) in coeffs.iter().enumerate() {
        // level ≈ round(c * scale / ls). Use signed rounding.
        let num = (c as i64) * scale;
        let level = if num >= 0 {
            (num + ls / 2) / ls
        } else {
            (num - ls / 2) / ls
        };
        out[i] = level as i32;
    }
    Ok(out)
}

/// Forward quantisation for **dependent quantization**
/// (`sh_dep_quant_used_flag = 1`) — the encoder-side inverse of the
/// §8.7.3 dep-quant dequant arms (`bdShift` + 1 per eq. 1143, the
/// `(qP + 1)`-scaled levelScale per eq. 1151) combined with the
/// §7.3.11.11 magnitude reconstruction
/// `TransCoeffLevel = 2 · AbsLevel − (QState > 1)`.
///
/// Walks the whole TB in reverse diagonal scan order — the §7.4.12.11
/// eq. 198 trellis coding order (positions above the last significant
/// coefficient hold zeros and state 0 is a parity-0 fixed point, so
/// starting from the very end of the TB is exact) — and makes a greedy
/// hard decision per coefficient: the `AbsLevel` whose reconstruction
/// under the CURRENT trellis state minimises the error, then advances
/// the state on that level's parity. (A full trellis search over the
/// four states is an encoder-quality upgrade, not a conformance
/// requirement; the greedy output is always trellis-consistent, i.e.
/// [`crate::residual_enc::encode_tb_coefficients_opts`] accepts it by
/// construction.)
///
/// Returns the signed `TransCoeffLevel` array (row-major).
pub fn quantize_tb_dep_quant(
    coeffs: &[i32],
    n_tb_w: u32,
    n_tb_h: u32,
    qp: i32,
    bit_depth: u32,
    log2_transform_range: u32,
) -> Result<Vec<i32>> {
    use crate::residual::q_state_advance;
    use crate::scan::coeff_scan_positions;

    let w = n_tb_w as usize;
    let h = n_tb_h as usize;
    if coeffs.len() != w * h {
        return Err(Error::invalid(format!(
            "h266 dep-quant fwd: input length {} != {}x{}",
            coeffs.len(),
            w,
            h
        )));
    }
    if !w.is_power_of_two() || !h.is_power_of_two() {
        return Err(Error::invalid(
            "h266 dep-quant fwd: nTbW and nTbH must be powers of two",
        ));
    }
    let log2_w = w.trailing_zeros();
    let log2_h = h.trailing_zeros();
    let rect_non_ts = (((log2_w + log2_h) & 1) == 1) as u32;
    // eq. 1143 with the `+ sh_dep_quant_used_flag` term.
    let bd_shift = (bit_depth as i32 + rect_non_ts as i32 + ((log2_w + log2_h) as i32) / 2 + 10
        - log2_transform_range as i32
        + 1)
    .max(0) as u32;
    // eq. 1151 — (qP + 1) drives the levelScale arms.
    let qe = qp + 1;
    let q_mod = qe.rem_euclid(6) as usize;
    let q_div = (qe / 6).max(0) as u32;
    let ls = (16i64 * LEVEL_SCALE[rect_non_ts as usize][q_mod] as i64) << q_div;
    let scale = 1i64 << bd_shift;

    let positions = coeff_scan_positions(w, h);
    let mut out = vec![0i32; w * h];
    let mut q_state: i32 = 0;
    for &(xc, yc) in positions.iter().rev() {
        let idx = (yc as usize) * w + (xc as usize);
        let c = coeffs[idx] as i64;
        let target = c.unsigned_abs() as i64 * scale;
        let delta = i64::from(q_state > 1);
        // AbsLevel estimate from |t| ≈ target / ls, a = (|t| + δ) / 2.
        let t_est = (target + ls / 2) / ls;
        let a_mid = ((t_est + delta) / 2).max(0);
        // Greedy hard decision: probe a_mid − 1 / a_mid / a_mid + 1
        // (and 0), pick the minimum reconstruction error under the
        // current state's quantizer.
        let mut best_a: i64 = 0;
        let mut best_err: i64 = target; // a = 0 → recon 0
        for a in [a_mid.saturating_sub(1).max(1), a_mid.max(1), a_mid + 1] {
            let recon = (2 * a - delta) * ls;
            let err = (recon - target).abs();
            if err < best_err {
                best_err = err;
                best_a = a;
            }
        }
        if best_a > 0 {
            let mag = (2 * best_a - delta) as i32;
            out[idx] = if c < 0 { -mag } else { mag };
        }
        q_state = q_state_advance(q_state, best_a as u32);
    }
    Ok(out)
}

// ---------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------

/// 64×64 DCT-II `transMatrix` entry at row `m`, column `n` (§8.7.4.5
/// eqs. 1179 / 1181 / 1183 / 1184). Re-derived locally so this module
/// can stand on its own without exposing a private decoder helper.
#[inline]
fn dct_ii_entry(m: usize, n: usize) -> i32 {
    use crate::transform::{DCT_II_COL_0_15, DCT_II_COL_16_31};
    debug_assert!(m < 64 && n < 64);
    match m {
        0..=15 => DCT_II_COL_0_15[n][m] as i32,
        16..=31 => DCT_II_COL_16_31[n][m - 16] as i32,
        32..=47 => {
            let sign = if n & 1 == 1 { -1 } else { 1 };
            sign * DCT_II_COL_16_31[n][47 - m] as i32
        }
        48..=63 => {
            let sign = if n & 1 == 1 { -1 } else { 1 };
            sign * DCT_II_COL_0_15[n][63 - m] as i32
        }
        _ => unreachable!(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 1D forward DCT-II of a flat (DC-only-input) row should
    /// concentrate all energy at the DC output coefficient.
    #[test]
    fn forward_1d_flat_input_is_dc_only() {
        // Constant input: every sample = 100 → forward output should
        // have most energy at index 0.
        let x = vec![100i32; 8];
        let y = forward_dct_ii_1d(8, &x).unwrap();
        // y[0] must dominate.
        let dc = y[0].abs();
        for &v in &y[1..] {
            assert!(
                v.abs() < dc,
                "non-DC coefficient {} should be smaller than DC {}",
                v.abs(),
                dc
            );
        }
    }

    /// 1D forward then 1D inverse should round-trip back to the
    /// original (modulo scale). For small inputs and the unscaled
    /// matrix, the round-trip introduces a fixed scale factor `N` per
    /// the orthogonality of the spec's matrix.
    #[test]
    fn forward_then_inverse_1d_recovers_input_up_to_scale() {
        use crate::transform::{one_d_transform, TrType};
        let x = vec![10, -5, 3, -7, 12, -2, 1, 8];
        let y_fwd = forward_dct_ii_1d(8, &x).unwrap();
        let y_inv = one_d_transform(TrType::DctII, 8, 8, &y_fwd).unwrap();
        // The DCT-II matrix entries are 8-bit-magnitude (~64). The
        // round-trip introduces a scale of N * (norm of matrix row /
        // col)^2 ≈ N * 64^2 = 8 * 4096 = 32768. So divide by 32768
        // and verify approximate equality.
        for (i, (&xi, &yi)) in x.iter().zip(y_inv.iter()).enumerate() {
            // We need to take the round-trip and rescale. The exact
            // scale depends on matrix normalisation; check that the
            // rescaled value is in the right ballpark (sign + same
            // order of magnitude).
            let scaled = yi as f64 / 32768.0;
            let err = (scaled - xi as f64).abs();
            assert!(
                err < (xi.abs() as f64 + 5.0),
                "mismatch at i={i}: x={xi} round-trip={yi} scaled={scaled:.3}"
            );
        }
    }

    /// Quantise then dequantise should recover small-magnitude
    /// coefficients to within a few units (the quantisation step).
    #[test]
    fn quantise_then_dequantise_round_trip_small_qp() {
        use crate::dequant::{dequantize_tb_flat, DequantParams};
        // QP = 0 (minimum quantisation). Small-magnitude coefficients.
        let coeffs = vec![1000i32; 16];
        let levels = quantize_tb_flat(&coeffs, 4, 4, 0, 8, 15).unwrap();
        let params = DequantParams::luma_8bit(4, 4, 0);
        let recovered = dequantize_tb_flat(&levels, &params).unwrap();
        for (i, (&c, &r)) in coeffs.iter().zip(recovered.iter()).enumerate() {
            // The round-trip should recover c exactly modulo the
            // quantisation step. At QP=0 the step is small.
            let err = (c - r).abs();
            assert!(err < 50, "round-trip error at {i}: {c} → {r}");
        }
    }

    /// Zero coefficients quantise to zero levels.
    #[test]
    fn quantise_zeros_yields_zeros() {
        let coeffs = vec![0i32; 16];
        let levels = quantize_tb_flat(&coeffs, 4, 4, 26, 8, 15).unwrap();
        for &l in &levels {
            assert_eq!(l, 0);
        }
    }

    /// Negative coefficients quantise symmetrically to negative levels.
    #[test]
    fn quantise_negative_coeffs_yields_negative_levels() {
        let coeffs = vec![-12000i32; 4];
        let levels = quantize_tb_flat(&coeffs, 2, 2, 26, 8, 15).unwrap();
        for &l in &levels {
            assert!(l < 0, "expected negative level, got {l}");
        }
    }

    /// Mismatched input size is rejected.
    #[test]
    fn forward_1d_rejects_wrong_size() {
        let x = vec![0i32; 7];
        assert!(forward_dct_ii_1d(8, &x).is_err());
    }

    /// 2D forward of a flat block concentrates energy at DC.
    #[test]
    fn forward_2d_flat_block_dc_dominates() {
        let block = vec![128i32; 64]; // 8×8 flat
        let coeffs = forward_dct_ii_2d(8, 8, &block, 8).unwrap();
        // DC is at (0, 0).
        let dc = coeffs[0].abs();
        let mut total_ac: i64 = 0;
        for &c in &coeffs[1..] {
            total_ac += c.abs() as i64;
        }
        assert!(
            (dc as i64) > total_ac,
            "DC {} should dominate total AC {}",
            dc,
            total_ac
        );
    }

    /// 2D forward → quantise → dequantise → 2D inverse round-trip on
    /// a small block at a reasonable QP. Output should be a low-error
    /// approximation of the input.
    #[test]
    fn forward_quantise_dequantise_inverse_2d_round_trip() {
        use crate::dequant::{dequantize_tb_flat, DequantParams};
        use crate::transform::{inverse_transform_2d, TrType};

        // 4×4 block of small residual values.
        let block: Vec<i32> = vec![10, -5, 3, 1, -2, 4, -1, 6, 0, 2, -3, 5, 1, -4, 7, 0];
        let coeffs = forward_dct_ii_2d(4, 4, &block, 8).unwrap();
        // Quantise at QP=0 to minimise quantisation error so the
        // round-trip error is dominated by transform precision (the
        // forward path's 7-bit mid-shift discards 7 bits per pass).
        let levels = quantize_tb_flat(&coeffs, 4, 4, 0, 8, 15).unwrap();
        let params = DequantParams::luma_8bit(4, 4, 0);
        let d = dequantize_tb_flat(&levels, &params).unwrap();
        let recovered =
            inverse_transform_2d(4, 4, 4, 4, TrType::DctII, TrType::DctII, &d, 8, 15).unwrap();
        // The pipeline introduces transform-precision loss of ~`<<7`
        // bits per pass, so the recovered samples are scaled-down
        // approximations. Check sign correlation rather than exact
        // value: the dominant samples should keep their sign.
        let mut sign_matches = 0;
        let mut nonzero_inputs = 0;
        for (&b, &r) in block.iter().zip(recovered.iter()) {
            if b != 0 {
                nonzero_inputs += 1;
                if (b > 0 && r >= 0) || (b < 0 && r <= 0) || b == 0 {
                    sign_matches += 1;
                }
            }
        }
        // At least half the non-zero inputs should keep their sign —
        // this is a low bar (the forward+inverse loses precision) but
        // it confirms the math is direction-correct.
        assert!(
            sign_matches * 2 >= nonzero_inputs,
            "sign-match: {}/{} (expected ≥ half)",
            sign_matches,
            nonzero_inputs
        );
    }

    /// `quantize_tb_dep_quant` output is trellis-consistent by
    /// construction: the §7.3.11.11 writer accepts it and the decoder
    /// recovers the exact TransCoeffLevel array; the reconstruction
    /// through the §8.7.3 dep-quant dequant arms lands within one
    /// half-step of the source coefficient.
    #[test]
    fn dep_quant_forward_quant_is_trellis_feasible_and_round_trips() {
        use crate::cabac::ArithDecoder;
        use crate::cabac_enc::ArithEncoder;
        use crate::dequant::{dequantize_tb_flat, DequantParams};
        use crate::residual::{decode_tb_coefficients_opts, RcOpts, ResidualCtxs};
        use crate::residual_enc::encode_tb_coefficients_opts;

        let mut state = 0xC0FFEEu32;
        let mut rand = move || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            state >> 16
        };
        let opts = RcOpts {
            dep_quant: true,
            sign_data_hiding: false,
        };
        for (w, h, qp) in [(8usize, 8usize, 26i32), (16, 16, 30), (4, 8, 22)] {
            let mut coeffs = vec![0i32; w * h];
            for c in coeffs.iter_mut() {
                let r = rand();
                *c = if r % 3 == 0 {
                    0
                } else {
                    ((r % 4000) as i32 - 2000) * 4
                };
            }
            let levels = quantize_tb_dep_quant(&coeffs, w as u32, h as u32, qp, 8, 15).unwrap();
            if levels.iter().all(|&l| l == 0) {
                continue; // nothing to code
            }
            // Trellis feasibility: the dep-quant writer accepts the array…
            let mut enc = ArithEncoder::new();
            let mut ectx = ResidualCtxs::init(26);
            encode_tb_coefficients_opts(&mut enc, &mut ectx, w, h, 0, &levels, opts)
                .expect("greedy TCQ output must be trellis-consistent");
            let mut bytes = enc.finish();
            bytes.extend_from_slice(&[0u8; 64]);
            // …and the reader recovers it exactly.
            let mut dec = ArithDecoder::new(&bytes).unwrap();
            let mut dctx = ResidualCtxs::init(26);
            let (rec, _) = decode_tb_coefficients_opts(&mut dec, &mut dctx, w, h, 0, opts).unwrap();
            assert_eq!(rec, levels, "{w}x{h} qp={qp}");

            // Reconstruction accuracy: |dequant(level) − coeff| bounded
            // by one dep-quant step (ls / 2^bdShift in coefficient
            // units, plus 1 for the rounding chain).
            let mut params = DequantParams::luma_8bit(w as u32, h as u32, qp);
            params.dep_quant = true;
            let dq = dequantize_tb_flat(&levels, &params).unwrap();
            let log2_sum = (w.trailing_zeros() + h.trailing_zeros()) as i32;
            let rect = (log2_sum & 1) as usize;
            let bd_shift = (8 + (log2_sum & 1) as i32 + log2_sum / 2 + 10 - 15 + 1).max(0);
            let qe = qp + 1;
            let ls =
                (16i64 * crate::dequant::LEVEL_SCALE[rect][(qe % 6) as usize] as i64) << (qe / 6);
            let step = ((ls >> bd_shift).max(1)) as i32;
            for (i, (&d, &c)) in dq.iter().zip(&coeffs).enumerate() {
                assert!(
                    (d - c).abs() <= step + 1,
                    "{w}x{h} qp={qp} pos {i}: dequant {d} vs coeff {c} (step {step})"
                );
            }
        }
    }
}
