//! VVC §8.7.3 "Scaling process for transform coefficients".
//!
//! This module implements the spec's dz → d dequantisation pipeline
//! (eqs. 1141 – 1156) that maps the decoded `TransCoeffLevel[]` integer
//! levels to the scaled-transform-coefficient array `d[]` consumed by
//! the inverse transform (§8.7.4).
//!
//! The flat-scaling-list subset — `sh_explicit_scaling_list_used_flag`
//! = 0 — is implemented in full. Non-flat scaling lists
//! (`ScalingMatrixRec[id]` / `ScalingMatrixDcRec[id-14]`) are surfaced
//! via [`dequantize_tb_with_scaling_list`] but the per-block
//! `id / log2MatrixSize` derivation from Table 38 is left to the
//! caller; this scaffold takes a pre-built `m[x][y]` array instead.
//!
//! Transform-skip (§8.7.3.1 with `rectNonTsFlag = 0`, `bdShift = 10`)
//! and BDPCM accumulation (eqs. 1153 / 1154) are now implemented; both
//! dequantise paths consume the [`DequantParams::transform_skip`] /
//! [`DequantParams::bdpcm`] flags and apply the spec-exact eqs.
//! `BdpcmDir` is wired via [`DequantParams::bdpcm_dir`] (false →
//! horizontal eq. 1153, true → vertical eq. 1154).
//!
//! Spec reference: ITU-T H.266 | ISO/IEC 23090-3 (V4, 01/2026).

use oxideav_core::{Error, Result};

/// `levelScale[rectNonTsFlag][qP % 6]` from §8.7.3 — the fixed
/// 2 × 6 table used by eq. 1152. Row 0 corresponds to square (or
/// `(log2W + log2H)` even) TBs, row 1 to rectangular TBs.
pub const LEVEL_SCALE: [[u32; 6]; 2] = [[40, 45, 51, 57, 64, 72], [57, 64, 72, 80, 90, 102]];

/// Prediction mode flavour used by [`scaling_matrix_id`] to index into
/// Table 38 (§8.7.3). INTER and IBC share the same row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PredModeKind {
    Intra,
    InterOrIbc,
}

/// §8.7.3 Table 38 derivation of the scaling-matrix identifier `id`
/// from `(predMode, cIdx, nTbW, nTbH)`. The result feeds eq. 1148
/// (`log2MatrixSize = id < 2 ? 1 : id < 8 ? 2 : 3`) and indexes
/// `ScalingMatrixRec[id]` in eq. 1149.
///
/// Returns `None` when the `(predMode, cIdx, max(nTbW, nTbH))` triple
/// is outside the table (e.g. INTRA with max side 2 — INTRA does not
/// use the 2×N chroma-only entries).
pub fn scaling_matrix_id(
    pred_mode: PredModeKind,
    c_idx: u32,
    n_tb_w: u32,
    n_tb_h: u32,
) -> Option<u32> {
    let max_side = n_tb_w.max(n_tb_h);
    match (pred_mode, c_idx, max_side) {
        // INTER / IBC, chroma-only Max = 2.
        (PredModeKind::InterOrIbc, 1, 2) => Some(0),
        (PredModeKind::InterOrIbc, 2, 2) => Some(1),
        // Max = 4.
        (PredModeKind::Intra, 0, 4) => Some(2),
        (PredModeKind::Intra, 1, 4) => Some(3),
        (PredModeKind::Intra, 2, 4) => Some(4),
        (PredModeKind::InterOrIbc, 0, 4) => Some(5),
        (PredModeKind::InterOrIbc, 1, 4) => Some(6),
        (PredModeKind::InterOrIbc, 2, 4) => Some(7),
        // Max = 8.
        (PredModeKind::Intra, 0, 8) => Some(8),
        (PredModeKind::Intra, 1, 8) => Some(9),
        (PredModeKind::Intra, 2, 8) => Some(10),
        (PredModeKind::InterOrIbc, 0, 8) => Some(11),
        (PredModeKind::InterOrIbc, 1, 8) => Some(12),
        (PredModeKind::InterOrIbc, 2, 8) => Some(13),
        // Max = 16.
        (PredModeKind::Intra, 0, 16) => Some(14),
        (PredModeKind::Intra, 1, 16) => Some(15),
        (PredModeKind::Intra, 2, 16) => Some(16),
        (PredModeKind::InterOrIbc, 0, 16) => Some(17),
        (PredModeKind::InterOrIbc, 1, 16) => Some(18),
        (PredModeKind::InterOrIbc, 2, 16) => Some(19),
        // Max = 32.
        (PredModeKind::Intra, 0, 32) => Some(20),
        (PredModeKind::Intra, 1, 32) => Some(21),
        (PredModeKind::Intra, 2, 32) => Some(22),
        (PredModeKind::InterOrIbc, 0, 32) => Some(23),
        (PredModeKind::InterOrIbc, 1, 32) => Some(24),
        (PredModeKind::InterOrIbc, 2, 32) => Some(25),
        // Max = 64: Y has its own id, chroma reuses Max=32's.
        (PredModeKind::Intra, 0, 64) => Some(26),
        (PredModeKind::Intra, 1, 64) => Some(21),
        (PredModeKind::Intra, 2, 64) => Some(22),
        (PredModeKind::InterOrIbc, 0, 64) => Some(27),
        (PredModeKind::InterOrIbc, 1, 64) => Some(24),
        (PredModeKind::InterOrIbc, 2, 64) => Some(25),
        _ => None,
    }
}

/// §8.7.3 eq. 1148 — `log2MatrixSize` derived from the scaling-matrix
/// `id`. Returns 1 (2×2 matrix), 2 (4×4) or 3 (8×8).
pub fn log2_matrix_size(id: u32) -> u32 {
    if id < 2 {
        1
    } else if id < 8 {
        2
    } else {
        3
    }
}

/// §8.7.3 eq. 1149 — expand a `matrixSize × matrixSize`
/// `ScalingMatrixRec` into a `n_tb_w × n_tb_h` row-major `m[]` array.
///
/// `scaling_matrix_rec` is the matrix as stored in the APS
/// (row-major `matrix_size * matrix_size` with `matrix_size = 1 << log2`).
/// If `dc_override` is `Some`, `m[0][0]` is replaced with its value
/// (eq. 1150, applied when the caller-side `id > 13`).
pub fn expand_scaling_matrix(
    scaling_matrix_rec: &[u8],
    log2_matrix_size: u32,
    n_tb_w: u32,
    n_tb_h: u32,
    dc_override: Option<u8>,
) -> Result<Vec<u32>> {
    let matrix_size = 1u32 << log2_matrix_size;
    if scaling_matrix_rec.len() as u32 != matrix_size * matrix_size {
        return Err(Error::invalid(
            "h266 dequant: scaling_matrix_rec length does not match matrix_size^2",
        ));
    }
    let log2_w = n_tb_w.trailing_zeros();
    let log2_h = n_tb_h.trailing_zeros();
    let mut m = vec![0u32; (n_tb_w as usize) * (n_tb_h as usize)];
    for y in 0..n_tb_h {
        for x in 0..n_tb_w {
            // eq. 1149: i = (x << log2MatrixSize) >> log2 nTbW; j similar.
            let i = (x << log2_matrix_size) >> log2_w;
            let j = (y << log2_matrix_size) >> log2_h;
            let idx = (j as usize) * (matrix_size as usize) + (i as usize);
            m[(y as usize) * (n_tb_w as usize) + (x as usize)] = scaling_matrix_rec[idx] as u32;
        }
    }
    if let Some(dc) = dc_override {
        m[0] = dc as u32;
    }
    Ok(m)
}

/// Parameter bundle for [`dequantize_tb_flat`] and
/// [`dequantize_tb_with_scaling_list`]. Mirrors the §8.7.3 inputs.
#[derive(Clone, Copy, Debug)]
pub struct DequantParams {
    /// Bit depth of the component samples (e.g. 8, 10, 12).
    pub bit_depth: u32,
    /// `Log2TransformRange` — typically 15 for the MSB profile (see
    /// §7.4.3.2 note: `Log2TransformRange = Max(15, BitDepth + 6)`).
    pub log2_transform_range: u32,
    /// Transform block width (power of two, ≤ 64).
    pub n_tb_w: u32,
    /// Transform block height (power of two, ≤ 64).
    pub n_tb_h: u32,
    /// `qP` after the §8.7.3 Qp′Y / Qp′Cb / Qp′Cr selection + the
    /// Clip3 of eq. 1141 (non-TS) or 1144 (TS). Caller is responsible
    /// for plumbing the Qp′ path.
    pub qp: i32,
    /// `sh_dep_quant_used_flag` — only affects the `bdShift` +1 term
    /// and `ls[x][y]` ladder shift (eqs. 1143 / 1151 / 1152).
    pub dep_quant: bool,
    /// `transform_skip_flag[x][y][cIdx]`. When set, the spec uses the
    /// shorter `bdShift = 10`, `rectNonTsFlag = 0` ladder and the qP is
    /// clipped to `QpPrimeTsMin` from below (eqs. 1144 – 1146). This
    /// scaffold treats `qp` as already clipped (callers handle the
    /// `QpPrimeTsMin` floor at parse time).
    pub transform_skip: bool,
    /// `BdpcmFlag[x][y][cIdx]`. When set, eqs. 1153 / 1154 accumulate
    /// `dz[x][y]` along the prediction direction before scaling.
    pub bdpcm: bool,
    /// `BdpcmDir[x][y][cIdx]` — false = horizontal accumulation
    /// (eq. 1153), true = vertical accumulation (eq. 1154). Only
    /// consulted when `bdpcm` is true.
    pub bdpcm_dir: bool,
}

impl DequantParams {
    /// Helper: construct a flat-scaling-list parameter bundle for luma
    /// at 8-bit depth.
    pub fn luma_8bit(n_tb_w: u32, n_tb_h: u32, qp: i32) -> Self {
        Self {
            bit_depth: 8,
            log2_transform_range: 15,
            n_tb_w,
            n_tb_h,
            qp,
            dep_quant: false,
            transform_skip: false,
            bdpcm: false,
            bdpcm_dir: false,
        }
    }
}

/// `CoeffMin` / `CoeffMax` per §7.4.11.11: the symmetric
/// `Log2TransformRange`-bit signed range.
fn coeff_min_max(log2_transform_range: u32) -> (i32, i32) {
    let max = (1i64 << log2_transform_range) - 1;
    ((-(max + 1)) as i32, max as i32)
}

/// Clip3(lo, hi, x) helper matching the spec's definition.
#[inline]
fn clip3(lo: i32, hi: i32, x: i64) -> i32 {
    x.clamp(lo as i64, hi as i64) as i32
}

/// §8.7.3 dequantisation with flat (all-16) scaling list — the
/// `sh_explicit_scaling_list_used_flag == 0` path.
///
/// Input `levels` is a row-major `(n_tb_w * n_tb_h)` array of
/// `TransCoeffLevel[]` values. Returns the `d[]` array of the same
/// shape, clipped to the `Log2TransformRange` signed range per
/// eq. 1156.
pub fn dequantize_tb_flat(levels: &[i32], params: &DequantParams) -> Result<Vec<i32>> {
    let w = params.n_tb_w as usize;
    let h = params.n_tb_h as usize;
    if levels.len() != w * h {
        return Err(Error::invalid(
            "h266 dequant: input level array size does not match nTbW*nTbH",
        ));
    }
    if !w.is_power_of_two() || !h.is_power_of_two() {
        return Err(Error::invalid(
            "h266 dequant: nTbW and nTbH must be powers of two",
        ));
    }
    let log2_w = w.trailing_zeros();
    let log2_h = h.trailing_zeros();
    // eqs. 1142 / 1143 vs. 1145 / 1146 — transform-skip forces
    // rectNonTsFlag = 0 and bdShift = 10.
    let (rect_non_ts, bd_shift) = if params.transform_skip {
        (0u32, 10u32)
    } else {
        let rect_non_ts = (((log2_w + log2_h) & 1) == 1) as u32;
        let bd_shift =
            (params.bit_depth as i32 + rect_non_ts as i32 + ((log2_w + log2_h) as i32) / 2 + 10
                - params.log2_transform_range as i32
                + params.dep_quant as i32)
                .max(0) as u32;
        (rect_non_ts, bd_shift)
    };
    // eq. 1147
    let bd_offset = if bd_shift == 0 {
        0
    } else {
        1i64 << (bd_shift - 1)
    };
    // levelScale[rectNonTsFlag][qP%6] << (qP/6) — or the dep_quant
    // variant that uses (qP+1)%6 / (qP+1)/6 (eq. 1151). dep_quant is
    // ignored in transform-skip mode (eq. 1152).
    let (q_mod, q_div) = if params.dep_quant && !params.transform_skip {
        ((params.qp + 1).rem_euclid(6) as u32, (params.qp + 1) / 6)
    } else {
        (params.qp.rem_euclid(6) as u32, params.qp / 6)
    };
    let level_scale = LEVEL_SCALE[rect_non_ts as usize][q_mod as usize] as i64;
    // For flat scaling list, m[x][y] = 16 everywhere (spec NOTE on
    // eq. 1148; transform_skip = 1 also forces m = 16).
    let m = 16i64;
    let q_shift = q_div.max(0) as u32;
    let ls = (m * level_scale) << q_shift;

    let (coeff_min, coeff_max) = coeff_min_max(params.log2_transform_range);

    // BDPCM accumulation (eqs. 1153 / 1154) is applied to dz BEFORE the
    // scaling step. Build a working dz[] buffer (in-place is fine because
    // accumulation uses the previously-updated cell).
    let mut dz = vec![0i64; levels.len()];
    for (i, &v) in levels.iter().enumerate() {
        dz[i] = v as i64;
    }
    if params.bdpcm {
        if !params.bdpcm_dir {
            // Horizontal accumulation (eq. 1153): dz[x][y] += dz[x-1][y]
            // for x > 0, with Clip3 to the CoeffMin/CoeffMax range.
            for y in 0..h {
                for x in 1..w {
                    let sum = dz[y * w + x - 1] + dz[y * w + x];
                    dz[y * w + x] = clip3(coeff_min, coeff_max, sum) as i64;
                }
            }
        } else {
            // Vertical accumulation (eq. 1154): dz[x][y] += dz[x][y-1]
            // for y > 0.
            for y in 1..h {
                for x in 0..w {
                    let sum = dz[(y - 1) * w + x] + dz[y * w + x];
                    dz[y * w + x] = clip3(coeff_min, coeff_max, sum) as i64;
                }
            }
        }
    }

    let mut out = vec![0i32; levels.len()];
    for y in 0..h {
        for x in 0..w {
            // eq. 1155: dnc = (dz * ls + bdOffset) >> bdShift.
            let dnc = if bd_shift == 0 {
                dz[y * w + x] * ls
            } else {
                (dz[y * w + x] * ls + bd_offset) >> bd_shift
            };
            out[y * w + x] = clip3(coeff_min, coeff_max, dnc);
        }
    }
    Ok(out)
}

/// §8.7.3 dequantisation with a caller-supplied `m[x][y]` scaling
/// matrix (row-major `nTbW * nTbH`). Enables the non-flat scaling-list
/// path once the caller has derived `ScalingMatrixRec[id][i][j]` from
/// Table 38 and the APS.
pub fn dequantize_tb_with_scaling_list(
    levels: &[i32],
    m: &[u32],
    params: &DequantParams,
) -> Result<Vec<i32>> {
    let w = params.n_tb_w as usize;
    let h = params.n_tb_h as usize;
    if levels.len() != w * h || m.len() != w * h {
        return Err(Error::invalid(
            "h266 dequant: input array size does not match nTbW*nTbH",
        ));
    }
    if !w.is_power_of_two() || !h.is_power_of_two() {
        return Err(Error::invalid(
            "h266 dequant: nTbW and nTbH must be powers of two",
        ));
    }
    let log2_w = w.trailing_zeros();
    let log2_h = h.trailing_zeros();
    // Transform-skip overrides per §8.7.3 eqs. 1145 / 1146.
    let (rect_non_ts, bd_shift) = if params.transform_skip {
        (0u32, 10u32)
    } else {
        let rect_non_ts = (((log2_w + log2_h) & 1) == 1) as u32;
        let bd_shift =
            (params.bit_depth as i32 + rect_non_ts as i32 + ((log2_w + log2_h) as i32) / 2 + 10
                - params.log2_transform_range as i32
                + params.dep_quant as i32)
                .max(0) as u32;
        (rect_non_ts, bd_shift)
    };
    let bd_offset = if bd_shift == 0 {
        0
    } else {
        1i64 << (bd_shift - 1)
    };
    let (q_mod, q_div) = if params.dep_quant && !params.transform_skip {
        ((params.qp + 1).rem_euclid(6) as u32, (params.qp + 1) / 6)
    } else {
        (params.qp.rem_euclid(6) as u32, params.qp / 6)
    };
    let level_scale = LEVEL_SCALE[rect_non_ts as usize][q_mod as usize] as i64;
    let q_shift = q_div.max(0) as u32;
    let (coeff_min, coeff_max) = coeff_min_max(params.log2_transform_range);

    // BDPCM accumulation on dz before scaling.
    let mut dz = vec![0i64; levels.len()];
    for (i, &v) in levels.iter().enumerate() {
        dz[i] = v as i64;
    }
    if params.bdpcm {
        if !params.bdpcm_dir {
            for y in 0..h {
                for x in 1..w {
                    let sum = dz[y * w + x - 1] + dz[y * w + x];
                    dz[y * w + x] = clip3(coeff_min, coeff_max, sum) as i64;
                }
            }
        } else {
            for y in 1..h {
                for x in 0..w {
                    let sum = dz[(y - 1) * w + x] + dz[y * w + x];
                    dz[y * w + x] = clip3(coeff_min, coeff_max, sum) as i64;
                }
            }
        }
    }

    let mut out = vec![0i32; levels.len()];
    for y in 0..h {
        for x in 0..w {
            let mxy = m[y * w + x] as i64;
            let ls = (mxy * level_scale) << q_shift;
            let dnc = if bd_shift == 0 {
                dz[y * w + x] * ls
            } else {
                (dz[y * w + x] * ls + bd_offset) >> bd_shift
            };
            out[y * w + x] = clip3(coeff_min, coeff_max, dnc);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// With qP = 26 (the Qp′Y offset used elsewhere in the crate),
    /// bit_depth = 8, dep_quant = 0, on a square 4×4 TB (log2 sum = 4
    /// → rectNonTsFlag = 0):
    ///   bdShift = 8 + 0 + 4/2 + 10 − 15 + 0 = 5
    ///   levelScale[0][26%6] = levelScale[0][2] = 51
    ///   q_shift = 26/6 = 4
    ///   ls = 16 * 51 << 4 = 13056
    ///   bdOffset = 1 << 4 = 16
    ///
    /// A level of 1 at DC maps to (1 * 13056 + 16) >> 5 = 13072 >> 5 = 408.
    #[test]
    fn dequant_flat_luma_4x4_qp26_dc_impulse() {
        let mut levels = vec![0i32; 16];
        levels[0] = 1;
        let params = DequantParams::luma_8bit(4, 4, 26);
        let d = dequantize_tb_flat(&levels, &params).unwrap();
        assert_eq!(d[0], 408);
        for v in &d[1..] {
            assert_eq!(*v, 0);
        }
    }

    /// Negative levels scale symmetrically.
    #[test]
    fn dequant_flat_negative_dc() {
        let mut levels = vec![0i32; 16];
        levels[0] = -1;
        let params = DequantParams::luma_8bit(4, 4, 26);
        let d = dequantize_tb_flat(&levels, &params).unwrap();
        // With bdOffset added: (-13056 + 16) >> 5. Arithmetic shift
        // right rounds toward -∞ so −13040 >> 5 = −408.
        assert_eq!(d[0], -408);
    }

    /// qP = 0 edge: bdShift = 8 + 0 + 2 + 10 − 15 = 5. Same bd_shift
    /// but levelScale[0][0] = 40, q_shift = 0 → ls = 640. Level 1
    /// maps to (640 + 16) >> 5 = 20.
    #[test]
    fn dequant_flat_qp0() {
        let mut levels = vec![0i32; 16];
        levels[0] = 1;
        let params = DequantParams::luma_8bit(4, 4, 0);
        let d = dequantize_tb_flat(&levels, &params).unwrap();
        assert_eq!(d[0], 20);
    }

    /// Rectangular TB (4×8) → log2 sum = 5 → rectNonTsFlag = 1 → use
    /// row 1 of levelScale; bdShift += 1 relative to 4×4.
    ///   bdShift = 8 + 1 + 5/2 + 10 − 15 = 6
    ///   levelScale[1][26%6] = 72
    ///   q_shift = 4, ls = 16 * 72 << 4 = 18432
    ///   bdOffset = 1 << 5 = 32
    ///   Level 1 → (18432 + 32) >> 6 = 288.
    #[test]
    fn dequant_flat_rect_4x8_qp26() {
        let mut levels = vec![0i32; 32];
        levels[0] = 1;
        let params = DequantParams {
            bit_depth: 8,
            log2_transform_range: 15,
            n_tb_w: 4,
            n_tb_h: 8,
            qp: 26,
            dep_quant: false,
            transform_skip: false,
            bdpcm: false,
            bdpcm_dir: false,
        };
        let d = dequantize_tb_flat(&levels, &params).unwrap();
        assert_eq!(d[0], 288);
    }

    /// Transform-skip dequant: §8.7.3 eqs. 1145 / 1146 force
    /// rectNonTsFlag = 0 and bdShift = 10. With the flat scaling list,
    /// `m = 16` and bdOffset = 1<<9 = 512. At qP = 26, levelScale[0][2]
    /// = 51, q_shift = 4, ls = 16 * 51 << 4 = 13056. Level = 1 maps to
    /// (13056 + 512) >> 10 = 13568 >> 10 = 13.
    #[test]
    fn dequant_flat_transform_skip_4x4_qp26() {
        let mut levels = vec![0i32; 16];
        levels[0] = 1;
        let params = DequantParams {
            transform_skip: true,
            ..DequantParams::luma_8bit(4, 4, 26)
        };
        let d = dequantize_tb_flat(&levels, &params).unwrap();
        assert_eq!(d[0], 13);
    }

    /// BDPCM horizontal accumulation (eq. 1153) on a row of 1's: after
    /// accumulation `dz` becomes [1, 2, 3, 4] before scaling. With
    /// transform-skip + qP=26 4×4 the row scales 1, 2, 3, 4 → 13, 26,
    /// 40, 53 (each (dz*13056 + 512) >> 10).
    #[test]
    fn dequant_flat_bdpcm_horizontal_row() {
        let mut levels = vec![0i32; 16];
        for x in 0..4 {
            levels[x] = 1;
        }
        let params = DequantParams {
            transform_skip: true,
            bdpcm: true,
            bdpcm_dir: false,
            ..DequantParams::luma_8bit(4, 4, 26)
        };
        let d = dequantize_tb_flat(&levels, &params).unwrap();
        // (1*13056 + 512) >> 10 = 13
        // (2*13056 + 512) >> 10 = 26568 >> 10 = 25 (NOT 26 — round down)
        // Recompute: 2 * 13056 = 26112; 26112 + 512 = 26624; 26624 >> 10 = 26.
        // (3*13056 + 512) >> 10 = (39168 + 512) >> 10 = 39680 >> 10 = 38.
        // (4*13056 + 512) >> 10 = (52224 + 512) >> 10 = 52736 >> 10 = 51.
        assert_eq!(d[0], 13);
        assert_eq!(d[1], 26);
        assert_eq!(d[2], 38);
        assert_eq!(d[3], 51);
    }

    /// BDPCM vertical accumulation (eq. 1154) on a column of 1's: each
    /// row accumulates downwards.
    #[test]
    fn dequant_flat_bdpcm_vertical_column() {
        let mut levels = vec![0i32; 16];
        for y in 0..4 {
            levels[y * 4] = 1;
        }
        let params = DequantParams {
            transform_skip: true,
            bdpcm: true,
            bdpcm_dir: true,
            ..DequantParams::luma_8bit(4, 4, 26)
        };
        let d = dequantize_tb_flat(&levels, &params).unwrap();
        // Same scaling per cell, accumulated dz = 1, 2, 3, 4 down col 0.
        assert_eq!(d[0 * 4 + 0], 13);
        assert_eq!(d[1 * 4 + 0], 26);
        assert_eq!(d[2 * 4 + 0], 38);
        assert_eq!(d[3 * 4 + 0], 51);
    }

    /// BDPCM with all-zero levels remains all-zero post-accumulation.
    #[test]
    fn dequant_flat_bdpcm_zero_input_stays_zero() {
        let levels = vec![0i32; 16];
        let params = DequantParams {
            transform_skip: true,
            bdpcm: true,
            bdpcm_dir: false,
            ..DequantParams::luma_8bit(4, 4, 26)
        };
        let d = dequantize_tb_flat(&levels, &params).unwrap();
        assert!(d.iter().all(|&v| v == 0));
    }

    /// Bad input length surfaces Invalid.
    #[test]
    fn dequant_flat_rejects_wrong_size() {
        let levels = vec![0i32; 15];
        let params = DequantParams::luma_8bit(4, 4, 26);
        assert!(dequantize_tb_flat(&levels, &params).is_err());
    }

    /// Non-power-of-two TB is rejected.
    #[test]
    fn dequant_flat_rejects_non_pow2() {
        let levels = vec![0i32; 12];
        let params = DequantParams {
            bit_depth: 8,
            log2_transform_range: 15,
            n_tb_w: 3,
            n_tb_h: 4,
            qp: 26,
            dep_quant: false,
            transform_skip: false,
            bdpcm: false,
            bdpcm_dir: false,
        };
        assert!(dequantize_tb_flat(&levels, &params).is_err());
    }

    /// Clipping to CoeffMin / CoeffMax at Log2TransformRange = 15:
    /// range is [-32768, 32767]. Feed a level deliberately sized to
    /// overflow after scaling and verify clip kicks in.
    #[test]
    fn dequant_flat_clips_to_coeff_range() {
        // bd_shift = 5, ls = 13056 at qP=26 4×4. A level of 1000 gives
        // dz*ls = 13056000; (13056000 + 16) >> 5 = 408000. Log2Range = 15
        // → max = 32767 → value clips to 32767.
        let mut levels = vec![0i32; 16];
        levels[0] = 1000;
        let params = DequantParams::luma_8bit(4, 4, 26);
        let d = dequantize_tb_flat(&levels, &params).unwrap();
        assert_eq!(d[0], 32767);

        levels[0] = -1000;
        let d = dequantize_tb_flat(&levels, &params).unwrap();
        assert_eq!(d[0], -32768);
    }

    /// With a non-flat scaling matrix (caller-provided m[]), verify
    /// that doubling a single DC coefficient m[0][0] doubles the
    /// output at that position relative to the flat result.
    #[test]
    fn dequant_scaling_list_doubles_dc_when_m_is_doubled() {
        let mut levels = vec![0i32; 16];
        levels[0] = 1;
        let mut m = vec![16u32; 16];
        m[0] = 32;
        let params = DequantParams::luma_8bit(4, 4, 26);
        let d = dequantize_tb_with_scaling_list(&levels, &m, &params).unwrap();
        // m=32 → ls doubles → output doubles pre-rounding.
        // ls = 32 * 51 << 4 = 26112; (26112 + 16) >> 5 = 26128 >> 5 = 816.
        assert_eq!(d[0], 816);
    }

    /// level_scale table values are exactly the spec's list.
    #[test]
    fn level_scale_table_matches_spec() {
        assert_eq!(LEVEL_SCALE[0], [40, 45, 51, 57, 64, 72]);
        assert_eq!(LEVEL_SCALE[1], [57, 64, 72, 80, 90, 102]);
    }

    /// Table 38 spot-checks: the diagonal "luma INTRA" ids are 2, 8, 14,
    /// 20, 26 at max sides 4, 8, 16, 32, 64.
    #[test]
    fn scaling_matrix_id_intra_luma_diagonal() {
        assert_eq!(scaling_matrix_id(PredModeKind::Intra, 0, 4, 4), Some(2));
        assert_eq!(scaling_matrix_id(PredModeKind::Intra, 0, 8, 8), Some(8));
        assert_eq!(scaling_matrix_id(PredModeKind::Intra, 0, 16, 16), Some(14));
        assert_eq!(scaling_matrix_id(PredModeKind::Intra, 0, 32, 32), Some(20));
        assert_eq!(scaling_matrix_id(PredModeKind::Intra, 0, 64, 64), Some(26));
    }

    /// Table 38: INTER luma diagonal is 5, 11, 17, 23, 27.
    #[test]
    fn scaling_matrix_id_inter_luma_diagonal() {
        assert_eq!(
            scaling_matrix_id(PredModeKind::InterOrIbc, 0, 4, 4),
            Some(5)
        );
        assert_eq!(
            scaling_matrix_id(PredModeKind::InterOrIbc, 0, 8, 8),
            Some(11)
        );
        assert_eq!(
            scaling_matrix_id(PredModeKind::InterOrIbc, 0, 16, 16),
            Some(17)
        );
        assert_eq!(
            scaling_matrix_id(PredModeKind::InterOrIbc, 0, 32, 32),
            Some(23)
        );
        assert_eq!(
            scaling_matrix_id(PredModeKind::InterOrIbc, 0, 64, 64),
            Some(27)
        );
    }

    /// Table 38: chroma rows for max = 64 reuse the max = 32 ids.
    #[test]
    fn scaling_matrix_id_chroma_64_reuses_32() {
        assert_eq!(scaling_matrix_id(PredModeKind::Intra, 1, 64, 64), Some(21));
        assert_eq!(scaling_matrix_id(PredModeKind::Intra, 2, 64, 64), Some(22));
        assert_eq!(
            scaling_matrix_id(PredModeKind::InterOrIbc, 1, 64, 64),
            Some(24)
        );
    }

    /// INTRA at max = 2 is outside the table.
    #[test]
    fn scaling_matrix_id_intra_max2_is_none() {
        assert_eq!(scaling_matrix_id(PredModeKind::Intra, 0, 2, 2), None);
    }

    /// log2_matrix_size matches eq. 1148: id<2 → 1, id<8 → 2, else 3.
    #[test]
    fn log2_matrix_size_table() {
        assert_eq!(log2_matrix_size(0), 1);
        assert_eq!(log2_matrix_size(1), 1);
        assert_eq!(log2_matrix_size(2), 2);
        assert_eq!(log2_matrix_size(7), 2);
        assert_eq!(log2_matrix_size(8), 3);
        assert_eq!(log2_matrix_size(27), 3);
    }

    /// expand_scaling_matrix on a uniform 2×2 matrix duplicated over a
    /// 4×4 TB yields a flat `m[]` of the same value.
    #[test]
    fn expand_scaling_matrix_uniform_2x2_over_4x4() {
        let rec = vec![32u8; 4]; // 2×2 matrix with all 32
        let m = expand_scaling_matrix(&rec, 1, 4, 4, None).unwrap();
        assert_eq!(m.len(), 16);
        for v in m {
            assert_eq!(v, 32);
        }
    }

    /// expand_scaling_matrix expands a 4×4 matrix over an 8×8 TB (each
    /// matrix entry is replicated 2×2).
    #[test]
    fn expand_scaling_matrix_4x4_over_8x8_replicates() {
        // 4×4 matrix: values = row*4 + col + 1 (1..16).
        let rec: Vec<u8> = (1u8..=16).collect();
        let m = expand_scaling_matrix(&rec, 2, 8, 8, None).unwrap();
        // Position (0,0) through (1,1) all come from matrix cell (0,0) = 1.
        assert_eq!(m[0], 1);
        assert_eq!(m[1], 1);
        assert_eq!(m[8], 1); // row 1
        assert_eq!(m[9], 1);
        // (2, 0) → i = (2<<2)>>3 = 1 → matrix[0][1] = 2.
        assert_eq!(m[2], 2);
        // Bottom-right block: (7, 7) → i = j = (7<<2)>>3 = 3 → matrix[3][3] = 16.
        assert_eq!(m[7 * 8 + 7], 16);
    }

    /// DC override (eq. 1150) replaces only position (0, 0).
    #[test]
    fn expand_scaling_matrix_dc_override_only_touches_dc() {
        let rec = vec![32u8; 4]; // 2×2 matrix with all 32
        let m = expand_scaling_matrix(&rec, 1, 4, 4, Some(200)).unwrap();
        assert_eq!(m[0], 200);
        assert_eq!(m[1], 32);
        assert_eq!(m[15], 32);
    }

    /// Wrong matrix length is rejected.
    #[test]
    fn expand_scaling_matrix_rejects_wrong_length() {
        let rec = vec![16u8; 3];
        assert!(expand_scaling_matrix(&rec, 1, 4, 4, None).is_err());
    }
}
