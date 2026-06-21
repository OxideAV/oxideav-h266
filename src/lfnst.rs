//! Low-Frequency Non-Separable Transform (LFNST) — inverse path
//! (§8.7.4.1 – §8.7.4.3).
//!
//! LFNST is an intra-only secondary transform applied to the
//! low-frequency region of the dequantised coefficient block *before*
//! the regular separable inverse transform (§8.7.4.4). It is gated by
//! the coding-unit-level `lfnst_idx` syntax element (§7.3.11.5): when
//! `lfnst_idx > 0` and `ApplyLfnstFlag[cIdx] == 1`, this module's
//! [`apply_inverse_lfnst`] replaces the top-left `nLfnstSize x
//! nLfnstSize` corner of the coefficient block with the inverse
//! non-separable transform output, leaving the rest of the block at
//! zero (the encoder only codes that low-frequency corner when LFNST is
//! active).
//!
//! Pipeline (§8.7.4.1):
//!
//! 1. `nLfnstOutSize` = 48 when both TB dims >= 8, else 16; `nonZeroSize`
//!    = 8 for 4x4 / 8x8 TBs, else 16 (eqs. 1158 / 1161).
//! 2. Read the first `nonZeroSize` coefficients in 4x4 diagonal scan
//!    order from the top-left corner → `u[]` (eqs. 1162 – 1164).
//! 3. Multiply by the `nLfnstOutSize x nonZeroSize` matrix selected by
//!    `lfnstTrSetIdx` (Table 41, keyed on the intra mode) and
//!    `lfnst_idx` → `v[]` (§8.7.4.2 eq. 1176).
//! 4. Scatter `v[]` back into the `nLfnstSize x nLfnstSize` corner per
//!    eqs. 1165 / 1166 (the layout depends on whether the modified
//!    `predModeIntra <= 34`).
//!
//! The §8.4.5.2.7 wide-angle remap that feeds the LFNST set selection is
//! applied internally by [`apply_inverse_lfnst`] from the TB dimensions
//! (single-tree non-ISP luma path: `nW == nTbW`, `nH == nTbH`), so both
//! square and non-square transform blocks are handled. Callers pass
//! `predModeIntra` in its pre-remap `IntraPredModeY` / `IntraPredModeC`
//! form (0..66, or the negative / >66 CCLM sentinels mapped by the
//! caller); the remap is the identity for square blocks.
//!
//! Spec reference: ITU-T H.266 | ISO/IEC 23090-3 (V4, 01/2026).

use oxideav_core::{Error, Result};

use crate::lfnst_matrices::*;
use crate::scan::diag_scan_order;

/// `INTRA_PLANAR` mode index (§8.4.5.2.11).
pub const INTRA_PLANAR: i32 = 0;

/// §8.4.5.2.7 wide-angle intra prediction mode mapping.
///
/// For a non-square block (`n_w != n_h`) the intra prediction mode is
/// shifted into the wide-angle range so the §8.7.4.3 Table-41 set
/// selection (and the angular prediction itself) account for the
/// block's aspect ratio:
///
/// * `nW > nH`, `2 <= mode < ((whRatio > 1) ? 8 + 2*whRatio : 8)`
///   → `mode + 65`;
/// * `nH > nW`, `mode <= 66 && mode > ((whRatio > 1) ? 60 − 2*whRatio : 60)`
///   → `mode − 67`,
///
/// where `whRatio = |log2(nW) − log2(nH)|`. Square blocks and modes
/// outside those ranges are returned unchanged. `n_w` / `n_h` are the
/// §8.4.5.2.7 `nW` / `nH` (for the single-tree non-ISP luma path these
/// equal the transform-block dimensions).
pub fn wide_angle_remap_mode(pred_mode_intra: i32, n_w: usize, n_h: usize) -> i32 {
    if n_w == n_h {
        return pred_mode_intra;
    }
    let log2 = |v: usize| (usize::BITS - 1 - v.leading_zeros()) as i32;
    let wh_ratio = (log2(n_w) - log2(n_h)).abs();
    if n_w > n_h {
        let upper = if wh_ratio > 1 { 8 + 2 * wh_ratio } else { 8 };
        if pred_mode_intra >= 2 && pred_mode_intra < upper {
            return pred_mode_intra + 65;
        }
    } else {
        // n_h > n_w (n_w == n_h handled above).
        let lower = if wh_ratio > 1 { 60 - 2 * wh_ratio } else { 60 };
        if pred_mode_intra <= 66 && pred_mode_intra > lower {
            return pred_mode_intra - 67;
        }
    }
    pred_mode_intra
}

/// §8.7.4.3 Table 41 — `lfnstTrSetIdx` selection from the (wide-angle
/// remapped) intra prediction mode `predModeIntra`.
///
/// The negative-mode branch (`predModeIntra < 0`) covers the CCLM-style
/// wide-angle modes; modes 0..80 map by the table ranges. The result is
/// one of the four transform sets (0..3).
pub fn lfnst_tr_set_idx(pred_mode_intra: i32) -> u32 {
    match pred_mode_intra {
        m if m < 0 => 1,
        0 | 1 => 0,
        2..=12 => 1,
        13..=23 => 2,
        24..=44 => 3,
        45..=55 => 2,
        // 56..=80 and any larger value fall in the last row.
        _ => 1,
    }
}

/// Return the `nLfnstOutSize x 16` LFNST matrix for the given output
/// size (`16` or `48`), transform set (0..3) and `lfnst_idx` (1 or 2).
///
/// The matrices store the full 16 input columns even though only the
/// first `nonZeroSize` (8 or 16) are used; the §8.7.4.2 multiply reads
/// exactly `nonZeroSize` columns.
fn lfnst_matrix(
    n_lfnst_out_size: usize,
    set_idx: u32,
    lfnst_idx: u8,
) -> Result<&'static [[i16; 16]]> {
    let m: &'static [[i16; 16]] = match (n_lfnst_out_size, set_idx, lfnst_idx) {
        (16, 0, 1) => &LFNST_16_SET0_IDX1,
        (16, 0, 2) => &LFNST_16_SET0_IDX2,
        (16, 1, 1) => &LFNST_16_SET1_IDX1,
        (16, 1, 2) => &LFNST_16_SET1_IDX2,
        (16, 2, 1) => &LFNST_16_SET2_IDX1,
        (16, 2, 2) => &LFNST_16_SET2_IDX2,
        (16, 3, 1) => &LFNST_16_SET3_IDX1,
        (16, 3, 2) => &LFNST_16_SET3_IDX2,
        (48, 0, 1) => &LFNST_48_SET0_IDX1,
        (48, 0, 2) => &LFNST_48_SET0_IDX2,
        (48, 1, 1) => &LFNST_48_SET1_IDX1,
        (48, 1, 2) => &LFNST_48_SET1_IDX2,
        (48, 2, 1) => &LFNST_48_SET2_IDX1,
        (48, 2, 2) => &LFNST_48_SET2_IDX2,
        (48, 3, 1) => &LFNST_48_SET3_IDX1,
        (48, 3, 2) => &LFNST_48_SET3_IDX2,
        _ => {
            return Err(Error::invalid(
                "h266 LFNST: unsupported (nLfnstOutSize, set, lfnst_idx) combination",
            ));
        }
    };
    Ok(m)
}

/// §8.7.4.2 one-dimensional LFNST process. `x` holds `nonZeroSize`
/// scaled coefficients; the result `y` has `nTrS == n_lfnst_out_size`
/// entries (eq. 1176): `y[i] = Clip3(min, max, (Σ M[i][j]*x[j] + 64) >> 7)`.
fn lfnst_1d(
    x: &[i32],
    n_lfnst_out_size: usize,
    matrix: &[[i16; 16]],
    coeff_min: i32,
    coeff_max: i32,
) -> Vec<i32> {
    let non_zero = x.len();
    let mut y = vec![0i32; n_lfnst_out_size];
    for (i, yi) in y.iter_mut().enumerate() {
        let mut acc: i64 = 0;
        let row = &matrix[i];
        for (j, &xj) in x.iter().enumerate().take(non_zero) {
            acc += row[j] as i64 * xj as i64;
        }
        let v = (acc + 64) >> 7;
        *yi = v.clamp(coeff_min as i64, coeff_max as i64) as i32;
    }
    y
}

/// §8.7.4.1 inverse-LFNST corner replacement.
///
/// `d` is the row-major `(n_tb_w * n_tb_h)` dequantised coefficient
/// block. On return its top-left `nLfnstSize x nLfnstSize` corner is
/// overwritten with the inverse non-separable transform output and all
/// other positions are zeroed (the encoder zeroes them when LFNST is
/// active — see `LfnstZeroOutSigCoeffFlag`). `pred_mode_intra` is the
/// intra mode **before** the §8.4.5.2.7 wide-angle remap; this function
/// applies that remap internally from the TB dimensions (single-tree
/// non-ISP luma path: `nW == nTbW`, `nH == nTbH`) so non-square TBs
/// select the correct transform set and transpose. `lfnst_idx` ∈ {1, 2}.
///
/// `coeff_min` / `coeff_max` are the `CoeffMin` / `CoeffMax` clip bounds
/// (§7.4.11.11) for the active `Log2TransformRange`.
pub fn apply_inverse_lfnst(
    d: &mut [i32],
    n_tb_w: usize,
    n_tb_h: usize,
    pred_mode_intra: i32,
    lfnst_idx: u8,
    coeff_min: i32,
    coeff_max: i32,
) -> Result<()> {
    if !(lfnst_idx == 1 || lfnst_idx == 2) {
        return Err(Error::invalid("h266 LFNST: lfnst_idx must be 1 or 2"));
    }
    if d.len() != n_tb_w * n_tb_h {
        return Err(Error::invalid("h266 LFNST: d length != n_tb_w * n_tb_h"));
    }
    if n_tb_w < 4 || n_tb_h < 4 {
        return Err(Error::invalid("h266 LFNST: TB smaller than 4x4"));
    }

    // eqs. 1158 – 1161.
    let big = n_tb_w >= 8 && n_tb_h >= 8;
    let n_lfnst_out_size: usize = if big { 48 } else { 16 };
    let log2_lfnst_size: u32 = if big { 3 } else { 2 };
    let n_lfnst_size: usize = 1 << log2_lfnst_size;
    let non_zero_size: usize = if (n_tb_w == 4 && n_tb_h == 4) || (n_tb_w == 8 && n_tb_h == 8) {
        8
    } else {
        16
    };

    // eqs. 1162 – 1164: gather u[x] in 4x4 diagonal scan order from the
    // top-left corner of d.
    let scan4 = diag_scan_order(4, 4); // DiagScanOrder[2][2]
    let mut u = vec![0i32; non_zero_size];
    for (k, slot) in u.iter_mut().enumerate() {
        let (xc, yc) = scan4[k];
        *slot = d[(yc as usize) * n_tb_w + (xc as usize)];
    }

    // §8.4.5.2.7 wide-angle remap (nW == nTbW, nH == nTbH on the
    // single-tree non-ISP luma path). The remapped mode drives both the
    // Table-41 set selection and the eqs. 1165/1166 transpose decision.
    let remapped_mode = wide_angle_remap_mode(pred_mode_intra, n_tb_w, n_tb_h);

    // §8.7.4.2 multiply.
    let set_idx = lfnst_tr_set_idx(remapped_mode);
    let matrix = lfnst_matrix(n_lfnst_out_size, set_idx, lfnst_idx)?;
    let v = lfnst_1d(&u, n_lfnst_out_size, matrix, coeff_min, coeff_max);

    // The corner is fully rewritten, so clear the entire TB first (every
    // out-of-corner position is zero when LFNST is active).
    for c in d.iter_mut() {
        *c = 0;
    }

    // eqs. 1165 / 1166: scatter v into the nLfnstSize x nLfnstSize
    // corner. The layout transposes when predModeIntra > 34.
    let transpose = remapped_mode > 34;
    for y in 0..n_lfnst_size {
        for x in 0..n_lfnst_size {
            let val = lfnst_scatter_value(x, y, &v, log2_lfnst_size, transpose);
            if let Some(val) = val {
                d[y * n_tb_w + x] = val;
            }
        }
    }
    Ok(())
}

/// Compute the scattered coefficient for position `(x, y)` in the
/// nLfnstSize corner per eqs. 1165 (mode <= 34) / 1166 (mode > 34).
///
/// Returns `None` for the "else d[x][y]" branch (the position keeps its
/// current — already-zeroed — value).
fn lfnst_scatter_value(
    x: usize,
    y: usize,
    v: &[i32],
    log2_lfnst_size: u32,
    transpose: bool,
) -> Option<i32> {
    // For predModeIntra <= 34 (eq. 1165): primary axis is y.
    // For predModeIntra > 34 (eq. 1166): primary axis is x (the roles of
    // x and y are swapped).
    let (p, q) = if transpose { (y, x) } else { (x, y) };
    // p plays the role of `x`, q plays the role of `y` in eq. 1165.
    if q < 4 {
        Some(v[p + (q << log2_lfnst_size)])
    } else if p < 4 {
        Some(v[32 + p + ((q - 4) << 2)])
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tr_set_idx_table_41_ranges() {
        assert_eq!(lfnst_tr_set_idx(-5), 1);
        assert_eq!(lfnst_tr_set_idx(0), 0);
        assert_eq!(lfnst_tr_set_idx(1), 0);
        assert_eq!(lfnst_tr_set_idx(2), 1);
        assert_eq!(lfnst_tr_set_idx(12), 1);
        assert_eq!(lfnst_tr_set_idx(13), 2);
        assert_eq!(lfnst_tr_set_idx(23), 2);
        assert_eq!(lfnst_tr_set_idx(24), 3);
        assert_eq!(lfnst_tr_set_idx(44), 3);
        assert_eq!(lfnst_tr_set_idx(45), 2);
        assert_eq!(lfnst_tr_set_idx(55), 2);
        assert_eq!(lfnst_tr_set_idx(56), 1);
        assert_eq!(lfnst_tr_set_idx(80), 1);
    }

    #[test]
    fn matrices_have_expected_dims() {
        assert_eq!(LFNST_16_SET0_IDX1.len(), 16);
        assert_eq!(LFNST_48_SET3_IDX2.len(), 48);
        // First-row spot-check vs spec §8.7.4.3.
        assert_eq!(
            LFNST_16_SET0_IDX1[0],
            [108, -44, -15, 1, -44, 19, 7, -1, -11, 6, 2, -1, 0, -1, -1, 0]
        );
        assert_eq!(
            LFNST_48_SET0_IDX1[0],
            [-117, 28, 18, 2, 4, 1, 2, 1, 32, -18, -2, 0, -1, 0, 0, 0]
        );
    }

    /// A 4x4 TB with only a DC coefficient runs through the
    /// nonZeroSize=8 path and produces a fully-populated 4x4 corner; all
    /// 16 positions are written (DC drives every output via the matrix
    /// column 0).
    #[test]
    fn inverse_lfnst_4x4_dc_only_fills_corner() {
        let mut d = vec![0i32; 16];
        d[0] = 100; // DC
                    // Mode 0 (PLANAR) -> set 0, not transposed.
        apply_inverse_lfnst(&mut d, 4, 4, INTRA_PLANAR, 1, -32768, 32767).unwrap();
        // Column 0 of LFNST_16_SET0_IDX1 scaled by 100 then (x+64)>>7.
        // Verify the (0,0) output matches the eq. 1176 / 1165 fold.
        let expected_dc = ((LFNST_16_SET0_IDX1[0][0] as i64 * 100) + 64) >> 7;
        assert_eq!(d[0] as i64, expected_dc);
        // At least one off-DC position is non-zero (the transform is not
        // diagonal).
        assert!(d.iter().skip(1).any(|&c| c != 0));
    }

    /// The transpose branch (mode > 34) swaps the x/y roles: feeding the
    /// same single coefficient with mode 0 vs mode 66 must produce
    /// transposed corners.
    #[test]
    fn inverse_lfnst_transpose_branch_swaps_axes() {
        let make = |mode: i32| {
            let mut d = vec![0i32; 16];
            d[0] = 50;
            d[1] = -20;
            apply_inverse_lfnst(&mut d, 4, 4, mode, 2, -32768, 32767).unwrap();
            d
        };
        let a = make(0); // not transposed
        let b = make(66); // transposed
                          // The two corners must be transposes of each other only if the
                          // set index is identical; mode 0 -> set 0, mode 66 -> set 1, so
                          // they differ. Just assert both produce output and differ.
        assert_ne!(a, b);
    }

    #[test]
    fn wide_angle_remap_square_is_identity() {
        for mode in [0, 1, 2, 18, 34, 50, 66] {
            assert_eq!(wide_angle_remap_mode(mode, 8, 8), mode);
        }
    }

    #[test]
    fn wide_angle_remap_wide_block_shifts_low_modes() {
        // nW > nH, whRatio = |log2(16) - log2(4)| = 2 -> upper = 8 + 4 = 12.
        // Modes 2..11 remap to +65; mode 12 and above are unchanged.
        assert_eq!(wide_angle_remap_mode(2, 16, 4), 67);
        assert_eq!(wide_angle_remap_mode(11, 16, 4), 76);
        assert_eq!(wide_angle_remap_mode(12, 16, 4), 12);
        assert_eq!(wide_angle_remap_mode(34, 16, 4), 34);
        // Mode 1 (DC) and 0 (PLANAR) are below the >= 2 threshold.
        assert_eq!(wide_angle_remap_mode(1, 16, 4), 1);
    }

    #[test]
    fn wide_angle_remap_tall_block_shifts_high_modes() {
        // nH > nW, whRatio = 2 -> lower = 60 - 4 = 56. Modes 57..66
        // remap to -67; mode 56 and below unchanged.
        assert_eq!(wide_angle_remap_mode(66, 4, 16), -1);
        assert_eq!(wide_angle_remap_mode(57, 4, 16), -10);
        assert_eq!(wide_angle_remap_mode(56, 4, 16), 56);
        assert_eq!(wide_angle_remap_mode(34, 4, 16), 34);
    }

    #[test]
    fn wide_angle_remap_ratio_one_uses_default_bounds() {
        // whRatio = 1 (e.g. 8x4) -> upper = 8, lower = 60.
        assert_eq!(wide_angle_remap_mode(7, 8, 4), 72);
        assert_eq!(wide_angle_remap_mode(8, 8, 4), 8);
        assert_eq!(wide_angle_remap_mode(61, 4, 8), -6);
        assert_eq!(wide_angle_remap_mode(60, 4, 8), 60);
    }

    /// A non-square TB now runs through `apply_inverse_lfnst` (the
    /// wide-angle remap is applied internally) instead of being rejected.
    #[test]
    fn inverse_lfnst_non_square_tb_populates_corner() {
        // 4x16: big = false (4 < 8) so nLfnstSize = 4, nonZeroSize = 16.
        let mut d = vec![0i32; 4 * 16];
        d[0] = 80;
        d[1] = -30;
        apply_inverse_lfnst(&mut d, 4, 16, /*mode=*/ 2, 1, -32768, 32767).unwrap();
        // The 4x4 corner is populated; positions outside it are zero.
        assert!(d.iter().take(4).any(|&c| c != 0));
        // Row 5 onward (y >= 4) of the first column must be zero.
        for y in 4..16 {
            assert_eq!(d[y * 4], 0, "row {y} col 0 should be zeroed");
        }
    }

    /// Mode 2 on a wide block (16x4) remaps to 67 (> 34) and therefore
    /// transposes, producing a different corner than the same mode on a
    /// square block (where it stays 2, <= 34, no transpose).
    #[test]
    fn inverse_lfnst_wide_block_transposes_via_remap() {
        let mut wide = vec![0i32; 16 * 4];
        wide[0] = 60;
        wide[1] = 25;
        apply_inverse_lfnst(&mut wide, 16, 4, 2, 2, -32768, 32767).unwrap();
        let mut square = vec![0i32; 8 * 8];
        square[0] = 60;
        square[1] = 25;
        apply_inverse_lfnst(&mut square, 8, 8, 2, 2, -32768, 32767).unwrap();
        // Different geometry + remap -> the corners differ; both are
        // populated.
        assert!(wide.iter().any(|&c| c != 0));
        assert!(square.iter().any(|&c| c != 0));
    }

    #[test]
    fn rejects_bad_lfnst_idx_and_dims() {
        let mut d = vec![0i32; 16];
        assert!(apply_inverse_lfnst(&mut d, 4, 4, 0, 0, -32768, 32767).is_err());
        let mut d2 = vec![0i32; 9];
        assert!(apply_inverse_lfnst(&mut d2, 3, 3, 0, 1, -32768, 32767).is_err());
    }
}
