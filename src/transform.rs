//! VVC inverse transform kernels (§8.7.4).
//!
//! Foundation drop: the small-size (4 / 8) kernels for DST-VII
//! (`trType = 1`) and DCT-VIII (`trType = 2`). The DCT-II kernel
//! (`trType = 0`) uses a 64×64 coefficient matrix and is deferred
//! along with the 16/32/64-point kernels — they add up to about
//! 4096 × (4 tables) spec coefficients; landing them cleanly is a
//! separate increment and they are not needed to begin exercising
//! the 4×4 residual path (4×4 intra luma TUs are DST-VII by
//! default, per §8.7.4.1 eqs. 1167/1168 when `nTbW` is in 4..=16).
//!
//! The exposed primitive is [`one_d_transform`], which implements
//! eq. 1178 from §8.7.4.4:
//!
//! ```text
//!   y[i] = sum_{j=0}^{nonZeroS-1} transMatrix[i][j] * x[j]
//! ```
//!
//! Matrix entries are exactly as printed in the spec (§8.7.4.5) with
//! sign-extended 8-bit values fitting in `i16`. Inputs are coefficient
//! arrays in `i32` (they may carry up to 15-bit magnitudes after
//! de-quantisation, plus room for the accumulation headroom).
//!
//! Composition into the full 2D inverse transform (`clause 8.7.4.1`)
//! — including the mid-shift of 7 and final shift to
//! `(20 - BitDepth)` — is out of this increment's scope and will land
//! next to the CU/TU walker that actually needs reconstructed residual.

use oxideav_core::{Error, Result};

/// Transform kernel identifier per §8.7.4.5.
///
/// * `DctII = 0` — inverse discrete cosine transform type II; used
///   for all sizes in 4..=64. Coefficient tables for this kernel
///   are deferred (see module docs).
/// * `DstVII = 1` — inverse discrete sine transform type VII; selected
///   by MTS (`mts_idx`) or the implicit small-TU default for widths
///   4..=16 (eqs. 1167/1168).
/// * `DctVIII = 2` — inverse discrete cosine transform type VIII;
///   selected by MTS only.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum TrType {
    DctII = 0,
    DstVII = 1,
    DctVIII = 2,
}

/// 4-point DST-VII matrix (eq. 1185).
#[rustfmt::skip]
pub const DST_VII_4: [[i16; 4]; 4] = [
    [ 29,  55,  74,  84],
    [ 74,  74,   0, -74],
    [ 84, -29, -74,  55],
    [ 55, -84,  74, -29],
];

/// 8-point DST-VII matrix (eq. 1186).
#[rustfmt::skip]
pub const DST_VII_8: [[i16; 8]; 8] = [
    [17,  32,  46,  60,  71,  78,  85,  86],
    [46,  78,  86,  71,  32, -17, -60, -85],
    [71,  85,  32, -46, -86, -60,  17,  78],
    [85,  46, -60, -78,  17,  86,  32, -71],
    [86, -17, -85,  32,  78, -46, -71,  60],
    [78, -71, -17,  85, -60, -32,  86, -46],
    [60, -86,  71, -17, -46,  85, -78,  32],
    [32, -60,  78, -86,  85, -71,  46, -17],
];

/// 4-point DCT-VIII matrix (eq. 1192).
#[rustfmt::skip]
pub const DCT_VIII_4: [[i16; 4]; 4] = [
    [84,  74,  55,  29],
    [74,   0, -74, -74],
    [55, -74, -29,  84],
    [29, -74,  84, -55],
];

/// 8-point DCT-VIII matrix (eq. 1193).
#[rustfmt::skip]
pub const DCT_VIII_8: [[i16; 8]; 8] = [
    [86,  85,  78,  71,  60,  46,  32,  17],
    [85,  60,  17, -32, -71, -86, -78, -46],
    [78,  17, -60, -86, -46,  32,  85,  71],
    [71, -32, -86, -17,  78,  60, -46, -85],
    [60, -71, -46,  78,  32, -85, -17,  86],
    [46, -86,  32,  60, -85,  17,  71, -78],
    [32, -78,  85, -46, -17,  71, -86,  60],
    [17, -46,  71, -85,  86, -78,  60, -32],
];

/// Apply one of the small-size DST-VII / DCT-VIII inverse transforms.
///
/// * `tr_type` — kernel selector. `DctII` is not implemented at this
///   increment and returns `Error::Unsupported`.
/// * `n_tb_s` — transform size; must be 4 or 8 for the non-DCT-II
///   kernels that are supported here. 16/32/64-point tables are
///   deferred.
/// * `non_zero_s` — horizontal size of non-zero coefficients; inputs
///   beyond this range are ignored by the sum.
/// * `x` — input coefficients (len ≥ `non_zero_s`).
///
/// Returns an owned `Vec<i32>` of length `n_tb_s`, storing `y[i]` per
/// eq. 1178.
pub fn one_d_transform(
    tr_type: TrType,
    n_tb_s: usize,
    non_zero_s: usize,
    x: &[i32],
) -> Result<Vec<i32>> {
    if non_zero_s > n_tb_s {
        return Err(Error::invalid(format!(
            "h266 transform: nonZeroS={non_zero_s} > nTbS={n_tb_s}"
        )));
    }
    if x.len() < non_zero_s {
        return Err(Error::invalid(format!(
            "h266 transform: input slice {} shorter than nonZeroS {non_zero_s}",
            x.len()
        )));
    }
    match (tr_type, n_tb_s) {
        (TrType::DstVII, 4) => Ok(apply_matrix(&DST_VII_4, non_zero_s, x)),
        (TrType::DstVII, 8) => Ok(apply_matrix(&DST_VII_8, non_zero_s, x)),
        (TrType::DctVIII, 4) => Ok(apply_matrix(&DCT_VIII_4, non_zero_s, x)),
        (TrType::DctVIII, 8) => Ok(apply_matrix(&DCT_VIII_8, non_zero_s, x)),
        (TrType::DctII, _) => Err(Error::unsupported(
            "h266 transform: DCT-II (trType=0) tables not yet landed",
        )),
        (TrType::DstVII, _) | (TrType::DctVIII, _) => Err(Error::unsupported(format!(
            "h266 transform: DST-VII / DCT-VIII at size {n_tb_s} not yet supported (only 4 / 8)"
        ))),
    }
}

fn apply_matrix<const N: usize>(matrix: &[[i16; N]; N], non_zero_s: usize, x: &[i32]) -> Vec<i32> {
    let mut y = vec![0i32; N];
    for (i, row) in matrix.iter().enumerate() {
        let mut acc: i64 = 0;
        for j in 0..non_zero_s {
            acc += row[j] as i64 * x[j] as i64;
        }
        // Fit into i32: VVC coefficients are ≤ 91 (8 bit signed),
        // non_zero_s ≤ 8 here, x[j] ≤ 2^15 by construction → acc ≤
        // 91 * 8 * 2^15 ≈ 2^25 — well within i32.
        y[i] = acc as i32;
    }
    y
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A DC-only input (`x[0]` non-zero, others zero) through DST-VII
    /// extracts the first column of the matrix, scaled by `x[0]`.
    #[test]
    fn dst_vii_4_dc_input_returns_first_column() {
        let x = [100, 0, 0, 0];
        let y = one_d_transform(TrType::DstVII, 4, 1, &x).unwrap();
        // First column of DST_VII_4: 29, 74, 84, 55.
        assert_eq!(y, vec![29 * 100, 74 * 100, 84 * 100, 55 * 100]);
    }

    /// Same for DCT-VIII 4×4.
    #[test]
    fn dct_viii_4_dc_input_returns_first_column() {
        let x = [50, 0, 0, 0];
        let y = one_d_transform(TrType::DctVIII, 4, 1, &x).unwrap();
        // First column: 84, 74, 55, 29.
        assert_eq!(y, vec![84 * 50, 74 * 50, 55 * 50, 29 * 50]);
    }

    /// Feeding an impulse in coefficient position k returns the k-th
    /// column of the matrix. Loop over all 8 positions of DST-VII 8.
    #[test]
    fn dst_vii_8_impulse_responses_match_columns() {
        for k in 0..8 {
            let mut x = [0i32; 8];
            x[k] = 1;
            let y = one_d_transform(TrType::DstVII, 8, 8, &x).unwrap();
            let expected: Vec<i32> = DST_VII_8.iter().map(|row| row[k] as i32).collect();
            assert_eq!(y, expected, "DST-VII-8 column {k} mismatch");
        }
    }

    /// Linearity: T(x + y) = T(x) + T(y). Verifies accumulation
    /// uses linear sums and not e.g. saturating math.
    #[test]
    fn dst_vii_is_linear() {
        let a = [3, -7, 12, 5];
        let b = [-2, 1, 8, -4];
        let sum: Vec<i32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();
        let ta = one_d_transform(TrType::DstVII, 4, 4, &a).unwrap();
        let tb = one_d_transform(TrType::DstVII, 4, 4, &b).unwrap();
        let tsum = one_d_transform(TrType::DstVII, 4, 4, &sum).unwrap();
        for i in 0..4 {
            assert_eq!(tsum[i], ta[i] + tb[i], "linearity fails at row {i}");
        }
    }

    /// non_zero_s clamps the inner sum: values past `non_zero_s`
    /// are ignored. Gives the decoder the freedom to skip
    /// arithmetic for zeroed high-frequency columns.
    #[test]
    fn non_zero_s_clips_input() {
        let x_full = [10, 20, 30, 40];
        let y_full = one_d_transform(TrType::DstVII, 4, 4, &x_full).unwrap();
        let x_trunc = [10, 20, 30, 40];
        // non_zero_s=2 → summation only over j=0..1.
        let y_trunc = one_d_transform(TrType::DstVII, 4, 2, &x_trunc).unwrap();
        // Manually compute the expected truncated sum.
        let expected: Vec<i32> = (0..4)
            .map(|i| (DST_VII_4[i][0] as i32 * 10) + (DST_VII_4[i][1] as i32 * 20))
            .collect();
        assert_eq!(y_trunc, expected);
        // And it should differ from y_full (unless by chance the
        // higher columns cancelled to zero, which they don't here).
        assert_ne!(y_trunc, y_full);
    }

    /// trType=DCT-II currently returns Unsupported.
    #[test]
    fn dct_ii_is_unsupported_pending_tables() {
        let x = [1, 2, 3, 4];
        let r = one_d_transform(TrType::DctII, 4, 4, &x);
        assert!(r.is_err());
    }

    /// Size-16 and beyond are deferred for the small-kernel drop.
    #[test]
    fn big_sizes_are_unsupported_pending_tables() {
        let x = vec![0i32; 16];
        assert!(one_d_transform(TrType::DstVII, 16, 16, &x).is_err());
        assert!(one_d_transform(TrType::DctVIII, 32, 32, &x).is_err());
    }

    /// Input validation: nonZeroS > nTbS is rejected.
    #[test]
    fn non_zero_s_out_of_range_is_rejected() {
        let x = [1, 2, 3, 4, 5];
        assert!(one_d_transform(TrType::DstVII, 4, 5, &x).is_err());
    }
}
