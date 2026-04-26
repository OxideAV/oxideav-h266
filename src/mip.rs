//! Matrix-based Intra Prediction (MIP) — §8.4.5.2.2 of ITU-T H.266 (V4).
//!
//! MIP is a luma-only intra mode that replaces the angular / planar / DC
//! prediction with a small *matrix multiplication* against a downsampled
//! reference-sample vector. It applies to luma TBs of size 4×4 up to
//! 64×64. The pipeline (per §8.4.5.2.2) is:
//!
//! 1. Build the unfiltered reference rows / columns `refT[0..nTbW-1]`,
//!    `refL[0..nTbH-1]`.
//! 2. Downsample to `redT[0..boundarySize-1]`, `redL[0..boundarySize-1]`
//!    via §8.4.5.2.3 (rectangular box filter, not a tap filter).
//! 3. Concatenate into a `pTemp[0..2*boundarySize-1]` vector, swapping
//!    halves when `intra_mip_transposed_flag == 1`.
//! 4. Build the input vector `p[0..inSize-1]` from `pTemp[]`:
//!    - If `mipSizeId == 2`: `p[i] = pTemp[i+1] - pTemp[0]`,
//!      so `inSize = 2*boundarySize - 1 = 7`.
//!    - Otherwise: `p[0] = (1<<(BitDepth-1)) - pTemp[0]`, and
//!      `p[i] = pTemp[i] - pTemp[0]` for `i = 1..inSize-1`, so
//!      `inSize = 2*boundarySize ∈ { 4, 8 }`.
//! 5. Compute the dense `predMip[predSize][predSize]` block via
//!    eq. (270): `predMip[x][y] = (((Σᵢ mWeight[i][y*predSize+x] *
//!    p[i]) + oW) >> 6) + pTemp[0]` with `oW = 32 − 32 * Σᵢ p[i]`.
//! 6. Clip to the pixel range with `Clip1`.
//! 7. If `isTransposed`, transpose `predMip` (eq. 272 / 273).
//! 8. If `nTbW > predSize` or `nTbH > predSize`, run the §8.4.5.2.5
//!    upsampling process to fill the full `nTbW × nTbH` block.
//!
//! The weight matrices `mWeight` come from §8.4.5.2.4 (Tables 276 .. 305)
//! and live in [`crate::mip_tables`]. They are indexed
//! `MIP_W_SZ{0,1,2}[mode][j][i]` where `j = y*predSize + x` and `i ∈
//! 0..inSize-1`, matching the spec's printed `{ {row 0}, ... }` ordering
//! exactly.
//!
//! Out of scope for this module: parsing `intra_mip_flag` /
//! `intra_mip_transposed_flag` / `intra_mip_mode` (handled in
//! [`crate::leaf_cu`]) and the cIdx-specific neighbour-availability
//! marking (the caller passes already-substituted `refT` / `refL`).

use crate::mip_tables::{MIP_W_SZ0, MIP_W_SZ1, MIP_W_SZ2};
use oxideav_core::error::{Error, Result};

/// Specification of `boundarySize` and `predSize` per `mipSizeId`
/// (§8.4.5.2.2 Table 22). Returns `(boundarySize, predSize)`.
#[inline]
pub fn mip_size_params(mip_size_id: u32) -> (u32, u32) {
    match mip_size_id {
        0 => (2, 4),
        1 => (4, 4),
        // mipSizeId == 2 covers all CB shapes with both nTbW > 4 and
        // nTbH > 4 (and not 8×8). predSize is 8 in this case.
        _ => (4, 8),
    }
}

/// Derive `mipSizeId` from the transform-block dimensions per §8.4.5.2.2.
///
/// * `mipSizeId == 0` when both `nTbW` and `nTbH` are equal to 4.
/// * `mipSizeId == 1` when either `nTbW` or `nTbH` is equal to 4, OR
///   both are equal to 8.
/// * `mipSizeId == 2` otherwise.
#[inline]
pub fn mip_size_id(n_tb_w: u32, n_tb_h: u32) -> u32 {
    if n_tb_w == 4 && n_tb_h == 4 {
        0
    } else if n_tb_w == 4 || n_tb_h == 4 || (n_tb_w == 8 && n_tb_h == 8) {
        1
    } else {
        2
    }
}

/// The number of MIP modes for each `mipSizeId` (Tables 276..305).
/// Used to validate the parsed `intra_mip_mode`.
#[inline]
pub fn num_mip_modes(mip_size_id: u32) -> u32 {
    match mip_size_id {
        0 => 16,
        1 => 8,
        _ => 6,
    }
}

/// Look up the weight matrix entry `mWeight[i][j]` for a given
/// `(mipSizeId, modeId)`. The output is widened to `i32` so that the
/// matrix-vector product in eq. (270) can be done in 32-bit arithmetic
/// without per-cell casts.
///
/// Indexing convention: the spec's printed form `{ {row 0 with inSize
/// entries}, {row 1 with inSize entries}, ... }` is stored as
/// `[mode][j][i]` with `j ∈ 0..predSize*predSize-1` and
/// `i ∈ 0..inSize-1`.
#[inline]
fn m_weight(mip_size_id: u32, mode_id: u32, j: usize, i: usize) -> i32 {
    match mip_size_id {
        0 => MIP_W_SZ0[mode_id as usize][j][i] as i32,
        1 => MIP_W_SZ1[mode_id as usize][j][i] as i32,
        _ => MIP_W_SZ2[mode_id as usize][j][i] as i32,
    }
}

/// §8.4.5.2.3 — MIP boundary sample downsampling.
///
/// `red_s[x] = ( Σᵢ refS[x*bDwn + i] + (1 << (Log2(bDwn)-1)) ) >> Log2(bDwn)`
/// for `i = 0..bDwn-1`, with `bDwn = nTbS / boundarySize`. When
/// `boundarySize == nTbS`, the input is copied through unchanged.
///
/// `ref_s` has length `nTbS`; the returned `Vec` has length
/// `boundarySize`.
pub fn downsample_boundary(ref_s: &[i16], boundary_size: usize) -> Vec<i32> {
    let n_tb_s = ref_s.len();
    if boundary_size == 0 {
        return Vec::new();
    }
    if boundary_size == n_tb_s {
        return ref_s.iter().map(|&v| v as i32).collect();
    }
    // Spec only ever asks us to downsample by a power-of-two factor — the
    // shift below assumes `bDwn` ∈ { 2, 4, 8, 16 }.
    let b_dwn = n_tb_s / boundary_size;
    debug_assert!(b_dwn.is_power_of_two() && b_dwn >= 2);
    let log2_b = b_dwn.trailing_zeros() as i32;
    let round = 1i32 << (log2_b - 1);
    let mut out = vec![0i32; boundary_size];
    for x in 0..boundary_size {
        let mut acc = 0i32;
        for i in 0..b_dwn {
            acc += ref_s[x * b_dwn + i] as i32;
        }
        out[x] = (acc + round) >> log2_b;
    }
    out
}

/// §8.4.5.2.5 — MIP prediction upsampling. Produces the full
/// `n_tb_w × n_tb_h` predicted-sample block from the dense `predSize ×
/// predSize` matrix and the original (pre-downsampling) reference
/// rows / columns. Output is row-major (`y * n_tb_w + x`).
///
/// Returns the input untouched when no upsampling is needed (`upHor ==
/// upVer == 1`).
pub fn upsample_prediction(
    pred_mip: &[i32],
    pred_size: usize,
    n_tb_w: usize,
    n_tb_h: usize,
    ref_t: &[i16],
    ref_l: &[i16],
) -> Vec<i32> {
    debug_assert_eq!(pred_mip.len(), pred_size * pred_size);
    let up_hor = n_tb_w / pred_size;
    let up_ver = n_tb_h / pred_size;
    if up_hor == 1 && up_ver == 1 {
        return pred_mip.to_vec();
    }

    // Allocate a (n_tb_w + 1) × (n_tb_h + 1) staging buffer that
    // co-locates the row/column of reference samples with the dense
    // prediction (the spec models them as `predSamples[-1][y]` and
    // `predSamples[x][-1]`). We stash the references at the topmost row
    // and leftmost column and refer to absolute coordinates +1 below.
    let stride = n_tb_w + 1;
    let height = n_tb_h + 1;
    let mut pred = vec![0i32; stride * height];
    let at = |x: usize, y: usize| (y + 1) * stride + (x + 1);
    let at_top = |x: usize| 0 * stride + (x + 1);
    let at_left = |y: usize| (y + 1) * stride + 0;

    // Sparse seeding: predSamples[(x+1)*upHor-1][(y+1)*upVer-1] = predMip[x][y]
    for y in 0..pred_size {
        for x in 0..pred_size {
            let xx = (x + 1) * up_hor - 1;
            let yy = (y + 1) * up_ver - 1;
            pred[at(xx, yy)] = pred_mip[y * pred_size + x];
        }
    }
    // Reference rows (refT at y = -1, refL at x = -1).
    for x in 0..n_tb_w {
        pred[at_top(x)] = ref_t[x] as i32;
    }
    for y in 0..n_tb_h {
        pred[at_left(y)] = ref_l[y] as i32;
    }

    // 1. Horizontal upsampling at each sparse row y_hor = (n+1)*upVer - 1
    //    for n = 0..predSize-1, between columns (m*upHor - 1, (m+1)*upHor - 1).
    if up_hor > 1 {
        for n in 1..=pred_size {
            let y_hor_abs = n * up_ver - 1;
            // y = -1 maps to top reference row. We always include the
            // sparse rows themselves; the top reference is automatically
            // available because we seeded `at_top`.
            for m in 0..pred_size {
                // Sparse columns m*upHor - 1 (left, may be -1) and
                // (m+1)*upHor - 1 (right). The "left -1" case is the
                // refL column at x = -1 — handled by `at_left`.
                let x_left_signed = (m as isize) * (up_hor as isize) - 1;
                let x_right = (m + 1) * up_hor - 1;
                let left_val = if x_left_signed < 0 {
                    pred[at_left(y_hor_abs)]
                } else {
                    pred[at(x_left_signed as usize, y_hor_abs)]
                };
                let right_val = pred[at(x_right, y_hor_abs)];
                for d_x in 1..up_hor {
                    let sum = ((up_hor - d_x) as i32) * left_val + (d_x as i32) * right_val;
                    let v = (sum + (up_hor as i32 / 2)) / (up_hor as i32);
                    let xx = if x_left_signed < 0 {
                        d_x - 1
                    } else {
                        x_left_signed as usize + d_x
                    };
                    pred[at(xx, y_hor_abs)] = v;
                }
            }
        }
    }

    // 2. Vertical upsampling at every column m = 0..n_tb_w-1, between
    //    sparse rows (n*upVer - 1) and ((n+1)*upVer - 1).
    if up_ver > 1 {
        for m in 0..n_tb_w {
            for n in 0..pred_size {
                let y_top_signed = (n as isize) * (up_ver as isize) - 1;
                let y_bottom = (n + 1) * up_ver - 1;
                let top_val = if y_top_signed < 0 {
                    pred[at_top(m)]
                } else {
                    pred[at(m, y_top_signed as usize)]
                };
                let bottom_val = pred[at(m, y_bottom)];
                for d_y in 1..up_ver {
                    let sum = ((up_ver - d_y) as i32) * top_val + (d_y as i32) * bottom_val;
                    let v = (sum + (up_ver as i32 / 2)) / (up_ver as i32);
                    let yy = if y_top_signed < 0 {
                        d_y - 1
                    } else {
                        y_top_signed as usize + d_y
                    };
                    pred[at(m, yy)] = v;
                }
            }
        }
    }

    // Pull out the n_tb_h × n_tb_w body (drop the reference frame).
    let mut out = vec![0i32; n_tb_w * n_tb_h];
    for y in 0..n_tb_h {
        for x in 0..n_tb_w {
            out[y * n_tb_w + x] = pred[at(x, y)];
        }
    }
    out
}

/// Generate a `n_tb_w × n_tb_h` MIP prediction block per §8.4.5.2.2.
/// The output is row-major luma samples clipped to `[0, (1 <<
/// bit_depth) - 1]`.
///
/// Inputs:
/// * `n_tb_w` / `n_tb_h` — transform-block size in luma samples
///   (4 ≤ side ≤ 64; both ≥ 4).
/// * `mode_id` — `intra_mip_mode` from the leaf-CU parser. Must be in
///   `[0, num_mip_modes(mipSizeId))`.
/// * `is_transposed` — `intra_mip_transposed_flag`.
/// * `ref_t` — `nTbW` reference samples from the row immediately above
///   the TB. The caller has already done availability marking +
///   substitution (§8.4.5.2.8 / §8.4.5.2.9).
/// * `ref_l` — `nTbH` reference samples from the column immediately
///   left of the TB.
/// * `bit_depth` — luma bit depth.
pub fn predict_mip(
    n_tb_w: usize,
    n_tb_h: usize,
    mode_id: u32,
    is_transposed: bool,
    ref_t: &[i16],
    ref_l: &[i16],
    bit_depth: u32,
) -> Result<Vec<i16>> {
    if !(4..=64).contains(&n_tb_w) || !(4..=64).contains(&n_tb_h) {
        return Err(Error::invalid(format!(
            "h266 MIP: TB size {}x{} out of range [4, 64]",
            n_tb_w, n_tb_h
        )));
    }
    if !n_tb_w.is_power_of_two() || !n_tb_h.is_power_of_two() {
        return Err(Error::invalid(format!(
            "h266 MIP: TB size {}x{} not power-of-two",
            n_tb_w, n_tb_h
        )));
    }
    if ref_t.len() != n_tb_w || ref_l.len() != n_tb_h {
        return Err(Error::invalid(format!(
            "h266 MIP: ref length mismatch refT={} refL={} expected {}/{}",
            ref_t.len(),
            ref_l.len(),
            n_tb_w,
            n_tb_h
        )));
    }
    let mip_size_id = mip_size_id(n_tb_w as u32, n_tb_h as u32);
    let n_modes = num_mip_modes(mip_size_id);
    if mode_id >= n_modes {
        return Err(Error::invalid(format!(
            "h266 MIP: mode {} >= max {} for mipSizeId={}",
            mode_id, n_modes, mip_size_id
        )));
    }
    let (boundary_size_u, pred_size_u) = mip_size_params(mip_size_id);
    let boundary_size = boundary_size_u as usize;
    let pred_size = pred_size_u as usize;
    // inSize = 2 * boundarySize - (mipSizeId == 2 ? 1 : 0).
    let in_size = if mip_size_id == 2 {
        2 * boundary_size - 1
    } else {
        2 * boundary_size
    };

    // 1. Boundary downsampling.
    let red_t = downsample_boundary(ref_t, boundary_size);
    let red_l = downsample_boundary(ref_l, boundary_size);

    // 2. Build pTemp[]. When isTransposed, the left half goes to the
    // top half of pTemp (eq. derived from the spec's bullet list above
    // eq. 267).
    let mut p_temp = vec![0i32; 2 * boundary_size];
    if is_transposed {
        for x in 0..boundary_size {
            p_temp[x] = red_l[x];
            p_temp[x + boundary_size] = red_t[x];
        }
    } else {
        for x in 0..boundary_size {
            p_temp[x] = red_t[x];
            p_temp[x + boundary_size] = red_l[x];
        }
    }

    // 3. p[] from pTemp[].
    let mut p = vec![0i32; in_size];
    if mip_size_id == 2 {
        // p[i] = pTemp[i+1] - pTemp[0]
        for i in 0..in_size {
            p[i] = p_temp[i + 1] - p_temp[0];
        }
    } else {
        // p[0] = (1 << (BitDepth-1)) - pTemp[0]
        // p[i] = pTemp[i] - pTemp[0] for i = 1..inSize-1
        let mid = 1i32 << (bit_depth - 1);
        p[0] = mid - p_temp[0];
        for i in 1..in_size {
            p[i] = p_temp[i] - p_temp[0];
        }
    }

    // 4. Matrix multiply + clip. Use i32 arithmetic — each weight is
    // ≤ 127, |p[i]| < 2^bit_depth, and inSize ≤ 8, so accumulation fits
    // comfortably in i32.
    let max_val = (1i32 << bit_depth) - 1;
    let sum_p: i32 = p.iter().sum();
    let o_w = 32 - 32 * sum_p;
    let mut pred_mip = vec![0i32; pred_size * pred_size];
    for y in 0..pred_size {
        for x in 0..pred_size {
            let j = y * pred_size + x;
            let mut acc: i32 = 0;
            for i in 0..in_size {
                acc += m_weight(mip_size_id, mode_id, j, i) * p[i];
            }
            let v = ((acc + o_w) >> 6) + p_temp[0];
            pred_mip[j] = v.clamp(0, max_val);
        }
    }

    // 5. Transpose if needed (eq. 272 / 273).
    if is_transposed {
        let mut t = vec![0i32; pred_size * pred_size];
        for y in 0..pred_size {
            for x in 0..pred_size {
                t[x * pred_size + y] = pred_mip[y * pred_size + x];
            }
        }
        pred_mip = t;
    }

    // 6. Upsample (or pass through). The output is then clipped a
    // second time only if the upsampling produced out-of-range values
    // — by construction the linear interpolation between in-range
    // endpoints stays in-range, so no extra clip is needed.
    let pred = if n_tb_w > pred_size || n_tb_h > pred_size {
        upsample_prediction(&pred_mip, pred_size, n_tb_w, n_tb_h, ref_t, ref_l)
    } else {
        pred_mip
    };

    // 7. Cast back to i16.
    Ok(pred
        .into_iter()
        .map(|v| v.clamp(0, max_val) as i16)
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `mip_size_id` Table-22 sanity.
    #[test]
    fn mip_size_id_table_22() {
        assert_eq!(mip_size_id(4, 4), 0);
        assert_eq!(mip_size_id(8, 8), 1);
        assert_eq!(mip_size_id(4, 16), 1);
        assert_eq!(mip_size_id(16, 4), 1);
        assert_eq!(mip_size_id(16, 16), 2);
        assert_eq!(mip_size_id(8, 16), 2);
        assert_eq!(mip_size_id(64, 64), 2);
    }

    /// Mode-count bounds: 16 / 8 / 6 modes per sizeId.
    #[test]
    fn num_mip_modes_per_size_id() {
        assert_eq!(num_mip_modes(0), 16);
        assert_eq!(num_mip_modes(1), 8);
        assert_eq!(num_mip_modes(2), 6);
    }

    /// Boundary downsampling identity when `boundary_size == nTbS`.
    #[test]
    fn boundary_downsample_pass_through() {
        let r = vec![10i16, 20, 30, 40];
        let d = downsample_boundary(&r, 4);
        assert_eq!(d, vec![10, 20, 30, 40]);
    }

    /// Boundary downsampling box-filter from 8 → 4 (each output is the
    /// rounded mean of two neighbouring inputs).
    #[test]
    fn boundary_downsample_8_to_4_round_to_nearest() {
        // Inputs (10, 20) → mean 15.
        let r = vec![10i16, 20, 0, 0, 100, 110, 200, 210];
        let d = downsample_boundary(&r, 4);
        // (10+20+1)>>1 = 15
        // (0+0+1)>>1 = 0
        // (100+110+1)>>1 = 105
        // (200+210+1)>>1 = 205
        assert_eq!(d, vec![15, 0, 105, 205]);
    }

    /// MIP weights all live in `[0, 127]` (sanity-check the table
    /// import — guard against spec-extraction off-by-ones).
    #[test]
    fn weight_table_value_range() {
        for mode in 0..16usize {
            for j in 0..16 {
                for i in 0..4 {
                    assert!(MIP_W_SZ0[mode][j][i] <= 127);
                }
            }
        }
        for mode in 0..8usize {
            for j in 0..16 {
                for i in 0..8 {
                    assert!(MIP_W_SZ1[mode][j][i] <= 127);
                }
            }
        }
        for mode in 0..6usize {
            for j in 0..64 {
                for i in 0..7 {
                    assert!(MIP_W_SZ2[mode][j][i] <= 127);
                }
            }
        }
    }

    /// Spot-check the spec-extracted `MIP_W_SZ0[0]` first row against
    /// the printed values in §8.4.5.2.4 eq. (276):
    /// `{ 32, 30, 90, 28}, ...`
    #[test]
    fn mip_w_sz0_mode0_matches_spec_row0() {
        assert_eq!(MIP_W_SZ0[0][0], [32, 30, 90, 28]);
        assert_eq!(MIP_W_SZ0[0][1], [32, 32, 72, 28]);
    }

    /// Spot-check `MIP_W_SZ2[0]` first row against §8.4.5.2.4 eq. (300):
    /// `{ 42, 37, 33, 27, 44, 33, 35}, ...`
    #[test]
    fn mip_w_sz2_mode0_matches_spec_row0() {
        assert_eq!(MIP_W_SZ2[0][0], [42, 37, 33, 27, 44, 33, 35]);
    }

    /// 4×4 MIP with a constant reference. The downsampling preserves
    /// the constant, p[i] = 0 for i ≥ 1 and p[0] = (1<<(bd-1)) - pTemp[0]
    /// = 0 when pTemp[0] == mid. Then `oW = 32 - 32*p[0] = 32`,
    /// `acc = mWeight[0][j] * 0 = 0`, so predMip[x][y] = (32 >> 6) +
    /// pTemp[0] = 0 + pTemp[0] = pTemp[0] when pTemp[0] = mid. Result
    /// should be all `mid` (no NaN, no overflow).
    #[test]
    fn predict_mip_4x4_constant_reference_returns_mid() {
        let bit_depth = 8;
        let mid = 1i16 << (bit_depth - 1);
        let ref_t = vec![mid; 4];
        let ref_l = vec![mid; 4];
        for mode in 0..16 {
            let pred = predict_mip(4, 4, mode, false, &ref_t, &ref_l, bit_depth).unwrap();
            assert_eq!(pred.len(), 16);
            for &v in &pred {
                assert_eq!(v, mid, "mode {} produced non-mid sample {}", mode, v);
            }
        }
    }

    /// 8×8 MIP with a constant reference (mipSizeId == 1). Same logic
    /// as the 4×4 case: p[0] = mid - mid = 0, p[i>=1] = 0, so the
    /// matrix product collapses to `pTemp[0] + (32 >> 6)` = mid.
    #[test]
    fn predict_mip_8x8_constant_reference_returns_mid() {
        let bit_depth = 8;
        let mid = 1i16 << (bit_depth - 1);
        let ref_t = vec![mid; 8];
        let ref_l = vec![mid; 8];
        for mode in 0..num_mip_modes(1) {
            let pred = predict_mip(8, 8, mode, false, &ref_t, &ref_l, bit_depth).unwrap();
            assert_eq!(pred.len(), 64);
            for &v in &pred {
                assert_eq!(v, mid);
            }
        }
    }

    /// 16×16 MIP with a constant reference (mipSizeId == 2, requires
    /// upsampling). For sz2 the p[] formula is p[i] = pTemp[i+1] -
    /// pTemp[0] = 0 for all i; oW = 32; acc = 0; predMip[x][y] = pTemp[0]
    /// = mid. Upsampling between mid-valued cells stays at mid.
    #[test]
    fn predict_mip_16x16_constant_reference_returns_mid() {
        let bit_depth = 10;
        let mid = 1i16 << (bit_depth - 1);
        let ref_t = vec![mid; 16];
        let ref_l = vec![mid; 16];
        for mode in 0..num_mip_modes(2) {
            let pred = predict_mip(16, 16, mode, false, &ref_t, &ref_l, bit_depth).unwrap();
            assert_eq!(pred.len(), 256);
            for &v in &pred {
                assert_eq!(v, mid);
            }
        }
    }

    /// Transposed flag swaps the boundary halves: with refT == 0 and
    /// refL == max, the transposed prediction is the spatial transpose
    /// of the un-transposed prediction (modulo clipping). Sanity check
    /// that the flag actually changes the output.
    #[test]
    fn predict_mip_transposed_changes_result() {
        let bit_depth = 8;
        let ref_t = vec![10i16; 4];
        let ref_l = vec![200i16; 4];
        let pred_a = predict_mip(4, 4, 0, false, &ref_t, &ref_l, bit_depth).unwrap();
        let pred_b = predict_mip(4, 4, 0, true, &ref_t, &ref_l, bit_depth).unwrap();
        assert_ne!(pred_a, pred_b);
    }

    /// All output samples are clipped to `[0, (1 << bd) - 1]` regardless
    /// of input distribution.
    #[test]
    fn predict_mip_output_in_range() {
        let bit_depth = 10;
        let max = (1i16 << bit_depth) - 1;
        let ref_t = vec![max; 16];
        let ref_l = vec![0i16; 16];
        for mode in 0..num_mip_modes(2) {
            let pred = predict_mip(16, 16, mode, false, &ref_t, &ref_l, bit_depth).unwrap();
            for &v in &pred {
                assert!((0..=max).contains(&v));
            }
        }
    }

    /// Bad TB sizes get rejected.
    #[test]
    fn predict_mip_rejects_bad_size() {
        let r = vec![128i16; 4];
        assert!(predict_mip(2, 2, 0, false, &r, &r, 8).is_err());
        let r3 = vec![128i16; 3];
        assert!(predict_mip(3, 3, 0, false, &r3, &r3, 8).is_err());
    }

    /// Mode index out of range is rejected.
    #[test]
    fn predict_mip_rejects_oversized_mode() {
        let r = vec![128i16; 4];
        // sizeId 0 has only 16 modes (0..15).
        assert!(predict_mip(4, 4, 16, false, &r, &r, 8).is_err());
        // sizeId 1 has only 8.
        let r8 = vec![128i16; 8];
        assert!(predict_mip(8, 8, 8, false, &r8, &r8, 8).is_err());
    }

    /// Asymmetric 16x8 MIP (mipSizeId == 2 → predSize == 8, requires
    /// horizontal upsampling but not vertical: upHor = 2, upVer = 1).
    /// Constant-reference invariant still holds: every output pixel is
    /// `mid` because `p[i] == 0` collapses both the matrix product and
    /// the upsampling to constant `pTemp[0] == mid`.
    #[test]
    fn predict_mip_16x8_constant_reference_returns_mid() {
        let bit_depth = 8;
        let mid = 1i16 << (bit_depth - 1);
        let ref_t = vec![mid; 16];
        let ref_l = vec![mid; 8];
        for mode in 0..num_mip_modes(2) {
            let pred = predict_mip(16, 8, mode, false, &ref_t, &ref_l, bit_depth).unwrap();
            assert_eq!(pred.len(), 128);
            for &v in &pred {
                assert_eq!(v, mid);
            }
        }
    }

    /// Asymmetric 8x16 MIP (vertical-only upsampling: upHor = 1,
    /// upVer = 2) — the perpendicular case to the previous test.
    #[test]
    fn predict_mip_8x16_constant_reference_returns_mid() {
        let bit_depth = 8;
        let mid = 1i16 << (bit_depth - 1);
        let ref_t = vec![mid; 8];
        let ref_l = vec![mid; 16];
        for mode in 0..num_mip_modes(2) {
            let pred = predict_mip(8, 16, mode, false, &ref_t, &ref_l, bit_depth).unwrap();
            assert_eq!(pred.len(), 128);
            for &v in &pred {
                assert_eq!(v, mid);
            }
        }
    }

    /// 64x64 — biggest legal MIP block (predSize=8, upHor=upVer=8).
    /// Constant-reference invariant still holds across the deepest
    /// upsampling factor.
    #[test]
    fn predict_mip_64x64_constant_reference_returns_mid() {
        let bit_depth = 10;
        let mid = 1i16 << (bit_depth - 1);
        let ref_t = vec![mid; 64];
        let ref_l = vec![mid; 64];
        for mode in 0..num_mip_modes(2) {
            let pred = predict_mip(64, 64, mode, false, &ref_t, &ref_l, bit_depth).unwrap();
            assert_eq!(pred.len(), 64 * 64);
            for &v in &pred {
                assert_eq!(v, mid);
            }
        }
    }

    /// Direct check of the upsampling step: a 4×4 dense block embedded
    /// in an 8×8 output uses `upHor = upVer = 2`. With
    /// `pred_mip == [const; 16]` and matching reference samples, all
    /// 64 output samples should equal that constant.
    #[test]
    fn upsample_constant_block_4_to_8() {
        let pred_mip = vec![100i32; 16];
        let ref_t = vec![100i16; 8];
        let ref_l = vec![100i16; 8];
        let out = upsample_prediction(&pred_mip, 4, 8, 8, &ref_t, &ref_l);
        assert_eq!(out.len(), 64);
        for v in out {
            assert_eq!(v, 100);
        }
    }
}
