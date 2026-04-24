//! VVC intra-picture reconstruction (§8.7.5) — per-TU path.
//!
//! Given a predicted sample array and a scaled / inverse-transformed
//! residual array of the same dimensions, compute the reconstructed
//! sample values and store them into a component plane.
//!
//! Formula (§8.7.5.1 eq. 1426):
//!
//! ```text
//!   recSamples[i][j] = Clip1(predSamples[i][j] + resSamples[i][j])
//! ```
//!
//! The clip range is `[0, (1 << BitDepth) - 1]`.
//!
//! A second helper, [`dequantise_block`], applies the spec's uniform
//! scalar dequantisation (eq. 1150 family) for a TB whose coefficients
//! were decoded with dep_quant off. It accepts a row-major
//! `(n_tb_w * n_tb_h)` coefficient slice, an integer QP, and a bit
//! depth, and produces the scaled transform coefficients ready for
//! the inverse-transform pipeline.

use oxideav_core::{Error, Result};

/// Clip to the pixel-value range for a given bit depth.
#[inline]
pub fn clip_pixel(v: i32, bit_depth: u32) -> i16 {
    let max = (1i32 << bit_depth) - 1;
    v.clamp(0, max) as i16
}

/// Uniform scalar dequantisation. Level-scale table from eq. 1150:
///   levelScale[rem6] = { 40, 45, 51, 57, 64, 72 }
/// with `rem6 = Qp % 6`.
/// `scale` = `levelScale[qp % 6] << (qp / 6)`; each coefficient is
/// then `(coeff * scale + offset) >> (shift - transform_bd_offset)`
/// with `shift = BitDepth + Log2(nTbS) - 5`.
///
/// This helper applies the minimal version: `d[i] = (coeff * scale
/// + (1 << (shift - 1))) >> shift`, which matches spec behaviour when
/// TuCResMode / JointCbCr scaling lists are all disabled.
pub fn dequantise_block(
    coeffs: &[i32],
    n_tb_w: usize,
    n_tb_h: usize,
    qp: i32,
    bit_depth: u32,
) -> Result<Vec<i32>> {
    if coeffs.len() != n_tb_w * n_tb_h {
        return Err(Error::invalid(format!(
            "h266 dequant: input length {} != {} x {}",
            coeffs.len(),
            n_tb_w,
            n_tb_h
        )));
    }
    let level_scale = [40, 45, 51, 57, 64, 72];
    let qp = qp.clamp(0, 63);
    let scale = level_scale[(qp % 6) as usize] << (qp / 6);
    let log2_w = n_tb_w.trailing_zeros() as i32;
    let log2_h = n_tb_h.trailing_zeros() as i32;
    // Per eq. 1150 (simplified):
    //   shift1 = BitDepth + ((log2_w + log2_h) / 2) - 5
    let shift = (bit_depth as i32) + (log2_w + log2_h) / 2 - 5;
    let shift = shift.max(1) as u32;
    let offset = 1i32 << (shift - 1);
    let mut out = vec![0i32; coeffs.len()];
    for i in 0..coeffs.len() {
        // Saturating arithmetic so a conformance-violating input
        // doesn't panic.
        let prod = (coeffs[i] as i64) * (scale as i64) + offset as i64;
        out[i] = (prod >> shift) as i32;
    }
    Ok(out)
}

/// Add prediction + residual and clip into a destination plane. The
/// destination is a row-major byte array of `dst_stride` columns; the
/// TB is placed at offset `(x, y)`.
pub fn reconstruct_tb_into(
    dst: &mut [u8],
    dst_stride: usize,
    dst_height: usize,
    x: usize,
    y: usize,
    pred: &[i16],
    residual: &[i32],
    n_tb_w: usize,
    n_tb_h: usize,
    bit_depth: u32,
) -> Result<()> {
    if pred.len() != n_tb_w * n_tb_h || residual.len() != n_tb_w * n_tb_h {
        return Err(Error::invalid(
            "h266 reconstruct: pred / residual size mismatch",
        ));
    }
    if x + n_tb_w > dst_stride || y + n_tb_h > dst_height {
        return Err(Error::invalid(
            "h266 reconstruct: TB does not fit in destination",
        ));
    }
    for row in 0..n_tb_h {
        for col in 0..n_tb_w {
            let p = pred[row * n_tb_w + col] as i32;
            let r = residual[row * n_tb_w + col];
            let v = clip_pixel(p + r, bit_depth);
            let dst_idx = (y + row) * dst_stride + (x + col);
            // 8-bit destination — cast down (we are outputting YUV420P).
            let v8 = if bit_depth > 8 {
                (v >> (bit_depth - 8)) as u8
            } else {
                v as u8
            };
            dst[dst_idx] = v8;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Clip at 8-bit: sub-zero → 0, saturated → 255.
    #[test]
    fn clip_pixel_8bit() {
        assert_eq!(clip_pixel(-5, 8), 0);
        assert_eq!(clip_pixel(256, 8), 255);
        assert_eq!(clip_pixel(123, 8), 123);
    }

    /// Dequantise a single DC coefficient and check the scaling.
    /// At qp = 24 (rem6=0, div6=4) → scale = 40 << 4 = 640.
    /// For N=4 (log2_w=log2_h=2), shift = bit_depth + (2+2)/2 - 5 = 10+2-5 = 7.
    /// offset = 64. d = (coeff * 640 + 64) >> 7.
    /// With coeff = 10 → (6400 + 64) >> 7 = 6464 >> 7 = 50.
    #[test]
    fn dequant_dc_coeff_4x4_qp24() {
        let mut c = vec![0i32; 16];
        c[0] = 10;
        let d = dequantise_block(&c, 4, 4, 24, 10).unwrap();
        assert_eq!(d[0], 50);
        for &v in &d[1..] {
            assert_eq!(v, 0);
        }
    }

    /// reconstruct_tb_into writes predicted sample when residual is zero.
    #[test]
    fn reconstruct_zero_residual_copies_pred() {
        let mut dst = vec![0u8; 16];
        let pred = vec![128i16; 16];
        let res = vec![0i32; 16];
        reconstruct_tb_into(&mut dst, 4, 4, 0, 0, &pred, &res, 4, 4, 8).unwrap();
        assert!(dst.iter().all(|&v| v == 128));
    }

    /// reconstruct clips overflow: prediction 250 + residual 20 → 255.
    #[test]
    fn reconstruct_clips_overflow() {
        let mut dst = vec![0u8; 4];
        let pred = vec![250i16; 4];
        let res = vec![20i32; 4];
        reconstruct_tb_into(&mut dst, 2, 2, 0, 0, &pred, &res, 2, 2, 8).unwrap();
        for &v in &dst {
            assert_eq!(v, 255);
        }
    }

    /// reconstruct clips underflow: prediction 5 + residual -20 → 0.
    #[test]
    fn reconstruct_clips_underflow() {
        let mut dst = vec![99u8; 4];
        let pred = vec![5i16; 4];
        let res = vec![-20i32; 4];
        reconstruct_tb_into(&mut dst, 2, 2, 0, 0, &pred, &res, 2, 2, 8).unwrap();
        for &v in &dst {
            assert_eq!(v, 0);
        }
    }

    /// 10-bit inputs are narrowed to 8-bit output via right-shift.
    #[test]
    fn reconstruct_10bit_narrows_to_8() {
        let mut dst = vec![0u8; 4];
        // predicted = 1020 (just under max 1023), residual 0.
        let pred = vec![1020i16; 4];
        let res = vec![0i32; 4];
        reconstruct_tb_into(&mut dst, 2, 2, 0, 0, &pred, &res, 2, 2, 10).unwrap();
        // 1020 >> 2 = 255.
        for &v in &dst {
            assert_eq!(v, 255);
        }
    }

    /// TB-out-of-bounds is surfaced as an error.
    #[test]
    fn reconstruct_out_of_bounds_errors() {
        let mut dst = vec![0u8; 16];
        let pred = vec![0i16; 16];
        let res = vec![0i32; 16];
        assert!(reconstruct_tb_into(&mut dst, 4, 4, 2, 2, &pred, &res, 4, 4, 8).is_err());
    }

    /// Mini end-to-end integration: DC-predicted 4x4 TB + a single
    /// dequantised DC coefficient + inverse DCT-II + reconstruct.
    /// Sanity check that all pieces compose without crashing and
    /// produce values in the valid 8-bit range.
    #[test]
    fn mini_end_to_end_dc_impulse() {
        use crate::intra::{predict_dc, IntraRefs};
        use crate::transform::{inverse_transform_2d, TrType};

        // 4x4 TB with constant-grey (128) neighbours → DC prediction 128.
        let above = vec![128i16; 5];
        let left = vec![128i16; 5];
        let refs = IntraRefs {
            above: &above,
            left: &left,
            top_left: 128,
        };
        let pred = predict_dc(4, 4, &refs).unwrap();
        assert_eq!(pred, vec![128; 16]);

        // Single DC coefficient → dequantise.
        let mut c = vec![0i32; 16];
        c[0] = 100;
        let d = dequantise_block(&c, 4, 4, 30, 8).unwrap();

        // Inverse 2D DCT-II on the dequantised coefficients.
        let res =
            inverse_transform_2d(4, 4, 1, 1, TrType::DctII, TrType::DctII, &d, 8, 15).unwrap();

        // Reconstruct into a 4x4 plane.
        let mut plane = vec![0u8; 16];
        reconstruct_tb_into(&mut plane, 4, 4, 0, 0, &pred, &res, 4, 4, 8).unwrap();

        // Every sample must be valid 8-bit; the DC-only input + constant
        // prediction guarantees a spatially uniform result.
        let first = plane[0];
        for &v in &plane {
            assert_eq!(v, first);
        }
    }
}
