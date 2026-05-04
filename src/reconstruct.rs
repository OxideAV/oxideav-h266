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
//!
//! In addition, this module hosts the picture-plane buffer
//! ([`PicturePlane`]) and the per-CU reference-sample fetcher
//! ([`fetch_intra_refs`]) used by the [`crate::ctu::CtuWalker`]
//! reconstruction pipeline. References are gathered from the
//! already-reconstructed plane (top + left edges of the current TB);
//! when a side is unavailable the spec's mid-grey substitution
//! (`1 << (BitDepth - 1)`, §8.4.5.2.8) is applied.

use oxideav_core::{Error, Result};

/// Component planes stored as 8-bit samples in row-major layout. The
/// `stride` field allows the caller to over-allocate columns (handy
/// for SIMD-friendly layouts later); for the basic decode path
/// `stride == width`. Picture-edge clipping below the spec's
/// `pic_width_luma` / `pic_height_luma` is the caller's responsibility.
#[derive(Clone, Debug)]
pub struct PicturePlane {
    /// Row-major sample storage, length = `stride * height`.
    pub samples: Vec<u8>,
    /// Number of bytes per row (>= width).
    pub stride: usize,
    /// Visible width in samples.
    pub width: usize,
    /// Visible height in samples.
    pub height: usize,
}

impl PicturePlane {
    /// Allocate a plane filled with the supplied seed sample value.
    /// Useful for initialising a fresh frame to mid-grey.
    pub fn filled(width: usize, height: usize, seed: u8) -> Self {
        Self {
            samples: vec![seed; width * height],
            stride: width,
            width,
            height,
        }
    }

    /// Read the sample at `(x, y)`. Returns `None` when out of bounds.
    pub fn get(&self, x: usize, y: usize) -> Option<u8> {
        if x >= self.width || y >= self.height {
            return None;
        }
        Some(self.samples[y * self.stride + x])
    }
}

/// Frame-level reconstruction buffer for 4:2:0 YUV. Chroma planes live
/// alongside luma at half the dimensions in each direction.
#[derive(Clone, Debug)]
pub struct PictureBuffer {
    pub luma: PicturePlane,
    pub cb: PicturePlane,
    pub cr: PicturePlane,
}

impl PictureBuffer {
    /// Allocate a 4:2:0 frame. Chroma planes are seeded to neutral
    /// (128) and luma to mid-grey (also 128 by default).
    pub fn yuv420_filled(luma_w: usize, luma_h: usize, seed: u8) -> Self {
        Self {
            luma: PicturePlane::filled(luma_w, luma_h, seed),
            cb: PicturePlane::filled(luma_w / 2, luma_h / 2, 128),
            cr: PicturePlane::filled(luma_w / 2, luma_h / 2, 128),
        }
    }
}

/// HBD twin of [`PicturePlane`] — Main10 / Main12 reconstruction needs
/// `u16` sample storage so MC and reconstruction can read / write the
/// full bit-depth value (the legacy 8-bit plane truncates Main10 by 2
/// bits). Layout mirrors [`PicturePlane`] one-to-one (row-major, an
/// over-allocated `stride >= width` allowed). Range is `[0,
/// (1 << bit_depth) - 1]`; `bit_depth` is carried at the buffer level
/// to make the clip range explicit and to let MC lift integer-pel
/// samples by the spec's `<< (14 - BitDepth)`.
#[derive(Clone, Debug)]
pub struct PicturePlane16 {
    /// Row-major sample storage, length = `stride * height`.
    pub samples: Vec<u16>,
    /// Number of samples per row (>= width).
    pub stride: usize,
    /// Visible width in samples.
    pub width: usize,
    /// Visible height in samples.
    pub height: usize,
    /// Bit depth of the stored samples (8..=16). `samples[i]` is
    /// guaranteed to lie in `[0, (1 << bit_depth) - 1]` for any
    /// value written through this module's helpers.
    pub bit_depth: u32,
}

impl PicturePlane16 {
    /// Allocate an HBD plane of `width × height` samples seeded to
    /// `seed`. Panics if `bit_depth` is outside `8..=16` or if the
    /// seed exceeds the bit-depth range.
    pub fn filled(width: usize, height: usize, seed: u16, bit_depth: u32) -> Self {
        assert!(
            (8..=16).contains(&bit_depth),
            "PicturePlane16: bit_depth {bit_depth} must be in 8..=16",
        );
        let max = (1u32 << bit_depth) - 1;
        assert!(
            seed as u32 <= max,
            "PicturePlane16: seed {seed} exceeds bit_depth-{bit_depth} max {max}",
        );
        Self {
            samples: vec![seed; width * height],
            stride: width,
            width,
            height,
            bit_depth,
        }
    }

    /// Read the sample at `(x, y)`. Returns `None` when out of bounds.
    pub fn get(&self, x: usize, y: usize) -> Option<u16> {
        if x >= self.width || y >= self.height {
            return None;
        }
        Some(self.samples[y * self.stride + x])
    }

    /// Write `v` at `(x, y)`. Returns `Err` when out of bounds or
    /// when `v` exceeds the bit-depth range.
    pub fn set(&mut self, x: usize, y: usize, v: u16) -> Result<()> {
        if x >= self.width || y >= self.height {
            return Err(Error::invalid(format!(
                "PicturePlane16: ({x},{y}) out of bounds {}x{}",
                self.width, self.height
            )));
        }
        let max = (1u32 << self.bit_depth) - 1;
        if v as u32 > max {
            return Err(Error::invalid(format!(
                "PicturePlane16: value {v} exceeds bit_depth-{} max {max}",
                self.bit_depth,
            )));
        }
        self.samples[y * self.stride + x] = v;
        Ok(())
    }

    /// Lossy projection to an 8-bit [`PicturePlane`] via right-shift by
    /// `bit_depth - 8` (the canonical narrowing used by the legacy
    /// 8-bit pipeline). When `bit_depth == 8` this is a copy.
    pub fn to_picture_plane_u8(&self) -> PicturePlane {
        let shift = self.bit_depth - 8;
        let mut out = vec![0u8; self.width * self.height];
        for y in 0..self.height {
            for x in 0..self.width {
                out[y * self.width + x] =
                    (self.samples[y * self.stride + x] >> shift).min(255) as u8;
            }
        }
        PicturePlane {
            samples: out,
            stride: self.width,
            width: self.width,
            height: self.height,
        }
    }
}

/// HBD twin of [`PictureBuffer`] — 4:2:0 frame at any spec-supported
/// bit depth (8..=16). At `bit_depth == 8` this is functionally
/// identical to [`PictureBuffer`] but stored in `u16` cells; for
/// Main10 / Main12 it preserves the full luma dynamic range that the
/// legacy 8-bit buffer would truncate.
#[derive(Clone, Debug)]
pub struct PictureBuffer16 {
    pub luma: PicturePlane16,
    pub cb: PicturePlane16,
    pub cr: PicturePlane16,
}

impl PictureBuffer16 {
    /// Allocate a 4:2:0 frame at the given `bit_depth`. Chroma planes
    /// are seeded to mid-grey (`1 << (bit_depth - 1)`); the luma seed
    /// is supplied by the caller.
    pub fn yuv420_filled(luma_w: usize, luma_h: usize, seed: u16, bit_depth: u32) -> Self {
        let mid = 1u16 << (bit_depth - 1);
        Self {
            luma: PicturePlane16::filled(luma_w, luma_h, seed, bit_depth),
            cb: PicturePlane16::filled(luma_w / 2, luma_h / 2, mid, bit_depth),
            cr: PicturePlane16::filled(luma_w / 2, luma_h / 2, mid, bit_depth),
        }
    }
}

/// Owned reference-sample bundle ready to feed into the intra
/// predictors of [`crate::intra`].
///
/// Per §8.4.5.2.8 the arrays are sized:
/// * `above`: `n_tb_w + 1` samples covering `p[0..nTbW][-1]`
/// * `left` : `n_tb_h + 1` samples covering `p[-1][0..nTbH]`
/// * `top_left` is `p[-1][-1]`.
///
/// Unavailable samples are substituted from the nearest available
/// neighbour, and when *every* side is unavailable each sample is set
/// to `1 << (BitDepth - 1)` (mid-grey).
#[derive(Clone, Debug)]
pub struct OwnedIntraRefs {
    pub above: Vec<i16>,
    pub left: Vec<i16>,
    pub top_left: i16,
}

impl OwnedIntraRefs {
    /// Build a reference-sample bundle for a `n_tb_w × n_tb_h` TB whose
    /// top-left luma sample sits at picture-absolute `(x0, y0)`. Reads
    /// already-reconstructed samples out of `plane`. Edges that fall
    /// outside the picture (or that the caller has marked unavailable
    /// via the booleans) are filled with mid-grey using the spec's
    /// substitution helper [`crate::intra::substitute_references`].
    pub fn from_plane(
        plane: &PicturePlane,
        x0: usize,
        y0: usize,
        n_tb_w: usize,
        n_tb_h: usize,
        above_avail: bool,
        left_avail: bool,
        bit_depth: u32,
    ) -> Self {
        let mid = 1i16 << (bit_depth - 1);
        let mut above = vec![mid; n_tb_w + 1];
        let mut left = vec![mid; n_tb_h + 1];
        let mut top_left = mid;

        let mut above_mask = vec![false; n_tb_w + 1];
        let mut left_mask = vec![false; n_tb_h + 1];
        let mut tl_mask = false;

        if above_avail && y0 > 0 {
            // p[x][-1] for x = 0..n_tb_w. Spec's planar uses x = nTbW
            // too (the corner-extension sample). Read it from the
            // reconstructed plane; if x falls outside the picture, the
            // entry stays masked-off so substitution kicks in.
            for x in 0..=n_tb_w {
                let xi = x0 + x;
                if let Some(v) = plane.get(xi, y0 - 1) {
                    above[x] = v as i16;
                    above_mask[x] = true;
                }
            }
        }
        if left_avail && x0 > 0 {
            for y in 0..=n_tb_h {
                let yi = y0 + y;
                if let Some(v) = plane.get(x0 - 1, yi) {
                    left[y] = v as i16;
                    left_mask[y] = true;
                }
            }
        }
        if above_avail && left_avail && x0 > 0 && y0 > 0 {
            if let Some(v) = plane.get(x0 - 1, y0 - 1) {
                top_left = v as i16;
                tl_mask = true;
            }
        }
        crate::intra::substitute_references(
            &mut above,
            &mut left,
            &mut top_left,
            &above_mask,
            &left_mask,
            tl_mask,
            mid,
        );
        Self {
            above,
            left,
            top_left,
        }
    }
}

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

/// HBD twin of [`reconstruct_tb_into`] — writes the eq. 1426
/// `Clip1(pred + res)` into a `u16` destination plane at the supplied
/// `bit_depth` (no narrowing). Used by the Main10 / Main12
/// reconstruction path; for the legacy 8-bit pipeline keep using the
/// `u8` overload (it is byte-identical at `bit_depth == 8`).
#[allow(clippy::too_many_arguments)]
pub fn reconstruct_tb_into_u16(
    dst: &mut [u16],
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
            "h266 reconstruct u16: pred / residual size mismatch",
        ));
    }
    if x + n_tb_w > dst_stride || y + n_tb_h > dst_height {
        return Err(Error::invalid(
            "h266 reconstruct u16: TB does not fit in destination",
        ));
    }
    if !(8..=16).contains(&bit_depth) {
        return Err(Error::invalid(format!(
            "h266 reconstruct u16: bit_depth {bit_depth} out of supported range 8..=16",
        )));
    }
    let max = (1i32 << bit_depth) - 1;
    for row in 0..n_tb_h {
        for col in 0..n_tb_w {
            let p = pred[row * n_tb_w + col] as i32;
            let r = residual[row * n_tb_w + col];
            let v = (p + r).clamp(0, max) as u16;
            dst[(y + row) * dst_stride + (x + col)] = v;
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

    /// `PicturePlane16::filled` round-trips a Main10 mid-grey seed
    /// (= 512) and rejects out-of-range seeds.
    #[test]
    fn picture_plane16_filled_main10() {
        let p = PicturePlane16::filled(8, 8, 512, 10);
        assert_eq!(p.bit_depth, 10);
        assert_eq!(p.width, 8);
        assert_eq!(p.samples.len(), 64);
        for &v in &p.samples {
            assert_eq!(v, 512);
        }
    }

    #[test]
    #[should_panic(expected = "exceeds bit_depth")]
    fn picture_plane16_filled_rejects_oversized_seed() {
        // seed = 2000 > (1 << 10) - 1 = 1023
        let _ = PicturePlane16::filled(4, 4, 2000, 10);
    }

    /// `PicturePlane16::set` enforces the bit-depth range.
    #[test]
    fn picture_plane16_set_clip_range() {
        let mut p = PicturePlane16::filled(4, 4, 0, 10);
        p.set(2, 2, 1023).unwrap();
        assert_eq!(p.get(2, 2), Some(1023));
        // 1024 exceeds the 10-bit range.
        assert!(p.set(0, 0, 1024).is_err());
        // out-of-bounds is also an error.
        assert!(p.set(4, 0, 0).is_err());
    }

    /// At `bit_depth == 8`, `reconstruct_tb_into_u16` is byte-identical
    /// to the legacy `reconstruct_tb_into` for the same pred/res inputs
    /// (samples just live in `u16` cells now).
    #[test]
    fn reconstruct_u16_bit8_matches_u8() {
        let mut u8_dst = vec![0u8; 16];
        let mut u16_dst = vec![0u16; 16];
        let pred: Vec<i16> = (0..16).map(|i| 100 + i as i16).collect();
        let res: Vec<i32> = (0..16).map(|i| if i % 3 == 0 { 5 } else { -3 }).collect();
        reconstruct_tb_into(&mut u8_dst, 4, 4, 0, 0, &pred, &res, 4, 4, 8).unwrap();
        reconstruct_tb_into_u16(&mut u16_dst, 4, 4, 0, 0, &pred, &res, 4, 4, 8).unwrap();
        for i in 0..16 {
            assert_eq!(u8_dst[i] as u16, u16_dst[i]);
        }
    }

    /// Main10 reconstruction preserves the full sub-1024 dynamic range
    /// — pred = 1000 + res = 20 = 1020 stays at 1020 (the legacy 8-bit
    /// path would right-shift this to 255).
    #[test]
    fn reconstruct_u16_main10_preserves_range() {
        let mut dst = vec![0u16; 16];
        let pred = vec![1000i16; 16];
        let res = vec![20i32; 16];
        reconstruct_tb_into_u16(&mut dst, 4, 4, 0, 0, &pred, &res, 4, 4, 10).unwrap();
        for &v in &dst {
            assert_eq!(v, 1020);
        }
    }

    /// Main10 reconstruction clips at `(1 << 10) - 1 = 1023`.
    #[test]
    fn reconstruct_u16_main10_clips_overflow() {
        let mut dst = vec![0u16; 4];
        let pred = vec![1020i16; 4];
        let res = vec![20i32; 4];
        reconstruct_tb_into_u16(&mut dst, 2, 2, 0, 0, &pred, &res, 2, 2, 10).unwrap();
        for &v in &dst {
            assert_eq!(v, 1023);
        }
    }

    /// `to_picture_plane_u8` is the canonical narrowing — Main10 1020
    /// becomes 8-bit 255.
    #[test]
    fn picture_plane16_narrow_to_u8() {
        let mut p = PicturePlane16::filled(4, 4, 0, 10);
        for y in 0..4 {
            for x in 0..4 {
                p.set(x, y, 1020).unwrap();
            }
        }
        let p8 = p.to_picture_plane_u8();
        for &v in &p8.samples {
            assert_eq!(v, 255);
        }
    }

    /// End-to-end Main10 reconstruction: DC-predicted 4x4 TB at a
    /// 10-bit mid-grey neighbour (512) + a 10-bit residual lands in a
    /// `u16` plane without any narrowing.
    #[test]
    fn mini_end_to_end_main10_dc_impulse() {
        use crate::intra::{predict_dc, IntraRefs};
        use crate::transform::{inverse_transform_2d, TrType};

        // Main10 mid-grey neighbours → DC prediction 512.
        let above = vec![512i16; 5];
        let left = vec![512i16; 5];
        let refs = IntraRefs {
            above: &above,
            left: &left,
            top_left: 512,
        };
        let pred = predict_dc(4, 4, &refs).unwrap();
        assert_eq!(pred, vec![512; 16]);

        let mut c = vec![0i32; 16];
        c[0] = 200;
        let d = dequantise_block(&c, 4, 4, 30, 10).unwrap();
        let res =
            inverse_transform_2d(4, 4, 1, 1, TrType::DctII, TrType::DctII, &d, 10, 17).unwrap();

        let mut buf = PictureBuffer16::yuv420_filled(4, 4, 512, 10);
        reconstruct_tb_into_u16(
            &mut buf.luma.samples,
            buf.luma.stride,
            buf.luma.height,
            0,
            0,
            &pred,
            &res,
            4,
            4,
            10,
        )
        .unwrap();
        // Every sample must remain in the Main10 range.
        for &v in &buf.luma.samples {
            assert!(v <= 1023, "Main10 sample {v} out of range [0, 1023]");
        }
        // DC-only input + constant prediction → spatially uniform.
        let first = buf.luma.samples[0];
        for &v in &buf.luma.samples {
            assert_eq!(v, first);
        }
        // The DC residual at QP=30, BD=10 lifts the prediction by a
        // small positive number — the result must exceed the legacy
        // 8-bit max-value 255 (i.e. genuinely benefits from u16).
        assert!(
            first > 255,
            "Main10 reconstruction must escape the 8-bit ceiling, got {first}",
        );
    }
}
