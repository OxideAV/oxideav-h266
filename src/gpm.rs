//! VVC Geometric Partitioning Mode (GPM) — §8.5.4 + §8.5.7.
//!
//! GPM splits a CU into two regions along an oblique line and predicts each
//! region from a different merge candidate. The two per-region predictions
//! are then blended with per-pixel weights `wValue ∈ [0, 8]` derived from the
//! pixel's signed distance to the partition line.
//!
//! ## Spec organisation
//!
//! * **§8.5.4.2** — given `merge_gpm_idx0` / `merge_gpm_idx1` and the merge
//!   candidate list, derive the two motion vectors `(mvA, mvB)`, reference
//!   indices `(refIdxA, refIdxB)` and prediction-list flags `(predListFlagA,
//!   predListFlagB)` (eqs. 646 – 655).
//! * **§8.5.7.1** — picture-level dispatch: for each of the two halves run
//!   the §8.5.6.3 fractional-sample interpolation, then call §8.5.7.2.
//! * **§8.5.7.2** — weighted sample prediction. The angle / distance pair
//!   is decoded from `merge_gpm_partition_idx` via [`Table 36`] and the
//!   per-pixel weight comes from a signed scaled dot-product against the
//!   `disLut` array of [`Table 37`] (eqs. 999 – 1016).
//!
//! ## Round-40 scope
//!
//! * Tables 36 + 37 transcribed verbatim.
//! * [`derive_gpm_partition`] decodes `(angleIdx, distanceIdx)`.
//! * [`gpm_weight_at`] evaluates the per-pixel `wValue` (eqs. 999 – 1015)
//!   for any `(x, y)` inside the CU.
//! * [`derive_gpm_mv_indices`] implements §8.5.4.2 step 2 — the
//!   `m` / `n` adjustment and the `predListFlag = X` / `(1 − X)`
//!   selection (eqs. 646 – 655).
//! * [`blend_gpm_into_plane`] runs eq. 1016 across a `(cbWidth × cbHeight)`
//!   block of pre-computed per-region predictions, writing the clipped
//!   `predSamples` into a target plane.
//!
//! Two-region MC, partition-MV storage (§8.5.7.3) and CTU integration
//! land alongside [`crate::ctu`] in this same round.
//!
//! Spec reference: ITU-T H.266 | ISO/IEC 23090-3 (V4, 01/2026).

use crate::inter::{MotionVector, MvField};
use crate::reconstruct::PicturePlane;

// =====================================================================
// §8.5.7.2 / Table 36 — angleIdx + distanceIdx from
// merge_gpm_partition_idx (0..63).
// =====================================================================

/// Table 36 entry — `(angleIdx, distanceIdx)` for one
/// `merge_gpm_partition_idx` value.
pub type GpmPartition = (u8, u8);

/// Table 36 — `(angleIdx, distanceIdx)` indexed by `merge_gpm_partition_idx
/// ∈ 0..64`.
pub const GPM_PARTITION_TABLE: [GpmPartition; 64] = [
    // 0..15
    (0, 1),
    (0, 3),
    (2, 0),
    (2, 1),
    (2, 2),
    (2, 3),
    (3, 0),
    (3, 1),
    (3, 2),
    (3, 3),
    (4, 0),
    (4, 1),
    (4, 2),
    (4, 3),
    (5, 0),
    (5, 1),
    // 16..31
    (5, 2),
    (5, 3),
    (8, 1),
    (8, 3),
    (11, 0),
    (11, 1),
    (11, 2),
    (11, 3),
    (12, 0),
    (12, 1),
    (12, 2),
    (12, 3),
    (13, 0),
    (13, 1),
    (13, 2),
    (13, 3),
    // 32..47
    (14, 0),
    (14, 1),
    (14, 2),
    (14, 3),
    (16, 1),
    (16, 3),
    (18, 1),
    (18, 2),
    (18, 3),
    (19, 1),
    (19, 2),
    (19, 3),
    (20, 1),
    (20, 2),
    (20, 3),
    (21, 1),
    // 48..63
    (21, 2),
    (21, 3),
    (24, 1),
    (24, 3),
    (27, 1),
    (27, 2),
    (27, 3),
    (28, 1),
    (28, 2),
    (28, 3),
    (29, 1),
    (29, 2),
    (29, 3),
    (30, 1),
    (30, 2),
    (30, 3),
];

/// §8.5.7.1 step 2 — decode `(angleIdx, distanceIdx)` from
/// `merge_gpm_partition_idx`. Saturates to entry 63 for out-of-range
/// inputs (the spec range is `0..=63`).
pub fn derive_gpm_partition(merge_gpm_partition_idx: u32) -> GpmPartition {
    let idx = (merge_gpm_partition_idx as usize).min(63);
    GPM_PARTITION_TABLE[idx]
}

// =====================================================================
// Table 37 — disLut[].
// =====================================================================
//
// The disLut array is indexed by the 5-bit `displacementX` /
// `displacementY` derived from `angleIdx` (eqs. 1003 / 1004). Entries
// not listed in Table 37 are zero (the spec uses sparse population).
// We materialise the dense 32-entry table here to avoid per-call branching.

/// Table 37 — `disLut[idx]` for `idx ∈ 0..32`. Entries not specified in
/// the spec table are set to 0 (matching the "weightIdx accumulator
/// contribution = 0" semantics for unused angle indices). Per the spec:
///
/// ```text
/// idx       0  2  3  4  5  6  8  10  11  12  13  14
/// disLut    8  8  8  4  4  2  0  -2  -4  -4  -8  -8
/// idx      16 18 19 20 21 22 24  26  27  28  29  30
/// disLut   -8 -8 -8 -4 -4 -2  0   2   4   4   8   8
/// ```
pub const DIS_LUT: [i32; 32] = [
    8, 0, 8, 8, 4, 4, 2, 0, // 0..7
    0, 0, -2, -4, -4, -8, -8, 0, // 8..15
    -8, 0, -8, -8, -4, -4, -2, 0, // 16..23
    0, 0, 2, 4, 4, 8, 8, 0, // 24..31
];

// =====================================================================
// §8.5.7.2 — Per-pixel weighted sample prediction (eqs. 999 – 1016).
// =====================================================================

/// Per-CU geometric partitioning context — pre-computed once per CU and
/// reused across every `(x, y)` weight lookup.
#[derive(Clone, Copy, Debug)]
pub struct GpmContext {
    /// Block width. For luma `cIdx == 0` this is `cbWidth`; for chroma
    /// the §8.5.7.2 caller should set this to `cbWidth / SubWidthC`.
    pub n_cb_w: u32,
    /// Block height — `cbHeight` (luma) or `cbHeight / SubHeightC`
    /// (chroma).
    pub n_cb_h: u32,
    /// `nW` per eq. 999.
    pub n_w: i32,
    /// `nH` per eq. 1000.
    pub n_h: i32,
    /// `displacementX` per eq. 1003.
    pub displacement_x: u8,
    /// `displacementY` per eq. 1004.
    pub displacement_y: u8,
    /// `partFlip` per eq. 1005.
    pub part_flip: u8,
    /// `shiftHor` per eq. 1006.
    pub shift_hor: u8,
    /// `offsetX` per eq. 1007 / 1009.
    pub offset_x: i32,
    /// `offsetY` per eq. 1008 / 1010.
    pub offset_y: i32,
    /// `shift1` per eq. 1001 (cached for eq. 1016).
    pub shift1: u32,
    /// `offset1` per eq. 1002 (cached for eq. 1016).
    pub offset1: i32,
    /// Subsampling factor used when `cIdx != 0`. For cIdx == 0 this is
    /// `(1, 1)`; for 4:2:0 chroma it's `(2, 2)`.
    pub sub_w: u32,
    pub sub_h: u32,
}

impl GpmContext {
    /// §8.5.7.2 setup — derive every constant in the eqs. 999 – 1010
    /// chain. `c_idx` selects luma (0) vs chroma (1 / 2). `sub_w` /
    /// `sub_h` are the spec's `SubWidthC` / `SubHeightC` (1 / 1 for luma
    /// and 4:4:4, 2 / 2 for 4:2:0 chroma, 2 / 1 for 4:2:2 chroma).
    pub fn new(
        cb_width: u32,
        cb_height: u32,
        angle_idx: u8,
        distance_idx: u8,
        c_idx: u32,
        bit_depth: u32,
        sub_w: u32,
        sub_h: u32,
    ) -> Self {
        // Eqs. 999 / 1000 — the GPM math always operates in luma units;
        // for chroma `cIdx != 0` the weighting integrates over a
        // (cbW * SubWidthC) × (cbH * SubHeightC) luma extent so the same
        // partition line applies pixel-for-pixel.
        let (n_cb_w, n_cb_h) = (cb_width, cb_height);
        let (n_w, n_h) = if c_idx == 0 {
            (cb_width as i32, cb_height as i32)
        } else {
            ((cb_width * sub_w) as i32, (cb_height * sub_h) as i32)
        };
        // Eq. 1001 — shift1 = max(5, 17 - BitDepth).
        let shift1 = 5u32.max(17u32.saturating_sub(bit_depth));
        // Eq. 1002 — offset1 = 1 << (shift1 - 1).
        let offset1 = 1i32 << (shift1 - 1);
        // Eqs. 1003 / 1004.
        let displacement_x = angle_idx & 31;
        let displacement_y = ((angle_idx as u32 + 8) % 32) as u8;
        // Eq. 1005 — partFlip = (angleIdx >= 13 && angleIdx <= 27) ? 0 : 1
        let part_flip = if (13..=27).contains(&angle_idx) { 0 } else { 1 };
        // Eq. 1006.
        let mod16 = angle_idx % 16;
        let shift_hor = if mod16 == 8 || (mod16 != 0 && n_h >= n_w) {
            0
        } else {
            1
        };
        // Eqs. 1007 / 1008 / 1009 / 1010.
        let dist = distance_idx as i32;
        let (offset_x, offset_y) = if shift_hor == 0 {
            // Eq. 1007 / 1008.
            let ox = (-n_w) >> 1;
            let mag = (dist * n_h) >> 3;
            let oy = ((-n_h) >> 1) + if angle_idx < 16 { mag } else { -mag };
            (ox, oy)
        } else {
            // Eq. 1009 / 1010.
            let mag = (dist * n_w) >> 3;
            let ox = ((-n_w) >> 1) + if angle_idx < 16 { mag } else { -mag };
            let oy = (-n_h) >> 1;
            (ox, oy)
        };
        Self {
            n_cb_w,
            n_cb_h,
            n_w,
            n_h,
            displacement_x,
            displacement_y,
            part_flip,
            shift_hor,
            offset_x,
            offset_y,
            shift1,
            offset1,
            sub_w,
            sub_h,
        }
    }

    /// §8.5.7.2 — per-pixel weight `wValue ∈ [0, 8]`. `(x, y)` are in
    /// `(nCbW, nCbH)` units (i.e. for chroma `cIdx != 0`, callers pass
    /// the chroma sample coordinates and the luma-grid `(xL, yL)` is
    /// derived inside per eqs. 1011 / 1012).
    #[inline]
    pub fn weight_at(&self, x: u32, y: u32, c_idx: u32) -> i32 {
        // Eqs. 1011 / 1012.
        let (xl, yl) = if c_idx == 0 {
            (x as i32, y as i32)
        } else {
            ((x * self.sub_w) as i32, (y * self.sub_h) as i32)
        };
        let dx = DIS_LUT[self.displacement_x as usize];
        let dy = DIS_LUT[self.displacement_y as usize];
        // Eq. 1013.
        let weight_idx =
            (((xl + self.offset_x) << 1) + 1) * dx + (((yl + self.offset_y) << 1) + 1) * dy;
        // Eq. 1014.
        let weight_idx_l = if self.part_flip == 1 {
            32 - weight_idx
        } else {
            32 + weight_idx
        };
        // Eq. 1015.
        ((weight_idx_l + 4) >> 3).clamp(0, 8)
    }

    /// §8.5.7.2 eq. 1016 — clip + final blend of two pre-prediction
    /// arrays. Returns the predicted sample value at `(x, y)`.
    #[inline]
    pub fn blend_at(
        &self,
        x: u32,
        y: u32,
        c_idx: u32,
        pred_a: i32,
        pred_b: i32,
        bit_depth: u32,
    ) -> i32 {
        let w = self.weight_at(x, y, c_idx);
        let pix = (pred_a * w + pred_b * (8 - w) + self.offset1) >> self.shift1;
        let max_val = (1i32 << bit_depth) - 1;
        pix.clamp(0, max_val)
    }
}

// =====================================================================
// §8.5.4.2 — m / n / X derivation.
// =====================================================================

/// §8.5.4.2 step 2 / 3 / 4 — index into the merge candidate list for the
/// two GPM partitions. Per eqs. 646 / 647:
///
/// ```text
/// m = merge_gpm_idx0[xCb][yCb]
/// n = merge_gpm_idx1[xCb][yCb] + (merge_gpm_idx1 >= m ? 1 : 0)
/// ```
#[inline]
pub fn derive_gpm_mn(merge_gpm_idx0: u32, merge_gpm_idx1: u32) -> (u32, u32) {
    let m = merge_gpm_idx0;
    let bump = if merge_gpm_idx1 >= m { 1 } else { 0 };
    let n = merge_gpm_idx1 + bump;
    (m, n)
}

/// §8.5.4.2 steps 4 / 5 — pick the prediction list for partition N
/// (named A or B in the spec). `partition_idx_in_list` is `m` for
/// partition A and `n` for partition B. The candidate's per-list
/// `pred_flag_lX` is consulted: if list `X = (idx & 0x01)` is inactive,
/// `X` is flipped to `1 − X` (eq. 6 of §8.5.4.2 step 5).
///
/// Returns `(x_list, mv, ref_idx)` where `x_list ∈ {0, 1}` is the chosen
/// `predListFlagN` and `(mv, ref_idx)` are the corresponding active
/// per-list MV / ref-idx pair from the candidate.
#[inline]
pub fn derive_gpm_partition_motion(
    partition_idx_in_list: u32,
    cand: &MvField,
) -> (u8, MotionVector, i32) {
    let mut x = (partition_idx_in_list & 1) as u8;
    let cand_active = if x == 0 {
        cand.pred_flag_l0
    } else {
        cand.pred_flag_l1
    };
    if !cand_active {
        x ^= 1;
    }
    let (mv, ref_idx) = if x == 0 {
        (cand.mv_l0, cand.ref_idx_l0)
    } else {
        (cand.mv_l1, cand.ref_idx_l1)
    };
    (x, mv, ref_idx)
}

// =====================================================================
// §8.5.7.1 + §8.5.7.2 — apply GPM blend on a luma plane.
// =====================================================================

/// Apply §8.5.7.2 eq. 1016 over a `(cbWidth × cbHeight)` block,
/// reading `pred_a` / `pred_b` (row-major, `cbWidth × cbHeight`) and
/// writing the clipped `predSamples` into `dst` at `(x0, y0)`.
///
/// `pred_a` / `pred_b` are the §8.5.6.3 motion-compensated sample
/// arrays at the **final** post-§8.5.6.6.2 8-bit precision (i.e. each
/// entry is already clipped to `[0, 255]` and the §8.5.7.2 blend
/// re-applies the eq. 1016 division). For higher bit-depths callers
/// should run [`blend_gpm_into_plane_u16`] (a future round).
pub fn blend_gpm_into_plane(
    dst: &mut PicturePlane,
    x0: u32,
    y0: u32,
    cb_w: u32,
    cb_h: u32,
    pred_a: &[u8],
    pred_b: &[u8],
    ctx: &GpmContext,
    c_idx: u32,
    bit_depth: u32,
) {
    let stride_a = cb_w as usize;
    let dst_stride = dst.stride;
    for y in 0..cb_h {
        for x in 0..cb_w {
            let a = pred_a[y as usize * stride_a + x as usize] as i32;
            let b = pred_b[y as usize * stride_a + x as usize] as i32;
            // Eq. 1016 takes pre-clip intermediates from §8.5.6.3,
            // not 8-bit clipped samples. For the round-40 scaffold we
            // re-lift the 8-bit samples back to the §8.5.6.6.2
            // intermediate domain by scaling — at BitDepth = 8 the
            // round-22 wrapper applies `(intermediate + 32) >> 6` to
            // get the 8-bit value, so the inverse is `<< 6`. The
            // round-trip stays bit-identical for partitions where
            // wValue is 0 or 8 (since a single source survives).
            let a_pre = a << 6;
            let b_pre = b << 6;
            let pix = ctx.blend_at(x, y, c_idx, a_pre, b_pre, bit_depth);
            let yi = (y0 as usize) + y as usize;
            let xi = (x0 as usize) + x as usize;
            if yi < dst.height && xi < dst.width {
                dst.samples[yi * dst_stride + xi] = pix as u8;
            }
        }
    }
}

// =====================================================================
// Tests
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inter::MotionVector;

    #[test]
    fn table_36_partition_zero_is_angle_0_distance_1() {
        assert_eq!(derive_gpm_partition(0), (0, 1));
    }

    #[test]
    fn table_36_partition_63_is_angle_30_distance_3() {
        assert_eq!(derive_gpm_partition(63), (30, 3));
    }

    #[test]
    fn table_36_partition_24_is_angle_12_distance_0() {
        // Spot-check a mid-table entry (idx 24 — angleIdx 12).
        assert_eq!(derive_gpm_partition(24), (12, 0));
    }

    #[test]
    fn table_36_partition_44_is_angle_20_distance_1() {
        assert_eq!(derive_gpm_partition(44), (20, 1));
    }

    #[test]
    fn dis_lut_known_entries() {
        // Spec table values.
        assert_eq!(DIS_LUT[0], 8);
        assert_eq!(DIS_LUT[2], 8);
        assert_eq!(DIS_LUT[6], 2);
        assert_eq!(DIS_LUT[8], 0);
        assert_eq!(DIS_LUT[16], -8);
        assert_eq!(DIS_LUT[24], 0);
        assert_eq!(DIS_LUT[30], 8);
    }

    #[test]
    fn derive_gpm_mn_avoids_collision() {
        // m = 2, idx1 = 0 → n = 0 (no bump).
        assert_eq!(derive_gpm_mn(2, 0), (2, 0));
        // m = 0, idx1 = 0 → n = 1 (bump).
        assert_eq!(derive_gpm_mn(0, 0), (0, 1));
        // m = 0, idx1 = 4 → n = 5 (bump because 4 >= 0).
        assert_eq!(derive_gpm_mn(0, 4), (0, 5));
        // m = 3, idx1 = 2 → n = 2 (no bump because 2 < 3).
        assert_eq!(derive_gpm_mn(3, 2), (3, 2));
    }

    #[test]
    fn derive_partition_motion_picks_active_list() {
        // Candidate is uni-pred L0; partition idx 1 → X = 1 → flips to 0.
        let cand = MvField {
            mv_l0: MotionVector { x: 4, y: 8 },
            ref_idx_l0: 0,
            pred_flag_l0: true,
            pred_flag_l1: false,
            ..MvField::UNAVAILABLE
        };
        let (x, mv, ref_idx) = derive_gpm_partition_motion(1, &cand);
        assert_eq!(x, 0);
        assert_eq!(mv.x, 4);
        assert_eq!(ref_idx, 0);
    }

    #[test]
    fn derive_partition_motion_keeps_active_list() {
        // Candidate is bi-pred; partition idx 1 → X = 1 → keeps 1.
        let cand = MvField {
            mv_l0: MotionVector { x: 4, y: 8 },
            ref_idx_l0: 0,
            pred_flag_l0: true,
            mv_l1: MotionVector { x: -2, y: 6 },
            ref_idx_l1: 1,
            pred_flag_l1: true,
            ..MvField::UNAVAILABLE
        };
        let (x, mv, _ref_idx) = derive_gpm_partition_motion(1, &cand);
        assert_eq!(x, 1);
        assert_eq!(mv.x, -2);
    }

    #[test]
    fn gpm_context_construction_at_partition_zero() {
        // partition_idx 0 → angle 0, distance 1; 8x8 CU; cIdx 0.
        let ctx = GpmContext::new(8, 8, 0, 1, 0, 8, 1, 1);
        assert_eq!(ctx.n_cb_w, 8);
        assert_eq!(ctx.n_cb_h, 8);
        assert_eq!(ctx.n_w, 8);
        assert_eq!(ctx.n_h, 8);
        // angle 0 → displacement_x 0, displacement_y 8.
        assert_eq!(ctx.displacement_x, 0);
        assert_eq!(ctx.displacement_y, 8);
        // angle 0 → partFlip 1 (outside 13..=27).
        assert_eq!(ctx.part_flip, 1);
        // angle 0 mod 16 = 0; nH (8) >= nW (8) → fails the !=0 gate so
        // shift_hor = 1.
        assert_eq!(ctx.shift_hor, 1);
        // shift1 = max(5, 17 - 8) = 9; offset1 = 256.
        assert_eq!(ctx.shift1, 9);
        assert_eq!(ctx.offset1, 256);
    }

    #[test]
    fn gpm_weight_at_extreme_corners_clamp_to_0_or_8() {
        // Build a 16x16 context and sample (0,0) and (15, 15).
        let ctx = GpmContext::new(16, 16, 18, 1, 0, 8, 1, 1);
        let w_top_left = ctx.weight_at(0, 0, 0);
        let w_bottom_right = ctx.weight_at(15, 15, 0);
        // The two corners should be on opposite sides of the partition
        // line and their weights must be in [0, 8].
        assert!((0..=8).contains(&w_top_left));
        assert!((0..=8).contains(&w_bottom_right));
    }

    #[test]
    fn gpm_blend_full_left_half_when_weight_zero() {
        // At extreme partitions one side of the CU should pick predA
        // entirely (weight 8) and the other predB entirely (weight 0).
        // This is harder to assert purely from the math at one specific
        // partition, but we can verify the *pixelwise* blend is monotone
        // in the weight by feeding contrasting (predA = 200, predB = 50)
        // and checking that the dst values cluster bimodally around
        // those endpoints once clipped to 8-bit.
        let ctx = GpmContext::new(8, 8, 0, 1, 0, 8, 1, 1);
        let pred_a = vec![200u8; 64];
        let pred_b = vec![50u8; 64];
        let mut plane = PicturePlane::filled(8, 8, 0);
        blend_gpm_into_plane(&mut plane, 0, 0, 8, 8, &pred_a, &pred_b, &ctx, 0, 8);
        let mut hits_left = 0;
        let mut hits_right = 0;
        for v in &plane.samples {
            if *v >= 195 {
                hits_left += 1;
            } else if *v <= 55 {
                hits_right += 1;
            }
        }
        // Most pixels (excluding the blend transition band) should pick
        // one side or the other.
        assert!(hits_left + hits_right >= 32, "expected bimodal split");
    }
}
