//! VVC inter prediction — round-21 P-slice subset (§7.3.11.5 + §8.5).
//!
//! This module covers the **smallest viable** P-slice path:
//!
//! * [`MotionVector`] / [`MvField`] / [`MotionField`] — per-4x4-block
//!   storage of the MV / refIdx / predFlag tuple a CU writes after its
//!   `merge_data()` resolves. Future rounds extend this to multiple
//!   reference pictures and bi-prediction; today only refIdx 0 of
//!   [`MotionField::list_l0`] is exercised.
//! * [`ReferencePicture`] — a 4:2:0 reference frame plus its POC. The
//!   round-21 walker carries a single reference (the last decoded IDR)
//!   and only supports `predFlagL0 == 1, predFlagL1 == 0` (P-slice).
//! * [`derive_spatial_merge_candidates`] — §8.5.2.3 5-position list
//!   (B1 → A1 → B0 → A0 → B2 with the spec's redundancy checks). The
//!   `Log2ParMrgLevel` shared-merge collapse is applied per the spec
//!   condition `xCb >> Log2ParMrgLevel == xNbX >> Log2ParMrgLevel &&
//!   yCb >> Log2ParMrgLevel == yNbX >> Log2ParMrgLevel`.
//! * [`build_merge_cand_list`] — §8.5.2.2 step 5 spatial-only
//!   assembly + step 9 zero-MV padding. The §8.5.2.4 pairwise average
//!   and §8.5.2.6 HMVP candidates are intentionally out of scope for
//!   this round (they will land alongside more aggressive
//!   merge-mode features in r22+); the temporal candidate (§8.5.2.11)
//!   needs collocated-picture machinery and is also out of scope.
//! * [`MergeData`] / [`InterCuInfo`] — parsed-syntax records produced
//!   by the leaf-CU reader for cu_skip_flag / regular_merge_flag /
//!   merge_idx; consumed by the CTU walker to drive merge derivation.
//! * [`mc_copy_block_int`] — integer-pel motion compensation. The
//!   round-21 path takes one block from a reference picture and writes
//!   it into the current picture buffer at integer (xCb + mvLX[0]/16,
//!   yCb + mvLX[1]/16). For all-zero MVs (the all-skip P-slice case)
//!   this collapses to a memcpy of the reference frame; the bilinear
//!   stub for chroma is used unconditionally because round-21 only
//!   exercises integer alignment, in which case the bilinear collapses
//!   to a copy too.
//!
//! Spec reference: ITU-T H.266 | ISO/IEC 23090-3 (V4, 01/2026). The
//! implementation is spec-only; no third-party VVC decoder source was
//! consulted.

use oxideav_core::{Error, Result};

use crate::reconstruct::{PictureBuffer, PicturePlane};

/// 1/16-pel motion vector (§8.5.2 fractional-sample accuracy).
///
/// Components are stored in spec-units of `1/16 sample`; integer pel
/// `(dx, dy)` is therefore `MotionVector { x: dx*16, y: dy*16 }`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MotionVector {
    pub x: i32,
    pub y: i32,
}

impl MotionVector {
    pub const ZERO: MotionVector = MotionVector { x: 0, y: 0 };

    /// Construct an integer-pel MV (auto-converted to 1/16 units).
    pub fn from_int_pel(dx: i32, dy: i32) -> Self {
        MotionVector {
            x: dx * 16,
            y: dy * 16,
        }
    }

    /// Integer-pel component derived as `mv >> 4` (spec §8.5.6.3.x;
    /// `xFracL = mvLX[0] & 15`, `xIntL = (mvLX[0] >> 4)`). Ignores
    /// the fractional remainder.
    pub fn int_x(self) -> i32 {
        self.x >> 4
    }
    pub fn int_y(self) -> i32 {
        self.y >> 4
    }

    /// True when both components are integer-pel aligned (no fractional
    /// remainder). Round-21 MC only handles this case.
    pub fn is_integer_pel(self) -> bool {
        (self.x & 0xF) == 0 && (self.y & 0xF) == 0
    }
}

/// Per-block (4x4 luma) motion field record. Mirrors the spec's per-
/// position arrays `MvLX[x][y]`, `RefIdxLX[x][y]`, `PredFlagLX[x][y]`
/// for `X = 0` (round-21 covers L0 only).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MvField {
    pub mv_l0: MotionVector,
    pub ref_idx_l0: i32,
    pub pred_flag_l0: bool,
    /// Marker for the per-CU `CuSkipFlag[x][y]` propagation needed by
    /// the §9.3.4.2.2 cu_skip_flag context derivation. Spec stores
    /// this on the per-CU grid, not the 4x4 grid; we conservatively
    /// replicate it onto every covered block.
    pub cu_skip_flag: bool,
    /// Marker for the per-CU `CuPredMode[x][y]` propagation. True =
    /// MODE_INTER, false = MODE_INTRA. Used by §9.3.4.2.2 pred_mode
    /// context derivation when the round-22+ CU walker brings non-skip
    /// inter CUs online.
    pub mode_inter: bool,
    /// True when this 4x4 block is "available" — i.e. some CU has
    /// written into it. Default `false` so picture-edge / pre-decode
    /// neighbours register as unavailable.
    pub available: bool,
}

impl MvField {
    /// "Unavailable" sentinel — all zeros + `available = false`.
    pub const UNAVAILABLE: MvField = MvField {
        mv_l0: MotionVector::ZERO,
        ref_idx_l0: -1,
        pred_flag_l0: false,
        cu_skip_flag: false,
        mode_inter: false,
        available: false,
    };
}

/// Per-picture motion field, sampled at 4x4 luma granularity (the
/// finest grid the spec writes per-block MV state at — §7.4.4 / Table
/// 15). The grid covers `pic_width_luma / 4` × `pic_height_luma / 4`
/// entries; CU writes broadcast a single MvField across every covered
/// 4x4 block.
#[derive(Clone, Debug)]
pub struct MotionField {
    /// Width in 4x4 blocks.
    pub blocks_w: u32,
    /// Height in 4x4 blocks.
    pub blocks_h: u32,
    /// Row-major storage; `field[y * blocks_w + x]`.
    pub field: Vec<MvField>,
}

impl MotionField {
    pub fn new(pic_w_luma: u32, pic_h_luma: u32) -> Self {
        let bw = pic_w_luma.div_ceil(4);
        let bh = pic_h_luma.div_ceil(4);
        Self {
            blocks_w: bw,
            blocks_h: bh,
            field: vec![MvField::UNAVAILABLE; (bw * bh) as usize],
        }
    }

    /// Sample at picture-absolute luma `(x, y)` — returns the MvField
    /// for the 4x4 block containing that sample; `UNAVAILABLE` when out
    /// of bounds.
    pub fn get_at_luma(&self, x: i32, y: i32) -> MvField {
        if x < 0 || y < 0 {
            return MvField::UNAVAILABLE;
        }
        let bx = (x as u32) / 4;
        let by = (y as u32) / 4;
        if bx >= self.blocks_w || by >= self.blocks_h {
            return MvField::UNAVAILABLE;
        }
        self.field[(by * self.blocks_w + bx) as usize]
    }

    /// Write the same MvField into every 4x4 block touched by the
    /// rectangle `[x, x+w) x [y, y+h)` (luma units). Out-of-bounds
    /// writes are clipped silently — the caller is responsible for
    /// passing a CU geometry that fits inside the picture.
    pub fn write_block(&mut self, x: u32, y: u32, w: u32, h: u32, mv: MvField) {
        let bx0 = x / 4;
        let by0 = y / 4;
        let bx1 = (x + w).div_ceil(4).min(self.blocks_w);
        let by1 = (y + h).div_ceil(4).min(self.blocks_h);
        for by in by0..by1 {
            for bx in bx0..bx1 {
                self.field[(by * self.blocks_w + bx) as usize] = mv;
            }
        }
    }
}

/// Reference picture used by the inter-prediction MC. Currently a 4:2:0
/// frame snapshot + a POC integer (only the order matters within a
/// `RefPicListN`; the exact value does not affect the integer-pel MC
/// path).
#[derive(Clone, Debug)]
pub struct ReferencePicture {
    pub poc: i32,
    pub frame: PictureBuffer,
}

/// Spatial-merge neighbour position labels used by §8.5.2.3.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpatialMergePos {
    B1,
    A1,
    B0,
    A0,
    B2,
}

/// Output of [`derive_spatial_merge_candidates`] — the per-position
/// availability flag + MvField captured from the spec's neighbouring
/// `(xNbX, yNbX)` lookups (eq. 492-516).
#[derive(Clone, Copy, Debug, Default)]
pub struct SpatialMergeCandidate {
    pub available: bool,
    pub field: MvField,
}

/// §8.5.2.3 spatial-merge candidate derivation — the 5-position
/// (B1, A1, B0, A0, B2) list with the spec's redundancy checks.
///
/// Inputs:
/// * `xcb / ycb` — top-left luma sample of the current CB (picture-
///   absolute).
/// * `cb_w / cb_h` — CB dimensions in luma samples.
/// * `mvf` — the per-picture MotionField covering the slice.
/// * `log2_par_mrg_level` — `Log2ParMrgLevel` from
///   `pps_log2_parallel_merge_level_minus2 + 2` (§7.4.3.5). Causes the
///   spec to mark a neighbour unavailable when both blocks share the
///   same parallel-merge tile.
///
/// Returns availability + MvField for each of the five positions in
/// the order they are evaluated by §8.5.2.3 (B1 first, then A1, B0,
/// A0, B2).
pub fn derive_spatial_merge_candidates(
    xcb: i32,
    ycb: i32,
    cb_w: i32,
    cb_h: i32,
    mvf: &MotionField,
    log2_par_mrg_level: u32,
) -> [SpatialMergeCandidate; 5] {
    let mut out = [SpatialMergeCandidate::default(); 5];
    let par_shift = log2_par_mrg_level;
    let same_par = |x1: i32, y1: i32, x2: i32, y2: i32| -> bool {
        (x1 >> par_shift) == (x2 >> par_shift) && (y1 >> par_shift) == (y2 >> par_shift)
    };

    // ---- B1: (xCb + cbWidth - 1, yCb - 1) ---------------------------
    let xb1 = xcb + cb_w - 1;
    let yb1 = ycb - 1;
    let raw_b1 = mvf.get_at_luma(xb1, yb1);
    let available_b1 = raw_b1.available && !same_par(xcb, ycb, xb1, yb1);
    out[0] = SpatialMergeCandidate {
        available: available_b1,
        field: if available_b1 {
            raw_b1
        } else {
            MvField::UNAVAILABLE
        },
    };

    // ---- A1: (xCb - 1, yCb + cbHeight - 1) --------------------------
    let xa1 = xcb - 1;
    let ya1 = ycb + cb_h - 1;
    let raw_a1 = mvf.get_at_luma(xa1, ya1);
    let mut available_a1 = raw_a1.available && !same_par(xcb, ycb, xa1, ya1);
    // Redundancy: drop A1 if it has same MV/refIdx as B1.
    if available_a1 && available_b1 && mvf_matches(&raw_a1, &out[0].field) {
        available_a1 = false;
    }
    out[1] = SpatialMergeCandidate {
        available: available_a1,
        field: if available_a1 {
            raw_a1
        } else {
            MvField::UNAVAILABLE
        },
    };

    // ---- B0: (xCb + cbWidth, yCb - 1) -------------------------------
    let xb0 = xcb + cb_w;
    let yb0 = ycb - 1;
    let raw_b0 = mvf.get_at_luma(xb0, yb0);
    let mut available_b0 = raw_b0.available && !same_par(xcb, ycb, xb0, yb0);
    if available_b0 && available_b1 && mvf_matches(&raw_b0, &out[0].field) {
        available_b0 = false;
    }
    out[2] = SpatialMergeCandidate {
        available: available_b0,
        field: if available_b0 {
            raw_b0
        } else {
            MvField::UNAVAILABLE
        },
    };

    // ---- A0: (xCb - 1, yCb + cbHeight) ------------------------------
    let xa0 = xcb - 1;
    let ya0 = ycb + cb_h;
    let raw_a0 = mvf.get_at_luma(xa0, ya0);
    let mut available_a0 = raw_a0.available && !same_par(xcb, ycb, xa0, ya0);
    if available_a0 && available_a1 && mvf_matches(&raw_a0, &out[1].field) {
        available_a0 = false;
    }
    out[3] = SpatialMergeCandidate {
        available: available_a0,
        field: if available_a0 {
            raw_a0
        } else {
            MvField::UNAVAILABLE
        },
    };

    // ---- B2: (xCb - 1, yCb - 1) -------------------------------------
    let xb2 = xcb - 1;
    let yb2 = ycb - 1;
    let raw_b2 = mvf.get_at_luma(xb2, yb2);
    let mut available_b2 = raw_b2.available && !same_par(xcb, ycb, xb2, yb2);
    // §8.5.2.3 last bullet: B2 dropped also when A0+A1+B0+B1 == 4
    // (i.e. all four prior candidates available).
    if available_b2 && available_a1 && mvf_matches(&raw_b2, &out[1].field) {
        available_b2 = false;
    }
    if available_b2 && available_b1 && mvf_matches(&raw_b2, &out[0].field) {
        available_b2 = false;
    }
    if available_b2 && available_a0 && available_a1 && available_b0 && available_b1 {
        available_b2 = false;
    }
    out[4] = SpatialMergeCandidate {
        available: available_b2,
        field: if available_b2 {
            raw_b2
        } else {
            MvField::UNAVAILABLE
        },
    };

    out
}

/// True when two MvFields encode the same (refIdx, mv) on L0. The
/// round-21 redundancy check ignores L1 since P-slices never set
/// `predFlagL1 = 1`.
fn mvf_matches(a: &MvField, b: &MvField) -> bool {
    a.pred_flag_l0 == b.pred_flag_l0 && a.ref_idx_l0 == b.ref_idx_l0 && a.mv_l0 == b.mv_l0
}

/// §8.5.2.2 step 5 + step 9 — spatial-only mergeCandList assembly with
/// zero-MV padding to `MaxNumMergeCand`. Pairwise average and HMVP are
/// intentionally not yet wired (r22+ scope).
///
/// The list is built in spec walk order:
///   1. If availableFlagB1 → push B1
///   2. If availableFlagA1 → push A1
///   3. If availableFlagB0 → push B0
///   4. If availableFlagA0 → push A0
///   5. If availableFlagB2 → push B2
///   6. While len < max → push zero-MV (refIdx 0).
pub fn build_merge_cand_list(
    spatial: &[SpatialMergeCandidate; 5],
    max_num_merge_cand: u32,
) -> Vec<MvField> {
    let mut out: Vec<MvField> = Vec::with_capacity(max_num_merge_cand as usize);
    for cand in spatial {
        if cand.available && (out.len() as u32) < max_num_merge_cand {
            out.push(cand.field);
        }
    }
    while (out.len() as u32) < max_num_merge_cand {
        out.push(MvField {
            mv_l0: MotionVector::ZERO,
            ref_idx_l0: 0,
            pred_flag_l0: true,
            cu_skip_flag: false,
            mode_inter: true,
            available: true,
        });
    }
    out
}

/// Parsed `merge_data()` syntax (round-21 subset — regular merge only).
/// Only the regular-merge sub-tree is exercised: `regular_merge_flag = 1`
/// + `merge_idx`. MMVD / CIIP / GPM / subblock merge are out of scope.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MergeData {
    /// `regular_merge_flag[x0][y0]` — `true` when the regular-merge
    /// branch is taken. Inferred to `true` in the round-21 walker
    /// (CIIP/GPM gates collapse, MMVD is disabled in the SPS).
    pub regular_merge_flag: bool,
    /// `merge_idx[x0][y0]` — index into `mergeCandList`.
    pub merge_idx: u32,
}

/// Per-CU parsed inter-syntax record (round-21 P-slice subset).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct InterCuInfo {
    /// `cu_skip_flag[x0][y0]`.
    pub cu_skip_flag: bool,
    /// `general_merge_flag[x0][y0]` — inferred to 1 when cu_skip_flag
    /// is 1 (§7.4.12.5).
    pub general_merge_flag: bool,
    pub merge_data: MergeData,
}

/// Block-copy MC: copy `(w, h)` luma samples from `ref_pic.luma` at
/// integer-pel offset `(ref_x, ref_y)` to `dst.luma` at `(dst_x,
/// dst_y)`. Out-of-bounds source coordinates are clamped to the
/// reference picture's edges (mirrors §8.5.6.3.2 picture-edge clipping
/// for integer positions).
pub fn mc_copy_block_int(
    dst: &mut PicturePlane,
    dst_x: u32,
    dst_y: u32,
    w: u32,
    h: u32,
    src: &PicturePlane,
    src_x: i32,
    src_y: i32,
) -> Result<()> {
    if dst_x as usize + w as usize > dst.width || dst_y as usize + h as usize > dst.height {
        return Err(Error::invalid(format!(
            "h266 MC: destination block ({},{}) {}x{} out of plane bounds {}x{}",
            dst_x, dst_y, w, h, dst.width, dst.height
        )));
    }
    let s_w = src.width as i32;
    let s_h = src.height as i32;
    let s_stride = src.stride;
    let d_stride = dst.stride;
    for r in 0..h as i32 {
        let sy = (src_y + r).clamp(0, s_h - 1) as usize;
        let dy = dst_y as usize + r as usize;
        for c in 0..w as i32 {
            let sx = (src_x + c).clamp(0, s_w - 1) as usize;
            let dx = dst_x as usize + c as usize;
            dst.samples[dy * d_stride + dx] = src.samples[sy * s_stride + sx];
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_field(w: u32, h: u32) -> MotionField {
        MotionField::new(w, h)
    }

    #[test]
    fn motion_vector_int_pel_helpers() {
        let mv = MotionVector::from_int_pel(3, -7);
        assert_eq!(mv.x, 48);
        assert_eq!(mv.y, -112);
        assert_eq!(mv.int_x(), 3);
        assert_eq!(mv.int_y(), -7);
        assert!(mv.is_integer_pel());
        let mv2 = MotionVector { x: 3, y: 0 };
        assert!(!mv2.is_integer_pel());
    }

    #[test]
    fn motion_field_get_at_luma_clipping() {
        let mut mf = empty_field(64, 64);
        let mvf = MvField {
            mv_l0: MotionVector::from_int_pel(1, 1),
            ref_idx_l0: 0,
            pred_flag_l0: true,
            available: true,
            ..Default::default()
        };
        mf.write_block(8, 8, 8, 8, mvf);
        // A sample inside the written block.
        let g = mf.get_at_luma(10, 12);
        assert!(g.available);
        assert_eq!(g.mv_l0.int_x(), 1);
        // A sample outside.
        let g_out = mf.get_at_luma(0, 0);
        assert!(!g_out.available);
        // Negative → unavailable.
        assert!(!mf.get_at_luma(-1, 5).available);
        assert!(!mf.get_at_luma(5, -1).available);
        // Beyond-edge → unavailable.
        assert!(!mf.get_at_luma(64, 0).available);
    }

    #[test]
    fn spatial_merge_top_left_corner_all_unavailable() {
        // CU at (0,0) size 16x16: every neighbour position is at a
        // negative coordinate, so all five candidates are unavailable.
        let mf = empty_field(64, 64);
        let cands = derive_spatial_merge_candidates(0, 0, 16, 16, &mf, 2);
        for c in &cands {
            assert!(!c.available);
        }
    }

    #[test]
    fn spatial_merge_left_only_picks_a1() {
        // CU at (16, 0) size 16x16. A1 lives at (15, 15) — write a
        // motion field there and verify A1 is picked up. B1/B0/B2
        // should all read from y=-1 (unavailable since negative).
        let mut mf = empty_field(64, 64);
        let mvf = MvField {
            mv_l0: MotionVector::from_int_pel(2, -1),
            ref_idx_l0: 0,
            pred_flag_l0: true,
            available: true,
            ..Default::default()
        };
        mf.write_block(0, 0, 16, 16, mvf);
        let cands = derive_spatial_merge_candidates(16, 0, 16, 16, &mf, 2);
        assert!(!cands[0].available, "B1 at y=-1 must be unavailable");
        assert!(cands[1].available, "A1 at (15,15) must be available");
        assert!(!cands[2].available);
        assert!(!cands[3].available);
        assert!(!cands[4].available);
        assert_eq!(cands[1].field.mv_l0.int_x(), 2);
        assert_eq!(cands[1].field.mv_l0.int_y(), -1);
    }

    #[test]
    fn spatial_merge_redundancy_drops_a1_when_matches_b1() {
        // Build a setup where both B1 (above-right) and A1 (left)
        // resolve to the same MV/refIdx. A1 should be dropped per
        // §8.5.2.3 redundancy check.
        let mut mf = empty_field(64, 64);
        let mvf = MvField {
            mv_l0: MotionVector::from_int_pel(0, 0),
            ref_idx_l0: 0,
            pred_flag_l0: true,
            available: true,
            ..Default::default()
        };
        // Write the SAME MV into both the above row (covers B1) and the
        // left column (covers A1).
        mf.write_block(16, 0, 16, 4, mvf); // above row (covers B1 at (31, -1)? No — B1 at (xCb + w - 1, yCb - 1) = (31, -1)... hmm needs y >= 0)
                                           // For B1 to be available, yCb must be > 0.
                                           // Use CU at (16, 16) size 16x16; B1 at (31, 15), A1 at (15, 31).
        mf.write_block(16, 12, 16, 4, mvf); // covers (31, 15)
        mf.write_block(12, 16, 4, 16, mvf); // covers (15, 31)
        let cands = derive_spatial_merge_candidates(16, 16, 16, 16, &mf, 2);
        assert!(cands[0].available, "B1 must be available");
        assert!(!cands[1].available, "A1 must be dropped — same MV as B1");
    }

    #[test]
    fn build_merge_list_pads_with_zero_mvs() {
        // No spatial candidates: list collapses to all zero-MV entries.
        let empty = [SpatialMergeCandidate::default(); 5];
        let list = build_merge_cand_list(&empty, 6);
        assert_eq!(list.len(), 6);
        for cand in &list {
            assert!(cand.pred_flag_l0);
            assert_eq!(cand.ref_idx_l0, 0);
            assert_eq!(cand.mv_l0, MotionVector::ZERO);
            assert!(cand.available);
        }
    }

    #[test]
    fn build_merge_list_inserts_spatials_in_walk_order() {
        let mut spatials = [SpatialMergeCandidate::default(); 5];
        // Make A1 available with a distinctive refIdx; the rest stay
        // unavailable.
        spatials[1] = SpatialMergeCandidate {
            available: true,
            field: MvField {
                mv_l0: MotionVector::from_int_pel(1, 1),
                ref_idx_l0: 0,
                pred_flag_l0: true,
                available: true,
                ..Default::default()
            },
        };
        let list = build_merge_cand_list(&spatials, 4);
        assert_eq!(list.len(), 4);
        // First entry is A1 (the only available spatial).
        assert_eq!(list[0].mv_l0.int_x(), 1);
        // Remaining three entries are zero-MV pads.
        for i in 1..4 {
            assert_eq!(list[i].mv_l0, MotionVector::ZERO);
        }
    }

    #[test]
    fn mc_copy_block_zero_mv_is_memcpy() {
        // 8x8 reference plane filled with a ramp; copy to identical
        // position in a fresh destination → exact match.
        let mut src = PicturePlane::filled(8, 8, 0);
        for y in 0..8 {
            for x in 0..8 {
                src.samples[y * 8 + x] = ((y * 8 + x) as u8).wrapping_add(1);
            }
        }
        let mut dst = PicturePlane::filled(8, 8, 0);
        mc_copy_block_int(&mut dst, 0, 0, 8, 8, &src, 0, 0).unwrap();
        assert_eq!(dst.samples, src.samples);
    }

    #[test]
    fn mc_copy_block_clamps_negative_source() {
        // Source position partly negative — clamped to plane edges.
        // Destination samples for negative-x columns should equal the
        // x=0 column of the source.
        let mut src = PicturePlane::filled(4, 4, 0);
        for x in 0..4 {
            for y in 0..4 {
                src.samples[y * 4 + x] = (x * 10 + y) as u8;
            }
        }
        let mut dst = PicturePlane::filled(4, 4, 0);
        mc_copy_block_int(&mut dst, 0, 0, 4, 4, &src, -2, 0).unwrap();
        // After offset -2: dst columns 0,1 read src column 0, dst
        // columns 2,3 read src columns 0,1.
        for y in 0..4 {
            assert_eq!(dst.samples[y * 4], src.samples[y * 4]); // dst col 0 → src col 0
            assert_eq!(dst.samples[y * 4 + 1], src.samples[y * 4]); // dst col 1 → src col 0 (clamped)
            assert_eq!(dst.samples[y * 4 + 2], src.samples[y * 4]); // dst col 2 → src col 0
            assert_eq!(dst.samples[y * 4 + 3], src.samples[y * 4 + 1]); // dst col 3 → src col 1
        }
    }
}
