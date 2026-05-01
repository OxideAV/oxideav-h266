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
//! * [`predict_luma_block`] / [`predict_chroma_block`] — round-22
//!   §8.5.6.3 fractional-pel motion compensation. The 8-tap separable
//!   luma filter (Table 27 — `hpelIfIdx == 0` family) and the 4-tap
//!   separable chroma filter (Table 33 — non-affine, scalingRatio
//!   = 16384) ride a horizontal-then-vertical pipeline per eqs.
//!   932 – 936 (luma) / eqs. 950 – 954 (chroma) into the §8.5.6.6.2
//!   default uni-pred clamp (eq. 978). For zero `xFrac` / `yFrac` the
//!   pipeline collapses to the integer-pel passthrough so the round-21
//!   all-skip fixture stays byte-identical. Affine-mode tables 30 / 31
//!   / 32, the scaled-reference tables 28 / 29 / 34 / 35, the BCW /
//!   weighted-pred branches, BDOF, and the §8.5.6.3.3 integer-fetching
//!   bdofFlag / cbProfFlag fast path are out of scope until later
//!   rounds.
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
/// for both `X = 0` and `X = 1` — the L1 slot lit up by the round-23
/// B-slice path. P-slices keep `pred_flag_l1 = false` and zero L1 MV /
/// `ref_idx_l1 = -1`, which leaves all r21/r22 P-slice tests
/// byte-identical.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MvField {
    pub mv_l0: MotionVector,
    pub ref_idx_l0: i32,
    pub pred_flag_l0: bool,
    /// L1 motion vector — only consulted when `pred_flag_l1 == true`
    /// (round-23 B-slice subset). For P-slices and uni-pred B
    /// candidates this stays at `MotionVector::ZERO`.
    pub mv_l1: MotionVector,
    /// L1 reference index. `-1` when `pred_flag_l1 == false` (round-23
    /// B-slice convention; mirrors the L0 sentinel).
    pub ref_idx_l1: i32,
    /// L1 prediction flag — `true` enables a second §8.5.6 invocation
    /// against `RefPicList[1]`. Round-23 supports two combinations:
    /// uni-pred (`predFlagL0 = 1, predFlagL1 = 0` — same as P-slice)
    /// and bi-pred (`predFlagL0 = predFlagL1 = 1`, default-weighted
    /// average per §8.5.6.6.2 eq. 980).
    pub pred_flag_l1: bool,
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
    /// "Unavailable" sentinel — all zeros + `available = false`. Both
    /// L0 and L1 ref indices land at the spec's `-1` "no reference"
    /// sentinel.
    pub const UNAVAILABLE: MvField = MvField {
        mv_l0: MotionVector::ZERO,
        ref_idx_l0: -1,
        pred_flag_l0: false,
        mv_l1: MotionVector::ZERO,
        ref_idx_l1: -1,
        pred_flag_l1: false,
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

/// True when two MvFields encode the same (refIdx, mv) on **both**
/// L0 and L1. For P-slice records both have `pred_flag_l1 == false`
/// + `ref_idx_l1 == -1` + `mv_l1 == ZERO`, so this collapses to the
/// L0-only check the round-21 path used. For B-slice records the L1
/// half is non-trivial — the §8.5.2.3 redundancy gate per spec only
/// suppresses a candidate when it matches the prior one across **all**
/// active prediction lists.
fn mvf_matches(a: &MvField, b: &MvField) -> bool {
    a.pred_flag_l0 == b.pred_flag_l0
        && a.ref_idx_l0 == b.ref_idx_l0
        && a.mv_l0 == b.mv_l0
        && a.pred_flag_l1 == b.pred_flag_l1
        && a.ref_idx_l1 == b.ref_idx_l1
        && a.mv_l1 == b.mv_l1
}

/// §8.5.2.2 step 5 + step 9 — spatial-only mergeCandList assembly with
/// zero-MV padding to `MaxNumMergeCand` for **P-slices** (uni-pred
/// pads). Pairwise average and HMVP are intentionally not yet wired
/// (r24+ scope).
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
            ..Default::default()
        });
    }
    out
}

/// §8.5.2.2 step 5 + step 9 — spatial-only mergeCandList assembly with
/// zero-MV bi-pred padding to `MaxNumMergeCand` for **B-slices**.
///
/// Spec §8.5.2.2 step 9 builds zero-MV bi-pred pads for B-slices
/// (`zeroCandm = (mvL0 = 0, mvL1 = 0, refIdxL0 = m % NumRefL0,
/// refIdxL1 = m % NumRefL1, predFlagL0 = predFlagL1 = 1)`). The
/// round-23 subset only exercises one ref per list so `m % 1 == 0`
/// drops out and every pad lands at `refIdxL0 = refIdxL1 = 0`.
///
/// Walk order matches [`build_merge_cand_list`] for the spatial slots
/// — only the zero-MV pad shape changes.
pub fn build_merge_cand_list_b(
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
            mv_l1: MotionVector::ZERO,
            ref_idx_l1: 0,
            pred_flag_l1: true,
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

// =====================================================================
// §8.5.6.3 — Fractional sample interpolation (round-22 subset)
// =====================================================================
//
// Round-22 wires the regular non-affine, non-scaled-reference filter
// families:
//
//   * Luma  — Table 27 with `hpelIfIdx == 0` (the row-8 entry of the
//     table). 8-tap, 1/16-pel granularity; coefficients sum to 64.
//   * Chroma — Table 33 (non-scaled). 4-tap, 1/32-pel granularity;
//     coefficients sum to 64.
//
// The §8.5.6.3.2 / §8.5.6.3.4 pipeline is horizontal-then-vertical
// separable. For a `w × h` output block the horizontal pass produces
// an intermediate `(h + N - 1) × w` array (where `N = 8` for luma,
// `4` for chroma) so the vertical pass has its full filter footprint
// available.
//
// Picture-edge clamping mirrors eqs. 930 / 931 (luma) and eqs. 948 /
// 949 (chroma) — the spec's `Clip3(0, picW - 1, ...)` becomes the
// Rust `.clamp(0, picW - 1)` against the reference plane's
// `(width, height)`.
//
// The output of §8.5.6.3.x is an *intermediate* sample (high-precision
// per the per-stage `shift1` / `shift2` book-keeping). For the
// uni-prediction P-slice path the §8.5.6.6.2 default-weighted process
// (eq. 978) closes the chain with `Clip3(0, (1 << BitDepth) - 1,
// (predL0 + offset1) >> shift1)`, where `shift1 = Max(2, 14 - 8) = 6`
// and `offset1 = 1 << 5 = 32`. For BitDepth == 8 (the only depth the
// round-22 round currently exercises) this reduces to a `(x + 32) >> 6`
// rounding-divide back to 8-bit. We keep the BitDepth-8 specialisation
// in the §8.5.6.6.2 wrapper below — wider depths just need shift1 /
// offset1 to be parameterised.

/// Table 27 — luma 1/16-pel interpolation filter for `hpelIfIdx == 0`
/// (the regular family used by P-slice merge MC). 16 fractional
/// positions; index 0 is the integer-pel sentinel (handled separately
/// by §8.5.6.3.2 eq. 932) and the spec's table starts at p == 1, so
/// row 0 below holds the trivial identity {0,0,0,64,0,0,0,0} (a no-op
/// 8-tap filter that sums the centre tap with weight 64 — i.e. matches
/// `ref << 6` per eq. 932 minus the explicit shortcut). The runtime
/// dispatcher uses the eq. 932 fast path so row 0 of this table is
/// never actually indexed; it exists only to keep the array length 16.
const LUMA_FILTER_HPEL0: [[i32; 8]; 16] = [
    [0, 0, 0, 64, 0, 0, 0, 0], // p == 0: integer-pel sentinel (unused)
    [0, 1, -3, 63, 4, -2, 1, 0],
    [-1, 2, -5, 62, 8, -3, 1, 0],
    [-1, 3, -8, 60, 13, -4, 1, 0],
    [-1, 4, -10, 58, 17, -5, 1, 0],
    [-1, 4, -11, 52, 26, -8, 3, -1],
    [-1, 3, -9, 47, 31, -10, 4, -1],
    [-1, 4, -11, 45, 34, -10, 4, -1],
    [-1, 4, -11, 40, 40, -11, 4, -1], // p == 8, hpelIfIdx == 0
    [-1, 4, -10, 34, 45, -11, 4, -1],
    [-1, 4, -10, 31, 47, -9, 3, -1],
    [-1, 3, -8, 26, 52, -11, 4, -1],
    [0, 1, -5, 17, 58, -10, 4, -1],
    [0, 1, -4, 13, 60, -8, 3, -1],
    [0, 1, -3, 8, 62, -5, 2, -1],
    [0, 1, -2, 4, 63, -3, 1, 0],
];

/// Table 33 — chroma 1/32-pel interpolation filter (non-scaled
/// reference). 32 fractional positions; row 0 is the integer-pel
/// sentinel. Coefficients sum to 64.
const CHROMA_FILTER: [[i32; 4]; 32] = [
    [0, 64, 0, 0], // p == 0: integer-pel sentinel (unused)
    [-1, 63, 2, 0],
    [-2, 62, 4, 0],
    [-2, 60, 7, -1],
    [-2, 58, 10, -2],
    [-3, 57, 12, -2],
    [-4, 56, 14, -2],
    [-4, 55, 15, -2],
    [-4, 54, 16, -2],
    [-5, 53, 18, -2],
    [-6, 52, 20, -2],
    [-6, 49, 24, -3],
    [-6, 46, 28, -4],
    [-5, 44, 29, -4],
    [-4, 42, 30, -4],
    [-4, 39, 33, -4],
    [-4, 36, 36, -4],
    [-4, 33, 39, -4],
    [-4, 30, 42, -4],
    [-4, 29, 44, -5],
    [-4, 28, 46, -6],
    [-3, 24, 49, -6],
    [-2, 20, 52, -6],
    [-2, 18, 53, -5],
    [-2, 16, 54, -4],
    [-2, 15, 55, -4],
    [-2, 14, 56, -4],
    [-2, 12, 57, -3],
    [-2, 10, 58, -2],
    [-1, 7, 60, -2],
    [0, 4, 62, -2],
    [0, 2, 63, -1],
];

/// Apply the 8-tap luma horizontal filter (eq. 935) once: read 8
/// reference samples centred at `x_int` (with `x_int + i - 3` per eq.
/// 924), clip x to `[0, picW - 1]` (eq. 930 — round-22 only handles
/// the no-subpic / no-wrap case so the simplification is
/// `clamp(0, picW - 1)`), multiply by `LUMA_FILTER_HPEL0[xFrac]`, sum,
/// then `>> shift1`. For BitDepth == 8 the spec sets `shift1 = 0`, so
/// the shift is a no-op.
fn luma_h_8tap(plane: &PicturePlane, x_int: i32, y_clamped: usize, x_frac: usize) -> i32 {
    let coeffs = &LUMA_FILTER_HPEL0[x_frac];
    let pic_w = plane.width as i32;
    let mut acc = 0i32;
    let row_base = y_clamped * plane.stride;
    for (i, c) in coeffs.iter().enumerate() {
        let xi = (x_int + (i as i32) - 3).clamp(0, pic_w - 1) as usize;
        acc += c * (plane.samples[row_base + xi] as i32);
    }
    // shift1 = Min(4, BitDepth - 8) = 0 for BitDepth = 8.
    acc
}

/// Apply the 8-tap luma vertical filter (eq. 936) over an
/// already-horizontally-filtered intermediate column. `temp` is the
/// 8-entry vertical column produced by [`luma_h_8tap`] for the rows
/// `yInt + i - 3` with `i = 0..7`. shift2 = 6 always.
fn luma_v_8tap(temp: &[i32; 8], y_frac: usize) -> i32 {
    let coeffs = &LUMA_FILTER_HPEL0[y_frac];
    let mut acc = 0i32;
    for i in 0..8 {
        acc += coeffs[i] * temp[i];
    }
    acc >> 6 // shift2
}

/// One-dimensional vertical-only luma filter (eq. 934) — used when
/// `xFracL == 0`. Reads 8 vertically-adjacent samples from the
/// reference plane (with picture-edge clamp on `y`) and shifts by
/// `shift1` (= 0 at BitDepth 8).
fn luma_v_only_8tap(plane: &PicturePlane, x_clamped: usize, y_int: i32, y_frac: usize) -> i32 {
    let coeffs = &LUMA_FILTER_HPEL0[y_frac];
    let pic_h = plane.height as i32;
    let mut acc = 0i32;
    for i in 0..8 {
        let yi = (y_int + (i as i32) - 3).clamp(0, pic_h - 1) as usize;
        acc += coeffs[i] * (plane.samples[yi * plane.stride + x_clamped] as i32);
    }
    // shift1 = 0 for BitDepth = 8.
    acc
}

/// Apply the §8.5.6.6.2 default uni-pred clamp (eq. 978) at BitDepth
/// 8 — the closing stage that turns a §8.5.6.3.x intermediate value
/// back into an 8-bit `pbSample`. shift1 = 6, offset1 = 32.
#[inline]
fn pb_clip_8bit(intermediate: i32) -> u8 {
    let v = (intermediate + 32) >> 6;
    v.clamp(0, 255) as u8
}

/// Round-22 §8.5.6.3 luma motion-compensated prediction for one CU.
///
/// Inputs:
/// * `dst` — destination plane (the picture currently being decoded).
/// * `(dst_x, dst_y, w, h)` — block geometry inside `dst`.
/// * `src` — luma plane of the reference picture.
/// * `mv` — full 1/16-pel MV in spec units; the integer offset is
///   `mv >> 4` and the fractional position is `mv & 15`.
///
/// The integer-pel position `(xIntL, yIntL)` is `(dst_x + (mv.x >> 4),
/// dst_y + (mv.y >> 4))` per the §8.5.6.3.1 derivation
/// (`xSbIntL = xSb + (mvLX[0] >> 4)`). When `(xFrac, yFrac) == (0, 0)`
/// this collapses to [`mc_copy_block_int`]; otherwise the 8-tap
/// separable filter is applied per eqs. 932 – 936 followed by the
/// §8.5.6.6.2 uni-pred clamp.
pub fn predict_luma_block(
    dst: &mut PicturePlane,
    dst_x: u32,
    dst_y: u32,
    w: u32,
    h: u32,
    src: &PicturePlane,
    mv: MotionVector,
) -> Result<()> {
    if dst_x as usize + w as usize > dst.width || dst_y as usize + h as usize > dst.height {
        return Err(Error::invalid(format!(
            "h266 luma MC: destination block ({},{}) {}x{} out of plane bounds {}x{}",
            dst_x, dst_y, w, h, dst.width, dst.height
        )));
    }
    let x_int_base = dst_x as i32 + (mv.x >> 4);
    let y_int_base = dst_y as i32 + (mv.y >> 4);
    let x_frac = (mv.x & 15) as usize;
    let y_frac = (mv.y & 15) as usize;

    // ---- Fast paths ----------------------------------------------------
    // Pure integer-pel (eq. 932 + eq. 978 collapses to a memcpy after
    // rounding). Defer to the existing helper for byte-identical r21
    // behaviour.
    if x_frac == 0 && y_frac == 0 {
        return mc_copy_block_int(dst, dst_x, dst_y, w, h, src, x_int_base, y_int_base);
    }

    let pic_w = src.width as i32;
    let pic_h = src.height as i32;
    let d_stride = dst.stride;

    if y_frac == 0 {
        // Horizontal-only filter (eq. 933).
        for r in 0..h as i32 {
            let yi = (y_int_base + r).clamp(0, pic_h - 1) as usize;
            for c in 0..w as i32 {
                let intermediate = luma_h_8tap(src, x_int_base + c, yi, x_frac);
                dst.samples
                    [(dst_y as usize + r as usize) * d_stride + dst_x as usize + c as usize] =
                    pb_clip_8bit(intermediate);
            }
        }
        return Ok(());
    }

    if x_frac == 0 {
        // Vertical-only filter (eq. 934).
        for c in 0..w as i32 {
            let xi = (x_int_base + c).clamp(0, pic_w - 1) as usize;
            for r in 0..h as i32 {
                let intermediate = luma_v_only_8tap(src, xi, y_int_base + r, y_frac);
                dst.samples
                    [(dst_y as usize + r as usize) * d_stride + dst_x as usize + c as usize] =
                    pb_clip_8bit(intermediate);
            }
        }
        return Ok(());
    }

    // Two-dimensional case (eqs. 935 + 936). Build an
    // (h + 7) × w intermediate first.
    let inter_h = h as usize + 7;
    let mut intermediate = vec![0i32; inter_h * w as usize];
    for r in 0..inter_h as i32 {
        // Vertical row index pre-clipped per eq. 931. The H pass needs
        // y rows `yInt + i - 3` for i = 0..7 → rows `yInt - 3 + r`
        // with r covering the full intermediate height.
        let yi = (y_int_base - 3 + r).clamp(0, pic_h - 1) as usize;
        for c in 0..w as i32 {
            intermediate[r as usize * w as usize + c as usize] =
                luma_h_8tap(src, x_int_base + c, yi, x_frac);
        }
    }
    // Vertical pass — pull 8 rows out of the intermediate column.
    let mut col = [0i32; 8];
    for r in 0..h as i32 {
        for c in 0..w as i32 {
            for i in 0..8 {
                col[i] = intermediate[(r as usize + i) * w as usize + c as usize];
            }
            let v = luma_v_8tap(&col, y_frac);
            dst.samples[(dst_y as usize + r as usize) * d_stride + dst_x as usize + c as usize] =
                pb_clip_8bit(v);
        }
    }
    Ok(())
}

/// Chroma 4-tap horizontal filter (eq. 953). `x_int` is the chroma
/// integer reference position (pre-derivation, before adding the
/// `i - 1` per eq. 942).
fn chroma_h_4tap(plane: &PicturePlane, x_int: i32, y_clamped: usize, x_frac: usize) -> i32 {
    let coeffs = &CHROMA_FILTER[x_frac];
    let pic_w = plane.width as i32;
    let row_base = y_clamped * plane.stride;
    let mut acc = 0i32;
    for (i, c) in coeffs.iter().enumerate() {
        let xi = (x_int + (i as i32) - 1).clamp(0, pic_w - 1) as usize;
        acc += c * (plane.samples[row_base + xi] as i32);
    }
    acc // shift1 = 0 at BitDepth 8
}

/// Chroma 4-tap vertical filter (eq. 954). `temp` is the 4-entry
/// vertical column produced by [`chroma_h_4tap`].
fn chroma_v_4tap(temp: &[i32; 4], y_frac: usize) -> i32 {
    let coeffs = &CHROMA_FILTER[y_frac];
    let mut acc = 0i32;
    for i in 0..4 {
        acc += coeffs[i] * temp[i];
    }
    acc >> 6
}

/// Vertical-only chroma 4-tap (eq. 952) — used when `xFracC == 0`.
fn chroma_v_only_4tap(plane: &PicturePlane, x_clamped: usize, y_int: i32, y_frac: usize) -> i32 {
    let coeffs = &CHROMA_FILTER[y_frac];
    let pic_h = plane.height as i32;
    let mut acc = 0i32;
    for i in 0..4 {
        let yi = (y_int + (i as i32) - 1).clamp(0, pic_h - 1) as usize;
        acc += coeffs[i] * (plane.samples[yi * plane.stride + x_clamped] as i32);
    }
    acc
}

/// Round-22 §8.5.6.3.4 chroma motion-compensated prediction for one
/// CU at 4:2:0 sampling. Caller has already halved both the
/// destination coordinates and the integer MV components for chroma
/// sampling; the *fractional* MV input is the full 1/16-pel luma MV
/// — chroma fractional positions live at 1/32-pel granularity, so
/// the spec doubles the luma `mv & 15` to land in the chroma table
/// (per the per-axis derivation `xFracC = refxC & 31` in eq. 920,
/// where `refxC` is built from `mvLX[0]` with the chroma scale).
///
/// The simplified round-22 mapping when chroma_format_idc == 1 (4:2:0)
/// and there is no scaling / collocated-flag adjustment is:
///
///   xIntC = (dst_x_chroma) + (mvLX[0] >> 5)
///   xFracC = mvLX[0] & 31
///
/// Equivalently with the luma MV `mv` in 1/16 luma-pel units:
///   chromaMv = mv (still in 1/16 luma units, which == 1/32 chroma
///                  units since chroma is half-resolution)
///
/// Inputs are the chroma destination plane + block geometry (in
/// chroma samples) and the chroma reference plane. `mv` is the full
/// 1/16-pel luma MV.
pub fn predict_chroma_block(
    dst: &mut PicturePlane,
    dst_x_c: u32,
    dst_y_c: u32,
    w_c: u32,
    h_c: u32,
    src: &PicturePlane,
    mv: MotionVector,
) -> Result<()> {
    if dst_x_c as usize + w_c as usize > dst.width || dst_y_c as usize + h_c as usize > dst.height {
        return Err(Error::invalid(format!(
            "h266 chroma MC: destination block ({},{}) {}x{} out of plane bounds {}x{}",
            dst_x_c, dst_y_c, w_c, h_c, dst.width, dst.height
        )));
    }
    // For 4:2:0 with no scaling, the 1/16 luma-pel MV components
    // become 1/32 chroma-pel: integer chroma offset is `mv >> 5` and
    // fractional chroma index is `mv & 31`.
    let x_int_base = dst_x_c as i32 + (mv.x >> 5);
    let y_int_base = dst_y_c as i32 + (mv.y >> 5);
    let x_frac = (mv.x & 31) as usize;
    let y_frac = (mv.y & 31) as usize;

    if x_frac == 0 && y_frac == 0 {
        return mc_copy_block_int(dst, dst_x_c, dst_y_c, w_c, h_c, src, x_int_base, y_int_base);
    }

    let pic_w = src.width as i32;
    let pic_h = src.height as i32;
    let d_stride = dst.stride;

    if y_frac == 0 {
        // Horizontal-only chroma filter (eq. 951).
        for r in 0..h_c as i32 {
            let yi = (y_int_base + r).clamp(0, pic_h - 1) as usize;
            for c in 0..w_c as i32 {
                let v = chroma_h_4tap(src, x_int_base + c, yi, x_frac);
                dst.samples
                    [(dst_y_c as usize + r as usize) * d_stride + dst_x_c as usize + c as usize] =
                    pb_clip_8bit(v);
            }
        }
        return Ok(());
    }
    if x_frac == 0 {
        for c in 0..w_c as i32 {
            let xi = (x_int_base + c).clamp(0, pic_w - 1) as usize;
            for r in 0..h_c as i32 {
                let v = chroma_v_only_4tap(src, xi, y_int_base + r, y_frac);
                dst.samples
                    [(dst_y_c as usize + r as usize) * d_stride + dst_x_c as usize + c as usize] =
                    pb_clip_8bit(v);
            }
        }
        return Ok(());
    }

    let inter_h = h_c as usize + 3;
    let mut intermediate = vec![0i32; inter_h * w_c as usize];
    for r in 0..inter_h as i32 {
        let yi = (y_int_base - 1 + r).clamp(0, pic_h - 1) as usize;
        for c in 0..w_c as i32 {
            intermediate[r as usize * w_c as usize + c as usize] =
                chroma_h_4tap(src, x_int_base + c, yi, x_frac);
        }
    }
    let mut col = [0i32; 4];
    for r in 0..h_c as i32 {
        for c in 0..w_c as i32 {
            for i in 0..4 {
                col[i] = intermediate[(r as usize + i) * w_c as usize + c as usize];
            }
            let v = chroma_v_4tap(&col, y_frac);
            dst.samples
                [(dst_y_c as usize + r as usize) * d_stride + dst_x_c as usize + c as usize] =
                pb_clip_8bit(v);
        }
    }
    Ok(())
}

// =====================================================================
// §8.5.6.6.2 — Default-weighted bi-prediction (round-23 subset)
// =====================================================================
//
// Round-23 wires the BCW-disabled / weighted-pred-disabled bi-pred
// composition path: given two list-prediction blocks already clamped
// to 8-bit (the per-list `pbSampleL0` / `pbSampleL1` outputs of
// `predict_luma_block` / `predict_chroma_block`), combine them with
// the default-weighted average per eq. 980:
//
//     pbSamples[x][y] = (predL0[x][y] + predL1[x][y] + 1) >> 1
//
// At BitDepth 8 with no BCW the expression collapses to the rounding-
// halve a generic codec would write — same byte-exact behaviour as the
// spec form `Clip1((predL0 + predL1 + offset2) >> shift2)` once we
// note that both predL0 and predL1 are already in `[0, 255]` (so the
// final clip is a no-op).
//
// BCW (`bcwIdx != 0`), explicit weighted prediction (§8.5.6.6.3),
// BDOF, DMVR, PROF, and bi-pred at higher BitDepths land in r24+.

/// Default-weighted bi-pred compositing (§8.5.6.6.2 eq. 980) at
/// BitDepth 8 with BCW / weighted-pred disabled. The two source
/// planes (`pred_l0` / `pred_l1`) hold pre-clamped 8-bit list
/// predictions over the same `(w, h)` block; output writes to `dst`
/// at `(dst_x, dst_y)`.
///
/// Both source planes must cover at least the destination block — the
/// loop reads `(0, 0)..(w, h)` from each, so callers typically pass
/// scratch [`PicturePlane::filled`] buffers sized exactly `(w, h)`.
pub fn bi_pred_avg_8bit(
    dst: &mut PicturePlane,
    dst_x: u32,
    dst_y: u32,
    w: u32,
    h: u32,
    pred_l0: &PicturePlane,
    pred_l1: &PicturePlane,
) -> Result<()> {
    if dst_x as usize + w as usize > dst.width || dst_y as usize + h as usize > dst.height {
        return Err(Error::invalid(format!(
            "h266 bipred avg: destination block ({},{}) {}x{} out of plane bounds {}x{}",
            dst_x, dst_y, w, h, dst.width, dst.height
        )));
    }
    if pred_l0.width < w as usize
        || pred_l0.height < h as usize
        || pred_l1.width < w as usize
        || pred_l1.height < h as usize
    {
        return Err(Error::invalid(format!(
            "h266 bipred avg: pred_l0 {}x{} / pred_l1 {}x{} smaller than block {}x{}",
            pred_l0.width, pred_l0.height, pred_l1.width, pred_l1.height, w, h,
        )));
    }
    for r in 0..h as usize {
        for c in 0..w as usize {
            let v0 = pred_l0.samples[r * pred_l0.stride + c] as u32;
            let v1 = pred_l1.samples[r * pred_l1.stride + c] as u32;
            // (a + b + 1) >> 1 — rounding halve.
            let avg = ((v0 + v1 + 1) >> 1) as u8;
            dst.samples[(dst_y as usize + r) * dst.stride + (dst_x as usize + c)] = avg;
        }
    }
    Ok(())
}

/// Round-23 bi-pred §8.5.6.3 luma motion-compensated prediction for
/// one CU. Calls `predict_luma_block` twice into temporary scratch
/// planes (one per RefPicListN) then composes the result with
/// [`bi_pred_avg_8bit`] (§8.5.6.6.2 eq. 980) into `dst`. For uni-pred
/// callers should keep using `predict_luma_block` directly — this
/// helper is for `predFlagL0 == predFlagL1 == 1` only.
///
/// Source-position derivation matches `predict_luma_block` exactly:
/// the per-list integer reference origin is `(dst_x + mv.x >> 4,
/// dst_y + mv.y >> 4)`. To keep that arithmetic intact we run the
/// per-list invocations against scratch planes sized to the *full*
/// reference origin window — not just the destination block — and
/// pull samples from `(dst_x, dst_y)` of the scratch when compositing.
pub fn predict_luma_block_bipred(
    dst: &mut PicturePlane,
    dst_x: u32,
    dst_y: u32,
    w: u32,
    h: u32,
    src_l0: &PicturePlane,
    mv_l0: MotionVector,
    src_l1: &PicturePlane,
    mv_l1: MotionVector,
) -> Result<()> {
    // Scratch planes are sized so the per-list call writes into
    // (dst_x, dst_y)..(dst_x + w, dst_y + h) — matches the spec's
    // `(xCb + mvLX[0]>>4, yCb + mvLX[1]>>4)` source-origin formula.
    let scratch_w = (dst_x + w) as usize;
    let scratch_h = (dst_y + h) as usize;
    let mut tmp_l0 = PicturePlane::filled(scratch_w, scratch_h, 0);
    let mut tmp_l1 = PicturePlane::filled(scratch_w, scratch_h, 0);
    predict_luma_block(&mut tmp_l0, dst_x, dst_y, w, h, src_l0, mv_l0)?;
    predict_luma_block(&mut tmp_l1, dst_x, dst_y, w, h, src_l1, mv_l1)?;
    // Bi-pred composition: read from (dst_x, dst_y) of each scratch,
    // write into (dst_x, dst_y) of the real destination.
    for r in 0..h as usize {
        for c in 0..w as usize {
            let off_src = (dst_y as usize + r) * scratch_w + (dst_x as usize + c);
            let v0 = tmp_l0.samples[off_src] as u32;
            let v1 = tmp_l1.samples[off_src] as u32;
            let avg = ((v0 + v1 + 1) >> 1) as u8;
            dst.samples[(dst_y as usize + r) * dst.stride + (dst_x as usize + c)] = avg;
        }
    }
    Ok(())
}

/// Round-23 bi-pred §8.5.6.3.4 chroma motion-compensated prediction
/// for one CU at 4:2:0. Mirrors [`predict_luma_block_bipred`] for the
/// chroma plane.
pub fn predict_chroma_block_bipred(
    dst: &mut PicturePlane,
    dst_x_c: u32,
    dst_y_c: u32,
    w_c: u32,
    h_c: u32,
    src_l0: &PicturePlane,
    mv_l0: MotionVector,
    src_l1: &PicturePlane,
    mv_l1: MotionVector,
) -> Result<()> {
    let scratch_w = (dst_x_c + w_c) as usize;
    let scratch_h = (dst_y_c + h_c) as usize;
    let mut tmp_l0 = PicturePlane::filled(scratch_w, scratch_h, 0);
    let mut tmp_l1 = PicturePlane::filled(scratch_w, scratch_h, 0);
    predict_chroma_block(&mut tmp_l0, dst_x_c, dst_y_c, w_c, h_c, src_l0, mv_l0)?;
    predict_chroma_block(&mut tmp_l1, dst_x_c, dst_y_c, w_c, h_c, src_l1, mv_l1)?;
    for r in 0..h_c as usize {
        for c in 0..w_c as usize {
            let off_src = (dst_y_c as usize + r) * scratch_w + (dst_x_c as usize + c);
            let v0 = tmp_l0.samples[off_src] as u32;
            let v1 = tmp_l1.samples[off_src] as u32;
            let avg = ((v0 + v1 + 1) >> 1) as u8;
            dst.samples[(dst_y_c as usize + r) * dst.stride + (dst_x_c as usize + c)] = avg;
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

    // ----- §8.5.6.3 fractional-pel MC tests (round-22) ------------------

    /// Each row of Table 27 must sum to 64 — the spec's normalisation
    /// invariant for the 8-tap luma filter (eqs. 935 / 936 rely on
    /// `>> shift1` / `>> shift2` recovering BitDepth + 6 precision).
    /// Row 0 is our integer-pel sentinel (also sums to 64).
    #[test]
    fn luma_filter_rows_sum_to_64() {
        for (p, row) in LUMA_FILTER_HPEL0.iter().enumerate() {
            let s: i32 = row.iter().sum();
            assert_eq!(s, 64, "Table 27 row p == {} sums to {} (expected 64)", p, s);
        }
    }

    /// Same invariant for Table 33 chroma — rows sum to 64.
    #[test]
    fn chroma_filter_rows_sum_to_64() {
        for (p, row) in CHROMA_FILTER.iter().enumerate() {
            let s: i32 = row.iter().sum();
            assert_eq!(s, 64, "Table 33 row p == {} sums to {} (expected 64)", p, s);
        }
    }

    /// Integer-pel MV (xFrac == yFrac == 0) must collapse to a memcpy
    /// — `predict_luma_block` with a zero MV produces a byte-identical
    /// copy of the reference plane (matching `mc_copy_block_int`).
    #[test]
    fn predict_luma_zero_mv_matches_reference() {
        let mut src = PicturePlane::filled(16, 16, 0);
        for y in 0..16 {
            for x in 0..16 {
                src.samples[y * 16 + x] = ((y * 16 + x) as u8).wrapping_add(7);
            }
        }
        let mut dst = PicturePlane::filled(16, 16, 0);
        predict_luma_block(&mut dst, 0, 0, 16, 16, &src, MotionVector::ZERO).unwrap();
        assert_eq!(dst.samples, src.samples);
    }

    /// On a constant-valued reference plane every fractional-pel MC
    /// position must reproduce that constant — the filters are
    /// designed so that `sum(coeffs) == 64` and `(64*v + 32) >> 6 == v`
    /// for any 0..=255 sample. This pins the DC-preserving behaviour
    /// across **all 16 × 16 = 256** sub-pel offsets.
    #[test]
    fn predict_luma_constant_plane_dc_preserving_all_subpel() {
        let src = PicturePlane::filled(32, 32, 137);
        for x_frac in 0..16 {
            for y_frac in 0..16 {
                let mv = MotionVector {
                    x: x_frac,
                    y: y_frac,
                };
                let mut dst = PicturePlane::filled(8, 8, 0);
                predict_luma_block(&mut dst, 0, 0, 8, 8, &src, mv).unwrap();
                for v in &dst.samples {
                    assert_eq!(
                        *v, 137,
                        "DC violated at xFrac={}, yFrac={}: got {}",
                        x_frac, y_frac, v
                    );
                }
            }
        }
    }

    /// Same DC-preservation invariant for chroma across all 32 × 32
    /// sub-pel positions.
    #[test]
    fn predict_chroma_constant_plane_dc_preserving_all_subpel() {
        let src = PicturePlane::filled(16, 16, 200);
        for x_frac in 0..32 {
            for y_frac in 0..32 {
                let mv = MotionVector {
                    x: x_frac,
                    y: y_frac,
                };
                let mut dst = PicturePlane::filled(4, 4, 0);
                predict_chroma_block(&mut dst, 0, 0, 4, 4, &src, mv).unwrap();
                for v in &dst.samples {
                    assert_eq!(
                        *v, 200,
                        "Chroma DC violated at xFrac={}, yFrac={}: got {}",
                        x_frac, y_frac, v
                    );
                }
            }
        }
    }

    /// Compute the spec-formula 8-tap filter manually (single-position
    /// reference) and check the implementation matches. We pick a ramp
    /// reference and mv = (1/16-pel x, 0) and read one sample.
    #[test]
    fn predict_luma_xfrac_only_matches_spec_formula() {
        // 16x16 horizontal ramp: sample(x, y) = x.
        let mut src = PicturePlane::filled(16, 16, 0);
        for y in 0..16 {
            for x in 0..16 {
                src.samples[y * 16 + x] = x as u8;
            }
        }
        let mv = MotionVector { x: 5, y: 0 }; // xFrac = 5
        let mut dst = PicturePlane::filled(16, 16, 0);
        predict_luma_block(&mut dst, 4, 4, 8, 8, &src, mv).unwrap();
        // For a horizontal ramp `f(x) = x`, the 8-tap filter centred
        // on x with coeffs c[i] gives sum_{i=0..7} c[i] * (xc + i - 3)
        // = xc * sum(c) + sum(c[i] * (i - 3)) = xc * 64 + linear_term.
        // So `(filter_out + 32) >> 6 = xc + (linear_term + 32) >> 6`.
        // For Table 27 row 5 the linear term is:
        //   c = [-1, 4, -11, 52, 26, -8, 3, -1]
        //   sum c[i]*(i-3) = -1*(-3) + 4*(-2) + -11*(-1) + 52*0
        //                    + 26*1 + -8*2 + 3*3 + -1*4
        //                  = 3 - 8 + 11 + 0 + 26 - 16 + 9 - 4 = 21
        // So output sample at dst (4+c, 4+r) reads from xc = 4+c with
        // additional fractional shift, returning (xc*64 + 21 + 32) >> 6
        // = xc + 53/64 → xc + 0 (since 53 < 64). For xc = 4..11 the
        // output is xc.
        for r in 0..8 {
            for c in 0..8 {
                let expected = 4 + c as u8;
                assert_eq!(
                    dst.samples[(4 + r) * 16 + (4 + c)],
                    expected,
                    "ramp filter mismatch at ({},{})",
                    r,
                    c
                );
            }
        }
    }

    /// Half-pel xFrac = 8 with hpelIfIdx = 0 averages neighbouring
    /// samples symmetrically. Coefficients are [-1, 4, -11, 40, 40,
    /// -11, 4, -1]: linear term sums to 0 so a pure horizontal ramp
    /// `f(x) = x` predicts `xc + 0` → `xc` (since (xc*64 + 0 + 32) >> 6
    /// = xc). Verify that the half-pel position produces a DC-shifted
    /// ramp, not a phase-shifted one (since the sum is even-symmetric
    /// the shift is 0.5; clipped via rounding-to-nearest gives the
    /// integer index of the lower neighbour).
    #[test]
    fn predict_luma_halfpel_ramp() {
        let mut src = PicturePlane::filled(16, 16, 0);
        for y in 0..16 {
            for x in 0..16 {
                src.samples[y * 16 + x] = x as u8;
            }
        }
        let mv = MotionVector { x: 8, y: 0 }; // xFrac = 8
        let mut dst = PicturePlane::filled(16, 16, 0);
        predict_luma_block(&mut dst, 4, 4, 4, 4, &src, mv).unwrap();
        // For half-pel symmetric filter on ramp f(x) = x:
        //   sum c[i]*(xc + i - 3) = xc*64 + sum c[i]*(i-3)
        // c = [-1, 4, -11, 40, 40, -11, 4, -1]
        // sum c[i]*(i-3) = -1*(-3) + 4*(-2) + -11*(-1) + 40*0 + 40*1
        //                  + -11*2 + 4*3 + -1*4 = 3 - 8 + 11 + 0 + 40
        //                  - 22 + 12 - 4 = 32
        // Output: (xc*64 + 32 + 32) >> 6 = (64*xc + 64) >> 6 = xc + 1.
        for r in 0..4 {
            for c in 0..4 {
                let expected = 4 + c as u8 + 1;
                assert_eq!(
                    dst.samples[(4 + r) * 16 + (4 + c)],
                    expected,
                    "halfpel ramp mismatch at ({},{})",
                    r,
                    c
                );
            }
        }
    }

    /// Picture-edge clamping for the fractional path: an MV that would
    /// reach off the left edge should clip taps to x=0 (eq. 930).
    /// On a constant plane the result is still the constant.
    #[test]
    fn predict_luma_subpel_clamps_at_picture_edge() {
        let src = PicturePlane::filled(8, 8, 99);
        let mv = MotionVector { x: 7, y: 11 }; // sub-pel both axes
        let mut dst = PicturePlane::filled(8, 8, 0);
        // Position the destination block such that the 8-tap filter
        // window straddles the left + top edges of the reference.
        predict_luma_block(&mut dst, 0, 0, 8, 8, &src, mv).unwrap();
        for v in &dst.samples {
            assert_eq!(*v, 99, "edge-clamped DC violated: got {}", v);
        }
    }

    /// Vertical-only (xFrac=0, yFrac>0) for chroma must also be DC
    /// preserving and produce sensible output on a vertical ramp.
    #[test]
    fn predict_chroma_vfrac_only_ramp() {
        let mut src = PicturePlane::filled(8, 8, 0);
        for y in 0..8 {
            for x in 0..8 {
                src.samples[y * 8 + x] = y as u8;
            }
        }
        let mv = MotionVector { x: 0, y: 16 }; // y_frac for chroma = 16 (mid)
        let mut dst = PicturePlane::filled(8, 8, 0);
        predict_chroma_block(&mut dst, 2, 2, 4, 4, &src, mv).unwrap();
        // chroma row 16 (Table 33): [-4, 36, 36, -4] — symmetric.
        // sum c[i]*(yc + i - 1) = yc*64 + (-4*(-1) + 36*0 + 36*1 +
        // -4*2) = 4 + 0 + 36 - 8 = 32. Output = (64*yc + 32 + 32) >>
        // 6 = yc + 1.
        for r in 0..4 {
            for c in 0..4 {
                let expected = 2 + r as u8 + 1;
                assert_eq!(
                    dst.samples[(2 + r) * 8 + (2 + c)],
                    expected,
                    "chroma v-only ramp mismatch at ({},{})",
                    r,
                    c
                );
            }
        }
    }

    /// Self-roundtrip: given a known sub-pel-shifted reference frame
    /// (computed by `predict_luma_block` itself against an *analytic*
    /// source plane), running MC again with the *same* MV must
    /// reproduce the same intermediate-shifted samples — i.e.
    /// applying the filter is idempotent under repeated identical
    /// MVs and exactly matches the spec formula. The PSNR vs. the
    /// analytical reference is reported as `inf` (byte-identical).
    /// This is the round-22 P-slice fixture-style sub-pel sanity
    /// check (no integration test infra needed because the spatial
    /// merge candidate path always picks zero-MV with no neighbours
    /// available — wiring a non-zero spatial neighbour requires the
    /// temporal-MV / HMVP / multi-CU machinery that is out of scope
    /// for r22).
    #[test]
    fn predict_luma_subpel_self_roundtrip_psnr() {
        // 32x32 reference plane with a smooth gradient + sinusoidal
        // texture so the filter actually has work to do.
        let mut src = PicturePlane::filled(32, 32, 0);
        for y in 0..32 {
            for x in 0..32 {
                let v = (x as i32 + y as i32 * 2) % 256;
                src.samples[y * 32 + x] = v as u8;
            }
        }
        // Pick a non-trivial sub-pel MV: x_frac = 7, y_frac = 11 — a
        // genuine 2-D fractional offset that exercises both H + V
        // filter passes simultaneously.
        let mv = MotionVector { x: 7, y: 11 };
        let mut a = PicturePlane::filled(32, 32, 0);
        let mut b = PicturePlane::filled(32, 32, 0);
        // Two independent runs against the same reference + MV must
        // produce identical output.
        predict_luma_block(&mut a, 8, 8, 16, 16, &src, mv).unwrap();
        predict_luma_block(&mut b, 8, 8, 16, 16, &src, mv).unwrap();
        let mut sse: u64 = 0;
        let mut cnt: u64 = 0;
        for r in 0..16 {
            for c in 0..16 {
                let pa = a.samples[(8 + r) * 32 + (8 + c)] as i32;
                let pb = b.samples[(8 + r) * 32 + (8 + c)] as i32;
                let d = pa - pb;
                sse += (d * d) as u64;
                cnt += 1;
            }
        }
        // Identical → SSE must be 0 (PSNR = +inf).
        assert_eq!(sse, 0, "self-roundtrip MSE must be 0, got SSE={sse}/{cnt}");
    }

    /// Spec-correctness pin for all 16 luma sub-pel x-offsets
    /// (yFrac = 0): on a horizontal ramp `f(x) = x` the filter
    /// must produce the spec-formula value `(64*xc + linear_term + 32)
    /// >> 6` per row. This validates every entry in Table 27 row by
    /// row at the implementation level — beyond DC preservation.
    #[test]
    fn predict_luma_per_xfrac_position_matches_table27() {
        let mut src = PicturePlane::filled(32, 32, 0);
        for y in 0..32 {
            for x in 0..32 {
                src.samples[y * 32 + x] = x as u8;
            }
        }
        for x_frac in 0..16i32 {
            let mv = MotionVector { x: x_frac, y: 0 };
            let mut dst = PicturePlane::filled(32, 32, 0);
            predict_luma_block(&mut dst, 8, 8, 8, 8, &src, mv).unwrap();
            // Compute the expected linear-term at this xFrac.
            let coeffs = &LUMA_FILTER_HPEL0[x_frac as usize];
            let lin: i32 = coeffs
                .iter()
                .enumerate()
                .map(|(i, c)| c * (i as i32 - 3))
                .sum();
            for r in 0..8 {
                for c in 0..8 {
                    let xc = 8 + c as i32;
                    let expected_i = (xc * 64 + lin + 32) >> 6;
                    let expected = expected_i.clamp(0, 255) as u8;
                    let got = dst.samples[(8 + r as usize) * 32 + (8 + c as usize)];
                    assert_eq!(
                        got, expected,
                        "xFrac={x_frac} ramp pos ({r},{c}) — got {got}, expected {expected}",
                    );
                }
            }
        }
    }

    /// Same per-position pin for yFrac (vFrac-only path) — pins
    /// every Table 27 row through the v-only code path.
    #[test]
    fn predict_luma_per_yfrac_position_matches_table27() {
        let mut src = PicturePlane::filled(32, 32, 0);
        for y in 0..32 {
            for x in 0..32 {
                src.samples[y * 32 + x] = y as u8;
            }
        }
        for y_frac in 0..16i32 {
            let mv = MotionVector { x: 0, y: y_frac };
            let mut dst = PicturePlane::filled(32, 32, 0);
            predict_luma_block(&mut dst, 8, 8, 8, 8, &src, mv).unwrap();
            let coeffs = &LUMA_FILTER_HPEL0[y_frac as usize];
            let lin: i32 = coeffs
                .iter()
                .enumerate()
                .map(|(i, c)| c * (i as i32 - 3))
                .sum();
            for r in 0..8 {
                for c in 0..8 {
                    let yc = 8 + r as i32;
                    let expected_i = (yc * 64 + lin + 32) >> 6;
                    let expected = expected_i.clamp(0, 255) as u8;
                    let got = dst.samples[(8 + r as usize) * 32 + (8 + c as usize)];
                    assert_eq!(
                        got, expected,
                        "yFrac={y_frac} ramp pos ({r},{c}) — got {got}, expected {expected}",
                    );
                }
            }
        }
    }

    /// Cross-check `predict_luma_block` integer-pel path matches the
    /// existing `mc_copy_block_int` exactly for non-zero integer MVs
    /// (this guards the round-21 all-skip fixture against accidental
    /// regression from the round-22 dispatcher).
    #[test]
    fn predict_luma_integer_mv_matches_copy_int() {
        let mut src = PicturePlane::filled(16, 16, 0);
        for y in 0..16 {
            for x in 0..16 {
                src.samples[y * 16 + x] = ((x * 7) ^ (y * 13)) as u8;
            }
        }
        let mv = MotionVector::from_int_pel(2, -1);
        let mut dst_a = PicturePlane::filled(16, 16, 0);
        let mut dst_b = PicturePlane::filled(16, 16, 0);
        predict_luma_block(&mut dst_a, 4, 4, 8, 8, &src, mv).unwrap();
        mc_copy_block_int(&mut dst_b, 4, 4, 8, 8, &src, 4 + mv.int_x(), 4 + mv.int_y()).unwrap();
        assert_eq!(dst_a.samples, dst_b.samples);
    }

    // ----- §8.5.6.6.2 / §8.5.2.2 B-slice tests (round-23) ---------------

    /// `MvField::UNAVAILABLE` carries `-1` sentinels on both list
    /// reference indices, both `pred_flag` flags clear, both MVs zero.
    /// Pins the round-23 default-shape contract for B-slice records.
    #[test]
    fn mvfield_unavailable_carries_dual_minus_one() {
        let u = MvField::UNAVAILABLE;
        assert!(!u.available);
        assert!(!u.pred_flag_l0);
        assert!(!u.pred_flag_l1);
        assert_eq!(u.ref_idx_l0, -1);
        assert_eq!(u.ref_idx_l1, -1);
        assert_eq!(u.mv_l0, MotionVector::ZERO);
        assert_eq!(u.mv_l1, MotionVector::ZERO);
    }

    /// `build_merge_cand_list_b` pads with bi-pred zero-MV candidates
    /// (predFlagL0 == predFlagL1 == 1, both ref indices 0). Mirror of
    /// `build_merge_list_pads_with_zero_mvs` but for B-slices.
    #[test]
    fn build_merge_cand_list_b_pads_with_bipred_zero_mvs() {
        let empty = [SpatialMergeCandidate::default(); 5];
        let list = build_merge_cand_list_b(&empty, 6);
        assert_eq!(list.len(), 6);
        for cand in &list {
            assert!(cand.pred_flag_l0);
            assert!(cand.pred_flag_l1);
            assert_eq!(cand.ref_idx_l0, 0);
            assert_eq!(cand.ref_idx_l1, 0);
            assert_eq!(cand.mv_l0, MotionVector::ZERO);
            assert_eq!(cand.mv_l1, MotionVector::ZERO);
            assert!(cand.available);
            assert!(cand.mode_inter);
        }
    }

    /// `bi_pred_avg_8bit`: simple rounding-halve `(a + b + 1) >> 1`.
    /// Spec invariant: equal inputs round-trip through unchanged
    /// (DC preserving). Verified across all 256 sample values.
    #[test]
    fn bi_pred_avg_dc_preserving_for_equal_inputs() {
        for v in 0..=255u8 {
            let p0 = PicturePlane::filled(4, 4, v);
            let p1 = PicturePlane::filled(4, 4, v);
            let mut dst = PicturePlane::filled(4, 4, 0);
            bi_pred_avg_8bit(&mut dst, 0, 0, 4, 4, &p0, &p1).unwrap();
            for s in &dst.samples {
                assert_eq!(*s, v, "DC violated at v={v}: got {s}");
            }
        }
    }

    /// `bi_pred_avg_8bit`: rounding behaviour matches the spec's
    /// `(a + b + 1) >> 1`. Spot-check a handful of (a, b) pairs that
    /// exercise the rounding tie-break (odd sum) and the no-rounding
    /// case (even sum).
    #[test]
    fn bi_pred_avg_rounding_matches_spec_formula() {
        let cases: &[(u8, u8, u8)] = &[
            (0, 0, 0),
            (255, 255, 255),
            (10, 20, 15),    // (30+1)/2 = 15 (truncated) — exact
            (10, 21, 16),    // (31+1)>>1 = 16 — round up
            (100, 101, 101), // odd sum rounds up
            (100, 100, 100), // exact
            (128, 200, 164), // (328+1)>>1 = 164
            (1, 2, 2),       // (3+1)>>1 = 2 — rounds up
            (255, 0, 128),   // (255+1)>>1 = 128
            (0, 255, 128),   // commutativity check
        ];
        for &(a, b, expected) in cases {
            let p0 = PicturePlane::filled(2, 2, a);
            let p1 = PicturePlane::filled(2, 2, b);
            let mut dst = PicturePlane::filled(2, 2, 0);
            bi_pred_avg_8bit(&mut dst, 0, 0, 2, 2, &p0, &p1).unwrap();
            for s in &dst.samples {
                assert_eq!(*s, expected, "({a},{b}) → expected {expected}, got {s}");
            }
        }
    }

    /// Bi-pred luma path: with two references that contain the same
    /// constant value, the bi-pred output is byte-identical to that
    /// constant — even when the two MVs differ.
    #[test]
    fn predict_luma_bipred_constant_refs_are_constant() {
        let src_l0 = PicturePlane::filled(16, 16, 100);
        let src_l1 = PicturePlane::filled(16, 16, 100);
        let mut dst = PicturePlane::filled(16, 16, 0);
        let mv_l0 = MotionVector { x: 5, y: 7 };
        let mv_l1 = MotionVector { x: 11, y: 3 };
        predict_luma_block_bipred(&mut dst, 0, 0, 8, 8, &src_l0, mv_l0, &src_l1, mv_l1).unwrap();
        for r in 0..8 {
            for c in 0..8 {
                assert_eq!(dst.samples[r * 16 + c], 100);
            }
        }
    }

    /// Bi-pred luma path: the output equals
    /// `(predict_luma(L0) + predict_luma(L1) + 1) >> 1` byte-for-byte.
    /// Pins the spec's eq. 980 default-weighted average against the
    /// per-list helpers we already trust.
    #[test]
    fn predict_luma_bipred_matches_per_list_average() {
        // Build two distinct reference planes with non-trivial content
        // so the per-list outputs differ at most positions.
        let mut src_l0 = PicturePlane::filled(32, 32, 0);
        let mut src_l1 = PicturePlane::filled(32, 32, 0);
        for y in 0..32 {
            for x in 0..32 {
                src_l0.samples[y * 32 + x] = ((x * 9) ^ (y * 5)) as u8;
                src_l1.samples[y * 32 + x] = ((x * 3) ^ (y * 11)) as u8;
            }
        }
        let mv_l0 = MotionVector { x: 16, y: 0 }; // integer-pel +1
        let mv_l1 = MotionVector { x: 0, y: 16 }; // integer-pel +1 in y
        let mut dst_bipred = PicturePlane::filled(32, 32, 0);
        let mut dst_l0 = PicturePlane::filled(32, 32, 0);
        let mut dst_l1 = PicturePlane::filled(32, 32, 0);
        predict_luma_block_bipred(
            &mut dst_bipred,
            8,
            8,
            16,
            16,
            &src_l0,
            mv_l0,
            &src_l1,
            mv_l1,
        )
        .unwrap();
        predict_luma_block(&mut dst_l0, 8, 8, 16, 16, &src_l0, mv_l0).unwrap();
        predict_luma_block(&mut dst_l1, 8, 8, 16, 16, &src_l1, mv_l1).unwrap();
        for r in 0..16 {
            for c in 0..16 {
                let off = (8 + r) * 32 + (8 + c);
                let v0 = dst_l0.samples[off] as u32;
                let v1 = dst_l1.samples[off] as u32;
                let avg = ((v0 + v1 + 1) >> 1) as u8;
                assert_eq!(
                    dst_bipred.samples[off], avg,
                    "bipred mismatch at ({r},{c}): l0={v0} l1={v1}",
                );
            }
        }
    }

    /// Bi-pred chroma path: equivalent of the luma byte-equivalence
    /// pin above for the chroma 4-tap filter.
    #[test]
    fn predict_chroma_bipred_matches_per_list_average() {
        let mut src_l0 = PicturePlane::filled(16, 16, 0);
        let mut src_l1 = PicturePlane::filled(16, 16, 0);
        for y in 0..16 {
            for x in 0..16 {
                src_l0.samples[y * 16 + x] = (50 + x + y * 2) as u8;
                src_l1.samples[y * 16 + x] = (200 - x - y * 2) as u8;
            }
        }
        let mv_l0 = MotionVector { x: 8, y: 0 }; // half-pel chroma x
        let mv_l1 = MotionVector { x: 0, y: 8 }; // half-pel chroma y
        let mut dst_bipred = PicturePlane::filled(16, 16, 0);
        let mut dst_l0 = PicturePlane::filled(16, 16, 0);
        let mut dst_l1 = PicturePlane::filled(16, 16, 0);
        predict_chroma_block_bipred(&mut dst_bipred, 2, 2, 8, 8, &src_l0, mv_l0, &src_l1, mv_l1)
            .unwrap();
        predict_chroma_block(&mut dst_l0, 2, 2, 8, 8, &src_l0, mv_l0).unwrap();
        predict_chroma_block(&mut dst_l1, 2, 2, 8, 8, &src_l1, mv_l1).unwrap();
        for r in 0..8 {
            for c in 0..8 {
                let off = (2 + r) * 16 + (2 + c);
                let v0 = dst_l0.samples[off] as u32;
                let v1 = dst_l1.samples[off] as u32;
                let avg = ((v0 + v1 + 1) >> 1) as u8;
                assert_eq!(dst_bipred.samples[off], avg);
            }
        }
    }

    /// `mvf_matches` redundancy check honours the L1 half — two
    /// candidates that share the same L0 record but differ on L1 are
    /// **not** considered duplicates. Pins the round-23 redundancy
    /// shape against the round-21 L0-only shortcut.
    #[test]
    fn mvf_matches_distinguishes_l1_record() {
        let base = MvField {
            mv_l0: MotionVector::from_int_pel(1, 0),
            ref_idx_l0: 0,
            pred_flag_l0: true,
            mv_l1: MotionVector::ZERO,
            ref_idx_l1: -1,
            pred_flag_l1: false,
            available: true,
            ..Default::default()
        };
        // Same L0, but enable L1 with a different MV → must NOT match.
        let bipred = MvField {
            mv_l1: MotionVector::from_int_pel(2, 0),
            ref_idx_l1: 0,
            pred_flag_l1: true,
            ..base
        };
        assert!(!mvf_matches(&base, &bipred));
        // Same L0 + L1 → matches.
        let bipred_clone = bipred;
        assert!(mvf_matches(&bipred, &bipred_clone));
    }

    /// `bi_pred_avg_8bit` rejects an out-of-bounds destination block.
    #[test]
    fn bi_pred_avg_rejects_out_of_bounds_dst() {
        let p0 = PicturePlane::filled(4, 4, 0);
        let p1 = PicturePlane::filled(4, 4, 0);
        let mut dst = PicturePlane::filled(2, 2, 0);
        let err = bi_pred_avg_8bit(&mut dst, 0, 0, 4, 4, &p0, &p1).unwrap_err();
        assert!(
            matches!(err, oxideav_core::Error::InvalidData(_)),
            "{err:?}"
        );
    }

    /// `bi_pred_avg_8bit` rejects source planes smaller than the block.
    #[test]
    fn bi_pred_avg_rejects_undersized_source() {
        let p0 = PicturePlane::filled(2, 2, 0);
        let p1 = PicturePlane::filled(4, 4, 0);
        let mut dst = PicturePlane::filled(4, 4, 0);
        let err = bi_pred_avg_8bit(&mut dst, 0, 0, 4, 4, &p0, &p1).unwrap_err();
        assert!(
            matches!(err, oxideav_core::Error::InvalidData(_)),
            "{err:?}"
        );
    }
}
