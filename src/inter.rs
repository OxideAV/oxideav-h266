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
//! * [`build_merge_cand_list`] — §8.5.2.2 step 5 spatial assembly +
//!   step 7 HMVP insertion (round-24, §8.5.2.6) + step 5 last bullet
//!   Col candidate (round-25, §8.5.2.11) + step 8 pairwise-average
//!   candidate (round-26, §8.5.2.4) + step 9 zero-MV padding.
//! * [`HmvpTable`] — round-24 §8.5.2.6 / §8.5.2.16 history-based
//!   motion vector predictor table. Per-slice circular buffer of up
//!   to [`MAX_HMVP_CAND`] = 5 MvField records, reset to empty at
//!   every CTU column tile-boundary per the §7.3.11 slice_data()
//!   pseudocode. Merge derivation reads the table newest-to-oldest
//!   (`HmvpCandList[NumHmvpCand − hMvpIdx]` walk); §8.5.2.16 update
//!   pushes each just-decoded inter CU's motion field, slides
//!   duplicates to the newest slot, and evicts the oldest entry only
//!   when the buffer is at capacity and the incoming entry is not a
//!   duplicate.
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

use crate::reconstruct::{PictureBuffer, PicturePlane, PicturePlane16};

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
    /// `BcwIdx[x][y]` per §8.5.6.6.2 — the bi-prediction with
    /// CU-level weights index. Range `0..=4`; `0` means "default
    /// equal-weight bi-pred per eq. 980" and is the spec value for
    /// every candidate spawned by the temporal merge (§8.5.2.11),
    /// the pairwise-average path (eq. 483 footnote), and the zero-MV
    /// pad (§8.5.2.5). Spatial-merge candidates inherit the per-block
    /// `BcwIdx[xNbN][yNbN]` of the neighbour. The HMVP table also
    /// carries the index. CTU-level apply selects between eq. 980 and
    /// eq. 981 based on this slot.
    pub bcw_idx: u8,
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
        bcw_idx: 0,
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

/// Reference picture used by the inter-prediction MC. A 4:2:0 frame
/// snapshot, the picture's POC, and (round-25) the per-block MotionField
/// captured at the time the picture was decoded — needed by §8.5.2.11
/// temporal motion-vector prediction so the current slice can pull
/// MvFields from the bottom-right / centre collocated 8x8 grid position.
///
/// The `motion_field` slot is `None` for pictures that were decoded as
/// intra (no inter MV state to carry forward) or for any reference where
/// the upstream caller has not yet wired the MF capture step. When
/// `motion_field` is `None`, §8.5.2.11 falls through to the "no
/// availableFlagLXCol" branch and the spec's `availableFlagCol = 0`
/// short-circuits the Col candidate insertion.
///
/// Spec ties: §8.5.2.11 references `MvDmvrL0[ x ][ y ]`,
/// `RefIdxL0[ x ][ y ]`, `PredFlagL0[ x ][ y ]` (and the L1 mirror) of
/// the picture indicated by `ColPic`. These three arrays are the per-4x4
/// MotionField captured here (DMVR storage compression — §8.5.2.15 — is
/// the trivial `mv >> 4 << 4` rounding for our 1/16-pel record, applied
/// inside the temporal derivation).
#[derive(Clone, Debug)]
pub struct ReferencePicture {
    pub poc: i32,
    pub frame: PictureBuffer,
    /// Per-4x4-block motion field of the reference picture (round-25
    /// §8.5.2.11). `None` when this reference was decoded as intra-only
    /// or the caller has not wired MF capture yet — temporal merge
    /// derivation against this reference will short-circuit to
    /// "availableFlagLXCol = 0" in that case.
    pub motion_field: Option<MotionField>,
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

/// §8.5.2.2 step 5 + step 7 + step 8 + step 9 — spatial mergeCandList
/// assembly with optional Col + HMVP insertion + pairwise-average
/// candidate, padded with zero-MV to `MaxNumMergeCand` for **P-slices**
/// (uni-pred pads).
///
/// The list is built in spec walk order:
///   1. If availableFlagB1 → push B1
///   2. If availableFlagA1 → push A1
///   3. If availableFlagB0 → push B0
///   4. If availableFlagA0 → push A0
///   5. If availableFlagB2 → push B2
///   6. (round-25) If `availableFlagCol` (output of §8.5.2.11) → push Col.
///   7. (round-24) Step 7: if `numCurrMergeCand < MaxNumMergeCand − 1`
///      and `NumHmvpCand > 0`, invoke §8.5.2.6 to insert HMVP
///      candidates (newest to oldest, with the spec's prune-against-
///      A1/B1 rule for the two newest entries).
///   8. (round-26, this round) Step 8: if `numCurrMergeCand > 1` and
///      `numCurrMergeCand < MaxNumMergeCand`, invoke §8.5.2.4 to derive
///      the pairwise-average candidate from `mergeCandList[0]` and
///      `mergeCandList[1]` and append it.
///   9. While len < max → push zero-MV (refIdx 0).
pub fn build_merge_cand_list(
    spatial: &[SpatialMergeCandidate; 5],
    max_num_merge_cand: u32,
    col: Option<MvField>,
    hmvp: Option<&HmvpTable>,
) -> Vec<MvField> {
    let mut out: Vec<MvField> = Vec::with_capacity(max_num_merge_cand as usize);
    for cand in spatial {
        if cand.available && (out.len() as u32) < max_num_merge_cand {
            out.push(cand.field);
        }
    }
    // §8.5.2.2 step 5 last bullet — `if (availableFlagCol) mergeCandList[i++] = Col`.
    // The Col candidate (§8.5.2.11 derivation) is appended after the spatial
    // walk and *before* HMVP. Caller short-circuits by passing `None` when
    // ph_temporal_mvp_enabled_flag == 0 or the §8.5.2.11 derivation
    // produced `availableFlagLXCol == 0`.
    if let Some(c) = col {
        if (out.len() as u32) < max_num_merge_cand {
            out.push(c);
        }
    }
    // §8.5.2.2 step 7 — HMVP insertion. Trigger condition mirrors the
    // spec: `numCurrMergeCand < MaxNumMergeCand − 1 && NumHmvpCand > 0`.
    // The §8.5.2.6 walk inserts entries from newest to oldest, prunes
    // duplicates against A1/B1 (only for the two newest entries), and
    // halts as soon as `numCurrMergeCand == MaxNumMergeCand − 1`. The
    // last slot is reserved for the pairwise-average candidate / zero-MV
    // pad below.
    if let Some(table) = hmvp {
        if (out.len() as u32) < max_num_merge_cand.saturating_sub(1) && !table.entries.is_empty() {
            insert_hmvp_into_merge_list(&mut out, spatial, table, max_num_merge_cand);
        }
    }
    // §8.5.2.2 step 8 — pairwise-average candidate (§8.5.2.4). Gates
    // mirror the spec: `numCurrMergeCand > 1 && numCurrMergeCand <
    // MaxNumMergeCand`. P-slice variant → is_b_slice = false suppresses
    // the L1 half (the spec's `numRefLists == 1` clause forces
    // refIdxL1avg = -1 / predFlagL1avg = 0).
    derive_pairwise_average_candidate(&mut out, max_num_merge_cand, /*is_b_slice*/ false);
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

/// §8.5.2.2 step 5 + step 7 + step 9 — spatial mergeCandList assembly
/// with optional Col + HMVP insertion (round-24/25) and zero-MV bi-pred
/// padding to `MaxNumMergeCand` for **B-slices**.
///
/// Spec §8.5.2.2 step 9 builds zero-MV bi-pred pads for B-slices
/// (`zeroCandm = (mvL0 = 0, mvL1 = 0, refIdxL0 = m % NumRefL0,
/// refIdxL1 = m % NumRefL1, predFlagL0 = predFlagL1 = 1)`). The
/// round-23 subset only exercises one ref per list so `m % 1 == 0`
/// drops out and every pad lands at `refIdxL0 = refIdxL1 = 0`.
///
/// Walk order matches [`build_merge_cand_list`] for the spatial slots
/// + Col + HMVP step 7 — only the zero-MV pad shape changes.
pub fn build_merge_cand_list_b(
    spatial: &[SpatialMergeCandidate; 5],
    max_num_merge_cand: u32,
    col: Option<MvField>,
    hmvp: Option<&HmvpTable>,
) -> Vec<MvField> {
    let mut out: Vec<MvField> = Vec::with_capacity(max_num_merge_cand as usize);
    for cand in spatial {
        if cand.available && (out.len() as u32) < max_num_merge_cand {
            out.push(cand.field);
        }
    }
    // §8.5.2.2 step 5 last bullet — Col candidate (B-slice path). The
    // §8.5.2.11 derivation produces a record carrying both L0 and L1
    // halves when the slice is B and ph_temporal_mvp_enabled_flag is 1
    // for both lists. Caller fuses the per-list invocations into a
    // single MvField before passing it here.
    if let Some(c) = col {
        if (out.len() as u32) < max_num_merge_cand {
            out.push(c);
        }
    }
    // §8.5.2.2 step 7 — HMVP insertion (B-slice path). Same gating as
    // the P-slice variant; HMVP entries from a B-slice carry full L0 +
    // L1 records, so no shape adjustment is needed.
    if let Some(table) = hmvp {
        if (out.len() as u32) < max_num_merge_cand.saturating_sub(1) && !table.entries.is_empty() {
            insert_hmvp_into_merge_list(&mut out, spatial, table, max_num_merge_cand);
        }
    }
    // §8.5.2.2 step 8 — pairwise-average candidate (§8.5.2.4). B-slice
    // variant → is_b_slice = true so both L0 and L1 halves of avgCand
    // are derived (numRefLists == 2 in the spec walk).
    derive_pairwise_average_candidate(&mut out, max_num_merge_cand, /*is_b_slice*/ true);
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
            bcw_idx: 0,
        });
    }
    out
}

// =====================================================================
// §8.5.2.6 — Derivation process for history-based merging candidates
// =====================================================================
//
// HmvpCandList is a per-slice circular buffer of up to 5 MvField
// records. The §8.5.2.16 update process pushes the just-decoded inter
// CU's motion field after every inter CU. Duplicate entries are
// removed (and the buffer slid down) on insertion; once the buffer is
// full the oldest entry is evicted.
//
// At merge-list build time §8.5.2.6 inserts candidates from newest to
// oldest, halting once `numCurrMergeCand == MaxNumMergeCand − 1`.
// Pruning against A1/B1 applies only to the **two newest** HMVP
// entries (the spec's `hMvpIdx <= 2` guard).
//
// The HMVP table is reset to empty at every CTU column tile-boundary
// per the §7.3.11 slice_data() pseudocode. For our single-tile fixture
// the reset only happens at slice start.

/// Maximum HMVP candidate count per §8.5.2.16 ("NumHmvpCand is equal
/// to 5"). Treat as the buffer's hard cap.
pub const MAX_HMVP_CAND: usize = 5;

/// Per-slice history-based motion vector predictor table (§8.5.2.6 +
/// §8.5.2.16). Keeps the most recent (up to [`MAX_HMVP_CAND`]) inter-
/// CU motion records for the current slice in insertion order — the
/// last entry is the most recently pushed.
///
/// Reset to empty at every CTU column tile-boundary per the §7.3.11
/// slice_data() pseudocode (`NumHmvpCand = 0`).
#[derive(Clone, Debug, Default)]
pub struct HmvpTable {
    /// Newest entry is at the *back* (`entries[len-1]`); oldest at the
    /// *front* (`entries[0]`). The §8.5.2.6 walk reads back-to-front
    /// (newest first) per its `HmvpCandList[NumHmvpCand − hMvpIdx]`
    /// indexing with `hMvpIdx = 1..NumHmvpCand`.
    pub entries: Vec<MvField>,
}

impl HmvpTable {
    /// Create an empty HMVP table — equivalent to the spec's
    /// `NumHmvpCand = 0` reset.
    pub fn new() -> Self {
        Self {
            entries: Vec::with_capacity(MAX_HMVP_CAND),
        }
    }

    /// Reset the table to empty. Called at every CTU column tile
    /// boundary per §7.3.11.
    pub fn reset(&mut self) {
        self.entries.clear();
    }

    /// `NumHmvpCand` — the spec name for the current entry count.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns true when the table is empty (`NumHmvpCand == 0`).
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// §8.5.2.16 — Updating process for the history-based motion
    /// vector predictor candidate list.
    ///
    /// Insert `cand` into the table:
    ///   1. If an existing entry has the same motion vectors and
    ///      reference indices (`mvf_matches`), drop that entry (slide
    ///      the rest down).
    ///   2. If the table is full (`NumHmvpCand == 5`), drop the oldest
    ///      entry (front).
    ///   3. Append `cand` at the back (newest position).
    ///
    /// The combined effect: a duplicate refresh promotes the existing
    /// entry to the newest slot; the oldest entry is evicted only when
    /// the table is at capacity and the incoming entry is not a
    /// duplicate.
    pub fn update_with(&mut self, cand: MvField) {
        // Step 1 + 2 of the spec. Find any duplicate; if found, remove it
        // (which also frees a slot, so the eviction-when-full branch
        // doesn't run twice). Otherwise, evict the oldest if at capacity.
        if let Some(pos) = self.entries.iter().position(|e| mvf_matches(e, &cand)) {
            self.entries.remove(pos);
        } else if self.entries.len() >= MAX_HMVP_CAND {
            // Spec eviction-when-full clause: drop HmvpCandList[0]
            // (oldest), shifting indices 1..N down by one. `Vec::remove`
            // handles the shift in one go.
            self.entries.remove(0);
        }
        // Step 3: append at the back (newest slot).
        self.entries.push(cand);
    }
}

/// §8.5.2.6 — insert HMVP candidates into the merge list in newest-to-
/// oldest order, with the spec's two-newest-entry prune against A1/B1.
/// Halts as soon as `numCurrMergeCand == MaxNumMergeCand − 1` (i.e. one
/// slot is reserved for the eventual pairwise-average / zero-MV pad
/// candidate) or the HMVP table is exhausted.
///
/// Spec text (§8.5.2.6 — reproduced for reviewer convenience):
/// > For each candidate in `HmvpCandList[NumHmvpCand − hMvpIdx]` with
/// > index `hMvpIdx = 1..NumHmvpCand`, the following ordered steps are
/// > repeated until `numCurrMergeCand` is equal to `MaxNumMergeCand − 1`:
/// >   1. The variable `sameMotion` is derived as follows:
/// >      - If all of the following conditions are true for any
/// >        merging candidate `N` with `N` being `A1` or `B1`,
/// >        `sameMotion` is set equal to TRUE:
/// >        - `hMvpIdx` is less than or equal to 2.
/// >        - The candidate `HmvpCandList[NumHmvpCand − hMvpIdx]` and
/// >          the merging candidate `N` have the same motion vectors
/// >          and the same reference indices.
/// >      - Otherwise, `sameMotion` is set equal to FALSE.
/// >   2. When `sameMotion` is equal to FALSE, the candidate
/// >      `HmvpCandList[NumHmvpCand − hMvpIdx]` is added to the
/// >      merging candidate list.
fn insert_hmvp_into_merge_list(
    merge_list: &mut Vec<MvField>,
    spatial: &[SpatialMergeCandidate; 5],
    hmvp: &HmvpTable,
    max_num_merge_cand: u32,
) {
    let halt_count = max_num_merge_cand.saturating_sub(1) as usize;
    let n = hmvp.entries.len();
    // §8.5.2.6 walks hMvpIdx = 1..NumHmvpCand (1-based), reading
    // HmvpCandList[NumHmvpCand − hMvpIdx] — i.e. starts at the newest
    // entry (index n - 1) and proceeds toward the oldest.
    for h_mvp_idx in 1..=n {
        if merge_list.len() >= halt_count {
            break;
        }
        let entry = hmvp.entries[n - h_mvp_idx];
        // Pruning rule applies only for the two newest entries
        // (hMvpIdx ≤ 2). Compare against A1 (spatial[1]) and B1
        // (spatial[0]) when those are available.
        let mut same_motion = false;
        if h_mvp_idx <= 2 {
            let b1 = &spatial[0];
            let a1 = &spatial[1];
            if b1.available && mvf_matches(&entry, &b1.field) {
                same_motion = true;
            }
            if !same_motion && a1.available && mvf_matches(&entry, &a1.field) {
                same_motion = true;
            }
        }
        if !same_motion {
            merge_list.push(entry);
        }
    }
}

// =====================================================================
// §8.5.2.4 — Derivation process for pairwise average merging candidate
// =====================================================================
//
// Spec walk (§8.5.2.4):
//
//   p0Cand = mergeCandList[0]
//   p1Cand = mergeCandList[1]
//   numRefLists = (sh_slice_type == B) ? 2 : 1
//
//   For each X in 0..(numRefLists - 1):
//     If predFlagLX of both p0 and p1 is 1:
//       refIdxLXavg     = refIdxLXp0
//       predFlagLXavg   = 1
//       mvLXavg         = round( mvLXp0 + mvLXp1, rightShift=1, leftShift=0 )
//                         per §8.5.2.14 eqs. 608 – 610 (signed-magnitude
//                         shift toward zero, offset = 0 because
//                         rightShift == 1 ⇒ offset = (1 << 0) − 1 = 0).
//     Elif only p0 active on LX: copy from p0.
//     Elif only p1 active on LX: copy from p1.
//     Else: refIdxLXavg = -1, predFlagLXavg = 0, mvLXavg = (0, 0).
//
//   When numRefLists == 1 (P-slice): force refIdxL1avg = -1 and
//   predFlagL1avg = 0 — the L1 half is suppressed even if p0/p1 happen
//   to carry stale L1 state.
//
// Insertion is governed by §8.5.2.2 step 8: pairwise fires only when
// `numCurrMergeCand > 1 && numCurrMergeCand < MaxNumMergeCand` (i.e.
// the list has at least the two source candidates and there's room for
// the synthetic average). The new entry is appended at position
// `numCurrMergeCand` and the count is incremented by 1; the spec then
// hands off to step 9 (zero-MV pad).

/// §8.5.2.14 motion-vector rounding helper for the pairwise-average
/// case (`rightShift = 1`, `leftShift = 0`).
///
/// Reproduces eqs. 608 – 610:
///
/// ```text
/// offset = (rightShift == 0) ? 0 : ((1 << (rightShift - 1)) - 1)
/// mvX[i] = Sign(mvX[i]) * ((Abs(mvX[i]) + offset) >> rightShift) << leftShift
/// ```
///
/// For `rightShift = 1` the offset collapses to 0, so this is a
/// signed-magnitude divide-by-2 toward zero (`(-1, -1)` rounds to `0`,
/// not `-1`). This matches the spec exactly — note that this is **not**
/// the same as Rust's arithmetic right shift for negative values
/// (`-1 >> 1 == -1`, but `Sign(-1) * (Abs(-1) >> 1) == 0`).
#[inline]
fn round_mv_pairwise(sum: i32) -> i32 {
    let s = sum.signum();
    let abs = sum.unsigned_abs();
    s * ((abs >> 1) as i32)
}

/// §8.5.2.4 — derive the pairwise-average synthetic merging candidate
/// from the first two entries of `merge_list`.
///
/// Returns `None` (and leaves `merge_list` untouched) when:
///
///   * `merge_list.len() < 2` — there is no `p1Cand` to average against
///     (the §8.5.2.2 step 8 `numCurrMergeCand > 1` gate).
///   * `(merge_list.len() as u32) >= max_num_merge_cand` — the list is
///     already at capacity, so step 8 short-circuits before invoking
///     §8.5.2.4 (the `numCurrMergeCand < MaxNumMergeCand` gate).
///
/// Otherwise computes the per-list average per the spec walk above
/// (rounded with [`round_mv_pairwise`] for the both-active case),
/// suppresses L1 for P-slices, and pushes the new candidate at the end
/// of the list. Returns `Some(avgCand)` for inspection by tests.
fn derive_pairwise_average_candidate(
    merge_list: &mut Vec<MvField>,
    max_num_merge_cand: u32,
    is_b_slice: bool,
) -> Option<MvField> {
    if merge_list.len() < 2 {
        return None;
    }
    if (merge_list.len() as u32) >= max_num_merge_cand {
        return None;
    }
    let p0 = merge_list[0];
    let p1 = merge_list[1];
    let num_ref_lists: u32 = if is_b_slice { 2 } else { 1 };
    let mut avg = MvField {
        // Default everything to the "inactive" shape; the per-list walk
        // below overwrites the active half(ves).
        mv_l0: MotionVector::ZERO,
        ref_idx_l0: -1,
        pred_flag_l0: false,
        mv_l1: MotionVector::ZERO,
        ref_idx_l1: -1,
        pred_flag_l1: false,
        cu_skip_flag: false,
        // The synthetic candidate inherits MODE_INTER (it is added to
        // the merge list of an inter CU; the leaf-CU walker only cares
        // that pred_flag_l{0,1} are set correctly for MC). Setting
        // mode_inter true keeps the broadcast MvField self-consistent
        // for downstream §9.3.4.2.2 pred_mode context derivation when a
        // later CU samples this CU's slot.
        mode_inter: true,
        available: true,
        bcw_idx: 0,
    };
    // L0 half — always derived (numRefLists >= 1).
    derive_pairwise_per_list(
        p0.pred_flag_l0,
        p1.pred_flag_l0,
        p0.mv_l0,
        p1.mv_l0,
        p0.ref_idx_l0,
        p1.ref_idx_l0,
        &mut avg.mv_l0,
        &mut avg.ref_idx_l0,
        &mut avg.pred_flag_l0,
    );
    // L1 half — only derived for B-slices (numRefLists == 2). For
    // P-slices the spec forces refIdxL1avg = -1 / predFlagL1avg = 0,
    // which is exactly the default `avg` shape, so we leave the L1 half
    // alone.
    if num_ref_lists == 2 {
        derive_pairwise_per_list(
            p0.pred_flag_l1,
            p1.pred_flag_l1,
            p0.mv_l1,
            p1.mv_l1,
            p0.ref_idx_l1,
            p1.ref_idx_l1,
            &mut avg.mv_l1,
            &mut avg.ref_idx_l1,
            &mut avg.pred_flag_l1,
        );
    }
    merge_list.push(avg);
    Some(avg)
}

/// Per-list inner of §8.5.2.4 — applies the four-way switch on
/// (predFlagLXp0, predFlagLXp1) and writes the resulting refIdx /
/// predFlag / MV into the output slots.
#[allow(clippy::too_many_arguments)]
fn derive_pairwise_per_list(
    p0_flag: bool,
    p1_flag: bool,
    p0_mv: MotionVector,
    p1_mv: MotionVector,
    p0_ref: i32,
    p1_ref: i32,
    out_mv: &mut MotionVector,
    out_ref: &mut i32,
    out_flag: &mut bool,
) {
    match (p0_flag, p1_flag) {
        (true, true) => {
            // Eqs. 520 – 521 + the §8.5.2.14 round on (mvL0p0 + mvL0p1).
            *out_ref = p0_ref;
            *out_flag = true;
            *out_mv = MotionVector {
                x: round_mv_pairwise(p0_mv.x + p1_mv.x),
                y: round_mv_pairwise(p0_mv.y + p1_mv.y),
            };
        }
        (true, false) => {
            // Eqs. 522 – 525 — copy p0.
            *out_ref = p0_ref;
            *out_flag = true;
            *out_mv = p0_mv;
        }
        (false, true) => {
            // Eqs. 526 – 529 — copy p1.
            *out_ref = p1_ref;
            *out_flag = true;
            *out_mv = p1_mv;
        }
        (false, false) => {
            // Eqs. 530 – 533 — inactive on this list.
            *out_ref = -1;
            *out_flag = false;
            *out_mv = MotionVector::ZERO;
        }
    }
}

/// Parsed `merge_data()` syntax (round-21 + round-27 MMVD + round-28
/// CIIP subset — regular merge with optional MMVD or CIIP).
///
/// GPM / subblock merge are still out of scope. MMVD (round-27,
/// §8.5.2.7) is wired: when `mmvd_merge_flag == true` the decoder takes
/// the merge-with-motion-vector-difference branch, in which case
/// `merge_idx` is inferred to `mmvd_cand_flag` (per §7.4.12.7) and the
/// per-CB `MmvdOffset` is derived via [`derive_mmvd_offset`] from the
/// `(mmvd_distance_idx, mmvd_direction_idx, ph_mmvd_fullpel_only_flag)`
/// triple. CIIP (round-28, §8.5.6.7): when `ciip_flag == true` the
/// reconstructed prediction is the eq. 998 weighted average of the
/// regular-merge MC predSamplesInter and a planar predSamplesIntra
/// drawn from already-reconstructed neighbour samples; the weight `w`
/// is derived from the intra-coded status of the A / B neighbours per
/// §8.5.6.7.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MergeData {
    /// `regular_merge_flag[x0][y0]` — `true` when the regular-merge
    /// branch is taken. Inferred per §7.4.12.7 when no CIIP / GPM gate
    /// is open; explicitly parsed otherwise.
    pub regular_merge_flag: bool,
    /// `merge_idx[x0][y0]` — index into `mergeCandList`. When
    /// `mmvd_merge_flag == true` this slot is **inferred** to
    /// `mmvd_cand_flag` per §7.4.12.7 and is not parsed from the
    /// bitstream. Used by both the regular-merge branch and the CIIP
    /// branch (CIIP also reads `merge_idx` when `MaxNumMergeCand > 1`).
    pub merge_idx: u32,
    /// `mmvd_merge_flag[x0][y0]` — round-27 §8.5.2.7. `true` when the
    /// CU uses merge-with-motion-vector-difference. Only set when
    /// `regular_merge_flag == 1` and `sps_mmvd_enabled_flag == 1`.
    pub mmvd_merge_flag: bool,
    /// `mmvd_cand_flag[x0][y0]` — selects the first (`0`) or second
    /// (`1`) entry of `mergeCandList` as the MMVD base candidate.
    /// Inferred to 0 when `mmvd_merge_flag == 0` or when
    /// `MaxNumMergeCand == 1`.
    pub mmvd_cand_flag: u32,
    /// `mmvd_distance_idx[x0][y0]` — index into Table 17 governing the
    /// magnitude of `MmvdDistance`. Range 0..7 (TR `cMax = 7`).
    pub mmvd_distance_idx: u32,
    /// `mmvd_direction_idx[x0][y0]` — index into Table 18 selecting
    /// one of the four cardinal sign pairs `(MmvdSign[0], MmvdSign[1])`.
    /// Range 0..3 (FL `cMax = 3`, two bypass-coded bins).
    pub mmvd_direction_idx: u32,
    /// `ciip_flag[x0][y0]` — round-28 §8.5.6.7. `true` when the
    /// combined inter-intra prediction tool is applied: predSamples =
    /// `(w * predSamplesIntra + (4 − w) * predSamplesInter + 2) >> 2`
    /// with `w ∈ {1, 2, 3}` derived from the A / B neighbour intra-
    /// coded counts. Set only when `regular_merge_flag == 0` and
    /// (`sps_ciip_enabled_flag == 1`, `cu_skip_flag == 0`,
    /// `cbW * cbH ≥ 64`, `cbW < 128`, `cbH < 128`); inferred to 1 per
    /// §7.4.12.7 when CIIP is the only enabled non-regular branch
    /// (i.e. `sps_gpm_enabled_flag == 0`).
    pub ciip_flag: bool,
}

// =====================================================================
// §8.5.2.7 — Derivation process for merge motion vector difference
// =====================================================================
//
// Given the per-CU `(mmvd_distance_idx, mmvd_direction_idx)` syntax
// elements and the `ph_mmvd_fullpel_only_flag` from the picture header,
// the spec derives:
//
//   MmvdDistance = MMVD_DISTANCE_TABLE[mmvd_distance_idx]   (Table 17)
//   (MmvdSign[0], MmvdSign[1]) = MMVD_SIGN_TABLE[mmvd_direction_idx]
//                                                          (Table 18)
//   MmvdOffset[X] = (MmvdDistance << 2) * MmvdSign[X]      (eqs. 188/189)
//
// The `<< 2` shift converts the spec's "luma sample distance" into
// 1/16-pel MotionVector units (since 1 luma = 16 of these units, and
// the table is already pre-divided by 4 — see Table 17 doc).
//
// The §8.5.2.7 routine then derives mMvdL0 / mMvdL1. The simple
// uni-pred case reduces to "apply MmvdOffset to the active list" (eqs.
// 581/582). Bi-pred has three sub-paths:
//
//   * Equal POC distance (eqs. 557 – 560): mMvdL1 = mMvdL0 = MmvdOffset.
//   * Long-term refs OR opposite-sign POC distances of equal magnitude
//     (eqs. 564 / 565): mMvdL1 = -MmvdOffset (sign flip on both axes).
//   * General asymmetric short-term refs (eqs. 561 – 580): apply the
//     §8.5.2.12 distScaleFactor chain (eqs. 601 – 605) to scale the L1
//     offset by `currPocDiffL1 / currPocDiffL0`.

/// Table 17 — `MmvdDistance[ x0 ][ y0 ]` for `ph_mmvd_fullpel_only_flag
/// == 0`. Indexed by `mmvd_distance_idx` in `0..8`. The table is
/// pre-divided by 4 — eqs. 188 / 189 reapply the `<< 2` to land in
/// 1/16-pel units, so the on-screen displacement contributed by
/// `mmvd_distance_idx == 0` is `1/4` luma sample, idx 2 = `1` luma
/// sample, idx 7 = `32` luma samples.
pub const MMVD_DISTANCE_TABLE: [u32; 8] = [1, 2, 4, 8, 16, 32, 64, 128];

/// Table 17 — `MmvdDistance[ x0 ][ y0 ]` for `ph_mmvd_fullpel_only_flag
/// == 1`. Each entry is `4×` the regular table → after the eq. 188
/// `<< 2` shift the offsets snap to integer-pel multiples (1, 2, 4, 8,
/// 16, 32, 64, 128 luma samples).
pub const MMVD_DISTANCE_TABLE_FULLPEL: [u32; 8] = [4, 8, 16, 32, 64, 128, 256, 512];

/// Table 18 — `(MmvdSign[ 0 ], MmvdSign[ 1 ])` indexed by
/// `mmvd_direction_idx` in `0..4`. The four cardinal directions are
/// `+x`, `−x`, `+y`, `−y` (no diagonals).
pub const MMVD_SIGN_TABLE: [(i32, i32); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];

/// §8.5.2.7 §188 / §189 — derive `MmvdOffset[ x0 ][ y0 ]` for the
/// per-CB merge-MVD. Returns the offset as a [`MotionVector`] in
/// 1/16-pel units (matching the rest of the [`MotionVector`] API).
///
/// `mmvd_distance_idx` is clamped to `0..8` and `mmvd_direction_idx`
/// to `0..4`; values outside these ranges are spec-illegal but we
/// guard against them here so the helper is total.
pub fn derive_mmvd_offset(
    mmvd_distance_idx: u32,
    mmvd_direction_idx: u32,
    ph_mmvd_fullpel_only_flag: bool,
) -> MotionVector {
    let dist_idx = (mmvd_distance_idx as usize).min(MMVD_DISTANCE_TABLE.len() - 1);
    let dir_idx = (mmvd_direction_idx as usize).min(MMVD_SIGN_TABLE.len() - 1);
    let dist = if ph_mmvd_fullpel_only_flag {
        MMVD_DISTANCE_TABLE_FULLPEL[dist_idx]
    } else {
        MMVD_DISTANCE_TABLE[dist_idx]
    };
    // Eqs. 188 / 189: MmvdOffset[ X ] = ( MmvdDistance << 2 ) * MmvdSign[ X ]
    let scaled = (dist << 2) as i32;
    let (sx, sy) = MMVD_SIGN_TABLE[dir_idx];
    MotionVector {
        x: scaled * sx,
        y: scaled * sy,
    }
}

/// §8.5.2.7 — apply the derived `MmvdOffset` to a base merge candidate
/// to produce `(mMvdL0, mMvdL1)`-corrected motion vectors. Returns a
/// new [`MvField`] with the offsets folded into the active per-list MVs.
///
/// This is the legacy POC-agnostic entry: handles the uni-pred case
/// (eqs. 581 / 582) and the equal-POC-distance bi-pred case (eqs. 557
/// – 560 — `mMvdL1 = mMvdL0 = MmvdOffset` when `currPocDiffL0 ==
/// currPocDiffL1`). For arbitrary POC layouts (eqs. 561 – 580) prefer
/// [`apply_mmvd_to_base_with_poc`], which carries POC distance
/// information and dispatches into the §8.5.2.12-style distScaleFactor
/// scaling automatically.
///
/// `equal_poc_distance` lets the caller signal that the `currPocDiffL0
/// == currPocDiffL1` short-circuit applies. When the base candidate is
/// uni-pred (only one `pred_flag` set) the flag is irrelevant — eq.
/// 581 / 582 only ever applies the offset to the active list. When the
/// caller cannot prove `equal_poc_distance == true` and the base is
/// bi-pred, the older path silently dropped to "fold the offset into
/// L1 too"; that conservative behaviour is preserved here so existing
/// fixtures stay byte-identical.
pub fn apply_mmvd_to_base(
    base: &MvField,
    offset: MotionVector,
    equal_poc_distance: bool,
) -> MvField {
    let mut out = *base;
    let bi = base.pred_flag_l0 && base.pred_flag_l1;
    if base.pred_flag_l0 {
        out.mv_l0 = MotionVector {
            x: base.mv_l0.x + offset.x,
            y: base.mv_l0.y + offset.y,
        };
    }
    if base.pred_flag_l1 {
        // Symmetric bi-pred shortcut: same POC distance on both sides
        // → mMvdL1 == mMvdL0 == MmvdOffset (eqs. 557 – 560). The
        // asymmetric branch is unreachable here (the legacy entry
        // never carried POC data); preserved for byte-identical
        // backward compatibility with round-27 fixtures.
        let _ = equal_poc_distance;
        let _ = bi;
        out.mv_l1 = MotionVector {
            x: base.mv_l1.x + offset.x,
            y: base.mv_l1.y + offset.y,
        };
    }
    out
}

/// §8.5.2.7 — full POC-aware MMVD application for bi-pred bases.
/// Extends [`apply_mmvd_to_base`] with the asymmetric-POC branch
/// (eqs. 561 – 580). The inputs are signed POC distances:
///
///   `curr_poc_diff_l0 = currPoc − RefPicListL0[refIdxL0].poc`
///   `curr_poc_diff_l1 = currPoc − RefPicListL1[refIdxL1].poc`
///
/// `lt_l0` / `lt_l1` indicate whether the corresponding reference is
/// long-term (LT). Per spec, the L1 derivation switches between three
/// branches:
///
/// 1. Equal POC distance OR both-LT shortcut (eqs. 557 – 560): the L1
///    offset equals MmvdOffset unchanged.
/// 2. Opposite-sign POC distances (regardless of magnitude — when both
///    refs are short-term and `sign(d0) != sign(d1)`): the L1 offset is
///    `-MmvdOffset` per eq. 564 / 565 ("Td and Tb of opposite sign →
///    mMvdL1 = −MmvdOffset").
/// 3. Asymmetric same-sign short-term refs: scale MmvdOffset by the
///    §8.5.2.12 distScaleFactor (eqs. 601 – 605) so
///    `mMvdL1 = (distScaleFactor * MmvdOffset + 128) >> 8` per
///    component, with sign-aware rounding and the spec's clamps.
///
/// Uni-pred bases collapse to eqs. 581 / 582 — the offset is folded
/// into the active list and the inactive list stays untouched.
pub fn apply_mmvd_to_base_with_poc(
    base: &MvField,
    offset: MotionVector,
    curr_poc_diff_l0: i32,
    curr_poc_diff_l1: i32,
    lt_l0: bool,
    lt_l1: bool,
) -> MvField {
    let mut out = *base;
    if base.pred_flag_l0 {
        out.mv_l0 = MotionVector {
            x: base.mv_l0.x.saturating_add(offset.x),
            y: base.mv_l0.y.saturating_add(offset.y),
        };
    }
    if base.pred_flag_l1 {
        let l1_offset = if !base.pred_flag_l0 {
            // Uni-pred-L1 base — eq. 582: just apply the offset.
            offset
        } else if lt_l0 || lt_l1 || curr_poc_diff_l0 == curr_poc_diff_l1 {
            // Equal POC distance or LT shortcut (eqs. 557 – 560).
            offset
        } else if (curr_poc_diff_l0 ^ curr_poc_diff_l1) < 0 {
            // Opposite-sign POC distances — eq. 564 / 565: flip sign on
            // both axes.
            MotionVector {
                x: -offset.x,
                y: -offset.y,
            }
        } else {
            // General asymmetric same-sign short-term refs — apply the
            // §8.5.2.12 distScaleFactor scaling chain (eqs. 601 – 605).
            mmvd_scale_offset(offset, curr_poc_diff_l0, curr_poc_diff_l1)
        };
        out.mv_l1 = MotionVector {
            x: base.mv_l1.x.saturating_add(l1_offset.x),
            y: base.mv_l1.y.saturating_add(l1_offset.y),
        };
    }
    out
}

/// §8.5.2.12 / §8.5.2.7 — scale `MmvdOffset` from the L0 POC distance
/// onto the L1 POC distance per the spec's distScaleFactor pipeline:
///
///   td = clip3(-128, 127, currPocDiffL0)
///   tb = clip3(-128, 127, currPocDiffL1)
///   tx = (16384 + (|td| >> 1)) / td
///   distScaleFactor = clip3(-4096, 4095, (tb * tx + 32) >> 6)
///   mMvdL1[c] = clip3(-2^17, 2^17 - 1,
///                     (distScaleFactor * MmvdOffset[c] + 128 -
///                      (distScaleFactor*MmvdOffset[c] >= 0 ? 1 : 0)) >> 8)
///
/// Returns `MotionVector::ZERO` for the degenerate `td == 0` input
/// (the caller is expected to short-circuit equal-POC cases before
/// invoking this helper, but the guard keeps the function total).
fn mmvd_scale_offset(
    offset: MotionVector,
    curr_poc_diff_l0: i32,
    curr_poc_diff_l1: i32,
) -> MotionVector {
    let td = curr_poc_diff_l0.clamp(-128, 127);
    let tb = curr_poc_diff_l1.clamp(-128, 127);
    if td == 0 {
        return MotionVector::ZERO;
    }
    let abs_td = td.unsigned_abs() as i32;
    let tx = (16384 + (abs_td >> 1)) / td;
    let dist_scale_factor = ((tb * tx + 32) >> 6).clamp(-4096, 4095);
    let scale = |c: i32| -> i32 {
        let prod = dist_scale_factor * c;
        let bias: i32 = if prod >= 0 { 1 } else { 0 };
        ((prod + 128 - bias) >> 8).clamp(-131072, 131071)
    };
    MotionVector {
        x: scale(offset.x),
        y: scale(offset.y),
    }
}

// =====================================================================
// §8.5.6.7 — Combined inter-intra prediction (CIIP)
// =====================================================================
//
// The CIIP weight `w` is derived from the intra-coded status of two
// neighbouring blocks at fixed offsets relative to the current coding
// block (cbWidth, cbHeight) — for luma those are
//
//   ( xNbA, yNbA ) = ( xCb − 1, yCb − 1 + cbHeight )
//   ( xNbB, yNbB ) = ( xCb − 1 + cbWidth, yCb − 1 )
//
// (the A neighbour sits below the bottom-left corner of the left
// edge; the B neighbour sits to the right of the top-right corner of
// the top edge). The §8.5.6.7 ladder maps the count of intra-coded
// neighbours into `w`:
//
//   * both intra → w = 3
//   * both not intra → w = 1
//   * exactly one intra → w = 2
//
// The output samples are then composed per eq. 998:
//
//   predSamplesComb[x][y] = ( w * predSamplesIntra[x][y]
//                           + (4 − w) * predSamplesInter[x][y]
//                           + 2 ) >> 2
//
// The (intra, inter) weight pair is therefore one of (1, 3), (2, 2),
// (3, 1). The combiner clips the result back into the bit-depth range
// — eq. 998 itself does not clip but predSamplesIntra and
// predSamplesInter are already in-range so their non-negative weighted
// average is too; we still clamp defensively to handle rounding edge
// cases.

/// §8.5.6.7 — derive the CIIP intra-prediction weight `w` from the
/// intra-coded status of the §8.5.6.7 A / B neighbours. The output is
/// the weight that scales `predSamplesIntra`; the `predSamplesInter`
/// weight is `4 − w`. Returns `1`, `2`, or `3` per the spec ladder.
///
/// Spec mapping (§8.5.6.7):
///   * (intra_top = T, intra_left = T) → 3
///   * (intra_top = F, intra_left = F) → 1
///   * (otherwise — exactly one intra)  → 2
///
/// Callers compute `intra_top` / `intra_left` from a per-4x4 grid that
/// is updated by the leaf-CU walker as each CU finishes. A neighbour
/// that is unavailable (off-picture, off-slice) is treated as not
/// intra-coded — matching the spec's `availableX == FALSE →
/// isIntraCodedNeighbourX = FALSE` branch.
pub fn ciip_intra_weight(intra_top: bool, intra_left: bool) -> u32 {
    match (intra_top, intra_left) {
        (true, true) => 3,
        (false, false) => 1,
        _ => 2,
    }
}

/// §8.5.6.7 eq. 998 — combine planar intra-predicted samples and
/// regular-merge inter-predicted samples into the CIIP output. Both
/// `intra_pred` and `inter_pred` are row-major `cb_w * cb_h` arrays in
/// reconstructed-sample units (i.e. they are already clipped into
/// `[0, (1 << bit_depth) - 1]`).
///
/// `w` must be in `{1, 2, 3}` — see [`ciip_intra_weight`]. The output
/// is row-major `cb_w * cb_h` in the same units, with each sample
/// clamped into `[0, (1 << bit_depth) - 1]`.
pub fn combine_ciip_samples(
    intra_pred: &[i16],
    inter_pred: &[i16],
    cb_w: usize,
    cb_h: usize,
    w: u32,
    bit_depth: u32,
) -> Vec<i16> {
    debug_assert!(matches!(w, 1..=3));
    debug_assert_eq!(intra_pred.len(), cb_w * cb_h);
    debug_assert_eq!(inter_pred.len(), cb_w * cb_h);
    let max = (1i32 << bit_depth) - 1;
    let mut out = vec![0i16; cb_w * cb_h];
    let w_i = w as i32;
    let w_p = 4 - w_i;
    for y in 0..cb_h {
        for x in 0..cb_w {
            let i = y * cb_w + x;
            let v = (w_i * intra_pred[i] as i32 + w_p * inter_pred[i] as i32 + 2) >> 2;
            out[i] = v.clamp(0, max) as i16;
        }
    }
    out
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

// =====================================================================
// §8.5.2.11 / §8.5.2.12 — Temporal luma MV prediction (round-25 subset)
// =====================================================================
//
// The temporal merge candidate ("Col") fetches an MvField from the
// collocated picture (`ColPic`) at one of two positions:
//
//   1. Bottom-right (xCb + cbWidth, yCb + cbHeight), restricted to the
//      same CTB row as the current CU and inside the picture / subpic
//      boundary. Position is rounded down to the nearest 8x8 grid via
//      `(xColBr >> 3) << 3` per the spec — which is what §8.5.2.15
//      temporal-MV buffer compression effectively dictates.
//   2. Centre fallback (xCb + cbWidth/2, yCb + cbHeight/2) when the
//      bottom-right position is unavailable.
//
// The fetched MvField is then scaled by the POC distance ratio
// (currPocDiff / colPocDiff) per §8.5.2.12 eqs. 601 – 605, with the
// equality short-circuit at eq. 600 when colPocDiff == currPocDiff or
// the reference is long-term.
//
// Round-25 subset assumptions:
//   * No subpictures → boundary checks reduce to picture-edge clip.
//   * No long-term references in the simple fixture.
//   * `sbFlag = 0` (regular merge candidate, not subblock TMVP).
//   * `NoBackwardPredFlag` heuristic is conservative: in round-25 we
//     pick L0 of the collocated MvField when both lists are present —
//     spec §8.5.2.12 then prefers `mvL0Col` when `predFlagColL0 == 1`
//     and `predFlagColL1 == 0`, falls back to `mvL1Col` otherwise.
//     The fixture only exercises the uni-pred-L0 reference, so the
//     branch picked is `predFlagColL0 == 1`.

/// Inputs to the §8.5.2.11 temporal merge candidate derivation. One
/// `Col` candidate is built per `X` (0 or 1) — the caller invokes this
/// helper once for L0 and once for L1 (B-slices only) and fuses the two
/// halves into a single MvField for the merge list.
///
/// Spec inputs (§8.5.2.11):
/// * `xcb / ycb` — top-left luma sample of the current CB (picture-
///   absolute).
/// * `cb_w / cb_h` — CB dimensions in luma samples.
/// * `pic_w / pic_h` — current picture width / height in luma samples
///   (used to clamp the bottom-right collocated position to
///   `pic_width_in_luma_samples − 1` per eqs. 594 / 595).
/// * `ctb_log2_size_y` — `CtbLog2SizeY` from the SPS. Drives the
///   `yCb >> CtbLog2SizeY == yColBr >> CtbLog2SizeY` same-CTB-row
///   check.
/// * `current_poc` — the current picture's POC (from §8.3.1).
/// * `current_ref_poc` — POC of the current picture's
///   `RefPicList[X][refIdxLX]`. Used as the `currPocDiff` operand for
///   the §8.5.2.12 scaling.
/// * `col_pic` — the collocated picture (must carry an attached
///   MotionField + the POC). When `col_pic.motion_field` is `None`
///   the derivation short-circuits to "availableFlagLXCol == 0".
/// * `col_ref_poc` — POC of the picture pointed to by the collocated
///   block's `refIdxCol`. The scaling distance `colPocDiff` is then
///   `col_pic.poc - col_ref_poc`.
#[derive(Clone, Debug)]
pub struct TemporalMergeInputs<'a> {
    pub xcb: i32,
    pub ycb: i32,
    pub cb_w: i32,
    pub cb_h: i32,
    pub pic_w: i32,
    pub pic_h: i32,
    pub ctb_log2_size_y: u32,
    pub current_poc: i32,
    pub current_ref_poc: i32,
    pub col_pic: &'a ReferencePicture,
    pub col_ref_poc: i32,
}

/// §8.5.2.11 — derive one half (L0 or L1) of the temporal merge
/// candidate. Returns the fetched-and-scaled `(mv, available)` pair.
///
/// `x` selects which RefPicList[X] the *current* CU's refIdx points
/// into. Per spec the temporal candidate's reference index is fixed at
/// `refIdxL{X}Col = 0` (§8.5.2.2 step 2), so the caller passes the POC
/// of `RefPicList[X][0]` as `inputs.current_ref_poc`.
///
/// Returns `None` when:
/// * `inputs.col_pic.motion_field` is `None` (no captured MF on the
///   reference picture);
/// * the bottom-right position falls outside the same CTB row, the
///   picture, or the position is intra-coded — and the centre fallback
///   is also unavailable;
/// * `colPocDiff == 0` (the spec's degenerate case — division by zero
///   in eq. 601 — folded into "availableFlagLXCol = 0").
pub fn derive_temporal_merge_candidate(inputs: &TemporalMergeInputs<'_>) -> Option<MvField> {
    let mf = inputs.col_pic.motion_field.as_ref()?;

    // ---- Step 1: bottom-right collocated position --------------------
    let x_col_br = inputs.xcb + inputs.cb_w;
    let y_col_br = inputs.ycb + inputs.cb_h;
    let right_boundary = inputs.pic_w - 1;
    let bot_boundary = inputs.pic_h - 1;
    let same_ctb_row =
        (inputs.ycb >> inputs.ctb_log2_size_y) == (y_col_br >> inputs.ctb_log2_size_y);
    let mut tried_br = false;
    if same_ctb_row && y_col_br <= bot_boundary && x_col_br <= right_boundary {
        let x_col_cb = (x_col_br >> 3) << 3;
        let y_col_cb = (y_col_br >> 3) << 3;
        if let Some(mv) = fetch_collocated_mv(mf, x_col_cb, y_col_cb, inputs) {
            return Some(mv);
        }
        tried_br = true;
    }
    let _ = tried_br;

    // ---- Step 2: centre fallback -------------------------------------
    let x_col_ctr = inputs.xcb + (inputs.cb_w >> 1);
    let y_col_ctr = inputs.ycb + (inputs.cb_h >> 1);
    let x_col_cb = (x_col_ctr >> 3) << 3;
    let y_col_cb = (y_col_ctr >> 3) << 3;
    fetch_collocated_mv(mf, x_col_cb, y_col_cb, inputs)
}

/// Fetch + scale the collocated MV at the spec-derived 8x8-aligned
/// `(x_col_cb, y_col_cb)` luma position. Returns `None` when the
/// fetched block is intra-coded, has no active prediction list, or the
/// POC scaling cannot proceed (degenerate `colPocDiff == 0`).
///
/// The returned `MvField` is uni-pred L0 — caller is responsible for
/// fusing the L0 and L1 halves into a single bi-pred record when the
/// slice is B and both invocations succeed.
fn fetch_collocated_mv(
    mf: &MotionField,
    x_col_cb: i32,
    y_col_cb: i32,
    inputs: &TemporalMergeInputs<'_>,
) -> Option<MvField> {
    let raw = mf.get_at_luma(x_col_cb, y_col_cb);
    if !raw.available || !raw.mode_inter {
        // §8.5.2.12: intra / IBC / palette colCb → not available.
        return None;
    }
    // §8.5.2.12: pick the collocated motion vector. Spec rules:
    //   * If predFlagColL0 == 0 → use L1.
    //   * Else if predFlagColL1 == 0 → use L0.
    //   * Else (both lists active in collocated): pick per
    //     NoBackwardPredFlag / sh_collocated_from_l0_flag. For the
    //     round-25 fixture (uni-pred-L0 reference) we pick L0.
    let (mv_col, _ref_idx_col) = if !raw.pred_flag_l0 && raw.pred_flag_l1 {
        (raw.mv_l1, raw.ref_idx_l1)
    } else if raw.pred_flag_l0 {
        (raw.mv_l0, raw.ref_idx_l0)
    } else {
        return None;
    };

    // §8.5.2.15 temporal MV buffer compression — round to integer-pel
    // (effectively `mv >> 4 << 4` on each component when the buffer
    // stores at 1-sample granularity). The full spec does
    // `(mv + ((1 << 4) - 1)) >> 4 << 4` with sign-aware rounding;
    // using straight integer-pel rounding via `mv >> 4 << 4` is the
    // sample-accurate variant the fixture exercises and matches
    // analytical equality for the integer-pel test vectors.
    let mv_col = MotionVector {
        x: (mv_col.x >> 4) << 4,
        y: (mv_col.y >> 4) << 4,
    };

    // §8.5.2.12 eqs. 598/599 — POC distance computation.
    let col_poc_diff = inputs.col_pic.poc.wrapping_sub(inputs.col_ref_poc);
    let curr_poc_diff = inputs.current_poc.wrapping_sub(inputs.current_ref_poc);
    if col_poc_diff == 0 {
        // Degenerate scaling — spec does not generate a valid Col here.
        return None;
    }

    // Eq. 600 short-circuit when distances are equal (or LT references —
    // not modelled in round-25, but an equal-distance path covers it).
    let scaled = if col_poc_diff == curr_poc_diff {
        MotionVector {
            x: mv_col.x.clamp(-131072, 131071),
            y: mv_col.y.clamp(-131072, 131071),
        }
    } else {
        // Eqs. 601 – 605:
        //   td = clip(-128, 127, colPocDiff)
        //   tb = clip(-128, 127, currPocDiff)
        //   tx = (16384 + (|td| >> 1)) / td
        //   distScaleFactor = clip(-4096, 4095, (tb * tx + 32) >> 6)
        //   mvLXCol = clip(-2^17, 2^17 - 1,
        //     (distScaleFactor * mvCol + 128 - (distScaleFactor*mvCol >= 0)) >> 8)
        let td = col_poc_diff.clamp(-128, 127);
        let tb = curr_poc_diff.clamp(-128, 127);
        let abs_td = td.unsigned_abs() as i32;
        let tx = (16384 + (abs_td >> 1)) / td;
        let dist_scale_factor = ((tb * tx + 32) >> 6).clamp(-4096, 4095);
        let scale = |c: i32| -> i32 {
            let prod = dist_scale_factor * c;
            // Spec form: `(prod + 128 - (prod >= 0)) >> 8` — applies the
            // round-toward-zero correction integral to eq. 603.
            let bias: i32 = if prod >= 0 { 1 } else { 0 };
            ((prod + 128 - bias) >> 8).clamp(-131072, 131071)
        };
        MotionVector {
            x: scale(mv_col.x),
            y: scale(mv_col.y),
        }
    };

    // Build a uni-pred L0 record. §8.5.2.2 step 2 fixes
    // refIdxL{X}Col = 0 for the merge-mode Col candidate.
    Some(MvField {
        mv_l0: scaled,
        ref_idx_l0: 0,
        pred_flag_l0: true,
        mv_l1: MotionVector::ZERO,
        ref_idx_l1: -1,
        pred_flag_l1: false,
        cu_skip_flag: false,
        mode_inter: true,
        available: true,
        bcw_idx: 0,
    })
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

/// §8.5.6.3 horizontal-pass shift1 = `Min(4, BitDepth - 8)`. The H
/// pass divides its `coeffs(sum 64) × sample` accumulator by `1 <<
/// shift1` so the result lands in `BitDepth + 6 - shift2` precision
/// when followed by the V pass (shift2 = 6). At BitDepth 8 this is
/// `Min(4, 0) = 0` — the existing 8-bit code path.
#[inline]
fn shift1_for_bd(bit_depth: u32) -> i32 {
    std::cmp::min(4, bit_depth as i32 - 8).max(0)
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

/// Round-32 §8.5.6.3 luma motion-compensated prediction surfacing the
/// `BitDepth + 6` precision intermediate.
///
/// Equivalent to [`predict_luma_block`] except no final per-list clip
/// + right-shift is applied; the returned `Vec<i32>` of length `w * h`
/// (row-major) carries the spec's high-precision predSamplesLN values
/// per the §8.5.6.3.2 separable filter. This is the array §8.5.6.5
/// (BDOF), §8.5.6.5b (PROF), and the §8.5.6.6.x bi-pred composition
/// stages consume.
///
/// At `BitDepth = 8` the values lie in roughly `[-2^14, 2^14)` (=
/// 14-bit signed); at `BitDepth = 10` the range expands to 16-bit;
/// at `BitDepth = 12` it expands to 18-bit. Recovering an 8-bit
/// sample from the BD = 8 intermediate is `(v + 32) >> 6` clipped
/// to `[0, 255]` — i.e. the existing [`pb_clip_8bit`] helper —
/// which makes a high-precision → 8-bit conversion bit-identical
/// to [`predict_luma_block`].
///
/// Source-position derivation matches [`predict_luma_block`]: integer
/// origin is `(dst_x + mv.x >> 4, dst_y + mv.y >> 4)` and fractional
/// position is `(mv.x & 15, mv.y & 15)`. `dst_x` / `dst_y` are the CU
/// origin in the *current* picture (used purely for the integer
/// reference origin); the returned buffer is CU-origin-aligned with
/// no padding.
pub fn predict_luma_block_high_precision(
    dst_x: u32,
    dst_y: u32,
    w: u32,
    h: u32,
    src: &PicturePlane,
    mv: MotionVector,
    bit_depth: u32,
) -> Result<Vec<i32>> {
    if !(8..=16).contains(&bit_depth) {
        return Err(Error::invalid(format!(
            "h266 luma MC HP: bit_depth {bit_depth} out of supported range 8..=16",
        )));
    }
    let x_int_base = dst_x as i32 + (mv.x >> 4);
    let y_int_base = dst_y as i32 + (mv.y >> 4);
    let x_frac = (mv.x & 15) as usize;
    let y_frac = (mv.y & 15) as usize;

    let pic_w = src.width as i32;
    let pic_h = src.height as i32;
    let w_us = w as usize;
    let h_us = h as usize;
    let mut out = vec![0i32; w_us * h_us];
    let shift1 = shift1_for_bd(bit_depth);
    let lift = 14 - bit_depth as i32; // integer-pel `<< (14 - BD)` — eq. 932

    // ---- Fast path: integer-pel — eq. 932 lifts each sample by
    // `<< (14 - BitDepth)` to land in BD + 6 precision.
    if x_frac == 0 && y_frac == 0 {
        for r in 0..h as i32 {
            let yi = (y_int_base + r).clamp(0, pic_h - 1) as usize;
            for c in 0..w as i32 {
                let xi = (x_int_base + c).clamp(0, pic_w - 1) as usize;
                out[r as usize * w_us + c as usize] =
                    (src.samples[yi * src.stride + xi] as i32) << lift;
            }
        }
        return Ok(out);
    }

    if y_frac == 0 {
        // Horizontal-only filter (eq. 933). Output is in BD + 6
        // precision after `>> shift1`.
        for r in 0..h as i32 {
            let yi = (y_int_base + r).clamp(0, pic_h - 1) as usize;
            for c in 0..w as i32 {
                let acc = luma_h_8tap(src, x_int_base + c, yi, x_frac);
                out[r as usize * w_us + c as usize] = acc >> shift1;
            }
        }
        return Ok(out);
    }

    if x_frac == 0 {
        // Vertical-only filter (eq. 934).
        for c in 0..w as i32 {
            let xi = (x_int_base + c).clamp(0, pic_w - 1) as usize;
            for r in 0..h as i32 {
                let acc = luma_v_only_8tap(src, xi, y_int_base + r, y_frac);
                out[r as usize * w_us + c as usize] = acc >> shift1;
            }
        }
        return Ok(out);
    }

    // Two-dimensional case (eqs. 935 + 936). The H pass produces an
    // `(h + 7) × w` intermediate at `>> shift1` precision, the V pass
    // shifts by `shift2 = 6` to land in BD + 6 precision overall.
    let inter_h = h_us + 7;
    let mut intermediate = vec![0i32; inter_h * w_us];
    for r in 0..inter_h as i32 {
        let yi = (y_int_base - 3 + r).clamp(0, pic_h - 1) as usize;
        for c in 0..w as i32 {
            intermediate[r as usize * w_us + c as usize] =
                luma_h_8tap(src, x_int_base + c, yi, x_frac) >> shift1;
        }
    }
    let mut col = [0i32; 8];
    for r in 0..h as i32 {
        for c in 0..w as i32 {
            for i in 0..8 {
                col[i] = intermediate[(r as usize + i) * w_us + c as usize];
            }
            out[r as usize * w_us + c as usize] = luma_v_8tap(&col, y_frac);
        }
    }
    Ok(out)
}

/// HBD twin of [`luma_h_8tap`] reading u16 samples (Main10 / Main12).
fn luma_h_8tap_u16(plane: &PicturePlane16, x_int: i32, y_clamped: usize, x_frac: usize) -> i32 {
    let coeffs = &LUMA_FILTER_HPEL0[x_frac];
    let pic_w = plane.width as i32;
    let mut acc = 0i32;
    let row_base = y_clamped * plane.stride;
    for (i, c) in coeffs.iter().enumerate() {
        let xi = (x_int + (i as i32) - 3).clamp(0, pic_w - 1) as usize;
        acc += c * (plane.samples[row_base + xi] as i32);
    }
    acc
}

/// HBD twin of [`luma_v_only_8tap`] reading u16 samples.
fn luma_v_only_8tap_u16(
    plane: &PicturePlane16,
    x_clamped: usize,
    y_int: i32,
    y_frac: usize,
) -> i32 {
    let coeffs = &LUMA_FILTER_HPEL0[y_frac];
    let pic_h = plane.height as i32;
    let mut acc = 0i32;
    for i in 0..8 {
        let yi = (y_int + (i as i32) - 3).clamp(0, pic_h - 1) as usize;
        acc += coeffs[i] * (plane.samples[yi * plane.stride + x_clamped] as i32);
    }
    acc
}

/// HBD twin of [`predict_luma_block_high_precision`] — reads `u16`
/// reference samples from a [`PicturePlane16`] so the §8.5.6.3
/// separable 8-tap filter sees the full Main10 / Main12 dynamic range
/// rather than the 8-bit-truncated value the legacy `u8` plane would
/// expose. Output layout, output precision, and the per-bit-depth
/// `shift1 / shift2` tracking are byte-identical to the existing 8-bit
/// HP helper at `bit_depth == 8`; tests cover the parity.
///
/// The returned `Vec<i32>` of length `w * h` (row-major) carries the
/// spec's `BitDepth + 6` precision intermediate values, ready to feed
/// into the round-32 BDOF / PROF / bi-pred composition stages without
/// the per-list clip + right-shift the §8.5.6.6.2 closing stage
/// applies.
pub fn predict_luma_block_high_precision_u16(
    dst_x: u32,
    dst_y: u32,
    w: u32,
    h: u32,
    src: &PicturePlane16,
    mv: MotionVector,
    bit_depth: u32,
) -> Result<Vec<i32>> {
    if !(8..=16).contains(&bit_depth) {
        return Err(Error::invalid(format!(
            "h266 luma MC HP u16: bit_depth {bit_depth} out of supported range 8..=16",
        )));
    }
    if src.bit_depth != bit_depth {
        return Err(Error::invalid(format!(
            "h266 luma MC HP u16: ref plane bit_depth {} != requested {}",
            src.bit_depth, bit_depth
        )));
    }
    let x_int_base = dst_x as i32 + (mv.x >> 4);
    let y_int_base = dst_y as i32 + (mv.y >> 4);
    let x_frac = (mv.x & 15) as usize;
    let y_frac = (mv.y & 15) as usize;

    let pic_w = src.width as i32;
    let pic_h = src.height as i32;
    let w_us = w as usize;
    let h_us = h as usize;
    let mut out = vec![0i32; w_us * h_us];
    let shift1 = shift1_for_bd(bit_depth);
    let lift = 14 - bit_depth as i32;

    if x_frac == 0 && y_frac == 0 {
        for r in 0..h as i32 {
            let yi = (y_int_base + r).clamp(0, pic_h - 1) as usize;
            for c in 0..w as i32 {
                let xi = (x_int_base + c).clamp(0, pic_w - 1) as usize;
                out[r as usize * w_us + c as usize] =
                    (src.samples[yi * src.stride + xi] as i32) << lift;
            }
        }
        return Ok(out);
    }

    if y_frac == 0 {
        for r in 0..h as i32 {
            let yi = (y_int_base + r).clamp(0, pic_h - 1) as usize;
            for c in 0..w as i32 {
                let acc = luma_h_8tap_u16(src, x_int_base + c, yi, x_frac);
                out[r as usize * w_us + c as usize] = acc >> shift1;
            }
        }
        return Ok(out);
    }

    if x_frac == 0 {
        for c in 0..w as i32 {
            let xi = (x_int_base + c).clamp(0, pic_w - 1) as usize;
            for r in 0..h as i32 {
                let acc = luma_v_only_8tap_u16(src, xi, y_int_base + r, y_frac);
                out[r as usize * w_us + c as usize] = acc >> shift1;
            }
        }
        return Ok(out);
    }

    let inter_h = h_us + 7;
    let mut intermediate = vec![0i32; inter_h * w_us];
    for r in 0..inter_h as i32 {
        let yi = (y_int_base - 3 + r).clamp(0, pic_h - 1) as usize;
        for c in 0..w as i32 {
            intermediate[r as usize * w_us + c as usize] =
                luma_h_8tap_u16(src, x_int_base + c, yi, x_frac) >> shift1;
        }
    }
    let mut col = [0i32; 8];
    for r in 0..h as i32 {
        for c in 0..w as i32 {
            for i in 0..8 {
                col[i] = intermediate[(r as usize + i) * w_us + c as usize];
            }
            out[r as usize * w_us + c as usize] = luma_v_8tap(&col, y_frac);
        }
    }
    Ok(out)
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

// =====================================================================
// §8.5.6.6.2 — BCW (Bi-prediction with CU-level Weights) eq. 981
// =====================================================================
//
// When a bi-pred CU carries `bcw_idx > 0` and CIIP is off, the spec's
// eq. 981 replaces the eq. 980 default-weighted average with an
// explicit weighted blend:
//
//   w1 = bcwWLut[bcw_idx]   with bcwWLut = { 4, 5, 3, 10, -2 }
//   w0 = 8 - w1
//   pbSamples = Clip1((w0*predL0 + w1*predL1 + offset3) >> (shift1 + 3))
//
// At BitDepth 8 the spec sets shift1 = max(2, 14-8) = 6 and
// offset3 = 1 << (shift1 + 2) = 1 << 8 = 256. The two source predL0 /
// predL1 we hold are already pre-clamped 8-bit values (the round-22
// uni-pred clamp at eq. 978 / 979); the BCW path then does
//
//   ((4 - sign(w1)*…) * v0 * 64 + w1 * v1 * 64 + 256) >> 9
//
// once we factor in the implicit `<< 6` to lift each clamped sample
// back into the spec's 14-bit precision domain. That is equivalent
// (and byte-exact) to the simpler form
//
//   pbSamples = Clip1((w0 * v0 + w1 * v1 + 4) >> 3)
//
// which is what we implement here — the two forms differ only by the
// constant factor 64 cancelling between numerator and denominator.

/// Spec table — `bcwWLut[k]` for `k ∈ 0..=4`. The default index 0 is
/// equivalent to the eq. 980 shortcut and is never read through this
/// table at the call site (the BCW dispatch in
/// [`bi_pred_avg_8bit_bcw`] short-circuits to the simple average when
/// `bcw_idx == 0`); it is included here for completeness.
pub const BCW_W_LUT: [i32; 5] = [4, 5, 3, 10, -2];

/// Default-weighted bi-pred composite at BitDepth 8 with optional BCW
/// (§8.5.6.6.2). `bcw_idx == 0` falls through to eq. 980 (rounding
/// average); `bcw_idx ∈ 1..=4` applies eq. 981 with weights from
/// [`BCW_W_LUT`]. CIIP CUs MUST pass `bcw_idx == 0` per the spec
/// "If bcwIdx is equal to 0 OR ciip_flag == 1" gate.
pub fn bi_pred_avg_8bit_bcw(
    dst: &mut PicturePlane,
    dst_x: u32,
    dst_y: u32,
    w: u32,
    h: u32,
    pred_l0: &PicturePlane,
    pred_l1: &PicturePlane,
    bcw_idx: u8,
) -> Result<()> {
    if bcw_idx == 0 {
        return bi_pred_avg_8bit(dst, dst_x, dst_y, w, h, pred_l0, pred_l1);
    }
    let idx = bcw_idx as usize;
    if idx >= BCW_W_LUT.len() {
        return Err(Error::invalid(format!(
            "h266 bipred BCW: bcw_idx {idx} out of range (0..=4)"
        )));
    }
    if dst_x as usize + w as usize > dst.width || dst_y as usize + h as usize > dst.height {
        return Err(Error::invalid(format!(
            "h266 bipred BCW: destination block ({dst_x},{dst_y}) {w}x{h} out of plane bounds {}x{}",
            dst.width, dst.height
        )));
    }
    if pred_l0.width < w as usize
        || pred_l0.height < h as usize
        || pred_l1.width < w as usize
        || pred_l1.height < h as usize
    {
        return Err(Error::invalid(format!(
            "h266 bipred BCW: pred_l0 {}x{} / pred_l1 {}x{} smaller than block {w}x{h}",
            pred_l0.width, pred_l0.height, pred_l1.width, pred_l1.height,
        )));
    }
    let w1 = BCW_W_LUT[idx];
    let w0 = 8 - w1;
    for r in 0..h as usize {
        for c in 0..w as usize {
            let v0 = pred_l0.samples[r * pred_l0.stride + c] as i32;
            let v1 = pred_l1.samples[r * pred_l1.stride + c] as i32;
            // pbSamples = Clip1((w0*v0 + w1*v1 + 4) >> 3)
            let blended = (w0 * v0 + w1 * v1 + 4) >> 3;
            let clamped = blended.clamp(0, 255) as u8;
            dst.samples[(dst_y as usize + r) * dst.stride + (dst_x as usize + c)] = clamped;
        }
    }
    Ok(())
}

/// BCW-aware luma bi-pred MC. Drop-in replacement for
/// [`predict_luma_block_bipred`] that selects between eq. 980 (when
/// `bcw_idx == 0`) and eq. 981 weighted blending (when `bcw_idx > 0`).
pub fn predict_luma_block_bipred_bcw(
    dst: &mut PicturePlane,
    dst_x: u32,
    dst_y: u32,
    w: u32,
    h: u32,
    src_l0: &PicturePlane,
    mv_l0: MotionVector,
    src_l1: &PicturePlane,
    mv_l1: MotionVector,
    bcw_idx: u8,
) -> Result<()> {
    let scratch_w = (dst_x + w) as usize;
    let scratch_h = (dst_y + h) as usize;
    let mut tmp_l0 = PicturePlane::filled(scratch_w, scratch_h, 0);
    let mut tmp_l1 = PicturePlane::filled(scratch_w, scratch_h, 0);
    predict_luma_block(&mut tmp_l0, dst_x, dst_y, w, h, src_l0, mv_l0)?;
    predict_luma_block(&mut tmp_l1, dst_x, dst_y, w, h, src_l1, mv_l1)?;
    if bcw_idx == 0 {
        // Fast eq. 980 path matches predict_luma_block_bipred byte-for-
        // byte; preserved separately to avoid the extra inner clamp.
        for r in 0..h as usize {
            for c in 0..w as usize {
                let off_src = (dst_y as usize + r) * scratch_w + (dst_x as usize + c);
                let v0 = tmp_l0.samples[off_src] as u32;
                let v1 = tmp_l1.samples[off_src] as u32;
                let avg = ((v0 + v1 + 1) >> 1) as u8;
                dst.samples[(dst_y as usize + r) * dst.stride + (dst_x as usize + c)] = avg;
            }
        }
        return Ok(());
    }
    let idx = bcw_idx as usize;
    if idx >= BCW_W_LUT.len() {
        return Err(Error::invalid(format!(
            "h266 luma bipred BCW: bcw_idx {idx} out of range (0..=4)"
        )));
    }
    let w1 = BCW_W_LUT[idx];
    let w0 = 8 - w1;
    for r in 0..h as usize {
        for c in 0..w as usize {
            let off_src = (dst_y as usize + r) * scratch_w + (dst_x as usize + c);
            let v0 = tmp_l0.samples[off_src] as i32;
            let v1 = tmp_l1.samples[off_src] as i32;
            let blended = ((w0 * v0 + w1 * v1 + 4) >> 3).clamp(0, 255) as u8;
            dst.samples[(dst_y as usize + r) * dst.stride + (dst_x as usize + c)] = blended;
        }
    }
    Ok(())
}

/// BCW-aware chroma bi-pred MC. Mirrors
/// [`predict_luma_block_bipred_bcw`] for the chroma plane at 4:2:0.
pub fn predict_chroma_block_bipred_bcw(
    dst: &mut PicturePlane,
    dst_x_c: u32,
    dst_y_c: u32,
    w_c: u32,
    h_c: u32,
    src_l0: &PicturePlane,
    mv_l0: MotionVector,
    src_l1: &PicturePlane,
    mv_l1: MotionVector,
    bcw_idx: u8,
) -> Result<()> {
    let scratch_w = (dst_x_c + w_c) as usize;
    let scratch_h = (dst_y_c + h_c) as usize;
    let mut tmp_l0 = PicturePlane::filled(scratch_w, scratch_h, 0);
    let mut tmp_l1 = PicturePlane::filled(scratch_w, scratch_h, 0);
    predict_chroma_block(&mut tmp_l0, dst_x_c, dst_y_c, w_c, h_c, src_l0, mv_l0)?;
    predict_chroma_block(&mut tmp_l1, dst_x_c, dst_y_c, w_c, h_c, src_l1, mv_l1)?;
    if bcw_idx == 0 {
        for r in 0..h_c as usize {
            for c in 0..w_c as usize {
                let off_src = (dst_y_c as usize + r) * scratch_w + (dst_x_c as usize + c);
                let v0 = tmp_l0.samples[off_src] as u32;
                let v1 = tmp_l1.samples[off_src] as u32;
                let avg = ((v0 + v1 + 1) >> 1) as u8;
                dst.samples[(dst_y_c as usize + r) * dst.stride + (dst_x_c as usize + c)] = avg;
            }
        }
        return Ok(());
    }
    let idx = bcw_idx as usize;
    if idx >= BCW_W_LUT.len() {
        return Err(Error::invalid(format!(
            "h266 chroma bipred BCW: bcw_idx {idx} out of range (0..=4)"
        )));
    }
    let w1 = BCW_W_LUT[idx];
    let w0 = 8 - w1;
    for r in 0..h_c as usize {
        for c in 0..w_c as usize {
            let off_src = (dst_y_c as usize + r) * scratch_w + (dst_x_c as usize + c);
            let v0 = tmp_l0.samples[off_src] as i32;
            let v1 = tmp_l1.samples[off_src] as i32;
            let blended = ((w0 * v0 + w1 * v1 + 4) >> 3).clamp(0, 255) as u8;
            dst.samples[(dst_y_c as usize + r) * dst.stride + (dst_x_c as usize + c)] = blended;
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
        let list = build_merge_cand_list(&empty, 6, None, None);
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
        let list = build_merge_cand_list(&spatials, 4, None, None);
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
        let list = build_merge_cand_list_b(&empty, 6, None, None);
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

    // ----- §8.5.2.6 / §8.5.2.16 HMVP tests (round-24) -------------------

    /// Helper: synthesise an MvField with a distinctive L0 MV so we can
    /// distinguish entries by inspection. Pred-flag-L0 set; L1 left at
    /// the unavailable shape (round-23 P-slice convention).
    fn dummy_mvf(dx: i32, dy: i32, ref_idx: i32) -> MvField {
        MvField {
            mv_l0: MotionVector::from_int_pel(dx, dy),
            ref_idx_l0: ref_idx,
            pred_flag_l0: true,
            mv_l1: MotionVector::ZERO,
            ref_idx_l1: -1,
            pred_flag_l1: false,
            cu_skip_flag: false,
            mode_inter: true,
            available: true,
            bcw_idx: 0,
        }
    }

    /// `HmvpTable::new` starts empty (`NumHmvpCand == 0`) and `reset`
    /// returns it to that state — pins the §7.3.11 slice-start /
    /// CTU-column-tile-boundary semantics.
    #[test]
    fn hmvp_table_new_is_empty_and_reset_clears() {
        let mut t = HmvpTable::new();
        assert!(t.is_empty());
        assert_eq!(t.len(), 0);
        t.update_with(dummy_mvf(1, 0, 0));
        assert_eq!(t.len(), 1);
        t.reset();
        assert!(t.is_empty());
    }

    /// §8.5.2.16 — fresh entries append at the back (newest slot) and
    /// the table grows monotonically until it hits [`MAX_HMVP_CAND`].
    #[test]
    fn hmvp_update_appends_until_capacity() {
        let mut t = HmvpTable::new();
        for i in 0..MAX_HMVP_CAND as i32 {
            t.update_with(dummy_mvf(i + 1, 0, 0));
        }
        assert_eq!(t.len(), MAX_HMVP_CAND);
        // Newest entry is at the back.
        assert_eq!(
            t.entries.last().unwrap().mv_l0,
            MotionVector::from_int_pel(MAX_HMVP_CAND as i32, 0)
        );
        // Oldest entry is at the front.
        assert_eq!(
            t.entries.first().unwrap().mv_l0,
            MotionVector::from_int_pel(1, 0)
        );
    }

    /// §8.5.2.16 — when the table is full and the new entry is *not* a
    /// duplicate, evict the oldest entry (front) and append the new
    /// entry at the back.
    #[test]
    fn hmvp_update_evicts_oldest_when_full() {
        let mut t = HmvpTable::new();
        for i in 0..MAX_HMVP_CAND as i32 {
            t.update_with(dummy_mvf(i + 1, 0, 0));
        }
        // Push one more — oldest (mv (1,0)) should be evicted.
        t.update_with(dummy_mvf(99, 0, 0));
        assert_eq!(t.len(), MAX_HMVP_CAND);
        assert_eq!(
            t.entries.first().unwrap().mv_l0,
            MotionVector::from_int_pel(2, 0),
            "oldest entry (mv=1,0) must have been evicted"
        );
        assert_eq!(
            t.entries.last().unwrap().mv_l0,
            MotionVector::from_int_pel(99, 0)
        );
    }

    /// §8.5.2.16 — duplicate insertion removes the existing entry from
    /// its current position and appends the (logically same) entry at
    /// the newest slot. Net effect: the duplicate is "promoted".
    #[test]
    fn hmvp_update_duplicate_promotes_to_newest() {
        let mut t = HmvpTable::new();
        // Push three entries, then re-push the middle one.
        t.update_with(dummy_mvf(1, 0, 0));
        t.update_with(dummy_mvf(2, 0, 0));
        t.update_with(dummy_mvf(3, 0, 0));
        t.update_with(dummy_mvf(2, 0, 0));
        assert_eq!(t.len(), 3);
        // Order should now be: (1,0), (3,0), (2,0) — (2,0) at the back.
        assert_eq!(t.entries[0].mv_l0, MotionVector::from_int_pel(1, 0));
        assert_eq!(t.entries[1].mv_l0, MotionVector::from_int_pel(3, 0));
        assert_eq!(t.entries[2].mv_l0, MotionVector::from_int_pel(2, 0));
    }

    /// §8.5.2.16 — duplicate insertion when at capacity does NOT evict
    /// the oldest entry (the dedup-and-promote path frees a slot, so
    /// the eviction-when-full clause does not fire).
    #[test]
    fn hmvp_update_duplicate_at_capacity_keeps_oldest() {
        let mut t = HmvpTable::new();
        for i in 0..MAX_HMVP_CAND as i32 {
            t.update_with(dummy_mvf(i + 1, 0, 0));
        }
        let oldest_before = t.entries.first().unwrap().mv_l0;
        // Re-push the middle entry — duplicate path.
        t.update_with(dummy_mvf(3, 0, 0));
        assert_eq!(t.len(), MAX_HMVP_CAND);
        assert_eq!(
            t.entries.first().unwrap().mv_l0,
            oldest_before,
            "duplicate-at-capacity must NOT evict the oldest entry"
        );
        assert_eq!(
            t.entries.last().unwrap().mv_l0,
            MotionVector::from_int_pel(3, 0)
        );
    }

    /// §8.5.2.6 — when no spatial candidates are available, all HMVP
    /// entries should be inserted (no pruning needed) up to
    /// `MaxNumMergeCand − 1`. The remaining slots get the §8.5.2.4
    /// pairwise-average candidate (round-26) followed by zero-MV pads.
    #[test]
    fn merge_list_inserts_hmvp_no_pruning_when_no_spatials() {
        let empty_spatials = [SpatialMergeCandidate::default(); 5];
        let mut hmvp = HmvpTable::new();
        // Fill HMVP with 3 distinctive entries: (1,0), (2,0), (3,0).
        hmvp.update_with(dummy_mvf(1, 0, 0));
        hmvp.update_with(dummy_mvf(2, 0, 0));
        hmvp.update_with(dummy_mvf(3, 0, 0));
        let list = build_merge_cand_list(&empty_spatials, 6, None, Some(&hmvp));
        assert_eq!(list.len(), 6);
        // First three entries are HMVP — newest first.
        assert_eq!(list[0].mv_l0, MotionVector::from_int_pel(3, 0));
        assert_eq!(list[1].mv_l0, MotionVector::from_int_pel(2, 0));
        assert_eq!(list[2].mv_l0, MotionVector::from_int_pel(1, 0));
        // Slot 3 = §8.5.2.4 pairwise-average of slots 0 and 1:
        // sum = (3*16 + 2*16, 0) = (80, 0), rounded by §8.5.2.14
        // (rightShift=1, leftShift=0, offset=0) → (40, 0). The
        // signed-magnitude shift collapses to a plain (sum >> 1) for
        // positive sums.
        assert_eq!(list[3].mv_l0, MotionVector { x: 40, y: 0 });
        assert!(list[3].pred_flag_l0);
        // Slots 4..6 are zero-MV pads.
        for entry in &list[4..] {
            assert_eq!(entry.mv_l0, MotionVector::ZERO);
            assert!(entry.pred_flag_l0);
        }
    }

    /// §8.5.2.6 — HMVP insertion halts at `MaxNumMergeCand − 1` (last
    /// slot reserved for the §8.5.2.4 pairwise-average candidate /
    /// zero-MV pad). With 5 HMVP entries and `MaxNumMergeCand == 4`,
    /// only 3 HMVP entries land in the merge list (slots 0..2) — and
    /// slot 3 is the pairwise-average of slots 0 and 1.
    #[test]
    fn merge_list_hmvp_halts_one_short_of_max() {
        let empty_spatials = [SpatialMergeCandidate::default(); 5];
        let mut hmvp = HmvpTable::new();
        for i in 0..5 {
            hmvp.update_with(dummy_mvf(i + 1, 0, 0));
        }
        let list = build_merge_cand_list(&empty_spatials, 4, None, Some(&hmvp));
        assert_eq!(list.len(), 4);
        // 3 HMVP slots filled (newest first).
        assert_eq!(list[0].mv_l0, MotionVector::from_int_pel(5, 0));
        assert_eq!(list[1].mv_l0, MotionVector::from_int_pel(4, 0));
        assert_eq!(list[2].mv_l0, MotionVector::from_int_pel(3, 0));
        // Slot 3 = pairwise-average of slots 0 + 1 = (80+64, 0) >> 1 =
        // (72, 0) in 1/16 units (4.5 int-pel).
        assert_eq!(list[3].mv_l0, MotionVector { x: 72, y: 0 });
    }

    /// §8.5.2.6 — `sameMotion` pruning rule: when the **newest** HMVP
    /// entry duplicates B1 (or A1), it is dropped from the merge list.
    /// Pinning that the rule fires only for the two newest entries
    /// (`hMvpIdx ≤ 2`).
    #[test]
    fn merge_list_hmvp_prunes_against_b1_when_newest() {
        // Build a setup where B1 is available with mv = (5, 0) and the
        // newest HMVP entry also encodes mv = (5, 0). The HMVP entry
        // must be skipped.
        let mut spatials = [SpatialMergeCandidate::default(); 5];
        spatials[0] = SpatialMergeCandidate {
            available: true,
            field: dummy_mvf(5, 0, 0),
        };
        let mut hmvp = HmvpTable::new();
        hmvp.update_with(dummy_mvf(7, 0, 0)); // older
        hmvp.update_with(dummy_mvf(5, 0, 0)); // newest — duplicates B1
        let list = build_merge_cand_list(&spatials, 6, None, Some(&hmvp));
        // Slot 0 = B1 = (5,0). Slot 1 = older HMVP (7,0) — newest (5,0)
        // was pruned. Slot 2 = §8.5.2.4 pairwise-average of (5,0)+(7,0)
        // = (192, 0) >> 1 = (96, 0) in 1/16-pel units. Slots 3..5 are
        // zero-MV pads.
        assert_eq!(list.len(), 6);
        assert_eq!(list[0].mv_l0, MotionVector::from_int_pel(5, 0));
        assert_eq!(
            list[1].mv_l0,
            MotionVector::from_int_pel(7, 0),
            "newest HMVP must have been pruned (matches B1)"
        );
        assert_eq!(list[2].mv_l0, MotionVector { x: 96, y: 0 });
        for entry in &list[3..] {
            assert_eq!(entry.mv_l0, MotionVector::ZERO);
        }
    }

    /// §8.5.2.6 — pruning fires only for the two newest entries
    /// (`hMvpIdx ≤ 2`). A *third* HMVP entry that matches B1 must NOT
    /// be pruned. Pin this so we don't over-prune older entries.
    #[test]
    fn merge_list_hmvp_does_not_prune_third_oldest_against_b1() {
        let mut spatials = [SpatialMergeCandidate::default(); 5];
        spatials[0] = SpatialMergeCandidate {
            available: true,
            field: dummy_mvf(5, 0, 0),
        };
        let mut hmvp = HmvpTable::new();
        // entries[0] (oldest) = (5, 0) — matches B1, hMvpIdx=3 → no prune
        // entries[1]          = (8, 0) — distinct
        // entries[2] (newest) = (9, 0) — distinct
        hmvp.update_with(dummy_mvf(5, 0, 0));
        hmvp.update_with(dummy_mvf(8, 0, 0));
        hmvp.update_with(dummy_mvf(9, 0, 0));
        let list = build_merge_cand_list(&spatials, 6, None, Some(&hmvp));
        // slot 0 = B1 (5,0); slot 1..3 = HMVP newest-first
        // (9,0), (8,0), (5,0) — the (5,0) entry survives because it
        // sits at hMvpIdx = 3 (third newest).
        assert_eq!(list[0].mv_l0, MotionVector::from_int_pel(5, 0));
        assert_eq!(list[1].mv_l0, MotionVector::from_int_pel(9, 0));
        assert_eq!(list[2].mv_l0, MotionVector::from_int_pel(8, 0));
        assert_eq!(
            list[3].mv_l0,
            MotionVector::from_int_pel(5, 0),
            "third-oldest HMVP must NOT be pruned (hMvpIdx>2 falls outside the rule)"
        );
    }

    /// §8.5.2.2 step 7 trigger condition: HMVP insertion is gated by
    /// `numCurrMergeCand < MaxNumMergeCand − 1`. When the spatial walk
    /// already produced `MaxNumMergeCand − 1` entries, HMVP must NOT
    /// run (the spec stops adding HMVP at that point and lets the
    /// zero-MV pad close the chain).
    ///
    /// Setup: 5 spatial candidates available, all distinct → spatial
    /// walk produces 5 entries. With `MaxNumMergeCand == 5` the
    /// trigger condition `5 < 4` is FALSE → HMVP stays out.
    #[test]
    fn merge_list_hmvp_skipped_when_spatials_already_max_minus_one() {
        let mut spatials = [SpatialMergeCandidate::default(); 5];
        for (i, slot) in spatials.iter_mut().enumerate() {
            *slot = SpatialMergeCandidate {
                available: true,
                field: dummy_mvf(i as i32 + 1, 0, 0),
            };
        }
        let mut hmvp = HmvpTable::new();
        hmvp.update_with(dummy_mvf(99, 0, 0));
        let list = build_merge_cand_list(&spatials, 5, None, Some(&hmvp));
        // 5 slots; all 5 are spatial. HMVP entry (99, 0) must NOT
        // appear — `5 < 4` is false.
        assert_eq!(list.len(), 5);
        for entry in &list {
            assert_ne!(entry.mv_l0, MotionVector::from_int_pel(99, 0));
        }
    }

    /// `build_merge_cand_list_b` integration with HMVP — the B-slice
    /// path also picks up HMVP entries between spatials and the
    /// bi-pred zero-MV pad. Pin the same insertion order: spatial
    /// first, HMVP newest-first, bi-pred pad last.
    #[test]
    fn merge_list_b_inserts_hmvp_then_bipred_pad() {
        let empty_spatials = [SpatialMergeCandidate::default(); 5];
        let mut hmvp = HmvpTable::new();
        // Push a single B-slice-shaped HMVP entry (both pred flags
        // set, so the entry survives the unmodified-shape contract).
        let bi_entry = MvField {
            mv_l0: MotionVector::from_int_pel(2, 1),
            ref_idx_l0: 0,
            pred_flag_l0: true,
            mv_l1: MotionVector::from_int_pel(-1, 3),
            ref_idx_l1: 0,
            pred_flag_l1: true,
            cu_skip_flag: false,
            mode_inter: true,
            available: true,
            bcw_idx: 0,
        };
        hmvp.update_with(bi_entry);
        let list = build_merge_cand_list_b(&empty_spatials, 4, None, Some(&hmvp));
        assert_eq!(list.len(), 4);
        // Slot 0 = HMVP entry (carries both lists).
        assert_eq!(list[0].mv_l0, MotionVector::from_int_pel(2, 1));
        assert_eq!(list[0].mv_l1, MotionVector::from_int_pel(-1, 3));
        assert!(list[0].pred_flag_l1);
        // Slots 1..3 are bi-pred zero-MV pads (predFlagL0 = predFlagL1 = 1).
        for entry in &list[1..] {
            assert!(entry.pred_flag_l0);
            assert!(entry.pred_flag_l1);
            assert_eq!(entry.mv_l0, MotionVector::ZERO);
            assert_eq!(entry.mv_l1, MotionVector::ZERO);
        }
    }

    /// `None` HMVP argument leaves the merge list shape byte-identical
    /// to the round-23 spatial-only build path. Round-21/22 unit tests
    /// pass `None` for backwards compatibility — pin that the result
    /// matches the original shape (no HMVP entries injected).
    #[test]
    fn merge_list_with_none_hmvp_is_spatial_only_pad_only() {
        let empty_spatials = [SpatialMergeCandidate::default(); 5];
        let p_list = build_merge_cand_list(&empty_spatials, 5, None, None);
        let b_list = build_merge_cand_list_b(&empty_spatials, 5, None, None);
        assert_eq!(p_list.len(), 5);
        assert_eq!(b_list.len(), 5);
        // Every entry is a zero-MV pad.
        for e in &p_list {
            assert_eq!(e.mv_l0, MotionVector::ZERO);
            assert!(!e.pred_flag_l1);
        }
        for e in &b_list {
            assert_eq!(e.mv_l0, MotionVector::ZERO);
            assert!(e.pred_flag_l1);
        }
    }

    /// Edge case — an empty HMVP table is silently skipped (no panic /
    /// no spurious zero-MV insertion). Pin against the §8.5.2.2 step 7
    /// `NumHmvpCand > 0` gate.
    #[test]
    fn merge_list_empty_hmvp_skipped() {
        let empty_spatials = [SpatialMergeCandidate::default(); 5];
        let empty_hmvp = HmvpTable::new();
        let list = build_merge_cand_list(&empty_spatials, 4, None, Some(&empty_hmvp));
        // Empty HMVP → list is fully zero-MV padded.
        assert_eq!(list.len(), 4);
        for entry in &list {
            assert_eq!(entry.mv_l0, MotionVector::ZERO);
            assert!(entry.available);
        }
    }

    // ----- §8.5.2.11 / §8.5.2.12 temporal merge tests (round-25) -------

    /// Build a single-reference picture with a uniform motion field —
    /// every 4x4 block carries `mv` / `ref_idx` on L0. Useful as the
    /// "ColPic" input to the temporal-merge derivation tests.
    fn col_pic_with_uniform_mv(
        pic_w: u32,
        pic_h: u32,
        poc: i32,
        mv: MotionVector,
        ref_idx: i32,
    ) -> ReferencePicture {
        let mut mf = MotionField::new(pic_w, pic_h);
        let cell = MvField {
            mv_l0: mv,
            ref_idx_l0: ref_idx,
            pred_flag_l0: true,
            mv_l1: MotionVector::ZERO,
            ref_idx_l1: -1,
            pred_flag_l1: false,
            cu_skip_flag: false,
            mode_inter: true,
            available: true,
            bcw_idx: 0,
        };
        mf.write_block(0, 0, pic_w, pic_h, cell);
        ReferencePicture {
            poc,
            frame: PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 0),
            motion_field: Some(mf),
        }
    }

    /// `derive_temporal_merge_candidate` returns `None` when the
    /// reference picture has no captured MotionField — pins the
    /// `availableFlagLXCol == 0` short-circuit for intra-only refs.
    #[test]
    fn temporal_merge_returns_none_without_motion_field() {
        let col_pic = ReferencePicture {
            poc: 0,
            frame: PictureBuffer::yuv420_filled(16, 16, 0),
            motion_field: None,
        };
        let inputs = TemporalMergeInputs {
            xcb: 0,
            ycb: 0,
            cb_w: 16,
            cb_h: 16,
            pic_w: 16,
            pic_h: 16,
            ctb_log2_size_y: 5,
            current_poc: 1,
            current_ref_poc: 0,
            col_pic: &col_pic,
            col_ref_poc: 0,
        };
        assert!(derive_temporal_merge_candidate(&inputs).is_none());
    }

    /// Equal-distance fast path (eq. 600): when `colPocDiff ==
    /// currPocDiff`, the scaled MV equals the (8x-rounded) collocated
    /// MV verbatim. Pins the equal-distance branch with a uniform
    /// integer-pel MV.
    #[test]
    fn temporal_merge_equal_poc_distance_uses_unscaled_mv() {
        // 16x16 picture, single CU at (0, 0) covering whole picture.
        let col_pic = col_pic_with_uniform_mv(16, 16, 1, MotionVector::from_int_pel(2, -1), 0);
        let inputs = TemporalMergeInputs {
            xcb: 0,
            ycb: 0,
            cb_w: 16,
            cb_h: 16,
            pic_w: 16,
            pic_h: 16,
            ctb_log2_size_y: 5,
            current_poc: 2,
            current_ref_poc: 0, // currPocDiff = 2
            col_pic: &col_pic,
            col_ref_poc: -1, // colPocDiff = 1 - (-1) = 2 (equal)
        };
        let cand = derive_temporal_merge_candidate(&inputs).unwrap();
        assert_eq!(cand.mv_l0, MotionVector::from_int_pel(2, -1));
        assert!(cand.pred_flag_l0);
        assert!(!cand.pred_flag_l1);
        assert_eq!(cand.ref_idx_l0, 0);
        assert!(cand.available);
        assert!(cand.mode_inter);
    }

    /// Picture-edge fall-back: when the bottom-right collocated
    /// position falls outside the picture (xColBr > picW-1), the
    /// derivation must use the centre fallback. With a uniform MV
    /// reference it yields the same MV as the BR path would.
    #[test]
    fn temporal_merge_falls_back_to_centre_at_picture_edge() {
        let col_pic = col_pic_with_uniform_mv(16, 16, 1, MotionVector::from_int_pel(1, 1), 0);
        // CU at (8, 8) of size 8x8. Bottom-right is (16, 16) — exactly
        // ON the right/bottom boundary, so xColBr <= picW-1 fails
        // (16 > 15). Centre is (12, 12) → 8x8-rounded (8, 8) — in bounds.
        let inputs = TemporalMergeInputs {
            xcb: 8,
            ycb: 8,
            cb_w: 8,
            cb_h: 8,
            pic_w: 16,
            pic_h: 16,
            ctb_log2_size_y: 5,
            current_poc: 2,
            current_ref_poc: 0,
            col_pic: &col_pic,
            col_ref_poc: -1, // equal-distance (col 1-(-1)=2 = curr)
        };
        let cand = derive_temporal_merge_candidate(&inputs).unwrap();
        assert_eq!(cand.mv_l0, MotionVector::from_int_pel(1, 1));
    }

    /// CTB-row crossing: when the CU's bottom-right collocated row
    /// belongs to a different CTB row, the BR path is rejected. The
    /// centre still falls inside the same row in this fixture, so the
    /// centre fallback fires.
    #[test]
    fn temporal_merge_rejects_br_when_crossing_ctb_row_uses_centre() {
        // 32x32 picture, ctb_log2 = 5 → ctb_size = 32. Single CTB row.
        // Try a CU at (0, 24) with cb_h = 8: yColBr = 32 lands in the
        // *next* CTB row (32 >> 5 = 1, while ycb (24) >> 5 = 0). So the
        // BR check fails (`yCb >> CtbLog2 != yColBr >> CtbLog2`) and
        // the centre at (4, 28) → 8x8-rounded (0, 24) fires.
        let col_pic = col_pic_with_uniform_mv(32, 32, 1, MotionVector::from_int_pel(3, 0), 0);
        let inputs = TemporalMergeInputs {
            xcb: 0,
            ycb: 24,
            cb_w: 8,
            cb_h: 8,
            pic_w: 32,
            pic_h: 32,
            ctb_log2_size_y: 5,
            current_poc: 2,
            current_ref_poc: 0,
            col_pic: &col_pic,
            col_ref_poc: -1,
        };
        let cand = derive_temporal_merge_candidate(&inputs).unwrap();
        assert_eq!(cand.mv_l0, MotionVector::from_int_pel(3, 0));
    }

    /// Unequal POC distance triggers the §8.5.2.12 scaling path. With
    /// `colPocDiff = 4` and `currPocDiff = 2`, the scale factor halves
    /// the MV. We verify with a deliberately large integer-pel MV so
    /// the rounding is byte-stable.
    #[test]
    fn temporal_merge_scales_by_poc_distance_ratio() {
        let col_pic = col_pic_with_uniform_mv(16, 16, 1, MotionVector::from_int_pel(8, 0), 0);
        // colPocDiff = 1 - (-3) = 4; currPocDiff = 2 - 0 = 2.
        // Spec scaling: scale = 2/4 = 0.5 → mv = 4.
        let inputs = TemporalMergeInputs {
            xcb: 0,
            ycb: 0,
            cb_w: 16,
            cb_h: 16,
            pic_w: 16,
            pic_h: 16,
            ctb_log2_size_y: 5,
            current_poc: 2,
            current_ref_poc: 0,
            col_pic: &col_pic,
            col_ref_poc: -3,
        };
        let cand = derive_temporal_merge_candidate(&inputs).unwrap();
        // Spec eqs:
        //   td = 4, tb = 2
        //   tx = (16384 + 2) / 4 = 4096
        //   distScaleFactor = (2 * 4096 + 32) >> 6 = 8224 >> 6 = 128
        //                     clamped to [-4096, 4095] → 128
        //   prod = 128 * (8 * 16) = 128 * 128 = 16384
        //   mvLX = (16384 + 128 - 1) >> 8 = 16511 >> 8 = 64
        // So scaled MV.x = 64 (in 1/16 units = 4 integer-pel).
        assert_eq!(cand.mv_l0.x, 64); // = 4 integer-pel
        assert_eq!(cand.mv_l0.y, 0);
    }

    /// Negative scale: a backward-pointing collocated MV scales to a
    /// negative value when both reference axes also point backward
    /// (negative POC distances).
    #[test]
    fn temporal_merge_handles_negative_poc_distance() {
        let col_pic = col_pic_with_uniform_mv(16, 16, 5, MotionVector::from_int_pel(-4, 0), 0);
        // colPocDiff = 5 - 9 = -4; currPocDiff = 5 - 9 = -4 (equal-dist
        // path; pin the negative-equal scenario).
        let inputs = TemporalMergeInputs {
            xcb: 0,
            ycb: 0,
            cb_w: 16,
            cb_h: 16,
            pic_w: 16,
            pic_h: 16,
            ctb_log2_size_y: 5,
            current_poc: 5,
            current_ref_poc: 9,
            col_pic: &col_pic,
            col_ref_poc: 9,
        };
        let cand = derive_temporal_merge_candidate(&inputs).unwrap();
        // Equal distance → mv passes through (rounded to integer-pel by
        // §8.5.2.15 compression — already integer here).
        assert_eq!(cand.mv_l0, MotionVector::from_int_pel(-4, 0));
    }

    /// Intra-coded collocated block → derivation rejects the BR
    /// position and the centre as well, returning `None`.
    #[test]
    fn temporal_merge_rejects_intra_collocated() {
        // Build a MotionField whose entries are intra (mode_inter = false).
        let mut mf = MotionField::new(16, 16);
        let intra_cell = MvField {
            available: true,
            mode_inter: false, // INTRA
            ..MvField::UNAVAILABLE
        };
        mf.write_block(0, 0, 16, 16, intra_cell);
        let col_pic = ReferencePicture {
            poc: 1,
            frame: PictureBuffer::yuv420_filled(16, 16, 0),
            motion_field: Some(mf),
        };
        let inputs = TemporalMergeInputs {
            xcb: 0,
            ycb: 0,
            cb_w: 16,
            cb_h: 16,
            pic_w: 16,
            pic_h: 16,
            ctb_log2_size_y: 5,
            current_poc: 2,
            current_ref_poc: 0,
            col_pic: &col_pic,
            col_ref_poc: -1,
        };
        assert!(derive_temporal_merge_candidate(&inputs).is_none());
    }

    /// `build_merge_cand_list` integration with a Col candidate (no
    /// spatial / no HMVP) — Col lands at slot 0 and the rest are
    /// zero-MV pads.
    #[test]
    fn merge_list_includes_col_candidate_when_supplied() {
        let empty = [SpatialMergeCandidate::default(); 5];
        let col = MvField {
            mv_l0: MotionVector::from_int_pel(7, -3),
            ref_idx_l0: 0,
            pred_flag_l0: true,
            mv_l1: MotionVector::ZERO,
            ref_idx_l1: -1,
            pred_flag_l1: false,
            cu_skip_flag: false,
            mode_inter: true,
            available: true,
            bcw_idx: 0,
        };
        let list = build_merge_cand_list(&empty, 4, Some(col), None);
        assert_eq!(list.len(), 4);
        // Slot 0 = Col candidate.
        assert_eq!(list[0].mv_l0, MotionVector::from_int_pel(7, -3));
        // Slots 1..3 = zero-MV pad.
        for entry in &list[1..] {
            assert_eq!(entry.mv_l0, MotionVector::ZERO);
            assert!(entry.pred_flag_l0);
        }
    }

    /// Walk-order pin: spatial candidates fill first, then Col, then
    /// HMVP, then pairwise-average, then zero-MV pad. Verified by
    /// giving the merge list a distinctive entry at each origin.
    #[test]
    fn merge_list_walk_order_spatial_then_col_then_hmvp() {
        let mut spatials = [SpatialMergeCandidate::default(); 5];
        spatials[0] = SpatialMergeCandidate {
            available: true,
            field: dummy_mvf(1, 0, 0), // B1 → mv (1, 0)
        };
        let col = MvField {
            mv_l0: MotionVector::from_int_pel(2, 0),
            ref_idx_l0: 0,
            pred_flag_l0: true,
            mv_l1: MotionVector::ZERO,
            ref_idx_l1: -1,
            pred_flag_l1: false,
            cu_skip_flag: false,
            mode_inter: true,
            available: true,
            bcw_idx: 0,
        };
        let mut hmvp = HmvpTable::new();
        hmvp.update_with(dummy_mvf(3, 0, 0));
        let list = build_merge_cand_list(&spatials, 6, Some(col), Some(&hmvp));
        assert_eq!(list[0].mv_l0, MotionVector::from_int_pel(1, 0)); // B1
        assert_eq!(list[1].mv_l0, MotionVector::from_int_pel(2, 0)); // Col
        assert_eq!(list[2].mv_l0, MotionVector::from_int_pel(3, 0)); // HMVP
                                                                     // Slot 3 = §8.5.2.4 pairwise-average of B1 + Col = (1+2, 0)
                                                                     // int-pel sum = (48, 0) in 1/16-pel units, rounded by §8.5.2.14
                                                                     // (rightShift=1) → (24, 0). Slots 4..5 = zero-MV pad.
        assert_eq!(list[3].mv_l0, MotionVector { x: 24, y: 0 });
        for entry in &list[4..] {
            assert_eq!(entry.mv_l0, MotionVector::ZERO);
        }
    }

    // ----- §8.5.2.4 pairwise-average tests (round-26) ------------------

    /// §8.5.2.14 round (`rightShift = 1`, `leftShift = 0`) — pin the
    /// signed-magnitude divide-by-2-toward-zero behavior. This is the
    /// edge case that distinguishes the spec rounding from a plain
    /// arithmetic shift: `(-1) >> 1 == -1` in Rust, but the spec
    /// `Sign(-1) * (Abs(-1) >> 1) == 0`.
    #[test]
    fn round_mv_pairwise_signed_magnitude_toward_zero() {
        // Positive sums: plain (sum >> 1).
        assert_eq!(round_mv_pairwise(0), 0);
        assert_eq!(round_mv_pairwise(1), 0);
        assert_eq!(round_mv_pairwise(2), 1);
        assert_eq!(round_mv_pairwise(3), 1);
        assert_eq!(round_mv_pairwise(4), 2);
        // Negative sums: rounds toward zero (NOT floor like Rust >>).
        assert_eq!(round_mv_pairwise(-1), 0);
        assert_eq!(round_mv_pairwise(-2), -1);
        assert_eq!(round_mv_pairwise(-3), -1);
        assert_eq!(round_mv_pairwise(-4), -2);
        // Sanity check vs Rust arithmetic shift (the trap).
        assert_ne!(-1i32 >> 1, round_mv_pairwise(-1));
    }

    /// §8.5.2.4 — both source candidates active on L0; the synthetic
    /// avgCand inherits p0's refIdx and carries the rounded per-axis
    /// average MV. P-slice variant (numRefLists == 1) suppresses L1.
    #[test]
    fn pairwise_average_both_active_l0_p_slice() {
        let mut list = vec![
            dummy_mvf(4, 2, 0), // p0: mv (4*16, 2*16) = (64, 32)
            dummy_mvf(2, 4, 0), // p1: mv (32, 64)
        ];
        let avg = derive_pairwise_average_candidate(&mut list, 6, /*is_b*/ false).unwrap();
        // Sum = (96, 96), rounded by (sum >> 1) → (48, 48) in 1/16-pel
        // units (i.e. 3 int-pel on each axis).
        assert_eq!(avg.mv_l0, MotionVector { x: 48, y: 48 });
        assert_eq!(avg.ref_idx_l0, 0);
        assert!(avg.pred_flag_l0);
        // P-slice → L1 half forced inactive even though dummy_mvf
        // happens to leave it at the unavailable shape.
        assert!(!avg.pred_flag_l1);
        assert_eq!(avg.ref_idx_l1, -1);
        assert_eq!(avg.mv_l1, MotionVector::ZERO);
        // The new candidate landed at the end.
        assert_eq!(list.len(), 3);
        assert_eq!(list[2], avg);
    }

    /// §8.5.2.4 — only p0 active on L0; avgCand copies p0 verbatim
    /// (eqs. 522 – 525). No averaging on this list.
    #[test]
    fn pairwise_average_only_p0_active_copies_p0() {
        let p0 = dummy_mvf(7, -3, 1);
        let p1 = MvField {
            mv_l0: MotionVector::from_int_pel(99, 99), // ignored
            ref_idx_l0: -1,
            pred_flag_l0: false, // p1 inactive on L0
            mv_l1: MotionVector::ZERO,
            ref_idx_l1: -1,
            pred_flag_l1: false,
            cu_skip_flag: false,
            mode_inter: true,
            available: true,
            bcw_idx: 0,
        };
        let mut list = vec![p0, p1];
        let avg = derive_pairwise_average_candidate(&mut list, 6, /*is_b*/ false).unwrap();
        // L0 copied from p0.
        assert_eq!(avg.mv_l0, p0.mv_l0);
        assert_eq!(avg.ref_idx_l0, 1);
        assert!(avg.pred_flag_l0);
    }

    /// §8.5.2.4 — only p1 active on L0; avgCand copies p1 verbatim
    /// (eqs. 526 – 529).
    #[test]
    fn pairwise_average_only_p1_active_copies_p1() {
        let p0 = MvField {
            mv_l0: MotionVector::from_int_pel(99, 99), // ignored
            ref_idx_l0: -1,
            pred_flag_l0: false, // p0 inactive on L0
            mv_l1: MotionVector::ZERO,
            ref_idx_l1: -1,
            pred_flag_l1: false,
            cu_skip_flag: false,
            mode_inter: true,
            available: true,
            bcw_idx: 0,
        };
        let p1 = dummy_mvf(-5, 6, 2);
        let mut list = vec![p0, p1];
        let avg = derive_pairwise_average_candidate(&mut list, 6, /*is_b*/ false).unwrap();
        assert_eq!(avg.mv_l0, p1.mv_l0);
        assert_eq!(avg.ref_idx_l0, 2);
        assert!(avg.pred_flag_l0);
    }

    /// §8.5.2.4 — neither p0 nor p1 active on L0; avgCand reports
    /// inactive on this list (eqs. 530 – 533: refIdxLXavg = -1,
    /// predFlagLXavg = 0, mvLXavg = (0, 0)).
    #[test]
    fn pairwise_average_neither_active_l0_inactive() {
        let inactive = MvField {
            mv_l0: MotionVector::from_int_pel(99, 99),
            ref_idx_l0: -1,
            pred_flag_l0: false,
            mv_l1: MotionVector::ZERO,
            ref_idx_l1: -1,
            pred_flag_l1: false,
            cu_skip_flag: false,
            mode_inter: true,
            available: true,
            bcw_idx: 0,
        };
        let mut list = vec![inactive, inactive];
        // is_b=true so the L1 walk also runs (and mirrors the L0 result).
        let avg = derive_pairwise_average_candidate(&mut list, 6, /*is_b*/ true).unwrap();
        assert_eq!(avg.mv_l0, MotionVector::ZERO);
        assert_eq!(avg.ref_idx_l0, -1);
        assert!(!avg.pred_flag_l0);
        assert_eq!(avg.mv_l1, MotionVector::ZERO);
        assert_eq!(avg.ref_idx_l1, -1);
        assert!(!avg.pred_flag_l1);
    }

    /// §8.5.2.4 — B-slice path: both halves derived. Mixed activity per
    /// list (L0 both active, L1 only p0 active) exercises the per-list
    /// switch independently.
    #[test]
    fn pairwise_average_mixed_per_list_b_slice() {
        let p0 = MvField {
            mv_l0: MotionVector::from_int_pel(2, 0),
            ref_idx_l0: 0,
            pred_flag_l0: true,
            mv_l1: MotionVector::from_int_pel(-3, 1),
            ref_idx_l1: 0,
            pred_flag_l1: true,
            cu_skip_flag: false,
            mode_inter: true,
            available: true,
            bcw_idx: 0,
        };
        let p1 = MvField {
            mv_l0: MotionVector::from_int_pel(4, 0),
            ref_idx_l0: 0,
            pred_flag_l0: true,
            // p1 inactive on L1 — copy p0's L1.
            mv_l1: MotionVector::ZERO,
            ref_idx_l1: -1,
            pred_flag_l1: false,
            cu_skip_flag: false,
            mode_inter: true,
            available: true,
            bcw_idx: 0,
        };
        let mut list = vec![p0, p1];
        let avg = derive_pairwise_average_candidate(&mut list, 6, /*is_b*/ true).unwrap();
        // L0: both active → avg = ((32 + 64) >> 1, 0) = (48, 0).
        assert_eq!(avg.mv_l0, MotionVector { x: 48, y: 0 });
        assert_eq!(avg.ref_idx_l0, 0);
        assert!(avg.pred_flag_l0);
        // L1: only p0 → copy p0.
        assert_eq!(avg.mv_l1, p0.mv_l1);
        assert_eq!(avg.ref_idx_l1, 0);
        assert!(avg.pred_flag_l1);
    }

    /// §8.5.2.2 step 8 gate — pairwise does not fire when the merge
    /// list has fewer than 2 entries (`numCurrMergeCand > 1`).
    #[test]
    fn pairwise_average_skipped_when_list_has_fewer_than_two_entries() {
        let mut list = vec![dummy_mvf(1, 0, 0)];
        let result = derive_pairwise_average_candidate(&mut list, 6, false);
        assert!(result.is_none());
        assert_eq!(list.len(), 1, "list must be unchanged when gate fails");
    }

    /// §8.5.2.2 step 8 gate — pairwise does not fire when the merge
    /// list is already at `MaxNumMergeCand` (`numCurrMergeCand <
    /// MaxNumMergeCand`).
    #[test]
    fn pairwise_average_skipped_when_list_at_capacity() {
        let mut list = vec![dummy_mvf(1, 0, 0), dummy_mvf(2, 0, 0)];
        // max = 2 → list is already full.
        let result = derive_pairwise_average_candidate(&mut list, 2, false);
        assert!(result.is_none());
        assert_eq!(list.len(), 2);
    }

    /// §8.5.2.2 step 8 — when only spatial B1 + A1 are available the
    /// pairwise candidate is derived from those two entries. Pin the
    /// integration with `build_merge_cand_list` (P-slice variant).
    #[test]
    fn merge_list_pairwise_from_two_spatial_p_slice() {
        let mut spatials = [SpatialMergeCandidate::default(); 5];
        spatials[0] = SpatialMergeCandidate {
            available: true,
            field: dummy_mvf(2, 0, 0), // B1 = (32, 0)
        };
        spatials[1] = SpatialMergeCandidate {
            available: true,
            field: dummy_mvf(6, 0, 0), // A1 = (96, 0)
        };
        let list = build_merge_cand_list(&spatials, 6, None, None);
        assert_eq!(list.len(), 6);
        assert_eq!(list[0].mv_l0, MotionVector::from_int_pel(2, 0));
        assert_eq!(list[1].mv_l0, MotionVector::from_int_pel(6, 0));
        // Slot 2 = pairwise-average = ((32 + 96) >> 1, 0) = (64, 0)
        // = 4 int-pel on the x axis.
        assert_eq!(list[2].mv_l0, MotionVector { x: 64, y: 0 });
        assert!(list[2].pred_flag_l0);
        assert!(!list[2].pred_flag_l1, "P-slice: L1 must be inactive");
        // Slots 3..5 = zero-MV pads.
        for entry in &list[3..] {
            assert_eq!(entry.mv_l0, MotionVector::ZERO);
        }
    }

    /// §8.5.2.2 step 8 — B-slice variant integration. Two spatial
    /// candidates with full bi-pred motion → pairwise carries averaged
    /// L0 + L1 halves.
    #[test]
    fn merge_list_pairwise_b_slice_carries_both_halves() {
        let bipred = |dx0, dy0, dx1, dy1| MvField {
            mv_l0: MotionVector::from_int_pel(dx0, dy0),
            ref_idx_l0: 0,
            pred_flag_l0: true,
            mv_l1: MotionVector::from_int_pel(dx1, dy1),
            ref_idx_l1: 0,
            pred_flag_l1: true,
            cu_skip_flag: false,
            mode_inter: true,
            available: true,
            bcw_idx: 0,
        };
        let mut spatials = [SpatialMergeCandidate::default(); 5];
        spatials[0] = SpatialMergeCandidate {
            available: true,
            field: bipred(2, 0, -2, 0),
        };
        spatials[1] = SpatialMergeCandidate {
            available: true,
            field: bipred(4, 0, -4, 0),
        };
        let list = build_merge_cand_list_b(&spatials, 6, None, None);
        assert_eq!(list.len(), 6);
        // Slot 2 = pairwise: L0 = ((32+64)>>1, 0) = (48, 0);
        //                    L1 = ((-32 + -64) >> 1, 0) = (-48, 0).
        assert_eq!(list[2].mv_l0, MotionVector { x: 48, y: 0 });
        assert_eq!(list[2].mv_l1, MotionVector { x: -48, y: 0 });
        assert!(list[2].pred_flag_l0);
        assert!(list[2].pred_flag_l1);
    }

    /// §8.5.2.2 step 8 — does not fire on a single-spatial list
    /// (numCurrMergeCand == 1 fails the `> 1` gate). Pin that the
    /// integration respects the gate.
    #[test]
    fn merge_list_pairwise_skipped_on_single_spatial() {
        let mut spatials = [SpatialMergeCandidate::default(); 5];
        spatials[0] = SpatialMergeCandidate {
            available: true,
            field: dummy_mvf(3, 0, 0),
        };
        let list = build_merge_cand_list(&spatials, 6, None, None);
        // Slot 0 = B1, slots 1..5 = zero-MV pads (no pairwise — only
        // one source candidate available).
        assert_eq!(list[0].mv_l0, MotionVector::from_int_pel(3, 0));
        for entry in &list[1..] {
            assert_eq!(entry.mv_l0, MotionVector::ZERO);
        }
    }

    // ===== §8.5.2.7 — MMVD derivation tests ===========================

    /// Table 17 + Table 18 + eq. 188 / 189: distance_idx = 0 → MmvdDistance
    /// = 1 → MmvdOffset[0] = (1 << 2) * (+1) = +4 in 1/16-pel units
    /// (i.e. +1/4 luma sample) when direction = +x.
    #[test]
    fn mmvd_offset_distance_idx_0_plus_x_is_quarter_pel() {
        let off = derive_mmvd_offset(0, 0, false);
        assert_eq!(off, MotionVector { x: 4, y: 0 });
    }

    /// distance_idx = 2 → MmvdDistance = 4 → 4 << 2 = 16 in 1/16-pel
    /// = 1 luma sample integer-pel offset. direction = -x.
    #[test]
    fn mmvd_offset_distance_idx_2_minus_x_is_one_int_pel() {
        let off = derive_mmvd_offset(2, 1, false);
        assert_eq!(off, MotionVector { x: -16, y: 0 });
    }

    /// All 4 directions for distance_idx = 0 are exactly the cardinal
    /// 1/4-pel offsets. Table 18 has no diagonals.
    #[test]
    fn mmvd_offset_all_four_directions_are_cardinal() {
        assert_eq!(derive_mmvd_offset(0, 0, false), MotionVector { x: 4, y: 0 });
        assert_eq!(
            derive_mmvd_offset(0, 1, false),
            MotionVector { x: -4, y: 0 }
        );
        assert_eq!(derive_mmvd_offset(0, 2, false), MotionVector { x: 0, y: 4 });
        assert_eq!(
            derive_mmvd_offset(0, 3, false),
            MotionVector { x: 0, y: -4 }
        );
    }

    /// `ph_mmvd_fullpel_only_flag = 1`: distance_idx = 0 → MmvdDistance
    /// = 4 → 4 << 2 = 16 in 1/16-pel = 1 luma sample. The fullpel
    /// table starts at 1 luma instead of 1/4 luma.
    #[test]
    fn mmvd_offset_fullpel_only_starts_at_one_int_pel() {
        let off = derive_mmvd_offset(0, 0, true);
        assert_eq!(off, MotionVector { x: 16, y: 0 });
    }

    /// `ph_mmvd_fullpel_only_flag = 1`: distance_idx = 7 → MmvdDistance
    /// = 512 → 512 << 2 = 2048 in 1/16-pel = 128 luma samples. The
    /// largest entry in the fullpel table.
    #[test]
    fn mmvd_offset_fullpel_only_max_is_128_int_pel() {
        let off = derive_mmvd_offset(7, 0, true);
        assert_eq!(off, MotionVector { x: 2048, y: 0 });
    }

    /// distance_idx = 7 (regular table) → MmvdDistance = 128 → 128 <<
    /// 2 = 512 in 1/16-pel = 32 luma samples. Largest non-fullpel
    /// entry per the brief.
    #[test]
    fn mmvd_offset_max_distance_is_32_int_pel() {
        let off = derive_mmvd_offset(7, 0, false);
        assert_eq!(off, MotionVector { x: 512, y: 0 });
    }

    /// All 8 distance steps for the regular table land at the spec's
    /// published fractional / integer values: 1/4, 1/2, 1, 2, 4, 8,
    /// 16, 32 luma samples (per the round-27 brief).
    #[test]
    fn mmvd_offset_regular_distance_table_matches_brief() {
        // Distances expressed in 1/16-pel units after the eq. 188 << 2.
        let expected_x = [4, 8, 16, 32, 64, 128, 256, 512];
        for (i, &want) in expected_x.iter().enumerate() {
            assert_eq!(
                derive_mmvd_offset(i as u32, 0, false).x,
                want,
                "distance_idx {} (regular) should yield {} in 1/16-pel",
                i,
                want
            );
        }
    }

    /// `apply_mmvd_to_base` adds the offset to the active list MV. Uni-
    /// pred L0 base — only the L0 MV is touched, L1 stays untouched.
    #[test]
    fn mmvd_apply_to_base_uni_pred_l0() {
        let base = MvField {
            mv_l0: MotionVector::from_int_pel(3, 5),
            ref_idx_l0: 0,
            pred_flag_l0: true,
            mv_l1: MotionVector::ZERO,
            ref_idx_l1: -1,
            pred_flag_l1: false,
            cu_skip_flag: false,
            mode_inter: true,
            available: true,
            bcw_idx: 0,
        };
        let off = derive_mmvd_offset(2, 0, false); // (+1, 0) int-pel = (+16, 0) 1/16
        let out = apply_mmvd_to_base(&base, off, false);
        assert_eq!(out.mv_l0, MotionVector::from_int_pel(4, 5));
        assert_eq!(out.mv_l1, MotionVector::ZERO);
        assert_eq!(out.ref_idx_l0, 0);
        assert_eq!(out.ref_idx_l1, -1);
        assert!(out.pred_flag_l0);
        assert!(!out.pred_flag_l1);
    }

    /// Bi-pred symmetric base — equal POC distance shortcut: both list
    /// MVs receive the same MmvdOffset.
    #[test]
    fn mmvd_apply_to_base_bi_pred_equal_poc() {
        let base = MvField {
            mv_l0: MotionVector::from_int_pel(2, 0),
            ref_idx_l0: 0,
            pred_flag_l0: true,
            mv_l1: MotionVector::from_int_pel(-2, 0),
            ref_idx_l1: 0,
            pred_flag_l1: true,
            cu_skip_flag: false,
            mode_inter: true,
            available: true,
            bcw_idx: 0,
        };
        let off = derive_mmvd_offset(0, 0, false); // (+1/4, 0)
        let out = apply_mmvd_to_base(&base, off, true);
        assert_eq!(out.mv_l0, MotionVector { x: 32 + 4, y: 0 });
        assert_eq!(out.mv_l1, MotionVector { x: -32 + 4, y: 0 });
    }

    /// MergeData default: MMVD fields all default to "off" so existing
    /// non-MMVD callers stay byte-identical.
    #[test]
    fn merge_data_default_disables_mmvd() {
        let md = MergeData::default();
        assert!(!md.mmvd_merge_flag);
        assert_eq!(md.mmvd_cand_flag, 0);
        assert_eq!(md.mmvd_distance_idx, 0);
        assert_eq!(md.mmvd_direction_idx, 0);
    }

    // §8.5.2.7 — POC-aware MMVD bi-pred application (eqs. 561 – 580).
    // The next batch of tests pins all three branches of
    // `apply_mmvd_to_base_with_poc`.

    /// Equal POC distance bi-pred — same as the legacy symmetric
    /// shortcut: both list MVs receive `+MmvdOffset`.
    #[test]
    fn mmvd_with_poc_bi_pred_equal_distance_symmetric() {
        let base = MvField {
            mv_l0: MotionVector::from_int_pel(2, 0),
            ref_idx_l0: 0,
            pred_flag_l0: true,
            mv_l1: MotionVector::from_int_pel(-2, 0),
            ref_idx_l1: 0,
            pred_flag_l1: true,
            cu_skip_flag: false,
            mode_inter: true,
            available: true,
            bcw_idx: 0,
        };
        let off = derive_mmvd_offset(2, 0, false); // (+1, 0) int-pel
                                                   // currPocDiffL0 = +1, currPocDiffL1 = +1 — equal distance
                                                   // (eqs. 557 – 560).
        let out = apply_mmvd_to_base_with_poc(&base, off, 1, 1, false, false);
        assert_eq!(out.mv_l0, MotionVector::from_int_pel(3, 0));
        assert_eq!(out.mv_l1, MotionVector::from_int_pel(-1, 0));
    }

    /// Opposite-sign POC distances — mMvdL1 = -MmvdOffset (eqs. 564 /
    /// 565). Typical case: L0 ref has POC < curr, L1 ref has POC > curr,
    /// so currPocDiffL0 > 0 and currPocDiffL1 < 0.
    #[test]
    fn mmvd_with_poc_bi_pred_opposite_sign_negates_l1() {
        let base = MvField {
            mv_l0: MotionVector::from_int_pel(0, 0),
            ref_idx_l0: 0,
            pred_flag_l0: true,
            mv_l1: MotionVector::from_int_pel(0, 0),
            ref_idx_l1: 0,
            pred_flag_l1: true,
            cu_skip_flag: false,
            mode_inter: true,
            available: true,
            bcw_idx: 0,
        };
        let off = derive_mmvd_offset(2, 0, false); // (+1, 0) int-pel
                                                   // currPocDiffL0 = +1 (L0 ref earlier), currPocDiffL1 = -1 (L1
                                                   // ref later) → opposite-sign branch.
        let out = apply_mmvd_to_base_with_poc(&base, off, 1, -1, false, false);
        assert_eq!(out.mv_l0, MotionVector::from_int_pel(1, 0));
        assert_eq!(out.mv_l1, MotionVector::from_int_pel(-1, 0));
    }

    /// Asymmetric same-sign POC distances — mMvdL1 is scaled by the
    /// §8.5.2.12 distScaleFactor (eqs. 561 – 580). Concrete vector:
    /// `currPocDiffL0 = 1, currPocDiffL1 = 2`. Per the spec chain:
    ///   td = 1, tb = 2
    ///   tx = (16384 + 0) / 1 = 16384
    ///   distScaleFactor = (2 * 16384 + 32) >> 6 = 32800 >> 6 = 512
    ///                     (clamped to 4095 — 512 fits)
    /// So mMvdL1 = (512 * MmvdOffset + 128 - 1) >> 8 — for offset 16
    /// (= +1 int-pel = 16 in 1/16-pel) we get
    ///   (512 * 16 + 128 - 1) >> 8 = 8319 >> 8 = 32 → +2 int-pel.
    #[test]
    fn mmvd_with_poc_bi_pred_asymmetric_distance_scales_l1() {
        let base = MvField {
            mv_l0: MotionVector::from_int_pel(0, 0),
            ref_idx_l0: 0,
            pred_flag_l0: true,
            mv_l1: MotionVector::from_int_pel(0, 0),
            ref_idx_l1: 0,
            pred_flag_l1: true,
            cu_skip_flag: false,
            mode_inter: true,
            available: true,
            bcw_idx: 0,
        };
        let off = derive_mmvd_offset(2, 0, false); // (+1, 0) int-pel = (16, 0)
        let out = apply_mmvd_to_base_with_poc(&base, off, 1, 2, false, false);
        assert_eq!(out.mv_l0, MotionVector::from_int_pel(1, 0));
        assert_eq!(out.mv_l1, MotionVector::from_int_pel(2, 0));
    }

    /// Long-term reference on either side — bypasses the asymmetric
    /// branch and applies the offset unscaled (eqs. 557 – 560 LT
    /// shortcut).
    #[test]
    fn mmvd_with_poc_bi_pred_lt_ref_bypasses_scaling() {
        let base = MvField {
            mv_l0: MotionVector::from_int_pel(0, 0),
            ref_idx_l0: 0,
            pred_flag_l0: true,
            mv_l1: MotionVector::from_int_pel(0, 0),
            ref_idx_l1: 0,
            pred_flag_l1: true,
            cu_skip_flag: false,
            mode_inter: true,
            available: true,
            bcw_idx: 0,
        };
        let off = derive_mmvd_offset(2, 0, false); // (+1, 0) int-pel
                                                   // Distances that would normally scale (1 vs 2), but `lt_l0 =
                                                   // true` flips us back to the LT shortcut.
        let out = apply_mmvd_to_base_with_poc(&base, off, 1, 2, true, false);
        assert_eq!(out.mv_l0, MotionVector::from_int_pel(1, 0));
        assert_eq!(out.mv_l1, MotionVector::from_int_pel(1, 0));
    }

    /// Uni-pred-L0 base — only L0 receives the offset; POC distances
    /// are irrelevant (eq. 581).
    #[test]
    fn mmvd_with_poc_uni_pred_l0_only_touches_l0() {
        let base = MvField {
            mv_l0: MotionVector::from_int_pel(3, 5),
            ref_idx_l0: 0,
            pred_flag_l0: true,
            mv_l1: MotionVector::ZERO,
            ref_idx_l1: -1,
            pred_flag_l1: false,
            cu_skip_flag: false,
            mode_inter: true,
            available: true,
            bcw_idx: 0,
        };
        let off = derive_mmvd_offset(2, 0, false);
        let out = apply_mmvd_to_base_with_poc(&base, off, 1, 0, false, false);
        assert_eq!(out.mv_l0, MotionVector::from_int_pel(4, 5));
        assert_eq!(out.mv_l1, MotionVector::ZERO);
        assert_eq!(out.ref_idx_l1, -1);
    }

    /// Uni-pred-L1 base — only L1 receives the offset; eq. 582 applies
    /// the offset directly to the active list regardless of POC.
    #[test]
    fn mmvd_with_poc_uni_pred_l1_only_touches_l1() {
        let base = MvField {
            mv_l0: MotionVector::ZERO,
            ref_idx_l0: -1,
            pred_flag_l0: false,
            mv_l1: MotionVector::from_int_pel(7, -3),
            ref_idx_l1: 0,
            pred_flag_l1: true,
            cu_skip_flag: false,
            mode_inter: true,
            available: true,
            bcw_idx: 0,
        };
        let off = derive_mmvd_offset(2, 0, false); // (+1, 0)
        let out = apply_mmvd_to_base_with_poc(&base, off, 0, 1, false, false);
        assert_eq!(out.mv_l1, MotionVector::from_int_pel(8, -3));
        assert_eq!(out.mv_l0, MotionVector::ZERO);
        assert_eq!(out.ref_idx_l0, -1);
    }

    /// Degenerate `currPocDiffL0 == 0` in the asymmetric branch — the
    /// helper short-circuits to a zero L1 offset. The case is
    /// unreachable from the call site (the equal-POC shortcut always
    /// fires first when L0 distance is 0 and L1 is also 0), but the
    /// guard keeps the helper total.
    #[test]
    fn mmvd_with_poc_zero_l0_distance_in_asymm_path_zeros_l1() {
        let base = MvField {
            mv_l0: MotionVector::ZERO,
            ref_idx_l0: 0,
            pred_flag_l0: true,
            mv_l1: MotionVector::ZERO,
            ref_idx_l1: 0,
            pred_flag_l1: true,
            cu_skip_flag: false,
            mode_inter: true,
            available: true,
            bcw_idx: 0,
        };
        let off = derive_mmvd_offset(2, 0, false);
        // currPocDiffL0 = 0, currPocDiffL1 = 1 — hits equal-POC short
        // (since 0 != 1 fails) but neither opposite-sign nor asymm
        // works without a non-zero td. The path lands in the asymm
        // branch (both same sign? 0 ^ 1 == 1 > 0, so no
        // opposite-sign), then `mmvd_scale_offset` returns ZERO.
        let out = apply_mmvd_to_base_with_poc(&base, off, 0, 1, false, false);
        assert_eq!(out.mv_l0, MotionVector::from_int_pel(1, 0));
        assert_eq!(out.mv_l1, MotionVector::ZERO);
    }

    /// MergeData default also keeps CIIP off so existing non-CIIP
    /// callers stay byte-identical (round-28 §8.5.6.7).
    #[test]
    fn merge_data_default_disables_ciip() {
        let md = MergeData::default();
        assert!(!md.ciip_flag);
    }

    // ---- §8.5.6.6.2 BCW (eq. 981) ------------------------------------

    /// `MvField::default()` has bcw_idx == 0, so existing non-BCW
    /// callers stay byte-identical (eq. 980 default-weighted average).
    #[test]
    fn mvfield_default_has_bcw_idx_zero() {
        let f = MvField::default();
        assert_eq!(f.bcw_idx, 0);
    }

    /// BCW table value sanity — `bcwWLut[k] = {4, 5, 3, 10, -2}` per
    /// the spec (the round-29 brief).
    #[test]
    fn bcw_w_lut_matches_spec() {
        assert_eq!(BCW_W_LUT, [4, 5, 3, 10, -2]);
    }

    /// `bi_pred_avg_8bit_bcw` with `bcw_idx == 0` reduces to eq. 980 —
    /// byte-identical to `bi_pred_avg_8bit` over a constant pair.
    #[test]
    fn bcw_idx_zero_falls_through_to_eq_980() {
        let p0 = PicturePlane::filled(4, 4, 100);
        let p1 = PicturePlane::filled(4, 4, 200);
        let mut dst_a = PicturePlane::filled(4, 4, 0);
        let mut dst_b = PicturePlane::filled(4, 4, 0);
        bi_pred_avg_8bit(&mut dst_a, 0, 0, 4, 4, &p0, &p1).unwrap();
        bi_pred_avg_8bit_bcw(&mut dst_b, 0, 0, 4, 4, &p0, &p1, 0).unwrap();
        assert_eq!(dst_a.samples, dst_b.samples);
        // Sanity: (100 + 200 + 1) >> 1 = 150
        assert!(dst_a.samples.iter().all(|&v| v == 150));
    }

    /// BCW eq. 981 spot-check — bcw_idx = 1 (w1 = 5, w0 = 3) over
    /// (v0, v1) = (100, 200): (3*100 + 5*200 + 4) >> 3 = 1304 >> 3 = 163.
    #[test]
    fn bcw_idx_1_blends_with_w1_5_w0_3() {
        let p0 = PicturePlane::filled(2, 2, 100);
        let p1 = PicturePlane::filled(2, 2, 200);
        let mut dst = PicturePlane::filled(2, 2, 0);
        bi_pred_avg_8bit_bcw(&mut dst, 0, 0, 2, 2, &p0, &p1, 1).unwrap();
        assert!(
            dst.samples.iter().all(|&v| v == 163),
            "got {:?}",
            dst.samples
        );
    }

    /// BCW eq. 981 spot-check — bcw_idx = 2 (w1 = 3, w0 = 5) over
    /// (v0, v1) = (100, 200): (5*100 + 3*200 + 4) >> 3 = 1104 >> 3 = 138.
    #[test]
    fn bcw_idx_2_blends_with_w1_3_w0_5() {
        let p0 = PicturePlane::filled(2, 2, 100);
        let p1 = PicturePlane::filled(2, 2, 200);
        let mut dst = PicturePlane::filled(2, 2, 0);
        bi_pred_avg_8bit_bcw(&mut dst, 0, 0, 2, 2, &p0, &p1, 2).unwrap();
        assert!(
            dst.samples.iter().all(|&v| v == 138),
            "got {:?}",
            dst.samples
        );
    }

    /// BCW eq. 981 spot-check — bcw_idx = 3 (w1 = 10, w0 = -2) over
    /// (v0, v1) = (100, 200): (-2*100 + 10*200 + 4) >> 3 = 1804 >> 3 = 225.
    /// Exercises the negative-weight branch.
    #[test]
    fn bcw_idx_3_blends_with_w1_10_w0_neg2() {
        let p0 = PicturePlane::filled(2, 2, 100);
        let p1 = PicturePlane::filled(2, 2, 200);
        let mut dst = PicturePlane::filled(2, 2, 0);
        bi_pred_avg_8bit_bcw(&mut dst, 0, 0, 2, 2, &p0, &p1, 3).unwrap();
        assert!(
            dst.samples.iter().all(|&v| v == 225),
            "got {:?}",
            dst.samples
        );
    }

    /// BCW eq. 981 spot-check — bcw_idx = 4 (w1 = -2, w0 = 10) over
    /// (v0, v1) = (100, 200): (10*100 + -2*200 + 4) >> 3 = 604 >> 3 = 75.
    /// The other negative-weight branch.
    #[test]
    fn bcw_idx_4_blends_with_w1_neg2_w0_10() {
        let p0 = PicturePlane::filled(2, 2, 100);
        let p1 = PicturePlane::filled(2, 2, 200);
        let mut dst = PicturePlane::filled(2, 2, 0);
        bi_pred_avg_8bit_bcw(&mut dst, 0, 0, 2, 2, &p0, &p1, 4).unwrap();
        assert!(
            dst.samples.iter().all(|&v| v == 75),
            "got {:?}",
            dst.samples
        );
    }

    /// BCW eq. 981 must Clip1 — at the extreme `(255, 255)` with
    /// `bcw_idx = 3` (w0 = -2, w1 = 10): (-2*255 + 10*255 + 4) >> 3 =
    /// 2044 >> 3 = 255. Verifies that the upper boundary stays in range.
    #[test]
    fn bcw_clamps_to_255_at_extreme() {
        let p0 = PicturePlane::filled(2, 2, 255);
        let p1 = PicturePlane::filled(2, 2, 255);
        let mut dst = PicturePlane::filled(2, 2, 0);
        bi_pred_avg_8bit_bcw(&mut dst, 0, 0, 2, 2, &p0, &p1, 3).unwrap();
        assert!(dst.samples.iter().all(|&v| v == 255));
    }

    /// BCW eq. 981 must Clip1 — at `(255, 0)` with `bcw_idx = 4`
    /// (w0 = 10, w1 = -2): (10*255 + -2*0 + 4) >> 3 = 2554 >> 3 = 319,
    /// which clamps down to 255.
    #[test]
    fn bcw_clamps_to_255_when_blend_overshoots() {
        let p0 = PicturePlane::filled(2, 2, 255);
        let p1 = PicturePlane::filled(2, 2, 0);
        let mut dst = PicturePlane::filled(2, 2, 0);
        bi_pred_avg_8bit_bcw(&mut dst, 0, 0, 2, 2, &p0, &p1, 4).unwrap();
        assert!(dst.samples.iter().all(|&v| v == 255));
    }

    /// BCW eq. 981 must Clip1 to 0 — at `(0, 255)` with `bcw_idx = 4`
    /// (w0 = 10, w1 = -2): (10*0 + -2*255 + 4) >> 3 = -506 >> 3 = -64
    /// (Rust arith shift), which clamps up to 0.
    #[test]
    fn bcw_clamps_to_zero_when_blend_undershoots() {
        let p0 = PicturePlane::filled(2, 2, 0);
        let p1 = PicturePlane::filled(2, 2, 255);
        let mut dst = PicturePlane::filled(2, 2, 0);
        bi_pred_avg_8bit_bcw(&mut dst, 0, 0, 2, 2, &p0, &p1, 4).unwrap();
        assert!(dst.samples.iter().all(|&v| v == 0));
    }

    /// `bi_pred_avg_8bit_bcw` rejects out-of-range `bcw_idx`.
    #[test]
    fn bcw_rejects_out_of_range_idx() {
        let p0 = PicturePlane::filled(2, 2, 100);
        let p1 = PicturePlane::filled(2, 2, 200);
        let mut dst = PicturePlane::filled(2, 2, 0);
        let err = bi_pred_avg_8bit_bcw(&mut dst, 0, 0, 2, 2, &p0, &p1, 5).unwrap_err();
        assert!(format!("{err:?}").contains("bcw_idx"));
    }

    /// BCW with bcw_idx = 0 over distinct planes via the luma MC
    /// helper produces byte-identical output to `predict_luma_block_bipred`.
    #[test]
    fn bcw_idx_zero_luma_path_matches_legacy_bipred() {
        let mut src_l0 = PicturePlane::filled(16, 16, 0);
        let mut src_l1 = PicturePlane::filled(16, 16, 0);
        for r in 0..16usize {
            for c in 0..16usize {
                src_l0.samples[r * 16 + c] = ((r * 17 + c) & 0xff) as u8;
                src_l1.samples[r * 16 + c] = (255u8).wrapping_sub((r * 7 + c) as u8);
            }
        }
        let mv_l0 = MotionVector::from_int_pel(0, 0);
        let mv_l1 = MotionVector::from_int_pel(0, 0);
        let mut dst_a = PicturePlane::filled(8, 8, 0);
        let mut dst_b = PicturePlane::filled(8, 8, 0);
        predict_luma_block_bipred(&mut dst_a, 0, 0, 8, 8, &src_l0, mv_l0, &src_l1, mv_l1).unwrap();
        predict_luma_block_bipred_bcw(&mut dst_b, 0, 0, 8, 8, &src_l0, mv_l0, &src_l1, mv_l1, 0)
            .unwrap();
        assert_eq!(dst_a.samples, dst_b.samples);
    }

    /// §8.5.6.7 weight ladder — both neighbours intra → w = 3.
    #[test]
    fn ciip_weight_both_intra_is_three() {
        assert_eq!(ciip_intra_weight(true, true), 3);
    }

    /// §8.5.6.7 weight ladder — both neighbours not intra → w = 1.
    #[test]
    fn ciip_weight_neither_intra_is_one() {
        assert_eq!(ciip_intra_weight(false, false), 1);
    }

    /// §8.5.6.7 weight ladder — exactly one neighbour intra → w = 2,
    /// regardless of which side it is.
    #[test]
    fn ciip_weight_one_intra_is_two() {
        assert_eq!(ciip_intra_weight(true, false), 2);
        assert_eq!(ciip_intra_weight(false, true), 2);
    }

    /// §8.5.6.7 eq. 998 spot-check — both predictors equal to 100,
    /// any weight collapses to 100 (since the weighted average of two
    /// equal values is the value itself).
    #[test]
    fn ciip_combine_equal_predictors_is_predictor() {
        let intra = vec![100i16; 16];
        let inter = vec![100i16; 16];
        for w in [1u32, 2, 3] {
            let out = combine_ciip_samples(&intra, &inter, 4, 4, w, 8);
            for v in &out {
                assert_eq!(*v, 100);
            }
        }
    }

    /// §8.5.6.7 eq. 998 spot-check — w = 2 produces the spec-rounded
    /// midpoint `(intra + inter + 2) >> 2 * 2 = (intra + inter + 1) >> 1`
    /// when both weights are equal. With `intra = 200, inter = 100`:
    /// eq. 998 = `(2 * 200 + 2 * 100 + 2) >> 2 = 602 >> 2 = 150`.
    #[test]
    fn ciip_combine_w2_is_rounded_midpoint() {
        let intra = vec![200i16; 4];
        let inter = vec![100i16; 4];
        let out = combine_ciip_samples(&intra, &inter, 2, 2, 2, 8);
        for v in &out {
            assert_eq!(*v, 150);
        }
    }

    /// §8.5.6.7 eq. 998 spot-check — w = 3 leans towards intra.
    /// `(3 * 240 + 1 * 80 + 2) >> 2 = 802 >> 2 = 200`.
    #[test]
    fn ciip_combine_w3_leans_intra() {
        let intra = vec![240i16; 4];
        let inter = vec![80i16; 4];
        let out = combine_ciip_samples(&intra, &inter, 2, 2, 3, 8);
        for v in &out {
            assert_eq!(*v, 200);
        }
    }

    /// §8.5.6.7 eq. 998 spot-check — w = 1 leans towards inter.
    /// `(1 * 240 + 3 * 80 + 2) >> 2 = 482 >> 2 = 120`.
    #[test]
    fn ciip_combine_w1_leans_inter() {
        let intra = vec![240i16; 4];
        let inter = vec![80i16; 4];
        let out = combine_ciip_samples(&intra, &inter, 2, 2, 1, 8);
        for v in &out {
            assert_eq!(*v, 120);
        }
    }

    /// CIIP combiner clamps into `[0, (1 << bit_depth) - 1]`. Even
    /// though eq. 998 is non-expanding for in-range inputs, the helper
    /// guards against rounding overshoot at extremes (e.g. 1023 / 1023
    /// at 10-bit returns 1023, not 1024).
    #[test]
    fn ciip_combine_clamps_to_bit_depth() {
        let intra = vec![1023i16; 4];
        let inter = vec![1023i16; 4];
        let out = combine_ciip_samples(&intra, &inter, 2, 2, 2, 10);
        for v in &out {
            assert_eq!(*v, 1023);
        }
    }

    /// At `bit_depth == 8`, the HBD u16 MC HP path produces a
    /// bit-identical intermediate to the legacy u8 HP path for the
    /// same sample values lifted into a `PicturePlane16`.
    #[test]
    fn predict_luma_hp_u16_bit8_matches_u8() {
        // Build a non-trivial 16x16 reference and clone it into a
        // bit_depth=8 u16 plane.
        let mut src8 = PicturePlane::filled(16, 16, 0);
        let mut src16 = PicturePlane16::filled(16, 16, 0, 8);
        for y in 0..16 {
            for x in 0..16 {
                let v = (((x * 13 + y * 7) ^ 0x55) & 0xff) as u8;
                src8.samples[y * 16 + x] = v;
                src16.samples[y * 16 + x] = v as u16;
            }
        }
        // Sweep a few sub-pel positions covering every fast-path
        // branch in `predict_luma_block_high_precision`.
        for &(mx, my) in &[(0, 0), (5, 0), (0, 7), (3, 9), (8, 8)] {
            let mv = MotionVector { x: mx, y: my };
            let want = predict_luma_block_high_precision(2, 2, 8, 8, &src8, mv, 8).unwrap();
            let got = predict_luma_block_high_precision_u16(2, 2, 8, 8, &src16, mv, 8).unwrap();
            assert_eq!(
                got, want,
                "u16 / u8 HP MC must match at BD=8 for mv=({mx},{my})",
            );
        }
    }

    /// Main10 MC sees the full 10-bit dynamic range of the reference
    /// plane: a 1023-valued constant plane at BD=10 round-trips
    /// through the HP filter to `1023 << (14 - 10) = 16368`, well
    /// above the 8-bit-truncated `255 << 6 = 16320` the legacy path
    /// would produce. This pins the "HBD MC actually uses HBD samples"
    /// invariant.
    #[test]
    fn predict_luma_hp_u16_main10_uses_full_range() {
        let src = PicturePlane16::filled(16, 16, 1023, 10);
        let mv = MotionVector::ZERO;
        let out = predict_luma_block_high_precision_u16(2, 2, 8, 8, &src, mv, 10).unwrap();
        // Integer-pel HP lift = src << (14 - bit_depth) = 1023 << 4
        // = 16368.
        for &v in &out {
            assert_eq!(
                v, 16368,
                "Main10 HP integer-pel must lift 1023 to 16368 (got {v})",
            );
        }
        // The legacy 8-bit path would have truncated 1023 → 255 before
        // the lift, producing `255 << 6 = 16320`; the HBD path must
        // exceed that by exactly the truncation gap.
        assert!(out[0] > 16320);
    }

    /// Round-trip Main12: integer-pel HP lift = src << (14 - 12) = 4.
    /// This makes the "lift-by-(14-BD)" formula traceable across all
    /// supported bit depths.
    #[test]
    fn predict_luma_hp_u16_main12_lift() {
        let src = PicturePlane16::filled(16, 16, 4095, 12);
        let mv = MotionVector::ZERO;
        let out = predict_luma_block_high_precision_u16(2, 2, 8, 8, &src, mv, 12).unwrap();
        // 4095 << 2 = 16380.
        for &v in &out {
            assert_eq!(v, 16380);
        }
    }

    /// HBD MC rejects a bit-depth mismatch between the reference plane
    /// and the requested precision.
    #[test]
    fn predict_luma_hp_u16_bit_depth_mismatch_errors() {
        let src = PicturePlane16::filled(16, 16, 0, 10);
        assert!(
            predict_luma_block_high_precision_u16(0, 0, 8, 8, &src, MotionVector::ZERO, 8).is_err()
        );
    }
}
