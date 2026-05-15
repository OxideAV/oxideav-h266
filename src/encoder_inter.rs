//! Round-58 — VVC P-slice encoder scaffold.
//! Round-59 — extends round 58 with sub-pel motion compensation (¼-pel
//! refinement after the integer-pel search).
//!
//! This module provides a **minimum-viable** P-slice encode + roundtrip
//! decode path that closes the round-57 "inter-frame P-slice pipeline"
//! lacks tail. It is a scaffold:
//!
//! * **One reference picture** in the DPB (the previous decoded frame).
//! * **Integer-pel motion search** (round 58, full-search ±N px window,
//!   SAD cost on luma 4×4 reference blocks — VVC §7.4.10 minimum-PU size).
//! * **Sub-pel refinement** (round 59) on top of the integer-pel result:
//!   probe the 8 half-pel positions around the best integer-pel candidate
//!   then the 8 quarter-pel positions around the best half-pel candidate.
//!   This achieves 1/4-pel (4-unit-of-1/16-pel) precision on the wire
//!   without paying for a full 1/16-pel exhaustive search. The MC step
//!   itself runs at the spec's full 1/16-pel granularity through the
//!   existing [`crate::inter::predict_luma_block`] (VVC §8.5.6.3.2,
//!   Table 27 8-tap luma filter `hpelIfIdx == 0`).
//! * **`PRED_L0` only** — uni-prediction from L0[0]. Bi-pred and B-slice
//!   pipeline is out of scope.
//! * **Spatial MVP** uses the `left` candidate when available, else the
//!   `above` candidate, else zero (§7.4.7.3 minimum).
//! * **Residual emit** reuses the existing forward-DCT + flat quant +
//!   CABAC three-pass coefficient writer from
//!   [`crate::residual_enc::encode_tb_coefficients`]. The decoder side
//!   uses [`crate::residual::decode_tb_coefficients`] / inverse DCT /
//!   reconstruction-clip from [`crate::dequant`] +
//!   [`crate::transform::inverse_transform_2d`].
//!
//! ## Motion-vector representation
//!
//! Internally MVs are stored in **1/16-luma-sample** units to match the
//! spec's §8.5.2 fractional accuracy. Integer-pel `(dx, dy)` is therefore
//! `(dx << 4, dy << 4)`; ¼-pel is `±4`; half-pel is `±8`. The `MvdLX`
//! emitted on the wire (§7.4.7.2) is the difference `mv - mvp` in these
//! same 1/16-pel units. AMVR (resolution selection) is deferred.
//!
//! ## Wire format (in-crate)
//!
//! Conceptually the wire follows the spec's slice-header → CU-syntax →
//! residual chain (same shape as round 58; the only delta in round 59
//! is that the MVD components now carry sub-pel magnitudes when the
//! refinement picks a fractional MV).

use oxideav_core::{Error, Result};

use crate::cabac::{ArithDecoder, ContextModel};
use crate::cabac_enc::ArithEncoder;
use crate::dequant::{dequantize_tb_flat, DequantParams};
use crate::inter::{predict_chroma_block, predict_luma_block, MotionVector};
use crate::reconstruct::{PictureBuffer, PicturePlane};
use crate::residual::{read_tu_y_coded_flag, ResidualCtxs};
use crate::residual_enc::{encode_tb_coefficients, write_tu_y_coded_flag};
use crate::tables::{init_contexts, SyntaxCtx};
use crate::transform::{inverse_transform_2d, TrType};
use crate::transform_fwd::{forward_dct_ii_2d, quantize_tb_flat};

// =====================================================================
// Magic / header constants
// =====================================================================

/// Magic prefix identifying an in-crate VVC P-slice payload built by
/// [`encode_p_slice`] and consumable by [`decode_p_slice`].
pub const PSLICE_MAGIC: &[u8; 14] = b"OXAV_VVC_PSLIC";

/// Magic prefix identifying an in-crate VVC B-slice payload built by
/// [`encode_b_slice`] and consumable by [`decode_b_slice`].
pub const BSLICE_MAGIC: &[u8; 14] = b"OXAV_VVC_BSLIC";

/// Block size in luma samples for the inter scaffold. VVC's
/// minimum inter PU is 4×4. We use `nTbS == 4` here because the
/// existing per-TB residual emit/decode pair (`encode_tb_coefficients`
/// / `decode_tb_coefficients`) is most heavily round-trip tested at
/// 4×4 (see `crate::residual_enc::tests::encode_decode_4x4_*`); the
/// larger 16×16 / 32×32 sizes have known scatter-coefficient
/// fragility that is out of scope for the round-58 scaffold to chase.
pub const INTER_BLOCK_W: usize = 4;
pub const INTER_BLOCK_H: usize = 4;

/// Default integer-pel search range (±N samples around the predicted
/// motion vector). 8 is the round-58 baseline.
pub const DEFAULT_SEARCH_RANGE: i32 = 8;

/// Context-init slice-QP for this scaffold. Shared by encoder + decoder.
const SCAFFOLD_SLICE_QP: i32 = 26;

/// Sub-pel encoding step in 1/16-luma-sample units. 8 = half-pel,
/// 4 = quarter-pel, 1 = 1/16-pel.
const HALF_PEL_STEP: i32 = 8;
const QUARTER_PEL_STEP: i32 = 4;

/// Round-62 — maximum reference pictures per list. VVC §A.4 Main 10
/// profile permits up to 15 active references per list; mainstream
/// profiles typically use 1..4. The scaffold caps at 4 to keep the
/// encoder ME RDO loop bounded; the wire-side truncated-unary encoding
/// scales to whatever active count the slice header advertises so the
/// constant is the encoder/test ceiling, not a wire-format limit.
pub const MAX_REF_PICS: usize = 4;

// =====================================================================
// PreparedCu — round-58 sibling of encoder_pipeline::PreparedCu
// =====================================================================

/// One inter-CU's prepared state, mirroring the round-57
/// [`crate::encoder_pipeline`] internal `PreparedCu` style. The round-58
/// `InterPSlice` variant carries everything the second-pass CABAC walk
/// needs to emit the wire-side syntax: the L0 reference index, the
/// motion vector (1/16-pel units), and the quantised luma residual
/// levels.
///
/// Only the `InterPSlice` variant is ever constructed in this module;
/// the variant exists alongside the round-57 leaf / BT / TT shapes in
/// the spirit of "every tree walker must learn the new variant".
#[derive(Clone, Debug)]
pub enum PreparedCu {
    /// P-slice inter CU: L0 ref + MV (1/16-pel units) + residual.
    InterPSlice {
        /// L0 reference index. The scaffold only emits 0 (single-ref
        /// DPB), but the field is kept explicit for spec parity.
        ref_idx: u8,
        /// Motion vector `(mv_x, mv_y)` in 1/16-luma-sample units.
        mv: (i32, i32),
        /// Spatial MV predictor used for `mvd = mv - mvp` (1/16-pel).
        mvp: (i32, i32),
        /// Luma TB width / height (always `INTER_BLOCK_W` /
        /// `INTER_BLOCK_H` in this scaffold).
        n_tb_w: usize,
        n_tb_h: usize,
        /// Quantised luma residual levels (length `n_tb_w * n_tb_h`).
        /// Empty when `cbf_y == 0`.
        levels: Vec<i32>,
        /// Whether this CU has a non-zero luma CBF on the wire.
        cbf_y: bool,
    },
    /// Round-60 — B-slice inter CU. Carries `inter_pred_idc` selecting
    /// uni-pred L0 / uni-pred L1 / bi-pred plus per-list MVs and the
    /// luma residual. Per-list MV / MVP are stored in 1/16-pel units.
    InterBSlice {
        /// `inter_pred_idc` per §7.4.7.2 (PRED_L0 / PRED_L1 / PRED_BI).
        inter_pred_idc: InterPredIdc,
        /// L0 reference index — 0 in the single-pic-per-list scaffold.
        ref_idx_l0: u8,
        /// L1 reference index — 0 in the single-pic-per-list scaffold.
        ref_idx_l1: u8,
        /// L0 motion vector (1/16-pel). Inferred zero when not active.
        mv_l0: (i32, i32),
        /// L1 motion vector (1/16-pel). Inferred zero when not active.
        mv_l1: (i32, i32),
        /// L0 spatial MV predictor (1/16-pel).
        mvp_l0: (i32, i32),
        /// L1 spatial MV predictor (1/16-pel).
        mvp_l1: (i32, i32),
        /// Luma TB width / height.
        n_tb_w: usize,
        n_tb_h: usize,
        /// Quantised luma residual levels; empty when `cbf_y == 0`.
        levels: Vec<i32>,
        /// `cbf_y`.
        cbf_y: bool,
    },
}

/// `inter_pred_idc` per VVC §7.4.7.2 — bi-prediction selector.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InterPredIdc {
    /// Uni-pred from L0 (`inter_pred_idc == PRED_L0`).
    L0 = 0,
    /// Uni-pred from L1 (`inter_pred_idc == PRED_L1`).
    L1 = 1,
    /// Bi-pred — both L0 and L1 active (`inter_pred_idc == PRED_BI`).
    Bi = 2,
}

// =====================================================================
// Block-matching motion search — full-search SAD on luma
// =====================================================================

/// Sum of absolute differences between a `w × h` block at `(cx, cy)` in
/// `curr` and the same-size block at `(cx + mv_x, cy + mv_y)` in `ref_p`
/// when `(mv_x, mv_y)` is in **integer** luma samples. Out-of-bound
/// reference samples are clamped to the picture edge.
fn sad_block(
    curr: &PicturePlane,
    cx: usize,
    cy: usize,
    w: usize,
    h: usize,
    ref_p: &PicturePlane,
    mv_x: i32,
    mv_y: i32,
) -> u32 {
    let mut sad: u32 = 0;
    let rw = ref_p.width as i32;
    let rh = ref_p.height as i32;
    for r in 0..h {
        let cy_r = cy + r;
        let ry = (cy as i32 + r as i32 + mv_y).clamp(0, rh - 1) as usize;
        for c in 0..w {
            let cx_c = cx + c;
            let rx = (cx as i32 + c as i32 + mv_x).clamp(0, rw - 1) as usize;
            let cur_s = curr.samples[cy_r * curr.stride + cx_c] as i32;
            let ref_s = ref_p.samples[ry * ref_p.stride + rx] as i32;
            sad += (cur_s - ref_s).unsigned_abs();
        }
    }
    sad
}

/// SAD between the `w × h` block at `(cx, cy)` in `curr` and the sub-pel
/// MC prediction from `ref_p` shifted by `mv_q16` (1/16-luma-sample
/// units). The MC prediction is built into a small scratch plane via
/// [`predict_luma_block`].
fn sad_block_subpel(
    curr: &PicturePlane,
    cx: usize,
    cy: usize,
    w: usize,
    h: usize,
    ref_p: &PicturePlane,
    mv_q16: (i32, i32),
) -> Result<u32> {
    let mut scratch = PicturePlane::filled(w, h, 0);
    let mv = MotionVector {
        x: mv_q16.0,
        y: mv_q16.1,
    };
    predict_luma_block(
        &mut scratch,
        0,
        0,
        w as u32,
        h as u32,
        ref_p,
        // predict_luma_block computes source position as
        // `(dst_x + (mv.x >> 4), dst_y + (mv.y >> 4))` — but our
        // destination is at scratch origin (0, 0), so we have to fold
        // the source origin `(cx, cy)` into the MV.
        MotionVector {
            x: mv.x + (cx as i32) * 16,
            y: mv.y + (cy as i32) * 16,
        },
    )?;
    let mut sad: u32 = 0;
    for r in 0..h {
        for c in 0..w {
            let cur_s = curr.samples[(cy + r) * curr.stride + (cx + c)] as i32;
            let p = scratch.samples[r * scratch.stride + c] as i32;
            sad += (cur_s - p).unsigned_abs();
        }
    }
    Ok(sad)
}

/// Full-search integer-pel motion estimation. Returns the integer
/// motion vector `(mv_x, mv_y)` in luma samples that minimises SAD
/// against `ref_p` in a `±range` window, plus the achieved SAD.
///
/// `(cx, cy)` is the top-left luma sample of the current block; `(w, h)`
/// is the block size. The search centre is the supplied `mvp` (integer
/// pel) so the search window is `[mvp_x - range, mvp_x + range]`.
pub fn full_search_int(
    curr: &PicturePlane,
    cx: usize,
    cy: usize,
    w: usize,
    h: usize,
    ref_p: &PicturePlane,
    mvp: (i16, i16),
    range: i32,
) -> ((i16, i16), u32) {
    let mut best_mv = (mvp.0, mvp.1);
    let mut best_sad = sad_block(curr, cx, cy, w, h, ref_p, mvp.0 as i32, mvp.1 as i32);
    for dy in -range..=range {
        for dx in -range..=range {
            let cand_x = mvp.0 as i32 + dx;
            let cand_y = mvp.1 as i32 + dy;
            if cand_x < i16::MIN as i32 || cand_x > i16::MAX as i32 {
                continue;
            }
            if cand_y < i16::MIN as i32 || cand_y > i16::MAX as i32 {
                continue;
            }
            let s = sad_block(curr, cx, cy, w, h, ref_p, cand_x, cand_y);
            if s < best_sad {
                best_sad = s;
                best_mv = (cand_x as i16, cand_y as i16);
            }
        }
    }
    (best_mv, best_sad)
}

/// Round-59 — sub-pel refinement. Starting from the integer-pel best
/// (`mv_int_pel` in luma samples), probe the 8 half-pel neighbours
/// (±8 in 1/16-pel units along x/y) and pick the best. Then probe the
/// 8 quarter-pel neighbours (±4) around that best half-pel candidate.
/// All sub-pel candidates are evaluated with the spec 8-tap luma
/// interpolation via [`predict_luma_block`].
///
/// Returns the refined MV in **1/16-luma-sample** units plus its SAD.
pub fn refine_subpel(
    curr: &PicturePlane,
    cx: usize,
    cy: usize,
    w: usize,
    h: usize,
    ref_p: &PicturePlane,
    mv_int_pel: (i16, i16),
    int_pel_sad: u32,
) -> Result<((i32, i32), u32)> {
    // Convert to 1/16-pel units.
    let mut best_mv: (i32, i32) = (mv_int_pel.0 as i32 * 16, mv_int_pel.1 as i32 * 16);
    let mut best_sad = int_pel_sad;

    // 8 half-pel offsets (excluding the integer-pel centre).
    const HALF_OFFSETS: [(i32, i32); 8] = [
        (-HALF_PEL_STEP, -HALF_PEL_STEP),
        (0, -HALF_PEL_STEP),
        (HALF_PEL_STEP, -HALF_PEL_STEP),
        (-HALF_PEL_STEP, 0),
        (HALF_PEL_STEP, 0),
        (-HALF_PEL_STEP, HALF_PEL_STEP),
        (0, HALF_PEL_STEP),
        (HALF_PEL_STEP, HALF_PEL_STEP),
    ];
    for (dx, dy) in HALF_OFFSETS.iter() {
        let cand = (best_mv.0 + dx, best_mv.1 + dy);
        let s = sad_block_subpel(curr, cx, cy, w, h, ref_p, cand)?;
        if s < best_sad {
            best_sad = s;
            best_mv = cand;
        }
    }

    // 8 quarter-pel offsets around the (possibly refined) best.
    const QUARTER_OFFSETS: [(i32, i32); 8] = [
        (-QUARTER_PEL_STEP, -QUARTER_PEL_STEP),
        (0, -QUARTER_PEL_STEP),
        (QUARTER_PEL_STEP, -QUARTER_PEL_STEP),
        (-QUARTER_PEL_STEP, 0),
        (QUARTER_PEL_STEP, 0),
        (-QUARTER_PEL_STEP, QUARTER_PEL_STEP),
        (0, QUARTER_PEL_STEP),
        (QUARTER_PEL_STEP, QUARTER_PEL_STEP),
    ];
    for (dx, dy) in QUARTER_OFFSETS.iter() {
        let cand = (best_mv.0 + dx, best_mv.1 + dy);
        let s = sad_block_subpel(curr, cx, cy, w, h, ref_p, cand)?;
        if s < best_sad {
            best_sad = s;
            best_mv = cand;
        }
    }

    Ok((best_mv, best_sad))
}

// =====================================================================
// MVP — minimal §7.4.7.3 spatial MVP picker
// =====================================================================

/// Picture-wide grid of per-block MVs in **1/16-luma-sample** units.
/// Used for the minimal §7.4.7.3 spatial MVP derivation: when filling
/// block `(bx, by)`, the predictor is the left neighbour's MV when
/// `bx > 0`, else the above neighbour's MV when `by > 0`, else zero.
///
/// `cells[by * cols + bx]` holds the MV of the block whose
/// top-left luma sample is `(bx * INTER_BLOCK_W, by * INTER_BLOCK_H)`.
#[derive(Clone, Debug)]
pub struct MvField {
    pub cols: usize,
    pub rows: usize,
    pub cells: Vec<(i32, i32)>,
}

impl MvField {
    pub fn new(cols: usize, rows: usize) -> Self {
        Self {
            cols,
            rows,
            cells: vec![(0, 0); cols * rows],
        }
    }

    /// Spatial MVP per the round-58 minimal rule.
    pub fn mvp_for(&self, bx: usize, by: usize) -> (i32, i32) {
        if bx > 0 {
            self.cells[by * self.cols + (bx - 1)]
        } else if by > 0 {
            self.cells[(by - 1) * self.cols + bx]
        } else {
            (0, 0)
        }
    }

    pub fn set(&mut self, bx: usize, by: usize, mv: (i32, i32)) {
        self.cells[by * self.cols + bx] = mv;
    }
}

// =====================================================================
// Motion compensation — integer-pel sample copy (legacy round-58 path)
// =====================================================================

/// Predict a `w × h` luma block at `(dx, dy)` in `dst` from `ref_p`
/// shifted by integer-pel motion vector `(mv_x, mv_y)`. Reference
/// samples outside the picture are clamped to the nearest picture edge
/// (matches the spec's `Clip3(0, picW - 1, ...)` for the no-subpic
/// no-wrap case). Used only by tests still on the integer-pel API; the
/// main encode + decode walk now uses [`predict_luma_block`] which
/// handles both integer-pel and sub-pel MVs through the spec 8-tap
/// luma filter (§8.5.6.3.2 Table 27, `hpelIfIdx == 0`).
pub fn mc_predict_int(
    dst: &mut PicturePlane,
    dx: usize,
    dy: usize,
    w: usize,
    h: usize,
    ref_p: &PicturePlane,
    mv_x: i16,
    mv_y: i16,
) {
    let rw = ref_p.width as i32;
    let rh = ref_p.height as i32;
    for r in 0..h {
        let dy_r = dy + r;
        let ry = (dy as i32 + r as i32 + mv_y as i32).clamp(0, rh - 1) as usize;
        for c in 0..w {
            let dx_c = dx + c;
            let rx = (dx as i32 + c as i32 + mv_x as i32).clamp(0, rw - 1) as usize;
            dst.samples[dy_r * dst.stride + dx_c] = ref_p.samples[ry * ref_p.stride + rx];
        }
    }
}

/// Predict a `w × h` luma block from `ref_p` at integer **or** sub-pel
/// MV (1/16-pel units) through the spec §8.5.6.3 luma interpolation
/// filter. Writes the prediction into a `w*h` row-major buffer.
fn mc_predict_subpel(
    pred: &mut [u8],
    ref_p: &PicturePlane,
    cx: usize,
    cy: usize,
    w: usize,
    h: usize,
    mv_q16: (i32, i32),
) -> Result<()> {
    debug_assert_eq!(pred.len(), w * h);
    let mut scratch = PicturePlane::filled(w, h, 0);
    predict_luma_block(
        &mut scratch,
        0,
        0,
        w as u32,
        h as u32,
        ref_p,
        MotionVector {
            x: mv_q16.0 + (cx as i32) * 16,
            y: mv_q16.1 + (cy as i32) * 16,
        },
    )?;
    for r in 0..h {
        for c in 0..w {
            pred[r * w + c] = scratch.samples[r * scratch.stride + c];
        }
    }
    Ok(())
}

/// Round-63 — chroma 4-tap sub-pel motion-compensated prediction for one
/// inter block (4:2:0, single chroma component).
///
/// Mirrors [`mc_predict_subpel`] but invokes the §8.5.6.3.4 4-tap chroma
/// interpolation filter (Table 28) via [`predict_chroma_block`]. Inputs:
///
///   * `pred_c` — output buffer, `w_c * h_c` chroma samples (row-major).
///   * `ref_c` — reference chroma plane (Cb or Cr).
///   * `cx_c` / `cy_c` — destination position in **chroma** samples.
///   * `w_c` / `h_c` — chroma block dimensions (luma `4 → 2` for 4:2:0).
///   * `mv_q16` — the **luma-domain** 1/16-pel MV. Per §8.5.6.3.4 the
///     same MV is reused for chroma; the 4:2:0 mapping doubles the
///     effective fractional resolution to 1/32 chroma samples
///     (`mv >> 5` integer chroma offset, `mv & 31` chroma frac index).
///     [`predict_chroma_block`] handles the conversion internally.
fn mc_predict_chroma_subpel(
    pred_c: &mut [u8],
    ref_c: &PicturePlane,
    cx_c: usize,
    cy_c: usize,
    w_c: usize,
    h_c: usize,
    mv_q16: (i32, i32),
) -> Result<()> {
    debug_assert_eq!(pred_c.len(), w_c * h_c);
    let mut scratch = PicturePlane::filled(w_c, h_c, 0);
    // predict_chroma_block computes the integer chroma source as
    // `(dst_x_c + (mv >> 5), dst_y_c + (mv >> 5))`. Our destination is
    // at scratch origin (0, 0), so fold the chroma source origin
    // `(cx_c, cy_c)` into the MV — but the MV is in luma-1/16 units
    // and the scaling to chroma-1/32 happens inside the helper, so we
    // pre-multiply each component by 2 (`<< 1`) to convert the chroma-
    // pixel offset into the luma-1/16 equivalent that the helper expects.
    predict_chroma_block(
        &mut scratch,
        0,
        0,
        w_c as u32,
        h_c as u32,
        ref_c,
        MotionVector {
            x: mv_q16.0 + ((cx_c as i32) << 5),
            y: mv_q16.1 + ((cy_c as i32) << 5),
        },
    )?;
    for r in 0..h_c {
        for c in 0..w_c {
            pred_c[r * w_c + c] = scratch.samples[r * scratch.stride + c];
        }
    }
    Ok(())
}

/// Round-63 — chroma 4-tap sub-pel BI prediction for one inter block.
/// Computes the per-list chroma predictions through
/// [`mc_predict_chroma_subpel`] and averages them per §8.5.6.4
/// (`(p0 + p1 + 1) >> 1`). Returns a fresh `w_c * h_c` buffer.
fn mc_predict_chroma_subpel_bi(
    ref_c_l0: &PicturePlane,
    mv_l0_q16: (i32, i32),
    ref_c_l1: &PicturePlane,
    mv_l1_q16: (i32, i32),
    cx_c: usize,
    cy_c: usize,
    w_c: usize,
    h_c: usize,
) -> Result<Vec<u8>> {
    let mut p0 = vec![0u8; w_c * h_c];
    let mut p1 = vec![0u8; w_c * h_c];
    mc_predict_chroma_subpel(&mut p0, ref_c_l0, cx_c, cy_c, w_c, h_c, mv_l0_q16)?;
    mc_predict_chroma_subpel(&mut p1, ref_c_l1, cx_c, cy_c, w_c, h_c, mv_l1_q16)?;
    let mut out = vec![0u8; w_c * h_c];
    for i in 0..w_c * h_c {
        out[i] = (((p0[i] as u16) + (p1[i] as u16) + 1) >> 1) as u8;
    }
    Ok(out)
}

// =====================================================================
// MVD coding — §7.4.7.2 / §9.3.3.7 (round-58 scaffold form)
// =====================================================================
//
// VVC's full mvd_coding() emits abs_mvd_greater0_flag + greater1_flag
// (CABAC-coded) + abs_mvd_minus2 (bypass EG1) + mvd_sign_flag (bypass).
// For the scaffold we collapse to a compact bypass-only form using
// exp-Golomb of order 1 for the magnitude (matches the spec's
// `abs_mvd_minus2 + EG1` shape, just with all bins bypass-coded so we
// don't have to thread two more contexts):
//   - 1 bypass bit `abs_zero_flag`. If 0, mvd is 0 (no further bins).
//   - else: EG1(abs - 1) bypass-coded, then 1 bypass `mvd_sign_flag`.
// This keeps the CABAC state shared with the residual stream while
// staying compact even for large |mvd|. Zero-cost when MV is exactly
// equal to the predictor (the common case after spatial MVP picks the
// neighbour's MV). Round 59: the same EG-1 bypass shape now carries
// sub-pel magnitudes (a ¼-pel MV → 4-unit absolute value, half-pel →
// 8, full-pel → 16, etc.), so on the wire the only delta is the
// magnitude payload — the CABAC schema is unchanged.

fn encode_eg_k(enc: &mut ArithEncoder, value: u32, k: u32) -> Result<()> {
    // Standard exp-Golomb of order k, bypass-coded.
    let mut v = value;
    let mut k = k;
    // Find the unary prefix length: smallest m such that
    // v < (1 << k) * (2^(m+1) - 1) -- equivalently, repeatedly emit a
    // 1-bit while v >= (1 << k), subtracting (1 << k) and bumping k.
    loop {
        let thresh = 1u32 << k;
        if v < thresh {
            break;
        }
        enc.encode_bypass(1)?;
        v -= thresh;
        k += 1;
    }
    enc.encode_bypass(0)?;
    // Emit the k LSBs of v (LSB first packing matches the decoder
    // below: we read MSB-first by `for i in (0..k).rev()`).
    for i in (0..k).rev() {
        enc.encode_bypass((v >> i) & 1)?;
    }
    Ok(())
}

fn decode_eg_k(dec: &mut ArithDecoder<'_>, k: u32) -> Result<u32> {
    let mut k = k;
    let mut value: u32 = 0;
    loop {
        let b = dec.decode_bypass()?;
        if b == 0 {
            break;
        }
        value += 1u32 << k;
        k += 1;
        // Defensive cap to avoid runaway decode on a corrupt stream.
        if k > 31 {
            return Err(Error::invalid("h266 P-slice EG decode: k overflow"));
        }
    }
    let mut tail: u32 = 0;
    for _ in 0..k {
        let b = dec.decode_bypass()?;
        tail = (tail << 1) | b;
    }
    Ok(value + tail)
}

fn encode_mvd_component(enc: &mut ArithEncoder, mvd: i32) -> Result<()> {
    if mvd == 0 {
        enc.encode_bypass(0)?;
        return Ok(());
    }
    enc.encode_bypass(1)?;
    let abs = mvd.unsigned_abs();
    encode_eg_k(enc, abs - 1, 1)?;
    enc.encode_bypass(if mvd < 0 { 1 } else { 0 })?;
    Ok(())
}

fn decode_mvd_component(dec: &mut ArithDecoder<'_>) -> Result<i32> {
    let zero = dec.decode_bypass()?;
    if zero == 0 {
        return Ok(0);
    }
    let abs = decode_eg_k(dec, 1)? + 1;
    let sign = dec.decode_bypass()?;
    let signed = abs as i32;
    Ok(if sign == 1 { -signed } else { signed })
}

// =====================================================================
// ref_idx_lX coding — round-62 truncated unary, §9.3.3.7
// =====================================================================
//
// VVC §9.3.3.7 / Table 132 codes `ref_idx_lX` as a truncated-unary
// binarization with `cMax = NumRefIdxActive[X] - 1`. The first bin is
// context-coded (two contexts indexed by `binIdx == 0 ? 0 : 1`); the
// remainder are bypass. For the round-62 scaffold we collapse the
// whole chain to bypass coding (matching the round-58/60 mvd schema's
// "all bypass for the magnitude" pattern). The shape on the wire is:
//
//   - if `num_active <= 1`: zero bins emitted (decoder infers 0).
//   - else: bins of value `min(ref_idx, num_active - 1)` zeros,
//     terminated by a single `1` bit when `ref_idx < num_active - 1`,
//     or by reaching `num_active - 1` zeros (no terminator needed).
//
// In other words: emit `ref_idx` zeros up to a cap of `num_active - 1`,
// then a `1` unless we hit the cap.

fn encode_ref_idx(enc: &mut ArithEncoder, ref_idx: u8, num_active: usize) -> Result<()> {
    if num_active <= 1 {
        // ref_idx is inferred to 0; nothing on the wire.
        return Ok(());
    }
    let cap = num_active - 1;
    let v = (ref_idx as usize).min(cap);
    for _ in 0..v {
        enc.encode_bypass(0)?; // unary "still bigger"
    }
    if v < cap {
        enc.encode_bypass(1)?; // terminator
    }
    Ok(())
}

fn decode_ref_idx(dec: &mut ArithDecoder<'_>, num_active: usize) -> Result<u8> {
    if num_active <= 1 {
        return Ok(0);
    }
    let cap = num_active - 1;
    let mut v: usize = 0;
    while v < cap {
        let b = dec.decode_bypass()?;
        if b == 1 {
            break;
        }
        v += 1;
    }
    Ok(v as u8)
}

// =====================================================================
// Per-block residual emit / decode — luma TB only
// =====================================================================

/// Forward-DCT + flat-quant + dequant + IDCT round-trip for one luma
/// TB. Returns:
///   - the quantised levels (for CABAC emit), and
///   - the reconstructed `pred + dequant_residual` clipped to `[0, 255]`
///     (so the encoder's reference frame for *next* P-slice matches
///     what the decoder will reconstruct).
fn prepare_inter_tb(
    src: &PicturePlane,
    pred: &[u8],
    cx: usize,
    cy: usize,
    n_tb_w: usize,
    n_tb_h: usize,
    qp: i32,
) -> Result<(Vec<i32>, Vec<u8>)> {
    debug_assert_eq!(pred.len(), n_tb_w * n_tb_h);
    let mut residual = vec![0i32; n_tb_w * n_tb_h];
    for ty in 0..n_tb_h {
        for tx in 0..n_tb_w {
            let s = src.samples[(cy + ty) * src.stride + (cx + tx)] as i32;
            let p = pred[ty * n_tb_w + tx] as i32;
            residual[ty * n_tb_w + tx] = s - p;
        }
    }
    let coeffs = forward_dct_ii_2d(n_tb_w, n_tb_h, &residual, 8)?;
    let levels = quantize_tb_flat(&coeffs, n_tb_w as u32, n_tb_h as u32, qp, 8, 15)?;
    let any_nz = levels.iter().any(|&l| l != 0);
    let recon = if any_nz {
        // Mirror the decoder side: dequantise → inverse transform →
        // pred + residual → clip.
        let dq = DequantParams::luma_8bit(n_tb_w as u32, n_tb_h as u32, qp);
        let d = dequantize_tb_flat(&levels, &dq)?;
        let r = inverse_transform_2d(
            n_tb_w,
            n_tb_h,
            n_tb_w,
            n_tb_h,
            TrType::DctII,
            TrType::DctII,
            &d,
            8,
            15,
        )?;
        let mut recon = vec![0u8; n_tb_w * n_tb_h];
        for ty in 0..n_tb_h {
            for tx in 0..n_tb_w {
                let p = pred[ty * n_tb_w + tx] as i32;
                let v = (p + r[ty * n_tb_w + tx]).clamp(0, 255) as u8;
                recon[ty * n_tb_w + tx] = v;
            }
        }
        recon
    } else {
        // All-zero levels — reconstruction is the prediction.
        pred.to_vec()
    };
    Ok((levels, recon))
}

/// Decoder-side inverse of [`prepare_inter_tb`]: reads the quantised
/// levels off the wire, dequantises + inverse-transforms, then adds to
/// `pred` and writes the reconstructed samples into `out_block`.
fn reconstruct_inter_tb_from_levels(
    levels: &[i32],
    pred: &[u8],
    n_tb_w: usize,
    n_tb_h: usize,
    qp: i32,
) -> Result<Vec<u8>> {
    debug_assert_eq!(pred.len(), n_tb_w * n_tb_h);
    if levels.iter().all(|&l| l == 0) {
        return Ok(pred.to_vec());
    }
    let dq = DequantParams::luma_8bit(n_tb_w as u32, n_tb_h as u32, qp);
    let d = dequantize_tb_flat(levels, &dq)?;
    let r = inverse_transform_2d(
        n_tb_w,
        n_tb_h,
        n_tb_w,
        n_tb_h,
        TrType::DctII,
        TrType::DctII,
        &d,
        8,
        15,
    )?;
    let mut out = vec![0u8; n_tb_w * n_tb_h];
    for ty in 0..n_tb_h {
        for tx in 0..n_tb_w {
            let p = pred[ty * n_tb_w + tx] as i32;
            let v = (p + r[ty * n_tb_w + tx]).clamp(0, 255) as u8;
            out[ty * n_tb_w + tx] = v;
        }
    }
    Ok(out)
}

// =====================================================================
// Per-block CABAC contexts — cu_skip_flag + merge_flag (round-58 minimal)
// =====================================================================

/// Round-58 P-slice context bundle — extends [`ResidualCtxs`] with the
/// `cu_skip_flag` and `merge_flag` ctxs the inter walker needs.
pub struct PSliceCtxs {
    pub residual: ResidualCtxs,
    pub cu_skip: Vec<ContextModel>,
    pub merge_flag: Vec<ContextModel>,
}

impl PSliceCtxs {
    pub fn init(slice_qp_y: i32) -> Self {
        Self {
            residual: ResidualCtxs::init(slice_qp_y),
            cu_skip: init_contexts(SyntaxCtx::CuSkipFlag, slice_qp_y),
            merge_flag: init_contexts(SyntaxCtx::GeneralMergeFlag, slice_qp_y),
        }
    }
}

// =====================================================================
// Slice-header bit prelude
// =====================================================================

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PSliceHeader {
    /// `slice_type` per §7.4.4 Table 8 — must be `1` (P).
    pub slice_type: u8,
    /// `slice_pic_order_cnt_lsb` (8-bit window for the scaffold).
    pub poc_lsb: u8,
    /// `num_ref_idx_l0_active_minus1` — single ref ⇒ 0.
    pub num_ref_idx_l0_active_minus1: u32,
    /// `slice_qp_delta` (signed `se(v)`).
    pub slice_qp_delta: i32,
    /// Picture width in luma samples (for decoder geometry).
    pub width: u32,
    /// Picture height in luma samples (for decoder geometry).
    pub height: u32,
}

fn write_pslice_header(hdr: &PSliceHeader) -> Vec<u8> {
    use crate::encoder::BitWriter;
    let mut bw = BitWriter::new();
    bw.write_bits(hdr.slice_type as u32, 3);
    bw.write_bits(hdr.poc_lsb as u32, 8);
    bw.write_ue(hdr.num_ref_idx_l0_active_minus1);
    bw.write_se(hdr.slice_qp_delta);
    bw.write_bits(hdr.width, 16);
    bw.write_bits(hdr.height, 16);
    bw.byte_alignment();
    bw.into_bytes()
}

fn read_pslice_header(bytes: &[u8]) -> Result<PSliceHeader> {
    use crate::bitreader::BitReader;
    let mut br = BitReader::new(bytes);
    let slice_type = br.u(3)? as u8;
    let poc_lsb = br.u(8)? as u8;
    let num_ref_idx_l0_active_minus1 = br.ue()?;
    let slice_qp_delta = br.se()?;
    let width = br.u(16)?;
    let height = br.u(16)?;
    Ok(PSliceHeader {
        slice_type,
        poc_lsb,
        num_ref_idx_l0_active_minus1,
        slice_qp_delta,
        width,
        height,
    })
}

// =====================================================================
// Slice-level encode
// =====================================================================

/// Round-58 / Round-59 — encode one P-slice for `curr` against single L0
/// reference `ref_buf` at QP `slice_qp_y`. Returns the wire bytes plus
/// the reconstructed luma-only [`PictureBuffer`] (chroma planes are
/// passed through from `ref_buf` since the scaffold does not encode
/// chroma residuals).
///
/// Behaviour:
/// 1. Walks `curr.luma` in 4×4 raster blocks.
/// 2. For each block, derives `mvp` from the spatial MV field (left
///    else above else zero) and runs an integer-pel full search ±N
///    around the predictor, **then refines to ¼-pel** via the §8.5.6.3
///    8-tap luma filter (round 59).
/// 3. MC-predicts the block from `ref_buf.luma` at the refined 1/16-pel
///    MV (`predict_luma_block`), computes residual, forward-DCTs +
///    flat-quants, and decides on `cbf_y` (any non-zero quantised
///    level ⇒ 1).
/// 4. Emits the per-block CABAC bins and the `tu_y_coded_flag`-gated
///    residual into the slice's single arithmetic stream.
pub fn encode_p_slice(
    curr: &PictureBuffer,
    ref_buf: &PictureBuffer,
    slice_qp_y: i32,
    poc_lsb: u8,
    search_range: i32,
) -> Result<(Vec<u8>, PictureBuffer)> {
    encode_p_slice_multi_ref(curr, &[ref_buf], slice_qp_y, poc_lsb, search_range)
}

/// Round-62 — encode one P-slice for `curr` against an L0 reference
/// list of N (up to [`MAX_REF_PICS`]) pictures. Per-block ME iterates
/// each candidate reference, runs the round-58 integer-pel SAD search
/// plus round-59 sub-pel refinement against each, and picks the
/// cheapest-SAD reference index. The wire-side per-block
/// `ref_idx_l0` (truncated-unary per §9.3.3.7) and the slice-header
/// `num_ref_idx_l0_active_minus1` (§7.4.4.2) are emitted accordingly.
///
/// `ref_list_l0` must contain at least one picture; all references in
/// the list must match `curr`'s luma geometry. When the list length is
/// 1 the wire reduces to the round-58 single-ref form (no `ref_idx_l0`
/// bins emitted; `num_ref_idx_l0_active_minus1 == 0`).
pub fn encode_p_slice_multi_ref(
    curr: &PictureBuffer,
    ref_list_l0: &[&PictureBuffer],
    slice_qp_y: i32,
    poc_lsb: u8,
    search_range: i32,
) -> Result<(Vec<u8>, PictureBuffer)> {
    if ref_list_l0.is_empty() {
        return Err(Error::invalid(
            "h266 P-slice: ref_list_l0 must contain at least one picture",
        ));
    }
    let w = curr.luma.width;
    let h = curr.luma.height;
    for r in ref_list_l0 {
        if w != r.luma.width || h != r.luma.height {
            return Err(Error::invalid(
                "h266 P-slice: current and reference picture dimensions differ",
            ));
        }
    }
    if w % INTER_BLOCK_W != 0 || h % INTER_BLOCK_H != 0 {
        return Err(Error::invalid(format!(
            "h266 P-slice scaffold requires picture {}x{} divisible by 16x16",
            w, h
        )));
    }
    let cols = w / INTER_BLOCK_W;
    let rows = h / INTER_BLOCK_H;
    let num_active_l0 = ref_list_l0.len();

    let hdr = PSliceHeader {
        slice_type: 1, // P per §7.4.4 Table 8 (I=2, P=1, B=0 in our local mapping)
        poc_lsb,
        num_ref_idx_l0_active_minus1: (num_active_l0 - 1) as u32,
        slice_qp_delta: slice_qp_y - SCAFFOLD_SLICE_QP,
        width: w as u32,
        height: h as u32,
    };
    let hdr_bytes = write_pslice_header(&hdr);

    let mut mv_field = MvField::new(cols, rows);
    let mut prepared: Vec<PreparedCu> = Vec::with_capacity(cols * rows);

    // Build the reconstruction skeleton from L0[0] for chroma + buffer
    // dimensions; luma is rewritten below. Round-63: chroma is also
    // rewritten via per-block chroma MC (4-tap §8.5.6.3.4 filter).
    let mut rec = ref_list_l0[0].clone();
    rec.luma.samples.fill(0);
    rec.cb.samples.fill(128);
    rec.cr.samples.fill(128);

    let mut enc = ArithEncoder::new();
    let mut ctxs = PSliceCtxs::init(slice_qp_y);

    for by in 0..rows {
        for bx in 0..cols {
            let cx = bx * INTER_BLOCK_W;
            let cy = by * INTER_BLOCK_H;
            let mvp_q16 = mv_field.mvp_for(bx, by);
            // Integer-pel search centre — round the predictor to the
            // nearest integer pel for the SAD search window (the
            // refinement step puts the fractional bits back).
            let mvp_int_pel = (
                ((mvp_q16.0 + 8) >> 4).clamp(i16::MIN as i32, i16::MAX as i32) as i16,
                ((mvp_q16.1 + 8) >> 4).clamp(i16::MIN as i32, i16::MAX as i32) as i16,
            );

            // --- Multi-ref ME (round-62): iterate every L0 picture,
            // refine to 1/16-pel, pick the cheapest by SAD. -----------
            let mut best_ref_idx: u8 = 0;
            let mut best_mv_q16: (i32, i32) = (0, 0);
            let mut best_sad: u32 = u32::MAX;
            for (idx, rp) in ref_list_l0.iter().enumerate() {
                let (int_mv, int_sad) = full_search_int(
                    &curr.luma,
                    cx,
                    cy,
                    INTER_BLOCK_W,
                    INTER_BLOCK_H,
                    &rp.luma,
                    mvp_int_pel,
                    search_range,
                );
                let (mv_q16, sad) = refine_subpel(
                    &curr.luma,
                    cx,
                    cy,
                    INTER_BLOCK_W,
                    INTER_BLOCK_H,
                    &rp.luma,
                    int_mv,
                    int_sad,
                )?;
                if sad < best_sad {
                    best_sad = sad;
                    best_mv_q16 = mv_q16;
                    best_ref_idx = idx as u8;
                }
            }
            let ref_p = ref_list_l0[best_ref_idx as usize];
            mv_field.set(bx, by, best_mv_q16);

            // --- MC prediction (sub-pel-aware) ---
            let mut pred = vec![0u8; INTER_BLOCK_W * INTER_BLOCK_H];
            mc_predict_subpel(
                &mut pred,
                &ref_p.luma,
                cx,
                cy,
                INTER_BLOCK_W,
                INTER_BLOCK_H,
                best_mv_q16,
            )?;

            // Round-63 — chroma 4-tap sub-pel MC on Cb + Cr at 4:2:0
            // (§8.5.6.3.4 / Table 28). The same luma 1/16-pel MV is
            // reused; predict_chroma_block handles the 4:2:0 mapping
            // (`mv >> 5` integer chroma offset, `mv & 31` chroma frac).
            const CW: usize = INTER_BLOCK_W / 2;
            const CH: usize = INTER_BLOCK_H / 2;
            let cx_c = cx / 2;
            let cy_c = cy / 2;
            let mut pred_cb = vec![0u8; CW * CH];
            mc_predict_chroma_subpel(&mut pred_cb, &ref_p.cb, cx_c, cy_c, CW, CH, best_mv_q16)?;
            let mut pred_cr = vec![0u8; CW * CH];
            mc_predict_chroma_subpel(&mut pred_cr, &ref_p.cr, cx_c, cy_c, CW, CH, best_mv_q16)?;
            for r in 0..CH {
                for c in 0..CW {
                    rec.cb.samples[(cy_c + r) * rec.cb.stride + (cx_c + c)] = pred_cb[r * CW + c];
                    rec.cr.samples[(cy_c + r) * rec.cr.stride + (cx_c + c)] = pred_cr[r * CW + c];
                }
            }

            // --- Residual: forward DCT + flat quant + reconstruct ---
            let (levels, recon) = prepare_inter_tb(
                &curr.luma,
                &pred,
                cx,
                cy,
                INTER_BLOCK_W,
                INTER_BLOCK_H,
                slice_qp_y,
            )?;
            // Write reconstruction into rec.luma.
            for r in 0..INTER_BLOCK_H {
                for c in 0..INTER_BLOCK_W {
                    rec.luma.samples[(cy + r) * rec.luma.stride + (cx + c)] =
                        recon[r * INTER_BLOCK_W + c];
                }
            }
            let cbf_y = levels.iter().any(|&l| l != 0);

            // --- Emit per-block CABAC bins ---
            let inc_skip = crate::ctx::ctx_inc_cu_skip_flag(false, false, false, false) as usize;
            let n_skip = ctxs.cu_skip.len() - 1;
            enc.encode_decision(&mut ctxs.cu_skip[inc_skip.min(n_skip)], 0)?;
            let inc_merge = crate::ctx::ctx_inc_general_merge_flag() as usize;
            let n_merge = ctxs.merge_flag.len() - 1;
            enc.encode_decision(&mut ctxs.merge_flag[inc_merge.min(n_merge)], 0)?;
            // inter_pred_idc = PRED_L0.
            enc.encode_bypass(0)?;
            // ref_idx_l0 — truncated-unary, §9.3.3.7. Only emitted when
            // the slice has more than one active L0 reference.
            encode_ref_idx(&mut enc, best_ref_idx, num_active_l0)?;
            // §7.4.7.2 — mvd_coding for x then y. Now in 1/16-pel units.
            let mvd = (best_mv_q16.0 - mvp_q16.0, best_mv_q16.1 - mvp_q16.1);
            encode_mvd_component(&mut enc, mvd.0)?;
            encode_mvd_component(&mut enc, mvd.1)?;

            // §7.4.10 — tu_y_coded_flag (CABAC).
            write_tu_y_coded_flag(&mut enc, &mut ctxs.residual, cbf_y, false, false, false)?;
            if cbf_y {
                encode_tb_coefficients(
                    &mut enc,
                    &mut ctxs.residual,
                    INTER_BLOCK_W,
                    INTER_BLOCK_H,
                    0,
                    &levels,
                )?;
            }

            prepared.push(PreparedCu::InterPSlice {
                ref_idx: best_ref_idx,
                mv: best_mv_q16,
                mvp: mvp_q16,
                n_tb_w: INTER_BLOCK_W,
                n_tb_h: INTER_BLOCK_H,
                levels: if cbf_y { levels } else { Vec::new() },
                cbf_y,
            });
        }
    }

    enc.encode_terminate(1)?;
    let cabac_bytes = enc.finish();

    // Round-63 — chroma was filled per-block above via the §8.5.6.3.4
    // 4-tap chroma MC; no slice-tail pass-through needed.

    // Wire layout: magic (14B) | hdr_len_le32 (4B) | hdr | cabac_len_le32 (4B) | cabac_bytes
    let mut out = Vec::with_capacity(PSLICE_MAGIC.len() + 8 + hdr_bytes.len() + cabac_bytes.len());
    out.extend_from_slice(PSLICE_MAGIC);
    out.extend_from_slice(&(hdr_bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(&hdr_bytes);
    out.extend_from_slice(&(cabac_bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(&cabac_bytes);

    let _ = prepared;
    Ok((out, rec))
}

// =====================================================================
// Slice-level decode — round-trip side
// =====================================================================

/// Round-58 / Round-59 — decode one P-slice produced by [`encode_p_slice`].
/// Reads the magic + slice header + CABAC stream, reconstructs each 4×4
/// inter block (sub-pel-aware MC predict from `ref_buf` shifted by the
/// recovered 1/16-pel MV + dequantised inverse-DCT residual), and
/// returns the reconstructed luma in a fresh [`PictureBuffer`] (chroma
/// is copied through from `ref_buf` per the encoder-side scaffold scope).
pub fn decode_p_slice(bytes: &[u8], ref_buf: &PictureBuffer) -> Result<PictureBuffer> {
    decode_p_slice_multi_ref(bytes, &[ref_buf])
}

/// Round-62 — decode one P-slice produced by [`encode_p_slice_multi_ref`].
/// Reads `num_ref_idx_l0_active_minus1` from the slice header, then for
/// each block reads the truncated-unary `ref_idx_l0` (§9.3.3.7) before
/// the MVD chain, and reconstructs from the chosen reference in
/// `ref_list_l0`. Backwards-compatible with the single-ref wire
/// produced by [`encode_p_slice`].
pub fn decode_p_slice_multi_ref(
    bytes: &[u8],
    ref_list_l0: &[&PictureBuffer],
) -> Result<PictureBuffer> {
    if ref_list_l0.is_empty() {
        return Err(Error::invalid(
            "h266 P-slice decode: ref_list_l0 must contain at least one picture",
        ));
    }
    if bytes.len() < PSLICE_MAGIC.len() + 8 {
        return Err(Error::invalid("h266 P-slice decode: payload too short"));
    }
    if &bytes[..PSLICE_MAGIC.len()] != PSLICE_MAGIC {
        return Err(Error::invalid("h266 P-slice decode: missing magic"));
    }
    let mut p = PSLICE_MAGIC.len();
    let hdr_len = u32::from_le_bytes(bytes[p..p + 4].try_into().unwrap()) as usize;
    p += 4;
    if p + hdr_len > bytes.len() {
        return Err(Error::invalid("h266 P-slice decode: header overflow"));
    }
    let hdr = read_pslice_header(&bytes[p..p + hdr_len])?;
    p += hdr_len;
    let cabac_len = u32::from_le_bytes(bytes[p..p + 4].try_into().unwrap()) as usize;
    p += 4;
    if p + cabac_len > bytes.len() {
        return Err(Error::invalid("h266 P-slice decode: cabac overflow"));
    }
    let mut cabac_bytes: Vec<u8> = bytes[p..p + cabac_len].to_vec();
    cabac_bytes.extend_from_slice(&[0u8; 256]);

    if hdr.slice_type != 1 {
        return Err(Error::invalid(format!(
            "h266 P-slice decode: expected slice_type=1, got {}",
            hdr.slice_type
        )));
    }
    let num_active_l0 = (hdr.num_ref_idx_l0_active_minus1 as usize) + 1;
    if num_active_l0 > ref_list_l0.len() {
        return Err(Error::invalid(format!(
            "h266 P-slice decode: slice header advertises {} active L0 refs but caller provided {}",
            num_active_l0,
            ref_list_l0.len(),
        )));
    }
    let w = hdr.width as usize;
    let h = hdr.height as usize;
    let ref0 = ref_list_l0[0];
    for r in ref_list_l0 {
        if w != r.luma.width || h != r.luma.height {
            return Err(Error::invalid(format!(
                "h266 P-slice decode: header geometry {}x{} vs reference {}x{}",
                w, h, r.luma.width, r.luma.height
            )));
        }
    }
    if w % INTER_BLOCK_W != 0 || h % INTER_BLOCK_H != 0 {
        return Err(Error::invalid(
            "h266 P-slice decode: dims not divisible by 16",
        ));
    }
    let cols = w / INTER_BLOCK_W;
    let rows = h / INTER_BLOCK_H;
    let slice_qp_y = SCAFFOLD_SLICE_QP + hdr.slice_qp_delta;

    let mut dec = ArithDecoder::new(&cabac_bytes)?;
    let mut ctxs = PSliceCtxs::init(slice_qp_y);
    let mut mv_field = MvField::new(cols, rows);

    let mut out = ref0.clone();
    out.luma.samples.fill(0);
    out.cb.samples.fill(128);
    out.cr.samples.fill(128);

    for by in 0..rows {
        for bx in 0..cols {
            let cx = bx * INTER_BLOCK_W;
            let cy = by * INTER_BLOCK_H;
            let mvp_q16 = mv_field.mvp_for(bx, by);

            // cu_skip_flag.
            let inc_skip = crate::ctx::ctx_inc_cu_skip_flag(false, false, false, false) as usize;
            let n_skip = ctxs.cu_skip.len() - 1;
            let _skip = dec.decode_decision(&mut ctxs.cu_skip[inc_skip.min(n_skip)])?;
            // general_merge_flag.
            let inc_merge = crate::ctx::ctx_inc_general_merge_flag() as usize;
            let n_merge = ctxs.merge_flag.len() - 1;
            let _merge = dec.decode_decision(&mut ctxs.merge_flag[inc_merge.min(n_merge)])?;
            // inter_pred_idc — bypass.
            let _ = dec.decode_bypass()?;
            // ref_idx_l0 — truncated-unary, only emitted when active > 1.
            let ref_idx_l0 = decode_ref_idx(&mut dec, num_active_l0)?;
            // mvd_coding x / y (now 1/16-pel units).
            let mvd_x = decode_mvd_component(&mut dec)?;
            let mvd_y = decode_mvd_component(&mut dec)?;
            let mv_q16 = (mvp_q16.0 + mvd_x, mvp_q16.1 + mvd_y);
            mv_field.set(bx, by, mv_q16);

            // tu_y_coded_flag.
            let cbf_y = read_tu_y_coded_flag(&mut dec, &mut ctxs.residual, false, false, false)?;
            // Predict the block from the reference + sub-pel MV.
            let ref_p = ref_list_l0[ref_idx_l0 as usize];
            let mut pred = vec![0u8; INTER_BLOCK_W * INTER_BLOCK_H];
            mc_predict_subpel(
                &mut pred,
                &ref_p.luma,
                cx,
                cy,
                INTER_BLOCK_W,
                INTER_BLOCK_H,
                mv_q16,
            )?;
            let recon = if cbf_y {
                let levels = crate::residual::decode_tb_coefficients(
                    &mut dec,
                    &mut ctxs.residual,
                    INTER_BLOCK_W,
                    INTER_BLOCK_H,
                    0,
                )?;
                reconstruct_inter_tb_from_levels(
                    &levels,
                    &pred,
                    INTER_BLOCK_W,
                    INTER_BLOCK_H,
                    slice_qp_y,
                )?
            } else {
                pred
            };
            for r in 0..INTER_BLOCK_H {
                for c in 0..INTER_BLOCK_W {
                    out.luma.samples[(cy + r) * out.luma.stride + (cx + c)] =
                        recon[r * INTER_BLOCK_W + c];
                }
            }

            // Round-63 — chroma 4-tap sub-pel MC mirroring the encoder.
            const CW: usize = INTER_BLOCK_W / 2;
            const CH: usize = INTER_BLOCK_H / 2;
            let cx_c = cx / 2;
            let cy_c = cy / 2;
            let mut pred_cb = vec![0u8; CW * CH];
            mc_predict_chroma_subpel(&mut pred_cb, &ref_p.cb, cx_c, cy_c, CW, CH, mv_q16)?;
            let mut pred_cr = vec![0u8; CW * CH];
            mc_predict_chroma_subpel(&mut pred_cr, &ref_p.cr, cx_c, cy_c, CW, CH, mv_q16)?;
            for r in 0..CH {
                for c in 0..CW {
                    out.cb.samples[(cy_c + r) * out.cb.stride + (cx_c + c)] = pred_cb[r * CW + c];
                    out.cr.samples[(cy_c + r) * out.cr.stride + (cx_c + c)] = pred_cr[r * CW + c];
                }
            }
        }
    }

    let _ = dec.decode_terminate()?;
    Ok(out)
}

// =====================================================================
// Round-60 / Round-61 — B-slice (bi-prediction) encoder + decoder
// =====================================================================
//
// The B-slice path mirrors the P-slice path but threads TWO reference
// lists (L0 + L1). For each 4×4 luma block:
//
//   1. Run integer-pel full-search SAD against L0[0] and L1[0]
//      independently (each produces its own integer-pel MV).
//   2. For each list, run a two-stage sub-pel refinement (round 61):
//      8 half-pel neighbours around the integer-pel best, then
//      8 quarter-pel neighbours around the half-pel best, each
//      probed with the §8.5.6.3.2 Table 27 8-tap luma filter
//      (`hpelIfIdx == 0`) via `predict_luma_block`. This produces
//      a 1/16-pel MV per list.
//   3. Form three candidate predictions: L0-only, L1-only, and BI
//      (`(predL0 + predL1 + 1) >> 1` per §8.5.6.4 — simple average,
//      weighted bi-pred is out of scope).
//   4. Pick the cheapest of {L0, L1, BI} by SSE (Lagrangian SSE + λ·R
//      with the per-mode "bits" approximated as a small fixed cost so
//      ties favour uni-pred). This is the encoder-side RDO.
//   5. Emit the chosen `inter_pred_idc` + per-list MVDs + residual.
//
// The decoder reverses the process: read `inter_pred_idc`, read each
// list's MVD chain, reconstruct each MV from the per-list MVP, MC-predict
// from `ref_l0` and/or `ref_l1`, average for BI, add residual.
//
// Multi-reference is implicit: L0 and L1 are TWO references. For this
// round, both lists hold a SINGLE picture — enough to validate the
// §7.4.7.2 / §8.5.6.4 syntax + reconstruction. Real multi-ref-per-list
// DPB plumbing comes later.

/// `inter_pred_idc` wire encoding — bypass-coded as two bins:
///   bit 0: 0 → uni-pred, 1 → bi-pred.
///   bit 1 (uni-pred only): 0 → L0, 1 → L1.
fn encode_inter_pred_idc(enc: &mut ArithEncoder, idc: InterPredIdc) -> Result<()> {
    match idc {
        InterPredIdc::L0 => {
            enc.encode_bypass(0)?;
            enc.encode_bypass(0)?;
        }
        InterPredIdc::L1 => {
            enc.encode_bypass(0)?;
            enc.encode_bypass(1)?;
        }
        InterPredIdc::Bi => {
            enc.encode_bypass(1)?;
        }
    }
    Ok(())
}

fn decode_inter_pred_idc(dec: &mut ArithDecoder<'_>) -> Result<InterPredIdc> {
    let bi = dec.decode_bypass()?;
    if bi == 1 {
        return Ok(InterPredIdc::Bi);
    }
    let which = dec.decode_bypass()?;
    if which == 0 {
        Ok(InterPredIdc::L0)
    } else {
        Ok(InterPredIdc::L1)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BSliceHeader {
    /// `slice_type` per §7.4.4 Table 8 — `0` for B.
    pub slice_type: u8,
    pub poc_lsb: u8,
    /// `num_ref_idx_l0_active_minus1` — single ref ⇒ 0.
    pub num_ref_idx_l0_active_minus1: u32,
    /// `num_ref_idx_l1_active_minus1` — single ref ⇒ 0.
    pub num_ref_idx_l1_active_minus1: u32,
    pub slice_qp_delta: i32,
    pub width: u32,
    pub height: u32,
}

fn write_bslice_header(hdr: &BSliceHeader) -> Vec<u8> {
    use crate::encoder::BitWriter;
    let mut bw = BitWriter::new();
    bw.write_bits(hdr.slice_type as u32, 3);
    bw.write_bits(hdr.poc_lsb as u32, 8);
    bw.write_ue(hdr.num_ref_idx_l0_active_minus1);
    bw.write_ue(hdr.num_ref_idx_l1_active_minus1);
    bw.write_se(hdr.slice_qp_delta);
    bw.write_bits(hdr.width, 16);
    bw.write_bits(hdr.height, 16);
    bw.byte_alignment();
    bw.into_bytes()
}

fn read_bslice_header(bytes: &[u8]) -> Result<BSliceHeader> {
    use crate::bitreader::BitReader;
    let mut br = BitReader::new(bytes);
    let slice_type = br.u(3)? as u8;
    let poc_lsb = br.u(8)? as u8;
    let num_ref_idx_l0_active_minus1 = br.ue()?;
    let num_ref_idx_l1_active_minus1 = br.ue()?;
    let slice_qp_delta = br.se()?;
    let width = br.u(16)?;
    let height = br.u(16)?;
    Ok(BSliceHeader {
        slice_type,
        poc_lsb,
        num_ref_idx_l0_active_minus1,
        num_ref_idx_l1_active_minus1,
        slice_qp_delta,
        width,
        height,
    })
}

/// Per-block SSE between the source luma and a candidate prediction.
fn sse_block(src: &PicturePlane, cx: usize, cy: usize, pred: &[u8], w: usize, h: usize) -> u32 {
    let mut sse: u32 = 0;
    for r in 0..h {
        for c in 0..w {
            let s = src.samples[(cy + r) * src.stride + (cx + c)] as i32;
            let p = pred[r * w + c] as i32;
            let d = s - p;
            sse = sse.saturating_add((d * d) as u32);
        }
    }
    sse
}

/// Average two predictions per §8.5.6.4 — `(p0 + p1 + 1) >> 1`.
fn average_bi(p0: &[u8], p1: &[u8], w: usize, h: usize) -> Vec<u8> {
    let mut out = vec![0u8; w * h];
    for i in 0..w * h {
        let a = p0[i] as u16;
        let b = p1[i] as u16;
        out[i] = ((a + b + 1) >> 1) as u8;
    }
    out
}

/// Round-60 / Round-61 — encode one B-slice for `curr` against L0 + L1
/// reference pictures `ref_l0` / `ref_l1` at QP `slice_qp_y`. Returns
/// the wire bytes plus the reconstructed luma-only [`PictureBuffer`].
/// Chroma is passed through from `ref_l0` (the scaffold does not
/// encode chroma residuals).
///
/// Round 61 adds per-list sub-pel ME refinement: integer-pel full
/// search then 8-neighbour ½-pel + 8-neighbour ¼-pel refinement, each
/// stage using the §8.5.6.3.2 Table 27 8-tap luma filter. The RDO over
/// `{L0, L1, BI}` runs after both lists have settled at 1/16-pel
/// precision; BI reconstruction is the §8.5.6.4 simple average
/// `(predL0 + predL1 + 1) >> 1`.
pub fn encode_b_slice(
    curr: &PictureBuffer,
    ref_l0: &PictureBuffer,
    ref_l1: &PictureBuffer,
    slice_qp_y: i32,
    poc_lsb: u8,
    search_range: i32,
) -> Result<(Vec<u8>, PictureBuffer)> {
    encode_b_slice_multi_ref(
        curr,
        &[ref_l0],
        &[ref_l1],
        slice_qp_y,
        poc_lsb,
        search_range,
    )
}

/// Round-62 — encode one B-slice for `curr` against L0 + L1 reference
/// lists of N pictures each (up to [`MAX_REF_PICS`]). Per-block ME
/// iterates every L0 and L1 reference independently, refines each to
/// 1/16-pel, then runs the §7.4.7.2 `{PRED_L0, PRED_L1, PRED_BI}` RDO
/// using each list's chosen best. Wire-side: slice header advertises
/// `num_ref_idx_l{0,1}_active_minus1` (§7.4.4.2); per-CU
/// `ref_idx_l0` / `ref_idx_l1` are emitted as truncated-unary
/// (§9.3.3.7) when the corresponding list has more than one active
/// reference. `ref_list_l1` may be the same list as `ref_list_l0` when
/// the picture is a low-delay-B with co-equal lists.
pub fn encode_b_slice_multi_ref(
    curr: &PictureBuffer,
    ref_list_l0: &[&PictureBuffer],
    ref_list_l1: &[&PictureBuffer],
    slice_qp_y: i32,
    poc_lsb: u8,
    search_range: i32,
) -> Result<(Vec<u8>, PictureBuffer)> {
    if ref_list_l0.is_empty() || ref_list_l1.is_empty() {
        return Err(Error::invalid(
            "h266 B-slice: both ref lists must contain at least one picture",
        ));
    }
    let w = curr.luma.width;
    let h = curr.luma.height;
    for rp in ref_list_l0.iter().chain(ref_list_l1.iter()) {
        if w != rp.luma.width || h != rp.luma.height {
            return Err(Error::invalid(
                "h266 B-slice: current and reference picture dimensions differ",
            ));
        }
    }
    if w % INTER_BLOCK_W != 0 || h % INTER_BLOCK_H != 0 {
        return Err(Error::invalid(format!(
            "h266 B-slice scaffold requires picture {}x{} divisible by {}x{}",
            w, h, INTER_BLOCK_W, INTER_BLOCK_H,
        )));
    }
    let cols = w / INTER_BLOCK_W;
    let rows = h / INTER_BLOCK_H;
    let num_active_l0 = ref_list_l0.len();
    let num_active_l1 = ref_list_l1.len();

    let hdr = BSliceHeader {
        slice_type: 0, // B per §7.4.4 Table 8 (B=0, P=1, I=2 in local mapping)
        poc_lsb,
        num_ref_idx_l0_active_minus1: (num_active_l0 - 1) as u32,
        num_ref_idx_l1_active_minus1: (num_active_l1 - 1) as u32,
        slice_qp_delta: slice_qp_y - SCAFFOLD_SLICE_QP,
        width: w as u32,
        height: h as u32,
    };
    let hdr_bytes = write_bslice_header(&hdr);

    let mut mv_field_l0 = MvField::new(cols, rows);
    let mut mv_field_l1 = MvField::new(cols, rows);
    let mut prepared: Vec<PreparedCu> = Vec::with_capacity(cols * rows);

    let mut rec = ref_list_l0[0].clone();
    rec.luma.samples.fill(0);
    rec.cb.samples.fill(128);
    rec.cr.samples.fill(128);

    let mut enc = ArithEncoder::new();
    let mut ctxs = PSliceCtxs::init(slice_qp_y);

    for by in 0..rows {
        for bx in 0..cols {
            let cx = bx * INTER_BLOCK_W;
            let cy = by * INTER_BLOCK_H;

            // ----- L0 list: MVP + per-ref integer-pel + sub-pel refine -----
            // Round 62 — iterate every L0 reference, refine each to
            // 1/16-pel, keep the cheapest by SAD.
            let mvp_l0_q16 = mv_field_l0.mvp_for(bx, by);
            let mvp_l0_int = (
                ((mvp_l0_q16.0 + 8) >> 4).clamp(i16::MIN as i32, i16::MAX as i32) as i16,
                ((mvp_l0_q16.1 + 8) >> 4).clamp(i16::MIN as i32, i16::MAX as i32) as i16,
            );
            let mut best_ref_idx_l0: u8 = 0;
            let mut mv_l0_q16: (i32, i32) = (0, 0);
            let mut best_sad_l0: u32 = u32::MAX;
            for (idx, rp) in ref_list_l0.iter().enumerate() {
                let (int_mv, int_sad) = full_search_int(
                    &curr.luma,
                    cx,
                    cy,
                    INTER_BLOCK_W,
                    INTER_BLOCK_H,
                    &rp.luma,
                    mvp_l0_int,
                    search_range,
                );
                let (mv_q16, sad) = refine_subpel(
                    &curr.luma,
                    cx,
                    cy,
                    INTER_BLOCK_W,
                    INTER_BLOCK_H,
                    &rp.luma,
                    int_mv,
                    int_sad,
                )?;
                if sad < best_sad_l0 {
                    best_sad_l0 = sad;
                    mv_l0_q16 = mv_q16;
                    best_ref_idx_l0 = idx as u8;
                }
            }
            let ref_l0_p = ref_list_l0[best_ref_idx_l0 as usize];

            // ----- L1 list: MVP + per-ref integer-pel + sub-pel refine -----
            let mvp_l1_q16 = mv_field_l1.mvp_for(bx, by);
            let mvp_l1_int = (
                ((mvp_l1_q16.0 + 8) >> 4).clamp(i16::MIN as i32, i16::MAX as i32) as i16,
                ((mvp_l1_q16.1 + 8) >> 4).clamp(i16::MIN as i32, i16::MAX as i32) as i16,
            );
            let mut best_ref_idx_l1: u8 = 0;
            let mut mv_l1_q16: (i32, i32) = (0, 0);
            let mut best_sad_l1: u32 = u32::MAX;
            for (idx, rp) in ref_list_l1.iter().enumerate() {
                let (int_mv, int_sad) = full_search_int(
                    &curr.luma,
                    cx,
                    cy,
                    INTER_BLOCK_W,
                    INTER_BLOCK_H,
                    &rp.luma,
                    mvp_l1_int,
                    search_range,
                );
                let (mv_q16, sad) = refine_subpel(
                    &curr.luma,
                    cx,
                    cy,
                    INTER_BLOCK_W,
                    INTER_BLOCK_H,
                    &rp.luma,
                    int_mv,
                    int_sad,
                )?;
                if sad < best_sad_l1 {
                    best_sad_l1 = sad;
                    mv_l1_q16 = mv_q16;
                    best_ref_idx_l1 = idx as u8;
                }
            }
            let ref_l1_p = ref_list_l1[best_ref_idx_l1 as usize];

            // ----- Form three candidate predictions -----
            let mut pred_l0 = vec![0u8; INTER_BLOCK_W * INTER_BLOCK_H];
            mc_predict_subpel(
                &mut pred_l0,
                &ref_l0_p.luma,
                cx,
                cy,
                INTER_BLOCK_W,
                INTER_BLOCK_H,
                mv_l0_q16,
            )?;
            let mut pred_l1 = vec![0u8; INTER_BLOCK_W * INTER_BLOCK_H];
            mc_predict_subpel(
                &mut pred_l1,
                &ref_l1_p.luma,
                cx,
                cy,
                INTER_BLOCK_W,
                INTER_BLOCK_H,
                mv_l1_q16,
            )?;
            let pred_bi = average_bi(&pred_l0, &pred_l1, INTER_BLOCK_W, INTER_BLOCK_H);

            // ----- RDO: pick cheapest of {L0, L1, BI} on SSE + bias -----
            // The bias term stands in for the per-mode bin-cost
            // difference (BI emits an extra MVD chain, ~+8 bits). A
            // small fixed bias keeps the encoder away from BI when the
            // uni-pred predictions are already perfect.
            const BI_BIAS: u32 = 1; // negligible vs typical SSE
            let sse_l0 = sse_block(&curr.luma, cx, cy, &pred_l0, INTER_BLOCK_W, INTER_BLOCK_H);
            let sse_l1 = sse_block(&curr.luma, cx, cy, &pred_l1, INTER_BLOCK_W, INTER_BLOCK_H);
            let sse_bi = sse_block(&curr.luma, cx, cy, &pred_bi, INTER_BLOCK_W, INTER_BLOCK_H)
                .saturating_add(BI_BIAS);

            let (idc, pred) = if sse_l0 <= sse_l1 && sse_l0 <= sse_bi {
                (InterPredIdc::L0, pred_l0)
            } else if sse_l1 <= sse_bi {
                (InterPredIdc::L1, pred_l1)
            } else {
                (InterPredIdc::Bi, pred_bi)
            };

            // ----- Wire-side MV/MVD: only emit active lists' MVDs -----
            let (active_l0, active_l1) = match idc {
                InterPredIdc::L0 => (true, false),
                InterPredIdc::L1 => (false, true),
                InterPredIdc::Bi => (true, true),
            };
            // For inactive lists, the MV is "inferred zero" — and the
            // mv_field cell is set to zero so neighbour MVPs reflect
            // what the decoder will see.
            if active_l0 {
                mv_field_l0.set(bx, by, mv_l0_q16);
            } else {
                mv_field_l0.set(bx, by, (0, 0));
            }
            if active_l1 {
                mv_field_l1.set(bx, by, mv_l1_q16);
            } else {
                mv_field_l1.set(bx, by, (0, 0));
            }

            // ----- Residual: forward DCT + flat quant + reconstruct -----
            let (levels, recon) = prepare_inter_tb(
                &curr.luma,
                &pred,
                cx,
                cy,
                INTER_BLOCK_W,
                INTER_BLOCK_H,
                slice_qp_y,
            )?;
            for r in 0..INTER_BLOCK_H {
                for c in 0..INTER_BLOCK_W {
                    rec.luma.samples[(cy + r) * rec.luma.stride + (cx + c)] =
                        recon[r * INTER_BLOCK_W + c];
                }
            }
            let cbf_y = levels.iter().any(|&l| l != 0);

            // ----- Round-63 — chroma 4-tap sub-pel MC (§8.5.6.3.4) -----
            // Per the chosen `idc` we predict chroma from L0, L1, or BI
            // average. The same per-list luma MVs are reused;
            // predict_chroma_block applies the 4:2:0 mapping internally.
            const CW: usize = INTER_BLOCK_W / 2;
            const CH: usize = INTER_BLOCK_H / 2;
            let cx_c = cx / 2;
            let cy_c = cy / 2;
            let (pred_cb, pred_cr) = match idc {
                InterPredIdc::L0 => {
                    let mut cb = vec![0u8; CW * CH];
                    let mut cr = vec![0u8; CW * CH];
                    mc_predict_chroma_subpel(&mut cb, &ref_l0_p.cb, cx_c, cy_c, CW, CH, mv_l0_q16)?;
                    mc_predict_chroma_subpel(&mut cr, &ref_l0_p.cr, cx_c, cy_c, CW, CH, mv_l0_q16)?;
                    (cb, cr)
                }
                InterPredIdc::L1 => {
                    let mut cb = vec![0u8; CW * CH];
                    let mut cr = vec![0u8; CW * CH];
                    mc_predict_chroma_subpel(&mut cb, &ref_l1_p.cb, cx_c, cy_c, CW, CH, mv_l1_q16)?;
                    mc_predict_chroma_subpel(&mut cr, &ref_l1_p.cr, cx_c, cy_c, CW, CH, mv_l1_q16)?;
                    (cb, cr)
                }
                InterPredIdc::Bi => {
                    let cb = mc_predict_chroma_subpel_bi(
                        &ref_l0_p.cb,
                        mv_l0_q16,
                        &ref_l1_p.cb,
                        mv_l1_q16,
                        cx_c,
                        cy_c,
                        CW,
                        CH,
                    )?;
                    let cr = mc_predict_chroma_subpel_bi(
                        &ref_l0_p.cr,
                        mv_l0_q16,
                        &ref_l1_p.cr,
                        mv_l1_q16,
                        cx_c,
                        cy_c,
                        CW,
                        CH,
                    )?;
                    (cb, cr)
                }
            };
            for r in 0..CH {
                for c in 0..CW {
                    rec.cb.samples[(cy_c + r) * rec.cb.stride + (cx_c + c)] = pred_cb[r * CW + c];
                    rec.cr.samples[(cy_c + r) * rec.cr.stride + (cx_c + c)] = pred_cr[r * CW + c];
                }
            }

            // ----- Emit per-block CABAC bins -----
            let inc_skip = crate::ctx::ctx_inc_cu_skip_flag(false, false, false, false) as usize;
            let n_skip = ctxs.cu_skip.len() - 1;
            enc.encode_decision(&mut ctxs.cu_skip[inc_skip.min(n_skip)], 0)?;
            let inc_merge = crate::ctx::ctx_inc_general_merge_flag() as usize;
            let n_merge = ctxs.merge_flag.len() - 1;
            enc.encode_decision(&mut ctxs.merge_flag[inc_merge.min(n_merge)], 0)?;
            // inter_pred_idc (compact bypass form).
            encode_inter_pred_idc(&mut enc, idc)?;
            if active_l0 {
                // ref_idx_l0 — truncated-unary, §9.3.3.7.
                encode_ref_idx(&mut enc, best_ref_idx_l0, num_active_l0)?;
                let mvd_l0 = (mv_l0_q16.0 - mvp_l0_q16.0, mv_l0_q16.1 - mvp_l0_q16.1);
                encode_mvd_component(&mut enc, mvd_l0.0)?;
                encode_mvd_component(&mut enc, mvd_l0.1)?;
            }
            if active_l1 {
                // ref_idx_l1 — truncated-unary, §9.3.3.7.
                encode_ref_idx(&mut enc, best_ref_idx_l1, num_active_l1)?;
                let mvd_l1 = (mv_l1_q16.0 - mvp_l1_q16.0, mv_l1_q16.1 - mvp_l1_q16.1);
                encode_mvd_component(&mut enc, mvd_l1.0)?;
                encode_mvd_component(&mut enc, mvd_l1.1)?;
            }
            write_tu_y_coded_flag(&mut enc, &mut ctxs.residual, cbf_y, false, false, false)?;
            if cbf_y {
                encode_tb_coefficients(
                    &mut enc,
                    &mut ctxs.residual,
                    INTER_BLOCK_W,
                    INTER_BLOCK_H,
                    0,
                    &levels,
                )?;
            }

            prepared.push(PreparedCu::InterBSlice {
                inter_pred_idc: idc,
                ref_idx_l0: if active_l0 { best_ref_idx_l0 } else { 0 },
                ref_idx_l1: if active_l1 { best_ref_idx_l1 } else { 0 },
                mv_l0: if active_l0 { mv_l0_q16 } else { (0, 0) },
                mv_l1: if active_l1 { mv_l1_q16 } else { (0, 0) },
                mvp_l0: mvp_l0_q16,
                mvp_l1: mvp_l1_q16,
                n_tb_w: INTER_BLOCK_W,
                n_tb_h: INTER_BLOCK_H,
                levels: if cbf_y { levels } else { Vec::new() },
                cbf_y,
            });
        }
    }

    enc.encode_terminate(1)?;
    let cabac_bytes = enc.finish();

    // Round-63 — chroma was filled per-block above via the 4-tap MC.

    let mut out = Vec::with_capacity(BSLICE_MAGIC.len() + 8 + hdr_bytes.len() + cabac_bytes.len());
    out.extend_from_slice(BSLICE_MAGIC);
    out.extend_from_slice(&(hdr_bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(&hdr_bytes);
    out.extend_from_slice(&(cabac_bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(&cabac_bytes);

    let _ = prepared;
    Ok((out, rec))
}

/// Round-60 / Round-61 — decode one B-slice produced by
/// [`encode_b_slice`]. Sub-pel MVs are handled transparently because
/// the per-list MC prediction goes through the same `mc_predict_subpel`
/// helper used by the P-slice decoder.
pub fn decode_b_slice(
    bytes: &[u8],
    ref_l0: &PictureBuffer,
    ref_l1: &PictureBuffer,
) -> Result<PictureBuffer> {
    decode_b_slice_multi_ref(bytes, &[ref_l0], &[ref_l1])
}

/// Round-62 — decode one B-slice produced by
/// [`encode_b_slice_multi_ref`]. Reads
/// `num_ref_idx_l{0,1}_active_minus1` from the slice header, then for
/// each block reads the truncated-unary `ref_idx_l0` / `ref_idx_l1`
/// before each active list's MVD chain, and reconstructs from the
/// chosen reference in `ref_list_l0` / `ref_list_l1`. Backwards-
/// compatible with the single-ref wire produced by [`encode_b_slice`].
pub fn decode_b_slice_multi_ref(
    bytes: &[u8],
    ref_list_l0: &[&PictureBuffer],
    ref_list_l1: &[&PictureBuffer],
) -> Result<PictureBuffer> {
    if ref_list_l0.is_empty() || ref_list_l1.is_empty() {
        return Err(Error::invalid(
            "h266 B-slice decode: both ref lists must contain at least one picture",
        ));
    }
    if bytes.len() < BSLICE_MAGIC.len() + 8 {
        return Err(Error::invalid("h266 B-slice decode: payload too short"));
    }
    if &bytes[..BSLICE_MAGIC.len()] != BSLICE_MAGIC {
        return Err(Error::invalid("h266 B-slice decode: missing magic"));
    }
    let mut p = BSLICE_MAGIC.len();
    let hdr_len = u32::from_le_bytes(bytes[p..p + 4].try_into().unwrap()) as usize;
    p += 4;
    if p + hdr_len > bytes.len() {
        return Err(Error::invalid("h266 B-slice decode: header overflow"));
    }
    let hdr = read_bslice_header(&bytes[p..p + hdr_len])?;
    p += hdr_len;
    let cabac_len = u32::from_le_bytes(bytes[p..p + 4].try_into().unwrap()) as usize;
    p += 4;
    if p + cabac_len > bytes.len() {
        return Err(Error::invalid("h266 B-slice decode: cabac overflow"));
    }
    let mut cabac_bytes: Vec<u8> = bytes[p..p + cabac_len].to_vec();
    cabac_bytes.extend_from_slice(&[0u8; 256]);

    if hdr.slice_type != 0 {
        return Err(Error::invalid(format!(
            "h266 B-slice decode: expected slice_type=0, got {}",
            hdr.slice_type
        )));
    }
    let num_active_l0 = (hdr.num_ref_idx_l0_active_minus1 as usize) + 1;
    let num_active_l1 = (hdr.num_ref_idx_l1_active_minus1 as usize) + 1;
    if num_active_l0 > ref_list_l0.len() || num_active_l1 > ref_list_l1.len() {
        return Err(Error::invalid(format!(
            "h266 B-slice decode: slice header advertises {} L0 / {} L1 active refs, caller provided {} / {}",
            num_active_l0,
            num_active_l1,
            ref_list_l0.len(),
            ref_list_l1.len(),
        )));
    }
    let w = hdr.width as usize;
    let h = hdr.height as usize;
    for rp in ref_list_l0.iter().chain(ref_list_l1.iter()) {
        if w != rp.luma.width || h != rp.luma.height {
            return Err(Error::invalid(format!(
                "h266 B-slice decode: header geometry {}x{} vs reference {}x{}",
                w, h, rp.luma.width, rp.luma.height,
            )));
        }
    }
    if w % INTER_BLOCK_W != 0 || h % INTER_BLOCK_H != 0 {
        return Err(Error::invalid(format!(
            "h266 B-slice decode: dims not divisible by {}",
            INTER_BLOCK_W,
        )));
    }
    let cols = w / INTER_BLOCK_W;
    let rows = h / INTER_BLOCK_H;
    let slice_qp_y = SCAFFOLD_SLICE_QP + hdr.slice_qp_delta;

    let mut dec = ArithDecoder::new(&cabac_bytes)?;
    let mut ctxs = PSliceCtxs::init(slice_qp_y);
    let mut mv_field_l0 = MvField::new(cols, rows);
    let mut mv_field_l1 = MvField::new(cols, rows);

    let mut out = ref_list_l0[0].clone();
    out.luma.samples.fill(0);
    out.cb.samples.fill(128);
    out.cr.samples.fill(128);

    for by in 0..rows {
        for bx in 0..cols {
            let cx = bx * INTER_BLOCK_W;
            let cy = by * INTER_BLOCK_H;
            let mvp_l0_q16 = mv_field_l0.mvp_for(bx, by);
            let mvp_l1_q16 = mv_field_l1.mvp_for(bx, by);

            // cu_skip_flag.
            let inc_skip = crate::ctx::ctx_inc_cu_skip_flag(false, false, false, false) as usize;
            let n_skip = ctxs.cu_skip.len() - 1;
            let _skip = dec.decode_decision(&mut ctxs.cu_skip[inc_skip.min(n_skip)])?;
            // general_merge_flag.
            let inc_merge = crate::ctx::ctx_inc_general_merge_flag() as usize;
            let n_merge = ctxs.merge_flag.len() - 1;
            let _merge = dec.decode_decision(&mut ctxs.merge_flag[inc_merge.min(n_merge)])?;
            // inter_pred_idc.
            let idc = decode_inter_pred_idc(&mut dec)?;
            let (active_l0, active_l1) = match idc {
                InterPredIdc::L0 => (true, false),
                InterPredIdc::L1 => (false, true),
                InterPredIdc::Bi => (true, true),
            };
            let mut mv_l0_q16 = (0i32, 0i32);
            let mut mv_l1_q16 = (0i32, 0i32);
            let mut ref_idx_l0: u8 = 0;
            let mut ref_idx_l1: u8 = 0;
            if active_l0 {
                ref_idx_l0 = decode_ref_idx(&mut dec, num_active_l0)?;
                let mvd_x = decode_mvd_component(&mut dec)?;
                let mvd_y = decode_mvd_component(&mut dec)?;
                mv_l0_q16 = (mvp_l0_q16.0 + mvd_x, mvp_l0_q16.1 + mvd_y);
            }
            if active_l1 {
                ref_idx_l1 = decode_ref_idx(&mut dec, num_active_l1)?;
                let mvd_x = decode_mvd_component(&mut dec)?;
                let mvd_y = decode_mvd_component(&mut dec)?;
                mv_l1_q16 = (mvp_l1_q16.0 + mvd_x, mvp_l1_q16.1 + mvd_y);
            }
            mv_field_l0.set(bx, by, mv_l0_q16);
            mv_field_l1.set(bx, by, mv_l1_q16);

            let cbf_y = read_tu_y_coded_flag(&mut dec, &mut ctxs.residual, false, false, false)?;
            let pred = match idc {
                InterPredIdc::L0 => {
                    let ref_p = ref_list_l0[ref_idx_l0 as usize];
                    let mut pred = vec![0u8; INTER_BLOCK_W * INTER_BLOCK_H];
                    mc_predict_subpel(
                        &mut pred,
                        &ref_p.luma,
                        cx,
                        cy,
                        INTER_BLOCK_W,
                        INTER_BLOCK_H,
                        mv_l0_q16,
                    )?;
                    pred
                }
                InterPredIdc::L1 => {
                    let ref_p = ref_list_l1[ref_idx_l1 as usize];
                    let mut pred = vec![0u8; INTER_BLOCK_W * INTER_BLOCK_H];
                    mc_predict_subpel(
                        &mut pred,
                        &ref_p.luma,
                        cx,
                        cy,
                        INTER_BLOCK_W,
                        INTER_BLOCK_H,
                        mv_l1_q16,
                    )?;
                    pred
                }
                InterPredIdc::Bi => {
                    let ref_l0_p = ref_list_l0[ref_idx_l0 as usize];
                    let ref_l1_p = ref_list_l1[ref_idx_l1 as usize];
                    let mut p0 = vec![0u8; INTER_BLOCK_W * INTER_BLOCK_H];
                    mc_predict_subpel(
                        &mut p0,
                        &ref_l0_p.luma,
                        cx,
                        cy,
                        INTER_BLOCK_W,
                        INTER_BLOCK_H,
                        mv_l0_q16,
                    )?;
                    let mut p1 = vec![0u8; INTER_BLOCK_W * INTER_BLOCK_H];
                    mc_predict_subpel(
                        &mut p1,
                        &ref_l1_p.luma,
                        cx,
                        cy,
                        INTER_BLOCK_W,
                        INTER_BLOCK_H,
                        mv_l1_q16,
                    )?;
                    average_bi(&p0, &p1, INTER_BLOCK_W, INTER_BLOCK_H)
                }
            };
            let recon = if cbf_y {
                let levels = crate::residual::decode_tb_coefficients(
                    &mut dec,
                    &mut ctxs.residual,
                    INTER_BLOCK_W,
                    INTER_BLOCK_H,
                    0,
                )?;
                reconstruct_inter_tb_from_levels(
                    &levels,
                    &pred,
                    INTER_BLOCK_W,
                    INTER_BLOCK_H,
                    slice_qp_y,
                )?
            } else {
                pred
            };
            for r in 0..INTER_BLOCK_H {
                for c in 0..INTER_BLOCK_W {
                    out.luma.samples[(cy + r) * out.luma.stride + (cx + c)] =
                        recon[r * INTER_BLOCK_W + c];
                }
            }

            // Round-63 — chroma 4-tap sub-pel MC (mirror encoder).
            const CW: usize = INTER_BLOCK_W / 2;
            const CH: usize = INTER_BLOCK_H / 2;
            let cx_c = cx / 2;
            let cy_c = cy / 2;
            let (pred_cb, pred_cr) = match idc {
                InterPredIdc::L0 => {
                    let ref_p = ref_list_l0[ref_idx_l0 as usize];
                    let mut cb = vec![0u8; CW * CH];
                    let mut cr = vec![0u8; CW * CH];
                    mc_predict_chroma_subpel(&mut cb, &ref_p.cb, cx_c, cy_c, CW, CH, mv_l0_q16)?;
                    mc_predict_chroma_subpel(&mut cr, &ref_p.cr, cx_c, cy_c, CW, CH, mv_l0_q16)?;
                    (cb, cr)
                }
                InterPredIdc::L1 => {
                    let ref_p = ref_list_l1[ref_idx_l1 as usize];
                    let mut cb = vec![0u8; CW * CH];
                    let mut cr = vec![0u8; CW * CH];
                    mc_predict_chroma_subpel(&mut cb, &ref_p.cb, cx_c, cy_c, CW, CH, mv_l1_q16)?;
                    mc_predict_chroma_subpel(&mut cr, &ref_p.cr, cx_c, cy_c, CW, CH, mv_l1_q16)?;
                    (cb, cr)
                }
                InterPredIdc::Bi => {
                    let ref_l0_p = ref_list_l0[ref_idx_l0 as usize];
                    let ref_l1_p = ref_list_l1[ref_idx_l1 as usize];
                    let cb = mc_predict_chroma_subpel_bi(
                        &ref_l0_p.cb,
                        mv_l0_q16,
                        &ref_l1_p.cb,
                        mv_l1_q16,
                        cx_c,
                        cy_c,
                        CW,
                        CH,
                    )?;
                    let cr = mc_predict_chroma_subpel_bi(
                        &ref_l0_p.cr,
                        mv_l0_q16,
                        &ref_l1_p.cr,
                        mv_l1_q16,
                        cx_c,
                        cy_c,
                        CW,
                        CH,
                    )?;
                    (cb, cr)
                }
            };
            for r in 0..CH {
                for c in 0..CW {
                    out.cb.samples[(cy_c + r) * out.cb.stride + (cx_c + c)] = pred_cb[r * CW + c];
                    out.cr.samples[(cy_c + r) * out.cr.stride + (cx_c + c)] = pred_cr[r * CW + c];
                }
            }
        }
    }

    let _ = dec.decode_terminate()?;
    Ok(out)
}

// =====================================================================
// Tests
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoder_pipeline::{encode_idr_with_residuals, psnr_y};

    fn translation_frame_pair(w: usize, h: usize, dx: i32) -> (PictureBuffer, PictureBuffer) {
        let mut a = PictureBuffer::yuv420_filled(w, h, 100);
        let mut b = PictureBuffer::yuv420_filled(w, h, 100);
        // Stripe pattern in the luma plane.
        for y in 0..h {
            for x in 0..w {
                let v = if (x / 8) % 2 == 0 { 80u8 } else { 180u8 };
                a.luma.samples[y * a.luma.stride + x] = v;
                let sx = ((x as i32 - dx).rem_euclid(w as i32)) as usize;
                let v2 = if (sx / 8) % 2 == 0 { 80u8 } else { 180u8 };
                b.luma.samples[y * b.luma.stride + x] = v2;
            }
        }
        (a, b)
    }

    #[test]
    fn round58_p_slice_translation_psnr_clears_35db() {
        // Frame A is the "I", frame B is the "P" with a 4-pixel
        // horizontal shift relative to A.
        let (frame_i, frame_p) = translation_frame_pair(64, 64, 4);
        let (_bs_i, rec_i) = encode_idr_with_residuals(&frame_i, 26).unwrap();
        let (bs_p, rec_p) = encode_p_slice(&frame_p, &rec_i, 26, 1, 8).unwrap();
        assert!(!bs_p.is_empty());
        let psnr = psnr_y(&frame_p.luma, &rec_p.luma).unwrap();
        assert!(
            psnr >= 35.0,
            "P-slice PSNR_Y {psnr:.2} dB < 35 dB on 4-px translation"
        );
    }

    #[test]
    fn round58_p_slice_no_motion_yields_small_payload() {
        let frame_a = {
            let (a, _) = translation_frame_pair(64, 64, 0);
            a
        };
        let (_, rec_i) = encode_idr_with_residuals(&frame_a, 26).unwrap();
        let (bs_p, rec_p) = encode_p_slice(&frame_a, &rec_i, 26, 1, 8).unwrap();
        // Allow generous headroom — the round-59 sub-pel refinement
        // adds at most a handful of bypass bins per block when it
        // declines the integer-pel result (which it generally does
        // not on identical frames).
        assert!(
            bs_p.len() < 800,
            "P-slice on identical frames ({} B) larger than the no-residual ceiling",
            bs_p.len(),
        );
        let psnr = psnr_y(&frame_a.luma, &rec_p.luma).unwrap();
        assert!(
            psnr >= 30.0,
            "Identical-frame P-slice PSNR_Y {psnr:.2} dB < 30 dB"
        );
    }

    #[test]
    fn round58_p_slice_decoder_roundtrips_through_own_decoder() {
        let (frame_i, frame_p) = translation_frame_pair(64, 64, 4);
        let (_, rec_i) = encode_idr_with_residuals(&frame_i, 26).unwrap();
        let (bs_p, enc_rec) = encode_p_slice(&frame_p, &rec_i, 26, 1, 8).unwrap();
        let dec_rec = decode_p_slice(&bs_p, &rec_i).unwrap();
        let mut diff_count = 0usize;
        let mut first_diff: Option<(usize, usize, u8, u8)> = None;
        for y in 0..frame_p.luma.height {
            for x in 0..frame_p.luma.width {
                let e = enc_rec.luma.samples[y * enc_rec.luma.stride + x];
                let d = dec_rec.luma.samples[y * dec_rec.luma.stride + x];
                if e != d {
                    diff_count += 1;
                    if first_diff.is_none() {
                        first_diff = Some((x, y, e, d));
                    }
                }
            }
        }
        assert_eq!(
            diff_count, 0,
            "encoder-side and decoder-side P-slice luma differ in {} samples (first at {:?})",
            diff_count, first_diff,
        );
    }

    #[test]
    fn round58_p_slice_synthetic_two_frame_fixture() {
        let make = |dx: i32| {
            let mut buf = PictureBuffer::yuv420_filled(64, 64, 100);
            for y in 16..32 {
                for x in 0..16 {
                    let xx = (16 + dx as usize + x).min(63);
                    buf.luma.samples[y * buf.luma.stride + xx] = 220;
                }
            }
            buf
        };
        let frame_i = make(0);
        let frame_p = make(4);
        let (_, rec_i) = encode_idr_with_residuals(&frame_i, 26).unwrap();
        let (bs_p, rec_p) = encode_p_slice(&frame_p, &rec_i, 26, 1, 8).unwrap();
        assert!(!bs_p.is_empty());
        let dec_rec = decode_p_slice(&bs_p, &rec_i).unwrap();
        assert_eq!(rec_p.luma.samples, dec_rec.luma.samples);
        let psnr = psnr_y(&frame_p.luma, &rec_p.luma).unwrap();
        assert!(
            psnr >= 35.0,
            "Synthetic 2-frame fixture PSNR_Y {psnr:.2} dB < 35 dB"
        );
    }

    #[test]
    fn full_search_int_finds_known_translation() {
        let mut a = PictureBuffer::yuv420_filled(64, 64, 100);
        let mut b = PictureBuffer::yuv420_filled(64, 64, 100);
        for y in 0..64 {
            for x in 0..64 {
                a.luma.samples[y * a.luma.stride + x] = if x < 32 { 60 } else { 200 };
                b.luma.samples[y * b.luma.stride + x] = if x < 28 { 60 } else { 200 };
            }
        }
        let (mv, sad) = full_search_int(
            &b.luma,
            26,
            16,
            INTER_BLOCK_W,
            INTER_BLOCK_H,
            &a.luma,
            (0, 0),
            8,
        );
        assert_eq!(mv.0, 4, "expected mv_x=4, got {mv:?} (sad={sad})");
        assert_eq!(sad, 0, "expected SAD=0 at the true mv, got {sad}");
    }

    #[test]
    fn mvp_for_reads_left_then_above_then_zero() {
        let mut f = MvField::new(2, 2);
        assert_eq!(f.mvp_for(0, 0), (0, 0));
        assert_eq!(f.mvp_for(1, 0), (0, 0));
        assert_eq!(f.mvp_for(0, 1), (0, 0));
        f.set(0, 0, (3, -2));
        assert_eq!(f.mvp_for(1, 0), (3, -2));
        assert_eq!(f.mvp_for(0, 1), (3, -2));
        assert_eq!(f.mvp_for(1, 1), (0, 0));
        f.set(0, 1, (-5, 7));
        assert_eq!(f.mvp_for(1, 1), (-5, 7));
    }

    #[test]
    fn mvd_component_round_trip() {
        // Sub-pel-magnitudes mixed with integer-pel and big values to
        // pin the round-59 1/16-pel encoding path.
        for &v in &[0i32, 1, -1, 4, -4, 8, -8, 12, -16, 64, -255, 1000, -1000] {
            let mut enc = ArithEncoder::new();
            encode_mvd_component(&mut enc, v).unwrap();
            let bytes = enc.finish();
            let mut padded = bytes.clone();
            padded.extend_from_slice(&[0u8; 32]);
            let mut dec = ArithDecoder::new(&padded).unwrap();
            let got = decode_mvd_component(&mut dec).unwrap();
            assert_eq!(got, v, "mvd round trip failed at v={v}");
        }
    }

    #[test]
    fn pslice_per_block_cabac_bins_roundtrip_two_blocks() {
        let qp = 26;
        let mut enc = ArithEncoder::new();
        let mut ctxs = PSliceCtxs::init(qp);
        let inc_skip = crate::ctx::ctx_inc_cu_skip_flag(false, false, false, false) as usize;
        let n_skip = ctxs.cu_skip.len() - 1;
        enc.encode_decision(&mut ctxs.cu_skip[inc_skip.min(n_skip)], 0)
            .unwrap();
        let inc_merge = crate::ctx::ctx_inc_general_merge_flag() as usize;
        let n_merge = ctxs.merge_flag.len() - 1;
        enc.encode_decision(&mut ctxs.merge_flag[inc_merge.min(n_merge)], 0)
            .unwrap();
        enc.encode_bypass(0).unwrap();
        enc.encode_bypass(0).unwrap();
        encode_mvd_component(&mut enc, -4).unwrap();
        encode_mvd_component(&mut enc, -8).unwrap();
        write_tu_y_coded_flag(&mut enc, &mut ctxs.residual, true, false, false, false).unwrap();
        let mut levels = vec![0i32; 64];
        levels[0] = 5;
        encode_tb_coefficients(&mut enc, &mut ctxs.residual, 8, 8, 0, &levels).unwrap();
        enc.encode_decision(&mut ctxs.cu_skip[inc_skip.min(n_skip)], 0)
            .unwrap();
        enc.encode_decision(&mut ctxs.merge_flag[inc_merge.min(n_merge)], 0)
            .unwrap();
        enc.encode_bypass(0).unwrap();
        enc.encode_bypass(0).unwrap();
        encode_mvd_component(&mut enc, 0).unwrap();
        encode_mvd_component(&mut enc, 0).unwrap();
        write_tu_y_coded_flag(&mut enc, &mut ctxs.residual, false, false, false, false).unwrap();
        enc.encode_terminate(1).unwrap();
        let mut bytes = enc.finish();
        bytes.extend_from_slice(&[0u8; 256]);

        let mut dec = ArithDecoder::new(&bytes).unwrap();
        let mut ctxs = PSliceCtxs::init(qp);
        let n_skip = ctxs.cu_skip.len() - 1;
        let n_merge = ctxs.merge_flag.len() - 1;
        let inc_skip = crate::ctx::ctx_inc_cu_skip_flag(false, false, false, false) as usize;
        let _ = dec
            .decode_decision(&mut ctxs.cu_skip[inc_skip.min(n_skip)])
            .unwrap();
        let inc_merge = crate::ctx::ctx_inc_general_merge_flag() as usize;
        let _ = dec
            .decode_decision(&mut ctxs.merge_flag[inc_merge.min(n_merge)])
            .unwrap();
        let _ = dec.decode_bypass().unwrap();
        let _ = dec.decode_bypass().unwrap();
        let mvd_x = decode_mvd_component(&mut dec).unwrap();
        let mvd_y = decode_mvd_component(&mut dec).unwrap();
        assert_eq!(mvd_x, -4);
        assert_eq!(mvd_y, -8);
        let cbf_y =
            read_tu_y_coded_flag(&mut dec, &mut ctxs.residual, false, false, false).unwrap();
        assert!(cbf_y);
        let recovered =
            crate::residual::decode_tb_coefficients(&mut dec, &mut ctxs.residual, 8, 8, 0).unwrap();
        assert_eq!(levels, recovered);
        let _ = dec
            .decode_decision(&mut ctxs.cu_skip[inc_skip.min(n_skip)])
            .unwrap();
        let _ = dec
            .decode_decision(&mut ctxs.merge_flag[inc_merge.min(n_merge)])
            .unwrap();
        let _ = dec.decode_bypass().unwrap();
        let _ = dec.decode_bypass().unwrap();
        let mvd_x = decode_mvd_component(&mut dec).unwrap();
        let mvd_y = decode_mvd_component(&mut dec).unwrap();
        assert_eq!(mvd_x, 0);
        assert_eq!(mvd_y, 0);
        let cbf_y =
            read_tu_y_coded_flag(&mut dec, &mut ctxs.residual, false, false, false).unwrap();
        assert!(!cbf_y);
        let _ = dec.decode_terminate().unwrap();
    }

    #[test]
    fn pslice_header_round_trip() {
        let hdr = PSliceHeader {
            slice_type: 1,
            poc_lsb: 7,
            num_ref_idx_l0_active_minus1: 0,
            slice_qp_delta: -3,
            width: 64,
            height: 48,
        };
        let bytes = write_pslice_header(&hdr);
        let got = read_pslice_header(&bytes).unwrap();
        assert_eq!(got, hdr);
    }

    // =================================================================
    // Round-59 — sub-pel motion compensation tests
    // =================================================================

    /// Build a two-frame pair with a sub-pel horizontal translation by
    /// pre-generating a high-resolution source at `oversample × width`,
    /// then resampling each frame at integer-pel positions in the
    /// up-sampled grid. `dx_q16` is the desired shift in 1/16-pel
    /// luma-sample units (e.g. 4 = ¼-pel, 8 = ½-pel).
    fn subpel_translation_pair(w: usize, h: usize, dx_q16: i32) -> (PictureBuffer, PictureBuffer) {
        // Use a sufficiently high oversample so 1/16-pel positions are
        // representable. 16× is the natural choice.
        let os = 16usize;
        let big_w = w * os;
        // Generate a smooth source: a gentle linear ramp with a few
        // band edges. Sub-pel-displacements of such a band-limited
        // signal can be reconstructed by an 8-tap filter to high PSNR.
        let big = |x_q16: i32| -> u8 {
            // Use a smoothly-varying brightness in the q16 axis.
            // Pattern: a sinusoid mixed with a low-amplitude offset so
            // dynamic range is healthy (~ 70 .. 180).
            let phase = (x_q16 as f64) / (big_w as f64) * (5.0 * std::f64::consts::PI);
            let v = 125.0 + 55.0 * phase.sin();
            v.clamp(0.0, 255.0) as u8
        };
        let mut a = PictureBuffer::yuv420_filled(w, h, 100);
        let mut b = PictureBuffer::yuv420_filled(w, h, 100);
        for y in 0..h {
            for x in 0..w {
                a.luma.samples[y * a.luma.stride + x] = big((x * os) as i32);
                // b is shifted by dx_q16/16 luma-samples.
                let xq = (x * os) as i32 - dx_q16;
                let xq = xq.clamp(0, (big_w - 1) as i32);
                b.luma.samples[y * b.luma.stride + x] = big(xq);
            }
        }
        (a, b)
    }

    #[test]
    fn round59_subpel_half_pel_translation_psnr() {
        // Half-pel translation: b samples are shifted by 0.5 luma px.
        let (frame_i, frame_p) = subpel_translation_pair(64, 64, 8);
        let (_, rec_i) = encode_idr_with_residuals(&frame_i, 26).unwrap();
        let (bs_p, rec_p) = encode_p_slice(&frame_p, &rec_i, 26, 1, 8).unwrap();
        assert!(!bs_p.is_empty());
        let psnr = psnr_y(&frame_p.luma, &rec_p.luma).unwrap();
        assert!(
            psnr >= 30.0,
            "Round-59 half-pel translation PSNR_Y {psnr:.2} dB < 30 dB"
        );
    }

    #[test]
    fn round59_subpel_quarter_pel_translation_psnr() {
        // Quarter-pel translation: b shifted by 0.25 luma px.
        let (frame_i, frame_p) = subpel_translation_pair(64, 64, 4);
        let (_, rec_i) = encode_idr_with_residuals(&frame_i, 26).unwrap();
        let (bs_p, rec_p) = encode_p_slice(&frame_p, &rec_i, 26, 1, 8).unwrap();
        assert!(!bs_p.is_empty());
        let psnr = psnr_y(&frame_p.luma, &rec_p.luma).unwrap();
        assert!(
            psnr >= 30.0,
            "Round-59 quarter-pel translation PSNR_Y {psnr:.2} dB < 30 dB"
        );
    }

    #[test]
    fn round59_integer_pel_regression_still_passes() {
        // The round-58 4-px-horizontal regression fixture: ensure the
        // sub-pel-aware path still reaches the >= 70 dB ballpark that
        // integer-pel MC achieves (the spec 8-tap filter at frac == 0
        // is the integer-pel sentinel and degrades to mc_copy_block_int).
        let (frame_i, frame_p) = translation_frame_pair(64, 64, 4);
        let (_, rec_i) = encode_idr_with_residuals(&frame_i, 26).unwrap();
        let (bs_p, rec_p) = encode_p_slice(&frame_p, &rec_i, 26, 1, 8).unwrap();
        assert!(!bs_p.is_empty());
        let psnr = psnr_y(&frame_p.luma, &rec_p.luma).unwrap();
        assert!(
            psnr >= 70.0,
            "Round-58 integer-pel fixture regressed to PSNR_Y {psnr:.2} dB (< 70 dB) after sub-pel wiring"
        );
    }

    #[test]
    fn round59_subpel_decoder_byte_identical() {
        // The decoder must reproduce the encoder's reconstruction
        // byte-for-byte even with sub-pel MVs in play.
        let (frame_i, frame_p) = subpel_translation_pair(64, 64, 8);
        let (_, rec_i) = encode_idr_with_residuals(&frame_i, 26).unwrap();
        let (bs_p, enc_rec) = encode_p_slice(&frame_p, &rec_i, 26, 1, 8).unwrap();
        let dec_rec = decode_p_slice(&bs_p, &rec_i).unwrap();
        assert_eq!(
            enc_rec.luma.samples, dec_rec.luma.samples,
            "round-59 encoder + decoder luma must match byte-for-byte at sub-pel MVs",
        );
    }

    // =================================================================
    // Round-60 — B-slice unit tests
    // =================================================================

    #[test]
    fn round60_bslice_header_round_trip() {
        let hdr = BSliceHeader {
            slice_type: 0,
            poc_lsb: 9,
            num_ref_idx_l0_active_minus1: 0,
            num_ref_idx_l1_active_minus1: 0,
            slice_qp_delta: 2,
            width: 64,
            height: 64,
        };
        let bytes = write_bslice_header(&hdr);
        let got = read_bslice_header(&bytes).unwrap();
        assert_eq!(got, hdr);
    }

    #[test]
    fn round60_inter_pred_idc_round_trip() {
        for idc in [InterPredIdc::L0, InterPredIdc::L1, InterPredIdc::Bi] {
            let mut enc = ArithEncoder::new();
            encode_inter_pred_idc(&mut enc, idc).unwrap();
            let mut bytes = enc.finish();
            bytes.extend_from_slice(&[0u8; 32]);
            let mut dec = ArithDecoder::new(&bytes).unwrap();
            let got = decode_inter_pred_idc(&mut dec).unwrap();
            assert_eq!(got, idc, "inter_pred_idc round-trip failed for {idc:?}");
        }
    }

    #[test]
    fn round60_average_bi_matches_spec_8_5_6_4() {
        // pred = (a + b + 1) >> 1 per §8.5.6.4 (simple-average bi-pred).
        let p0: Vec<u8> = (0..16).map(|i| i * 8).collect();
        let p1: Vec<u8> = (0..16).map(|i| 128 + i).collect();
        let avg = average_bi(&p0, &p1, 4, 4);
        for i in 0..16 {
            let expected = ((p0[i] as u16 + p1[i] as u16 + 1) >> 1) as u8;
            assert_eq!(avg[i], expected, "mismatch at i={i}");
        }
    }

    // =================================================================
    // Round-62 — multi-ref DPB unit tests
    // =================================================================

    #[test]
    fn round62_ref_idx_truncated_unary_round_trip_all_sizes() {
        // For each list length 1..=MAX_REF_PICS, every ref_idx in
        // [0, num_active) round-trips through encode_ref_idx +
        // decode_ref_idx with zero residual bits.
        for num_active in 1..=MAX_REF_PICS {
            for ref_idx in 0..num_active {
                let mut enc = ArithEncoder::new();
                encode_ref_idx(&mut enc, ref_idx as u8, num_active).unwrap();
                let mut bytes = enc.finish();
                bytes.extend_from_slice(&[0u8; 32]);
                let mut dec = ArithDecoder::new(&bytes).unwrap();
                let got = decode_ref_idx(&mut dec, num_active).unwrap();
                assert_eq!(
                    got, ref_idx as u8,
                    "ref_idx round-trip failed at num_active={num_active}, ref_idx={ref_idx}",
                );
            }
        }
    }

    #[test]
    fn round62_ref_idx_single_active_emits_no_bins() {
        // num_active == 1 ⇒ ref_idx is inferred to 0; the encoder must
        // not consume any bypass bins. Pin this via "encode followed by
        // a sentinel bypass round-trips clean".
        let mut enc = ArithEncoder::new();
        encode_ref_idx(&mut enc, 0, 1).unwrap();
        // Emit a sentinel that the decoder must read.
        enc.encode_bypass(1).unwrap();
        let mut bytes = enc.finish();
        bytes.extend_from_slice(&[0u8; 32]);
        let mut dec = ArithDecoder::new(&bytes).unwrap();
        let got = decode_ref_idx(&mut dec, 1).unwrap();
        assert_eq!(got, 0);
        let sentinel = dec.decode_bypass().unwrap();
        assert_eq!(sentinel, 1, "single-active ref_idx leaked a bin");
    }

    #[test]
    fn round62_ref_idx_clamps_at_cap() {
        // Trying to emit ref_idx == num_active - 1 (the largest valid
        // value) writes no terminator; an over-cap value clamps to cap.
        let mut enc = ArithEncoder::new();
        encode_ref_idx(&mut enc, 3, 3).unwrap(); // clamps to 2
        let mut bytes = enc.finish();
        bytes.extend_from_slice(&[0u8; 32]);
        let mut dec = ArithDecoder::new(&bytes).unwrap();
        let got = decode_ref_idx(&mut dec, 3).unwrap();
        assert_eq!(got, 2, "ref_idx clamp at cap not honoured");
    }

    #[test]
    fn round59_refine_subpel_returns_zero_at_perfect_int_match() {
        // When the integer-pel SAD is already 0, sub-pel refinement
        // must not waste effort moving the MV away from the integer
        // optimum (SAD ties are broken in favour of the existing best).
        let mut a = PictureBuffer::yuv420_filled(64, 64, 100);
        let mut b = PictureBuffer::yuv420_filled(64, 64, 100);
        for y in 0..64 {
            for x in 0..64 {
                a.luma.samples[y * a.luma.stride + x] = if x < 32 { 60 } else { 200 };
                b.luma.samples[y * b.luma.stride + x] = if x < 28 { 60 } else { 200 };
            }
        }
        // True integer-pel MV is +4 (see the round-58 test above).
        let (mv_q16, sad) = refine_subpel(
            &b.luma,
            26,
            16,
            INTER_BLOCK_W,
            INTER_BLOCK_H,
            &a.luma,
            (4, 0),
            0,
        )
        .unwrap();
        assert_eq!(sad, 0);
        assert_eq!(mv_q16, (4 * 16, 0));
    }

    // =================================================================
    // Round-63 — chroma sub-pel MC unit tests
    // =================================================================

    /// Integer-pel chroma MC (`mv_q16 == (0, 0)` and the destination
    /// equals the source position) reproduces the source byte-for-byte.
    #[test]
    fn round63_chroma_mc_zero_mv_is_identity() {
        let mut src = PicturePlane::filled(8, 8, 0);
        for y in 0..8 {
            for x in 0..8 {
                src.samples[y * src.stride + x] = (10 + x as u8 * 7 + y as u8 * 3) & 0xFF;
            }
        }
        let mut dst = vec![0u8; 4];
        mc_predict_chroma_subpel(&mut dst, &src, 2, 2, 2, 2, (0, 0)).unwrap();
        let mut expected: Vec<u8> = Vec::with_capacity(4);
        for r in 0..2 {
            for c in 0..2 {
                expected.push(src.samples[(2 + r) * src.stride + (2 + c)]);
            }
        }
        assert_eq!(dst, expected, "zero-MV chroma MC must be identity");
    }

    /// Integer-pel chroma MC with a non-zero integer MV reproduces the
    /// shifted block byte-for-byte (the §8.5.6.3.4 helper falls through
    /// to `mc_copy_block_int` at `xFracC == 0 && yFracC == 0`).
    #[test]
    fn round63_chroma_mc_int_mv_translates_block() {
        let mut src = PicturePlane::filled(16, 16, 0);
        for y in 0..16 {
            for x in 0..16 {
                src.samples[y * src.stride + x] = (x as u8) ^ ((y as u8) << 1);
            }
        }
        // mv = (+2 chroma px, -1 chroma px) in luma 1/16 units (chroma
        // 1/32 = 2 luma 1/16 ⇒ +2 chroma px = +64 luma 1/16).
        let mv_q16 = (2 * 32, -32);
        let mut dst = vec![0u8; 9];
        mc_predict_chroma_subpel(&mut dst, &src, 4, 4, 3, 3, mv_q16).unwrap();
        for r in 0..3 {
            for c in 0..3 {
                let sx = (4 + c + 2) as usize;
                let sy = (4 + r) as i32 - 1;
                let sy = sy.max(0) as usize; // clamp at top edge
                let want = src.samples[sy * src.stride + sx];
                assert_eq!(
                    dst[r * 3 + c],
                    want,
                    "int-MV chroma MC mismatch at (r={r}, c={c})",
                );
            }
        }
    }

    /// Half-pel chroma MC on a constant plane reconstructs the constant
    /// (DC-preserving property of any normalised interpolation filter).
    #[test]
    fn round63_chroma_mc_half_pel_constant_plane_dc_preserving() {
        let src = PicturePlane::filled(16, 16, 137);
        for x_frac in 0..32 {
            for y_frac in 0..32 {
                let mv_q16 = (x_frac, y_frac);
                let mut dst = vec![0u8; 4];
                mc_predict_chroma_subpel(&mut dst, &src, 4, 4, 2, 2, mv_q16).unwrap();
                for &v in &dst {
                    assert_eq!(
                        v, 137,
                        "chroma MC at frac (x={x_frac}, y={y_frac}) corrupted DC plane",
                    );
                }
            }
        }
    }

    /// BI chroma helper produces the per-list rounding average (eq.
    /// 8.5.6.4 form: `(p0 + p1 + 1) >> 1`).
    #[test]
    fn round63_chroma_bi_averages_per_list_predictions() {
        let mut s0 = PicturePlane::filled(8, 8, 0);
        let mut s1 = PicturePlane::filled(8, 8, 0);
        for i in 0..64 {
            s0.samples[i] = (i * 3) as u8;
            s1.samples[i] = (255 - i * 2) as u8;
        }
        let mut p0 = vec![0u8; 4];
        let mut p1 = vec![0u8; 4];
        mc_predict_chroma_subpel(&mut p0, &s0, 2, 2, 2, 2, (0, 0)).unwrap();
        mc_predict_chroma_subpel(&mut p1, &s1, 2, 2, 2, 2, (0, 0)).unwrap();
        let bi = mc_predict_chroma_subpel_bi(&s0, (0, 0), &s1, (0, 0), 2, 2, 2, 2).unwrap();
        for i in 0..4 {
            let want = (((p0[i] as u16) + (p1[i] as u16) + 1) >> 1) as u8;
            assert_eq!(bi[i], want, "BI chroma average mismatch at i={i}");
        }
    }
}
