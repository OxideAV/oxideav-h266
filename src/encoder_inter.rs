//! Round-58 — VVC P-slice encoder scaffold (single-reference, integer-pel only).
//!
//! This module provides a **minimum-viable** P-slice encode + roundtrip
//! decode path that closes the round-57 "inter-frame P-slice pipeline"
//! lacks tail. It is a scaffold:
//!
//! * **One reference picture** in the DPB (the previous decoded frame).
//! * **Integer-pel motion search only** (full-search ±N px window, SAD
//!   cost on luma 4×4 reference blocks — VVC §7.4.10 minimum-PU size).
//!   Sub-pel / fractional MVs is deferred to round 59.
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
//! ## Wire format (in-crate)
//!
//! Conceptually the wire follows the spec's slice-header → CU-syntax →
//! residual chain:
//!
//! 1. **Slice header (§7.4.4)** — bit-prelude (BitWriter):
//!    - `slice_type` (3 bits) — `1` for P (matches `SliceType::P` raw
//!      value, §7.4.4 Table 8).
//!    - `slice_pic_order_cnt_lsb` (u8) — picture-order-count low byte.
//!    - `num_ref_idx_l0_active_minus1` (`ue(v)`) — single ref so always 0.
//!    - `slice_qp_delta` (`se(v)`) — relative to the IDR slice QP.
//!    - byte_alignment().
//! 2. **Per-CU CABAC stream** — one [`crate::cabac_enc::ArithEncoder`]
//!    threading every block of the slice:
//!    - For each 4×4 inter block (in raster scan order):
//!      - `cu_skip_flag` — context-coded bin (§7.4.10).
//!      - `merge_flag` — context-coded bin.
//!      - When `merge_flag == 1`: `merge_idx` — bypass bin (we only
//!        emit `merge_idx == 0` so this is one bin).
//!      - When `merge_flag == 0`:
//!        - `inter_pred_idc` — bypass bin (`PRED_L0 == 0`).
//!        - `ref_idx_l0` — `ue(v)`-style bypass (we only emit 0).
//!        - `mvd_coding(mvd_x)` — `mvd_sign_flag` + `mvd_extra_bits`
//!          per §7.4.7.2 / §9.3.3.7. Implemented as: 1 bypass bit for
//!          sign, then magnitude as `ue(v)`-style bypass.
//!        - `mvd_coding(mvd_y)` — same.
//!      - `tu_y_coded_flag` — context-coded bin (Cb / Cr CBFs are
//!        forced 0 in this scaffold).
//!      - When `tu_y_coded_flag == 1`: residual coefficients via
//!        [`crate::residual_enc::encode_tb_coefficients`].
//!    - Stream terminated with `encode_terminate(1)` + `finish()`.
//! 3. **Wire layout** — the bit-prelude bytes are length-prefixed
//!    (little-endian `u32`) followed by the CABAC byte payload. Both
//!    pieces are wrapped in a fixed `OXAV_VVC_PSLICE` magic.
//!
//! The custom in-crate wire is **not** wrapped in a NAL unit — the
//! intent of round 58 is the codec building blocks (MC search, MV
//! emit, residual emit, MV reconstruction at the decoder, MC predict,
//! residual add) being correctness-verified. The full Annex-B NAL
//! integration of P-slices is gated on a future round once the
//! existing IDR `encode_idr_with_residuals_cfg` pipeline can be
//! threaded with the multi-frame DPB plumbing.

use oxideav_core::{Error, Result};

use crate::cabac::{ArithDecoder, ContextModel};
use crate::cabac_enc::ArithEncoder;
use crate::dequant::{dequantize_tb_flat, DequantParams};
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

/// Block size in luma samples for the round-58 inter scaffold. VVC's
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

// =====================================================================
// PreparedCu — round-58 sibling of encoder_pipeline::PreparedCu
// =====================================================================

/// One inter-CU's prepared state, mirroring the round-57
/// [`crate::encoder_pipeline`] internal `PreparedCu` style. The round-58
/// `InterPSlice` variant carries everything the second-pass CABAC walk
/// needs to emit the wire-side syntax: the L0 reference index, the
/// integer-pel motion vector, and the quantised luma residual levels.
///
/// Only the `InterPSlice` variant is ever constructed in this module;
/// the variant exists alongside the round-57 leaf / BT / TT shapes in
/// the spirit of "every tree walker must learn the new variant".
#[derive(Clone, Debug)]
pub enum PreparedCu {
    /// Round-58 P-slice inter CU: L0 ref + integer-pel MV + residual.
    InterPSlice {
        /// L0 reference index. The scaffold only emits 0 (single-ref
        /// DPB), but the field is kept explicit for spec parity.
        ref_idx: u8,
        /// Integer-pel motion vector `(mv_x, mv_y)` in luma samples.
        mv: (i16, i16),
        /// Spatial MV predictor used for `mvd = mv - mvp`.
        mvp: (i16, i16),
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
}

// =====================================================================
// Block-matching motion search — full-search SAD on luma
// =====================================================================

/// Sum of absolute differences between a `w × h` block at `(cx, cy)` in
/// `curr` and the same-size block at `(cx + mv_x, cy + mv_y)` in `ref_p`.
/// Out-of-bound reference samples are clamped to the picture edge.
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

/// Full-search integer-pel motion estimation. Returns the integer
/// motion vector `(mv_x, mv_y)` in luma samples that minimises SAD
/// against `ref_p` in a `±range` window, plus the achieved SAD.
///
/// `(cx, cy)` is the top-left luma sample of the current block; `(w, h)`
/// is the block size. The search centre is the supplied `mvp` (so the
/// search window is `[mvp_x - range, mvp_x + range]`); when `mvp ==
/// (0, 0)` this is the classical zero-motion-centred full search.
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

// =====================================================================
// MVP — minimal §7.4.7.3 spatial MVP picker
// =====================================================================

/// Picture-wide grid of per-block `(mv_x, mv_y)` values. Used for the
/// minimal §7.4.7.3 spatial MVP derivation: when filling block `(bx,
/// by)`, the predictor is the left neighbour's MV when `bx > 0`, else
/// the above neighbour's MV when `by > 0`, else zero.
///
/// `cells[by * cols + bx]` holds the integer-pel MV of the block whose
/// top-left luma sample is `(bx * INTER_BLOCK_W, by * INTER_BLOCK_H)`.
#[derive(Clone, Debug)]
pub struct MvField {
    pub cols: usize,
    pub rows: usize,
    pub cells: Vec<(i16, i16)>,
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
    pub fn mvp_for(&self, bx: usize, by: usize) -> (i16, i16) {
        if bx > 0 {
            self.cells[by * self.cols + (bx - 1)]
        } else if by > 0 {
            self.cells[(by - 1) * self.cols + bx]
        } else {
            (0, 0)
        }
    }

    pub fn set(&mut self, bx: usize, by: usize, mv: (i16, i16)) {
        self.cells[by * self.cols + bx] = mv;
    }
}

// =====================================================================
// Motion compensation — integer-pel sample copy
// =====================================================================

/// Predict a `w × h` luma block at `(dx, dy)` in `dst` from `ref_p`
/// shifted by integer-pel motion vector `(mv_x, mv_y)`. Reference
/// samples outside the picture are clamped to the nearest picture edge
/// (matches the spec's `Clip3(0, picW - 1, ...)` for the no-subpic
/// no-wrap case).
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
// neighbour's MV).

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

/// Round-58 — encode one P-slice for `curr` against single L0 reference
/// `ref_buf` at QP `slice_qp_y`. Returns the wire bytes plus the
/// reconstructed luma-only [`PictureBuffer`] (chroma planes are passed
/// through from `ref_buf` since the round-58 scaffold does not encode
/// chroma residuals).
///
/// Behaviour:
/// 1. Walks `curr.luma` in 4×4 raster blocks.
/// 2. For each block, derives `mvp` from the spatial MV field (left
///    else above else zero) and runs an integer-pel full search ±N
///    around the predictor.
/// 3. MC-predicts the block from `ref_buf.luma`, computes residual,
///    forward-DCTs + flat-quants, and decides on `cbf_y` (any non-zero
///    quantised level ⇒ 1).
/// 4. Emits the per-block CABAC bins and the `tu_y_coded_flag`-gated
///    residual into the slice's single arithmetic stream.
pub fn encode_p_slice(
    curr: &PictureBuffer,
    ref_buf: &PictureBuffer,
    slice_qp_y: i32,
    poc_lsb: u8,
    search_range: i32,
) -> Result<(Vec<u8>, PictureBuffer)> {
    let w = curr.luma.width;
    let h = curr.luma.height;
    if w != ref_buf.luma.width || h != ref_buf.luma.height {
        return Err(Error::invalid(
            "h266 P-slice: current and reference picture dimensions differ",
        ));
    }
    if w % INTER_BLOCK_W != 0 || h % INTER_BLOCK_H != 0 {
        return Err(Error::invalid(format!(
            "h266 P-slice scaffold requires picture {}x{} divisible by 16x16",
            w, h
        )));
    }
    let cols = w / INTER_BLOCK_W;
    let rows = h / INTER_BLOCK_H;

    let hdr = PSliceHeader {
        slice_type: 1, // P per §7.4.4 Table 8 (I=2, P=1, B=0 in our local mapping)
        poc_lsb,
        num_ref_idx_l0_active_minus1: 0,
        slice_qp_delta: slice_qp_y - SCAFFOLD_SLICE_QP,
        width: w as u32,
        height: h as u32,
    };
    let hdr_bytes = write_pslice_header(&hdr);

    let mut mv_field = MvField::new(cols, rows);
    let mut prepared: Vec<PreparedCu> = Vec::with_capacity(cols * rows);

    let mut rec = ref_buf.clone();
    // Reset the reconstructed luma plane to the reference and fill in
    // per-block (so an early block's reconstruction is visible to a
    // later block — though MC reads from `ref_buf` which is the
    // reference frame, not the in-progress reconstruction).
    rec.luma.samples.fill(0);

    let mut enc = ArithEncoder::new();
    let mut ctxs = PSliceCtxs::init(slice_qp_y);

    for by in 0..rows {
        for bx in 0..cols {
            let cx = bx * INTER_BLOCK_W;
            let cy = by * INTER_BLOCK_H;
            let mvp = mv_field.mvp_for(bx, by);

            // --- Motion search ---
            let (mv, _sad) = full_search_int(
                &curr.luma,
                cx,
                cy,
                INTER_BLOCK_W,
                INTER_BLOCK_H,
                &ref_buf.luma,
                mvp,
                search_range,
            );
            mv_field.set(bx, by, mv);

            // --- MC prediction into a scratch buffer ---
            let mut pred = vec![0u8; INTER_BLOCK_W * INTER_BLOCK_H];
            for r in 0..INTER_BLOCK_H {
                let ry = (cy as i32 + r as i32 + mv.1 as i32)
                    .clamp(0, ref_buf.luma.height as i32 - 1) as usize;
                for c in 0..INTER_BLOCK_W {
                    let rx = (cx as i32 + c as i32 + mv.0 as i32)
                        .clamp(0, ref_buf.luma.width as i32 - 1)
                        as usize;
                    pred[r * INTER_BLOCK_W + c] =
                        ref_buf.luma.samples[ry * ref_buf.luma.stride + rx];
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
            // §7.4.10 — cu_skip_flag = 0 (we always emit residual /
            // explicit MVD even when the MV equals the MVP, to keep
            // the round-58 walker uniform). The ctxInc derivation
            // (§9.3.4.2.2 eq. 1551) reads the left + above CU skip
            // flags; we have neither so both are `false`.
            let inc_skip = crate::ctx::ctx_inc_cu_skip_flag(false, false, false, false) as usize;
            let n_skip = ctxs.cu_skip.len() - 1;
            enc.encode_decision(&mut ctxs.cu_skip[inc_skip.min(n_skip)], 0)?;

            // §7.4.10 — general_merge_flag = 0 (explicit MVD path).
            let inc_merge = crate::ctx::ctx_inc_general_merge_flag() as usize;
            let n_merge = ctxs.merge_flag.len() - 1;
            enc.encode_decision(&mut ctxs.merge_flag[inc_merge.min(n_merge)], 0)?;

            // §7.4.10 — inter_pred_idc = PRED_L0 (encoded as 1 bypass
            // bit "0" — single-list ⇒ unambiguous).
            enc.encode_bypass(0)?;

            // §7.4.10 — ref_idx_l0 (single ref ⇒ 0; emit 1 bypass bit
            // "0" as the truncated-rice "value 0" path).
            enc.encode_bypass(0)?;

            // §7.4.7.2 — mvd_coding for x then y.
            let mvd = (mv.0 as i32 - mvp.0 as i32, mv.1 as i32 - mvp.1 as i32);
            encode_mvd_component(&mut enc, mvd.0)?;
            encode_mvd_component(&mut enc, mvd.1)?;

            // §7.4.10 — tu_y_coded_flag (CABAC). Signature is
            // (enc, ctxs, coded, bdpcm_y, isp_split, prev_tu_cbf_y).
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
                ref_idx: 0,
                mv,
                mvp,
                n_tb_w: INTER_BLOCK_W,
                n_tb_h: INTER_BLOCK_H,
                levels: if cbf_y { levels } else { Vec::new() },
                cbf_y,
            });
        }
    }

    // Terminate the slice stream.
    enc.encode_terminate(1)?;
    let cabac_bytes = enc.finish();

    // Round-58 silences the chroma path: pass chroma planes through
    // from the reference, since chroma residual emit is out of scope.
    rec.cb = ref_buf.cb.clone();
    rec.cr = ref_buf.cr.clone();

    // Wire layout: magic (14B) | hdr_len_le32 (4B) | hdr | cabac_len_le32 (4B) | cabac_bytes
    let mut out = Vec::with_capacity(PSLICE_MAGIC.len() + 8 + hdr_bytes.len() + cabac_bytes.len());
    out.extend_from_slice(PSLICE_MAGIC);
    out.extend_from_slice(&(hdr_bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(&hdr_bytes);
    out.extend_from_slice(&(cabac_bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(&cabac_bytes);

    let _ = prepared; // Round-58 keeps the prepared vector alive for
                      // future-round symmetry with the encoder_pipeline
                      // second-pass walk; the data is not yet consumed
                      // outside the per-block CABAC emit above.
    Ok((out, rec))
}

// =====================================================================
// Slice-level decode — round-trip side
// =====================================================================

/// Round-58 — decode one P-slice produced by [`encode_p_slice`]. Reads
/// the magic + slice header + CABAC stream, reconstructs each 4×4
/// inter block (MC predict from `ref_buf` shifted by the recovered MV
/// + dequantised inverse-DCT residual), and returns the reconstructed
/// luma in a fresh [`PictureBuffer`] (chroma is copied through from
/// `ref_buf` per the encoder-side scaffold scope).
pub fn decode_p_slice(bytes: &[u8], ref_buf: &PictureBuffer) -> Result<PictureBuffer> {
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
    // CABAC decoder reads ahead — append generous zero-pad so the
    // renormalisation tail never runs off the end of the slice. The
    // existing residual_enc tests use 64 B; we use 256 B to be safe
    // for the chained per-block reads of an entire P-slice.
    cabac_bytes.extend_from_slice(&[0u8; 256]);

    if hdr.slice_type != 1 {
        return Err(Error::invalid(format!(
            "h266 P-slice decode: expected slice_type=1, got {}",
            hdr.slice_type
        )));
    }
    let w = hdr.width as usize;
    let h = hdr.height as usize;
    if w != ref_buf.luma.width || h != ref_buf.luma.height {
        return Err(Error::invalid(format!(
            "h266 P-slice decode: header geometry {}x{} vs reference {}x{}",
            w, h, ref_buf.luma.width, ref_buf.luma.height
        )));
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

    let mut out = ref_buf.clone();
    out.luma.samples.fill(0);

    for by in 0..rows {
        for bx in 0..cols {
            let cx = bx * INTER_BLOCK_W;
            let cy = by * INTER_BLOCK_H;
            let mvp = mv_field.mvp_for(bx, by);

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
            // ref_idx_l0 — bypass.
            let _ = dec.decode_bypass()?;
            // mvd_coding x / y.
            let mvd_x = decode_mvd_component(&mut dec)?;
            let mvd_y = decode_mvd_component(&mut dec)?;
            let mv = (
                (mvp.0 as i32 + mvd_x).clamp(i16::MIN as i32, i16::MAX as i32) as i16,
                (mvp.1 as i32 + mvd_y).clamp(i16::MIN as i32, i16::MAX as i32) as i16,
            );
            mv_field.set(bx, by, mv);

            // tu_y_coded_flag.
            let cbf_y = read_tu_y_coded_flag(&mut dec, &mut ctxs.residual, false, false, false)?;
            // Predict the block from the reference + MV.
            let mut pred = vec![0u8; INTER_BLOCK_W * INTER_BLOCK_H];
            for r in 0..INTER_BLOCK_H {
                let ry = (cy as i32 + r as i32 + mv.1 as i32)
                    .clamp(0, ref_buf.luma.height as i32 - 1) as usize;
                for c in 0..INTER_BLOCK_W {
                    let rx = (cx as i32 + c as i32 + mv.0 as i32)
                        .clamp(0, ref_buf.luma.width as i32 - 1)
                        as usize;
                    pred[r * INTER_BLOCK_W + c] =
                        ref_buf.luma.samples[ry * ref_buf.luma.stride + rx];
                }
            }
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
        }
    }

    let _ = dec.decode_terminate()?;
    out.cb = ref_buf.cb.clone();
    out.cr = ref_buf.cr.clone();
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
        // Encode the IDR (frame I) to seed a real reconstructed reference.
        let (_bs_i, rec_i) = encode_idr_with_residuals(&frame_i, 26).unwrap();
        // Encode the P-slice using rec_i as the L0 reference.
        let (bs_p, rec_p) = encode_p_slice(&frame_p, &rec_i, 26, 1, 8).unwrap();
        assert!(!bs_p.is_empty());
        // The rec_p luma should match frame_p.luma to within rounding.
        let psnr = psnr_y(&frame_p.luma, &rec_p.luma).unwrap();
        assert!(
            psnr >= 35.0,
            "P-slice PSNR_Y {psnr:.2} dB < 35 dB on 4-px translation"
        );
    }

    #[test]
    fn round58_p_slice_no_motion_yields_small_payload() {
        // Same frame for I and P — every block should pick MV=(0,0)
        // and the residual should be near-zero, so the per-block
        // CABAC overhead is dominated by the cu_skip / merge_flag /
        // ref_idx / mvd_zero / tu_y_coded(false) bins (no residual).
        let frame_a = {
            let (a, _) = translation_frame_pair(64, 64, 0);
            a
        };
        let (_, rec_i) = encode_idr_with_residuals(&frame_a, 26).unwrap();
        let (bs_p, rec_p) = encode_p_slice(&frame_a, &rec_i, 26, 1, 8).unwrap();
        // The P-slice payload is bounded by the per-block syntax
        // overhead × number of blocks, plus the slice header. With
        // 4×4 blocks on a 64×64 picture that's 256 blocks × ~9 bins
        // each ≈ 290 B. Allow generous headroom for CABAC encoding
        // overhead.
        assert!(
            bs_p.len() < 600,
            "P-slice on identical frames ({} B) larger than the no-residual ceiling",
            bs_p.len(),
        );
        // Reconstruction must match the reference (which already
        // approximates the source after IDR encode) within rounding.
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
        // Feed the encoded P-slice back through our own decoder; the
        // resulting reconstruction must be byte-identical to the one
        // the encoder kept internally.
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
        // Synthetic "two-frame fixture": a single bright square that
        // moves 4 px to the right between frames. Mirrors what an
        // ffmpeg -c:v libvvenc -an -frames:v 2 fixture would look
        // like at the byte level for a moving object on a flat
        // background.
        let make = |dx: i32| {
            let mut buf = PictureBuffer::yuv420_filled(64, 64, 100);
            // 16×16 bright square at (16 + dx, 16). Multiple of the
            // round-58 8×8 block size so the search block fully sits
            // on the moving content.
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
        // Build a synthetic pair with a vertical edge so SAD has a
        // unique minimum at the true MV. `a` has a vertical edge at
        // column 32; `b` has the same edge at column 28 (shifted left
        // by 4). Position the search block straddling b's edge
        // (cols 26..29 with 4-wide block) so any MV ≠ +4 mismatches.
        let mut a = PictureBuffer::yuv420_filled(64, 64, 100);
        let mut b = PictureBuffer::yuv420_filled(64, 64, 100);
        for y in 0..64 {
            for x in 0..64 {
                a.luma.samples[y * a.luma.stride + x] = if x < 32 { 60 } else { 200 };
                b.luma.samples[y * b.luma.stride + x] = if x < 28 { 60 } else { 200 };
            }
        }
        // Block at b's cols 26..29 contains "60 60 200 200" (b's edge
        // at 28). The matching window in a (which has its edge at 32)
        // is cols 30..33 → mv_x = +4 places a's window at b's pos.
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
        // Default: zero everywhere.
        assert_eq!(f.mvp_for(0, 0), (0, 0));
        assert_eq!(f.mvp_for(1, 0), (0, 0));
        assert_eq!(f.mvp_for(0, 1), (0, 0));
        // Set (0, 0) → mvp(1, 0) reads left.
        f.set(0, 0, (3, -2));
        assert_eq!(f.mvp_for(1, 0), (3, -2));
        // mvp(0, 1) reads above (also (0, 0)).
        assert_eq!(f.mvp_for(0, 1), (3, -2));
        // mvp(1, 1) reads left ((0, 1) which is still 0).
        assert_eq!(f.mvp_for(1, 1), (0, 0));
        f.set(0, 1, (-5, 7));
        assert_eq!(f.mvp_for(1, 1), (-5, 7));
    }

    #[test]
    fn mvd_component_round_trip() {
        for &v in &[0i32, 1, -1, 7, -13, 64, -255, 1000, -1000] {
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
        // Mirror the encode_p_slice per-block bin emission exactly,
        // then decode the same sequence and assert match. This isolates
        // any CABAC en/dec asymmetry that the full P-slice walker
        // would otherwise drown out.
        let qp = 26;
        let mut enc = ArithEncoder::new();
        let mut ctxs = PSliceCtxs::init(qp);
        // Block 1 — emit cu_skip(0), merge(0), inter_pred(0), ref_idx(0),
        // mvd_x = -4, mvd_y = -8, tu_y_coded(true), and a tiny levels
        // block to stand in for residual.
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
        // Block 2 — same shape, mvd = (0, 0), no residual.
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
        // Block 1.
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
        // Block 2.
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
}
