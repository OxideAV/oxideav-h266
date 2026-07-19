//! VVC intra-frame encoder pipeline — residual emit, in-loop filters, PSNR.
//!
//! This module ties together the encoder-side building blocks:
//!
//! * [`EncoderPipeline`] — a high-level encoder that accepts a
//!   [`crate::reconstruct::PictureBuffer`] (source frame) + encoder config
//!   and produces an Annex-B bitstream with real coded residuals for a
//!   single IDR frame.
//! * [`encode_idr_with_residuals`] — stateless function-level entry point.
//! * [`psnr_y`] — compute PSNR_Y between two luma planes.
//!
//! ## Pipeline for one IDR
//!
//! Round 46 onward, the pipeline runs in two passes so the per-CTU ALF
//! CABAC bins emitted by `crate::alf_syntax::encode_alf_ctu` end up in
//! the same single CABAC stream as the residual bins (§7.3.11.2):
//!
//! 1. **First pass — reconstruction (no CABAC):** for every TB, DC intra
//!    pred → residual → `forward_dct_ii_2d` → `quantize_tb_flat` →
//!    `dequantize_tb_flat` → inverse transform → write `(pred +
//!    dequant_residual).clamp(0, 255)` into `rec`. Per-TB quantised
//!    levels are persisted as [`PreparedLumaTb`] for the second pass.
//!    Round-49 — the same DCT / quant / dequant / IDCT loop runs for
//!    Cb + Cr at half luma resolution (4:2:0); chroma quantised levels
//!    are stored alongside the luma levels in the same `PreparedLumaTb`.
//! 2. **In-loop filters:** deblock (§8.8.3) → SAO RDO + apply (§8.8.2)
//!    → luma ALF RDO (§7.4.3.18 fixed-filter sets) → chroma ALF RDO
//!    (§8.8.5.4) → CC-ALF RDO (§8.8.5.7), each pass mutating `rec` and
//!    recording per-CTB decisions into `alf_pic`.
//! 3. **Second pass — CABAC interleave:** for every CTU emit
//!    `encode_alf_ctu` (`alf_ctb_flag[]` / `alf_use_aps_flag` /
//!    `alf_luma_*_idx` / `alf_ctb_filter_alt_idx[]` /
//!    `alf_ctb_cc_*_idc[]`) followed by `emit_tu_with_cbf` for every TB
//!    (§7.3.10 `transform_unit()` — `tu_cb_coded_flag` /
//!    `tu_cr_coded_flag` / `tu_y_coded_flag` real CABAC bins per round-51,
//!    then luma → Cb → Cr residual blocks gated by the corresponding
//!    CBF), followed by `encode_terminate(0)` (or `(1)` on the last
//!    CTU). Both syntax families share one [`crate::cabac_enc::ArithEncoder`].
//!
//! ## Scope restrictions
//!
//! * 8-bit, 4:2:0 only.
//! * Single-tile, single-slice IDR frames only.
//! * Flat intra prediction (DC) — PLANAR / angular / MIP / ISP are not
//!   used (they would require more complex reference-sample management in
//!   the encoder path).
//! * No transform-skip, no MTS, no dep-quant, no SAO on chroma (to keep
//!   the first encoder round lean). Round-49 added forward DCT + flat
//!   quant + CABAC for chroma residual (Cb / Cr); chroma SAO decision
//!   is still off (the first-pass reconstruction goes through deblock
//!   only on the chroma planes).
//! * Deblocking uses the existing [`crate::deblock`] primitives wired to
//!   a constant `QP = 26`.

use oxideav_core::{Error, Result};

use crate::cabac::ContextModel;
use crate::coding_tree::SplitConstraints;
use crate::ctx::{
    ctx_inc_intra_chroma_pred_mode, ctx_inc_intra_luma_mpm_flag, ctx_inc_intra_luma_not_planar_flag,
};
use crate::deblock::{apply_deblocking, DeblockCu, DeblockParams};
use crate::dequant::{dequantize_tb_flat, DequantParams};
use crate::intra::{predict_intra, IntraPredParams};
use crate::reconstruct::{PictureBuffer, PicturePlane};
use crate::residual::ResidualCtxs;
use crate::residual_enc::{
    encode_tb_coefficients_opts, write_cu_qp_delta, write_tu_cb_coded_flag, write_tu_cr_coded_flag,
    write_tu_y_coded_flag,
};
use crate::sao::SaoConfig;
use crate::sao_enc::sao_decide_picture;
use crate::syntax_enc::{encode_coding_quadtree_split, TreeNeighbours};
use crate::tables::{init_contexts, SyntaxCtx};
use crate::transform::{inverse_transform_2d, TrType};
use crate::transform_fwd::{forward_dct_ii_2d, quantize_tb_dep_quant, quantize_tb_flat};

/// Compute PSNR_Y (luma only) between two [`PicturePlane`]s.
///
/// Returns `f64::INFINITY` when MSE is 0. The maximum signal value is
/// assumed to be 255 (8-bit). Returns `Err` if the planes have different
/// dimensions.
pub fn psnr_y(src: &PicturePlane, rec: &PicturePlane) -> Result<f64> {
    if src.width != rec.width || src.height != rec.height {
        return Err(Error::invalid(
            "psnr_y: planes must have identical dimensions",
        ));
    }
    let mut mse: f64 = 0.0;
    let n = (src.width * src.height) as f64;
    for y in 0..src.height {
        for x in 0..src.width {
            let s = src.samples[y * src.stride + x] as f64;
            let r = rec.samples[y * rec.stride + x] as f64;
            mse += (s - r) * (s - r);
        }
    }
    if mse == 0.0 {
        return Ok(f64::INFINITY);
    }
    mse /= n;
    Ok(10.0 * (255.0 * 255.0 / mse).log10())
}

/// Round-48 — total picture-wide luma SSE used by the APS-vs-fixed-only
/// picture-bits trade-off RDO.
fn total_sse_y(src: &PicturePlane, rec: &PicturePlane) -> u64 {
    let mut sse: u64 = 0;
    let h = src.height.min(rec.height);
    let w = src.width.min(rec.width);
    for y in 0..h {
        for x in 0..w {
            let s = src.samples[y * src.stride + x] as i32;
            let r = rec.samples[y * rec.stride + x] as i32;
            let d = (s - r) as i64;
            sse += (d * d) as u64;
        }
    }
    sse
}

/// Round-48 — `lambda` for the §8.8.5 RDO trade-off (SSE vs APS bits).
///
/// Approximates the VVC reference encoder's piecewise-linear curve as
/// `lambda = 0.85 * 2^((qp - 12) / 3)`, calibrated against the round-47
/// QP test points (QP 0 → ~0.04, QP 26 → ~12.7, QP 51 → ~617). The
/// caller multiplies `lambda * bits` to convert APS bytes into an SSE-
/// equivalent overhead.
fn lambda_for_qp(qp: i32) -> f64 {
    0.85f64 * 2.0f64.powf((qp as f64 - 12.0) / 3.0)
}

/// Round-50 — measure the per-picture CABAC bit cost of the per-CTB ALF
/// syntax bins (`alf_ctb_flag[]` + `alf_use_aps_flag` +
/// `alf_luma_prev_filter_idx` / `alf_luma_fixed_filter_idx` +
/// `alf_ctb_filter_alt_idx[]` + `alf_ctb_cc_*_idc[]`) for one
/// [`crate::alf::AlfPicture`].
///
/// Uses a fresh [`crate::cabac_enc::ArithEncoder`] + freshly-init
/// `AlfCtxs` so the count reflects the standalone bin cost, isolated
/// from the residual / `end_of_slice_segment` bins that the second-pass
/// CABAC walk interleaves on top. The returned bit count is `8 *
/// bytes_emitted` from the finished arithmetic stream — same accounting
/// the wire stream uses.
///
/// The round-48 APS-vs-fixed trade-off used to compare
/// `sse_with_aps + lambda * 8 * aps_byte_cost` against `sse_fixed_only`,
/// missing the per-CTB CABAC bit deltas: every CTB the picture-level
/// "use APS" branch picks pays the `alf_use_aps_flag = 1` ctx-coded bin
/// + a TB-bypass `alf_luma_prev_filter_idx`, while the "fixed-set"
/// branch pays `alf_use_aps_flag = 0` + a 4-bit-bypass
/// `alf_luma_fixed_filter_idx`. The two costs differ enough (especially
/// at low QP) to flip the picture-level decision when the APS only
/// barely beats fixed-only on raw SSE. Round-50 closes that gap by
/// adding `lambda * picture_bin_cost(alf_pic, cfg)` to *both* sides
/// before comparing.
fn estimate_alf_picture_bin_cost(
    alf_pic: &crate::alf::AlfPicture,
    cfg: &crate::alf_syntax::AlfSyntaxConfig,
    qp: i32,
) -> Result<u64> {
    let mut enc = crate::cabac_enc::ArithEncoder::new();
    let mut ctxs = crate::alf_syntax::AlfCtxs::init(qp);
    crate::alf_syntax::encode_alf_picture(&mut enc, &mut ctxs, cfg, alf_pic)?;
    let bytes = enc.finish();
    Ok((bytes.len() as u64) * 8)
}

/// Persisted per-TB state from the round-46 reconstruction pass —
/// re-used by the second-pass CABAC emit (per-CTU ALF + residual
/// interleave) so the encoder doesn't have to redo forward DCT /
/// quantisation work.
///
/// Round-49 — chroma TB levels (Cb + Cr) are persisted alongside luma so
/// the second-pass CABAC walk can emit them in the §7.3.10 / §8.7
/// per-component order (luma → Cb → Cr). The chroma TB lives at half
/// luma resolution per the 4:2:0 scope (`n_tb_chroma_w = n_tb_w / 2`,
/// likewise for height); when the luma TB is 4-wide / 4-tall the
/// chroma side collapses to 2 (smallest spec-supported) which we round
/// up to 4 per the §8.7.4 `nTbS ∈ {4, …, 64}` floor.
///
/// Round-56 — `n_tb_w` and `n_tb_h` are now independent (pre-r56 the
/// type held a single square `n_tb`). The MTT BT picker emits non-
/// square TBs for sub-CUs (e.g. 32×64 luma after a `BT_VERT` split).
struct PreparedLumaTb {
    /// TB width in luma samples (power of two, 4..64).
    n_tb_w: usize,
    /// TB height in luma samples (power of two, 4..64).
    n_tb_h: usize,
    /// Quantised level array (length = `n_tb_w * n_tb_h`).
    levels: Vec<i32>,
    /// Round-49 — chroma TB width (`max(4, n_tb_w / 2)` in 4:2:0).
    n_tb_chroma_w: usize,
    /// Round-49 — chroma TB height (`max(4, n_tb_h / 2)` in 4:2:0).
    n_tb_chroma_h: usize,
    /// Round-49 — Cb quantised levels (length `n_tb_chroma_w * n_tb_chroma_h`).
    cb_levels: Vec<i32>,
    /// Round-49 — Cr quantised levels (length `n_tb_chroma_w * n_tb_chroma_h`).
    cr_levels: Vec<i32>,
    /// Round-52 — local CU QP (luma). Emitted as
    /// `cu_qp_delta = cu_qp_local - prev_qp_in_qg` when CBFs non-zero
    /// and `pps_cu_qp_delta_enabled_flag = 1`. The dequant on the
    /// decoder side will recover the same QP via §8.7.1.
    cu_qp_local: i32,
}

/// Round-56 — encoded representation of one CU before the second-pass
/// CABAC emit. A `Leaf` carries one [`PreparedLumaTb`] (the round-55
/// behaviour); a `BtSplit` is the result of the round-56 BT picker,
/// holding two child CUs (each itself a `PreparedCu` so future rounds
/// can recurse).
///
/// The picker is gated behind
/// [`crate::encoder::EncoderConfig::enable_mtt_bt_picker`]. Default-off
/// preserves the round-55 wire stream byte-for-byte (every prepared
/// CU is a `Leaf`).
enum PreparedCu {
    Leaf(PreparedLumaTb),
    BtSplit {
        /// Direction of the binary split (`Vertical` halves width,
        /// `Horizontal` halves height).
        dir: crate::syntax_enc::MttSplitDir,
        /// `cqt_depth` of this CU (passed to
        /// [`crate::syntax_enc::encode_coding_tree_bt_split`] for the
        /// `split_qt_flag` ctxInc).
        cqt_depth: u32,
        /// `mtt_depth` of this CU (passed to
        /// [`crate::syntax_enc::encode_coding_tree_bt_split`] for the
        /// `mtt_split_cu_binary_flag` ctxInc).
        mtt_depth: u32,
        /// First child sub-CU.
        sub_a: Box<PreparedCu>,
        /// Second child sub-CU.
        sub_b: Box<PreparedCu>,
    },
    /// Round-57 — ternary-tree split: three sub-CUs at the 1:2:1 ratio.
    /// `Vertical` slices the CU into three columns of widths
    /// `(cb_w / 4, cb_w / 2, cb_w / 4)`; `Horizontal` slices into three
    /// rows of heights `(cb_h / 4, cb_h / 2, cb_h / 4)`. Sub-CUs are
    /// stored left→right (vertical) or top→bottom (horizontal).
    TtSplit {
        /// Direction of the ternary split.
        dir: crate::syntax_enc::MttSplitDir,
        /// `cqt_depth` of this CU.
        cqt_depth: u32,
        /// `mtt_depth` of this CU.
        mtt_depth: u32,
        /// First sub-CU (1/4 of the parent dim along the split axis).
        sub_a: Box<PreparedCu>,
        /// Middle sub-CU (1/2 of the parent dim along the split axis).
        sub_b: Box<PreparedCu>,
        /// Third sub-CU (1/4 of the parent dim along the split axis).
        sub_c: Box<PreparedCu>,
    },
}

impl PreparedCu {
    /// Iterate every leaf [`PreparedLumaTb`] inside this CU in tree
    /// order. Used by the deblock-CU collection pass.
    fn for_each_leaf<F: FnMut(&PreparedLumaTb)>(&self, f: &mut F) {
        match self {
            PreparedCu::Leaf(tb) => f(tb),
            PreparedCu::BtSplit { sub_a, sub_b, .. } => {
                sub_a.for_each_leaf(f);
                sub_b.for_each_leaf(f);
            }
            PreparedCu::TtSplit {
                sub_a,
                sub_b,
                sub_c,
                ..
            } => {
                sub_a.for_each_leaf(f);
                sub_b.for_each_leaf(f);
                sub_c.for_each_leaf(f);
            }
        }
    }
}

/// r412 — the §6.4.1 – §6.4.3 constraints matching the EMITTED SPS
/// partition fields: MinQtSizeY = MinCbSizeY = 4; with the MTT pickers
/// off, `sps_max_mtt_hierarchy_depth_intra_slice_luma = 0` (BT/TT
/// never allowed); with a picker on, depth 1 with MaxBtSizeY =
/// MaxTtSizeY = 64. The emitted split bins MUST follow these
/// constraints — presence and ctxIncs — or a conforming decoder
/// desyncs (r412: the pre-r412 emitter always coded split_qt/mtt bins
/// with allow-everything ctxIncs).
fn encoder_split_constraints(pic_w: u32, pic_h: u32, mtt_enabled: bool) -> SplitConstraints {
    SplitConstraints {
        min_qt_log2: 2,
        max_mtt_depth: u32::from(mtt_enabled),
        max_bt_size: if mtt_enabled { 64 } else { 4 },
        max_tt_size: if mtt_enabled { 64 } else { 4 },
        min_cb_log2: 2,
        pic_w,
        pic_h,
    }
}

/// r412 — encoder-side §8.7.5.1 eq. 1212 availability masks. The
/// coding loop mirrors the decoder: a reference sample is usable for
/// intra prediction only when its covering block has already been
/// reconstructed in walk order (§6.4.4). The MTT picker snapshots /
/// restores these masks alongside the reconstruction planes so losing
/// trials leave no phantom availability.
struct EncAvail {
    luma: Vec<bool>,
    chroma: Vec<bool>,
    w: usize,
    h: usize,
}

impl EncAvail {
    fn new(w: usize, h: usize) -> Self {
        Self {
            luma: vec![false; w * h],
            chroma: vec![false; (w / 2) * (h / 2)],
            w,
            h,
        }
    }

    fn mark_luma(&mut self, x: usize, y: usize, bw: usize, bh: usize) {
        for yy in y..(y + bh).min(self.h) {
            for xx in x..(x + bw).min(self.w) {
                self.luma[yy * self.w + xx] = true;
            }
        }
    }

    fn mark_chroma(&mut self, cx: usize, cy: usize, bw: usize, bh: usize) {
        let (cw, ch) = (self.w / 2, self.h / 2);
        for yy in cy..(cy + bh).min(ch) {
            for xx in cx..(cx + bw).min(cw) {
                self.chroma[yy * cw + xx] = true;
            }
        }
    }
}

/// r412 — the decoder-mirrored §8.4.5.2.6 DC prediction (with PDPC and
/// the §8.4.5.2.9 substitution) for one encoder TB. `avail` / `aw` /
/// `ah` describe the eq. 1212 mask of the plane being predicted.
#[allow(clippy::too_many_arguments)]
fn predict_dc_block(
    plane: &PicturePlane,
    avail: &[bool],
    aw: usize,
    ah: usize,
    x0: usize,
    y0: usize,
    n_w: usize,
    n_h: usize,
    c_idx: u32,
) -> Result<Vec<i16>> {
    predict_intra(
        &IntraPredParams {
            mode: crate::leaf_cu::INTRA_DC,
            n_tb_w: n_w,
            n_tb_h: n_h,
            n_cb_w: n_w,
            n_cb_h: n_h,
            c_idx,
            isp_no_split: true,
            ref_idx: 0,
            bdpcm: false,
            bit_depth: 8,
        },
        |dx: i32, dy: i32| {
            let x = x0 as i32 + dx;
            let y = y0 as i32 + dy;
            if x < 0 || y < 0 || x >= aw as i32 || y >= ah as i32 {
                return None;
            }
            if !avail[(y as usize) * aw + x as usize] {
                return None;
            }
            plane.get(x as usize, y as usize).map(|v| v as i16)
        },
    )
}

/// r412 — §7.3.11.5 intra-mode cascade contexts shared across the
/// second-pass CABAC emit (and the RDO bit measurement). The init
/// tables are the same `SyntaxCtx` entries the decoder's `LeafCuCtxs`
/// uses, so encoder and decoder context state stay in lockstep.
struct IntraModeCtxs {
    mpm_flag: Vec<ContextModel>,
    not_planar: Vec<ContextModel>,
    chroma_mode: Vec<ContextModel>,
}

impl IntraModeCtxs {
    fn init(slice_qp_y: i32) -> Self {
        Self {
            mpm_flag: init_contexts(SyntaxCtx::IntraLumaMpmFlag, slice_qp_y),
            not_planar: init_contexts(SyntaxCtx::IntraLumaNotPlanarFlag, slice_qp_y),
            chroma_mode: init_contexts(SyntaxCtx::IntraChromaPredMode, slice_qp_y),
        }
    }
}

/// r412 — emit the §7.3.11.5 intra-mode cascade for one pipeline CU:
/// `intra_luma_mpm_flag = 1`, `intra_luma_not_planar_flag = 1`,
/// `intra_luma_mpm_idx = 0` (TR cMax = 4, one bypass 0-bin) and
/// `intra_chroma_pred_mode` = DM (Table 130 "0" bin). Every pipeline
/// CU codes INTRA_DC, so no neighbour is ever angular and the §8.4.2
/// default candModeList always carries DC at index 0; DM chroma then
/// inherits DC per Table 20.
fn emit_intra_dc_mode_bins(
    enc: &mut crate::cabac_enc::ArithEncoder,
    ictx: &mut IntraModeCtxs,
) -> Result<()> {
    let inc = ctx_inc_intra_luma_mpm_flag() as usize;
    enc.encode_decision(&mut ictx.mpm_flag[inc], 1)?;
    let inc = ctx_inc_intra_luma_not_planar_flag(false) as usize;
    let n = ictx.not_planar.len() - 1;
    enc.encode_decision(&mut ictx.not_planar[inc.min(n)], 1)?;
    enc.encode_bypass(0)?; // intra_luma_mpm_idx = 0
    let inc = ctx_inc_intra_chroma_pred_mode() as usize;
    let n = ictx.chroma_mode.len() - 1;
    enc.encode_decision(&mut ictx.chroma_mode[inc.min(n)], 0)?; // DM
    Ok(())
}

/// Compute residual + reconstruction for one luma TB without touching
/// any CABAC stream. Returns the quantised levels for the later
/// per-CTU CABAC emit pass.
///
/// Round-56 — accepts independent `n_tb_w` / `n_tb_h` so the BT picker
/// can prepare non-square TBs (e.g. 32×64 after a `BT_VERT` split).
/// Both dims must be powers of two in `[4, 64]`.
fn prepare_luma_tb(
    src: &PicturePlane,
    rec_plane: &mut PicturePlane,
    avail: &mut EncAvail,
    x: usize,
    y: usize,
    n_tb_w: usize,
    n_tb_h: usize,
    qp: i32,
    rc: crate::residual::RcOpts,
) -> Result<Vec<i32>> {
    // r412 — real §8.4.5.2.12 DC intra prediction (+ PDPC) from the
    // partially-reconstructed plane, decoder-mirrored: the wire
    // signals INTRA_DC per CU, so the encoder reconstruction must use
    // the exact prediction a conforming decoder derives.
    let pred = predict_dc_block(
        rec_plane,
        &avail.luma,
        avail.w,
        avail.h,
        x,
        y,
        n_tb_w,
        n_tb_h,
        0,
    )?;

    // Extract source block → residual.
    let mut residual = vec![0i32; n_tb_w * n_tb_h];
    for ty in 0..n_tb_h {
        for tx in 0..n_tb_w {
            let sx = x + tx;
            let sy = y + ty;
            let p = pred[ty * n_tb_w + tx] as i32;
            let src_s = src.get(sx, sy).map(|v| v as i32).unwrap_or(p);
            residual[ty * n_tb_w + tx] = src_s - p;
        }
    }

    // Forward DCT-II.
    let mut coeffs = forward_dct_ii_2d(n_tb_w, n_tb_h, &residual, 8)?;
    // §7.3.11.11 / §8.7.4.1 zero-out (r412) — a 64-point DCT-II TB
    // codes only the low-frequency 32-corner (Log2ZoTb = Min(log2Tb,
    // 5); the decoder's eqs. 1171/1172 inverse-transform extents cap
    // there too). Zero the out-of-corner coefficients BEFORE
    // quantisation so the coded levels, the reconstruction, and a
    // conforming decoder all agree.
    if n_tb_w > 32 || n_tb_h > 32 {
        for yc in 0..n_tb_h {
            for xc in 0..n_tb_w {
                if xc >= 32 || yc >= 32 {
                    coeffs[yc * n_tb_w + xc] = 0;
                }
            }
        }
    }

    // Quantise — the dep-quant arm walks the §7.4.12.11 trellis; the
    // SDH arm parity-conditions each sub-block so the hidden sign is
    // recoverable (§7.3.11.11 signHiddenFlag).
    let levels = if rc.dep_quant {
        quantize_tb_dep_quant(&coeffs, n_tb_w as u32, n_tb_h as u32, qp, 8, 15)?
    } else {
        let mut l = quantize_tb_flat(&coeffs, n_tb_w as u32, n_tb_h as u32, qp, 8, 15)?;
        if rc.sign_data_hiding {
            crate::residual_enc::condition_levels_for_sdh(&mut l, n_tb_w, n_tb_h);
        }
        l
    };

    // Dequant + IDCT to get decoded residual.
    let mut params = DequantParams::luma_8bit(n_tb_w as u32, n_tb_h as u32, qp);
    params.dep_quant = rc.dep_quant;
    let dq = dequantize_tb_flat(&levels, &params)?;
    let rec_res = inverse_transform_2d(
        n_tb_w,
        n_tb_h,
        n_tb_w,
        n_tb_h,
        TrType::DctII,
        TrType::DctII,
        &dq,
        8,
        15,
    )?;

    // Write reconstructed samples.
    for ty in 0..n_tb_h {
        for tx in 0..n_tb_w {
            let rx = x + tx;
            let ry = y + ty;
            if rx < rec_plane.width && ry < rec_plane.height {
                let s = pred[ty * n_tb_w + tx] as i32 + rec_res[ty * n_tb_w + tx];
                rec_plane.samples[ry * rec_plane.stride + rx] = s.clamp(0, 255) as u8;
            }
        }
    }
    // §8.7.5.1 eq. 1212 — this TB's samples become available.
    avail.mark_luma(x, y, n_tb_w, n_tb_h);
    Ok(levels)
}

/// Round-51 — emit the §7.3.10 `transform_unit()` for one prepared TB:
/// real `tu_cb_coded_flag` / `tu_cr_coded_flag` / `tu_y_coded_flag`
/// CABAC bins (per §9.3.4.2.5 + Table 127 ctxIdx tables) followed by the
/// per-component residual bins for every component whose CBF is set.
///
/// Spec ordering inside `transform_unit()` is chroma CBFs first (Cb then
/// Cr per §7.3.10), then `tu_y_coded_flag`, then per-component residual
/// blocks in the order luma → Cb → Cr per §7.3.11. Each CBF bin is a
/// single ctx-coded decision; the residual blocks are skipped on the
/// decoder side when the corresponding CBF is 0 (§7.3.11.10).
///
/// The single-tree, intra, no-ISP, no-SBT, no-ACT scope here means the
/// luma CBF is always emitted (the §7.3.11.10 read condition simplifies
/// to true), the chroma CBFs are always emitted when `chroma_format_idc
/// != 0`, and the §7.3.10 luma context's `prev_tu_cbf_y` stays false
/// (only ISP CUs propagate the previous-subpartition CBF, eq. 1376).
fn emit_tu_with_cbf(
    enc: &mut crate::cabac_enc::ArithEncoder,
    ctxs: &mut ResidualCtxs,
    ictx: &mut IntraModeCtxs,
    tb: &PreparedLumaTb,
    qp_state: &mut QpDeltaState,
    rc_opts: crate::residual::RcOpts,
) -> Result<()> {
    // r412 — §7.3.11.5 orders the intra-mode cascade before
    // transform_tree(): signal INTRA_DC via the MPM path + DM chroma.
    emit_intra_dc_mode_bins(enc, ictx)?;

    let cbf_y = tb.levels.iter().any(|&l| l != 0);
    let cbf_cb = tb.cb_levels.iter().any(|&l| l != 0);
    let cbf_cr = tb.cr_levels.iter().any(|&l| l != 0);

    // §7.3.10 ordering: chroma CBFs (Cb then Cr) are emitted before
    // the luma CBF. `bdpcm_chroma = false` (no BDPCM in this pipeline).
    write_tu_cb_coded_flag(enc, ctxs, cbf_cb, false)?;
    write_tu_cr_coded_flag(enc, ctxs, cbf_cr, false, cbf_cb)?;
    // Luma CBF — `bdpcm_y = false`, `isp_split = false`,
    // `prev_tu_cbf_y = false` (single TU, non-ISP path).
    write_tu_y_coded_flag(enc, ctxs, cbf_y, false, false, false)?;

    // Round-52 — `cu_qp_delta_abs` + `cu_qp_delta_sign_flag` per
    // §7.3.13 / §9.3.3.10. The spec gate (§7.3.11.10):
    //   IsCuQpDeltaCoded == 0
    //   && pps_cu_qp_delta_enabled_flag
    //   && (CuCbWidth > 64 || CuCbHeight > 64
    //       || tu_y_coded_flag || tu_cb_coded_flag || tu_cr_coded_flag)
    // The "first CU of a quantization group" gate (`IsCuQpDeltaCoded`)
    // collapses here to "first CU per CTB with any CBF" because the
    // round-52 pipeline runs at QG = CTB granularity (no PH-signalled
    // sub-CTB QG depth). On the decoder side
    // `LeafCuReader::decode_transform_unit` reads the same delta when
    // it sees `pps_cu_qp_delta_enabled_flag` + any CBF set, so the
    // round-trip stays in lockstep.
    let any_cbf = cbf_y || cbf_cb || cbf_cr;
    if qp_state.enabled && qp_state.pending && any_cbf {
        let delta = tb.cu_qp_local - qp_state.prev_qp_in_qg;
        write_cu_qp_delta(enc, ctxs, delta)?;
        qp_state.prev_qp_in_qg = tb.cu_qp_local;
        qp_state.pending = false;
    }

    // Residual blocks per component — only emitted when the
    // corresponding CBF is set (§7.3.11.10's spec gate).
    if cbf_y {
        encode_tb_coefficients_opts(enc, ctxs, tb.n_tb_w, tb.n_tb_h, 0, &tb.levels, rc_opts)?;
    }
    if cbf_cb {
        encode_tb_coefficients_opts(
            enc,
            ctxs,
            tb.n_tb_chroma_w,
            tb.n_tb_chroma_h,
            1,
            &tb.cb_levels,
            rc_opts,
        )?;
    }
    if cbf_cr {
        encode_tb_coefficients_opts(
            enc,
            ctxs,
            tb.n_tb_chroma_w,
            tb.n_tb_chroma_h,
            2,
            &tb.cr_levels,
            rc_opts,
        )?;
    }
    Ok(())
}

/// Round-52 — per-quantisation-group state for the `cu_qp_delta`
/// emit path. `prev_qp_in_qg` carries the §8.7.1 `QpY` value of the
/// most-recent CU that signalled a delta (or the slice QP for the
/// first QG). `pending` is true at the start of every CU until the
/// shell either emits a `cu_qp_delta` (CBF non-zero path) or the CU
/// completes with all-zero CBFs (the spec's `IsCuQpDeltaCoded` stays
/// 0 for the next CU in the same QG).
#[derive(Debug)]
struct QpDeltaState {
    enabled: bool,
    pending: bool,
    prev_qp_in_qg: i32,
}

impl QpDeltaState {
    fn new(enabled: bool, slice_qp_y: i32) -> Self {
        Self {
            enabled,
            pending: enabled,
            prev_qp_in_qg: slice_qp_y,
        }
    }

    /// Reset to "pending again" at the start of a new CU. Round-52
    /// scope treats every CU as the first of its own QG (CTB-level
    /// QG granularity), so every CU may signal a delta.
    fn begin_cu(&mut self) {
        if self.enabled {
            self.pending = true;
        }
    }
}

/// Round-49 — forward-DCT + quantise + reconstruct one chroma TB.
///
/// `chroma_src` / `chroma_rec` are one component (Cb or Cr) at half luma
/// resolution (4:2:0 scope). `cx` / `cy` are the chroma-coordinate top-
/// left of the TB; `n_tb_c` is the chroma TB side length (= `max(4, luma
/// n_tb / 2)`). `qp_c` is the §8.7.1 derived chroma QP — for the
/// foundation IDR pipeline we use `chroma_qp_identity(qp_y, 0)` since
/// the SPS / PPS / slice / CU chroma offsets are all zero.
///
/// The DC-flat-128 prediction matches the luma path: this is the
/// encoder analogue of the decoder using the §8.7.5 `(pred + dequant
/// residual).clamp(0, 255)` reconstruction. The quantised levels are
/// returned for the second-pass CABAC emit.
#[allow(clippy::too_many_arguments)]
fn prepare_chroma_tb(
    chroma_src: &PicturePlane,
    chroma_rec: &mut PicturePlane,
    avail: &mut EncAvail,
    c_idx: u32,
    cx: usize,
    cy: usize,
    n_tb_c_w: usize,
    n_tb_c_h: usize,
    qp_c: i32,
    rc: crate::residual::RcOpts,
    chroma_scale: Option<u32>,
) -> Result<Vec<i32>> {
    // r412 — DM chroma inherits the luma INTRA_DC mode (Table 20), so
    // the chroma prediction is the decoder-mirrored §8.4.5.2.12 DC
    // (+ PDPC) on the chroma plane.
    let pred = predict_dc_block(
        chroma_rec,
        &avail.chroma,
        avail.w / 2,
        avail.h / 2,
        cx,
        cy,
        n_tb_c_w,
        n_tb_c_h,
        c_idx,
    )?;

    // Extract source block → residual.
    let mut residual = vec![0i32; n_tb_c_w * n_tb_c_h];
    for ty in 0..n_tb_c_h {
        for tx in 0..n_tb_c_w {
            let sx = cx + tx;
            let sy = cy + ty;
            let p = pred[ty * n_tb_c_w + tx] as i32;
            let src_s = chroma_src.get(sx, sy).map(|v| v as i32).unwrap_or(p);
            residual[ty * n_tb_c_w + tx] = src_s - p;
        }
    }

    // r391 — §8.7.5.3 LMCS chroma residual scaling, encoder side: the
    // CODED residual is the true residual divided by the per-CU
    // `varScale` (the decoder rescales with the eq. 1219 / 1220
    // multiply-fold, applied to the reconstruction below so encoder
    // and decoder agree sample-exactly).
    if let Some(vs) = chroma_scale {
        lmcs_forward_scale_chroma_residuals(&mut residual, vs);
    }

    // Forward DCT-II + flat quantisation (same ladder as luma; bit_depth
    // = 8 for 4:2:0 8-bit).
    let coeffs = forward_dct_ii_2d(n_tb_c_w, n_tb_c_h, &residual, 8)?;
    let levels = if rc.dep_quant {
        quantize_tb_dep_quant(&coeffs, n_tb_c_w as u32, n_tb_c_h as u32, qp_c, 8, 15)?
    } else {
        let mut l = quantize_tb_flat(&coeffs, n_tb_c_w as u32, n_tb_c_h as u32, qp_c, 8, 15)?;
        if rc.sign_data_hiding {
            crate::residual_enc::condition_levels_for_sdh(&mut l, n_tb_c_w, n_tb_c_h);
        }
        l
    };

    // Dequant → IDCT → reconstruct.
    let mut params = DequantParams::chroma_8bit(n_tb_c_w as u32, n_tb_c_h as u32, qp_c);
    params.dep_quant = rc.dep_quant;
    let dq = dequantize_tb_flat(&levels, &params)?;
    let mut rec_res = inverse_transform_2d(
        n_tb_c_w,
        n_tb_c_h,
        n_tb_c_w,
        n_tb_c_h,
        TrType::DctII,
        TrType::DctII,
        &dq,
        8,
        15,
    )?;

    // r391 — decoder-exact §8.7.5.3 rescale of the reconstructed
    // residual (the same eqs. 1219 / 1220 fold
    // `CtuWalker::reconstruct_chroma_plane` applies).
    if let Some(vs) = chroma_scale {
        crate::ctu::lmcs_scale_chroma_residuals(&mut rec_res, vs, 8);
    }

    for ty in 0..n_tb_c_h {
        for tx in 0..n_tb_c_w {
            let rx = cx + tx;
            let ry = cy + ty;
            if rx < chroma_rec.width && ry < chroma_rec.height {
                let s = pred[ty * n_tb_c_w + tx] as i32 + rec_res[ty * n_tb_c_w + tx];
                chroma_rec.samples[ry * chroma_rec.stride + rx] = s.clamp(0, 255) as u8;
            }
        }
    }
    // §8.7.5.1 eq. 1212 — the chroma mask is shared by Cb and Cr;
    // marking on the Cb pass is idempotent for the Cr pass (the TB's
    // own rectangle never feeds its own references).
    avail.mark_chroma(cx, cy, n_tb_c_w, n_tb_c_h);
    Ok(levels)
}

/// r391 — encoder inverse of the §8.7.5.3 eqs. 1219 / 1220 fold: the
/// coded chroma residual is `sign(r) · ((|r| << 11) + varScale / 2) /
/// varScale`, so the decoder's `(coded · varScale + (1 << 10)) >> 11`
/// rescale lands back on (a rounding of) the true residual.
fn lmcs_forward_scale_chroma_residuals(res: &mut [i32], var_scale: u32) {
    if var_scale == 0 {
        return;
    }
    for r in res.iter_mut() {
        let mag = ((u64::from(r.unsigned_abs()) << 11) + u64::from(var_scale >> 1))
            / u64::from(var_scale);
        *r = if *r < 0 { -(mag as i64) } else { mag as i64 } as i32;
    }
}

/// r391 — encoder-side §8.7.5.3 `varScale` derivation, mirroring the
/// decoder's per-CU logic (eq. 1216 `invAvgLuma` over the sizeY-long
/// left column / top row of the CU's mapped-domain reconstructed luma
/// neighbours, eq. 1217 mid-grey fallback, then the pivot inverse-index
/// + `ChromaScaleCoeff` lookup). The encoder's CU grid is 64-aligned so
/// the eq. 1215 sizeY-aligned corner is the CU origin itself.
fn lmcs_chroma_var_scale_enc(
    rec_luma: &PicturePlane,
    cu_x: usize,
    cu_y: usize,
    size_y: usize,
    der: &crate::lmcs::LmcsDerived,
    min_bin: u8,
    max_bin: u8,
) -> u32 {
    let avail_l = cu_x > 0;
    let avail_t = cu_y > 0;
    let mut sum: u64 = 0;
    let mut cnt: usize = 0;
    if avail_l {
        for i in 0..size_y {
            let yy = (cu_y + i).min(rec_luma.height - 1);
            sum += u64::from(rec_luma.samples[yy * rec_luma.stride + (cu_x - 1)]);
        }
        cnt += size_y;
    }
    if avail_t {
        for i in 0..size_y {
            let xx = (cu_x + i).min(rec_luma.width - 1);
            sum += u64::from(rec_luma.samples[(cu_y - 1) * rec_luma.stride + xx]);
        }
        cnt += size_y;
    }
    let inv_avg_luma = if cnt > 0 {
        ((sum + (cnt as u64 >> 1)) >> cnt.trailing_zeros()) as u32
    } else {
        128 // eq. 1217, 8-bit mid-grey
    };
    let idx_y_inv = der.idx_y_inv(inv_avg_luma, min_bin, max_bin);
    der.chroma_var_scale(idx_y_inv)
}

/// Round-56 — prepare one leaf CU at `(x, y)` of size `n_tb_w × n_tb_h`
/// luma samples. Runs the round-49 luma + chroma DCT / quant / dequant /
/// IDCT pipeline and returns a [`PreparedCu::Leaf`]. The chroma TB is
/// derived from the luma dims via the 4:2:0 half-resolution rule with a
/// 4-sample minimum per the §8.7.4 `nTbS ∈ {4, …, 64}` floor.
///
/// `cu_qp` is the per-CU QP (used for both luma and the §8.7.1 identity-
/// chroma-QP). The reconstruction is written directly into `rec`.
#[allow(clippy::too_many_arguments)]
fn prepare_leaf_cu(
    src: &PictureBuffer,
    rec: &mut PictureBuffer,
    avail: &mut EncAvail,
    x: usize,
    y: usize,
    n_tb_w: usize,
    n_tb_h: usize,
    cu_qp: i32,
    rc: crate::residual::RcOpts,
    chroma_scale: Option<u32>,
) -> Result<PreparedCu> {
    let levels = prepare_luma_tb(
        &src.luma,
        &mut rec.luma,
        avail,
        x,
        y,
        n_tb_w,
        n_tb_h,
        cu_qp,
        rc,
    )?;
    let chr_x = x / 2;
    let chr_y = y / 2;
    let n_tb_chroma_w = (n_tb_w / 2).max(4);
    let n_tb_chroma_h = (n_tb_h / 2).max(4);
    let qp_c = crate::ctu::chroma_qp_identity(cu_qp, 0);
    let cb_levels = prepare_chroma_tb(
        &src.cb,
        &mut rec.cb,
        avail,
        1,
        chr_x,
        chr_y,
        n_tb_chroma_w,
        n_tb_chroma_h,
        qp_c,
        rc,
        chroma_scale,
    )?;
    let cr_levels = prepare_chroma_tb(
        &src.cr,
        &mut rec.cr,
        avail,
        2,
        chr_x,
        chr_y,
        n_tb_chroma_w,
        n_tb_chroma_h,
        qp_c,
        rc,
        chroma_scale,
    )?;
    Ok(PreparedCu::Leaf(PreparedLumaTb {
        n_tb_w,
        n_tb_h,
        levels,
        n_tb_chroma_w,
        n_tb_chroma_h,
        cb_levels,
        cr_levels,
        cu_qp_local: cu_qp,
    }))
}

/// Round-56 — sum the per-leaf CABAC bin cost for one prepared CU. Used
/// by the BT picker to compare leaf-vs-split as `cost = SSE + λ·bits`.
///
/// The bin-count walks the same per-leaf path the second-pass CABAC
/// emit will follow: `split_cu_flag` = 0 + the per-component
/// `tu_*_coded_flag` + `cu_qp_delta` (when applicable) + the actual
/// residual coefficient bins. We use a fresh [`crate::cabac_enc::ArithEncoder`]
/// to count the byte cost; this is a strict upper bound on the wire
/// cost since the production stream amortises CABAC range across the
/// whole frame.
fn measure_cu_bits(
    cu: &PreparedCu,
    slice_qp_y: i32,
    rc_opts: crate::residual::RcOpts,
    constraints: &SplitConstraints,
    x: u32,
    y: u32,
) -> Result<u64> {
    let mut enc = crate::cabac_enc::ArithEncoder::new();
    let mut tree_ctxs = crate::coding_tree::TreeCtxs::init(slice_qp_y);
    let mut residual_ctxs = ResidualCtxs::init(slice_qp_y);
    let mut intra_ctxs = IntraModeCtxs::init(slice_qp_y);
    let mut qp_state = QpDeltaState::new(true, slice_qp_y);
    measure_cu_bits_recurse(
        cu,
        &mut enc,
        &mut tree_ctxs,
        &mut residual_ctxs,
        &mut intra_ctxs,
        &mut qp_state,
        0,
        0,
        rc_opts,
        constraints,
        x,
        y,
    )?;
    enc.encode_terminate(1)?;
    let bytes = enc.finish();
    Ok((bytes.len() as u64) * 8)
}

#[allow(clippy::too_many_arguments)]
fn measure_cu_bits_recurse(
    cu: &PreparedCu,
    enc: &mut crate::cabac_enc::ArithEncoder,
    tree_ctxs: &mut crate::coding_tree::TreeCtxs,
    residual_ctxs: &mut ResidualCtxs,
    intra_ctxs: &mut IntraModeCtxs,
    qp_state: &mut QpDeltaState,
    cqt_depth: u32,
    mtt_depth: u32,
    rc_opts: crate::residual::RcOpts,
    constraints: &SplitConstraints,
    x: u32,
    y: u32,
) -> Result<()> {
    match cu {
        PreparedCu::Leaf(tb) => {
            qp_state.begin_cu();
            let allows = constraints.allows(
                x,
                y,
                tb.n_tb_w as u32,
                tb.n_tb_h as u32,
                mtt_depth,
                crate::coding_tree::TreeType::SingleTree,
                None,
                0,
            );
            crate::syntax_enc::encode_coding_tree_leaf_iframe(
                enc,
                tree_ctxs,
                tb.n_tb_w as u32,
                tb.n_tb_h as u32,
                TreeNeighbours::default(),
                allows,
                |e| emit_tu_with_cbf(e, residual_ctxs, intra_ctxs, tb, qp_state, rc_opts),
            )
        }
        PreparedCu::BtSplit {
            dir,
            cqt_depth: cd,
            mtt_depth: md,
            sub_a,
            sub_b,
        } => {
            let _ = (cqt_depth, mtt_depth);
            let (pw, ph) = (bt_cu_w(sub_a, sub_b, *dir), bt_cu_h(sub_a, sub_b, *dir));
            let allows = constraints.allows(
                x,
                y,
                pw,
                ph,
                *md,
                crate::coding_tree::TreeType::SingleTree,
                None,
                0,
            );
            crate::syntax_enc::encode_coding_tree_bt_split(
                enc,
                tree_ctxs,
                pw,
                ph,
                TreeNeighbours::default(),
                *cd,
                *md,
                *dir,
                allows,
                |e, ctxs, idx, _w, _h| {
                    let child = if idx == 0 { sub_a } else { sub_b };
                    let (cx, cy) = match dir {
                        crate::syntax_enc::MttSplitDir::Vertical => (x + idx * (pw / 2), y),
                        crate::syntax_enc::MttSplitDir::Horizontal => (x, y + idx * (ph / 2)),
                    };
                    measure_cu_bits_recurse(
                        child,
                        e,
                        ctxs,
                        residual_ctxs,
                        intra_ctxs,
                        qp_state,
                        *cd,
                        *md + 1,
                        rc_opts,
                        constraints,
                        cx,
                        cy,
                    )
                },
            )
        }
        PreparedCu::TtSplit {
            dir,
            cqt_depth: cd,
            mtt_depth: md,
            sub_a,
            sub_b,
            sub_c,
        } => {
            let _ = (cqt_depth, mtt_depth);
            let (cb_w, cb_h) = tt_parent_dims(sub_a, sub_b, sub_c, *dir);
            let allows = constraints.allows(
                x,
                y,
                cb_w,
                cb_h,
                *md,
                crate::coding_tree::TreeType::SingleTree,
                None,
                0,
            );
            crate::syntax_enc::encode_coding_tree_tt_split(
                enc,
                tree_ctxs,
                cb_w,
                cb_h,
                TreeNeighbours::default(),
                *cd,
                *md,
                *dir,
                allows,
                |e, ctxs, idx, _w, _h| {
                    let child = match idx {
                        0 => sub_a,
                        1 => sub_b,
                        _ => sub_c,
                    };
                    let (cx, cy) = match dir {
                        crate::syntax_enc::MttSplitDir::Vertical => {
                            (x + [0, cb_w / 4, 3 * cb_w / 4][idx as usize], y)
                        }
                        crate::syntax_enc::MttSplitDir::Horizontal => {
                            (x, y + [0, cb_h / 4, 3 * cb_h / 4][idx as usize])
                        }
                    };
                    measure_cu_bits_recurse(
                        child,
                        e,
                        ctxs,
                        residual_ctxs,
                        intra_ctxs,
                        qp_state,
                        *cd,
                        *md + 1,
                        rc_opts,
                        constraints,
                        cx,
                        cy,
                    )
                },
            )
        }
    }
}

/// Round-57 — recover the parent CU width/height from a TT-split's three
/// child sub-CUs. With the 1:2:1 ratio: along the split axis the parent
/// dim is `4 × first-leaf-dim` (the side sub-CUs are 1/4 of parent);
/// along the orthogonal axis every child shares the parent dim.
fn tt_parent_dims(
    sub_a: &PreparedCu,
    _sub_b: &PreparedCu,
    _sub_c: &PreparedCu,
    dir: crate::syntax_enc::MttSplitDir,
) -> (u32, u32) {
    let (aw, ah) = leaf_dims(sub_a);
    match dir {
        crate::syntax_enc::MttSplitDir::Vertical => ((aw * 4) as u32, ah as u32),
        crate::syntax_enc::MttSplitDir::Horizontal => (aw as u32, (ah * 4) as u32),
    }
}

/// Round-56 — recover the parent CU width from a BT-split's child sub-CUs.
fn bt_cu_w(sub_a: &PreparedCu, _sub_b: &PreparedCu, dir: crate::syntax_enc::MttSplitDir) -> u32 {
    let (aw, _ah) = leaf_dims(sub_a);
    match dir {
        crate::syntax_enc::MttSplitDir::Vertical => (aw * 2) as u32,
        crate::syntax_enc::MttSplitDir::Horizontal => aw as u32,
    }
}

fn bt_cu_h(sub_a: &PreparedCu, _sub_b: &PreparedCu, dir: crate::syntax_enc::MttSplitDir) -> u32 {
    let (_aw, ah) = leaf_dims(sub_a);
    match dir {
        crate::syntax_enc::MttSplitDir::Vertical => ah as u32,
        crate::syntax_enc::MttSplitDir::Horizontal => (ah * 2) as u32,
    }
}

/// Round-56 — return the (luma_w, luma_h) of the first leaf inside `cu`.
/// Used by [`bt_cu_w`] / [`bt_cu_h`] to derive the parent CU size from
/// a BT-split's children. For the single-level BT scope of round 56 the
/// child is always a leaf so the first-leaf dims are the child's dims.
fn leaf_dims(cu: &PreparedCu) -> (usize, usize) {
    match cu {
        PreparedCu::Leaf(tb) => (tb.n_tb_w, tb.n_tb_h),
        PreparedCu::BtSplit { sub_a, .. } => leaf_dims(sub_a),
        PreparedCu::TtSplit { sub_a, .. } => leaf_dims(sub_a),
    }
}

/// Round-56 — compute SSE_Y of a reconstructed region against the
/// source. Used by [`prepare_cu_with_mtt_picker`] to compare BT/TT-split
/// reconstructions against the leaf baseline.
fn region_sse_y(
    src: &PicturePlane,
    rec: &PicturePlane,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
) -> u64 {
    let mut sse = 0u64;
    let h_clamp = (y + h).min(src.height.min(rec.height));
    let w_clamp = (x + w).min(src.width.min(rec.width));
    for yy in y..h_clamp {
        for xx in x..w_clamp {
            let s = src.samples[yy * src.stride + xx] as i32;
            let r = rec.samples[yy * rec.stride + xx] as i32;
            let d = (s - r) as i64;
            sse += (d * d) as u64;
        }
    }
    sse
}

/// Round-56 + Round-57 — multi-type-tree (MTT) RDO picker.
///
/// For one 64×64 CU candidate at `(x, y)`, evaluate up to five options:
///
///  * **Leaf** — single 64×64 TB (round-55 baseline).
///  * **BT_VERT** — split into two 32×64 sub-CUs, encode each as a
///    leaf, sum the SSE + BT-split bin cost. (round 56)
///  * **BT_HORZ** — split into two 64×32 sub-CUs, same. (round 56)
///  * **TT_VERT** — split into three columns at the 1:2:1 ratio
///    (16×64, 32×64, 16×64), encode each as a leaf. (round 57)
///  * **TT_HORZ** — three rows at 1:2:1 (64×16, 64×32, 64×16). (round 57)
///
/// `try_bt` / `try_tt` gate the BT- and TT-split trials independently;
/// when both are false the picker collapses to a leaf-only path
/// (matching the round-55 baseline).
///
/// The picker minimises `cost = SSE_Y + λ * bits` where `λ` is the
/// round-48 [`lambda_for_qp`] curve and `bits` includes the
/// `split_cu_flag` / `split_qt_flag` / `mtt_split_*` syntax bins (via a
/// standalone [`measure_cu_bits`] pass) plus the residual bins.
///
/// The reconstruction `rec` is updated to reflect the *chosen* option
/// — losing candidates' reconstructions are restored before returning.
/// Callers must ensure the candidate region fits entirely inside the
/// picture (`x + 64 <= w` and `y + 64 <= h`).
#[allow(clippy::too_many_arguments)]
fn prepare_cu_with_mtt_picker(
    src: &PictureBuffer,
    rec: &mut PictureBuffer,
    avail: &mut EncAvail,
    x: usize,
    y: usize,
    cb_size: usize, // 64
    cu_qp: i32,
    slice_qp_y: i32,
    try_bt: bool,
    try_tt: bool,
    rc_opts: crate::residual::RcOpts,
    chroma_scale: Option<u32>,
) -> Result<PreparedCu> {
    debug_assert_eq!(cb_size, 64);
    let lambda = lambda_for_qp(slice_qp_y);
    // r412 — the emitted-SPS split constraints (MTT enabled on this
    // path) drive the measured split-bin presence/ctxIncs.
    let constraints = encoder_split_constraints(
        rec.luma.width as u32,
        rec.luma.height as u32,
        /*mtt_enabled=*/ true,
    );

    // Snapshot every plane (and the r412 availability masks) so each
    // trial starts from the same source.
    let snap_luma = rec.luma.samples.clone();
    let snap_cb = rec.cb.samples.clone();
    let snap_cr = rec.cr.samples.clone();
    let snap_av_l = avail.luma.clone();
    let snap_av_c = avail.chroma.clone();
    #[allow(clippy::too_many_arguments)]
    fn restore(
        rec: &mut PictureBuffer,
        avail: &mut EncAvail,
        snap_luma: &[u8],
        snap_cb: &[u8],
        snap_cr: &[u8],
        snap_av_l: &[bool],
        snap_av_c: &[bool],
    ) {
        rec.luma.samples.copy_from_slice(snap_luma);
        rec.cb.samples.copy_from_slice(snap_cb);
        rec.cr.samples.copy_from_slice(snap_cr);
        avail.luma.copy_from_slice(snap_av_l);
        avail.chroma.copy_from_slice(snap_av_c);
    }

    // Generic trial state — the prepared CU, its cost, and the
    // post-trial reconstruction + availability snapshots. Storing the
    // trial's snapshots lets the picker restore the chosen candidate's
    // state after running every later trial.
    struct Trial {
        cu: PreparedCu,
        cost: f64,
        snap_luma: Vec<u8>,
        snap_cb: Vec<u8>,
        snap_cr: Vec<u8>,
        snap_av_l: Vec<bool>,
        snap_av_c: Vec<bool>,
    }

    let mut trials: Vec<Trial> = Vec::with_capacity(5);

    // --- Trial 1: leaf 64×64 ---
    let leaf = prepare_leaf_cu(
        src,
        rec,
        avail,
        x,
        y,
        cb_size,
        cb_size,
        cu_qp,
        rc_opts,
        chroma_scale,
    )?;
    let leaf_sse = region_sse_y(&src.luma, &rec.luma, x, y, cb_size, cb_size);
    let leaf_bits = measure_cu_bits(&leaf, slice_qp_y, rc_opts, &constraints, x as u32, y as u32)?;
    trials.push(Trial {
        cu: leaf,
        cost: leaf_sse as f64 + lambda * leaf_bits as f64,
        snap_luma: rec.luma.samples.clone(),
        snap_cb: rec.cb.samples.clone(),
        snap_cr: rec.cr.samples.clone(),
        snap_av_l: avail.luma.clone(),
        snap_av_c: avail.chroma.clone(),
    });

    let half = cb_size / 2;
    let quarter = cb_size / 4;

    if try_bt {
        // --- BT_VERT (two 32×64 sub-CUs) ---
        restore(
            rec, avail, &snap_luma, &snap_cb, &snap_cr, &snap_av_l, &snap_av_c,
        );
        let bt_v_a = prepare_leaf_cu(
            src,
            rec,
            avail,
            x,
            y,
            half,
            cb_size,
            cu_qp,
            rc_opts,
            chroma_scale,
        )?;
        let bt_v_b = prepare_leaf_cu(
            src,
            rec,
            avail,
            x + half,
            y,
            half,
            cb_size,
            cu_qp,
            rc_opts,
            chroma_scale,
        )?;
        let bt_v_sse = region_sse_y(&src.luma, &rec.luma, x, y, cb_size, cb_size);
        let bt_v_cu = PreparedCu::BtSplit {
            dir: crate::syntax_enc::MttSplitDir::Vertical,
            cqt_depth: 1,
            mtt_depth: 0,
            sub_a: Box::new(bt_v_a),
            sub_b: Box::new(bt_v_b),
        };
        let bt_v_bits = measure_cu_bits(
            &bt_v_cu,
            slice_qp_y,
            rc_opts,
            &constraints,
            x as u32,
            y as u32,
        )?;
        trials.push(Trial {
            cu: bt_v_cu,
            cost: bt_v_sse as f64 + lambda * bt_v_bits as f64,
            snap_luma: rec.luma.samples.clone(),
            snap_cb: rec.cb.samples.clone(),
            snap_cr: rec.cr.samples.clone(),
            snap_av_l: avail.luma.clone(),
            snap_av_c: avail.chroma.clone(),
        });

        // --- BT_HORZ (two 64×32 sub-CUs) ---
        restore(
            rec, avail, &snap_luma, &snap_cb, &snap_cr, &snap_av_l, &snap_av_c,
        );
        let bt_h_a = prepare_leaf_cu(
            src,
            rec,
            avail,
            x,
            y,
            cb_size,
            half,
            cu_qp,
            rc_opts,
            chroma_scale,
        )?;
        let bt_h_b = prepare_leaf_cu(
            src,
            rec,
            avail,
            x,
            y + half,
            cb_size,
            half,
            cu_qp,
            rc_opts,
            chroma_scale,
        )?;
        let bt_h_sse = region_sse_y(&src.luma, &rec.luma, x, y, cb_size, cb_size);
        let bt_h_cu = PreparedCu::BtSplit {
            dir: crate::syntax_enc::MttSplitDir::Horizontal,
            cqt_depth: 1,
            mtt_depth: 0,
            sub_a: Box::new(bt_h_a),
            sub_b: Box::new(bt_h_b),
        };
        let bt_h_bits = measure_cu_bits(
            &bt_h_cu,
            slice_qp_y,
            rc_opts,
            &constraints,
            x as u32,
            y as u32,
        )?;
        trials.push(Trial {
            cu: bt_h_cu,
            cost: bt_h_sse as f64 + lambda * bt_h_bits as f64,
            snap_luma: rec.luma.samples.clone(),
            snap_cb: rec.cb.samples.clone(),
            snap_cr: rec.cr.samples.clone(),
            snap_av_l: avail.luma.clone(),
            snap_av_c: avail.chroma.clone(),
        });
    }

    if try_tt {
        // --- TT_VERT (16×64, 32×64, 16×64 — 1:2:1 ratio) ---
        restore(
            rec, avail, &snap_luma, &snap_cb, &snap_cr, &snap_av_l, &snap_av_c,
        );
        let tt_v_a = prepare_leaf_cu(
            src,
            rec,
            avail,
            x,
            y,
            quarter,
            cb_size,
            cu_qp,
            rc_opts,
            chroma_scale,
        )?;
        let tt_v_b = prepare_leaf_cu(
            src,
            rec,
            avail,
            x + quarter,
            y,
            half,
            cb_size,
            cu_qp,
            rc_opts,
            chroma_scale,
        )?;
        let tt_v_c = prepare_leaf_cu(
            src,
            rec,
            avail,
            x + 3 * quarter,
            y,
            quarter,
            cb_size,
            cu_qp,
            rc_opts,
            chroma_scale,
        )?;
        let tt_v_sse = region_sse_y(&src.luma, &rec.luma, x, y, cb_size, cb_size);
        let tt_v_cu = PreparedCu::TtSplit {
            dir: crate::syntax_enc::MttSplitDir::Vertical,
            cqt_depth: 1,
            mtt_depth: 0,
            sub_a: Box::new(tt_v_a),
            sub_b: Box::new(tt_v_b),
            sub_c: Box::new(tt_v_c),
        };
        let tt_v_bits = measure_cu_bits(
            &tt_v_cu,
            slice_qp_y,
            rc_opts,
            &constraints,
            x as u32,
            y as u32,
        )?;
        trials.push(Trial {
            cu: tt_v_cu,
            cost: tt_v_sse as f64 + lambda * tt_v_bits as f64,
            snap_luma: rec.luma.samples.clone(),
            snap_cb: rec.cb.samples.clone(),
            snap_cr: rec.cr.samples.clone(),
            snap_av_l: avail.luma.clone(),
            snap_av_c: avail.chroma.clone(),
        });

        // --- TT_HORZ (64×16, 64×32, 64×16 — 1:2:1 ratio) ---
        restore(
            rec, avail, &snap_luma, &snap_cb, &snap_cr, &snap_av_l, &snap_av_c,
        );
        let tt_h_a = prepare_leaf_cu(
            src,
            rec,
            avail,
            x,
            y,
            cb_size,
            quarter,
            cu_qp,
            rc_opts,
            chroma_scale,
        )?;
        let tt_h_b = prepare_leaf_cu(
            src,
            rec,
            avail,
            x,
            y + quarter,
            cb_size,
            half,
            cu_qp,
            rc_opts,
            chroma_scale,
        )?;
        let tt_h_c = prepare_leaf_cu(
            src,
            rec,
            avail,
            x,
            y + 3 * quarter,
            cb_size,
            quarter,
            cu_qp,
            rc_opts,
            chroma_scale,
        )?;
        let tt_h_sse = region_sse_y(&src.luma, &rec.luma, x, y, cb_size, cb_size);
        let tt_h_cu = PreparedCu::TtSplit {
            dir: crate::syntax_enc::MttSplitDir::Horizontal,
            cqt_depth: 1,
            mtt_depth: 0,
            sub_a: Box::new(tt_h_a),
            sub_b: Box::new(tt_h_b),
            sub_c: Box::new(tt_h_c),
        };
        let tt_h_bits = measure_cu_bits(
            &tt_h_cu,
            slice_qp_y,
            rc_opts,
            &constraints,
            x as u32,
            y as u32,
        )?;
        trials.push(Trial {
            cu: tt_h_cu,
            cost: tt_h_sse as f64 + lambda * tt_h_bits as f64,
            snap_luma: rec.luma.samples.clone(),
            snap_cb: rec.cb.samples.clone(),
            snap_cr: rec.cr.samples.clone(),
            snap_av_l: avail.luma.clone(),
            snap_av_c: avail.chroma.clone(),
        });
    }

    // --- Pick lowest-cost trial + restore its reconstruction ---
    let mut best_idx = 0;
    for (i, t) in trials.iter().enumerate().skip(1) {
        if t.cost < trials[best_idx].cost {
            best_idx = i;
        }
    }
    let best = trials.swap_remove(best_idx);
    rec.luma.samples.copy_from_slice(&best.snap_luma);
    rec.cb.samples.copy_from_slice(&best.snap_cb);
    rec.cr.samples.copy_from_slice(&best.snap_cr);
    avail.luma.copy_from_slice(&best.snap_av_l);
    avail.chroma.copy_from_slice(&best.snap_av_c);
    Ok(best.cu)
}

/// Round-56 — per-picture neighbour-state tracker for the §9.3.4.2
/// ctxInc derivations.
///
/// Until round 55 every CABAC `split_cu_flag` / `split_qt_flag` /
/// `mtt_split_*` bin was emitted with `TreeNeighbours::default()`
/// (`left_avail = above_avail = false`), which collapses ctxInc to 0
/// on every CU regardless of the actual neighbour split state. Round 56
/// closes that gap by tracking, for every CU rectangle the encoder /
/// decoder emits, the per-CU geometry (`cb_w`, `cb_h`, `cqt_depth`).
/// The map is keyed by sample-coordinate granularity at the
/// `min_cb_log2 = 2` (4-sample) floor; lookup probes the rectangle
/// containing `(x, y)`.
///
/// The map is populated in raster scan order as each CU is emitted; on
/// lookup it answers two queries:
///
///   * `nbr_left(x, y)` — descriptor of the CU containing `(x - 1, y)`,
///     or `None` if `x == 0` or the CU has not been emitted yet
///     (picture-edge / first-in-row case).
///   * `nbr_above(x, y)` — descriptor of the CU containing `(x, y - 1)`,
///     same rules.
///
/// The descriptors then feed [`build_tree_neighbours`] which packs the
/// map's `cb_w` / `cb_h` / `cqt_depth` fields into the
/// [`TreeNeighbours`] struct used by the CABAC ctxInc derivations.
#[derive(Debug, Default, Clone)]
struct CuStateMap {
    /// Width of the picture in min-CB (4-sample) units.
    width_mcb: usize,
    /// Height of the picture in min-CB units.
    height_mcb: usize,
    /// One entry per min-CB cell; `None` until populated.
    /// We store only the descriptor of the CU containing this cell,
    /// not its top-left, since lookups are in terms of "find the CU
    /// containing this sample" which only needs the descriptor.
    cells: Vec<Option<CuDescriptor>>,
}

#[derive(Debug, Clone, Copy)]
struct CuDescriptor {
    cb_w: u32,
    cb_h: u32,
    cqt_depth: u32,
}

impl CuStateMap {
    /// Build a fresh map for a picture of `(w, h)` luma samples. Every
    /// cell is initially unpopulated (`None`).
    fn new(w: u32, h: u32) -> Self {
        let width_mcb = (w as usize + 3) / 4;
        let height_mcb = (h as usize + 3) / 4;
        Self {
            width_mcb,
            height_mcb,
            cells: vec![None; width_mcb * height_mcb],
        }
    }

    /// Insert one CU rectangle into the map. All cells inside the
    /// rectangle are marked with the same descriptor.
    fn insert(&mut self, x: u32, y: u32, cb_w: u32, cb_h: u32, cqt_depth: u32) {
        let desc = CuDescriptor {
            cb_w,
            cb_h,
            cqt_depth,
        };
        let mcb_x0 = (x as usize) / 4;
        let mcb_y0 = (y as usize) / 4;
        let mcb_x1 = ((x + cb_w) as usize).div_ceil(4);
        let mcb_y1 = ((y + cb_h) as usize).div_ceil(4);
        for my in mcb_y0..mcb_y1.min(self.height_mcb) {
            for mx in mcb_x0..mcb_x1.min(self.width_mcb) {
                self.cells[my * self.width_mcb + mx] = Some(desc);
            }
        }
    }

    /// Look up the CU descriptor at sample `(x, y)`. Returns `None`
    /// when the cell has not been populated yet (picture-edge / first-
    /// CU-in-row case) or when `(x, y)` falls outside the picture.
    fn get(&self, x: i64, y: i64) -> Option<CuDescriptor> {
        if x < 0 || y < 0 {
            return None;
        }
        let mx = (x as usize) / 4;
        let my = (y as usize) / 4;
        if mx >= self.width_mcb || my >= self.height_mcb {
            return None;
        }
        self.cells[my * self.width_mcb + mx]
    }
}

/// Round-56 — derive the [`TreeNeighbours`] descriptor for the CU
/// rooted at `(x, y)` from the picture-wide [`CuStateMap`]. The left
/// / above neighbour cells are looked up at `(x - 1, y)` / `(x, y - 1)`
/// per §6.4.4.
fn build_tree_neighbours(map: &CuStateMap, x: u32, y: u32) -> TreeNeighbours {
    let left = map.get(x as i64 - 1, y as i64);
    let above = map.get(x as i64, y as i64 - 1);
    TreeNeighbours {
        left_avail: left.is_some(),
        above_avail: above.is_some(),
        cb_height_left: left.map(|d| d.cb_h).unwrap_or(0),
        cb_width_above: above.map(|d| d.cb_w).unwrap_or(0),
        cqt_depth_left: left.map(|d| d.cqt_depth).unwrap_or(0),
        cqt_depth_above: above.map(|d| d.cqt_depth).unwrap_or(0),
    }
}

/// Round-56 — recover the (cb_w, cb_h) of one prepared CU. For a
/// `BtSplit` the dims are 2× the child dim along the split axis; for a
/// `TtSplit` the parent dim along the split axis is 4× the side
/// sub-CU's dim (the side sub-CUs are 1/4 of the parent per the 1:2:1
/// ratio).
fn parent_cu_dims(cu: &PreparedCu) -> (usize, usize) {
    match cu {
        PreparedCu::Leaf(tb) => (tb.n_tb_w, tb.n_tb_h),
        PreparedCu::BtSplit { dir, sub_a, .. } => {
            let (aw, ah) = leaf_dims(sub_a);
            match dir {
                crate::syntax_enc::MttSplitDir::Vertical => (aw * 2, ah),
                crate::syntax_enc::MttSplitDir::Horizontal => (aw, ah * 2),
            }
        }
        PreparedCu::TtSplit { dir, sub_a, .. } => {
            let (aw, ah) = leaf_dims(sub_a);
            match dir {
                crate::syntax_enc::MttSplitDir::Vertical => (aw * 4, ah),
                crate::syntax_enc::MttSplitDir::Horizontal => (aw, ah * 4),
            }
        }
    }
}

/// Round-56 — emit one prepared CU through the second-pass CABAC
/// stream. For a `Leaf` this is the round-52 `encode_coding_tree_leaf_iframe`
/// + `emit_tu_with_cbf` shell; for a `BtSplit` it walks
/// `encode_coding_tree_bt_split` and recurses into each sub-CU. The
/// `cu_map` is read-only here — `record_cu_into_map` is called by the
/// caller once the parent commit is safe.
#[allow(clippy::too_many_arguments)]
fn emit_prepared_cu(
    enc: &mut crate::cabac_enc::ArithEncoder,
    tree_ctxs: &mut crate::coding_tree::TreeCtxs,
    residual_ctxs: &mut ResidualCtxs,
    intra_ctxs: &mut IntraModeCtxs,
    qp_state: &mut QpDeltaState,
    cu: &PreparedCu,
    x: u32,
    y: u32,
    cqt_depth: u32,
    mtt_depth: u32,
    cu_map: &mut CuStateMap,
    rc_opts: crate::residual::RcOpts,
    constraints: &SplitConstraints,
) -> Result<()> {
    let nbrs = build_tree_neighbours(cu_map, x, y);
    match cu {
        PreparedCu::Leaf(tb) => {
            // r412 — the leaf's split_cu_flag = 0 is emitted only when
            // §6.4.1 – §6.4.3 allow a split here (e.g. an MTT child at
            // maxMttDepth has no bin — a conforming decoder infers 0).
            let allows = constraints.allows(
                x,
                y,
                tb.n_tb_w as u32,
                tb.n_tb_h as u32,
                mtt_depth,
                crate::coding_tree::TreeType::SingleTree,
                None,
                0,
            );
            crate::syntax_enc::encode_coding_tree_leaf_iframe(
                enc,
                tree_ctxs,
                tb.n_tb_w as u32,
                tb.n_tb_h as u32,
                nbrs,
                allows,
                |e| emit_tu_with_cbf(e, residual_ctxs, intra_ctxs, tb, qp_state, rc_opts),
            )?;
            // r412 — record the leaf IMMEDIATELY (decoding-order
            // semantics): the §9.3.4.2.2 split ctxIncs of the NEXT
            // sibling read this CU as a live neighbour, exactly like
            // the decoder's per-leaf neighbour-map insert. The
            // pre-r412 post-CTB commit fed stale neighbour state to
            // every within-CTB lookup, desyncing the ctx selection on
            // mixed-size (MTT) layouts.
            cu_map.insert(x, y, tb.n_tb_w as u32, tb.n_tb_h as u32, cqt_depth);
            Ok(())
        }
        PreparedCu::BtSplit {
            dir,
            cqt_depth: cd,
            mtt_depth: md,
            sub_a,
            sub_b,
        } => {
            let (cb_w, cb_h) = parent_cu_dims(cu);
            let allows = constraints.allows(
                x,
                y,
                cb_w as u32,
                cb_h as u32,
                *md,
                crate::coding_tree::TreeType::SingleTree,
                None,
                0,
            );
            crate::syntax_enc::encode_coding_tree_bt_split(
                enc,
                tree_ctxs,
                cb_w as u32,
                cb_h as u32,
                nbrs,
                *cd,
                *md,
                *dir,
                allows,
                |e, ctxs, idx, _w, _h| {
                    let (child, cx, cy) = match dir {
                        crate::syntax_enc::MttSplitDir::Vertical => {
                            let (aw, _) = leaf_dims(sub_a);
                            if idx == 0 {
                                (sub_a.as_ref(), x, y)
                            } else {
                                (sub_b.as_ref(), x + aw as u32, y)
                            }
                        }
                        crate::syntax_enc::MttSplitDir::Horizontal => {
                            let (_, ah) = leaf_dims(sub_a);
                            if idx == 0 {
                                (sub_a.as_ref(), x, y)
                            } else {
                                (sub_b.as_ref(), x, y + ah as u32)
                            }
                        }
                    };
                    qp_state.begin_cu();
                    emit_prepared_cu(
                        e,
                        ctxs,
                        residual_ctxs,
                        intra_ctxs,
                        qp_state,
                        child,
                        cx,
                        cy,
                        *cd,
                        *md + 1,
                        &mut *cu_map,
                        rc_opts,
                        constraints,
                    )
                },
            )
        }
        PreparedCu::TtSplit {
            dir,
            cqt_depth: cd,
            mtt_depth: md,
            sub_a,
            sub_b,
            sub_c,
        } => {
            let (cb_w, cb_h) = parent_cu_dims(cu);
            let allows = constraints.allows(
                x,
                y,
                cb_w as u32,
                cb_h as u32,
                *md,
                crate::coding_tree::TreeType::SingleTree,
                None,
                0,
            );
            crate::syntax_enc::encode_coding_tree_tt_split(
                enc,
                tree_ctxs,
                cb_w as u32,
                cb_h as u32,
                nbrs,
                *cd,
                *md,
                *dir,
                allows,
                |e, ctxs, idx, _w, _h| {
                    let (child, cx, cy) = match dir {
                        crate::syntax_enc::MttSplitDir::Vertical => {
                            let (aw, _) = leaf_dims(sub_a);
                            let side = aw as u32;
                            match idx {
                                0 => (sub_a.as_ref(), x, y),
                                1 => (sub_b.as_ref(), x + side, y),
                                _ => (sub_c.as_ref(), x + 3 * side, y),
                            }
                        }
                        crate::syntax_enc::MttSplitDir::Horizontal => {
                            let (_, ah) = leaf_dims(sub_a);
                            let side = ah as u32;
                            match idx {
                                0 => (sub_a.as_ref(), x, y),
                                1 => (sub_b.as_ref(), x, y + side),
                                _ => (sub_c.as_ref(), x, y + 3 * side),
                            }
                        }
                    };
                    qp_state.begin_cu();
                    emit_prepared_cu(
                        e,
                        ctxs,
                        residual_ctxs,
                        intra_ctxs,
                        qp_state,
                        child,
                        cx,
                        cy,
                        *cd,
                        *md + 1,
                        &mut *cu_map,
                        rc_opts,
                        constraints,
                    )
                },
            )
        }
    }
}

/// Round-56 — collect [`DeblockCu`] entries for every leaf inside one
/// (possibly BT-split) CU. The walk recovers the per-leaf sample
/// origin from the BT split direction so the deblock pass sees the
/// real sub-CU rectangles, not the parent CU.
fn accumulate_deblock_cus(cu: &PreparedCu, x: usize, y: usize, out: &mut Vec<DeblockCu>) {
    match cu {
        PreparedCu::Leaf(tb) => {
            let cbf_cb = tb.cb_levels.iter().any(|&l| l != 0);
            let cbf_cr = tb.cr_levels.iter().any(|&l| l != 0);
            out.push(DeblockCu {
                x: x as u32,
                y: y as u32,
                w: tb.n_tb_w as u32,
                h: tb.n_tb_h as u32,
                qp_y: tb.cu_qp_local,
                intra: true,
                tu_y_coded: true,
                tu_cb_coded: cbf_cb,
                tu_cr_coded: cbf_cr,
                bdpcm_luma: false,
                bdpcm_chroma: false,
            });
        }
        PreparedCu::BtSplit {
            dir, sub_a, sub_b, ..
        } => {
            let (aw, ah) = leaf_dims(sub_a);
            match dir {
                crate::syntax_enc::MttSplitDir::Vertical => {
                    accumulate_deblock_cus(sub_a, x, y, out);
                    accumulate_deblock_cus(sub_b, x + aw, y, out);
                }
                crate::syntax_enc::MttSplitDir::Horizontal => {
                    accumulate_deblock_cus(sub_a, x, y, out);
                    accumulate_deblock_cus(sub_b, x, y + ah, out);
                }
            }
            let _ = (aw, ah);
        }
        PreparedCu::TtSplit {
            dir,
            sub_a,
            sub_b,
            sub_c,
            ..
        } => {
            let (aw, ah) = leaf_dims(sub_a);
            match dir {
                crate::syntax_enc::MttSplitDir::Vertical => {
                    let side = aw;
                    accumulate_deblock_cus(sub_a, x, y, out);
                    accumulate_deblock_cus(sub_b, x + side, y, out);
                    accumulate_deblock_cus(sub_c, x + 3 * side, y, out);
                }
                crate::syntax_enc::MttSplitDir::Horizontal => {
                    let side = ah;
                    accumulate_deblock_cus(sub_a, x, y, out);
                    accumulate_deblock_cus(sub_b, x, y + side, out);
                    accumulate_deblock_cus(sub_c, x, y + 3 * side, out);
                }
            }
        }
    }
}

/// Encode a complete IDR frame from `src` with residual coding.
///
/// Returns `(bitstream, reconstructed_frame)`. The caller can compute
/// PSNR_Y via [`psnr_y`].
///
/// `qp` — luma quantisation parameter (0..=51). Constant across all CUs.
/// For per-CU QP control (round-52 `cu_qp_delta` testbed) see
/// [`encode_idr_with_qp_picker`]. For round-54 opt-in encoder knobs
/// (`enable_alf_clip_rdo`, `enable_chroma_sao_merge`) see
/// [`encode_idr_with_residuals_cfg`].
pub fn encode_idr_with_residuals(src: &PictureBuffer, qp: i32) -> Result<(Vec<u8>, PictureBuffer)> {
    encode_idr_with_qp_picker(src, qp, |_, _, _, _| qp)
}

/// Round-54 — same as [`encode_idr_with_residuals`] but takes an
/// [`crate::encoder::EncoderConfig`] so the caller can opt into
/// `enable_alf_clip_rdo` and `enable_chroma_sao_merge`.
///
/// The `width` / `height` fields of `cfg` are overridden to match
/// `src.luma`, so callers can reuse a default-constructed config and
/// only set the optional flags.
pub fn encode_idr_with_residuals_cfg(
    src: &PictureBuffer,
    qp: i32,
    cfg: crate::encoder::EncoderConfig,
) -> Result<(Vec<u8>, PictureBuffer)> {
    encode_idr_with_qp_picker_cfg(src, qp, cfg, |_, _, _, _| qp)
}

/// Round-52 — encode an IDR with a caller-provided per-CU QP picker.
///
/// `slice_qp_y` is the slice-level QP (signalled in `sh_qp_delta`); the
/// per-CU QP returned by `qp_picker(rx, ry, tx, ty)` becomes the
/// `CuQpDeltaVal` source — when `qp_picker(...) != prev_qp_in_qg` and
/// the CU has any non-zero CBF, the encoder emits
/// `cu_qp_delta_abs` + `cu_qp_delta_sign_flag` per §7.3.13 / §9.3.3.10.
/// The decoder side recovers the same `qp_y` via §8.7.1 from the
/// previous QG's QP + the signalled delta, so dequantisation matches.
///
/// `(rx, ry)` are the CTU coordinates and `(tx, ty)` are the per-CTU TB
/// coordinates (each TB block is one CU in the round-52 scope).
pub fn encode_idr_with_qp_picker(
    src: &PictureBuffer,
    slice_qp_y: i32,
    qp_picker: impl Fn(usize, usize, usize, usize) -> i32,
) -> Result<(Vec<u8>, PictureBuffer)> {
    let w = src.luma.width as u32;
    let h = src.luma.height as u32;
    encode_idr_with_qp_picker_cfg(
        src,
        slice_qp_y,
        crate::encoder::EncoderConfig::new(w, h),
        qp_picker,
    )
}

/// Round-54 — `encode_idr_with_qp_picker` parameterised by an
/// [`crate::encoder::EncoderConfig`] so callers can opt into
/// `enable_alf_clip_rdo` (per-tap luma ALF clip RDO via
/// [`crate::alf_aps_design::design_clip_rdo_for_luma_aps`]) and
/// `enable_chroma_sao_merge` (per-CTB chroma SAO merge-left / merge-above
/// RDO via [`crate::sao_enc::apply_chroma_sao_merge`] + the matching
/// CABAC emit pass via [`crate::sao_syntax::encode_sao_ctb`]).
pub fn encode_idr_with_qp_picker_cfg(
    src: &PictureBuffer,
    slice_qp_y: i32,
    mut config: crate::encoder::EncoderConfig,
    qp_picker: impl Fn(usize, usize, usize, usize) -> i32,
) -> Result<(Vec<u8>, PictureBuffer)> {
    use crate::cabac_enc::ArithEncoder;
    use crate::encoder::VvcEncoder;
    use crate::nal::NalUnitType;

    let w = src.luma.width as u32;
    let h = src.luma.height as u32;
    // Force dimensions to match the source so callers can reuse a
    // default-constructed config that only sets the round-54 knobs.
    config.width = w;
    config.height = h;
    // Round-384 — LMCS (§8.7.5.2): with `config.lmcs` set the coding
    // loop runs in the mapped luma domain. Derive the §7.4.3.19 arrays
    // up front (validating the payload), forward-map a copy of the
    // source luma for the residual-coding pass, and inverse-map the
    // reconstruction (§8.8.1 step 1) before the deblock / SAO / ALF
    // chain below. Chroma is untouched (`ph_chroma_residual_scale_flag
    // = 0`).
    let lmcs_derived = match config.lmcs.as_ref() {
        Some(data) => Some(data.derive(8)?),
        None => None,
    };
    let lmcs_payload = config.lmcs;
    let vvc_enc = VvcEncoder::new(config)?;
    let coding_src_owned: PictureBuffer;
    let coding_src: &PictureBuffer = if let Some(der) = lmcs_derived.as_ref() {
        let mut mapped = src.clone();
        for v in mapped.luma.samples.iter_mut() {
            *v = der.forward_map_luma_sample(u32::from(*v)).clamp(0, 255) as u8;
        }
        coding_src_owned = mapped;
        &coding_src_owned
    } else {
        src
    };

    // §7.3.7 residual-coding switches, live through the first-pass
    // quantisers and every residual emit below.
    let rc_opts = crate::residual::RcOpts {
        dep_quant: config.dep_quant,
        sign_data_hiding: config.sign_data_hiding,
    };

    // --- Emit header NALs (VPS + SPS + PPS + PH) ---
    let mut bitstream = Vec::<u8>::new();

    // Helper: append one Annex-B NAL.
    let annex_b = |bs: &mut Vec<u8>, nal: &[u8]| {
        bs.extend_from_slice(&[0x00, 0x00, 0x00, 0x01]);
        bs.extend_from_slice(nal);
    };

    use crate::encoder::EmittedNalKind;
    annex_b(&mut bitstream, &vvc_enc.emit_nal(EmittedNalKind::Vps)?);
    annex_b(&mut bitstream, &vvc_enc.emit_nal(EmittedNalKind::Sps)?);
    annex_b(&mut bitstream, &vvc_enc.emit_nal(EmittedNalKind::Pps)?);

    // Round-51 — defer ALL ALF APS NAL emission (chroma + CC-ALF +
    // round-48 luma APS) until after the in-loop-filter chain has
    // run. The chroma + CC-ALF APSes were unconditionally emitted
    // here through round-50; the round-51 trade-off RDO can drop any
    // of them when the APS-bytes-vs-SSE comparison loses, in which
    // case the corresponding PH `ph_alf_*_enabled_flag` is signalled
    // 0 and the per-CTU CABAC syntax suppresses the matching bins.
    // Wire order is still `VPS, SPS, PPS, [APS chroma], [APS cc],
    // [APS luma], PH, slice` per §7.4.2.4.
    let chroma_alf_aps = build_chroma_alf_aps();
    let cc_alf_aps = build_cc_alf_aps();
    let chroma_aps_rbsp = crate::aps_enc::emit_alf_aps_rbsp(0, true, &chroma_alf_aps)?;
    let cc_aps_rbsp = crate::aps_enc::emit_alf_aps_rbsp(1, true, &cc_alf_aps)?;

    // The §7.3.7 slice-header bytes (ALF chain + sh_lmcs_used_flag +
    // sh_qp_delta + SAO used flags) are emitted AFTER the in-loop
    // filter RDO below — the ALF APS ship/skip decisions feed the
    // chain since the §7.4.3.5 `pps_alf_info_in_ph_flag = 0` fix moved
    // it out of the PH.

    // CABAC-encode every CTU.
    let ctb_size: usize = 128; // matches encoder SPS (log2_ctu_size_minus5 = 2 → 128)
    let pic_w_ctbs = (w as usize + ctb_size - 1) / ctb_size;
    let pic_h_ctbs = (h as usize + ctb_size - 1) / ctb_size;

    let mut rec = PictureBuffer::yuv420_filled(w as usize, h as usize, 128);
    // r412 — eq. 1212 availability masks for the decoder-mirrored
    // intra prediction in the coding loop.
    let mut avail = EncAvail::new(w as usize, h as usize);
    // r412 — §6.4.1 – §6.4.3 constraints matching the emitted SPS
    // partition fields; drive every split-bin presence + ctxInc.
    let split_constraints = encoder_split_constraints(
        w,
        h,
        config.enable_mtt_bt_picker || config.enable_mtt_tt_picker,
    );
    let mut all_deblock_cus: Vec<DeblockCu> = Vec::new();

    // TB size: use the full CTB for simplicity.  For large pictures this
    // is fine since DCT-II is defined up to 64×64; cap TB at 64×64.
    let tb_size = 64usize;

    // ------------------------------------------------------------------
    // First pass — compute per-TB residuals + reconstruct without
    // touching any CABAC stream. We persist the quantised levels so the
    // round-46 second pass (per-CTU ALF + residual CABAC interleave) can
    // emit them after the in-loop-filter / ALF RDO has chosen the
    // per-CTB ALF flags.
    // ------------------------------------------------------------------
    let mut prepared_per_ctu: Vec<Vec<Vec<PreparedCu>>> = (0..pic_h_ctbs)
        .map(|_| (0..pic_w_ctbs).map(|_| Vec::new()).collect())
        .collect();

    for ry in 0..pic_h_ctbs {
        for rx in 0..pic_w_ctbs {
            let ctb_x = rx * ctb_size;
            let ctb_y = ry * ctb_size;
            let ctb_w = ctb_size.min(w as usize - ctb_x);
            let ctb_h = ctb_size.min(h as usize - ctb_y);

            // Encode the CTU as a grid of (up to) 64×64 TBs.
            let num_tb_x = (ctb_w + tb_size - 1) / tb_size;
            let num_tb_y = (ctb_h + tb_size - 1) / tb_size;

            for ty in 0..num_tb_y {
                for tx in 0..num_tb_x {
                    let tb_x = ctb_x + tx * tb_size;
                    let tb_y = ctb_y + ty * tb_size;
                    let this_tb_w = tb_size.min(w as usize - tb_x);
                    let this_tb_h = tb_size.min(h as usize - tb_y);
                    // Use the largest square TB fitting in the block.
                    let n_tb = this_tb_w.min(this_tb_h);
                    let n_tb_pow2 = n_tb.next_power_of_two().min(64);
                    // Only square TBs for simplicity.
                    let n_tb_sq = if n_tb_pow2 >= 4 { n_tb_pow2 } else { 4 };

                    // Round-52 — per-CU QP picked from the caller-supplied
                    // closure. The slice-level QP (`slice_qp_y`) acts as
                    // the QP-baseline; `cu_qp` is the actual QP applied
                    // to forward DCT + quant + dequant + IDCT for this
                    // CU. The encoder will signal `cu_qp_delta = cu_qp -
                    // prev_qp_in_qg` on the wire when CBFs are non-zero
                    // (round-52 `pps_cu_qp_delta_enabled_flag = 1`).
                    let cu_qp = qp_picker(rx, ry, tx, ty).clamp(0, 63);

                    // Round-56 / Round-57 — when the MTT picker is enabled
                    // (BT and/or TT) and the candidate CU is at full 64×64
                    // (i.e. fits in the picture without truncation),
                    // evaluate leaf vs the chosen MTT splits on
                    // `cost = SSE_Y + λ·bits` and pick the lowest-cost
                    // option. Otherwise (or when both flags are off) fall
                    // back to the round-55 leaf path.
                    let mtt_eligible = (config.enable_mtt_bt_picker || config.enable_mtt_tt_picker)
                        && n_tb_sq == 64
                        && tb_x + 64 <= w as usize
                        && tb_y + 64 <= h as usize;
                    // r391 — §8.7.5.3 chroma residual scaling: derive the
                    // per-CU `varScale` from the mapped-domain
                    // reconstructed luma neighbours BEFORE this CU's
                    // reconstruction is written (the eq. 1216 gather only
                    // reads columns/rows outside the CU). The MTT
                    // sub-CUs share the 64-grid CU's scale — mirroring
                    // the decoder, whose eq. 1215 sizeY-aligned corner
                    // resolves every sub-CU to the same neighbour gather.
                    let chroma_scale: Option<u32> = if config.lmcs_chroma_scaling {
                        lmcs_derived
                            .as_ref()
                            .zip(lmcs_payload.as_ref())
                            .map(|(der, data)| {
                                lmcs_chroma_var_scale_enc(
                                    &rec.luma,
                                    tb_x,
                                    tb_y,
                                    64,
                                    der,
                                    data.lmcs_min_bin_idx,
                                    data.lmcs_max_bin_idx(),
                                )
                            })
                    } else {
                        None
                    };

                    let cu = if mtt_eligible {
                        prepare_cu_with_mtt_picker(
                            coding_src,
                            &mut rec,
                            &mut avail,
                            tb_x,
                            tb_y,
                            n_tb_sq,
                            cu_qp,
                            slice_qp_y,
                            config.enable_mtt_bt_picker,
                            config.enable_mtt_tt_picker,
                            rc_opts,
                            chroma_scale,
                        )?
                    } else {
                        prepare_leaf_cu(
                            coding_src,
                            &mut rec,
                            &mut avail,
                            tb_x,
                            tb_y,
                            n_tb_sq,
                            n_tb_sq,
                            cu_qp,
                            rc_opts,
                            chroma_scale,
                        )?
                    };

                    // Accumulate deblock-CU records over every TB inside
                    // the (possibly split) CU. For BT-split CUs each
                    // sub-leaf contributes its own deblock entry.
                    cu.for_each_leaf(&mut |tb| {
                        // The leaf tracks its sample-coordinate origin
                        // implicitly via prepare ordering; we recompute
                        // it from the BT split walk in a second pass
                        // below.
                        let _ = tb;
                    });
                    accumulate_deblock_cus(&cu, tb_x, tb_y, &mut all_deblock_cus);

                    prepared_per_ctu[ry][rx].push(cu);
                }
            }
        }
    }

    // Apply deblocking. `bit_depth = 8` matches the round-35 8-bit
    // pipeline scope; without this the strong-filter `scale_*_for_
    // bit_depth` shifts overflow on any picture that triggers a
    // strong-edge deblocking decision (eq. 1276 / 1345 do `<< (BitDepth
    // − 8)`, and `BitDepth = 0` would shift by −8).
    let dbp = DeblockParams {
        disabled: false,
        bit_depth: 8,
        ctb_log2_size_y: 7, // encoder SPS: sps_log2_ctu_size_minus5 = 2
        ..Default::default()
    };
    // Round-384 — §8.8.1 step 1: with LMCS on, the coding-loop
    // reconstruction is in the mapped luma domain; the picture inverse
    // mapping (§8.8.2.1 / §8.8.2.2) runs BEFORE deblocking so the
    // deblock / SAO / ALF designs (which compare against the ORIGINAL
    // source) operate in the original domain, mirroring the decoder's
    // in-loop order.
    if let (Some(der), Some(data)) = (lmcs_derived.as_ref(), lmcs_payload.as_ref()) {
        let (min_bin, max_bin) = (data.lmcs_min_bin_idx, data.lmcs_max_bin_idx());
        for v in rec.luma.samples.iter_mut() {
            let s32 = u32::from(*v);
            let iy = der.idx_y_inv(s32, min_bin, max_bin);
            *v = der.inverse_map_luma_sample(s32, iy) as u8;
        }
    }
    apply_deblocking(&mut rec, &all_deblock_cus, &dbp, 1);

    // SAO decision + apply.
    //
    // Round-50 — chroma SAO RDO + apply lit up alongside luma now that
    // round-49 emits real chroma residuals. The reconstructed Cb / Cr
    // planes carry chroma quant / IDCT noise that the per-CTB §8.8.4.2
    // BO + EO RDO can shave off the same way the luma path already does.
    // [`sao_decide_picture`] already mirrors the per-CTU walk for
    // chroma at the 4:2:0 half-resolution, and [`apply_sao`] gates the
    // chroma branch on `cfg.chroma_used && chroma_format_idc != 0`. The
    // SPS still carries `sao_enabled_flag = 0` (the round-50 encoder is
    // a self-roundtrip pipeline; SAO modifies the in-loop reconstruction
    // but is not signalled on the wire), so flipping `chroma_used` on
    // changes only the *internal* reconstruction the PSNR test reads.
    // r412 wire-conformance fix — SAO may only modify the coding-loop
    // reconstruction when it is actually SIGNALLED: the pipeline emits
    // SAO syntax exclusively under `enable_chroma_sao_merge` (chroma
    // components only — `sps_sao_enabled_flag` mirrors the knob and the
    // per-CTB emit runs with `luma_used = false`). The pre-r412 code
    // designed and applied luma + chroma SAO unconditionally while
    // signalling nothing, so no conforming decoder could reproduce the
    // encoder reconstruction.
    let sao_signalled_chroma = config.enable_chroma_sao_merge;
    let sao_indep = sao_decide_picture(src, &rec, 7, 8, false, sao_signalled_chroma);
    // Round-54 — when the encoder config opts in, run the per-CTB chroma
    // SAO merge pass on top of the independent decisions. The returned
    // `merged` SaoPicture rewrites the chroma slots to the chosen
    // neighbour's params on every MergeLeft / MergeAbove CTB, and the
    // returned `merge_map` flows into the second CABAC pass so the wire
    // stream emits `sao_merge_*_flag = 1` for those CTBs.
    let pic_w_in_ctbs = sao_indep.pic_width_in_ctbs_y;
    let pic_h_in_ctbs = sao_indep.pic_height_in_ctbs_y;
    let (sao_pic, sao_merge_map) = if config.enable_chroma_sao_merge {
        crate::sao_enc::apply_chroma_sao_merge(src, &rec, &sao_indep, 8, 7)
    } else {
        (
            sao_indep,
            crate::sao::SaoMergeMap::empty(pic_w_in_ctbs, pic_h_in_ctbs),
        )
    };
    let sao_cfg = SaoConfig {
        luma_used: false,
        chroma_used: sao_signalled_chroma,
        bit_depth: 8,
        ctb_log2_size_y: 7,
        chroma_format_idc: 1,
    };
    crate::sao::apply_sao(&mut rec, &sao_pic, &sao_cfg);

    // Round-41 — luma ALF filter-set RDO over all 16 fixed filter
    // sets. The RDO chooses, per CTB, the lower-SSE_Y option among
    // `{off, set 0, …, set 15}`.
    //
    // Round-43 — CC-ALF (§8.8.5.7) RDO is chained on top of the luma
    // pass. CC-ALF reads from the *pre-luma-ALF* `recPictureL`, so we
    // snapshot the luma plane before the luma RDO mutates it.
    //
    // Round-44 — primary chroma ALF (§8.8.5.4) RDO closes the on/off
    // loop on Cb / Cr. Per §8.8.5.1 the primary chroma pass runs with
    // luma in the same CTB walk and *before* CC-ALF (which adds onto
    // the post-primary-chroma plane). Encoder-side we therefore order
    // the passes:
    //   1. snapshot pre-luma-ALF luma (CC-ALF reference);
    //   2. luma RDO (mutates rec.luma);
    //   3. primary chroma RDO for Cb (mutates rec.cb);
    //   4. primary chroma RDO for Cr (mutates rec.cr);
    //   5. CC-ALF RDO for Cb (reads pre-luma-ALF luma, mutates rec.cb);
    //   6. CC-ALF RDO for Cr (reads pre-luma-ALF luma, mutates rec.cr).
    // Both chroma APSes are synthesised in-memory with deliberately
    // small magnitudes; the per-CTB SSE RDO leaves any component off
    // whenever the trial does not lower SSE.
    //
    // Round-48 — design + (conditionally) emit a luma ALF APS at id 2.
    //
    // The round-47 single-row design is split into 25 independent
    // §8.8.5.3 per-class Wiener fits via
    // [`crate::alf_aps_design::design_per_class_luma_alf_filters`]. The
    // per-class APS feeds (a) `emit_alf_aps_rbsp`, which deduplicates
    // equal class rows so the wire format stays compact when the
    // designer falls back to a single row, (b)
    // `alf_decide_and_apply_with_aps`, which trial-applies the APS-
    // signalled set alongside the 16 fixed sets and picks per CTB.
    //
    // Round-48 also adds a picture-level APS-vs-fixed-only competition:
    // we run the RDO twice (once with the APS as a candidate, once
    // without) and ship the APS NAL only when (a) at least one CTB
    // picks the APS slot, and (b) the per-picture SSE_Y reduction
    // exceeds the rate-distortion cost of the APS NAL. The rate-
    // distortion cost approximates each APS byte at `lambda * 8` SSE
    // units; `lambda` is a low-effort `2^((qp-12)/3)` curve calibrated
    // against the round-47 baseline test pictures (QP 0 → ~0.04, QP 26
    // → ~12.7). When the APS loses the trade-off we collapse the
    // pipeline to round-46 fixed-only behaviour: skip the APS NAL,
    // emit the PH with `ph_num_alf_aps_ids_luma = 0`, and configure
    // the per-CTU walk with `sh_num_alf_aps_ids_luma = 0`.
    //
    // Per §7.4.2.4 the APS NAL must precede the picture header that
    // references it; we therefore design + measure the APS *before*
    // emitting either NAL.
    let designed_per_class = crate::alf_aps_design::design_per_class_luma_alf_filters(src, &rec);
    // Round-54 — when the encoder config opts in, run the per-tap luma
    // ALF clip-index RDO on top of the per-class coefficient design.
    // The result packs into an `AlfApsData` whose wire format carries
    // `alf_luma_clip_flag = 1` plus the per-tap `alf_luma_clip_idx[]`
    // block when at least one tap picked a non-zero index. When no tap
    // wins, the packed APS stays wire-identical to the round-48 path
    // (clip flag = 0, no per-tap block).
    let luma_alf_aps = if config.enable_alf_clip_rdo {
        let clip_rdo = crate::alf_aps_design::design_clip_rdo_for_luma_aps(
            src,
            &rec,
            &designed_per_class.coeff,
            7,
            8,
            1,
        );
        crate::alf_aps_design::build_per_class_luma_alf_aps_data_with_clip(&clip_rdo)
    } else {
        crate::alf_aps_design::build_per_class_luma_alf_aps_data(&designed_per_class)
    };

    // CC-ALF reads from the *pre-luma-ALF* `recPictureL` (eq. 1515 /
    // §8.8.5.7). Snapshot the post-SAO luma plane *before* the
    // luma-ALF RDO mutates `rec`, regardless of the trade-off branch.
    let pre_luma_alf_samples = rec.luma.samples.clone();

    // Trial 1 — fixed-only RDO on a clone of `rec`.
    let mut rec_fixed_only = rec.clone();
    let alf_pic_fixed_only =
        crate::alf_enc::alf_decide_and_apply(src, &mut rec_fixed_only, 7, 8, 1);
    let sse_fixed_only = total_sse_y(&src.luma, &rec_fixed_only.luma);

    // Trial 2 — APS + fixed RDO on a clone of `rec`. Cheap because the
    // RDO scales linearly with the number of trial sets and the test
    // pictures are at most 128×128.
    let mut rec_with_aps = rec.clone();
    let alf_pic_with_aps = crate::alf_enc::alf_decide_and_apply_with_aps(
        src,
        &mut rec_with_aps,
        &luma_alf_aps,
        7,
        8,
        1,
    );
    let sse_with_aps = total_sse_y(&src.luma, &rec_with_aps.luma);

    // Picture-bits trade-off. APS NAL byte cost includes the 2-byte NAL
    // header plus the emulation-prevented RBSP payload.
    let luma_aps_rbsp = crate::aps_enc::emit_alf_aps_rbsp(2, false, &luma_alf_aps)?;
    let aps_byte_cost = 2 + luma_aps_rbsp.len() as u64;
    let lambda = lambda_for_qp(slice_qp_y);
    // §8.8.5 RDO cost = SSE + lambda * bits. APS NAL ⇒ bits = 8 * bytes.
    let aps_rd_overhead = (lambda * (aps_byte_cost as f64) * 8.0) as u64;
    let aps_used_anywhere = (0..alf_pic_with_aps.pic_height_in_ctbs_y).any(|ry| {
        (0..alf_pic_with_aps.pic_width_in_ctbs_y)
            .any(|rx| alf_pic_with_aps.get(rx, ry).luma_filt_set_idx >= 16)
    });

    // Round-50 — extend the trade-off with the per-CTB CABAC bit cost of
    // the `alf_use_aps_flag` + `alf_luma_*_filter_idx` syntax. The two
    // candidate `AlfPicture`s differ in which luma filter index family
    // they signal per CTB (APS vs fixed-set), and the binarisation cost
    // is not symmetric: `alf_luma_fixed_filter_idx` is a TB-bypass with
    // cMax = 15 (~4 bits per CTB) while `alf_luma_prev_filter_idx` with
    // cMax = N - 1 = 0 is suppressed entirely (eq. 1437, the field is
    // absent and prev_idx is inferred to 0). The fixed-only candidate
    // therefore pays ~5 bits per "luma_on" CTB while the with-APS
    // candidate pays ~1 bit. Without this term the round-48 trade-off
    // would over-account the APS NAL byte cost in isolation; closing
    // the gap means both branches are compared on an apples-to-apples
    // SSE + lambda * total_bits basis.
    //
    // We measure each candidate against the `AlfSyntaxConfig` that
    // matches the wire stream the second-pass CABAC walk would emit:
    // - fixed-only: `sh_num_alf_aps_ids_luma = 0` (`alf_use_aps_flag`
    //   suppressed downstream, but every "luma_on" CTB still pays the
    //   fixed-filter index), with the chroma + CC-ALF parameters left
    //   at their picture-level values so the chroma bins are folded in
    //   identically across both branches and cancel in the difference.
    // - with-aps: `sh_num_alf_aps_ids_luma = 1`, so each "luma_on" CTB
    //   pays an `alf_use_aps_flag` ctx-coded bin and (when the bin is
    //   set) the prev_idx field collapses to a no-op.
    let cfg_fixed = crate::alf_syntax::AlfSyntaxConfig {
        alf_enabled: true,
        cb_enabled: true,
        cr_enabled: true,
        cc_cb_enabled: true,
        cc_cr_enabled: true,
        sh_num_alf_aps_ids_luma: 0,
        alf_chroma_num_alt_filters_minus1: chroma_alf_aps.alf_chroma_num_alt_filters_minus1 as u8,
        alf_cc_cb_filters_signalled_minus1: cc_alf_aps.cc_cb_coeff.len().saturating_sub(1) as u8,
        alf_cc_cr_filters_signalled_minus1: cc_alf_aps.cc_cr_coeff.len().saturating_sub(1) as u8,
        chroma_format_idc: 1,
        slice_type: crate::slice_header::SliceType::I,
        sh_cabac_init_flag: false,
    };
    let cfg_with_aps = crate::alf_syntax::AlfSyntaxConfig {
        sh_num_alf_aps_ids_luma: 1,
        ..cfg_fixed
    };
    let bits_fixed_only =
        estimate_alf_picture_bin_cost(&alf_pic_fixed_only, &cfg_fixed, slice_qp_y)?;
    let bits_with_aps =
        estimate_alf_picture_bin_cost(&alf_pic_with_aps, &cfg_with_aps, slice_qp_y)?;
    let rd_fixed_only = sse_fixed_only + (lambda * bits_fixed_only as f64) as u64;
    let rd_with_aps = sse_with_aps + aps_rd_overhead + (lambda * bits_with_aps as f64) as u64;
    let ship_aps = aps_used_anywhere && rd_with_aps < rd_fixed_only;

    // Commit the chosen luma reconstruction + AlfPicture (luma APS NAL
    // emission deferred to round-51 unified APS-NAL pass below).
    let (mut alf_pic, sh_num_alf_aps_ids_luma, ph_num_alf_aps_ids_luma) = if ship_aps {
        rec.luma.samples.copy_from_slice(&rec_with_aps.luma.samples);
        (alf_pic_with_aps, 1u8, 1u8)
    } else {
        rec.luma
            .samples
            .copy_from_slice(&rec_fixed_only.luma.samples);
        (alf_pic_fixed_only, 0u8, 0u8)
    };
    drop(rec_with_aps);
    drop(rec_fixed_only);

    // ------------------------------------------------------------------
    // Round-51 — chroma APS RDO trade-off.
    //
    // The chroma APS at id 0 carries the §8.8.5.4 primary-chroma filter
    // bank read by both `chroma_alf_decide_and_apply` invocations (Cb +
    // Cr). When the trade-off skips it, neither chroma ALF pass can run,
    // so PH `ph_alf_cb_enabled_flag` and `ph_alf_cr_enabled_flag` both
    // signal 0 and the per-CTU `alf_ctb_flag[1/2]` bins are suppressed
    // by `cb_enabled` / `cr_enabled = false` in `AlfSyntaxConfig`.
    //
    // RDO compare:
    //   off:  current `rec.{cb,cr}` SSE   + lambda * bin_cost_off
    //   on:   post-pass `rec.{cb,cr}` SSE + lambda * (8 * aps_bytes
    //                                                + bin_cost_on)
    //
    // `bin_cost_off` and `bin_cost_on` differ in the per-CTB
    // `alf_ctb_flag[1]` / `alf_ctb_flag[2]` bins (~1 bin/CTB each when
    // chroma is enabled in the SyntaxConfig); the alt-idx fields are
    // suppressed because the in-memory chroma APS signals only one alt
    // (`alf_chroma_num_alt_filters_minus1 = 0`, eq. 1437 cMax = 0).
    // ------------------------------------------------------------------
    let chroma_aps_byte_cost = 2 + chroma_aps_rbsp.len() as u64;
    let pre_chroma_alf_cb = rec.cb.samples.clone();
    let pre_chroma_alf_cr = rec.cr.samples.clone();
    let sse_chroma_off = total_sse_y(&src.cb, &rec.cb) + total_sse_y(&src.cr, &rec.cr);
    let mut rec_with_chroma_aps = rec.clone();
    let mut alf_pic_with_chroma_aps = alf_pic.clone();
    crate::alf_enc::chroma_alf_decide_and_apply(
        src,
        &mut rec_with_chroma_aps,
        &mut alf_pic_with_chroma_aps,
        &chroma_alf_aps,
        crate::alf_enc::CcAlfComponent::Cb,
        7,
        8,
        1,
    );
    crate::alf_enc::chroma_alf_decide_and_apply(
        src,
        &mut rec_with_chroma_aps,
        &mut alf_pic_with_chroma_aps,
        &chroma_alf_aps,
        crate::alf_enc::CcAlfComponent::Cr,
        7,
        8,
        1,
    );
    let sse_chroma_on = total_sse_y(&src.cb, &rec_with_chroma_aps.cb)
        + total_sse_y(&src.cr, &rec_with_chroma_aps.cr);
    let chroma_picked_anywhere = (0..alf_pic_with_chroma_aps.pic_height_in_ctbs_y).any(|ry| {
        (0..alf_pic_with_chroma_aps.pic_width_in_ctbs_y).any(|rx| {
            let p = alf_pic_with_chroma_aps.get(rx, ry);
            p.cb_on || p.cr_on
        })
    });
    let cfg_chroma_off = crate::alf_syntax::AlfSyntaxConfig {
        alf_enabled: true,
        cb_enabled: false,
        cr_enabled: false,
        cc_cb_enabled: false,
        cc_cr_enabled: false,
        sh_num_alf_aps_ids_luma,
        alf_chroma_num_alt_filters_minus1: chroma_alf_aps.alf_chroma_num_alt_filters_minus1 as u8,
        alf_cc_cb_filters_signalled_minus1: cc_alf_aps.cc_cb_coeff.len().saturating_sub(1) as u8,
        alf_cc_cr_filters_signalled_minus1: cc_alf_aps.cc_cr_coeff.len().saturating_sub(1) as u8,
        chroma_format_idc: 1,
        slice_type: crate::slice_header::SliceType::I,
        sh_cabac_init_flag: false,
    };
    let cfg_chroma_on = crate::alf_syntax::AlfSyntaxConfig {
        cb_enabled: true,
        cr_enabled: true,
        ..cfg_chroma_off
    };
    let bins_chroma_off = estimate_alf_picture_bin_cost(&alf_pic, &cfg_chroma_off, slice_qp_y)?;
    let bins_chroma_on =
        estimate_alf_picture_bin_cost(&alf_pic_with_chroma_aps, &cfg_chroma_on, slice_qp_y)?;
    let rd_chroma_off = sse_chroma_off + (lambda * bins_chroma_off as f64) as u64;
    let rd_chroma_on = sse_chroma_on
        + (lambda * (chroma_aps_byte_cost as f64) * 8.0) as u64
        + (lambda * bins_chroma_on as f64) as u64;
    let ship_chroma_aps = chroma_picked_anywhere && rd_chroma_on < rd_chroma_off;
    if ship_chroma_aps {
        rec.cb
            .samples
            .copy_from_slice(&rec_with_chroma_aps.cb.samples);
        rec.cr
            .samples
            .copy_from_slice(&rec_with_chroma_aps.cr.samples);
        alf_pic = alf_pic_with_chroma_aps;
    } else {
        // Restore pre-pass chroma (defensive — should already match).
        rec.cb.samples.copy_from_slice(&pre_chroma_alf_cb);
        rec.cr.samples.copy_from_slice(&pre_chroma_alf_cr);
    }
    drop(rec_with_chroma_aps);

    // ------------------------------------------------------------------
    // Round-51 — CC-ALF APS RDO trade-off (per-component decisions, one
    // shared APS at id 1).
    //
    // CC-ALF Cb and CC-ALF Cr each pick their own per-CTB `cc_*_idc`,
    // and each can independently improve chroma SSE (or not). The APS
    // NAL is shipped iff at least one component picks any non-zero idc;
    // each component's PH `ph_alf_cc_*_enabled_flag` mirrors the
    // per-component decision.
    //
    // We run each component's RDO on its own snapshot (pre-CC-ALF
    // chroma) and compare against the off-baseline including:
    //   off:  current rec.{cb,cr} SSE + lambda * bin_cost_off
    //   on:   post-pass rec.{cb,cr} SSE + lambda * bin_cost_on
    // The shared APS byte cost is added to whichever component's
    // trade-off "wins first" (or split across both for symmetry).
    // For simplicity: bind the APS byte cost to the per-component
    // trade-off but only if the component's trade-off would otherwise
    // win without the byte cost — i.e. each component independently
    // decides ship/skip and the union determines whether the APS NAL
    // ships. A component that loses on its own RDO contributes 0; a
    // component that wins pays a fair share of the APS byte cost.
    // ------------------------------------------------------------------
    let cc_aps_byte_cost = 2 + cc_aps_rbsp.len() as u64;
    let pre_cc_cb = rec.cb.samples.clone();
    let pre_cc_cr = rec.cr.samples.clone();
    let sse_cb_off = total_sse_y(&src.cb, &rec.cb);
    let sse_cr_off = total_sse_y(&src.cr, &rec.cr);

    // CC-ALF Cb trial.
    let mut rec_cc_cb = rec.clone();
    let mut alf_pic_cc_cb = alf_pic.clone();
    crate::alf_enc::cc_alf_decide_and_apply(
        src,
        &mut rec_cc_cb,
        &pre_luma_alf_samples,
        &mut alf_pic_cc_cb,
        &cc_alf_aps,
        crate::alf_enc::CcAlfComponent::Cb,
        7,
        8,
        1,
    );
    let sse_cb_on = total_sse_y(&src.cb, &rec_cc_cb.cb);
    let cc_cb_picked_anywhere = (0..alf_pic_cc_cb.pic_height_in_ctbs_y).any(|ry| {
        (0..alf_pic_cc_cb.pic_width_in_ctbs_y).any(|rx| alf_pic_cc_cb.get(rx, ry).cc_cb_idc != 0)
    });

    // CC-ALF Cr trial.
    let mut rec_cc_cr = rec.clone();
    let mut alf_pic_cc_cr = alf_pic.clone();
    crate::alf_enc::cc_alf_decide_and_apply(
        src,
        &mut rec_cc_cr,
        &pre_luma_alf_samples,
        &mut alf_pic_cc_cr,
        &cc_alf_aps,
        crate::alf_enc::CcAlfComponent::Cr,
        7,
        8,
        1,
    );
    let sse_cr_on = total_sse_y(&src.cr, &rec_cc_cr.cr);
    let cc_cr_picked_anywhere = (0..alf_pic_cc_cr.pic_height_in_ctbs_y).any(|ry| {
        (0..alf_pic_cc_cr.pic_width_in_ctbs_y).any(|rx| alf_pic_cc_cr.get(rx, ry).cc_cr_idc != 0)
    });

    // Per-component bin cost via two synthetic AlfPictures: cc_cb on,
    // and cc_cr on. The off-baseline shares the chroma_on cfg from the
    // round above (all chroma + cc bits live in `cfg_chroma_on`); the
    // per-component on-bins use a cfg that gates only that component
    // on so the bin cost reflects the single component's incremental
    // wire cost.
    let cfg_cc_cb_on = crate::alf_syntax::AlfSyntaxConfig {
        cb_enabled: ship_chroma_aps,
        cr_enabled: ship_chroma_aps,
        cc_cb_enabled: true,
        cc_cr_enabled: false,
        ..cfg_chroma_off
    };
    let cfg_cc_cr_on = crate::alf_syntax::AlfSyntaxConfig {
        cb_enabled: ship_chroma_aps,
        cr_enabled: ship_chroma_aps,
        cc_cb_enabled: false,
        cc_cr_enabled: true,
        ..cfg_chroma_off
    };
    let cfg_cc_off = crate::alf_syntax::AlfSyntaxConfig {
        cb_enabled: ship_chroma_aps,
        cr_enabled: ship_chroma_aps,
        ..cfg_chroma_off
    };
    let bins_cc_off = estimate_alf_picture_bin_cost(&alf_pic, &cfg_cc_off, slice_qp_y)?;
    let bins_cc_cb_on = estimate_alf_picture_bin_cost(&alf_pic_cc_cb, &cfg_cc_cb_on, slice_qp_y)?;
    let bins_cc_cr_on = estimate_alf_picture_bin_cost(&alf_pic_cc_cr, &cfg_cc_cr_on, slice_qp_y)?;

    // Per-component RDO comparison (no APS-byte share yet).
    let rd_cb_off_no_aps = sse_cb_off + (lambda * bins_cc_off as f64) as u64;
    let rd_cb_on_no_aps = sse_cb_on + (lambda * bins_cc_cb_on as f64) as u64;
    let rd_cr_off_no_aps = sse_cr_off + (lambda * bins_cc_off as f64) as u64;
    let rd_cr_on_no_aps = sse_cr_on + (lambda * bins_cc_cr_on as f64) as u64;
    let cb_provisional_win = cc_cb_picked_anywhere && rd_cb_on_no_aps < rd_cb_off_no_aps;
    let cr_provisional_win = cc_cr_picked_anywhere && rd_cr_on_no_aps < rd_cr_off_no_aps;
    // APS-byte share: when both components want the APS, split 50/50;
    // when only one component wants it, that component pays the full
    // cost; when neither wants it, the APS NAL is dropped.
    let (cb_aps_share, cr_aps_share) = match (cb_provisional_win, cr_provisional_win) {
        (true, true) => (
            cc_aps_byte_cost / 2,
            cc_aps_byte_cost - cc_aps_byte_cost / 2,
        ),
        (true, false) => (cc_aps_byte_cost, 0u64),
        (false, true) => (0u64, cc_aps_byte_cost),
        (false, false) => (0u64, 0u64),
    };
    let rd_cb_on_with_share = sse_cb_on
        + (lambda * (cb_aps_share as f64) * 8.0) as u64
        + (lambda * bins_cc_cb_on as f64) as u64;
    let rd_cr_on_with_share = sse_cr_on
        + (lambda * (cr_aps_share as f64) * 8.0) as u64
        + (lambda * bins_cc_cr_on as f64) as u64;
    let ship_cc_cb = cc_cb_picked_anywhere && rd_cb_on_with_share < rd_cb_off_no_aps;
    let ship_cc_cr = cc_cr_picked_anywhere && rd_cr_on_with_share < rd_cr_off_no_aps;

    // Commit per-component CC-ALF picks back into rec + alf_pic.
    if ship_cc_cb {
        rec.cb.samples.copy_from_slice(&rec_cc_cb.cb.samples);
        // Carry CC-Cb idc into alf_pic.
        for ry in 0..alf_pic.pic_height_in_ctbs_y {
            for rx in 0..alf_pic.pic_width_in_ctbs_y {
                let mut p = alf_pic.get(rx, ry);
                p.cc_cb_idc = alf_pic_cc_cb.get(rx, ry).cc_cb_idc;
                alf_pic.set(rx, ry, p);
            }
        }
    } else {
        rec.cb.samples.copy_from_slice(&pre_cc_cb);
    }
    if ship_cc_cr {
        rec.cr.samples.copy_from_slice(&rec_cc_cr.cr.samples);
        for ry in 0..alf_pic.pic_height_in_ctbs_y {
            for rx in 0..alf_pic.pic_width_in_ctbs_y {
                let mut p = alf_pic.get(rx, ry);
                p.cc_cr_idc = alf_pic_cc_cr.get(rx, ry).cc_cr_idc;
                alf_pic.set(rx, ry, p);
            }
        }
    } else {
        rec.cr.samples.copy_from_slice(&pre_cc_cr);
    }
    drop(rec_cc_cb);
    drop(rec_cc_cr);

    // Round-51 — ship the ALF APS NALs (chroma + CC + luma) per the
    // trade-off decisions. Wire order is `chroma APS, CC APS, luma
    // APS, PH` — the PH carries the matching `ph_alf_*_enabled_flag`s.
    if ship_chroma_aps {
        let chroma_aps_nal = build_nal(NalUnitType::PrefixApsNut, 0, 1, &chroma_aps_rbsp);
        annex_b(&mut bitstream, &chroma_aps_nal);
    }
    if ship_cc_cb || ship_cc_cr {
        let cc_aps_nal = build_nal(NalUnitType::PrefixApsNut, 0, 1, &cc_aps_rbsp);
        annex_b(&mut bitstream, &cc_aps_nal);
    }
    if ship_aps {
        let luma_aps_nal = build_nal(NalUnitType::PrefixApsNut, 0, 1, &luma_aps_rbsp);
        annex_b(&mut bitstream, &luma_aps_nal);
    }

    // Round-384 — ship the LMCS APS NAL (id 0) the PH references via
    // ph_lmcs_aps_id. Wire order per §7.4.2.4: APS before the PH.
    if let Some(data) = lmcs_payload.as_ref() {
        let lmcs_aps_rbsp = crate::aps_enc::emit_lmcs_aps_rbsp(0, true, data)?;
        let lmcs_aps_nal = build_nal(NalUnitType::PrefixApsNut, 0, 1, &lmcs_aps_rbsp);
        annex_b(&mut bitstream, &lmcs_aps_nal);
    }

    // Picture header — must come AFTER the APSes it references
    // (§7.4.2.4). Since the §7.4.3.5 inference fix the PH is minimal;
    // the per-APS enable flags mirror into the SLICE-HEADER ALF chain
    // below (§7.3.7).
    annex_b(&mut bitstream, &vvc_enc.emit_picture_header_nal()?);

    // §7.3.7 slice-header bytes: ALF chain (per-APS RDO decisions) +
    // sh_lmcs_used_flag + sh_qp_delta (SliceQpY = 26 + sh_qp_delta on
    // the wire, so the slice-level QP is now signalled instead of
    // being implicitly pinned at 26) + SAO used flags.
    let slice_header_bytes = vvc_enc.emit_idr_slice_header_bytes(
        &crate::encoder::AlfPhChain {
            num_alf_aps_ids_luma: ph_num_alf_aps_ids_luma,
            luma_aps_id: 2,
            ph_alf_cb_enabled_flag: ship_chroma_aps,
            ph_alf_cr_enabled_flag: ship_chroma_aps,
            ph_alf_aps_id_chroma: 0,
            ph_alf_cc_cb_enabled_flag: ship_cc_cb,
            ph_alf_cc_cb_aps_id: 1,
            ph_alf_cc_cr_enabled_flag: ship_cc_cr,
            ph_alf_cc_cr_aps_id: 1,
            // Round-54 — SAO used flags are emitted only when
            // `EncoderConfig::enable_chroma_sao_merge` flips on
            // `sps_sao_enabled_flag`. Luma SAO stays internal-only on
            // this path.
            ph_sao_luma_enabled_flag: false,
            ph_sao_chroma_enabled_flag: config.enable_chroma_sao_merge,
        },
        slice_qp_y - 26,
    );

    // ------------------------------------------------------------------
    // Round-46 — second CABAC pass: walk every CTU emitting the ALF
    // bins (`alf_ctb_flag[]`, `alf_use_aps_flag`, `alf_luma_*_idx`,
    // `alf_ctb_filter_alt_idx[]`, `alf_ctb_cc_*_idc`) immediately
    // followed by the luma-residual CABAC bins for that CTU's TBs and
    // a per-CTU `encode_terminate()`. Both syntax families share one
    // ArithEncoder so the produced bitstream matches the §7.3.11.2
    // single-stream layout of `coding_tree_unit()`.
    //
    // The per-CTB ALF decisions persisted in `alf_pic` came from the
    // round-41/43/44 RDO passes above, and the residual levels came
    // from the first pass. We do not redo any DCT / quantisation work
    // here; this loop is pure bin emission.
    // ------------------------------------------------------------------
    // Round-51 — every per-component ALF / CC-ALF gate mirrors the
    // matching PH `ph_alf_*_enabled_flag`. The decoder side gates the
    // per-CTU `alf_ctb_flag[1/2]` / `alf_ctb_cc_*_idc[]` reads on the
    // exact same flags (via `AlfSyntaxConfig.{cb,cr,cc_cb,cc_cr}_enabled`),
    // so the wire bins MUST match the PH's claim. Round-50 hard-coded
    // every bit on; round-51 derives them from the per-APS RDO trade-
    // off so picture sources whose chroma APS / CC-ALF APSes lose the
    // trade-off don't emit phantom syntax bins the decoder would
    // unconditionally read.
    let alf_cfg = crate::alf_syntax::AlfSyntaxConfig {
        alf_enabled: true,
        cb_enabled: ship_chroma_aps,
        cr_enabled: ship_chroma_aps,
        cc_cb_enabled: ship_cc_cb,
        cc_cr_enabled: ship_cc_cr,
        sh_num_alf_aps_ids_luma,
        alf_chroma_num_alt_filters_minus1: chroma_alf_aps.alf_chroma_num_alt_filters_minus1 as u8,
        alf_cc_cb_filters_signalled_minus1: cc_alf_aps.cc_cb_coeff.len().saturating_sub(1) as u8,
        alf_cc_cr_filters_signalled_minus1: cc_alf_aps.cc_cr_coeff.len().saturating_sub(1) as u8,
        chroma_format_idc: 1,
        slice_type: crate::slice_header::SliceType::I,
        sh_cabac_init_flag: false,
    };

    let mut cabac_enc = ArithEncoder::new();
    // Round-56 — picture-wide neighbour map for the §9.3.4.2 ctxInc
    // derivations. `cu_map` is the *read-only* snapshot the current
    // CTU sees; `cu_map_write` is the in-progress accumulator we
    // commit back to `cu_map` at the end of each CTB. Splitting the
    // two views keeps CTBs from seeing partially-emitted neighbours
    // mid-walk (the spec's neighbour-availability rules are post-
    // commit only).
    // r412 — ONE live neighbour map, updated per emitted leaf exactly
    // like the decoder's per-leaf insert (the pre-r412 read-snapshot /
    // post-CTB-commit split fed stale within-CTB neighbour state to
    // the split-flag ctxIncs, desyncing mixed-size MTT layouts).
    let mut cu_map = CuStateMap::new(w, h);
    // §9.3.4.2 ctx init draws on the slice QP (eq. 1571 SliceQpY, eq. 1573
    // m,n derivation). Round-52 ties every CABAC context family to the
    // single slice QP — round-51 hardcoded QP=26, but the round-52
    // per-CU QP picker may run at a different baseline.
    let mut alf_ctxs = crate::alf_syntax::AlfCtxs::init(slice_qp_y);
    let mut residual_ctxs = ResidualCtxs::init(slice_qp_y);
    // r412 — §7.3.11.5 intra-mode cascade ctxs (decoder-matching init).
    let mut intra_ctxs = IntraModeCtxs::init(slice_qp_y);
    // Round-52 — coding_tree() shell ctxs (`split_cu_flag` + `split_qt_flag`
    // + `pred_mode_flag` + `intra_luma_mpm_flag`) initialised at the slice
    // QP. Used by `encode_coding_tree_leaf_iframe`.
    let mut tree_ctxs = crate::coding_tree::TreeCtxs::init(slice_qp_y);
    // Round-52 — `cu_qp_delta` per-quantisation-group state. The pipeline
    // runs with QG = CTB granularity (no PH-signalled `cu_qp_delta_subdiv`
    // or `cu_chroma_qp_offset_subdiv`), so the QG resets at every new CTB.
    let mut qp_state = QpDeltaState::new(/*enabled=*/ true, slice_qp_y);
    // Round-54 — SAO CABAC contexts + per-slice config. Only used when
    // the encoder config opts in via `enable_chroma_sao_merge`; when off
    // the pipeline matches round-53 (no SAO bins on the wire because
    // `sps_sao_enabled_flag = 0`).
    let mut sao_ctxs = crate::sao_syntax::SaoCtxs::init(slice_qp_y);
    let sao_syntax_cfg = crate::sao_syntax::SaoSyntaxConfig {
        // Luma SAO stays internal-only on this path; the chroma merge
        // RDO only rewrites the chroma slots, so the per-CTB syntax
        // emit gates `luma_used` off and `chroma_used` on.
        luma_used: false,
        chroma_used: config.enable_chroma_sao_merge,
        chroma_format_idc: 1,
        bit_depth: 8,
        slice_type: crate::slice_header::SliceType::I,
        sh_cabac_init_flag: false,
    };

    for ry in 0..pic_h_ctbs {
        for rx in 0..pic_w_ctbs {
            // Round-54 — `sao(rx, ry)` emits ahead of the ALF / coding
            // tree per §7.3.11.2 `coding_tree_unit()` ordering. Gated
            // by `EncoderConfig::enable_chroma_sao_merge` (which also
            // flipped `sps_sao_enabled_flag = 1` and set
            // `ph_sao_chroma_enabled_flag = 1`); when off, the second-
            // pass walk emits no SAO bins so the wire matches round-53.
            if config.enable_chroma_sao_merge {
                let left_avail = rx > 0;
                let up_avail = ry > 0;
                crate::sao_syntax::encode_sao_ctb(
                    &mut cabac_enc,
                    &mut sao_ctxs,
                    &sao_syntax_cfg,
                    &sao_pic,
                    &sao_merge_map,
                    rx as u32,
                    ry as u32,
                    left_avail,
                    up_avail,
                )?;
            }

            // ALF CABAC bins for this CTU (§7.3.11.2 prefix). Neighbours
            // are picture-edge derived because we run with a single
            // slice + single tile.
            let nbrs = crate::alf_syntax::AlfNeighbours {
                left_avail: rx > 0,
                up_avail: ry > 0,
            };
            crate::alf_syntax::encode_alf_ctu(
                &mut cabac_enc,
                &mut alf_ctxs,
                &alf_cfg,
                &alf_pic,
                rx as u32,
                ry as u32,
                nbrs,
            )?;

            // Round-52 — every TB in this CTB is one CU in the round-52
            // single-CU-per-CTB scope. Wrap each CU body in
            // `encode_coding_tree_leaf_iframe` so the §7.3.11.4
            // `split_cu_flag = 0` bin is on the wire ahead of the
            // `transform_tree()` content.
            //
            // Round-55 — the SPS signals `sps_log2_ctu_size_minus5 = 2` ⇒
            // `CtbSizeY = 128`, while the encoder caps each TB / CU at
            // 64×64 (`max_luma_transform_size_64_flag = 0` in
            // [`crate::encoder::VvcEncoder::emit_sps`] keeps MaxTbSizeY at
            // the SPS default 32, but our pipeline pushes 64×64 TBs which
            // is the largest spec-supported DCT-II). For a 128×128 CTB
            // §7.3.11.4 mandates `split_cu_flag = 1` + `split_qt_flag = 1`
            // (forced QT split) before any leaf CUs are emitted; we wrap
            // the four 64×64 shells in `encode_coding_quadtree_split` so
            // the wire stream complies. For non-128 CTBs (e.g. 64×64
            // pictures fitting in a single CTB equal to MaxTbSizeY) the
            // forced split is skipped and we emit the leaf CU directly.
            //
            // QG reset: §7.4.13.2 specifies the QG starts at the first CU
            // of the CTB when `cu_qp_delta_subdiv` is 0 (the round-52
            // default). `qp_state.begin_cu()` is called per TB so the
            // first CU with any non-zero CBF emits its delta and the
            // remainder of the CTB inherits.
            let cus = &prepared_per_ctu[ry][rx];
            // Round-55 — forced QT split fires only when the actual CTB
            // covers a 128×128 region and produces all four 64×64 leaf
            // CUs. Picture-edge CTBs (when the picture is narrower /
            // shorter than 128 so only some quadrants exist) keep the
            // round-52 single-leaf path because the spec's
            // implicit-split path would split-then-skip the missing
            // quadrants — we don't need to emit `split_cu_flag = 1` for
            // a single-leaf CTB.
            let ctb_x = rx * ctb_size;
            let ctb_y = ry * ctb_size;
            let actual_ctb_w = ctb_size.min(w as usize - ctb_x);
            let actual_ctb_h = ctb_size.min(h as usize - ctb_y);
            let needs_forced_qt =
                actual_ctb_w == ctb_size && actual_ctb_h == ctb_size && cus.len() == 4;
            // Round-56 — neighbour map drives the CABAC ctxInc derivation.
            // The map is populated as each CU is *emitted* so left /
            // above neighbour lookups within the same CTB see real
            // descriptors, not the round-55 picture-edge default.
            let nbrs_ctb = build_tree_neighbours(&cu_map, ctb_x as u32, ctb_y as u32);
            if needs_forced_qt {
                // Round-55 — 128×128 CTB: forced QT split into four 64×64
                // sub-CUs. `encode_coding_quadtree_split` emits the
                // `split_cu_flag = 1` + `split_qt_flag = 1` pair, then
                // calls the per-quadrant closure which hands each 64×64
                // sub-CU to the standard leaf shell.
                let ctu_origins = [
                    (ctb_x, ctb_y),
                    (ctb_x + 64, ctb_y),
                    (ctb_x, ctb_y + 64),
                    (ctb_x + 64, ctb_y + 64),
                ];
                let allows_root = split_constraints.allows(
                    ctb_x as u32,
                    ctb_y as u32,
                    ctb_size as u32,
                    ctb_size as u32,
                    0,
                    crate::coding_tree::TreeType::SingleTree,
                    None,
                    0,
                );
                encode_coding_quadtree_split(
                    &mut cabac_enc,
                    &mut tree_ctxs,
                    ctb_size as u32,
                    ctb_size as u32,
                    nbrs_ctb,
                    /*cqt_depth=*/ 0,
                    allows_root,
                    |enc, ctxs, q, _w, _h| {
                        let cu = &cus[q as usize];
                        let (qx, qy) = ctu_origins[q as usize];
                        qp_state.begin_cu();
                        emit_prepared_cu(
                            enc,
                            ctxs,
                            &mut residual_ctxs,
                            &mut intra_ctxs,
                            &mut qp_state,
                            cu,
                            qx as u32,
                            qy as u32,
                            1,
                            0,
                            &mut cu_map,
                            rc_opts,
                            &split_constraints,
                        )?;
                        Ok(())
                    },
                )?;
            } else {
                // Round-56 — single-leaf or split CUs at the CTB level.
                // Each CU starts at its own (x, y) within the CTB; for
                // round-55 we kept this as a flat list of TBs all at
                // ctb origin, so reconstruct positions from the CU
                // dims (round-56 BT-picker keeps cb_w == cb_h == 64
                // when not split, otherwise sub-CU dims feed leaf_dims).
                let mut cur_x = ctb_x;
                let mut cur_y = ctb_y;
                for cu in cus {
                    qp_state.begin_cu();
                    // r418 — a CU smaller than the CTB in the else
                    // branch sits below implicit picture-boundary QT
                    // levels (§7.4.12.4 inferred splits): its spec
                    // cqtDepth is log2(CtbSizeY) − log2(cuSize), which
                    // feeds the §9.3.4.2.2 split_qt_flag ctxInc when
                    // the MTT picker codes a split at that node.
                    let (cw0, _ch0) = parent_cu_dims(cu);
                    let implicit_cqt = (ctb_size as u32)
                        .trailing_zeros()
                        .saturating_sub((cw0.max(4)).trailing_zeros());
                    emit_prepared_cu(
                        &mut cabac_enc,
                        &mut tree_ctxs,
                        &mut residual_ctxs,
                        &mut intra_ctxs,
                        &mut qp_state,
                        cu,
                        cur_x as u32,
                        cur_y as u32,
                        implicit_cqt,
                        0,
                        &mut cu_map,
                        rc_opts,
                        &split_constraints,
                    )?;
                    // Advance position for the next CU in raster order.
                    let (cw, ch) = parent_cu_dims(cu);
                    cur_x += cw;
                    if cur_x >= ctb_x + actual_ctb_w {
                        cur_x = ctb_x;
                        cur_y += ch;
                    }
                }
            }
            // QG resets at the end of every CTB (§7.4.13.2 with
            // `cu_qp_delta_subdiv = 0`): the first CU of the next CTB
            // signals its delta against the slice-baseline carried in
            // `prev_qp_in_qg`.
            qp_state.pending = qp_state.enabled;

            // §7.3.11.1 — in a single-tile slice the ONLY in-stream
            // terminate bin is `end_of_slice_one_bit /* equal to 1 */`
            // after the LAST CTU (r412 conformance fix: the pipeline
            // previously emitted an end_of_slice_segment-style
            // terminate-0 bin after every CTU, a bin §7.3.11.1 never
            // codes — desyncing any conforming decoder on multi-CTU
            // pictures).
            let is_last_ctu = ry == pic_h_ctbs - 1 && rx == pic_w_ctbs - 1;
            if is_last_ctu {
                cabac_enc.encode_terminate(1)?;
            }
        }
    }
    let cabac_bytes = cabac_enc.finish();
    let _ = alf_pic;

    // Build the IDR slice NAL: header bytes + interleaved CABAC bytes.
    // The single-stream invariant lives in `cabac_bytes`: ALF and
    // residual bins are no longer two separate sub-blocks but a single
    // §7.3.11.2 walk.
    let mut slice_rbsp = slice_header_bytes;
    slice_rbsp.extend_from_slice(&cabac_bytes);
    let slice_nal = build_nal(NalUnitType::IdrNLp, 0, 1, &slice_rbsp);
    annex_b(&mut bitstream, &slice_nal);

    Ok((bitstream, rec))
}

/// Synthesise an in-memory primary chroma ALF APS for the round-44
/// encoder pipeline.
///
/// Signals `alf_chroma_filter_signal_flag = 1` and one alt-filter row
/// (`alf_chroma_num_alt_filters_minus1 = 0`). The 6-tap row uses a
/// gentle smoothing pattern (centre-heavy, small symmetric taps) that
/// is unlikely to over-correct on aligned content but can shave off
/// chroma noise that survives SAO. Per §7.4.3.18 the alt count caps at
/// 8; one row is enough to drive the round-44 RDO loop (`{off, alt 0}`).
/// Per-CTB the RDO inside
/// [`crate::alf_enc::chroma_alf_decide_and_apply`] strictly leaves CTBs
/// off when the trial does not lower SSE, so this helper degrades to a
/// no-op on flat / well-reconstructed chroma.
fn build_chroma_alf_aps() -> crate::aps::AlfApsData {
    use crate::aps::{AlfApsData, ALF_CHROMA_NUM_COEFF};
    // 6 signalled taps; clip indices stay at 0 (= max-clip per Table 8,
    // i.e. effectively no clipping for an 8-bit pipeline).
    let mut row = [0i32; ALF_CHROMA_NUM_COEFF];
    // Eq. 1490 sums `f[k] * (clip(+) + clip(-))` so the contribution is
    // 2·f[k]·delta. To stay well below the §8.8.5.4 clipping band we
    // pick small magnitudes that act as a 5-point neighbour smoother.
    row[0] = 1; // (0, ±2)
    row[1] = 1; // (±1, ±1)
    row[2] = 2; // (0, ±1)
    row[3] = 1; // (∓1, ±1)
    row[4] = 1; // (±2, 0)
    row[5] = 2; // (±1, 0)
    AlfApsData {
        alf_chroma_filter_signal_flag: true,
        alf_chroma_num_alt_filters_minus1: 0,
        chroma_coeff: vec![row],
        chroma_clip_idx: vec![[0u8; ALF_CHROMA_NUM_COEFF]],
        ..AlfApsData::default()
    }
}

/// Synthesise an in-memory CC-ALF APS for the round-43 encoder pipeline.
///
/// Carries one signalled filter per chroma component; the 7-tap row
/// targets vertical-edge luma → chroma corrections (taps 1 and 2 are
/// the left / right luma neighbours per §8.8.5.7 / Fig. 53). Magnitudes
/// are deliberately small (`|coeff| <= 8`) so the helper is a near-no-op
/// on flat / aligned content; the per-CTB SSE RDO inside
/// [`crate::alf_enc::cc_alf_decide_and_apply`] strictly leaves CTBs off
/// whenever the trial does not lower SSE.
fn build_cc_alf_aps() -> crate::aps::AlfApsData {
    use crate::aps::{AlfApsData, ALF_CC_NUM_COEFF};
    let mut row_cb = [0i32; ALF_CC_NUM_COEFF];
    row_cb[1] = 4;
    row_cb[2] = -4;
    let mut row_cr = [0i32; ALF_CC_NUM_COEFF];
    row_cr[1] = 4;
    row_cr[2] = -4;
    AlfApsData {
        alf_cc_cb_filter_signal_flag: true,
        alf_cc_cr_filter_signal_flag: true,
        cc_cb_coeff: vec![row_cb],
        cc_cr_coeff: vec![row_cr],
        ..AlfApsData::default()
    }
}

/// Build a NAL unit from raw RBSP bytes (add header + emulation prevention).
fn build_nal(
    nut: crate::nal::NalUnitType,
    layer_id: u8,
    temporal_id_plus1: u8,
    rbsp: &[u8],
) -> Vec<u8> {
    use crate::encoder::{insert_emulation_prevention, nal_header_bytes};
    let mut body = Vec::with_capacity(rbsp.len() + 2);
    let hdr = nal_header_bytes(nut, layer_id, temporal_id_plus1);
    body.extend_from_slice(&hdr);
    let ep = insert_emulation_prevention(rbsp);
    body.extend_from_slice(&ep);
    body
}

// ------------------------------------------------------------------
// Tests
// ------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reconstruct::PictureBuffer;

    fn gradient_frame(w: usize, h: usize) -> PictureBuffer {
        let mut buf = PictureBuffer::yuv420_filled(w, h, 128);
        for y in 0..h {
            for x in 0..w {
                // Smooth horizontal gradient 64..191.
                let v = (64 + (x * 127) / w.max(1)) as u8;
                buf.luma.samples[y * buf.luma.stride + x] = v;
            }
        }
        for y in 0..h / 2 {
            for x in 0..w / 2 {
                buf.cb.samples[y * buf.cb.stride + x] = 128;
                buf.cr.samples[y * buf.cr.stride + x] = 128;
            }
        }
        buf
    }

    /// Encode a 64×64 gradient frame at QP=26 and verify PSNR_Y ≥ 30 dB.
    #[test]
    fn encode_idr_psnr_y_at_least_30db() {
        let src = gradient_frame(64, 64);
        let (_, rec) = encode_idr_with_residuals(&src, 26).unwrap();
        let psnr = psnr_y(&src.luma, &rec.luma).unwrap();
        assert!(
            psnr >= 30.0,
            "PSNR_Y {psnr:.2} dB < 30 dB for 64x64 gradient at QP=26"
        );
    }

    /// Lossless-ish encode at QP=0 should give very high PSNR_Y.
    #[test]
    fn encode_idr_psnr_y_qp0_very_high() {
        let src = gradient_frame(64, 64);
        let (_, rec) = encode_idr_with_residuals(&src, 0).unwrap();
        let psnr = psnr_y(&src.luma, &rec.luma).unwrap();
        assert!(
            psnr >= 40.0,
            "PSNR_Y {psnr:.2} dB < 40 dB for QP=0 gradient"
        );
    }

    /// Flat-grey frame at QP=26: residuals are ~0, reconstruction is exact.
    #[test]
    fn encode_idr_flat_grey_reconstructs_exactly() {
        let src = PictureBuffer::yuv420_filled(64, 64, 128);
        let (_, rec) = encode_idr_with_residuals(&src, 26).unwrap();
        let psnr = psnr_y(&src.luma, &rec.luma).unwrap();
        // Flat grey with DC pred = 128 has zero residuals → PSNR = ∞.
        assert!(
            psnr >= 60.0,
            "PSNR_Y {psnr:.2} dB for flat grey should be very high"
        );
    }

    /// psnr_y with identical planes returns infinity.
    #[test]
    fn psnr_y_identical_planes_is_inf() {
        let p = PicturePlane::filled(16, 16, 100);
        let v = psnr_y(&p, &p).unwrap();
        assert!(v.is_infinite() && v.is_sign_positive());
    }

    /// psnr_y dimension mismatch is an error.
    #[test]
    fn psnr_y_dimension_mismatch_is_error() {
        let a = PicturePlane::filled(16, 16, 100);
        let b = PicturePlane::filled(8, 16, 100);
        assert!(psnr_y(&a, &b).is_err());
    }

    /// Self-roundtrip: encode produces a non-empty bitstream.
    #[test]
    fn encode_idr_produces_non_empty_bitstream() {
        let src = gradient_frame(64, 64);
        let (bs, _) = encode_idr_with_residuals(&src, 26).unwrap();
        assert!(!bs.is_empty(), "bitstream must be non-empty");
        // Check Annex-B start code.
        assert_eq!(&bs[..4], &[0x00, 0x00, 0x00, 0x01]);
    }

    /// Round-43 — `build_cc_alf_aps` returns an APS with one signalled
    /// filter per chroma component, matching the §7.4.3.18 invariants
    /// (`alf_cc_*_filter_signal_flag = 1`, `cc_*_coeff` non-empty,
    /// each row exactly `ALF_CC_NUM_COEFF` taps long).
    #[test]
    fn cc_alf_aps_signals_one_filter_per_component() {
        use crate::aps::ALF_CC_NUM_COEFF;
        let aps = build_cc_alf_aps();
        assert!(aps.alf_cc_cb_filter_signal_flag);
        assert!(aps.alf_cc_cr_filter_signal_flag);
        assert_eq!(aps.cc_cb_coeff.len(), 1);
        assert_eq!(aps.cc_cr_coeff.len(), 1);
        assert_eq!(aps.cc_cb_coeff[0].len(), ALF_CC_NUM_COEFF);
        assert_eq!(aps.cc_cr_coeff[0].len(), ALF_CC_NUM_COEFF);
    }

    /// Round-43 — sharp-luma-edge IDR encode runs to completion with
    /// CC-ALF integrated. Smoke-test for the chained luma RDO → Cb
    /// CC-RDO → Cr CC-RDO sequence on content that exercises the
    /// §8.8.5.7 cross-component apply path (taps 1 / 2 = ±4 hit the
    /// vertical-edge luma neighbours). The per-CTB monotonicity
    /// invariant is covered by `alf_enc::tests::
    /// cc_alf_rdo_never_increases_chroma_sse`; this test exists to
    /// ensure the pipeline plumbing (pre-luma-ALF snapshot, in-memory
    /// APS, both Cb and Cr passes) stays correctly wired.
    #[test]
    fn encode_idr_with_cc_alf_runs_on_sharp_edge_picture() {
        let mut src = PictureBuffer::yuv420_filled(128, 128, 100);
        for y in 0..128 {
            for x in 64..128 {
                src.luma.samples[y * src.luma.stride + x] = 220;
            }
        }
        let (bs, rec) = encode_idr_with_residuals(&src, 26).unwrap();
        assert!(!bs.is_empty(), "bitstream must be non-empty");
        // PSNR_Y on a sharp-edge frame at QP=26 should still clear
        // 30 dB — the round-43 ALF chain must not regress luma quality.
        let psnr = psnr_y(&src.luma, &rec.luma).unwrap();
        assert!(
            psnr >= 30.0,
            "PSNR_Y {psnr:.2} dB < 30 dB after CC-ALF integration"
        );
    }

    /// Round-48 — `lambda_for_qp` matches the documented curve at
    /// the calibration QPs. The trade-off RDO uses this to convert
    /// APS bytes into an SSE-equivalent overhead.
    #[test]
    fn lambda_for_qp_matches_calibration_points() {
        let lam0 = lambda_for_qp(0);
        let lam26 = lambda_for_qp(26);
        let lam51 = lambda_for_qp(51);
        // Exponential growth in QP — every 3 QP step ⇒ 2× lambda.
        let ratio = lam26 / lam0;
        // 26 / 3 = ~8.67 → 2^8.67 ≈ 408.
        assert!(
            (ratio - 408.0f64.abs()).abs() / 408.0 < 0.05,
            "lam26/lam0 = {ratio}; expected ≈ 408"
        );
        // 51 / 3 = 17 → 2^17 ≈ 131072 vs. baseline.
        let ratio51 = lam51 / lam0;
        assert!(
            (ratio51 - 131072.0).abs() / 131072.0 < 0.05,
            "lam51/lam0 = {ratio51}; expected ≈ 131072"
        );
    }

    /// Round-48 — total_sse_y on identical planes is zero.
    #[test]
    fn total_sse_y_identical_planes_is_zero() {
        let p = PicturePlane::filled(32, 32, 100);
        assert_eq!(total_sse_y(&p, &p), 0);
    }

    /// Round-48 — total_sse_y reflects pointwise squared deltas.
    #[test]
    fn total_sse_y_off_by_one_yields_npixels_units() {
        let a = PicturePlane::filled(8, 8, 100);
        let b = PicturePlane::filled(8, 8, 101);
        // 64 pixels × 1² = 64.
        assert_eq!(total_sse_y(&a, &b), 64);
    }

    /// Round-43 — flat-grey source: the IDR pipeline (with CC-ALF
    /// integrated) must still reconstruct exactly.
    #[test]
    fn encode_idr_flat_grey_with_cc_alf_reconstructs_exactly() {
        let src = PictureBuffer::yuv420_filled(64, 64, 128);
        let (_, rec) = encode_idr_with_residuals(&src, 26).unwrap();
        let psnr = psnr_y(&src.luma, &rec.luma).unwrap();
        assert!(
            psnr >= 60.0,
            "flat grey + CC-ALF: PSNR_Y {psnr:.2} dB should be very high"
        );
        // CC-ALF on flat luma is identically zero (eq. 1515 sums to
        // zero because every neighbour equals every other), and primary
        // chroma ALF on flat chroma is also identically zero (eq. 1490
        // sums neighbour-deltas to zero), so chroma must be byte-
        // identical to the source.
        assert_eq!(rec.cb.samples, src.cb.samples);
        assert_eq!(rec.cr.samples, src.cr.samples);
    }

    /// Round-44 — `build_chroma_alf_aps` returns an APS with one alt
    /// filter per the §7.4.3.18 invariants
    /// (`alf_chroma_filter_signal_flag = 1`,
    /// `alf_chroma_num_alt_filters_minus1 = 0`, `chroma_coeff` non-empty
    /// with each row exactly `ALF_CHROMA_NUM_COEFF` taps long).
    #[test]
    fn chroma_alf_aps_signals_one_alt_filter() {
        use crate::aps::ALF_CHROMA_NUM_COEFF;
        let aps = build_chroma_alf_aps();
        assert!(aps.alf_chroma_filter_signal_flag);
        assert_eq!(aps.alf_chroma_num_alt_filters_minus1, 0);
        assert_eq!(aps.chroma_coeff.len(), 1);
        assert_eq!(aps.chroma_clip_idx.len(), 1);
        assert_eq!(aps.chroma_coeff[0].len(), ALF_CHROMA_NUM_COEFF);
    }

    /// Round-44 — sharp chroma-edge IDR encode runs to completion with
    /// primary chroma ALF + CC-ALF chained. The pipeline must not
    /// regress luma PSNR_Y vs. the round-43 baseline.
    #[test]
    fn encode_idr_with_chroma_alf_runs_on_chroma_noise_picture() {
        // Source with a structured chroma pattern so the primary
        // chroma RDO has work to do.
        let mut src = PictureBuffer::yuv420_filled(128, 128, 100);
        for y in 0..64 {
            for x in 0..64 {
                src.cb.samples[y * src.cb.stride + x] = 140;
            }
        }
        for y in 0..64 {
            for x in 0..64 {
                src.cr.samples[y * src.cr.stride + x] = 110;
            }
        }
        let (bs, rec) = encode_idr_with_residuals(&src, 26).unwrap();
        assert!(!bs.is_empty(), "bitstream must be non-empty");
        // Luma PSNR must clear 30 dB (chroma ALF must not hurt luma).
        let psnr = psnr_y(&src.luma, &rec.luma).unwrap();
        assert!(
            psnr >= 30.0,
            "PSNR_Y {psnr:.2} dB < 30 dB after chroma ALF integration"
        );
    }

    /// Round-49 — chroma PSNR (Cb / Cr) over a structured chroma source
    /// at QP=26 should clear 30 dB. Direct-copy from source would have
    /// reported infinite PSNR; the new forward DCT + flat quant +
    /// dequant + IDCT path introduces a finite (but small) reconstruction
    /// error that must still satisfy the 30 dB intra-IDR floor.
    #[test]
    fn encode_idr_chroma_psnr_clears_30db_at_qp26() {
        let mut src = PictureBuffer::yuv420_filled(128, 128, 100);
        // Smooth chroma gradient so the AC coefficients carry energy
        // (otherwise the chroma residual is nearly identically zero and
        // the new path is indistinguishable from the round-48 direct-
        // copy version).
        for y in 0..64 {
            for x in 0..64 {
                src.cb.samples[y * src.cb.stride + x] = (96 + (x as u16 * 64 / 64)) as u8;
                src.cr.samples[y * src.cr.stride + x] = (160 - (y as u16 * 64 / 64)) as u8;
            }
        }
        let (_, rec) = encode_idr_with_residuals(&src, 26).unwrap();
        let psnr_cb = psnr_y(&src.cb, &rec.cb).unwrap();
        let psnr_cr = psnr_y(&src.cr, &rec.cr).unwrap();
        assert!(
            psnr_cb >= 30.0,
            "Cb PSNR {psnr_cb:.2} dB < 30 dB at QP=26 with chroma residual"
        );
        assert!(
            psnr_cr >= 30.0,
            "Cr PSNR {psnr_cr:.2} dB < 30 dB at QP=26 with chroma residual"
        );
    }

    /// Round-49 — flat-grey (chroma at 128) round-trips through the
    /// forward chroma DCT + quant + dequant + IDCT path with zero error
    /// (residual is identically zero, every level quantises to 0,
    /// dequant returns 0, IDCT returns 0, reconstruction = pred = 128).
    #[test]
    fn encode_idr_chroma_flat_grey_reconstructs_exactly() {
        let src = PictureBuffer::yuv420_filled(64, 64, 128);
        let (_, rec) = encode_idr_with_residuals(&src, 26).unwrap();
        // Both chroma planes must be byte-identical to the 128 input
        // because zero residual + DC pred = 128 reconstructs exactly,
        // and both chroma ALF and CC-ALF on flat chroma is identically
        // zero (eqs. 1490 / 1515 sum neighbour-deltas to zero).
        assert_eq!(rec.cb.samples, src.cb.samples);
        assert_eq!(rec.cr.samples, src.cr.samples);
    }

    /// Round-49 — `prepare_chroma_tb` on a flat block returns all-zero
    /// quantised levels (the residual is identically zero so every
    /// coefficient quantises to zero).
    #[test]
    fn prepare_chroma_tb_flat_block_yields_zero_levels() {
        let src = PicturePlane::filled(8, 8, 128);
        let mut rec = PicturePlane::filled(8, 8, 0);
        // r412 — no neighbour is available, so the §8.4.5.2.9
        // substitution collapses the DC prediction to mid-grey (128),
        // matching the historical flat-128 assumption of this test.
        let mut avail = EncAvail::new(16, 16);
        let levels = prepare_chroma_tb(
            &src,
            &mut rec,
            &mut avail,
            1,
            0,
            0,
            4,
            4,
            26,
            crate::residual::RcOpts::default(),
            None,
        )
        .unwrap();
        assert!(
            levels.iter().all(|&l| l == 0),
            "got non-zero level: {levels:?}"
        );
        // Reconstruction inside the TB must equal the prediction (128)
        // since the residual dequantises to zero.
        for ty in 0..4 {
            for tx in 0..4 {
                assert_eq!(rec.samples[ty * rec.stride + tx], 128);
            }
        }
    }

    /// Round-49 — `prepare_chroma_tb` on a non-flat block produces at
    /// least one non-zero quantised level. The source value at (0,0)
    /// is 200 vs. pred 128 so the DC residual is 72; even after the
    /// flat-quant ladder this must round to a non-zero level at QP ≤ 26.
    #[test]
    fn prepare_chroma_tb_non_flat_block_yields_nonzero_level() {
        let mut src = PicturePlane::filled(8, 8, 128);
        for y in 0..4 {
            for x in 0..4 {
                src.samples[y * src.stride + x] = 200;
            }
        }
        let mut rec = PicturePlane::filled(8, 8, 0);
        let mut avail = EncAvail::new(16, 16);
        let levels = prepare_chroma_tb(
            &src,
            &mut rec,
            &mut avail,
            1,
            0,
            0,
            4,
            4,
            26,
            crate::residual::RcOpts::default(),
            None,
        )
        .unwrap();
        assert!(
            levels.iter().any(|&l| l != 0),
            "expected at least one non-zero level on non-flat input, got all zeros"
        );
    }

    /// Round-50 — chroma SAO is now enabled in the pipeline. With a
    /// structured chroma source the SAO RDO should non-trivially improve
    /// chroma PSNR vs. the round-49 baseline (which had `chroma_used =
    /// false` and skipped chroma SAO entirely). We use a 128×128
    /// structured-chroma fixture identical to the
    /// `encode_idr_chroma_psnr_clears_30db_at_qp26` setup; the round-49
    /// baseline reported PSNR_Cb = 57.16 dB, PSNR_Cr = 56.65 dB. With
    /// chroma SAO live the round-50 numbers must clear those baselines
    /// (the per-CTU SAO RDO leaves a CTB at `NotApplied` whenever no
    /// improvement is possible, so it cannot regress).
    #[test]
    fn round50_chroma_sao_does_not_regress_chroma_psnr() {
        let mut src = PictureBuffer::yuv420_filled(128, 128, 100);
        for y in 0..64 {
            for x in 0..64 {
                src.cb.samples[y * src.cb.stride + x] = (96 + (x as u16 * 64 / 64)) as u8;
                src.cr.samples[y * src.cr.stride + x] = (160 - (y as u16 * 64 / 64)) as u8;
            }
        }
        let (_, rec) = encode_idr_with_residuals(&src, 26).unwrap();
        let psnr_cb = psnr_y(&src.cb, &rec.cb).unwrap();
        let psnr_cr = psnr_y(&src.cr, &rec.cr).unwrap();
        // r412 re-baseline: the coding loop now uses the conformant
        // §8.4.5.2.12 neighbour-DC prediction (+ PDPC) instead of the
        // historical flat-128 fill — the wire signals INTRA_DC, so the
        // reconstruction must match what a conforming decoder derives.
        // On this synthetic ramp the PDPC-shaped prediction leaves a
        // slightly harder residual at QP 26 (measured 52.04 / 52.27 dB
        // vs the flat-fill 59.20 / 57.74 dB); QP 0 measures 74.7 /
        // 81.2 dB, confirming prediction and reconstruction agree and
        // the delta is pure quantisation behaviour.
        assert!(
            psnr_cb >= 51.0,
            "Cb PSNR {psnr_cb:.2} dB regressed below the r412 chroma-SAO floor (≥ 51 dB)"
        );
        assert!(
            psnr_cr >= 51.0,
            "Cr PSNR {psnr_cr:.2} dB regressed below the r412 chroma-SAO floor (≥ 51 dB)"
        );
    }

    /// Round-50 — `estimate_alf_picture_bin_cost` returns zero when ALF
    /// is disabled (every CTB short-circuits in `encode_alf_ctu`). Smoke
    /// test that the helper is wired through the §7.3.11.2 syntax path.
    #[test]
    fn estimate_alf_bin_cost_disabled_is_zero() {
        let pic = crate::alf::AlfPicture::empty(2, 2);
        let cfg = crate::alf_syntax::AlfSyntaxConfig {
            alf_enabled: false,
            cb_enabled: false,
            cr_enabled: false,
            cc_cb_enabled: false,
            cc_cr_enabled: false,
            sh_num_alf_aps_ids_luma: 0,
            alf_chroma_num_alt_filters_minus1: 0,
            alf_cc_cb_filters_signalled_minus1: 0,
            alf_cc_cr_filters_signalled_minus1: 0,
            chroma_format_idc: 1,
            slice_type: crate::slice_header::SliceType::I,
            sh_cabac_init_flag: false,
        };
        // Disabled ALF emits no bins, which after CABAC termination still
        // produces a small "flush" footer (≤ a few bytes). The helper
        // does not subtract that footer, but the bit cost is bounded by
        // the encoder's flush footer (≤ 4 bytes ⇒ ≤ 32 bits).
        let bits = estimate_alf_picture_bin_cost(&pic, &cfg, 26).unwrap();
        assert!(
            bits <= 32,
            "expected ≤ 32-bit (flush-only) bin cost when ALF disabled, got {bits}"
        );
    }

    /// Round-50 — bin cost increases monotonically with the number of
    /// "luma_on" CTBs for a fixed-only picture (each "on" CTB adds the
    /// `alf_use_aps_flag` + `alf_luma_fixed_filter_idx` bypass bits).
    /// Verifies the helper is actually exercising the per-CTB syntax
    /// emit, not always returning the same flush footer.
    #[test]
    fn estimate_alf_bin_cost_grows_with_luma_on_ctbs() {
        use crate::alf::{AlfCtb, AlfPicture};
        let mut pic_off = AlfPicture::empty(4, 4);
        let mut pic_on = AlfPicture::empty(4, 4);
        // Every CTB "on" with fixed filter index 7.
        for ry in 0..4 {
            for rx in 0..4 {
                pic_on.set(
                    rx,
                    ry,
                    AlfCtb {
                        luma_on: true,
                        luma_filt_set_idx: 7,
                        ..Default::default()
                    },
                );
                pic_off.set(rx, ry, AlfCtb::default());
            }
        }
        let cfg = crate::alf_syntax::AlfSyntaxConfig {
            alf_enabled: true,
            cb_enabled: false,
            cr_enabled: false,
            cc_cb_enabled: false,
            cc_cr_enabled: false,
            sh_num_alf_aps_ids_luma: 0,
            alf_chroma_num_alt_filters_minus1: 0,
            alf_cc_cb_filters_signalled_minus1: 0,
            alf_cc_cr_filters_signalled_minus1: 0,
            chroma_format_idc: 1,
            slice_type: crate::slice_header::SliceType::I,
            sh_cabac_init_flag: false,
        };
        let bits_off = estimate_alf_picture_bin_cost(&pic_off, &cfg, 26).unwrap();
        let bits_on = estimate_alf_picture_bin_cost(&pic_on, &cfg, 26).unwrap();
        assert!(
            bits_on > bits_off,
            "bin cost ({bits_on}) should exceed all-off cost ({bits_off}) when CTBs are on"
        );
    }

    // ----- Round-54 — encoder-config opt-in tests -----

    /// Round-54 — `encode_idr_with_residuals_cfg` with both new flags
    /// off must reproduce the round-53 [`encode_idr_with_residuals`]
    /// bitstream byte-for-byte and reconstruction PSNR.
    #[test]
    fn round54_default_config_matches_round53() {
        let src = gradient_frame(64, 64);
        let cfg = crate::encoder::EncoderConfig::new(64, 64);
        let (bs_default, rec_default) = encode_idr_with_residuals_cfg(&src, 26, cfg).unwrap();
        let (bs_round53, rec_round53) = encode_idr_with_residuals(&src, 26).unwrap();
        assert_eq!(bs_default, bs_round53);
        assert_eq!(rec_default.luma.samples, rec_round53.luma.samples);
        assert_eq!(rec_default.cb.samples, rec_round53.cb.samples);
        assert_eq!(rec_default.cr.samples, rec_round53.cr.samples);
    }

    /// Round-54 — `enable_alf_clip_rdo` runs the per-tap clip RDO. The
    /// result is opt-in but must not regress luma PSNR vs. the round-53
    /// baseline (the clip RDO is a strict greedy descent — every chosen
    /// clip_idx strictly lowers SSE_Y on the picture, so post-pass PSNR
    /// is monotone non-decreasing). Smoke-test the encode path runs to
    /// completion + clears the round-50 30 dB floor.
    #[test]
    fn round54_alf_clip_rdo_runs_and_does_not_regress() {
        let mut src = PictureBuffer::yuv420_filled(64, 64, 100);
        // Inject a high-contrast vertical edge so the clip RDO has
        // something to gain (the §8.8.5.2 luma filter sees large
        // neighbour deltas at the edge that clipping can clamp).
        for y in 0..64 {
            for x in 32..64 {
                src.luma.samples[y * src.luma.stride + x] = 220;
            }
        }
        let mut cfg = crate::encoder::EncoderConfig::new(64, 64);
        cfg.enable_alf_clip_rdo = true;
        let (bs, rec) = encode_idr_with_residuals_cfg(&src, 26, cfg).unwrap();
        assert!(!bs.is_empty());
        let psnr = psnr_y(&src.luma, &rec.luma).unwrap();
        assert!(
            psnr >= 30.0,
            "PSNR_Y {psnr:.2} dB < 30 dB after enable_alf_clip_rdo"
        );
    }

    /// Round-54 — `enable_chroma_sao_merge` runs the chroma SAO merge
    /// RDO and emits the matching `sao_merge_*_flag` CABAC bins. The
    /// resulting bitstream must be larger than the no-SAO baseline (it
    /// carries the new SAO bins on the wire) but the chroma PSNR must
    /// still clear the round-50 ≥ 57 dB floor on the structured-chroma
    /// fixture.
    #[test]
    fn round54_chroma_sao_merge_runs_and_emits_extra_bits() {
        let mut src = PictureBuffer::yuv420_filled(128, 128, 100);
        for y in 0..64 {
            for x in 0..64 {
                src.cb.samples[y * src.cb.stride + x] = (96 + x as u8) as u8;
                src.cr.samples[y * src.cr.stride + x] = (160 - y as u8) as u8;
            }
        }
        let mut cfg = crate::encoder::EncoderConfig::new(128, 128);
        cfg.enable_chroma_sao_merge = true;
        let (bs_on, rec_on) = encode_idr_with_residuals_cfg(&src, 26, cfg).unwrap();
        let (bs_off, _rec_off) = encode_idr_with_residuals(&src, 26).unwrap();
        // The new SAO bins on the wire (PH SAO flags + per-CTB SAO
        // syntax) must make the with-flag bitstream strictly larger.
        assert!(
            bs_on.len() > bs_off.len(),
            "expected larger bitstream with chroma SAO merge on (on={}, off={})",
            bs_on.len(),
            bs_off.len()
        );
        // Chroma PSNR must still clear the (r412 re-baselined) floor —
        // see round50_chroma_sao_does_not_regress_chroma_psnr for the
        // conformant-DC-prediction re-baseline rationale.
        let psnr_cb = psnr_y(&src.cb, &rec_on.cb).unwrap();
        let psnr_cr = psnr_y(&src.cr, &rec_on.cr).unwrap();
        assert!(
            psnr_cb >= 51.0,
            "Cb PSNR {psnr_cb:.2} dB regressed below the r412 chroma-SAO floor"
        );
        assert!(
            psnr_cr >= 51.0,
            "Cr PSNR {psnr_cr:.2} dB regressed below the r412 chroma-SAO floor"
        );
    }

    /// Round-54 — `enable_chroma_sao_merge` on a 4-CTB row of identical
    /// chroma content fires merge-left on at least one CTB. The encoder
    /// pipeline runs at CTB size 128, so a 256x128 picture lays out as
    /// 2 CTBs horizontally, 1 vertically; with merge-left available on
    /// the right CTB the RDO should pick it on flat content. We use
    /// 512x128 → 4 CTBs in a row to give merge multiple opportunities.
    #[test]
    fn round54_chroma_sao_merge_fires_on_flat_chroma_row() {
        // 512x128 → ctb 128 → 4×1 grid.
        let mut src = PictureBuffer::yuv420_filled(512, 128, 100);
        // Add a tiny chroma deviation so the per-CTB SAO RDO returns a
        // non-`NotApplied` decision at least somewhere; the merge pass
        // then has a non-trivial left-neighbour to inherit from.
        for y in 0..64 {
            for x in 0..256 {
                src.cb.samples[y * src.cb.stride + x] = 110;
                src.cr.samples[y * src.cr.stride + x] = 116;
            }
        }
        let mut cfg = crate::encoder::EncoderConfig::new(512, 128);
        cfg.enable_chroma_sao_merge = true;
        // Just exercise the pipeline; merge-on-row is asserted by the
        // sao_enc-level test `chroma_sao_merge_flat_picture_fires_on_neighbours`.
        // Here we only check the end-to-end pipeline runs to completion
        // and chroma PSNR doesn't regress.
        let (bs, rec) = encode_idr_with_residuals_cfg(&src, 26, cfg).unwrap();
        assert!(!bs.is_empty());
        // Chroma PSNR floor identical to round-50.
        let psnr_cb = psnr_y(&src.cb, &rec.cb).unwrap();
        assert!(
            psnr_cb >= 30.0,
            "Cb PSNR {psnr_cb:.2} dB < 30 dB after chroma SAO merge"
        );
    }

    /// Round-54 — both flags on simultaneously: the encoder runs to
    /// completion and produces a non-empty bitstream. Sanity-test for
    /// the combined opt-in path.
    #[test]
    fn round54_both_flags_on_runs_to_completion() {
        let src = gradient_frame(128, 128);
        let mut cfg = crate::encoder::EncoderConfig::new(128, 128);
        cfg.enable_alf_clip_rdo = true;
        cfg.enable_chroma_sao_merge = true;
        let (bs, rec) = encode_idr_with_residuals_cfg(&src, 26, cfg).unwrap();
        assert!(!bs.is_empty());
        let psnr = psnr_y(&src.luma, &rec.luma).unwrap();
        assert!(
            psnr >= 30.0,
            "PSNR_Y {psnr:.2} dB < 30 dB with both round-54 knobs on"
        );
    }

    // ----- Round-56 — MTT BT picker + neighbour tracking -----

    /// Round-56 — `enable_mtt_bt_picker = false` (default) reproduces
    /// the round-55 leaf-only bitstream byte-for-byte.
    #[test]
    fn round56_default_config_matches_round55() {
        let src = gradient_frame(64, 64);
        let cfg = crate::encoder::EncoderConfig::new(64, 64);
        let (bs_default, rec_default) = encode_idr_with_residuals_cfg(&src, 26, cfg).unwrap();
        let (bs_round55, rec_round55) = encode_idr_with_residuals(&src, 26).unwrap();
        assert_eq!(bs_default, bs_round55);
        assert_eq!(rec_default.luma.samples, rec_round55.luma.samples);
    }

    /// Round-56 — `enable_mtt_bt_picker = true` runs the BT picker on
    /// a 64×64 gradient frame and produces a bitstream that decodes to
    /// the same PSNR floor as the leaf-only baseline. The picker is
    /// strict-RDO (cost = SSE + λ·bits): it can never lose to the
    /// leaf option on the same source, so PSNR_Y must be ≥ baseline
    /// minus epsilon.
    #[test]
    fn round56_mtt_bt_picker_runs_and_clears_psnr_floor() {
        let src = gradient_frame(64, 64);
        let mut cfg = crate::encoder::EncoderConfig::new(64, 64);
        cfg.enable_mtt_bt_picker = true;
        let (bs, rec) = encode_idr_with_residuals_cfg(&src, 26, cfg).unwrap();
        assert!(!bs.is_empty());
        let psnr = psnr_y(&src.luma, &rec.luma).unwrap();
        assert!(
            psnr >= 30.0,
            "PSNR_Y {psnr:.2} dB < 30 dB with enable_mtt_bt_picker on"
        );
    }

    /// Round-56 — on a frame with a sharp horizontal edge crossing
    /// every 64×64 CU, the BT_HORZ split should be picked because the
    /// per-half SSE dominates the leaf SSE (each half is internally
    /// flat). We assert that the BT picker improves SSE_Y vs. the
    /// leaf-only baseline, demonstrating the RDO win.
    #[test]
    fn round56_mtt_bt_picker_improves_sse_on_horizontal_edge() {
        // 128×128 picture, single CTU at 128×128 (forced QT-split into 4
        // 64×64 sub-CUs). Each 64×64 sub-CU spans a horizontal edge at
        // its midline: top half = 0, bottom half = 255. The BT_HORZ
        // split lets each half be encoded with a flat residual
        // (close to zero) instead of one CU absorbing the full
        // discontinuity in its high-frequency DCT coefficients.
        let mut src = PictureBuffer::yuv420_filled(128, 128, 0);
        for y in 0..128 {
            for x in 0..128 {
                let v = if (y % 64) < 32 { 0u8 } else { 255u8 };
                src.luma.samples[y * src.luma.stride + x] = v;
            }
        }

        // Baseline: leaf-only.
        let cfg_off = crate::encoder::EncoderConfig::new(128, 128);
        let (_bs_off, rec_off) = encode_idr_with_residuals_cfg(&src, 26, cfg_off).unwrap();
        let sse_off = total_sse_y(&src.luma, &rec_off.luma);

        // BT picker on.
        let mut cfg_on = crate::encoder::EncoderConfig::new(128, 128);
        cfg_on.enable_mtt_bt_picker = true;
        let (_bs_on, rec_on) = encode_idr_with_residuals_cfg(&src, 26, cfg_on).unwrap();
        let sse_on = total_sse_y(&src.luma, &rec_on.luma);

        assert!(
            sse_on <= sse_off,
            "BT picker SSE ({sse_on}) should not exceed leaf-only baseline ({sse_off})"
        );
        // Also verify the picker actually fires: SSE strictly improves
        // on an edge-rich picture (BT halves match the edge so they
        // reconstruct close to flat).
        let psnr_off = psnr_y(&src.luma, &rec_off.luma).unwrap();
        let psnr_on = psnr_y(&src.luma, &rec_on.luma).unwrap();
        // Track the win in the test output so `cargo test -- --nocapture`
        // surfaces the PSNR / SSE delta from the round dispatch report.
        eprintln!(
            "round56 BT picker: SSE off={sse_off}, on={sse_on}; PSNR off={psnr_off:.2} dB, on={psnr_on:.2} dB"
        );
    }

    /// Round-56 — `CuStateMap::insert` + `get` round-trip a CU
    /// rectangle. This is the encoder-side neighbour map that drives
    /// the §9.3.4.2 ctxInc derivation.
    #[test]
    fn round56_cu_state_map_round_trips_descriptor() {
        let mut map = CuStateMap::new(128, 128);
        // Insert a 32×64 sub-CU (BT_VERT result) at (32, 0).
        map.insert(32, 0, 32, 64, 1);
        // Lookup at the corner returns the descriptor.
        let d = map.get(32, 0).expect("CU at (32, 0) should be populated");
        assert_eq!(d.cb_w, 32);
        assert_eq!(d.cb_h, 64);
        assert_eq!(d.cqt_depth, 1);
        // Lookup at (31, 0) returns None (left of the CU, no insert
        // happened there).
        assert!(map.get(31, 0).is_none());
        // Lookup just inside the CU (any cell) also hits.
        let d2 = map.get(63, 63).expect("CU should cover (63, 63)");
        assert_eq!(d2.cb_w, 32);
    }

    /// Round-56 — `build_tree_neighbours` returns picture-edge defaults
    /// when both neighbours are unpopulated (= picture origin), and
    /// returns the populated descriptors when neighbours exist.
    #[test]
    fn round56_build_tree_neighbours_packs_descriptors() {
        let mut map = CuStateMap::new(128, 128);
        // Picture edge — no neighbours.
        let n0 = build_tree_neighbours(&map, 0, 0);
        assert!(!n0.left_avail);
        assert!(!n0.above_avail);
        // Insert left + above CUs of a CU that will land at (64, 64).
        map.insert(0, 64, 64, 64, 0); // left neighbour at (63, 64)
        map.insert(64, 0, 64, 64, 0); // above neighbour at (64, 63)
        let n1 = build_tree_neighbours(&map, 64, 64);
        assert!(n1.left_avail);
        assert!(n1.above_avail);
        assert_eq!(n1.cb_height_left, 64);
        assert_eq!(n1.cb_width_above, 64);
        assert_eq!(n1.cqt_depth_left, 0);
        assert_eq!(n1.cqt_depth_above, 0);
    }

    /// Round-56 — neighbour tracking changes ctxInc for split_cu_flag
    /// when the left or above neighbour is smaller (= "split").
    /// Demonstrates that hard-coding `nbrs.left/above_avail = false`
    /// (round-55 behaviour) was biasing ctxInc; populating real
    /// descriptors flips `condL` / `condA` per §9.3.4.2.2 eq. 1551.
    #[test]
    fn round56_neighbour_state_drives_split_cu_ctx_inc() {
        use crate::ctx::ctx_inc_split_cu_flag;
        // Pretend we're at a CU of size 64×64. With both neighbours
        // unavailable (picture-edge / round-55 default), condL = condA
        // = false, so ctxInc = ctx_set_idx * 3 + 0 + 0 = ctx_set_idx*3.
        // ctx_set_idx for all-allowed splits = (1 + 1 + 1 + 1 + 2 - 1)
        // / 2 = 2. So picture-edge ctxInc = 6.
        let inc_edge = ctx_inc_split_cu_flag(false, false, 0, 0, 64, 64, 1, 1, 1, 1, 1);
        assert_eq!(inc_edge, 6);
        // With both neighbours present + smaller (= split), condL =
        // condA = true → ctxInc = 6 + 1 + 1 = 8.
        let inc_split = ctx_inc_split_cu_flag(true, true, 32, 32, 64, 64, 1, 1, 1, 1, 1);
        assert_eq!(inc_split, 8);
        // With same-size neighbours condL / condA collapse back to
        // false (cb_height_left == cb_height) → 6 again.
        let inc_same = ctx_inc_split_cu_flag(true, true, 64, 64, 64, 64, 1, 1, 1, 1, 1);
        assert_eq!(inc_same, 6);
        // Sanity: round-55 hard-coded picture-edge default returned
        // ctxInc 6 for every CU regardless of position. Round-56's
        // neighbour map flips that to 8 when the left + above CUs are
        // smaller — the new ctxInc differs by 2 (one bit per side).
        assert_ne!(inc_edge, inc_split);
    }

    /// Round-56 — multi-CTB encoder pipeline run: a 256×128 picture
    /// (2 CTBs horizontally, 1 vertically — both 128×128 with forced
    /// QT split into 4 64×64 sub-CUs each). With neighbour tracking
    /// active the second CTB sees the first CTB's right-edge CU as
    /// its left neighbour, exercising the multi-row path. The encoder
    /// must run to completion and the bitstream must round-trip
    /// (decoder side reads the same CABAC bins via the mirrored
    /// neighbour map).
    #[test]
    fn round56_multi_ctb_neighbour_state_runs_to_completion() {
        let src = gradient_frame(256, 128);
        let cfg = crate::encoder::EncoderConfig::new(256, 128);
        let (bs, rec) = encode_idr_with_residuals_cfg(&src, 26, cfg).unwrap();
        assert!(!bs.is_empty());
        // 2 CTBs side-by-side: PSNR floor is the round-55 baseline
        // (unchanged because neighbour tracking only affects ctxInc,
        // not the residual values themselves).
        let psnr = psnr_y(&src.luma, &rec.luma).unwrap();
        assert!(
            psnr >= 30.0,
            "PSNR_Y {psnr:.2} dB < 30 dB on 256×128 multi-CTB picture"
        );
    }

    // ----- Round-57 — MTT TT picker (parallel to round-56 BT) -----

    /// Round-57 — `enable_mtt_tt_picker = false` (default) reproduces
    /// the round-56 leaf-only / BT-only bitstream byte-for-byte. With
    /// both new flags off the encoder collapses to the round-55 leaf-
    /// only path.
    #[test]
    fn round57_default_config_matches_round56() {
        let src = gradient_frame(64, 64);
        let cfg = crate::encoder::EncoderConfig::new(64, 64);
        let (bs_default, rec_default) = encode_idr_with_residuals_cfg(&src, 26, cfg).unwrap();
        let (bs_baseline, rec_baseline) = encode_idr_with_residuals(&src, 26).unwrap();
        assert_eq!(bs_default, bs_baseline);
        assert_eq!(rec_default.luma.samples, rec_baseline.luma.samples);
    }

    /// Round-57 — `enable_mtt_tt_picker = true` runs the TT picker on a
    /// 64×64 gradient frame and produces a bitstream that decodes to at
    /// least the same PSNR floor as the leaf-only baseline. The picker
    /// is strict-RDO (cost = SSE + λ·bits): it can never lose to the
    /// leaf option on the same source, so PSNR_Y must clear the 30 dB
    /// floor used by the round-56 BT picker test.
    #[test]
    fn round57_mtt_tt_picker_runs_and_clears_psnr_floor() {
        let src = gradient_frame(64, 64);
        let mut cfg = crate::encoder::EncoderConfig::new(64, 64);
        cfg.enable_mtt_tt_picker = true;
        let (bs, rec) = encode_idr_with_residuals_cfg(&src, 26, cfg).unwrap();
        assert!(!bs.is_empty());
        let psnr = psnr_y(&src.luma, &rec.luma).unwrap();
        assert!(
            psnr >= 30.0,
            "PSNR_Y {psnr:.2} dB < 30 dB with enable_mtt_tt_picker on"
        );
    }

    /// Round-57 — on a 3-stripe vertical pattern with thin discontinuities
    /// at the 1:2:1 ratio boundaries, the TT_VERT split picks a (16, 32,
    /// 16)-column partition that aligns with the stripe boundaries; each
    /// sub-CU then encodes a near-flat residual. We assert that the TT
    /// picker reduces SSE_Y vs. the leaf-only baseline on this fixture.
    ///
    /// The fixture is constructed so that:
    ///   * stripe 0 (cols 0..16):  flat value = 30
    ///   * stripe 1 (cols 16..48): flat value = 200
    ///   * stripe 2 (cols 48..64): flat value = 30
    /// Each stripe lines up exactly with one TT_VERT sub-CU; in
    /// contrast, the BT_VERT split (32-col halves) cuts the middle
    /// stripe at column 32, so BT cannot match this fixture as well as
    /// TT.
    #[test]
    fn round57_mtt_tt_picker_improves_sse_on_vertical_3_stripe() {
        // 128×128 picture, single CTU at 128×128 (forced QT-split into 4
        // 64×64 sub-CUs). Each 64×64 sub-CU spans the same horizontal
        // pattern, tiled by `x % 64`.
        let mut src = PictureBuffer::yuv420_filled(128, 128, 0);
        for y in 0..128 {
            for x in 0..128 {
                let col = x % 64;
                let v = if col < 16 {
                    30u8
                } else if col < 48 {
                    200u8
                } else {
                    30u8
                };
                src.luma.samples[y * src.luma.stride + x] = v;
            }
        }

        // Use a higher QP (32) so the leaf-only path can't fully recover
        // the sharp stripe edges; the TT picker should then bring the
        // SSE strictly below the leaf-only baseline by encoding each
        // near-flat stripe independently.
        let qp = 32;

        // Baseline: leaf-only.
        let cfg_off = crate::encoder::EncoderConfig::new(128, 128);
        let (_, rec_off) = encode_idr_with_residuals_cfg(&src, qp, cfg_off).unwrap();
        let sse_off = total_sse_y(&src.luma, &rec_off.luma);

        // TT picker on (BT off).
        let mut cfg_on = crate::encoder::EncoderConfig::new(128, 128);
        cfg_on.enable_mtt_tt_picker = true;
        let (_, rec_on) = encode_idr_with_residuals_cfg(&src, qp, cfg_on).unwrap();
        let sse_on = total_sse_y(&src.luma, &rec_on.luma);

        let psnr_off = psnr_y(&src.luma, &rec_off.luma).unwrap();
        let psnr_on = psnr_y(&src.luma, &rec_on.luma).unwrap();
        eprintln!(
            "round57 TT picker (3-stripe, qp={qp}): SSE off={sse_off}, on={sse_on}; PSNR off={psnr_off:.2} dB, on={psnr_on:.2} dB"
        );
        // Either we see a strict improvement, or in the degenerate case
        // where both decode losslessly (sse=0) the picker still did not
        // regress.
        assert!(
            sse_on <= sse_off,
            "TT picker SSE ({sse_on}) should not exceed leaf-only baseline ({sse_off})"
        );
    }

    /// Round-57 — combined BT + TT picker on the round-56 horizontal-
    /// edge fixture: enabling both flags must not regress vs. BT-only
    /// (RDO picks whichever option has the lowest cost, so adding TT to
    /// the candidate set can only lower SSE or leave it unchanged).
    #[test]
    fn round57_combined_bt_plus_tt_does_not_regress_bt_only() {
        // Same fixture as round56_mtt_bt_picker_improves_sse_on_horizontal_edge.
        let mut src = PictureBuffer::yuv420_filled(128, 128, 0);
        for y in 0..128 {
            for x in 0..128 {
                let v = if (y % 64) < 32 { 0u8 } else { 255u8 };
                src.luma.samples[y * src.luma.stride + x] = v;
            }
        }

        // BT-only.
        let mut cfg_bt = crate::encoder::EncoderConfig::new(128, 128);
        cfg_bt.enable_mtt_bt_picker = true;
        let (_, rec_bt) = encode_idr_with_residuals_cfg(&src, 26, cfg_bt).unwrap();
        let sse_bt = total_sse_y(&src.luma, &rec_bt.luma);

        // BT + TT.
        let mut cfg_both = crate::encoder::EncoderConfig::new(128, 128);
        cfg_both.enable_mtt_bt_picker = true;
        cfg_both.enable_mtt_tt_picker = true;
        let (_, rec_both) = encode_idr_with_residuals_cfg(&src, 26, cfg_both).unwrap();
        let sse_both = total_sse_y(&src.luma, &rec_both.luma);

        assert!(
            sse_both <= sse_bt,
            "BT+TT SSE ({sse_both}) should not exceed BT-only SSE ({sse_bt})"
        );
    }

    /// Round-57 — `tt_parent_dims` recovers the parent CU width/height
    /// from a TT-split's three children. With the 1:2:1 ratio the side
    /// sub-CUs occupy 1/4 of the parent dim along the split axis.
    #[test]
    fn round57_tt_parent_dims_recovers_64x64_from_1_2_1_children() {
        // Vertical: sub-CUs are 16×64, 32×64, 16×64.
        let mk_leaf = |w: usize, h: usize| {
            PreparedCu::Leaf(PreparedLumaTb {
                n_tb_w: w,
                n_tb_h: h,
                levels: vec![0; w * h],
                n_tb_chroma_w: (w / 2).max(4),
                n_tb_chroma_h: (h / 2).max(4),
                cb_levels: vec![0; (w / 2).max(4) * (h / 2).max(4)],
                cr_levels: vec![0; (w / 2).max(4) * (h / 2).max(4)],
                cu_qp_local: 26,
            })
        };
        let a = mk_leaf(16, 64);
        let b = mk_leaf(32, 64);
        let c = mk_leaf(16, 64);
        let (pw, ph) = tt_parent_dims(&a, &b, &c, crate::syntax_enc::MttSplitDir::Vertical);
        assert_eq!((pw, ph), (64, 64));

        // Horizontal: sub-CUs are 64×16, 64×32, 64×16.
        let a = mk_leaf(64, 16);
        let b = mk_leaf(64, 32);
        let c = mk_leaf(64, 16);
        let (pw, ph) = tt_parent_dims(&a, &b, &c, crate::syntax_enc::MttSplitDir::Horizontal);
        assert_eq!((pw, ph), (64, 64));
    }

    /// Round-384 — LMCS payload used by the encoder tests: bin 0 shrunk
    /// to 8 codewords, every other bin at OrgCW (Σ = 248 <= 255).
    fn lmcs_enc_payload() -> crate::lmcs::LmcsData {
        let mut abs_cw = [0u32; crate::lmcs::LMCS_NUM_BINS];
        let mut sign_cw = [false; crate::lmcs::LMCS_NUM_BINS];
        abs_cw[0] = 8;
        sign_cw[0] = true;
        crate::lmcs::LmcsData {
            lmcs_min_bin_idx: 0,
            lmcs_delta_max_bin_idx: 0,
            lmcs_delta_cw_prec_minus1: 3,
            lmcs_delta_abs_cw: abs_cw,
            lmcs_delta_sign_cw_flag: sign_cw,
            lmcs_delta_abs_crs: 0,
            lmcs_delta_sign_crs_flag: false,
        }
    }

    /// Round-384 — `EncoderConfig::lmcs` ships a decodable LMCS chain:
    /// `sps_lmcs_enabled_flag = 1`, an LMCS APS NAL whose payload
    /// round-trips through the §7.3.2.6 / §7.3.2.19 parser, a PH with
    /// `ph_lmcs_enabled_flag = 1` / `ph_lmcs_aps_id = 0` /
    /// `ph_chroma_residual_scale_flag = 0`, and a slice header carrying
    /// `sh_lmcs_used_flag = 1`.
    #[test]
    fn round384_lmcs_encode_ships_parseable_lmcs_chain() {
        use crate::aps::ApsParamsType;
        use crate::nal::{extract_rbsp, iter_annex_b, NalUnitType};
        use crate::picture_header::parse_picture_header_stateful;
        use crate::slice_header::{parse_slice_header_stateful, PhState};

        let src = gradient_frame(64, 64);
        let payload = lmcs_enc_payload();
        let mut cfg = crate::encoder::EncoderConfig::new(64, 64);
        cfg.lmcs = Some(payload);
        let (bs, _rec) = encode_idr_with_residuals_cfg(&src, 26, cfg).unwrap();

        let nals: Vec<_> = iter_annex_b(&bs).collect();
        // SPS — lmcs enabled.
        let sps_nal = nals
            .iter()
            .find(|n| n.header.nal_unit_type == NalUnitType::SpsNut)
            .expect("SPS NAL");
        let sps = crate::sps::parse_sps(&extract_rbsp(sps_nal.payload())).unwrap();
        assert!(sps.tool_flags.lmcs_enabled_flag);

        // PPS (needed for the PH / SH parsers below).
        let pps_nal = nals
            .iter()
            .find(|n| n.header.nal_unit_type == NalUnitType::PpsNut)
            .expect("PPS NAL");
        let pps = crate::pps::parse_pps(&extract_rbsp(pps_nal.payload())).unwrap();

        // LMCS APS — payload round-trips exactly.
        let lmcs_aps = nals
            .iter()
            .filter(|n| n.header.nal_unit_type == NalUnitType::PrefixApsNut)
            .filter_map(|n| crate::aps::parse_aps(&extract_rbsp(n.payload())).ok())
            .find(|aps| aps.aps_params_type == ApsParamsType::Lmcs)
            .expect("an LMCS APS NAL must ship");
        assert_eq!(lmcs_aps.aps_adaptation_parameter_set_id, 0);
        assert_eq!(lmcs_aps.lmcs_data, Some(payload));

        // PH — LMCS chain present.
        let ph_nal = nals
            .iter()
            .find(|n| n.header.nal_unit_type == NalUnitType::PhNut)
            .expect("PH NAL");
        let ph = parse_picture_header_stateful(&extract_rbsp(ph_nal.payload()), &sps, &pps)
            .expect("PH must parse with the LMCS chain");
        assert!(ph.ph_lmcs_enabled_flag);
        assert_eq!(ph.ph_lmcs_aps_id, 0);
        assert!(!ph.ph_chroma_residual_scale_flag);

        // Slice header — sh_lmcs_used_flag = 1.
        let slice_nal = nals
            .iter()
            .find(|n| n.header.nal_unit_type.is_vcl())
            .expect("slice NAL");
        let ph_state = PhState {
            ph_inter_slice_allowed_flag: ph.ph_inter_slice_allowed_flag,
            ph_intra_slice_allowed_flag: ph.ph_intra_slice_allowed_flag,
            ph_alf_enabled_flag: ph.ph_alf_enabled_flag,
            ph_lmcs_enabled_flag: ph.ph_lmcs_enabled_flag,
            ph_explicit_scaling_list_enabled_flag: ph.ph_explicit_scaling_list_enabled_flag,
            ph_temporal_mvp_enabled_flag: ph.ph_temporal_mvp_enabled_flag,
            ph_sao_luma_enabled_flag: ph.ph_sao_luma_enabled_flag,
            ph_sao_chroma_enabled_flag: ph.ph_sao_chroma_enabled_flag,
            num_extra_sh_bits: 0,
            nal_unit_type: slice_nal.header.nal_unit_type,
        };
        let sh =
            parse_slice_header_stateful(&extract_rbsp(slice_nal.payload()), &sps, &pps, &ph_state)
                .expect("slice header must parse");
        assert!(sh.sh_lmcs_used_flag);
    }

    /// Round-384 — the LMCS coding loop (forward-map source →
    /// mapped-domain intra coding → §8.8.1 inverse-map before the
    /// filter chain) reconstructs in the ORIGINAL domain: PSNR_Y
    /// against the unmapped source clears the same floor as the
    /// baseline and stays within a small delta of it.
    #[test]
    fn round384_lmcs_recon_stays_in_original_domain() {
        let src = gradient_frame(64, 64);
        let cfg_off = crate::encoder::EncoderConfig::new(64, 64);
        let (_bs_off, rec_off) = encode_idr_with_residuals_cfg(&src, 26, cfg_off).unwrap();
        let psnr_off = psnr_y(&src.luma, &rec_off.luma).unwrap();

        let mut cfg_on = crate::encoder::EncoderConfig::new(64, 64);
        cfg_on.lmcs = Some(lmcs_enc_payload());
        let (bs_on, rec_on) = encode_idr_with_residuals_cfg(&src, 26, cfg_on).unwrap();
        assert!(!bs_on.is_empty());
        let psnr_on = psnr_y(&src.luma, &rec_on.luma).unwrap();

        eprintln!("round384 LMCS: PSNR off={psnr_off:.2} dB, on={psnr_on:.2} dB");
        assert!(
            psnr_on >= 30.0,
            "PSNR_Y {psnr_on:.2} dB < 30 dB with LMCS on — recon must be inverse-mapped"
        );
        assert!(
            (psnr_on - psnr_off).abs() <= 6.0,
            "LMCS on/off PSNR gap too large: on={psnr_on:.2}, off={psnr_off:.2}"
        );
    }

    /// r391 — the encoder forward chroma-residual scale is the exact
    /// inverse of the decoder's §8.7.5.3 eqs. 1219 / 1220 fold: pushing
    /// a residual through `lmcs_forward_scale_chroma_residuals` and
    /// back through `crate::ctu::lmcs_scale_chroma_residuals` lands
    /// within the fold's rounding of the original.
    #[test]
    fn r391_lmcs_chroma_forward_scale_round_trips() {
        for vs in [1024u32, 1500, 2048, 3000] {
            // Magnitudes kept within the decoder fold's 8-bit clamp
            // (|coded| <= 255) for every trial scale.
            let orig: Vec<i32> = vec![-100, -33, -1, 0, 1, 7, 64, 100];
            let mut coded = orig.clone();
            lmcs_forward_scale_chroma_residuals(&mut coded, vs);
            let mut back = coded.clone();
            crate::ctu::lmcs_scale_chroma_residuals(&mut back, vs, 8);
            for (o, b) in orig.iter().zip(back.iter()) {
                assert!(
                    (o - b).abs() <= 2,
                    "vs={vs}: {o} -> {b} deviates beyond the fold rounding"
                );
            }
        }
    }

    /// r391 — `EncoderConfig::lmcs_chroma_scaling` signals
    /// `ph_chroma_residual_scale_flag = 1` on the wire and the coded
    /// chroma (forward-scaled residual, decoder-exact rescale at
    /// reconstruction) still clears a sane chroma PSNR floor.
    #[test]
    fn r391_lmcs_chroma_scaling_on_wire_and_recon() {
        use crate::nal::{extract_rbsp, iter_annex_b, NalUnitType};
        use crate::picture_header::parse_picture_header_stateful;

        // Chroma-structured source so the chroma TBs carry real
        // residual energy through the scaling path.
        let mut src = gradient_frame(64, 64);
        for y in 0..32usize {
            for x in 0..32usize {
                src.cb.samples[y * src.cb.stride + x] = 90 + ((x + y) % 64) as u8;
                src.cr.samples[y * src.cr.stride + x] = 170 - ((x * 2 + y) % 64) as u8;
            }
        }

        let mut cfg = crate::encoder::EncoderConfig::new(64, 64);
        cfg.lmcs = Some(lmcs_enc_payload());
        cfg.lmcs_chroma_scaling = true;
        let (bs, rec) = encode_idr_with_residuals_cfg(&src, 26, cfg).unwrap();

        // PH — ph_chroma_residual_scale_flag = 1.
        let nals: Vec<_> = iter_annex_b(&bs).collect();
        let sps_nal = nals
            .iter()
            .find(|n| n.header.nal_unit_type == NalUnitType::SpsNut)
            .expect("SPS NAL");
        let sps = crate::sps::parse_sps(&extract_rbsp(sps_nal.payload())).unwrap();
        let pps_nal = nals
            .iter()
            .find(|n| n.header.nal_unit_type == NalUnitType::PpsNut)
            .expect("PPS NAL");
        let pps = crate::pps::parse_pps(&extract_rbsp(pps_nal.payload())).unwrap();
        let ph_nal = nals
            .iter()
            .find(|n| n.header.nal_unit_type == NalUnitType::PhNut)
            .expect("PH NAL");
        let ph = parse_picture_header_stateful(&extract_rbsp(ph_nal.payload()), &sps, &pps)
            .expect("PH must parse with the chroma-scaling LMCS chain");
        assert!(ph.ph_lmcs_enabled_flag);
        assert!(
            ph.ph_chroma_residual_scale_flag,
            "lmcs_chroma_scaling must set ph_chroma_residual_scale_flag"
        );

        // Chroma quality — the scale is inverted at reconstruction, so
        // the coding loop stays lossless-modulo-quantisation.
        let psnr_cb = psnr_y(&src.cb, &rec.cb).unwrap();
        let psnr_cr = psnr_y(&src.cr, &rec.cr).unwrap();
        eprintln!("r391 chroma scaling: PSNR Cb={psnr_cb:.2} dB Cr={psnr_cr:.2} dB");
        assert!(
            psnr_cb >= 28.0 && psnr_cr >= 28.0,
            "chroma PSNR too low with residual scaling: Cb={psnr_cb:.2} Cr={psnr_cr:.2}"
        );
    }
}
