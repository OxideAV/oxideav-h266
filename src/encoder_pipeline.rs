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
//! 2. **In-loop filters:** deblock (§8.8.3) → SAO RDO + apply (§8.8.2)
//!    → luma ALF RDO (§7.4.3.18 fixed-filter sets) → chroma ALF RDO
//!    (§8.8.5.4) → CC-ALF RDO (§8.8.5.7), each pass mutating `rec` and
//!    recording per-CTB decisions into `alf_pic`.
//! 3. **Second pass — CABAC interleave:** for every CTU emit
//!    `encode_alf_ctu` (`alf_ctb_flag[]` / `alf_use_aps_flag` /
//!    `alf_luma_*_idx` / `alf_ctb_filter_alt_idx[]` /
//!    `alf_ctb_cc_*_idc[]`) followed by `emit_luma_tb_residual` for
//!    every TB followed by `encode_terminate(0)` (or `(1)` on the last
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
//!   the first encoder round lean).
//! * Deblocking uses the existing [`crate::deblock`] primitives wired to
//!   a constant `QP = 26`.

use oxideav_core::{Error, Result};

use crate::deblock::{apply_deblocking, DeblockCu, DeblockParams};
use crate::dequant::{dequantize_tb_flat, DequantParams};
use crate::reconstruct::{PictureBuffer, PicturePlane};
use crate::residual::ResidualCtxs;
use crate::residual_enc::encode_tb_coefficients;
use crate::sao::SaoConfig;
use crate::sao_enc::sao_decide_picture;
use crate::transform::{inverse_transform_2d, TrType};
use crate::transform_fwd::{forward_dct_ii_2d, quantize_tb_flat};

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

/// Persisted per-TB state from the round-46 reconstruction pass —
/// re-used by the second-pass CABAC emit (per-CTU ALF + residual
/// interleave) so the encoder doesn't have to redo forward DCT /
/// quantisation work.
struct PreparedLumaTb {
    /// Square TB side length in luma samples.
    n_tb: usize,
    /// Quantised level array (length = `n_tb * n_tb`).
    levels: Vec<i32>,
}

/// Compute residual + reconstruction for one luma TB without touching
/// any CABAC stream. Returns the quantised levels for the later
/// per-CTU CABAC emit pass.
fn prepare_luma_tb(
    src: &PicturePlane,
    rec_plane: &mut PicturePlane,
    x: usize,
    y: usize,
    n_tb: usize,
    qp: i32,
) -> Result<Vec<i32>> {
    // DC intra prediction: flat fill with mid-grey (128).
    let pred_val = 128u8;

    // Extract source block → residual.
    let mut residual = vec![0i32; n_tb * n_tb];
    for ty in 0..n_tb {
        for tx in 0..n_tb {
            let sx = x + tx;
            let sy = y + ty;
            let src_s = src.get(sx, sy).unwrap_or(pred_val) as i32;
            residual[ty * n_tb + tx] = src_s - pred_val as i32;
        }
    }

    // Forward DCT-II.
    let coeffs = forward_dct_ii_2d(n_tb, n_tb, &residual, 8)?;

    // Quantise.
    let levels = quantize_tb_flat(&coeffs, n_tb as u32, n_tb as u32, qp, 8, 15)?;

    // Dequant + IDCT to get decoded residual.
    let params = DequantParams::luma_8bit(n_tb as u32, n_tb as u32, qp);
    let dq = dequantize_tb_flat(&levels, &params)?;
    let rec_res = inverse_transform_2d(
        n_tb,
        n_tb,
        n_tb,
        n_tb,
        TrType::DctII,
        TrType::DctII,
        &dq,
        8,
        15,
    )?;

    // Write reconstructed samples.
    for ty in 0..n_tb {
        for tx in 0..n_tb {
            let rx = x + tx;
            let ry = y + ty;
            if rx < rec_plane.width && ry < rec_plane.height {
                let s = pred_val as i32 + rec_res[ty * n_tb + tx];
                rec_plane.samples[ry * rec_plane.stride + rx] = s.clamp(0, 255) as u8;
            }
        }
    }
    Ok(levels)
}

/// Round-46 — emit the residual CABAC bins for one prepared luma TB.
/// CBF is implicit (only emit coefficients when a non-zero level
/// exists); the spec's `tu_y_coded_flag` is gated outside of this helper
/// in the foundation IDR pipeline.
fn emit_luma_tb_residual(
    enc: &mut crate::cabac_enc::ArithEncoder,
    ctxs: &mut ResidualCtxs,
    tb: &PreparedLumaTb,
) -> Result<()> {
    let has_nonzero = tb.levels.iter().any(|&l| l != 0);
    if has_nonzero {
        encode_tb_coefficients(enc, ctxs, tb.n_tb, tb.n_tb, 0, &tb.levels)?;
    }
    Ok(())
}

/// Encode a complete IDR frame from `src` with residual coding.
///
/// Returns `(bitstream, reconstructed_frame)`. The caller can compute
/// PSNR_Y via [`psnr_y`].
///
/// `qp` — luma quantisation parameter (0..=51).
pub fn encode_idr_with_residuals(src: &PictureBuffer, qp: i32) -> Result<(Vec<u8>, PictureBuffer)> {
    use crate::cabac_enc::ArithEncoder;
    use crate::encoder::{BitWriter, EncoderConfig, VvcEncoder};
    use crate::nal::NalUnitType;

    let w = src.luma.width as u32;
    let h = src.luma.height as u32;

    let config = EncoderConfig::new(w, h);
    let vvc_enc = VvcEncoder::new(config)?;

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

    // Round-45 — emit the synthesised chroma + CC-ALF APSes ahead of
    // the picture header so the decoder can resolve sh_alf_aps_id_chroma
    // / sh_alf_cc_*_aps_id at slice-parse time. APS NUTs use type 17
    // (prefix); both APSes share the chroma payload via
    // `aps_chroma_present_flag = 1`.
    //
    // Round-47 — the luma APS at id 2 (referenced by
    // `ph_alf_aps_id_luma[0] = 2`) is designed AFTER the post-SAO
    // reconstruction so the §8.8.5.2 design pass sees the same pixels
    // the apply pass will read. We therefore defer all NAL emission
    // until after the in-loop-filter chain has run; the order on the
    // wire will still be `VPS, SPS, PPS, APS(0), APS(1), APS(2), PH,
    // slice` per §7.4.2.4 (and the round-47 PH carries
    // `ph_num_alf_aps_ids_luma = 1` referencing APS id 2).
    let chroma_alf_aps = build_chroma_alf_aps();
    let cc_alf_aps = build_cc_alf_aps();
    let chroma_aps_rbsp = crate::aps_enc::emit_alf_aps_rbsp(0, true, &chroma_alf_aps)?;
    let cc_aps_rbsp = crate::aps_enc::emit_alf_aps_rbsp(1, true, &cc_alf_aps)?;
    let chroma_aps_nal = build_nal(NalUnitType::PrefixApsNut, 0, 1, &chroma_aps_rbsp);
    let cc_aps_nal = build_nal(NalUnitType::PrefixApsNut, 0, 1, &cc_aps_rbsp);
    annex_b(&mut bitstream, &chroma_aps_nal);
    annex_b(&mut bitstream, &cc_aps_nal);

    // --- Build the slice RBSP ---
    let mut bw = BitWriter::new();
    // Slice header: sh_picture_header_in_slice_header_flag = 0,
    //               sh_no_output_of_prior_pics_flag = 0 (IDR).
    bw.write_bit(0); // sh_picture_header_in_slice_header_flag
    bw.write_bit(0); // sh_no_output_of_prior_pics_flag
                     // byte_alignment() — 1-stop-bit + pad to byte.
    bw.byte_alignment();
    // CABAC engine starts here.
    let slice_header_bytes = bw.into_bytes();

    // CABAC-encode every CTU.
    let ctb_size: usize = 128; // matches encoder SPS (log2_ctu_size_minus5 = 2 → 128)
    let pic_w_ctbs = (w as usize + ctb_size - 1) / ctb_size;
    let pic_h_ctbs = (h as usize + ctb_size - 1) / ctb_size;

    let mut rec = PictureBuffer::yuv420_filled(w as usize, h as usize, 128);
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
    let mut prepared_per_ctu: Vec<Vec<Vec<PreparedLumaTb>>> = (0..pic_h_ctbs)
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

                    let levels =
                        prepare_luma_tb(&src.luma, &mut rec.luma, tb_x, tb_y, n_tb_sq, qp)?;
                    prepared_per_ctu[ry][rx].push(PreparedLumaTb {
                        n_tb: n_tb_sq,
                        levels,
                    });

                    // Chroma: flat fill (mid-grey, no residual for now).
                    let chr_x = tb_x / 2;
                    let chr_y = tb_y / 2;
                    let chr_n = n_tb_sq / 2;
                    for py in 0..chr_n {
                        for px in 0..chr_n {
                            if chr_x + px < rec.cb.width && chr_y + py < rec.cb.height {
                                // Predict chroma from source.
                                if let Some(s) = src.cb.get(chr_x + px, chr_y + py) {
                                    rec.cb.samples[(chr_y + py) * rec.cb.stride + chr_x + px] = s;
                                }
                                if let Some(s) = src.cr.get(chr_x + px, chr_y + py) {
                                    rec.cr.samples[(chr_y + py) * rec.cr.stride + chr_x + px] = s;
                                }
                            }
                        }
                    }

                    // Accumulate deblock CU info.
                    all_deblock_cus.push(DeblockCu {
                        x: tb_x as u32,
                        y: tb_y as u32,
                        w: n_tb_sq as u32,
                        h: n_tb_sq as u32,
                        qp_y: qp,
                        intra: true,
                        tu_y_coded: true,
                        tu_cb_coded: false,
                        tu_cr_coded: false,
                        bdpcm_luma: false,
                        bdpcm_chroma: false,
                    });
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
        ..Default::default()
    };
    apply_deblocking(&mut rec, &all_deblock_cus, &dbp, 1);

    // SAO decision + apply (luma only for this round).
    let sao_pic = sao_decide_picture(src, &rec, 7, 8, true, false);
    let sao_cfg = SaoConfig {
        luma_used: true,
        chroma_used: false,
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
    // Round-47 — design + emit a luma ALF APS at id 2. The
    // [`crate::alf_aps_design::design_luma_alf_filter`] pass solves the
    // 12×12 normal-equations system over the post-SAO recon vs. the
    // source for a single signalled filter row that minimises picture-
    // wide squared error in the eq. 1449 / 1450 luma diamond. The
    // resulting [`AlfApsData`] feeds (a) `emit_alf_aps_rbsp` so the
    // decoder side can resolve `AlfCtbFiltSetIdxY = 16` to the
    // designed coefficients, and (b) `alf_decide_and_apply_with_aps`
    // so the per-CTB RDO compares the learned filter against the 16
    // fixed sets and picks the lower-SSE option. Per §7.4.2.4 the APS
    // NAL must precede the picture header that references it; we emit
    // it here, then the picture header, then the slice.
    let luma_alf_aps = {
        let designed = crate::alf_aps_design::design_luma_alf_filter(src, &rec);
        crate::alf_aps_design::build_luma_alf_aps_data(&designed)
    };
    let luma_aps_rbsp = crate::aps_enc::emit_alf_aps_rbsp(2, false, &luma_alf_aps)?;
    let luma_aps_nal = build_nal(NalUnitType::PrefixApsNut, 0, 1, &luma_aps_rbsp);
    annex_b(&mut bitstream, &luma_aps_nal);

    // Picture header — must come AFTER the APSes it references
    // (§7.4.2.4). Round-47 PH carries `ph_num_alf_aps_ids_luma = 1` /
    // `ph_alf_aps_id_luma[0] = 2` so the slice CABAC walk's
    // `alf_use_aps_flag = 1` branch resolves to the freshly emitted
    // APS at id 2.
    annex_b(
        &mut bitstream,
        &vvc_enc.emit_nal(EmittedNalKind::PictureHeader)?,
    );

    let pre_luma_alf_samples = rec.luma.samples.clone();
    let mut alf_pic =
        crate::alf_enc::alf_decide_and_apply_with_aps(src, &mut rec, &luma_alf_aps, 7, 8, 1);

    crate::alf_enc::chroma_alf_decide_and_apply(
        src,
        &mut rec,
        &mut alf_pic,
        &chroma_alf_aps,
        crate::alf_enc::CcAlfComponent::Cb,
        7,
        8,
        1,
    );
    crate::alf_enc::chroma_alf_decide_and_apply(
        src,
        &mut rec,
        &mut alf_pic,
        &chroma_alf_aps,
        crate::alf_enc::CcAlfComponent::Cr,
        7,
        8,
        1,
    );

    crate::alf_enc::cc_alf_decide_and_apply(
        src,
        &mut rec,
        &pre_luma_alf_samples,
        &mut alf_pic,
        &cc_alf_aps,
        crate::alf_enc::CcAlfComponent::Cb,
        7,
        8,
        1,
    );
    crate::alf_enc::cc_alf_decide_and_apply(
        src,
        &mut rec,
        &pre_luma_alf_samples,
        &mut alf_pic,
        &cc_alf_aps,
        crate::alf_enc::CcAlfComponent::Cr,
        7,
        8,
        1,
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
    // Round-47 — `sh_num_alf_aps_ids_luma = 1` matches the PH-level
    // chain (`ph_num_alf_aps_ids_luma = 1`). The per-CTU
    // `alf_luma_prev_filter_idx` field uses cMax = N - 1 = 0 (so the
    // field is suppressed and `prev_idx` is inferred to 0), which is
    // the round-47 behaviour we want — every "use APS" CTB resolves to
    // APS slot 0 (= APS id 2).
    let alf_cfg = crate::alf_syntax::AlfSyntaxConfig {
        alf_enabled: true,
        cb_enabled: true,
        cr_enabled: true,
        cc_cb_enabled: true,
        cc_cr_enabled: true,
        sh_num_alf_aps_ids_luma: 1,
        alf_chroma_num_alt_filters_minus1: chroma_alf_aps.alf_chroma_num_alt_filters_minus1 as u8,
        alf_cc_cb_filters_signalled_minus1: cc_alf_aps.cc_cb_coeff.len().saturating_sub(1) as u8,
        alf_cc_cr_filters_signalled_minus1: cc_alf_aps.cc_cr_coeff.len().saturating_sub(1) as u8,
        chroma_format_idc: 1,
        slice_type: crate::slice_header::SliceType::I,
        sh_cabac_init_flag: false,
    };

    let mut cabac_enc = ArithEncoder::new();
    let mut alf_ctxs = crate::alf_syntax::AlfCtxs::init(26);
    let mut residual_ctxs = ResidualCtxs::init(26);

    for ry in 0..pic_h_ctbs {
        for rx in 0..pic_w_ctbs {
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

            // Residual CABAC bins for every TB inside this CTU.
            for tb in &prepared_per_ctu[ry][rx] {
                emit_luma_tb_residual(&mut cabac_enc, &mut residual_ctxs, tb)?;
            }

            // CABAC end_of_slice_segment_flag for CTU termination.
            let is_last_ctu = ry == pic_h_ctbs - 1 && rx == pic_w_ctbs - 1;
            cabac_enc.encode_terminate(if is_last_ctu { 1 } else { 0 })?;
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
}
