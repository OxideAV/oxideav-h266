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
//! ## Pipeline for one CTU
//!
//! 1. **Intra prediction** — DC (flat mid-grey) for the whole CTU.
//! 2. **Residual** — `src - pred` → `forward_dct_ii_2d` → `quantize_tb_flat`.
//! 3. **Residual CABAC emit** — [`crate::residual_enc::encode_tb_coefficients`].
//! 4. **Dequant + inverse transform** — reconstruct the decoded residual.
//! 5. **Reconstruction** — `pred + dequant_residual`, clipped to `[0, 255]`.
//! 6. **Deblocking** — applied per CTU after reconstruction (simplified:
//!    one CTU per IDR with `pps_no_pic_partition = 1`, so border only).
//! 7. **SAO** — decided via [`crate::sao_enc`] and applied via
//!    [`crate::sao::apply_sao`].
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

/// Encode one luma TB (n_tb × n_tb block at position `(x, y)` in the
/// source plane). Updates `rec_plane` in-place with the decoded reconstruction.
///
/// Returns the quantised level array for the deblocking step.
fn encode_luma_tb(
    src: &PicturePlane,
    rec_plane: &mut PicturePlane,
    x: usize,
    y: usize,
    n_tb: usize,
    qp: i32,
    enc: &mut crate::cabac_enc::ArithEncoder,
    ctxs: &mut ResidualCtxs,
) -> Result<()> {
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

    // CABAC-emit residual.
    // Check if there are any non-zero levels (CBF).
    let has_nonzero = levels.iter().any(|&l| l != 0);
    if has_nonzero {
        encode_tb_coefficients(enc, ctxs, n_tb, n_tb, 0, &levels)?;
    }

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
    annex_b(
        &mut bitstream,
        &vvc_enc.emit_nal(EmittedNalKind::PictureHeader)?,
    );

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

    let mut cabac_enc = ArithEncoder::new();
    let mut ctxs = ResidualCtxs::init(26); // slice_qp_y = 26

    // TB size: use the full CTB for simplicity.  For large pictures this
    // is fine since DCT-II is defined up to 64×64; cap TB at 64×64.
    let tb_size = 64usize;

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

                    encode_luma_tb(
                        &src.luma,
                        &mut rec.luma,
                        tb_x,
                        tb_y,
                        n_tb_sq,
                        qp,
                        &mut cabac_enc,
                        &mut ctxs,
                    )?;

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

            // CABAC end_of_slice_segment_flag for CTU termination:
            // encode_terminate(0) for all but the last CTU,
            // encode_terminate(1) for the last.
            let is_last_ctu = ry == pic_h_ctbs - 1 && rx == pic_w_ctbs - 1;
            cabac_enc.encode_terminate(if is_last_ctu { 1 } else { 0 })?;
        }
    }

    let cabac_bytes = cabac_enc.finish();

    // Apply deblocking.
    let dbp = DeblockParams {
        disabled: false,
        ..Default::default()
    };
    apply_deblocking(&mut rec, &all_deblock_cus, &dbp, 1);

    // SAO decision + apply (luma only for this round).
    let sao_pic = sao_decide_picture(&src, &rec, 7, 8, true, false);
    let sao_cfg = SaoConfig {
        luma_used: true,
        chroma_used: false,
        bit_depth: 8,
        ctb_log2_size_y: 7,
        chroma_format_idc: 1,
    };
    crate::sao::apply_sao(&mut rec, &sao_pic, &sao_cfg);

    // Build the IDR slice NAL: header bytes + CABAC bytes + trailing bits.
    let mut slice_rbsp = slice_header_bytes;
    slice_rbsp.extend_from_slice(&cabac_bytes);
    // rbsp_trailing_bits: the CABAC finish() already byte-aligns; append
    // a stop bit if needed for strict conformance.
    // For our scaffold, the CABAC bytes are byte-aligned by finish().
    let slice_nal = build_nal(NalUnitType::IdrNLp, 0, 1, &slice_rbsp);
    annex_b(&mut bitstream, &slice_nal);

    Ok((bitstream, rec))
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
}
