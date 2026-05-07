//! Encoder-side ALF RDO — round-40.
//!
//! This module wraps the decode-side [`crate::alf::apply_alf`] pipeline
//! in a per-CTB on/off RDO loop. The encoder picks
//! `AlfCtbFiltSetIdxY ∈ 0..16` (a §7.4.3.18 fixed filter set; no APS
//! signalling required) when the post-ALF SSE_Y is lower than the
//! pre-ALF SSE_Y, and leaves luma ALF off otherwise. Chroma ALF +
//! CC-ALF are not exercised by this round — both `cb_enabled` and
//! `cr_enabled` stay false, and the per-CTB chroma slots / CC-IDC
//! arrays are zero.
//!
//! ## Why fixed filter sets only
//!
//! VVC's full-fat ALF requires APS signalling — the encoder side has
//! to derive 25 per-class 12-tap filters from the picture statistics,
//! quantise them with the spec's piecewise-linear lattice, and emit
//! the `alf_data()` syntax. That entire pipeline (Wiener-filter +
//! lattice quant + APS bin emit) is a multi-round project in its own
//! right.
//!
//! The 16 §7.4.3.18 *fixed filter sets* (eqs. 90 / 91 + the
//! `AlfFixFiltCoeff` table) are pre-defined linear filters that the
//! decoder applies without any APS dependency. They give the encoder
//! a free per-CTB on/off knob: try one of the 16 fixed sets, measure
//! the SSE delta against the source frame, and pick the lower-SSE
//! option per CTB. Round-40 picks fixed set 0 (the spec's "smooth
//! area" candidate); evaluating multiple fixed sets per CTB is a
//! straightforward loop extension.

use crate::alf::{apply_alf, AlfApsBinding, AlfConfig, AlfCtb, AlfPicture};
use crate::reconstruct::{PictureBuffer, PicturePlane};

/// Per-CTB SSE summed over the luma rectangle covering this CTB. Used
/// by the on/off RDO.
fn ctb_sse_y(
    src: &PicturePlane,
    rec: &PicturePlane,
    x0: usize,
    y0: usize,
    w: usize,
    h: usize,
) -> u64 {
    let mut sse: u64 = 0;
    for y in 0..h {
        let yi = y0 + y;
        if yi >= src.height || yi >= rec.height {
            break;
        }
        for x in 0..w {
            let xi = x0 + x;
            if xi >= src.width || xi >= rec.width {
                break;
            }
            let s = src.samples[yi * src.stride + xi] as i32;
            let r = rec.samples[yi * rec.stride + xi] as i32;
            let d = (s - r) as i64;
            sse += (d * d) as u64;
        }
    }
    sse
}

/// Build a fully-on AlfPicture using fixed filter set 0 (the §7.4.3.18
/// "smooth area" table). Used by [`alf_decide_picture`] as the trial
/// candidate against which per-CTB SSE deltas are measured.
fn alf_picture_all_on_fixed_set0(pic_w_in_ctbs: u32, pic_h_in_ctbs: u32) -> AlfPicture {
    let mut pic = AlfPicture::empty(pic_w_in_ctbs, pic_h_in_ctbs);
    for ry in 0..pic_h_in_ctbs {
        for rx in 0..pic_w_in_ctbs {
            pic.set(
                rx,
                ry,
                AlfCtb {
                    luma_on: true,
                    luma_filt_set_idx: 0, // fixed filter set 0
                    cb_on: false,
                    cr_on: false,
                    cb_alt_idx: 0,
                    cr_alt_idx: 0,
                    cc_cb_idc: 0,
                    cc_cr_idc: 0,
                },
            );
        }
    }
    pic
}

/// Encoder ALF on/off RDO + apply.
///
/// Inputs:
/// * `src` — original (uncompressed) picture, used for the per-CTB SSE
///   distortion metric.
/// * `rec` — post-deblock + post-SAO reconstructed picture; mutated
///   in place when ALF is judged to lower SSE_Y.
/// * `ctb_log2_size_y` — CtbLog2SizeY (typically 7 for 128×128 CTUs).
/// * `bit_depth` — luma bit depth (8 for the round-40 scaffold).
/// * `chroma_format_idc` — 0 / 1 / 2 / 3 per §7.4.3.3.
///
/// Returns the [`AlfPicture`] describing the per-CTB on/off decisions
/// (so the caller can emit the matching slice-header / per-CTB CABAC
/// syntax in a follow-up round).
pub fn alf_decide_and_apply(
    src: &PictureBuffer,
    rec: &mut PictureBuffer,
    ctb_log2_size_y: u32,
    bit_depth: u32,
    chroma_format_idc: u32,
) -> AlfPicture {
    let ctb_size_y = 1u32 << ctb_log2_size_y;
    let pic_w_in_ctbs = (rec.luma.width as u32).div_ceil(ctb_size_y);
    let pic_h_in_ctbs = (rec.luma.height as u32).div_ceil(ctb_size_y);

    // Stage 1: collect per-CTB SSE_Y on the *current* `rec` (pre-ALF).
    let pre_alf_sse = {
        let mut tab = vec![0u64; (pic_w_in_ctbs * pic_h_in_ctbs) as usize];
        for ry in 0..pic_h_in_ctbs {
            for rx in 0..pic_w_in_ctbs {
                let x0 = (rx * ctb_size_y) as usize;
                let y0 = (ry * ctb_size_y) as usize;
                tab[(ry * pic_w_in_ctbs + rx) as usize] = ctb_sse_y(
                    &src.luma,
                    &rec.luma,
                    x0,
                    y0,
                    ctb_size_y as usize,
                    ctb_size_y as usize,
                );
            }
        }
        tab
    };

    // Stage 2: clone `rec` and apply ALF picture-wide with fixed filter
    // set 0 (the spec's `AlfCtbFiltSetIdxY = 0` row of `AlfClassToFiltMap`).
    let mut rec_with_alf = rec.clone();
    let trial_pic = alf_picture_all_on_fixed_set0(pic_w_in_ctbs, pic_h_in_ctbs);
    let cfg = AlfConfig {
        alf_enabled: true,
        cb_enabled: false,
        cr_enabled: false,
        bit_depth,
        ctb_log2_size_y,
        chroma_format_idc,
    };
    apply_alf(
        &mut rec_with_alf,
        &trial_pic,
        &cfg,
        &AlfApsBinding::default(),
    );

    // Stage 3: per-CTB on/off — pick the post-ALF copy when its SSE_Y
    // is strictly lower than the pre-ALF SSE_Y. The choice is recorded
    // in `final_pic` so the bitstream emit (future round) can mirror it.
    let mut final_pic = AlfPicture::empty(pic_w_in_ctbs, pic_h_in_ctbs);
    for ry in 0..pic_h_in_ctbs {
        for rx in 0..pic_w_in_ctbs {
            let x0 = (rx * ctb_size_y) as usize;
            let y0 = (ry * ctb_size_y) as usize;
            let post = ctb_sse_y(
                &src.luma,
                &rec_with_alf.luma,
                x0,
                y0,
                ctb_size_y as usize,
                ctb_size_y as usize,
            );
            let pre = pre_alf_sse[(ry * pic_w_in_ctbs + rx) as usize];
            if post < pre {
                // Commit ALF for this CTB — copy the post-ALF luma
                // samples back into `rec`.
                let w = (ctb_size_y as usize).min(rec.luma.width.saturating_sub(x0));
                let h = (ctb_size_y as usize).min(rec.luma.height.saturating_sub(y0));
                for y in 0..h {
                    let yi = y0 + y;
                    let dst = &mut rec.luma.samples
                        [yi * rec.luma.stride + x0..yi * rec.luma.stride + x0 + w];
                    let src_alf = &rec_with_alf.luma.samples[yi * rec_with_alf.luma.stride + x0
                        ..yi * rec_with_alf.luma.stride + x0 + w];
                    dst.copy_from_slice(src_alf);
                }
                final_pic.set(
                    rx,
                    ry,
                    AlfCtb {
                        luma_on: true,
                        luma_filt_set_idx: 0,
                        ..Default::default()
                    },
                );
            }
        }
    }
    final_pic
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reconstruct::PictureBuffer;

    fn smooth_gradient_picture(w: usize, h: usize) -> PictureBuffer {
        let mut buf = PictureBuffer::yuv420_filled(w, h, 128);
        for y in 0..h {
            for x in 0..w {
                let v = (32 + (x as f32 / w.max(1) as f32 * 191.0) as i32) as u8;
                buf.luma.samples[y * buf.luma.stride + x] = v;
            }
        }
        buf
    }

    #[test]
    fn alf_rdo_returns_picture_geometry() {
        let src = smooth_gradient_picture(128, 128);
        let mut rec = src.clone();
        // Inject a small distortion so RDO has a reason to flip CTB
        // entries on (a flat-source rec leaves every CTB pre-ALF SSE = 0
        // and ALF can only hurt or break-even).
        rec.luma.samples[0] ^= 0x10;
        let pic = alf_decide_and_apply(&src, &mut rec, 7, 8, 1);
        assert_eq!(pic.pic_width_in_ctbs_y, 1);
        assert_eq!(pic.pic_height_in_ctbs_y, 1);
    }

    #[test]
    fn alf_rdo_on_flat_source_does_not_hurt_psnr() {
        // With a flat-grey source, ALF can only hurt; the RDO must
        // choose "off" everywhere.
        let src = PictureBuffer::yuv420_filled(128, 128, 128);
        let mut rec = src.clone();
        // Distort one sample so the rec differs from src and ALF could
        // theoretically run.
        rec.luma.samples[10] ^= 0x40;
        let pre_sse: u64 = src
            .luma
            .samples
            .iter()
            .zip(rec.luma.samples.iter())
            .map(|(s, r)| {
                let d = *s as i64 - *r as i64;
                (d * d) as u64
            })
            .sum();
        alf_decide_and_apply(&src, &mut rec, 7, 8, 1);
        let post_sse: u64 = src
            .luma
            .samples
            .iter()
            .zip(rec.luma.samples.iter())
            .map(|(s, r)| {
                let d = *s as i64 - *r as i64;
                (d * d) as u64
            })
            .sum();
        // RDO is monotone-improving by construction (any CTB whose
        // post-ALF SSE >= pre-ALF SSE stays off).
        assert!(post_sse <= pre_sse, "ALF RDO must never increase SSE_Y");
    }
}
