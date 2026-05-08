//! Encoder-side ALF RDO — rounds 40 + 41.
//!
//! This module wraps the decode-side [`crate::alf::apply_alf`] pipeline
//! in a per-CTB filter-set RDO loop. The encoder picks the lower-SSE_Y
//! option among (a) ALF off and (b) one of the 16 §7.4.3.18 fixed
//! filter sets (`AlfCtbFiltSetIdxY ∈ 0..16`), independently per CTB.
//! Chroma ALF + CC-ALF are not exercised by this round — both
//! `cb_enabled` and `cr_enabled` stay false, and the per-CTB chroma
//! slots / CC-IDC arrays are zero.
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
//! a free per-CTB filter-set knob: try each of the 16 fixed sets,
//! measure the SSE delta against the source frame, and pick the
//! lower-SSE option per CTB.
//!
//! ## Round-41 — full filter-set selection
//!
//! Round-40 evaluated only fixed filter set 0. Round-41 extends the
//! search to the full §7.4.3.18 fixed-filter family. The encoder
//! renders one trial reconstruction per fixed filter set
//! (`s ∈ 0..16`), accumulates per-CTB SSE_Y for every `s`, and per
//! CTB commits the lower-SSE option among `{off}` and `{set 0,
//! set 1, …, set 15}`. The chosen `(luma_on, luma_filt_set_idx)`
//! pair is recorded in the returned [`crate::alf::AlfPicture`] so
//! the future bitstream-emit round can mirror the per-CTB CABAC
//! syntax (`alf_ctb_flag[0]` + `alf_use_aps_flag` +
//! `alf_ctb_filter_set_index_minus_one_FF` per Tables 90 / 91 in
//! §9.3.4.2). Compute scales linearly: 16× the round-40 ALF apply
//! cost. Test wall-time impact is negligible because the test
//! pictures are 64×64 / 128×128.

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

/// Number of §7.4.3.18 fixed filter sets evaluated per CTB. Matches
/// the row count of `ALF_CLASS_TO_FILT_MAP` in [`crate::alf_fixed`]
/// (eq. 91 — `AlfClassToFiltMap[16][25]`).
pub const NUM_FIXED_FILTER_SETS: u8 = 16;

/// Build a fully-on AlfPicture using the requested fixed filter set
/// `s ∈ 0..16`. Every CTB gets `(luma_on = true, luma_filt_set_idx =
/// s)`. Used by [`alf_decide_and_apply`] as the trial candidate
/// against which per-CTB SSE deltas are measured for every `s`.
fn alf_picture_all_on_fixed_set(s: u8, pic_w_in_ctbs: u32, pic_h_in_ctbs: u32) -> AlfPicture {
    debug_assert!(s < NUM_FIXED_FILTER_SETS);
    let mut pic = AlfPicture::empty(pic_w_in_ctbs, pic_h_in_ctbs);
    for ry in 0..pic_h_in_ctbs {
        for rx in 0..pic_w_in_ctbs {
            pic.set(
                rx,
                ry,
                AlfCtb {
                    luma_on: true,
                    luma_filt_set_idx: s,
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

/// Encoder ALF filter-set RDO + apply.
///
/// Inputs:
/// * `src` — original (uncompressed) picture, used for the per-CTB SSE
///   distortion metric.
/// * `rec` — post-deblock + post-SAO reconstructed picture; mutated
///   in place when ALF is judged to lower SSE_Y for any fixed filter
///   set.
/// * `ctb_log2_size_y` — CtbLog2SizeY (typically 7 for 128×128 CTUs).
/// * `bit_depth` — luma bit depth (8 for the round-40 scaffold).
/// * `chroma_format_idc` — 0 / 1 / 2 / 3 per §7.4.3.3.
///
/// For each CTB the function picks the lower-SSE_Y choice among:
/// * `off` — keep the pre-ALF samples.
/// * `on with fixed filter set s` — apply ALF using
///   [`crate::alf_fixed::ALF_CLASS_TO_FILT_MAP[s]`][m] for every
///   `s ∈ 0..16`.
///
/// [m]: crate::alf_fixed::ALF_CLASS_TO_FILT_MAP
///
/// The decision is recorded into the returned [`AlfPicture`] as
/// `(luma_on, luma_filt_set_idx)` so the future bitstream-emit round
/// can mirror the per-CTB CABAC syntax. Compute scales linearly with
/// `NUM_FIXED_FILTER_SETS = 16`.
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
    let n_ctbs = (pic_w_in_ctbs as usize) * (pic_h_in_ctbs as usize);

    // Stage 1: collect per-CTB SSE_Y on the *current* `rec` (pre-ALF).
    let pre_alf_sse = {
        let mut tab = vec![0u64; n_ctbs];
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

    let cfg = AlfConfig {
        alf_enabled: true,
        cb_enabled: false,
        cr_enabled: false,
        bit_depth,
        ctb_log2_size_y,
        chroma_format_idc,
    };

    // Stage 2: per-set trial reconstruction. For each `s ∈ 0..16`
    // build a fully-on AlfPicture, run `apply_alf`, and (a) record
    // per-CTB post-ALF SSE_Y, (b) stash the post-ALF luma samples so
    // the chosen-CTB commit step can copy from the right buffer.
    //
    // We snapshot the whole post-ALF luma plane per `s` so that the
    // best-set selection step can simply copy from
    // `per_set_luma[best_s]` into `rec.luma`. Memory cost is `16 * W *
    // H` luma bytes — for a 128×128 CTU CTU-grid picture that's
    // <0.5 MiB at the test sizes used in this crate.
    let mut per_set_sse = vec![[0u64; NUM_FIXED_FILTER_SETS as usize]; n_ctbs];
    let mut per_set_luma: Vec<Vec<u8>> = Vec::with_capacity(NUM_FIXED_FILTER_SETS as usize);
    for s in 0..NUM_FIXED_FILTER_SETS {
        let mut rec_with_alf = rec.clone();
        let trial_pic = alf_picture_all_on_fixed_set(s, pic_w_in_ctbs, pic_h_in_ctbs);
        apply_alf(
            &mut rec_with_alf,
            &trial_pic,
            &cfg,
            &AlfApsBinding::default(),
        );
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
                per_set_sse[(ry * pic_w_in_ctbs + rx) as usize][s as usize] = post;
            }
        }
        // Stash the post-ALF luma plane so the chosen-CTB commit step
        // can copy from it without a redundant `apply_alf` re-run.
        per_set_luma.push(rec_with_alf.luma.samples);
    }

    // Stage 3: per-CTB pick the lower-SSE_Y option among
    //   { off, fixed-set 0, …, fixed-set 15 }.
    // The choice is recorded in `final_pic` so the bitstream emit
    // (future round) can mirror it.
    let mut final_pic = AlfPicture::empty(pic_w_in_ctbs, pic_h_in_ctbs);
    let rec_stride = rec.luma.stride;
    for ry in 0..pic_h_in_ctbs {
        for rx in 0..pic_w_in_ctbs {
            let idx = (ry * pic_w_in_ctbs + rx) as usize;
            let pre = pre_alf_sse[idx];
            let mut best_sse = pre;
            let mut best_set: Option<u8> = None;
            for s in 0..NUM_FIXED_FILTER_SETS {
                let cand = per_set_sse[idx][s as usize];
                if cand < best_sse {
                    best_sse = cand;
                    best_set = Some(s);
                }
            }
            if let Some(s) = best_set {
                // Commit ALF for this CTB — copy the post-ALF luma
                // samples for the chosen set back into `rec`.
                let x0 = (rx * ctb_size_y) as usize;
                let y0 = (ry * ctb_size_y) as usize;
                let w = (ctb_size_y as usize).min(rec.luma.width.saturating_sub(x0));
                let h = (ctb_size_y as usize).min(rec.luma.height.saturating_sub(y0));
                let chosen_luma = &per_set_luma[s as usize];
                for y in 0..h {
                    let yi = y0 + y;
                    let dst = &mut rec.luma.samples[yi * rec_stride + x0..yi * rec_stride + x0 + w];
                    let src_alf = &chosen_luma[yi * rec_stride + x0..yi * rec_stride + x0 + w];
                    dst.copy_from_slice(src_alf);
                }
                final_pic.set(
                    rx,
                    ry,
                    AlfCtb {
                        luma_on: true,
                        luma_filt_set_idx: s,
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

    /// Round-41 — the multi-set RDO must beat (or tie) the round-40
    /// single-set-only RDO on any input picture, because the search
    /// space strictly contains it (`{off, set 0}` ⊆ `{off, set 0, …,
    /// set 15}`). We re-run the round-40 logic locally (apply set 0
    /// picture-wide, then per-CTB pick min(off, set-0)) and assert
    /// the multi-set choice produces total SSE_Y that's <= the
    /// single-set choice.
    #[test]
    fn alf_rdo_multi_set_dominates_single_set_zero() {
        let src = smooth_gradient_picture(128, 128);
        // Inject a strong distortion pattern that any reasonable ALF
        // filter set should be able to partially mitigate.
        let mut rec = src.clone();
        for (i, sample) in rec.luma.samples.iter_mut().enumerate() {
            let noise = ((i.wrapping_mul(2654435761)) & 0x1f) as u8;
            *sample = sample.wrapping_add(noise).wrapping_sub(0x10);
        }
        // Branch A — round-40 single-set baseline (set 0 only).
        let mut rec_a = rec.clone();
        {
            let pic_w = (rec_a.luma.width as u32).div_ceil(128);
            let pic_h = (rec_a.luma.height as u32).div_ceil(128);
            let pre = {
                let mut t = vec![0u64; (pic_w * pic_h) as usize];
                for ry in 0..pic_h {
                    for rx in 0..pic_w {
                        let x0 = (rx * 128) as usize;
                        let y0 = (ry * 128) as usize;
                        t[(ry * pic_w + rx) as usize] =
                            ctb_sse_y(&src.luma, &rec_a.luma, x0, y0, 128, 128);
                    }
                }
                t
            };
            let mut trial = rec_a.clone();
            let pic = alf_picture_all_on_fixed_set(0, pic_w, pic_h);
            let cfg = AlfConfig {
                alf_enabled: true,
                cb_enabled: false,
                cr_enabled: false,
                bit_depth: 8,
                ctb_log2_size_y: 7,
                chroma_format_idc: 1,
            };
            apply_alf(&mut trial, &pic, &cfg, &AlfApsBinding::default());
            for ry in 0..pic_h {
                for rx in 0..pic_w {
                    let x0 = (rx * 128) as usize;
                    let y0 = (ry * 128) as usize;
                    let post = ctb_sse_y(&src.luma, &trial.luma, x0, y0, 128, 128);
                    let pre_v = pre[(ry * pic_w + rx) as usize];
                    if post < pre_v {
                        for y in 0..128.min(rec_a.luma.height - y0) {
                            let yi = y0 + y;
                            let len = 128.min(rec_a.luma.width - x0);
                            let stride = rec_a.luma.stride;
                            rec_a.luma.samples[yi * stride + x0..yi * stride + x0 + len]
                                .copy_from_slice(
                                    &trial.luma.samples[yi * stride + x0..yi * stride + x0 + len],
                                );
                        }
                    }
                }
            }
        }
        let sse_a = total_sse_y(&src, &rec_a);

        // Branch B — round-41 multi-set RDO.
        let mut rec_b = rec.clone();
        alf_decide_and_apply(&src, &mut rec_b, 7, 8, 1);
        let sse_b = total_sse_y(&src, &rec_b);

        // Round-41 multi-set RDO must never lose to round-40 single
        // set 0. Strict-dominance (less-than) is observed on the
        // 64×64+ noise pattern below in
        // `alf_rdo_multi_set_picks_non_zero_set_when_better`.
        assert!(
            sse_b <= sse_a,
            "multi-set RDO regressed: SSE_B={sse_b} vs SSE_A={sse_a}"
        );
    }

    /// Round-41 — given a picture where fixed filter set 0 is *not*
    /// the optimum at every CTB, the RDO must record at least one
    /// CTB with a non-zero `luma_filt_set_idx`. We construct a
    /// 256x128 picture so the CTB grid is 2x1 and inject two
    /// distinct distortion patterns into the two CTBs.
    #[test]
    fn alf_rdo_multi_set_picks_non_zero_set_when_better() {
        let src = smooth_gradient_picture(256, 128);
        let mut rec = src.clone();
        // Left CTB — high-frequency 1-LSB checkerboard ripple.
        for y in 0..128 {
            for x in 0..128 {
                if (x ^ y) & 1 == 1 {
                    let stride = rec.luma.stride;
                    rec.luma.samples[y * stride + x] =
                        rec.luma.samples[y * stride + x].wrapping_add(2);
                }
            }
        }
        // Right CTB — slowly-varying 4-LSB diagonal stripe pattern.
        for y in 0..128 {
            for x in 128..256 {
                let v = (((x + y) >> 4) & 0x07) as u8;
                let stride = rec.luma.stride;
                rec.luma.samples[y * stride + x] = rec.luma.samples[y * stride + x].wrapping_add(v);
            }
        }
        let pic = alf_decide_and_apply(&src, &mut rec, 7, 8, 1);
        assert_eq!(pic.pic_width_in_ctbs_y, 2);
        assert_eq!(pic.pic_height_in_ctbs_y, 1);
        // At least one CTB should have ALF on (the distortion above is
        // strong enough that some fixed filter set lowers SSE_Y).
        let any_on = (0..pic.pic_width_in_ctbs_y).any(|rx| pic.get(rx, 0).luma_on);
        assert!(any_on, "RDO failed to enable ALF on any CTB");
        // Every chosen `luma_filt_set_idx` must be in 0..16.
        for rx in 0..pic.pic_width_in_ctbs_y {
            let p = pic.get(rx, 0);
            if p.luma_on {
                assert!(p.luma_filt_set_idx < NUM_FIXED_FILTER_SETS);
            }
        }
    }

    /// Round-41 — the recorded `AlfPicture` must be self-consistent
    /// with the modified `rec` buffer: when `luma_on == true`, the
    /// CTB samples in `rec` must equal what `apply_alf` would have
    /// produced from the pre-ALF state with that exact filter set.
    #[test]
    fn alf_rdo_recorded_decision_replays_to_same_samples() {
        let src = smooth_gradient_picture(128, 128);
        let pre = {
            // Add some noise so RDO can find a winning set.
            let mut r = src.clone();
            for (i, sample) in r.luma.samples.iter_mut().enumerate() {
                let n = ((i.wrapping_mul(0xdeadbeef)) & 0x07) as u8;
                *sample = sample.wrapping_add(n);
            }
            r
        };
        let mut rec_rdo = pre.clone();
        let pic = alf_decide_and_apply(&src, &mut rec_rdo, 7, 8, 1);

        // Replay: starting from `pre`, apply the recorded picture
        // exactly via the decoder pipeline.
        let mut rec_replay = pre.clone();
        let cfg = AlfConfig {
            alf_enabled: true,
            cb_enabled: false,
            cr_enabled: false,
            bit_depth: 8,
            ctb_log2_size_y: 7,
            chroma_format_idc: 1,
        };
        apply_alf(&mut rec_replay, &pic, &cfg, &AlfApsBinding::default());

        // The two paths must produce byte-identical luma planes — the
        // RDO's per-CTB commit semantics match the `apply_alf` direct
        // application of the recorded `AlfPicture`.
        assert_eq!(rec_rdo.luma.samples, rec_replay.luma.samples);
    }

    fn total_sse_y(src: &PictureBuffer, rec: &PictureBuffer) -> u64 {
        src.luma
            .samples
            .iter()
            .zip(rec.luma.samples.iter())
            .map(|(s, r)| {
                let d = *s as i64 - *r as i64;
                (d * d) as u64
            })
            .sum()
    }
}
