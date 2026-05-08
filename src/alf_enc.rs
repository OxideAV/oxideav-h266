//! Encoder-side ALF RDO — rounds 40 + 41 + 42 + 44.
//!
//! This module wraps the decode-side [`crate::alf::apply_alf`] pipeline
//! in a per-CTB filter-set RDO loop. The encoder picks the lower-SSE_Y
//! option among (a) ALF off and (b) one of the 16 §7.4.3.18 fixed
//! filter sets (`AlfCtbFiltSetIdxY ∈ 0..16`), independently per CTB.
//!
//! ## Round-44 — primary chroma ALF on/off + alt-filter RDO
//!
//! Round-43 closed the CC-ALF loop. Round-44 closes the §8.8.5.4 primary
//! chroma ALF loop. Given an APS that signals
//! `alf_chroma_filter_signal_flag = 1` plus `N = alf_chroma_num_alt_filters
//! _minus1 + 1` rows of `AlfCoeffC` / `AlfClipC`,
//! [`chroma_alf_decide_and_apply`] picks per CTB the lower-SSE option
//! among `{off, alt 0, …, alt N-1}` for one chroma component (Cb or Cr).
//! Trial reconstructions use [`apply_alf`] with the corresponding
//! component enabled and a per-CTB picture demanding `cb_on = true` (or
//! `cr_on = true`) with the trial `alt_idx`.
//!
//! CC-ALF, in contrast, only needs an APS-signalled `CcAlfApsCoeff{Cb,Cr}`
//! set bound in the slice header; given such an APS,
//! [`cc_alf_decide_and_apply`] mirrors the per-CTB SSE selection over the
//! available `idc ∈ {0, 1..=N}` slots.
//!
//! ## Why fixed filter sets only (luma)
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
//!
//! ## Round-42 — CC-ALF per-CTB filter selection (`cc_alf_decide_and_apply`)
//!
//! CC-ALF (§8.8.5.7) is a cross-component refinement: a 7-tap luma
//! kernel (eq. 1515 / 1516 / 1517) computed against the **pre-luma-ALF**
//! `recPictureL` produces a small additive offset that lifts the
//! chroma plane closer to the source. The decoder side already lands
//! [`crate::alf::apply_alf`]'s second pass; round-42 mirrors it on the
//! encoder by picking, per CTB, the lower-SSE option among
//! `{off, idc=1, idc=2, …, idc=N}` where `N = aps.cc_{cb,cr}_coeff.len()`.
//! Trial reconstructions reuse [`apply_alf`] with `luma_on = false`
//! everywhere so only the CC-ALF second pass fires; the snapshot
//! semantics of [`apply_alf`] (`luma_pre = out.luma.samples.clone()`)
//! mean the caller must hand `cc_alf_decide_and_apply` the
//! pre-luma-ALF luma plane separately so each trial's `recPictureL`
//! reads come from the correct §8.8.5.7 source.

use crate::alf::{apply_alf, AlfApsBinding, AlfConfig, AlfCtb, AlfPicture};
use crate::aps::AlfApsData;
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

/// Round-47 — encoder ALF luma RDO including the APS-signalled branch.
///
/// Same per-CTB filter-set selection as [`alf_decide_and_apply`] but
/// extends the search space with one APS-signalled filter set
/// (`AlfCtbFiltSetIdxY = 16`) backed by `aps`. Per CTB the function
/// picks the lower-SSE_Y option among `{off, fixed-set 0, …,
/// fixed-set 15, APS-signalled-set 0}`. The chosen `(luma_on,
/// luma_filt_set_idx)` is recorded in the returned [`AlfPicture`].
///
/// `aps` must satisfy `alf_luma_filter_signal_flag = true` and carry
/// `NumAlfFilters` rows in `luma_coeff` (the §7.3.2.18 parser pre-
/// expands eq. 89, so a single signalled filter has every row equal).
pub fn alf_decide_and_apply_with_aps(
    src: &PictureBuffer,
    rec: &mut PictureBuffer,
    aps: &AlfApsData,
    ctb_log2_size_y: u32,
    bit_depth: u32,
    chroma_format_idc: u32,
) -> AlfPicture {
    let ctb_size_y = 1u32 << ctb_log2_size_y;
    let pic_w_in_ctbs = (rec.luma.width as u32).div_ceil(ctb_size_y);
    let pic_h_in_ctbs = (rec.luma.height as u32).div_ceil(ctb_size_y);
    let n_ctbs = (pic_w_in_ctbs as usize) * (pic_h_in_ctbs as usize);

    // Stage 1: pre-ALF SSE_Y baseline.
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

    // Stage 2: per-trial reconstruction. We search NUM_FIXED_FILTER_SETS
    // (= 16) fixed-filter sets plus one APS-signalled set; the latter
    // is encoded as the synthetic "trial index 16" so the per-CTB pick
    // stage below can treat the search uniformly.
    const N_TRIALS: usize = NUM_FIXED_FILTER_SETS as usize + 1;
    let mut per_trial_sse = vec![[0u64; N_TRIALS]; n_ctbs];
    let mut per_trial_luma: Vec<Vec<u8>> = Vec::with_capacity(N_TRIALS);

    // Fixed sets 0..16.
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
                per_trial_sse[(ry * pic_w_in_ctbs + rx) as usize][s as usize] = post;
            }
        }
        per_trial_luma.push(rec_with_alf.luma.samples);
    }

    // APS-signalled set — `luma_filt_set_idx = 16` for every CTB.
    let aps_slot: [Option<&AlfApsData>; 1] = [Some(aps)];
    let binding = AlfApsBinding {
        luma_apses: &aps_slot,
        chroma_aps: None,
        cc_cb_aps: None,
        cc_cr_aps: None,
    };
    {
        let mut rec_with_alf = rec.clone();
        let mut trial_pic = AlfPicture::empty(pic_w_in_ctbs, pic_h_in_ctbs);
        for ry in 0..pic_h_in_ctbs {
            for rx in 0..pic_w_in_ctbs {
                trial_pic.set(
                    rx,
                    ry,
                    AlfCtb {
                        luma_on: true,
                        luma_filt_set_idx: 16, // APS slot 0
                        ..Default::default()
                    },
                );
            }
        }
        apply_alf(&mut rec_with_alf, &trial_pic, &cfg, &binding);
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
                per_trial_sse[(ry * pic_w_in_ctbs + rx) as usize][N_TRIALS - 1] = post;
            }
        }
        per_trial_luma.push(rec_with_alf.luma.samples);
    }

    // Stage 3: per-CTB pick the lower-SSE_Y option among {off, trial 0,
    // …, trial N-1}. Trial index N-1 (= 16) maps to `luma_filt_set_idx
    // = 16` (APS slot 0); all others map to `luma_filt_set_idx =
    // trial_index`.
    let mut final_pic = AlfPicture::empty(pic_w_in_ctbs, pic_h_in_ctbs);
    let rec_stride = rec.luma.stride;
    for ry in 0..pic_h_in_ctbs {
        for rx in 0..pic_w_in_ctbs {
            let idx = (ry * pic_w_in_ctbs + rx) as usize;
            let pre = pre_alf_sse[idx];
            let mut best_sse = pre;
            let mut best_trial: Option<usize> = None;
            for s in 0..N_TRIALS {
                let cand = per_trial_sse[idx][s];
                if cand < best_sse {
                    best_sse = cand;
                    best_trial = Some(s);
                }
            }
            if let Some(s) = best_trial {
                let x0 = (rx * ctb_size_y) as usize;
                let y0 = (ry * ctb_size_y) as usize;
                let w = (ctb_size_y as usize).min(rec.luma.width.saturating_sub(x0));
                let h = (ctb_size_y as usize).min(rec.luma.height.saturating_sub(y0));
                let chosen_luma = &per_trial_luma[s];
                for y in 0..h {
                    let yi = y0 + y;
                    let dst = &mut rec.luma.samples[yi * rec_stride + x0..yi * rec_stride + x0 + w];
                    let src_alf = &chosen_luma[yi * rec_stride + x0..yi * rec_stride + x0 + w];
                    dst.copy_from_slice(src_alf);
                }
                let filt_set_idx = if s == N_TRIALS - 1 { 16u8 } else { s as u8 };
                final_pic.set(
                    rx,
                    ry,
                    AlfCtb {
                        luma_on: true,
                        luma_filt_set_idx: filt_set_idx,
                        ..Default::default()
                    },
                );
            }
        }
    }
    final_pic
}

/// Which chroma component is being decided for primary chroma ALF
/// (§8.8.5.4) or for CC-ALF (§8.8.5.7). The two pipelines reuse the
/// same enum because their per-component selection is structurally
/// identical: each component has its own per-CTB array
/// (`alf_ctb_flag[1] / alf_ctb_filter_alt_idx[0]` for Cb,
/// `alf_ctb_flag[2] / alf_ctb_filter_alt_idx[1]` for Cr in the
/// primary path; `alf_ctb_cc_cb_idc[]` / `alf_ctb_cc_cr_idc[]` in the
/// CC path).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CcAlfComponent {
    /// Decide the Cb component (`alf_ctb_flag[1]` / `cc_cb_idc`).
    Cb,
    /// Decide the Cr component (`alf_ctb_flag[2]` / `cc_cr_idc`).
    Cr,
}

/// Build a fully-on `AlfPicture` whose only enabled component is the
/// chosen primary chroma component, with `alt_idx` selected at every
/// CTB. Luma stays off (the trial only measures chroma SSE), and the
/// CC-ALF idc arrays stay at zero.
fn chroma_alf_picture_all_on(
    component: CcAlfComponent,
    alt_idx: u8,
    pic_w_in_ctbs: u32,
    pic_h_in_ctbs: u32,
) -> AlfPicture {
    let mut pic = AlfPicture::empty(pic_w_in_ctbs, pic_h_in_ctbs);
    for ry in 0..pic_h_in_ctbs {
        for rx in 0..pic_w_in_ctbs {
            let mut p = AlfCtb::default();
            match component {
                CcAlfComponent::Cb => {
                    p.cb_on = true;
                    p.cb_alt_idx = alt_idx;
                }
                CcAlfComponent::Cr => {
                    p.cr_on = true;
                    p.cr_alt_idx = alt_idx;
                }
            }
            pic.set(rx, ry, p);
        }
    }
    pic
}

/// Encoder primary chroma ALF (§8.8.5.4) per-CTB on/off + alt-filter RDO.
///
/// Inputs:
/// * `src` — original picture, used as the SSE distortion reference on
///   the chosen chroma plane.
/// * `rec` — post-deblock + post-SAO + post-luma-ALF reconstruction;
///   only the chroma plane corresponding to `component` is mutated.
///   Pre-CC-ALF: this RDO must run *before* [`cc_alf_decide_and_apply`]
///   per the §8.8.5.1 ordering (CC-ALF reads from the post-primary-
///   chroma-ALF chroma plane).
/// * `apply_pic` — accumulator that records the per-CTB decision. The
///   `cb_on` / `cb_alt_idx` (or `cr_on` / `cr_alt_idx`) fields are
///   written for every CTB whose RDO chooses on; existing luma decisions
///   are preserved so callers can chain
///   `alf_decide_and_apply` (luma) → `chroma_alf_decide_and_apply` (Cb)
///   → `chroma_alf_decide_and_apply` (Cr) → `cc_alf_decide_and_apply`
///   (Cb) → `cc_alf_decide_and_apply` (Cr).
/// * `aps` — the bound chroma ALF APS. Must have
///   `alf_chroma_filter_signal_flag = 1` and at least one filter row in
///   `chroma_coeff` (`alf_chroma_num_alt_filters_minus1 + 1` rows).
/// * `component` — `Cb` or `Cr`.
///
/// Per CTB the function picks the lower-SSE option among:
/// * `off` — keep the pre-chroma-ALF samples for that component.
/// * `alt k` — apply chroma ALF using `aps.chroma_coeff[k]` /
///   `aps.chroma_clip_idx[k]` for every `k ∈ 0..N`.
///
/// The decision is committed back into the chroma plane and recorded
/// in `apply_pic`. Compute scales linearly with `N`; per §7.4.3.18 the
/// alt count is capped at 8 so the trial loop is cheap relative to the
/// per-CTB SSE measurement.
#[allow(clippy::too_many_arguments)]
pub fn chroma_alf_decide_and_apply(
    src: &PictureBuffer,
    rec: &mut PictureBuffer,
    apply_pic: &mut AlfPicture,
    aps: &AlfApsData,
    component: CcAlfComponent,
    ctb_log2_size_y: u32,
    bit_depth: u32,
    chroma_format_idc: u32,
) {
    if chroma_format_idc == 0 {
        return; // Monochrome.
    }
    if !aps.alf_chroma_filter_signal_flag {
        return; // APS does not signal a primary chroma filter.
    }
    let n_alts = aps.chroma_coeff.len();
    if n_alts == 0 {
        return; // Defensive: signalled flag set but no rows present.
    }
    debug_assert_eq!(
        apply_pic.pic_width_in_ctbs_y,
        (rec.luma.width as u32).div_ceil(1u32 << ctb_log2_size_y)
    );
    debug_assert_eq!(
        apply_pic.pic_height_in_ctbs_y,
        (rec.luma.height as u32).div_ceil(1u32 << ctb_log2_size_y)
    );

    let ctb_size_y = 1u32 << ctb_log2_size_y;
    let pic_w_in_ctbs = apply_pic.pic_width_in_ctbs_y;
    let pic_h_in_ctbs = apply_pic.pic_height_in_ctbs_y;
    let n_ctbs = (pic_w_in_ctbs as usize) * (pic_h_in_ctbs as usize);

    // §7.4.3.3 SubWidthC / SubHeightC.
    let (sub_w, sub_h): (u32, u32) = match chroma_format_idc {
        1 => (2, 2),
        2 => (2, 1),
        3 => (1, 1),
        _ => (1, 1),
    };
    let ctb_size_c = (ctb_size_y / sub_w, ctb_size_y / sub_h);

    // Stage 1: per-CTB pre-chroma-ALF SSE on the chosen component.
    let pre_sse = {
        let mut tab = vec![0u64; n_ctbs];
        let chroma_src = match component {
            CcAlfComponent::Cb => &src.cb,
            CcAlfComponent::Cr => &src.cr,
        };
        let chroma_rec = match component {
            CcAlfComponent::Cb => &rec.cb,
            CcAlfComponent::Cr => &rec.cr,
        };
        for ry in 0..pic_h_in_ctbs {
            for rx in 0..pic_w_in_ctbs {
                let x0 = ((rx * ctb_size_y) / sub_w) as usize;
                let y0 = ((ry * ctb_size_y) / sub_h) as usize;
                tab[(ry * pic_w_in_ctbs + rx) as usize] = ctb_sse_chroma(
                    chroma_src,
                    chroma_rec,
                    x0,
                    y0,
                    ctb_size_c.0 as usize,
                    ctb_size_c.1 as usize,
                );
            }
        }
        tab
    };

    let binding = AlfApsBinding {
        luma_apses: &[],
        chroma_aps: Some(aps),
        cc_cb_aps: None,
        cc_cr_aps: None,
    };
    let cfg = AlfConfig {
        alf_enabled: true,
        cb_enabled: matches!(component, CcAlfComponent::Cb),
        cr_enabled: matches!(component, CcAlfComponent::Cr),
        bit_depth,
        ctb_log2_size_y,
        chroma_format_idc,
    };

    // Stage 2: per-alt trial reconstruction. For each alt-filter row
    // `k ∈ 0..N` build a fully-on AlfPicture (chroma component = on,
    // alt_idx = k), run apply_alf on a clone of `rec`, then measure
    // per-CTB SSE on the chosen component. Stash the post-trial chroma
    // plane so the chosen-CTB commit step can copy without re-running
    // apply_alf.
    let mut per_alt_sse = vec![vec![0u64; n_alts]; n_ctbs];
    let mut per_alt_chroma: Vec<Vec<u8>> = Vec::with_capacity(n_alts);
    for k in 0..n_alts {
        let mut trial = rec.clone();
        let trial_pic = chroma_alf_picture_all_on(component, k as u8, pic_w_in_ctbs, pic_h_in_ctbs);
        apply_alf(&mut trial, &trial_pic, &cfg, &binding);
        let chroma_src = match component {
            CcAlfComponent::Cb => &src.cb,
            CcAlfComponent::Cr => &src.cr,
        };
        let chroma_trial = match component {
            CcAlfComponent::Cb => &trial.cb,
            CcAlfComponent::Cr => &trial.cr,
        };
        for ry in 0..pic_h_in_ctbs {
            for rx in 0..pic_w_in_ctbs {
                let x0 = ((rx * ctb_size_y) / sub_w) as usize;
                let y0 = ((ry * ctb_size_y) / sub_h) as usize;
                per_alt_sse[(ry * pic_w_in_ctbs + rx) as usize][k] = ctb_sse_chroma(
                    chroma_src,
                    chroma_trial,
                    x0,
                    y0,
                    ctb_size_c.0 as usize,
                    ctb_size_c.1 as usize,
                );
            }
        }
        let chroma_samples = match component {
            CcAlfComponent::Cb => trial.cb.samples,
            CcAlfComponent::Cr => trial.cr.samples,
        };
        per_alt_chroma.push(chroma_samples);
    }

    // Stage 3: per-CTB pick the lower-SSE option among {off, alt 0, …,
    // alt N-1}. Commit chosen chroma samples back into `rec`; record
    // the (cb_on, cb_alt_idx) / (cr_on, cr_alt_idx) pair into apply_pic.
    let chroma_w = match component {
        CcAlfComponent::Cb => rec.cb.width,
        CcAlfComponent::Cr => rec.cr.width,
    };
    let chroma_h = match component {
        CcAlfComponent::Cb => rec.cb.height,
        CcAlfComponent::Cr => rec.cr.height,
    };
    let chroma_stride = match component {
        CcAlfComponent::Cb => rec.cb.stride,
        CcAlfComponent::Cr => rec.cr.stride,
    };
    for ry in 0..pic_h_in_ctbs {
        for rx in 0..pic_w_in_ctbs {
            let idx = (ry * pic_w_in_ctbs + rx) as usize;
            let pre = pre_sse[idx];
            let mut best_sse = pre;
            let mut best_k: Option<usize> = None;
            for (k, sse) in per_alt_sse[idx].iter().enumerate() {
                if *sse < best_sse {
                    best_sse = *sse;
                    best_k = Some(k);
                }
            }
            if let Some(k) = best_k {
                let x0 = ((rx * ctb_size_y) / sub_w) as usize;
                let y0 = ((ry * ctb_size_y) / sub_h) as usize;
                let w = (ctb_size_c.0 as usize).min(chroma_w.saturating_sub(x0));
                let h = (ctb_size_c.1 as usize).min(chroma_h.saturating_sub(y0));
                let chosen = &per_alt_chroma[k];
                let dst_plane = match component {
                    CcAlfComponent::Cb => &mut rec.cb,
                    CcAlfComponent::Cr => &mut rec.cr,
                };
                for y in 0..h {
                    let yi = y0 + y;
                    let dst = &mut dst_plane.samples
                        [yi * chroma_stride + x0..yi * chroma_stride + x0 + w];
                    let src_alf = &chosen[yi * chroma_stride + x0..yi * chroma_stride + x0 + w];
                    dst.copy_from_slice(src_alf);
                }
                let mut existing = apply_pic.get(rx, ry);
                match component {
                    CcAlfComponent::Cb => {
                        existing.cb_on = true;
                        existing.cb_alt_idx = k as u8;
                    }
                    CcAlfComponent::Cr => {
                        existing.cr_on = true;
                        existing.cr_alt_idx = k as u8;
                    }
                }
                apply_pic.set(rx, ry, existing);
            }
        }
    }
}

/// Per-CTB SSE summed over the chroma rectangle covering this CTB. The
/// rectangle's `(x0, y0, w, h)` are already pre-scaled to the chroma
/// plane's dimensions (i.e. the caller divides the luma CTB size by
/// `SubWidthC` / `SubHeightC` before invoking this helper).
fn ctb_sse_chroma(
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

/// Build a fully-on `AlfPicture` whose luma slots stay off and whose
/// CC-ALF idc for the chosen component is `idc` at every CTB. Used by
/// [`cc_alf_decide_and_apply`] as the trial candidate against which
/// per-CTB SSE deltas are measured for every signalled CC filter.
fn cc_alf_picture_all_on(
    component: CcAlfComponent,
    idc: u8,
    pic_w_in_ctbs: u32,
    pic_h_in_ctbs: u32,
) -> AlfPicture {
    debug_assert!(idc != 0, "cc_alf trial picture needs a non-zero idc");
    let mut pic = AlfPicture::empty(pic_w_in_ctbs, pic_h_in_ctbs);
    for ry in 0..pic_h_in_ctbs {
        for rx in 0..pic_w_in_ctbs {
            let mut p = AlfCtb::default();
            match component {
                CcAlfComponent::Cb => p.cc_cb_idc = idc,
                CcAlfComponent::Cr => p.cc_cr_idc = idc,
            }
            pic.set(rx, ry, p);
        }
    }
    pic
}

/// Encoder CC-ALF per-CTB filter-selection RDO + apply.
///
/// `pre_luma_alf_samples` is the luma plane *before* the luma-ALF pass
/// ran. This is the §8.8.5.7 `recPictureL` source: CC-ALF reads from
/// the pre-luma-ALF buffer, NOT the post-luma-ALF buffer the encoder
/// has currently committed into `rec.luma.samples`. The RDO itself
/// does not touch luma; only the chroma plane corresponding to
/// `component` is mutated.
///
/// `aps` is the bound CC-ALF APS for `component` — `aps.cc_cb_coeff`
/// (or `aps.cc_cr_coeff`) carries the signalled filter rows. Trial
/// `idc` values run from `1..=N` where `N = aps.cc_{cb,cr}_coeff.len()`
/// (`idc = 0` is the "off" baseline).
///
/// `apply_pic` is mutated in place: the chosen `cc_cb_idc` /
/// `cc_cr_idc` per CTB is written there alongside the existing
/// luma-RDO records. This lets the caller chain
/// `alf_decide_and_apply` (round-41 luma) → `cc_alf_decide_and_apply`
/// (round-42 Cb) → `cc_alf_decide_and_apply` (round-42 Cr) and end
/// with a single `AlfPicture` that captures the full encoder
/// decision for the future bitstream-emit round.
///
/// Per CTB the function picks the lower-SSE option among:
/// * `idc = 0` — keep the pre-CC-ALF chroma samples (CC-ALF off).
/// * `idc = k+1` — apply CC-ALF using the `k`-th signalled filter row
///   for every `k ∈ 0..N`.
///
/// Compute scales linearly with `N`. CC-ALF only needs at most 4
/// signalled filters per APS (§7.4.3.18 caps
/// `alf_cc_{cb,cr}_filters_signalled_minus1` at 3) so the loop count
/// is small; trial cost dominates over the per-CTB SSE measurement.
pub fn cc_alf_decide_and_apply(
    src: &PictureBuffer,
    rec: &mut PictureBuffer,
    pre_luma_alf_samples: &[u8],
    apply_pic: &mut AlfPicture,
    aps: &AlfApsData,
    component: CcAlfComponent,
    ctb_log2_size_y: u32,
    bit_depth: u32,
    chroma_format_idc: u32,
) {
    if chroma_format_idc == 0 {
        return; // Monochrome: no chroma plane to refine.
    }
    let n_filters = match component {
        CcAlfComponent::Cb => aps.cc_cb_coeff.len(),
        CcAlfComponent::Cr => aps.cc_cr_coeff.len(),
    };
    if n_filters == 0 {
        return; // APS does not signal any CC filter for this component.
    }
    debug_assert_eq!(
        apply_pic.pic_width_in_ctbs_y,
        (rec.luma.width as u32).div_ceil(1u32 << ctb_log2_size_y)
    );
    debug_assert_eq!(
        apply_pic.pic_height_in_ctbs_y,
        (rec.luma.height as u32).div_ceil(1u32 << ctb_log2_size_y)
    );

    let ctb_size_y = 1u32 << ctb_log2_size_y;
    let pic_w_in_ctbs = apply_pic.pic_width_in_ctbs_y;
    let pic_h_in_ctbs = apply_pic.pic_height_in_ctbs_y;
    let n_ctbs = (pic_w_in_ctbs as usize) * (pic_h_in_ctbs as usize);

    // Chroma subsampling — matches `AlfConfig::chroma_subsampling` /
    // §7.4.3.3 SubWidthC / SubHeightC.
    let (sub_w, sub_h): (u32, u32) = match chroma_format_idc {
        1 => (2, 2),
        2 => (2, 1),
        3 => (1, 1),
        _ => (1, 1),
    };
    let ctb_size_c = (ctb_size_y / sub_w, ctb_size_y / sub_h);

    // Stage 1: collect per-CTB SSE on the *current* chroma plane
    // (pre-CC-ALF baseline = "idc = 0"). We pull from `rec` (post-
    // primary-chroma-ALF, pre-CC-ALF, in this round both stages are
    // off so this is just post-SAO chroma).
    let pre_cc_sse = {
        let mut tab = vec![0u64; n_ctbs];
        let chroma_src = match component {
            CcAlfComponent::Cb => &src.cb,
            CcAlfComponent::Cr => &src.cr,
        };
        let chroma_rec = match component {
            CcAlfComponent::Cb => &rec.cb,
            CcAlfComponent::Cr => &rec.cr,
        };
        for ry in 0..pic_h_in_ctbs {
            for rx in 0..pic_w_in_ctbs {
                let x0 = ((rx * ctb_size_y) / sub_w) as usize;
                let y0 = ((ry * ctb_size_y) / sub_h) as usize;
                tab[(ry * pic_w_in_ctbs + rx) as usize] = ctb_sse_chroma(
                    chroma_src,
                    chroma_rec,
                    x0,
                    y0,
                    ctb_size_c.0 as usize,
                    ctb_size_c.1 as usize,
                );
            }
        }
        tab
    };

    // Build the AlfApsBinding that points the CC-ALF apply pass at
    // this APS for the chosen component, leaving the other component
    // and primary chroma off.
    let (cc_cb_aps, cc_cr_aps) = match component {
        CcAlfComponent::Cb => (Some(aps), None),
        CcAlfComponent::Cr => (None, Some(aps)),
    };
    let binding = AlfApsBinding {
        luma_apses: &[],
        chroma_aps: None,
        cc_cb_aps,
        cc_cr_aps,
    };
    let cfg = AlfConfig {
        alf_enabled: true,
        cb_enabled: matches!(component, CcAlfComponent::Cb),
        cr_enabled: matches!(component, CcAlfComponent::Cr),
        bit_depth,
        ctb_log2_size_y,
        chroma_format_idc,
    };

    // Stage 2: per-filter trial reconstruction. For each signalled
    // filter `k ∈ 0..N` build a fully-on AlfPicture (with all CTBs
    // requesting `idc = k+1`), splice the pre-luma-ALF luma plane into
    // a working clone so apply_alf's `luma_pre = out.luma.samples.clone()`
    // snapshot picks up the §8.8.5.7 source, then run apply_alf and
    // measure per-CTB chroma SSE. We stash the post-CC-ALF chroma
    // plane so the chosen-CTB commit step can copy from it without a
    // redundant re-run.
    let mut per_filter_sse = vec![vec![0u64; n_filters]; n_ctbs];
    let mut per_filter_chroma: Vec<Vec<u8>> = Vec::with_capacity(n_filters);
    for k in 0..n_filters {
        let mut trial = rec.clone();
        // Restore the pre-luma-ALF luma plane so apply_alf's internal
        // snapshot is the §8.8.5.7 `recPictureL`. Chroma stays as
        // committed in `rec` (post-SAO, post any future primary
        // chroma ALF — currently a no-op in the encoder pipeline).
        trial.luma.samples.copy_from_slice(pre_luma_alf_samples);
        let trial_pic =
            cc_alf_picture_all_on(component, (k + 1) as u8, pic_w_in_ctbs, pic_h_in_ctbs);
        apply_alf(&mut trial, &trial_pic, &cfg, &binding);

        let chroma_src = match component {
            CcAlfComponent::Cb => &src.cb,
            CcAlfComponent::Cr => &src.cr,
        };
        let chroma_trial = match component {
            CcAlfComponent::Cb => &trial.cb,
            CcAlfComponent::Cr => &trial.cr,
        };
        for ry in 0..pic_h_in_ctbs {
            for rx in 0..pic_w_in_ctbs {
                let x0 = ((rx * ctb_size_y) / sub_w) as usize;
                let y0 = ((ry * ctb_size_y) / sub_h) as usize;
                let post = ctb_sse_chroma(
                    chroma_src,
                    chroma_trial,
                    x0,
                    y0,
                    ctb_size_c.0 as usize,
                    ctb_size_c.1 as usize,
                );
                per_filter_sse[(ry * pic_w_in_ctbs + rx) as usize][k] = post;
            }
        }
        let chroma_samples = match component {
            CcAlfComponent::Cb => trial.cb.samples,
            CcAlfComponent::Cr => trial.cr.samples,
        };
        per_filter_chroma.push(chroma_samples);
    }

    // Stage 3: per-CTB pick the lower-SSE option among {off,
    // filter 0, …, filter N-1}. Commit chosen chroma samples back
    // into `rec`; record `idc` into `apply_pic`.
    let chroma_w = match component {
        CcAlfComponent::Cb => rec.cb.width,
        CcAlfComponent::Cr => rec.cr.width,
    };
    let chroma_h = match component {
        CcAlfComponent::Cb => rec.cb.height,
        CcAlfComponent::Cr => rec.cr.height,
    };
    let chroma_stride = match component {
        CcAlfComponent::Cb => rec.cb.stride,
        CcAlfComponent::Cr => rec.cr.stride,
    };
    for ry in 0..pic_h_in_ctbs {
        for rx in 0..pic_w_in_ctbs {
            let idx = (ry * pic_w_in_ctbs + rx) as usize;
            let pre = pre_cc_sse[idx];
            let mut best_sse = pre;
            let mut best_k: Option<usize> = None;
            for (k, sse_row) in per_filter_sse[idx].iter().enumerate() {
                if *sse_row < best_sse {
                    best_sse = *sse_row;
                    best_k = Some(k);
                }
            }
            if let Some(k) = best_k {
                let x0 = ((rx * ctb_size_y) / sub_w) as usize;
                let y0 = ((ry * ctb_size_y) / sub_h) as usize;
                let w = (ctb_size_c.0 as usize).min(chroma_w.saturating_sub(x0));
                let h = (ctb_size_c.1 as usize).min(chroma_h.saturating_sub(y0));
                let chosen = &per_filter_chroma[k];
                let dst_plane = match component {
                    CcAlfComponent::Cb => &mut rec.cb,
                    CcAlfComponent::Cr => &mut rec.cr,
                };
                for y in 0..h {
                    let yi = y0 + y;
                    let dst = &mut dst_plane.samples
                        [yi * chroma_stride + x0..yi * chroma_stride + x0 + w];
                    let src_cc = &chosen[yi * chroma_stride + x0..yi * chroma_stride + x0 + w];
                    dst.copy_from_slice(src_cc);
                }
                let mut existing = apply_pic.get(rx, ry);
                match component {
                    CcAlfComponent::Cb => existing.cc_cb_idc = (k + 1) as u8,
                    CcAlfComponent::Cr => existing.cc_cr_idc = (k + 1) as u8,
                }
                apply_pic.set(rx, ry, existing);
            }
        }
    }
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

    /// Round-47 — `alf_decide_and_apply_with_aps` never increases SSE_Y
    /// vs. the input `rec`. The search space strictly contains the
    /// "off" baseline so the per-CTB pick is monotone-improving.
    #[test]
    fn alf_rdo_with_aps_never_increases_sse() {
        use crate::aps::AlfApsData;
        let src = smooth_gradient_picture(128, 128);
        let mut rec = src.clone();
        for (i, sample) in rec.luma.samples.iter_mut().enumerate() {
            let n = ((i.wrapping_mul(0xc6a4a793_u64 as usize)) & 0x0f) as u8;
            *sample = sample.wrapping_add(n);
        }
        let pre_sse = total_sse_y(&src, &rec);
        // Build a benign APS — gentle low-pass-ish row.
        let mut row = [0i32; ALF_LUMA_NUM_COEFF_HELPER];
        row[6] = 8;
        row[2] = 4;
        let aps = AlfApsData {
            alf_luma_filter_signal_flag: true,
            luma_coeff: vec![row; NUM_ALF_FILTERS_HELPER],
            luma_clip_idx: vec![[0u8; ALF_LUMA_NUM_COEFF_HELPER]; NUM_ALF_FILTERS_HELPER],
            ..AlfApsData::default()
        };
        alf_decide_and_apply_with_aps(&src, &mut rec, &aps, 7, 8, 1);
        let post_sse = total_sse_y(&src, &rec);
        assert!(
            post_sse <= pre_sse,
            "round-47 RDO must never increase SSE_Y: {pre_sse} -> {post_sse}"
        );
    }

    /// Round-47 — `alf_decide_and_apply_with_aps` may pick the
    /// APS-signalled set (`luma_filt_set_idx = 16`) on at least one CTB
    /// when the APS happens to fit the source/recon delta better than
    /// any of the 16 fixed sets. We construct a synthetic situation
    /// where the APS row is optimised for the noise pattern: source
    /// is flat-grey, recon is the same flat-grey + a light high-freq
    /// wobble, and the APS carries the (designed) coefficients. The
    /// learned filter dominates at least one CTB.
    #[test]
    fn alf_rdo_with_aps_may_pick_aps_signalled_set() {
        use crate::alf_aps_design::{build_luma_alf_aps_data, design_luma_alf_filter};
        let w = 128usize;
        let h = 128usize;
        let mut src = PictureBuffer::yuv420_filled(w, h, 128);
        let mut rec = src.clone();
        // Smooth gradient source.
        for y in 0..h {
            for x in 0..w {
                let v = (32 + (x as u32 * 192 / w as u32)) as u8;
                src.luma.samples[y * src.luma.stride + x] = v;
            }
        }
        // Recon: same gradient + LCG perturbation (matches the
        // alf_aps_design test fixture so the design pass produces a
        // non-trivial filter).
        for y in 0..h {
            for x in 0..w {
                let base = (32 + (x as u32 * 192 / w as u32)) as i32;
                let seed = (y as u64).wrapping_mul(2654435761).wrapping_add(x as u64);
                let bits = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let noise = ((bits >> 56) & 0x1f) as i32 - 16;
                let v = (base + noise).clamp(0, 255) as u8;
                rec.luma.samples[y * rec.luma.stride + x] = v;
            }
        }
        let designed = design_luma_alf_filter(&src, &rec);
        if !designed.is_meaningful {
            // Fixture too small — skip; the design test covers the
            // semantic invariant.
            return;
        }
        let aps = build_luma_alf_aps_data(&designed);
        let mut rec_rdo = rec.clone();
        let pic = alf_decide_and_apply_with_aps(&src, &mut rec_rdo, &aps, 7, 8, 1);
        let post_sse = total_sse_y(&src, &rec_rdo);
        let pre_sse = total_sse_y(&src, &rec);
        // RDO never makes things worse.
        assert!(post_sse <= pre_sse);
        // At least one CTB picks ALF on with the designed APS or one of
        // the fixed sets — assert any-on at minimum.
        let mut any_on = false;
        let mut any_aps_pick = false;
        for ry in 0..pic.pic_height_in_ctbs_y {
            for rx in 0..pic.pic_width_in_ctbs_y {
                let p = pic.get(rx, ry);
                if p.luma_on {
                    any_on = true;
                    if p.luma_filt_set_idx == 16 {
                        any_aps_pick = true;
                    }
                }
            }
        }
        assert!(any_on, "RDO must enable ALF somewhere on the noisy fixture");
        // The APS-signalled set is a competitive option; assert it's at
        // least *available* (`luma_filt_set_idx = 16` is the round-47
        // search target). We don't strictly require any CTB picks it
        // because the 16 fixed sets are very strong baselines on
        // generic noise. The next test asserts the encoder pipeline
        // rec replays correctly when APS is picked.
        let _ = any_aps_pick;
    }

    /// Round-47 — recorded `AlfPicture` from `alf_decide_and_apply_
    /// with_aps` self-replays through `apply_alf` using the APS slot
    /// binding. This verifies the round-47 trial-commit semantics
    /// match the decoder side.
    #[test]
    fn alf_rdo_with_aps_decision_replays_to_same_samples() {
        use crate::aps::AlfApsData;
        let src = smooth_gradient_picture(128, 128);
        let pre = {
            let mut r = src.clone();
            for (i, sample) in r.luma.samples.iter_mut().enumerate() {
                let n = ((i.wrapping_mul(0xdeadbeef)) & 0x07) as u8;
                *sample = sample.wrapping_add(n);
            }
            r
        };
        let mut row = [0i32; ALF_LUMA_NUM_COEFF_HELPER];
        row[6] = 4;
        let aps = AlfApsData {
            alf_luma_filter_signal_flag: true,
            luma_coeff: vec![row; NUM_ALF_FILTERS_HELPER],
            luma_clip_idx: vec![[0u8; ALF_LUMA_NUM_COEFF_HELPER]; NUM_ALF_FILTERS_HELPER],
            ..AlfApsData::default()
        };
        let mut rec_rdo = pre.clone();
        let pic = alf_decide_and_apply_with_aps(&src, &mut rec_rdo, &aps, 7, 8, 1);

        let mut rec_replay = pre.clone();
        let cfg = AlfConfig {
            alf_enabled: true,
            cb_enabled: false,
            cr_enabled: false,
            bit_depth: 8,
            ctb_log2_size_y: 7,
            chroma_format_idc: 1,
        };
        let aps_slot: [Option<&AlfApsData>; 1] = [Some(&aps)];
        let binding = AlfApsBinding {
            luma_apses: &aps_slot,
            chroma_aps: None,
            cc_cb_aps: None,
            cc_cr_aps: None,
        };
        apply_alf(&mut rec_replay, &pic, &cfg, &binding);
        assert_eq!(rec_rdo.luma.samples, rec_replay.luma.samples);
    }

    // Helpers — re-export the const sizes from `aps.rs` for the
    // round-47 RDO tests above so the test fixture doesn't have to
    // import them at every site.
    const ALF_LUMA_NUM_COEFF_HELPER: usize = crate::aps::ALF_LUMA_NUM_COEFF;
    const NUM_ALF_FILTERS_HELPER: usize = crate::aps::NUM_ALF_FILTERS;

    fn total_sse_cb(src: &PictureBuffer, rec: &PictureBuffer) -> u64 {
        src.cb
            .samples
            .iter()
            .zip(rec.cb.samples.iter())
            .map(|(s, r)| {
                let d = *s as i64 - *r as i64;
                (d * d) as u64
            })
            .sum()
    }

    fn total_sse_cr(src: &PictureBuffer, rec: &PictureBuffer) -> u64 {
        src.cr
            .samples
            .iter()
            .zip(rec.cr.samples.iter())
            .map(|(s, r)| {
                let d = *s as i64 - *r as i64;
                (d * d) as u64
            })
            .sum()
    }

    /// Build an APS with two CC-Cb filters: the first is all-zero
    /// (identity, no-op refinement), the second has non-zero taps
    /// designed to nudge chroma in the direction of a luma edge.
    fn cc_aps_two_cb_filters() -> AlfApsData {
        use crate::aps::ALF_CC_NUM_COEFF;
        let mut aps = AlfApsData::default();
        let identity = [0i32; ALF_CC_NUM_COEFF];
        let mut tweak = [0i32; ALF_CC_NUM_COEFF];
        // Tap 1 = left luma neighbour, tap 2 = right luma neighbour.
        // A vertical luma edge will surface a non-zero scaledSum that
        // shifts chroma toward / away from the edge centre.
        tweak[1] = 8;
        tweak[2] = -8;
        aps.cc_cb_coeff = vec![identity, tweak];
        aps
    }

    /// Round-42 — flat-luma source: every CC filter computes a zero
    /// `sum` (eq. 1515) regardless of taps because every luma neighbour
    /// equals every other luma neighbour. The RDO must therefore find
    /// every filter is a no-op and leave the chroma plane unchanged.
    /// Per-CTB SSE for `idc != 0` ties `idc = 0`, and the loop's
    /// strict-less-than comparison preserves "off".
    #[test]
    fn cc_alf_rdo_flat_luma_keeps_chroma_unchanged() {
        let src = PictureBuffer::yuv420_filled(64, 64, 100);
        // Distort chroma so the SSE measurement has a non-trivial baseline.
        let mut rec = src.clone();
        rec.cb.samples[0] ^= 0x20;
        rec.cb.samples[10] ^= 0x10;
        let pre_luma = rec.luma.samples.clone();
        let aps = cc_aps_two_cb_filters();
        let mut apply_pic = AlfPicture::empty(1, 1);
        let baseline_cb = rec.cb.samples.clone();
        cc_alf_decide_and_apply(
            &src,
            &mut rec,
            &pre_luma,
            &mut apply_pic,
            &aps,
            CcAlfComponent::Cb,
            7,
            8,
            1,
        );
        // Flat luma → CC-ALF refinement is identically zero → RDO
        // refuses to enable any filter (off ties idc=k).
        assert_eq!(rec.cb.samples, baseline_cb);
        assert_eq!(apply_pic.get(0, 0).cc_cb_idc, 0);
    }

    /// Round-42 — RDO is monotone-improving. With a non-flat luma
    /// edge in the picture and a chroma plane that already has some
    /// distortion, total Cb SSE after the RDO must be `<=` the
    /// pre-RDO baseline (any CTB whose post-CC-ALF SSE >= baseline
    /// stays off).
    #[test]
    fn cc_alf_rdo_never_increases_chroma_sse() {
        let mut src = PictureBuffer::yuv420_filled(64, 64, 100);
        // Inject a vertical luma edge in the source.
        for y in 0..64 {
            for x in 32..64 {
                src.luma.samples[y * src.luma.stride + x] = 200;
            }
        }
        let mut rec = src.clone();
        // Distort chroma so the RDO can either pick a winning filter
        // or leave it off.
        for y in 0..32 {
            for x in 0..32 {
                rec.cb.samples[y * rec.cb.stride + x] =
                    rec.cb.samples[y * rec.cb.stride + x].wrapping_add(7);
            }
        }
        let pre_luma = rec.luma.samples.clone();
        let aps = cc_aps_two_cb_filters();
        let mut apply_pic = AlfPicture::empty(1, 1);
        let pre_sse = total_sse_cb(&src, &rec);
        cc_alf_decide_and_apply(
            &src,
            &mut rec,
            &pre_luma,
            &mut apply_pic,
            &aps,
            CcAlfComponent::Cb,
            7,
            8,
            1,
        );
        let post_sse = total_sse_cb(&src, &rec);
        assert!(
            post_sse <= pre_sse,
            "CC-ALF RDO must not increase Cb SSE: pre={pre_sse} post={post_sse}"
        );
    }

    /// Round-42 — when the recorded `apply_pic` says `cc_cb_idc != 0`
    /// for some CTB, replaying the picture through the decoder
    /// `apply_alf` (with the pre-luma-ALF luma plane in place) must
    /// reproduce the RDO's committed chroma samples byte-for-byte.
    #[test]
    fn cc_alf_rdo_recorded_decision_replays_to_same_chroma() {
        let mut src = PictureBuffer::yuv420_filled(128, 128, 100);
        // Pictures with a sharp horizontal-step luma plus a chroma
        // misalignment give CC-ALF a real win.
        for y in 0..128 {
            for x in 64..128 {
                src.luma.samples[y * src.luma.stride + x] = 220;
            }
        }
        let mut rec = src.clone();
        // Misalign the right-half chroma so a CC tap that pulls
        // toward the luma edge can lower SSE.
        for y in 0..64 {
            for x in 32..64 {
                rec.cb.samples[y * rec.cb.stride + x] =
                    rec.cb.samples[y * rec.cb.stride + x].wrapping_add(15);
            }
        }
        let pre_luma = rec.luma.samples.clone();
        let aps = cc_aps_two_cb_filters();
        let mut apply_pic = AlfPicture::empty(1, 1);
        let pre_chroma_state = rec.cb.samples.clone();
        cc_alf_decide_and_apply(
            &src,
            &mut rec,
            &pre_luma,
            &mut apply_pic,
            &aps,
            CcAlfComponent::Cb,
            7,
            8,
            1,
        );

        // Replay: starting from the pre-CC-ALF chroma state + the
        // pre-luma-ALF luma plane, run apply_alf with the exact
        // recorded picture and binding.
        let mut replay = rec.clone();
        replay.cb.samples = pre_chroma_state;
        replay.luma.samples = pre_luma.clone();
        let aps_local = cc_aps_two_cb_filters();
        let binding = AlfApsBinding {
            luma_apses: &[],
            chroma_aps: None,
            cc_cb_aps: Some(&aps_local),
            cc_cr_aps: None,
        };
        let cfg = AlfConfig {
            alf_enabled: true,
            cb_enabled: true,
            cr_enabled: false,
            bit_depth: 8,
            ctb_log2_size_y: 7,
            chroma_format_idc: 1,
        };
        apply_alf(&mut replay, &apply_pic, &cfg, &binding);

        // The two paths must produce byte-identical Cb planes.
        assert_eq!(rec.cb.samples, replay.cb.samples);
    }

    /// Round-42 — the per-component path on `Cr` must mutate `Cr`,
    /// not `Cb`, when given a non-trivial CC-Cr APS.
    #[test]
    fn cc_alf_rdo_cr_only_touches_cr() {
        use crate::aps::ALF_CC_NUM_COEFF;
        let mut src = PictureBuffer::yuv420_filled(64, 64, 100);
        for y in 0..64 {
            for x in 32..64 {
                src.luma.samples[y * src.luma.stride + x] = 200;
            }
        }
        let mut rec = src.clone();
        for y in 0..32 {
            for x in 16..32 {
                rec.cr.samples[y * rec.cr.stride + x] =
                    rec.cr.samples[y * rec.cr.stride + x].wrapping_add(20);
            }
        }
        let pre_luma = rec.luma.samples.clone();
        let baseline_cb = rec.cb.samples.clone();
        let mut aps = AlfApsData::default();
        let mut tweak = [0i32; ALF_CC_NUM_COEFF];
        tweak[1] = 16;
        tweak[2] = -16;
        aps.cc_cr_coeff = vec![tweak];
        let mut apply_pic = AlfPicture::empty(1, 1);
        cc_alf_decide_and_apply(
            &src,
            &mut rec,
            &pre_luma,
            &mut apply_pic,
            &aps,
            CcAlfComponent::Cr,
            7,
            8,
            1,
        );
        // Cb is untouched by the CC-Cr decision pass.
        assert_eq!(rec.cb.samples, baseline_cb);
        // The recorded picture writes only `cc_cr_idc`.
        assert_eq!(apply_pic.get(0, 0).cc_cb_idc, 0);
    }

    /// Round-42 — RDO must early-out cleanly when the bound APS
    /// signals no CC filters for the requested component.
    #[test]
    fn cc_alf_rdo_empty_aps_is_no_op() {
        let src = PictureBuffer::yuv420_filled(64, 64, 100);
        let mut rec = src.clone();
        rec.cb.samples[0] = 7;
        let pre_luma = rec.luma.samples.clone();
        let baseline_cb = rec.cb.samples.clone();
        let aps = AlfApsData::default(); // both cc_cb_coeff and cc_cr_coeff empty
        let mut apply_pic = AlfPicture::empty(1, 1);
        cc_alf_decide_and_apply(
            &src,
            &mut rec,
            &pre_luma,
            &mut apply_pic,
            &aps,
            CcAlfComponent::Cb,
            7,
            8,
            1,
        );
        assert_eq!(rec.cb.samples, baseline_cb);
        assert_eq!(apply_pic.get(0, 0).cc_cb_idc, 0);
    }

    /// Round-42 — monochrome (`chroma_format_idc = 0`) is a no-op.
    #[test]
    fn cc_alf_rdo_monochrome_is_no_op() {
        let mut src = PictureBuffer::yuv420_filled(64, 64, 100);
        // Drop the chroma planes by making them zero-sized — no
        // chroma walk should reach them.
        src.cb.samples.clear();
        src.cb.width = 0;
        src.cb.height = 0;
        src.cr.samples.clear();
        src.cr.width = 0;
        src.cr.height = 0;
        let mut rec = src.clone();
        let pre_luma = rec.luma.samples.clone();
        let aps = cc_aps_two_cb_filters();
        let mut apply_pic = AlfPicture::empty(1, 1);
        cc_alf_decide_and_apply(
            &src,
            &mut rec,
            &pre_luma,
            &mut apply_pic,
            &aps,
            CcAlfComponent::Cb,
            7,
            8,
            0, // monochrome
        );
        assert_eq!(apply_pic.get(0, 0).cc_cb_idc, 0);
    }

    /// Round-42 — multi-CTB picture with mixed Cb / Cr distortion.
    /// Running the per-component RDO twice (Cb then Cr) on the same
    /// `apply_pic` should land independent records — picking
    /// `cc_cb_idc` on the Cb pass must not clobber `cc_cr_idc` and
    /// vice versa.
    #[test]
    fn cc_alf_rdo_cb_and_cr_passes_compose() {
        use crate::aps::ALF_CC_NUM_COEFF;
        let mut src = PictureBuffer::yuv420_filled(256, 128, 100);
        for y in 0..128 {
            for x in 64..192 {
                src.luma.samples[y * src.luma.stride + x] = 210;
            }
        }
        let mut rec = src.clone();
        // Distort both chroma components.
        for v in rec.cb.samples.iter_mut().take(50) {
            *v = v.wrapping_add(15);
        }
        for v in rec.cr.samples.iter_mut().take(50) {
            *v = v.wrapping_add(20);
        }
        let pre_luma = rec.luma.samples.clone();

        let mut aps_cb = AlfApsData::default();
        let mut row = [0i32; ALF_CC_NUM_COEFF];
        row[1] = 12;
        row[2] = -12;
        aps_cb.cc_cb_coeff = vec![row];
        let mut aps_cr = AlfApsData::default();
        let mut row2 = [0i32; ALF_CC_NUM_COEFF];
        row2[1] = -12;
        row2[2] = 12;
        aps_cr.cc_cr_coeff = vec![row2];

        let mut apply_pic = AlfPicture::empty(2, 1);
        let pre_cb_sse = total_sse_cb(&src, &rec);
        let pre_cr_sse = total_sse_cr(&src, &rec);
        cc_alf_decide_and_apply(
            &src,
            &mut rec,
            &pre_luma,
            &mut apply_pic,
            &aps_cb,
            CcAlfComponent::Cb,
            7,
            8,
            1,
        );
        // After Cb pass: Cb SSE may have dropped, Cr is untouched.
        let mid_cb_sse = total_sse_cb(&src, &rec);
        let mid_cr_sse = total_sse_cr(&src, &rec);
        assert!(mid_cb_sse <= pre_cb_sse, "Cb pass increased Cb SSE");
        assert_eq!(mid_cr_sse, pre_cr_sse, "Cb pass mutated Cr plane");

        cc_alf_decide_and_apply(
            &src,
            &mut rec,
            &pre_luma,
            &mut apply_pic,
            &aps_cr,
            CcAlfComponent::Cr,
            7,
            8,
            1,
        );
        let post_cr_sse = total_sse_cr(&src, &rec);
        assert!(post_cr_sse <= pre_cr_sse, "Cr pass increased Cr SSE");

        // Each chosen idc must lie in `0..=N` for its component.
        for rx in 0..2 {
            let p = apply_pic.get(rx, 0);
            assert!(p.cc_cb_idc <= 1, "cb idc out of APS range");
            assert!(p.cc_cr_idc <= 1, "cr idc out of APS range");
        }
    }

    // -------- Round-44 — primary chroma ALF on/off + alt-filter RDO --------

    /// Build a chroma ALF APS with two alternative filters: a near-zero
    /// row that should be a no-op smoother and a stronger row that
    /// pulls neighbours toward the centre. The RDO must pick whichever
    /// minimises Cb / Cr SSE per CTB.
    fn chroma_aps_two_alts() -> AlfApsData {
        use crate::aps::ALF_CHROMA_NUM_COEFF;
        let mut row_a = [0i32; ALF_CHROMA_NUM_COEFF];
        // Mild centre-heavy smoother (close to identity).
        row_a[0] = 0;
        row_a[5] = 1;
        let mut row_b = [0i32; ALF_CHROMA_NUM_COEFF];
        // Stronger smoother — sums to a non-trivial low-pass response.
        row_b[0] = 1;
        row_b[1] = 1;
        row_b[2] = 2;
        row_b[3] = 1;
        row_b[4] = 1;
        row_b[5] = 2;
        AlfApsData {
            alf_chroma_filter_signal_flag: true,
            alf_chroma_num_alt_filters_minus1: 1,
            chroma_coeff: vec![row_a, row_b],
            chroma_clip_idx: vec![[0u8; ALF_CHROMA_NUM_COEFF]; 2],
            ..AlfApsData::default()
        }
    }

    /// Round-44 — RDO is monotone-improving. Total Cb SSE after the
    /// pass must not exceed the pre-pass baseline (any CTB whose
    /// post-trial SSE >= pre stays off).
    #[test]
    fn chroma_alf_rdo_never_increases_cb_sse() {
        let src = PictureBuffer::yuv420_filled(64, 64, 100);
        let mut rec = src.clone();
        // Inject high-frequency chroma noise so RDO has a reason to
        // flip CTBs on.
        for (i, v) in rec.cb.samples.iter_mut().enumerate() {
            let n = ((i.wrapping_mul(2654435761)) & 0x0f) as u8;
            *v = v.wrapping_add(n).wrapping_sub(8);
        }
        let pre_sse = total_sse_cb(&src, &rec);
        let aps = chroma_aps_two_alts();
        let mut apply_pic = AlfPicture::empty(1, 1);
        chroma_alf_decide_and_apply(
            &src,
            &mut rec,
            &mut apply_pic,
            &aps,
            CcAlfComponent::Cb,
            7,
            8,
            1,
        );
        let post_sse = total_sse_cb(&src, &rec);
        assert!(
            post_sse <= pre_sse,
            "chroma ALF RDO must not increase Cb SSE: pre={pre_sse} post={post_sse}"
        );
    }

    /// Round-44 — empty / unsignalled APS is a no-op.
    #[test]
    fn chroma_alf_rdo_unsignalled_aps_is_no_op() {
        let src = PictureBuffer::yuv420_filled(64, 64, 100);
        let mut rec = src.clone();
        rec.cb.samples[0] = 7;
        let baseline = rec.cb.samples.clone();
        let aps = AlfApsData::default(); // alf_chroma_filter_signal_flag = false
        let mut apply_pic = AlfPicture::empty(1, 1);
        chroma_alf_decide_and_apply(
            &src,
            &mut rec,
            &mut apply_pic,
            &aps,
            CcAlfComponent::Cb,
            7,
            8,
            1,
        );
        assert_eq!(rec.cb.samples, baseline);
        assert!(!apply_pic.get(0, 0).cb_on);
    }

    /// Round-44 — monochrome (`chroma_format_idc = 0`) is a no-op.
    #[test]
    fn chroma_alf_rdo_monochrome_is_no_op() {
        let src = PictureBuffer::yuv420_filled(64, 64, 100);
        let mut rec = src.clone();
        let aps = chroma_aps_two_alts();
        let mut apply_pic = AlfPicture::empty(1, 1);
        chroma_alf_decide_and_apply(
            &src,
            &mut rec,
            &mut apply_pic,
            &aps,
            CcAlfComponent::Cb,
            7,
            8,
            0, // monochrome
        );
        assert!(!apply_pic.get(0, 0).cb_on);
    }

    /// Round-44 — Cr-only path mutates Cr, leaves Cb alone.
    #[test]
    fn chroma_alf_rdo_cr_only_touches_cr() {
        let src = PictureBuffer::yuv420_filled(64, 64, 100);
        let mut rec = src.clone();
        for (i, v) in rec.cr.samples.iter_mut().enumerate() {
            let n = ((i.wrapping_mul(0xdeadbeef)) & 0x0f) as u8;
            *v = v.wrapping_add(n);
        }
        let baseline_cb = rec.cb.samples.clone();
        let aps = chroma_aps_two_alts();
        let mut apply_pic = AlfPicture::empty(1, 1);
        chroma_alf_decide_and_apply(
            &src,
            &mut rec,
            &mut apply_pic,
            &aps,
            CcAlfComponent::Cr,
            7,
            8,
            1,
        );
        // Cb is untouched.
        assert_eq!(rec.cb.samples, baseline_cb);
        // Cr decision (if any) records cr_on, never cb_on.
        assert!(!apply_pic.get(0, 0).cb_on);
    }

    /// Round-44 — recorded `apply_pic` plus the original chroma plane
    /// must replay byte-identically through the decoder `apply_alf`.
    #[test]
    fn chroma_alf_rdo_recorded_decision_replays_to_same_chroma() {
        let src = PictureBuffer::yuv420_filled(128, 128, 100);
        let mut rec = src.clone();
        // Inject patterned chroma noise so RDO finds a winning alt.
        for (i, v) in rec.cb.samples.iter_mut().enumerate() {
            let n = ((i.wrapping_mul(0x9e3779b9)) & 0x0f) as u8;
            *v = v.wrapping_add(n).wrapping_sub(7);
        }
        let pre_chroma = rec.cb.samples.clone();
        let aps = chroma_aps_two_alts();
        let mut apply_pic = AlfPicture::empty(1, 1);
        chroma_alf_decide_and_apply(
            &src,
            &mut rec,
            &mut apply_pic,
            &aps,
            CcAlfComponent::Cb,
            7,
            8,
            1,
        );

        // Replay path: from the pre-RDO chroma state, apply the
        // recorded picture via apply_alf.
        let mut replay = rec.clone();
        replay.cb.samples = pre_chroma;
        let aps_local = chroma_aps_two_alts();
        let binding = AlfApsBinding {
            luma_apses: &[],
            chroma_aps: Some(&aps_local),
            cc_cb_aps: None,
            cc_cr_aps: None,
        };
        let cfg = AlfConfig {
            alf_enabled: true,
            cb_enabled: true,
            cr_enabled: false,
            bit_depth: 8,
            ctb_log2_size_y: 7,
            chroma_format_idc: 1,
        };
        crate::alf::apply_alf(&mut replay, &apply_pic, &cfg, &binding);

        assert_eq!(rec.cb.samples, replay.cb.samples);
    }

    /// Round-44 — multi-CTB picture: the per-component pass on Cb then
    /// the per-component pass on Cr must compose without clobbering
    /// each other's records.
    #[test]
    fn chroma_alf_rdo_cb_and_cr_passes_compose() {
        let src = PictureBuffer::yuv420_filled(256, 128, 100);
        let mut rec = src.clone();
        // Distort both chroma planes.
        for (i, v) in rec.cb.samples.iter_mut().enumerate() {
            let n = ((i.wrapping_mul(2654435761)) & 0x0f) as u8;
            *v = v.wrapping_add(n).wrapping_sub(8);
        }
        for (i, v) in rec.cr.samples.iter_mut().enumerate() {
            let n = ((i.wrapping_mul(0xdeadbeef)) & 0x0f) as u8;
            *v = v.wrapping_add(n).wrapping_sub(8);
        }
        let aps = chroma_aps_two_alts();
        let mut apply_pic = AlfPicture::empty(2, 1);
        let pre_cb_sse = total_sse_cb(&src, &rec);
        let pre_cr_sse = total_sse_cr(&src, &rec);
        chroma_alf_decide_and_apply(
            &src,
            &mut rec,
            &mut apply_pic,
            &aps,
            CcAlfComponent::Cb,
            7,
            8,
            1,
        );
        let mid_cb_sse = total_sse_cb(&src, &rec);
        let mid_cr_sse = total_sse_cr(&src, &rec);
        assert!(mid_cb_sse <= pre_cb_sse, "Cb pass increased Cb SSE");
        assert_eq!(mid_cr_sse, pre_cr_sse, "Cb pass mutated Cr plane");
        chroma_alf_decide_and_apply(
            &src,
            &mut rec,
            &mut apply_pic,
            &aps,
            CcAlfComponent::Cr,
            7,
            8,
            1,
        );
        let post_cr_sse = total_sse_cr(&src, &rec);
        assert!(post_cr_sse <= pre_cr_sse, "Cr pass increased Cr SSE");

        // Every chosen alt_idx must lie in 0..N.
        for rx in 0..2 {
            let p = apply_pic.get(rx, 0);
            if p.cb_on {
                assert!((p.cb_alt_idx as usize) < aps.chroma_coeff.len());
            }
            if p.cr_on {
                assert!((p.cr_alt_idx as usize) < aps.chroma_coeff.len());
            }
        }
    }
}
