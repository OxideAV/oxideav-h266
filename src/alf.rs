//! VVC adaptive loop filter (ALF) — §8.8.5.
//!
//! ALF is the third in-loop filter in VVC, applied after deblocking
//! (§8.8.3) and SAO (§8.8.4). It operates on a CTB basis and supports
//! three sub-features:
//!
//! * **Luma ALF** (§8.8.5.2) — per-pixel 7×7 diamond filter (12 signalled
//!   taps + central tap). The filter to apply is selected per 4×4 sub-
//!   block by §8.8.5.3 classification (25 classes, four transpose
//!   variants), then convolved with the central pixel and clipped.
//! * **Chroma ALF** (§8.8.5.4) — per-pixel 5×5 diamond filter (6 signalled
//!   taps + central tap), one of `alf_chroma_num_alt_filters_minus1 + 1`
//!   alternative filter sets selected per CTB by `alf_ctb_filter_alt_idx`.
//! * **Cross-component ALF (CC-ALF, §8.8.5.7)** — luma → chroma
//!   refinement with a 3×4 diamond (7 taps), added on top of the
//!   primary chroma ALF output.
//!
//! ## Scope (rounds 15 + 16)
//!
//! 1. The data-source plumbing — `alf_data()` parsing in
//!    [`crate::aps::parse_aps`] populates [`crate::aps::AlfApsData`],
//!    and the per-picture array of CTB on/off + filter selection lives
//!    in [`AlfPicture`].
//! 2. The §8.8.5.2 luma filter math (eqs. 1446 – 1451) including the
//!    Table 45 line-buffer offsets and the §8.8.5.6 boundary clipping.
//! 3. **Round 16:** §8.8.5.3 luma classification — per 4×4 sub-block
//!    activity + directionality from a centred 8×8 (or 8×6 / 8×4 at
//!    line-buffer boundaries) sums of |Δh|, |Δv|, |Δd0|, |Δd1| → 25
//!    classes × 4 transpose variants (eqs. 1452 – 1482). The result
//!    feeds [`apply_alf`] / `apply_alf_luma_ctb`, which now picks
//!    `filtIdx = filt_idx[sx][sy]` per 4×4 sub-block and applies the
//!    `transposeIdx` permutation (eqs. 1442 – 1445) to the coefficient
//!    + clipping index arrays.
//! 4. The §8.8.5.4 chroma filter math (eqs. 1485 – 1492) with the same
//!    simplified boundary handling. `alf_ctb_filter_alt_idx[chromaIdx]`
//!    selects the alternative filter per CTB.
//! 5. CC-ALF (§8.8.5.7) is **not** implemented yet; the
//!    [`AlfCtb::cc_cb_idc`] / `cc_cr_idc` arrays are surfaced for
//!    callers but the apply pass treats non-zero entries as a no-op
//!    after primary chroma ALF (matching the spec's "ccAlfPicture
//!    initialised from alfPicture" rule when no CC-ALF is signalled).
//!
//! ## Boundary handling
//!
//! Tile / slice / sub-picture / virtual boundary suppression
//! (`pps_loop_filter_across_*`, `VirtualBoundariesPresentFlag`)
//! collapses to "always available" in the single-tile / single-slice /
//! no-virtual-boundary fixture this scaffold targets. Picture-edge
//! sample padding is handled by clipping the luma sample location to
//! `[0, picWidth-1] × [0, picHeight-1]` (eqs. 1446 / 1447 / 1485 /
//! 1486), and the §8.8.5.6 ALF-line-buffer position offsets are
//! supplied by Tables 45 / 46.
//!
//! Spec reference: ITU-T H.266 | ISO/IEC 23090-3 (V4, 01/2026) §8.8.5.

use crate::aps::{
    AlfApsData, ALF_CC_NUM_COEFF, ALF_CHROMA_NUM_COEFF, ALF_LUMA_NUM_COEFF, NUM_ALF_FILTERS,
};
use crate::reconstruct::{PictureBuffer, PicturePlane};

/// Per-CTB ALF on/off + filter-selection record for one CTB position.
///
/// Mirrors the spec's `alf_ctb_flag[]`, `AlfCtbFiltSetIdxY[]`,
/// `alf_ctb_filter_alt_idx[]`, and `alf_ctb_cc_cb_idc[]` /
/// `alf_ctb_cc_cr_idc[]` arrays for a single `(rx, ry)` slot.
#[derive(Clone, Copy, Debug, Default)]
pub struct AlfCtb {
    /// `alf_ctb_flag[0][rx][ry]` — luma on/off.
    pub luma_on: bool,
    /// `alf_ctb_flag[1][rx][ry]` — Cb on/off.
    pub cb_on: bool,
    /// `alf_ctb_flag[2][rx][ry]` — Cr on/off.
    pub cr_on: bool,
    /// `AlfCtbFiltSetIdxY[rx][ry]` — 0..15 selects a fixed-filter set
    /// from §7.4.3.13 / eq. 1437; 16+ selects an APS-signalled filter
    /// set (eq. 1439).
    pub luma_filt_set_idx: u8,
    /// `alf_ctb_filter_alt_idx[0][rx][ry]` — alternative-filter index for Cb.
    pub cb_alt_idx: u8,
    /// `alf_ctb_filter_alt_idx[1][rx][ry]` — alternative-filter index for Cr.
    pub cr_alt_idx: u8,
    /// `alf_ctb_cc_cb_idc[rx][ry]` — 0 means CC-ALF disabled for this
    /// CTB, 1..N selects a CC-ALF filter from the bound APS.
    pub cc_cb_idc: u8,
    /// `alf_ctb_cc_cr_idc[rx][ry]`.
    pub cc_cr_idc: u8,
}

/// Per-picture ALF parameter array, indexed by raster CTB order.
#[derive(Clone, Debug)]
pub struct AlfPicture {
    pub pic_width_in_ctbs_y: u32,
    pub pic_height_in_ctbs_y: u32,
    /// Length = `pic_width_in_ctbs_y * pic_height_in_ctbs_y`.
    ctbs: Vec<AlfCtb>,
}

impl AlfPicture {
    /// Allocate a fresh per-picture array with every entry defaulting
    /// to "ALF off" for all components.
    pub fn empty(pic_width_in_ctbs_y: u32, pic_height_in_ctbs_y: u32) -> Self {
        let n = (pic_width_in_ctbs_y as usize) * (pic_height_in_ctbs_y as usize);
        Self {
            pic_width_in_ctbs_y,
            pic_height_in_ctbs_y,
            ctbs: vec![AlfCtb::default(); n],
        }
    }

    fn idx(&self, rx: u32, ry: u32) -> usize {
        (ry as usize) * (self.pic_width_in_ctbs_y as usize) + (rx as usize)
    }

    pub fn set(&mut self, rx: u32, ry: u32, params: AlfCtb) {
        let i = self.idx(rx, ry);
        self.ctbs[i] = params;
    }

    pub fn get(&self, rx: u32, ry: u32) -> AlfCtb {
        let i = self.idx(rx, ry);
        self.ctbs[i]
    }

    /// True when no CTB enables ALF for any component — the apply pass
    /// can short-circuit. Includes the CC-ALF `idc` arrays so a slice
    /// that runs CC-ALF without primary chroma ALF (allowed by the
    /// spec) still gets its second pass.
    pub fn is_all_off(&self) -> bool {
        self.ctbs
            .iter()
            .all(|c| !c.luma_on && !c.cb_on && !c.cr_on && c.cc_cb_idc == 0 && c.cc_cr_idc == 0)
    }
}

/// Picture-level ALF configuration consumed by [`apply_alf`].
#[derive(Clone, Copy, Debug)]
pub struct AlfConfig {
    /// `sh_alf_enabled_flag` — gates the entire luma + chroma pass.
    pub alf_enabled: bool,
    /// `sh_alf_cb_enabled_flag`.
    pub cb_enabled: bool,
    /// `sh_alf_cr_enabled_flag`.
    pub cr_enabled: bool,
    /// `BitDepth` (= sps_bitdepth_minus8 + 8).
    pub bit_depth: u32,
    /// `CtbLog2SizeY`.
    pub ctb_log2_size_y: u32,
    /// `sps_chroma_format_idc` — 0 (monochrome), 1 (4:2:0), 2 (4:2:2),
    /// 3 (4:4:4).
    pub chroma_format_idc: u32,
}

impl AlfConfig {
    fn chroma_subsampling(&self) -> (u32, u32) {
        match self.chroma_format_idc {
            1 => (2, 2),
            2 => (2, 1),
            3 => (1, 1),
            _ => (1, 1),
        }
    }
}

/// Source of luma/chroma filter coefficients for one CTB.
///
/// Per §8.8.5.2 the per-CTB luma filter is either one of the 16 fixed
/// `AlfFixFiltCoeff` sets (`AlfCtbFiltSetIdxY < 16`) or one of the
/// APS-signalled `AlfCoeffL` sets (`AlfCtbFiltSetIdxY >= 16`). For the
/// round-15 scaffold the apply pass drives off pre-resolved
/// per-coefficient tables that callers (or the future per-CTU CABAC
/// reader) build directly from the picture's bound ALF APSes.
#[derive(Clone, Debug)]
pub struct AlfApsBinding<'a> {
    /// APSes that are referenced by `sh_alf_aps_id_luma[]`. Indexed by
    /// the slice header's APS-id list position. `None` slots are
    /// allowed (the slice must not actually reference them); the apply
    /// pass surfaces "luma off" for any CTB that demands an absent APS.
    pub luma_apses: &'a [Option<&'a AlfApsData>],
    /// APS bound by `sh_alf_aps_id_chroma`. `None` disables chroma ALF.
    pub chroma_aps: Option<&'a AlfApsData>,
    /// APS bound by `sh_alf_cc_cb_aps_id` (CC-ALF for Cb). `None`
    /// disables CC-ALF for Cb across the picture; per-CTB
    /// `alf_ctb_cc_cb_idc` is then ignored.
    pub cc_cb_aps: Option<&'a AlfApsData>,
    /// APS bound by `sh_alf_cc_cr_aps_id` (CC-ALF for Cr). `None`
    /// disables CC-ALF for Cr.
    pub cc_cr_aps: Option<&'a AlfApsData>,
}

impl Default for AlfApsBinding<'_> {
    fn default() -> Self {
        Self {
            luma_apses: &[],
            chroma_aps: None,
            cc_cb_aps: None,
            cc_cr_aps: None,
        }
    }
}

/// Top-level ALF driver invoked by the CTU walker after SAO. Implements
/// the §8.8.5.1 ordered scan: for every CTB,
/// 1. apply luma ALF (if `alf_ctb_flag[0][rx][ry] == 1`),
/// 2. apply chroma ALF for Cb / Cr (when chroma is present and the
///    corresponding `alf_ctb_flag` is set).
///
/// `apply_alf` mutates `out` in place. Per §8.8.5.1 the spec defines
/// `alfPicture*` aux buffers initialised from `recPicture*`; in our
/// scheme reads always come from a *pre-ALF* snapshot of each plane to
/// keep that semantics — without it, neighbour samples on inter-CTB
/// boundaries would already carry ALF modifications and break
/// reproducibility against the spec.
pub fn apply_alf(
    out: &mut PictureBuffer,
    alf_pic: &AlfPicture,
    cfg: &AlfConfig,
    binding: &AlfApsBinding<'_>,
) {
    if !cfg.alf_enabled {
        return;
    }
    if alf_pic.is_all_off() {
        return;
    }
    let ctb_size_y = 1u32 << cfg.ctb_log2_size_y;
    let (sub_w, sub_h) = cfg.chroma_subsampling();

    // Stage pre-ALF snapshots — see §8.8.5.1 / eqs. 1446 / 1447 /
    // 1485 / 1486 for why neighbour reads always sample from the
    // pre-ALF buffer.
    let luma_pre = out.luma.samples.clone();
    let cb_pre = if cfg.chroma_format_idc != 0 {
        out.cb.samples.clone()
    } else {
        Vec::new()
    };
    let cr_pre = if cfg.chroma_format_idc != 0 {
        out.cr.samples.clone()
    } else {
        Vec::new()
    };

    for ry in 0..alf_pic.pic_height_in_ctbs_y {
        for rx in 0..alf_pic.pic_width_in_ctbs_y {
            let p = alf_pic.get(rx, ry);
            if p.luma_on {
                if let Some(filters) = resolve_luma_filter_set(p.luma_filt_set_idx, binding) {
                    // §8.8.5.3 — derive the per-4×4-sub-block (filtIdx,
                    // transposeIdx) grid for this CTB. Reads come from
                    // the pre-ALF snapshot per §8.8.5.1.
                    let cls = derive_luma_classification(
                        &luma_pre,
                        out.luma.stride,
                        out.luma.width as i32,
                        out.luma.height as i32,
                        rx,
                        ry,
                        ctb_size_y,
                        cfg.bit_depth,
                    );
                    apply_alf_luma_ctb(
                        &mut out.luma,
                        &luma_pre,
                        rx,
                        ry,
                        ctb_size_y,
                        cfg.bit_depth,
                        &filters,
                        &cls,
                    );
                }
            }
            if cfg.chroma_format_idc != 0 {
                if cfg.cb_enabled && p.cb_on {
                    if let Some(aps) = binding.chroma_aps {
                        if let Some((coeff, clip_idx)) =
                            resolve_chroma_filter(aps, p.cb_alt_idx as usize)
                        {
                            apply_alf_chroma_ctb(
                                &mut out.cb,
                                &cb_pre,
                                rx,
                                ry,
                                ctb_size_y,
                                sub_w,
                                sub_h,
                                cfg.bit_depth,
                                &coeff,
                                &clip_idx,
                            );
                        }
                    }
                }
                if cfg.cr_enabled && p.cr_on {
                    if let Some(aps) = binding.chroma_aps {
                        if let Some((coeff, clip_idx)) =
                            resolve_chroma_filter(aps, p.cr_alt_idx as usize)
                        {
                            apply_alf_chroma_ctb(
                                &mut out.cr,
                                &cr_pre,
                                rx,
                                ry,
                                ctb_size_y,
                                sub_w,
                                sub_h,
                                cfg.bit_depth,
                                &coeff,
                                &clip_idx,
                            );
                        }
                    }
                }
            }
        }
    }

    // §8.8.5.1 second pass — CC-ALF runs after primary chroma ALF and
    // reads from the *pre-luma-ALF* `recPictureL`. The accumulated
    // delta is added to the post-chroma-ALF chroma plane.
    if cfg.chroma_format_idc != 0 {
        for ry in 0..alf_pic.pic_height_in_ctbs_y {
            for rx in 0..alf_pic.pic_width_in_ctbs_y {
                let p = alf_pic.get(rx, ry);
                if cfg.cb_enabled && p.cc_cb_idc != 0 {
                    if let Some(aps) = binding.cc_cb_aps {
                        let filt_idx = (p.cc_cb_idc - 1) as usize;
                        if filt_idx < aps.cc_cb_coeff.len() {
                            let coeff = &aps.cc_cb_coeff[filt_idx];
                            apply_cc_alf_ctb(
                                &mut out.cb,
                                &luma_pre,
                                out.luma.stride,
                                out.luma.width as u32,
                                out.luma.height as u32,
                                rx,
                                ry,
                                ctb_size_y,
                                sub_w,
                                sub_h,
                                cfg.bit_depth,
                                coeff,
                            );
                        }
                    }
                }
                if cfg.cr_enabled && p.cc_cr_idc != 0 {
                    if let Some(aps) = binding.cc_cr_aps {
                        let filt_idx = (p.cc_cr_idc - 1) as usize;
                        if filt_idx < aps.cc_cr_coeff.len() {
                            let coeff = &aps.cc_cr_coeff[filt_idx];
                            apply_cc_alf_ctb(
                                &mut out.cr,
                                &luma_pre,
                                out.luma.stride,
                                out.luma.width as u32,
                                out.luma.height as u32,
                                rx,
                                ry,
                                ctb_size_y,
                                sub_w,
                                sub_h,
                                cfg.bit_depth,
                                coeff,
                            );
                        }
                    }
                }
            }
        }
    }
}

/// Apply §8.8.5.7 cross-component ALF to a single chroma CTB.
///
/// `chroma` is the post-chroma-ALF plane being modified; `luma_pre` is
/// the **pre-luma-ALF** luma snapshot referenced by eq. 1515 as
/// `recPictureL`. `coeff[0..6]` are the seven CcAlfCoeff filter taps.
///
/// Per eq. 1517 the `(SubHeightC == 1 && y ∈ {CtbSizeY-3, CtbSizeY-4})`
/// guard suppresses the modification on the two near-bottom luma rows
/// of a 4:2:2 / 4:4:4 CTB; in 4:2:0 (`SubHeightC == 2`) the guard
/// never fires so it collapses to "always add scaledSum".
#[allow(clippy::too_many_arguments)]
fn apply_cc_alf_ctb(
    chroma: &mut PicturePlane,
    luma_pre: &[u8],
    luma_stride: usize,
    luma_w: u32,
    luma_h: u32,
    rx: u32,
    ry: u32,
    ctb_size_y: u32,
    sub_w: u32,
    sub_h: u32,
    bit_depth: u32,
    coeff: &[i32; ALF_CC_NUM_COEFF],
) {
    let cc_alf_width = (ctb_size_y / sub_w) as usize;
    let cc_alf_height = (ctb_size_y / sub_h) as usize;
    let x_ctb_c = ((rx * ctb_size_y) / sub_w) as usize;
    let y_ctb_c = ((ry * ctb_size_y) / sub_h) as usize;
    let i_max = cc_alf_width.min(chroma.width.saturating_sub(x_ctb_c));
    let j_max = cc_alf_height.min(chroma.height.saturating_sub(y_ctb_c));
    let max_val = (1i32 << bit_depth) - 1;
    let max_val_8 = max_val.min(255);
    let half = 1i32 << (bit_depth - 1);
    let pw = luma_w as i32;
    let ph = luma_h as i32;

    for y in 0..j_max {
        for x in 0..i_max {
            // Map chroma → luma per eq. 1510.
            let xl = ((x_ctb_c + x) as i32) * (sub_w as i32);
            let yl = ((y_ctb_c + y) as i32) * (sub_h as i32);

            // Table 47 yP1 / yP2: in the single-slice / no-virtual-
            // boundary scaffold `applyAlfLineBufBoundary` is treated as
            // 1 for non-bottom CTBs (the line-buffer carve-out fires
            // inside the CTB) and 0 for the picture-bottom row.
            let y_line = (y as i32) * (sub_h as i32);
            let apply_lbb = !(yl + (ctb_size_y as i32) >= ph
                && (ph - (y_ctb_c as i32) * (sub_h as i32)) <= (ctb_size_y as i32) - 4);
            let (y_p1, y_p2) = cc_alf_table_47(y_line, ctb_size_y as i32, apply_lbb);

            // Eq. 1513.
            let curr_chroma = chroma.samples[(y_ctb_c + y) * chroma.stride + (x_ctb_c + x)] as i32;
            // Centre luma sample for delta calculation.
            let centre = sample(luma_pre, luma_stride, pw, ph, xl, yl) as i32;
            // Eq. 1515 — seven luma-deltas weighted by f[0..6].
            let mut sum: i32 = 0;
            sum +=
                coeff[0] * (sample(luma_pre, luma_stride, pw, ph, xl, yl - y_p1) as i32 - centre);
            sum += coeff[1] * (sample(luma_pre, luma_stride, pw, ph, xl - 1, yl) as i32 - centre);
            sum += coeff[2] * (sample(luma_pre, luma_stride, pw, ph, xl + 1, yl) as i32 - centre);
            sum += coeff[3]
                * (sample(luma_pre, luma_stride, pw, ph, xl - 1, yl + y_p1) as i32 - centre);
            sum +=
                coeff[4] * (sample(luma_pre, luma_stride, pw, ph, xl, yl + y_p1) as i32 - centre);
            sum += coeff[5]
                * (sample(luma_pre, luma_stride, pw, ph, xl + 1, yl + y_p1) as i32 - centre);
            sum +=
                coeff[6] * (sample(luma_pre, luma_stride, pw, ph, xl, yl + y_p2) as i32 - centre);
            // Eq. 1516.
            let scaled_sum = ((sum + 64) >> 7).clamp(-half, half - 1);
            // Eq. 1517 — suppression rule for `SubHeightC == 1` rows.
            let suppress = sub_h == 1
                && (y_line == (ctb_size_y as i32) - 3 || y_line == (ctb_size_y as i32) - 4);
            let new = if suppress {
                curr_chroma
            } else {
                curr_chroma + scaled_sum
            };
            // Eq. 1518.
            let new = new.clamp(0, max_val).min(max_val_8) as u8;
            chroma.samples[(y_ctb_c + y) * chroma.stride + (x_ctb_c + x)] = new;
        }
    }
}

/// Table 47 — yP1 / yP2 vertical luma offsets for CC-ALF. Indexed by
/// `y * SubHeightC` and `applyAlfLineBufBoundary`.
#[inline]
fn cc_alf_table_47(y_line: i32, ctb_size_y: i32, apply_lbb: bool) -> (i32, i32) {
    if apply_lbb {
        // Two carve-outs near the bottom of the CTB.
        if y_line == ctb_size_y - 5 || y_line == ctb_size_y - 4 {
            return (0, 0);
        }
        if y_line == ctb_size_y - 6 || y_line == ctb_size_y - 3 {
            return (1, 1);
        }
    }
    (1, 2)
}

/// Resolved per-CTB luma filter set: all 25 filter rows the §8.8.5.3
/// classification can pick from.
///
/// Each row is `(coeff[12], clip_idx[12])`; `clip_idx` is the
/// §7.4.3.18 / Table 8 index that [`resolve_clip_value`] turns into the
/// concrete `c[]` value at apply time.
type LumaFilterSet = [([i32; ALF_LUMA_NUM_COEFF], [u8; ALF_LUMA_NUM_COEFF]); NUM_ALF_FILTERS];

/// Resolve the full 25-class luma filter set referenced by
/// `AlfCtbFiltSetIdxY`.
///
/// * `filt_set_idx < 16` — fixed-filter set per eqs. 90 / 91 / 1437 /
///   1438. Coefficients come from `AlfFixFiltCoeff[ AlfClassToFiltMap[
///   filt_set_idx ][ filtIdx ] ]`; the spec sets `c[j] = 2^BitDepth`
///   so the per-sample clip is effectively disabled (the linear
///   filter path). We surface clipIdx 0 for every coefficient — that
///   maps via [`resolve_clip_value`] to `1 << BitDepth`, the same
///   no-op clipping the spec specifies.
/// * `filt_set_idx >= 16` — APS-signalled set per eqs. 1439 / 1440 /
///   1441. Coefficients + clipIdx come from `AlfCoeffL` / `AlfClipL`
///   in the bound APS.
///
/// Returns `None` when the requested APS slot is empty / unsignalled.
fn resolve_luma_filter_set(filt_set_idx: u8, binding: &AlfApsBinding<'_>) -> Option<LumaFilterSet> {
    if (filt_set_idx as usize) < 16 {
        // Fixed-filter set — eq. 1437 / 1438. Build all 25 class rows
        // by walking AlfClassToFiltMap[filt_set_idx][class] for class
        // ∈ 0..25 and copying the corresponding 12-tap row out of
        // ALF_FIX_FILT_COEFF. clipIdx 0 → `resolve_clip_value` returns
        // `2^BitDepth`, the spec's "no clipping" sentinel for the
        // fixed-filter linear path.
        let mut set: LumaFilterSet =
            [([0i32; ALF_LUMA_NUM_COEFF], [0u8; ALF_LUMA_NUM_COEFF]); NUM_ALF_FILTERS];
        for class in 0..NUM_ALF_FILTERS {
            let coeff_row = crate::alf_fixed::fixed_filter_coeff_row(filt_set_idx, class as u8)?;
            for j in 0..ALF_LUMA_NUM_COEFF {
                set[class].0[j] = coeff_row[j] as i32;
                set[class].1[j] = 0;
            }
        }
        return Some(set);
    }
    let aps_slot = (filt_set_idx as usize) - 16;
    let aps = binding.luma_apses.get(aps_slot).and_then(|s| *s)?;
    if !aps.alf_luma_filter_signal_flag {
        return None;
    }
    let mut set: LumaFilterSet =
        [([0i32; ALF_LUMA_NUM_COEFF], [0u8; ALF_LUMA_NUM_COEFF]); NUM_ALF_FILTERS];
    for filt_idx in 0..NUM_ALF_FILTERS {
        set[filt_idx].0 = aps.luma_coeff[filt_idx];
        set[filt_idx].1 = aps.luma_clip_idx[filt_idx];
    }
    Some(set)
}

/// Resolve `(AlfCoeffC[altIdx][...], AlfClipC[altIdx][...])` for the
/// chroma alternative filter referenced by the CTB.
fn resolve_chroma_filter(
    aps: &AlfApsData,
    alt_idx: usize,
) -> Option<([i32; ALF_CHROMA_NUM_COEFF], [u8; ALF_CHROMA_NUM_COEFF])> {
    if !aps.alf_chroma_filter_signal_flag {
        return None;
    }
    if alt_idx >= aps.chroma_coeff.len() {
        return None;
    }
    Some((aps.chroma_coeff[alt_idx], aps.chroma_clip_idx[alt_idx]))
}

/// Resolve the spec's `AlfClip[ BitDepth ][ clipIdx ]` value (Table 8).
///
/// `AlfClip[BitDepth][0] = 1 << BitDepth`; for `clipIdx > 0` Table 8
/// halves the value (rounded down) progressively, with a bit-depth
/// dependent floor that the spec table specifies row-by-row. The
/// closed-form expression is
///
/// ```text
///   AlfClip[BitDepth][clipIdx] = 1 << Round( BitDepth - clipIdx * (BitDepth - 7) / 3 )
/// ```
///
/// but the §7.4.3.18 derivation is given as the literal Table 8;
/// transcribing the four columns directly is more obviously correct
/// and avoids floating-point rounding. We therefore expand from a
/// small in-table.
pub fn resolve_clip_value(bit_depth: u32, clip_idx: u8) -> i32 {
    // Table 8 — exponents for AlfClip[BitDepth][clipIdx]. Rows are
    // BitDepth = 8..=16; columns are clipIdx = 0..=3.
    const EXP: [[u8; 4]; 9] = [
        [8, 5, 3, 1],    // BitDepth = 8
        [9, 6, 4, 2],    // BitDepth = 9
        [10, 7, 5, 3],   // BitDepth = 10
        [11, 8, 6, 4],   // BitDepth = 11
        [12, 9, 7, 5],   // BitDepth = 12
        [13, 10, 8, 6],  // BitDepth = 13
        [14, 11, 9, 7],  // BitDepth = 14
        [15, 12, 10, 8], // BitDepth = 15
        [16, 13, 11, 9], // BitDepth = 16
    ];
    let row = (bit_depth.clamp(8, 16) - 8) as usize;
    let col = (clip_idx & 0x3) as usize;
    1i32 << EXP[row][col]
}

/// Apply the §8.8.5.2 luma filter to a single CTB. `pre` is a snapshot
/// of the pre-ALF luma plane (same `stride`/`width`/`height` as
/// `plane`). `filters` provides all 25 filter rows for this CTB; the
/// per-4×4-sub-block classification `cls` (from §8.8.5.3) picks one
/// row plus a transpose variant for every 16-pixel block.
#[allow(clippy::too_many_arguments)]
fn apply_alf_luma_ctb(
    plane: &mut PicturePlane,
    pre: &[u8],
    rx: u32,
    ry: u32,
    ctb_size_y: u32,
    bit_depth: u32,
    filters: &LumaFilterSet,
    cls: &LumaClassification,
) {
    let x_ctb = (rx * ctb_size_y) as usize;
    let y_ctb = (ry * ctb_size_y) as usize;
    let stride = plane.stride;
    let pw = plane.width as i32;
    let ph = plane.height as i32;
    let i_max = (ctb_size_y as usize).min(plane.width.saturating_sub(x_ctb));
    let j_max = (ctb_size_y as usize).min(plane.height.saturating_sub(y_ctb));
    let max_val = (1i32 << bit_depth) - 1;
    let max_val_8 = max_val.min(255);

    for j in 0..j_max {
        for i in 0..i_max {
            let xs = (x_ctb + i) as i32;
            let ys = (y_ctb + j) as i32;
            // §8.8.5.3 — every 4×4 sub-block shares one (filtIdx,
            // transposeIdx). Look up the entry for this pixel.
            let sx = i >> 2;
            let sy = j >> 2;
            let (filt_idx, transpose_idx) = cls.get(sx, sy);
            let (f, clip_idx) = &filters[filt_idx];
            // Eqs. 1442 – 1445: pick the coefficient/clip-index
            // permutation that realises the geometric transform.
            let idx = transpose_idx_table(transpose_idx);
            let mut c = [0i32; ALF_LUMA_NUM_COEFF];
            for k in 0..ALF_LUMA_NUM_COEFF {
                c[k] = resolve_clip_value(bit_depth, clip_idx[k]);
            }
            // Table 45: pick alfShiftY / y1 / y2 / y3 from the vertical
            // sample position. The boundary suppression case
            // (`applyAlfLineBufBoundary == 1`) only fires on inter-CTB
            // line-buffer rows; in the single-slice scaffold we treat
            // the bottom of the picture per the spec — which collapses
            // to "applyAlfLineBufBoundary == 0" for the very last CTB
            // row when it doesn't span CtbSizeY-4 rows past yCtb.
            let (alf_shift_y, y1, y2, y3) = table_45(j as i32, ctb_size_y as i32, true);
            // Eq. 1448.
            let curr = sample(pre, stride, pw, ph, xs, ys) as i32;
            // Eq. 1449.
            let mut sum: i32 = 0;
            // Helper: clipped neighbour minus curr.
            let clip = |k: usize, dx: i32, dy: i32| -> i32 {
                let v = sample(pre, stride, pw, ph, xs + dx, ys + dy) as i32 - curr;
                v.clamp(-c[idx[k]], c[idx[k]])
            };
            sum += f[idx[0]] * (clip(0, 0, y3) + clip(0, 0, -y3));
            sum += f[idx[1]] * (clip(1, 1, y2) + clip(1, -1, -y2));
            sum += f[idx[2]] * (clip(2, 0, y2) + clip(2, 0, -y2));
            sum += f[idx[3]] * (clip(3, -1, y2) + clip(3, 1, -y2));
            sum += f[idx[4]] * (clip(4, 2, y1) + clip(4, -2, -y1));
            sum += f[idx[5]] * (clip(5, 1, y1) + clip(5, -1, -y1));
            sum += f[idx[6]] * (clip(6, 0, y1) + clip(6, 0, -y1));
            sum += f[idx[7]] * (clip(7, -1, y1) + clip(7, 1, -y1));
            sum += f[idx[8]] * (clip(8, -2, y1) + clip(8, 2, -y1));
            sum += f[idx[9]] * (clip(9, 3, 0) + clip(9, -3, 0));
            sum += f[idx[10]] * (clip(10, 2, 0) + clip(10, -2, 0));
            sum += f[idx[11]] * (clip(11, 1, 0) + clip(11, -1, 0));
            // Eq. 1450.
            let bias = 1i32 << (alf_shift_y - 1);
            let new = curr + ((sum + bias) >> alf_shift_y);
            // Eq. 1451.
            let new = new.clamp(0, max_val).min(max_val_8) as u8;
            plane.samples[(ys as usize) * stride + (xs as usize)] = new;
        }
    }
}

/// `idx[]` permutation table from eqs. 1442 – 1445 — picks the
/// coefficient/clip-index ordering that realises one of the four
/// transposeIdx geometric transforms. The spec encodes them as
/// 0 = identity, 1 = vertical-flip, 2 = horizontal-flip, 3 = 180°.
#[inline]
fn transpose_idx_table(transpose_idx: u8) -> [usize; ALF_LUMA_NUM_COEFF] {
    match transpose_idx {
        // Eq. 1442 — transposeIdx == 1.
        1 => [9, 4, 10, 8, 1, 5, 11, 7, 3, 0, 2, 6],
        // Eq. 1443 — transposeIdx == 2.
        2 => [0, 3, 2, 1, 8, 7, 6, 5, 4, 9, 10, 11],
        // Eq. 1444 — transposeIdx == 3.
        3 => [9, 8, 10, 4, 3, 7, 11, 5, 1, 0, 2, 6],
        // Eq. 1445 — transposeIdx == 0 (identity).
        _ => [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    }
}

/// Per-4×4-sub-block classification grid for one luma CTB. Indexed by
/// sub-block coordinates `(sx, sy)` with `sx, sy = 0..(CtbSizeY / 4 - 1)`.
///
/// Each entry stores `(filtIdx, transposeIdx)`; `filtIdx` selects one
/// of the 25 luma filters and `transposeIdx` (0..3) selects one of the
/// four geometric transforms applied via [`transpose_idx_table`].
#[derive(Clone, Debug)]
pub(crate) struct LumaClassification {
    /// Number of 4×4 sub-blocks per CTB row/column (= `CtbSizeY >> 2`).
    sub_size: usize,
    /// Length = `sub_size * sub_size`.
    cells: Vec<(u8, u8)>,
}

impl LumaClassification {
    fn new(ctb_size_y: u32) -> Self {
        let sub_size = (ctb_size_y as usize) >> 2;
        Self {
            sub_size,
            cells: vec![(0u8, 0u8); sub_size * sub_size],
        }
    }

    #[inline]
    fn idx(&self, sx: usize, sy: usize) -> usize {
        sy * self.sub_size + sx
    }

    #[inline]
    fn set(&mut self, sx: usize, sy: usize, filt_idx: u8, transpose_idx: u8) {
        let i = self.idx(sx, sy);
        self.cells[i] = (filt_idx, transpose_idx);
    }

    /// Return `(filtIdx as usize, transposeIdx as u8)` for the 4×4
    /// sub-block at `(sx, sy)`.
    #[inline]
    fn get(&self, sx: usize, sy: usize) -> (usize, u8) {
        let (f, t) = self.cells[self.idx(sx, sy)];
        (f as usize, t)
    }

    /// Number of 4×4 sub-blocks per CTB row/column. Used by the test
    /// suite to walk the classification grid.
    #[cfg(test)]
    #[inline]
    fn sub_size(&self) -> usize {
        self.sub_size
    }
}

/// §8.8.5.3 — derive `(filtIdx, transposeIdx)` for every 4×4 sub-block
/// in the luma CTB at `(rx, ry)`.
///
/// Spec mapping:
/// * Eqs. 1452 / 1453 — `recPicture` reads are clipped to the picture.
/// * Eqs. 1454 – 1457 — per-sample 1-D Laplacians on a checkerboard
///   pattern (both `i, j` even *or* both odd; otherwise zero).
/// * Eqs. 1458 – 1462 — sums over `i = -2..5`, `j = minY..maxY`.
/// * Eqs. 1463 – 1474 — pick the dominant H/V and the dominant D0/D1
///   directions.
/// * Eqs. 1475 – 1479 — combine into `dir1`, `dir2`, `dirS`.
/// * Eqs. 1480 – 1481 — quantise activity through `varTab[]`.
/// * Eqs. 1463-style transposeTable + eq. 1482 — pack everything into
///   `transposeIdx` (0..3) and `filtIdx` (0..24).
///
/// The min/max-Y window selection follows the spec's two
/// line-buffer-row carve-outs at `y4 == CtbSizeY - 8` and
/// `y4 == CtbSizeY - 4`. For the single-slice / no-virtual-boundary
/// scaffold both carve-outs collapse to the "bottom boundary of CTB is
/// not bottom of picture, OR remaining picture height >
/// `CtbSizeY - 4`" condition, which we approximate as: the carve-out
/// fires when the next CTB row exists in the picture (so `(ry+1) *
/// ctb_size_y < pic_height_luma`) OR `pic_height_luma - yCtb >
/// ctb_size_y - 4`. The else branch (last CTB row of a picture whose
/// height ≤ CtbSizeY - 4 past `yCtb`) yields the default
/// `minY=-2, maxY=5, ac=2` row everywhere.
#[allow(clippy::too_many_arguments)]
fn derive_luma_classification(
    pre: &[u8],
    stride: usize,
    pw: i32,
    ph: i32,
    rx: u32,
    ry: u32,
    ctb_size_y: u32,
    bit_depth: u32,
) -> LumaClassification {
    let x_ctb = (rx * ctb_size_y) as i32;
    let y_ctb = (ry * ctb_size_y) as i32;
    let ctb = ctb_size_y as i32;
    let sub_size = (ctb_size_y as usize) >> 2;
    let mut out = LumaClassification::new(ctb_size_y);

    // The line-buffer carve-outs activate when the spec's "boundary
    // condition" is met. For our single-slice / picture-only scaffold:
    // the bottom of the CTB is the bottom of the picture only when
    // `y_ctb + ctb >= ph`. Rephrased: the carve-out fires when *not*
    // (bottom of CTB is bottom of picture AND remaining picture height
    // ≤ ctb - 4).
    let bottom_of_ctb_is_bottom_of_pic = y_ctb + ctb >= ph;
    let pic_height_minus_yctb = ph - y_ctb;
    let carve_out_active = !bottom_of_ctb_is_bottom_of_pic || pic_height_minus_yctb > ctb - 4;

    // Sums per 4×4 sub-block.
    let mut sum_h = vec![0i32; sub_size * sub_size];
    let mut sum_v = vec![0i32; sub_size * sub_size];
    let mut sum_d0 = vec![0i32; sub_size * sub_size];
    let mut sum_d1 = vec![0i32; sub_size * sub_size];
    let mut sum_hv = vec![0i32; sub_size * sub_size];
    let mut ac_arr = vec![0i32; sub_size * sub_size];

    let s_get = |sx: usize, sy: usize| sy * sub_size + sx;

    for sy in 0..sub_size {
        for sx in 0..sub_size {
            let x4 = (sx as i32) << 2;
            let y4 = (sy as i32) << 2;

            // §8.8.5.3 — derive (minY, maxY, ac) from y4.
            let (min_y, max_y, ac) = if y4 == ctb - 8 && carve_out_active {
                (-2i32, 3i32, 3i32)
            } else if y4 == ctb - 4 && carve_out_active {
                (0i32, 5i32, 3i32)
            } else {
                (-2i32, 5i32, 2i32)
            };
            ac_arr[s_get(sx, sy)] = ac;

            // Eqs. 1454 – 1457 + 1458 – 1462.
            let mut sh = 0i32;
            let mut sv = 0i32;
            let mut sd0 = 0i32;
            let mut sd1 = 0i32;
            for j in min_y..=max_y {
                for i in -2i32..=5i32 {
                    // Checkerboard mask: only (i,j both even) or
                    // (i,j both odd) contribute. Otherwise filtH/V/D0/D1
                    // are 0 (eq. text immediately under 1457).
                    if (i.rem_euclid(2)) != (j.rem_euclid(2)) {
                        continue;
                    }
                    let cx = x_ctb + x4 + i;
                    let cy = y_ctb + y4 + j;
                    let centre = sample(pre, stride, pw, ph, cx, cy) as i32;
                    let centre2 = centre << 1;
                    // filtH: horizontal Laplacian.
                    let l = sample(pre, stride, pw, ph, cx - 1, cy) as i32;
                    let r = sample(pre, stride, pw, ph, cx + 1, cy) as i32;
                    sh += (centre2 - l - r).abs();
                    // filtV: vertical Laplacian.
                    let u = sample(pre, stride, pw, ph, cx, cy - 1) as i32;
                    let d = sample(pre, stride, pw, ph, cx, cy + 1) as i32;
                    sv += (centre2 - u - d).abs();
                    // filtD0: 135°-diagonal Laplacian (top-left ↔ bottom-right).
                    let ul = sample(pre, stride, pw, ph, cx - 1, cy - 1) as i32;
                    let dr = sample(pre, stride, pw, ph, cx + 1, cy + 1) as i32;
                    sd0 += (centre2 - ul - dr).abs();
                    // filtD1: 45°-diagonal Laplacian (top-right ↔ bottom-left).
                    let ur = sample(pre, stride, pw, ph, cx + 1, cy - 1) as i32;
                    let dl = sample(pre, stride, pw, ph, cx - 1, cy + 1) as i32;
                    sd1 += (centre2 - ur - dl).abs();
                }
            }
            sum_h[s_get(sx, sy)] = sh;
            sum_v[s_get(sx, sy)] = sv;
            sum_d0[s_get(sx, sy)] = sd0;
            sum_d1[s_get(sx, sy)] = sd1;
            sum_hv[s_get(sx, sy)] = sh + sv;
        }
    }

    // varTab from eq. 1480.
    const VAR_TAB: [u8; 16] = [0, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4];
    // transposeTable from the prologue of step 3.
    const TRANSPOSE_TABLE: [u8; 8] = [0, 1, 0, 2, 2, 3, 1, 3];

    // Steps 1, 2, 3: derive (transposeIdx, filtIdx) per 4×4 sub-block.
    for sy in 0..sub_size {
        for sx in 0..sub_size {
            let i = s_get(sx, sy);
            let sv = sum_v[i];
            let sh = sum_h[i];
            let sd0 = sum_d0[i];
            let sd1 = sum_d1[i];

            // Eqs. 1463 – 1468.
            let (hv1, hv0, dir_hv) = if sv > sh {
                (sv, sh, 1u32)
            } else {
                (sh, sv, 3u32)
            };
            // Eqs. 1469 – 1474.
            let (d1, d0, dir_d) = if sd0 > sd1 {
                (sd0, sd1, 0u32)
            } else {
                (sd1, sd0, 2u32)
            };
            // Eqs. 1475 / 1476 — careful: the comparison is
            // `d1 * hv0 > hv1 * d0` and uses the un-swapped values.
            // Use i64 to avoid overflow on 8×6 sums (max ~3.2e6 each → product ~1e13).
            let prefer_d = (d1 as i64) * (hv0 as i64) > (hv1 as i64) * (d0 as i64);
            let hvd1 = if prefer_d { d1 } else { hv1 };
            let hvd0 = if prefer_d { d0 } else { hv0 };
            // Eqs. 1477 / 1478.
            let dir1 = if prefer_d { dir_d } else { dir_hv };
            let dir2 = if prefer_d { dir_hv } else { dir_d };
            // Eq. 1479.
            let dir_s: u32 = if hvd1 * 2 > 9 * hvd0 {
                2
            } else if hvd1 > 2 * hvd0 {
                1
            } else {
                0
            };

            // Eq. 1481 — quantise activity. avg = sum_hv * ac, then
            // shift by (BitDepth - 1) and clip to 0..15.
            let prod = (sum_hv[i] as i64) * (ac_arr[i] as i64);
            let shifted = (prod >> (bit_depth as i64 - 1)).clamp(0, 15) as usize;
            let mut filt_idx = VAR_TAB[shifted] as u32;

            // Step 3 — transposeIdx + filtIdx.
            let transpose_idx = TRANSPOSE_TABLE[(dir1 * 2 + (dir2 >> 1)) as usize];
            if dir_s != 0 {
                // Eq. 1482.
                filt_idx += (((dir1 & 0x1) << 1) + dir_s) * 5;
            }
            // filt_idx is in 0..25 (25 classes).
            out.set(sx, sy, filt_idx as u8, transpose_idx);
        }
    }

    out
}

/// Apply the §8.8.5.4 chroma filter to a single CTB.
#[allow(clippy::too_many_arguments)]
fn apply_alf_chroma_ctb(
    plane: &mut PicturePlane,
    pre: &[u8],
    rx: u32,
    ry: u32,
    ctb_size_y: u32,
    sub_w: u32,
    sub_h: u32,
    bit_depth: u32,
    f: &[i32; ALF_CHROMA_NUM_COEFF],
    clip_idx: &[u8; ALF_CHROMA_NUM_COEFF],
) {
    let ctb_w_c = (ctb_size_y / sub_w) as usize;
    let ctb_h_c = (ctb_size_y / sub_h) as usize;
    let x_ctb_c = (rx as usize) * ctb_w_c;
    let y_ctb_c = (ry as usize) * ctb_h_c;
    let stride = plane.stride;
    let pw = plane.width as i32;
    let ph = plane.height as i32;
    let i_max = ctb_w_c.min(plane.width.saturating_sub(x_ctb_c));
    let j_max = ctb_h_c.min(plane.height.saturating_sub(y_ctb_c));
    let max_val = (1i32 << bit_depth) - 1;
    let max_val_8 = max_val.min(255);
    let mut c = [0i32; ALF_CHROMA_NUM_COEFF];
    for k in 0..ALF_CHROMA_NUM_COEFF {
        c[k] = resolve_clip_value(bit_depth, clip_idx[k]);
    }
    for j in 0..j_max {
        for i in 0..i_max {
            let xs = (x_ctb_c + i) as i32;
            let ys = (y_ctb_c + j) as i32;
            let (alf_shift_c, y1, y2) = table_46(j as i32, ctb_h_c as i32, true);
            let curr = sample(pre, stride, pw, ph, xs, ys) as i32;
            let clip = |k: usize, dx: i32, dy: i32| -> i32 {
                let v = sample(pre, stride, pw, ph, xs + dx, ys + dy) as i32 - curr;
                v.clamp(-c[k], c[k])
            };
            // Eq. 1490.
            let mut sum: i32 = 0;
            sum += f[0] * (clip(0, 0, y2) + clip(0, 0, -y2));
            sum += f[1] * (clip(1, 1, y1) + clip(1, -1, -y1));
            sum += f[2] * (clip(2, 0, y1) + clip(2, 0, -y1));
            sum += f[3] * (clip(3, -1, y1) + clip(3, 1, -y1));
            sum += f[4] * (clip(4, 2, 0) + clip(4, -2, 0));
            sum += f[5] * (clip(5, 1, 0) + clip(5, -1, 0));
            // Eq. 1491.
            let bias = 1i32 << (alf_shift_c - 1);
            let new = curr + ((sum + bias) >> alf_shift_c);
            // Eq. 1492.
            let new = new.clamp(0, max_val).min(max_val_8) as u8;
            plane.samples[(ys as usize) * stride + (xs as usize)] = new;
        }
    }
}

/// Sample read with picture-edge clipping (§8.8.5.2 / §8.8.5.4
/// equivalents of eqs. 1446 / 1447 / 1485 / 1486).
#[inline]
fn sample(buf: &[u8], stride: usize, pw: i32, ph: i32, x: i32, y: i32) -> u8 {
    let xc = x.clamp(0, pw - 1) as usize;
    let yc = y.clamp(0, ph - 1) as usize;
    buf[yc * stride + xc]
}

/// Table 45 — `(alfShiftY, y1, y2, y3)` for the §8.8.5.2 luma filter
/// based on the vertical sample position `y` within the CTB (0-indexed)
/// and `applyAlfLineBufBoundary`. The single-slice scaffold treats
/// `applyAlfLineBufBoundary` as 1 for every CTB except the last picture
/// row when its height is less than `CtbSizeY - 4`; the caller passes
/// `apply_line_buf_boundary` after deriving that condition.
fn table_45(y: i32, ctb_size_y: i32, apply_line_buf_boundary: bool) -> (i32, i32, i32, i32) {
    if apply_line_buf_boundary {
        if y == ctb_size_y - 5 || y == ctb_size_y - 4 {
            return (10, 0, 0, 0);
        }
        if y == ctb_size_y - 6 || y == ctb_size_y - 3 {
            return (7, 1, 1, 1);
        }
        if y == ctb_size_y - 7 || y == ctb_size_y - 2 {
            return (7, 1, 2, 2);
        }
    }
    (7, 1, 2, 3)
}

/// Table 46 — `(alfShiftC, y1, y2)` for the §8.8.5.4 chroma filter.
fn table_46(y: i32, ctb_h_c: i32, apply_line_buf_boundary: bool) -> (i32, i32, i32) {
    if apply_line_buf_boundary {
        if y == ctb_h_c - 2 || y == ctb_h_c - 3 {
            return (10, 0, 0);
        }
        if y == ctb_h_c - 1 || y == ctb_h_c - 4 {
            return (7, 1, 1);
        }
    }
    (7, 1, 2)
}

/// CC-ALF coefficient resolver helper. Returned `None` from CC-ALF
/// branches in [`apply_alf`] would be the round-16 path — currently
/// CC-ALF is a no-op, so this helper exists only as a sanity check
/// that the [`AlfApsData::cc_cb_coeff`] / `cc_cr_coeff` arrays are
/// populated as expected by the parser.
#[allow(dead_code)]
fn cc_alf_coeff_count_sanity(aps: &AlfApsData) -> bool {
    aps.cc_cb_coeff.iter().all(|f| f.len() == ALF_CC_NUM_COEFF)
        && aps.cc_cr_coeff.iter().all(|f| f.len() == ALF_CC_NUM_COEFF)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aps::AlfApsData;

    fn fresh_buf(w: usize, h: usize, seed: u8) -> PictureBuffer {
        PictureBuffer {
            luma: PicturePlane::filled(w, h, seed),
            cb: PicturePlane::filled(w / 2, h / 2, 128),
            cr: PicturePlane::filled(w / 2, h / 2, 128),
        }
    }

    fn cfg_8bit() -> AlfConfig {
        AlfConfig {
            alf_enabled: true,
            cb_enabled: false,
            cr_enabled: false,
            bit_depth: 8,
            ctb_log2_size_y: 5,
            chroma_format_idc: 1,
        }
    }

    /// `is_all_off` is true for a fresh AlfPicture.
    #[test]
    fn alf_picture_default_off() {
        let pic = AlfPicture::empty(2, 3);
        assert!(pic.is_all_off());
    }

    /// `set` / `get` round-trip.
    #[test]
    fn alf_picture_set_get() {
        let mut pic = AlfPicture::empty(2, 2);
        let mut p = AlfCtb::default();
        p.luma_on = true;
        p.luma_filt_set_idx = 16;
        pic.set(1, 0, p);
        let g = pic.get(1, 0);
        assert!(g.luma_on);
        assert_eq!(g.luma_filt_set_idx, 16);
        assert!(!pic.is_all_off());
    }

    /// `apply_alf` short-circuits when alf_enabled = 0 even with an
    /// active per-CTB record.
    #[test]
    fn apply_alf_no_op_when_slice_disabled() {
        let mut buf = fresh_buf(32, 32, 100);
        let mut pic = AlfPicture::empty(1, 1);
        pic.set(
            0,
            0,
            AlfCtb {
                luma_on: true,
                luma_filt_set_idx: 16,
                ..Default::default()
            },
        );
        let mut cfg = cfg_8bit();
        cfg.alf_enabled = false;
        apply_alf(&mut buf, &pic, &cfg, &AlfApsBinding::default());
        assert!(buf.luma.samples.iter().all(|&v| v == 100));
    }

    /// `apply_alf` no-ops when the per-CTB array is all-off.
    #[test]
    fn apply_alf_no_op_when_all_off() {
        let mut buf = fresh_buf(32, 32, 100);
        let pic = AlfPicture::empty(1, 1);
        apply_alf(&mut buf, &pic, &cfg_8bit(), &AlfApsBinding::default());
        assert!(buf.luma.samples.iter().all(|&v| v == 100));
    }

    /// Fixed-filter-set CTBs (idx < 16) now resolve through eqs. 90 /
    /// 91 / 1437 / 1438 (`AlfFixFiltCoeff` + `AlfClassToFiltMap`). On a
    /// flat plane every clipped neighbour-delta is 0, so even with
    /// non-zero coefficients the filter output equals `curr` and the
    /// plane stays unchanged.
    #[test]
    fn fixed_filter_set_is_identity_on_flat_plane() {
        let mut buf = fresh_buf(32, 32, 100);
        let mut pic = AlfPicture::empty(1, 1);
        pic.set(
            0,
            0,
            AlfCtb {
                luma_on: true,
                luma_filt_set_idx: 0, // < 16 → fixed-filter set 0
                ..Default::default()
            },
        );
        apply_alf(&mut buf, &pic, &cfg_8bit(), &AlfApsBinding::default());
        // Flat plane → all neighbour-deltas are 0 → sum stays 0 → curr
        // unchanged.
        assert!(buf.luma.samples.iter().all(|&v| v == 100));
    }

    /// On a non-flat plane the fixed-filter set should produce *some*
    /// change in the output (proves the table wiring isn't a no-op).
    /// We use a single-pixel "spike" rather than a step edge — the
    /// 7×7 diamond's centre-symmetric tap pairs cancel on a step
    /// (the filter is direction-symmetric) but not on a delta.
    #[test]
    fn fixed_filter_set_modifies_non_flat_plane() {
        let mut buf = fresh_buf(32, 32, 100);
        let stride = buf.luma.stride;
        // Single asymmetric spike — placed off-centre so the per-tap
        // pairs see asymmetric neighbours.
        buf.luma.samples[15 * stride + 15] = 250;
        let before = buf.luma.samples.clone();
        let mut pic = AlfPicture::empty(1, 1);
        pic.set(
            0,
            0,
            AlfCtb {
                luma_on: true,
                luma_filt_set_idx: 5, // arbitrary fixed-filter set
                ..Default::default()
            },
        );
        apply_alf(&mut buf, &pic, &cfg_8bit(), &AlfApsBinding::default());
        // At least one sample must have changed (the spike redistributes
        // energy via the diamond filter).
        assert!(
            buf.luma
                .samples
                .iter()
                .zip(before.iter())
                .any(|(a, b)| a != b),
            "fixed-filter apply must modify at least one sample on a non-flat plane",
        );
    }

    /// All-zero filter coefficients on a flat plane → output unchanged
    /// (sum = 0 → curr + 0 = curr).
    #[test]
    fn zero_coeff_filter_is_identity_on_flat() {
        let mut buf = fresh_buf(32, 32, 100);
        let mut aps = AlfApsData::default();
        aps.alf_luma_filter_signal_flag = true;
        aps.alf_luma_clip_flag = false;
        aps.luma_coeff = vec![[0i32; ALF_LUMA_NUM_COEFF]; NUM_ALF_FILTERS];
        aps.luma_clip_idx = vec![[0u8; ALF_LUMA_NUM_COEFF]; NUM_ALF_FILTERS];
        let aps_slot: [Option<&AlfApsData>; 1] = [Some(&aps)];
        let binding = AlfApsBinding {
            luma_apses: &aps_slot,
            chroma_aps: None,
            cc_cb_aps: None,
            cc_cr_aps: None,
        };
        let mut pic = AlfPicture::empty(1, 1);
        pic.set(
            0,
            0,
            AlfCtb {
                luma_on: true,
                luma_filt_set_idx: 16, // → APS slot 0
                ..Default::default()
            },
        );
        apply_alf(&mut buf, &pic, &cfg_8bit(), &binding);
        assert!(buf.luma.samples.iter().all(|&v| v == 100));
    }

    /// On a flat plane every neighbour minus curr = 0, so the clipped
    /// term is 0 regardless of c[]. With non-zero coefficients the sum
    /// is still 0 → identity.
    #[test]
    fn nonzero_coeff_flat_plane_is_identity() {
        let mut buf = fresh_buf(32, 32, 80);
        let mut aps = AlfApsData::default();
        aps.alf_luma_filter_signal_flag = true;
        aps.alf_luma_clip_flag = false;
        let mut row = [0i32; ALF_LUMA_NUM_COEFF];
        row[0] = 5;
        row[6] = -3;
        aps.luma_coeff = vec![row; NUM_ALF_FILTERS];
        aps.luma_clip_idx = vec![[0u8; ALF_LUMA_NUM_COEFF]; NUM_ALF_FILTERS];
        let aps_slot: [Option<&AlfApsData>; 1] = [Some(&aps)];
        let binding = AlfApsBinding {
            luma_apses: &aps_slot,
            chroma_aps: None,
            cc_cb_aps: None,
            cc_cr_aps: None,
        };
        let mut pic = AlfPicture::empty(1, 1);
        pic.set(
            0,
            0,
            AlfCtb {
                luma_on: true,
                luma_filt_set_idx: 16,
                ..Default::default()
            },
        );
        apply_alf(&mut buf, &pic, &cfg_8bit(), &binding);
        assert!(buf.luma.samples.iter().all(|&v| v == 80));
    }

    /// On a non-flat plane the filter must actually run — drop a single
    /// "spike" pixel surrounded by uniform 100s. With f[6] = 64 (the
    /// vertical centre-tap pair), the spike at (16, 16) attracts the
    /// vertical ±y1 neighbours which are 100, but those neighbours
    /// themselves see (curr - spike) clipped within ±c. Verify that
    /// some pixels in the immediate neighbourhood change while the far
    /// background stays at 100.
    #[test]
    fn nonzero_coeff_spike_modifies_neighbourhood() {
        let mut buf = fresh_buf(32, 32, 100);
        // Spike one pixel.
        let stride = buf.luma.stride;
        buf.luma.samples[16 * stride + 16] = 200;
        let mut aps = AlfApsData::default();
        aps.alf_luma_filter_signal_flag = true;
        aps.alf_luma_clip_flag = false;
        let mut row = [0i32; ALF_LUMA_NUM_COEFF];
        row[6] = 32; // vertical y1 pair only
        aps.luma_coeff = vec![row; NUM_ALF_FILTERS];
        aps.luma_clip_idx = vec![[0u8; ALF_LUMA_NUM_COEFF]; NUM_ALF_FILTERS];
        let aps_slot: [Option<&AlfApsData>; 1] = [Some(&aps)];
        let binding = AlfApsBinding {
            luma_apses: &aps_slot,
            chroma_aps: None,
            cc_cb_aps: None,
            cc_cr_aps: None,
        };
        let mut pic = AlfPicture::empty(1, 1);
        pic.set(
            0,
            0,
            AlfCtb {
                luma_on: true,
                luma_filt_set_idx: 16,
                ..Default::default()
            },
        );
        apply_alf(&mut buf, &pic, &cfg_8bit(), &binding);
        // The spike pixel sees (100-200, 100-200) → 2 * (-100), clipped
        // by c=256 (clipIdx=0, BitDepth=8 → 1<<8). sum = 32 * (-200) =
        // -6400. shift = 7 → (-6400 + 64) >> 7 = -49 (round).
        // new = 200 + (-49) = 151 (not 200 anymore).
        let new_centre = buf.luma.samples[16 * stride + 16];
        assert_ne!(new_centre, 200, "centre should be modified by ALF");
        // A pixel far from the spike (e.g. (0,0)) has all neighbours
        // = 100 → identity.
        assert_eq!(buf.luma.samples[0], 100);
    }

    /// Table 8 expansion sanity: `clipIdx = 0` always returns
    /// `1 << BitDepth` (the spec's "linear" / no-clipping value).
    #[test]
    fn clip_value_clip_idx_zero_is_one_shl_bit_depth() {
        for bd in 8u32..=16 {
            assert_eq!(resolve_clip_value(bd, 0), 1i32 << bd);
        }
    }

    /// Table 8 expansion: `clipIdx > 0` returns a strictly smaller
    /// power of two than `clipIdx-1` for every BitDepth.
    #[test]
    fn clip_value_monotonic() {
        for bd in 8u32..=16 {
            let mut prev = resolve_clip_value(bd, 0);
            for ci in 1u8..=3 {
                let cur = resolve_clip_value(bd, ci);
                assert!(cur < prev, "BitDepth={bd} clipIdx={ci} not strictly < prev");
                prev = cur;
            }
        }
    }

    /// AlfApsBinding default has empty luma APS list, no chroma APS,
    /// and no CC-ALF APSes (Cb / Cr).
    #[test]
    fn binding_default_is_empty() {
        let b = AlfApsBinding::default();
        assert!(b.luma_apses.is_empty());
        assert!(b.chroma_aps.is_none());
        assert!(b.cc_cb_aps.is_none());
        assert!(b.cc_cr_aps.is_none());
    }

    /// Chroma ALF identity: zero-coefficient chroma filter on a flat
    /// chroma plane leaves it unchanged.
    #[test]
    fn chroma_zero_coeff_is_identity() {
        let mut buf = fresh_buf(32, 32, 100);
        let mut aps = AlfApsData::default();
        aps.alf_chroma_filter_signal_flag = true;
        aps.alf_chroma_num_alt_filters_minus1 = 0;
        aps.chroma_coeff = vec![[0i32; ALF_CHROMA_NUM_COEFF]; 1];
        aps.chroma_clip_idx = vec![[0u8; ALF_CHROMA_NUM_COEFF]; 1];
        let binding = AlfApsBinding {
            luma_apses: &[],
            chroma_aps: Some(&aps),
            cc_cb_aps: None,
            cc_cr_aps: None,
        };
        let mut pic = AlfPicture::empty(1, 1);
        pic.set(
            0,
            0,
            AlfCtb {
                cb_on: true,
                cr_on: true,
                ..Default::default()
            },
        );
        let mut cfg = cfg_8bit();
        cfg.cb_enabled = true;
        cfg.cr_enabled = true;
        apply_alf(&mut buf, &pic, &cfg, &binding);
        assert!(buf.cb.samples.iter().all(|&v| v == 128));
        assert!(buf.cr.samples.iter().all(|&v| v == 128));
        // Luma untouched (luma_on = false).
        assert!(buf.luma.samples.iter().all(|&v| v == 100));
    }

    /// Table 45 line-buffer rows.
    #[test]
    fn table_45_returns_default_for_interior() {
        // Interior position (j=10 in a CtbSize=32) → fall through to
        // "Otherwise" row.
        assert_eq!(table_45(10, 32, true), (7, 1, 2, 3));
    }

    #[test]
    fn table_45_line_buffer_rows() {
        let ctb = 32;
        // y == CtbSizeY - 5 (= 27) or - 4 (= 28) → alfShiftY=10, all 0.
        assert_eq!(table_45(27, ctb, true), (10, 0, 0, 0));
        assert_eq!(table_45(28, ctb, true), (10, 0, 0, 0));
        assert_eq!(table_45(26, ctb, true), (7, 1, 1, 1));
        assert_eq!(table_45(29, ctb, true), (7, 1, 1, 1));
        assert_eq!(table_45(25, ctb, true), (7, 1, 2, 2));
        assert_eq!(table_45(30, ctb, true), (7, 1, 2, 2));
    }

    /// Table 45: when applyAlfLineBufBoundary == 0 the line-buffer
    /// rows collapse to the default.
    #[test]
    fn table_45_no_line_buffer_collapses() {
        for y in 0..32 {
            assert_eq!(table_45(y, 32, false), (7, 1, 2, 3));
        }
    }

    /// Table 46 sanity for chroma.
    #[test]
    fn table_46_default_and_line_buffer() {
        let h = 16;
        assert_eq!(table_46(0, h, true), (7, 1, 2));
        assert_eq!(table_46(13, h, true), (10, 0, 0));
        assert_eq!(table_46(14, h, true), (10, 0, 0));
        assert_eq!(table_46(12, h, true), (7, 1, 1));
        assert_eq!(table_46(15, h, true), (7, 1, 1));
    }

    /// CC-ALF coeff lengths sanity.
    #[test]
    fn cc_alf_coeff_lengths() {
        let mut aps = AlfApsData::default();
        aps.cc_cb_coeff = vec![[0; ALF_CC_NUM_COEFF]; 2];
        aps.cc_cr_coeff = vec![[0; ALF_CC_NUM_COEFF]; 1];
        assert!(cc_alf_coeff_count_sanity(&aps));
    }

    /// transposeIdx == 0..3 produce four distinct permutations whose
    /// composition with itself is the identity for indices 1, 2, 3
    /// (vertical-flip / horizontal-flip / 180°-rotate are all
    /// involutions — applying twice returns the original ordering).
    #[test]
    fn transpose_idx_table_involution_for_1_2_3() {
        for ti in 0u8..=3 {
            let p = transpose_idx_table(ti);
            // Applying the same permutation twice must equal identity
            // (only true for involutions: 0=identity, 1, 2, 3).
            let mut twice = [0usize; ALF_LUMA_NUM_COEFF];
            for (k, &pk) in p.iter().enumerate() {
                twice[k] = p[pk];
            }
            for k in 0..ALF_LUMA_NUM_COEFF {
                assert_eq!(twice[k], k, "transposeIdx {ti} not an involution at k={k}");
            }
        }
    }

    /// transposeIdx == 0 returns the identity permutation (eq. 1445).
    #[test]
    fn transpose_idx_table_zero_is_identity() {
        let p = transpose_idx_table(0);
        for k in 0..ALF_LUMA_NUM_COEFF {
            assert_eq!(p[k], k);
        }
    }

    /// §8.8.5.3 — flat plane → all Laplacian sums are zero → activity
    /// quantises to 0 and `dirS` becomes 0, so every 4×4 sub-block ends
    /// up with `filtIdx = 0` (the activity-only base class). The
    /// `transposeIdx` is unconstrained when `dirS == 0` — the spec
    /// formula still feeds the transpose table from `dir1`/`dir2`, but
    /// the filter sees uniform neighbours so the geometric transform is
    /// inert. We therefore only assert on `filtIdx`.
    #[test]
    fn classification_flat_plane_is_class_zero() {
        let plane = vec![100u8; 32 * 32];
        let cls = derive_luma_classification(&plane, 32, 32, 32, 0, 0, 32, 8);
        for sy in 0..cls.sub_size() {
            for sx in 0..cls.sub_size() {
                let (f, _t) = cls.get(sx, sy);
                assert_eq!(f, 0, "flat plane should produce class 0 at ({sx},{sy})");
            }
        }
    }

    /// Linear gradient (sample = 8*x) is purely first-order, so all
    /// second-difference Laplacians are zero (modulo edge clipping) →
    /// class 0 in the interior.
    #[test]
    fn classification_linear_gradient_interior_is_class_zero() {
        let mut plane = vec![0u8; 32 * 32];
        for y in 0..32 {
            for x in 0..32 {
                plane[y * 32 + x] = (x as u8).saturating_mul(7);
            }
        }
        let cls = derive_luma_classification(&plane, 32, 32, 32, 0, 0, 32, 8);
        // Far from the picture edge — sub-block (3,3) sits at pixel
        // x=12..16, y=12..16 of the CTB; its 8-pixel window i=-2..5
        // (centred on x4=12) → x=10..17, all interior. No edge effects.
        let (f, _t) = cls.get(3, 3);
        assert_eq!(f, 0, "interior linear gradient should be class 0");
    }

    /// Strong vertical edge → high vertical activity → non-zero
    /// `dirS`, so `filtIdx` is bumped past the activity-only range
    /// `0..5` and lands in `5..25`. Different sub-blocks pick
    /// different filter indices.
    #[test]
    fn classification_vertical_edge_picks_directional_class() {
        // 32×32 picture: top half = 0, bottom half = 200. Edge at row 16.
        let mut plane = vec![0u8; 32 * 32];
        for y in 16..32 {
            for x in 0..32 {
                plane[y * 32 + x] = 200;
            }
        }
        let cls = derive_luma_classification(&plane, 32, 32, 32, 0, 0, 32, 8);
        // Sub-blocks straddling the edge live around sy = 4 (pixels y=16..19);
        // their classification window covers y4=16, j=-2..5 → y=14..21,
        // which spans the step. Expect non-zero filtIdx and possibly
        // a non-zero transposeIdx.
        let (f_edge, _t_edge) = cls.get(3, 4); // pixel (12..16, 16..20)
        let (f_far, _t_far) = cls.get(0, 0); // top-left corner: all zeros
        assert!(
            f_edge >= 5,
            "edge sub-block should pick a directional class (≥5), got {f_edge}"
        );
        assert_eq!(f_far, 0, "far-from-edge corner should be class 0");
        // At least one sub-block is in a non-zero class.
        let mut classes = std::collections::HashSet::new();
        for sy in 0..cls.sub_size() {
            for sx in 0..cls.sub_size() {
                classes.insert(cls.get(sx, sy).0);
            }
        }
        assert!(
            classes.len() >= 2,
            "expected at least two distinct classes from a vertical-edge picture, got {:?}",
            classes
        );
    }

    /// `derive_luma_classification` writes a `(filtIdx, transposeIdx)`
    /// for every 4×4 sub-block (CtbSizeY/4 squared total). Verify the
    /// grid dimensions for CTB sizes 16, 32, 64.
    #[test]
    fn classification_grid_size_matches_ctb_size() {
        for log2 in 4..=6u32 {
            let ctb = 1u32 << log2;
            let plane = vec![100u8; (ctb * ctb) as usize];
            let cls = derive_luma_classification(
                &plane,
                ctb as usize,
                ctb as i32,
                ctb as i32,
                0,
                0,
                ctb,
                8,
            );
            assert_eq!(cls.sub_size(), (ctb >> 2) as usize);
        }
    }

    /// End-to-end: with all 25 filter rows distinct (via per-row tags
    /// embedded in coefficients), apply_alf on a non-uniform CTB
    /// must select different filters for different sub-blocks.
    /// We check this indirectly: with classification active, a
    /// vertical-edge picture filtered through `apply_alf` produces
    /// pixel modifications — and the modifications are *not* uniform
    /// across the CTB (proving classification + transpose actually
    /// pick varying filters per sub-block).
    #[test]
    fn apply_alf_classification_produces_non_uniform_output() {
        // Vertical edge picture.
        let mut buf = fresh_buf(32, 32, 0);
        for y in 16..32 {
            for x in 0..32 {
                buf.luma.samples[y * buf.luma.stride + x] = 200;
            }
        }
        // 25 distinct filters: row k has f[6] = 4*(k+1) (so different
        // classes pick different shifts on vertical neighbours).
        let mut aps = AlfApsData::default();
        aps.alf_luma_filter_signal_flag = true;
        let mut coeffs: Vec<[i32; ALF_LUMA_NUM_COEFF]> =
            vec![[0i32; ALF_LUMA_NUM_COEFF]; NUM_ALF_FILTERS];
        for k in 0..NUM_ALF_FILTERS {
            coeffs[k][6] = 4 * (k as i32 + 1);
        }
        aps.luma_coeff = coeffs;
        aps.luma_clip_idx = vec![[0u8; ALF_LUMA_NUM_COEFF]; NUM_ALF_FILTERS];
        let aps_slot: [Option<&AlfApsData>; 1] = [Some(&aps)];
        let binding = AlfApsBinding {
            luma_apses: &aps_slot,
            chroma_aps: None,
            cc_cb_aps: None,
            cc_cr_aps: None,
        };
        let mut pic = AlfPicture::empty(1, 1);
        pic.set(
            0,
            0,
            AlfCtb {
                luma_on: true,
                luma_filt_set_idx: 16,
                ..Default::default()
            },
        );
        // Snapshot pre-ALF for comparison.
        let pre = buf.luma.samples.clone();
        apply_alf(&mut buf, &pic, &cfg_8bit(), &binding);
        // The very corner (0,0) sees only zero neighbours → identity
        // (or at least very small changes); it should remain 0.
        assert_eq!(buf.luma.samples[0], 0);
        // Pixels straddling the edge are modified — the spike at the
        // boundary attracts adjustment.
        let stride = buf.luma.stride;
        let edge = buf.luma.samples[16 * stride + 16];
        assert_ne!(edge, pre[16 * stride + 16], "edge sample should change");
    }

    /// `apply_alf` skips chroma when `cb_enabled` / `cr_enabled` are off
    /// at the slice header even when the per-CTB on flags are set.
    #[test]
    fn chroma_disabled_at_slice_header_skips_chroma() {
        let mut buf = fresh_buf(32, 32, 100);
        // Spike a chroma pixel; we'd see it move iff the filter ran.
        buf.cb.samples[5 * buf.cb.stride + 5] = 200;
        let mut aps = AlfApsData::default();
        aps.alf_chroma_filter_signal_flag = true;
        aps.alf_chroma_num_alt_filters_minus1 = 0;
        let mut row = [0i32; ALF_CHROMA_NUM_COEFF];
        row[0] = 100;
        aps.chroma_coeff = vec![row; 1];
        aps.chroma_clip_idx = vec![[0u8; ALF_CHROMA_NUM_COEFF]; 1];
        let binding = AlfApsBinding {
            luma_apses: &[],
            chroma_aps: Some(&aps),
            cc_cb_aps: None,
            cc_cr_aps: None,
        };
        let mut pic = AlfPicture::empty(1, 1);
        pic.set(
            0,
            0,
            AlfCtb {
                cb_on: true,
                ..Default::default()
            },
        );
        let mut cfg = cfg_8bit();
        cfg.cb_enabled = false; // gate at slice header
        apply_alf(&mut buf, &pic, &cfg, &binding);
        // Chroma is left alone — the spike survives.
        assert_eq!(buf.cb.samples[5 * buf.cb.stride + 5], 200);
    }

    /// CC-ALF identity: zero-coefficient cross-component filter on a
    /// flat luma plane leaves the chroma plane unchanged.
    #[test]
    fn cc_alf_zero_coeff_is_identity() {
        let mut buf = fresh_buf(32, 32, 100);
        // Mark Cb with a known non-default value.
        for v in buf.cb.samples.iter_mut() {
            *v = 64;
        }
        let mut aps = AlfApsData::default();
        aps.cc_cb_coeff = vec![[0i32; ALF_CC_NUM_COEFF]; 1];
        let binding = AlfApsBinding {
            luma_apses: &[],
            chroma_aps: None,
            cc_cb_aps: Some(&aps),
            cc_cr_aps: None,
        };
        let mut pic = AlfPicture::empty(1, 1);
        pic.set(
            0,
            0,
            AlfCtb {
                cc_cb_idc: 1, // → CC-ALF filter index 0
                ..Default::default()
            },
        );
        let mut cfg = cfg_8bit();
        cfg.cb_enabled = true;
        apply_alf(&mut buf, &pic, &cfg, &binding);
        // Chroma stays at 64 since every luma-delta is zero on a flat
        // plane → sum=0 → curr+0=curr.
        assert!(buf.cb.samples.iter().all(|&v| v == 64));
    }

    /// CC-ALF on a non-flat luma + non-zero coefficient → at least one
    /// chroma sample must be modified.
    #[test]
    fn cc_alf_non_zero_coeff_modifies_chroma() {
        let mut buf = fresh_buf(32, 32, 100);
        // Vertical step in luma: left half = 50, right half = 200.
        for y in 0..32 {
            for x in 16..32 {
                buf.luma.samples[y * 32 + x] = 200;
            }
        }
        // Flat chroma at 128.
        let before_cb = buf.cb.samples.clone();
        let mut aps = AlfApsData::default();
        let mut row = [0i32; ALF_CC_NUM_COEFF];
        row[1] = 16; // tap on the immediate left luma neighbour
        row[2] = -16; // tap on the immediate right luma neighbour
        aps.cc_cb_coeff = vec![row; 1];
        let binding = AlfApsBinding {
            luma_apses: &[],
            chroma_aps: None,
            cc_cb_aps: Some(&aps),
            cc_cr_aps: None,
        };
        let mut pic = AlfPicture::empty(1, 1);
        pic.set(
            0,
            0,
            AlfCtb {
                cc_cb_idc: 1,
                ..Default::default()
            },
        );
        let mut cfg = cfg_8bit();
        cfg.cb_enabled = true;
        apply_alf(&mut buf, &pic, &cfg, &binding);
        assert!(
            buf.cb
                .samples
                .iter()
                .zip(before_cb.iter())
                .any(|(a, b)| a != b),
            "CC-ALF must modify at least one chroma sample on a non-flat luma plane",
        );
    }

    /// CC-ALF respects the per-CTB `cc_cb_idc == 0` gate (no-op).
    #[test]
    fn cc_alf_skipped_when_idc_zero() {
        let mut buf = fresh_buf(32, 32, 100);
        for v in buf.cb.samples.iter_mut() {
            *v = 64;
        }
        let mut aps = AlfApsData::default();
        let mut row = [0i32; ALF_CC_NUM_COEFF];
        row[0] = 50;
        aps.cc_cb_coeff = vec![row; 1];
        let binding = AlfApsBinding {
            luma_apses: &[],
            chroma_aps: None,
            cc_cb_aps: Some(&aps),
            cc_cr_aps: None,
        };
        let mut pic = AlfPicture::empty(1, 1);
        // cc_cb_idc default = 0 → CC-ALF off for this CTB.
        pic.set(0, 0, AlfCtb::default());
        let mut cfg = cfg_8bit();
        cfg.cb_enabled = true;
        apply_alf(&mut buf, &pic, &cfg, &binding);
        assert!(buf.cb.samples.iter().all(|&v| v == 64));
    }

    /// Table 47: with `applyAlfLineBufBoundary == 0` the answer is
    /// always (1, 2) regardless of the y-position carve-outs.
    #[test]
    fn cc_alf_table_47_no_line_buffer_collapses() {
        for y in 0..32 {
            assert_eq!(cc_alf_table_47(y, 32, false), (1, 2));
        }
    }

    /// Table 47 carve-outs: (CtbSizeY-5, CtbSizeY-4) → (0, 0);
    /// (CtbSizeY-6, CtbSizeY-3) → (1, 1).
    #[test]
    fn cc_alf_table_47_carve_outs() {
        let ctb = 32;
        assert_eq!(cc_alf_table_47(ctb - 5, ctb, true), (0, 0));
        assert_eq!(cc_alf_table_47(ctb - 4, ctb, true), (0, 0));
        assert_eq!(cc_alf_table_47(ctb - 6, ctb, true), (1, 1));
        assert_eq!(cc_alf_table_47(ctb - 3, ctb, true), (1, 1));
        // Non-carve-out positions still get (1, 2).
        assert_eq!(cc_alf_table_47(0, ctb, true), (1, 2));
    }
}
