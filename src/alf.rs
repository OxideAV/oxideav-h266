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
//! ## Scope of this round (round 15)
//!
//! Per the round-15 plan we land:
//!
//! 1. The data-source plumbing — `alf_data()` parsing in
//!    [`crate::aps::parse_aps`] populates [`crate::aps::AlfApsData`],
//!    and the per-picture array of CTB on/off + filter selection lives
//!    in [`AlfPicture`].
//! 2. The §8.8.5.2 luma filter math (eqs. 1446 – 1451) including the
//!    Table 45 line-buffer offsets and the §8.8.5.6 boundary clipping.
//!    The §8.8.5.3 classification is **not** implemented this round —
//!    every sample is filtered with `filtIdx = 0` and `transposeIdx = 0`.
//!    [`AlfPicture`] callers can still drive arbitrary filter sets per
//!    CTB; classification can be hung off the same data path in a
//!    later round.
//! 3. The §8.8.5.4 chroma filter math (eqs. 1485 – 1492) with the same
//!    simplified boundary handling. `alf_ctb_filter_alt_idx[chromaIdx]`
//!    selects the alternative filter per CTB.
//! 4. CC-ALF (§8.8.5.7) is **not** implemented this round; the
//!    [`AlfPicture::cc_cb_idc`] / `cc_cr_idc` arrays are surfaced for
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
    /// can short-circuit.
    pub fn is_all_off(&self) -> bool {
        self.ctbs.iter().all(|c| !c.luma_on && !c.cb_on && !c.cr_on)
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
}

impl Default for AlfApsBinding<'_> {
    fn default() -> Self {
        Self {
            luma_apses: &[],
            chroma_aps: None,
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
                if let Some((coeff, clip_idx)) =
                    resolve_luma_filter(p.luma_filt_set_idx, 0, binding)
                {
                    apply_alf_luma_ctb(
                        &mut out.luma,
                        &luma_pre,
                        rx,
                        ry,
                        ctb_size_y,
                        cfg.bit_depth,
                        &coeff,
                        &clip_idx,
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
}

/// Resolve `(AlfCoeffL[i][filtIdx][...], AlfClipL[i][filtIdx][...])` for
/// the luma filter selected by `AlfCtbFiltSetIdxY` / `filt_idx`. Returns
/// `None` when the CTB references an absent APS or a fixed-filter set
/// (round-15 does not yet ship the 64-row §7.4.3.18 `AlfFixFiltCoeff`
/// table; fixed filters fall back to "no filter" — i.e. luma off — for
/// this CTB).
fn resolve_luma_filter(
    filt_set_idx: u8,
    filt_idx: usize,
    binding: &AlfApsBinding<'_>,
) -> Option<([i32; ALF_LUMA_NUM_COEFF], [u8; ALF_LUMA_NUM_COEFF])> {
    if (filt_set_idx as usize) < 16 {
        // Fixed-filter set — round-15 does not ship `AlfFixFiltCoeff`
        // (the 64-row 12-coefficient table from eq. 90). When the
        // caller installs a fixed-filter CTB the apply pass currently
        // skips it. Future round will add the table and switch this
        // branch to read `AlfFixFiltCoeff[AlfClassToFiltMap[i][filtIdx]]`.
        return None;
    }
    let aps_slot = (filt_set_idx as usize) - 16;
    let aps = binding.luma_apses.get(aps_slot).and_then(|s| *s)?;
    if !aps.alf_luma_filter_signal_flag {
        return None;
    }
    if filt_idx >= NUM_ALF_FILTERS {
        return None;
    }
    let coeff = aps.luma_coeff[filt_idx];
    let clip = aps.luma_clip_idx[filt_idx];
    Some((coeff, clip))
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
/// `plane`). The caller passes the resolved 12-tap filter coefficients
/// and Table 8 clipping indices.
#[allow(clippy::too_many_arguments)]
fn apply_alf_luma_ctb(
    plane: &mut PicturePlane,
    pre: &[u8],
    rx: u32,
    ry: u32,
    ctb_size_y: u32,
    bit_depth: u32,
    f: &[i32; ALF_LUMA_NUM_COEFF],
    clip_idx: &[u8; ALF_LUMA_NUM_COEFF],
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

    // Identity transposeIdx (round-15 single-class scaffold). Eq. 1445.
    let idx = [0usize, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
    // Resolve clipping values once per CTB.
    let mut c = [0i32; ALF_LUMA_NUM_COEFF];
    for k in 0..ALF_LUMA_NUM_COEFF {
        c[k] = resolve_clip_value(bit_depth, clip_idx[k]);
    }

    for j in 0..j_max {
        for i in 0..i_max {
            let xs = (x_ctb + i) as i32;
            let ys = (y_ctb + j) as i32;
            // Table 45: pick alfShiftY / y1 / y2 / y3 from the vertical
            // sample position. The boundary suppression case
            // (`applyAlfLineBufBoundary == 1`) only fires on inter-CTB
            // line-buffer rows; in the round-15 single-slice scaffold
            // we treat the bottom of the picture per the spec — which
            // collapses to "applyAlfLineBufBoundary == 0" for the very
            // last CTB row when it doesn't span CtbSizeY-4 rows past
            // yCtb. For the interior we always use the "Otherwise" row
            // (alfShiftY = 7, y1 = 1, y2 = 2, y3 = 3).
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

    /// Fixed-filter-set CTBs are currently treated as luma-off (the
    /// 64-row AlfFixFiltCoeff table is not yet shipped in this round).
    #[test]
    fn fixed_filter_set_falls_through() {
        let mut buf = fresh_buf(32, 32, 100);
        let mut pic = AlfPicture::empty(1, 1);
        pic.set(
            0,
            0,
            AlfCtb {
                luma_on: true,
                luma_filt_set_idx: 0, // < 16 → fixed filter
                ..Default::default()
            },
        );
        apply_alf(&mut buf, &pic, &cfg_8bit(), &AlfApsBinding::default());
        // Luma untouched.
        assert!(buf.luma.samples.iter().all(|&v| v == 100));
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

    /// AlfApsBinding default has empty luma APS list and no chroma APS.
    #[test]
    fn binding_default_is_empty() {
        let b = AlfApsBinding::default();
        assert!(b.luma_apses.is_empty());
        assert!(b.chroma_aps.is_none());
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
}
