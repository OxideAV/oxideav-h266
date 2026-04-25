//! VVC sample adaptive offset (SAO) — §8.8.4.
//!
//! SAO sits between the §8.8.3 deblocking filter and the §8.8.5 ALF in
//! the in-loop filter chain. It operates on a CTB basis; each CTB
//! independently selects between three modes per colour component:
//!
//! * `SaoTypeIdx = 0` — not applied (the CTB is unchanged).
//! * `SaoTypeIdx = 1` — band offset (BO). The 0..(1<<BitDepth)-1 sample
//!   range is partitioned into 32 equal bands; four contiguous bands
//!   starting at `sao_band_position` receive non-zero offsets.
//! * `SaoTypeIdx = 2` — edge offset (EO). Each sample is classified
//!   into one of five categories (none, four EO categories) based on
//!   the comparison between it and its two neighbours along one of four
//!   directions selected by `sao_eo_class`. Categories 1..4 receive
//!   per-class offsets; category 0 leaves the sample unmodified.
//!
//! ## Sub-modules
//!
//! * [`SaoCtb`] — per-CTB SAO parameters for one component (luma OR
//!   one chroma plane).
//! * [`SaoCtbParams`] — the three [`SaoCtb`]s for a single CTB.
//! * [`SaoPicture`] — per-picture array of [`SaoCtbParams`], indexed
//!   by `ry * pic_width_in_ctbs_y + rx`.
//! * [`apply_sao`] — top-level driver invoked by
//!   [`crate::ctu::CtuWalker::apply_in_loop_filters`] *after*
//!   deblocking. Honours `sh_sao_*_used_flag` from the slice header.
//!
//! ## Scope of this round
//!
//! * Both EO + BO arithmetic per §8.8.4.2 (eqs. 1424 – 1435 + Table 44).
//! * Cross-CTB neighbour reads via the picture sample array (single-
//!   slice / single-tile fixture only — the boundary-suppression flags
//!   from §8.8.4.2 collapse to "always available").
//! * Bit-depth scaling for `SaoOffsetVal` per eq. 153.
//!
//! ## Out of scope
//!
//! * The SAO syntax `sao(rx, ry)` (§7.3.11.3) — this requires CABAC
//!   contexts (Table 58). The round-13 walker therefore does not parse
//!   per-CTB SAO bins; callers may *programmatically* populate
//!   [`SaoPicture`] for testing and for fixtures that synthesise SAO.
//! * Virtual / subpicture / tile / slice boundary suppression
//!   (`pps_loop_filter_across_*_enabled_flag`, virtual boundaries).
//!   With the single-slice/single-tile scaffold these all collapse to
//!   "available unless outside the picture".
//!
//! Spec reference: ITU-T H.266 | ISO/IEC 23090-3 (V4, 01/2026) §8.8.4.

use crate::reconstruct::{PictureBuffer, PicturePlane};

/// SAO offset type for a single CTB and component (Table 10).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum SaoTypeIdx {
    /// Not applied — the CTB is left unchanged.
    #[default]
    NotApplied,
    /// Band offset.
    BandOffset,
    /// Edge offset.
    EdgeOffset,
}

impl SaoTypeIdx {
    /// Decode from the integer value emitted by the spec syntax (0/1/2).
    pub fn from_u32(v: u32) -> Self {
        match v {
            1 => SaoTypeIdx::BandOffset,
            2 => SaoTypeIdx::EdgeOffset,
            _ => SaoTypeIdx::NotApplied,
        }
    }
}

/// SAO edge-offset class (Table 11). Determines the pair of neighbour
/// offsets `hPos` / `vPos` per Table 44.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum SaoEoClass {
    /// 1D 0-degree (horizontal). hPos = (-1, +1), vPos = (0, 0).
    #[default]
    Horizontal,
    /// 1D 90-degree (vertical). hPos = (0, 0), vPos = (-1, +1).
    Vertical,
    /// 1D 135-degree. hPos = (-1, +1), vPos = (-1, +1).
    Deg135,
    /// 1D 45-degree. hPos = (+1, -1), vPos = (-1, +1).
    Deg45,
}

impl SaoEoClass {
    /// Decode from the integer value emitted by the spec syntax (0..3).
    pub fn from_u32(v: u32) -> Self {
        match v {
            1 => SaoEoClass::Vertical,
            2 => SaoEoClass::Deg135,
            3 => SaoEoClass::Deg45,
            _ => SaoEoClass::Horizontal,
        }
    }

    /// Table 44 — `(hPos[0], vPos[0])` for this EO class.
    #[inline]
    pub fn pos0(self) -> (i32, i32) {
        match self {
            SaoEoClass::Horizontal => (-1, 0),
            SaoEoClass::Vertical => (0, -1),
            SaoEoClass::Deg135 => (-1, -1),
            SaoEoClass::Deg45 => (1, -1),
        }
    }

    /// Table 44 — `(hPos[1], vPos[1])` for this EO class.
    #[inline]
    pub fn pos1(self) -> (i32, i32) {
        match self {
            SaoEoClass::Horizontal => (1, 0),
            SaoEoClass::Vertical => (0, 1),
            SaoEoClass::Deg135 => (1, 1),
            SaoEoClass::Deg45 => (-1, 1),
        }
    }
}

/// Per-CTB SAO parameters for *one* component.
///
/// `offset_val` is the spec's `SaoOffsetVal[cIdx][rx][ry][i]` array
/// after eq. 153 has applied the bit-depth shift; entry 0 is reserved
/// (always 0) and entries 1..=4 carry the four offsets — for EO they
/// correspond to edge categories 1..=4, and for BO they correspond to
/// the four bands at positions `sao_band_position .. sao_band_position
/// + 3` (mod 32).
///
/// When `sao_type_idx` is `NotApplied`, the other fields are ignored
/// and the CTB is passed through unchanged.
#[derive(Clone, Copy, Debug, Default)]
pub struct SaoCtb {
    pub sao_type_idx: SaoTypeIdx,
    pub eo_class: SaoEoClass,
    /// `sao_band_position[cIdx][rx][ry]` — 0..31. Only meaningful when
    /// `sao_type_idx == BandOffset`.
    pub band_position: u8,
    /// Bit-depth-scaled `SaoOffsetVal[cIdx][rx][ry][0..=4]` per eq. 153.
    /// `offset_val[0]` is always 0.
    pub offset_val: [i32; 5],
}

impl SaoCtb {
    /// Construct a band-offset CTB from raw spec fields (`sao_offset_abs`
    /// in the 0..31 range, `sao_offset_sign_flag` in {0, 1}). Applies
    /// the eq. 153 bit-depth shift.
    pub fn band_offset(
        band_position: u8,
        offset_abs: [u32; 4],
        offset_sign_flag: [u32; 4],
        bit_depth: u32,
    ) -> Self {
        Self {
            sao_type_idx: SaoTypeIdx::BandOffset,
            eo_class: SaoEoClass::Horizontal,
            band_position: band_position & 31,
            offset_val: derive_offset_val(offset_abs, offset_sign_flag, bit_depth),
        }
    }

    /// Construct an edge-offset CTB from raw spec fields. For EO,
    /// categories 1 + 2 are positive offsets (sign forced to 0) and
    /// categories 3 + 4 are negative (sign forced to 1) — the spec
    /// inferences in §7.4.12.3 are applied automatically here when the
    /// caller passes `[0; 4]` for `offset_sign_flag`.
    pub fn edge_offset(eo_class: SaoEoClass, offset_abs: [u32; 4], bit_depth: u32) -> Self {
        // Per §7.4.12.3 — "If i is equal to 0 or 1, sign is inferred 0;
        // otherwise sign is inferred 1" when EO is selected.
        let sign = [0, 0, 1, 1];
        Self {
            sao_type_idx: SaoTypeIdx::EdgeOffset,
            eo_class,
            band_position: 0,
            offset_val: derive_offset_val(offset_abs, sign, bit_depth),
        }
    }

    /// A "not applied" CTB — unchanged samples.
    pub fn not_applied() -> Self {
        Self::default()
    }
}

/// SAO parameters for *one* CTB across all three components.
#[derive(Clone, Copy, Debug, Default)]
pub struct SaoCtbParams {
    pub luma: SaoCtb,
    pub cb: SaoCtb,
    pub cr: SaoCtb,
}

impl SaoCtbParams {
    /// Pick the per-component params for a given `cIdx` (0/1/2).
    #[inline]
    pub fn component(&self, c_idx: u32) -> &SaoCtb {
        match c_idx {
            0 => &self.luma,
            1 => &self.cb,
            _ => &self.cr,
        }
    }
}

/// Per-picture SAO parameter array, indexed in raster CTB order.
///
/// Construct empty (every CTB defaulting to `NotApplied`) and then mutate
/// individual entries via [`Self::set`] for testing / fixture-driven use.
#[derive(Clone, Debug)]
pub struct SaoPicture {
    pub pic_width_in_ctbs_y: u32,
    pub pic_height_in_ctbs_y: u32,
    /// Length = `pic_width_in_ctbs_y * pic_height_in_ctbs_y`.
    ctbs: Vec<SaoCtbParams>,
}

impl SaoPicture {
    /// Allocate a fresh per-picture array with every entry defaulting
    /// to "not applied" for all three components.
    pub fn empty(pic_width_in_ctbs_y: u32, pic_height_in_ctbs_y: u32) -> Self {
        let n = (pic_width_in_ctbs_y as usize) * (pic_height_in_ctbs_y as usize);
        Self {
            pic_width_in_ctbs_y,
            pic_height_in_ctbs_y,
            ctbs: vec![SaoCtbParams::default(); n],
        }
    }

    fn idx(&self, rx: u32, ry: u32) -> usize {
        (ry as usize) * (self.pic_width_in_ctbs_y as usize) + (rx as usize)
    }

    /// Set the SAO parameters for the CTB at grid position `(rx, ry)`.
    pub fn set(&mut self, rx: u32, ry: u32, params: SaoCtbParams) {
        let i = self.idx(rx, ry);
        self.ctbs[i] = params;
    }

    /// Read the SAO parameters for the CTB at grid position `(rx, ry)`.
    pub fn get(&self, rx: u32, ry: u32) -> SaoCtbParams {
        let i = self.idx(rx, ry);
        self.ctbs[i]
    }

    /// True iff every entry is "not applied" for every component. Lets
    /// the apply driver short-circuit the whole pass.
    pub fn is_all_not_applied(&self) -> bool {
        self.ctbs.iter().all(|p| {
            p.luma.sao_type_idx == SaoTypeIdx::NotApplied
                && p.cb.sao_type_idx == SaoTypeIdx::NotApplied
                && p.cr.sao_type_idx == SaoTypeIdx::NotApplied
        })
    }
}

/// Picture-level configuration the SAO driver needs.
#[derive(Clone, Copy, Debug)]
pub struct SaoConfig {
    /// `sh_sao_luma_used_flag` — gates `cIdx = 0`.
    pub luma_used: bool,
    /// `sh_sao_chroma_used_flag` — gates `cIdx = 1` and `cIdx = 2`.
    pub chroma_used: bool,
    pub bit_depth: u32,
    /// `CtbLog2SizeY` — log2 of the CTB size in luma samples.
    pub ctb_log2_size_y: u32,
    /// `sps_chroma_format_idc` — 0 (monochrome), 1 (4:2:0), 2 (4:2:2),
    /// 3 (4:4:4). Determines the per-component `SubWidthC` / `SubHeightC`
    /// scaling.
    pub chroma_format_idc: u32,
}

impl SaoConfig {
    /// `(SubWidthC, SubHeightC)` for the chroma planes per §6.2.
    pub fn chroma_subsampling(&self) -> (u32, u32) {
        match self.chroma_format_idc {
            1 => (2, 2), // 4:2:0
            2 => (2, 1), // 4:2:2
            3 => (1, 1), // 4:4:4
            _ => (1, 1),
        }
    }
}

/// Eq. 153 — `SaoOffsetVal[i + 1] = (1 - 2 * sign) * (abs << shift)` with
/// `shift = BitDepth - Min(10, BitDepth)`. `SaoOffsetVal[0] = 0`.
fn derive_offset_val(offset_abs: [u32; 4], offset_sign_flag: [u32; 4], bit_depth: u32) -> [i32; 5] {
    let shift = (bit_depth as i32) - bit_depth.min(10) as i32;
    let mut out = [0i32; 5];
    for i in 0..4 {
        let sign_term = 1 - 2 * (offset_sign_flag[i] as i32 & 1);
        let mag = (offset_abs[i] as i32) << shift;
        out[i + 1] = sign_term * mag;
    }
    out
}

/// Top-level SAO driver invoked by the CTU walker. Implements the
/// §8.8.4.1 ordered scan: for every CTB, apply the §8.8.4.2 CTB
/// modification process per component, gated on `sh_sao_*_used_flag`.
///
/// The picture is modified *in place*. Per §8.8.4.1 the spec defines a
/// separate `saoPicture` aux buffer initialised from `recPicture`; in
/// our path SAO reads always come from the *pre-SAO* sample values, so
/// we stage one copy of each plane up front and read the neighbour
/// samples from that copy. Without this staging, an EO classification on
/// CTB N would see SAO-modified samples in CTB N − 1 along the inter-CTB
/// boundary, breaking the "operates on the recPicture, writes to
/// saoPicture" semantics.
pub fn apply_sao(out: &mut PictureBuffer, sao_pic: &SaoPicture, cfg: &SaoConfig) {
    if sao_pic.is_all_not_applied() {
        return;
    }
    if !cfg.luma_used && !cfg.chroma_used {
        return;
    }
    // Stage pre-SAO copies of each plane that has anything to do. This
    // mirrors `saoPicture` in eq. 1426 / 1429: reads come from the
    // pre-SAO buffer, writes hit the live frame.
    let luma_pre = if cfg.luma_used {
        Some(out.luma.samples.clone())
    } else {
        None
    };
    let cb_pre = if cfg.chroma_used && cfg.chroma_format_idc != 0 {
        Some(out.cb.samples.clone())
    } else {
        None
    };
    let cr_pre = if cfg.chroma_used && cfg.chroma_format_idc != 0 {
        Some(out.cr.samples.clone())
    } else {
        None
    };

    let (sub_w, sub_h) = cfg.chroma_subsampling();
    let ctb_size_y = 1u32 << cfg.ctb_log2_size_y;

    for ry in 0..sao_pic.pic_height_in_ctbs_y {
        for rx in 0..sao_pic.pic_width_in_ctbs_y {
            let p = sao_pic.get(rx, ry);
            if cfg.luma_used && p.luma.sao_type_idx != SaoTypeIdx::NotApplied {
                let pre = luma_pre.as_ref().unwrap();
                apply_sao_ctb(
                    &mut out.luma,
                    pre,
                    rx,
                    ry,
                    ctb_size_y,
                    ctb_size_y,
                    cfg.bit_depth,
                    &p.luma,
                );
            }
            if cfg.chroma_used && cfg.chroma_format_idc != 0 {
                let n_ctb_sw = ctb_size_y / sub_w;
                let n_ctb_sh = ctb_size_y / sub_h;
                if p.cb.sao_type_idx != SaoTypeIdx::NotApplied {
                    let pre = cb_pre.as_ref().unwrap();
                    apply_sao_ctb(
                        &mut out.cb,
                        pre,
                        rx,
                        ry,
                        n_ctb_sw,
                        n_ctb_sh,
                        cfg.bit_depth,
                        &p.cb,
                    );
                }
                if p.cr.sao_type_idx != SaoTypeIdx::NotApplied {
                    let pre = cr_pre.as_ref().unwrap();
                    apply_sao_ctb(
                        &mut out.cr,
                        pre,
                        rx,
                        ry,
                        n_ctb_sw,
                        n_ctb_sh,
                        cfg.bit_depth,
                        &p.cr,
                    );
                }
            }
        }
    }
}

/// §8.8.4.2 CTB modification process. Reads the *pre-SAO* sample buffer
/// `pre` (laid out at the same `stride / width / height` as `plane`),
/// classifies each sample, and writes the offset-adjusted, clipped
/// result into `plane`.
#[allow(clippy::too_many_arguments)]
fn apply_sao_ctb(
    plane: &mut PicturePlane,
    pre: &[u8],
    rx: u32,
    ry: u32,
    n_ctb_sw: u32,
    n_ctb_sh: u32,
    bit_depth: u32,
    params: &SaoCtb,
) {
    // Eq. 1426: top-left of this CTB in component-space samples.
    let x_ctb = (rx * n_ctb_sw) as usize;
    let y_ctb = (ry * n_ctb_sh) as usize;
    let stride = plane.stride;
    let pw = plane.width as i32;
    let ph = plane.height as i32;
    // Clip the iteration to the picture interior (edge CTBs may not
    // span the full nCtbS{w,h}).
    let i_max = (n_ctb_sw as usize).min(plane.width.saturating_sub(x_ctb));
    let j_max = (n_ctb_sh as usize).min(plane.height.saturating_sub(y_ctb));
    let max_val = (1i32 << bit_depth) - 1;
    let max_val_8 = max_val.min(255);

    match params.sao_type_idx {
        SaoTypeIdx::NotApplied => {}
        SaoTypeIdx::EdgeOffset => {
            let (h0, v0) = params.eo_class.pos0();
            let (h1, v1) = params.eo_class.pos1();
            for j in 0..j_max {
                for i in 0..i_max {
                    let xs = (x_ctb + i) as i32;
                    let ys = (y_ctb + j) as i32;
                    // Boundary check (eq. 1429): if either neighbour
                    // would land outside the picture, edgeIdx = 0
                    // (no-op).
                    let nx0 = xs + h0;
                    let ny0 = ys + v0;
                    let nx1 = xs + h1;
                    let ny1 = ys + v1;
                    let outside = nx0 < 0
                        || ny0 < 0
                        || nx1 < 0
                        || ny1 < 0
                        || nx0 >= pw
                        || ny0 >= ph
                        || nx1 >= pw
                        || ny1 >= ph;
                    if outside {
                        // edgeIdx = 0 → offset_val[0] = 0. Copy through.
                        let v = pre[(ys as usize) * stride + (xs as usize)] as i32;
                        plane.samples[(ys as usize) * stride + (xs as usize)] =
                            v.clamp(0, max_val_8) as u8;
                        continue;
                    }
                    // Eq. 1431: edgeIdx = 2 + sign(p − n0) + sign(p − n1).
                    let p = pre[(ys as usize) * stride + (xs as usize)] as i32;
                    let n0 = pre[(ny0 as usize) * stride + (nx0 as usize)] as i32;
                    let n1 = pre[(ny1 as usize) * stride + (nx1 as usize)] as i32;
                    let mut edge_idx = 2 + sign(p - n0) + sign(p - n1);
                    // Eq. 1432: remap [0, 1, 2] → [1, 2, 0]; [3, 4]
                    // unchanged.
                    if edge_idx <= 2 {
                        edge_idx = if edge_idx == 2 { 0 } else { edge_idx + 1 };
                    }
                    let off = if (1..=4).contains(&edge_idx) {
                        params.offset_val[edge_idx as usize]
                    } else {
                        0
                    };
                    let new = (p + off).clamp(0, max_val);
                    plane.samples[(ys as usize) * stride + (xs as usize)] =
                        new.min(max_val_8) as u8;
                }
            }
        }
        SaoTypeIdx::BandOffset => {
            // Eq. 1434: build the band table — entries 0..3 of the four
            // active bands carry offsets indexed 1..4.
            let band_shift = (bit_depth as i32) - 5;
            let mut band_table = [0i32; 32];
            let left_class = params.band_position as i32;
            for k in 0..4 {
                band_table[((k + left_class) & 31) as usize] = k + 1;
            }
            for j in 0..j_max {
                for i in 0..i_max {
                    let xs = x_ctb + i;
                    let ys = y_ctb + j;
                    let p = pre[ys * stride + xs] as i32;
                    let band_idx = if band_shift >= 0 {
                        band_table[(p >> band_shift) as usize & 31]
                    } else {
                        // bit_depth < 5 — treat as no shift (degenerate
                        // path that the spec never reaches; clamp the
                        // sample first).
                        band_table[(p & 31) as usize]
                    };
                    let off = if (1..=4).contains(&band_idx) {
                        params.offset_val[band_idx as usize]
                    } else {
                        0
                    };
                    let new = (p + off).clamp(0, max_val);
                    plane.samples[ys * stride + xs] = new.min(max_val_8) as u8;
                }
            }
        }
    }
}

/// `Sign(x)` per the spec convention: −1 for x < 0, 0 for x = 0, 1 for
/// x > 0.
#[inline]
fn sign(x: i32) -> i32 {
    match x.cmp(&0) {
        std::cmp::Ordering::Less => -1,
        std::cmp::Ordering::Equal => 0,
        std::cmp::Ordering::Greater => 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fresh_buf(w: usize, h: usize, seed: u8) -> PictureBuffer {
        PictureBuffer {
            luma: PicturePlane::filled(w, h, seed),
            cb: PicturePlane::filled(w / 2, h / 2, 128),
            cr: PicturePlane::filled(w / 2, h / 2, 128),
        }
    }

    fn cfg_8bit() -> SaoConfig {
        SaoConfig {
            luma_used: true,
            chroma_used: false,
            bit_depth: 8,
            ctb_log2_size_y: 5, // 32x32 CTB
            chroma_format_idc: 1,
        }
    }

    /// Eq. 153 sanity: 8-bit shift = 8 - min(10, 8) = 0 → no scaling.
    #[test]
    fn offset_val_8bit_no_shift() {
        let v = derive_offset_val([1, 2, 3, 4], [0, 1, 0, 1], 8);
        assert_eq!(v, [0, 1, -2, 3, -4]);
    }

    /// Eq. 153 at 10-bit: shift = 0; at 12-bit shift = 2 → ×4.
    #[test]
    fn offset_val_high_bit_depth_shift() {
        let v = derive_offset_val([1, 1, 1, 1], [0, 0, 0, 0], 12);
        assert_eq!(v, [0, 4, 4, 4, 4]);
    }

    /// Table 11 / 44: each EO class returns the spec-defined neighbour
    /// position pair.
    #[test]
    fn eo_class_table_44() {
        assert_eq!(SaoEoClass::Horizontal.pos0(), (-1, 0));
        assert_eq!(SaoEoClass::Horizontal.pos1(), (1, 0));
        assert_eq!(SaoEoClass::Vertical.pos0(), (0, -1));
        assert_eq!(SaoEoClass::Vertical.pos1(), (0, 1));
        assert_eq!(SaoEoClass::Deg135.pos0(), (-1, -1));
        assert_eq!(SaoEoClass::Deg135.pos1(), (1, 1));
        assert_eq!(SaoEoClass::Deg45.pos0(), (1, -1));
        assert_eq!(SaoEoClass::Deg45.pos1(), (-1, 1));
    }

    /// Empty SaoPicture short-circuits — no plane is mutated.
    #[test]
    fn apply_sao_no_op_when_all_not_applied() {
        let mut buf = fresh_buf(32, 32, 100);
        let sao_pic = SaoPicture::empty(1, 1);
        apply_sao(&mut buf, &sao_pic, &cfg_8bit());
        assert!(buf.luma.samples.iter().all(|&v| v == 100));
    }

    /// `luma_used = false` AND `chroma_used = false` → no-op even when
    /// CTB params are populated.
    #[test]
    fn apply_sao_no_op_when_flags_off() {
        let mut buf = fresh_buf(32, 32, 100);
        let mut sao_pic = SaoPicture::empty(1, 1);
        sao_pic.set(
            0,
            0,
            SaoCtbParams {
                luma: SaoCtb::band_offset(0, [10, 0, 0, 0], [0, 0, 0, 0], 8),
                ..Default::default()
            },
        );
        let mut cfg = cfg_8bit();
        cfg.luma_used = false;
        cfg.chroma_used = false;
        apply_sao(&mut buf, &sao_pic, &cfg);
        assert!(buf.luma.samples.iter().all(|&v| v == 100));
    }

    /// BO with a single non-zero offset on band containing seed value
    /// 100 should shift all samples by +10. Band index of 100 is
    /// 100 >> 3 = 12. Pick `band_position = 12` → band-0 of the four
    /// active bands lands on band 12 → bandIdx = 1 → offset[1] = +10.
    #[test]
    fn band_offset_shifts_matching_band() {
        let mut buf = fresh_buf(32, 32, 100);
        let mut sao_pic = SaoPicture::empty(1, 1);
        sao_pic.set(
            0,
            0,
            SaoCtbParams {
                luma: SaoCtb::band_offset(12, [10, 0, 0, 0], [0, 0, 0, 0], 8),
                ..Default::default()
            },
        );
        apply_sao(&mut buf, &sao_pic, &cfg_8bit());
        assert!(buf.luma.samples.iter().all(|&v| v == 110));
    }

    /// BO with a non-matching band leaves the samples untouched.
    #[test]
    fn band_offset_non_matching_band_no_op() {
        let mut buf = fresh_buf(32, 32, 100);
        let mut sao_pic = SaoPicture::empty(1, 1);
        // band of 100 = 12; pick band_position = 0 → bands 0..3 active,
        // none match → bandIdx = 0 → offset_val[0] = 0.
        sao_pic.set(
            0,
            0,
            SaoCtbParams {
                luma: SaoCtb::band_offset(0, [50, 50, 50, 50], [0, 0, 0, 0], 8),
                ..Default::default()
            },
        );
        apply_sao(&mut buf, &sao_pic, &cfg_8bit());
        assert!(buf.luma.samples.iter().all(|&v| v == 100));
    }

    /// BO clipping: a strong positive offset must clip to 255 (8-bit).
    #[test]
    fn band_offset_clips_to_max() {
        let mut buf = fresh_buf(32, 32, 250);
        let mut sao_pic = SaoPicture::empty(1, 1);
        // 250 >> 3 = 31. band_position = 31 → bands 31, 0, 1, 2 active;
        // band 31 → bandIdx = 1 → +50.
        sao_pic.set(
            0,
            0,
            SaoCtbParams {
                luma: SaoCtb::band_offset(31, [50, 0, 0, 0], [0, 0, 0, 0], 8),
                ..Default::default()
            },
        );
        apply_sao(&mut buf, &sao_pic, &cfg_8bit());
        assert!(buf.luma.samples.iter().all(|&v| v == 255));
    }

    /// EO horizontal class on a flat plane: every interior sample sees
    /// `p == n0 == n1` → edgeIdx = 2 + 0 + 0 = 2 → remapped to 0 → no
    /// offset applied. (Pixels on the left/right column also yield 0
    /// via the boundary rule.) Therefore the plane is unchanged.
    #[test]
    fn edge_offset_flat_plane_no_op() {
        let mut buf = fresh_buf(32, 32, 100);
        let mut sao_pic = SaoPicture::empty(1, 1);
        sao_pic.set(
            0,
            0,
            SaoCtbParams {
                luma: SaoCtb::edge_offset(SaoEoClass::Horizontal, [10, 10, 10, 10], 8),
                ..Default::default()
            },
        );
        apply_sao(&mut buf, &sao_pic, &cfg_8bit());
        assert!(buf.luma.samples.iter().all(|&v| v == 100));
    }

    /// EO horizontal: a 32-row pattern `..., 100, 110, 100, 110, ...`
    /// causes each "100" with neighbours 110, 110 to yield edgeIdx =
    /// 2 + (-1) + (-1) = 0 → remapped to 1 → category 1. With offset[1]
    /// = +5, that sample becomes 105.
    /// Each "110" with neighbours 100, 100 yields edgeIdx = 2 + 1 + 1 =
    /// 4 → category 4. With offset[4] = -2 (sign forced to 1 for i=3),
    /// that sample becomes 108.
    #[test]
    fn edge_offset_horizontal_pattern() {
        let mut buf = fresh_buf(32, 1, 0);
        // Build a 1-row, 32-col plane with alternating 100 / 110.
        for x in 0..32 {
            buf.luma.samples[x] = if x % 2 == 0 { 100 } else { 110 };
        }
        let mut sao_pic = SaoPicture::empty(1, 1);
        sao_pic.set(
            0,
            0,
            SaoCtbParams {
                // EO offsets per spec semantics: cats 1, 2 are positive,
                // cats 3, 4 negative.
                luma: SaoCtb::edge_offset(SaoEoClass::Horizontal, [5, 0, 0, 2], 8),
                ..Default::default()
            },
        );
        // CTB is 32x32 but the plane is only 32x1 — j_max clamps to 1.
        apply_sao(&mut buf, &sao_pic, &cfg_8bit());
        // Interior columns (x in 1..31): 100 → 105, 110 → 108.
        // Column 0 has no left neighbour → boundary → unchanged (100).
        // Column 31 has no right neighbour → boundary → unchanged (110).
        assert_eq!(buf.luma.samples[0], 100);
        assert_eq!(buf.luma.samples[31], 110);
        for x in 1..31 {
            let want = if x % 2 == 0 { 105 } else { 108 };
            assert_eq!(buf.luma.samples[x], want, "x = {}", x);
        }
    }

    /// `SaoTypeIdx::from_u32` decodes the spec values 0..2.
    #[test]
    fn sao_type_idx_decode() {
        assert_eq!(SaoTypeIdx::from_u32(0), SaoTypeIdx::NotApplied);
        assert_eq!(SaoTypeIdx::from_u32(1), SaoTypeIdx::BandOffset);
        assert_eq!(SaoTypeIdx::from_u32(2), SaoTypeIdx::EdgeOffset);
        // Unknown values clamp to NotApplied per defensive default.
        assert_eq!(SaoTypeIdx::from_u32(99), SaoTypeIdx::NotApplied);
    }

    /// `SaoEoClass::from_u32` decodes the spec values 0..3.
    #[test]
    fn sao_eo_class_decode() {
        assert_eq!(SaoEoClass::from_u32(0), SaoEoClass::Horizontal);
        assert_eq!(SaoEoClass::from_u32(1), SaoEoClass::Vertical);
        assert_eq!(SaoEoClass::from_u32(2), SaoEoClass::Deg135);
        assert_eq!(SaoEoClass::from_u32(3), SaoEoClass::Deg45);
    }

    /// SaoPicture get/set round-trip.
    #[test]
    fn sao_picture_set_get() {
        let mut sao_pic = SaoPicture::empty(2, 2);
        let p = SaoCtbParams {
            luma: SaoCtb::edge_offset(SaoEoClass::Vertical, [1, 2, 3, 4], 8),
            ..Default::default()
        };
        sao_pic.set(1, 0, p);
        let g = sao_pic.get(1, 0);
        assert_eq!(g.luma.sao_type_idx, SaoTypeIdx::EdgeOffset);
        assert_eq!(g.luma.eo_class, SaoEoClass::Vertical);
        // Other entries default.
        let g0 = sao_pic.get(0, 0);
        assert_eq!(g0.luma.sao_type_idx, SaoTypeIdx::NotApplied);
        assert!(!sao_pic.is_all_not_applied());
    }

    /// EO 135-degree on a 3x3 ramp: only the centre sample is interior.
    /// Plane =
    ///   100 100 100
    ///   100 110 100
    ///   100 100 100
    /// 135-deg neighbours are (-1,-1) and (+1,+1) — both are 100, centre
    /// is 110. edgeIdx = 2 + 1 + 1 = 4 → category 4 → -2 offset → 108.
    #[test]
    fn edge_offset_135deg_peak() {
        let mut buf = PictureBuffer {
            luma: PicturePlane::filled(3, 3, 100),
            cb: PicturePlane::filled(2, 2, 128),
            cr: PicturePlane::filled(2, 2, 128),
        };
        buf.luma.samples[1 * 3 + 1] = 110;
        let mut sao_pic = SaoPicture::empty(1, 1);
        sao_pic.set(
            0,
            0,
            SaoCtbParams {
                luma: SaoCtb::edge_offset(SaoEoClass::Deg135, [5, 0, 0, 2], 8),
                ..Default::default()
            },
        );
        apply_sao(&mut buf, &sao_pic, &cfg_8bit());
        assert_eq!(buf.luma.samples[1 * 3 + 1], 108);
        // Corners and edges are boundary → unchanged.
        assert_eq!(buf.luma.samples[0], 100);
    }

    /// Chroma SAO with `chroma_used = true`. 4:2:0 chroma plane is half
    /// the size — feed a flat band that matches a 128 seed value.
    /// 128 >> 3 = 16 → band_position = 16 → bandIdx = 1 → +20.
    #[test]
    fn chroma_band_offset_4_2_0() {
        let mut buf = fresh_buf(32, 32, 100);
        let mut sao_pic = SaoPicture::empty(1, 1);
        sao_pic.set(
            0,
            0,
            SaoCtbParams {
                cb: SaoCtb::band_offset(16, [20, 0, 0, 0], [0, 0, 0, 0], 8),
                cr: SaoCtb::band_offset(16, [20, 0, 0, 0], [0, 0, 0, 0], 8),
                ..Default::default()
            },
        );
        let mut cfg = cfg_8bit();
        cfg.luma_used = false;
        cfg.chroma_used = true;
        apply_sao(&mut buf, &sao_pic, &cfg);
        // Cb / Cr were seeded to 128 → all shift to 148.
        assert!(buf.cb.samples.iter().all(|&v| v == 148));
        assert!(buf.cr.samples.iter().all(|&v| v == 148));
        // Luma untouched (luma_used = false).
        assert!(buf.luma.samples.iter().all(|&v| v == 100));
    }

    /// SaoCtbParams::component routes correctly.
    #[test]
    fn ctb_params_component_routing() {
        let p = SaoCtbParams {
            luma: SaoCtb::band_offset(1, [1, 0, 0, 0], [0, 0, 0, 0], 8),
            cb: SaoCtb::band_offset(2, [2, 0, 0, 0], [0, 0, 0, 0], 8),
            cr: SaoCtb::band_offset(3, [3, 0, 0, 0], [0, 0, 0, 0], 8),
        };
        assert_eq!(p.component(0).band_position, 1);
        assert_eq!(p.component(1).band_position, 2);
        assert_eq!(p.component(2).band_position, 3);
    }

    /// EO vertical class — symmetrical to horizontal but neighbours
    /// are above / below.
    #[test]
    fn edge_offset_vertical_pattern() {
        // Build a 1-col, 32-row plane with alternating 100 / 110.
        let mut buf = PictureBuffer {
            luma: PicturePlane::filled(1, 32, 0),
            cb: PicturePlane::filled(1, 16, 128),
            cr: PicturePlane::filled(1, 16, 128),
        };
        for y in 0..32 {
            buf.luma.samples[y] = if y % 2 == 0 { 100 } else { 110 };
        }
        let mut sao_pic = SaoPicture::empty(1, 1);
        sao_pic.set(
            0,
            0,
            SaoCtbParams {
                luma: SaoCtb::edge_offset(SaoEoClass::Vertical, [5, 0, 0, 2], 8),
                ..Default::default()
            },
        );
        apply_sao(&mut buf, &sao_pic, &cfg_8bit());
        assert_eq!(buf.luma.samples[0], 100);
        assert_eq!(buf.luma.samples[31], 110);
        for y in 1..31 {
            let want = if y % 2 == 0 { 105 } else { 108 };
            assert_eq!(buf.luma.samples[y], want, "y = {}", y);
        }
    }
}
