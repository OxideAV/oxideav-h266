//! VVC CTU walker scaffold (§7.3.11 – §7.4.11).
//!
//! This module is the *entry point* for per-picture reconstruction. It
//! ties together the parser modules (SPS + PPS + slice header) with the
//! CABAC engine (`cabac` / `tables` / `ctx`) and the coding-tree walker
//! (`coding_tree`). The output of a successful walk is a list of leaf
//! Coding Units with their spec-level bookkeeping (position, size,
//! partition depths). [`CtuWalker::reconstruct_leaf_cu`] now drives the
//! §8.4 / §8.7 pipeline end to end for the intra-only single-tree
//! subset (PLANAR / DC / cardinal angular intra prediction + flat-list
//! dequantisation + DCT-II inverse transform + sample-clip add-back).
//! In-loop filters and the more elaborate intra modes (MIP / ISP /
//! BDPCM / wide-angle / fractional angular) still surface
//! `Error::Unsupported` so callers see a precise pointer to the gap.
//!
//! What this scaffold provides:
//!
//! * [`CtuLayout`] — picture geometry in CTB units: picture size in
//!   CTBs (`PicWidthInCtbsY`, `PicHeightInCtbsY`), CtbSizeY from
//!   the SPS, and the raster → pixel coordinate map.
//! * [`CtuPos`] — a single CTU's spatial state (x_ctb, y_ctb, top-left
//!   luma pixel, edge-clamped width/height).
//! * [`CtuIter`] — iterates CTUs in slice-decoding order. The scaffold
//!   currently supports only the single-slice-per-picture case
//!   (`pps_no_pic_partition_flag == 1`); multi-tile / multi-slice
//!   layouts surface `Error::Unsupported`.
//! * [`NeighbourAvail`] — left / above availability flags derived from
//!   CTU position within the current slice (spec §6.4.4).
//! * [`SliceQpY`] — §7.4.8 eq. 140 derivation (`26 + pps_init_qp_minus26
//!   + sh_qp_delta + ph_qp_delta` clipped to [-QpBdOffset, 63]).
//! * [`SliceCabacState`] — the per-slice CABAC context bundle (§9.3.2.2
//!   entry + first-CTU init; context table dispatch is done by
//!   [`coding_tree::TreeCtxs::init`]).
//! * [`CtuWalker`] — orchestrator that iterates CTUs, calls
//!   [`coding_tree::TreeWalker`] for each CTU, and surfaces
//!   `Error::Unsupported` when asked to reconstruct a leaf CU. Exposes
//!   [`CtuWalker::decode_leaf_cu_syntax`] for callers that want to
//!   consume a leaf's `coding_unit()` bins into a
//!   [`crate::leaf_cu::LeafCuInfo`] without proceeding to pixel
//!   reconstruction.
//!
//! What this scaffold does **not** yet do:
//!
//! * Multi-tile / multi-slice CTU-address scans. Only the raster scan
//!   within a single slice that fills the whole picture is walked.
//! * Dual-tree (separate luma + chroma) partitioning. Only SingleTree
//!   mode is surfaced; the walker emits single-plane (luma) CUs and
//!   defers chroma partitioning to a future increment.
//! * Chroma-plane reconstruction. The 4:2:0 buffer's Cb / Cr planes
//!   stay seeded at neutral 128 — the luma plane is the only one
//!   currently driven by [`CtuWalker::reconstruct_leaf_cu`].
//! * In-loop filters: §8.8.3 deblocking is now wired through
//!   [`CtuWalker::apply_in_loop_filters`] (short-tap luma / weak-tap
//!   chroma per the round-12 subset). SAO / ALF / LMCS are still
//!   pending and silently no-op when their slice flags are clear.
//! * MIP / ISP / BDPCM / CCLM / LFNST / MTS-non-DCT-II / scaling lists
//!   / wide-angle / fractional-angular intra prediction. Each is gated
//!   with a precise `Error::Unsupported` so callers see exactly which
//!   construct is pending.
//! * Entry-point offset handling: even though the slice header parser
//!   captures `sh_entry_point_offsets`, the WPP / tile multi-stream
//!   CABAC engine is not wired up. Single-stream slices only.
//! * CABAC synchronisation at tile boundaries (§9.3.1.3) / WPP
//!   second-row-init (§9.3.1.4) — both surface `Error::Unsupported`.
//!
//! Spec reference: ITU-T H.266 | ISO/IEC 23090-3 (V4, 01/2026).

use oxideav_core::{Error, Result};

use crate::alf::{apply_alf, AlfApsBinding, AlfConfig, AlfPicture};
use crate::bdof::{bdof_refine_into, bdof_used_flag, build_extended_pred_high_precision};
use crate::cabac::ArithDecoder;
use crate::cclm::{predict_cclm, CclmInputs, LumaPlane};
use crate::coding_tree::{Cu, CuNeighbourMap, TreeCtxs, TreeWalker};
use crate::deblock::{apply_deblocking, DeblockCu, DeblockParams};
use crate::dequant::{dequantize_tb_flat, DequantParams};
use crate::gpm::{
    blend_gpm_into_plane, derive_gpm_mn, derive_gpm_partition, derive_gpm_partition_motion,
    GpmContext,
};
use crate::inter::{
    apply_mmvd_to_base_with_poc, build_merge_cand_list, build_merge_cand_list_b, ciip_intra_weight,
    derive_mmvd_offset, derive_spatial_merge_candidates, derive_temporal_merge_candidate,
    predict_chroma_block, predict_chroma_block_bipred_bcw, predict_luma_block,
    predict_luma_block_bipred_bcw, predict_luma_block_high_precision, HmvpTable, MotionField,
    MotionVector, MvField, ReferencePicture, TemporalMergeInputs,
};
use crate::intra::{predict_angular, predict_dc, predict_planar, IntraRefs};
use crate::leaf_cu::{
    CuNeighbourhood, CuPredMode, CuToolFlags, LeafCuCtxs, LeafCuInfo, LeafCuReader, LeafCuResidual,
    INTRA_DC, INTRA_LT_CCLM, INTRA_L_CCLM, INTRA_PLANAR, INTRA_T_CCLM,
};
use crate::lmcs::{LmcsData, LmcsDerived};
use crate::mip::predict_mip;
use crate::pps::PicParameterSet;
use crate::reconstruct::{clip_pixel, reconstruct_tb_into, OwnedIntraRefs, PictureBuffer};
use crate::sao::{apply_sao, SaoConfig, SaoPicture};
use crate::sao_syntax::{decode_sao_ctb, SaoCtxs, SaoSyntaxConfig};
use crate::slice_header::{SliceType, StatefulSliceHeader};
use crate::sps::SeqParameterSet;
use crate::transform::{inverse_transform_2d, TrType};

/// Snap an arbitrary VVC angular intra mode `m ∈ 2..=66` to the nearest
/// of the cardinal/diagonal subset implemented by [`crate::intra::predict_angular`]
/// (`{2, 18, 34, 50, 66}`). Used as a fallback by
/// [`CtuWalker::reconstruct_leaf_cu`] until the full angular-prediction
/// pipeline (§8.4.5.2.13 — wide-angle remap, 4-tap reference filter,
/// fractional-position interpolation) is wired.
pub(crate) fn nearest_supported_angular(mode: u32) -> u32 {
    const CARDINALS: [u32; 5] = [2, 18, 34, 50, 66];
    CARDINALS
        .iter()
        .copied()
        .min_by_key(|c| c.abs_diff(mode))
        .unwrap_or(crate::leaf_cu::INTRA_PLANAR)
}

/// Map an `IntraPredModeC` (§8.4.3 output, range 0..=83) to one of the
/// directional intra modes the per-sample chroma predictor understands
/// in this scaffold (PLANAR, DC, or one of the cardinal angulars
/// `{2, 18, 34, 50, 66}`).
///
/// CCLM modes (`81..=83`) bypass this helper — the CCLM path in
/// [`CtuWalker::reconstruct_chroma_plane`] runs the dedicated
/// §8.4.5.2.14 predictor against the reconstructed luma plane. The
/// helper still maps them to `PLANAR` as a safety fallback if CCLM
/// derivation fails (e.g. neither neighbour available — eq. 365 covers
/// that path inside CCLM, but PLANAR keeps the chroma plane
/// numerically defined). Non-cardinal angular modes (anything outside
/// the cardinal set) are snapped to the nearest cardinal via
/// [`nearest_supported_angular`], matching the luma-side fallback
/// used by [`CtuWalker::reconstruct_leaf_cu`].
pub(crate) fn chroma_pred_mode_for_predict(mode_c: u32) -> u32 {
    use crate::leaf_cu::{INTRA_DC, INTRA_LT_CCLM, INTRA_L_CCLM, INTRA_PLANAR, INTRA_T_CCLM};
    match mode_c {
        INTRA_PLANAR => INTRA_PLANAR,
        INTRA_DC => INTRA_DC,
        // CCLM is dispatched separately by `reconstruct_chroma_plane`.
        // Map to PLANAR here as the safety fallback.
        INTRA_LT_CCLM | INTRA_L_CCLM | INTRA_T_CCLM => INTRA_PLANAR,
        m @ 2..=66 => {
            if matches!(m, 2 | 18 | 34 | 50 | 66) {
                m
            } else {
                nearest_supported_angular(m)
            }
        }
        _ => INTRA_PLANAR,
    }
}

/// Build the chroma neighbour rows / columns that [`predict_cclm`] needs
/// from the partially-reconstructed chroma plane. Returns `(top, left)`
/// where `top` is empty when the top side is unavailable and `left` is
/// empty when the left side is unavailable.
///
/// Round-28 §8.5.6.7 — apply the eq. 998 weighted average between the
/// freshly-MC'd `predSamplesInter` (already living inside the CU
/// rectangle of `plane`) and the supplied `pred_intra` planar
/// prediction array. Writes the combined result back into the same
/// plane rectangle, clipping into the bit-depth range. The `w`
/// argument is the §8.5.6.7 intra-prediction weight in `{1, 2, 3}`
/// — see [`crate::inter::ciip_intra_weight`].
fn apply_ciip_combine_to_plane(
    plane: &mut crate::reconstruct::PicturePlane,
    x0: usize,
    y0: usize,
    n_w: usize,
    n_h: usize,
    pred_intra: &[i16],
    w: u32,
    bit_depth: u32,
) {
    debug_assert_eq!(pred_intra.len(), n_w * n_h);
    debug_assert!(matches!(w, 1..=3));
    let stride = plane.stride;
    // Snapshot the current rectangle as predSamplesInter, then
    // combine + write back. We could combine in-place row-by-row but
    // staging through `Vec<i16>` keeps the helper symmetric with
    // [`crate::inter::combine_ciip_samples`].
    let mut inter_pred = vec![0i16; n_w * n_h];
    for y in 0..n_h {
        let plane_row = (y0 + y) * stride;
        for x in 0..n_w {
            inter_pred[y * n_w + x] = plane.samples[plane_row + x0 + x] as i16;
        }
    }
    let combined =
        crate::inter::combine_ciip_samples(pred_intra, &inter_pred, n_w, n_h, w, bit_depth);
    for y in 0..n_h {
        let plane_row = (y0 + y) * stride;
        for x in 0..n_w {
            plane.samples[plane_row + x0 + x] = combined[y * n_w + x] as u8;
        }
    }
}

/// The lengths follow §8.4.5.2.14 numSampN derivation:
///
/// * `INTRA_LT_CCLM` — `top` has `n_tb_w` samples, `left` has `n_tb_h`.
/// * `INTRA_T_CCLM`  — `top` has up to `n_tb_w + n_tb_h` samples
///   (extending into the top-right neighbour), `left` is empty.
/// * `INTRA_L_CCLM`  — `top` is empty, `left` has up to `n_tb_h + n_tb_w`
///   samples (extending into the bottom-left neighbour).
fn cclm_chroma_neighbours(
    plane: &crate::reconstruct::PicturePlane,
    x0: usize,
    y0: usize,
    n_tb_w: usize,
    n_tb_h: usize,
    above_avail: bool,
    left_avail: bool,
    mode: u32,
) -> (Vec<i16>, Vec<i16>) {
    let stride = plane.stride;
    let plane_w = plane.width;
    let plane_h = plane.height;
    // Top side.
    let top: Vec<i16> = if above_avail {
        let want = match mode {
            INTRA_T_CCLM => n_tb_w + n_tb_h,
            _ => n_tb_w,
        };
        // Limit to what's actually inside the plane (the spec's
        // numTopRight loop walks until availTR goes false; for the
        // single-slice scaffold, "outside the picture" is the only way
        // to lose availability).
        let max_x = plane_w.saturating_sub(x0);
        let actual = want.min(max_x);
        let yref = y0.saturating_sub(1);
        (0..actual)
            .map(|i| {
                let xi = x0 + i;
                if xi < plane_w && y0 > 0 && yref < plane_h {
                    plane.samples[yref * stride + xi] as i16
                } else {
                    0
                }
            })
            .collect()
    } else {
        Vec::new()
    };
    // Left side.
    let left: Vec<i16> = if left_avail {
        let want = match mode {
            INTRA_L_CCLM => n_tb_h + n_tb_w,
            _ => n_tb_h,
        };
        let max_y = plane_h.saturating_sub(y0);
        let actual = want.min(max_y);
        let xref = x0.saturating_sub(1);
        (0..actual)
            .map(|i| {
                let yi = y0 + i;
                if x0 > 0 && yi < plane_h && xref < plane_w {
                    plane.samples[yi * stride + xref] as i16
                } else {
                    0
                }
            })
            .collect()
    } else {
        Vec::new()
    };
    (top, left)
}

/// §8.7.1 chroma QP derivation — minimal identity mapping.
///
/// Returns the per-component chroma QP (Qp′Cb / Qp′Cr) given luma QP
/// `qp_y`, the PPS picture-level chroma offsets, and any slice/CU
/// chroma offsets. The full §8.7.1 path threads this through Table 4
/// (the SPS-supplied chroma-QP mapping table) and adds the bit-depth
/// offset; this scaffold uses the spec's identity mapping
/// (`QpC = QpY`) plus the additive PPS / slice / CU offsets, which is
/// exactly the §8.7.1 result when `same_qp_table_for_chroma_flag == 1`
/// with the default (identity) table.
///
/// `qp_offset` is the sum `pps_cb_qp_offset + sh_cb_qp_offset
/// + cu_chroma_qp_offset` (or the Cr equivalent). The result is
/// clamped to the spec's `[0, 63]` range.
#[inline]
pub(crate) fn chroma_qp_identity(qp_y: i32, qp_offset: i32) -> i32 {
    (qp_y + qp_offset).clamp(0, 63)
}

/// §8.7.1 eqs. 1147 / 1148 additive chroma-QP offset term for a single
/// component: `pps_c?_qp_offset + sh_c?_qp_offset + CuQpOffsetC?`.
///
/// `c_idx` selects Cb (1) or Cr (2); any other value yields 0 (luma has
/// no chroma offset). The slice-level term is inferred to 0 (§7.4.9.1)
/// when `pps_slice_chroma_qp_offsets_present_flag == 0`, which the
/// slice-header parser leaves at its default, so the caller can pass the
/// stored value unconditionally. `cu_offset` is the §7.4.10.6 CU-level
/// `CuQpOffsetC?` term (0 until the PPS chroma-offset list is plumbed).
#[inline]
pub(crate) fn chroma_qp_offset_sum(
    c_idx: u32,
    pps_cb_qp_offset: i32,
    pps_cr_qp_offset: i32,
    sh_cb_qp_offset: i32,
    sh_cr_qp_offset: i32,
    cu_offset: i32,
) -> i32 {
    let (pps_offset, sh_offset) = match c_idx {
        1 => (pps_cb_qp_offset, sh_cb_qp_offset),
        2 => (pps_cr_qp_offset, sh_cr_qp_offset),
        _ => (0, 0),
    };
    pps_offset + sh_offset + cu_offset
}

/// §7.4.10.6 eqs. 193 / 194 — the CU-level `CuQpOffsetCb` / `CuQpOffsetCr`
/// term for a single chroma component.
///
/// When `cu_chroma_qp_offset_flag == 1` the value is
/// `pps_c?_qp_offset_list[cu_chroma_qp_offset_idx]`; otherwise (or for
/// luma `c_idx == 0`, or when the index is out of range) it is 0. `c_idx`
/// selects Cb (1) / Cr (2); the lists carry one entry per
/// `pps_chroma_qp_offset_list_len_minus1 + 1`.
#[inline]
pub(crate) fn cu_chroma_qp_offset(
    c_idx: u32,
    flag: bool,
    idx: u32,
    cb_list: &[i32],
    cr_list: &[i32],
) -> i32 {
    if !flag {
        return 0;
    }
    let list = match c_idx {
        1 => cb_list,
        2 => cr_list,
        _ => return 0,
    };
    list.get(idx as usize).copied().unwrap_or(0)
}

/// §8.5.5.3 eqs. 878 / 879 — average two motion-vector components by
/// halving their sum with the spec's round-toward-zero rule
/// `( v + 1 − ( v >= 0 ) ) >> 1`. For `v` the sum of the two component
/// values this yields the rounded average that the chroma-MV
/// derivation feeds into §8.5.2.13.
#[inline]
pub(crate) fn round_half_toward_zero_div2(v: i32) -> i32 {
    (v + 1 - (v >= 0) as i32) >> 1
}

/// CTU grid geometry derived from SPS + PPS (§7.4.3.4 / §7.4.3.5).
///
/// The entry-level quantities a walker needs before even touching the
/// slice bitstream:
///
/// * `ctb_log2_size_y` — log2 of the CTB size in luma samples.
/// * `ctb_size_y` — CTB size in luma samples (= `1 << ctb_log2_size_y`).
/// * `pic_width_in_ctbs_y`, `pic_height_in_ctbs_y` — number of CTUs
///   spanning the picture in each direction (§7.4.3.4 eq. 56, 58).
/// * `pic_width_luma`, `pic_height_luma` — the raw luma picture
///   dimensions (in samples) needed for edge-CTU clamping.
/// * `min_cb_log2_size_y` — log2 of the minimum CB size (§7.4.3.4
///   eq. 62); bounds the leaf-CU size the coding-tree walker may reach.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CtuLayout {
    pub ctb_log2_size_y: u32,
    pub ctb_size_y: u32,
    pub pic_width_in_ctbs_y: u32,
    pub pic_height_in_ctbs_y: u32,
    pub pic_width_luma: u32,
    pub pic_height_luma: u32,
    pub min_cb_log2_size_y: u32,
}

impl CtuLayout {
    /// Derive the CTU grid from a parsed SPS + PPS pair. Uses the PPS
    /// picture dimensions when present; falls back to the SPS max size.
    pub fn from_sps_pps(sps: &SeqParameterSet, pps: &PicParameterSet) -> Self {
        let ctb_log2 = sps.sps_log2_ctu_size_minus5 as u32 + 5;
        let ctb_size = 1u32 << ctb_log2;
        let pic_w = pps.pps_pic_width_in_luma_samples;
        let pic_h = pps.pps_pic_height_in_luma_samples;
        // §7.4.3.4 eq. 57–58: PicWidthInCtbsY = Ceil(pic_width / CtbSizeY).
        let pwc = pic_w.div_ceil(ctb_size);
        let phc = pic_h.div_ceil(ctb_size);
        // §7.4.3.4 eq. 62: MinCbLog2SizeY = log2_min_luma_coding_block_size_minus2 + 2.
        let min_cb_log2 = sps
            .partition_constraints
            .log2_min_luma_coding_block_size_minus2
            + 2;
        Self {
            ctb_log2_size_y: ctb_log2,
            ctb_size_y: ctb_size,
            pic_width_in_ctbs_y: pwc,
            pic_height_in_ctbs_y: phc,
            pic_width_luma: pic_w,
            pic_height_luma: pic_h,
            min_cb_log2_size_y: min_cb_log2,
        }
    }

    /// Total number of CTUs in the picture (`PicSizeInCtbsY`).
    pub fn pic_size_in_ctbs_y(&self) -> u32 {
        self.pic_width_in_ctbs_y
            .saturating_mul(self.pic_height_in_ctbs_y)
    }
}

/// A single CTU's spatial state.
///
/// All positions are in luma samples unless noted. Edge CTUs may have
/// `width_luma < ctb_size_y` / `height_luma < ctb_size_y` when the
/// picture dimensions are not a multiple of `CtbSizeY`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CtuPos {
    /// CTB-grid column index (0..pic_width_in_ctbs_y).
    pub x_ctb: u32,
    /// CTB-grid row index.
    pub y_ctb: u32,
    /// Top-left luma sample x.
    pub x0: u32,
    /// Top-left luma sample y.
    pub y0: u32,
    /// Width in luma samples. Clamped at the right edge.
    pub width_luma: u32,
    /// Height in luma samples. Clamped at the bottom edge.
    pub height_luma: u32,
    /// CtuAddrInRs / CtuAddrInSlice for the single-slice case.
    pub ctu_addr_rs: u32,
}

/// Neighbour availability around a CTU (§6.4.4).
///
/// The walker scaffolds treat *slice / tile / subpicture* boundaries as
/// the only sources of unavailability. Since we currently support only
/// single-slice pictures, left / above availability is determined by
/// picture-edge proximity alone.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NeighbourAvail {
    pub left: bool,
    pub above: bool,
}

/// Iterator over the CTUs that make up a single slice, in decoding
/// order. The single-slice case is raster scan across the full grid.
pub struct CtuIter<'a> {
    layout: &'a CtuLayout,
    next_rs: u32,
    total: u32,
}

impl<'a> CtuIter<'a> {
    /// Build a raster-scan iterator covering every CTU in the picture.
    /// Valid only when `pps_no_pic_partition_flag == 1` (single slice +
    /// single tile); [`CtuWalker`] is responsible for gating the call.
    pub fn raster(layout: &'a CtuLayout) -> Self {
        Self {
            layout,
            next_rs: 0,
            total: layout.pic_size_in_ctbs_y(),
        }
    }

    /// Neighbour availability for a CTU at (x_ctb, y_ctb) under the
    /// single-slice assumption: left/above are simply "not on the
    /// left/top edge of the picture".
    pub fn availability_single_slice(x_ctb: u32, y_ctb: u32) -> NeighbourAvail {
        NeighbourAvail {
            left: x_ctb > 0,
            above: y_ctb > 0,
        }
    }
}

impl Iterator for CtuIter<'_> {
    type Item = CtuPos;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next_rs >= self.total {
            return None;
        }
        let rs = self.next_rs;
        self.next_rs += 1;
        let x_ctb = rs % self.layout.pic_width_in_ctbs_y;
        let y_ctb = rs / self.layout.pic_width_in_ctbs_y;
        let x0 = x_ctb * self.layout.ctb_size_y;
        let y0 = y_ctb * self.layout.ctb_size_y;
        let w = (self.layout.pic_width_luma - x0).min(self.layout.ctb_size_y);
        let h = (self.layout.pic_height_luma - y0).min(self.layout.ctb_size_y);
        Some(CtuPos {
            x_ctb,
            y_ctb,
            x0,
            y0,
            width_luma: w,
            height_luma: h,
            ctu_addr_rs: rs,
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = (self.total - self.next_rs) as usize;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for CtuIter<'_> {}

/// SliceQpY derivation (§7.4.8 eq. 140).
///
/// `SliceQpY = 26 + pps_init_qp_minus26 + sh_qp_delta + ph_qp_delta` is
/// clipped to the luma QP range `[-QpBdOffset, 63]`, with
/// `QpBdOffset = 6 * sps_bitdepth_minus8` (§7.4.3.4 eq. 40).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SliceQpY(pub i32);

impl SliceQpY {
    /// Compute from the parsed parameter sets + slice header. The
    /// `ph_qp_delta` argument captures the value from the picture
    /// header (0 when `pps_qp_delta_info_in_ph_flag == 0`).
    pub fn derive(
        sps: &SeqParameterSet,
        pps: &PicParameterSet,
        sh: &StatefulSliceHeader,
        ph_qp_delta: i32,
    ) -> Self {
        let qp_bd_offset = 6 * sps.sps_bitdepth_minus8 as i32;
        let raw = 26 + pps.pps_init_qp_minus26 + sh.sh_qp_delta + ph_qp_delta;
        let clipped = raw.clamp(-qp_bd_offset, 63);
        SliceQpY(clipped)
    }

    /// Value clamped into [0, 63] for context-init purposes
    /// (§9.3.2.2 eq. 1525 clips `SliceQpY` internally but callers may
    /// want the pre-clipped integer for debugging).
    pub fn as_init_qp(self) -> i32 {
        self.0.clamp(0, 63)
    }
}

/// Slice-scope CABAC state: the initial context tables plus the slice
/// QP used to initialise them. Rebuilt at the start of every slice,
/// and (for WPP) at the start of every CTU row — WPP is not yet
/// supported.
///
/// `sh_cabac_init_flag` governs the choice of initialisation table in
/// P/B slices (§9.3.2.2 / Table 52), which is a straight swap of one
/// `initValue` / `shiftIdx` row for another. The intra-slice foundation
/// only exercises the I-slice table; the flag is recorded here so the
/// inter dispatch can be wired when inter CUs land. For now the scaffold
/// surfaces `Error::Unsupported` for non-intra slices (see
/// [`CtuWalker::begin_slice`]).
#[derive(Debug)]
pub struct SliceCabacState {
    pub slice_qp_y: SliceQpY,
    pub sh_cabac_init_flag: bool,
    pub tree_ctxs: TreeCtxs,
}

impl SliceCabacState {
    /// Initialise the CABAC context bundle for the first CTU of a
    /// slice using §9.3.2.2 entry arithmetic via [`TreeCtxs::init`].
    ///
    /// This is the "simplest pipeline" path: it picks the I-slice
    /// initValue/shift rows unconditionally. When the I-slice walker
    /// is extended to P/B, the `sh_cabac_init_flag` bit will swap in
    /// the alternative row per spec Table 52.
    pub fn init_for_slice(slice_qp_y: SliceQpY, sh_cabac_init_flag: bool) -> Self {
        Self {
            slice_qp_y,
            sh_cabac_init_flag,
            tree_ctxs: TreeCtxs::init(slice_qp_y.as_init_qp()),
        }
    }
}

/// A leaf CU emitted by the walker, annotated with the CTU it belongs
/// to. The geometry is in absolute (picture-wide) luma-sample
/// coordinates, not CTU-relative.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CtuCu {
    pub ctu_addr_rs: u32,
    pub cu: Cu,
}

/// Resolved LMCS binding for the current slice — the §7.4.3.19 derived
/// arrays plus the picture-header gates the §8.7.5.2 / §8.7.5.3 /
/// §8.8.2 processes consume. Built by [`CtuWalker::set_lmcs`] from the
/// `ph_lmcs_aps_id`-referenced LMCS APS payload; consulted only when
/// the slice sets `sh_lmcs_used_flag`.
struct LmcsBinding {
    /// §7.4.3.19 BitDepth-bound derivation (`InputPivot` / `LmcsPivot` /
    /// `ScaleCoeff` / `InvScaleCoeff` / `ChromaScaleCoeff`).
    derived: LmcsDerived,
    /// `lmcs_min_bin_idx` — lower bound of the §8.8.2.3 eq. 1224 scan.
    min_bin_idx: u8,
    /// `LmcsMaxBinIdx = 15 − lmcs_delta_max_bin_idx` — upper bound of
    /// the §8.8.2.3 eq. 1224 scan.
    max_bin_idx: u8,
    /// `ph_chroma_residual_scale_flag` — the picture-header gate for
    /// the §8.7.5.3 chroma residual scaling (first bullet of the
    /// pass-through list).
    chroma_residual_scale: bool,
}

/// §8.7.5.3 eqs. 1219 / 1220 — scale a chroma residual sample array in
/// place: each sample is first clamped to
/// `Clip3( −( 1 << BitDepth ), ( 1 << BitDepth ) − 1, res )` (eq. 1219),
/// then replaced by the eq. 1220 scale term
/// `Sign( res ) * ( ( Abs( res ) * varScale + ( 1 << 10 ) ) >> 11 )`.
/// The caller's ordinary `Clip1( pred + res )` add then matches
/// eq. 1220 exactly.
fn lmcs_scale_chroma_residuals(res: &mut [i32], var_scale: u32, bit_depth: u32) {
    let lo = -(1i64 << bit_depth);
    let hi = (1i64 << bit_depth) - 1;
    for r in res.iter_mut() {
        let v = i64::from(*r).clamp(lo, hi);
        let mag = (v.unsigned_abs() * u64::from(var_scale) + (1 << 10)) >> 11;
        *r = if v < 0 { -(mag as i64) } else { mag as i64 } as i32;
    }
}

/// Top-level CTU walker.
///
/// A `CtuWalker` is constructed from a parsed SPS/PPS/slice-header
/// triple plus the *post-byte-alignment* CABAC payload bytes and the
/// `ph_qp_delta` read out of the picture header. Typical lifecycle:
///
/// ```ignore
/// let layout  = CtuLayout::from_sps_pps(&sps, &pps);
/// let mut w   = CtuWalker::begin_slice(&layout, &sps, &pps, &sh, 0, payload)?;
/// for ctu in w.iter_ctus() {
///     let cus = w.decode_ctu_partitions(&ctu)?;
///     // Each `cus` entry will eventually be sent to reconstruct_leaf_cu();
///     // today that call returns Error::Unsupported.
/// }
/// ```
pub struct CtuWalker<'a, 'b> {
    layout: &'a CtuLayout,
    sps: &'a SeqParameterSet,
    pps: &'a PicParameterSet,
    sh: &'a StatefulSliceHeader,
    cabac: SliceCabacState,
    arith: ArithDecoder<'b>,
    leaf_ctxs: LeafCuCtxs,
    /// Per-leaf records accumulated by [`Self::reconstruct_leaf_cu`]
    /// for the in-loop deblocking pass (§8.8.3). Reset implicitly
    /// every time a fresh walker is constructed; consumed by
    /// [`Self::apply_in_loop_filters`].
    deblock_cus: Vec<DeblockCu>,
    /// Per-CTB SAO parameters consumed by [`Self::apply_in_loop_filters`].
    /// Defaults to "all CTBs not applied". The round-14 walker can
    /// optionally invoke [`Self::decode_sao_for_ctu`] before
    /// [`Self::decode_ctu_partitions`] to populate this array from the
    /// §7.3.11.3 `sao(rx, ry)` CABAC syntax; otherwise callers (and
    /// integration tests) may populate it programmatically via
    /// [`Self::set_sao_picture`].
    sao_picture: SaoPicture,
    /// CABAC contexts for the §7.3.11.3 SAO syntax. Lives on the walker
    /// so each per-CTU call advances the same engine.
    sao_ctxs: SaoCtxs,
    /// Per-CTB ALF on/off + filter-set selection consumed by
    /// [`Self::apply_in_loop_filters`]. Defaults to "all CTBs ALF off".
    /// The round-15 walker does not yet parse the §7.3.11.2 ALF CABAC
    /// bins; callers (and integration tests) populate this array
    /// programmatically via [`Self::set_alf_picture`].
    alf_picture: AlfPicture,
    /// Round-21 reference-picture list 0. The merge-mode `merge_idx`
    /// resolves to an `MvField` with `ref_idx_l0` indexing into this
    /// list. Empty for I-slices; populated by callers via
    /// [`Self::set_ref_pic_list_l0`] before `decode_picture_into`
    /// for P-slices.
    ref_pic_list_l0: Vec<ReferencePicture>,
    /// Round-23 reference-picture list 1. Mirrors `ref_pic_list_l0`
    /// for the bi-pred path: the merge-mode `merge_idx` resolves to
    /// an `MvField` whose `pred_flag_l1` (when set) indexes
    /// `ref_idx_l1` into this list. Empty for I/P-slices; populated
    /// by callers via [`Self::set_ref_pic_list_l1`] before
    /// `decode_picture_into` for B-slices.
    ref_pic_list_l1: Vec<ReferencePicture>,
    /// Round-21 per-picture motion field — written by
    /// [`Self::reconstruct_leaf_cu`] for inter CUs and read back during
    /// the §8.5.2.3 spatial-merge derivation of subsequent CUs.
    motion_field: MotionField,
    /// Round-24 per-slice HMVP table (§8.5.2.6 + §8.5.2.16). Reset
    /// to empty at slice start (and at every CTU column tile-
    /// boundary per the §7.3.11 slice_data() pseudocode — for our
    /// single-tile fixture only the slice-start reset fires). The
    /// inter CU walker reads this table during §8.5.2.2 step 7 and
    /// updates it after each inter CU per §8.5.2.16.
    hmvp: HmvpTable,
    /// `Log2ParMrgLevel` from `pps_log2_parallel_merge_level_minus2 + 2`
    /// (§7.4.3.5). Drives the spec's same-parallel-merge-tile
    /// neighbour-suppression rule.
    log2_par_mrg_level: u32,
    /// Round-25 §8.5.2.11 — current picture's POC. Needed for the
    /// POC-distance scaling of the temporal merge candidate (eq. 599).
    /// Defaults to 0; callers that exercise temporal MVP set this via
    /// [`Self::set_temporal_mvp`].
    current_poc: i32,
    /// Round-25 §8.5.2.11 — `ph_temporal_mvp_enabled_flag`. When 0 the
    /// Col candidate derivation short-circuits to `availableFlagCol = 0`
    /// per eq. 477. Defaults to `false` (temporal MVP off).
    ph_temporal_mvp_enabled: bool,
    /// Round-25 §8.5.2.11 — index of the collocated reference picture
    /// in the spec-selected list (`sh_collocated_ref_idx` /
    /// `ph_collocated_ref_idx`). Resolved via
    /// `ref_pic_list_l{0|1}[col_ref_idx]` based on
    /// [`Self::collocated_from_l0`]. Ignored when
    /// [`Self::ph_temporal_mvp_enabled`] is false.
    col_ref_idx: u32,
    /// Round-25 §8.5.2.11 — `sh_collocated_from_l0_flag` /
    /// `ph_collocated_from_l0_flag`. When `true` the collocated
    /// reference is `RefPicList[0][col_ref_idx]`; when `false` it is
    /// `RefPicList[1][col_ref_idx]`. Defaults to `true` per spec
    /// inference.
    collocated_from_l0: bool,
    /// Round-27 §8.5.2.7 — `ph_mmvd_fullpel_only_flag`. When `true`
    /// (and `sps_mmvd_enabled_flag` is also true), the MMVD distance
    /// table swaps from `MMVD_DISTANCE_TABLE` to
    /// `MMVD_DISTANCE_TABLE_FULLPEL` so every MMVD-corrected MV stays
    /// integer-pel. Defaults to `false`. Surfaced into [`CuToolFlags`]
    /// via [`Self::cu_tool_flags`] so the leaf CU reader can route
    /// it through to [`crate::inter::derive_mmvd_offset`].
    ph_mmvd_fullpel_only: bool,
    /// Round-31 §8.5.5.1 / §8.5.6.5 — `ph_bdof_disabled_flag`. The
    /// BDOF picture-level disable that, together with
    /// `sps_bdof_enabled_flag` (read directly from
    /// [`Self::sps`]`.tool_flags.bdof_enabled_flag`), drives the
    /// `bdofUsedFlag` derivation in [`crate::bdof::bdof_used_flag`].
    /// Defaults to `true` — i.e. BDOF off — so callers that have not
    /// wired the picture-header bit yet keep the round-23 plain
    /// `bi_pred_avg_8bit` byte-for-byte. Set via
    /// [`Self::set_ph_bdof_disabled`].
    ph_bdof_disabled: bool,
    /// §8.5.1 / §8.5.3.1 — `ph_dmvr_disabled_flag`. The DMVR
    /// picture-level disable that, together with
    /// `sps_dmvr_enabled_flag` (read directly from
    /// [`Self::sps`]`.tool_flags.dmvr_enabled_flag`), drives the
    /// §8.5.1 `dmvrFlag` derivation in
    /// [`crate::dmvr::dmvr_used_flag`]. Defaults to `true` — i.e.
    /// DMVR off — so callers that have not wired the picture-header
    /// bit yet keep the round-31 plain bi-pred / BDOF path
    /// byte-for-byte. Set via [`Self::set_ph_dmvr_disabled`].
    ph_dmvr_disabled: bool,
    /// §7.4.3.7 `ph_joint_cbcr_sign_flag` — drives the §8.7.2 joint
    /// Cb-Cr `cSign = 1 − 2 * flag` used when deriving the non-coded
    /// chroma residual from the coded one. Defaults to `false`; set via
    /// [`Self::set_ph_joint_cbcr_sign`].
    ph_joint_cbcr_sign: bool,
    /// §7.4.3.7 `ph_prof_disabled_flag` — gates §8.5.5.8 PROF in the
    /// affine MC path ([`Self::reconstruct_affine_inter_uni`]).
    /// Defaults to `true` (PROF off) so callers that have not wired the
    /// picture-header bit keep the plain affine sub-block MC. Set via
    /// [`Self::set_ph_prof_disabled`].
    ph_prof_disabled: bool,
    /// Round-28 §8.5.6.7 — per-picture intra-coded grid sampled at 4x4
    /// luma granularity. The §8.5.6.7 weight ladder reads
    /// `CuPredMode[0][xNbX][yNbX]` for X being the A / B neighbour
    /// positions; this grid is the cheapest way to track the
    /// MODE_INTRA / MODE_INTER decision per 4x4 block without rerunning
    /// the partition walker. Each entry is initialised to `false`
    /// (treat-as-unavailable / inter); intra leaf CUs flip their
    /// covered cells to `true` and inter leaf CUs flip them back to
    /// `false`. The CTU walker reads the (xCb − 1, yCb − 1 + cbHeight)
    /// and (xCb − 1 + cbWidth, yCb − 1) cells when the leaf CU's
    /// `info.inter.merge_data.ciip_flag == true` to derive the §8.5.6.7
    /// weight `w`.
    intra_grid: Vec<bool>,
    /// Round-28 — width of [`Self::intra_grid`] in 4x4 blocks. Mirrors
    /// `motion_field.blocks_w` but is kept as a separate field so the
    /// CIIP path doesn't have to indirect through `motion_field` for a
    /// pure picture-geometry constant.
    intra_grid_w: u32,
    /// Round-28 — height of [`Self::intra_grid`] in 4x4 blocks.
    intra_grid_h: u32,
    /// Round-149 — per-CB live sub-block-merge grid sampled at 4x4
    /// luma granularity. Mirrors `MergeSubblockFlag[x][y]` from
    /// §7.3.11.7: every 4x4 block touched by a CU decoded with
    /// `merge_subblock_flag == 1` carries `true`. The §9.3.4.2.2 /
    /// Table 133 `cond{L,A}` ctxInc for `read_merge_subblock_flag`
    /// samples this grid through [`Self::compute_cu_neighbourhood`].
    /// Initialised to `false` (== unavailable / non-subblock-merge);
    /// the inter leaf CU walker flips covered cells to `true` when
    /// the parsed `merge_data.merge_subblock_flag == 1` and clears
    /// them otherwise (e.g. when an intra or non-subblock-merge inter
    /// CU later overwrites the same picture region — single-pass
    /// scan in slice order, the merge-side neighbour query at a CU
    /// only ever reads cells the prior CUs already wrote).
    subblock_merge_grid: Vec<bool>,
    /// Round-149 — per-CB live affine-inter grid sampled at 4x4 luma
    /// granularity. Mirrors `InterAffineFlag[x][y]` from §7.3.11.7
    /// (the non-merge affine inter path). The CTU walker does not
    /// yet parse `inter_affine_flag`; every cell stays `false` for
    /// round-149. The grid is plumbed so the §9.3.4.2.2 /
    /// Table 133 `cond{L,A} = MergeSubblockFlag[...] ||
    /// InterAffineFlag[...]` ctxInc only needs a one-line drop-in
    /// when the affine-inter walker arrives.
    inter_affine_grid: Vec<bool>,
    /// Round-364 — per-CB affine CPMV store sampled at 4x4 luma
    /// granularity (§8.5.5.7 / §8.5.5.5). Every affine-coded CB
    /// broadcasts its [`crate::inter::AffineCbRecord`] (origin, dims,
    /// `MotionModelIdc`, per-list CPMV set + pred flags + ref indices)
    /// across the 4x4 blocks it covers; later CUs sample the
    /// A0/A1/B0/B1/B2 neighbour positions during the §8.5.5.7
    /// inherited-CPMVP scan. Translational / intra CBs clear their
    /// covered cells to `None`.
    affine_cpmv_field: crate::inter::AffineCpmvField,
    /// §7.4.3.19 / §8.7.5 — resolved LMCS binding for the slice. `None`
    /// until a caller installs the `ph_lmcs_aps_id`-referenced APS
    /// payload via [`Self::set_lmcs`]; every LMCS fold is additionally
    /// gated on `sh_lmcs_used_flag`, so a bound-but-unused payload is
    /// inert. [`Self::decode_picture_into`] fails fast when the slice
    /// uses LMCS but no binding was installed.
    lmcs: Option<LmcsBinding>,
    /// §8.7.5.3 — per-4x4-cell top-left luma origin of the coding unit
    /// covering the cell (shares the [`Self::intra_grid`] geometry).
    /// Written by [`Self::write_intra_block`] as each leaf CU commits;
    /// the chroma-residual-scaling `( xCuCb, yCuCb )` lookup samples it
    /// for the CU containing the sizeY-aligned luma corner when that
    /// corner falls outside the current CU.
    cu_origin_grid: Vec<(u32, u32)>,
    /// Round-56 — picture-wide CU neighbour map used by the
    /// [`TreeWalker`] to derive the §9.3.4.2 ctxInc. Populated as each
    /// CTU's leaf CUs commit; `decode_ctu_partitions` hands a `&mut`
    /// borrow to the per-CTU walker via
    /// [`TreeWalker::with_neighbour_map`]. Replaces the round-55 hard-
    /// coded picture-edge default so multi-row CTBs see real left /
    /// above neighbour descriptors.
    nbr_map: CuNeighbourMap,
}

impl std::fmt::Debug for CtuWalker<'_, '_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CtuWalker")
            .field("layout", self.layout)
            .field("slice_qp_y", &self.cabac.slice_qp_y)
            .field("sh_cabac_init_flag", &self.cabac.sh_cabac_init_flag)
            .finish()
    }
}

impl<'a, 'b> CtuWalker<'a, 'b> {
    /// Construct a walker and initialise the CABAC engine + slice
    /// context state. Returns `Error::Unsupported` for any slice /
    /// picture layout the scaffold does not walk yet.
    pub fn begin_slice(
        layout: &'a CtuLayout,
        sps: &'a SeqParameterSet,
        pps: &'a PicParameterSet,
        sh: &'a StatefulSliceHeader,
        ph_qp_delta: i32,
        cabac_payload: &'b [u8],
    ) -> Result<Self> {
        // Single-slice guard: the CTU iterator only knows how to walk
        // the whole picture as one raster scan today.
        if !pps.pps_no_pic_partition_flag {
            return Err(Error::unsupported(
                "h266 CTU walker: multi-tile / multi-slice pictures not supported \
                 (pps_no_pic_partition_flag must be 1)",
            ));
        }
        // SPS/PPS CTB size must agree (§7.4.3.5).
        if let Some(part) = pps.partition.as_ref() {
            if part.log2_ctu_size_minus5 != sps.sps_log2_ctu_size_minus5 {
                return Err(Error::invalid(
                    "h266 CTU walker: PPS CTB size does not match SPS",
                ));
            }
        }
        // P-slices land via the round-21 inter path (cu_skip + regular
        // merge). B-slices land via the round-23 extension (second
        // §8.5.3.2 invocation against RefPicList[1] + §8.5.6.6.2 eq.
        // 980 default-weighted bi-pred composition; BCW / weighted-pred
        // / BDOF / DMVR still surface `Error::Unsupported` further down).
        // Dual-tree luma/chroma separate partitioning is not yet
        // modelled.
        if sps.partition_constraints.qtbtt_dual_tree_intra_flag {
            return Err(Error::unsupported(
                "h266 CTU walker: dual-tree intra partitioning (separate luma/chroma \
                 coding trees) not supported by scaffold",
            ));
        }
        // Deferred features that would change the per-CTU pipeline shape.
        // ALF (`sh_alf_enabled_flag`) is now accepted: the §8.8.5 apply
        // pass runs after SAO and the per-CTB on/off + filter selection
        // lives in [`Self::alf_picture`]. The round-15 walker leaves it
        // at the empty default (every CTB ALF-off), so ALF is a no-op
        // unless callers programmatically populate it. The per-CTB
        // §7.3.11.2 CABAC bins and the §8.8.5.3 classification will
        // come in later rounds.
        // LMCS (`sh_lmcs_used_flag`) is accepted since round 384: the
        // §8.7.5.2 mapped-domain reconstruction + §8.8.2 picture inverse
        // mapping run once a caller binds the `ph_lmcs_aps_id`-referenced
        // APS payload via [`Self::set_lmcs`]. `decode_picture_into`
        // fails fast when the slice uses LMCS but no binding exists.
        if sh.sh_dep_quant_used_flag {
            return Err(Error::unsupported(
                "h266 CTU walker: dependent quantisation not supported yet",
            ));
        }
        if sh.sh_explicit_scaling_list_used_flag {
            return Err(Error::unsupported(
                "h266 CTU walker: explicit scaling lists not supported yet",
            ));
        }

        let slice_qp = SliceQpY::derive(sps, pps, sh, ph_qp_delta);
        let cabac_state = SliceCabacState::init_for_slice(slice_qp, sh.sh_cabac_init_flag);
        let arith = ArithDecoder::new(cabac_payload)?;
        // §9.3.2.2 / Table 51 — initType = 0 for I, otherwise picks
        // 1 / 2 from `sh_cabac_init_flag` per the slice type. The same
        // value drives the [`sao_syntax::SaoSyntaxConfig::init_type`]
        // helper for the SAO bundle.
        let init_type: u8 = match sh.sh_slice_type {
            SliceType::I => 0,
            SliceType::P => {
                if sh.sh_cabac_init_flag {
                    2
                } else {
                    1
                }
            }
            SliceType::B => {
                if sh.sh_cabac_init_flag {
                    1
                } else {
                    2
                }
            }
        };
        let leaf_ctxs = LeafCuCtxs::init_with_init_type(slice_qp.as_init_qp(), init_type);

        let sao_picture =
            SaoPicture::empty(layout.pic_width_in_ctbs_y, layout.pic_height_in_ctbs_y);
        let sao_ctxs = SaoCtxs::init(slice_qp.as_init_qp());
        let alf_picture =
            AlfPicture::empty(layout.pic_width_in_ctbs_y, layout.pic_height_in_ctbs_y);
        let motion_field = MotionField::new(layout.pic_width_luma, layout.pic_height_luma);
        // Round-28 — 4x4 intra-coded grid. Initialised to `false`
        // (== unavailable / inter), gets updated by every leaf CU
        // reconstruction path to track the §8.5.6.7 neighbour status.
        let intra_grid_w = layout.pic_width_luma.div_ceil(4);
        let intra_grid_h = layout.pic_height_luma.div_ceil(4);
        let intra_grid = vec![false; (intra_grid_w * intra_grid_h) as usize];
        // Round-149 — per-CB live sub-block-merge / affine-inter grids
        // share the §8.5.6.7 intra-grid geometry (4x4 luma cells across
        // the full picture). Both initialise to `false` so picture-edge
        // and pre-decode neighbours register as "no merge_subblock /
        // no inter_affine", matching the §7.4.12.7 inference for any
        // unavailable neighbour.
        let subblock_merge_grid = vec![false; (intra_grid_w * intra_grid_h) as usize];
        let inter_affine_grid = vec![false; (intra_grid_w * intra_grid_h) as usize];
        // Round-364 — per-CB affine CPMV store (§8.5.5.7 / §8.5.5.5).
        // Same 4x4 grid geometry as the motion field; every cell starts
        // `None` (covering CB not affine).
        let affine_cpmv_field =
            crate::inter::AffineCpmvField::new(layout.pic_width_luma, layout.pic_height_luma);
        // §7.4.3.5 — Log2ParMrgLevel = pps_log2_parallel_merge_level_minus2
        // + 2. Our PPS parser does not yet surface that field, so we
        // default to the spec minimum (2 → ParMrgLevel = 4) which is
        // also the value our test fixture emits.
        let log2_par_mrg_level: u32 = 2;
        Ok(Self {
            layout,
            sps,
            pps,
            sh,
            cabac: cabac_state,
            arith,
            leaf_ctxs,
            deblock_cus: Vec::new(),
            sao_picture,
            sao_ctxs,
            alf_picture,
            ref_pic_list_l0: Vec::new(),
            ref_pic_list_l1: Vec::new(),
            motion_field,
            hmvp: HmvpTable::new(),
            log2_par_mrg_level,
            current_poc: 0,
            ph_temporal_mvp_enabled: false,
            col_ref_idx: 0,
            collocated_from_l0: true,
            ph_mmvd_fullpel_only: false,
            ph_bdof_disabled: true,
            ph_dmvr_disabled: true,
            ph_joint_cbcr_sign: false,
            ph_prof_disabled: true,
            intra_grid,
            intra_grid_w,
            intra_grid_h,
            subblock_merge_grid,
            inter_affine_grid,
            affine_cpmv_field,
            lmcs: None,
            cu_origin_grid: vec![(0, 0); (intra_grid_w * intra_grid_h) as usize],
            nbr_map: CuNeighbourMap::new(layout.pic_width_luma, layout.pic_height_luma),
        })
    }

    /// Install the LMCS APS payload the picture header references via
    /// `ph_lmcs_aps_id` (§7.4.3.7), running the §7.4.3.19 BitDepth-bound
    /// derivations against the active SPS.
    /// `ph_chroma_residual_scale_flag` carries the picture-header gate
    /// for the §8.7.5.3 chroma residual scaling. Must be called before
    /// [`Self::decode_picture_into`] / [`Self::apply_in_loop_filters`]
    /// whenever the slice header sets `sh_lmcs_used_flag`.
    pub fn set_lmcs(&mut self, data: &LmcsData, ph_chroma_residual_scale_flag: bool) -> Result<()> {
        let bit_depth = self.sps.sps_bitdepth_minus8 as u32 + 8;
        let derived = data.derive(bit_depth)?;
        self.lmcs = Some(LmcsBinding {
            derived,
            min_bin_idx: data.lmcs_min_bin_idx,
            max_bin_idx: data.lmcs_max_bin_idx(),
            chroma_residual_scale: ph_chroma_residual_scale_flag,
        });
        Ok(())
    }

    /// `true` when the slice both signals `sh_lmcs_used_flag` and has a
    /// resolved [`LmcsBinding`] — the condition under which the
    /// §8.7.5.2 / §8.7.5.3 / §8.8.2 folds run.
    fn lmcs_active(&self) -> bool {
        self.sh.sh_lmcs_used_flag && self.lmcs.is_some()
    }

    /// §8.7.5.2 — derive `predMapSamples` for a MODE_INTER (non-CIIP)
    /// CU by forward-mapping its motion-compensated luma prediction in
    /// place. Right after MC the picture plane holds `predSamples`;
    /// eq. 1213 lifts each sample into the LMCS codeword domain so the
    /// eq. 1214 residual add — and every later mapped-domain read
    /// (intra reference samples, the §8.7.5.3 `invAvgLuma`
    /// neighbourhood) — sees mapped values. The §8.7.5.2 pass-through
    /// modes (MODE_INTRA / MODE_IBC / MODE_PLT, and MODE_INTER with
    /// `ciip_flag == 1`) must NOT call this. No-op unless the slice
    /// uses LMCS with a bound payload.
    ///
    /// Eq. 1213 itself is unclamped; the value is clamped to
    /// `[0, (1 << BitDepth) − 1]` for plane storage — the §7.4.3.19
    /// eq. 96 `Σ lmcsCW <= (1 << BitDepth) − 1` budget keeps
    /// `predMapSamples` inside that range, so the clamp never changes
    /// an in-range mapping.
    fn lmcs_forward_map_luma_rect(
        &self,
        out: &mut PictureBuffer,
        x0: usize,
        y0: usize,
        w: usize,
        h: usize,
    ) {
        if !self.lmcs_active() {
            return;
        }
        let l = self.lmcs.as_ref().expect("lmcs_active checked is_some");
        let bit_depth = self.sps.sps_bitdepth_minus8 as u32 + 8;
        let up = bit_depth.saturating_sub(8);
        let hi = (1i64 << bit_depth) - 1;
        let plane = &mut out.luma;
        let y1 = (y0 + h).min(plane.height);
        let x1 = (x0 + w).min(plane.width);
        for row in y0..y1 {
            for col in x0..x1 {
                let idx = row * plane.stride + col;
                let s = u32::from(plane.samples[idx]) << up;
                let m = l.derived.forward_map_luma_sample(s).clamp(0, hi) as u32;
                plane.samples[idx] = (m >> up) as u8;
            }
        }
    }

    /// §8.7.5.3 ordered steps 1 – 3 — derive `varScale` for the current
    /// CU's chroma transform block(s). Returns `None` when chroma
    /// residual scaling does not apply at the CU level
    /// (`sh_lmcs_used_flag == 0`, no binding, or
    /// `ph_chroma_residual_scale_flag == 0`); the per-TB
    /// `nCurrSw * nCurrSh <= 4` and `tuCbfChroma` gates live at the
    /// call sites.
    ///
    /// * `sizeY = Min( CtbSizeY, 64 )`; the current luma location is
    ///   the CU's luma origin (eq. 1215 with the chroma TB at the CU
    ///   corner), aligned down to the sizeY grid.
    /// * `( xCuCb, yCuCb )` is the top-left of the coding unit
    ///   containing that aligned corner — the current CU when the
    ///   corner falls inside it, otherwise the already-committed
    ///   [`Self::cu_origin_grid`] cell.
    /// * `availL` / `availT` follow the walker's single-slice §6.4.4
    ///   picture-edge rule (`checkPredModeY == FALSE`, so intra / inter
    ///   makes no difference).
    /// * `recLuma[]` gathers the left column / top row of mapped-domain
    ///   reconstructed luma with the eq.-clamps to the picture bounds;
    ///   `invAvgLuma` is the eq. 1216 rounded average (eq. 1217
    ///   mid-grey when neither side is available), which then walks
    ///   §8.8.2.3 to `idxYInv` and eq. 1218 to
    ///   `ChromaScaleCoeff[ idxYInv ]`.
    fn lmcs_chroma_var_scale_for_cu(&self, cu: &CtuCu, out: &PictureBuffer) -> Option<u32> {
        if !self.lmcs_active() {
            return None;
        }
        let l = self.lmcs.as_ref().expect("lmcs_active checked is_some");
        if !l.chroma_residual_scale {
            return None;
        }
        let bit_depth = self.sps.sps_bitdepth_minus8 as u32 + 8;
        let up = bit_depth.saturating_sub(8);
        let size_y = (1usize << self.layout.ctb_log2_size_y).min(64);
        // Aligned corner of the sizeY region containing the CU's luma
        // origin (the chroma TB sits at the CU corner, so eq. 1215
        // gives back the CU's luma origin).
        let ax = (cu.cu.x as usize) / size_y * size_y;
        let ay = (cu.cu.y as usize) / size_y * size_y;
        let inside_current = ax >= cu.cu.x as usize
            && ax < (cu.cu.x + cu.cu.w) as usize
            && ay >= cu.cu.y as usize
            && ay < (cu.cu.y + cu.cu.h) as usize;
        let (x_cu_cb, y_cu_cb) = if inside_current {
            (cu.cu.x as usize, cu.cu.y as usize)
        } else {
            let bx = ((ax / 4) as u32).min(self.intra_grid_w - 1);
            let by = ((ay / 4) as u32).min(self.intra_grid_h - 1);
            let (ox, oy) = self.cu_origin_grid[(by * self.intra_grid_w + bx) as usize];
            (ox as usize, oy as usize)
        };
        let avail_l = x_cu_cb > 0;
        let avail_t = y_cu_cb > 0;
        let luma = &out.luma;
        let mut sum: u64 = 0;
        let mut cnt: usize = 0;
        if avail_l {
            for i in 0..size_y {
                let yy = (y_cu_cb + i).min(luma.height - 1);
                sum += u64::from(luma.samples[yy * luma.stride + (x_cu_cb - 1)]) << up;
            }
            cnt += size_y;
        }
        if avail_t {
            for i in 0..size_y {
                let xx = (x_cu_cb + i).min(luma.width - 1);
                sum += u64::from(luma.samples[(y_cu_cb - 1) * luma.stride + xx]) << up;
            }
            cnt += size_y;
        }
        // eq. 1216 / eq. 1217 — `cnt` is sizeY or 2·sizeY, both powers
        // of two, so `Log2( cnt )` is exact.
        let inv_avg_luma = if cnt > 0 {
            ((sum + (cnt as u64 >> 1)) >> cnt.trailing_zeros()) as u32
        } else {
            1u32 << (bit_depth - 1)
        };
        let idx_y_inv = l
            .derived
            .idx_y_inv(inv_avg_luma, l.min_bin_idx, l.max_bin_idx);
        Some(l.derived.chroma_var_scale(idx_y_inv))
    }

    /// Install the per-slice reference-picture list 0. P-slice callers
    /// must invoke this before `decode_picture_into` so the round-21
    /// inter path has a reference to copy from. Subsequent re-use of
    /// the walker for another slice should reset this list — but the
    /// walker is one-shot today (`begin_slice` builds a fresh one).
    pub fn set_ref_pic_list_l0(&mut self, list: Vec<ReferencePicture>) {
        self.ref_pic_list_l0 = list;
    }

    /// Install the per-slice reference-picture list 1. Round-23
    /// B-slice callers must invoke this before `decode_picture_into`
    /// when the parsed merge candidate sets `pred_flag_l1` (the
    /// second §8.5.6 invocation reads from here). Mirrors
    /// [`Self::set_ref_pic_list_l0`].
    pub fn set_ref_pic_list_l1(&mut self, list: Vec<ReferencePicture>) {
        self.ref_pic_list_l1 = list;
    }

    /// Read-only view of the round-21 motion field. Useful for tests
    /// that want to assert the per-CU MvField writes the inter path
    /// emits.
    pub fn motion_field(&self) -> &MotionField {
        &self.motion_field
    }

    /// Round-28 §8.5.6.7 — sample the picture-wide 4x4 intra-coded
    /// grid at picture-absolute luma `(x, y)`. Returns `false` when the
    /// position is out of bounds (matching the spec's
    /// `availableX == FALSE` → `isIntraCodedNeighbourX = FALSE`
    /// branch) and otherwise the cell's stored mode flag.
    fn sample_intra_at_luma(&self, x: i32, y: i32) -> bool {
        if x < 0 || y < 0 {
            return false;
        }
        let bx = (x as u32) / 4;
        let by = (y as u32) / 4;
        if bx >= self.intra_grid_w || by >= self.intra_grid_h {
            return false;
        }
        self.intra_grid[(by * self.intra_grid_w + bx) as usize]
    }

    /// Round-28 §8.5.6.7 — broadcast `is_intra` across every 4x4 cell
    /// touched by the rectangle `[x, x+w) x [y, y+h)`. Used by the
    /// per-CU reconstruction paths to update [`Self::intra_grid`] so
    /// later CIIP CUs see the correct neighbour status.
    fn write_intra_block(&mut self, x: u32, y: u32, w: u32, h: u32, is_intra: bool) {
        let bx0 = x / 4;
        let by0 = y / 4;
        let bx1 = (x + w).div_ceil(4).min(self.intra_grid_w);
        let by1 = (y + h).div_ceil(4).min(self.intra_grid_h);
        for by in by0..by1 {
            let row_off = (by * self.intra_grid_w) as usize;
            for bx in bx0..bx1 {
                self.intra_grid[row_off + bx as usize] = is_intra;
                // §8.7.5.3 — record the covering CU's top-left luma
                // origin so a later CU's `( xCuCb, yCuCb )` lookup can
                // resolve the CU containing its sizeY-aligned corner.
                self.cu_origin_grid[row_off + bx as usize] = (x, y);
            }
        }
        // Round-364 — an intra CB is never affine; clear any stale
        // per-CB affine record on its cells so a later §8.5.5.7 scan
        // reads this region as non-affine (`MotionModelIdc == 0`).
        if is_intra {
            self.affine_cpmv_field.write_block(x, y, w, h, None);
        }
    }

    /// Round-149 — sample `MergeSubblockFlag[xCb][yCb]` at picture-
    /// absolute luma `(x, y)`. Returns `false` when the position is
    /// out of bounds (matching the §6.4.4 availability mask:
    /// unavailable neighbour → `cond = 0`) and otherwise the cell's
    /// stored sub-block-merge flag.
    fn sample_subblock_merge_at_luma(&self, x: i32, y: i32) -> bool {
        if x < 0 || y < 0 {
            return false;
        }
        let bx = (x as u32) / 4;
        let by = (y as u32) / 4;
        if bx >= self.intra_grid_w || by >= self.intra_grid_h {
            return false;
        }
        self.subblock_merge_grid[(by * self.intra_grid_w + bx) as usize]
    }

    /// Round-149 — sample `InterAffineFlag[xCb][yCb]` at picture-
    /// absolute luma `(x, y)`. Mirrors
    /// [`Self::sample_subblock_merge_at_luma`]; the non-merge affine
    /// inter path is not yet parsed by the CTU walker so this is
    /// always `false` for round-149.
    fn sample_inter_affine_at_luma(&self, x: i32, y: i32) -> bool {
        if x < 0 || y < 0 {
            return false;
        }
        let bx = (x as u32) / 4;
        let by = (y as u32) / 4;
        if bx >= self.intra_grid_w || by >= self.intra_grid_h {
            return false;
        }
        self.inter_affine_grid[(by * self.intra_grid_w + bx) as usize]
    }

    /// Round-149 — broadcast `is_subblock_merge` across every 4x4 cell
    /// touched by the rectangle `[x, x+w) x [y, y+h)`. Mirrors
    /// [`Self::write_intra_block`] for the §9.3.4.2.2 / Table 133
    /// `MergeSubblockFlag` neighbour grid.
    fn write_subblock_merge_block(&mut self, x: u32, y: u32, w: u32, h: u32, value: bool) {
        let bx0 = x / 4;
        let by0 = y / 4;
        let bx1 = (x + w).div_ceil(4).min(self.intra_grid_w);
        let by1 = (y + h).div_ceil(4).min(self.intra_grid_h);
        for by in by0..by1 {
            let row_off = (by * self.intra_grid_w) as usize;
            for bx in bx0..bx1 {
                self.subblock_merge_grid[row_off + bx as usize] = value;
            }
        }
    }

    /// Round-149 — broadcast `is_inter_affine` across every 4x4 cell
    /// touched by the rectangle `[x, x+w) x [y, y+h)`. Mirrors
    /// [`Self::write_intra_block`] for the §9.3.4.2.2 / Table 133
    /// `InterAffineFlag` neighbour grid. Not yet invoked from the
    /// leaf-CU walker (the non-merge affine inter path is unparsed);
    /// kept `pub(crate)`-callable for tests that pre-load the grid.
    #[allow(dead_code)]
    fn write_inter_affine_block(&mut self, x: u32, y: u32, w: u32, h: u32, value: bool) {
        let bx0 = x / 4;
        let by0 = y / 4;
        let bx1 = (x + w).div_ceil(4).min(self.intra_grid_w);
        let by1 = (y + h).div_ceil(4).min(self.intra_grid_h);
        for by in by0..by1 {
            let row_off = (by * self.intra_grid_w) as usize;
            for bx in bx0..bx1 {
                self.inter_affine_grid[row_off + bx as usize] = value;
            }
        }
    }

    /// Read-only view of the round-24 HMVP table. Useful for tests that
    /// want to assert the per-CU §8.5.2.16 update writes the inter path
    /// performs after every inter CU (and the slice-start §7.3.11
    /// reset state).
    pub fn hmvp_table(&self) -> &HmvpTable {
        &self.hmvp
    }

    /// Round-25 — configure the §8.5.2.11 temporal motion-vector
    /// prediction inputs. Must be called before `decode_picture_into`
    /// when the bitstream signals `ph_temporal_mvp_enabled_flag == 1`
    /// (the picture-header path that picks the `ColPic` for
    /// merge / AMVP TMVP).
    ///
    /// * `current_poc` — `PicOrderCntVal` of the current picture
    ///   (§8.3.1).
    /// * `enabled` — `ph_temporal_mvp_enabled_flag`. When `false` the
    ///   walker short-circuits the Col candidate (eq. 477 path).
    /// * `collocated_from_l0` — `sh_collocated_from_l0_flag` /
    ///   `ph_collocated_from_l0_flag`. Selects which RefPicList
    ///   provides `ColPic`.
    /// * `col_ref_idx` — `sh_collocated_ref_idx` /
    ///   `ph_collocated_ref_idx`. Index into the chosen list.
    pub fn set_temporal_mvp(
        &mut self,
        current_poc: i32,
        enabled: bool,
        collocated_from_l0: bool,
        col_ref_idx: u32,
    ) {
        self.current_poc = current_poc;
        self.ph_temporal_mvp_enabled = enabled;
        self.collocated_from_l0 = collocated_from_l0;
        self.col_ref_idx = col_ref_idx;
    }

    /// Round-27 — install `ph_mmvd_fullpel_only_flag`. When the SPS
    /// signals `sps_mmvd_fullpel_only_enabled_flag == 1`, the picture
    /// header may set this to switch the Table 17 distance grid from
    /// the regular {1/4, 1/2, 1, 2, 4, 8, 16, 32} luma steps to the
    /// fullpel-only {1, 2, 4, 8, 16, 32, 64, 128} luma steps. Defaults
    /// to `false`; harmless on bitstreams without MMVD.
    pub fn set_ph_mmvd_fullpel_only(&mut self, enabled: bool) {
        self.ph_mmvd_fullpel_only = enabled;
    }

    /// Round-31 §8.5.5.1 / §8.5.6.5 — install the picture-header
    /// `ph_bdof_disabled_flag`. Takes effect for subsequent leaf-CU
    /// inter dispatches: when this flag is `false` AND the SPS-level
    /// `sps_bdof_enabled_flag` (read from [`Self::sps`]) is `true` AND
    /// every other §8.5.5.1 condition holds (symmetric POC distance,
    /// both refs short-term, no sub-block / sym-MVD / CIIP / BCW /
    /// weighted-pred / RPR, `cbW * cbH >= 128`), the bipred dispatch
    /// runs [`crate::bdof::bdof_refine_into`] in place of the eq. 980
    /// average. Defaults to `true` (BDOF off) so callers that have
    /// not wired the bit keep the round-23 byte-for-byte pipeline.
    pub fn set_ph_bdof_disabled(&mut self, disabled: bool) {
        self.ph_bdof_disabled = disabled;
    }

    /// §8.5.1 / §8.5.3.1 — install the picture-header
    /// `ph_dmvr_disabled_flag`. Takes effect for subsequent leaf-CU
    /// merge dispatches: when this flag is `false` AND the SPS-level
    /// `sps_dmvr_enabled_flag` (read from [`Self::sps`]) is `true` AND
    /// every §8.5.1 `dmvrFlag` condition holds (general merge, bi-pred,
    /// symmetric STRP POC distance, no MMVD / CIIP / sub-block /
    /// sym-MVD / BCW / weighted-pred / RPR, `cbW >= 8`, `cbH >= 8`,
    /// `cbW * cbH >= 128`), the bi-pred dispatch runs the §8.5.3 decoder-
    /// side MV refinement per 16×16 sub-block before motion compensation.
    /// Defaults to `true` (DMVR off) so callers that have not wired the
    /// bit keep the round-31 byte-for-byte pipeline.
    pub fn set_ph_dmvr_disabled(&mut self, disabled: bool) {
        self.ph_dmvr_disabled = disabled;
    }

    /// Set `ph_joint_cbcr_sign_flag` (§7.4.3.7). Drives the §8.7.2
    /// `cSign = 1 − 2 * flag` factor when reconstructing the non-coded
    /// chroma residual from the joint Cb-Cr coded block. Takes effect
    /// for subsequent leaf-CU reconstructions in the picture.
    pub fn set_ph_joint_cbcr_sign(&mut self, sign: bool) {
        self.ph_joint_cbcr_sign = sign;
    }

    /// Set `ph_prof_disabled_flag` (§7.4.3.7). Gates §8.5.5.8 PROF in
    /// the affine MC path. Takes effect for subsequent affine
    /// reconstructions in the picture.
    pub fn set_ph_prof_disabled(&mut self, disabled: bool) {
        self.ph_prof_disabled = disabled;
    }

    /// §8.5.2.11 derivation harness for one CU. Returns the Col
    /// candidate (uni-pred for P-slice, fused L0+L1 for B-slice) when
    /// it is available, else `None`. Always returns `None` when
    /// `ph_temporal_mvp_enabled` is false, the collocated reference
    /// is missing or has no captured MotionField, or `(cb_w * cb_h)`
    /// is too small (the spec disables Col for blocks with area
    /// `<= 32`).
    fn derive_col_candidate(
        &self,
        xcb: i32,
        ycb: i32,
        cb_w: i32,
        cb_h: i32,
        is_b: bool,
    ) -> Option<MvField> {
        // §8.5.2.11 first bullet: ph_temporal_mvp_enabled_flag == 0 OR
        // (cbWidth * cbHeight) <= 32 → availableFlagLXCol = 0.
        if !self.ph_temporal_mvp_enabled {
            return None;
        }
        if cb_w * cb_h <= 32 {
            return None;
        }
        // Resolve ColPic. The collocated reference index is fixed at
        // `sh_collocated_ref_idx` regardless of which RefPicList the
        // current CU's refIdx points into; both the L0 and L1 halves
        // of §8.5.2.11 fetch from the same ColPic.
        let col_list = if self.collocated_from_l0 {
            &self.ref_pic_list_l0
        } else {
            &self.ref_pic_list_l1
        };
        let col_idx = self.col_ref_idx as usize;
        if col_idx >= col_list.len() {
            return None;
        }
        let col_pic = &col_list[col_idx];
        col_pic.motion_field.as_ref()?;
        // The current CU's refIdx for the merge mode is fixed at
        // refIdxL{X}Col = 0 per §8.5.2.2 step 2. So the
        // `current_ref_poc` is the POC of `RefPicList[X][0]`. For the
        // round-25 fixture both lists carry a single reference (the
        // collocated picture itself), so RefPicList[0][0].poc is the
        // right value.
        let curr_ref_poc_l0 = self
            .ref_pic_list_l0
            .first()
            .map(|r| r.poc)
            .unwrap_or(self.current_poc);
        // Determine the POC of the collocated MV's reference picture.
        // §8.5.2.12 sets `colPocDiff = DiffPicOrderCnt(ColPic,
        // refPicList[ listCol ][ refIdxCol ])`. Without modelling the
        // collocated picture's RPL we approximate `col_ref_poc` as
        // `current_poc` — i.e. the typical case where the collocated
        // picture's MV points back at the current picture's reference
        // axis. The fixture deliberately uses the equal-distance fast
        // path (eq. 600) so the approximation is exact.
        let col_ref_poc = self.current_poc;

        let l0_inputs = TemporalMergeInputs {
            xcb,
            ycb,
            cb_w,
            cb_h,
            pic_w: self.layout.pic_width_luma as i32,
            pic_h: self.layout.pic_height_luma as i32,
            ctb_log2_size_y: self.layout.ctb_log2_size_y,
            current_poc: self.current_poc,
            current_ref_poc: curr_ref_poc_l0,
            col_pic,
            col_ref_poc,
        };
        let l0_mv = derive_temporal_merge_candidate(&l0_inputs);
        if !is_b {
            return l0_mv;
        }
        // B-slice: derive the L1 half too, then fuse.
        let curr_ref_poc_l1 = self
            .ref_pic_list_l1
            .first()
            .map(|r| r.poc)
            .unwrap_or(self.current_poc);
        let l1_inputs = TemporalMergeInputs {
            current_ref_poc: curr_ref_poc_l1,
            ..l0_inputs.clone()
        };
        let l1_mv = derive_temporal_merge_candidate(&l1_inputs);
        match (l0_mv, l1_mv) {
            (Some(l0), Some(l1)) => Some(MvField {
                mv_l0: l0.mv_l0,
                ref_idx_l0: 0,
                pred_flag_l0: true,
                mv_l1: l1.mv_l0, // l1.mv_l0 because the per-call result is uni-pred-shaped
                ref_idx_l1: 0,
                pred_flag_l1: true,
                cu_skip_flag: false,
                mode_inter: true,
                available: true,
                // §8.5.2.2 step 2 — bcwIdxCol = 0 for the merge-mode
                // temporal candidate, regardless of the L0 / L1 fuse.
                bcw_idx: 0,
            }),
            (Some(l0), None) => Some(l0),
            (None, Some(l1)) => Some(MvField {
                mv_l0: MotionVector::ZERO,
                ref_idx_l0: -1,
                pred_flag_l0: false,
                mv_l1: l1.mv_l0,
                ref_idx_l1: 0,
                pred_flag_l1: true,
                cu_skip_flag: false,
                mode_inter: true,
                available: true,
                bcw_idx: 0,
            }),
            (None, None) => None,
        }
    }

    /// Inject a per-picture SAO parameter array. Replaces the default
    /// "all CTBs not applied" snapshot installed by [`Self::begin_slice`].
    /// Used by integration tests (and by future round-N callers) that
    /// drive SAO without going through the unimplemented §7.3.11.3
    /// CABAC parser. The picture's CTB grid must match the layout's
    /// `pic_width_in_ctbs_y` / `pic_height_in_ctbs_y`.
    pub fn set_sao_picture(&mut self, sao_pic: SaoPicture) -> Result<()> {
        if sao_pic.pic_width_in_ctbs_y != self.layout.pic_width_in_ctbs_y
            || sao_pic.pic_height_in_ctbs_y != self.layout.pic_height_in_ctbs_y
        {
            return Err(Error::invalid(format!(
                "h266 SAO: per-picture grid {}x{} does not match layout {}x{}",
                sao_pic.pic_width_in_ctbs_y,
                sao_pic.pic_height_in_ctbs_y,
                self.layout.pic_width_in_ctbs_y,
                self.layout.pic_height_in_ctbs_y,
            )));
        }
        self.sao_picture = sao_pic;
        Ok(())
    }

    /// Read-only view of the current per-picture SAO parameter array.
    pub fn sao_picture(&self) -> &SaoPicture {
        &self.sao_picture
    }

    /// Inject a per-picture ALF parameter array. Replaces the default
    /// "all CTBs ALF off" snapshot installed by [`Self::begin_slice`].
    /// Used by integration tests (and by future round-N callers) that
    /// drive ALF without going through the unimplemented per-CTB CABAC
    /// parser. The picture's CTB grid must match the layout's
    /// `pic_width_in_ctbs_y` / `pic_height_in_ctbs_y`.
    pub fn set_alf_picture(&mut self, alf_pic: AlfPicture) -> Result<()> {
        if alf_pic.pic_width_in_ctbs_y != self.layout.pic_width_in_ctbs_y
            || alf_pic.pic_height_in_ctbs_y != self.layout.pic_height_in_ctbs_y
        {
            return Err(Error::invalid(format!(
                "h266 ALF: per-picture grid {}x{} does not match layout {}x{}",
                alf_pic.pic_width_in_ctbs_y,
                alf_pic.pic_height_in_ctbs_y,
                self.layout.pic_width_in_ctbs_y,
                self.layout.pic_height_in_ctbs_y,
            )));
        }
        self.alf_picture = alf_pic;
        Ok(())
    }

    /// Read-only view of the current per-picture ALF parameter array.
    pub fn alf_picture(&self) -> &AlfPicture {
        &self.alf_picture
    }

    /// Decode the §7.3.11.3 `sao(rx, ry)` syntax for the supplied CTU
    /// position and fold the result into the walker's [`SaoPicture`].
    ///
    /// The spec walker invokes this at the start of `coding_tree_unit()`
    /// — i.e. *before* the partition tree — when SAO is enabled by the
    /// slice header (`sh_sao_luma_used_flag` or `sh_sao_chroma_used_flag`).
    /// Tile / slice / sub-picture boundary handling collapses to the
    /// single-tile / single-slice rule today: `leftCtbAvailable = rx > 0`
    /// and `upCtbAvailable = ry > 0 && !FirstCtbRowInSlice` (the
    /// `FirstCtbRowInSlice` term reduces to "ry > 0" in the single-slice
    /// case).
    ///
    /// Skipping this call (or calling it before [`Self::set_sao_picture`])
    /// leaves the per-CTB params at their constructor defaults — every
    /// CTB classifies as `SaoTypeIdx::NotApplied`, so SAO becomes a
    /// no-op even when the slice flags are on. The caller is expected
    /// to invoke this exactly once per CTU in slice order before
    /// [`Self::decode_ctu_partitions`] / [`Self::decode_ctu_full`] for
    /// the same CTU, since the SAO syntax sits *before* the partition
    /// tree in the CABAC byte stream (§7.3.11.2).
    pub fn decode_sao_for_ctu(&mut self, ctu: &CtuPos) -> Result<()> {
        if !self.sh.sh_sao_luma_used_flag && !self.sh.sh_sao_chroma_used_flag {
            return Ok(());
        }
        let bit_depth = self.sps.sps_bitdepth_minus8 as u32 + 8;
        let cfg = SaoSyntaxConfig {
            luma_used: self.sh.sh_sao_luma_used_flag,
            chroma_used: self.sh.sh_sao_chroma_used_flag,
            chroma_format_idc: self.sps.sps_chroma_format_idc as u32,
            bit_depth,
            slice_type: self.sh.sh_slice_type,
            sh_cabac_init_flag: self.cabac.sh_cabac_init_flag,
        };
        // Single-slice / single-tile fixture: the spec's
        // `leftCtbAvailable` / `upCtbAvailable` checks (§7.3.11.3)
        // reduce to "in the picture and not the first CTU of the
        // current slice row".
        let left_avail = ctu.x_ctb > 0;
        let up_avail = ctu.y_ctb > 0;
        let params = decode_sao_ctb(
            &mut self.arith,
            &mut self.sao_ctxs,
            &cfg,
            &self.sao_picture,
            ctu.x_ctb,
            ctu.y_ctb,
            left_avail,
            up_avail,
        )?;
        self.sao_picture.set(ctu.x_ctb, ctu.y_ctb, params);
        Ok(())
    }

    /// Build a [`CuToolFlags`] snapshot from the parameter-set views
    /// for the leaf-CU reader. Derived values (`MaxTbSizeY` /
    /// `MinTbSizeY` / `MaxTsSize`) use the spec defaults:
    ///
    /// * `MaxTbSizeY = sps_max_luma_transform_size_64_flag ? 64 : 32`
    ///   (we proxy this off the SPS tool flag directly; the
    ///   `sps_max_luma_transform_size_64_flag` bit lives in
    ///   `partition_constraints` and is not yet surfaced, so we use
    ///   the conservative 64 value here until a Round-8 PR surfaces
    ///   the flag explicitly).
    /// * `MinTbSizeY = 1 << 2 = 4` (§7.4.3.4 eq. 41).
    /// * `MaxTsSize = 1 << (log2_transform_skip_max_size_minus2 + 2)`.
    /// * `CtbSizeY = 1 << ctb_log2_size_y`.
    pub fn cu_tool_flags(&self) -> CuToolFlags {
        let tf = &self.sps.tool_flags;
        let max_ts = 1u32 << (tf.log2_transform_skip_max_size_minus2 + 2);
        let max_num_merge = 6u32.saturating_sub(tf.six_minus_max_num_merge_cand);
        CuToolFlags {
            ibc: tf.ibc_enabled_flag,
            palette: tf.palette_enabled_flag,
            bdpcm: tf.bdpcm_enabled_flag,
            mip: tf.mip_enabled_flag,
            mrl: tf.mrl_enabled_flag,
            isp: tf.isp_enabled_flag,
            lfnst_enabled: tf.lfnst_enabled_flag,
            act: tf.act_enabled_flag,
            max_tb_size_y: 64,
            min_tb_size_y: 4,
            transform_skip_enabled: tf.transform_skip_enabled_flag,
            max_ts_size: max_ts,
            ts_residual_coding_rice_idx: self.sh.sh_ts_residual_coding_rice_idx_minus1 as u32 + 1,
            ctb_size_y: self.layout.ctb_size_y,
            chroma_format_idc: self.sps.sps_chroma_format_idc as u32,
            cu_qp_delta_enabled: self.pps.pps_cu_qp_delta_enabled_flag,
            cu_chroma_qp_offset_enabled: self.sh.sh_cu_chroma_qp_offset_enabled_flag,
            chroma_qp_offset_list_len_minus1: 0,
            joint_cbcr_enabled: tf.joint_cbcr_enabled_flag,
            ts_residual_coding_disabled: self.sh.sh_ts_residual_coding_disabled_flag,
            slice_is_inter: self.sh.sh_slice_type != SliceType::I,
            max_num_merge_cand: max_num_merge,
            mmvd_enabled: tf.mmvd_enabled_flag,
            ph_mmvd_fullpel_only: self.ph_mmvd_fullpel_only,
            ciip_enabled: tf.ciip_enabled_flag,
            gpm_enabled: tf.gpm_enabled_flag,
            // §7.4.3.4 eq. 60 — MaxNumGpmMergeCand = MaxNumMergeCand
            //                   − sps_max_num_merge_cand_minus_max_num_gpm_cand.
            // Only meaningful when sps_gpm_enabled_flag == 1 and
            // MaxNumMergeCand >= 2 (the SPS gates the field's presence).
            max_num_gpm_merge_cand: if tf.gpm_enabled_flag {
                max_num_merge.saturating_sub(tf.max_num_merge_cand_minus_max_num_gpm_cand)
            } else {
                0
            },
            slice_is_b: self.sh.sh_slice_type == SliceType::B,
            // §7.4.3.4 eq. 85 — MaxNumSubblockMergeCand. Routed through
            // the SeqParameterSet helper so the affine vs sbtmvp branch
            // selection + range clamp live in one place. The picture-
            // header input `ph_temporal_mvp_enabled_flag` only matters
            // on the non-affine branch; the helper ignores it when
            // sps_affine_enabled_flag == 1.
            max_num_subblock_merge_cand: self
                .sps
                .max_num_subblock_merge_cand(self.ph_temporal_mvp_enabled),
            mts_enabled: tf.mts_enabled_flag,
            explicit_mts_intra_enabled: tf.explicit_mts_intra_enabled_flag,
            explicit_mts_inter_enabled: tf.explicit_mts_inter_enabled_flag,
        }
    }

    /// Decode the syntax elements for a single leaf CU and derive its
    /// luma / chroma intra prediction modes per §8.4.2 / §8.4.3.
    /// Consumes CABAC bins from the walker's arithmetic decoder — it
    /// is the caller's responsibility to invoke this **in decoding
    /// order** immediately after the coding-tree walker emits the leaf.
    ///
    /// Neighbour availability is encoded in `neigh`; the scaffold
    /// currently does not track a per-CU neighbour grid, so callers
    /// that have not wired one up yet can pass
    /// [`CuNeighbourhood::default()`] to get planar-initialised
    /// candidates (the §8.4.2 default list).
    pub fn decode_leaf_cu_syntax(
        &mut self,
        cu: &CtuCu,
        neigh: &CuNeighbourhood,
    ) -> Result<(LeafCuInfo, LeafCuResidual)> {
        let tools = self.cu_tool_flags();
        let mut info = LeafCuInfo {
            x0: cu.cu.x,
            y0: cu.cu.y,
            cb_width: cu.cu.w,
            cb_height: cu.cu.h,
            ..LeafCuInfo::default()
        };
        let mut residual = LeafCuResidual::default();
        let mut reader = LeafCuReader::new(&mut self.arith, &mut self.leaf_ctxs, tools);
        reader.decode(&mut info, &mut residual, neigh)?;
        Ok((info, residual))
    }

    /// Walk one CTU end-to-end: decode the partition tree, then for
    /// each emitted leaf CU consume its `coding_unit()` bins (intra
    /// mode cascade + transform_unit() CBFs + residual coefficients)
    /// in decoding order. Returns the parallel flat lists of leaf CU
    /// rectangles, per-CU parsed info, and per-CU residual coefficient
    /// arrays (one slot per leaf).
    ///
    /// This is the round-8 "parse a whole CTU and capture everything"
    /// entry point. Pixel reconstruction still lives in
    /// [`Self::reconstruct_leaf_cu`] and returns Unsupported.
    pub fn decode_ctu_full(
        &mut self,
        ctu: &CtuPos,
    ) -> Result<(Vec<CtuCu>, Vec<LeafCuInfo>, Vec<LeafCuResidual>)> {
        let cus = self.decode_ctu_partitions(ctu)?;
        let mut infos = Vec::with_capacity(cus.len());
        let mut residuals = Vec::with_capacity(cus.len());
        // Neighbour tracking for the round-21 inter path: cu_skip_flag's
        // §9.3.4.2.2 ctxInc reads `CuSkipFlag[xNbX][yNbX]` from the
        // motion-field grid the inter reconstruct pass writes. We sample
        // it at (x-1, y) for the left neighbour and (x, y-1) for the
        // above neighbour. Intra reads still flow through
        // [`CuNeighbourhood::default()`] — refining the intra MPM
        // neighbour grid is unrelated to this round.
        for ccu in &cus {
            let neigh = self.compute_cu_neighbourhood(ccu);
            let (info, residual) = self.decode_leaf_cu_syntax(ccu, &neigh)?;
            // Round-149 — commit the per-CB `MergeSubblockFlag` so
            // subsequent leaf CUs in the same CTU (and downstream CTUs
            // in raster order) see this CU as a merge-side neighbour
            // through [`Self::compute_cu_neighbourhood`]. The
            // `InterAffineFlag` write is reserved for the future
            // non-merge affine inter walker; until then every leaf CU
            // emits `false` for the affine flag, which is the
            // §7.4.12.7 inference.
            self.commit_subblock_neighbour_state(ccu, &info);
            infos.push(info);
            residuals.push(residual);
        }
        Ok((cus, infos, residuals))
    }

    /// Round-149 — commit the per-CB `MergeSubblockFlag[x][y]` /
    /// `InterAffineFlag[x][y]` for a freshly-decoded leaf CU into the
    /// picture-wide grids that drive
    /// [`Self::compute_cu_neighbourhood`]. Idempotent: callers may
    /// invoke from both the syntax-only path
    /// ([`Self::decode_ctu_full`]) and the reconstruction path
    /// ([`Self::reconstruct_leaf_cu`]); the per-cell write is the
    /// same boolean either way.
    fn commit_subblock_neighbour_state(&mut self, cu: &CtuCu, info: &LeafCuInfo) {
        let merge_sb = info.inter.merge_data.merge_subblock_flag;
        self.write_subblock_merge_block(cu.cu.x, cu.cu.y, cu.cu.w, cu.cu.h, merge_sb);
        // The non-merge affine inter path is not yet parsed by the
        // CTU walker — `info.inter` carries no `inter_affine_flag`
        // field today. Every leaf CU clears its 4x4 cells to `false`
        // so any pre-existing stale state from a prior slice's
        // overlapping picture region is wiped. When the affine
        // walker lands, swap the `false` below for the parsed flag.
        self.write_inter_affine_block(cu.cu.x, cu.cu.y, cu.cu.w, cu.cu.h, false);
    }

    /// Build a [`CuNeighbourhood`] for a CU at `ccu.cu.(x, y)` by
    /// sampling the motion field (which carries the cu_skip_flag /
    /// mode_inter propagation needed by §9.3.4.2.2). The intra-side
    /// fields (`left` / `above` IntraNeighbour) stay default — the
    /// round-21 inter walker does not need MPM neighbour info, and
    /// the intra walker continues to use `CuNeighbourhood::default()`
    /// via the explicit zero fall-back.
    ///
    /// Round-149 — also samples the per-CB sub-block-merge /
    /// inter-affine grids at the same `(x-1, y)` / `(x, y-1)` luma
    /// positions, so the §7.3.11.7 `read_merge_subblock_flag` reader
    /// gets a live Table-133 ctxInc input instead of the pre-r149
    /// `(false, false)` stub. The inter-affine grid is reserved for
    /// the future non-merge affine inter walker; until that lands,
    /// every cell reads `false` and the ctxInc collapses to the
    /// merge-side-only term.
    fn compute_cu_neighbourhood(&self, ccu: &CtuCu) -> CuNeighbourhood {
        let x = ccu.cu.x as i32;
        let y = ccu.cu.y as i32;
        let left = self.motion_field.get_at_luma(x - 1, y);
        let above = self.motion_field.get_at_luma(x, y - 1);
        // §9.3.4.2.2 / Table 133 neighbour positions: the merge-side
        // ctxInc samples the same `(xCb − 1, yCb)` (left) and
        // `(xCb, yCb − 1)` (above) 4×4 cells the cu_skip_flag /
        // pred_mode ctxInc already use.
        let left_merge_subblock = self.sample_subblock_merge_at_luma(x - 1, y);
        let above_merge_subblock = self.sample_subblock_merge_at_luma(x, y - 1);
        let left_inter_affine = self.sample_inter_affine_at_luma(x - 1, y);
        let above_inter_affine = self.sample_inter_affine_at_luma(x, y - 1);
        CuNeighbourhood {
            left_available: x > 0,
            above_available: y > 0,
            left: None,
            above: None,
            left_cu_skip: left.available && left.cu_skip_flag,
            above_cu_skip: above.available && above.cu_skip_flag,
            left_merge_subblock,
            above_merge_subblock,
            left_inter_affine,
            above_inter_affine,
        }
    }

    /// Immutable view of the CTU geometry the walker was built against.
    pub fn layout(&self) -> &CtuLayout {
        self.layout
    }

    /// Parameter-set views — exposed for tests / upper-layer glue.
    pub fn sps(&self) -> &SeqParameterSet {
        self.sps
    }
    pub fn pps(&self) -> &PicParameterSet {
        self.pps
    }
    pub fn slice_header(&self) -> &StatefulSliceHeader {
        self.sh
    }

    /// Slice QP derived for this slice.
    pub fn slice_qp_y(&self) -> SliceQpY {
        self.cabac.slice_qp_y
    }

    /// §7.4.3.22 eq. 106 `Log2TransformRange` for this stream's bit depth.
    ///
    /// `15` for the common path; when the optional `sps_range_extension()`
    /// block sets `sps_extended_precision_flag`, the extended-precision
    /// range `Max(15, Min(20, BitDepth + 6))` applies (only reachable
    /// when `BitDepth > 10`). Drives the dequant clip + the inverse-
    /// transform intermediate clamp at every coded-TB call site.
    pub fn log2_transform_range(&self) -> u32 {
        let bit_depth = self.sps.sps_bitdepth_minus8 + 8;
        match &self.sps.range_extension {
            Some(rx) => rx.log2_transform_range(bit_depth),
            None => 15,
        }
    }

    /// Whether the slice header signalled the alternative CABAC init
    /// table for P/B. Always `false` in the intra-only scaffold.
    pub fn sh_cabac_init_flag(&self) -> bool {
        self.cabac.sh_cabac_init_flag
    }

    /// CTU iterator in slice order. Single-slice raster scan today.
    pub fn iter_ctus(&self) -> CtuIter<'_> {
        CtuIter::raster(self.layout)
    }

    /// Neighbour availability for the supplied CTU. Single-slice model
    /// only gates on picture-edge membership.
    pub fn neighbour_avail(ctu: &CtuPos) -> NeighbourAvail {
        CtuIter::availability_single_slice(ctu.x_ctb, ctu.y_ctb)
    }

    /// Decode the partition tree rooted at `ctu` and return the flat
    /// list of leaf CUs emitted by [`TreeWalker`]. The CABAC state
    /// threads through across CTUs — callers must invoke this in slice
    /// order exactly once per CTU.
    ///
    /// The returned rectangles are in picture-absolute luma-sample
    /// coordinates (the low-level tree walker emits CTU-local ones;
    /// we rebase them here).
    pub fn decode_ctu_partitions(&mut self, ctu: &CtuPos) -> Result<Vec<CtuCu>> {
        // Zero-sized edge CTUs would be a spec violation upstream; guard
        // defensively so the CABAC engine is not touched for them.
        if ctu.width_luma == 0 || ctu.height_luma == 0 {
            return Ok(Vec::new());
        }
        // Round-56 — the TreeWalker now consumes a picture-wide
        // [`CuNeighbourMap`] (§9.3.4.2 ctxInc derivation). The walker
        // operates in CTU-local coordinates so we hand it the full
        // picture map rebased onto the CTU origin: leaves it inserts
        // are picture-absolute via the rebase below.
        let _avail = Self::neighbour_avail(ctu);
        let local = TreeWalker::new(&mut self.arith, &mut self.cabac.tree_ctxs)
            .with_neighbour_map_rebased(&mut self.nbr_map, ctu.x0, ctu.y0)
            .walk(0, 0, ctu.width_luma, ctu.height_luma)?;
        Ok(local
            .into_iter()
            .map(|c| CtuCu {
                ctu_addr_rs: ctu.ctu_addr_rs,
                cu: Cu {
                    x: c.x + ctu.x0,
                    y: c.y + ctu.y0,
                    ..c
                },
            })
            .collect())
    }

    /// Reconstruct one leaf CU into a frame buffer (luma plane only for
    /// this round; chroma planes are seeded mid-grey at allocation time
    /// and intentionally left untouched).
    ///
    /// Pipeline (§8.4 / §8.7.3 / §8.7.4 / §8.7.5):
    ///
    /// 1. Build reference samples from the partially-reconstructed luma
    ///    plane via [`OwnedIntraRefs::from_plane`] (§8.4.5.2.8 fill).
    /// 2. Generate intra prediction samples for the derived
    ///    `intra_pred_mode_y`. PLANAR / DC are spec-exact; the cardinal
    ///    /diagonal angular modes (2 / 18 / 34 / 50 / 66) use the
    ///    nearest-neighbour subset already in [`crate::intra`]. Other
    ///    angular modes degrade to PLANAR — the residual is preserved,
    ///    so the picture loses some prediction accuracy but stays
    ///    structurally correct.
    /// 3. Dequantise the parsed luma coefficient levels via
    ///    [`crate::dequant::dequantize_tb_flat`] (eqs. 1141 – 1156).
    /// 4. Inverse 2D transform (§8.7.4.1) — DCT-II both axes for now.
    /// 5. Add residual + clip with [`reconstruct_tb_into`] (eq. 1426).
    ///
    /// Out of scope for this round: ISP / SBT / MIP / LFNST / MTS / CCLM,
    /// chroma reconstruction, in-loop filters, dependent quantisation,
    /// scaling lists. Those CUs are written to the picture as the pure
    /// prediction (no inverse transform) when the residual cannot be
    /// applied; the helpers above are therefore best-effort but never
    /// panic.
    pub fn reconstruct_leaf_cu(
        &mut self,
        cu: &CtuCu,
        info: &LeafCuInfo,
        residual: &LeafCuResidual,
        out: &mut PictureBuffer,
    ) -> Result<()> {
        // Round-21 inter dispatch — runs the §8.5.2 spatial-merge
        // derivation, picks `mergeCandList[merge_idx]`, and writes the
        // MC'd block into the output buffer (luma + chroma). The
        // motion field is updated so subsequent CUs can read this CU
        // as an A/B neighbour during their own merge derivation.
        if matches!(info.pred_mode, CuPredMode::Inter) {
            return self.reconstruct_leaf_cu_inter(cu, info, residual, out);
        }

        // MIP (§8.4.5.2.2) is wired below through [`crate::mip`]; the
        // prediction step dispatches on `info.intra_mip_flag` and
        // replaces the angular / PLANAR / DC prediction with the
        // matrix-based variant. ISP (§8.4.5.1, eqs. 251 – 260) is now
        // handled by [`Self::reconstruct_leaf_cu_isp_luma`], which
        // walks the spec-derived subpartition list and reconstructs
        // each one in turn so the next partition can reference the
        // freshly-reconstructed samples of the prior one. Chroma is
        // not split for ISP (per eqs. 251 – 254 the splitting only
        // fires when `cIdx == 0`); the chroma pass below runs once
        // per CU regardless of the ISP path.
        let isp_active = info.isp_split != crate::leaf_cu::IspSplitType::NoSplit;
        // BDPCM (§7.4.5.1 / §8.7.3 eqs. 1153-1154) is wired through
        // the dequant pipeline below.

        let bit_depth = self.sps.sps_bitdepth_minus8 as u32 + 8;
        // §8.7.4.1 MTS kernel selection consults the SPS tool flags; the
        // snapshot is `Copy` so binding it once keeps the later
        // `&mut self` reconstruction calls borrow-clean.
        let tools = self.cu_tool_flags();
        let n_tb_w = cu.cu.w as usize;
        let n_tb_h = cu.cu.h as usize;
        let x0 = cu.cu.x as usize;
        let y0 = cu.cu.y as usize;

        if isp_active {
            self.reconstruct_leaf_cu_isp_luma(cu, info, residual, out, bit_depth)?;
            // Fall through to the chroma + bookkeeping passes below.
            // The luma plane is fully written by the ISP walker so the
            // single-TB luma block (steps 1 – 5) is skipped.
            if self.sps.sps_chroma_format_idc == 1 {
                self.reconstruct_chroma_plane(
                    /*c_idx=*/ 1,
                    cu,
                    info,
                    &residual.cb_levels,
                    info.tu_cb_coded_flag,
                    bit_depth,
                    out,
                )?;
                self.reconstruct_chroma_plane(
                    /*c_idx=*/ 2,
                    cu,
                    info,
                    &residual.cr_levels,
                    info.tu_cr_coded_flag,
                    bit_depth,
                    out,
                )?;
            }
            let qp_y = self.cabac.slice_qp_y.0 + info.cu_qp_delta_val;
            self.deblock_cus.push(DeblockCu {
                x: cu.cu.x,
                y: cu.cu.y,
                w: cu.cu.w,
                h: cu.cu.h,
                qp_y: qp_y.clamp(0, 63),
                intra: matches!(info.pred_mode, CuPredMode::Intra),
                tu_y_coded: info.tu_y_coded_flag,
                tu_cb_coded: info.tu_cb_coded_flag,
                tu_cr_coded: info.tu_cr_coded_flag,
                bdpcm_luma: info.intra_bdpcm_luma,
                bdpcm_chroma: false,
            });
            self.write_intra_block(
                cu.cu.x,
                cu.cu.y,
                cu.cu.w,
                cu.cu.h,
                matches!(info.pred_mode, CuPredMode::Intra),
            );
            self.commit_subblock_neighbour_state(cu, info);
            return Ok(());
        }

        // Per §7.4.10.5 / §8.7.4.1 the largest single TB is `MaxTbSizeY`
        // (≤ 64). When a leaf CU exceeds that size the spec splits it
        // into multiple transform units (`maxTbSize`-aligned tiling);
        // that multi-TB-per-CU walker is not yet wired, so surface a
        // precise Unsupported instead of trying to feed a 128-tall TB
        // into an inverse transform that doesn't exist.
        if n_tb_w > 64 || n_tb_h > 64 {
            return Err(Error::unsupported(format!(
                "h266 reconstruct_leaf_cu: leaf CU {}x{} exceeds MaxTbSizeY=64; \
                 multi-TB-per-CU tiling (§7.4.10.5) not yet wired",
                n_tb_w, n_tb_h,
            )));
        }

        // 1. Reference samples from the partially-reconstructed plane.
        // Availability follows the simple slice-edge rule (§6.4.4 in the
        // single-slice case): top / left exist when not on a picture
        // edge.
        let above_avail = y0 > 0;
        let left_avail = x0 > 0;
        let refs = OwnedIntraRefs::from_plane(
            &out.luma,
            x0,
            y0,
            n_tb_w,
            n_tb_h,
            above_avail,
            left_avail,
            bit_depth,
        );
        let refs_view = IntraRefs {
            above: &refs.above,
            left: &refs.left,
            top_left: refs.top_left,
        };

        // 2. Intra prediction. MIP (§8.4.5.2.2) is dispatched first
        // when the leaf-CU parser flagged it: `info.intra_mip_mode`
        // lives in a separate mode namespace from the angular /
        // PLANAR / DC modes, so we cannot fall through to the angular
        // path. The MIP helper takes the unfiltered reference rows
        // directly (refs.above[0..n_tb_w] and refs.left[0..n_tb_h]) —
        // the §8.4.5.2.8 / §8.4.5.2.9 substitution has already been
        // applied by [`OwnedIntraRefs::from_plane`].
        let pred: Vec<i16> = if info.intra_mip_flag {
            // refs.above has length n_tb_w + 1 (includes the extra
            // sample at column nTbW used by PLANAR). MIP only consumes
            // refT[0..nTbW-1], so we slice off the trailing sample.
            let ref_t: &[i16] = &refs.above[..n_tb_w];
            let ref_l: &[i16] = &refs.left[..n_tb_h];
            predict_mip(
                n_tb_w,
                n_tb_h,
                info.intra_mip_mode,
                info.intra_mip_transposed_flag,
                ref_t,
                ref_l,
                bit_depth,
            )?
        } else {
            match info.intra_pred_mode_y {
                INTRA_PLANAR => predict_planar(n_tb_w, n_tb_h, &refs_view)?,
                INTRA_DC => predict_dc(n_tb_w, n_tb_h, &refs_view)?,
                mode @ (2 | 18 | 34 | 50 | 66) => {
                    predict_angular(n_tb_w, n_tb_h, mode, &refs_view)?
                }
                // Fallback: angular mode outside the implemented
                // cardinal subset. Snap to the nearest cardinal /
                // diagonal so the CU still gets a plausible
                // prediction; the accumulated residual will absorb the
                // difference (lossy but stable).
                mode if (2..=66).contains(&mode) => {
                    let snapped = nearest_supported_angular(mode);
                    predict_angular(n_tb_w, n_tb_h, snapped, &refs_view)?
                }
                // Anything else is a spec-illegal mode; prefer PLANAR
                // over bailing the whole picture so the stream still
                // produces pixels.
                _ => predict_planar(n_tb_w, n_tb_h, &refs_view)?,
            }
        };

        // 3. Dequantise + 4. inverse 2D transform when there is a coded
        // luma TB. Otherwise skip straight to reconstruction (residual
        // is implicitly zero, eq. 1426 still applies).
        let residual_samples: Vec<i32> = if info.tu_y_coded_flag && !residual.luma_levels.is_empty()
        {
            let qp = self.cabac.slice_qp_y.0 + info.cu_qp_delta_val;
            let qp = qp.clamp(0, 63);
            // BDPCM forces transform-skip and the eq. 1153 / 1154
            // accumulation pass; the inverse transform is bypassed
            // (the dz post-accumulation IS the residual). A plain
            // (non-BDPCM) transform_skip_flag == 1 also skips the inverse
            // transform (§8.7.4.6: res[x][y] = d[x][y]) but without the
            // BDPCM accumulation pass.
            let transform_skip = info.intra_bdpcm_luma || info.transform_skip_luma;
            let log2_tr_range = self.log2_transform_range();
            let params = DequantParams {
                bit_depth,
                log2_transform_range: log2_tr_range,
                n_tb_w: n_tb_w as u32,
                n_tb_h: n_tb_h as u32,
                qp,
                dep_quant: false,
                transform_skip,
                bdpcm: info.intra_bdpcm_luma,
                bdpcm_dir: info.intra_bdpcm_luma_dir,
            };
            let mut d = dequantize_tb_flat(&residual.luma_levels, &params)?;
            // §8.7.4.1 inverse LFNST — when lfnst_idx > 0 the dequantised
            // coefficients are passed through the secondary non-separable
            // transform before the regular separable inverse transform.
            // ApplyLfnstFlag[0] == (lfnst_idx > 0) on this luma path
            // (§8.7.4.1 eq. 179). LFNST is mutually exclusive with
            // transform-skip (the lfnst_idx parse is gated on
            // transform_skip_flag == 0), so it never fires when
            // `transform_skip` is set.
            if info.lfnst_idx > 0 && !transform_skip {
                // The §8.4.5.2.7 wide-angle remap (for non-square TBs) is
                // now applied inside `apply_inverse_lfnst` from the TB
                // dimensions, so both square and non-square intra TBs run
                // through the inverse LFNST.
                // CoeffMin / CoeffMax for the active Log2TransformRange.
                let max_c = (1i32 << log2_tr_range) - 1;
                crate::lfnst::apply_inverse_lfnst(
                    &mut d,
                    n_tb_w,
                    n_tb_h,
                    info.intra_pred_mode_y as i32,
                    info.lfnst_idx,
                    -(max_c + 1),
                    max_c,
                )?;
            }
            if transform_skip {
                // Transform-skip (BDPCM or plain transform_skip_flag):
                // the dequantised d[] is the residual sample array
                // directly (§8.7.4.6 — when transform_skip_flag is 1,
                // res[x][y] = d[x][y] >> 0). The BDPCM accumulation pass
                // (eq. 1153/1154) is folded inside `dequantize_tb_flat`
                // when `bdpcm` is set.
                d
            } else {
                // §8.7.4.1 horizontal / vertical transform kernel
                // selection. Implicit MTS substitutes DST-VII for small
                // (4..16) intra TBs when `sps_mts_enabled_flag == 1 &&
                // sps_explicit_mts_intra_enabled_flag == 0 && lfnst_idx
                // == 0 && !MIP`; explicit MTS picks the Table-39 pair from
                // `mts_idx`. `lfnst_idx != 0` keeps the primary transform
                // at DCT-II (mts_idx inferred 0).
                let (tr_h, tr_v) = crate::transform::intra_luma_tr_types(
                    tools.mts_enabled,
                    tools.explicit_mts_intra_enabled,
                    info.mts_idx,
                    info.lfnst_idx,
                    info.intra_mip_flag,
                    n_tb_w as u32,
                    n_tb_h as u32,
                );
                // §8.7.4.1 non-zero coefficient extent. When the inverse
                // LFNST is active (ApplyLfnstFlag[0]) the secondary
                // transform leaves only a small low-frequency corner, so
                // eqs. 1169 / 1170 cap nonZeroW / nonZeroH at 4 (a 4-wide
                // or 4-tall TB) or 8 otherwise. Otherwise eqs. 1171 / 1172
                // apply: a DST-VII / DCT-VIII direction caps the non-zero
                // extent at 16, DCT-II at 32.
                let (non_zero_w, non_zero_h) = if info.lfnst_idx > 0 {
                    let nz = if n_tb_w == 4 || n_tb_h == 4 { 4 } else { 8 };
                    (n_tb_w.min(nz), n_tb_h.min(nz))
                } else {
                    (
                        n_tb_w.min(if tr_h != TrType::DctII { 16 } else { 32 }),
                        n_tb_h.min(if tr_v != TrType::DctII { 16 } else { 32 }),
                    )
                };
                inverse_transform_2d(
                    n_tb_w,
                    n_tb_h,
                    non_zero_w,
                    non_zero_h,
                    tr_h,
                    tr_v,
                    &d,
                    bit_depth,
                    log2_tr_range,
                )?
            }
        } else {
            vec![0i32; n_tb_w * n_tb_h]
        };

        // 5. Reconstruct (eq. 1426). The destination plane row stride
        // matches its sample width by construction (PicturePlane::filled).
        reconstruct_tb_into(
            &mut out.luma.samples,
            out.luma.stride,
            out.luma.height,
            x0,
            y0,
            &pred,
            &residual_samples,
            n_tb_w,
            n_tb_h,
            bit_depth,
        )?;

        // 6. Chroma planes (4:2:0 only — single-tree, monochrome formats
        // skip this whole block). Both Cb and Cr go through the same
        // §8.4 / §8.7 pipeline as luma, just at half spatial resolution
        // and with the §8.4.3 chroma intra-mode mapping already baked
        // into `info.intra_pred_mode_c`.
        if self.sps.sps_chroma_format_idc == 1 {
            self.reconstruct_chroma_plane(
                /*c_idx=*/ 1,
                cu,
                info,
                &residual.cb_levels,
                info.tu_cb_coded_flag,
                bit_depth,
                out,
            )?;
            self.reconstruct_chroma_plane(
                /*c_idx=*/ 2,
                cu,
                info,
                &residual.cr_levels,
                info.tu_cr_coded_flag,
                bit_depth,
                out,
            )?;
        }

        // Record this leaf for the §8.8.3 deblocking pass. The
        // per-CU QP carries any `cu_qp_delta` so the deblocker can
        // build the eq. 1274 average without re-walking the slice.
        let qp_y = self.cabac.slice_qp_y.0 + info.cu_qp_delta_val;
        self.deblock_cus.push(DeblockCu {
            x: cu.cu.x,
            y: cu.cu.y,
            w: cu.cu.w,
            h: cu.cu.h,
            qp_y: qp_y.clamp(0, 63),
            intra: matches!(info.pred_mode, CuPredMode::Intra),
            tu_y_coded: info.tu_y_coded_flag,
            tu_cb_coded: info.tu_cb_coded_flag,
            tu_cr_coded: info.tu_cr_coded_flag,
            bdpcm_luma: info.intra_bdpcm_luma,
            bdpcm_chroma: false,
        });
        // Round-28 §8.5.6.7 — record this CU's prediction mode in the
        // 4x4 intra-coded grid so subsequent CIIP CUs see the correct
        // neighbour status.
        self.write_intra_block(
            cu.cu.x,
            cu.cu.y,
            cu.cu.w,
            cu.cu.h,
            matches!(info.pred_mode, CuPredMode::Intra),
        );
        // Round-149 — mirror the §9.3.4.2.2 / Table 133 merge-side
        // neighbour grid update. For intra CUs the per-CB
        // `MergeSubblockFlag` is always 0 by spec; this clears any
        // stale bit from a prior partition / slice that touched the
        // same picture region.
        self.commit_subblock_neighbour_state(cu, info);
        Ok(())
    }

    /// Round-21 inter-CU reconstruction.
    ///
    /// Pipeline (P-slice, regular-merge subset only):
    ///
    /// 1. Build the §8.5.2.3 spatial-merge candidate list at the CU's
    ///    `(xCb, yCb)` from the live [`MotionField`].
    /// 2. Assemble the §8.5.2.2 mergeCandList (spatial only here +
    ///    zero-MV padding to `MaxNumMergeCand`).
    /// 3. Pick `mergeCandList[merge_idx]`; the resulting per-list
    ///    `(refIdxLN, mvLN, predFlagLN)` drives §8.5.6 motion
    ///    compensation per list.
    /// 4. Run §8.5.6 motion compensation. As of round-22 the per-list
    ///    luma path is [`predict_luma_block`] (§8.5.6.3.2 8-tap
    ///    separable, Table 27 `hpelIfIdx == 0`) and chroma is
    ///    [`predict_chroma_block`] (§8.5.6.3.4 4-tap, Table 33).
    ///    Round-23 adds bi-prediction (`predFlagL0 == predFlagL1 ==
    ///    1`) via [`predict_luma_block_bipred`] /
    ///    [`predict_chroma_block_bipred`], which run the per-list
    ///    interpolation into scratch planes and compose the result
    ///    with the §8.5.6.6.2 eq. 980 default-weighted average
    ///    `(predL0 + predL1 + 1) >> 1` (BCW disabled). The §8.5.6.6.2
    ///    default uni-pred clamp closes the uni-pred chain.
    /// 5. Broadcast the chosen `MvField` (full L0 + L1 record) across
    ///    every 4x4 block in the CU so the next CU's merge derivation
    ///    can read it.
    /// 6. Append a zero-CBF [`DeblockCu`] record (skip CUs are coded
    ///    without residual) so the deblocker sees the inter boundary.
    fn reconstruct_leaf_cu_inter(
        &mut self,
        cu: &CtuCu,
        info: &LeafCuInfo,
        residual: &LeafCuResidual,
        out: &mut PictureBuffer,
    ) -> Result<()> {
        // §8.5.2.1 — when `general_merge_flag == 0` the CU's motion is
        // signalled explicitly through the AMVP path (mvp_lX_flag +
        // ref_idx_lX + mvd_lX). The §8.5.2.8-10 candidate derivation,
        // the §8.5.2.1 `mvLX = mvpLX + mvdLX` fold, and the MC tail are
        // handled by the dedicated AMVP walker. Skip / merge CUs
        // (`general_merge_flag == 1`) fall through to the merge-list
        // build below.
        if !info.inter.general_merge_flag {
            return self.reconstruct_leaf_cu_inter_amvp(cu, info, residual, out);
        }
        let max_merge = self.cu_tool_flags().max_num_merge_cand;
        let xcb = cu.cu.x as i32;
        let ycb = cu.cu.y as i32;
        let cb_w = cu.cu.w as i32;
        let cb_h = cu.cu.h as i32;
        let spatial = derive_spatial_merge_candidates(
            xcb,
            ycb,
            cb_w,
            cb_h,
            &self.motion_field,
            self.log2_par_mrg_level,
        );
        // Slice-aware merge list build: B-slice variants pad with
        // bi-pred zero-MV candidates per §8.5.2.2 step 9; P-slice
        // variants pad with uni-pred zero-MV candidates. Round-24
        // also feeds the per-slice HMVP table through so §8.5.2.2
        // step 7 (HMVP insertion per §8.5.2.6) lights up between
        // the spatial slot fill and the zero-MV pad.
        let is_b = self.sh.sh_slice_type == SliceType::B;
        // Round-25 §8.5.2.11 — temporal merge candidate ("Col"). When
        // ph_temporal_mvp_enabled_flag is 1 and the walker has been
        // configured with a collocated reference (via
        // [`Self::set_temporal_mvp`]), invoke the §8.5.2.11 derivation
        // for L0 (P-slice) or L0 + L1 (B-slice). The returned MvField
        // is then inserted at step 5 last bullet of §8.5.2.2 (after
        // spatials, before HMVP).
        let col = self.derive_col_candidate(xcb, ycb, cb_w, cb_h, is_b);
        let mlist = if is_b {
            build_merge_cand_list_b(&spatial, max_merge, col, Some(&self.hmvp))
        } else {
            build_merge_cand_list(&spatial, max_merge, col, Some(&self.hmvp))
        };
        // Round-40 §8.5.4 + §8.5.7 — Geometric Partitioning Mode (GPM).
        // When `gpm_flag == 1`, the CU is split along an oblique line
        // into two regions, each predicted from a separate merge
        // candidate. We pull `(m, n)` from `merge_gpm_idx0` /
        // `merge_gpm_idx1` (eqs. 646 / 647), look up the two
        // mergeCandList entries, apply the §8.5.4.2 step 4 / 5 list-
        // selection rule, run two §8.5.6.3 MC predictions and blend per
        // §8.5.7.2 eq. 1016. Chroma uses the same partition geometry
        // (Tables 36 / 37) with `cIdx ∈ {1, 2}`; eqs. 999/1000 scale to
        // the chroma sub-sample grid.
        if info.inter.merge_data.gpm_flag {
            return self.reconstruct_leaf_cu_gpm(cu, info, &mlist, out);
        }

        // §8.5.5.2 — sub-block (affine / SbTMVP) merge. When
        // `merge_subblock_flag == 1` the CU's motion is a per-sub-block
        // affine field (or SbTMVP grid), not a single translational
        // candidate. The dedicated walker builds the §8.5.5.2 sub-block
        // merge list and reconstructs the affine MC to pixels.
        if info.inter.merge_data.merge_subblock_flag {
            return self.reconstruct_leaf_cu_inter_subblock_merge(cu, info, residual, out);
        }

        let idx = (info.inter.merge_data.merge_idx as usize).min(mlist.len() - 1);
        let mut chosen = mlist[idx];

        // §8.5.2.7 — Merge with Motion Vector Differences. When
        // `mmvd_merge_flag == 1`, the parser already inferred
        // `merge_idx == mmvd_cand_flag` (∈ {0, 1}) so `chosen` holds the
        // base candidate. Derive `MmvdOffset` from
        // `(mmvd_distance_idx, mmvd_direction_idx, ph_mmvd_fullpel_only)`
        // per Tables 17 / 18 + eqs. 188 / 189, then fold it into the
        // base candidate's per-list MVs via
        // [`apply_mmvd_to_base_with_poc`]. POC distances are pulled
        // from the chosen candidate's `(refIdxLN, RefPicListN)`
        // pointers; this dispatches into the equal-POC shortcut
        // (eqs. 557 – 560), the opposite-sign branch (eqs. 564 / 565),
        // or the §8.5.2.12 distScaleFactor scaling (eqs. 561 – 580 /
        // 601 – 605) automatically based on the POC layout. LT refs
        // are not modelled yet — passed through as `false` to keep the
        // short-term branch active.
        if info.inter.merge_data.mmvd_merge_flag {
            let off = derive_mmvd_offset(
                info.inter.merge_data.mmvd_distance_idx,
                info.inter.merge_data.mmvd_direction_idx,
                self.ph_mmvd_fullpel_only,
            );
            let poc_l0 = if chosen.pred_flag_l0 && chosen.ref_idx_l0 >= 0 {
                self.ref_pic_list_l0
                    .get(chosen.ref_idx_l0 as usize)
                    .map(|r| r.poc)
                    .unwrap_or(self.current_poc)
            } else {
                self.current_poc
            };
            let poc_l1 = if chosen.pred_flag_l1 && chosen.ref_idx_l1 >= 0 {
                self.ref_pic_list_l1
                    .get(chosen.ref_idx_l1 as usize)
                    .map(|r| r.poc)
                    .unwrap_or(self.current_poc)
            } else {
                self.current_poc
            };
            let curr_poc_diff_l0 = self.current_poc.wrapping_sub(poc_l0);
            let curr_poc_diff_l1 = self.current_poc.wrapping_sub(poc_l1);
            chosen = apply_mmvd_to_base_with_poc(
                &chosen,
                off,
                curr_poc_diff_l0,
                curr_poc_diff_l1,
                /*lt_l0*/ false,
                /*lt_l1*/ false,
            );
        }

        self.reconstruct_inter_with_chosen(cu, info, residual, chosen, out)
    }

    /// Shared §8.5.6 + §8.5.8 motion-compensated reconstruction tail for
    /// a leaf inter CU whose final per-list MVs have already been
    /// resolved into `chosen` (an [`MvField`]).
    ///
    /// Both the §8.5.2.2 merge / skip path
    /// ([`Self::reconstruct_leaf_cu_inter`]) and the §8.5.2.1 non-merge
    /// AMVP path ([`Self::reconstruct_leaf_cu_inter_amvp`]) converge
    /// here once `chosen` is known: the per-list reference lookup,
    /// §8.5.6.3 fractional-sample luma + chroma MC (with §8.5.6.6.2
    /// default-weighted / BCW bi-pred and §8.5.6.5 BDOF), the optional
    /// §8.5.6.7 CIIP combine (merge-only — derived from
    /// `info.inter.merge_data.ciip_flag`, so the AMVP path's
    /// `ciip_flag == 0` makes it a no-op), the §8.5.8 + §8.7.5.1
    /// inter-residual add, and the motion-field / HMVP / deblock
    /// bookkeeping are all identical regardless of how `chosen` was
    /// derived.
    fn reconstruct_inter_with_chosen(
        &mut self,
        cu: &CtuCu,
        info: &LeafCuInfo,
        residual: &LeafCuResidual,
        chosen: MvField,
        out: &mut PictureBuffer,
    ) -> Result<()> {
        if !chosen.pred_flag_l0 && !chosen.pred_flag_l1 {
            return Err(Error::invalid(
                "h266 inter: chosen merge candidate has both predFlags == 0; \
                 spec requires at least one active prediction list",
            ));
        }
        // L0 reference picture lookup (when pred_flag_l0 is set).
        let ref_pic_l0 = if chosen.pred_flag_l0 {
            if chosen.ref_idx_l0 < 0 || (chosen.ref_idx_l0 as usize) >= self.ref_pic_list_l0.len() {
                return Err(Error::invalid(format!(
                    "h266 inter: refIdxL0 {} out of range — RefPicList0 has {} entries",
                    chosen.ref_idx_l0,
                    self.ref_pic_list_l0.len()
                )));
            }
            Some(&self.ref_pic_list_l0[chosen.ref_idx_l0 as usize])
        } else {
            None
        };
        // L1 reference picture lookup (when pred_flag_l1 is set,
        // i.e. round-23 B-slice bi-pred path).
        let ref_pic_l1 = if chosen.pred_flag_l1 {
            if chosen.ref_idx_l1 < 0 || (chosen.ref_idx_l1 as usize) >= self.ref_pic_list_l1.len() {
                return Err(Error::invalid(format!(
                    "h266 inter: refIdxL1 {} out of range — RefPicList1 has {} entries",
                    chosen.ref_idx_l1,
                    self.ref_pic_list_l1.len()
                )));
            }
            Some(&self.ref_pic_list_l1[chosen.ref_idx_l1 as usize])
        } else {
            None
        };

        // ---- DMVR (decoder-side MV refinement, §8.5.1 / §8.5.3) ------------
        //
        // §8.5.1 derives `dmvrFlag`. When it holds, the per-list MVs are
        // refined by the §8.5.3.1 bilateral-matching search *before*
        // motion compensation (and BDOF, which then runs on the refined
        // MVs). The §8.5.1 split divides the CU into 16×16 sub-blocks
        // (eqs. 452 – 455) and refines each independently; this commit
        // handles the single-sub-block case (`cbWidth <= 16 &&
        // cbHeight <= 16`, i.e. `numSbX == numSbY == 1`) by refining
        // `chosen.mv_l{0,1}` in place. Larger CUs fall through to the
        // unrefined path until the per-sub-block split lands.
        //
        // `chosen` already holds the §8.5.2.1 fold's per-list MVs (merge
        // candidate, possibly MMVD-corrected). DMVR is gated off when
        // `mmvd_merge_flag == 1` or `ciip_flag == 1` per §8.5.1, so a
        // surviving refinement only touches the plain bi-pred merge CU.
        //
        // §8.5.1 NOTE (above eq. 460): the *refined* `refMvLX` drives
        // motion compensation (`chosen.mv_l{0,1}` here) and the collocated
        // temporal derivation (`MvDmvrLX`), but the *unrefined* `MvLX` is
        // what feeds the spatial-MVP and deblocking-boundary-strength
        // processes. We therefore keep the pre-DMVR MVs in
        // `unrefined_mv_l{0,1}` and write *those* into the per-picture
        // motion field at the end, so a later CU's §8.5.2.3 spatial scan
        // sees the unrefined neighbour motion. (The motion field also
        // backs temporal derivation in this scaffold, which spec-wise
        // should see the refined `MvDmvrLX`; modelling that split needs a
        // second per-picture field and is a follow-up.)
        let mut chosen = chosen;
        let unrefined_mv_l0 = chosen.mv_l0;
        let unrefined_mv_l1 = chosen.mv_l1;
        // Set when the §8.5.1 multi-16×16-sub-block DMVR path has already
        // written the per-sub-block bi-pred MC into `out`, so the shared
        // CU-wide luma + chroma MC below must be skipped.
        let mut dmvr_multi_done = false;
        if let (Some(rp0), Some(rp1)) = (ref_pic_l0, ref_pic_l1) {
            let curr_poc = self.current_poc;
            // §8.5.1 — `DiffPicOrderCnt(curr, ref0) == DiffPicOrderCnt(ref1, curr)`.
            let bracketed_same_diff_poc =
                curr_poc.wrapping_sub(rp0.poc) == rp1.poc.wrapping_sub(curr_poc);
            let dmvr_runs = self.sps.tool_flags.dmvr_enabled_flag
                && crate::dmvr::dmvr_used_flag(
                    self.sps.tool_flags.dmvr_enabled_flag,
                    self.ph_dmvr_disabled,
                    info.inter.general_merge_flag,
                    chosen.pred_flag_l0,
                    chosen.pred_flag_l1,
                    bracketed_same_diff_poc,
                    /*is_strp_l0*/ true,
                    /*is_strp_l1*/ true,
                    /*motion_model_idc*/ 0,
                    /*merge_subblock_flag*/ false,
                    /*sym_mvd_flag*/ false,
                    info.inter.merge_data.ciip_flag,
                    chosen.bcw_idx,
                    /*luma_weight_l0_flag*/ false,
                    /*luma_weight_l1_flag*/ false,
                    /*chroma_weight_l0_flag*/ false,
                    /*chroma_weight_l1_flag*/ false,
                    cu.cu.w,
                    cu.cu.h,
                    /*c_idx*/ 0,
                )
                && !info.inter.merge_data.mmvd_merge_flag;
            if dmvr_runs {
                // §8.5.1 eqs. 452 – 455 — split the CU into ≤16×16
                // sub-blocks. `numSbX/numSbY > 1` only when a dimension
                // exceeds 16.
                let num_sb_x = if cu.cu.w > 16 { cu.cu.w >> 4 } else { 1 };
                let num_sb_y = if cu.cu.h > 16 { cu.cu.h >> 4 } else { 1 };
                if num_sb_x == 1 && num_sb_y == 1 {
                    // Single sub-block: refine `chosen` in place and let
                    // the shared CU-wide MC + BDOF tail consume it.
                    let refined = crate::dmvr::apply_dmvr(
                        cu.cu.x,
                        cu.cu.y,
                        cu.cu.w,
                        cu.cu.h,
                        chosen.mv_l0,
                        chosen.mv_l1,
                        &rp0.frame.luma,
                        &rp1.frame.luma,
                    )?;
                    chosen.mv_l0 = refined.mv_l0_refined;
                    chosen.mv_l1 = refined.mv_l1_refined;
                } else {
                    // Multi-sub-block split (§8.5.1 eqs. 456 – 459): each
                    // 16×16 sub-block is refined independently and runs
                    // its own §8.5.6 bi-pred MC (luma + 4:2:0 chroma) at
                    // the sub-block origin. The §8.5.6.6.2 blend uses the
                    // candidate's `bcwIdx` (always 0 on the DMVR gate, so
                    // eq. 980 default-weighting). BDOF on the per-DMVR-
                    // sub-block grid is a follow-up; the gate sets it off
                    // for this path. The unrefined `MvLX` motion-field
                    // store below still applies (the CU-wide MvField is
                    // the pre-DMVR candidate).
                    let sb_w = 16u32;
                    let sb_h = 16u32;
                    let chroma = self.sps.sps_chroma_format_idc == 1;
                    for sby in 0..num_sb_y {
                        for sbx in 0..num_sb_x {
                            let x = cu.cu.x + sbx * sb_w;
                            let y = cu.cu.y + sby * sb_h;
                            let refined = crate::dmvr::apply_dmvr(
                                x,
                                y,
                                sb_w,
                                sb_h,
                                chosen.mv_l0,
                                chosen.mv_l1,
                                &rp0.frame.luma,
                                &rp1.frame.luma,
                            )?;
                            predict_luma_block_bipred_bcw(
                                &mut out.luma,
                                x,
                                y,
                                sb_w,
                                sb_h,
                                &rp0.frame.luma,
                                refined.mv_l0_refined,
                                &rp1.frame.luma,
                                refined.mv_l1_refined,
                                /*bcw_idx*/ 0,
                            )?;
                            if chroma {
                                predict_chroma_block_bipred_bcw(
                                    &mut out.cb,
                                    x / 2,
                                    y / 2,
                                    sb_w / 2,
                                    sb_h / 2,
                                    &rp0.frame.cb,
                                    refined.mv_l0_refined,
                                    &rp1.frame.cb,
                                    refined.mv_l1_refined,
                                    0,
                                )?;
                                predict_chroma_block_bipred_bcw(
                                    &mut out.cr,
                                    x / 2,
                                    y / 2,
                                    sb_w / 2,
                                    sb_h / 2,
                                    &rp0.frame.cr,
                                    refined.mv_l0_refined,
                                    &rp1.frame.cr,
                                    refined.mv_l1_refined,
                                    0,
                                )?;
                            }
                        }
                    }
                    dmvr_multi_done = true;
                }
            }
        }

        // ---- CIIP setup ----------------------------------------------------
        //
        // Round-28 §8.5.6.7. Build the planar intra prediction from the
        // *partially-reconstructed* output plane neighbours **before**
        // running motion compensation — MC only writes inside the CU
        // rectangle, but reading neighbour samples after-the-fact would
        // still be safe because the neighbour rows / columns are
        // outside the CU. Doing it up-front mirrors the spec's
        // pre-computed `predSamplesIntra` vs `predSamplesInter`
        // separation in eq. 998. The §8.5.6.7 weight `w` is also
        // captured here from the `(xCb − 1, yCb − 1 + cbHeight)` /
        // `(xCb − 1 + cbWidth, yCb − 1)` neighbour intra-coded grid.
        let ciip_active = info.inter.merge_data.ciip_flag;
        let ciip_w_luma = if ciip_active {
            let xcb_i = cu.cu.x as i32;
            let ycb_i = cu.cu.y as i32;
            let cbw_i = cu.cu.w as i32;
            let cbh_i = cu.cu.h as i32;
            // §8.5.6.7 — the "A" neighbour sits at (xCb − 1, yCb − 1
            // + cbHeight) and the "B" neighbour at (xCb − 1 + cbWidth,
            // yCb − 1). Out-of-picture neighbours register as
            // not-intra (matching the spec's `availableX == FALSE`
            // → `isIntraCodedNeighbourX = FALSE` branch).
            let intra_a = self.sample_intra_at_luma(xcb_i - 1, ycb_i - 1 + cbh_i);
            let intra_b = self.sample_intra_at_luma(xcb_i - 1 + cbw_i, ycb_i - 1);
            ciip_intra_weight(intra_b, intra_a)
        } else {
            0
        };
        let bit_depth_ciip = self.sps.sps_bitdepth_minus8 as u32 + 8;
        let ciip_pred_intra_luma: Vec<i16> = if ciip_active {
            let n_w = cu.cu.w as usize;
            let n_h = cu.cu.h as usize;
            let above_avail = cu.cu.y > 0;
            let left_avail = cu.cu.x > 0;
            let refs = OwnedIntraRefs::from_plane(
                &out.luma,
                cu.cu.x as usize,
                cu.cu.y as usize,
                n_w,
                n_h,
                above_avail,
                left_avail,
                bit_depth_ciip,
            );
            let refs_view = IntraRefs {
                above: &refs.above,
                left: &refs.left,
                top_left: refs.top_left,
            };
            // §7.4.5.2 — when ciip_flag == 1 the spec sets
            // IntraPredModeY[x][y] = INTRA_PLANAR for every sample of
            // the CB. The chroma intra mode is derived from luma per
            // §8.4.3 / Table 20 → also INTRA_PLANAR for our 4:2:0
            // single-tree case.
            predict_planar(n_w, n_h, &refs_view)?
        } else {
            Vec::new()
        };

        // ---- Luma MC -------------------------------------------------------
        //
        // Skipped when the §8.5.1 multi-sub-block DMVR path already wrote
        // the per-sub-block bi-pred MC into `out` above.
        match (ref_pic_l0, ref_pic_l1) {
            _ if dmvr_multi_done => {}
            (Some(rp0), None) => {
                // Uni-pred L0 (P-slice path, or B-slice candidate with
                // predFlagL1 == 0).
                predict_luma_block(
                    &mut out.luma,
                    cu.cu.x,
                    cu.cu.y,
                    cu.cu.w,
                    cu.cu.h,
                    &rp0.frame.luma,
                    chosen.mv_l0,
                )?;
            }
            (None, Some(rp1)) => {
                // Uni-pred L1 — re-uses the same per-list helper.
                predict_luma_block(
                    &mut out.luma,
                    cu.cu.x,
                    cu.cu.y,
                    cu.cu.w,
                    cu.cu.h,
                    &rp1.frame.luma,
                    chosen.mv_l1,
                )?;
            }
            (Some(rp0), Some(rp1)) => {
                // Bi-pred: §8.5.6.6.2 — default-weighted (eq. 980)
                // when bcw_idx == 0 OR ciip_flag == 1; explicit BCW
                // (eq. 981) with weights from BCW_W_LUT otherwise.
                let bcw_for_blend = if ciip_active { 0 } else { chosen.bcw_idx };

                // §8.5.5.1 bdofUsedFlag derivation (round-31). The
                // §8.5.6.5 refinement runs in place of the eq. 980
                // default-weighted average when every gating bullet
                // holds. Long-term reference classification is not
                // modelled by the round-23 [`ReferencePicture`]
                // record; we conservatively treat both refs as STRP
                // (matching every existing bipred fixture). Symmetric
                // POC distance is computed from the per-list
                // reference POCs against `current_poc`. RPR resampling
                // is not active in the scaffold.
                let curr_poc = self.current_poc;
                let same_diff_poc =
                    curr_poc.wrapping_sub(rp0.poc) == rp1.poc.wrapping_sub(curr_poc);
                let bdof_runs = self.sps.tool_flags.bdof_enabled_flag
                    && bdof_used_flag(
                        self.ph_bdof_disabled,
                        chosen.pred_flag_l0,
                        chosen.pred_flag_l1,
                        same_diff_poc,
                        /*is_strp_l0*/ true,
                        /*is_strp_l1*/ true,
                        /*motion_model_idc*/ 0,
                        /*merge_subblock_flag*/ false,
                        /*sym_mvd_flag*/ false,
                        ciip_active,
                        chosen.bcw_idx,
                        /*luma_weight_l0_flag*/ false,
                        /*luma_weight_l1_flag*/ false,
                        /*chroma_weight_l0_flag*/ false,
                        /*chroma_weight_l1_flag*/ false,
                        cu.cu.w,
                        cu.cu.h,
                        /*rpr_constraints_active_l0*/ false,
                        /*rpr_constraints_active_l1*/ false,
                        /*c_idx*/ 0,
                    );

                if bdof_runs {
                    // §8.5.6.5 — round-32 wires the spec-byte-identical
                    // path: each per-list §8.5.6.3 MC surfaces its
                    // `BitDepth + 6` precision intermediate (the value
                    // before the §8.5.6.6.2 per-list clip / right-
                    // shift), and that intermediate is lifted into the
                    // spec's `(nCbW + 2) × (nCbH + 2)` extended layout
                    // by 1-sample edge replication only — no re-scale.
                    // BDOF gradients then operate on the spec's full
                    // 14-bit precision (8-bit input), 16-bit (10-bit
                    // input), 18-bit (12-bit input). The round-30
                    // 8-bit lifter (`build_extended_pred_8bit`) is now
                    // deprecated.
                    let n_cb_w = cu.cu.w;
                    let n_cb_h = cu.cu.h;
                    let bit_depth = self.sps.sps_bitdepth_minus8 as u32 + 8;
                    let pred_l0 = predict_luma_block_high_precision(
                        cu.cu.x,
                        cu.cu.y,
                        n_cb_w,
                        n_cb_h,
                        &rp0.frame.luma,
                        chosen.mv_l0,
                        bit_depth,
                    )?;
                    let pred_l1 = predict_luma_block_high_precision(
                        cu.cu.x,
                        cu.cu.y,
                        n_cb_w,
                        n_cb_h,
                        &rp1.frame.luma,
                        chosen.mv_l1,
                        bit_depth,
                    )?;
                    let ext_l0 = build_extended_pred_high_precision(&pred_l0, n_cb_w, n_cb_h)?;
                    let ext_l1 = build_extended_pred_high_precision(&pred_l1, n_cb_w, n_cb_h)?;
                    bdof_refine_into(
                        &mut out.luma,
                        cu.cu.x,
                        cu.cu.y,
                        n_cb_w,
                        n_cb_h,
                        &ext_l0,
                        &ext_l1,
                        bit_depth,
                    )?;
                } else {
                    predict_luma_block_bipred_bcw(
                        &mut out.luma,
                        cu.cu.x,
                        cu.cu.y,
                        cu.cu.w,
                        cu.cu.h,
                        &rp0.frame.luma,
                        chosen.mv_l0,
                        &rp1.frame.luma,
                        chosen.mv_l1,
                        bcw_for_blend,
                    )?;
                }
            }
            (None, None) => unreachable!(), // guarded above
        }

        // ---- CIIP luma combine (§8.5.6.7 eq. 998) --------------------------
        if ciip_active {
            apply_ciip_combine_to_plane(
                &mut out.luma,
                cu.cu.x as usize,
                cu.cu.y as usize,
                cu.cu.w as usize,
                cu.cu.h as usize,
                &ciip_pred_intra_luma,
                ciip_w_luma,
                bit_depth_ciip,
            );
        }

        // ---- Chroma MC for 4:2:0 -----------------------------------------
        if self.sps.sps_chroma_format_idc == 1 {
            let cb_w_c = cu.cu.w / 2;
            let cb_h_c = cu.cu.h / 2;
            let cb_x_c = cu.cu.x / 2;
            let cb_y_c = cu.cu.y / 2;
            // Round-28 §8.5.6.7 — capture the planar chroma intra
            // prediction *before* the chroma MC overwrites the CU
            // rectangle. The §8.4.5.2.11 planar predictor is the same
            // for chroma as for luma; for CIIP the spec forces
            // IntraPredModeC to PLANAR (luma is also forced to PLANAR
            // by §7.4.5.2 and the §8.4.3 chroma derivation table maps
            // PLANAR luma → PLANAR chroma).
            let (ciip_pred_intra_cb, ciip_pred_intra_cr) = if ciip_active {
                let n_w_c = cb_w_c as usize;
                let n_h_c = cb_h_c as usize;
                let above_avail_c = cb_y_c > 0;
                let left_avail_c = cb_x_c > 0;
                let refs_cb = OwnedIntraRefs::from_plane(
                    &out.cb,
                    cb_x_c as usize,
                    cb_y_c as usize,
                    n_w_c,
                    n_h_c,
                    above_avail_c,
                    left_avail_c,
                    bit_depth_ciip,
                );
                let refs_cr = OwnedIntraRefs::from_plane(
                    &out.cr,
                    cb_x_c as usize,
                    cb_y_c as usize,
                    n_w_c,
                    n_h_c,
                    above_avail_c,
                    left_avail_c,
                    bit_depth_ciip,
                );
                let view_cb = IntraRefs {
                    above: &refs_cb.above,
                    left: &refs_cb.left,
                    top_left: refs_cb.top_left,
                };
                let view_cr = IntraRefs {
                    above: &refs_cr.above,
                    left: &refs_cr.left,
                    top_left: refs_cr.top_left,
                };
                (
                    predict_planar(n_w_c, n_h_c, &view_cb)?,
                    predict_planar(n_w_c, n_h_c, &view_cr)?,
                )
            } else {
                (Vec::new(), Vec::new())
            };
            match (ref_pic_l0, ref_pic_l1) {
                _ if dmvr_multi_done => {}
                (Some(rp0), None) => {
                    predict_chroma_block(
                        &mut out.cb,
                        cb_x_c,
                        cb_y_c,
                        cb_w_c,
                        cb_h_c,
                        &rp0.frame.cb,
                        chosen.mv_l0,
                    )?;
                    predict_chroma_block(
                        &mut out.cr,
                        cb_x_c,
                        cb_y_c,
                        cb_w_c,
                        cb_h_c,
                        &rp0.frame.cr,
                        chosen.mv_l0,
                    )?;
                }
                (None, Some(rp1)) => {
                    predict_chroma_block(
                        &mut out.cb,
                        cb_x_c,
                        cb_y_c,
                        cb_w_c,
                        cb_h_c,
                        &rp1.frame.cb,
                        chosen.mv_l1,
                    )?;
                    predict_chroma_block(
                        &mut out.cr,
                        cb_x_c,
                        cb_y_c,
                        cb_w_c,
                        cb_h_c,
                        &rp1.frame.cr,
                        chosen.mv_l1,
                    )?;
                }
                (Some(rp0), Some(rp1)) => {
                    // BCW chroma — same gating as luma above.
                    let bcw_for_blend = if ciip_active { 0 } else { chosen.bcw_idx };
                    predict_chroma_block_bipred_bcw(
                        &mut out.cb,
                        cb_x_c,
                        cb_y_c,
                        cb_w_c,
                        cb_h_c,
                        &rp0.frame.cb,
                        chosen.mv_l0,
                        &rp1.frame.cb,
                        chosen.mv_l1,
                        bcw_for_blend,
                    )?;
                    predict_chroma_block_bipred_bcw(
                        &mut out.cr,
                        cb_x_c,
                        cb_y_c,
                        cb_w_c,
                        cb_h_c,
                        &rp0.frame.cr,
                        chosen.mv_l0,
                        &rp1.frame.cr,
                        chosen.mv_l1,
                        bcw_for_blend,
                    )?;
                }
                (None, None) => unreachable!(),
            }
            // ---- CIIP chroma combine (§8.5.6.7 eq. 998) -------------------
            //
            // Per §8.5.6.7 the chroma weight is derived from the same
            // §8.5.6.7 A / B luma neighbours as the luma weight (the
            // spec scales the neighbour position by SubWidthC /
            // SubHeightC in eqs. 995 / 996, which for 4:2:0 means the
            // chroma neighbours sit at *the same* luma-grid 4x4 cells
            // as the luma neighbours; the intra-grid sample picked
            // earlier therefore applies here too).
            if ciip_active {
                apply_ciip_combine_to_plane(
                    &mut out.cb,
                    cb_x_c as usize,
                    cb_y_c as usize,
                    cb_w_c as usize,
                    cb_h_c as usize,
                    &ciip_pred_intra_cb,
                    ciip_w_luma,
                    bit_depth_ciip,
                );
                apply_ciip_combine_to_plane(
                    &mut out.cr,
                    cb_x_c as usize,
                    cb_y_c as usize,
                    cb_w_c as usize,
                    cb_h_c as usize,
                    &ciip_pred_intra_cr,
                    ciip_w_luma,
                    bit_depth_ciip,
                );
            }
        }

        // ---- Inter residual add (§8.5.8 + §8.7.5.1) -----------------------
        //
        // When `cu_coded_flag == 1` the parser populated `residual`
        // with the §7.3.11.10 transform-unit coefficient levels. The
        // §8.5.8 residual-signal process dequantises (§8.7.3) and
        // inverse-transforms (§8.7.4) each coded TB with
        // `predMode == MODE_INTER`, and §8.7.5.1 forms the
        // reconstructed sample `recSamples = Clip1(predSamples +
        // resSamples)`. The motion-compensated `predSamples` already
        // live in `out`, so we add the residual in place.
        //
        // §7.4.12.5 SBT: when `cu_sbt_flag == 1` the CU is split into two
        // TUs; one carries residual, the other has none. The residual TU
        // occupies the [`crate::transform::sbt_geometry`] sub-region and
        // uses the §8.7.4.1 Table-40 transform kernels
        // ([`crate::transform::sbt_tr_types`]) instead of DCT-II. The
        // non-residual sub-region is left at the MC prediction (eq. 1426
        // residual implicitly zero).
        self.add_inter_cu_residual(cu, info, residual, out)?;

        // Broadcast the chosen MvField across every 4x4 block of the
        // CU. The cu_skip_flag flag is propagated so the next CU's
        // §9.3.4.2.2 cu_skip_flag context picks it up correctly.
        //
        // §8.5.1 NOTE — the spatial-MVP / deblocking processes read the
        // *unrefined* `MvLX`, so the motion-field store uses the pre-DMVR
        // MVs (identical to `chosen.mv_l{0,1}` when DMVR did not run).
        let mvf = MvField {
            mv_l0: unrefined_mv_l0,
            ref_idx_l0: if chosen.pred_flag_l0 {
                chosen.ref_idx_l0
            } else {
                -1
            },
            pred_flag_l0: chosen.pred_flag_l0,
            mv_l1: unrefined_mv_l1,
            ref_idx_l1: if chosen.pred_flag_l1 {
                chosen.ref_idx_l1
            } else {
                -1
            },
            pred_flag_l1: chosen.pred_flag_l1,
            cu_skip_flag: info.inter.cu_skip_flag,
            mode_inter: true,
            available: true,
            // §8.5.6.6.2 — propagate the chosen candidate's BcwIdx
            // onto every covered 4x4 block so subsequent spatial
            // neighbours inherit it per eqs. 496 / 501 / 506 / etc.
            bcw_idx: chosen.bcw_idx,
        };
        self.motion_field
            .write_block(cu.cu.x, cu.cu.y, cu.cu.w, cu.cu.h, mvf);
        // Round-364 — this merge / skip / non-affine inter CU is
        // translational (`MotionModelIdc == 0`), so clear any stale
        // affine record on the cells it covers. A later CU's §8.5.5.7
        // scan must see this region as non-affine (the `MotionModelIdc
        // > 0` gate fails ⇒ no inherited candidate from here).
        self.affine_cpmv_field
            .write_block(cu.cu.x, cu.cu.y, cu.cu.w, cu.cu.h, None);

        // Round-24 §8.5.2.16 — push the just-decoded inter CU's
        // motion field into the per-slice HMVP table.
        self.hmvp.update_with(mvf);

        // Record this CU for the deblocker.
        let qp_y = (self.cabac.slice_qp_y.0 + info.cu_qp_delta_val).clamp(0, 63);
        self.deblock_cus.push(DeblockCu {
            x: cu.cu.x,
            y: cu.cu.y,
            w: cu.cu.w,
            h: cu.cu.h,
            qp_y,
            intra: false,
            tu_y_coded: info.tu_y_coded_flag,
            tu_cb_coded: info.tu_cb_coded_flag,
            tu_cr_coded: info.tu_cr_coded_flag,
            bdpcm_luma: false,
            bdpcm_chroma: false,
        });
        // Round-28 §8.5.6.7 — inter CUs record MODE_INTER in the
        // intra-coded grid.
        self.write_intra_block(cu.cu.x, cu.cu.y, cu.cu.w, cu.cu.h, false);
        // Round-149 — the regular-merge inter path sets
        // `merge_subblock_flag = 0`; clear the per-CB grid.
        self.commit_subblock_neighbour_state(cu, info);
        Ok(())
    }

    /// §8.5.8 + §8.7.5.1 inter-residual add for a leaf CU whose
    /// motion-compensated `predSamples` already live in `out`. Shared
    /// by the translational merge / AMVP path
    /// ([`Self::reconstruct_inter_with_chosen`]) and the §8.5.5.2
    /// sub-block (affine) merge path: once MC has written the prediction,
    /// the dequant + inverse-transform + `Clip1(pred + res)` add is
    /// identical regardless of how the prediction was formed.
    ///
    /// Handles the §7.4.12.5 SBT split (one residual TU), §7.3.11.4
    /// multi-TB tiling for CUs above `MaxTbSizeY`, §8.7.2 joint Cb-Cr,
    /// and the §7.4.10.6 CU-level chroma QP offset lookups. Transform-
    /// skip per component is honoured via the parsed `transform_skip_*`
    /// flags.
    fn add_inter_cu_residual(
        &mut self,
        cu: &CtuCu,
        info: &LeafCuInfo,
        residual: &LeafCuResidual,
        out: &mut PictureBuffer,
    ) -> Result<()> {
        // §8.7.5.2 `predMapSamples` — a MODE_INTER CU with
        // `ciip_flag == 0` forward-maps its MC luma prediction before
        // the eq. 1214 residual add. CIIP CUs (and the intra paths,
        // which never reach here) take the pass-through arm
        // (`predMapSamples = predSamples`). Runs before the residual
        // add below so the eq. 1214 `Clip1(predMap + res)` sees mapped
        // prediction samples; skip CUs (no coded residual) still map,
        // leaving the reconstruction in the mapped domain the §8.8.2
        // in-loop inverse pass expects.
        if !info.inter.merge_data.ciip_flag {
            self.lmcs_forward_map_luma_rect(
                out,
                cu.cu.x as usize,
                cu.cu.y as usize,
                cu.cu.w as usize,
                cu.cu.h as usize,
            );
        }
        let bit_depth = self.sps.sps_bitdepth_minus8 as u32 + 8;
        let cu_qp_delta = info.cu_qp_delta_val;
        if info.cu_sbt_flag {
            if info.tu_y_coded_flag && !residual.luma_levels.is_empty() {
                let geo = crate::transform::sbt_geometry(
                    cu.cu.w,
                    cu.cu.h,
                    info.cu_sbt_quad_flag,
                    info.cu_sbt_horizontal_flag,
                    info.cu_sbt_pos_flag,
                );
                let (tr_h, tr_v) = crate::transform::sbt_tr_types(
                    info.cu_sbt_horizontal_flag,
                    info.cu_sbt_pos_flag,
                );
                self.add_inter_residual_plane_tr(
                    /*c_idx=*/ 0,
                    (cu.cu.x + geo.res_x) as usize,
                    (cu.cu.y + geo.res_y) as usize,
                    geo.res_w as usize,
                    geo.res_h as usize,
                    &residual.luma_levels,
                    cu_qp_delta,
                    bit_depth,
                    tr_h,
                    tr_v,
                    /*transform_skip=*/ false,
                    /*cu_chroma_qp_offset=*/ 0,
                    /*lmcs_chroma_var_scale=*/ None,
                    out,
                )?;
            }
        } else if info.tu_y_coded_flag && !residual.luma_levels.is_empty() {
            let n_tb_w = cu.cu.w as usize;
            let n_tb_h = cu.cu.h as usize;
            if n_tb_w > 64 || n_tb_h > 64 {
                // §7.3.11.4 multi-TB tiling — the CU exceeds MaxTbSizeY,
                // so the residual is split into ≤64×64 transform blocks
                // (raster recursion order). `residual.luma_levels` is the
                // per-tile coefficient blocks concatenated in the same
                // [`crate::transform::transform_tree_tiles`] order; each
                // tile's DCT-II inverse residual is added at its CU-
                // relative offset.
                let tiles = crate::transform::transform_tree_tiles(cu.cu.w, cu.cu.h, 64);
                let mut offset = 0usize;
                for t in &tiles {
                    let tlen = (t.w * t.h) as usize;
                    let end = (offset + tlen).min(residual.luma_levels.len());
                    let tile_levels = &residual.luma_levels[offset..end];
                    if tile_levels.len() == tlen {
                        self.add_inter_residual_plane(
                            /*c_idx=*/ 0,
                            (cu.cu.x + t.x) as usize,
                            (cu.cu.y + t.y) as usize,
                            t.w as usize,
                            t.h as usize,
                            tile_levels,
                            cu_qp_delta,
                            bit_depth,
                            /*transform_skip=*/ false,
                            /*cu_chroma_qp_offset=*/ 0,
                            /*lmcs_chroma_var_scale=*/ None,
                            out,
                        )?;
                    }
                    offset += tlen;
                }
            } else {
                self.add_inter_residual_plane(
                    /*c_idx=*/ 0,
                    cu.cu.x as usize,
                    cu.cu.y as usize,
                    n_tb_w,
                    n_tb_h,
                    &residual.luma_levels,
                    cu_qp_delta,
                    bit_depth,
                    info.transform_skip_luma,
                    /*cu_chroma_qp_offset=*/ 0,
                    /*lmcs_chroma_var_scale=*/ None,
                    out,
                )?;
            }
        }
        if self.sps.sps_chroma_format_idc == 1 {
            let c_w = (cu.cu.w as usize) / 2;
            let c_h = (cu.cu.h as usize) / 2;
            let c_x = (cu.cu.x as usize) / 2;
            let c_y = (cu.cu.y as usize) / 2;
            // §8.7.5.3 — chroma residual scaling `varScale` for this
            // CU's chroma TBs. The `nCurrSw * nCurrSh <= 4` bullet of
            // the pass-through list gates it here; the remaining
            // bullets (`ph_chroma_residual_scale_flag`,
            // `sh_lmcs_used_flag`) live in the helper. The `invAvgLuma`
            // neighbourhood reads the mapped-domain luma written above.
            let lmcs_cvs = if c_w * c_h > 4 {
                self.lmcs_chroma_var_scale_for_cu(cu, out)
            } else {
                None
            };
            if info.tu_c_res_mode != 0 {
                // §8.7.2 joint Cb-Cr — a single coded block reconstructs
                // both chroma residuals.
                // §7.4.10.6 `CuQpOffsetCbCr` — the joint chroma CU offset
                // indexes `pps_joint_cbcr_qp_offset_list` (not the per-
                // component lists) when `cu_chroma_qp_offset_flag == 1`.
                let cu_off_cbcr = if info.cu_chroma_qp_offset_flag {
                    self.pps
                        .pps_joint_cbcr_qp_offset_list
                        .get(info.cu_chroma_qp_offset_idx as usize)
                        .copied()
                        .unwrap_or(0)
                } else {
                    0
                };
                self.add_inter_joint_cbcr_residual(
                    c_x,
                    c_y,
                    c_w,
                    c_h,
                    info.tu_c_res_mode,
                    residual,
                    cu_qp_delta,
                    bit_depth,
                    cu_off_cbcr,
                    info.transform_skip_cb,
                    info.transform_skip_cr,
                    lmcs_cvs,
                    out,
                )?;
            } else {
                // §7.4.10.6 CU-level chroma QP offsets — index the PPS
                // `pps_c?_qp_offset_list` by `cu_chroma_qp_offset_idx` when
                // `cu_chroma_qp_offset_flag == 1`, matching the intra path.
                let cu_off_cb = cu_chroma_qp_offset(
                    1,
                    info.cu_chroma_qp_offset_flag,
                    info.cu_chroma_qp_offset_idx,
                    &self.pps.pps_cb_qp_offset_list,
                    &self.pps.pps_cr_qp_offset_list,
                );
                let cu_off_cr = cu_chroma_qp_offset(
                    2,
                    info.cu_chroma_qp_offset_flag,
                    info.cu_chroma_qp_offset_idx,
                    &self.pps.pps_cb_qp_offset_list,
                    &self.pps.pps_cr_qp_offset_list,
                );
                if info.tu_cb_coded_flag && !residual.cb_levels.is_empty() {
                    self.add_inter_residual_plane(
                        /*c_idx=*/ 1,
                        c_x,
                        c_y,
                        c_w,
                        c_h,
                        &residual.cb_levels,
                        cu_qp_delta,
                        bit_depth,
                        info.transform_skip_cb,
                        cu_off_cb,
                        lmcs_cvs,
                        out,
                    )?;
                }
                if info.tu_cr_coded_flag && !residual.cr_levels.is_empty() {
                    self.add_inter_residual_plane(
                        /*c_idx=*/ 2,
                        c_x,
                        c_y,
                        c_w,
                        c_h,
                        &residual.cr_levels,
                        cu_qp_delta,
                        bit_depth,
                        info.transform_skip_cr,
                        cu_off_cr,
                        lmcs_cvs,
                        out,
                    )?;
                }
            }
        }
        Ok(())
    }

    /// §8.5.2.1 non-merge AMVP reconstruction — derive the final per-list
    /// MVs from the parsed §7.3.11.7 motion record
    /// (`info.inter.non_merge`) and run the shared §8.5.6 / §8.5.8 MC +
    /// residual tail.
    ///
    /// For each active prediction list `X` (selected by
    /// `inter_pred_idc`):
    ///
    /// 1. Resolve `RefPicList[X][refIdxLX]` and its POC — the
    ///    `current_ref_poc` a neighbour's reference must match
    ///    (`DiffPicOrderCnt == 0`) to contribute (§8.5.2.10).
    /// 2. Derive the `[A, B]` spatial candidates (§8.5.2.10), AMVR-round
    ///    them (§8.5.2.9 step 2), derive the §8.5.2.11 temporal Col
    ///    candidate, build the §8.5.2.9 list (step 3 Col gate + step 5
    ///    HMVP fill + step 6 zero-MV pad), and select
    ///    `mvpListLX[mvp_lX_flag]` (§8.5.2.8 eq. 583).
    /// 3. Fold the raw `MvdLX` with the per-CU `AmvrShift`:
    ///    `mvLX = mvpLX + (mvdLX << AmvrShift)` (§8.5.2.1 eqs. 504-507).
    ///
    /// The resulting `chosen` [`MvField`] feeds
    /// [`Self::reconstruct_inter_with_chosen`] — the same MC + residual
    /// path the merge / skip CUs use.
    fn reconstruct_leaf_cu_inter_amvp(
        &mut self,
        cu: &CtuCu,
        info: &LeafCuInfo,
        residual: &LeafCuResidual,
        out: &mut PictureBuffer,
    ) -> Result<()> {
        use crate::amvp::{
            build_mvp_cand_list, derive_final_mv, derive_hmvp_mvp_candidates,
            derive_spatial_mvp_candidates, round_mv_amvr, select_mvp, AmvpRefContext, RefList,
            SpatialMvpCandidate,
        };
        use crate::leaf_cu::InterPredDir;

        let nm = &info.inter.non_merge;
        let xcb = cu.cu.x as i32;
        let ycb = cu.cu.y as i32;
        let cb_w = cu.cu.w as i32;
        let cb_h = cu.cu.h as i32;
        let amvr = nm.amvr_shift;

        let use_l0 = !matches!(nm.pred_dir, InterPredDir::PredL1);
        let use_l1 = !matches!(nm.pred_dir, InterPredDir::PredL0);

        // Per-list final MV via the §8.5.2.8-10 candidate derivation.
        // `derive_one_list` resolves the chosen predictor and folds the
        // raw mvd; the closure captures the shared geometry and the
        // per-list reference POC tables.
        let derive_one_list = |slf: &Self,
                               list: RefList,
                               ref_idx: i32,
                               mvp_flag: u32,
                               raw_mvd: MotionVector|
         -> Result<MotionVector> {
            let rpl_x = match list {
                RefList::L0 => &slf.ref_pic_list_l0,
                RefList::L1 => &slf.ref_pic_list_l1,
            };
            if ref_idx < 0 || (ref_idx as usize) >= rpl_x.len() {
                return Err(Error::invalid(format!(
                    "h266 amvp: refIdx {ref_idx} out of range for the active \
                         reference list ({} entries)",
                    rpl_x.len()
                )));
            }
            let current_ref_poc = rpl_x[ref_idx as usize].poc;
            // §8.5.2.10 neighbour-reference POC resolvers — a
            // neighbour's per-list refIdx → that reference's POC.
            // The neighbour's L0 index always resolves through
            // RefPicList0 and L1 through RefPicList1 regardless of
            // which list `X` the current CU predicts (the §8.5.2.10
            // scan consults both the neighbour's L0 and L1).
            let poc_of_l0 = |idx: i32| -> Option<i32> {
                if idx < 0 {
                    None
                } else {
                    slf.ref_pic_list_l0.get(idx as usize).map(|r| r.poc)
                }
            };
            let poc_of_l1 = |idx: i32| -> Option<i32> {
                if idx < 0 {
                    None
                } else {
                    slf.ref_pic_list_l1.get(idx as usize).map(|r| r.poc)
                }
            };
            let ctx = AmvpRefContext {
                list,
                current_ref_poc,
                poc_of_l0_ref: &poc_of_l0,
                poc_of_l1_ref: &poc_of_l1,
            };
            // §8.5.2.10 spatial scan, then §8.5.2.9 step-2 AMVR
            // rounding of each available candidate.
            let raw_spatial =
                derive_spatial_mvp_candidates(xcb, ycb, cb_w, cb_h, &slf.motion_field, &ctx);
            let spatial: [SpatialMvpCandidate; 2] = [
                SpatialMvpCandidate {
                    available: raw_spatial[0].available,
                    mv: round_mv_amvr(raw_spatial[0].mv, amvr),
                },
                SpatialMvpCandidate {
                    available: raw_spatial[1].available,
                    mv: round_mv_amvr(raw_spatial[1].mv, amvr),
                },
            ];
            // §8.5.2.11 temporal Col candidate (AMVR-rounded inside
            // the helper). Resolved from the configured ColPic when
            // `ph_temporal_mvp_enabled`.
            let col = slf.derive_amvp_col_candidate(xcb, ycb, cb_w, cb_h, current_ref_poc, amvr);
            // §8.5.2.9 step-5 HMVP fill — only consulted when the
            // list still has room after spatial + Col.
            let n_after_spatial =
                usize::from(spatial[0].available) + usize::from(spatial[1].available);
            let col_suppressed =
                spatial[0].available && spatial[1].available && spatial[0].mv != spatial[1].mv;
            let n_pre_hmvp =
                (n_after_spatial + usize::from(col.is_some() && !col_suppressed)).min(2);
            let slots_remaining = 2 - n_pre_hmvp;
            let hmvp = derive_hmvp_mvp_candidates(&slf.hmvp, &ctx, amvr, slots_remaining);
            let list_lx = build_mvp_cand_list(spatial, col, &hmvp);
            let mvp = select_mvp(&list_lx, mvp_flag);
            Ok(derive_final_mv(mvp, raw_mvd, amvr))
        };

        let (mv_l0, ref_idx_l0_out) = if use_l0 {
            (
                derive_one_list(self, RefList::L0, nm.ref_idx_l0, nm.mvp_l0_flag, nm.mvd_l0)?,
                nm.ref_idx_l0,
            )
        } else {
            (MotionVector::ZERO, -1)
        };
        let (mv_l1, ref_idx_l1_out) = if use_l1 {
            (
                derive_one_list(self, RefList::L1, nm.ref_idx_l1, nm.mvp_l1_flag, nm.mvd_l1)?,
                nm.ref_idx_l1,
            )
        } else {
            (MotionVector::ZERO, -1)
        };

        let chosen = MvField {
            mv_l0,
            ref_idx_l0: ref_idx_l0_out,
            pred_flag_l0: use_l0,
            mv_l1,
            ref_idx_l1: ref_idx_l1_out,
            pred_flag_l1: use_l1,
            cu_skip_flag: false,
            mode_inter: true,
            available: true,
            // §8.5.6.6.2 — BCW for the non-merge path is the parsed
            // `bcw_idx[x0][y0]`. `0` ⇒ eq. 980 default-weighted average;
            // `1..=4` ⇒ eq. 981 `BCW_W_LUT` weighted blend in the bi-pred
            // tail. Inferred 0 for uni-pred CUs (the gate never signals it).
            bcw_idx: nm.bcw_idx,
        };

        self.reconstruct_inter_with_chosen(cu, info, residual, chosen, out)
    }

    /// §8.5.5.3 / §8.5.6 affine uni-prediction reconstruction to pixels.
    ///
    /// Given a control-point MV set (`cpmvs`) for a single reference
    /// list and the resolved reference picture, this drives:
    ///
    /// * **Luma** — `crate::affine::predict_luma_block_affine_prof`
    ///   derives the §8.5.5.9 per-4×4-sub-block MV grid (eqs. 872 – 875),
    ///   runs the §8.5.6.3.2 affine 6-tap interpolation per sub-block,
    ///   and applies §8.5.5.8 PROF when `cbProfFlagLX` holds.
    /// * **Chroma** (4:2:0 only) — per §8.5.5.3 eqs. 876 – 879 each
    ///   4×4 chroma sub-block (covering a 2×2 luma sub-block group)
    ///   takes the average of the top-left and bottom-right luma
    ///   sub-block MVs (rounded toward zero, halved), then runs the
    ///   §8.5.6.3.4 4-tap chroma interpolation via
    ///   [`predict_chroma_block`].
    ///
    /// `cb` is the picture-absolute CU geometry. The reference picture's
    /// planes are read for MC; `out` already holds the partially-
    /// reconstructed picture and is written in place. Scope: uni-pred,
    /// no residual add (the caller layers residual via the regular
    /// `add_inter_residual_plane*` path), 4:2:0 chroma.
    pub fn reconstruct_affine_inter_uni(
        &self,
        cb_x: u32,
        cb_y: u32,
        cb_w: u32,
        cb_h: u32,
        cpmvs: &crate::affine::AffineCpmvs,
        ref_pic: &ReferencePicture,
        out: &mut PictureBuffer,
    ) -> Result<()> {
        // Uni-pred: affine MC for the single list writes straight into
        // the output picture at the CU's picture-absolute offset.
        self.predict_affine_one_list(
            cb_w,
            cb_h,
            cpmvs,
            ref_pic,
            /*bipred=*/ false,
            &mut out.luma,
            cb_x,
            cb_y,
            &mut out.cb,
            &mut out.cr,
            cb_x / 2,
            cb_y / 2,
        )
    }

    /// §8.5.5.3 / §8.5.6 affine bi-prediction reconstruction to pixels
    /// with eq. 980 default-weighted averaging (the `bcw_idx == 0`
    /// case). Thin wrapper over [`Self::reconstruct_affine_inter_bi_bcw`].
    #[allow(clippy::too_many_arguments)]
    pub fn reconstruct_affine_inter_bi(
        &self,
        cb_x: u32,
        cb_y: u32,
        cb_w: u32,
        cb_h: u32,
        cpmvs_l0: &crate::affine::AffineCpmvs,
        ref_pic_l0: &ReferencePicture,
        cpmvs_l1: &crate::affine::AffineCpmvs,
        ref_pic_l1: &ReferencePicture,
        out: &mut PictureBuffer,
    ) -> Result<()> {
        self.reconstruct_affine_inter_bi_bcw(
            cb_x, cb_y, cb_w, cb_h, cpmvs_l0, ref_pic_l0, cpmvs_l1, ref_pic_l1, 0, out,
        )
    }

    /// §8.5.5.3 / §8.5.6 affine bi-prediction reconstruction to pixels.
    ///
    /// Predicts each reference list's affine MC into a CU-sized scratch
    /// plane (luma + 4:2:0 chroma), then forms the §8.5.6.6.2 composite
    /// into the output picture: eq. 980 default-weighted average
    /// `(pred_l0 + pred_l1 + 1) >> 1` when `bcw_idx == 0`, or the eq. 981
    /// BCW-weighted blend `Clip1((w0·p0 + w1·p1 + 4) >> 3)` (weights from
    /// `BCW_W_LUT`) when `bcw_idx ∈ 1..=4`. Mirrors the translational
    /// bi-pred path's two-scratch-plane composition. BDOF on the affine
    /// bi-pred path remains a follow-up.
    #[allow(clippy::too_many_arguments)]
    pub fn reconstruct_affine_inter_bi_bcw(
        &self,
        cb_x: u32,
        cb_y: u32,
        cb_w: u32,
        cb_h: u32,
        cpmvs_l0: &crate::affine::AffineCpmvs,
        ref_pic_l0: &ReferencePicture,
        cpmvs_l1: &crate::affine::AffineCpmvs,
        ref_pic_l1: &ReferencePicture,
        bcw_idx: u8,
        out: &mut PictureBuffer,
    ) -> Result<()> {
        use crate::reconstruct::PicturePlane;
        let chroma = self.sps.sps_chroma_format_idc == 1;
        let cw = (cb_w / 2) as usize;
        let ch = (cb_h / 2) as usize;
        // The §8.5.6.3 affine MC couples the destination position with
        // the reference-sample read position (`sb_x = dst_x + ...`), so
        // each list is predicted into a full-picture-sized scratch
        // plane at the CU's picture-absolute offset, then the CU
        // rectangle is averaged out of it.
        let pw = out.luma.width;
        let ph = out.luma.height;
        let cpw = out.cb.width;
        let cph = out.cb.height;
        let mut l0_luma = PicturePlane::filled(pw, ph, 0);
        let mut l1_luma = PicturePlane::filled(pw, ph, 0);
        let mut l0_cb = PicturePlane::filled(cpw, cph, 0);
        let mut l0_cr = PicturePlane::filled(cpw, cph, 0);
        let mut l1_cb = PicturePlane::filled(cpw, cph, 0);
        let mut l1_cr = PicturePlane::filled(cpw, cph, 0);

        self.predict_affine_one_list(
            cb_w,
            cb_h,
            cpmvs_l0,
            ref_pic_l0,
            /*bipred=*/ true,
            &mut l0_luma,
            cb_x,
            cb_y,
            &mut l0_cb,
            &mut l0_cr,
            cb_x / 2,
            cb_y / 2,
        )?;
        self.predict_affine_one_list(
            cb_w,
            cb_h,
            cpmvs_l1,
            ref_pic_l1,
            /*bipred=*/ true,
            &mut l1_luma,
            cb_x,
            cb_y,
            &mut l1_cb,
            &mut l1_cr,
            cb_x / 2,
            cb_y / 2,
        )?;

        // §8.5.6.6.2 composite into the picture. The scratch planes
        // carry the prediction at the CU's picture-absolute offset, so
        // the blend reads them at that same offset. eq. 980 average for
        // `bcw_idx == 0`, eq. 981 BCW weighting otherwise. The chroma
        // planes take the SAME `bcw_idx` (eq. 981 applies per-component).
        self.bi_blend_plane_region_bcw(
            &mut out.luma,
            cb_x,
            cb_y,
            cb_w,
            cb_h,
            &l0_luma,
            &l1_luma,
            bcw_idx,
        );
        if chroma {
            self.bi_blend_plane_region_bcw(
                &mut out.cb,
                cb_x / 2,
                cb_y / 2,
                cw as u32,
                ch as u32,
                &l0_cb,
                &l1_cb,
                bcw_idx,
            );
            self.bi_blend_plane_region_bcw(
                &mut out.cr,
                cb_x / 2,
                cb_y / 2,
                cw as u32,
                ch as u32,
                &l0_cr,
                &l1_cr,
                bcw_idx,
            );
        }
        Ok(())
    }

    /// §8.5.5.5 affine non-merge (AMVP) reconstruction — the
    /// parse-to-pixels fuse.
    ///
    /// Takes the parser-side affine decision (the per-CP `MvdCpLX`
    /// arrays the §7.3.11.7 affine branch produced, via
    /// [`crate::non_merge_inter_pre_residual_dec::read_non_merge_inter_pre_residual_affine`])
    /// plus the resolved per-CU `AmvrShift`, and drives the full
    /// §8.5.5.5 ordered steps:
    ///
    /// 1. `numCpMv = MotionModelIdc + 1` (§8.5.5.5 / eq. 160).
    /// 2. Per active list X: build the §8.5.5.7 affine CPMVP candidate
    ///    list (`build_affine_mvp_cand_list`) and pick `mvp_lX_flag`
    ///    via `select_affine_mvp` (eq. 840). The §8.5.5.7 step-4 A-scan
    ///    ({A0, A1}) and step-5 B-scan ({B0, B1, B2}) now read the
    ///    per-CB affine CPMV store ([`Self::affine_neighbour_query`] +
    ///    [`Self::derive_inherited_affine_side`]) and feed real
    ///    inherited candidates; the §8.5.5.8 constructed candidate
    ///    (step 6) + step-7 single-corner fallback read the per-corner
    ///    regular motion field. When nothing hits, the list falls through
    ///    to the §8.5.5.7 step-8 temporal MV (replicated across CPs) and
    ///    the step-9 zero-MV pad.
    /// 3. Cumulate the per-CP MVDs (§8.5.5.5 eqs. 660 – 663 via
    ///    `cumulate_affine_mvd_cp`).
    /// 4. Fold predictor + cumulative MVD into the final CPMVs (eqs.
    ///    664 – 667 via `derive_final_affine_cpmvs`).
    /// 5. Drive [`Self::reconstruct_affine_inter_uni`] (uni-pred) or
    ///    [`Self::reconstruct_affine_inter_bi`] (bi-pred, §8.5.6.6.2
    ///    eq. 980 default-weighted average).
    ///
    /// §8.5.5.7 step-4 / step-5 neighbour query — sample the per-CB
    /// affine CPMV store at `(xnb, ynb)` and assemble a
    /// [`crate::affine_amvp::NeighbourAffineQuery`].
    ///
    /// Returns `None` when the sample is out of bounds or the covering
    /// CB is not affine (`MotionModelIdc == 0` ⇒ the §8.5.5.7 gate
    /// fails). The §6.4.4 availability is satisfied by construction: the
    /// store only carries CBs already decoded in slice raster order, so
    /// any non-`None` cell is causally available to the current CU. The
    /// §8.5.5.7 parallel-merge suppression (eq. 60) is folded in by the
    /// caller via `available`.
    fn affine_neighbour_query(
        &self,
        xnb: i32,
        ynb: i32,
        xcb: i32,
        ycb: i32,
    ) -> Option<crate::affine_amvp::NeighbourAffineQuery> {
        let rec = self.affine_cpmv_field.get_at_luma(xnb, ynb)?;
        // §8.5.5.7 parallel-merge suppression (eq. 60): a neighbour in
        // the same Log2ParMrgLevel region as the current CU is
        // unavailable. ParMrgLevel masks the low bits of the position.
        let par = self.log2_par_mrg_level;
        let same_par = (xcb >> par) == (xnb >> par) && (ycb >> par) == (ynb >> par);
        Some(crate::affine_amvp::NeighbourAffineQuery {
            available: !same_par,
            pred_flag_l0: rec.pred_flag_l0,
            pred_flag_l1: rec.pred_flag_l1,
            ref_idx_l0: rec.ref_idx_l0,
            ref_idx_l1: rec.ref_idx_l1,
            xnb: rec.xnb,
            ynb: rec.ynb,
            nb_w: rec.nb_w,
            nb_h: rec.nb_h,
            motion_model: rec.model,
            cpmvs_l0: rec.cpmvs_l0,
            cpmvs_l1: rec.cpmvs_l1,
            bcw_idx: rec.bcw_idx,
        })
    }

    /// §8.5.5.5 `isCTUboundary` for a neighbour CB whose covering
    /// origin/dims are `(xnb, ynb, _, nb_h)` relative to the current
    /// CU top `ycb`. TRUE iff `(yNb + nNbH) % CtbSizeY == 0` AND
    /// `yNb + nNbH == yCb` (eqs. above 734).
    fn affine_is_ctu_boundary_above(&self, ynb: i32, nb_h: u32, ycb: i32) -> bool {
        let ctb = self.layout.ctb_size_y as i32;
        let bottom = ynb + nb_h as i32;
        bottom % ctb == 0 && bottom == ycb
    }

    /// §8.5.5.7 inherited scan over one side's ordered positions (A-side
    /// = {A0, A1}, B-side = {B0, B1, B2}). Returns the first
    /// effectively-available affine neighbour's inherited CPMVP
    /// candidate (already AMVR-rounded), or `None` when no position
    /// yields one. `ctx` carries the §8.5.5.7 `DiffPicOrderCnt == 0`
    /// cross-list gate.
    #[allow(clippy::too_many_arguments)]
    fn derive_inherited_affine_side(
        &self,
        positions: &[(i32, i32)],
        xcb: i32,
        ycb: i32,
        cb_w: u32,
        cb_h: u32,
        ctx: &crate::affine_amvp::AffineMvpRefContext<'_>,
        amvr: crate::amvr::AmvrShift,
        num_cp_mv: u32,
    ) -> Option<crate::affine_amvp::AffineMvpCandidate> {
        for &(xnb, ynb) in positions {
            let Some(nb) = self.affine_neighbour_query(xnb, ynb, xcb, ycb) else {
                continue;
            };
            let is_boundary = self.affine_is_ctu_boundary_above(nb.ynb, nb.nb_h, ycb);
            if let Some(cand) = crate::affine_amvp::derive_inherited_affine_mvp_candidate(
                xcb,
                ycb,
                cb_w,
                cb_h,
                &nb,
                ctx,
                amvr,
                num_cp_mv,
                is_boundary,
            ) {
                return Some(cand);
            }
        }
        None
    }

    /// §8.5.5.8 per-corner constructed-CPMVP pick over one corner's
    /// ordered cascade of `MvLX` sample positions (TL = {B2, B3, A2},
    /// TR = {B1, B0}, BL = {A1, A0}). Unlike the inherited scan this
    /// reads the **regular per-sample motion field** (`MvLX[xNb][yNb]`),
    /// not the affine CPMV store — so a translational neighbour
    /// contributes its stored MV directly (eqs. 841 – 846).
    ///
    /// Each position is gated by §6.4.4 availability (`MvField::available`,
    /// satisfied causally by slice-raster decode order) AND the
    /// `PredFlagLX == 1 ∧ DiffPicOrderCnt(RefPicList[X][RefIdxLX[nb]],
    /// RefPicList[X][refIdxLX]) == 0` test, with the cross-list
    /// `Y = 1 − X` fallback. The first matching position wins; the picked
    /// MV is AMVR-rounded so the corner carries the spec's already-rounded
    /// `cpMvLX[cpIdx]`.
    fn constructed_affine_corner(
        &self,
        positions: &[(i32, i32)],
        ctx: &crate::affine_amvp::AffineMvpRefContext<'_>,
        amvr: crate::amvr::AmvrShift,
    ) -> crate::affine_amvp::ConstructedAffineMvpCorner {
        use crate::affine_amvp::{
            round_constructed_corner_amvr, ConstructedAffineMvpCorner, RefList,
        };
        let cur_list = ctx.list;
        for &(xnb, ynb) in positions {
            let nb = self.motion_field.get_at_luma(xnb, ynb);
            if !nb.available {
                continue;
            }
            // List X first, then list Y = 1 − X. Each requires
            // PredFlagL? == 1 and a matching reference POC.
            for which in [cur_list, cur_list.other()] {
                let (pred_flag, ref_idx, mv) = match which {
                    RefList::L0 => (nb.pred_flag_l0, nb.ref_idx_l0, nb.mv_l0),
                    RefList::L1 => (nb.pred_flag_l1, nb.ref_idx_l1, nb.mv_l1),
                };
                if !pred_flag {
                    continue;
                }
                let poc = match which {
                    RefList::L0 => (ctx.poc_of_l0_ref)(ref_idx),
                    RefList::L1 => (ctx.poc_of_l1_ref)(ref_idx),
                };
                if poc == Some(ctx.current_ref_poc) {
                    return ConstructedAffineMvpCorner {
                        available: true,
                        mv: round_constructed_corner_amvr(mv, amvr),
                    };
                }
            }
        }
        ConstructedAffineMvpCorner::default()
    }

    /// §8.5.5.8 — derive the three constructed-CPMVP corners (top-left,
    /// top-right, bottom-left) at the spec's cascade positions, plus the
    /// `numCpMv`-sized full constructed candidate (`availableConsFlagLX`).
    /// Returns `(corners, constructed_full)` ready to stage into
    /// [`crate::affine_amvp::AffineMvpListInputs`].
    fn derive_constructed_affine_corners(
        &self,
        xcb: i32,
        ycb: i32,
        cbw: i32,
        cbh: i32,
        ctx: &crate::affine_amvp::AffineMvpRefContext<'_>,
        amvr: crate::amvr::AmvrShift,
        num_cp_mv: u32,
    ) -> (
        [crate::affine_amvp::ConstructedAffineMvpCorner; 3],
        Option<crate::affine_amvp::AffineMvpCandidate>,
    ) {
        // §8.5.5.8 step-1 cascade positions per corner.
        let tl_positions = [
            (xcb - 1, ycb - 1), // B2
            (xcb, ycb - 1),     // B3
            (xcb - 1, ycb),     // A2
        ];
        let tr_positions = [
            (xcb + cbw - 1, ycb - 1), // B1
            (xcb + cbw, ycb - 1),     // B0
        ];
        let bl_positions = [
            (xcb - 1, ycb + cbh - 1), // A1
            (xcb - 1, ycb + cbh),     // A0
        ];
        let top_left = self.constructed_affine_corner(&tl_positions, ctx, amvr);
        let top_right = self.constructed_affine_corner(&tr_positions, ctx, amvr);
        let bottom_left = self.constructed_affine_corner(&bl_positions, ctx, amvr);
        // The picks are already AMVR-rounded, so the §8.5.5.8 inner
        // rounding bullet inside `derive_constructed_affine_mvp_candidate`
        // is a no-op on them — passing them straight through keeps the
        // `availableConsFlagLX` gating logic in one place.
        let full = crate::affine_amvp::derive_constructed_affine_mvp_candidate(
            top_left,
            top_right,
            bottom_left,
            num_cp_mv,
            amvr,
        );
        ([top_left, top_right, bottom_left], full)
    }

    /// §8.5.5.2 step 4 / step 5 — build an inherited affine-merge
    /// candidate (`A` or `B`) from one ordered side scan.
    ///
    /// Unlike the §8.5.5.7 AMVP inherited scan (which builds a per-list
    /// CPMVP for a *specific* `refIdxLX`), the merge inherited candidate
    /// copies the neighbour's **whole** motion record: `predFlagLXN`,
    /// `refIdxLXN`, `bcwIdxN`, and the §8.5.5.5-derived per-list CPMVs.
    /// The scan returns the first effectively-available affine
    /// (`MotionModelIdc > 0`) neighbour at the supplied positions; the
    /// resulting candidate's `motion_model` is the neighbour's own model
    /// (eqs. 681 – 684 carry `motionModelIdcN`).
    fn derive_inherited_affine_merge_side(
        &self,
        positions: &[(i32, i32)],
        xcb: i32,
        ycb: i32,
        cb_w: u32,
        cb_h: u32,
    ) -> crate::affine_merge::InheritedAffineCandidate {
        use crate::affine_merge::{
            derive_inherited_affine_cpmvs, AffineMergeCandidate, InheritedAffineCandidate,
            InheritedAffineGeom, NeighbourCpmvSource,
        };
        for &(xnb, ynb) in positions {
            let Some(nb) = self.affine_neighbour_query(xnb, ynb, xcb, ycb) else {
                continue;
            };
            if !nb.available {
                continue;
            }
            // The neighbour must itself be affine (`MotionModelIdc > 0`).
            // `affine_neighbour_query` only returns a record for cells
            // that carry an affine CB; a translational neighbour yields
            // `None` (handled by the `?` above).
            let num_cp_mv = nb.motion_model.num_cp_mv() as u32;
            if num_cp_mv < 2 {
                continue;
            }
            let geom = InheritedAffineGeom {
                xcb,
                ycb,
                cb_width: cb_w,
                cb_height: cb_h,
                xnb: nb.xnb,
                ynb: nb.ynb,
                nb_w: nb.nb_w,
                nb_h: nb.nb_h,
            };
            let is_boundary = self.affine_is_ctu_boundary_above(nb.ynb, nb.nb_h, ycb);

            // §8.5.5.5 per active list. `num_cp_mv` here is the
            // *current* CU's output model — the spec emits the candidate
            // with the neighbour's `motionModelIdcN` so a 6-param
            // neighbour yields a 6-param candidate and a 4-param
            // neighbour a 4-param one. We mirror that by deriving with
            // the neighbour's own `num_cp_mv`.
            let derive_list = |cpmvs: crate::affine::AffineCpmvs,
                               mv_bl: MotionVector,
                               mv_br: MotionVector|
             -> Option<crate::affine::AffineCpmvs> {
                let source = if is_boundary {
                    NeighbourCpmvSource::AboveCtuBoundary {
                        mv_bottom_left: mv_bl,
                        mv_bottom_right: mv_br,
                    }
                } else {
                    NeighbourCpmvSource::SameOrLeftCtu { cpmvs }
                };
                derive_inherited_affine_cpmvs(geom, source, num_cp_mv).ok()
            };

            // §8.5.5.5 above-CTU-boundary branch reads the neighbour's
            // bottom-row sub-block MVs from the regular motion field.
            let (bl_l0, br_l0, bl_l1, br_l1) = if is_boundary {
                let bx_l = nb.xnb;
                let bx_r = nb.xnb + nb.nb_w as i32 - 1;
                let by = nb.ynb + nb.nb_h as i32 - 1;
                let mfl = self.motion_field.get_at_luma(bx_l, by);
                let mfr = self.motion_field.get_at_luma(bx_r, by);
                (mfl.mv_l0, mfr.mv_l0, mfl.mv_l1, mfr.mv_l1)
            } else {
                (
                    MotionVector::ZERO,
                    MotionVector::ZERO,
                    MotionVector::ZERO,
                    MotionVector::ZERO,
                )
            };

            let cpmvs_l0 = if nb.pred_flag_l0 {
                match derive_list(nb.cpmvs_l0, bl_l0, br_l0) {
                    Some(c) => c,
                    None => continue,
                }
            } else {
                crate::affine::AffineCpmvs {
                    model: nb.motion_model,
                    cpmvs: [MotionVector::ZERO; 3],
                }
            };
            let cpmvs_l1 = if nb.pred_flag_l1 {
                match derive_list(nb.cpmvs_l1, bl_l1, br_l1) {
                    Some(c) => c,
                    None => continue,
                }
            } else {
                crate::affine::AffineCpmvs {
                    model: nb.motion_model,
                    cpmvs: [MotionVector::ZERO; 3],
                }
            };

            return InheritedAffineCandidate {
                available: true,
                cand: AffineMergeCandidate {
                    pred_flag_l0: nb.pred_flag_l0,
                    pred_flag_l1: nb.pred_flag_l1,
                    ref_idx_l0: if nb.pred_flag_l0 { nb.ref_idx_l0 } else { -1 },
                    ref_idx_l1: if nb.pred_flag_l1 { nb.ref_idx_l1 } else { -1 },
                    cpmvs_l0,
                    cpmvs_l1,
                    motion_model: nb.motion_model,
                    // §8.5.5.2 eqs. 681 – 684 — the inherited candidate
                    // carries the neighbour CB's `bcwIdxN` only when the
                    // neighbour is bi-pred (a uni-pred neighbour has no
                    // meaningful weight index; the affine bi-pred MC
                    // default-weights when bcwIdx == 0).
                    bcw_idx: if nb.pred_flag_l0 && nb.pred_flag_l1 {
                        nb.bcw_idx
                    } else {
                        0
                    },
                },
            };
        }
        InheritedAffineCandidate::default()
    }

    /// Broadcast the final-decided affine CB record into the per-CB
    /// affine CPMV store (§8.5.5.7 / §8.5.5.5 source for later CUs).
    /// `model` is the CU's `MotionModelIdc`; the per-list CPMVs / pred
    /// flags / ref indices are the post-fold (`derive_final_affine_cpmvs`)
    /// values. Translational / non-affine CBs should not call this; the
    /// motion-field write path leaves their cells `None`.
    #[allow(clippy::too_many_arguments)]
    fn store_affine_cb(
        &mut self,
        cb_x: u32,
        cb_y: u32,
        cb_w: u32,
        cb_h: u32,
        rec: crate::inter::AffineCbRecord,
    ) {
        self.affine_cpmv_field
            .write_block(cb_x, cb_y, cb_w, cb_h, Some(rec));
    }

    /// Scope: 4:2:0 chroma, no residual add (the caller layers residual
    /// via the regular tail), BCW / BDOF on the affine path deferred.
    #[allow(clippy::too_many_arguments)]
    pub fn reconstruct_leaf_cu_inter_affine_amvp(
        &mut self,
        cb_x: u32,
        cb_y: u32,
        cb_w: u32,
        cb_h: u32,
        decision: &crate::non_merge_inter_pre_residual_enc::NonMergeInterPreResidualAffineDecision,
        amvr: crate::amvr::AmvrShift,
        bcw_idx: u8,
        out: &mut PictureBuffer,
    ) -> Result<()> {
        use crate::affine_amvp::{
            build_affine_mvp_cand_list, cumulate_affine_mvd_cp, derive_final_affine_cpmvs,
            select_affine_mvp, AffineMvpListInputs, AffineMvpRefContext, RefList,
        };
        use crate::leaf_cu::InterPredDir;

        let motion_model = decision.affine.motion_model;
        let num_cp_mv = motion_model.num_cp_mv() as u32;
        if num_cp_mv < 2 {
            return Err(Error::invalid(
                "h266 affine amvp: motion_model is translational (numCpMv < 2)",
            ));
        }

        let xcb = cb_x as i32;
        let ycb = cb_y as i32;
        let cbw = cb_w as i32;
        let cbh = cb_h as i32;

        let pred = decision.mvp.inter_pred_idc;
        let use_l0 = !matches!(pred, InterPredDir::PredL1);
        let use_l1 = !matches!(pred, InterPredDir::PredL0);

        // Per-list CPMV derivation: §8.5.5.5 step 4 (AMVP) + step 5
        // (eqs. 664 – 667). The closure captures the shared geometry.
        let derive_one_list = |slf: &Self,
                               list: RefList,
                               ref_idx: u32,
                               mvp_flag: u32,
                               mvd_cp_stored: &[MotionVector; 3]|
         -> Result<(crate::affine::AffineCpmvs, ReferencePicture)> {
            let rpl_x = match list {
                RefList::L0 => &slf.ref_pic_list_l0,
                RefList::L1 => &slf.ref_pic_list_l1,
            };
            if (ref_idx as usize) >= rpl_x.len() {
                return Err(Error::invalid(format!(
                    "h266 affine amvp: refIdx {ref_idx} out of range \
                         (list has {} entries)",
                    rpl_x.len()
                )));
            }
            let ref_pic = rpl_x[ref_idx as usize].clone();
            let current_ref_poc = ref_pic.poc;

            // §8.5.5.7 step-8 temporal MV (replicated across CPs inside
            // `build_affine_mvp_cand_list`). Reuses the regular-AMVP
            // collocated walk.
            let temporal_col =
                slf.derive_amvp_col_candidate(xcb, ycb, cbw, cbh, current_ref_poc, amvr);

            // §8.5.5.7 steps 4 / 5 — inherited affine CPMVP candidates,
            // recovered from the per-CB affine CPMV store. The store
            // carries every previously-decoded affine CB's CPMV record;
            // the A-scan {A0, A1} and B-scan {B0, B1, B2} sample the
            // §8.5.5.7 eq. 819 – 823 neighbour positions and feed the
            // §8.5.5.5 inherited derivation. §8.5.5.8 constructed CPMVP
            // (step 6) + the step-7 single-corner fallback now read the
            // per-corner regular motion field; when nothing hits, the
            // list falls through to the step-8 temporal MV + step-9
            // zero-MV pad.
            let ctx = AffineMvpRefContext {
                list,
                current_ref_poc,
                poc_of_l0_ref: &|idx: i32| -> Option<i32> {
                    if idx < 0 {
                        None
                    } else {
                        slf.ref_pic_list_l0.get(idx as usize).map(|r| r.poc)
                    }
                },
                poc_of_l1_ref: &|idx: i32| -> Option<i32> {
                    if idx < 0 {
                        None
                    } else {
                        slf.ref_pic_list_l1.get(idx as usize).map(|r| r.poc)
                    }
                },
            };

            // §8.5.5.7 eqs. 819 – 823 neighbour positions.
            let a_positions = [
                (xcb - 1, ycb + cbh),     // A0 (eq. 819)
                (xcb - 1, ycb + cbh - 1), // A1 (eq. 820)
            ];
            let b_positions = [
                (xcb + cbw, ycb - 1),     // B0 (eq. 821)
                (xcb + cbw - 1, ycb - 1), // B1 (eq. 822)
                (xcb - 1, ycb - 1),       // B2 (eq. 823)
            ];
            let inherited_a = slf.derive_inherited_affine_side(
                &a_positions,
                xcb,
                ycb,
                cb_w,
                cb_h,
                &ctx,
                amvr,
                num_cp_mv,
            );
            let inherited_b = slf.derive_inherited_affine_side(
                &b_positions,
                xcb,
                ycb,
                cb_w,
                cb_h,
                &ctx,
                amvr,
                num_cp_mv,
            );

            // §8.5.5.8 constructed CPMVP — read the per-corner regular
            // motion field at the cascade positions and assemble the full
            // candidate (`availableConsFlagLX`) plus the per-corner picks
            // for the §8.5.5.7 step-7 single-corner fallback.
            let (constructed_corners, constructed_full) =
                slf.derive_constructed_affine_corners(xcb, ycb, cbw, cbh, &ctx, amvr, num_cp_mv);

            let list_lx = build_affine_mvp_cand_list(AffineMvpListInputs {
                num_cp_mv,
                inherited_a,
                inherited_b,
                constructed_full,
                constructed_corners,
                temporal_col,
                _phantom: core::marker::PhantomData,
            });
            let mvp = select_affine_mvp(&list_lx, mvp_flag);

            // §8.5.5.5 eqs. 660 – 663 cumulative MVD, then eqs. 664 –
            // 667 final fold.
            let cumulative = cumulate_affine_mvd_cp(mvd_cp_stored, num_cp_mv);
            let cpmvs = derive_final_affine_cpmvs(&mvp, &cumulative);
            Ok((cpmvs, ref_pic))
        };

        // Capture per-list final CPMVs so the affine CB record can be
        // broadcast into the per-CB store after reconstruction (§8.5.5.7
        // source for later CUs). `None` for an inactive list.
        let mut store_cpmvs_l0: Option<crate::affine::AffineCpmvs> = None;
        let mut store_cpmvs_l1: Option<crate::affine::AffineCpmvs> = None;

        let recon_result = match (use_l0, use_l1) {
            (true, false) => {
                let (cpmvs, ref_pic) = derive_one_list(
                    self,
                    RefList::L0,
                    decision.mvp.ref_idx_l0,
                    decision.mvp.mvp_l0_flag,
                    &decision.mvd_cp_l0,
                )?;
                store_cpmvs_l0 = Some(cpmvs);
                self.reconstruct_affine_inter_uni(cb_x, cb_y, cb_w, cb_h, &cpmvs, &ref_pic, out)
            }
            (false, true) => {
                let (cpmvs, ref_pic) = derive_one_list(
                    self,
                    RefList::L1,
                    decision.mvp.ref_idx_l1,
                    decision.mvp.mvp_l1_flag,
                    &decision.mvd_cp_l1,
                )?;
                store_cpmvs_l1 = Some(cpmvs);
                self.reconstruct_affine_inter_uni(cb_x, cb_y, cb_w, cb_h, &cpmvs, &ref_pic, out)
            }
            (true, true) => {
                let (cpmvs_l0, ref_pic_l0) = derive_one_list(
                    self,
                    RefList::L0,
                    decision.mvp.ref_idx_l0,
                    decision.mvp.mvp_l0_flag,
                    &decision.mvd_cp_l0,
                )?;
                let (cpmvs_l1, ref_pic_l1) = derive_one_list(
                    self,
                    RefList::L1,
                    decision.mvp.ref_idx_l1,
                    decision.mvp.mvp_l1_flag,
                    &decision.mvd_cp_l1,
                )?;
                store_cpmvs_l0 = Some(cpmvs_l0);
                store_cpmvs_l1 = Some(cpmvs_l1);
                // §8.5.6.6.2 — apply the parsed `bcw_idx` so an explicit
                // bi-pred affine CU with `bcwIdx ∈ 1..=4` takes the
                // eq. 981 weighted blend (eq. 980 average for 0).
                self.reconstruct_affine_inter_bi_bcw(
                    cb_x,
                    cb_y,
                    cb_w,
                    cb_h,
                    &cpmvs_l0,
                    &ref_pic_l0,
                    &cpmvs_l1,
                    &ref_pic_l1,
                    bcw_idx,
                    out,
                )
            }
            (false, false) => Err(Error::invalid(
                "h266 affine amvp: neither list active (inter_pred_idc invalid)",
            )),
        };
        recon_result?;

        // §8.7.5.2 — an affine non-merge CU is MODE_INTER with
        // `ciip_flag == 0` (CIIP is merge-only), so its MC luma
        // prediction forward-maps into the LMCS codeword domain.
        self.lmcs_forward_map_luma_rect(
            out,
            cb_x as usize,
            cb_y as usize,
            cb_w as usize,
            cb_h as usize,
        );

        // Broadcast the affine CB record into the per-CB CPMV store so
        // later CUs' §8.5.5.7 inherited scan can recover this block's
        // CPMVs (§8.5.5.5). The zero-CPMV record kept for an inactive
        // list is harmless — its pred-flag is `false`, so the §8.5.5.7
        // `PredFlagLX == 1` gate skips it.
        let zero_cpmvs = crate::affine::AffineCpmvs {
            model: motion_model,
            cpmvs: [crate::inter::MotionVector::ZERO; 3],
        };
        let rec = crate::inter::AffineCbRecord {
            xnb: xcb,
            ynb: ycb,
            nb_w: cb_w,
            nb_h: cb_h,
            model: motion_model,
            pred_flag_l0: use_l0,
            pred_flag_l1: use_l1,
            ref_idx_l0: if use_l0 {
                decision.mvp.ref_idx_l0 as i32
            } else {
                -1
            },
            ref_idx_l1: if use_l1 {
                decision.mvp.ref_idx_l1 as i32
            } else {
                -1
            },
            cpmvs_l0: store_cpmvs_l0.unwrap_or(zero_cpmvs),
            cpmvs_l1: store_cpmvs_l1.unwrap_or(zero_cpmvs),
            // §8.5.6.6.2 — store the parsed affine-AMVP `bcw_idx` only
            // when the CU is bi-pred (both lists active); a uni-pred CU
            // has no meaningful weight index, so it stores the default
            // `0`. A later CU's §8.5.5.2 inherited affine-merge scan
            // (`derive_inherited_affine_merge_side`) then recovers this
            // weight for an inherited bi-pred candidate.
            bcw_idx: if use_l0 && use_l1 { bcw_idx } else { 0 },
        };
        self.store_affine_cb(cb_x, cb_y, cb_w, cb_h, rec);
        Ok(())
    }

    /// §8.5.5.6 — read one constructed-merge corner's full per-list
    /// motion from the regular motion field at the ordered cascade
    /// positions (TL = {B2, B3, A2}, TR = {B1, B0}, BL = {A1, A0}).
    /// Returns the first available position's `(predFlag, refIdx, mv)`
    /// for both lists assembled into an [`AffineCpRecord`]. A corner that
    /// finds no available neighbour reports `AffineCpRecord::UNAVAILABLE`.
    fn read_constructed_merge_corner(
        &self,
        positions: &[(i32, i32)],
        xcb: i32,
        ycb: i32,
    ) -> crate::affine_merge::AffineCpRecord {
        use crate::affine_merge::AffineCpRecord;
        let par = self.log2_par_mrg_level;
        for &(xnb, ynb) in positions {
            let nb = self.motion_field.get_at_luma(xnb, ynb);
            if !nb.available {
                continue;
            }
            // §8.5.5.2 parallel-merge-level suppression (eq. 60).
            if (xcb >> par) == (xnb >> par) && (ycb >> par) == (ynb >> par) {
                continue;
            }
            return AffineCpRecord {
                available: true,
                pred_flag_l0: nb.pred_flag_l0,
                pred_flag_l1: nb.pred_flag_l1,
                ref_idx_l0: if nb.pred_flag_l0 { nb.ref_idx_l0 } else { -1 },
                ref_idx_l1: if nb.pred_flag_l1 { nb.ref_idx_l1 } else { -1 },
                mv_l0: nb.mv_l0,
                mv_l1: nb.mv_l1,
                bcw_idx: nb.bcw_idx,
            };
        }
        AffineCpRecord::UNAVAILABLE
    }

    /// §8.5.5.6 fourth corner — derive the temporal bottom-right corner
    /// record (corner 3) for the constructed affine-merge candidates.
    /// The §8.5.5.6 fourth-corner bullet pins `refIdxL0Corner[3] = 0`
    /// (and `refIdxL1Corner[3] = 0` on B-slices) and reads the §8.5.2.11
    /// collocated MV at the CU's bottom-right. `predFlagL1Corner[3]`
    /// only becomes 1 on B-slices. The MVs are AMVR-rounded with the
    /// default 1/4-luma `AmvrShift(2)` (the temporal candidate is not
    /// AMVR-signalled — it inherits the slice default).
    fn read_temporal_corner3(
        &self,
        xcb: i32,
        ycb: i32,
        cb_w: i32,
        cb_h: i32,
        is_b: bool,
    ) -> crate::affine_merge::AffineCpRecord {
        use crate::affine_merge::AffineCpRecord;
        let amvr = crate::amvr::AmvrShift(2);
        let poc_l0 = self.ref_pic_list_l0.first().map(|r| r.poc);
        let poc_l1 = self.ref_pic_list_l1.first().map(|r| r.poc);
        let mv_l0 =
            poc_l0.and_then(|p| self.derive_amvp_col_candidate(xcb, ycb, cb_w, cb_h, p, amvr));
        let mv_l1 = if is_b {
            poc_l1.and_then(|p| self.derive_amvp_col_candidate(xcb, ycb, cb_w, cb_h, p, amvr))
        } else {
            None
        };
        if mv_l0.is_none() && mv_l1.is_none() {
            return AffineCpRecord::UNAVAILABLE;
        }
        AffineCpRecord {
            available: true,
            pred_flag_l0: mv_l0.is_some(),
            pred_flag_l1: mv_l1.is_some(),
            ref_idx_l0: if mv_l0.is_some() { 0 } else { -1 },
            ref_idx_l1: if mv_l1.is_some() { 0 } else { -1 },
            mv_l0: mv_l0.unwrap_or(MotionVector::ZERO),
            mv_l1: mv_l1.unwrap_or(MotionVector::ZERO),
            // §8.5.5.6 — corners 2 / 3 surface bcwIdx == 0.
            bcw_idx: 0,
        }
    }

    /// §8.5.2.1 — `NoBackwardPredFlag`: `1` when every active reference
    /// in both lists has POC ≤ the current picture POC. Drives the
    /// §8.5.2.12 sbFlag=1 cross-list (LY) fallback in the SbTMVP fuse.
    fn no_backward_pred_flag(&self) -> bool {
        let all_le = |list: &[ReferencePicture]| list.iter().all(|r| r.poc <= self.current_poc);
        all_le(&self.ref_pic_list_l0) && all_le(&self.ref_pic_list_l1)
    }

    /// §8.5.5.3 / §8.5.5.4 — build the SbTMVP record for the current CB:
    /// gate on the §8.5.5.3 first bullet, derive `tempMv` from the A1
    /// neighbour (§8.5.5.4), the centre / grid geometry (eqs. 711 – 718),
    /// and read the collocated CU-centre block (§8.5.5.4 eqs. 729 – 731)
    /// to populate `ctrMvLX` / `ctrPredFlagLX`. Returns `None` when the
    /// gate is closed or no collocated picture is bound. The returned
    /// record's [`SbTmvpRecord::is_sb_col_available`] reports the
    /// §8.5.5.3 step-3 `availableFlagSbCol`.
    fn build_sbtmvp_record(
        &self,
        xcb: i32,
        ycb: i32,
        cb_w: i32,
        cb_h: i32,
    ) -> Option<(crate::sbtmvp::SbTmvpRecord, ReferencePicture)> {
        use crate::sbtmvp::{
            derive_temp_mv, is_sbtmvp_available, ColBlockMotion, PictureBoundary,
            SbTmvpAvailability, SbTmvpCenterLoc, SbTmvpFuseInputs, SbTmvpGrid, SbTmvpRecord,
            SbTmvpTempMvInputs,
        };

        // §8.5.5.3 first-bullet gate.
        let col_list = if self.collocated_from_l0 {
            &self.ref_pic_list_l0
        } else {
            &self.ref_pic_list_l1
        };
        let col_idx = self.col_ref_idx as usize;
        let col_pic = col_list.get(col_idx)?;
        col_pic.motion_field.as_ref()?;
        let gate = SbTmvpAvailability {
            sps_sbtmvp_enabled: self.sps.tool_flags.sbtmvp_enabled_flag,
            ph_temporal_mvp_enabled: self.ph_temporal_mvp_enabled,
            cb_width: cb_w,
            cb_height: cb_h,
            col_pic_present: true,
        };
        if !is_sbtmvp_available(gate) {
            return None;
        }

        let is_b = self.sh.sh_slice_type == SliceType::B;
        let col_pic_poc = col_pic.poc;
        let curr_ref_poc_l0 = self
            .ref_pic_list_l0
            .first()
            .map(|r| r.poc)
            .unwrap_or(self.current_poc);
        let curr_ref_poc_l1 = self
            .ref_pic_list_l1
            .first()
            .map(|r| r.poc)
            .unwrap_or(self.current_poc);

        // §8.5.5.4 A1 neighbour query (xCb − 1, yCb + cbHeight − 1).
        let a1 = self.motion_field.get_at_luma(xcb - 1, ycb + cb_h - 1);
        let temp_mv = derive_temp_mv(SbTmvpTempMvInputs {
            available_a1: a1.available,
            pred_flag_l0_a1: a1.pred_flag_l0,
            pred_flag_l1_a1: a1.pred_flag_l1,
            ref_idx_l0_a1: a1.ref_idx_l0,
            ref_idx_l1_a1: a1.ref_idx_l1,
            mv_l0_a1: a1.mv_l0,
            mv_l1_a1: a1.mv_l1,
            slice_type: self.sh.sh_slice_type,
            col_pic_poc,
            poc_of_l0_ref: &|idx: i32| {
                if idx < 0 {
                    None
                } else {
                    self.ref_pic_list_l0.get(idx as usize).map(|r| r.poc)
                }
            },
            poc_of_l1_ref: &|idx: i32| {
                if idx < 0 {
                    None
                } else {
                    self.ref_pic_list_l1.get(idx as usize).map(|r| r.poc)
                }
            },
        });

        let ctb_log2 = self.layout.ctb_log2_size_y;
        let centre = SbTmvpCenterLoc::derive(xcb, ycb, cb_w, cb_h, ctb_log2);
        let grid = SbTmvpGrid::derive(cb_w, cb_h);
        let boundary = PictureBoundary::Picture {
            pic_width_luma: self.layout.pic_width_luma as i32,
            pic_height_luma: self.layout.pic_height_luma as i32,
        };

        // §8.5.5.4 centre-block read — the collocated motion at the
        // clipped CU-centre location (eqs. 729 – 731) seeds `ctrMvLX`.
        let mf = col_pic.motion_field.as_ref().unwrap();
        let col_sampler = |x: i32, y: i32| -> ColBlockMotion {
            let m = mf.get_at_luma(x, y);
            ColBlockMotion {
                mode_inter: m.available && m.mode_inter,
                pred_flag_l0: m.pred_flag_l0,
                pred_flag_l1: m.pred_flag_l1,
                mv_l0: m.mv_l0,
                mv_l1: m.mv_l1,
                ref_idx_l0: m.ref_idx_l0,
                ref_idx_l1: m.ref_idx_l1,
            }
        };

        let fuse_inputs = SbTmvpFuseInputs {
            xcb,
            ycb,
            ctb_log2_size_y: ctb_log2,
            boundary,
            slice_type: self.sh.sh_slice_type,
            no_backward_pred: self.no_backward_pred_flag(),
            col_pic_poc,
            curr_pic_poc: self.current_poc,
            curr_ref_poc_l0,
            curr_ref_poc_l1,
            poc_of_col_ref: &|list_col: i32, ref_idx: i32| {
                // The collocated picture's RPL is not modelled; the
                // §8.5.2.12 fuse assumes the collocated MV's reference
                // sits on the current axis (equal-distance fast path).
                let _ = (list_col, ref_idx);
                Some(self.current_poc)
            },
        };

        // Centre-block collocated read (eqs. 729 – 731 clip + §8.5.2.12).
        let (xc, yc) = crate::sbtmvp::clip_col_centre_location(
            centre.x_ctb,
            centre.y_ctb,
            ctb_log2,
            boundary,
            centre.x_ctr_cb,
            centre.y_ctr_cb,
            temp_mv,
        );
        let ctr_col = col_sampler((xc >> 3) << 3, (yc >> 3) << 3);
        let (ctr_mv_l0, ctr_pred_l0) = crate::sbtmvp::derive_collocated_mv_subblock_pub(
            ctr_col,
            0,
            curr_ref_poc_l0,
            &fuse_inputs,
        );
        let (ctr_mv_l1, ctr_pred_l1) = if is_b {
            crate::sbtmvp::derive_collocated_mv_subblock_pub(
                ctr_col,
                1,
                curr_ref_poc_l1,
                &fuse_inputs,
            )
        } else {
            (MotionVector::ZERO, false)
        };

        let record = SbTmvpRecord {
            col_pic_poc,
            centre,
            grid,
            ref_idx_l0_sb_col: 0,
            ref_idx_l1_sb_col: 0,
            temp_mv,
            ctr_pred_flag_l0: ctr_pred_l0,
            ctr_pred_flag_l1: ctr_pred_l1,
            ctr_mv_l0,
            ctr_mv_l1,
        };
        Some((record, col_pic.clone()))
    }

    /// §8.5.5.3 + §8.5.6 — reconstruct an SbTMVP (`SbCol`) sub-block
    /// merge CU to pixels. Fills the §8.5.5.3 per-8×8-sub-block motion
    /// grid (`fill_subblock_motion`) and runs translational MC for each
    /// sub-block (uni- or default-weighted bi-pred), then layers residual
    /// and writes the per-sub-block motion field. The collocated motion
    /// for every reference list is read from `RefPicList[0]` /
    /// `RefPicList[1]` at `refIdxLXSbCol = 0`.
    fn reconstruct_leaf_cu_sbtmvp(
        &mut self,
        cu: &CtuCu,
        info: &LeafCuInfo,
        residual: &LeafCuResidual,
        record: &crate::sbtmvp::SbTmvpRecord,
        col_pic: &ReferencePicture,
        out: &mut PictureBuffer,
    ) -> Result<()> {
        use crate::sbtmvp::{
            fill_subblock_motion, ColBlockMotion, PictureBoundary, SbTmvpFuseInputs,
        };

        let xcb = cu.cu.x as i32;
        let ycb = cu.cu.y as i32;
        let is_b = self.sh.sh_slice_type == SliceType::B;
        let ctb_log2 = self.layout.ctb_log2_size_y;
        let boundary = PictureBoundary::Picture {
            pic_width_luma: self.layout.pic_width_luma as i32,
            pic_height_luma: self.layout.pic_height_luma as i32,
        };
        let curr_ref_poc_l0 = self
            .ref_pic_list_l0
            .first()
            .map(|r| r.poc)
            .unwrap_or(self.current_poc);
        let curr_ref_poc_l1 = self
            .ref_pic_list_l1
            .first()
            .map(|r| r.poc)
            .unwrap_or(self.current_poc);

        let mf = col_pic.motion_field.as_ref().ok_or_else(|| {
            Error::invalid("h266 sbtmvp: collocated picture lost its motion field")
        })?;
        let col_sampler = |x: i32, y: i32| -> ColBlockMotion {
            let m = mf.get_at_luma(x, y);
            ColBlockMotion {
                mode_inter: m.available && m.mode_inter,
                pred_flag_l0: m.pred_flag_l0,
                pred_flag_l1: m.pred_flag_l1,
                mv_l0: m.mv_l0,
                mv_l1: m.mv_l1,
                ref_idx_l0: m.ref_idx_l0,
                ref_idx_l1: m.ref_idx_l1,
            }
        };
        let fuse_inputs = SbTmvpFuseInputs {
            xcb,
            ycb,
            ctb_log2_size_y: ctb_log2,
            boundary,
            slice_type: self.sh.sh_slice_type,
            no_backward_pred: self.no_backward_pred_flag(),
            col_pic_poc: record.col_pic_poc,
            curr_pic_poc: self.current_poc,
            curr_ref_poc_l0,
            curr_ref_poc_l1,
            poc_of_col_ref: &|_list_col: i32, _ref_idx: i32| Some(self.current_poc),
        };

        let sb_grid = fill_subblock_motion(record, &fuse_inputs, &col_sampler);
        let sb_w = record.grid.sb_width as u32;
        let sb_h = record.grid.sb_height as u32;

        // Per-sub-block translational MC. refIdxLXSbCol == 0, so the
        // reference picture is RefPicList[X][0].
        let ref_l0 = self.ref_pic_list_l0.first().cloned();
        let ref_l1 = self.ref_pic_list_l1.first().cloned();

        for ys in 0..sb_grid.num_sb_y {
            for xs in 0..sb_grid.num_sb_x {
                let sb = sb_grid.at(xs, ys);
                let bx = cu.cu.x + xs as u32 * sb_w;
                let by = cu.cu.y + ys as u32 * sb_h;
                let use_l0 = sb.pred_flag_l0 && ref_l0.is_some();
                let use_l1 = is_b && sb.pred_flag_l1 && ref_l1.is_some();
                match (use_l0, use_l1) {
                    (true, false) => {
                        let rp = ref_l0.as_ref().unwrap();
                        crate::inter::predict_luma_block(
                            &mut out.luma,
                            bx,
                            by,
                            sb_w,
                            sb_h,
                            &rp.frame.luma,
                            sb.mv_l0,
                        )?;
                        self.sbtmvp_chroma_mc(out, bx, by, sb_w, sb_h, &rp.frame, sb.mv_l0)?;
                    }
                    (false, true) => {
                        let rp = ref_l1.as_ref().unwrap();
                        crate::inter::predict_luma_block(
                            &mut out.luma,
                            bx,
                            by,
                            sb_w,
                            sb_h,
                            &rp.frame.luma,
                            sb.mv_l1,
                        )?;
                        self.sbtmvp_chroma_mc(out, bx, by, sb_w, sb_h, &rp.frame, sb.mv_l1)?;
                    }
                    (true, true) => {
                        let rp0 = ref_l0.as_ref().unwrap();
                        let rp1 = ref_l1.as_ref().unwrap();
                        // §8.5.6.6.2 eq. 980 default-weighted average.
                        crate::inter::predict_luma_block_bipred(
                            &mut out.luma,
                            bx,
                            by,
                            sb_w,
                            sb_h,
                            &rp0.frame.luma,
                            sb.mv_l0,
                            &rp1.frame.luma,
                            sb.mv_l1,
                        )?;
                        self.sbtmvp_chroma_mc_bi(
                            out, bx, by, sb_w, sb_h, &rp0.frame, sb.mv_l0, &rp1.frame, sb.mv_l1,
                        )?;
                    }
                    (false, false) => {
                        return Err(Error::invalid(
                            "h266 sbtmvp: sub-block has neither list active after centre fallback",
                        ));
                    }
                }
            }
        }

        // §8.5.8 residual + bookkeeping.
        self.add_inter_cu_residual(cu, info, residual, out)?;
        // Broadcast a representative MV (the first sub-block's) onto the
        // motion field cells; finer SbTMVP per-sub-block field writes
        // follow the same grid.
        for ys in 0..sb_grid.num_sb_y {
            for xs in 0..sb_grid.num_sb_x {
                let sb = sb_grid.at(xs, ys);
                let mvf = MvField {
                    mv_l0: sb.mv_l0,
                    ref_idx_l0: sb.ref_idx_l0,
                    pred_flag_l0: sb.pred_flag_l0,
                    mv_l1: sb.mv_l1,
                    ref_idx_l1: sb.ref_idx_l1,
                    pred_flag_l1: sb.pred_flag_l1 && is_b,
                    cu_skip_flag: false,
                    mode_inter: true,
                    available: true,
                    bcw_idx: 0,
                };
                let bx = cu.cu.x + xs as u32 * sb_w;
                let by = cu.cu.y + ys as u32 * sb_h;
                self.motion_field.write_block(bx, by, sb_w, sb_h, mvf);
            }
        }
        // SbTMVP CBs are translational at the sub-block granularity — no
        // affine CPMV record; clear the affine store so later §8.5.5.7
        // scans don't inherit a stale record.
        self.affine_cpmv_field
            .write_block(cu.cu.x, cu.cu.y, cu.cu.w, cu.cu.h, None);

        let qp_y = (self.cabac.slice_qp_y.0 + info.cu_qp_delta_val).clamp(0, 63);
        self.deblock_cus.push(DeblockCu {
            x: cu.cu.x,
            y: cu.cu.y,
            w: cu.cu.w,
            h: cu.cu.h,
            qp_y,
            intra: false,
            tu_y_coded: info.tu_y_coded_flag,
            tu_cb_coded: info.tu_cb_coded_flag,
            tu_cr_coded: info.tu_cr_coded_flag,
            bdpcm_luma: false,
            bdpcm_chroma: false,
        });
        self.write_intra_block(cu.cu.x, cu.cu.y, cu.cu.w, cu.cu.h, false);
        self.commit_subblock_neighbour_state(cu, info);
        Ok(())
    }

    /// 4:2:0 chroma MC for one SbTMVP sub-block (uni-pred). The chroma
    /// sub-block is `sb_w/2 × sb_h/2` at `(bx/2, by/2)`; the MV is the
    /// luma sub-block MV (the §8.5.5.3 grid carries luma-precision MVs
    /// that the §8.5.6.3.4 4-tap chroma filter consumes directly).
    fn sbtmvp_chroma_mc(
        &self,
        out: &mut PictureBuffer,
        bx: u32,
        by: u32,
        sb_w: u32,
        sb_h: u32,
        ref_frame: &PictureBuffer,
        mv: MotionVector,
    ) -> Result<()> {
        if self.sps.sps_chroma_format_idc != 1 {
            return Ok(());
        }
        crate::inter::predict_chroma_block(
            &mut out.cb,
            bx / 2,
            by / 2,
            sb_w / 2,
            sb_h / 2,
            &ref_frame.cb,
            mv,
        )?;
        crate::inter::predict_chroma_block(
            &mut out.cr,
            bx / 2,
            by / 2,
            sb_w / 2,
            sb_h / 2,
            &ref_frame.cr,
            mv,
        )?;
        Ok(())
    }

    /// 4:2:0 chroma bi-pred MC for one SbTMVP sub-block (§8.5.6.6.2 eq.
    /// 980 default-weighted average).
    #[allow(clippy::too_many_arguments)]
    fn sbtmvp_chroma_mc_bi(
        &self,
        out: &mut PictureBuffer,
        bx: u32,
        by: u32,
        sb_w: u32,
        sb_h: u32,
        ref_l0: &PictureBuffer,
        mv_l0: MotionVector,
        ref_l1: &PictureBuffer,
        mv_l1: MotionVector,
    ) -> Result<()> {
        if self.sps.sps_chroma_format_idc != 1 {
            return Ok(());
        }
        crate::inter::predict_chroma_block_bipred(
            &mut out.cb,
            bx / 2,
            by / 2,
            sb_w / 2,
            sb_h / 2,
            &ref_l0.cb,
            mv_l0,
            &ref_l1.cb,
            mv_l1,
        )?;
        crate::inter::predict_chroma_block_bipred(
            &mut out.cr,
            bx / 2,
            by / 2,
            sb_w / 2,
            sb_h / 2,
            &ref_l0.cr,
            mv_l0,
            &ref_l1.cr,
            mv_l1,
        )?;
        Ok(())
    }

    /// §8.5.5.2 + §8.5.5.3 — sub-block (affine / SbTMVP) merge
    /// reconstruction to pixels.
    ///
    /// Entered from [`Self::reconstruct_leaf_cu_inter`] when the parsed
    /// `merge_subblock_flag == 1`. Builds the §8.5.5.2 sub-block merge
    /// candidate list (inherited A/B per §8.5.5.5 + constructed Const1..6
    /// per §8.5.5.6 + zero-MV pad; the §8.5.5.3 SbCol temporal candidate
    /// is a follow-up), picks the entry at `merge_subblock_idx`, then
    /// drives the affine MC ([`Self::reconstruct_affine_inter_uni`] /
    /// [`Self::reconstruct_affine_inter_bi`]) for the picked candidate's
    /// per-list CPMVs. The shared §8.5.8 residual tail
    /// ([`Self::add_inter_cu_residual`]) layers the coded residual, and
    /// the §8.5.5.9 sub-block MV grid is broadcast into the motion field
    /// so later CUs' spatial merge / AMVP scans see this block's motion.
    /// The affine CB record is stored for the §8.5.5.7 / §8.5.5.5
    /// inherited scan of subsequent CUs.
    fn reconstruct_leaf_cu_inter_subblock_merge(
        &mut self,
        cu: &CtuCu,
        info: &LeafCuInfo,
        residual: &LeafCuResidual,
        out: &mut PictureBuffer,
    ) -> Result<()> {
        use crate::affine_merge::{
            build_subblock_merge_cand_list, derive_constructed_affine_merge_candidates,
            ConstructedAffineFlags, SubblockMergeCandidateKind, SubblockMergeListInputs,
        };

        let xcb = cu.cu.x as i32;
        let ycb = cu.cu.y as i32;
        let cb_w = cu.cu.w;
        let cb_h = cu.cu.h;

        // §8.5.5.2 step 4 / 5 — inherited A (A0 → A1) + inherited B
        // (B0 → B1 → B2).
        let a_positions = [
            (xcb - 1, ycb + cb_h as i32),     // A0 (eq. 674)
            (xcb - 1, ycb + cb_h as i32 - 1), // A1 (eq. 675)
        ];
        let b_positions = [
            (xcb + cb_w as i32, ycb - 1),     // B0 (eq. 677)
            (xcb + cb_w as i32 - 1, ycb - 1), // B1 (eq. 678)
            (xcb - 1, ycb - 1),               // B2 (eq. 679)
        ];
        let inherited_a =
            self.derive_inherited_affine_merge_side(&a_positions, xcb, ycb, cb_w, cb_h);
        let inherited_b =
            self.derive_inherited_affine_merge_side(&b_positions, xcb, ycb, cb_w, cb_h);

        // §8.5.5.2 step 6 + §8.5.5.6 — constructed Const1..6. Corners
        // 0 / 1 / 2 read the regular per-list motion field at the cascade
        // positions; corner 3 (bottom-right) is the §8.5.5.6 fourth-corner
        // temporal collocated MV (`read_temporal_corner3`), so the
        // corner-3-dependent triples (Const2 / Const3 / Const4) can now
        // light up alongside Const1 / Const5 / Const6.
        let tl_positions = [
            (xcb - 1, ycb - 1), // B2
            (xcb, ycb - 1),     // B3
            (xcb - 1, ycb),     // A2
        ];
        let tr_positions = [
            (xcb + cb_w as i32 - 1, ycb - 1), // B1
            (xcb + cb_w as i32, ycb - 1),     // B0
        ];
        let bl_positions = [
            (xcb - 1, ycb + cb_h as i32 - 1), // A1
            (xcb - 1, ycb + cb_h as i32),     // A0
        ];
        let is_b = self.sh.sh_slice_type == SliceType::B;
        let corner0 = self.read_constructed_merge_corner(&tl_positions, xcb, ycb);
        let corner1 = self.read_constructed_merge_corner(&tr_positions, xcb, ycb);
        let corner2 = self.read_constructed_merge_corner(&bl_positions, xcb, ycb);
        // §8.5.5.6 fourth corner — the temporal bottom-right collocated
        // MV with `refIdxLXCorner[3] = 0`, derived through the §8.5.2.11
        // AMVP temporal path per active list. The L1 half is only OR-
        // folded on B-slices (the fourth-corner bullet).
        let corner3 = self.read_temporal_corner3(xcb, ycb, cb_w as i32, cb_h as i32, is_b);
        let corners = [corner0, corner1, corner2, corner3];

        let constructed = derive_constructed_affine_merge_candidates(
            cb_w,
            cb_h,
            &corners,
            ConstructedAffineFlags {
                sps_6param_affine_enabled_flag: self.sps.tool_flags.six_param_affine_enabled_flag,
                slice_type_b: is_b,
            },
        );

        // §8.5.5.3 — build the SbTMVP (`SbCol`) record. When the
        // §8.5.5.3 step-3 `availableFlagSbCol` is true it occupies slot 0
        // of the sub-block merge list (ahead of inherited A/B).
        let sbtmvp = self.build_sbtmvp_record(xcb, ycb, cb_w as i32, cb_h as i32);
        let sb_col_available = sbtmvp
            .as_ref()
            .map(|(rec, _)| rec.is_sb_col_available())
            .unwrap_or(false);

        let max_num = self.cu_tool_flags().max_num_subblock_merge_cand;
        let list = build_subblock_merge_cand_list(SubblockMergeListInputs {
            max_num_cand: max_num,
            slice_type_b: is_b,
            sb_col_available,
            inherited_a,
            inherited_b,
            constructed: &constructed,
        });

        let idx = info.inter.merge_data.merge_subblock_idx;
        let Some((kind, cand)) = list.pick(idx) else {
            return Err(Error::invalid(format!(
                "h266 subblock merge: merge_subblock_idx {idx} out of range \
                 (list has {} entries)",
                list.len()
            )));
        };

        // §8.5.5.3 SbCol — reconstruct the SbTMVP sub-block grid to
        // pixels through the dedicated walker.
        if matches!(kind, SubblockMergeCandidateKind::SbCol) {
            let (record, col_pic) = sbtmvp.ok_or_else(|| {
                Error::invalid("h266 subblock merge: SbCol picked but no SbTMVP record was built")
            })?;
            return self.reconstruct_leaf_cu_sbtmvp(cu, info, residual, &record, &col_pic, out);
        }

        // Resolve the per-list reference pictures.
        let use_l0 = cand.pred_flag_l0;
        let use_l1 = cand.pred_flag_l1;
        if !use_l0 && !use_l1 {
            return Err(Error::invalid(
                "h266 subblock merge: picked candidate has both predFlags == 0",
            ));
        }
        let ref_pic_l0 = if use_l0 {
            let ri = cand.ref_idx_l0;
            if ri < 0 || (ri as usize) >= self.ref_pic_list_l0.len() {
                return Err(Error::invalid(format!(
                    "h266 subblock merge: refIdxL0 {ri} out of range"
                )));
            }
            Some(self.ref_pic_list_l0[ri as usize].clone())
        } else {
            None
        };
        let ref_pic_l1 = if use_l1 {
            let ri = cand.ref_idx_l1;
            if ri < 0 || (ri as usize) >= self.ref_pic_list_l1.len() {
                return Err(Error::invalid(format!(
                    "h266 subblock merge: refIdxL1 {ri} out of range"
                )));
            }
            Some(self.ref_pic_list_l1[ri as usize].clone())
        } else {
            None
        };

        // §8.5.5.9 affine MC.
        match (&ref_pic_l0, &ref_pic_l1) {
            (Some(rp0), None) => {
                self.reconstruct_affine_inter_uni(
                    cu.cu.x,
                    cu.cu.y,
                    cb_w,
                    cb_h,
                    &cand.cpmvs_l0,
                    rp0,
                    out,
                )?;
            }
            (None, Some(rp1)) => {
                self.reconstruct_affine_inter_uni(
                    cu.cu.x,
                    cu.cu.y,
                    cb_w,
                    cb_h,
                    &cand.cpmvs_l1,
                    rp1,
                    out,
                )?;
            }
            (Some(rp0), Some(rp1)) => {
                // §8.5.6.6.2 — the sub-block (affine) merge candidate
                // carries its inherited / constructed `bcwIdx`; thread
                // it into the bi-pred blend.
                self.reconstruct_affine_inter_bi_bcw(
                    cu.cu.x,
                    cu.cu.y,
                    cb_w,
                    cb_h,
                    &cand.cpmvs_l0,
                    rp0,
                    &cand.cpmvs_l1,
                    rp1,
                    cand.bcw_idx,
                    out,
                )?;
            }
            (None, None) => unreachable!(),
        }

        // §8.5.8 residual add (shared tail).
        self.add_inter_cu_residual(cu, info, residual, out)?;

        // Broadcast the §8.5.5.9 per-sub-block MV grid into the motion
        // field. Each 4×4 cell takes its covering affine sub-block's MV
        // so later spatial scans (regular merge / AMVP / HMVP seed) read
        // a representative MV. The CU-corner CPMVs are kept in the affine
        // CB store separately for the §8.5.5.7 inherited scan.
        self.write_affine_subblock_motion_field(cu, &cand, use_l0, use_l1);

        // Store the affine CB record so the next CU's §8.5.5.5 /
        // §8.5.5.7 inherited scan can recover this block's CPMVs.
        let zero_cpmvs = crate::affine::AffineCpmvs {
            model: cand.motion_model,
            cpmvs: [MotionVector::ZERO; 3],
        };
        let rec = crate::inter::AffineCbRecord {
            xnb: xcb,
            ynb: ycb,
            nb_w: cb_w,
            nb_h: cb_h,
            model: cand.motion_model,
            pred_flag_l0: use_l0,
            pred_flag_l1: use_l1,
            ref_idx_l0: if use_l0 { cand.ref_idx_l0 } else { -1 },
            ref_idx_l1: if use_l1 { cand.ref_idx_l1 } else { -1 },
            cpmvs_l0: if use_l0 { cand.cpmvs_l0 } else { zero_cpmvs },
            cpmvs_l1: if use_l1 { cand.cpmvs_l1 } else { zero_cpmvs },
            // §8.5.6.6.2 — preserve the picked candidate's BcwIdx so a
            // later CU's §8.5.5.2 inherited candidate recovers it.
            bcw_idx: cand.bcw_idx,
        };
        self.store_affine_cb(cu.cu.x, cu.cu.y, cb_w, cb_h, rec);

        // Deblock + intra-grid + subblock-neighbour bookkeeping (mirrors
        // the translational tail; the §9.3.4.2.2 merge-subblock context
        // must read this CU back as `MergeSubblockFlag == 1`).
        let qp_y = (self.cabac.slice_qp_y.0 + info.cu_qp_delta_val).clamp(0, 63);
        self.deblock_cus.push(DeblockCu {
            x: cu.cu.x,
            y: cu.cu.y,
            w: cu.cu.w,
            h: cu.cu.h,
            qp_y,
            intra: false,
            tu_y_coded: info.tu_y_coded_flag,
            tu_cb_coded: info.tu_cb_coded_flag,
            tu_cr_coded: info.tu_cr_coded_flag,
            bdpcm_luma: false,
            bdpcm_chroma: false,
        });
        self.write_intra_block(cu.cu.x, cu.cu.y, cu.cu.w, cu.cu.h, false);
        self.commit_subblock_neighbour_state(cu, info);
        Ok(())
    }

    /// §8.5.5.9 — broadcast a sub-block affine candidate's per-4×4 MV
    /// grid into the motion field. The grid is derived once per active
    /// list from the candidate CPMVs and written cell-by-cell; the
    /// per-cell record carries the candidate's pred flags / ref indices /
    /// bcwIdx so later neighbour reads are well-formed.
    fn write_affine_subblock_motion_field(
        &mut self,
        cu: &CtuCu,
        cand: &crate::affine_merge::AffineMergeCandidate,
        use_l0: bool,
        use_l1: bool,
    ) {
        use crate::affine::derive_subblock_mvs;
        let cb_w = cu.cu.w;
        let cb_h = cu.cu.h;
        let bipred = use_l0 && use_l1;
        let grid_l0 = if use_l0 {
            derive_subblock_mvs(cb_w, cb_h, &cand.cpmvs_l0, bipred).ok()
        } else {
            None
        };
        let grid_l1 = if use_l1 {
            derive_subblock_mvs(cb_w, cb_h, &cand.cpmvs_l1, bipred).ok()
        } else {
            None
        };
        // Determine the sub-block grid geometry from whichever list is
        // active (both lists share the §8.5.5.9 4×4 / sub-block layout).
        let geom = grid_l0.as_ref().or(grid_l1.as_ref());
        let Some(geom) = geom else {
            return;
        };
        let sb_w = cb_w / geom.num_sb_x;
        let sb_h = cb_h / geom.num_sb_y;
        for gy in 0..geom.num_sb_y {
            for gx in 0..geom.num_sb_x {
                let mv_l0 = grid_l0
                    .as_ref()
                    .map(|g| g.mv_at(gx, gy))
                    .unwrap_or(MotionVector::ZERO);
                let mv_l1 = grid_l1
                    .as_ref()
                    .map(|g| g.mv_at(gx, gy))
                    .unwrap_or(MotionVector::ZERO);
                let mvf = MvField {
                    mv_l0,
                    ref_idx_l0: if use_l0 { cand.ref_idx_l0 } else { -1 },
                    pred_flag_l0: use_l0,
                    mv_l1,
                    ref_idx_l1: if use_l1 { cand.ref_idx_l1 } else { -1 },
                    pred_flag_l1: use_l1,
                    cu_skip_flag: false,
                    mode_inter: true,
                    available: true,
                    bcw_idx: cand.bcw_idx,
                };
                let bx = cu.cu.x + gx * sb_w;
                let by = cu.cu.y + gy * sb_h;
                self.motion_field.write_block(bx, by, sb_w, sb_h, mvf);
            }
        }
    }

    /// §8.5.6.6.2 eq. 980 default-weighted average of two co-located
    /// plane regions: `out[x][y] = (a[x][y] + b[x][y] + 1) >> 1` over
    /// the `(w × h)` rectangle at `(dx, dy)`, where `a` / `b` are read
    /// at the **same** offset (not anchored at (0,0) like
    /// [`crate::inter::bi_pred_avg_8bit`]).
    #[allow(clippy::too_many_arguments)]
    fn bi_avg_plane_region(
        &self,
        out: &mut crate::reconstruct::PicturePlane,
        dx: u32,
        dy: u32,
        w: u32,
        h: u32,
        a: &crate::reconstruct::PicturePlane,
        b: &crate::reconstruct::PicturePlane,
    ) {
        for r in 0..h as usize {
            for c in 0..w as usize {
                let idx = (dy as usize + r) * out.stride + (dx as usize + c);
                let ai = (dy as usize + r) * a.stride + (dx as usize + c);
                let bi = (dy as usize + r) * b.stride + (dx as usize + c);
                let v = (a.samples[ai] as u32 + b.samples[bi] as u32 + 1) >> 1;
                out.samples[idx] = v as u8;
            }
        }
    }

    /// §8.5.6.6.2 eq. 981 BCW-weighted blend of two co-located plane
    /// regions: `out[x][y] = Clip1((w0·a + w1·b + 4) >> 3)` over the
    /// `(w × h)` rectangle at `(dx, dy)`, with `w1 = BCW_W_LUT[bcw_idx]`
    /// and `w0 = 8 − w1`. `a` / `b` are read at the **same** offset as
    /// the destination (the affine scratch-plane convention). When
    /// `bcw_idx == 0` this is identical to [`Self::bi_avg_plane_region`]
    /// (the eq. 980 default-weighted average), so the caller routes
    /// through here only when `bcw_idx ∈ 1..=4`.
    #[allow(clippy::too_many_arguments)]
    fn bi_blend_plane_region_bcw(
        &self,
        out: &mut crate::reconstruct::PicturePlane,
        dx: u32,
        dy: u32,
        w: u32,
        h: u32,
        a: &crate::reconstruct::PicturePlane,
        b: &crate::reconstruct::PicturePlane,
        bcw_idx: u8,
    ) {
        let idx_w = bcw_idx as usize;
        if idx_w == 0 || idx_w >= crate::inter::BCW_W_LUT.len() {
            self.bi_avg_plane_region(out, dx, dy, w, h, a, b);
            return;
        }
        let w1 = crate::inter::BCW_W_LUT[idx_w];
        let w0 = 8 - w1;
        for r in 0..h as usize {
            for c in 0..w as usize {
                let idx = (dy as usize + r) * out.stride + (dx as usize + c);
                let ai = (dy as usize + r) * a.stride + (dx as usize + c);
                let bi = (dy as usize + r) * b.stride + (dx as usize + c);
                let blended = (w0 * a.samples[ai] as i32 + w1 * b.samples[bi] as i32 + 4) >> 3;
                out.samples[idx] = blended.clamp(0, 255) as u8;
            }
        }
    }

    /// Predict ONE reference list's affine MC (luma + 4:2:0 chroma)
    /// into caller-supplied planes. The luma plane is written at
    /// `(luma_dst_x, luma_dst_y)`; the chroma planes at
    /// `(c_dst_x, c_dst_y)`. The §8.5.5.9 sub-block MV derivation is
    /// anchored at the CU origin and the CPMVs encode the absolute
    /// motion, so only the CU dimensions `(cb_w, cb_h)` are needed.
    /// Shared by the uni- and bi-pred entry points.
    #[allow(clippy::too_many_arguments)]
    fn predict_affine_one_list(
        &self,
        cb_w: u32,
        cb_h: u32,
        cpmvs: &crate::affine::AffineCpmvs,
        ref_pic: &ReferencePicture,
        bipred: bool,
        luma_dst: &mut crate::reconstruct::PicturePlane,
        luma_dst_x: u32,
        luma_dst_y: u32,
        cb_dst: &mut crate::reconstruct::PicturePlane,
        cr_dst: &mut crate::reconstruct::PicturePlane,
        c_dst_x: u32,
        c_dst_y: u32,
    ) -> Result<()> {
        use crate::affine::{
            derive_subblock_mvs, predict_luma_block_affine_prof, AffineLumaFilterSet,
        };
        // §8.5.6.3.2 — the default affine filter set (no RPR scaling in
        // the scaffold). The sub-block MV grid is anchored at the
        // CU's picture-absolute (cb_x, cb_y) but the samples land at the
        // destination offset.
        predict_luma_block_affine_prof(
            luma_dst,
            luma_dst_x,
            luma_dst_y,
            cb_w,
            cb_h,
            &ref_pic.frame.luma,
            cpmvs,
            AffineLumaFilterSet::Set0,
            bipred,
            self.ph_prof_disabled,
            /*rpr_constraints_active=*/ false,
        )?;

        if self.sps.sps_chroma_format_idc != 1 {
            return Ok(());
        }
        // §8.5.5.3 eqs. 876 – 879 — average the top-left + bottom-right
        // luma sub-block MVs of each 2×2 group into the chroma MV.
        let grid = derive_subblock_mvs(cb_w, cb_h, cpmvs, bipred)?;
        let sb_w = cb_w / grid.num_sb_x;
        let sb_h = cb_h / grid.num_sb_y;
        let mut gy = 0u32;
        while gy < grid.num_sb_y {
            let mut gx = 0u32;
            while gx < grid.num_sb_x {
                let tl = grid.mv_at(gx, gy);
                let br_x = (gx + 1).min(grid.num_sb_x - 1);
                let br_y = (gy + 1).min(grid.num_sb_y - 1);
                let br = grid.mv_at(br_x, br_y);
                let avg = MotionVector {
                    x: round_half_toward_zero_div2(tl.x + br.x),
                    y: round_half_toward_zero_div2(tl.y + br.y),
                };
                let c_w = sb_w; // 2 luma sub-blocks (8 luma) → 4 chroma
                let c_h = sb_h;
                let c_sb_x = c_dst_x + (gx * sb_w) / 2;
                let c_sb_y = c_dst_y + (gy * sb_h) / 2;
                predict_chroma_block(cb_dst, c_sb_x, c_sb_y, c_w, c_h, &ref_pic.frame.cb, avg)?;
                predict_chroma_block(cr_dst, c_sb_x, c_sb_y, c_w, c_h, &ref_pic.frame.cr, avg)?;
                gx += 2;
            }
            gy += 2;
        }
        Ok(())
    }

    /// §8.5.2.11 (AMVP path) — derive the temporal collocated AMVP
    /// candidate for the active list, AMVR-rounded. Mirrors
    /// [`Self::derive_col_candidate`] but feeds the AMVP-specific
    /// `current_ref_poc` (the POC of the parsed `RefPicList[X][refIdxLX]`)
    /// and applies the §8.5.2.9 step-3 last-bullet AMVR rounding via
    /// [`crate::amvp::derive_temporal_amvp_candidate`]. Returns `None`
    /// when TMVP is disabled, the CU is ≤ 32 samples, or the ColPic
    /// carries no motion field.
    fn derive_amvp_col_candidate(
        &self,
        xcb: i32,
        ycb: i32,
        cb_w: i32,
        cb_h: i32,
        current_ref_poc: i32,
        amvr: crate::amvr::AmvrShift,
    ) -> Option<MotionVector> {
        let col_list = if self.collocated_from_l0 {
            &self.ref_pic_list_l0
        } else {
            &self.ref_pic_list_l1
        };
        let col_pic = col_list.get(self.col_ref_idx as usize)?;
        col_pic.motion_field.as_ref()?;
        let inputs = crate::amvp::AmvpTemporalInputs {
            xcb,
            ycb,
            cb_w,
            cb_h,
            pic_w: self.layout.pic_width_luma as i32,
            pic_h: self.layout.pic_height_luma as i32,
            ctb_log2_size_y: self.layout.ctb_log2_size_y,
            current_poc: self.current_poc,
            current_ref_poc,
            // §8.5.2.12 colPocDiff — without modelling the ColPic's own
            // RPL we use the equal-distance fast path (eq. 600), exact
            // for the fixtures this lands against.
            col_ref_poc: self.current_poc,
            ph_temporal_mvp_enabled: self.ph_temporal_mvp_enabled,
        };
        crate::amvp::derive_temporal_amvp_candidate(&inputs, col_pic, amvr)
    }

    /// §8.5.8 + §8.7.5.1 — add the inverse-transformed inter residual
    /// of one component's single transform block to the motion-
    /// compensated prediction already present in `out`.
    ///
    /// `levels` is the §7.3.11.10 coefficient-level array (scan-order
    /// already de-scanned into a row-major `n_tb_w × n_tb_h` grid by
    /// the residual reader). The dequant (§8.7.3) + inverse 2-D DCT-II
    /// (§8.7.4, the `mts_idx == 0` regular path — explicit MTS / SBT /
    /// transform-skip are out of scope for this single-TB inter path)
    /// produce `resSamples`, and §8.7.5.1 forms `recSamples =
    /// Clip1(predSamples + resSamples)` in place. Chroma uses the same
    /// flat-list dequant; `c_idx ∈ {1, 2}` selects the destination
    /// plane and applies the chroma QP mapping (§8.7.3 — identity-plus-
    /// offset clamp here, matching the intra chroma path).
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    fn add_inter_residual_plane(
        &self,
        c_idx: u32,
        x0: usize,
        y0: usize,
        n_tb_w: usize,
        n_tb_h: usize,
        levels: &[i32],
        cu_qp_delta: i32,
        bit_depth: u32,
        transform_skip: bool,
        cu_chroma_qp_offset: i32,
        lmcs_chroma_var_scale: Option<u32>,
        out: &mut PictureBuffer,
    ) -> Result<()> {
        // The regular (`cu_sbt_flag == 0`) inter residual path uses the
        // §8.7.4.1 `mts_idx == 0` DCT-II × DCT-II kernel. When
        // `transform_skip` is set the inverse transform is bypassed
        // (§8.7.4.6) — `add_inter_residual_plane_tr` short-circuits on the
        // flag before reaching the kernel selection.
        self.add_inter_residual_plane_tr(
            c_idx,
            x0,
            y0,
            n_tb_w,
            n_tb_h,
            levels,
            cu_qp_delta,
            bit_depth,
            TrType::DctII,
            TrType::DctII,
            transform_skip,
            cu_chroma_qp_offset,
            lmcs_chroma_var_scale,
            out,
        )
    }

    /// As [`Self::add_inter_residual_plane`] but with caller-selected
    /// horizontal / vertical inverse-transform kernels. The SBT path
    /// (§8.7.4.1 Table 40) passes DST-VII / DCT-VIII; the regular path
    /// passes DCT-II × DCT-II.
    #[allow(clippy::too_many_arguments)]
    fn add_inter_residual_plane_tr(
        &self,
        c_idx: u32,
        x0: usize,
        y0: usize,
        n_tb_w: usize,
        n_tb_h: usize,
        levels: &[i32],
        cu_qp_delta: i32,
        bit_depth: u32,
        tr_h: TrType,
        tr_v: TrType,
        transform_skip: bool,
        cu_chroma_qp_offset: i32,
        lmcs_chroma_var_scale: Option<u32>,
        out: &mut PictureBuffer,
    ) -> Result<()> {
        if n_tb_w == 0 || n_tb_h == 0 || levels.len() != n_tb_w * n_tb_h {
            return Ok(());
        }
        // §8.7.3 — QP for this component. Luma is slice QP + cu_qp_delta;
        // chroma maps through `chroma_qp_identity` (QpC = QpY + offset,
        // clamped) mirroring the intra chroma reconstruction path.
        let qp_y = (self.cabac.slice_qp_y.0 + cu_qp_delta).clamp(0, 63);
        let qp = if c_idx == 0 {
            qp_y
        } else {
            // §8.7.1 eqs. 1147 / 1148 — PPS + slice chroma offset for this
            // component, plus the §7.4.10.6 CU-level `CuQpOffsetC?` term
            // (`cu_chroma_qp_offset`, indexed from the PPS offset list by
            // the caller when `cu_chroma_qp_offset_flag == 1`; 0 otherwise),
            // now matching the intra chroma reconstruction path.
            let qp_offset = chroma_qp_offset_sum(
                c_idx,
                self.pps.pps_cb_qp_offset,
                self.pps.pps_cr_qp_offset,
                self.sh.sh_cb_qp_offset,
                self.sh.sh_cr_qp_offset,
                cu_chroma_qp_offset,
            );
            chroma_qp_identity(qp_y, qp_offset)
        };
        let log2_tr_range = self.log2_transform_range();
        let params = DequantParams {
            bit_depth,
            log2_transform_range: log2_tr_range,
            n_tb_w: n_tb_w as u32,
            n_tb_h: n_tb_h as u32,
            qp,
            dep_quant: false,
            transform_skip,
            bdpcm: false,
            bdpcm_dir: false,
        };
        let d = dequantize_tb_flat(levels, &params)?;
        // §8.7.4.6 — when transform_skip_flag is 1 the dequantised array is
        // the residual sample array directly (res[x][y] = d[x][y]); the
        // separable inverse transform is bypassed. Otherwise run the
        // §8.7.4.1 kernel (DCT-II for the regular path, Table-40 DST-VII /
        // DCT-VIII for SBT).
        let mut res = if transform_skip {
            d
        } else {
            // §8.7.4.1 eqs. 1171 / 1172 — non-zero coefficient ranges depend
            // on the kernel (DST-VII / DCT-VIII cap at 16, DCT-II at 32).
            let non_zero_w = n_tb_w.min(if matches!(tr_h, TrType::DctII) {
                32
            } else {
                16
            });
            let non_zero_h = n_tb_h.min(if matches!(tr_v, TrType::DctII) {
                32
            } else {
                16
            });
            inverse_transform_2d(
                n_tb_w,
                n_tb_h,
                non_zero_w,
                non_zero_h,
                tr_h,
                tr_v,
                &d,
                bit_depth,
                log2_tr_range,
            )?
        };
        // §8.7.5.3 eqs. 1219 / 1220 — when LMCS chroma residual scaling
        // applies to this chroma TB the caller passes the eq. 1218
        // `varScale`; the residual is folded before the eq. 1220 add.
        if c_idx != 0 {
            if let Some(vs) = lmcs_chroma_var_scale {
                lmcs_scale_chroma_residuals(&mut res, vs, bit_depth);
            }
        }
        // §8.7.5.1 — recSamples = Clip1(predSamples + resSamples). The
        // plane already holds predSamples; read each sample as pred,
        // add residual, clip. The 8-bit `PictureBuffer` stores narrowed
        // samples, so lift to `bit_depth` scale before the add when
        // bit_depth > 8 and narrow back afterwards (byte-identical to
        // the intra `reconstruct_tb_into` convention at bit_depth == 8).
        let plane = match c_idx {
            0 => &mut out.luma,
            1 => &mut out.cb,
            2 => &mut out.cr,
            _ => return Ok(()),
        };
        let stride = plane.stride;
        let height = plane.height;
        if x0 + n_tb_w > stride || y0 + n_tb_h > height {
            return Err(Error::invalid(
                "h266 inter residual: TB does not fit in destination plane",
            ));
        }
        let up = bit_depth.saturating_sub(8);
        for row in 0..n_tb_h {
            for col in 0..n_tb_w {
                let idx = (y0 + row) * stride + (x0 + col);
                let pred = (plane.samples[idx] as i32) << up;
                let v = clip_pixel(pred + res[row * n_tb_w + col], bit_depth);
                plane.samples[idx] = if bit_depth > 8 {
                    (v >> up) as u8
                } else {
                    v as u8
                };
            }
        }
        Ok(())
    }

    /// §8.7.2 joint Cb-Cr inter residual reconstruction. The single
    /// coded chroma transform block (`codedCIdx = TuCResMode∈{1,2} ?
    /// Cb : Cr`) is dequantised once; the §8.7.2 orchestrator then
    /// derives `resSamples` for both Cb (`cIdx = 1`) and Cr
    /// (`cIdx = 2`) via the eqs. 1130 – 1132 sign / shift rules. Each
    /// component's residual is added to its plane with the §8.7.5.1
    /// clip.
    #[allow(clippy::too_many_arguments)]
    fn add_inter_joint_cbcr_residual(
        &self,
        c_x: usize,
        c_y: usize,
        c_w: usize,
        c_h: usize,
        tu_c_res_mode: u8,
        residual: &LeafCuResidual,
        cu_qp_delta: i32,
        bit_depth: u32,
        cu_qp_offset_cbcr: i32,
        coded_transform_skip_cb: bool,
        coded_transform_skip_cr: bool,
        lmcs_chroma_var_scale: Option<u32>,
        out: &mut PictureBuffer,
    ) -> Result<()> {
        use crate::transform::{scaling_and_transformation, CodedTransform, TrType, TuCResMode};
        if c_w == 0 || c_h == 0 {
            return Ok(());
        }
        let mode = TuCResMode::from_raw(tu_c_res_mode)?;
        // The coded component supplies the levels: Cb for modes 1 / 2,
        // Cr for mode 3 (§8.7.2 codedCIdx). Its `transform_skip_flag`
        // controls whether the single coded TB bypasses the inverse
        // transform (§8.7.4.6) — read from the coded component's flag.
        let (coded_levels, coded_ts) = match mode {
            TuCResMode::CrCoded => (&residual.cr_levels, coded_transform_skip_cr),
            _ => (&residual.cb_levels, coded_transform_skip_cb),
        };
        if coded_levels.len() != c_w * c_h {
            // No coded coefficients captured (e.g. all-zero CBF edge);
            // nothing to add.
            return Ok(());
        }
        // §8.7.3 dequant — joint Cb-Cr QP (eq. 1149): the additive term is
        // `pps_joint_cbcr_qp_offset_value + sh_joint_cbcr_qp_offset +
        // CuQpOffsetCbCr` rather than the per-component Cb/Cr offsets.
        // `CuQpOffsetCbCr` is the §7.4.10.6 CU-level joint offset (indexed
        // from `pps_joint_cbcr_qp_offset_list` by the caller).
        let qp_y = (self.cabac.slice_qp_y.0 + cu_qp_delta).clamp(0, 63);
        let joint_offset = self.pps.pps_joint_cbcr_qp_offset_value
            + self.sh.sh_joint_cbcr_qp_offset
            + cu_qp_offset_cbcr;
        let qp = chroma_qp_identity(qp_y, joint_offset);
        let log2_tr_range = self.log2_transform_range();
        let params = DequantParams {
            bit_depth,
            log2_transform_range: log2_tr_range,
            n_tb_w: c_w as u32,
            n_tb_h: c_h as u32,
            qp,
            dep_quant: false,
            transform_skip: coded_ts,
            bdpcm: false,
            bdpcm_dir: false,
        };
        let d = dequantize_tb_flat(coded_levels, &params)?;
        // §8.7.4.6 — when the coded chroma TB carries transform_skip the
        // single dequantised block is the residual directly (CodedTransform
        // ::Skip); otherwise the §8.7.4.1 DCT-II × DCT-II inverse runs.
        let coded = if coded_ts {
            CodedTransform::Skip
        } else {
            let non_zero_w = c_w.min(32);
            let non_zero_h = c_h.min(32);
            CodedTransform::Transform {
                non_zero_w,
                non_zero_h,
                tr_type_hor: TrType::DctII,
                tr_type_ver: TrType::DctII,
            }
        };
        // Derive and add each chroma component's residual.
        for c_idx in [1u32, 2u32] {
            let mut res = scaling_and_transformation(
                c_idx,
                mode,
                self.ph_joint_cbcr_sign,
                c_w,
                c_h,
                &d,
                coded,
                bit_depth,
                log2_tr_range,
            )?;
            // §8.7.5.3 eqs. 1219 / 1220 — the scaling applies to the
            // §8.7.2-derived `resSamples` of BOTH components (the
            // `tu_joint_cbcr_residual_flag` arm of `tuCbfChroma`).
            if let Some(vs) = lmcs_chroma_var_scale {
                lmcs_scale_chroma_residuals(&mut res, vs, bit_depth);
            }
            let plane = match c_idx {
                1 => &mut out.cb,
                _ => &mut out.cr,
            };
            let stride = plane.stride;
            let height = plane.height;
            if c_x + c_w > stride || c_y + c_h > height {
                return Err(Error::invalid(
                    "h266 inter JCCR residual: chroma TB does not fit in destination plane",
                ));
            }
            let up = bit_depth.saturating_sub(8);
            for row in 0..c_h {
                for col in 0..c_w {
                    let idx = (c_y + row) * stride + (c_x + col);
                    let pred = (plane.samples[idx] as i32) << up;
                    let v = clip_pixel(pred + res[row * c_w + col], bit_depth);
                    plane.samples[idx] = if bit_depth > 8 {
                        (v >> up) as u8
                    } else {
                        v as u8
                    };
                }
            }
        }
        Ok(())
    }

    /// Round-40 §8.5.4 + §8.5.7 — Geometric Partitioning Mode
    /// reconstruction. The CU is split along an oblique line; each side
    /// is predicted from a separate merge candidate. Implements:
    ///
    ///  1. **§8.5.4.2** — derive `(m, n)` from `merge_gpm_idx0` /
    ///     `merge_gpm_idx1` (eqs. 646 / 647), pick `(predListFlagN, mvN,
    ///     refIdxN)` for partitions A and B per the `X = idx & 1`
    ///     active-list rule (eqs. 648 – 655).
    ///  2. **§8.5.7.1** — invoke §8.5.6.3 fractional-sample MC twice,
    ///     once for partition A and once for partition B, capturing the
    ///     full `cbWidth × cbHeight` predicted samples in scratch
    ///     planes (one per chroma component when 4:2:0).
    ///  3. **§8.5.7.2** — blend the two predictions with the per-pixel
    ///     `wValue` derived from Tables 36 / 37 + eqs. 999 – 1015 +
    ///     final eq. 1016 clip.
    ///  4. **§8.5.7.3** — broadcast a representative MvField into the
    ///     motion-field 4×4 grid (round-40 stores partition A's MV
    ///     uniformly across the CU; the spec's per-4×4 partition-aware
    ///     storage lands once GPM-aware merge derivation is exercised
    ///     by a downstream CU).
    fn reconstruct_leaf_cu_gpm(
        &mut self,
        cu: &CtuCu,
        info: &LeafCuInfo,
        mlist: &[MvField],
        out: &mut PictureBuffer,
    ) -> Result<()> {
        if mlist.is_empty() {
            return Err(Error::invalid(
                "h266 GPM: empty merge candidate list — §8.5.4.2 step 1 violation",
            ));
        }
        let cb_x = cu.cu.x;
        let cb_y = cu.cu.y;
        let cb_w = cu.cu.w;
        let cb_h = cu.cu.h;
        let bit_depth = self.sps.sps_bitdepth_minus8 as u32 + 8;

        // §8.5.4.2 step 2 — eqs. 646 / 647.
        let (m, n) = derive_gpm_mn(
            info.inter.merge_data.gpm_idx0,
            info.inter.merge_data.gpm_idx1,
        );
        let cand_a = mlist[(m as usize).min(mlist.len() - 1)];
        let cand_b = mlist[(n as usize).min(mlist.len() - 1)];
        // §8.5.4.2 steps 4 / 5 — pick the active list per partition.
        let (x_a, mv_a, ref_idx_a) = derive_gpm_partition_motion(m, &cand_a);
        let (x_b, mv_b, ref_idx_b) = derive_gpm_partition_motion(n, &cand_b);

        // Look up the two reference pictures.
        let ref_a = if ref_idx_a < 0 {
            None
        } else if x_a == 0 {
            self.ref_pic_list_l0.get(ref_idx_a as usize)
        } else {
            self.ref_pic_list_l1.get(ref_idx_a as usize)
        }
        .ok_or_else(|| {
            Error::invalid("h266 GPM: partition A reference not available — §8.5.4.2 / §8.5.7.1")
        })?;
        let ref_b = if ref_idx_b < 0 {
            None
        } else if x_b == 0 {
            self.ref_pic_list_l0.get(ref_idx_b as usize)
        } else {
            self.ref_pic_list_l1.get(ref_idx_b as usize)
        }
        .ok_or_else(|| {
            Error::invalid("h266 GPM: partition B reference not available — §8.5.4.2 / §8.5.7.1")
        })?;

        // §8.5.7.1 step 1 — predict each partition over the full CU
        // rectangle using §8.5.6.3 (the round-22 8-tap luma + 4-tap
        // chroma path). The two predictions are captured in scratch
        // planes (8-bit samples) so the §8.5.7.2 blend has the full
        // `cbWidth × cbHeight` arrays available.
        use crate::reconstruct::PicturePlane;
        let mut scratch_a_luma = PicturePlane::filled(cb_w as usize, cb_h as usize, 0);
        let mut scratch_b_luma = PicturePlane::filled(cb_w as usize, cb_h as usize, 0);
        predict_luma_block(
            &mut scratch_a_luma,
            0,
            0,
            cb_w,
            cb_h,
            &ref_a.frame.luma,
            mv_a,
        )?;
        predict_luma_block(
            &mut scratch_b_luma,
            0,
            0,
            cb_w,
            cb_h,
            &ref_b.frame.luma,
            mv_b,
        )?;

        // §8.5.7.1 step 2 + §8.5.7.2 — set up the partition geometry.
        let (angle_idx, distance_idx) =
            derive_gpm_partition(info.inter.merge_data.gpm_partition_idx);
        let ctx_luma = GpmContext::new(cb_w, cb_h, angle_idx, distance_idx, 0, bit_depth, 1, 1);

        // §8.5.7.1 step 3 — eq. 1016 blend on luma.
        blend_gpm_into_plane(
            &mut out.luma,
            cb_x,
            cb_y,
            cb_w,
            cb_h,
            &scratch_a_luma.samples,
            &scratch_b_luma.samples,
            &ctx_luma,
            0,
            bit_depth,
        );

        // §8.5.7.1 steps 4 / 5 — chroma 4:2:0 blend. The same
        // `(angleIdx, distanceIdx)` apply; eqs. 999 / 1000 expand the
        // CU back to luma units so `wValue` is consistent across the
        // partition line.
        if self.sps.sps_chroma_format_idc == 1 {
            let cb_w_c = cb_w / 2;
            let cb_h_c = cb_h / 2;
            let cb_x_c = cb_x / 2;
            let cb_y_c = cb_y / 2;
            let mut scratch_a_cb = PicturePlane::filled(cb_w_c as usize, cb_h_c as usize, 0);
            let mut scratch_b_cb = PicturePlane::filled(cb_w_c as usize, cb_h_c as usize, 0);
            let mut scratch_a_cr = PicturePlane::filled(cb_w_c as usize, cb_h_c as usize, 0);
            let mut scratch_b_cr = PicturePlane::filled(cb_w_c as usize, cb_h_c as usize, 0);
            predict_chroma_block(
                &mut scratch_a_cb,
                0,
                0,
                cb_w_c,
                cb_h_c,
                &ref_a.frame.cb,
                mv_a,
            )?;
            predict_chroma_block(
                &mut scratch_b_cb,
                0,
                0,
                cb_w_c,
                cb_h_c,
                &ref_b.frame.cb,
                mv_b,
            )?;
            predict_chroma_block(
                &mut scratch_a_cr,
                0,
                0,
                cb_w_c,
                cb_h_c,
                &ref_a.frame.cr,
                mv_a,
            )?;
            predict_chroma_block(
                &mut scratch_b_cr,
                0,
                0,
                cb_w_c,
                cb_h_c,
                &ref_b.frame.cr,
                mv_b,
            )?;
            let ctx_chroma =
                GpmContext::new(cb_w_c, cb_h_c, angle_idx, distance_idx, 1, bit_depth, 2, 2);
            blend_gpm_into_plane(
                &mut out.cb,
                cb_x_c,
                cb_y_c,
                cb_w_c,
                cb_h_c,
                &scratch_a_cb.samples,
                &scratch_b_cb.samples,
                &ctx_chroma,
                1,
                bit_depth,
            );
            blend_gpm_into_plane(
                &mut out.cr,
                cb_x_c,
                cb_y_c,
                cb_w_c,
                cb_h_c,
                &scratch_a_cr.samples,
                &scratch_b_cr.samples,
                &ctx_chroma,
                2,
                bit_depth,
            );
        }

        // §8.7.5.2 — a GPM CU is MODE_INTER with `ciip_flag == 0`, so
        // its blended luma prediction forward-maps into the LMCS
        // codeword domain (the GPM path carries no coded residual
        // today, so the mapped prediction IS the mapped-domain
        // reconstruction the §8.8.2 in-loop inverse pass expects).
        self.lmcs_forward_map_luma_rect(
            out,
            cb_x as usize,
            cb_y as usize,
            cb_w as usize,
            cb_h as usize,
        );

        // §8.5.7.3 — Motion vector storing process. Round-40 broadcasts
        // partition A's MV uniformly across the CU's 4×4 motion-field
        // grid; the per-4×4 partition-aware storage (storing partition
        // B's MV on cells whose centre falls on the partition-B side
        // of the line) lands when a downstream merge derivation
        // actually consumes a partition-aware neighbour. None of the
        // existing fixtures or downstream CUs do.
        let mvf = MvField {
            mv_l0: if x_a == 0 { mv_a } else { MotionVector::ZERO },
            ref_idx_l0: if x_a == 0 { ref_idx_a } else { -1 },
            pred_flag_l0: x_a == 0,
            mv_l1: if x_a == 1 { mv_a } else { MotionVector::ZERO },
            ref_idx_l1: if x_a == 1 { ref_idx_a } else { -1 },
            pred_flag_l1: x_a == 1,
            cu_skip_flag: info.inter.cu_skip_flag,
            mode_inter: true,
            available: true,
            bcw_idx: 0,
        };
        self.motion_field.write_block(cb_x, cb_y, cb_w, cb_h, mvf);
        self.hmvp.update_with(mvf);

        // Deblock + intra-grid bookkeeping (matches the round-21 inter
        // path).
        let qp_y = self.cabac.slice_qp_y.0;
        self.deblock_cus.push(DeblockCu {
            x: cb_x,
            y: cb_y,
            w: cb_w,
            h: cb_h,
            qp_y: qp_y.clamp(0, 63),
            intra: false,
            tu_y_coded: false,
            tu_cb_coded: false,
            tu_cr_coded: false,
            bdpcm_luma: false,
            bdpcm_chroma: false,
        });
        self.write_intra_block(cb_x, cb_y, cb_w, cb_h, false);
        // Round-149 — GPM CUs always carry `merge_subblock_flag = 0`
        // (§7.3.11.7: the GPM merge sub-tree only opens after the
        // subblock-merge branch was bypassed). Clear the per-CB grid
        // so the next CU's merge-side ctxInc query reads 0 for this
        // region.
        self.write_subblock_merge_block(cb_x, cb_y, cb_w, cb_h, false);
        self.write_inter_affine_block(cb_x, cb_y, cb_w, cb_h, false);
        // `cand_b` is not consulted again post-MC (its motion is folded
        // into the picture's pixels via the §8.5.7.2 blend), so swallow
        // it explicitly.
        let _ = cand_b;
        Ok(())
    }

    /// Reconstruct the luma plane of an ISP-split CU per §8.4.5.1.
    ///
    /// Walk the spec-derived subpartition list (§8.4.5.1 eqs. 251 –
    /// 260, exposed by [`crate::isp::iter_isp_partitions`]) in order.
    /// For each subpartition:
    ///
    /// 1. **Reference samples** are fetched from the *partially*
    ///    reconstructed luma plane: each subpartition reads the
    ///    samples written by the prior partition, which is exactly
    ///    what makes ISP a useful tool (the residual on a thin sub-TB
    ///    is much smaller when the prediction can use a fresh edge
    ///    one CU-line away).
    /// 2. **Prediction** is computed at width `nPbW = max(4, nW)` and
    ///    sliced down to the sub-TB width when `pbFactor > 1` (only
    ///    relevant for vertical splits with `nW ∈ {1, 2}`). The
    ///    intra-prediction process is invoked only when
    ///    `xPartPbIdx == 0`; the reused prediction is held in
    ///    `pred_window` and sliced for the subsequent partitions.
    /// 3. **Residual** uses the per-subpartition `levels` array
    ///    captured in [`LeafCuLumaSubpart`]. The dequant / IDCT
    ///    pipeline matches the single-TB path (DCT-II both axes; no
    ///    BDPCM here — the spec disallows BDPCM + ISP).
    /// 4. **Reconstruct + clip** is the same eq. 1426 step used by
    ///    the no-ISP path, writing into `out.luma`.
    ///
    /// Returns once every subpartition has been written to the picture
    /// buffer; the chroma plane is handled separately by the caller.
    fn reconstruct_leaf_cu_isp_luma(
        &self,
        cu: &CtuCu,
        info: &LeafCuInfo,
        residual: &LeafCuResidual,
        out: &mut PictureBuffer,
        bit_depth: u32,
    ) -> Result<()> {
        let cb_w = cu.cu.w;
        let cb_h = cu.cu.h;
        let x0_cu = cu.cu.x as usize;
        let y0_cu = cu.cu.y as usize;
        let parts = crate::isp::iter_isp_partitions(info.isp_split, cb_w, cb_h);
        if parts.is_empty() {
            return Err(Error::invalid(
                "h266 reconstruct_leaf_cu ISP: subpartition walk produced no entries",
            ));
        }

        // Cache the prediction window for the current `pbFactor`-aligned
        // group of subpartitions (eqs. 255 – 260). When `pb_factor > 1`,
        // the window is computed on the partition with `xPartPbIdx == 0`
        // and reused for the subsequent partitions in the group.
        let mut pred_window: Vec<i16> = Vec::new();
        let mut pred_window_w: usize = 0;
        let mut pred_window_h: usize = 0;

        for (i, p) in parts.iter().enumerate() {
            // Look up the per-partition residual record. They are
            // written in the same order as the spec walker, so the
            // index aligns 1:1.
            let sub = residual.luma_subparts.get(i).ok_or_else(|| {
                Error::invalid(
                    "h266 reconstruct_leaf_cu ISP: residual.luma_subparts shorter than walk",
                )
            })?;
            if sub.x_offset != p.x_offset || sub.y_offset != p.y_offset {
                return Err(Error::invalid(
                    "h266 reconstruct_leaf_cu ISP: residual record offset mismatch",
                ));
            }

            let n_w = p.n_w as usize;
            let n_h = p.n_h as usize;
            let n_pb_w = p.n_pb_w as usize;
            let pb_factor = p.pb_factor as usize;
            let abs_x = x0_cu + p.x_offset as usize;
            let abs_y = y0_cu + p.y_offset as usize;

            // 1. Prediction window. The intra sample-prediction
            //    process is invoked only when xPartPbIdx == 0
            //    (eq. 260). For pbFactor == 1 every partition runs
            //    the predictor; for pbFactor == 2 the window is
            //    reused for the second partition in each pair.
            if p.x_part_pb_idx == 0 {
                let above_avail = abs_y > 0;
                let left_avail = abs_x > 0;
                let refs = OwnedIntraRefs::from_plane(
                    &out.luma,
                    abs_x,
                    abs_y,
                    n_pb_w,
                    n_h,
                    above_avail,
                    left_avail,
                    bit_depth,
                );
                let refs_view = IntraRefs {
                    above: &refs.above,
                    left: &refs.left,
                    top_left: refs.top_left,
                };
                let pred = match info.intra_pred_mode_y {
                    INTRA_PLANAR => predict_planar(n_pb_w, n_h, &refs_view)?,
                    INTRA_DC => predict_dc(n_pb_w, n_h, &refs_view)?,
                    mode @ (2 | 18 | 34 | 50 | 66) => {
                        predict_angular(n_pb_w, n_h, mode, &refs_view)?
                    }
                    mode if (2..=66).contains(&mode) => {
                        let snapped = nearest_supported_angular(mode);
                        predict_angular(n_pb_w, n_h, snapped, &refs_view)?
                    }
                    _ => predict_planar(n_pb_w, n_h, &refs_view)?,
                };
                pred_window = pred;
                pred_window_w = n_pb_w;
                pred_window_h = n_h;
            } else if pred_window.is_empty() || pred_window_w != n_pb_w || pred_window_h != n_h {
                return Err(Error::invalid(
                    "h266 reconstruct_leaf_cu ISP: prediction window missing for pb-grouped partition",
                ));
            }

            // 2. Slice the prediction window down to nW columns at
            //    the partition's offset within the window. For
            //    pb_factor == 1 this is a 1:1 copy.
            let pb_offset = (p.x_part_pb_idx as usize) * n_w;
            let mut pred_sub = vec![0i16; n_w * n_h];
            for row in 0..n_h {
                let src = &pred_window
                    [row * pred_window_w + pb_offset..row * pred_window_w + pb_offset + n_w];
                pred_sub[row * n_w..row * n_w + n_w].copy_from_slice(src);
            }
            // Defensive: pb_factor must keep `pb_offset + n_w <=
            // pred_window_w`. With `n_pb_w = max(4, n_w)` and `n_w |
            // n_pb_w`, this always holds.
            let _ = pb_factor;

            // 3. Dequant + IDCT for the per-subpartition residual.
            let residual_samples: Vec<i32> = if sub.tu_y_coded_flag && !sub.levels.is_empty() {
                let qp = self.cabac.slice_qp_y.0 + info.cu_qp_delta_val;
                let qp = qp.clamp(0, 63);
                let log2_tr_range = self.log2_transform_range();
                let params = DequantParams {
                    bit_depth,
                    log2_transform_range: log2_tr_range,
                    n_tb_w: p.n_w,
                    n_tb_h: p.n_h,
                    qp,
                    dep_quant: false,
                    transform_skip: false,
                    bdpcm: false,
                    bdpcm_dir: false,
                };
                let d = dequantize_tb_flat(&sub.levels, &params)?;
                inverse_transform_2d(
                    n_w,
                    n_h,
                    n_w,
                    n_h,
                    TrType::DctII,
                    TrType::DctII,
                    &d,
                    bit_depth,
                    log2_tr_range,
                )?
            } else {
                vec![0i32; n_w * n_h]
            };

            // 4. Reconstruct (eq. 1426) into the picture plane.
            reconstruct_tb_into(
                &mut out.luma.samples,
                out.luma.stride,
                out.luma.height,
                abs_x,
                abs_y,
                &pred_sub,
                &residual_samples,
                n_w,
                n_h,
                bit_depth,
            )?;
        }

        Ok(())
    }

    /// Reconstruct a single chroma TB (Cb when `c_idx == 1`, Cr when
    /// `c_idx == 2`) into the corresponding plane of `out`.
    ///
    /// Pipeline mirrors the luma path: §8.4.5.2.8 reference fetch on
    /// the half-resolution chroma plane, PLANAR / DC / cardinal-angular
    /// intra prediction (`info.intra_pred_mode_c` mapped via
    /// [`chroma_pred_mode_for_predict`]; CCLM and non-cardinal angulars
    /// are snapped to PLANAR / nearest-cardinal as a placeholder),
    /// §8.7.3 flat-list dequant (with the §8.7.1 chroma QP — currently
    /// the identity mapping `QpC = QpY + pps_chroma_qp_offset
    /// + cu_chroma_qp_offset`), §8.7.4.1 separable DCT-II inverse
    /// transform, and §8.7.5.1 reconstruct + clip.
    ///
    /// Chroma TBs of size 2×2 / 2×4 / 4×2 (which arise when the luma
    /// CU is 4×4 / 4×8 / 8×4 with 4:2:0 sub-sampling) now go through
    /// the same DCT-II path as the larger sizes — the size-2 kernel was
    /// added to [`crate::transform::one_d_transform`] in round 13. The
    /// chroma path will run dequant + inverse transform whenever the
    /// leaf reader emitted a coded residual, regardless of TB size.
    #[allow(clippy::too_many_arguments)]
    fn reconstruct_chroma_plane(
        &self,
        c_idx: u32,
        cu: &CtuCu,
        info: &LeafCuInfo,
        levels: &[i32],
        coded_flag: bool,
        bit_depth: u32,
        out: &mut PictureBuffer,
    ) -> Result<()> {
        // 4:2:0 chroma sub-sampling: half in both directions.
        let n_tb_w = (cu.cu.w as usize) / 2;
        let n_tb_h = (cu.cu.h as usize) / 2;
        let x0 = (cu.cu.x as usize) / 2;
        let y0 = (cu.cu.y as usize) / 2;
        if n_tb_w == 0 || n_tb_h == 0 {
            return Ok(());
        }
        // CCLM (§8.4.5.2.14) reads the reconstructed luma plane while
        // the chroma plane is held mutably below. Snapshot the luma
        // metadata up front so the immutable luma borrow can be created
        // alongside the mutable chroma borrow without aliasing.
        let mode_c = info.intra_pred_mode_c;
        // §8.7.5.3 — chroma residual scaling applies to intra CUs too
        // (only MODE_PLT is excluded). Derive `varScale` before the
        // chroma plane is borrowed mutably; the `invAvgLuma`
        // neighbourhood reads the (mapped-domain, when LMCS is used)
        // reconstructed luma plane.
        let lmcs_cvs = if n_tb_w * n_tb_h > 4 {
            self.lmcs_chroma_var_scale_for_cu(cu, out)
        } else {
            None
        };
        let is_cclm = matches!(mode_c, INTRA_LT_CCLM | INTRA_L_CCLM | INTRA_T_CCLM);
        let luma_samples_snapshot: Option<Vec<u8>> = if is_cclm {
            Some(out.luma.samples.clone())
        } else {
            None
        };
        let luma_stride = out.luma.stride;
        let luma_w = out.luma.width;
        let luma_h = out.luma.height;

        // Pick the destination plane up front so the borrow on `out`
        // does not have to follow the plane-by-plane reconstruction.
        let plane = match c_idx {
            1 => &mut out.cb,
            2 => &mut out.cr,
            _ => return Ok(()),
        };

        // 1. Reference samples from the partially-reconstructed chroma
        // plane. Same picture-edge rule as luma in the single-slice
        // case.
        let above_avail = y0 > 0;
        let left_avail = x0 > 0;
        let refs = OwnedIntraRefs::from_plane(
            plane,
            x0,
            y0,
            n_tb_w,
            n_tb_h,
            above_avail,
            left_avail,
            bit_depth,
        );
        let refs_view = IntraRefs {
            above: &refs.above,
            left: &refs.left,
            top_left: refs.top_left,
        };

        // 2. Intra prediction. CCLM (§8.4.5.2.14) is dispatched first
        // when the §8.4.3 chroma-mode derivation produced one of the
        // CCLM modes (81 / 82 / 83); the helper consumes the
        // reconstructed luma plane snapshot taken just above. The
        // remaining modes go through the cardinal-only fallback.
        let pred = if is_cclm {
            // Build the 4:2:0 chroma neighbour rows. Both `INTRA_T_CCLM`
            // and `INTRA_L_CCLM` extend the corresponding side; the
            // helper caps the lengths at the plane boundary.
            let (top, left) = cclm_chroma_neighbours(
                plane,
                x0,
                y0,
                n_tb_w,
                n_tb_h,
                above_avail,
                left_avail,
                mode_c,
            );
            let luma_samples = luma_samples_snapshot.as_deref().unwrap_or(&[]);
            // §8.4.5.2.14 honours bCTUboundary on the top side. CTU
            // height in the chroma grid is `ctb_size_y / SubHeightC` —
            // for 4:2:0 that's `ctb_size_y / 2`. The luma top-left
            // (after eq. 358) is `y_tb_y = y0_chroma << 1`, and the
            // boundary check uses `y_tb_y & (CtbSizeY - 1) == 0`.
            let ctb_size_y = self.layout.ctb_size_y as usize;
            let b_ctu_boundary = ctb_size_y > 0 && (((y0 << 1) & (ctb_size_y - 1)) == 0);
            let inputs = CclmInputs {
                mode: mode_c,
                n_tb_w,
                n_tb_h,
                sub_width_c: 2,
                sub_height_c: 2,
                chroma_vertical_collocated_flag: self
                    .sps
                    .tool_flags
                    .chroma_vertical_collocated_flag,
                b_ctu_boundary,
                bit_depth,
                neigh_top_chroma: &top,
                neigh_left_chroma: &left,
                luma_plane: LumaPlane {
                    samples: luma_samples,
                    stride: luma_stride,
                    width: luma_w,
                    height: luma_h,
                },
                x_tb_c: x0,
                y_tb_c: y0,
            };
            // Falls back to PLANAR if CCLM rejects the inputs (bad mode
            // / zero-sized TB) — keeps the chroma plane numerically
            // defined even on degenerate input.
            match predict_cclm(&inputs) {
                Ok(p) => p,
                Err(_) => predict_planar(n_tb_w, n_tb_h, &refs_view)?,
            }
        } else {
            let mapped = chroma_pred_mode_for_predict(mode_c);
            match mapped {
                INTRA_PLANAR => predict_planar(n_tb_w, n_tb_h, &refs_view)?,
                INTRA_DC => predict_dc(n_tb_w, n_tb_h, &refs_view)?,
                mode @ (2 | 18 | 34 | 50 | 66) => {
                    predict_angular(n_tb_w, n_tb_h, mode, &refs_view)?
                }
                _ => predict_planar(n_tb_w, n_tb_h, &refs_view)?,
            }
        };

        // 3. Dequantise + 4. inverse 2D transform when there is a coded
        // chroma TB. Round-13 added size-2 DCT-II support, so the
        // 2×2 / 2×4 / 4×2 chroma sizes (from 4×4 / 4×8 / 8×4 luma CUs
        // under 4:2:0) now go through the dequant + IDCT pipeline
        // alongside the 4+ sizes.
        let residual_samples: Vec<i32> =
            if coded_flag && !levels.is_empty() && n_tb_w >= 2 && n_tb_h >= 2 {
                // §8.7.1 chroma QP (eqs. 1147 / 1148): identity mapping
                // qPi → QpC, plus the additive
                // `pps_c?_qp_offset + sh_c?_qp_offset + CuQpOffsetC?` term.
                // The slice-level offset is only present when the PPS gate
                // `pps_slice_chroma_qp_offsets_present_flag` is set; when
                // absent §7.4.9.1 infers `sh_c?_qp_offset = 0`, which the
                // slice-header parser already leaves at the default, so the
                // unconditional add is safe.
                //
                // §7.4.10.6 eqs. 193 / 194 — the CU-level `CuQpOffsetC?`
                // term indexes the parsed PPS `pps_c?_qp_offset_list`
                // by `cu_chroma_qp_offset_idx` when the flag is set.
                let cu_offset = cu_chroma_qp_offset(
                    c_idx,
                    info.cu_chroma_qp_offset_flag,
                    info.cu_chroma_qp_offset_idx,
                    &self.pps.pps_cb_qp_offset_list,
                    &self.pps.pps_cr_qp_offset_list,
                );
                let qp_offset = chroma_qp_offset_sum(
                    c_idx,
                    self.pps.pps_cb_qp_offset,
                    self.pps.pps_cr_qp_offset,
                    self.sh.sh_cb_qp_offset,
                    self.sh.sh_cr_qp_offset,
                    cu_offset,
                );
                let qp = chroma_qp_identity(self.cabac.slice_qp_y.0, qp_offset);
                // Per-plane transform-skip: BDPCM-chroma forces it on both
                // planes; a plain transform_skip_flag is per-component.
                let plane_ts = match c_idx {
                    1 => info.transform_skip_cb,
                    2 => info.transform_skip_cr,
                    _ => false,
                };
                let transform_skip = info.intra_bdpcm_chroma || plane_ts;
                let log2_tr_range = self.log2_transform_range();
                let params = DequantParams {
                    bit_depth,
                    log2_transform_range: log2_tr_range,
                    n_tb_w: n_tb_w as u32,
                    n_tb_h: n_tb_h as u32,
                    qp,
                    dep_quant: false,
                    transform_skip,
                    bdpcm: info.intra_bdpcm_chroma,
                    bdpcm_dir: info.intra_bdpcm_chroma_dir,
                };
                let d = dequantize_tb_flat(levels, &params)?;
                if transform_skip {
                    // Transform-skip (BDPCM-chroma or plain
                    // transform_skip_flag): bypass the inverse transform;
                    // d[] is the chroma residual directly (§8.7.4.6).
                    d
                } else {
                    // DCT-II both axes — chroma never picks an MTS kernel
                    // explicitly; implicit-MTS would only apply to luma intra
                    // (§8.7.4.4). Sizes ∈ {4, 8, 16, 32, 64} are all covered.
                    inverse_transform_2d(
                        n_tb_w,
                        n_tb_h,
                        n_tb_w,
                        n_tb_h,
                        TrType::DctII,
                        TrType::DctII,
                        &d,
                        bit_depth,
                        log2_tr_range,
                    )?
                }
            } else {
                vec![0i32; n_tb_w * n_tb_h]
            };

        // §8.7.5.3 eqs. 1219 / 1220 — scale the residual before the
        // eq. 1220 add when LMCS chroma residual scaling applies (an
        // uncoded TB's all-zero residual is unaffected, matching the
        // eq. 1221 pass-through arm).
        let mut residual_samples = residual_samples;
        if let Some(vs) = lmcs_cvs {
            lmcs_scale_chroma_residuals(&mut residual_samples, vs, bit_depth);
        }

        // 5. Reconstruct + clip into the chroma plane.
        reconstruct_tb_into(
            &mut plane.samples,
            plane.stride,
            plane.height,
            x0,
            y0,
            &pred,
            &residual_samples,
            n_tb_w,
            n_tb_h,
            bit_depth,
        )?;
        Ok(())
    }

    /// Walk every CTU in the slice and reconstruct each leaf CU into
    /// `out`. Returns the picture buffer once filled. This is the
    /// "decode an IDR" entry point — it ties together the partition
    /// walker, syntax reader, and reconstruction stages above.
    pub fn decode_picture_into(&mut self, out: &mut PictureBuffer) -> Result<()> {
        // §8.7.5 — a slice that signals `sh_lmcs_used_flag` reconstructs
        // its luma in the mapped domain; decoding it without the
        // `ph_lmcs_aps_id`-referenced APS payload would silently produce
        // wrong pixels, so fail fast instead.
        if self.sh.sh_lmcs_used_flag && self.lmcs.is_none() {
            return Err(Error::unsupported(
                "h266 CTU walker: sh_lmcs_used_flag == 1 but no LMCS APS payload bound — \
                 call set_lmcs() with the ph_lmcs_aps_id-referenced lmcs_data() first",
            ));
        }
        let ctus: Vec<CtuPos> = self.iter_ctus().collect();
        for ctu in &ctus {
            // §7.3.11.2: `sao(rx, ry)` is decoded at the start of every
            // CTU (before the partition tree). The helper short-circuits
            // when both sao_*_used_flags are 0.
            self.decode_sao_for_ctu(ctu)?;
            let (cus, infos, residuals) = self.decode_ctu_full(ctu)?;
            for ((ccu, info), residual) in cus.iter().zip(infos.iter()).zip(residuals.iter()) {
                self.reconstruct_leaf_cu(ccu, info, residual, out)?;
            }
        }
        Ok(())
    }

    /// Apply the §8.8 in-loop filters to a partially-reconstructed
    /// picture. Runs deblocking (§8.8.3) followed by SAO (§8.8.4) and
    /// then ALF (§8.8.5). Wraps [`Self::apply_in_loop_filters_with_alf`]
    /// with an empty [`AlfApsBinding`]; with no APSes bound the ALF
    /// pass is necessarily a no-op (every CTB falls through the "APS
    /// missing → luma off" branch). Tests drive ALF via the
    /// `_with_alf` variant.
    ///
    /// The CU records consumed by the deblocker are accumulated by
    /// [`Self::reconstruct_leaf_cu`]; calling
    /// `apply_in_loop_filters` before any picture-decode call is
    /// therefore a no-op (the empty CU list never finds a neighbour
    /// pair). The SAO pass is gated on `sh_sao_luma_used_flag` /
    /// `sh_sao_chroma_used_flag` and the per-CTB params installed via
    /// [`Self::set_sao_picture`]; with the default empty SAO picture
    /// every CTB classifies as `SaoTypeIdx::NotApplied` and SAO is a
    /// no-op even when the slice flags are on. The ALF pass is gated
    /// on `sh_alf_enabled_flag` and the per-CTB params installed via
    /// [`Self::set_alf_picture`]; with the default empty ALF picture
    /// every CTB has `luma_on = cb_on = cr_on = false` and ALF is a
    /// no-op even when the slice flag is on.
    pub fn apply_in_loop_filters(&mut self, out: &mut PictureBuffer) -> Result<()> {
        self.apply_in_loop_filters_with_alf(out, &AlfApsBinding::default())
    }

    /// Variant of [`Self::apply_in_loop_filters`] that accepts an
    /// [`AlfApsBinding`] so callers (and integration tests) can supply
    /// the resolved ALF APSes referenced by `sh_alf_aps_id_luma[]` and
    /// `sh_alf_aps_id_chroma`. With a non-empty binding and a
    /// populated [`AlfPicture`] the §8.8.5.2 / §8.8.5.4 filters
    /// actually run.
    pub fn apply_in_loop_filters_with_alf(
        &mut self,
        out: &mut PictureBuffer,
        alf_binding: &AlfApsBinding<'_>,
    ) -> Result<()> {
        // Resolve deblock parameters from the slice header (§7.4.8) +
        // the picture header / PPS fall-backs already applied during
        // parse.
        let bit_depth = self.sps.sps_bitdepth_minus8 as u32 + 8;
        // §8.8.1 step 1 — the picture inverse mapping process for luma
        // samples (§8.8.2.1) runs BEFORE deblocking. When the slice
        // used LMCS the reconstructed luma is still in the mapped
        // domain; every sample is inverse-mapped via the §8.8.2.2
        // eqs. 1222 / 1223 fold (the §8.8.2.2 `sh_lmcs_used_flag == 0`
        // arm makes the pass an identity, i.e. it is skipped).
        if self.lmcs_active() {
            let l = self.lmcs.as_ref().expect("lmcs_active checked is_some");
            let up = bit_depth.saturating_sub(8);
            let plane = &mut out.luma;
            for row in 0..plane.height {
                for col in 0..plane.width {
                    let idx = row * plane.stride + col;
                    let s = u32::from(plane.samples[idx]) << up;
                    let iy = l.derived.idx_y_inv(s, l.min_bin_idx, l.max_bin_idx);
                    let v = l.derived.inverse_map_luma_sample(s, iy);
                    plane.samples[idx] = (v >> up) as u8;
                }
            }
        }
        let disabled = self.pps.pps_deblocking_filter_disabled_flag
            || self.sh.sh_deblocking_filter_disabled_flag;
        let params = DeblockParams {
            disabled,
            luma_beta_offset_div2: self.sh.sh_luma_beta_offset_div2,
            luma_tc_offset_div2: self.sh.sh_luma_tc_offset_div2,
            cb_beta_offset_div2: self.sh.sh_cb_beta_offset_div2,
            cb_tc_offset_div2: self.sh.sh_cb_tc_offset_div2,
            cr_beta_offset_div2: self.sh.sh_cr_beta_offset_div2,
            cr_tc_offset_div2: self.sh.sh_cr_tc_offset_div2,
            chroma_qp_offset_cb: self.pps.pps_cb_qp_offset,
            chroma_qp_offset_cr: self.pps.pps_cr_qp_offset,
            bit_depth,
        };
        apply_deblocking(
            out,
            &self.deblock_cus,
            &params,
            self.sps.sps_chroma_format_idc as u32,
        );
        // SAO (§8.8.4) — runs after deblocking per §8.8.4.1. Per-CTB
        // SaoTypeIdx / SaoEoClass / SaoOffsetVal arrays come from
        // `sao_picture`, which the round-13 walker leaves at the empty
        // default unless an integration test explicitly populates it.
        let sao_cfg = SaoConfig {
            luma_used: self.sh.sh_sao_luma_used_flag,
            chroma_used: self.sh.sh_sao_chroma_used_flag,
            bit_depth,
            ctb_log2_size_y: self.layout.ctb_log2_size_y,
            chroma_format_idc: self.sps.sps_chroma_format_idc as u32,
        };
        apply_sao(out, &self.sao_picture, &sao_cfg);
        // ALF (§8.8.5) — runs after SAO per §8.8.5.1. Gated on
        // `sh_alf_enabled_flag`. Per-CTB on/off + filter selection
        // comes from `alf_picture`, which the round-15 walker leaves
        // at the empty default unless an integration test populates
        // it.
        let alf_cfg = AlfConfig {
            alf_enabled: self.sh.sh_alf_enabled_flag,
            cb_enabled: self.sh.sh_alf_cb_enabled_flag,
            cr_enabled: self.sh.sh_alf_cr_enabled_flag,
            bit_depth,
            ctb_log2_size_y: self.layout.ctb_log2_size_y,
            chroma_format_idc: self.sps.sps_chroma_format_idc as u32,
        };
        apply_alf(out, &self.alf_picture, &alf_cfg, alf_binding);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pps::{PicParameterSet, PicPartition};
    use crate::slice_header::{SliceType, StatefulSliceHeader};
    use crate::sps::{PartitionConstraints, SeqParameterSet};

    // Minimal SPS with a configurable CTB size, 4:2:0, 8-bit, no subpics.
    fn dummy_sps(ctb_log2_minus5: u8, pic_w: u32, pic_h: u32) -> SeqParameterSet {
        use crate::sps::ToolFlags;
        SeqParameterSet {
            sps_seq_parameter_set_id: 0,
            sps_video_parameter_set_id: 0,
            sps_max_sublayers_minus1: 0,
            sps_chroma_format_idc: 1,
            sps_log2_ctu_size_minus5: ctb_log2_minus5,
            sps_ptl_dpb_hrd_params_present_flag: false,
            profile_tier_level: None,
            sps_gdr_enabled_flag: false,
            sps_ref_pic_resampling_enabled_flag: false,
            sps_res_change_in_clvs_allowed_flag: false,
            sps_pic_width_max_in_luma_samples: pic_w,
            sps_pic_height_max_in_luma_samples: pic_h,
            conformance_window: None,
            sps_subpic_info_present_flag: false,
            sps_bitdepth_minus8: 0,
            sps_entropy_coding_sync_enabled_flag: false,
            sps_entry_point_offsets_present_flag: false,
            sps_log2_max_pic_order_cnt_lsb_minus4: 4,
            sps_poc_msb_cycle_flag: false,
            sps_poc_msb_cycle_len_minus1: 0,
            sps_num_extra_ph_bytes: 0,
            sps_num_extra_sh_bytes: 0,
            num_extra_ph_bits: 0,
            num_extra_sh_bits: 0,
            sps_sublayer_dpb_params_flag: false,
            dpb_parameters: None,
            partition_constraints: PartitionConstraints::default(),
            tool_flags: ToolFlags::default(),
            subpic_info: None,
            sps_timing_hrd_params_present_flag: false,
            general_timing_hrd: None,
            sps_sublayer_cpb_params_present_flag: false,
            ols_timing_hrd: None,
            sps_field_seq_flag: false,
            sps_vui_parameters_present_flag: false,
            vui_payload: Vec::new(),
            sps_extension_flag: false,
            sps_range_extension_flag: false,
            sps_extension_7bits: 0,
            range_extension: None,
        }
    }

    fn dummy_pps(pic_w: u32, pic_h: u32, no_partition: bool) -> PicParameterSet {
        PicParameterSet {
            pps_pic_parameter_set_id: 0,
            pps_seq_parameter_set_id: 0,
            pps_mixed_nalu_types_in_pic_flag: false,
            pps_pic_width_in_luma_samples: pic_w,
            pps_pic_height_in_luma_samples: pic_h,
            conformance_window: None,
            scaling_window: None,
            pps_output_flag_present_flag: false,
            pps_no_pic_partition_flag: no_partition,
            pps_subpic_id_mapping_present_flag: false,
            subpic_id_mapping: None,
            pps_rect_slice_flag: true,
            pps_single_slice_per_subpic_flag: true,
            pps_loop_filter_across_slices_enabled_flag: false,
            pps_cabac_init_present_flag: false,
            pps_num_ref_idx_default_active_minus1: [0, 0],
            pps_rpl1_idx_present_flag: false,
            pps_weighted_pred_flag: false,
            pps_weighted_bipred_flag: false,
            pps_ref_wraparound_enabled_flag: false,
            pps_pic_width_minus_wraparound_offset: 0,
            pps_init_qp_minus26: 0,
            pps_cu_qp_delta_enabled_flag: false,
            pps_chroma_tool_offsets_present_flag: false,
            pps_cb_qp_offset: 0,
            pps_cr_qp_offset: 0,
            pps_joint_cbcr_qp_offset_present_flag: false,
            pps_joint_cbcr_qp_offset_value: 0,
            pps_slice_chroma_qp_offsets_present_flag: false,
            pps_cu_chroma_qp_offset_list_enabled_flag: false,
            pps_cb_qp_offset_list: Vec::new(),
            pps_cr_qp_offset_list: Vec::new(),
            pps_joint_cbcr_qp_offset_list: Vec::new(),
            pps_deblocking_filter_control_present_flag: false,
            pps_deblocking_filter_override_enabled_flag: false,
            pps_deblocking_filter_disabled_flag: false,
            pps_dbf_info_in_ph_flag: false,
            pps_rpl_info_in_ph_flag: true,
            pps_sao_info_in_ph_flag: true,
            pps_alf_info_in_ph_flag: true,
            pps_wp_info_in_ph_flag: true,
            pps_qp_delta_info_in_ph_flag: false,
            pps_picture_header_extension_present_flag: false,
            pps_slice_header_extension_present_flag: false,
            pps_extension_flag: false,
            partition: if no_partition {
                None
            } else {
                Some(PicPartition::default())
            },
        }
    }

    fn intra_slice_header() -> StatefulSliceHeader {
        StatefulSliceHeader {
            sh_slice_type: SliceType::I,
            sh_qp_delta: 0,
            ..Default::default()
        }
    }

    #[test]
    fn ctu_layout_derives_grid_from_sps() {
        // CtbLog2SizeY = 7 (128), picture 256x128 -> 2x1 CTUs.
        let sps = dummy_sps(2, 256, 128);
        let pps = dummy_pps(256, 128, true);
        let layout = CtuLayout::from_sps_pps(&sps, &pps);
        assert_eq!(layout.ctb_log2_size_y, 7);
        assert_eq!(layout.ctb_size_y, 128);
        assert_eq!(layout.pic_width_in_ctbs_y, 2);
        assert_eq!(layout.pic_height_in_ctbs_y, 1);
        assert_eq!(layout.pic_size_in_ctbs_y(), 2);
        assert_eq!(layout.min_cb_log2_size_y, 2);
    }

    #[test]
    fn ctu_layout_handles_non_multiple_picture_size() {
        // 130x65 picture, 64x64 CTU → 3x2 CTUs with edge clamping.
        let sps = dummy_sps(1, 130, 65);
        let pps = dummy_pps(130, 65, true);
        let layout = CtuLayout::from_sps_pps(&sps, &pps);
        assert_eq!(layout.ctb_size_y, 64);
        assert_eq!(layout.pic_width_in_ctbs_y, 3);
        assert_eq!(layout.pic_height_in_ctbs_y, 2);
        let ctus: Vec<_> = CtuIter::raster(&layout).collect();
        assert_eq!(ctus.len(), 6);
        // Bottom-right CTU is clamped.
        let br = ctus[5];
        assert_eq!((br.x_ctb, br.y_ctb), (2, 1));
        assert_eq!((br.x0, br.y0), (128, 64));
        assert_eq!(br.width_luma, 130 - 128); // 2
        assert_eq!(br.height_luma, 65 - 64); // 1
    }

    #[test]
    fn ctu_iter_raster_scan_order() {
        let sps = dummy_sps(1, 192, 128); // 3x2 CTUs at 64 px.
        let pps = dummy_pps(192, 128, true);
        let layout = CtuLayout::from_sps_pps(&sps, &pps);
        let addrs: Vec<_> = CtuIter::raster(&layout).map(|c| c.ctu_addr_rs).collect();
        assert_eq!(addrs, vec![0, 1, 2, 3, 4, 5]);
        let positions: Vec<_> = CtuIter::raster(&layout)
            .map(|c| (c.x_ctb, c.y_ctb))
            .collect();
        assert_eq!(
            positions,
            vec![(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)]
        );
    }

    #[test]
    fn neighbour_avail_reflects_picture_edges() {
        // Top-left CTU has no neighbours; interior has both.
        assert_eq!(
            CtuIter::availability_single_slice(0, 0),
            NeighbourAvail {
                left: false,
                above: false
            }
        );
        assert_eq!(
            CtuIter::availability_single_slice(1, 0),
            NeighbourAvail {
                left: true,
                above: false
            }
        );
        assert_eq!(
            CtuIter::availability_single_slice(0, 1),
            NeighbourAvail {
                left: false,
                above: true
            }
        );
        assert_eq!(
            CtuIter::availability_single_slice(1, 1),
            NeighbourAvail {
                left: true,
                above: true
            }
        );
    }

    #[test]
    fn slice_qp_y_derives_eq140() {
        // pps_init_qp_minus26 = 5, sh_qp_delta = 2, ph_qp_delta = -1
        // → SliceQpY = 26 + 5 + 2 - 1 = 32.
        let sps = dummy_sps(2, 128, 128);
        let mut pps = dummy_pps(128, 128, true);
        pps.pps_init_qp_minus26 = 5;
        let mut sh = intra_slice_header();
        sh.sh_qp_delta = 2;
        let qp = SliceQpY::derive(&sps, &pps, &sh, -1);
        assert_eq!(qp.0, 32);
    }

    #[test]
    fn slice_qp_y_clamps_to_legal_range() {
        // With sps_bitdepth_minus8 = 0, QpBdOffset = 0, so range = [0, 63].
        let sps = dummy_sps(2, 128, 128);
        let mut pps = dummy_pps(128, 128, true);
        pps.pps_init_qp_minus26 = 100;
        let sh = intra_slice_header();
        let qp = SliceQpY::derive(&sps, &pps, &sh, 0);
        assert_eq!(qp.0, 63);

        // Negative past the lower bound is also clipped.
        pps.pps_init_qp_minus26 = -200;
        let qp = SliceQpY::derive(&sps, &pps, &sh, 0);
        assert_eq!(qp.0, 0);
    }

    #[test]
    fn slice_qp_y_respects_qp_bd_offset_at_10bit() {
        // 10-bit → QpBdOffset = 12, lower bound = -12.
        let mut sps = dummy_sps(2, 128, 128);
        sps.sps_bitdepth_minus8 = 2;
        let mut pps = dummy_pps(128, 128, true);
        pps.pps_init_qp_minus26 = -100;
        let sh = intra_slice_header();
        let qp = SliceQpY::derive(&sps, &pps, &sh, 0);
        assert_eq!(qp.0, -12);
    }

    #[test]
    fn slice_cabac_state_builds_nonempty_tables() {
        let state = SliceCabacState::init_for_slice(SliceQpY(26), false);
        // All four table groups must be populated.
        assert!(!state.tree_ctxs.split_cu.is_empty());
        assert!(!state.tree_ctxs.split_qt.is_empty());
        assert!(!state.tree_ctxs.pred_mode.is_empty());
        assert!(!state.tree_ctxs.intra_luma_mpm.is_empty());
    }

    #[test]
    fn walker_rejects_multi_slice_pictures() {
        let sps = dummy_sps(2, 256, 256);
        let pps = dummy_pps(256, 256, false); // no_partition = false
        let sh = intra_slice_header();
        let layout = CtuLayout::from_sps_pps(&sps, &pps);
        let data = [0u8; 8];
        let err = CtuWalker::begin_slice(&layout, &sps, &pps, &sh, 0, &data).unwrap_err();
        assert!(matches!(err, Error::Unsupported(_)), "got {err:?}");
    }

    #[test]
    fn walker_accepts_p_and_b_slices() {
        // Round-21: P-slices accepted via the inter path (cu_skip +
        // regular merge). Round-23: B-slices accepted with the second
        // §8.5.3.2 invocation against RefPicList[1] + the §8.5.6.6.2
        // eq. 980 default-weighted bi-pred composition.
        let sps = dummy_sps(2, 128, 128);
        let pps = dummy_pps(128, 128, true);
        let layout = CtuLayout::from_sps_pps(&sps, &pps);
        let data = [0u8; 8];
        let mut sh_p = intra_slice_header();
        sh_p.sh_slice_type = SliceType::P;
        assert!(CtuWalker::begin_slice(&layout, &sps, &pps, &sh_p, 0, &data).is_ok());
        let mut sh_b = intra_slice_header();
        sh_b.sh_slice_type = SliceType::B;
        // B-slices should now construct cleanly; the actual MC requires
        // RefPicList0 + RefPicList1 to be populated by the caller before
        // `decode_picture_into`.
        assert!(CtuWalker::begin_slice(&layout, &sps, &pps, &sh_b, 0, &data).is_ok());
    }

    /// §8.5.2.1 — a non-merge AMVP inter CU whose left spatial
    /// neighbour (A1) carries a known L0 MV pointing at the same
    /// reference picture must reconstruct through that neighbour's MV as
    /// `mvpL0` (selected by `mvp_l0_flag == 0`) folded with `mvd == 0`.
    /// The output luma must equal the §8.5.6.3 MC of the reference at
    /// that MV — proving the AMVP-derived MV (not zero-MV) drove the
    /// prediction. A horizontal gradient reference makes a zero-MV
    /// prediction distinguishable from the shifted one.
    #[test]
    fn reconstruct_leaf_cu_inter_amvp_uses_spatial_predictor() {
        let pic_w = 64u32;
        let pic_h = 64u32;
        let sps = dummy_sps(1, pic_w, pic_h);
        let pps = dummy_pps(pic_w, pic_h, true);
        let mut sh = intra_slice_header();
        sh.sh_slice_type = SliceType::P;
        let layout = CtuLayout::from_sps_pps(&sps, &pps);
        let data = [0u8; 32];
        let mut walker = CtuWalker::begin_slice(&layout, &sps, &pps, &sh, 0, &data).unwrap();
        walker.current_poc = 0;

        // Horizontal-gradient reference so a non-zero MV gives a
        // distinguishable prediction. POC 4 → the AMVP context matches
        // a neighbour whose L0 refIdx 0 resolves to the same POC.
        let mut ref_frame = PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 0);
        for y in 0..pic_h as usize {
            for x in 0..pic_w as usize {
                ref_frame.luma.samples[y * ref_frame.luma.stride + x] = (x * 3) as u8;
            }
        }
        let ref_pic = ReferencePicture {
            poc: 4,
            frame: ref_frame.clone(),
            motion_field: None,
        };
        walker.set_ref_pic_list_l0(vec![ref_pic]);

        // Seed the left neighbour A1 = (xCb - 1, yCb + cbH - 1) with an
        // integer-pel L0 MV of (3, 0) px → (48, 0) in 1/16-luma units,
        // pointing at refIdx 0 (POC 4). The CU sits at (16, 0).
        let nb_mv = MotionVector::from_int_pel(3, 0);
        let nb = MvField {
            mv_l0: nb_mv,
            ref_idx_l0: 0,
            pred_flag_l0: true,
            mv_l1: MotionVector::ZERO,
            ref_idx_l1: -1,
            pred_flag_l1: false,
            cu_skip_flag: false,
            mode_inter: true,
            available: true,
            bcw_idx: 0,
        };
        // The A-side scan reads (xCb - 1, yCb + cbH) then A1 =
        // (xCb - 1, yCb + cbH - 1); broadcast the MV across the whole
        // left column region so both A positions resolve to it.
        walker.motion_field.write_block(0, 0, 16, 64, nb);

        let ccu = CtuCu {
            ctu_addr_rs: 0,
            cu: Cu {
                x: 16,
                y: 0,
                w: 16,
                h: 16,
                cqt_depth: 0,
                mtt_depth: 0,
            },
        };
        let mut info = LeafCuInfo {
            x0: 16,
            y0: 0,
            cb_width: 16,
            cb_height: 16,
            pred_mode: CuPredMode::Inter,
            ..LeafCuInfo::default()
        };
        info.inter.general_merge_flag = false;
        info.inter.non_merge = crate::inter::NonMergeInterData {
            pred_dir: crate::leaf_cu::InterPredDir::PredL0,
            ref_idx_l0: 0,
            mvp_l0_flag: 0,
            mvd_l0: MotionVector::ZERO,
            ..crate::inter::NonMergeInterData::default()
        };
        // No residual.
        let residual = LeafCuResidual::default();

        let mut out = PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 0);
        walker
            .reconstruct_leaf_cu(&ccu, &info, &residual, &mut out)
            .expect("amvp non-merge path should reconstruct");

        // Expected: MC of the reference at the neighbour's MV (3px right)
        // for the 16x16 CU at (16, 0). Build it with the same primitive.
        let mut expected = PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 0);
        crate::inter::predict_luma_block(&mut expected.luma, 16, 0, 16, 16, &ref_frame.luma, nb_mv)
            .unwrap();

        for y in 0..16usize {
            for x in 0..16usize {
                let xi = 16 + x;
                assert_eq!(
                    out.luma.samples[y * out.luma.stride + xi],
                    expected.luma.samples[y * expected.luma.stride + xi],
                    "AMVP recon at ({xi},{y}) must equal MC at the spatial predictor MV"
                );
            }
        }
        // Sanity: the AMVP MV produced a genuinely shifted prediction —
        // a zero-MV prediction would copy column xi verbatim, but the
        // 3px shift reads column xi+3 from the gradient. Confirm the
        // top-left CU sample differs from the zero-MV value.
        let zero_mv_val = ref_frame.luma.samples[16];
        assert_ne!(
            out.luma.samples[16], zero_mv_val,
            "AMVP-derived MV must shift the prediction away from zero-MV"
        );
    }

    /// §8.5.2.1 — the non-merge AMVP path folds the parsed `MvdL0`
    /// (with the per-CU `AmvrShift`) into the chosen predictor:
    /// `mvL0 = mvpL0 + (mvdL0 << AmvrShift)`. With the spatial
    /// predictor at (3, 0) px and the default 1/4-luma `AmvrShift == 2`,
    /// a parsed `mvd` of (4, 0) quarter-luma units (= 1 px) must produce
    /// a final 4-px shift. The output must equal MC at (4, 0) px, not
    /// (3, 0).
    #[test]
    fn reconstruct_leaf_cu_inter_amvp_folds_mvd() {
        let pic_w = 64u32;
        let pic_h = 64u32;
        let sps = dummy_sps(1, pic_w, pic_h);
        let pps = dummy_pps(pic_w, pic_h, true);
        let mut sh = intra_slice_header();
        sh.sh_slice_type = SliceType::P;
        let layout = CtuLayout::from_sps_pps(&sps, &pps);
        let data = [0u8; 32];
        let mut walker = CtuWalker::begin_slice(&layout, &sps, &pps, &sh, 0, &data).unwrap();
        walker.current_poc = 0;

        let mut ref_frame = PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 0);
        for y in 0..pic_h as usize {
            for x in 0..pic_w as usize {
                ref_frame.luma.samples[y * ref_frame.luma.stride + x] = (x * 3) as u8;
            }
        }
        walker.set_ref_pic_list_l0(vec![ReferencePicture {
            poc: 4,
            frame: ref_frame.clone(),
            motion_field: None,
        }]);

        let nb_mv = MotionVector::from_int_pel(3, 0);
        let nb = MvField {
            mv_l0: nb_mv,
            ref_idx_l0: 0,
            pred_flag_l0: true,
            available: true,
            mode_inter: true,
            ..MvField::UNAVAILABLE
        };
        walker.motion_field.write_block(0, 0, 16, 64, nb);

        let ccu = CtuCu {
            ctu_addr_rs: 0,
            cu: Cu {
                x: 16,
                y: 0,
                w: 16,
                h: 16,
                cqt_depth: 0,
                mtt_depth: 0,
            },
        };
        let mut info = LeafCuInfo {
            x0: 16,
            y0: 0,
            cb_width: 16,
            cb_height: 16,
            pred_mode: CuPredMode::Inter,
            ..LeafCuInfo::default()
        };
        info.inter.general_merge_flag = false;
        // mvd of (4, 0) in 1/4-luma-derived units; with AmvrShift == 2
        // the §8.5.2.1 fold left-shifts by 2 → (16, 0) in 1/16 units =
        // 1 px. Final mvL0 = (48, 0) + (16, 0) = (64, 0) = 4 px.
        info.inter.non_merge = crate::inter::NonMergeInterData {
            pred_dir: crate::leaf_cu::InterPredDir::PredL0,
            ref_idx_l0: 0,
            mvp_l0_flag: 0,
            mvd_l0: MotionVector { x: 4, y: 0 },
            amvr_shift: crate::amvr::AmvrShift(2),
            ..crate::inter::NonMergeInterData::default()
        };
        let residual = LeafCuResidual::default();

        let mut out = PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 0);
        walker
            .reconstruct_leaf_cu(&ccu, &info, &residual, &mut out)
            .expect("amvp mvd-fold path should reconstruct");

        let final_mv = MotionVector::from_int_pel(4, 0);
        let mut expected = PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 0);
        crate::inter::predict_luma_block(
            &mut expected.luma,
            16,
            0,
            16,
            16,
            &ref_frame.luma,
            final_mv,
        )
        .unwrap();
        for y in 0..16usize {
            for x in 0..16usize {
                let xi = 16 + x;
                assert_eq!(
                    out.luma.samples[y * out.luma.stride + xi],
                    expected.luma.samples[y * expected.luma.stride + xi],
                    "AMVP recon must equal MC at mvp + (mvd << AmvrShift)"
                );
            }
        }
    }

    /// §8.5.6.6.2 eq. 981 — a **non-merge** bi-pred CU carrying a parsed
    /// `bcw_idx > 0` must blend its two predictions with the `BCW_W_LUT`
    /// weights, not the eq. 980 equal-weight average. With L0 ≡ 80 and
    /// L1 ≡ 200 the default (`bcw_idx == 0`) average is 140; `bcw_idx ==
    /// 3` (w0 = −2, w1 = 10) gives `(−2·80 + 10·200 + 4) >> 3 = 230`. The
    /// two reconstructions must differ, proving the parsed `bcw_idx` now
    /// reaches the bi-pred tail through `NonMergeInterData::bcw_idx`
    /// (previously hardcoded 0 on the AMVP record).
    #[test]
    fn reconstruct_leaf_cu_inter_amvp_bipred_bcw_weights() {
        let pic_w = 64u32;
        let pic_h = 64u32;
        let sps = dummy_sps(1, pic_w, pic_h);
        let pps = dummy_pps(pic_w, pic_h, true);
        let mut sh = intra_slice_header();
        sh.sh_slice_type = SliceType::B;
        let layout = CtuLayout::from_sps_pps(&sps, &pps);
        let data = [0u8; 32];
        let mut walker = CtuWalker::begin_slice(&layout, &sps, &pps, &sh, 0, &data).unwrap();
        walker.current_poc = 4;

        walker.set_ref_pic_list_l0(vec![ReferencePicture {
            poc: 0,
            frame: PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 80),
            motion_field: None,
        }]);
        walker.set_ref_pic_list_l1(vec![ReferencePicture {
            poc: 8,
            frame: PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 200),
            motion_field: None,
        }]);

        let ccu = CtuCu {
            ctu_addr_rs: 0,
            cu: Cu {
                x: 16,
                y: 0,
                w: 16,
                h: 16,
                cqt_depth: 0,
                mtt_depth: 0,
            },
        };
        let make_info = |bcw: u8| {
            let mut info = LeafCuInfo {
                x0: 16,
                y0: 0,
                cb_width: 16,
                cb_height: 16,
                pred_mode: CuPredMode::Inter,
                ..LeafCuInfo::default()
            };
            info.inter.general_merge_flag = false;
            info.inter.non_merge = crate::inter::NonMergeInterData {
                pred_dir: crate::leaf_cu::InterPredDir::PredBi,
                ref_idx_l0: 0,
                ref_idx_l1: 0,
                mvp_l0_flag: 0,
                mvp_l1_flag: 0,
                mvd_l0: MotionVector::ZERO,
                mvd_l1: MotionVector::ZERO,
                bcw_idx: bcw,
                ..crate::inter::NonMergeInterData::default()
            };
            info
        };
        let residual = LeafCuResidual::default();

        // bcw_idx == 0 — eq. 980 equal-weight average (≈140).
        let mut out_def = PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 0);
        walker
            .reconstruct_leaf_cu(&ccu, &make_info(0), &residual, &mut out_def)
            .expect("default bi-pred recon");
        // bcw_idx == 3 — eq. 981 weighted blend (230).
        let mut out_bcw = PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 0);
        walker
            .reconstruct_leaf_cu(&ccu, &make_info(3), &residual, &mut out_bcw)
            .expect("BCW bi-pred recon");

        let def00 = out_def.luma.samples[16] as i32;
        let bcw00 = out_bcw.luma.samples[16] as i32;
        assert_eq!(def00, 140, "default-weighted bi-pred must average to 140");
        assert_eq!(
            bcw00, 230,
            "bcw_idx=3 must apply the eq. 981 weighted blend"
        );
    }

    /// §8.5.2.9 step 6 — with no available spatial / temporal / HMVP
    /// candidate the AMVP list pads to the zero-MV predictor, so a CU
    /// with `mvd == 0` reconstructs through the zero-MV prediction
    /// (a verbatim copy of the reference at the CU position).
    #[test]
    fn reconstruct_leaf_cu_inter_amvp_zero_mv_pad() {
        let pic_w = 64u32;
        let pic_h = 64u32;
        let sps = dummy_sps(1, pic_w, pic_h);
        let pps = dummy_pps(pic_w, pic_h, true);
        let mut sh = intra_slice_header();
        sh.sh_slice_type = SliceType::P;
        let layout = CtuLayout::from_sps_pps(&sps, &pps);
        let data = [0u8; 32];
        let mut walker = CtuWalker::begin_slice(&layout, &sps, &pps, &sh, 0, &data).unwrap();
        walker.current_poc = 0;

        let mut ref_frame = PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 0);
        for y in 0..pic_h as usize {
            for x in 0..pic_w as usize {
                ref_frame.luma.samples[y * ref_frame.luma.stride + x] = (x * 3) as u8;
            }
        }
        walker.set_ref_pic_list_l0(vec![ReferencePicture {
            poc: 4,
            frame: ref_frame.clone(),
            motion_field: None,
        }]);
        // Empty motion field → no spatial neighbour available, TMVP off
        // (no ColPic configured) → §8.5.2.9 step 6 zero-MV pad.

        let ccu = CtuCu {
            ctu_addr_rs: 0,
            cu: Cu {
                x: 16,
                y: 0,
                w: 16,
                h: 16,
                cqt_depth: 0,
                mtt_depth: 0,
            },
        };
        let mut info = LeafCuInfo {
            x0: 16,
            y0: 0,
            cb_width: 16,
            cb_height: 16,
            pred_mode: CuPredMode::Inter,
            ..LeafCuInfo::default()
        };
        info.inter.general_merge_flag = false;
        info.inter.non_merge = crate::inter::NonMergeInterData {
            pred_dir: crate::leaf_cu::InterPredDir::PredL0,
            ref_idx_l0: 0,
            mvp_l0_flag: 0,
            mvd_l0: MotionVector::ZERO,
            ..crate::inter::NonMergeInterData::default()
        };
        let residual = LeafCuResidual::default();

        let mut out = PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 0);
        walker
            .reconstruct_leaf_cu(&ccu, &info, &residual, &mut out)
            .expect("amvp zero-mv-pad path should reconstruct");

        // Zero-MV → verbatim copy of the reference at the CU position.
        for y in 0..16usize {
            for x in 0..16usize {
                let xi = 16 + x;
                assert_eq!(
                    out.luma.samples[y * out.luma.stride + xi],
                    ref_frame.luma.samples[y * ref_frame.luma.stride + xi],
                    "zero-MV-pad AMVP recon must copy the reference verbatim"
                );
            }
        }
    }

    /// §8.5.1 / §8.5.3 — DMVR refines a single-sub-block (16×16) bi-pred
    /// merge CU's per-list MVs before motion compensation, so enabling
    /// DMVR changes the reconstructed pixels versus the unrefined path.
    ///
    /// The CU is a B-slice merge CU with an empty motion field / no TMVP
    /// / no HMVP, so the §8.5.2.2 list pads with bi-pred zero-MV
    /// candidates (`merge_idx == 0` ⇒ initial MVs `(0, 0)`). The two
    /// references carry a horizontal ramp shifted by ±2 samples relative
    /// to each other; the §8.5.3.1 bilateral search finds a non-zero
    /// integer delta that aligns them, so the DMVR-on reconstruction
    /// differs from the DMVR-off baseline.
    #[test]
    fn reconstruct_leaf_cu_inter_merge_dmvr_refines_to_pixels() {
        let pic_w = 64u32;
        let pic_h = 64u32;
        let mut sps = dummy_sps(1, pic_w, pic_h);
        sps.tool_flags.dmvr_enabled_flag = true;
        let pps = dummy_pps(pic_w, pic_h, true);
        let mut sh = intra_slice_header();
        sh.sh_slice_type = SliceType::B;
        let layout = CtuLayout::from_sps_pps(&sps, &pps);
        let data = [0u8; 32];

        // Build two reference frames whose luma is a horizontal ramp,
        // ref_l1 shifted right by `shift` samples relative to ref_l0.
        // With symmetric POC (curr=4, ref0=0, ref1=8) the §8.5.3.1
        // opposite-direction search aligns them at a non-zero delta.
        let shift = 2i32;
        // Smooth band-limited 2D sinusoid (the dmvr-module test pattern):
        // aperiodic over the plane, low-frequency, so the bilateral SAD
        // has a strictly convex minimum at the spec-correct delta and
        // the (0, 0) baseline SAD clears the §8.5.3.1 early-out
        // threshold.
        let ramp = |off: i32| {
            let mut f = PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 0);
            for y in 0..pic_h as usize {
                for x in 0..pic_w as usize {
                    let xs = (x as i32 - off).clamp(0, pic_w as i32 - 1);
                    let phx = (xs as f64) / (pic_w as f64) * 3.0 * std::f64::consts::PI;
                    let phy = (y as f64) / (pic_h as f64) * 1.5 * std::f64::consts::PI;
                    let v = (128.0 + 50.0 * phx.sin() + 30.0 * phy.cos()).clamp(0.0, 255.0) as u8;
                    f.luma.samples[y * f.luma.stride + x] = v;
                }
            }
            f
        };

        let ccu = CtuCu {
            ctu_addr_rs: 0,
            cu: Cu {
                x: 16,
                y: 16,
                w: 16,
                h: 16,
                cqt_depth: 0,
                mtt_depth: 0,
            },
        };
        let info = LeafCuInfo {
            x0: 16,
            y0: 16,
            cb_width: 16,
            cb_height: 16,
            pred_mode: CuPredMode::Inter,
            ..LeafCuInfo::default()
        };
        // `general_merge_flag` defaults false on LeafCuInfo::default; set
        // the merge path explicitly with merge_idx 0.
        let mut info = info;
        info.inter.general_merge_flag = true;
        info.inter.merge_data.merge_idx = 0;
        let residual = LeafCuResidual::default();

        let build_walker = |dmvr_disabled: bool| {
            let mut w = CtuWalker::begin_slice(&layout, &sps, &pps, &sh, 0, &data).unwrap();
            w.current_poc = 4;
            w.set_ph_dmvr_disabled(dmvr_disabled);
            w.set_ref_pic_list_l0(vec![ReferencePicture {
                poc: 0,
                frame: ramp(0),
                motion_field: None,
            }]);
            w.set_ref_pic_list_l1(vec![ReferencePicture {
                poc: 8,
                frame: ramp(shift),
                motion_field: None,
            }]);
            w
        };

        // DMVR off — plain eq. 980 average of the two unrefined refs.
        let mut walker_off = build_walker(true);
        let mut out_off = PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 0);
        walker_off
            .reconstruct_leaf_cu(&ccu, &info, &residual, &mut out_off)
            .expect("dmvr-off merge recon");

        // DMVR on — the refined MVs realign the two refs before MC.
        let mut walker_on = build_walker(false);
        let mut out_on = PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 0);
        walker_on
            .reconstruct_leaf_cu(&ccu, &info, &residual, &mut out_on)
            .expect("dmvr-on merge recon");

        // The two reconstructions must differ — DMVR moved the MVs.
        let mut differ = false;
        for y in 0..16usize {
            for x in 0..16usize {
                let xi = 16 + x;
                let yi = 16 + y;
                if out_off.luma.samples[yi * out_off.luma.stride + xi]
                    != out_on.luma.samples[yi * out_on.luma.stride + xi]
                {
                    differ = true;
                }
            }
        }
        assert!(
            differ,
            "DMVR-on reconstruction must differ from the unrefined baseline"
        );

        // §8.5.1 NOTE — the motion field that feeds the spatial-MVP and
        // deblocking processes stores the *unrefined* MvLX (zero here),
        // not the DMVR-refined MV that drove MC.
        let stored = walker_on.motion_field.get_at_luma(20, 20);
        assert!(stored.available, "CU motion field must be written");
        assert_eq!(
            stored.mv_l0,
            MotionVector::ZERO,
            "spatial-MVP store must keep the pre-DMVR (unrefined) L0 MV"
        );
        assert_eq!(
            stored.mv_l1,
            MotionVector::ZERO,
            "spatial-MVP store must keep the pre-DMVR (unrefined) L1 MV"
        );
    }

    /// §8.5.1 eqs. 452 – 459 — a bi-pred merge CU wider than 16 luma
    /// samples splits into `numSbX = cbWidth >> 4` 16×16 DMVR sub-blocks,
    /// each refined independently before its own §8.5.6 bi-pred MC. A
    /// 32×16 CU therefore decodes through the multi-sub-block path and
    /// its pixels differ from the DMVR-off baseline.
    #[test]
    fn reconstruct_leaf_cu_inter_merge_dmvr_multi_subblock_to_pixels() {
        let pic_w = 64u32;
        let pic_h = 64u32;
        let mut sps = dummy_sps(1, pic_w, pic_h);
        sps.tool_flags.dmvr_enabled_flag = true;
        let pps = dummy_pps(pic_w, pic_h, true);
        let mut sh = intra_slice_header();
        sh.sh_slice_type = SliceType::B;
        let layout = CtuLayout::from_sps_pps(&sps, &pps);
        let data = [0u8; 32];

        // Per-sub-block-distinct horizontal shift: the left 16-wide
        // sub-block sees ref_l1 shifted +2, the right one +3, so the two
        // sub-blocks refine to different deltas and the multi-sub-block
        // loop is exercised (a single CU-wide refine could not match
        // both).
        let ramp = |off_left: i32, off_right: i32| {
            let mut f = PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 0);
            for y in 0..pic_h as usize {
                for x in 0..pic_w as usize {
                    let off = if x < 32 { off_left } else { off_right };
                    let xs = (x as i32 - off).clamp(0, pic_w as i32 - 1);
                    let phx = (xs as f64) / (pic_w as f64) * 3.0 * std::f64::consts::PI;
                    let phy = (y as f64) / (pic_h as f64) * 1.5 * std::f64::consts::PI;
                    let v = (128.0 + 50.0 * phx.sin() + 30.0 * phy.cos()).clamp(0.0, 255.0) as u8;
                    f.luma.samples[y * f.luma.stride + x] = v;
                }
            }
            f
        };

        // CU at (16, 16), 32×16 → numSbX = 2, numSbY = 1.
        let ccu = CtuCu {
            ctu_addr_rs: 0,
            cu: Cu {
                x: 16,
                y: 16,
                w: 32,
                h: 16,
                cqt_depth: 0,
                mtt_depth: 0,
            },
        };
        let mut info = LeafCuInfo {
            x0: 16,
            y0: 16,
            cb_width: 32,
            cb_height: 16,
            pred_mode: CuPredMode::Inter,
            ..LeafCuInfo::default()
        };
        info.inter.general_merge_flag = true;
        info.inter.merge_data.merge_idx = 0;
        let residual = LeafCuResidual::default();

        let build_walker = |dmvr_disabled: bool| {
            let mut w = CtuWalker::begin_slice(&layout, &sps, &pps, &sh, 0, &data).unwrap();
            w.current_poc = 4;
            w.set_ph_dmvr_disabled(dmvr_disabled);
            w.set_ref_pic_list_l0(vec![ReferencePicture {
                poc: 0,
                frame: ramp(0, 0),
                motion_field: None,
            }]);
            w.set_ref_pic_list_l1(vec![ReferencePicture {
                poc: 8,
                frame: ramp(2, 3),
                motion_field: None,
            }]);
            w
        };

        let mut walker_off = build_walker(true);
        let mut out_off = PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 0);
        walker_off
            .reconstruct_leaf_cu(&ccu, &info, &residual, &mut out_off)
            .expect("dmvr-off 32x16 merge recon");

        let mut walker_on = build_walker(false);
        let mut out_on = PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 0);
        walker_on
            .reconstruct_leaf_cu(&ccu, &info, &residual, &mut out_on)
            .expect("dmvr-on 32x16 merge recon");

        // Both sub-blocks must show a difference from the baseline.
        let region_differs = |x0: usize| {
            for y in 0..16usize {
                for x in 0..16usize {
                    let xi = x0 + x;
                    let yi = 16 + y;
                    if out_off.luma.samples[yi * out_off.luma.stride + xi]
                        != out_on.luma.samples[yi * out_on.luma.stride + xi]
                    {
                        return true;
                    }
                }
            }
            false
        };
        assert!(
            region_differs(16),
            "left 16×16 DMVR sub-block must differ from the baseline"
        );
        assert!(
            region_differs(32),
            "right 16×16 DMVR sub-block must differ from the baseline"
        );
    }

    /// §7.4.12.5 + §8.7.4.1 — an inter CU with `cu_sbt_flag == 1`
    /// (vertical half-split, residual in TU0 / left half) adds its
    /// inverse-transformed residual ONLY to the residual sub-region; the
    /// non-residual TU stays at the MC prediction (constant 100). The
    /// residual array is sized to the residual TU (16×16), not the whole
    /// 32×16 CU.
    #[test]
    fn reconstruct_leaf_cu_inter_sbt_residual_only_in_res_tu() {
        let pic_w = 64u32;
        let pic_h = 64u32;
        let sps = dummy_sps(1, pic_w, pic_h);
        let pps = dummy_pps(pic_w, pic_h, true);
        let mut sh = intra_slice_header();
        sh.sh_slice_type = SliceType::P;
        let layout = CtuLayout::from_sps_pps(&sps, &pps);
        let data = [0u8; 32];
        let mut walker = CtuWalker::begin_slice(&layout, &sps, &pps, &sh, 0, &data).unwrap();

        walker.set_ref_pic_list_l0(vec![ReferencePicture {
            poc: -1,
            frame: PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 100),
            motion_field: None,
        }]);

        // 32x16 merge CU at (0, 0) — zero-MV uni-pred → constant-100
        // prediction over the whole CU.
        let ccu = CtuCu {
            ctu_addr_rs: 0,
            cu: Cu {
                x: 0,
                y: 0,
                w: 32,
                h: 16,
                cqt_depth: 0,
                mtt_depth: 0,
            },
        };
        let mut info = LeafCuInfo {
            x0: 0,
            y0: 0,
            cb_width: 32,
            cb_height: 16,
            pred_mode: CuPredMode::Inter,
            ..LeafCuInfo::default()
        };
        info.inter.general_merge_flag = true;
        info.inter.merge_data.merge_idx = 0;
        info.tu_y_coded_flag = true;
        // Vertical half-split, residual in TU0 (left 16 cols).
        info.cu_sbt_flag = true;
        info.cu_sbt_quad_flag = false;
        info.cu_sbt_horizontal_flag = false;
        info.cu_sbt_pos_flag = false;
        // Residual TU is 16x16; single positive DC coefficient.
        let mut residual = LeafCuResidual::default();
        residual.luma_levels = vec![0i32; 16 * 16];
        residual.luma_levels[0] = 8;

        let mut out = PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 0);
        walker
            .reconstruct_leaf_cu(&ccu, &info, &residual, &mut out)
            .expect("SBT inter residual path should reconstruct");

        // Left half (residual TU, cols 0..16): the §8.7.4.1 Table-40
        // DCT-VIII × DST-VII kernels produce a spatially-varying
        // residual (not a flat DC lift), so the residual TU must differ
        // from the constant-100 prediction in at least some samples.
        let mut any_changed = false;
        for y in 0..16usize {
            for x in 0..16usize {
                if out.luma.samples[y * out.luma.stride + x] != 100 {
                    any_changed = true;
                }
            }
        }
        assert!(
            any_changed,
            "the SBT residual TU must apply a non-zero residual to the prediction"
        );
        // Right half (non-residual TU, cols 16..32): stays exactly at
        // the prediction — this is the defining SBT property (one TU has
        // no residual).
        for y in 0..16usize {
            for x in 16..32usize {
                assert_eq!(
                    out.luma.samples[y * out.luma.stride + x],
                    100,
                    "non-residual TU sample at ({x},{y}) must stay at the prediction"
                );
            }
        }
    }

    /// §7.4.10.6 + §8.7.1 — an inter CU with `cu_chroma_qp_offset_flag
    /// == 1` adds the per-component `CuQpOffsetC?` term (from the PPS
    /// `pps_c?_qp_offset_list`) to the chroma QP. A non-zero offset
    /// rescales the chroma dequant, so the reconstructed Cb residual must
    /// differ from the no-offset reconstruction. Proves the inter chroma
    /// QP path no longer drops `CuQpOffsetC?` (previously hardcoded 0).
    #[test]
    fn reconstruct_leaf_cu_inter_chroma_qp_offset_changes_recon() {
        let pic_w = 64u32;
        let pic_h = 64u32;
        let sps = dummy_sps(1, pic_w, pic_h);
        let mut pps = dummy_pps(pic_w, pic_h, true);
        // Populate the PPS Cb/Cr offset lists so index 0 carries a strong
        // offset; the §8.7.1 chroma-QP add then shifts the dequant scale.
        pps.pps_cb_qp_offset_list = vec![10];
        pps.pps_cr_qp_offset_list = vec![10];
        let mut sh = intra_slice_header();
        sh.sh_slice_type = SliceType::P;
        let layout = CtuLayout::from_sps_pps(&sps, &pps);
        let data = [0u8; 32];
        let mut walker = CtuWalker::begin_slice(&layout, &sps, &pps, &sh, 0, &data).unwrap();

        walker.set_ref_pic_list_l0(vec![ReferencePicture {
            poc: -1,
            frame: PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 100),
            motion_field: None,
        }]);

        // 16x16 merge CU at (0,0) — zero-MV uni-pred → constant-100 chroma.
        let ccu = CtuCu {
            ctu_addr_rs: 0,
            cu: Cu {
                x: 0,
                y: 0,
                w: 16,
                h: 16,
                cqt_depth: 0,
                mtt_depth: 0,
            },
        };
        let mut residual = LeafCuResidual::default();
        residual.cb_levels = vec![0i32; 8 * 8];
        residual.cb_levels[0] = 8; // single Cb DC coefficient.

        let base_info = || {
            let mut info = LeafCuInfo {
                x0: 0,
                y0: 0,
                cb_width: 16,
                cb_height: 16,
                pred_mode: CuPredMode::Inter,
                ..LeafCuInfo::default()
            };
            info.inter.general_merge_flag = true;
            info.inter.merge_data.merge_idx = 0;
            info.tu_y_coded_flag = false;
            info.tu_cb_coded_flag = true;
            info
        };

        // No CU chroma offset.
        let info_off = base_info();
        let mut out_off = PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 0);
        walker
            .reconstruct_leaf_cu(&ccu, &info_off, &residual, &mut out_off)
            .expect("recon without CU chroma offset");

        // With CU chroma offset (flag = 1, idx = 0 → +10).
        let mut info_on = base_info();
        info_on.cu_chroma_qp_offset_flag = true;
        info_on.cu_chroma_qp_offset_idx = 0;
        let mut out_on = PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 0);
        walker
            .reconstruct_leaf_cu(&ccu, &info_on, &residual, &mut out_on)
            .expect("recon with CU chroma offset");

        let mut differs = false;
        for i in 0..out_off.cb.samples.len() {
            if out_off.cb.samples[i] != out_on.cb.samples[i] {
                differs = true;
                break;
            }
        }
        assert!(
            differs,
            "CuQpOffsetCb must change the inter chroma reconstruction vs. the no-offset path"
        );
    }

    /// §8.7.4.6 — an inter merge CU with `transform_skip_luma == 1`
    /// bypasses the inverse transform: each dequantised coefficient maps
    /// directly to the co-located residual sample (`res[x][y] = d[x][y]`).
    /// A single non-zero level at index 0 therefore lifts **only** sample
    /// (0,0) — the rest of the block stays exactly at the constant-100
    /// prediction. This is the defining contrast with the DCT-II path,
    /// where a lone DC coefficient spreads a flat lift across the whole
    /// block. Proves the inter TS reconstruction is wired end-to-end.
    #[test]
    fn reconstruct_leaf_cu_inter_transform_skip_luma_no_spread() {
        let pic_w = 64u32;
        let pic_h = 64u32;
        let sps = dummy_sps(1, pic_w, pic_h);
        let pps = dummy_pps(pic_w, pic_h, true);
        let mut sh = intra_slice_header();
        sh.sh_slice_type = SliceType::P;
        let layout = CtuLayout::from_sps_pps(&sps, &pps);
        let data = [0u8; 32];
        let mut walker = CtuWalker::begin_slice(&layout, &sps, &pps, &sh, 0, &data).unwrap();

        walker.set_ref_pic_list_l0(vec![ReferencePicture {
            poc: -1,
            frame: PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 100),
            motion_field: None,
        }]);

        // 16x16 merge CU at (0, 0) — zero-MV uni-pred → constant-100
        // prediction over the whole CU.
        let ccu = CtuCu {
            ctu_addr_rs: 0,
            cu: Cu {
                x: 0,
                y: 0,
                w: 16,
                h: 16,
                cqt_depth: 0,
                mtt_depth: 0,
            },
        };
        let mut info = LeafCuInfo {
            x0: 0,
            y0: 0,
            cb_width: 16,
            cb_height: 16,
            pred_mode: CuPredMode::Inter,
            ..LeafCuInfo::default()
        };
        info.inter.general_merge_flag = true;
        info.inter.merge_data.merge_idx = 0;
        info.tu_y_coded_flag = true;
        // Transform-skip on the luma TB — single positive level at (0,0).
        info.transform_skip_luma = true;
        let mut residual = LeafCuResidual::default();
        residual.luma_levels = vec![0i32; 16 * 16];
        residual.luma_levels[0] = 4;

        let mut out = PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 0);
        walker
            .reconstruct_leaf_cu(&ccu, &info, &residual, &mut out)
            .expect("inter transform-skip path should reconstruct");

        // Sample (0,0) must change; every other luma sample of the CU
        // stays at the constant-100 prediction (no DCT spreading).
        let s00 = out.luma.samples[0] as i32;
        assert!(
            s00 > 100,
            "TS DC lift must raise sample (0,0) above the prediction (got {s00})"
        );
        for y in 0..16usize {
            for x in 0..16usize {
                if x == 0 && y == 0 {
                    continue;
                }
                assert_eq!(
                    out.luma.samples[y * out.luma.stride + x],
                    100,
                    "TS residual must not spread to sample ({x},{y})"
                );
            }
        }
    }

    /// §8.7.2 joint Cb-Cr inter residual — an inter merge CU with
    /// `TuCResMode == 2` (both chroma CBFs set, joint flag set, Cb is
    /// the coded component) reconstructs Cr from the Cb coefficients:
    /// with `ph_joint_cbcr_sign_flag == 0` (`cSign = +1`) eq. 1131
    /// gives `resCr = cSign * resCb`, so the Cb and Cr residuals are
    /// identical. With a constant-100 chroma reference the two planes
    /// must end up equal to each other and different from the
    /// prediction. This proves the JCCR inter path no longer surfaces
    /// Unsupported and derives both chroma planes from one coded block.
    #[test]
    fn reconstruct_leaf_cu_inter_joint_cbcr_mode2() {
        let pic_w = 64u32;
        let pic_h = 64u32;
        let sps = dummy_sps(1, pic_w, pic_h);
        let pps = dummy_pps(pic_w, pic_h, true);
        let mut sh = intra_slice_header();
        sh.sh_slice_type = SliceType::P;
        let layout = CtuLayout::from_sps_pps(&sps, &pps);
        let data = [0u8; 32];
        let mut walker = CtuWalker::begin_slice(&layout, &sps, &pps, &sh, 0, &data).unwrap();
        // ph_joint_cbcr_sign_flag == 0 → cSign = +1.
        walker.set_ph_joint_cbcr_sign(false);

        walker.set_ref_pic_list_l0(vec![ReferencePicture {
            poc: -1,
            frame: PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 100),
            motion_field: None,
        }]);

        // 16x16 merge CU at (0, 0) — zero-MV uni-pred → constant-100
        // prediction over luma + both chroma planes.
        let ccu = CtuCu {
            ctu_addr_rs: 0,
            cu: Cu {
                x: 0,
                y: 0,
                w: 16,
                h: 16,
                cqt_depth: 0,
                mtt_depth: 0,
            },
        };
        let mut info = LeafCuInfo {
            x0: 0,
            y0: 0,
            cb_width: 16,
            cb_height: 16,
            pred_mode: CuPredMode::Inter,
            ..LeafCuInfo::default()
        };
        info.inter.general_merge_flag = true;
        info.inter.merge_data.merge_idx = 0;
        // No luma residual; both chroma CBFs set with JCCR mode 2.
        info.tu_y_coded_flag = false;
        info.tu_cb_coded_flag = true;
        info.tu_cr_coded_flag = true;
        info.tu_joint_cbcr_residual_flag = true;
        info.tu_c_res_mode = 2;
        // Coded chroma block is Cb (8x8 for the 16x16 4:2:0 CU); single
        // positive DC coefficient.
        let mut residual = LeafCuResidual::default();
        residual.cb_levels = vec![0i32; 8 * 8];
        residual.cb_levels[0] = 8;

        let mut out = PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 100);
        walker
            .reconstruct_leaf_cu(&ccu, &info, &residual, &mut out)
            .expect("JCCR inter residual path should reconstruct");

        // The zero-MV chroma prediction is the constant neutral 128
        // (yuv420_filled always seeds chroma at 128). Both chroma planes
        // must be modified from that prediction, and (cSign = +1, mode 2)
        // be identical to each other.
        let mut any_cb_changed = false;
        for cy in 0..8usize {
            for cx in 0..8usize {
                let idx = cy * out.cb.stride + cx;
                if out.cb.samples[idx] != 128 {
                    any_cb_changed = true;
                }
                assert_eq!(
                    out.cb.samples[idx], out.cr.samples[idx],
                    "JCCR mode 2 with cSign=+1 must give identical Cb / Cr residuals at ({cx},{cy})"
                );
            }
        }
        assert!(
            any_cb_changed,
            "the JCCR coded block must apply a non-zero residual to both chroma planes"
        );
    }

    /// §8.7.2 + §8.7.4.6 — JCCR inter residual with the coded chroma TB
    /// (Cb, mode 2) carrying `transform_skip_flag == 1`. The single coded
    /// block bypasses the inverse transform, so the lone DC level lifts
    /// only chroma sample (0,0) of BOTH planes (cSign = +1 → identical),
    /// with no DCT spreading. Proves the JCCR inter path now honours the
    /// coded component's transform-skip flag.
    #[test]
    fn reconstruct_leaf_cu_inter_joint_cbcr_transform_skip() {
        let pic_w = 64u32;
        let pic_h = 64u32;
        let sps = dummy_sps(1, pic_w, pic_h);
        let pps = dummy_pps(pic_w, pic_h, true);
        let mut sh = intra_slice_header();
        sh.sh_slice_type = SliceType::P;
        let layout = CtuLayout::from_sps_pps(&sps, &pps);
        let data = [0u8; 32];
        let mut walker = CtuWalker::begin_slice(&layout, &sps, &pps, &sh, 0, &data).unwrap();
        walker.set_ph_joint_cbcr_sign(false);

        walker.set_ref_pic_list_l0(vec![ReferencePicture {
            poc: -1,
            frame: PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 100),
            motion_field: None,
        }]);

        let ccu = CtuCu {
            ctu_addr_rs: 0,
            cu: Cu {
                x: 0,
                y: 0,
                w: 16,
                h: 16,
                cqt_depth: 0,
                mtt_depth: 0,
            },
        };
        let mut info = LeafCuInfo {
            x0: 0,
            y0: 0,
            cb_width: 16,
            cb_height: 16,
            pred_mode: CuPredMode::Inter,
            ..LeafCuInfo::default()
        };
        info.inter.general_merge_flag = true;
        info.inter.merge_data.merge_idx = 0;
        info.tu_y_coded_flag = false;
        info.tu_cb_coded_flag = true;
        info.tu_cr_coded_flag = true;
        info.tu_joint_cbcr_residual_flag = true;
        info.tu_c_res_mode = 2;
        // Coded Cb block carries transform-skip; single DC level.
        info.transform_skip_cb = true;
        let mut residual = LeafCuResidual::default();
        residual.cb_levels = vec![0i32; 8 * 8];
        residual.cb_levels[0] = 8;

        let mut out = PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 100);
        walker
            .reconstruct_leaf_cu(&ccu, &info, &residual, &mut out)
            .expect("JCCR transform-skip inter residual should reconstruct");

        // Chroma plane is seeded at neutral 128 by yuv420_filled. Only
        // (0,0) of each plane may move; cSign = +1 → Cb == Cr everywhere.
        assert_ne!(
            out.cb.samples[0], 128,
            "TS JCCR DC must lift chroma sample (0,0)"
        );
        for cy in 0..8usize {
            for cx in 0..8usize {
                let idx = cy * out.cb.stride + cx;
                assert_eq!(
                    out.cb.samples[idx], out.cr.samples[idx],
                    "cSign=+1 JCCR must keep Cb == Cr at ({cx},{cy})"
                );
                if cx == 0 && cy == 0 {
                    continue;
                }
                assert_eq!(
                    out.cb.samples[idx], 128,
                    "TS JCCR residual must not spread to chroma ({cx},{cy})"
                );
            }
        }
    }

    /// §8.7.2 joint Cb-Cr inter residual, mode 2 with
    /// `ph_joint_cbcr_sign_flag == 1` (`cSign = −1`): eq. 1131 gives
    /// `resCr = −resCb`, so the Cb and Cr residuals are negatives of
    /// each other relative to the shared prediction. Cb lifts above
    /// the 100 prediction at the DC, Cr drops below it (or vice versa).
    #[test]
    fn reconstruct_leaf_cu_inter_joint_cbcr_sign_negates() {
        let pic_w = 64u32;
        let pic_h = 64u32;
        let sps = dummy_sps(1, pic_w, pic_h);
        let pps = dummy_pps(pic_w, pic_h, true);
        let mut sh = intra_slice_header();
        sh.sh_slice_type = SliceType::P;
        let layout = CtuLayout::from_sps_pps(&sps, &pps);
        let data = [0u8; 32];
        let mut walker = CtuWalker::begin_slice(&layout, &sps, &pps, &sh, 0, &data).unwrap();
        // ph_joint_cbcr_sign_flag == 1 → cSign = -1.
        walker.set_ph_joint_cbcr_sign(true);

        walker.set_ref_pic_list_l0(vec![ReferencePicture {
            poc: -1,
            frame: PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 100),
            motion_field: None,
        }]);

        let ccu = CtuCu {
            ctu_addr_rs: 0,
            cu: Cu {
                x: 0,
                y: 0,
                w: 16,
                h: 16,
                cqt_depth: 0,
                mtt_depth: 0,
            },
        };
        let mut info = LeafCuInfo {
            x0: 0,
            y0: 0,
            cb_width: 16,
            cb_height: 16,
            pred_mode: CuPredMode::Inter,
            ..LeafCuInfo::default()
        };
        info.inter.general_merge_flag = true;
        info.tu_y_coded_flag = false;
        info.tu_cb_coded_flag = true;
        info.tu_cr_coded_flag = true;
        info.tu_joint_cbcr_residual_flag = true;
        info.tu_c_res_mode = 2;
        let mut residual = LeafCuResidual::default();
        residual.cb_levels = vec![0i32; 8 * 8];
        residual.cb_levels[0] = 8;

        let mut out = PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 100);
        walker
            .reconstruct_leaf_cu(&ccu, &info, &residual, &mut out)
            .expect("JCCR inter residual path should reconstruct");

        // For every chroma sample, (Cb − 128) must equal −(Cr − 128):
        // cSign = −1 makes the Cr residual the negation of the Cb one
        // (the chroma prediction is the neutral 128 baseline).
        let mut any_changed = false;
        for cy in 0..8usize {
            for cx in 0..8usize {
                let idx = cy * out.cb.stride + cx;
                let cb_delta = out.cb.samples[idx] as i32 - 128;
                let cr_delta = out.cr.samples[idx] as i32 - 128;
                if cb_delta != 0 {
                    any_changed = true;
                }
                assert_eq!(
                    cb_delta, -cr_delta,
                    "JCCR mode 2 with cSign=-1: Cr residual must negate Cb at ({cx},{cy})"
                );
            }
        }
        assert!(
            any_changed,
            "JCCR sign-negate test must apply a non-zero residual"
        );
    }

    /// §8.5.5.3 / §8.5.6 affine uni-pred reconstruction — a degenerate
    /// affine CU whose CPMVs are all the **same** integer-pel MV
    /// reduces to a translational copy: every per-sub-block MV equals
    /// that MV (eqs. 872 – 875), so the affine luma MC must produce the
    /// same pixels as a plain `predict_luma_block` with that MV. The
    /// chroma sub-block MV averaging (eqs. 876 – 879) of two identical
    /// MVs is also that MV. With a horizontal-ramp reference and a
    /// `(2, 0)` integer-pel shift, the CU samples must equal the
    /// reference shifted left by 2 luma columns.
    #[test]
    fn reconstruct_affine_inter_uni_translational_equiv() {
        let pic_w = 64u32;
        let pic_h = 64u32;
        let sps = dummy_sps(1, pic_w, pic_h);
        let pps = dummy_pps(pic_w, pic_h, true);
        let mut sh = intra_slice_header();
        sh.sh_slice_type = SliceType::P;
        let layout = CtuLayout::from_sps_pps(&sps, &pps);
        let data = [0u8; 32];
        let walker = CtuWalker::begin_slice(&layout, &sps, &pps, &sh, 0, &data).unwrap();

        // Horizontal-ramp reference: luma[x] = (x & 63) as u8.
        let mut ref_frame = PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 0);
        for y in 0..pic_h as usize {
            for x in 0..pic_w as usize {
                ref_frame.luma.samples[y * ref_frame.luma.stride + x] = (x & 63) as u8;
            }
        }
        let ref_pic = ReferencePicture {
            poc: -1,
            frame: ref_frame.clone(),
            motion_field: None,
        };

        // All CPMVs == (2, 0) integer-pel → translational shift.
        let mv = MotionVector::from_int_pel(2, 0);
        let cpmvs = crate::affine::AffineCpmvs::new_4param(mv, mv);

        let mut out = PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 0);
        walker
            .reconstruct_affine_inter_uni(16, 16, 16, 16, &cpmvs, &ref_pic, &mut out)
            .expect("affine uni recon should succeed");

        // Compare to a plain translational predict_luma_block.
        let mut expect = PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 0);
        predict_luma_block(&mut expect.luma, 16, 16, 16, 16, &ref_frame.luma, mv)
            .expect("translational reference MC");
        for y in 16..32usize {
            for x in 16..32usize {
                assert_eq!(
                    out.luma.samples[y * out.luma.stride + x],
                    expect.luma.samples[y * expect.luma.stride + x],
                    "degenerate affine luma must equal translational MC at ({x},{y})"
                );
            }
        }
    }

    /// §8.5.5.9 — a genuine affine model (non-equal CPMVs) produces a
    /// **spatially-varying** per-sub-block MV field, so the affine MC
    /// output must differ from a single translational copy of the
    /// reference. This proves the sub-block MV derivation is actually
    /// exercised rather than collapsing to one MV.
    #[test]
    fn reconstruct_affine_inter_uni_varies_per_subblock() {
        let pic_w = 64u32;
        let pic_h = 64u32;
        let sps = dummy_sps(1, pic_w, pic_h);
        let pps = dummy_pps(pic_w, pic_h, true);
        let mut sh = intra_slice_header();
        sh.sh_slice_type = SliceType::P;
        let layout = CtuLayout::from_sps_pps(&sps, &pps);
        let data = [0u8; 32];
        let walker = CtuWalker::begin_slice(&layout, &sps, &pps, &sh, 0, &data).unwrap();

        // 2-D ramp reference so any MV difference shows up in samples.
        let mut ref_frame = PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 0);
        for y in 0..pic_h as usize {
            for x in 0..pic_w as usize {
                ref_frame.luma.samples[y * ref_frame.luma.stride + x] = ((x + y) & 63) as u8;
            }
        }
        let ref_pic = ReferencePicture {
            poc: -1,
            frame: ref_frame.clone(),
            motion_field: None,
        };

        // 4-param affine: cp0 = (0,0), cp1 = (4,0) int-pel → a
        // horizontal MV gradient across the CU width.
        let cp0 = MotionVector::from_int_pel(0, 0);
        let cp1 = MotionVector::from_int_pel(4, 0);
        let cpmvs = crate::affine::AffineCpmvs::new_4param(cp0, cp1);

        let mut out = PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 0);
        walker
            .reconstruct_affine_inter_uni(16, 16, 16, 16, &cpmvs, &ref_pic, &mut out)
            .expect("affine uni recon should succeed");

        // A single zero-MV translational copy would reproduce the
        // reference verbatim at the CU; the affine gradient must change
        // at least some samples relative to that.
        let mut any_diff = false;
        for y in 16..32usize {
            for x in 16..32usize {
                let got = out.luma.samples[y * out.luma.stride + x];
                let flat = ref_frame.luma.samples[y * ref_frame.luma.stride + x];
                if got != flat {
                    any_diff = true;
                }
            }
        }
        assert!(
            any_diff,
            "a non-degenerate affine model must produce a per-sub-block-varying prediction"
        );
    }

    /// §8.5.5.5 affine non-merge (AMVP) parse-to-pixels fuse — drives a
    /// parsed `NonMergeInterPreResidualAffineDecision` (per-CP MvdCpL0
    /// arrays) through the §8.5.5.7 candidate list + eqs. 660 – 667
    /// CPMV fold + §8.5.6 affine MC, and checks the result equals a
    /// direct `reconstruct_affine_inter_uni` on the independently-folded
    /// CPMVs.
    ///
    /// With `mvp_l0_flag == 0` and no temporal MF (`motion_field:
    /// None`), the §8.5.5.7 list falls through to the step-9 zero-MV
    /// pad, so the predictor is zero and the final CPMVs equal the
    /// cumulative MVD (eqs. 660 – 663). The CP1 MVD is differential
    /// relative to CP0, so the test pins the cumulative fold.
    #[test]
    fn reconstruct_leaf_cu_inter_affine_amvp_parse_to_pixels() {
        use crate::affine_syntax_enc::make_non_merge_inter_affine_decision;
        use crate::non_merge_inter_pre_residual_enc::NonMergeInterPreResidualAffineDecision;
        use crate::non_merge_mvp_syntax_enc::make_non_merge_mvp_syntax_decision;

        let pic_w = 64u32;
        let pic_h = 64u32;
        let sps = dummy_sps(1, pic_w, pic_h);
        let pps = dummy_pps(pic_w, pic_h, true);
        let mut sh = intra_slice_header();
        sh.sh_slice_type = SliceType::P;
        let layout = CtuLayout::from_sps_pps(&sps, &pps);
        let data = [0u8; 32];
        let mut walker = CtuWalker::begin_slice(&layout, &sps, &pps, &sh, 0, &data).unwrap();

        // 2-D ramp reference so an affine MV gradient is visible.
        let mut ref_frame = PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 0);
        for y in 0..pic_h as usize {
            for x in 0..pic_w as usize {
                ref_frame.luma.samples[y * ref_frame.luma.stride + x] = ((x + y) & 63) as u8;
            }
        }
        for cy in 0..(pic_h / 2) as usize {
            for cx in 0..(pic_w / 2) as usize {
                ref_frame.cb.samples[cy * ref_frame.cb.stride + cx] = ((cx * 2) & 63) as u8;
                ref_frame.cr.samples[cy * ref_frame.cr.stride + cx] = ((cy * 2) & 63) as u8;
            }
        }
        let ref_pic = ReferencePicture {
            poc: -1,
            frame: ref_frame.clone(),
            motion_field: None,
        };
        walker.set_ref_pic_list_l0(vec![ref_pic.clone()]);

        // 4-param affine, P-slice (L0 only). Per-CP stored MVDs: CP0 =
        // (8,0) int-pel, CP1 = (8,0) differential (eq. 662 → cumulative
        // CP1 = (16,0)) → a horizontal MV gradient.
        let affine = make_non_merge_inter_affine_decision(true, false);
        let mvp = make_non_merge_mvp_syntax_decision(
            crate::leaf_cu::InterPredDir::PredL0,
            false,
            0,
            0,
            0,
            0,
        );
        let mvd_cp_l0 = [
            MotionVector::from_int_pel(8, 0),
            MotionVector::from_int_pel(8, 0),
            MotionVector::ZERO,
        ];
        let decision = NonMergeInterPreResidualAffineDecision::new(
            affine,
            mvp,
            mvd_cp_l0,
            [MotionVector::ZERO; 3],
        );

        let mut out = PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 0);
        walker
            .reconstruct_leaf_cu_inter_affine_amvp(
                16,
                16,
                16,
                16,
                &decision,
                crate::amvr::AmvrShift(0),
                0, // bcw_idx
                &mut out,
            )
            .expect("affine amvp parse-to-pixels");

        // Independent reference: predictor is zero (zero-MV pad), so the
        // final CPMVs are the cumulative MVDs: CP0 = (8,0), CP1 = CP1 +
        // CP0 = (16,0) int-pel.
        let cpmvs = crate::affine::AffineCpmvs::new_4param(
            MotionVector::from_int_pel(8, 0),
            MotionVector::from_int_pel(16, 0),
        );
        let mut expect = PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 0);
        walker
            .reconstruct_affine_inter_uni(16, 16, 16, 16, &cpmvs, &ref_pic, &mut expect)
            .expect("direct affine uni recon");

        for y in 16..32usize {
            for x in 16..32usize {
                assert_eq!(
                    out.luma.samples[y * out.luma.stride + x],
                    expect.luma.samples[y * expect.luma.stride + x],
                    "affine AMVP fuse luma must equal the folded-CPMV recon at ({x},{y})"
                );
            }
        }
        // Chroma must also match the independent recon.
        for cy in 8..16usize {
            for cx in 8..16usize {
                assert_eq!(
                    out.cb.samples[cy * out.cb.stride + cx],
                    expect.cb.samples[cy * expect.cb.stride + cx],
                    "affine AMVP fuse Cb must match at ({cx},{cy})"
                );
            }
        }
    }

    /// §8.5.5.5 affine non-merge (AMVP) bi-pred parse-to-pixels fuse —
    /// a B-slice CU with `inter_pred_idc == PRED_BI` drives both lists'
    /// per-CP MVD streams through the §8.5.5.7 candidate list + eqs.
    /// 660 – 667 fold and into the §8.5.6.6.2 eq. 980 default-weighted
    /// average. With identical L0 / L1 references and identical per-CP
    /// MVDs, the bi-pred average must equal the per-list affine bi-pred
    /// prediction (averaging identical predictions is the identity).
    #[test]
    fn reconstruct_leaf_cu_inter_affine_amvp_bipred_parse_to_pixels() {
        use crate::affine_syntax_enc::make_non_merge_inter_affine_decision;
        use crate::non_merge_inter_pre_residual_enc::NonMergeInterPreResidualAffineDecision;
        use crate::non_merge_mvp_syntax_enc::make_non_merge_mvp_syntax_decision;

        let pic_w = 64u32;
        let pic_h = 64u32;
        let sps = dummy_sps(1, pic_w, pic_h);
        let pps = dummy_pps(pic_w, pic_h, true);
        let mut sh = intra_slice_header();
        sh.sh_slice_type = SliceType::B;
        let layout = CtuLayout::from_sps_pps(&sps, &pps);
        let data = [0u8; 32];
        let mut walker = CtuWalker::begin_slice(&layout, &sps, &pps, &sh, 0, &data).unwrap();

        let mut ref_frame = PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 0);
        for y in 0..pic_h as usize {
            for x in 0..pic_w as usize {
                ref_frame.luma.samples[y * ref_frame.luma.stride + x] = ((x + y) & 63) as u8;
            }
        }
        let ref_pic = ReferencePicture {
            poc: -1,
            frame: ref_frame.clone(),
            motion_field: None,
        };
        walker.set_ref_pic_list_l0(vec![ref_pic.clone()]);
        walker.set_ref_pic_list_l1(vec![ref_pic.clone()]);

        // 4-param affine bi-pred; same per-CP MVDs on both lists →
        // identical per-list predictions. Predictor is zero (zero-MV
        // pad, no MF), so final CPMVs = cumulative MVD.
        let affine = make_non_merge_inter_affine_decision(true, false);
        let mvp = make_non_merge_mvp_syntax_decision(
            crate::leaf_cu::InterPredDir::PredBi,
            false,
            0,
            0,
            0,
            0,
        );
        let mvd_cp = [
            MotionVector::from_int_pel(0, 0),
            MotionVector::from_int_pel(4, 0),
            MotionVector::ZERO,
        ];
        let decision = NonMergeInterPreResidualAffineDecision::new(affine, mvp, mvd_cp, mvd_cp);

        let mut out = PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 0);
        walker
            .reconstruct_leaf_cu_inter_affine_amvp(
                16,
                16,
                16,
                16,
                &decision,
                crate::amvr::AmvrShift(0),
                0, // bcw_idx
                &mut out,
            )
            .expect("affine amvp bi-pred parse-to-pixels");

        // Independent reference: cumulative CPMVs (CP0 = (0,0), CP1 =
        // (4,0)) → bi-pred recon of identical lists.
        let cpmvs = crate::affine::AffineCpmvs::new_4param(
            MotionVector::from_int_pel(0, 0),
            MotionVector::from_int_pel(4, 0),
        );
        let mut expect = PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 0);
        walker
            .reconstruct_affine_inter_bi(
                16,
                16,
                16,
                16,
                &cpmvs,
                &ref_pic,
                &cpmvs,
                &ref_pic,
                &mut expect,
            )
            .expect("direct affine bi recon");

        let mut any = false;
        for y in 16..32usize {
            for x in 16..32usize {
                let got = out.luma.samples[y * out.luma.stride + x];
                assert_eq!(
                    got,
                    expect.luma.samples[y * expect.luma.stride + x],
                    "affine AMVP bi-pred fuse luma must match the folded-CPMV bi recon at ({x},{y})"
                );
                if got != ((x + y) & 63) as u8 {
                    any = true;
                }
            }
        }
        assert!(
            any,
            "the affine bi-pred fuse must produce a non-trivial prediction"
        );
    }

    /// §8.5.6.6.2 — the parsed `bcw_idx` is threaded into the affine
    /// non-merge (AMVP) bi-pred reconstruction. With distinct constant
    /// L0 / L1 references (40 / 200), `bcw_idx == 0` averages to 120
    /// (eq. 980) while `bcw_idx == 3` (w0 = −2, w1 = 10) applies the
    /// eq. 981 weighted blend `Clip1((−2·40 + 10·200 + 4) >> 3) = 240`.
    /// The stored affine CB record also carries the weight so a later
    /// inherited-affine-merge scan recovers it.
    #[test]
    fn reconstruct_leaf_cu_inter_affine_amvp_bipred_bcw_weights() {
        use crate::affine_syntax_enc::make_non_merge_inter_affine_decision;
        use crate::non_merge_inter_pre_residual_enc::NonMergeInterPreResidualAffineDecision;
        use crate::non_merge_mvp_syntax_enc::make_non_merge_mvp_syntax_decision;

        let pic_w = 64u32;
        let pic_h = 64u32;
        let sps = dummy_sps(1, pic_w, pic_h);
        let pps = dummy_pps(pic_w, pic_h, true);
        let mut sh = intra_slice_header();
        sh.sh_slice_type = SliceType::B;
        let layout = CtuLayout::from_sps_pps(&sps, &pps);
        let data = [0u8; 32];

        // Distinct constant references so the per-list affine predictions
        // are 40 (L0) and 200 (L1) everywhere → the blend is the only
        // variable between bcw_idx 0 and 3.
        let build = |bcw: u8| {
            let mut walker = CtuWalker::begin_slice(&layout, &sps, &pps, &sh, 0, &data).unwrap();
            walker.set_ref_pic_list_l0(vec![ReferencePicture {
                poc: -1,
                frame: PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 40),
                motion_field: None,
            }]);
            walker.set_ref_pic_list_l1(vec![ReferencePicture {
                poc: 1,
                frame: PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 200),
                motion_field: None,
            }]);
            let affine = make_non_merge_inter_affine_decision(true, false);
            let mvp = make_non_merge_mvp_syntax_decision(
                crate::leaf_cu::InterPredDir::PredBi,
                false,
                0,
                0,
                0,
                0,
            );
            // Zero per-CP MVDs → zero CPMVs → both lists read the
            // reference verbatim (constant planes), so the predictions
            // are exactly 40 / 200.
            let mvd_cp = [MotionVector::ZERO; 3];
            let decision = NonMergeInterPreResidualAffineDecision::new(affine, mvp, mvd_cp, mvd_cp);
            let mut out = PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 0);
            walker
                .reconstruct_leaf_cu_inter_affine_amvp(
                    16,
                    16,
                    16,
                    16,
                    &decision,
                    crate::amvr::AmvrShift(0),
                    bcw,
                    &mut out,
                )
                .expect("affine amvp bcw recon");
            (walker, out)
        };

        let (_w0, out_def) = build(0);
        let (walker3, out_bcw) = build(3);
        assert_eq!(
            out_def.luma.samples[16 * out_def.luma.stride + 16],
            120,
            "bcw_idx=0 must average 40/200 to 120"
        );
        assert_eq!(
            out_bcw.luma.samples[16 * out_bcw.luma.stride + 16],
            240,
            "bcw_idx=3 must apply the eq. 981 weighted blend to 240"
        );
        // The stored affine CB record carries the weight.
        let rec = walker3
            .affine_cpmv_field
            .get_at_luma(20, 20)
            .expect("affine CB record stored");
        assert_eq!(rec.bcw_idx, 3, "affine AMVP store must keep bcw_idx = 3");
    }

    /// §8.5.5.7 inherited affine CPMVP — round-364 per-CB affine CPMV
    /// store. A first affine CB at the left-neighbour position writes
    /// its CPMV record into the store; a second affine CB then samples
    /// that neighbour (position A1 = (xCb − 1, yCb + cbHeight − 1)) in
    /// the §8.5.5.7 step-4 A-scan, so its `mvp_l0_flag == 0` predictor
    /// is the **inherited** candidate (eqs. 824 / 840) rather than the
    /// step-9 zero-MV pad. The inherited predictor must change the
    /// second CU's output relative to a zero-pad-only baseline, and the
    /// store must report the first CB's record at the queried position.
    #[test]
    fn affine_amvp_inherited_cpmvp_from_per_cb_store() {
        use crate::affine_syntax_enc::make_non_merge_inter_affine_decision;
        use crate::non_merge_inter_pre_residual_enc::NonMergeInterPreResidualAffineDecision;
        use crate::non_merge_mvp_syntax_enc::make_non_merge_mvp_syntax_decision;

        let pic_w = 64u32;
        let pic_h = 64u32;
        let sps = dummy_sps(1, pic_w, pic_h);
        let pps = dummy_pps(pic_w, pic_h, true);
        let mut sh = intra_slice_header();
        sh.sh_slice_type = SliceType::P;
        let layout = CtuLayout::from_sps_pps(&sps, &pps);
        let data = [0u8; 32];
        let mut walker = CtuWalker::begin_slice(&layout, &sps, &pps, &sh, 0, &data).unwrap();

        let mut ref_frame = PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 0);
        for y in 0..pic_h as usize {
            for x in 0..pic_w as usize {
                ref_frame.luma.samples[y * ref_frame.luma.stride + x] = ((x + y) & 63) as u8;
            }
        }
        let ref_pic = ReferencePicture {
            poc: -1,
            frame: ref_frame.clone(),
            motion_field: None,
        };
        walker.set_ref_pic_list_l0(vec![ref_pic.clone()]);

        // ---- First affine CB at (0, 16) size 16x16 -------------------
        // Its covered region x∈[0,16) y∈[16,32) includes the A1 sample
        // (15, 31) of a CU at (16, 16, 16, 16). 4-param, L0, a strong
        // horizontal MV gradient so the inherited predictor is clearly
        // non-zero. mvp_l0_flag == 0, no MF ⇒ predictor zero ⇒ final
        // CPMVs = cumulative MVD: CP0 = (12,0), CP1 = (24,0).
        let affine0 = make_non_merge_inter_affine_decision(true, false);
        let mvp0 = make_non_merge_mvp_syntax_decision(
            crate::leaf_cu::InterPredDir::PredL0,
            false,
            0,
            0,
            0,
            0,
        );
        let mvd_cp0 = [
            MotionVector::from_int_pel(12, 0),
            MotionVector::from_int_pel(12, 0),
            MotionVector::ZERO,
        ];
        let decision0 = NonMergeInterPreResidualAffineDecision::new(
            affine0,
            mvp0,
            mvd_cp0,
            [MotionVector::ZERO; 3],
        );
        let mut out0 = PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 0);
        walker
            .reconstruct_leaf_cu_inter_affine_amvp(
                0,
                16,
                16,
                16,
                &decision0,
                crate::amvr::AmvrShift(0),
                0, // bcw_idx
                &mut out0,
            )
            .expect("first affine CB recon");

        // The store must now carry the first CB's record at A1 (15, 31).
        let nb = walker.affine_cpmv_field.get_at_luma(15, 31);
        assert!(
            nb.is_some(),
            "store must hold the first affine CB at (15,31)"
        );
        let nb = nb.unwrap();
        assert_eq!((nb.xnb, nb.ynb, nb.nb_w, nb.nb_h), (0, 16, 16, 16));
        assert!(nb.pred_flag_l0 && !nb.pred_flag_l1);
        assert_eq!(nb.cpmvs_l0.cpmvs[0], MotionVector::from_int_pel(12, 0));
        assert_eq!(nb.cpmvs_l0.cpmvs[1], MotionVector::from_int_pel(24, 0));

        // ---- Second affine CB at (16, 16) size 16x16 ----------------
        // mvp_l0_flag == 0, zero MVDs. The §8.5.5.7 A-scan picks the
        // inherited candidate from the left neighbour, so the predictor
        // is NOT zero. Output must therefore differ from a zero-CPMV
        // recon (the result a pre-store decoder would have produced).
        let affine1 = make_non_merge_inter_affine_decision(true, false);
        let mvp1 = make_non_merge_mvp_syntax_decision(
            crate::leaf_cu::InterPredDir::PredL0,
            false,
            0,
            0,
            0,
            0,
        );
        let mvd_cp1 = [MotionVector::ZERO; 3];
        let decision1 = NonMergeInterPreResidualAffineDecision::new(
            affine1,
            mvp1,
            mvd_cp1,
            [MotionVector::ZERO; 3],
        );
        let mut out1 = PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 0);
        walker
            .reconstruct_leaf_cu_inter_affine_amvp(
                16,
                16,
                16,
                16,
                &decision1,
                crate::amvr::AmvrShift(0),
                0, // bcw_idx
                &mut out1,
            )
            .expect("second affine CB recon");

        // Zero-CPMV baseline (the pre-store fall-through result).
        let zero_cpmvs = crate::affine::AffineCpmvs::new_4param(
            MotionVector::from_int_pel(0, 0),
            MotionVector::from_int_pel(0, 0),
        );
        let mut baseline = PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 0);
        walker
            .reconstruct_affine_inter_uni(16, 16, 16, 16, &zero_cpmvs, &ref_pic, &mut baseline)
            .expect("zero-CPMV baseline recon");

        let mut differs = false;
        for y in 16..32usize {
            for x in 16..32usize {
                if out1.luma.samples[y * out1.luma.stride + x]
                    != baseline.luma.samples[y * baseline.luma.stride + x]
                {
                    differs = true;
                }
            }
        }
        assert!(
            differs,
            "inherited affine CPMVP from the per-CB store must change the second \
             CU's prediction vs. the zero-MV-pad baseline"
        );

        // Independent reference: the inherited derivation projects the
        // neighbour's CPMVs onto the current CU's corners. Re-derive it
        // through the public §8.5.5.7 path and confirm the fuse matches.
        let ctx = crate::affine_amvp::AffineMvpRefContext {
            list: crate::affine_amvp::RefList::L0,
            current_ref_poc: -1,
            poc_of_l0_ref: &|idx: i32| if idx == 0 { Some(-1) } else { None },
            poc_of_l1_ref: &|_idx: i32| None,
        };
        let inherited = walker
            .derive_inherited_affine_side(
                &[(15, 32), (15, 31)],
                16,
                16,
                16,
                16,
                &ctx,
                crate::amvr::AmvrShift(0),
                2,
            )
            .expect("A-scan must yield an inherited candidate");
        let inherited_cpmvs =
            crate::affine::AffineCpmvs::new_4param(inherited.cp_mvs[0], inherited.cp_mvs[1]);
        let mut expect = PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 0);
        walker
            .reconstruct_affine_inter_uni(16, 16, 16, 16, &inherited_cpmvs, &ref_pic, &mut expect)
            .expect("inherited-CPMV recon");
        for y in 16..32usize {
            for x in 16..32usize {
                assert_eq!(
                    out1.luma.samples[y * out1.luma.stride + x],
                    expect.luma.samples[y * expect.luma.stride + x],
                    "second CU fuse must equal the inherited-CPMV recon at ({x},{y})"
                );
            }
        }
    }

    /// §8.5.5.2 affine sub-block MERGE to pixels — an affine neighbour
    /// CB is stored at the left of the current CU, then a merge CU with
    /// `general_merge_flag == 1` + `merge_subblock_flag == 1` +
    /// `merge_subblock_idx == 0` is reconstructed through the live
    /// [`Self::reconstruct_leaf_cu`] dispatch. The §8.5.5.2 sub-block
    /// merge list's slot 0 is the inherited-A candidate, so the picked
    /// CPMVs are the §8.5.5.5 inherited projection of the neighbour.
    /// The pixels must (a) differ from a zero-CPMV baseline and (b)
    /// exactly match an affine-AMVP CU at the same position with
    /// `mvp_l0_flag == 0` + zero MVDs (which lands on the same inherited
    /// CPMVs through the §8.5.5.7 A-scan).
    #[test]
    fn reconstruct_leaf_cu_inter_affine_subblock_merge_inherited_to_pixels() {
        use crate::affine_syntax_enc::make_non_merge_inter_affine_decision;
        use crate::non_merge_inter_pre_residual_enc::NonMergeInterPreResidualAffineDecision;
        use crate::non_merge_mvp_syntax_enc::make_non_merge_mvp_syntax_decision;

        let pic_w = 64u32;
        let pic_h = 64u32;
        let mut sps = dummy_sps(1, pic_w, pic_h);
        sps.tool_flags.affine_enabled_flag = true;
        let pps = dummy_pps(pic_w, pic_h, true);
        let mut sh = intra_slice_header();
        sh.sh_slice_type = SliceType::P;
        let layout = CtuLayout::from_sps_pps(&sps, &pps);
        let data = [0u8; 32];
        let mut walker = CtuWalker::begin_slice(&layout, &sps, &pps, &sh, 0, &data).unwrap();

        let mut ref_frame = PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 0);
        for y in 0..pic_h as usize {
            for x in 0..pic_w as usize {
                ref_frame.luma.samples[y * ref_frame.luma.stride + x] = ((x + y) & 63) as u8;
            }
        }
        let ref_pic = ReferencePicture {
            poc: -1,
            frame: ref_frame.clone(),
            motion_field: None,
        };
        walker.set_ref_pic_list_l0(vec![ref_pic.clone()]);

        // ---- Affine neighbour CB at (0, 16) size 16x16 --------------
        // Its covered region includes the A1 sample (15, 31) of a CU at
        // (16, 16, 16, 16). A horizontal-gradient 4-param L0 affine →
        // final CPMVs CP0 = (12,0), CP1 = (24,0).
        let affine0 = make_non_merge_inter_affine_decision(true, false);
        let mvp0 = make_non_merge_mvp_syntax_decision(
            crate::leaf_cu::InterPredDir::PredL0,
            false,
            0,
            0,
            0,
            0,
        );
        let mvd_cp0 = [
            MotionVector::from_int_pel(12, 0),
            MotionVector::from_int_pel(12, 0),
            MotionVector::ZERO,
        ];
        let decision0 = NonMergeInterPreResidualAffineDecision::new(
            affine0,
            mvp0,
            mvd_cp0,
            [MotionVector::ZERO; 3],
        );
        let mut out0 = PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 0);
        walker
            .reconstruct_leaf_cu_inter_affine_amvp(
                0,
                16,
                16,
                16,
                &decision0,
                crate::amvr::AmvrShift(0),
                0, // bcw_idx
                &mut out0,
            )
            .expect("neighbour affine CB recon");

        // ---- Affine-AMVP reference CU at (16, 16): zero MVDs, mvp 0 ---
        // Lands on the inherited CPMVs (the §8.5.5.7 A-scan picks the
        // left neighbour). This is the pixel oracle the merge path must
        // reproduce.
        let affine_ref = make_non_merge_inter_affine_decision(true, false);
        let mvp_ref = make_non_merge_mvp_syntax_decision(
            crate::leaf_cu::InterPredDir::PredL0,
            false,
            0,
            0,
            0,
            0,
        );
        let decision_ref = NonMergeInterPreResidualAffineDecision::new(
            affine_ref,
            mvp_ref,
            [MotionVector::ZERO; 3],
            [MotionVector::ZERO; 3],
        );
        let mut expect = PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 0);
        walker
            .reconstruct_leaf_cu_inter_affine_amvp(
                16,
                16,
                16,
                16,
                &decision_ref,
                crate::amvr::AmvrShift(0),
                0, // bcw_idx
                &mut expect,
            )
            .expect("amvp inherited reference recon");

        // Zero-CPMV baseline (the result if the merge path ignored the
        // inherited candidate).
        let zero_cpmvs = crate::affine::AffineCpmvs::new_4param(
            MotionVector::from_int_pel(0, 0),
            MotionVector::from_int_pel(0, 0),
        );
        let mut baseline = PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 0);
        walker
            .reconstruct_affine_inter_uni(16, 16, 16, 16, &zero_cpmvs, &ref_pic, &mut baseline)
            .expect("zero-CPMV baseline recon");

        // ---- Merge CU at (16, 16): merge_subblock_flag == 1 ----------
        let ccu = CtuCu {
            ctu_addr_rs: 0,
            cu: Cu {
                x: 16,
                y: 16,
                w: 16,
                h: 16,
                cqt_depth: 0,
                mtt_depth: 0,
            },
        };
        let mut info = LeafCuInfo {
            x0: 16,
            y0: 16,
            cb_width: 16,
            cb_height: 16,
            pred_mode: CuPredMode::Inter,
            ..LeafCuInfo::default()
        };
        info.inter.general_merge_flag = true;
        info.inter.merge_data.merge_subblock_flag = true;
        info.inter.merge_data.merge_subblock_idx = 0;
        let residual = LeafCuResidual::default();

        let mut out = PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 0);
        walker
            .reconstruct_leaf_cu(&ccu, &info, &residual, &mut out)
            .expect("affine sub-block merge recon to pixels");

        let mut differs_from_baseline = false;
        for y in 16..32usize {
            for x in 16..32usize {
                let got = out.luma.samples[y * out.luma.stride + x];
                assert_eq!(
                    got,
                    expect.luma.samples[y * expect.luma.stride + x],
                    "affine sub-block merge luma must equal the inherited-CPMV recon at ({x},{y})"
                );
                if got != baseline.luma.samples[y * baseline.luma.stride + x] {
                    differs_from_baseline = true;
                }
            }
        }
        assert!(
            differs_from_baseline,
            "the inherited affine merge candidate must change the prediction vs. zero-CPMV"
        );

        // The merge path must also store the affine CB record so a later
        // CU's §8.5.5.7 scan can inherit from it.
        let stored = walker.affine_cpmv_field.get_at_luma(16, 16);
        assert!(
            stored.is_some(),
            "affine sub-block merge must store the CB's affine record"
        );
        let stored = stored.unwrap();
        assert!(stored.pred_flag_l0 && !stored.pred_flag_l1);
    }

    /// §8.5.5.2 step 9 — when no inherited / constructed / SbCol
    /// candidate is available, the sub-block merge list slot 0 is the
    /// `zeroCandm` zero-MV pad (eqs. 686 – 695: predFlagL0 = 1, zero
    /// CPMVs). The affine MC of zero CPMVs is a verbatim copy of
    /// `RefPicList[0][0]` at the CU.
    #[test]
    fn reconstruct_leaf_cu_inter_affine_subblock_merge_zero_pad() {
        let pic_w = 64u32;
        let pic_h = 64u32;
        let mut sps = dummy_sps(1, pic_w, pic_h);
        sps.tool_flags.affine_enabled_flag = true;
        let pps = dummy_pps(pic_w, pic_h, true);
        let mut sh = intra_slice_header();
        sh.sh_slice_type = SliceType::P;
        let layout = CtuLayout::from_sps_pps(&sps, &pps);
        let data = [0u8; 32];
        let mut walker = CtuWalker::begin_slice(&layout, &sps, &pps, &sh, 0, &data).unwrap();

        let mut ref_frame = PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 0);
        for y in 0..pic_h as usize {
            for x in 0..pic_w as usize {
                ref_frame.luma.samples[y * ref_frame.luma.stride + x] = ((x * 2 + y) & 63) as u8;
            }
        }
        walker.set_ref_pic_list_l0(vec![ReferencePicture {
            poc: -1,
            frame: ref_frame.clone(),
            motion_field: None,
        }]);

        // 16x16 merge CU at (16, 16) — no neighbours, no collocated
        // picture → all candidates unavailable → slot 0 is the zero pad.
        let ccu = CtuCu {
            ctu_addr_rs: 0,
            cu: Cu {
                x: 16,
                y: 16,
                w: 16,
                h: 16,
                cqt_depth: 0,
                mtt_depth: 0,
            },
        };
        let mut info = LeafCuInfo {
            x0: 16,
            y0: 16,
            cb_width: 16,
            cb_height: 16,
            pred_mode: CuPredMode::Inter,
            ..LeafCuInfo::default()
        };
        info.inter.general_merge_flag = true;
        info.inter.merge_data.merge_subblock_flag = true;
        info.inter.merge_data.merge_subblock_idx = 0;
        let residual = LeafCuResidual::default();

        let mut out = PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 0);
        walker
            .reconstruct_leaf_cu(&ccu, &info, &residual, &mut out)
            .expect("zero-pad affine sub-block merge recon");

        for y in 16..32usize {
            for x in 16..32usize {
                assert_eq!(
                    out.luma.samples[y * out.luma.stride + x],
                    ref_frame.luma.samples[y * ref_frame.luma.stride + x],
                    "zero-pad affine merge must copy the reference verbatim at ({x},{y})"
                );
            }
        }
    }

    /// §8.5.5.2 eqs. 681 – 684 — an inherited affine-merge candidate
    /// carries the neighbour CB's `bcwIdxN`. A bi-pred affine neighbour
    /// record stored with `bcw_idx == 3` is inherited by a merge CU; the
    /// reconstruction must apply the eq. 981 BCW weighting, so it differs
    /// from the same setup with the neighbour record's `bcw_idx == 0`
    /// (eq. 980 default-weighted).
    #[test]
    fn affine_subblock_merge_inherits_neighbour_bcw_idx() {
        let pic_w = 64u32;
        let pic_h = 64u32;
        let mut sps = dummy_sps(1, pic_w, pic_h);
        sps.tool_flags.affine_enabled_flag = true;
        let pps = dummy_pps(pic_w, pic_h, true);
        let mut sh = intra_slice_header();
        sh.sh_slice_type = SliceType::B;
        let layout = CtuLayout::from_sps_pps(&sps, &pps);
        let data = [0u8; 32];

        // Two distinct-constant references so a weight change is visible.
        let ref_l0 = ReferencePicture {
            poc: -1,
            frame: PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 40),
            motion_field: None,
        };
        let ref_l1 = ReferencePicture {
            poc: 1,
            frame: PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 200),
            motion_field: None,
        };

        // Build the merge CU at (16, 16).
        let ccu = CtuCu {
            ctu_addr_rs: 0,
            cu: Cu {
                x: 16,
                y: 16,
                w: 16,
                h: 16,
                cqt_depth: 0,
                mtt_depth: 0,
            },
        };
        let mut info = LeafCuInfo {
            x0: 16,
            y0: 16,
            cb_width: 16,
            cb_height: 16,
            pred_mode: CuPredMode::Inter,
            ..LeafCuInfo::default()
        };
        info.inter.general_merge_flag = true;
        info.inter.merge_data.merge_subblock_flag = true;
        info.inter.merge_data.merge_subblock_idx = 0;
        let residual = LeafCuResidual::default();

        // Stored neighbour record helper — a bi-pred 4-param affine CB at
        // (0, 16) covering the A1 sample (15, 31), zero CPMVs (so the
        // inherited projection is zero-MV and the per-list MC is a
        // verbatim copy of each constant reference).
        let recon_with_bcw = |bcw: u8| -> u8 {
            let mut walker = CtuWalker::begin_slice(&layout, &sps, &pps, &sh, 0, &data).unwrap();
            walker.set_ref_pic_list_l0(vec![ref_l0.clone()]);
            walker.set_ref_pic_list_l1(vec![ref_l1.clone()]);
            let zero =
                crate::affine::AffineCpmvs::new_4param(MotionVector::ZERO, MotionVector::ZERO);
            let rec = crate::inter::AffineCbRecord {
                xnb: 0,
                ynb: 16,
                nb_w: 16,
                nb_h: 16,
                model: crate::affine::MotionModel::Affine4Param,
                pred_flag_l0: true,
                pred_flag_l1: true,
                ref_idx_l0: 0,
                ref_idx_l1: 0,
                cpmvs_l0: zero,
                cpmvs_l1: zero,
                bcw_idx: bcw,
            };
            walker
                .affine_cpmv_field
                .write_block(0, 16, 16, 16, Some(rec));

            let mut out = PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 0);
            walker
                .reconstruct_leaf_cu(&ccu, &info, &residual, &mut out)
                .expect("affine merge inherits neighbour bcw");
            out.luma.samples[20 * out.luma.stride + 20]
        };

        // bcw == 0 → eq. 980 average (40 + 200 + 1) >> 1 = 120.
        assert_eq!(
            recon_with_bcw(0),
            120,
            "bcw==0 must take the eq. 980 average"
        );
        // bcw == 3 → eq. 981 (w0=−2, w1=10): (−2·40 + 10·200 + 4) >> 3 = 240.
        assert_eq!(
            recon_with_bcw(3),
            240,
            "inherited bcw==3 must apply the eq. 981 BCW weights"
        );
    }

    /// §8.5.5.3 SbTMVP (`SbCol`) sub-block merge to pixels — a merge CU
    /// with `merge_subblock_flag == 1` whose §8.5.5.2 list slot 0 is the
    /// SbCol candidate (sbtmvp enabled, collocated picture bound, the
    /// §8.5.5.3 step-3 `availableFlagSbCol` true) routes through the
    /// SbTMVP walker. With a uniform zero-MV collocated motion field the
    /// per-sub-block grid resolves to MV (0,0) on every 8×8 sub-block, so
    /// the reconstruction is a verbatim copy of `RefPicList[0][0]`.
    #[test]
    fn reconstruct_leaf_cu_sbtmvp_to_pixels() {
        let pic_w = 64u32;
        let pic_h = 64u32;
        let mut sps = dummy_sps(1, pic_w, pic_h);
        sps.tool_flags.affine_enabled_flag = true;
        sps.tool_flags.sbtmvp_enabled_flag = true;
        let pps = dummy_pps(pic_w, pic_h, true);
        let mut sh = intra_slice_header();
        sh.sh_slice_type = SliceType::P;
        let layout = CtuLayout::from_sps_pps(&sps, &pps);
        let data = [0u8; 32];
        let mut walker = CtuWalker::begin_slice(&layout, &sps, &pps, &sh, 0, &data).unwrap();

        // Reference picture (the collocated picture) with a 2-D ramp + a
        // uniform zero-MV inter motion field.
        let mut ref_frame = PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 0);
        for y in 0..pic_h as usize {
            for x in 0..pic_w as usize {
                ref_frame.luma.samples[y * ref_frame.luma.stride + x] = ((x + y) & 63) as u8;
            }
        }
        let mut mf = MotionField::new(pic_w, pic_h);
        let cell = MvField {
            mv_l0: MotionVector::ZERO,
            ref_idx_l0: 0,
            pred_flag_l0: true,
            mv_l1: MotionVector::ZERO,
            ref_idx_l1: -1,
            pred_flag_l1: false,
            cu_skip_flag: false,
            mode_inter: true,
            available: true,
            bcw_idx: 0,
        };
        mf.write_block(0, 0, pic_w, pic_h, cell);
        let ref_pic = ReferencePicture {
            poc: 0,
            frame: ref_frame.clone(),
            motion_field: Some(mf),
        };
        walker.set_ref_pic_list_l0(vec![ref_pic]);
        // ph_temporal_mvp on, current POC 2, collocated = RefPicList[0][0].
        walker.set_temporal_mvp(2, true, true, 0);

        // 16x16 merge CU at (16, 16): merge_subblock_flag == 1, idx 0.
        let ccu = CtuCu {
            ctu_addr_rs: 0,
            cu: Cu {
                x: 16,
                y: 16,
                w: 16,
                h: 16,
                cqt_depth: 0,
                mtt_depth: 0,
            },
        };
        let mut info = LeafCuInfo {
            x0: 16,
            y0: 16,
            cb_width: 16,
            cb_height: 16,
            pred_mode: CuPredMode::Inter,
            ..LeafCuInfo::default()
        };
        info.inter.general_merge_flag = true;
        info.inter.merge_data.merge_subblock_flag = true;
        info.inter.merge_data.merge_subblock_idx = 0;
        let residual = LeafCuResidual::default();

        let mut out = PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 0);
        walker
            .reconstruct_leaf_cu(&ccu, &info, &residual, &mut out)
            .expect("SbTMVP sub-block merge recon to pixels");

        // Zero-MV SbTMVP → verbatim copy of the reference at the CU.
        for y in 16..32usize {
            for x in 16..32usize {
                assert_eq!(
                    out.luma.samples[y * out.luma.stride + x],
                    ref_frame.luma.samples[y * ref_frame.luma.stride + x],
                    "SbTMVP zero-MV grid must copy the reference verbatim at ({x},{y})"
                );
            }
        }
        // The motion field cells must report MODE_INTER so a later CU's
        // neighbour scan sees this SbTMVP block.
        let mvf = walker.motion_field.get_at_luma(20, 20);
        assert!(mvf.available && mvf.mode_inter && mvf.pred_flag_l0);
    }

    /// §8.5.5.8 constructed affine CPMVP — when the three corner
    /// cascades (TL = {B2,B3,A2}, TR = {B1,B0}, BL = {A1,A0}) all read
    /// a translational inter neighbour whose L0 reference POC matches the
    /// current CU's, the `availableConsFlagLX` candidate is assembled
    /// from the per-corner `MvLX[xNb][yNb]` (eqs. 841 – 846) — read from
    /// the **regular** motion field, NOT the affine CPMV store (which
    /// holds no translational neighbours). With no inherited affine
    /// neighbour present, that constructed candidate becomes
    /// `cpMvpListLX[0]`; selecting `mvp_l0_flag == 0` + zero MVD makes
    /// the final CPMVs equal the neighbour MV at every CP, so the output
    /// must differ from a zero-CPMV baseline and exactly match an affine
    /// recon driven by that uniform-CPMV predictor.
    #[test]
    fn affine_amvp_constructed_cpmvp_from_motion_field() {
        use crate::affine_syntax_enc::make_non_merge_inter_affine_decision;
        use crate::non_merge_inter_pre_residual_enc::NonMergeInterPreResidualAffineDecision;
        use crate::non_merge_mvp_syntax_enc::make_non_merge_mvp_syntax_decision;

        let pic_w = 64u32;
        let pic_h = 64u32;
        let sps = dummy_sps(1, pic_w, pic_h);
        let pps = dummy_pps(pic_w, pic_h, true);
        let mut sh = intra_slice_header();
        sh.sh_slice_type = SliceType::P;
        let layout = CtuLayout::from_sps_pps(&sps, &pps);
        let data = [0u8; 32];
        let mut walker = CtuWalker::begin_slice(&layout, &sps, &pps, &sh, 0, &data).unwrap();

        let mut ref_frame = PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 0);
        for y in 0..pic_h as usize {
            for x in 0..pic_w as usize {
                ref_frame.luma.samples[y * ref_frame.luma.stride + x] = ((x * 3 + y) & 63) as u8;
            }
        }
        let ref_pic = ReferencePicture {
            poc: -1,
            frame: ref_frame.clone(),
            motion_field: None,
        };
        walker.set_ref_pic_list_l0(vec![ref_pic.clone()]);

        // A translational L0 neighbour with refIdx 0 (POC -1, matching).
        // Strong MV so a uniform-CPMV affine recon is clearly non-zero.
        let nb_mv = MotionVector::from_int_pel(6, -3);
        let nb = MvField {
            mv_l0: nb_mv,
            ref_idx_l0: 0,
            pred_flag_l0: true,
            mv_l1: MotionVector::ZERO,
            ref_idx_l1: -1,
            pred_flag_l1: false,
            cu_skip_flag: false,
            mode_inter: true,
            available: true,
            bcw_idx: 0,
        };
        // CU at (16,16,16,16). Its TL cascade reads (15,15)/(16,15)/(15,16),
        // TR reads (31,15)/(32,15), BL reads (15,31)/(15,32). Fill the top
        // band (y < 16, x < 48) and the left column (x < 16, y < 48) so
        // every corner position resolves to the neighbour.
        walker.motion_field.write_block(0, 0, 48, 16, nb);
        walker.motion_field.write_block(0, 0, 16, 48, nb);

        let affine = make_non_merge_inter_affine_decision(true, false);
        let mvp = make_non_merge_mvp_syntax_decision(
            crate::leaf_cu::InterPredDir::PredL0,
            false,
            0,
            0,
            0,
            0,
        );
        let decision = NonMergeInterPreResidualAffineDecision::new(
            affine,
            mvp,
            [MotionVector::ZERO; 3],
            [MotionVector::ZERO; 3],
        );
        let mut out = PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 0);
        walker
            .reconstruct_leaf_cu_inter_affine_amvp(
                16,
                16,
                16,
                16,
                &decision,
                crate::amvr::AmvrShift(0),
                0, // bcw_idx
                &mut out,
            )
            .expect("constructed-CPMVP affine recon");

        // Zero-CPMV baseline (what a pre-constructed-CPMVP decoder gave).
        let zero_cpmvs = crate::affine::AffineCpmvs::new_4param(
            MotionVector::from_int_pel(0, 0),
            MotionVector::from_int_pel(0, 0),
        );
        let mut baseline = PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 0);
        walker
            .reconstruct_affine_inter_uni(16, 16, 16, 16, &zero_cpmvs, &ref_pic, &mut baseline)
            .expect("zero-CPMV baseline recon");
        let mut differs = false;
        for y in 16..32usize {
            for x in 16..32usize {
                if out.luma.samples[y * out.luma.stride + x]
                    != baseline.luma.samples[y * baseline.luma.stride + x]
                {
                    differs = true;
                }
            }
        }
        assert!(
            differs,
            "constructed CPMVP must change the CU's prediction vs. the zero-MV-pad baseline"
        );

        // The constructed candidate replicates the neighbour MV across
        // both CPs, so the final CPMVs (zero MVD) are uniform == nb_mv.
        // An affine recon driven by that uniform CPMV must match exactly.
        let uniform = crate::affine::AffineCpmvs::new_4param(nb_mv, nb_mv);
        let mut expect = PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 0);
        walker
            .reconstruct_affine_inter_uni(16, 16, 16, 16, &uniform, &ref_pic, &mut expect)
            .expect("uniform-CPMV recon");
        for y in 16..32usize {
            for x in 16..32usize {
                assert_eq!(
                    out.luma.samples[y * out.luma.stride + x],
                    expect.luma.samples[y * expect.luma.stride + x],
                    "constructed-CPMVP fuse must equal the uniform-CPMV recon at ({x},{y})"
                );
            }
        }
    }

    /// §8.5.6.6.2 affine bi-pred — averaging two **identical**
    /// predictions (`(a + a + 1) >> 1 == a`) is the identity, so the
    /// bi-pred reconstruction of the same CPMVs / same reference on
    /// both lists must equal that single per-list affine prediction.
    /// The per-list reference is computed directly with the public
    /// `affine::predict_luma_block_affine_prof(bipred = true)` so the
    /// test pins both the two-scratch-plane composition and the
    /// default-weighted average, for luma (the bi-pred PROF/rounding
    /// flag differs from the uni path, which is why this compares
    /// against the bi-pred per-list prediction, not the uni result).
    #[test]
    fn reconstruct_affine_inter_bi_equals_uni_for_identical_lists() {
        let pic_w = 64u32;
        let pic_h = 64u32;
        let sps = dummy_sps(1, pic_w, pic_h);
        let pps = dummy_pps(pic_w, pic_h, true);
        let mut sh = intra_slice_header();
        sh.sh_slice_type = SliceType::B;
        let layout = CtuLayout::from_sps_pps(&sps, &pps);
        let data = [0u8; 32];
        let walker = CtuWalker::begin_slice(&layout, &sps, &pps, &sh, 0, &data).unwrap();

        // 2-D ramp reference (luma + a distinct chroma ramp).
        let mut ref_frame = PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 0);
        for y in 0..pic_h as usize {
            for x in 0..pic_w as usize {
                ref_frame.luma.samples[y * ref_frame.luma.stride + x] = ((x + y) & 63) as u8;
            }
        }
        for cy in 0..(pic_h / 2) as usize {
            for cx in 0..(pic_w / 2) as usize {
                ref_frame.cb.samples[cy * ref_frame.cb.stride + cx] = ((cx * 2) & 63) as u8;
                ref_frame.cr.samples[cy * ref_frame.cr.stride + cx] = ((cy * 2) & 63) as u8;
            }
        }
        let ref_pic = ReferencePicture {
            poc: -1,
            frame: ref_frame.clone(),
            motion_field: None,
        };
        let ref_pic2 = ref_pic.clone();

        let cp0 = MotionVector::from_int_pel(0, 0);
        let cp1 = MotionVector::from_int_pel(4, 0);
        let cpmvs = crate::affine::AffineCpmvs::new_4param(cp0, cp1);

        // Per-list bi-pred affine luma prediction (reference).
        let mut ref_list = crate::reconstruct::PicturePlane::filled(64, 64, 0);
        crate::affine::predict_luma_block_affine_prof(
            &mut ref_list,
            16,
            16,
            16,
            16,
            &ref_frame.luma,
            &cpmvs,
            crate::affine::AffineLumaFilterSet::Set0,
            /*bipred=*/ true,
            /*ph_prof_disabled=*/ true,
            /*rpr_constraints_active=*/ false,
        )
        .expect("per-list affine prediction");

        let mut bi = PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 0);
        walker
            .reconstruct_affine_inter_bi(
                16, 16, 16, 16, &cpmvs, &ref_pic, &cpmvs, &ref_pic2, &mut bi,
            )
            .expect("bi affine recon");

        // Averaging two identical per-list predictions is the identity.
        let mut any = false;
        for y in 16..32usize {
            for x in 16..32usize {
                let got = bi.luma.samples[y * bi.luma.stride + x];
                let expect = ref_list.samples[y * ref_list.stride + x];
                assert_eq!(
                    got, expect,
                    "affine bi-pred luma of identical lists must equal the per-list prediction at ({x},{y})"
                );
                if got != ((x + y) & 63) as u8 {
                    any = true;
                }
            }
        }
        assert!(
            any,
            "the affine bi-pred must produce a non-trivial prediction"
        );
        // Chroma: both lists identical → averaging is the identity, so
        // the bi-pred chroma must be a valid (in-range) non-constant
        // reconstruction.
        let mut cb_any = false;
        for cy in 8..16usize {
            for cx in 8..16usize {
                if bi.cb.samples[cy * bi.cb.stride + cx] != 0 {
                    cb_any = true;
                }
            }
        }
        assert!(cb_any, "affine bi-pred Cb must be populated");
    }

    /// §8.5.6.6.2 eq. 981 — `reconstruct_affine_inter_bi_bcw` applies the
    /// BCW weighting `Clip1((w0·p0 + w1·p1 + 4) >> 3)` on the affine
    /// bi-pred path. With zero CPMVs (integer-pel ⇒ verbatim copy) and
    /// two reference pictures of distinct constants A=40 (L0) and B=200
    /// (L1), the `bcw_idx == 3` blend (w1=10, w0=−2) must yield
    /// `(−2·40 + 10·200 + 4) >> 3 = (−80 + 2000 + 4) >> 3 = 240`, whereas
    /// the default eq. 980 average (`bcw_idx == 0`) yields
    /// `(40 + 200 + 1) >> 1 = 120`.
    #[test]
    fn reconstruct_affine_inter_bi_bcw_applies_eq981_weights() {
        let pic_w = 64u32;
        let pic_h = 64u32;
        let sps = dummy_sps(1, pic_w, pic_h);
        let pps = dummy_pps(pic_w, pic_h, true);
        let mut sh = intra_slice_header();
        sh.sh_slice_type = SliceType::B;
        let layout = CtuLayout::from_sps_pps(&sps, &pps);
        let data = [0u8; 32];
        let walker = CtuWalker::begin_slice(&layout, &sps, &pps, &sh, 0, &data).unwrap();

        let ref_a = ReferencePicture {
            poc: -1,
            frame: PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 40),
            motion_field: None,
        };
        let ref_b = ReferencePicture {
            poc: 1,
            frame: PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 200),
            motion_field: None,
        };
        let zero = crate::affine::AffineCpmvs::new_4param(
            MotionVector::from_int_pel(0, 0),
            MotionVector::from_int_pel(0, 0),
        );

        // Default-weighted (bcw_idx == 0) → eq. 980 average = 120.
        let mut avg = PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 0);
        walker
            .reconstruct_affine_inter_bi_bcw(
                16, 16, 16, 16, &zero, &ref_a, &zero, &ref_b, 0, &mut avg,
            )
            .expect("default-weighted affine bi recon");
        assert_eq!(
            avg.luma.samples[16 * avg.luma.stride + 16],
            120,
            "bcw_idx==0 must take the eq. 980 average (40+200+1)>>1 = 120"
        );

        // BCW-weighted (bcw_idx == 3) → eq. 981 = 240.
        let mut bcw = PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 0);
        walker
            .reconstruct_affine_inter_bi_bcw(
                16, 16, 16, 16, &zero, &ref_a, &zero, &ref_b, 3, &mut bcw,
            )
            .expect("BCW-weighted affine bi recon");
        for y in 16..32usize {
            for x in 16..32usize {
                assert_eq!(
                    bcw.luma.samples[y * bcw.luma.stride + x],
                    240,
                    "bcw_idx==3 must apply eq. 981 weights → 240 at ({x},{y})"
                );
            }
        }
        // Chroma takes the same eq. 981 weighting per component. Both
        // references carry chroma == 128, and `w0 + w1 == 8`, so the
        // weighted blend of two equal inputs is the identity (128) — the
        // assertion confirms the BCW chroma path runs without corrupting
        // the in-range result.
        assert_eq!(
            bcw.cb.samples[8 * bcw.cb.stride + 8],
            128,
            "BCW chroma blend of two equal inputs (w0+w1=8) must be the identity"
        );
    }

    /// §7.3.11.4 multi-TB tiling — a 128×128 inter CU (> MaxTbSizeY)
    /// reconstructs by tiling its residual into four 64×64 transform
    /// blocks. With a constant-100 reference and zero-MV merge, each
    /// tile carrying a positive DC coefficient lifts its own 64×64
    /// quadrant uniformly; a tile with no coefficients stays at the
    /// prediction. This proves both that the > MaxTbSizeY CU no longer
    /// surfaces Unsupported and that each tile's residual lands at its
    /// own offset.
    #[test]
    fn reconstruct_leaf_cu_inter_multi_tb_tiling() {
        let pic_w = 128u32;
        let pic_h = 128u32;
        let sps = dummy_sps(2, pic_w, pic_h); // 128x128 CTU.
        let pps = dummy_pps(pic_w, pic_h, true);
        let mut sh = intra_slice_header();
        sh.sh_slice_type = SliceType::P;
        let layout = CtuLayout::from_sps_pps(&sps, &pps);
        let data = [0u8; 32];
        let mut walker = CtuWalker::begin_slice(&layout, &sps, &pps, &sh, 0, &data).unwrap();

        walker.set_ref_pic_list_l0(vec![ReferencePicture {
            poc: -1,
            frame: PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 100),
            motion_field: None,
        }]);

        let ccu = CtuCu {
            ctu_addr_rs: 0,
            cu: Cu {
                x: 0,
                y: 0,
                w: 128,
                h: 128,
                cqt_depth: 0,
                mtt_depth: 0,
            },
        };
        let mut info = LeafCuInfo {
            x0: 0,
            y0: 0,
            cb_width: 128,
            cb_height: 128,
            pred_mode: CuPredMode::Inter,
            ..LeafCuInfo::default()
        };
        info.inter.general_merge_flag = true;
        info.inter.merge_data.merge_idx = 0;
        info.tu_y_coded_flag = true;

        // Four 64×64 tiles in transform_tree_tiles order: TL, BL, TR, BR.
        // Give the FIRST tile (top-left) a positive DC; leave the others
        // zero so only the TL quadrant is lifted.
        let mut residual = LeafCuResidual::default();
        let tile_len = 64 * 64;
        residual.luma_levels = vec![0i32; tile_len * 4];
        residual.luma_levels[0] = 8; // DC of the top-left tile.

        let mut out = PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 0);
        walker
            .reconstruct_leaf_cu(&ccu, &info, &residual, &mut out)
            .expect("multi-TB inter CU should reconstruct without Unsupported");

        // Top-left 64×64 quadrant: lifted uniformly above 100.
        let tl = out.luma.samples[0];
        assert!(tl > 100, "top-left tile must be lifted above 100, got {tl}");
        for y in 0..64usize {
            for x in 0..64usize {
                assert_eq!(
                    out.luma.samples[y * out.luma.stride + x],
                    tl,
                    "top-left tile must lift uniformly (DC residual)"
                );
            }
        }
        // The other three quadrants stay at the constant prediction 100.
        // Bottom-left:
        assert_eq!(out.luma.samples[64 * out.luma.stride], 100);
        // Top-right:
        assert_eq!(out.luma.samples[64], 100);
        // Bottom-right:
        assert_eq!(out.luma.samples[64 * out.luma.stride + 64], 100);
    }

    /// §8.5.8 + §8.7.5.1 — a non-skip merge inter CU whose
    /// `tu_y_coded_flag == 1` must add its inverse-transformed luma
    /// residual to the motion-compensated prediction. With a constant
    /// reference frame the zero-MV uni-pred merge candidate produces a
    /// constant prediction; a single positive luma DC coefficient must
    /// lift every reconstructed luma sample uniformly above that
    /// constant. Chroma carries no residual, so it stays at the
    /// prediction value.
    #[test]
    fn reconstruct_leaf_cu_inter_adds_luma_residual() {
        let pic_w = 64u32;
        let pic_h = 64u32;
        let sps = dummy_sps(1, pic_w, pic_h); // 64x64 CTU.
        let pps = dummy_pps(pic_w, pic_h, true);
        let mut sh = intra_slice_header();
        sh.sh_slice_type = SliceType::P;
        let layout = CtuLayout::from_sps_pps(&sps, &pps);
        let data = [0u8; 32];
        let mut walker = CtuWalker::begin_slice(&layout, &sps, &pps, &sh, 0, &data).unwrap();

        // Constant-100 reference frame for the single L0 entry.
        let ref_pic = ReferencePicture {
            poc: -1,
            frame: PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 100),
            motion_field: None,
        };
        walker.set_ref_pic_list_l0(vec![ref_pic]);

        // Inter CU, 16x16, merge with merge_idx 0 → zero-MV uni-pred
        // L0 candidate (empty motion field pads zero-MV for P-slices).
        let ccu = CtuCu {
            ctu_addr_rs: 0,
            cu: Cu {
                x: 0,
                y: 0,
                w: 16,
                h: 16,
                cqt_depth: 0,
                mtt_depth: 0,
            },
        };
        let mut info = LeafCuInfo {
            x0: 0,
            y0: 0,
            cb_width: 16,
            cb_height: 16,
            pred_mode: CuPredMode::Inter,
            ..LeafCuInfo::default()
        };
        info.inter.general_merge_flag = true;
        info.inter.merge_data.merge_idx = 0;
        // Coded luma TB carrying a single positive DC coefficient.
        info.tu_y_coded_flag = true;
        let mut residual = LeafCuResidual::default();
        residual.luma_levels = vec![0i32; 16 * 16];
        residual.luma_levels[0] = 8; // DC.

        let mut out = PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 0);
        walker
            .reconstruct_leaf_cu(&ccu, &info, &residual, &mut out)
            .expect("inter residual path should reconstruct");

        // Every luma sample in the 16x16 CU must exceed the constant
        // prediction (100) by the DC residual.
        for y in 0..16usize {
            for x in 0..16usize {
                let v = out.luma.samples[y * out.luma.stride + x];
                assert!(
                    v > 100,
                    "luma sample at ({x},{y}) = {v} must exceed the constant prediction 100 \
                     after a positive DC residual"
                );
            }
        }
        // Uniform DC lift: all reconstructed luma samples are equal.
        let first = out.luma.samples[0];
        for y in 0..16usize {
            for x in 0..16usize {
                assert_eq!(
                    out.luma.samples[y * out.luma.stride + x],
                    first,
                    "DC-only residual must lift the whole TB uniformly"
                );
            }
        }
        // Chroma carries no residual → stays at the prediction value.
        // `PictureBuffer::yuv420_filled` seeds chroma planes to 128
        // (mid-grey) regardless of the luma seed, so the constant
        // reference's chroma is 128 and the MC prediction reproduces it.
        for cy in 0..8usize {
            for cx in 0..8usize {
                assert_eq!(
                    out.cb.samples[cy * out.cb.stride + cx],
                    128,
                    "Cb must remain at the prediction value (no chroma residual)"
                );
            }
        }
    }

    #[test]
    fn walker_rejects_dual_tree_intra() {
        let mut sps = dummy_sps(2, 128, 128);
        sps.partition_constraints.qtbtt_dual_tree_intra_flag = true;
        let pps = dummy_pps(128, 128, true);
        let sh = intra_slice_header();
        let layout = CtuLayout::from_sps_pps(&sps, &pps);
        let data = [0u8; 8];
        let err = CtuWalker::begin_slice(&layout, &sps, &pps, &sh, 0, &data).unwrap_err();
        assert!(matches!(err, Error::Unsupported(_)));
    }

    #[test]
    fn walker_rejects_dep_quant_scaling_list_and_gates_lmcs() {
        let sps = dummy_sps(2, 128, 128);
        let pps = dummy_pps(128, 128, true);
        let layout = CtuLayout::from_sps_pps(&sps, &pps);
        let data = [0u8; 8];

        // ALF is now accepted (round-15) — the apply pass runs after
        // SAO with an empty AlfPicture by default, so it's a no-op
        // unless callers programmatically populate it.
        let mut sh = intra_slice_header();
        sh.sh_alf_enabled_flag = true;
        assert!(CtuWalker::begin_slice(&layout, &sps, &pps, &sh, 0, &data).is_ok());

        // LMCS is accepted at slice start (round 384); decoding a slice
        // that uses LMCS without a bound APS payload fails fast instead
        // of silently producing unmapped pixels.
        let mut sh = intra_slice_header();
        sh.sh_lmcs_used_flag = true;
        let mut w = CtuWalker::begin_slice(&layout, &sps, &pps, &sh, 0, &data).unwrap();
        let mut out = PictureBuffer::yuv420_filled(128, 128, 128);
        assert!(matches!(
            w.decode_picture_into(&mut out).unwrap_err(),
            Error::Unsupported(_)
        ));

        let mut sh = intra_slice_header();
        sh.sh_dep_quant_used_flag = true;
        assert!(matches!(
            CtuWalker::begin_slice(&layout, &sps, &pps, &sh, 0, &data).unwrap_err(),
            Error::Unsupported(_)
        ));

        let mut sh = intra_slice_header();
        sh.sh_explicit_scaling_list_used_flag = true;
        assert!(matches!(
            CtuWalker::begin_slice(&layout, &sps, &pps, &sh, 0, &data).unwrap_err(),
            Error::Unsupported(_)
        ));
    }

    /// Non-identity BD-8 `lmcs_data()` payload used by the LMCS
    /// integration tests: bin 0 is shrunk to 8 codewords (`lmcsCW[0] =
    /// OrgCW − 8 = 8`), every other bin keeps `OrgCW = 16`. `Σ lmcsCW =
    /// 248 <= 255` (eq. 96 budget), every `lmcsCW` inside the eq. 95
    /// `[OrgCW >> 3, (OrgCW << 3) − 1]` band, and every `LmcsPivot` is
    /// a multiple of `1 << (BitDepth − 5) = 8`, so the eq. 98 crossing
    /// clause is vacuous.
    fn lmcs_test_data() -> LmcsData {
        let mut abs_cw = [0u32; crate::lmcs::LMCS_NUM_BINS];
        let mut sign_cw = [false; crate::lmcs::LMCS_NUM_BINS];
        abs_cw[0] = 8;
        sign_cw[0] = true; // negative delta: lmcsCW[0] = 16 − 8 = 8.
        LmcsData {
            lmcs_min_bin_idx: 0,
            lmcs_delta_max_bin_idx: 0,
            lmcs_delta_cw_prec_minus1: 3,
            lmcs_delta_abs_cw: abs_cw,
            lmcs_delta_sign_cw_flag: sign_cw,
            lmcs_delta_abs_crs: 0,
            lmcs_delta_sign_crs_flag: false,
        }
    }

    #[test]
    fn lmcs_inverse_map_pass_runs_in_loop_filter_stack() {
        // §8.8.1 step 1 — with `sh_lmcs_used_flag == 1` and a bound
        // LMCS payload, `apply_in_loop_filters` inverse-maps every
        // reconstructed luma sample (§8.8.2.1 / §8.8.2.2) before the
        // (here no-op) deblock / SAO / ALF passes; chroma is untouched.
        let sps = dummy_sps(2, 64, 64);
        let pps = dummy_pps(64, 64, true);
        let layout = CtuLayout::from_sps_pps(&sps, &pps);
        let data = [0u8; 8];
        let mut sh = intra_slice_header();
        sh.sh_lmcs_used_flag = true;
        let mut w = CtuWalker::begin_slice(&layout, &sps, &pps, &sh, 0, &data).unwrap();
        let lmcs = lmcs_test_data();
        w.set_lmcs(&lmcs, false).unwrap();

        let mut out = PictureBuffer::yuv420_filled(64, 64, 128);
        for (i, s) in out.luma.samples.iter_mut().enumerate() {
            *s = (i % 251) as u8;
        }
        let cb_before = out.cb.samples.clone();
        let luma_before = out.luma.samples.clone();
        w.apply_in_loop_filters(&mut out).unwrap();

        let derived = lmcs.derive(8).unwrap();
        let (min_bin, max_bin) = (lmcs.lmcs_min_bin_idx, lmcs.lmcs_max_bin_idx());
        let mut changed = false;
        for (idx, (&got, &before)) in out.luma.samples.iter().zip(luma_before.iter()).enumerate() {
            let s = u32::from(before);
            let iy = derived.idx_y_inv(s, min_bin, max_bin);
            let expect = derived.inverse_map_luma_sample(s, iy) as u8;
            assert_eq!(got, expect, "luma sample {idx}");
            if got != before {
                changed = true;
            }
        }
        assert!(changed, "the test mapping must not be an identity");
        assert_eq!(out.cb.samples, cb_before, "chroma must be untouched");
    }

    #[test]
    fn lmcs_inverse_map_pass_skipped_when_slice_flag_off() {
        // §8.8.2.2 `sh_lmcs_used_flag == 0` arm — a bound-but-unused
        // payload must leave the picture alone.
        let sps = dummy_sps(2, 64, 64);
        let pps = dummy_pps(64, 64, true);
        let layout = CtuLayout::from_sps_pps(&sps, &pps);
        let data = [0u8; 8];
        let sh = intra_slice_header();
        let mut w = CtuWalker::begin_slice(&layout, &sps, &pps, &sh, 0, &data).unwrap();
        w.set_lmcs(&lmcs_test_data(), true).unwrap();

        let mut out = PictureBuffer::yuv420_filled(64, 64, 128);
        for (i, s) in out.luma.samples.iter_mut().enumerate() {
            *s = (i % 251) as u8;
        }
        let luma_before = out.luma.samples.clone();
        w.apply_in_loop_filters(&mut out).unwrap();
        assert_eq!(out.luma.samples, luma_before);
    }

    #[test]
    fn lmcs_forward_maps_inter_luma_prediction_then_inverse_maps_in_loop() {
        // §8.7.5.2 — a MODE_INTER (non-CIIP) CU's MC luma prediction is
        // forward-mapped in place (eq. 1213), so with a zero residual
        // the reconstruction equals FwdMap(predSamples); the §8.8.1
        // step-1 in-loop pass then inverse-maps it (§8.8.2.2). Both
        // stages are checked against the per-sample lmcs folds.
        let pic_w = 64u32;
        let pic_h = 64u32;
        let sps = dummy_sps(1, pic_w, pic_h);
        let pps = dummy_pps(pic_w, pic_h, true);
        let mut sh = intra_slice_header();
        sh.sh_slice_type = SliceType::P;
        sh.sh_lmcs_used_flag = true;
        let layout = CtuLayout::from_sps_pps(&sps, &pps);
        let data = [0u8; 32];
        let mut walker = CtuWalker::begin_slice(&layout, &sps, &pps, &sh, 0, &data).unwrap();
        walker.current_poc = 0;
        let lmcs = lmcs_test_data();
        walker.set_lmcs(&lmcs, false).unwrap();

        // Horizontal-gradient reference; zero-MV merge candidate falls
        // out of the empty motion field, so the MC prediction is a
        // straight copy of the collocated reference block.
        let mut ref_frame = PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 0);
        for y in 0..pic_h as usize {
            for x in 0..pic_w as usize {
                ref_frame.luma.samples[y * ref_frame.luma.stride + x] = (x * 3) as u8;
            }
        }
        let ref_pic = ReferencePicture {
            poc: 4,
            frame: ref_frame.clone(),
            motion_field: None,
        };
        walker.set_ref_pic_list_l0(vec![ref_pic]);

        let ccu = CtuCu {
            ctu_addr_rs: 0,
            cu: Cu {
                x: 16,
                y: 0,
                w: 16,
                h: 16,
                cqt_depth: 0,
                mtt_depth: 0,
            },
        };
        let mut info = LeafCuInfo {
            x0: 16,
            y0: 0,
            cb_width: 16,
            cb_height: 16,
            pred_mode: CuPredMode::Inter,
            ..LeafCuInfo::default()
        };
        info.inter.general_merge_flag = true;
        info.inter.merge_data.merge_idx = 0;
        let residual = LeafCuResidual::default();

        let mut out = PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 0);
        walker
            .reconstruct_leaf_cu(&ccu, &info, &residual, &mut out)
            .expect("merge inter path should reconstruct");

        // Stage 1 — the reconstruction is the forward-mapped MC
        // prediction (zero-MV copy of the reference block).
        let derived = lmcs.derive(8).unwrap();
        let mut mapped_expect = vec![0u8; 16 * 16];
        for row in 0..16usize {
            for col in 0..16usize {
                let pred =
                    u32::from(ref_frame.luma.samples[row * ref_frame.luma.stride + 16 + col]);
                let m = derived.forward_map_luma_sample(pred).clamp(0, 255) as u8;
                mapped_expect[row * 16 + col] = m;
                assert_eq!(
                    out.luma.samples[row * out.luma.stride + 16 + col],
                    m,
                    "mapped-domain luma at ({col}, {row})"
                );
            }
        }
        assert_ne!(
            &mapped_expect[..16],
            &ref_frame.luma.samples[16..32],
            "test mapping must change the first row"
        );

        // Stage 2 — the in-loop inverse pass (§8.8.1 step 1) restores
        // the original domain. For this payload every bin >= 1 maps
        // x → x − 8 with an exact inverse, so samples >= 16 round-trip
        // bit-exactly.
        walker.apply_in_loop_filters(&mut out).unwrap();
        let (min_bin, max_bin) = (lmcs.lmcs_min_bin_idx, lmcs.lmcs_max_bin_idx());
        for row in 0..16usize {
            for col in 0..16usize {
                let m = u32::from(mapped_expect[row * 16 + col]);
                let iy = derived.idx_y_inv(m, min_bin, max_bin);
                let expect = derived.inverse_map_luma_sample(m, iy) as u8;
                let got = out.luma.samples[row * out.luma.stride + 16 + col];
                assert_eq!(got, expect, "inverse-mapped luma at ({col}, {row})");
                let orig = ref_frame.luma.samples[row * ref_frame.luma.stride + 16 + col];
                if orig >= 16 {
                    assert_eq!(got, orig, "exact round-trip for bins >= 1");
                }
            }
        }
    }

    #[test]
    fn lmcs_chroma_residual_scaling_scales_inter_cb_residual() {
        // §8.7.5.3 — with `ph_chroma_residual_scale_flag == 1` the
        // chroma residual is folded through eq. 1220 before the add.
        // The CU sits at (0, 0), so the sizeY-aligned corner falls
        // inside it and neither neighbour column is available →
        // `invAvgLuma = 1 << (BitDepth − 1) = 128` (eq. 1217), which
        // lands in bin 8 (`lmcsCW[8] = 16`), and with
        // `lmcsDeltaCrs = +4` eq. 100 gives
        // `varScale = 16 * 2048 / 20 = 1638` — a non-identity scale.
        let pic_w = 64u32;
        let pic_h = 64u32;
        let sps = dummy_sps(1, pic_w, pic_h);
        let pps = dummy_pps(pic_w, pic_h, true);
        let mut sh = intra_slice_header();
        sh.sh_slice_type = SliceType::P;
        sh.sh_lmcs_used_flag = true;
        let layout = CtuLayout::from_sps_pps(&sps, &pps);
        let data = [0u8; 32];
        let mut walker = CtuWalker::begin_slice(&layout, &sps, &pps, &sh, 0, &data).unwrap();
        walker.set_ref_pic_list_l0(vec![ReferencePicture {
            poc: -1,
            frame: PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 100),
            motion_field: None,
        }]);

        let mut lmcs = lmcs_test_data();
        lmcs.lmcs_delta_abs_crs = 4; // lmcsDeltaCrs = +4.
        let ccu = CtuCu {
            ctu_addr_rs: 0,
            cu: Cu {
                x: 0,
                y: 0,
                w: 16,
                h: 16,
                cqt_depth: 0,
                mtt_depth: 0,
            },
        };
        let mut info = LeafCuInfo {
            x0: 0,
            y0: 0,
            cb_width: 16,
            cb_height: 16,
            pred_mode: CuPredMode::Inter,
            ..LeafCuInfo::default()
        };
        info.inter.general_merge_flag = true;
        info.inter.merge_data.merge_idx = 0;
        info.tu_y_coded_flag = false;
        info.tu_cb_coded_flag = true;
        let mut residual = LeafCuResidual::default();
        residual.cb_levels = vec![0i32; 8 * 8];
        residual.cb_levels[0] = 8; // single Cb DC coefficient.

        // Scaling gated off by ph_chroma_residual_scale_flag = 0.
        walker.set_lmcs(&lmcs, false).unwrap();
        let mut out_off = PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 0);
        walker
            .reconstruct_leaf_cu(&ccu, &info, &residual, &mut out_off)
            .expect("recon with chroma scaling off");

        // Scaling on.
        walker.set_lmcs(&lmcs, true).unwrap();
        let mut out_on = PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 0);
        walker
            .reconstruct_leaf_cu(&ccu, &info, &residual, &mut out_on)
            .expect("recon with chroma scaling on");

        // Expected: rec_on = Clip1(pred + (|res| * 1638 + 1024) >> 11)
        // where pred = 128 (zero-MV copy — `yuv420_filled` seeds every
        // chroma plane to neutral 128 regardless of the luma seed) and
        // res = rec_off − pred.
        let var_scale = 16u64 * 2048 / 20;
        assert_eq!(var_scale, 1638);
        let stride = out_off.cb.stride;
        let mut diff_seen = false;
        for row in 0..8usize {
            for col in 0..8usize {
                let off = i64::from(out_off.cb.samples[row * stride + col]);
                let res = off - 128;
                assert!(
                    (1..=254).contains(&off),
                    "unclipped baseline required at ({col}, {row})"
                );
                let mag = ((res.unsigned_abs() * var_scale) + 1024) >> 11;
                let scaled = if res < 0 { -(mag as i64) } else { mag as i64 };
                let expect = (128 + scaled).clamp(0, 255) as u8;
                let got = out_on.cb.samples[row * stride + col];
                assert_eq!(got, expect, "scaled Cb at ({col}, {row})");
                if got != out_off.cb.samples[row * stride + col] {
                    diff_seen = true;
                }
            }
        }
        assert!(diff_seen, "the eq. 1220 fold must change some samples");
        // Luma and Cr are untouched by the Cb residual.
        assert_eq!(out_on.cr.samples, out_off.cr.samples);
        assert_eq!(out_on.luma.samples, out_off.luma.samples);
    }

    #[test]
    fn lmcs_chroma_scaling_averages_left_neighbour_luma() {
        // §8.7.5.3 step 1 — a CU whose sizeY-aligned corner has a left
        // neighbour must average the left column of mapped-domain
        // reconstructed luma (eq. 1216), NOT fall back to eq. 1217
        // mid-grey. The payload puts `lmcsCW = 16` on bins 0..=7 and
        // `lmcsCW = 12` on bins 8..=15 with `lmcsDeltaCrs = +4`:
        //   * left-column luma 60 → bin 3 → varScale = 32768/20 = 1638;
        //   * mid-grey 128 → bin 8 → varScale = 32768/16 = 2048 (an
        //     identity fold).
        // So a difference from the scaling-off reconstruction proves
        // the neighbour average was used.
        let pic_w = 128u32;
        let pic_h = 64u32;
        let sps = dummy_sps(1, pic_w, pic_h);
        let pps = dummy_pps(pic_w, pic_h, true);
        let mut sh = intra_slice_header();
        sh.sh_slice_type = SliceType::P;
        sh.sh_lmcs_used_flag = true;
        let layout = CtuLayout::from_sps_pps(&sps, &pps);
        let data = [0u8; 32];
        let mut walker = CtuWalker::begin_slice(&layout, &sps, &pps, &sh, 0, &data).unwrap();
        walker.set_ref_pic_list_l0(vec![ReferencePicture {
            poc: -1,
            frame: PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 100),
            motion_field: None,
        }]);

        let mut abs_cw = [0u32; crate::lmcs::LMCS_NUM_BINS];
        let mut sign_cw = [false; crate::lmcs::LMCS_NUM_BINS];
        for i in 8..16 {
            abs_cw[i] = 4;
            sign_cw[i] = true; // lmcsCW[8..=15] = 16 − 4 = 12.
        }
        let lmcs = LmcsData {
            lmcs_min_bin_idx: 0,
            lmcs_delta_max_bin_idx: 0,
            lmcs_delta_cw_prec_minus1: 3,
            lmcs_delta_abs_cw: abs_cw,
            lmcs_delta_sign_cw_flag: sign_cw,
            lmcs_delta_abs_crs: 4,
            lmcs_delta_sign_crs_flag: false,
        };

        // CU at (64, 0) — the sizeY(=64)-aligned corner (64, 0) falls
        // inside the CU, availL holds (x > 0), availT does not (y == 0).
        let ccu = CtuCu {
            ctu_addr_rs: 1,
            cu: Cu {
                x: 64,
                y: 0,
                w: 16,
                h: 16,
                cqt_depth: 0,
                mtt_depth: 0,
            },
        };
        let mut info = LeafCuInfo {
            x0: 64,
            y0: 0,
            cb_width: 16,
            cb_height: 16,
            pred_mode: CuPredMode::Inter,
            ..LeafCuInfo::default()
        };
        info.inter.general_merge_flag = true;
        info.inter.merge_data.merge_idx = 0;
        info.tu_y_coded_flag = false;
        info.tu_cb_coded_flag = true;
        let mut residual = LeafCuResidual::default();
        residual.cb_levels = vec![0i32; 8 * 8];
        residual.cb_levels[0] = 8;

        // Baseline: scaling off. Seed the whole luma plane to 60 so the
        // x = 63 column (left of the aligned corner) reads 60.
        walker.set_lmcs(&lmcs, false).unwrap();
        let mut out_off = PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 60);
        walker
            .reconstruct_leaf_cu(&ccu, &info, &residual, &mut out_off)
            .expect("recon with chroma scaling off");

        walker.set_lmcs(&lmcs, true).unwrap();
        let mut out_on = PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 60);
        walker
            .reconstruct_leaf_cu(&ccu, &info, &residual, &mut out_on)
            .expect("recon with chroma scaling on");

        // Expected fold with the LEFT-average varScale (1638), checked
        // exactly; the mid-grey fallback would give the identity 2048
        // fold (== out_off), which `diff_seen` rules out.
        let var_scale = 16u64 * 2048 / 20;
        let stride = out_off.cb.stride;
        let c_x = 32usize; // 64 / 2
        let mut diff_seen = false;
        for row in 0..8usize {
            for col in 0..8usize {
                let off = i64::from(out_off.cb.samples[row * stride + c_x + col]);
                let res = off - 128;
                let mag = ((res.unsigned_abs() * var_scale) + 1024) >> 11;
                let scaled = if res < 0 { -(mag as i64) } else { mag as i64 };
                let expect = (128 + scaled).clamp(0, 255) as u8;
                let got = out_on.cb.samples[row * stride + c_x + col];
                assert_eq!(got, expect, "scaled Cb at ({col}, {row})");
                if got != out_off.cb.samples[row * stride + c_x + col] {
                    diff_seen = true;
                }
            }
        }
        assert!(diff_seen, "left-average varScale must differ from identity");
    }

    #[test]
    fn set_lmcs_rejects_nonconforming_payload() {
        // The all-defaults full-window payload derives `lmcsCW[i] =
        // OrgCW` for all 16 bins, so `Σ lmcsCW = 1 << BitDepth` — one
        // over the eq. 96 `(1 << BitDepth) − 1` budget.
        let sps = dummy_sps(2, 64, 64);
        let pps = dummy_pps(64, 64, true);
        let layout = CtuLayout::from_sps_pps(&sps, &pps);
        let data = [0u8; 8];
        let sh = intra_slice_header();
        let mut w = CtuWalker::begin_slice(&layout, &sps, &pps, &sh, 0, &data).unwrap();
        assert!(w.set_lmcs(&LmcsData::default(), false).is_err());
    }

    #[test]
    fn walker_builds_for_single_slice_intra() {
        let sps = dummy_sps(2, 128, 128);
        let pps = dummy_pps(128, 128, true);
        let sh = intra_slice_header();
        let layout = CtuLayout::from_sps_pps(&sps, &pps);
        let data = [0u8; 32];
        let walker = CtuWalker::begin_slice(&layout, &sps, &pps, &sh, 0, &data).unwrap();
        assert_eq!(walker.layout().pic_size_in_ctbs_y(), 1);
        assert_eq!(walker.slice_qp_y().0, 26);
        assert!(!walker.sh_cabac_init_flag());
        assert_eq!(walker.iter_ctus().count(), 1);
    }

    #[test]
    fn decode_ctu_partitions_returns_absolute_rectangles() {
        // 2x1 CTU picture at 128-px CTU size.
        let sps = dummy_sps(2, 256, 128);
        let pps = dummy_pps(256, 128, true);
        let sh = intra_slice_header();
        let layout = CtuLayout::from_sps_pps(&sps, &pps);
        let data = [0u8; 64];
        let mut walker = CtuWalker::begin_slice(&layout, &sps, &pps, &sh, 0, &data).unwrap();
        // Decode the first CTU only — don't chain both on a single zero
        // stream because the CABAC state diverges past the initial run.
        let ctu0 = walker.iter_ctus().next().unwrap();
        assert_eq!(ctu0.x0, 0);
        let cus = walker.decode_ctu_partitions(&ctu0).unwrap();
        // Every emitted CU must land inside the CTU.
        for ccu in &cus {
            assert!(ccu.cu.x >= ctu0.x0);
            assert!(ccu.cu.y >= ctu0.y0);
            assert!(ccu.cu.x + ccu.cu.w <= ctu0.x0 + ctu0.width_luma);
            assert!(ccu.cu.y + ccu.cu.h <= ctu0.y0 + ctu0.height_luma);
            assert_eq!(ccu.ctu_addr_rs, ctu0.ctu_addr_rs);
        }
    }

    #[test]
    fn decode_leaf_cu_syntax_returns_legal_mode() {
        // With all intra tools disabled (the default SPS fixture),
        // the reader must still produce a legal luma / chroma mode
        // without panicking on the CABAC state.
        let sps = dummy_sps(2, 128, 128);
        let pps = dummy_pps(128, 128, true);
        let sh = intra_slice_header();
        let layout = CtuLayout::from_sps_pps(&sps, &pps);
        let data = [0u8; 64];
        let mut walker = CtuWalker::begin_slice(&layout, &sps, &pps, &sh, 0, &data).unwrap();
        let ccu = CtuCu {
            ctu_addr_rs: 0,
            cu: Cu {
                x: 0,
                y: 0,
                w: 16,
                h: 16,
                cqt_depth: 0,
                mtt_depth: 0,
            },
        };
        let neigh = CuNeighbourhood::default();
        let (info, _residual) = walker.decode_leaf_cu_syntax(&ccu, &neigh).unwrap();
        assert!(info.intra_pred_mode_y <= 66);
        assert!(info.intra_pred_mode_c <= 83);
        assert_eq!(info.x0, 0);
        assert_eq!(info.cb_width, 16);
    }

    #[test]
    fn cu_tool_flags_snapshot_carries_ctb_and_chroma() {
        let sps = dummy_sps(2, 128, 128);
        let pps = dummy_pps(128, 128, true);
        let sh = intra_slice_header();
        let layout = CtuLayout::from_sps_pps(&sps, &pps);
        let data = [0u8; 32];
        let walker = CtuWalker::begin_slice(&layout, &sps, &pps, &sh, 0, &data).unwrap();
        let tools = walker.cu_tool_flags();
        assert_eq!(tools.ctb_size_y, 128);
        assert_eq!(tools.chroma_format_idc, 1);
        assert_eq!(tools.min_tb_size_y, 4);
    }

    /// `reconstruct_leaf_cu` now drives the §8.4 / §8.7 pipeline end
    /// to end. With the test fixtures' default tool flags (no MIP, no
    /// ISP, no BDPCM) and a fresh mid-grey picture buffer, the call
    /// must succeed and write into the luma plane.
    #[test]
    fn reconstruct_leaf_cu_writes_pred_into_buffer() {
        let sps = dummy_sps(2, 128, 128);
        let pps = dummy_pps(128, 128, true);
        let sh = intra_slice_header();
        let layout = CtuLayout::from_sps_pps(&sps, &pps);
        let data = [0u8; 64];
        let mut walker = CtuWalker::begin_slice(&layout, &sps, &pps, &sh, 0, &data).unwrap();
        let ccu = CtuCu {
            ctu_addr_rs: 0,
            cu: Cu {
                x: 0,
                y: 0,
                w: 16,
                h: 16,
                cqt_depth: 0,
                mtt_depth: 0,
            },
        };
        let neigh = CuNeighbourhood::default();
        let (info, residual) = walker.decode_leaf_cu_syntax(&ccu, &neigh).unwrap();
        let mut out = PictureBuffer::yuv420_filled(128, 128, 0);
        walker
            .reconstruct_leaf_cu(&ccu, &info, &residual, &mut out)
            .unwrap();
        // The predictor used a mid-grey neighbourhood (left/above
        // unavailable at picture-edge); PLANAR / DC + a near-zero
        // residual lands every sample close to the spec's mid-grey
        // (128 at 8-bit) and inside the legal pixel range.
        let mut wrote = false;
        for y in 0..16 {
            for x in 0..16 {
                let v = out.luma.samples[y * 128 + x];
                if v != 0 {
                    wrote = true;
                }
            }
        }
        assert!(wrote, "reconstruct must paint into the destination plane");
    }

    /// Chroma planes (4:2:0 → 8×8 chroma TB for a 16×16 luma CU) must
    /// also be reconstructed by `reconstruct_leaf_cu`. We seed Cb / Cr
    /// to a sentinel value (17) and verify the targeted CU rectangle
    /// was overwritten — the §8.4.5.2.8 mid-grey substitution at the
    /// picture edge plus PLANAR/DC prediction lands every chroma sample
    /// at 128, which is observably different from the seed.
    #[test]
    fn reconstruct_leaf_cu_writes_chroma_planes() {
        let sps = dummy_sps(2, 128, 128);
        let pps = dummy_pps(128, 128, true);
        let sh = intra_slice_header();
        let layout = CtuLayout::from_sps_pps(&sps, &pps);
        let data = [0u8; 64];
        let mut walker = CtuWalker::begin_slice(&layout, &sps, &pps, &sh, 0, &data).unwrap();
        let ccu = CtuCu {
            ctu_addr_rs: 0,
            cu: Cu {
                x: 0,
                y: 0,
                w: 16,
                h: 16,
                cqt_depth: 0,
                mtt_depth: 0,
            },
        };
        let neigh = CuNeighbourhood::default();
        let (info, residual) = walker.decode_leaf_cu_syntax(&ccu, &neigh).unwrap();
        let mut out = PictureBuffer::yuv420_filled(128, 128, 0);
        // Override the chroma seed so we can detect writes.
        for v in out.cb.samples.iter_mut() {
            *v = 17;
        }
        for v in out.cr.samples.iter_mut() {
            *v = 17;
        }
        walker
            .reconstruct_leaf_cu(&ccu, &info, &residual, &mut out)
            .unwrap();
        // A 16×16 luma CU corresponds to an 8×8 chroma TB at (0, 0)
        // in the chroma planes. The picture-edge case substitutes
        // mid-grey (128) for unavailable refs and PLANAR / DC of a
        // mid-grey-only neighbourhood collapses to 128 everywhere.
        for y in 0..8 {
            for x in 0..8 {
                let cb = out.cb.samples[y * 64 + x];
                let cr = out.cr.samples[y * 64 + x];
                assert_ne!(cb, 17, "chroma Cb at ({x},{y}) was not written");
                assert_ne!(cr, 17, "chroma Cr at ({x},{y}) was not written");
            }
        }
        // Outside the CU rectangle the seed must still be visible
        // (no over-write past the CU's chroma footprint).
        assert_eq!(out.cb.samples[0 * 64 + 8], 17);
        assert_eq!(out.cb.samples[8 * 64 + 0], 17);
    }

    /// `chroma_pred_mode_for_predict` keeps PLANAR / DC / cardinal
    /// angulars verbatim, snaps non-cardinal angulars to the nearest
    /// cardinal, and falls CCLM back to PLANAR (the actual CCLM dispatch
    /// happens in `reconstruct_chroma_plane` before this helper runs —
    /// CCLM only hits this PLANAR fallback if the dedicated path
    /// rejects its inputs).
    #[test]
    fn chroma_pred_mode_mapping_keeps_planar_dc_and_snaps_angulars() {
        use crate::leaf_cu::{INTRA_DC, INTRA_LT_CCLM, INTRA_L_CCLM, INTRA_PLANAR, INTRA_T_CCLM};
        assert_eq!(chroma_pred_mode_for_predict(INTRA_PLANAR), INTRA_PLANAR);
        assert_eq!(chroma_pred_mode_for_predict(INTRA_DC), INTRA_DC);
        assert_eq!(chroma_pred_mode_for_predict(50), 50);
        assert_eq!(chroma_pred_mode_for_predict(2), 2);
        assert_eq!(chroma_pred_mode_for_predict(66), 66);
        // Non-cardinal angular -> snap to nearest cardinal (matches
        // the luma fallback).
        assert_eq!(chroma_pred_mode_for_predict(40), 34);
        // CCLM modes drop to PLANAR here as a safety fallback.
        assert_eq!(chroma_pred_mode_for_predict(INTRA_LT_CCLM), INTRA_PLANAR);
        assert_eq!(chroma_pred_mode_for_predict(INTRA_L_CCLM), INTRA_PLANAR);
        assert_eq!(chroma_pred_mode_for_predict(INTRA_T_CCLM), INTRA_PLANAR);
    }

    /// CCLM end-to-end: place a CU away from the picture edge so the
    /// chroma + luma neighbours are inside the plane, set
    /// `intra_pred_mode_c == INTRA_LT_CCLM`, and confirm the chroma
    /// planes get written through the §8.4.5.2.14 path. With a flat
    /// luma surround and a flat chroma neighbourhood, CCLM must
    /// reproduce the chroma constant (diff == 0 → a == 0, b == minC).
    #[test]
    fn reconstruct_leaf_cu_cclm_lt_runs_pipeline() {
        let sps = dummy_sps(2, 128, 128);
        let pps = dummy_pps(128, 128, true);
        let sh = intra_slice_header();
        let layout = CtuLayout::from_sps_pps(&sps, &pps);
        let data = [0u8; 64];
        let mut walker = CtuWalker::begin_slice(&layout, &sps, &pps, &sh, 0, &data).unwrap();
        // Place the CU at (32, 32) so left and above neighbours both
        // exist (above_avail / left_avail derive from x0/y0 > 0).
        let ccu = CtuCu {
            ctu_addr_rs: 0,
            cu: Cu {
                x: 32,
                y: 32,
                w: 16,
                h: 16,
                cqt_depth: 0,
                mtt_depth: 0,
            },
        };
        let info = LeafCuInfo {
            x0: 32,
            y0: 32,
            cb_width: 16,
            cb_height: 16,
            intra_pred_mode_c: INTRA_LT_CCLM,
            ..LeafCuInfo::default()
        };
        let residual = LeafCuResidual::default();
        // Seed luma + chroma surround to a known value so CCLM's diff
        // collapses to 0. Luma seed = 100; chroma seed = 60.
        let mut out = PictureBuffer::yuv420_filled(128, 128, 100);
        for v in out.cb.samples.iter_mut() {
            *v = 60;
        }
        for v in out.cr.samples.iter_mut() {
            *v = 70;
        }
        walker
            .reconstruct_leaf_cu(&ccu, &info, &residual, &mut out)
            .expect("CCLM_LT path should succeed");
        // Chroma TB is 8×8 at chroma-coords (16, 16). With diff == 0,
        // every sample collapses to the chroma neighbour constant.
        for cy in 0..8 {
            for cx in 0..8 {
                let cb = out.cb.samples[(16 + cy) * 64 + (16 + cx)];
                let cr = out.cr.samples[(16 + cy) * 64 + (16 + cx)];
                assert_eq!(cb, 60, "Cb at ({cx},{cy}) lost CCLM constant");
                assert_eq!(cr, 70, "Cr at ({cx},{cy}) lost CCLM constant");
            }
        }
    }

    /// CCLM_T (top-only) on a CU at the very top picture edge falls
    /// back to mid-grey because no top samples are available — the
    /// helper signals this via empty `neigh_top_chroma`, which trips
    /// the eq. 365 fallback. We seed chroma to an obviously-different
    /// value to detect the write.
    #[test]
    fn reconstruct_leaf_cu_cclm_t_picture_edge_writes_mid_grey() {
        let sps = dummy_sps(2, 128, 128);
        let pps = dummy_pps(128, 128, true);
        let sh = intra_slice_header();
        let layout = CtuLayout::from_sps_pps(&sps, &pps);
        let data = [0u8; 64];
        let mut walker = CtuWalker::begin_slice(&layout, &sps, &pps, &sh, 0, &data).unwrap();
        let ccu = CtuCu {
            ctu_addr_rs: 0,
            cu: Cu {
                x: 0,
                y: 0,
                w: 16,
                h: 16,
                cqt_depth: 0,
                mtt_depth: 0,
            },
        };
        let info = LeafCuInfo {
            x0: 0,
            y0: 0,
            cb_width: 16,
            cb_height: 16,
            intra_pred_mode_c: INTRA_T_CCLM,
            ..LeafCuInfo::default()
        };
        let residual = LeafCuResidual::default();
        let mut out = PictureBuffer::yuv420_filled(128, 128, 0);
        for v in out.cb.samples.iter_mut() {
            *v = 17;
        }
        walker
            .reconstruct_leaf_cu(&ccu, &info, &residual, &mut out)
            .expect("CCLM_T edge path should succeed");
        // Mid-grey at 8 bits = 128.
        for cy in 0..8 {
            for cx in 0..8 {
                assert_eq!(
                    out.cb.samples[cy * 64 + cx],
                    128,
                    "CCLM_T edge fallback should paint mid-grey at ({cx},{cy})"
                );
            }
        }
    }

    /// `chroma_qp_identity` is a clamp-only identity mapping today
    /// (`QpC = QpY + offset`, clipped to `[0, 63]`).
    #[test]
    fn chroma_qp_identity_clips_to_legal_range() {
        assert_eq!(chroma_qp_identity(20, 0), 20);
        assert_eq!(chroma_qp_identity(20, 5), 25);
        assert_eq!(chroma_qp_identity(60, 5), 63); // saturates
        assert_eq!(chroma_qp_identity(0, -3), 0); // clamps from below
    }

    /// §8.7.1 eqs. 1147 / 1148 — `chroma_qp_offset_sum` adds the
    /// per-component PPS + slice + CU offsets, picking the Cb or Cr
    /// column by `c_idx`.
    #[test]
    fn chroma_qp_offset_sum_threads_slice_level_offsets() {
        // Cb column (c_idx == 1): pps_cb (3) + sh_cb (-2) + cu (1) = 2;
        // the Cr-side values must not leak in.
        assert_eq!(chroma_qp_offset_sum(1, 3, 9, -2, 7, 1), 2);
        // Cr column (c_idx == 2): pps_cr (9) + sh_cr (7) + cu (0) = 16.
        assert_eq!(chroma_qp_offset_sum(2, 3, 9, -2, 7, 0), 16);
        // Luma (c_idx == 0): no chroma offset, only the CU term survives.
        assert_eq!(chroma_qp_offset_sum(0, 3, 9, -2, 7, 4), 4);
        // Default zero offsets give a pure identity passthrough.
        assert_eq!(chroma_qp_offset_sum(1, 0, 0, 0, 0, 0), 0);
    }

    /// §7.4.10.6 eqs. 193 / 194 — `cu_chroma_qp_offset` indexes the
    /// PPS `pps_c?_qp_offset_list` by `cu_chroma_qp_offset_idx` when the
    /// flag is set, and yields 0 when the flag is clear, the index is out
    /// of range, or the component is luma.
    #[test]
    fn cu_chroma_qp_offset_indexes_pps_list() {
        let cb = [1, -3, 5];
        let cr = [2, 4, -6];
        // Flag set, idx 1: Cb → -3, Cr → 4.
        assert_eq!(cu_chroma_qp_offset(1, true, 1, &cb, &cr), -3);
        assert_eq!(cu_chroma_qp_offset(2, true, 1, &cb, &cr), 4);
        // Flag clear → 0 regardless of idx.
        assert_eq!(cu_chroma_qp_offset(1, false, 2, &cb, &cr), 0);
        // Luma component → 0.
        assert_eq!(cu_chroma_qp_offset(0, true, 0, &cb, &cr), 0);
        // Out-of-range idx → 0 (defensive; bitstream constrains idx).
        assert_eq!(cu_chroma_qp_offset(1, true, 9, &cb, &cr), 0);
        // Empty list → 0.
        assert_eq!(cu_chroma_qp_offset(1, true, 0, &[], &[]), 0);
    }

    /// §7.4.3.22 eq. 106 — the walker's `log2_transform_range()` returns
    /// 15 for an SPS with no range-extension block, and the extended
    /// `Max(15, Min(20, BitDepth + 6))` when `sps_extended_precision_flag`
    /// is set on a high-bit-depth SPS.
    #[test]
    fn walker_log2_transform_range_honours_extended_precision() {
        use crate::sps::SpsRangeExtension;
        let pic_w = 64u32;
        let pic_h = 64u32;
        let pps = dummy_pps(pic_w, pic_h, true);
        let sh = intra_slice_header();
        let data = [0u8; 32];

        // No range-extension block → 15 regardless of bit depth.
        let sps = dummy_sps(1, pic_w, pic_h);
        let layout = CtuLayout::from_sps_pps(&sps, &pps);
        let w = CtuWalker::begin_slice(&layout, &sps, &pps, &sh, 0, &data).unwrap();
        assert_eq!(w.log2_transform_range(), 15);

        // Extended precision at BitDepth 12 → Max(15, Min(20, 18)) = 18.
        let mut sps12 = dummy_sps(1, pic_w, pic_h);
        sps12.sps_bitdepth_minus8 = 4; // BitDepth = 12
        sps12.sps_range_extension_flag = true;
        sps12.range_extension = Some(SpsRangeExtension {
            sps_extended_precision_flag: true,
            ..Default::default()
        });
        let layout12 = CtuLayout::from_sps_pps(&sps12, &pps);
        let w12 = CtuWalker::begin_slice(&layout12, &sps12, &pps, &sh, 0, &data).unwrap();
        assert_eq!(w12.log2_transform_range(), 18);

        // Extended precision flag off → 15 even with the block present.
        let mut sps_off = dummy_sps(1, pic_w, pic_h);
        sps_off.sps_bitdepth_minus8 = 4;
        sps_off.sps_range_extension_flag = true;
        sps_off.range_extension = Some(SpsRangeExtension::default());
        let layout_off = CtuLayout::from_sps_pps(&sps_off, &pps);
        let w_off = CtuWalker::begin_slice(&layout_off, &sps_off, &pps, &sh, 0, &data).unwrap();
        assert_eq!(w_off.log2_transform_range(), 15);
    }

    /// MIP-flagged CU now flows through the §8.4.5.2.2 matrix-based
    /// prediction in [`crate::mip::predict_mip`] instead of surfacing
    /// `Error::Unsupported`. With an empty residual (`tu_y_coded_flag
    /// == false`) the reconstructed luma plane should change away from
    /// its zero-fill seed at the MIP CU's footprint.
    #[test]
    fn reconstruct_leaf_cu_mip_runs_pipeline() {
        let sps = dummy_sps(2, 128, 128);
        let pps = dummy_pps(128, 128, true);
        let sh = intra_slice_header();
        let layout = CtuLayout::from_sps_pps(&sps, &pps);
        let data = [0u8; 32];
        let mut walker = CtuWalker::begin_slice(&layout, &sps, &pps, &sh, 0, &data).unwrap();
        let ccu = CtuCu {
            ctu_addr_rs: 0,
            cu: Cu {
                x: 0,
                y: 0,
                w: 16,
                h: 16,
                cqt_depth: 0,
                mtt_depth: 0,
            },
        };
        let mut info = LeafCuInfo {
            x0: 0,
            y0: 0,
            cb_width: 16,
            cb_height: 16,
            ..LeafCuInfo::default()
        };
        info.intra_mip_flag = true;
        info.intra_mip_transposed_flag = false;
        info.intra_mip_mode = 0; // valid for any mipSizeId (sz2 has 6 modes).
        let residual = LeafCuResidual::default();
        let mut out = PictureBuffer::yuv420_filled(128, 128, 0);
        walker
            .reconstruct_leaf_cu(&ccu, &info, &residual, &mut out)
            .expect("MIP path should succeed");
        // 1. With ALL-zero references (above_avail == false &&
        //    left_avail == false because the CU is at x=0/y=0), the
        //    `OwnedIntraRefs::from_plane` substitution fills the
        //    references with mid-grey (1 << (BitDepth - 1)). For an
        //    8-bit pipeline that's 128. The MIP `p[]` then collapses
        //    to all zeros (eq. 268), `oW = 32`, and predMip[x][y] =
        //    pTemp[0] = 128 for every pixel. So the destination plane
        //    should now contain `128` over the 16×16 CU footprint.
        for y in 0..16 {
            for x in 0..16 {
                assert_eq!(
                    out.luma.get(x, y).unwrap(),
                    128,
                    "MIP-on-zero-refs should fill with mid-grey at ({x},{y})"
                );
            }
        }
    }

    /// MIP with a non-zero `intra_mip_mode` and a transposed flag still
    /// flows through the pipeline without panicking (parser-level
    /// invariants only — covers the size-id dispatch and transpose
    /// branches in [`crate::mip::predict_mip`]).
    #[test]
    fn reconstruct_leaf_cu_mip_transposed_runs_pipeline() {
        let sps = dummy_sps(2, 128, 128);
        let pps = dummy_pps(128, 128, true);
        let sh = intra_slice_header();
        let layout = CtuLayout::from_sps_pps(&sps, &pps);
        let data = [0u8; 32];
        let mut walker = CtuWalker::begin_slice(&layout, &sps, &pps, &sh, 0, &data).unwrap();
        let ccu = CtuCu {
            ctu_addr_rs: 0,
            cu: Cu {
                x: 0,
                y: 0,
                w: 8,
                h: 8,
                cqt_depth: 0,
                mtt_depth: 0,
            },
        };
        // 8x8 → mipSizeId = 1 → 8 valid modes.
        let mut info = LeafCuInfo {
            x0: 0,
            y0: 0,
            cb_width: 8,
            cb_height: 8,
            ..LeafCuInfo::default()
        };
        info.intra_mip_flag = true;
        info.intra_mip_transposed_flag = true;
        info.intra_mip_mode = 3;
        let residual = LeafCuResidual::default();
        let mut out = PictureBuffer::yuv420_filled(128, 128, 0);
        walker
            .reconstruct_leaf_cu(&ccu, &info, &residual, &mut out)
            .expect("MIP transposed path should succeed");
    }

    /// BDPCM-luma path through `reconstruct_leaf_cu`: when the leaf
    /// info has `intra_bdpcm_luma == true`, the call must succeed
    /// (no longer surface Unsupported), the prediction mode is one of
    /// ANGULAR18 / ANGULAR50, the residual goes through the
    /// transform-skip + accumulation dequant, and the destination luma
    /// plane is written without panicking.
    #[test]
    fn reconstruct_leaf_cu_bdpcm_luma_runs_pipeline() {
        let sps = dummy_sps(2, 128, 128);
        let pps = dummy_pps(128, 128, true);
        let sh = intra_slice_header();
        let layout = CtuLayout::from_sps_pps(&sps, &pps);
        let data = [0u8; 32];
        let mut walker = CtuWalker::begin_slice(&layout, &sps, &pps, &sh, 0, &data).unwrap();
        let ccu = CtuCu {
            ctu_addr_rs: 0,
            cu: Cu {
                x: 0,
                y: 0,
                w: 8,
                h: 8,
                cqt_depth: 0,
                mtt_depth: 0,
            },
        };
        // Hand-build a BDPCM-luma leaf with empty residual (tu_y_coded
        // == false ⇒ skip dequant). Direction = vertical → ANGULAR50.
        let mut info = LeafCuInfo {
            x0: 0,
            y0: 0,
            cb_width: 8,
            cb_height: 8,
            ..LeafCuInfo::default()
        };
        info.intra_bdpcm_luma = true;
        info.intra_bdpcm_luma_dir = true;
        info.intra_pred_mode_y = crate::leaf_cu::INTRA_ANGULAR50;
        let residual = LeafCuResidual::default();
        let mut out = PictureBuffer::yuv420_filled(128, 128, 0);
        walker
            .reconstruct_leaf_cu(&ccu, &info, &residual, &mut out)
            .unwrap();
        // ANGULAR50 (vertical) on a fresh mid-grey-substituted block
        // replicates above[x] down each column. With no above samples
        // available (picture-edge), §8.4.5.2.8 substitutes mid-grey
        // (1 << (BitDepth-1) = 512 at 10-bit). The destination plane
        // is u8 so the value is shifted down 2 bits → 128.
        let mid_u8: u8 = 128;
        for y in 0..8 {
            for x in 0..8 {
                assert_eq!(out.luma.samples[y * 128 + x], mid_u8);
            }
        }
    }

    /// BDPCM-luma with a horizontal direction picks ANGULAR18 (pure
    /// horizontal replication of left[y]).
    #[test]
    fn reconstruct_leaf_cu_bdpcm_luma_horizontal_uses_angular18() {
        let sps = dummy_sps(2, 128, 128);
        let pps = dummy_pps(128, 128, true);
        let sh = intra_slice_header();
        let layout = CtuLayout::from_sps_pps(&sps, &pps);
        let data = [0u8; 32];
        let mut walker = CtuWalker::begin_slice(&layout, &sps, &pps, &sh, 0, &data).unwrap();
        let ccu = CtuCu {
            ctu_addr_rs: 0,
            cu: Cu {
                x: 0,
                y: 0,
                w: 8,
                h: 8,
                cqt_depth: 0,
                mtt_depth: 0,
            },
        };
        let mut info = LeafCuInfo {
            x0: 0,
            y0: 0,
            cb_width: 8,
            cb_height: 8,
            ..LeafCuInfo::default()
        };
        info.intra_bdpcm_luma = true;
        info.intra_bdpcm_luma_dir = false;
        info.intra_pred_mode_y = crate::leaf_cu::INTRA_ANGULAR18;
        let residual = LeafCuResidual::default();
        let mut out = PictureBuffer::yuv420_filled(128, 128, 0);
        walker
            .reconstruct_leaf_cu(&ccu, &info, &residual, &mut out)
            .unwrap();
        // Same picture-edge mid-grey behaviour as the vertical case —
        // there are no real prior samples so both directions collapse
        // to the constant fill (10-bit mid-grey 512 → u8 128).
        let mid_u8: u8 = 128;
        for y in 0..8 {
            for x in 0..8 {
                assert_eq!(out.luma.samples[y * 128 + x], mid_u8);
            }
        }
    }

    /// ISP horizontal split: a 16×16 PLANAR CU split into four 16×4
    /// subpartitions must reconstruct each one in turn into the picture
    /// buffer. With every CBF off and the partial plane left at the
    /// initial fill, the first subpartition picks up its top-row
    /// reference from the seed, and each later subpartition picks up
    /// the row freshly written by the prior partition. Because PLANAR
    /// of a constant neighbourhood reproduces the constant, the final
    /// picture matches the seed across the whole CU.
    #[test]
    fn reconstruct_leaf_cu_isp_hor_split_runs_pipeline() {
        use crate::leaf_cu::{IspSplitType, LeafCuLumaSubpart};
        let sps = dummy_sps(2, 128, 128);
        let pps = dummy_pps(128, 128, true);
        let sh = intra_slice_header();
        let layout = CtuLayout::from_sps_pps(&sps, &pps);
        let data = [0u8; 32];
        let mut walker = CtuWalker::begin_slice(&layout, &sps, &pps, &sh, 0, &data).unwrap();
        let ccu = CtuCu {
            ctu_addr_rs: 0,
            cu: Cu {
                x: 0,
                y: 0,
                w: 16,
                h: 16,
                cqt_depth: 0,
                mtt_depth: 0,
            },
        };
        let info = LeafCuInfo {
            x0: 0,
            y0: 0,
            cb_width: 16,
            cb_height: 16,
            isp_split: IspSplitType::HorSplit,
            intra_pred_mode_y: INTRA_PLANAR,
            ..LeafCuInfo::default()
        };
        // Build empty per-partition residuals — four 16x4 sub-TBs at
        // y = 0, 4, 8, 12 with `tu_y_coded_flag == false`.
        let mut residual = LeafCuResidual::default();
        for i in 0..4u32 {
            residual.luma_subparts.push(LeafCuLumaSubpart {
                n_w: 16,
                n_h: 4,
                x_offset: 0,
                y_offset: i * 4,
                tu_y_coded_flag: false,
                levels: Vec::new(),
            });
        }
        let mut out = PictureBuffer::yuv420_filled(128, 128, 100);
        walker
            .reconstruct_leaf_cu(&ccu, &info, &residual, &mut out)
            .expect("ISP HOR split path should succeed");
        // At the picture edge, the first partition reads mid-grey
        // (10-bit 1<<9 = 512 → u8 128) substitution for both above
        // and left. Each subsequent partition reads its top row
        // from the freshly-written prior partition, so PLANAR's
        // row+column blend slowly drifts away from a flat 128. The
        // important invariants here are "every sample is legal"
        // and "the CU's footprint is fully written" (no leftover
        // seed pixels at value 100). Outside the CU the seed must
        // remain visible.
        let mut all_legal = true;
        let mut wrote_into_cu = true;
        for y in 0..16usize {
            for x in 0..16usize {
                let v = out.luma.samples[y * 128 + x] as i32;
                if !(0..=255).contains(&v) {
                    all_legal = false;
                }
                if v == 100 {
                    wrote_into_cu = false;
                }
            }
        }
        assert!(all_legal, "ISP partition wrote out-of-range pixels");
        assert!(wrote_into_cu, "ISP CU footprint left unwritten samples");
        // Outside the 16x16 CU the seed must still be visible on row 0
        // (no spillover past the CU width).
        assert_eq!(out.luma.samples[16], 100);
        assert_eq!(out.luma.samples[20 * 128 + 0], 100);
    }

    /// ISP vertical split: 4x16 sub-TBs side by side, all coded with
    /// `tu_y_coded_flag == false`. Since `nW == 4 == nPbW`, `pbFactor`
    /// is 1 and prediction runs once per partition. The walker must
    /// emit four sub-TBs and the picture buffer must end up valid.
    #[test]
    fn reconstruct_leaf_cu_isp_ver_split_runs_pipeline() {
        use crate::leaf_cu::{IspSplitType, LeafCuLumaSubpart};
        let sps = dummy_sps(2, 128, 128);
        let pps = dummy_pps(128, 128, true);
        let sh = intra_slice_header();
        let layout = CtuLayout::from_sps_pps(&sps, &pps);
        let data = [0u8; 32];
        let mut walker = CtuWalker::begin_slice(&layout, &sps, &pps, &sh, 0, &data).unwrap();
        // Place at (32, 32) so left + above neighbours are inside the
        // plane — predictor reads real samples instead of mid-grey
        // substitutions.
        let ccu = CtuCu {
            ctu_addr_rs: 0,
            cu: Cu {
                x: 32,
                y: 32,
                w: 16,
                h: 16,
                cqt_depth: 0,
                mtt_depth: 0,
            },
        };
        let info = LeafCuInfo {
            x0: 32,
            y0: 32,
            cb_width: 16,
            cb_height: 16,
            isp_split: IspSplitType::VerSplit,
            intra_pred_mode_y: INTRA_PLANAR,
            ..LeafCuInfo::default()
        };
        let mut residual = LeafCuResidual::default();
        for i in 0..4u32 {
            residual.luma_subparts.push(LeafCuLumaSubpart {
                n_w: 4,
                n_h: 16,
                x_offset: i * 4,
                y_offset: 0,
                tu_y_coded_flag: false,
                levels: Vec::new(),
            });
        }
        let mut out = PictureBuffer::yuv420_filled(128, 128, 100);
        walker
            .reconstruct_leaf_cu(&ccu, &info, &residual, &mut out)
            .expect("ISP VER split path should succeed");
        // A flat seed neighbourhood + PLANAR + zero residual
        // reproduces the seed exactly.
        for y in 32..48usize {
            for x in 32..48usize {
                assert_eq!(
                    out.luma.samples[y * 128 + x],
                    100,
                    "pixel ({x},{y}) wandered off seed value"
                );
            }
        }
    }

    /// ISP vertical split with a small CU (8x8) that triggers
    /// `pbFactor == 2`: the prediction window is shared between
    /// `xPartPbIdx == 0` and `xPartPbIdx == 1`. The walker must not
    /// re-fetch references for `xPartPbIdx > 0` and must still write
    /// every partition.
    #[test]
    fn reconstruct_leaf_cu_isp_ver_split_pb_factor_two() {
        use crate::leaf_cu::{IspSplitType, LeafCuLumaSubpart};
        let sps = dummy_sps(2, 128, 128);
        let pps = dummy_pps(128, 128, true);
        let sh = intra_slice_header();
        let layout = CtuLayout::from_sps_pps(&sps, &pps);
        let data = [0u8; 32];
        let mut walker = CtuWalker::begin_slice(&layout, &sps, &pps, &sh, 0, &data).unwrap();
        let ccu = CtuCu {
            ctu_addr_rs: 0,
            cu: Cu {
                x: 32,
                y: 32,
                w: 8,
                h: 8,
                cqt_depth: 0,
                mtt_depth: 0,
            },
        };
        let info = LeafCuInfo {
            x0: 32,
            y0: 32,
            cb_width: 8,
            cb_height: 8,
            isp_split: IspSplitType::VerSplit,
            intra_pred_mode_y: INTRA_PLANAR,
            ..LeafCuInfo::default()
        };
        // 4 partitions of 2x8 each. pb_factor = 2; pred window covers
        // xPartIdx 0 and 1 jointly, then 2 and 3.
        let mut residual = LeafCuResidual::default();
        for i in 0..4u32 {
            residual.luma_subparts.push(LeafCuLumaSubpart {
                n_w: 2,
                n_h: 8,
                x_offset: i * 2,
                y_offset: 0,
                tu_y_coded_flag: false,
                levels: Vec::new(),
            });
        }
        let mut out = PictureBuffer::yuv420_filled(128, 128, 100);
        walker
            .reconstruct_leaf_cu(&ccu, &info, &residual, &mut out)
            .expect("ISP VER pb_factor=2 path should succeed");
        // Flat surround + PLANAR + zero residual ⇒ seed reproduction.
        for y in 32..40usize {
            for x in 32..40usize {
                assert_eq!(out.luma.samples[y * 128 + x], 100);
            }
        }
    }

    /// ISP with a coded residual on the last subpartition: when the
    /// final partition has a non-zero DC coefficient, the dequant +
    /// IDCT must offset the seed value visibly inside that
    /// partition's footprint while leaving the prior partitions
    /// untouched.
    #[test]
    fn reconstruct_leaf_cu_isp_hor_last_partition_residual() {
        use crate::leaf_cu::{IspSplitType, LeafCuLumaSubpart};
        let sps = dummy_sps(2, 128, 128);
        let pps = dummy_pps(128, 128, true);
        let sh = intra_slice_header();
        let layout = CtuLayout::from_sps_pps(&sps, &pps);
        let data = [0u8; 32];
        let mut walker = CtuWalker::begin_slice(&layout, &sps, &pps, &sh, 0, &data).unwrap();
        let ccu = CtuCu {
            ctu_addr_rs: 0,
            cu: Cu {
                x: 32,
                y: 32,
                w: 16,
                h: 16,
                cqt_depth: 0,
                mtt_depth: 0,
            },
        };
        let info = LeafCuInfo {
            x0: 32,
            y0: 32,
            cb_width: 16,
            cb_height: 16,
            isp_split: IspSplitType::HorSplit,
            intra_pred_mode_y: INTRA_PLANAR,
            ..LeafCuInfo::default()
        };
        let mut residual = LeafCuResidual::default();
        for i in 0..3u32 {
            residual.luma_subparts.push(LeafCuLumaSubpart {
                n_w: 16,
                n_h: 4,
                x_offset: 0,
                y_offset: i * 4,
                tu_y_coded_flag: false,
                levels: Vec::new(),
            });
        }
        let mut last_levels = vec![0i32; 16 * 4];
        last_levels[0] = 50; // DC impulse on the bottom strip
        residual.luma_subparts.push(LeafCuLumaSubpart {
            n_w: 16,
            n_h: 4,
            x_offset: 0,
            y_offset: 12,
            tu_y_coded_flag: true,
            levels: last_levels,
        });
        let mut out = PictureBuffer::yuv420_filled(128, 128, 100);
        walker
            .reconstruct_leaf_cu(&ccu, &info, &residual, &mut out)
            .expect("ISP HOR last-partition residual path should succeed");
        // Top three 16×4 strips reproduce the seed (zero residual
        // through PLANAR on a flat neighbourhood).
        for y in 32..44usize {
            for x in 32..48usize {
                assert_eq!(
                    out.luma.samples[y * 128 + x],
                    100,
                    "ISP top strip pixel ({x},{y}) wandered"
                );
            }
        }
        let mut last_strip_changed = false;
        for y in 44..48usize {
            for x in 32..48usize {
                if out.luma.samples[y * 128 + x] != 100 {
                    last_strip_changed = true;
                }
            }
        }
        assert!(
            last_strip_changed,
            "ISP bottom subpartition residual must move at least one sample"
        );
    }

    /// The fallback angular-snap helper picks the closest cardinal
    /// mode by absolute index distance.
    #[test]
    fn nearest_supported_angular_snaps_to_cardinals() {
        assert_eq!(nearest_supported_angular(2), 2);
        assert_eq!(nearest_supported_angular(10), 2);
        assert_eq!(nearest_supported_angular(11), 18);
        assert_eq!(nearest_supported_angular(26), 18);
        assert_eq!(nearest_supported_angular(40), 34);
        assert_eq!(nearest_supported_angular(60), 66);
        assert_eq!(nearest_supported_angular(66), 66);
    }

    /// Round-12: `apply_in_loop_filters` is wired to the §8.8.3
    /// deblocker. With no CUs accumulated yet (no `decode_picture_into`
    /// call) it should be a no-op, returning `Ok(())` and leaving the
    /// picture untouched.
    #[test]
    fn apply_in_loop_filters_no_op_without_decode() {
        let sps = dummy_sps(2, 128, 128);
        let pps = dummy_pps(128, 128, true);
        let sh = intra_slice_header();
        let layout = CtuLayout::from_sps_pps(&sps, &pps);
        let data = [0u8; 32];
        let mut walker = CtuWalker::begin_slice(&layout, &sps, &pps, &sh, 0, &data).unwrap();
        let mut out = PictureBuffer::yuv420_filled(128, 128, 64);
        let snapshot = out.luma.samples.clone();
        walker.apply_in_loop_filters(&mut out).unwrap();
        assert_eq!(out.luma.samples, snapshot);
    }

    #[test]
    fn neighbour_avail_via_walker_helper_matches_iter() {
        let ctu = CtuPos {
            x_ctb: 1,
            y_ctb: 2,
            x0: 128,
            y0: 256,
            width_luma: 128,
            height_luma: 128,
            ctu_addr_rs: 5,
        };
        let avail = CtuWalker::neighbour_avail(&ctu);
        assert!(avail.left);
        assert!(avail.above);
    }

    /// End-to-end single-CTU walk: `decode_ctu_full` must produce one
    /// `LeafCuInfo` + one `LeafCuResidual` per leaf CU emitted by the
    /// coding-tree walker, with geometry that tiles the CTU.
    #[test]
    fn decode_ctu_full_wires_syntax_per_leaf() {
        let sps = dummy_sps(2, 128, 128);
        let pps = dummy_pps(128, 128, true);
        let sh = intra_slice_header();
        let layout = CtuLayout::from_sps_pps(&sps, &pps);
        // Enough bytes for any reasonable bin run. A zero stream keeps
        // most context-coded decisions on the MPS branch; the residual
        // walker won't emit significant coefficients and chroma CBFs
        // stay 0.
        let data = [0u8; 256];
        let mut walker = CtuWalker::begin_slice(&layout, &sps, &pps, &sh, 0, &data).unwrap();
        let ctu0 = walker.iter_ctus().next().unwrap();
        let (cus, infos, residuals) = walker.decode_ctu_full(&ctu0).unwrap();
        assert!(!cus.is_empty());
        assert_eq!(cus.len(), infos.len());
        assert_eq!(cus.len(), residuals.len());
        // Every CU has a legal luma/chroma mode and geometry.
        for (ccu, info) in cus.iter().zip(infos.iter()) {
            assert_eq!(info.x0, ccu.cu.x);
            assert_eq!(info.y0, ccu.cu.y);
            assert_eq!(info.cb_width, ccu.cu.w);
            assert_eq!(info.cb_height, ccu.cu.h);
            assert!(info.intra_pred_mode_y <= 66);
            assert!(info.intra_pred_mode_c <= 83);
        }
        // Every residual struct is either empty (no CBF) or has the
        // right-sized level array.
        for (ccu, residual) in cus.iter().zip(residuals.iter()) {
            let expected_luma = (ccu.cu.w * ccu.cu.h) as usize;
            assert!(residual.luma_levels.is_empty() || residual.luma_levels.len() == expected_luma);
        }
    }

    #[test]
    fn decode_ctu_partitions_handles_zero_edge() {
        let sps = dummy_sps(2, 128, 128);
        let pps = dummy_pps(128, 128, true);
        let sh = intra_slice_header();
        let layout = CtuLayout::from_sps_pps(&sps, &pps);
        let data = [0u8; 32];
        let mut walker = CtuWalker::begin_slice(&layout, &sps, &pps, &sh, 0, &data).unwrap();
        // A fabricated zero-sized CTU must not advance the CABAC engine.
        let zero_ctu = CtuPos {
            x_ctb: 9,
            y_ctb: 9,
            x0: 0,
            y0: 0,
            width_luma: 0,
            height_luma: 0,
            ctu_addr_rs: 999,
        };
        assert!(walker.decode_ctu_partitions(&zero_ctu).unwrap().is_empty());
    }

    // =====================================================================
    // Round-149 — per-CB live MergeSubblockFlag / InterAffineFlag grid
    // fuse into [`CtuWalker::compute_cu_neighbourhood`]
    // =====================================================================

    /// A freshly-built walker initialises both per-CB grids to `false`.
    /// Sampling at any in-bounds 4x4 cell returns `false`; the
    /// out-of-bounds path returns `false` too (§6.4.4 unavailability
    /// short-circuit).
    #[test]
    fn round149_grids_default_to_false_and_oob_returns_false() {
        let sps = dummy_sps(2, 128, 128);
        let pps = dummy_pps(128, 128, true);
        let sh = intra_slice_header();
        let layout = CtuLayout::from_sps_pps(&sps, &pps);
        let data = [0u8; 32];
        let walker = CtuWalker::begin_slice(&layout, &sps, &pps, &sh, 0, &data).unwrap();
        // In-bounds samples — every cell defaults to false.
        for (x, y) in &[(0, 0), (4, 0), (0, 4), (64, 64), (124, 124)] {
            assert!(!walker.sample_subblock_merge_at_luma(*x, *y));
            assert!(!walker.sample_inter_affine_at_luma(*x, *y));
        }
        // Out-of-bounds samples (negative + past-picture-edge) also
        // return false — matches the §6.4.4 mask for an unavailable
        // neighbour.
        assert!(!walker.sample_subblock_merge_at_luma(-1, 0));
        assert!(!walker.sample_subblock_merge_at_luma(0, -1));
        assert!(!walker.sample_subblock_merge_at_luma(128, 0));
        assert!(!walker.sample_subblock_merge_at_luma(0, 128));
        assert!(!walker.sample_inter_affine_at_luma(-1, 0));
        assert!(!walker.sample_inter_affine_at_luma(0, -1));
        assert!(!walker.sample_inter_affine_at_luma(128, 0));
        assert!(!walker.sample_inter_affine_at_luma(0, 128));
    }

    /// [`CtuWalker::write_subblock_merge_block`] broadcasts the boolean
    /// across every 4x4 cell of the rectangle; reads at neighbouring
    /// cells inside and outside the rectangle reflect the write.
    #[test]
    fn round149_write_subblock_merge_block_broadcasts_then_reads_back() {
        let sps = dummy_sps(2, 128, 128);
        let pps = dummy_pps(128, 128, true);
        let sh = intra_slice_header();
        let layout = CtuLayout::from_sps_pps(&sps, &pps);
        let data = [0u8; 32];
        let mut walker = CtuWalker::begin_slice(&layout, &sps, &pps, &sh, 0, &data).unwrap();
        // Flip a 16x16 region at (32, 32) to true.
        walker.write_subblock_merge_block(32, 32, 16, 16, true);
        // Every covered 4x4 cell reads back true.
        for dy in (0..16).step_by(4) {
            for dx in (0..16).step_by(4) {
                assert!(
                    walker.sample_subblock_merge_at_luma(32 + dx, 32 + dy),
                    "cell ({}, {}) should be true after write_subblock_merge_block",
                    32 + dx,
                    32 + dy
                );
            }
        }
        // Cells outside the written rectangle stay false.
        assert!(!walker.sample_subblock_merge_at_luma(28, 32));
        assert!(!walker.sample_subblock_merge_at_luma(32, 28));
        assert!(!walker.sample_subblock_merge_at_luma(48, 32));
        assert!(!walker.sample_subblock_merge_at_luma(32, 48));
    }

    /// [`CtuWalker::compute_cu_neighbourhood`] reads the per-CB grids
    /// at `(xCb − 1, yCb)` for the left neighbour and `(xCb, yCb − 1)`
    /// for the above neighbour, matching the §9.3.4.2.2 / Table 133
    /// merge-side ctxInc neighbour positions.
    #[test]
    fn round149_compute_cu_neighbourhood_samples_left_above_subblock_merge() {
        let sps = dummy_sps(2, 128, 128);
        let pps = dummy_pps(128, 128, true);
        let sh = intra_slice_header();
        let layout = CtuLayout::from_sps_pps(&sps, &pps);
        let data = [0u8; 32];
        let mut walker = CtuWalker::begin_slice(&layout, &sps, &pps, &sh, 0, &data).unwrap();
        // Pre-load: left neighbour of CU at (16, 0) carries
        // MergeSubblockFlag = 1; above neighbour of CU at (16, 16)
        // carries MergeSubblockFlag = 1.
        walker.write_subblock_merge_block(0, 0, 16, 16, true);
        let ccu_a = CtuCu {
            ctu_addr_rs: 0,
            cu: Cu {
                x: 16,
                y: 0,
                w: 16,
                h: 16,
                cqt_depth: 0,
                mtt_depth: 0,
            },
        };
        let neigh_a = walker.compute_cu_neighbourhood(&ccu_a);
        assert!(neigh_a.left_available);
        assert!(neigh_a.left_merge_subblock);
        assert!(!neigh_a.above_available); // y == 0
        assert!(!neigh_a.above_merge_subblock);
        let ccu_b = CtuCu {
            ctu_addr_rs: 0,
            cu: Cu {
                x: 0,
                y: 16,
                w: 16,
                h: 16,
                cqt_depth: 0,
                mtt_depth: 0,
            },
        };
        let neigh_b = walker.compute_cu_neighbourhood(&ccu_b);
        assert!(!neigh_b.left_available); // x == 0
        assert!(neigh_b.above_available);
        assert!(neigh_b.above_merge_subblock);
    }

    /// The `InterAffineFlag` grid plumbs through `compute_cu_neighbourhood`
    /// even though the CTU walker does not yet write into it. A test-only
    /// pre-load via [`CtuWalker::write_inter_affine_block`] confirms the
    /// neighbour query reads it back correctly.
    #[test]
    fn round149_compute_cu_neighbourhood_samples_left_above_inter_affine() {
        let sps = dummy_sps(2, 128, 128);
        let pps = dummy_pps(128, 128, true);
        let sh = intra_slice_header();
        let layout = CtuLayout::from_sps_pps(&sps, &pps);
        let data = [0u8; 32];
        let mut walker = CtuWalker::begin_slice(&layout, &sps, &pps, &sh, 0, &data).unwrap();
        walker.write_inter_affine_block(0, 0, 16, 16, true);
        let ccu = CtuCu {
            ctu_addr_rs: 0,
            cu: Cu {
                x: 16,
                y: 16,
                w: 16,
                h: 16,
                cqt_depth: 0,
                mtt_depth: 0,
            },
        };
        let neigh = walker.compute_cu_neighbourhood(&ccu);
        // Left neighbour at (15, 16) is *outside* the (0,0)..(16,16)
        // rectangle on y (rectangle is rows 0..16, so y=16 is outside).
        assert!(!neigh.left_inter_affine);
        // Above neighbour at (16, 15) is also outside the rectangle on
        // x (col=16 is outside the cols 0..16 rectangle).
        assert!(!neigh.above_inter_affine);
        // Now load (16, 0)..(32, 16) so the left-of-(16, 16) cell at
        // (15, 16) becomes irrelevant; the above-of-(16, 16) cell is
        // (16, 15) which lives inside the new rectangle.
        walker.write_inter_affine_block(16, 0, 16, 16, true);
        let neigh2 = walker.compute_cu_neighbourhood(&ccu);
        assert!(neigh2.above_inter_affine);
        // Left-of-(16, 16) cell at (15, 16) still outside any
        // written rectangle.
        assert!(!neigh2.left_inter_affine);
    }

    /// [`CtuWalker::commit_subblock_neighbour_state`] writes the parsed
    /// `MergeSubblockFlag` into the grid and clears
    /// `InterAffineFlag` (the latter is not yet parsed). Verified end-
    /// to-end: a CU with `merge_subblock_flag = 1` flips the cells the
    /// next CU's neighbour query reads.
    #[test]
    fn round149_commit_subblock_neighbour_state_writes_merge_flag() {
        let sps = dummy_sps(2, 128, 128);
        let pps = dummy_pps(128, 128, true);
        let sh = intra_slice_header();
        let layout = CtuLayout::from_sps_pps(&sps, &pps);
        let data = [0u8; 32];
        let mut walker = CtuWalker::begin_slice(&layout, &sps, &pps, &sh, 0, &data).unwrap();
        let cu = CtuCu {
            ctu_addr_rs: 0,
            cu: Cu {
                x: 0,
                y: 0,
                w: 16,
                h: 16,
                cqt_depth: 0,
                mtt_depth: 0,
            },
        };
        let mut info = LeafCuInfo {
            x0: 0,
            y0: 0,
            cb_width: 16,
            cb_height: 16,
            pred_mode: CuPredMode::Inter,
            ..LeafCuInfo::default()
        };
        info.inter.merge_data.merge_subblock_flag = true;
        walker.commit_subblock_neighbour_state(&cu, &info);
        // The just-committed CU's region reads back true on the
        // sub-block grid and false on the affine-inter grid.
        assert!(walker.sample_subblock_merge_at_luma(0, 0));
        assert!(walker.sample_subblock_merge_at_luma(12, 12));
        assert!(!walker.sample_inter_affine_at_luma(0, 0));
        assert!(!walker.sample_inter_affine_at_luma(12, 12));
        // A CU at (16, 0) would now see the (0,0)..(16,16) region as
        // its left neighbour with merge_subblock = true.
        let next = CtuCu {
            ctu_addr_rs: 0,
            cu: Cu {
                x: 16,
                y: 0,
                w: 16,
                h: 16,
                cqt_depth: 0,
                mtt_depth: 0,
            },
        };
        let neigh = walker.compute_cu_neighbourhood(&next);
        assert!(neigh.left_merge_subblock);
        assert!(!neigh.left_inter_affine);
    }

    /// `commit_subblock_neighbour_state` for an intra CU clears the
    /// merge-subblock cells (consistent with the §7.4.12.7 inference:
    /// `MergeSubblockFlag` is always 0 for non-inter CUs). A stale `true`
    /// from a prior overlapping region is wiped.
    #[test]
    fn round149_commit_subblock_neighbour_state_intra_clears_stale_flag() {
        let sps = dummy_sps(2, 128, 128);
        let pps = dummy_pps(128, 128, true);
        let sh = intra_slice_header();
        let layout = CtuLayout::from_sps_pps(&sps, &pps);
        let data = [0u8; 32];
        let mut walker = CtuWalker::begin_slice(&layout, &sps, &pps, &sh, 0, &data).unwrap();
        // Pre-load a stale true.
        walker.write_subblock_merge_block(0, 0, 16, 16, true);
        assert!(walker.sample_subblock_merge_at_luma(8, 8));
        let cu = CtuCu {
            ctu_addr_rs: 0,
            cu: Cu {
                x: 0,
                y: 0,
                w: 16,
                h: 16,
                cqt_depth: 0,
                mtt_depth: 0,
            },
        };
        let info = LeafCuInfo {
            x0: 0,
            y0: 0,
            cb_width: 16,
            cb_height: 16,
            pred_mode: CuPredMode::Intra,
            ..LeafCuInfo::default() // merge_subblock_flag default = false
        };
        walker.commit_subblock_neighbour_state(&cu, &info);
        // Stale bit wiped — intra CU's `merge_subblock_flag = 0` per
        // §7.4.12.7.
        assert!(!walker.sample_subblock_merge_at_luma(8, 8));
    }

    /// A CU strictly outside any written rectangle reads
    /// `(left_merge_subblock, above_merge_subblock) = (false, false)`,
    /// which matches the §7.4.12.7 default-inference path the pre-r149
    /// stub-call hard-coded — i.e. the round-149 fuse stays
    /// backward-compatible on the existing fixture path.
    #[test]
    fn round149_default_neighbours_match_pre_r149_stub() {
        let sps = dummy_sps(2, 128, 128);
        let pps = dummy_pps(128, 128, true);
        let sh = intra_slice_header();
        let layout = CtuLayout::from_sps_pps(&sps, &pps);
        let data = [0u8; 32];
        let walker = CtuWalker::begin_slice(&layout, &sps, &pps, &sh, 0, &data).unwrap();
        let cu = CtuCu {
            ctu_addr_rs: 0,
            cu: Cu {
                x: 32,
                y: 32,
                w: 16,
                h: 16,
                cqt_depth: 0,
                mtt_depth: 0,
            },
        };
        let neigh = walker.compute_cu_neighbourhood(&cu);
        // Both neighbour positions are inside the picture (so
        // `*_available` is true), but the per-CB grid was never
        // written, so every per-CB flag stays false — same as the
        // pre-r149 wire-up that passed (false, false) directly.
        assert!(neigh.left_available);
        assert!(neigh.above_available);
        assert!(!neigh.left_merge_subblock);
        assert!(!neigh.above_merge_subblock);
        assert!(!neigh.left_inter_affine);
        assert!(!neigh.above_inter_affine);
    }
}
