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

use crate::cabac::ArithDecoder;
use crate::coding_tree::{Cu, TreeCtxs, TreeWalker};
use crate::deblock::{apply_deblocking, DeblockCu, DeblockParams};
use crate::dequant::{dequantize_tb_flat, DequantParams};
use crate::intra::{predict_angular, predict_dc, predict_planar, IntraRefs};
use crate::leaf_cu::{
    CuNeighbourhood, CuPredMode, CuToolFlags, LeafCuCtxs, LeafCuInfo, LeafCuReader, LeafCuResidual,
    INTRA_DC, INTRA_PLANAR,
};
use crate::pps::PicParameterSet;
use crate::reconstruct::{reconstruct_tb_into, OwnedIntraRefs, PictureBuffer};
use crate::slice_header::{SliceType, StatefulSliceHeader};
use crate::sps::SeqParameterSet;
use crate::transform::{implicit_mts_tr_types, inverse_transform_2d, TrType};

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
/// CCLM modes (`81..=83`) are not yet wired — they collapse to PLANAR.
/// Non-cardinal angular modes (anything outside the cardinal set) are
/// snapped to the nearest cardinal via [`nearest_supported_angular`],
/// matching the luma-side fallback used by
/// [`CtuWalker::reconstruct_leaf_cu`].
pub(crate) fn chroma_pred_mode_for_predict(mode_c: u32) -> u32 {
    use crate::leaf_cu::{INTRA_DC, INTRA_LT_CCLM, INTRA_L_CCLM, INTRA_PLANAR, INTRA_T_CCLM};
    match mode_c {
        INTRA_PLANAR => INTRA_PLANAR,
        INTRA_DC => INTRA_DC,
        // CCLM (§8.4.5.2.14) needs the luma reconstruction + linear
        // model derivation; not in this round.
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
        // I-slice only for the scaffold. P/B need inter-specific CABAC
        // tables + reference picture list construction which are out
        // of scope for this increment.
        if sh.sh_slice_type != SliceType::I {
            return Err(Error::unsupported(format!(
                "h266 CTU walker: slice type {:?} not supported by scaffold \
                 (intra-only path implemented; inter needs RPL + inter-CABAC tables)",
                sh.sh_slice_type,
            )));
        }
        // Dual-tree luma/chroma separate partitioning is not yet
        // modelled.
        if sps.partition_constraints.qtbtt_dual_tree_intra_flag {
            return Err(Error::unsupported(
                "h266 CTU walker: dual-tree intra partitioning (separate luma/chroma \
                 coding trees) not supported by scaffold",
            ));
        }
        // Deferred features that would change the per-CTU pipeline shape.
        if sh.sh_alf_enabled_flag {
            return Err(Error::unsupported(
                "h266 CTU walker: per-slice ALF not supported yet",
            ));
        }
        if sh.sh_lmcs_used_flag {
            return Err(Error::unsupported(
                "h266 CTU walker: LMCS (luma mapping / chroma scaling) not supported yet",
            ));
        }
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
        let leaf_ctxs = LeafCuCtxs::init(slice_qp.as_init_qp());

        Ok(Self {
            layout,
            sps,
            pps,
            sh,
            cabac: cabac_state,
            arith,
            leaf_ctxs,
            deblock_cus: Vec::new(),
        })
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
        CuToolFlags {
            ibc: tf.ibc_enabled_flag,
            palette: tf.palette_enabled_flag,
            bdpcm: tf.bdpcm_enabled_flag,
            mip: tf.mip_enabled_flag,
            mrl: tf.mrl_enabled_flag,
            isp: tf.isp_enabled_flag,
            act: tf.act_enabled_flag,
            max_tb_size_y: 64,
            min_tb_size_y: 4,
            max_ts_size: max_ts,
            ctb_size_y: self.layout.ctb_size_y,
            chroma_format_idc: self.sps.sps_chroma_format_idc as u32,
            cu_qp_delta_enabled: self.pps.pps_cu_qp_delta_enabled_flag,
            cu_chroma_qp_offset_enabled: self.sh.sh_cu_chroma_qp_offset_enabled_flag,
            chroma_qp_offset_list_len_minus1: 0,
            joint_cbcr_enabled: tf.joint_cbcr_enabled_flag,
            ts_residual_coding_disabled: self.sh.sh_ts_residual_coding_disabled_flag,
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
        // Neighbour tracking is rudimentary in this round: we pass
        // the default empty neighbourhood so every CU decodes as if it
        // were at the top-left of the slice. The MPM candidate list
        // will therefore fall back to the "neither above DC" branch
        // (PLANAR / 50 / 18 / 46 / 54), which is spec-legal — it's
        // just not the finest context the decoder could have. Refining
        // this will happen when the left/above neighbour grid lands.
        let neigh = CuNeighbourhood::default();
        for ccu in &cus {
            let (info, residual) = self.decode_leaf_cu_syntax(ccu, &neigh)?;
            infos.push(info);
            residuals.push(residual);
        }
        Ok((cus, infos, residuals))
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
        // NB: the low-level TreeWalker ignores neighbour availability
        // today; this is fine for the root of each CTU but biases the
        // ctxInc derivation for internal splits. That precision is
        // tracked in the TreeWalker module.
        let _avail = Self::neighbour_avail(ctu);
        let local = TreeWalker::new(&mut self.arith, &mut self.cabac.tree_ctxs).walk(
            0,
            0,
            ctu.width_luma,
            ctu.height_luma,
        )?;
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
        // MIP / ISP / BDPCM are out of scope for this round — leaf-CU
        // syntax surfaces these only when SPS opts in, so the typical
        // intra-only fixture never hits these branches.
        if info.intra_mip_flag {
            return Err(Error::unsupported(
                "h266 reconstruct_leaf_cu: MIP prediction (§8.4.5.2.15) not yet wired",
            ));
        }
        if info.isp_split != crate::leaf_cu::IspSplitType::NoSplit {
            return Err(Error::unsupported(
                "h266 reconstruct_leaf_cu: ISP split (§8.4.5.2.5) not yet wired",
            ));
        }
        if info.intra_bdpcm_luma {
            return Err(Error::unsupported(
                "h266 reconstruct_leaf_cu: intra BDPCM (§8.7.3 eq. 1153) not yet wired",
            ));
        }

        let bit_depth = self.sps.sps_bitdepth_minus8 as u32 + 8;
        let n_tb_w = cu.cu.w as usize;
        let n_tb_h = cu.cu.h as usize;
        let x0 = cu.cu.x as usize;
        let y0 = cu.cu.y as usize;

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

        // 2. Intra prediction.
        let pred = match info.intra_pred_mode_y {
            INTRA_PLANAR => predict_planar(n_tb_w, n_tb_h, &refs_view)?,
            INTRA_DC => predict_dc(n_tb_w, n_tb_h, &refs_view)?,
            mode @ (2 | 18 | 34 | 50 | 66) => predict_angular(n_tb_w, n_tb_h, mode, &refs_view)?,
            // Fallback: angular mode outside the implemented cardinal
            // subset. Snap to the nearest cardinal/diagonal so the CU
            // still gets a plausible prediction; the accumulated
            // residual will absorb the difference (lossy but stable).
            mode if (2..=66).contains(&mode) => {
                let snapped = nearest_supported_angular(mode);
                predict_angular(n_tb_w, n_tb_h, snapped, &refs_view)?
            }
            // Anything else is a spec-illegal mode; prefer PLANAR over
            // bailing the whole picture so the stream still produces
            // pixels.
            _ => predict_planar(n_tb_w, n_tb_h, &refs_view)?,
        };

        // 3. Dequantise + 4. inverse 2D transform when there is a coded
        // luma TB. Otherwise skip straight to reconstruction (residual
        // is implicitly zero, eq. 1426 still applies).
        let residual_samples: Vec<i32> = if info.tu_y_coded_flag && !residual.luma_levels.is_empty()
        {
            let qp = self.cabac.slice_qp_y.0 + info.cu_qp_delta_val;
            let qp = qp.clamp(0, 63);
            let params = DequantParams {
                bit_depth,
                log2_transform_range: 15,
                n_tb_w: n_tb_w as u32,
                n_tb_h: n_tb_h as u32,
                qp,
                dep_quant: false,
                transform_skip: false,
                bdpcm: false,
            };
            let d = dequantize_tb_flat(&residual.luma_levels, &params)?;
            // For the first cut we always pick DCT-II / DCT-II per
            // §8.7.4.4 (the `mts_idx == 0` branch). Implicit MTS
            // would substitute DST-VII for small intra TBs, but
            // sticking with DCT-II keeps the path symmetrical with
            // the encoder scaffold and matches the Common Test
            // Conditions baseline. nTbS in {4, 8, 16, 32} are
            // supported by inverse_transform_2d today.
            // DCT-II / DCT-II for nTbS ∈ {4, 8, 16, 32, 64}. The size-64
            // path is now wired through `apply_dct_ii` against the full
            // 64×64 trMatrix (§8.7.4.2 eq. 1184).
            let (tr_h, tr_v) = (TrType::DctII, TrType::DctII);
            let _ = implicit_mts_tr_types(n_tb_w as u32, n_tb_h as u32);
            inverse_transform_2d(
                n_tb_w, n_tb_h, n_tb_w, n_tb_h, tr_h, tr_v, &d, bit_depth, 15,
            )?
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
    /// Chroma TBs smaller than 4×4 (e.g. when the luma CU is 4×4 with
    /// 4:2:0 → 2×2 chroma) have no coded residual emitted by the leaf
    /// reader and bypass dequant + inverse transform — the prediction
    /// is written through unchanged. This keeps the path stable for
    /// minimum-CB sizes without touching the still-pending size-2
    /// transform support.
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

        // 2. Intra prediction. Map the §8.4.3 derived `IntraPredModeC`
        // through the cardinal-only fallback used by the luma path.
        let mapped = chroma_pred_mode_for_predict(info.intra_pred_mode_c);
        let pred = match mapped {
            INTRA_PLANAR => predict_planar(n_tb_w, n_tb_h, &refs_view)?,
            INTRA_DC => predict_dc(n_tb_w, n_tb_h, &refs_view)?,
            mode @ (2 | 18 | 34 | 50 | 66) => predict_angular(n_tb_w, n_tb_h, mode, &refs_view)?,
            _ => predict_planar(n_tb_w, n_tb_h, &refs_view)?,
        };

        // 3. Dequantise + 4. inverse 2D transform when there is a coded
        // chroma TB and the chroma TB is at least 4 samples on a side.
        // Sizes 2×2 / 2×4 / 4×2 still go through the prediction path
        // but skip dequant + IDCT (the leaf reader does not emit the
        // residual for those sizes today, and the transform module
        // does not yet implement size-2 kernels).
        let residual_samples: Vec<i32> =
            if coded_flag && !levels.is_empty() && n_tb_w >= 4 && n_tb_h >= 4 {
                // §8.7.1 chroma QP: identity mapping qPi → QpC, plus the
                // additive PPS / slice / CU offsets. Slice-level chroma
                // offsets (`sh_cb_qp_offset` / `sh_cr_qp_offset`) are
                // captured in the slice header but not yet plumbed to this
                // call site; only the PPS + CU offsets contribute today.
                let pps_offset = match c_idx {
                    1 => self.pps.pps_cb_qp_offset,
                    2 => self.pps.pps_cr_qp_offset,
                    _ => 0,
                };
                let cu_offset = if info.cu_chroma_qp_offset_flag {
                    // Per §7.4.10.6 the CU offset value is read from the
                    // PPS chroma-offset list; the list itself is not yet
                    // plumbed (round-10 surfaces the flag + index only).
                    // Identity value 0 is the safe default.
                    0
                } else {
                    0
                };
                let qp = chroma_qp_identity(self.cabac.slice_qp_y.0, pps_offset + cu_offset);
                let params = DequantParams {
                    bit_depth,
                    log2_transform_range: 15,
                    n_tb_w: n_tb_w as u32,
                    n_tb_h: n_tb_h as u32,
                    qp,
                    dep_quant: false,
                    transform_skip: false,
                    bdpcm: false,
                };
                let d = dequantize_tb_flat(levels, &params)?;
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
                    15,
                )?
            } else {
                vec![0i32; n_tb_w * n_tb_h]
            };

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
        let ctus: Vec<CtuPos> = self.iter_ctus().collect();
        for ctu in &ctus {
            let (cus, infos, residuals) = self.decode_ctu_full(ctu)?;
            for ((ccu, info), residual) in cus.iter().zip(infos.iter()).zip(residuals.iter()) {
                self.reconstruct_leaf_cu(ccu, info, residual, out)?;
            }
        }
        Ok(())
    }

    /// Apply the §8.8 in-loop filters to a partially-reconstructed
    /// picture. Round-12 wires only the §8.8.3 deblocking pass; SAO
    /// (§8.8.4) and ALF (§8.8.5) remain pending and are skipped (they
    /// are no-ops when their slice / SPS enable-flags are off, which
    /// holds in every fixture this scaffold currently emits).
    ///
    /// The CU records consumed by the deblocker are accumulated by
    /// [`Self::reconstruct_leaf_cu`]; calling
    /// `apply_in_loop_filters` before any picture-decode call is
    /// therefore a no-op (the empty CU list never finds a neighbour
    /// pair).
    pub fn apply_in_loop_filters(&mut self, out: &mut PictureBuffer) -> Result<()> {
        // Resolve deblock parameters from the slice header (§7.4.8) +
        // the picture header / PPS fall-backs already applied during
        // parse.
        let bit_depth = self.sps.sps_bitdepth_minus8 as u32 + 8;
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
        // SAO (§8.8.4) — needs sh_sao_*_used_flag + per-CTB
        // SaoTypeIdx / SaoEoClass / SaoOffsetVal arrays from the
        // slice header. Not emitted by the round-12 syntax walker
        // yet; running the spec's "type 0" no-op is the spec-correct
        // path when those parameters are absent.
        // ALF (§8.8.5) — gated on `sh_alf_enabled_flag`, which the
        // CTU-walker constructor already rejects (see `begin_slice`).
        let _ = (
            self.sh.sh_sao_luma_used_flag,
            self.sh.sh_sao_chroma_used_flag,
        );
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
    fn walker_rejects_inter_slice_types() {
        let sps = dummy_sps(2, 128, 128);
        let pps = dummy_pps(128, 128, true);
        let mut sh = intra_slice_header();
        sh.sh_slice_type = SliceType::P;
        let layout = CtuLayout::from_sps_pps(&sps, &pps);
        let data = [0u8; 8];
        let err = CtuWalker::begin_slice(&layout, &sps, &pps, &sh, 0, &data).unwrap_err();
        assert!(matches!(err, Error::Unsupported(_)));
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
    fn walker_rejects_alf_lmcs_dep_quant_scaling_list() {
        let sps = dummy_sps(2, 128, 128);
        let pps = dummy_pps(128, 128, true);
        let layout = CtuLayout::from_sps_pps(&sps, &pps);
        let data = [0u8; 8];

        let mut sh = intra_slice_header();
        sh.sh_alf_enabled_flag = true;
        assert!(matches!(
            CtuWalker::begin_slice(&layout, &sps, &pps, &sh, 0, &data).unwrap_err(),
            Error::Unsupported(_)
        ));

        let mut sh = intra_slice_header();
        sh.sh_lmcs_used_flag = true;
        assert!(matches!(
            CtuWalker::begin_slice(&layout, &sps, &pps, &sh, 0, &data).unwrap_err(),
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

    /// `chroma_pred_mode_for_predict` collapses CCLM modes to PLANAR
    /// and snaps non-cardinal angulars to the nearest cardinal. PLANAR /
    /// DC / cardinals pass through verbatim.
    #[test]
    fn chroma_pred_mode_mapping_collapses_cclm_and_snaps_angulars() {
        use crate::leaf_cu::{INTRA_DC, INTRA_LT_CCLM, INTRA_L_CCLM, INTRA_PLANAR, INTRA_T_CCLM};
        assert_eq!(chroma_pred_mode_for_predict(INTRA_PLANAR), INTRA_PLANAR);
        assert_eq!(chroma_pred_mode_for_predict(INTRA_DC), INTRA_DC);
        assert_eq!(chroma_pred_mode_for_predict(50), 50);
        assert_eq!(chroma_pred_mode_for_predict(2), 2);
        assert_eq!(chroma_pred_mode_for_predict(66), 66);
        // Non-cardinal angular -> snap to nearest cardinal (matches
        // the luma fallback).
        assert_eq!(chroma_pred_mode_for_predict(40), 34);
        // CCLM modes collapse to PLANAR until §8.4.5.2.14 lands.
        assert_eq!(chroma_pred_mode_for_predict(INTRA_LT_CCLM), INTRA_PLANAR);
        assert_eq!(chroma_pred_mode_for_predict(INTRA_L_CCLM), INTRA_PLANAR);
        assert_eq!(chroma_pred_mode_for_predict(INTRA_T_CCLM), INTRA_PLANAR);
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

    /// The unsupported-error escape hatch still fires for the MIP path
    /// even though the surrounding pipeline now succeeds.
    #[test]
    fn reconstruct_leaf_cu_mip_is_unsupported() {
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
        let residual = LeafCuResidual::default();
        let mut out = PictureBuffer::yuv420_filled(128, 128, 0);
        let err = walker
            .reconstruct_leaf_cu(&ccu, &info, &residual, &mut out)
            .unwrap_err();
        assert!(matches!(err, Error::Unsupported(_)), "got {err:?}");
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
}
