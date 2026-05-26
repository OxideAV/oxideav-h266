//! VVC leaf coding-unit syntax + intra-mode derivation.
//!
//! This module covers the *per-CU* parsing hop between the coding-tree
//! walker (which produces leaf rectangles) and the reconstruction
//! pipeline (which turns an intra mode + residual into pixels). The
//! focus of this round is strictly the syntax reads from the spec's
//! `coding_unit()` structure (§7.3.11.5) plus the derivations
//! in §8.4.2 (luma intra mode) and §8.4.3 (chroma intra mode).
//!
//! Covered syntax elements (I-slice, single-tree subset):
//!
//! * `pred_mode_flag`, `pred_mode_ibc_flag`, `pred_mode_plt_flag` —
//!   coarse prediction-mode dispatch. In I-slices `pred_mode_flag` is
//!   not signalled and is inferred to 1 (MODE_INTRA) per spec §7.4.12.2.
//! * `intra_bdpcm_luma_flag` / `intra_bdpcm_luma_dir_flag` —
//!   block DPCM (§7.4.12.2). Only parsed when the SPS enables it.
//! * `intra_mip_flag` + `intra_mip_transposed_flag` + `intra_mip_mode` —
//!   Matrix-based intra prediction (§8.4.5.2.15). Parsed when the SPS
//!   enables it and the CU size qualifies.
//! * `intra_luma_ref_idx` — multi-reference-line selection (§7.4.12.2).
//!   Parsed via TR(cMax=2) when MRL is enabled and `y0 % CtbSizeY > 0`.
//! * `intra_subpartitions_mode_flag` / `intra_subpartitions_split_flag`
//!   — ISP selector (§7.4.12.2, Table 13).
//! * `intra_luma_mpm_flag`, `intra_luma_not_planar_flag`,
//!   `intra_luma_mpm_idx`, `intra_luma_mpm_remainder` — the MPM cascade
//!   that the luma intra-mode derivation (§8.4.2) reads out of.
//! * `intra_chroma_pred_mode` — Table 130 binarisation (ctx bin 0 +
//!   two bypass bins). Mapped to `IntraPredModeC` via Table 20 in the
//!   caller's derivation step.
//!
//! Out of scope (still returns `Error::Unsupported` one layer up):
//!
//! * CCLM (`cclm_mode_flag` / `cclm_mode_idx`) — chroma from luma.
//! * IBC / PLT CUs — IBC parsing + palette coding.
//! * CBF / residual coding / transforms / inverse quantisation.
//! * Dual-tree luma / chroma split. Chroma intra mode is derived for
//!   `treeType == SINGLE_TREE` only.
//!
//! The module is intentionally "parse + derive, don't reconstruct" —
//! [`LeafCuInfo`] captures everything a follow-up round needs to build
//! reference samples + drive the intra predictor, but nothing here
//! touches pixels.
//!
//! Spec reference: ITU-T H.266 | ISO/IEC 23090-3 (V4, 01/2026). The
//! implementation is spec-only; no third-party VVC decoder source was
//! consulted.

use oxideav_core::{Error, Result};

use crate::cabac::{ArithDecoder, ContextModel};
use crate::ctx::{
    ctx_inc_abs_mvd_greater0_flag, ctx_inc_abs_mvd_greater1_flag, ctx_inc_bcw_idx,
    ctx_inc_cu_skip_flag, ctx_inc_general_merge_flag, ctx_inc_inter_affine_flag,
    ctx_inc_inter_pred_idc_bin0, ctx_inc_inter_pred_idc_bin1, ctx_inc_intra_bdpcm_chroma_dir_flag,
    ctx_inc_intra_bdpcm_chroma_flag, ctx_inc_intra_bdpcm_luma_dir_flag,
    ctx_inc_intra_bdpcm_luma_flag, ctx_inc_intra_chroma_pred_mode, ctx_inc_intra_luma_mpm_flag,
    ctx_inc_intra_luma_not_planar_flag, ctx_inc_intra_luma_ref_idx, ctx_inc_intra_mip_flag,
    ctx_inc_intra_subpartitions_mode_flag, ctx_inc_intra_subpartitions_split_flag,
    ctx_inc_merge_subblock_flag, ctx_inc_merge_subblock_idx, ctx_inc_mvp_lx_flag,
    ctx_inc_ref_idx_lx, ctx_inc_regular_merge_flag, ctx_inc_sym_mvd_flag,
};
use crate::inter::{InterCuInfo, MergeData, MotionVector, MvField};
use crate::residual::{
    decode_tb_coefficients, read_cu_chroma_qp_offset, read_cu_qp_delta, read_tu_cb_coded_flag,
    read_tu_cr_coded_flag, read_tu_y_coded_flag, ResidualCtxs,
};
use crate::tables::{init_contexts, SyntaxCtx};

/// CU prediction mode (§7.4.12.2 — `MODE_*`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CuPredMode {
    /// Regular intra-predicted CU.
    Intra,
    /// Inter-predicted (not reachable in I-slices).
    Inter,
    /// Intra-block-copy CU (IBC).
    Ibc,
    /// Palette-mode CU.
    Plt,
}

/// Inter prediction direction (§7.4.12.4 / Table for `inter_pred_idc`).
/// The numeric value matches the spec's `inter_pred_idc` syntax value
/// (`PRED_L0 = 0`, `PRED_L1 = 1`, `PRED_BI = 2`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InterPredDir {
    /// `PRED_L0` — uni-prediction from reference list 0.
    PredL0,
    /// `PRED_L1` — uni-prediction from reference list 1.
    PredL1,
    /// `PRED_BI` — bi-prediction from both reference lists.
    PredBi,
}

impl InterPredDir {
    /// The §7.4.12.4 numeric `inter_pred_idc` value.
    pub fn value(self) -> u32 {
        match self {
            InterPredDir::PredL0 => 0,
            InterPredDir::PredL1 => 1,
            InterPredDir::PredBi => 2,
        }
    }

    fn from_value(v: u32) -> Self {
        match v {
            0 => InterPredDir::PredL0,
            1 => InterPredDir::PredL1,
            _ => InterPredDir::PredBi,
        }
    }
}

/// §7.3.10.5 gate inputs for the `bcw_idx[x0][y0]` syntax element.
///
/// The spec's `coding_unit()` else-branch only signals `bcw_idx` when
/// *all* of these conditions hold (verbatim from §7.3.10.5):
///
/// ```text
/// if( sps_bcw_enabled_flag && inter_pred_idc[x0][y0] == PRED_BI &&
///     luma_weight_l0_flag[ ref_idx_l0[x0][y0] ]   == 0 &&
///     luma_weight_l1_flag[ ref_idx_l1[x0][y0] ]   == 0 &&
///     chroma_weight_l0_flag[ ref_idx_l0[x0][y0] ] == 0 &&
///     chroma_weight_l1_flag[ ref_idx_l1[x0][y0] ] == 0 &&
///     cbWidth * cbHeight >= 256 )
///   bcw_idx[x0][y0]                                    ae(v)
/// ```
///
/// When the gate is closed the value is inferred 0 per §7.4.12.5
/// ("When `bcw_idx[x0][y0]` is not present, it is inferred to be equal
/// to 0").
///
/// The caller (the CTU walker, once it brings the non-merge inter path
/// online) fills this struct from live per-CB state: `inter_pred_idc`
/// comes from [`LeafCuReader::read_inter_pred_idc`], the luma /
/// chroma weight flags from the parsed `pred_weight_table()`
/// (round-29), and `cb_w` / `cb_h` from the leaf rectangle. The
/// `no_backward_pred_flag` field is `NoBackwardPredFlag` from
/// §7.4.11.6 (true ⇔ none of the slice's L1 references are temporally
/// after the current picture, so the BCW weight table's "backward"
/// indices are unreachable and `cMax = 4` instead of `2`).
#[derive(Clone, Copy, Debug, Default)]
pub struct BcwIdxGate {
    /// `sps_bcw_enabled_flag` from the active SPS.
    pub sps_bcw_enabled: bool,
    /// `inter_pred_idc[x0][y0]` — the gate closes unless this is
    /// [`InterPredDir::PredBi`].
    pub inter_pred_idc: Option<InterPredDir>,
    /// `luma_weight_l0_flag[ ref_idx_l0[x0][y0] ]` — true closes the
    /// gate (explicit weighted-prediction on L0 luma).
    pub luma_weight_l0_flag: bool,
    /// `luma_weight_l1_flag[ ref_idx_l1[x0][y0] ]` — true closes the
    /// gate.
    pub luma_weight_l1_flag: bool,
    /// `chroma_weight_l0_flag[ ref_idx_l0[x0][y0] ]` — true closes the
    /// gate.
    pub chroma_weight_l0_flag: bool,
    /// `chroma_weight_l1_flag[ ref_idx_l1[x0][y0] ]` — true closes the
    /// gate.
    pub chroma_weight_l1_flag: bool,
    /// CU luma-block width `cbWidth`.
    pub cb_width: u32,
    /// CU luma-block height `cbHeight`.
    pub cb_height: u32,
    /// `NoBackwardPredFlag` per §7.4.11.6. Threads straight into the
    /// `cMax` selection inside [`LeafCuReader::read_bcw_idx`].
    pub no_backward_pred_flag: bool,
}

impl BcwIdxGate {
    /// `true` iff the §7.3.10.5 conditional opens, i.e. the next bin
    /// in the bitstream really is `bcw_idx[x0][y0]`.
    pub fn is_open(&self) -> bool {
        self.sps_bcw_enabled
            && self.inter_pred_idc == Some(InterPredDir::PredBi)
            && !self.luma_weight_l0_flag
            && !self.luma_weight_l1_flag
            && !self.chroma_weight_l0_flag
            && !self.chroma_weight_l1_flag
            && (self.cb_width.saturating_mul(self.cb_height)) >= 256
    }
}

/// Intra subpartitions split type (Table 13).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IspSplitType {
    /// `ISP_NO_SPLIT` — ISP disabled for this CU.
    NoSplit,
    /// `ISP_HOR_SPLIT` — horizontal partition.
    HorSplit,
    /// `ISP_VER_SPLIT` — vertical partition.
    VerSplit,
}

/// Tool-enable flags that gate the `coding_unit()` syntax reads in
/// the SPS. Grouped here for test convenience; production callers pass
/// a [`crate::sps::SeqParameterSet`] view into [`LeafCuReader::new`].
#[derive(Clone, Copy, Debug, Default)]
pub struct CuToolFlags {
    /// `sps_ibc_enabled_flag`.
    pub ibc: bool,
    /// `sps_palette_enabled_flag`.
    pub palette: bool,
    /// `sps_bdpcm_enabled_flag`.
    pub bdpcm: bool,
    /// `sps_mip_enabled_flag`.
    pub mip: bool,
    /// `sps_mrl_enabled_flag`.
    pub mrl: bool,
    /// `sps_isp_enabled_flag`.
    pub isp: bool,
    /// `sps_act_enabled_flag`.
    pub act: bool,
    /// `MaxTbSizeY` — max transform block size in luma samples.
    pub max_tb_size_y: u32,
    /// `MinTbSizeY` — min transform block size in luma samples.
    pub min_tb_size_y: u32,
    /// `MaxTsSize` — max transform-skip block size in luma samples
    /// (§7.4.3.4 via `sps_transform_skip_enabled_flag`).
    pub max_ts_size: u32,
    /// `CtbSizeY`.
    pub ctb_size_y: u32,
    /// `sps_chroma_format_idc` (0 = monochrome).
    pub chroma_format_idc: u32,
    /// `pps_cu_qp_delta_enabled_flag` — gates `cu_qp_delta_abs` reads.
    pub cu_qp_delta_enabled: bool,
    /// `sh_cu_chroma_qp_offset_enabled_flag` — gates
    /// `cu_chroma_qp_offset_flag` reads.
    pub cu_chroma_qp_offset_enabled: bool,
    /// `pps_chroma_qp_offset_list_len_minus1` — cMax for the
    /// `cu_chroma_qp_offset_idx` TR read.
    pub chroma_qp_offset_list_len_minus1: u32,
    /// `sps_joint_cbcr_enabled_flag` — if true, `tu_joint_cbcr_residual_flag`
    /// may appear (currently surfaced as Unsupported when it would be read).
    pub joint_cbcr_enabled: bool,
    /// `sh_ts_residual_coding_disabled_flag`. When set the
    /// `transform_skip_flag` path is elided in the residual walker.
    pub ts_residual_coding_disabled: bool,
    /// True for P/B-slices (i.e. `sh_slice_type != I`). Routes the
    /// reader through the round-21 cu_skip_flag → merge_data path
    /// instead of the I-slice intra path.
    pub slice_is_inter: bool,
    /// `MaxNumMergeCand` derived per §7.4.3.4 eq. 58 (`6 -
    /// sps_six_minus_max_num_merge_cand`). Only meaningful when
    /// `slice_is_inter == true`.
    pub max_num_merge_cand: u32,
    /// `sps_mmvd_enabled_flag` — round-27 §8.5.2.7. Gates the
    /// `mmvd_merge_flag` parse inside `merge_data()` per §7.3.11.7.
    /// When false, the MMVD branch collapses (matches the round-21..26
    /// behaviour where MMVD was not yet wired).
    pub mmvd_enabled: bool,
    /// `ph_mmvd_fullpel_only_flag` — picture-header switch that swaps
    /// Table 17 from the regular `MMVD_DISTANCE_TABLE` to the fullpel-
    /// scaled `MMVD_DISTANCE_TABLE_FULLPEL`. Only meaningful when
    /// `mmvd_enabled` is true. Round-27 plumbs this through
    /// [`crate::inter::derive_mmvd_offset`] at reconstruction time.
    pub ph_mmvd_fullpel_only: bool,
    /// `sps_ciip_enabled_flag` — round-28 §8.5.6.7. Gates the
    /// `regular_merge_flag` parse inside `merge_data()` (without
    /// CIIP / GPM the gate collapses and `regular_merge_flag` is
    /// inferred to 1) and the §7.4.12.7 `ciip_flag` inference. When
    /// `false` the merge-data parser ignores CIIP entirely.
    pub ciip_enabled: bool,
    /// `sps_gpm_enabled_flag` — geometric partitioning merge. Round-40
    /// (this round) wires the §7.3.11.7 GPM branch end-to-end: the
    /// `merge_gpm_partition_idx`, `merge_gpm_idx0`, `merge_gpm_idx1`
    /// syntax elements are parsed and the §8.5.4 / §8.5.7 reconstruction
    /// path runs through [`crate::gpm`]. Pre-round-40 callers can still
    /// leave this `false` to fall back to the round-28 CIIP-only
    /// behaviour.
    pub gpm_enabled: bool,
    /// `MaxNumGpmMergeCand` — derived per §7.4.3.4 eq. 60 from
    /// `sps_max_num_merge_cand_minus_max_num_gpm_cand`. Caps the GPM
    /// merge-cand list size and drives the TR binarisation of
    /// `merge_gpm_idx0` (`cMax = MaxNumGpmMergeCand − 1`) and
    /// `merge_gpm_idx1` (`cMax = MaxNumGpmMergeCand − 2`).
    pub max_num_gpm_merge_cand: u32,
    /// True for B-slices. The §7.3.11.7 GPM branch is gated on this
    /// (P-slices never enter GPM); the round-21..30 paths only ever
    /// consulted `slice_is_inter`.
    pub slice_is_b: bool,
    /// `MaxNumSubblockMergeCand` derived per §7.4.3.4 eq. 85 from
    /// `sps_affine_enabled_flag`, `sps_five_minus_max_num_subblock_merge_cand`,
    /// `sps_sbtmvp_enabled_flag` and `ph_temporal_mvp_enabled_flag`
    /// (see [`crate::sps::SeqParameterSet::max_num_subblock_merge_cand`]).
    /// Round-146 wire-up: gates the §7.3.11.7
    /// `merge_subblock_flag` parse (only emitted when this is `> 0`
    /// AND `cbW >= 8 && cbH >= 8`) and caps the `merge_subblock_idx`
    /// TR binarisation (`cMax = MaxNumSubblockMergeCand − 1`). When
    /// `0` the subblock-merge branch collapses entirely and the
    /// §7.4.12.7 inference forces `merge_subblock_flag = 0`,
    /// re-routing the merge sub-tree through the regular / MMVD /
    /// CIIP / GPM path. Only meaningful when `slice_is_inter == true`.
    pub max_num_subblock_merge_cand: u32,
}

/// Parsed + derived per-CU state for an intra leaf CU.
///
/// Holds every syntax element read by [`LeafCuReader`] for a single
/// coding unit, plus the spec-derived luma / chroma intra prediction
/// modes (§8.4.2 / §8.4.3). The decoder pipeline in later rounds will
/// consume this struct to drive reference-sample fetch + prediction
/// without replaying the CABAC reads.
///
/// CBFs and the raw coefficient level array live in a sibling
/// [`LeafCuResidual`] because they are not `Copy` (they carry a
/// `Vec<i32>`). The residual struct stays `None` when the CU has no
/// residual to decode (e.g. BDPCM pure-prediction path or a CU where
/// `tu_y_coded_flag == 0` and chroma CBFs are 0).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LeafCuInfo {
    /// Top-left luma x (picture-absolute).
    pub x0: u32,
    /// Top-left luma y (picture-absolute).
    pub y0: u32,
    /// CB width in luma samples.
    pub cb_width: u32,
    /// CB height in luma samples.
    pub cb_height: u32,
    /// `CuPredMode[chType][x0][y0]`.
    pub pred_mode: CuPredMode,
    /// `BdpcmFlag[x0][y0][0]`.
    pub intra_bdpcm_luma: bool,
    /// `BdpcmDir[x0][y0][0]` — direction (false = horizontal, true = vertical).
    pub intra_bdpcm_luma_dir: bool,
    /// `BdpcmFlag[x0][y0][1..2]` — BDPCM applied to both Cb and Cr.
    pub intra_bdpcm_chroma: bool,
    /// `BdpcmDir[x0][y0][1..2]` — chroma BDPCM direction (false = horizontal,
    /// true = vertical).
    pub intra_bdpcm_chroma_dir: bool,
    /// `IntraMipFlag[x0][y0]`.
    pub intra_mip_flag: bool,
    /// `intra_mip_transposed_flag`.
    pub intra_mip_transposed_flag: bool,
    /// `intra_mip_mode`.
    pub intra_mip_mode: u32,
    /// `IntraLumaRefLineIdx[x0][y0]`.
    pub intra_luma_ref_idx: u32,
    /// Derived ISP split type.
    pub isp_split: IspSplitType,
    /// `intra_luma_mpm_flag[x0][y0]` (inferred 1 if not present).
    pub intra_luma_mpm_flag: bool,
    /// `intra_luma_not_planar_flag[x0][y0]` (inferred 1 if not present).
    pub intra_luma_not_planar_flag: bool,
    /// `intra_luma_mpm_idx[x0][y0]` (0 when MPM flag is 0).
    pub intra_luma_mpm_idx: u32,
    /// `intra_luma_mpm_remainder[x0][y0]` (present only when MPM flag == 0).
    pub intra_luma_mpm_remainder: u32,
    /// `intra_chroma_pred_mode` (inferred 0 when not present).
    pub intra_chroma_pred_mode: u32,
    /// `IntraPredModeY[x0][y0]` derived per §8.4.2 — 0..66 for regular
    /// intra, encoded verbatim when MIP is in use (MIP has its own
    /// mode namespace `intra_mip_mode`).
    pub intra_pred_mode_y: u32,
    /// `IntraPredModeC[x0][y0]` derived per §8.4.3 / Table 20. For
    /// monochrome sequences this remains 0 and callers should ignore
    /// the chroma plane.
    pub intra_pred_mode_c: u32,
    /// `tu_y_coded_flag[x0][y0]` — luma CBF for the single-TB CU case.
    pub tu_y_coded_flag: bool,
    /// `tu_cb_coded_flag[x0][y0]`.
    pub tu_cb_coded_flag: bool,
    /// `tu_cr_coded_flag[x0][y0]`.
    pub tu_cr_coded_flag: bool,
    /// `CuQpDeltaVal` (signed, §7.4.11.8). 0 when `cu_qp_delta_abs == 0`
    /// or the flag was not present in the slice.
    pub cu_qp_delta_val: i32,
    /// `cu_chroma_qp_offset_flag`.
    pub cu_chroma_qp_offset_flag: bool,
    /// `cu_chroma_qp_offset_idx` (0 when the flag is 0).
    pub cu_chroma_qp_offset_idx: u32,
    /// `LastSignificantCoeffX` for the luma TB (0 when no residual).
    pub last_sig_x: u32,
    /// `LastSignificantCoeffY` for the luma TB.
    pub last_sig_y: u32,
    /// Inter-coding side state populated by the P/B-slice parse path
    /// (round-21). Empty/default when the CU is intra.
    pub inter: InterCuInfo,
}

impl Default for LeafCuInfo {
    fn default() -> Self {
        Self {
            x0: 0,
            y0: 0,
            cb_width: 0,
            cb_height: 0,
            pred_mode: CuPredMode::Intra,
            intra_bdpcm_luma: false,
            intra_bdpcm_luma_dir: false,
            intra_bdpcm_chroma: false,
            intra_bdpcm_chroma_dir: false,
            intra_mip_flag: false,
            intra_mip_transposed_flag: false,
            intra_mip_mode: 0,
            intra_luma_ref_idx: 0,
            isp_split: IspSplitType::NoSplit,
            intra_luma_mpm_flag: true,
            intra_luma_not_planar_flag: true,
            intra_luma_mpm_idx: 0,
            intra_luma_mpm_remainder: 0,
            intra_chroma_pred_mode: 0,
            intra_pred_mode_y: INTRA_PLANAR,
            intra_pred_mode_c: INTRA_PLANAR,
            tu_y_coded_flag: false,
            tu_cb_coded_flag: false,
            tu_cr_coded_flag: false,
            cu_qp_delta_val: 0,
            cu_chroma_qp_offset_flag: false,
            cu_chroma_qp_offset_idx: 0,
            last_sig_x: 0,
            last_sig_y: 0,
            inter: InterCuInfo {
                cu_skip_flag: false,
                general_merge_flag: false,
                merge_data: MergeData {
                    regular_merge_flag: false,
                    merge_idx: 0,
                    mmvd_merge_flag: false,
                    mmvd_cand_flag: 0,
                    mmvd_distance_idx: 0,
                    mmvd_direction_idx: 0,
                    ciip_flag: false,
                    gpm_flag: false,
                    gpm_partition_idx: 0,
                    gpm_idx0: 0,
                    gpm_idx1: 0,
                    merge_subblock_flag: false,
                    merge_subblock_idx: 0,
                },
            },
        }
    }
}

/// Per-subpartition luma residual entry used when ISP is active.
///
/// One [`LeafCuLumaSubpart`] is emitted per ISP subpartition, in
/// spec walk order (top-to-bottom for `ISP_HOR_SPLIT`, left-to-right
/// for `ISP_VER_SPLIT`). The `levels` array is row-major and sized
/// `(n_w * n_h)`; it stays empty when the subpartition's
/// `tu_y_coded_flag` is 0 (which can happen for partitions
/// `0..numParts-2` — the last partition has its CBF inferred per
/// `InferTuCbfLuma` if every prior partition was zero).
///
/// Spec: §8.4.5.1 eqs. 251 – 260 derive `n_w` / `n_h` per partition.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LeafCuLumaSubpart {
    /// Sub-TB width in luma samples.
    pub n_w: u32,
    /// Sub-TB height in luma samples.
    pub n_h: u32,
    /// CB-relative top-left luma position of the subpartition.
    pub x_offset: u32,
    /// CB-relative top-left luma position of the subpartition.
    pub y_offset: u32,
    /// Per-partition `tu_y_coded_flag[x0][y0]`.
    pub tu_y_coded_flag: bool,
    /// Coefficient levels (row-major), empty when not coded.
    pub levels: Vec<i32>,
}

/// Residual side-state for a leaf CU that has at least one coded TB.
/// Held alongside [`LeafCuInfo`] because the level arrays require an
/// allocation.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LeafCuResidual {
    /// Luma coefficient levels, row-major `(cb_width * cb_height)`.
    /// Empty when `tu_y_coded_flag == 0` or when ISP is active (in
    /// which case [`Self::luma_subparts`] holds the per-partition
    /// arrays instead).
    pub luma_levels: Vec<i32>,
    /// Per-subpartition luma residual when ISP is active. Empty when
    /// the CU is not ISP-split (in which case [`Self::luma_levels`]
    /// holds the single-TB residual).
    pub luma_subparts: Vec<LeafCuLumaSubpart>,
    /// Chroma Cb coefficient levels (row-major, chroma-plane size).
    pub cb_levels: Vec<i32>,
    /// Chroma Cr coefficient levels (row-major, chroma-plane size).
    pub cr_levels: Vec<i32>,
}

/// §7.3.11.5 syntax values for the intra-mode constants (Table 19).
pub const INTRA_PLANAR: u32 = 0;
pub const INTRA_DC: u32 = 1;
pub const INTRA_ANGULAR18: u32 = 18;
pub const INTRA_ANGULAR46: u32 = 46;
pub const INTRA_ANGULAR50: u32 = 50;
pub const INTRA_ANGULAR54: u32 = 54;

/// Chroma-direct modes emitted by Table 20 when `cclm_mode_flag == 1`.
pub const INTRA_LT_CCLM: u32 = 81;
pub const INTRA_L_CCLM: u32 = 82;
pub const INTRA_T_CCLM: u32 = 83;

/// CABAC context bundle used by [`LeafCuReader`].
pub struct LeafCuCtxs {
    pub intra_bdpcm_luma_flag: Vec<ContextModel>,
    pub intra_bdpcm_luma_dir_flag: Vec<ContextModel>,
    pub intra_bdpcm_chroma_flag: Vec<ContextModel>,
    pub intra_bdpcm_chroma_dir_flag: Vec<ContextModel>,
    pub intra_mip_flag: Vec<ContextModel>,
    pub intra_luma_ref_idx: Vec<ContextModel>,
    pub intra_subpartitions_mode_flag: Vec<ContextModel>,
    pub intra_subpartitions_split_flag: Vec<ContextModel>,
    pub intra_luma_mpm_flag: Vec<ContextModel>,
    pub intra_luma_not_planar_flag: Vec<ContextModel>,
    pub intra_chroma_pred_mode: Vec<ContextModel>,
    /// `cu_skip_flag` (Table 64) — full 9-entry bundle. The per-slice
    /// `init_type` selects the slot used at parse time: `init_type * 3
    /// + ctxInc`.
    pub cu_skip_flag: Vec<ContextModel>,
    /// `general_merge_flag` (Table 82) — 3-entry, indexed by
    /// `init_type`.
    pub general_merge_flag: Vec<ContextModel>,
    /// `regular_merge_flag` (Table 102) — 4 entries spread across the
    /// non-I initTypes (slots 0-1 for P, 2-3 for B). Indexed at parse
    /// time as `(init_type - 1) * 2 + ctxInc` (init_type 1 / 2 only).
    pub regular_merge_flag: Vec<ContextModel>,
    /// `merge_idx` (Table 109) — 3-entry, indexed by `init_type`.
    pub merge_idx: Vec<ContextModel>,
    /// `abs_mvd_greater0_flag` (Table 110) — 3-entry, one per initType.
    /// Indexed at parse time by `init_type`. Single ctx-coded bin (FL
    /// `cMax = 1`) per Table 132 with `ctxInc = 0`; read once per
    /// component inside `mvd_coding()` §7.3.10.10.
    pub abs_mvd_greater0_flag: Vec<ContextModel>,
    /// `abs_mvd_greater1_flag` (Table 111) — 3-entry, one per initType.
    /// Same shape as `abs_mvd_greater0_flag`.
    pub abs_mvd_greater1_flag: Vec<ContextModel>,
    /// `mmvd_merge_flag` (Table 103) — 2-entry, one per non-I
    /// initType. Indexed at parse time as `(init_type - 1)` (init_type
    /// 1 / 2 only — MMVD is never signalled in I slices).
    pub mmvd_merge_flag: Vec<ContextModel>,
    /// `mmvd_cand_flag` (Table 104) — 2-entry, one per non-I initType.
    /// Indexed as `(init_type - 1)`.
    pub mmvd_cand_flag: Vec<ContextModel>,
    /// `mmvd_distance_idx` (Table 105) — 2-entry, one per non-I
    /// initType. Indexed as `(init_type - 1)`. Only the first bin is
    /// ctx-coded; remaining TR bins (cMax = 7) are bypass.
    pub mmvd_distance_idx: Vec<ContextModel>,
    /// `ciip_flag` (Table 106) — 2-entry, one per non-I initType.
    /// Indexed at parse time as `(init_type - 1)`. Single ctx-coded
    /// bin (FL `cMax = 1`) per Table 132.
    pub ciip_flag: Vec<ContextModel>,
    /// `cu_coded_flag` (Table 92) — 3-entry, one per initType. Used
    /// by non-skip merge / non-merge inter CUs to gate the
    /// `transform_tree()` body. Single ctx-coded bin (FL `cMax = 1`)
    /// per Table 132.
    pub cu_coded_flag: Vec<ContextModel>,
    /// `inter_pred_idc` (Table 83) — 12 entries, 6 per non-I initType.
    /// Indexed at parse time as `(init_type - 1) * 6 + ctxInc` (the
    /// per-bin ctxInc is derived from cbWidth / cbHeight per §9.3.3.9).
    /// Only signalled for B slices.
    pub inter_pred_idc: Vec<ContextModel>,
    /// `sym_mvd_flag` (Table 86) — 2 entries, one per non-I initType.
    /// Indexed as `(init_type - 1)`. Single ctx-coded bin (FL
    /// `cMax = 1`).
    pub sym_mvd_flag: Vec<ContextModel>,
    /// `ref_idx_l0` / `ref_idx_l1` (Table 87) — 4 entries, 2 per non-I
    /// initType. Indexed as `(init_type - 1) * 2 + ctxInc` with ctxInc
    /// ∈ {0, 1} for the first two TR bins (bins 2.. bypass).
    pub ref_idx_lx: Vec<ContextModel>,
    /// `mvp_l0_flag` / `mvp_l1_flag` (Table 88) — 3 entries, one per
    /// initType. Indexed by `init_type`. Single ctx-coded bin (FL
    /// `cMax = 1`).
    pub mvp_lx_flag: Vec<ContextModel>,
    /// `bcw_idx` (Table 91) — 2 entries, one per non-I initType.
    /// Indexed at parse time as `(init_type - 1)`. Bin 0 of the TR
    /// sequence is the only context-coded bin (`ctxInc = 0`); the
    /// remaining bins (the syntax element's TR `cMax = NoBackwardPredFlag
    /// ? 4 : 2`) are bypass-coded per Table 132. Only signalled in the
    /// `coding_unit()` else-branch behind the §7.3.10.5 gate
    /// `sps_bcw_enabled_flag && inter_pred_idc == PRED_BI &&
    ///  luma_weight_lX_flag[refIdxLX] == 0 (X = 0, 1) &&
    ///  chroma_weight_lX_flag[refIdxLX] == 0 (X = 0, 1) &&
    ///  cbWidth * cbHeight >= 256`.
    pub bcw_idx: Vec<ContextModel>,
    /// `merge_subblock_flag` (Table 107) — 6 entries split as 3 ctx
    /// slots per non-I initType (initType 1 → slots 0..2, initType 2 →
    /// slots 3..5). Indexed at parse time as `(init_type - 1) * 3 +
    /// ctxInc` with the ctxInc derived via [`ctx_inc_merge_subblock_flag`]
    /// per §9.3.4.2.2 / Table 133. Single ctx-coded bin (FL `cMax = 1`)
    /// gated by §7.3.11.7's `MaxNumSubblockMergeCand > 0 && cbW >= 8 &&
    /// cbH >= 8` size check; merge data is never signalled in I slices.
    pub merge_subblock_flag: Vec<ContextModel>,
    /// `merge_subblock_idx` (Table 108) — 2 entries, one per non-I
    /// initType. Indexed at parse time as `init_type - 1`. Only bin 0
    /// of the TR sequence is context-coded (`ctxInc = 0` per Table
    /// 132); bins 1.. (when `MaxNumSubblockMergeCand ≥ 3`) are
    /// bypass-coded. The TR `cMax = MaxNumSubblockMergeCand − 1`,
    /// `cRiceParam = 0`; only emitted when `merge_subblock_flag == 1
    /// && MaxNumSubblockMergeCand > 1`.
    pub merge_subblock_idx: Vec<ContextModel>,
    /// Round-152 — `inter_affine_flag` (Table 84) — 6 entries split as
    /// 3 ctx slots per non-I initType (initType 1 → slots 0..2,
    /// initType 2 → slots 3..5). Indexed at parse time as
    /// `(init_type - 1) * 3 + ctxInc` with the ctxInc derived via
    /// [`ctx_inc_inter_affine_flag`] per §9.3.4.2.2 / Table 133 (whose
    /// `condL` / `condA` predicates are identical to the
    /// `merge_subblock_flag` row). Single ctx-coded bin (FL `cMax = 1`)
    /// gated by §7.3.11.7's `sps_affine_enabled_flag && cbWidth >= 16
    /// && cbHeight >= 16`; never signalled in I slices nor for the
    /// `general_merge_flag == 1` branch.
    pub inter_affine_flag: Vec<ContextModel>,
    /// Slice initialisation type (§9.3.2.2 / Table 51) — 0 for I,
    /// 1 / 2 for P / B based on `sh_cabac_init_flag`. Used by the
    /// inter-syntax reads to pick the right slot inside the
    /// per-Table CABAC bundles.
    pub init_type: u8,
    /// CBF + residual + last-sig-coeff context arrays. Shared with
    /// [`crate::residual`] so the leaf reader + the residual reader
    /// advance the same CABAC state machine.
    pub residual: ResidualCtxs,
}

impl std::fmt::Debug for LeafCuCtxs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LeafCuCtxs")
            .field("intra_mip_flag.len", &self.intra_mip_flag.len())
            .field("intra_luma_ref_idx.len", &self.intra_luma_ref_idx.len())
            .field(
                "intra_subpartitions_mode_flag.len",
                &self.intra_subpartitions_mode_flag.len(),
            )
            .field(
                "intra_subpartitions_split_flag.len",
                &self.intra_subpartitions_split_flag.len(),
            )
            .field("intra_luma_mpm_flag.len", &self.intra_luma_mpm_flag.len())
            .field(
                "intra_luma_not_planar_flag.len",
                &self.intra_luma_not_planar_flag.len(),
            )
            .field(
                "intra_chroma_pred_mode.len",
                &self.intra_chroma_pred_mode.len(),
            )
            .field("residual.tu_y_coded.len", &self.residual.tu_y_coded.len())
            .finish()
    }
}

impl LeafCuCtxs {
    /// Build all context arrays using the supplied SliceQpY. Defaults
    /// to `init_type = 0` (I-slice). For P/B-slices the caller must
    /// invoke [`Self::init_with_init_type`] so the inter-syntax reads
    /// land in the correct row of Tables 64 / 82 / 102 / 109.
    pub fn init(slice_qp_y: i32) -> Self {
        Self::init_with_init_type(slice_qp_y, 0)
    }

    /// Build all context arrays using the supplied SliceQpY and the
    /// per-slice `init_type` (§9.3.2.2 / Table 51 — 0 for I, 1 / 2 for
    /// P / B based on `sh_cabac_init_flag`).
    pub fn init_with_init_type(slice_qp_y: i32, init_type: u8) -> Self {
        Self {
            intra_bdpcm_luma_flag: init_contexts(SyntaxCtx::IntraBdpcmLumaFlag, slice_qp_y),
            intra_bdpcm_luma_dir_flag: init_contexts(SyntaxCtx::IntraBdpcmLumaDirFlag, slice_qp_y),
            intra_bdpcm_chroma_flag: init_contexts(SyntaxCtx::IntraBdpcmChromaFlag, slice_qp_y),
            intra_bdpcm_chroma_dir_flag: init_contexts(
                SyntaxCtx::IntraBdpcmChromaDirFlag,
                slice_qp_y,
            ),
            intra_mip_flag: init_contexts(SyntaxCtx::IntraMipFlag, slice_qp_y),
            intra_luma_ref_idx: init_contexts(SyntaxCtx::IntraLumaRefIdx, slice_qp_y),
            intra_subpartitions_mode_flag: init_contexts(
                SyntaxCtx::IntraSubpartitionsModeFlag,
                slice_qp_y,
            ),
            intra_subpartitions_split_flag: init_contexts(
                SyntaxCtx::IntraSubpartitionsSplitFlag,
                slice_qp_y,
            ),
            intra_luma_mpm_flag: init_contexts(SyntaxCtx::IntraLumaMpmFlag, slice_qp_y),
            intra_luma_not_planar_flag: init_contexts(
                SyntaxCtx::IntraLumaNotPlanarFlag,
                slice_qp_y,
            ),
            intra_chroma_pred_mode: init_contexts(SyntaxCtx::IntraChromaPredMode, slice_qp_y),
            cu_skip_flag: init_contexts(SyntaxCtx::CuSkipFlag, slice_qp_y),
            general_merge_flag: init_contexts(SyntaxCtx::GeneralMergeFlag, slice_qp_y),
            regular_merge_flag: init_contexts(SyntaxCtx::RegularMergeFlag, slice_qp_y),
            merge_idx: init_contexts(SyntaxCtx::MergeIdx, slice_qp_y),
            abs_mvd_greater0_flag: init_contexts(SyntaxCtx::AbsMvdGreater0Flag, slice_qp_y),
            abs_mvd_greater1_flag: init_contexts(SyntaxCtx::AbsMvdGreater1Flag, slice_qp_y),
            mmvd_merge_flag: init_contexts(SyntaxCtx::MmvdMergeFlag, slice_qp_y),
            mmvd_cand_flag: init_contexts(SyntaxCtx::MmvdCandFlag, slice_qp_y),
            mmvd_distance_idx: init_contexts(SyntaxCtx::MmvdDistanceIdx, slice_qp_y),
            ciip_flag: init_contexts(SyntaxCtx::CiipFlag, slice_qp_y),
            cu_coded_flag: init_contexts(SyntaxCtx::CuCodedFlag, slice_qp_y),
            inter_pred_idc: init_contexts(SyntaxCtx::InterPredIdc, slice_qp_y),
            sym_mvd_flag: init_contexts(SyntaxCtx::SymMvdFlag, slice_qp_y),
            ref_idx_lx: init_contexts(SyntaxCtx::RefIdxLx, slice_qp_y),
            mvp_lx_flag: init_contexts(SyntaxCtx::MvpLxFlag, slice_qp_y),
            bcw_idx: init_contexts(SyntaxCtx::BcwIdx, slice_qp_y),
            merge_subblock_flag: init_contexts(SyntaxCtx::MergeSubblockFlag, slice_qp_y),
            merge_subblock_idx: init_contexts(SyntaxCtx::MergeSubblockIdx, slice_qp_y),
            inter_affine_flag: init_contexts(SyntaxCtx::InterAffineFlag, slice_qp_y),
            init_type,
            residual: ResidualCtxs::init(slice_qp_y),
        }
    }
}

/// Neighbour state used by §8.4.2 (MPM derivation) and the `intra_mip`
/// context derivation. One of these is held per leaf CU that has been
/// parsed so far; the reader looks up the immediate left / above
/// neighbour entries by `(x0, y0)`.
///
/// Off-picture / off-slice neighbours are encoded as `None`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct IntraNeighbour {
    /// `IntraPredModeY[x][y]` of the neighbour (0..66). Only valid
    /// when `pred_mode == Intra` and `mip == false`.
    pub intra_pred_mode_y: u32,
    /// `IntraMipFlag[x][y]`.
    pub mip: bool,
    /// `CuPredMode[0][x][y]`.
    pub pred_mode: Option<CuPredMode>,
}

/// Derive §8.4.2's `candIntraPredModeX` for a given side.
///
/// * `available_x` — §6.4.4 availability flag.
/// * `neighbour` — the neighbour's intra info (ignored if unavailable).
/// * `side_b` — true when deriving the "above" (B) candidate; used to
///   trigger the CTB-row boundary rule that forces PLANAR when the
///   neighbour sits in a prior CTB row.
/// * `crosses_ctb_row` — spec condition
///   `yCb − 1 < (yCb >> CtbLog2SizeY) << CtbLog2SizeY`, i.e. the above
///   neighbour is in a different CTB row than the current CU.
fn cand_intra_mode_for_side(
    available_x: bool,
    neighbour: Option<IntraNeighbour>,
    side_b: bool,
    crosses_ctb_row: bool,
) -> u32 {
    let Some(n) = neighbour else {
        return INTRA_PLANAR;
    };
    if !available_x {
        return INTRA_PLANAR;
    }
    if n.pred_mode != Some(CuPredMode::Intra) {
        return INTRA_PLANAR;
    }
    if n.mip {
        return INTRA_PLANAR;
    }
    if side_b && crosses_ctb_row {
        return INTRA_PLANAR;
    }
    n.intra_pred_mode_y
}

/// Build the 5-entry `candModeList[]` per §8.4.2. Spec equations 210 –
/// 240, covering every branch of the A == B / one-above-DC / neither-
/// above-DC decision tree.
pub fn build_mpm_cand_list(cand_a: u32, cand_b: u32) -> [u32; 5] {
    let mut list = [INTRA_PLANAR; 5];
    if cand_b == cand_a && cand_a > INTRA_DC {
        // eqs 210–214.
        list[0] = cand_a;
        list[1] = 2 + ((cand_a + 61) % 64);
        list[2] = 2 + ((cand_a.wrapping_sub(1)) % 64);
        list[3] = 2 + ((cand_a + 60) % 64);
        list[4] = 2 + (cand_a % 64);
        return list;
    }
    if cand_a != cand_b && (cand_a > INTRA_DC || cand_b > INTRA_DC) {
        let min_ab = cand_a.min(cand_b);
        let max_ab = cand_a.max(cand_b);
        if cand_a > INTRA_DC && cand_b > INTRA_DC {
            // Both > DC (eqs 217 – 230).
            list[0] = cand_a;
            list[1] = cand_b;
            let diff = max_ab - min_ab;
            if diff == 1 {
                list[2] = 2 + ((min_ab + 61) % 64);
                list[3] = 2 + ((max_ab - 1) % 64);
                list[4] = 2 + ((min_ab + 60) % 64);
            } else if diff >= 62 {
                list[2] = 2 + ((min_ab.wrapping_sub(1)) % 64);
                list[3] = 2 + ((max_ab + 61) % 64);
                list[4] = 2 + (min_ab % 64);
            } else if diff == 2 {
                list[2] = 2 + ((min_ab - 1) % 64);
                list[3] = 2 + ((min_ab + 61) % 64);
                list[4] = 2 + ((max_ab - 1) % 64);
            } else {
                list[2] = 2 + ((min_ab + 61) % 64);
                list[3] = 2 + ((min_ab - 1) % 64);
                list[4] = 2 + ((max_ab + 61) % 64);
            }
        } else {
            // Only one > DC (eqs 231 – 235).
            list[0] = max_ab;
            list[1] = 2 + ((max_ab + 61) % 64);
            list[2] = 2 + ((max_ab - 1) % 64);
            list[3] = 2 + ((max_ab + 60) % 64);
            list[4] = 2 + (max_ab % 64);
        }
        return list;
    }
    // Neither > DC (eqs 236 – 240).
    list[0] = INTRA_DC;
    list[1] = INTRA_ANGULAR50;
    list[2] = INTRA_ANGULAR18;
    list[3] = INTRA_ANGULAR46;
    list[4] = INTRA_ANGULAR54;
    list
}

/// §8.4.2 final-step `IntraPredModeY` derivation.
///
/// * `not_planar_flag == 0` → returns `INTRA_PLANAR`.
/// * `bdpcm == true` → returns `INTRA_ANGULAR50` for vertical direction,
///   `INTRA_ANGULAR18` for horizontal.
/// * MPM matched (`mpm_flag == true`): `candModeList[mpm_idx]`.
/// * MPM remainder path: incremented + offset-skipped value.
pub fn derive_intra_pred_mode_y(
    not_planar_flag: bool,
    bdpcm: bool,
    bdpcm_vertical: bool,
    mpm_flag: bool,
    mpm_idx: u32,
    mpm_remainder: u32,
    cand_list: &[u32; 5],
) -> u32 {
    if !not_planar_flag {
        return INTRA_PLANAR;
    }
    if bdpcm {
        return if bdpcm_vertical {
            INTRA_ANGULAR50
        } else {
            INTRA_ANGULAR18
        };
    }
    if mpm_flag {
        // Spec step 4.a: pick candModeList[mpm_idx].
        return cand_list[(mpm_idx as usize).min(4)];
    }
    // Step 4.b: sort candModeList ascending, then add 1, then skip any
    // candidate <= current to produce the final mode outside the MPM
    // set.
    let mut sorted = *cand_list;
    sorted.sort_unstable();
    let mut mode = mpm_remainder + 1;
    for &c in sorted.iter() {
        if mode >= c {
            mode += 1;
        }
    }
    mode
}

/// §8.4.3 / Table 20 chroma intra-mode derivation for `cclm_mode_flag == 0`.
///
/// Table 20 (lower half) maps `intra_chroma_pred_mode ∈ 0..=4` and a
/// representative `lumaIntraPredMode` to the final `IntraPredModeC`.
/// Entries in the table where the luma mode equals 0 / 50 / 18 / 1
/// re-target to a dedicated "alt" mode; the remainder passes the luma
/// mode through untouched.
///
/// Handles the common SINGLE_TREE + non-CCLM + non-BDPCM +
/// `cu_act_enabled_flag == 0` path used in tests.
pub fn derive_intra_pred_mode_c(intra_chroma_pred_mode: u32, luma_intra_pred_mode: u32) -> u32 {
    // Bottom half of Table 20 (cclm_mode_flag == 0).
    match intra_chroma_pred_mode {
        0 => match luma_intra_pred_mode {
            INTRA_PLANAR => INTRA_ANGULAR66,
            _ => INTRA_PLANAR,
        },
        1 => match luma_intra_pred_mode {
            INTRA_ANGULAR50 => INTRA_ANGULAR66,
            _ => INTRA_ANGULAR50,
        },
        2 => match luma_intra_pred_mode {
            INTRA_ANGULAR18 => INTRA_ANGULAR66,
            _ => INTRA_ANGULAR18,
        },
        3 => match luma_intra_pred_mode {
            INTRA_DC => INTRA_ANGULAR66,
            _ => INTRA_DC,
        },
        4 => luma_intra_pred_mode, // DM_CHROMA — direct luma inherit.
        _ => INTRA_PLANAR,
    }
}

/// INTRA_ANGULAR66 (top-right diagonal) — Table 20's "alt" direct mode.
pub const INTRA_ANGULAR66: u32 = 66;

/// Neighbourhood snapshot passed into [`LeafCuReader`] per CU.
#[derive(Clone, Copy, Debug, Default)]
pub struct CuNeighbourhood {
    pub left_available: bool,
    pub above_available: bool,
    pub left: Option<IntraNeighbour>,
    pub above: Option<IntraNeighbour>,
    /// `CuSkipFlag[xNbL][yNbL]` — used by §9.3.4.2.2 cu_skip_flag
    /// context derivation.
    pub left_cu_skip: bool,
    /// `CuSkipFlag[xNbA][yNbA]`.
    pub above_cu_skip: bool,
    /// Round-149 — `MergeSubblockFlag[xNbL][yNbL]` per §9.3.4.2.2 /
    /// Table 133 merge-side row. Folded with `left_inter_affine` into
    /// the ctxInc input for [`Self::read_merge_subblock_flag`] (the
    /// §7.3.11.7 wire-up). Spec semantics: the per-CB merge-side
    /// neighbour flag — `true` only when the neighbouring CB was itself
    /// decoded with `merge_subblock_flag == 1`.
    pub left_merge_subblock: bool,
    /// Round-149 — `MergeSubblockFlag[xNbA][yNbA]` (above neighbour
    /// mirror of [`Self::left_merge_subblock`]).
    pub above_merge_subblock: bool,
    /// Round-149 — `InterAffineFlag[xNbL][yNbL]` per §9.3.4.2.2 /
    /// Table 133. The encoder-side affine inter (non-merge) path is
    /// not yet parsed by the CTU walker, so for round-149 this stays
    /// `false` for every neighbour; the field is plumbed now so the
    /// §7.3.11.7 wire-up no longer hard-codes the `(false, false)`
    /// default and the eventual affine-inter walker has a one-line
    /// drop-in point.
    pub left_inter_affine: bool,
    /// Round-149 — `InterAffineFlag[xNbA][yNbA]` (above neighbour
    /// mirror of [`Self::left_inter_affine`]).
    pub above_inter_affine: bool,
}

/// Leaf-CU syntax reader. Stateless w.r.t. the spec (each call to
/// [`Self::decode`] consumes the full `coding_unit()` bin stream); the
/// caller feeds CABAC state + CU geometry + SPS tool flags.
pub struct LeafCuReader<'a, 'b> {
    dec: &'a mut ArithDecoder<'b>,
    ctxs: &'a mut LeafCuCtxs,
    tools: CuToolFlags,
}

impl<'a, 'b> LeafCuReader<'a, 'b> {
    pub fn new(
        dec: &'a mut ArithDecoder<'b>,
        ctxs: &'a mut LeafCuCtxs,
        tools: CuToolFlags,
    ) -> Self {
        Self { dec, ctxs, tools }
    }

    /// Decode the `coding_unit()` body for a leaf CU in an I-slice,
    /// single-tree pipeline. `info.x0 / y0 / cb_width / cb_height`
    /// must be populated by the caller; every other field is written
    /// by this method. The residual-level storage (per-plane
    /// coefficient arrays) is filled into `residual` when any CBF
    /// fires; it stays empty when the CU carries no coded residual
    /// (e.g. all CBFs 0 on a zero-stream test fixture).
    pub fn decode(
        &mut self,
        info: &mut LeafCuInfo,
        residual: &mut LeafCuResidual,
        neigh: &CuNeighbourhood,
    ) -> Result<()> {
        info.pred_mode = CuPredMode::Intra; // I-slice default (§7.4.12.2).

        // ---- Round-21 P/B-slice inter path ------------------------------
        //
        // §7.3.11.5 cu_skip_flag is read only outside the
        // `cbWidth==4 && cbHeight==4` (and IBC corner) cases. In an
        // inter slice we always read it for the supported sizes; for
        // 4x4 CUs we infer cu_skip_flag = 0 (no IBC support yet).
        if self.tools.slice_is_inter {
            return self.decode_inter(info, residual, neigh);
        }

        // IBC / palette are not currently walked: the flags are
        // grouped here to surface `Error::Unsupported` the moment the
        // SPS opts into them, rather than silently skipping them.
        if self.tools.ibc {
            return Err(Error::unsupported(
                "h266 leaf CU: IBC (sps_ibc_enabled_flag) not supported by this round",
            ));
        }
        if self.tools.palette {
            return Err(Error::unsupported(
                "h266 leaf CU: palette coding (sps_palette_enabled_flag) not supported",
            ));
        }

        // pred_mode_plt_flag: ignored for now (palette disabled above).
        // cu_act_enabled_flag: only present when sps_act_enabled_flag &&
        // treeType == SINGLE_TREE. We gate it similarly.
        if self.tools.act {
            return Err(Error::unsupported(
                "h266 leaf CU: ACT (sps_act_enabled_flag) not supported",
            ));
        }

        // intra_bdpcm_luma_flag gate: spec condition is
        // "cbWidth <= MaxTsSize && cbHeight <= MaxTsSize".
        let bdpcm_gate = self.tools.bdpcm
            && info.cb_width <= self.tools.max_ts_size
            && info.cb_height <= self.tools.max_ts_size;
        if bdpcm_gate {
            // Table 132 — ctxInc 0; Table 69 init values plumbed via
            // SyntaxCtx::IntraBdpcmLumaFlag.
            let inc = ctx_inc_intra_bdpcm_luma_flag() as usize;
            let n = self.ctxs.intra_bdpcm_luma_flag.len() - 1;
            let bit = self
                .dec
                .decode_decision(&mut self.ctxs.intra_bdpcm_luma_flag[inc.min(n)])?;
            info.intra_bdpcm_luma = bit == 1;
            if info.intra_bdpcm_luma {
                // intra_bdpcm_luma_dir_flag — Table 70.
                let inc = ctx_inc_intra_bdpcm_luma_dir_flag() as usize;
                let n = self.ctxs.intra_bdpcm_luma_dir_flag.len() - 1;
                let bit = self
                    .dec
                    .decode_decision(&mut self.ctxs.intra_bdpcm_luma_dir_flag[inc.min(n)])?;
                info.intra_bdpcm_luma_dir = bit == 1;
                // §8.4.2 / §8.4.3: BdpcmDir == 0 → ANGULAR18 (horizontal),
                // BdpcmDir == 1 → ANGULAR50 (vertical). Chroma is derived
                // independently if intra_bdpcm_chroma_flag fires later;
                // otherwise the §8.4.3 mapping below uses this luma mode.
                info.intra_pred_mode_y = if info.intra_bdpcm_luma_dir {
                    INTRA_ANGULAR50
                } else {
                    INTRA_ANGULAR18
                };
                self.derive_chroma(info);
                return Ok(());
            }
        }

        // intra_mip_flag.
        if self.tools.mip {
            let inc = ctx_inc_intra_mip_flag(
                info.cb_width,
                info.cb_height,
                neigh.left_available,
                neigh.above_available,
                neigh.left.map(|n| n.mip).unwrap_or(false),
                neigh.above.map(|n| n.mip).unwrap_or(false),
            ) as usize;
            let n = self.ctxs.intra_mip_flag.len() - 1;
            let bit = self
                .dec
                .decode_decision(&mut self.ctxs.intra_mip_flag[inc.min(n)])?;
            info.intra_mip_flag = bit == 1;
        }

        if info.intra_mip_flag {
            // intra_mip_transposed_flag — bypass (Table 132).
            info.intra_mip_transposed_flag = self.dec.decode_bypass()? == 1;
            // intra_mip_mode — TB with size-dependent cMax.
            let c_max = Self::mip_mode_cmax(info.cb_width, info.cb_height);
            info.intra_mip_mode = decode_tb(self.dec, c_max)?;
            // MIP has its own mode namespace — we still store 0 for
            // intra_pred_mode_y so callers know to branch on the flag.
            info.intra_pred_mode_y = 0;
            self.derive_chroma(info);
            return Ok(());
        }

        // intra_luma_ref_idx — only when MRL is enabled and y0 % CtbSizeY > 0.
        // TR(cMax=2, cRice=0) is a context-coded truncated-unary up to 2
        // bins. Each bin gets its own ctxInc per
        // [`ctx_inc_intra_luma_ref_idx`].
        if self.tools.mrl && (info.y0 % self.tools.ctb_size_y) > 0 {
            let ctxs = &mut self.ctxs.intra_luma_ref_idx;
            let n = ctxs.len() - 1;
            let mut val = 0u32;
            for bin_idx in 0..2u32 {
                let inc = ctx_inc_intra_luma_ref_idx(bin_idx) as usize;
                let bit = self.dec.decode_decision(&mut ctxs[inc.min(n)])?;
                if bit == 0 {
                    break;
                }
                val += 1;
            }
            info.intra_luma_ref_idx = val;
        }

        // intra_subpartitions_mode_flag — ISP gated by:
        //   sps_isp_enabled_flag && intra_luma_ref_idx == 0 &&
        //   cbWidth <= MaxTbSizeY && cbHeight <= MaxTbSizeY &&
        //   (cbWidth * cbHeight) > (MinTbSizeY * MinTbSizeY).
        let isp_gate = self.tools.isp
            && info.intra_luma_ref_idx == 0
            && info.cb_width <= self.tools.max_tb_size_y
            && info.cb_height <= self.tools.max_tb_size_y
            && info.cb_width * info.cb_height > self.tools.min_tb_size_y * self.tools.min_tb_size_y;
        let mut isp_mode_flag = false;
        if isp_gate {
            let inc = ctx_inc_intra_subpartitions_mode_flag() as usize;
            let n = self.ctxs.intra_subpartitions_mode_flag.len() - 1;
            let bit = self
                .dec
                .decode_decision(&mut self.ctxs.intra_subpartitions_mode_flag[inc.min(n)])?;
            isp_mode_flag = bit == 1;
        }
        if isp_mode_flag {
            let inc = ctx_inc_intra_subpartitions_split_flag() as usize;
            let n = self.ctxs.intra_subpartitions_split_flag.len() - 1;
            let split_bit = self
                .dec
                .decode_decision(&mut self.ctxs.intra_subpartitions_split_flag[inc.min(n)])?;
            info.isp_split = if split_bit == 1 {
                IspSplitType::VerSplit
            } else {
                IspSplitType::HorSplit
            };
        } else {
            info.isp_split = IspSplitType::NoSplit;
        }

        // intra_luma_mpm_flag — only present when intra_luma_ref_idx == 0.
        if info.intra_luma_ref_idx == 0 {
            let inc = ctx_inc_intra_luma_mpm_flag() as usize;
            let bit = self
                .dec
                .decode_decision(&mut self.ctxs.intra_luma_mpm_flag[inc])?;
            info.intra_luma_mpm_flag = bit == 1;
        } else {
            // MRL index > 0 implies the MPM is used (not_planar and mpm_idx).
            info.intra_luma_mpm_flag = true;
        }

        if info.intra_luma_mpm_flag {
            if info.intra_luma_ref_idx == 0 {
                // intra_luma_not_planar_flag.
                let inc = ctx_inc_intra_luma_not_planar_flag(isp_mode_flag) as usize;
                let n = self.ctxs.intra_luma_not_planar_flag.len() - 1;
                let bit = self
                    .dec
                    .decode_decision(&mut self.ctxs.intra_luma_not_planar_flag[inc.min(n)])?;
                info.intra_luma_not_planar_flag = bit == 1;
            } else {
                info.intra_luma_not_planar_flag = true;
            }
            if info.intra_luma_not_planar_flag {
                // intra_luma_mpm_idx — bypass TR(cMax=4).
                info.intra_luma_mpm_idx = decode_tr_bypass(self.dec, 4, 0)?;
            }
        } else {
            // intra_luma_mpm_remainder — bypass TB(cMax=60).
            info.intra_luma_mpm_remainder = decode_tb_bypass(self.dec, 60)?;
        }

        // Derive luma mode per §8.4.2.
        let crosses_ctb = {
            let ctb_row = (info.y0 / self.tools.ctb_size_y) * self.tools.ctb_size_y;
            info.y0 == ctb_row
        };
        let cand_a = cand_intra_mode_for_side(
            neigh.left_available,
            neigh.left,
            /*side_b=*/ false,
            /*crosses_ctb_row=*/ false,
        );
        let cand_b = cand_intra_mode_for_side(
            neigh.above_available,
            neigh.above,
            /*side_b=*/ true,
            crosses_ctb,
        );
        let cand_list = build_mpm_cand_list(cand_a, cand_b);
        info.intra_pred_mode_y = derive_intra_pred_mode_y(
            info.intra_luma_not_planar_flag,
            info.intra_bdpcm_luma,
            info.intra_bdpcm_luma_dir,
            info.intra_luma_mpm_flag,
            info.intra_luma_mpm_idx,
            info.intra_luma_mpm_remainder,
            &cand_list,
        );

        // Chroma — skip for monochrome.
        self.derive_chroma(info);

        // --- transform_unit() reads (§7.3.11.10) ---
        self.decode_transform_unit(info, residual)?;

        Ok(())
    }

    /// Round-21 P/B-slice inter-CU parser. Reads `cu_skip_flag` →
    /// `general_merge_flag` → `merge_data()` (regular-merge subset
    /// only) per §7.3.11.5 + §7.3.11.7. Non-merge inter CUs (which
    /// would carry `mvd_coding()` and friends) surface
    /// `Error::Unsupported` so callers see exactly which construct is
    /// pending — the round-21 decoder pipeline only handles the all-
    /// merge / all-skip case.
    fn decode_inter(
        &mut self,
        info: &mut LeafCuInfo,
        residual: &mut LeafCuResidual,
        neigh: &CuNeighbourhood,
    ) -> Result<()> {
        // §7.3.11.5: cu_skip_flag is signalled when the CU is not
        // 4x4 (and treeType != DUAL_TREE_CHROMA). For 4x4 it is
        // implicitly 0 (the IBC corner only applies when sps_ibc_enabled).
        let cb_is_4x4 = info.cb_width == 4 && info.cb_height == 4;
        let init_type_offset = (self.ctxs.init_type as usize) * 3;
        let cu_skip = if !cb_is_4x4 {
            let inc = ctx_inc_cu_skip_flag(
                neigh.left_available,
                neigh.above_available,
                neigh.left_cu_skip,
                neigh.above_cu_skip,
            ) as usize;
            let n = self.ctxs.cu_skip_flag.len() - 1;
            let bit = self
                .dec
                .decode_decision(&mut self.ctxs.cu_skip_flag[(init_type_offset + inc).min(n)])?;
            bit == 1
        } else {
            false
        };
        info.inter.cu_skip_flag = cu_skip;

        // pred_mode_flag is signalled when cu_skip_flag == 0 in P/B for
        // CUs that are not 4x4 and treeType is single. Round-21 does
        // not yet support non-merge inter CUs (they need MVD coding).
        // Force MODE_INTER. When cu_skip_flag == 1 the CU is also
        // necessarily MODE_INTER (and general_merge_flag is inferred 1).
        info.pred_mode = CuPredMode::Inter;

        // pred_mode_ibc_flag: not parsed (sps_ibc_enabled_flag = false
        // in r21 fixtures).

        // §7.3.11.5 inter else-branch: if !cu_skip_flag, read
        // general_merge_flag. Otherwise infer it to 1.
        let general_merge_flag = if !cu_skip {
            let inc = ctx_inc_general_merge_flag() as usize;
            let n = self.ctxs.general_merge_flag.len() - 1;
            let ctx_idx = (init_type_offset + inc).min(n);
            let bit = self
                .dec
                .decode_decision(&mut self.ctxs.general_merge_flag[ctx_idx])?;
            bit == 1
        } else {
            true
        };
        info.inter.general_merge_flag = general_merge_flag;

        if !general_merge_flag {
            // Non-merge inter CU — needs mvd_coding() + ref_idx + amvr.
            // Out of scope for round-21 (covered by r22+).
            return Err(Error::unsupported(
                "h266 leaf CU inter: non-merge inter CUs (mvd_coding) not supported in round-21 \
                 (only cu_skip / regular-merge are wired)",
            ));
        }

        // ---- merge_data() (§7.3.11.7) -----------------------------------
        //
        // Round-146 prologue: §7.3.11.7 opens with the subblock-merge
        // branch *before* the regular_merge_flag tree. The spec text
        // (V4, 01/2026) is:
        //
        //   if (MaxNumSubblockMergeCand > 0 && cbWidth >= 8 && cbHeight >= 8)
        //       merge_subblock_flag[x0][y0]                    ae(v)
        //   if (merge_subblock_flag[x0][y0] == 1) {
        //       if (MaxNumSubblockMergeCand > 1)
        //           merge_subblock_idx[x0][y0]                 ae(v)
        //   } else { /* regular / MMVD / CIIP / GPM tree */ }
        //
        // §7.4.12.7 inference:
        //   * `merge_subblock_flag` not present → 0 (gate closed → no
        //     subblock-merge candidate, fall through to the regular
        //     branch).
        //   * `merge_subblock_idx` not present → 0 (either
        //     `merge_subblock_flag == 0` OR `MaxNumSubblockMergeCand
        //     <= 1`, in which case the single available subblock
        //     candidate is implicit).
        //   * `regular_merge_flag` inferred to
        //     `general_merge_flag && !merge_subblock_flag` — so
        //     `merge_subblock_flag == 1` short-circuits the entire
        //     downstream regular / MMVD / CIIP / GPM tree.
        //
        // Round-149: neighbour state for the §9.3.4.2.2 / Table 133
        // ctxInc derivation (`cond{L,A} = MergeSubblockFlag[{L,A}] ||
        // InterAffineFlag[{L,A}]`) is now sourced from the per-CB
        // [`CuNeighbourhood::{left,above}_merge_subblock`] +
        // [`CuNeighbourhood::{left,above}_inter_affine`] slots that
        // the [`CtuWalker`] populates from the picture-wide live
        // sub-block-merge / affine-inter grid. The walker writes
        // `merge_subblock_flag` per leaf CU after decode (the affine-
        // inter path is not yet parsed, so `inter_affine_flag` stays
        // 0 across the whole picture — the field is plumbed for the
        // future drop-in). When the §7.3.11.7 size gate is closed,
        // §7.4.12.7 infers `merge_subblock_flag = 0` and we skip the
        // reader entirely.
        let max_sb_merge = self.tools.max_num_subblock_merge_cand;
        let subblock_gate_open = max_sb_merge > 0 && info.cb_width >= 8 && info.cb_height >= 8;
        let merge_subblock_flag = if subblock_gate_open {
            self.read_merge_subblock_flag(
                neigh.left_merge_subblock,
                neigh.left_inter_affine,
                neigh.left_available,
                neigh.above_merge_subblock,
                neigh.above_inter_affine,
                neigh.above_available,
            )?
        } else {
            false
        };
        info.inter.merge_data.merge_subblock_flag = merge_subblock_flag;
        if merge_subblock_flag {
            // §7.3.11.7 — merge_subblock_idx is only emitted when
            // MaxNumSubblockMergeCand > 1 (a single candidate degenerates
            // the TR(cMax = 0) syntax). The reader returns 0 without
            // consuming bits in that case, matching the §7.4.12.7
            // inference.
            let merge_subblock_idx = self.read_merge_subblock_idx(max_sb_merge)?;
            info.inter.merge_data.merge_subblock_idx = merge_subblock_idx;
            // Subblock-merge CUs carry no residual on the regular /
            // skip path: cu_skip_flag may be 0 or 1, but the rest of
            // merge_data() (regular_merge_flag, MMVD, CIIP, GPM, the
            // merge_idx parse) is bypassed because §7.4.12.7 infers
            // `regular_merge_flag = general_merge_flag && !merge_subblock_flag`
            // = 0. The cu_coded_flag handling at the end of decode_inter
            // still runs for non-skip CUs.
            info.tu_y_coded_flag = false;
            info.tu_cb_coded_flag = false;
            info.tu_cr_coded_flag = false;
            let _ = residual;
            if !cu_skip {
                let n = self.ctxs.cu_coded_flag.len() - 1;
                let slot = (self.ctxs.init_type as usize).min(n);
                let cu_coded = self
                    .dec
                    .decode_decision(&mut self.ctxs.cu_coded_flag[slot])?
                    == 1;
                if cu_coded {
                    return Err(Error::unsupported(
                        "h266 leaf CU inter: subblock-merge CU with cu_coded_flag == 1 (residual \
                         transform_tree) not yet supported (round-146 only handles \
                         cu_coded_flag == 0)",
                    ));
                }
            }
            return Ok(());
        }
        // §7.3.11.7 — `merge_subblock_flag == 0` falls through to the
        // regular / MMVD / CIIP / GPM tree (rounds 21 / 27 / 28 / 40).
        //
        // Round-28: §7.3.11.7 `regular_merge_flag` gate now light up
        // when CIIP and / or GPM is enabled in the SPS. The gate per
        // §7.3.11.7 reads (paraphrased):
        //
        //   if cbWidth < 128 && cbHeight < 128 &&
        //      ((sps_ciip_enabled && cu_skip_flag == 0 && cbW*cbH >= 64) ||
        //       (sps_gpm_enabled && B-slice && cbW >= 8 && cbH >= 8 &&
        //         cbW < 8*cbH && cbH < 8*cbW))
        //     → parse regular_merge_flag
        //   otherwise → regular_merge_flag inferred to 1
        //
        // When `regular_merge_flag == 1`, the spec optionally inserts
        // the round-27 §8.5.2.7 MMVD sub-tree (gated by
        // `sps_mmvd_enabled_flag`); when `regular_merge_flag == 0`,
        // round-28 picks the §8.5.6.7 CIIP branch:
        //
        //   if sps_ciip && sps_gpm && B-slice && cu_skip_flag == 0
        //      && cbW >= 8 && cbH >= 8 && cbW < 8*cbH && cbH < 8*cbW
        //      && cbW < 128 && cbH < 128 → parse ciip_flag
        //   otherwise → ciip_flag inferred per §7.4.12.7 (= 1 if the
        //              CIIP gates are met and CIIP is the only enabled
        //              non-regular branch, else 0).
        //   if (ciip_flag && MaxNumMergeCand > 1) → parse merge_idx
        //   if (!ciip_flag) → GPM branch (parse merge_gpm_*) — out of
        //                     scope; surfaces Error::Unsupported.
        //
        // Per §7.4.12.7, when `mmvd_merge_flag == 1`, `merge_idx` is
        // inferred to `mmvd_cand_flag`; otherwise it's inferred to 0.
        // The downstream pipeline reads `mergeCandList[merge_idx]`
        // uniformly regardless of which branch (regular / MMVD / CIIP)
        // produced the index.
        let cb_w = info.cb_width;
        let cb_h = info.cb_height;
        let cb_under_128 = cb_w < 128 && cb_h < 128;
        let ciip_size_ok = (cb_w as u64 * cb_h as u64) >= 64;
        // Pure-CIIP regular_merge_flag gate (P-slice or non-GPM B-slice).
        let ciip_branch_open = self.tools.ciip_enabled && !cu_skip && ciip_size_ok && cb_under_128;
        // §7.3.11.7 GPM gate (round-40): B-slice only, square-ish CU
        // (8 ≤ side < 8 × other_side), under 128, !cu_skip, GPM enabled in
        // SPS with at least 2 GPM merge candidates.
        let gpm_size_ok = cb_w >= 8
            && cb_h >= 8
            && (cb_w as u64) < 8u64 * cb_h as u64
            && (cb_h as u64) < 8u64 * cb_w as u64;
        let gpm_branch_open = self.tools.gpm_enabled
            && self.tools.slice_is_b
            && self.tools.max_num_gpm_merge_cand >= 2
            && !cu_skip
            && cb_under_128
            && gpm_size_ok;

        let regular_merge_flag = if cb_under_128 && (ciip_branch_open || gpm_branch_open) {
            let inc = ctx_inc_regular_merge_flag(cu_skip) as usize;
            // Table 102: 4 entries split as initType 1 → slots 0-1,
            // initType 2 → slots 2-3. Indexed at parse time as
            // `(init_type - 1) * 2 + ctxInc`.
            let init_off = (self.ctxs.init_type as usize).saturating_sub(1) * 2;
            let n = self.ctxs.regular_merge_flag.len() - 1;
            let slot = (init_off + inc).min(n);
            let bit = self
                .dec
                .decode_decision(&mut self.ctxs.regular_merge_flag[slot])?;
            bit == 1
        } else {
            true
        };
        info.inter.merge_data.regular_merge_flag = regular_merge_flag;

        let max_merge = self.tools.max_num_merge_cand;

        if regular_merge_flag {
            let mmvd_merge_flag = if self.tools.mmvd_enabled {
                let n = self.ctxs.mmvd_merge_flag.len();
                // MMVD is only signalled in inter slices; init_type ∈
                // {1, 2} → ctx slot 0 / 1.
                let slot = (self.ctxs.init_type as usize).saturating_sub(1).min(n - 1);
                self.dec
                    .decode_decision(&mut self.ctxs.mmvd_merge_flag[slot])?
                    == 1
            } else {
                false
            };
            info.inter.merge_data.mmvd_merge_flag = mmvd_merge_flag;

            if mmvd_merge_flag {
                // §8.5.2.7 — MMVD sub-tree.
                let mmvd_cand_flag = if max_merge > 1 {
                    let n = self.ctxs.mmvd_cand_flag.len();
                    let slot = (self.ctxs.init_type as usize).saturating_sub(1).min(n - 1);
                    self.dec
                        .decode_decision(&mut self.ctxs.mmvd_cand_flag[slot])?
                } else {
                    0
                };
                info.inter.merge_data.mmvd_cand_flag = mmvd_cand_flag as u32;
                info.inter.merge_data.mmvd_distance_idx = self.read_mmvd_distance_idx()?;
                info.inter.merge_data.mmvd_direction_idx = self.read_mmvd_direction_idx()?;
                // §7.4.12.7 — merge_idx is inferred to mmvd_cand_flag.
                info.inter.merge_data.merge_idx = mmvd_cand_flag as u32;
            } else {
                let merge_idx = if max_merge > 1 {
                    self.read_merge_idx(max_merge)?
                } else {
                    0
                };
                info.inter.merge_data.merge_idx = merge_idx;
            }
        } else {
            // §7.3.11.7 — `regular_merge_flag == 0` branch. Either CIIP
            // or GPM fires. Round-40 lights up the GPM path; CIIP stays
            // wired from round-28. The §7.3.11.7 ciip_flag bin is parsed
            // **only** when both CIIP and GPM gates are open
            // simultaneously (forcing the decoder to disambiguate); when
            // only one of the two is open, §7.4.12.7 infers ciip_flag
            // (= 1 for CIIP-only, = 0 for GPM-only).
            let ciip_parse_gate = ciip_branch_open && gpm_branch_open;
            let ciip_flag = if ciip_parse_gate {
                let n = self.ctxs.ciip_flag.len();
                let slot = (self.ctxs.init_type as usize).saturating_sub(1).min(n - 1);
                self.dec.decode_decision(&mut self.ctxs.ciip_flag[slot])? == 1
            } else {
                // §7.4.12.7 inference: when only the CIIP gate is open
                // → ciip_flag = 1; when only the GPM gate is open
                // → ciip_flag = 0; if neither is open, the
                // regular_merge_flag inference (above) already selected
                // regular_merge = 1 so this branch is unreachable.
                ciip_branch_open && !gpm_branch_open
            };
            info.inter.merge_data.ciip_flag = ciip_flag;

            if ciip_flag {
                // CIIP branch — read merge_idx if MaxNumMergeCand > 1.
                let merge_idx = if max_merge > 1 {
                    self.read_merge_idx(max_merge)?
                } else {
                    0
                };
                info.inter.merge_data.merge_idx = merge_idx;
            } else if gpm_branch_open {
                // §7.3.11.7 GPM branch — read merge_gpm_partition_idx,
                // merge_gpm_idx0, merge_gpm_idx1.
                info.inter.merge_data.gpm_flag = true;
                info.inter.merge_data.gpm_partition_idx = self.read_merge_gpm_partition_idx()?;
                let max_gpm = self.tools.max_num_gpm_merge_cand;
                info.inter.merge_data.gpm_idx0 = self.read_merge_gpm_idx0(max_gpm)?;
                info.inter.merge_data.gpm_idx1 = if max_gpm >= 3 {
                    self.read_merge_gpm_idx1(max_gpm)?
                } else {
                    0
                };
                // §7.4.12.7 — merge_idx is unused in the GPM branch,
                // mergeCandList[m] / [n] are looked up directly.
                info.inter.merge_data.merge_idx = 0;
            } else {
                // Truly neither branch open and regular_merge_flag was
                // explicitly parsed as 0 — spec-illegal in any well-
                // formed VVC stream, but bail cleanly.
                return Err(Error::invalid(
                    "h266 leaf CU inter: regular_merge_flag = 0 with neither CIIP nor GPM \
                     branches open (spec §7.3.11.7 violation)",
                ));
            }
        }

        // Skip CUs (cu_skip_flag == 1) carry no residual. Non-skip
        // merge / CIIP CUs read `cu_coded_flag` to gate the
        // `transform_tree()` body. Round-28 wires that read but does
        // not yet decode an actual `transform_tree()` body — when the
        // flag comes back 1 the reader surfaces Unsupported. The CIIP
        // acceptance fixture pins the cu_coded_flag = 0 path, which
        // matches the spec's eq. 998 application to the
        // (intra+inter)-only prediction.
        info.tu_y_coded_flag = false;
        info.tu_cb_coded_flag = false;
        info.tu_cr_coded_flag = false;
        let _ = residual;
        if !cu_skip {
            // §7.3.11.5 — for a merge CU with cu_skip_flag == 0 the
            // next syntax element is `cu_coded_flag` (Table 92, single
            // ctx bin, ctxInc = 0 per Table 132).
            let n = self.ctxs.cu_coded_flag.len() - 1;
            let slot = (self.ctxs.init_type as usize).min(n);
            let cu_coded = self
                .dec
                .decode_decision(&mut self.ctxs.cu_coded_flag[slot])?
                == 1;
            if cu_coded {
                return Err(Error::unsupported(
                    "h266 leaf CU inter: non-skip merge CUs with cu_coded_flag == 1 (residual \
                     transform_tree) not yet supported (round-28 only handles cu_coded_flag == 0)",
                ));
            }
        }

        Ok(())
    }

    /// Decode `mmvd_distance_idx[x0][y0]` — TR binarisation with
    /// `cMax = 7, cRiceParam = 0`. Bin 0 is ctx-coded (ctx 0 of
    /// `MmvdDistanceIdx` per Table 132); bins 1..6 are bypass-coded.
    /// Returns the decoded index in `0..=7`.
    ///
    /// TR with cRiceParam = 0 behaves like a truncated-unary code over
    /// `[0, 7]`: emit a `1` to advance, terminate with a `0` until you
    /// reach `cMax = 7`, where the decoder exits without consuming a
    /// terminator bin.
    fn read_mmvd_distance_idx(&mut self) -> Result<u32> {
        let cmax = 7u32;
        let n = self.ctxs.mmvd_distance_idx.len();
        let slot = (self.ctxs.init_type as usize).saturating_sub(1).min(n - 1);
        let bit0 = self
            .dec
            .decode_decision(&mut self.ctxs.mmvd_distance_idx[slot])?;
        if bit0 == 0 {
            return Ok(0);
        }
        let mut val = 1u32;
        while val < cmax {
            let bit = self.dec.decode_bypass()?;
            if bit == 0 {
                break;
            }
            val += 1;
        }
        Ok(val)
    }

    /// Decode `mmvd_direction_idx[x0][y0]` — FL binarisation with
    /// `cMax = 3` (Table 132 entry: 2 bypass-coded bins, no ctx).
    /// Returns the decoded index in `0..=3`.
    ///
    /// FL(cMax = 3) emits exactly `ceil(log2(4)) = 2` bins, MSB first,
    /// per §9.3.3.5.
    fn read_mmvd_direction_idx(&mut self) -> Result<u32> {
        let b0 = self.dec.decode_bypass()? as u32;
        let b1 = self.dec.decode_bypass()? as u32;
        Ok((b0 << 1) | b1)
    }

    /// Decode `merge_idx[x0][y0]` per Table 132 — bin 0 context-coded
    /// (ctx 0 of MergeIdx table), bins 1..4 bypass-coded. Truncated-
    /// unary binarisation with `cMax = max_merge - 1`. Returns the
    /// decoded index (0..=cMax).
    fn read_merge_idx(&mut self, max_merge: u32) -> Result<u32> {
        if max_merge < 2 {
            return Ok(0);
        }
        let cmax = max_merge - 1;
        let init_type_offset = self.ctxs.init_type as usize;
        let n = self.ctxs.merge_idx.len() - 1;
        let ctx_idx = init_type_offset.min(n);
        let bit0 = self
            .dec
            .decode_decision(&mut self.ctxs.merge_idx[ctx_idx])?;
        if bit0 == 0 {
            return Ok(0);
        }
        let mut val = 1u32;
        while val < cmax {
            let bit = self.dec.decode_bypass()?;
            if bit == 0 {
                break;
            }
            val += 1;
        }
        Ok(val)
    }

    /// Decode the absolute magnitude `abs_mvd_minus2` per §9.3.3.14 —
    /// the *limited* k-th order Exp-Golomb binarisation (§9.3.3.6) with
    /// `k = 1`, `maxPreExtLen = 15`, `truncSuffixLen = 17`. Returns the
    /// raw `abs_mvd_minus2` value (so the caller adds 2 for the absolute
    /// MVD component). All bins are bypass-coded.
    ///
    /// The decode mirrors the §9.3.3.6 binarisation in reverse: read the
    /// `preExtLen` unary prefix (a run of `1`s, capped at `maxPreExtLen`,
    /// terminated by a `0` unless the cap is reached), pick the suffix
    /// length (`truncSuffixLen` at the cap, else `preExtLen + k`), read
    /// that many fixed-length suffix bits MSB-first, then reconstruct
    /// `symbolVal = suffix + (((1 << preExtLen) − 1) << k)`.
    fn read_abs_mvd_minus2(&mut self) -> Result<u32> {
        const K: u32 = 1;
        const MAX_PRE_EXT_LEN: u32 = 15;
        const TRUNC_SUFFIX_LEN: u32 = 17;

        let mut pre_ext_len = 0u32;
        while pre_ext_len < MAX_PRE_EXT_LEN {
            let bit = self.dec.decode_bypass()?;
            if bit == 0 {
                break;
            }
            pre_ext_len += 1;
        }
        let escape_length = if pre_ext_len == MAX_PRE_EXT_LEN {
            // The cap was hit; the spec emits no terminating `0` and the
            // suffix is the fixed `truncSuffixLen`-bit escape field.
            TRUNC_SUFFIX_LEN
        } else {
            // We already consumed the terminating `0` above when the loop
            // broke on a 0 bin.
            pre_ext_len + K
        };
        let mut suffix = 0u32;
        for _ in 0..escape_length {
            let bit = self.dec.decode_bypass()? as u32;
            suffix = (suffix << 1) | bit;
        }
        // symbolVal = symbolVal_low + ((( 1 << preExtLen ) − 1 ) << k):
        // `suffix` carries the low `escapeLength` bits of `symbolVal`,
        // and the prefix run contributes the `(2^preExtLen − 1) << k`
        // base. (For preExtLen == maxPreExtLen the base uses the same
        // formula — the escape path widens only the suffix.)
        let base = ((1u32 << pre_ext_len) - 1) << K;
        Ok(suffix + base)
    }

    /// Decode one `mvd_coding(x0, y0, refList, cpIdx)` syntax structure
    /// (§7.3.10.10) and return the resulting `lMvd[0..1]` pair packed
    /// into a [`MotionVector`] (`x = lMvd[0]`, `y = lMvd[1]`). These are
    /// the raw, pre-AMVR motion-vector differences in the spec's
    /// 1/16-pel storage convention (the §7.4.11.6 AMVR shift, when
    /// signalled, is applied separately by [`crate::amvr`]).
    ///
    /// Bin order per §7.3.10.10:
    ///   1. `abs_mvd_greater0_flag[0]`, `abs_mvd_greater0_flag[1]`
    ///      (both ctx-coded, Table 110 slot = `init_type`).
    ///   2. for each component whose greater0 flag is 1:
    ///      `abs_mvd_greater1_flag[c]` (ctx-coded, Table 111).
    ///   3. for each component whose greater0 flag is 1:
    ///      `abs_mvd_minus2[c]` (only when greater1 == 1, §9.3.3.14
    ///      bypass EG) then `mvd_sign_flag[c]` (bypass).
    ///
    /// `lMvd[c] = greater0 ? (abs_mvd_minus2 + 2) * (1 − 2*sign) : 0`
    /// (eq. 190), where `abs_mvd_minus2` is inferred to −1 when greater1
    /// is 0 (so the magnitude collapses to 1) and to its decoded value
    /// otherwise.
    ///
    /// Exposed `pub` so the (still-deferred) non-merge inter / affine
    /// AMVP CU paths and the round-103 conformance tests can drive the
    /// shared CABAC engine through one `mvd_coding()` structure without
    /// duplicating the bin sequence.
    pub fn read_mvd_coding(&mut self) -> Result<MotionVector> {
        let init_type = self.ctxs.init_type as usize;
        let g0n = self.ctxs.abs_mvd_greater0_flag.len() - 1;
        let g1n = self.ctxs.abs_mvd_greater1_flag.len() - 1;
        let g0_slot = (init_type + ctx_inc_abs_mvd_greater0_flag() as usize).min(g0n);
        let g1_slot = (init_type + ctx_inc_abs_mvd_greater1_flag() as usize).min(g1n);

        // Step 1: both greater0 flags, components 0 then 1.
        let greater0 = [
            self.dec
                .decode_decision(&mut self.ctxs.abs_mvd_greater0_flag[g0_slot])?
                == 1,
            self.dec
                .decode_decision(&mut self.ctxs.abs_mvd_greater0_flag[g0_slot])?
                == 1,
        ];

        // Step 2: greater1 flag per component that was non-zero (in the
        // spec's component-major order: c0 greater1, then c1 greater1).
        let mut greater1 = [false; 2];
        for c in 0..2 {
            if greater0[c] {
                greater1[c] = self
                    .dec
                    .decode_decision(&mut self.ctxs.abs_mvd_greater1_flag[g1_slot])?
                    == 1;
            }
        }

        // Step 3: per non-zero component, the magnitude tail then sign.
        let mut lmvd = [0i32; 2];
        for c in 0..2 {
            if !greater0[c] {
                // abs_mvd_minus2 inferred −1, sign inferred 0 ⇒ lMvd = 0.
                continue;
            }
            // abs_mvd_minus2 is present only when greater1; otherwise it
            // is inferred to −1 so the absolute magnitude is exactly 1.
            let abs = if greater1[c] {
                self.read_abs_mvd_minus2()? as i32 + 2
            } else {
                1
            };
            let sign = self.dec.decode_bypass()?;
            lmvd[c] = if sign == 1 { -abs } else { abs };
        }

        Ok(MotionVector {
            x: lmvd[0],
            y: lmvd[1],
        })
    }

    /// Decode `inter_pred_idc[x0][y0]` per §9.3.3.9 / Table 131. The
    /// binarisation depends on `cbWidth + cbHeight`:
    ///
    /// * `> 12` — bin 0 distinguishes `PRED_BI` (`1`) from the uni-pred
    ///   pair (`0`); when the uni-pred pair is taken bin 1 picks
    ///   `PRED_L0` (`0`) vs `PRED_L1` (`1`), giving `PRED_L0 = 00`,
    ///   `PRED_L1 = 01`, `PRED_BI = 1`.
    /// * `== 12` — `PRED_BI` is not allowed; a single bin gives
    ///   `PRED_L0 = 0` / `PRED_L1 = 1`.
    ///
    /// Bin 0's ctxInc is the cbWidth / cbHeight expression
    /// (`ctx_inc_inter_pred_idc_bin0`); bin 1's ctxInc is fixed 5
    /// (`ctx_inc_inter_pred_idc_bin1`). Per Table 51 the per-initType
    /// slot block is `(init_type − 1) * 6` (initType 1 → 0..5,
    /// initType 2 → 6..11; the I-slice initType-0 block is unused since
    /// `inter_pred_idc` is only signalled for inter slices, and the
    /// two-bin form only for B slices).
    pub fn read_inter_pred_idc(&mut self, cb_width: u32, cb_height: u32) -> Result<InterPredDir> {
        let block = (self.ctxs.init_type as usize).saturating_sub(1) * 6;
        let n = self.ctxs.inter_pred_idc.len();
        if cb_width + cb_height > 12 {
            let inc0 = ctx_inc_inter_pred_idc_bin0(cb_width, cb_height) as usize;
            let slot0 = (block + inc0).min(n - 1);
            let bin0 = self
                .dec
                .decode_decision(&mut self.ctxs.inter_pred_idc[slot0])?;
            if bin0 == 1 {
                return Ok(InterPredDir::PredBi);
            }
            let inc1 = ctx_inc_inter_pred_idc_bin1() as usize;
            let slot1 = (block + inc1).min(n - 1);
            let bin1 = self
                .dec
                .decode_decision(&mut self.ctxs.inter_pred_idc[slot1])?;
            Ok(InterPredDir::from_value(bin1))
        } else {
            // (cbWidth + cbHeight) == 12: single bin, PRED_BI suppressed.
            let inc0 = ctx_inc_inter_pred_idc_bin0(cb_width, cb_height) as usize;
            let slot0 = (block + inc0).min(n - 1);
            let bin0 = self
                .dec
                .decode_decision(&mut self.ctxs.inter_pred_idc[slot0])?;
            Ok(InterPredDir::from_value(bin0))
        }
    }

    /// Decode `sym_mvd_flag[x0][y0]` per Table 132 — FL `cMax = 1`, a
    /// single ctx-coded bin with `ctxInc = 0`. Per Table 51 the slot is
    /// `init_type - 1` (only signalled in inter slices). Returns the
    /// flag as a `bool`.
    pub fn read_sym_mvd_flag(&mut self) -> Result<bool> {
        let _ = ctx_inc_sym_mvd_flag();
        let slot = (self.ctxs.init_type as usize)
            .saturating_sub(1)
            .min(self.ctxs.sym_mvd_flag.len() - 1);
        let bit = self
            .dec
            .decode_decision(&mut self.ctxs.sym_mvd_flag[slot])?;
        Ok(bit == 1)
    }

    /// Decode `ref_idx_l0[x0][y0]` / `ref_idx_l1[x0][y0]` per Table 127 —
    /// TR binarisation with `cMax = NumRefIdxActive[X] − 1`,
    /// `cRiceParam = 0`. Bin 0 ctxInc = 0, bin 1 ctxInc = 1, bins 2..
    /// bypass-coded (Table 132). Per Table 51 the slot block is
    /// `(init_type - 1) * 2`. `num_ref_idx_active` is `NumRefIdxActive[X]`
    /// (the value of the syntax element, not minus one); a value of 1
    /// (or 0) makes `cMax == 0` so no bins are read and the index is 0.
    pub fn read_ref_idx_lx(&mut self, num_ref_idx_active: u32) -> Result<u32> {
        let cmax = num_ref_idx_active.saturating_sub(1);
        if cmax == 0 {
            return Ok(0);
        }
        let block = (self.ctxs.init_type as usize).saturating_sub(1) * 2;
        let n = self.ctxs.ref_idx_lx.len();
        let mut val = 0u32;
        while val < cmax {
            let inc = ctx_inc_ref_idx_lx(val) as usize;
            let bit = if val < 2 {
                let slot = (block + inc).min(n - 1);
                self.dec.decode_decision(&mut self.ctxs.ref_idx_lx[slot])?
            } else {
                self.dec.decode_bypass()?
            };
            if bit == 0 {
                break;
            }
            val += 1;
        }
        Ok(val)
    }

    /// Decode `mvp_l0_flag[x0][y0]` / `mvp_l1_flag[x0][y0]` per Table 132
    /// — FL `cMax = 1`, a single ctx-coded bin with `ctxInc = 0`. Per
    /// Table 51 the slot is `init_type`. The returned flag is the AMVP
    /// candidate-list index in `0..=1`.
    pub fn read_mvp_lx_flag(&mut self) -> Result<u32> {
        let _ = ctx_inc_mvp_lx_flag();
        let slot = (self.ctxs.init_type as usize).min(self.ctxs.mvp_lx_flag.len() - 1);
        let bit = self.dec.decode_decision(&mut self.ctxs.mvp_lx_flag[slot])?;
        Ok(bit)
    }

    /// Decode `bcw_idx[x0][y0]` per §7.3.10.5 / Table 91 / Table 132.
    ///
    /// TR binarisation (`cMax = NoBackwardPredFlag ? 4 : 2`,
    /// `cRiceParam = 0`): only bin 0 is context-coded against the
    /// Table 91 slot `init_type - 1` (initType 1 → ctxIdx 0,
    /// initType 2 → ctxIdx 1) with `ctxInc = 0`; the remaining TR bins
    /// are bypass-coded. When `cMax == 0` no bins are read and the
    /// value is 0 (the `NoBackwardPredFlag == 0 && cMax == 2` case is
    /// the common B-slice path; with backward prediction absent
    /// `cMax = 4`).
    ///
    /// The caller is responsible for gating this read behind the
    /// §7.3.10.5 conditional (`sps_bcw_enabled_flag &&
    /// inter_pred_idc == PRED_BI && no per-list weighted-prediction
    /// flags set on the chosen reference indices && cbWidth * cbHeight
    /// >= 256`). When the gate is closed `bcw_idx` is inferred 0 per
    /// §7.4.12.5 (the caller skips the read and assigns 0 directly —
    /// the existing per-block default in [`crate::ctu`] and
    /// [`crate::affine_merge`] already pin `bcw_idx == 0` on inferred
    /// paths).
    ///
    /// Returns the decoded `bcw_idx` value in `0..=cMax`. The caller
    /// maps the value into the eq. 981 BCW weight lookup
    /// `bcwWLut[k] = {4, 5, 3, 10, -2}` (see the `inter::bcw_lut`
    /// note in the round-29 work).
    pub fn read_bcw_idx(&mut self, no_backward_pred_flag: bool) -> Result<u32> {
        let cmax = if no_backward_pred_flag { 4u32 } else { 2u32 };
        if cmax == 0 {
            return Ok(0);
        }
        // Bin 0 — context-coded against Table 91, slot
        // `init_type - 1`. Per Table 132 the ctxInc is fixed 0; the
        // `ctx_inc_bcw_idx` helper enforces this via debug_assert.
        let block = (self.ctxs.init_type as usize).saturating_sub(1);
        let n = self.ctxs.bcw_idx.len();
        let inc = ctx_inc_bcw_idx(0) as usize;
        let slot = (block + inc).min(n - 1);
        let bit0 = self.dec.decode_decision(&mut self.ctxs.bcw_idx[slot])?;
        if bit0 == 0 {
            return Ok(0);
        }
        // Bins 1..cMax — all bypass per Table 132. TR with
        // cRiceParam = 0: a `0` terminates the truncated-unary
        // sequence; reaching `cMax - 1` ones implies the value is
        // exactly `cMax` (the truncation point — no trailing zero
        // sent).
        let mut val = 1u32;
        while val < cmax {
            let bit = self.dec.decode_bypass()?;
            if bit == 0 {
                break;
            }
            val += 1;
        }
        Ok(val)
    }

    /// Read `bcw_idx[x0][y0]` *with the §7.3.10.5 gate evaluated for
    /// you*: when [`BcwIdxGate::is_open`] is true the reader is invoked
    /// against the supplied `no_backward_pred_flag` (so the TR uses
    /// `cMax = NoBackwardPredFlag ? 4 : 2`); otherwise the syntax
    /// element is not signalled and the inferred value 0 is returned
    /// (per §7.4.12.5 "When `bcw_idx[ x0 ][ y0 ]` is not present, it is
    /// inferred to be equal to 0").
    ///
    /// This is the fuse the round-126 reader-side note called out: the
    /// CTU walker fills in a [`BcwIdxGate`] from live per-CB state
    /// (`(sps_bcw_enabled, inter_pred_idc, luma_weight_lX,
    /// chroma_weight_lX, cb_w * cb_h)`), this routine decides whether to
    /// pull bins, and the returned value drops straight into
    /// [`MvField::bcw_idx`].
    ///
    /// Returns the decoded (or inferred) `bcw_idx` value. The caller
    /// is responsible for broadcasting the value across every 4x4 block
    /// the CU covers (the existing CTU writer in [`crate::ctu`] does
    /// this for the merge path; the non-merge inter path consumes this
    /// helper).
    pub fn read_bcw_idx_gated(&mut self, gate: BcwIdxGate) -> Result<u32> {
        if !gate.is_open() {
            return Ok(0);
        }
        self.read_bcw_idx(gate.no_backward_pred_flag)
    }

    /// Convenience wrapper around [`Self::read_bcw_idx_gated`] that
    /// also writes the decoded value into the supplied
    /// [`MvField::bcw_idx`] slot in place. Returns the value written
    /// (for assertion / RDO bookkeeping). This is the spec's "set
    /// `BcwIdx[x0][y0] = bcw_idx[x0][y0]`" assignment from §7.4.12.5 +
    /// the §8.5.2.1 final paragraph (the latter pins `bcwIdx = 0` when
    /// the spec's symmetric / parallel-merge collapse fires; that
    /// collapse is the CTU walker's responsibility — this helper only
    /// handles the per-CU read / infer step).
    pub fn read_bcw_idx_into(&mut self, gate: BcwIdxGate, mvf: &mut MvField) -> Result<u32> {
        let v = self.read_bcw_idx_gated(gate)?;
        mvf.bcw_idx = v as u8;
        Ok(v)
    }

    /// Decode `merge_subblock_flag[x0][y0]` per §7.3.11.7 / Table 107 /
    /// Table 132.
    ///
    /// Binarisation: FL `cMax = 1` — a single ctx-coded bin per Table
    /// 132. The ctx-slot is `(init_type - 1) * 3 + ctxInc` with the
    /// ctxInc derived by [`ctx_inc_merge_subblock_flag`] (§9.3.4.2.2 /
    /// eq. 1551 with the Table 133 merge-side row
    /// `cond{L,A} = MergeSubblockFlag[{L,A}] || InterAffineFlag[{L,A}]`).
    /// Returns the decoded flag as a `bool`.
    ///
    /// **The caller is responsible for the §7.3.11.7 size gate**
    /// (`MaxNumSubblockMergeCand > 0 && cbW >= 8 && cbH >= 8`). When
    /// the gate is closed, `merge_subblock_flag` is inferred to 0
    /// per §7.4.12.7 and this reader must NOT be invoked. Likewise the
    /// reader assumes the slice is non-I (initType ∈ {1, 2}); merge
    /// data is never signalled in I slices.
    pub fn read_merge_subblock_flag(
        &mut self,
        left_merge_subblock: bool,
        left_inter_affine: bool,
        left_available: bool,
        above_merge_subblock: bool,
        above_inter_affine: bool,
        above_available: bool,
    ) -> Result<bool> {
        let inc = ctx_inc_merge_subblock_flag(
            left_merge_subblock,
            left_inter_affine,
            left_available,
            above_merge_subblock,
            above_inter_affine,
            above_available,
        ) as usize;
        // Table 107: 6 entries split as initType 1 → slots 0..2, initType
        // 2 → slots 3..5 (per Table 51). Indexed as `(init_type - 1) * 3
        // + ctxInc`.
        let init_off = (self.ctxs.init_type as usize).saturating_sub(1) * 3;
        let n = self.ctxs.merge_subblock_flag.len() - 1;
        let slot = (init_off + inc).min(n);
        let bit = self
            .dec
            .decode_decision(&mut self.ctxs.merge_subblock_flag[slot])?;
        Ok(bit == 1)
    }

    /// Decode `merge_subblock_idx[x0][y0]` per §7.3.11.7 / Table 108 /
    /// Table 132.
    ///
    /// Binarisation: TR with `cMax = MaxNumSubblockMergeCand − 1`,
    /// `cRiceParam = 0`. Bin 0 is ctx-coded against the Table 108 slot
    /// `init_type - 1` (initType 1 → ctxIdx 0, initType 2 → ctxIdx 1)
    /// with `ctxInc = 0` per Table 132; bins 1.. (only present when
    /// `MaxNumSubblockMergeCand ≥ 3`) are bypass-coded.
    ///
    /// `max_num_subblock_merge_cand` is `MaxNumSubblockMergeCand` per
    /// §7.4.3.4 eq. 85 (clipped to `[0, 5]`). §7.3.11.7 only emits this
    /// syntax element when `MaxNumSubblockMergeCand > 1`; with
    /// `max_num_subblock_merge_cand ≤ 1` the reader returns 0 without
    /// consuming a bit (matching the §7.4.12.7 inference). Returns the
    /// decoded sub-block-merge-candidate index in `0..=cMax`.
    pub fn read_merge_subblock_idx(&mut self, max_num_subblock_merge_cand: u32) -> Result<u32> {
        if max_num_subblock_merge_cand <= 1 {
            return Ok(0);
        }
        let cmax = max_num_subblock_merge_cand - 1;
        // Bin 0 — context-coded against Table 108, slot `init_type - 1`.
        // Per Table 132 the ctxInc is fixed 0.
        let _ = ctx_inc_merge_subblock_idx();
        let block = (self.ctxs.init_type as usize).saturating_sub(1);
        let n = self.ctxs.merge_subblock_idx.len();
        let slot = block.min(n - 1);
        let bin0 = self
            .dec
            .decode_decision(&mut self.ctxs.merge_subblock_idx[slot])?;
        if bin0 == 0 {
            return Ok(0);
        }
        // Bypass tail.
        let mut val = 1u32;
        while val < cmax {
            let bit = self.dec.decode_bypass()?;
            if bit == 0 {
                break;
            }
            val += 1;
        }
        Ok(val)
    }

    /// Round-152 — Decode `inter_affine_flag[x0][y0]` per §7.3.11.7 /
    /// Table 84 / Table 132.
    ///
    /// Binarisation: FL `cMax = 1` — a single ctx-coded bin per Table
    /// 132. The ctx-slot is `(init_type - 1) * 3 + ctxInc` with the
    /// ctxInc derived by [`ctx_inc_inter_affine_flag`] (§9.3.4.2.2 /
    /// eq. 1551 with the Table 133 row whose `condL` / `condA` rules
    /// are identical to `merge_subblock_flag`:
    /// `cond{L,A} = MergeSubblockFlag[{L,A}] || InterAffineFlag[{L,A}]`).
    /// Returns the decoded flag as a `bool`.
    ///
    /// **The caller is responsible for the §7.3.11.7 gates**:
    /// `sps_affine_enabled_flag && cbWidth >= 16 && cbHeight >= 16` AND
    /// the surrounding `general_merge_flag == 0` (`inter_affine_flag` is
    /// only parsed on the non-merge inter branch). When any gate is
    /// closed the syntax element is not present and §7.4.12.7 infers
    /// it to 0 — this reader must NOT be invoked. Likewise the reader
    /// assumes the slice is non-I (initType ∈ {1, 2}); affine flags are
    /// never signalled in I slices.
    pub fn read_inter_affine_flag(
        &mut self,
        left_merge_subblock: bool,
        left_inter_affine: bool,
        left_available: bool,
        above_merge_subblock: bool,
        above_inter_affine: bool,
        above_available: bool,
    ) -> Result<bool> {
        let inc = ctx_inc_inter_affine_flag(
            left_merge_subblock,
            left_inter_affine,
            left_available,
            above_merge_subblock,
            above_inter_affine,
            above_available,
        ) as usize;
        // Table 84: 6 entries split as initType 1 → slots 0..2, initType
        // 2 → slots 3..5 (per Table 51). Indexed as `(init_type - 1) * 3
        // + ctxInc`.
        let init_off = (self.ctxs.init_type as usize).saturating_sub(1) * 3;
        let n = self.ctxs.inter_affine_flag.len() - 1;
        let slot = (init_off + inc).min(n);
        let bit = self
            .dec
            .decode_decision(&mut self.ctxs.inter_affine_flag[slot])?;
        Ok(bit == 1)
    }

    /// Decode `merge_gpm_partition_idx[x0][y0]` per Table 132 — FL
    /// binarisation with `cMax = 63` (six bypass-coded bins, MSB first).
    /// Returns the decoded value in `0..=63`.
    fn read_merge_gpm_partition_idx(&mut self) -> Result<u32> {
        let mut val = 0u32;
        for _ in 0..6 {
            let bit = self.dec.decode_bypass()? as u32;
            val = (val << 1) | bit;
        }
        Ok(val)
    }

    /// Decode `merge_gpm_idx0[x0][y0]` per Table 132 — TR binarisation
    /// with `cMax = MaxNumGpmMergeCand − 1` and `cRiceParam = 0`. Bin 0
    /// is ctx-coded against the Table 109 `merge_idx` slot (initType-
    /// indexed); subsequent bins are bypass-coded. Returns the decoded
    /// index in `0..=cMax`.
    fn read_merge_gpm_idx0(&mut self, max_gpm: u32) -> Result<u32> {
        if max_gpm < 2 {
            return Ok(0);
        }
        let cmax = max_gpm - 1;
        let init_type_offset = self.ctxs.init_type as usize;
        let n = self.ctxs.merge_idx.len() - 1;
        let ctx_idx = init_type_offset.min(n);
        let bit0 = self
            .dec
            .decode_decision(&mut self.ctxs.merge_idx[ctx_idx])?;
        if bit0 == 0 {
            return Ok(0);
        }
        let mut val = 1u32;
        while val < cmax {
            let bit = self.dec.decode_bypass()?;
            if bit == 0 {
                break;
            }
            val += 1;
        }
        Ok(val)
    }

    /// Decode `merge_gpm_idx1[x0][y0]` per Table 132 — TR binarisation
    /// with `cMax = MaxNumGpmMergeCand − 2` and `cRiceParam = 0`. Bin 0
    /// is ctx-coded against the same Table 109 slot as
    /// `merge_gpm_idx0`; subsequent bins are bypass-coded. Returns the
    /// decoded index in `0..=cMax`. The §8.5.4.2 eq. 647 increment past
    /// `merge_gpm_idx0` is applied later by [`crate::gpm::derive_gpm_mn`].
    fn read_merge_gpm_idx1(&mut self, max_gpm: u32) -> Result<u32> {
        if max_gpm < 3 {
            return Ok(0);
        }
        let cmax = max_gpm - 2;
        let init_type_offset = self.ctxs.init_type as usize;
        let n = self.ctxs.merge_idx.len() - 1;
        let ctx_idx = init_type_offset.min(n);
        let bit0 = self
            .dec
            .decode_decision(&mut self.ctxs.merge_idx[ctx_idx])?;
        if bit0 == 0 {
            return Ok(0);
        }
        let mut val = 1u32;
        while val < cmax {
            let bit = self.dec.decode_bypass()?;
            if bit == 0 {
                break;
            }
            val += 1;
        }
        Ok(val)
    }

    /// Read the transform_unit()-level syntax (CBFs, cu_qp_delta,
    /// cu_chroma_qp_offset_*) and drive the residual walker.
    ///
    /// Scope: single-TB CU and ISP-split CU (no SBT). The CBF read
    /// gates per §7.3.11.10 reduce to:
    ///   * chroma CBFs are always read when the chroma format is not
    ///     monochrome and the CU is in single-tree mode. For ISP CUs
    ///     they are read **only on the last** subpartition (subTuIndex
    ///     == NumIntraSubPartitions − 1).
    ///   * luma CBF is always read in the intra path. For ISP CUs
    ///     `tu_y_coded_flag` is read per subpartition; the last
    ///     partition's CBF is inferred to 1 when every prior
    ///     partition's CBF was 0 (§7.3.11.10's `InferTuCbfLuma`).
    ///
    /// The residual decode path surfaces `Error::Unsupported` if
    /// `sps_joint_cbcr_enabled_flag` is set and the joint-cbcr flag
    /// would be read — that binarisation has a separate context init
    /// that is not plumbed yet.
    fn decode_transform_unit(
        &mut self,
        info: &mut LeafCuInfo,
        residual: &mut LeafCuResidual,
    ) -> Result<()> {
        if info.isp_split != IspSplitType::NoSplit {
            return self.decode_transform_unit_isp(info, residual);
        }
        let cb_w = info.cb_width as usize;
        let cb_h = info.cb_height as usize;
        let chroma = self.tools.chroma_format_idc != 0;
        // Chroma dims: 4:2:0 halves both, 4:2:2 halves width only, 4:4:4 keeps.
        // We only support 4:2:0 here (chroma_format_idc == 1); other formats
        // surface Unsupported the moment the code would diverge.
        let (sub_w, sub_h) = match self.tools.chroma_format_idc {
            0 => (1, 1),
            1 => (2, 2),
            2 => (2, 1),
            3 => (1, 1),
            _ => {
                return Err(Error::invalid(
                    "h266 leaf CU: unknown sps_chroma_format_idc value",
                ));
            }
        };
        // Read tu_cb/tu_cr_coded first (spec order — chroma CBFs come
        // before luma within transform_unit()).
        if chroma {
            info.tu_cb_coded_flag = read_tu_cb_coded_flag(
                self.dec,
                &mut self.ctxs.residual,
                /*bdpcm_chroma=*/ false,
            )?;
            info.tu_cr_coded_flag = read_tu_cr_coded_flag(
                self.dec,
                &mut self.ctxs.residual,
                /*bdpcm_chroma=*/ false,
                info.tu_cb_coded_flag,
            )?;
        }
        // Luma CBF: always read in the intra path (per §7.3.11.10 the
        // condition simplifies to true when pred_mode == INTRA, ISP is
        // NO_SPLIT, SBT is off and ACT is off).
        info.tu_y_coded_flag = read_tu_y_coded_flag(
            self.dec,
            &mut self.ctxs.residual,
            /*bdpcm_y=*/ info.intra_bdpcm_luma,
            /*isp_split=*/ info.isp_split != IspSplitType::NoSplit,
            /*prev_tu_cbf_y=*/ false,
        )?;

        // cu_qp_delta_abs + sign (spec gates on CB > 64 || any CBF + enable).
        let any_cbf = info.tu_y_coded_flag || info.tu_cb_coded_flag || info.tu_cr_coded_flag;
        if self.tools.cu_qp_delta_enabled && any_cbf {
            info.cu_qp_delta_val = read_cu_qp_delta(self.dec, &mut self.ctxs.residual)?;
        }
        // cu_chroma_qp_offset_flag + idx.
        let chroma_cbf = info.tu_cb_coded_flag || info.tu_cr_coded_flag;
        if self.tools.cu_chroma_qp_offset_enabled && chroma && chroma_cbf {
            let (flag, idx) = read_cu_chroma_qp_offset(
                self.dec,
                &mut self.ctxs.residual,
                self.tools.chroma_qp_offset_list_len_minus1,
            )?;
            info.cu_chroma_qp_offset_flag = flag;
            info.cu_chroma_qp_offset_idx = idx;
        }
        // joint_cbcr_residual_flag — gate exists but parsing not
        // plumbed; surface Unsupported when it would actually fire.
        if self.tools.joint_cbcr_enabled
            && chroma
            && ((info.pred_mode == CuPredMode::Intra && chroma_cbf)
                || (info.tu_cb_coded_flag && info.tu_cr_coded_flag))
        {
            return Err(Error::unsupported(
                "h266 leaf CU: tu_joint_cbcr_residual_flag parsing not plumbed yet",
            ));
        }

        // Residual decode per plane.
        if info.tu_y_coded_flag {
            let levels = decode_tb_coefficients(self.dec, &mut self.ctxs.residual, cb_w, cb_h, 0)?;
            // Capture the last-sig position derived by the residual
            // reader by re-reading the info from the coefficient
            // array: the last non-zero in scan order is the last-sig.
            // (The residual reader already reads last_sig_coeff_*; we
            // expose a simplified record here.)
            // Find max (x, y) with non-zero level as a best-effort
            // proxy for LastSignificantCoeffX/Y.
            let mut lx = 0u32;
            let mut ly = 0u32;
            for y in 0..cb_h {
                for x in 0..cb_w {
                    if levels[y * cb_w + x] != 0 {
                        lx = lx.max(x as u32);
                        ly = ly.max(y as u32);
                    }
                }
            }
            info.last_sig_x = lx;
            info.last_sig_y = ly;
            residual.luma_levels = levels;
        }
        if info.tu_cb_coded_flag && chroma {
            let cw = cb_w / sub_w;
            let ch = cb_h / sub_h;
            if cw >= 2 && ch >= 2 {
                residual.cb_levels =
                    decode_tb_coefficients(self.dec, &mut self.ctxs.residual, cw, ch, 1)?;
            }
        }
        if info.tu_cr_coded_flag && chroma {
            let cw = cb_w / sub_w;
            let ch = cb_h / sub_h;
            if cw >= 2 && ch >= 2 {
                residual.cr_levels =
                    decode_tb_coefficients(self.dec, &mut self.ctxs.residual, cw, ch, 2)?;
            }
        }
        Ok(())
    }

    /// `transform_tree() + transform_unit()` walk for an ISP-split CU
    /// (§7.3.11.9 + §7.3.11.10). Each subpartition issues its own
    /// `transform_unit()` call with `subTuIndex = partIdx`; chroma
    /// CBFs and chroma residuals are read only on the last
    /// subpartition. The luma CBF for the last subpartition is
    /// **inferred** to 1 when every prior subpartition reported
    /// `tu_y_coded_flag == 0` (the spec's `InferTuCbfLuma`).
    ///
    /// This routine does *not* perform `cu_sbt_*` reads — ISP and SBT
    /// are mutually exclusive (§7.3.11.5) and the leaf-CU path here
    /// always has `cu_sbt_flag == 0`.
    fn decode_transform_unit_isp(
        &mut self,
        info: &mut LeafCuInfo,
        residual: &mut LeafCuResidual,
    ) -> Result<()> {
        let cb_w = info.cb_width;
        let cb_h = info.cb_height;
        let chroma = self.tools.chroma_format_idc != 0;
        let (sub_w, sub_h) = match self.tools.chroma_format_idc {
            0 => (1u32, 1u32),
            1 => (2, 2),
            2 => (2, 1),
            3 => (1, 1),
            _ => {
                return Err(Error::invalid(
                    "h266 leaf CU ISP: unknown sps_chroma_format_idc value",
                ));
            }
        };
        let parts = crate::isp::iter_isp_partitions(info.isp_split, cb_w, cb_h);
        if parts.is_empty() {
            return Err(Error::invalid(
                "h266 leaf CU ISP: subpartition walk produced no entries",
            ));
        }
        let num_parts = parts.len();
        residual.luma_subparts.clear();
        residual.luma_subparts.reserve(num_parts);

        // Walk subpartitions in order. We accumulate the per-part
        // CBFs first to honour spec read ordering — chroma CBFs come
        // *before* luma CBFs only on the last partition.
        let mut infer_tu_cbf_luma = true;
        let mut last_tu_cbf_y = false;
        // Per-partition luma CBFs we read along the way. Chroma CBFs
        // are deferred until the final partition.
        let mut part_cbfs: Vec<bool> = Vec::with_capacity(num_parts);
        // We also collect the full per-partition records so the
        // residual decode (which interleaves with CBF reads in spec
        // order) can be invoked at the right moment.
        let mut sub_records: Vec<LeafCuLumaSubpart> = Vec::with_capacity(num_parts);

        for (i, p) in parts.iter().enumerate() {
            let is_last = i == num_parts - 1;

            // Spec §7.3.11.10: chroma CBFs are read only on the last
            // subpartition (and only for SINGLE_TREE / DUAL_TREE_CHROMA
            // with chroma format != monochrome).
            if is_last && chroma {
                info.tu_cb_coded_flag = read_tu_cb_coded_flag(
                    self.dec,
                    &mut self.ctxs.residual,
                    /*bdpcm_chroma=*/ false,
                )?;
                info.tu_cr_coded_flag = read_tu_cr_coded_flag(
                    self.dec,
                    &mut self.ctxs.residual,
                    /*bdpcm_chroma=*/ false,
                    info.tu_cb_coded_flag,
                )?;
            }

            // Spec §7.3.11.10: tu_y_coded_flag read condition for ISP
            // simplifies to `subTuIndex < NumIntraSubPartitions - 1
            // || !InferTuCbfLuma`. When neither is true (last
            // partition, all priors zero) the flag is inferred to 1.
            let read_y_cbf = !is_last || !infer_tu_cbf_luma;
            let cbf_y = if read_y_cbf {
                read_tu_y_coded_flag(
                    self.dec,
                    &mut self.ctxs.residual,
                    /*bdpcm_y=*/ info.intra_bdpcm_luma,
                    /*isp_split=*/ true,
                    /*prev_tu_cbf_y=*/ last_tu_cbf_y,
                )?
            } else {
                true
            };
            last_tu_cbf_y = cbf_y;
            // §7.3.11.10: InferTuCbfLuma stays true only while every
            // partition has reported zero.
            infer_tu_cbf_luma = infer_tu_cbf_luma && !cbf_y;
            part_cbfs.push(cbf_y);

            // The cu_qp_delta / cu_chroma_qp_offset reads are gated
            // on the same conditions as the single-TB path; we read
            // them on the first partition where the CBFs (or any of
            // the size escapes) make them available, which for ISP
            // means: as soon as we have a CBF == 1 in luma, or once
            // chroma CBFs are read (last partition).
            if i == num_parts - 1 {
                // Defer to the trailing pass below.
            }

            // Decode this partition's luma residual now (spec orders
            // residual_coding *after* the CBF reads of *this*
            // partition's transform_unit; chroma CBFs only appear in
            // the last partition's TU and the per-partition luma
            // residual lives inside the same TU body).
            let n_w = p.n_w as usize;
            let n_h = p.n_h as usize;
            let levels = if cbf_y {
                decode_tb_coefficients(self.dec, &mut self.ctxs.residual, n_w, n_h, 0)?
            } else {
                Vec::new()
            };
            sub_records.push(LeafCuLumaSubpart {
                n_w: p.n_w,
                n_h: p.n_h,
                x_offset: p.x_offset,
                y_offset: p.y_offset,
                tu_y_coded_flag: cbf_y,
                levels,
            });
        }

        // The combined `tu_y_coded_flag` flag captured into `LeafCuInfo`
        // is the OR across partitions: the deblocker / chroma-CBF
        // logic only cares about "did this CU produce any luma
        // residual".
        info.tu_y_coded_flag = part_cbfs.iter().any(|&b| b);

        // Trailing per-CU reads (cu_qp_delta_*, cu_chroma_qp_offset_*,
        // joint_cbcr) — these are only signalled once per CU.
        let any_cbf = info.tu_y_coded_flag || info.tu_cb_coded_flag || info.tu_cr_coded_flag;
        if self.tools.cu_qp_delta_enabled && any_cbf {
            info.cu_qp_delta_val = read_cu_qp_delta(self.dec, &mut self.ctxs.residual)?;
        }
        let chroma_cbf = info.tu_cb_coded_flag || info.tu_cr_coded_flag;
        if self.tools.cu_chroma_qp_offset_enabled && chroma && chroma_cbf {
            let (flag, idx) = read_cu_chroma_qp_offset(
                self.dec,
                &mut self.ctxs.residual,
                self.tools.chroma_qp_offset_list_len_minus1,
            )?;
            info.cu_chroma_qp_offset_flag = flag;
            info.cu_chroma_qp_offset_idx = idx;
        }
        if self.tools.joint_cbcr_enabled
            && chroma
            && ((info.pred_mode == CuPredMode::Intra && chroma_cbf)
                || (info.tu_cb_coded_flag && info.tu_cr_coded_flag))
        {
            return Err(Error::unsupported(
                "h266 leaf CU ISP: tu_joint_cbcr_residual_flag parsing not plumbed yet",
            ));
        }

        // Chroma residuals (single-pass at the CU level — chroma is
        // not split for ISP per eqs. 251 – 254).
        if info.tu_cb_coded_flag && chroma {
            let cw = (cb_w / sub_w) as usize;
            let ch = (cb_h / sub_h) as usize;
            if cw >= 2 && ch >= 2 {
                residual.cb_levels =
                    decode_tb_coefficients(self.dec, &mut self.ctxs.residual, cw, ch, 1)?;
            }
        }
        if info.tu_cr_coded_flag && chroma {
            let cw = (cb_w / sub_w) as usize;
            let ch = (cb_h / sub_h) as usize;
            if cw >= 2 && ch >= 2 {
                residual.cr_levels =
                    decode_tb_coefficients(self.dec, &mut self.ctxs.residual, cw, ch, 2)?;
            }
        }

        residual.luma_subparts = sub_records;
        Ok(())
    }

    fn derive_chroma(&mut self, info: &mut LeafCuInfo) {
        if self.tools.chroma_format_idc == 0 {
            info.intra_pred_mode_c = 0;
            return;
        }
        // intra_bdpcm_chroma_flag — gated by:
        //   cbW/SubWidthC <= MaxTsSize && cbH/SubHeightC <= MaxTsSize
        //   && sps_bdpcm_enabled_flag && !cu_act_enabled_flag.
        // We approximate SubWidthC/SubHeightC as 2 for 4:2:0 (the only
        // chroma layout currently supported by reconstruct_leaf_cu).
        let (sub_w, sub_h) = match self.tools.chroma_format_idc {
            1 => (2, 2),
            2 => (2, 1),
            _ => (1, 1),
        };
        let bdpcm_chroma_gate = self.tools.bdpcm
            && info.cb_width / sub_w <= self.tools.max_ts_size
            && info.cb_height / sub_h <= self.tools.max_ts_size;
        if bdpcm_chroma_gate {
            let inc = ctx_inc_intra_bdpcm_chroma_flag() as usize;
            let n = self.ctxs.intra_bdpcm_chroma_flag.len() - 1;
            let bit = self
                .dec
                .decode_decision(&mut self.ctxs.intra_bdpcm_chroma_flag[inc.min(n)])
                .unwrap_or(0);
            info.intra_bdpcm_chroma = bit == 1;
            if info.intra_bdpcm_chroma {
                let inc = ctx_inc_intra_bdpcm_chroma_dir_flag() as usize;
                let n = self.ctxs.intra_bdpcm_chroma_dir_flag.len() - 1;
                let bit = self
                    .dec
                    .decode_decision(&mut self.ctxs.intra_bdpcm_chroma_dir_flag[inc.min(n)])
                    .unwrap_or(0);
                info.intra_bdpcm_chroma_dir = bit == 1;
                // Chroma uses the same vertical/horizontal mapping as
                // luma BDPCM (§7.4.5.1 + chroma derivation in §8.4.3).
                info.intra_pred_mode_c = if info.intra_bdpcm_chroma_dir {
                    INTRA_ANGULAR50
                } else {
                    INTRA_ANGULAR18
                };
                return;
            }
        }
        // Read intra_chroma_pred_mode — Table 130, bin 0 ctx, bins 1-2 bypass.
        // If MIP is active the chroma derivation path inherits luma
        // (MipChromaDirectFlag can force this), but reading the mode
        // element is still gated on cclm_mode_flag == 0. For this round
        // we always take the non-CCLM path (cclm tool disabled in tests).
        // When intra_bdpcm_chroma_flag is not present or is 0 the
        // element is read.
        // We bail the read to a best-effort: if there is insufficient
        // bitstream, error up the chain.
        // Best-effort: missing bitstream tail means defaults.
        let icp = self.read_intra_chroma_pred_mode().unwrap_or_default();
        info.intra_chroma_pred_mode = icp;
        let luma = info.intra_pred_mode_y;
        info.intra_pred_mode_c = derive_intra_pred_mode_c(icp, luma);
    }

    fn read_intra_chroma_pred_mode(&mut self) -> Result<u32> {
        // Binarisation Table 130:
        //   val==4 → "0", val∈0..3 → "1" + FL(2 bits).
        let inc = ctx_inc_intra_chroma_pred_mode() as usize;
        let n = self.ctxs.intra_chroma_pred_mode.len() - 1;
        let bin0 = self
            .dec
            .decode_decision(&mut self.ctxs.intra_chroma_pred_mode[inc.min(n)])?;
        if bin0 == 0 {
            return Ok(4);
        }
        let bin1 = self.dec.decode_bypass()?;
        let bin2 = self.dec.decode_bypass()?;
        Ok((bin1 << 1) | bin2)
    }

    fn mip_mode_cmax(cb_w: u32, cb_h: u32) -> u32 {
        if cb_w == 4 && cb_h == 4 {
            15
        } else if cb_w == 4 || cb_h == 4 || (cb_w == 8 && cb_h == 8) {
            7
        } else {
            5
        }
    }
}

/// Truncated-Rice binarisation decode for the bypass-only variant
/// (intra_luma_mpm_idx).
fn decode_tr_bypass(dec: &mut ArithDecoder<'_>, c_max: u32, c_rice: u32) -> Result<u32> {
    let prefix_max = c_max >> c_rice;
    let mut prefix = 0u32;
    for _ in 0..prefix_max {
        let bit = dec.decode_bypass()?;
        if bit == 0 {
            break;
        }
        prefix += 1;
    }
    if c_rice == 0 {
        return Ok(prefix);
    }
    let suffix = dec.decode_bypass_bits(c_rice)?;
    Ok((prefix << c_rice) | suffix)
}

/// Truncated-binary binarisation decode (§9.3.3.4) — context-coded
/// bins, not currently used.
#[allow(dead_code)]
fn decode_tb(dec: &mut ArithDecoder<'_>, c_max: u32) -> Result<u32> {
    decode_tb_bypass(dec, c_max)
}

/// Truncated-binary binarisation decode for the bypass-only
/// variant (§9.3.3.4). `cMax == 0` decodes to 0 with no bin consumed.
fn decode_tb_bypass(dec: &mut ArithDecoder<'_>, c_max: u32) -> Result<u32> {
    if c_max == 0 {
        return Ok(0);
    }
    let n = c_max + 1;
    let k = 31 - n.leading_zeros(); // floor(log2(n))
    let u = (1u32 << (k + 1)) - n;
    // Read k bits.
    let prefix = dec.decode_bypass_bits(k)?;
    if prefix < u {
        return Ok(prefix);
    }
    // Read one more bit and remap.
    let extra = dec.decode_bypass()?;
    let combined = (prefix << 1) | extra;
    Ok(combined - u)
}

#[cfg(test)]
mod tests {
    use super::*;

    // === MPM candidate-list derivation tests (§8.4.2 eqs 210–240) ===

    #[test]
    fn mpm_list_both_above_dc_equal() {
        // candA == candB == 34 (angular). eqs 210–214.
        let l = build_mpm_cand_list(34, 34);
        assert_eq!(l[0], 34);
        assert_eq!(l[1], 2 + ((34 + 61) % 64)); // = 2 + 31 = 33
        assert_eq!(l[2], 2 + ((34 - 1) % 64)); // = 2 + 33 = 35
        assert_eq!(l[3], 2 + ((34 + 60) % 64)); // = 2 + 30 = 32
        assert_eq!(l[4], 2 + (34 % 64)); // = 2 + 34 = 36
    }

    #[test]
    fn mpm_list_both_above_dc_diff_1() {
        // candA=10, candB=11. diff=1. eqs 217-221.
        let l = build_mpm_cand_list(10, 11);
        assert_eq!(l[0], 10); // candA
        assert_eq!(l[1], 11); // candB
                              // minAB=10, maxAB=11.
        assert_eq!(l[2], 2 + ((10 + 61) % 64)); // = 2 + 7 = 9
        assert_eq!(l[3], 2 + ((11 - 1) % 64)); // = 2 + 10 = 12
        assert_eq!(l[4], 2 + ((10 + 60) % 64)); // = 2 + 6 = 8
    }

    #[test]
    fn mpm_list_both_above_dc_diff_ge_62() {
        // candA=3, candB=65 → diff=62.
        let l = build_mpm_cand_list(3, 65);
        assert_eq!(l[0], 3); // candA
        assert_eq!(l[1], 65); // candB
                              // eqs 222-224.
        assert_eq!(l[2], 2 + ((3u32.wrapping_sub(1)) % 64)); // = 2 + 2 = 4
        assert_eq!(l[3], 2 + ((65 + 61) % 64)); // = 2 + 62 = 64
        assert_eq!(l[4], 2 + (3 % 64)); // = 2 + 3 = 5
    }

    #[test]
    fn mpm_list_both_above_dc_diff_2() {
        // candA=10, candB=12 → diff=2. eqs 225-227.
        let l = build_mpm_cand_list(10, 12);
        assert_eq!(l[0], 10);
        assert_eq!(l[1], 12);
        assert_eq!(l[2], 2 + ((10 - 1) % 64)); // 11
        assert_eq!(l[3], 2 + ((10 + 61) % 64)); // 9
        assert_eq!(l[4], 2 + ((12 - 1) % 64)); // 13
    }

    #[test]
    fn mpm_list_both_above_dc_other_diff() {
        // candA=10, candB=20 → diff=10 (not 1, not 2, not ≥62). eqs 228-230.
        let l = build_mpm_cand_list(10, 20);
        assert_eq!(l[0], 10);
        assert_eq!(l[1], 20);
        assert_eq!(l[2], 2 + ((10 + 61) % 64)); // 9
        assert_eq!(l[3], 2 + ((10 - 1) % 64)); // 11
        assert_eq!(l[4], 2 + ((20 + 61) % 64)); // 19
    }

    #[test]
    fn mpm_list_one_above_dc() {
        // A planar, B angular. eqs 231-235 with maxAB=50.
        let l = build_mpm_cand_list(INTRA_PLANAR, 50);
        assert_eq!(l[0], 50);
        assert_eq!(l[1], 2 + ((50 + 61) % 64));
        assert_eq!(l[2], 2 + ((50 - 1) % 64));
        assert_eq!(l[3], 2 + ((50 + 60) % 64));
        assert_eq!(l[4], 2 + (50 % 64));
    }

    #[test]
    fn mpm_list_neither_above_dc() {
        // Both planar → eqs 236-240.
        let l = build_mpm_cand_list(INTRA_PLANAR, INTRA_DC);
        assert_eq!(
            l,
            [
                INTRA_DC,
                INTRA_ANGULAR50,
                INTRA_ANGULAR18,
                INTRA_ANGULAR46,
                INTRA_ANGULAR54
            ]
        );

        let l = build_mpm_cand_list(INTRA_PLANAR, INTRA_PLANAR);
        assert_eq!(
            l,
            [
                INTRA_DC,
                INTRA_ANGULAR50,
                INTRA_ANGULAR18,
                INTRA_ANGULAR46,
                INTRA_ANGULAR54
            ]
        );
    }

    // === derive_intra_pred_mode_y ===

    #[test]
    fn derive_luma_mode_planar_shortcircuit() {
        let mode = derive_intra_pred_mode_y(
            /*not_planar=*/ false,
            /*bdpcm=*/ false,
            /*bdpcm_v=*/ false,
            /*mpm_flag=*/ true,
            /*mpm_idx=*/ 3,
            /*remainder=*/ 0,
            &[10, 20, 30, 40, 50],
        );
        assert_eq!(mode, INTRA_PLANAR);
    }

    #[test]
    fn derive_luma_mode_bdpcm_vertical_and_horizontal() {
        let mode_v = derive_intra_pred_mode_y(true, true, true, false, 0, 0, &[0; 5]);
        let mode_h = derive_intra_pred_mode_y(true, true, false, false, 0, 0, &[0; 5]);
        assert_eq!(mode_v, INTRA_ANGULAR50);
        assert_eq!(mode_h, INTRA_ANGULAR18);
    }

    #[test]
    fn derive_luma_mode_mpm_match_picks_list_entry() {
        let list = [10, 20, 30, 40, 50];
        let mode = derive_intra_pred_mode_y(true, false, false, true, 2, 0, &list);
        assert_eq!(mode, 30);
    }

    #[test]
    fn derive_luma_mode_remainder_increments_and_skips() {
        // Sorted candModeList = [2, 5, 10, 20, 50].
        // remainder = 0 → mode starts at 1. Not >= 2, so stays at 1.
        let l = [10, 20, 2, 5, 50];
        let sorted = [2, 5, 10, 20, 50];
        // remainder=0 → mode = 1.
        let mode = derive_intra_pred_mode_y(true, false, false, false, 0, 0, &l);
        assert_eq!(mode, 1);

        // remainder = 1 → initial 2; equals sorted[0]=2, so ++ → 3; not ≥5, stops → 3.
        let mode = derive_intra_pred_mode_y(true, false, false, false, 0, 1, &l);
        assert_eq!(mode, 3);

        // remainder = 4 → initial 5; ≥ sorted[0]=2 → 6; ≥ sorted[1]=5 → 7; not ≥ 10 → 7.
        // Wait: sorted has [2,5,10,20,50]. At mode=5, compare against sorted[0]=2: 5>=2, mode=6.
        // sorted[1]=5: 6>=5, mode=7. sorted[2]=10: 7<10, no change. 7<20 and 7<50.
        let mode = derive_intra_pred_mode_y(true, false, false, false, 0, 4, &l);
        assert_eq!(mode, 7);

        // Verify list sort worked: compare against manual calculation.
        assert_eq!(
            {
                let mut s = l;
                s.sort_unstable();
                s
            },
            sorted
        );
    }

    // === derive_intra_pred_mode_c (Table 20) ===

    #[test]
    fn chroma_derivation_table_20_rows() {
        // Row (icp=0): luma=0 → 66, luma=50/18/1 → 0.
        assert_eq!(derive_intra_pred_mode_c(0, 0), 66);
        assert_eq!(derive_intra_pred_mode_c(0, 50), 0);
        assert_eq!(derive_intra_pred_mode_c(0, 18), 0);
        assert_eq!(derive_intra_pred_mode_c(0, 1), 0);
        // Row (icp=1): luma=50 → 66, else 50.
        assert_eq!(derive_intra_pred_mode_c(1, 0), 50);
        assert_eq!(derive_intra_pred_mode_c(1, 50), 66);
        assert_eq!(derive_intra_pred_mode_c(1, 18), 50);
        // Row (icp=2): luma=18 → 66, else 18.
        assert_eq!(derive_intra_pred_mode_c(2, 0), 18);
        assert_eq!(derive_intra_pred_mode_c(2, 18), 66);
        assert_eq!(derive_intra_pred_mode_c(2, 50), 18);
        // Row (icp=3): luma=1 → 66, else 1.
        assert_eq!(derive_intra_pred_mode_c(3, 0), 1);
        assert_eq!(derive_intra_pred_mode_c(3, 1), 66);
        // Row (icp=4): passthrough.
        assert_eq!(derive_intra_pred_mode_c(4, 0), 0);
        assert_eq!(derive_intra_pred_mode_c(4, 50), 50);
        assert_eq!(derive_intra_pred_mode_c(4, 25), 25);
    }

    // === cand_intra_mode_for_side corners ===

    #[test]
    fn cand_mode_unavailable_is_planar() {
        let v = cand_intra_mode_for_side(false, None, false, false);
        assert_eq!(v, INTRA_PLANAR);
        let v = cand_intra_mode_for_side(
            false,
            Some(IntraNeighbour {
                intra_pred_mode_y: 30,
                mip: false,
                pred_mode: Some(CuPredMode::Intra),
            }),
            false,
            false,
        );
        assert_eq!(v, INTRA_PLANAR);
    }

    #[test]
    fn cand_mode_mip_neighbour_is_planar() {
        let nb = IntraNeighbour {
            intra_pred_mode_y: 30,
            mip: true,
            pred_mode: Some(CuPredMode::Intra),
        };
        let v = cand_intra_mode_for_side(true, Some(nb), false, false);
        assert_eq!(v, INTRA_PLANAR);
    }

    #[test]
    fn cand_mode_non_intra_neighbour_is_planar() {
        let nb = IntraNeighbour {
            intra_pred_mode_y: 30,
            mip: false,
            pred_mode: Some(CuPredMode::Inter),
        };
        let v = cand_intra_mode_for_side(true, Some(nb), false, false);
        assert_eq!(v, INTRA_PLANAR);
    }

    #[test]
    fn cand_mode_side_b_crossing_ctb_row_is_planar() {
        let nb = IntraNeighbour {
            intra_pred_mode_y: 30,
            mip: false,
            pred_mode: Some(CuPredMode::Intra),
        };
        let v = cand_intra_mode_for_side(true, Some(nb), true, true);
        assert_eq!(v, INTRA_PLANAR);
        // Side A is never subject to the CTB-row rule, even if the
        // crossing flag is somehow set.
        let v = cand_intra_mode_for_side(true, Some(nb), false, true);
        assert_eq!(v, 30);
    }

    // === TB decode correctness (bypass path) ===

    #[test]
    fn tb_cmax_60_decodes_full_range() {
        // A carefully crafted 64-byte all-zero stream returns 0 from
        // every bypass bit (ivlOffset stays 0 forever). A bypass-TB
        // decode with cMax=60 needs k=5, u=64-61=3. Reading 5 zero
        // bits yields prefix=0 < 3 → returns 0. Good.
        let data = [0u8; 64];
        let mut dec = ArithDecoder::new(&data).unwrap();
        let v = decode_tb_bypass(&mut dec, 60).unwrap();
        assert_eq!(v, 0);
    }

    // === CuToolFlags defaults: all disabled, chroma 4:2:0 ===

    #[test]
    fn leaf_cu_default_round_trip_without_mip_isp_mrl() {
        // Set up a stream with known deterministic behaviour: all zero.
        // With all tools disabled, the reader must read:
        //   - intra_luma_mpm_flag (ctx) → 0 (depending on init bias)
        //   - depending on flag value, either not_planar + mpm_idx or
        //     mpm_remainder.
        //   - intra_chroma_pred_mode.
        // On all-zero streams the specific answers depend on init; we
        // just check that decoding succeeds and produces legal modes.
        let tools = CuToolFlags {
            ibc: false,
            palette: false,
            bdpcm: false,
            mip: false,
            mrl: false,
            isp: false,
            act: false,
            max_tb_size_y: 64,
            min_tb_size_y: 4,
            max_ts_size: 32,
            ctb_size_y: 128,
            chroma_format_idc: 1,
            cu_qp_delta_enabled: false,
            cu_chroma_qp_offset_enabled: false,
            chroma_qp_offset_list_len_minus1: 0,
            joint_cbcr_enabled: false,
            ts_residual_coding_disabled: false,
            slice_is_inter: false,
            max_num_merge_cand: 6,
            mmvd_enabled: false,
            ph_mmvd_fullpel_only: false,
            ciip_enabled: false,
            gpm_enabled: false,
            max_num_gpm_merge_cand: 0,
            slice_is_b: false,
            max_num_subblock_merge_cand: 0,
        };
        let data = [0u8; 128];
        let mut dec = ArithDecoder::new(&data).unwrap();
        let mut ctxs = LeafCuCtxs::init(26);
        let mut info = LeafCuInfo {
            x0: 0,
            y0: 0,
            cb_width: 16,
            cb_height: 16,
            ..LeafCuInfo::default()
        };
        let mut residual = LeafCuResidual::default();
        let neigh = CuNeighbourhood::default();
        let mut reader = LeafCuReader::new(&mut dec, &mut ctxs, tools);
        reader.decode(&mut info, &mut residual, &neigh).unwrap();
        assert_eq!(info.pred_mode, CuPredMode::Intra);
        // Legal luma mode range.
        assert!(info.intra_pred_mode_y <= 66);
        assert!(info.intra_pred_mode_c <= 83);
    }

    #[test]
    fn leaf_cu_with_mip_reads_transposed_and_mode() {
        let tools = CuToolFlags {
            mip: true,
            chroma_format_idc: 1,
            ctb_size_y: 128,
            max_tb_size_y: 64,
            min_tb_size_y: 4,
            max_ts_size: 32,
            ..CuToolFlags::default()
        };
        let data = [0u8; 128];
        let mut dec = ArithDecoder::new(&data).unwrap();
        let mut ctxs = LeafCuCtxs::init(26);
        let mut info = LeafCuInfo {
            x0: 0,
            y0: 0,
            cb_width: 8,
            cb_height: 8,
            ..LeafCuInfo::default()
        };
        let mut residual = LeafCuResidual::default();
        let neigh = CuNeighbourhood::default();
        let mut reader = LeafCuReader::new(&mut dec, &mut ctxs, tools);
        reader.decode(&mut info, &mut residual, &neigh).unwrap();
        // With 8x8 CU + MIP enabled, cMax = 7 → 3 bypass bits; parse
        // must yield mip_mode ∈ 0..=7.
        if info.intra_mip_flag {
            assert!(info.intra_mip_mode <= 7);
        }
    }

    /// BDPCM-luma path: when `sps_bdpcm_enabled_flag` is set and the
    /// CU fits inside `MaxTsSize`, the reader consumes the
    /// `intra_bdpcm_luma_flag` (Table 69) and — when set — the
    /// `intra_bdpcm_luma_dir_flag` (Table 70). The luma intra mode is
    /// then derived per §8.4.2 from `BdpcmDir`: 0 → ANGULAR18,
    /// 1 → ANGULAR50. With a zero-only stream the LPS path is not
    /// taken, so the flag stays at 0; the test still verifies the new
    /// code path no longer surfaces Unsupported (compared to r17
    /// behaviour).
    #[test]
    fn bdpcm_luma_flag_parses_zero_stream_does_not_error() {
        let tools = CuToolFlags {
            bdpcm: true,
            chroma_format_idc: 1,
            ctb_size_y: 128,
            max_tb_size_y: 64,
            min_tb_size_y: 4,
            max_ts_size: 32,
            ..CuToolFlags::default()
        };
        let data = [0u8; 64];
        let mut dec = ArithDecoder::new(&data).unwrap();
        let mut ctxs = LeafCuCtxs::init(26);
        let mut info = LeafCuInfo {
            cb_width: 8,
            cb_height: 8,
            ..LeafCuInfo::default()
        };
        let mut residual = LeafCuResidual::default();
        let neigh = CuNeighbourhood::default();
        let mut reader = LeafCuReader::new(&mut dec, &mut ctxs, tools);
        // The CU fits the BDPCM gate, the flag is read, and the
        // pipeline must not surface the previous "Table 69 init not
        // plumbed" Unsupported.
        let res = reader.decode(&mut info, &mut residual, &neigh);
        match res {
            Ok(()) => {}
            Err(Error::Unsupported(msg)) => {
                assert!(
                    !msg.contains("BDPCM"),
                    "BDPCM flag parsing should no longer be Unsupported: {msg}"
                );
            }
            Err(_) => {
                // Other errors (e.g. truncated bitstream past the CU
                // header) are acceptable — we only care that BDPCM is
                // not the blocker.
            }
        }
        // When BDPCM-luma did fire, the derived intra mode must be one
        // of the two angular modes mandated by §8.4.2.
        if info.intra_bdpcm_luma {
            assert!(
                info.intra_pred_mode_y == INTRA_ANGULAR18
                    || info.intra_pred_mode_y == INTRA_ANGULAR50
            );
        }
    }

    /// Tools that disable BDPCM at the SPS level (`sps_bdpcm_enabled_flag
    /// == 0`) leave the BDPCM flag unread and `intra_bdpcm_luma`
    /// stays at the default `false`.
    #[test]
    fn bdpcm_disabled_skips_flag_read() {
        let tools = CuToolFlags {
            bdpcm: false,
            chroma_format_idc: 1,
            ctb_size_y: 128,
            max_tb_size_y: 64,
            min_tb_size_y: 4,
            max_ts_size: 32,
            ..CuToolFlags::default()
        };
        let data = [0u8; 64];
        let mut dec = ArithDecoder::new(&data).unwrap();
        let mut ctxs = LeafCuCtxs::init(26);
        let mut info = LeafCuInfo {
            cb_width: 8,
            cb_height: 8,
            ..LeafCuInfo::default()
        };
        let mut residual = LeafCuResidual::default();
        let neigh = CuNeighbourhood::default();
        let mut reader = LeafCuReader::new(&mut dec, &mut ctxs, tools);
        let _ = reader.decode(&mut info, &mut residual, &neigh);
        assert!(!info.intra_bdpcm_luma);
        assert!(!info.intra_bdpcm_luma_dir);
    }

    // === MPM-remainder swap + skip behaviour end-to-end via derivation ===

    #[test]
    fn end_to_end_neither_above_dc_remainder() {
        // Both neighbours planar → candModeList per eq 236-240.
        let cand_list = build_mpm_cand_list(INTRA_PLANAR, INTRA_PLANAR);
        // sorted = [1, 18, 46, 50, 54] (after sorting DC, 50, 18, 46, 54).
        // remainder 0 → mode 1 → ≥1 → 2 → <18 → stop → 2.
        let mode = derive_intra_pred_mode_y(true, false, false, false, 0, 0, &cand_list);
        assert_eq!(mode, 2);
        // remainder 15 → initial 16 → ≥1 → 17 → <18 → stop → 17.
        let mode = derive_intra_pred_mode_y(true, false, false, false, 0, 15, &cand_list);
        assert_eq!(mode, 17);
        // remainder 20 → initial 21 → ≥1 → 22 → ≥18 → 23 → <46 → stop → 23.
        let mode = derive_intra_pred_mode_y(true, false, false, false, 0, 20, &cand_list);
        assert_eq!(mode, 23);
    }

    #[test]
    fn mip_mode_cmax_sizes() {
        assert_eq!(LeafCuReader::mip_mode_cmax(4, 4), 15);
        assert_eq!(LeafCuReader::mip_mode_cmax(8, 4), 7);
        assert_eq!(LeafCuReader::mip_mode_cmax(8, 8), 7);
        assert_eq!(LeafCuReader::mip_mode_cmax(16, 16), 5);
        assert_eq!(LeafCuReader::mip_mode_cmax(32, 16), 5);
    }

    #[test]
    fn ibc_and_palette_tools_surface_unsupported() {
        let data = [0u8; 32];
        let mut dec = ArithDecoder::new(&data).unwrap();
        let mut ctxs = LeafCuCtxs::init(26);
        let mut info = LeafCuInfo {
            cb_width: 16,
            cb_height: 16,
            ..LeafCuInfo::default()
        };
        let mut residual = LeafCuResidual::default();
        let neigh = CuNeighbourhood::default();

        let mut tools = CuToolFlags {
            chroma_format_idc: 1,
            ctb_size_y: 128,
            max_tb_size_y: 64,
            min_tb_size_y: 4,
            max_ts_size: 32,
            ibc: true,
            ..CuToolFlags::default()
        };
        let mut reader = LeafCuReader::new(&mut dec, &mut ctxs, tools);
        assert!(matches!(
            reader.decode(&mut info, &mut residual, &neigh),
            Err(Error::Unsupported(_))
        ));

        tools.ibc = false;
        tools.palette = true;
        let mut reader = LeafCuReader::new(&mut dec, &mut ctxs, tools);
        assert!(matches!(
            reader.decode(&mut info, &mut residual, &neigh),
            Err(Error::Unsupported(_))
        ));
    }

    #[test]
    fn mpm_idx_exercises_all_five_candidates_via_derivation() {
        // Take a cand_list with distinct modes and verify every index
        // picks the right entry.
        let list = [20u32, 30, 40, 50, 60];
        for idx in 0..5u32 {
            let m = derive_intra_pred_mode_y(true, false, false, true, idx, 0, &list);
            assert_eq!(m, list[idx as usize], "mpm_idx={idx} failed");
        }
    }

    /// Hand-built 4×4 intra-only CU decode: verifies that the full
    /// coding_unit() + transform_unit() chain runs end-to-end on a
    /// zero-stream without panicking, producing CBFs and a coefficient
    /// array of the right shape.
    #[test]
    fn intra_4x4_cu_runs_full_decode_chain() {
        let tools = CuToolFlags {
            chroma_format_idc: 1,
            ctb_size_y: 128,
            max_tb_size_y: 64,
            min_tb_size_y: 4,
            max_ts_size: 32,
            ..CuToolFlags::default()
        };
        let data = [0u8; 256];
        let mut dec = ArithDecoder::new(&data).unwrap();
        let mut ctxs = LeafCuCtxs::init(26);
        let mut info = LeafCuInfo {
            cb_width: 4,
            cb_height: 4,
            ..LeafCuInfo::default()
        };
        let mut residual = LeafCuResidual::default();
        let neigh = CuNeighbourhood::default();
        let mut reader = LeafCuReader::new(&mut dec, &mut ctxs, tools);
        reader.decode(&mut info, &mut residual, &neigh).unwrap();
        // CBFs read as MPS on the zero-stream for this QP; regardless
        // of exact values the reader must produce legal output.
        // When tu_y_coded_flag is true, the luma residual array must
        // have the expected shape.
        if info.tu_y_coded_flag {
            assert_eq!(residual.luma_levels.len(), 16);
        } else {
            assert!(residual.luma_levels.is_empty());
        }
        // Chroma CBFs must also yield either 0 or a 2x2 level array
        // (4:2:0 chroma plane of a 4x4 luma is 2x2).
        if info.tu_cb_coded_flag {
            assert_eq!(residual.cb_levels.len(), 4);
        }
        if info.tu_cr_coded_flag {
            assert_eq!(residual.cr_levels.len(), 4);
        }
        // QP delta stays 0 because cu_qp_delta_enabled is false.
        assert_eq!(info.cu_qp_delta_val, 0);
    }

    /// QP delta path: enable `cu_qp_delta_enabled` and verify that a
    /// zero-stream at least runs the read. Cannot assert on the exact
    /// value without hand-crafting the bit pattern.
    #[test]
    fn cu_qp_delta_is_read_when_enabled_and_cbf_fires() {
        let tools = CuToolFlags {
            chroma_format_idc: 1,
            ctb_size_y: 128,
            max_tb_size_y: 64,
            min_tb_size_y: 4,
            max_ts_size: 32,
            cu_qp_delta_enabled: true,
            ..CuToolFlags::default()
        };
        let data = [0u8; 256];
        let mut dec = ArithDecoder::new(&data).unwrap();
        let mut ctxs = LeafCuCtxs::init(26);
        let mut info = LeafCuInfo {
            cb_width: 8,
            cb_height: 8,
            ..LeafCuInfo::default()
        };
        let mut residual = LeafCuResidual::default();
        let neigh = CuNeighbourhood::default();
        let mut reader = LeafCuReader::new(&mut dec, &mut ctxs, tools);
        reader.decode(&mut info, &mut residual, &neigh).unwrap();
        // cu_qp_delta_val is in the legal signed range; on zero stream
        // it's almost always 0 (MPS branch). We can't bound it tighter
        // without re-implementing the CABAC engine math here.
        assert!(info.cu_qp_delta_val.abs() < 64);
    }

    #[test]
    fn tb_bypass_reads_small_cmax_cases() {
        // cMax = 3 → n=4 → k=2 → u=0. pure FL(2 bits).
        // All-zero stream → decodes 0.
        let data = [0u8; 4];
        let mut dec = ArithDecoder::new(&data).unwrap();
        assert_eq!(decode_tb_bypass(&mut dec, 3).unwrap(), 0);

        // cMax = 4 → n=5 → k=2 → u=8-5=3. prefix∈[0..3) is FL 2-bit
        // (values 0..2); ≥3 needs an extra bit. 0-stream → prefix=0<3 → v=0.
        let data = [0u8; 4];
        let mut dec = ArithDecoder::new(&data).unwrap();
        assert_eq!(decode_tb_bypass(&mut dec, 4).unwrap(), 0);
    }

    // === mvd_coding() syntax (§7.3.10.10 / §9.3.3.14) ===

    use crate::cabac_enc::ArithEncoder;

    /// Encode `abs_mvd_minus2` per the §9.3.3.6 limited k-th order
    /// Exp-Golomb binarisation with `k = 1`, `maxPreExtLen = 15`,
    /// `truncSuffixLen = 17` — the encode counterpart of
    /// `read_abs_mvd_minus2`. All bins bypass-coded.
    fn encode_abs_mvd_minus2(enc: &mut ArithEncoder, symbol_val: u32) {
        const K: u32 = 1;
        const MAX_PRE_EXT_LEN: u32 = 15;
        const TRUNC_SUFFIX_LEN: u32 = 17;

        let code_value = symbol_val >> K;
        let mut pre_ext_len = 0u32;
        while pre_ext_len < MAX_PRE_EXT_LEN && code_value > ((2u32 << pre_ext_len) - 2) {
            pre_ext_len += 1;
            enc.encode_bypass(1).unwrap();
        }
        let escape_length = if pre_ext_len == MAX_PRE_EXT_LEN {
            TRUNC_SUFFIX_LEN
        } else {
            enc.encode_bypass(0).unwrap();
            pre_ext_len + K
        };
        let val = symbol_val - (((1u32 << pre_ext_len) - 1) << K);
        for i in (0..escape_length).rev() {
            enc.encode_bypass((val >> i) & 1).unwrap();
        }
    }

    /// Encode one `mvd_coding()` structure bin-for-bin per §7.3.10.10,
    /// driving the supplied context bundle (the mirror of
    /// `read_mvd_coding`). `lmvd` is the `(lMvd[0], lMvd[1])` pair.
    fn encode_mvd_coding(enc: &mut ArithEncoder, ctxs: &mut LeafCuCtxs, lmvd: (i32, i32)) {
        let init_type = ctxs.init_type as usize;
        let g0_slot = init_type.min(ctxs.abs_mvd_greater0_flag.len() - 1);
        let g1_slot = init_type.min(ctxs.abs_mvd_greater1_flag.len() - 1);
        let comp = [lmvd.0, lmvd.1];
        let greater0 = [comp[0] != 0, comp[1] != 0];
        let greater1 = [comp[0].abs() > 1, comp[1].abs() > 1];

        // Step 1: both greater0 flags.
        for c in 0..2 {
            enc.encode_decision(&mut ctxs.abs_mvd_greater0_flag[g0_slot], greater0[c] as u32)
                .unwrap();
        }
        // Step 2: greater1 per non-zero component.
        for c in 0..2 {
            if greater0[c] {
                enc.encode_decision(&mut ctxs.abs_mvd_greater1_flag[g1_slot], greater1[c] as u32)
                    .unwrap();
            }
        }
        // Step 3: magnitude tail + sign per non-zero component.
        for c in 0..2 {
            if greater0[c] {
                if greater1[c] {
                    encode_abs_mvd_minus2(enc, comp[c].unsigned_abs() - 2);
                }
                enc.encode_bypass((comp[c] < 0) as u32).unwrap();
            }
        }
    }

    fn mvd_coding_round_trip(lmvd: (i32, i32), init_type: u8) -> MotionVector {
        let mut enc = ArithEncoder::new();
        let mut enc_ctxs = LeafCuCtxs::init_with_init_type(26, init_type);
        encode_mvd_coding(&mut enc, &mut enc_ctxs, lmvd);
        enc.encode_terminate(1).unwrap();
        let bytes = enc.finish();
        let mut padded = bytes;
        padded.extend_from_slice(&[0u8; 32]);

        let mut dec = ArithDecoder::new(&padded).unwrap();
        let mut dec_ctxs = LeafCuCtxs::init_with_init_type(26, init_type);
        let tools = CuToolFlags::default();
        let mut reader = LeafCuReader::new(&mut dec, &mut dec_ctxs, tools);
        reader.read_mvd_coding().unwrap()
    }

    #[test]
    fn mvd_coding_zero_both_components() {
        let mv = mvd_coding_round_trip((0, 0), 1);
        assert_eq!(mv, MotionVector { x: 0, y: 0 });
    }

    #[test]
    fn mvd_coding_unit_magnitude_skips_minus2() {
        // |lMvd| == 1 ⇒ greater1 == 0 ⇒ no abs_mvd_minus2 bin, just the
        // sign. Both signs and one positive / one negative.
        assert_eq!(
            mvd_coding_round_trip((1, -1), 1),
            MotionVector { x: 1, y: -1 }
        );
        assert_eq!(
            mvd_coding_round_trip((-1, 1), 2),
            MotionVector { x: -1, y: 1 }
        );
    }

    #[test]
    fn mvd_coding_mixed_zero_and_nonzero() {
        // x == 0 (no greater1/minus2/sign for x), y == 5 (greater1 set,
        // abs_mvd_minus2 == 3).
        assert_eq!(
            mvd_coding_round_trip((0, 5), 1),
            MotionVector { x: 0, y: 5 }
        );
        assert_eq!(
            mvd_coding_round_trip((-7, 0), 2),
            MotionVector { x: -7, y: 0 }
        );
    }

    #[test]
    fn mvd_coding_large_magnitudes_round_trip() {
        // Values spanning small, sub-pel-scale, and large magnitudes to
        // exercise the §9.3.3.6 unary-prefix growth of the EG1 suffix.
        for &(x, y) in &[
            (2, 2),
            (-2, 3),
            (16, -16),
            (255, -255),
            (1000, -1234),
            (65535, -65535),
            (131070, -131070), // 2^17 − 2: the max |lMvd| for abs_mvd
        ] {
            let mv = mvd_coding_round_trip((x, y), 1);
            assert_eq!(
                mv,
                MotionVector { x, y },
                "mvd round trip failed at ({x},{y})"
            );
        }
    }

    #[test]
    fn mvd_coding_eq190_derivation_matches_components() {
        // Spot-check eq. 190 directly: lMvd[c] = greater0 *
        // (abs_mvd_minus2 + 2) * (1 − 2*sign). For lMvd = 9 the encoder
        // emits abs_mvd_minus2 = 7, sign = 0; the decoder must rebuild 9.
        // For lMvd = −9 sign = 1.
        assert_eq!(
            mvd_coding_round_trip((9, -9), 2),
            MotionVector { x: 9, y: -9 }
        );
    }

    #[test]
    fn abs_mvd_minus2_limited_egk_round_trip() {
        // Directly exercise the §9.3.3.6 limited-EGk codec across the
        // value range, including the escape boundary (maxPreExtLen).
        for sym in [0u32, 1, 2, 3, 7, 8, 100, 1000, 65533, 131068] {
            let mut enc = ArithEncoder::new();
            encode_abs_mvd_minus2(&mut enc, sym);
            enc.encode_terminate(1).unwrap();
            let mut padded = enc.finish();
            padded.extend_from_slice(&[0u8; 32]);
            let mut dec = ArithDecoder::new(&padded).unwrap();
            let mut ctxs = LeafCuCtxs::init_with_init_type(26, 1);
            let tools = CuToolFlags::default();
            let mut reader = LeafCuReader::new(&mut dec, &mut ctxs, tools);
            let got = reader.read_abs_mvd_minus2().unwrap();
            assert_eq!(got, sym, "abs_mvd_minus2 round trip failed at {sym}");
        }
    }

    // ---- Round-108: inter MVP-side syntax (inter_pred_idc /
    // sym_mvd_flag / ref_idx_lX / mvp_lX_flag) ----

    /// Encoder mirror of `read_inter_pred_idc` driving the same context
    /// bundle bin-for-bin (§9.3.3.9 / Table 131).
    fn encode_inter_pred_idc(
        enc: &mut ArithEncoder,
        ctxs: &mut LeafCuCtxs,
        dir: InterPredDir,
        cb_width: u32,
        cb_height: u32,
    ) {
        let block = (ctxs.init_type as usize).saturating_sub(1) * 6;
        let n = ctxs.inter_pred_idc.len();
        if cb_width + cb_height > 12 {
            let inc0 = ctx_inc_inter_pred_idc_bin0(cb_width, cb_height) as usize;
            let slot0 = (block + inc0).min(n - 1);
            let bin0 = if dir == InterPredDir::PredBi { 1 } else { 0 };
            enc.encode_decision(&mut ctxs.inter_pred_idc[slot0], bin0)
                .unwrap();
            if dir != InterPredDir::PredBi {
                let inc1 = ctx_inc_inter_pred_idc_bin1() as usize;
                let slot1 = (block + inc1).min(n - 1);
                let bin1 = if dir == InterPredDir::PredL1 { 1 } else { 0 };
                enc.encode_decision(&mut ctxs.inter_pred_idc[slot1], bin1)
                    .unwrap();
            }
        } else {
            let inc0 = ctx_inc_inter_pred_idc_bin0(cb_width, cb_height) as usize;
            let slot0 = (block + inc0).min(n - 1);
            let bin0 = if dir == InterPredDir::PredL1 { 1 } else { 0 };
            enc.encode_decision(&mut ctxs.inter_pred_idc[slot0], bin0)
                .unwrap();
        }
    }

    fn inter_pred_idc_round_trip(
        dir: InterPredDir,
        cb_width: u32,
        cb_height: u32,
        init_type: u8,
    ) -> InterPredDir {
        let mut enc = ArithEncoder::new();
        let mut enc_ctxs = LeafCuCtxs::init_with_init_type(26, init_type);
        encode_inter_pred_idc(&mut enc, &mut enc_ctxs, dir, cb_width, cb_height);
        enc.encode_terminate(1).unwrap();
        let mut padded = enc.finish();
        padded.extend_from_slice(&[0u8; 32]);
        let mut dec = ArithDecoder::new(&padded).unwrap();
        let mut dec_ctxs = LeafCuCtxs::init_with_init_type(26, init_type);
        let tools = CuToolFlags::default();
        let mut reader = LeafCuReader::new(&mut dec, &mut dec_ctxs, tools);
        reader.read_inter_pred_idc(cb_width, cb_height).unwrap()
    }

    #[test]
    fn inter_pred_idc_two_bin_form_round_trips() {
        // (cbWidth + cbHeight) > 12: PRED_L0 = 00, PRED_L1 = 01,
        // PRED_BI = 1. 32x32 sum = 64 > 12. B-slice init_type = 2.
        for dir in [
            InterPredDir::PredL0,
            InterPredDir::PredL1,
            InterPredDir::PredBi,
        ] {
            assert_eq!(inter_pred_idc_round_trip(dir, 32, 32, 2), dir);
        }
    }

    #[test]
    fn inter_pred_idc_single_bin_form_when_sum_is_12() {
        // (cbWidth + cbHeight) == 12: PRED_BI suppressed, single bin
        // gives PRED_L0 = 0 / PRED_L1 = 1. 4x8 sum = 12.
        assert_eq!(
            inter_pred_idc_round_trip(InterPredDir::PredL0, 4, 8, 2),
            InterPredDir::PredL0
        );
        assert_eq!(
            inter_pred_idc_round_trip(InterPredDir::PredL1, 8, 4, 2),
            InterPredDir::PredL1
        );
    }

    #[test]
    fn inter_pred_idc_bin0_ctx_inc_matches_spec() {
        // Bin 0 ctxInc = 7 − ((1 + Log2(W) + Log2(H)) >> 1) for sum > 12,
        // else 5. Spot-check a few power-of-two sizes.
        // 32x32: 7 − ((1 + 5 + 5) >> 1) = 7 − (11 >> 1) = 7 − 5 = 2.
        assert_eq!(ctx_inc_inter_pred_idc_bin0(32, 32), 2);
        // 16x16: 7 − ((1 + 4 + 4) >> 1) = 7 − 4 = 3.
        assert_eq!(ctx_inc_inter_pred_idc_bin0(16, 16), 3);
        // 8x8: 7 − ((1 + 3 + 3) >> 1) = 7 − 3 = 4. sum = 16 > 12.
        assert_eq!(ctx_inc_inter_pred_idc_bin0(8, 8), 4);
        // 64x64: 7 − ((1 + 6 + 6) >> 1) = 7 − 6 = 1.
        assert_eq!(ctx_inc_inter_pred_idc_bin0(64, 64), 1);
        // sum == 12 → 5.
        assert_eq!(ctx_inc_inter_pred_idc_bin0(4, 8), 5);
        // Both P (init 1) and B (init 2) slot blocks must stay in range.
        // Block offset is (init_type - 1) * 6 per Table 51.
        for it in [1u8, 2] {
            let ctxs = LeafCuCtxs::init_with_init_type(26, it);
            let block = (it as usize - 1) * 6;
            let inc = ctx_inc_inter_pred_idc_bin0(64, 64) as usize;
            assert!(block + inc < ctxs.inter_pred_idc.len());
            // Bin 1 ctxInc = 5 → top slot of the per-initType block.
            assert!(block + 5 < ctxs.inter_pred_idc.len());
        }
    }

    fn sym_mvd_flag_round_trip(flag: bool, init_type: u8) -> bool {
        let mut enc = ArithEncoder::new();
        let mut enc_ctxs = LeafCuCtxs::init_with_init_type(26, init_type);
        let slot = (init_type as usize)
            .saturating_sub(1)
            .min(enc_ctxs.sym_mvd_flag.len() - 1);
        enc.encode_decision(&mut enc_ctxs.sym_mvd_flag[slot], flag as u32)
            .unwrap();
        enc.encode_terminate(1).unwrap();
        let mut padded = enc.finish();
        padded.extend_from_slice(&[0u8; 32]);
        let mut dec = ArithDecoder::new(&padded).unwrap();
        let mut dec_ctxs = LeafCuCtxs::init_with_init_type(26, init_type);
        let tools = CuToolFlags::default();
        let mut reader = LeafCuReader::new(&mut dec, &mut dec_ctxs, tools);
        reader.read_sym_mvd_flag().unwrap()
    }

    #[test]
    fn sym_mvd_flag_round_trips_both_init_types() {
        for it in [1u8, 2] {
            assert!(!sym_mvd_flag_round_trip(false, it));
            assert!(sym_mvd_flag_round_trip(true, it));
        }
    }

    /// Encoder mirror of `read_ref_idx_lx` (TR, cMax = numActive − 1).
    fn encode_ref_idx_lx(
        enc: &mut ArithEncoder,
        ctxs: &mut LeafCuCtxs,
        value: u32,
        num_ref_idx_active: u32,
    ) {
        let cmax = num_ref_idx_active.saturating_sub(1);
        if cmax == 0 {
            return;
        }
        let block = (ctxs.init_type as usize).saturating_sub(1) * 2;
        let n = ctxs.ref_idx_lx.len();
        let mut i = 0u32;
        while i < cmax {
            let bit = if i < value { 1 } else { 0 };
            if i < 2 {
                let inc = ctx_inc_ref_idx_lx(i) as usize;
                let slot = (block + inc).min(n - 1);
                enc.encode_decision(&mut ctxs.ref_idx_lx[slot], bit)
                    .unwrap();
            } else {
                enc.encode_bypass(bit).unwrap();
            }
            if bit == 0 {
                break;
            }
            i += 1;
        }
    }

    fn ref_idx_lx_round_trip(value: u32, num_ref_idx_active: u32, init_type: u8) -> u32 {
        let mut enc = ArithEncoder::new();
        let mut enc_ctxs = LeafCuCtxs::init_with_init_type(26, init_type);
        encode_ref_idx_lx(&mut enc, &mut enc_ctxs, value, num_ref_idx_active);
        enc.encode_terminate(1).unwrap();
        let mut padded = enc.finish();
        padded.extend_from_slice(&[0u8; 32]);
        let mut dec = ArithDecoder::new(&padded).unwrap();
        let mut dec_ctxs = LeafCuCtxs::init_with_init_type(26, init_type);
        let tools = CuToolFlags::default();
        let mut reader = LeafCuReader::new(&mut dec, &mut dec_ctxs, tools);
        reader.read_ref_idx_lx(num_ref_idx_active).unwrap()
    }

    #[test]
    fn ref_idx_lx_single_ref_reads_no_bins() {
        // cMax = 0 ⇒ no bins, always 0. Verified by feeding an all-zero
        // stream and reading numActive = 1; the result must be exactly 0
        // regardless of bytes available.
        for it in [1u8, 2] {
            assert_eq!(ref_idx_lx_round_trip(0, 1, it), 0);
        }
    }

    #[test]
    fn ref_idx_lx_truncated_unary_round_trip() {
        // numActive = 4 ⇒ cMax = 3. Values 0..=3 cover ctx-coded bins 0/1
        // plus the bypass tail (bin 2) and the cMax-truncation (value 3
        // stops without a terminating zero bin).
        for it in [1u8, 2] {
            for v in 0..=3u32 {
                assert_eq!(
                    ref_idx_lx_round_trip(v, 4, it),
                    v,
                    "ref_idx_lx round trip failed for v={v} init={it}"
                );
            }
        }
    }

    #[test]
    fn ref_idx_lx_two_ref_uses_only_ctx_bin0() {
        // numActive = 2 ⇒ cMax = 1 ⇒ a single ctx-coded bin (ctxInc 0).
        for it in [1u8, 2] {
            assert_eq!(ref_idx_lx_round_trip(0, 2, it), 0);
            assert_eq!(ref_idx_lx_round_trip(1, 2, it), 1);
        }
    }

    fn mvp_lx_flag_round_trip(value: u32, init_type: u8) -> u32 {
        let mut enc = ArithEncoder::new();
        let mut enc_ctxs = LeafCuCtxs::init_with_init_type(26, init_type);
        let slot = (init_type as usize).min(enc_ctxs.mvp_lx_flag.len() - 1);
        enc.encode_decision(&mut enc_ctxs.mvp_lx_flag[slot], value)
            .unwrap();
        enc.encode_terminate(1).unwrap();
        let mut padded = enc.finish();
        padded.extend_from_slice(&[0u8; 32]);
        let mut dec = ArithDecoder::new(&padded).unwrap();
        let mut dec_ctxs = LeafCuCtxs::init_with_init_type(26, init_type);
        let tools = CuToolFlags::default();
        let mut reader = LeafCuReader::new(&mut dec, &mut dec_ctxs, tools);
        reader.read_mvp_lx_flag().unwrap()
    }

    #[test]
    fn mvp_lx_flag_round_trips_all_init_types() {
        for it in [0u8, 1, 2] {
            assert_eq!(mvp_lx_flag_round_trip(0, it), 0);
            assert_eq!(mvp_lx_flag_round_trip(1, it), 1);
        }
    }

    // ---- Round-126: §7.3.10.5 bcw_idx CABAC reader ----

    /// Encoder mirror of `read_bcw_idx`. TR with `cMax = NoBackwardPredFlag
    /// ? 4 : 2`, `cRiceParam = 0`: bin 0 is ctx-coded against Table 91
    /// slot `init_type - 1` (ctxInc = 0), bins 1.. are bypass-coded.
    fn encode_bcw_idx(
        enc: &mut ArithEncoder,
        ctxs: &mut LeafCuCtxs,
        value: u32,
        no_backward_pred_flag: bool,
    ) {
        let cmax = if no_backward_pred_flag { 4u32 } else { 2u32 };
        if cmax == 0 {
            return;
        }
        let block = (ctxs.init_type as usize).saturating_sub(1);
        let n = ctxs.bcw_idx.len();
        let slot = block.min(n - 1);
        let bin0 = if value == 0 { 0 } else { 1 };
        enc.encode_decision(&mut ctxs.bcw_idx[slot], bin0).unwrap();
        if bin0 == 0 {
            return;
        }
        // bypass tail
        let mut i = 1u32;
        while i < cmax {
            let bit = if i < value { 1 } else { 0 };
            enc.encode_bypass(bit).unwrap();
            if bit == 0 {
                break;
            }
            i += 1;
        }
    }

    fn bcw_idx_round_trip(value: u32, no_backward_pred_flag: bool, init_type: u8) -> u32 {
        let mut enc = ArithEncoder::new();
        let mut enc_ctxs = LeafCuCtxs::init_with_init_type(26, init_type);
        encode_bcw_idx(&mut enc, &mut enc_ctxs, value, no_backward_pred_flag);
        enc.encode_terminate(1).unwrap();
        let mut padded = enc.finish();
        padded.extend_from_slice(&[0u8; 32]);
        let mut dec = ArithDecoder::new(&padded).unwrap();
        let mut dec_ctxs = LeafCuCtxs::init_with_init_type(26, init_type);
        let tools = CuToolFlags::default();
        let mut reader = LeafCuReader::new(&mut dec, &mut dec_ctxs, tools);
        reader.read_bcw_idx(no_backward_pred_flag).unwrap()
    }

    #[test]
    fn bcw_idx_b_slice_round_trips_all_values_cmax_2() {
        // NoBackwardPredFlag == 0 ⇒ cMax = 2. Three legal values
        // {0, 1, 2}. value 0 = bin0=0; value 1 = bin0=1, bypass=0;
        // value 2 = bin0=1, bypass=1 (truncation point, no trailing
        // zero). Cover both non-I initTypes (1 = P, 2 = B).
        for it in [1u8, 2] {
            for v in 0..=2u32 {
                assert_eq!(
                    bcw_idx_round_trip(v, false, it),
                    v,
                    "cMax=2 round trip failed for v={v} init={it}"
                );
            }
        }
    }

    #[test]
    fn bcw_idx_no_backward_pred_round_trips_all_values_cmax_4() {
        // NoBackwardPredFlag == 1 ⇒ cMax = 4. Five legal values
        // {0, 1, 2, 3, 4}. value 4 hits the truncation point — four
        // bypass-1 bins with no terminating zero.
        for it in [1u8, 2] {
            for v in 0..=4u32 {
                assert_eq!(
                    bcw_idx_round_trip(v, true, it),
                    v,
                    "cMax=4 round trip failed for v={v} init={it}"
                );
            }
        }
    }

    #[test]
    fn bcw_idx_value_zero_reads_exactly_one_ctx_bin() {
        // Spec sanity: value 0 corresponds to `bin0 = 0`, no bypass
        // tail. Build a stream with exactly that one bin (no termination
        // emitted past the single context decision) — the reader must
        // still return 0 and consume no bypass bits.
        let init_type = 2;
        let mut enc = ArithEncoder::new();
        let mut enc_ctxs = LeafCuCtxs::init_with_init_type(26, init_type);
        let slot = (init_type as usize - 1).min(enc_ctxs.bcw_idx.len() - 1);
        enc.encode_decision(&mut enc_ctxs.bcw_idx[slot], 0).unwrap();
        // Mark a sentinel bypass bit AFTER the bcw_idx so we can prove
        // the reader did not consume it.
        enc.encode_bypass(1).unwrap();
        enc.encode_terminate(1).unwrap();
        let mut padded = enc.finish();
        padded.extend_from_slice(&[0u8; 32]);

        let mut dec = ArithDecoder::new(&padded).unwrap();
        let mut dec_ctxs = LeafCuCtxs::init_with_init_type(26, init_type);
        let tools = CuToolFlags::default();
        let mut reader = LeafCuReader::new(&mut dec, &mut dec_ctxs, tools);
        let bcw = reader.read_bcw_idx(false).unwrap();
        assert_eq!(bcw, 0);
        // The sentinel bypass bit must still be in the stream — confirm
        // by reading it via the decoder directly.
        let sentinel = reader.dec.decode_bypass().unwrap();
        assert_eq!(
            sentinel, 1,
            "read_bcw_idx must not consume bypass bins when bin0 == 0"
        );
    }

    #[test]
    fn bcw_idx_ctx_inc_is_fixed_zero() {
        // Table 132 pins bin 0's ctxInc to 0; the helper must agree.
        assert_eq!(crate::ctx::ctx_inc_bcw_idx(0), 0);
    }

    #[test]
    fn bcw_idx_table_91_init_matches_spec() {
        // Round-126 transcription: initValue = [4, 5], shiftIdx = [1, 1].
        assert_eq!(crate::tables::BCW_IDX_INIT, &[4u8, 5]);
        assert_eq!(crate::tables::BCW_IDX_SHIFT, &[1u8, 1]);
        assert_eq!(crate::tables::ctx_count(SyntaxCtx::BcwIdx), 2);
    }

    #[test]
    fn bcw_idx_per_inittype_slot_isolation() {
        // P-slice (init 1) and B-slice (init 2) must address different
        // ctxIdx slots so a single CLVS replay does not bleed pState
        // updates between the two slice-type initialisation bundles.
        // The bundle is 2 entries — slot 0 is initType 1 (P), slot 1
        // is initType 2 (B) per Table 51.
        let p_ctxs = LeafCuCtxs::init_with_init_type(26, 1);
        let b_ctxs = LeafCuCtxs::init_with_init_type(26, 2);
        assert_eq!(p_ctxs.bcw_idx.len(), 2);
        assert_eq!(b_ctxs.bcw_idx.len(), 2);
        // Round-trip every legal value on both initTypes to prove the
        // slot wiring keeps the two bundles independent (any mis-
        // address would cause one initType's CABAC state to drift
        // against the encoder's mirror and corrupt the result).
        for v in 0..=2u32 {
            assert_eq!(bcw_idx_round_trip(v, false, 1), v);
            assert_eq!(bcw_idx_round_trip(v, false, 2), v);
        }
    }

    // ---- Round-129: §7.3.10.5 bcw_idx gate evaluator + MvField fuse ----

    /// Build a fully-open gate: all five "weighted-prediction off"
    /// preconditions met, `inter_pred_idc == PRED_BI`, SPS bit set,
    /// CU 16x16 = 256 luma samples (the boundary).
    fn open_gate() -> BcwIdxGate {
        BcwIdxGate {
            sps_bcw_enabled: true,
            inter_pred_idc: Some(InterPredDir::PredBi),
            luma_weight_l0_flag: false,
            luma_weight_l1_flag: false,
            chroma_weight_l0_flag: false,
            chroma_weight_l1_flag: false,
            cb_width: 16,
            cb_height: 16,
            no_backward_pred_flag: false,
        }
    }

    #[test]
    fn bcw_idx_gate_open_when_all_conditions_met() {
        let g = open_gate();
        assert!(g.is_open());
    }

    #[test]
    fn bcw_idx_gate_closes_on_sps_disable() {
        let mut g = open_gate();
        g.sps_bcw_enabled = false;
        assert!(!g.is_open());
    }

    #[test]
    fn bcw_idx_gate_closes_on_uni_pred() {
        // PRED_L0 and PRED_L1 both close the gate (only PRED_BI opens).
        let mut g = open_gate();
        g.inter_pred_idc = Some(InterPredDir::PredL0);
        assert!(!g.is_open());
        g.inter_pred_idc = Some(InterPredDir::PredL1);
        assert!(!g.is_open());
        // None (no inter_pred_idc decoded yet) also closes.
        g.inter_pred_idc = None;
        assert!(!g.is_open());
    }

    #[test]
    fn bcw_idx_gate_closes_on_any_weighted_prediction_flag() {
        // Each of the four weighted-prediction flags closes the gate
        // individually — there's no "AND" relaxation in the §7.3.10.5
        // conditional.
        for set in [
            |g: &mut BcwIdxGate| g.luma_weight_l0_flag = true,
            |g: &mut BcwIdxGate| g.luma_weight_l1_flag = true,
            |g: &mut BcwIdxGate| g.chroma_weight_l0_flag = true,
            |g: &mut BcwIdxGate| g.chroma_weight_l1_flag = true,
        ] {
            let mut g = open_gate();
            set(&mut g);
            assert!(!g.is_open(), "any wp flag should close the gate");
        }
    }

    #[test]
    fn bcw_idx_gate_area_threshold_is_inclusive_at_256() {
        // cbWidth * cbHeight >= 256 — the spec uses `>=`, so the 256-
        // sample minimum (e.g. 16x16, 8x32, 32x8) opens; 8x16 = 128 and
        // 16x8 = 128 close.
        for (w, h, expect) in [
            (16u32, 16u32, true),
            (8, 32, true),
            (32, 8, true),
            (8, 16, false),
            (16, 8, false),
            (8, 8, false),
            (32, 16, true),
            (4, 64, true),
            (4, 32, false),
        ] {
            let mut g = open_gate();
            g.cb_width = w;
            g.cb_height = h;
            assert_eq!(g.is_open(), expect, "gate for {w}x{h} ({} samples)", w * h);
        }
    }

    #[test]
    fn read_bcw_idx_gated_returns_zero_without_consuming_when_closed() {
        // Gate closed ⇒ no bins consumed ⇒ value inferred 0 per
        // §7.4.12.5. Prove the bitstream pointer hasn't moved by
        // letting the reader bypass-decode a sentinel bit right after.
        let init_type = 2;
        let mut enc = ArithEncoder::new();
        let _ctxs_for_enc = LeafCuCtxs::init_with_init_type(26, init_type);
        // Sentinel-only stream: just a single bypass `1`.
        enc.encode_bypass(1).unwrap();
        enc.encode_terminate(1).unwrap();
        let mut padded = enc.finish();
        padded.extend_from_slice(&[0u8; 32]);
        let mut dec = ArithDecoder::new(&padded).unwrap();
        let mut dec_ctxs = LeafCuCtxs::init_with_init_type(26, init_type);
        let tools = CuToolFlags::default();
        let mut reader = LeafCuReader::new(&mut dec, &mut dec_ctxs, tools);

        // Closed gate: any of the §7.3.10.5 preconditions failing.
        let mut g = open_gate();
        g.sps_bcw_enabled = false;
        assert_eq!(reader.read_bcw_idx_gated(g).unwrap(), 0);
        // Stream still parked at the sentinel bypass bit.
        assert_eq!(reader.dec.decode_bypass().unwrap(), 1);
    }

    #[test]
    fn read_bcw_idx_gated_reads_when_open() {
        // Gate open with `no_backward_pred_flag = false` (cMax = 2).
        // Encode value 1 (bin0=1, bypass=0) and verify the gated read
        // returns it.
        let init_type = 2;
        let mut enc = ArithEncoder::new();
        let mut enc_ctxs = LeafCuCtxs::init_with_init_type(26, init_type);
        encode_bcw_idx(&mut enc, &mut enc_ctxs, 1, false);
        enc.encode_terminate(1).unwrap();
        let mut padded = enc.finish();
        padded.extend_from_slice(&[0u8; 32]);
        let mut dec = ArithDecoder::new(&padded).unwrap();
        let mut dec_ctxs = LeafCuCtxs::init_with_init_type(26, init_type);
        let tools = CuToolFlags::default();
        let mut reader = LeafCuReader::new(&mut dec, &mut dec_ctxs, tools);
        assert_eq!(reader.read_bcw_idx_gated(open_gate()).unwrap(), 1);
    }

    #[test]
    fn read_bcw_idx_into_writes_decoded_value_into_mvfield() {
        // Round-trip value 2 with cMax = 2 (max-truncation point) and
        // confirm the MvField slot is updated in place.
        let init_type = 2;
        let mut enc = ArithEncoder::new();
        let mut enc_ctxs = LeafCuCtxs::init_with_init_type(26, init_type);
        encode_bcw_idx(&mut enc, &mut enc_ctxs, 2, false);
        enc.encode_terminate(1).unwrap();
        let mut padded = enc.finish();
        padded.extend_from_slice(&[0u8; 32]);
        let mut dec = ArithDecoder::new(&padded).unwrap();
        let mut dec_ctxs = LeafCuCtxs::init_with_init_type(26, init_type);
        let tools = CuToolFlags::default();
        let mut reader = LeafCuReader::new(&mut dec, &mut dec_ctxs, tools);

        let mut mvf = MvField::UNAVAILABLE;
        // Seed with a non-zero "stale" value to prove the helper
        // actually writes (not merely defaults).
        mvf.bcw_idx = 7;
        let returned = reader.read_bcw_idx_into(open_gate(), &mut mvf).unwrap();
        assert_eq!(returned, 2);
        assert_eq!(mvf.bcw_idx, 2);
    }

    #[test]
    fn read_bcw_idx_into_clears_stale_when_gate_closed() {
        // Closed gate ⇒ MvField.bcw_idx must be set to 0 per §7.4.12.5
        // (the spec's "inferred to be equal to 0" rule applies to the
        // *array slot*, not just the local variable — the CTU writer
        // broadcasts this 0 across every 4x4 covered block).
        let init_type = 2;
        let mut enc = ArithEncoder::new();
        enc.encode_terminate(1).unwrap();
        let mut padded = enc.finish();
        padded.extend_from_slice(&[0u8; 32]);
        let mut dec = ArithDecoder::new(&padded).unwrap();
        let mut dec_ctxs = LeafCuCtxs::init_with_init_type(26, init_type);
        let tools = CuToolFlags::default();
        let mut reader = LeafCuReader::new(&mut dec, &mut dec_ctxs, tools);

        let mut mvf = MvField::UNAVAILABLE;
        mvf.bcw_idx = 3;
        let mut g = open_gate();
        g.cb_width = 8;
        g.cb_height = 16; // 128 < 256 ⇒ closed
        let returned = reader.read_bcw_idx_into(g, &mut mvf).unwrap();
        assert_eq!(returned, 0);
        assert_eq!(mvf.bcw_idx, 0);
    }

    #[test]
    fn bcw_idx_gate_threads_no_backward_pred_flag_into_cmax() {
        // When `no_backward_pred_flag = true` the reader must see
        // `cMax = 4`, allowing values up to 4. Confirm via end-to-end
        // gated round-trip of value 4 (the truncation-point max).
        let init_type = 1; // P-slice with NoBackwardPredFlag = 1
        let mut enc = ArithEncoder::new();
        let mut enc_ctxs = LeafCuCtxs::init_with_init_type(26, init_type);
        encode_bcw_idx(&mut enc, &mut enc_ctxs, 4, true);
        enc.encode_terminate(1).unwrap();
        let mut padded = enc.finish();
        padded.extend_from_slice(&[0u8; 32]);
        let mut dec = ArithDecoder::new(&padded).unwrap();
        let mut dec_ctxs = LeafCuCtxs::init_with_init_type(26, init_type);
        let tools = CuToolFlags::default();
        let mut reader = LeafCuReader::new(&mut dec, &mut dec_ctxs, tools);
        let mut g = open_gate();
        g.no_backward_pred_flag = true;
        assert_eq!(reader.read_bcw_idx_gated(g).unwrap(), 4);
    }

    // ---- Round-139: §7.3.11.7 `merge_subblock_flag` + `merge_subblock_idx`
    // reader round-trips ----

    /// Encoder mirror of `read_merge_subblock_flag` driving the same
    /// context bundle bin-for-bin per §9.3.4.2.2 / eq. 1551 with the
    /// Table 133 merge-side row. Used by the round-trip tests below.
    fn encode_merge_subblock_flag(
        enc: &mut ArithEncoder,
        ctxs: &mut LeafCuCtxs,
        flag: bool,
        left_merge_subblock: bool,
        left_inter_affine: bool,
        left_available: bool,
        above_merge_subblock: bool,
        above_inter_affine: bool,
        above_available: bool,
    ) {
        let inc = crate::ctx::ctx_inc_merge_subblock_flag(
            left_merge_subblock,
            left_inter_affine,
            left_available,
            above_merge_subblock,
            above_inter_affine,
            above_available,
        ) as usize;
        let init_off = (ctxs.init_type as usize).saturating_sub(1) * 3;
        let n = ctxs.merge_subblock_flag.len() - 1;
        let slot = (init_off + inc).min(n);
        let bit = if flag { 1 } else { 0 };
        enc.encode_decision(&mut ctxs.merge_subblock_flag[slot], bit)
            .unwrap();
    }

    /// Encoder mirror of `read_merge_subblock_idx` driving the same
    /// context bundle bin-for-bin per Table 108 / Table 132.
    fn encode_merge_subblock_idx(
        enc: &mut ArithEncoder,
        ctxs: &mut LeafCuCtxs,
        value: u32,
        max_num_subblock_merge_cand: u32,
    ) {
        if max_num_subblock_merge_cand <= 1 {
            // §7.3.11.7 suppression — nothing on the wire.
            return;
        }
        let cmax = max_num_subblock_merge_cand - 1;
        let block = (ctxs.init_type as usize).saturating_sub(1);
        let n = ctxs.merge_subblock_idx.len();
        let slot = block.min(n - 1);
        let bin0 = if value == 0 { 0 } else { 1 };
        enc.encode_decision(&mut ctxs.merge_subblock_idx[slot], bin0)
            .unwrap();
        if bin0 == 0 {
            return;
        }
        // Bypass tail.
        let mut i = 1u32;
        while i < cmax {
            let bit = if i < value { 1 } else { 0 };
            enc.encode_bypass(bit).unwrap();
            if bit == 0 {
                break;
            }
            i += 1;
        }
    }

    /// Per-bin round-trip helper for `merge_subblock_flag`.
    fn merge_subblock_flag_round_trip(
        flag: bool,
        init_type: u8,
        left_merge_subblock: bool,
        left_inter_affine: bool,
        left_available: bool,
        above_merge_subblock: bool,
        above_inter_affine: bool,
        above_available: bool,
    ) -> bool {
        let mut enc = ArithEncoder::new();
        let mut enc_ctxs = LeafCuCtxs::init_with_init_type(26, init_type);
        encode_merge_subblock_flag(
            &mut enc,
            &mut enc_ctxs,
            flag,
            left_merge_subblock,
            left_inter_affine,
            left_available,
            above_merge_subblock,
            above_inter_affine,
            above_available,
        );
        enc.encode_terminate(1).unwrap();
        let mut padded = enc.finish();
        padded.extend_from_slice(&[0u8; 32]);
        let mut dec = ArithDecoder::new(&padded).unwrap();
        let mut dec_ctxs = LeafCuCtxs::init_with_init_type(26, init_type);
        let tools = CuToolFlags::default();
        let mut reader = LeafCuReader::new(&mut dec, &mut dec_ctxs, tools);
        reader
            .read_merge_subblock_flag(
                left_merge_subblock,
                left_inter_affine,
                left_available,
                above_merge_subblock,
                above_inter_affine,
                above_available,
            )
            .unwrap()
    }

    /// Per-value round-trip helper for `merge_subblock_idx`.
    fn merge_subblock_idx_round_trip(
        value: u32,
        max_num_subblock_merge_cand: u32,
        init_type: u8,
    ) -> u32 {
        let mut enc = ArithEncoder::new();
        let mut enc_ctxs = LeafCuCtxs::init_with_init_type(26, init_type);
        encode_merge_subblock_idx(&mut enc, &mut enc_ctxs, value, max_num_subblock_merge_cand);
        enc.encode_terminate(1).unwrap();
        let mut padded = enc.finish();
        padded.extend_from_slice(&[0u8; 32]);
        let mut dec = ArithDecoder::new(&padded).unwrap();
        let mut dec_ctxs = LeafCuCtxs::init_with_init_type(26, init_type);
        let tools = CuToolFlags::default();
        let mut reader = LeafCuReader::new(&mut dec, &mut dec_ctxs, tools);
        reader
            .read_merge_subblock_idx(max_num_subblock_merge_cand)
            .unwrap()
    }

    #[test]
    fn merge_subblock_flag_round_trips_no_neighbours_both_init_types() {
        // No neighbour info (cond_l = cond_a = 0). Both P-slice (init 1)
        // and B-slice (init 2) initTypes round-trip the flag value.
        for it in [1u8, 2] {
            for flag in [false, true] {
                assert_eq!(
                    merge_subblock_flag_round_trip(
                        flag, it, false, false, true, false, false, true,
                    ),
                    flag,
                    "merge_subblock_flag round trip failed at init={it} flag={flag}"
                );
            }
        }
    }

    #[test]
    fn merge_subblock_flag_round_trips_with_one_active_neighbour() {
        // Left neighbour was a sub-block-merge or affine CU → cond_l = 1
        // → ctxInc = 1.
        for flag in [false, true] {
            assert_eq!(
                merge_subblock_flag_round_trip(
                    flag, 2, // B-slice
                    true, false, true, // left MergeSubblockFlag, available
                    false, false, true,
                ),
                flag
            );
            // Above neighbour was an affine CU → cond_a = 1 → ctxInc = 1.
            assert_eq!(
                merge_subblock_flag_round_trip(flag, 2, false, false, true, false, true, true,),
                flag
            );
        }
    }

    #[test]
    fn merge_subblock_flag_round_trips_with_both_active_neighbours() {
        // Both neighbours contribute → ctxInc = 2 (max).
        for flag in [false, true] {
            for it in [1u8, 2] {
                assert_eq!(
                    merge_subblock_flag_round_trip(flag, it, true, false, true, false, true, true,),
                    flag
                );
            }
        }
    }

    #[test]
    fn merge_subblock_flag_unavailable_neighbour_masks_active_flags() {
        // MergeSubblockFlag set on left but available_l = false should
        // produce ctxInc = 0 → still round-trips through that slot.
        for flag in [false, true] {
            assert_eq!(
                merge_subblock_flag_round_trip(flag, 2, true, true, false, false, false, true,),
                flag
            );
        }
    }

    #[test]
    fn merge_subblock_idx_round_trips_max_cand_2() {
        // MaxNumSubblockMergeCand = 2 → cMax = 1 → only the ctx bin
        // (value 0 vs 1) is on the wire.
        for it in [1u8, 2] {
            for v in 0..=1u32 {
                assert_eq!(
                    merge_subblock_idx_round_trip(v, 2, it),
                    v,
                    "merge_subblock_idx round trip failed at init={it} v={v} cMax=1"
                );
            }
        }
    }

    #[test]
    fn merge_subblock_idx_round_trips_full_range_cand_5() {
        // MaxNumSubblockMergeCand = 5 (clip cap of eq. 85) → cMax = 4
        // → values 0..=4, with value 4 hitting the TR truncation point
        // (four bins, no trailing zero).
        for it in [1u8, 2] {
            for v in 0..=4u32 {
                assert_eq!(
                    merge_subblock_idx_round_trip(v, 5, it),
                    v,
                    "merge_subblock_idx round trip failed at init={it} v={v} cMax=4"
                );
            }
        }
    }

    #[test]
    fn merge_subblock_idx_suppressed_when_max_cand_le_1() {
        // §7.3.11.7 only emits `merge_subblock_idx` when
        // `MaxNumSubblockMergeCand > 1`. With 0 or 1 candidate, the
        // reader returns 0 without consuming any bits — prove this by
        // letting a sentinel bypass bit survive past the call.
        for max_cand in [0u32, 1] {
            let init_type = 2;
            let mut enc = ArithEncoder::new();
            // Sentinel-only stream: a single bypass `1`.
            enc.encode_bypass(1).unwrap();
            enc.encode_terminate(1).unwrap();
            let mut padded = enc.finish();
            padded.extend_from_slice(&[0u8; 32]);
            let mut dec = ArithDecoder::new(&padded).unwrap();
            let mut dec_ctxs = LeafCuCtxs::init_with_init_type(26, init_type);
            let tools = CuToolFlags::default();
            let mut reader = LeafCuReader::new(&mut dec, &mut dec_ctxs, tools);
            assert_eq!(reader.read_merge_subblock_idx(max_cand).unwrap(), 0);
            // Sentinel bit must still be on the wire.
            assert_eq!(reader.dec.decode_bypass().unwrap(), 1);
        }
    }

    #[test]
    fn merge_subblock_idx_value_zero_reads_exactly_one_ctx_bin() {
        // Spec sanity: value 0 corresponds to `bin0 = 0`, no bypass
        // tail. Encode exactly that bin then a sentinel bypass-1; the
        // reader must return 0 and leave the sentinel for the next read.
        let init_type = 2;
        let mut enc = ArithEncoder::new();
        let mut enc_ctxs = LeafCuCtxs::init_with_init_type(26, init_type);
        let slot = (init_type as usize - 1).min(enc_ctxs.merge_subblock_idx.len() - 1);
        enc.encode_decision(&mut enc_ctxs.merge_subblock_idx[slot], 0)
            .unwrap();
        enc.encode_bypass(1).unwrap();
        enc.encode_terminate(1).unwrap();
        let mut padded = enc.finish();
        padded.extend_from_slice(&[0u8; 32]);
        let mut dec = ArithDecoder::new(&padded).unwrap();
        let mut dec_ctxs = LeafCuCtxs::init_with_init_type(26, init_type);
        let tools = CuToolFlags::default();
        let mut reader = LeafCuReader::new(&mut dec, &mut dec_ctxs, tools);
        // cMax = 4 chosen to make sure the reader doesn't accidentally
        // consume the sentinel as part of the bypass tail.
        assert_eq!(reader.read_merge_subblock_idx(5).unwrap(), 0);
        assert_eq!(reader.dec.decode_bypass().unwrap(), 1);
    }

    #[test]
    fn merge_subblock_flag_per_ctx_slots_are_addressable() {
        // Defensive: confirm that for every legal (init_type, ctxInc)
        // pair the slot index lands inside the per-Table 107 6-entry
        // bundle. This guards against future expansions of the table or
        // a stale clamp.
        for it in [1u8, 2] {
            let ctxs = LeafCuCtxs::init_with_init_type(26, it);
            let init_off = (it as usize - 1) * 3;
            for inc in 0..=2usize {
                let slot = init_off + inc;
                assert!(
                    slot < ctxs.merge_subblock_flag.len(),
                    "slot {slot} out of range for init_type {it}"
                );
            }
        }
    }

    // ---- Round-152: §7.3.11.7 `inter_affine_flag` reader round-trips ----
    //
    // The encoder mirror duplicates the reader's CABAC bin emission so a
    // round-trip can be staged without a live encoder pipeline. The same
    // pattern is used by the round-139 `merge_subblock_flag` tests above.

    /// Encoder mirror of `read_inter_affine_flag` driving the same
    /// context bundle bin-for-bin per §9.3.4.2.2 / eq. 1551 with the
    /// Table 133 row identical to `merge_subblock_flag` and the Table 84
    /// per-initType slot.
    fn encode_inter_affine_flag(
        enc: &mut ArithEncoder,
        ctxs: &mut LeafCuCtxs,
        flag: bool,
        left_merge_subblock: bool,
        left_inter_affine: bool,
        left_available: bool,
        above_merge_subblock: bool,
        above_inter_affine: bool,
        above_available: bool,
    ) {
        let inc = crate::ctx::ctx_inc_inter_affine_flag(
            left_merge_subblock,
            left_inter_affine,
            left_available,
            above_merge_subblock,
            above_inter_affine,
            above_available,
        ) as usize;
        let init_off = (ctxs.init_type as usize).saturating_sub(1) * 3;
        let n = ctxs.inter_affine_flag.len() - 1;
        let slot = (init_off + inc).min(n);
        let bit = if flag { 1 } else { 0 };
        enc.encode_decision(&mut ctxs.inter_affine_flag[slot], bit)
            .unwrap();
    }

    /// Per-bin round-trip helper for `inter_affine_flag`. Builds a
    /// fresh CABAC stream with one bin and reads it back through the
    /// matching context slot.
    #[allow(clippy::too_many_arguments)]
    fn inter_affine_flag_round_trip(
        flag: bool,
        init_type: u8,
        left_merge_subblock: bool,
        left_inter_affine: bool,
        left_available: bool,
        above_merge_subblock: bool,
        above_inter_affine: bool,
        above_available: bool,
    ) -> bool {
        let mut enc = ArithEncoder::new();
        let mut enc_ctxs = LeafCuCtxs::init_with_init_type(26, init_type);
        encode_inter_affine_flag(
            &mut enc,
            &mut enc_ctxs,
            flag,
            left_merge_subblock,
            left_inter_affine,
            left_available,
            above_merge_subblock,
            above_inter_affine,
            above_available,
        );
        enc.encode_terminate(1).unwrap();
        let mut padded = enc.finish();
        padded.extend_from_slice(&[0u8; 32]);
        let mut dec = ArithDecoder::new(&padded).unwrap();
        let mut dec_ctxs = LeafCuCtxs::init_with_init_type(26, init_type);
        let tools = CuToolFlags::default();
        let mut reader = LeafCuReader::new(&mut dec, &mut dec_ctxs, tools);
        reader
            .read_inter_affine_flag(
                left_merge_subblock,
                left_inter_affine,
                left_available,
                above_merge_subblock,
                above_inter_affine,
                above_available,
            )
            .unwrap()
    }

    #[test]
    fn inter_affine_flag_round_trips_no_neighbours_both_init_types() {
        // No neighbour info (cond_l = cond_a = 0). Both P-slice (init 1)
        // and B-slice (init 2) initTypes round-trip the flag value.
        for it in [1u8, 2] {
            for flag in [false, true] {
                assert_eq!(
                    inter_affine_flag_round_trip(flag, it, false, false, true, false, false, true,),
                    flag,
                    "inter_affine_flag round trip failed at init={it} flag={flag}"
                );
            }
        }
    }

    #[test]
    fn inter_affine_flag_round_trips_with_one_active_neighbour() {
        // Left neighbour was an affine inter CU → cond_l = 1 → ctxInc = 1.
        for flag in [false, true] {
            assert_eq!(
                inter_affine_flag_round_trip(
                    flag, 2, // B-slice
                    false, true, true, // left InterAffineFlag, available
                    false, false, true,
                ),
                flag
            );
            // Above neighbour was a sub-block-merge CU → cond_a = 1.
            assert_eq!(
                inter_affine_flag_round_trip(flag, 2, false, false, true, true, false, true,),
                flag
            );
        }
    }

    #[test]
    fn inter_affine_flag_round_trips_with_both_active_neighbours() {
        // Both neighbours contribute → ctxInc = 2 (max).
        for flag in [false, true] {
            for it in [1u8, 2] {
                assert_eq!(
                    inter_affine_flag_round_trip(flag, it, true, false, true, false, true, true,),
                    flag
                );
            }
        }
    }

    #[test]
    fn inter_affine_flag_unavailable_neighbour_masks_active_flags() {
        // InterAffineFlag set on left but available_l = false should
        // contribute 0 to ctxInc — the round-trip still works.
        for flag in [false, true] {
            assert_eq!(
                inter_affine_flag_round_trip(
                    flag, 2, false, true, false, // left affine but unavail
                    false, true, true, // above affine + avail
                ),
                flag
            );
        }
    }

    #[test]
    fn inter_affine_flag_per_ctx_slots_are_addressable() {
        // Defensive: confirm that for every legal (init_type, ctxInc)
        // pair the slot index lands inside the per-Table 84 6-entry
        // bundle. Mirrors `merge_subblock_flag_per_ctx_slots_are_addressable`
        // because the two tables share shape.
        for it in [1u8, 2] {
            let ctxs = LeafCuCtxs::init_with_init_type(26, it);
            let init_off = (it as usize - 1) * 3;
            for inc in 0..=2usize {
                let slot = init_off + inc;
                assert!(
                    slot < ctxs.inter_affine_flag.len(),
                    "slot {slot} out of range for init_type {it}"
                );
            }
        }
    }

    #[test]
    fn inter_affine_flag_ctx_bundle_disjoint_from_merge_subblock_flag() {
        // The reader uses *separate* CABAC state machines for the two
        // syntax elements (Table 84 vs Table 107). Confirm the two
        // bundles are independent vectors of contexts — flipping one
        // must not perturb the other's `pState`.
        let mut ctxs = LeafCuCtxs::init_with_init_type(26, 1);
        let baseline_subblock: Vec<_> = ctxs
            .merge_subblock_flag
            .iter()
            .map(|c| (c.p_state_idx0, c.p_state_idx1, c.shift_idx))
            .collect();
        let baseline_affine: Vec<_> = ctxs
            .inter_affine_flag
            .iter()
            .map(|c| (c.p_state_idx0, c.p_state_idx1, c.shift_idx))
            .collect();
        // Sanity: the two tables must NOT initialise to identical
        // pState pairs — Table 84 and Table 107 carry different
        // initValue / shiftIdx rows per the spec transcription.
        assert_ne!(
            baseline_subblock, baseline_affine,
            "Tables 84 and 107 should initialise to different ctx state"
        );
        // Drive a couple of decisions through one bundle and confirm
        // the other is untouched.
        let mut enc = ArithEncoder::new();
        enc.encode_decision(&mut ctxs.inter_affine_flag[0], 1)
            .unwrap();
        enc.encode_decision(&mut ctxs.inter_affine_flag[1], 0)
            .unwrap();
        let after_subblock: Vec<_> = ctxs
            .merge_subblock_flag
            .iter()
            .map(|c| (c.p_state_idx0, c.p_state_idx1, c.shift_idx))
            .collect();
        assert_eq!(
            after_subblock, baseline_subblock,
            "driving inter_affine_flag must not perturb merge_subblock_flag state"
        );
    }

    // ---- Round-146: §7.3.11.7 merge_data() wire-up of
    // merge_subblock_flag / merge_subblock_idx ----

    /// Build a hand-rolled CABAC payload that drives `decode_inter()`
    /// through the subblock-merge prologue: cu_skip_flag = 0,
    /// general_merge_flag = 1, then merge_subblock_flag plus
    /// (optionally) merge_subblock_idx. The CU is non-skip so the
    /// trailing cu_coded_flag(0) is also encoded.
    fn build_subblock_merge_payload(
        slice_qp: i32,
        init_type: u8,
        merge_subblock_flag: bool,
        merge_subblock_idx: Option<(u32, u32)>, // (value, max_num_subblock_merge_cand)
    ) -> Vec<u8> {
        use crate::cabac_enc::ArithEncoder;
        use crate::ctx::{
            ctx_inc_cu_skip_flag, ctx_inc_general_merge_flag, ctx_inc_merge_subblock_flag,
        };

        let mut enc = ArithEncoder::new();
        let mut ctxs = LeafCuCtxs::init_with_init_type(slice_qp, init_type);

        // cu_skip_flag(0). ctxInc derivation: no neighbours available.
        let skip_inc = ctx_inc_cu_skip_flag(false, false, false, false) as usize;
        let skip_slot = (init_type as usize) * 3 + skip_inc;
        enc.encode_decision(&mut ctxs.cu_skip_flag[skip_slot], 0)
            .unwrap();

        // general_merge_flag(1).
        let gm_inc = ctx_inc_general_merge_flag() as usize;
        let gm_n = ctxs.general_merge_flag.len() - 1;
        let gm_slot = ((init_type as usize) * 3 + gm_inc).min(gm_n);
        enc.encode_decision(&mut ctxs.general_merge_flag[gm_slot], 1)
            .unwrap();

        // merge_subblock_flag — Table 107 / §9.3.4.2.2 ctxInc with no
        // neighbours (all-false).
        let inc = ctx_inc_merge_subblock_flag(false, false, false, false, false, false) as usize;
        let init_off = (init_type as usize).saturating_sub(1) * 3;
        let n = ctxs.merge_subblock_flag.len() - 1;
        let sb_slot = (init_off + inc).min(n);
        let sb_bit = if merge_subblock_flag { 1 } else { 0 };
        enc.encode_decision(&mut ctxs.merge_subblock_flag[sb_slot], sb_bit)
            .unwrap();

        if merge_subblock_flag {
            if let Some((value, max_cand)) = merge_subblock_idx {
                encode_merge_subblock_idx(&mut enc, &mut ctxs, value, max_cand);
            }
            // cu_coded_flag(0) — trailing bin for non-skip merge CU.
            let cu_n = ctxs.cu_coded_flag.len() - 1;
            let cu_slot = (init_type as usize).min(cu_n);
            enc.encode_decision(&mut ctxs.cu_coded_flag[cu_slot], 0)
                .unwrap();
        }

        enc.encode_terminate(1).unwrap();
        let mut padded = enc.finish();
        padded.extend_from_slice(&[0u8; 32]);
        padded
    }

    fn tools_for_subblock_merge(max_num_subblock_merge_cand: u32) -> CuToolFlags {
        CuToolFlags {
            chroma_format_idc: 1,
            ctb_size_y: 128,
            max_tb_size_y: 64,
            min_tb_size_y: 4,
            max_ts_size: 32,
            slice_is_inter: true,
            max_num_merge_cand: 6,
            max_num_subblock_merge_cand,
            ..CuToolFlags::default()
        }
    }

    /// Round-146: §7.3.11.7 size gate closed (`cbW < 8` or `cbH < 8`)
    /// → `merge_subblock_flag` is NOT parsed and inferred to 0
    /// per §7.4.12.7; the reader falls through to the regular-merge
    /// path. We pick a 4×8 CU so the gate `cbWidth >= 8 && cbHeight
    /// >= 8` is closed by the width side.
    #[test]
    fn merge_subblock_gate_closed_by_cb_width_no_bin_consumed() {
        let slice_qp = 26;
        let init_type = 1u8;
        // Payload built for a *non*-subblock path (only cu_skip + gm).
        use crate::cabac_enc::ArithEncoder;
        use crate::ctx::{ctx_inc_cu_skip_flag, ctx_inc_general_merge_flag};
        let mut enc = ArithEncoder::new();
        let mut ctxs = LeafCuCtxs::init_with_init_type(slice_qp, init_type);
        let skip_inc = ctx_inc_cu_skip_flag(false, false, false, false) as usize;
        let skip_slot = (init_type as usize) * 3 + skip_inc;
        enc.encode_decision(&mut ctxs.cu_skip_flag[skip_slot], 0)
            .unwrap();
        let gm_inc = ctx_inc_general_merge_flag() as usize;
        let gm_n = ctxs.general_merge_flag.len() - 1;
        let gm_slot = ((init_type as usize) * 3 + gm_inc).min(gm_n);
        enc.encode_decision(&mut ctxs.general_merge_flag[gm_slot], 1)
            .unwrap();
        enc.encode_terminate(1).unwrap();
        let mut payload = enc.finish();
        payload.extend_from_slice(&[0u8; 64]);

        let tools = tools_for_subblock_merge(5);
        let mut dec = ArithDecoder::new(&payload).unwrap();
        let mut dec_ctxs = LeafCuCtxs::init_with_init_type(slice_qp, init_type);
        let mut info = LeafCuInfo {
            cb_width: 4,
            cb_height: 8,
            ..LeafCuInfo::default()
        };
        let mut residual = LeafCuResidual::default();
        let neigh = CuNeighbourhood::default();
        let mut reader = LeafCuReader::new(&mut dec, &mut dec_ctxs, tools);
        // Falls through to the regular-merge path. The remaining
        // syntax may surface Unsupported (round-30 cu_coded_flag etc.)
        // — what matters here is the merge_subblock_flag inference.
        let _ = reader.decode(&mut info, &mut residual, &neigh);
        assert!(
            !info.inter.merge_data.merge_subblock_flag,
            "gate closed by cb_width=4 should infer merge_subblock_flag = 0"
        );
        assert_eq!(info.inter.merge_data.merge_subblock_idx, 0);
    }

    /// Round-146: `MaxNumSubblockMergeCand == 0` (eq. 85 produces 0
    /// when both `sps_affine_enabled_flag == 0` and
    /// `sps_sbtmvp_enabled_flag * ph_temporal_mvp_enabled_flag == 0`)
    /// → §7.3.11.7 gate closed; `merge_subblock_flag` not parsed,
    /// inferred to 0.
    #[test]
    fn merge_subblock_gate_closed_by_max_cand_zero() {
        let slice_qp = 26;
        let init_type = 1u8;
        use crate::cabac_enc::ArithEncoder;
        use crate::ctx::{ctx_inc_cu_skip_flag, ctx_inc_general_merge_flag};
        let mut enc = ArithEncoder::new();
        let mut ctxs = LeafCuCtxs::init_with_init_type(slice_qp, init_type);
        let skip_inc = ctx_inc_cu_skip_flag(false, false, false, false) as usize;
        let skip_slot = (init_type as usize) * 3 + skip_inc;
        enc.encode_decision(&mut ctxs.cu_skip_flag[skip_slot], 0)
            .unwrap();
        let gm_inc = ctx_inc_general_merge_flag() as usize;
        let gm_n = ctxs.general_merge_flag.len() - 1;
        let gm_slot = ((init_type as usize) * 3 + gm_inc).min(gm_n);
        enc.encode_decision(&mut ctxs.general_merge_flag[gm_slot], 1)
            .unwrap();
        enc.encode_terminate(1).unwrap();
        let mut payload = enc.finish();
        payload.extend_from_slice(&[0u8; 64]);

        let tools = tools_for_subblock_merge(0);
        let mut dec = ArithDecoder::new(&payload).unwrap();
        let mut dec_ctxs = LeafCuCtxs::init_with_init_type(slice_qp, init_type);
        let mut info = LeafCuInfo {
            cb_width: 16,
            cb_height: 16,
            ..LeafCuInfo::default()
        };
        let mut residual = LeafCuResidual::default();
        let neigh = CuNeighbourhood::default();
        let mut reader = LeafCuReader::new(&mut dec, &mut dec_ctxs, tools);
        let _ = reader.decode(&mut info, &mut residual, &neigh);
        assert!(!info.inter.merge_data.merge_subblock_flag);
        assert_eq!(info.inter.merge_data.merge_subblock_idx, 0);
    }

    /// Round-146: gate OPEN (cbW=cbH=16, MaxNumSubblockMergeCand=5)
    /// and the wire-side `merge_subblock_flag` decodes 0; the reader
    /// must not consume a `merge_subblock_idx` bin and must continue
    /// into the regular-merge tree. We can't easily assert on the
    /// downstream tree outcome here (the all-zero / hand-crafted
    /// stream may surface Unsupported on later syntax), so we just
    /// verify that the wired flag is 0.
    #[test]
    fn merge_subblock_gate_open_flag_zero_falls_through() {
        let slice_qp = 26;
        let init_type = 1u8;
        let payload = build_subblock_merge_payload(slice_qp, init_type, false, None);

        let tools = tools_for_subblock_merge(5);
        let mut dec = ArithDecoder::new(&payload).unwrap();
        let mut dec_ctxs = LeafCuCtxs::init_with_init_type(slice_qp, init_type);
        let mut info = LeafCuInfo {
            cb_width: 16,
            cb_height: 16,
            ..LeafCuInfo::default()
        };
        let mut residual = LeafCuResidual::default();
        let neigh = CuNeighbourhood::default();
        let mut reader = LeafCuReader::new(&mut dec, &mut dec_ctxs, tools);
        let _ = reader.decode(&mut info, &mut residual, &neigh);
        assert!(!info.inter.merge_data.merge_subblock_flag);
        assert_eq!(info.inter.merge_data.merge_subblock_idx, 0);
    }

    /// Round-146: gate OPEN, `merge_subblock_flag = 1`, then the
    /// `merge_subblock_idx` TR bin selects slot 0. The reader takes
    /// the subblock-merge branch and short-circuits the regular tree
    /// (no `regular_merge_flag` bin consumed). Trailing
    /// `cu_coded_flag(0)` lands on `tu_y_coded_flag == false`.
    #[test]
    fn merge_subblock_gate_open_flag_one_idx_zero_decodes_clean() {
        let slice_qp = 26;
        let init_type = 1u8;
        let payload = build_subblock_merge_payload(slice_qp, init_type, true, Some((0, 5)));

        let tools = tools_for_subblock_merge(5);
        let mut dec = ArithDecoder::new(&payload).unwrap();
        let mut dec_ctxs = LeafCuCtxs::init_with_init_type(slice_qp, init_type);
        let mut info = LeafCuInfo {
            cb_width: 16,
            cb_height: 16,
            ..LeafCuInfo::default()
        };
        let mut residual = LeafCuResidual::default();
        let neigh = CuNeighbourhood::default();
        let mut reader = LeafCuReader::new(&mut dec, &mut dec_ctxs, tools);
        reader
            .decode(&mut info, &mut residual, &neigh)
            .expect("subblock-merge path with merge_subblock_idx=0 + cu_coded=0 must decode clean");
        assert!(info.inter.merge_data.merge_subblock_flag);
        assert_eq!(info.inter.merge_data.merge_subblock_idx, 0);
        assert!(!info.inter.merge_data.regular_merge_flag); // not parsed.
        assert!(!info.inter.merge_data.mmvd_merge_flag);
        assert!(!info.inter.merge_data.ciip_flag);
        assert!(!info.inter.merge_data.gpm_flag);
        assert!(!info.tu_y_coded_flag);
        assert!(!info.tu_cb_coded_flag);
        assert!(!info.tu_cr_coded_flag);
    }

    /// Round-146: gate OPEN, `merge_subblock_flag = 1`, with a non-
    /// zero `merge_subblock_idx` (value 3 out of cMax = 4). Exercises
    /// the TR ctx-coded bin0 + bypass tail to drive the
    /// `subblockMergeCandList` slot selector beyond the head.
    #[test]
    fn merge_subblock_gate_open_flag_one_idx_three_decodes_clean() {
        let slice_qp = 26;
        let init_type = 2u8; // also exercise init_type-2 slot offset.
        let payload = build_subblock_merge_payload(slice_qp, init_type, true, Some((3, 5)));

        let tools = tools_for_subblock_merge(5);
        let mut dec = ArithDecoder::new(&payload).unwrap();
        let mut dec_ctxs = LeafCuCtxs::init_with_init_type(slice_qp, init_type);
        let mut info = LeafCuInfo {
            cb_width: 32,
            cb_height: 32,
            ..LeafCuInfo::default()
        };
        let mut residual = LeafCuResidual::default();
        let neigh = CuNeighbourhood::default();
        let mut reader = LeafCuReader::new(&mut dec, &mut dec_ctxs, tools);
        reader
            .decode(&mut info, &mut residual, &neigh)
            .expect("subblock-merge path with merge_subblock_idx=3 must decode clean");
        assert!(info.inter.merge_data.merge_subblock_flag);
        assert_eq!(info.inter.merge_data.merge_subblock_idx, 3);
    }

    /// Round-146: gate OPEN, `merge_subblock_flag = 1`, and
    /// `MaxNumSubblockMergeCand == 1` — §7.3.11.7 suppresses the
    /// `merge_subblock_idx` parse (cMax = 0). The reader's existing
    /// `read_merge_subblock_idx` returns 0 without consuming bits,
    /// matching the §7.4.12.7 inference.
    #[test]
    fn merge_subblock_idx_suppressed_when_max_cand_equals_one() {
        let slice_qp = 26;
        let init_type = 1u8;
        let payload = build_subblock_merge_payload(slice_qp, init_type, true, None);

        let tools = tools_for_subblock_merge(1);
        let mut dec = ArithDecoder::new(&payload).unwrap();
        let mut dec_ctxs = LeafCuCtxs::init_with_init_type(slice_qp, init_type);
        let mut info = LeafCuInfo {
            cb_width: 8,
            cb_height: 8,
            ..LeafCuInfo::default()
        };
        let mut residual = LeafCuResidual::default();
        let neigh = CuNeighbourhood::default();
        let mut reader = LeafCuReader::new(&mut dec, &mut dec_ctxs, tools);
        reader
            .decode(&mut info, &mut residual, &neigh)
            .expect("subblock-merge path with cMax=0 idx-suppression must decode clean");
        assert!(info.inter.merge_data.merge_subblock_flag);
        assert_eq!(info.inter.merge_data.merge_subblock_idx, 0);
    }

    /// Round-146: confirm that `CuToolFlags::default()` keeps
    /// `max_num_subblock_merge_cand == 0` so pre-r146 tests
    /// (slice_is_inter = false / intra-only) never accidentally open
    /// the new gate.
    #[test]
    fn cu_tool_flags_default_keeps_max_num_subblock_merge_cand_zero() {
        let tools = CuToolFlags::default();
        assert_eq!(tools.max_num_subblock_merge_cand, 0);
    }
}
