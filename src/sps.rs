//! VVC Sequence Parameter Set parser (§7.3.2.4).
//!
//! This parser walks the SPS RBSP far enough for slice and picture
//! headers to make state-dependent decisions:
//!
//! * identifier / format / conformance-window / bit-depth / POC /
//!   extra-header-bits (§7.3.2.4 head),
//! * `dpb_parameters()` (§7.3.4) when signalled,
//! * **partition-constraints block** — the min-CB / MTT / dual-tree /
//!   chroma-QP-table fields at §7.3.2.4 tail (spec lines beginning with
//!   `sps_log2_min_luma_coding_block_size_minus2` through
//!   `sps_min_qp_prime_ts`),
//! * **tool-enable flags** (`sps_sao_enabled_flag`, `sps_alf_enabled_flag`,
//!   `sps_transform_skip_enabled_flag`, `sps_mts_enabled_flag`,
//!   `sps_lfnst_enabled_flag`, inter-prediction tool group, IBC /
//!   palette, LADF, explicit scaling list, virtual boundaries, etc.).
//!
//! Still intentionally out of scope for this increment:
//!
//! * `sps_subpic_info_present_flag == 1` streams — the subpic syntax
//!   contains `u(v)` fields whose width depends on `CtbSizeY`; we return
//!   `Error::Unsupported` before touching them (§7.3.2.4, subpic block),
//! * HRD timing, VUI payload, and `sps_extension_*`. The tail reader
//!   stops after the virtual-boundaries block.
//!
//! Round 3 adds `ref_pic_list_struct(listIdx, rplsIdx)` (§7.3.10 /
//! §7.4.11) so SPSes that signal one or more candidate reference-picture
//! lists (ST / LT / ILRP mixtures) can be walked end-to-end.

use oxideav_core::{Error, Result};

use crate::bitreader::BitReader;
use crate::hrd::{
    parse_general_timing_hrd_parameters, parse_ols_timing_hrd_parameters,
    GeneralTimingHrdParameters, OlsTimingHrdParameters,
};
use crate::ptl::{parse_profile_tier_level, ProfileTierLevel};

#[derive(Clone, Debug)]
pub struct SeqParameterSet {
    pub sps_seq_parameter_set_id: u8,
    pub sps_video_parameter_set_id: u8,
    pub sps_max_sublayers_minus1: u8,
    pub sps_chroma_format_idc: u8,
    pub sps_log2_ctu_size_minus5: u8,
    pub sps_ptl_dpb_hrd_params_present_flag: bool,
    pub profile_tier_level: Option<ProfileTierLevel>,
    pub sps_gdr_enabled_flag: bool,
    pub sps_ref_pic_resampling_enabled_flag: bool,
    pub sps_res_change_in_clvs_allowed_flag: bool,
    pub sps_pic_width_max_in_luma_samples: u32,
    pub sps_pic_height_max_in_luma_samples: u32,
    pub conformance_window: Option<ConformanceWindow>,
    pub sps_subpic_info_present_flag: bool,
    pub sps_bitdepth_minus8: u32,
    pub sps_entropy_coding_sync_enabled_flag: bool,
    pub sps_entry_point_offsets_present_flag: bool,
    pub sps_log2_max_pic_order_cnt_lsb_minus4: u8,
    pub sps_poc_msb_cycle_flag: bool,
    pub sps_poc_msb_cycle_len_minus1: u32,
    pub sps_num_extra_ph_bytes: u8,
    pub sps_num_extra_sh_bytes: u8,
    /// §7.4.3.4 eq. 41 `NumExtraPhBits` — the count of
    /// `sps_extra_ph_bit_present_flag[i]` equal to 1. Drives the
    /// `ph_extra_bit[i]` skip loop in the picture header.
    pub num_extra_ph_bits: u8,
    /// `NumExtraShBits` — the count of `sps_extra_sh_bit_present_flag[i]`
    /// equal to 1. Drives the `sh_extra_bit[i]` skip loop in the slice
    /// header (surfaced via `PhState::num_extra_sh_bits`).
    pub num_extra_sh_bits: u8,
    pub sps_sublayer_dpb_params_flag: bool,
    pub dpb_parameters: Option<DpbParameters>,
    pub partition_constraints: PartitionConstraints,
    pub tool_flags: ToolFlags,
    /// Subpicture info block (§7.4.3.4). `None` when
    /// `sps_subpic_info_present_flag == 0`.
    pub subpic_info: Option<SubpicInfo>,
    /// `sps_timing_hrd_params_present_flag` — only transmitted when
    /// `sps_ptl_dpb_hrd_params_present_flag == 1`.
    pub sps_timing_hrd_params_present_flag: bool,
    /// `general_timing_hrd_parameters()` block from §7.3.5.1. Present
    /// when `sps_timing_hrd_params_present_flag == 1`.
    pub general_timing_hrd: Option<GeneralTimingHrdParameters>,
    /// `sps_sublayer_cpb_params_present_flag` — controls `firstSubLayer`
    /// passed to `ols_timing_hrd_parameters()`.
    pub sps_sublayer_cpb_params_present_flag: bool,
    /// `ols_timing_hrd_parameters()` block from §7.3.5.2. Present
    /// alongside `general_timing_hrd`.
    pub ols_timing_hrd: Option<OlsTimingHrdParameters>,
    /// `sps_field_seq_flag` (§7.4.3.4).
    pub sps_field_seq_flag: bool,
    /// `sps_vui_parameters_present_flag` (§7.4.3.4). When set, the
    /// `vui_parameters()` bytes are captured verbatim in
    /// [`Self::vui_payload`] (the reference decoder treats them as
    /// opaque — see ITU-T H.274).
    pub sps_vui_parameters_present_flag: bool,
    /// Raw VUI payload bytes. The first byte starts at
    /// `vui_payload_bit_offset` within the SPS RBSP — kept for
    /// later increments that wire the H.274 parser in.
    pub vui_payload: Vec<u8>,
    /// `sps_extension_flag` (§7.4.3.4).
    pub sps_extension_flag: bool,
    /// `sps_range_extension_flag` — only transmitted under
    /// `sps_extension_flag == 1`.
    pub sps_range_extension_flag: bool,
    /// `sps_extension_7bits` (§7.4.3.4).
    pub sps_extension_7bits: u8,
    /// `sps_range_extension()` block (§7.3.2.22). `Some` exactly when
    /// `sps_range_extension_flag == 1`. Fields default to the
    /// §7.4.3.22 "When not present" inferences when absent.
    pub range_extension: Option<SpsRangeExtension>,
}

/// `sps_range_extension()` payload (§7.3.2.22 + §7.4.3.22).
///
/// The five flags here control transform-coefficient dynamic range,
/// alternate Rice-parameter derivation, persistent Rice adaptation
/// across TUs, and the `sh_reverse_last_sig_coeff_flag` plumbing in
/// the slice header. The block is only present when
/// `sps_range_extension_flag == 1` (§7.4.3.4), which in turn requires
/// `BitDepth > 10` (§7.4.3.4 constraint).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SpsRangeExtension {
    /// `sps_extended_precision_flag` — selects the §7.4.3.22 eq. 106
    /// `Log2TransformRange = Max( 15, Min( 20, BitDepth + 6 ) )`
    /// extended-precision branch. When `0`, `Log2TransformRange = 15`.
    pub sps_extended_precision_flag: bool,
    /// `sps_ts_residual_coding_rice_present_in_sh_flag` — gates
    /// `sh_ts_residual_coding_rice_idx_minus1` in slice headers. Only
    /// transmitted when `sps_transform_skip_enabled_flag == 1`;
    /// inferred to `0` otherwise (§7.4.3.22).
    pub sps_ts_residual_coding_rice_present_in_sh_flag: bool,
    /// `sps_rrc_rice_extension_flag` — selects the alternative Rice
    /// parameter derivation for `abs_remaining[]` / `dec_abs_level[]`
    /// (§9.3.3.10 `baseLevel` branch — `baseLevel = 4` when `0`,
    /// `baseLevel ∈ {0, 2, 4}` per the eq. block when `1`).
    pub sps_rrc_rice_extension_flag: bool,
    /// `sps_persistent_rice_adaptation_enabled_flag` — when `1`, the
    /// per-component `StatCoeff[]` accumulator is carried across TUs
    /// and seeds the Rice parameter at TU start (§9.3.3.10 eqs. 1521 /
    /// HistValue / updateHist).
    pub sps_persistent_rice_adaptation_enabled_flag: bool,
    /// `sps_reverse_last_sig_coeff_enabled_flag` — gates
    /// `sh_reverse_last_sig_coeff_flag` in slice headers (§7.3.7).
    pub sps_reverse_last_sig_coeff_enabled_flag: bool,
}

impl SpsRangeExtension {
    /// §7.4.3.22 eq. 106 `Log2TransformRange` derivation.
    ///
    /// When `sps_extended_precision_flag == 1`, returns
    /// `Max( 15, Min( 20, BitDepth + 6 ) )`; otherwise `15`.
    pub fn log2_transform_range(&self, bit_depth: u32) -> u32 {
        if self.sps_extended_precision_flag {
            // §7.4.3.22 eq. 106: Max( 15, Min( 20, BitDepth + 6 ) ).
            // Equivalent to clamping into the inclusive range [15, 20].
            (bit_depth + 6).clamp(15, 20)
        } else {
            15
        }
    }
}

/// Subpicture info block (§7.4.3.4, gated by
/// `sps_subpic_info_present_flag == 1`).
#[derive(Clone, Debug, Default)]
pub struct SubpicInfo {
    pub num_subpics_minus1: u32,
    pub independent_subpics_flag: bool,
    pub subpic_same_size_flag: bool,
    pub subpics: Vec<SubpicEntry>,
    pub subpic_id_len_minus1: u32,
    pub subpic_id_mapping_explicitly_signalled_flag: bool,
    pub subpic_id_mapping_present_flag: bool,
    /// `sps_subpic_id[i]` values when the mapping is signalled in-SPS.
    pub subpic_ids: Vec<u32>,
}

/// Per-subpicture entry. Fields inherit the spec's inference rules
/// (§7.4.3.4, subpic block): a missing top-left offset is inferred
/// to 0 or a position derived from the same-size grid; a missing
/// width/height defaults to the residual of the picture-relative CTU
/// grid; `treated_as_pic_flag` defaults to 1 and
/// `loop_filter_across_subpic_enabled_flag` to 0 when not present.
#[derive(Clone, Copy, Debug, Default)]
pub struct SubpicEntry {
    pub ctu_top_left_x: u32,
    pub ctu_top_left_y: u32,
    pub width_minus1: u32,
    pub height_minus1: u32,
    pub treated_as_pic_flag: bool,
    pub loop_filter_across_subpic_enabled_flag: bool,
}

/// SPS conformance-cropping window (§7.4.3.4).
#[derive(Clone, Copy, Debug)]
pub struct ConformanceWindow {
    pub left_offset: u32,
    pub right_offset: u32,
    pub top_offset: u32,
    pub bottom_offset: u32,
}

/// DPB sublayer parameters (§7.3.4 / §7.4.5).
#[derive(Clone, Debug, Default)]
pub struct DpbParameters {
    /// Per-sublayer triples ordered as read out of the bitstream
    /// (`dpb_max_dec_pic_buffering_minus1`, `dpb_max_num_reorder_pics`,
    /// `dpb_max_latency_increase_plus1`).
    pub sublayers: Vec<DpbSublayer>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct DpbSublayer {
    pub max_dec_pic_buffering_minus1: u32,
    pub max_num_reorder_pics: u32,
    pub max_latency_increase_plus1: u32,
}

/// SPS chroma QP-table entry (§7.3.2.4, §8.7.1).
#[derive(Clone, Debug, Default)]
pub struct ChromaQpTable {
    pub qp_table_start_minus26: i32,
    /// Paired `(delta_qp_in_val_minus1, delta_qp_diff_val)` runs.
    pub entries: Vec<(u32, u32)>,
}

impl ChromaQpTable {
    /// §7.4.3.4 eqs. 54 – 57 — build the `ChromaQpTable[k]` mapping array
    /// for `k ∈ [−QpBdOffset, 63]`. The returned vector is indexed by
    /// `k + qp_bd_offset` (so `ChromaQpTable[k]` lives at index
    /// `(k + qp_bd_offset) as usize`); use [`Self::map_qp`] for a
    /// clamped lookup that mirrors §8.7.1.
    ///
    /// The single-point table this crate emits (`same_qp_table_for_chroma`,
    /// `num_points = 1`, `delta_qp_in_val_minus1 = 0`,
    /// `delta_qp_diff_val = 1`) derives to the identity
    /// `ChromaQpTable[k] = Clip3(−QpBdOffset, 63, k)`. NB an all-zero
    /// point does NOT: `qpOutVal` would hold at the start value and the
    /// tail extrapolation lands on `k − 1` above it (r415).
    pub fn build(&self, qp_bd_offset: i32) -> Vec<i32> {
        let num_points = self.entries.len(); // sps_num_points_in_qp_table_minus1 + 1
                                             // qpInVal / qpOutVal pivot arrays (length num_points + 1).
        let mut qp_in = vec![0i32; num_points + 1];
        let mut qp_out = vec![0i32; num_points + 1];
        qp_in[0] = self.qp_table_start_minus26 + 26;
        qp_out[0] = qp_in[0];
        for (j, &(d_in_minus1, d_diff)) in self.entries.iter().enumerate() {
            let d_in = d_in_minus1 as i32; // sps_delta_qp_in_val_minus1[j]
            qp_in[j + 1] = qp_in[j] + d_in + 1;
            // qpOutVal += (delta_qp_in_val_minus1 ^ delta_qp_diff_val).
            qp_out[j + 1] = qp_out[j] + (d_in_minus1 ^ d_diff) as i32;
        }

        // ChromaQpTable index space is k ∈ [−QpBdOffset, 63].
        let lo = -qp_bd_offset;
        let len = (63 - lo + 1) as usize;
        let mut table = vec![0i32; len];
        let idx = |k: i32| (k - lo) as usize;

        // Eq. 56: anchor at qpInVal[0].
        table[idx(qp_in[0])] = qp_out[0];
        // Eq. 57: fill below the anchor.
        let mut k = qp_in[0] - 1;
        while k >= lo {
            table[idx(k)] = (table[idx(k + 1)] - 1).clamp(-qp_bd_offset, 63);
            k -= 1;
        }
        // Linear interpolation between pivots.
        for (j, &(d_in_minus1, _)) in self.entries.iter().enumerate() {
            let denom = d_in_minus1 as i32 + 1; // sps_delta_qp_in_val_minus1 + 1
            let sh = denom >> 1;
            let mut m = 1i32;
            let anchor = table[idx(qp_in[j])];
            let out_delta = qp_out[j + 1] - qp_out[j];
            let mut k = qp_in[j] + 1;
            while k <= qp_in[j + 1] {
                table[idx(k)] = anchor + (out_delta * m + sh) / denom;
                m += 1;
                k += 1;
            }
        }
        // Fill above the last pivot.
        let mut k = qp_in[num_points] + 1;
        while k <= 63 {
            table[idx(k)] = (table[idx(k - 1)] + 1).clamp(-qp_bd_offset, 63);
            k += 1;
        }
        table
    }

    /// §8.7.1 chroma-QP lookup: clamp `qp_i` into `[−QpBdOffset, 63]`,
    /// then read `ChromaQpTable[qp_i]`. `built` is the array from
    /// [`Self::build`] (indexed by `k + qp_bd_offset`).
    pub fn map_qp(built: &[i32], qp_i: i32, qp_bd_offset: i32) -> i32 {
        let clamped = qp_i.clamp(-qp_bd_offset, 63);
        built[(clamped + qp_bd_offset) as usize]
    }
}

/// Luma-Adaptive Deblocking Filter parameters (§7.3.2.4).
#[derive(Clone, Debug, Default)]
pub struct LadfParameters {
    pub num_intervals_minus2: u8,
    pub lowest_interval_qp_offset: i32,
    /// Paired `(qp_offset, delta_threshold_minus1)` per interval.
    pub intervals: Vec<(i32, u32)>,
}

/// Explicit SPS virtual boundary positions (§7.3.2.4).
#[derive(Clone, Debug, Default)]
pub struct VirtualBoundaries {
    pub pos_x_minus1: Vec<u32>,
    pub pos_y_minus1: Vec<u32>,
}

/// Partition-constraints block (§7.3.2.4 / §7.4.3.4).
///
/// Mirrors the spec structure field-for-field so downstream consumers
/// (slice-header partitioning override, CTU walker) can access the raw
/// values; derived quantities such as `MinCbLog2SizeY`, `MinQtLog2` etc.
/// are computed on demand in the CTU walker.
#[derive(Clone, Debug, Default)]
pub struct PartitionConstraints {
    pub log2_min_luma_coding_block_size_minus2: u32,
    pub partition_constraints_override_enabled_flag: bool,
    pub log2_diff_min_qt_min_cb_intra_slice_luma: u32,
    pub max_mtt_hierarchy_depth_intra_slice_luma: u32,
    pub log2_diff_max_bt_min_qt_intra_slice_luma: u32,
    pub log2_diff_max_tt_min_qt_intra_slice_luma: u32,
    pub qtbtt_dual_tree_intra_flag: bool,
    pub log2_diff_min_qt_min_cb_intra_slice_chroma: u32,
    pub max_mtt_hierarchy_depth_intra_slice_chroma: u32,
    pub log2_diff_max_bt_min_qt_intra_slice_chroma: u32,
    pub log2_diff_max_tt_min_qt_intra_slice_chroma: u32,
    pub log2_diff_min_qt_min_cb_inter_slice: u32,
    pub max_mtt_hierarchy_depth_inter_slice: u32,
    pub log2_diff_max_bt_min_qt_inter_slice: u32,
    pub log2_diff_max_tt_min_qt_inter_slice: u32,
    pub max_luma_transform_size_64_flag: bool,
}

/// A single entry of `ref_pic_list_struct(listIdx, rplsIdx)` — §7.3.10 /
/// §7.4.11. An entry is exactly one of:
///
/// * **ST** (short-term reference) — carries `abs_delta_poc_st` and an
///   optional `strp_entry_sign_flag`. The composed signed POC delta is
///   exposed as [`RefPicListEntry::delta_poc_val_st`] (equation 151).
/// * **LT** (long-term reference) — either the LSB is listed inline as
///   `rpls_poc_lsb_lt` (when `ltrp_in_header_flag == 0`) or is deferred
///   to the containing PH/slice header (when `ltrp_in_header_flag == 1`).
/// * **ILRP** (inter-layer reference) — indexes the direct-reference-layer
///   list via `ilrp_idx`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RefPicListEntry {
    /// Short-term reference picture entry (§7.4.11).
    /// `abs_delta_poc_st` is the raw syntax element; `delta_poc_val_st`
    /// already has equation (150) applied (+1 when not gated by weighted
    /// pred on i != 0) and sign inversion from `strp_entry_sign_flag`.
    ShortTerm {
        abs_delta_poc_st: u32,
        strp_entry_sign_flag: bool,
        /// Signed POC delta `DeltaPocValSt` (equation 151). `None` when
        /// `AbsDeltaPocSt == 0` because the sign flag is absent and the
        /// delta is defined as 0.
        delta_poc_val_st: i32,
    },
    /// Long-term reference picture entry (§7.4.11). When
    /// `ltrp_in_header_flag` was 1 the POC LSB is deferred to the
    /// containing PH/SH; the `poc_lsb_lt` field is then `None`.
    LongTerm { poc_lsb_lt: Option<u32> },
    /// Inter-layer reference picture entry. `ilrp_idx` indexes the
    /// direct-reference-layer list (§7.4.11).
    InterLayer { ilrp_idx: u32 },
}

impl RefPicListEntry {
    pub fn is_short_term(&self) -> bool {
        matches!(self, RefPicListEntry::ShortTerm { .. })
    }
    pub fn is_long_term(&self) -> bool {
        matches!(self, RefPicListEntry::LongTerm { .. })
    }
    pub fn is_inter_layer(&self) -> bool {
        matches!(self, RefPicListEntry::InterLayer { .. })
    }
}

/// `ref_pic_list_struct(listIdx, rplsIdx)` — §7.3.10.
///
/// `num_ref_entries` is always equal to `entries.len()`. `ltrp_in_header_flag`
/// is the *effective* value after inference (§7.4.11: inferred to 1 when
/// `rplsIdx == sps_num_ref_pic_lists[listIdx]` and
/// `sps_long_term_ref_pics_flag == 1`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RefPicListStruct {
    pub entries: Vec<RefPicListEntry>,
    pub ltrp_in_header_flag: bool,
}

impl RefPicListStruct {
    /// Number of LTRP entries in this list (§7.4.11, equation 149).
    pub fn num_ltrp_entries(&self) -> usize {
        self.entries.iter().filter(|e| e.is_long_term()).count()
    }

    pub fn num_strp_entries(&self) -> usize {
        self.entries.iter().filter(|e| e.is_short_term()).count()
    }

    pub fn num_ilrp_entries(&self) -> usize {
        self.entries.iter().filter(|e| e.is_inter_layer()).count()
    }
}

/// SPS tool-enable flags (§7.3.2.4 tail).
///
/// These are the gates slice / picture headers consult to decide which
/// derivation paths to take. Field names mirror the spec 1:1 minus the
/// `sps_` prefix.
#[derive(Clone, Debug, Default)]
pub struct ToolFlags {
    // Transforms.
    pub transform_skip_enabled_flag: bool,
    pub log2_transform_skip_max_size_minus2: u32,
    pub bdpcm_enabled_flag: bool,
    pub mts_enabled_flag: bool,
    pub explicit_mts_intra_enabled_flag: bool,
    pub explicit_mts_inter_enabled_flag: bool,
    pub lfnst_enabled_flag: bool,

    // Chroma / QP.
    pub joint_cbcr_enabled_flag: bool,
    pub same_qp_table_for_chroma_flag: bool,
    pub chroma_qp_tables: Vec<ChromaQpTable>,

    // Loop filters + LMCS.
    pub sao_enabled_flag: bool,
    pub alf_enabled_flag: bool,
    pub ccalf_enabled_flag: bool,
    pub lmcs_enabled_flag: bool,

    // Weighted / long-term reference picture support.
    pub weighted_pred_flag: bool,
    pub weighted_bipred_flag: bool,
    pub long_term_ref_pics_flag: bool,
    pub inter_layer_prediction_enabled_flag: bool,
    pub idr_rpl_present_flag: bool,
    pub rpl1_same_as_rpl0_flag: bool,
    /// `sps_num_ref_pic_lists[i]` for i = 0 and (if signalled) i = 1.
    pub num_ref_pic_lists: [u32; 2],
    /// `ref_pic_list_struct(listIdx, rplsIdx)` per §7.3.10 — outer index
    /// is `listIdx` (0 or 1), inner index is `rplsIdx`. When
    /// `rpl1_same_as_rpl0_flag` is set, only `ref_pic_lists[0]` is
    /// populated and the caller is expected to alias list 1 to list 0.
    pub ref_pic_lists: [Vec<RefPicListStruct>; 2],

    // Inter / motion tools.
    pub ref_wraparound_enabled_flag: bool,
    pub temporal_mvp_enabled_flag: bool,
    pub sbtmvp_enabled_flag: bool,
    pub amvr_enabled_flag: bool,
    pub bdof_enabled_flag: bool,
    pub bdof_control_present_in_ph_flag: bool,
    pub smvd_enabled_flag: bool,
    pub dmvr_enabled_flag: bool,
    pub dmvr_control_present_in_ph_flag: bool,
    pub mmvd_enabled_flag: bool,
    pub mmvd_fullpel_only_enabled_flag: bool,
    pub six_minus_max_num_merge_cand: u32,
    pub sbt_enabled_flag: bool,
    pub affine_enabled_flag: bool,
    pub five_minus_max_num_subblock_merge_cand: u32,
    pub six_param_affine_enabled_flag: bool,
    pub affine_amvr_enabled_flag: bool,
    pub affine_prof_enabled_flag: bool,
    pub prof_control_present_in_ph_flag: bool,
    pub bcw_enabled_flag: bool,
    pub ciip_enabled_flag: bool,
    pub gpm_enabled_flag: bool,
    pub max_num_merge_cand_minus_max_num_gpm_cand: u32,
    pub log2_parallel_merge_level_minus2: u32,

    // Intra tools.
    pub isp_enabled_flag: bool,
    pub mrl_enabled_flag: bool,
    pub mip_enabled_flag: bool,
    pub cclm_enabled_flag: bool,
    pub chroma_horizontal_collocated_flag: bool,
    pub chroma_vertical_collocated_flag: bool,
    pub palette_enabled_flag: bool,
    pub act_enabled_flag: bool,
    pub min_qp_prime_ts: u32,

    // IBC, LADF, scaling list, virtual boundaries, dep quant.
    pub ibc_enabled_flag: bool,
    pub six_minus_max_num_ibc_merge_cand: u32,
    pub ladf_enabled_flag: bool,
    pub ladf: Option<LadfParameters>,
    pub explicit_scaling_list_enabled_flag: bool,
    pub scaling_matrix_for_lfnst_disabled_flag: bool,
    pub scaling_matrix_for_alternative_colour_space_disabled_flag: bool,
    pub scaling_matrix_designated_colour_space_flag: bool,
    pub dep_quant_enabled_flag: bool,
    pub sign_data_hiding_enabled_flag: bool,
    pub virtual_boundaries_enabled_flag: bool,
    pub virtual_boundaries_present_flag: bool,
    pub virtual_boundaries: Option<VirtualBoundaries>,
}

impl SeqParameterSet {
    /// CTB size in luma samples. §7.4.3.4: `CtbSizeY = 1 << (log2_ctu_size_minus5 + 5)`.
    pub fn ctb_size(&self) -> u32 {
        1 << (self.sps_log2_ctu_size_minus5 as u32 + 5)
    }

    /// Bit depth of luma samples (= `sps_bitdepth_minus8 + 8`).
    pub fn bit_depth_y(&self) -> u32 {
        self.sps_bitdepth_minus8 + 8
    }

    /// Bit depth of chroma samples. VVC signals a single bit depth for
    /// both planes (§7.4.3.4).
    pub fn bit_depth_c(&self) -> u32 {
        self.bit_depth_y()
    }

    /// `MaxNumMergeCand` per §7.4.3.4: `6 − sps_six_minus_max_num_merge_cand`,
    /// clipped to `[1, 6]` (the spec range bullet — and the SPS u(e) range
    /// is implicitly `0..=5` for a conforming bitstream).
    ///
    /// This is the regular-merge candidate count and is also the source for
    /// `MaxNumGpmMergeCand` via the SPS
    /// `max_num_merge_cand_minus_max_num_gpm_cand` syntax element.
    pub fn max_num_merge_cand(&self) -> u32 {
        (6i32 - self.tool_flags.six_minus_max_num_merge_cand as i32).clamp(1, 6) as u32
    }

    /// `MaxNumSubblockMergeCand` per §7.4.3.4 eq. 85.
    ///
    /// The spec derivation reads:
    /// * `if( sps_affine_enabled_flag )`
    ///   `MaxNumSubblockMergeCand = 5 − sps_five_minus_max_num_subblock_merge_cand`
    /// * `else`
    ///   `MaxNumSubblockMergeCand = sps_sbtmvp_enabled_flag && ph_temporal_mvp_enabled_flag`
    ///
    /// The §7.4.3.4 trailing constraint clamps the result to `[0, 5]`.
    ///
    /// The `ph_temporal_mvp_enabled_flag` argument comes from the picture
    /// header — when not yet known (e.g. SPS-only validation), pass `false`
    /// to get the lower-bound estimate (which still produces the correct
    /// answer whenever `sps_affine_enabled_flag == 1`, since the eq.-85
    /// affine branch ignores the PH input).
    pub fn max_num_subblock_merge_cand(&self, ph_temporal_mvp_enabled_flag: bool) -> u32 {
        let raw = if self.tool_flags.affine_enabled_flag {
            5i32 - self.tool_flags.five_minus_max_num_subblock_merge_cand as i32
        } else if self.tool_flags.sbtmvp_enabled_flag && ph_temporal_mvp_enabled_flag {
            1
        } else {
            0
        };
        raw.clamp(0, 5) as u32
    }

    /// Luma width after the conformance-window crop, using SubWidthC /
    /// SubHeightC derived from `sps_chroma_format_idc`. Per Table 2 in
    /// §6.2: 4:2:0 → (2,2); 4:2:2 → (2,1); 4:4:4 → (1,1); mono → (1,1).
    pub fn cropped_width(&self) -> u32 {
        let sub_x = match self.sps_chroma_format_idc {
            1 | 2 => 2,
            _ => 1,
        };
        let crop = self
            .conformance_window
            .map(|c| sub_x * (c.left_offset + c.right_offset))
            .unwrap_or(0);
        self.sps_pic_width_max_in_luma_samples.saturating_sub(crop)
    }

    pub fn cropped_height(&self) -> u32 {
        let sub_y = match self.sps_chroma_format_idc {
            1 => 2,
            _ => 1,
        };
        let crop = self
            .conformance_window
            .map(|c| sub_y * (c.top_offset + c.bottom_offset))
            .unwrap_or(0);
        self.sps_pic_height_max_in_luma_samples.saturating_sub(crop)
    }
}

/// Parse an SPS NAL RBSP payload (the bytes after the 2-byte NAL header,
/// with emulation-prevention bytes already stripped).
pub fn parse_sps(rbsp: &[u8]) -> Result<SeqParameterSet> {
    let mut br = BitReader::new(rbsp);
    let sps_seq_parameter_set_id = br.u(4)? as u8;
    let sps_video_parameter_set_id = br.u(4)? as u8;
    let sps_max_sublayers_minus1 = br.u(3)? as u8;
    if sps_max_sublayers_minus1 > 6 {
        return Err(Error::invalid(
            "h266 SPS: sps_max_sublayers_minus1 must be <= 6",
        ));
    }
    let sps_chroma_format_idc = br.u(2)? as u8;
    let sps_log2_ctu_size_minus5 = br.u(2)? as u8;
    // §7.4.3.4: sps_log2_ctu_size_minus5 shall be <= 2 (CtbLog2Size in 5..7,
    // i.e. CTUs of 32 / 64 / 128 samples). The u(2) field can encode 0..3
    // so we explicitly reject 3.
    if sps_log2_ctu_size_minus5 > 2 {
        return Err(Error::invalid(format!(
            "h266 SPS: sps_log2_ctu_size_minus5 out of range ({sps_log2_ctu_size_minus5})"
        )));
    }
    let sps_ptl_dpb_hrd_params_present_flag = br.u1()? == 1;
    let profile_tier_level = if sps_ptl_dpb_hrd_params_present_flag {
        Some(parse_profile_tier_level(
            &mut br,
            true,
            sps_max_sublayers_minus1,
        )?)
    } else {
        None
    };
    let sps_gdr_enabled_flag = br.u1()? == 1;
    let sps_ref_pic_resampling_enabled_flag = br.u1()? == 1;
    let sps_res_change_in_clvs_allowed_flag = if sps_ref_pic_resampling_enabled_flag {
        br.u1()? == 1
    } else {
        false
    };
    let sps_pic_width_max_in_luma_samples = br.ue()?;
    let sps_pic_height_max_in_luma_samples = br.ue()?;
    if sps_pic_width_max_in_luma_samples == 0
        || sps_pic_height_max_in_luma_samples == 0
        || sps_pic_width_max_in_luma_samples > 16384
        || sps_pic_height_max_in_luma_samples > 16384
    {
        return Err(Error::invalid(format!(
            "h266 SPS: implausible picture size {sps_pic_width_max_in_luma_samples}x{sps_pic_height_max_in_luma_samples}"
        )));
    }
    let sps_conformance_window_flag = br.u1()? == 1;
    let conformance_window = if sps_conformance_window_flag {
        Some(ConformanceWindow {
            left_offset: br.ue()?,
            right_offset: br.ue()?,
            top_offset: br.ue()?,
            bottom_offset: br.ue()?,
        })
    } else {
        None
    };
    let sps_subpic_info_present_flag = br.u1()? == 1;
    // CtbSizeY = 1 << (log2_ctu_size_minus5 + 5).
    let ctb_size_y: u32 = 1u32 << (sps_log2_ctu_size_minus5 as u32 + 5);
    let subpic_info = if sps_subpic_info_present_flag {
        if sps_res_change_in_clvs_allowed_flag {
            return Err(Error::invalid(
                "h266 SPS: sps_res_change_in_clvs_allowed_flag == 1 disallows sps_subpic_info_present_flag == 1",
            ));
        }
        Some(parse_subpic_info(
            &mut br,
            sps_pic_width_max_in_luma_samples,
            sps_pic_height_max_in_luma_samples,
            ctb_size_y,
        )?)
    } else {
        None
    };
    let sps_bitdepth_minus8 = br.ue()?;
    if sps_bitdepth_minus8 > 8 {
        return Err(Error::invalid(format!(
            "h266 SPS: sps_bitdepth_minus8 out of range ({sps_bitdepth_minus8})"
        )));
    }
    let sps_entropy_coding_sync_enabled_flag = br.u1()? == 1;
    let sps_entry_point_offsets_present_flag = br.u1()? == 1;
    let sps_log2_max_pic_order_cnt_lsb_minus4 = br.u(4)? as u8;
    if sps_log2_max_pic_order_cnt_lsb_minus4 > 12 {
        return Err(Error::invalid(format!(
            "h266 SPS: sps_log2_max_pic_order_cnt_lsb_minus4 out of range ({sps_log2_max_pic_order_cnt_lsb_minus4})"
        )));
    }
    let sps_poc_msb_cycle_flag = br.u1()? == 1;
    let sps_poc_msb_cycle_len_minus1 = if sps_poc_msb_cycle_flag { br.ue()? } else { 0 };
    let sps_num_extra_ph_bytes = br.u(2)? as u8;
    // §7.4.3.4 eq. 41 — NumExtraPhBits = Σ sps_extra_ph_bit_present_flag[i].
    let mut num_extra_ph_bits = 0u8;
    for _ in 0..(sps_num_extra_ph_bytes as u32 * 8) {
        if br.u1()? == 1 {
            num_extra_ph_bits += 1;
        }
    }
    let sps_num_extra_sh_bytes = br.u(2)? as u8;
    // NumExtraShBits = Σ sps_extra_sh_bit_present_flag[i].
    let mut num_extra_sh_bits = 0u8;
    for _ in 0..(sps_num_extra_sh_bytes as u32 * 8) {
        if br.u1()? == 1 {
            num_extra_sh_bits += 1;
        }
    }

    // ---- dpb_parameters (§7.3.4) ----
    let mut sps_sublayer_dpb_params_flag = false;
    let dpb_parameters = if sps_ptl_dpb_hrd_params_present_flag {
        if sps_max_sublayers_minus1 > 0 {
            sps_sublayer_dpb_params_flag = br.u1()? == 1;
        }
        Some(parse_dpb_parameters(
            &mut br,
            sps_max_sublayers_minus1,
            sps_sublayer_dpb_params_flag,
        )?)
    } else {
        None
    };

    // ---- Partition constraints + tool flags (§7.3.2.4 tail) ----
    let partition_constraints =
        parse_partition_constraints(&mut br, sps_chroma_format_idc, sps_log2_ctu_size_minus5)?;
    let tool_flags = parse_tool_flags(
        &mut br,
        sps_chroma_format_idc,
        sps_video_parameter_set_id,
        partition_constraints.max_luma_transform_size_64_flag,
        sps_log2_max_pic_order_cnt_lsb_minus4,
    )?;

    // ---- HRD timing block (§7.3.2.4 / §7.3.5) ----
    let mut sps_timing_hrd_params_present_flag = false;
    let mut general_timing_hrd: Option<GeneralTimingHrdParameters> = None;
    let mut sps_sublayer_cpb_params_present_flag = false;
    let mut ols_timing_hrd: Option<OlsTimingHrdParameters> = None;
    if sps_ptl_dpb_hrd_params_present_flag {
        sps_timing_hrd_params_present_flag = br.u1()? == 1;
        if sps_timing_hrd_params_present_flag {
            let g = parse_general_timing_hrd_parameters(&mut br)?;
            if sps_max_sublayers_minus1 > 0 {
                sps_sublayer_cpb_params_present_flag = br.u1()? == 1;
            }
            let first_sub_layer = if sps_sublayer_cpb_params_present_flag {
                0u8
            } else {
                sps_max_sublayers_minus1
            };
            let o = parse_ols_timing_hrd_parameters(
                &mut br,
                &g,
                first_sub_layer,
                sps_max_sublayers_minus1,
            )?;
            general_timing_hrd = Some(g);
            ols_timing_hrd = Some(o);
        }
    }

    // ---- field_seq / VUI / extension tail (§7.3.2.4) ----
    let sps_field_seq_flag = br.u1()? == 1;
    let sps_vui_parameters_present_flag = br.u1()? == 1;
    let vui_payload = if sps_vui_parameters_present_flag {
        let payload_size = br.ue()? + 1;
        // vui_alignment_zero_bits to the next byte boundary.
        while !br.is_byte_aligned() {
            if br.u1()? != 0 {
                return Err(Error::invalid(
                    "h266 SPS: sps_vui_alignment_zero_bit shall be 0",
                ));
            }
        }
        // Cap payload_size — runaway values would exhaust the buffer.
        let max_vui = (br.bits_remaining() / 8) as u32;
        if payload_size > max_vui {
            return Err(Error::invalid(format!(
                "h266 SPS: sps_vui_payload_size_minus1 + 1 ({payload_size}) exceeds remaining bytes ({max_vui})"
            )));
        }
        let mut buf = Vec::with_capacity(payload_size as usize);
        for _ in 0..payload_size {
            buf.push(br.u(8)? as u8);
        }
        buf
    } else {
        Vec::new()
    };
    let sps_extension_flag = br.u1()? == 1;
    let mut sps_range_extension_flag = false;
    let mut sps_extension_7bits: u8 = 0;
    let mut range_extension: Option<SpsRangeExtension> = None;
    if sps_extension_flag {
        sps_range_extension_flag = br.u1()? == 1;
        sps_extension_7bits = br.u(7)? as u8;
        if sps_range_extension_flag {
            // §7.4.3.4 constraint: "When BitDepth is less than or
            // equal to 10, the value of sps_range_extension_flag
            // shall be equal to 0." The block is only meaningful at
            // > 10-bit depth; flag a malformed bitstream up front.
            let bit_depth = sps_bitdepth_minus8 + 8;
            if bit_depth <= 10 {
                return Err(Error::invalid(format!(
                    "h266 SPS: sps_range_extension_flag = 1 disallowed at BitDepth = {bit_depth} (§7.4.3.4 requires BitDepth > 10)"
                )));
            }
            // §7.3.2.22 `sps_range_extension()` body.
            let sps_extended_precision_flag = br.u1()? == 1;
            // The `sps_ts_residual_coding_rice_present_in_sh_flag`
            // bin is only present when `sps_transform_skip_enabled_flag
            // == 1`; the §7.4.3.22 inference rule gives `0` otherwise.
            let sps_ts_residual_coding_rice_present_in_sh_flag =
                if tool_flags.transform_skip_enabled_flag {
                    br.u1()? == 1
                } else {
                    false
                };
            let sps_rrc_rice_extension_flag = br.u1()? == 1;
            let sps_persistent_rice_adaptation_enabled_flag = br.u1()? == 1;
            let sps_reverse_last_sig_coeff_enabled_flag = br.u1()? == 1;
            range_extension = Some(SpsRangeExtension {
                sps_extended_precision_flag,
                sps_ts_residual_coding_rice_present_in_sh_flag,
                sps_rrc_rice_extension_flag,
                sps_persistent_rice_adaptation_enabled_flag,
                sps_reverse_last_sig_coeff_enabled_flag,
            });
        }
    }
    // `if (sps_extension_7bits) while(more_rbsp_data()) u(1)` — consume
    // the extension-data flag tail. Our `has_more_rbsp_data` probe
    // already handles the stop-bit detection, so this loop is a no-op
    // when only the rbsp_stop_one_bit remains.
    if sps_extension_7bits != 0 {
        while br.has_more_rbsp_data() {
            br.u1()?;
        }
    }

    Ok(SeqParameterSet {
        sps_seq_parameter_set_id,
        sps_video_parameter_set_id,
        sps_max_sublayers_minus1,
        sps_chroma_format_idc,
        sps_log2_ctu_size_minus5,
        sps_ptl_dpb_hrd_params_present_flag,
        profile_tier_level,
        sps_gdr_enabled_flag,
        sps_ref_pic_resampling_enabled_flag,
        sps_res_change_in_clvs_allowed_flag,
        sps_pic_width_max_in_luma_samples,
        sps_pic_height_max_in_luma_samples,
        conformance_window,
        sps_subpic_info_present_flag,
        sps_bitdepth_minus8,
        sps_entropy_coding_sync_enabled_flag,
        sps_entry_point_offsets_present_flag,
        sps_log2_max_pic_order_cnt_lsb_minus4,
        sps_poc_msb_cycle_flag,
        sps_poc_msb_cycle_len_minus1,
        sps_num_extra_ph_bytes,
        sps_num_extra_sh_bytes,
        num_extra_ph_bits,
        num_extra_sh_bits,
        sps_sublayer_dpb_params_flag,
        dpb_parameters,
        partition_constraints,
        tool_flags,
        subpic_info,
        sps_timing_hrd_params_present_flag,
        general_timing_hrd,
        sps_sublayer_cpb_params_present_flag,
        ols_timing_hrd,
        sps_field_seq_flag,
        sps_vui_parameters_present_flag,
        vui_payload,
        sps_extension_flag,
        sps_range_extension_flag,
        sps_extension_7bits,
        range_extension,
    })
}

/// Helper: `Ceil( Log2( n ) )` for non-negative `n` (spec §4 notation).
/// Zero and one map to 0; otherwise returns the smallest non-negative
/// integer `k` such that `(1 << k) >= n`.
fn ceil_log2(n: u32) -> u32 {
    if n <= 1 {
        return 0;
    }
    32 - (n - 1).leading_zeros()
}

/// Parse the subpicture block from §7.4.3.4 (gated by
/// `sps_subpic_info_present_flag == 1`).
fn parse_subpic_info(
    br: &mut BitReader<'_>,
    pic_width_max: u32,
    pic_height_max: u32,
    ctb_size_y: u32,
) -> Result<SubpicInfo> {
    let num_subpics_minus1 = br.ue()?;
    // §7.4.3.4: sps_num_subpics_minus1 < MaxSlicesPerAu. MaxSlicesPerAu
    // is profile-dependent but ≤ 600; we accept up to 4095 to stay
    // forward-compatible without inviting runaway allocation.
    if num_subpics_minus1 > 4095 {
        return Err(Error::invalid(format!(
            "h266 SPS: sps_num_subpics_minus1 out of range ({num_subpics_minus1})"
        )));
    }
    let (independent_subpics_flag, subpic_same_size_flag) = if num_subpics_minus1 > 0 {
        (br.u1()? == 1, br.u1()? == 1)
    } else {
        // When not present they are inferred to 1 and 0 respectively
        // (§7.4.3.4, subpic semantics).
        (true, false)
    };

    let tmp_width_val = (pic_width_max + ctb_size_y - 1) / ctb_size_y;
    let tmp_height_val = (pic_height_max + ctb_size_y - 1) / ctb_size_y;
    let top_left_x_len = ceil_log2(tmp_width_val);
    let top_left_y_len = ceil_log2(tmp_height_val);
    let width_len = top_left_x_len;
    let height_len = top_left_y_len;

    let mut subpics: Vec<SubpicEntry> = Vec::with_capacity((num_subpics_minus1 + 1) as usize);
    // numSubpicCols derived when subpic_same_size_flag == 1.
    let mut num_subpic_cols: u32 = 1;
    for i in 0..=num_subpics_minus1 {
        let mut entry = SubpicEntry::default();
        if !subpic_same_size_flag || i == 0 {
            if i > 0 && pic_width_max > ctb_size_y && top_left_x_len > 0 {
                entry.ctu_top_left_x = br.u(top_left_x_len)?;
            }
            if i > 0 && pic_height_max > ctb_size_y && top_left_y_len > 0 {
                entry.ctu_top_left_y = br.u(top_left_y_len)?;
            }
            if i < num_subpics_minus1 && pic_width_max > ctb_size_y && width_len > 0 {
                entry.width_minus1 = br.u(width_len)?;
            } else {
                // Inferred residual (§7.4.3.4): tmpWidthVal - top_left_x - 1.
                entry.width_minus1 = tmp_width_val
                    .saturating_sub(entry.ctu_top_left_x)
                    .saturating_sub(1);
            }
            if i < num_subpics_minus1 && pic_height_max > ctb_size_y && height_len > 0 {
                entry.height_minus1 = br.u(height_len)?;
            } else {
                entry.height_minus1 = tmp_height_val
                    .saturating_sub(entry.ctu_top_left_y)
                    .saturating_sub(1);
            }
        } else {
            // subpic_same_size_flag == 1 and i > 0 → all geometry is
            // derived from entry 0 plus the subpicture index.
            let w0 = subpics[0].width_minus1 + 1;
            let h0 = subpics[0].height_minus1 + 1;
            if i == 1 {
                // Compute numSubpicCols exactly once, when we have the
                // first entry's geometry in hand (equation 37).
                num_subpic_cols = tmp_width_val / w0;
                if num_subpic_cols == 0 {
                    return Err(Error::invalid(
                        "h266 SPS: subpic_same_size_flag => tmpWidthVal / (w0+1) must be > 0",
                    ));
                }
            }
            entry.ctu_top_left_x = (i % num_subpic_cols) * w0;
            entry.ctu_top_left_y = (i / num_subpic_cols) * h0;
            entry.width_minus1 = subpics[0].width_minus1;
            entry.height_minus1 = subpics[0].height_minus1;
        }
        if !independent_subpics_flag {
            entry.treated_as_pic_flag = br.u1()? == 1;
            entry.loop_filter_across_subpic_enabled_flag = br.u1()? == 1;
        } else {
            // Inferred per §7.4.3.4.
            entry.treated_as_pic_flag = true;
            entry.loop_filter_across_subpic_enabled_flag = false;
        }
        subpics.push(entry);
    }

    let subpic_id_len_minus1 = br.ue()?;
    if subpic_id_len_minus1 > 15 {
        return Err(Error::invalid(format!(
            "h266 SPS: sps_subpic_id_len_minus1 out of range ({subpic_id_len_minus1})"
        )));
    }
    let subpic_id_mapping_explicitly_signalled_flag = br.u1()? == 1;
    let (subpic_id_mapping_present_flag, subpic_ids) =
        if subpic_id_mapping_explicitly_signalled_flag {
            let present = br.u1()? == 1;
            let mut ids = Vec::new();
            if present {
                let id_width = subpic_id_len_minus1 + 1;
                ids.reserve((num_subpics_minus1 + 1) as usize);
                for _ in 0..=num_subpics_minus1 {
                    ids.push(br.u(id_width)?);
                }
            }
            (present, ids)
        } else {
            (false, Vec::new())
        };

    Ok(SubpicInfo {
        num_subpics_minus1,
        independent_subpics_flag,
        subpic_same_size_flag,
        subpics,
        subpic_id_len_minus1,
        subpic_id_mapping_explicitly_signalled_flag,
        subpic_id_mapping_present_flag,
        subpic_ids,
    })
}

/// `dpb_parameters( MaxSubLayersMinus1, subLayerInfoFlag )` — §7.3.4.
fn parse_dpb_parameters(
    br: &mut BitReader<'_>,
    max_sublayers_minus1: u8,
    sublayer_info_flag: bool,
) -> Result<DpbParameters> {
    let start = if sublayer_info_flag {
        0u32
    } else {
        max_sublayers_minus1 as u32
    };
    let end = max_sublayers_minus1 as u32;
    let mut sublayers = Vec::with_capacity((end - start + 1) as usize);
    for _ in start..=end {
        sublayers.push(DpbSublayer {
            max_dec_pic_buffering_minus1: br.ue()?,
            max_num_reorder_pics: br.ue()?,
            max_latency_increase_plus1: br.ue()?,
        });
    }
    Ok(DpbParameters { sublayers })
}

/// §7.3.2.4 partition-constraints block.
fn parse_partition_constraints(
    br: &mut BitReader<'_>,
    chroma_format_idc: u8,
    log2_ctu_size_minus5: u8,
) -> Result<PartitionConstraints> {
    let log2_min_luma_coding_block_size_minus2 = br.ue()?;
    let partition_constraints_override_enabled_flag = br.u1()? == 1;
    let log2_diff_min_qt_min_cb_intra_slice_luma = br.ue()?;
    let max_mtt_hierarchy_depth_intra_slice_luma = br.ue()?;
    let (log2_diff_max_bt_min_qt_intra_slice_luma, log2_diff_max_tt_min_qt_intra_slice_luma) =
        if max_mtt_hierarchy_depth_intra_slice_luma != 0 {
            (br.ue()?, br.ue()?)
        } else {
            (0, 0)
        };
    let qtbtt_dual_tree_intra_flag = if chroma_format_idc != 0 {
        br.u1()? == 1
    } else {
        false
    };
    let (
        log2_diff_min_qt_min_cb_intra_slice_chroma,
        max_mtt_hierarchy_depth_intra_slice_chroma,
        log2_diff_max_bt_min_qt_intra_slice_chroma,
        log2_diff_max_tt_min_qt_intra_slice_chroma,
    ) = if qtbtt_dual_tree_intra_flag {
        let min_qt_c = br.ue()?;
        let max_depth_c = br.ue()?;
        let (bt_c, tt_c) = if max_depth_c != 0 {
            (br.ue()?, br.ue()?)
        } else {
            (0, 0)
        };
        (min_qt_c, max_depth_c, bt_c, tt_c)
    } else {
        (0, 0, 0, 0)
    };
    let log2_diff_min_qt_min_cb_inter_slice = br.ue()?;
    let max_mtt_hierarchy_depth_inter_slice = br.ue()?;
    let (log2_diff_max_bt_min_qt_inter_slice, log2_diff_max_tt_min_qt_inter_slice) =
        if max_mtt_hierarchy_depth_inter_slice != 0 {
            (br.ue()?, br.ue()?)
        } else {
            (0, 0)
        };
    // `CtbSizeY = 1 << (log2_ctu_size_minus5 + 5)` — the 64-flag is only
    // signalled when CtbSizeY > 32, i.e. when log2_ctu_size_minus5 >= 1.
    let max_luma_transform_size_64_flag = if log2_ctu_size_minus5 >= 1 {
        br.u1()? == 1
    } else {
        false
    };
    Ok(PartitionConstraints {
        log2_min_luma_coding_block_size_minus2,
        partition_constraints_override_enabled_flag,
        log2_diff_min_qt_min_cb_intra_slice_luma,
        max_mtt_hierarchy_depth_intra_slice_luma,
        log2_diff_max_bt_min_qt_intra_slice_luma,
        log2_diff_max_tt_min_qt_intra_slice_luma,
        qtbtt_dual_tree_intra_flag,
        log2_diff_min_qt_min_cb_intra_slice_chroma,
        max_mtt_hierarchy_depth_intra_slice_chroma,
        log2_diff_max_bt_min_qt_intra_slice_chroma,
        log2_diff_max_tt_min_qt_intra_slice_chroma,
        log2_diff_min_qt_min_cb_inter_slice,
        max_mtt_hierarchy_depth_inter_slice,
        log2_diff_max_bt_min_qt_inter_slice,
        log2_diff_max_tt_min_qt_inter_slice,
        max_luma_transform_size_64_flag,
    })
}

/// §7.3.2.4 tool-enable flags, stopping before HRD timing / VUI /
/// `sps_extension_flag`.
#[allow(clippy::field_reassign_with_default)]
fn parse_tool_flags(
    br: &mut BitReader<'_>,
    chroma_format_idc: u8,
    video_parameter_set_id: u8,
    max_luma_transform_size_64_flag: bool,
    log2_max_pic_order_cnt_lsb_minus4: u8,
) -> Result<ToolFlags> {
    let mut t = ToolFlags::default();

    // -- transforms --
    t.transform_skip_enabled_flag = br.u1()? == 1;
    if t.transform_skip_enabled_flag {
        t.log2_transform_skip_max_size_minus2 = br.ue()?;
        t.bdpcm_enabled_flag = br.u1()? == 1;
    }
    t.mts_enabled_flag = br.u1()? == 1;
    if t.mts_enabled_flag {
        t.explicit_mts_intra_enabled_flag = br.u1()? == 1;
        t.explicit_mts_inter_enabled_flag = br.u1()? == 1;
    }
    t.lfnst_enabled_flag = br.u1()? == 1;

    // -- chroma / QP tables --
    if chroma_format_idc != 0 {
        t.joint_cbcr_enabled_flag = br.u1()? == 1;
        t.same_qp_table_for_chroma_flag = br.u1()? == 1;
        let num_qp_tables = if t.same_qp_table_for_chroma_flag {
            1
        } else if t.joint_cbcr_enabled_flag {
            3
        } else {
            2
        };
        for _ in 0..num_qp_tables {
            let start = br.se()?;
            let num_points_minus1 = br.ue()?;
            let mut entries = Vec::with_capacity((num_points_minus1 as usize) + 1);
            for _ in 0..=num_points_minus1 {
                let a = br.ue()?;
                let b = br.ue()?;
                entries.push((a, b));
            }
            t.chroma_qp_tables.push(ChromaQpTable {
                qp_table_start_minus26: start,
                entries,
            });
        }
    }

    // -- loop filters + LMCS --
    t.sao_enabled_flag = br.u1()? == 1;
    t.alf_enabled_flag = br.u1()? == 1;
    if t.alf_enabled_flag && chroma_format_idc != 0 {
        t.ccalf_enabled_flag = br.u1()? == 1;
    }
    t.lmcs_enabled_flag = br.u1()? == 1;

    // -- weighted / long-term refs --
    t.weighted_pred_flag = br.u1()? == 1;
    t.weighted_bipred_flag = br.u1()? == 1;
    t.long_term_ref_pics_flag = br.u1()? == 1;
    if video_parameter_set_id > 0 {
        t.inter_layer_prediction_enabled_flag = br.u1()? == 1;
    }
    t.idr_rpl_present_flag = br.u1()? == 1;
    t.rpl1_same_as_rpl0_flag = br.u1()? == 1;
    let rpl_loops = if t.rpl1_same_as_rpl0_flag { 1 } else { 2 };
    for i in 0..rpl_loops {
        let n = br.ue()?;
        // §7.4.3.4: sps_num_ref_pic_lists[i] shall be in the range of 0
        // to 64, inclusive. Reject runaway ue(v) before we allocate.
        if n > 64 {
            return Err(Error::invalid(format!(
                "h266 SPS: sps_num_ref_pic_lists[{i}] out of range ({n})"
            )));
        }
        t.num_ref_pic_lists[i] = n;
        let mut lists = Vec::with_capacity(n as usize);
        for j in 0..n {
            lists.push(parse_ref_pic_list_struct(
                br,
                i as u8,
                j,
                n,
                t.long_term_ref_pics_flag,
                t.inter_layer_prediction_enabled_flag,
                t.weighted_pred_flag,
                t.weighted_bipred_flag,
                log2_max_pic_order_cnt_lsb_minus4,
            )?);
        }
        t.ref_pic_lists[i] = lists;
    }

    // -- inter / motion tools --
    t.ref_wraparound_enabled_flag = br.u1()? == 1;
    t.temporal_mvp_enabled_flag = br.u1()? == 1;
    if t.temporal_mvp_enabled_flag {
        t.sbtmvp_enabled_flag = br.u1()? == 1;
    }
    t.amvr_enabled_flag = br.u1()? == 1;
    t.bdof_enabled_flag = br.u1()? == 1;
    if t.bdof_enabled_flag {
        t.bdof_control_present_in_ph_flag = br.u1()? == 1;
    }
    t.smvd_enabled_flag = br.u1()? == 1;
    t.dmvr_enabled_flag = br.u1()? == 1;
    if t.dmvr_enabled_flag {
        t.dmvr_control_present_in_ph_flag = br.u1()? == 1;
    }
    t.mmvd_enabled_flag = br.u1()? == 1;
    if t.mmvd_enabled_flag {
        t.mmvd_fullpel_only_enabled_flag = br.u1()? == 1;
    }
    t.six_minus_max_num_merge_cand = br.ue()?;
    t.sbt_enabled_flag = br.u1()? == 1;
    t.affine_enabled_flag = br.u1()? == 1;
    if t.affine_enabled_flag {
        t.five_minus_max_num_subblock_merge_cand = br.ue()?;
        t.six_param_affine_enabled_flag = br.u1()? == 1;
        if t.amvr_enabled_flag {
            t.affine_amvr_enabled_flag = br.u1()? == 1;
        }
        t.affine_prof_enabled_flag = br.u1()? == 1;
        if t.affine_prof_enabled_flag {
            t.prof_control_present_in_ph_flag = br.u1()? == 1;
        }
    }
    t.bcw_enabled_flag = br.u1()? == 1;
    t.ciip_enabled_flag = br.u1()? == 1;

    // sps_gpm is gated on `MaxNumMergeCand >= 2`, derived from
    // `six_minus_max_num_merge_cand` (§7.4.3.4: MaxNumMergeCand =
    // 6 − six_minus_max_num_merge_cand). The gpm fields are only
    // present when that derived value is >= 2.
    let max_num_merge_cand = 6i32 - t.six_minus_max_num_merge_cand as i32;
    if max_num_merge_cand >= 2 {
        t.gpm_enabled_flag = br.u1()? == 1;
        if t.gpm_enabled_flag && max_num_merge_cand >= 3 {
            t.max_num_merge_cand_minus_max_num_gpm_cand = br.ue()?;
        }
    }
    t.log2_parallel_merge_level_minus2 = br.ue()?;

    // -- intra tools --
    t.isp_enabled_flag = br.u1()? == 1;
    t.mrl_enabled_flag = br.u1()? == 1;
    t.mip_enabled_flag = br.u1()? == 1;
    if chroma_format_idc != 0 {
        t.cclm_enabled_flag = br.u1()? == 1;
    }
    if chroma_format_idc == 1 {
        t.chroma_horizontal_collocated_flag = br.u1()? == 1;
        t.chroma_vertical_collocated_flag = br.u1()? == 1;
    }
    t.palette_enabled_flag = br.u1()? == 1;
    if chroma_format_idc == 3 && !max_luma_transform_size_64_flag {
        t.act_enabled_flag = br.u1()? == 1;
    }
    if t.transform_skip_enabled_flag || t.palette_enabled_flag {
        t.min_qp_prime_ts = br.ue()?;
    }

    // -- IBC / LADF / scaling list / virtual boundaries --
    t.ibc_enabled_flag = br.u1()? == 1;
    if t.ibc_enabled_flag {
        t.six_minus_max_num_ibc_merge_cand = br.ue()?;
    }
    t.ladf_enabled_flag = br.u1()? == 1;
    if t.ladf_enabled_flag {
        let num_intervals_minus2 = br.u(2)? as u8;
        let lowest = br.se()?;
        let mut intervals = Vec::with_capacity((num_intervals_minus2 as usize) + 1);
        for _ in 0..(num_intervals_minus2 as u32 + 1) {
            let qp_off = br.se()?;
            let thresh = br.ue()?;
            intervals.push((qp_off, thresh));
        }
        t.ladf = Some(LadfParameters {
            num_intervals_minus2,
            lowest_interval_qp_offset: lowest,
            intervals,
        });
    }

    t.explicit_scaling_list_enabled_flag = br.u1()? == 1;
    if t.lfnst_enabled_flag && t.explicit_scaling_list_enabled_flag {
        t.scaling_matrix_for_lfnst_disabled_flag = br.u1()? == 1;
    }
    if t.act_enabled_flag && t.explicit_scaling_list_enabled_flag {
        t.scaling_matrix_for_alternative_colour_space_disabled_flag = br.u1()? == 1;
    }
    if t.scaling_matrix_for_alternative_colour_space_disabled_flag {
        t.scaling_matrix_designated_colour_space_flag = br.u1()? == 1;
    }
    t.dep_quant_enabled_flag = br.u1()? == 1;
    t.sign_data_hiding_enabled_flag = br.u1()? == 1;

    t.virtual_boundaries_enabled_flag = br.u1()? == 1;
    if t.virtual_boundaries_enabled_flag {
        t.virtual_boundaries_present_flag = br.u1()? == 1;
        if t.virtual_boundaries_present_flag {
            let num_ver = br.ue()?;
            let mut pos_x = Vec::with_capacity(num_ver as usize);
            for _ in 0..num_ver {
                pos_x.push(br.ue()?);
            }
            let num_hor = br.ue()?;
            let mut pos_y = Vec::with_capacity(num_hor as usize);
            for _ in 0..num_hor {
                pos_y.push(br.ue()?);
            }
            t.virtual_boundaries = Some(VirtualBoundaries {
                pos_x_minus1: pos_x,
                pos_y_minus1: pos_y,
            });
        }
    }
    Ok(t)
}

/// Parse a single `ref_pic_list_struct(listIdx, rplsIdx)` — §7.3.10.
///
/// * `list_idx` / `rpls_idx` — outer / inner indices as referenced by the
///   spec. `num_ref_pic_lists_listidx` is the value of
///   `sps_num_ref_pic_lists[listIdx]` — needed to decide whether
///   `ltrp_in_header_flag` is transmitted or inferred per §7.4.11.
/// * `long_term_ref_pics_flag` / `inter_layer_prediction_enabled_flag` —
///   SPS tool gates that control which sub-flags are present.
/// * `weighted_pred_flag` / `weighted_bipred_flag` — feed equation 150
///   (`AbsDeltaPocSt = abs_delta_poc_st + (gated ? 0 : 1)`).
/// * `log2_max_pic_order_cnt_lsb_minus4` — width of `rpls_poc_lsb_lt`
///   (§7.4.11: `u(v)` with `v = log2_max_pic_order_cnt_lsb_minus4 + 4`).
pub fn parse_ref_pic_list_struct(
    br: &mut BitReader<'_>,
    list_idx: u8,
    rpls_idx: u32,
    num_ref_pic_lists_listidx: u32,
    long_term_ref_pics_flag: bool,
    inter_layer_prediction_enabled_flag: bool,
    weighted_pred_flag: bool,
    weighted_bipred_flag: bool,
    log2_max_pic_order_cnt_lsb_minus4: u8,
) -> Result<RefPicListStruct> {
    let _ = list_idx; // reserved for future cross-list validation hooks.
    let num_ref_entries = br.ue()?;
    // §7.4.11: num_ref_entries shall be <= MaxDpbSize + 13. MaxDpbSize is
    // profile-dependent (A.4.2) but is capped at 8 for any defined level,
    // so the hard upper bound is 21. We accept a slightly higher
    // threshold (32) to stay forward-compatible, and reject grossly
    // out-of-range values to avoid runaway allocation.
    if num_ref_entries > 32 {
        return Err(Error::invalid(format!(
            "h266 RPL[{list_idx}][{rpls_idx}]: num_ref_entries out of range ({num_ref_entries})"
        )));
    }

    // `ltrp_in_header_flag` is only transmitted when the list *could*
    // carry LTRPs (sps_long_term_ref_pics_flag == 1) and we are not on
    // the synthesized "picture-header supplies the list" candidate
    // (rplsIdx < sps_num_ref_pic_lists[listIdx]) and there is at least
    // one entry. Otherwise §7.4.11 infers it to 1 when LT refs are
    // enabled and rplsIdx == sps_num_ref_pic_lists[listIdx], else 0.
    let ltrp_in_header_flag =
        if long_term_ref_pics_flag && rpls_idx < num_ref_pic_lists_listidx && num_ref_entries > 0 {
            br.u1()? == 1
        } else if long_term_ref_pics_flag && rpls_idx == num_ref_pic_lists_listidx {
            true
        } else {
            false
        };

    let poc_lsb_lt_width = log2_max_pic_order_cnt_lsb_minus4 as u32 + 4;

    let mut entries = Vec::with_capacity(num_ref_entries as usize);
    for i in 0..num_ref_entries {
        let inter_layer_ref_pic_flag = if inter_layer_prediction_enabled_flag {
            br.u1()? == 1
        } else {
            false
        };
        if !inter_layer_ref_pic_flag {
            // Default: entry is an STRP (st_ref_pic_flag inferred to 1
            // when not present — §7.4.11).
            let st_ref_pic_flag = if long_term_ref_pics_flag {
                br.u1()? == 1
            } else {
                true
            };
            if st_ref_pic_flag {
                let abs_delta_poc_st = br.ue()?;
                // §7.4.11: abs_delta_poc_st shall be in [0, 2^15 - 1].
                if abs_delta_poc_st > 0x7FFF {
                    return Err(Error::invalid(format!(
                        "h266 RPL[{list_idx}][{rpls_idx}][{i}]: abs_delta_poc_st out of range ({abs_delta_poc_st})"
                    )));
                }
                // Equation (150): +1 unless gated by weighted pred on i>0.
                let gated = (weighted_pred_flag || weighted_bipred_flag) && i != 0;
                let abs = if gated {
                    abs_delta_poc_st
                } else {
                    abs_delta_poc_st + 1
                };
                let (sign_flag, delta) = if abs > 0 {
                    let s = br.u1()? == 1;
                    let signed = if s { -(abs as i32) } else { abs as i32 };
                    (s, signed)
                } else {
                    (false, 0)
                };
                entries.push(RefPicListEntry::ShortTerm {
                    abs_delta_poc_st,
                    strp_entry_sign_flag: sign_flag,
                    delta_poc_val_st: delta,
                });
            } else {
                // LT entry. POC LSB is either inlined (u(v)) or deferred
                // to the containing PH/slice header.
                let poc_lsb_lt = if !ltrp_in_header_flag {
                    Some(br.u(poc_lsb_lt_width)?)
                } else {
                    None
                };
                entries.push(RefPicListEntry::LongTerm { poc_lsb_lt });
            }
        } else {
            let ilrp_idx = br.ue()?;
            entries.push(RefPicListEntry::InterLayer { ilrp_idx });
        }
    }

    Ok(RefPicListStruct {
        entries,
        ltrp_in_header_flag,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Push `v` as `n` bits MSB-first.
    fn push_u(bits: &mut Vec<u8>, v: u64, n: u32) {
        for i in (0..n).rev() {
            bits.push(((v >> i) & 1) as u8);
        }
    }

    /// Encode an unsigned Exp-Golomb `ue(v)` into the bit stream.
    fn push_ue(bits: &mut Vec<u8>, value: u32) {
        // length = 2 * floor(log2(value+1)) + 1
        let code_num = value as u64 + 1;
        let mut zeros: u32 = 0;
        while (1u64 << (zeros + 1)) <= code_num {
            zeros += 1;
        }
        // zeros leading zeros, then the `zeros+1`-bit representation of `code_num`.
        for _ in 0..zeros {
            bits.push(0);
        }
        push_u(bits, code_num, zeros + 1);
    }

    /// Encode a signed Exp-Golomb `se(v)`.
    fn push_se(bits: &mut Vec<u8>, value: i32) {
        let code = if value <= 0 {
            (-(value as i64) * 2) as u32
        } else {
            (value as i64 * 2 - 1) as u32
        };
        push_ue(bits, code);
    }

    fn pack(bits: &[u8]) -> Vec<u8> {
        let mut padded = bits.to_vec();
        while padded.len() % 8 != 0 {
            padded.push(0);
        }
        let mut bytes = Vec::with_capacity(padded.len() / 8);
        for chunk in padded.chunks(8) {
            let mut b = 0u8;
            for (i, &bit) in chunk.iter().enumerate() {
                b |= bit << (7 - i);
            }
            bytes.push(b);
        }
        bytes
    }

    /// Build a minimal SPS bit-sequence with all tools disabled, using
    /// `chroma_format_idc = 1` (4:2:0), `log2_ctu_size_minus5 = 2`
    /// (CtbSizeY = 128), and no PTL/DPB. This is the canonical "zero-
    /// everything" tail used as a baseline for several tests.
    fn build_minimal_sps_bits() -> Vec<u8> {
        let mut bits: Vec<u8> = Vec::new();
        push_u(&mut bits, 0, 4); // sps_id
        push_u(&mut bits, 0, 4); // vps_id
        push_u(&mut bits, 0, 3); // max_sublayers_minus1
        push_u(&mut bits, 1, 2); // chroma = 4:2:0
        push_u(&mut bits, 2, 2); // log2_ctu - 5 = 2 → CTB=128
        push_u(&mut bits, 0, 1); // ptl_dpb_hrd_present = 0
        push_u(&mut bits, 0, 1); // gdr_enabled
        push_u(&mut bits, 0, 1); // ref_pic_resampling
        push_ue(&mut bits, 320); // pic_width
        push_ue(&mut bits, 240); // pic_height
        push_u(&mut bits, 0, 1); // conformance_window_flag
        push_u(&mut bits, 0, 1); // subpic_info_present
        push_ue(&mut bits, 2); // bitdepth_minus8 = 2 → 10-bit
        push_u(&mut bits, 0, 1); // entropy_coding_sync
        push_u(&mut bits, 0, 1); // entry_point_offsets
        push_u(&mut bits, 4, 4); // log2_max_poc_lsb_minus4
        push_u(&mut bits, 0, 1); // poc_msb_cycle_flag
        push_u(&mut bits, 0, 2); // num_extra_ph_bytes
        push_u(&mut bits, 0, 2); // num_extra_sh_bytes

        // ---- partition constraints (§7.3.2.4 tail) ----
        push_ue(&mut bits, 0); // log2_min_luma_cb_size_minus2
        push_u(&mut bits, 0, 1); // partition_constraints_override_enabled
        push_ue(&mut bits, 0); // log2_diff_min_qt_min_cb_intra_luma
        push_ue(&mut bits, 0); // max_mtt_depth_intra_luma (=> skip bt/tt)
                               // chroma_format_idc = 1 ≠ 0 → emit qtbtt_dual_tree_intra_flag
        push_u(&mut bits, 0, 1); // qtbtt_dual_tree_intra = 0 → skip chroma block
        push_ue(&mut bits, 0); // log2_diff_min_qt_min_cb_inter
        push_ue(&mut bits, 0); // max_mtt_depth_inter (=> skip bt/tt)
                               // CtbSizeY = 128 > 32 → emit max_luma_transform_size_64_flag
        push_u(&mut bits, 0, 1); // max_luma_transform_size_64_flag

        // ---- tool flags (§7.3.2.4 tail) ----
        push_u(&mut bits, 0, 1); // transform_skip_enabled
        push_u(&mut bits, 0, 1); // mts_enabled
        push_u(&mut bits, 0, 1); // lfnst_enabled
                                 // chroma_format_idc ≠ 0 → joint_cbcr + same_qp_table + tables
        push_u(&mut bits, 0, 1); // joint_cbcr_enabled
        push_u(&mut bits, 1, 1); // same_qp_table_for_chroma (→ 1 table)
        push_se(&mut bits, 0); // qp_table_start_minus26 (se)
        push_ue(&mut bits, 0); // num_points_minus1 = 0 (=> 1 point)
        push_ue(&mut bits, 0); // delta_qp_in_val_minus1[0][0]
        push_ue(&mut bits, 0); // delta_qp_diff_val[0][0]
        push_u(&mut bits, 0, 1); // sao_enabled
        push_u(&mut bits, 0, 1); // alf_enabled (=> no ccalf)
        push_u(&mut bits, 0, 1); // lmcs_enabled
        push_u(&mut bits, 0, 1); // weighted_pred
        push_u(&mut bits, 0, 1); // weighted_bipred
        push_u(&mut bits, 0, 1); // long_term_ref_pics
                                 // vps_id = 0 → no inter_layer_prediction_enabled
        push_u(&mut bits, 0, 1); // idr_rpl_present
        push_u(&mut bits, 0, 1); // rpl1_same_as_rpl0 = 0 → 2 loops
        push_ue(&mut bits, 0); // num_ref_pic_lists[0] = 0
        push_ue(&mut bits, 0); // num_ref_pic_lists[1] = 0
        push_u(&mut bits, 0, 1); // ref_wraparound
        push_u(&mut bits, 0, 1); // temporal_mvp (=> no sbtmvp)
        push_u(&mut bits, 0, 1); // amvr
        push_u(&mut bits, 0, 1); // bdof (=> no bdof_ctrl)
        push_u(&mut bits, 0, 1); // smvd
        push_u(&mut bits, 0, 1); // dmvr (=> no dmvr_ctrl)
        push_u(&mut bits, 0, 1); // mmvd (=> no mmvd_fullpel)
        push_ue(&mut bits, 0); // six_minus_max_num_merge_cand → MaxNumMergeCand=6
        push_u(&mut bits, 0, 1); // sbt
        push_u(&mut bits, 0, 1); // affine (=> skip affine block)
        push_u(&mut bits, 0, 1); // bcw
        push_u(&mut bits, 0, 1); // ciip
                                 // MaxNumMergeCand=6 >= 2 → gpm_enabled
        push_u(&mut bits, 0, 1); // gpm_enabled = 0
        push_ue(&mut bits, 0); // log2_parallel_merge_level_minus2
        push_u(&mut bits, 0, 1); // isp
        push_u(&mut bits, 0, 1); // mrl
        push_u(&mut bits, 0, 1); // mip
        push_u(&mut bits, 0, 1); // cclm (chroma_format_idc ≠ 0)
                                 // chroma_format_idc == 1 → chroma-collocated flags
        push_u(&mut bits, 0, 1); // chroma_horizontal_collocated
        push_u(&mut bits, 0, 1); // chroma_vertical_collocated
        push_u(&mut bits, 0, 1); // palette
                                 // chroma_format_idc != 3 → no act flag
                                 // transform_skip=0 && palette=0 → no min_qp_prime_ts
        push_u(&mut bits, 0, 1); // ibc
        push_u(&mut bits, 0, 1); // ladf
        push_u(&mut bits, 0, 1); // explicit_scaling_list
        push_u(&mut bits, 0, 1); // dep_quant
        push_u(&mut bits, 0, 1); // sign_data_hiding
        push_u(&mut bits, 0, 1); // virtual_boundaries_enabled
                                 // Tail block (§7.3.2.4): no HRD timing (ptl_dpb_hrd=0 →
                                 // sps_timing_hrd_params_present_flag is not emitted), then
                                 // sps_field_seq_flag=0, sps_vui_parameters_present_flag=0,
                                 // sps_extension_flag=0.
        push_u(&mut bits, 0, 1); // sps_field_seq_flag
        push_u(&mut bits, 0, 1); // sps_vui_parameters_present_flag
        push_u(&mut bits, 0, 1); // sps_extension_flag
        bits
    }

    /// Variant of [`build_minimal_sps_bits`] that injects
    /// `sps_num_extra_ph_bytes` / `sps_num_extra_sh_bytes` (each 1 byte)
    /// with the supplied present-flag bit patterns, so the §7.4.3.4
    /// `NumExtra{Ph,Sh}Bits` derivation can be exercised.
    fn build_sps_bits_with_extra_bits(ph_flags: &[u8], sh_flags: &[u8]) -> Vec<u8> {
        let mut bits: Vec<u8> = Vec::new();
        push_u(&mut bits, 0, 4); // sps_id
        push_u(&mut bits, 0, 4); // vps_id
        push_u(&mut bits, 0, 3); // max_sublayers_minus1
        push_u(&mut bits, 1, 2); // chroma = 4:2:0
        push_u(&mut bits, 2, 2); // log2_ctu - 5 = 2 → CTB=128
        push_u(&mut bits, 0, 1); // ptl_dpb_hrd_present = 0
        push_u(&mut bits, 0, 1); // gdr_enabled
        push_u(&mut bits, 0, 1); // ref_pic_resampling
        push_ue(&mut bits, 320); // pic_width
        push_ue(&mut bits, 240); // pic_height
        push_u(&mut bits, 0, 1); // conformance_window_flag
        push_u(&mut bits, 0, 1); // subpic_info_present
        push_ue(&mut bits, 2); // bitdepth_minus8 = 2 → 10-bit
        push_u(&mut bits, 0, 1); // entropy_coding_sync
        push_u(&mut bits, 0, 1); // entry_point_offsets
        push_u(&mut bits, 4, 4); // log2_max_poc_lsb_minus4
        push_u(&mut bits, 0, 1); // poc_msb_cycle_flag
        push_u(&mut bits, 1, 2); // num_extra_ph_bytes = 1 → 8 present flags
        for &f in ph_flags {
            push_u(&mut bits, f as u64, 1);
        }
        push_u(&mut bits, 1, 2); // num_extra_sh_bytes = 1 → 8 present flags
        for &f in sh_flags {
            push_u(&mut bits, f as u64, 1);
        }
        // Remaining body is identical to build_minimal_sps_bits from the
        // partition-constraints block onward.
        push_ue(&mut bits, 0); // log2_min_luma_cb_size_minus2
        push_u(&mut bits, 0, 1); // partition_constraints_override_enabled
        push_ue(&mut bits, 0); // log2_diff_min_qt_min_cb_intra_luma
        push_ue(&mut bits, 0); // max_mtt_depth_intra_luma
        push_u(&mut bits, 0, 1); // qtbtt_dual_tree_intra = 0
        push_ue(&mut bits, 0); // log2_diff_min_qt_min_cb_inter
        push_ue(&mut bits, 0); // max_mtt_depth_inter
        push_u(&mut bits, 0, 1); // max_luma_transform_size_64_flag
        push_u(&mut bits, 0, 1); // transform_skip_enabled
        push_u(&mut bits, 0, 1); // mts_enabled
        push_u(&mut bits, 0, 1); // lfnst_enabled
        push_u(&mut bits, 0, 1); // joint_cbcr_enabled
        push_u(&mut bits, 1, 1); // same_qp_table_for_chroma
        push_se(&mut bits, 0); // qp_table_start_minus26
        push_ue(&mut bits, 0); // num_points_minus1
        push_ue(&mut bits, 0); // delta_qp_in_val_minus1[0][0]
        push_ue(&mut bits, 0); // delta_qp_diff_val[0][0]
        push_u(&mut bits, 0, 1); // sao_enabled
        push_u(&mut bits, 0, 1); // alf_enabled
        push_u(&mut bits, 0, 1); // lmcs_enabled
        push_u(&mut bits, 0, 1); // weighted_pred
        push_u(&mut bits, 0, 1); // weighted_bipred
        push_u(&mut bits, 0, 1); // long_term_ref_pics
        push_u(&mut bits, 0, 1); // idr_rpl_present
        push_u(&mut bits, 0, 1); // rpl1_same_as_rpl0 = 0
        push_ue(&mut bits, 0); // num_ref_pic_lists[0]
        push_ue(&mut bits, 0); // num_ref_pic_lists[1]
        push_u(&mut bits, 0, 1); // ref_wraparound
        push_u(&mut bits, 0, 1); // temporal_mvp
        push_u(&mut bits, 0, 1); // amvr
        push_u(&mut bits, 0, 1); // bdof
        push_u(&mut bits, 0, 1); // smvd
        push_u(&mut bits, 0, 1); // dmvr
        push_u(&mut bits, 0, 1); // mmvd
        push_ue(&mut bits, 0); // six_minus_max_num_merge_cand
        push_u(&mut bits, 0, 1); // sbt
        push_u(&mut bits, 0, 1); // affine
        push_u(&mut bits, 0, 1); // bcw
        push_u(&mut bits, 0, 1); // ciip
        push_u(&mut bits, 0, 1); // gpm_enabled
        push_ue(&mut bits, 0); // log2_parallel_merge_level_minus2
        push_u(&mut bits, 0, 1); // isp
        push_u(&mut bits, 0, 1); // mrl
        push_u(&mut bits, 0, 1); // mip
        push_u(&mut bits, 0, 1); // cclm
        push_u(&mut bits, 0, 1); // chroma_horizontal_collocated
        push_u(&mut bits, 0, 1); // chroma_vertical_collocated
        push_u(&mut bits, 0, 1); // palette
        push_u(&mut bits, 0, 1); // ibc
        push_u(&mut bits, 0, 1); // ladf
        push_u(&mut bits, 0, 1); // explicit_scaling_list
        push_u(&mut bits, 0, 1); // dep_quant
        push_u(&mut bits, 0, 1); // sign_data_hiding
        push_u(&mut bits, 0, 1); // virtual_boundaries_enabled
        push_u(&mut bits, 0, 1); // sps_field_seq_flag
        push_u(&mut bits, 0, 1); // sps_vui_parameters_present_flag
        push_u(&mut bits, 0, 1); // sps_extension_flag
        bits
    }

    /// §7.4.3.4 eq. 41 — `NumExtraPhBits` / `NumExtraShBits` count the
    /// `sps_extra_{ph,sh}_bit_present_flag[i]` equal to 1, not the
    /// `sps_num_extra_{ph,sh}_bytes * 8` upper bound.
    #[test]
    fn sps_derives_num_extra_ph_sh_bits_from_present_flags() {
        // 3 PH flags set (positions 0, 2, 5), 2 SH flags set (1, 7).
        let ph = [1u8, 0, 1, 0, 0, 1, 0, 0];
        let sh = [0u8, 1, 0, 0, 0, 0, 0, 1];
        let bits = build_sps_bits_with_extra_bits(&ph, &sh);
        let bytes = pack(&bits);
        let sps = parse_sps(&bytes).unwrap();
        assert_eq!(sps.sps_num_extra_ph_bytes, 1);
        assert_eq!(sps.sps_num_extra_sh_bytes, 1);
        assert_eq!(sps.num_extra_ph_bits, 3);
        assert_eq!(sps.num_extra_sh_bits, 2);

        // All-zero present flags → counts are 0 even though a byte each
        // was signalled.
        let zero = [0u8; 8];
        let bits0 = build_sps_bits_with_extra_bits(&zero, &zero);
        let sps0 = parse_sps(&pack(&bits0)).unwrap();
        assert_eq!(sps0.num_extra_ph_bits, 0);
        assert_eq!(sps0.num_extra_sh_bits, 0);
    }

    /// §7.4.3.4 eqs. 54 – 57 — the single-point all-zero-delta table
    /// (`qpIn = [26, 27]`, `qpOut = [26, 26]`) is identity below the
    /// start QP 26, plateaus at 26 across `k ∈ {26, 27}`, then increments
    /// by 1 per step above. (Note: this is NOT the previous
    /// `chroma_qp_identity` clamp — that approximation diverged from the
    /// spec table at `k >= 27`.)
    #[test]
    fn chroma_qp_table_single_zero_point_matches_spec() {
        use crate::sps::ChromaQpTable;
        let t = ChromaQpTable {
            qp_table_start_minus26: 0,
            entries: vec![(0, 0)],
        };
        let qp_bd_offset = 0; // 8-bit
        let built = t.build(qp_bd_offset);
        // Identity below the start anchor.
        for k in 0..=26 {
            assert_eq!(ChromaQpTable::map_qp(&built, k, qp_bd_offset), k);
        }
        // Plateau: T[27] = 26, then +1 per step above.
        assert_eq!(ChromaQpTable::map_qp(&built, 27, qp_bd_offset), 26);
        assert_eq!(ChromaQpTable::map_qp(&built, 28, qp_bd_offset), 27);
        assert_eq!(ChromaQpTable::map_qp(&built, 63, qp_bd_offset), 62);
        // Clamp below / above the index range.
        assert_eq!(ChromaQpTable::map_qp(&built, -5, qp_bd_offset), 0);
        assert_eq!(ChromaQpTable::map_qp(&built, 99, qp_bd_offset), 62);
    }

    /// §7.4.3.4 eqs. 54 – 57 — a non-identity pivot table maps qpInVal to
    /// the derived qpOutVal and interpolates between pivots. With start 26
    /// and one point `(delta_in_minus1 = 3, delta_diff = 1)`:
    ///   qpIn  = [26, 30], qpOut = [26, 26 + (3 ^ 1)] = [26, 28].
    /// The anchor `ChromaQpTable[26] = 26`; interpolation over k=27..30
    /// adds `(2 * m + sh) / 4` with `sh = (3 + 1) >> 1 = 2`.
    #[test]
    fn chroma_qp_table_interpolates_between_pivots() {
        use crate::sps::ChromaQpTable;
        let t = ChromaQpTable {
            qp_table_start_minus26: 0, // start 26
            entries: vec![(3, 1)],     // delta_in_minus1 = 3, delta_diff = 1
        };
        let built = t.build(0);
        // Anchor.
        assert_eq!(ChromaQpTable::map_qp(&built, 26, 0), 26);
        // Interp: anchor + (out_delta * m + sh) / denom, out_delta = 2,
        // denom = 4, sh = 2.
        // k=27 m=1: 26 + (2*1 + 2)/4 = 26 + 1 = 27
        // k=28 m=2: 26 + (2*2 + 2)/4 = 26 + 1 = 27
        // k=29 m=3: 26 + (2*3 + 2)/4 = 26 + 2 = 28
        // k=30 m=4: 26 + (2*4 + 2)/4 = 26 + 2 = 28 (= qpOut[1])
        assert_eq!(ChromaQpTable::map_qp(&built, 27, 0), 27);
        assert_eq!(ChromaQpTable::map_qp(&built, 28, 0), 27);
        assert_eq!(ChromaQpTable::map_qp(&built, 29, 0), 28);
        assert_eq!(ChromaQpTable::map_qp(&built, 30, 0), 28);
        // Above the last pivot increments by 1 (clamped at 63).
        assert_eq!(ChromaQpTable::map_qp(&built, 31, 0), 29);
        assert_eq!(ChromaQpTable::map_qp(&built, 33, 0), 31);
        // Below the start anchor decrements by 1 (clamped at 0).
        assert_eq!(ChromaQpTable::map_qp(&built, 25, 0), 25);
        assert_eq!(ChromaQpTable::map_qp(&built, 0, 0), 0);
    }

    #[test]
    fn minimal_sps_320x240_10bit_tail_parses() {
        let bits = build_minimal_sps_bits();
        let bytes = pack(&bits);
        let sps = parse_sps(&bytes).unwrap();

        // Identifier / format sanity (regression for round-1 fields).
        assert_eq!(sps.sps_chroma_format_idc, 1);
        assert_eq!(sps.ctb_size(), 128);
        assert_eq!(sps.sps_pic_width_max_in_luma_samples, 320);
        assert_eq!(sps.sps_pic_height_max_in_luma_samples, 240);
        assert_eq!(sps.bit_depth_y(), 10);

        // No DPB because sps_ptl_dpb_hrd_params_present_flag = 0.
        assert!(sps.dpb_parameters.is_none());

        // Partition constraints: all zero, max_luma_transform_size_64 present.
        let p = &sps.partition_constraints;
        assert_eq!(p.log2_min_luma_coding_block_size_minus2, 0);
        assert!(!p.partition_constraints_override_enabled_flag);
        assert!(!p.qtbtt_dual_tree_intra_flag);
        assert!(!p.max_luma_transform_size_64_flag);

        // Tool flags: one QP table (same_qp_table_for_chroma=1),
        // everything else off.
        let t = &sps.tool_flags;
        assert!(!t.transform_skip_enabled_flag);
        assert!(!t.mts_enabled_flag);
        assert!(!t.lfnst_enabled_flag);
        assert!(t.same_qp_table_for_chroma_flag);
        assert_eq!(t.chroma_qp_tables.len(), 1);
        assert_eq!(t.chroma_qp_tables[0].qp_table_start_minus26, 0);
        assert_eq!(t.chroma_qp_tables[0].entries.len(), 1);
        assert!(!t.sao_enabled_flag);
        assert!(!t.alf_enabled_flag);
        assert!(!t.ccalf_enabled_flag);
        assert!(!t.lmcs_enabled_flag);
        assert_eq!(t.num_ref_pic_lists, [0, 0]);
        assert_eq!(t.six_minus_max_num_merge_cand, 0);
        assert!(!t.gpm_enabled_flag);
        assert!(!t.affine_enabled_flag);
        assert!(!t.palette_enabled_flag);
        assert!(!t.ibc_enabled_flag);
        assert!(!t.ladf_enabled_flag);
        assert!(!t.explicit_scaling_list_enabled_flag);
        assert!(!t.dep_quant_enabled_flag);
        assert!(!t.virtual_boundaries_enabled_flag);
    }

    /// An SPS with `sps_subpic_info_present_flag = 1` and a single
    /// subpicture (num_subpics_minus1 = 0) parses successfully, with
    /// the derived geometry inferred from the picture dimensions.
    #[test]
    fn single_subpicture_sps_parses() {
        // Build a minimal SPS but flip subpic_info_present = 1 and
        // supply `sps_num_subpics_minus1 = 0` + `sps_subpic_id_len_minus1 = 0`
        // + `sps_subpic_id_mapping_explicitly_signalled_flag = 0`. Nothing
        // in the per-subpicture loop is transmitted because all gates
        // close on i == 0 / i == num_subpics_minus1.
        let mut bits: Vec<u8> = Vec::new();
        push_u(&mut bits, 0, 4); // sps_id
        push_u(&mut bits, 0, 4); // vps_id
        push_u(&mut bits, 0, 3); // max_sublayers
        push_u(&mut bits, 1, 2); // chroma = 4:2:0
        push_u(&mut bits, 2, 2); // log2_ctu - 5 = 2 → CTB = 128
        push_u(&mut bits, 0, 1); // ptl_present
        push_u(&mut bits, 0, 1); // gdr
        push_u(&mut bits, 0, 1); // ref_pic_resampling (=> no res_change)
        push_ue(&mut bits, 320); // width
        push_ue(&mut bits, 240); // height
        push_u(&mut bits, 0, 1); // conformance_window_flag
        push_u(&mut bits, 1, 1); // subpic_info_present = 1
                                 // Subpic block: num_subpics_minus1 = 0 → no indep/same flags,
                                 // no per-entry loop body.
        push_ue(&mut bits, 0);
        push_ue(&mut bits, 0); // subpic_id_len_minus1 = 0
        push_u(&mut bits, 0, 1); // subpic_id_mapping_explicitly_signalled_flag = 0
                                 // Continue with the usual identifier tail.
        push_ue(&mut bits, 2); // bitdepth_minus8
        push_u(&mut bits, 0, 1); // entropy_coding_sync
        push_u(&mut bits, 0, 1); // entry_point_offsets
        push_u(&mut bits, 4, 4); // log2_max_poc_lsb_minus4 = 4
        push_u(&mut bits, 0, 1); // poc_msb_cycle
        push_u(&mut bits, 0, 2); // num_extra_ph_bytes
        push_u(&mut bits, 0, 2); // num_extra_sh_bytes
                                 // Partition constraints — all-zero.
        push_ue(&mut bits, 0);
        push_u(&mut bits, 0, 1);
        push_ue(&mut bits, 0);
        push_ue(&mut bits, 0);
        push_u(&mut bits, 0, 1); // qtbtt_dual_tree_intra = 0
        push_ue(&mut bits, 0);
        push_ue(&mut bits, 0);
        push_u(&mut bits, 0, 1); // max_luma_transform_size_64 (CtbSizeY=128)
                                 // Tool flags — mirror build_minimal_sps_bits' minimal tail.
        push_u(&mut bits, 0, 1); // transform_skip
        push_u(&mut bits, 0, 1); // mts
        push_u(&mut bits, 0, 1); // lfnst
        push_u(&mut bits, 0, 1); // joint_cbcr
        push_u(&mut bits, 1, 1); // same_qp_table_for_chroma
        push_se(&mut bits, 0);
        push_ue(&mut bits, 0);
        push_ue(&mut bits, 0);
        push_ue(&mut bits, 0);
        push_u(&mut bits, 0, 1); // sao
        push_u(&mut bits, 0, 1); // alf
        push_u(&mut bits, 0, 1); // lmcs
        push_u(&mut bits, 0, 1); // weighted_pred
        push_u(&mut bits, 0, 1); // weighted_bipred
        push_u(&mut bits, 0, 1); // long_term_ref_pics
        push_u(&mut bits, 0, 1); // idr_rpl
        push_u(&mut bits, 0, 1); // rpl1_same_as_rpl0
        push_ue(&mut bits, 0);
        push_ue(&mut bits, 0);
        push_u(&mut bits, 0, 1); // ref_wraparound
        push_u(&mut bits, 0, 1); // temporal_mvp
        push_u(&mut bits, 0, 1); // amvr
        push_u(&mut bits, 0, 1); // bdof
        push_u(&mut bits, 0, 1); // smvd
        push_u(&mut bits, 0, 1); // dmvr
        push_u(&mut bits, 0, 1); // mmvd
        push_ue(&mut bits, 0);
        push_u(&mut bits, 0, 1); // sbt
        push_u(&mut bits, 0, 1); // affine
        push_u(&mut bits, 0, 1); // bcw
        push_u(&mut bits, 0, 1); // ciip
        push_u(&mut bits, 0, 1); // gpm
        push_ue(&mut bits, 0);
        push_u(&mut bits, 0, 1); // isp
        push_u(&mut bits, 0, 1); // mrl
        push_u(&mut bits, 0, 1); // mip
        push_u(&mut bits, 0, 1); // cclm
        push_u(&mut bits, 0, 1); // chroma_horizontal_collocated
        push_u(&mut bits, 0, 1); // chroma_vertical_collocated
        push_u(&mut bits, 0, 1); // palette
        push_u(&mut bits, 0, 1); // ibc
        push_u(&mut bits, 0, 1); // ladf
        push_u(&mut bits, 0, 1); // explicit_scaling_list
        push_u(&mut bits, 0, 1); // dep_quant
        push_u(&mut bits, 0, 1); // sign_data_hiding
        push_u(&mut bits, 0, 1); // virtual_boundaries_enabled
        push_u(&mut bits, 0, 1); // sps_field_seq_flag
        push_u(&mut bits, 0, 1); // sps_vui_parameters_present_flag
        push_u(&mut bits, 0, 1); // sps_extension_flag

        let bytes = pack(&bits);
        let sps = parse_sps(&bytes).unwrap();
        assert!(sps.sps_subpic_info_present_flag);
        let sp = sps.subpic_info.as_ref().expect("subpic block parsed");
        assert_eq!(sp.num_subpics_minus1, 0);
        assert_eq!(sp.subpics.len(), 1);
        // Inferred: treated_as_pic_flag = 1 (independent subpics flag
        // inferred to 1), loop-filter flag = 0.
        assert!(sp.subpics[0].treated_as_pic_flag);
        assert!(!sp.subpics[0].loop_filter_across_subpic_enabled_flag);
        // tmpWidthVal = ceil(320/128) = 3, height = ceil(240/128) = 2.
        // Since i == 0 and i == num_subpics_minus1, width/height are
        // inferred to (tmpWidthVal - top_left - 1), i.e. full grid.
        assert_eq!(sp.subpics[0].width_minus1, 2);
        assert_eq!(sp.subpics[0].height_minus1, 1);
        assert!(!sp.subpic_id_mapping_explicitly_signalled_flag);
    }

    /// SPS with `sps_res_change_in_clvs_allowed_flag = 1` cannot have
    /// subpic info — the parser rejects that combination.
    #[test]
    fn subpic_info_with_res_change_rejected() {
        let mut bits: Vec<u8> = Vec::new();
        push_u(&mut bits, 0, 4); // sps_id
        push_u(&mut bits, 0, 4); // vps_id
        push_u(&mut bits, 0, 3); // max_sublayers
        push_u(&mut bits, 1, 2); // chroma
        push_u(&mut bits, 0, 2); // log2_ctu - 5 = 0 → 32
        push_u(&mut bits, 0, 1); // ptl_present
        push_u(&mut bits, 0, 1); // gdr
        push_u(&mut bits, 1, 1); // ref_pic_resampling = 1
        push_u(&mut bits, 1, 1); // res_change_in_clvs_allowed = 1
        push_ue(&mut bits, 320);
        push_ue(&mut bits, 240);
        push_u(&mut bits, 0, 1); // conformance_window_flag
        push_u(&mut bits, 1, 1); // subpic_info_present = 1
        let bytes = pack(&bits);
        assert!(parse_sps(&bytes).is_err());
    }

    /// Enabling SAO + ALF + CCALF + LMCS should flip the parsed bits
    /// exactly. Because `ccalf_enabled_flag` is only emitted when
    /// `alf_enabled_flag` is set *and* `chroma_format_idc != 0`, this
    /// also regression-guards the conditional read.
    #[test]
    fn loop_filter_enable_flags_are_honoured() {
        // Start from the minimal 4:2:0 SPS and patch the three flags.
        let mut bits = build_minimal_sps_bits();
        // Locate the exact offsets: the minimal builder emits
        //   ...lfnst(1), joint_cbcr(1), same_qp(1), se(0)=1, ue(0)=1,
        //   ue(0)=1, ue(0)=1, sao(1), alf(1), [ccalf only if alf=1],
        //   lmcs(1)...
        // To keep the test robust, rebuild the sequence from scratch
        // with the different flag values rather than patch bits in place.
        //
        // We inline the full sequence from `build_minimal_sps_bits` up to
        // the filter block and change sao/alf/lmcs.
        bits.clear();
        push_u(&mut bits, 0, 4);
        push_u(&mut bits, 0, 4);
        push_u(&mut bits, 0, 3);
        push_u(&mut bits, 1, 2);
        push_u(&mut bits, 2, 2);
        push_u(&mut bits, 0, 1);
        push_u(&mut bits, 0, 1);
        push_u(&mut bits, 0, 1);
        push_ue(&mut bits, 320);
        push_ue(&mut bits, 240);
        push_u(&mut bits, 0, 1);
        push_u(&mut bits, 0, 1);
        push_ue(&mut bits, 2);
        push_u(&mut bits, 0, 1);
        push_u(&mut bits, 0, 1);
        push_u(&mut bits, 4, 4);
        push_u(&mut bits, 0, 1);
        push_u(&mut bits, 0, 2);
        push_u(&mut bits, 0, 2);

        // Partition constraints — same as minimal.
        push_ue(&mut bits, 0);
        push_u(&mut bits, 0, 1);
        push_ue(&mut bits, 0);
        push_ue(&mut bits, 0);
        push_u(&mut bits, 0, 1); // qtbtt_dual_tree_intra = 0
        push_ue(&mut bits, 0);
        push_ue(&mut bits, 0);
        push_u(&mut bits, 0, 1); // max_luma_transform_size_64_flag

        // Tool flags with SAO + ALF + CCALF + LMCS enabled.
        push_u(&mut bits, 0, 1); // transform_skip
        push_u(&mut bits, 0, 1); // mts
        push_u(&mut bits, 0, 1); // lfnst
        push_u(&mut bits, 0, 1); // joint_cbcr
        push_u(&mut bits, 1, 1); // same_qp_table_for_chroma
        push_se(&mut bits, 0);
        push_ue(&mut bits, 0);
        push_ue(&mut bits, 0);
        push_ue(&mut bits, 0);
        push_u(&mut bits, 1, 1); // sao_enabled = 1
        push_u(&mut bits, 1, 1); // alf_enabled = 1
        push_u(&mut bits, 1, 1); // ccalf_enabled = 1 (alf=1 && chroma!=0)
        push_u(&mut bits, 1, 1); // lmcs_enabled = 1
        push_u(&mut bits, 0, 1);
        push_u(&mut bits, 0, 1);
        push_u(&mut bits, 0, 1);
        push_u(&mut bits, 0, 1); // idr_rpl
        push_u(&mut bits, 0, 1); // rpl1_same_as_rpl0
        push_ue(&mut bits, 0);
        push_ue(&mut bits, 0);
        push_u(&mut bits, 0, 1);
        push_u(&mut bits, 0, 1);
        push_u(&mut bits, 0, 1);
        push_u(&mut bits, 0, 1);
        push_u(&mut bits, 0, 1);
        push_u(&mut bits, 0, 1);
        push_u(&mut bits, 0, 1);
        push_ue(&mut bits, 0);
        push_u(&mut bits, 0, 1); // sbt
        push_u(&mut bits, 0, 1); // affine
        push_u(&mut bits, 0, 1); // bcw
        push_u(&mut bits, 0, 1); // ciip
        push_u(&mut bits, 0, 1); // gpm
        push_ue(&mut bits, 0); // log2_parallel_merge_level
        push_u(&mut bits, 0, 1);
        push_u(&mut bits, 0, 1);
        push_u(&mut bits, 0, 1);
        push_u(&mut bits, 0, 1); // cclm
        push_u(&mut bits, 0, 1);
        push_u(&mut bits, 0, 1);
        push_u(&mut bits, 0, 1); // palette
        push_u(&mut bits, 0, 1); // ibc
        push_u(&mut bits, 0, 1); // ladf
        push_u(&mut bits, 0, 1); // explicit_scaling_list
        push_u(&mut bits, 0, 1); // dep_quant
        push_u(&mut bits, 0, 1); // sign_data_hiding
        push_u(&mut bits, 0, 1); // virtual_boundaries_enabled
                                 // Tail: field_seq=0, vui=0, ext=0.
        push_u(&mut bits, 0, 1); // sps_field_seq_flag
        push_u(&mut bits, 0, 1); // sps_vui_parameters_present_flag
        push_u(&mut bits, 0, 1); // sps_extension_flag

        let bytes = pack(&bits);
        let sps = parse_sps(&bytes).unwrap();
        let t = &sps.tool_flags;
        assert!(t.sao_enabled_flag);
        assert!(t.alf_enabled_flag);
        assert!(t.ccalf_enabled_flag);
        assert!(t.lmcs_enabled_flag);
    }

    /// Common preamble used by the rpl tests — emits everything from the
    /// SPS identifier block through the `long_term_ref_pics_flag` gate,
    /// parameterised on the three flags that change RPL semantics.
    fn emit_sps_preamble_for_rpl(
        bits: &mut Vec<u8>,
        long_term: bool,
        inter_layer: bool,
        weighted_pred: bool,
    ) {
        push_u(bits, 0, 4); // sps_id
                            // `inter_layer_prediction_enabled_flag` is only transmitted when
                            // `vps_id > 0`, so use vps_id=1 when we want ILRP support.
        push_u(bits, if inter_layer { 1 } else { 0 }, 4); // vps_id
        push_u(bits, 0, 3); // max_sublayers_minus1
        push_u(bits, 1, 2); // chroma = 4:2:0
        push_u(bits, 2, 2); // log2_ctu - 5 = 2 → CTB=128
        push_u(bits, 0, 1); // ptl_dpb_hrd_present
        push_u(bits, 0, 1); // gdr
        push_u(bits, 0, 1); // ref_pic_resampling
        push_ue(bits, 320);
        push_ue(bits, 240);
        push_u(bits, 0, 1); // conformance_window
        push_u(bits, 0, 1); // subpic_info_present
        push_ue(bits, 2); // bitdepth_minus8 = 2 (10-bit)
        push_u(bits, 0, 1); // entropy_coding_sync
        push_u(bits, 0, 1); // entry_point_offsets
        push_u(bits, 4, 4); // log2_max_poc_lsb_minus4 = 4 → 8-bit LSB
        push_u(bits, 0, 1); // poc_msb_cycle_flag
        push_u(bits, 0, 2); // num_extra_ph_bytes
        push_u(bits, 0, 2); // num_extra_sh_bytes

        // Partition constraints — all-zero.
        push_ue(bits, 0);
        push_u(bits, 0, 1);
        push_ue(bits, 0);
        push_ue(bits, 0);
        push_u(bits, 0, 1); // qtbtt_dual_tree_intra = 0
        push_ue(bits, 0);
        push_ue(bits, 0);
        push_u(bits, 0, 1); // max_luma_transform_size_64 = 0

        // Tool flags up through long_term_ref_pics_flag / ILRP / idr_rpl /
        // rpl1_same_as_rpl0.
        push_u(bits, 0, 1); // transform_skip
        push_u(bits, 0, 1); // mts
        push_u(bits, 0, 1); // lfnst
        push_u(bits, 0, 1); // joint_cbcr
        push_u(bits, 1, 1); // same_qp_table_for_chroma → 1 QP table
        push_se(bits, 0);
        push_ue(bits, 0);
        push_ue(bits, 0);
        push_ue(bits, 0);
        push_u(bits, 0, 1); // sao
        push_u(bits, 0, 1); // alf (=> no ccalf)
        push_u(bits, 0, 1); // lmcs
        push_u(bits, if weighted_pred { 1 } else { 0 }, 1); // weighted_pred
        push_u(bits, 0, 1); // weighted_bipred
        push_u(bits, if long_term { 1 } else { 0 }, 1); // long_term_ref_pics
        if inter_layer {
            // vps_id > 0 → transmit inter_layer_prediction_enabled_flag.
            push_u(bits, 1, 1);
        }
        push_u(bits, 0, 1); // idr_rpl_present
        push_u(bits, 1, 1); // rpl1_same_as_rpl0 = 1 (single loop over i=0)
    }

    fn emit_sps_tail_after_rpl(bits: &mut Vec<u8>) {
        // ref_wraparound through virtual_boundaries_enabled — everything
        // off except tools implicitly required by §7.3.2.4 ordering.
        push_u(bits, 0, 1); // ref_wraparound
        push_u(bits, 0, 1); // temporal_mvp (=> no sbtmvp)
        push_u(bits, 0, 1); // amvr
        push_u(bits, 0, 1); // bdof
        push_u(bits, 0, 1); // smvd
        push_u(bits, 0, 1); // dmvr
        push_u(bits, 0, 1); // mmvd
        push_ue(bits, 0); // six_minus_max_num_merge_cand
        push_u(bits, 0, 1); // sbt
        push_u(bits, 0, 1); // affine
        push_u(bits, 0, 1); // bcw
        push_u(bits, 0, 1); // ciip
        push_u(bits, 0, 1); // gpm
        push_ue(bits, 0); // log2_parallel_merge_level
        push_u(bits, 0, 1); // isp
        push_u(bits, 0, 1); // mrl
        push_u(bits, 0, 1); // mip
        push_u(bits, 0, 1); // cclm
        push_u(bits, 0, 1); // chroma_horizontal_collocated
        push_u(bits, 0, 1); // chroma_vertical_collocated
        push_u(bits, 0, 1); // palette
        push_u(bits, 0, 1); // ibc
        push_u(bits, 0, 1); // ladf
        push_u(bits, 0, 1); // explicit_scaling_list
        push_u(bits, 0, 1); // dep_quant
        push_u(bits, 0, 1); // sign_data_hiding
        push_u(bits, 0, 1); // virtual_boundaries_enabled
                            // Tail: no HRD (ptl_dpb_hrd=0), field_seq=0, vui=0, ext=0.
        push_u(bits, 0, 1); // sps_field_seq_flag
        push_u(bits, 0, 1); // sps_vui_parameters_present_flag
        push_u(bits, 0, 1); // sps_extension_flag
    }

    #[test]
    fn rpl_st_only_two_entries_parses() {
        // sps_long_term_ref_pics_flag = 0, no ILRP. One list of two STRPs
        // with deltas +1 and -2. Because weighted_pred=0 the AbsDeltaPocSt
        // formula is `abs_delta_poc_st + 1` for every i → we encode 0 / 1
        // to get magnitudes 1 / 2.
        let mut bits: Vec<u8> = Vec::new();
        emit_sps_preamble_for_rpl(&mut bits, false, false, false);
        // num_ref_pic_lists[0] = 1 (rpl1_same_as_rpl0=1 so only one loop).
        push_ue(&mut bits, 1);
        // ---- ref_pic_list_struct(0, 0) ----
        push_ue(&mut bits, 2); // num_ref_entries = 2
                               // long_term=0 → no ltrp_in_header_flag, no st_ref_pic_flag.
                               // Entry 0: abs_delta_poc_st = 0 → Abs = 0+1 = 1, sign=0 (+1)
        push_ue(&mut bits, 0);
        push_u(&mut bits, 0, 1); // strp_entry_sign_flag (+)
                                 // Entry 1: abs_delta_poc_st = 1 → Abs = 1+1 = 2, sign=1 (-2)
        push_ue(&mut bits, 1);
        push_u(&mut bits, 1, 1); // strp_entry_sign_flag (-)
        emit_sps_tail_after_rpl(&mut bits);

        let bytes = pack(&bits);
        let sps = parse_sps(&bytes).unwrap();
        assert_eq!(sps.tool_flags.num_ref_pic_lists, [1, 0]);
        let l0 = &sps.tool_flags.ref_pic_lists[0];
        assert_eq!(l0.len(), 1);
        assert_eq!(l0[0].entries.len(), 2);
        // When LT refs are disabled the inferred ltrp_in_header_flag is 0.
        assert!(!l0[0].ltrp_in_header_flag);
        match l0[0].entries[0] {
            RefPicListEntry::ShortTerm {
                abs_delta_poc_st,
                strp_entry_sign_flag,
                delta_poc_val_st,
            } => {
                assert_eq!(abs_delta_poc_st, 0);
                assert!(!strp_entry_sign_flag);
                assert_eq!(delta_poc_val_st, 1);
            }
            other => panic!("entry 0 must be STRP, got {other:?}"),
        }
        match l0[0].entries[1] {
            RefPicListEntry::ShortTerm {
                abs_delta_poc_st,
                strp_entry_sign_flag,
                delta_poc_val_st,
            } => {
                assert_eq!(abs_delta_poc_st, 1);
                assert!(strp_entry_sign_flag);
                assert_eq!(delta_poc_val_st, -2);
            }
            other => panic!("entry 1 must be STRP, got {other:?}"),
        }
        assert_eq!(l0[0].num_strp_entries(), 2);
        assert_eq!(l0[0].num_ltrp_entries(), 0);
        assert_eq!(l0[0].num_ilrp_entries(), 0);
    }

    #[test]
    fn rpl_lt_only_inline_poc_lsb_parses() {
        // sps_long_term_ref_pics_flag = 1, ltrp_in_header_flag = 0, two
        // LTRP entries with 8-bit POC LSBs (log2_max_poc_lsb_minus4=4).
        let mut bits: Vec<u8> = Vec::new();
        emit_sps_preamble_for_rpl(&mut bits, true, false, false);
        push_ue(&mut bits, 1); // num_ref_pic_lists[0] = 1
                               // ---- ref_pic_list_struct(0, 0) ----
                               // rplsIdx=0 < num_ref_pic_lists_listidx=1 and num_ref_entries>0 →
                               // ltrp_in_header_flag is transmitted.
        push_ue(&mut bits, 2); // num_ref_entries = 2
        push_u(&mut bits, 0, 1); // ltrp_in_header_flag = 0 → POC LSBs inlined.
                                 // Entry 0: st_ref_pic_flag = 0 → LT with inlined POC LSB.
        push_u(&mut bits, 0, 1);
        push_u(&mut bits, 0xA5, 8); // rpls_poc_lsb_lt[0] = 0xA5
                                    // Entry 1: st_ref_pic_flag = 0 → LT with inlined POC LSB.
        push_u(&mut bits, 0, 1);
        push_u(&mut bits, 0x42, 8); // rpls_poc_lsb_lt[1] = 0x42
        emit_sps_tail_after_rpl(&mut bits);

        let bytes = pack(&bits);
        let sps = parse_sps(&bytes).unwrap();
        let rpl = &sps.tool_flags.ref_pic_lists[0][0];
        assert!(!rpl.ltrp_in_header_flag);
        assert_eq!(rpl.entries.len(), 2);
        assert_eq!(rpl.num_ltrp_entries(), 2);
        assert_eq!(rpl.num_strp_entries(), 0);
        match rpl.entries[0] {
            RefPicListEntry::LongTerm { poc_lsb_lt } => {
                assert_eq!(poc_lsb_lt, Some(0xA5));
            }
            other => panic!("entry 0 must be LTRP, got {other:?}"),
        }
        match rpl.entries[1] {
            RefPicListEntry::LongTerm { poc_lsb_lt } => {
                assert_eq!(poc_lsb_lt, Some(0x42));
            }
            other => panic!("entry 1 must be LTRP, got {other:?}"),
        }
    }

    #[test]
    fn rpl_mixed_st_lt_ilrp_parses() {
        // long_term=1, ILRP enabled, ltrp_in_header=1 so LT entries defer
        // their POC LSB. Build an RPL with 3 entries: ST (+3), ILRP idx=2,
        // LT (deferred).
        let mut bits: Vec<u8> = Vec::new();
        emit_sps_preamble_for_rpl(&mut bits, true, true, false);
        push_ue(&mut bits, 1); // num_ref_pic_lists[0] = 1
                               // ---- ref_pic_list_struct(0, 0) ----
        push_ue(&mut bits, 3); // num_ref_entries = 3
        push_u(&mut bits, 1, 1); // ltrp_in_header_flag = 1 → no inlined POC LSB
                                 // Entry 0: ILRP flag=0 → not ILRP; st_ref_pic_flag=1 → STRP; Abs=2+1=3.
        push_u(&mut bits, 0, 1); // inter_layer_ref_pic_flag
        push_u(&mut bits, 1, 1); // st_ref_pic_flag
        push_ue(&mut bits, 2); // abs_delta_poc_st → Abs = 3
        push_u(&mut bits, 0, 1); // sign = 0 → +3
                                 // Entry 1: ILRP flag=1 → ilrp_idx = 2
        push_u(&mut bits, 1, 1);
        push_ue(&mut bits, 2);
        // Entry 2: ILRP flag=0, st_ref_pic_flag=0 → LT, deferred POC LSB.
        push_u(&mut bits, 0, 1);
        push_u(&mut bits, 0, 1);
        emit_sps_tail_after_rpl(&mut bits);

        let bytes = pack(&bits);
        let sps = parse_sps(&bytes).unwrap();
        let t = &sps.tool_flags;
        assert!(t.long_term_ref_pics_flag);
        assert!(t.inter_layer_prediction_enabled_flag);
        let rpl = &t.ref_pic_lists[0][0];
        assert!(rpl.ltrp_in_header_flag);
        assert_eq!(rpl.entries.len(), 3);
        assert_eq!(rpl.num_strp_entries(), 1);
        assert_eq!(rpl.num_ltrp_entries(), 1);
        assert_eq!(rpl.num_ilrp_entries(), 1);
        match rpl.entries[0] {
            RefPicListEntry::ShortTerm {
                delta_poc_val_st, ..
            } => assert_eq!(delta_poc_val_st, 3),
            other => panic!("entry 0 must be STRP, got {other:?}"),
        }
        match rpl.entries[1] {
            RefPicListEntry::InterLayer { ilrp_idx } => assert_eq!(ilrp_idx, 2),
            other => panic!("entry 1 must be ILRP, got {other:?}"),
        }
        match rpl.entries[2] {
            RefPicListEntry::LongTerm { poc_lsb_lt } => assert_eq!(poc_lsb_lt, None),
            other => panic!("entry 2 must be LTRP, got {other:?}"),
        }
    }

    #[test]
    fn ladf_block_parses_when_enabled() {
        // Rebuild a full SPS; toggle ladf_enabled = 1 and supply
        // num_intervals_minus2 = 0 (→ 1 interval), lowest = -3, one
        // (qp_offset = 2, delta_threshold_minus1 = 0) entry.
        let mut bits: Vec<u8> = Vec::new();
        push_u(&mut bits, 0, 4);
        push_u(&mut bits, 0, 4);
        push_u(&mut bits, 0, 3);
        push_u(&mut bits, 1, 2);
        push_u(&mut bits, 2, 2);
        push_u(&mut bits, 0, 1);
        push_u(&mut bits, 0, 1);
        push_u(&mut bits, 0, 1);
        push_ue(&mut bits, 320);
        push_ue(&mut bits, 240);
        push_u(&mut bits, 0, 1);
        push_u(&mut bits, 0, 1);
        push_ue(&mut bits, 2);
        push_u(&mut bits, 0, 1);
        push_u(&mut bits, 0, 1);
        push_u(&mut bits, 4, 4);
        push_u(&mut bits, 0, 1);
        push_u(&mut bits, 0, 2);
        push_u(&mut bits, 0, 2);
        // Partition constraints.
        push_ue(&mut bits, 0);
        push_u(&mut bits, 0, 1);
        push_ue(&mut bits, 0);
        push_ue(&mut bits, 0);
        push_u(&mut bits, 0, 1);
        push_ue(&mut bits, 0);
        push_ue(&mut bits, 0);
        push_u(&mut bits, 0, 1);
        // Tool flags.
        push_u(&mut bits, 0, 1); // ts
        push_u(&mut bits, 0, 1); // mts
        push_u(&mut bits, 0, 1); // lfnst
        push_u(&mut bits, 0, 1); // joint_cbcr
        push_u(&mut bits, 1, 1); // same_qp
        push_se(&mut bits, 0);
        push_ue(&mut bits, 0);
        push_ue(&mut bits, 0);
        push_ue(&mut bits, 0);
        push_u(&mut bits, 0, 1); // sao
        push_u(&mut bits, 0, 1); // alf
        push_u(&mut bits, 0, 1); // lmcs
        push_u(&mut bits, 0, 1);
        push_u(&mut bits, 0, 1);
        push_u(&mut bits, 0, 1);
        push_u(&mut bits, 0, 1); // idr_rpl
        push_u(&mut bits, 0, 1); // rpl1_same_as_rpl0
        push_ue(&mut bits, 0);
        push_ue(&mut bits, 0);
        push_u(&mut bits, 0, 1);
        push_u(&mut bits, 0, 1);
        push_u(&mut bits, 0, 1);
        push_u(&mut bits, 0, 1); // bdof
        push_u(&mut bits, 0, 1); // smvd
        push_u(&mut bits, 0, 1); // dmvr
        push_u(&mut bits, 0, 1); // mmvd
        push_ue(&mut bits, 0);
        push_u(&mut bits, 0, 1);
        push_u(&mut bits, 0, 1); // affine
        push_u(&mut bits, 0, 1); // bcw
        push_u(&mut bits, 0, 1); // ciip
        push_u(&mut bits, 0, 1); // gpm
        push_ue(&mut bits, 0);
        push_u(&mut bits, 0, 1);
        push_u(&mut bits, 0, 1);
        push_u(&mut bits, 0, 1);
        push_u(&mut bits, 0, 1); // cclm
        push_u(&mut bits, 0, 1);
        push_u(&mut bits, 0, 1);
        push_u(&mut bits, 0, 1); // palette
        push_u(&mut bits, 0, 1); // ibc
        push_u(&mut bits, 1, 1); // ladf_enabled = 1
        push_u(&mut bits, 0, 2); // num_intervals_minus2 = 0
        push_se(&mut bits, -3); // lowest_interval_qp_offset
        push_se(&mut bits, 2); // ladf_qp_offset[0]
        push_ue(&mut bits, 0); // ladf_delta_threshold_minus1[0]
        push_u(&mut bits, 0, 1); // explicit_scaling_list
        push_u(&mut bits, 0, 1); // dep_quant
        push_u(&mut bits, 0, 1); // sign_data_hiding
        push_u(&mut bits, 0, 1); // virtual_boundaries_enabled
                                 // Tail: field_seq=0, vui=0, ext=0.
        push_u(&mut bits, 0, 1); // sps_field_seq_flag
        push_u(&mut bits, 0, 1); // sps_vui_parameters_present_flag
        push_u(&mut bits, 0, 1); // sps_extension_flag

        let bytes = pack(&bits);
        let sps = parse_sps(&bytes).unwrap();
        let t = &sps.tool_flags;
        assert!(t.ladf_enabled_flag);
        let ladf = t.ladf.as_ref().expect("ladf parsed");
        assert_eq!(ladf.num_intervals_minus2, 0);
        assert_eq!(ladf.lowest_interval_qp_offset, -3);
        assert_eq!(ladf.intervals.len(), 1);
        assert_eq!(ladf.intervals[0], (2, 0));
    }

    // -----------------------------------------------------------------
    // §7.4.3.4 eq. 85 — MaxNumSubblockMergeCand derivation
    // -----------------------------------------------------------------

    /// Build a `SeqParameterSet` with only the three fields the eq.-85
    /// derivation reads pre-populated. Built off the canonical minimal SPS
    /// so every other field is realistic; the three eq.-85 inputs are then
    /// overwritten directly.
    fn sps_with_subblock_inputs(
        affine: bool,
        sbtmvp: bool,
        five_minus_cand: u32,
    ) -> SeqParameterSet {
        let bits = build_minimal_sps_bits();
        let bytes = pack(&bits);
        let mut s = parse_sps(&bytes).expect("minimal SPS parses");
        s.tool_flags.affine_enabled_flag = affine;
        s.tool_flags.sbtmvp_enabled_flag = sbtmvp;
        s.tool_flags.five_minus_max_num_subblock_merge_cand = five_minus_cand;
        s
    }

    #[test]
    fn max_num_subblock_merge_cand_affine_branch_full_range() {
        // §7.4.3.4 eq. 85 affine branch: 5 − five_minus_max_num_subblock_merge_cand.
        // PH ph_temporal_mvp_enabled_flag is ignored in this branch.
        for k in 0u32..=5 {
            let sps = sps_with_subblock_inputs(true, false, k);
            assert_eq!(
                sps.max_num_subblock_merge_cand(false),
                5 - k,
                "five_minus={k}, ph=false"
            );
            assert_eq!(
                sps.max_num_subblock_merge_cand(true),
                5 - k,
                "five_minus={k}, ph=true (ignored in affine branch)"
            );
        }
    }

    #[test]
    fn max_num_subblock_merge_cand_affine_branch_clamps_negative() {
        // sps_five_minus_max_num_subblock_merge_cand > 5 would yield negative;
        // the §7.4.3.4 range constraint forces the result to 0.
        let sps = sps_with_subblock_inputs(true, false, 6);
        assert_eq!(sps.max_num_subblock_merge_cand(false), 0);
        let sps = sps_with_subblock_inputs(true, true, 100);
        assert_eq!(sps.max_num_subblock_merge_cand(true), 0);
    }

    #[test]
    fn max_num_subblock_merge_cand_non_affine_branch_sbtmvp_and_ph() {
        // §7.4.3.4 eq. 85 else branch: sbtmvp && ph_temporal_mvp_enabled.
        // Both on → 1; either off → 0. five_minus value is ignored.
        let sps = sps_with_subblock_inputs(false, true, 3);
        assert_eq!(sps.max_num_subblock_merge_cand(true), 1);
        assert_eq!(sps.max_num_subblock_merge_cand(false), 0);

        let sps = sps_with_subblock_inputs(false, false, 0);
        assert_eq!(sps.max_num_subblock_merge_cand(true), 0);
        assert_eq!(sps.max_num_subblock_merge_cand(false), 0);

        // sbtmvp=on but ph=off (the SPS-only lower-bound case) → 0.
        let sps = sps_with_subblock_inputs(false, true, 0);
        assert_eq!(sps.max_num_subblock_merge_cand(false), 0);
    }

    #[test]
    fn max_num_subblock_merge_cand_result_stays_within_zero_to_five() {
        // §7.4.3.4 trailing bullet: "shall be in the range of 0 to 5, inclusive".
        for affine in [false, true] {
            for sbtmvp in [false, true] {
                for five_minus in 0u32..=8 {
                    for ph in [false, true] {
                        let sps = sps_with_subblock_inputs(affine, sbtmvp, five_minus);
                        let v = sps.max_num_subblock_merge_cand(ph);
                        assert!(
                            v <= 5,
                            "out of range: affine={affine} sbtmvp={sbtmvp} \
                             five_minus={five_minus} ph={ph} → {v}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn max_num_subblock_merge_cand_drives_merge_subblock_idx_cmax() {
        // §7.3.11.7 / Table 110 link: `merge_subblock_idx` has cMax =
        // MaxNumSubblockMergeCand − 1, and the §7.3.11.7 size gate +
        // §7.4.12.7 inference fold to 0 when MaxNumSubblockMergeCand ≤ 1.
        // This test pins the derivation values the reader switches on.
        let sps = sps_with_subblock_inputs(true, false, 0);
        assert_eq!(sps.max_num_subblock_merge_cand(false), 5); // cMax=4
        let sps = sps_with_subblock_inputs(true, false, 3);
        assert_eq!(sps.max_num_subblock_merge_cand(false), 2); // cMax=1
        let sps = sps_with_subblock_inputs(true, false, 4);
        assert_eq!(sps.max_num_subblock_merge_cand(false), 1); // cMax suppressed → infer 0
        let sps = sps_with_subblock_inputs(true, false, 5);
        assert_eq!(sps.max_num_subblock_merge_cand(false), 0); // cMax suppressed → infer 0
    }

    #[test]
    fn max_num_merge_cand_full_range() {
        // §7.4.3.4: MaxNumMergeCand = 6 − sps_six_minus_max_num_merge_cand,
        // legal range 1..=6. Pins the regular-merge cand derivation that
        // gates the SPS gpm_enabled_flag emission (≥ 2 ≥ 3 thresholds in
        // parse_tool_flags).
        let bits = build_minimal_sps_bits();
        let bytes = pack(&bits);
        let mut s = parse_sps(&bytes).expect("minimal SPS parses");
        for k in 0u32..=5 {
            s.tool_flags.six_minus_max_num_merge_cand = k;
            assert_eq!(s.max_num_merge_cand(), 6 - k);
        }
        // Clamp: out-of-range values still produce a legal 1..=6 result.
        s.tool_flags.six_minus_max_num_merge_cand = 10;
        assert_eq!(s.max_num_merge_cand(), 1);
    }

    /// Configurable knobs for [`build_range_extension_sps_bits`].
    ///
    /// `bitdepth_minus8` controls the §7.4.3.4 `BitDepth > 10`
    /// constraint check (set `>= 3` for legal range-extension cases).
    /// `transform_skip_enabled` switches the §7.3.2.22 conditional
    /// `sps_ts_residual_coding_rice_present_in_sh_flag` bin in/out.
    #[derive(Clone, Copy)]
    struct RangeExtCfg {
        bitdepth_minus8: u32,
        transform_skip_enabled: bool,
        ext_precision: bool,
        ts_rrc_in_sh: bool,
        rrc_rice_ext: bool,
        persistent_rice: bool,
        reverse_last_sig: bool,
    }

    /// Build a minimal SPS whose `sps_range_extension_flag = 1` and
    /// whose §7.3.2.22 payload bins are driven by `cfg`. Used to pin
    /// the five-flag range-extension parser at both the all-zero and
    /// all-one extremes, plus the transform-skip-gated conditional
    /// bin.
    fn build_range_extension_sps_bits(cfg: RangeExtCfg) -> Vec<u8> {
        let mut bits: Vec<u8> = Vec::new();
        push_u(&mut bits, 0, 4); // sps_id
        push_u(&mut bits, 0, 4); // vps_id
        push_u(&mut bits, 0, 3); // max_sublayers_minus1
        push_u(&mut bits, 1, 2); // chroma = 4:2:0
        push_u(&mut bits, 2, 2); // log2_ctu - 5 = 2 → CTB=128
        push_u(&mut bits, 0, 1); // ptl_dpb_hrd_present = 0
        push_u(&mut bits, 0, 1); // gdr_enabled
        push_u(&mut bits, 0, 1); // ref_pic_resampling
        push_ue(&mut bits, 320);
        push_ue(&mut bits, 240);
        push_u(&mut bits, 0, 1); // conformance_window_flag
        push_u(&mut bits, 0, 1); // subpic_info_present
        push_ue(&mut bits, cfg.bitdepth_minus8); // bitdepth_minus8
        push_u(&mut bits, 0, 1); // entropy_coding_sync
        push_u(&mut bits, 0, 1); // entry_point_offsets
        push_u(&mut bits, 4, 4); // log2_max_poc_lsb_minus4
        push_u(&mut bits, 0, 1); // poc_msb_cycle_flag
        push_u(&mut bits, 0, 2); // num_extra_ph_bytes
        push_u(&mut bits, 0, 2); // num_extra_sh_bytes

        // ---- partition constraints (§7.3.2.4 tail) ----
        push_ue(&mut bits, 0);
        push_u(&mut bits, 0, 1); // partition_constraints_override_enabled
        push_ue(&mut bits, 0); // log2_diff_min_qt_min_cb_intra_luma
        push_ue(&mut bits, 0); // max_mtt_depth_intra_luma
        push_u(&mut bits, 0, 1); // qtbtt_dual_tree_intra_flag
        push_ue(&mut bits, 0); // log2_diff_min_qt_min_cb_inter
        push_ue(&mut bits, 0); // max_mtt_depth_inter
        push_u(&mut bits, 0, 1); // max_luma_transform_size_64_flag

        // ---- tool flags (§7.3.2.4 tail) ----
        push_u(&mut bits, cfg.transform_skip_enabled as u64, 1); // transform_skip_enabled
        if cfg.transform_skip_enabled {
            push_ue(&mut bits, 0); // log2_transform_skip_max_size_minus2
            push_u(&mut bits, 0, 1); // bdpcm_enabled_flag
        }
        push_u(&mut bits, 0, 1); // mts_enabled
        push_u(&mut bits, 0, 1); // lfnst_enabled
        push_u(&mut bits, 0, 1); // joint_cbcr_enabled
        push_u(&mut bits, 1, 1); // same_qp_table_for_chroma
        push_se(&mut bits, 0); // qp_table_start_minus26
        push_ue(&mut bits, 0); // num_points_minus1 = 0
        push_ue(&mut bits, 0); // delta_qp_in_val_minus1[0][0]
        push_ue(&mut bits, 0); // delta_qp_diff_val[0][0]
        push_u(&mut bits, 0, 1); // sao_enabled
        push_u(&mut bits, 0, 1); // alf_enabled
        push_u(&mut bits, 0, 1); // lmcs_enabled
        push_u(&mut bits, 0, 1); // weighted_pred
        push_u(&mut bits, 0, 1); // weighted_bipred
        push_u(&mut bits, 0, 1); // long_term_ref_pics
        push_u(&mut bits, 0, 1); // idr_rpl_present
        push_u(&mut bits, 0, 1); // rpl1_same_as_rpl0 = 0
        push_ue(&mut bits, 0); // num_ref_pic_lists[0]
        push_ue(&mut bits, 0); // num_ref_pic_lists[1]
        push_u(&mut bits, 0, 1); // ref_wraparound
        push_u(&mut bits, 0, 1); // temporal_mvp
        push_u(&mut bits, 0, 1); // amvr
        push_u(&mut bits, 0, 1); // bdof
        push_u(&mut bits, 0, 1); // smvd
        push_u(&mut bits, 0, 1); // dmvr
        push_u(&mut bits, 0, 1); // mmvd
        push_ue(&mut bits, 0); // six_minus_max_num_merge_cand
        push_u(&mut bits, 0, 1); // sbt
        push_u(&mut bits, 0, 1); // affine
        push_u(&mut bits, 0, 1); // bcw
        push_u(&mut bits, 0, 1); // ciip
        push_u(&mut bits, 0, 1); // gpm_enabled
        push_ue(&mut bits, 0); // log2_parallel_merge_level_minus2
        push_u(&mut bits, 0, 1); // isp
        push_u(&mut bits, 0, 1); // mrl
        push_u(&mut bits, 0, 1); // mip
        push_u(&mut bits, 0, 1); // cclm
        push_u(&mut bits, 0, 1); // chroma_horizontal_collocated
        push_u(&mut bits, 0, 1); // chroma_vertical_collocated
        push_u(&mut bits, 0, 1); // palette
                                 // transform_skip || palette → min_qp_prime_ts
        if cfg.transform_skip_enabled {
            push_ue(&mut bits, 0); // min_qp_prime_ts
        }
        push_u(&mut bits, 0, 1); // ibc
        push_u(&mut bits, 0, 1); // ladf
        push_u(&mut bits, 0, 1); // explicit_scaling_list
        push_u(&mut bits, 0, 1); // dep_quant
        push_u(&mut bits, 0, 1); // sign_data_hiding
        push_u(&mut bits, 0, 1); // virtual_boundaries_enabled
        push_u(&mut bits, 0, 1); // sps_field_seq_flag
        push_u(&mut bits, 0, 1); // sps_vui_parameters_present_flag

        // ---- extension block (§7.3.2.4 tail) ----
        push_u(&mut bits, 1, 1); // sps_extension_flag = 1
        push_u(&mut bits, 1, 1); // sps_range_extension_flag = 1
        push_u(&mut bits, 0, 7); // sps_extension_7bits = 0

        // ---- sps_range_extension() body (§7.3.2.22) ----
        push_u(&mut bits, cfg.ext_precision as u64, 1);
        if cfg.transform_skip_enabled {
            push_u(&mut bits, cfg.ts_rrc_in_sh as u64, 1);
        }
        push_u(&mut bits, cfg.rrc_rice_ext as u64, 1);
        push_u(&mut bits, cfg.persistent_rice as u64, 1);
        push_u(&mut bits, cfg.reverse_last_sig as u64, 1);
        bits
    }

    /// `sps_range_extension()` payload reads back as all-zeros when
    /// every bin is emitted as `0`. Also pins the §7.4.3.22 eq. 106
    /// `Log2TransformRange = 15` (non-extended-precision) branch.
    #[test]
    fn sps_range_extension_all_zero_round_trip() {
        let cfg = RangeExtCfg {
            bitdepth_minus8: 4, // BitDepth = 12 (> 10 ⇒ range_ext legal)
            transform_skip_enabled: false,
            ext_precision: false,
            ts_rrc_in_sh: false,
            rrc_rice_ext: false,
            persistent_rice: false,
            reverse_last_sig: false,
        };
        let bytes = pack(&build_range_extension_sps_bits(cfg));
        let sps = parse_sps(&bytes).expect("range-ext SPS parses");

        assert!(sps.sps_extension_flag);
        assert!(sps.sps_range_extension_flag);
        assert_eq!(sps.sps_extension_7bits, 0);

        let re = sps.range_extension.expect("range extension populated");
        assert!(!re.sps_extended_precision_flag);
        assert!(!re.sps_ts_residual_coding_rice_present_in_sh_flag);
        assert!(!re.sps_rrc_rice_extension_flag);
        assert!(!re.sps_persistent_rice_adaptation_enabled_flag);
        assert!(!re.sps_reverse_last_sig_coeff_enabled_flag);

        // §7.4.3.22 eq. 106: Log2TransformRange = 15 when ext_prec = 0.
        assert_eq!(re.log2_transform_range(sps.bit_depth_y()), 15);
    }

    /// `sps_range_extension()` payload reads back as all-ones when
    /// every bin is emitted as `1`, including the `transform_skip`-
    /// gated `sps_ts_residual_coding_rice_present_in_sh_flag` bin.
    /// Also pins the §7.4.3.22 eq. 106 extended-precision branch for
    /// `BitDepth = 12 → Log2TransformRange = Max(15, Min(20, 18)) = 18`.
    #[test]
    fn sps_range_extension_all_one_with_ts_round_trip() {
        let cfg = RangeExtCfg {
            bitdepth_minus8: 4, // BitDepth = 12
            transform_skip_enabled: true,
            ext_precision: true,
            ts_rrc_in_sh: true,
            rrc_rice_ext: true,
            persistent_rice: true,
            reverse_last_sig: true,
        };
        let bytes = pack(&build_range_extension_sps_bits(cfg));
        let sps = parse_sps(&bytes).expect("range-ext SPS parses");

        let re = sps.range_extension.expect("range extension populated");
        assert!(re.sps_extended_precision_flag);
        assert!(re.sps_ts_residual_coding_rice_present_in_sh_flag);
        assert!(re.sps_rrc_rice_extension_flag);
        assert!(re.sps_persistent_rice_adaptation_enabled_flag);
        assert!(re.sps_reverse_last_sig_coeff_enabled_flag);

        // BitDepth = 12 ⇒ Max(15, Min(20, 18)) = 18.
        assert_eq!(re.log2_transform_range(sps.bit_depth_y()), 18);
        // BitDepth = 16 ⇒ Max(15, Min(20, 22)) = 20 (clamped).
        assert_eq!(re.log2_transform_range(16), 20);
        // BitDepth = 8 ⇒ Max(15, Min(20, 14)) = 15 (clamped).
        assert_eq!(re.log2_transform_range(8), 15);
    }

    /// When `sps_transform_skip_enabled_flag == 0`, the
    /// `sps_ts_residual_coding_rice_present_in_sh_flag` bin is NOT
    /// transmitted in the §7.3.2.22 body and §7.4.3.22 infers it to
    /// 0. The bit-position math is sensitive to this — getting the
    /// gate wrong corrupts the next-emitted bin
    /// (`sps_rrc_rice_extension_flag`). Pins both: TS-disabled SPS
    /// with `sps_rrc_rice_extension_flag = 1` and the inferred 0.
    #[test]
    fn sps_range_extension_ts_disabled_skips_rrc_bin() {
        let cfg = RangeExtCfg {
            bitdepth_minus8: 4,
            transform_skip_enabled: false, // ⇒ skip ts_rrc bin
            ext_precision: false,
            ts_rrc_in_sh: true, // value ignored — bin not emitted
            rrc_rice_ext: true,
            persistent_rice: false,
            reverse_last_sig: true,
        };
        let bytes = pack(&build_range_extension_sps_bits(cfg));
        let sps = parse_sps(&bytes).expect("range-ext SPS parses");

        let re = sps.range_extension.expect("range extension populated");
        assert!(!re.sps_extended_precision_flag);
        // Bin was NOT emitted ⇒ §7.4.3.22 inference = 0.
        assert!(!re.sps_ts_residual_coding_rice_present_in_sh_flag);
        // The following three bins must align with the cfg values —
        // a misaligned reader would put `rrc_rice = false` here.
        assert!(re.sps_rrc_rice_extension_flag);
        assert!(!re.sps_persistent_rice_adaptation_enabled_flag);
        assert!(re.sps_reverse_last_sig_coeff_enabled_flag);
    }

    /// §7.4.3.4 constraint: `sps_range_extension_flag` shall be 0
    /// when `BitDepth <= 10`. The parser rejects a malformed
    /// bitstream that flips it at 10-bit.
    #[test]
    fn sps_range_extension_flag_at_10bit_rejected() {
        let cfg = RangeExtCfg {
            bitdepth_minus8: 2, // BitDepth = 10
            transform_skip_enabled: false,
            ext_precision: false,
            ts_rrc_in_sh: false,
            rrc_rice_ext: false,
            persistent_rice: false,
            reverse_last_sig: false,
        };
        let bytes = pack(&build_range_extension_sps_bits(cfg));
        let err = parse_sps(&bytes).expect_err("BitDepth = 10 disallows range-ext flag");
        let msg = format!("{err}");
        assert!(
            msg.contains("sps_range_extension_flag") && msg.contains("BitDepth"),
            "unexpected error message: {msg}"
        );
    }
}
