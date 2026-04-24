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
//! * `ref_pic_list_struct()` — we refuse SPSes that signal any
//!   `sps_num_ref_pic_lists[i]` > 0 (the structure needs its own parser),
//! * HRD timing, VUI payload, and `sps_extension_*`. The tail reader
//!   stops after the virtual-boundaries block.

use oxideav_core::{Error, Result};

use crate::bitreader::BitReader;
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
    pub sps_sublayer_dpb_params_flag: bool,
    pub dpb_parameters: Option<DpbParameters>,
    pub partition_constraints: PartitionConstraints,
    pub tool_flags: ToolFlags,
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
    if sps_subpic_info_present_flag {
        // The subpic sub-syntax contains u(v) fields whose width is
        // derived from `CtbSizeY` and the picture dimensions. Decoding
        // those correctly is out of foundation scope; we surface an
        // Unsupported error so callers know the SPS parse was aborted.
        return Err(Error::unsupported(
            "h266 SPS: sps_subpic_info_present_flag = 1 (subpicture streams not yet supported)",
        ));
    }
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
    for _ in 0..(sps_num_extra_ph_bytes as u32 * 8) {
        br.skip(1)?;
    }
    let sps_num_extra_sh_bytes = br.u(2)? as u8;
    for _ in 0..(sps_num_extra_sh_bytes as u32 * 8) {
        br.skip(1)?;
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
    )?;

    // The rest of the SPS (HRD timing, VUI payload, sps_extension_*) is
    // intentionally not parsed by this scaffold. Callers that need it
    // will extend the parser in a follow-up increment.
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
        sps_sublayer_dpb_params_flag,
        dpb_parameters,
        partition_constraints,
        tool_flags,
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
        t.num_ref_pic_lists[i] = n;
        if n > 0 {
            // `ref_pic_list_struct(i, j)` is a non-trivial syntax
            // (LTRPs, inter-layer flags, delta POCs); parsing it is out
            // of scope for this increment. Surface Unsupported so the
            // caller does not silently consume misaligned bits.
            return Err(Error::unsupported(
                "h266 SPS: sps_num_ref_pic_lists > 0 — ref_pic_list_struct() not yet parsed",
            ));
        }
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
        bits
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

    #[test]
    fn subpic_streams_are_rejected() {
        // Same preamble as above but flip subpic_info_present to 1; the
        // parser should return Unsupported before reading anything else.
        let mut bits: Vec<u8> = Vec::new();
        push_u(&mut bits, 0, 4); // sps_id
        push_u(&mut bits, 0, 4); // vps_id
        push_u(&mut bits, 0, 3); // max_sublayers
        push_u(&mut bits, 1, 2); // chroma
        push_u(&mut bits, 0, 2); // log2_ctu - 5 = 0 → 32
        push_u(&mut bits, 0, 1); // ptl_present
        push_u(&mut bits, 0, 1); // gdr
        push_u(&mut bits, 0, 1); // ref_pic_resampling
        push_ue(&mut bits, 320); // width
        push_ue(&mut bits, 240); // height
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

        let bytes = pack(&bits);
        let sps = parse_sps(&bytes).unwrap();
        let t = &sps.tool_flags;
        assert!(t.sao_enabled_flag);
        assert!(t.alf_enabled_flag);
        assert!(t.ccalf_enabled_flag);
        assert!(t.lmcs_enabled_flag);
    }

    #[test]
    fn ref_pic_list_struct_streams_are_unsupported() {
        // Minimal SPS where num_ref_pic_lists[0] = 1; parsing should
        // surface Unsupported because ref_pic_list_struct() isn't
        // implemented yet.
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
        // Tool flags up to num_ref_pic_lists[0] = 1.
        push_u(&mut bits, 0, 1);
        push_u(&mut bits, 0, 1);
        push_u(&mut bits, 0, 1);
        push_u(&mut bits, 0, 1);
        push_u(&mut bits, 1, 1);
        push_se(&mut bits, 0);
        push_ue(&mut bits, 0);
        push_ue(&mut bits, 0);
        push_ue(&mut bits, 0);
        push_u(&mut bits, 0, 1);
        push_u(&mut bits, 0, 1);
        push_u(&mut bits, 0, 1);
        push_u(&mut bits, 0, 1);
        push_u(&mut bits, 0, 1);
        push_u(&mut bits, 0, 1);
        push_u(&mut bits, 0, 1);
        push_u(&mut bits, 0, 1); // rpl1_same_as_rpl0 = 0
        push_ue(&mut bits, 1); // num_ref_pic_lists[0] = 1 → Unsupported
        let bytes = pack(&bits);
        let err = parse_sps(&bytes).unwrap_err();
        assert!(matches!(err, Error::Unsupported(_)));
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
}
