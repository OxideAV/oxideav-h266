//! VVC Picture Header structure parser (§7.3.2.7 — `picture_header_rbsp()`
//! / §7.3.2.8 — `picture_header_structure()`).
//!
//! Two entry points are offered:
//!
//! * [`parse_picture_header`] — byte-level parse through
//!   `ph_pic_parameter_set_id` only, preserved for callers that still
//!   want the opaque-tail view from rounds 1-4.
//! * [`parse_picture_header_stateful`] — full §7.3.2.8 walk, requires
//!   the SPS + PPS the PH references. Produces a populated
//!   [`PictureHeader`] struct; the `tail_bits_*` fields then cover only
//!   the bits not consumed by the PH body (e.g. the
//!   `rbsp_trailing_bits()` pad on a standalone PH NAL, or the slice
//!   header payload when the PH is embedded).

use oxideav_core::{Error, Result};

use crate::bitreader::BitReader;
use crate::pps::PicParameterSet;
use crate::ref_pic_list::{parse_ref_pic_lists, HeaderRefPicList};
use crate::sps::SeqParameterSet;

/// Virtual-boundary positions supplied by the picture header (§7.3.2.8).
#[derive(Clone, Debug, Default)]
pub struct PhVirtualBoundaries {
    pub num_ver: u32,
    pub pos_x_minus1: Vec<u32>,
    pub num_hor: u32,
    pub pos_y_minus1: Vec<u32>,
}

/// Partition-constraints override fields (§7.3.2.8). All values are
/// optional because they are transmitted piecewise under a chain of
/// `if` gates.
#[derive(Clone, Debug, Default)]
pub struct PhPartitionOverride {
    /// Intra-slice luma constraints (present when
    /// `ph_intra_slice_allowed_flag && ph_partition_constraints_override_flag`).
    pub log2_diff_min_qt_min_cb_intra_slice_luma: u32,
    pub max_mtt_hierarchy_depth_intra_slice_luma: u32,
    pub log2_diff_max_bt_min_qt_intra_slice_luma: u32,
    pub log2_diff_max_tt_min_qt_intra_slice_luma: u32,
    /// Chroma constraints, only when `sps_qtbtt_dual_tree_intra_flag` is set.
    pub log2_diff_min_qt_min_cb_intra_slice_chroma: u32,
    pub max_mtt_hierarchy_depth_intra_slice_chroma: u32,
    pub log2_diff_max_bt_min_qt_intra_slice_chroma: u32,
    pub log2_diff_max_tt_min_qt_intra_slice_chroma: u32,
    /// Inter-slice constraints.
    pub log2_diff_min_qt_min_cb_inter_slice: u32,
    pub max_mtt_hierarchy_depth_inter_slice: u32,
    pub log2_diff_max_bt_min_qt_inter_slice: u32,
    pub log2_diff_max_tt_min_qt_inter_slice: u32,
}

/// Deblocking parameters transmitted in the PH (§7.3.2.8).
#[derive(Clone, Debug, Default)]
pub struct PhDeblockingParams {
    pub present_flag: bool,
    pub filter_disabled_flag: bool,
    pub luma_beta_offset_div2: i32,
    pub luma_tc_offset_div2: i32,
    pub cb_beta_offset_div2: i32,
    pub cb_tc_offset_div2: i32,
    pub cr_beta_offset_div2: i32,
    pub cr_tc_offset_div2: i32,
}

/// Full picture-header structure (§7.3.2.8) after a stateful parse.
///
/// Fields that the spec infers when absent are populated with their
/// inferred values (documented per-field). Lists that are skipped by
/// their gates stay empty.
#[derive(Clone, Debug, Default)]
pub struct PictureHeader {
    pub ph_gdr_or_irap_pic_flag: bool,
    pub ph_non_ref_pic_flag: bool,
    pub ph_gdr_pic_flag: bool,
    pub ph_inter_slice_allowed_flag: bool,
    /// `true` when intra slices are allowed (inferred to `true` when
    /// `ph_inter_slice_allowed_flag == 0`, per §7.3.2.8).
    pub ph_intra_slice_allowed_flag: bool,
    pub ph_pic_parameter_set_id: u32,
    pub ph_pic_order_cnt_lsb: u32,
    /// Only present when `ph_gdr_pic_flag == 1`; `0` otherwise.
    pub ph_recovery_poc_cnt: u32,
    /// Raw `ph_extra_bit[i]` values in transmission order. Length equals
    /// `NumExtraPhBits` (§7.4.3.4). Bit values are preserved but VVC
    /// reserves them — consumers typically ignore.
    pub ph_extra_bits: Vec<u8>,
    pub ph_poc_msb_cycle_present_flag: bool,
    pub ph_poc_msb_cycle_val: u32,

    // ALF (§7.3.2.8).
    pub ph_alf_enabled_flag: bool,
    pub ph_num_alf_aps_ids_luma: u8,
    pub ph_alf_aps_id_luma: Vec<u8>,
    pub ph_alf_cb_enabled_flag: bool,
    pub ph_alf_cr_enabled_flag: bool,
    pub ph_alf_aps_id_chroma: u8,
    pub ph_alf_cc_cb_enabled_flag: bool,
    pub ph_alf_cc_cb_aps_id: u8,
    pub ph_alf_cc_cr_enabled_flag: bool,
    pub ph_alf_cc_cr_aps_id: u8,

    // LMCS.
    pub ph_lmcs_enabled_flag: bool,
    pub ph_lmcs_aps_id: u8,
    pub ph_chroma_residual_scale_flag: bool,

    // Explicit scaling list.
    pub ph_explicit_scaling_list_enabled_flag: bool,
    pub ph_scaling_list_aps_id: u8,

    // Virtual boundaries.
    pub ph_virtual_boundaries_present_flag: bool,
    pub ph_virtual_boundaries: Option<PhVirtualBoundaries>,

    // Picture-output flag (§7.3.2.8).
    pub ph_pic_output_flag: bool,

    /// `ref_pic_lists()` block (§7.3.9) — populated iff
    /// `pps_rpl_info_in_ph_flag == 1`.
    pub ref_pic_lists: Option<[HeaderRefPicList; 2]>,

    pub ph_partition_constraints_override_flag: bool,
    pub partition_override: Option<PhPartitionOverride>,

    pub ph_cu_qp_delta_subdiv_intra_slice: u32,
    pub ph_cu_chroma_qp_offset_subdiv_intra_slice: u32,
    pub ph_cu_qp_delta_subdiv_inter_slice: u32,
    pub ph_cu_chroma_qp_offset_subdiv_inter_slice: u32,

    // Temporal MVP + collocated reference (only under ph_inter_slice_allowed).
    pub ph_temporal_mvp_enabled_flag: bool,
    pub ph_collocated_from_l0_flag: bool,
    pub ph_collocated_ref_idx: u32,

    // MMVD / MVD-L1-zero / BDOF / DMVR / PROF.
    pub ph_mmvd_fullpel_only_flag: bool,
    pub ph_mvd_l1_zero_flag: bool,
    pub ph_bdof_disabled_flag: bool,
    pub ph_dmvr_disabled_flag: bool,
    pub ph_prof_disabled_flag: bool,

    /// Raw bytes of the `pred_weight_table()` block when it appears in
    /// the PH — captured opaque because the weighted-prediction table
    /// grammar (§7.3.8) is not walked yet.
    pub pred_weight_table_bytes: Vec<u8>,
    pub pred_weight_table_bit_len: u32,

    pub ph_qp_delta: i32,
    pub ph_joint_cbcr_sign_flag: bool,

    pub ph_sao_luma_enabled_flag: bool,
    pub ph_sao_chroma_enabled_flag: bool,

    pub deblocking: PhDeblockingParams,

    /// `ph_extension_length` + the raw extension bytes. Retained so that
    /// later increments can round-trip the header. Present only when
    /// `pps_picture_header_extension_present_flag == 1`.
    pub ph_extension_bytes: Vec<u8>,

    /// Remaining RBSP bytes after the parsed PH body (i.e. whatever the
    /// containing NAL has after `picture_header_structure()` — for a
    /// standalone PH NAL this is `rbsp_trailing_bits()`; for an
    /// embedded PH it is the rest of the slice header).
    pub payload_tail: Vec<u8>,
    pub payload_tail_bit_offset: u8,
    /// Total number of bits consumed from the input buffer before
    /// `payload_tail` begins.
    pub consumed_bits: u64,
}

impl PictureHeader {
    /// True when the PH carries an embedded `ref_pic_lists()` block.
    pub fn has_embedded_rpl(&self) -> bool {
        self.ref_pic_lists.is_some()
    }
}

/// Legacy byte-level view returned by [`parse_picture_header`].
///
/// This struct is kept as a thin wrapper around the header-level fields
/// consumed up to `ph_pic_parameter_set_id` — it lets pre-round-5
/// callers keep compiling while they migrate to
/// [`parse_picture_header_stateful`].
#[derive(Clone, Debug)]
pub struct PictureHeaderLead {
    pub ph_gdr_or_irap_pic_flag: bool,
    pub ph_non_ref_pic_flag: bool,
    pub ph_gdr_pic_flag: bool,
    pub ph_inter_slice_allowed_flag: bool,
    pub ph_intra_slice_allowed_flag: bool,
    pub ph_pic_parameter_set_id: u32,
    pub payload_tail: Vec<u8>,
    pub payload_tail_bit_offset: u8,
    pub consumed_bits: u64,
}

/// Parse the leading fields of a `picture_header_structure()` body
/// (§7.3.2.8) up to `ph_pic_parameter_set_id`. Preserved for callers
/// that do not yet have a paired SPS + PPS to feed
/// [`parse_picture_header_stateful`].
pub fn parse_picture_header(rbsp: &[u8]) -> Result<PictureHeaderLead> {
    if rbsp.is_empty() {
        return Err(Error::invalid("h266 PH: empty RBSP"));
    }
    let mut br = BitReader::new(rbsp);
    let ph_gdr_or_irap_pic_flag = br.u1()? == 1;
    let ph_non_ref_pic_flag = br.u1()? == 1;
    let ph_gdr_pic_flag = if ph_gdr_or_irap_pic_flag {
        br.u1()? == 1
    } else {
        false
    };
    let ph_inter_slice_allowed_flag = br.u1()? == 1;
    let ph_intra_slice_allowed_flag = if ph_inter_slice_allowed_flag {
        br.u1()? == 1
    } else {
        true
    };
    let ph_pic_parameter_set_id = br.ue()?;
    if ph_pic_parameter_set_id > 63 {
        return Err(Error::invalid(format!(
            "h266 PH: ph_pic_parameter_set_id out of range ({ph_pic_parameter_set_id})"
        )));
    }
    let bit_pos = br.bit_position();
    let byte_off = (bit_pos / 8) as usize;
    let bit_off = (bit_pos % 8) as u8;
    let tail = if byte_off < rbsp.len() {
        rbsp[byte_off..].to_vec()
    } else {
        Vec::new()
    };
    Ok(PictureHeaderLead {
        ph_gdr_or_irap_pic_flag,
        ph_non_ref_pic_flag,
        ph_gdr_pic_flag,
        ph_inter_slice_allowed_flag,
        ph_intra_slice_allowed_flag,
        ph_pic_parameter_set_id,
        payload_tail: tail,
        payload_tail_bit_offset: bit_off,
        consumed_bits: bit_pos,
    })
}

/// Parse a `picture_header_rbsp()` body (§7.3.2.7). Identical to
/// `parse_picture_header`, but the name signals that the caller has
/// a PH_NUT RBSP in hand rather than a slice-embedded PH.
pub fn parse_picture_header_rbsp(rbsp: &[u8]) -> Result<PictureHeaderLead> {
    parse_picture_header(rbsp)
}

/// Full stateful parse of §7.3.2.8 picture_header_structure().
///
/// Returns a populated [`PictureHeader`] struct with every field
/// resolved (including spec-defined inferences). The input bytes are
/// the RBSP that starts at the `ph_gdr_or_irap_pic_flag` bit — either
/// a standalone PH NAL RBSP or the sub-buffer that a caller has
/// already advanced past `sh_picture_header_in_slice_header_flag`.
///
/// When `pps_picture_header_extension_present_flag == 1`, the caller
/// is expected to pass an SPS with partition-constraints already
/// populated (the default [`crate::sps::PartitionConstraints`] block
/// is sufficient when the stream relies on SPS-level defaults).
pub fn parse_picture_header_stateful(
    rbsp: &[u8],
    sps: &SeqParameterSet,
    pps: &PicParameterSet,
) -> Result<PictureHeader> {
    if rbsp.is_empty() {
        return Err(Error::invalid("h266 PH: empty RBSP"));
    }
    let mut br = BitReader::new(rbsp);
    let mut ph = PictureHeader::default();

    ph.ph_gdr_or_irap_pic_flag = br.u1()? == 1;
    ph.ph_non_ref_pic_flag = br.u1()? == 1;
    if ph.ph_gdr_or_irap_pic_flag {
        ph.ph_gdr_pic_flag = br.u1()? == 1;
    }
    ph.ph_inter_slice_allowed_flag = br.u1()? == 1;
    ph.ph_intra_slice_allowed_flag = if ph.ph_inter_slice_allowed_flag {
        br.u1()? == 1
    } else {
        true
    };
    ph.ph_pic_parameter_set_id = br.ue()?;
    if ph.ph_pic_parameter_set_id > 63 {
        return Err(Error::invalid(format!(
            "h266 PH: ph_pic_parameter_set_id out of range ({})",
            ph.ph_pic_parameter_set_id
        )));
    }

    // ph_pic_order_cnt_lsb — u(v) with width = log2_max_pic_order_cnt_lsb_minus4 + 4.
    let poc_lsb_width = sps.sps_log2_max_pic_order_cnt_lsb_minus4 as u32 + 4;
    ph.ph_pic_order_cnt_lsb = br.u(poc_lsb_width)?;

    if ph.ph_gdr_pic_flag {
        ph.ph_recovery_poc_cnt = br.ue()?;
    }

    // NumExtraPhBits = count of sps_extra_ph_bit_present_flag[i] set.
    // Our SPS parser retains `sps_num_extra_ph_bytes` but not the
    // individual flag values, so we assume the "all present" upper
    // bound of `sps_num_extra_ph_bytes * 8` bits. Consumers that need
    // the exact count can override via the SPS fields when the parser
    // grows per-bit retention.
    let num_extra_ph_bits = (sps.sps_num_extra_ph_bytes as u32) * 8;
    ph.ph_extra_bits.reserve(num_extra_ph_bits as usize);
    for _ in 0..num_extra_ph_bits {
        ph.ph_extra_bits.push(br.u1()? as u8);
    }

    if sps.sps_poc_msb_cycle_flag {
        ph.ph_poc_msb_cycle_present_flag = br.u1()? == 1;
        if ph.ph_poc_msb_cycle_present_flag {
            let w = sps.sps_poc_msb_cycle_len_minus1 + 1;
            ph.ph_poc_msb_cycle_val = br.u(w)?;
        }
    }

    // ALF block (§7.3.2.8).
    if sps.tool_flags.alf_enabled_flag && pps.pps_alf_info_in_ph_flag {
        ph.ph_alf_enabled_flag = br.u1()? == 1;
        if ph.ph_alf_enabled_flag {
            ph.ph_num_alf_aps_ids_luma = br.u(3)? as u8;
            for _ in 0..ph.ph_num_alf_aps_ids_luma {
                ph.ph_alf_aps_id_luma.push(br.u(3)? as u8);
            }
            if sps.sps_chroma_format_idc != 0 {
                ph.ph_alf_cb_enabled_flag = br.u1()? == 1;
                ph.ph_alf_cr_enabled_flag = br.u1()? == 1;
            }
            if ph.ph_alf_cb_enabled_flag || ph.ph_alf_cr_enabled_flag {
                ph.ph_alf_aps_id_chroma = br.u(3)? as u8;
            }
            if sps.tool_flags.ccalf_enabled_flag {
                ph.ph_alf_cc_cb_enabled_flag = br.u1()? == 1;
                if ph.ph_alf_cc_cb_enabled_flag {
                    ph.ph_alf_cc_cb_aps_id = br.u(3)? as u8;
                }
                ph.ph_alf_cc_cr_enabled_flag = br.u1()? == 1;
                if ph.ph_alf_cc_cr_enabled_flag {
                    ph.ph_alf_cc_cr_aps_id = br.u(3)? as u8;
                }
            }
        }
    }

    if sps.tool_flags.lmcs_enabled_flag {
        ph.ph_lmcs_enabled_flag = br.u1()? == 1;
        if ph.ph_lmcs_enabled_flag {
            ph.ph_lmcs_aps_id = br.u(2)? as u8;
            if sps.sps_chroma_format_idc != 0 {
                ph.ph_chroma_residual_scale_flag = br.u1()? == 1;
            }
        }
    }

    if sps.tool_flags.explicit_scaling_list_enabled_flag {
        ph.ph_explicit_scaling_list_enabled_flag = br.u1()? == 1;
        if ph.ph_explicit_scaling_list_enabled_flag {
            ph.ph_scaling_list_aps_id = br.u(3)? as u8;
        }
    }

    if sps.tool_flags.virtual_boundaries_enabled_flag
        && !sps.tool_flags.virtual_boundaries_present_flag
    {
        ph.ph_virtual_boundaries_present_flag = br.u1()? == 1;
        if ph.ph_virtual_boundaries_present_flag {
            let mut vb = PhVirtualBoundaries::default();
            vb.num_ver = br.ue()?;
            for _ in 0..vb.num_ver {
                vb.pos_x_minus1.push(br.ue()?);
            }
            vb.num_hor = br.ue()?;
            for _ in 0..vb.num_hor {
                vb.pos_y_minus1.push(br.ue()?);
            }
            ph.ph_virtual_boundaries = Some(vb);
        }
    }

    if pps.pps_output_flag_present_flag && !ph.ph_non_ref_pic_flag {
        ph.ph_pic_output_flag = br.u1()? == 1;
    } else {
        // §7.4.3.8: inferred to 1 when not present.
        ph.ph_pic_output_flag = true;
    }

    if pps.pps_rpl_info_in_ph_flag {
        let rpls = parse_ref_pic_lists(&mut br, sps, pps)?;
        ph.ref_pic_lists = Some(rpls);
    }

    if sps
        .partition_constraints
        .partition_constraints_override_enabled_flag
    {
        ph.ph_partition_constraints_override_flag = br.u1()? == 1;
    }

    let mut override_block = PhPartitionOverride::default();
    let mut saw_override = false;

    if ph.ph_intra_slice_allowed_flag {
        if ph.ph_partition_constraints_override_flag {
            saw_override = true;
            override_block.log2_diff_min_qt_min_cb_intra_slice_luma = br.ue()?;
            override_block.max_mtt_hierarchy_depth_intra_slice_luma = br.ue()?;
            if override_block.max_mtt_hierarchy_depth_intra_slice_luma != 0 {
                override_block.log2_diff_max_bt_min_qt_intra_slice_luma = br.ue()?;
                override_block.log2_diff_max_tt_min_qt_intra_slice_luma = br.ue()?;
            }
            if sps.partition_constraints.qtbtt_dual_tree_intra_flag {
                override_block.log2_diff_min_qt_min_cb_intra_slice_chroma = br.ue()?;
                override_block.max_mtt_hierarchy_depth_intra_slice_chroma = br.ue()?;
                if override_block.max_mtt_hierarchy_depth_intra_slice_chroma != 0 {
                    override_block.log2_diff_max_bt_min_qt_intra_slice_chroma = br.ue()?;
                    override_block.log2_diff_max_tt_min_qt_intra_slice_chroma = br.ue()?;
                }
            }
        }
        if pps.pps_cu_qp_delta_enabled_flag {
            ph.ph_cu_qp_delta_subdiv_intra_slice = br.ue()?;
        }
        if pps.pps_cu_chroma_qp_offset_list_enabled_flag {
            ph.ph_cu_chroma_qp_offset_subdiv_intra_slice = br.ue()?;
        }
    }

    if ph.ph_inter_slice_allowed_flag {
        if ph.ph_partition_constraints_override_flag {
            saw_override = true;
            override_block.log2_diff_min_qt_min_cb_inter_slice = br.ue()?;
            override_block.max_mtt_hierarchy_depth_inter_slice = br.ue()?;
            if override_block.max_mtt_hierarchy_depth_inter_slice != 0 {
                override_block.log2_diff_max_bt_min_qt_inter_slice = br.ue()?;
                override_block.log2_diff_max_tt_min_qt_inter_slice = br.ue()?;
            }
        }
        if pps.pps_cu_qp_delta_enabled_flag {
            ph.ph_cu_qp_delta_subdiv_inter_slice = br.ue()?;
        }
        if pps.pps_cu_chroma_qp_offset_list_enabled_flag {
            ph.ph_cu_chroma_qp_offset_subdiv_inter_slice = br.ue()?;
        }

        if sps.tool_flags.temporal_mvp_enabled_flag {
            ph.ph_temporal_mvp_enabled_flag = br.u1()? == 1;
            if ph.ph_temporal_mvp_enabled_flag && pps.pps_rpl_info_in_ph_flag {
                let rpls = ph.ref_pic_lists.as_ref().ok_or_else(|| {
                    Error::invalid(
                        "h266 PH: temporal MVP gate requires ref_pic_lists() that wasn't parsed",
                    )
                })?;
                let num_entries_l1 = rpls[1].rpls.entries.len() as u32;
                let num_entries_l0 = rpls[0].rpls.entries.len() as u32;
                let mut collocated_from_l0 = true;
                if num_entries_l1 > 0 {
                    collocated_from_l0 = br.u1()? == 1;
                    ph.ph_collocated_from_l0_flag = collocated_from_l0;
                } else {
                    // Inferred to 1 when list 1 is empty (§7.4.3.8).
                    ph.ph_collocated_from_l0_flag = true;
                }
                let ref_idx_emitted = (collocated_from_l0 && num_entries_l0 > 1)
                    || (!collocated_from_l0 && num_entries_l1 > 1);
                if ref_idx_emitted {
                    ph.ph_collocated_ref_idx = br.ue()?;
                }
            }
        }

        if sps.tool_flags.mmvd_fullpel_only_enabled_flag {
            ph.ph_mmvd_fullpel_only_flag = br.u1()? == 1;
        }

        // presenceFlag gate from §7.3.2.8: mirrors whether the PH
        // carries the (ph_mvd_l1_zero_flag + optional BDOF/DMVR/PROF)
        // block. `!pps_rpl_info_in_ph_flag` always emits; otherwise
        // emit when num_ref_entries[1][RplsIdx[1]] > 0.
        let presence_flag = if !pps.pps_rpl_info_in_ph_flag {
            true
        } else {
            ph.ref_pic_lists
                .as_ref()
                .map(|r| !r[1].rpls.entries.is_empty())
                .unwrap_or(false)
        };
        if presence_flag {
            ph.ph_mvd_l1_zero_flag = br.u1()? == 1;
            if sps.tool_flags.bdof_control_present_in_ph_flag {
                ph.ph_bdof_disabled_flag = br.u1()? == 1;
            }
            if sps.tool_flags.dmvr_control_present_in_ph_flag {
                ph.ph_dmvr_disabled_flag = br.u1()? == 1;
            }
        }

        if sps.tool_flags.prof_control_present_in_ph_flag {
            ph.ph_prof_disabled_flag = br.u1()? == 1;
        }

        if (pps.pps_weighted_pred_flag || pps.pps_weighted_bipred_flag)
            && pps.pps_wp_info_in_ph_flag
        {
            // pred_weight_table() — grammar §7.3.8 is not walked here;
            // we refuse rather than silently advance, because the
            // table's bit length can't be pre-computed without parsing
            // it. Caller that wants this needs a later increment.
            return Err(Error::unsupported(
                "h266 PH: pred_weight_table() in-header is not yet parsed",
            ));
        }
    }

    if saw_override {
        ph.partition_override = Some(override_block);
    }

    if pps.pps_qp_delta_info_in_ph_flag {
        ph.ph_qp_delta = br.se()?;
    }

    if sps.tool_flags.joint_cbcr_enabled_flag {
        ph.ph_joint_cbcr_sign_flag = br.u1()? == 1;
    }

    if sps.tool_flags.sao_enabled_flag && pps.pps_sao_info_in_ph_flag {
        ph.ph_sao_luma_enabled_flag = br.u1()? == 1;
        if sps.sps_chroma_format_idc != 0 {
            ph.ph_sao_chroma_enabled_flag = br.u1()? == 1;
        }
    }

    if pps.pps_dbf_info_in_ph_flag {
        ph.deblocking.present_flag = br.u1()? == 1;
        if ph.deblocking.present_flag {
            if !pps.pps_deblocking_filter_disabled_flag {
                ph.deblocking.filter_disabled_flag = br.u1()? == 1;
            } else {
                ph.deblocking.filter_disabled_flag = true;
            }
            if !ph.deblocking.filter_disabled_flag {
                ph.deblocking.luma_beta_offset_div2 = br.se()?;
                ph.deblocking.luma_tc_offset_div2 = br.se()?;
                if pps.pps_chroma_tool_offsets_present_flag {
                    ph.deblocking.cb_beta_offset_div2 = br.se()?;
                    ph.deblocking.cb_tc_offset_div2 = br.se()?;
                    ph.deblocking.cr_beta_offset_div2 = br.se()?;
                    ph.deblocking.cr_tc_offset_div2 = br.se()?;
                }
            }
        }
    }

    if pps.pps_picture_header_extension_present_flag {
        let ext_len = br.ue()?;
        if ext_len > 256 {
            return Err(Error::invalid(format!(
                "h266 PH: ph_extension_length out of range ({ext_len})"
            )));
        }
        ph.ph_extension_bytes.reserve(ext_len as usize);
        for _ in 0..ext_len {
            ph.ph_extension_bytes.push(br.u(8)? as u8);
        }
    }

    // Capture the remaining bits as the tail — for a standalone PH
    // NAL this is where `rbsp_trailing_bits()` lives; for an embedded
    // PH it's the rest of the slice header.
    let bit_pos = br.bit_position();
    let byte_off = (bit_pos / 8) as usize;
    let bit_off = (bit_pos % 8) as u8;
    ph.payload_tail = if byte_off < rbsp.len() {
        rbsp[byte_off..].to_vec()
    } else {
        Vec::new()
    };
    ph.payload_tail_bit_offset = bit_off;
    ph.consumed_bits = bit_pos;
    Ok(ph)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// PH leading byte: ph_gdr_or_irap = 1 (IRAP), ph_non_ref = 0,
    /// ph_gdr_pic = 0 (present since gdr_or_irap=1), ph_inter_slice = 0,
    /// ph_pic_parameter_set_id = 0 (ue(v) → "1" = 1 bit).
    ///
    /// Bits: 1 0 0 0 | 1 0 0 0 = 0x88
    #[test]
    fn irap_picture_header() {
        let data = [0x88u8];
        let ph = parse_picture_header(&data).unwrap();
        assert!(ph.ph_gdr_or_irap_pic_flag);
        assert!(!ph.ph_non_ref_pic_flag);
        assert!(!ph.ph_gdr_pic_flag);
        assert!(!ph.ph_inter_slice_allowed_flag);
        // intra_slice_allowed inferred to true when inter not allowed.
        assert!(ph.ph_intra_slice_allowed_flag);
        assert_eq!(ph.ph_pic_parameter_set_id, 0);
    }

    /// Bits: 0 0 1 | 1 | 1 = 0b00111000 = 0x38
    /// ph_gdr_or_irap=0, ph_non_ref=0, ph_inter_slice=1,
    /// ph_intra_slice=1, ph_pic_parameter_set_id=0.
    #[test]
    fn inter_allowed_picture_header() {
        let data = [0b00111000u8];
        let ph = parse_picture_header(&data).unwrap();
        assert!(!ph.ph_gdr_or_irap_pic_flag);
        assert!(!ph.ph_gdr_pic_flag);
        assert!(ph.ph_inter_slice_allowed_flag);
        assert!(ph.ph_intra_slice_allowed_flag);
        assert_eq!(ph.ph_pic_parameter_set_id, 0);
    }

    // ---- Stateful parser tests ----

    fn push_u(bits: &mut Vec<u8>, v: u64, n: u32) {
        for i in (0..n).rev() {
            bits.push(((v >> i) & 1) as u8);
        }
    }
    fn push_ue(bits: &mut Vec<u8>, value: u32) {
        let code_num = value as u64 + 1;
        let mut zeros: u32 = 0;
        while (1u64 << (zeros + 1)) <= code_num {
            zeros += 1;
        }
        for _ in 0..zeros {
            bits.push(0);
        }
        push_u(bits, code_num, zeros + 1);
    }
    fn pack(bits: &[u8]) -> Vec<u8> {
        let mut padded = bits.to_vec();
        while padded.len() % 8 != 0 {
            padded.push(0);
        }
        let mut out = Vec::with_capacity(padded.len() / 8);
        for chunk in padded.chunks(8) {
            let mut b = 0u8;
            for (i, &bit) in chunk.iter().enumerate() {
                b |= bit << (7 - i);
            }
            out.push(b);
        }
        out
    }

    fn synthetic_sps_pps() -> (SeqParameterSet, PicParameterSet) {
        use crate::sps::{PartitionConstraints, ToolFlags};
        let sps = SeqParameterSet {
            sps_seq_parameter_set_id: 0,
            sps_video_parameter_set_id: 0,
            sps_max_sublayers_minus1: 0,
            sps_chroma_format_idc: 1,
            sps_log2_ctu_size_minus5: 2,
            sps_ptl_dpb_hrd_params_present_flag: false,
            profile_tier_level: None,
            sps_gdr_enabled_flag: false,
            sps_ref_pic_resampling_enabled_flag: false,
            sps_res_change_in_clvs_allowed_flag: false,
            sps_pic_width_max_in_luma_samples: 320,
            sps_pic_height_max_in_luma_samples: 240,
            conformance_window: None,
            sps_subpic_info_present_flag: false,
            sps_bitdepth_minus8: 2,
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
        };
        let pps = PicParameterSet {
            pps_pic_parameter_set_id: 0,
            pps_seq_parameter_set_id: 0,
            pps_mixed_nalu_types_in_pic_flag: false,
            pps_pic_width_in_luma_samples: 320,
            pps_pic_height_in_luma_samples: 240,
            conformance_window: None,
            scaling_window: None,
            pps_output_flag_present_flag: false,
            pps_no_pic_partition_flag: true,
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
            pps_dbf_info_in_ph_flag: true,
            pps_rpl_info_in_ph_flag: false, // inline-empty; flag forces SH RPL.
            pps_sao_info_in_ph_flag: true,
            pps_alf_info_in_ph_flag: true,
            pps_wp_info_in_ph_flag: true,
            pps_qp_delta_info_in_ph_flag: true,
            pps_picture_header_extension_present_flag: false,
            pps_slice_header_extension_present_flag: false,
            pps_extension_flag: false,
            partition: None,
        };
        (sps, pps)
    }

    /// Minimal intra-only IRAP PH under a no-tools SPS + no-partition PPS.
    /// Tests that the happy path reads through to the dbf block cleanly.
    #[test]
    fn stateful_intra_only_minimal_ph() {
        let (sps, pps) = synthetic_sps_pps();
        let mut bits: Vec<u8> = Vec::new();
        // ph_gdr_or_irap_pic_flag = 1, ph_non_ref_pic_flag = 0,
        // ph_gdr_pic_flag = 0 (since gdr_or_irap=1 but not gdr),
        // ph_inter_slice_allowed_flag = 0.
        push_u(&mut bits, 1, 1);
        push_u(&mut bits, 0, 1);
        push_u(&mut bits, 0, 1); // ph_gdr_pic_flag
        push_u(&mut bits, 0, 1); // ph_inter_slice_allowed_flag = 0
        // ph_pic_parameter_set_id = 0 → ue "1"
        push_ue(&mut bits, 0);
        // ph_pic_order_cnt_lsb: 8 bits (log2_max = 4 + 4 = 8).
        push_u(&mut bits, 0b0000_0101, 8);
        // NumExtraPhBits = 0 (sps_num_extra_ph_bytes = 0) → no bits.
        // sps_poc_msb_cycle_flag = 0 → skip.
        // ALF: sps_alf_enabled_flag = 0 → skip.
        // LMCS: sps_lmcs_enabled_flag = 0 → skip.
        // Explicit scaling: 0 → skip.
        // Virtual boundaries: 0 → skip.
        // pps_output_flag_present_flag = 0 → ph_pic_output_flag inferred.
        // pps_rpl_info_in_ph_flag = 0 → skip ref_pic_lists().
        // partition_constraints_override_enabled_flag = 0 → skip override flag.
        // ph_intra_slice_allowed_flag (inferred true) block:
        //   ph_partition_constraints_override_flag = false → no override block.
        //   pps_cu_qp_delta_enabled_flag = 0 → skip.
        //   pps_cu_chroma_qp_offset_list_enabled_flag = 0 → skip.
        // ph_inter_slice_allowed_flag = 0 → skip whole inter block.
        // pps_qp_delta_info_in_ph_flag = 1 → read ph_qp_delta.
        push_ue(&mut bits, 0); // ph_qp_delta = 0 → se "1"
        // sps_joint_cbcr_enabled_flag = 0 → skip sign flag.
        // sps_sao_enabled_flag = 0 → skip.
        // pps_dbf_info_in_ph_flag = 1 → read deblocking block gate.
        push_u(&mut bits, 0, 1); // ph_deblocking_params_present_flag = 0
        // pps_picture_header_extension_present_flag = 0 → skip.
        let bytes = pack(&bits);
        let ph = parse_picture_header_stateful(&bytes, &sps, &pps).unwrap();
        assert!(ph.ph_gdr_or_irap_pic_flag);
        assert!(ph.ph_intra_slice_allowed_flag);
        assert!(!ph.ph_inter_slice_allowed_flag);
        assert_eq!(ph.ph_pic_order_cnt_lsb, 0b0000_0101);
        assert_eq!(ph.ph_qp_delta, 0);
        assert!(ph.ref_pic_lists.is_none());
        assert!(ph.ph_pic_output_flag); // inferred to 1
    }

    /// SAO gate: sps_sao_enabled = 1, pps_sao_info_in_ph = 1 → the
    /// stateful parser must read ph_sao_luma + ph_sao_chroma.
    #[test]
    fn stateful_ph_reads_sao_flags() {
        let (mut sps, pps) = synthetic_sps_pps();
        sps.tool_flags.sao_enabled_flag = true;
        let mut bits: Vec<u8> = Vec::new();
        push_u(&mut bits, 1, 1); // ph_gdr_or_irap
        push_u(&mut bits, 0, 1);
        push_u(&mut bits, 0, 1); // ph_gdr_pic
        push_u(&mut bits, 0, 1); // ph_inter_slice_allowed = 0
        push_ue(&mut bits, 0);
        push_u(&mut bits, 0, 8); // ph_pic_order_cnt_lsb = 0
        // ph_qp_delta = 0
        push_ue(&mut bits, 0);
        // sao_luma + sao_chroma
        push_u(&mut bits, 1, 1);
        push_u(&mut bits, 0, 1);
        // dbf gate = 0
        push_u(&mut bits, 0, 1);
        let bytes = pack(&bits);
        let ph = parse_picture_header_stateful(&bytes, &sps, &pps).unwrap();
        assert!(ph.ph_sao_luma_enabled_flag);
        assert!(!ph.ph_sao_chroma_enabled_flag);
    }
}
