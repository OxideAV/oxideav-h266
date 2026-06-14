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

/// Per-list weighted-prediction parameters within
/// [`PredWeightTable`]. Mirrors the §7.3.8 syntax for one direction
/// (L0 or L1).
///
/// `luma_weight_lN_flag` / `chroma_weight_lN_flag` are stored as
/// `Vec<bool>` of length `NumWeightsLN` (§7.4.7). When the flag is
/// `true`, the corresponding `delta_luma_weight_lN[i]` /
/// `luma_offset_lN[i]` (or chroma equivalents) are also captured.
/// Per spec §7.4.7: when `luma_weight_lN_flag[i] == 0`,
/// `LumaWeightLN[i]` is inferred to `2^luma_log2_weight_denom` and
/// `luma_offset_lN[i]` is inferred to 0.
#[derive(Clone, Debug, Default)]
pub struct PredWeightTableList {
    /// `num_lN_weights` (the field is parsed when
    /// `pps_wp_info_in_ph_flag == 1`; otherwise `NumWeightsLN` is
    /// inferred to `NumRefIdxActive[N]` per §7.4.7 and this slot is
    /// left at 0). Round-29: in the PH-carried path
    /// (pps_wp_info_in_ph_flag == 1) we pin this to the parsed value.
    pub num_weights: u32,
    /// `luma_weight_lN_flag[i]`, length `NumWeightsLN`.
    pub luma_weight_flag: Vec<bool>,
    /// `chroma_weight_lN_flag[i]`, length `NumWeightsLN` (skipped
    /// entirely when chroma is absent).
    pub chroma_weight_flag: Vec<bool>,
    /// `delta_luma_weight_lN[i]` for each `i` where
    /// `luma_weight_flag[i] == 1`; collected sparsely as
    /// `(i, delta_luma_weight, luma_offset)` triples.
    pub luma: Vec<LumaWeight>,
    /// `delta_chroma_weight_lN[i][j]` / `delta_chroma_offset_lN[i][j]`
    /// for each `i` where `chroma_weight_flag[i] == 1`. `j ∈ {0, 1}`
    /// for `(Cb, Cr)`.
    pub chroma: Vec<ChromaWeight>,
}

/// One luma weighting record in `pred_weight_table()`.
#[derive(Clone, Copy, Debug, Default)]
pub struct LumaWeight {
    /// Index `i` into `RefPicList[N]` this record applies to.
    pub ref_idx: u32,
    /// `delta_luma_weight_lN[i]`. Spec range: -128..=127.
    pub delta_luma_weight: i32,
    /// `luma_offset_lN[i]`. Spec range: -128..=127.
    pub luma_offset: i32,
}

/// One chroma weighting record in `pred_weight_table()`. Carries both
/// the Cb and Cr deltas because §7.3.8 emits them paired under one
/// `chroma_weight_lN_flag[i]` gate.
#[derive(Clone, Copy, Debug, Default)]
pub struct ChromaWeight {
    /// Index `i` into `RefPicList[N]` this record applies to.
    pub ref_idx: u32,
    /// `delta_chroma_weight_lN[i][0]` (Cb).
    pub delta_chroma_weight_cb: i32,
    /// `delta_chroma_offset_lN[i][0]` (Cb).
    pub delta_chroma_offset_cb: i32,
    /// `delta_chroma_weight_lN[i][1]` (Cr).
    pub delta_chroma_weight_cr: i32,
    /// `delta_chroma_offset_lN[i][1]` (Cr).
    pub delta_chroma_offset_cr: i32,
}

/// `pred_weight_table()` per §7.3.8 — explicit weighted prediction
/// parameters carried in the PH (when `pps_wp_info_in_ph_flag == 1`)
/// or in the slice header. Round-29 wires the PH-carried path.
///
/// Per §7.4.7, `LumaWeightLN[i] = (1 << luma_log2_weight_denom) +
/// delta_luma_weight_lN[i]` (when `luma_weight_lN_flag[i] == 1`) or
/// `2^luma_log2_weight_denom` (when the flag is 0). `ChromaLog2WeightDenom
/// = luma_log2_weight_denom + delta_chroma_log2_weight_denom`. The
/// `ChromaOffsetLN[i][j]` reconstruction uses eq. 144:
///   ChromaOffsetLN[i][j] = Clip3(-128, 127,
///     128 + delta_chroma_offset_lN[i][j] -
///     ((128 * ChromaWeightLN[i][j]) >> ChromaLog2WeightDenom))
#[derive(Clone, Debug, Default)]
pub struct PredWeightTable {
    /// `luma_log2_weight_denom`. Spec range 0..=7.
    pub luma_log2_weight_denom: u32,
    /// `delta_chroma_log2_weight_denom`. Only signalled when
    /// `sps_chroma_format_idc != 0`. Spec range -7..=7 with the sum
    /// `luma_log2_weight_denom + delta_chroma_log2_weight_denom` in
    /// `0..=7`.
    pub delta_chroma_log2_weight_denom: i32,
    /// L0 weighting block.
    pub l0: PredWeightTableList,
    /// L1 weighting block. Empty when `pps_weighted_bipred_flag == 0`
    /// or when `num_ref_entries[1][RplsIdx[1]] == 0` (P-slice path).
    pub l1: PredWeightTableList,
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
    /// the PH — kept for round-tripping the bitstream verbatim.
    pub pred_weight_table_bytes: Vec<u8>,
    pub pred_weight_table_bit_len: u32,
    /// Parsed `pred_weight_table()` per §7.3.8 (round-29). Populated
    /// when the table actually appears in the PH; left at the empty
    /// default otherwise. The walker advances bit-position by the
    /// table's own structural fields rather than relying on
    /// `pred_weight_table_bit_len`.
    pub pred_weight_table: PredWeightTable,

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
            // pred_weight_table() — round-29 walks §7.3.8 in the PH
            // path. The number of L1 entries used to gate the L1 part
            // comes from `num_ref_entries[1][RplsIdx[1]]` per spec —
            // which we read off the parsed PH-level RPL.
            let num_ref_entries_l1 = ph
                .ref_pic_lists
                .as_ref()
                .map(|r| r[1].rpls.entries.len() as u32)
                .unwrap_or(0);
            ph.pred_weight_table = parse_pred_weight_table(
                &mut br,
                sps.sps_chroma_format_idc != 0,
                pps.pps_weighted_bipred_flag,
                pps.pps_wp_info_in_ph_flag,
                num_ref_entries_l1,
            )?;
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

/// Parse `pred_weight_table()` per §7.3.8 starting at the reader's
/// current bit position.
///
/// Inputs:
/// * `chroma_present` — `sps_chroma_format_idc != 0`. When false the
///   chroma fields are skipped entirely (not even the
///   `delta_chroma_log2_weight_denom` `se(v)` is consumed).
/// * `pps_weighted_bipred_flag` — gates the L1 walk's
///   `num_l1_weights` parse along with the next two predicates.
/// * `pps_wp_info_in_ph_flag` — `true` when this table lives in the
///   PH (round-29 path; `num_l0_weights` / `num_l1_weights` are
///   parsed). When `false`, `NumWeightsLN` is the SPS-derived
///   `NumRefIdxActive[N]`, which the caller threads in via
///   `num_ref_entries_l1` when relevant.
/// * `num_ref_entries_l1` — `num_ref_entries[1][RplsIdx[1]]` from
///   the §7.3.10 RPL struct. Only consulted when
///   `pps_weighted_bipred_flag && pps_wp_info_in_ph_flag` to gate
///   the L1 weighting block (§7.3.8 final `if`).
///
/// Returns the populated [`PredWeightTable`].
pub fn parse_pred_weight_table(
    br: &mut BitReader<'_>,
    chroma_present: bool,
    pps_weighted_bipred_flag: bool,
    pps_wp_info_in_ph_flag: bool,
    num_ref_entries_l1: u32,
) -> Result<PredWeightTable> {
    let mut pwt = PredWeightTable::default();
    pwt.luma_log2_weight_denom = br.ue()?;
    if pwt.luma_log2_weight_denom > 7 {
        return Err(Error::invalid(format!(
            "h266 pred_weight_table: luma_log2_weight_denom {} exceeds spec max 7",
            pwt.luma_log2_weight_denom
        )));
    }
    if chroma_present {
        pwt.delta_chroma_log2_weight_denom = br.se()?;
        let chroma_denom = pwt.luma_log2_weight_denom as i32 + pwt.delta_chroma_log2_weight_denom;
        if !(0..=7).contains(&chroma_denom) {
            return Err(Error::invalid(format!(
                "h266 pred_weight_table: ChromaLog2WeightDenom {chroma_denom} \
                 out of spec range 0..=7",
            )));
        }
    }
    // L0 walk -----------------------------------------------------------
    if pps_wp_info_in_ph_flag {
        pwt.l0.num_weights = br.ue()?;
        if pwt.l0.num_weights == 0 || pwt.l0.num_weights > 15 {
            return Err(Error::invalid(format!(
                "h266 pred_weight_table: num_l0_weights {} out of spec range 1..=15",
                pwt.l0.num_weights
            )));
        }
    } else {
        // Caller must have populated NumRefIdxActive[0] elsewhere; we
        // expect the slice-header path to call this with
        // pps_wp_info_in_ph_flag == false and the count threaded
        // separately. Round-29 only wires the PH path, where
        // num_l0_weights is parsed; the slice-header path is left as a
        // no-op here.
        pwt.l0.num_weights = 0;
    }
    let n_l0 = pwt.l0.num_weights as usize;
    pwt.l0.luma_weight_flag.resize(n_l0, false);
    for slot in pwt.l0.luma_weight_flag.iter_mut().take(n_l0) {
        *slot = br.u1()? == 1;
    }
    if chroma_present {
        pwt.l0.chroma_weight_flag.resize(n_l0, false);
        for slot in pwt.l0.chroma_weight_flag.iter_mut().take(n_l0) {
            *slot = br.u1()? == 1;
        }
    }
    for i in 0..n_l0 {
        if pwt.l0.luma_weight_flag[i] {
            let dlw = br.se()?;
            let lo = br.se()?;
            if !(-128..=127).contains(&dlw) || !(-128..=127).contains(&lo) {
                return Err(Error::invalid(format!(
                    "h266 pred_weight_table: L0 luma weight ({dlw}, {lo}) at i={i} \
                     out of spec range -128..=127",
                )));
            }
            pwt.l0.luma.push(LumaWeight {
                ref_idx: i as u32,
                delta_luma_weight: dlw,
                luma_offset: lo,
            });
        }
        if chroma_present && pwt.l0.chroma_weight_flag.get(i).copied().unwrap_or(false) {
            let cb_dw = br.se()?;
            let cb_do = br.se()?;
            let cr_dw = br.se()?;
            let cr_do = br.se()?;
            for v in [cb_dw, cb_do, cr_dw, cr_do] {
                if !(-128..=127).contains(&v) {
                    return Err(Error::invalid(format!(
                        "h266 pred_weight_table: L0 chroma value {v} at i={i} \
                         out of spec range -128..=127",
                    )));
                }
            }
            pwt.l0.chroma.push(ChromaWeight {
                ref_idx: i as u32,
                delta_chroma_weight_cb: cb_dw,
                delta_chroma_offset_cb: cb_do,
                delta_chroma_weight_cr: cr_dw,
                delta_chroma_offset_cr: cr_do,
            });
        }
    }
    // L1 walk -----------------------------------------------------------
    let l1_gate = pps_weighted_bipred_flag && pps_wp_info_in_ph_flag && num_ref_entries_l1 > 0;
    if l1_gate {
        pwt.l1.num_weights = br.ue()?;
        if pwt.l1.num_weights == 0 || pwt.l1.num_weights > 15 {
            return Err(Error::invalid(format!(
                "h266 pred_weight_table: num_l1_weights {} out of spec range 1..=15",
                pwt.l1.num_weights
            )));
        }
        let n_l1 = pwt.l1.num_weights as usize;
        pwt.l1.luma_weight_flag.resize(n_l1, false);
        for slot in pwt.l1.luma_weight_flag.iter_mut().take(n_l1) {
            *slot = br.u1()? == 1;
        }
        if chroma_present {
            pwt.l1.chroma_weight_flag.resize(n_l1, false);
            for slot in pwt.l1.chroma_weight_flag.iter_mut().take(n_l1) {
                *slot = br.u1()? == 1;
            }
        }
        for i in 0..n_l1 {
            if pwt.l1.luma_weight_flag[i] {
                let dlw = br.se()?;
                let lo = br.se()?;
                if !(-128..=127).contains(&dlw) || !(-128..=127).contains(&lo) {
                    return Err(Error::invalid(format!(
                        "h266 pred_weight_table: L1 luma weight ({dlw}, {lo}) at i={i} \
                         out of spec range -128..=127",
                    )));
                }
                pwt.l1.luma.push(LumaWeight {
                    ref_idx: i as u32,
                    delta_luma_weight: dlw,
                    luma_offset: lo,
                });
            }
            if chroma_present && pwt.l1.chroma_weight_flag.get(i).copied().unwrap_or(false) {
                let cb_dw = br.se()?;
                let cb_do = br.se()?;
                let cr_dw = br.se()?;
                let cr_do = br.se()?;
                for v in [cb_dw, cb_do, cr_dw, cr_do] {
                    if !(-128..=127).contains(&v) {
                        return Err(Error::invalid(format!(
                            "h266 pred_weight_table: L1 chroma value {v} at i={i} \
                             out of spec range -128..=127",
                        )));
                    }
                }
                pwt.l1.chroma.push(ChromaWeight {
                    ref_idx: i as u32,
                    delta_chroma_weight_cb: cb_dw,
                    delta_chroma_offset_cb: cb_do,
                    delta_chroma_weight_cr: cr_dw,
                    delta_chroma_offset_cr: cr_do,
                });
            }
        }
    }
    Ok(pwt)
}

/// §7.4.7 — derive `LumaWeightLN[i]` for record `i`. When the per-i
/// `luma_weight_lN_flag == 1`, returns `(1 << luma_log2_weight_denom) +
/// delta_luma_weight_lN[i]`; otherwise the spec-inferred
/// `2^luma_log2_weight_denom`.
pub fn derive_luma_weight(table: &PredWeightTableList, log2_weight_denom: u32, i: u32) -> i32 {
    let base = 1i32 << log2_weight_denom;
    if let Some(rec) = table.luma.iter().find(|l| l.ref_idx == i) {
        base + rec.delta_luma_weight
    } else {
        base
    }
}

/// §7.4.7 — derive `LumaOffsetLN[i]` for record `i`. Returns
/// `luma_offset_lN[i]` when present, else 0 (the spec-inferred value).
pub fn derive_luma_offset(table: &PredWeightTableList, i: u32) -> i32 {
    table
        .luma
        .iter()
        .find(|l| l.ref_idx == i)
        .map(|l| l.luma_offset)
        .unwrap_or(0)
}

/// §7.4.7 — derive `ChromaWeightLN[i][j]` for record `i`, component
/// `j ∈ {0 (Cb), 1 (Cr)}`. When the per-i `chroma_weight_lN_flag == 1`,
/// returns `(1 << chroma_log2_weight_denom) + delta_chroma_weight_lN[i][j]`;
/// otherwise the spec-inferred `2^ChromaLog2WeightDenom`. `j` outside
/// `{0, 1}` returns the inferred base.
pub fn derive_chroma_weight(
    table: &PredWeightTableList,
    chroma_log2_weight_denom: u32,
    i: u32,
    j: u32,
) -> i32 {
    let base = 1i32 << chroma_log2_weight_denom;
    match table.chroma.iter().find(|c| c.ref_idx == i) {
        Some(rec) if j == 0 => base + rec.delta_chroma_weight_cb,
        Some(rec) if j == 1 => base + rec.delta_chroma_weight_cr,
        _ => base,
    }
}

/// §7.4.7 eq. 144 — derive `ChromaOffsetLN[i][j]` for record `i`,
/// component `j ∈ {0 (Cb), 1 (Cr)}`:
///
/// ```text
/// ChromaOffsetLN[i][j] = Clip3(-128, 127,
///     128 + delta_chroma_offset_lN[i][j]
///     - ((128 * ChromaWeightLN[i][j]) >> ChromaLog2WeightDenom))
/// ```
///
/// When the per-i `chroma_weight_lN_flag == 0` (no record), the
/// spec infers `ChromaOffsetLN[i][j] = 0`.
pub fn derive_chroma_offset(
    table: &PredWeightTableList,
    chroma_log2_weight_denom: u32,
    i: u32,
    j: u32,
) -> i32 {
    let Some(rec) = table.chroma.iter().find(|c| c.ref_idx == i) else {
        return 0;
    };
    let delta_offset = match j {
        0 => rec.delta_chroma_offset_cb,
        1 => rec.delta_chroma_offset_cr,
        _ => return 0,
    };
    let chroma_weight = derive_chroma_weight(table, chroma_log2_weight_denom, i, j);
    let raw = 128 + delta_offset - ((128 * chroma_weight) >> chroma_log2_weight_denom);
    raw.clamp(-128, 127)
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
            range_extension: None,
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

    // ---- §7.3.8 pred_weight_table parser tests -----------------------

    fn push_se(bits: &mut Vec<u8>, value: i32) {
        let code = if value <= 0 {
            (-(value as i64) * 2) as u32
        } else {
            (value as i64 * 2 - 1) as u32
        };
        push_ue(bits, code);
    }

    /// Minimal PH-carried table: `luma_log2_weight_denom = 6`,
    /// `delta_chroma_log2_weight_denom = 0`, `num_l0_weights = 1`,
    /// `luma_weight_l0_flag[0] = 0`, `chroma_weight_l0_flag[0] = 0`,
    /// `pps_weighted_bipred_flag = false` so L1 is skipped. Verify the
    /// inferred `LumaWeightL0[0] = 64` (= 1 << 6).
    #[test]
    fn pred_weight_table_minimal_l0_only() {
        let mut bits: Vec<u8> = Vec::new();
        push_ue(&mut bits, 6); // luma_log2_weight_denom = 6
        push_se(&mut bits, 0); // delta_chroma_log2_weight_denom = 0
        push_ue(&mut bits, 1); // num_l0_weights = 1
        push_u(&mut bits, 0, 1); // luma_weight_l0_flag[0] = 0
        push_u(&mut bits, 0, 1); // chroma_weight_l0_flag[0] = 0
        let bytes = pack(&bits);
        let mut br = BitReader::new(&bytes);
        let pwt = parse_pred_weight_table(&mut br, true, false, true, 0).unwrap();
        assert_eq!(pwt.luma_log2_weight_denom, 6);
        assert_eq!(pwt.delta_chroma_log2_weight_denom, 0);
        assert_eq!(pwt.l0.num_weights, 1);
        assert_eq!(pwt.l0.luma_weight_flag, vec![false]);
        assert_eq!(pwt.l0.chroma_weight_flag, vec![false]);
        assert!(pwt.l0.luma.is_empty());
        assert!(pwt.l0.chroma.is_empty());
        // Inferred LumaWeightL0[0] = 1 << 6 = 64.
        assert_eq!(
            derive_luma_weight(&pwt.l0, pwt.luma_log2_weight_denom, 0),
            64
        );
        assert_eq!(derive_luma_offset(&pwt.l0, 0), 0);
        // L1 is skipped (pps_weighted_bipred_flag = false).
        assert_eq!(pwt.l1.num_weights, 0);
    }

    /// L0 record with explicit luma weight: delta_luma = -8, offset = 16.
    /// Result: LumaWeightL0[0] = 64 + (-8) = 56.
    #[test]
    fn pred_weight_table_explicit_luma_weight() {
        let mut bits: Vec<u8> = Vec::new();
        push_ue(&mut bits, 6); // luma_log2_weight_denom = 6
        push_se(&mut bits, 0); // delta_chroma_log2_weight_denom
        push_ue(&mut bits, 1); // num_l0_weights = 1
        push_u(&mut bits, 1, 1); // luma_weight_l0_flag[0] = 1
        push_u(&mut bits, 0, 1); // chroma_weight_l0_flag[0] = 0
        push_se(&mut bits, -8); // delta_luma_weight_l0[0]
        push_se(&mut bits, 16); // luma_offset_l0[0]
        let bytes = pack(&bits);
        let mut br = BitReader::new(&bytes);
        let pwt = parse_pred_weight_table(&mut br, true, false, true, 0).unwrap();
        assert_eq!(pwt.l0.luma.len(), 1);
        let rec = pwt.l0.luma[0];
        assert_eq!(rec.ref_idx, 0);
        assert_eq!(rec.delta_luma_weight, -8);
        assert_eq!(rec.luma_offset, 16);
        // Eq. derived: LumaWeightL0[0] = 64 + (-8) = 56.
        assert_eq!(
            derive_luma_weight(&pwt.l0, pwt.luma_log2_weight_denom, 0),
            56
        );
        assert_eq!(derive_luma_offset(&pwt.l0, 0), 16);
    }

    /// L1 walk fires when pps_weighted_bipred_flag = true and
    /// num_ref_entries_l1 > 0. Verify two-record L0 + one-record L1.
    #[test]
    fn pred_weight_table_l0_and_l1_walks() {
        let mut bits: Vec<u8> = Vec::new();
        push_ue(&mut bits, 5); // luma_log2_weight_denom = 5
        push_se(&mut bits, 1); // delta_chroma_log2 = 1 → ChromaLog2 = 6
                               // L0
        push_ue(&mut bits, 2); // num_l0_weights = 2
        push_u(&mut bits, 1, 1); // luma_weight_l0_flag[0] = 1
        push_u(&mut bits, 0, 1); // luma_weight_l0_flag[1] = 0
        push_u(&mut bits, 0, 1); // chroma_weight_l0_flag[0] = 0
        push_u(&mut bits, 0, 1); // chroma_weight_l0_flag[1] = 0
        push_se(&mut bits, 4); // delta_luma_weight_l0[0]
        push_se(&mut bits, -2); // luma_offset_l0[0]
                                // L1
        push_ue(&mut bits, 1); // num_l1_weights = 1
        push_u(&mut bits, 0, 1); // luma_weight_l1_flag[0] = 0
        push_u(&mut bits, 0, 1); // chroma_weight_l1_flag[0] = 0
        let bytes = pack(&bits);
        let mut br = BitReader::new(&bytes);
        let pwt = parse_pred_weight_table(&mut br, true, true, true, 1).unwrap();
        assert_eq!(pwt.l0.num_weights, 2);
        assert_eq!(pwt.l0.luma.len(), 1);
        assert_eq!(pwt.l0.luma[0].delta_luma_weight, 4);
        assert_eq!(pwt.l0.luma[0].luma_offset, -2);
        assert_eq!(pwt.l1.num_weights, 1);
        assert!(pwt.l1.luma.is_empty()); // luma_weight_l1_flag[0] == 0
                                         // Inferred values: LumaWeightL0[1] = 32 (1 << 5), L1[0] = 32.
        assert_eq!(
            derive_luma_weight(&pwt.l0, pwt.luma_log2_weight_denom, 1),
            32
        );
        assert_eq!(
            derive_luma_weight(&pwt.l1, pwt.luma_log2_weight_denom, 0),
            32
        );
    }

    /// L1 block is suppressed when `num_ref_entries_l1 == 0` even
    /// with `pps_weighted_bipred_flag = true` (P-slice in disguise).
    #[test]
    fn pred_weight_table_l1_suppressed_when_no_l1_refs() {
        let mut bits: Vec<u8> = Vec::new();
        push_ue(&mut bits, 4); // luma_log2_weight_denom
        push_se(&mut bits, 0);
        push_ue(&mut bits, 1); // num_l0_weights = 1
        push_u(&mut bits, 0, 1);
        push_u(&mut bits, 0, 1);
        let bytes = pack(&bits);
        let mut br = BitReader::new(&bytes);
        // num_ref_entries_l1 = 0 — L1 walk skipped.
        let pwt = parse_pred_weight_table(&mut br, true, true, true, 0).unwrap();
        assert_eq!(pwt.l1.num_weights, 0);
    }

    /// Out-of-spec `luma_log2_weight_denom > 7` is rejected.
    #[test]
    fn pred_weight_table_rejects_out_of_range_luma_denom() {
        let mut bits: Vec<u8> = Vec::new();
        push_ue(&mut bits, 8); // > 7 — invalid
        let bytes = pack(&bits);
        let mut br = BitReader::new(&bytes);
        let err = parse_pred_weight_table(&mut br, false, false, true, 0).unwrap_err();
        assert!(format!("{err:?}").contains("luma_log2_weight_denom"));
    }

    /// Chroma is skipped entirely when `chroma_present == false` —
    /// `delta_chroma_log2_weight_denom` is NOT consumed.
    #[test]
    fn pred_weight_table_skips_chroma_when_not_present() {
        let mut bits: Vec<u8> = Vec::new();
        push_ue(&mut bits, 3); // luma_log2_weight_denom
                               // chroma fields skipped.
        push_ue(&mut bits, 1); // num_l0_weights = 1
        push_u(&mut bits, 0, 1); // luma_weight_l0_flag[0] = 0
                                 // No chroma_weight_l0_flag emitted because chroma_present = false.
        let bytes = pack(&bits);
        let mut br = BitReader::new(&bytes);
        let pwt = parse_pred_weight_table(&mut br, false, false, true, 0).unwrap();
        assert_eq!(pwt.luma_log2_weight_denom, 3);
        assert_eq!(pwt.delta_chroma_log2_weight_denom, 0); // default
        assert_eq!(pwt.l0.num_weights, 1);
        assert!(pwt.l0.chroma_weight_flag.is_empty());
    }

    /// `num_l0_weights` must be in 1..=15. Reject `num_l0_weights = 0`.
    #[test]
    fn pred_weight_table_rejects_zero_num_l0_weights() {
        let mut bits: Vec<u8> = Vec::new();
        push_ue(&mut bits, 6);
        push_se(&mut bits, 0);
        push_ue(&mut bits, 0); // num_l0_weights = 0 — invalid
        let bytes = pack(&bits);
        let mut br = BitReader::new(&bytes);
        let err = parse_pred_weight_table(&mut br, true, false, true, 0).unwrap_err();
        assert!(format!("{err:?}").contains("num_l0_weights"));
    }

    /// `num_l0_weights` upper bound: rejects 16.
    #[test]
    fn pred_weight_table_rejects_too_large_num_l0_weights() {
        let mut bits: Vec<u8> = Vec::new();
        push_ue(&mut bits, 6);
        push_se(&mut bits, 0);
        push_ue(&mut bits, 16); // num_l0_weights = 16 — invalid
        let bytes = pack(&bits);
        let mut br = BitReader::new(&bytes);
        let err = parse_pred_weight_table(&mut br, true, false, true, 0).unwrap_err();
        assert!(format!("{err:?}").contains("num_l0_weights"));
    }

    // ---- §7.4.7 chroma weight / offset (eq. 144) derivations ----------

    /// Absent record → inferred `ChromaWeightLN = 2^ChromaLog2WeightDenom`
    /// and `ChromaOffsetLN = 0` for both Cb (j=0) and Cr (j=1).
    #[test]
    fn chroma_weight_offset_inferred_when_absent() {
        let table = PredWeightTableList::default();
        for j in 0..2 {
            assert_eq!(derive_chroma_weight(&table, 6, 0, j), 1 << 6);
            assert_eq!(derive_chroma_offset(&table, 6, 0, j), 0);
        }
    }

    /// Present record: ChromaWeightLN[i][j] = (1 << denom) + delta.
    /// denom = 6 → base 64. Cb delta +12 → 76; Cr delta −8 → 56.
    #[test]
    fn chroma_weight_present_record() {
        let mut table = PredWeightTableList::default();
        table.chroma.push(ChromaWeight {
            ref_idx: 0,
            delta_chroma_weight_cb: 12,
            delta_chroma_offset_cb: 0,
            delta_chroma_weight_cr: -8,
            delta_chroma_offset_cr: 0,
        });
        assert_eq!(derive_chroma_weight(&table, 6, 0, 0), 76);
        assert_eq!(derive_chroma_weight(&table, 6, 0, 1), 56);
    }

    /// eq. 144 for Cb: denom = 6, ChromaWeight = 64 (delta 0),
    /// delta_chroma_offset = 10:
    ///   Clip3(-128, 127, 128 + 10 - ((128 * 64) >> 6))
    ///   = Clip3(-128, 127, 138 - 128) = 10.
    #[test]
    fn chroma_offset_eq144_neutral_weight() {
        let mut table = PredWeightTableList::default();
        table.chroma.push(ChromaWeight {
            ref_idx: 0,
            delta_chroma_weight_cb: 0,
            delta_chroma_offset_cb: 10,
            delta_chroma_weight_cr: 0,
            delta_chroma_offset_cr: -10,
        });
        assert_eq!(derive_chroma_offset(&table, 6, 0, 0), 10);
        assert_eq!(derive_chroma_offset(&table, 6, 0, 1), -10);
    }

    /// eq. 144 with a non-neutral weight. denom = 6, Cb delta_weight
    /// = 32 → ChromaWeight = 96, delta_offset = 0:
    ///   Clip3(-128, 127, 128 + 0 - ((128 * 96) >> 6))
    ///   = Clip3(-128, 127, 128 - 192) = -64.
    #[test]
    fn chroma_offset_eq144_with_weight() {
        let mut table = PredWeightTableList::default();
        table.chroma.push(ChromaWeight {
            ref_idx: 0,
            delta_chroma_weight_cb: 32,
            delta_chroma_offset_cb: 0,
            delta_chroma_weight_cr: 0,
            delta_chroma_offset_cr: 0,
        });
        assert_eq!(derive_chroma_offset(&table, 6, 0, 0), -64);
    }

    /// eq. 144 saturates to the [-128, 127] range. A large positive
    /// delta_offset clamps to 127.
    #[test]
    fn chroma_offset_eq144_clamps() {
        let mut table = PredWeightTableList::default();
        table.chroma.push(ChromaWeight {
            ref_idx: 0,
            delta_chroma_weight_cb: 0,
            delta_chroma_offset_cb: 4 * 127,
            delta_chroma_weight_cr: 0,
            delta_chroma_offset_cr: 0,
        });
        // 128 + 508 - 128 = 508 → clamp 127.
        assert_eq!(derive_chroma_offset(&table, 6, 0, 0), 127);
    }
}
