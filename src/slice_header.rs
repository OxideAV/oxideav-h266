//! VVC Slice Header parser (§7.3.7 — `slice_header()`).
//!
//! Two entry points are offered:
//!
//! * [`parse_slice_header`] — scaffold-level parse that walks only the
//!   embedded `sh_picture_header_in_slice_header_flag` (+ optional
//!   `picture_header_structure()`) and returns the rest of the RBSP as
//!   an opaque tail. Keeps backwards compatibility with the earlier
//!   rounds.
//! * [`parse_slice_header_stateful`] — stateful parse requiring the SPS,
//!   PPS, and a [`PhState`] projection of the picture header. Walks the
//!   slice header from `sh_subpic_id` through the `sh_deblocking_params`
//!   block / `sh_qp_delta` tail (§7.3.7), honouring the "info_in_ph"
//!   inference rules surfaced by the PPS parser. The scaffold currently
//!   enforces `pps_no_pic_partition_flag == 1`, i.e. a single slice per
//!   picture — that lets the slice-address / tile-count derivations
//!   collapse to "no such field" without needing the full partition-
//!   subpicture layout state to be in hand.
//!
//! The entry-point offset list, CABAC zero-word padding, and
//! `slice_header_extension_data_byte[]` (when
//! `pps_slice_header_extension_present_flag == 1`) are captured as
//! opaque byte vectors because their interpretation needs the full CTB
//! / entry-point model that lives outside this module.

use oxideav_core::{Error, Result};

use crate::bitreader::BitReader;
use crate::nal::NalUnitType;
use crate::picture_header::{
    parse_picture_header, parse_picture_header_stateful, parse_pred_weight_table_sh, PictureHeader,
    PictureHeaderLead, PredWeightTable,
};
use crate::pps::PicParameterSet;
use crate::ref_pic_list::{parse_ref_pic_lists, HeaderRefPicList};
use crate::sps::SeqParameterSet;

#[derive(Clone, Debug)]
pub struct SliceHeader {
    pub sh_picture_header_in_slice_header_flag: bool,
    /// Present only when `sh_picture_header_in_slice_header_flag == 1`.
    pub embedded_picture_header: Option<PictureHeaderLead>,
    /// Remaining RBSP bytes after the parsed leading bits.
    pub payload_tail: Vec<u8>,
    /// Bit offset within `payload_tail[0]` where the tail begins.
    pub payload_tail_bit_offset: u8,
}

/// Projection of the picture header needed by the slice-header parser.
///
/// These are the flags and widths whose values are derived from the PH
/// and that the slice-header syntax gates on. Several of them are
/// straight copies of `ph_*_enabled_flag` / `ph_*_used_flag` fields and
/// one (`num_extra_sh_bits`) is a bitstream-level count read out of the
/// SPS. The caller builds this struct once after parsing the PH and
/// hands it to [`parse_slice_header_stateful`].
#[derive(Clone, Copy, Debug)]
pub struct PhState {
    pub ph_inter_slice_allowed_flag: bool,
    pub ph_intra_slice_allowed_flag: bool,
    pub ph_alf_enabled_flag: bool,
    pub ph_lmcs_enabled_flag: bool,
    pub ph_explicit_scaling_list_enabled_flag: bool,
    pub ph_temporal_mvp_enabled_flag: bool,
    /// Round-54 — `ph_sao_luma_enabled_flag` (§7.4.3.7). Used by
    /// [`parse_slice_header_stateful`] to infer `sh_sao_luma_used_flag`
    /// when `pps_sao_info_in_ph_flag == 1` (the SH does not transmit the
    /// flag in that branch — it inherits from the PH).
    pub ph_sao_luma_enabled_flag: bool,
    /// Round-54 — `ph_sao_chroma_enabled_flag` (§7.4.3.7).
    pub ph_sao_chroma_enabled_flag: bool,
    /// `NumExtraShBits` — the count of `sps_extra_sh_bit_present_flag[i]`
    /// entries that are equal to 1 (§7.4.3.4). Zero if
    /// `sps_num_extra_sh_bytes == 0`. Our SPS parser does not keep the
    /// individual flag values yet, so callers that genuinely need >0
    /// must override this.
    pub num_extra_sh_bits: u8,
    /// NAL unit type of the slice NAL — controls whether
    /// `sh_no_output_of_prior_pics_flag` is transmitted.
    pub nal_unit_type: NalUnitType,
    /// r434 — `num_ref_entries[i][RplsIdx[i]]` of the PH-selected RPLs
    /// when `pps_rpl_info_in_ph_flag == 1` (the slice header still
    /// gates its `sh_num_ref_idx_active_override_flag` block on these
    /// counts even though `ref_pic_lists()` itself lives in the PH).
    /// Leave at `[0, 0]` for I-only callers.
    pub num_ref_entries: [u32; 2],
    /// r434 — `ph_collocated_from_l0_flag` (§7.4.3.7), consulted by the
    /// §7.4.8 `sh_collocated_from_l0_flag` inference for B slices.
    pub ph_collocated_from_l0_flag: bool,
    /// r434 — `ph_collocated_ref_idx`, consulted by the §7.4.8
    /// `sh_collocated_ref_idx` inference when
    /// `pps_rpl_info_in_ph_flag == 1`.
    pub ph_collocated_ref_idx: u32,
    /// r456 — the PH-carried ALF selection (§7.3.2.8), inherited by
    /// every slice when `pps_alf_info_in_ph_flag == 1` (§7.4.8: the
    /// absent `sh_alf_*` elements infer to their `ph_alf_*` values).
    pub ph_alf: PhAlfState,
}

/// r456 — `ph_alf_*` (§7.4.3.7) as the slice header inherits them
/// under `pps_alf_info_in_ph_flag == 1`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PhAlfState {
    pub num_alf_aps_ids_luma: u8,
    /// `ph_alf_aps_id_luma[ i ]` for `i < num_alf_aps_ids_luma`
    /// (`u(3)` count, so at most 7).
    pub alf_aps_id_luma: [u8; 7],
    pub alf_cb_enabled_flag: bool,
    pub alf_cr_enabled_flag: bool,
    pub alf_aps_id_chroma: u8,
    pub alf_cc_cb_enabled_flag: bool,
    pub alf_cc_cb_aps_id: u8,
    pub alf_cc_cr_enabled_flag: bool,
    pub alf_cc_cr_aps_id: u8,
}

impl PhAlfState {
    /// Project a parsed picture header's ALF fields.
    pub fn from_ph(ph: &crate::picture_header::PictureHeader) -> Self {
        let mut ids = [0u8; 7];
        for (slot, &id) in ids.iter_mut().zip(&ph.ph_alf_aps_id_luma) {
            *slot = id;
        }
        Self {
            num_alf_aps_ids_luma: ph.ph_num_alf_aps_ids_luma.min(7),
            alf_aps_id_luma: ids,
            alf_cb_enabled_flag: ph.ph_alf_cb_enabled_flag,
            alf_cr_enabled_flag: ph.ph_alf_cr_enabled_flag,
            alf_aps_id_chroma: ph.ph_alf_aps_id_chroma,
            alf_cc_cb_enabled_flag: ph.ph_alf_cc_cb_enabled_flag,
            alf_cc_cb_aps_id: ph.ph_alf_cc_cb_aps_id,
            alf_cc_cr_enabled_flag: ph.ph_alf_cc_cr_enabled_flag,
            alf_cc_cr_aps_id: ph.ph_alf_cc_cr_aps_id,
        }
    }
}

impl Default for PhState {
    fn default() -> Self {
        Self {
            ph_inter_slice_allowed_flag: false,
            ph_intra_slice_allowed_flag: true,
            ph_alf_enabled_flag: false,
            ph_lmcs_enabled_flag: false,
            ph_explicit_scaling_list_enabled_flag: false,
            ph_temporal_mvp_enabled_flag: false,
            ph_sao_luma_enabled_flag: false,
            ph_sao_chroma_enabled_flag: false,
            num_extra_sh_bits: 0,
            nal_unit_type: NalUnitType::TrailNut,
            num_ref_entries: [0, 0],
            ph_collocated_from_l0_flag: true,
            ph_collocated_ref_idx: 0,
            ph_alf: PhAlfState::default(),
        }
    }
}

/// Slice types per §7.4.8.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum SliceType {
    /// Bi-predictive (= 0 in ue(v)).
    B,
    /// Predictive (= 1).
    P,
    /// Intra (= 2).
    #[default]
    I,
}

impl SliceType {
    pub fn from_ue(v: u32) -> Result<Self> {
        match v {
            0 => Ok(SliceType::B),
            1 => Ok(SliceType::P),
            2 => Ok(SliceType::I),
            other => Err(Error::invalid(format!(
                "h266 SH: sh_slice_type out of range ({other})"
            ))),
        }
    }
}

/// Stateful slice header (§7.3.7). Covers the path from
/// `sh_picture_header_in_slice_header_flag` through
/// `sh_deblocking_params_present_flag` / `sh_qp_delta` — the scope
/// specified by round-4 goal (1). The tail past that point (dep_quant,
/// sign_data_hiding, ts_residual_coding, reverse_last_sig_coeff,
/// slice-header extension bytes, entry-point offsets) is captured as
/// `trailing_bits` for a later increment to walk.
#[derive(Clone, Debug, Default)]
pub struct StatefulSliceHeader {
    pub sh_picture_header_in_slice_header_flag: bool,
    pub embedded_picture_header: Option<PictureHeaderLead>,
    /// `sh_subpic_id` (§7.4.8) — `None` when not transmitted.
    pub sh_subpic_id: Option<u32>,
    /// `sh_slice_address` (§7.4.8). 0 when not transmitted (single-slice
    /// / single-tile case).
    pub sh_slice_address: u32,
    /// r434 — `CurrSubpicIdx` (§7.4.8): the subpicture index resolved
    /// from `sh_subpic_id` via eq. 75; 0 without subpicture info.
    pub curr_subpic_idx: u32,
    /// r434 — the picture-level slice index (§3.103): for rectangular
    /// layouts, `sh_slice_address` mapped through the §6.5.1 eq. 23
    /// lists; equals the raster `sh_slice_address` otherwise.
    pub pic_level_slice_idx: u32,
    /// `sh_num_tiles_in_slice_minus1` — emitted only under raster-scan
    /// multi-tile slices.
    pub sh_num_tiles_in_slice_minus1: u32,
    /// Slice type (§7.4.8). `I` when `ph_inter_slice_allowed_flag == 0`.
    pub sh_slice_type: SliceType,
    pub sh_no_output_of_prior_pics_flag: bool,
    pub sh_alf_enabled_flag: bool,
    pub sh_num_alf_aps_ids_luma: u8,
    pub sh_alf_aps_id_luma: Vec<u8>,
    pub sh_alf_cb_enabled_flag: bool,
    pub sh_alf_cr_enabled_flag: bool,
    pub sh_alf_aps_id_chroma: u8,
    pub sh_alf_cc_cb_enabled_flag: bool,
    pub sh_alf_cc_cb_aps_id: u8,
    pub sh_alf_cc_cr_enabled_flag: bool,
    pub sh_alf_cc_cr_aps_id: u8,
    pub sh_lmcs_used_flag: bool,
    pub sh_explicit_scaling_list_used_flag: bool,
    /// r434 — the slice-header-carried `ref_pic_lists()` block
    /// (§7.3.9), present when `pps_rpl_info_in_ph_flag == 0` and the
    /// slice is not an IDR without `sps_idr_rpl_present_flag`.
    pub sh_ref_pic_lists: Option<[HeaderRefPicList; 2]>,
    /// r434 — `sh_num_ref_idx_active_override_flag` (§7.4.8; inferred
    /// to 1 when the enclosing block is absent).
    pub sh_num_ref_idx_active_override_flag: bool,
    /// r434 — transmitted / inferred `sh_num_ref_idx_active_minus1[i]`.
    pub sh_num_ref_idx_active_minus1: [u32; 2],
    /// r434 — `NumRefIdxActive[i]` per §7.4.8 eq. 139.
    pub num_ref_idx_active: [u32; 2],
    /// r434 — `sh_collocated_from_l0_flag` with the §7.4.8 inferences
    /// applied (B: `ph_collocated_from_l0_flag`; P: 1).
    pub sh_collocated_from_l0_flag: bool,
    /// r434 — `sh_collocated_ref_idx` with the §7.4.8 inferences
    /// applied.
    pub sh_collocated_ref_idx: u32,
    /// r434 — slice-header-carried `pred_weight_table()` (§7.3.8),
    /// present under `!pps_wp_info_in_ph_flag` for weighted P/B slices.
    pub sh_pred_weight_table: Option<PredWeightTable>,
    /// r434 — fully parsed embedded `picture_header_structure()` when
    /// `sh_picture_header_in_slice_header_flag == 1` (the legacy
    /// [`Self::embedded_picture_header`] lead is kept alongside for
    /// earlier-round callers).
    pub embedded_ph: Option<Box<PictureHeader>>,
    pub sh_cabac_init_flag: bool,
    /// sh_qp_delta (§7.4.8). Only transmitted when
    /// `pps_qp_delta_info_in_ph_flag == 0`; inferred to 0 otherwise.
    pub sh_qp_delta: i32,
    pub sh_cb_qp_offset: i32,
    pub sh_cr_qp_offset: i32,
    pub sh_joint_cbcr_qp_offset: i32,
    pub sh_cu_chroma_qp_offset_enabled_flag: bool,
    pub sh_sao_luma_used_flag: bool,
    pub sh_sao_chroma_used_flag: bool,
    pub sh_deblocking_params_present_flag: bool,
    pub sh_deblocking_filter_disabled_flag: bool,
    pub sh_luma_beta_offset_div2: i32,
    pub sh_luma_tc_offset_div2: i32,
    pub sh_cb_beta_offset_div2: i32,
    pub sh_cb_tc_offset_div2: i32,
    pub sh_cr_beta_offset_div2: i32,
    pub sh_cr_tc_offset_div2: i32,
    /// `sh_dep_quant_used_flag` (§7.4.8). Inferred to 0 when not present.
    pub sh_dep_quant_used_flag: bool,
    /// `sh_sign_data_hiding_used_flag` (§7.4.8). Inferred to 0 when
    /// not present (including when `sh_dep_quant_used_flag == 1`).
    pub sh_sign_data_hiding_used_flag: bool,
    /// `sh_ts_residual_coding_disabled_flag` — gated by TS +
    /// !dep_quant + !sign-hiding. Inferred to 0 when absent.
    pub sh_ts_residual_coding_disabled_flag: bool,
    /// `sh_ts_residual_coding_rice_idx_minus1` (§7.4.8). Only read when
    /// `sps_ts_residual_coding_rice_present_in_sh_flag` is set, which
    /// lives in the SPS range extension block. Our SPS parser does not
    /// walk the range extension, so the flag is effectively 0 and this
    /// field stays 0.
    pub sh_ts_residual_coding_rice_idx_minus1: u8,
    /// `sh_reverse_last_sig_coeff_flag` — same caveat as the rice idx
    /// above (gated by `sps_reverse_last_sig_coeff_enabled_flag`).
    pub sh_reverse_last_sig_coeff_flag: bool,
    /// Raw bytes of `sh_slice_header_extension_data_byte[]`. Length
    /// equals the transmitted `sh_slice_header_extension_length`.
    pub sh_slice_header_extension_bytes: Vec<u8>,
    /// `sh_entry_offset_len_minus1 + 1` (valid only when
    /// `num_entry_points > 0`). Captured so downstream walkers can
    /// decode `sh_entry_point_offset_minus1[]` without re-parsing.
    pub sh_entry_offset_len: u8,
    /// `sh_entry_point_offset_minus1[i] + 1` — length = `NumEntryPoints`.
    pub sh_entry_point_offsets: Vec<u64>,
    /// Bit position of the first `byte_alignment()` bit within the
    /// input RBSP, i.e. where the `rbsp_slice_trailing_bits()` starts.
    /// Useful for callers that want to locate the start of the CABAC
    /// slice data.
    pub byte_alignment_bit_pos: u64,
    /// Raw tail bytes remaining *after* `byte_alignment()` (i.e. the
    /// start of the coded slice data payload).
    pub trailing_bits: Vec<u8>,
    pub trailing_bit_offset: u8,
}

/// Parse a slice header RBSP bytes-only (foundation scope). Preserved
/// for backwards compatibility with earlier-round callers that only
/// need the embedded PH flag.
pub fn parse_slice_header(rbsp: &[u8]) -> Result<SliceHeader> {
    if rbsp.is_empty() {
        return Err(Error::invalid("h266 SH: empty RBSP"));
    }
    let mut br = BitReader::new(rbsp);
    let sh_picture_header_in_slice_header_flag = br.u1()? == 1;
    let embedded_picture_header = if sh_picture_header_in_slice_header_flag {
        let bit_pos = br.bit_position();
        let tail = collect_bits(rbsp, bit_pos)?;
        let ph = parse_picture_header(&tail)?;
        for _ in 0..ph.consumed_bits {
            br.u1()?;
        }
        Some(ph)
    } else {
        None
    };
    let bit_pos = br.bit_position();
    let byte_off = (bit_pos / 8) as usize;
    let bit_off = (bit_pos % 8) as u8;
    let tail = if byte_off < rbsp.len() {
        rbsp[byte_off..].to_vec()
    } else {
        Vec::new()
    };
    Ok(SliceHeader {
        sh_picture_header_in_slice_header_flag,
        embedded_picture_header,
        payload_tail: tail,
        payload_tail_bit_offset: bit_off,
    })
}

/// Stateful slice header parser — round-4 goal (1). Walks §7.3.7 as
/// far as `sh_qp_delta` / the deblocking block. Requires the current
/// SPS + PPS + a [`PhState`] projection of the picture header.
///
/// Assumptions:
///
/// * `pps_no_pic_partition_flag == 1` (single slice) — the
///   `sh_slice_address` / `sh_num_tiles_in_slice_minus1` branches are
///   elided because `NumSlicesInSubpic == 1` and `NumTilesInPic == 1`.
/// * `pps_*_info_in_ph_flag` values are inferred to 1 under the same
///   assumption (§7.4.3.5) — the RPL / SAO / ALF / WP / QP-delta
///   branches that would require per-slice state therefore collapse.
pub fn parse_slice_header_stateful(
    rbsp: &[u8],
    sps: &SeqParameterSet,
    pps: &PicParameterSet,
    ph_state: &PhState,
) -> Result<StatefulSliceHeader> {
    if rbsp.is_empty() {
        return Err(Error::invalid("h266 SH: empty RBSP"));
    }
    let mut br = BitReader::new(rbsp);
    let sh_picture_header_in_slice_header_flag = br.u1()? == 1;
    // r434 — an embedded `picture_header_structure()` is parsed in
    // FULL through the stateful PH parser (the pre-r434 lead-only parse
    // desynced every stream whose PH carries more than the lead
    // fields, which is all real PH-in-SH wires). The legacy lead
    // projection is kept for earlier-round callers.
    let embedded_ph: Option<Box<PictureHeader>> = if sh_picture_header_in_slice_header_flag {
        let bit_pos = br.bit_position();
        let tail = collect_bits(rbsp, bit_pos)?;
        let ph = parse_picture_header_stateful(&tail, sps, pps)?;
        for _ in 0..ph.consumed_bits {
            br.u1()?;
        }
        Some(Box::new(ph))
    } else {
        None
    };
    let embedded_picture_header = embedded_ph.as_ref().map(|ph| PictureHeaderLead {
        ph_gdr_or_irap_pic_flag: ph.ph_gdr_or_irap_pic_flag,
        ph_non_ref_pic_flag: ph.ph_non_ref_pic_flag,
        ph_gdr_pic_flag: ph.ph_gdr_pic_flag,
        ph_inter_slice_allowed_flag: ph.ph_inter_slice_allowed_flag,
        ph_intra_slice_allowed_flag: ph.ph_intra_slice_allowed_flag,
        ph_pic_parameter_set_id: ph.ph_pic_parameter_set_id,
        payload_tail: ph.payload_tail.clone(),
        payload_tail_bit_offset: ph.payload_tail_bit_offset,
        consumed_bits: ph.consumed_bits,
    });

    // r434 — the slice-header syntax below gates on PH state. When the
    // PH is embedded in this very slice header the caller cannot have
    // known it in advance, so the effective projection is rebuilt from
    // the just-parsed PH (keeping the caller's NAL type + extra-bit
    // count, which come from the NAL header / SPS respectively).
    let eff: PhState = match embedded_ph.as_deref() {
        Some(ph) => PhState {
            ph_inter_slice_allowed_flag: ph.ph_inter_slice_allowed_flag,
            ph_intra_slice_allowed_flag: ph.ph_intra_slice_allowed_flag,
            ph_alf_enabled_flag: ph.ph_alf_enabled_flag,
            ph_lmcs_enabled_flag: ph.ph_lmcs_enabled_flag,
            ph_explicit_scaling_list_enabled_flag: ph.ph_explicit_scaling_list_enabled_flag,
            ph_temporal_mvp_enabled_flag: ph.ph_temporal_mvp_enabled_flag,
            ph_sao_luma_enabled_flag: ph.ph_sao_luma_enabled_flag,
            ph_sao_chroma_enabled_flag: ph.ph_sao_chroma_enabled_flag,
            num_extra_sh_bits: ph_state.num_extra_sh_bits,
            nal_unit_type: ph_state.nal_unit_type,
            num_ref_entries: [
                ph.ref_pic_lists
                    .as_ref()
                    .map(|r| r[0].rpls.entries.len() as u32)
                    .unwrap_or(0),
                ph.ref_pic_lists
                    .as_ref()
                    .map(|r| r[1].rpls.entries.len() as u32)
                    .unwrap_or(0),
            ],
            ph_collocated_from_l0_flag: ph.ph_collocated_from_l0_flag,
            ph_collocated_ref_idx: ph.ph_collocated_ref_idx,
            ph_alf: PhAlfState::from_ph(ph),
        },
        None => *ph_state,
    };
    let ph_state = &eff;

    // sh_subpic_id — present iff sps_subpic_info_present_flag. Width =
    // sps_subpic_id_len_minus1 + 1 per §7.4.8 (inherits the SPS width).
    let sh_subpic_id = if sps.sps_subpic_info_present_flag {
        let id_len = sps
            .subpic_info
            .as_ref()
            .map(|s| s.subpic_id_len_minus1)
            .unwrap_or(0)
            + 1;
        Some(br.u(id_len)?)
    } else {
        None
    };

    // sh_slice_address width (§7.4.8): Ceil(Log2(NumSlicesInSubpic
    // [CurrSubpicIdx])) when rect-slice, Ceil(Log2(NumTilesInPic))
    // otherwise. The field is only emitted when the relevant count
    // is > 1. r434 — CurrSubpicIdx resolves `sh_subpic_id` against the
    // §7.4.3.4 eq. 75 `SubpicIdVal[]`, and NumSlicesInSubpic comes
    // from the §6.5.1 eq. 23 lists (multi-subpicture layouts
    // previously read the whole-picture slice count here and
    // desynced).
    let num_tiles_in_pic = pps
        .partition
        .as_ref()
        .map(|p| p.num_tiles_in_pic)
        .unwrap_or(1);
    let scan = crate::tile_scan::TileScan::derive(sps, pps)?;
    let curr_subpic_idx = match (sh_subpic_id, sps.subpic_info.as_ref()) {
        (Some(id), Some(info)) => {
            let n = info.num_subpics_minus1 + 1;
            let mut found = None;
            for i in 0..n {
                let val = if info.subpic_id_mapping_explicitly_signalled_flag {
                    if pps.pps_subpic_id_mapping_present_flag {
                        pps.subpic_id_mapping
                            .as_ref()
                            .and_then(|m| m.pps_subpic_id.get(i as usize).copied())
                            .ok_or_else(|| {
                                Error::invalid("h266 SH: PPS subpic id mapping too short")
                            })?
                    } else {
                        info.subpic_ids.get(i as usize).copied().ok_or_else(|| {
                            Error::invalid("h266 SH: SPS subpic id mapping too short")
                        })?
                    }
                } else {
                    i
                };
                if val == id {
                    found = Some(i);
                    break;
                }
            }
            found.ok_or_else(|| {
                Error::invalid(format!("h266 SH: sh_subpic_id {id} matches no SubpicIdVal"))
            })?
        }
        _ => 0,
    };
    let num_slices_in_subpic = scan
        .num_slices_in_subpic
        .get(curr_subpic_idx as usize)
        .copied()
        .unwrap_or(1);
    let emit_slice_address = (pps.pps_rect_slice_flag && num_slices_in_subpic > 1)
        || (!pps.pps_rect_slice_flag && num_tiles_in_pic > 1);
    let mut sh_slice_address: u32 = 0;
    if emit_slice_address {
        let width = if pps.pps_rect_slice_flag {
            ceil_log2(num_slices_in_subpic)
        } else {
            ceil_log2(num_tiles_in_pic)
        };
        sh_slice_address = br.u(width)?;
    }
    // Picture-level slice index (§3.103) — identity for
    // single-subpicture rectangular layouts, eq. 23-mapped otherwise.
    let pic_level_slice_idx = if pps.pps_rect_slice_flag || pps.partition.is_none() {
        scan.pic_level_slice_idx(curr_subpic_idx, sh_slice_address)?
    } else {
        sh_slice_address
    };

    // sh_extra_bit loop (SPS-side count).
    for _ in 0..ph_state.num_extra_sh_bits {
        let _ = br.u1()?;
    }

    // sh_num_tiles_in_slice_minus1 — only when the slice spans more
    // than one tile under the raster-scan layout (§7.3.7).
    let mut sh_num_tiles_in_slice_minus1: u32 = 0;
    if !pps.pps_rect_slice_flag && num_tiles_in_pic - sh_slice_address > 1 {
        sh_num_tiles_in_slice_minus1 = br.ue()?;
    }

    // sh_slice_type — only transmitted when `ph_inter_slice_allowed_flag`.
    // Inferred to I otherwise.
    let sh_slice_type = if ph_state.ph_inter_slice_allowed_flag {
        SliceType::from_ue(br.ue()?)?
    } else {
        SliceType::I
    };

    let sh_no_output_of_prior_pics_flag = matches!(
        ph_state.nal_unit_type,
        NalUnitType::IdrWRadl | NalUnitType::IdrNLp | NalUnitType::CraNut | NalUnitType::GdrNut
    ) && br.u1()? == 1;

    // ALF — transmitted only when `sps_alf_enabled_flag &&
    // !pps_alf_info_in_ph_flag`; otherwise (r456) every `sh_alf_*`
    // element infers to the PH's `ph_alf_*` value (§7.4.8).
    let mut out = StatefulSliceHeader::default();
    out.sh_picture_header_in_slice_header_flag = sh_picture_header_in_slice_header_flag;
    out.embedded_picture_header = embedded_picture_header;
    out.embedded_ph = embedded_ph;
    out.sh_subpic_id = sh_subpic_id;
    out.sh_slice_address = sh_slice_address;
    out.curr_subpic_idx = curr_subpic_idx;
    out.pic_level_slice_idx = pic_level_slice_idx;
    out.sh_num_tiles_in_slice_minus1 = sh_num_tiles_in_slice_minus1;
    out.sh_slice_type = sh_slice_type;
    out.sh_no_output_of_prior_pics_flag = sh_no_output_of_prior_pics_flag;

    if sps.tool_flags.alf_enabled_flag && !pps.pps_alf_info_in_ph_flag {
        out.sh_alf_enabled_flag = br.u1()? == 1;
        if out.sh_alf_enabled_flag {
            out.sh_num_alf_aps_ids_luma = br.u(3)? as u8;
            for _ in 0..out.sh_num_alf_aps_ids_luma {
                out.sh_alf_aps_id_luma.push(br.u(3)? as u8);
            }
            if sps.sps_chroma_format_idc != 0 {
                out.sh_alf_cb_enabled_flag = br.u1()? == 1;
                out.sh_alf_cr_enabled_flag = br.u1()? == 1;
            }
            if out.sh_alf_cb_enabled_flag || out.sh_alf_cr_enabled_flag {
                out.sh_alf_aps_id_chroma = br.u(3)? as u8;
            }
            if sps.tool_flags.ccalf_enabled_flag {
                out.sh_alf_cc_cb_enabled_flag = br.u1()? == 1;
                if out.sh_alf_cc_cb_enabled_flag {
                    out.sh_alf_cc_cb_aps_id = br.u(3)? as u8;
                }
                out.sh_alf_cc_cr_enabled_flag = br.u1()? == 1;
                if out.sh_alf_cc_cr_enabled_flag {
                    out.sh_alf_cc_cr_aps_id = br.u(3)? as u8;
                }
            }
        }
    } else if sps.tool_flags.alf_enabled_flag && pps.pps_alf_info_in_ph_flag {
        out.sh_alf_enabled_flag = ph_state.ph_alf_enabled_flag;
        if out.sh_alf_enabled_flag {
            let a = &ph_state.ph_alf;
            out.sh_num_alf_aps_ids_luma = a.num_alf_aps_ids_luma;
            out.sh_alf_aps_id_luma =
                a.alf_aps_id_luma[..usize::from(a.num_alf_aps_ids_luma)].to_vec();
            out.sh_alf_cb_enabled_flag = a.alf_cb_enabled_flag;
            out.sh_alf_cr_enabled_flag = a.alf_cr_enabled_flag;
            out.sh_alf_aps_id_chroma = a.alf_aps_id_chroma;
            out.sh_alf_cc_cb_enabled_flag = a.alf_cc_cb_enabled_flag;
            out.sh_alf_cc_cb_aps_id = a.alf_cc_cb_aps_id;
            out.sh_alf_cc_cr_enabled_flag = a.alf_cc_cr_enabled_flag;
            out.sh_alf_cc_cr_aps_id = a.alf_cc_cr_aps_id;
        }
    }

    if ph_state.ph_lmcs_enabled_flag && !sh_picture_header_in_slice_header_flag {
        out.sh_lmcs_used_flag = br.u1()? == 1;
    } else {
        // §7.4.8 — when not present, sh_lmcs_used_flag is inferred to
        // `sh_picture_header_in_slice_header_flag ?
        // ph_lmcs_enabled_flag : 0` (a PH carried in the slice header
        // speaks for exactly this one slice, so the PH enable IS the
        // per-slice use).
        out.sh_lmcs_used_flag =
            sh_picture_header_in_slice_header_flag && ph_state.ph_lmcs_enabled_flag;
    }
    if ph_state.ph_explicit_scaling_list_enabled_flag && !sh_picture_header_in_slice_header_flag {
        out.sh_explicit_scaling_list_used_flag = br.u1()? == 1;
    } else {
        // §7.4.8 — same PH-in-SH inference shape as sh_lmcs_used_flag.
        out.sh_explicit_scaling_list_used_flag = sh_picture_header_in_slice_header_flag
            && ph_state.ph_explicit_scaling_list_enabled_flag;
    }

    // §7.3.7 — `ref_pic_lists()` in the slice header: present when the
    // PPS keeps RPL info out of the PH and the slice is not an IDR
    // without `sps_idr_rpl_present_flag`.
    let is_idr = matches!(
        ph_state.nal_unit_type,
        NalUnitType::IdrWRadl | NalUnitType::IdrNLp
    );
    if !pps.pps_rpl_info_in_ph_flag && (!is_idr || sps.tool_flags.idr_rpl_present_flag) {
        out.sh_ref_pic_lists = Some(parse_ref_pic_lists(&mut br, sps, pps)?);
    }

    // `num_ref_entries[i][RplsIdx[i]]` — from the SH-carried RPLs when
    // present, else from the PH-carried ones the caller projected in.
    let num_ref_entries: [u32; 2] = match out.sh_ref_pic_lists.as_ref() {
        Some(rpls) => [
            rpls[0].rpls.entries.len() as u32,
            rpls[1].rpls.entries.len() as u32,
        ],
        None => ph_state.num_ref_entries,
    };

    // §7.3.7 — the sh_num_ref_idx_active_override block. The flag is
    // inferred to 1 when the block is absent; the per-list minus1
    // values are inferred to 0 when absent within the block (§7.4.8).
    out.sh_num_ref_idx_active_override_flag = true;
    out.sh_num_ref_idx_active_minus1 = [0, 0];
    if (sh_slice_type != SliceType::I && num_ref_entries[0] > 1)
        || (sh_slice_type == SliceType::B && num_ref_entries[1] > 1)
    {
        out.sh_num_ref_idx_active_override_flag = br.u1()? == 1;
        if out.sh_num_ref_idx_active_override_flag {
            let lists = if sh_slice_type == SliceType::B { 2 } else { 1 };
            for i in 0..lists {
                if num_ref_entries[i] > 1 {
                    let v = br.ue()?;
                    if v > 14 {
                        return Err(Error::invalid(format!(
                            "h266 SH: sh_num_ref_idx_active_minus1[{i}] = {v} out of range 0..=14"
                        )));
                    }
                    out.sh_num_ref_idx_active_minus1[i] = v;
                }
            }
        }
    }

    // §7.4.8 eq. 139 — NumRefIdxActive[i].
    for i in 0..2 {
        out.num_ref_idx_active[i] =
            if sh_slice_type == SliceType::B || (sh_slice_type == SliceType::P && i == 0) {
                if out.sh_num_ref_idx_active_override_flag {
                    out.sh_num_ref_idx_active_minus1[i] + 1
                } else {
                    let default_active = pps.pps_num_ref_idx_default_active_minus1[i] + 1;
                    if num_ref_entries[i] >= default_active {
                        default_active
                    } else {
                        num_ref_entries[i]
                    }
                }
            } else {
                0
            };
    }

    // §7.3.7 `if( sh_slice_type != I )` block — cabac_init /
    // collocated / pred_weight_table.
    //
    // §7.4.8 inference defaults first (P: collocated from L0; B:
    // inherit the PH values; idx: PH value when rpl-in-PH else 0).
    out.sh_collocated_from_l0_flag = if sh_slice_type == SliceType::B {
        ph_state.ph_collocated_from_l0_flag
    } else {
        true
    };
    out.sh_collocated_ref_idx = if pps.pps_rpl_info_in_ph_flag {
        ph_state.ph_collocated_ref_idx
    } else {
        0
    };
    if sh_slice_type != SliceType::I {
        if pps.pps_cabac_init_present_flag {
            out.sh_cabac_init_flag = br.u1()? == 1;
        }
        if ph_state.ph_temporal_mvp_enabled_flag && !pps.pps_rpl_info_in_ph_flag {
            if sh_slice_type == SliceType::B {
                out.sh_collocated_from_l0_flag = br.u1()? == 1;
            }
            if (out.sh_collocated_from_l0_flag && out.num_ref_idx_active[0] > 1)
                || (!out.sh_collocated_from_l0_flag && out.num_ref_idx_active[1] > 1)
            {
                out.sh_collocated_ref_idx = br.ue()?;
            }
        }
        if !pps.pps_wp_info_in_ph_flag
            && ((pps.pps_weighted_pred_flag && sh_slice_type == SliceType::P)
                || (pps.pps_weighted_bipred_flag && sh_slice_type == SliceType::B))
        {
            // §7.4.7 eqs. 144 / 145 — NumWeightsL0 = NumRefIdxActive[0];
            // NumWeightsL1 = NumRefIdxActive[1] for weighted-bipred B
            // slices, 0 otherwise.
            let n_l1 = if pps.pps_weighted_bipred_flag && sh_slice_type == SliceType::B {
                out.num_ref_idx_active[1]
            } else {
                0
            };
            out.sh_pred_weight_table = Some(parse_pred_weight_table_sh(
                &mut br,
                sps.sps_chroma_format_idc != 0,
                out.num_ref_idx_active[0],
                n_l1,
            )?);
        }
    }

    if !pps.pps_qp_delta_info_in_ph_flag {
        out.sh_qp_delta = br.se()?;
    }
    if pps.pps_slice_chroma_qp_offsets_present_flag {
        out.sh_cb_qp_offset = br.se()?;
        out.sh_cr_qp_offset = br.se()?;
        if sps.tool_flags.joint_cbcr_enabled_flag {
            out.sh_joint_cbcr_qp_offset = br.se()?;
        }
    }
    if pps.pps_cu_chroma_qp_offset_list_enabled_flag {
        out.sh_cu_chroma_qp_offset_enabled_flag = br.u1()? == 1;
    }

    if sps.tool_flags.sao_enabled_flag && !pps.pps_sao_info_in_ph_flag {
        out.sh_sao_luma_used_flag = br.u1()? == 1;
        if sps.sps_chroma_format_idc != 0 {
            out.sh_sao_chroma_used_flag = br.u1()? == 1;
        }
    } else if sps.tool_flags.sao_enabled_flag && pps.pps_sao_info_in_ph_flag {
        // Round-54 — §7.4.8 inference: when `pps_sao_info_in_ph_flag == 1`
        // the slice header does not transmit `sh_sao_*_used_flag`; both
        // values are inferred to equal the PH's `ph_sao_*_enabled_flag`.
        out.sh_sao_luma_used_flag = ph_state.ph_sao_luma_enabled_flag;
        out.sh_sao_chroma_used_flag = ph_state.ph_sao_chroma_enabled_flag;
    }

    if pps.pps_deblocking_filter_override_enabled_flag && !pps.pps_dbf_info_in_ph_flag {
        out.sh_deblocking_params_present_flag = br.u1()? == 1;
    }
    if out.sh_deblocking_params_present_flag {
        if !pps.pps_deblocking_filter_disabled_flag {
            out.sh_deblocking_filter_disabled_flag = br.u1()? == 1;
        } else {
            out.sh_deblocking_filter_disabled_flag = true;
        }
        if !out.sh_deblocking_filter_disabled_flag {
            out.sh_luma_beta_offset_div2 = br.se()?;
            out.sh_luma_tc_offset_div2 = br.se()?;
            if pps.pps_chroma_tool_offsets_present_flag {
                out.sh_cb_beta_offset_div2 = br.se()?;
                out.sh_cb_tc_offset_div2 = br.se()?;
                out.sh_cr_beta_offset_div2 = br.se()?;
                out.sh_cr_tc_offset_div2 = br.se()?;
            } else {
                out.sh_cb_beta_offset_div2 = out.sh_luma_beta_offset_div2;
                out.sh_cb_tc_offset_div2 = out.sh_luma_tc_offset_div2;
                out.sh_cr_beta_offset_div2 = out.sh_luma_beta_offset_div2;
                out.sh_cr_tc_offset_div2 = out.sh_luma_tc_offset_div2;
            }
        }
    } else {
        // r434 — §7.4.8 inheritance: absent slice deblocking parameters
        // take the PH-carried values (when `pps_dbf_info_in_ph_flag`
        // routes them there and the PH transmitted them) else the
        // PPS-level values. Pre-r434 they silently stayed at 0 /
        // enabled, mis-filtering every wire with non-default PPS
        // offsets.
        let ph_deb = out
            .embedded_ph
            .as_ref()
            .filter(|_| pps.pps_dbf_info_in_ph_flag)
            .map(|ph| ph.deblocking)
            .filter(|d| d.present_flag);
        match ph_deb {
            Some(d) => {
                out.sh_deblocking_filter_disabled_flag = d.filter_disabled_flag;
                out.sh_luma_beta_offset_div2 = d.luma_beta_offset_div2;
                out.sh_luma_tc_offset_div2 = d.luma_tc_offset_div2;
                out.sh_cb_beta_offset_div2 = d.cb_beta_offset_div2;
                out.sh_cb_tc_offset_div2 = d.cb_tc_offset_div2;
                out.sh_cr_beta_offset_div2 = d.cr_beta_offset_div2;
                out.sh_cr_tc_offset_div2 = d.cr_tc_offset_div2;
            }
            None => {
                out.sh_deblocking_filter_disabled_flag = pps.pps_deblocking_filter_disabled_flag;
                out.sh_luma_beta_offset_div2 = pps.pps_luma_beta_offset_div2;
                out.sh_luma_tc_offset_div2 = pps.pps_luma_tc_offset_div2;
                out.sh_cb_beta_offset_div2 = pps.pps_cb_beta_offset_div2;
                out.sh_cb_tc_offset_div2 = pps.pps_cb_tc_offset_div2;
                out.sh_cr_beta_offset_div2 = pps.pps_cr_beta_offset_div2;
                out.sh_cr_tc_offset_div2 = pps.pps_cr_tc_offset_div2;
            }
        }
    }

    // sh_dep_quant_used_flag, sh_sign_data_hiding_used_flag,
    // sh_ts_residual_coding_disabled_flag, sh_ts_residual_coding_rice_idx_minus1,
    // sh_reverse_last_sig_coeff_flag — §7.3.7 tail. The two range-extension
    // gates (`sps_ts_residual_coding_rice_present_in_sh_flag`,
    // `sps_reverse_last_sig_coeff_enabled_flag`) live in the optional
    // `sps_range_extension()` block (§7.3.2.22); when the block is absent
    // both infer to 0 (§7.4.3.22) and the two reads below are skipped.
    if sps.tool_flags.dep_quant_enabled_flag {
        out.sh_dep_quant_used_flag = br.u1()? == 1;
    }
    if sps.tool_flags.sign_data_hiding_enabled_flag && !out.sh_dep_quant_used_flag {
        out.sh_sign_data_hiding_used_flag = br.u1()? == 1;
    }
    if sps.tool_flags.transform_skip_enabled_flag
        && !out.sh_dep_quant_used_flag
        && !out.sh_sign_data_hiding_used_flag
    {
        out.sh_ts_residual_coding_disabled_flag = br.u1()? == 1;
    }
    // §7.3.7: `sh_ts_residual_coding_rice_idx_minus1` is present iff
    // `!sh_ts_residual_coding_disabled_flag &&
    //  sps_ts_residual_coding_rice_present_in_sh_flag`. When absent it
    // infers to 0 (§7.4.8). `sh_reverse_last_sig_coeff_flag` is present
    // iff `sps_reverse_last_sig_coeff_enabled_flag`; absent → 0 (§7.4.8).
    // Both range-extension gates come from the `sps_range_extension()`
    // payload, which is `None` unless `sps_range_extension_flag == 1`.
    let (rice_present_in_sh, reverse_last_sig_enabled) = match &sps.range_extension {
        Some(rx) => (
            rx.sps_ts_residual_coding_rice_present_in_sh_flag,
            rx.sps_reverse_last_sig_coeff_enabled_flag,
        ),
        None => (false, false),
    };
    if !out.sh_ts_residual_coding_disabled_flag && rice_present_in_sh {
        out.sh_ts_residual_coding_rice_idx_minus1 = br.u(3)? as u8;
    }
    if reverse_last_sig_enabled {
        out.sh_reverse_last_sig_coeff_flag = br.u1()? == 1;
    }

    // Slice-header extension: length + data bytes.
    if pps.pps_slice_header_extension_present_flag {
        let ext_len = br.ue()?;
        if ext_len > 256 {
            return Err(Error::invalid(format!(
                "h266 SH: sh_slice_header_extension_length out of range ({ext_len})"
            )));
        }
        for _ in 0..ext_len {
            out.sh_slice_header_extension_bytes.push(br.u(8)? as u8);
        }
    }

    // Entry-point offsets (§7.4.8 eq. 141) — r418: derived through the
    // §6.5.1 tile scan. The slice's CTB list comes from
    // `CtbAddrInSlice[sh_slice_address]` for rectangular layouts
    // (single-subpicture profile: the subpicture-level address IS the
    // picture-level slice index) or from the
    // `sh_slice_address` / `sh_num_tiles_in_slice_minus1` tile run for
    // raster-scan layouts; NumEntryPoints counts the eq. 141 tile
    // transitions plus, with `sps_entropy_coding_sync_enabled_flag`
    // (WPP), every CTU-row change.
    let num_entry_points = if sps.sps_entry_point_offsets_present_flag {
        let ctbs: Vec<u32> = if pps.pps_rect_slice_flag || pps.partition.is_none() {
            scan.ctb_addr_in_slice
                .get(pic_level_slice_idx as usize)
                .cloned()
                .ok_or_else(|| {
                    Error::invalid(format!(
                        "h266 SH: slice index {pic_level_slice_idx} has no derived CTB list"
                    ))
                })?
        } else {
            scan.raster_slice_ctbs(sh_slice_address, sh_num_tiles_in_slice_minus1 + 1)?
        };
        scan.num_entry_points(&ctbs, sps.sps_entropy_coding_sync_enabled_flag)
    } else {
        0
    };
    if num_entry_points > 0 {
        let len_minus1 = br.ue()?;
        if len_minus1 > 31 {
            return Err(Error::invalid(format!(
                "h266 SH: sh_entry_offset_len_minus1 out of range ({len_minus1})"
            )));
        }
        out.sh_entry_offset_len = len_minus1 as u8 + 1;
        for _ in 0..num_entry_points {
            // u(v) with v = len_minus1 + 1 ≤ 32 bits.
            let off = u64::from(br.u(len_minus1 + 1)?);
            out.sh_entry_point_offsets.push(off + 1);
        }
    }

    // byte_alignment() — §7.3.2.17: a single "1" bit followed by zero or
    // more "0" bits until byte aligned.
    out.byte_alignment_bit_pos = br.bit_position();
    let stop_bit = br.u1()?;
    if stop_bit != 1 {
        return Err(Error::invalid("h266 SH: byte_alignment stop bit != 1"));
    }
    while br.bit_position() % 8 != 0 {
        let pad = br.u1()?;
        if pad != 0 {
            return Err(Error::invalid("h266 SH: byte_alignment padding bit != 0"));
        }
    }

    let bit_pos = br.bit_position();
    let byte_off = (bit_pos / 8) as usize;
    let bit_off = (bit_pos % 8) as u8;
    out.trailing_bits = if byte_off < rbsp.len() {
        rbsp[byte_off..].to_vec()
    } else {
        Vec::new()
    };
    out.trailing_bit_offset = bit_off;
    Ok(out)
}

/// `Ceil(Log2(n))` per the VVC convention (`Ceil(Log2(0)) = 0`).
fn ceil_log2(n: u32) -> u32 {
    if n <= 1 {
        return 0;
    }
    32 - (n - 1).leading_zeros()
}

/// Build a fresh byte-aligned buffer that contains the bits of `rbsp`
/// starting at bit offset `from_bit`. Useful when delegating to a
/// bit-aligned sub-parser.
fn collect_bits(rbsp: &[u8], from_bit: u64) -> Result<Vec<u8>> {
    let total_bits = rbsp.len() as u64 * 8;
    if from_bit > total_bits {
        return Err(Error::invalid("h266 SH: embed offset out of range"));
    }
    let remaining = total_bits - from_bit;
    let mut out = vec![0u8; ((remaining + 7) / 8) as usize];
    let mut src = BitReader::new(rbsp);
    src.skip(from_bit as u32)?;
    let mut bits_written: u64 = 0;
    while bits_written < remaining {
        let n = core::cmp::min(8, (remaining - bits_written) as u32);
        let v = src.u(n)? as u8;
        let byte_idx = (bits_written / 8) as usize;
        let shift = 8 - n;
        out[byte_idx] |= v << shift;
        bits_written += n as u64;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// No embedded PH — first bit = 0, remainder opaque.
    #[test]
    fn no_embedded_ph() {
        let data = [0b0111_0000u8, 0xAB];
        let sh = parse_slice_header(&data).unwrap();
        assert!(!sh.sh_picture_header_in_slice_header_flag);
        assert!(sh.embedded_picture_header.is_none());
        assert_eq!(sh.payload_tail_bit_offset, 1);
        assert_eq!(sh.payload_tail.len(), 2);
    }

    /// Embedded PH: flag=1 followed by the IRAP picture header from the
    /// picture_header tests (0x88 = 1000_1000).
    /// Bits: 1 | 1000_1000 = 0b1100_0100_0... → byte0 = 0xC4.
    #[test]
    fn embedded_ph_flag() {
        let data = [0xC4u8, 0x00];
        let sh = parse_slice_header(&data).unwrap();
        assert!(sh.sh_picture_header_in_slice_header_flag);
        let ph = sh.embedded_picture_header.unwrap();
        assert!(ph.ph_gdr_or_irap_pic_flag);
        assert!(!ph.ph_inter_slice_allowed_flag);
        assert_eq!(ph.ph_pic_parameter_set_id, 0);
    }

    /// Helpers shared by the stateful-parser tests.
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
    /// Append a `byte_alignment()` (a single `1` bit + zero pad to byte
    /// boundary) to the bit vector. Mirrors §7.3.2.17.
    fn push_byte_align(bits: &mut Vec<u8>) {
        bits.push(1);
        while bits.len() % 8 != 0 {
            bits.push(0);
        }
    }

    /// Build a synthetic SPS + PPS pair suitable for exercising the
    /// stateful slice-header parser. Both structs are hand-assembled
    /// (no bitstream round-trip) because we only need a few flags on
    /// the consumer side.
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
            pps_cb_qp_offset_list: Vec::new(),
            pps_cr_qp_offset_list: Vec::new(),
            pps_joint_cbcr_qp_offset_list: Vec::new(),
            pps_deblocking_filter_control_present_flag: false,
            pps_deblocking_filter_override_enabled_flag: false,
            pps_deblocking_filter_disabled_flag: false,
            pps_luma_beta_offset_div2: 0,
            pps_luma_tc_offset_div2: 0,
            pps_cb_beta_offset_div2: 0,
            pps_cb_tc_offset_div2: 0,
            pps_cr_beta_offset_div2: 0,
            pps_cr_tc_offset_div2: 0,
            pps_dbf_info_in_ph_flag: true,
            pps_rpl_info_in_ph_flag: true,
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

    /// r418 — §7.4.8 eq. 141: WPP entry points parsed through the
    /// §6.5.1 tile scan. The 320x240 / 128-CTB picture spans 2 CTU
    /// rows, so with `sps_entry_point_offsets_present_flag` +
    /// `sps_entropy_coding_sync_enabled_flag` the single slice has
    /// NumEntryPoints = 1 — the SH must consume
    /// `sh_entry_offset_len_minus1` and one offset. (Pre-r418 this
    /// configuration was refused as unsupported.)
    #[test]
    fn stateful_wpp_entry_point_offsets_are_parsed() {
        let (mut sps, pps) = synthetic_sps_pps();
        sps.sps_entry_point_offsets_present_flag = true;
        sps.sps_entropy_coding_sync_enabled_flag = true;
        let ph_state = PhState {
            ph_inter_slice_allowed_flag: false,
            ph_intra_slice_allowed_flag: true,
            num_extra_sh_bits: 0,
            nal_unit_type: NalUnitType::IdrWRadl,
            ..Default::default()
        };

        let mut bits: Vec<u8> = Vec::new();
        push_u(&mut bits, 0, 1); // sh_ph_in_sh_flag
        push_u(&mut bits, 0, 1); // sh_no_output_of_prior_pics_flag
        push_ue(&mut bits, 7); // sh_entry_offset_len_minus1 = 7 → 8 bits
        push_u(&mut bits, 0x2a, 8); // sh_entry_point_offset_minus1[0]
        push_byte_align(&mut bits);
        let bytes = pack(&bits);

        let sh = parse_slice_header_stateful(&bytes, &sps, &pps, &ph_state).unwrap();
        assert_eq!(sh.sh_entry_offset_len, 8);
        assert_eq!(sh.sh_entry_point_offsets, vec![0x2a + 1]);
    }

    /// r418 — tiles without WPP: a 2x1-tile raster-scan slice covering
    /// both tiles has one tile transition → one entry point.
    #[test]
    fn stateful_tile_entry_point_parsed_for_raster_slice() {
        use crate::pps::PicPartition;
        let (mut sps, mut pps) = synthetic_sps_pps();
        sps.sps_entry_point_offsets_present_flag = true;
        pps.pps_rect_slice_flag = false; // raster scan
        pps.pps_no_pic_partition_flag = false;
        pps.partition = Some(PicPartition {
            log2_ctu_size_minus5: 2,
            explicit_col_widths: vec![1, 2],
            explicit_row_heights: vec![2],
            col_width_ctbs: vec![1, 2],
            row_height_ctbs: vec![2],
            num_tile_columns: 2,
            num_tile_rows: 1,
            num_tiles_in_pic: 2,
            pps_loop_filter_across_tiles_enabled_flag: false,
            num_slices_in_pic: 1,
            slice_top_left_tile_idx: vec![],
            num_slices_in_subpic: vec![1],
            tile_idx_delta_present_flag: false,
            slice_width_in_tiles: vec![],
            slice_height_in_tiles: vec![],
            slice_height_in_ctus: vec![],
            slice_ctb_row_offset_in_tile: vec![],
        });
        let ph_state = PhState {
            ph_inter_slice_allowed_flag: false,
            ph_intra_slice_allowed_flag: true,
            num_extra_sh_bits: 0,
            nal_unit_type: NalUnitType::IdrWRadl,
            ..Default::default()
        };

        let mut bits: Vec<u8> = Vec::new();
        push_u(&mut bits, 0, 1); // sh_ph_in_sh_flag
        push_u(&mut bits, 0, 1); // sh_slice_address = 0 (1 bit: 2 tiles)
        push_ue(&mut bits, 1); // sh_num_tiles_in_slice_minus1 = 1
        push_u(&mut bits, 0, 1); // sh_no_output_of_prior_pics_flag
        push_ue(&mut bits, 15); // sh_entry_offset_len_minus1 = 15 → 16 bits
        push_u(&mut bits, 0x0102, 16); // sh_entry_point_offset_minus1[0]
        push_byte_align(&mut bits);
        let bytes = pack(&bits);

        let sh = parse_slice_header_stateful(&bytes, &sps, &pps, &ph_state).unwrap();
        assert_eq!(sh.sh_slice_address, 0);
        assert_eq!(sh.sh_num_tiles_in_slice_minus1, 1);
        assert_eq!(sh.sh_entry_offset_len, 16);
        assert_eq!(sh.sh_entry_point_offsets, vec![0x0102 + 1]);
    }

    /// Intra-only IRAP slice: ph_inter_slice_allowed = 0 → sh_slice_type
    /// is inferred to I and NOT transmitted; pps_qp_delta_info_in_ph = 1
    /// skips sh_qp_delta; everything else collapses.
    #[test]
    fn stateful_idr_intra_slice_is_empty_tail() {
        let (sps, pps) = synthetic_sps_pps();
        let ph_state = PhState {
            ph_inter_slice_allowed_flag: false,
            ph_intra_slice_allowed_flag: true,
            num_extra_sh_bits: 0,
            nal_unit_type: NalUnitType::IdrWRadl,
            ..Default::default()
        };

        // Build a minimal slice_header(): sh_ph_in_sh_flag = 0, then
        // immediately sh_no_output_of_prior_pics_flag = 0 (because
        // IDR_W_RADL matches), nothing else emitted until the opaque
        // tail.
        let mut bits: Vec<u8> = Vec::new();
        push_u(&mut bits, 0, 1); // sh_ph_in_sh_flag
        push_u(&mut bits, 0, 1); // sh_no_output_of_prior_pics_flag
        push_byte_align(&mut bits);
        let bytes = pack(&bits);

        let sh = parse_slice_header_stateful(&bytes, &sps, &pps, &ph_state).unwrap();
        assert!(!sh.sh_picture_header_in_slice_header_flag);
        assert_eq!(sh.sh_slice_type, SliceType::I);
        assert!(!sh.sh_no_output_of_prior_pics_flag);
        assert_eq!(sh.sh_qp_delta, 0);
        assert!(!sh.sh_dep_quant_used_flag);
        assert!(sh.sh_slice_header_extension_bytes.is_empty());
    }

    /// §7.4.8 — `sh_lmcs_used_flag` / `sh_explicit_scaling_list_used_flag`
    /// are not transmitted when the picture header rides in the slice
    /// header; both are inferred to
    /// `sh_picture_header_in_slice_header_flag ? ph_*_enabled_flag : 0`.
    #[test]
    fn stateful_ph_in_sh_infers_lmcs_and_scaling_list_flags() {
        // r434 — the embedded PH is now parsed in FULL, so the fixture
        // must carry a complete `picture_header_structure()` and the
        // inference reads the EMBEDDED PH's enables (the caller-side
        // PhState no longer matters for a PH-in-SH slice).
        let (mut sps, pps) = synthetic_sps_pps();
        sps.tool_flags.lmcs_enabled_flag = true;
        sps.tool_flags.explicit_scaling_list_enabled_flag = true;
        let ph_state = PhState {
            ph_inter_slice_allowed_flag: false,
            ph_intra_slice_allowed_flag: true,
            num_extra_sh_bits: 0,
            nal_unit_type: NalUnitType::IdrWRadl,
            ..Default::default()
        };

        // Build one full embedded PH; `enables` toggles the LMCS +
        // scaling-list PH flags.
        let build = |enables: bool| -> Vec<u8> {
            let mut bits: Vec<u8> = Vec::new();
            push_u(&mut bits, 1, 1); // sh_picture_header_in_slice_header_flag
            push_u(&mut bits, 1, 1); // ph_gdr_or_irap_pic_flag
            push_u(&mut bits, 0, 1); // ph_non_ref_pic_flag
            push_u(&mut bits, 0, 1); // ph_gdr_pic_flag
            push_u(&mut bits, 0, 1); // ph_inter_slice_allowed_flag
            push_ue(&mut bits, 0); // ph_pic_parameter_set_id
            push_u(&mut bits, 0, 8); // ph_pic_order_cnt_lsb (u(8))
            if enables {
                push_u(&mut bits, 1, 1); // ph_lmcs_enabled_flag = 1
                push_u(&mut bits, 0, 2); // ph_lmcs_aps_id
                push_u(&mut bits, 0, 1); // ph_chroma_residual_scale_flag
                push_u(&mut bits, 1, 1); // ph_explicit_scaling_list_enabled_flag = 1
                push_u(&mut bits, 0, 3); // ph_scaling_list_aps_id
            } else {
                push_u(&mut bits, 0, 1); // ph_lmcs_enabled_flag = 0
                push_u(&mut bits, 0, 1); // ph_explicit_scaling_list_enabled_flag = 0
            }
            // ref_pic_lists() — pps_rpl_info_in_ph = 1 in the fixture;
            // sps_num_ref_pic_lists = [0, 0] → both lists are inline
            // empty ref_pic_list_struct()s (one ue(0) each).
            push_ue(&mut bits, 0);
            push_ue(&mut bits, 0);
            push_se(&mut bits, 0); // ph_qp_delta (pps_qp_delta_info_in_ph = 1)
            push_u(&mut bits, 0, 1); // ph_deblocking_params_present_flag = 0
                                     // — PH ends; slice header resumes.
            push_u(&mut bits, 0, 1); // sh_no_output_of_prior_pics_flag
            push_byte_align(&mut bits);
            pack(&bits)
        };

        let sh = parse_slice_header_stateful(&build(true), &sps, &pps, &ph_state).unwrap();
        assert!(sh.sh_picture_header_in_slice_header_flag);
        let emb = sh.embedded_ph.as_ref().expect("full embedded PH parsed");
        assert!(emb.ph_lmcs_enabled_flag);
        assert!(emb.ph_explicit_scaling_list_enabled_flag);
        assert!(sh.sh_lmcs_used_flag, "inferred from embedded ph_lmcs");
        assert!(
            sh.sh_explicit_scaling_list_used_flag,
            "inferred from embedded ph_explicit_scaling_list"
        );

        // Same slice with the embedded PH enables off — both infer 0.
        let sh = parse_slice_header_stateful(&build(false), &sps, &pps, &ph_state).unwrap();
        assert!(!sh.sh_lmcs_used_flag);
        assert!(!sh.sh_explicit_scaling_list_used_flag);
    }

    /// r434 — §7.4.8: a slice without its own deblocking parameters
    /// inherits the PPS-level β / tC offsets and disabled flag (they
    /// previously stayed at 0 / enabled, mis-filtering every wire with
    /// non-default PPS offsets).
    #[test]
    fn stateful_deblock_params_inherit_pps() {
        let (sps, mut pps) = synthetic_sps_pps();
        pps.pps_deblocking_filter_control_present_flag = true;
        pps.pps_luma_beta_offset_div2 = 3;
        pps.pps_luma_tc_offset_div2 = -2;
        pps.pps_cb_beta_offset_div2 = 1;
        pps.pps_cb_tc_offset_div2 = 2;
        pps.pps_cr_beta_offset_div2 = -1;
        pps.pps_cr_tc_offset_div2 = 4;
        let ph_state = PhState {
            ph_inter_slice_allowed_flag: false,
            ph_intra_slice_allowed_flag: true,
            num_extra_sh_bits: 0,
            nal_unit_type: NalUnitType::IdrWRadl,
            ..Default::default()
        };
        let mut bits: Vec<u8> = Vec::new();
        push_u(&mut bits, 0, 1); // sh_ph_in_sh_flag
        push_u(&mut bits, 0, 1); // sh_no_output_of_prior_pics_flag
        push_byte_align(&mut bits);
        let bytes = pack(&bits);
        let sh = parse_slice_header_stateful(&bytes, &sps, &pps, &ph_state).unwrap();
        assert!(!sh.sh_deblocking_params_present_flag);
        assert!(!sh.sh_deblocking_filter_disabled_flag);
        assert_eq!(sh.sh_luma_beta_offset_div2, 3);
        assert_eq!(sh.sh_luma_tc_offset_div2, -2);
        assert_eq!(sh.sh_cb_beta_offset_div2, 1);
        assert_eq!(sh.sh_cb_tc_offset_div2, 2);
        assert_eq!(sh.sh_cr_beta_offset_div2, -1);
        assert_eq!(sh.sh_cr_tc_offset_div2, 4);

        // Disabled at PPS level → the slice inherits the disable.
        pps.pps_deblocking_filter_disabled_flag = true;
        let sh = parse_slice_header_stateful(&bytes, &sps, &pps, &ph_state).unwrap();
        assert!(sh.sh_deblocking_filter_disabled_flag);
    }

    /// §7.3.7 — without an embedded PH the two flags ARE transmitted
    /// when the corresponding PH enable is set (and inferred 0
    /// otherwise, which the intra-slice test above already covers).
    #[test]
    fn stateful_lmcs_and_scaling_list_flags_read_when_ph_separate() {
        let (sps, pps) = synthetic_sps_pps();
        let ph_state = PhState {
            ph_inter_slice_allowed_flag: false,
            ph_intra_slice_allowed_flag: true,
            ph_lmcs_enabled_flag: true,
            ph_explicit_scaling_list_enabled_flag: true,
            num_extra_sh_bits: 0,
            nal_unit_type: NalUnitType::IdrWRadl,
            ..Default::default()
        };
        let mut bits: Vec<u8> = Vec::new();
        push_u(&mut bits, 0, 1); // sh_ph_in_sh_flag
        push_u(&mut bits, 0, 1); // sh_no_output_of_prior_pics_flag
        push_u(&mut bits, 1, 1); // sh_lmcs_used_flag = 1
        push_u(&mut bits, 0, 1); // sh_explicit_scaling_list_used_flag = 0
        push_byte_align(&mut bits);
        let bytes = pack(&bits);

        let sh = parse_slice_header_stateful(&bytes, &sps, &pps, &ph_state).unwrap();
        assert!(sh.sh_lmcs_used_flag);
        assert!(!sh.sh_explicit_scaling_list_used_flag);
    }

    /// Inter slice (B, sh_slice_type = 0). ph_inter_slice_allowed = 1 →
    /// sh_slice_type is transmitted. pps_cabac_init_present = 1 forces
    /// sh_cabac_init_flag to be read.
    #[test]
    fn stateful_b_slice_reads_cabac_init() {
        let (sps, mut pps) = synthetic_sps_pps();
        pps.pps_cabac_init_present_flag = true;
        let ph_state = PhState {
            ph_inter_slice_allowed_flag: true,
            ph_intra_slice_allowed_flag: true,
            num_extra_sh_bits: 0,
            nal_unit_type: NalUnitType::TrailNut, // not IDR/CRA/GDR
            ..Default::default()
        };
        let mut bits: Vec<u8> = Vec::new();
        push_u(&mut bits, 0, 1); // sh_ph_in_sh_flag
        push_ue(&mut bits, 0); // sh_slice_type = B
                               // no sh_no_output_of_prior_pics_flag (NalUnitType::TrailNut)
        push_u(&mut bits, 1, 1); // sh_cabac_init_flag = 1
        push_byte_align(&mut bits);
        let bytes = pack(&bits);

        let sh = parse_slice_header_stateful(&bytes, &sps, &pps, &ph_state).unwrap();
        assert_eq!(sh.sh_slice_type, SliceType::B);
        assert!(sh.sh_cabac_init_flag);
    }

    /// When sao_enabled_flag is set AND pps_sao_info_in_ph_flag is 0
    /// (override), sh_sao_luma_used_flag (+ chroma) must be read.
    #[test]
    fn stateful_sao_flags_are_read_when_override_on() {
        let (mut sps, mut pps) = synthetic_sps_pps();
        sps.tool_flags.sao_enabled_flag = true;
        pps.pps_sao_info_in_ph_flag = false;
        let ph_state = PhState {
            ph_inter_slice_allowed_flag: false,
            ph_intra_slice_allowed_flag: true,
            num_extra_sh_bits: 0,
            nal_unit_type: NalUnitType::IdrNLp,
            ..Default::default()
        };
        let mut bits: Vec<u8> = Vec::new();
        push_u(&mut bits, 0, 1); // sh_ph_in_sh_flag
        push_u(&mut bits, 0, 1); // sh_no_output_of_prior_pics_flag
        push_u(&mut bits, 1, 1); // sh_sao_luma_used_flag
        push_u(&mut bits, 1, 1); // sh_sao_chroma_used_flag
        push_byte_align(&mut bits);
        let bytes = pack(&bits);

        let sh = parse_slice_header_stateful(&bytes, &sps, &pps, &ph_state).unwrap();
        assert!(sh.sh_sao_luma_used_flag);
        assert!(sh.sh_sao_chroma_used_flag);
    }

    /// Deblocking override path — when override is enabled in the PPS
    /// and dbf_info_in_ph is 0, the slice header must read the
    /// deblocking-params presence flag + following se(v) offsets.
    #[test]
    fn stateful_deblocking_override_is_parsed() {
        let (sps, mut pps) = synthetic_sps_pps();
        pps.pps_deblocking_filter_control_present_flag = true;
        pps.pps_deblocking_filter_override_enabled_flag = true;
        pps.pps_dbf_info_in_ph_flag = false;
        pps.pps_deblocking_filter_disabled_flag = false;
        let ph_state = PhState {
            ph_inter_slice_allowed_flag: false,
            ph_intra_slice_allowed_flag: true,
            num_extra_sh_bits: 0,
            nal_unit_type: NalUnitType::IdrNLp,
            ..Default::default()
        };
        let mut bits: Vec<u8> = Vec::new();
        push_u(&mut bits, 0, 1); // sh_ph_in_sh_flag
        push_u(&mut bits, 0, 1); // sh_no_output_of_prior_pics_flag
        push_u(&mut bits, 1, 1); // sh_deblocking_params_present_flag
        push_u(&mut bits, 0, 1); // sh_deblocking_filter_disabled_flag = 0
        push_se(&mut bits, 2); // sh_luma_beta_offset_div2
        push_se(&mut bits, -1); // sh_luma_tc_offset_div2
        push_byte_align(&mut bits);
        let bytes = pack(&bits);

        let sh = parse_slice_header_stateful(&bytes, &sps, &pps, &ph_state).unwrap();
        assert!(sh.sh_deblocking_params_present_flag);
        assert!(!sh.sh_deblocking_filter_disabled_flag);
        assert_eq!(sh.sh_luma_beta_offset_div2, 2);
        assert_eq!(sh.sh_luma_tc_offset_div2, -1);
    }

    /// sh_dep_quant_used_flag path: with sps_dep_quant_enabled_flag set,
    /// the stateful parser must read the flag. Once set it suppresses
    /// sh_sign_data_hiding_used_flag + sh_ts_residual_coding_disabled_flag.
    #[test]
    fn stateful_tail_reads_dep_quant_and_gates_followers() {
        let (mut sps, pps) = synthetic_sps_pps();
        sps.tool_flags.dep_quant_enabled_flag = true;
        sps.tool_flags.sign_data_hiding_enabled_flag = true;
        sps.tool_flags.transform_skip_enabled_flag = true;
        let ph_state = PhState {
            ph_inter_slice_allowed_flag: false,
            ph_intra_slice_allowed_flag: true,
            num_extra_sh_bits: 0,
            nal_unit_type: NalUnitType::IdrNLp,
            ..Default::default()
        };
        let mut bits: Vec<u8> = Vec::new();
        push_u(&mut bits, 0, 1); // sh_ph_in_sh_flag
        push_u(&mut bits, 0, 1); // sh_no_output_of_prior_pics_flag
        push_u(&mut bits, 1, 1); // sh_dep_quant_used_flag = 1
                                 // sh_sign_data_hiding_used_flag + sh_ts_residual_coding_disabled_flag
                                 // both suppressed because dep_quant == 1.
        push_byte_align(&mut bits);
        let bytes = pack(&bits);

        let sh = parse_slice_header_stateful(&bytes, &sps, &pps, &ph_state).unwrap();
        assert!(sh.sh_dep_quant_used_flag);
        assert!(!sh.sh_sign_data_hiding_used_flag);
        assert!(!sh.sh_ts_residual_coding_disabled_flag);
    }

    /// §7.3.7 range-extension tail: with `sps_range_extension()` present
    /// and both gates set, the parser must consume the `u(3)`
    /// `sh_ts_residual_coding_rice_idx_minus1` (only when TS residual
    /// coding is *not* disabled) and the `u(1)`
    /// `sh_reverse_last_sig_coeff_flag`.
    #[test]
    fn stateful_tail_reads_range_extension_gates() {
        use crate::sps::SpsRangeExtension;
        let (mut sps, pps) = synthetic_sps_pps();
        sps.tool_flags.transform_skip_enabled_flag = true;
        sps.sps_range_extension_flag = true;
        sps.range_extension = Some(SpsRangeExtension {
            sps_extended_precision_flag: false,
            sps_ts_residual_coding_rice_present_in_sh_flag: true,
            sps_rrc_rice_extension_flag: false,
            sps_persistent_rice_adaptation_enabled_flag: false,
            sps_reverse_last_sig_coeff_enabled_flag: true,
        });
        let ph_state = PhState {
            ph_inter_slice_allowed_flag: false,
            ph_intra_slice_allowed_flag: true,
            num_extra_sh_bits: 0,
            nal_unit_type: NalUnitType::IdrNLp,
            ..Default::default()
        };
        let mut bits: Vec<u8> = Vec::new();
        push_u(&mut bits, 0, 1); // sh_ph_in_sh_flag
        push_u(&mut bits, 0, 1); // sh_no_output_of_prior_pics_flag
        push_u(&mut bits, 0, 1); // sh_ts_residual_coding_disabled_flag = 0
        push_u(&mut bits, 5, 3); // sh_ts_residual_coding_rice_idx_minus1 = 5
        push_u(&mut bits, 1, 1); // sh_reverse_last_sig_coeff_flag = 1
        push_byte_align(&mut bits);
        let bytes = pack(&bits);

        let sh = parse_slice_header_stateful(&bytes, &sps, &pps, &ph_state).unwrap();
        assert!(!sh.sh_ts_residual_coding_disabled_flag);
        assert_eq!(sh.sh_ts_residual_coding_rice_idx_minus1, 5);
        assert!(sh.sh_reverse_last_sig_coeff_flag);
    }

    /// §7.3.7 range-extension tail, disabled-TS branch: when
    /// `sh_ts_residual_coding_disabled_flag == 1` the `u(3)` rice idx is
    /// *not* present even though the SPS gate is set; only the reverse-
    /// last-sig flag (its own independent gate) is read.
    #[test]
    fn stateful_tail_skips_rice_idx_when_ts_disabled() {
        use crate::sps::SpsRangeExtension;
        let (mut sps, pps) = synthetic_sps_pps();
        sps.tool_flags.transform_skip_enabled_flag = true;
        sps.sps_range_extension_flag = true;
        sps.range_extension = Some(SpsRangeExtension {
            sps_extended_precision_flag: false,
            sps_ts_residual_coding_rice_present_in_sh_flag: true,
            sps_rrc_rice_extension_flag: false,
            sps_persistent_rice_adaptation_enabled_flag: false,
            sps_reverse_last_sig_coeff_enabled_flag: true,
        });
        let ph_state = PhState {
            ph_inter_slice_allowed_flag: false,
            ph_intra_slice_allowed_flag: true,
            num_extra_sh_bits: 0,
            nal_unit_type: NalUnitType::IdrNLp,
            ..Default::default()
        };
        let mut bits: Vec<u8> = Vec::new();
        push_u(&mut bits, 0, 1); // sh_ph_in_sh_flag
        push_u(&mut bits, 0, 1); // sh_no_output_of_prior_pics_flag
        push_u(&mut bits, 1, 1); // sh_ts_residual_coding_disabled_flag = 1
                                 // rice idx suppressed (TS coding disabled)
        push_u(&mut bits, 1, 1); // sh_reverse_last_sig_coeff_flag = 1
        push_byte_align(&mut bits);
        let bytes = pack(&bits);

        let sh = parse_slice_header_stateful(&bytes, &sps, &pps, &ph_state).unwrap();
        assert!(sh.sh_ts_residual_coding_disabled_flag);
        assert_eq!(sh.sh_ts_residual_coding_rice_idx_minus1, 0);
        assert!(sh.sh_reverse_last_sig_coeff_flag);
    }

    /// Multi-tile partitioned PPS: when NumTilesInPic > 1 under
    /// raster-scan slices, sh_slice_address + sh_num_tiles_in_slice_minus1
    /// must be read.
    #[test]
    fn stateful_reads_slice_address_under_partition() {
        use crate::pps::PicPartition;
        let (sps, mut pps) = synthetic_sps_pps();
        pps.pps_rect_slice_flag = false; // raster scan
        pps.partition = Some(PicPartition {
            log2_ctu_size_minus5: 2,
            // r434 — geometry must actually cover the 320x240 picture
            // (3x2 CTBs at CtbSizeY = 128): the §6.5.1 tile-scan
            // derivation now runs in the SH parser and validates it.
            explicit_col_widths: vec![1, 2],
            explicit_row_heights: vec![2],
            col_width_ctbs: vec![1, 2],
            row_height_ctbs: vec![2],
            num_tile_columns: 2,
            num_tile_rows: 1,
            num_tiles_in_pic: 2,
            pps_loop_filter_across_tiles_enabled_flag: false,
            num_slices_in_pic: 1,
            slice_top_left_tile_idx: vec![],
            num_slices_in_subpic: vec![1],
            tile_idx_delta_present_flag: false,
            slice_width_in_tiles: vec![],
            slice_height_in_tiles: vec![],
            slice_height_in_ctus: vec![],
            slice_ctb_row_offset_in_tile: vec![],
        });
        let ph_state = PhState {
            ph_inter_slice_allowed_flag: false,
            ph_intra_slice_allowed_flag: true,
            num_extra_sh_bits: 0,
            nal_unit_type: NalUnitType::IdrNLp,
            ..Default::default()
        };
        let mut bits: Vec<u8> = Vec::new();
        push_u(&mut bits, 0, 1); // sh_ph_in_sh_flag
                                 // sh_slice_address is ceil(log2(NumTilesInPic)) = 1 bit.
        push_u(&mut bits, 1, 1); // sh_slice_address = 1 (second tile)
                                 // sh_num_tiles_in_slice_minus1: NumTilesInPic - slice_address = 1,
                                 // so `> 1` fails → not emitted.
        push_u(&mut bits, 0, 1); // sh_no_output_of_prior_pics_flag
        push_byte_align(&mut bits);
        let bytes = pack(&bits);
        let sh = parse_slice_header_stateful(&bytes, &sps, &pps, &ph_state).unwrap();
        assert_eq!(sh.sh_slice_address, 1);
        assert_eq!(sh.sh_num_tiles_in_slice_minus1, 0);
    }

    /// sh_slice_header_extension path: with the PPS extension flag set,
    /// the parser reads ext_len + ext_len bytes of opaque data.
    #[test]
    fn stateful_tail_reads_slice_header_extension() {
        let (sps, mut pps) = synthetic_sps_pps();
        pps.pps_slice_header_extension_present_flag = true;
        let ph_state = PhState {
            ph_inter_slice_allowed_flag: false,
            ph_intra_slice_allowed_flag: true,
            num_extra_sh_bits: 0,
            nal_unit_type: NalUnitType::IdrNLp,
            ..Default::default()
        };
        let mut bits: Vec<u8> = Vec::new();
        push_u(&mut bits, 0, 1); // sh_ph_in_sh_flag
        push_u(&mut bits, 0, 1); // sh_no_output_of_prior_pics_flag
                                 // sh_slice_header_extension_length = 2 → ue "011".
        push_ue(&mut bits, 2);
        for _ in 0..8 {
            bits.push(1); // sh_slice_header_extension_data_byte[0] = 0xFF
        }
        for _ in 0..8 {
            bits.push(0); // sh_slice_header_extension_data_byte[1] = 0x00
        }
        push_byte_align(&mut bits);
        let bytes = pack(&bits);

        let sh = parse_slice_header_stateful(&bytes, &sps, &pps, &ph_state).unwrap();
        assert_eq!(sh.sh_slice_header_extension_bytes, vec![0xFF, 0x00]);
    }

    /// r456 — §7.4.8: under `pps_alf_info_in_ph_flag == 1` the slice
    /// header carries no ALF syntax and every `sh_alf_*` element
    /// infers to the PH's `ph_alf_*` value (SUBPIC_C_1 signals ALF in
    /// the PH; the pre-r456 parser left the slice ALF-off and desynced
    /// on the first CTU's ALF bins).
    #[test]
    fn stateful_alf_inherits_ph_when_alf_info_in_ph() {
        let (mut sps, pps) = synthetic_sps_pps();
        sps.tool_flags.alf_enabled_flag = true;
        sps.tool_flags.ccalf_enabled_flag = true;
        assert!(pps.pps_alf_info_in_ph_flag);
        let mut ids = [0u8; 7];
        ids[0] = 7;
        ids[1] = 2;
        let ph_state = PhState {
            ph_inter_slice_allowed_flag: false,
            ph_intra_slice_allowed_flag: true,
            ph_alf_enabled_flag: true,
            num_extra_sh_bits: 0,
            nal_unit_type: NalUnitType::IdrWRadl,
            ph_alf: PhAlfState {
                num_alf_aps_ids_luma: 2,
                alf_aps_id_luma: ids,
                alf_cb_enabled_flag: true,
                alf_cr_enabled_flag: false,
                alf_aps_id_chroma: 5,
                alf_cc_cb_enabled_flag: false,
                alf_cc_cb_aps_id: 0,
                alf_cc_cr_enabled_flag: true,
                alf_cc_cr_aps_id: 3,
            },
            ..Default::default()
        };
        let mut bits: Vec<u8> = Vec::new();
        push_u(&mut bits, 0, 1); // sh_ph_in_sh_flag
        push_u(&mut bits, 0, 1); // sh_no_output_of_prior_pics_flag
        push_byte_align(&mut bits);
        let bytes = pack(&bits);
        let sh = parse_slice_header_stateful(&bytes, &sps, &pps, &ph_state).unwrap();
        assert!(sh.sh_alf_enabled_flag);
        assert_eq!(sh.sh_num_alf_aps_ids_luma, 2);
        assert_eq!(sh.sh_alf_aps_id_luma, vec![7, 2]);
        assert!(sh.sh_alf_cb_enabled_flag);
        assert!(!sh.sh_alf_cr_enabled_flag);
        assert_eq!(sh.sh_alf_aps_id_chroma, 5);
        assert!(!sh.sh_alf_cc_cb_enabled_flag);
        assert!(sh.sh_alf_cc_cr_enabled_flag);
        assert_eq!(sh.sh_alf_cc_cr_aps_id, 3);
        // PH ALF off → the slice inherits "off" (no bins consumed).
        let off = PhState {
            ph_alf_enabled_flag: false,
            ..ph_state
        };
        let sh = parse_slice_header_stateful(&bytes, &sps, &pps, &off).unwrap();
        assert!(!sh.sh_alf_enabled_flag);
        assert!(sh.sh_alf_aps_id_luma.is_empty());
    }
}
