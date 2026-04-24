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
use crate::picture_header::{parse_picture_header, PictureHeaderLead};
use crate::pps::PicParameterSet;
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
    /// `NumExtraShBits` — the count of `sps_extra_sh_bit_present_flag[i]`
    /// entries that are equal to 1 (§7.4.3.4). Zero if
    /// `sps_num_extra_sh_bytes == 0`. Our SPS parser does not keep the
    /// individual flag values yet, so callers that genuinely need >0
    /// must override this.
    pub num_extra_sh_bits: u8,
    /// NAL unit type of the slice NAL — controls whether
    /// `sh_no_output_of_prior_pics_flag` is transmitted.
    pub nal_unit_type: NalUnitType,
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
            num_extra_sh_bits: 0,
            nal_unit_type: NalUnitType::TrailNut,
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
    if !pps.pps_no_pic_partition_flag {
        return Err(Error::unsupported(
            "h266 SH: stateful parser requires pps_no_pic_partition_flag = 1",
        ));
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

    // sh_slice_address / sh_extra_bit / sh_num_tiles_in_slice_minus1:
    // the first and third are skipped because the single-slice assumption
    // implies `NumSlicesInSubpic == 1` and `NumTilesInPic == 1`. The
    // sh_extra_bit loop still runs — the SPS holds the count.
    for _ in 0..ph_state.num_extra_sh_bits {
        let _ = br.u1()?;
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

    // ALF — only when `sps_alf_enabled_flag && !pps_alf_info_in_ph_flag`.
    // Under our assumption pps_alf_info_in_ph_flag is always 1 so this
    // whole block is skipped.
    let mut out = StatefulSliceHeader::default();
    out.sh_picture_header_in_slice_header_flag = sh_picture_header_in_slice_header_flag;
    out.embedded_picture_header = embedded_picture_header;
    out.sh_subpic_id = sh_subpic_id;
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
    }

    if ph_state.ph_lmcs_enabled_flag && !sh_picture_header_in_slice_header_flag {
        out.sh_lmcs_used_flag = br.u1()? == 1;
    }
    if ph_state.ph_explicit_scaling_list_enabled_flag && !sh_picture_header_in_slice_header_flag {
        out.sh_explicit_scaling_list_used_flag = br.u1()? == 1;
    }

    // ref_pic_lists() — skipped because pps_rpl_info_in_ph_flag = 1 in
    // our single-slice scenario. sh_num_ref_idx_active_override_flag is
    // also skipped (num_ref_entries[0/1] are known from the PH-level RPL
    // and resolved by the RefPicList builder).

    // `if( sh_slice_type != I )` block — skipped (cabac_init /
    // collocated / pred_weight_table). With pps_rpl_info_in_ph_flag = 1
    // the collocated fields live in the PH. Weighted pred table is also
    // handled via pps_wp_info_in_ph_flag = 1.
    //
    // Honour the few elements that *are* still emitted:
    if sh_slice_type != SliceType::I && pps.pps_cabac_init_present_flag {
        out.sh_cabac_init_flag = br.u1()? == 1;
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
            }
        }
    }

    // sh_dep_quant_used_flag, sh_sign_data_hiding_used_flag,
    // sh_ts_residual_coding_disabled_flag, sh_ts_residual_coding_rice_idx_minus1,
    // sh_reverse_last_sig_coeff_flag — §7.3.7 tail. The SPS range-extension
    // gates (`sps_ts_residual_coding_rice_present_in_sh_flag`,
    // `sps_reverse_last_sig_coeff_enabled_flag`) are not yet parsed by
    // our SPS, so both are treated as 0 (the inference when
    // `sps_range_extension_flag == 0`, §7.4.3.22).
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
    // sh_ts_residual_coding_rice_idx_minus1 / sh_reverse_last_sig_coeff_flag
    // are gated by range-extension SPS flags that default to 0; left unset.

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

    // Entry-point offsets (§7.4.8). Under the single-slice / single-tile
    // assumption enforced at the top of this function (NumTilesInPic = 1,
    // NumSlicesInSubpic = 1), NumEntryPoints is 0 whenever
    // sps_entry_point_offsets_present_flag = 0. When the flag is set and
    // the slice spans more than one CTU row with entropy-coding-sync, the
    // count is non-zero — derivation needs the full CTU layout that the
    // current scaffold does not carry. We refuse the ambiguous case so
    // the caller can't silently consume the wrong number of bits.
    if sps.sps_entry_point_offsets_present_flag && sps.sps_entropy_coding_sync_enabled_flag {
        return Err(Error::unsupported(
            "h266 SH: entry-point offsets with entropy-coding-sync are not yet walked",
        ));
    }
    // NumEntryPoints = 0 in every supported configuration → skip the
    // entire `if( NumEntryPoints > 0 )` block.

    // byte_alignment() — §7.3.2.17: a single "1" bit followed by zero or
    // more "0" bits until byte aligned.
    out.byte_alignment_bit_pos = br.bit_position();
    let stop_bit = br.u1()?;
    if stop_bit != 1 {
        return Err(Error::invalid(
            "h266 SH: byte_alignment stop bit != 1",
        ));
    }
    while br.bit_position() % 8 != 0 {
        let pad = br.u1()?;
        if pad != 0 {
            return Err(Error::invalid(
                "h266 SH: byte_alignment padding bit != 0",
            ));
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
            pps_rpl_info_in_ph_flag: true,
            pps_sao_info_in_ph_flag: true,
            pps_alf_info_in_ph_flag: true,
            pps_wp_info_in_ph_flag: true,
            pps_qp_delta_info_in_ph_flag: true,
            pps_picture_header_extension_present_flag: false,
            pps_slice_header_extension_present_flag: false,
            pps_extension_flag: false,
        };
        (sps, pps)
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
}
