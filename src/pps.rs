//! VVC Picture Parameter Set parser (§7.3.2.5).
//!
//! Foundation scope: identifier + picture dimension fields, conformance
//! and scaling windows, the partition-gating flags. The tile/slice
//! layout sub-syntax (under `!pps_no_pic_partition_flag`) references
//! derived variables (`NumTilesInPic`, `SliceTopLeftTileIdx[]`, …) whose
//! population requires full SPS context; we refuse to walk it here and
//! surface an `Unsupported` error when the PPS signals per-picture
//! tiling. A single-tile / single-slice PPS is the common case for
//! conformance fixtures.

use oxideav_core::{Error, Result};

use crate::bitreader::BitReader;

#[derive(Clone, Copy, Debug)]
pub struct ConformanceWindow {
    pub left_offset: u32,
    pub right_offset: u32,
    pub top_offset: u32,
    pub bottom_offset: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct ScalingWindow {
    pub left_offset: i32,
    pub right_offset: i32,
    pub top_offset: i32,
    pub bottom_offset: i32,
}

#[derive(Clone, Debug)]
pub struct PicParameterSet {
    pub pps_pic_parameter_set_id: u8,
    pub pps_seq_parameter_set_id: u8,
    pub pps_mixed_nalu_types_in_pic_flag: bool,
    pub pps_pic_width_in_luma_samples: u32,
    pub pps_pic_height_in_luma_samples: u32,
    pub conformance_window: Option<ConformanceWindow>,
    pub scaling_window: Option<ScalingWindow>,
    pub pps_output_flag_present_flag: bool,
    pub pps_no_pic_partition_flag: bool,
    pub pps_subpic_id_mapping_present_flag: bool,
    /// `pps_rect_slice_flag` — inferred to `true` when the partitioning
    /// block is skipped (§7.4.3.5 inference: defaults to 1 when not
    /// transmitted). `pps_no_pic_partition_flag == 1` also forces
    /// `NumTilesInPic = 1` / `pps_num_slices_in_pic_minus1 = 0`.
    pub pps_rect_slice_flag: bool,
    pub pps_single_slice_per_subpic_flag: bool,
    pub pps_loop_filter_across_slices_enabled_flag: bool,
    pub pps_cabac_init_present_flag: bool,
    pub pps_num_ref_idx_default_active_minus1: [u32; 2],
    pub pps_rpl1_idx_present_flag: bool,
    pub pps_weighted_pred_flag: bool,
    pub pps_weighted_bipred_flag: bool,
    pub pps_ref_wraparound_enabled_flag: bool,
    pub pps_pic_width_minus_wraparound_offset: u32,
    pub pps_init_qp_minus26: i32,
    pub pps_cu_qp_delta_enabled_flag: bool,
    pub pps_chroma_tool_offsets_present_flag: bool,
    pub pps_cb_qp_offset: i32,
    pub pps_cr_qp_offset: i32,
    pub pps_joint_cbcr_qp_offset_present_flag: bool,
    pub pps_joint_cbcr_qp_offset_value: i32,
    pub pps_slice_chroma_qp_offsets_present_flag: bool,
    pub pps_cu_chroma_qp_offset_list_enabled_flag: bool,
    pub pps_deblocking_filter_control_present_flag: bool,
    pub pps_deblocking_filter_override_enabled_flag: bool,
    pub pps_deblocking_filter_disabled_flag: bool,
    pub pps_dbf_info_in_ph_flag: bool,
    /// Present only when `pps_no_pic_partition_flag == 0`; inferred to
    /// `true` otherwise (§7.4.3.5 / §7.3.2.5: the "in PH" flags default
    /// to 1 when not signalled, forcing the slice header to skip the
    /// corresponding per-slice fields).
    pub pps_rpl_info_in_ph_flag: bool,
    pub pps_sao_info_in_ph_flag: bool,
    pub pps_alf_info_in_ph_flag: bool,
    pub pps_wp_info_in_ph_flag: bool,
    pub pps_qp_delta_info_in_ph_flag: bool,
    pub pps_picture_header_extension_present_flag: bool,
    pub pps_slice_header_extension_present_flag: bool,
    pub pps_extension_flag: bool,
}

/// Parse a PPS NAL RBSP payload (the bytes after the 2-byte NAL header,
/// already stripped of emulation-prevention bytes).
///
/// Foundation build: streams signalling
/// `pps_no_pic_partition_flag == 0` or
/// `pps_subpic_id_mapping_present_flag == 1` are rejected with
/// `Error::Unsupported` because walking the partition / subpic
/// mapping sub-syntax requires derived state that the scaffold does
/// not carry.
pub fn parse_pps(rbsp: &[u8]) -> Result<PicParameterSet> {
    let mut br = BitReader::new(rbsp);
    let pps_pic_parameter_set_id = br.u(6)? as u8;
    let pps_seq_parameter_set_id = br.u(4)? as u8;
    let pps_mixed_nalu_types_in_pic_flag = br.u1()? == 1;
    let pps_pic_width_in_luma_samples = br.ue()?;
    let pps_pic_height_in_luma_samples = br.ue()?;
    if pps_pic_width_in_luma_samples == 0
        || pps_pic_height_in_luma_samples == 0
        || pps_pic_width_in_luma_samples > 16384
        || pps_pic_height_in_luma_samples > 16384
    {
        return Err(Error::invalid(format!(
            "h266 PPS: implausible picture size {pps_pic_width_in_luma_samples}x{pps_pic_height_in_luma_samples}"
        )));
    }
    let pps_conformance_window_flag = br.u1()? == 1;
    let conformance_window = if pps_conformance_window_flag {
        Some(ConformanceWindow {
            left_offset: br.ue()?,
            right_offset: br.ue()?,
            top_offset: br.ue()?,
            bottom_offset: br.ue()?,
        })
    } else {
        None
    };
    let pps_scaling_window_explicit_signalling_flag = br.u1()? == 1;
    let scaling_window = if pps_scaling_window_explicit_signalling_flag {
        Some(ScalingWindow {
            left_offset: br.se()?,
            right_offset: br.se()?,
            top_offset: br.se()?,
            bottom_offset: br.se()?,
        })
    } else {
        None
    };
    let pps_output_flag_present_flag = br.u1()? == 1;
    let pps_no_pic_partition_flag = br.u1()? == 1;
    let pps_subpic_id_mapping_present_flag = br.u1()? == 1;
    if pps_subpic_id_mapping_present_flag {
        return Err(Error::unsupported(
            "h266 PPS: pps_subpic_id_mapping_present_flag = 1 (subpicture streams not yet supported)",
        ));
    }
    if !pps_no_pic_partition_flag {
        return Err(Error::unsupported(
            "h266 PPS: per-picture tile / slice partitioning not yet supported (pps_no_pic_partition_flag = 0)",
        ));
    }
    // With `pps_no_pic_partition_flag = 1`, the partition/slice block is
    // entirely skipped. The "…in_ph_flag" switches (rpl / sao / alf / wp
    // / qp_delta / dbf) are only transmitted under the partition block;
    // when that block is skipped §7.4.3.5 infers them to 1, meaning the
    // slice header defers all the corresponding fields to the PH.
    let pps_rect_slice_flag = true; // inferred
    let pps_single_slice_per_subpic_flag = true; // inferred
    let pps_loop_filter_across_slices_enabled_flag = false; // irrelevant w/ single slice

    let pps_cabac_init_present_flag = br.u1()? == 1;
    let pps_num_ref_idx_default_active_minus1 = [br.ue()?, br.ue()?];
    let pps_rpl1_idx_present_flag = br.u1()? == 1;
    let pps_weighted_pred_flag = br.u1()? == 1;
    let pps_weighted_bipred_flag = br.u1()? == 1;
    let pps_ref_wraparound_enabled_flag = br.u1()? == 1;
    let pps_pic_width_minus_wraparound_offset = if pps_ref_wraparound_enabled_flag {
        br.ue()?
    } else {
        0
    };
    let pps_init_qp_minus26 = br.se()?;
    let pps_cu_qp_delta_enabled_flag = br.u1()? == 1;
    let pps_chroma_tool_offsets_present_flag = br.u1()? == 1;
    let mut pps_cb_qp_offset: i32 = 0;
    let mut pps_cr_qp_offset: i32 = 0;
    let mut pps_joint_cbcr_qp_offset_present_flag = false;
    let mut pps_joint_cbcr_qp_offset_value: i32 = 0;
    let mut pps_slice_chroma_qp_offsets_present_flag = false;
    let mut pps_cu_chroma_qp_offset_list_enabled_flag = false;
    if pps_chroma_tool_offsets_present_flag {
        pps_cb_qp_offset = br.se()?;
        pps_cr_qp_offset = br.se()?;
        pps_joint_cbcr_qp_offset_present_flag = br.u1()? == 1;
        if pps_joint_cbcr_qp_offset_present_flag {
            pps_joint_cbcr_qp_offset_value = br.se()?;
        }
        pps_slice_chroma_qp_offsets_present_flag = br.u1()? == 1;
        pps_cu_chroma_qp_offset_list_enabled_flag = br.u1()? == 1;
        if pps_cu_chroma_qp_offset_list_enabled_flag {
            let len_minus1 = br.ue()?;
            // Walk the pairs / triples — we don't retain them in this
            // pass (only the gate matters for slice-header decoding).
            if len_minus1 > 64 {
                return Err(Error::invalid(format!(
                    "h266 PPS: pps_chroma_qp_offset_list_len_minus1 out of range ({len_minus1})"
                )));
            }
            for _ in 0..=len_minus1 {
                let _ = br.se()?; // pps_cb_qp_offset_list[i]
                let _ = br.se()?; // pps_cr_qp_offset_list[i]
                if pps_joint_cbcr_qp_offset_present_flag {
                    let _ = br.se()?; // pps_joint_cbcr_qp_offset_list[i]
                }
            }
        }
    }
    let pps_deblocking_filter_control_present_flag = br.u1()? == 1;
    let mut pps_deblocking_filter_override_enabled_flag = false;
    let mut pps_deblocking_filter_disabled_flag = false;
    let pps_dbf_info_in_ph_flag = true; // inferred when not present
    if pps_deblocking_filter_control_present_flag {
        pps_deblocking_filter_override_enabled_flag = br.u1()? == 1;
        pps_deblocking_filter_disabled_flag = br.u1()? == 1;
        // pps_dbf_info_in_ph_flag is only transmitted when
        // `!pps_no_pic_partition_flag && pps_deblocking_filter_override_enabled_flag`
        // — both false with pps_no_pic_partition_flag = 1, so this flag
        // stays at its inferred value (1).
        if !pps_deblocking_filter_disabled_flag {
            let _ = br.se()?; // pps_luma_beta_offset_div2
            let _ = br.se()?; // pps_luma_tc_offset_div2
            if pps_chroma_tool_offsets_present_flag {
                let _ = br.se()?;
                let _ = br.se()?;
                let _ = br.se()?;
                let _ = br.se()?;
            }
        }
    }
    // `if (!pps_no_pic_partition_flag)` block is skipped — the four
    // info_in_ph_flag members are inferred:
    //    pps_rpl_info_in_ph_flag = 1, pps_sao_info_in_ph_flag = 1,
    //    pps_alf_info_in_ph_flag = 1, pps_wp_info_in_ph_flag = 1,
    //    pps_qp_delta_info_in_ph_flag = 1.
    let pps_rpl_info_in_ph_flag = true;
    let pps_sao_info_in_ph_flag = true;
    let pps_alf_info_in_ph_flag = true;
    let pps_wp_info_in_ph_flag = true;
    let pps_qp_delta_info_in_ph_flag = true;

    let pps_picture_header_extension_present_flag = br.u1()? == 1;
    let pps_slice_header_extension_present_flag = br.u1()? == 1;
    let pps_extension_flag = br.u1()? == 1;
    if pps_extension_flag {
        // Consume the extension-data bits until the stop bit (§7.3.2.5).
        while br.has_more_rbsp_data() {
            br.u1()?;
        }
    }

    Ok(PicParameterSet {
        pps_pic_parameter_set_id,
        pps_seq_parameter_set_id,
        pps_mixed_nalu_types_in_pic_flag,
        pps_pic_width_in_luma_samples,
        pps_pic_height_in_luma_samples,
        conformance_window,
        scaling_window,
        pps_output_flag_present_flag,
        pps_no_pic_partition_flag,
        pps_subpic_id_mapping_present_flag,
        pps_rect_slice_flag,
        pps_single_slice_per_subpic_flag,
        pps_loop_filter_across_slices_enabled_flag,
        pps_cabac_init_present_flag,
        pps_num_ref_idx_default_active_minus1,
        pps_rpl1_idx_present_flag,
        pps_weighted_pred_flag,
        pps_weighted_bipred_flag,
        pps_ref_wraparound_enabled_flag,
        pps_pic_width_minus_wraparound_offset,
        pps_init_qp_minus26,
        pps_cu_qp_delta_enabled_flag,
        pps_chroma_tool_offsets_present_flag,
        pps_cb_qp_offset,
        pps_cr_qp_offset,
        pps_joint_cbcr_qp_offset_present_flag,
        pps_joint_cbcr_qp_offset_value,
        pps_slice_chroma_qp_offsets_present_flag,
        pps_cu_chroma_qp_offset_list_enabled_flag,
        pps_deblocking_filter_control_present_flag,
        pps_deblocking_filter_override_enabled_flag,
        pps_deblocking_filter_disabled_flag,
        pps_dbf_info_in_ph_flag,
        pps_rpl_info_in_ph_flag,
        pps_sao_info_in_ph_flag,
        pps_alf_info_in_ph_flag,
        pps_wp_info_in_ph_flag,
        pps_qp_delta_info_in_ph_flag,
        pps_picture_header_extension_present_flag,
        pps_slice_header_extension_present_flag,
        pps_extension_flag,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn push_u(bits: &mut Vec<u8>, v: u64, n: u32) {
        for i in (0..n).rev() {
            bits.push(((v >> i) & 1) as u8);
        }
    }

    fn pack_bits(bits: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity((bits.len() + 7) / 8);
        let mut cur = 0u8;
        for (i, &bit) in bits.iter().enumerate() {
            cur |= bit << (7 - (i % 8));
            if i % 8 == 7 {
                out.push(cur);
                cur = 0;
            }
        }
        if bits.len() % 8 != 0 {
            out.push(cur);
        }
        out
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

    /// Minimal single-slice 320x240 PPS (no partitioning, no subpic map).
    #[test]
    fn minimal_pps_roundtrip() {
        let mut bits: Vec<u8> = Vec::new();
        push_u(&mut bits, 0, 6); // pps_id
        push_u(&mut bits, 0, 4); // sps_id
        push_u(&mut bits, 0, 1); // mixed_nalu_types
        push_ue(&mut bits, 320);
        push_ue(&mut bits, 240);
        push_u(&mut bits, 0, 1); // conformance_window_flag
        push_u(&mut bits, 0, 1); // scaling_window
        push_u(&mut bits, 0, 1); // output_flag_present
        push_u(&mut bits, 1, 1); // no_pic_partition = 1 (foundation path)
        push_u(&mut bits, 0, 1); // subpic_id_mapping_present = 0
                                 // Cabac + ref-idx defaults + rpl1 flag.
        push_u(&mut bits, 0, 1); // cabac_init_present
        push_ue(&mut bits, 0); // num_ref_idx_default_active_minus1[0]
        push_ue(&mut bits, 0); // num_ref_idx_default_active_minus1[1]
        push_u(&mut bits, 0, 1); // rpl1_idx_present
        push_u(&mut bits, 0, 1); // weighted_pred
        push_u(&mut bits, 0, 1); // weighted_bipred
        push_u(&mut bits, 0, 1); // ref_wraparound
        push_se(&mut bits, 0); // init_qp_minus26
        push_u(&mut bits, 0, 1); // cu_qp_delta_enabled
        push_u(&mut bits, 0, 1); // chroma_tool_offsets_present
        push_u(&mut bits, 0, 1); // deblocking_filter_control_present
                                 // partition block skipped. Extension-flag tail.
        push_u(&mut bits, 0, 1); // picture_header_ext_present
        push_u(&mut bits, 0, 1); // slice_header_ext_present
        push_u(&mut bits, 0, 1); // pps_extension_flag
        let bytes = pack_bits(&bits);
        let pps = parse_pps(&bytes).unwrap();
        assert_eq!(pps.pps_pic_parameter_set_id, 0);
        assert_eq!(pps.pps_seq_parameter_set_id, 0);
        assert_eq!(pps.pps_pic_width_in_luma_samples, 320);
        assert_eq!(pps.pps_pic_height_in_luma_samples, 240);
        assert!(pps.pps_no_pic_partition_flag);
        assert!(!pps.pps_subpic_id_mapping_present_flag);
        // The info_in_ph flags must be inferred to true when the
        // partition block was skipped.
        assert!(pps.pps_rpl_info_in_ph_flag);
        assert!(pps.pps_sao_info_in_ph_flag);
        assert!(pps.pps_alf_info_in_ph_flag);
        assert!(pps.pps_qp_delta_info_in_ph_flag);
        assert_eq!(pps.pps_init_qp_minus26, 0);
    }

    #[test]
    fn partitioned_pps_is_rejected() {
        let mut bits: Vec<u8> = Vec::new();
        push_u(&mut bits, 0, 6); // pps_id
        push_u(&mut bits, 0, 4); // sps_id
        push_u(&mut bits, 0, 1); // mixed_nalu_types
        push_u(&mut bits, 0b00000000010100_0001, 17); // ue(320)
        push_u(&mut bits, 0b000000011110001, 15); // ue(240)
        push_u(&mut bits, 0, 1); // conformance_window_flag
        push_u(&mut bits, 0, 1); // scaling_window
        push_u(&mut bits, 0, 1); // output_flag_present
        push_u(&mut bits, 0, 1); // no_pic_partition = 0 → partitioning
        push_u(&mut bits, 0, 1); // subpic_id_mapping_present
        let bytes = pack_bits(&bits);
        assert!(parse_pps(&bytes).is_err());
    }
}
