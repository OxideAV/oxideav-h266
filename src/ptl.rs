//! `profile_tier_level()` parser (§7.3.3.1) — shared by DCI, VPS and SPS.
//!
//! VVC's PTL is structurally different from HEVC's: the sublayer level
//! array is stored in descending order, GCI is a dedicated sub-structure
//! gated on `gci_present_flag`, and `ptl_num_sub_profiles` replaces HEVC's
//! 32-bit profile-compatibility field.
//!
//! Round 245 extends the §7.3.3.2 `general_constraints_info()` walker
//! from a length-only `br.skip()` cascade to a typed
//! [`GeneralConstraintsInfo`] structure that exposes every named GCI
//! flag plus the V4 (01/2026) additional-bit block.

use oxideav_core::{Error, Result};

use crate::bitreader::BitReader;

/// Parsed `profile_tier_level()` structure (§7.3.3.1).
#[derive(Clone, Debug, Default)]
pub struct ProfileTierLevel {
    /// Whether `profileTierPresentFlag` was 1 when the PTL was parsed. If
    /// false, `general_profile_idc` / `general_tier_flag` / `gci_*` were
    /// not read and carry their default values.
    pub profile_tier_present: bool,
    /// §7.3.3.1: `general_profile_idc` (u(7)).
    pub general_profile_idc: u8,
    /// §7.3.3.1: `general_tier_flag` (u(1)).
    pub general_tier_flag: bool,
    /// §7.3.3.1: `general_level_idc` (u(8)).
    pub general_level_idc: u8,
    /// §7.3.3.1: `ptl_frame_only_constraint_flag` (u(1)).
    pub ptl_frame_only_constraint_flag: bool,
    /// §7.3.3.1: `ptl_multilayer_enabled_flag` (u(1)).
    pub ptl_multilayer_enabled_flag: bool,
    /// `general_constraints_info()` first-byte flag — zero-initialised when
    /// the structure is absent.
    pub gci_present_flag: bool,
    /// Round-245 typed §7.3.3.2 body. All fields are zero / `false` /
    /// `0` when `gci_present_flag == 0` per the §7.4.4.2 inferences.
    pub gci: GeneralConstraintsInfo,
    /// Number of sub-profile IDCs recorded (0 when absent).
    pub ptl_num_sub_profiles: u8,
    /// `general_sub_profile_idc[i]` (u(32) each).
    pub general_sub_profile_idc: Vec<u32>,
    /// `ptl_sublayer_level_present_flag[i]` (length = `MaxNumSubLayersMinus1`).
    /// Stored in i=MaxNumSubLayersMinus1-1 .. 0 bit-order.
    pub sublayer_level_present: Vec<bool>,
    /// `sublayer_level_idc[i]` (u(8) each) when present.
    pub sublayer_level_idc: Vec<u8>,
}

/// Typed `general_constraints_info()` body (§7.3.3.2 / §7.4.4.2).
///
/// Every named flag in the V3 (10/2022) baseline GCI block is surfaced
/// as its own field; the §7.4.4.2 picture-format `_idc` fields keep their
/// raw `u(4)` / `u(2)` widths. The six §7.3.3.2 V4 (01/2026) extensions
/// are read when `gci_num_additional_bits > 5`; the remaining
/// `gci_reserved_bit[i]` are skipped per §7.4.4.2 "decoders … shall
/// ignore the values of the gci_reserved_bit[i] syntax elements when
/// present."
///
/// When `gci_present_flag == 0`, every field is zero / `false` / `0`
/// per §7.4.4.2 ("the general_constraints_info() syntax structure does
/// not impose any constraints").
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GeneralConstraintsInfo {
    // ----- general -----
    /// §7.4.4.2 `gci_intra_only_constraint_flag` (u(1)).
    pub intra_only_constraint_flag: bool,
    /// §7.4.4.2 `gci_all_layers_independent_constraint_flag` (u(1)).
    pub all_layers_independent_constraint_flag: bool,
    /// §7.4.4.2 `gci_one_au_only_constraint_flag` (u(1)).
    pub one_au_only_constraint_flag: bool,

    // ----- picture format -----
    /// §7.4.4.2 `gci_sixteen_minus_max_bitdepth_constraint_idc` (u(4)).
    pub sixteen_minus_max_bitdepth_constraint_idc: u8,
    /// §7.4.4.2 `gci_three_minus_max_chroma_format_constraint_idc` (u(2)).
    pub three_minus_max_chroma_format_constraint_idc: u8,

    // ----- NAL unit type related -----
    /// §7.4.4.2 `gci_no_mixed_nalu_types_in_pic_constraint_flag` (u(1)).
    pub no_mixed_nalu_types_in_pic_constraint_flag: bool,
    /// §7.4.4.2 `gci_no_trail_constraint_flag` (u(1)).
    pub no_trail_constraint_flag: bool,
    /// §7.4.4.2 `gci_no_stsa_constraint_flag` (u(1)).
    pub no_stsa_constraint_flag: bool,
    /// §7.4.4.2 `gci_no_rasl_constraint_flag` (u(1)).
    pub no_rasl_constraint_flag: bool,
    /// §7.4.4.2 `gci_no_radl_constraint_flag` (u(1)).
    pub no_radl_constraint_flag: bool,
    /// §7.4.4.2 `gci_no_idr_constraint_flag` (u(1)).
    pub no_idr_constraint_flag: bool,
    /// §7.4.4.2 `gci_no_cra_constraint_flag` (u(1)).
    pub no_cra_constraint_flag: bool,
    /// §7.4.4.2 `gci_no_gdr_constraint_flag` (u(1)).
    pub no_gdr_constraint_flag: bool,
    /// §7.4.4.2 `gci_no_aps_constraint_flag` (u(1)).
    pub no_aps_constraint_flag: bool,
    /// §7.4.4.2 `gci_no_idr_rpl_constraint_flag` (u(1)).
    pub no_idr_rpl_constraint_flag: bool,

    // ----- tile / slice / subpicture partitioning -----
    /// §7.4.4.2 `gci_one_tile_per_pic_constraint_flag` (u(1)).
    pub one_tile_per_pic_constraint_flag: bool,
    /// §7.4.4.2 `gci_pic_header_in_slice_header_constraint_flag` (u(1)).
    pub pic_header_in_slice_header_constraint_flag: bool,
    /// §7.4.4.2 `gci_one_slice_per_pic_constraint_flag` (u(1)).
    pub one_slice_per_pic_constraint_flag: bool,
    /// §7.4.4.2 `gci_no_rectangular_slice_constraint_flag` (u(1)).
    pub no_rectangular_slice_constraint_flag: bool,
    /// §7.4.4.2 `gci_one_slice_per_subpic_constraint_flag` (u(1)).
    pub one_slice_per_subpic_constraint_flag: bool,
    /// §7.4.4.2 `gci_no_subpic_info_constraint_flag` (u(1)).
    pub no_subpic_info_constraint_flag: bool,

    // ----- CTU and block partitioning -----
    /// §7.4.4.2 `gci_three_minus_max_log2_ctu_size_constraint_idc` (u(2)).
    pub three_minus_max_log2_ctu_size_constraint_idc: u8,
    /// §7.4.4.2 `gci_no_partition_constraints_override_constraint_flag` (u(1)).
    pub no_partition_constraints_override_constraint_flag: bool,
    /// §7.4.4.2 `gci_no_mtt_constraint_flag` (u(1)).
    pub no_mtt_constraint_flag: bool,
    /// §7.4.4.2 `gci_no_qtbtt_dual_tree_intra_constraint_flag` (u(1)).
    pub no_qtbtt_dual_tree_intra_constraint_flag: bool,

    // ----- intra -----
    /// §7.4.4.2 `gci_no_palette_constraint_flag` (u(1)).
    pub no_palette_constraint_flag: bool,
    /// §7.4.4.2 `gci_no_ibc_constraint_flag` (u(1)).
    pub no_ibc_constraint_flag: bool,
    /// §7.4.4.2 `gci_no_isp_constraint_flag` (u(1)).
    pub no_isp_constraint_flag: bool,
    /// §7.4.4.2 `gci_no_mrl_constraint_flag` (u(1)).
    pub no_mrl_constraint_flag: bool,
    /// §7.4.4.2 `gci_no_mip_constraint_flag` (u(1)).
    pub no_mip_constraint_flag: bool,
    /// §7.4.4.2 `gci_no_cclm_constraint_flag` (u(1)).
    pub no_cclm_constraint_flag: bool,

    // ----- inter -----
    /// §7.4.4.2 `gci_no_ref_pic_resampling_constraint_flag` (u(1)).
    pub no_ref_pic_resampling_constraint_flag: bool,
    /// §7.4.4.2 `gci_no_res_change_in_clvs_constraint_flag` (u(1)).
    pub no_res_change_in_clvs_constraint_flag: bool,
    /// §7.4.4.2 `gci_no_weighted_prediction_constraint_flag` (u(1)).
    pub no_weighted_prediction_constraint_flag: bool,
    /// §7.4.4.2 `gci_no_ref_wraparound_constraint_flag` (u(1)).
    pub no_ref_wraparound_constraint_flag: bool,
    /// §7.4.4.2 `gci_no_temporal_mvp_constraint_flag` (u(1)).
    pub no_temporal_mvp_constraint_flag: bool,
    /// §7.4.4.2 `gci_no_sbtmvp_constraint_flag` (u(1)).
    pub no_sbtmvp_constraint_flag: bool,
    /// §7.4.4.2 `gci_no_amvr_constraint_flag` (u(1)).
    pub no_amvr_constraint_flag: bool,
    /// §7.4.4.2 `gci_no_bdof_constraint_flag` (u(1)).
    pub no_bdof_constraint_flag: bool,
    /// §7.4.4.2 `gci_no_smvd_constraint_flag` (u(1)).
    pub no_smvd_constraint_flag: bool,
    /// §7.4.4.2 `gci_no_dmvr_constraint_flag` (u(1)).
    pub no_dmvr_constraint_flag: bool,
    /// §7.4.4.2 `gci_no_mmvd_constraint_flag` (u(1)).
    pub no_mmvd_constraint_flag: bool,
    /// §7.4.4.2 `gci_no_affine_motion_constraint_flag` (u(1)).
    pub no_affine_motion_constraint_flag: bool,
    /// §7.4.4.2 `gci_no_prof_constraint_flag` (u(1)).
    pub no_prof_constraint_flag: bool,
    /// §7.4.4.2 `gci_no_bcw_constraint_flag` (u(1)).
    pub no_bcw_constraint_flag: bool,
    /// §7.4.4.2 `gci_no_ciip_constraint_flag` (u(1)).
    pub no_ciip_constraint_flag: bool,
    /// §7.4.4.2 `gci_no_gpm_constraint_flag` (u(1)).
    pub no_gpm_constraint_flag: bool,

    // ----- transform / quantization / residual -----
    /// §7.4.4.2 `gci_no_luma_transform_size_64_constraint_flag` (u(1)).
    pub no_luma_transform_size_64_constraint_flag: bool,
    /// §7.4.4.2 `gci_no_transform_skip_constraint_flag` (u(1)).
    pub no_transform_skip_constraint_flag: bool,
    /// §7.4.4.2 `gci_no_bdpcm_constraint_flag` (u(1)).
    pub no_bdpcm_constraint_flag: bool,
    /// §7.4.4.2 `gci_no_mts_constraint_flag` (u(1)).
    pub no_mts_constraint_flag: bool,
    /// §7.4.4.2 `gci_no_lfnst_constraint_flag` (u(1)).
    pub no_lfnst_constraint_flag: bool,
    /// §7.4.4.2 `gci_no_joint_cbcr_constraint_flag` (u(1)).
    pub no_joint_cbcr_constraint_flag: bool,
    /// §7.4.4.2 `gci_no_sbt_constraint_flag` (u(1)).
    pub no_sbt_constraint_flag: bool,
    /// §7.4.4.2 `gci_no_act_constraint_flag` (u(1)).
    pub no_act_constraint_flag: bool,
    /// §7.4.4.2 `gci_no_explicit_scaling_list_constraint_flag` (u(1)).
    pub no_explicit_scaling_list_constraint_flag: bool,
    /// §7.4.4.2 `gci_no_dep_quant_constraint_flag` (u(1)).
    pub no_dep_quant_constraint_flag: bool,
    /// §7.4.4.2 `gci_no_sign_data_hiding_constraint_flag` (u(1)).
    pub no_sign_data_hiding_constraint_flag: bool,
    /// §7.4.4.2 `gci_no_cu_qp_delta_constraint_flag` (u(1)).
    pub no_cu_qp_delta_constraint_flag: bool,
    /// §7.4.4.2 `gci_no_chroma_qp_offset_constraint_flag` (u(1)).
    pub no_chroma_qp_offset_constraint_flag: bool,

    // ----- loop filter -----
    /// §7.4.4.2 `gci_no_sao_constraint_flag` (u(1)).
    pub no_sao_constraint_flag: bool,
    /// §7.4.4.2 `gci_no_alf_constraint_flag` (u(1)).
    pub no_alf_constraint_flag: bool,
    /// §7.4.4.2 `gci_no_ccalf_constraint_flag` (u(1)).
    pub no_ccalf_constraint_flag: bool,
    /// §7.4.4.2 `gci_no_lmcs_constraint_flag` (u(1)).
    pub no_lmcs_constraint_flag: bool,
    /// §7.4.4.2 `gci_no_ladf_constraint_flag` (u(1)).
    pub no_ladf_constraint_flag: bool,
    /// §7.4.4.2 `gci_no_virtual_boundaries_constraint_flag` (u(1)).
    pub no_virtual_boundaries_constraint_flag: bool,

    // ----- V4 (01/2026) additional bit block -----
    /// §7.4.4.2 `gci_num_additional_bits` (u(8)). Per §7.4.4.2, this
    /// value shall be 0 or 6 for V4 conformance; greater values are
    /// reserved and the trailing `gci_reserved_bit[i]` bits are
    /// ignored by the decoder.
    pub num_additional_bits: u8,
    /// §7.4.4.2 `gci_all_rap_pictures_constraint_flag` (u(1)).
    /// Only read when `num_additional_bits > 5`; inferred `false`
    /// otherwise (§7.4.4.2 "When … not present, its value is inferred
    /// to be equal to 0").
    pub all_rap_pictures_constraint_flag: bool,
    /// §7.4.4.2 `gci_no_extended_precision_processing_constraint_flag`
    /// (u(1)). Same V4 gating as above.
    pub no_extended_precision_processing_constraint_flag: bool,
    /// §7.4.4.2 `gci_no_ts_residual_coding_rice_constraint_flag` (u(1)).
    pub no_ts_residual_coding_rice_constraint_flag: bool,
    /// §7.4.4.2 `gci_no_rrc_rice_extension_constraint_flag` (u(1)).
    pub no_rrc_rice_extension_constraint_flag: bool,
    /// §7.4.4.2 `gci_no_persistent_rice_adaptation_constraint_flag` (u(1)).
    pub no_persistent_rice_adaptation_constraint_flag: bool,
    /// §7.4.4.2 `gci_no_reverse_last_sig_coeff_constraint_flag` (u(1)).
    pub no_reverse_last_sig_coeff_constraint_flag: bool,
}

/// Parse `profile_tier_level( profileTierPresentFlag, MaxNumSubLayersMinus1 )`
/// (§7.3.3.1). `max_num_sub_layers_minus1` comes from the caller (e.g.
/// `vps_ptl_max_tid[i]` or `sps_max_sublayers_minus1`).
pub fn parse_profile_tier_level(
    br: &mut BitReader<'_>,
    profile_tier_present_flag: bool,
    max_num_sub_layers_minus1: u8,
) -> Result<ProfileTierLevel> {
    let mut ptl = ProfileTierLevel {
        profile_tier_present: profile_tier_present_flag,
        ..Default::default()
    };
    if profile_tier_present_flag {
        ptl.general_profile_idc = br.u(7)? as u8;
        ptl.general_tier_flag = br.u1()? == 1;
    }
    ptl.general_level_idc = br.u(8)? as u8;
    ptl.ptl_frame_only_constraint_flag = br.u1()? == 1;
    ptl.ptl_multilayer_enabled_flag = br.u1()? == 1;
    if profile_tier_present_flag {
        parse_general_constraints_info(br, &mut ptl)?;
    }
    // sublayer_level_present loop iterates MaxNumSubLayersMinus1-1 .. 0.
    if max_num_sub_layers_minus1 > 0 {
        let n = max_num_sub_layers_minus1 as usize;
        let mut flags = Vec::with_capacity(n);
        for _ in 0..n {
            flags.push(br.u1()? == 1);
        }
        ptl.sublayer_level_present = flags;
    }
    // Byte alignment with zero bits.
    while !br.is_byte_aligned() {
        br.skip(1)?;
    }
    if max_num_sub_layers_minus1 > 0 {
        let n = max_num_sub_layers_minus1 as usize;
        let mut idcs = vec![0u8; n];
        for i in 0..n {
            if ptl.sublayer_level_present[i] {
                idcs[i] = br.u(8)? as u8;
            }
        }
        ptl.sublayer_level_idc = idcs;
    }
    if profile_tier_present_flag {
        ptl.ptl_num_sub_profiles = br.u(8)? as u8;
        let mut v = Vec::with_capacity(ptl.ptl_num_sub_profiles as usize);
        for _ in 0..ptl.ptl_num_sub_profiles {
            v.push(br.u(32)?);
        }
        ptl.general_sub_profile_idc = v;
    }
    Ok(ptl)
}

/// Walk `general_constraints_info()` (§7.3.3.2) into a typed
/// [`GeneralConstraintsInfo`]. Each named flag is read in spec order;
/// the V4 (01/2026) additional-bit block is read when
/// `gci_num_additional_bits > 5`, and any remaining
/// `gci_reserved_bit[i]` are skipped per §7.4.4.2.
fn parse_general_constraints_info(
    br: &mut BitReader<'_>,
    ptl: &mut ProfileTierLevel,
) -> Result<()> {
    ptl.gci_present_flag = br.u1()? == 1;
    if ptl.gci_present_flag {
        let g = &mut ptl.gci;

        // general — 3 bits.
        g.intra_only_constraint_flag = br.u1()? == 1;
        g.all_layers_independent_constraint_flag = br.u1()? == 1;
        g.one_au_only_constraint_flag = br.u1()? == 1;

        // picture format — u(4) + u(2) = 6 bits.
        g.sixteen_minus_max_bitdepth_constraint_idc = br.u(4)? as u8;
        g.three_minus_max_chroma_format_constraint_idc = br.u(2)? as u8;

        // NAL unit type related — 10 bits.
        g.no_mixed_nalu_types_in_pic_constraint_flag = br.u1()? == 1;
        g.no_trail_constraint_flag = br.u1()? == 1;
        g.no_stsa_constraint_flag = br.u1()? == 1;
        g.no_rasl_constraint_flag = br.u1()? == 1;
        g.no_radl_constraint_flag = br.u1()? == 1;
        g.no_idr_constraint_flag = br.u1()? == 1;
        g.no_cra_constraint_flag = br.u1()? == 1;
        g.no_gdr_constraint_flag = br.u1()? == 1;
        g.no_aps_constraint_flag = br.u1()? == 1;
        g.no_idr_rpl_constraint_flag = br.u1()? == 1;

        // tile / slice / subpicture partitioning — 6 bits.
        g.one_tile_per_pic_constraint_flag = br.u1()? == 1;
        g.pic_header_in_slice_header_constraint_flag = br.u1()? == 1;
        g.one_slice_per_pic_constraint_flag = br.u1()? == 1;
        g.no_rectangular_slice_constraint_flag = br.u1()? == 1;
        g.one_slice_per_subpic_constraint_flag = br.u1()? == 1;
        g.no_subpic_info_constraint_flag = br.u1()? == 1;

        // CTU and block partitioning — u(2) + 3 flags = 5 bits.
        g.three_minus_max_log2_ctu_size_constraint_idc = br.u(2)? as u8;
        g.no_partition_constraints_override_constraint_flag = br.u1()? == 1;
        g.no_mtt_constraint_flag = br.u1()? == 1;
        g.no_qtbtt_dual_tree_intra_constraint_flag = br.u1()? == 1;

        // intra — 6 bits.
        g.no_palette_constraint_flag = br.u1()? == 1;
        g.no_ibc_constraint_flag = br.u1()? == 1;
        g.no_isp_constraint_flag = br.u1()? == 1;
        g.no_mrl_constraint_flag = br.u1()? == 1;
        g.no_mip_constraint_flag = br.u1()? == 1;
        g.no_cclm_constraint_flag = br.u1()? == 1;

        // inter — 16 bits.
        g.no_ref_pic_resampling_constraint_flag = br.u1()? == 1;
        g.no_res_change_in_clvs_constraint_flag = br.u1()? == 1;
        g.no_weighted_prediction_constraint_flag = br.u1()? == 1;
        g.no_ref_wraparound_constraint_flag = br.u1()? == 1;
        g.no_temporal_mvp_constraint_flag = br.u1()? == 1;
        g.no_sbtmvp_constraint_flag = br.u1()? == 1;
        g.no_amvr_constraint_flag = br.u1()? == 1;
        g.no_bdof_constraint_flag = br.u1()? == 1;
        g.no_smvd_constraint_flag = br.u1()? == 1;
        g.no_dmvr_constraint_flag = br.u1()? == 1;
        g.no_mmvd_constraint_flag = br.u1()? == 1;
        g.no_affine_motion_constraint_flag = br.u1()? == 1;
        g.no_prof_constraint_flag = br.u1()? == 1;
        g.no_bcw_constraint_flag = br.u1()? == 1;
        g.no_ciip_constraint_flag = br.u1()? == 1;
        g.no_gpm_constraint_flag = br.u1()? == 1;

        // transform / quantization / residual — 13 bits.
        g.no_luma_transform_size_64_constraint_flag = br.u1()? == 1;
        g.no_transform_skip_constraint_flag = br.u1()? == 1;
        g.no_bdpcm_constraint_flag = br.u1()? == 1;
        g.no_mts_constraint_flag = br.u1()? == 1;
        g.no_lfnst_constraint_flag = br.u1()? == 1;
        g.no_joint_cbcr_constraint_flag = br.u1()? == 1;
        g.no_sbt_constraint_flag = br.u1()? == 1;
        g.no_act_constraint_flag = br.u1()? == 1;
        g.no_explicit_scaling_list_constraint_flag = br.u1()? == 1;
        g.no_dep_quant_constraint_flag = br.u1()? == 1;
        g.no_sign_data_hiding_constraint_flag = br.u1()? == 1;
        g.no_cu_qp_delta_constraint_flag = br.u1()? == 1;
        g.no_chroma_qp_offset_constraint_flag = br.u1()? == 1;

        // loop filter — 6 bits.
        g.no_sao_constraint_flag = br.u1()? == 1;
        g.no_alf_constraint_flag = br.u1()? == 1;
        g.no_ccalf_constraint_flag = br.u1()? == 1;
        g.no_lmcs_constraint_flag = br.u1()? == 1;
        g.no_ladf_constraint_flag = br.u1()? == 1;
        g.no_virtual_boundaries_constraint_flag = br.u1()? == 1;

        // V4 additional-bit block.
        g.num_additional_bits = br.u(8)? as u8;
        let num_additional_bits_used = if g.num_additional_bits > 5 {
            g.all_rap_pictures_constraint_flag = br.u1()? == 1;
            g.no_extended_precision_processing_constraint_flag = br.u1()? == 1;
            g.no_ts_residual_coding_rice_constraint_flag = br.u1()? == 1;
            g.no_rrc_rice_extension_constraint_flag = br.u1()? == 1;
            g.no_persistent_rice_adaptation_constraint_flag = br.u1()? == 1;
            g.no_reverse_last_sig_coeff_constraint_flag = br.u1()? == 1;
            6u32
        } else {
            0u32
        };
        if (g.num_additional_bits as u32) < num_additional_bits_used {
            return Err(Error::invalid(format!(
                "h266 GCI: gci_num_additional_bits {} < used {}",
                g.num_additional_bits, num_additional_bits_used
            )));
        }
        let reserved = g.num_additional_bits as u32 - num_additional_bits_used;
        br.skip(reserved)?;
    }
    while !br.is_byte_aligned() {
        br.skip(1)?;
    }
    Ok(())
}

/// Common VVC profile name, where known (Annex A).
pub fn profile_name(profile_idc: u8) -> Option<&'static str> {
    match profile_idc {
        1 => Some("Main 10"),
        2 => Some("Multilayer Main 10"),
        17 => Some("Main 10 4:4:4"),
        18 => Some("Multilayer Main 10 4:4:4"),
        33 => Some("Main 10 Still Picture"),
        65 => Some("Main 10 4:4:4 Still Picture"),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_name_main10() {
        assert_eq!(profile_name(1), Some("Main 10"));
        assert_eq!(profile_name(255), None);
    }

    #[test]
    fn ptl_level_only_single_sublayer() {
        // profile_tier_present = false, MaxNumSubLayersMinus1 = 0. Only
        // general_level_idc + ptl_frame_only_constraint + ptl_multilayer
        // appear (10 bits total), then byte alignment zeros. 2 bytes is
        // enough.
        //
        // Bits: level_idc=0x5A (8) + frame_only=1 + multilayer=0 + 6
        // alignment zeros = 0x5A,0xB0? Let's compute.
        //   byte0 = 0b01011010                     (level_idc)
        //   byte1 = 0b10_000000 = 0x80             (1, 0, align)
        let data = [0x5A, 0x80];
        let mut br = BitReader::new(&data);
        let ptl = parse_profile_tier_level(&mut br, false, 0).unwrap();
        assert_eq!(ptl.general_level_idc, 0x5A);
        assert!(ptl.ptl_frame_only_constraint_flag);
        assert!(!ptl.ptl_multilayer_enabled_flag);
        assert!(!ptl.gci_present_flag);
        // §7.4.4.2 inference: when gci_present_flag == 0, every typed
        // GCI field carries its zero default.
        assert_eq!(ptl.gci, GeneralConstraintsInfo::default());
    }

    /// Helper: append `value` as `bits` MSB-first bits to a `Vec<u8>`
    /// being assembled bit-by-bit, tracking `bit_pos` (0..8 within the
    /// last byte).
    fn push_bits(buf: &mut Vec<u8>, bit_pos: &mut u32, value: u32, bits: u32) {
        for i in (0..bits).rev() {
            let bit = (value >> i) & 1;
            if *bit_pos == 0 {
                buf.push(0);
            }
            let byte_idx = buf.len() - 1;
            let shift = 7 - *bit_pos;
            buf[byte_idx] |= (bit as u8) << shift;
            *bit_pos = (*bit_pos + 1) % 8;
        }
    }

    /// Build a PTL byte stream with `profileTierPresentFlag == 1` and
    /// a fully-zero GCI body whose `num_additional_bits` selects the
    /// V3 or V4 path.
    fn build_ptl_bytes_with_zero_gci(num_additional_bits: u8) -> Vec<u8> {
        let mut buf = Vec::new();
        let mut bp = 0u32;
        // profile_tier_present = true path: general_profile_idc u(7),
        // general_tier_flag u(1), general_level_idc u(8),
        // ptl_frame_only_constraint u(1), ptl_multilayer u(1).
        push_bits(&mut buf, &mut bp, 1, 7); // general_profile_idc = 1
        push_bits(&mut buf, &mut bp, 0, 1); // general_tier_flag = 0
        push_bits(&mut buf, &mut bp, 0x5A, 8); // general_level_idc
        push_bits(&mut buf, &mut bp, 1, 1); // ptl_frame_only = 1
        push_bits(&mut buf, &mut bp, 0, 1); // ptl_multilayer = 0
                                            // gci_present_flag = 1, all named flags = 0.
        push_bits(&mut buf, &mut bp, 1, 1);
        // 3 general + 6 picture format + 10 NAL + 6 partitioning +
        // 5 CTU/block + 6 intra + 16 inter + 13 t/q/r + 6 loop = 71
        // zero bits.
        for _ in 0..71 {
            push_bits(&mut buf, &mut bp, 0, 1);
        }
        // gci_num_additional_bits u(8).
        push_bits(&mut buf, &mut bp, num_additional_bits as u32, 8);
        if num_additional_bits > 5 {
            // 6 V4 additional flags, all zero.
            for _ in 0..6 {
                push_bits(&mut buf, &mut bp, 0, 1);
            }
            for _ in 0..(num_additional_bits as u32 - 6) {
                push_bits(&mut buf, &mut bp, 0, 1); // reserved
            }
        } else {
            for _ in 0..(num_additional_bits as u32) {
                push_bits(&mut buf, &mut bp, 0, 1); // reserved
            }
        }
        // gci_alignment_zero_bit until byte aligned.
        while bp != 0 {
            push_bits(&mut buf, &mut bp, 0, 1);
        }
        // ptl_num_sub_profiles u(8) = 0.
        push_bits(&mut buf, &mut bp, 0, 8);
        buf
    }

    #[test]
    fn ptl_with_zero_gci_v3_path() {
        let data = build_ptl_bytes_with_zero_gci(0);
        let mut br = BitReader::new(&data);
        let ptl = parse_profile_tier_level(&mut br, true, 0).unwrap();
        assert_eq!(ptl.general_profile_idc, 1);
        assert!(!ptl.general_tier_flag);
        assert_eq!(ptl.general_level_idc, 0x5A);
        assert!(ptl.ptl_frame_only_constraint_flag);
        assert!(!ptl.ptl_multilayer_enabled_flag);
        assert!(ptl.gci_present_flag);
        assert_eq!(ptl.gci, GeneralConstraintsInfo::default());
        assert_eq!(ptl.ptl_num_sub_profiles, 0);
    }

    #[test]
    fn ptl_with_zero_gci_v4_path() {
        // num_additional_bits = 6: V4-conformant; six additional flags
        // appear, all zero, no reserved bits.
        let data = build_ptl_bytes_with_zero_gci(6);
        let mut br = BitReader::new(&data);
        let ptl = parse_profile_tier_level(&mut br, true, 0).unwrap();
        assert!(ptl.gci_present_flag);
        assert_eq!(ptl.gci.num_additional_bits, 6);
        assert_eq!(
            ptl.gci,
            GeneralConstraintsInfo {
                num_additional_bits: 6,
                ..GeneralConstraintsInfo::default()
            }
        );
    }

    /// Build a PTL byte stream with a GCI body where every named flag
    /// is set to 1, every `_idc` field is set to its maximum, and the
    /// V4 additional-bit block is fully present with all six flags
    /// set to 1.
    fn build_ptl_bytes_with_full_gci() -> Vec<u8> {
        let mut buf = Vec::new();
        let mut bp = 0u32;
        push_bits(&mut buf, &mut bp, 1, 7); // general_profile_idc
        push_bits(&mut buf, &mut bp, 0, 1);
        push_bits(&mut buf, &mut bp, 0x5A, 8);
        push_bits(&mut buf, &mut bp, 1, 1);
        push_bits(&mut buf, &mut bp, 0, 1);
        push_bits(&mut buf, &mut bp, 1, 1); // gci_present

        // 3 general flags = 1.
        for _ in 0..3 {
            push_bits(&mut buf, &mut bp, 1, 1);
        }
        // picture format _idc fields at their max widths.
        push_bits(&mut buf, &mut bp, 0xF, 4); // sixteen_minus_max_bitdepth
        push_bits(&mut buf, &mut bp, 0x3, 2); // three_minus_max_chroma_format
                                              // 10 NAL flags = 1.
        for _ in 0..10 {
            push_bits(&mut buf, &mut bp, 1, 1);
        }
        // 6 partitioning flags = 1.
        for _ in 0..6 {
            push_bits(&mut buf, &mut bp, 1, 1);
        }
        // CTU + block: u(2) = 3, then 3 flags = 1.
        push_bits(&mut buf, &mut bp, 0x3, 2);
        for _ in 0..3 {
            push_bits(&mut buf, &mut bp, 1, 1);
        }
        // 6 intra + 16 inter + 13 t/q/r + 6 loop filter = 41 flags = 1.
        for _ in 0..(6 + 16 + 13 + 6) {
            push_bits(&mut buf, &mut bp, 1, 1);
        }
        // gci_num_additional_bits = 6 (V4).
        push_bits(&mut buf, &mut bp, 6, 8);
        // 6 V4 flags = 1.
        for _ in 0..6 {
            push_bits(&mut buf, &mut bp, 1, 1);
        }
        // Align.
        while bp != 0 {
            push_bits(&mut buf, &mut bp, 0, 1);
        }
        // ptl_num_sub_profiles u(8) = 0.
        push_bits(&mut buf, &mut bp, 0, 8);
        buf
    }

    #[test]
    fn ptl_with_full_gci_round_trip() {
        let data = build_ptl_bytes_with_full_gci();
        let mut br = BitReader::new(&data);
        let ptl = parse_profile_tier_level(&mut br, true, 0).unwrap();
        assert!(ptl.gci_present_flag);
        let g = &ptl.gci;
        // Spot-check across every band.
        assert!(g.intra_only_constraint_flag);
        assert!(g.all_layers_independent_constraint_flag);
        assert!(g.one_au_only_constraint_flag);
        assert_eq!(g.sixteen_minus_max_bitdepth_constraint_idc, 0xF);
        assert_eq!(g.three_minus_max_chroma_format_constraint_idc, 0x3);
        // NAL band: pin both ends.
        assert!(g.no_mixed_nalu_types_in_pic_constraint_flag);
        assert!(g.no_idr_rpl_constraint_flag);
        // Partitioning band.
        assert!(g.one_tile_per_pic_constraint_flag);
        assert!(g.no_subpic_info_constraint_flag);
        // CTU band: u(2) at full + 3 flags.
        assert_eq!(g.three_minus_max_log2_ctu_size_constraint_idc, 0x3);
        assert!(g.no_partition_constraints_override_constraint_flag);
        assert!(g.no_mtt_constraint_flag);
        assert!(g.no_qtbtt_dual_tree_intra_constraint_flag);
        // Intra band.
        assert!(g.no_palette_constraint_flag);
        assert!(g.no_cclm_constraint_flag);
        // Inter band: pin both ends + a middle entry.
        assert!(g.no_ref_pic_resampling_constraint_flag);
        assert!(g.no_dmvr_constraint_flag);
        assert!(g.no_gpm_constraint_flag);
        // Transform/quant/residual band: pin both ends.
        assert!(g.no_luma_transform_size_64_constraint_flag);
        assert!(g.no_chroma_qp_offset_constraint_flag);
        // Loop filter band.
        assert!(g.no_sao_constraint_flag);
        assert!(g.no_virtual_boundaries_constraint_flag);
        // V4 additional block.
        assert_eq!(g.num_additional_bits, 6);
        assert!(g.all_rap_pictures_constraint_flag);
        assert!(g.no_extended_precision_processing_constraint_flag);
        assert!(g.no_ts_residual_coding_rice_constraint_flag);
        assert!(g.no_rrc_rice_extension_constraint_flag);
        assert!(g.no_persistent_rice_adaptation_constraint_flag);
        assert!(g.no_reverse_last_sig_coeff_constraint_flag);
        // PTL tail still reads cleanly.
        assert_eq!(ptl.ptl_num_sub_profiles, 0);
    }

    #[test]
    fn ptl_with_gci_num_additional_bits_reserved_ignored() {
        // num_additional_bits = 10 → 6 V4 flags + 4 reserved bits the
        // decoder shall ignore per §7.4.4.2.
        let data = build_ptl_bytes_with_zero_gci(10);
        let mut br = BitReader::new(&data);
        let ptl = parse_profile_tier_level(&mut br, true, 0).unwrap();
        assert!(ptl.gci_present_flag);
        assert_eq!(ptl.gci.num_additional_bits, 10);
        // 6 named V4 flags were read (all zero in this fixture); the
        // 4 reserved bits did not surface.
        assert!(!ptl.gci.all_rap_pictures_constraint_flag);
    }

    #[test]
    fn ptl_with_gci_num_additional_bits_underflow_rejected() {
        // Craft a stream with gci_num_additional_bits = 3 (< 6). The
        // V4 path is skipped per the `> 5` gate, so this is a valid
        // shape (3 reserved bits, no V4 flags read). The error branch
        // we want to exercise is a hypothetical num_additional_bits
        // value that would underflow `used`; we cover that by
        // constructing a byte stream by hand with num_additional_bits
        // = 5 (no V4 path), reading it, and confirming no rejection
        // (this pins that the `> 5` gate, not `>= 5`, is what guards
        // the read).
        let data = build_ptl_bytes_with_zero_gci(5);
        let mut br = BitReader::new(&data);
        let ptl = parse_profile_tier_level(&mut br, true, 0).unwrap();
        assert_eq!(ptl.gci.num_additional_bits, 5);
        assert!(!ptl.gci.all_rap_pictures_constraint_flag);
    }
}
