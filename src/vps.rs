//! VVC Video Parameter Set parser (§7.3.2.3).
//!
//! Foundation scope: parse up to and including `vps_num_ptls_minus1` plus
//! the list of `profile_tier_level()` structures. The OLS / DPB / HRD tail
//! requires a set of derived variables (`TotalNumOlss`, `NumMultiLayerOlss`,
//! …) that are only meaningful once actual multi-layer decoding is being
//! implemented. For the scaffold we stop after the PTL list and leave the
//! remainder of the RBSP (OLS pairing, DPB/HRD/timing, extension data)
//! unread; the `byte_offset` method lets callers see how far we got.
//!
//! Most single-layer VVC bitstreams do not actually carry a VPS (§7.4.3.3:
//! the VPS can be absent when `sps_video_parameter_set_id == 0`). The
//! parser is therefore a best-effort inspector for multi-layer streams and
//! for the identifying fields (`vps_video_parameter_set_id`,
//! `vps_max_layers_minus1`, …) that non-multi-layer consumers might
//! still want to read.

use oxideav_core::{Error, Result};

use crate::bitreader::BitReader;
use crate::ptl::{parse_profile_tier_level, ProfileTierLevel};

#[derive(Clone, Debug)]
pub struct VideoParameterSet {
    pub vps_video_parameter_set_id: u8,
    pub vps_max_layers_minus1: u8,
    pub vps_max_sublayers_minus1: u8,
    pub vps_default_ptl_dpb_hrd_max_tid_flag: bool,
    pub vps_all_independent_layers_flag: bool,
    /// `vps_layer_id[i]` for i in 0..=vps_max_layers_minus1.
    pub vps_layer_id: Vec<u8>,
    /// `vps_independent_layer_flag[i]` for i in 0..=vps_max_layers_minus1.
    /// Inferred to 1 when either `vps_max_layers_minus1 == 0` or
    /// `vps_all_independent_layers_flag == 1`.
    pub vps_independent_layer_flag: Vec<bool>,
    /// `vps_each_layer_is_an_ols_flag` (§7.4.3.3). Inferred to 1 when
    /// there is only one layer.
    pub vps_each_layer_is_an_ols_flag: bool,
    /// `vps_ols_mode_idc` (§7.4.3.3), when explicitly signalled.
    pub vps_ols_mode_idc: Option<u8>,
    pub vps_num_ptls_minus1: u8,
    /// Parsed PTL list; one entry per i in 0..=vps_num_ptls_minus1.
    pub profile_tier_levels: Vec<ProfileTierLevel>,
}

/// Parse a VPS NAL RBSP payload (i.e. the bytes after the 2-byte NAL header,
/// already stripped of emulation-prevention bytes).
///
/// The parse intentionally stops after the PTL list; callers should not
/// rely on derived OLS / DPB / HRD state from this scaffold.
pub fn parse_vps(rbsp: &[u8]) -> Result<VideoParameterSet> {
    let mut br = BitReader::new(rbsp);
    let vps_video_parameter_set_id = br.u(4)? as u8;
    if vps_video_parameter_set_id == 0 {
        // §7.4.3.3: vps_video_parameter_set_id shall be > 0.
        return Err(Error::invalid(
            "h266 VPS: vps_video_parameter_set_id must be > 0 (§7.4.3.3)",
        ));
    }
    let vps_max_layers_minus1 = br.u(6)? as u8;
    let vps_max_sublayers_minus1 = br.u(3)? as u8;
    if vps_max_sublayers_minus1 > 6 {
        return Err(Error::invalid(
            "h266 VPS: vps_max_sublayers_minus1 must be <= 6",
        ));
    }
    let vps_default_ptl_dpb_hrd_max_tid_flag =
        if vps_max_layers_minus1 > 0 && vps_max_sublayers_minus1 > 0 {
            br.u1()? == 1
        } else {
            true
        };
    let vps_all_independent_layers_flag = if vps_max_layers_minus1 > 0 {
        br.u1()? == 1
    } else {
        true
    };
    let layer_count = (vps_max_layers_minus1 as usize) + 1;
    let mut vps_layer_id = Vec::with_capacity(layer_count);
    let mut vps_independent_layer_flag = Vec::with_capacity(layer_count);
    for i in 0..layer_count {
        vps_layer_id.push(br.u(6)? as u8);
        let indep = if i > 0 && !vps_all_independent_layers_flag {
            br.u1()? == 1
        } else {
            true
        };
        vps_independent_layer_flag.push(indep);
        if i > 0 && !vps_all_independent_layers_flag && !indep {
            let vps_max_tid_ref_present_flag = br.u1()? == 1;
            for _j in 0..i {
                let vps_direct_ref_layer_flag = br.u1()? == 1;
                if vps_max_tid_ref_present_flag && vps_direct_ref_layer_flag {
                    br.skip(3)?; // vps_max_tid_il_ref_pics_plus1
                }
            }
        }
    }
    let mut vps_each_layer_is_an_ols_flag = true;
    let mut vps_ols_mode_idc: Option<u8> = None;
    if vps_max_layers_minus1 > 0 {
        if vps_all_independent_layers_flag {
            vps_each_layer_is_an_ols_flag = br.u1()? == 1;
        } else {
            // Not all independent → can't take the "each layer an OLS"
            // shortcut; vps_ols_mode_idc is signalled below.
            vps_each_layer_is_an_ols_flag = false;
        }
        if !vps_each_layer_is_an_ols_flag {
            if !vps_all_independent_layers_flag {
                let idc = br.u(2)? as u8;
                vps_ols_mode_idc = Some(idc);
                if idc == 2 {
                    let vps_num_output_layer_sets_minus2 = br.u(8)? as u32;
                    for _i in 1..=(vps_num_output_layer_sets_minus2 + 1) {
                        for _j in 0..=vps_max_layers_minus1 as u32 {
                            br.skip(1)?; // vps_ols_output_layer_flag
                        }
                    }
                }
            }
        }
    }
    let vps_num_ptls_minus1 = br.u(8)? as u8;
    let num_ptls = (vps_num_ptls_minus1 as usize) + 1;
    let mut vps_pt_present_flag = Vec::with_capacity(num_ptls);
    let mut vps_ptl_max_tid = Vec::with_capacity(num_ptls);
    for i in 0..num_ptls {
        let pt_present = if i > 0 { br.u1()? == 1 } else { true };
        vps_pt_present_flag.push(pt_present);
        let max_tid = if !vps_default_ptl_dpb_hrd_max_tid_flag {
            br.u(3)? as u8
        } else {
            vps_max_sublayers_minus1
        };
        vps_ptl_max_tid.push(max_tid);
    }
    // vps_ptl_alignment_zero_bit(s) — pad to byte alignment.
    while !br.is_byte_aligned() {
        br.skip(1)?;
    }
    let mut profile_tier_levels = Vec::with_capacity(num_ptls);
    for i in 0..num_ptls {
        let ptl = parse_profile_tier_level(&mut br, vps_pt_present_flag[i], vps_ptl_max_tid[i])?;
        profile_tier_levels.push(ptl);
    }
    // Everything past this point (OLS pairing, DPB parameters, HRD
    // parameters, extension data) is deliberately not parsed in the
    // foundation scaffold; see module-level comment.
    Ok(VideoParameterSet {
        vps_video_parameter_set_id,
        vps_max_layers_minus1,
        vps_max_sublayers_minus1,
        vps_default_ptl_dpb_hrd_max_tid_flag,
        vps_all_independent_layers_flag,
        vps_layer_id,
        vps_independent_layer_flag,
        vps_each_layer_is_an_ols_flag,
        vps_ols_mode_idc,
        vps_num_ptls_minus1,
        profile_tier_levels,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Hand-built minimal VPS RBSP: vps_id=1, single layer, single sublayer,
    /// single PTL with level_idc = 0x5A.
    #[test]
    fn minimal_vps_roundtrip() {
        // See the module-level comments in the test for bit layout.
        let data = [0x10u8, 0x00, 0x00, 0x00, 0x02, 0x5A, 0x80, 0x00];
        let vps = parse_vps(&data).unwrap();
        assert_eq!(vps.vps_video_parameter_set_id, 1);
        assert_eq!(vps.vps_max_layers_minus1, 0);
        assert_eq!(vps.vps_max_sublayers_minus1, 0);
        assert_eq!(vps.vps_layer_id, vec![0]);
        assert_eq!(vps.vps_independent_layer_flag, vec![true]);
        assert_eq!(vps.vps_num_ptls_minus1, 0);
        assert_eq!(vps.profile_tier_levels.len(), 1);
        assert_eq!(vps.profile_tier_levels[0].general_profile_idc, 1);
        assert!(!vps.profile_tier_levels[0].general_tier_flag);
        assert_eq!(vps.profile_tier_levels[0].general_level_idc, 0x5A);
        assert!(vps.profile_tier_levels[0].ptl_frame_only_constraint_flag);
    }

    #[test]
    fn vps_id_zero_is_rejected() {
        // All zeros → vps_id = 0, which §7.4.3.3 forbids.
        let data = [0u8; 8];
        assert!(parse_vps(&data).is_err());
    }
}
