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
    // Everything past this point (cabac/init, ref_idx defaults, QP /
    // deblocking / SAO / ALF control, extension flags) is deliberately
    // not parsed in the foundation pass. Surface-level PPS inspection
    // is already sufficient to pair a picture with its SPS.
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

    /// Minimal single-slice 320x240 PPS (no partitioning, no subpic map).
    #[test]
    fn minimal_pps_roundtrip() {
        let mut bits: Vec<u8> = Vec::new();
        push_u(&mut bits, 0, 6); // pps_id
        push_u(&mut bits, 0, 4); // sps_id
        push_u(&mut bits, 0, 1); // mixed_nalu_types
        push_u(&mut bits, 0b00000000010100_0001, 17); // ue(320)
        push_u(&mut bits, 0b000000011110001, 15); // ue(240)
        push_u(&mut bits, 0, 1); // conformance_window_flag
        push_u(&mut bits, 0, 1); // scaling_window
        push_u(&mut bits, 0, 1); // output_flag_present
        push_u(&mut bits, 1, 1); // no_pic_partition = 1 (foundation path)
        push_u(&mut bits, 0, 1); // subpic_id_mapping_present = 0
        let bytes = pack_bits(&bits);
        let pps = parse_pps(&bytes).unwrap();
        assert_eq!(pps.pps_pic_parameter_set_id, 0);
        assert_eq!(pps.pps_seq_parameter_set_id, 0);
        assert_eq!(pps.pps_pic_width_in_luma_samples, 320);
        assert_eq!(pps.pps_pic_height_in_luma_samples, 240);
        assert!(pps.pps_no_pic_partition_flag);
        assert!(!pps.pps_subpic_id_mapping_present_flag);
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
