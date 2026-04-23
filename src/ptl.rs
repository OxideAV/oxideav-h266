//! `profile_tier_level()` parser (§7.3.3.1) — shared by DCI, VPS and SPS.
//!
//! VVC's PTL is structurally different from HEVC's: the sublayer level
//! array is stored in descending order, GCI is a dedicated sub-structure
//! gated on `gci_present_flag`, and `ptl_num_sub_profiles` replaces HEVC's
//! 32-bit profile-compatibility field.

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

/// Walk `general_constraints_info()` (§7.3.3.2). We don't surface all 74+
/// constraint flags here — foundation code only needs to advance the bit
/// position correctly so the surrounding PTL / VPS / SPS parse stays
/// aligned. `gci_present_flag` is exposed; if false, the syntax is empty
/// except for the byte-alignment zeros.
fn parse_general_constraints_info(br: &mut BitReader<'_>, ptl: &mut ProfileTierLevel) -> Result<()> {
    ptl.gci_present_flag = br.u1()? == 1;
    if ptl.gci_present_flag {
        // 3 general flags.
        br.skip(3)?;
        // Picture format: 4 + 2 = 6 bits.
        br.skip(4 + 2)?;
        // NAL unit type related: 11 bits.
        br.skip(11)?;
        // Tile / slice / subpicture partitioning: 6 bits.
        br.skip(6)?;
        // CTU and block partitioning: 2 + 3 = 5 bits.
        br.skip(2 + 3)?;
        // Intra: 6 bits.
        br.skip(6)?;
        // Inter: 16 bits (ref_pic_resampling..gpm).
        br.skip(16)?;
        // Transform/quant/residual: 13 bits.
        br.skip(13)?;
        // Loop filter: 6 bits (sao..virtual_boundaries).
        br.skip(6)?;
        let gci_num_additional_bits = br.u(8)? as u32;
        let num_additional_bits_used = if gci_num_additional_bits > 5 {
            // 6 additional flags defined for v4.
            br.skip(6)?;
            6
        } else {
            0
        };
        if gci_num_additional_bits < num_additional_bits_used {
            return Err(Error::invalid(format!(
                "h266 GCI: gci_num_additional_bits {gci_num_additional_bits} < used {num_additional_bits_used}"
            )));
        }
        let reserved = gci_num_additional_bits - num_additional_bits_used;
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
    }
}
