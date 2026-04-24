//! VVC Decoding Capability Information parser (§7.3.2.1 / §7.4.3.1).
//!
//! The DCI RBSP is a small, standalone parameter set that advertises the
//! set of profile / tier / level combinations the bitstream can conform
//! to. Its semantics (§7.4.3.1) explicitly state that the information
//! is **not** necessary for the decoding process — decoders may ignore
//! it — but surfacing it is useful for container-level muxers, player
//! capability negotiation, and conformance tooling.
//!
//! Structure (§7.3.2.1):
//!
//! ```text
//!   dci_reserved_zero_4bits    u(4)
//!   dci_num_ptls_minus1        u(4)
//!   for i in 0..=dci_num_ptls_minus1:
//!       profile_tier_level(1, 0)
//!   dci_extension_flag         u(1)
//!   if dci_extension_flag:
//!       while more_rbsp_data():
//!           dci_extension_data_flag  u(1)
//!   rbsp_trailing_bits()
//! ```
//!
//! §7.4.3.1: `dci_num_ptls_minus1` is in the range `0..=14`; value 15
//! is reserved. `dci_extension_flag` shall be 0 in bitstreams
//! conforming to this version of the spec, but decoders must tolerate
//! `== 1` and ignore the extension data bits.

use oxideav_core::{Error, Result};

use crate::bitreader::BitReader;
use crate::ptl::{parse_profile_tier_level, ProfileTierLevel};

/// Parsed DCI RBSP contents.
#[derive(Clone, Debug)]
pub struct DecodingCapabilityInformation {
    /// `dci_num_ptls_minus1` (§7.4.3.1): the number of PTL structures is
    /// `dci_num_ptls_minus1 + 1`. The raw value is retained for callers
    /// that want to echo it.
    pub dci_num_ptls_minus1: u8,
    /// One `profile_tier_level(1, 0)` per entry.
    pub profile_tier_levels: Vec<ProfileTierLevel>,
    /// `dci_extension_flag` (§7.4.3.1). Must be 0 in v1/v2/v3/v4 bitstreams
    /// but decoders must still tolerate `== 1`.
    pub dci_extension_flag: bool,
}

/// Parse a DCI RBSP (bytes after the 2-byte NAL header, emulation
/// prevention stripped).
pub fn parse_dci(rbsp: &[u8]) -> Result<DecodingCapabilityInformation> {
    if rbsp.is_empty() {
        return Err(Error::invalid("h266 DCI: empty RBSP"));
    }
    let mut br = BitReader::new(rbsp);
    // dci_reserved_zero_4bits — §7.4.3.1 says decoders shall ignore its
    // value, so we read and drop it.
    let _reserved = br.u(4)?;
    let dci_num_ptls_minus1 = br.u(4)? as u8;
    if dci_num_ptls_minus1 == 15 {
        // §7.4.3.1 reserves the value 15 for future use; a conforming
        // decoder should surface that rather than blow past it.
        return Err(Error::invalid(
            "h266 DCI: dci_num_ptls_minus1 = 15 is reserved (§7.4.3.1)",
        ));
    }
    let num_ptls = dci_num_ptls_minus1 as usize + 1;
    let mut profile_tier_levels = Vec::with_capacity(num_ptls);
    for _ in 0..num_ptls {
        // DCI PTL entries always pass profileTierPresentFlag = 1 and
        // MaxNumSubLayersMinus1 = 0 (§7.3.2.1).
        let ptl = parse_profile_tier_level(&mut br, true, 0)?;
        profile_tier_levels.push(ptl);
    }
    let dci_extension_flag = br.u1()? == 1;
    // We deliberately do not walk the `while(more_rbsp_data())` loop of
    // dci_extension_data_flag bits — §7.4.3.1 says those bits must be
    // ignored by this version's decoders, and walking them would
    // require stepping through rbsp_trailing_bits() state which this
    // crate doesn't yet model.
    Ok(DecodingCapabilityInformation {
        dci_num_ptls_minus1,
        profile_tier_levels,
        dci_extension_flag,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build bits MSB-first into a Vec<u8> buffer.
    fn push_u(bits: &mut Vec<u8>, v: u64, n: u32) {
        for i in (0..n).rev() {
            bits.push(((v >> i) & 1) as u8);
        }
    }

    fn pack_bits(bits: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(bits.len().div_ceil(8));
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

    /// Build one PTL with `profileTierPresentFlag = 1` and
    /// `MaxNumSubLayersMinus1 = 0`. Layout (§7.3.3.1):
    ///   general_profile_idc            u(7) = 1
    ///   general_tier_flag              u(1) = 0
    ///   general_level_idc              u(8) = 0x5A
    ///   ptl_frame_only_constraint_flag u(1) = 1
    ///   ptl_multilayer_enabled_flag    u(1) = 0
    ///   gci_present_flag               u(1) = 0 (skip GCI)
    ///   byte-align zero bits           (align to next byte)
    ///   ptl_num_sub_profiles           u(8) = 0
    ///   (sublayer levels absent because MaxNumSubLayersMinus1 = 0)
    fn push_minimal_ptl(bits: &mut Vec<u8>) {
        push_u(bits, 1, 7); // profile_idc = 1 (main)
        push_u(bits, 0, 1); // tier = main
        push_u(bits, 0x5A, 8); // level_idc
        push_u(bits, 1, 1); // frame_only
        push_u(bits, 0, 1); // multilayer_enabled
        push_u(bits, 0, 1); // gci_present
        // Align to byte boundary.
        while bits.len() % 8 != 0 {
            bits.push(0);
        }
        push_u(bits, 0, 8); // ptl_num_sub_profiles
    }

    #[test]
    fn minimal_dci_one_ptl() {
        let mut bits: Vec<u8> = Vec::new();
        push_u(&mut bits, 0, 4); // dci_reserved_zero_4bits
        push_u(&mut bits, 0, 4); // dci_num_ptls_minus1 = 0 → 1 PTL
        push_minimal_ptl(&mut bits);
        push_u(&mut bits, 0, 1); // dci_extension_flag = 0
        // rbsp_trailing_bits() — emit a 1 + zeros to byte-align. Not
        // needed for the parse itself (we stop at dci_extension_flag).
        let bytes = pack_bits(&bits);

        let dci = parse_dci(&bytes).unwrap();
        assert_eq!(dci.dci_num_ptls_minus1, 0);
        assert_eq!(dci.profile_tier_levels.len(), 1);
        let ptl = &dci.profile_tier_levels[0];
        assert_eq!(ptl.general_profile_idc, 1);
        assert!(!ptl.general_tier_flag);
        assert_eq!(ptl.general_level_idc, 0x5A);
        assert!(ptl.ptl_frame_only_constraint_flag);
        assert!(!ptl.gci_present_flag);
        assert!(!dci.dci_extension_flag);
    }

    #[test]
    fn dci_two_ptls() {
        let mut bits: Vec<u8> = Vec::new();
        push_u(&mut bits, 0, 4); // reserved
        push_u(&mut bits, 1, 4); // num_ptls_minus1 = 1 → 2 PTLs
        push_minimal_ptl(&mut bits);
        push_minimal_ptl(&mut bits);
        push_u(&mut bits, 0, 1); // extension_flag
        let bytes = pack_bits(&bits);

        let dci = parse_dci(&bytes).unwrap();
        assert_eq!(dci.dci_num_ptls_minus1, 1);
        assert_eq!(dci.profile_tier_levels.len(), 2);
    }

    #[test]
    fn reserved_num_ptls_is_rejected() {
        // dci_num_ptls_minus1 = 15 is reserved (§7.4.3.1).
        let mut bits: Vec<u8> = Vec::new();
        push_u(&mut bits, 0, 4);
        push_u(&mut bits, 15, 4);
        let bytes = pack_bits(&bits);
        assert!(parse_dci(&bytes).is_err());
    }

    #[test]
    fn empty_rbsp_is_rejected() {
        assert!(parse_dci(&[]).is_err());
    }
}
