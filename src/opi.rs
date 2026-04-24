//! VVC Operating Point Information parser (§7.3.2.2 / §7.4.3.2).
//!
//! An OPI NAL unit advertises the operating point at which the decoder
//! is running — either the target OLS index or the highest temporal
//! sublayer to decode, or both. §7.4.3.2 states that at least one of
//! `opi_ols_info_present_flag`, `opi_htid_info_present_flag`, and
//! `opi_extension_flag` must be 1; we enforce that constraint.
//!
//! Syntax (§7.3.2.2):
//!
//! ```text
//!   opi_ols_info_present_flag   u(1)
//!   opi_htid_info_present_flag  u(1)
//!   if opi_ols_info_present_flag:
//!       opi_ols_idx             ue(v)
//!   if opi_htid_info_present_flag:
//!       opi_htid_plus1          u(3)
//!   opi_extension_flag          u(1)
//!   if opi_extension_flag:
//!       while more_rbsp_data():
//!           opi_extension_data_flag  u(1)
//!   rbsp_trailing_bits()
//! ```

use oxideav_core::{Error, Result};

use crate::bitreader::BitReader;

/// Parsed OPI RBSP contents.
#[derive(Clone, Copy, Debug)]
pub struct OperatingPointInformation {
    pub opi_ols_info_present_flag: bool,
    pub opi_htid_info_present_flag: bool,
    /// Present when `opi_ols_info_present_flag` is 1.
    pub opi_ols_idx: Option<u32>,
    /// Present when `opi_htid_info_present_flag` is 1 (3-bit field).
    pub opi_htid_plus1: Option<u8>,
    pub opi_extension_flag: bool,
}

/// Parse an OPI RBSP (bytes after the 2-byte NAL header, emulation-
/// prevention bytes stripped).
pub fn parse_opi(rbsp: &[u8]) -> Result<OperatingPointInformation> {
    if rbsp.is_empty() {
        return Err(Error::invalid("h266 OPI: empty RBSP"));
    }
    let mut br = BitReader::new(rbsp);
    let opi_ols_info_present_flag = br.u1()? == 1;
    let opi_htid_info_present_flag = br.u1()? == 1;
    let opi_ols_idx = if opi_ols_info_present_flag {
        Some(br.ue()?)
    } else {
        None
    };
    let opi_htid_plus1 = if opi_htid_info_present_flag {
        Some(br.u(3)? as u8)
    } else {
        None
    };
    let opi_extension_flag = br.u1()? == 1;
    // §7.4.3.2: "One or more of opi_htid_info_present_flag,
    // opi_ols_info_present_flag, and opi_extension_flag shall be equal
    // to 1." Flag an OPI that advertises nothing.
    if !opi_ols_info_present_flag && !opi_htid_info_present_flag && !opi_extension_flag {
        return Err(Error::invalid(
            "h266 OPI: all of opi_ols_info_present_flag, opi_htid_info_present_flag, \
             opi_extension_flag are 0 (§7.4.3.2 forbids)",
        ));
    }
    // Extension bits (`while more_rbsp_data()`) are intentionally not
    // walked here: §7.4.3.2 says decoders must ignore them, and doing
    // so cleanly would require walking rbsp_trailing_bits() state.
    Ok(OperatingPointInformation {
        opi_ols_info_present_flag,
        opi_htid_info_present_flag,
        opi_ols_idx,
        opi_htid_plus1,
        opi_extension_flag,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Bits MSB-first. Layout:
    /// ols_info=1, htid_info=1, ols_idx=ue(0)="1", htid_plus1=u3(3)="011",
    /// extension_flag=0.
    /// Stream: 1 1 1 011 0 → 1110_1100 → 0xEC
    #[test]
    fn opi_ols_and_htid() {
        let data = [0b1110_1100u8];
        let opi = parse_opi(&data).unwrap();
        assert!(opi.opi_ols_info_present_flag);
        assert!(opi.opi_htid_info_present_flag);
        assert_eq!(opi.opi_ols_idx, Some(0));
        assert_eq!(opi.opi_htid_plus1, Some(3));
        assert!(!opi.opi_extension_flag);
    }

    /// Only htid info: ols_info=0, htid_info=1, htid_plus1=u3(0)="000",
    /// extension_flag=0 — but that leaves all three flags 0 except htid → valid.
    /// Stream: 0 1 000 0 _ = 0100_0000 → 0x40
    #[test]
    fn opi_only_htid() {
        let data = [0b0100_0000u8];
        let opi = parse_opi(&data).unwrap();
        assert!(!opi.opi_ols_info_present_flag);
        assert!(opi.opi_htid_info_present_flag);
        assert_eq!(opi.opi_ols_idx, None);
        assert_eq!(opi.opi_htid_plus1, Some(0));
        assert!(!opi.opi_extension_flag);
    }

    /// All three "information" flags 0 — §7.4.3.2 forbids.
    /// Stream: 0 0 0 (extension_flag=0) = 0000_0000 → 0x00
    #[test]
    fn all_flags_zero_is_rejected() {
        let data = [0x00u8];
        assert!(parse_opi(&data).is_err());
    }

    /// Extension-only OPI (conformance for future versions):
    /// ols_info=0, htid_info=0, extension_flag=1. Stream: 001 → 0010_0000 → 0x20
    #[test]
    fn extension_only_opi_is_accepted() {
        let data = [0b0010_0000u8];
        let opi = parse_opi(&data).unwrap();
        assert!(!opi.opi_ols_info_present_flag);
        assert!(!opi.opi_htid_info_present_flag);
        assert!(opi.opi_extension_flag);
    }

    #[test]
    fn empty_rbsp_is_rejected() {
        assert!(parse_opi(&[]).is_err());
    }
}
