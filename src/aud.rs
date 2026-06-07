//! VVC Access Unit Delimiter parser (§7.3.2.10 / §7.4.3.10).
//!
//! An AUD NAL unit (NAL type 20, `AudNut`) sits at the start of an
//! access unit and signals two pieces of information:
//!
//! * `aud_irap_or_gdr_flag` — whether the AU is an IRAP or GDR AU.
//! * `aud_pic_type` — restricts the set of `sh_slice_type` values
//!   that may be present in the AU per Table 7:
//!     - `0` → `I` only,
//!     - `1` → `{P, I}`,
//!     - `2` → `{B, P, I}`.
//!
//! §7.4.3.10 constrains `aud_pic_type` to `0..=2` in bitstreams
//! conforming to this version of the spec; other values are reserved.
//! Per the same clause, conforming decoders shall ignore reserved
//! values rather than reject. The parser surfaces the raw 3-bit value
//! so callers can inspect it; a separate accessor classifies whether
//! the value falls in the conforming `0..=2` range.
//!
//! Syntax (§7.3.2.10):
//!
//! ```text
//!   aud_irap_or_gdr_flag   u(1)
//!   aud_pic_type           u(3)
//!   rbsp_trailing_bits()
//! ```
//!
//! When the bitstream contains only one layer, there is no normative
//! decoding process associated with the AUD; the parser is a syntax
//! inspector that mirrors what a multi-layer / sub-bitstream extractor
//! would consult per Annex C.6.

use oxideav_core::{Error, Result};

use crate::bitreader::BitReader;

/// Picture-type classification carried by `aud_pic_type` per §7.4.3.10
/// Table 7. Reserved values (`3..=7`) are preserved as `Reserved(raw)`
/// rather than mapped to `Err`, matching the spec's "decoders shall
/// ignore reserved values" handling rule.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AudPicType {
    /// `aud_pic_type == 0` — only `I` slices may appear in the AU.
    IOnly,
    /// `aud_pic_type == 1` — only `{P, I}` slices may appear in the AU.
    PorI,
    /// `aud_pic_type == 2` — any of `{B, P, I}` may appear in the AU.
    BPorI,
    /// `aud_pic_type` in `3..=7` — reserved for future use, decoders
    /// must ignore per §7.4.3.10. Raw value is preserved.
    Reserved(u8),
}

impl AudPicType {
    /// Map the raw 3-bit `aud_pic_type` field into the enum.
    pub fn from_raw(raw: u8) -> Self {
        match raw {
            0 => AudPicType::IOnly,
            1 => AudPicType::PorI,
            2 => AudPicType::BPorI,
            n => AudPicType::Reserved(n),
        }
    }

    /// Recover the 3-bit raw field value.
    pub fn as_raw(self) -> u8 {
        match self {
            AudPicType::IOnly => 0,
            AudPicType::PorI => 1,
            AudPicType::BPorI => 2,
            AudPicType::Reserved(n) => n,
        }
    }

    /// `true` iff the value lies in the `0..=2` range required of
    /// conforming bitstreams per §7.4.3.10.
    pub fn is_conforming(self) -> bool {
        matches!(
            self,
            AudPicType::IOnly | AudPicType::PorI | AudPicType::BPorI
        )
    }
}

/// Parsed AUD RBSP contents (§7.3.2.10).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AccessUnitDelimiter {
    /// `aud_irap_or_gdr_flag` — `true` iff the AU containing the AUD
    /// is an IRAP or GDR AU.
    pub aud_irap_or_gdr_flag: bool,
    /// `aud_pic_type` classified per Table 7.
    pub aud_pic_type: AudPicType,
}

/// Parse an AUD RBSP (bytes after the 2-byte NAL header, with
/// emulation-prevention bytes already stripped).
///
/// The parser consumes exactly the `u(1)` + `u(3)` body of §7.3.2.10
/// and does **not** verify the trailing `rbsp_trailing_bits()`. Callers
/// that need a strict full-NAL conformance check can chain
/// `BitReader::next_bits_are_rbsp_trailing()` (or equivalent) after
/// retrieving the parser's bit offset.
pub fn parse_access_unit_delimiter(rbsp: &[u8]) -> Result<AccessUnitDelimiter> {
    if rbsp.is_empty() {
        return Err(Error::invalid("h266 AUD: empty RBSP"));
    }
    let mut br = BitReader::new(rbsp);
    let aud_irap_or_gdr_flag = br.u1()? == 1;
    let raw_pic_type = br.u(3)? as u8;
    Ok(AccessUnitDelimiter {
        aud_irap_or_gdr_flag,
        aud_pic_type: AudPicType::from_raw(raw_pic_type),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `aud_irap_or_gdr_flag = 1`, `aud_pic_type = 0` (I-only AU).
    /// Body bits: `1 000` → followed by the rbsp_trailing_bits pad
    /// `1 000` for the byte: `1000_1000` = `0x88`.
    #[test]
    fn irap_with_i_only_pic_type() {
        let data = [0b1000_1000u8];
        let aud = parse_access_unit_delimiter(&data).unwrap();
        assert!(aud.aud_irap_or_gdr_flag);
        assert_eq!(aud.aud_pic_type, AudPicType::IOnly);
        assert!(aud.aud_pic_type.is_conforming());
        assert_eq!(aud.aud_pic_type.as_raw(), 0);
    }

    /// `aud_irap_or_gdr_flag = 0`, `aud_pic_type = 1` (`{P, I}` AU).
    /// Body bits: `0 001` → `0001_1000` (`1` is the rbsp_trailing
    /// stop bit) → `0x18`.
    #[test]
    fn non_irap_with_p_or_i_pic_type() {
        let data = [0b0001_1000u8];
        let aud = parse_access_unit_delimiter(&data).unwrap();
        assert!(!aud.aud_irap_or_gdr_flag);
        assert_eq!(aud.aud_pic_type, AudPicType::PorI);
        assert!(aud.aud_pic_type.is_conforming());
    }

    /// `aud_irap_or_gdr_flag = 0`, `aud_pic_type = 2` (`{B, P, I}` AU).
    /// Body bits: `0 010` → `0010_1000` → `0x28`.
    #[test]
    fn b_p_or_i_pic_type() {
        let data = [0b0010_1000u8];
        let aud = parse_access_unit_delimiter(&data).unwrap();
        assert!(!aud.aud_irap_or_gdr_flag);
        assert_eq!(aud.aud_pic_type, AudPicType::BPorI);
        assert!(aud.aud_pic_type.is_conforming());
        assert_eq!(aud.aud_pic_type.as_raw(), 2);
    }

    /// `aud_pic_type = 5` — reserved range (`3..=7`). §7.4.3.10 mandates
    /// that conforming decoders ignore reserved values; the parser
    /// surfaces them as `Reserved(raw)` so callers can apply the
    /// "ignore" semantics without losing the original byte.
    /// Body bits: `1 101` → `1101_1000` → `0xD8`.
    #[test]
    fn reserved_pic_type_is_preserved_not_rejected() {
        let data = [0b1101_1000u8];
        let aud = parse_access_unit_delimiter(&data).unwrap();
        assert!(aud.aud_irap_or_gdr_flag);
        assert_eq!(aud.aud_pic_type, AudPicType::Reserved(5));
        assert!(!aud.aud_pic_type.is_conforming());
        assert_eq!(aud.aud_pic_type.as_raw(), 5);
    }

    /// `aud_pic_type = 7` boundary on the reserved range.
    /// Body bits: `0 111` → `0111_1000` → `0x78`.
    #[test]
    fn reserved_pic_type_seven_boundary() {
        let data = [0b0111_1000u8];
        let aud = parse_access_unit_delimiter(&data).unwrap();
        assert!(!aud.aud_irap_or_gdr_flag);
        assert_eq!(aud.aud_pic_type, AudPicType::Reserved(7));
        assert!(!aud.aud_pic_type.is_conforming());
    }

    #[test]
    fn empty_rbsp_is_rejected() {
        assert!(parse_access_unit_delimiter(&[]).is_err());
    }

    /// Round-trip every conforming `aud_pic_type` value: the parser
    /// classifies, `as_raw()` recovers the source value, and
    /// `from_raw()` survives a full round.
    #[test]
    fn conforming_pic_type_round_trip() {
        for raw in 0u8..=2 {
            let typed = AudPicType::from_raw(raw);
            assert!(typed.is_conforming());
            assert_eq!(typed.as_raw(), raw);
            assert_eq!(AudPicType::from_raw(typed.as_raw()), typed);
        }
    }

    /// All reserved values survive `from_raw` → `as_raw` round-trip.
    #[test]
    fn reserved_pic_type_round_trip() {
        for raw in 3u8..=7 {
            let typed = AudPicType::from_raw(raw);
            assert!(!typed.is_conforming());
            assert_eq!(typed.as_raw(), raw);
            assert_eq!(AudPicType::from_raw(typed.as_raw()), typed);
        }
    }
}
