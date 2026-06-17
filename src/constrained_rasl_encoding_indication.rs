//! VVC constrained RASL encoding indication SEI message parser
//! (§D.10.1 / §D.10.2).
//!
//! The constrained RASL encoding indication (CREI) SEI message
//! (`payloadType == 207`) is one of the few `sei_payload()` bodies
//! specified directly in this Specification — alongside the SEI manifest
//! (`payloadType == 200`), the SEI prefix indication
//! (`payloadType == 201`), the subpicture level information
//! (`payloadType == 203`), and the DU information
//! (`payloadType == 130`). Most other payload types defer to Rec.
//! ITU-T H.274 | ISO/IEC 23002-7.
//!
//! Unlike those, the CREI body carries **no syntax elements at all**
//! (§D.10.1):
//!
//! ```text
//!   constrained_rasl_encoding_indication( payloadSize ) {            Descriptor
//!   }
//! ```
//!
//! The semantics are conveyed purely by the message's *presence*. Per
//! §D.10.2, the presence of a CREI SEI message in a CVS indicates that a
//! set of encoding constraints applies for all RASL pictures in the CVS:
//!
//! * each RASL picture's PH syntax structure has `ph_dmvr_disabled_flag`
//!   equal to `1`;
//! * the PPS referred to by each RASL picture has
//!   `pps_ref_wraparound_enabled_flag` equal to `0`;
//! * no CU in a slice with `sh_slice_type` equal to `0` (B) or `1` (P)
//!   has `cclm_mode_flag` equal to `1`;
//! * no collocated reference picture precedes the CRA picture associated
//!   with the RASL picture in decoding order.
//!
//! These constraints (NOTE 2 in §D.10.2) make open-GOP IRAP (CRA-with-
//! RASL) pictures more amenable to bitstream switching in bit-rate
//! adaptive streaming, by limiting the visible artefacts in RASL pictures
//! decoded after a switch.
//!
//! Persistence (Table D.1): the CVS containing the SEI message. §D.10.2
//! further requires that when a CREI SEI message is present for any
//! picture of an AU of a CVS, a CREI SEI message shall be present for the
//! first picture of the CVSS AU, and that it persists in decoding order
//! from the current AU until the end of the CVS. Those AU-level
//! consistency checks are a higher-level concern than this single-payload
//! parser; this module decodes the (empty) body and validates the §D.2.1
//! `sei_payload()` framing so the message can be recognised and surfaced.
//!
//! ## Empty body, `sei_payload()` framing
//!
//! Because the §D.10.1 body has no syntax elements, the bit reader sits
//! at the start of the payload immediately, byte aligned. §D.2.1's
//! general SEI payload syntax then admits, when `payloadSize` carries
//! extra bytes (`SeiExtensionBitsPresentFlag || more_data_in_payload()`),
//! a single `sei_payload_bit_equal_to_one` (`f(1)`) followed by
//! `sei_payload_bit_equal_to_zero` bits to the byte boundary. For an
//! empty body with `payloadSize == 0` there are no further bits at all.
//! §7.4.6 requires the consumed length to equal `payloadSize`. This
//! parser reuses the shared §D.2.1 trailing-bits validator so a non-`1`
//! one-bit, a non-`0` zero-padding bit, or leftover bytes beyond the
//! framing surface as a parse error rather than silently desynchronising
//! the enclosing `sei_rbsp()` walk.
//!
//! Spec reference: ITU-T H.266 | ISO/IEC 23090-3 (V4, 01/2026),
//! §D.10.1, §D.10.2, §D.2 (`sei_payload()` dispatch), §D.2.1, §7.4.6,
//! Table D.1.
//!
//! No third-party VVC decoder source was consulted; the implementation
//! is spec-only and reads the payload through the crate's own
//! [`BitReader`].

use oxideav_core::{Error, Result};

use crate::bitreader::BitReader;

/// `payloadType` value that selects the constrained RASL encoding
/// indication body in the §D.2 `sei_payload()` dispatch.
pub const CONSTRAINED_RASL_ENCODING_INDICATION_PAYLOAD_TYPE: u32 = 207;

/// A parsed constrained RASL encoding indication SEI message (§D.10.1).
///
/// The §D.10.1 body has no syntax elements, so this is a zero-field
/// marker type: its mere construction records that a well-formed CREI
/// SEI message was present. The set of encoding constraints it asserts
/// (§D.10.2) is documented on the module and exposed through the
/// associated [`CREI_CONSTRAINTS`] description for consumers that want to
/// surface the meaning without re-deriving it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ConstrainedRaslEncodingIndication;

/// The four encoding constraints that the presence of a CREI SEI message
/// asserts for every RASL picture in the CVS (§D.10.2), in spec order.
///
/// Provided as plain prose so a consumer can describe the message's
/// effect (e.g. in a stream report) without restating §D.10.2.
pub const CREI_CONSTRAINTS: [&str; 4] = [
    "PH ph_dmvr_disabled_flag == 1",
    "PPS pps_ref_wraparound_enabled_flag == 0",
    "no B/P-slice CU has cclm_mode_flag == 1",
    "no collocated reference picture precedes the associated CRA picture in decoding order",
];

impl ConstrainedRaslEncodingIndication {
    /// The §D.10.2 constraints asserted by this message's presence, in
    /// spec order. A convenience accessor mirroring [`CREI_CONSTRAINTS`].
    pub fn constraints(&self) -> &'static [&'static str; 4] {
        &CREI_CONSTRAINTS
    }
}

/// Parse a `constrained_rasl_encoding_indication( payloadSize )` body
/// (§D.10.1) from the raw SEI payload bytes carried by a
/// `payloadType == 207` `sei_message()`.
///
/// `payload` is the `sei_payload()` argument region — the `payloadSize`
/// bytes that follow the `sei_message()` header, with emulation-
/// prevention bytes already removed. The §D.10.1 body is empty, so the
/// only work is to validate the §D.2.1 `sei_payload()` framing: the
/// reader starts byte aligned at the front of the payload, and any
/// trailing bytes must be the canonical `sei_payload_bit_equal_to_one`
/// + `sei_payload_bit_equal_to_zero` padding (or, for `payloadSize == 0`,
/// no bytes at all).
///
/// Errors:
/// * a malformed §D.2.1 trailing-bits region (a non-`1`
///   `sei_payload_bit_equal_to_one`, a non-`0` zero-padding bit, or a
///   consumed length that does not equal `payloadSize`) is rejected.
pub fn parse_constrained_rasl_encoding_indication(
    payload: &[u8],
) -> Result<ConstrainedRaslEncodingIndication> {
    const ERR: &str =
        "h266 constrained_rasl_encoding_indication: malformed sei_payload trailing bits / length (§D.2.1, §7.4.6)";

    let mut reader = BitReader::new(payload);

    // §D.10.1: the body has no syntax elements, so the reader sits at the
    // front of the payload, byte aligned, at bit position 0.
    //
    // §D.2.1 then evaluates `SeiExtensionBitsPresentFlag ||
    // more_data_in_payload()`. `SeiExtensionBitsPresentFlag` is set to 0
    // at the top of `sei_payload()`, and §7.4.6 defines
    // `more_data_in_payload()` as FALSE iff the reader is byte aligned and
    // sitting exactly at `8 * payloadSize` bits. Here:
    //
    // * `payloadSize == 0` (an empty `payload`): position 0 == 8*0 and the
    //   reader is byte aligned, so `more_data_in_payload()` is FALSE — the
    //   trailing-bits block does not run and there are no further bits.
    // * `payloadSize > 0`: the reader is byte aligned at position 0 but
    //   0 != 8*payloadSize, so `more_data_in_payload()` is TRUE and the
    //   trailing-bits block runs: `payload_extension_present()` is FALSE
    //   for a CREI body (the current position already *is* the position of
    //   the trailing `sei_payload_bit_equal_to_one`, since §D.10.1 wrote
    //   no syntax bits), so no `sei_reserved_payload_extension_data` is
    //   read; then a single `sei_payload_bit_equal_to_one` (`f(1)`, == 1)
    //   followed by `sei_payload_bit_equal_to_zero` bits to the byte
    //   boundary. The canonical one-byte encoding of that padding is
    //   `0x80`.
    //
    // §7.4.6 requires the consumed length to equal `payloadSize`, i.e. the
    // reader must end sitting exactly at the end of `payload`.
    if reader.bits_remaining() != 0 {
        // more_data_in_payload(): the padding block runs.
        // sei_payload_bit_equal_to_one
        if reader.u1()? != 1 {
            return Err(Error::invalid(ERR));
        }
        // sei_payload_bit_equal_to_zero* up to the byte boundary.
        while !reader.is_byte_aligned() {
            if reader.u1()? != 0 {
                return Err(Error::invalid(ERR));
            }
        }
        // No bytes may remain beyond the single padding run (§7.4.6).
        if reader.bits_remaining() != 0 {
            return Err(Error::invalid(ERR));
        }
    }

    Ok(ConstrainedRaslEncodingIndication)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `CONSTRAINED_RASL_ENCODING_INDICATION_PAYLOAD_TYPE` matches the
    /// §D.2 dispatch value.
    #[test]
    fn payload_type_is_207() {
        assert_eq!(CONSTRAINED_RASL_ENCODING_INDICATION_PAYLOAD_TYPE, 207);
    }

    /// An empty payload (`payloadSize == 0`) is the canonical CREI body:
    /// no syntax elements, no trailing bits.
    #[test]
    fn empty_payload_is_valid() {
        let crei = parse_constrained_rasl_encoding_indication(&[]).unwrap();
        assert_eq!(crei, ConstrainedRaslEncodingIndication);
    }

    /// The §D.10.2 constraints are surfaced in spec order, four of them.
    #[test]
    fn constraints_are_surfaced_in_spec_order() {
        let crei = ConstrainedRaslEncodingIndication;
        let c = crei.constraints();
        assert_eq!(c.len(), 4);
        assert!(c[0].contains("ph_dmvr_disabled_flag"));
        assert!(c[1].contains("pps_ref_wraparound_enabled_flag"));
        assert!(c[2].contains("cclm_mode_flag"));
        assert!(c[3].contains("collocated reference picture"));
        // The associated const and the accessor agree.
        assert_eq!(c, &CREI_CONSTRAINTS);
    }

    /// A payload that carries §D.2.1 extension/padding bytes is accepted
    /// when the padding is the canonical `1` then `0`-to-byte-boundary
    /// encoding. A single `0x80` byte is `sei_payload_bit_equal_to_one`
    /// (`1`) followed by seven `sei_payload_bit_equal_to_zero` bits.
    #[test]
    fn canonical_trailing_padding_byte_is_valid() {
        let crei = parse_constrained_rasl_encoding_indication(&[0x80]).unwrap();
        assert_eq!(crei, ConstrainedRaslEncodingIndication);
    }

    /// A trailing padding byte whose `sei_payload_bit_equal_to_one` is `0`
    /// is rejected (§D.2.1).
    #[test]
    fn trailing_byte_with_zero_one_bit_is_rejected() {
        // 0x00: first bit is 0, not the required sei_payload_bit_equal_to_one.
        assert!(parse_constrained_rasl_encoding_indication(&[0x00]).is_err());
    }

    /// A trailing padding byte whose `sei_payload_bit_equal_to_zero` pad
    /// is non-zero is rejected (§D.2.1).
    #[test]
    fn trailing_byte_with_nonzero_pad_is_rejected() {
        // 0xC0: 1 (one-bit) then 1 in the next pad position — must be 0.
        assert!(parse_constrained_rasl_encoding_indication(&[0xC0]).is_err());
    }

    /// Leftover bytes beyond the single §D.2.1 framing byte are rejected
    /// as a length mismatch (§7.4.6): the framing ends after the first
    /// byte but a second byte remains.
    #[test]
    fn extra_trailing_byte_is_rejected() {
        assert!(parse_constrained_rasl_encoding_indication(&[0x80, 0x80]).is_err());
    }

    /// The marker type is `Copy` + `Default`, so it can be stored and
    /// cloned trivially by a higher-level CVS-state tracker.
    #[test]
    fn marker_is_copy_and_default() {
        let a = ConstrainedRaslEncodingIndication;
        let b = a; // Copy
        assert_eq!(a, b);
        let d: ConstrainedRaslEncodingIndication = Default::default();
        assert_eq!(d, a);
    }
}
