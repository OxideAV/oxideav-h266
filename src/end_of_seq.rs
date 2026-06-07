//! VVC End of Sequence RBSP parser (§7.3.2.11 / §7.4.3.11).
//!
//! An End of Sequence NAL unit (NAL type 21, `EosNut` per Table 5)
//! signals to the decoder that the next subsequent PU in the same
//! layer (if any) shall be an IRAP or GDR PU. §7.4.3.11 states
//! explicitly: *"The syntax content of the SODB and RBSP for the EOS
//! RBSP are empty."*
//!
//! Syntax (§7.3.2.11):
//!
//! ```text
//!   end_of_seq_rbsp() {                                              Descriptor
//!   }
//! ```
//!
//! The §7.3.2.11 table has zero syntax elements — there is no
//! `rbsp_trailing_bits()` invocation at the tail, in contrast with
//! §7.3.2.10 (AUD) / §7.3.2.13 (filler). The on-wire RBSP for an EOS
//! NAL unit is therefore zero bytes after the 2-byte NAL header
//! (§7.3.1.2) once emulation-prevention bytes have been stripped.
//!
//! ## Semantic side-effects (informative)
//!
//! §7.4.3.11 attaches two consequences to a present EOS NAL unit:
//!
//! * The next subsequent PU in the same layer as the EOS NAL unit
//!   (if any) shall be an IRAP or GDR PU.
//! * For each layer that has the EOS NAL unit's layer as a reference
//!   layer, the first picture in that layer in decoding order in an
//!   AU following the EOS NAL unit's AU shall be a CLVSS picture.
//!
//! Those constraints are reference-list / DPB bookkeeping concerns
//! and are out of scope for this parser; the parser's only job is to
//! validate that the RBSP body is empty, since any non-empty content
//! would indicate an upstream framing bug.

use oxideav_core::{Error, Result};

/// Parsed End of Sequence RBSP contents (§7.3.2.11).
///
/// The §7.3.2.11 syntax table is empty, so the struct itself is a
/// unit-shaped marker — it exists so the parse function can return a
/// typed value (matching the shape used by [`crate::aud`] /
/// [`crate::filler_data`] / etc.) and so callers can pattern-match
/// against a name rather than an `()`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EndOfSeq;

/// Parse an End of Sequence RBSP (bytes after the 2-byte NAL header,
/// with emulation-prevention bytes already stripped).
///
/// §7.4.3.11 mandates that the SODB and RBSP for the EOS NAL unit
/// are empty. The parser therefore accepts only a zero-length RBSP
/// and rejects any byte at all — including a stray `0x80`
/// `rbsp_trailing_bits()` byte, which is the canonical encoding when
/// the syntax table appends one but is absent from §7.3.2.11.
///
/// This is stricter than a conformance decoder's tolerance — §7.4.3.11
/// has no "ignore" clause for non-empty EOS bodies — but matches the
/// project-wide pattern of rejecting framing bugs at parse time so
/// they surface here rather than silently corrupting downstream
/// reference-list logic.
pub fn parse_end_of_seq(rbsp: &[u8]) -> Result<EndOfSeq> {
    if !rbsp.is_empty() {
        return Err(Error::invalid(
            "h266 end_of_seq_rbsp: §7.3.2.11 specifies an empty RBSP",
        ));
    }
    Ok(EndOfSeq)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Canonical case — zero-byte RBSP.
    #[test]
    fn empty_rbsp_is_accepted() {
        assert_eq!(parse_end_of_seq(&[]).unwrap(), EndOfSeq);
    }

    /// A single byte (e.g. someone mistakenly appended an
    /// `rbsp_trailing_bits()` byte) is rejected — §7.3.2.11 has no
    /// trailing-bits invocation.
    #[test]
    fn trailing_bits_byte_is_rejected() {
        assert!(parse_end_of_seq(&[0x80]).is_err());
    }

    /// Any non-zero byte is rejected.
    #[test]
    fn arbitrary_single_byte_is_rejected() {
        assert!(parse_end_of_seq(&[0x00]).is_err());
        assert!(parse_end_of_seq(&[0xFF]).is_err());
        assert!(parse_end_of_seq(&[0x55]).is_err());
    }

    /// Multi-byte payloads are rejected.
    #[test]
    fn multi_byte_payload_is_rejected() {
        assert!(parse_end_of_seq(&[0x00, 0x00]).is_err());
        assert!(parse_end_of_seq(&[0xFF, 0x80]).is_err());
        assert!(parse_end_of_seq(&[0x01, 0x02, 0x03, 0x04, 0x05]).is_err());
    }

    /// The struct is `Default + Copy + Eq` so callers can stash it in
    /// option/result chains without manual construction.
    #[test]
    fn marker_struct_is_default_constructible() {
        let a: EndOfSeq = EndOfSeq;
        let b: EndOfSeq = Default::default();
        assert_eq!(a, b);
    }
}
