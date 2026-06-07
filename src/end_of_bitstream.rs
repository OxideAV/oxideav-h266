//! VVC End of Bitstream RBSP parser (§7.3.2.12 / §7.4.3.12).
//!
//! An End of Bitstream NAL unit (NAL type 22, `EobNut` per Table 5)
//! indicates that no additional NAL units are present in the
//! bitstream subsequent to this NAL unit in decoding order. §7.4.3.12
//! states explicitly: *"The syntax content of the SODB and RBSP for
//! the EOB RBSP are empty."*
//!
//! Syntax (§7.3.2.12):
//!
//! ```text
//!   end_of_bitstream_rbsp() {                                        Descriptor
//!   }
//! ```
//!
//! The §7.3.2.12 table has zero syntax elements — there is no
//! `rbsp_trailing_bits()` invocation at the tail, in contrast with
//! §7.3.2.10 (AUD) / §7.3.2.13 (filler). The on-wire RBSP for an EOB
//! NAL unit is therefore zero bytes after the 2-byte NAL header
//! (§7.3.1.2) once emulation-prevention bytes have been stripped.
//!
//! ## Bitstream-termination semantics (informative)
//!
//! §7.4.3.12 attaches one consequence to a present EOB NAL unit:
//! no further NAL units may appear in the bitstream in decoding
//! order. Together with the §7.4.2.4.4 ordering rule that places
//! the EOB NAL unit at the end of the AU (and the end of the CVS,
//! per the §3 CVS definition), the EOB serves as a sentinel that a
//! framework-level NAL iterator can use to terminate its scan
//! without consulting reference-list / DPB state.
//!
//! That termination-marker behaviour is a framework-level concern
//! and is out of scope for this parser; the parser's only job is to
//! validate that the RBSP body is empty, since any non-empty content
//! would indicate an upstream framing bug.

use oxideav_core::{Error, Result};

/// Parsed End of Bitstream RBSP contents (§7.3.2.12).
///
/// The §7.3.2.12 syntax table is empty, so the struct itself is a
/// unit-shaped marker — it exists so the parse function can return a
/// typed value (matching the shape used by [`crate::aud`] /
/// [`crate::filler_data`] / [`crate::end_of_seq`]) and so callers can
/// pattern-match against a name rather than an `()`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EndOfBitstream;

/// Parse an End of Bitstream RBSP (bytes after the 2-byte NAL header,
/// with emulation-prevention bytes already stripped).
///
/// §7.4.3.12 mandates that the SODB and RBSP for the EOB NAL unit
/// are empty. The parser therefore accepts only a zero-length RBSP
/// and rejects any byte at all — including a stray `0x80`
/// `rbsp_trailing_bits()` byte, which is the canonical encoding when
/// the syntax table appends one but is absent from §7.3.2.12.
///
/// This is stricter than a conformance decoder's tolerance — §7.4.3.12
/// has no "ignore" clause for non-empty EOB bodies — but matches the
/// project-wide pattern of rejecting framing bugs at parse time so
/// they surface here rather than silently corrupting downstream
/// bitstream-termination logic.
pub fn parse_end_of_bitstream(rbsp: &[u8]) -> Result<EndOfBitstream> {
    if !rbsp.is_empty() {
        return Err(Error::invalid(
            "h266 end_of_bitstream_rbsp: §7.3.2.12 specifies an empty RBSP",
        ));
    }
    Ok(EndOfBitstream)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Canonical case — zero-byte RBSP.
    #[test]
    fn empty_rbsp_is_accepted() {
        assert_eq!(parse_end_of_bitstream(&[]).unwrap(), EndOfBitstream);
    }

    /// A single byte (e.g. someone mistakenly appended an
    /// `rbsp_trailing_bits()` byte) is rejected — §7.3.2.12 has no
    /// trailing-bits invocation.
    #[test]
    fn trailing_bits_byte_is_rejected() {
        assert!(parse_end_of_bitstream(&[0x80]).is_err());
    }

    /// Any non-zero byte is rejected.
    #[test]
    fn arbitrary_single_byte_is_rejected() {
        assert!(parse_end_of_bitstream(&[0x00]).is_err());
        assert!(parse_end_of_bitstream(&[0xFF]).is_err());
        assert!(parse_end_of_bitstream(&[0x55]).is_err());
    }

    /// Multi-byte payloads are rejected.
    #[test]
    fn multi_byte_payload_is_rejected() {
        assert!(parse_end_of_bitstream(&[0x00, 0x00]).is_err());
        assert!(parse_end_of_bitstream(&[0xFF, 0x80]).is_err());
        assert!(parse_end_of_bitstream(&[0x01, 0x02, 0x03, 0x04, 0x05]).is_err());
    }

    /// The struct is `Default + Copy + Eq` so callers can stash it in
    /// option/result chains without manual construction.
    #[test]
    fn marker_struct_is_default_constructible() {
        let a: EndOfBitstream = EndOfBitstream;
        let b: EndOfBitstream = Default::default();
        assert_eq!(a, b);
    }

    /// The EOS and EOB markers are distinct types — even though their
    /// underlying parsers share the empty-RBSP shape, they classify
    /// different NAL unit types (21 / EOS_NUT vs 22 / EOB_NUT) and
    /// have different downstream consequences per §7.4.3.11 vs
    /// §7.4.3.12. Confirm they don't accidentally collapse to the
    /// same type.
    #[test]
    fn end_of_bitstream_is_a_distinct_marker_type() {
        // The two markers carry no data, but the type system keeps
        // them apart so a caller cannot accidentally fan one into the
        // other's downstream pipeline. The check is intentionally
        // shaped so the test would fail to compile (rather than fail
        // at runtime) if a future refactor collapsed the marker
        // structs into a single shared type via `type` alias.
        fn assert_eob(_: EndOfBitstream) {}
        fn assert_eos(_: crate::end_of_seq::EndOfSeq) {}
        assert_eob(EndOfBitstream);
        assert_eos(crate::end_of_seq::EndOfSeq);
    }
}
