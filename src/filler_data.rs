//! VVC Filler Data RBSP parser (§7.3.2.13 / §7.4.3.13).
//!
//! A Filler Data NAL unit (NAL type 25, `FdNut`) carries one or more
//! bytes of value `0xFF` followed by the trailing-bits byte that
//! `rbsp_trailing_bits()` produces for a byte-aligned reader. §7.4.3.13
//! is explicit that filler data has no normative decoding semantics:
//! the payload is for bit-rate / buffer-padding only and decoders may
//! discard it once parsed.
//!
//! Syntax (§7.3.2.13):
//!
//! ```text
//!   filler_data_rbsp() {
//!     while( next_bits(8) == 0xFF )
//!       fd_ff_byte                                                  f(8)
//!     rbsp_trailing_bits()
//!   }
//! ```
//!
//! After emulation-prevention byte removal the RBSP is therefore
//! a run of N×`0xFF` bytes followed by a single `0x80` byte (the
//! byte-aligned `rbsp_trailing_bits()` encoding: a `1` stop bit then
//! seven zero-bits of pad). The parser returns the count `N` so
//! callers that care about the on-wire padding budget (e.g. HRD
//! bookkeeping) can inspect it; callers that don't can ignore the
//! payload entirely.

use oxideav_core::{Error, Result};

/// Parsed filler-data RBSP contents.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FillerData {
    /// Number of `0xFF` (`fd_ff_byte`) bytes carried in the RBSP.
    ///
    /// May be zero — the spec's `while( next_bits(8) == 0xFF )` loop
    /// is permitted to never enter, in which case the RBSP collapses
    /// to the single `rbsp_trailing_bits()` byte.
    pub num_ff_bytes: u32,
}

/// Parse a Filler Data RBSP (bytes after the 2-byte NAL header, with
/// emulation-prevention bytes already stripped).
///
/// The parser consumes every leading `0xFF` byte (each one is the
/// `fd_ff_byte` produced by the §7.3.2.13 `while( next_bits(8) ==
/// 0xFF )` loop), then verifies that the final byte is exactly `0x80`
/// — the canonical byte-aligned encoding of `rbsp_trailing_bits()`:
/// the `1` stop bit followed by seven zero pad bits.
///
/// Any deviation — an interior byte that is neither `0xFF` nor the
/// terminating `0x80`, a missing trailing byte, an `0xFF`-only RBSP
/// with no trailing-bits byte at all, or a trailing-bits byte with
/// non-zero pad — is rejected as malformed input. The check is
/// stricter than a conformance-decoder's "ignore everything" rule
/// (§7.4.3.13 says filler has no normative semantics) on purpose:
/// it surfaces upstream framing bugs that would otherwise be silently
/// swallowed.
pub fn parse_filler_data(rbsp: &[u8]) -> Result<FillerData> {
    if rbsp.is_empty() {
        return Err(Error::invalid("h266 filler_data: empty RBSP"));
    }
    // The last byte must be the byte-aligned `rbsp_trailing_bits()`
    // encoding (0x80). Everything before it must be `fd_ff_byte` (0xFF).
    let (body, trailing) = rbsp.split_at(rbsp.len() - 1);
    if trailing[0] != 0x80 {
        return Err(Error::invalid(
            "h266 filler_data: rbsp_trailing_bits byte != 0x80",
        ));
    }
    for &b in body {
        if b != 0xFF {
            return Err(Error::invalid(
                "h266 filler_data: non-0xFF byte before rbsp_trailing_bits",
            ));
        }
    }
    Ok(FillerData {
        num_ff_bytes: body.len() as u32,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Zero `fd_ff_byte` bytes — the `while` loop never enters, the
    /// RBSP collapses to the single `rbsp_trailing_bits()` byte.
    #[test]
    fn zero_ff_bytes_just_trailing() {
        let rbsp = [0x80u8];
        let fd = parse_filler_data(&rbsp).unwrap();
        assert_eq!(fd.num_ff_bytes, 0);
    }

    /// One `fd_ff_byte` followed by trailing-bits.
    #[test]
    fn single_ff_byte() {
        let rbsp = [0xFFu8, 0x80];
        let fd = parse_filler_data(&rbsp).unwrap();
        assert_eq!(fd.num_ff_bytes, 1);
    }

    /// Multiple `fd_ff_byte` bytes survive the loop and count up.
    #[test]
    fn many_ff_bytes() {
        let rbsp = [0xFFu8, 0xFF, 0xFF, 0xFF, 0xFF, 0x80];
        let fd = parse_filler_data(&rbsp).unwrap();
        assert_eq!(fd.num_ff_bytes, 5);
    }

    /// Large padding run — sanity-check the loop count.
    #[test]
    fn large_filler_run() {
        let mut rbsp = vec![0xFFu8; 1024];
        rbsp.push(0x80);
        let fd = parse_filler_data(&rbsp).unwrap();
        assert_eq!(fd.num_ff_bytes, 1024);
    }

    /// Empty RBSP is rejected (the spec requires at least the
    /// `rbsp_trailing_bits()` byte).
    #[test]
    fn empty_rbsp_is_rejected() {
        assert!(parse_filler_data(&[]).is_err());
    }

    /// A trailing byte that is not the canonical `0x80` is rejected.
    /// The spec's `rbsp_trailing_bits()` always produces `0x80` from
    /// a byte-aligned position.
    #[test]
    fn non_canonical_trailing_byte_is_rejected() {
        // Final 0x40 has the stop bit in the wrong position.
        let rbsp = [0xFFu8, 0xFF, 0x40];
        assert!(parse_filler_data(&rbsp).is_err());
    }

    /// A non-`0xFF` byte before the trailing-bits byte is rejected as
    /// a framing bug.
    #[test]
    fn non_ff_body_byte_is_rejected() {
        let rbsp = [0xFFu8, 0xAA, 0xFF, 0x80];
        assert!(parse_filler_data(&rbsp).is_err());
    }

    /// A single `0xFF` byte with no trailing-bits byte (length == 1
    /// and content is `0xFF`) is rejected: the split would put `0xFF`
    /// in the trailing slot, which is not `0x80`.
    #[test]
    fn ff_without_trailing_is_rejected() {
        let rbsp = [0xFFu8];
        assert!(parse_filler_data(&rbsp).is_err());
    }

    /// Round-trip a synthesised buffer of `n` `fd_ff_byte` bytes
    /// across a useful range. Confirms the count matches the source
    /// for every n we care about.
    #[test]
    fn count_round_trip_range() {
        for n in 0u32..=32 {
            let mut rbsp = vec![0xFFu8; n as usize];
            rbsp.push(0x80);
            let fd = parse_filler_data(&rbsp).unwrap();
            assert_eq!(fd.num_ff_bytes, n);
        }
    }
}
