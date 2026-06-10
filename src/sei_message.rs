//! VVC SEI message header parser (§7.3.6 / §7.4.6).
//!
//! Supplemental Enhancement Information (SEI) is carried in
//! Prefix-SEI (`PrefixSeiNut`, NAL type 23) and Suffix-SEI
//! (`SuffixSeiNut`, NAL type 24) NAL units. The §7.3.2.9 `sei_rbsp()`
//! structure wraps a `do { sei_message() } while( more_rbsp_data() )`
//! loop terminated by `rbsp_trailing_bits()`; this module implements
//! the single-message `sei_message()` *header* of §7.3.6 — the two
//! byte-accumulation loops that derive `payloadType` and `payloadSize`
//! — and isolates the `payloadSize` payload bytes without interpreting
//! them (the per-type `sei_payload()` body lives in Annex D and is a
//! later increment).
//!
//! Syntax (§7.3.6):
//!
//! ```text
//!   sei_message() {
//!     payloadType = 0
//!     do {
//!       payload_type_byte                                           u(8)
//!       payloadType += payload_type_byte
//!     } while( payload_type_byte == 0xFF )
//!     payloadSize = 0
//!     do {
//!       payload_size_byte                                           u(8)
//!       payloadSize += payload_size_byte
//!     } while( payload_size_byte == 0xFF )
//!     sei_payload( payloadType, payloadSize )
//!   }
//! ```
//!
//! §7.4.6: `payload_type_byte` is a byte of the payload type and
//! `payload_size_byte` is a byte of the payload size of an SEI message.
//! Both accumulate by simple addition across as many `0xFF`
//! continuation bytes as the encoder emitted (the last non-`0xFF` byte
//! terminates each loop). §7.4.6 further notes that the derived
//! `payloadSize` is specified in RBSP bytes and "shall be equal to the
//! number of RBSP bytes in the SEI message payload" — emulation-
//! prevention bytes are already stripped from the RBSP this parser
//! consumes, so the `payloadSize` count maps directly onto a byte slice
//! of the post-emulation-removal buffer.
//!
//! Because both loops are byte-granular and an SEI message begins
//! byte-aligned within `sei_rbsp()`, the parser operates directly on
//! the RBSP byte slice (no `BitReader` is required) and returns the
//! number of bytes consumed so a caller can advance to the next
//! `sei_message()` in the §7.3.2.9 loop.

use oxideav_core::{Error, Result};

/// One parsed `sei_message()` header (§7.3.6) plus a borrow of its
/// payload bytes.
///
/// The lifetime ties the `payload` slice to the source RBSP buffer so
/// no copy is made; the per-type interpretation of those bytes
/// (Annex D `sei_payload()`) is intentionally out of scope for this
/// header-level parser.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SeiMessage<'a> {
    /// `payloadType` — the accumulated SEI payload type (§7.4.6).
    /// Derived as the sum of every `payload_type_byte` in the first
    /// `do/while` loop.
    pub payload_type: u32,
    /// `payloadSize` — the accumulated SEI payload size in RBSP bytes
    /// (§7.4.6). Derived as the sum of every `payload_size_byte` in the
    /// second `do/while` loop; equals `payload.len()`.
    pub payload_size: u32,
    /// The `payloadSize` payload bytes (the `sei_payload()` argument
    /// region), borrowed from the source RBSP.
    pub payload: &'a [u8],
    /// Number of source bytes this message occupies in the RBSP: the
    /// `payload_type_byte` run + the `payload_size_byte` run + the
    /// payload itself. Lets a caller advance to the next
    /// `sei_message()` in the §7.3.2.9 `sei_rbsp()` loop.
    pub consumed_bytes: usize,
}

/// Accumulate a §7.3.6 byte-extension run starting at `offset`.
///
/// Reads `u(8)` bytes, summing them into the running total, until a
/// byte that is not `0xFF` terminates the `do/while` loop (the loop
/// body always runs at least once). Returns the accumulated value and
/// the number of bytes read. The sum is computed in `u64` to avoid an
/// overflow on a pathological all-`0xFF` run before the `u32` range
/// check is applied by the caller.
fn accumulate_extension_bytes(data: &[u8], offset: usize, what: &str) -> Result<(u64, usize)> {
    let mut total: u64 = 0;
    let mut idx = offset;
    loop {
        let byte = *data.get(idx).ok_or_else(|| {
            Error::invalid(match what {
                "type" => "h266 sei_message: truncated payload_type_byte run (§7.3.6)",
                _ => "h266 sei_message: truncated payload_size_byte run (§7.3.6)",
            })
        })?;
        idx += 1;
        total += byte as u64;
        // The do/while terminates on the first non-0xFF byte.
        if byte != 0xFF {
            break;
        }
    }
    Ok((total, idx - offset))
}

/// Parse one `sei_message()` (§7.3.6) starting at the front of `rbsp`.
///
/// `rbsp` is the SEI RBSP byte stream (post-NAL-header, with
/// emulation-prevention bytes already stripped), positioned at the
/// start of a `sei_message()`. The parser walks the two byte-extension
/// loops to recover `payloadType` / `payloadSize`, then borrows the
/// `payloadSize` payload bytes that follow.
///
/// Errors:
/// * an empty input, or a `payload_type_byte` / `payload_size_byte` run
///   that runs off the end of `rbsp`, is rejected as truncated;
/// * a `payloadSize` that exceeds the bytes remaining after the size
///   loop is rejected (the §7.4.6 "shall be equal to the number of RBSP
///   bytes in the SEI message payload" constraint cannot hold);
/// * a `payloadType` accumulation that overflows the `u32` range is
///   rejected.
pub fn parse_sei_message(rbsp: &[u8]) -> Result<SeiMessage<'_>> {
    if rbsp.is_empty() {
        return Err(Error::invalid("h266 sei_message: empty RBSP (§7.3.6)"));
    }

    // payloadType = sum of payload_type_byte while == 0xFF.
    let (payload_type_sum, type_len) = accumulate_extension_bytes(rbsp, 0, "type")?;
    let payload_type = u32::try_from(payload_type_sum)
        .map_err(|_| Error::invalid("h266 sei_message: payloadType exceeds u32 range (§7.4.6)"))?;

    // payloadSize = sum of payload_size_byte while == 0xFF.
    let (payload_size_sum, size_len) = accumulate_extension_bytes(rbsp, type_len, "size")?;
    let payload_size = u32::try_from(payload_size_sum)
        .map_err(|_| Error::invalid("h266 sei_message: payloadSize exceeds u32 range (§7.4.6)"))?;

    let header_len = type_len + size_len;
    let payload_size_usize = payload_size as usize;
    let available = rbsp.len() - header_len;
    if payload_size_usize > available {
        return Err(Error::invalid(
            "h266 sei_message: payloadSize exceeds bytes remaining in RBSP (§7.4.6)",
        ));
    }

    let payload = &rbsp[header_len..header_len + payload_size_usize];
    Ok(SeiMessage {
        payload_type,
        payload_size,
        payload,
        consumed_bytes: header_len + payload_size_usize,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal single-byte type + single-byte size message: type 1,
    /// size 0, no payload bytes.
    #[test]
    fn single_byte_type_and_size_empty_payload() {
        let rbsp = [0x01u8, 0x00];
        let msg = parse_sei_message(&rbsp).unwrap();
        assert_eq!(msg.payload_type, 1);
        assert_eq!(msg.payload_size, 0);
        assert!(msg.payload.is_empty());
        assert_eq!(msg.consumed_bytes, 2);
    }

    /// type 5, size 3, with a 3-byte payload that is borrowed verbatim.
    #[test]
    fn type_size_with_payload() {
        let rbsp = [0x05u8, 0x03, 0xAA, 0xBB, 0xCC];
        let msg = parse_sei_message(&rbsp).unwrap();
        assert_eq!(msg.payload_type, 5);
        assert_eq!(msg.payload_size, 3);
        assert_eq!(msg.payload, &[0xAA, 0xBB, 0xCC]);
        assert_eq!(msg.consumed_bytes, 5);
    }

    /// `payloadType` extension: one `0xFF` continuation byte then a
    /// `0x0A` terminator accumulates to `255 + 10 = 265`.
    #[test]
    fn payload_type_extension_accumulates() {
        let rbsp = [0xFFu8, 0x0A, 0x00];
        let msg = parse_sei_message(&rbsp).unwrap();
        assert_eq!(msg.payload_type, 265);
        assert_eq!(msg.payload_size, 0);
        assert_eq!(msg.consumed_bytes, 3);
    }

    /// `payloadSize` extension: a `0xFF` continuation then `0x02`
    /// accumulates to `255 + 2 = 257`; supply that many payload bytes.
    #[test]
    fn payload_size_extension_accumulates() {
        let mut rbsp = vec![0x07u8, 0xFF, 0x02];
        rbsp.resize(rbsp.len() + 257, 0x5Au8);
        let msg = parse_sei_message(&rbsp).unwrap();
        assert_eq!(msg.payload_type, 7);
        assert_eq!(msg.payload_size, 257);
        assert_eq!(msg.payload.len(), 257);
        assert!(msg.payload.iter().all(|&b| b == 0x5A));
        assert_eq!(msg.consumed_bytes, 3 + 257);
    }

    /// Both loops extend simultaneously: type = 255+255+1 = 511,
    /// size = 255+1 = 256.
    #[test]
    fn both_loops_extend() {
        let mut rbsp = vec![0xFFu8, 0xFF, 0x01, 0xFF, 0x01];
        rbsp.resize(rbsp.len() + 256, 0x11u8);
        let msg = parse_sei_message(&rbsp).unwrap();
        assert_eq!(msg.payload_type, 511);
        assert_eq!(msg.payload_size, 256);
        assert_eq!(msg.payload.len(), 256);
        assert_eq!(msg.consumed_bytes, 5 + 256);
    }

    /// Empty RBSP is rejected.
    #[test]
    fn empty_rbsp_is_rejected() {
        assert!(parse_sei_message(&[]).is_err());
    }

    /// A `payload_type_byte` run that never terminates before the end
    /// of the buffer (all `0xFF`) is rejected as truncated.
    #[test]
    fn unterminated_type_run_is_rejected() {
        let rbsp = [0xFFu8, 0xFF, 0xFF];
        assert!(parse_sei_message(&rbsp).is_err());
    }

    /// A `payload_size_byte` run that runs off the end after a valid
    /// type byte is rejected as truncated.
    #[test]
    fn unterminated_size_run_is_rejected() {
        // type = 0x01 terminates, then the size loop hits EOF mid-run.
        let rbsp = [0x01u8, 0xFF];
        assert!(parse_sei_message(&rbsp).is_err());
    }

    /// A `payloadSize` larger than the bytes remaining in the RBSP is
    /// rejected per §7.4.6.
    #[test]
    fn payload_size_exceeding_remaining_is_rejected() {
        // type 0, size 4, but only 2 payload bytes supplied.
        let rbsp = [0x00u8, 0x04, 0xAA, 0xBB];
        assert!(parse_sei_message(&rbsp).is_err());
    }

    /// `payloadSize` exactly equal to the remaining bytes is accepted
    /// and leaves no trailing bytes unconsumed.
    #[test]
    fn payload_size_exact_remaining_is_accepted() {
        let rbsp = [0x00u8, 0x02, 0xAA, 0xBB];
        let msg = parse_sei_message(&rbsp).unwrap();
        assert_eq!(msg.payload, &[0xAA, 0xBB]);
        assert_eq!(msg.consumed_bytes, rbsp.len());
    }

    /// `consumed_bytes` lets a caller chain a second `sei_message()`
    /// from the same buffer (the §7.3.2.9 `sei_rbsp()` loop shape).
    #[test]
    fn consumed_bytes_advances_to_next_message() {
        // Message A: type 1, size 1, payload [0xAA].
        // Message B: type 2, size 0.
        let rbsp = [0x01u8, 0x01, 0xAA, 0x02, 0x00];
        let a = parse_sei_message(&rbsp).unwrap();
        assert_eq!(a.payload_type, 1);
        assert_eq!(a.payload, &[0xAA]);
        let b = parse_sei_message(&rbsp[a.consumed_bytes..]).unwrap();
        assert_eq!(b.payload_type, 2);
        assert_eq!(b.payload_size, 0);
    }

    /// Round-trip a range of single-byte (`< 0xFF`) type/size values to
    /// confirm the non-extended path is exact.
    #[test]
    fn single_byte_round_trip_range() {
        for t in 0u8..=0xFE {
            for s in [0u8, 1, 2, 0xFE] {
                let mut rbsp = vec![t, s];
                rbsp.resize(rbsp.len() + s as usize, 0x33u8);
                let msg = parse_sei_message(&rbsp).unwrap();
                assert_eq!(msg.payload_type, t as u32);
                assert_eq!(msg.payload_size, s as u32);
                assert_eq!(msg.payload.len(), s as usize);
            }
        }
    }
}
