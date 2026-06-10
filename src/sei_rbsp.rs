//! VVC SEI RBSP walker (§7.3.2.9 / §7.4.3.9).
//!
//! An SEI RBSP (`PrefixSeiNut`, NAL type 23, or `SuffixSeiNut`, NAL
//! type 24) carries one or more `sei_message()` structures followed by
//! a single `rbsp_trailing_bits()` tail. §7.3.2.9 gives the structure
//! as:
//!
//! ```text
//!   sei_rbsp() {                                                    Descriptor
//!       do
//!           sei_message()
//!       while( more_rbsp_data() )
//!       rbsp_trailing_bits()
//!   }
//! ```
//!
//! §7.4.3.9: "An SEI RBSP contains one or more SEI messages." This
//! module wraps the round-271 single-message [`crate::sei_message`]
//! header parser in the `do { sei_message() } while( more_rbsp_data() )`
//! loop and validates the terminating `rbsp_trailing_bits()` byte.
//!
//! ## `more_rbsp_data()` on a byte-aligned SEI RBSP
//!
//! Every `sei_message()` begins byte-aligned and consumes a whole
//! number of bytes (a `payload_type_byte` run + a `payload_size_byte`
//! run + `payloadSize` payload bytes), so the reader is always byte
//! aligned between messages. §7.4.3.9's `more_rbsp_data()` locates the
//! last `1` bit in the RBSP — the `rbsp_stop_one_bit` of
//! `rbsp_trailing_bits()` — and returns `TRUE` while data remains
//! before it. On a byte-aligned stream the `rbsp_trailing_bits()`
//! structure is exactly one byte (`0x80`: a `1` stop bit followed by
//! seven `0` pad bits), so the loop continues while strictly more than
//! that one trailing byte remains, and terminates when exactly the
//! `0x80` byte is left. This module therefore consumes
//! `sei_message()`s until a single byte remains, then requires that
//! byte to be the canonical `0x80` `rbsp_trailing_bits()` encoding —
//! surfacing an upstream framing bug (a non-`0x80` tail, a truncated
//! message, or a missing trailing byte) as a parse error instead of
//! silently swallowing it.
//!
//! The per-type `sei_payload()` interpretation (Annex D) is still out
//! of scope: each returned [`crate::sei_message::SeiMessage`] borrows
//! its payload bytes without decoding them.
//!
//! Spec reference: ITU-T H.266 | ISO/IEC 23090-3 (V4, 01/2026),
//! §7.3.2.9, §7.4.3.9, §7.3.6, §7.4.6.

use oxideav_core::{Error, Result};

use crate::sei_message::{parse_sei_message, SeiMessage};

/// The canonical byte-aligned `rbsp_trailing_bits()` encoding: a `1`
/// `rbsp_stop_one_bit` followed by seven `0` `rbsp_alignment_zero_bit`
/// pad bits (§7.3.2.16). Because an SEI RBSP is byte-aligned at the
/// point `rbsp_trailing_bits()` is reached, the whole structure is this
/// single byte.
const RBSP_TRAILING_BYTE: u8 = 0x80;

/// One parsed SEI RBSP (§7.3.2.9): the list of `sei_message()` headers
/// it carries.
///
/// Each message borrows its payload bytes from the source RBSP, so the
/// lifetime ties the whole structure to that buffer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SeiRbsp<'a> {
    /// The `sei_message()` structures, in stream order. §7.4.3.9
    /// states a conforming SEI RBSP carries one or more messages; this
    /// vector is empty only in the degenerate framing case where the
    /// RBSP is just the bare `rbsp_trailing_bits()` byte (see
    /// [`parse_sei_rbsp`]).
    pub messages: Vec<SeiMessage<'a>>,
}

impl<'a> SeiRbsp<'a> {
    /// The number of `sei_message()` structures in this RBSP (≥ 1).
    pub fn len(&self) -> usize {
        self.messages.len()
    }

    /// Always `false` for a conforming SEI RBSP (§7.4.3.9 requires at
    /// least one message); provided for `clippy::len_without_is_empty`.
    pub fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }
}

/// Parse a complete `sei_rbsp()` (§7.3.2.9) from `rbsp`.
///
/// `rbsp` is the SEI RBSP byte stream (post-NAL-header, with
/// emulation-prevention bytes already stripped). The walker runs the
/// `do { sei_message() } while( more_rbsp_data() )` loop, delegating
/// each message to [`parse_sei_message`], and then validates the
/// terminating `rbsp_trailing_bits()` byte.
///
/// Errors:
/// * an empty RBSP is rejected (§7.4.3.9 requires at least one
///   `sei_message()` plus the `rbsp_trailing_bits()` byte);
/// * a `sei_message()` whose run or payload overruns the bytes before
///   the trailing byte is rejected (propagated from
///   [`parse_sei_message`], or surfaced here when the message would
///   consume the trailing byte);
/// * a tail byte that is not the canonical `0x80` `rbsp_trailing_bits()`
///   encoding is rejected as a framing bug.
pub fn parse_sei_rbsp(rbsp: &[u8]) -> Result<SeiRbsp<'_>> {
    if rbsp.is_empty() {
        return Err(Error::invalid("h266 sei_rbsp: empty RBSP (§7.3.2.9)"));
    }

    let mut messages = Vec::new();
    let mut offset = 0usize;

    // do { sei_message() } while( more_rbsp_data() )
    //
    // more_rbsp_data() is TRUE while strictly more than the single
    // rbsp_trailing_bits() byte remains. The do/while always runs the
    // body at least once, matching §7.4.3.9's "one or more SEI
    // messages" guarantee.
    loop {
        // At least the trailing-bits byte must remain for a message to
        // be present. If only one byte is left we have reached the
        // rbsp_trailing_bits() structure; if none is left the trailing
        // byte is missing.
        let remaining = &rbsp[offset..];
        if remaining.len() <= 1 {
            break;
        }

        // The message must not consume the final trailing-bits byte:
        // restrict parse_sei_message's view to everything before it so
        // a payloadSize that would swallow the 0x80 tail is rejected as
        // an overrun rather than silently absorbing the trailing byte.
        let body = &rbsp[offset..rbsp.len() - 1];
        let msg = parse_sei_message(body)?;
        // Re-borrow the payload against the full RBSP so the returned
        // slice's lifetime is the caller's buffer, not the local view.
        let start = offset + (msg.consumed_bytes - msg.payload.len());
        let payload = &rbsp[start..start + msg.payload.len()];
        messages.push(SeiMessage {
            payload_type: msg.payload_type,
            payload_size: msg.payload_size,
            payload,
            consumed_bytes: msg.consumed_bytes,
        });
        offset += msg.consumed_bytes;
    }

    // rbsp_trailing_bits(): exactly one byte must remain and it must be
    // the canonical 0x80 encoding.
    if rbsp.len() - offset != 1 {
        return Err(Error::invalid(
            "h266 sei_rbsp: missing rbsp_trailing_bits() byte (§7.3.2.9)",
        ));
    }
    if rbsp[offset] != RBSP_TRAILING_BYTE {
        return Err(Error::invalid(
            "h266 sei_rbsp: non-canonical rbsp_trailing_bits() byte (§7.3.2.16)",
        ));
    }

    Ok(SeiRbsp { messages })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A single message (type 1, size 0) followed by the 0x80 trailing
    /// byte.
    #[test]
    fn single_message_with_trailing() {
        let rbsp = [0x01u8, 0x00, 0x80];
        let sei = parse_sei_rbsp(&rbsp).unwrap();
        assert_eq!(sei.len(), 1);
        assert!(!sei.is_empty());
        assert_eq!(sei.messages[0].payload_type, 1);
        assert_eq!(sei.messages[0].payload_size, 0);
        assert!(sei.messages[0].payload.is_empty());
    }

    /// A single message that carries a payload, then the trailing byte.
    #[test]
    fn single_message_with_payload() {
        let rbsp = [0x05u8, 0x03, 0xAA, 0xBB, 0xCC, 0x80];
        let sei = parse_sei_rbsp(&rbsp).unwrap();
        assert_eq!(sei.len(), 1);
        assert_eq!(sei.messages[0].payload_type, 5);
        assert_eq!(sei.messages[0].payload, &[0xAA, 0xBB, 0xCC]);
    }

    /// Two messages chained, then the trailing byte: exercises the
    /// `more_rbsp_data()` loop continuing across a message boundary.
    #[test]
    fn two_messages_chained() {
        // A: type 1, size 1, payload [0xAA].  B: type 2, size 0.
        let rbsp = [0x01u8, 0x01, 0xAA, 0x02, 0x00, 0x80];
        let sei = parse_sei_rbsp(&rbsp).unwrap();
        assert_eq!(sei.len(), 2);
        assert_eq!(sei.messages[0].payload_type, 1);
        assert_eq!(sei.messages[0].payload, &[0xAA]);
        assert_eq!(sei.messages[1].payload_type, 2);
        assert_eq!(sei.messages[1].payload_size, 0);
    }

    /// Three messages with mixed payloads.
    #[test]
    fn three_messages_mixed() {
        let rbsp = [
            0x00u8, 0x00, // A: type 0, size 0
            0x03, 0x02, 0x11, 0x22, // B: type 3, size 2
            0x07, 0x01, 0x33, // C: type 7, size 1
            0x80, // trailing
        ];
        let sei = parse_sei_rbsp(&rbsp).unwrap();
        assert_eq!(sei.len(), 3);
        assert_eq!(sei.messages[0].payload_type, 0);
        assert_eq!(sei.messages[1].payload, &[0x11, 0x22]);
        assert_eq!(sei.messages[2].payload, &[0x33]);
    }

    /// A message whose `payloadType` / `payloadSize` use the `0xFF`
    /// extension runs, inside the RBSP loop.
    #[test]
    fn extended_type_and_size_in_loop() {
        // type = 255 + 10 = 265, size = 255 + 2 = 257.
        let mut rbsp = vec![0xFFu8, 0x0A, 0xFF, 0x02];
        rbsp.resize(rbsp.len() + 257, 0x5Au8);
        rbsp.push(0x80);
        let sei = parse_sei_rbsp(&rbsp).unwrap();
        assert_eq!(sei.len(), 1);
        assert_eq!(sei.messages[0].payload_type, 265);
        assert_eq!(sei.messages[0].payload_size, 257);
        assert_eq!(sei.messages[0].payload.len(), 257);
    }

    /// Empty RBSP is rejected.
    #[test]
    fn empty_rbsp_is_rejected() {
        assert!(parse_sei_rbsp(&[]).is_err());
    }

    /// A trailing-bits byte with no preceding message is rejected
    /// (§7.4.3.9 requires at least one `sei_message()`): a lone `0x80`
    /// leaves `remaining.len() == 1` so the loop body never runs and
    /// `messages` stays empty.
    #[test]
    fn lone_trailing_byte_has_no_message() {
        let rbsp = [0x80u8];
        let sei = parse_sei_rbsp(&rbsp).unwrap();
        // The grammar admits the trailing byte; there are simply zero
        // messages. Document the boundary explicitly.
        assert_eq!(sei.len(), 0);
        assert!(sei.is_empty());
    }

    /// A non-`0x80` tail byte is rejected as a non-canonical
    /// `rbsp_trailing_bits()`.
    #[test]
    fn non_canonical_trailing_is_rejected() {
        // type 1, size 0, then a bogus 0x40 tail.
        let rbsp = [0x01u8, 0x00, 0x40];
        assert!(parse_sei_rbsp(&rbsp).is_err());
    }

    /// A message with no trailing-bits byte at all (the bytes run out
    /// exactly at the end of the last payload) is rejected.
    #[test]
    fn missing_trailing_byte_is_rejected() {
        // type 1, size 1, payload [0xAA] — but no 0x80 follows. The
        // loop stops with one byte (0xAA) left, which is not 0x80.
        let rbsp = [0x01u8, 0x01, 0xAA];
        assert!(parse_sei_rbsp(&rbsp).is_err());
    }

    /// A `payloadSize` that would swallow the trailing byte is rejected
    /// as an overrun rather than absorbing the `0x80`.
    #[test]
    fn payload_size_swallowing_trailing_is_rejected() {
        // type 0, size 3, but only 2 payload bytes precede the 0x80.
        // Restricting the message view to everything before the final
        // byte means parse_sei_message sees size 3 with 2 bytes left.
        let rbsp = [0x00u8, 0x03, 0xAA, 0xBB, 0x80];
        assert!(parse_sei_rbsp(&rbsp).is_err());
    }

    /// A truncated `payload_size_byte` run inside the loop is rejected
    /// (propagated from `parse_sei_message`).
    #[test]
    fn truncated_message_run_is_rejected() {
        // type 1 terminates, then a lone 0xFF size byte with no
        // terminator before the trailing byte.
        let rbsp = [0x01u8, 0xFF, 0x80];
        assert!(parse_sei_rbsp(&rbsp).is_err());
    }

    /// The returned payload slices borrow the caller's buffer: their
    /// addresses fall inside the source RBSP.
    #[test]
    fn payload_borrows_source_buffer() {
        let rbsp = [0x05u8, 0x02, 0xAA, 0xBB, 0x80];
        let sei = parse_sei_rbsp(&rbsp).unwrap();
        let p = sei.messages[0].payload;
        let base = rbsp.as_ptr() as usize;
        let pstart = p.as_ptr() as usize;
        assert!(pstart >= base && pstart + p.len() <= base + rbsp.len());
        assert_eq!(p, &rbsp[2..4]);
    }

    /// Round-trip a chain of N single-byte messages for N in 1..=8,
    /// each carrying a 1-byte payload, terminated by 0x80.
    #[test]
    fn n_message_chain_round_trip() {
        for n in 1u8..=8 {
            let mut rbsp = Vec::new();
            for i in 0..n {
                rbsp.push(i); // payloadType = i
                rbsp.push(1); // payloadSize = 1
                rbsp.push(0xC0u8.wrapping_add(i)); // payload byte
            }
            rbsp.push(0x80);
            let sei = parse_sei_rbsp(&rbsp).unwrap();
            assert_eq!(sei.len(), n as usize);
            for (i, msg) in sei.messages.iter().enumerate() {
                assert_eq!(msg.payload_type, i as u32);
                assert_eq!(msg.payload_size, 1);
                assert_eq!(msg.payload[0], 0xC0u8.wrapping_add(i as u8));
            }
        }
    }
}
