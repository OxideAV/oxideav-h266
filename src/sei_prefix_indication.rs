//! VVC SEI prefix indication SEI message parser (§D.9.1 / §D.9.2).
//!
//! The SEI prefix indication SEI message (`payloadType == 201`) carries
//! one or more *SEI prefix indications* for SEI messages of a particular
//! value of `payloadType`. Each prefix indication is a bit string that
//! follows the SEI payload syntax of that `payloadType` and contains a
//! number of complete syntax elements starting from the first syntax
//! element in the SEI payload — i.e. a starting prefix of an SEI payload
//! that one or more SEI messages of that type are expected to begin
//! with. It lets transport- or systems-layer elements decide, without a
//! full decode, whether a CVS is suitable for delivery (e.g. whether a
//! particular frame-unpacking capability is needed or whether a caption
//! language is present).
//!
//! Like the SEI manifest (`payloadType == 200`), this is one of the few
//! `sei_payload()` bodies specified directly in this Specification (most
//! other payload types defer to Rec. ITU-T H.274 | ISO/IEC 23002-7), so
//! it is parseable from its own payload bytes with no external
//! buffering-period / HRD context.
//!
//! Syntax (§D.9.1):
//!
//! ```text
//!   sei_prefix_indication( payloadSize ) {                            Descriptor
//!       prefix_sei_payload_type                                          u(16)
//!       num_sei_prefix_indications_minus1                                u(8)
//!       for( i = 0; i <= num_sei_prefix_indications_minus1; i++ ) {
//!           num_bits_in_prefix_indication_minus1[ i ]                    u(16)
//!           for( j = 0; j <= num_bits_in_prefix_indication_minus1[ i ]; j++ )
//!               sei_prefix_data_bit[ i ][ j ]                            u(1)
//!           while( !byte_aligned( ) )
//!               byte_alignment_bit_equal_to_one /* equal to 1 */         f(1)
//!       }
//!   }
//! ```
//!
//! §D.9.2 semantics modelled here:
//!
//! * `prefix_sei_payload_type` — the `payloadType` value of the SEI
//!   messages for which prefix indications are provided. (The §D.9.2
//!   cross-constraint against `manifest_sei_payload_type[ m ]` /
//!   `manifest_sei_description[ m ]` requires an applicable SEI manifest
//!   to be in hand, which is a higher-level CVS concern than this
//!   single-payload parser; the field is surfaced verbatim for that
//!   check.)
//! * `num_sei_prefix_indications_minus1` plus 1 — the number of SEI
//!   prefix indications carried.
//! * `num_bits_in_prefix_indication_minus1[ i ]` plus 1 — the number of
//!   bits in the i-th SEI prefix indication. Those bits
//!   (`sei_prefix_data_bit[ i ][ 0 .. ]`) follow the syntax of the SEI
//!   payload with `payloadType == prefix_sei_payload_type` and are
//!   preserved here as an opaque bit string (their per-`payloadType`
//!   interpretation is the consumer's job).
//! * `byte_alignment_bit_equal_to_one` — §D.9.2 requires each padding
//!   bit appended after an indication's data bits to be `1`; the parser
//!   reads and verifies them rather than discarding them, surfacing a
//!   non-`1` padding bit as a parse error.
//!
//! Each indication's data-bit run is followed by `byte_alignment(...)`
//! padding, so every indication ends on a byte boundary and the whole
//! body occupies a whole number of bytes. §7.4.6 requires the derived
//! `payloadSize` to equal the number of payload bytes; this parser
//! cross-checks the consumed length against `payloadSize` so a truncated
//! or over-long body surfaces as a parse error instead of silently
//! desynchronising the enclosing `sei_rbsp()` walk.
//!
//! Spec reference: ITU-T H.266 | ISO/IEC 23090-3 (V4, 01/2026),
//! §D.9.1, §D.9.2, §D.2 (`sei_payload()` dispatch), §7.4.6.
//!
//! No third-party VVC decoder source was consulted; the implementation
//! is spec-only and reads the payload through the crate's own
//! [`BitReader`].

use oxideav_core::{Error, Result};

use crate::bitreader::BitReader;

/// `payloadType` value that selects the SEI prefix indication body in
/// the §D.2 `sei_payload()` dispatch.
pub const SEI_PREFIX_INDICATION_PAYLOAD_TYPE: u32 = 201;

/// One SEI prefix indication: an opaque bit string of
/// `num_bits_in_prefix_indication_minus1[ i ] + 1` bits
/// (`sei_prefix_data_bit[ i ][ j ]` for `j` in `0 ..= minus1`).
///
/// The bits follow the SEI payload syntax of the enclosing message's
/// `prefix_sei_payload_type`; this struct stores them verbatim so a
/// consumer with the appropriate per-`payloadType` decoder can interpret
/// the leading syntax elements. They are stored MSB-first packed into
/// bytes, with [`PrefixIndication::num_bits`] giving the exact valid bit
/// count (the final byte may be partially used).
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct PrefixIndication {
    /// Number of valid `sei_prefix_data_bit` bits, i.e.
    /// `num_bits_in_prefix_indication_minus1[ i ] + 1` (always ≥ 1).
    num_bits: usize,
    /// The data bits, MSB-first, packed into `ceil(num_bits / 8)` bytes.
    /// Bits beyond `num_bits` in the final byte are zero.
    bits: Vec<u8>,
}

impl PrefixIndication {
    /// The number of valid data bits (`num_bits_in_prefix_indication_minus1 + 1`).
    pub fn num_bits(&self) -> usize {
        self.num_bits
    }

    /// The packed data bytes (MSB-first); the final byte may be only
    /// partially significant — use [`PrefixIndication::num_bits`] for the
    /// exact valid bit count.
    pub fn packed_bits(&self) -> &[u8] {
        &self.bits
    }

    /// The `j`-th `sei_prefix_data_bit` (`0` or `1`), or `None` if `j` is
    /// out of range.
    pub fn bit(&self, j: usize) -> Option<u8> {
        if j >= self.num_bits {
            return None;
        }
        let byte = self.bits[j / 8];
        Some((byte >> (7 - (j % 8))) & 1)
    }
}

/// A parsed SEI prefix indication SEI message (§D.9.1).
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct SeiPrefixIndication {
    /// `prefix_sei_payload_type` — the `payloadType` value of the SEI
    /// messages the indications describe.
    pub payload_type: u16,
    /// The SEI prefix indications, in stream order; `indications.len()`
    /// equals `num_sei_prefix_indications_minus1 + 1` (always ≥ 1).
    pub indications: Vec<PrefixIndication>,
}

impl SeiPrefixIndication {
    /// `num_sei_prefix_indications_minus1 + 1` — the number of SEI prefix
    /// indications carried (always ≥ 1 for a conforming message).
    pub fn num_indications(&self) -> usize {
        self.indications.len()
    }
}

/// Parse a `sei_prefix_indication( payloadSize )` body (§D.9.1) from the
/// raw SEI payload bytes carried by a `payloadType == 201`
/// `sei_message()`.
///
/// `payload` is the `sei_payload()` argument region — the `payloadSize`
/// bytes that follow the `sei_message()` header, with emulation-
/// prevention bytes already removed. Each prefix indication's data-bit
/// run is followed by `byte_alignment(...)` padding, so the body is a
/// whole number of bytes; the parser requires the bit reader to land
/// exactly at `payload.len()` after the announced number of indications
/// so a framing error does not desynchronise the enclosing `sei_rbsp()`
/// walk.
///
/// Errors:
/// * a `payload` shorter than the 3-byte fixed header
///   (`prefix_sei_payload_type` u(16) + `num_sei_prefix_indications_minus1`
///   u(8)) is rejected as truncated;
/// * a data-bit run or its alignment padding that overruns `payload` is
///   rejected (propagated from the bit reader);
/// * a `byte_alignment_bit_equal_to_one` that is not `1` violates §D.9.2
///   and is rejected;
/// * a body whose consumed length does not equal `payloadSize` (trailing
///   bytes beyond the §D.9.1 structure, or a short body) is rejected.
pub fn parse_sei_prefix_indication(payload: &[u8]) -> Result<SeiPrefixIndication> {
    // Fixed header: prefix_sei_payload_type u(16) + num..minus1 u(8).
    if payload.len() < 3 {
        return Err(Error::invalid(
            "h266 sei_prefix_indication: payload too short for fixed header (§D.9.1)",
        ));
    }

    let mut reader = BitReader::new(payload);
    let payload_type = reader.u(16)? as u16;
    let num_indications = reader.u(8)? as usize + 1;

    let mut indications = Vec::with_capacity(num_indications);
    for _ in 0..num_indications {
        let num_bits = reader.u(16)? as usize + 1;

        // Read the num_bits data bits MSB-first into packed bytes.
        let nbytes = num_bits.div_ceil(8);
        let mut bits = vec![0u8; nbytes];
        for j in 0..num_bits {
            let bit = reader.u1()? as u8;
            bits[j / 8] |= bit << (7 - (j % 8));
        }

        // byte_alignment(): while not byte-aligned, read a padding bit
        // that §D.9.2 requires to be 1.
        while !reader.is_byte_aligned() {
            let pad = reader.u1()?;
            if pad != 1 {
                return Err(Error::invalid(
                    "h266 sei_prefix_indication: byte_alignment_bit_equal_to_one was 0 (§D.9.2)",
                ));
            }
        }

        indications.push(PrefixIndication { num_bits, bits });
    }

    // §7.4.6: payloadSize equals the body length. Each indication ends
    // byte-aligned, so after the announced count the reader must sit
    // exactly at the end of the payload — no leftover bytes, no overrun.
    if reader.bits_remaining() != 0 {
        return Err(Error::invalid(
            "h266 sei_prefix_indication: trailing bytes beyond §D.9.1 structure (§7.4.6)",
        ));
    }

    Ok(SeiPrefixIndication {
        payload_type,
        indications,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A single indication of one data bit (`= 1`); the body is the
    /// 3-byte header plus a u(16) length plus the bit and its alignment
    /// padding.
    #[test]
    fn single_indication_one_bit() {
        // prefix_sei_payload_type = 45 (0x002D),
        // num_sei_prefix_indications_minus1 = 0,
        // num_bits_in_prefix_indication_minus1[0] = 0 (so 1 bit),
        // sei_prefix_data_bit[0][0] = 1, then 7 alignment-one bits.
        // The data byte is 0b1111_1111 = 0xFF (1 data bit + 7 pad ones).
        let payload = [0x00u8, 0x2D, 0x00, 0x00, 0x00, 0xFF];
        let p = parse_sei_prefix_indication(&payload).unwrap();
        assert_eq!(p.payload_type, 45);
        assert_eq!(p.num_indications(), 1);
        let ind = &p.indications[0];
        assert_eq!(ind.num_bits(), 1);
        assert_eq!(ind.bit(0), Some(1));
        assert_eq!(ind.bit(1), None);
    }

    /// A single indication carrying a full byte (8 data bits): no
    /// alignment padding is needed.
    #[test]
    fn single_indication_full_byte_no_padding() {
        // payload_type = 4, count-1 = 0, num_bits-1 = 7 (so 8 bits),
        // data byte = 0xA5.
        let payload = [0x00u8, 0x04, 0x00, 0x00, 0x07, 0xA5];
        let p = parse_sei_prefix_indication(&payload).unwrap();
        assert_eq!(p.payload_type, 4);
        let ind = &p.indications[0];
        assert_eq!(ind.num_bits(), 8);
        assert_eq!(ind.packed_bits(), &[0xA5]);
        // 0xA5 = 1010_0101, MSB first.
        let expected = [1u8, 0, 1, 0, 0, 1, 0, 1];
        for (j, &e) in expected.iter().enumerate() {
            assert_eq!(ind.bit(j), Some(e), "bit {j}");
        }
        assert_eq!(ind.bit(8), None);
    }

    /// An indication of 12 bits spans two bytes; the final 4 bits of the
    /// second byte are alignment-one padding.
    #[test]
    fn indication_twelve_bits_spans_two_bytes() {
        // payload_type = 137, count-1 = 0, num_bits-1 = 11 (so 12 bits).
        // 12 data bits = 0xAB, 0xC.. ; full data 0xAB 0xCX where the low
        // 4 bits of the second byte are pad ones -> 0xAB, 0xCF.
        let payload = [0x00u8, 0x89, 0x00, 0x00, 0x0B, 0xAB, 0xCF];
        let p = parse_sei_prefix_indication(&payload).unwrap();
        assert_eq!(p.payload_type, 137);
        let ind = &p.indications[0];
        assert_eq!(ind.num_bits(), 12);
        // First 8 bits = 0xAB; next 4 bits = high nibble of 0xCF = 0xC.
        assert_eq!(ind.packed_bits(), &[0xAB, 0xC0]);
        assert_eq!(ind.bit(8), Some(1)); // 0xC = 1100
        assert_eq!(ind.bit(9), Some(1));
        assert_eq!(ind.bit(10), Some(0));
        assert_eq!(ind.bit(11), Some(0));
        assert_eq!(ind.bit(12), None);
    }

    /// Two indications chained: each ends byte-aligned so the second
    /// starts cleanly.
    #[test]
    fn two_indications_chained() {
        // payload_type = 0, count-1 = 1 (so 2 indications).
        // Ind A: num_bits-1 = 3 (4 bits), data nibble 0b1010 + 4 pad
        //   ones -> 0xAF.
        // Ind B: num_bits-1 = 7 (8 bits), data byte 0x3C (no padding).
        let payload = [
            0x00u8, 0x00, // payload_type = 0
            0x01, // count-1 = 1
            0x00, 0x03, 0xAF, // ind A: 4 bits + padding
            0x00, 0x07, 0x3C, // ind B: 8 bits
        ];
        let p = parse_sei_prefix_indication(&payload).unwrap();
        assert_eq!(p.payload_type, 0);
        assert_eq!(p.num_indications(), 2);
        assert_eq!(p.indications[0].num_bits(), 4);
        assert_eq!(p.indications[0].bit(0), Some(1));
        assert_eq!(p.indications[0].bit(1), Some(0));
        assert_eq!(p.indications[0].bit(2), Some(1));
        assert_eq!(p.indications[0].bit(3), Some(0));
        assert_eq!(p.indications[1].num_bits(), 8);
        assert_eq!(p.indications[1].packed_bits(), &[0x3C]);
    }

    /// A `payload_type` using the high half of the u(16) range round-trips.
    #[test]
    fn large_payload_type() {
        // payload_type = 0xBEEF, count-1 = 0, num_bits-1 = 0 (1 bit).
        let payload = [0xBEu8, 0xEF, 0x00, 0x00, 0x00, 0xFF];
        let p = parse_sei_prefix_indication(&payload).unwrap();
        assert_eq!(p.payload_type, 0xBEEF);
    }

    /// A truncated fixed header is rejected.
    #[test]
    fn truncated_header_rejected() {
        assert!(parse_sei_prefix_indication(&[]).is_err());
        assert!(parse_sei_prefix_indication(&[0x00]).is_err());
        assert!(parse_sei_prefix_indication(&[0x00, 0x2D]).is_err());
    }

    /// A data-bit run that overruns the payload is rejected.
    #[test]
    fn data_run_overrun_rejected() {
        // count-1 = 0, num_bits-1 = 15 (16 bits) but only 1 data byte.
        let payload = [0x00u8, 0x2D, 0x00, 0x00, 0x0F, 0xAB];
        assert!(parse_sei_prefix_indication(&payload).is_err());
    }

    /// A non-`1` `byte_alignment_bit_equal_to_one` is rejected per §D.9.2.
    #[test]
    fn non_one_alignment_bit_rejected() {
        // count-1 = 0, num_bits-1 = 0 (1 data bit), data byte 0x80:
        // first bit = 1 (data), remaining 7 pad bits = 0 -> violates
        // "equal to 1".
        let payload = [0x00u8, 0x2D, 0x00, 0x00, 0x00, 0x80];
        assert!(parse_sei_prefix_indication(&payload).is_err());
    }

    /// A body with trailing bytes beyond the §D.9.1 structure is rejected.
    #[test]
    fn trailing_bytes_rejected() {
        // Valid single 1-bit indication (0xFF) plus an extra 0x00 byte.
        let payload = [0x00u8, 0x2D, 0x00, 0x00, 0x00, 0xFF, 0x00];
        assert!(parse_sei_prefix_indication(&payload).is_err());
    }

    /// A body that ends one byte short of a complete final indication is
    /// rejected (the alignment / data run overruns).
    #[test]
    fn short_body_rejected() {
        // count-1 = 1 (2 indications) but only one indication's bytes.
        let payload = [0x00u8, 0x2D, 0x01, 0x00, 0x00, 0xFF];
        assert!(parse_sei_prefix_indication(&payload).is_err());
    }

    /// `SEI_PREFIX_INDICATION_PAYLOAD_TYPE` matches the §D.2 dispatch
    /// value.
    #[test]
    fn payload_type_constant() {
        assert_eq!(SEI_PREFIX_INDICATION_PAYLOAD_TYPE, 201);
    }

    /// `bit()` returns `None` for out-of-range indices and the correct
    /// value across an 8-bit indication.
    #[test]
    fn bit_accessor_bounds() {
        let ind = PrefixIndication {
            num_bits: 3,
            bits: vec![0b1010_0000],
        };
        assert_eq!(ind.bit(0), Some(1));
        assert_eq!(ind.bit(1), Some(0));
        assert_eq!(ind.bit(2), Some(1));
        assert_eq!(ind.bit(3), None);
        assert_eq!(ind.bit(100), None);
    }
}
