//! VVC `rbsp_trailing_bits()` (§7.3.2.16 / §7.4.3.16),
//! `rbsp_slice_trailing_bits()` (§7.3.2.15 / §7.4.3.15), and
//! `byte_alignment()` (§7.3.2.17 / §7.4.3.17) reader-side validators.
//!
//! The `rbsp_trailing_bits()` and `byte_alignment()` grammars are
//! textually identical — they each begin with a single `1` bit and
//! then pad zero bits until the next byte boundary. They differ only
//! in which higher-level grammar invokes them, so the semantics
//! §7.4.3.16 / §7.4.3.17 spell out the same two constraints:
//!
//! * `rbsp_stop_one_bit` (resp. `byte_alignment_bit_equal_to_one`)
//!   shall be equal to 1.
//! * `rbsp_alignment_zero_bit` (resp. `byte_alignment_bit_equal_to_zero`)
//!   shall be equal to 0.
//!
//! `rbsp_slice_trailing_bits()` is the slice-layer wrapper: it runs
//! `rbsp_trailing_bits()` first (so the body bits get a stop bit + zero
//! pad to a byte boundary), then while `more_rbsp_trailing_data()` is
//! true it consumes byte-aligned `rbsp_cabac_zero_word` values, each a
//! 16-bit `0x0000` (§7.3.2.15). The §7.4.3.15 semantics text spells out
//! that `rbsp_cabac_zero_word` "is a byte-aligned sequence of two bytes
//! equal to 0x0000".
//!
//! The writer side is already covered by
//! [`crate::encoder::BitWriter::rbsp_trailing_bits`] /
//! [`crate::encoder::BitWriter::byte_alignment`]. This module adds the
//! reader-side dual: a typed verifier that consumes the stop-bit + zero
//! pad and surfaces a clear error when an upstream framing bug leaves
//! a non-1 stop bit or a non-0 alignment bit.
//!
//! ## Syntax
//!
//! ```text
//!   rbsp_trailing_bits() {                                          Descriptor
//!       rbsp_stop_one_bit       /* equal to 1 */                    f(1)
//!       while (!byte_aligned())
//!           rbsp_alignment_zero_bit  /* equal to 0 */               f(1)
//!   }
//!
//!   byte_alignment() {                                              Descriptor
//!       byte_alignment_bit_equal_to_one  /* equal to 1 */           f(1)
//!       while (!byte_aligned())
//!           byte_alignment_bit_equal_to_zero /* equal to 0 */       f(1)
//!   }
//! ```
//!
//! ## Usage shape
//!
//! Higher-level RBSP parsers in this crate generally consume the body
//! bits of the syntax table and stop without verifying the trailing
//! pad — `parse_access_unit_delimiter` is the worked example (see its
//! docs). Once the body has been read, callers that need full
//! conformance can pass the partially-consumed [`BitReader`] to
//! [`validate_rbsp_trailing_bits`] / [`validate_byte_alignment`] to
//! discharge the §7.3.2.16 / §7.3.2.17 contract.
//!
//! Spec reference: ITU-T H.266 | ISO/IEC 23090-3 (V4, 01/2026),
//! §7.3.2.16, §7.3.2.17, §7.4.3.16, §7.4.3.17.

use oxideav_core::{Error, Result};

use crate::bitreader::BitReader;

/// Consume an `rbsp_trailing_bits()` sequence from `br` per §7.3.2.16.
///
/// Reads exactly one bit (the `rbsp_stop_one_bit`) followed by zero or
/// more zero bits up to the next byte boundary
/// (`rbsp_alignment_zero_bit`s). Returns `Ok(consumed)` with the total
/// number of bits consumed (always in `1..=8`) on success.
///
/// Errors:
///
/// * The reader is already at end-of-stream and cannot supply the
///   stop bit.
/// * `rbsp_stop_one_bit` is 0 (§7.4.3.16 — "shall be equal to 1").
/// * Any `rbsp_alignment_zero_bit` is 1 (§7.4.3.16 — "shall be equal
///   to 0").
/// * The reader runs out of bits before reaching byte alignment.
///
/// The reader is left positioned at the byte boundary that the
/// trailing-bits sequence terminates at.
pub fn validate_rbsp_trailing_bits(br: &mut BitReader<'_>) -> Result<u32> {
    let stop = br.u1().map_err(|_| {
        Error::invalid("h266 rbsp_trailing_bits: missing rbsp_stop_one_bit (§7.3.2.16)")
    })?;
    if stop != 1 {
        return Err(Error::invalid(
            "h266 rbsp_trailing_bits: rbsp_stop_one_bit shall be equal to 1 (§7.4.3.16)",
        ));
    }
    let mut consumed: u32 = 1;
    while !br.is_byte_aligned() {
        let pad = br.u1().map_err(|_| {
            Error::invalid("h266 rbsp_trailing_bits: out of bits before byte alignment (§7.3.2.16)")
        })?;
        if pad != 0 {
            return Err(Error::invalid(
                "h266 rbsp_trailing_bits: rbsp_alignment_zero_bit shall be equal to 0 (§7.4.3.16)",
            ));
        }
        consumed += 1;
    }
    Ok(consumed)
}

/// Consume a `byte_alignment()` sequence from `br` per §7.3.2.17.
///
/// Same wire shape as [`validate_rbsp_trailing_bits`]: one `1` bit
/// (`byte_alignment_bit_equal_to_one`) followed by zero or more `0`
/// bits (`byte_alignment_bit_equal_to_zero`) up to the next byte
/// boundary. The §7.4.3.17 names differ but the constraints are
/// identical.
///
/// Returns `Ok(consumed)` with the total number of bits consumed
/// (always in `1..=8`) on success.
///
/// Errors:
///
/// * The reader cannot supply the alignment-1 bit (end of stream).
/// * `byte_alignment_bit_equal_to_one` is 0 (§7.4.3.17 — "shall be
///   equal to 1").
/// * Any `byte_alignment_bit_equal_to_zero` is 1 (§7.4.3.17 — "shall
///   be equal to 0").
/// * The reader runs out of bits before byte alignment.
pub fn validate_byte_alignment(br: &mut BitReader<'_>) -> Result<u32> {
    let one = br.u1().map_err(|_| {
        Error::invalid("h266 byte_alignment: missing byte_alignment_bit_equal_to_one (§7.3.2.17)")
    })?;
    if one != 1 {
        return Err(Error::invalid(
            "h266 byte_alignment: byte_alignment_bit_equal_to_one shall be equal to 1 (§7.4.3.17)",
        ));
    }
    let mut consumed: u32 = 1;
    while !br.is_byte_aligned() {
        let pad = br.u1().map_err(|_| {
            Error::invalid("h266 byte_alignment: out of bits before byte alignment (§7.3.2.17)")
        })?;
        if pad != 0 {
            return Err(Error::invalid(
                "h266 byte_alignment: byte_alignment_bit_equal_to_zero shall be equal to 0 (§7.4.3.17)",
            ));
        }
        consumed += 1;
    }
    Ok(consumed)
}

/// Consume an `rbsp_slice_trailing_bits()` sequence from `br` per
/// §7.3.2.15.
///
/// The grammar is
///
/// ```text
///   rbsp_slice_trailing_bits() {
///       rbsp_trailing_bits()
///       while (more_rbsp_trailing_data())
///           rbsp_cabac_zero_word /* equal to 0x0000 */  f(16)
///   }
/// ```
///
/// After `rbsp_trailing_bits()` the reader sits on a byte boundary
/// (validated by [`validate_rbsp_trailing_bits`]). Per §7.4.3.15,
/// `rbsp_cabac_zero_word` is "a byte-aligned sequence of two bytes
/// equal to 0x0000". `more_rbsp_trailing_data()` is the §7.2 hook
/// that returns true exactly when there is more data in the RBSP —
/// once the body + trailing-bits byte has been consumed, "more data"
/// after a byte boundary means at least one further byte. We accept
/// any number of `0x0000` cabac-zero-words (zero or more), but
/// reject a partial 16-bit tail or a non-zero word as a framing bug.
///
/// Returns `Ok((trailing_bits_consumed, num_cabac_zero_words))` on
/// success, where the first component is the §7.3.2.16 return value
/// (bits consumed by `rbsp_trailing_bits()`, in `1..=8`) and the
/// second is the count of `rbsp_cabac_zero_word` 16-bit words that
/// followed (in `0..`).
///
/// Errors:
///
/// * `rbsp_trailing_bits()` itself is malformed (delegates to
///   [`validate_rbsp_trailing_bits`]).
/// * A trailing-data run is not a whole number of bytes (an odd
///   trailing byte cannot be a `0x0000` 16-bit word).
/// * Any 16-bit trailing word is not `0x0000` (§7.4.3.15 — "equal
///   to 0x0000").
pub fn validate_rbsp_slice_trailing_bits(br: &mut BitReader<'_>) -> Result<(u32, u32)> {
    let trailing_bits = validate_rbsp_trailing_bits(br)?;
    // After rbsp_trailing_bits() the reader is byte-aligned (we just
    // validated that). more_rbsp_trailing_data() reduces to "any
    // further bits remaining"; at a byte boundary that's equivalent
    // to "any further bytes remaining".
    debug_assert!(br.is_byte_aligned());
    let mut cabac_zero_words: u32 = 0;
    while br.bits_remaining() > 0 {
        let remaining = br.bits_remaining();
        if remaining < 16 {
            return Err(Error::invalid(
                "h266 rbsp_slice_trailing_bits: partial trailing byte after cabac-zero-words (§7.3.2.15)",
            ));
        }
        let word = br.u(16).map_err(|_| {
            Error::invalid(
                "h266 rbsp_slice_trailing_bits: out of bits inside rbsp_cabac_zero_word (§7.3.2.15)",
            )
        })?;
        if word != 0 {
            return Err(Error::invalid(
                "h266 rbsp_slice_trailing_bits: rbsp_cabac_zero_word shall be equal to 0x0000 (§7.4.3.15)",
            ));
        }
        cabac_zero_words = cabac_zero_words.saturating_add(1);
    }
    Ok((trailing_bits, cabac_zero_words))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Byte-aligned stop bit followed by 7 alignment zeros: `1000_0000`
    /// consumes exactly 8 bits, leaves the reader at end-of-stream.
    #[test]
    fn rbsp_trailing_byte_aligned_full_pad() {
        let data = [0b1000_0000u8];
        let mut br = BitReader::new(&data);
        let consumed = validate_rbsp_trailing_bits(&mut br).unwrap();
        assert_eq!(consumed, 8);
        assert_eq!(br.bits_remaining(), 0);
    }

    /// Stop bit at bit-offset 3 of a byte: caller already read 3 body
    /// bits, the pad is bits `1 0000` (stop + 4 zeros). After the
    /// validator the reader sits on a byte boundary.
    #[test]
    fn rbsp_trailing_mid_byte_stop() {
        // Body 3 bits (e.g. AUD-like) + trailing `1 0000` → byte = `bbb 1 0000`
        // With body = `110`, byte = `1101_0000` = `0xD0`.
        let data = [0b1101_0000u8];
        let mut br = BitReader::new(&data);
        let _ = br.u(3).unwrap(); // consume body bits
        let consumed = validate_rbsp_trailing_bits(&mut br).unwrap();
        // 1 stop bit + 4 zero pad bits = 5 bits.
        assert_eq!(consumed, 5);
        assert!(br.is_byte_aligned());
        assert_eq!(br.bits_remaining(), 0);
    }

    /// Stop bit is the last bit of a byte → zero pad bits follow.
    /// Body = 7 bits (`0000_000`), then stop bit `1` at the LSB.
    #[test]
    fn rbsp_trailing_stop_at_byte_end() {
        let data = [0b0000_0001u8];
        let mut br = BitReader::new(&data);
        let _ = br.u(7).unwrap();
        let consumed = validate_rbsp_trailing_bits(&mut br).unwrap();
        // Just the stop bit, no zero pad bits.
        assert_eq!(consumed, 1);
        assert!(br.is_byte_aligned());
        assert_eq!(br.bits_remaining(), 0);
    }

    /// `rbsp_stop_one_bit = 0` violates §7.4.3.16.
    #[test]
    fn rbsp_trailing_stop_bit_must_be_one() {
        let data = [0b0000_0000u8];
        let mut br = BitReader::new(&data);
        let err = validate_rbsp_trailing_bits(&mut br).unwrap_err();
        // Make sure the error message references §7.4.3.16.
        assert!(format!("{err:?}").contains("rbsp_stop_one_bit"));
    }

    /// A non-zero alignment pad bit violates §7.4.3.16.
    #[test]
    fn rbsp_trailing_alignment_zero_must_be_zero() {
        // `1100_0000` — stop bit ok, but the next pad bit is 1, not 0.
        let data = [0b1100_0000u8];
        let mut br = BitReader::new(&data);
        let err = validate_rbsp_trailing_bits(&mut br).unwrap_err();
        assert!(format!("{err:?}").contains("rbsp_alignment_zero_bit"));
    }

    /// End-of-stream before the stop bit is reported.
    #[test]
    fn rbsp_trailing_empty_input_is_rejected() {
        let data: [u8; 0] = [];
        let mut br = BitReader::new(&data);
        let err = validate_rbsp_trailing_bits(&mut br).unwrap_err();
        assert!(format!("{err:?}").contains("rbsp_stop_one_bit"));
    }

    /// `byte_alignment()` mirrors `rbsp_trailing_bits()` for the
    /// canonical full-pad case.
    #[test]
    fn byte_alignment_byte_aligned_full_pad() {
        let data = [0b1000_0000u8];
        let mut br = BitReader::new(&data);
        let consumed = validate_byte_alignment(&mut br).unwrap();
        assert_eq!(consumed, 8);
        assert_eq!(br.bits_remaining(), 0);
    }

    /// `byte_alignment()` at a mid-byte position.
    #[test]
    fn byte_alignment_mid_byte_stop() {
        // Body 5 bits + trailing `1 00` → byte = `bbbbb 1 00`
        // With body = `10101`, byte = `1010_1100` = `0xAC`.
        let data = [0b1010_1100u8];
        let mut br = BitReader::new(&data);
        let _ = br.u(5).unwrap();
        let consumed = validate_byte_alignment(&mut br).unwrap();
        // 1 stop bit + 2 zero pad bits.
        assert_eq!(consumed, 3);
        assert!(br.is_byte_aligned());
    }

    /// `byte_alignment_bit_equal_to_one = 0` violates §7.4.3.17.
    #[test]
    fn byte_alignment_one_bit_must_be_one() {
        let data = [0b0000_0000u8];
        let mut br = BitReader::new(&data);
        let err = validate_byte_alignment(&mut br).unwrap_err();
        assert!(format!("{err:?}").contains("byte_alignment_bit_equal_to_one"));
    }

    /// `byte_alignment_bit_equal_to_zero = 1` violates §7.4.3.17.
    #[test]
    fn byte_alignment_zero_bit_must_be_zero() {
        let data = [0b1100_0000u8];
        let mut br = BitReader::new(&data);
        let err = validate_byte_alignment(&mut br).unwrap_err();
        assert!(format!("{err:?}").contains("byte_alignment_bit_equal_to_zero"));
    }

    /// Both validators consume exactly the trailing pad and leave the
    /// reader byte-aligned, regardless of starting bit offset 0..7.
    #[test]
    fn rbsp_trailing_consumes_correct_count_at_every_offset() {
        // For each bit offset `k` in 0..=7, build a byte where the
        // first `k` bits are body (here `1`s for visibility) and the
        // remaining `8 - k` bits are `rbsp_trailing_bits()` (a `1`
        // followed by zero pads).
        for k in 0u32..=7 {
            // Body bits = all 1s (k bits), then stop bit 1, then
            // (8 - k - 1) zero pad bits, but if k = 7 there is exactly
            // 1 bit left for the stop and no pad bits.
            // Compose the byte from MSB.
            let mut byte: u8 = 0;
            let mut pos = 7;
            for _ in 0..k {
                byte |= 1 << pos;
                pos -= 1;
            }
            // Stop bit.
            byte |= 1 << pos;
            // Remaining pad bits are 0 — already zero in `byte`.

            let data = [byte];
            let mut br = BitReader::new(&data);
            if k > 0 {
                let _ = br.u(k).unwrap();
            }
            let consumed = validate_rbsp_trailing_bits(&mut br).unwrap();
            assert_eq!(
                consumed,
                8 - k,
                "offset {k}: expected {} consumed, got {consumed}",
                8 - k
            );
            assert!(
                br.is_byte_aligned(),
                "offset {k}: not byte-aligned after validate"
            );
        }
    }

    /// Cross-check against the encoder-side `rbsp_trailing_bits()`:
    /// emit + read should round-trip for every starting offset.
    #[test]
    fn rbsp_trailing_round_trip_against_encoder() {
        use crate::encoder::BitWriter;
        for k in 0u32..=7 {
            let mut bw = BitWriter::new();
            // Write `k` body bits = all 1s.
            for _ in 0..k {
                bw.write_bit(1);
            }
            bw.rbsp_trailing_bits();
            let bytes = bw.into_bytes();
            assert_eq!(
                bytes.len(),
                1,
                "encoder produced {} bytes for offset {k}",
                bytes.len()
            );

            let mut br = BitReader::new(&bytes);
            if k > 0 {
                assert_eq!(br.u(k).unwrap(), (1u32 << k) - 1);
            }
            let consumed = validate_rbsp_trailing_bits(&mut br).unwrap();
            assert_eq!(consumed, 8 - k);
            assert!(br.is_byte_aligned());
            assert_eq!(br.bits_remaining(), 0);
        }
    }

    // ---------------- rbsp_slice_trailing_bits() (§7.3.2.15) -----------------

    /// Zero cabac-zero-words case: just `rbsp_trailing_bits()` and no
    /// extra trailing bytes.
    #[test]
    fn rbsp_slice_trailing_no_cabac_zero_words() {
        let data = [0b1000_0000u8];
        let mut br = BitReader::new(&data);
        let (trail, n) = validate_rbsp_slice_trailing_bits(&mut br).unwrap();
        assert_eq!(trail, 8);
        assert_eq!(n, 0);
        assert_eq!(br.bits_remaining(), 0);
    }

    /// One cabac-zero-word: a stop-bit byte followed by `0x00 0x00`.
    #[test]
    fn rbsp_slice_trailing_one_cabac_zero_word() {
        let data = [0b1000_0000u8, 0x00, 0x00];
        let mut br = BitReader::new(&data);
        let (trail, n) = validate_rbsp_slice_trailing_bits(&mut br).unwrap();
        assert_eq!(trail, 8);
        assert_eq!(n, 1);
        assert_eq!(br.bits_remaining(), 0);
    }

    /// Several cabac-zero-words back to back: stop byte + N pairs of
    /// `0x00`.
    #[test]
    fn rbsp_slice_trailing_many_cabac_zero_words() {
        for n_words in 0u32..=4 {
            let mut data = vec![0b1000_0000u8];
            for _ in 0..n_words {
                data.push(0x00);
                data.push(0x00);
            }
            let mut br = BitReader::new(&data);
            let (trail, n) = validate_rbsp_slice_trailing_bits(&mut br).unwrap();
            assert_eq!(trail, 8, "n_words={n_words}");
            assert_eq!(n, n_words, "n_words={n_words}");
            assert_eq!(br.bits_remaining(), 0, "n_words={n_words}");
        }
    }

    /// A 16-bit trailing word that is not `0x0000` violates §7.4.3.15.
    #[test]
    fn rbsp_slice_trailing_non_zero_cabac_word_rejected() {
        let data = [0b1000_0000u8, 0x00, 0x01];
        let mut br = BitReader::new(&data);
        let err = validate_rbsp_slice_trailing_bits(&mut br).unwrap_err();
        assert!(format!("{err:?}").contains("rbsp_cabac_zero_word"));
    }

    /// A single dangling byte after the trailing-bits byte cannot form a
    /// 16-bit `rbsp_cabac_zero_word`; this is a framing bug.
    #[test]
    fn rbsp_slice_trailing_partial_word_rejected() {
        let data = [0b1000_0000u8, 0x00];
        let mut br = BitReader::new(&data);
        let err = validate_rbsp_slice_trailing_bits(&mut br).unwrap_err();
        assert!(format!("{err:?}").contains("partial trailing byte"));
    }

    /// A bad `rbsp_trailing_bits()` is surfaced unchanged.
    #[test]
    fn rbsp_slice_trailing_propagates_inner_trailing_error() {
        // `0b0000_0000` — missing stop bit.
        let data = [0b0000_0000u8];
        let mut br = BitReader::new(&data);
        let err = validate_rbsp_slice_trailing_bits(&mut br).unwrap_err();
        assert!(format!("{err:?}").contains("rbsp_stop_one_bit"));
    }

    /// Works at a non-zero starting bit offset just like the wrapped
    /// `rbsp_trailing_bits()`: after the body bits we still land on the
    /// next byte boundary and then read the cabac-zero-word words.
    #[test]
    fn rbsp_slice_trailing_mid_byte_then_zero_words() {
        // Body 5 bits + trailing `1 00` → byte = `bbbbb 1 00`
        // With body = `10101`, byte = `1010_1100` = `0xAC`. Then
        // two cabac-zero-words = 4 zero bytes.
        let data = [0b1010_1100u8, 0x00, 0x00, 0x00, 0x00];
        let mut br = BitReader::new(&data);
        let _ = br.u(5).unwrap();
        let (trail, n) = validate_rbsp_slice_trailing_bits(&mut br).unwrap();
        assert_eq!(trail, 3); // stop bit + 2 zero pad bits
        assert_eq!(n, 2);
        assert_eq!(br.bits_remaining(), 0);
    }

    /// End-to-end round-trip using the encoder-side `rbsp_trailing_bits()`
    /// to build the stop byte, then manually appending zero-words and
    /// reading the whole sequence back.
    #[test]
    fn rbsp_slice_trailing_round_trip_against_encoder() {
        use crate::encoder::BitWriter;
        for k in 0u32..=7 {
            for n_words in 0u32..=3 {
                let mut bw = BitWriter::new();
                for _ in 0..k {
                    bw.write_bit(1);
                }
                bw.rbsp_trailing_bits();
                let mut bytes = bw.into_bytes();
                // Append n_words 16-bit cabac-zero-words (= 2 * n_words bytes of 0).
                for _ in 0..n_words {
                    bytes.push(0x00);
                    bytes.push(0x00);
                }
                let mut br = BitReader::new(&bytes);
                if k > 0 {
                    assert_eq!(br.u(k).unwrap(), (1u32 << k) - 1);
                }
                let (trail, n) = validate_rbsp_slice_trailing_bits(&mut br).unwrap();
                assert_eq!(trail, 8 - k, "k={k}, n_words={n_words}");
                assert_eq!(n, n_words, "k={k}, n_words={n_words}");
                assert_eq!(br.bits_remaining(), 0, "k={k}, n_words={n_words}");
            }
        }
    }
}
