//! VVC subpicture level information SEI message parser (§D.7.1 / §D.7.2).
//!
//! The subpicture level information (SLI) SEI message
//! (`payloadType == 203`) conveys, for the set of CVSs of the OLSs to
//! which it applies, the level that each subpicture sequence conforms to
//! when the extracted sub-bitstream containing that subpicture sequence
//! is conformance-tested per Annex A — plus the per-subpicture fractions
//! of the OLS-level bitstream limits used by the §D.7.2 CPB-size and
//! bit-rate derivations (eqs. 1637–1641).
//!
//! Like the SEI manifest (`payloadType == 200`) and the SEI prefix
//! indication (`payloadType == 201`), this is one of the few
//! `sei_payload()` bodies specified directly in this Specification (most
//! other payload types defer to Rec. ITU-T H.274 | ISO/IEC 23002-7), so
//! the *syntax* is parseable from its own payload bytes with no external
//! HRD / VPS context. (The §D.7.2 derivations that consume the parsed
//! fields — `SubpicCpbSize*`, `SubpicBitRate*`, `SubpicLevelIdc` — need
//! the OLS / level context and remain a higher-layer concern; this module
//! surfaces the raw syntax elements for them.)
//!
//! Syntax (§D.7.1):
//!
//! ```text
//!   subpic_level_info( payloadSize ) {                                Descriptor
//!       sli_num_ref_levels_minus1                                        u(3)
//!       sli_cbr_constraint_flag                                          u(1)
//!       sli_explicit_fraction_present_flag                               u(1)
//!       if( sli_explicit_fraction_present_flag )
//!           sli_num_subpics_minus1                                       ue(v)
//!       sli_max_sublayers_minus1                                         u(3)
//!       sli_sublayer_info_present_flag                                   u(1)
//!       while( !byte_aligned( ) )
//!           sli_alignment_zero_bit /* equal to 0 */                      f(1)
//!       for( k = sli_sublayer_info_present_flag ? 0 : sli_max_sublayers_minus1;
//!            k <= sli_max_sublayers_minus1; k++ )
//!           for( i = 0; i <= sli_num_ref_levels_minus1; i++ ) {
//!               sli_non_subpic_layers_fraction[ i ][ k ]                 u(8)
//!               sli_ref_level_idc[ i ][ k ]                              u(8)
//!               if( sli_explicit_fraction_present_flag )
//!                   for( j = 0; j <= sli_num_subpics_minus1; j++ )
//!                       sli_ref_level_fraction_minus1[ i ][ j ][ k ]     u(8)
//!           }
//!   }
//! ```
//!
//! §D.7.2 semantics modelled here:
//!
//! * `sli_num_ref_levels_minus1` plus 1 — the number of reference levels
//!   signalled for each subpicture sequence.
//! * `sli_cbr_constraint_flag` — `1` if the HSS operates in constant
//!   bit-rate mode for the extracted sub-bitstreams, `0` for intermittent
//!   bit-rate mode.
//! * `sli_explicit_fraction_present_flag` — `1` if the
//!   `sli_ref_level_fraction_minus1[ i ][ j ][ k ]` elements are present.
//! * `sli_num_subpics_minus1` plus 1 — the number of subpictures in the
//!   pictures of the `multiSubpicLayers`. Present only when
//!   `sli_explicit_fraction_present_flag` is `1`; when absent the parser
//!   leaves it `None` (no fractions are then carried).
//! * `sli_max_sublayers_minus1` plus 1 — the maximum number of temporal
//!   sublayers for which level information is indicated.
//! * `sli_sublayer_info_present_flag` — `1` if information is present for
//!   sublayer representations `0 ..= sli_max_sublayers_minus1`; `0` if
//!   only for the `sli_max_sublayers_minus1`-th sublayer. The first
//!   sublayer for which entries are read is
//!   `sli_sublayer_info_present_flag ? 0 : sli_max_sublayers_minus1`.
//! * `sli_alignment_zero_bit` — §D.7.2 requires each padding bit to be
//!   `0`; the parser reads and verifies them rather than discarding them.
//! * `sli_non_subpic_layers_fraction[ i ][ k ]`, `sli_ref_level_idc[ i ][ k ]`,
//!   and (when present) `sli_ref_level_fraction_minus1[ i ][ j ][ k ]` —
//!   stored per sublayer `k` in a [`SubpicLevelSublayer`].
//!
//! The body is byte-aligned at the point the per-sublayer loops begin
//! (the `while( !byte_aligned() )` padding guarantees it) and is then a
//! whole number of `u(8)` elements, so the whole body occupies a whole
//! number of bytes. §7.4.6 requires the derived `payloadSize` to equal
//! the number of payload bytes; this parser cross-checks the consumed
//! length against `payloadSize` so a truncated or over-long body surfaces
//! as a parse error instead of silently desynchronising the enclosing
//! `sei_rbsp()` walk.
//!
//! Spec reference: ITU-T H.266 | ISO/IEC 23090-3 (V4, 01/2026),
//! §D.7.1, §D.7.2, §D.2 (`sei_payload()` dispatch), §7.4.6.
//!
//! No third-party VVC decoder source was consulted; the implementation
//! is spec-only and reads the payload through the crate's own
//! [`BitReader`].

use oxideav_core::{Error, Result};

use crate::bitreader::BitReader;

/// `payloadType` value that selects the subpicture level information body
/// in the §D.2 `sei_payload()` dispatch.
pub const SUBPIC_LEVEL_INFO_PAYLOAD_TYPE: u32 = 203;

/// One reference-level entry within a sublayer (`i`-th of
/// `sli_num_ref_levels_minus1 + 1`).
///
/// Holds `sli_non_subpic_layers_fraction[ i ][ k ]` and
/// `sli_ref_level_idc[ i ][ k ]` plus, when
/// `sli_explicit_fraction_present_flag` is `1`, the
/// `sli_ref_level_fraction_minus1[ i ][ j ][ k ]` run over the
/// `sli_num_subpics_minus1 + 1` subpictures (index `j`).
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct SubpicLevelRefEntry {
    /// `sli_non_subpic_layers_fraction[ i ][ k ]` — the `i`-th fraction of
    /// the bitstream level limits associated with layers that have
    /// `sps_num_subpics_minus1 == 0` when `Htid == k`.
    pub non_subpic_layers_fraction: u8,
    /// `sli_ref_level_idc[ i ][ k ]` — the `i`-th level each subpicture
    /// sequence conforms to (per Annex A) when `Htid == k`.
    pub ref_level_idc: u8,
    /// `sli_ref_level_fraction_minus1[ i ][ j ][ k ]` for
    /// `j` in `0 ..= sli_num_subpics_minus1` — empty when
    /// `sli_explicit_fraction_present_flag` is `0`.
    pub ref_level_fraction_minus1: Vec<u8>,
}

/// The per-sublayer block of reference-level entries for one value of the
/// sublayer index `k`.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct SubpicLevelSublayer {
    /// The sublayer index `k` this block describes (`Htid == k`).
    pub sublayer_idx: u8,
    /// The `sli_num_ref_levels_minus1 + 1` reference-level entries (index
    /// `i`).
    pub entries: Vec<SubpicLevelRefEntry>,
}

/// A parsed subpicture level information SEI message (§D.7.1).
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct SubpicLevelInfo {
    /// `sli_cbr_constraint_flag`.
    pub cbr_constraint_flag: bool,
    /// `sli_explicit_fraction_present_flag` — `true` when the
    /// `sli_ref_level_fraction_minus1` elements are carried.
    pub explicit_fraction_present_flag: bool,
    /// `sli_num_subpics_minus1` — present only when
    /// `explicit_fraction_present_flag` is `true`.
    pub num_subpics_minus1: Option<u32>,
    /// `sli_max_sublayers_minus1`.
    pub max_sublayers_minus1: u8,
    /// `sli_sublayer_info_present_flag`.
    pub sublayer_info_present_flag: bool,
    /// The per-sublayer blocks, in the stream order the `k` loop visits
    /// them (`sublayer_info_present_flag ? 0 : max_sublayers_minus1`
    /// up to `max_sublayers_minus1`).
    pub sublayers: Vec<SubpicLevelSublayer>,
}

impl SubpicLevelInfo {
    /// `sli_num_ref_levels_minus1 + 1` — the number of reference levels
    /// signalled for each subpicture sequence. Derived from the first
    /// sublayer block (every block carries the same count); `0` if no
    /// sublayer blocks were parsed.
    pub fn num_ref_levels(&self) -> usize {
        self.sublayers.first().map_or(0, |s| s.entries.len())
    }

    /// `sli_num_subpics_minus1 + 1` — the number of subpictures the
    /// explicit fractions span, or `None` when
    /// `explicit_fraction_present_flag` is `false`.
    pub fn num_subpics(&self) -> Option<u32> {
        self.num_subpics_minus1.map(|v| v + 1)
    }

    /// The per-sublayer block describing sublayer index `k`, if present.
    pub fn sublayer(&self, k: u8) -> Option<&SubpicLevelSublayer> {
        self.sublayers.iter().find(|s| s.sublayer_idx == k)
    }
}

/// Parse a `subpic_level_info( payloadSize )` body (§D.7.1) from the raw
/// SEI payload bytes carried by a `payloadType == 203` `sei_message()`.
///
/// `payload` is the `sei_payload()` argument region — the `payloadSize`
/// bytes that follow the `sei_message()` header, with emulation-
/// prevention bytes already removed. After the bit-prefixed header the
/// `while( !byte_aligned() )` padding lands the reader on a byte boundary,
/// and the per-sublayer loops then consume a whole number of `u(8)`
/// elements, so the body occupies a whole number of bytes; the parser
/// requires the bit reader to land exactly at `payload.len()` so a framing
/// error does not desynchronise the enclosing `sei_rbsp()` walk.
///
/// Errors:
/// * a `payload` too short for the bit-prefixed header is rejected
///   (propagated from the bit reader);
/// * a non-`0` `sli_alignment_zero_bit` violates §D.7.2 and is rejected;
/// * a per-sublayer `u(8)` run that overruns `payload` is rejected
///   (propagated from the bit reader);
/// * a body whose consumed length does not equal `payloadSize` (trailing
///   bytes beyond the §D.7.1 structure) is rejected.
pub fn parse_subpic_level_info(payload: &[u8]) -> Result<SubpicLevelInfo> {
    let mut reader = BitReader::new(payload);

    let num_ref_levels_minus1 = reader.u(3)? as usize;
    let cbr_constraint_flag = reader.u1()? != 0;
    let explicit_fraction_present_flag = reader.u1()? != 0;

    let num_subpics_minus1 = if explicit_fraction_present_flag {
        Some(reader.ue()?)
    } else {
        None
    };

    let max_sublayers_minus1 = reader.u(3)? as u8;
    let sublayer_info_present_flag = reader.u1()? != 0;

    // while( !byte_aligned() ) sli_alignment_zero_bit /* equal to 0 */
    while !reader.is_byte_aligned() {
        if reader.u1()? != 0 {
            return Err(Error::invalid(
                "h266 subpic_level_info: sli_alignment_zero_bit was 1 (§D.7.2)",
            ));
        }
    }

    // The `j` loop count (only meaningful when explicit fractions are
    // present). `num_subpics_minus1` is then `Some`.
    let num_subpics = num_subpics_minus1.map_or(0usize, |v| v as usize + 1);

    // for( k = sublayer_info_present ? 0 : max_sublayers_minus1;
    //      k <= max_sublayers_minus1; k++ )
    let first_k = if sublayer_info_present_flag {
        0
    } else {
        max_sublayers_minus1
    };

    let mut sublayers = Vec::new();
    for k in first_k..=max_sublayers_minus1 {
        let mut entries = Vec::with_capacity(num_ref_levels_minus1 + 1);
        for _ in 0..=num_ref_levels_minus1 {
            let non_subpic_layers_fraction = reader.u(8)? as u8;
            let ref_level_idc = reader.u(8)? as u8;
            let ref_level_fraction_minus1 = if explicit_fraction_present_flag {
                let mut fracs = Vec::with_capacity(num_subpics);
                for _ in 0..num_subpics {
                    fracs.push(reader.u(8)? as u8);
                }
                fracs
            } else {
                Vec::new()
            };
            entries.push(SubpicLevelRefEntry {
                non_subpic_layers_fraction,
                ref_level_idc,
                ref_level_fraction_minus1,
            });
        }
        sublayers.push(SubpicLevelSublayer {
            sublayer_idx: k,
            entries,
        });
    }

    // §7.4.6: payloadSize equals the body length. The body ends
    // byte-aligned (the header padding made it so and every later element
    // is a whole byte), so the reader must sit exactly at the end of the
    // payload — no leftover bytes, no overrun.
    if reader.bits_remaining() != 0 {
        return Err(Error::invalid(
            "h266 subpic_level_info: trailing bytes beyond §D.7.1 structure (§7.4.6)",
        ));
    }

    Ok(SubpicLevelInfo {
        cbr_constraint_flag,
        explicit_fraction_present_flag,
        num_subpics_minus1,
        max_sublayers_minus1,
        sublayer_info_present_flag,
        sublayers,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build the bit-prefixed header byte from its fields. The header is
    /// `u(3) u(1) u(1)` = 5 bits, optionally followed by `ue(v)`; the
    /// helpers below assemble whole payloads by hand for clarity.
    ///
    /// Minimal case: `explicit = 0`, `sublayer_info = 0`, so only the
    /// `max_sublayers_minus1`-th sublayer is present and each entry is
    /// just `(non_subpic_fraction, ref_level_idc)` = 2 bytes.
    #[test]
    fn single_ref_level_no_explicit_no_sublayer_info() {
        // sli_num_ref_levels_minus1 = 0 (u3 = 000)
        // sli_cbr_constraint_flag   = 1 (u1)
        // sli_explicit_fraction     = 0 (u1)
        // sli_max_sublayers_minus1  = 0 (u3 = 000)
        // sli_sublayer_info_present = 0 (u1)
        // Header bits: 000 1 0 000 0 = 0b0001_0000 0 -> 9 bits.
        // Pack: first byte 0b0001_0000 (8 bits), then 1 more bit (0) +
        //   7 alignment-zero bits -> second byte 0b0000_0000.
        // Then k = max_sublayers_minus1 = 0 only; one entry:
        //   non_subpic_fraction = 0x10, ref_level_idc = 0x33.
        let payload = [0b0001_0000u8, 0b0000_0000, 0x10, 0x33];
        let sli = parse_subpic_level_info(&payload).unwrap();
        assert!(sli.cbr_constraint_flag);
        assert!(!sli.explicit_fraction_present_flag);
        assert_eq!(sli.num_subpics_minus1, None);
        assert_eq!(sli.num_subpics(), None);
        assert_eq!(sli.max_sublayers_minus1, 0);
        assert!(!sli.sublayer_info_present_flag);
        assert_eq!(sli.num_ref_levels(), 1);
        assert_eq!(sli.sublayers.len(), 1);
        let s = &sli.sublayers[0];
        assert_eq!(s.sublayer_idx, 0);
        assert_eq!(s.entries.len(), 1);
        assert_eq!(s.entries[0].non_subpic_layers_fraction, 0x10);
        assert_eq!(s.entries[0].ref_level_idc, 0x33);
        assert!(s.entries[0].ref_level_fraction_minus1.is_empty());
        assert_eq!(sli.sublayer(0).map(|s| s.entries.len()), Some(1));
        assert_eq!(sli.sublayer(1), None);
    }

    /// Two reference levels, no explicit fractions, single sublayer.
    #[test]
    fn two_ref_levels_no_explicit() {
        // num_ref_levels_minus1 = 1 (u3 = 001)
        // cbr = 0, explicit = 0
        // max_sublayers_minus1 = 0 (u3 = 000), sublayer_info = 0 (u1)
        // Header bits: 001 0 0 000 0 = 0b0010_0000 0 (9 bits)
        // byte0 = 0b0010_0000, byte1 = bit(0) + 7 zero pad = 0x00.
        // Entries (i=0,1): (0x05,0x40), (0x80,0x51).
        let payload = [0b0010_0000u8, 0x00, 0x05, 0x40, 0x80, 0x51];
        let sli = parse_subpic_level_info(&payload).unwrap();
        assert!(!sli.cbr_constraint_flag);
        assert_eq!(sli.num_ref_levels(), 2);
        let s = &sli.sublayers[0];
        assert_eq!(s.entries[0].non_subpic_layers_fraction, 0x05);
        assert_eq!(s.entries[0].ref_level_idc, 0x40);
        assert_eq!(s.entries[1].non_subpic_layers_fraction, 0x80);
        assert_eq!(s.entries[1].ref_level_idc, 0x51);
    }

    /// Explicit fractions present: one ref level, two subpictures.
    #[test]
    fn explicit_fractions_two_subpics() {
        // num_ref_levels_minus1 = 0 (u3 = 000)
        // cbr = 1
        // explicit = 1
        // then ue(v) sli_num_subpics_minus1 = 1 -> code "010"
        // max_sublayers_minus1 = 0 (u3 = 000), sublayer_info = 0 (u1)
        // Header bits: 000 1 1 010 000 0
        //   = 000 1 1   (5 bits: num_ref/cbr/explicit)
        //     010       (ue=1)
        //     000       (max_sublayers_minus1)
        //     0         (sublayer_info)
        //   = 0001_1010 0000  -> 12 bits.
        // byte0 = 0b0001_1010, then 4 more bits 0000 + 4 alignment-zero
        //   bits -> byte1 = 0b0000_0000.
        // One sublayer (k=0), one entry: non_frac=0x20, idc=0x60,
        //   then two fraction bytes for j=0,1: 0x11, 0x22.
        let payload = [0b0001_1010u8, 0b0000_0000, 0x20, 0x60, 0x11, 0x22];
        let sli = parse_subpic_level_info(&payload).unwrap();
        assert!(sli.cbr_constraint_flag);
        assert!(sli.explicit_fraction_present_flag);
        assert_eq!(sli.num_subpics_minus1, Some(1));
        assert_eq!(sli.num_subpics(), Some(2));
        assert_eq!(sli.num_ref_levels(), 1);
        let e = &sli.sublayers[0].entries[0];
        assert_eq!(e.non_subpic_layers_fraction, 0x20);
        assert_eq!(e.ref_level_idc, 0x60);
        assert_eq!(e.ref_level_fraction_minus1, vec![0x11, 0x22]);
    }

    /// `sublayer_info_present_flag == 1` with `max_sublayers_minus1 == 1`
    /// yields two sublayer blocks (k = 0 and k = 1).
    #[test]
    fn sublayer_info_present_two_sublayers() {
        // num_ref_levels_minus1 = 0 (u3 = 000)
        // cbr = 0, explicit = 0
        // max_sublayers_minus1 = 1 (u3 = 001)
        // sublayer_info_present = 1 (u1)
        // Header bits: 000 0 0 001 1 = 0b0000_0001 1 (9 bits)
        // byte0 = 0b0000_0001 (the max_sublayers_minus1 = 001 ends in the
        //   low bit), then the 9th bit (sublayer_info = 1) + 7 alignment
        //   zero bits -> byte1 = 0b1000_0000 = 0x80.
        // k loop: 0..=1, each one entry of 2 bytes.
        //   k=0: (0x01,0x10); k=1: (0x02,0x20).
        let payload = [0b0000_0001u8, 0x80, 0x01, 0x10, 0x02, 0x20];
        let sli = parse_subpic_level_info(&payload).unwrap();
        assert_eq!(sli.max_sublayers_minus1, 1);
        assert!(sli.sublayer_info_present_flag);
        assert_eq!(sli.sublayers.len(), 2);
        assert_eq!(sli.sublayers[0].sublayer_idx, 0);
        assert_eq!(sli.sublayers[0].entries[0].non_subpic_layers_fraction, 0x01);
        assert_eq!(sli.sublayers[0].entries[0].ref_level_idc, 0x10);
        assert_eq!(sli.sublayers[1].sublayer_idx, 1);
        assert_eq!(sli.sublayers[1].entries[0].non_subpic_layers_fraction, 0x02);
        assert_eq!(sli.sublayers[1].entries[0].ref_level_idc, 0x20);
        assert_eq!(sli.sublayer(1).map(|s| s.sublayer_idx), Some(1));
    }

    /// `sublayer_info_present_flag == 0` with `max_sublayers_minus1 == 2`
    /// yields a single block for k = max_sublayers_minus1 (= 2).
    #[test]
    fn no_sublayer_info_uses_top_sublayer_only() {
        // num_ref_levels_minus1 = 0, cbr = 0, explicit = 0,
        // max_sublayers_minus1 = 2 (u3 = 010), sublayer_info = 0
        // Header bits: 000 0 0 010 0 = 0b0000_0010 0 (9 bits)
        // byte0 = 0b0000_0010, byte1 = bit(0)+7 zero = 0x00.
        // Single sublayer block, sublayer_idx = 2: entry (0x07,0x70).
        let payload = [0b0000_0010u8, 0x00, 0x07, 0x70];
        let sli = parse_subpic_level_info(&payload).unwrap();
        assert_eq!(sli.max_sublayers_minus1, 2);
        assert!(!sli.sublayer_info_present_flag);
        assert_eq!(sli.sublayers.len(), 1);
        assert_eq!(sli.sublayers[0].sublayer_idx, 2);
        assert_eq!(sli.sublayers[0].entries[0].ref_level_idc, 0x70);
    }

    /// A non-`0` `sli_alignment_zero_bit` is rejected per §D.7.2.
    #[test]
    fn non_zero_alignment_bit_rejected() {
        // Same header as single_ref_level_no_explicit_no_sublayer_info
        // but flip one alignment pad bit to 1: byte1 = 0b0000_0001.
        let payload = [0b0001_0000u8, 0b0000_0001, 0x10, 0x33];
        assert!(parse_subpic_level_info(&payload).is_err());
    }

    /// A per-sublayer `u(8)` run that overruns the payload is rejected.
    #[test]
    fn entry_run_overrun_rejected() {
        // num_ref_levels_minus1 = 1 (needs 2 entries = 4 bytes) but only
        // 2 bytes of entry data supplied.
        let payload = [0b0010_0000u8, 0x00, 0x05, 0x40];
        assert!(parse_subpic_level_info(&payload).is_err());
    }

    /// A body with trailing bytes beyond the §D.7.1 structure is rejected.
    #[test]
    fn trailing_bytes_rejected() {
        // Valid minimal body plus one extra byte.
        let payload = [0b0001_0000u8, 0b0000_0000, 0x10, 0x33, 0xAB];
        assert!(parse_subpic_level_info(&payload).is_err());
    }

    /// A body too short even for the entry run after the header is
    /// rejected (the bit reader runs out of bits).
    #[test]
    fn truncated_entry_rejected() {
        // Header announces one entry (2 bytes) but supplies one byte.
        let payload = [0b0001_0000u8, 0b0000_0000, 0x10];
        assert!(parse_subpic_level_info(&payload).is_err());
    }

    /// Explicit fractions with several subpictures across two ref levels.
    #[test]
    fn explicit_fractions_two_levels_three_subpics() {
        // num_ref_levels_minus1 = 1 (u3 = 001)
        // cbr = 1, explicit = 1
        // ue(v) num_subpics_minus1 = 2 -> code "011"
        // max_sublayers_minus1 = 0 (000), sublayer_info = 0 (0)
        // Header bits: 001 1 1 011 000 0
        //   001 1 1   (num_ref/cbr/explicit)
        //   011       (ue = 2)
        //   000       (max_sublayers_minus1)
        //   0         (sublayer_info)
        //   = 0011_1011 0000 -> 12 bits.
        // byte0 = 0b0011_1011, byte1 = 0000 + 4 zero pad = 0x00.
        // One sublayer, two entries; each entry = non_frac + idc + 3 frac.
        //   i=0: 0x10,0x40, fracs 0x01,0x02,0x03
        //   i=1: 0x20,0x41, fracs 0x04,0x05,0x06
        let payload = [
            0b0011_1011u8,
            0x00,
            0x10,
            0x40,
            0x01,
            0x02,
            0x03,
            0x20,
            0x41,
            0x04,
            0x05,
            0x06,
        ];
        let sli = parse_subpic_level_info(&payload).unwrap();
        assert!(sli.explicit_fraction_present_flag);
        assert_eq!(sli.num_subpics(), Some(3));
        assert_eq!(sli.num_ref_levels(), 2);
        let s = &sli.sublayers[0];
        assert_eq!(
            s.entries[0].ref_level_fraction_minus1,
            vec![0x01, 0x02, 0x03]
        );
        assert_eq!(s.entries[1].non_subpic_layers_fraction, 0x20);
        assert_eq!(s.entries[1].ref_level_idc, 0x41);
        assert_eq!(
            s.entries[1].ref_level_fraction_minus1,
            vec![0x04, 0x05, 0x06]
        );
    }

    /// `SUBPIC_LEVEL_INFO_PAYLOAD_TYPE` matches the §D.2 dispatch value.
    #[test]
    fn payload_type_constant() {
        assert_eq!(SUBPIC_LEVEL_INFO_PAYLOAD_TYPE, 203);
    }
}
