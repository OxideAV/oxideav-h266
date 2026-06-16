//! VVC DU information SEI message parser (§D.5.1 / §D.5.2).
//!
//! The decoding unit information (DUI) SEI message
//! (`payloadType == 130`) provides CPB removal delay information for the
//! decoding unit (DU) associated with the SEI message. Like the SEI
//! manifest (`payloadType == 200`), the SEI prefix indication
//! (`payloadType == 201`), and the subpicture level information
//! (`payloadType == 203`), it is one of the few `sei_payload()` bodies
//! specified directly in this Specification (most other payload types
//! defer to Rec. ITU-T H.274 | ISO/IEC 23002-7).
//!
//! Unlike those three, the DUI body is *not* self-describing: several of
//! its syntax elements are present only under conditions carried by the
//! buffering period (BP) SEI message, and two of them are `u(v)` with a
//! length taken from the BP SEI message. So this parser takes a
//! [`DuiContext`] supplying the §D.3.2 BP-derived values that gate /
//! size the DUI elements, plus the `TemporalId` of the SEI NAL unit
//! carrying the message (which sets the lower bound of the per-sublayer
//! loop).
//!
//! Syntax (§D.5.1):
//!
//! ```text
//!   decoding_unit_info( payloadSize ) {                               Descriptor
//!       dui_decoding_unit_idx                                            ue(v)
//!       if( !bp_du_cpb_params_in_pic_timing_sei_flag )
//!           for( i = TemporalId; i <= bp_max_sublayers_minus1; i++ ) {
//!               if( i < bp_max_sublayers_minus1 )
//!                   dui_sublayer_delays_present_flag[ i ]                u(1)
//!               if( dui_sublayer_delays_present_flag[ i ] )
//!                   dui_du_cpb_removal_delay_increment[ i ]             u(v)
//!           }
//!       if( !bp_du_dpb_params_in_pic_timing_sei_flag )
//!           dui_dpb_output_du_delay_present_flag                         u(1)
//!       if( dui_dpb_output_du_delay_present_flag )
//!           dui_dpb_output_du_delay                                      u(v)
//!   }
//! ```
//!
//! §D.5.2 semantics modelled here:
//!
//! * `dui_decoding_unit_idx` — the index (from 0) into the list of DUs
//!   in the current AU of the DU associated with this message.
//! * `dui_sublayer_delays_present_flag[ i ]` — `1` if
//!   `dui_du_cpb_removal_delay_increment[ i ]` is present for the
//!   sublayer with `TemporalId == i`. Inference (§D.5.2): when not
//!   present, it is `1` if `bp_du_cpb_params_in_pic_timing_sei_flag` is
//!   `0` and `i == bp_max_sublayers_minus1`, otherwise `0`. In the
//!   syntax the flag is only read for `i < bp_max_sublayers_minus1`, so
//!   the top sublayer's flag is always the inferred one (`1` here, since
//!   the whole loop is gated on `!bp_du_cpb_params_in_pic_timing_sei_flag`).
//! * `dui_du_cpb_removal_delay_increment[ i ]` — the inter-DU CPB
//!   removal delay, in clock sub-ticks, `u(v)` with length
//!   `bp_du_cpb_removal_delay_increment_length_minus1 + 1`.
//! * `dui_dpb_output_du_delay_present_flag` — `1` if
//!   `dui_dpb_output_du_delay` is present. When not present (i.e. when
//!   `bp_du_dpb_params_in_pic_timing_sei_flag` is `1`) it is inferred
//!   `0`.
//! * `dui_dpb_output_du_delay` — `u(v)` with length
//!   `bp_dpb_output_delay_du_length_minus1 + 1`.
//!
//! The whole DUI body is a bit-packed structure with no internal
//! byte-alignment; §7.4.6 requires the derived `payloadSize` to equal
//! the number of payload bytes, with the `sei_payload()` tail
//! (`sei_payload_bit_equal_to_one` then zero padding to a byte boundary,
//! per §D.2.1) accounting for the bits between the end of the structure
//! and the byte boundary. This parser consumes the structure and then
//! validates that tail: exactly one `1` bit followed by `0` bits to the
//! byte boundary, landing exactly at `payload.len()`. A framing error
//! therefore surfaces as a parse error rather than desynchronising the
//! enclosing `sei_rbsp()` walk.
//!
//! Spec reference: ITU-T H.266 | ISO/IEC 23090-3 (V4, 01/2026),
//! §D.5.1, §D.5.2, §D.3.2 (BP-derived context), §D.2.1 (`sei_payload()`
//! dispatch + trailing bits), §7.4.6.
//!
//! No third-party VVC decoder source was consulted; the implementation
//! is spec-only and reads the payload through the crate's own
//! [`BitReader`].

use oxideav_core::{Error, Result};

use crate::bitreader::BitReader;

/// `payloadType` value that selects the DU information body in the §D.2
/// `sei_payload()` dispatch.
pub const DECODING_UNIT_INFO_PAYLOAD_TYPE: u32 = 130;

/// The buffering-period-derived context required to parse a DUI SEI
/// message (§D.5.1).
///
/// These values come from the BP SEI message (§D.3.1) applicable to the
/// operation point(s) to which the DUI message applies, plus the
/// `TemporalId` of the SEI NAL unit carrying the DUI message (§D.5.2:
/// "The `TemporalId` in the DUI SEI message syntax is the `TemporalId`
/// of the SEI NAL unit containing the DUI SEI message").
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DuiContext {
    /// `TemporalId` of the SEI NAL unit carrying the DUI message — the
    /// lower bound of the per-sublayer `i` loop.
    pub temporal_id: u8,
    /// `bp_max_sublayers_minus1` — the upper bound of the per-sublayer
    /// `i` loop (§D.3.2).
    pub bp_max_sublayers_minus1: u8,
    /// `bp_du_cpb_params_in_pic_timing_sei_flag` (§D.3.2) — when `true`
    /// the whole per-sublayer CPB-removal-delay loop is absent (the
    /// equivalent values live in the PT SEI message instead).
    pub bp_du_cpb_params_in_pic_timing_sei_flag: bool,
    /// `bp_du_dpb_params_in_pic_timing_sei_flag` (§D.3.2) — when `true`
    /// `dui_dpb_output_du_delay_present_flag` / `dui_dpb_output_du_delay`
    /// are absent.
    pub bp_du_dpb_params_in_pic_timing_sei_flag: bool,
    /// `bp_du_cpb_removal_delay_increment_length_minus1` (§D.3.2) — the
    /// `u(v)` length of `dui_du_cpb_removal_delay_increment[ i ]` is this
    /// plus 1.
    pub bp_du_cpb_removal_delay_increment_length_minus1: u8,
    /// `bp_dpb_output_delay_du_length_minus1` (§D.3.2) — the `u(v)`
    /// length of `dui_dpb_output_du_delay` is this plus 1.
    pub bp_dpb_output_delay_du_length_minus1: u8,
}

/// One per-sublayer entry of the DUI CPB-removal-delay loop (§D.5.1),
/// for the sublayer with `TemporalId == sublayer_idx`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct DuiSublayerDelay {
    /// The sublayer index `i` this entry describes (`TemporalId == i`).
    pub sublayer_idx: u8,
    /// `dui_sublayer_delays_present_flag[ i ]` — read for
    /// `i < bp_max_sublayers_minus1`, inferred for the top sublayer
    /// (§D.5.2).
    pub sublayer_delays_present_flag: bool,
    /// `dui_du_cpb_removal_delay_increment[ i ]` — present (non-`None`)
    /// only when `sublayer_delays_present_flag` is `true`.
    pub du_cpb_removal_delay_increment: Option<u32>,
}

/// A parsed DU information SEI message (§D.5.1).
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct DecodingUnitInfo {
    /// `dui_decoding_unit_idx`.
    pub decoding_unit_idx: u32,
    /// The per-sublayer CPB-removal-delay entries, in the stream order
    /// the `i` loop visits them (`TemporalId ..= bp_max_sublayers_minus1`).
    /// Empty when `bp_du_cpb_params_in_pic_timing_sei_flag` is `true`.
    pub sublayer_delays: Vec<DuiSublayerDelay>,
    /// `dui_dpb_output_du_delay_present_flag` — inferred `false` when
    /// `bp_du_dpb_params_in_pic_timing_sei_flag` is `true` (the flag is
    /// then absent from the bitstream).
    pub dpb_output_du_delay_present_flag: bool,
    /// `dui_dpb_output_du_delay` — present (non-`None`) only when
    /// `dpb_output_du_delay_present_flag` is `true`.
    pub dpb_output_du_delay: Option<u32>,
}

impl DecodingUnitInfo {
    /// The per-sublayer entry describing sublayer index `i`, if present.
    pub fn sublayer_delay(&self, i: u8) -> Option<&DuiSublayerDelay> {
        self.sublayer_delays.iter().find(|s| s.sublayer_idx == i)
    }
}

/// Read the `sei_payload()` trailing bits (§D.2.1) and require the reader
/// to land exactly at `payload.len()`.
///
/// After the structured body, `sei_payload()` carries (when the body did
/// not already end on a byte boundary, or when extension bits are
/// present) a single `sei_payload_bit_equal_to_one` (`f(1)`) followed by
/// `sei_payload_bit_equal_to_zero` bits to the byte boundary. For a body
/// whose end is *not* byte aligned the `1` bit is mandatory; for a body
/// that already ended byte aligned there are no further bits. §7.4.6
/// then requires the consumed length to equal `payloadSize`.
fn finish_sei_payload(reader: &mut BitReader, what: &str) -> Result<()> {
    if reader.is_byte_aligned() {
        // Body ended on a byte boundary: there must be no trailing bits.
        if reader.bits_remaining() != 0 {
            return Err(Error::invalid(what));
        }
        return Ok(());
    }
    // sei_payload_bit_equal_to_one
    if reader.u1()? != 1 {
        return Err(Error::invalid(what));
    }
    // sei_payload_bit_equal_to_zero* up to the byte boundary.
    while !reader.is_byte_aligned() {
        if reader.u1()? != 0 {
            return Err(Error::invalid(what));
        }
    }
    if reader.bits_remaining() != 0 {
        return Err(Error::invalid(what));
    }
    Ok(())
}

/// Parse a `decoding_unit_info( payloadSize )` body (§D.5.1) from the raw
/// SEI payload bytes carried by a `payloadType == 130` `sei_message()`.
///
/// `payload` is the `sei_payload()` argument region — the `payloadSize`
/// bytes that follow the `sei_message()` header, with emulation-
/// prevention bytes already removed. `ctx` supplies the §D.3.2
/// BP-derived gating flags / `u(v)` lengths plus the SEI NAL unit's
/// `TemporalId`.
///
/// Errors:
/// * a `payload` too short for the structure is rejected (propagated
///   from the bit reader);
/// * a malformed §D.2.1 trailing-bits region (missing
///   `sei_payload_bit_equal_to_one`, a non-`0` zero-padding bit, or a
///   consumed length that does not equal `payloadSize`) is rejected;
/// * `ctx.temporal_id > ctx.bp_max_sublayers_minus1` makes the
///   per-sublayer loop empty (a degenerate but well-formed case).
pub fn parse_decoding_unit_info(payload: &[u8], ctx: &DuiContext) -> Result<DecodingUnitInfo> {
    let mut reader = BitReader::new(payload);

    let decoding_unit_idx = reader.ue()?;

    let cpb_inc_len = u32::from(ctx.bp_du_cpb_removal_delay_increment_length_minus1) + 1;
    let dpb_delay_len = u32::from(ctx.bp_dpb_output_delay_du_length_minus1) + 1;

    let mut sublayer_delays = Vec::new();
    if !ctx.bp_du_cpb_params_in_pic_timing_sei_flag {
        // for( i = TemporalId; i <= bp_max_sublayers_minus1; i++ )
        for i in ctx.temporal_id..=ctx.bp_max_sublayers_minus1 {
            // if( i < bp_max_sublayers_minus1 )
            //     dui_sublayer_delays_present_flag[ i ]   u(1)
            // else inferred: 1 (since this loop runs only when
            //     bp_du_cpb_params_in_pic_timing_sei_flag == 0, the
            //     §D.5.2 inference for i == bp_max_sublayers_minus1
            //     gives 1).
            let present = if i < ctx.bp_max_sublayers_minus1 {
                reader.u1()? != 0
            } else {
                true
            };
            let du_cpb_removal_delay_increment = if present {
                Some(reader.u(cpb_inc_len)?)
            } else {
                None
            };
            sublayer_delays.push(DuiSublayerDelay {
                sublayer_idx: i,
                sublayer_delays_present_flag: present,
                du_cpb_removal_delay_increment,
            });
        }
    }

    // if( !bp_du_dpb_params_in_pic_timing_sei_flag )
    //     dui_dpb_output_du_delay_present_flag    u(1)
    // else inferred 0.
    let dpb_output_du_delay_present_flag = if ctx.bp_du_dpb_params_in_pic_timing_sei_flag {
        false
    } else {
        reader.u1()? != 0
    };

    let dpb_output_du_delay = if dpb_output_du_delay_present_flag {
        Some(reader.u(dpb_delay_len)?)
    } else {
        None
    };

    finish_sei_payload(
        &mut reader,
        "h266 decoding_unit_info: malformed sei_payload trailing bits / length (§D.2.1, §7.4.6)",
    )?;

    Ok(DecodingUnitInfo {
        decoding_unit_idx,
        sublayer_delays,
        dpb_output_du_delay_present_flag,
        dpb_output_du_delay,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A small bit packer for building DUI test payloads.
    struct BitWriter {
        bits: Vec<u8>,
    }

    impl BitWriter {
        fn new() -> Self {
            Self { bits: Vec::new() }
        }

        fn put(&mut self, value: u32, n: u32) {
            for k in (0..n).rev() {
                self.bits.push(((value >> k) & 1) as u8);
            }
        }

        /// Exp-Golomb ue(v).
        fn ue(&mut self, value: u32) {
            let code = value + 1;
            let len = 32 - code.leading_zeros();
            // len - 1 leading zeros.
            for _ in 0..(len - 1) {
                self.bits.push(0);
            }
            self.put(code, len);
        }

        /// Append `sei_payload()` trailing bits (§D.2.1) and pack to bytes.
        fn finish_payload(mut self) -> Vec<u8> {
            if self.bits.len() % 8 != 0 {
                self.bits.push(1); // sei_payload_bit_equal_to_one
                while self.bits.len() % 8 != 0 {
                    self.bits.push(0);
                }
            }
            let mut out = Vec::with_capacity(self.bits.len() / 8);
            for chunk in self.bits.chunks(8) {
                let mut b = 0u8;
                for &bit in chunk {
                    b = (b << 1) | bit;
                }
                out.push(b);
            }
            out
        }
    }

    fn ctx_default() -> DuiContext {
        DuiContext {
            temporal_id: 0,
            bp_max_sublayers_minus1: 0,
            bp_du_cpb_params_in_pic_timing_sei_flag: false,
            bp_du_dpb_params_in_pic_timing_sei_flag: false,
            bp_du_cpb_removal_delay_increment_length_minus1: 7, // 8-bit u(v)
            bp_dpb_output_delay_du_length_minus1: 7,            // 8-bit u(v)
        }
    }

    /// Single-sublayer case (`bp_max_sublayers_minus1 == 0`): the loop
    /// runs once for the top sublayer whose presence flag is inferred 1,
    /// so the increment is always read, followed by the DPB-output delay.
    #[test]
    fn single_sublayer_with_dpb_delay() {
        let ctx = ctx_default();
        let mut w = BitWriter::new();
        w.ue(2); // dui_decoding_unit_idx
                 // i = 0 == bp_max_sublayers_minus1 -> flag inferred 1, no u(1).
        w.put(0x55, 8); // dui_du_cpb_removal_delay_increment[0]
        w.put(1, 1); // dui_dpb_output_du_delay_present_flag
        w.put(0xAB, 8); // dui_dpb_output_du_delay
        let payload = w.finish_payload();

        let dui = parse_decoding_unit_info(&payload, &ctx).unwrap();
        assert_eq!(dui.decoding_unit_idx, 2);
        assert_eq!(dui.sublayer_delays.len(), 1);
        let s = &dui.sublayer_delays[0];
        assert_eq!(s.sublayer_idx, 0);
        assert!(s.sublayer_delays_present_flag);
        assert_eq!(s.du_cpb_removal_delay_increment, Some(0x55));
        assert!(dui.dpb_output_du_delay_present_flag);
        assert_eq!(dui.dpb_output_du_delay, Some(0xAB));
        assert_eq!(dui.sublayer_delay(0).map(|s| s.sublayer_idx), Some(0));
        assert_eq!(dui.sublayer_delay(1), None);
    }

    /// `dui_dpb_output_du_delay_present_flag == 0` suppresses the delay.
    #[test]
    fn single_sublayer_no_dpb_delay() {
        let ctx = ctx_default();
        let mut w = BitWriter::new();
        w.ue(0);
        w.put(0x10, 8); // increment for the inferred top sublayer
        w.put(0, 1); // dpb_output_du_delay_present_flag = 0
        let payload = w.finish_payload();

        let dui = parse_decoding_unit_info(&payload, &ctx).unwrap();
        assert_eq!(dui.decoding_unit_idx, 0);
        assert!(!dui.dpb_output_du_delay_present_flag);
        assert_eq!(dui.dpb_output_du_delay, None);
        assert_eq!(
            dui.sublayer_delays[0].du_cpb_removal_delay_increment,
            Some(0x10)
        );
    }

    /// Multiple sublayers: lower sublayers carry an explicit
    /// `dui_sublayer_delays_present_flag`, the top one is inferred 1.
    #[test]
    fn multi_sublayer_explicit_flags() {
        let ctx = DuiContext {
            bp_max_sublayers_minus1: 2,
            bp_du_cpb_removal_delay_increment_length_minus1: 3, // 4-bit u(v)
            ..ctx_default()
        };
        let mut w = BitWriter::new();
        w.ue(1); // decoding_unit_idx
                 // i = 0: i < 2 -> read flag = 1, then 4-bit increment 0x3
        w.put(1, 1);
        w.put(0x3, 4);
        // i = 1: i < 2 -> read flag = 0, no increment
        w.put(0, 1);
        // i = 2 == bp_max_sublayers_minus1 -> inferred 1, 4-bit 0x7
        w.put(0x7, 4);
        // dpb_output_du_delay_present_flag = 0
        w.put(0, 1);
        let payload = w.finish_payload();

        let dui = parse_decoding_unit_info(&payload, &ctx).unwrap();
        assert_eq!(dui.sublayer_delays.len(), 3);
        assert_eq!(dui.sublayer_delays[0].sublayer_idx, 0);
        assert!(dui.sublayer_delays[0].sublayer_delays_present_flag);
        assert_eq!(
            dui.sublayer_delays[0].du_cpb_removal_delay_increment,
            Some(0x3)
        );
        assert_eq!(dui.sublayer_delays[1].sublayer_idx, 1);
        assert!(!dui.sublayer_delays[1].sublayer_delays_present_flag);
        assert_eq!(dui.sublayer_delays[1].du_cpb_removal_delay_increment, None);
        assert_eq!(dui.sublayer_delays[2].sublayer_idx, 2);
        assert!(dui.sublayer_delays[2].sublayer_delays_present_flag);
        assert_eq!(
            dui.sublayer_delays[2].du_cpb_removal_delay_increment,
            Some(0x7)
        );
    }

    /// `temporal_id > 0` raises the loop's lower bound: the lower
    /// sublayers are skipped entirely.
    #[test]
    fn temporal_id_raises_loop_lower_bound() {
        let ctx = DuiContext {
            temporal_id: 2,
            bp_max_sublayers_minus1: 2,
            bp_du_cpb_removal_delay_increment_length_minus1: 7,
            ..ctx_default()
        };
        let mut w = BitWriter::new();
        w.ue(0);
        // Only i = 2 == bp_max_sublayers_minus1 -> inferred flag 1.
        w.put(0x9, 8);
        w.put(0, 1); // dpb flag
        let payload = w.finish_payload();

        let dui = parse_decoding_unit_info(&payload, &ctx).unwrap();
        assert_eq!(dui.sublayer_delays.len(), 1);
        assert_eq!(dui.sublayer_delays[0].sublayer_idx, 2);
        assert_eq!(
            dui.sublayer_delays[0].du_cpb_removal_delay_increment,
            Some(0x9)
        );
    }

    /// `bp_du_cpb_params_in_pic_timing_sei_flag == 1` removes the whole
    /// per-sublayer loop.
    #[test]
    fn cpb_params_in_pt_sei_skips_sublayer_loop() {
        let ctx = DuiContext {
            bp_du_cpb_params_in_pic_timing_sei_flag: true,
            ..ctx_default()
        };
        let mut w = BitWriter::new();
        w.ue(3);
        w.put(1, 1); // dpb_output_du_delay_present_flag
        w.put(0x42, 8);
        let payload = w.finish_payload();

        let dui = parse_decoding_unit_info(&payload, &ctx).unwrap();
        assert_eq!(dui.decoding_unit_idx, 3);
        assert!(dui.sublayer_delays.is_empty());
        assert_eq!(dui.dpb_output_du_delay, Some(0x42));
    }

    /// `bp_du_dpb_params_in_pic_timing_sei_flag == 1` removes the
    /// DPB-output-delay flag + value (inferred present-flag 0).
    #[test]
    fn dpb_params_in_pt_sei_skips_dpb_delay() {
        let ctx = DuiContext {
            bp_du_dpb_params_in_pic_timing_sei_flag: true,
            ..ctx_default()
        };
        let mut w = BitWriter::new();
        w.ue(0);
        w.put(0x12, 8); // top-sublayer increment (inferred flag 1)
        let payload = w.finish_payload();

        let dui = parse_decoding_unit_info(&payload, &ctx).unwrap();
        assert!(!dui.dpb_output_du_delay_present_flag);
        assert_eq!(dui.dpb_output_du_delay, None);
        assert_eq!(
            dui.sublayer_delays[0].du_cpb_removal_delay_increment,
            Some(0x12)
        );
    }

    /// Variable `u(v)` lengths are honoured: a 13-bit increment and a
    /// 20-bit DPB delay round-trip their values.
    #[test]
    fn variable_length_uv_fields() {
        let ctx = DuiContext {
            bp_du_cpb_removal_delay_increment_length_minus1: 12, // 13-bit
            bp_dpb_output_delay_du_length_minus1: 19,            // 20-bit
            ..ctx_default()
        };
        let mut w = BitWriter::new();
        w.ue(0);
        w.put(0x1ABC, 13); // 13-bit increment
        w.put(1, 1);
        w.put(0xFEDC0, 20); // 20-bit DPB delay
        let payload = w.finish_payload();

        let dui = parse_decoding_unit_info(&payload, &ctx).unwrap();
        assert_eq!(
            dui.sublayer_delays[0].du_cpb_removal_delay_increment,
            Some(0x1ABC)
        );
        assert_eq!(dui.dpb_output_du_delay, Some(0xFEDC0));
    }

    /// A body that ends exactly on a byte boundary carries no trailing
    /// bits and must still parse (the §D.2.1 `1`/`0` tail is absent).
    #[test]
    fn byte_aligned_body_no_trailing_bits() {
        // Choose lengths so the structure is a whole number of bits.
        // ue(0) = 1 bit, top-sublayer increment = 6 bits, dpb flag = 1
        // bit -> 8 bits total, already byte aligned, no trailing tail.
        let ctx = DuiContext {
            bp_du_cpb_removal_delay_increment_length_minus1: 5, // 6-bit
            bp_du_dpb_params_in_pic_timing_sei_flag: false,
            ..ctx_default()
        };
        let mut w = BitWriter::new();
        w.ue(0); // 1 bit
        w.put(0x2A, 6); // 6 bits
        w.put(0, 1); // dpb present flag = 0 -> 1 bit; total 8.
        assert_eq!(w.bits.len(), 8);
        let payload = w.finish_payload();
        assert_eq!(payload.len(), 1);

        let dui = parse_decoding_unit_info(&payload, &ctx).unwrap();
        assert_eq!(
            dui.sublayer_delays[0].du_cpb_removal_delay_increment,
            Some(0x2A)
        );
        assert!(!dui.dpb_output_du_delay_present_flag);
    }

    /// A truncated body (not enough bits for the announced `u(v)`) is
    /// rejected.
    #[test]
    fn truncated_body_rejected() {
        let ctx = ctx_default();
        // Only one byte: ue(0)=1 bit then 7 bits, not enough for the
        // 8-bit increment plus the dpb flag.
        let payload = [0b0000_0000u8];
        assert!(parse_decoding_unit_info(&payload, &ctx).is_err());
    }

    /// A body with extra trailing bytes beyond the §D.2.1 tail is
    /// rejected (the length cross-check fails).
    #[test]
    fn trailing_bytes_rejected() {
        let ctx = ctx_default();
        let mut w = BitWriter::new();
        w.ue(0);
        w.put(0x10, 8);
        w.put(0, 1);
        let mut payload = w.finish_payload();
        payload.push(0xAB); // stray extra byte
        assert!(parse_decoding_unit_info(&payload, &ctx).is_err());
    }

    /// A malformed trailing region (zero where the
    /// `sei_payload_bit_equal_to_one` should be) is rejected.
    #[test]
    fn missing_payload_stop_bit_rejected() {
        let ctx = ctx_default();
        // Build a body that does not end byte aligned (so a 1-stop-bit
        // is required), but hand-pack the final byte with a 0 there.
        // ue(0)=1, increment 8 bits, dpb flag 1 bit -> 10 bits, needs
        // a 1 then 6 zeros. Replace the stop bit with 0.
        let mut bits: Vec<u8> = Vec::new();
        bits.push(1); // ue(0)
        for k in (0..8).rev() {
            bits.push((0x10u32 >> k) as u8 & 1);
        }
        bits.push(0); // dpb present flag = 0 -> 10 bits
        bits.push(0); // WRONG: should be sei_payload_bit_equal_to_one (1)
        while bits.len() % 8 != 0 {
            bits.push(0);
        }
        let mut payload = Vec::new();
        for chunk in bits.chunks(8) {
            let mut b = 0u8;
            for &bit in chunk {
                b = (b << 1) | bit;
            }
            payload.push(b);
        }
        assert!(parse_decoding_unit_info(&payload, &ctx).is_err());
    }

    /// `DECODING_UNIT_INFO_PAYLOAD_TYPE` matches the §D.2 dispatch value.
    #[test]
    fn payload_type_constant() {
        assert_eq!(DECODING_UNIT_INFO_PAYLOAD_TYPE, 130);
    }
}
