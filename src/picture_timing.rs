//! §D.4.1 / §D.4.2 — Picture timing (PT) SEI message (`payloadType == 1`).
//!
//! The PT SEI message provides the per-AU CPB removal delay and DPB
//! output delay information for the HRD (§D.4.2). Together with the
//! buffering period (BP, §D.3, [`crate::buffering_period`]) and the DU
//! information (DUI, §D.5, [`crate::decoding_unit_info`]) SEI messages
//! it forms the HRD-timing SEI family specified directly in this
//! Specification rather than deferred to Rec. ITU-T H.274 |
//! ISO/IEC 23002-7.
//!
//! Like the DUI message, the PT message has no self-describing lengths:
//! every `u(v)` field's bit width and every conditional branch is gated
//! by the BP SEI message applicable to the AU. The §D.3.2 derivations
//! are surfaced from a parsed [`crate::buffering_period::BufferingPeriod`]
//! through [`crate::buffering_period::BufferingPeriod::pt_context`] into
//! a [`PtContext`]; only the SEI NAL unit's own `TemporalId` is supplied
//! externally (§D.4.2: "The `TemporalId` in the PT SEI message syntax is
//! the `TemporalId` of the SEI NAL unit containing the PT SEI message").
//!
//! Scope: this parser walks the §D.4.1 syntax end-to-end — the
//! sublayer CPB-removal-delay loop (with the `pt_cpb_removal_delay_delta`
//! / explicit-`minus1` branch), `pt_dpb_output_delay`, the
//! `bp_alt_cpb_params_present_flag`-gated alt-CPB timing block, the
//! `bp_du_*_in_pic_timing_sei_flag`-gated DU DPB / CPB blocks (including
//! the per-DU NAL-count + increment loops), the concatenation flag, and
//! the trailing `pt_display_elemental_periods_minus1` `u(8)`. The §D.2.1
//! `sei_payload()` trailing bits are verified and the §7.4.6
//! `payloadSize` cross-checked by requiring the reader to land exactly
//! at the payload end. The implementation is spec-only and reads the
//! payload through the crate's own [`BitReader`].

use oxideav_core::{Error, Result};

use crate::bitreader::BitReader;

/// `payloadType` value that selects the picture timing body in the §D.2
/// `sei_payload()` dispatch.
pub const PICTURE_TIMING_PAYLOAD_TYPE: u32 = 1;

/// The buffering-period-derived context required to parse a PT SEI
/// message (§D.4.1).
///
/// These values come from the BP SEI message (§D.3.1) applicable to the
/// operation point(s) to which the PT message applies, plus the
/// `TemporalId` of the SEI NAL unit carrying the PT message (§D.4.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PtContext {
    /// `TemporalId` of the SEI NAL unit carrying the PT message — the
    /// lower bound of the per-sublayer `i` loop.
    pub temporal_id: u8,
    /// `bp_max_sublayers_minus1` (§D.3.2) — the upper bound of the
    /// per-sublayer / per-DU loops.
    pub bp_max_sublayers_minus1: u8,
    /// `bp_cpb_removal_delay_length_minus1` (§D.3.2) — the `u(v)` length
    /// of `pt_cpb_removal_delay_minus1[ i ]` is this plus 1.
    pub bp_cpb_removal_delay_length_minus1: u8,
    /// `bp_dpb_output_delay_length_minus1` (§D.3.2) — the `u(v)` length
    /// of `pt_dpb_output_delay` is this plus 1.
    pub bp_dpb_output_delay_length_minus1: u8,
    /// `bp_cpb_initial_removal_delay_length_minus1` (§D.3.1) — the `u(v)`
    /// length of the alt-CPB `pt_*_cpb_alt_initial_removal_delay/offset
    /// _delta[ i ][ j ]` fields is this plus 1 (§D.4.2).
    pub bp_cpb_initial_removal_delay_length_minus1: u8,
    /// `bp_cpb_removal_delay_deltas_present_flag` (§D.3.2) — gates the
    /// `pt_cpb_removal_delay_delta_enabled_flag[ i ]` read.
    pub bp_cpb_removal_delay_deltas_present_flag: bool,
    /// `bp_num_cpb_removal_delay_deltas_minus1` (§D.3.2) — `> 0` gates the
    /// `pt_cpb_removal_delay_delta_idx[ i ]` read; its `u(v)` length is
    /// `Ceil( Log2( bp_num_cpb_removal_delay_deltas_minus1 + 1 ) )` bits.
    pub bp_num_cpb_removal_delay_deltas_minus1: u32,
    /// `bp_alt_cpb_params_present_flag` (§D.3.2) — gates the alt-CPB
    /// timing-info block.
    pub bp_alt_cpb_params_present_flag: bool,
    /// `bp_nal_hrd_params_present_flag` (§D.3.1) — gates the NAL alt-CPB
    /// sublayer loop.
    pub bp_nal_hrd_params_present_flag: bool,
    /// `bp_vcl_hrd_params_present_flag` (§D.3.1) — gates the VCL alt-CPB
    /// sublayer loop.
    pub bp_vcl_hrd_params_present_flag: bool,
    /// `bp_sublayer_initial_cpb_removal_delay_present_flag` (§D.3.1) —
    /// selects the lower bound of the alt-CPB sublayer loops.
    pub bp_sublayer_initial_cpb_removal_delay_present_flag: bool,
    /// `bp_cpb_cnt_minus1` (§D.3.1) — the per-CPB-index loop bound of the
    /// alt-CPB delay/offset-delta pairs.
    pub bp_cpb_cnt_minus1: u32,
    /// `bp_du_hrd_params_present_flag` (§D.3.1) — gates both DU blocks.
    pub bp_du_hrd_params_present_flag: bool,
    /// `bp_du_dpb_params_in_pic_timing_sei_flag` (§D.3.2) — with
    /// `bp_du_hrd_params_present_flag`, gates `pt_dpb_output_du_delay`.
    pub bp_du_dpb_params_in_pic_timing_sei_flag: bool,
    /// `bp_du_cpb_params_in_pic_timing_sei_flag` (§D.3.2) — with
    /// `bp_du_hrd_params_present_flag`, gates the DU CPB block.
    pub bp_du_cpb_params_in_pic_timing_sei_flag: bool,
    /// `bp_du_cpb_removal_delay_increment_length_minus1` (§D.3.2) — the
    /// `u(v)` length of `pt_du_common_cpb_removal_delay_increment_minus1`
    /// / `pt_du_cpb_removal_delay_increment_minus1` is this plus 1.
    pub bp_du_cpb_removal_delay_increment_length_minus1: u8,
    /// `bp_dpb_output_delay_du_length_minus1` (§D.3.2) — the `u(v)` length
    /// of `pt_dpb_output_du_delay` is this plus 1.
    pub bp_dpb_output_delay_du_length_minus1: u8,
    /// `bp_additional_concatenation_info_present_flag` (§D.3.1) — gates
    /// `pt_delay_for_concatenation_ensured_flag`.
    pub bp_additional_concatenation_info_present_flag: bool,
}

/// One per-sublayer entry of the PT CPB-removal-delay loop (§D.4.1),
/// for the sublayer with `TemporalId == sublayer_idx`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct PtSublayerDelay {
    /// The sublayer index `i` this entry describes (`TemporalId == i`).
    pub sublayer_idx: u8,
    /// `pt_sublayer_delays_present_flag[ i ]`.
    pub sublayer_delays_present_flag: bool,
    /// `pt_cpb_removal_delay_delta_enabled_flag[ i ]` (inferred `false`
    /// when absent, §D.4.2).
    pub cpb_removal_delay_delta_enabled_flag: bool,
    /// `pt_cpb_removal_delay_delta_idx[ i ]` — present (non-`None`) only
    /// when `cpb_removal_delay_delta_enabled_flag` is `true`.
    pub cpb_removal_delay_delta_idx: Option<u32>,
    /// `pt_cpb_removal_delay_minus1[ i ]` — present (non-`None`) only when
    /// `sublayer_delays_present_flag` is `true` and the delta path is not
    /// taken.
    pub cpb_removal_delay_minus1: Option<u32>,
}

/// One per-CPB-index alt-CPB delay/offset-delta pair (§D.4.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct AltCpbEntry {
    /// `pt_{nal,vcl}_cpb_alt_initial_removal_delay_delta[ i ][ j ]`.
    pub initial_removal_delay_delta: u32,
    /// `pt_{nal,vcl}_cpb_alt_initial_removal_offset_delta[ i ][ j ]`.
    pub initial_removal_offset_delta: u32,
}

/// The per-sublayer alt-CPB timing block (§D.4.1), one for NAL and one
/// for VCL HRD when present.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct AltCpbSublayer {
    /// The sublayer index `i` this entry describes.
    pub sublayer_idx: u8,
    /// `pt_{nal,vcl}_cpb_alt_initial_removal_delay/offset_delta[ i ][ j ]`
    /// pairs (`j = 0 .. bp_cpb_cnt_minus1`).
    pub entries: Vec<AltCpbEntry>,
    /// `pt_{nal,vcl}_cpb_delay_offset[ i ]`.
    pub cpb_delay_offset: u32,
    /// `pt_{nal,vcl}_dpb_delay_offset[ i ]`.
    pub dpb_delay_offset: u32,
}

/// One per-DU entry of the §D.4.1 DU CPB loop.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct PtDecodingUnit {
    /// `pt_num_nalus_in_du_minus1[ i ]`.
    pub num_nalus_in_du_minus1: u32,
    /// `pt_du_cpb_removal_delay_increment_minus1[ i ][ j ]` — the
    /// per-sublayer increments read when the common-delay flag is `false`
    /// and `i < pt_num_decoding_units_minus1`. Empty otherwise.
    pub cpb_removal_delay_increment_minus1: Vec<u32>,
}

/// A parsed picture timing SEI message (§D.4.1).
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct PictureTiming {
    /// The per-sublayer CPB-removal-delay entries, in the order the `i`
    /// loop visits them (`TemporalId ..= bp_max_sublayers_minus1`). The
    /// first entry carries `pt_cpb_removal_delay_minus1[ bp_max_sublayers
    /// _minus1 ]` (the unconditional leading field) when the loop is
    /// empty; see [`Self::top_cpb_removal_delay_minus1`].
    pub sublayer_delays: Vec<PtSublayerDelay>,
    /// `pt_cpb_removal_delay_minus1[ bp_max_sublayers_minus1 ]` — the
    /// unconditional leading `u(v)` field of §D.4.1.
    pub top_cpb_removal_delay_minus1: u32,
    /// `pt_dpb_output_delay`.
    pub dpb_output_delay: u32,
    /// `pt_cpb_alt_timing_info_present_flag` (inferred `false` when the
    /// alt-CPB block is absent).
    pub cpb_alt_timing_info_present_flag: bool,
    /// NAL alt-CPB per-sublayer block (empty when absent).
    pub nal_alt_cpb: Vec<AltCpbSublayer>,
    /// VCL alt-CPB per-sublayer block (empty when absent).
    pub vcl_alt_cpb: Vec<AltCpbSublayer>,
    /// `pt_dpb_output_du_delay` — present only when the DU DPB block is
    /// signalled.
    pub dpb_output_du_delay: Option<u32>,
    /// `pt_num_decoding_units_minus1` — present only when the DU CPB block
    /// is signalled.
    pub num_decoding_units_minus1: Option<u32>,
    /// `pt_du_common_cpb_removal_delay_flag` (only meaningful when the DU
    /// CPB block is present and `num_decoding_units_minus1 > 0`).
    pub du_common_cpb_removal_delay_flag: bool,
    /// `pt_du_common_cpb_removal_delay_increment_minus1[ i ]` per sublayer
    /// (read only when `du_common_cpb_removal_delay_flag` is `true`).
    pub du_common_cpb_removal_delay_increment_minus1: Vec<u32>,
    /// The per-DU entries (empty when the DU CPB block is absent or
    /// `num_decoding_units_minus1 == 0`).
    pub decoding_units: Vec<PtDecodingUnit>,
    /// `pt_delay_for_concatenation_ensured_flag` — present only when
    /// `bp_additional_concatenation_info_present_flag` is `true`.
    pub delay_for_concatenation_ensured_flag: Option<bool>,
    /// `pt_display_elemental_periods_minus1` — the trailing `u(8)`.
    pub display_elemental_periods_minus1: u8,
}

/// `Ceil( Log2( n ) )` for `n >= 1` (§D.4.2 length formula).
fn ceil_log2(n: u32) -> u32 {
    if n <= 1 {
        0
    } else {
        32 - (n - 1).leading_zeros()
    }
}

/// Read the `sei_payload()` trailing bits (§D.2.1) and require the reader
/// to land exactly at `payload.len()` (§7.4.6 `payloadSize` cross-check).
fn finish_sei_payload(reader: &mut BitReader, what: &'static str) -> Result<()> {
    if reader.is_byte_aligned() {
        if reader.bits_remaining() != 0 {
            return Err(Error::invalid(what));
        }
        return Ok(());
    }
    if reader.u1()? != 1 {
        return Err(Error::invalid(what));
    }
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

/// Parse a `pic_timing( payloadSize )` body (§D.4.1) from the raw SEI
/// payload bytes carried by a `payloadType == 1` `sei_message()`.
///
/// `payload` is the `sei_payload()` argument region — the `payloadSize`
/// bytes that follow the `sei_message()` header, with emulation-
/// prevention bytes already removed. `ctx` supplies the §D.3.2 BP-derived
/// gating flags / `u(v)` lengths plus the SEI NAL unit's `TemporalId`.
///
/// Errors:
/// * a `payload` too short for the structure is rejected (propagated from
///   the bit reader);
/// * a malformed §D.2.1 trailing-bits region (missing
///   `sei_payload_bit_equal_to_one`, a non-`0` zero-padding bit, or a
///   consumed length that does not equal `payloadSize`) is rejected.
pub fn parse_picture_timing(payload: &[u8], ctx: &PtContext) -> Result<PictureTiming> {
    let mut reader = BitReader::new(payload);
    let mut pt = PictureTiming::default();

    let cpb_delay_len = u32::from(ctx.bp_cpb_removal_delay_length_minus1) + 1;
    let dpb_delay_len = u32::from(ctx.bp_dpb_output_delay_length_minus1) + 1;
    let init_delay_len = u32::from(ctx.bp_cpb_initial_removal_delay_length_minus1) + 1;
    let delta_idx_len = ceil_log2(ctx.bp_num_cpb_removal_delay_deltas_minus1 + 1);

    // Unconditional leading field:
    //   pt_cpb_removal_delay_minus1[ bp_max_sublayers_minus1 ]   u(v)
    pt.top_cpb_removal_delay_minus1 = reader.u(cpb_delay_len)?;

    // for( i = TemporalId; i < bp_max_sublayers_minus1; i++ )
    if ctx.temporal_id < ctx.bp_max_sublayers_minus1 {
        for i in ctx.temporal_id..ctx.bp_max_sublayers_minus1 {
            let mut entry = PtSublayerDelay {
                sublayer_idx: i,
                ..PtSublayerDelay::default()
            };
            // pt_sublayer_delays_present_flag[ i ]   u(1)
            entry.sublayer_delays_present_flag = reader.u1()? != 0;
            if entry.sublayer_delays_present_flag {
                // if( bp_cpb_removal_delay_deltas_present_flag )
                //     pt_cpb_removal_delay_delta_enabled_flag[ i ]   u(1)
                if ctx.bp_cpb_removal_delay_deltas_present_flag {
                    entry.cpb_removal_delay_delta_enabled_flag = reader.u1()? != 0;
                }
                if entry.cpb_removal_delay_delta_enabled_flag {
                    // if( bp_num_cpb_removal_delay_deltas_minus1 > 0 )
                    //     pt_cpb_removal_delay_delta_idx[ i ]   u(v)
                    // else inferred 0 (§D.4.2).
                    let idx = if ctx.bp_num_cpb_removal_delay_deltas_minus1 > 0 {
                        reader.u(delta_idx_len)?
                    } else {
                        0
                    };
                    entry.cpb_removal_delay_delta_idx = Some(idx);
                } else {
                    // pt_cpb_removal_delay_minus1[ i ]   u(v)
                    entry.cpb_removal_delay_minus1 = Some(reader.u(cpb_delay_len)?);
                }
            }
            pt.sublayer_delays.push(entry);
        }
    }

    // pt_dpb_output_delay   u(v)
    pt.dpb_output_delay = reader.u(dpb_delay_len)?;

    // if( bp_alt_cpb_params_present_flag ) { ... }
    if ctx.bp_alt_cpb_params_present_flag {
        pt.cpb_alt_timing_info_present_flag = reader.u1()? != 0;
        if pt.cpb_alt_timing_info_present_flag {
            let lo = if ctx.bp_sublayer_initial_cpb_removal_delay_present_flag {
                0
            } else {
                ctx.bp_max_sublayers_minus1
            };
            if ctx.bp_nal_hrd_params_present_flag {
                pt.nal_alt_cpb = read_alt_cpb_block(
                    &mut reader,
                    lo,
                    ctx.bp_max_sublayers_minus1,
                    ctx.bp_cpb_cnt_minus1,
                    init_delay_len,
                    cpb_delay_len,
                    dpb_delay_len,
                )?;
            }
            if ctx.bp_vcl_hrd_params_present_flag {
                pt.vcl_alt_cpb = read_alt_cpb_block(
                    &mut reader,
                    lo,
                    ctx.bp_max_sublayers_minus1,
                    ctx.bp_cpb_cnt_minus1,
                    init_delay_len,
                    cpb_delay_len,
                    dpb_delay_len,
                )?;
            }
        }
    }

    let du_inc_len = u32::from(ctx.bp_du_cpb_removal_delay_increment_length_minus1) + 1;
    let dpb_du_delay_len = u32::from(ctx.bp_dpb_output_delay_du_length_minus1) + 1;

    // if( bp_du_hrd_params_present_flag && bp_du_dpb_params_in_pic_timing_sei_flag )
    //     pt_dpb_output_du_delay   u(v)
    if ctx.bp_du_hrd_params_present_flag && ctx.bp_du_dpb_params_in_pic_timing_sei_flag {
        pt.dpb_output_du_delay = Some(reader.u(dpb_du_delay_len)?);
    }

    // if( bp_du_hrd_params_present_flag && bp_du_cpb_params_in_pic_timing_sei_flag ) { ... }
    if ctx.bp_du_hrd_params_present_flag && ctx.bp_du_cpb_params_in_pic_timing_sei_flag {
        let num_du_minus1 = reader.ue()?;
        pt.num_decoding_units_minus1 = Some(num_du_minus1);
        if num_du_minus1 > 0 {
            // pt_du_common_cpb_removal_delay_flag   u(1)
            pt.du_common_cpb_removal_delay_flag = reader.u1()? != 0;
            if pt.du_common_cpb_removal_delay_flag {
                // for( i = TemporalId; i <= bp_max_sublayers_minus1; i++ )
                //   if( pt_sublayer_delays_present_flag[ i ] )
                //     pt_du_common_cpb_removal_delay_increment_minus1[ i ]   u(v)
                for i in ctx.temporal_id..=ctx.bp_max_sublayers_minus1 {
                    if pt_sublayer_delays_present(&pt, ctx, i) {
                        pt.du_common_cpb_removal_delay_increment_minus1
                            .push(reader.u(du_inc_len)?);
                    }
                }
            }
            // for( i = 0; i <= pt_num_decoding_units_minus1; i++ )
            for i in 0..=num_du_minus1 {
                let mut du = PtDecodingUnit {
                    num_nalus_in_du_minus1: reader.ue()?,
                    ..PtDecodingUnit::default()
                };
                if !pt.du_common_cpb_removal_delay_flag && i < num_du_minus1 {
                    for j in ctx.temporal_id..=ctx.bp_max_sublayers_minus1 {
                        if pt_sublayer_delays_present(&pt, ctx, j) {
                            du.cpb_removal_delay_increment_minus1
                                .push(reader.u(du_inc_len)?);
                        }
                    }
                }
                pt.decoding_units.push(du);
            }
        }
    }

    // if( bp_additional_concatenation_info_present_flag )
    //     pt_delay_for_concatenation_ensured_flag   u(1)
    if ctx.bp_additional_concatenation_info_present_flag {
        pt.delay_for_concatenation_ensured_flag = Some(reader.u1()? != 0);
    }

    // pt_display_elemental_periods_minus1   u(8)
    pt.display_elemental_periods_minus1 = u8::try_from(reader.u(8)?).unwrap_or(0);

    finish_sei_payload(
        &mut reader,
        "h266 picture_timing: malformed §D.2.1 sei_payload() trailing bits or payloadSize mismatch",
    )?;
    Ok(pt)
}

/// §D.4.2 — `pt_sublayer_delays_present_flag[ i ]`: read for
/// `i < bp_max_sublayers_minus1`, inferred to 1 for the top sublayer.
fn pt_sublayer_delays_present(pt: &PictureTiming, ctx: &PtContext, i: u8) -> bool {
    if i == ctx.bp_max_sublayers_minus1 {
        return true;
    }
    pt.sublayer_delays
        .iter()
        .find(|s| s.sublayer_idx == i)
        .map(|s| s.sublayer_delays_present_flag)
        .unwrap_or(false)
}

/// Read one §D.4.1 alt-CPB per-sublayer block (NAL or VCL).
#[allow(clippy::too_many_arguments)]
fn read_alt_cpb_block(
    reader: &mut BitReader,
    lo: u8,
    hi: u8,
    cpb_cnt_minus1: u32,
    init_delay_len: u32,
    cpb_delay_len: u32,
    dpb_delay_len: u32,
) -> Result<Vec<AltCpbSublayer>> {
    let mut block = Vec::new();
    if lo > hi {
        return Ok(block);
    }
    for i in lo..=hi {
        let mut sub = AltCpbSublayer {
            sublayer_idx: i,
            ..AltCpbSublayer::default()
        };
        // The delta pairs are bp_cpb_initial_removal_delay_length_minus1
        // + 1 bits (§D.4.2); the cpb/dpb delay offsets use the
        // removal/output delay lengths respectively.
        for _ in 0..=cpb_cnt_minus1 {
            sub.entries.push(AltCpbEntry {
                initial_removal_delay_delta: reader.u(init_delay_len)?,
                initial_removal_offset_delta: reader.u(init_delay_len)?,
            });
        }
        sub.cpb_delay_offset = reader.u(cpb_delay_len)?;
        sub.dpb_delay_offset = reader.u(dpb_delay_len)?;
        block.push(sub);
    }
    Ok(block)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A small bit packer for building PT test payloads.
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

        fn ue(&mut self, value: u32) {
            let code = value + 1;
            let len = 32 - code.leading_zeros();
            for _ in 0..(len - 1) {
                self.bits.push(0);
            }
            self.put(code, len);
        }

        /// Append `sei_payload()` trailing bits (§D.2.1) and pack to bytes.
        fn finish_payload(mut self) -> Vec<u8> {
            if self.bits.len() % 8 != 0 {
                self.bits.push(1);
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

    /// A minimal single-sublayer context: no alt-CPB, no DU HRD, no
    /// concatenation. All `u(v)` lengths set to 8 bits for easy hand
    /// encoding.
    fn ctx_minimal() -> PtContext {
        PtContext {
            temporal_id: 0,
            bp_max_sublayers_minus1: 0,
            bp_cpb_removal_delay_length_minus1: 7,
            bp_dpb_output_delay_length_minus1: 7,
            bp_cpb_initial_removal_delay_length_minus1: 7,
            bp_cpb_removal_delay_deltas_present_flag: false,
            bp_num_cpb_removal_delay_deltas_minus1: 0,
            bp_alt_cpb_params_present_flag: false,
            bp_nal_hrd_params_present_flag: false,
            bp_vcl_hrd_params_present_flag: false,
            bp_sublayer_initial_cpb_removal_delay_present_flag: false,
            bp_cpb_cnt_minus1: 0,
            bp_du_hrd_params_present_flag: false,
            bp_du_dpb_params_in_pic_timing_sei_flag: false,
            bp_du_cpb_params_in_pic_timing_sei_flag: false,
            bp_du_cpb_removal_delay_increment_length_minus1: 7,
            bp_dpb_output_delay_du_length_minus1: 7,
            bp_additional_concatenation_info_present_flag: false,
        }
    }

    #[test]
    fn payload_type_constant_matches_spec() {
        assert_eq!(PICTURE_TIMING_PAYLOAD_TYPE, 1);
    }

    #[test]
    fn ceil_log2_matches_spec_formula() {
        assert_eq!(ceil_log2(1), 0);
        assert_eq!(ceil_log2(2), 1);
        assert_eq!(ceil_log2(3), 2);
        assert_eq!(ceil_log2(4), 2);
        assert_eq!(ceil_log2(5), 3);
        assert_eq!(ceil_log2(16), 4);
    }

    /// Single-sublayer minimal message: the unconditional leading
    /// `pt_cpb_removal_delay_minus1`, the per-sublayer loop is empty
    /// (`TemporalId == bp_max_sublayers_minus1`), `pt_dpb_output_delay`,
    /// then the trailing `pt_display_elemental_periods_minus1` `u(8)`.
    #[test]
    fn minimal_single_sublayer() {
        let ctx = ctx_minimal();
        let mut w = BitWriter::new();
        w.put(0x11, 8); // pt_cpb_removal_delay_minus1[top]
        w.put(0x22, 8); // pt_dpb_output_delay
        w.put(0x33, 8); // pt_display_elemental_periods_minus1
        let payload = w.finish_payload();

        let pt = parse_picture_timing(&payload, &ctx).unwrap();
        assert_eq!(pt.top_cpb_removal_delay_minus1, 0x11);
        assert!(pt.sublayer_delays.is_empty());
        assert_eq!(pt.dpb_output_delay, 0x22);
        assert_eq!(pt.display_elemental_periods_minus1, 0x33);
        assert!(!pt.cpb_alt_timing_info_present_flag);
        assert!(pt.dpb_output_du_delay.is_none());
        assert!(pt.num_decoding_units_minus1.is_none());
        assert!(pt.delay_for_concatenation_ensured_flag.is_none());
    }

    /// Two-sublayer message exercising the per-sublayer loop with the
    /// explicit `pt_cpb_removal_delay_minus1[i]` branch (no deltas).
    #[test]
    fn two_sublayers_explicit_delay() {
        let mut ctx = ctx_minimal();
        ctx.bp_max_sublayers_minus1 = 1;
        let mut w = BitWriter::new();
        w.put(0x11, 8); // pt_cpb_removal_delay_minus1[top=1]
                        // i = 0 (< 1): pt_sublayer_delays_present_flag[0]
        w.put(1, 1);
        // deltas not present -> no enabled flag; delta_enabled inferred 0
        // -> explicit pt_cpb_removal_delay_minus1[0]
        w.put(0x44, 8);
        w.put(0x22, 8); // pt_dpb_output_delay
        w.put(0x33, 8); // display_elemental_periods_minus1
        let payload = w.finish_payload();

        let pt = parse_picture_timing(&payload, &ctx).unwrap();
        assert_eq!(pt.sublayer_delays.len(), 1);
        let s = &pt.sublayer_delays[0];
        assert_eq!(s.sublayer_idx, 0);
        assert!(s.sublayer_delays_present_flag);
        assert!(!s.cpb_removal_delay_delta_enabled_flag);
        assert_eq!(s.cpb_removal_delay_minus1, Some(0x44));
        assert!(s.cpb_removal_delay_delta_idx.is_none());
        assert_eq!(pt.dpb_output_delay, 0x22);
    }

    /// Delta path: deltas present + `bp_num_cpb_removal_delay_deltas
    /// _minus1 > 0`, so the enabled flag is read and the idx is read
    /// with `Ceil(Log2(N))` bits.
    #[test]
    fn two_sublayers_delta_idx() {
        let mut ctx = ctx_minimal();
        ctx.bp_max_sublayers_minus1 = 1;
        ctx.bp_cpb_removal_delay_deltas_present_flag = true;
        ctx.bp_num_cpb_removal_delay_deltas_minus1 = 3; // idx len = ceil_log2(4) = 2
        let mut w = BitWriter::new();
        w.put(0x11, 8); // top delay
        w.put(1, 1); // sublayer_delays_present_flag[0]
        w.put(1, 1); // cpb_removal_delay_delta_enabled_flag[0]
        w.put(2, 2); // cpb_removal_delay_delta_idx[0] (2 bits)
        w.put(0x22, 8); // dpb_output_delay
        w.put(0x33, 8); // display periods
        let payload = w.finish_payload();

        let pt = parse_picture_timing(&payload, &ctx).unwrap();
        let s = &pt.sublayer_delays[0];
        assert!(s.cpb_removal_delay_delta_enabled_flag);
        assert_eq!(s.cpb_removal_delay_delta_idx, Some(2));
        assert!(s.cpb_removal_delay_minus1.is_none());
    }

    /// Alt-CPB NAL block: `bp_alt_cpb_params_present_flag` + NAL HRD
    /// with a single sublayer and one CPB index.
    #[test]
    fn alt_cpb_nal_block() {
        let mut ctx = ctx_minimal();
        ctx.bp_alt_cpb_params_present_flag = true;
        ctx.bp_nal_hrd_params_present_flag = true;
        ctx.bp_sublayer_initial_cpb_removal_delay_present_flag = true; // lo = 0
        ctx.bp_cpb_cnt_minus1 = 0; // one CPB index.
        let mut w = BitWriter::new();
        w.put(0x11, 8); // top delay
        w.put(0x22, 8); // dpb_output_delay
        w.put(1, 1); // pt_cpb_alt_timing_info_present_flag
                     // NAL HRD, i = 0..=0, j = 0..=0:
        w.put(0xA1, 8); // nal alt initial removal delay delta[0][0]
        w.put(0xA2, 8); // nal alt initial removal offset delta[0][0]
        w.put(0xA3, 8); // nal cpb delay offset[0]
        w.put(0xA4, 8); // nal dpb delay offset[0]
        w.put(0x33, 8); // display periods
        let payload = w.finish_payload();

        let pt = parse_picture_timing(&payload, &ctx).unwrap();
        assert!(pt.cpb_alt_timing_info_present_flag);
        assert_eq!(pt.nal_alt_cpb.len(), 1);
        let sub = &pt.nal_alt_cpb[0];
        assert_eq!(sub.entries.len(), 1);
        assert_eq!(sub.entries[0].initial_removal_delay_delta, 0xA1);
        assert_eq!(sub.entries[0].initial_removal_offset_delta, 0xA2);
        assert_eq!(sub.cpb_delay_offset, 0xA3);
        assert_eq!(sub.dpb_delay_offset, 0xA4);
        assert!(pt.vcl_alt_cpb.is_empty());
    }

    /// DU CPB + DPB blocks with the `bp_du_*_in_pic_timing_sei_flag`
    /// gating, common-delay path, two decoding units.
    #[test]
    fn du_blocks_common_delay() {
        let mut ctx = ctx_minimal();
        ctx.bp_du_hrd_params_present_flag = true;
        ctx.bp_du_dpb_params_in_pic_timing_sei_flag = true;
        ctx.bp_du_cpb_params_in_pic_timing_sei_flag = true;
        let mut w = BitWriter::new();
        w.put(0x11, 8); // top delay
        w.put(0x22, 8); // dpb_output_delay
                        // DU DPB block:
        w.put(0x55, 8); // pt_dpb_output_du_delay
                        // DU CPB block: num_decoding_units_minus1 = 1 (ue)
        w.ue(1);
        w.put(1, 1); // pt_du_common_cpb_removal_delay_flag
                     // common increment loop: i = 0..=0, top sublayer present -> read 1
        w.put(0x66, 8); // pt_du_common_cpb_removal_delay_increment_minus1[0]
                        // for i in 0..=1: pt_num_nalus_in_du_minus1[i]
        w.ue(3); // du0
        w.ue(4); // du1
        w.put(0x33, 8); // display periods
        let payload = w.finish_payload();

        let pt = parse_picture_timing(&payload, &ctx).unwrap();
        assert_eq!(pt.dpb_output_du_delay, Some(0x55));
        assert_eq!(pt.num_decoding_units_minus1, Some(1));
        assert!(pt.du_common_cpb_removal_delay_flag);
        assert_eq!(pt.du_common_cpb_removal_delay_increment_minus1, vec![0x66]);
        assert_eq!(pt.decoding_units.len(), 2);
        assert_eq!(pt.decoding_units[0].num_nalus_in_du_minus1, 3);
        assert_eq!(pt.decoding_units[1].num_nalus_in_du_minus1, 4);
        assert!(pt.decoding_units[0]
            .cpb_removal_delay_increment_minus1
            .is_empty());
    }

    /// Concatenation flag tail.
    #[test]
    fn concatenation_flag() {
        let mut ctx = ctx_minimal();
        ctx.bp_additional_concatenation_info_present_flag = true;
        let mut w = BitWriter::new();
        w.put(0x11, 8); // top delay
        w.put(0x22, 8); // dpb_output_delay
        w.put(1, 1); // pt_delay_for_concatenation_ensured_flag
        w.put(0x33, 8); // display periods
        let payload = w.finish_payload();

        let pt = parse_picture_timing(&payload, &ctx).unwrap();
        assert_eq!(pt.delay_for_concatenation_ensured_flag, Some(true));
    }

    /// Trailing-bytes / mis-alignment rejection: an extra payload byte
    /// beyond the structure must fail the §D.2.1 framing check.
    #[test]
    fn extra_trailing_byte_rejected() {
        let ctx = ctx_minimal();
        let mut w = BitWriter::new();
        w.put(0x11, 8);
        w.put(0x22, 8);
        w.put(0x33, 8);
        let mut payload = w.finish_payload();
        payload.push(0x00); // stray byte.
        assert!(parse_picture_timing(&payload, &ctx).is_err());
    }

    /// Truncated body rejection: not enough bytes for the structure.
    #[test]
    fn truncated_body_rejected() {
        let ctx = ctx_minimal();
        let mut w = BitWriter::new();
        w.put(0x11, 8); // only the top delay; dpb_output_delay missing.
        let payload = w.finish_payload();
        assert!(parse_picture_timing(&payload, &ctx).is_err());
    }
}
