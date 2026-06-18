//! VVC buffering period (BP) SEI message parser (§D.3.1 / §D.3.2).
//!
//! The buffering period SEI message (`payloadType == 0`) provides the
//! initial CPB removal delay / offset information used to initialise the
//! HRD at the position of the associated AU. Like the SEI manifest
//! (`payloadType == 200`), the SEI prefix indication
//! (`payloadType == 201`), the subpicture level information
//! (`payloadType == 203`), the DU information (`payloadType == 130`), and
//! the constrained RASL encoding indication (`payloadType == 207`), it is
//! one of the `sei_payload()` bodies specified directly in this
//! Specification (most other payload types defer to Rec. ITU-T H.274 |
//! ISO/IEC 23002-7).
//!
//! The BP message is also the source of the §D.3.2 derivations that gate
//! and size the DU information ([`crate::decoding_unit_info`]) and picture
//! timing SEI messages: every `u(v)`-length syntax element in those
//! messages takes its bit length from a `bp_*_length_minus1 + 1` field
//! parsed here, and the per-sublayer / DU-presence loops are gated on
//! flags parsed here. [`BufferingPeriod::dui_context`] surfaces exactly
//! the [`crate::decoding_unit_info::DuiContext`] needed to parse a
//! companion DUI message.
//!
//! Syntax (§D.3.1):
//!
//! ```text
//!   buffering_period( payloadSize ) {                                  Descriptor
//!       bp_nal_hrd_params_present_flag                                    u(1)
//!       bp_vcl_hrd_params_present_flag                                    u(1)
//!       bp_cpb_initial_removal_delay_length_minus1                        u(5)
//!       bp_cpb_removal_delay_length_minus1                                u(5)
//!       bp_dpb_output_delay_length_minus1                                 u(5)
//!       bp_du_hrd_params_present_flag                                     u(1)
//!       if( bp_du_hrd_params_present_flag ) {
//!           bp_du_cpb_removal_delay_increment_length_minus1               u(5)
//!           bp_dpb_output_delay_du_length_minus1                          u(5)
//!           bp_du_cpb_params_in_pic_timing_sei_flag                       u(1)
//!           bp_du_dpb_params_in_pic_timing_sei_flag                       u(1)
//!       }
//!       bp_concatenation_flag                                             u(1)
//!       bp_additional_concatenation_info_present_flag                     u(1)
//!       if( bp_additional_concatenation_info_present_flag )
//!           bp_max_initial_removal_delay_for_concatenation                u(v)
//!       bp_cpb_removal_delay_delta_minus1                                 u(v)
//!       bp_max_sublayers_minus1                                           u(3)
//!       if( bp_max_sublayers_minus1 > 0 )
//!           bp_cpb_removal_delay_deltas_present_flag                      u(1)
//!       if( bp_cpb_removal_delay_deltas_present_flag ) {
//!           bp_num_cpb_removal_delay_deltas_minus1                        ue(v)
//!           for( i = 0; i <= bp_num_cpb_removal_delay_deltas_minus1; i++ )
//!               bp_cpb_removal_delay_delta_val[ i ]                       u(v)
//!       }
//!       bp_cpb_cnt_minus1                                                 ue(v)
//!       if( bp_max_sublayers_minus1 > 0 )
//!           bp_sublayer_initial_cpb_removal_delay_present_flag            u(1)
//!       for( i = ( bp_sublayer_initial_cpb_removal_delay_present_flag ?
//!               0 : bp_max_sublayers_minus1 );
//!            i <= bp_max_sublayers_minus1; i++ ) {
//!           if( bp_nal_hrd_params_present_flag )
//!               for( j = 0; j < bp_cpb_cnt_minus1 + 1; j++ ) {
//!                   bp_nal_initial_cpb_removal_delay[ i ][ j ]            u(v)
//!                   bp_nal_initial_cpb_removal_offset[ i ][ j ]           u(v)
//!                   if( bp_du_hrd_params_present_flag ) {
//!                       bp_nal_initial_alt_cpb_removal_delay[ i ][ j ]    u(v)
//!                       bp_nal_initial_alt_cpb_removal_offset[ i ][ j ]   u(v)
//!                   }
//!               }
//!           if( bp_vcl_hrd_params_present_flag )
//!               for( j = 0; j < bp_cpb_cnt_minus1 + 1; j++ ) {
//!                   bp_vcl_initial_cpb_removal_delay[ i ][ j ]            u(v)
//!                   bp_vcl_initial_cpb_removal_offset[ i ][ j ]           u(v)
//!                   if( bp_du_hrd_params_present_flag ) {
//!                       bp_vcl_initial_alt_cpb_removal_delay[ i ][ j ]    u(v)
//!                       bp_vcl_initial_alt_cpb_removal_offset[ i ][ j ]   u(v)
//!                   }
//!               }
//!       }
//!       if( bp_max_sublayers_minus1 > 0 )
//!           bp_sublayer_dpb_output_offsets_present_flag                   u(1)
//!       if( bp_sublayer_dpb_output_offsets_present_flag )
//!           for( i = 0; i < bp_max_sublayers_minus1; i++ )
//!               bp_dpb_output_tid_offset[ i ]                             ue(v)
//!       bp_alt_cpb_params_present_flag                                    u(1)
//!       if( bp_alt_cpb_params_present_flag )
//!           bp_use_alt_cpb_params_flag                                    u(1)
//!   }
//! ```
//!
//! §D.3.2 `u(v)` lengths and inferences modelled here:
//!
//! * `bp_max_initial_removal_delay_for_concatenation`,
//!   `bp_nal_initial_cpb_removal_delay/offset[ i ][ j ]`,
//!   `bp_nal_initial_alt_cpb_removal_delay/offset[ i ][ j ]`, and the VCL
//!   equivalents are each `bp_cpb_initial_removal_delay_length_minus1 + 1`
//!   bits.
//! * `bp_cpb_removal_delay_delta_minus1` and
//!   `bp_cpb_removal_delay_delta_val[ i ]` are
//!   `bp_cpb_removal_delay_length_minus1 + 1` bits.
//! * `bp_num_cpb_removal_delay_deltas_minus1` shall be in 0..=15;
//!   `bp_cpb_cnt_minus1` shall be in 0..=31. Out-of-range values are
//!   rejected.
//! * The alt-CPB delay/offset pairs are present in the loop only when
//!   `bp_du_hrd_params_present_flag` is 1.
//! * When `bp_du_hrd_params_present_flag` is 0 the four DU-length /
//!   DU-flag fields are absent; `bp_du_cpb_removal_delay_increment_length_
//!   minus1` / `bp_dpb_output_delay_du_length_minus1` are inferred 23 and
//!   the two `*_in_pic_timing_sei_flag`s are inferred 0 (§D.3.2).
//! * `bp_cpb_removal_delay_deltas_present_flag`,
//!   `bp_sublayer_initial_cpb_removal_delay_present_flag`,
//!   `bp_sublayer_dpb_output_offsets_present_flag`, and
//!   `bp_use_alt_cpb_params_flag` are inferred 0 when not present.
//!
//! The whole body is a bit-packed structure with no internal
//! byte-alignment; the §D.2.1 `sei_payload()` trailing-bits tail
//! (`sei_payload_bit_equal_to_one` then zero padding to a byte boundary)
//! is validated and the consumed length is required to equal
//! `payloadSize` per §7.4.6, so a framing error surfaces as a parse error
//! rather than desynchronising the enclosing `sei_rbsp()` walk.
//!
//! Spec reference: ITU-T H.266 | ISO/IEC 23090-3 (V4, 01/2026), §D.3.1,
//! §D.3.2, §D.2.1 (`sei_payload()` dispatch + trailing bits), §7.4.6.
//!
//! No third-party VVC decoder source was consulted; the implementation is
//! spec-only and reads the payload through the crate's own [`BitReader`].

use oxideav_core::{Error, Result};

use crate::bitreader::BitReader;
use crate::decoding_unit_info::DuiContext;

/// `payloadType` value that selects the buffering period body in the §D.2
/// `sei_payload()` dispatch.
pub const BUFFERING_PERIOD_PAYLOAD_TYPE: u32 = 0;

/// Maximum value of `bp_num_cpb_removal_delay_deltas_minus1` (§D.3.2).
const MAX_NUM_CPB_REMOVAL_DELAY_DELTAS_MINUS1: u32 = 15;

/// Maximum value of `bp_cpb_cnt_minus1` (§D.3.2).
const MAX_CPB_CNT_MINUS1: u32 = 31;

/// One initial-CPB-removal-delay/offset entry pair for a single
/// `( i, j )` sublayer/CPB index of one HRD type (§D.3.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct InitialCpbRemovalEntry {
    /// `bp_{nal,vcl}_initial_cpb_removal_delay[ i ][ j ]`.
    pub initial_cpb_removal_delay: u32,
    /// `bp_{nal,vcl}_initial_cpb_removal_offset[ i ][ j ]`.
    pub initial_cpb_removal_offset: u32,
    /// `bp_{nal,vcl}_initial_alt_cpb_removal_delay[ i ][ j ]` — present
    /// (non-`None`) only when `bp_du_hrd_params_present_flag` is 1.
    pub initial_alt_cpb_removal_delay: Option<u32>,
    /// `bp_{nal,vcl}_initial_alt_cpb_removal_offset[ i ][ j ]` — present
    /// (non-`None`) only when `bp_du_hrd_params_present_flag` is 1.
    pub initial_alt_cpb_removal_offset: Option<u32>,
}

/// The per-sublayer initial CPB removal delay / offset entries for one
/// temporal sublayer `i` (§D.3.1).
///
/// `nal` is populated only when `bp_nal_hrd_params_present_flag` is 1,
/// `vcl` only when `bp_vcl_hrd_params_present_flag` is 1. Each holds
/// `bp_cpb_cnt_minus1 + 1` entries (the `j` loop).
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct SublayerInitialCpb {
    /// The sublayer index `i` this entry describes.
    pub sublayer_idx: u8,
    /// NAL HRD `( i, j )` entries (empty when
    /// `bp_nal_hrd_params_present_flag` is 0).
    pub nal: Vec<InitialCpbRemovalEntry>,
    /// VCL HRD `( i, j )` entries (empty when
    /// `bp_vcl_hrd_params_present_flag` is 0).
    pub vcl: Vec<InitialCpbRemovalEntry>,
}

/// A parsed buffering period SEI message (§D.3.1).
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct BufferingPeriod {
    /// `bp_nal_hrd_params_present_flag`.
    pub nal_hrd_params_present_flag: bool,
    /// `bp_vcl_hrd_params_present_flag`.
    pub vcl_hrd_params_present_flag: bool,
    /// `bp_cpb_initial_removal_delay_length_minus1`.
    pub cpb_initial_removal_delay_length_minus1: u8,
    /// `bp_cpb_removal_delay_length_minus1`.
    pub cpb_removal_delay_length_minus1: u8,
    /// `bp_dpb_output_delay_length_minus1`.
    pub dpb_output_delay_length_minus1: u8,
    /// `bp_du_hrd_params_present_flag` (inferred 0 when absent).
    pub du_hrd_params_present_flag: bool,
    /// `bp_du_cpb_removal_delay_increment_length_minus1` (inferred 23 when
    /// `bp_du_hrd_params_present_flag` is 0).
    pub du_cpb_removal_delay_increment_length_minus1: u8,
    /// `bp_dpb_output_delay_du_length_minus1` (inferred 23 when
    /// `bp_du_hrd_params_present_flag` is 0).
    pub dpb_output_delay_du_length_minus1: u8,
    /// `bp_du_cpb_params_in_pic_timing_sei_flag` (inferred 0 when absent).
    pub du_cpb_params_in_pic_timing_sei_flag: bool,
    /// `bp_du_dpb_params_in_pic_timing_sei_flag` (inferred 0 when absent).
    pub du_dpb_params_in_pic_timing_sei_flag: bool,
    /// `bp_concatenation_flag`.
    pub concatenation_flag: bool,
    /// `bp_additional_concatenation_info_present_flag`.
    pub additional_concatenation_info_present_flag: bool,
    /// `bp_max_initial_removal_delay_for_concatenation` — present
    /// (non-`None`) only when
    /// `additional_concatenation_info_present_flag` is `true`.
    pub max_initial_removal_delay_for_concatenation: Option<u32>,
    /// `bp_cpb_removal_delay_delta_minus1`.
    pub cpb_removal_delay_delta_minus1: u32,
    /// `bp_max_sublayers_minus1`.
    pub max_sublayers_minus1: u8,
    /// `bp_cpb_removal_delay_deltas_present_flag` (inferred 0 when absent).
    pub cpb_removal_delay_deltas_present_flag: bool,
    /// `bp_cpb_removal_delay_delta_val[ i ]` — `bp_num_cpb_removal_delay_
    /// deltas_minus1 + 1` entries when
    /// `cpb_removal_delay_deltas_present_flag` is `true`, else empty.
    pub cpb_removal_delay_delta_vals: Vec<u32>,
    /// `bp_cpb_cnt_minus1`.
    pub cpb_cnt_minus1: u32,
    /// `bp_sublayer_initial_cpb_removal_delay_present_flag` (inferred 0
    /// when absent).
    pub sublayer_initial_cpb_removal_delay_present_flag: bool,
    /// The per-sublayer initial CPB removal delay/offset entries, in the
    /// stream order the `i` loop visits them.
    pub sublayer_initial_cpb: Vec<SublayerInitialCpb>,
    /// `bp_sublayer_dpb_output_offsets_present_flag` (inferred 0 when
    /// absent).
    pub sublayer_dpb_output_offsets_present_flag: bool,
    /// `bp_dpb_output_tid_offset[ i ]` — `bp_max_sublayers_minus1` entries
    /// (`i = 0 .. bp_max_sublayers_minus1 - 1`) when
    /// `sublayer_dpb_output_offsets_present_flag` is `true`, else empty.
    pub dpb_output_tid_offsets: Vec<u32>,
    /// `bp_alt_cpb_params_present_flag`.
    pub alt_cpb_params_present_flag: bool,
    /// `bp_use_alt_cpb_params_flag` (inferred 0 when absent).
    pub use_alt_cpb_params_flag: bool,
}

impl BufferingPeriod {
    /// Build the [`DuiContext`] a companion DU information SEI message
    /// (`payloadType == 130`) needs, given the `TemporalId` of the SEI NAL
    /// unit carrying that DUI message (§D.3.2 / §D.5.2).
    ///
    /// All five fields are §D.3.2 derivations carried by this BP message;
    /// only the DUI's own NAL unit `TemporalId` is supplied externally.
    pub fn dui_context(&self, temporal_id: u8) -> DuiContext {
        DuiContext {
            temporal_id,
            bp_max_sublayers_minus1: self.max_sublayers_minus1,
            bp_du_cpb_params_in_pic_timing_sei_flag: self.du_cpb_params_in_pic_timing_sei_flag,
            bp_du_dpb_params_in_pic_timing_sei_flag: self.du_dpb_params_in_pic_timing_sei_flag,
            bp_du_cpb_removal_delay_increment_length_minus1: self
                .du_cpb_removal_delay_increment_length_minus1,
            bp_dpb_output_delay_du_length_minus1: self.dpb_output_delay_du_length_minus1,
        }
    }
}

/// Read the `sei_payload()` trailing bits (§D.2.1) and require the reader
/// to land exactly at `payload.len()`.
///
/// After the structured body, `sei_payload()` carries (when the body did
/// not already end on a byte boundary) a single
/// `sei_payload_bit_equal_to_one` (`f(1)`) followed by
/// `sei_payload_bit_equal_to_zero` bits to the byte boundary. §7.4.6 then
/// requires the consumed length to equal `payloadSize`.
fn finish_sei_payload(reader: &mut BitReader, what: &str) -> Result<()> {
    if reader.is_byte_aligned() {
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

/// Read the `bp_cpb_cnt_minus1 + 1` `( i, j )` entries of one HRD type for
/// sublayer `i`, each a `u(v)` delay/offset pair plus the optional alt
/// pair (present only when `du_hrd` is `true`), all
/// `init_len`-bit `u(v)`.
fn read_initial_cpb_entries(
    reader: &mut BitReader,
    cpb_cnt: u32,
    init_len: u32,
    du_hrd: bool,
) -> Result<Vec<InitialCpbRemovalEntry>> {
    let mut entries = Vec::with_capacity(cpb_cnt as usize);
    for _ in 0..cpb_cnt {
        let initial_cpb_removal_delay = reader.u(init_len)?;
        let initial_cpb_removal_offset = reader.u(init_len)?;
        let (initial_alt_cpb_removal_delay, initial_alt_cpb_removal_offset) = if du_hrd {
            (Some(reader.u(init_len)?), Some(reader.u(init_len)?))
        } else {
            (None, None)
        };
        entries.push(InitialCpbRemovalEntry {
            initial_cpb_removal_delay,
            initial_cpb_removal_offset,
            initial_alt_cpb_removal_delay,
            initial_alt_cpb_removal_offset,
        });
    }
    Ok(entries)
}

/// Parse a `buffering_period( payloadSize )` body (§D.3.1) from the raw
/// SEI payload bytes carried by a `payloadType == 0` `sei_message()`.
///
/// `payload` is the `sei_payload()` argument region — the `payloadSize`
/// bytes that follow the `sei_message()` header, with emulation-prevention
/// bytes already removed.
///
/// Errors:
/// * a `payload` too short for the structure is rejected (propagated from
///   the bit reader);
/// * `bp_num_cpb_removal_delay_deltas_minus1 > 15` or
///   `bp_cpb_cnt_minus1 > 31` (out of the §D.3.2 ranges) is rejected;
/// * a malformed §D.2.1 trailing-bits region (missing
///   `sei_payload_bit_equal_to_one`, a non-`0` zero-padding bit, or a
///   consumed length that does not equal `payloadSize`) is rejected.
pub fn parse_buffering_period(payload: &[u8]) -> Result<BufferingPeriod> {
    let mut reader = BitReader::new(payload);

    let nal_hrd_params_present_flag = reader.u1()? != 0;
    let vcl_hrd_params_present_flag = reader.u1()? != 0;
    let cpb_initial_removal_delay_length_minus1 = reader.u(5)? as u8;
    let cpb_removal_delay_length_minus1 = reader.u(5)? as u8;
    let dpb_output_delay_length_minus1 = reader.u(5)? as u8;

    let du_hrd_params_present_flag = reader.u1()? != 0;
    // Inferences (§D.3.2): when bp_du_hrd_params_present_flag is 0 the
    // two DU-length fields are inferred 23 and the two
    // *_in_pic_timing_sei_flag are inferred 0.
    let mut du_cpb_removal_delay_increment_length_minus1 = 23u8;
    let mut dpb_output_delay_du_length_minus1 = 23u8;
    let mut du_cpb_params_in_pic_timing_sei_flag = false;
    let mut du_dpb_params_in_pic_timing_sei_flag = false;
    if du_hrd_params_present_flag {
        du_cpb_removal_delay_increment_length_minus1 = reader.u(5)? as u8;
        dpb_output_delay_du_length_minus1 = reader.u(5)? as u8;
        du_cpb_params_in_pic_timing_sei_flag = reader.u1()? != 0;
        du_dpb_params_in_pic_timing_sei_flag = reader.u1()? != 0;
    }

    let concatenation_flag = reader.u1()? != 0;
    let additional_concatenation_info_present_flag = reader.u1()? != 0;

    // §D.3.2: the length of bp_max_initial_removal_delay_for_concatenation
    // and the initial CPB removal delay/offset fields is
    // bp_cpb_initial_removal_delay_length_minus1 + 1 bits.
    let init_len = u32::from(cpb_initial_removal_delay_length_minus1) + 1;
    // §D.3.2: bp_cpb_removal_delay_delta_minus1 and
    // bp_cpb_removal_delay_delta_val[i] are
    // bp_cpb_removal_delay_length_minus1 + 1 bits.
    let removal_delay_len = u32::from(cpb_removal_delay_length_minus1) + 1;

    let max_initial_removal_delay_for_concatenation = if additional_concatenation_info_present_flag
    {
        Some(reader.u(init_len)?)
    } else {
        None
    };

    let cpb_removal_delay_delta_minus1 = reader.u(removal_delay_len)?;

    let max_sublayers_minus1 = reader.u(3)? as u8;

    // bp_cpb_removal_delay_deltas_present_flag is only read when
    // bp_max_sublayers_minus1 > 0 (inferred 0 otherwise).
    let cpb_removal_delay_deltas_present_flag = if max_sublayers_minus1 > 0 {
        reader.u1()? != 0
    } else {
        false
    };

    let mut cpb_removal_delay_delta_vals = Vec::new();
    if cpb_removal_delay_deltas_present_flag {
        let num_minus1 = reader.ue()?;
        if num_minus1 > MAX_NUM_CPB_REMOVAL_DELAY_DELTAS_MINUS1 {
            return Err(Error::invalid(
                "h266 buffering_period: bp_num_cpb_removal_delay_deltas_minus1 > 15 (§D.3.2)",
            ));
        }
        for _ in 0..=num_minus1 {
            cpb_removal_delay_delta_vals.push(reader.u(removal_delay_len)?);
        }
    }

    let cpb_cnt_minus1 = reader.ue()?;
    if cpb_cnt_minus1 > MAX_CPB_CNT_MINUS1 {
        return Err(Error::invalid(
            "h266 buffering_period: bp_cpb_cnt_minus1 > 31 (§D.3.2)",
        ));
    }
    let cpb_cnt = cpb_cnt_minus1 + 1;

    let sublayer_initial_cpb_removal_delay_present_flag = if max_sublayers_minus1 > 0 {
        reader.u1()? != 0
    } else {
        false
    };

    // for( i = ( sublayer_present ? 0 : bp_max_sublayers_minus1 );
    //      i <= bp_max_sublayers_minus1; i++ )
    let i_start = if sublayer_initial_cpb_removal_delay_present_flag {
        0u8
    } else {
        max_sublayers_minus1
    };
    let mut sublayer_initial_cpb = Vec::new();
    for i in i_start..=max_sublayers_minus1 {
        let nal = if nal_hrd_params_present_flag {
            read_initial_cpb_entries(&mut reader, cpb_cnt, init_len, du_hrd_params_present_flag)?
        } else {
            Vec::new()
        };
        let vcl = if vcl_hrd_params_present_flag {
            read_initial_cpb_entries(&mut reader, cpb_cnt, init_len, du_hrd_params_present_flag)?
        } else {
            Vec::new()
        };
        sublayer_initial_cpb.push(SublayerInitialCpb {
            sublayer_idx: i,
            nal,
            vcl,
        });
    }

    let sublayer_dpb_output_offsets_present_flag = if max_sublayers_minus1 > 0 {
        reader.u1()? != 0
    } else {
        false
    };

    let mut dpb_output_tid_offsets = Vec::new();
    if sublayer_dpb_output_offsets_present_flag {
        // for( i = 0; i < bp_max_sublayers_minus1; i++ )
        for _ in 0..max_sublayers_minus1 {
            dpb_output_tid_offsets.push(reader.ue()?);
        }
    }

    let alt_cpb_params_present_flag = reader.u1()? != 0;
    let use_alt_cpb_params_flag = if alt_cpb_params_present_flag {
        reader.u1()? != 0
    } else {
        false
    };

    finish_sei_payload(
        &mut reader,
        "h266 buffering_period: malformed sei_payload trailing bits / length (§D.2.1, §7.4.6)",
    )?;

    Ok(BufferingPeriod {
        nal_hrd_params_present_flag,
        vcl_hrd_params_present_flag,
        cpb_initial_removal_delay_length_minus1,
        cpb_removal_delay_length_minus1,
        dpb_output_delay_length_minus1,
        du_hrd_params_present_flag,
        du_cpb_removal_delay_increment_length_minus1,
        dpb_output_delay_du_length_minus1,
        du_cpb_params_in_pic_timing_sei_flag,
        du_dpb_params_in_pic_timing_sei_flag,
        concatenation_flag,
        additional_concatenation_info_present_flag,
        max_initial_removal_delay_for_concatenation,
        cpb_removal_delay_delta_minus1,
        max_sublayers_minus1,
        cpb_removal_delay_deltas_present_flag,
        cpb_removal_delay_delta_vals,
        cpb_cnt_minus1,
        sublayer_initial_cpb_removal_delay_present_flag,
        sublayer_initial_cpb,
        sublayer_dpb_output_offsets_present_flag,
        dpb_output_tid_offsets,
        alt_cpb_params_present_flag,
        use_alt_cpb_params_flag,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A small bit packer for building BP test payloads.
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

    /// Write the fixed-prefix common to the test payloads:
    /// nal/vcl flags, the three length fields, then the du_hrd flag.
    #[allow(clippy::too_many_arguments)]
    fn write_prefix(
        w: &mut BitWriter,
        nal: u32,
        vcl: u32,
        init_len_m1: u32,
        rem_len_m1: u32,
        dpb_len_m1: u32,
        du_hrd: u32,
    ) {
        w.put(nal, 1);
        w.put(vcl, 1);
        w.put(init_len_m1, 5);
        w.put(rem_len_m1, 5);
        w.put(dpb_len_m1, 5);
        w.put(du_hrd, 1);
    }

    /// Minimal single-sublayer NAL-only message: no DU HRD, no
    /// concatenation extras, no deltas, one CPB entry.
    #[test]
    fn minimal_nal_only() {
        let mut w = BitWriter::new();
        // nal=1, vcl=0, init_len_m1=7 (8-bit), rem_len_m1=7 (8-bit),
        // dpb_len_m1=7, du_hrd=0.
        write_prefix(&mut w, 1, 0, 7, 7, 7, 0);
        w.put(0, 1); // concatenation_flag
        w.put(0, 1); // additional_concatenation_info_present_flag
        w.put(0x12, 8); // cpb_removal_delay_delta_minus1 (8-bit)
        w.put(0, 3); // max_sublayers_minus1 = 0 -> no deltas_present flag
        w.ue(0); // cpb_cnt_minus1 = 0 -> cpb_cnt = 1
                 // max_sublayers_minus1 == 0: no sublayer_present flag,
                 // i_start = 0, single i = 0; NAL entries only.
        w.put(0xAB, 8); // nal_initial_cpb_removal_delay[0][0]
        w.put(0xCD, 8); // nal_initial_cpb_removal_offset[0][0]
        w.put(0, 1); // alt_cpb_params_present_flag
        let payload = w.finish_payload();

        let bp = parse_buffering_period(&payload).unwrap();
        assert!(bp.nal_hrd_params_present_flag);
        assert!(!bp.vcl_hrd_params_present_flag);
        assert_eq!(bp.cpb_initial_removal_delay_length_minus1, 7);
        assert_eq!(bp.cpb_removal_delay_length_minus1, 7);
        assert_eq!(bp.dpb_output_delay_length_minus1, 7);
        assert!(!bp.du_hrd_params_present_flag);
        // Inferred when du_hrd absent.
        assert_eq!(bp.du_cpb_removal_delay_increment_length_minus1, 23);
        assert_eq!(bp.dpb_output_delay_du_length_minus1, 23);
        assert!(!bp.du_cpb_params_in_pic_timing_sei_flag);
        assert!(!bp.du_dpb_params_in_pic_timing_sei_flag);
        assert_eq!(bp.cpb_removal_delay_delta_minus1, 0x12);
        assert_eq!(bp.max_sublayers_minus1, 0);
        assert!(!bp.cpb_removal_delay_deltas_present_flag);
        assert_eq!(bp.cpb_cnt_minus1, 0);
        assert_eq!(bp.sublayer_initial_cpb.len(), 1);
        let s = &bp.sublayer_initial_cpb[0];
        assert_eq!(s.sublayer_idx, 0);
        assert_eq!(s.nal.len(), 1);
        assert!(s.vcl.is_empty());
        assert_eq!(s.nal[0].initial_cpb_removal_delay, 0xAB);
        assert_eq!(s.nal[0].initial_cpb_removal_offset, 0xCD);
        assert_eq!(s.nal[0].initial_alt_cpb_removal_delay, None);
        assert!(!bp.alt_cpb_params_present_flag);
    }

    /// DU HRD present: the four DU fields are read, the alt delay/offset
    /// pairs appear in the per-(i,j) loop, and `dui_context` surfaces them.
    #[test]
    fn du_hrd_present_with_alt_pairs() {
        let mut w = BitWriter::new();
        // nal=1, vcl=0, init_len_m1=7, rem_len_m1=7, dpb_len_m1=7,
        // du_hrd=1.
        write_prefix(&mut w, 1, 0, 7, 7, 7, 1);
        w.put(9, 5); // du_cpb_removal_delay_increment_length_minus1
        w.put(11, 5); // dpb_output_delay_du_length_minus1
        w.put(1, 1); // du_cpb_params_in_pic_timing_sei_flag
        w.put(0, 1); // du_dpb_params_in_pic_timing_sei_flag
        w.put(0, 1); // concatenation_flag
        w.put(0, 1); // additional_concatenation_info_present_flag
        w.put(0x05, 8); // cpb_removal_delay_delta_minus1
        w.put(0, 3); // max_sublayers_minus1 = 0
        w.ue(0); // cpb_cnt_minus1 = 0
                 // single i = 0, NAL: delay, offset, alt_delay, alt_offset.
        w.put(0x11, 8);
        w.put(0x22, 8);
        w.put(0x33, 8); // alt delay
        w.put(0x44, 8); // alt offset
        w.put(0, 1); // alt_cpb_params_present_flag
        let payload = w.finish_payload();

        let bp = parse_buffering_period(&payload).unwrap();
        assert!(bp.du_hrd_params_present_flag);
        assert_eq!(bp.du_cpb_removal_delay_increment_length_minus1, 9);
        assert_eq!(bp.dpb_output_delay_du_length_minus1, 11);
        assert!(bp.du_cpb_params_in_pic_timing_sei_flag);
        assert!(!bp.du_dpb_params_in_pic_timing_sei_flag);
        let e = &bp.sublayer_initial_cpb[0].nal[0];
        assert_eq!(e.initial_cpb_removal_delay, 0x11);
        assert_eq!(e.initial_cpb_removal_offset, 0x22);
        assert_eq!(e.initial_alt_cpb_removal_delay, Some(0x33));
        assert_eq!(e.initial_alt_cpb_removal_offset, Some(0x44));

        // dui_context surfaces the §D.3.2 DU-derived values.
        let ctx = bp.dui_context(2);
        assert_eq!(ctx.temporal_id, 2);
        assert_eq!(ctx.bp_max_sublayers_minus1, 0);
        assert!(ctx.bp_du_cpb_params_in_pic_timing_sei_flag);
        assert!(!ctx.bp_du_dpb_params_in_pic_timing_sei_flag);
        assert_eq!(ctx.bp_du_cpb_removal_delay_increment_length_minus1, 9);
        assert_eq!(ctx.bp_dpb_output_delay_du_length_minus1, 11);
    }

    /// `additional_concatenation_info_present_flag == 1` reads the
    /// `init_len`-bit max-initial-removal-delay field.
    #[test]
    fn additional_concatenation_info() {
        let mut w = BitWriter::new();
        write_prefix(&mut w, 1, 0, 11, 7, 7, 0); // init_len_m1=11 -> 12-bit
        w.put(1, 1); // concatenation_flag
        w.put(1, 1); // additional_concatenation_info_present_flag
        w.put(0xABC, 12); // bp_max_initial_removal_delay_for_concatenation
        w.put(0x07, 8); // cpb_removal_delay_delta_minus1
        w.put(0, 3); // max_sublayers_minus1 = 0
        w.ue(0); // cpb_cnt_minus1 = 0
        w.put(0x55, 12); // nal_initial_cpb_removal_delay (12-bit)
        w.put(0x66, 12); // nal_initial_cpb_removal_offset (12-bit)
        w.put(0, 1); // alt_cpb_params_present_flag
        let payload = w.finish_payload();

        let bp = parse_buffering_period(&payload).unwrap();
        assert!(bp.concatenation_flag);
        assert!(bp.additional_concatenation_info_present_flag);
        assert_eq!(bp.max_initial_removal_delay_for_concatenation, Some(0xABC));
        assert_eq!(
            bp.sublayer_initial_cpb[0].nal[0].initial_cpb_removal_delay,
            0x55
        );
    }

    /// Multi-sublayer with deltas and sublayer-present flag set: the `i`
    /// loop covers 0..=max, NAL+VCL both present, plus the
    /// dpb-output-tid-offset list.
    #[test]
    fn multi_sublayer_nal_and_vcl_with_deltas() {
        let mut w = BitWriter::new();
        // nal=1, vcl=1, init_len_m1=7, rem_len_m1=3 (4-bit deltas), du=0.
        write_prefix(&mut w, 1, 1, 7, 3, 7, 0);
        w.put(0, 1); // concatenation_flag
        w.put(0, 1); // additional_concatenation_info_present_flag
        w.put(0x9, 4); // cpb_removal_delay_delta_minus1 (4-bit)
        w.put(2, 3); // max_sublayers_minus1 = 2
        w.put(1, 1); // cpb_removal_delay_deltas_present_flag
        w.ue(1); // num_cpb_removal_delay_deltas_minus1 = 1 -> 2 vals
        w.put(0x3, 4); // delta_val[0]
        w.put(0x4, 4); // delta_val[1]
        w.ue(1); // cpb_cnt_minus1 = 1 -> cpb_cnt = 2
        w.put(1, 1); // sublayer_initial_cpb_removal_delay_present_flag = 1

        // i = 0..=2, each with NAL then VCL, 2 entries each, no alt pairs.
        let mut tag = 0u32;
        for _i in 0..=2 {
            // NAL: 2 (delay,offset) pairs.
            for _j in 0..2 {
                w.put(tag, 8);
                tag += 1;
                w.put(tag, 8);
                tag += 1;
            }
            // VCL: 2 (delay,offset) pairs.
            for _j in 0..2 {
                w.put(tag, 8);
                tag += 1;
                w.put(tag, 8);
                tag += 1;
            }
        }

        w.put(1, 1); // sublayer_dpb_output_offsets_present_flag = 1
        w.ue(5); // dpb_output_tid_offset[0]
        w.ue(6); // dpb_output_tid_offset[1]  (i < max_sublayers_minus1=2)
        w.put(0, 1); // alt_cpb_params_present_flag
        let payload = w.finish_payload();

        let bp = parse_buffering_period(&payload).unwrap();
        assert!(bp.nal_hrd_params_present_flag);
        assert!(bp.vcl_hrd_params_present_flag);
        assert_eq!(bp.max_sublayers_minus1, 2);
        assert!(bp.cpb_removal_delay_deltas_present_flag);
        assert_eq!(bp.cpb_removal_delay_delta_vals, vec![0x3, 0x4]);
        assert_eq!(bp.cpb_cnt_minus1, 1);
        assert!(bp.sublayer_initial_cpb_removal_delay_present_flag);
        assert_eq!(bp.sublayer_initial_cpb.len(), 3);
        for (idx, s) in bp.sublayer_initial_cpb.iter().enumerate() {
            assert_eq!(s.sublayer_idx, idx as u8);
            assert_eq!(s.nal.len(), 2);
            assert_eq!(s.vcl.len(), 2);
        }
        // First NAL entry of sublayer 0 is delay=0, offset=1.
        assert_eq!(
            bp.sublayer_initial_cpb[0].nal[0].initial_cpb_removal_delay,
            0
        );
        assert_eq!(
            bp.sublayer_initial_cpb[0].nal[0].initial_cpb_removal_offset,
            1
        );
        assert!(bp.sublayer_dpb_output_offsets_present_flag);
        assert_eq!(bp.dpb_output_tid_offsets, vec![5, 6]);
    }

    /// `sublayer_initial_cpb_removal_delay_present_flag == 0` with
    /// `max_sublayers_minus1 > 0` makes the `i` loop run for the single
    /// top sublayer only (i_start == max).
    #[test]
    fn sublayer_present_flag_zero_top_only() {
        let mut w = BitWriter::new();
        write_prefix(&mut w, 1, 0, 7, 7, 7, 0);
        w.put(0, 1); // concatenation_flag
        w.put(0, 1); // additional_concatenation_info_present_flag
        w.put(0x01, 8); // cpb_removal_delay_delta_minus1
        w.put(3, 3); // max_sublayers_minus1 = 3
        w.put(0, 1); // cpb_removal_delay_deltas_present_flag = 0
        w.ue(0); // cpb_cnt_minus1 = 0
        w.put(0, 1); // sublayer_initial_cpb_removal_delay_present_flag = 0
                     // i_start = 3, single i = 3.
        w.put(0x7E, 8); // nal delay
        w.put(0x7F, 8); // nal offset
        w.put(0, 1); // sublayer_dpb_output_offsets_present_flag = 0
        w.put(0, 1); // alt_cpb_params_present_flag
        let payload = w.finish_payload();

        let bp = parse_buffering_period(&payload).unwrap();
        assert_eq!(bp.max_sublayers_minus1, 3);
        assert!(!bp.sublayer_initial_cpb_removal_delay_present_flag);
        assert_eq!(bp.sublayer_initial_cpb.len(), 1);
        assert_eq!(bp.sublayer_initial_cpb[0].sublayer_idx, 3);
        assert_eq!(
            bp.sublayer_initial_cpb[0].nal[0].initial_cpb_removal_delay,
            0x7E
        );
    }

    /// `alt_cpb_params_present_flag == 1` reads the use-alt flag.
    #[test]
    fn alt_cpb_params_use_flag() {
        let mut w = BitWriter::new();
        write_prefix(&mut w, 1, 0, 7, 7, 7, 0);
        w.put(0, 1);
        w.put(0, 1);
        w.put(0x00, 8);
        w.put(0, 3); // max_sublayers_minus1 = 0
        w.ue(0); // cpb_cnt_minus1 = 0
        w.put(0x01, 8);
        w.put(0x02, 8);
        w.put(1, 1); // alt_cpb_params_present_flag = 1
        w.put(1, 1); // use_alt_cpb_params_flag = 1
        let payload = w.finish_payload();

        let bp = parse_buffering_period(&payload).unwrap();
        assert!(bp.alt_cpb_params_present_flag);
        assert!(bp.use_alt_cpb_params_flag);
    }

    /// `bp_cpb_cnt_minus1 > 31` is rejected (§D.3.2 range).
    #[test]
    fn cpb_cnt_out_of_range_rejected() {
        let mut w = BitWriter::new();
        write_prefix(&mut w, 1, 0, 7, 7, 7, 0);
        w.put(0, 1);
        w.put(0, 1);
        w.put(0x00, 8);
        w.put(0, 3);
        w.ue(32); // cpb_cnt_minus1 = 32 -> out of range
        let payload = w.finish_payload();
        assert!(parse_buffering_period(&payload).is_err());
    }

    /// `bp_num_cpb_removal_delay_deltas_minus1 > 15` is rejected.
    #[test]
    fn num_deltas_out_of_range_rejected() {
        let mut w = BitWriter::new();
        write_prefix(&mut w, 1, 0, 7, 7, 7, 0);
        w.put(0, 1);
        w.put(0, 1);
        w.put(0x00, 8);
        w.put(1, 3); // max_sublayers_minus1 = 1 -> deltas_present read
        w.put(1, 1); // cpb_removal_delay_deltas_present_flag = 1
        w.ue(16); // num_cpb_removal_delay_deltas_minus1 = 16 -> out of range
        let payload = w.finish_payload();
        assert!(parse_buffering_period(&payload).is_err());
    }

    /// A truncated body is rejected.
    #[test]
    fn truncated_body_rejected() {
        let payload = [0b1000_0000u8];
        assert!(parse_buffering_period(&payload).is_err());
    }

    /// Extra trailing bytes beyond the §D.2.1 tail are rejected.
    #[test]
    fn trailing_bytes_rejected() {
        let mut w = BitWriter::new();
        write_prefix(&mut w, 1, 0, 7, 7, 7, 0);
        w.put(0, 1);
        w.put(0, 1);
        w.put(0x00, 8);
        w.put(0, 3);
        w.ue(0);
        w.put(0x01, 8);
        w.put(0x02, 8);
        w.put(0, 1);
        let mut payload = w.finish_payload();
        payload.push(0xAB);
        assert!(parse_buffering_period(&payload).is_err());
    }

    /// `BUFFERING_PERIOD_PAYLOAD_TYPE` matches the §D.2 dispatch value.
    #[test]
    fn payload_type_constant() {
        assert_eq!(BUFFERING_PERIOD_PAYLOAD_TYPE, 0);
    }
}
