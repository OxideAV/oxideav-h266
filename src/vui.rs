//! Video usability information — `vui_parameters()` of Rec. ITU-T
//! H.274 | ISO/IEC 23002-7 §7.2 / §7.3, carried in the H.266 SPS
//! inside `vui_payload()` (H.266 §7.3.2.21 / §7.4.3.21).
//!
//! The SPS states the payload size in bytes
//! (`sps_vui_payload_size_minus1 + 1`, at most 1024), so the parser
//! works on exactly that slice: the H.274 syntax, then — when bits
//! remain — the reserved extension data, the
//! `vui_payload_bit_equal_to_one` stop bit and zero padding. A
//! decoder ignores the extension data (§7.4.3.21), which is what
//! makes payloads written by later versions readable here.
//!
//! [`VuiParameters::color_signal`] maps the colour description onto
//! the core [`ColorSignal`]; [`VuiParameters::write_payload`] is the
//! encoder side.

use oxideav_core::{ColorSignal, Error, Result};

use crate::bitreader::BitReader;

/// §7.4.3.4: `sps_vui_payload_size_minus1` is in `0..=1023`.
pub const MAX_VUI_PAYLOAD_BYTES: usize = 1024;

/// `vui_aspect_ratio_idc` value that announces an explicit
/// `vui_sar_width : vui_sar_height` pair.
pub const ASPECT_RATIO_IDC_EXPLICIT: u8 = 255;

/// Source scan type signalled by `vui_progressive_source_flag` /
/// `vui_interlaced_source_flag` (H.274 §7.3).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum SourceScan {
    /// Both flags 0: unknown, unspecified or set by external means.
    #[default]
    Unspecified,
    /// Progressive only.
    Progressive,
    /// Interlaced only.
    Interlaced,
    /// Both flags 1: stated per picture by `ffi_source_scan_type` in a
    /// frame-field information SEI message.
    PerPicture,
}

/// The sample aspect ratio fields (`vui_aspect_ratio_*`, `vui_sar_*`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct VuiAspectRatio {
    /// `vui_aspect_ratio_constant_flag` — the ratio holds for every
    /// picture of the CLVS (no SARI SEI message overrides it).
    pub constant_flag: bool,
    /// `vui_aspect_ratio_idc` — a Rec. ITU-T H.273 `SampleAspectRatio`
    /// code point; [`ASPECT_RATIO_IDC_EXPLICIT`] selects
    /// [`Self::sar_width`] / [`Self::sar_height`].
    pub idc: u8,
    /// `vui_sar_width` (0 when `idc != 255`).
    pub sar_width: u16,
    /// `vui_sar_height` (0 when `idc != 255`).
    pub sar_height: u16,
}

impl VuiAspectRatio {
    /// An explicit `width : height` sample aspect ratio.
    pub const fn explicit(sar_width: u16, sar_height: u16) -> Self {
        Self {
            constant_flag: true,
            idc: ASPECT_RATIO_IDC_EXPLICIT,
            sar_width,
            sar_height,
        }
    }

    /// The explicit ratio, when one is signalled and specified
    /// (`idc == 255`, neither term 0). An `idc` code point other than
    /// 255 is an index into the Rec. ITU-T H.273 table and is reported
    /// as-is through [`Self::idc`].
    pub fn explicit_ratio(&self) -> Option<(u16, u16)> {
        (self.idc == ASPECT_RATIO_IDC_EXPLICIT && self.sar_width != 0 && self.sar_height != 0)
            .then_some((self.sar_width, self.sar_height))
    }
}

/// The colour description (`vui_colour_primaries`,
/// `vui_transfer_characteristics`, `vui_matrix_coeffs`,
/// `vui_full_range_flag`) — Rec. ITU-T H.273 code points.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct VuiColourDescription {
    pub colour_primaries: u8,
    pub transfer_characteristics: u8,
    pub matrix_coeffs: u8,
    pub full_range_flag: bool,
}

impl VuiColourDescription {
    /// Build from the three code points and the range flag.
    pub const fn new(
        colour_primaries: u8,
        transfer_characteristics: u8,
        matrix_coeffs: u8,
        full_range_flag: bool,
    ) -> Self {
        Self {
            colour_primaries,
            transfer_characteristics,
            matrix_coeffs,
            full_range_flag,
        }
    }

    /// The description of a core [`ColorSignal`]; `None` when the
    /// signal says nothing (every field unspecified), in which case no
    /// colour description needs to be written. An unspecified range
    /// with specified code points is written as limited range — the
    /// value `vui_full_range_flag` is inferred to when absent.
    pub fn from_color_signal(signal: &ColorSignal) -> Option<Self> {
        if *signal == ColorSignal::unspecified() {
            return None;
        }
        Some(Self {
            colour_primaries: signal.primaries.0,
            transfer_characteristics: signal.transfer.0,
            matrix_coeffs: signal.matrix.0,
            full_range_flag: signal.range.full_range_flag().unwrap_or(false),
        })
    }

    /// The same description as a core [`ColorSignal`].
    pub const fn color_signal(&self) -> ColorSignal {
        ColorSignal::from_code_points(
            self.colour_primaries,
            self.transfer_characteristics,
            self.matrix_coeffs,
            self.full_range_flag,
        )
    }
}

/// Chroma sample location (`vui_chroma_sample_loc_type_*`), values
/// `0..=6`; meaningful for 4:2:0 only.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum VuiChromaLoc {
    /// `vui_chroma_sample_loc_type_frame` (progressive-only source).
    Frame(u8),
    /// `vui_chroma_sample_loc_type_top_field` /
    /// `vui_chroma_sample_loc_type_bottom_field`.
    Fields { top: u8, bottom: u8 },
}

/// A parsed `vui_parameters()` structure.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[non_exhaustive]
pub struct VuiParameters {
    /// `vui_progressive_source_flag`.
    pub progressive_source_flag: bool,
    /// `vui_interlaced_source_flag`.
    pub interlaced_source_flag: bool,
    /// `vui_non_packed_constraint_flag` — no frame packing arrangement
    /// SEI message applies to the CLVS.
    pub non_packed_constraint_flag: bool,
    /// `vui_non_projected_constraint_flag` — no projection SEI message
    /// applies to the CLVS.
    pub non_projected_constraint_flag: bool,
    /// Present when `vui_aspect_ratio_info_present_flag`.
    pub aspect_ratio: Option<VuiAspectRatio>,
    /// `vui_overscan_appropriate_flag`, present when
    /// `vui_overscan_info_present_flag`.
    pub overscan_appropriate_flag: Option<bool>,
    /// Present when `vui_colour_description_present_flag`.
    pub colour_description: Option<VuiColourDescription>,
    /// Present when `vui_chroma_loc_info_present_flag`.
    pub chroma_loc: Option<VuiChromaLoc>,
}

impl VuiParameters {
    /// A structure with every optional part absent and all flags 0.
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder: mark the source progressive-only.
    pub fn with_progressive_source(mut self) -> Self {
        self.progressive_source_flag = true;
        self.interlaced_source_flag = false;
        self
    }

    /// Builder: set the colour description.
    pub fn with_colour_description(mut self, colour: VuiColourDescription) -> Self {
        self.colour_description = Some(colour);
        self
    }

    /// Builder: set the sample aspect ratio fields.
    pub fn with_aspect_ratio(mut self, aspect: VuiAspectRatio) -> Self {
        self.aspect_ratio = Some(aspect);
        self
    }

    /// Builder: set the chroma sample location (each value `0..=6`).
    pub fn with_chroma_loc(mut self, loc: VuiChromaLoc) -> Self {
        self.chroma_loc = Some(loc);
        self
    }

    /// The source scan type the two source flags describe.
    pub fn source_scan(&self) -> SourceScan {
        match (self.progressive_source_flag, self.interlaced_source_flag) {
            (true, false) => SourceScan::Progressive,
            (false, true) => SourceScan::Interlaced,
            (false, false) => SourceScan::Unspecified,
            (true, true) => SourceScan::PerPicture,
        }
    }

    /// The colour description as a core [`ColorSignal`]. Without a
    /// colour description the three code points and the range are
    /// unspecified: H.274 infers the code points to 2 ("unknown or
    /// determined by other means"), and the inferred
    /// `vui_full_range_flag = 0` is not a statement the stream made.
    pub fn color_signal(&self) -> ColorSignal {
        self.colour_description
            .map(|c| c.color_signal())
            .unwrap_or_else(ColorSignal::unspecified)
    }

    /// Parse a `vui_payload()` — exactly the
    /// `sps_vui_payload_size_minus1 + 1` bytes the SPS carries.
    pub fn parse_payload(payload: &[u8]) -> Result<Self> {
        if payload.is_empty() || payload.len() > MAX_VUI_PAYLOAD_BYTES {
            return Err(Error::invalid(format!(
                "h266 VUI: payload size {} outside 1..={MAX_VUI_PAYLOAD_BYTES}",
                payload.len()
            )));
        }
        let mut br = BitReader::new(payload);
        let mut vui = VuiParameters {
            progressive_source_flag: br.u1()? == 1,
            interlaced_source_flag: br.u1()? == 1,
            non_packed_constraint_flag: br.u1()? == 1,
            non_projected_constraint_flag: br.u1()? == 1,
            ..Default::default()
        };
        if br.u1()? == 1 {
            let constant_flag = br.u1()? == 1;
            let idc = br.u(8)? as u8;
            let (sar_width, sar_height) = if idc == ASPECT_RATIO_IDC_EXPLICIT {
                (br.u(16)? as u16, br.u(16)? as u16)
            } else {
                (0, 0)
            };
            vui.aspect_ratio = Some(VuiAspectRatio {
                constant_flag,
                idc,
                sar_width,
                sar_height,
            });
        }
        if br.u1()? == 1 {
            vui.overscan_appropriate_flag = Some(br.u1()? == 1);
        }
        if br.u1()? == 1 {
            vui.colour_description = Some(VuiColourDescription {
                colour_primaries: br.u(8)? as u8,
                transfer_characteristics: br.u(8)? as u8,
                matrix_coeffs: br.u(8)? as u8,
                full_range_flag: br.u1()? == 1,
            });
        }
        if br.u1()? == 1 {
            let mut loc = || -> Result<u8> {
                let v = br.ue()?;
                if v > 6 {
                    return Err(Error::invalid(format!(
                        "h266 VUI: chroma sample location type {v} outside 0..=6"
                    )));
                }
                Ok(v as u8)
            };
            vui.chroma_loc = Some(
                if vui.progressive_source_flag && !vui.interlaced_source_flag {
                    VuiChromaLoc::Frame(loc()?)
                } else {
                    let top = loc()?;
                    let bottom = loc()?;
                    VuiChromaLoc::Fields { top, bottom }
                },
            );
        }
        // H.266 §7.3.2.21: when bits remain, they are reserved
        // extension data closed by a `1` stop bit and zero padding —
        // i.e. the last set bit of the payload is the stop bit. A
        // remainder with no set bit at all cannot be that.
        let consumed = br.bit_position() as usize;
        if consumed < payload.len() * 8 {
            let tail_has_one =
                (consumed..payload.len() * 8).any(|bit| payload[bit / 8] >> (7 - bit % 8) & 1 == 1);
            if !tail_has_one {
                return Err(Error::invalid(
                    "h266 VUI: payload tail carries no vui_payload_bit_equal_to_one",
                ));
            }
        }
        Ok(vui)
    }

    /// Serialise as a `vui_payload()`: the H.274 syntax, then the stop
    /// bit and zero padding when the syntax does not end on a byte
    /// boundary. Returns the payload bytes
    /// (`sps_vui_payload_size_minus1 + 1` of them).
    pub fn write_payload(&self) -> Vec<u8> {
        let mut bits = BitSink::default();
        bits.put(u32::from(self.progressive_source_flag), 1);
        bits.put(u32::from(self.interlaced_source_flag), 1);
        bits.put(u32::from(self.non_packed_constraint_flag), 1);
        bits.put(u32::from(self.non_projected_constraint_flag), 1);
        bits.put(u32::from(self.aspect_ratio.is_some()), 1);
        if let Some(a) = &self.aspect_ratio {
            bits.put(u32::from(a.constant_flag), 1);
            bits.put(u32::from(a.idc), 8);
            if a.idc == ASPECT_RATIO_IDC_EXPLICIT {
                bits.put(u32::from(a.sar_width), 16);
                bits.put(u32::from(a.sar_height), 16);
            }
        }
        bits.put(u32::from(self.overscan_appropriate_flag.is_some()), 1);
        if let Some(f) = self.overscan_appropriate_flag {
            bits.put(u32::from(f), 1);
        }
        bits.put(u32::from(self.colour_description.is_some()), 1);
        if let Some(c) = &self.colour_description {
            bits.put(u32::from(c.colour_primaries), 8);
            bits.put(u32::from(c.transfer_characteristics), 8);
            bits.put(u32::from(c.matrix_coeffs), 8);
            bits.put(u32::from(c.full_range_flag), 1);
        }
        bits.put(u32::from(self.chroma_loc.is_some()), 1);
        if let Some(loc) = self.chroma_loc {
            // The syntax branch follows the source flags, not the
            // variant: a frame value on a non-progressive source is
            // written for both fields, and vice versa the top value.
            let progressive = self.progressive_source_flag && !self.interlaced_source_flag;
            match (progressive, loc) {
                (true, VuiChromaLoc::Frame(v)) | (true, VuiChromaLoc::Fields { top: v, .. }) => {
                    bits.put_ue(u32::from(v.min(6)));
                }
                (false, VuiChromaLoc::Frame(v)) => {
                    bits.put_ue(u32::from(v.min(6)));
                    bits.put_ue(u32::from(v.min(6)));
                }
                (false, VuiChromaLoc::Fields { top, bottom }) => {
                    bits.put_ue(u32::from(top.min(6)));
                    bits.put_ue(u32::from(bottom.min(6)));
                }
            }
        }
        if !bits.len.is_multiple_of(8) {
            bits.put(1, 1); // vui_payload_bit_equal_to_one
            while !bits.len.is_multiple_of(8) {
                bits.put(0, 1); // vui_payload_bit_equal_to_zero
            }
        }
        bits.bytes
    }
}

/// MSB-first bit accumulator for [`VuiParameters::write_payload`].
#[derive(Default)]
struct BitSink {
    bytes: Vec<u8>,
    len: usize,
}

impl BitSink {
    fn put(&mut self, value: u32, n: u32) {
        for i in (0..n).rev() {
            if self.len.is_multiple_of(8) {
                self.bytes.push(0);
            }
            let bit = (value >> i & 1) as u8;
            *self.bytes.last_mut().unwrap() |= bit << (7 - self.len % 8);
            self.len += 1;
        }
    }

    /// 0-th order Exp-Golomb, `value < 2^31`.
    fn put_ue(&mut self, value: u32) {
        let v = value + 1;
        let bits = 32 - v.leading_zeros();
        self.put(0, bits - 1);
        self.put(v, bits);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::ColorRange;

    #[test]
    fn empty_flags_round_trip_in_one_byte() {
        let vui = VuiParameters::new();
        let bytes = vui.write_payload();
        // 8 zero flag bits: byte aligned, no stop bit needed.
        assert_eq!(bytes, [0x00]);
        assert_eq!(VuiParameters::parse_payload(&bytes).unwrap(), vui);
        assert_eq!(vui.color_signal(), ColorSignal::unspecified());
        assert_eq!(vui.source_scan(), SourceScan::Unspecified);
    }

    #[test]
    fn colour_description_round_trips_and_maps_to_the_core_signal() {
        let vui = VuiParameters::new()
            .with_progressive_source()
            .with_colour_description(VuiColourDescription::new(9, 16, 9, false));
        let bytes = vui.write_payload();
        let back = VuiParameters::parse_payload(&bytes).unwrap();
        assert_eq!(back, vui);
        let s = back.color_signal();
        assert_eq!(
            (s.primaries.0, s.transfer.0, s.matrix.0, s.range),
            (9, 16, 9, ColorRange::Limited)
        );
        assert_eq!(back.source_scan(), SourceScan::Progressive);
        assert_eq!(
            VuiColourDescription::from_color_signal(&s),
            vui.colour_description
        );
        assert_eq!(
            VuiColourDescription::from_color_signal(&ColorSignal::unspecified()),
            None
        );
    }

    #[test]
    fn every_optional_part_round_trips() {
        for progressive in [false, true] {
            let mut vui = VuiParameters::new()
                .with_aspect_ratio(VuiAspectRatio::explicit(64, 45))
                .with_colour_description(VuiColourDescription::new(1, 13, 0, true));
            vui.progressive_source_flag = progressive;
            vui.interlaced_source_flag = true;
            vui.non_packed_constraint_flag = true;
            vui.overscan_appropriate_flag = Some(true);
            vui.chroma_loc = Some(VuiChromaLoc::Fields { top: 2, bottom: 5 });
            let back = VuiParameters::parse_payload(&vui.write_payload()).unwrap();
            assert_eq!(back, vui);
            assert_eq!(back.aspect_ratio.unwrap().explicit_ratio(), Some((64, 45)));
        }
        let vui = VuiParameters::new()
            .with_progressive_source()
            .with_chroma_loc(VuiChromaLoc::Frame(6));
        let back = VuiParameters::parse_payload(&vui.write_payload()).unwrap();
        assert_eq!(back.chroma_loc, Some(VuiChromaLoc::Frame(6)));
    }

    #[test]
    fn extension_data_is_ignored() {
        let mut bytes = VuiParameters::new()
            .with_colour_description(VuiColourDescription::new(1, 1, 1, false))
            .write_payload();
        let reference = VuiParameters::parse_payload(&bytes).unwrap();
        // Later-version payload: more bytes after ours, closed by a
        // stop bit.
        let last = bytes.len() - 1;
        bytes[last] &= 0x80; // keep our one used bit, drop the stop bit
        bytes.extend_from_slice(&[0xab, 0xcd, 0x80]);
        assert_eq!(VuiParameters::parse_payload(&bytes).unwrap(), reference);
    }

    #[test]
    fn malformed_payloads_are_errors() {
        assert!(VuiParameters::parse_payload(&[]).is_err());
        assert!(VuiParameters::parse_payload(&[0u8; MAX_VUI_PAYLOAD_BYTES + 1]).is_err());
        // Aspect ratio present, idc 255, but the SAR is cut off.
        assert!(VuiParameters::parse_payload(&[0x0f, 0xff]).is_err());
        // Unused tail without a stop bit.
        assert!(VuiParameters::parse_payload(&[0x00, 0x00]).is_err());
        // Chroma location 7 (ue code 0001000) is out of range.
        // flags: 0000 0 0 0 1 | ue(7) = 0001000
        assert!(VuiParameters::parse_payload(&[0x01, 0b0001_0001]).is_err());
        for len in 0..40usize {
            for seed in 0..64u32 {
                let bytes: Vec<u8> = (0..len)
                    .map(|i| (seed.wrapping_mul(2654435761).rotate_left(i as u32 * 5) >> 7) as u8)
                    .collect();
                let _ = VuiParameters::parse_payload(&bytes);
            }
        }
    }
}
