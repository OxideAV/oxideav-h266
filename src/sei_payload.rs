//! Typed SEI payloads — the messages Rec. ITU-T H.274 | ISO/IEC
//! 23002-7 defines and H.266 Annex D (`sei_payload()`, §D.2.1) carries
//! that mean something to a decoder or to whoever presents its
//! output:
//!
//! | `payloadType` | message | H.274 |
//! |---|---|---|
//! | 3 | filler payload | §8.2 |
//! | 4 | user data registered by Rec. ITU-T T.35 | §8.3 |
//! | 5 | user data unregistered | §8.4 |
//! | 129 | parameter sets inclusion indication | §8.7 |
//! | 132 | decoded picture hash (suffix SEI only) | §8.8 |
//! | 137 | mastering display colour volume | §8.9 |
//! | 144 | content light level information | §8.10 |
//! | 145 | dependent random access point indication | §8.11 |
//! | 147 | alternative transfer characteristics | §8.12 |
//! | 148 | ambient viewing environment | §8.13 |
//! | 149 | content colour volume | §8.14 |
//! | 168 | frame-field information | §8.16 |
//! | 204 | sample aspect ratio information | §8.17 |
//! | 47 | display orientation | §8.25 |
//!
//! [`parse_sei_payload`] turns one [`SeiMessage`](crate::sei_message::SeiMessage)
//! body into a [`SeiPayload`]; every other type comes back as
//! [`SeiPayload::Other`] with its `payloadType`. Bits after the
//! syntax a message defines (the §D.2.1 reserved extension data and
//! the stop bit) are ignored, as §D.2.2 requires of a decoder.
//!
//! Every parser reads from the payload slice only — the size is
//! bounded by the NAL unit that carried it, and the variable-length
//! bodies (user data) are capped at [`MAX_USER_DATA_BYTES`] before
//! they are copied.

use oxideav_core::{Error, Result};

use crate::bitreader::BitReader;

/// `payloadType` codes of the messages parsed here.
pub mod payload_type {
    pub const FILLER_PAYLOAD: u32 = 3;
    pub const USER_DATA_REGISTERED_ITU_T_T35: u32 = 4;
    pub const USER_DATA_UNREGISTERED: u32 = 5;
    pub const DISPLAY_ORIENTATION: u32 = 47;
    pub const PARAMETER_SETS_INCLUSION_INDICATION: u32 = 129;
    pub const DECODED_PICTURE_HASH: u32 = 132;
    pub const MASTERING_DISPLAY_COLOUR_VOLUME: u32 = 137;
    pub const CONTENT_LIGHT_LEVEL_INFO: u32 = 144;
    pub const DEPENDENT_RAP_INDICATION: u32 = 145;
    pub const ALTERNATIVE_TRANSFER_CHARACTERISTICS: u32 = 147;
    pub const AMBIENT_VIEWING_ENVIRONMENT: u32 = 148;
    pub const CONTENT_COLOUR_VOLUME: u32 = 149;
    pub const FRAME_FIELD_INFO: u32 = 168;
    pub const SAMPLE_ASPECT_RATIO_INFO: u32 = 204;
}

/// Largest user-data body kept by [`parse_sei_payload`]; a longer one
/// is truncated to this many bytes (the message says so through
/// `truncated`).
pub const MAX_USER_DATA_BYTES: usize = 1 << 20;

/// One component's value in a decoded picture hash message.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum PictureHashValue {
    /// `dph_sei_picture_md5[ cIdx ]`.
    Md5([u8; 16]),
    /// `dph_sei_picture_crc[ cIdx ]`.
    Crc(u16),
    /// `dph_sei_picture_checksum[ cIdx ]`.
    Checksum(u32),
}

/// `decoded_picture_hash()` — H.274 §8.8.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct DecodedPictureHash {
    /// `dph_sei_hash_type`: 0 MD5, 1 CRC, 2 checksum; other values are
    /// reserved and carry no components.
    pub hash_type: u8,
    /// `dph_sei_single_component_flag` (shall equal
    /// `ChromaFormatIdc == 0`).
    pub single_component_flag: bool,
    /// One value per colour component (1 or 3); empty for a reserved
    /// `hash_type`.
    pub components: Vec<PictureHashValue>,
}

impl DecodedPictureHash {
    /// Serialise as the SEI payload body.
    pub fn write_payload(&self) -> Vec<u8> {
        let mut out = vec![self.hash_type, u8::from(self.single_component_flag) << 7];
        for c in &self.components {
            match c {
                PictureHashValue::Md5(d) => out.extend_from_slice(d),
                PictureHashValue::Crc(v) => out.extend_from_slice(&v.to_be_bytes()),
                PictureHashValue::Checksum(v) => out.extend_from_slice(&v.to_be_bytes()),
            }
        }
        out
    }
}

/// `mastering_display_colour_volume()` — H.274 §8.9. Chromaticities
/// are in increments of 0.00002, luminances in units of
/// 0.0001 cd/m².
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[non_exhaustive]
pub struct MasteringDisplayColourVolume {
    /// `mdcv_display_primaries_x[ c ]` / `_y[ c ]`, `c = 0..3`
    /// (suggested order: green, blue, red).
    pub display_primaries: [(u16, u16); 3],
    /// `mdcv_white_point_x` / `_y`.
    pub white_point: (u16, u16),
    /// `mdcv_max_display_mastering_luminance`.
    pub max_display_mastering_luminance: u32,
    /// `mdcv_min_display_mastering_luminance`.
    pub min_display_mastering_luminance: u32,
}

impl MasteringDisplayColourVolume {
    /// Build from the primaries, the white point and the luminance
    /// range, all in the message's own units.
    pub const fn new(
        display_primaries: [(u16, u16); 3],
        white_point: (u16, u16),
        max_display_mastering_luminance: u32,
        min_display_mastering_luminance: u32,
    ) -> Self {
        Self {
            display_primaries,
            white_point,
            max_display_mastering_luminance,
            min_display_mastering_luminance,
        }
    }

    /// Maximum luminance in cd/m², when the value is in the range the
    /// semantics give a meaning to (50 000 ..= 100 000 000).
    pub fn max_luminance_nits(&self) -> Option<f64> {
        (50_000..=100_000_000)
            .contains(&self.max_display_mastering_luminance)
            .then(|| f64::from(self.max_display_mastering_luminance) * 0.0001)
    }

    /// Minimum luminance in cd/m², when in 1 ..= 50 000.
    pub fn min_luminance_nits(&self) -> Option<f64> {
        (1..=50_000)
            .contains(&self.min_display_mastering_luminance)
            .then(|| f64::from(self.min_display_mastering_luminance) * 0.0001)
    }

    /// Serialise as the SEI payload body (24 bytes).
    pub fn write_payload(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(24);
        for (x, y) in self.display_primaries {
            out.extend_from_slice(&x.to_be_bytes());
            out.extend_from_slice(&y.to_be_bytes());
        }
        out.extend_from_slice(&self.white_point.0.to_be_bytes());
        out.extend_from_slice(&self.white_point.1.to_be_bytes());
        out.extend_from_slice(&self.max_display_mastering_luminance.to_be_bytes());
        out.extend_from_slice(&self.min_display_mastering_luminance.to_be_bytes());
        out
    }
}

/// `content_light_level_info()` — H.274 §8.10, both in cd/m²
/// (0 = unknown).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[non_exhaustive]
pub struct ContentLightLevelInfo {
    /// `clli_max_content_light_level`.
    pub max_content_light_level: u16,
    /// `clli_max_pic_average_light_level`.
    pub max_pic_average_light_level: u16,
}

impl ContentLightLevelInfo {
    pub const fn new(max_content_light_level: u16, max_pic_average_light_level: u16) -> Self {
        Self {
            max_content_light_level,
            max_pic_average_light_level,
        }
    }

    /// Serialise as the SEI payload body (4 bytes).
    pub fn write_payload(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(4);
        out.extend_from_slice(&self.max_content_light_level.to_be_bytes());
        out.extend_from_slice(&self.max_pic_average_light_level.to_be_bytes());
        out
    }
}

/// `ambient_viewing_environment()` — H.274 §8.13.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[non_exhaustive]
pub struct AmbientViewingEnvironment {
    /// `ambient_illuminance`, units of 0.0001 lux.
    pub ambient_illuminance: u32,
    /// `ambient_light_x` / `ambient_light_y`, increments of 0.00002.
    pub ambient_light: (u16, u16),
}

/// `content_colour_volume()` — H.274 §8.14.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[non_exhaustive]
pub struct ContentColourVolume {
    /// `ccv_cancel_flag` — cancels the persistence of an earlier
    /// message; every other field is then absent.
    pub cancel_flag: bool,
    /// `ccv_persistence_flag`.
    pub persistence_flag: bool,
    /// `ccv_primaries_x[ c ]` / `_y[ c ]` (increments of 0.00002).
    pub primaries: Option<[(i32, i32); 3]>,
    /// `ccv_min_luminance_value` (units of 0.0000001 cd/m²).
    pub min_luminance_value: Option<u32>,
    /// `ccv_max_luminance_value`.
    pub max_luminance_value: Option<u32>,
    /// `ccv_avg_luminance_value`.
    pub avg_luminance_value: Option<u32>,
}

/// `frame_field_info()` — H.274 §8.16.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[non_exhaustive]
pub struct FrameFieldInfo {
    /// `ffi_field_pic_flag` — the picture is a field.
    pub field_pic_flag: bool,
    /// `ffi_bottom_field_flag` (fields only).
    pub bottom_field_flag: bool,
    /// `ffi_paired_with_next_field_flag`, present when
    /// `ffi_pairing_indicated_flag` (fields only).
    pub paired_with_next_field_flag: Option<bool>,
    /// `ffi_display_fields_from_frame_flag` (frames only).
    pub display_fields_from_frame_flag: bool,
    /// `ffi_top_field_first_flag`, present when the frame is to be
    /// displayed as fields.
    pub top_field_first_flag: Option<bool>,
    /// `ffi_display_elemental_periods_minus1` (frames only).
    pub display_elemental_periods_minus1: u8,
    /// `ffi_source_scan_type`: 0 interlaced, 1 progressive,
    /// 2 unspecified, 3 reserved.
    pub source_scan_type: u8,
    /// `ffi_duplicate_flag` — the picture repeats the previous one in
    /// output order.
    pub duplicate_flag: bool,
}

/// `sample_aspect_ratio_info()` — H.274 §8.17.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[non_exhaustive]
pub struct SampleAspectRatioInfo {
    /// `sari_cancel_flag`.
    pub cancel_flag: bool,
    /// `sari_persistence_flag`.
    pub persistence_flag: bool,
    /// `sari_aspect_ratio_idc` (255 = explicit).
    pub aspect_ratio_idc: u8,
    /// `sari_sar_width` / `sari_sar_height` (explicit ratio only).
    pub sar: (u16, u16),
}

/// `display_orientation()` — H.274 §8.25.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[non_exhaustive]
pub struct DisplayOrientation {
    /// `display_orientation_cancel_flag`.
    pub cancel_flag: bool,
    /// `display_orientation_persistence_flag`.
    pub persistence_flag: bool,
    /// `display_orientation_transform_type` (0..=7).
    pub transform_type: u8,
}

/// `user_data_registered_itu_t_t35()` — H.274 §8.3.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct UserDataRegistered {
    /// `itu_t_t35_country_code`.
    pub country_code: u8,
    /// `itu_t_t35_country_code_extension_byte`, present when the
    /// country code is 0xFF.
    pub country_code_extension: Option<u8>,
    /// The `itu_t_t35_payload_byte`s.
    pub data: Vec<u8>,
    /// The body was longer than [`MAX_USER_DATA_BYTES`] and was cut.
    pub truncated: bool,
}

/// `user_data_unregistered()` — H.274 §8.4.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct UserDataUnregistered {
    /// `uuid_iso_iec_11578`.
    pub uuid: [u8; 16],
    /// The `user_data_payload_byte`s.
    pub data: Vec<u8>,
    /// The body was longer than [`MAX_USER_DATA_BYTES`] and was cut.
    pub truncated: bool,
}

/// One parsed SEI payload.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum SeiPayload {
    /// `filler_payload()` of this many bytes.
    Filler(usize),
    UserDataRegistered(UserDataRegistered),
    UserDataUnregistered(UserDataUnregistered),
    DisplayOrientation(DisplayOrientation),
    /// `psii_self_contained_clvs_flag`.
    ParameterSetsInclusionIndication {
        self_contained_clvs_flag: bool,
    },
    DecodedPictureHash(DecodedPictureHash),
    MasteringDisplayColourVolume(MasteringDisplayColourVolume),
    ContentLightLevelInfo(ContentLightLevelInfo),
    DependentRapIndication,
    /// `preferred_transfer_characteristics` — a Rec. ITU-T H.273 code
    /// point preferred over `vui_transfer_characteristics`.
    AlternativeTransferCharacteristics {
        preferred_transfer_characteristics: u8,
    },
    AmbientViewingEnvironment(AmbientViewingEnvironment),
    ContentColourVolume(ContentColourVolume),
    FrameFieldInfo(FrameFieldInfo),
    SampleAspectRatioInfo(SampleAspectRatioInfo),
    /// A message this module does not interpret.
    Other {
        payload_type: u32,
    },
}

fn cap_user_data(bytes: &[u8]) -> (Vec<u8>, bool) {
    let keep = bytes.len().min(MAX_USER_DATA_BYTES);
    (bytes[..keep].to_vec(), keep < bytes.len())
}

/// Parse one SEI message body. `suffix` is `true` for a message from
/// a suffix SEI NAL unit: §D.2.1 gives the decoded picture hash a
/// meaning only there (in a prefix SEI NAL unit type 132 is a reserved
/// message and comes back as [`SeiPayload::Other`]).
pub fn parse_sei_payload(payload_type: u32, payload: &[u8], suffix: bool) -> Result<SeiPayload> {
    use self::payload_type as pt;
    let mut br = BitReader::new(payload);
    Ok(match payload_type {
        pt::FILLER_PAYLOAD => SeiPayload::Filler(payload.len()),
        pt::USER_DATA_REGISTERED_ITU_T_T35 => {
            let country_code = br.u(8)? as u8;
            let (country_code_extension, start) = if country_code == 0xff {
                (Some(br.u(8)? as u8), 2)
            } else {
                (None, 1)
            };
            let (data, truncated) = cap_user_data(&payload[start..]);
            SeiPayload::UserDataRegistered(UserDataRegistered {
                country_code,
                country_code_extension,
                data,
                truncated,
            })
        }
        pt::USER_DATA_UNREGISTERED => {
            let uuid: [u8; 16] = payload
                .get(..16)
                .ok_or_else(|| {
                    Error::invalid("h266 SEI: user_data_unregistered shorter than its UUID")
                })?
                .try_into()
                .unwrap();
            let (data, truncated) = cap_user_data(&payload[16..]);
            SeiPayload::UserDataUnregistered(UserDataUnregistered {
                uuid,
                data,
                truncated,
            })
        }
        pt::DISPLAY_ORIENTATION => {
            let cancel_flag = br.u1()? == 1;
            let mut d = DisplayOrientation {
                cancel_flag,
                ..Default::default()
            };
            if !cancel_flag {
                d.persistence_flag = br.u1()? == 1;
                d.transform_type = br.u(3)? as u8;
                br.u(3)?; // display_orientation_reserved_zero_3bits
            }
            SeiPayload::DisplayOrientation(d)
        }
        pt::PARAMETER_SETS_INCLUSION_INDICATION => SeiPayload::ParameterSetsInclusionIndication {
            self_contained_clvs_flag: br.u1()? == 1,
        },
        pt::DECODED_PICTURE_HASH if suffix => {
            let hash_type = br.u(8)? as u8;
            let single_component_flag = br.u1()? == 1;
            br.u(7)?; // dph_sei_reserved_zero_7bits (ignored)
            let n = if single_component_flag { 1 } else { 3 };
            let mut components = Vec::with_capacity(n);
            for _ in 0..n {
                match hash_type {
                    0 => {
                        let mut d = [0u8; 16];
                        for b in &mut d {
                            *b = br.u(8)? as u8;
                        }
                        components.push(PictureHashValue::Md5(d));
                    }
                    1 => components.push(PictureHashValue::Crc(br.u(16)? as u16)),
                    2 => components.push(PictureHashValue::Checksum(br.u(32)?)),
                    _ => break, // reserved: no component syntax
                }
            }
            SeiPayload::DecodedPictureHash(DecodedPictureHash {
                hash_type,
                single_component_flag,
                components,
            })
        }
        pt::MASTERING_DISPLAY_COLOUR_VOLUME => {
            let mut m = MasteringDisplayColourVolume::default();
            for p in &mut m.display_primaries {
                *p = (br.u(16)? as u16, br.u(16)? as u16);
            }
            m.white_point = (br.u(16)? as u16, br.u(16)? as u16);
            m.max_display_mastering_luminance = br.u(32)?;
            m.min_display_mastering_luminance = br.u(32)?;
            SeiPayload::MasteringDisplayColourVolume(m)
        }
        pt::CONTENT_LIGHT_LEVEL_INFO => SeiPayload::ContentLightLevelInfo(ContentLightLevelInfo {
            max_content_light_level: br.u(16)? as u16,
            max_pic_average_light_level: br.u(16)? as u16,
        }),
        pt::DEPENDENT_RAP_INDICATION => SeiPayload::DependentRapIndication,
        pt::ALTERNATIVE_TRANSFER_CHARACTERISTICS => {
            SeiPayload::AlternativeTransferCharacteristics {
                preferred_transfer_characteristics: br.u(8)? as u8,
            }
        }
        pt::AMBIENT_VIEWING_ENVIRONMENT => {
            SeiPayload::AmbientViewingEnvironment(AmbientViewingEnvironment {
                ambient_illuminance: br.u(32)?,
                ambient_light: (br.u(16)? as u16, br.u(16)? as u16),
            })
        }
        pt::CONTENT_COLOUR_VOLUME => {
            let mut c = ContentColourVolume {
                cancel_flag: br.u1()? == 1,
                ..Default::default()
            };
            if !c.cancel_flag {
                c.persistence_flag = br.u1()? == 1;
                let primaries = br.u1()? == 1;
                let min = br.u1()? == 1;
                let max = br.u1()? == 1;
                let avg = br.u1()? == 1;
                br.u(2)?; // ccv_reserved_zero_2bits
                if primaries {
                    let mut p = [(0i32, 0i32); 3];
                    for slot in &mut p {
                        *slot = (br.u(32)? as i32, br.u(32)? as i32);
                    }
                    c.primaries = Some(p);
                }
                if min {
                    c.min_luminance_value = Some(br.u(32)?);
                }
                if max {
                    c.max_luminance_value = Some(br.u(32)?);
                }
                if avg {
                    c.avg_luminance_value = Some(br.u(32)?);
                }
            }
            SeiPayload::ContentColourVolume(c)
        }
        pt::FRAME_FIELD_INFO => {
            let mut f = FrameFieldInfo {
                field_pic_flag: br.u1()? == 1,
                ..Default::default()
            };
            if f.field_pic_flag {
                f.bottom_field_flag = br.u1()? == 1;
                if br.u1()? == 1 {
                    f.paired_with_next_field_flag = Some(br.u1()? == 1);
                }
            } else {
                f.display_fields_from_frame_flag = br.u1()? == 1;
                if f.display_fields_from_frame_flag {
                    f.top_field_first_flag = Some(br.u1()? == 1);
                }
                f.display_elemental_periods_minus1 = br.u(8)? as u8;
            }
            f.source_scan_type = br.u(2)? as u8;
            f.duplicate_flag = br.u1()? == 1;
            SeiPayload::FrameFieldInfo(f)
        }
        pt::SAMPLE_ASPECT_RATIO_INFO => {
            let mut s = SampleAspectRatioInfo {
                cancel_flag: br.u1()? == 1,
                ..Default::default()
            };
            if !s.cancel_flag {
                s.persistence_flag = br.u1()? == 1;
                s.aspect_ratio_idc = br.u(8)? as u8;
                if s.aspect_ratio_idc == 255 {
                    s.sar = (br.u(16)? as u16, br.u(16)? as u16);
                }
            }
            SeiPayload::SampleAspectRatioInfo(s)
        }
        other => SeiPayload::Other {
            payload_type: other,
        },
    })
}

/// Serialise one `sei_message()` (§7.3.6): the `payloadType` and
/// `payloadSize` in their 0xFF-extended byte form, then the body.
pub fn write_sei_message(payload_type: u32, body: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(body.len() + 8);
    for mut v in [payload_type as usize, body.len()] {
        while v >= 255 {
            out.push(0xff);
            v -= 255;
        }
        out.push(v as u8);
    }
    out.extend_from_slice(body);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sei_rbsp::parse_sei_rbsp;

    #[test]
    fn picture_hash_bodies_round_trip() {
        let cases = [
            DecodedPictureHash {
                hash_type: 0,
                single_component_flag: false,
                components: vec![
                    PictureHashValue::Md5([1; 16]),
                    PictureHashValue::Md5([2; 16]),
                    PictureHashValue::Md5([3; 16]),
                ],
            },
            DecodedPictureHash {
                hash_type: 1,
                single_component_flag: true,
                components: vec![PictureHashValue::Crc(0xbeef)],
            },
            DecodedPictureHash {
                hash_type: 2,
                single_component_flag: false,
                components: vec![
                    PictureHashValue::Checksum(1),
                    PictureHashValue::Checksum(0xdead_beef),
                    PictureHashValue::Checksum(u32::MAX),
                ],
            },
        ];
        for h in cases {
            let body = h.write_payload();
            assert_eq!(
                parse_sei_payload(132, &body, true).unwrap(),
                SeiPayload::DecodedPictureHash(h.clone())
            );
            // Not a picture hash in a prefix SEI NAL unit.
            assert_eq!(
                parse_sei_payload(132, &body, false).unwrap(),
                SeiPayload::Other { payload_type: 132 }
            );
            // Truncation is an error, never a panic.
            for cut in 0..body.len() {
                assert!(parse_sei_payload(132, &body[..cut], true).is_err());
            }
        }
        // Reserved hash type: parsed, no components.
        let SeiPayload::DecodedPictureHash(h) =
            parse_sei_payload(132, &[9, 0, 1, 2], true).unwrap()
        else {
            panic!()
        };
        assert!(h.components.is_empty());
    }

    #[test]
    fn hdr_bodies_round_trip_through_an_sei_rbsp() {
        let mdcv = MasteringDisplayColourVolume::new(
            [(13250, 34500), (7500, 3000), (34000, 16000)],
            (15635, 16450),
            10_000_000,
            50,
        );
        let cll = ContentLightLevelInfo::new(1000, 400);
        let mut rbsp = write_sei_message(137, &mdcv.write_payload());
        rbsp.extend(write_sei_message(144, &cll.write_payload()));
        rbsp.extend(write_sei_message(147, &[18]));
        rbsp.push(0x80);
        let sei = parse_sei_rbsp(&rbsp).unwrap();
        let parsed: Vec<SeiPayload> = sei
            .messages
            .iter()
            .map(|m| parse_sei_payload(m.payload_type, m.payload, false).unwrap())
            .collect();
        assert_eq!(
            parsed,
            vec![
                SeiPayload::MasteringDisplayColourVolume(mdcv),
                SeiPayload::ContentLightLevelInfo(cll),
                SeiPayload::AlternativeTransferCharacteristics {
                    preferred_transfer_characteristics: 18
                },
            ]
        );
        assert_eq!(mdcv.max_luminance_nits(), Some(1000.0));
        assert_eq!(mdcv.min_luminance_nits(), Some(0.005));
        assert_eq!(
            MasteringDisplayColourVolume::default().max_luminance_nits(),
            None
        );
    }

    #[test]
    fn frame_field_and_aspect_bodies() {
        // Frame, displayed as fields, top first, 1 elemental period,
        // progressive source, not a duplicate:
        // 0 1 1 00000000 01 0 -> 0110 0000 0000 1000
        let SeiPayload::FrameFieldInfo(f) = parse_sei_payload(168, &[0x60, 0x08], false).unwrap()
        else {
            panic!()
        };
        assert!(!f.field_pic_flag && f.display_fields_from_frame_flag);
        assert_eq!(f.top_field_first_flag, Some(true));
        assert_eq!(f.display_elemental_periods_minus1, 0);
        assert_eq!(f.source_scan_type, 1);
        assert!(!f.duplicate_flag);
        // Bottom field, paired with the next, interlaced source,
        // duplicate: 1 1 1 1 00 1 -> 1111 0010
        let SeiPayload::FrameFieldInfo(f) = parse_sei_payload(168, &[0xf2], false).unwrap() else {
            panic!()
        };
        assert!(f.field_pic_flag && f.bottom_field_flag && f.duplicate_flag);
        assert_eq!(f.paired_with_next_field_flag, Some(true));
        assert_eq!(f.source_scan_type, 0);

        // SARI: not cancelled, persistent, idc 255, 4:3.
        // 0 1 11111111 0000000000000100 0000000000000011
        let body = [0x7f, 0xc0, 0x01, 0x00, 0x00, 0xc0];
        let SeiPayload::SampleAspectRatioInfo(s) = parse_sei_payload(204, &body, false).unwrap()
        else {
            panic!()
        };
        assert!(!s.cancel_flag && s.persistence_flag);
        assert_eq!((s.aspect_ratio_idc, s.sar), (255, (4, 3)));
        assert!(parse_sei_payload(204, &body[..3], false).is_err());
    }

    #[test]
    fn user_data_and_unknown_types() {
        let mut body = vec![7u8; 16];
        body.extend_from_slice(b"hello");
        let SeiPayload::UserDataUnregistered(u) = parse_sei_payload(5, &body, false).unwrap()
        else {
            panic!()
        };
        assert_eq!(
            (u.uuid, u.data.as_slice(), u.truncated),
            ([7; 16], &b"hello"[..], false)
        );
        assert!(parse_sei_payload(5, &body[..15], false).is_err());

        let SeiPayload::UserDataRegistered(r) =
            parse_sei_payload(4, &[0xff, 0x12, 1, 2, 3], true).unwrap()
        else {
            panic!()
        };
        assert_eq!(
            (r.country_code, r.country_code_extension, r.data.as_slice()),
            (0xff, Some(0x12), &[1u8, 2, 3][..])
        );
        assert!(parse_sei_payload(4, &[], false).is_err());
        assert_eq!(
            parse_sei_payload(1234, &[1, 2, 3], false).unwrap(),
            SeiPayload::Other { payload_type: 1234 }
        );
        assert_eq!(
            parse_sei_payload(3, &[0xff; 9], false).unwrap(),
            SeiPayload::Filler(9)
        );
    }

    #[test]
    fn long_type_and_size_codes_round_trip() {
        let body = vec![0xaa; 600];
        let mut rbsp = write_sei_message(300, &body);
        rbsp.push(0x80);
        let sei = parse_sei_rbsp(&rbsp).unwrap();
        assert_eq!(sei.messages.len(), 1);
        assert_eq!(sei.messages[0].payload_type, 300);
        assert_eq!(sei.messages[0].payload, &body[..]);
    }
}
