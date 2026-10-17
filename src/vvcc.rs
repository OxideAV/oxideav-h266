//! ISO/IEC 14496-15 §11.2.4 `VvcDecoderConfigurationRecord` — the
//! `vvcC` box payload of a `vvc1` / `vvi1` sample entry (and of a
//! HEIF VVC item): builder, parser and sample framing.
//!
//! The record is externally framed; it carries the NAL length-prefix
//! size of the samples, an optional profile / tier / level block
//! (`VvcPTLRecord`, §11.2.4.1) with the chroma format, bit depth and
//! maximum picture size, and arrays of initialisation non-VCL NAL
//! units (DCI, OPI, VPS, SPS, PPS, prefix APS, prefix SEI — §11.2.4.2
//! recommends that order).
//!
//! [`VvcDecoderConfigurationRecord::from_annex_b`] builds the record
//! from an Annex B elementary stream (the parameter sets this crate's
//! encoder emits in-band), [`annex_b_to_length_prefixed`] turns the
//! same stream into §4.2.3 length-prefixed sample data, and
//! [`VvcDecoderConfigurationRecord::parse`] reads a record a container
//! hands the registry decoder as `CodecParameters::extradata`.

use oxideav_core::{Error, Result};

use crate::bitreader::BitReader;
use crate::nal::{extract_rbsp, iter_annex_b, NalHeader, NalUnitType};
use crate::ptl::ProfileTierLevel;
use crate::sps::{parse_sps, SeqParameterSet};
use crate::vps::parse_vps;

/// One `nal_unit_type` array of the record.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct NalArray {
    /// `array_completeness` — every NAL unit of this type lives in the
    /// record (a `vvc1` sample entry requires it for SPS / PPS).
    pub array_completeness: bool,
    /// The §7.4.2.2 `nal_unit_type` of every unit in the array.
    pub nal_unit_type: u8,
    /// Complete NAL units (2-byte header + emulation-prevented
    /// payload, no start code).
    pub nal_units: Vec<Vec<u8>>,
}

impl NalArray {
    /// An array of `nal_unit_type` units.
    pub fn new(nal_unit_type: u8, array_completeness: bool, nal_units: Vec<Vec<u8>>) -> Self {
        Self {
            array_completeness,
            nal_unit_type,
            nal_units,
        }
    }
}

/// The `ptl_present_flag = 1` block: `VvcPTLRecord` plus the stream
/// facts around it (§11.2.4.2.2).
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct VvccPtl {
    /// `ols_idx` — the output layer set the record describes.
    pub ols_idx: u16,
    /// `num_sublayers` (1 = not temporally scalable, 0 = unknown).
    pub num_sublayers: u8,
    /// `constant_frame_rate` (0 = unknown, 1 = constant, 2 = per
    /// sublayer).
    pub constant_frame_rate: u8,
    /// `chroma_format_idc` = the SPS `sps_chroma_format_idc`.
    pub chroma_format_idc: u8,
    /// `bit_depth_minus8` = the SPS `sps_bitdepth_minus8`.
    pub bit_depth_minus8: u8,
    /// `general_profile_idc` (u(7)).
    pub general_profile_idc: u8,
    /// `general_tier_flag`.
    pub general_tier_flag: bool,
    /// `general_level_idc`.
    pub general_level_idc: u8,
    /// `ptl_frame_only_constraint_flag`.
    pub ptl_frame_only_constraint_flag: bool,
    /// `ptl_multilayer_enabled_flag`.
    pub ptl_multilayer_enabled_flag: bool,
    /// `general_constraint_info` — the §7.3.3.2
    /// `general_constraints_info()` bits as they sit in the parameter
    /// set, packed MSB-first into `num_bytes_constraint_info` bytes
    /// whose first two bits (the slot of the two flags above in the
    /// SPS byte layout) are zero. One zero byte when the SPS carries
    /// `gci_present_flag = 0`.
    pub general_constraint_info: Vec<u8>,
    /// `sublayer_level_idc[i]` for `i = 0 .. num_sublayers − 1`
    /// (`None` where `ptl_sublayer_level_present_flag[i] = 0`).
    pub sublayer_level_idc: Vec<Option<u8>>,
    /// `general_sub_profile_idc[j]`.
    pub general_sub_profile_idc: Vec<u32>,
    /// `max_picture_width` = `sps_pic_width_max_in_luma_samples`.
    pub max_picture_width: u16,
    /// `max_picture_height` = `sps_pic_height_max_in_luma_samples`.
    pub max_picture_height: u16,
    /// `avg_frame_rate` in frames per 256 seconds (0 = unspecified).
    pub avg_frame_rate: u16,
}

impl VvccPtl {
    /// Fill the block from a parsed SPS (`ols_idx = 0`,
    /// `constant_frame_rate = 0`, `avg_frame_rate = 0`). `sps_rbsp`
    /// is the SPS RBSP the SPS was parsed from — the raw
    /// `general_constraints_info()` bytes are lifted from it (they
    /// start at bit 34 of every SPS: four fixed bytes of ids / format
    /// / CTU size / profile / level, then the two PTL flags). An SPS
    /// without a `profile_tier_level()`
    /// (`sps_ptl_dpb_hrd_params_present_flag = 0`) is refused — use
    /// [`Self::from_parameter_sets`] with the VPS PTL then.
    pub fn from_sps(sps: &SeqParameterSet, sps_rbsp: &[u8]) -> Result<Self> {
        Self::from_parameter_sets(sps, sps_rbsp, None)
    }

    /// As [`Self::from_sps`], falling back to `vps_ptl` (the VPS's
    /// `profile_tier_level()` for the output layer set) when the SPS
    /// carries none. A VPS-sourced block carries an all-zero
    /// `general_constraint_info` (one byte): a zero bit claims no
    /// constraint, which §11.2.4.2.3 always permits.
    pub fn from_parameter_sets(
        sps: &SeqParameterSet,
        sps_rbsp: &[u8],
        vps_ptl: Option<&ProfileTierLevel>,
    ) -> Result<Self> {
        let (ptl, general_constraint_info) = match (sps.profile_tier_level.as_ref(), vps_ptl) {
            (Some(p), _) => (p, gci_bytes_from_sps_rbsp(sps_rbsp)?),
            (None, Some(p)) => (p, vec![0u8]),
            (None, None) => {
                return Err(Error::invalid(
                    "vvcC: neither the SPS nor a VPS carries a profile_tier_level()",
                ))
            }
        };
        let n = usize::from(sps.sps_max_sublayers_minus1) + 1;
        // `sublayer_level_idc[i]` is parsed for i = 0 .. n − 2 in the
        // SPS (bit order n − 2 .. 0); map present flags across.
        let mut sublayer_level_idc = vec![None; n.saturating_sub(1)];
        for (i, slot) in sublayer_level_idc.iter_mut().enumerate() {
            if ptl.sublayer_level_present.get(i).copied().unwrap_or(false) {
                *slot = ptl.sublayer_level_idc.get(i).copied();
            }
        }
        Ok(Self {
            ols_idx: 0,
            num_sublayers: n as u8,
            constant_frame_rate: 0,
            chroma_format_idc: sps.sps_chroma_format_idc,
            bit_depth_minus8: sps.sps_bitdepth_minus8 as u8,
            general_profile_idc: ptl.general_profile_idc,
            general_tier_flag: ptl.general_tier_flag,
            general_level_idc: ptl.general_level_idc,
            ptl_frame_only_constraint_flag: ptl.ptl_frame_only_constraint_flag,
            ptl_multilayer_enabled_flag: ptl.ptl_multilayer_enabled_flag,
            general_constraint_info,
            sublayer_level_idc,
            general_sub_profile_idc: ptl.general_sub_profile_idc.clone(),
            max_picture_width: sps.sps_pic_width_max_in_luma_samples.min(0xFFFF) as u16,
            max_picture_height: sps.sps_pic_height_max_in_luma_samples.min(0xFFFF) as u16,
            avg_frame_rate: 0,
        })
    }
}

/// The raw `general_constraints_info()` bytes of an SPS RBSP (see
/// [`VvccPtl::general_constraint_info`]).
fn gci_bytes_from_sps_rbsp(rbsp: &[u8]) -> Result<Vec<u8>> {
    // §7.3.2.4: sps_seq_parameter_set_id u(4), sps_video_parameter_set_id
    // u(4), sps_max_sublayers_minus1 u(3), sps_chroma_format_idc u(2),
    // sps_log2_ctu_size_minus5 u(2), sps_ptl_dpb_hrd_params_present_flag
    // u(1) = 16 bits; then profile_tier_level(1, …): general_profile_idc
    // u(7), general_tier_flag u(1), general_level_idc u(8),
    // ptl_frame_only_constraint_flag u(1), ptl_multilayer_enabled_flag
    // u(1) = 18 bits → general_constraints_info() starts at bit 34.
    let mut br = BitReader::new(rbsp);
    br.skip(16)?;
    let present = br.u1()? == 1;
    if !present {
        return Err(Error::invalid(
            "vvcC: the SPS carries no profile_tier_level() (sps_ptl_dpb_hrd_params_present_flag = 0)",
        ));
    }
    br.skip(17)?;
    debug_assert_eq!(br.bit_position(), 34);
    crate::ptl::skip_general_constraints_info(&mut br)?;
    let end = br.bit_position();
    debug_assert_eq!(end % 8, 0);
    let end_byte = (end / 8) as usize;
    let mut bytes = rbsp
        .get(4..end_byte)
        .ok_or_else(|| Error::invalid("vvcC: SPS truncated inside general_constraints_info()"))?
        .to_vec();
    if let Some(first) = bytes.first_mut() {
        *first &= 0x3F;
    }
    Ok(bytes)
}

/// `VvcDecoderConfigurationRecord` (§11.2.4.2).
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct VvcDecoderConfigurationRecord {
    /// `LengthSizeMinusOne` — the sample NAL length-prefix size minus
    /// one (0, 1 or 3).
    pub length_size_minus_one: u8,
    /// The `ptl_present_flag = 1` block, when present.
    pub ptl: Option<VvccPtl>,
    /// The NAL unit arrays, in record order.
    pub arrays: Vec<NalArray>,
}

impl VvcDecoderConfigurationRecord {
    /// An empty record with 4-byte length prefixes and no arrays.
    pub fn new() -> Self {
        Self {
            length_size_minus_one: 3,
            ptl: None,
            arrays: Vec::new(),
        }
    }

    /// Set the sample NAL length-prefix size (1, 2 or 4 bytes).
    pub fn with_length_size(mut self, length_size: u8) -> Result<Self> {
        self.length_size_minus_one = match length_size {
            1 => 0,
            2 => 1,
            4 => 3,
            other => {
                return Err(Error::invalid(format!(
                    "vvcC: length size must be 1, 2 or 4 bytes, got {other}"
                )))
            }
        };
        Ok(self)
    }

    /// The sample NAL length-prefix size in bytes.
    pub fn length_size(&self) -> u8 {
        self.length_size_minus_one + 1
    }

    /// Build the record from an Annex B elementary stream: every DCI,
    /// OPI, VPS, SPS, PPS and prefix APS NAL unit goes into its array
    /// (duplicates by byte identity collapse to one; the arrays are
    /// `array_completeness = 1` as a `vvc1` sample entry requires),
    /// and the first SPS fills the PTL block. Prefix SEI units stay
    /// in the samples (they are rarely declarative).
    pub fn from_annex_b(annex_b: &[u8]) -> Result<Self> {
        let mut rec = Self::new();
        let mut first_sps: Option<(SeqParameterSet, Vec<u8>)> = None;
        let mut vps_ptl: Option<ProfileTierLevel> = None;
        for nal in iter_annex_b(annex_b) {
            let t = nal.header.nal_unit_type;
            if !matches!(
                t,
                NalUnitType::DciNut
                    | NalUnitType::OpiNut
                    | NalUnitType::VpsNut
                    | NalUnitType::SpsNut
                    | NalUnitType::PpsNut
                    | NalUnitType::PrefixApsNut
            ) {
                continue;
            }
            if t == NalUnitType::SpsNut && first_sps.is_none() {
                let rbsp = extract_rbsp(nal.payload());
                let sps = parse_sps(&rbsp)?;
                first_sps = Some((sps, rbsp));
            }
            if t == NalUnitType::VpsNut && vps_ptl.is_none() {
                let vps = parse_vps(&extract_rbsp(nal.payload()))?;
                vps_ptl = vps.profile_tier_levels.first().cloned();
            }
            rec.push_nal_unit(t.as_u8(), nal.raw);
        }
        let (sps, rbsp) =
            first_sps.ok_or_else(|| Error::invalid("vvcC: the stream carries no SPS NAL unit"))?;
        rec.ptl = Some(VvccPtl::from_parameter_sets(&sps, &rbsp, vps_ptl.as_ref())?);
        Ok(rec)
    }

    /// Append `raw` (a complete NAL unit) to the array of its type,
    /// creating the array (`array_completeness = 1`) on first use;
    /// a byte-identical unit already in the array is not added again.
    /// DCI / OPI arrays hold exactly one unit (the record has no
    /// count field for them) — a second one replaces the first.
    pub fn push_nal_unit(&mut self, nal_unit_type: u8, raw: &[u8]) {
        let single = nal_unit_type == NalUnitType::DciNut.as_u8()
            || nal_unit_type == NalUnitType::OpiNut.as_u8();
        if let Some(arr) = self
            .arrays
            .iter_mut()
            .find(|a| a.nal_unit_type == nal_unit_type)
        {
            if single {
                arr.nal_units = vec![raw.to_vec()];
            } else if !arr.nal_units.iter().any(|u| u == raw) {
                arr.nal_units.push(raw.to_vec());
            }
        } else {
            self.arrays
                .push(NalArray::new(nal_unit_type, true, vec![raw.to_vec()]));
            self.sort_arrays();
        }
    }

    /// §11.2.4.2 recommended array order: DCI, OPI, VPS, SPS, PPS,
    /// prefix APS, prefix SEI (others after, by type).
    fn sort_arrays(&mut self) {
        fn rank(t: u8) -> u8 {
            match NalUnitType::from_u8(t) {
                NalUnitType::DciNut => 0,
                NalUnitType::OpiNut => 1,
                NalUnitType::VpsNut => 2,
                NalUnitType::SpsNut => 3,
                NalUnitType::PpsNut => 4,
                NalUnitType::PrefixApsNut => 5,
                NalUnitType::PrefixSeiNut => 6,
                _ => 7,
            }
        }
        self.arrays
            .sort_by_key(|a| (rank(a.nal_unit_type), a.nal_unit_type));
    }

    /// Every NAL unit of every array, in record order (what a decoder
    /// pushes before the first sample).
    pub fn nal_units(&self) -> impl Iterator<Item = &[u8]> {
        self.arrays
            .iter()
            .flat_map(|a| a.nal_units.iter().map(Vec::as_slice))
    }

    /// The units of one `nal_unit_type` array, if present.
    pub fn nal_units_of_type(&self, nal_unit_type: u8) -> &[Vec<u8>] {
        self.arrays
            .iter()
            .find(|a| a.nal_unit_type == nal_unit_type)
            .map(|a| a.nal_units.as_slice())
            .unwrap_or(&[])
    }

    /// Serialise (§11.2.4.2.2).
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut w = BitWriter::default();
        w.bits(0b11111, 5);
        w.bits(u32::from(self.length_size_minus_one & 3), 2);
        w.bits(u32::from(self.ptl.is_some()), 1);
        if let Some(p) = &self.ptl {
            w.bits(u32::from(p.ols_idx), 9);
            w.bits(u32::from(p.num_sublayers), 3);
            w.bits(u32::from(p.constant_frame_rate), 2);
            w.bits(u32::from(p.chroma_format_idc), 2);
            w.bits(u32::from(p.bit_depth_minus8), 3);
            w.bits(0b11111, 5);
            // VvcPTLRecord(num_sublayers)
            let nb = p.general_constraint_info.len();
            if nb == 0 || nb > 63 {
                return Err(Error::invalid(format!(
                    "vvcC: num_bytes_constraint_info must be 1..=63, got {nb}"
                )));
            }
            w.bits(0, 2);
            w.bits(nb as u32, 6);
            w.bits(u32::from(p.general_profile_idc), 7);
            w.bits(u32::from(p.general_tier_flag), 1);
            w.bits(u32::from(p.general_level_idc), 8);
            w.bits(u32::from(p.ptl_frame_only_constraint_flag), 1);
            w.bits(u32::from(p.ptl_multilayer_enabled_flag), 1);
            for (i, b) in p.general_constraint_info.iter().enumerate() {
                if i == 0 {
                    w.bits(u32::from(b & 0x3F), 6);
                } else {
                    w.bits(u32::from(*b), 8);
                }
            }
            let ns = usize::from(p.num_sublayers);
            if ns > 1 {
                for i in (0..ns - 1).rev() {
                    let present = p.sublayer_level_idc.get(i).copied().flatten().is_some();
                    w.bits(u32::from(present), 1);
                }
                for _ in ns..=8 {
                    w.bits(0, 1);
                }
                for i in (0..ns - 1).rev() {
                    if let Some(idc) = p.sublayer_level_idc.get(i).copied().flatten() {
                        w.bits(u32::from(idc), 8);
                    }
                }
            }
            w.bits(p.general_sub_profile_idc.len() as u32, 8);
            for &sp in &p.general_sub_profile_idc {
                w.bits(sp, 32);
            }
            w.bits(u32::from(p.max_picture_width), 16);
            w.bits(u32::from(p.max_picture_height), 16);
            w.bits(u32::from(p.avg_frame_rate), 16);
        }
        if self.arrays.len() > 255 {
            return Err(Error::invalid("vvcC: more than 255 NAL unit arrays"));
        }
        w.bits(self.arrays.len() as u32, 8);
        for a in &self.arrays {
            w.bits(u32::from(a.array_completeness), 1);
            w.bits(0, 2);
            w.bits(u32::from(a.nal_unit_type & 0x1F), 5);
            let single = a.nal_unit_type == NalUnitType::DciNut.as_u8()
                || a.nal_unit_type == NalUnitType::OpiNut.as_u8();
            if single {
                if a.nal_units.len() != 1 {
                    return Err(Error::invalid(
                        "vvcC: a DCI / OPI array holds exactly one NAL unit",
                    ));
                }
            } else {
                if a.nal_units.len() > 0xFFFF {
                    return Err(Error::invalid(
                        "vvcC: more than 65535 NAL units in an array",
                    ));
                }
                w.bits(a.nal_units.len() as u32, 16);
            }
            for u in &a.nal_units {
                if u.len() > 0xFFFF {
                    return Err(Error::invalid("vvcC: NAL unit longer than 65535 bytes"));
                }
                w.bits(u.len() as u32, 16);
                for &b in u {
                    w.bits(u32::from(b), 8);
                }
            }
        }
        Ok(w.into_bytes())
    }

    /// Parse a record (§11.2.4.2.2). Arrays of reserved / unpermitted
    /// NAL unit types are kept verbatim (readers "should ignore"
    /// them); a truncated record is an error.
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        let mut br = BitReader::new(bytes);
        let e = |what: &str| Error::invalid(format!("vvcC: truncated inside {what}"));
        br.u(5).map_err(|_| e("the header"))?;
        let length_size_minus_one = br.u(2).map_err(|_| e("the header"))? as u8;
        if length_size_minus_one == 2 {
            return Err(Error::invalid(
                "vvcC: LengthSizeMinusOne = 2 (3-byte lengths) is not a permitted value",
            ));
        }
        let ptl_present = br.u1().map_err(|_| e("the header"))? == 1;
        let ptl = if ptl_present {
            let ols_idx = br.u(9).map_err(|_| e("the PTL block"))? as u16;
            let num_sublayers = br.u(3).map_err(|_| e("the PTL block"))? as u8;
            let constant_frame_rate = br.u(2).map_err(|_| e("the PTL block"))? as u8;
            let chroma_format_idc = br.u(2).map_err(|_| e("the PTL block"))? as u8;
            let bit_depth_minus8 = br.u(3).map_err(|_| e("the PTL block"))? as u8;
            br.u(5).map_err(|_| e("the PTL block"))?;
            br.u(2).map_err(|_| e("VvcPTLRecord"))?;
            let nb = br.u(6).map_err(|_| e("VvcPTLRecord"))? as usize;
            if nb == 0 {
                return Err(Error::invalid(
                    "vvcC: num_bytes_constraint_info shall be greater than 0",
                ));
            }
            let general_profile_idc = br.u(7).map_err(|_| e("VvcPTLRecord"))? as u8;
            let general_tier_flag = br.u1().map_err(|_| e("VvcPTLRecord"))? == 1;
            let general_level_idc = br.u(8).map_err(|_| e("VvcPTLRecord"))? as u8;
            let ptl_frame_only_constraint_flag = br.u1().map_err(|_| e("VvcPTLRecord"))? == 1;
            let ptl_multilayer_enabled_flag = br.u1().map_err(|_| e("VvcPTLRecord"))? == 1;
            let mut general_constraint_info = Vec::with_capacity(nb);
            general_constraint_info.push(br.u(6).map_err(|_| e("general_constraint_info"))? as u8);
            for _ in 1..nb {
                general_constraint_info
                    .push(br.u(8).map_err(|_| e("general_constraint_info"))? as u8);
            }
            let ns = usize::from(num_sublayers);
            let mut present = vec![false; ns.saturating_sub(1)];
            if ns > 1 {
                for i in (0..ns - 1).rev() {
                    present[i] = br.u1().map_err(|_| e("sublayer flags"))? == 1;
                }
                for _ in ns..=8 {
                    br.u1().map_err(|_| e("sublayer flags"))?;
                }
            }
            let mut sublayer_level_idc = vec![None; ns.saturating_sub(1)];
            if ns > 1 {
                for i in (0..ns - 1).rev() {
                    if present[i] {
                        sublayer_level_idc[i] =
                            Some(br.u(8).map_err(|_| e("sublayer_level_idc"))? as u8);
                    }
                }
            }
            let nsp = br.u(8).map_err(|_| e("ptl_num_sub_profiles"))? as usize;
            let mut general_sub_profile_idc = Vec::with_capacity(nsp);
            for _ in 0..nsp {
                general_sub_profile_idc.push(br.u(32).map_err(|_| e("general_sub_profile_idc"))?);
            }
            let max_picture_width = br.u(16).map_err(|_| e("max_picture_width"))? as u16;
            let max_picture_height = br.u(16).map_err(|_| e("max_picture_height"))? as u16;
            let avg_frame_rate = br.u(16).map_err(|_| e("avg_frame_rate"))? as u16;
            Some(VvccPtl {
                ols_idx,
                num_sublayers,
                constant_frame_rate,
                chroma_format_idc,
                bit_depth_minus8,
                general_profile_idc,
                general_tier_flag,
                general_level_idc,
                ptl_frame_only_constraint_flag,
                ptl_multilayer_enabled_flag,
                general_constraint_info,
                sublayer_level_idc,
                general_sub_profile_idc,
                max_picture_width,
                max_picture_height,
                avg_frame_rate,
            })
        } else {
            None
        };
        let num_arrays = br.u(8).map_err(|_| e("num_of_arrays"))? as usize;
        let mut arrays = Vec::with_capacity(num_arrays.min(16));
        for _ in 0..num_arrays {
            let array_completeness = br.u1().map_err(|_| e("an array header"))? == 1;
            br.u(2).map_err(|_| e("an array header"))?;
            let nal_unit_type = br.u(5).map_err(|_| e("an array header"))? as u8;
            let single = nal_unit_type == NalUnitType::DciNut.as_u8()
                || nal_unit_type == NalUnitType::OpiNut.as_u8();
            let num_nalus = if single {
                1
            } else {
                br.u(16).map_err(|_| e("num_nalus"))? as usize
            };
            let mut nal_units = Vec::with_capacity(num_nalus.min(64));
            for _ in 0..num_nalus {
                let len = br.u(16).map_err(|_| e("nal_unit_length"))? as usize;
                debug_assert!(br.is_byte_aligned());
                let at = (br.bit_position() / 8) as usize;
                let unit = bytes
                    .get(at..at + len)
                    .ok_or_else(|| e("a NAL unit"))?
                    .to_vec();
                br.skip((len * 8) as u32).map_err(|_| e("a NAL unit"))?;
                if len < 2 {
                    return Err(Error::invalid(
                        "vvcC: a NAL unit shorter than its 2-byte header",
                    ));
                }
                NalHeader::parse(&unit)?;
                nal_units.push(unit);
            }
            arrays.push(NalArray {
                array_completeness,
                nal_unit_type,
                nal_units,
            });
        }
        Ok(Self {
            length_size_minus_one,
            ptl,
            arrays,
        })
    }

    /// The first SPS of the record, parsed (what a decoder needs to
    /// label its output before the first sample).
    pub fn parsed_sps(&self) -> Result<Option<SeqParameterSet>> {
        match self.nal_units_of_type(NalUnitType::SpsNut.as_u8()).first() {
            Some(raw) => {
                let payload = raw.get(2..).unwrap_or(&[]);
                Ok(Some(parse_sps(&extract_rbsp(payload))?))
            }
            None => Ok(None),
        }
    }
}

impl Default for VvcDecoderConfigurationRecord {
    fn default() -> Self {
        Self::new()
    }
}

/// Re-frame an Annex B access unit as §4.2.3 length-prefixed sample
/// data: every NAL unit gets a big-endian `length_size`-byte length.
/// With `strip_parameter_sets` the DCI / OPI / VPS / SPS / PPS units
/// are dropped (they live in the record — the `vvc1` shape); every
/// other unit, AUD included, is kept.
pub fn annex_b_to_length_prefixed(
    annex_b: &[u8],
    length_size: u8,
    strip_parameter_sets: bool,
) -> Result<Vec<u8>> {
    if !matches!(length_size, 1 | 2 | 4) {
        return Err(Error::invalid(format!(
            "vvcC: length size must be 1, 2 or 4 bytes, got {length_size}"
        )));
    }
    let mut out = Vec::with_capacity(annex_b.len() + 16);
    for nal in iter_annex_b(annex_b) {
        let t = nal.header.nal_unit_type;
        if strip_parameter_sets
            && matches!(
                t,
                NalUnitType::DciNut
                    | NalUnitType::OpiNut
                    | NalUnitType::VpsNut
                    | NalUnitType::SpsNut
                    | NalUnitType::PpsNut
            )
        {
            continue;
        }
        let len = nal.raw.len();
        let max = match length_size {
            1 => 0xFF,
            2 => 0xFFFF,
            _ => u32::MAX as usize,
        };
        if len > max {
            return Err(Error::invalid(format!(
                "vvcC: NAL unit of {len} bytes does not fit a {length_size}-byte length prefix"
            )));
        }
        let be = (len as u32).to_be_bytes();
        out.extend_from_slice(&be[4 - usize::from(length_size)..]);
        out.extend_from_slice(nal.raw);
    }
    Ok(out)
}

/// MSB-first bit packer (local, to keep this module free of the
/// encoder's `BitWriter` exp-Golomb surface).
#[derive(Default)]
struct BitWriter {
    bytes: Vec<u8>,
    acc: u64,
    n: u32,
}

impl BitWriter {
    fn bits(&mut self, v: u32, n: u32) {
        debug_assert!(n <= 32);
        let mask = if n == 32 {
            u64::MAX >> 32
        } else {
            (1u64 << n) - 1
        };
        self.acc = (self.acc << n) | (u64::from(v) & mask);
        self.n += n;
        while self.n >= 8 {
            self.n -= 8;
            self.bytes.push((self.acc >> self.n) as u8);
        }
        self.acc &= (1u64 << self.n) - 1;
    }

    fn into_bytes(mut self) -> Vec<u8> {
        if self.n > 0 {
            let pad = 8 - self.n;
            self.bits(0, pad);
        }
        self.bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_stream() -> Vec<u8> {
        let src = crate::reconstruct::PictureBuffer::yuv420_filled(32, 24, 77);
        crate::encoder_pipeline::encode_idr_with_residuals(&src, 30)
            .expect("encode")
            .0
    }

    #[test]
    fn record_round_trips_through_bytes() {
        let bs = sample_stream();
        let rec = VvcDecoderConfigurationRecord::from_annex_b(&bs).expect("build");
        let p = rec.ptl.as_ref().expect("ptl");
        assert_eq!(p.chroma_format_idc, 1);
        assert_eq!(p.bit_depth_minus8, 0);
        assert_eq!(p.max_picture_width, 32);
        assert_eq!(p.max_picture_height, 24);
        assert_eq!(p.general_constraint_info, vec![0]);
        let types: Vec<u8> = rec.arrays.iter().map(|a| a.nal_unit_type).collect();
        assert_eq!(
            types,
            vec![
                NalUnitType::VpsNut.as_u8(),
                NalUnitType::SpsNut.as_u8(),
                NalUnitType::PpsNut.as_u8()
            ]
        );
        let bytes = rec.to_bytes().expect("serialise");
        assert_eq!(bytes[0] >> 3, 0b11111);
        assert_eq!(bytes[0] & 0x07, 0b111); // LengthSizeMinusOne = 3, ptl_present
        let back = VvcDecoderConfigurationRecord::parse(&bytes).expect("parse");
        assert_eq!(back, rec);
        assert_eq!(back.length_size(), 4);
        let sps = back.parsed_sps().expect("sps parses").expect("sps present");
        assert_eq!(sps.cropped_width(), 32);
    }

    #[test]
    fn length_prefixed_samples_carry_every_non_parameter_set_nal() {
        let bs = sample_stream();
        let sample = annex_b_to_length_prefixed(&bs, 4, true).expect("frame");
        let nals = crate::nal::iter_length_prefixed(&sample, 4).expect("walk");
        let types: Vec<_> = nals.iter().map(|n| n.header.nal_unit_type).collect();
        assert!(!types.contains(&NalUnitType::SpsNut));
        assert!(types.contains(&NalUnitType::IdrNLp));
        let full = annex_b_to_length_prefixed(&bs, 2, false).expect("frame");
        let nals = crate::nal::iter_length_prefixed(&full, 2).expect("walk");
        assert!(nals
            .iter()
            .any(|n| n.header.nal_unit_type == NalUnitType::SpsNut));
        assert!(annex_b_to_length_prefixed(&bs, 3, false).is_err());
    }

    #[test]
    fn truncated_and_hostile_records_are_typed_errors() {
        let bs = sample_stream();
        let bytes = VvcDecoderConfigurationRecord::from_annex_b(&bs)
            .unwrap()
            .to_bytes()
            .unwrap();
        for cut in 0..bytes.len() {
            assert!(VvcDecoderConfigurationRecord::parse(&bytes[..cut]).is_err());
        }
        // LengthSizeMinusOne = 2 is not permitted.
        let mut bad = bytes.clone();
        bad[0] = (bad[0] & !0x06) | 0x04;
        assert!(VvcDecoderConfigurationRecord::parse(&bad).is_err());
        // A no-PTL record with zero arrays is the minimal legal shape.
        let min = VvcDecoderConfigurationRecord::parse(&[0xFE, 0]).unwrap();
        assert!(min.ptl.is_none());
        assert!(min.arrays.is_empty());
        assert_eq!(min.length_size(), 4);
    }

    #[test]
    fn bit_writer_packs_msb_first() {
        let mut w = BitWriter::default();
        w.bits(0b101, 3);
        w.bits(0b11111, 5);
        w.bits(0xABCD, 16);
        w.bits(1, 1);
        assert_eq!(w.into_bytes(), vec![0xBF, 0xAB, 0xCD, 0x80]);
    }
}
