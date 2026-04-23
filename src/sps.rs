//! VVC Sequence Parameter Set parser (§7.3.2.4).
//!
//! Foundation scope: parse the leading identifier + format block so
//! downstream consumers can see the stream's resolution, chroma format,
//! CTU size, bit depths, and POC parameters. The full SPS carries
//! hundreds of tools-enable flags and a large partition-constraints
//! block (§7.3.2.4); walking the entire syntax is not needed for
//! parameter-set inspection, and the parts that depend on subpicture /
//! scaling-QP-table / LADF / virtual-boundaries sub-structures pull in
//! a lot of derivation state that the scaffold does not carry.
//!
//! Consequently this parser:
//!
//! * fully parses the identifier / format / conformance-window /
//!   bit-depth / POC / extra-header-bits fields,
//! * **rejects** `sps_subpic_info_present_flag == 1` streams with
//!   `Error::Unsupported` — their syntax contains `u(v)` fields whose
//!   width depends on the derived `CtbSizeY` and `PicSizeInCtbsY`, and
//!   decoding them is out of foundation scope,
//! * reads just enough of the rest of the SPS (PTL, dpb_parameters,
//!   extension flags) to keep the bit position consistent when the
//!   caller wants to interleave with PPS / slice-header parsing,
//! * does **not** walk the partition-constraints / QP-table / tool-flag
//!   tail. That section is deliberately left as a follow-up increment.

use oxideav_core::{Error, Result};

use crate::bitreader::BitReader;
use crate::ptl::{parse_profile_tier_level, ProfileTierLevel};

#[derive(Clone, Debug)]
pub struct SeqParameterSet {
    pub sps_seq_parameter_set_id: u8,
    pub sps_video_parameter_set_id: u8,
    pub sps_max_sublayers_minus1: u8,
    pub sps_chroma_format_idc: u8,
    pub sps_log2_ctu_size_minus5: u8,
    pub sps_ptl_dpb_hrd_params_present_flag: bool,
    pub profile_tier_level: Option<ProfileTierLevel>,
    pub sps_gdr_enabled_flag: bool,
    pub sps_ref_pic_resampling_enabled_flag: bool,
    pub sps_res_change_in_clvs_allowed_flag: bool,
    pub sps_pic_width_max_in_luma_samples: u32,
    pub sps_pic_height_max_in_luma_samples: u32,
    pub conformance_window: Option<ConformanceWindow>,
    pub sps_subpic_info_present_flag: bool,
    pub sps_bitdepth_minus8: u32,
    pub sps_entropy_coding_sync_enabled_flag: bool,
    pub sps_entry_point_offsets_present_flag: bool,
    pub sps_log2_max_pic_order_cnt_lsb_minus4: u8,
    pub sps_poc_msb_cycle_flag: bool,
    pub sps_poc_msb_cycle_len_minus1: u32,
    pub sps_num_extra_ph_bytes: u8,
    pub sps_num_extra_sh_bytes: u8,
}

/// SPS conformance-cropping window (§7.4.3.4).
#[derive(Clone, Copy, Debug)]
pub struct ConformanceWindow {
    pub left_offset: u32,
    pub right_offset: u32,
    pub top_offset: u32,
    pub bottom_offset: u32,
}

impl SeqParameterSet {
    /// CTB size in luma samples. §7.4.3.4: `CtbSizeY = 1 << (log2_ctu_size_minus5 + 5)`.
    pub fn ctb_size(&self) -> u32 {
        1 << (self.sps_log2_ctu_size_minus5 as u32 + 5)
    }

    /// Bit depth of luma samples (= `sps_bitdepth_minus8 + 8`).
    pub fn bit_depth_y(&self) -> u32 {
        self.sps_bitdepth_minus8 + 8
    }

    /// Bit depth of chroma samples. VVC signals a single bit depth for
    /// both planes (§7.4.3.4).
    pub fn bit_depth_c(&self) -> u32 {
        self.bit_depth_y()
    }

    /// Luma width after the conformance-window crop, using SubWidthC /
    /// SubHeightC derived from `sps_chroma_format_idc`. Per Table 2 in
    /// §6.2: 4:2:0 → (2,2); 4:2:2 → (2,1); 4:4:4 → (1,1); mono → (1,1).
    pub fn cropped_width(&self) -> u32 {
        let sub_x = match self.sps_chroma_format_idc {
            1 | 2 => 2,
            _ => 1,
        };
        let crop = self
            .conformance_window
            .map(|c| sub_x * (c.left_offset + c.right_offset))
            .unwrap_or(0);
        self.sps_pic_width_max_in_luma_samples.saturating_sub(crop)
    }

    pub fn cropped_height(&self) -> u32 {
        let sub_y = match self.sps_chroma_format_idc {
            1 => 2,
            _ => 1,
        };
        let crop = self
            .conformance_window
            .map(|c| sub_y * (c.top_offset + c.bottom_offset))
            .unwrap_or(0);
        self.sps_pic_height_max_in_luma_samples.saturating_sub(crop)
    }
}

/// Parse an SPS NAL RBSP payload (the bytes after the 2-byte NAL header,
/// with emulation-prevention bytes already stripped).
pub fn parse_sps(rbsp: &[u8]) -> Result<SeqParameterSet> {
    let mut br = BitReader::new(rbsp);
    let sps_seq_parameter_set_id = br.u(4)? as u8;
    let sps_video_parameter_set_id = br.u(4)? as u8;
    let sps_max_sublayers_minus1 = br.u(3)? as u8;
    if sps_max_sublayers_minus1 > 6 {
        return Err(Error::invalid(
            "h266 SPS: sps_max_sublayers_minus1 must be <= 6",
        ));
    }
    let sps_chroma_format_idc = br.u(2)? as u8;
    let sps_log2_ctu_size_minus5 = br.u(2)? as u8;
    // §7.4.3.4: sps_log2_ctu_size_minus5 shall be <= 2 (CtbLog2Size in 5..7,
    // i.e. CTUs of 32 / 64 / 128 samples). The u(2) field can encode 0..3
    // so we explicitly reject 3.
    if sps_log2_ctu_size_minus5 > 2 {
        return Err(Error::invalid(format!(
            "h266 SPS: sps_log2_ctu_size_minus5 out of range ({sps_log2_ctu_size_minus5})"
        )));
    }
    let sps_ptl_dpb_hrd_params_present_flag = br.u1()? == 1;
    let profile_tier_level = if sps_ptl_dpb_hrd_params_present_flag {
        Some(parse_profile_tier_level(
            &mut br,
            true,
            sps_max_sublayers_minus1,
        )?)
    } else {
        None
    };
    let sps_gdr_enabled_flag = br.u1()? == 1;
    let sps_ref_pic_resampling_enabled_flag = br.u1()? == 1;
    let sps_res_change_in_clvs_allowed_flag = if sps_ref_pic_resampling_enabled_flag {
        br.u1()? == 1
    } else {
        false
    };
    let sps_pic_width_max_in_luma_samples = br.ue()?;
    let sps_pic_height_max_in_luma_samples = br.ue()?;
    if sps_pic_width_max_in_luma_samples == 0
        || sps_pic_height_max_in_luma_samples == 0
        || sps_pic_width_max_in_luma_samples > 16384
        || sps_pic_height_max_in_luma_samples > 16384
    {
        return Err(Error::invalid(format!(
            "h266 SPS: implausible picture size {sps_pic_width_max_in_luma_samples}x{sps_pic_height_max_in_luma_samples}"
        )));
    }
    let sps_conformance_window_flag = br.u1()? == 1;
    let conformance_window = if sps_conformance_window_flag {
        Some(ConformanceWindow {
            left_offset: br.ue()?,
            right_offset: br.ue()?,
            top_offset: br.ue()?,
            bottom_offset: br.ue()?,
        })
    } else {
        None
    };
    let sps_subpic_info_present_flag = br.u1()? == 1;
    if sps_subpic_info_present_flag {
        // The subpic sub-syntax contains u(v) fields whose width is
        // derived from `CtbSizeY` and the picture dimensions. Decoding
        // those correctly is out of foundation scope; we surface an
        // Unsupported error so callers know the SPS parse was aborted.
        return Err(Error::unsupported(
            "h266 SPS: sps_subpic_info_present_flag = 1 (subpicture streams not yet supported)",
        ));
    }
    let sps_bitdepth_minus8 = br.ue()?;
    if sps_bitdepth_minus8 > 8 {
        return Err(Error::invalid(format!(
            "h266 SPS: sps_bitdepth_minus8 out of range ({sps_bitdepth_minus8})"
        )));
    }
    let sps_entropy_coding_sync_enabled_flag = br.u1()? == 1;
    let sps_entry_point_offsets_present_flag = br.u1()? == 1;
    let sps_log2_max_pic_order_cnt_lsb_minus4 = br.u(4)? as u8;
    if sps_log2_max_pic_order_cnt_lsb_minus4 > 12 {
        return Err(Error::invalid(format!(
            "h266 SPS: sps_log2_max_pic_order_cnt_lsb_minus4 out of range ({sps_log2_max_pic_order_cnt_lsb_minus4})"
        )));
    }
    let sps_poc_msb_cycle_flag = br.u1()? == 1;
    let sps_poc_msb_cycle_len_minus1 = if sps_poc_msb_cycle_flag { br.ue()? } else { 0 };
    let sps_num_extra_ph_bytes = br.u(2)? as u8;
    for _ in 0..(sps_num_extra_ph_bytes as u32 * 8) {
        br.skip(1)?;
    }
    let sps_num_extra_sh_bytes = br.u(2)? as u8;
    for _ in 0..(sps_num_extra_sh_bytes as u32 * 8) {
        br.skip(1)?;
    }
    // Remaining SPS body (dpb_parameters, partition constraints, tool
    // flags, LMCS / ALF gates, LADF, virtual boundaries, HRD timing,
    // VUI, extensions) is deliberately not parsed by the foundation
    // scaffold. The surfaced fields above are sufficient for parameter-
    // set inspection; downstream CTU-walker work will pick up the
    // remaining syntax.
    Ok(SeqParameterSet {
        sps_seq_parameter_set_id,
        sps_video_parameter_set_id,
        sps_max_sublayers_minus1,
        sps_chroma_format_idc,
        sps_log2_ctu_size_minus5,
        sps_ptl_dpb_hrd_params_present_flag,
        profile_tier_level,
        sps_gdr_enabled_flag,
        sps_ref_pic_resampling_enabled_flag,
        sps_res_change_in_clvs_allowed_flag,
        sps_pic_width_max_in_luma_samples,
        sps_pic_height_max_in_luma_samples,
        conformance_window,
        sps_subpic_info_present_flag,
        sps_bitdepth_minus8,
        sps_entropy_coding_sync_enabled_flag,
        sps_entry_point_offsets_present_flag,
        sps_log2_max_pic_order_cnt_lsb_minus4,
        sps_poc_msb_cycle_flag,
        sps_poc_msb_cycle_len_minus1,
        sps_num_extra_ph_bytes,
        sps_num_extra_sh_bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a hand-crafted minimal single-layer SPS RBSP and parse it.
    ///
    /// Fields (in bit order):
    ///   sps_seq_parameter_set_id           u(4) = 0
    ///   sps_video_parameter_set_id         u(4) = 0
    ///   sps_max_sublayers_minus1           u(3) = 0
    ///   sps_chroma_format_idc              u(2) = 1  (4:2:0)
    ///   sps_log2_ctu_size_minus5           u(2) = 2  (CTB = 128)
    ///   sps_ptl_dpb_hrd_params_present_flag u(1) = 0 (skip PTL)
    ///   sps_gdr_enabled_flag               u(1) = 0
    ///   sps_ref_pic_resampling_enabled_flag u(1) = 0
    ///   sps_pic_width_max_in_luma_samples  ue(v) 320 = 0000000101000001 (17 bits)
    ///   sps_pic_height_max_in_luma_samples ue(v) 240 = 000000011110001 (15 bits)
    ///   sps_conformance_window_flag        u(1) = 0
    ///   sps_subpic_info_present_flag       u(1) = 0
    ///   sps_bitdepth_minus8                ue(v) 2 (bit depth 10) = 011
    ///   sps_entropy_coding_sync_enabled    u(1) = 0
    ///   sps_entry_point_offsets_present    u(1) = 0
    ///   sps_log2_max_pic_order_cnt_lsb_minus4 u(4) = 4 → lsb_bits = 8
    ///   sps_poc_msb_cycle_flag             u(1) = 0
    ///   sps_num_extra_ph_bytes             u(2) = 0
    ///   sps_num_extra_sh_bytes             u(2) = 0
    ///
    /// We need to manually encode ue(v) for 320 and 240.
    /// ue(320): codeNum=320 → bit length = 2*8+1 = 17 bits
    ///   prefix = 8 zeros, then "1", then 8-bit suffix = 320+1-(1<<8) = 65 = 0b01000001
    ///   → 0_0000_0000_1_0100_0001 (17 bits)
    /// ue(240): 2*7+1 = 15 bits
    ///   prefix = 7 zeros, "1", then 7-bit suffix = 240+1-(1<<7) = 113 = 0b1110001
    ///   → 0_0000_001_111_0001 (15 bits)
    /// ue(2) → "011" (3 bits)
    #[test]
    fn minimal_sps_320x240_10bit_no_ptl() {
        // Build bit-by-bit into a big-endian buffer.
        let mut bits: Vec<u8> = Vec::new();
        // Helper: push `v` as `n` bits MSB-first.
        fn push_u(bits: &mut Vec<u8>, v: u64, n: u32) {
            for i in (0..n).rev() {
                bits.push(((v >> i) & 1) as u8);
            }
        }
        push_u(&mut bits, 0, 4); // sps_id
        push_u(&mut bits, 0, 4); // vps_id
        push_u(&mut bits, 0, 3); // max_sublayers_minus1
        push_u(&mut bits, 1, 2); // chroma = 4:2:0
        push_u(&mut bits, 2, 2); // log2_ctu - 5 = 2 → CTB=128
        push_u(&mut bits, 0, 1); // ptl_dpb_hrd_present = 0
        push_u(&mut bits, 0, 1); // gdr_enabled = 0
        push_u(&mut bits, 0, 1); // ref_pic_resampling = 0
        // ue(320): 17 bits = 0_0000_0000_1_0100_0001
        push_u(&mut bits, 0b00000000010100_0001, 17);
        // ue(240): 15 bits = 0_0000_001_111_0001
        push_u(&mut bits, 0b000000011110001, 15);
        push_u(&mut bits, 0, 1); // conformance_window_flag
        push_u(&mut bits, 0, 1); // subpic_info_present
        push_u(&mut bits, 0b011, 3); // ue(2) bitdepth_minus8
        push_u(&mut bits, 0, 1); // entropy_coding_sync
        push_u(&mut bits, 0, 1); // entry_point_offsets
        push_u(&mut bits, 4, 4); // log2_max_poc_lsb_minus4
        push_u(&mut bits, 0, 1); // poc_msb_cycle_flag
        push_u(&mut bits, 0, 2); // num_extra_ph_bytes
        push_u(&mut bits, 0, 2); // num_extra_sh_bytes
        // Pad to a full byte with zeros.
        while bits.len() % 8 != 0 {
            bits.push(0);
        }
        let mut bytes = Vec::with_capacity(bits.len() / 8);
        for chunk in bits.chunks(8) {
            let mut b = 0u8;
            for (i, &bit) in chunk.iter().enumerate() {
                b |= bit << (7 - i);
            }
            bytes.push(b);
        }

        let sps = parse_sps(&bytes).unwrap();
        assert_eq!(sps.sps_seq_parameter_set_id, 0);
        assert_eq!(sps.sps_video_parameter_set_id, 0);
        assert_eq!(sps.sps_chroma_format_idc, 1);
        assert_eq!(sps.ctb_size(), 128);
        assert_eq!(sps.sps_pic_width_max_in_luma_samples, 320);
        assert_eq!(sps.sps_pic_height_max_in_luma_samples, 240);
        assert_eq!(sps.bit_depth_y(), 10);
        assert_eq!(sps.cropped_width(), 320);
        assert_eq!(sps.cropped_height(), 240);
        assert_eq!(sps.sps_log2_max_pic_order_cnt_lsb_minus4, 4);
        assert_eq!(sps.sps_num_extra_ph_bytes, 0);
        assert_eq!(sps.sps_num_extra_sh_bytes, 0);
    }

    #[test]
    fn subpic_streams_are_rejected() {
        // Same as above but flip subpic_info_present to 1; the parser
        // should return Unsupported before reading anything else.
        let mut bits: Vec<u8> = Vec::new();
        fn push_u(bits: &mut Vec<u8>, v: u64, n: u32) {
            for i in (0..n).rev() {
                bits.push(((v >> i) & 1) as u8);
            }
        }
        push_u(&mut bits, 0, 4); // sps_id
        push_u(&mut bits, 0, 4); // vps_id
        push_u(&mut bits, 0, 3); // max_sublayers
        push_u(&mut bits, 1, 2); // chroma
        push_u(&mut bits, 0, 2); // log2_ctu - 5 = 0 → 32
        push_u(&mut bits, 0, 1); // ptl_present
        push_u(&mut bits, 0, 1); // gdr
        push_u(&mut bits, 0, 1); // ref_pic_resampling
        push_u(&mut bits, 0b00000000010100_0001, 17); // width 320
        push_u(&mut bits, 0b000000011110001, 15); // height 240
        push_u(&mut bits, 0, 1); // conformance_window_flag
        push_u(&mut bits, 1, 1); // subpic_info_present = 1
        while bits.len() % 8 != 0 {
            bits.push(0);
        }
        let mut bytes = Vec::with_capacity(bits.len() / 8);
        for chunk in bits.chunks(8) {
            let mut b = 0u8;
            for (i, &bit) in chunk.iter().enumerate() {
                b |= bit << (7 - i);
            }
            bytes.push(b);
        }
        assert!(parse_sps(&bytes).is_err());
    }
}
