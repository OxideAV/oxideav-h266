//! VVC Picture Header structure parser (§7.3.2.7 — `picture_header_rbsp()`
//! / §7.3.2.8 — `picture_header_structure()`).
//!
//! Foundation scope: parse the leading fixed fields through
//! `ph_pic_parameter_set_id` so callers can pair a PH with its PPS.
//! Everything past that point — `ph_pic_order_cnt_lsb` (u(v) width from
//! SPS), recovery POC, alf / lmcs / scaling / virtual-boundaries / RPL
//! gates, partition-constraint overrides, weighted-prediction table,
//! deblocking deltas, picture header extension — is **not** walked
//! because the widths and presence of every subsequent field depend on
//! SPS / PPS context that this parser does not receive.
//!
//! That is still useful: most PH-NUTs appear immediately after the PPS
//! in a conformance stream, so once the caller has the PPS in hand the
//! remaining bits (captured as `payload_tail`) can be rewound through a
//! stateful decoder.

use oxideav_core::{Error, Result};

use crate::bitreader::BitReader;

#[derive(Clone, Debug)]
pub struct PictureHeader {
    pub ph_gdr_or_irap_pic_flag: bool,
    pub ph_non_ref_pic_flag: bool,
    /// Present only when `ph_gdr_or_irap_pic_flag == 1`. Defaults to
    /// false when the flag was not signalled.
    pub ph_gdr_pic_flag: bool,
    pub ph_inter_slice_allowed_flag: bool,
    /// Present only when `ph_inter_slice_allowed_flag == 1`. Inferred
    /// to true (intra-only pictures) when not signalled.
    pub ph_intra_slice_allowed_flag: bool,
    pub ph_pic_parameter_set_id: u32,
    /// Remaining RBSP bytes (starting from the next bit after
    /// `ph_pic_parameter_set_id`). Foundation consumers treat this as
    /// opaque — it still needs SPS + PPS context to be parsed further.
    pub payload_tail: Vec<u8>,
    /// Bit offset within `payload_tail`'s first byte where the
    /// not-yet-parsed syntax resumes (0..=7).
    pub payload_tail_bit_offset: u8,
    /// Total number of bits consumed from the input buffer before
    /// `payload_tail` begins. Useful for callers that need to stitch
    /// this back into a larger parse (e.g. slice-header-embedded PH).
    pub consumed_bits: u64,
}

/// Parse the leading fields of a `picture_header_structure()` body
/// (§7.3.2.8). The `rbsp` slice is the content of a PH_NUT RBSP
/// (i.e. the bytes after the NAL header, with emulation-prevention
/// bytes already stripped).
pub fn parse_picture_header(rbsp: &[u8]) -> Result<PictureHeader> {
    if rbsp.is_empty() {
        return Err(Error::invalid("h266 PH: empty RBSP"));
    }
    let mut br = BitReader::new(rbsp);
    let ph_gdr_or_irap_pic_flag = br.u1()? == 1;
    let ph_non_ref_pic_flag = br.u1()? == 1;
    let ph_gdr_pic_flag = if ph_gdr_or_irap_pic_flag {
        br.u1()? == 1
    } else {
        false
    };
    let ph_inter_slice_allowed_flag = br.u1()? == 1;
    let ph_intra_slice_allowed_flag = if ph_inter_slice_allowed_flag {
        br.u1()? == 1
    } else {
        true
    };
    let ph_pic_parameter_set_id = br.ue()?;
    if ph_pic_parameter_set_id > 63 {
        return Err(Error::invalid(format!(
            "h266 PH: ph_pic_parameter_set_id out of range ({ph_pic_parameter_set_id})"
        )));
    }
    let bit_pos = br.bit_position();
    let byte_off = (bit_pos / 8) as usize;
    let bit_off = (bit_pos % 8) as u8;
    let tail = if byte_off < rbsp.len() {
        rbsp[byte_off..].to_vec()
    } else {
        Vec::new()
    };
    Ok(PictureHeader {
        ph_gdr_or_irap_pic_flag,
        ph_non_ref_pic_flag,
        ph_gdr_pic_flag,
        ph_inter_slice_allowed_flag,
        ph_intra_slice_allowed_flag,
        ph_pic_parameter_set_id,
        payload_tail: tail,
        payload_tail_bit_offset: bit_off,
        consumed_bits: bit_pos,
    })
}

/// Parse a `picture_header_rbsp()` body (§7.3.2.7). Identical to
/// `parse_picture_header`, but the name signals that the caller has
/// a PH_NUT RBSP in hand rather than a slice-embedded PH.
pub fn parse_picture_header_rbsp(rbsp: &[u8]) -> Result<PictureHeader> {
    parse_picture_header(rbsp)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// PH leading byte: ph_gdr_or_irap = 1 (IRAP), ph_non_ref = 0,
    /// ph_gdr_pic = 0 (present since gdr_or_irap=1), ph_inter_slice = 0,
    /// ph_pic_parameter_set_id = 0 (ue(v) → "1" = 1 bit).
    ///
    /// Bits: 1 0 0 0 | 1 0 0 0 = 0x88
    #[test]
    fn irap_picture_header() {
        let data = [0x88u8];
        let ph = parse_picture_header(&data).unwrap();
        assert!(ph.ph_gdr_or_irap_pic_flag);
        assert!(!ph.ph_non_ref_pic_flag);
        assert!(!ph.ph_gdr_pic_flag);
        assert!(!ph.ph_inter_slice_allowed_flag);
        // intra_slice_allowed inferred to true when inter not allowed.
        assert!(ph.ph_intra_slice_allowed_flag);
        assert_eq!(ph.ph_pic_parameter_set_id, 0);
    }

    /// Bits: 0 0 1 | 1 | 1 = 0b00111000 = 0x38
    /// ph_gdr_or_irap=0, ph_non_ref=0, ph_inter_slice=1,
    /// ph_intra_slice=1, ph_pic_parameter_set_id=0.
    #[test]
    fn inter_allowed_picture_header() {
        let data = [0b00111000u8];
        let ph = parse_picture_header(&data).unwrap();
        assert!(!ph.ph_gdr_or_irap_pic_flag);
        assert!(!ph.ph_gdr_pic_flag);
        assert!(ph.ph_inter_slice_allowed_flag);
        assert!(ph.ph_intra_slice_allowed_flag);
        assert_eq!(ph.ph_pic_parameter_set_id, 0);
    }
}
