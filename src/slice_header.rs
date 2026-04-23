//! VVC Slice Header parser (§7.3.7 — `slice_header()`).
//!
//! Foundation scope: parse the leading fixed bit —
//! `sh_picture_header_in_slice_header_flag` — and, when set,
//! delegate to the picture-header parser. The remainder of the slice
//! header (sh_subpic_id, sh_slice_address, sh_extra_bit,
//! sh_num_tiles_in_slice_minus1, sh_slice_type, and the large tail of
//! deblocking / ALF / LMCS / RPL / weighted-prediction / QP / SAO /
//! CABAC-init / entry-point / extension syntax) requires SPS, PPS,
//! and picture-header state whose widths / presence the scaffold does
//! not carry here. The raw tail is captured for a later stateful
//! parser to walk.

use oxideav_core::{Error, Result};

use crate::bitreader::BitReader;
use crate::picture_header::{parse_picture_header, PictureHeader};

#[derive(Clone, Debug)]
pub struct SliceHeader {
    pub sh_picture_header_in_slice_header_flag: bool,
    /// Present only when `sh_picture_header_in_slice_header_flag == 1`.
    pub embedded_picture_header: Option<PictureHeader>,
    /// Remaining RBSP bytes after the parsed leading bits.
    pub payload_tail: Vec<u8>,
    /// Bit offset within `payload_tail[0]` where the tail begins.
    pub payload_tail_bit_offset: u8,
}

/// Parse a slice header RBSP (bytes after the NAL header, emulation-
/// prevention stripped). Foundation decode only reads the leading
/// `sh_picture_header_in_slice_header_flag` and optionally the
/// embedded picture_header_structure(); everything else is returned
/// as an opaque tail.
pub fn parse_slice_header(rbsp: &[u8]) -> Result<SliceHeader> {
    if rbsp.is_empty() {
        return Err(Error::invalid("h266 SH: empty RBSP"));
    }
    let mut br = BitReader::new(rbsp);
    let sh_picture_header_in_slice_header_flag = br.u1()? == 1;
    let embedded_picture_header = if sh_picture_header_in_slice_header_flag {
        // The embedded picture_header_structure() starts immediately
        // after the flag bit. Our `parse_picture_header` expects a
        // byte-aligned buffer — rebuild one starting from the next bit.
        let bit_pos = br.bit_position();
        let tail = collect_bits(rbsp, bit_pos)?;
        let ph = parse_picture_header(&tail)?;
        // Fast-forward the main reader by `ph.consumed_bits`.
        for _ in 0..ph.consumed_bits {
            br.u1()?;
        }
        Some(ph)
    } else {
        None
    };
    let bit_pos = br.bit_position();
    let byte_off = (bit_pos / 8) as usize;
    let bit_off = (bit_pos % 8) as u8;
    let tail = if byte_off < rbsp.len() {
        rbsp[byte_off..].to_vec()
    } else {
        Vec::new()
    };
    Ok(SliceHeader {
        sh_picture_header_in_slice_header_flag,
        embedded_picture_header,
        payload_tail: tail,
        payload_tail_bit_offset: bit_off,
    })
}

/// Build a fresh byte-aligned buffer that contains the bits of `rbsp`
/// starting at bit offset `from_bit`. Useful when delegating to a
/// bit-aligned sub-parser.
fn collect_bits(rbsp: &[u8], from_bit: u64) -> Result<Vec<u8>> {
    let total_bits = rbsp.len() as u64 * 8;
    if from_bit > total_bits {
        return Err(Error::invalid("h266 SH: embed offset out of range"));
    }
    let remaining = total_bits - from_bit;
    let mut out = vec![0u8; ((remaining + 7) / 8) as usize];
    let mut src = BitReader::new(rbsp);
    src.skip(from_bit as u32)?;
    let mut bits_written: u64 = 0;
    while bits_written < remaining {
        let n = core::cmp::min(8, (remaining - bits_written) as u32);
        let v = src.u(n)? as u8;
        let byte_idx = (bits_written / 8) as usize;
        let shift = 8 - n;
        out[byte_idx] |= v << shift;
        bits_written += n as u64;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// No embedded PH — first bit = 0, remainder opaque.
    #[test]
    fn no_embedded_ph() {
        let data = [0b0111_0000u8, 0xAB];
        let sh = parse_slice_header(&data).unwrap();
        assert!(!sh.sh_picture_header_in_slice_header_flag);
        assert!(sh.embedded_picture_header.is_none());
        // Tail offset should be bit 1 within byte 0.
        assert_eq!(sh.payload_tail_bit_offset, 1);
        assert_eq!(sh.payload_tail.len(), 2);
    }

    /// Embedded PH: flag=1 followed by the IRAP picture header from
    /// the picture_header tests (0x88 = 1000_1000).
    /// Bits: 1 | 1000_1000 = 0b1100_0100_0... → byte0 = 0xC4.
    #[test]
    fn embedded_ph_flag() {
        let data = [0xC4u8, 0x00];
        let sh = parse_slice_header(&data).unwrap();
        assert!(sh.sh_picture_header_in_slice_header_flag);
        let ph = sh.embedded_picture_header.unwrap();
        assert!(ph.ph_gdr_or_irap_pic_flag);
        assert!(!ph.ph_inter_slice_allowed_flag);
        assert_eq!(ph.ph_pic_parameter_set_id, 0);
    }
}
