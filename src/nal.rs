//! VVC / H.266 NAL unit framing (§7.3.1 / §7.4.2 / Annex B).
//!
//! Two byte-stream formats are commonly seen:
//!
//! * **Annex B** (ITU-T H.266 Annex B / `.266` / `.vvc` files, MPEG-TS):
//!   each NAL unit is preceded by a 3- or 4-byte start code prefix
//!   (`0x000001` or `0x00000001`).
//! * **Length-prefixed** (ISOBMFF `vvc1` / `vvi1`): each NAL unit begins
//!   with an N-byte big-endian length field, where N = `length_size_minus_one
//!   + 1` from the VvcDecoderConfigurationRecord (typically 4).
//!
//! The 2-byte NAL header (§7.3.1.2) packs:
//!
//! ```text
//!   forbidden_zero_bit       f(1)   — must be 0
//!   nuh_reserved_zero_bit    u(1)   — must be 0 in this spec version;
//!                                     decoders shall ignore NALs where
//!                                     this bit is 1 (§7.4.2.2).
//!   nuh_layer_id             u(6)   — 0..55 (others reserved)
//!   nal_unit_type            u(5)   — see Table 5
//!   nuh_temporal_id_plus1    u(3)   — must be != 0
//! ```
//!
//! After framing, callers should strip emulation-prevention bytes via
//! [`extract_rbsp`] before applying the bit reader.

use oxideav_core::{Error, Result};

/// VVC NAL unit type codes (§7.4.2.2, Table 5). 5-bit values, 0..31.
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NalUnitType {
    /// 0 — coded slice of a trailing picture / subpicture.
    TrailNut,
    /// 1 — coded slice of an STSA picture / subpicture.
    StsaNut,
    /// 2 — coded slice of a RADL picture / subpicture.
    RadlNut,
    /// 3 — coded slice of a RASL picture / subpicture.
    RaslNut,
    /// 4..6 — reserved non-IRAP VCL NAL unit types.
    RsvVcl4,
    RsvVcl5,
    RsvVcl6,
    /// 7 — coded slice of an IDR picture / subpicture (leading pictures allowed).
    IdrWRadl,
    /// 8 — coded slice of an IDR picture / subpicture (no leading pictures).
    IdrNLp,
    /// 9 — coded slice of a CRA picture / subpicture.
    CraNut,
    /// 10 — coded slice of a GDR picture / subpicture.
    GdrNut,
    /// 11 — reserved IRAP VCL NAL unit type.
    RsvIrap11,
    /// 12 — operating point information.
    OpiNut,
    /// 13 — decoding capability information.
    DciNut,
    /// 14 — video parameter set.
    VpsNut,
    /// 15 — sequence parameter set.
    SpsNut,
    /// 16 — picture parameter set.
    PpsNut,
    /// 17 — prefix adaptation parameter set.
    PrefixApsNut,
    /// 18 — suffix adaptation parameter set.
    SuffixApsNut,
    /// 19 — picture header.
    PhNut,
    /// 20 — AU delimiter.
    AudNut,
    /// 21 — end of sequence.
    EosNut,
    /// 22 — end of bitstream.
    EobNut,
    /// 23 — prefix SEI.
    PrefixSeiNut,
    /// 24 — suffix SEI.
    SuffixSeiNut,
    /// 25 — filler data.
    FdNut,
    /// 26 / 27 — reserved non-VCL NAL unit types.
    RsvNvcl26,
    RsvNvcl27,
    /// 28..31 — unspecified non-VCL NAL unit types.
    Unspec(u8),
}

impl NalUnitType {
    pub fn from_u8(v: u8) -> Self {
        use NalUnitType::*;
        match v {
            0 => TrailNut,
            1 => StsaNut,
            2 => RadlNut,
            3 => RaslNut,
            4 => RsvVcl4,
            5 => RsvVcl5,
            6 => RsvVcl6,
            7 => IdrWRadl,
            8 => IdrNLp,
            9 => CraNut,
            10 => GdrNut,
            11 => RsvIrap11,
            12 => OpiNut,
            13 => DciNut,
            14 => VpsNut,
            15 => SpsNut,
            16 => PpsNut,
            17 => PrefixApsNut,
            18 => SuffixApsNut,
            19 => PhNut,
            20 => AudNut,
            21 => EosNut,
            22 => EobNut,
            23 => PrefixSeiNut,
            24 => SuffixSeiNut,
            25 => FdNut,
            26 => RsvNvcl26,
            27 => RsvNvcl27,
            other @ 28..=31 => Unspec(other),
            // Out-of-range — the 5-bit field can only hold 0..31 so this
            // branch only triggers if callers synthesise a value.
            other => Unspec(other),
        }
    }

    pub fn as_u8(self) -> u8 {
        use NalUnitType::*;
        match self {
            TrailNut => 0,
            StsaNut => 1,
            RadlNut => 2,
            RaslNut => 3,
            RsvVcl4 => 4,
            RsvVcl5 => 5,
            RsvVcl6 => 6,
            IdrWRadl => 7,
            IdrNLp => 8,
            CraNut => 9,
            GdrNut => 10,
            RsvIrap11 => 11,
            OpiNut => 12,
            DciNut => 13,
            VpsNut => 14,
            SpsNut => 15,
            PpsNut => 16,
            PrefixApsNut => 17,
            SuffixApsNut => 18,
            PhNut => 19,
            AudNut => 20,
            EosNut => 21,
            EobNut => 22,
            PrefixSeiNut => 23,
            SuffixSeiNut => 24,
            FdNut => 25,
            RsvNvcl26 => 26,
            RsvNvcl27 => 27,
            Unspec(v) => v,
        }
    }

    /// Whether this NAL holds coded slice data ("VCL"): types 0..11.
    pub fn is_vcl(self) -> bool {
        matches!(self.as_u8(), 0..=11)
    }

    /// Whether this is an IRAP (intra random access point) picture NAL:
    /// IDR_W_RADL, IDR_N_LP, CRA_NUT, or RSV_IRAP_11 (§3.IRAP).
    pub fn is_irap(self) -> bool {
        matches!(self.as_u8(), 7..=11)
    }

    /// Whether this is one of the parameter-set NAL types (VPS / SPS / PPS
    /// / prefix+suffix APS / DCI / OPI).
    pub fn is_parameter_set(self) -> bool {
        matches!(self.as_u8(), 12..=18)
    }
}

/// Parsed 2-byte VVC NAL header (§7.3.1.2).
#[derive(Clone, Copy, Debug)]
pub struct NalHeader {
    pub nal_unit_type: NalUnitType,
    pub nuh_layer_id: u8,
    pub nuh_temporal_id_plus1: u8,
}

impl NalHeader {
    /// Parse the 2-byte NAL header. The caller must pass at least 2 bytes —
    /// the bytes immediately after the start code (or after the length
    /// prefix in length-prefixed mode).
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 2 {
            return Err(Error::invalid("h266: NAL header < 2 bytes"));
        }
        let b0 = bytes[0];
        let b1 = bytes[1];
        // forbidden_zero_bit (§7.4.2.2)
        if b0 & 0x80 != 0 {
            return Err(Error::invalid(
                "h266: NAL forbidden_zero_bit must be 0 (corrupt or non-VVC)",
            ));
        }
        // nuh_reserved_zero_bit — must be 0. §7.4.2.2 says decoders shall
        // ignore NALs with this bit set; we surface it as an error so the
        // caller can log and resync rather than silently discard.
        if b0 & 0x40 != 0 {
            return Err(Error::invalid(
                "h266: nuh_reserved_zero_bit must be 0 (§7.4.2.2)",
            ));
        }
        let nuh_layer_id = b0 & 0x3F;
        if nuh_layer_id > 55 {
            // §7.4.2.2: values 56..63 are reserved. Not strictly an error
            // (decoders "shall ignore"), but the 2-byte header layout gives
            // a 6-bit field so the raw value fits; we surface anything >55
            // so the caller can discard the NAL.
            return Err(Error::invalid(format!(
                "h266: nuh_layer_id {nuh_layer_id} > 55 is reserved (§7.4.2.2)"
            )));
        }
        let nal_unit_type = (b1 >> 3) & 0x1F;
        let tid_plus1 = b1 & 0x07;
        if tid_plus1 == 0 {
            return Err(Error::invalid(
                "h266: nuh_temporal_id_plus1 must be > 0 (§7.4.2.2)",
            ));
        }
        Ok(Self {
            nal_unit_type: NalUnitType::from_u8(nal_unit_type),
            nuh_layer_id,
            nuh_temporal_id_plus1: tid_plus1,
        })
    }

    /// `TemporalId` = `nuh_temporal_id_plus1` − 1.
    pub fn temporal_id(self) -> u8 {
        self.nuh_temporal_id_plus1 - 1
    }
}

/// One NAL unit located in a buffer (zero-copy slice).
#[derive(Clone, Copy, Debug)]
pub struct NalRef<'a> {
    pub header: NalHeader,
    /// Body bytes including the 2-byte NAL header (still emulation-prevented).
    pub raw: &'a [u8],
}

impl<'a> NalRef<'a> {
    /// Bytes after the 2-byte NAL header — still emulation-prevented;
    /// callers normally want `extract_rbsp(self.payload())`.
    pub fn payload(&self) -> &'a [u8] {
        &self.raw[2..]
    }
}

/// Iterate over Annex B NAL units. Start codes are `0x000001` (3 bytes) or
/// `0x00000001` (4 bytes). Trailing zero stuffing is tolerated.
pub fn iter_annex_b(data: &[u8]) -> AnnexBIter<'_> {
    AnnexBIter { data, pos: 0 }
}

pub struct AnnexBIter<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Iterator for AnnexBIter<'a> {
    type Item = NalRef<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let (sc_off, sc_len) = find_start_code(self.data, self.pos)?;
        let body_start = sc_off + sc_len;
        // Find the next start code (or EOF) to bound the body.
        let body_end = match find_start_code(self.data, body_start) {
            Some((next_off, _)) => trim_trailing_zeros(&self.data[..next_off], body_start),
            None => trim_trailing_zeros(self.data, body_start),
        };
        self.pos = body_end;
        if body_end <= body_start || body_end - body_start < 2 {
            // Empty or too-short NAL — keep scanning so callers don't lock up.
            return self.next();
        }
        let raw = &self.data[body_start..body_end];
        // Skip silently on a malformed header — the iterator should be
        // tolerant; malformed NALs can be logged by the consumer.
        let header = NalHeader::parse(raw).ok()?;
        Some(NalRef { header, raw })
    }
}

/// Trim trailing `0x00` bytes from `slice[..end]` down to (but not below)
/// `min_end`. Annex B may pad NAL bodies with zero stuffing.
fn trim_trailing_zeros(slice: &[u8], min_end: usize) -> usize {
    let mut end = slice.len();
    while end > min_end && slice[end - 1] == 0 {
        end -= 1;
    }
    end
}

/// Search forward from `from` for the start of the next start-code prefix
/// (`0x000001` or `0x00000001`). Returns `(offset_of_first_zero, prefix_len)`.
pub fn find_start_code(data: &[u8], from: usize) -> Option<(usize, usize)> {
    let mut i = from;
    while i + 3 <= data.len() {
        if data[i] == 0 && data[i + 1] == 0 {
            // Walk over an arbitrary number of zero stuffing bytes.
            let mut j = i + 2;
            while j < data.len() && data[j] == 0 {
                j += 1;
            }
            if j < data.len() && data[j] == 0x01 {
                let prefix_len = j - i + 1;
                return Some((i, prefix_len));
            }
            i = j.max(i + 1);
            continue;
        }
        i += 1;
    }
    None
}

/// Iterate length-prefixed NAL units (ISOBMFF `vvc1`/`vvi1` sample data).
/// `length_size` is the number of bytes used to encode each length field —
/// typically 4.
pub fn iter_length_prefixed(data: &[u8], length_size: u8) -> Result<Vec<NalRef<'_>>> {
    if !matches!(length_size, 1 | 2 | 4) {
        return Err(Error::invalid(format!(
            "h266: invalid length_size_minus_one (must be 0, 1 or 3): got {length_size}"
        )));
    }
    let n = length_size as usize;
    let mut out = Vec::new();
    let mut i = 0;
    while i + n <= data.len() {
        let mut len: usize = 0;
        for k in 0..n {
            len = (len << 8) | data[i + k] as usize;
        }
        i += n;
        if len < 2 || i + len > data.len() {
            return Err(Error::invalid(format!(
                "h266: length-prefixed NAL out of bounds (len={len}, remaining={})",
                data.len() - i
            )));
        }
        let raw = &data[i..i + len];
        let header = NalHeader::parse(raw)?;
        out.push(NalRef { header, raw });
        i += len;
    }
    if i != data.len() {
        return Err(Error::invalid(format!(
            "h266: trailing {} bytes after length-prefixed NAL stream",
            data.len() - i
        )));
    }
    Ok(out)
}

/// Strip VVC emulation-prevention bytes. Per §7.4.1, any sequence
/// `0x00 0x00 0x03` inside NAL data is decoded by removing the `0x03`.
/// Identical to HEVC and AVC.
pub fn extract_rbsp(nal_payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(nal_payload.len());
    let mut i = 0;
    while i < nal_payload.len() {
        if i + 2 < nal_payload.len()
            && nal_payload[i] == 0
            && nal_payload[i + 1] == 0
            && nal_payload[i + 2] == 0x03
        {
            out.push(0);
            out.push(0);
            i += 3;
            continue;
        }
        out.push(nal_payload[i]);
        i += 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2-byte VVC header builder for tests.
    ///
    /// Layout (MSB-first):
    /// `b0 = [F:1][R:1][layer:6]`, `b1 = [type:5][tid+1:3]`.
    fn mk_header(nut: u8, layer: u8, tid_plus1: u8) -> [u8; 2] {
        let b0 = (layer & 0x3F) & 0x3F; // F=0, R=0
        let b1 = ((nut & 0x1F) << 3) | (tid_plus1 & 0x07);
        [b0, b1]
    }

    #[test]
    fn parse_header_vps() {
        // VPS is type 14; layer 0, tid+1 = 1 → b1 = (14<<3)|1 = 0x71.
        let bytes = mk_header(14, 0, 1);
        let h = NalHeader::parse(&bytes).unwrap();
        assert_eq!(h.nal_unit_type, NalUnitType::VpsNut);
        assert_eq!(h.nuh_layer_id, 0);
        assert_eq!(h.nuh_temporal_id_plus1, 1);
        assert_eq!(h.temporal_id(), 0);
    }

    #[test]
    fn parse_header_sps_pps_aps_ph() {
        let h = NalHeader::parse(&mk_header(15, 0, 1)).unwrap();
        assert_eq!(h.nal_unit_type, NalUnitType::SpsNut);
        let h = NalHeader::parse(&mk_header(16, 0, 1)).unwrap();
        assert_eq!(h.nal_unit_type, NalUnitType::PpsNut);
        let h = NalHeader::parse(&mk_header(17, 0, 1)).unwrap();
        assert_eq!(h.nal_unit_type, NalUnitType::PrefixApsNut);
        let h = NalHeader::parse(&mk_header(18, 0, 1)).unwrap();
        assert_eq!(h.nal_unit_type, NalUnitType::SuffixApsNut);
        let h = NalHeader::parse(&mk_header(19, 0, 1)).unwrap();
        assert_eq!(h.nal_unit_type, NalUnitType::PhNut);
    }

    #[test]
    fn vcl_and_irap_classification() {
        assert!(NalUnitType::TrailNut.is_vcl());
        assert!(NalUnitType::IdrWRadl.is_vcl());
        assert!(NalUnitType::IdrWRadl.is_irap());
        assert!(NalUnitType::CraNut.is_irap());
        assert!(!NalUnitType::VpsNut.is_vcl());
        assert!(!NalUnitType::VpsNut.is_irap());
        assert!(NalUnitType::SpsNut.is_parameter_set());
        assert!(!NalUnitType::TrailNut.is_parameter_set());
    }

    #[test]
    fn forbidden_bit_is_rejected() {
        // High bit set on b0
        let bytes = [0x80, 0x71];
        assert!(NalHeader::parse(&bytes).is_err());
    }

    #[test]
    fn reserved_zero_bit_is_rejected() {
        // R bit (bit 6 of b0) set
        let bytes = [0x40, 0x71];
        assert!(NalHeader::parse(&bytes).is_err());
    }

    #[test]
    fn layer_out_of_range_is_rejected() {
        // layer_id = 56 is reserved
        let bytes = [56 & 0x3F, 0x71];
        assert!(NalHeader::parse(&bytes).is_err());
    }

    #[test]
    fn temporal_id_zero_is_rejected() {
        // tid_plus1 = 0 in b1
        let bytes = [0x00, 0x70];
        assert!(NalHeader::parse(&bytes).is_err());
    }

    #[test]
    fn rbsp_strip() {
        let input = [0x00u8, 0x00, 0x03, 0xAB, 0x00, 0x00, 0x03, 0xCD];
        let out = extract_rbsp(&input);
        assert_eq!(out, vec![0x00, 0x00, 0xAB, 0x00, 0x00, 0xCD]);
    }

    #[test]
    fn rbsp_strip_no_change() {
        let input = [0x00u8, 0x00, 0x02, 0xAB];
        let out = extract_rbsp(&input);
        assert_eq!(out, input);
    }

    #[test]
    fn annex_b_iter_finds_three_nals() {
        // VPS, SPS, PPS in sequence.
        let vps = mk_header(14, 0, 1);
        let sps = mk_header(15, 0, 1);
        let pps = mk_header(16, 0, 1);
        let data = [
            0x00, 0x00, 0x00, 0x01, vps[0], vps[1], 0xAA, // VPS
            0x00, 0x00, 0x01, sps[0], sps[1], 0xBB, 0xCC, // SPS
            0x00, 0x00, 0x01, pps[0], pps[1], 0xDD, // PPS
        ];
        let nals: Vec<_> = iter_annex_b(&data).collect();
        assert_eq!(nals.len(), 3);
        assert_eq!(nals[0].header.nal_unit_type, NalUnitType::VpsNut);
        assert_eq!(nals[1].header.nal_unit_type, NalUnitType::SpsNut);
        assert_eq!(nals[2].header.nal_unit_type, NalUnitType::PpsNut);
    }

    #[test]
    fn length_prefixed_roundtrip() {
        let vps = mk_header(14, 0, 1);
        let sps = mk_header(15, 0, 1);
        let data = [
            0x00u8, 0x00, 0x00, 0x03, vps[0], vps[1], 0xAA, // first NAL, len=3
            0x00, 0x00, 0x00, 0x02, sps[0], sps[1], // second NAL, len=2
        ];
        let nals = iter_length_prefixed(&data, 4).unwrap();
        assert_eq!(nals.len(), 2);
        assert_eq!(nals[0].header.nal_unit_type, NalUnitType::VpsNut);
        assert_eq!(nals[1].header.nal_unit_type, NalUnitType::SpsNut);
    }

    #[test]
    fn length_prefixed_overflow_is_rejected() {
        let data = [0x00u8, 0x00, 0x00, 0xFF, 0x71, 0x01];
        assert!(iter_length_prefixed(&data, 4).is_err());
    }
}
