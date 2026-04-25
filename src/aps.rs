//! VVC Adaptation Parameter Set parser (§7.3.2.6).
//!
//! Foundation scope: the APS header — `aps_params_type`,
//! `aps_adaptation_parameter_set_id`, and `aps_chroma_present_flag`.
//!
//! For ALF APSes we additionally walk the §7.3.2.18 `alf_data()` payload
//! and turn it into the spec's [`AlfApsData`] arrays
//! (`AlfCoeffL[]`, `AlfClipL[]`, `AlfCoeffC[]`, `AlfClipC[]`,
//! `CcAlfApsCoeffCb[]`, `CcAlfApsCoeffCr[]`) so the §8.8.5 ALF apply
//! pass can index into them by APS id. LMCS / scaling-list payloads
//! remain opaque (still kept around verbatim).

use oxideav_core::{Error, Result};

use crate::bitreader::BitReader;

/// Number of ALF luma classes per APS — §7.4.3.18 sets `NumAlfFilters = 25`.
pub const NUM_ALF_FILTERS: usize = 25;

/// Number of luma coefficients per filter (7×7 diamond → 13 taps; the
/// central tap is implicit, so 12 are signalled).
pub const ALF_LUMA_NUM_COEFF: usize = 12;

/// Number of chroma coefficients per filter (5×5 diamond → 7 taps; the
/// central tap is implicit, so 6 are signalled).
pub const ALF_CHROMA_NUM_COEFF: usize = 6;

/// Number of CC-ALF coefficients per filter (3×4 diamond → 8 taps; the
/// central tap is implicit, so 7 are signalled).
pub const ALF_CC_NUM_COEFF: usize = 7;

/// The three APS types defined in Table 6 (§7.4.3.6).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ApsParamsType {
    /// `ALF_APS` = 0 — adaptive loop filter parameters (§7.4.9 / §7.4.3.13).
    Alf,
    /// `LMCS_APS` = 1 — luma mapping with chroma scaling parameters.
    Lmcs,
    /// `SCALING_APS` = 2 — scaling list parameters.
    Scaling,
    /// Reserved values (3..=7). Decoders shall ignore NAL units with
    /// these types (§7.4.3.6).
    Reserved(u8),
}

impl ApsParamsType {
    pub fn from_u8(v: u8) -> Self {
        match v {
            0 => Self::Alf,
            1 => Self::Lmcs,
            2 => Self::Scaling,
            other => Self::Reserved(other),
        }
    }

    pub fn as_u8(self) -> u8 {
        match self {
            Self::Alf => 0,
            Self::Lmcs => 1,
            Self::Scaling => 2,
            Self::Reserved(v) => v,
        }
    }
}

/// Decoded `alf_data()` payload — §7.3.2.18 / §7.4.3.18.
///
/// Owns the spec's `filtCoeff` / `AlfCoeffL` / `AlfClipL` / `AlfCoeffC` /
/// `AlfClipC` / `CcAlfApsCoeffCb` / `CcAlfApsCoeffCr` for one ALF APS.
/// `AlfClip*` is laid out as `clipIdx` integer codes (0..=3); the
/// per-bit-depth Table 8 expansion is performed on demand by
/// [`crate::alf::resolve_clip_value`].
#[derive(Clone, Debug, Default)]
pub struct AlfApsData {
    pub alf_luma_filter_signal_flag: bool,
    pub alf_chroma_filter_signal_flag: bool,
    pub alf_cc_cb_filter_signal_flag: bool,
    pub alf_cc_cr_filter_signal_flag: bool,
    pub alf_luma_clip_flag: bool,
    pub alf_chroma_clip_flag: bool,
    /// Number of alternative chroma filters minus 1 (0..=7 per §7.4.3.18).
    pub alf_chroma_num_alt_filters_minus1: u32,
    /// `AlfCoeffL[ filtIdx ][ j ]` — eq. 89 expansion. `NumAlfFilters` rows
    /// of `ALF_LUMA_NUM_COEFF` entries, valid only when
    /// `alf_luma_filter_signal_flag == 1`.
    pub luma_coeff: Vec<[i32; ALF_LUMA_NUM_COEFF]>,
    /// `AlfClipL[ filtIdx ][ j ]` (clipIdx 0..=3, awaiting Table 8
    /// bit-depth expansion).
    pub luma_clip_idx: Vec<[u8; ALF_LUMA_NUM_COEFF]>,
    /// `AlfCoeffC[ altIdx ][ j ]` — eq. 92 expansion.
    pub chroma_coeff: Vec<[i32; ALF_CHROMA_NUM_COEFF]>,
    /// `AlfClipC[ altIdx ][ j ]` (clipIdx 0..=3).
    pub chroma_clip_idx: Vec<[u8; ALF_CHROMA_NUM_COEFF]>,
    /// `CcAlfApsCoeffCb[ k ][ j ]` — see §7.4.3.18; `k` indexes the
    /// signalled filters (0..=3).
    pub cc_cb_coeff: Vec<[i32; ALF_CC_NUM_COEFF]>,
    /// `CcAlfApsCoeffCr[ k ][ j ]`.
    pub cc_cr_coeff: Vec<[i32; ALF_CC_NUM_COEFF]>,
}

#[derive(Clone, Debug)]
pub struct AdaptationParameterSet {
    pub aps_params_type: ApsParamsType,
    pub aps_adaptation_parameter_set_id: u8,
    pub aps_chroma_present_flag: bool,
    /// Decoded ALF payload — `Some` iff `aps_params_type == Alf` and the
    /// payload was parsed successfully.
    pub alf_data: Option<AlfApsData>,
    /// Raw APS payload bytes (full RBSP, including the 9-bit header).
    /// Useful for LMCS / scaling-list APSes that this round does not
    /// dispatch into a typed structure.
    pub payload: Vec<u8>,
}

/// Parse an APS RBSP (after NAL header + emulation-prevention strip).
pub fn parse_aps(rbsp: &[u8]) -> Result<AdaptationParameterSet> {
    if rbsp.is_empty() {
        return Err(Error::invalid("h266 APS: empty RBSP"));
    }
    let mut br = BitReader::new(rbsp);
    let aps_params_type_raw = br.u(3)? as u8;
    let aps_adaptation_parameter_set_id = br.u(5)? as u8;
    let aps_chroma_present_flag = br.u1()? == 1;
    let aps_params_type = ApsParamsType::from_u8(aps_params_type_raw);
    // Validate id range for the three defined types.
    match aps_params_type {
        ApsParamsType::Alf | ApsParamsType::Scaling => {
            if aps_adaptation_parameter_set_id > 7 {
                return Err(Error::invalid(format!(
                    "h266 APS: aps_adaptation_parameter_set_id out of range for type {aps_params_type_raw} (expected 0..7, got {aps_adaptation_parameter_set_id})"
                )));
            }
        }
        ApsParamsType::Lmcs => {
            if aps_adaptation_parameter_set_id > 3 {
                return Err(Error::invalid(format!(
                    "h266 APS: LMCS aps_adaptation_parameter_set_id out of range (expected 0..3, got {aps_adaptation_parameter_set_id})"
                )));
            }
        }
        ApsParamsType::Reserved(_) => {
            // §7.4.3.6: decoders shall ignore APS NAL units with
            // reserved values of aps_params_type. Header is surfaced;
            // the payload is left opaque.
        }
    }
    let alf_data = if matches!(aps_params_type, ApsParamsType::Alf) {
        Some(parse_alf_data(&mut br, aps_chroma_present_flag)?)
    } else {
        None
    };
    Ok(AdaptationParameterSet {
        aps_params_type,
        aps_adaptation_parameter_set_id,
        aps_chroma_present_flag,
        alf_data,
        payload: rbsp.to_vec(),
    })
}

/// Parse the §7.3.2.18 `alf_data()` payload.
fn parse_alf_data(br: &mut BitReader<'_>, aps_chroma_present_flag: bool) -> Result<AlfApsData> {
    let mut out = AlfApsData::default();
    out.alf_luma_filter_signal_flag = br.u1()? == 1;
    if aps_chroma_present_flag {
        out.alf_chroma_filter_signal_flag = br.u1()? == 1;
        out.alf_cc_cb_filter_signal_flag = br.u1()? == 1;
        out.alf_cc_cr_filter_signal_flag = br.u1()? == 1;
    }
    if !out.alf_luma_filter_signal_flag
        && !out.alf_chroma_filter_signal_flag
        && !out.alf_cc_cb_filter_signal_flag
        && !out.alf_cc_cr_filter_signal_flag
    {
        // §7.4.3.18: at least one signal flag must be 1.
        return Err(Error::invalid(
            "h266 ALF APS: at least one of alf_luma/chroma/cc_cb/cc_cr filter_signal_flag \
             must be 1",
        ));
    }
    if out.alf_luma_filter_signal_flag {
        out.alf_luma_clip_flag = br.u1()? == 1;
        let alf_luma_num_filters_signalled_minus1 = br.ue()?;
        if (alf_luma_num_filters_signalled_minus1 as usize) >= NUM_ALF_FILTERS {
            return Err(Error::invalid(format!(
                "h266 ALF APS: alf_luma_num_filters_signalled_minus1 out of range \
                 ({alf_luma_num_filters_signalled_minus1} >= NumAlfFilters = {NUM_ALF_FILTERS})"
            )));
        }
        // alf_luma_coeff_delta_idx[ filtIdx ] — only when more than one
        // filter is signalled.
        let mut delta_idx = [0u32; NUM_ALF_FILTERS];
        if alf_luma_num_filters_signalled_minus1 > 0 {
            let bits = ceil_log2(alf_luma_num_filters_signalled_minus1 as u64 + 1);
            for filt_idx in 0..NUM_ALF_FILTERS {
                delta_idx[filt_idx] = br.u(bits)?;
                if delta_idx[filt_idx] > alf_luma_num_filters_signalled_minus1 {
                    return Err(Error::invalid(format!(
                        "h266 ALF APS: alf_luma_coeff_delta_idx[{filt_idx}]={} > num filters minus1 = {alf_luma_num_filters_signalled_minus1}",
                        delta_idx[filt_idx]
                    )));
                }
            }
        }
        // filtCoeff[ sfIdx ][ j ] — eq. 88. Read absolute value (ue(v))
        // and a sign bit; combine into a signed integer.
        let n = alf_luma_num_filters_signalled_minus1 as usize + 1;
        let mut filt_coeff = vec![[0i32; ALF_LUMA_NUM_COEFF]; n];
        for sf_idx in 0..n {
            for j in 0..ALF_LUMA_NUM_COEFF {
                let abs = br.ue()?;
                if abs > 128 {
                    return Err(Error::invalid(format!(
                        "h266 ALF APS: alf_luma_coeff_abs[{sf_idx}][{j}] = {abs} > 128"
                    )));
                }
                let mut v = abs as i32;
                if abs > 0 && br.u1()? == 1 {
                    v = -v;
                }
                filt_coeff[sf_idx][j] = v;
            }
        }
        // alf_luma_clip_idx (only when alf_luma_clip_flag).
        let mut filt_clip = vec![[0u8; ALF_LUMA_NUM_COEFF]; n];
        if out.alf_luma_clip_flag {
            for sf_idx in 0..n {
                for j in 0..ALF_LUMA_NUM_COEFF {
                    filt_clip[sf_idx][j] = br.u(2)? as u8;
                }
            }
        }
        // Eq. 89: AlfCoeffL[ filtIdx ][ j ] = filtCoeff[ delta_idx[filtIdx] ][ j ].
        // Same indirection for AlfClipL via alf_luma_clip_idx[ delta_idx[filtIdx] ][ j ].
        let mut luma_coeff = vec![[0i32; ALF_LUMA_NUM_COEFF]; NUM_ALF_FILTERS];
        let mut luma_clip = vec![[0u8; ALF_LUMA_NUM_COEFF]; NUM_ALF_FILTERS];
        for filt_idx in 0..NUM_ALF_FILTERS {
            let sf = delta_idx[filt_idx] as usize;
            luma_coeff[filt_idx] = filt_coeff[sf];
            if out.alf_luma_clip_flag {
                luma_clip[filt_idx] = filt_clip[sf];
            }
        }
        out.luma_coeff = luma_coeff;
        out.luma_clip_idx = luma_clip;
    }
    if out.alf_chroma_filter_signal_flag {
        out.alf_chroma_clip_flag = br.u1()? == 1;
        out.alf_chroma_num_alt_filters_minus1 = br.ue()?;
        if out.alf_chroma_num_alt_filters_minus1 > 7 {
            return Err(Error::invalid(format!(
                "h266 ALF APS: alf_chroma_num_alt_filters_minus1 = {} > 7",
                out.alf_chroma_num_alt_filters_minus1
            )));
        }
        let n_alt = out.alf_chroma_num_alt_filters_minus1 as usize + 1;
        out.chroma_coeff = vec![[0i32; ALF_CHROMA_NUM_COEFF]; n_alt];
        out.chroma_clip_idx = vec![[0u8; ALF_CHROMA_NUM_COEFF]; n_alt];
        for alt_idx in 0..n_alt {
            for j in 0..ALF_CHROMA_NUM_COEFF {
                let abs = br.ue()?;
                if abs > 128 {
                    return Err(Error::invalid(format!(
                        "h266 ALF APS: alf_chroma_coeff_abs[{alt_idx}][{j}] = {abs} > 128"
                    )));
                }
                let mut v = abs as i32;
                if abs > 0 && br.u1()? == 1 {
                    v = -v;
                }
                out.chroma_coeff[alt_idx][j] = v;
            }
            if out.alf_chroma_clip_flag {
                for j in 0..ALF_CHROMA_NUM_COEFF {
                    out.chroma_clip_idx[alt_idx][j] = br.u(2)? as u8;
                }
            }
        }
    }
    if out.alf_cc_cb_filter_signal_flag {
        let n_minus1 = br.ue()?;
        if n_minus1 > 3 {
            return Err(Error::invalid(format!(
                "h266 ALF APS: alf_cc_cb_filters_signalled_minus1 = {n_minus1} > 3"
            )));
        }
        let n = n_minus1 as usize + 1;
        out.cc_cb_coeff = vec![[0i32; ALF_CC_NUM_COEFF]; n];
        for k in 0..n {
            for j in 0..ALF_CC_NUM_COEFF {
                let mapped_abs = br.u(3)?;
                let mut v = mapped_abs as i32;
                if mapped_abs > 0 && br.u1()? == 1 {
                    v = -v;
                }
                out.cc_cb_coeff[k][j] = v;
            }
        }
    }
    if out.alf_cc_cr_filter_signal_flag {
        let n_minus1 = br.ue()?;
        if n_minus1 > 3 {
            return Err(Error::invalid(format!(
                "h266 ALF APS: alf_cc_cr_filters_signalled_minus1 = {n_minus1} > 3"
            )));
        }
        let n = n_minus1 as usize + 1;
        out.cc_cr_coeff = vec![[0i32; ALF_CC_NUM_COEFF]; n];
        for k in 0..n {
            for j in 0..ALF_CC_NUM_COEFF {
                let mapped_abs = br.u(3)?;
                let mut v = mapped_abs as i32;
                if mapped_abs > 0 && br.u1()? == 1 {
                    v = -v;
                }
                out.cc_cr_coeff[k][j] = v;
            }
        }
    }
    Ok(out)
}

/// Compute `Ceil(Log2(x))` for `x >= 1`. Used for the `u(v)` length of
/// `alf_luma_coeff_delta_idx` (length =
/// `Ceil(Log2(alf_luma_num_filters_signalled_minus1 + 1))`).
fn ceil_log2(x: u64) -> u32 {
    debug_assert!(x >= 1);
    if x == 1 {
        0
    } else {
        64 - (x - 1).leading_zeros()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alf_aps_header() {
        // u(3)=0 (ALF), u(5)=3 (id), u(1)=1 (chroma_present), then
        // a minimal alf_data tail with all signal flags off except luma=1
        // and a single filter.
        //   byte0 = 0000_0011 (high 8 bits: type=000, id=00011)
        //   byte1 high bit = 1 (chroma_present)
        //   then alf_data starts:
        //     u(1) alf_luma_filter_signal_flag = 1
        //     u(1) alf_chroma_filter_signal_flag = 0
        //     u(1) alf_cc_cb_filter_signal_flag = 0
        //     u(1) alf_cc_cr_filter_signal_flag = 0
        //     u(1) alf_luma_clip_flag = 0
        //     ue(v) alf_luma_num_filters_signalled_minus1 = 0  -> "1" bit
        //     for sfIdx=0, j=0..11: alf_luma_coeff_abs = 0 (just "1" each), sign skipped
        // So bits past header: "1 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1" — 18 bits.
        // We assemble carefully into a fresh byte stream.
        let header_byte = 0b0000_0011u8;
        let mut bits = String::new();
        bits.push('1'); // chroma_present
                        // alf_data
        bits.push('1'); // alf_luma_filter_signal_flag
        bits.push('0'); // alf_chroma
        bits.push('0'); // cc_cb
        bits.push('0'); // cc_cr
        bits.push('0'); // alf_luma_clip_flag
        bits.push('1'); // ue(v) = 0 → "1"
                        // 12 ue(v) = 0 each
        for _ in 0..12 {
            bits.push('1');
        }
        // pad to byte boundary
        while bits.len() % 8 != 0 {
            bits.push('0');
        }
        let mut data = vec![header_byte];
        for chunk in bits.as_bytes().chunks(8) {
            let mut b = 0u8;
            for &c in chunk {
                b = (b << 1) | (c - b'0');
            }
            data.push(b);
        }
        let aps = parse_aps(&data).unwrap();
        assert_eq!(aps.aps_params_type, ApsParamsType::Alf);
        assert_eq!(aps.aps_adaptation_parameter_set_id, 3);
        assert!(aps.aps_chroma_present_flag);
        let alf = aps.alf_data.as_ref().unwrap();
        assert!(alf.alf_luma_filter_signal_flag);
        assert!(!alf.alf_chroma_filter_signal_flag);
        assert_eq!(alf.luma_coeff.len(), NUM_ALF_FILTERS);
        // All filter coefficients are zero (every alf_luma_coeff_abs = 0,
        // sign skipped per spec).
        assert!(alf.luma_coeff.iter().all(|row| row.iter().all(|&v| v == 0)));
    }

    #[test]
    fn lmcs_id_out_of_range() {
        // type=1 (LMCS), id=5 (> 3 → rejected)
        //   byte0 = 001_00101 = 0b0010_0101
        //   byte1 chroma = 0
        let data = [0b0010_0101, 0];
        assert!(parse_aps(&data).is_err());
    }

    #[test]
    fn reserved_type_accepted_without_validation() {
        // type=5 (reserved). The parser doesn't fail — it surfaces the
        // value so callers can skip the NAL explicitly.
        //   byte0 = 101_00001 = 0b1010_0001
        //   byte1 = 0 (chroma=0, rest is "payload")
        let data = [0b1010_0001, 0];
        let aps = parse_aps(&data).unwrap();
        assert!(matches!(aps.aps_params_type, ApsParamsType::Reserved(5)));
        assert!(aps.alf_data.is_none());
    }

    #[test]
    fn ceil_log2_basic() {
        assert_eq!(ceil_log2(1), 0);
        assert_eq!(ceil_log2(2), 1);
        assert_eq!(ceil_log2(3), 2);
        assert_eq!(ceil_log2(4), 2);
        assert_eq!(ceil_log2(5), 3);
        assert_eq!(ceil_log2(8), 3);
        assert_eq!(ceil_log2(25), 5);
    }

    #[test]
    fn alf_data_rejects_all_signal_flags_zero() {
        // ALF type, id=0, chroma_present=1, then alf_data with all four
        // signal flags = 0. Spec §7.4.3.18 requires at least one to be 1.
        //   byte0 = 000_00000 = 0
        //   byte1 = 1 (chroma_present), 0,0,0,0 (all signal flags), then padding
        //         = 1 0 0 0 0 0 0 0 = 0x80
        let data = [0u8, 0x80];
        assert!(parse_aps(&data).is_err());
    }
}
