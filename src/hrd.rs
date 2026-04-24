//! VVC HRD timing parameter parsers (§7.3.5).
//!
//! Three sub-structures are defined in the spec:
//!
//! * `general_timing_hrd_parameters()` (§7.3.5.1) — carries the sequence-
//!   level `num_units_in_tick`, `time_scale`, the NAL / VCL HRD enable
//!   flags, the DU (decoding-unit) HRD flag, and the common scale / count
//!   fields that the two downstream sub-structures consume.
//! * `ols_timing_hrd_parameters(firstSubLayer, MaxSubLayersVal)`
//!   (§7.3.5.2) — per-sublayer `fixed_pic_rate_*` / `low_delay_hrd_flag`
//!   fields plus optional `sublayer_hrd_parameters` pairs for NAL and
//!   VCL paths.
//! * `sublayer_hrd_parameters(subLayerId)` (§7.3.5.3) — per-CPB-count
//!   `(bit_rate, cpb_size, cbr_flag)` triples.
//!
//! The VVC reference foundation consumes these primarily as an opaque
//! "walk past me" block so the SPS parser can reach `sps_field_seq_flag`
//! / `sps_vui_parameters_present_flag` / `sps_extension_flag`. We still
//! store the parsed values verbatim so later increments can wire timing
//! into a CPB model.

use oxideav_core::{Error, Result};

use crate::bitreader::BitReader;

/// `general_timing_hrd_parameters()` — §7.3.5.1.
#[derive(Clone, Debug)]
pub struct GeneralTimingHrdParameters {
    pub num_units_in_tick: u32,
    pub time_scale: u32,
    pub general_nal_hrd_params_present_flag: bool,
    pub general_vcl_hrd_params_present_flag: bool,
    /// Present only when either NAL or VCL HRD flags is set.
    pub general_same_pic_timing_in_all_ols_flag: bool,
    pub general_du_hrd_params_present_flag: bool,
    /// Present only when `general_du_hrd_params_present_flag == 1`.
    pub tick_divisor_minus2: u8,
    pub bit_rate_scale: u8,
    pub cpb_size_scale: u8,
    /// Present only when `general_du_hrd_params_present_flag == 1`.
    pub cpb_size_du_scale: u8,
    pub hrd_cpb_cnt_minus1: u32,
}

impl Default for GeneralTimingHrdParameters {
    fn default() -> Self {
        Self {
            num_units_in_tick: 0,
            time_scale: 0,
            general_nal_hrd_params_present_flag: false,
            general_vcl_hrd_params_present_flag: false,
            general_same_pic_timing_in_all_ols_flag: false,
            general_du_hrd_params_present_flag: false,
            tick_divisor_minus2: 0,
            bit_rate_scale: 0,
            cpb_size_scale: 0,
            cpb_size_du_scale: 0,
            hrd_cpb_cnt_minus1: 0,
        }
    }
}

/// `ols_timing_hrd_parameters(firstSubLayer, MaxSubLayersVal)` — §7.3.5.2.
///
/// `sublayers` is indexed from 0 and always starts at the sublayer id
/// corresponding to `firstSubLayer` (the spec's i-loop). When
/// `sub_layer_cpb_params_present_flag == 0`, `firstSubLayer == MaxSubLayersVal`
/// and there is exactly one entry at index 0.
#[derive(Clone, Debug, Default)]
pub struct OlsTimingHrdParameters {
    pub sublayers: Vec<OlsSublayerTiming>,
}

#[derive(Clone, Debug, Default)]
pub struct OlsSublayerTiming {
    pub fixed_pic_rate_general_flag: bool,
    pub fixed_pic_rate_within_cvs_flag: bool,
    /// Present only when `fixed_pic_rate_within_cvs_flag == 1`.
    pub elemental_duration_in_tc_minus1: u32,
    /// Present only when neither fixed-pic-rate flag is set and
    /// `hrd_cpb_cnt_minus1 == 0`.
    pub low_delay_hrd_flag: bool,
    pub nal_hrd: Option<SublayerHrdParameters>,
    pub vcl_hrd: Option<SublayerHrdParameters>,
}

/// `sublayer_hrd_parameters(subLayerId)` — §7.3.5.3. One entry per CPB
/// specification (0..=`hrd_cpb_cnt_minus1`).
#[derive(Clone, Debug, Default)]
pub struct SublayerHrdParameters {
    pub entries: Vec<CpbSpec>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CpbSpec {
    pub bit_rate_value_minus1: u32,
    pub cpb_size_value_minus1: u32,
    pub cpb_size_du_value_minus1: u32,
    pub bit_rate_du_value_minus1: u32,
    pub cbr_flag: bool,
}

/// Parse §7.3.5.1 into [`GeneralTimingHrdParameters`].
pub fn parse_general_timing_hrd_parameters(
    br: &mut BitReader<'_>,
) -> Result<GeneralTimingHrdParameters> {
    let num_units_in_tick = br.u(32)?;
    let time_scale = br.u(32)?;
    if time_scale == 0 {
        return Err(Error::invalid(
            "h266 HRD: time_scale shall be greater than 0",
        ));
    }
    let general_nal_hrd_params_present_flag = br.u1()? == 1;
    let general_vcl_hrd_params_present_flag = br.u1()? == 1;
    let mut out = GeneralTimingHrdParameters {
        num_units_in_tick,
        time_scale,
        general_nal_hrd_params_present_flag,
        general_vcl_hrd_params_present_flag,
        ..Default::default()
    };
    if general_nal_hrd_params_present_flag || general_vcl_hrd_params_present_flag {
        out.general_same_pic_timing_in_all_ols_flag = br.u1()? == 1;
        out.general_du_hrd_params_present_flag = br.u1()? == 1;
        if out.general_du_hrd_params_present_flag {
            out.tick_divisor_minus2 = br.u(8)? as u8;
        }
        out.bit_rate_scale = br.u(4)? as u8;
        out.cpb_size_scale = br.u(4)? as u8;
        if out.general_du_hrd_params_present_flag {
            out.cpb_size_du_scale = br.u(4)? as u8;
        }
        out.hrd_cpb_cnt_minus1 = br.ue()?;
        // §7.4.6.1: hrd_cpb_cnt_minus1 shall be in the range 0..=31.
        if out.hrd_cpb_cnt_minus1 > 31 {
            return Err(Error::invalid(format!(
                "h266 HRD: hrd_cpb_cnt_minus1 out of range ({})",
                out.hrd_cpb_cnt_minus1
            )));
        }
    }
    Ok(out)
}

/// Parse §7.3.5.3 — `sublayer_hrd_parameters(subLayerId)`.
pub fn parse_sublayer_hrd_parameters(
    br: &mut BitReader<'_>,
    hrd_cpb_cnt_minus1: u32,
    general_du_hrd_params_present_flag: bool,
) -> Result<SublayerHrdParameters> {
    let mut entries = Vec::with_capacity((hrd_cpb_cnt_minus1 as usize) + 1);
    for _ in 0..=hrd_cpb_cnt_minus1 {
        let bit_rate_value_minus1 = br.ue()?;
        let cpb_size_value_minus1 = br.ue()?;
        let (cpb_size_du_value_minus1, bit_rate_du_value_minus1) =
            if general_du_hrd_params_present_flag {
                (br.ue()?, br.ue()?)
            } else {
                (0, 0)
            };
        let cbr_flag = br.u1()? == 1;
        entries.push(CpbSpec {
            bit_rate_value_minus1,
            cpb_size_value_minus1,
            cpb_size_du_value_minus1,
            bit_rate_du_value_minus1,
            cbr_flag,
        });
    }
    Ok(SublayerHrdParameters { entries })
}

/// Parse §7.3.5.2 — `ols_timing_hrd_parameters(firstSubLayer, MaxSubLayersVal)`.
///
/// `general` supplies the global HRD enable flags + CPB count required
/// to decide which sub-fields are emitted.
pub fn parse_ols_timing_hrd_parameters(
    br: &mut BitReader<'_>,
    general: &GeneralTimingHrdParameters,
    first_sub_layer: u8,
    max_sub_layers_val: u8,
) -> Result<OlsTimingHrdParameters> {
    if first_sub_layer > max_sub_layers_val {
        return Err(Error::invalid(format!(
            "h266 HRD: firstSubLayer ({first_sub_layer}) > MaxSubLayersVal ({max_sub_layers_val})"
        )));
    }
    let mut sublayers =
        Vec::with_capacity((max_sub_layers_val as usize - first_sub_layer as usize) + 1);
    for _ in first_sub_layer..=max_sub_layers_val {
        let mut s = OlsSublayerTiming::default();
        s.fixed_pic_rate_general_flag = br.u1()? == 1;
        if !s.fixed_pic_rate_general_flag {
            s.fixed_pic_rate_within_cvs_flag = br.u1()? == 1;
        } else {
            // §7.4.6.2: when fixed_pic_rate_general_flag is 1,
            // fixed_pic_rate_within_cvs_flag is inferred to be 1.
            s.fixed_pic_rate_within_cvs_flag = true;
        }
        if s.fixed_pic_rate_within_cvs_flag {
            s.elemental_duration_in_tc_minus1 = br.ue()?;
        } else if (general.general_nal_hrd_params_present_flag
            || general.general_vcl_hrd_params_present_flag)
            && general.hrd_cpb_cnt_minus1 == 0
        {
            s.low_delay_hrd_flag = br.u1()? == 1;
        }
        if general.general_nal_hrd_params_present_flag {
            s.nal_hrd = Some(parse_sublayer_hrd_parameters(
                br,
                general.hrd_cpb_cnt_minus1,
                general.general_du_hrd_params_present_flag,
            )?);
        }
        if general.general_vcl_hrd_params_present_flag {
            s.vcl_hrd = Some(parse_sublayer_hrd_parameters(
                br,
                general.hrd_cpb_cnt_minus1,
                general.general_du_hrd_params_present_flag,
            )?);
        }
        sublayers.push(s);
    }
    Ok(OlsTimingHrdParameters { sublayers })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn push_u(bits: &mut Vec<u8>, v: u64, n: u32) {
        for i in (0..n).rev() {
            bits.push(((v >> i) & 1) as u8);
        }
    }

    fn push_ue(bits: &mut Vec<u8>, value: u32) {
        let code_num = value as u64 + 1;
        let mut zeros: u32 = 0;
        while (1u64 << (zeros + 1)) <= code_num {
            zeros += 1;
        }
        for _ in 0..zeros {
            bits.push(0);
        }
        push_u(bits, code_num, zeros + 1);
    }

    fn pack(bits: &[u8]) -> Vec<u8> {
        let mut padded = bits.to_vec();
        while padded.len() % 8 != 0 {
            padded.push(0);
        }
        let mut out = Vec::with_capacity(padded.len() / 8);
        for chunk in padded.chunks(8) {
            let mut b = 0u8;
            for (i, &bit) in chunk.iter().enumerate() {
                b |= bit << (7 - i);
            }
            out.push(b);
        }
        out
    }

    /// General HRD with neither NAL nor VCL flag set — only the three
    /// leading fields are consumed.
    #[test]
    fn general_hrd_without_nal_or_vcl_only_reads_header() {
        let mut bits = Vec::new();
        push_u(&mut bits, 1, 32); // num_units_in_tick
        push_u(&mut bits, 30, 32); // time_scale (must be > 0)
        push_u(&mut bits, 0, 1); // general_nal_hrd = 0
        push_u(&mut bits, 0, 1); // general_vcl_hrd = 0
        let bytes = pack(&bits);
        let mut br = BitReader::new(&bytes);
        let g = parse_general_timing_hrd_parameters(&mut br).unwrap();
        assert_eq!(g.num_units_in_tick, 1);
        assert_eq!(g.time_scale, 30);
        assert!(!g.general_nal_hrd_params_present_flag);
        assert!(!g.general_vcl_hrd_params_present_flag);
        assert_eq!(g.hrd_cpb_cnt_minus1, 0);
    }

    /// General HRD with NAL flag set — the conditional block must be
    /// walked. We omit the DU branch (flag=0) for simplicity.
    #[test]
    fn general_hrd_with_nal_flag_walks_conditional_block() {
        let mut bits = Vec::new();
        push_u(&mut bits, 1, 32); // num_units_in_tick
        push_u(&mut bits, 60000, 32); // time_scale
        push_u(&mut bits, 1, 1); // general_nal_hrd = 1
        push_u(&mut bits, 0, 1); // general_vcl_hrd = 0
        push_u(&mut bits, 0, 1); // general_same_pic_timing_in_all_ols_flag
        push_u(&mut bits, 0, 1); // general_du_hrd_params_present_flag
        push_u(&mut bits, 0b0011, 4); // bit_rate_scale = 3
        push_u(&mut bits, 0b0100, 4); // cpb_size_scale = 4
        push_ue(&mut bits, 2); // hrd_cpb_cnt_minus1 = 2
        let bytes = pack(&bits);
        let mut br = BitReader::new(&bytes);
        let g = parse_general_timing_hrd_parameters(&mut br).unwrap();
        assert_eq!(g.num_units_in_tick, 1);
        assert_eq!(g.time_scale, 60000);
        assert!(g.general_nal_hrd_params_present_flag);
        assert!(!g.general_vcl_hrd_params_present_flag);
        assert!(!g.general_du_hrd_params_present_flag);
        assert_eq!(g.bit_rate_scale, 3);
        assert_eq!(g.cpb_size_scale, 4);
        assert_eq!(g.hrd_cpb_cnt_minus1, 2);
    }

    /// Zero time_scale is rejected.
    #[test]
    fn general_hrd_rejects_zero_time_scale() {
        let mut bits = Vec::new();
        push_u(&mut bits, 0, 32);
        push_u(&mut bits, 0, 32);
        push_u(&mut bits, 0, 1);
        push_u(&mut bits, 0, 1);
        let bytes = pack(&bits);
        let mut br = BitReader::new(&bytes);
        assert!(parse_general_timing_hrd_parameters(&mut br).is_err());
    }

    /// OLS timing with no NAL/VCL HRD (both flags off) and a single
    /// sublayer: emits only `fixed_pic_rate_general_flag` +
    /// `fixed_pic_rate_within_cvs_flag` + optional elemental duration.
    #[test]
    fn ols_timing_single_sublayer_no_hrd() {
        let general = GeneralTimingHrdParameters {
            num_units_in_tick: 1,
            time_scale: 30,
            general_nal_hrd_params_present_flag: false,
            general_vcl_hrd_params_present_flag: false,
            ..Default::default()
        };
        let mut bits = Vec::new();
        // sublayer 0: fixed_pic_rate_general=1 → within_cvs inferred to 1
        // → elemental_duration_in_tc_minus1 = 0.
        push_u(&mut bits, 1, 1);
        push_ue(&mut bits, 0);
        let bytes = pack(&bits);
        let mut br = BitReader::new(&bytes);
        let o = parse_ols_timing_hrd_parameters(&mut br, &general, 0, 0).unwrap();
        assert_eq!(o.sublayers.len(), 1);
        assert!(o.sublayers[0].fixed_pic_rate_general_flag);
        assert!(o.sublayers[0].fixed_pic_rate_within_cvs_flag);
        assert_eq!(o.sublayers[0].elemental_duration_in_tc_minus1, 0);
        assert!(o.sublayers[0].nal_hrd.is_none());
        assert!(o.sublayers[0].vcl_hrd.is_none());
    }

    /// OLS timing with NAL HRD enabled and one CPB → one entry per
    /// sublayer carrying (bit_rate, cpb_size, cbr_flag).
    #[test]
    fn ols_timing_single_sublayer_nal_one_cpb() {
        let general = GeneralTimingHrdParameters {
            num_units_in_tick: 1,
            time_scale: 30,
            general_nal_hrd_params_present_flag: true,
            general_vcl_hrd_params_present_flag: false,
            general_same_pic_timing_in_all_ols_flag: false,
            general_du_hrd_params_present_flag: false,
            hrd_cpb_cnt_minus1: 0,
            ..Default::default()
        };
        let mut bits = Vec::new();
        // sublayer 0: fixed_pic_rate_general=0, within_cvs=0, and because
        // cpb_cnt_minus1 == 0 → low_delay_hrd_flag is emitted.
        push_u(&mut bits, 0, 1);
        push_u(&mut bits, 0, 1);
        push_u(&mut bits, 1, 1); // low_delay_hrd_flag = 1
                                 // NAL sublayer_hrd_parameters: one CPB, no DU branch.
        push_ue(&mut bits, 17); // bit_rate_value_minus1
        push_ue(&mut bits, 33); // cpb_size_value_minus1
        push_u(&mut bits, 1, 1); // cbr_flag
        let bytes = pack(&bits);
        let mut br = BitReader::new(&bytes);
        let o = parse_ols_timing_hrd_parameters(&mut br, &general, 0, 0).unwrap();
        assert_eq!(o.sublayers.len(), 1);
        let s = &o.sublayers[0];
        assert!(!s.fixed_pic_rate_general_flag);
        assert!(!s.fixed_pic_rate_within_cvs_flag);
        assert!(s.low_delay_hrd_flag);
        let nal = s.nal_hrd.as_ref().unwrap();
        assert_eq!(nal.entries.len(), 1);
        assert_eq!(nal.entries[0].bit_rate_value_minus1, 17);
        assert_eq!(nal.entries[0].cpb_size_value_minus1, 33);
        assert!(nal.entries[0].cbr_flag);
    }
}
