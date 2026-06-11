//! VVC Luma Mapping with Chroma Scaling data syntax parser — §7.3.2.19 /
//! §7.4.3.19.
//!
//! Scope (round 278): the `lmcs_data()` **field walk + parse-time range
//! validation** carried in an LMCS APS (`aps_params_type == LMCS_APS`),
//! plus the two pure-data sign folds that need no SPS context:
//! eq. 94 (`lmcsDeltaCW[ i ]`) and eq. 99 (`lmcsDeltaCrs`).
//!
//! Deliberately **not** here yet (BitDepth-dependent §7.4.3.19
//! derivations + the §8.7.4.2 / §8.7.5.2 reshaping processes; follow-up
//! rounds): eq. 93 `OrgCW`, eq. 95 `lmcsCW[ i ]` + its
//! `OrgCW >> 3 ..= (OrgCW << 3) − 1` conformance band, the eq. 96
//! `Σ lmcsCW[ i ] <= (1 << BitDepth) − 1` budget, eqs. 97 / 98
//! `InputPivot` / `LmcsPivot` / `ScaleCoeff` / `InvScaleCoeff`, the
//! `LmcsPivot` bin-crossing conformance clause, the
//! `lmcsCW[ i ] + lmcsDeltaCrs` joint band, and eq. 100
//! `ChromaScaleCoeff`. Those all need `BitDepth` (an SPS-side quantity
//! the APS cannot see), so they belong to the picture-level fuse that
//! binds an LMCS APS to an active SPS.

use oxideav_core::{Error, Result};

use crate::bitreader::BitReader;

/// Number of LMCS codeword bins — the §7.3.2.19 loop and every
/// §7.4.3.19 derivation run over bin indices `0..=15`.
pub const LMCS_NUM_BINS: usize = 16;

/// Decoded `lmcs_data()` payload — §7.3.2.19 / §7.4.3.19.
///
/// Bins outside `lmcs_min_bin_idx ..= lmcs_max_bin_idx()` keep the
/// §7.4.3.19 inferred defaults (`lmcs_delta_abs_cw = 0`,
/// `lmcs_delta_sign_cw_flag = 0`), matching the "set equal 0" arms of
/// the `lmcsCW[ i ]` derivation.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LmcsData {
    /// `lmcs_min_bin_idx` — minimum bin index used in the LMCS
    /// construction process. Range 0..=15 (§7.4.3.19).
    pub lmcs_min_bin_idx: u8,
    /// `lmcs_delta_max_bin_idx` — delta between 15 and the maximum bin
    /// index `LmcsMaxBinIdx`. Range 0..=15 (§7.4.3.19).
    pub lmcs_delta_max_bin_idx: u8,
    /// `lmcs_delta_cw_prec_minus1` — plus 1 gives the bit width of each
    /// `lmcs_delta_abs_cw[ i ]` field. Range 0..=14 (§7.4.3.19).
    pub lmcs_delta_cw_prec_minus1: u8,
    /// `lmcs_delta_abs_cw[ i ]` — absolute delta codeword value for the
    /// i-th bin. Signalled as `u(lmcs_delta_cw_prec_minus1 + 1)` for
    /// `i = lmcs_min_bin_idx ..= LmcsMaxBinIdx`; 0 elsewhere.
    pub lmcs_delta_abs_cw: [u32; LMCS_NUM_BINS],
    /// `lmcs_delta_sign_cw_flag[ i ]` — sign of `lmcsDeltaCW[ i ]`
    /// (`false` ⇒ positive). Inferred `false` when not present
    /// (§7.4.3.19), i.e. whenever `lmcs_delta_abs_cw[ i ] == 0`.
    pub lmcs_delta_sign_cw_flag: [bool; LMCS_NUM_BINS],
    /// `lmcs_delta_abs_crs` — absolute codeword value of `lmcsDeltaCrs`
    /// (`u(3)`, chroma-residual-scaling offset). Inferred 0 when not
    /// present, i.e. when `aps_chroma_present_flag == 0` (§7.4.3.19).
    pub lmcs_delta_abs_crs: u8,
    /// `lmcs_delta_sign_crs_flag` — sign of `lmcsDeltaCrs`. Inferred
    /// `false` when not present (§7.4.3.19).
    pub lmcs_delta_sign_crs_flag: bool,
}

impl LmcsData {
    /// `LmcsMaxBinIdx = 15 − lmcs_delta_max_bin_idx` (§7.4.3.19).
    pub fn lmcs_max_bin_idx(&self) -> u8 {
        15 - self.lmcs_delta_max_bin_idx
    }

    /// Bit width of each `lmcs_delta_abs_cw[ i ]` field:
    /// `lmcs_delta_cw_prec_minus1 + 1` (§7.4.3.19). Always in 1..=15.
    pub fn delta_cw_bit_width(&self) -> u32 {
        u32::from(self.lmcs_delta_cw_prec_minus1) + 1
    }

    /// Eq. 94: `lmcsDeltaCW[ i ] =
    /// ( 1 − 2 * lmcs_delta_sign_cw_flag[ i ] ) * lmcs_delta_abs_cw[ i ]`.
    ///
    /// Valid for any `i < 16`; bins outside the signalled range fold to
    /// 0 via their inferred defaults.
    pub fn lmcs_delta_cw(&self, i: usize) -> i32 {
        let sign = if self.lmcs_delta_sign_cw_flag[i] {
            -1
        } else {
            1
        };
        sign * self.lmcs_delta_abs_cw[i] as i32
    }

    /// Eq. 99: `lmcsDeltaCrs =
    /// ( 1 − 2 * lmcs_delta_sign_crs_flag ) * lmcs_delta_abs_crs`.
    pub fn lmcs_delta_crs(&self) -> i32 {
        let sign = if self.lmcs_delta_sign_crs_flag { -1 } else { 1 };
        sign * i32::from(self.lmcs_delta_abs_crs)
    }
}

/// Parse a §7.3.2.19 `lmcs_data()` payload from the bit position the
/// reader currently sits at (immediately after the APS header fields
/// when invoked from `adaptation_parameter_set_rbsp()`).
///
/// `aps_chroma_present_flag` is the gating APS-header flag for the
/// chroma-residual-scaling tail (`lmcs_delta_abs_crs` /
/// `lmcs_delta_sign_crs_flag`).
///
/// Parse-time validation (§7.4.3.19):
/// * `lmcs_min_bin_idx` shall be in 0..=15;
/// * `lmcs_delta_max_bin_idx` shall be in 0..=15;
/// * `LmcsMaxBinIdx (= 15 − lmcs_delta_max_bin_idx)` shall be `>=`
///   `lmcs_min_bin_idx`;
/// * `lmcs_delta_cw_prec_minus1` shall be in 0..=14.
pub fn parse_lmcs_data(br: &mut BitReader<'_>, aps_chroma_present_flag: bool) -> Result<LmcsData> {
    let mut out = LmcsData::default();

    let lmcs_min_bin_idx = br.ue()?;
    if lmcs_min_bin_idx > 15 {
        return Err(Error::invalid(format!(
            "h266 LMCS: lmcs_min_bin_idx out of range (expected 0..=15, got {lmcs_min_bin_idx})"
        )));
    }
    out.lmcs_min_bin_idx = lmcs_min_bin_idx as u8;

    let lmcs_delta_max_bin_idx = br.ue()?;
    if lmcs_delta_max_bin_idx > 15 {
        return Err(Error::invalid(format!(
            "h266 LMCS: lmcs_delta_max_bin_idx out of range (expected 0..=15, got {lmcs_delta_max_bin_idx})"
        )));
    }
    out.lmcs_delta_max_bin_idx = lmcs_delta_max_bin_idx as u8;

    // §7.4.3.19: LmcsMaxBinIdx shall be >= lmcs_min_bin_idx.
    let max_bin_idx = out.lmcs_max_bin_idx();
    if max_bin_idx < out.lmcs_min_bin_idx {
        return Err(Error::invalid(format!(
            "h266 LMCS: LmcsMaxBinIdx ({max_bin_idx}) < lmcs_min_bin_idx ({})",
            out.lmcs_min_bin_idx
        )));
    }

    let lmcs_delta_cw_prec_minus1 = br.ue()?;
    if lmcs_delta_cw_prec_minus1 > 14 {
        return Err(Error::invalid(format!(
            "h266 LMCS: lmcs_delta_cw_prec_minus1 out of range (expected 0..=14, got {lmcs_delta_cw_prec_minus1})"
        )));
    }
    out.lmcs_delta_cw_prec_minus1 = lmcs_delta_cw_prec_minus1 as u8;

    let width = out.delta_cw_bit_width();
    for i in usize::from(out.lmcs_min_bin_idx)..=usize::from(max_bin_idx) {
        out.lmcs_delta_abs_cw[i] = br.u(width)?;
        if out.lmcs_delta_abs_cw[i] > 0 {
            out.lmcs_delta_sign_cw_flag[i] = br.u1()? == 1;
        }
    }

    if aps_chroma_present_flag {
        out.lmcs_delta_abs_crs = br.u(3)? as u8;
        if out.lmcs_delta_abs_crs > 0 {
            out.lmcs_delta_sign_crs_flag = br.u1()? == 1;
        }
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoder::BitWriter;

    /// Encoder-mirror helper: emit one `lmcs_data()` payload bin-for-bin
    /// per the §7.3.2.19 listing.
    fn write_lmcs_data(bw: &mut BitWriter, d: &LmcsData, aps_chroma_present_flag: bool) {
        bw.write_ue(u32::from(d.lmcs_min_bin_idx));
        bw.write_ue(u32::from(d.lmcs_delta_max_bin_idx));
        bw.write_ue(u32::from(d.lmcs_delta_cw_prec_minus1));
        let width = u32::from(d.lmcs_delta_cw_prec_minus1) + 1;
        for i in usize::from(d.lmcs_min_bin_idx)..=usize::from(15 - d.lmcs_delta_max_bin_idx) {
            bw.write_bits(d.lmcs_delta_abs_cw[i], width);
            if d.lmcs_delta_abs_cw[i] > 0 {
                bw.write_bit(u8::from(d.lmcs_delta_sign_cw_flag[i]));
            }
        }
        if aps_chroma_present_flag {
            bw.write_bits(u32::from(d.lmcs_delta_abs_crs), 3);
            if d.lmcs_delta_abs_crs > 0 {
                bw.write_bit(u8::from(d.lmcs_delta_sign_crs_flag));
            }
        }
    }

    fn round_trip(d: &LmcsData, chroma: bool) -> LmcsData {
        let mut bw = BitWriter::new();
        write_lmcs_data(&mut bw, d, chroma);
        bw.rbsp_trailing_bits();
        let bytes = bw.into_bytes();
        let mut br = BitReader::new(&bytes);
        parse_lmcs_data(&mut br, chroma).expect("lmcs_data must round-trip")
    }

    #[test]
    fn all_defaults_full_bin_range_no_chroma() {
        // min = 0, delta_max = 0 (LmcsMaxBinIdx = 15), prec_minus1 = 0,
        // every lmcs_delta_abs_cw = 0 (1-bit fields, signs absent).
        let d = LmcsData::default();
        let got = round_trip(&d, false);
        assert_eq!(got, d);
        assert_eq!(got.lmcs_max_bin_idx(), 15);
        assert_eq!(got.delta_cw_bit_width(), 1);
        assert_eq!(got.lmcs_delta_crs(), 0);
        for i in 0..LMCS_NUM_BINS {
            assert_eq!(got.lmcs_delta_cw(i), 0);
        }
    }

    #[test]
    fn narrowed_bin_range_round_trip() {
        // min = 2, delta_max = 5 (LmcsMaxBinIdx = 10): only bins 2..=10
        // are signalled; bins outside keep inferred zeros.
        let mut d = LmcsData {
            lmcs_min_bin_idx: 2,
            lmcs_delta_max_bin_idx: 5,
            lmcs_delta_cw_prec_minus1: 3, // 4-bit abs fields
            ..Default::default()
        };
        for i in 2..=10usize {
            d.lmcs_delta_abs_cw[i] = (i as u32) % 16;
            d.lmcs_delta_sign_cw_flag[i] = d.lmcs_delta_abs_cw[i] > 0 && i % 2 == 0;
        }
        let got = round_trip(&d, false);
        assert_eq!(got, d);
        // Eq. 94 spot checks: bin 3 (abs 3, sign 0) → +3; bin 4 (abs 4,
        // sign 1) → −4; bin 0 unsignalled → 0.
        assert_eq!(got.lmcs_delta_cw(3), 3);
        assert_eq!(got.lmcs_delta_cw(4), -4);
        assert_eq!(got.lmcs_delta_cw(0), 0);
    }

    #[test]
    fn single_bin_range_min_equals_max() {
        // min = 7, delta_max = 8 → LmcsMaxBinIdx = 7: exactly one bin.
        let mut d = LmcsData {
            lmcs_min_bin_idx: 7,
            lmcs_delta_max_bin_idx: 8,
            lmcs_delta_cw_prec_minus1: 0,
            ..Default::default()
        };
        d.lmcs_delta_abs_cw[7] = 1;
        d.lmcs_delta_sign_cw_flag[7] = true;
        let got = round_trip(&d, false);
        assert_eq!(got, d);
        assert_eq!(got.lmcs_max_bin_idx(), 7);
        assert_eq!(got.lmcs_delta_cw(7), -1);
    }

    #[test]
    fn max_precision_abs_fields() {
        // prec_minus1 = 14 → 15-bit abs fields up to 2^15 − 1.
        let mut d = LmcsData {
            lmcs_min_bin_idx: 15,
            lmcs_delta_max_bin_idx: 0,
            lmcs_delta_cw_prec_minus1: 14,
            ..Default::default()
        };
        d.lmcs_delta_abs_cw[15] = (1 << 15) - 1;
        d.lmcs_delta_sign_cw_flag[15] = true;
        let got = round_trip(&d, false);
        assert_eq!(got, d);
        assert_eq!(got.delta_cw_bit_width(), 15);
        assert_eq!(got.lmcs_delta_cw(15), -((1 << 15) - 1));
    }

    #[test]
    fn chroma_tail_round_trip_both_signs() {
        for (abs, sign, want) in [
            (0u8, false, 0i32),
            (3, false, 3),
            (5, true, -5),
            (7, true, -7),
        ] {
            let d = LmcsData {
                lmcs_delta_abs_crs: abs,
                lmcs_delta_sign_crs_flag: sign,
                ..Default::default()
            };
            let got = round_trip(&d, true);
            assert_eq!(got, d, "abs={abs} sign={sign}");
            assert_eq!(
                got.lmcs_delta_crs(),
                want,
                "eq. 99 fold for abs={abs} sign={sign}"
            );
        }
    }

    #[test]
    fn chroma_tail_absent_when_flag_zero() {
        // Same payload parsed with aps_chroma_present_flag = 0 must leave
        // the §7.4.3.19 inferred defaults and consume no chroma bits.
        let d = LmcsData::default();
        let mut bw = BitWriter::new();
        write_lmcs_data(&mut bw, &d, false);
        bw.rbsp_trailing_bits();
        let bytes = bw.into_bytes();
        let mut br = BitReader::new(&bytes);
        let pos_before = br.bit_position();
        let got = parse_lmcs_data(&mut br, false).unwrap();
        assert_eq!(got.lmcs_delta_abs_crs, 0);
        assert!(!got.lmcs_delta_sign_crs_flag);
        // 3 ue(v) zeros (1 bit each) + 16 one-bit abs fields = 19 bits.
        assert_eq!(br.bit_position() - pos_before, 19);
    }

    #[test]
    fn rejects_min_bin_idx_out_of_range() {
        // ue(v) = 16 > 15.
        let mut bw = BitWriter::new();
        bw.write_ue(16);
        bw.rbsp_trailing_bits();
        let bytes = bw.into_bytes();
        let mut br = BitReader::new(&bytes);
        assert!(parse_lmcs_data(&mut br, false).is_err());
    }

    #[test]
    fn rejects_delta_max_bin_idx_out_of_range() {
        let mut bw = BitWriter::new();
        bw.write_ue(0); // lmcs_min_bin_idx
        bw.write_ue(16); // lmcs_delta_max_bin_idx > 15
        bw.rbsp_trailing_bits();
        let bytes = bw.into_bytes();
        let mut br = BitReader::new(&bytes);
        assert!(parse_lmcs_data(&mut br, false).is_err());
    }

    #[test]
    fn rejects_max_bin_idx_below_min_bin_idx() {
        // min = 9, delta_max = 8 → LmcsMaxBinIdx = 7 < 9.
        let mut bw = BitWriter::new();
        bw.write_ue(9);
        bw.write_ue(8);
        bw.rbsp_trailing_bits();
        let bytes = bw.into_bytes();
        let mut br = BitReader::new(&bytes);
        assert!(parse_lmcs_data(&mut br, false).is_err());
    }

    #[test]
    fn accepts_max_bin_idx_equal_to_min_bin_idx_boundary() {
        // min = 8, delta_max = 7 → LmcsMaxBinIdx = 8 == min: allowed
        // (the §7.4.3.19 constraint is >=, not >).
        let mut bw = BitWriter::new();
        bw.write_ue(8);
        bw.write_ue(7);
        bw.write_ue(0); // prec_minus1 = 0 → 1-bit abs
        bw.write_bit(0); // lmcs_delta_abs_cw[8] = 0
        bw.rbsp_trailing_bits();
        let bytes = bw.into_bytes();
        let mut br = BitReader::new(&bytes);
        let got = parse_lmcs_data(&mut br, false).unwrap();
        assert_eq!(got.lmcs_min_bin_idx, 8);
        assert_eq!(got.lmcs_max_bin_idx(), 8);
    }

    #[test]
    fn rejects_cw_prec_minus1_out_of_range() {
        let mut bw = BitWriter::new();
        bw.write_ue(0);
        bw.write_ue(0);
        bw.write_ue(15); // > 14
        bw.rbsp_trailing_bits();
        let bytes = bw.into_bytes();
        let mut br = BitReader::new(&bytes);
        assert!(parse_lmcs_data(&mut br, false).is_err());
    }

    #[test]
    fn rejects_truncated_abs_cw_run() {
        // Header promises 16 4-bit abs fields but the buffer ends early.
        let mut bw = BitWriter::new();
        bw.write_ue(0); // min = 0
        bw.write_ue(0); // delta_max = 0 → 16 bins
        bw.write_ue(3); // prec_minus1 = 3 → 4-bit abs
        bw.write_bits(0, 4); // only one of the 16 abs fields
        let bytes = bw.into_bytes(); // no trailing fill beyond byte pad
        let mut br = BitReader::new(&bytes);
        assert!(parse_lmcs_data(&mut br, false).is_err());
    }

    #[test]
    fn sign_flag_inferred_zero_when_abs_zero() {
        // A zero abs value carries no sign bit; the parser must leave the
        // inferred false rather than consuming the next bin's bits.
        let mut d = LmcsData {
            lmcs_min_bin_idx: 0,
            lmcs_delta_max_bin_idx: 14, // bins 0..=1
            lmcs_delta_cw_prec_minus1: 1,
            ..Default::default()
        };
        d.lmcs_delta_abs_cw[0] = 0; // no sign bit
        d.lmcs_delta_abs_cw[1] = 2;
        d.lmcs_delta_sign_cw_flag[1] = true;
        let got = round_trip(&d, false);
        assert_eq!(got, d);
        assert!(!got.lmcs_delta_sign_cw_flag[0]);
        assert_eq!(got.lmcs_delta_cw(1), -2);
    }

    #[test]
    fn exhaustive_bin_window_round_trip() {
        // Every legal (min, delta_max) window at prec_minus1 = 2 with a
        // deterministic abs/sign fill.
        for min in 0..=15u8 {
            for delta_max in 0..=(15 - min) {
                let mut d = LmcsData {
                    lmcs_min_bin_idx: min,
                    lmcs_delta_max_bin_idx: delta_max,
                    lmcs_delta_cw_prec_minus1: 2,
                    ..Default::default()
                };
                for i in usize::from(min)..=usize::from(15 - delta_max) {
                    d.lmcs_delta_abs_cw[i] = (i as u32 * 3) % 8;
                    d.lmcs_delta_sign_cw_flag[i] = d.lmcs_delta_abs_cw[i] > 0 && i % 3 == 0;
                }
                let got = round_trip(&d, (min + delta_max) % 2 == 0);
                assert_eq!(got, d, "window min={min} delta_max={delta_max}");
            }
        }
    }
}
