//! VVC Luma Mapping with Chroma Scaling data syntax parser — §7.3.2.19 /
//! §7.4.3.19.
//!
//! Scope (round 278): the `lmcs_data()` **field walk + parse-time range
//! validation** carried in an LMCS APS (`aps_params_type == LMCS_APS`),
//! plus the two pure-data sign folds that need no SPS context:
//! eq. 94 (`lmcsDeltaCW[ i ]`) and eq. 99 (`lmcsDeltaCrs`).
//!
//! Round 281 adds the **BitDepth-dependent §7.4.3.19 derivations** as
//! [`LmcsDerived`] / [`derive_lmcs`] — the picture-level fuse that binds
//! an LMCS APS to an active SPS' `BitDepth` (eq. 38): eq. 93 `OrgCW`,
//! eq. 95 `lmcsCW[ i ]` + its `OrgCW >> 3 ..= (OrgCW << 3) − 1`
//! conformance band, the eq. 96 `Σ lmcsCW[ i ] <= (1 << BitDepth) − 1`
//! budget, eq. 97 `InputPivot`, eq. 98 `LmcsPivot` / `ScaleCoeff` /
//! `InvScaleCoeff`, the `LmcsPivot` bin-crossing conformance clause, the
//! `lmcsCW[ i ] + lmcsDeltaCrs` joint band, and eq. 100
//! `ChromaScaleCoeff`.
//!
//! Round 290 lands the **sample-domain LMCS processes** that consume the
//! [`LmcsDerived`] arrays, as pure per-sample folds on [`LmcsDerived`]:
//! the §8.7.5.2 forward luma mapping (eq. 1213 `idxY` /
//! `predMapSamples`), the §8.8.2.3 piecewise-function-index
//! identification (eq. 1224 `idxYInv`), the §8.8.2.2 inverse luma
//! mapping (eqs. 1222 / 1223 `invSample` / `invLumaSample`), the
//! §8.7.5.3 chroma residual scaling (eq. 1218 `varScale` lookup +
//! eqs. 1219 / 1220 `Clip3` clamp + `Sign`/`Abs` scale fold).
//!
//! Deliberately **not** here yet (follow-up rounds): the picture-level
//! orchestration — the §8.7.5.1 `sh_lmcs_used_flag` / `CuPredMode` /
//! `ciip_flag` gating, the §8.7.5.3 step-1 `invAvgLuma` neighbour
//! averaging (which needs §6.4.4 availability + the partially-
//! reconstructed luma plane), and the §8.8.2.1 whole-picture inverse
//! pass — all of which the CTU walker drives once it wires these
//! per-sample folds in.

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
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
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

    /// Run the BitDepth-dependent §7.4.3.19 derivations against an
    /// active SPS' `BitDepth` (eq. 38). See [`derive_lmcs`].
    pub fn derive(&self, bit_depth: u32) -> Result<LmcsDerived> {
        derive_lmcs(self, bit_depth)
    }
}

/// BitDepth-dependent §7.4.3.19 derived variables for one LMCS APS
/// bound to an active SPS — the inputs of the §8.7.4 luma mapping and
/// §8.7.5.3 chroma-residual scaling processes.
///
/// Produced by [`derive_lmcs`] / [`LmcsData::derive`]; every conformance
/// constraint the §7.4.3.19 semantics attach to these variables is
/// checked at derivation time, so a successfully constructed value
/// carries spec-conforming arrays.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LmcsDerived {
    /// `BitDepth` the derivation ran against (eq. 38; 8..=16).
    pub bit_depth: u32,
    /// Eq. 93: `OrgCW = ( 1 << BitDepth ) / 16`.
    pub org_cw: u32,
    /// Eq. 95 (and its two "set equal 0" arms): per-bin codeword count.
    /// `lmcsCW[ i ] = OrgCW + lmcsDeltaCW[ i ]` inside the signalled
    /// window, 0 outside.
    pub lmcs_cw: [u32; LMCS_NUM_BINS],
    /// Eq. 97: `InputPivot[ i ] = i * OrgCW`, `i = 0..15`.
    pub input_pivot: [u32; LMCS_NUM_BINS],
    /// Eq. 98: `LmcsPivot[ 0 ] = 0`;
    /// `LmcsPivot[ i + 1 ] = LmcsPivot[ i ] + lmcsCW[ i ]`, `i = 0..15`.
    pub lmcs_pivot: [u32; LMCS_NUM_BINS + 1],
    /// Eq. 98: `ScaleCoeff[ i ] = ( lmcsCW[ i ] * ( 1 << 11 ) +
    /// ( 1 << ( Log2( OrgCW ) − 1 ) ) ) >> Log2( OrgCW )`.
    pub scale_coeff: [u32; LMCS_NUM_BINS],
    /// Eq. 98: `InvScaleCoeff[ i ] = OrgCW * ( 1 << 11 ) / lmcsCW[ i ]`
    /// when `lmcsCW[ i ] != 0`, else 0.
    pub inv_scale_coeff: [u32; LMCS_NUM_BINS],
    /// Eq. 99 fold carried over from the parsed payload.
    pub lmcs_delta_crs: i32,
    /// Eq. 100: `ChromaScaleCoeff[ i ] = OrgCW * ( 1 << 11 ) /
    /// ( lmcsCW[ i ] + lmcsDeltaCrs )` when `lmcsCW[ i ] != 0`, else
    /// `1 << 11`.
    pub chroma_scale_coeff: [u32; LMCS_NUM_BINS],
}

impl LmcsDerived {
    /// `Clip1( x ) = Clip3( 0, ( 1 << BitDepth ) − 1, x )` (eq. 3) for
    /// the `BitDepth` this derivation was built against.
    fn clip1(&self, x: i64) -> u32 {
        let hi = (1i64 << self.bit_depth) - 1;
        x.clamp(0, hi) as u32
    }

    /// `Log2( OrgCW ) = BitDepth − 4` (eq. 93 gives `OrgCW =
    /// ( 1 << BitDepth ) / 16`, a power of two, so its log2 is exact).
    fn log2_org_cw(&self) -> u32 {
        self.bit_depth - 4
    }

    /// §8.7.5.2 eq. 1213 — forward-map one predicted luma sample for an
    /// inter CU that uses LMCS (`MODE_INTER`, `ciip_flag == 0`).
    ///
    /// `predSample` is the prediction-domain luma sample (range
    /// `0..=( 1 << BitDepth ) − 1`); the returned value is the mapped
    /// predicted sample `predMapSamples[ i ][ j ]`. For the four
    /// §8.7.5.2 carry-through modes (`MODE_INTRA` / `MODE_IBC` /
    /// `MODE_PLT`, and CIIP inter) the caller passes the prediction
    /// through unchanged instead of calling this fold.
    ///
    /// `idxY = predSample >> Log2( OrgCW )` is in `0..=15` for any
    /// in-range sample because `OrgCW = ( 1 << BitDepth ) / 16`, so no
    /// clamp on the index is needed.
    ///
    /// Returns the *unclamped* `predMapSamples` intermediate — eq. 1213
    /// applies no `Clip1`; that happens only at eq. 1214 once the
    /// residual is added (see [`Self::reconstruct_mapped_luma_sample`]).
    pub fn forward_map_luma_sample(&self, pred_sample: u32) -> i64 {
        let idx_y = (pred_sample >> self.log2_org_cw()) as usize;
        // eq. 1213. ScaleCoeff <= 16384 and (predSample − InputPivot) is
        // bounded by OrgCW << 3, so the product fits comfortably in i64.
        let prod = i64::from(self.scale_coeff[idx_y])
            * (i64::from(pred_sample) - i64::from(self.input_pivot[idx_y]))
            + (1 << 10);
        i64::from(self.lmcs_pivot[idx_y]) + (prod >> 11)
    }

    /// §8.7.5.2 eq. 1214 — combine a mapped predicted luma sample with
    /// its residual to produce a reconstructed luma sample:
    /// `Clip1( predMapSample + resSample )`.
    pub fn reconstruct_mapped_luma_sample(&self, pred_map_sample: i64, res_sample: i64) -> u32 {
        self.clip1(pred_map_sample + res_sample)
    }

    /// §8.8.2.3 eq. 1224 — identify the piece-wise function index
    /// `idxYInv` a (reconstructed-domain) luma sample belongs to.
    ///
    /// Scans `lmcs_min_bin_idx ..= LmcsMaxBinIdx` for the first
    /// `LmcsPivot[ idxYInv + 1 ]` strictly greater than `lumaSample`,
    /// then clamps the result to 15.
    pub fn idx_y_inv(&self, luma_sample: u32, lmcs_min_bin_idx: u8, lmcs_max_bin_idx: u8) -> usize {
        let min = usize::from(lmcs_min_bin_idx);
        let max = usize::from(lmcs_max_bin_idx);
        let mut idx = min;
        while idx <= max {
            if luma_sample < self.lmcs_pivot[idx + 1] {
                break;
            }
            idx += 1;
        }
        idx.min(15)
    }

    /// §8.8.2.2 eqs. 1222 / 1223 — inverse-map one reconstructed luma
    /// sample (the in-loop pre-filter pass of §8.8.2.1).
    ///
    /// `idxYInv` is the §8.8.2.3 piece index for `lumaSample` (obtain it
    /// via [`Self::idx_y_inv`]). Returns
    /// `Clip1( InputPivot[ idxYInv ] +
    ///   ( ( InvScaleCoeff[ idxYInv ] *
    ///     ( lumaSample − LmcsPivot[ idxYInv ] ) + ( 1 << 10 ) ) >> 11 ) )`.
    pub fn inverse_map_luma_sample(&self, luma_sample: u32, idx_y_inv: usize) -> u32 {
        // eq. 1222.
        let prod = i64::from(self.inv_scale_coeff[idx_y_inv])
            * (i64::from(luma_sample) - i64::from(self.lmcs_pivot[idx_y_inv]))
            + (1 << 10);
        let inv_sample = i64::from(self.input_pivot[idx_y_inv]) + (prod >> 11);
        // eq. 1223.
        self.clip1(inv_sample)
    }

    /// §8.7.5.3 eq. 1218 — `varScale = ChromaScaleCoeff[ idxYInv ]`,
    /// the chroma-residual scale for a chroma block whose collocated
    /// average reconstructed luma falls in piece `idxYInv`.
    pub fn chroma_var_scale(&self, idx_y_inv: usize) -> u32 {
        self.chroma_scale_coeff[idx_y_inv]
    }

    /// §8.7.5.3 eqs. 1219 / 1220 — reconstruct one chroma sample when
    /// chroma residual scaling applies (the `tuCbfChroma == 1 ||
    /// cu_act_enabled_flag == 1` branch).
    ///
    /// `varScale` is [`Self::chroma_var_scale`] for the block.
    /// The residual is first clamped to `Clip3( −( 1 << BitDepth ),
    /// ( 1 << BitDepth ) − 1, resSample )` (eq. 1219), then the scaled
    /// residual `Sign( res ) * ( ( Abs( res ) * varScale +
    /// ( 1 << 10 ) ) >> 11 )` is added to the prediction and `Clip1`'d
    /// (eq. 1220).
    pub fn scale_chroma_residual_sample(
        &self,
        pred_sample: i64,
        res_sample: i64,
        var_scale: u32,
    ) -> u32 {
        // eq. 1219.
        let lo = -(1i64 << self.bit_depth);
        let hi = (1i64 << self.bit_depth) - 1;
        let res = res_sample.clamp(lo, hi);
        // eq. 1220: Sign( res ) * ( ( Abs( res ) * varScale + 2^10 ) >> 11 ).
        let scaled = if res == 0 {
            0
        } else {
            let mag = (res.unsigned_abs() * u64::from(var_scale) + (1 << 10)) >> 11;
            if res < 0 {
                -(mag as i64)
            } else {
                mag as i64
            }
        };
        self.clip1(pred_sample + scaled)
    }
}

/// Run the BitDepth-dependent §7.4.3.19 derivations (eqs. 93 / 95 – 98 /
/// 100) for a parsed [`LmcsData`] payload bound to an active SPS'
/// `BitDepth` (eq. 38: `8 + sps_bitdepth_minus8`, so 8..=16 given the
/// §7.4.3.4 `sps_bitdepth_minus8` 0..=8 range).
///
/// The §7.4.3.19 bitstream-conformance constraints that only become
/// checkable once `BitDepth` is known are enforced here and surface as
/// `Error::invalid`:
/// * eq. 95 band — `lmcsCW[ i ]` shall be in
///   `OrgCW >> 3 ..= ( OrgCW << 3 ) − 1` for every signalled bin;
/// * eq. 96 budget — `Σ lmcsCW[ i ] <= ( 1 << BitDepth ) − 1`;
/// * the eq. 98 follow-on clause — for `i =
///   lmcs_min_bin_idx..LmcsMaxBinIdx`, when `LmcsPivot[ i ]` is not a
///   multiple of `1 << ( BitDepth − 5 )`, `LmcsPivot[ i ] >>
///   ( BitDepth − 5 )` shall differ from `LmcsPivot[ i + 1 ] >>
///   ( BitDepth − 5 )`;
/// * the eq. 99 follow-on joint band — when `lmcsCW[ i ] != 0`,
///   `lmcsCW[ i ] + lmcsDeltaCrs` shall be in
///   `OrgCW >> 3 ..= ( OrgCW << 3 ) − 1` (this also keeps the eq. 100
///   divisor strictly positive).
pub fn derive_lmcs(data: &LmcsData, bit_depth: u32) -> Result<LmcsDerived> {
    if !(8..=16).contains(&bit_depth) {
        return Err(Error::invalid(format!(
            "h266 LMCS: BitDepth out of range (expected 8..=16, got {bit_depth})"
        )));
    }

    // Eq. 93. (1 << BitDepth) is a power of two >= 256, so the /16 is
    // exact: OrgCW = 1 << (BitDepth - 4) and Log2(OrgCW) = BitDepth - 4.
    let org_cw = (1u32 << bit_depth) / 16;
    let log2_org_cw = bit_depth - 4;
    let band_lo = org_cw >> 3;
    let band_hi = (org_cw << 3) - 1;

    let min = usize::from(data.lmcs_min_bin_idx);
    let max = usize::from(data.lmcs_max_bin_idx());

    // Eq. 95 with its two "set equal 0" arms + the per-bin band check.
    let mut lmcs_cw = [0u32; LMCS_NUM_BINS];
    for (i, cw) in lmcs_cw.iter_mut().enumerate().take(max + 1).skip(min) {
        let v = org_cw as i64 + i64::from(data.lmcs_delta_cw(i));
        if v < i64::from(band_lo) || v > i64::from(band_hi) {
            return Err(Error::invalid(format!(
                "h266 LMCS: lmcsCW[{i}] = {v} outside OrgCW >> 3 ..= (OrgCW << 3) - 1 \
                 ({band_lo}..={band_hi}) at BitDepth {bit_depth}"
            )));
        }
        *cw = v as u32;
    }

    // Eq. 96 codeword budget.
    let total: u32 = lmcs_cw.iter().sum();
    if total > (1u32 << bit_depth) - 1 {
        return Err(Error::invalid(format!(
            "h266 LMCS: sum of lmcsCW ({total}) exceeds (1 << BitDepth) - 1 ({})",
            (1u32 << bit_depth) - 1
        )));
    }

    // Eq. 97.
    let mut input_pivot = [0u32; LMCS_NUM_BINS];
    for (i, p) in input_pivot.iter_mut().enumerate() {
        *p = i as u32 * org_cw;
    }

    // Eq. 98 loop: pivots + forward/inverse luma scale coefficients.
    let mut lmcs_pivot = [0u32; LMCS_NUM_BINS + 1];
    let mut scale_coeff = [0u32; LMCS_NUM_BINS];
    let mut inv_scale_coeff = [0u32; LMCS_NUM_BINS];
    for i in 0..LMCS_NUM_BINS {
        lmcs_pivot[i + 1] = lmcs_pivot[i] + lmcs_cw[i];
        scale_coeff[i] = (lmcs_cw[i] * (1 << 11) + (1 << (log2_org_cw - 1))) >> log2_org_cw;
        // The checked_div zero arm is exactly the eq. 98
        // `if( lmcsCW[ i ] == 0 ) InvScaleCoeff[ i ] = 0` branch.
        inv_scale_coeff[i] = (org_cw * (1 << 11)).checked_div(lmcs_cw[i]).unwrap_or(0);
    }

    // Eq. 98 follow-on bin-crossing conformance clause.
    let shift = bit_depth - 5;
    for i in min..=max {
        if lmcs_pivot[i] % (1u32 << shift) != 0
            && (lmcs_pivot[i] >> shift) == (lmcs_pivot[i + 1] >> shift)
        {
            return Err(Error::invalid(format!(
                "h266 LMCS: LmcsPivot[{i}] = {} is not a multiple of 1 << (BitDepth - 5) and \
                 LmcsPivot[{i}] >> (BitDepth - 5) == LmcsPivot[{}] >> (BitDepth - 5) ({})",
                lmcs_pivot[i],
                i + 1,
                lmcs_pivot[i] >> shift
            )));
        }
    }

    // Eq. 99 follow-on joint band + eq. 100 chroma scale coefficients.
    let lmcs_delta_crs = data.lmcs_delta_crs();
    let mut chroma_scale_coeff = [0u32; LMCS_NUM_BINS];
    for i in 0..LMCS_NUM_BINS {
        chroma_scale_coeff[i] = if lmcs_cw[i] == 0 {
            1 << 11
        } else {
            let joint = i64::from(lmcs_cw[i]) + i64::from(lmcs_delta_crs);
            if joint < i64::from(band_lo) || joint > i64::from(band_hi) {
                return Err(Error::invalid(format!(
                    "h266 LMCS: lmcsCW[{i}] + lmcsDeltaCrs = {joint} outside \
                     OrgCW >> 3 ..= (OrgCW << 3) - 1 ({band_lo}..={band_hi}) at BitDepth {bit_depth}"
                )));
            }
            org_cw * (1 << 11) / joint as u32
        };
    }

    Ok(LmcsDerived {
        bit_depth,
        org_cw,
        lmcs_cw,
        input_pivot,
        lmcs_pivot,
        scale_coeff,
        inv_scale_coeff,
        lmcs_delta_crs,
        chroma_scale_coeff,
    })
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

    /// Test-fixture payload: full window at BitDepth 8 with a single
    /// −1 delta on bin 0 so the eq. 96 budget (255) is met exactly.
    fn near_identity_bd8() -> LmcsData {
        let mut d = LmcsData {
            lmcs_delta_cw_prec_minus1: 0,
            ..Default::default()
        };
        d.lmcs_delta_abs_cw[0] = 1;
        d.lmcs_delta_sign_cw_flag[0] = true;
        d
    }

    #[test]
    fn derive_org_cw_across_bit_depths() {
        // Eq. 93 at every legal BitDepth (§7.4.3.4: 8..=16), using a
        // single-bin window with a zero delta (sum = OrgCW <= budget).
        let d = LmcsData {
            lmcs_delta_max_bin_idx: 15, // window = bin 0 only
            ..Default::default()
        };
        for (bd, want) in [
            (8u32, 16u32),
            (9, 32),
            (10, 64),
            (11, 128),
            (12, 256),
            (13, 512),
            (14, 1024),
            (15, 2048),
            (16, 4096),
        ] {
            let got = derive_lmcs(&d, bd).unwrap();
            assert_eq!(got.org_cw, want, "OrgCW at BitDepth {bd}");
            assert_eq!(got.lmcs_cw[0], want);
            assert_eq!(
                got.scale_coeff[0],
                1 << 11,
                "identity bin scales to 1 << 11"
            );
            assert_eq!(got.inv_scale_coeff[0], 1 << 11);
        }
    }

    #[test]
    fn derive_rejects_bit_depth_out_of_range() {
        let d = near_identity_bd8();
        assert!(derive_lmcs(&d, 7).is_err());
        assert!(derive_lmcs(&d, 17).is_err());
    }

    #[test]
    fn derive_near_identity_full_window_bd8() {
        // OrgCW = 16; lmcsCW = [15, 16, 16, ...]; budget 255 met exactly.
        let got = near_identity_bd8().derive(8).unwrap();
        assert_eq!(got.org_cw, 16);
        assert_eq!(got.lmcs_cw[0], 15);
        for i in 1..LMCS_NUM_BINS {
            assert_eq!(got.lmcs_cw[i], 16);
        }
        assert_eq!(got.lmcs_cw.iter().sum::<u32>(), 255);
        // Eq. 97: InputPivot[ i ] = i * OrgCW.
        for i in 0..LMCS_NUM_BINS {
            assert_eq!(got.input_pivot[i], i as u32 * 16);
        }
        // Eq. 98 pivots: 0, 15, 31, ..., 255.
        assert_eq!(got.lmcs_pivot[0], 0);
        for i in 1..=LMCS_NUM_BINS {
            assert_eq!(got.lmcs_pivot[i], 15 + 16 * (i as u32 - 1));
        }
        // Eq. 98 scale coefficients, worked by hand at Log2(OrgCW) = 4:
        // bin 0: (15 * 2048 + 8) >> 4 = 1920; inv = 32768 / 15 = 2184.
        assert_eq!(got.scale_coeff[0], 1920);
        assert_eq!(got.inv_scale_coeff[0], 2184);
        for i in 1..LMCS_NUM_BINS {
            assert_eq!(got.scale_coeff[i], 2048);
            assert_eq!(got.inv_scale_coeff[i], 2048);
        }
        // Eq. 100 with lmcsDeltaCrs = 0 mirrors InvScaleCoeff here.
        assert_eq!(got.lmcs_delta_crs, 0);
        assert_eq!(got.chroma_scale_coeff[0], 2184);
        for i in 1..LMCS_NUM_BINS {
            assert_eq!(got.chroma_scale_coeff[i], 2048);
        }
    }

    #[test]
    fn derive_rejects_full_window_identity_budget() {
        // All-zero deltas over the full window put Σ lmcsCW at exactly
        // 1 << BitDepth — one codeword past the eq. 96 budget — at every
        // bit depth, so the all-default payload is non-conforming.
        let d = LmcsData::default();
        for bd in 8..=16 {
            let err = derive_lmcs(&d, bd).unwrap_err();
            assert!(
                err.to_string().contains("sum of lmcsCW"),
                "BitDepth {bd}: {err}"
            );
        }
    }

    #[test]
    fn derive_narrowed_window_zero_bins_bd10() {
        // BitDepth 10 (OrgCW = 64), window bins 1..=14, all deltas zero:
        // outside bins take the eq. 95 "set equal 0" arms.
        let d = LmcsData {
            lmcs_min_bin_idx: 1,
            lmcs_delta_max_bin_idx: 1,
            ..Default::default()
        };
        let got = d.derive(10).unwrap();
        assert_eq!(got.lmcs_cw[0], 0);
        assert_eq!(got.lmcs_cw[15], 0);
        for i in 1..=14 {
            assert_eq!(got.lmcs_cw[i], 64);
            assert_eq!(got.scale_coeff[i], 2048);
            assert_eq!(got.inv_scale_coeff[i], 2048);
            assert_eq!(got.chroma_scale_coeff[i], 2048);
        }
        // Zero bins: ScaleCoeff rounds (0 + 32) >> 6 to 0; InvScaleCoeff
        // takes the eq. 98 zero arm; ChromaScaleCoeff the eq. 100
        // 1 << 11 arm.
        for i in [0usize, 15] {
            assert_eq!(got.scale_coeff[i], 0);
            assert_eq!(got.inv_scale_coeff[i], 0);
            assert_eq!(got.chroma_scale_coeff[i], 1 << 11);
        }
        // Pivot stays flat across zero bins: 0, 0, 64, ..., 896, 896.
        assert_eq!(got.lmcs_pivot[1], 0);
        assert_eq!(got.lmcs_pivot[15], 896);
        assert_eq!(got.lmcs_pivot[16], 896);
    }

    #[test]
    fn derive_lmcs_cw_band_boundaries_bd8() {
        // Band at BitDepth 8 is OrgCW >> 3 ..= (OrgCW << 3) − 1 = 2..=127.
        let mut d = LmcsData {
            lmcs_delta_max_bin_idx: 15, // single-bin window
            lmcs_delta_cw_prec_minus1: 6,
            ..Default::default()
        };
        // lmcsCW = 16 − 14 = 2: lower boundary accepted.
        d.lmcs_delta_abs_cw[0] = 14;
        d.lmcs_delta_sign_cw_flag[0] = true;
        let got = d.derive(8).unwrap();
        assert_eq!(got.lmcs_cw[0], 2);
        assert_eq!(got.scale_coeff[0], 256); // (2 * 2048 + 8) >> 4
        assert_eq!(got.inv_scale_coeff[0], 16384); // 32768 / 2
                                                   // lmcsCW = 16 − 15 = 1 < 2: rejected.
        d.lmcs_delta_abs_cw[0] = 15;
        assert!(d.derive(8).is_err());
        // lmcsCW = 16 + 111 = 127: upper boundary accepted.
        d.lmcs_delta_abs_cw[0] = 111;
        d.lmcs_delta_sign_cw_flag[0] = false;
        let got = d.derive(8).unwrap();
        assert_eq!(got.lmcs_cw[0], 127);
        assert_eq!(got.scale_coeff[0], 16256); // (127 * 2048 + 8) >> 4
        assert_eq!(got.inv_scale_coeff[0], 258); // 32768 / 127
                                                 // lmcsCW = 16 + 112 = 128 > 127: rejected.
        d.lmcs_delta_abs_cw[0] = 112;
        assert!(d.derive(8).is_err());
    }

    #[test]
    fn derive_band_upper_boundary_bd16() {
        // BitDepth 16: OrgCW = 4096, band 512..=32767, budget 65535.
        let mut d = LmcsData {
            lmcs_delta_max_bin_idx: 15,
            lmcs_delta_cw_prec_minus1: 14,
            ..Default::default()
        };
        d.lmcs_delta_abs_cw[0] = 28671; // lmcsCW = 32767 = band max
        let got = d.derive(16).unwrap();
        assert_eq!(got.lmcs_cw[0], 32767);
        // (32767 * 2048 + 2048) >> 12 = 2^26 >> 12 = 16384.
        assert_eq!(got.scale_coeff[0], 16384);
        assert_eq!(got.inv_scale_coeff[0], 256); // 8388608 / 32767
        d.lmcs_delta_abs_cw[0] = 28672; // 32768: one past the band
        assert!(d.derive(16).is_err());
    }

    #[test]
    fn derive_pivot_bin_crossing_clause_bd8() {
        // Window bins 0..=1 with lmcsCW = [2, 2]: LmcsPivot[1] = 2 is not
        // a multiple of 1 << (BitDepth − 5) = 8 and LmcsPivot[1] >> 3 ==
        // LmcsPivot[2] >> 3 (both 0) → non-conforming per the eq. 98
        // follow-on clause.
        let mut d = LmcsData {
            lmcs_delta_max_bin_idx: 14,
            lmcs_delta_cw_prec_minus1: 6,
            ..Default::default()
        };
        d.lmcs_delta_abs_cw[0] = 14;
        d.lmcs_delta_sign_cw_flag[0] = true;
        d.lmcs_delta_abs_cw[1] = 14;
        d.lmcs_delta_sign_cw_flag[1] = true;
        let err = d.derive(8).unwrap_err();
        assert!(err.to_string().contains("LmcsPivot"), "{err}");
        // Same LmcsPivot[1] = 2 but lmcsCW[1] = 127 pushes LmcsPivot[2]
        // to 129 → the two >> 3 values differ (0 vs 16) → conforming.
        d.lmcs_delta_abs_cw[1] = 111;
        d.lmcs_delta_sign_cw_flag[1] = false;
        let got = d.derive(8).unwrap();
        assert_eq!(got.lmcs_pivot[1], 2);
        assert_eq!(got.lmcs_pivot[2], 129);
    }

    #[test]
    fn derive_chroma_joint_band_and_eq100_bd8() {
        // lmcsDeltaCrs = −7 on the near-identity payload: joint values
        // 15 − 7 = 8 and 16 − 7 = 9 stay inside 2..=127.
        let mut d = near_identity_bd8();
        d.lmcs_delta_abs_crs = 7;
        d.lmcs_delta_sign_crs_flag = true;
        let got = d.derive(8).unwrap();
        assert_eq!(got.lmcs_delta_crs, -7);
        assert_eq!(got.chroma_scale_coeff[0], 4096); // 32768 / 8
        for i in 1..LMCS_NUM_BINS {
            assert_eq!(got.chroma_scale_coeff[i], 3640); // 32768 / 9
        }
        // Luma arrays are unaffected by lmcsDeltaCrs.
        assert_eq!(got.inv_scale_coeff[1], 2048);
    }

    #[test]
    fn derive_rejects_joint_band_violations_bd8() {
        // lmcsCW = 2 (band floor) + lmcsDeltaCrs = −1 → joint 1 < 2.
        let mut d = LmcsData {
            lmcs_delta_max_bin_idx: 15,
            lmcs_delta_cw_prec_minus1: 6,
            ..Default::default()
        };
        d.lmcs_delta_abs_cw[0] = 14;
        d.lmcs_delta_sign_cw_flag[0] = true;
        d.lmcs_delta_abs_crs = 1;
        d.lmcs_delta_sign_crs_flag = true;
        let err = d.derive(8).unwrap_err();
        assert!(err.to_string().contains("lmcsDeltaCrs"), "{err}");
        // lmcsCW = 127 (band ceiling) + lmcsDeltaCrs = +1 → joint 128.
        d.lmcs_delta_abs_cw[0] = 111;
        d.lmcs_delta_sign_cw_flag[0] = false;
        d.lmcs_delta_sign_crs_flag = false;
        assert!(d.derive(8).is_err());
    }

    #[test]
    fn derive_joint_band_skips_zero_bins() {
        // A non-zero lmcsDeltaCrs must not trip the joint band on bins
        // where lmcsCW = 0 (the constraint is gated on lmcsCW != 0); the
        // zero bins still take the eq. 100 1 << 11 arm.
        let mut d = LmcsData {
            lmcs_min_bin_idx: 4,
            lmcs_delta_max_bin_idx: 11, // window = bin 4 only
            ..Default::default()
        };
        d.lmcs_delta_abs_crs = 7;
        d.lmcs_delta_sign_crs_flag = true;
        let got = d.derive(8).unwrap();
        assert_eq!(got.chroma_scale_coeff[4], 3640); // 32768 / (16 − 7)
        for i in (0..LMCS_NUM_BINS).filter(|&i| i != 4) {
            assert_eq!(got.chroma_scale_coeff[i], 1 << 11);
        }
    }

    #[test]
    fn parse_then_derive_integration() {
        // Wire-level §7.3.2.19 payload → parse_lmcs_data → derive_lmcs,
        // matching a direct derive on the equivalent struct.
        let d = near_identity_bd8();
        let mut bw = BitWriter::new();
        write_lmcs_data(&mut bw, &d, false);
        bw.rbsp_trailing_bits();
        let bytes = bw.into_bytes();
        let mut br = BitReader::new(&bytes);
        let parsed = parse_lmcs_data(&mut br, false).unwrap();
        let got = parsed.derive(8).unwrap();
        assert_eq!(got, derive_lmcs(&d, 8).unwrap());
        assert_eq!(got.lmcs_pivot[16], 255);
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

    #[test]
    fn forward_map_identity_pivot_bins_bd8() {
        // near-identity payload at BitDepth 8: every bin except bin 0 has
        // lmcsCW = 16 = OrgCW, so ScaleCoeff = 2048 and InputPivot ==
        // LmcsPivot is offset by the single −1 on bin 0. Pick a sample in
        // bin 5: idxY = 80 >> 4 = 5; InputPivot[5] = 80; LmcsPivot[5] =
        // 15 + 16*4 = 79; ScaleCoeff[5] = 2048.
        let d = near_identity_bd8().derive(8).unwrap();
        // eq. 1213: 79 + ((2048 * (80 − 80) + 1024) >> 11) = 79 + 0 = 79.
        assert_eq!(d.forward_map_luma_sample(80), 79);
        // sample 85 in bin 5: 79 + ((2048 * 5 + 1024) >> 11) = 79 + 5 = 84.
        assert_eq!(d.forward_map_luma_sample(85), 84);
        // bin 0 (sample 8): ScaleCoeff[0] = 1920, InputPivot[0] = 0,
        // LmcsPivot[0] = 0 → ((1920 * 8 + 1024) >> 11) = (16384 >> 11) = 8.
        assert_eq!(d.forward_map_luma_sample(8), 8);
    }

    #[test]
    fn idx_y_inv_piecewise_lookup_bd8() {
        // near-identity payload: pivots are 0, 15, 31, 47, ..., 255.
        let d = near_identity_bd8().derive(8).unwrap();
        // sample 0 < LmcsPivot[1] = 15 → idx 0.
        assert_eq!(d.idx_y_inv(0, 0, 15), 0);
        // sample 15 not < 15; < LmcsPivot[2] = 31 → idx 1.
        assert_eq!(d.idx_y_inv(15, 0, 15), 1);
        assert_eq!(d.idx_y_inv(30, 0, 15), 1);
        assert_eq!(d.idx_y_inv(31, 0, 15), 2);
        // sample 255 runs off the loop → Min(idx, 15) = 15.
        assert_eq!(d.idx_y_inv(255, 0, 15), 15);
        assert_eq!(d.idx_y_inv(1000, 0, 15), 15);
    }

    #[test]
    fn forward_inverse_luma_round_trip_bd8() {
        // Forward-map then identify-and-inverse-map must recover the
        // original prediction-domain sample to within the piecewise
        // quantisation. On the near-identity payload (every bin scale is
        // a strict 1:1 except the single shortened bin 0) the inverse is
        // exact for samples that map cleanly across the pivots.
        let d = near_identity_bd8().derive(8).unwrap();
        for s in 16u32..=255 {
            let mapped = d.forward_map_luma_sample(s) as u32;
            let idx = d.idx_y_inv(mapped, 0, 15);
            let back = d.inverse_map_luma_sample(mapped, idx);
            // Bins 1..=15 are 1:1 (ScaleCoeff == InvScaleCoeff == 2048),
            // so the recovered sample equals the input exactly.
            assert_eq!(back, s, "sample {s} mapped to {mapped} idx {idx}");
        }
    }

    #[test]
    fn inverse_map_clamps_to_bit_depth() {
        // A reconstructed sample beyond the top pivot still inverse-maps
        // through the idx-15 piece and Clip1's to (1 << BitDepth) − 1.
        let d = near_identity_bd8().derive(8).unwrap();
        let back = d.inverse_map_luma_sample(255, 15);
        assert!(back <= 255);
    }

    #[test]
    fn chroma_residual_scale_sign_and_abs_bd8() {
        // lmcsDeltaCrs = −7 payload: ChromaScaleCoeff[5] = 32768 / 9 =
        // 3640 (bin 5, lmcsCW = 16). varScale = 3640.
        let mut d = near_identity_bd8();
        d.lmcs_delta_abs_crs = 7;
        d.lmcs_delta_sign_crs_flag = true;
        let der = d.derive(8).unwrap();
        let var_scale = der.chroma_var_scale(5);
        assert_eq!(var_scale, 3640);
        // eq. 1220 with res = +16: (16 * 3640 + 1024) >> 11 =
        // (58240 + 1024) >> 11 = 59264 >> 11 = 28. pred 100 → 128.
        assert_eq!(der.scale_chroma_residual_sample(100, 16, var_scale), 128);
        // res = −16: Sign folds to −28 → pred 100 → 72.
        assert_eq!(der.scale_chroma_residual_sample(100, -16, var_scale), 72);
        // res = 0 → no change.
        assert_eq!(der.scale_chroma_residual_sample(100, 0, var_scale), 100);
    }

    #[test]
    fn chroma_residual_scale_clip3_residual_bd8() {
        // eq. 1219 clamps the residual to ±(1 << BitDepth) before the
        // scale fold; eq. 1220 Clip1's the sum to 0..=255.
        let der = near_identity_bd8().derive(8).unwrap();
        let var_scale = der.chroma_var_scale(5); // 2048 (lmcsDeltaCrs 0)
        assert_eq!(var_scale, 2048);
        // varScale 2048 ⇒ scale fold is identity: (Abs * 2048 + 1024)
        // >> 11 = Abs. A huge positive residual clamps to 255 (Clip1).
        assert_eq!(der.scale_chroma_residual_sample(200, 1000, var_scale), 255);
        // A huge negative residual clamps to 0.
        assert_eq!(der.scale_chroma_residual_sample(50, -1000, var_scale), 0);
        // The eq. 1219 residual clamp itself: res = 300 > (1 << 8) − 1 =
        // 255 is clamped to 255 before scaling; pred 0 + 255 → 255.
        assert_eq!(der.scale_chroma_residual_sample(0, 300, var_scale), 255);
    }

    #[test]
    fn forward_inverse_round_trip_bd10_narrowed_window() {
        // BitDepth 10, window bins 1..=14 (OrgCW = 64): the active bins
        // are 1:1 so forward∘inverse is exact for samples inside them.
        let d = LmcsData {
            lmcs_min_bin_idx: 1,
            lmcs_delta_max_bin_idx: 1,
            ..Default::default()
        }
        .derive(10)
        .unwrap();
        // Bins 1..=14 cover samples 64..=959 (LmcsPivot[1]=0,
        // [15]=896). Pick samples inside an active, 1:1 bin.
        for s in [64u32, 100, 500, 895] {
            let mapped = d.forward_map_luma_sample(s) as u32;
            let idx = d.idx_y_inv(mapped, 1, 14);
            let back = d.inverse_map_luma_sample(mapped, idx);
            assert_eq!(back, s, "sample {s} mapped {mapped} idx {idx}");
        }
    }
}
