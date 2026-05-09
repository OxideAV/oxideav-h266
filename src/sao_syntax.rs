//! VVC sample adaptive offset (SAO) syntax — §7.3.11.3.
//!
//! Decodes the per-CTB `sao(rx, ry)` syntax (§7.3.11.3) using the
//! Table 132 ctxInc assignments and the Table 57 / Table 58 init values
//! transcribed in [`crate::tables`]. The output is a populated
//! [`crate::sao::SaoCtbParams`] that callers fold into the per-picture
//! [`crate::sao::SaoPicture`] for the §8.8.4 apply pass.
//!
//! ## Binarisation summary (Table 127)
//!
//! | Syntax element              | Binarisation | cMax                                    |
//! |-----------------------------|--------------|-----------------------------------------|
//! | `sao_merge_left_flag`       | FL           | 1                                       |
//! | `sao_merge_up_flag`         | FL           | 1                                       |
//! | `sao_type_idx_luma`         | TR           | 2 (cRice = 0)                           |
//! | `sao_type_idx_chroma`       | TR           | 2 (cRice = 0)                           |
//! | `sao_offset_abs[]`          | TR           | (1 << (Min(BitDepth, 10) − 5)) − 1      |
//! | `sao_offset_sign_flag[]`    | FL           | 1                                       |
//! | `sao_band_position[]`       | FL           | 31                                      |
//! | `sao_eo_class_luma`         | FL           | 3                                       |
//! | `sao_eo_class_chroma`       | FL           | 3                                       |
//!
//! ## ctxInc assignments (Table 132)
//!
//! * `sao_merge_left_flag`, `sao_merge_up_flag` — single context (ctx 0).
//! * `sao_type_idx_luma`, `sao_type_idx_chroma` — bin 0 uses ctx 0, the
//!   second TR bin (when present) is **bypass**.
//! * `sao_offset_abs[]`, `sao_offset_sign_flag[]`, `sao_band_position[]`,
//!   `sao_eo_class_*` — entirely **bypass**.
//!
//! ## Merge inference
//!
//! When `sao_merge_left_flag == 1` the entire SAO param block (all three
//! components) inherits from the CTB to the left; same for
//! `sao_merge_up_flag` with the CTB above. The §7.4.12.3 inference also
//! supplies default zeros for parameters that simply weren't sent.
//!
//! Spec reference: ITU-T H.266 | ISO/IEC 23090-3 (V4, 01/2026) §7.3.11.3
//! + §7.4.12.3 + §9.3.4.2.1 (Table 132).

use oxideav_core::Result;

use crate::cabac::{ArithDecoder, ContextModel};
use crate::cabac_enc::ArithEncoder;
use crate::sao::{
    SaoCtb, SaoCtbParams, SaoEoClass, SaoMergeChoice, SaoMergeMap, SaoPicture, SaoTypeIdx,
};
use crate::slice_header::SliceType;
use crate::tables::{init_contexts, SyntaxCtx};

/// CABAC context bundle used by the SAO syntax parser. Sized off
/// Table 51 — both arrays carry one ctxIdx per initType but the parser
/// only uses the slot for the current slice's initType.
pub struct SaoCtxs {
    /// Table 57 — `sao_merge_left_flag` and `sao_merge_up_flag` share
    /// the same per-initType slots.
    pub merge_flag: Vec<ContextModel>,
    /// Table 58 — `sao_type_idx_luma` and `sao_type_idx_chroma` share
    /// the same per-initType slots.
    pub type_idx: Vec<ContextModel>,
}

impl std::fmt::Debug for SaoCtxs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SaoCtxs")
            .field("merge_flag.len", &self.merge_flag.len())
            .field("type_idx.len", &self.type_idx.len())
            .finish()
    }
}

impl SaoCtxs {
    /// Build the SAO context arrays for the given slice QP.
    pub fn init(slice_qp_y: i32) -> Self {
        Self {
            merge_flag: init_contexts(SyntaxCtx::SaoMergeFlag, slice_qp_y),
            type_idx: init_contexts(SyntaxCtx::SaoTypeIdx, slice_qp_y),
        }
    }
}

/// Per-slice configuration the SAO syntax parser needs at decode time.
#[derive(Clone, Copy, Debug)]
pub struct SaoSyntaxConfig {
    /// `sh_sao_luma_used_flag`.
    pub luma_used: bool,
    /// `sh_sao_chroma_used_flag`.
    pub chroma_used: bool,
    /// `sps_chroma_format_idc` — 0 for monochrome.
    pub chroma_format_idc: u32,
    /// `BitDepth` — needed for the `sao_offset_abs` cMax derivation.
    pub bit_depth: u32,
    /// Slice type — selects the initType for Tables 57 / 58.
    pub slice_type: SliceType,
    /// `sh_cabac_init_flag` — flips initType for P/B slices per eq. 1527.
    pub sh_cabac_init_flag: bool,
}

impl SaoSyntaxConfig {
    /// Eq. 1527 — derive initType from slice type + `sh_cabac_init_flag`.
    /// I → 0, P → (cabac_init ? 2 : 1), B → (cabac_init ? 1 : 2).
    pub fn init_type(&self) -> usize {
        match self.slice_type {
            SliceType::I => 0,
            SliceType::P => {
                if self.sh_cabac_init_flag {
                    2
                } else {
                    1
                }
            }
            SliceType::B => {
                if self.sh_cabac_init_flag {
                    1
                } else {
                    2
                }
            }
        }
    }

    /// `sao_offset_abs` cMax per Table 127 — `(1 << (Min(BitDepth, 10)
    /// − 5)) − 1`. For 8-bit this is `(1 << 3) − 1 = 7`; for 10-bit it
    /// is `(1 << 5) − 1 = 31`.
    pub fn offset_abs_cmax(&self) -> u32 {
        let m = self.bit_depth.min(10);
        if m < 5 {
            return 0;
        }
        (1u32 << (m - 5)) - 1
    }
}

/// Decode `sao(rx, ry)` per §7.3.11.3 for a single CTB. Returns the
/// fully-populated [`SaoCtbParams`] (with merge inference applied
/// against the supplied `sao_pic` snapshot for left / up neighbours).
///
/// Inputs:
/// * `dec` — slice CABAC engine (advanced past prior syntax).
/// * `ctxs` — per-slice SAO context bundle.
/// * `cfg` — per-slice SAO config (flags, bit depth, init type).
/// * `sao_pic` — the per-picture SAO array already populated for prior
///   CTBs in slice order. Used to resolve `sao_merge_left/up_flag` and
///   the §7.4.12.3 inferences.
/// * `rx`, `ry` — the CTB grid position.
/// * `left_avail` / `up_avail` — neighbour availability in the spec
///   sense (§7.3.11.3 `leftCtbAvailable` / `upCtbAvailable`). The caller
///   must clear these for the picture-edge / first-row-in-slice / tile
///   boundary cases.
pub fn decode_sao_ctb(
    dec: &mut ArithDecoder<'_>,
    ctxs: &mut SaoCtxs,
    cfg: &SaoSyntaxConfig,
    sao_pic: &SaoPicture,
    rx: u32,
    ry: u32,
    left_avail: bool,
    up_avail: bool,
) -> Result<SaoCtbParams> {
    // §7.4.12.3 default inference: if neither sh_sao_*_used_flag is set,
    // sao(rx, ry) is not invoked at all by the spec walker, so callers
    // should gate on those flags before calling us. As a defence, return
    // the default "not applied" params.
    if !cfg.luma_used && !cfg.chroma_used {
        return Ok(SaoCtbParams::default());
    }

    let init_type = cfg.init_type();
    let ctx_idx = init_type.min(ctxs.merge_flag.len() - 1);

    // sao_merge_left_flag — bit 0 ctx 0 of Table 57. Only present when
    // rx > 0 and `leftCtbAvailable` (the "in-tile" check).
    let mut sao_merge_left = false;
    if rx > 0 && left_avail {
        let bit = dec.decode_decision(&mut ctxs.merge_flag[ctx_idx])?;
        sao_merge_left = bit == 1;
    }

    // sao_merge_up_flag — only when ry > 0 AND !sao_merge_left_flag AND
    // `upCtbAvailable` (the "in-tile and not first-row-in-slice" check).
    let mut sao_merge_up = false;
    if ry > 0 && !sao_merge_left && up_avail {
        let bit = dec.decode_decision(&mut ctxs.merge_flag[ctx_idx])?;
        sao_merge_up = bit == 1;
    }

    // Merge — inherit parameters from the named neighbour and skip the
    // remainder of the syntax block.
    if sao_merge_left {
        return Ok(sao_pic.get(rx - 1, ry));
    }
    if sao_merge_up {
        return Ok(sao_pic.get(rx, ry - 1));
    }

    // No merge: parse the per-component params for every selected cIdx.
    let mut out = SaoCtbParams::default();
    let n_comp = if cfg.chroma_format_idc != 0 { 3 } else { 1 };
    for c_idx in 0..n_comp {
        // The spec's nested gate: parse this component only when its
        // slice flag is on (§7.3.11.3 inner if).
        let active = match c_idx {
            0 => cfg.luma_used,
            _ => cfg.chroma_used,
        };
        if !active {
            continue;
        }

        // sao_type_idx_luma (cIdx == 0) / sao_type_idx_chroma (cIdx == 1
        // — applies to both Cb and Cr per §7.4.12.3). cIdx == 2 reuses
        // the value parsed for cIdx == 1.
        let type_idx_val = if c_idx == 2 {
            // Cr inherits from Cb's parsed sao_type_idx_chroma.
            match out.cb.sao_type_idx {
                SaoTypeIdx::NotApplied => 0,
                SaoTypeIdx::BandOffset => 1,
                SaoTypeIdx::EdgeOffset => 2,
            }
        } else {
            decode_sao_type_idx(dec, &mut ctxs.type_idx[ctx_idx])?
        };
        let sao_type = SaoTypeIdx::from_u32(type_idx_val);
        if sao_type == SaoTypeIdx::NotApplied {
            // Component left at default ("not applied"), per the
            // §7.4.12.3 inference for sao_offset_abs == 0.
            continue;
        }

        // sao_offset_abs[i] for i ∈ 0..3. TR bypass binarisation with
        // cMax = (1 << (Min(BitDepth, 10) - 5)) - 1.
        let abs_cmax = cfg.offset_abs_cmax();
        let mut offset_abs = [0u32; 4];
        for i in 0..4 {
            offset_abs[i] = decode_tr_bypass(dec, abs_cmax)?;
        }

        // Component-specific tail: BO carries sign + band_position; EO
        // carries eo_class.
        let (offset_sign, eo_class, band_position) = match sao_type {
            SaoTypeIdx::BandOffset => {
                let mut signs = [0u32; 4];
                for i in 0..4 {
                    if offset_abs[i] != 0 {
                        signs[i] = dec.decode_bypass()?;
                    }
                }
                let band = dec.decode_bypass_bits(5)?; // FL cMax=31 → 5 bits
                (signs, SaoEoClass::Horizontal, band as u8)
            }
            SaoTypeIdx::EdgeOffset => {
                // sao_eo_class_luma when c_idx == 0 is the parsed value;
                // for c_idx == 1 (Cb) we parse sao_eo_class_chroma; for
                // c_idx == 2 (Cr) the spec inherits from c_idx == 1.
                let class_val = if c_idx == 2 {
                    eo_class_to_u32(out.cb.eo_class)
                } else {
                    dec.decode_bypass_bits(2)?
                };
                let class = SaoEoClass::from_u32(class_val);
                // EO sign inference: i ∈ {0, 1} → 0 (positive),
                // i ∈ {2, 3} → 1 (negative). Encoded via SaoCtb::edge_offset.
                ([0, 0, 1, 1], class, 0)
            }
            SaoTypeIdx::NotApplied => unreachable!(),
        };

        let ctb = SaoCtb {
            sao_type_idx: sao_type,
            eo_class,
            band_position,
            offset_val: offset_val_from_raw(offset_abs, offset_sign, cfg.bit_depth),
        };
        match c_idx {
            0 => out.luma = ctb,
            1 => out.cb = ctb,
            _ => out.cr = ctb,
        }
    }
    Ok(out)
}

/// Eq. 153 — bit-depth-scaled `SaoOffsetVal` from raw `sao_offset_abs`
/// + `sao_offset_sign_flag` arrays. Local copy of the helper in
/// [`crate::sao`] used by the syntax path.
#[inline]
fn offset_val_from_raw(abs: [u32; 4], sign: [u32; 4], bit_depth: u32) -> [i32; 5] {
    let shift = (bit_depth as i32) - bit_depth.min(10) as i32;
    let mut out = [0i32; 5];
    for i in 0..4 {
        let s = 1 - 2 * (sign[i] as i32 & 1);
        let mag = (abs[i] as i32) << shift;
        out[i + 1] = s * mag;
    }
    out
}

/// `sao_type_idx_*` decode — bin 0 is contextual (ctx 0 of Table 58),
/// the second bin (when emitted) is bypass per Table 132. TR(cMax=2)
/// gives bin strings: `0` → 0, `10` → 1, `11` → 2.
fn decode_sao_type_idx(dec: &mut ArithDecoder<'_>, ctx: &mut ContextModel) -> Result<u32> {
    let bin0 = dec.decode_decision(ctx)?;
    if bin0 == 0 {
        return Ok(0);
    }
    let bin1 = dec.decode_bypass()?;
    if bin1 == 0 {
        Ok(1)
    } else {
        Ok(2)
    }
}

/// TR(cMax, cRice = 0) entirely under bypass coding (§9.3.3.3). All
/// SAO TR fields use this path because Table 132 marks every
/// `sao_offset_abs` bin as bypass.
fn decode_tr_bypass(dec: &mut ArithDecoder<'_>, c_max: u32) -> Result<u32> {
    if c_max == 0 {
        return Ok(0);
    }
    // TR with cRice=0 reduces to truncated-unary. Read up to cMax bins;
    // count the leading 1-bits before a 0 (or stop after cMax).
    let mut v = 0u32;
    while v < c_max {
        let b = dec.decode_bypass()?;
        if b == 0 {
            return Ok(v);
        }
        v += 1;
    }
    Ok(c_max)
}

/// Inverse of `SaoEoClass::from_u32`.
fn eo_class_to_u32(c: SaoEoClass) -> u32 {
    match c {
        SaoEoClass::Horizontal => 0,
        SaoEoClass::Vertical => 1,
        SaoEoClass::Deg135 => 2,
        SaoEoClass::Deg45 => 3,
    }
}

// ----------------------------------------------------------------------
// Round-54 — encoder counterpart of `decode_sao_ctb`.
// ----------------------------------------------------------------------

/// Round-54 — emit `sao(rx, ry)` per §7.3.11.3 for a single CTB.
///
/// This is the encoder mirror of [`decode_sao_ctb`]: both functions
/// share the same CABAC context family ([`SaoCtxs`]) and binarisation
/// table (Table 127). The encoder side consumes the encoder-RDO-picked
/// [`SaoCtbParams`] from `sao_pic.get(rx, ry)` and the merge decision
/// from the supplied [`SaoMergeMap`], then walks the same syntax order
/// the decoder reads.
///
/// Inputs:
/// * `enc` — slice CABAC encoder (advanced past prior syntax).
/// * `ctxs` — per-slice SAO context bundle (Tables 57 / 58 — same
///   `SaoCtxs::init` the decoder uses).
/// * `cfg` — per-slice SAO config (must match what the decoder side
///   built from SH/PPS/SPS bits).
/// * `sao_pic` — per-picture SAO array, populated for *this* CTB and
///   for prior CTBs (the merge inheritance reads from neighbours).
/// * `merge_map` — per-CTB merge decision per §7.3.11.3. The encoder
///   honours this bit-for-bit: `MergeLeft` → `sao_merge_left_flag = 1`
///   and the per-component params for this CTB are skipped (the decoder
///   inherits from the left neighbour); `MergeAbove` → `sao_merge_left_flag
///   = 0`, `sao_merge_up_flag = 1`; `Independent` → both merge bits = 0
///   and the per-component params are emitted in full.
/// * `rx`, `ry` — CTB grid position.
/// * `left_avail` / `up_avail` — neighbour availability per the
///   `leftCtbAvailable` / `upCtbAvailable` derivation. Must match the
///   decoder's gates.
///
/// Per §9.3.4.2.1 + Table 124 (legacy "Table 132" in older drafts), the
/// merge bit's `ctxInc` is 0 (single context, no neighbour-conditional
/// adjustment); the type-idx bit-0 also uses ctx 0; everything else is
/// bypass.
pub fn encode_sao_ctb(
    enc: &mut ArithEncoder,
    ctxs: &mut SaoCtxs,
    cfg: &SaoSyntaxConfig,
    sao_pic: &SaoPicture,
    merge_map: &SaoMergeMap,
    rx: u32,
    ry: u32,
    left_avail: bool,
    up_avail: bool,
) -> Result<()> {
    // §7.4.12.3 inference: when neither sh_sao_*_used_flag is set the
    // spec walker never invokes sao(rx, ry), so the encoder side just
    // emits nothing (mirroring the decoder's early return).
    if !cfg.luma_used && !cfg.chroma_used {
        return Ok(());
    }

    let init_type = cfg.init_type();
    let ctx_idx = init_type.min(ctxs.merge_flag.len() - 1);

    let choice = merge_map.get(rx, ry);
    let want_merge_left = matches!(choice, SaoMergeChoice::MergeLeft);
    let want_merge_above = matches!(choice, SaoMergeChoice::MergeAbove);

    // sao_merge_left_flag — bit 0 ctx 0 of Table 57. Only present when
    // rx > 0 and `leftCtbAvailable`. The encoder enforces the same
    // availability gate the decoder reads under (any merge choice that
    // is not actually permitted by neighbour availability collapses to
    // Independent on the wire).
    let left_emit = rx > 0 && left_avail && want_merge_left;
    if rx > 0 && left_avail {
        enc.encode_decision(&mut ctxs.merge_flag[ctx_idx], left_emit as u32)?;
    }

    // sao_merge_up_flag — only when ry > 0 AND !sao_merge_left_flag AND
    // `upCtbAvailable`.
    let up_emit = ry > 0 && up_avail && !left_emit && want_merge_above;
    if ry > 0 && !left_emit && up_avail {
        enc.encode_decision(&mut ctxs.merge_flag[ctx_idx], up_emit as u32)?;
    }

    // Merge — entire per-component block is skipped (decoder will
    // inherit from the named neighbour).
    if left_emit || up_emit {
        return Ok(());
    }

    // No merge — emit the per-component params in the same order the
    // decoder reads them.
    let params = sao_pic.get(rx, ry);
    let n_comp = if cfg.chroma_format_idc != 0 { 3 } else { 1 };
    for c_idx in 0..n_comp {
        let active = match c_idx {
            0 => cfg.luma_used,
            _ => cfg.chroma_used,
        };
        if !active {
            continue;
        }

        let ctb = match c_idx {
            0 => &params.luma,
            1 => &params.cb,
            _ => &params.cr,
        };

        // sao_type_idx_luma (cIdx == 0) / sao_type_idx_chroma (cIdx == 1
        // — value shared with cIdx == 2; the spec's Cr branch inherits
        // the cIdx == 1 parsed value rather than re-emitting). Bin 0 is
        // ctx 0 of Table 58, bin 1 (when present) is bypass.
        let type_val = match ctb.sao_type_idx {
            SaoTypeIdx::NotApplied => 0u32,
            SaoTypeIdx::BandOffset => 1u32,
            SaoTypeIdx::EdgeOffset => 2u32,
        };
        if c_idx != 2 {
            // TR(cMax=2) with bin 0 contextual + bin 1 bypass.
            // Bin 0: 0 → 0; otherwise 1.
            let bin0 = if type_val == 0 { 0 } else { 1 };
            enc.encode_decision(&mut ctxs.type_idx[ctx_idx], bin0)?;
            if bin0 == 1 {
                let bin1 = if type_val == 1 { 0 } else { 1 };
                enc.encode_bypass(bin1)?;
            }
        } else {
            // c_idx == 2: spec inherits sao_type_idx_chroma from c_idx == 1
            // (the Cb pass already emitted it). Sanity: the Cr CTB's
            // type must equal the Cb CTB's type — the encoder RDO is
            // expected to honour this constraint (the per-CTB chroma
            // RDO picks one type for both Cb + Cr in the round-50
            // pipeline, and the round-53 merge path inherits the
            // neighbour's whole CTB so Cb + Cr stay aligned). We do not
            // re-check at runtime; the assertion is inherent to the
            // spec.
        }

        if matches!(ctb.sao_type_idx, SaoTypeIdx::NotApplied) {
            // Component left at default → no further bins.
            continue;
        }

        // sao_offset_abs[i] for i ∈ 0..3. TR(cMax = (1 << (Min(BitDepth,
        // 10) − 5)) − 1) entirely under bypass coding.
        let abs_cmax = cfg.offset_abs_cmax();
        let (offset_abs, offset_sign) = recover_raw_offsets(ctb.offset_val, cfg.bit_depth);
        for i in 0..4 {
            encode_tr_bypass(enc, abs_cmax, offset_abs[i])?;
        }

        // Component-specific tail.
        match ctb.sao_type_idx {
            SaoTypeIdx::BandOffset => {
                for i in 0..4 {
                    if offset_abs[i] != 0 {
                        enc.encode_bypass(offset_sign[i])?;
                    }
                }
                // sao_band_position — FL(cMax=31) → 5 bits, MSB first.
                for k in (0..5).rev() {
                    enc.encode_bypass(((ctb.band_position as u32) >> k) & 1)?;
                }
            }
            SaoTypeIdx::EdgeOffset => {
                if c_idx != 2 {
                    // sao_eo_class_luma (c_idx == 0) / sao_eo_class_chroma
                    // (c_idx == 1, shared with c_idx == 2). FL(cMax=3) →
                    // 2 bits, MSB first.
                    let class_val = eo_class_to_u32(ctb.eo_class);
                    for k in (0..2).rev() {
                        enc.encode_bypass((class_val >> k) & 1)?;
                    }
                }
                // No sign for EO (§7.4.12.3 inference). No band_position.
            }
            SaoTypeIdx::NotApplied => unreachable!(),
        }
    }
    Ok(())
}

/// Round-54 — invert eq. 153 to recover the raw `(sao_offset_abs[i],
/// sao_offset_sign_flag[i])` pair from the bit-depth-scaled
/// `SaoOffsetVal[i + 1]` array stored in [`SaoCtb`].
///
/// Matches the bidirectional encoder/decoder contract: the decoder maps
/// `(abs, sign)` → `offset_val` via `derive_offset_val`; the encoder
/// inverts that using `abs = abs(offset_val) >> shift`, `sign = 1` when
/// `offset_val < 0`. For EO the spec's §7.4.12.3 inference forces sign
/// to be 0/0/1/1 by category; the recovery here returns the actual stored
/// sign (which the encoder code path sets via `SaoCtb::edge_offset`).
fn recover_raw_offsets(offset_val: [i32; 5], bit_depth: u32) -> ([u32; 4], [u32; 4]) {
    let shift = (bit_depth as i32) - bit_depth.min(10) as i32;
    let mut abs = [0u32; 4];
    let mut sign = [0u32; 4];
    for i in 0..4 {
        let v = offset_val[i + 1];
        sign[i] = if v < 0 { 1 } else { 0 };
        let mag = v.unsigned_abs();
        abs[i] = mag >> shift;
    }
    (abs, sign)
}

/// Round-54 — emit a TR(cMax, cRice = 0) value entirely under bypass
/// coding. Mirror of [`decode_tr_bypass`] (truncated unary): emit `value`
/// 1-bits then a terminating 0-bit, capping at `c_max` total bins.
fn encode_tr_bypass(enc: &mut ArithEncoder, c_max: u32, value: u32) -> Result<()> {
    if c_max == 0 {
        return Ok(());
    }
    let v = value.min(c_max);
    for _ in 0..v {
        enc.encode_bypass(1)?;
    }
    if v < c_max {
        enc.encode_bypass(0)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Eq. 1527: I-slice always selects initType 0; P/B flip on
    /// `sh_cabac_init_flag`.
    #[test]
    fn init_type_derivation() {
        let mut cfg = SaoSyntaxConfig {
            luma_used: true,
            chroma_used: true,
            chroma_format_idc: 1,
            bit_depth: 8,
            slice_type: SliceType::I,
            sh_cabac_init_flag: false,
        };
        assert_eq!(cfg.init_type(), 0);
        cfg.sh_cabac_init_flag = true;
        // Still 0 — I-slice ignores the flag.
        assert_eq!(cfg.init_type(), 0);

        cfg.slice_type = SliceType::P;
        cfg.sh_cabac_init_flag = false;
        assert_eq!(cfg.init_type(), 1);
        cfg.sh_cabac_init_flag = true;
        assert_eq!(cfg.init_type(), 2);

        cfg.slice_type = SliceType::B;
        cfg.sh_cabac_init_flag = false;
        assert_eq!(cfg.init_type(), 2);
        cfg.sh_cabac_init_flag = true;
        assert_eq!(cfg.init_type(), 1);
    }

    /// Table 127 cMax derivation for `sao_offset_abs`.
    #[test]
    fn sao_offset_abs_cmax() {
        let mut cfg = SaoSyntaxConfig {
            luma_used: true,
            chroma_used: true,
            chroma_format_idc: 1,
            bit_depth: 8,
            slice_type: SliceType::I,
            sh_cabac_init_flag: false,
        };
        // 8-bit: (1 << 3) - 1 = 7.
        assert_eq!(cfg.offset_abs_cmax(), 7);
        cfg.bit_depth = 10;
        assert_eq!(cfg.offset_abs_cmax(), 31);
        cfg.bit_depth = 12;
        // Min(BitDepth, 10) caps to 10 → still 31.
        assert_eq!(cfg.offset_abs_cmax(), 31);
    }

    /// Empty SAO contexts are 3-entries — one per initType slot.
    #[test]
    fn sao_ctxs_init_sizes() {
        let c = SaoCtxs::init(26);
        assert_eq!(c.merge_flag.len(), 3);
        assert_eq!(c.type_idx.len(), 3);
    }

    /// Disabled flags short-circuit to default.
    #[test]
    fn decode_sao_ctb_no_op_when_flags_off() {
        let data = [0u8; 8];
        let mut dec = ArithDecoder::new(&data).unwrap();
        let mut ctxs = SaoCtxs::init(26);
        let cfg = SaoSyntaxConfig {
            luma_used: false,
            chroma_used: false,
            chroma_format_idc: 1,
            bit_depth: 8,
            slice_type: SliceType::I,
            sh_cabac_init_flag: false,
        };
        let pic = SaoPicture::empty(2, 2);
        let p = decode_sao_ctb(&mut dec, &mut ctxs, &cfg, &pic, 0, 0, false, false).unwrap();
        assert_eq!(p.luma.sao_type_idx, SaoTypeIdx::NotApplied);
    }

    /// Merge-left propagation: when the parsed merge bit is 1, the
    /// returned params equal the left CTB's. Build a contrived stream
    /// where the contextual decision yields 1 (high-MPS context).
    #[test]
    fn decode_sao_ctb_merge_left_inherits() {
        // Build a SaoPicture with non-default left CTB.
        let mut pic = SaoPicture::empty(2, 1);
        pic.set(
            0,
            0,
            SaoCtbParams {
                luma: SaoCtb::band_offset(7, [3, 0, 0, 0], [0, 0, 0, 0], 8),
                ..Default::default()
            },
        );
        // Stream: ivlOffset starts 0, contexts initialised at QP=63 with
        // initValue=60 → biased toward MPS=1, so the first decision is 1.
        let data = [0u8; 8];
        let mut dec = ArithDecoder::new(&data).unwrap();
        let mut ctxs = SaoCtxs::init(63);
        // The default context at initType 0 / QP 63 with initValue=60:
        //   slope = 60 >> 3 = 7, m = 3
        //   offset = 60 & 7 = 4, n = 73
        //   pre = ((3 * (63 - 16)) >> 1) + 73 = 70 + 73 = 143 → clip 127
        //   pStateIdx0 = 127 << 3 = 1016, pStateIdx1 = 127 << 7 = 16256
        //   p_state = 16256 + 16 * 1016 = 32512 → val_mps = 32512 >> 14 = 1
        // ivlOffset=0 < new range → bin = MPS = 1 → sao_merge_left_flag = 1.
        let cfg = SaoSyntaxConfig {
            luma_used: true,
            chroma_used: true,
            chroma_format_idc: 1,
            bit_depth: 8,
            slice_type: SliceType::I,
            sh_cabac_init_flag: false,
        };
        let p = decode_sao_ctb(&mut dec, &mut ctxs, &cfg, &pic, 1, 0, true, false).unwrap();
        // Merge-left → equals pic.get(0, 0).luma.
        assert_eq!(p.luma.sao_type_idx, SaoTypeIdx::BandOffset);
        assert_eq!(p.luma.band_position, 7);
    }

    /// Truncated-unary (TR with cRice=0) decode in pure bypass on an
    /// all-zero stream returns 0 every time.
    #[test]
    fn decode_tr_bypass_zero_stream() {
        let data = [0u8; 8];
        let mut dec = ArithDecoder::new(&data).unwrap();
        // First few values: bin0 = 0 → returns 0; over and over.
        for _ in 0..6 {
            assert_eq!(decode_tr_bypass(&mut dec, 7).unwrap(), 0);
        }
    }

    /// `decode_tr_bypass(dec, 0)` is the cMax=0 fast-path: returns 0 with
    /// no bins consumed.
    #[test]
    fn decode_tr_bypass_cmax_zero_consumes_nothing() {
        let data = [0u8; 4];
        let mut dec = ArithDecoder::new(&data).unwrap();
        let p = dec.position();
        assert_eq!(decode_tr_bypass(&mut dec, 0).unwrap(), 0);
        assert_eq!(dec.position(), p);
    }

    /// Round-trip: `eo_class_to_u32` and `SaoEoClass::from_u32` are
    /// inverses across all four classes.
    #[test]
    fn eo_class_round_trip() {
        for v in 0..4u32 {
            assert_eq!(eo_class_to_u32(SaoEoClass::from_u32(v)), v);
        }
    }

    /// `offset_val_from_raw` matches `derive_offset_val` semantics
    /// (eq. 153) at 8 / 10 / 12 bit.
    #[test]
    fn offset_val_eq_153_smoke() {
        let v = offset_val_from_raw([1, 2, 3, 4], [0, 1, 0, 1], 8);
        assert_eq!(v, [0, 1, -2, 3, -4]);
        let v12 = offset_val_from_raw([1, 1, 1, 1], [0, 0, 0, 0], 12);
        assert_eq!(v12, [0, 4, 4, 4, 4]);
    }

    // ----- Round-54 — encoder mirror tests -----

    fn make_cfg() -> SaoSyntaxConfig {
        SaoSyntaxConfig {
            luma_used: true,
            chroma_used: true,
            chroma_format_idc: 1,
            bit_depth: 8,
            slice_type: SliceType::I,
            sh_cabac_init_flag: false,
        }
    }

    /// Round-54 — encoder/decoder round-trip: emit a single CTB's SAO
    /// params, then re-decode them and assert equality. NotApplied
    /// for all components is the simplest case.
    #[test]
    fn encode_sao_ctb_roundtrip_not_applied() {
        let cfg = make_cfg();
        let pic = SaoPicture::empty(1, 1);
        let merge = SaoMergeMap::empty(1, 1);

        let mut enc = ArithEncoder::new();
        let mut enc_ctxs = SaoCtxs::init(26);
        encode_sao_ctb(
            &mut enc,
            &mut enc_ctxs,
            &cfg,
            &pic,
            &merge,
            0,
            0,
            false,
            false,
        )
        .unwrap();
        // Terminate to align the bit-stream.
        enc.encode_terminate(1).unwrap();
        let bytes = enc.finish();

        let mut dec = ArithDecoder::new(&bytes).unwrap();
        let mut dec_ctxs = SaoCtxs::init(26);
        let dec_pic = SaoPicture::empty(1, 1);
        let p =
            decode_sao_ctb(&mut dec, &mut dec_ctxs, &cfg, &dec_pic, 0, 0, false, false).unwrap();
        assert_eq!(p.luma.sao_type_idx, SaoTypeIdx::NotApplied);
        assert_eq!(p.cb.sao_type_idx, SaoTypeIdx::NotApplied);
        assert_eq!(p.cr.sao_type_idx, SaoTypeIdx::NotApplied);
    }

    /// Round-54 — encoder/decoder round-trip with one band-offset CTB at
    /// the picture origin (no merge possible). Verifies the BO type +
    /// offset_abs/sign + band_position bins all round-trip.
    #[test]
    fn encode_sao_ctb_roundtrip_band_offset() {
        let cfg = make_cfg();
        let mut pic = SaoPicture::empty(1, 1);
        let bo_luma = SaoCtb::band_offset(7, [3, 0, 1, 2], [0, 0, 1, 0], 8);
        let bo_cb = SaoCtb::band_offset(15, [1, 1, 0, 0], [0, 1, 0, 0], 8);
        let bo_cr = SaoCtb::band_offset(15, [1, 1, 0, 0], [0, 1, 0, 0], 8);
        pic.set(
            0,
            0,
            SaoCtbParams {
                luma: bo_luma,
                cb: bo_cb,
                cr: bo_cr,
            },
        );
        let merge = SaoMergeMap::empty(1, 1);

        let mut enc = ArithEncoder::new();
        let mut enc_ctxs = SaoCtxs::init(26);
        encode_sao_ctb(
            &mut enc,
            &mut enc_ctxs,
            &cfg,
            &pic,
            &merge,
            0,
            0,
            false,
            false,
        )
        .unwrap();
        enc.encode_terminate(1).unwrap();
        let bytes = enc.finish();

        let mut dec = ArithDecoder::new(&bytes).unwrap();
        let mut dec_ctxs = SaoCtxs::init(26);
        let dec_pic = SaoPicture::empty(1, 1);
        let p =
            decode_sao_ctb(&mut dec, &mut dec_ctxs, &cfg, &dec_pic, 0, 0, false, false).unwrap();
        assert_eq!(p.luma.sao_type_idx, SaoTypeIdx::BandOffset);
        assert_eq!(p.luma.band_position, 7);
        assert_eq!(p.luma.offset_val, bo_luma.offset_val);
        assert_eq!(p.cb.sao_type_idx, SaoTypeIdx::BandOffset);
        assert_eq!(p.cb.band_position, 15);
        assert_eq!(p.cb.offset_val, bo_cb.offset_val);
    }

    /// Round-54 — encoder/decoder round-trip with EO type. Verifies the
    /// `sao_eo_class_*` bypass bin + the §7.4.12.3 sign inference both
    /// round-trip.
    #[test]
    fn encode_sao_ctb_roundtrip_edge_offset() {
        let cfg = make_cfg();
        let mut pic = SaoPicture::empty(1, 1);
        let eo_luma = SaoCtb::edge_offset(SaoEoClass::Vertical, [2, 1, 1, 2], 8);
        let eo_cb = SaoCtb::edge_offset(SaoEoClass::Deg45, [1, 0, 0, 1], 8);
        let eo_cr = SaoCtb::edge_offset(SaoEoClass::Deg45, [1, 0, 0, 1], 8);
        pic.set(
            0,
            0,
            SaoCtbParams {
                luma: eo_luma,
                cb: eo_cb,
                cr: eo_cr,
            },
        );
        let merge = SaoMergeMap::empty(1, 1);

        let mut enc = ArithEncoder::new();
        let mut enc_ctxs = SaoCtxs::init(26);
        encode_sao_ctb(
            &mut enc,
            &mut enc_ctxs,
            &cfg,
            &pic,
            &merge,
            0,
            0,
            false,
            false,
        )
        .unwrap();
        enc.encode_terminate(1).unwrap();
        let bytes = enc.finish();

        let mut dec = ArithDecoder::new(&bytes).unwrap();
        let mut dec_ctxs = SaoCtxs::init(26);
        let dec_pic = SaoPicture::empty(1, 1);
        let p =
            decode_sao_ctb(&mut dec, &mut dec_ctxs, &cfg, &dec_pic, 0, 0, false, false).unwrap();
        assert_eq!(p.luma.sao_type_idx, SaoTypeIdx::EdgeOffset);
        assert_eq!(p.luma.eo_class, SaoEoClass::Vertical);
        assert_eq!(p.luma.offset_val, eo_luma.offset_val);
        assert_eq!(p.cb.sao_type_idx, SaoTypeIdx::EdgeOffset);
        assert_eq!(p.cb.eo_class, SaoEoClass::Deg45);
    }

    /// Round-54 — multi-CTB row where CTBs 1..3 all merge-left from
    /// CTB 0. Verifies that the encoder writes the merge flag = 1 and
    /// the decoder inherits the params bit-for-bit.
    #[test]
    fn encode_sao_ctb_merge_left_roundtrip_row() {
        let cfg = make_cfg();
        let mut pic_enc = SaoPicture::empty(4, 1);
        let bo_params = SaoCtbParams {
            luma: SaoCtb::band_offset(3, [1, 0, 0, 0], [0, 0, 0, 0], 8),
            cb: SaoCtb::band_offset(5, [2, 0, 0, 0], [0, 0, 0, 0], 8),
            cr: SaoCtb::band_offset(5, [2, 0, 0, 0], [0, 0, 0, 0], 8),
        };
        // CTB 0 carries the params, CTBs 1..3 will merge from it.
        pic_enc.set(0, 0, bo_params);
        // The merge map encodes the encoder's intent; the per-CTB params
        // for merged CTBs are unused on the wire (decoder inherits) but
        // we still set them to the inherited values for parity.
        for rx in 1..4 {
            pic_enc.set(rx, 0, bo_params);
        }
        let mut merge = SaoMergeMap::empty(4, 1);
        for rx in 1..4 {
            merge.set(rx, 0, SaoMergeChoice::MergeLeft);
        }

        let mut enc = ArithEncoder::new();
        let mut enc_ctxs = SaoCtxs::init(26);
        for rx in 0..4 {
            let left_avail = rx > 0;
            encode_sao_ctb(
                &mut enc,
                &mut enc_ctxs,
                &cfg,
                &pic_enc,
                &merge,
                rx,
                0,
                left_avail,
                false,
            )
            .unwrap();
        }
        enc.encode_terminate(1).unwrap();
        let bytes = enc.finish();

        let mut dec = ArithDecoder::new(&bytes).unwrap();
        let mut dec_ctxs = SaoCtxs::init(26);
        let mut dec_pic = SaoPicture::empty(4, 1);
        for rx in 0..4 {
            let left_avail = rx > 0;
            let p = decode_sao_ctb(
                &mut dec,
                &mut dec_ctxs,
                &cfg,
                &dec_pic,
                rx,
                0,
                left_avail,
                false,
            )
            .unwrap();
            dec_pic.set(rx, 0, p);
        }
        // Every CTB inherits the same params.
        for rx in 0..4 {
            let p = dec_pic.get(rx, 0);
            assert_eq!(p.luma.band_position, 3);
            assert_eq!(p.cb.band_position, 5);
            assert_eq!(p.cr.band_position, 5);
        }
    }

    /// Round-54 — `recover_raw_offsets` is the strict inverse of
    /// [`crate::sao::SaoCtb::band_offset`]'s eq. 153 mapping at 8 bit.
    #[test]
    fn recover_raw_offsets_inverts_eq_153_8bit() {
        let raw_abs = [3u32, 1, 0, 2];
        let raw_sign = [0u32, 1, 0, 1];
        let ctb = SaoCtb::band_offset(0, raw_abs, raw_sign, 8);
        let (abs, sign) = recover_raw_offsets(ctb.offset_val, 8);
        assert_eq!(abs, raw_abs);
        // Sign for zero magnitudes is undefined per the spec — the
        // recovery returns 0 for them, matching what the encoder emits
        // (zero magnitudes skip the sign bin entirely).
        for i in 0..4 {
            if raw_abs[i] != 0 {
                assert_eq!(sign[i], raw_sign[i]);
            }
        }
    }
}
