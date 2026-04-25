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
use crate::sao::{SaoCtb, SaoCtbParams, SaoEoClass, SaoPicture, SaoTypeIdx};
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
}
