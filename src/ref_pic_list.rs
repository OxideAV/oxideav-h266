//! Reference picture list construction (§8.3.2).
//!
//! This module takes the two RPL candidates selected by a slice / picture
//! header (typically via `rpl_sps_flag[i]` + `rpl_idx[i]` or an inlined
//! `ref_pic_list_struct()`), plus the current picture's POC, and builds
//! `RefPicList[0]` and `RefPicList[1]` exactly as the spec prescribes.
//!
//! The decoded-picture buffer (DPB) is not modelled here — resolving a
//! concrete reference picture from a `(layer_id, POC)` key is a later
//! increment. What this module produces instead is a list of
//! [`RefPicListEntry`] values, each annotated with the predicted POC and
//! a classification (STRP / LTRP / ILRP). A higher-level wrapper can then
//! walk the DPB and substitute "no reference picture" for missing entries.
//!
//! Equation references match the V4 (01/2026) text.

use oxideav_core::{Error, Result};

use crate::bitreader::BitReader;
use crate::pps::PicParameterSet;
use crate::sps::{
    parse_ref_pic_list_struct, RefPicListEntry as SpsRefPicListEntry, RefPicListStruct,
    SeqParameterSet,
};

/// Classification of a built [`RefPicListEntry`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RefPicKind {
    /// Short-term reference — POC matched exactly.
    ShortTerm,
    /// Long-term reference — either the full POC LSB (when
    /// `delta_poc_msb_cycle_present_flag == 0`) or a full reconstructed
    /// POC (otherwise) was used.
    LongTerm,
    /// Inter-layer reference — AU-sharing picture with a different
    /// `nuh_layer_id` indicated by `ilrp_idx`.
    InterLayer,
}

/// A single entry in the built RefPicList (§8.3.2). Resolving the entry
/// to a concrete picture in the DPB is the caller's responsibility.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BuiltRefPicEntry {
    /// Classification of this entry.
    pub kind: RefPicKind,
    /// Predicted full POC of the reference picture (§8.3.2 equation
    /// 209). For STRP entries this is the absolute POC derived from
    /// `pocBase + DeltaPocValSt`. For LTRP entries this is either the
    /// POC LSB (if no msb cycle was supplied) or the reconstructed full
    /// POC. For ILRP entries this equals the current picture's POC
    /// (same AU, §8.3.2).
    pub poc: i32,
    /// Applies only when `kind == LongTerm`: `true` when the POC value
    /// above is the LSB-only form (§8.3.2: matched by
    /// `PicOrderCntVal & (MaxPicOrderCntLsb - 1) == PocLsbLt[i][k]`).
    pub is_poc_lsb_only: bool,
    /// Applies only when `kind == InterLayer`: the
    /// `ilrp_idx[i][RplsIdx[i]][j]` value — later mapped via
    /// `DirectRefLayerIdx` + `vps_layer_id` to resolve a DPB entry.
    pub ilrp_idx: u32,
}

impl BuiltRefPicEntry {
    pub fn short_term(poc: i32) -> Self {
        Self {
            kind: RefPicKind::ShortTerm,
            poc,
            is_poc_lsb_only: false,
            ilrp_idx: 0,
        }
    }
    pub fn long_term(poc: i32, is_poc_lsb_only: bool) -> Self {
        Self {
            kind: RefPicKind::LongTerm,
            poc,
            is_poc_lsb_only,
            ilrp_idx: 0,
        }
    }
    pub fn inter_layer(current_poc: i32, ilrp_idx: u32) -> Self {
        Self {
            kind: RefPicKind::InterLayer,
            poc: current_poc,
            is_poc_lsb_only: false,
            ilrp_idx,
        }
    }
}

/// Extra per-LT-entry info supplied by the slice / picture header's
/// `ref_pic_lists()` block (§7.3.9). One entry per long-term reference
/// in `RefPicList[i]`, in the order they appear in the RPL struct.
#[derive(Clone, Copy, Debug, Default)]
pub struct LongTermHeaderInfo {
    /// `poc_lsb_lt[i][k]` — only used when the containing list's
    /// `ltrp_in_header_flag` is 1 (POC LSB deferred to header). Ignored
    /// otherwise.
    pub poc_lsb_lt_header: u32,
    /// `delta_poc_msb_cycle_present_flag[i][k]`.
    pub delta_poc_msb_cycle_present: bool,
    /// `delta_poc_msb_cycle_lt[i][k]`. Used only when the present flag
    /// above is 1.
    pub delta_poc_msb_cycle_lt: u32,
}

/// Inputs to the RefPicList builder for one list (`i = 0` or `i = 1`).
#[derive(Clone, Debug)]
pub struct RefPicListInputs<'a> {
    /// POC of the current picture (equation 206).
    pub current_poc: i32,
    /// MaxPicOrderCntLsb = 1 << (sps_log2_max_pic_order_cnt_lsb_minus4 + 4).
    pub max_poc_lsb: u32,
    /// The `ref_pic_list_struct()` selected for list i.
    pub rpls: &'a RefPicListStruct,
    /// Header-side info for each LT entry, in RPL order. Must have
    /// `rpls.num_ltrp_entries()` elements.
    pub lt_info: &'a [LongTermHeaderInfo],
}

/// Build `RefPicList[i]` from the inputs (§8.3.2).
///
/// Errors are raised for mismatches between the RPL struct and the
/// header-side side-channel (wrong LT info length, missing
/// `poc_lsb_lt`).
pub fn build_ref_pic_list(inputs: &RefPicListInputs<'_>) -> Result<Vec<BuiltRefPicEntry>> {
    let rpls = inputs.rpls;
    let expected_lt = rpls.num_ltrp_entries();
    if inputs.lt_info.len() != expected_lt {
        return Err(Error::invalid(format!(
            "h266 RPL build: expected {expected_lt} LT info entries, got {}",
            inputs.lt_info.len()
        )));
    }
    let mut out: Vec<BuiltRefPicEntry> = Vec::with_capacity(rpls.entries.len());
    let mut poc_base: i32 = inputs.current_poc;
    let mut lt_k: usize = 0;
    let max_poc_lsb = inputs.max_poc_lsb as i64;
    for entry in &rpls.entries {
        match entry {
            SpsRefPicListEntry::ShortTerm {
                delta_poc_val_st, ..
            } => {
                let new_poc = poc_base.wrapping_add(*delta_poc_val_st);
                out.push(BuiltRefPicEntry::short_term(new_poc));
                poc_base = new_poc;
            }
            SpsRefPicListEntry::LongTerm { poc_lsb_lt } => {
                let info = inputs.lt_info.get(lt_k).ok_or_else(|| {
                    Error::invalid(format!(
                        "h266 RPL build: LT info index {lt_k} out of range (len = {})",
                        inputs.lt_info.len()
                    ))
                })?;
                // §7.4.11: when ltrp_in_header_flag = 1 the POC LSB
                // lives in the PH/SH, otherwise in the RPL struct.
                let poc_lsb = if rpls.ltrp_in_header_flag {
                    info.poc_lsb_lt_header
                } else {
                    match poc_lsb_lt {
                        Some(v) => *v,
                        None => {
                            return Err(Error::invalid(
                                "h266 RPL build: LT entry with neither inline POC LSB nor ltrp_in_header",
                            ))
                        }
                    }
                };
                if poc_lsb as i64 >= max_poc_lsb {
                    return Err(Error::invalid(format!(
                        "h266 RPL build: poc_lsb_lt ({poc_lsb}) out of range (MaxPicOrderCntLsb = {max_poc_lsb})"
                    )));
                }
                if info.delta_poc_msb_cycle_present {
                    // FullPocLt[i][k] = PicOrderCntVal of the current
                    // picture − delta_poc_msb_cycle_lt[i][k] *
                    // MaxPicOrderCntLsb − (PicOrderCntVal of the current
                    // picture & (MaxPicOrderCntLsb − 1)) + poc_lsb_lt.
                    // (Derivation in §7.4.11 / §8.3.2 paraphrased.)
                    let curr_lsb = (inputs.current_poc as i64) & (max_poc_lsb - 1);
                    let delta = info.delta_poc_msb_cycle_lt as i64 * max_poc_lsb;
                    let full = (inputs.current_poc as i64) - delta - curr_lsb + poc_lsb as i64;
                    out.push(BuiltRefPicEntry::long_term(full as i32, false));
                } else {
                    // No MSB cycle → the POC LSB alone identifies the
                    // reference (match `PicOrderCntVal & (MaxPicOrderCntLsb
                    // - 1)`, §8.3.2).
                    out.push(BuiltRefPicEntry::long_term(poc_lsb as i32, true));
                }
                lt_k += 1;
            }
            SpsRefPicListEntry::InterLayer { ilrp_idx } => {
                out.push(BuiltRefPicEntry::inter_layer(inputs.current_poc, *ilrp_idx));
            }
        }
    }
    Ok(out)
}

/// Result of parsing a header-level `ref_pic_lists()` block (§7.3.9).
///
/// Two sub-shapes exist for each list `i`:
///
/// * `rpl_sps_flag[i] == 1` — the header selects one of the SPS-owned
///   `ref_pic_list_struct` candidates by index. The chosen struct is
///   referred to as `RplsIdx[i]` below, matching the spec's variable.
/// * `rpl_sps_flag[i] == 0` — the header inlines a brand-new
///   `ref_pic_list_struct(i, sps_num_ref_pic_lists[i])` which we keep
///   alongside the parsed struct. Per §7.4.10, `RplsIdx[i]` equals
///   `sps_num_ref_pic_lists[i]` in that case.
#[derive(Clone, Debug)]
pub struct HeaderRefPicList {
    /// True when the list was selected from the SPS candidates
    /// (`rpl_sps_flag[i] == 1`). False when inlined in the header.
    pub rpl_sps_flag: bool,
    /// `rpl_idx[i]` when signalled / inferred, `None` when the list was
    /// inlined (§7.3.9: the "synthesised" candidate index is
    /// `sps_num_ref_pic_lists[i]`).
    pub rpl_idx: Option<u32>,
    /// Effective `RplsIdx[i]` (equation 146). Always populated — for
    /// inline lists this is `sps_num_ref_pic_lists[i]`.
    pub rpls_idx: u32,
    /// Cloned / parsed `ref_pic_list_struct()`. When `rpl_sps_flag ==
    /// 1` this is a clone of the SPS-owned struct; when inlined it is
    /// a freshly parsed one. Either way the caller can feed it to
    /// [`build_ref_pic_list`] without additional lookups.
    pub rpls: RefPicListStruct,
    /// Per-LTRP side-channel info transmitted by the header after the
    /// RPL struct (§7.3.9). Length equals `rpls.num_ltrp_entries()`.
    pub lt_info: Vec<LongTermHeaderInfo>,
}

/// Parse a header-level `ref_pic_lists()` block (§7.3.9) starting at the
/// reader's current bit position. The reader is advanced past the block
/// (byte alignment is *not* applied — the caller resumes with whatever
/// syntax follows `ref_pic_lists()` in the containing PH / slice header).
///
/// Behaviour notes:
///
/// * `rpl_sps_flag[i]` is only signalled when the SPS carries at least
///   one candidate for list `i` (and, for `i = 1`, when
///   `pps_rpl1_idx_present_flag == 1`). Otherwise §7.4.10 inference
///   rules apply.
/// * When `rpl_sps_flag[i]` is inferred to `rpl_sps_flag[0]` (§7.4.10),
///   the `rpl_idx[i]` value is likewise inferred to `rpl_idx[0]`.
/// * Per §7.4.10, when `rpl_sps_flag[i] == 1` and
///   `sps_num_ref_pic_lists[i] == 1`, `rpl_idx[i]` is not transmitted
///   and is inferred to 0.
pub fn parse_ref_pic_lists(
    br: &mut BitReader<'_>,
    sps: &SeqParameterSet,
    pps: &PicParameterSet,
) -> Result<[HeaderRefPicList; 2]> {
    // Start with list 0 so that list-1 inference can refer to it (§7.4.10).
    let mut out: [Option<HeaderRefPicList>; 2] = [None, None];

    // Effective `sps_num_ref_pic_lists[i]` with the "rpl1_same_as_rpl0"
    // aliasing applied: per §7.4.3.4 list 1 mirrors list 0 under that
    // flag, and the SPS parser populates ref_pic_lists[1] via the
    // alias. We read the raw SPS count array but cap list 1 to the
    // length of the populated candidate set so indexing stays valid.
    let num_rpls: [u32; 2] = [
        sps.tool_flags.num_ref_pic_lists[0],
        sps.tool_flags.num_ref_pic_lists[1],
    ];

    for i in 0..2 {
        let signalled = num_rpls[i] > 0 && (i == 0 || pps.pps_rpl1_idx_present_flag);
        let rpl_sps_flag = if signalled {
            br.u1()? == 1
        } else if num_rpls[i] == 0 {
            // §7.4.10: inferred to 0 when sps_num_ref_pic_lists[i] == 0.
            false
        } else {
            // §7.4.10: i == 1, pps_rpl1_idx_present_flag == 0 → mirror list 0.
            out[0].as_ref().map(|p| p.rpl_sps_flag).unwrap_or(false)
        };

        let (rpl_idx_signalled, rpls): (Option<u32>, RefPicListStruct) = if rpl_sps_flag {
            // Pick from the SPS candidates.
            let signalled_idx = num_rpls[i] > 1 && (i == 0 || pps.pps_rpl1_idx_present_flag);
            let rpl_idx = if signalled_idx {
                let w = ceil_log2(num_rpls[i]);
                let v = br.u(w)?;
                if v >= num_rpls[i] {
                    return Err(Error::invalid(format!(
                        "h266 RPL: rpl_idx[{i}] = {v} out of range (sps_num_ref_pic_lists = {})",
                        num_rpls[i]
                    )));
                }
                Some(v)
            } else if num_rpls[i] == 1 {
                Some(0)
            } else {
                // num_rpls[i] > 1 && i == 1 && !pps_rpl1_idx_present_flag
                // → inferred to rpl_idx[0].
                out[0]
                    .as_ref()
                    .and_then(|p| p.rpl_idx)
                    .map(Some)
                    .unwrap_or(Some(0))
            };
            let idx = rpl_idx.unwrap_or(0);
            // List 1 may have empty candidate storage under
            // `rpl1_same_as_rpl0_flag`; fall back to list 0's candidates
            // in that case to mirror the spec's aliasing.
            let candidates = if sps.tool_flags.ref_pic_lists[i].is_empty() {
                &sps.tool_flags.ref_pic_lists[0]
            } else {
                &sps.tool_flags.ref_pic_lists[i]
            };
            let rpls = candidates.get(idx as usize).cloned().ok_or_else(|| {
                Error::invalid(format!(
                    "h266 RPL: rpl_idx[{i}] = {idx} references missing SPS candidate"
                ))
            })?;
            (rpl_idx, rpls)
        } else {
            // Inline: ref_pic_list_struct(i, sps_num_ref_pic_lists[i]).
            let rpls = parse_ref_pic_list_struct(
                br,
                i as u8,
                num_rpls[i],
                num_rpls[i],
                sps.tool_flags.long_term_ref_pics_flag,
                sps.tool_flags.inter_layer_prediction_enabled_flag,
                pps.pps_weighted_pred_flag,
                pps.pps_weighted_bipred_flag,
                sps.sps_log2_max_pic_order_cnt_lsb_minus4,
            )?;
            (None, rpls)
        };

        let rpls_idx = if rpl_sps_flag {
            rpl_idx_signalled.unwrap_or(0)
        } else {
            num_rpls[i]
        };

        // Per-LT side-channel info. Width of poc_lsb_lt equals
        // sps_log2_max_pic_order_cnt_lsb_minus4 + 4 bits (§7.4.10).
        let poc_lsb_width = sps.sps_log2_max_pic_order_cnt_lsb_minus4 as u32 + 4;
        let mut lt_info: Vec<LongTermHeaderInfo> = Vec::with_capacity(rpls.num_ltrp_entries());
        for _ in 0..rpls.num_ltrp_entries() {
            let poc_lsb_lt_header = if rpls.ltrp_in_header_flag {
                br.u(poc_lsb_width)?
            } else {
                0
            };
            let delta_poc_msb_cycle_present = br.u1()? == 1;
            let delta_poc_msb_cycle_lt = if delta_poc_msb_cycle_present {
                br.ue()?
            } else {
                0
            };
            lt_info.push(LongTermHeaderInfo {
                poc_lsb_lt_header,
                delta_poc_msb_cycle_present,
                delta_poc_msb_cycle_lt,
            });
        }

        out[i] = Some(HeaderRefPicList {
            rpl_sps_flag,
            rpl_idx: rpl_idx_signalled,
            rpls_idx,
            rpls,
            lt_info,
        });
    }

    // `out` is guaranteed populated by the loop above.
    Ok([out[0].take().unwrap(), out[1].take().unwrap()])
}

/// `Ceil(Log2(n))` with the VVC convention `Ceil(Log2(0)) = 0`.
fn ceil_log2(n: u32) -> u32 {
    if n <= 1 {
        return 0;
    }
    32 - (n - 1).leading_zeros()
}

/// Convenience wrapper that resolves `rpl_sps_flag[i]` / `rpl_idx[i]`
/// semantics against the SPS. Returns the `ref_pic_list_struct` that
/// the slice / picture header has selected, or `None` when the stream
/// opted for an inline list (handled by the caller).
pub fn pick_rpls_from_sps<'s>(
    sps: &'s SeqParameterSet,
    list_idx: usize,
    rpls_idx: u32,
) -> Result<&'s RefPicListStruct> {
    if list_idx > 1 {
        return Err(Error::invalid(format!(
            "h266 RPL build: list_idx out of range ({list_idx})"
        )));
    }
    let lists = &sps.tool_flags.ref_pic_lists[list_idx];
    let idx = rpls_idx as usize;
    lists
        .get(idx)
        .ok_or_else(|| Error::invalid(format!("h266 RPL build: rpls_idx {idx} out of range")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sps::{RefPicListEntry, RefPicListStruct};

    fn strp(delta: i32) -> RefPicListEntry {
        RefPicListEntry::ShortTerm {
            abs_delta_poc_st: (delta.unsigned_abs()) - 1,
            strp_entry_sign_flag: delta < 0,
            delta_poc_val_st: delta,
        }
    }

    /// Two STRPs with deltas +1 and -2 against current POC = 10 →
    /// RefPicPocList = [11, 9] (pocBase updates after each STRP).
    #[test]
    fn strp_only_chains_poc_deltas() {
        let rpls = RefPicListStruct {
            entries: vec![strp(1), strp(-2)],
            ltrp_in_header_flag: false,
        };
        let inputs = RefPicListInputs {
            current_poc: 10,
            max_poc_lsb: 256,
            rpls: &rpls,
            lt_info: &[],
        };
        let out = build_ref_pic_list(&inputs).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].kind, RefPicKind::ShortTerm);
        assert_eq!(out[0].poc, 11);
        assert_eq!(out[1].kind, RefPicKind::ShortTerm);
        assert_eq!(out[1].poc, 9);
    }

    /// LT with inline POC LSB and no delta MSB cycle — the LSB itself
    /// is the returned POC and `is_poc_lsb_only` is true.
    #[test]
    fn lt_inline_no_msb_cycle_returns_lsb() {
        let rpls = RefPicListStruct {
            entries: vec![RefPicListEntry::LongTerm {
                poc_lsb_lt: Some(0x42),
            }],
            ltrp_in_header_flag: false,
        };
        let lt_info = [LongTermHeaderInfo {
            poc_lsb_lt_header: 0,
            delta_poc_msb_cycle_present: false,
            delta_poc_msb_cycle_lt: 0,
        }];
        let inputs = RefPicListInputs {
            current_poc: 500,
            max_poc_lsb: 256,
            rpls: &rpls,
            lt_info: &lt_info,
        };
        let out = build_ref_pic_list(&inputs).unwrap();
        assert_eq!(out[0].kind, RefPicKind::LongTerm);
        assert_eq!(out[0].poc, 0x42);
        assert!(out[0].is_poc_lsb_only);
    }

    /// LT with `ltrp_in_header_flag = 1`: the POC LSB lives in the
    /// header-side info block. With a MSB cycle of 1 and current POC
    /// = 500, Max = 256, lsb = 0x10 → full = 500 - 1*256 - (500 & 255)
    /// + 0x10 = 500 - 256 - 244 + 16 = 16.
    #[test]
    fn lt_header_with_msb_cycle_reconstructs_full_poc() {
        let rpls = RefPicListStruct {
            entries: vec![RefPicListEntry::LongTerm { poc_lsb_lt: None }],
            ltrp_in_header_flag: true,
        };
        let lt_info = [LongTermHeaderInfo {
            poc_lsb_lt_header: 0x10,
            delta_poc_msb_cycle_present: true,
            delta_poc_msb_cycle_lt: 1,
        }];
        let inputs = RefPicListInputs {
            current_poc: 500,
            max_poc_lsb: 256,
            rpls: &rpls,
            lt_info: &lt_info,
        };
        let out = build_ref_pic_list(&inputs).unwrap();
        assert_eq!(out[0].kind, RefPicKind::LongTerm);
        assert_eq!(out[0].poc, 16);
        assert!(!out[0].is_poc_lsb_only);
    }

    /// ILRP entries inherit the current picture's POC and forward the
    /// `ilrp_idx` so the caller can resolve the cross-layer picture.
    #[test]
    fn ilrp_uses_current_poc_and_preserves_idx() {
        let rpls = RefPicListStruct {
            entries: vec![RefPicListEntry::InterLayer { ilrp_idx: 2 }],
            ltrp_in_header_flag: false,
        };
        let inputs = RefPicListInputs {
            current_poc: 33,
            max_poc_lsb: 256,
            rpls: &rpls,
            lt_info: &[],
        };
        let out = build_ref_pic_list(&inputs).unwrap();
        assert_eq!(out[0].kind, RefPicKind::InterLayer);
        assert_eq!(out[0].poc, 33);
        assert_eq!(out[0].ilrp_idx, 2);
    }

    /// LT info vector size must match the number of LT entries in the
    /// selected RPL struct; mismatches are rejected.
    #[test]
    fn lt_info_length_mismatch_is_rejected() {
        let rpls = RefPicListStruct {
            entries: vec![RefPicListEntry::LongTerm {
                poc_lsb_lt: Some(0),
            }],
            ltrp_in_header_flag: false,
        };
        let inputs = RefPicListInputs {
            current_poc: 0,
            max_poc_lsb: 256,
            rpls: &rpls,
            lt_info: &[],
        };
        assert!(build_ref_pic_list(&inputs).is_err());
    }

    /// A mixed STRP + LT + ILRP list reproduces the expected sequence
    /// of POCs / kinds (regression guard for the outer dispatch).
    #[test]
    fn mixed_list_preserves_order_and_classification() {
        let rpls = RefPicListStruct {
            entries: vec![
                strp(2),
                RefPicListEntry::LongTerm {
                    poc_lsb_lt: Some(7),
                },
                RefPicListEntry::InterLayer { ilrp_idx: 1 },
                strp(-1),
            ],
            ltrp_in_header_flag: false,
        };
        let lt_info = [LongTermHeaderInfo::default()];
        let inputs = RefPicListInputs {
            current_poc: 20,
            max_poc_lsb: 256,
            rpls: &rpls,
            lt_info: &lt_info,
        };
        let out = build_ref_pic_list(&inputs).unwrap();
        // STRP(+2): pocBase 20 → 22.
        assert_eq!(out[0].kind, RefPicKind::ShortTerm);
        assert_eq!(out[0].poc, 22);
        // LT inline 7.
        assert_eq!(out[1].kind, RefPicKind::LongTerm);
        assert_eq!(out[1].poc, 7);
        // ILRP uses current POC = 20.
        assert_eq!(out[2].kind, RefPicKind::InterLayer);
        assert_eq!(out[2].poc, 20);
        // STRP(-1): pocBase is the *last STRP*'s POC (22) → 21.
        assert_eq!(out[3].kind, RefPicKind::ShortTerm);
        assert_eq!(out[3].poc, 21);
    }

    // ---- Header-level ref_pic_lists() tests ----

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

    fn synth_sps_pps() -> (crate::sps::SeqParameterSet, crate::pps::PicParameterSet) {
        use crate::sps::{PartitionConstraints, ToolFlags};
        let mut sps = crate::sps::SeqParameterSet {
            sps_seq_parameter_set_id: 0,
            sps_video_parameter_set_id: 0,
            sps_max_sublayers_minus1: 0,
            sps_chroma_format_idc: 1,
            sps_log2_ctu_size_minus5: 2,
            sps_ptl_dpb_hrd_params_present_flag: false,
            profile_tier_level: None,
            sps_gdr_enabled_flag: false,
            sps_ref_pic_resampling_enabled_flag: false,
            sps_res_change_in_clvs_allowed_flag: false,
            sps_pic_width_max_in_luma_samples: 320,
            sps_pic_height_max_in_luma_samples: 240,
            conformance_window: None,
            sps_subpic_info_present_flag: false,
            sps_bitdepth_minus8: 2,
            sps_entropy_coding_sync_enabled_flag: false,
            sps_entry_point_offsets_present_flag: false,
            sps_log2_max_pic_order_cnt_lsb_minus4: 4,
            sps_poc_msb_cycle_flag: false,
            sps_poc_msb_cycle_len_minus1: 0,
            sps_num_extra_ph_bytes: 0,
            sps_num_extra_sh_bytes: 0,
            sps_sublayer_dpb_params_flag: false,
            dpb_parameters: None,
            partition_constraints: PartitionConstraints::default(),
            tool_flags: ToolFlags::default(),
            subpic_info: None,
            sps_timing_hrd_params_present_flag: false,
            general_timing_hrd: None,
            sps_sublayer_cpb_params_present_flag: false,
            ols_timing_hrd: None,
            sps_field_seq_flag: false,
            sps_vui_parameters_present_flag: false,
            vui_payload: Vec::new(),
            sps_extension_flag: false,
            sps_range_extension_flag: false,
            sps_extension_7bits: 0,
        };
        sps.tool_flags.num_ref_pic_lists = [2, 1];
        sps.tool_flags.ref_pic_lists[0] = vec![
            RefPicListStruct {
                entries: vec![strp(1)],
                ltrp_in_header_flag: false,
            },
            RefPicListStruct {
                entries: vec![strp(-1)],
                ltrp_in_header_flag: false,
            },
        ];
        sps.tool_flags.ref_pic_lists[1] = vec![RefPicListStruct {
            entries: vec![strp(2)],
            ltrp_in_header_flag: false,
        }];
        let pps = crate::pps::PicParameterSet {
            pps_pic_parameter_set_id: 0,
            pps_seq_parameter_set_id: 0,
            pps_mixed_nalu_types_in_pic_flag: false,
            pps_pic_width_in_luma_samples: 320,
            pps_pic_height_in_luma_samples: 240,
            conformance_window: None,
            scaling_window: None,
            pps_output_flag_present_flag: false,
            pps_no_pic_partition_flag: true,
            pps_subpic_id_mapping_present_flag: false,
            pps_rect_slice_flag: true,
            pps_single_slice_per_subpic_flag: true,
            pps_loop_filter_across_slices_enabled_flag: false,
            pps_cabac_init_present_flag: false,
            pps_num_ref_idx_default_active_minus1: [0, 0],
            pps_rpl1_idx_present_flag: true,
            pps_weighted_pred_flag: false,
            pps_weighted_bipred_flag: false,
            pps_ref_wraparound_enabled_flag: false,
            pps_pic_width_minus_wraparound_offset: 0,
            pps_init_qp_minus26: 0,
            pps_cu_qp_delta_enabled_flag: false,
            pps_chroma_tool_offsets_present_flag: false,
            pps_cb_qp_offset: 0,
            pps_cr_qp_offset: 0,
            pps_joint_cbcr_qp_offset_present_flag: false,
            pps_joint_cbcr_qp_offset_value: 0,
            pps_slice_chroma_qp_offsets_present_flag: false,
            pps_cu_chroma_qp_offset_list_enabled_flag: false,
            pps_deblocking_filter_control_present_flag: false,
            pps_deblocking_filter_override_enabled_flag: false,
            pps_deblocking_filter_disabled_flag: false,
            pps_dbf_info_in_ph_flag: true,
            pps_rpl_info_in_ph_flag: true,
            pps_sao_info_in_ph_flag: true,
            pps_alf_info_in_ph_flag: true,
            pps_wp_info_in_ph_flag: true,
            pps_qp_delta_info_in_ph_flag: true,
            pps_picture_header_extension_present_flag: false,
            pps_slice_header_extension_present_flag: false,
            pps_extension_flag: false,
            partition: None,
        };
        (sps, pps)
    }

    /// List 0: rpl_sps_flag = 1, rpl_idx = 1 (selecting the second SPS
    /// candidate). List 1: rpl_sps_flag = 1, rpl_idx = 0 (the single
    /// candidate, inference applies).
    #[test]
    fn parses_rpl_sps_flag_selection() {
        use crate::bitreader::BitReader;
        let (sps, pps) = synth_sps_pps();
        let mut bits: Vec<u8> = Vec::new();
        // List 0: rpl_sps_flag = 1, rpl_idx width = ceil(log2(2)) = 1.
        push_u(&mut bits, 1, 1);
        push_u(&mut bits, 1, 1);
        // List 1: rpl_sps_flag = 1 (pps_rpl1_idx_present_flag = 1, num = 1).
        // rpl_idx not signalled (num = 1 → inferred to 0).
        push_u(&mut bits, 1, 1);
        // Pad.
        push_u(&mut bits, 0, 5);
        let bytes = pack(&bits);
        let mut br = BitReader::new(&bytes);
        let rpl = parse_ref_pic_lists(&mut br, &sps, &pps).unwrap();
        assert!(rpl[0].rpl_sps_flag);
        assert_eq!(rpl[0].rpl_idx, Some(1));
        assert_eq!(rpl[0].rpls_idx, 1);
        assert_eq!(rpl[0].rpls.entries.len(), 1);
        assert!(rpl[1].rpl_sps_flag);
        assert_eq!(rpl[1].rpl_idx, Some(0));
    }

    /// Inline list 0 (rpl_sps_flag = 0): parses a 1-entry STRP struct
    /// from the header. List 1 is still SPS-selected.
    #[test]
    fn parses_inline_rpl_struct() {
        use crate::bitreader::BitReader;
        let (mut sps, mut pps) = synth_sps_pps();
        // Only allow one SPS candidate on list 0 to simplify.
        sps.tool_flags.num_ref_pic_lists = [1, 1];
        sps.tool_flags.ref_pic_lists[0] = vec![RefPicListStruct {
            entries: vec![strp(3)],
            ltrp_in_header_flag: false,
        }];
        pps.pps_rpl1_idx_present_flag = true;
        let mut bits: Vec<u8> = Vec::new();
        // List 0: rpl_sps_flag = 0 (inline).
        push_u(&mut bits, 0, 1);
        // Inline ref_pic_list_struct(0, 1):
        //   num_ref_entries = 1 (ue = 010? no, ue(1) = "010" → value 1)
        push_ue(&mut bits, 1);
        // ltrp_in_header_flag not transmitted (long_term_ref_pics_flag = 0).
        // Entry loop (1 iteration):
        //   no inter-layer flag (sps_inter_layer_prediction_enabled_flag = 0).
        //   no st_ref_pic_flag (sps_long_term_ref_pics_flag = 0, inferred true).
        //   abs_delta_poc_st = 0 → ue "1" (1 bit).
        push_ue(&mut bits, 0);
        // abs = abs_delta_poc_st + 1 = 1 > 0, strp_entry_sign_flag present.
        push_u(&mut bits, 0, 1); // sign = 0 → positive.
                                 // List 1: rpl_sps_flag = 1 (num = 1, single SPS candidate).
        push_u(&mut bits, 1, 1);
        // rpl_idx inferred to 0 (num = 1).
        // No LT info block.
        push_u(&mut bits, 0, 3); // pad
        let bytes = pack(&bits);
        let mut br = BitReader::new(&bytes);
        let rpl = parse_ref_pic_lists(&mut br, &sps, &pps).unwrap();
        assert!(!rpl[0].rpl_sps_flag);
        assert_eq!(rpl[0].rpl_idx, None);
        assert_eq!(rpl[0].rpls_idx, 1);
        assert_eq!(rpl[0].rpls.entries.len(), 1);
        match rpl[0].rpls.entries[0] {
            crate::sps::RefPicListEntry::ShortTerm {
                delta_poc_val_st, ..
            } => assert_eq!(delta_poc_val_st, 1),
            other => panic!("unexpected entry {other:?}"),
        }
        assert!(rpl[1].rpl_sps_flag);
    }
}
