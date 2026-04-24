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

use crate::sps::{RefPicListEntry as SpsRefPicListEntry, RefPicListStruct, SeqParameterSet};

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
}
