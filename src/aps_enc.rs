//! VVC Adaptation Parameter Set encoder — emit side of the §7.3.2.6
//! `adaptation_parameter_set_rbsp()` + §7.3.2.18 `alf_data()` payload.
//!
//! Round-45 scope: ALF APS only. The encoder pipeline synthesises
//! chroma + CC-ALF APSes in-memory (`build_chroma_alf_aps` /
//! `build_cc_alf_aps` in [`crate::encoder_pipeline`]); this module turns
//! those [`crate::aps::AlfApsData`] structs back into the wire-format
//! RBSP bytes the §7.3.2.6 parser reads. LMCS / scaling-list APSes are
//! out of scope.
//!
//! ## Wire layout (§7.3.2.6)
//!
//! ```text
//!   aps_params_type:                       u(3)
//!   aps_adaptation_parameter_set_id:       u(5)
//!   aps_chroma_present_flag:               u(1)
//!   if (aps_params_type == ALF_APS) alf_data()  // §7.3.2.18
//!   aps_extension_flag:                    u(1) (= 0; no tail)
//!   rbsp_trailing_bits()
//! ```
//!
//! ## `alf_data()` (§7.3.2.18)
//!
//! ```text
//!   alf_luma_filter_signal_flag:                u(1)
//!   if (aps_chroma_present_flag) {
//!     alf_chroma_filter_signal_flag:            u(1)
//!     alf_cc_cb_filter_signal_flag:             u(1)
//!     alf_cc_cr_filter_signal_flag:             u(1)
//!   }
//!   if (alf_luma_filter_signal_flag) {
//!     alf_luma_clip_flag:                       u(1)
//!     alf_luma_num_filters_signalled_minus1:    ue(v)
//!     if (alf_luma_num_filters_signalled_minus1 > 0)
//!       for (filtIdx = 0; filtIdx < NumAlfFilters; filtIdx++)
//!         alf_luma_coeff_delta_idx[ filtIdx ]:  u(v) (Ceil(Log2(...)) bits)
//!     for (sfIdx = 0; sfIdx <= alf_luma_num_filters_signalled_minus1; sfIdx++)
//!       for (j = 0; j < ALF_LUMA_NUM_COEFF; j++) {
//!         alf_luma_coeff_abs[sfIdx][j]:         ue(v)
//!         if (alf_luma_coeff_abs[sfIdx][j] > 0)
//!           alf_luma_coeff_sign[sfIdx][j]:      u(1)
//!       }
//!     if (alf_luma_clip_flag)
//!       for (sfIdx, j)
//!         alf_luma_clip_idx[sfIdx][j]:          u(2)
//!   }
//!   if (alf_chroma_filter_signal_flag) {
//!     alf_chroma_clip_flag:                     u(1)
//!     alf_chroma_num_alt_filters_minus1:        ue(v)
//!     for (altIdx = 0..n_alt) {
//!       for (j = 0..ALF_CHROMA_NUM_COEFF) {
//!         alf_chroma_coeff_abs[altIdx][j]:      ue(v)
//!         if (... > 0) alf_chroma_coeff_sign:   u(1)
//!       }
//!       if (alf_chroma_clip_flag)
//!         for (j) alf_chroma_clip_idx:          u(2)
//!     }
//!   }
//!   if (alf_cc_cb_filter_signal_flag) {
//!     alf_cc_cb_filters_signalled_minus1:       ue(v)
//!     for (k = 0..n) for (j = 0..7) {
//!       alf_cc_cb_mapped_coeff_abs[k][j]:       u(3)
//!       if (... > 0) alf_cc_cb_coeff_sign:      u(1)
//!     }
//!   }
//!   if (alf_cc_cr_filter_signal_flag) ... (same shape)
//! ```
//!
//! Spec reference: ITU-T H.266 | ISO/IEC 23090-3 (V4, 01/2026)
//! §7.3.2.6 + §7.3.2.18 + Table 6.

use oxideav_core::{Error, Result};

use crate::aps::{
    AlfApsData, ALF_CC_NUM_COEFF, ALF_CHROMA_NUM_COEFF, ALF_LUMA_NUM_COEFF, NUM_ALF_FILTERS,
};
use crate::encoder::BitWriter;

/// Top-level ALF APS RBSP emitter. Mirrors the §7.3.2.6 parser in
/// [`crate::aps::parse_aps`].
///
/// `aps_id` must be in `0..=7` (per Table 6 / §7.4.3.6 ALF range).
pub fn emit_alf_aps_rbsp(
    aps_id: u8,
    aps_chroma_present_flag: bool,
    alf: &AlfApsData,
) -> Result<Vec<u8>> {
    if aps_id > 7 {
        return Err(Error::invalid(format!(
            "h266 ALF APS: aps_adaptation_parameter_set_id out of range (got {aps_id})"
        )));
    }
    if !alf.alf_luma_filter_signal_flag
        && !alf.alf_chroma_filter_signal_flag
        && !alf.alf_cc_cb_filter_signal_flag
        && !alf.alf_cc_cr_filter_signal_flag
    {
        // §7.4.3.18 invariant: at least one of the four signal flags
        // must be 1.
        return Err(Error::invalid(
            "h266 ALF APS: at least one of alf_luma/chroma/cc_cb/cc_cr filter_signal_flag \
             must be 1",
        ));
    }
    if (alf.alf_chroma_filter_signal_flag
        || alf.alf_cc_cb_filter_signal_flag
        || alf.alf_cc_cr_filter_signal_flag)
        && !aps_chroma_present_flag
    {
        return Err(Error::invalid(
            "h266 ALF APS: chroma / CC-ALF signal flag set but aps_chroma_present_flag = 0",
        ));
    }

    let mut bw = BitWriter::new();
    bw.write_bits(0u32, 3); // aps_params_type = ALF_APS = 0
    bw.write_bits(aps_id as u32, 5);
    bw.write_bit(if aps_chroma_present_flag { 1 } else { 0 });

    write_alf_data(&mut bw, aps_chroma_present_flag, alf)?;

    // aps_extension_flag = 0 (no extension).
    bw.write_bit(0);
    bw.rbsp_trailing_bits();
    Ok(bw.into_bytes())
}

/// Emit the §7.3.2.18 `alf_data()` body into `bw`. Mirrors
/// [`crate::aps::parse_alf_data`] in reverse.
fn write_alf_data(
    bw: &mut BitWriter,
    aps_chroma_present_flag: bool,
    alf: &AlfApsData,
) -> Result<()> {
    bw.write_bit(if alf.alf_luma_filter_signal_flag {
        1
    } else {
        0
    });
    if aps_chroma_present_flag {
        bw.write_bit(if alf.alf_chroma_filter_signal_flag {
            1
        } else {
            0
        });
        bw.write_bit(if alf.alf_cc_cb_filter_signal_flag {
            1
        } else {
            0
        });
        bw.write_bit(if alf.alf_cc_cr_filter_signal_flag {
            1
        } else {
            0
        });
    }

    if alf.alf_luma_filter_signal_flag {
        if alf.luma_coeff.len() != NUM_ALF_FILTERS {
            return Err(Error::invalid(format!(
                "h266 ALF APS: luma_coeff len = {} != NumAlfFilters = {NUM_ALF_FILTERS}",
                alf.luma_coeff.len()
            )));
        }
        bw.write_bit(if alf.alf_luma_clip_flag { 1 } else { 0 });

        // Recover the spec's `alf_luma_num_filters_signalled_minus1`
        // by counting unique rows in `luma_coeff` (the parser
        // pre-expanded eq. 89). For the round-45 encoder pipeline the
        // synthesised APS uses a single signalled filter (n=1),
        // matching the round-41 helper — we still allow N up to
        // NumAlfFilters for arbitrary inputs.
        let cl = compress_luma(alf)?;
        let n_minus1 = (cl.n_signalled - 1) as u32;
        bw.write_ue(n_minus1);
        if n_minus1 > 0 {
            let bits = ceil_log2(n_minus1 as u64 + 1);
            for di in &cl.delta_idx {
                bw.write_bits(*di, bits);
            }
        }
        for sf_idx in 0..cl.n_signalled {
            for j in 0..ALF_LUMA_NUM_COEFF {
                let v = cl.sf_coeff[sf_idx][j];
                let abs = v.unsigned_abs();
                bw.write_ue(abs);
                if abs > 0 {
                    bw.write_bit(if v < 0 { 1 } else { 0 });
                }
            }
        }
        if alf.alf_luma_clip_flag {
            for sf_idx in 0..cl.n_signalled {
                for j in 0..ALF_LUMA_NUM_COEFF {
                    bw.write_bits(cl.sf_clip[sf_idx][j] as u32, 2);
                }
            }
        }
    }

    if alf.alf_chroma_filter_signal_flag {
        bw.write_bit(if alf.alf_chroma_clip_flag { 1 } else { 0 });
        bw.write_ue(alf.alf_chroma_num_alt_filters_minus1);
        let n_alt = alf.alf_chroma_num_alt_filters_minus1 as usize + 1;
        if alf.chroma_coeff.len() < n_alt {
            return Err(Error::invalid(format!(
                "h266 ALF APS: chroma_coeff has {} rows but {n_alt} were requested",
                alf.chroma_coeff.len()
            )));
        }
        for alt_idx in 0..n_alt {
            for j in 0..ALF_CHROMA_NUM_COEFF {
                let v = alf.chroma_coeff[alt_idx][j];
                let abs = v.unsigned_abs();
                bw.write_ue(abs);
                if abs > 0 {
                    bw.write_bit(if v < 0 { 1 } else { 0 });
                }
            }
            if alf.alf_chroma_clip_flag {
                if alf.chroma_clip_idx.len() < n_alt {
                    return Err(Error::invalid(
                        "h266 ALF APS: chroma_clip_idx shorter than chroma_coeff",
                    ));
                }
                for j in 0..ALF_CHROMA_NUM_COEFF {
                    bw.write_bits(alf.chroma_clip_idx[alt_idx][j] as u32, 2);
                }
            }
        }
    }

    if alf.alf_cc_cb_filter_signal_flag {
        if alf.cc_cb_coeff.is_empty() || alf.cc_cb_coeff.len() > 4 {
            return Err(Error::invalid(format!(
                "h266 ALF APS: alf_cc_cb_filters_signalled out of range (n = {})",
                alf.cc_cb_coeff.len()
            )));
        }
        let n = alf.cc_cb_coeff.len();
        bw.write_ue((n - 1) as u32);
        for k in 0..n {
            for j in 0..ALF_CC_NUM_COEFF {
                let v = alf.cc_cb_coeff[k][j];
                let abs = v.unsigned_abs();
                if abs > 7 {
                    return Err(Error::invalid(format!(
                        "h266 ALF APS: alf_cc_cb_coeff[{k}][{j}] = {v} (|v| > 7)"
                    )));
                }
                bw.write_bits(abs, 3);
                if abs > 0 {
                    bw.write_bit(if v < 0 { 1 } else { 0 });
                }
            }
        }
    }

    if alf.alf_cc_cr_filter_signal_flag {
        if alf.cc_cr_coeff.is_empty() || alf.cc_cr_coeff.len() > 4 {
            return Err(Error::invalid(format!(
                "h266 ALF APS: alf_cc_cr_filters_signalled out of range (n = {})",
                alf.cc_cr_coeff.len()
            )));
        }
        let n = alf.cc_cr_coeff.len();
        bw.write_ue((n - 1) as u32);
        for k in 0..n {
            for j in 0..ALF_CC_NUM_COEFF {
                let v = alf.cc_cr_coeff[k][j];
                let abs = v.unsigned_abs();
                if abs > 7 {
                    return Err(Error::invalid(format!(
                        "h266 ALF APS: alf_cc_cr_coeff[{k}][{j}] = {v} (|v| > 7)"
                    )));
                }
                bw.write_bits(abs, 3);
                if abs > 0 {
                    bw.write_bit(if v < 0 { 1 } else { 0 });
                }
            }
        }
    }

    Ok(())
}

/// Output of [`compress_luma`] — the spec's `(filtCoeff[],
/// alf_luma_clip_idx[], alf_luma_coeff_delta_idx[])` tuple after
/// deduplication.
struct CompressedLuma {
    n_signalled: usize,
    delta_idx: Vec<u32>,
    sf_coeff: Vec<[i32; ALF_LUMA_NUM_COEFF]>,
    sf_clip: Vec<[u8; ALF_LUMA_NUM_COEFF]>,
}

/// Compress the parser-expanded `luma_coeff[ NumAlfFilters ]` into the
/// spec's signalled-filter tuple per eq. 89. Equal rows are
/// deduplicated to minimise the signalled count.
fn compress_luma(alf: &AlfApsData) -> Result<CompressedLuma> {
    let mut sf_coeff: Vec<[i32; ALF_LUMA_NUM_COEFF]> = Vec::new();
    let mut sf_clip: Vec<[u8; ALF_LUMA_NUM_COEFF]> = Vec::new();
    let mut delta_idx = vec![0u32; NUM_ALF_FILTERS];
    let clip_used = alf.alf_luma_clip_flag;
    let zero_clip = [0u8; ALF_LUMA_NUM_COEFF];
    for filt_idx in 0..NUM_ALF_FILTERS {
        let coeff_row = alf.luma_coeff[filt_idx];
        let clip_row = if clip_used && filt_idx < alf.luma_clip_idx.len() {
            alf.luma_clip_idx[filt_idx]
        } else {
            zero_clip
        };
        // Search for an already-seen row.
        let mut found: Option<usize> = None;
        for (sf_idx, existing) in sf_coeff.iter().enumerate() {
            let same_coeff = existing == &coeff_row;
            let same_clip = !clip_used || sf_clip[sf_idx] == clip_row;
            if same_coeff && same_clip {
                found = Some(sf_idx);
                break;
            }
        }
        let sf = match found {
            Some(s) => s,
            None => {
                sf_coeff.push(coeff_row);
                sf_clip.push(clip_row);
                sf_coeff.len() - 1
            }
        };
        delta_idx[filt_idx] = sf as u32;
    }
    Ok(CompressedLuma {
        n_signalled: sf_coeff.len(),
        delta_idx,
        sf_coeff,
        sf_clip,
    })
}

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
    use crate::aps::{parse_aps, ApsParamsType};

    fn flat_chroma_aps() -> AlfApsData {
        let mut row = [0i32; ALF_CHROMA_NUM_COEFF];
        row[0] = 1;
        row[1] = 1;
        row[2] = 2;
        row[3] = 1;
        row[4] = 1;
        row[5] = 2;
        AlfApsData {
            alf_chroma_filter_signal_flag: true,
            alf_chroma_num_alt_filters_minus1: 0,
            chroma_coeff: vec![row],
            chroma_clip_idx: vec![[0u8; ALF_CHROMA_NUM_COEFF]],
            ..AlfApsData::default()
        }
    }

    fn cc_aps() -> AlfApsData {
        let mut row_cb = [0i32; ALF_CC_NUM_COEFF];
        row_cb[1] = 4;
        row_cb[2] = -4;
        let mut row_cr = [0i32; ALF_CC_NUM_COEFF];
        row_cr[1] = 4;
        row_cr[2] = -4;
        AlfApsData {
            alf_cc_cb_filter_signal_flag: true,
            alf_cc_cr_filter_signal_flag: true,
            cc_cb_coeff: vec![row_cb],
            cc_cr_coeff: vec![row_cr],
            ..AlfApsData::default()
        }
    }

    /// Round-45 — chroma-only ALF APS round-trips through the parser.
    #[test]
    fn chroma_alf_aps_round_trips() {
        let alf = flat_chroma_aps();
        let bytes = emit_alf_aps_rbsp(3, true, &alf).unwrap();
        let parsed = parse_aps(&bytes).expect("parse_aps must accept emitted ALF APS");
        assert_eq!(parsed.aps_params_type, ApsParamsType::Alf);
        assert_eq!(parsed.aps_adaptation_parameter_set_id, 3);
        assert!(parsed.aps_chroma_present_flag);
        let p = parsed.alf_data.as_ref().expect("alf_data");
        assert!(!p.alf_luma_filter_signal_flag);
        assert!(p.alf_chroma_filter_signal_flag);
        assert_eq!(p.alf_chroma_num_alt_filters_minus1, 0);
        assert_eq!(p.chroma_coeff.len(), 1);
        assert_eq!(p.chroma_coeff[0], alf.chroma_coeff[0]);
    }

    /// Round-45 — CC-ALF APS round-trips through the parser, including
    /// signed coefficients (taps 2 = -4 stress-tests the sign bit).
    #[test]
    fn cc_alf_aps_round_trips() {
        let alf = cc_aps();
        let bytes = emit_alf_aps_rbsp(5, true, &alf).unwrap();
        let parsed = parse_aps(&bytes).expect("parse_aps must accept emitted CC-ALF APS");
        let p = parsed.alf_data.as_ref().expect("alf_data");
        assert!(p.alf_cc_cb_filter_signal_flag);
        assert!(p.alf_cc_cr_filter_signal_flag);
        assert_eq!(p.cc_cb_coeff.len(), 1);
        assert_eq!(p.cc_cr_coeff.len(), 1);
        assert_eq!(p.cc_cb_coeff[0], alf.cc_cb_coeff[0]);
        assert_eq!(p.cc_cr_coeff[0], alf.cc_cr_coeff[0]);
    }

    /// Round-45 — combined chroma + CC-ALF APS round-trips.
    #[test]
    fn combined_chroma_and_cc_alf_aps_round_trips() {
        let mut alf = flat_chroma_aps();
        let cc = cc_aps();
        alf.alf_cc_cb_filter_signal_flag = true;
        alf.alf_cc_cr_filter_signal_flag = true;
        alf.cc_cb_coeff = cc.cc_cb_coeff;
        alf.cc_cr_coeff = cc.cc_cr_coeff;
        let bytes = emit_alf_aps_rbsp(0, true, &alf).unwrap();
        let parsed = parse_aps(&bytes).expect("parse_aps");
        let p = parsed.alf_data.as_ref().expect("alf_data");
        assert!(p.alf_chroma_filter_signal_flag);
        assert!(p.alf_cc_cb_filter_signal_flag);
        assert!(p.alf_cc_cr_filter_signal_flag);
        assert_eq!(p.chroma_coeff[0], alf.chroma_coeff[0]);
        assert_eq!(p.cc_cb_coeff[0], alf.cc_cb_coeff[0]);
        assert_eq!(p.cc_cr_coeff[0], alf.cc_cr_coeff[0]);
    }

    /// Round-45 — luma-only ALF APS with a single signalled filter
    /// round-trips.
    #[test]
    fn luma_alf_aps_round_trips() {
        // Build a luma APS with one signalled filter — every row of
        // luma_coeff equals the same row pattern.
        let mut row = [0i32; ALF_LUMA_NUM_COEFF];
        row[0] = 2;
        row[5] = -3;
        row[11] = 7;
        let alf = AlfApsData {
            alf_luma_filter_signal_flag: true,
            alf_luma_clip_flag: false,
            luma_coeff: vec![row; NUM_ALF_FILTERS],
            luma_clip_idx: vec![[0u8; ALF_LUMA_NUM_COEFF]; NUM_ALF_FILTERS],
            ..AlfApsData::default()
        };
        let bytes = emit_alf_aps_rbsp(2, false, &alf).unwrap();
        let parsed = parse_aps(&bytes).expect("parse_aps");
        let p = parsed.alf_data.as_ref().expect("alf_data");
        assert!(p.alf_luma_filter_signal_flag);
        assert!(!p.alf_chroma_filter_signal_flag);
        // Every row must match the original.
        for r in &p.luma_coeff {
            assert_eq!(r, &row);
        }
    }

    /// Round-45 — APS with all signal flags off must be rejected.
    #[test]
    fn empty_alf_aps_is_rejected() {
        let alf = AlfApsData::default();
        assert!(emit_alf_aps_rbsp(0, true, &alf).is_err());
    }

    /// Round-45 — APS-id range validation.
    #[test]
    fn aps_id_out_of_range() {
        let alf = flat_chroma_aps();
        assert!(emit_alf_aps_rbsp(8, true, &alf).is_err());
    }

    /// Round-45 — chroma signal flag set but aps_chroma_present_flag = 0
    /// is rejected (the parser would skip the chroma block).
    #[test]
    fn chroma_flag_requires_chroma_present() {
        let alf = flat_chroma_aps();
        assert!(emit_alf_aps_rbsp(0, false, &alf).is_err());
    }
}
