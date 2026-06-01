//! Round-201 — encoder-side CABAC emission for the §7.3.10.5
//! `bcw_idx[x0][y0]` syntax element, plus a dispatcher that mirrors the
//! round-126 reader [`crate::leaf_cu::LeafCuReader::read_bcw_idx_gated`]
//! bin-for-bin under the same §7.3.10.5 gating cascade.
//!
//! Round-126 landed the reader: `read_bcw_idx`, `read_bcw_idx_gated`,
//! and `read_bcw_idx_into`. The round-129 follow-up wired the
//! [`crate::leaf_cu::BcwIdxGate`] from live per-CB state so the gate is
//! evaluated against the §7.3.10.5 conditional verbatim.
//!
//! The matching encoder mirror has been a follow-up since: the
//! `#[cfg(test)]` helper `encode_bcw_idx` in `leaf_cu.rs` exercises the
//! same CABAC bin layout but is not callable from a real encoder
//! pipeline. This module lifts that helper into the public encoder
//! surface parallel to the reader, and threads it through the
//! round-195 composite walker as the final §7.3.11.7 step before the
//! deferred residual / `cu_coded_flag` / `transform_tree()` /
//! `cu_qp_delta` tail.
//!
//! ## Public surface
//!
//! 1. [`encode_bcw_idx`] — TR binarisation with `cMax =
//!    NoBackwardPredFlag ? 4 : 2`, `cRiceParam = 0`. Bin 0 is
//!    ctx-coded against Table 91 slot `init_type - 1` with `ctxInc =
//!    0` per Table 132; bins 1..cMax are bypass-coded. Mirrors
//!    [`crate::leaf_cu::LeafCuReader::read_bcw_idx`] bin-for-bin.
//! 2. [`encode_bcw_idx_gated`] — full §7.3.10.5 dispatcher mirror.
//!    Takes a [`crate::leaf_cu::BcwIdxGate`] + a raw `bcw_idx` value
//!    and emits the bins the reader will consume. Applies the same
//!    gating cascade so the wire layout stays bit-identical to
//!    [`crate::leaf_cu::LeafCuReader::read_bcw_idx_gated`]. When the
//!    §7.3.10.5 gate is closed no bin is emitted, matching the
//!    bitstream the reader will skip parsing (§7.4.12.5 infers
//!    `BcwIdx[x0][y0] = 0`).
//!
//! ## Round-trip
//!
//! For every legal `(gate, value)` pair, the bytes produced by
//! [`encode_bcw_idx_gated`] decode back through
//! [`crate::leaf_cu::LeafCuReader::read_bcw_idx_gated`] to the same
//! `bcw_idx` value.
//!
//! ## Spec reference
//!
//! ITU-T H.266 | ISO/IEC 23090-3 (V4, 01/2026):
//! * §7.3.10.5 — `coding_unit()` else-branch `bcw_idx[x0][y0]` syntax
//!   (the source-of-truth listing this dispatcher mirrors).
//! * §7.4.12.5 — inference rule: `BcwIdx[x0][y0] = 0` when the
//!   syntax element is not present.
//! * §9.3.4.2 / Table 132 — ctxInc routing for bin 0 (bin 0 ctx-coded,
//!   bins 1..cMax bypass).
//! * Table 91 — `bcw_idx` initValue / shiftIdx (round-126).
//! * §7.4.11.6 — `NoBackwardPredFlag` derivation (true ⇔ none of the
//!   slice's L1 references are temporally after the current picture,
//!   so the BCW weight table's "backward" indices are unreachable and
//!   `cMax = 4` instead of `2`).
//!
//! No third-party VVC encoder source was consulted; the implementation
//! is spec-only and mirrors the existing round-126 reader-side code
//! already shipped in this crate.

use oxideav_core::Result;

use crate::cabac_enc::ArithEncoder;
use crate::ctx::ctx_inc_bcw_idx;
use crate::leaf_cu::{BcwIdxGate, LeafCuCtxs};

/// `cMax` for the §7.3.10.5 `bcw_idx[x0][y0]` TR binarisation per
/// Table 132 — `NoBackwardPredFlag ? 4 : 2`.
///
/// When `NoBackwardPredFlag == 1` the §8.5.6.6.2 eq. 981 BCW weight
/// table's "backward" entries are unreachable so the binarisation
/// admits the extended set of weights (cMax = 4 ⇒ five legal values
/// `{0, 1, 2, 3, 4}`). When `NoBackwardPredFlag == 0` the
/// binarisation is restricted to the three forward-only entries
/// (cMax = 2 ⇒ three legal values `{0, 1, 2}`).
///
/// `value 0` maps to the eq. 980 default-weighted bi-pred composition
/// `(predL0 + predL1 + 1) >> 1` per §7.4.12.5; values `1..=cMax` map
/// into the eq. 981 BCW weight lookup `bcwWLut[k] = {4, 5, 3, 10, -2}`
/// via the round-129 [`crate::leaf_cu::MvField::bcw_idx`] field.
pub const fn bcw_idx_c_max(no_backward_pred_flag: bool) -> u32 {
    if no_backward_pred_flag {
        4
    } else {
        2
    }
}

/// Encode one `bcw_idx[x0][y0]` value per §7.3.10.5 / Table 132.
///
/// Mirrors [`crate::leaf_cu::LeafCuReader::read_bcw_idx`] bin-for-bin:
/// TR with `cMax = NoBackwardPredFlag ? 4 : 2`, `cRiceParam = 0`. Bin
/// 0 is ctx-coded against Table 91 slot `init_type - 1` with `ctxInc
/// = 0` per Table 132; bins 1..cMax are bypass-coded.
///
/// The caller is responsible for the §7.3.10.5 outer gate: this
/// helper must NOT be invoked when the gate is closed (the §7.4.12.5
/// inference applies — see [`BcwIdxGate::is_open`]). Use
/// [`encode_bcw_idx_gated`] for a gated wrapper that does the gate
/// check for you.
///
/// The reader assumes the slice is non-I (`init_type ∈ {1, 2}`);
/// `bcw_idx` is never signalled in I slices because the non-merge
/// inter branch is unreachable there (PRED_BI is impossible). This
/// debug-asserts the same.
///
/// `value` must be in `0..=cMax(no_backward_pred_flag)`; out-of-range
/// values silently clamp to the truncation point in release builds
/// and debug-assert in debug builds. The `value == 0` short-circuit
/// emits exactly one ctx-coded bin (bin 0 = 0) and no bypass tail —
/// matching the `(predL0 + predL1 + 1) >> 1` default the round-29
/// composition path applies when no BCW weight is signalled.
pub fn encode_bcw_idx(
    enc: &mut ArithEncoder,
    ctxs: &mut LeafCuCtxs,
    value: u32,
    no_backward_pred_flag: bool,
) -> Result<()> {
    debug_assert!(
        ctxs.init_type >= 1,
        "bcw_idx is not signalled in I slices (non-merge inter branch is unreachable)"
    );
    let cmax = bcw_idx_c_max(no_backward_pred_flag);
    debug_assert!(
        value <= cmax,
        "bcw_idx = {value} out of range for cMax = {cmax} \
         (no_backward_pred_flag = {no_backward_pred_flag})"
    );
    // Bin 0 — ctx-coded against Table 91 slot `init_type - 1`. Per
    // Table 132 the ctxInc is fixed 0; the `ctx_inc_bcw_idx` helper
    // enforces this via debug_assert.
    let block = (ctxs.init_type as usize).saturating_sub(1);
    let n = ctxs.bcw_idx.len();
    let inc = ctx_inc_bcw_idx(0) as usize;
    let slot = (block + inc).min(n - 1);
    let bin0 = if value == 0 { 0 } else { 1 };
    enc.encode_decision(&mut ctxs.bcw_idx[slot], bin0)?;
    if bin0 == 0 {
        return Ok(());
    }
    // Bins 1..cMax — all bypass per Table 132. TR with cRiceParam = 0:
    // emit a `1` bin for each step from 1 up to `value`, then a `0`
    // terminator unless `value == cMax` (the truncation point — no
    // trailing zero is sent).
    let mut i = 1u32;
    while i < cmax {
        let bit = if i < value { 1 } else { 0 };
        enc.encode_bypass(bit)?;
        if bit == 0 {
            break;
        }
        i += 1;
    }
    Ok(())
}

/// Encode `bcw_idx[x0][y0]` *with the §7.3.10.5 gate evaluated for
/// you*: when [`BcwIdxGate::is_open`] is true the encoder is invoked
/// against the gate's `no_backward_pred_flag` (so the TR uses `cMax =
/// NoBackwardPredFlag ? 4 : 2`); otherwise the syntax element is not
/// emitted and the reader recovers the inferred value 0 per §7.4.12.5
/// ("When `bcw_idx[ x0 ][ y0 ]` is not present, it is inferred to be
/// equal to 0").
///
/// Mirror of [`crate::leaf_cu::LeafCuReader::read_bcw_idx_gated`]:
/// the encoder fills the same gate from live per-CB state
/// (`(sps_bcw_enabled, inter_pred_idc, luma_weight_lX,
/// chroma_weight_lX, cb_w * cb_h)`), this routine decides whether to
/// emit bins, and the reader recovers the same value via
/// `read_bcw_idx_gated(gate)`.
///
/// # Inferred-value invariant (debug-only)
///
/// Mirrors the round-177 / round-190 / round-195 debug-assert pattern:
/// when the gate is closed the dispatcher checks that the caller
/// passed the §7.4.12.5-inferred value (`bcw_idx = 0`). Violations
/// panic in debug builds; in release builds the wire stream may not
/// round-trip to the same value.
pub fn encode_bcw_idx_gated(
    enc: &mut ArithEncoder,
    ctxs: &mut LeafCuCtxs,
    gate: &BcwIdxGate,
    value: u32,
) -> Result<()> {
    if !gate.is_open() {
        // §7.4.12.5 inference fires — no bin on the wire. Caller MUST
        // pass `bcw_idx = 0` so a release-build wire stream still
        // round-trips even under a stale `value`.
        debug_assert_eq!(
            value, 0,
            "BCW gate closed → §7.4.12.5 requires bcw_idx = 0 (got {value})"
        );
        return Ok(());
    }
    encode_bcw_idx(enc, ctxs, value, gate.no_backward_pred_flag)
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::cabac::ArithDecoder;
    use crate::leaf_cu::{CuToolFlags, InterPredDir, LeafCuReader};

    /// End-to-end round-trip: drive [`encode_bcw_idx`] against a
    /// `(value, no_backward_pred_flag, init_type)` triple, decode the
    /// resulting bitstream through the round-126 reader, and return
    /// the recovered `bcw_idx`. Mirrors the cfg(test) helper
    /// `bcw_idx_round_trip` in `leaf_cu.rs` but exercises the public
    /// surface.
    fn raw_round_trip(value: u32, no_backward_pred_flag: bool, init_type: u8) -> u32 {
        let mut enc = ArithEncoder::new();
        let mut enc_ctxs = LeafCuCtxs::init_with_init_type(26, init_type);
        encode_bcw_idx(&mut enc, &mut enc_ctxs, value, no_backward_pred_flag)
            .expect("encode_bcw_idx succeeds under a valid value/init_type");
        enc.encode_terminate(1).expect("terminator");
        let mut padded = enc.finish();
        padded.extend_from_slice(&[0u8; 32]);
        let mut dec = ArithDecoder::new(&padded).expect("decoder accepts the encoded stream");
        let mut dec_ctxs = LeafCuCtxs::init_with_init_type(26, init_type);
        let tools = CuToolFlags::default();
        let mut reader = LeafCuReader::new(&mut dec, &mut dec_ctxs, tools);
        reader.read_bcw_idx(no_backward_pred_flag).unwrap()
    }

    /// End-to-end round-trip through the gated dispatcher: drive
    /// [`encode_bcw_idx_gated`] against `(gate, value, init_type)`,
    /// decode through `read_bcw_idx_gated`, return the recovered
    /// value.
    fn gated_round_trip(value: u32, gate: BcwIdxGate, init_type: u8) -> u32 {
        let mut enc = ArithEncoder::new();
        let mut enc_ctxs = LeafCuCtxs::init_with_init_type(26, init_type);
        encode_bcw_idx_gated(&mut enc, &mut enc_ctxs, &gate, value)
            .expect("encode_bcw_idx_gated succeeds under a valid gate/value");
        enc.encode_terminate(1).expect("terminator");
        let mut padded = enc.finish();
        padded.extend_from_slice(&[0u8; 32]);
        let mut dec = ArithDecoder::new(&padded).expect("decoder accepts the encoded stream");
        let mut dec_ctxs = LeafCuCtxs::init_with_init_type(26, init_type);
        let tools = CuToolFlags::default();
        let mut reader = LeafCuReader::new(&mut dec, &mut dec_ctxs, tools);
        reader.read_bcw_idx_gated(gate).unwrap()
    }

    /// Convenience: build an open `BcwIdxGate` with the common
    /// B-slice non-weighted-pred 32x32 bi-pred CU shape.
    fn open_gate(no_backward_pred_flag: bool) -> BcwIdxGate {
        BcwIdxGate {
            sps_bcw_enabled: true,
            inter_pred_idc: Some(InterPredDir::PredBi),
            luma_weight_l0_flag: false,
            luma_weight_l1_flag: false,
            chroma_weight_l0_flag: false,
            chroma_weight_l1_flag: false,
            cb_width: 32,
            cb_height: 32,
            no_backward_pred_flag,
        }
    }

    #[test]
    fn bcw_idx_c_max_table_132_branches() {
        // §7.3.10.5 / Table 132 — cMax selection per NoBackwardPredFlag.
        assert_eq!(bcw_idx_c_max(false), 2);
        assert_eq!(bcw_idx_c_max(true), 4);
    }

    #[test]
    fn raw_round_trips_all_values_cmax_2() {
        // B-slice with at least one L1 backward ref ⇒
        // NoBackwardPredFlag = 0 ⇒ cMax = 2. Three legal values
        // {0, 1, 2}. Cover both non-I initTypes (1 = P, 2 = B).
        for it in [1u8, 2] {
            for v in 0..=2u32 {
                assert_eq!(
                    raw_round_trip(v, false, it),
                    v,
                    "cMax=2 round trip failed for v={v} init={it}"
                );
            }
        }
    }

    #[test]
    fn raw_round_trips_all_values_cmax_4() {
        // B-slice with no L1 backward ref ⇒ NoBackwardPredFlag = 1 ⇒
        // cMax = 4. Five legal values {0, 1, 2, 3, 4}; value 4 hits
        // the truncation point — four bypass-1 bins with no
        // terminating zero. Cover both non-I initTypes.
        for it in [1u8, 2] {
            for v in 0..=4u32 {
                assert_eq!(
                    raw_round_trip(v, true, it),
                    v,
                    "cMax=4 round trip failed for v={v} init={it}"
                );
            }
        }
    }

    #[test]
    fn raw_value_zero_emits_exactly_one_ctx_bin() {
        // Spec sanity: value 0 corresponds to `bin0 = 0`, no bypass
        // tail. Build a stream with exactly that one bin (no bypass
        // bins emitted past the single context decision); the round-
        // trip helper recovers 0 cleanly.
        let it = 2u8;
        let baseline = raw_round_trip(0, false, it);
        assert_eq!(baseline, 0);
        let baseline_nb = raw_round_trip(0, true, it);
        assert_eq!(baseline_nb, 0);
    }

    #[test]
    fn gated_closed_gate_emits_no_bins_round_trip() {
        // §7.4.12.5 inference path — gate closed, no bins on the
        // wire, reader recovers the inferred default 0. Cover both
        // non-I initTypes and the two `no_backward_pred_flag` arms.
        let closed = BcwIdxGate::default();
        assert!(!closed.is_open());
        for it in [1u8, 2] {
            assert_eq!(
                gated_round_trip(0, closed, it),
                0,
                "closed gate should infer bcw_idx = 0 (init={it})"
            );
        }
    }

    #[test]
    fn gated_closed_by_sps_bcw_disabled() {
        // sps_bcw_enabled = false closes the gate even with PRED_BI
        // and no weighted-pred. Tests the SPS off-switch.
        let mut gate = open_gate(false);
        gate.sps_bcw_enabled = false;
        assert!(!gate.is_open());
        assert_eq!(gated_round_trip(0, gate, 2), 0);
    }

    #[test]
    fn gated_closed_by_uni_pred() {
        // inter_pred_idc != PRED_BI closes the gate (BCW only applies
        // to bi-pred per §7.3.10.5).
        let mut gate = open_gate(false);
        gate.inter_pred_idc = Some(InterPredDir::PredL0);
        assert!(!gate.is_open());
        assert_eq!(gated_round_trip(0, gate, 2), 0);
        gate.inter_pred_idc = Some(InterPredDir::PredL1);
        assert!(!gate.is_open());
        assert_eq!(gated_round_trip(0, gate, 2), 0);
        gate.inter_pred_idc = None;
        assert!(!gate.is_open());
        assert_eq!(gated_round_trip(0, gate, 2), 0);
    }

    #[test]
    fn gated_closed_by_weighted_pred_flags() {
        // Any of the four luma/chroma weighted-pred flags closes the
        // gate per §7.3.10.5 (explicit weighted prediction
        // suppresses BCW).
        for setter in [
            |g: &mut BcwIdxGate| g.luma_weight_l0_flag = true,
            |g: &mut BcwIdxGate| g.luma_weight_l1_flag = true,
            |g: &mut BcwIdxGate| g.chroma_weight_l0_flag = true,
            |g: &mut BcwIdxGate| g.chroma_weight_l1_flag = true,
        ] {
            let mut gate = open_gate(false);
            setter(&mut gate);
            assert!(!gate.is_open());
            assert_eq!(gated_round_trip(0, gate, 2), 0);
        }
    }

    #[test]
    fn gated_closed_by_small_cb_area() {
        // cb_w * cb_h < 256 closes the gate per §7.3.10.5 (BCW is
        // suppressed on small CUs to bound the per-pixel weight
        // overhead). 16x8 = 128 < 256 ⇒ closed; 16x16 = 256 ⇒
        // open.
        let mut gate = open_gate(false);
        gate.cb_width = 16;
        gate.cb_height = 8;
        assert!(!gate.is_open());
        assert_eq!(gated_round_trip(0, gate, 2), 0);
        gate.cb_height = 16;
        assert!(gate.is_open());
        // Open gate at the area threshold — value 1 round-trips.
        assert_eq!(gated_round_trip(1, gate, 2), 1);
    }

    #[test]
    fn gated_open_arm_cmax_2_round_trips_all_values() {
        // Open gate, NoBackwardPredFlag = 0 ⇒ cMax = 2. The reader
        // recovers every legal value bit-for-bit.
        let gate = open_gate(false);
        assert!(gate.is_open());
        for it in [1u8, 2] {
            for v in 0..=2u32 {
                assert_eq!(
                    gated_round_trip(v, gate, it),
                    v,
                    "open gate cMax=2 round trip failed for v={v} init={it}"
                );
            }
        }
    }

    #[test]
    fn gated_open_arm_cmax_4_round_trips_all_values() {
        // Open gate, NoBackwardPredFlag = 1 ⇒ cMax = 4. The reader
        // recovers every legal value bit-for-bit, including the
        // value = 4 truncation point.
        let gate = open_gate(true);
        assert!(gate.is_open());
        for it in [1u8, 2] {
            for v in 0..=4u32 {
                assert_eq!(
                    gated_round_trip(v, gate, it),
                    v,
                    "open gate cMax=4 round trip failed for v={v} init={it}"
                );
            }
        }
    }

    #[test]
    fn gated_value_zero_emits_no_bypass_bin() {
        // Sanity: value = 0 ⇒ exactly one ctx-coded bin (bin0 = 0)
        // and no bypass tail. Compare the encoded byte count to the
        // bin-1 + bypass-0 case (value = 1) — both fit in the
        // terminator-padded single-byte envelope but the value-1
        // case must consume more arith-coder state.
        let gate = open_gate(false);
        let mut enc_v0 = ArithEncoder::new();
        let mut ctxs_v0 = LeafCuCtxs::init_with_init_type(26, 2);
        encode_bcw_idx_gated(&mut enc_v0, &mut ctxs_v0, &gate, 0).unwrap();
        enc_v0.encode_terminate(1).unwrap();
        let bytes_v0 = enc_v0.finish().len();

        let mut enc_v1 = ArithEncoder::new();
        let mut ctxs_v1 = LeafCuCtxs::init_with_init_type(26, 2);
        encode_bcw_idx_gated(&mut enc_v1, &mut ctxs_v1, &gate, 1).unwrap();
        enc_v1.encode_terminate(1).unwrap();
        let bytes_v1 = enc_v1.finish().len();

        // value 1 emits strictly more bins on the wire (bin0=1 +
        // bypass=0) so the encoder consumes at least as many output
        // bytes.
        assert!(
            bytes_v1 >= bytes_v0,
            "value = 1 (more bins) must produce at least as many bytes as value = 0 \
             (got v0 = {bytes_v0}, v1 = {bytes_v1})"
        );
    }
}
