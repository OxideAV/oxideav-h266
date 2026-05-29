//! Round-177 — encoder-side CABAC emission for the §7.3.11.7 non-merge
//! inter affine syntax (`inter_affine_flag` + `cu_affine_type_flag`),
//! plus a dispatcher that mirrors the round-164 reader
//! [`crate::leaf_cu::LeafCuReader::read_non_merge_inter_affine`] bin-for-
//! bin under the same §7.3.11.7 gating cascade.
//!
//! Two readers landed in earlier rounds:
//!
//! * Round-152: [`crate::leaf_cu::LeafCuReader::read_inter_affine_flag`]
//!   parses the §7.3.11.7 outer flag against Table 84's 6-entry context
//!   bundle with the §9.3.4.2.2 / Table 133 ctxInc derivation
//!   ([`crate::ctx::ctx_inc_inter_affine_flag`]).
//! * Round-159: [`crate::leaf_cu::LeafCuReader::read_cu_affine_type_flag`]
//!   parses the §7.3.11.7 inner 6-param flag against Table 85's 2-entry
//!   bundle with the deterministic `ctxInc = 0` per Table 133
//!   ([`crate::ctx::ctx_inc_cu_affine_type_flag`]).
//!
//! Round-164 composed both readers into a single
//! [`crate::leaf_cu::LeafCuReader::read_non_merge_inter_affine`]
//! dispatcher that applies §7.4.12.7 inferences (`false` for any flag
//! whose gate was closed) and folds the two decisions through
//! [`crate::affine::derive_motion_model_idc`] (§8.5.5.2 eq. 160). The
//! encoder-side mirror has been a follow-up since: the
//! `#[cfg(test)]` helpers `encode_inter_affine_flag` /
//! `encode_cu_affine_type_flag` / `dispatch_non_merge_inter_affine_round_trip`
//! in `leaf_cu.rs` exercise the same CABAC bin layout but are not
//! callable from a real encoder pipeline.
//!
//! This module lifts those helpers into a public encoder surface
//! parallel to the reader:
//!
//! 1. [`encode_inter_affine_flag`] — single ctx-coded FL `cMax = 1` bin
//!    for the Table 84 slot picked via
//!    `(init_type - 1) * 3 + ctxInc` with
//!    [`crate::ctx::ctx_inc_inter_affine_flag`] supplying ctxInc from
//!    the §9.3.4.2.2 / Table 133 `condL` / `condA` neighbour state.
//! 2. [`encode_cu_affine_type_flag`] — single ctx-coded FL `cMax = 1`
//!    bin for the Table 85 slot picked via `init_type - 1`
//!    (deterministic `ctxInc = 0` per Table 133).
//! 3. [`encode_non_merge_inter_affine`] — full dispatcher mirror.
//!    Takes a [`crate::leaf_cu::NonMergeInterAffineGate`] + a
//!    [`crate::leaf_cu::NonMergeInterAffineDecision`] and emits the
//!    bins the reader will consume, applying the same §7.3.11.7
//!    gating cascade so the wire layout stays bit-identical to the
//!    reader's expectation. When a §7.4.12.7 inference holds (the
//!    matching gate is closed) no bin is emitted, matching the
//!    bitstream the reader will skip parsing.
//!
//! ## Spec reference
//!
//! ITU-T H.266 | ISO/IEC 23090-3 (V4, 01/2026):
//! * §7.3.11.7 — `coding_unit()` non-merge inter affine syntax (the
//!   `inter_affine_flag` + `cu_affine_type_flag` source-of-truth
//!   listing for both gates and inferences).
//! * §7.4.12.7 — inference rules for missing flags (`false` for both).
//! * §8.5.5.2 — `MotionModelIdc` derivation, eq. 160
//!   (`MotionModelIdc = inter_affine_flag + cu_affine_type_flag`).
//! * §9.3.4.2.2 / Table 132 / Table 133 — context derivation rules
//!   for both flags.
//! * Table 84 — `inter_affine_flag` initValue / shiftIdx (round-152).
//! * Table 85 — `cu_affine_type_flag` initValue / shiftIdx (round-159).
//!
//! No third-party VVC encoder source was consulted; the implementation
//! is spec-only and mirrors the existing round-152 / round-159 /
//! round-164 reader-side code already shipped in this crate.

use oxideav_core::Result;

use crate::affine::derive_motion_model_idc;
use crate::cabac_enc::ArithEncoder;
use crate::ctx::{
    ctx_inc_cu_affine_type_flag, ctx_inc_inter_affine_flag, ctx_inc_merge_subblock_flag,
};
use crate::leaf_cu::{LeafCuCtxs, NonMergeInterAffineDecision, NonMergeInterAffineGate};

/// Encode one `inter_affine_flag[x0][y0]` bin per §7.3.11.7 / Table 132.
///
/// Mirrors [`crate::leaf_cu::LeafCuReader::read_inter_affine_flag`] bin-
/// for-bin:
///
/// * Selects the Table 84 ctx slot via
///   `(init_type - 1) * 3 + ctxInc`, with ctxInc supplied by
///   [`crate::ctx::ctx_inc_inter_affine_flag`] from the §9.3.4.2.2 /
///   Table 133 `condL = MergeSubblockFlag[L] || InterAffineFlag[L]`
///   (same row as `merge_subblock_flag` — the two derivations share
///   `condL` / `condA` per the spec).
/// * Emits a single ctx-coded FL `cMax = 1` bin per Table 132.
///
/// # Preconditions
///
/// The caller is responsible for the §7.3.11.7 outer gate
/// (`sps_affine_enabled_flag && cbWidth >= 16 && cbHeight >= 16`) and
/// the surrounding `general_merge_flag == 0` branch — when any of
/// those is false §7.4.12.7 infers `inter_affine_flag = 0` and this
/// function must not be invoked. Use [`encode_non_merge_inter_affine`]
/// for the full dispatcher that respects these gates automatically.
///
/// `init_type` is taken from the CABAC context bundle's `init_type`
/// field per §9.3.2.2 / Table 51.
#[allow(clippy::too_many_arguments)]
pub fn encode_inter_affine_flag(
    enc: &mut ArithEncoder,
    ctxs: &mut LeafCuCtxs,
    flag: bool,
    left_merge_subblock: bool,
    left_inter_affine: bool,
    left_available: bool,
    above_merge_subblock: bool,
    above_inter_affine: bool,
    above_available: bool,
) -> Result<()> {
    debug_assert!(
        ctxs.init_type >= 1,
        "inter_affine_flag never signalled in I slices"
    );
    let inc = ctx_inc_inter_affine_flag(
        left_merge_subblock,
        left_inter_affine,
        left_available,
        above_merge_subblock,
        above_inter_affine,
        above_available,
    ) as usize;
    // The Table 84 bundle holds 6 entries: 3 per non-I initType.
    // Slot index: (init_type - 1) * 3 + ctxInc, clamped defensively
    // against the bundle length for parity with the reader-side
    // `LeafCuReader::read_inter_affine_flag`.
    let init_off = (ctxs.init_type as usize).saturating_sub(1) * 3;
    let n = ctxs.inter_affine_flag.len() - 1;
    let slot = (init_off + inc).min(n);
    let bit = if flag { 1 } else { 0 };
    enc.encode_decision(&mut ctxs.inter_affine_flag[slot], bit)
}

/// Encode one `cu_affine_type_flag[x0][y0]` bin per §7.3.11.7 / Table
/// 132.
///
/// Mirrors [`crate::leaf_cu::LeafCuReader::read_cu_affine_type_flag`]
/// bin-for-bin:
///
/// * Selects the Table 85 ctx slot via `init_type - 1` (the bundle
///   only carries 2 entries, one per non-I initType — `cu_affine_type_flag`
///   is never signalled in I slices nor in the merge branch).
/// * `ctxInc = 0` per Table 132 / Table 133 — there is no
///   §9.3.4.2.2 neighbour lookup for this flag. The
///   [`crate::ctx::ctx_inc_cu_affine_type_flag`] helper exists for
///   spec traceability and is routed through here so a future Table
///   133 amendment that introduces a non-trivial derivation is caught
///   in one place.
/// * Emits a single ctx-coded FL `cMax = 1` bin per Table 132.
///
/// # Preconditions
///
/// The caller is responsible for the §7.3.11.7 inner 6-param gate
/// (`sps_6param_affine_enabled_flag && inter_affine_flag == 1`) — when
/// the gate is closed §7.4.12.7 infers `cu_affine_type_flag = 0` and
/// this function must not be invoked. Use
/// [`encode_non_merge_inter_affine`] for the full dispatcher that
/// respects this gate automatically.
pub fn encode_cu_affine_type_flag(
    enc: &mut ArithEncoder,
    ctxs: &mut LeafCuCtxs,
    flag: bool,
) -> Result<()> {
    debug_assert!(
        ctxs.init_type >= 1,
        "cu_affine_type_flag never signalled in I slices"
    );
    // Route through `ctx_inc_cu_affine_type_flag` for spec traceability;
    // the deterministic `0` is stable per the round-159 transcription.
    let inc = ctx_inc_cu_affine_type_flag() as usize;
    debug_assert_eq!(
        inc, 0,
        "Table 133 lists deterministic ctxInc = 0 for cu_affine_type_flag"
    );
    let n = ctxs.cu_affine_type_flag.len() - 1;
    let slot = (ctxs.init_type as usize).saturating_sub(1).min(n);
    let bit = if flag { 1 } else { 0 };
    enc.encode_decision(&mut ctxs.cu_affine_type_flag[slot], bit)
}

/// Round-177 — encoder-side mirror of
/// [`crate::leaf_cu::LeafCuReader::read_non_merge_inter_affine`].
///
/// Emits the §7.3.11.7 non-merge inter affine syntax for one CU:
///
/// 1. If the outer affine gate is open
///    (`sps_affine_enabled_flag && cbWidth >= 16 && cbHeight >= 16`),
///    emit one [`encode_inter_affine_flag`] bin carrying
///    `decision.inter_affine_flag`. Otherwise emit nothing — the
///    reader will infer `inter_affine_flag = 0` per §7.4.12.7 and
///    `decision.inter_affine_flag` MUST therefore equal `false`.
/// 2. If the inner 6-param gate is open against the (effective)
///    `inter_affine_flag` value
///    (`sps_6param_affine_enabled_flag && inter_affine_flag == 1`),
///    emit one [`encode_cu_affine_type_flag`] bin carrying
///    `decision.cu_affine_type_flag`. Otherwise emit nothing — the
///    reader will infer `cu_affine_type_flag = 0` per §7.4.12.7 and
///    `decision.cu_affine_type_flag` MUST therefore equal `false`.
/// 3. Verify (via `debug_assert!`) that
///    `decision.motion_model == derive_motion_model_idc(decision.inter_affine_flag,
///    decision.cu_affine_type_flag)` so callers cannot accidentally
///    encode a flag pair that disagrees with the typed enum.
///
/// The reader and encoder share `NonMergeInterAffineGate` /
/// `NonMergeInterAffineDecision` types so the caller can pass the same
/// struct through both sides of a round-trip with no field
/// translation.
///
/// # `decision` validity
///
/// In a release build a `decision` whose flag pair disagrees with its
/// `motion_model` enum still encodes correctly — only the two flag
/// bools drive the wire, the `motion_model` field is redundant
/// (recomputable). The debug assertion catches encoder-side caller
/// bugs without inflating release binaries.
///
/// # Inferred-flag invariants (debug-only)
///
/// When the outer gate is closed the caller MUST pass
/// `decision.inter_affine_flag == false`; when the inner gate is
/// closed against the resulting `inter_affine_flag` the caller MUST
/// pass `decision.cu_affine_type_flag == false`. These match the
/// §7.4.12.7 inferences the reader will apply on the decode side; in
/// debug builds violations are caught by `debug_assert!`. In release
/// builds a violating `decision` will still encode 0 bins for the
/// inferred flag, but the resulting wire stream will round-trip back
/// to a different `NonMergeInterAffineDecision` than the encoder
/// asked for — which is the same failure mode the reader-side
/// dispatcher already documents.
pub fn encode_non_merge_inter_affine(
    enc: &mut ArithEncoder,
    ctxs: &mut LeafCuCtxs,
    gate: &NonMergeInterAffineGate,
    decision: &NonMergeInterAffineDecision,
) -> Result<()> {
    // Round-trip safety: the typed `motion_model` field MUST agree with
    // the two raw flag bools per §8.5.5.2 eq. 160. Caught only in debug
    // — release callers paying for production codepaths get the flag
    // bools straight onto the wire.
    debug_assert_eq!(
        decision.motion_model,
        derive_motion_model_idc(decision.inter_affine_flag, decision.cu_affine_type_flag),
        "NonMergeInterAffineDecision.motion_model disagrees with its flag pair"
    );

    // Step 1 — outer gate.
    let outer = gate.outer_affine_gate_open();
    if outer {
        encode_inter_affine_flag(
            enc,
            ctxs,
            decision.inter_affine_flag,
            gate.left_merge_subblock,
            gate.left_inter_affine,
            gate.left_available,
            gate.above_merge_subblock,
            gate.above_inter_affine,
            gate.above_available,
        )?;
    } else {
        // Outer gate closed → §7.4.12.7 infers `inter_affine_flag = 0`.
        // Caller must reflect that in the decision so the round-trip
        // recovers the same value bit-identically.
        debug_assert!(
            !decision.inter_affine_flag,
            "outer affine gate closed → §7.4.12.7 requires inter_affine_flag = 0"
        );
    }

    // Effective `inter_affine_flag` after the §7.4.12.7 inference — when
    // the outer gate was closed the reader will see 0 regardless of
    // what `decision` claims.
    let effective_inter_affine = if outer {
        decision.inter_affine_flag
    } else {
        false
    };

    // Step 2 — inner 6-param gate.
    if gate.inner_6param_gate_open(effective_inter_affine) {
        encode_cu_affine_type_flag(enc, ctxs, decision.cu_affine_type_flag)?;
    } else {
        // Inner gate closed → §7.4.12.7 infers `cu_affine_type_flag = 0`.
        debug_assert!(
            !decision.cu_affine_type_flag,
            "inner 6-param affine gate closed → §7.4.12.7 requires cu_affine_type_flag = 0"
        );
    }

    Ok(())
}

/// Convenience constructor: build a [`NonMergeInterAffineDecision`]
/// from the two raw flag bools, folding `motion_model` through
/// [`derive_motion_model_idc`] (§8.5.5.2 eq. 160).
///
/// This is the natural input to [`encode_non_merge_inter_affine`] when
/// the encoder has already picked the affine motion mode for the CU —
/// it removes a place where the typed enum and the flag pair could
/// drift apart.
pub fn make_non_merge_inter_affine_decision(
    inter_affine_flag: bool,
    cu_affine_type_flag: bool,
) -> NonMergeInterAffineDecision {
    NonMergeInterAffineDecision {
        inter_affine_flag,
        cu_affine_type_flag,
        motion_model: derive_motion_model_idc(inter_affine_flag, cu_affine_type_flag),
    }
}

/// Spec-traceability re-export of the shared §9.3.4.2.2 / Table 133
/// `condL = MergeSubblockFlag[L] || InterAffineFlag[L]` row used by
/// both `merge_subblock_flag` and `inter_affine_flag`. The encoder
/// reuses it when a caller wants to assert that a specific neighbour
/// pair yields a known ctxInc without having to import the lower-
/// level [`crate::ctx`] helper directly.
///
/// Both syntax elements share the row; the reader-side
/// [`crate::ctx::ctx_inc_inter_affine_flag`] delegates to
/// [`crate::ctx::ctx_inc_merge_subblock_flag`] for exactly this
/// reason.
pub fn ctx_inc_shared_merge_subblock_inter_affine(
    left_merge_subblock: bool,
    left_inter_affine: bool,
    left_available: bool,
    above_merge_subblock: bool,
    above_inter_affine: bool,
    above_available: bool,
) -> u32 {
    ctx_inc_merge_subblock_flag(
        left_merge_subblock,
        left_inter_affine,
        left_available,
        above_merge_subblock,
        above_inter_affine,
        above_available,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::affine::MotionModel;
    use crate::cabac::ArithDecoder;
    use crate::leaf_cu::{CuToolFlags, LeafCuReader};

    /// End-to-end round-trip: drive the encoder dispatcher against a
    /// gate + decision, decode the resulting bitstream through the
    /// reader dispatcher, and assert the recovered decision matches
    /// the input bit-for-bit.
    fn dispatcher_round_trip(
        init_type: u8,
        gate: NonMergeInterAffineGate,
        decision: NonMergeInterAffineDecision,
    ) -> NonMergeInterAffineDecision {
        let mut enc = ArithEncoder::new();
        let mut enc_ctxs = LeafCuCtxs::init_with_init_type(26, init_type);
        encode_non_merge_inter_affine(&mut enc, &mut enc_ctxs, &gate, &decision)
            .expect("encode_non_merge_inter_affine succeeds under a valid gate/decision pair");
        enc.encode_terminate(1).expect("terminator");
        let mut padded = enc.finish();
        padded.extend_from_slice(&[0u8; 32]);
        let mut dec = ArithDecoder::new(&padded).expect("decoder accepts the encoded stream");
        let mut dec_ctxs = LeafCuCtxs::init_with_init_type(26, init_type);
        let tools = CuToolFlags::default();
        let mut reader = LeafCuReader::new(&mut dec, &mut dec_ctxs, tools);
        reader
            .read_non_merge_inter_affine(&gate)
            .expect("reader dispatcher succeeds")
    }

    #[test]
    fn make_decision_folds_motion_model_idc_per_eq_160() {
        // §8.5.5.2 eq. 160 truth table — exhaustive over the two-bool
        // input space.
        let d00 = make_non_merge_inter_affine_decision(false, false);
        assert!(!d00.inter_affine_flag && !d00.cu_affine_type_flag);
        assert_eq!(d00.motion_model, MotionModel::Translational);
        assert_eq!(d00.motion_model.idc(), 0);

        let d10 = make_non_merge_inter_affine_decision(true, false);
        assert!(d10.inter_affine_flag && !d10.cu_affine_type_flag);
        assert_eq!(d10.motion_model, MotionModel::Affine4Param);
        assert_eq!(d10.motion_model.idc(), 1);

        let d11 = make_non_merge_inter_affine_decision(true, true);
        assert!(d11.inter_affine_flag && d11.cu_affine_type_flag);
        assert_eq!(d11.motion_model, MotionModel::Affine6Param);
        assert_eq!(d11.motion_model.idc(), 2);

        // `(false, true)` is unreachable per §7.4.12.7 (the inner gate
        // can only open when inter_affine_flag == 1) but the helper
        // still folds it to Translational per `derive_motion_model_idc`
        // — this matches the round-164 dispatcher's defensive contract.
        let d01 = make_non_merge_inter_affine_decision(false, true);
        assert!(!d01.inter_affine_flag && d01.cu_affine_type_flag);
        assert_eq!(d01.motion_model, MotionModel::Translational);
    }

    #[test]
    fn outer_gate_closed_by_sps_emits_zero_bins_and_round_trips_to_translational() {
        // sps_affine_enabled_flag == 0 → outer gate closed → §7.4.12.7
        // infers both flags 0 → no bins emitted → reader recovers
        // Translational without consuming any stream bits.
        let gate = NonMergeInterAffineGate {
            sps_affine_enabled: false,
            sps_6param_affine_enabled: true,
            cb_width: 32,
            cb_height: 32,
            left_available: true,
            above_available: true,
            ..NonMergeInterAffineGate::default()
        };
        assert!(!gate.outer_affine_gate_open());
        let decision = make_non_merge_inter_affine_decision(false, false);
        let recovered = dispatcher_round_trip(2, gate, decision);
        assert_eq!(recovered, decision);
        assert_eq!(recovered.motion_model, MotionModel::Translational);
    }

    #[test]
    fn outer_gate_closed_by_block_size_round_trips_to_translational() {
        // All sub-16 block sizes close the outer gate even with
        // sps_affine_enabled_flag == 1. Sweep the four boundary
        // shapes the round-164 reader test covers.
        for (cb_w, cb_h) in [(8u32, 32u32), (32, 8), (16, 8), (8, 16)] {
            let gate = NonMergeInterAffineGate {
                sps_affine_enabled: true,
                sps_6param_affine_enabled: true,
                cb_width: cb_w,
                cb_height: cb_h,
                left_available: true,
                above_available: true,
                ..NonMergeInterAffineGate::default()
            };
            assert!(!gate.outer_affine_gate_open());
            let decision = make_non_merge_inter_affine_decision(false, false);
            let recovered = dispatcher_round_trip(2, gate, decision);
            assert_eq!(recovered, decision);
            assert_eq!(recovered.motion_model, MotionModel::Translational);
        }
    }

    #[test]
    fn outer_gate_open_inter_affine_zero_round_trips() {
        // Outer gate open, encoder picks Translational → one
        // inter_affine_flag = 0 bin emitted, inner gate stays closed,
        // no cu_affine_type_flag bin → reader recovers Translational.
        let gate = NonMergeInterAffineGate {
            sps_affine_enabled: true,
            sps_6param_affine_enabled: true,
            cb_width: 16,
            cb_height: 16,
            left_available: true,
            above_available: true,
            ..NonMergeInterAffineGate::default()
        };
        let decision = make_non_merge_inter_affine_decision(false, false);
        let recovered = dispatcher_round_trip(2, gate, decision);
        assert_eq!(recovered, decision);
    }

    #[test]
    fn inner_gate_closed_by_sps_round_trips_to_affine4param() {
        // sps_6param_affine_enabled_flag == 0 → only inter_affine_flag
        // is signalled. inter_affine_flag = 1 → MotionModelIdc = 1.
        let gate = NonMergeInterAffineGate {
            sps_affine_enabled: true,
            sps_6param_affine_enabled: false,
            cb_width: 32,
            cb_height: 16,
            left_available: true,
            above_available: true,
            ..NonMergeInterAffineGate::default()
        };
        let decision = make_non_merge_inter_affine_decision(true, false);
        let recovered = dispatcher_round_trip(2, gate, decision);
        assert_eq!(recovered, decision);
        assert_eq!(recovered.motion_model, MotionModel::Affine4Param);
    }

    #[test]
    fn both_gates_open_inner_zero_round_trips_to_affine4param() {
        // Both gates open; encoder emits inter_affine_flag = 1 then
        // cu_affine_type_flag = 0 → MotionModelIdc = 1.
        let gate = NonMergeInterAffineGate {
            sps_affine_enabled: true,
            sps_6param_affine_enabled: true,
            cb_width: 16,
            cb_height: 32,
            left_available: true,
            above_available: true,
            ..NonMergeInterAffineGate::default()
        };
        let decision = make_non_merge_inter_affine_decision(true, false);
        let recovered = dispatcher_round_trip(2, gate, decision);
        assert_eq!(recovered, decision);
        assert_eq!(recovered.motion_model, MotionModel::Affine4Param);
    }

    #[test]
    fn both_gates_open_inner_one_round_trips_to_affine6param() {
        // Both gates open; encoder emits inter_affine_flag = 1 then
        // cu_affine_type_flag = 1 → MotionModelIdc = 2.
        let gate = NonMergeInterAffineGate {
            sps_affine_enabled: true,
            sps_6param_affine_enabled: true,
            cb_width: 64,
            cb_height: 64,
            left_available: true,
            above_available: true,
            ..NonMergeInterAffineGate::default()
        };
        let decision = make_non_merge_inter_affine_decision(true, true);
        let recovered = dispatcher_round_trip(2, gate, decision);
        assert_eq!(recovered, decision);
        assert_eq!(recovered.motion_model, MotionModel::Affine6Param);
    }

    #[test]
    fn neighbour_state_threads_through_round_trip() {
        // Drive every §9.3.4.2.2 / Table 133 condL/condA tuple and
        // verify the encoder picks the same ctx slot the reader will
        // read against — the round-trip recovers the flag value in
        // every case.
        let base = NonMergeInterAffineGate {
            sps_affine_enabled: true,
            sps_6param_affine_enabled: true,
            cb_width: 16,
            cb_height: 16,
            left_available: true,
            above_available: true,
            ..NonMergeInterAffineGate::default()
        };
        // (left_msb, left_aff, above_msb, above_aff) over the 16
        // combinations; each yields a deterministic ctxInc in
        // {0, 1, 2} per the shared Table 133 row.
        for left_msb in [false, true] {
            for left_aff in [false, true] {
                for above_msb in [false, true] {
                    for above_aff in [false, true] {
                        let mut gate = base;
                        gate.left_merge_subblock = left_msb;
                        gate.left_inter_affine = left_aff;
                        gate.above_merge_subblock = above_msb;
                        gate.above_inter_affine = above_aff;
                        for flag in [false, true] {
                            let decision = make_non_merge_inter_affine_decision(flag, false);
                            let recovered = dispatcher_round_trip(2, gate, decision);
                            assert_eq!(
                                recovered, decision,
                                "round trip lost decision at neighbours \
                                 (lm={left_msb}, la={left_aff}, am={above_msb}, aa={above_aff}, flag={flag})"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn unavailable_neighbours_round_trip() {
        // §6.4.4 masks `MergeSubblockFlag[N]` / `InterAffineFlag[N]` to
        // 0 when the neighbour is unavailable. Driving a "stale"
        // neighbour flag with `available = false` must still recover
        // the decision because both encoder and reader compute the
        // same masked ctxInc.
        let gate = NonMergeInterAffineGate {
            sps_affine_enabled: true,
            sps_6param_affine_enabled: true,
            cb_width: 16,
            cb_height: 16,
            left_available: false,
            above_available: false,
            // The next four fields are intentionally non-zero — the
            // §6.4.4 mask zeroes them out for ctxInc purposes.
            left_merge_subblock: true,
            left_inter_affine: true,
            above_merge_subblock: true,
            above_inter_affine: true,
        };
        for it in [1u8, 2] {
            for inter in [false, true] {
                for ty in [false, true] {
                    // Skip the unreachable `(0, 1)` pair on the encoder
                    // side — the inner gate would close it on decode.
                    let effective_ty = if inter { ty } else { false };
                    let decision = make_non_merge_inter_affine_decision(inter, effective_ty);
                    let recovered = dispatcher_round_trip(it, gate, decision);
                    assert_eq!(recovered, decision);
                }
            }
        }
    }

    #[test]
    fn both_init_types_round_trip_across_all_reachable_decisions() {
        // Sweep the five reachable (gate, decision) pairs against both
        // non-I initTypes. The Table 84 / Table 85 per-initType slots
        // both transcribe distinct probabilities so this test pins
        // that the encoder hits the same slot the reader reads.
        let gate_full = NonMergeInterAffineGate {
            sps_affine_enabled: true,
            sps_6param_affine_enabled: true,
            cb_width: 32,
            cb_height: 32,
            left_available: true,
            above_available: true,
            ..NonMergeInterAffineGate::default()
        };
        let gate_no_6p = NonMergeInterAffineGate {
            sps_6param_affine_enabled: false,
            ..gate_full
        };
        let gate_no_affine = NonMergeInterAffineGate {
            sps_affine_enabled: false,
            ..gate_full
        };
        let cases: &[(NonMergeInterAffineGate, bool, bool)] = &[
            (gate_no_affine, false, false), // outer gate closed
            (gate_no_6p, false, false),
            (gate_no_6p, true, false),
            (gate_full, false, false),
            (gate_full, true, false),
            (gate_full, true, true),
        ];
        for it in [1u8, 2] {
            for (gate, inter, ty) in cases.iter().copied() {
                let decision = make_non_merge_inter_affine_decision(inter, ty);
                let recovered = dispatcher_round_trip(it, gate, decision);
                assert_eq!(
                    recovered, decision,
                    "round trip failed at init_type {it} inter={inter} ty={ty}"
                );
            }
        }
    }

    #[test]
    fn ctx_inc_shared_helper_matches_inter_affine_ctx_inc() {
        // The encoder re-exports the shared §9.3.4.2.2 / Table 133 row
        // for spec traceability; it MUST agree with
        // `ctx_inc_inter_affine_flag` for every neighbour-state input
        // (the reader-side `ctx_inc_inter_affine_flag` delegates to
        // `ctx_inc_merge_subblock_flag` for exactly this reason).
        for left_msb in [false, true] {
            for left_aff in [false, true] {
                for left_avail in [false, true] {
                    for above_msb in [false, true] {
                        for above_aff in [false, true] {
                            for above_avail in [false, true] {
                                let shared = ctx_inc_shared_merge_subblock_inter_affine(
                                    left_msb,
                                    left_aff,
                                    left_avail,
                                    above_msb,
                                    above_aff,
                                    above_avail,
                                );
                                let via_inter_affine = ctx_inc_inter_affine_flag(
                                    left_msb,
                                    left_aff,
                                    left_avail,
                                    above_msb,
                                    above_aff,
                                    above_avail,
                                );
                                assert_eq!(
                                    shared, via_inter_affine,
                                    "shared row drifted from ctx_inc_inter_affine_flag at \
                                     (lm={left_msb}, la={left_aff}, lA={left_avail}, am={above_msb}, aa={above_aff}, aA={above_avail})"
                                );
                                assert!(shared <= 2, "ctxInc out of range");
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn encoder_does_not_emit_bin_when_outer_gate_closed() {
        // A direct verification of the §7.4.12.7 inference path on the
        // encoder side: when the outer gate is closed and the
        // decision carries the §7.4.12.7-required `false` flags, the
        // resulting CABAC byte stream is identical to one produced by
        // *not* calling the dispatcher at all (only the terminator
        // separates them). This pins that the dispatcher does not
        // accidentally emit bins that would desync the reader.
        let mut enc_with = ArithEncoder::new();
        let mut enc_no = ArithEncoder::new();
        let mut ctxs_with = LeafCuCtxs::init_with_init_type(26, 2);
        let gate = NonMergeInterAffineGate {
            sps_affine_enabled: false,
            sps_6param_affine_enabled: true,
            cb_width: 32,
            cb_height: 32,
            left_available: true,
            above_available: true,
            ..NonMergeInterAffineGate::default()
        };
        let decision = make_non_merge_inter_affine_decision(false, false);
        encode_non_merge_inter_affine(&mut enc_with, &mut ctxs_with, &gate, &decision)
            .expect("encode under closed outer gate");
        enc_with.encode_terminate(1).expect("terminator");
        enc_no.encode_terminate(1).expect("terminator");
        assert_eq!(
            enc_with.finish(),
            enc_no.finish(),
            "outer-gate-closed dispatcher emitted bins where the reader expects none"
        );
    }

    #[test]
    fn encoder_emits_exactly_one_bin_when_inner_gate_closed() {
        // sps_6param_affine_enabled_flag == 0 → only one bin should be
        // emitted (the outer `inter_affine_flag`). Compare against a
        // direct single-bin emission via `encode_inter_affine_flag` to
        // confirm the dispatcher does not double-emit.
        let gate = NonMergeInterAffineGate {
            sps_affine_enabled: true,
            sps_6param_affine_enabled: false,
            cb_width: 16,
            cb_height: 16,
            left_available: true,
            above_available: true,
            ..NonMergeInterAffineGate::default()
        };
        for flag in [false, true] {
            let mut enc_disp = ArithEncoder::new();
            let mut ctxs_disp = LeafCuCtxs::init_with_init_type(26, 2);
            let decision = make_non_merge_inter_affine_decision(flag, false);
            encode_non_merge_inter_affine(&mut enc_disp, &mut ctxs_disp, &gate, &decision)
                .expect("dispatcher encode");
            enc_disp.encode_terminate(1).expect("terminator");

            let mut enc_direct = ArithEncoder::new();
            let mut ctxs_direct = LeafCuCtxs::init_with_init_type(26, 2);
            encode_inter_affine_flag(
                &mut enc_direct,
                &mut ctxs_direct,
                flag,
                false,
                false,
                true,
                false,
                false,
                true,
            )
            .expect("direct encode");
            enc_direct.encode_terminate(1).expect("terminator");

            assert_eq!(
                enc_disp.finish(),
                enc_direct.finish(),
                "dispatcher diverged from direct single-bin emission at flag={flag}"
            );
        }
    }
}
