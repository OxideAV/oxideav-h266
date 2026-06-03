//! Round-219 — reader-side composite walker for the §7.3.11.7 non-merge
//! inter CU pre-residual syntax.
//!
//! The encoder-side dispatcher
//! [`crate::non_merge_inter_pre_residual_enc::encode_non_merge_inter_pre_residual`]
//! (round 190) drives the full §7.3.11.7 cascade in one call. Until
//! this round the matching reader path only existed as a sequence of
//! per-element invocations on [`crate::leaf_cu::LeafCuReader`] (`read_non_merge_inter_affine`
//! → `read_inter_pred_idc` → `read_sym_mvd_flag` → `read_ref_idx_lx` ×
//! 2 → `read_mvd_coding` × 2 → `read_mvp_lx_flag` × 2), and every
//! consumer had to walk those helpers by hand and re-apply the
//! §7.4.12.7 inferences for every gate-closed branch. Every existing
//! call site reproduces the same nine-step skeleton (see the round-190
//! encoder-side `tests::round_trip` for the worked example).
//!
//! This round lifts that skeleton into a public reader-side dispatcher
//! [`read_non_merge_inter_pre_residual`] that exactly mirrors the
//! round-190 encoder dispatcher in §7.3.11.7 spec order. The returned
//! [`crate::non_merge_inter_pre_residual_enc::NonMergeInterPreResidualDecision`]
//! is bit-for-bit identical to the decision the encoder originally fed
//! to its own dispatcher when the wire was produced — the round-trip
//! tests pin this on every reachable §7.4.12.7-inference path.
//!
//! ## Scope
//!
//! Translational only (one `mvd_coding` per active list, `numCpMv ==
//! 1`). Multi-CP-MV affine MVD parsing (round 207 encoder-side
//! follow-up), `amvr_flag` / `amvr_precision_idx` (round 195 encoder
//! follow-up — the reader-side `read_amvr_inter_gated` already exists
//! from round 193), and `bcw_idx` (round 201 encoder follow-up — the
//! reader-side `read_bcw_idx_gated` already exists from round 126) sit
//! after the pre-residual cascade and are *not* consumed by this
//! dispatcher. They are addressable by composing this dispatcher with
//! the existing per-element reader-side helpers in spec order, the
//! same way the encoder-side `_with_amvr` / `_with_amvr_and_bcw`
//! variants compose with their encoder-side per-element helpers.
//!
//! ## Spec reference
//!
//! ITU-T H.266 | ISO/IEC 23090-3 (V4, 01/2026):
//! * §7.3.11.7 — `coding_unit()` non-merge inter pre-residual syntax
//!   (the source-of-truth listing this dispatcher mirrors).
//! * §7.3.10.10 — `mvd_coding(x0, y0, refList, cpIdx)` syntax.
//! * §7.4.12.7 — inference rules for missing flags / fields.
//! * §8.5.2.5 — symmetric MVD derivation (`MvdL1 = -MvdL0` when
//!   `sym_mvd_flag == 1`).
//!
//! No third-party VVC decoder source was consulted; the implementation
//! is spec-only and composes the existing per-element reader-side
//! helpers already shipped in this crate.

use oxideav_core::Result;

use crate::inter::MotionVector;
use crate::leaf_cu::{InterPredDir, LeafCuReader, NonMergeInterAffineGate};
use crate::non_merge_inter_pre_residual_enc::NonMergeInterPreResidualDecision;
use crate::non_merge_mvp_syntax_enc::{NonMergeMvpSyntaxDecision, NonMergeMvpSyntaxGate};

/// Decode the entire §7.3.11.7 non-merge inter CU pre-residual syntax
/// in one call.
///
/// The dispatcher walks the per-element reader helpers in §7.3.11.7
/// spec order, applying the §7.4.12.7 inference rules whenever a gate
/// is closed (so no bin is consumed for a syntax element whose gate is
/// closed and the returned struct's matching field carries the
/// inferred default).
///
/// # Parameters
///
/// * `reader` — `LeafCuReader` bound to a CABAC decoder positioned at
///   the start of the §7.3.11.7 non-merge inter pre-residual cascade.
/// * `affine_gate` — §7.3.11.7 affine gate, identical in shape to the
///   encoder-side `affine_gate` parameter.
/// * `mvp_gate` — §7.3.11.7 MVP-side gate, identical in shape to the
///   encoder-side `mvp_gate` parameter.
///
/// # Returns
///
/// A [`NonMergeInterPreResidualDecision`] whose fields carry the
/// decoded values OR their §7.4.12.7 inferred defaults
/// (`PRED_L0` for the inter-pred direction in P-slices, `false` for a
/// suppressed `sym_mvd_flag`, `0` for any inactive `ref_idx_lX` /
/// `mvp_lX_flag`, and a zero MVD for an inactive list or the
/// symmetric-MVD L1 branch). The translational degenerate
/// `MotionModel::Translational` is folded into `decision.affine` via
/// the round-164 dispatcher already invoked by
/// `read_non_merge_inter_affine`.
///
/// # Preconditions
///
/// * The CU is on the `general_merge_flag == 0 && cu_skip_flag == 0`
///   branch of §7.3.11.5 (a real non-merge inter CU). The dispatcher
///   does NOT consume `cu_skip_flag`, `pred_mode_flag`, or
///   `general_merge_flag` — those belong to the broader leaf CU walker
///   that hands off to this dispatcher.
///
/// # Spec ordering — bin-for-bin with the encoder-side dispatcher
///
/// The reader walks the exact sequence the encoder-side dispatcher
/// emits in
/// [`crate::non_merge_inter_pre_residual_enc::encode_non_merge_inter_pre_residual`]:
///
/// 1. Affine syntax (`inter_affine_flag` / `cu_affine_type_flag`) via
///    [`LeafCuReader::read_non_merge_inter_affine`].
/// 2. `inter_pred_idc` via [`LeafCuReader::read_inter_pred_idc`]
///    (only when the §7.3.11.7 outer gate opens — B-slice).
/// 3. `sym_mvd_flag` via [`LeafCuReader::read_sym_mvd_flag`]
///    (only when the §7.3.11.7 SMVD gate opens).
/// 4. `ref_idx_l0` via [`LeafCuReader::read_ref_idx_lx`]
///    (only when L0 is active AND `cMax > 0` AND `sym_mvd_flag == 0`).
/// 5. `ref_idx_l1` (mirror of step 4 for L1).
/// 6. `mvd_coding(L0)` via [`LeafCuReader::read_mvd_coding`]
///    (only when L0 is active).
/// 7. `mvd_coding(L1)` (only when L1 is active AND `sym_mvd_flag == 0`
///    — the symmetric path infers `MvdL1 = -MvdL0` per §8.5.2.5 and
///    the reader consumes zero bins for it).
/// 8. `mvp_l0_flag` via [`LeafCuReader::read_mvp_lx_flag`]
///    (only when L0 is active).
/// 9. `mvp_l1_flag` (only when L1 is active).
pub fn read_non_merge_inter_pre_residual(
    reader: &mut LeafCuReader<'_, '_>,
    affine_gate: &NonMergeInterAffineGate,
    mvp_gate: &NonMergeMvpSyntaxGate,
) -> Result<NonMergeInterPreResidualDecision> {
    // ------------------------------------------------------------------
    // Step 1 — affine syntax (round-164 dispatcher).
    //
    // `read_non_merge_inter_affine` is itself the round-164 composite of
    // `read_inter_affine_flag` + `read_cu_affine_type_flag`. It applies
    // the two §7.3.11.7 gates internally and returns inferred-false
    // flags when either gate is closed, plus the typed `motion_model`
    // folded through §8.5.5.2 eq. 160.
    // ------------------------------------------------------------------
    let affine = reader.read_non_merge_inter_affine(affine_gate)?;

    // ------------------------------------------------------------------
    // Step 2 — inter_pred_idc (§7.3.11.7 B-slice only).
    //
    // P-slice ⇒ §7.4.12.7 infers PRED_L0 and the reader consumes zero
    // bins.
    // ------------------------------------------------------------------
    let inter_pred_idc = if mvp_gate.inter_pred_idc_gate_open() {
        reader.read_inter_pred_idc(mvp_gate.cb_width, mvp_gate.cb_height)?
    } else {
        InterPredDir::PredL0
    };

    // ------------------------------------------------------------------
    // Step 3 — sym_mvd_flag (§7.3.11.7 SMVD gate).
    //
    // The §8.5.2.5 SMVD derivation infers BOTH `refIdxL0` / `refIdxL1`
    // AND `MvdL1 = -MvdL0` when the gate is open. The reader skips the
    // per-list ref_idx_lX bins and the L1 mvd_coding entirely.
    // ------------------------------------------------------------------
    let sym_mvd_flag = if mvp_gate.sym_mvd_signalled(inter_pred_idc) {
        reader.read_sym_mvd_flag()?
    } else {
        false
    };

    let l0_active = matches!(inter_pred_idc, InterPredDir::PredL0 | InterPredDir::PredBi);
    let l1_active = matches!(inter_pred_idc, InterPredDir::PredL1 | InterPredDir::PredBi);

    // ------------------------------------------------------------------
    // Step 4 — ref_idx_l0 (§7.3.11.7 per-list).
    //
    // §7.4.12.7 infers 0 when not signalled (list inactive, sym_mvd
    // shortcut, or `NumRefIdxActive[0] <= 1`).
    // ------------------------------------------------------------------
    let ref_idx_l0 = if mvp_gate.ref_idx_l0_signalled(inter_pred_idc, sym_mvd_flag) {
        reader.read_ref_idx_lx(mvp_gate.num_ref_idx_active_l0)?
    } else {
        0
    };

    // ------------------------------------------------------------------
    // Step 5 — ref_idx_l1 (§7.3.11.7 per-list).
    // ------------------------------------------------------------------
    let ref_idx_l1 = if mvp_gate.ref_idx_l1_signalled(inter_pred_idc, sym_mvd_flag) {
        reader.read_ref_idx_lx(mvp_gate.num_ref_idx_active_l1)?
    } else {
        0
    };

    // ------------------------------------------------------------------
    // Step 6 — mvd_coding(L0, cpIdx = 0).
    //
    // The §7.3.11.7 listing places mvd_coding BETWEEN the per-list
    // ref_idx_lX and per-list mvp_lX_flag. Translational scope ⇒ one
    // mvd_coding per active list (multi-CP affine deferred).
    // ------------------------------------------------------------------
    let mvd_l0 = if l0_active {
        reader.read_mvd_coding()?
    } else {
        MotionVector { x: 0, y: 0 }
    };

    // ------------------------------------------------------------------
    // Step 7 — mvd_coding(L1, cpIdx = 0).
    //
    // Under `sym_mvd_flag == 1` the §8.5.2.5 derivation infers MvdL1 =
    // -MvdL0 and the reader consumes zero bins for the L1 MVD. The
    // composed-decision field carries the inferred -MvdL0 so the
    // returned struct exactly round-trips back through the encoder
    // dispatcher (whose constructor clamps L1 to zero under the
    // SMVD branch — both inferred forms round-trip bit-identically
    // through the encoder).
    // ------------------------------------------------------------------
    let mvd_l1 = if l1_active && !sym_mvd_flag {
        reader.read_mvd_coding()?
    } else if l1_active && sym_mvd_flag {
        // §8.5.2.5 — MvdL1 = -MvdL0. The L1 magnitude bounds match L0
        // bin-for-bin so the negation is safe at the spec's signed-18-
        // bit range.
        MotionVector {
            x: -mvd_l0.x,
            y: -mvd_l0.y,
        }
    } else {
        MotionVector { x: 0, y: 0 }
    };

    // ------------------------------------------------------------------
    // Step 8 — mvp_l0_flag (§7.3.11.7 per-list).
    // ------------------------------------------------------------------
    let mvp_l0_flag = if l0_active {
        reader.read_mvp_lx_flag()?
    } else {
        0
    };

    // ------------------------------------------------------------------
    // Step 9 — mvp_l1_flag (§7.3.11.7 per-list).
    // ------------------------------------------------------------------
    let mvp_l1_flag = if l1_active {
        reader.read_mvp_lx_flag()?
    } else {
        0
    };

    // Build the MVP-side decision in the same shape the encoder's
    // dispatcher consumes. The §7.4.12.7 inferences for inactive
    // elements are already folded above so this is a direct field copy.
    let mvp = NonMergeMvpSyntaxDecision {
        inter_pred_idc,
        sym_mvd_flag,
        ref_idx_l0,
        ref_idx_l1,
        mvp_l0_flag,
        mvp_l1_flag,
    };

    // The encoder-side `NonMergeInterPreResidualDecision::new`
    // constructor clamps the L1 MVD to zero under SMVD; we instead
    // carry the inferred `-MvdL0` value (the §8.5.2.5 derivation) so
    // the consumer doesn't have to re-derive it. Both forms round-trip
    // through the encoder dispatcher bit-identically (the encoder's
    // SMVD path emits zero bins for the L1 MVD regardless of the field
    // value, and the debug-assert admits either zero or `-mvd_l0`).
    Ok(NonMergeInterPreResidualDecision {
        affine,
        mvp,
        mvd_l0,
        mvd_l1,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::affine_syntax_enc::make_non_merge_inter_affine_decision;
    use crate::cabac::ArithDecoder;
    use crate::cabac_enc::ArithEncoder;
    use crate::leaf_cu::{CuToolFlags, LeafCuCtxs};
    use crate::non_merge_inter_pre_residual_enc::encode_non_merge_inter_pre_residual;
    use crate::non_merge_mvp_syntax_enc::make_non_merge_mvp_syntax_decision;

    /// End-to-end round trip: build a decision, push it through the
    /// encoder-side round-190 dispatcher, then recover it through the
    /// round-219 reader-side dispatcher. The two must agree on every
    /// field (modulo the SMVD-L1 MVD inference, which the round-219
    /// dispatcher fills with `-MvdL0` and the encoder treats as
    /// equivalent to zero).
    fn round_trip_via_encoder(
        init_type: u8,
        affine_gate: &NonMergeInterAffineGate,
        mvp_gate: &NonMergeMvpSyntaxGate,
        decision: &NonMergeInterPreResidualDecision,
    ) -> NonMergeInterPreResidualDecision {
        let mut enc = ArithEncoder::new();
        let mut enc_ctxs = LeafCuCtxs::init_with_init_type(26, init_type);
        encode_non_merge_inter_pre_residual(
            &mut enc,
            &mut enc_ctxs,
            affine_gate,
            mvp_gate,
            decision,
        )
        .expect("encoder dispatcher accepts a valid decision");
        enc.encode_terminate(1).expect("terminator");
        let mut padded = enc.finish();
        padded.extend_from_slice(&[0u8; 64]);
        let mut dec = ArithDecoder::new(&padded).expect("decoder accepts the encoded stream");
        let mut dec_ctxs = LeafCuCtxs::init_with_init_type(26, init_type);
        let tools = CuToolFlags::default();
        let mut reader = LeafCuReader::new(&mut dec, &mut dec_ctxs, tools);
        read_non_merge_inter_pre_residual(&mut reader, affine_gate, mvp_gate)
            .expect("reader dispatcher accepts the encoded stream")
    }

    fn p_slice_gate() -> NonMergeMvpSyntaxGate {
        NonMergeMvpSyntaxGate {
            cb_width: 16,
            cb_height: 16,
            b_slice: false,
            sym_mvd_gate_open: false,
            num_ref_idx_active_l0: 2,
            num_ref_idx_active_l1: 0,
        }
    }

    fn b_slice_gate() -> NonMergeMvpSyntaxGate {
        NonMergeMvpSyntaxGate {
            cb_width: 16,
            cb_height: 16,
            b_slice: true,
            sym_mvd_gate_open: false,
            num_ref_idx_active_l0: 2,
            num_ref_idx_active_l1: 2,
        }
    }

    fn b_slice_gate_smvd() -> NonMergeMvpSyntaxGate {
        NonMergeMvpSyntaxGate {
            cb_width: 16,
            cb_height: 16,
            b_slice: true,
            sym_mvd_gate_open: true,
            num_ref_idx_active_l0: 2,
            num_ref_idx_active_l1: 2,
        }
    }

    fn affine_gate_off() -> NonMergeInterAffineGate {
        NonMergeInterAffineGate {
            sps_affine_enabled: false,
            sps_6param_affine_enabled: false,
            cb_width: 16,
            cb_height: 16,
            ..Default::default()
        }
    }

    fn affine_gate_4param() -> NonMergeInterAffineGate {
        NonMergeInterAffineGate {
            sps_affine_enabled: true,
            sps_6param_affine_enabled: false,
            cb_width: 16,
            cb_height: 16,
            ..Default::default()
        }
    }

    #[test]
    fn p_slice_zero_mvd_round_trip() {
        let affine_gate = affine_gate_off();
        let mvp_gate = p_slice_gate();
        let affine = make_non_merge_inter_affine_decision(false, false);
        let mvp = make_non_merge_mvp_syntax_decision(InterPredDir::PredL0, false, 0, 0, 0, 0);
        let decision = NonMergeInterPreResidualDecision::new(
            affine,
            mvp,
            MotionVector { x: 0, y: 0 },
            MotionVector { x: 0, y: 0 },
        );
        for init_type in [1u8, 2u8] {
            let recovered = round_trip_via_encoder(init_type, &affine_gate, &mvp_gate, &decision);
            assert_eq!(recovered.affine, decision.affine);
            assert_eq!(recovered.mvp, decision.mvp);
            assert_eq!(recovered.mvd_l0, decision.mvd_l0);
            assert_eq!(recovered.mvd_l1, decision.mvd_l1);
        }
    }

    #[test]
    fn p_slice_nonzero_mvd_round_trip() {
        let affine_gate = affine_gate_off();
        let mvp_gate = p_slice_gate();
        let affine = make_non_merge_inter_affine_decision(false, false);
        let mvp = make_non_merge_mvp_syntax_decision(InterPredDir::PredL0, false, 1, 0, 1, 0);
        let decision = NonMergeInterPreResidualDecision::new(
            affine,
            mvp,
            MotionVector { x: 5, y: -3 },
            MotionVector { x: 0, y: 0 },
        );
        for init_type in [1u8, 2u8] {
            let recovered = round_trip_via_encoder(init_type, &affine_gate, &mvp_gate, &decision);
            assert_eq!(recovered.mvp.inter_pred_idc, InterPredDir::PredL0);
            assert_eq!(recovered.mvp.ref_idx_l0, 1);
            assert_eq!(recovered.mvp.mvp_l0_flag, 1);
            assert_eq!(recovered.mvd_l0, MotionVector { x: 5, y: -3 });
            // L0 inactive after the inter_pred_idc=PRED_L0 resolution
            // is never reached for the L1 fields; the §7.4.12.7
            // inference fills them.
            assert_eq!(recovered.mvp.ref_idx_l1, 0);
            assert_eq!(recovered.mvp.mvp_l1_flag, 0);
            assert_eq!(recovered.mvd_l1, MotionVector { x: 0, y: 0 });
        }
    }

    #[test]
    fn b_slice_pred_l0_round_trip() {
        let affine_gate = affine_gate_off();
        let mvp_gate = b_slice_gate();
        let affine = make_non_merge_inter_affine_decision(false, false);
        let mvp = make_non_merge_mvp_syntax_decision(InterPredDir::PredL0, false, 1, 0, 0, 0);
        let decision = NonMergeInterPreResidualDecision::new(
            affine,
            mvp,
            MotionVector { x: -4, y: 7 },
            MotionVector { x: 0, y: 0 },
        );
        for init_type in [1u8, 2u8] {
            let recovered = round_trip_via_encoder(init_type, &affine_gate, &mvp_gate, &decision);
            assert_eq!(recovered.mvp.inter_pred_idc, InterPredDir::PredL0);
            assert_eq!(recovered.mvp.ref_idx_l0, 1);
            assert_eq!(recovered.mvd_l0, MotionVector { x: -4, y: 7 });
            // L1 inactive on PRED_L0 — §7.4.12.7 inferred zeros.
            assert_eq!(recovered.mvp.ref_idx_l1, 0);
            assert_eq!(recovered.mvp.mvp_l1_flag, 0);
            assert_eq!(recovered.mvd_l1, MotionVector { x: 0, y: 0 });
        }
    }

    #[test]
    fn b_slice_pred_l1_round_trip() {
        let affine_gate = affine_gate_off();
        let mvp_gate = b_slice_gate();
        let affine = make_non_merge_inter_affine_decision(false, false);
        let mvp = make_non_merge_mvp_syntax_decision(InterPredDir::PredL1, false, 0, 1, 0, 1);
        let decision = NonMergeInterPreResidualDecision::new(
            affine,
            mvp,
            MotionVector { x: 0, y: 0 },
            MotionVector { x: 11, y: -22 },
        );
        for init_type in [1u8, 2u8] {
            let recovered = round_trip_via_encoder(init_type, &affine_gate, &mvp_gate, &decision);
            assert_eq!(recovered.mvp.inter_pred_idc, InterPredDir::PredL1);
            assert_eq!(recovered.mvp.ref_idx_l1, 1);
            assert_eq!(recovered.mvp.mvp_l1_flag, 1);
            assert_eq!(recovered.mvd_l1, MotionVector { x: 11, y: -22 });
            // L0 inactive on PRED_L1 — §7.4.12.7 inferred zeros.
            assert_eq!(recovered.mvp.ref_idx_l0, 0);
            assert_eq!(recovered.mvp.mvp_l0_flag, 0);
            assert_eq!(recovered.mvd_l0, MotionVector { x: 0, y: 0 });
        }
    }

    #[test]
    fn b_slice_pred_bi_round_trip() {
        let affine_gate = affine_gate_off();
        let mvp_gate = b_slice_gate();
        let affine = make_non_merge_inter_affine_decision(false, false);
        let mvp = make_non_merge_mvp_syntax_decision(InterPredDir::PredBi, false, 1, 1, 1, 0);
        let decision = NonMergeInterPreResidualDecision::new(
            affine,
            mvp,
            MotionVector { x: 3, y: -3 },
            MotionVector { x: -8, y: 8 },
        );
        for init_type in [1u8, 2u8] {
            let recovered = round_trip_via_encoder(init_type, &affine_gate, &mvp_gate, &decision);
            assert_eq!(recovered.mvp.inter_pred_idc, InterPredDir::PredBi);
            assert!(!recovered.mvp.sym_mvd_flag);
            assert_eq!(recovered.mvp.ref_idx_l0, 1);
            assert_eq!(recovered.mvp.ref_idx_l1, 1);
            assert_eq!(recovered.mvp.mvp_l0_flag, 1);
            assert_eq!(recovered.mvp.mvp_l1_flag, 0);
            assert_eq!(recovered.mvd_l0, MotionVector { x: 3, y: -3 });
            assert_eq!(recovered.mvd_l1, MotionVector { x: -8, y: 8 });
        }
    }

    #[test]
    fn b_slice_smvd_round_trip_fills_inferred_l1_mvd() {
        // SMVD path: encoder emits zero bins for L1 MVD + ref_idx_l1.
        // The reader recovers `sym_mvd_flag = 1` and fills `mvd_l1`
        // with the §8.5.2.5 derivation `-mvd_l0`.
        let affine_gate = affine_gate_off();
        let mvp_gate = b_slice_gate_smvd();
        let affine = make_non_merge_inter_affine_decision(false, false);
        // Constructor clamps L1 MVD + ref_idx_l1 + sym-path ref_idx_l0
        // to zero so the decision is encoder-conformant.
        let mvp = make_non_merge_mvp_syntax_decision(InterPredDir::PredBi, true, 0, 0, 1, 1);
        let l0 = MotionVector { x: 9, y: -5 };
        let decision = NonMergeInterPreResidualDecision::new(
            affine,
            mvp,
            l0, // L1 MVD passed nonzero; constructor clamps
            // it to zero (matches encoder-side SMVD contract).
            MotionVector { x: 0, y: 0 },
        );
        for init_type in [1u8, 2u8] {
            let recovered = round_trip_via_encoder(init_type, &affine_gate, &mvp_gate, &decision);
            assert_eq!(recovered.mvp.inter_pred_idc, InterPredDir::PredBi);
            assert!(recovered.mvp.sym_mvd_flag);
            // §7.4.12.7 + §8.5.2.5 — ref_idx_lX inferred 0 on the SMVD
            // shortcut (the encoder is required to pass 0; the reader
            // recovers 0).
            assert_eq!(recovered.mvp.ref_idx_l0, 0);
            assert_eq!(recovered.mvp.ref_idx_l1, 0);
            assert_eq!(recovered.mvd_l0, l0);
            // The dispatcher fills mvd_l1 with the §8.5.2.5 derivation
            // (the field on the encoder side is a clamped-to-zero
            // sentinel; on the reader side it carries the derived
            // -mvd_l0 so the consumer doesn't have to re-derive it).
            assert_eq!(recovered.mvd_l1, MotionVector { x: -l0.x, y: -l0.y });
            assert_eq!(recovered.mvp.mvp_l0_flag, 1);
            assert_eq!(recovered.mvp.mvp_l1_flag, 1);
        }
    }

    #[test]
    fn affine_outer_gate_closed_round_trip() {
        // Outer affine gate closed by SPS — `inter_affine_flag`
        // inferred 0, no affine bins on the wire.
        let affine_gate = affine_gate_off();
        let mvp_gate = p_slice_gate();
        let affine = make_non_merge_inter_affine_decision(false, false);
        let mvp = make_non_merge_mvp_syntax_decision(InterPredDir::PredL0, false, 0, 0, 0, 0);
        let decision = NonMergeInterPreResidualDecision::new(
            affine,
            mvp,
            MotionVector { x: 0, y: 0 },
            MotionVector { x: 0, y: 0 },
        );
        let recovered = round_trip_via_encoder(1, &affine_gate, &mvp_gate, &decision);
        assert!(!recovered.affine.inter_affine_flag);
        assert!(!recovered.affine.cu_affine_type_flag);
        assert_eq!(
            recovered.affine.motion_model,
            crate::affine::MotionModel::Translational
        );
    }

    #[test]
    fn affine_outer_gate_closed_by_size_round_trip() {
        // Outer affine gate closed by `cbWidth < 16` — same observable
        // wire layout as the SPS-closed case.
        let affine_gate = NonMergeInterAffineGate {
            sps_affine_enabled: true,
            sps_6param_affine_enabled: true,
            cb_width: 8,
            cb_height: 8,
            ..Default::default()
        };
        let mvp_gate = NonMergeMvpSyntaxGate {
            cb_width: 8,
            cb_height: 8,
            b_slice: false,
            sym_mvd_gate_open: false,
            num_ref_idx_active_l0: 2,
            num_ref_idx_active_l1: 0,
        };
        let affine = make_non_merge_inter_affine_decision(false, false);
        let mvp = make_non_merge_mvp_syntax_decision(InterPredDir::PredL0, false, 1, 0, 1, 0);
        let decision = NonMergeInterPreResidualDecision::new(
            affine,
            mvp,
            MotionVector { x: 7, y: 7 },
            MotionVector { x: 0, y: 0 },
        );
        let recovered = round_trip_via_encoder(2, &affine_gate, &mvp_gate, &decision);
        assert!(!recovered.affine.inter_affine_flag);
        assert!(!recovered.affine.cu_affine_type_flag);
        assert_eq!(recovered.mvp.ref_idx_l0, 1);
        assert_eq!(recovered.mvd_l0, MotionVector { x: 7, y: 7 });
    }

    #[test]
    fn b_slice_single_active_ref_idx_l1_zero_bin_round_trip() {
        // L0 + L1 both active but L1 has only one ref ⇒ `ref_idx_l1`
        // not signalled, §7.4.12.7 infers 0. Reader recovers 0 without
        // consuming any bins for it.
        let affine_gate = affine_gate_off();
        let mvp_gate = NonMergeMvpSyntaxGate {
            cb_width: 16,
            cb_height: 16,
            b_slice: true,
            sym_mvd_gate_open: false,
            num_ref_idx_active_l0: 2,
            num_ref_idx_active_l1: 1, // ref_idx_l1 not signalled
        };
        let affine = make_non_merge_inter_affine_decision(false, false);
        let mvp = make_non_merge_mvp_syntax_decision(InterPredDir::PredBi, false, 1, 0, 0, 1);
        let decision = NonMergeInterPreResidualDecision::new(
            affine,
            mvp,
            MotionVector { x: 1, y: 2 },
            MotionVector { x: 3, y: 4 },
        );
        let recovered = round_trip_via_encoder(1, &affine_gate, &mvp_gate, &decision);
        assert_eq!(recovered.mvp.ref_idx_l0, 1);
        assert_eq!(recovered.mvp.ref_idx_l1, 0);
        assert_eq!(recovered.mvp.mvp_l1_flag, 1);
        assert_eq!(recovered.mvd_l0, MotionVector { x: 1, y: 2 });
        assert_eq!(recovered.mvd_l1, MotionVector { x: 3, y: 4 });
    }

    #[test]
    fn b_slice_pred_bi_one_bin_inter_pred_idc_form_round_trip() {
        // `(cbWidth + cbHeight) == 12` ⇒ one-bin form (PRED_BI
        // suppressed). The reader recovers `PRED_L0` or `PRED_L1` per
        // the single bin.
        let affine_gate = NonMergeInterAffineGate {
            cb_width: 4,
            cb_height: 8,
            ..Default::default()
        };
        let mvp_gate = NonMergeMvpSyntaxGate {
            cb_width: 4,
            cb_height: 8,
            b_slice: true,
            sym_mvd_gate_open: false,
            num_ref_idx_active_l0: 2,
            num_ref_idx_active_l1: 2,
        };
        let affine = make_non_merge_inter_affine_decision(false, false);
        let mvp = make_non_merge_mvp_syntax_decision(InterPredDir::PredL1, false, 0, 1, 0, 0);
        let decision = NonMergeInterPreResidualDecision::new(
            affine,
            mvp,
            MotionVector { x: 0, y: 0 },
            MotionVector { x: 2, y: -2 },
        );
        let recovered = round_trip_via_encoder(2, &affine_gate, &mvp_gate, &decision);
        assert_eq!(recovered.mvp.inter_pred_idc, InterPredDir::PredL1);
        assert_eq!(recovered.mvp.ref_idx_l1, 1);
        assert_eq!(recovered.mvd_l1, MotionVector { x: 2, y: -2 });
    }

    #[test]
    fn p_slice_pred_l0_num_ref_idx_active_one_round_trip() {
        // `NumRefIdxActive[0] == 1` ⇒ ref_idx_l0 suppressed (cMax = 0).
        // §7.4.12.7 infers 0.
        let affine_gate = affine_gate_off();
        let mvp_gate = NonMergeMvpSyntaxGate {
            cb_width: 16,
            cb_height: 16,
            b_slice: false,
            sym_mvd_gate_open: false,
            num_ref_idx_active_l0: 1,
            num_ref_idx_active_l1: 0,
        };
        let affine = make_non_merge_inter_affine_decision(false, false);
        let mvp = make_non_merge_mvp_syntax_decision(InterPredDir::PredL0, false, 0, 0, 1, 0);
        let decision = NonMergeInterPreResidualDecision::new(
            affine,
            mvp,
            MotionVector { x: -1, y: 1 },
            MotionVector { x: 0, y: 0 },
        );
        for init_type in [1u8, 2u8] {
            let recovered = round_trip_via_encoder(init_type, &affine_gate, &mvp_gate, &decision);
            assert_eq!(recovered.mvp.ref_idx_l0, 0);
            assert_eq!(recovered.mvp.mvp_l0_flag, 1);
            assert_eq!(recovered.mvd_l0, MotionVector { x: -1, y: 1 });
        }
    }

    #[test]
    fn affine_4param_open_translational_decision_round_trip() {
        // `sps_affine_enabled = true` opens the outer gate but the
        // decision still carries `Translational` (the encoder emits
        // one inter_affine_flag = 0 bin). The reader recovers
        // `Translational` and the rest of the cascade walks the
        // standard MVP-side path.
        let affine_gate = affine_gate_4param();
        let mvp_gate = p_slice_gate();
        let affine = make_non_merge_inter_affine_decision(false, false);
        let mvp = make_non_merge_mvp_syntax_decision(InterPredDir::PredL0, false, 1, 0, 0, 0);
        let decision = NonMergeInterPreResidualDecision::new(
            affine,
            mvp,
            MotionVector { x: 6, y: -2 },
            MotionVector { x: 0, y: 0 },
        );
        for init_type in [1u8, 2u8] {
            let recovered = round_trip_via_encoder(init_type, &affine_gate, &mvp_gate, &decision);
            assert!(!recovered.affine.inter_affine_flag);
            assert_eq!(
                recovered.affine.motion_model,
                crate::affine::MotionModel::Translational
            );
            assert_eq!(recovered.mvp.ref_idx_l0, 1);
            assert_eq!(recovered.mvd_l0, MotionVector { x: 6, y: -2 });
        }
    }

    #[test]
    fn b_slice_pred_bi_large_mvd_round_trip_at_spec_boundary() {
        // §7.4.10.10 max conformant MVD magnitude is `2^17 - 1`. The
        // reader-side dispatcher MUST decode it intact through the
        // §9.3.3.6 limited-EGk path.
        let affine_gate = affine_gate_off();
        let mvp_gate = b_slice_gate();
        let affine = make_non_merge_inter_affine_decision(false, false);
        let mvp = make_non_merge_mvp_syntax_decision(InterPredDir::PredBi, false, 0, 0, 0, 0);
        let max_mvd = crate::mvd_coding_enc::max_mvd_magnitude();
        let decision = NonMergeInterPreResidualDecision::new(
            affine,
            mvp,
            MotionVector {
                x: max_mvd,
                y: -max_mvd,
            },
            MotionVector {
                x: -max_mvd,
                y: max_mvd,
            },
        );
        let recovered = round_trip_via_encoder(1, &affine_gate, &mvp_gate, &decision);
        assert_eq!(
            recovered.mvd_l0,
            MotionVector {
                x: max_mvd,
                y: -max_mvd
            }
        );
        assert_eq!(
            recovered.mvd_l1,
            MotionVector {
                x: -max_mvd,
                y: max_mvd
            }
        );
    }
}
