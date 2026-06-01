//! Round-190 — encoder-side composite walker for the §7.3.11.7 non-merge
//! inter CU pre-residual syntax.
//!
//! Companion to the three pre-existing public encoder-syntax modules:
//!
//! * Round-177 — [`crate::affine_syntax_enc`] covers the §7.3.11.7
//!   `inter_affine_flag` + `cu_affine_type_flag` pair plus their
//!   dispatcher [`crate::affine_syntax_enc::encode_non_merge_inter_affine`].
//! * Round-183 — [`crate::non_merge_mvp_syntax_enc`] covers
//!   `inter_pred_idc` / `sym_mvd_flag` / `ref_idx_lX` / `mvp_lX_flag`
//!   plus their dispatcher
//!   [`crate::non_merge_mvp_syntax_enc::encode_non_merge_mvp_syntax`].
//! * Round-187 — [`crate::mvd_coding_enc`] covers
//!   [`crate::mvd_coding_enc::encode_mvd_coding`] for one §7.3.10.10
//!   `mvd_coding(x0, y0, refList, cpIdx)` structure.
//!
//! Each prior round's report explicitly named the same follow-up: a
//! "CTU-walker / encoder-pipeline call-site that consumes the three
//! encoder-syntax modules together". This round delivers that surface:
//! one public function, [`encode_non_merge_inter_pre_residual`], that
//! drives the full §7.3.11.7 non-merge inter pre-residual cascade in
//! one call, leaving the residual / `cu_coded_flag` /
//! `transform_tree()` / `cu_qp_delta` tail to later rounds. The
//! dispatcher is deliberately purely compositional: it does not
//! emit any bin that the three underlying dispatchers cannot already
//! emit on their own. Its job is to interleave them in the spec order
//! the reader-side §7.3.11.7 walk expects.
//!
//! ## Scope
//!
//! `inter_affine_flag` ⇒ `cu_affine_type_flag`
//! ⇒ `inter_pred_idc` ⇒ `sym_mvd_flag`
//! ⇒ `ref_idx_l0` ⇒ `ref_idx_l1`
//! ⇒ `mvd_coding(L0, cpIdx=0)` ⇒ `mvd_coding(L1, cpIdx=0)`
//! ⇒ `mvp_l0_flag` ⇒ `mvp_l1_flag`.
//!
//! The §7.3.11.7 listing places `mvp_lX_flag` AFTER the `mvd_coding()`
//! invocations on each list, so this module hand-walks the per-element
//! encoder helpers in interleaved spec order rather than calling
//! `encode_non_merge_mvp_syntax` (which collapses the entire MVP-side
//! cascade — ref-idx then mvp — into one pass and explicitly steps
//! across mvd_coding per the spec note in
//! [`crate::non_merge_mvp_syntax_enc::encode_non_merge_mvp_syntax`]).
//!
//! Per-list `mvd_coding` is invoked at most once per list (cpIdx=0).
//! Multi-CP-MV affine MVD emission (`numCpMv > 1` ⇒ one `mvd_coding()`
//! per control point per list, per §7.3.11.7) is a follow-up.
//! `amvr_flag` / `bcw_idx` reads sit between the §7.3.11.7
//! pre-residual cascade covered here and the residual tree; they are
//! also follow-ups and are *not* emitted by this dispatcher.
//!
//! ## Round-trip
//!
//! The wire is bit-identical to the corresponding sequence of reader
//! calls in §7.3.11.7 spec order:
//!
//! 1. [`crate::leaf_cu::LeafCuReader::read_non_merge_inter_affine`]
//! 2. [`crate::leaf_cu::LeafCuReader::read_inter_pred_idc`]
//!    (only when the §7.3.11.7 outer gate opens — B-slice)
//! 3. [`crate::leaf_cu::LeafCuReader::read_sym_mvd_flag`]
//!    (only when the §7.3.11.7 SMVD gate opens)
//! 4. [`crate::leaf_cu::LeafCuReader::read_ref_idx_lx`] for L0
//!    (only when L0 is active AND `cMax > 0` AND `sym_mvd_flag == 0`)
//! 5. [`crate::leaf_cu::LeafCuReader::read_ref_idx_lx`] for L1
//!    (only when L1 is active AND `cMax > 0` AND `sym_mvd_flag == 0`)
//! 6. [`crate::leaf_cu::LeafCuReader::read_mvd_coding`] for L0
//!    (only when L0 is active)
//! 7. [`crate::leaf_cu::LeafCuReader::read_mvd_coding`] for L1
//!    (only when L1 is active AND `sym_mvd_flag == 0` — the symmetric
//!    path infers `MvdL1 = -MvdL0` per §8.5.2.5 and the reader
//!    consumes zero bins for it)
//! 8. [`crate::leaf_cu::LeafCuReader::read_mvp_lx_flag`] for L0
//!    (only when L0 is active)
//! 9. [`crate::leaf_cu::LeafCuReader::read_mvp_lx_flag`] for L1
//!    (only when L1 is active)
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
//! No third-party VVC encoder source was consulted; the implementation
//! is spec-only and composes the existing round-177 / round-183 /
//! round-187 encoder-side code already shipped in this crate.

use oxideav_core::Result;

use crate::affine::{derive_motion_model_idc, MotionModel};
use crate::affine_syntax_enc::encode_non_merge_inter_affine;
use crate::cabac_enc::ArithEncoder;
use crate::inter::MotionVector;
use crate::leaf_cu::{
    InterPredDir, LeafCuCtxs, NonMergeInterAffineDecision, NonMergeInterAffineGate,
};
use crate::mvd_coding_enc::encode_mvd_coding;
use crate::non_merge_mvp_syntax_enc::{
    encode_inter_pred_idc, encode_mvp_lx_flag, encode_ref_idx_lx, encode_sym_mvd_flag,
    NonMergeMvpSyntaxDecision, NonMergeMvpSyntaxGate,
};

/// Round-190 — input to [`encode_non_merge_inter_pre_residual`].
///
/// Carries both gates (`affine` + `mvp`) and both decisions (`affine`
/// + `mvp`) plus the per-list MVDs. The struct is intentionally pure
/// data: the §7.3.11.7 control flow is in the dispatcher itself.
///
/// The MVDs are the *post-AMVR* `lMvd[c]` values per
/// §7.4.10.10 — the bitstream-conformant signed-18-bit range
/// `[-2^17, 2^17 - 1]` matches
/// [`crate::mvd_coding_enc::max_mvd_magnitude`]. The §7.4.11.6 AMVR
/// shift, when signalled, is applied by [`crate::amvr`] *before* the
/// dispatcher runs.
#[derive(Clone, Copy, Debug)]
pub struct NonMergeInterPreResidualDecision {
    /// Affine syntax decision (§7.3.11.7 affine pair).
    pub affine: NonMergeInterAffineDecision,
    /// MVP-side syntax decision (§7.3.11.7 inter_pred_idc /
    /// sym_mvd_flag / ref_idx_lX / mvp_lX_flag).
    pub mvp: NonMergeMvpSyntaxDecision,
    /// `lMvd[L0]` — post-AMVR `(x, y)`. Only consulted when the L0
    /// list is active (`inter_pred_idc ∈ {PRED_L0, PRED_BI}`). When
    /// inactive callers should pass `MotionVector { x: 0, y: 0 }`; a
    /// non-zero value is silently ignored on the wire.
    pub mvd_l0: MotionVector,
    /// `lMvd[L1]` — post-AMVR `(x, y)`. Only consulted when the L1
    /// list is active AND `sym_mvd_flag == 0`. When `sym_mvd_flag ==
    /// 1` the spec derives `MvdL1 = -MvdL0` per §8.5.2.5 and the
    /// dispatcher emits zero bins for the L1 MVD; callers should
    /// either pass `-mvd_l0` (so a debug-assertion mirrors the
    /// invariant) or `MotionVector { x: 0, y: 0 }`. A non-zero value
    /// that disagrees with `-mvd_l0` is silently ignored on the wire.
    pub mvd_l1: MotionVector,
}

impl NonMergeInterPreResidualDecision {
    /// Convenience constructor that clamps inactive-list MVDs to zero
    /// so the struct can't carry stale state past the §7.3.11.7 gates.
    ///
    /// Matches the shape of
    /// [`crate::non_merge_mvp_syntax_enc::make_non_merge_mvp_syntax_decision`]:
    /// the L1 MVD is also clamped under `sym_mvd_flag == 1` (the
    /// symmetric path derives it from L0 per §8.5.2.5).
    pub fn new(
        affine: NonMergeInterAffineDecision,
        mvp: NonMergeMvpSyntaxDecision,
        mvd_l0: MotionVector,
        mvd_l1: MotionVector,
    ) -> Self {
        let l0 = matches!(
            mvp.inter_pred_idc,
            InterPredDir::PredL0 | InterPredDir::PredBi
        );
        let l1 = matches!(
            mvp.inter_pred_idc,
            InterPredDir::PredL1 | InterPredDir::PredBi
        );
        let l0_mvd = if l0 {
            mvd_l0
        } else {
            MotionVector { x: 0, y: 0 }
        };
        let l1_mvd = if l1 && !mvp.sym_mvd_flag {
            mvd_l1
        } else {
            MotionVector { x: 0, y: 0 }
        };
        Self {
            affine,
            mvp,
            mvd_l0: l0_mvd,
            mvd_l1: l1_mvd,
        }
    }
}

/// Encode the entire §7.3.11.7 non-merge inter CU pre-residual syntax
/// in one call.
///
/// The dispatcher walks the per-element encoder helpers in §7.3.11.7
/// spec order, applying the standard §7.4.12.7 inference rules so that
/// no bin is emitted for a syntax element whose gate is closed.
///
/// # Parameters
///
/// * `enc` — shared CABAC encoder state.
/// * `ctxs` — slice-scope context bundle. Mutated as the dispatcher
///   walks the affine / MVP-side context bins.
/// * `affine_gate` — §7.3.11.7 affine gate (round-164 / round-177).
/// * `mvp_gate` — §7.3.11.7 MVP-side gate (round-183).
/// * `decision` — combined affine + MVP + per-list `lMvd` decision.
///
/// # Preconditions
///
/// * The CU is on the `general_merge_flag == 0 && cu_skip_flag == 0`
///   branch of §7.3.11.5 (i.e. a real non-merge inter CU). The
///   dispatcher does NOT emit `cu_skip_flag`, `pred_mode_flag`, or
///   `general_merge_flag` — those belong to the broader leaf CU walk
///   that calls into this dispatcher on the non-merge branch.
/// * `decision.affine.motion_model` agrees with the two raw flags via
///   §8.5.5.2 eq. 160 (`derive_motion_model_idc`).
/// * `decision.mvp` carries spec-conformant inferences for any
///   element whose gate is closed (`PRED_L0` for P-slices, `false`
///   for `sym_mvd_flag` outside the SMVD branch, `0` for any inactive
///   reference index / `mvp_lX_flag`). The
///   [`NonMergeInterPreResidualDecision::new`] constructor takes care
///   of this for the MVDs; the matching MVP-side clamping happens in
///   [`crate::non_merge_mvp_syntax_enc::make_non_merge_mvp_syntax_decision`].
///
/// # Inferred-flag invariants (debug-only)
///
/// Mirrors round-177's and round-183's debug-assert pattern: when a
/// gate is closed the dispatcher checks that the caller passed the
/// §7.4.12.7-inferred value. Violations panic in debug builds; in
/// release builds the wire stream may not round-trip to the same
/// decision.
pub fn encode_non_merge_inter_pre_residual(
    enc: &mut ArithEncoder,
    ctxs: &mut LeafCuCtxs,
    affine_gate: &NonMergeInterAffineGate,
    mvp_gate: &NonMergeMvpSyntaxGate,
    decision: &NonMergeInterPreResidualDecision,
) -> Result<()> {
    // Round-trip safety: the typed `motion_model` field MUST agree
    // with the two raw affine flag bools per §8.5.5.2 eq. 160.
    debug_assert_eq!(
        decision.affine.motion_model,
        derive_motion_model_idc(
            decision.affine.inter_affine_flag,
            decision.affine.cu_affine_type_flag
        ),
        "NonMergeInterPreResidualDecision.affine.motion_model disagrees with its flag pair"
    );
    // Multi-CP-MV affine MVD emission is a follow-up; this round only
    // supports translational MVD (one mvd_coding per active list).
    debug_assert!(
        matches!(decision.affine.motion_model, MotionModel::Translational),
        "round-190 dispatcher supports translational motion only; \
         affine multi-CP MVD emission is a follow-up"
    );

    // ------------------------------------------------------------------
    // Step 1 — affine syntax (round-177 dispatcher).
    //
    // Emits zero, one, or two bins per §7.3.11.7 + §7.4.12.7 inferences.
    // ------------------------------------------------------------------
    encode_non_merge_inter_affine(enc, ctxs, affine_gate, &decision.affine)?;

    // ------------------------------------------------------------------
    // Step 2 — inter_pred_idc (§7.3.11.7 B-slice only).
    //
    // P-slice ⇒ §7.4.12.7 infers PRED_L0 and we emit zero bins.
    // ------------------------------------------------------------------
    let outer = mvp_gate.inter_pred_idc_gate_open();
    if outer {
        encode_inter_pred_idc(
            enc,
            ctxs,
            decision.mvp.inter_pred_idc,
            mvp_gate.cb_width,
            mvp_gate.cb_height,
        )?;
    } else {
        debug_assert_eq!(
            decision.mvp.inter_pred_idc,
            InterPredDir::PredL0,
            "P-slice → §7.4.12.7 requires inter_pred_idc = PRED_L0"
        );
    }
    let effective_inter_pred_idc = if outer {
        decision.mvp.inter_pred_idc
    } else {
        InterPredDir::PredL0
    };

    // ------------------------------------------------------------------
    // Step 3 — sym_mvd_flag (§7.3.11.7 SMVD gate).
    //
    // The §8.5.2.5 SMVD derivation infers BOTH `refIdxL0` / `refIdxL1`
    // AND `MvdL1 = -MvdL0` when the gate is open. The reader skips the
    // per-list ref_idx_lX bins and the L1 mvd_coding entirely; this
    // dispatcher mirrors that.
    // ------------------------------------------------------------------
    if mvp_gate.sym_mvd_signalled(effective_inter_pred_idc) {
        encode_sym_mvd_flag(enc, ctxs, decision.mvp.sym_mvd_flag)?;
    } else {
        debug_assert!(
            !decision.mvp.sym_mvd_flag,
            "sym_mvd_flag gate closed → §7.4.12.7 requires sym_mvd_flag = 0"
        );
    }
    let effective_sym_mvd_flag = if mvp_gate.sym_mvd_signalled(effective_inter_pred_idc) {
        decision.mvp.sym_mvd_flag
    } else {
        false
    };

    let l0_active = matches!(
        effective_inter_pred_idc,
        InterPredDir::PredL0 | InterPredDir::PredBi
    );
    let l1_active = matches!(
        effective_inter_pred_idc,
        InterPredDir::PredL1 | InterPredDir::PredBi
    );

    // ------------------------------------------------------------------
    // Step 4 — ref_idx_l0 (§7.3.11.7 per-list).
    // ------------------------------------------------------------------
    if mvp_gate.ref_idx_l0_signalled(effective_inter_pred_idc, effective_sym_mvd_flag) {
        encode_ref_idx_lx(
            enc,
            ctxs,
            decision.mvp.ref_idx_l0,
            mvp_gate.num_ref_idx_active_l0,
        )?;
    } else {
        debug_assert_eq!(
            decision.mvp.ref_idx_l0, 0,
            "ref_idx_l0 not signalled → §7.4.12.7 requires ref_idx_l0 = 0"
        );
    }

    // ------------------------------------------------------------------
    // Step 5 — ref_idx_l1 (§7.3.11.7 per-list).
    // ------------------------------------------------------------------
    if mvp_gate.ref_idx_l1_signalled(effective_inter_pred_idc, effective_sym_mvd_flag) {
        encode_ref_idx_lx(
            enc,
            ctxs,
            decision.mvp.ref_idx_l1,
            mvp_gate.num_ref_idx_active_l1,
        )?;
    } else {
        debug_assert_eq!(
            decision.mvp.ref_idx_l1, 0,
            "ref_idx_l1 not signalled → §7.4.12.7 requires ref_idx_l1 = 0"
        );
    }

    // ------------------------------------------------------------------
    // Step 6 — mvd_coding(L0, cpIdx = 0).
    //
    // The §7.3.11.7 listing places mvd_coding BETWEEN the per-list
    // ref_idx_lX and per-list mvp_lX_flag. The translational scope of
    // this round implies one mvd_coding per active list (multi-CP
    // affine is deferred).
    // ------------------------------------------------------------------
    if l0_active {
        encode_mvd_coding(enc, ctxs, decision.mvd_l0)?;
    } else {
        debug_assert_eq!(
            (decision.mvd_l0.x, decision.mvd_l0.y),
            (0, 0),
            "L0 inactive → mvd_l0 must be zero per the dispatcher's contract"
        );
    }

    // ------------------------------------------------------------------
    // Step 7 — mvd_coding(L1, cpIdx = 0).
    //
    // Under sym_mvd_flag == 1 the §8.5.2.5 derivation infers MvdL1 =
    // -MvdL0 and the reader consumes zero bins for the L1 MVD; the
    // dispatcher mirrors that.
    // ------------------------------------------------------------------
    if l1_active && !effective_sym_mvd_flag {
        encode_mvd_coding(enc, ctxs, decision.mvd_l1)?;
    } else if l1_active && effective_sym_mvd_flag {
        // Caller-conformance check: an SMVD CU's mvd_l1 should
        // either be the inferred `-mvd_l0` or zero (the
        // [`NonMergeInterPreResidualDecision::new`] clamp). Either way
        // the dispatcher emits zero bins for the L1 MVD.
        let inferred_match =
            decision.mvd_l1.x == -decision.mvd_l0.x && decision.mvd_l1.y == -decision.mvd_l0.y;
        let zero = decision.mvd_l1.x == 0 && decision.mvd_l1.y == 0;
        debug_assert!(
            inferred_match || zero,
            "sym_mvd_flag = 1 → mvd_l1 must be inferred -mvd_l0 or zero (was ({}, {}) for mvd_l0 = ({}, {}))",
            decision.mvd_l1.x,
            decision.mvd_l1.y,
            decision.mvd_l0.x,
            decision.mvd_l0.y
        );
    } else {
        debug_assert_eq!(
            (decision.mvd_l1.x, decision.mvd_l1.y),
            (0, 0),
            "L1 inactive → mvd_l1 must be zero per the dispatcher's contract"
        );
    }

    // ------------------------------------------------------------------
    // Step 8 — mvp_l0_flag (§7.3.11.7 per-list).
    // ------------------------------------------------------------------
    if l0_active {
        encode_mvp_lx_flag(enc, ctxs, decision.mvp.mvp_l0_flag)?;
    } else {
        debug_assert_eq!(
            decision.mvp.mvp_l0_flag, 0,
            "L0 inactive → §7.4.12.7 requires mvp_l0_flag = 0"
        );
    }

    // ------------------------------------------------------------------
    // Step 9 — mvp_l1_flag (§7.3.11.7 per-list).
    // ------------------------------------------------------------------
    if l1_active {
        encode_mvp_lx_flag(enc, ctxs, decision.mvp.mvp_l1_flag)?;
    } else {
        debug_assert_eq!(
            decision.mvp.mvp_l1_flag, 0,
            "L1 inactive → §7.4.12.7 requires mvp_l1_flag = 0"
        );
    }

    Ok(())
}

/// Round-195 — composite walker variant that ALSO emits the
/// §7.3.10.10 `amvr_flag` / `amvr_precision_idx` cascade after the
/// per-list `mvp_lX_flag` step (and before the deferred `bcw_idx` +
/// residual tree).
///
/// Spec ordering (§7.3.11.7) places the AMVR cascade between the
/// last `mvp_lX_flag` and the `bcw_idx` / residual tree. This module
/// covers everything up to and including AMVR; `bcw_idx` and the
/// residual tree remain follow-ups for the broader non-merge inter
/// CU encoder.
///
/// The dispatcher composes [`encode_non_merge_inter_pre_residual`]
/// with [`crate::amvr_enc::encode_amvr_inter_gated`]. The
/// `amvr_gate` parameter MUST agree with the per-list / per-CP MVD
/// non-zero state implied by `decision` — the round-trip tests pin
/// this. `amvr_decision` is the encoder mirror of the round-193
/// reader's `(amvr_flag, amvr_precision_idx, AmvrShift)` triple,
/// surfaced as [`crate::amvr_enc::AmvrDecision`].
///
/// The MVDs in `decision` (`mvd_l0` / `mvd_l1`) are still the
/// *post-AMVR* `lMvd[c]` values per §7.4.10.10 — the dispatcher
/// emits them via [`encode_mvd_coding`] without consulting
/// `amvr_decision.shift`. The AMVR shift is signalled separately by
/// the AMVR cascade and the reader recovers it via the round-193
/// `read_amvr_inter_gated` dispatcher.
///
/// # Preconditions
///
/// All preconditions of [`encode_non_merge_inter_pre_residual`] apply
/// unchanged. Additionally:
///
/// * `amvr_gate.inter_affine_flag` MUST match
///   `decision.affine.inter_affine_flag` (the AMVR arm follows the
///   affine flag). The dispatcher debug-asserts this.
/// * `amvr_gate.any_mvd_l0_l1_nonzero` MUST match the actual
///   non-zero state of `decision.mvd_l0` / `decision.mvd_l1` on the
///   regular arm.
/// * `amvr_gate.any_mvd_cp_l0_l1_nonzero` SHOULD reflect the affine
///   per-CP MVD non-zero state; this round's translational-only
///   scope means callers on the affine arm typically set it from a
///   round-187 multi-CP-MV emission that hasn't landed yet (the
///   dispatcher still emits AMVR bins per the gate verbatim).
pub fn encode_non_merge_inter_pre_residual_with_amvr(
    enc: &mut ArithEncoder,
    ctxs: &mut LeafCuCtxs,
    affine_gate: &NonMergeInterAffineGate,
    mvp_gate: &NonMergeMvpSyntaxGate,
    amvr_gate: &crate::leaf_cu::AmvrGate,
    decision: &NonMergeInterPreResidualDecision,
    amvr_decision: &crate::amvr_enc::AmvrDecision,
) -> Result<()> {
    debug_assert_eq!(
        amvr_gate.inter_affine_flag, decision.affine.inter_affine_flag,
        "amvr_gate.inter_affine_flag must match decision.affine.inter_affine_flag \
         (got amvr_gate = {}, decision = {})",
        amvr_gate.inter_affine_flag, decision.affine.inter_affine_flag,
    );
    // Steps 1–9 — the pre-AMVR cascade. Identical to round-190.
    encode_non_merge_inter_pre_residual(enc, ctxs, affine_gate, mvp_gate, decision)?;
    // Step 10 — the §7.3.10.10 AMVR cascade.
    crate::amvr_enc::encode_amvr_inter_gated(enc, ctxs, amvr_gate, amvr_decision)?;
    Ok(())
}

/// Round-201 — composite walker variant that ALSO emits the
/// §7.3.10.5 `bcw_idx[x0][y0]` cascade after the §7.3.10.10 AMVR
/// step (and before the deferred residual / `cu_coded_flag` /
/// `transform_tree()` / `cu_qp_delta` tail).
///
/// Spec ordering (§7.3.11.7 / §7.3.10.5) places the BCW cascade after
/// AMVR and before the residual tree. This module covers everything
/// up to and including BCW; the residual tree remains a follow-up for
/// the broader non-merge inter CU encoder.
///
/// The dispatcher composes [`encode_non_merge_inter_pre_residual_with_amvr`]
/// with [`crate::bcw_idx_enc::encode_bcw_idx_gated`]. The `bcw_gate`
/// parameter MUST agree with the per-CB state implied by `decision`
/// (in particular `bcw_gate.inter_pred_idc` must match the
/// inter_pred_idc resolved by the dispatcher's MVP-side cascade); the
/// round-trip tests pin this. `bcw_idx_value` is the raw `bcw_idx`
/// value the encoder picks for this CU — the same value the reader
/// recovers via [`crate::leaf_cu::LeafCuReader::read_bcw_idx_gated`]
/// when the gate is open, or the §7.4.12.5 inferred 0 when the gate
/// is closed.
///
/// # Preconditions
///
/// All preconditions of [`encode_non_merge_inter_pre_residual_with_amvr`]
/// apply unchanged. Additionally:
///
/// * `bcw_gate.cb_width` / `bcw_gate.cb_height` SHOULD match the
///   `affine_gate.cb_width` / `affine_gate.cb_height` (this is the
///   same CU). The dispatcher debug-asserts this.
/// * `bcw_gate.inter_pred_idc` MUST match the inter_pred_idc the MVP-
///   side dispatcher emitted (the value in `decision.mvp.inter_pred_idc`
///   for B-slices, or the P-slice inferred `PRED_L0` when the outer
///   gate is closed). The dispatcher debug-asserts this.
/// * When `bcw_gate.is_open()` is false the caller MUST pass
///   `bcw_idx_value = 0` (the §7.4.12.5 inferred default). The
///   dispatcher debug-asserts this.
pub fn encode_non_merge_inter_pre_residual_with_amvr_and_bcw(
    enc: &mut ArithEncoder,
    ctxs: &mut LeafCuCtxs,
    affine_gate: &NonMergeInterAffineGate,
    mvp_gate: &NonMergeMvpSyntaxGate,
    amvr_gate: &crate::leaf_cu::AmvrGate,
    bcw_gate: &crate::leaf_cu::BcwIdxGate,
    decision: &NonMergeInterPreResidualDecision,
    amvr_decision: &crate::amvr_enc::AmvrDecision,
    bcw_idx_value: u32,
) -> Result<()> {
    debug_assert_eq!(
        bcw_gate.cb_width, affine_gate.cb_width,
        "bcw_gate.cb_width must match affine_gate.cb_width (same CU)"
    );
    debug_assert_eq!(
        bcw_gate.cb_height, affine_gate.cb_height,
        "bcw_gate.cb_height must match affine_gate.cb_height (same CU)"
    );
    // The MVP-side dispatcher resolves inter_pred_idc per the §7.3.11.7
    // outer gate (PRED_L0 inferred for P-slices). bcw_gate.inter_pred_idc
    // MUST match that resolved value.
    let effective_inter_pred_idc = if mvp_gate.inter_pred_idc_gate_open() {
        decision.mvp.inter_pred_idc
    } else {
        InterPredDir::PredL0
    };
    debug_assert_eq!(
        bcw_gate.inter_pred_idc,
        Some(effective_inter_pred_idc),
        "bcw_gate.inter_pred_idc must match the MVP-side resolved inter_pred_idc \
         (got bcw_gate = {:?}, effective = {:?})",
        bcw_gate.inter_pred_idc,
        effective_inter_pred_idc,
    );

    // Steps 1–10 — the pre-BCW cascade. Identical to round-195.
    encode_non_merge_inter_pre_residual_with_amvr(
        enc,
        ctxs,
        affine_gate,
        mvp_gate,
        amvr_gate,
        decision,
        amvr_decision,
    )?;
    // Step 11 — the §7.3.10.5 BCW cascade.
    crate::bcw_idx_enc::encode_bcw_idx_gated(enc, ctxs, bcw_gate, bcw_idx_value)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::affine_syntax_enc::make_non_merge_inter_affine_decision;
    use crate::amvr_enc::AmvrDecision;
    use crate::cabac::ArithDecoder;
    use crate::leaf_cu::{AmvrGate, CuToolFlags, LeafCuReader};
    use crate::non_merge_mvp_syntax_enc::make_non_merge_mvp_syntax_decision;

    /// Decoded snapshot for the reader-side §7.3.11.7 walk used to
    /// verify the dispatcher's wire layout.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct ReaderSnapshot {
        affine: NonMergeInterAffineDecision,
        inter_pred_idc: InterPredDir,
        sym_mvd_flag: bool,
        ref_idx_l0: u32,
        ref_idx_l1: u32,
        mvd_l0: MotionVector,
        mvd_l1: MotionVector,
        mvp_l0_flag: u32,
        mvp_l1_flag: u32,
    }

    /// Drive the dispatcher and read back the resulting stream
    /// through the reader-side `LeafCuReader`. Walks the same
    /// §7.3.11.7 spec order the dispatcher emits.
    fn round_trip(
        init_type: u8,
        affine_gate: &NonMergeInterAffineGate,
        mvp_gate: &NonMergeMvpSyntaxGate,
        decision: &NonMergeInterPreResidualDecision,
    ) -> ReaderSnapshot {
        let mut enc = ArithEncoder::new();
        let mut enc_ctxs = LeafCuCtxs::init_with_init_type(26, init_type);
        encode_non_merge_inter_pre_residual(
            &mut enc,
            &mut enc_ctxs,
            affine_gate,
            mvp_gate,
            decision,
        )
        .expect("encode_non_merge_inter_pre_residual succeeds under a valid gate/decision");
        enc.encode_terminate(1).expect("terminator");
        let mut padded = enc.finish();
        padded.extend_from_slice(&[0u8; 64]);
        let mut dec = ArithDecoder::new(&padded).expect("decoder accepts the encoded stream");
        let mut dec_ctxs = LeafCuCtxs::init_with_init_type(26, init_type);
        let tools = CuToolFlags::default();
        let mut reader = LeafCuReader::new(&mut dec, &mut dec_ctxs, tools);

        // Step 1 — affine syntax.
        let affine = reader
            .read_non_merge_inter_affine(affine_gate)
            .expect("reader reads affine syntax");

        // Step 2 — inter_pred_idc.
        let inter_pred_idc = if mvp_gate.inter_pred_idc_gate_open() {
            reader
                .read_inter_pred_idc(mvp_gate.cb_width, mvp_gate.cb_height)
                .expect("reader reads inter_pred_idc")
        } else {
            InterPredDir::PredL0
        };

        // Step 3 — sym_mvd_flag.
        let sym_mvd_flag = if mvp_gate.sym_mvd_signalled(inter_pred_idc) {
            reader
                .read_sym_mvd_flag()
                .expect("reader reads sym_mvd_flag")
        } else {
            false
        };

        let l0_active = matches!(inter_pred_idc, InterPredDir::PredL0 | InterPredDir::PredBi);
        let l1_active = matches!(inter_pred_idc, InterPredDir::PredL1 | InterPredDir::PredBi);

        // Step 4 — ref_idx_l0.
        let ref_idx_l0 = if mvp_gate.ref_idx_l0_signalled(inter_pred_idc, sym_mvd_flag) {
            reader
                .read_ref_idx_lx(mvp_gate.num_ref_idx_active_l0)
                .expect("reader reads ref_idx_l0")
        } else {
            0
        };

        // Step 5 — ref_idx_l1.
        let ref_idx_l1 = if mvp_gate.ref_idx_l1_signalled(inter_pred_idc, sym_mvd_flag) {
            reader
                .read_ref_idx_lx(mvp_gate.num_ref_idx_active_l1)
                .expect("reader reads ref_idx_l1")
        } else {
            0
        };

        // Step 6 — mvd_coding(L0).
        let mvd_l0 = if l0_active {
            reader.read_mvd_coding().expect("reader reads mvd L0")
        } else {
            MotionVector { x: 0, y: 0 }
        };

        // Step 7 — mvd_coding(L1).
        let mvd_l1 = if l1_active && !sym_mvd_flag {
            reader.read_mvd_coding().expect("reader reads mvd L1")
        } else {
            MotionVector { x: 0, y: 0 }
        };

        // Step 8 — mvp_l0_flag.
        let mvp_l0_flag = if l0_active {
            reader.read_mvp_lx_flag().expect("reader reads mvp_l0_flag")
        } else {
            0
        };

        // Step 9 — mvp_l1_flag.
        let mvp_l1_flag = if l1_active {
            reader.read_mvp_lx_flag().expect("reader reads mvp_l1_flag")
        } else {
            0
        };

        ReaderSnapshot {
            affine,
            inter_pred_idc,
            sym_mvd_flag,
            ref_idx_l0,
            ref_idx_l1,
            mvd_l0,
            mvd_l1,
            mvp_l0_flag,
            mvp_l1_flag,
        }
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

    fn affine_gate_off() -> NonMergeInterAffineGate {
        NonMergeInterAffineGate {
            sps_affine_enabled: false,
            sps_6param_affine_enabled: false,
            cb_width: 16,
            cb_height: 16,
            ..Default::default()
        }
    }

    #[test]
    fn p_slice_translational_zero_mvd_round_trip() {
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
            let snap = round_trip(init_type, &affine_gate, &mvp_gate, &decision);
            assert_eq!(snap.affine, affine);
            assert_eq!(snap.inter_pred_idc, InterPredDir::PredL0);
            assert!(!snap.sym_mvd_flag);
            assert_eq!(snap.ref_idx_l0, 0);
            assert_eq!(snap.ref_idx_l1, 0);
            assert_eq!(snap.mvd_l0, MotionVector { x: 0, y: 0 });
            assert_eq!(snap.mvd_l1, MotionVector { x: 0, y: 0 });
            assert_eq!(snap.mvp_l0_flag, 0);
            assert_eq!(snap.mvp_l1_flag, 0);
        }
    }

    #[test]
    fn p_slice_translational_nonzero_mvd_round_trip() {
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
            let snap = round_trip(init_type, &affine_gate, &mvp_gate, &decision);
            assert_eq!(snap.ref_idx_l0, 1);
            assert_eq!(snap.mvp_l0_flag, 1);
            assert_eq!(snap.mvd_l0, MotionVector { x: 5, y: -3 });
            assert_eq!(snap.mvd_l1, MotionVector { x: 0, y: 0 });
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
        let snap = round_trip(1, &affine_gate, &mvp_gate, &decision);
        assert_eq!(snap.inter_pred_idc, InterPredDir::PredL0);
        assert_eq!(snap.ref_idx_l0, 1);
        assert_eq!(snap.ref_idx_l1, 0);
        assert_eq!(snap.mvd_l0, MotionVector { x: -4, y: 7 });
        assert_eq!(snap.mvd_l1, MotionVector { x: 0, y: 0 });
        assert_eq!(snap.mvp_l0_flag, 0);
        assert_eq!(snap.mvp_l1_flag, 0);
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
            MotionVector { x: 3, y: 9 },
        );
        let snap = round_trip(2, &affine_gate, &mvp_gate, &decision);
        assert_eq!(snap.inter_pred_idc, InterPredDir::PredL1);
        assert_eq!(snap.ref_idx_l0, 0);
        assert_eq!(snap.ref_idx_l1, 1);
        assert_eq!(snap.mvd_l0, MotionVector { x: 0, y: 0 });
        assert_eq!(snap.mvd_l1, MotionVector { x: 3, y: 9 });
        assert_eq!(snap.mvp_l1_flag, 1);
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
            MotionVector { x: -10, y: 11 },
            MotionVector { x: 12, y: -13 },
        );
        let snap = round_trip(1, &affine_gate, &mvp_gate, &decision);
        assert_eq!(snap.inter_pred_idc, InterPredDir::PredBi);
        assert_eq!(snap.ref_idx_l0, 1);
        assert_eq!(snap.ref_idx_l1, 1);
        assert_eq!(snap.mvd_l0, MotionVector { x: -10, y: 11 });
        assert_eq!(snap.mvd_l1, MotionVector { x: 12, y: -13 });
        assert_eq!(snap.mvp_l0_flag, 1);
        assert_eq!(snap.mvp_l1_flag, 0);
    }

    #[test]
    fn b_slice_sym_mvd_round_trip() {
        let affine_gate = affine_gate_off();
        let mvp_gate = NonMergeMvpSyntaxGate {
            sym_mvd_gate_open: true,
            ..b_slice_gate()
        };
        let affine = make_non_merge_inter_affine_decision(false, false);
        // Under sym_mvd the §8.5.2.5 derivation fixes refIdxLX and
        // infers MvdL1 = -MvdL0. The decision-maker constructor clamps
        // the ref_idx_lX fields to 0 on the sym path. The dispatcher
        // also emits zero bins for MvdL1; we pass `-MvdL0` to exercise
        // the inferred-match debug-assert path.
        let mvp = make_non_merge_mvp_syntax_decision(InterPredDir::PredBi, true, 0, 0, 1, 1);
        let mvd_l0 = MotionVector { x: 4, y: -8 };
        let decision = NonMergeInterPreResidualDecision::new(
            affine,
            mvp,
            mvd_l0,
            MotionVector {
                x: -mvd_l0.x,
                y: -mvd_l0.y,
            },
        );
        let snap = round_trip(1, &affine_gate, &mvp_gate, &decision);
        assert_eq!(snap.inter_pred_idc, InterPredDir::PredBi);
        assert!(snap.sym_mvd_flag);
        // Reader-side ref_idx is suppressed under sym_mvd; inferred 0.
        assert_eq!(snap.ref_idx_l0, 0);
        assert_eq!(snap.ref_idx_l1, 0);
        assert_eq!(snap.mvd_l0, mvd_l0);
        // L1 mvd is NOT on the wire under sym_mvd; reader-snap returns
        // the zero default.
        assert_eq!(snap.mvd_l1, MotionVector { x: 0, y: 0 });
        assert_eq!(snap.mvp_l0_flag, 1);
        assert_eq!(snap.mvp_l1_flag, 1);
    }

    #[test]
    fn b_slice_sym_mvd_zero_l1_mvd_round_trip() {
        // Alternative caller pattern: pass `MotionVector { 0, 0 }` for
        // the L1 MVD under sym_mvd_flag — the constructor clamps it.
        let affine_gate = affine_gate_off();
        let mvp_gate = NonMergeMvpSyntaxGate {
            sym_mvd_gate_open: true,
            ..b_slice_gate()
        };
        let affine = make_non_merge_inter_affine_decision(false, false);
        let mvp = make_non_merge_mvp_syntax_decision(InterPredDir::PredBi, true, 0, 0, 0, 0);
        let decision = NonMergeInterPreResidualDecision::new(
            affine,
            mvp,
            MotionVector { x: -1, y: 1 },
            MotionVector { x: 100, y: 100 }, // stale, gets clamped.
        );
        // The constructor's clamp put L1 MVD to zero.
        assert_eq!(decision.mvd_l1, MotionVector { x: 0, y: 0 });
        let snap = round_trip(1, &affine_gate, &mvp_gate, &decision);
        assert!(snap.sym_mvd_flag);
        assert_eq!(snap.mvd_l0, MotionVector { x: -1, y: 1 });
        assert_eq!(snap.mvd_l1, MotionVector { x: 0, y: 0 });
    }

    #[test]
    fn affine_outer_gate_closed_zero_affine_bins() {
        // Outer affine gate closed: no affine bins, decision must
        // carry the all-false inferences.
        let affine_gate = NonMergeInterAffineGate {
            sps_affine_enabled: false,
            sps_6param_affine_enabled: false,
            cb_width: 16,
            cb_height: 16,
            ..Default::default()
        };
        let mvp_gate = p_slice_gate();
        let affine = make_non_merge_inter_affine_decision(false, false);
        let mvp = make_non_merge_mvp_syntax_decision(InterPredDir::PredL0, false, 0, 0, 0, 0);
        let decision = NonMergeInterPreResidualDecision::new(
            affine,
            mvp,
            MotionVector { x: 0, y: 0 },
            MotionVector { x: 0, y: 0 },
        );
        let snap = round_trip(1, &affine_gate, &mvp_gate, &decision);
        assert_eq!(snap.affine.motion_model, MotionModel::Translational);
        assert!(!snap.affine.inter_affine_flag);
        assert!(!snap.affine.cu_affine_type_flag);
    }

    #[test]
    fn one_bin_inter_pred_idc_form_4x8() {
        // (cbWidth + cbHeight) == 12 — single-bin inter_pred_idc form.
        // PRED_BI is forbidden in this binarisation per Table 131.
        let affine_gate = affine_gate_off();
        let mvp_gate = NonMergeMvpSyntaxGate {
            cb_width: 4,
            cb_height: 8,
            b_slice: true,
            sym_mvd_gate_open: false,
            num_ref_idx_active_l0: 1,
            num_ref_idx_active_l1: 1,
        };
        let affine = make_non_merge_inter_affine_decision(false, false);
        let mvp = make_non_merge_mvp_syntax_decision(InterPredDir::PredL1, false, 0, 0, 1, 1);
        let decision = NonMergeInterPreResidualDecision::new(
            affine,
            mvp,
            MotionVector { x: 0, y: 0 },
            MotionVector { x: 2, y: -2 },
        );
        let snap = round_trip(1, &affine_gate, &mvp_gate, &decision);
        assert_eq!(snap.inter_pred_idc, InterPredDir::PredL1);
        assert_eq!(snap.mvd_l1, MotionVector { x: 2, y: -2 });
        assert_eq!(snap.mvp_l1_flag, 1);
    }

    #[test]
    fn ref_idx_lx_max1_suppressed() {
        // num_ref_idx_active = 1 ⇒ cMax = 0 ⇒ no ref_idx bin emitted
        // (the reader infers it as 0).
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
        let mvp = make_non_merge_mvp_syntax_decision(InterPredDir::PredL0, false, 0, 0, 0, 0);
        let decision = NonMergeInterPreResidualDecision::new(
            affine,
            mvp,
            MotionVector { x: 1, y: 1 },
            MotionVector { x: 0, y: 0 },
        );
        let snap = round_trip(1, &affine_gate, &mvp_gate, &decision);
        assert_eq!(snap.ref_idx_l0, 0);
        assert_eq!(snap.mvd_l0, MotionVector { x: 1, y: 1 });
    }

    #[test]
    fn p_slice_large_mvd_at_spec_max() {
        // Drive |lMvd| close to the §7.4.10.10 spec bound.
        let affine_gate = affine_gate_off();
        let mvp_gate = p_slice_gate();
        let affine = make_non_merge_inter_affine_decision(false, false);
        let mvp = make_non_merge_mvp_syntax_decision(InterPredDir::PredL0, false, 1, 0, 1, 0);
        let big = crate::mvd_coding_enc::max_mvd_magnitude();
        let decision = NonMergeInterPreResidualDecision::new(
            affine,
            mvp,
            MotionVector { x: big, y: -big },
            MotionVector { x: 0, y: 0 },
        );
        let snap = round_trip(1, &affine_gate, &mvp_gate, &decision);
        assert_eq!(snap.mvd_l0.x, big);
        assert_eq!(snap.mvd_l0.y, -big);
    }

    #[test]
    fn new_clamps_inactive_l1_mvd() {
        // Constructor contract: inactive L1 ⇒ mvd_l1 is forced to zero
        // regardless of the caller's stale value.
        let affine = make_non_merge_inter_affine_decision(false, false);
        let mvp = make_non_merge_mvp_syntax_decision(InterPredDir::PredL0, false, 0, 0, 0, 0);
        let decision = NonMergeInterPreResidualDecision::new(
            affine,
            mvp,
            MotionVector { x: 1, y: 2 },
            MotionVector { x: 99, y: -99 },
        );
        assert_eq!(decision.mvd_l1, MotionVector { x: 0, y: 0 });
    }

    #[test]
    fn new_clamps_inactive_l0_mvd() {
        // Symmetric to the L1 case: PRED_L1 ⇒ L0 is inactive and
        // mvd_l0 is forced to zero.
        let affine = make_non_merge_inter_affine_decision(false, false);
        let mvp = make_non_merge_mvp_syntax_decision(InterPredDir::PredL1, false, 0, 0, 0, 0);
        let decision = NonMergeInterPreResidualDecision::new(
            affine,
            mvp,
            MotionVector { x: 99, y: -99 },
            MotionVector { x: 1, y: 2 },
        );
        assert_eq!(decision.mvd_l0, MotionVector { x: 0, y: 0 });
        assert_eq!(decision.mvd_l1, MotionVector { x: 1, y: 2 });
    }

    #[test]
    fn new_clamps_sym_mvd_l1() {
        // Under sym_mvd_flag the constructor clamps L1 MVD to zero
        // (the spec derives it from L0, so the wire carries nothing).
        let affine = make_non_merge_inter_affine_decision(false, false);
        let mvp = make_non_merge_mvp_syntax_decision(InterPredDir::PredBi, true, 0, 0, 0, 0);
        let decision = NonMergeInterPreResidualDecision::new(
            affine,
            mvp,
            MotionVector { x: 7, y: -7 },
            MotionVector { x: 99, y: -99 },
        );
        // Even though L1 is active under PRED_BI, sym_mvd_flag clamps
        // the L1 mvd to zero.
        assert_eq!(decision.mvd_l1, MotionVector { x: 0, y: 0 });
    }

    // ---- Round-195: encode_non_merge_inter_pre_residual_with_amvr ----

    /// Reader-side composite snapshot extended with the §7.3.10.10
    /// AMVR triple consumed after `mvp_lX_flag`.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct ReaderSnapshotWithAmvr {
        base: ReaderSnapshot,
        amvr_flag: bool,
        amvr_precision_idx: u32,
        amvr_shift: crate::amvr::AmvrShift,
    }

    /// Drive the round-195 AMVR-aware dispatcher and read back the
    /// resulting stream through the reader-side `LeafCuReader`,
    /// finishing with `read_amvr_inter_gated` per §7.3.11.7 spec
    /// order.
    fn round_trip_with_amvr(
        init_type: u8,
        affine_gate: &NonMergeInterAffineGate,
        mvp_gate: &NonMergeMvpSyntaxGate,
        amvr_gate: &AmvrGate,
        decision: &NonMergeInterPreResidualDecision,
        amvr_decision: &AmvrDecision,
    ) -> ReaderSnapshotWithAmvr {
        let mut enc = ArithEncoder::new();
        let mut enc_ctxs = LeafCuCtxs::init_with_init_type(26, init_type);
        encode_non_merge_inter_pre_residual_with_amvr(
            &mut enc,
            &mut enc_ctxs,
            affine_gate,
            mvp_gate,
            amvr_gate,
            decision,
            amvr_decision,
        )
        .expect("encode_non_merge_inter_pre_residual_with_amvr succeeds");
        enc.encode_terminate(1).expect("terminator");
        let mut padded = enc.finish();
        padded.extend_from_slice(&[0u8; 64]);
        let mut dec = ArithDecoder::new(&padded).expect("decoder accepts the encoded stream");
        let mut dec_ctxs = LeafCuCtxs::init_with_init_type(26, init_type);
        let tools = CuToolFlags::default();
        let mut reader = LeafCuReader::new(&mut dec, &mut dec_ctxs, tools);

        // Step 1 — affine syntax.
        let affine = reader
            .read_non_merge_inter_affine(affine_gate)
            .expect("reader reads affine syntax");
        // Step 2 — inter_pred_idc.
        let inter_pred_idc = if mvp_gate.inter_pred_idc_gate_open() {
            reader
                .read_inter_pred_idc(mvp_gate.cb_width, mvp_gate.cb_height)
                .expect("reader reads inter_pred_idc")
        } else {
            InterPredDir::PredL0
        };
        // Step 3 — sym_mvd_flag.
        let sym_mvd_flag = if mvp_gate.sym_mvd_signalled(inter_pred_idc) {
            reader
                .read_sym_mvd_flag()
                .expect("reader reads sym_mvd_flag")
        } else {
            false
        };
        let l0_active = matches!(inter_pred_idc, InterPredDir::PredL0 | InterPredDir::PredBi);
        let l1_active = matches!(inter_pred_idc, InterPredDir::PredL1 | InterPredDir::PredBi);
        // Step 4 — ref_idx_l0.
        let ref_idx_l0 = if mvp_gate.ref_idx_l0_signalled(inter_pred_idc, sym_mvd_flag) {
            reader
                .read_ref_idx_lx(mvp_gate.num_ref_idx_active_l0)
                .expect("reader reads ref_idx_l0")
        } else {
            0
        };
        // Step 5 — ref_idx_l1.
        let ref_idx_l1 = if mvp_gate.ref_idx_l1_signalled(inter_pred_idc, sym_mvd_flag) {
            reader
                .read_ref_idx_lx(mvp_gate.num_ref_idx_active_l1)
                .expect("reader reads ref_idx_l1")
        } else {
            0
        };
        // Step 6 — mvd_coding(L0).
        let mvd_l0 = if l0_active {
            reader.read_mvd_coding().expect("reader reads mvd L0")
        } else {
            MotionVector { x: 0, y: 0 }
        };
        // Step 7 — mvd_coding(L1).
        let mvd_l1 = if l1_active && !sym_mvd_flag {
            reader.read_mvd_coding().expect("reader reads mvd L1")
        } else {
            MotionVector { x: 0, y: 0 }
        };
        // Step 8 — mvp_l0_flag.
        let mvp_l0_flag = if l0_active {
            reader.read_mvp_lx_flag().expect("reader reads mvp_l0_flag")
        } else {
            0
        };
        // Step 9 — mvp_l1_flag.
        let mvp_l1_flag = if l1_active {
            reader.read_mvp_lx_flag().expect("reader reads mvp_l1_flag")
        } else {
            0
        };
        // Step 10 (round-195) — §7.3.10.10 AMVR cascade.
        let (amvr_flag, amvr_precision_idx, amvr_shift) = reader
            .read_amvr_inter_gated(amvr_gate)
            .expect("reader reads amvr cascade");

        ReaderSnapshotWithAmvr {
            base: ReaderSnapshot {
                affine,
                inter_pred_idc,
                sym_mvd_flag,
                ref_idx_l0,
                ref_idx_l1,
                mvd_l0,
                mvd_l1,
                mvp_l0_flag,
                mvp_l1_flag,
            },
            amvr_flag,
            amvr_precision_idx,
            amvr_shift,
        }
    }

    #[test]
    fn p_slice_amvr_closed_gate_round_trip() {
        // P-slice, regular AMVR, all-zero MVDs → AMVR outer gate is
        // closed (no MVD non-zero) and the §7.4.12.7 inference fires
        // (amvr_flag = 0, amvr_precision_idx = 0, AmvrShift = 2).
        let affine_gate = affine_gate_off();
        let mvp_gate = p_slice_gate();
        let amvr_gate = AmvrGate {
            sps_amvr_enabled: true,
            sps_affine_amvr_enabled: false,
            inter_affine_flag: false,
            any_mvd_l0_l1_nonzero: false,
            any_mvd_cp_l0_l1_nonzero: false,
        };
        assert!(!amvr_gate.is_open());
        let affine = make_non_merge_inter_affine_decision(false, false);
        let mvp = make_non_merge_mvp_syntax_decision(InterPredDir::PredL0, false, 0, 0, 0, 0);
        let decision = NonMergeInterPreResidualDecision::new(
            affine,
            mvp,
            MotionVector { x: 0, y: 0 },
            MotionVector { x: 0, y: 0 },
        );
        let amvr_decision = AmvrDecision::default_inferred();
        for init_type in [1u8, 2] {
            let snap = round_trip_with_amvr(
                init_type,
                &affine_gate,
                &mvp_gate,
                &amvr_gate,
                &decision,
                &amvr_decision,
            );
            assert_eq!(snap.base.inter_pred_idc, InterPredDir::PredL0);
            assert_eq!(snap.base.mvd_l0, MotionVector { x: 0, y: 0 });
            assert!(!snap.amvr_flag);
            assert_eq!(snap.amvr_precision_idx, 0);
            assert_eq!(snap.amvr_shift, crate::amvr::AmvrShift(2));
        }
    }

    #[test]
    fn p_slice_amvr_open_regular_precision_2_round_trip() {
        // P-slice, regular AMVR open (sps + non-zero MVD), amvr_flag =
        // 1, prec = 2 (4-luma) → AmvrShift = 6. Round-trip across both
        // non-I initTypes.
        let affine_gate = affine_gate_off();
        let mvp_gate = p_slice_gate();
        let amvr_gate = AmvrGate {
            sps_amvr_enabled: true,
            sps_affine_amvr_enabled: false,
            inter_affine_flag: false,
            any_mvd_l0_l1_nonzero: true,
            any_mvd_cp_l0_l1_nonzero: false,
        };
        assert!(amvr_gate.is_open());
        let affine = make_non_merge_inter_affine_decision(false, false);
        let mvp = make_non_merge_mvp_syntax_decision(InterPredDir::PredL0, false, 1, 0, 1, 0);
        let decision = NonMergeInterPreResidualDecision::new(
            affine,
            mvp,
            MotionVector { x: 5, y: -3 },
            MotionVector { x: 0, y: 0 },
        );
        let amvr_decision = AmvrDecision::new(true, 2, false);
        for init_type in [1u8, 2] {
            let snap = round_trip_with_amvr(
                init_type,
                &affine_gate,
                &mvp_gate,
                &amvr_gate,
                &decision,
                &amvr_decision,
            );
            // Steps 1–9 still round-trip.
            assert_eq!(snap.base.mvd_l0, MotionVector { x: 5, y: -3 });
            assert_eq!(snap.base.ref_idx_l0, 1);
            assert_eq!(snap.base.mvp_l0_flag, 1);
            // Step 10 — AMVR triple lands.
            assert!(snap.amvr_flag);
            assert_eq!(snap.amvr_precision_idx, 2);
            assert_eq!(snap.amvr_shift, crate::amvr::AmvrShift(6));
        }
    }

    // ---- Round-201: encode_non_merge_inter_pre_residual_with_amvr_and_bcw ----

    /// Reader-side composite snapshot extended with the §7.3.10.5
    /// `bcw_idx` value consumed after the §7.3.10.10 AMVR cascade.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct ReaderSnapshotWithBcw {
        amvr: ReaderSnapshotWithAmvr,
        bcw_idx: u32,
    }

    /// Drive the round-201 BCW-aware dispatcher and read back the
    /// resulting stream through the reader-side `LeafCuReader`,
    /// finishing with `read_bcw_idx_gated` per §7.3.11.7 / §7.3.10.5
    /// spec order.
    #[allow(clippy::too_many_arguments)]
    fn round_trip_with_bcw(
        init_type: u8,
        affine_gate: &NonMergeInterAffineGate,
        mvp_gate: &NonMergeMvpSyntaxGate,
        amvr_gate: &AmvrGate,
        bcw_gate: &crate::leaf_cu::BcwIdxGate,
        decision: &NonMergeInterPreResidualDecision,
        amvr_decision: &AmvrDecision,
        bcw_idx_value: u32,
    ) -> ReaderSnapshotWithBcw {
        let mut enc = ArithEncoder::new();
        let mut enc_ctxs = LeafCuCtxs::init_with_init_type(26, init_type);
        encode_non_merge_inter_pre_residual_with_amvr_and_bcw(
            &mut enc,
            &mut enc_ctxs,
            affine_gate,
            mvp_gate,
            amvr_gate,
            bcw_gate,
            decision,
            amvr_decision,
            bcw_idx_value,
        )
        .expect("encode_non_merge_inter_pre_residual_with_amvr_and_bcw succeeds");
        enc.encode_terminate(1).expect("terminator");
        let mut padded = enc.finish();
        padded.extend_from_slice(&[0u8; 64]);
        let mut dec = ArithDecoder::new(&padded).expect("decoder accepts the encoded stream");
        let mut dec_ctxs = LeafCuCtxs::init_with_init_type(26, init_type);
        let tools = CuToolFlags::default();
        let mut reader = LeafCuReader::new(&mut dec, &mut dec_ctxs, tools);

        // Step 1 — affine syntax.
        let affine = reader
            .read_non_merge_inter_affine(affine_gate)
            .expect("reader reads affine syntax");
        // Step 2 — inter_pred_idc.
        let inter_pred_idc = if mvp_gate.inter_pred_idc_gate_open() {
            reader
                .read_inter_pred_idc(mvp_gate.cb_width, mvp_gate.cb_height)
                .expect("reader reads inter_pred_idc")
        } else {
            InterPredDir::PredL0
        };
        // Step 3 — sym_mvd_flag.
        let sym_mvd_flag = if mvp_gate.sym_mvd_signalled(inter_pred_idc) {
            reader
                .read_sym_mvd_flag()
                .expect("reader reads sym_mvd_flag")
        } else {
            false
        };
        let l0_active = matches!(inter_pred_idc, InterPredDir::PredL0 | InterPredDir::PredBi);
        let l1_active = matches!(inter_pred_idc, InterPredDir::PredL1 | InterPredDir::PredBi);
        // Step 4 — ref_idx_l0.
        let ref_idx_l0 = if mvp_gate.ref_idx_l0_signalled(inter_pred_idc, sym_mvd_flag) {
            reader
                .read_ref_idx_lx(mvp_gate.num_ref_idx_active_l0)
                .expect("reader reads ref_idx_l0")
        } else {
            0
        };
        // Step 5 — ref_idx_l1.
        let ref_idx_l1 = if mvp_gate.ref_idx_l1_signalled(inter_pred_idc, sym_mvd_flag) {
            reader
                .read_ref_idx_lx(mvp_gate.num_ref_idx_active_l1)
                .expect("reader reads ref_idx_l1")
        } else {
            0
        };
        // Step 6 — mvd_coding(L0).
        let mvd_l0 = if l0_active {
            reader.read_mvd_coding().expect("reader reads mvd L0")
        } else {
            MotionVector { x: 0, y: 0 }
        };
        // Step 7 — mvd_coding(L1).
        let mvd_l1 = if l1_active && !sym_mvd_flag {
            reader.read_mvd_coding().expect("reader reads mvd L1")
        } else {
            MotionVector { x: 0, y: 0 }
        };
        // Step 8 — mvp_l0_flag.
        let mvp_l0_flag = if l0_active {
            reader.read_mvp_lx_flag().expect("reader reads mvp_l0_flag")
        } else {
            0
        };
        // Step 9 — mvp_l1_flag.
        let mvp_l1_flag = if l1_active {
            reader.read_mvp_lx_flag().expect("reader reads mvp_l1_flag")
        } else {
            0
        };
        // Step 10 — §7.3.10.10 AMVR cascade.
        let (amvr_flag, amvr_precision_idx, amvr_shift) = reader
            .read_amvr_inter_gated(amvr_gate)
            .expect("reader reads amvr cascade");
        // Step 11 (round-201) — §7.3.10.5 BCW cascade.
        let bcw_idx = reader
            .read_bcw_idx_gated(*bcw_gate)
            .expect("reader reads bcw cascade");

        ReaderSnapshotWithBcw {
            amvr: ReaderSnapshotWithAmvr {
                base: ReaderSnapshot {
                    affine,
                    inter_pred_idc,
                    sym_mvd_flag,
                    ref_idx_l0,
                    ref_idx_l1,
                    mvd_l0,
                    mvd_l1,
                    mvp_l0_flag,
                    mvp_l1_flag,
                },
                amvr_flag,
                amvr_precision_idx,
                amvr_shift,
            },
            bcw_idx,
        }
    }

    /// Common open BCW gate for a B-slice 32x32 PRED_BI CU with no
    /// weighted-pred flags set.
    fn open_bcw_gate(no_backward_pred_flag: bool) -> crate::leaf_cu::BcwIdxGate {
        crate::leaf_cu::BcwIdxGate {
            sps_bcw_enabled: true,
            inter_pred_idc: Some(InterPredDir::PredBi),
            luma_weight_l0_flag: false,
            luma_weight_l1_flag: false,
            chroma_weight_l0_flag: false,
            chroma_weight_l1_flag: false,
            cb_width: 16,
            cb_height: 16,
            no_backward_pred_flag,
        }
    }

    #[test]
    fn p_slice_bcw_closed_gate_round_trip() {
        // P-slice → inter_pred_idc inferred PRED_L0 → BCW gate closed
        // (§7.3.10.5 requires PRED_BI). All-zero MVDs ⇒ AMVR gate
        // closed too. Reader recovers bcw_idx = 0.
        let affine_gate = affine_gate_off();
        let mvp_gate = p_slice_gate();
        let amvr_gate = AmvrGate {
            sps_amvr_enabled: true,
            sps_affine_amvr_enabled: false,
            inter_affine_flag: false,
            any_mvd_l0_l1_nonzero: false,
            any_mvd_cp_l0_l1_nonzero: false,
        };
        // BCW gate for P-slice: inter_pred_idc = PRED_L0 ⇒ closed.
        let bcw_gate = crate::leaf_cu::BcwIdxGate {
            sps_bcw_enabled: true,
            inter_pred_idc: Some(InterPredDir::PredL0),
            cb_width: 16,
            cb_height: 16,
            ..Default::default()
        };
        assert!(!bcw_gate.is_open());
        let affine = make_non_merge_inter_affine_decision(false, false);
        let mvp = make_non_merge_mvp_syntax_decision(InterPredDir::PredL0, false, 0, 0, 0, 0);
        let decision = NonMergeInterPreResidualDecision::new(
            affine,
            mvp,
            MotionVector { x: 0, y: 0 },
            MotionVector { x: 0, y: 0 },
        );
        let amvr_decision = AmvrDecision::default_inferred();
        for init_type in [1u8, 2] {
            let snap = round_trip_with_bcw(
                init_type,
                &affine_gate,
                &mvp_gate,
                &amvr_gate,
                &bcw_gate,
                &decision,
                &amvr_decision,
                0,
            );
            assert_eq!(snap.amvr.base.inter_pred_idc, InterPredDir::PredL0);
            assert!(!snap.amvr.amvr_flag);
            assert_eq!(snap.bcw_idx, 0);
        }
    }

    #[test]
    fn b_slice_bi_pred_bcw_open_round_trip_all_values_cmax_2() {
        // B-slice, PRED_BI, 16x16 ⇒ BCW gate open with
        // NoBackwardPredFlag = 0 ⇒ cMax = 2. AMVR closed via the
        // non-zero MVD branch toggled off. Exhaustive over the three
        // legal bcw_idx values.
        let affine_gate = affine_gate_off();
        let mvp_gate = b_slice_gate();
        let amvr_gate = AmvrGate {
            sps_amvr_enabled: true,
            sps_affine_amvr_enabled: false,
            inter_affine_flag: false,
            any_mvd_l0_l1_nonzero: true,
            any_mvd_cp_l0_l1_nonzero: false,
        };
        let bcw_gate = open_bcw_gate(false);
        assert!(bcw_gate.is_open());
        let affine = make_non_merge_inter_affine_decision(false, false);
        let mvp = make_non_merge_mvp_syntax_decision(InterPredDir::PredBi, false, 1, 1, 0, 0);
        let decision = NonMergeInterPreResidualDecision::new(
            affine,
            mvp,
            MotionVector { x: 1, y: -1 },
            MotionVector { x: -1, y: 1 },
        );
        let amvr_decision = AmvrDecision::new(false, 0, false);
        for init_type in [1u8, 2] {
            for bcw in 0..=2u32 {
                let snap = round_trip_with_bcw(
                    init_type,
                    &affine_gate,
                    &mvp_gate,
                    &amvr_gate,
                    &bcw_gate,
                    &decision,
                    &amvr_decision,
                    bcw,
                );
                assert_eq!(snap.amvr.base.inter_pred_idc, InterPredDir::PredBi);
                assert_eq!(snap.amvr.base.mvd_l0, MotionVector { x: 1, y: -1 });
                assert_eq!(snap.amvr.base.mvd_l1, MotionVector { x: -1, y: 1 });
                assert_eq!(
                    snap.bcw_idx, bcw,
                    "BCW round-trip failed at init={init_type} bcw={bcw}"
                );
            }
        }
    }

    #[test]
    fn b_slice_bi_pred_bcw_open_round_trip_all_values_cmax_4() {
        // B-slice, PRED_BI, 16x16 ⇒ BCW gate open with
        // NoBackwardPredFlag = 1 ⇒ cMax = 4. Exhaustive over the
        // five legal bcw_idx values, including the truncation point
        // value = 4.
        let affine_gate = affine_gate_off();
        let mvp_gate = b_slice_gate();
        let amvr_gate = AmvrGate {
            sps_amvr_enabled: false,
            sps_affine_amvr_enabled: false,
            inter_affine_flag: false,
            any_mvd_l0_l1_nonzero: true,
            any_mvd_cp_l0_l1_nonzero: false,
        };
        let bcw_gate = open_bcw_gate(true);
        assert!(bcw_gate.is_open());
        let affine = make_non_merge_inter_affine_decision(false, false);
        let mvp = make_non_merge_mvp_syntax_decision(InterPredDir::PredBi, false, 0, 0, 0, 0);
        let decision = NonMergeInterPreResidualDecision::new(
            affine,
            mvp,
            MotionVector { x: 2, y: 0 },
            MotionVector { x: 0, y: -2 },
        );
        let amvr_decision = AmvrDecision::default_inferred();
        for init_type in [1u8, 2] {
            for bcw in 0..=4u32 {
                let snap = round_trip_with_bcw(
                    init_type,
                    &affine_gate,
                    &mvp_gate,
                    &amvr_gate,
                    &bcw_gate,
                    &decision,
                    &amvr_decision,
                    bcw,
                );
                assert_eq!(
                    snap.bcw_idx, bcw,
                    "BCW cMax=4 round-trip failed at init={init_type} bcw={bcw}"
                );
            }
        }
    }

    #[test]
    fn b_slice_bi_pred_bcw_gate_closed_by_weighted_pred_round_trip() {
        // B-slice PRED_BI with luma_weight_l0_flag = 1 (explicit
        // weighted prediction on L0 luma) closes the BCW gate per
        // §7.3.10.5. Reader recovers the inferred bcw_idx = 0.
        let affine_gate = affine_gate_off();
        let mvp_gate = b_slice_gate();
        let amvr_gate = AmvrGate {
            sps_amvr_enabled: false,
            sps_affine_amvr_enabled: false,
            inter_affine_flag: false,
            any_mvd_l0_l1_nonzero: true,
            any_mvd_cp_l0_l1_nonzero: false,
        };
        let bcw_gate = crate::leaf_cu::BcwIdxGate {
            luma_weight_l0_flag: true,
            ..open_bcw_gate(false)
        };
        assert!(!bcw_gate.is_open());
        let affine = make_non_merge_inter_affine_decision(false, false);
        let mvp = make_non_merge_mvp_syntax_decision(InterPredDir::PredBi, false, 0, 0, 0, 0);
        let decision = NonMergeInterPreResidualDecision::new(
            affine,
            mvp,
            MotionVector { x: 3, y: -3 },
            MotionVector { x: -3, y: 3 },
        );
        let amvr_decision = AmvrDecision::default_inferred();
        let snap = round_trip_with_bcw(
            1,
            &affine_gate,
            &mvp_gate,
            &amvr_gate,
            &bcw_gate,
            &decision,
            &amvr_decision,
            0,
        );
        assert_eq!(snap.bcw_idx, 0);
    }

    #[test]
    fn b_slice_bi_pred_bcw_small_cu_round_trip() {
        // B-slice PRED_BI 16x8 ⇒ cb_w*cb_h = 128 < 256 ⇒ BCW gate
        // closed per §7.3.10.5. AMVR also closed (no MVD non-zero).
        let affine_gate = NonMergeInterAffineGate {
            cb_width: 16,
            cb_height: 8,
            ..Default::default()
        };
        let mvp_gate = NonMergeMvpSyntaxGate {
            cb_width: 16,
            cb_height: 8,
            b_slice: true,
            sym_mvd_gate_open: false,
            num_ref_idx_active_l0: 2,
            num_ref_idx_active_l1: 2,
        };
        let amvr_gate = AmvrGate {
            sps_amvr_enabled: false,
            sps_affine_amvr_enabled: false,
            inter_affine_flag: false,
            any_mvd_l0_l1_nonzero: false,
            any_mvd_cp_l0_l1_nonzero: false,
        };
        let bcw_gate = crate::leaf_cu::BcwIdxGate {
            sps_bcw_enabled: true,
            inter_pred_idc: Some(InterPredDir::PredBi),
            cb_width: 16,
            cb_height: 8,
            ..Default::default()
        };
        assert!(!bcw_gate.is_open());
        let affine = make_non_merge_inter_affine_decision(false, false);
        let mvp = make_non_merge_mvp_syntax_decision(InterPredDir::PredBi, false, 0, 0, 0, 0);
        let decision = NonMergeInterPreResidualDecision::new(
            affine,
            mvp,
            MotionVector { x: 0, y: 0 },
            MotionVector { x: 0, y: 0 },
        );
        let amvr_decision = AmvrDecision::default_inferred();
        let snap = round_trip_with_bcw(
            2,
            &affine_gate,
            &mvp_gate,
            &amvr_gate,
            &bcw_gate,
            &decision,
            &amvr_decision,
            0,
        );
        assert_eq!(snap.bcw_idx, 0);
    }

    #[test]
    fn b_slice_uni_pred_bcw_gate_closed_round_trip() {
        // B-slice PRED_L0 (uni-pred) → BCW gate closed per §7.3.10.5
        // (requires PRED_BI). Reader recovers bcw_idx = 0.
        let affine_gate = affine_gate_off();
        let mvp_gate = b_slice_gate();
        let amvr_gate = AmvrGate {
            sps_amvr_enabled: false,
            sps_affine_amvr_enabled: false,
            inter_affine_flag: false,
            any_mvd_l0_l1_nonzero: true,
            any_mvd_cp_l0_l1_nonzero: false,
        };
        let bcw_gate = crate::leaf_cu::BcwIdxGate {
            sps_bcw_enabled: true,
            inter_pred_idc: Some(InterPredDir::PredL0),
            cb_width: 16,
            cb_height: 16,
            ..Default::default()
        };
        assert!(!bcw_gate.is_open());
        let affine = make_non_merge_inter_affine_decision(false, false);
        let mvp = make_non_merge_mvp_syntax_decision(InterPredDir::PredL0, false, 1, 0, 0, 0);
        let decision = NonMergeInterPreResidualDecision::new(
            affine,
            mvp,
            MotionVector { x: 4, y: 4 },
            MotionVector { x: 0, y: 0 },
        );
        let amvr_decision = AmvrDecision::default_inferred();
        let snap = round_trip_with_bcw(
            1,
            &affine_gate,
            &mvp_gate,
            &amvr_gate,
            &bcw_gate,
            &decision,
            &amvr_decision,
            0,
        );
        assert_eq!(snap.amvr.base.inter_pred_idc, InterPredDir::PredL0);
        assert_eq!(snap.bcw_idx, 0);
    }

    #[test]
    fn b_slice_bi_pred_bcw_with_amvr_open_round_trip() {
        // Both AMVR and BCW gates open simultaneously: B-slice
        // PRED_BI 16x16 with non-zero MVDs, regular AMVR signalled
        // (prec = 1 ⇒ AmvrShift = 4 = 1-luma), bcw_idx = 1.
        let affine_gate = affine_gate_off();
        let mvp_gate = b_slice_gate();
        let amvr_gate = AmvrGate {
            sps_amvr_enabled: true,
            sps_affine_amvr_enabled: false,
            inter_affine_flag: false,
            any_mvd_l0_l1_nonzero: true,
            any_mvd_cp_l0_l1_nonzero: false,
        };
        assert!(amvr_gate.is_open());
        let bcw_gate = open_bcw_gate(false);
        assert!(bcw_gate.is_open());
        let affine = make_non_merge_inter_affine_decision(false, false);
        let mvp = make_non_merge_mvp_syntax_decision(InterPredDir::PredBi, false, 1, 1, 0, 1);
        let decision = NonMergeInterPreResidualDecision::new(
            affine,
            mvp,
            MotionVector { x: 7, y: -2 },
            MotionVector { x: -5, y: 3 },
        );
        let amvr_decision = AmvrDecision::new(true, 1, false);
        let snap = round_trip_with_bcw(
            1,
            &affine_gate,
            &mvp_gate,
            &amvr_gate,
            &bcw_gate,
            &decision,
            &amvr_decision,
            1,
        );
        assert_eq!(snap.amvr.base.inter_pred_idc, InterPredDir::PredBi);
        assert_eq!(snap.amvr.base.mvd_l0, MotionVector { x: 7, y: -2 });
        assert_eq!(snap.amvr.base.mvd_l1, MotionVector { x: -5, y: 3 });
        assert!(snap.amvr.amvr_flag);
        assert_eq!(snap.amvr.amvr_precision_idx, 1);
        assert_eq!(snap.amvr.amvr_shift, crate::amvr::AmvrShift(4));
        assert_eq!(snap.bcw_idx, 1);
    }
}
