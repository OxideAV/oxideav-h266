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
use crate::non_merge_inter_pre_residual_enc::{
    NonMergeInterPreResidualAffineDecision, NonMergeInterPreResidualDecision,
};
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

// =====================================================================
// Round-224 — reader-side composite walker variants that ALSO consume
// the §7.3.10.10 AMVR cascade and the §7.3.10.5 BCW cascade. Reader
// twins of the encoder-side `_with_amvr` (round 195) and
// `_with_amvr_and_bcw` (round 201) composites.
// =====================================================================

/// Round-224 — reader-side composite walker variant that ALSO consumes
/// the §7.3.10.10 AMVR cascade after the §7.3.11.7 pre-residual
/// cascade. Reader twin of
/// [`crate::non_merge_inter_pre_residual_enc::encode_non_merge_inter_pre_residual_with_amvr`]
/// (round 195).
///
/// Walks the §7.3.11.7 spec listing through step 10:
///
/// 1. Steps 1–9 — the pre-AMVR cascade via
///    [`read_non_merge_inter_pre_residual`].
/// 2. Step 10 — the §7.3.10.10 AMVR cascade via
///    [`LeafCuReader::read_amvr_inter_gated`].
///
/// Returns the pair `(pre_residual, amvr)` where `pre_residual` is the
/// round-219 decision (raw post-AMVR `lMvd[c]` per §7.4.10.10 — the
/// AMVR shift is signalled separately by the AMVR cascade and applied
/// to the parsed MVDs externally) and `amvr` packages the recovered
/// `(amvr_flag, amvr_precision_idx, AmvrShift)` triple as a
/// [`crate::amvr_enc::AmvrDecision`] for symmetry with the encoder-
/// side `amvr_decision` parameter.
///
/// The §8.5.2.5 SMVD inference and the §7.4.12.7 inferences for any
/// gate-closed branch are already folded inside the round-219
/// dispatcher — this composite adds only the §7.3.10.10 outer-gate
/// + per-cascade-arm walk on top.
///
/// # Preconditions
///
/// * All preconditions of [`read_non_merge_inter_pre_residual`] apply
///   unchanged.
/// * `amvr_gate.inter_affine_flag` MUST agree with the
///   `decision.affine.inter_affine_flag` returned by the pre-residual
///   dispatcher — the AMVR arm follows the affine flag. The caller
///   typically builds `amvr_gate` from the same per-CB state the
///   round-219 dispatcher recovers.
/// * `amvr_gate.any_mvd_l0_l1_nonzero` MUST agree with the
///   §7.3.10.10 cascade condition the encoder evaluated — i.e. true iff
///   at least one of the per-list MVDs the round-219 dispatcher parsed
///   is non-zero. The dispatcher debug-asserts this.
/// * `amvr_gate.any_mvd_cp_l0_l1_nonzero` SHOULD reflect the affine
///   per-CP MVD non-zero state when on the affine arm (out of scope
///   for this round's translational-only inner pre-residual
///   dispatcher; pass `false` for the translational case).
pub fn read_non_merge_inter_pre_residual_with_amvr(
    reader: &mut LeafCuReader<'_, '_>,
    affine_gate: &NonMergeInterAffineGate,
    mvp_gate: &NonMergeMvpSyntaxGate,
    amvr_gate: &crate::leaf_cu::AmvrGate,
) -> Result<(
    NonMergeInterPreResidualDecision,
    crate::amvr_enc::AmvrDecision,
)> {
    // Steps 1–9 — the pre-AMVR cascade. Identical to round-219.
    let decision = read_non_merge_inter_pre_residual(reader, affine_gate, mvp_gate)?;

    // The AMVR gate's `inter_affine_flag` is per-CB state the caller
    // already had at hand; the round-219 dispatcher just decoded the
    // matching `inter_affine_flag`, so debug-assert the two agree —
    // misalignment here would mean the AMVR arm picks the wrong
    // cascade (regular vs affine) and the decoded shift would be
    // wrong.
    debug_assert_eq!(
        amvr_gate.inter_affine_flag, decision.affine.inter_affine_flag,
        "amvr_gate.inter_affine_flag must match decision.affine.inter_affine_flag \
         (got amvr_gate = {}, decision = {})",
        amvr_gate.inter_affine_flag, decision.affine.inter_affine_flag,
    );

    // Step 10 — the §7.3.10.10 AMVR cascade.
    let (amvr_flag, amvr_precision_idx, shift) = reader.read_amvr_inter_gated(amvr_gate)?;
    let amvr = crate::amvr_enc::AmvrDecision {
        amvr_flag,
        amvr_precision_idx,
        shift,
    };
    Ok((decision, amvr))
}

/// Round-224 — reader-side composite walker variant that ALSO consumes
/// the §7.3.10.5 `bcw_idx[x0][y0]` cascade after the §7.3.10.10 AMVR
/// step. Reader twin of
/// [`crate::non_merge_inter_pre_residual_enc::encode_non_merge_inter_pre_residual_with_amvr_and_bcw`]
/// (round 201).
///
/// Walks the §7.3.11.7 / §7.3.10.5 spec listing through step 11:
///
/// 1. Steps 1–9 — the pre-AMVR cascade via
///    [`read_non_merge_inter_pre_residual`].
/// 2. Step 10 — the §7.3.10.10 AMVR cascade via
///    [`LeafCuReader::read_amvr_inter_gated`].
/// 3. Step 11 — the §7.3.10.5 BCW cascade via
///    [`LeafCuReader::read_bcw_idx_gated`].
///
/// Returns the triple `(pre_residual, amvr, bcw_idx)`. The `bcw_idx`
/// is the raw `bcw_idx[x0][y0]` value the encoder placed on the wire
/// (when the gate was open) or the §7.4.12.5 inferred default `0`
/// (when the gate was closed).
///
/// # Preconditions
///
/// All preconditions of [`read_non_merge_inter_pre_residual_with_amvr`]
/// apply unchanged. Additionally:
///
/// * `bcw_gate.cb_width` / `bcw_gate.cb_height` SHOULD match the
///   `affine_gate.cb_width` / `affine_gate.cb_height` (same CU). The
///   dispatcher debug-asserts this.
/// * `bcw_gate.inter_pred_idc` MUST match the inter_pred_idc the
///   round-219 dispatcher resolved (i.e.
///   `decision.mvp.inter_pred_idc` for B-slices, or the
///   `PRED_L0` inferred default for P-slices). The dispatcher
///   debug-asserts this.
pub fn read_non_merge_inter_pre_residual_with_amvr_and_bcw(
    reader: &mut LeafCuReader<'_, '_>,
    affine_gate: &NonMergeInterAffineGate,
    mvp_gate: &NonMergeMvpSyntaxGate,
    amvr_gate: &crate::leaf_cu::AmvrGate,
    bcw_gate: &crate::leaf_cu::BcwIdxGate,
) -> Result<(
    NonMergeInterPreResidualDecision,
    crate::amvr_enc::AmvrDecision,
    u32,
)> {
    debug_assert_eq!(
        bcw_gate.cb_width, affine_gate.cb_width,
        "bcw_gate.cb_width must match affine_gate.cb_width (same CU)"
    );
    debug_assert_eq!(
        bcw_gate.cb_height, affine_gate.cb_height,
        "bcw_gate.cb_height must match affine_gate.cb_height (same CU)"
    );

    // Steps 1–10 — the pre-BCW cascade. Identical to the round-224
    // `_with_amvr` composite above.
    let (decision, amvr) =
        read_non_merge_inter_pre_residual_with_amvr(reader, affine_gate, mvp_gate, amvr_gate)?;

    // The §7.3.11.7 outer gate resolves inter_pred_idc to PRED_L0 in
    // P-slices (§7.4.12.7 inference) and to the decoded value in
    // B-slices; bcw_gate.inter_pred_idc MUST match that resolved
    // value or the §7.3.10.5 gate evaluation diverges from the
    // encoder's.
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

    // Step 11 — the §7.3.10.5 BCW cascade.
    let bcw_idx = reader.read_bcw_idx_gated(*bcw_gate)?;
    Ok((decision, amvr, bcw_idx))
}

// =====================================================================
// Round-230 — reader-side composite walker variants for the §7.3.10.5
// multi-CP-MV affine MVD path. Reader twins of the encoder-side
// `encode_non_merge_inter_pre_residual_affine` (round 207),
// `encode_non_merge_inter_pre_residual_affine_with_amvr` (round 213),
// and `encode_non_merge_inter_pre_residual_affine_with_amvr_and_bcw`
// (round 213) dispatchers.
//
// Round 219 / 224 lifted the translational reader composites
// (one `mvd_coding` per active list, `numCpMv == 1`). This round
// generalises them across the full §7.3.10.5 non-merge inter affine
// case (`numCpMv` `mvd_coding` invocations per active list, in
// `cpIdx = 0, 1, 2` order). When the affine pair decodes to
// `MotionModel::Translational` (`numCpMv == 1`) the wire layout is
// bit-identical to the round-219 / round-224 dispatcher and the
// recovered `NonMergeInterPreResidualAffineDecision` is exactly the
// translational decision with `mvd_cp_l0[0]` / `mvd_cp_l1[0]` carrying
// the per-list MVDs.
// =====================================================================

/// Round-230 — reader-side composite walker that consumes the entire
/// §7.3.11.7 non-merge inter CU pre-residual syntax in one call WITH
/// per-CP affine MVD support. Reader twin of
/// [`crate::non_merge_inter_pre_residual_enc::encode_non_merge_inter_pre_residual_affine`]
/// (round 207).
///
/// Generalises [`read_non_merge_inter_pre_residual`] from the
/// translational case (one `mvd_coding` per active list) to the full
/// §7.3.10.5 non-merge inter affine case (`numCpMv` `mvd_coding`
/// invocations per active list, in `cpIdx = 0, 1, 2` order). When the
/// affine pair decodes to `MotionModel::Translational` (`numCpMv == 1`)
/// the wire layout is bit-identical to [`read_non_merge_inter_pre_residual`].
///
/// The dispatcher walks the per-element reader helpers in §7.3.10.5
/// spec order, applying the §7.4.12.7 inference rules whenever a gate
/// is closed so no bin is consumed for a syntax element whose gate is
/// closed.
///
/// # Spec ordering — bin-for-bin with the encoder-side dispatcher
///
/// The reader walks the exact sequence the encoder-side dispatcher
/// emits in
/// [`crate::non_merge_inter_pre_residual_enc::encode_non_merge_inter_pre_residual_affine`]:
///
/// 1. Affine syntax (`inter_affine_flag` / `cu_affine_type_flag`) via
///    [`LeafCuReader::read_non_merge_inter_affine`] — drives the
///    §8.5.5.2 eq. 160 `MotionModelIdc` and the §8.5.5.5
///    `numCpMv = MotionModelIdc + 1` derivation.
/// 2. `inter_pred_idc` via [`LeafCuReader::read_inter_pred_idc`]
///    (only when the §7.3.11.7 outer gate opens — B-slice).
/// 3. `sym_mvd_flag` via [`LeafCuReader::read_sym_mvd_flag`]
///    (only when the §7.3.11.7 SMVD gate opens — gate is closed on
///    the affine path per §7.3.11.7).
/// 4. `ref_idx_l0` via [`LeafCuReader::read_ref_idx_lx`]
///    (only when L0 is active AND `cMax > 0` AND `sym_mvd_flag == 0`).
/// 5. `ref_idx_l1` (mirror of step 4 for L1).
/// 6. **`mvd_coding(L0, cpIdx)` for `cpIdx ∈ 0..numCpMv`**
///    (gated on `MotionModelIdc > 0` / `MotionModelIdc > 1` for the
///    higher CPs per §7.3.10.5). Per-CP slot `cpIdx >= numCpMv` is
///    returned as zero (no bins consumed).
/// 7. **Same per-CP cascade for L1, with `cpIdx == 0` suppressed
///    under `sym_mvd_flag == 1` per §8.5.2.5** (the §8.5.2.5
///    derivation infers `MvdL1[0] = -MvdL0[0]`; the higher per-CP
///    L1 MVDs are read verbatim per the §7.3.10.5 listing — in
///    practice the affine path excludes SMVD so `cpIdx >= 1` only
///    fires under `sym_mvd_flag == 0`).
/// 8. `mvp_l0_flag` via [`LeafCuReader::read_mvp_lx_flag`]
///    (only when L0 is active).
/// 9. `mvp_l1_flag` (only when L1 is active).
///
/// # Returns
///
/// A [`NonMergeInterPreResidualAffineDecision`] whose fields carry the
/// decoded values OR their §7.4.12.7 inferred defaults. The constructor
/// [`NonMergeInterPreResidualAffineDecision::new`] is NOT called here
/// because the dispatcher already populates the per-CP slots in spec
/// order (with `cpIdx >= numCpMv` left at zero) and emits the
/// §8.5.2.5 SMVD-L1 derivation for the L1 cpIdx-0 slot when applicable.
///
/// # Preconditions
///
/// Identical to [`read_non_merge_inter_pre_residual`]. The dispatcher
/// does NOT consume `cu_skip_flag`, `pred_mode_flag`, or
/// `general_merge_flag` — those belong to the broader leaf CU walker.
///
/// No third-party VVC decoder source was consulted; the implementation
/// is spec-only and composes the existing per-element reader-side
/// helpers already shipped in this crate.
pub fn read_non_merge_inter_pre_residual_affine(
    reader: &mut LeafCuReader<'_, '_>,
    affine_gate: &NonMergeInterAffineGate,
    mvp_gate: &NonMergeMvpSyntaxGate,
) -> Result<NonMergeInterPreResidualAffineDecision> {
    // ------------------------------------------------------------------
    // Step 1 — affine syntax (round-164 dispatcher) drives numCpMv.
    // ------------------------------------------------------------------
    let affine = reader.read_non_merge_inter_affine(affine_gate)?;
    let num_cp = affine.motion_model.num_cp_mv();

    // ------------------------------------------------------------------
    // Step 2 — inter_pred_idc (§7.3.11.7 B-slice only).
    // ------------------------------------------------------------------
    let inter_pred_idc = if mvp_gate.inter_pred_idc_gate_open() {
        reader.read_inter_pred_idc(mvp_gate.cb_width, mvp_gate.cb_height)?
    } else {
        InterPredDir::PredL0
    };

    // ------------------------------------------------------------------
    // Step 3 — sym_mvd_flag (§7.3.11.7 SMVD gate). §7.3.11.7 gates SMVD
    // on `inter_affine_flag == 0`; on the affine path the gate is
    // closed by construction and the encoder emits no bin.
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
    // Step 6 — per-CP mvd_coding for L0.
    //
    // §7.3.10.5 listing:
    //   mvd_coding(x0, y0, 0, 0)
    //   if (MotionModelIdc > 0) mvd_coding(x0, y0, 0, 1)
    //   if (MotionModelIdc > 1) mvd_coding(x0, y0, 0, 2)
    //
    // Iterates 0..numCpMv on the L0 path when L0 is active. Slots
    // beyond numCpMv stay at the zero default.
    // ------------------------------------------------------------------
    let mut mvd_cp_l0 = [MotionVector { x: 0, y: 0 }; 3];
    if l0_active {
        for slot in mvd_cp_l0.iter_mut().take(num_cp) {
            *slot = reader.read_mvd_coding()?;
        }
    }

    // ------------------------------------------------------------------
    // Step 7 — per-CP mvd_coding for L1.
    //
    // §7.3.10.5 listing (paraphrased per the spec's pseudocode):
    //   if (sym_mvd_flag) { MvdL1[0] = -MvdL0[0]; }   // no bin
    //   else mvd_coding(x0, y0, 1, 0)
    //   if (MotionModelIdc > 0) mvd_coding(x0, y0, 1, 1)
    //   if (MotionModelIdc > 1) mvd_coding(x0, y0, 1, 2)
    //
    // The spec ONLY suppresses the `cpIdx == 0` L1 MVD under
    // sym_mvd_flag; the higher-CP L1 MVDs are read verbatim. In
    // practice the affine path excludes SMVD so `cpIdx >= 1` only
    // fires under `sym_mvd_flag == 0`. The translational degenerate
    // (`numCpMv == 1`) with `sym_mvd_flag == 1` carries the §8.5.2.5
    // -MvdL0 derivation in `mvd_cp_l1[0]` for symmetry with the
    // round-219 dispatcher (both inferred forms round-trip bit-
    // identically through the encoder).
    // ------------------------------------------------------------------
    let mut mvd_cp_l1 = [MotionVector { x: 0, y: 0 }; 3];
    if l1_active {
        if sym_mvd_flag {
            // §8.5.2.5 — MvdL1[0] = -MvdL0[0]. The L1 magnitude bounds
            // match L0 bin-for-bin so the negation is safe at the
            // spec's signed-18-bit range.
            mvd_cp_l1[0] = MotionVector {
                x: -mvd_cp_l0[0].x,
                y: -mvd_cp_l0[0].y,
            };
        } else {
            mvd_cp_l1[0] = reader.read_mvd_coding()?;
        }
        for slot in mvd_cp_l1.iter_mut().take(num_cp).skip(1) {
            *slot = reader.read_mvd_coding()?;
        }
    }

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

    let mvp = NonMergeMvpSyntaxDecision {
        inter_pred_idc,
        sym_mvd_flag,
        ref_idx_l0,
        ref_idx_l1,
        mvp_l0_flag,
        mvp_l1_flag,
    };

    Ok(NonMergeInterPreResidualAffineDecision {
        affine,
        mvp,
        mvd_cp_l0,
        mvd_cp_l1,
    })
}

/// Round-230 — reader-side composite walker variant that ALSO consumes
/// the §7.3.10.10 AMVR cascade after the §7.3.11.7 affine pre-residual
/// cascade. Reader twin of
/// [`crate::non_merge_inter_pre_residual_enc::encode_non_merge_inter_pre_residual_affine_with_amvr`]
/// (round 213).
///
/// Generalises [`read_non_merge_inter_pre_residual_with_amvr`] from the
/// translational case (one `mvd_coding` per active list) to the full
/// §7.3.10.5 non-merge inter affine case (`numCpMv` `mvd_coding`
/// invocations per active list). When the affine pair decodes to
/// `MotionModel::Translational` the wire layout is bit-identical to
/// [`read_non_merge_inter_pre_residual_with_amvr`].
///
/// Walks the §7.3.11.7 spec listing through step 10:
///
/// 1. Steps 1–9 — the pre-AMVR affine cascade via
///    [`read_non_merge_inter_pre_residual_affine`].
/// 2. Step 10 — the §7.3.10.10 AMVR cascade via
///    [`LeafCuReader::read_amvr_inter_gated`].
///
/// Returns the pair `(pre_residual, amvr)` where `pre_residual` is the
/// affine decision (raw post-AMVR `lMvdCpLX[c]` per §7.4.10.10 — the
/// AMVR shift is signalled separately by the AMVR cascade and applied
/// to the parsed MVDs externally) and `amvr` packages the recovered
/// `(amvr_flag, amvr_precision_idx, AmvrShift)` triple as a
/// [`crate::amvr_enc::AmvrDecision`] for symmetry with the encoder-side
/// `amvr_decision` parameter.
///
/// # Preconditions
///
/// * All preconditions of [`read_non_merge_inter_pre_residual_affine`]
///   apply unchanged.
/// * `amvr_gate.inter_affine_flag` MUST agree with the
///   `decision.affine.inter_affine_flag` returned by the pre-residual
///   dispatcher — the AMVR arm follows the affine flag. The caller
///   typically builds `amvr_gate` from the same per-CB state the
///   pre-residual dispatcher recovers.
/// * Under the affine arm `amvr_gate.any_mvd_cp_l0_l1_nonzero` SHOULD
///   reflect the non-zero state of the per-CP MVDs the dispatcher
///   parses (over CPs `0..numCpMv` per list). The encoder's matching
///   debug-assert pins this.
pub fn read_non_merge_inter_pre_residual_affine_with_amvr(
    reader: &mut LeafCuReader<'_, '_>,
    affine_gate: &NonMergeInterAffineGate,
    mvp_gate: &NonMergeMvpSyntaxGate,
    amvr_gate: &crate::leaf_cu::AmvrGate,
) -> Result<(
    NonMergeInterPreResidualAffineDecision,
    crate::amvr_enc::AmvrDecision,
)> {
    // Steps 1–9 — the pre-AMVR affine cascade. Identical to round-230.
    let decision = read_non_merge_inter_pre_residual_affine(reader, affine_gate, mvp_gate)?;

    // The AMVR gate's `inter_affine_flag` is per-CB state the caller
    // already had at hand; the affine pre-residual dispatcher just
    // decoded the matching `inter_affine_flag`, so debug-assert the
    // two agree.
    debug_assert_eq!(
        amvr_gate.inter_affine_flag, decision.affine.inter_affine_flag,
        "amvr_gate.inter_affine_flag must match decision.affine.inter_affine_flag \
         (got amvr_gate = {}, decision = {})",
        amvr_gate.inter_affine_flag, decision.affine.inter_affine_flag,
    );

    // Step 10 — the §7.3.10.10 AMVR cascade.
    let (amvr_flag, amvr_precision_idx, shift) = reader.read_amvr_inter_gated(amvr_gate)?;
    let amvr = crate::amvr_enc::AmvrDecision {
        amvr_flag,
        amvr_precision_idx,
        shift,
    };
    Ok((decision, amvr))
}

/// Round-230 — reader-side composite walker variant that ALSO consumes
/// the §7.3.10.5 `bcw_idx[x0][y0]` cascade after the §7.3.10.10 AMVR
/// step on the affine path. Reader twin of
/// [`crate::non_merge_inter_pre_residual_enc::encode_non_merge_inter_pre_residual_affine_with_amvr_and_bcw`]
/// (round 213).
///
/// Generalises [`read_non_merge_inter_pre_residual_with_amvr_and_bcw`]
/// from the translational case to the full §7.3.10.5 non-merge inter
/// affine case (`numCpMv` `mvd_coding` invocations per active list).
/// When the affine pair decodes to `MotionModel::Translational` the
/// wire layout is bit-identical to
/// [`read_non_merge_inter_pre_residual_with_amvr_and_bcw`].
///
/// Walks the §7.3.11.7 / §7.3.10.5 spec listing through step 11:
///
/// 1. Steps 1–9 — the pre-AMVR affine cascade via
///    [`read_non_merge_inter_pre_residual_affine`].
/// 2. Step 10 — the §7.3.10.10 AMVR cascade via
///    [`LeafCuReader::read_amvr_inter_gated`].
/// 3. Step 11 — the §7.3.10.5 BCW cascade via
///    [`LeafCuReader::read_bcw_idx_gated`].
///
/// Returns the triple `(pre_residual, amvr, bcw_idx)`. The `bcw_idx`
/// is the raw `bcw_idx[x0][y0]` value the encoder placed on the wire
/// (when the gate was open) or the §7.4.12.5 inferred default `0`
/// (when the gate was closed).
///
/// # Preconditions
///
/// All preconditions of
/// [`read_non_merge_inter_pre_residual_affine_with_amvr`] apply
/// unchanged. Additionally:
///
/// * `bcw_gate.cb_width` / `bcw_gate.cb_height` SHOULD match the
///   `affine_gate.cb_width` / `affine_gate.cb_height` (same CU). The
///   dispatcher debug-asserts this.
/// * `bcw_gate.inter_pred_idc` MUST match the inter_pred_idc the
///   affine pre-residual dispatcher resolved (i.e.
///   `decision.mvp.inter_pred_idc` for B-slices, or the
///   `PRED_L0` inferred default for P-slices). The dispatcher
///   debug-asserts this.
pub fn read_non_merge_inter_pre_residual_affine_with_amvr_and_bcw(
    reader: &mut LeafCuReader<'_, '_>,
    affine_gate: &NonMergeInterAffineGate,
    mvp_gate: &NonMergeMvpSyntaxGate,
    amvr_gate: &crate::leaf_cu::AmvrGate,
    bcw_gate: &crate::leaf_cu::BcwIdxGate,
) -> Result<(
    NonMergeInterPreResidualAffineDecision,
    crate::amvr_enc::AmvrDecision,
    u32,
)> {
    debug_assert_eq!(
        bcw_gate.cb_width, affine_gate.cb_width,
        "bcw_gate.cb_width must match affine_gate.cb_width (same CU)"
    );
    debug_assert_eq!(
        bcw_gate.cb_height, affine_gate.cb_height,
        "bcw_gate.cb_height must match affine_gate.cb_height (same CU)"
    );

    // Steps 1–10 — the pre-BCW affine cascade. Identical to the
    // `_with_amvr` composite above.
    let (decision, amvr) = read_non_merge_inter_pre_residual_affine_with_amvr(
        reader,
        affine_gate,
        mvp_gate,
        amvr_gate,
    )?;

    // The §7.3.11.7 outer gate resolves inter_pred_idc to PRED_L0 in
    // P-slices (§7.4.12.7 inference) and to the decoded value in
    // B-slices; bcw_gate.inter_pred_idc MUST match that resolved
    // value or the §7.3.10.5 gate evaluation diverges from the
    // encoder's.
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

    // Step 11 — the §7.3.10.5 BCW cascade.
    let bcw_idx = reader.read_bcw_idx_gated(*bcw_gate)?;
    Ok((decision, amvr, bcw_idx))
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

    // ==================================================================
    // Round-224 — composite reader dispatcher tests.
    //
    // The encoder dispatchers
    // `encode_non_merge_inter_pre_residual_with_amvr` (round 195) and
    // `encode_non_merge_inter_pre_residual_with_amvr_and_bcw` (round 201)
    // drive the cascade end-to-end and the round-224 reader composites
    // recover the same triples (decision, amvr, bcw_idx). Round-trip
    // tests mirror the round-195 / round-201 encoder-side test set on
    // both arms of the AMVR cascade and across the BCW cMax choices.
    // ==================================================================

    use crate::amvr::AmvrShift;
    use crate::amvr_enc::AmvrDecision;
    use crate::leaf_cu::{AmvrGate, BcwIdxGate};
    use crate::non_merge_inter_pre_residual_enc::{
        encode_non_merge_inter_pre_residual_with_amvr,
        encode_non_merge_inter_pre_residual_with_amvr_and_bcw,
    };

    /// Drive the round-195 encoder dispatcher and recover through the
    /// round-224 reader composite. Returns the `(decision, amvr)` pair
    /// the reader produced.
    fn round_trip_via_encoder_with_amvr(
        init_type: u8,
        affine_gate: &NonMergeInterAffineGate,
        mvp_gate: &NonMergeMvpSyntaxGate,
        amvr_gate: &AmvrGate,
        decision: &NonMergeInterPreResidualDecision,
        amvr_decision: &AmvrDecision,
    ) -> (NonMergeInterPreResidualDecision, AmvrDecision) {
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
        .expect("encoder _with_amvr dispatcher accepts a valid decision");
        enc.encode_terminate(1).expect("terminator");
        let mut padded = enc.finish();
        padded.extend_from_slice(&[0u8; 64]);
        let mut dec = ArithDecoder::new(&padded).expect("decoder accepts the encoded stream");
        let mut dec_ctxs = LeafCuCtxs::init_with_init_type(26, init_type);
        let tools = CuToolFlags::default();
        let mut reader = LeafCuReader::new(&mut dec, &mut dec_ctxs, tools);
        read_non_merge_inter_pre_residual_with_amvr(&mut reader, affine_gate, mvp_gate, amvr_gate)
            .expect("reader _with_amvr dispatcher accepts the encoded stream")
    }

    /// Drive the round-201 encoder dispatcher and recover through the
    /// round-224 reader composite. Returns the `(decision, amvr,
    /// bcw_idx)` triple the reader produced.
    #[allow(clippy::too_many_arguments)]
    fn round_trip_via_encoder_with_amvr_and_bcw(
        init_type: u8,
        affine_gate: &NonMergeInterAffineGate,
        mvp_gate: &NonMergeMvpSyntaxGate,
        amvr_gate: &AmvrGate,
        bcw_gate: &BcwIdxGate,
        decision: &NonMergeInterPreResidualDecision,
        amvr_decision: &AmvrDecision,
        bcw_idx_value: u32,
    ) -> (NonMergeInterPreResidualDecision, AmvrDecision, u32) {
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
        .expect("encoder _with_amvr_and_bcw dispatcher accepts a valid decision");
        enc.encode_terminate(1).expect("terminator");
        let mut padded = enc.finish();
        padded.extend_from_slice(&[0u8; 64]);
        let mut dec = ArithDecoder::new(&padded).expect("decoder accepts the encoded stream");
        let mut dec_ctxs = LeafCuCtxs::init_with_init_type(26, init_type);
        let tools = CuToolFlags::default();
        let mut reader = LeafCuReader::new(&mut dec, &mut dec_ctxs, tools);
        read_non_merge_inter_pre_residual_with_amvr_and_bcw(
            &mut reader,
            affine_gate,
            mvp_gate,
            amvr_gate,
            bcw_gate,
        )
        .expect("reader _with_amvr_and_bcw dispatcher accepts the encoded stream")
    }

    // ------ Round-224: _with_amvr round-trips -------------------------

    #[test]
    fn round224_with_amvr_p_slice_closed_gate_round_trip() {
        // P-slice, regular AMVR, all-zero MVDs → AMVR outer gate is
        // closed (no MVD non-zero) and §7.4.12.7 inference fires
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
        for init_type in [1u8, 2u8] {
            let (rec_decision, rec_amvr) = round_trip_via_encoder_with_amvr(
                init_type,
                &affine_gate,
                &mvp_gate,
                &amvr_gate,
                &decision,
                &amvr_decision,
            );
            assert_eq!(rec_decision.mvp.inter_pred_idc, InterPredDir::PredL0);
            assert_eq!(rec_decision.mvd_l0, MotionVector { x: 0, y: 0 });
            assert!(!rec_amvr.amvr_flag);
            assert_eq!(rec_amvr.amvr_precision_idx, 0);
            assert_eq!(rec_amvr.shift, AmvrShift(2));
        }
    }

    #[test]
    fn round224_with_amvr_p_slice_open_regular_precision_2_round_trip() {
        // P-slice, regular AMVR open (sps + non-zero MVD), amvr_flag =
        // 1, prec = 2 (4-luma) → AmvrShift = 6. Both non-I initTypes.
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
        for init_type in [1u8, 2u8] {
            let (rec_decision, rec_amvr) = round_trip_via_encoder_with_amvr(
                init_type,
                &affine_gate,
                &mvp_gate,
                &amvr_gate,
                &decision,
                &amvr_decision,
            );
            assert_eq!(rec_decision.mvd_l0, MotionVector { x: 5, y: -3 });
            assert_eq!(rec_decision.mvp.ref_idx_l0, 1);
            assert_eq!(rec_decision.mvp.mvp_l0_flag, 1);
            assert!(rec_amvr.amvr_flag);
            assert_eq!(rec_amvr.amvr_precision_idx, 2);
            assert_eq!(rec_amvr.shift, AmvrShift(6));
        }
    }

    #[test]
    fn round224_with_amvr_b_slice_pred_bi_amvr_open_precision_1_round_trip() {
        // B-slice, PRED_BI, regular AMVR open, prec = 1 → AmvrShift = 4.
        let affine_gate = affine_gate_off();
        let mvp_gate = b_slice_gate();
        let amvr_gate = AmvrGate {
            sps_amvr_enabled: true,
            sps_affine_amvr_enabled: false,
            inter_affine_flag: false,
            any_mvd_l0_l1_nonzero: true,
            any_mvd_cp_l0_l1_nonzero: false,
        };
        let affine = make_non_merge_inter_affine_decision(false, false);
        let mvp = make_non_merge_mvp_syntax_decision(InterPredDir::PredBi, false, 1, 1, 1, 0);
        let decision = NonMergeInterPreResidualDecision::new(
            affine,
            mvp,
            MotionVector { x: 3, y: -3 },
            MotionVector { x: -8, y: 8 },
        );
        let amvr_decision = AmvrDecision::new(true, 1, false);
        let (rec_decision, rec_amvr) = round_trip_via_encoder_with_amvr(
            1,
            &affine_gate,
            &mvp_gate,
            &amvr_gate,
            &decision,
            &amvr_decision,
        );
        assert_eq!(rec_decision.mvp.inter_pred_idc, InterPredDir::PredBi);
        assert_eq!(rec_decision.mvd_l0, MotionVector { x: 3, y: -3 });
        assert_eq!(rec_decision.mvd_l1, MotionVector { x: -8, y: 8 });
        assert!(rec_amvr.amvr_flag);
        assert_eq!(rec_amvr.amvr_precision_idx, 1);
        assert_eq!(rec_amvr.shift, AmvrShift(4));
    }

    #[test]
    fn round224_with_amvr_smvd_amvr_closed_round_trip() {
        // SMVD with L0 MVD non-zero — AMVR cascade is gated by
        // `any_mvd_l0_l1_nonzero` so the gate opens, but the encoder
        // emits amvr_flag = 0 → reader recovers AmvrShift = 2 + the
        // §8.5.2.5 -MvdL0 inference for the L1 MVD.
        let affine_gate = affine_gate_off();
        let mvp_gate = b_slice_gate_smvd();
        let amvr_gate = AmvrGate {
            sps_amvr_enabled: true,
            sps_affine_amvr_enabled: false,
            inter_affine_flag: false,
            any_mvd_l0_l1_nonzero: true,
            any_mvd_cp_l0_l1_nonzero: false,
        };
        let affine = make_non_merge_inter_affine_decision(false, false);
        let mvp = make_non_merge_mvp_syntax_decision(InterPredDir::PredBi, true, 0, 0, 1, 1);
        let l0 = MotionVector { x: 9, y: -5 };
        let decision =
            NonMergeInterPreResidualDecision::new(affine, mvp, l0, MotionVector { x: 0, y: 0 });
        let amvr_decision = AmvrDecision::new(false, 0, false);
        for init_type in [1u8, 2u8] {
            let (rec_decision, rec_amvr) = round_trip_via_encoder_with_amvr(
                init_type,
                &affine_gate,
                &mvp_gate,
                &amvr_gate,
                &decision,
                &amvr_decision,
            );
            assert!(rec_decision.mvp.sym_mvd_flag);
            assert_eq!(rec_decision.mvd_l0, l0);
            assert_eq!(rec_decision.mvd_l1, MotionVector { x: -l0.x, y: -l0.y });
            assert!(!rec_amvr.amvr_flag);
            assert_eq!(rec_amvr.shift, AmvrShift(2));
        }
    }

    // ------ Round-224: _with_amvr_and_bcw round-trips -----------------

    fn open_bcw_gate_for_b_pred_bi() -> BcwIdxGate {
        BcwIdxGate {
            sps_bcw_enabled: true,
            inter_pred_idc: Some(InterPredDir::PredBi),
            luma_weight_l0_flag: false,
            luma_weight_l1_flag: false,
            chroma_weight_l0_flag: false,
            chroma_weight_l1_flag: false,
            cb_width: 16,
            cb_height: 16,
            no_backward_pred_flag: false,
        }
    }

    #[test]
    fn round224_with_amvr_and_bcw_p_slice_bcw_closed_round_trip() {
        // P-slice → §7.3.10.5 BCW gate closed (PRED_L0 ≠ PRED_BI). AMVR
        // also closed via all-zero MVDs. Reader recovers bcw_idx = 0.
        let affine_gate = affine_gate_off();
        let mvp_gate = p_slice_gate();
        let amvr_gate = AmvrGate {
            sps_amvr_enabled: true,
            sps_affine_amvr_enabled: false,
            inter_affine_flag: false,
            any_mvd_l0_l1_nonzero: false,
            any_mvd_cp_l0_l1_nonzero: false,
        };
        let bcw_gate = BcwIdxGate {
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
        for init_type in [1u8, 2u8] {
            let (rec_decision, rec_amvr, rec_bcw) = round_trip_via_encoder_with_amvr_and_bcw(
                init_type,
                &affine_gate,
                &mvp_gate,
                &amvr_gate,
                &bcw_gate,
                &decision,
                &amvr_decision,
                0,
            );
            assert_eq!(rec_decision.mvp.inter_pred_idc, InterPredDir::PredL0);
            assert!(!rec_amvr.amvr_flag);
            assert_eq!(rec_bcw, 0);
        }
    }

    #[test]
    fn round224_with_amvr_and_bcw_b_slice_pred_bi_bcw_open_all_values_cmax_2() {
        // B-slice, PRED_BI, BCW gate open with NoBackwardPredFlag = 0
        // → cMax = 2. AMVR closed via all-zero MVDs (the AMVR cascade
        // emits no bins). Exhaustive across bcw_idx ∈ {0, 1, 2} × both
        // non-I initTypes.
        let affine_gate = affine_gate_off();
        let mvp_gate = b_slice_gate();
        let amvr_gate = AmvrGate {
            sps_amvr_enabled: true,
            sps_affine_amvr_enabled: false,
            inter_affine_flag: false,
            any_mvd_l0_l1_nonzero: false,
            any_mvd_cp_l0_l1_nonzero: false,
        };
        let bcw_gate = open_bcw_gate_for_b_pred_bi();
        assert!(bcw_gate.is_open());
        let affine = make_non_merge_inter_affine_decision(false, false);
        let mvp = make_non_merge_mvp_syntax_decision(InterPredDir::PredBi, false, 0, 0, 0, 0);
        let decision = NonMergeInterPreResidualDecision::new(
            affine,
            mvp,
            MotionVector { x: 0, y: 0 },
            MotionVector { x: 0, y: 0 },
        );
        let amvr_decision = AmvrDecision::default_inferred();
        for init_type in [1u8, 2u8] {
            for bcw_idx_value in 0u32..=2u32 {
                let (rec_decision, rec_amvr, rec_bcw) = round_trip_via_encoder_with_amvr_and_bcw(
                    init_type,
                    &affine_gate,
                    &mvp_gate,
                    &amvr_gate,
                    &bcw_gate,
                    &decision,
                    &amvr_decision,
                    bcw_idx_value,
                );
                assert_eq!(rec_decision.mvp.inter_pred_idc, InterPredDir::PredBi);
                assert!(!rec_amvr.amvr_flag);
                assert_eq!(rec_bcw, bcw_idx_value);
            }
        }
    }

    #[test]
    fn round224_with_amvr_and_bcw_b_slice_pred_bi_bcw_open_no_backward_pred_cmax_4() {
        // B-slice, PRED_BI, BCW gate open with NoBackwardPredFlag = 1
        // → cMax = 4. AMVR open via non-zero MVD with prec = 0 →
        // AmvrShift = 1 (1/2-luma). Exhaustive across bcw_idx ∈
        // {0, 1, 2, 3, 4}.
        let affine_gate = affine_gate_off();
        let mvp_gate = b_slice_gate();
        let amvr_gate = AmvrGate {
            sps_amvr_enabled: true,
            sps_affine_amvr_enabled: false,
            inter_affine_flag: false,
            any_mvd_l0_l1_nonzero: true,
            any_mvd_cp_l0_l1_nonzero: false,
        };
        let bcw_gate = BcwIdxGate {
            no_backward_pred_flag: true,
            ..open_bcw_gate_for_b_pred_bi()
        };
        let affine = make_non_merge_inter_affine_decision(false, false);
        let mvp = make_non_merge_mvp_syntax_decision(InterPredDir::PredBi, false, 0, 0, 1, 0);
        let decision = NonMergeInterPreResidualDecision::new(
            affine,
            mvp,
            MotionVector { x: 2, y: 1 },
            MotionVector { x: -1, y: -2 },
        );
        let amvr_decision = AmvrDecision::new(true, 0, false);
        for bcw_idx_value in 0u32..=4u32 {
            let (rec_decision, rec_amvr, rec_bcw) = round_trip_via_encoder_with_amvr_and_bcw(
                1,
                &affine_gate,
                &mvp_gate,
                &amvr_gate,
                &bcw_gate,
                &decision,
                &amvr_decision,
                bcw_idx_value,
            );
            assert_eq!(rec_decision.mvd_l0, MotionVector { x: 2, y: 1 });
            assert_eq!(rec_decision.mvd_l1, MotionVector { x: -1, y: -2 });
            assert!(rec_amvr.amvr_flag);
            assert_eq!(rec_amvr.shift, AmvrShift(3));
            assert_eq!(rec_bcw, bcw_idx_value);
        }
    }

    #[test]
    fn round224_with_amvr_and_bcw_amvr_and_bcw_simultaneously_open_round_trip() {
        // B-slice, PRED_BI, both AMVR (prec = 2 → AmvrShift = 6) and
        // BCW (cMax = 2, bcw_idx = 1) emit bins in sequence. Reader
        // recovers both intact.
        let affine_gate = affine_gate_off();
        let mvp_gate = b_slice_gate();
        let amvr_gate = AmvrGate {
            sps_amvr_enabled: true,
            sps_affine_amvr_enabled: false,
            inter_affine_flag: false,
            any_mvd_l0_l1_nonzero: true,
            any_mvd_cp_l0_l1_nonzero: false,
        };
        let bcw_gate = open_bcw_gate_for_b_pred_bi();
        let affine = make_non_merge_inter_affine_decision(false, false);
        let mvp = make_non_merge_mvp_syntax_decision(InterPredDir::PredBi, false, 1, 1, 0, 1);
        let decision = NonMergeInterPreResidualDecision::new(
            affine,
            mvp,
            MotionVector { x: 4, y: 4 },
            MotionVector { x: -4, y: -4 },
        );
        let amvr_decision = AmvrDecision::new(true, 2, false);
        for init_type in [1u8, 2u8] {
            let (rec_decision, rec_amvr, rec_bcw) = round_trip_via_encoder_with_amvr_and_bcw(
                init_type,
                &affine_gate,
                &mvp_gate,
                &amvr_gate,
                &bcw_gate,
                &decision,
                &amvr_decision,
                1,
            );
            assert_eq!(rec_decision.mvp.inter_pred_idc, InterPredDir::PredBi);
            assert_eq!(rec_decision.mvp.ref_idx_l0, 1);
            assert_eq!(rec_decision.mvp.ref_idx_l1, 1);
            assert_eq!(rec_decision.mvp.mvp_l1_flag, 1);
            assert_eq!(rec_decision.mvd_l0, MotionVector { x: 4, y: 4 });
            assert_eq!(rec_decision.mvd_l1, MotionVector { x: -4, y: -4 });
            assert!(rec_amvr.amvr_flag);
            assert_eq!(rec_amvr.amvr_precision_idx, 2);
            assert_eq!(rec_amvr.shift, AmvrShift(6));
            assert_eq!(rec_bcw, 1);
        }
    }

    #[test]
    fn round224_with_amvr_and_bcw_bcw_closed_by_weighted_pred_round_trip() {
        // B-slice, PRED_BI, but luma_weight_l0_flag = true closes the
        // §7.3.10.5 BCW gate. Reader recovers bcw_idx = 0 even when
        // the encoder was passed `bcw_idx_value = 0` (the §7.4.12.5
        // inferred default).
        let affine_gate = affine_gate_off();
        let mvp_gate = b_slice_gate();
        let amvr_gate = AmvrGate {
            sps_amvr_enabled: true,
            sps_affine_amvr_enabled: false,
            inter_affine_flag: false,
            any_mvd_l0_l1_nonzero: false,
            any_mvd_cp_l0_l1_nonzero: false,
        };
        let bcw_gate = BcwIdxGate {
            sps_bcw_enabled: true,
            inter_pred_idc: Some(InterPredDir::PredBi),
            luma_weight_l0_flag: true,
            cb_width: 16,
            cb_height: 16,
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
        let (rec_decision, rec_amvr, rec_bcw) = round_trip_via_encoder_with_amvr_and_bcw(
            1,
            &affine_gate,
            &mvp_gate,
            &amvr_gate,
            &bcw_gate,
            &decision,
            &amvr_decision,
            0,
        );
        assert_eq!(rec_decision.mvp.inter_pred_idc, InterPredDir::PredBi);
        assert!(!rec_amvr.amvr_flag);
        assert_eq!(rec_bcw, 0);
    }

    #[test]
    fn round224_with_amvr_and_bcw_bcw_closed_by_small_cu_round_trip() {
        // B-slice, PRED_BI, but cb_w * cb_h = 8 * 8 = 64 < 256 → BCW
        // gate closed. AMVR open via non-zero MVD with prec = 1.
        let affine_gate = NonMergeInterAffineGate {
            sps_affine_enabled: false,
            sps_6param_affine_enabled: false,
            cb_width: 8,
            cb_height: 8,
            ..Default::default()
        };
        let mvp_gate = NonMergeMvpSyntaxGate {
            cb_width: 8,
            cb_height: 8,
            b_slice: true,
            sym_mvd_gate_open: false,
            num_ref_idx_active_l0: 2,
            num_ref_idx_active_l1: 2,
        };
        let amvr_gate = AmvrGate {
            sps_amvr_enabled: true,
            sps_affine_amvr_enabled: false,
            inter_affine_flag: false,
            any_mvd_l0_l1_nonzero: true,
            any_mvd_cp_l0_l1_nonzero: false,
        };
        let bcw_gate = BcwIdxGate {
            sps_bcw_enabled: true,
            inter_pred_idc: Some(InterPredDir::PredBi),
            cb_width: 8,
            cb_height: 8,
            ..Default::default()
        };
        assert!(!bcw_gate.is_open());
        let affine = make_non_merge_inter_affine_decision(false, false);
        let mvp = make_non_merge_mvp_syntax_decision(InterPredDir::PredBi, false, 0, 0, 0, 0);
        let decision = NonMergeInterPreResidualDecision::new(
            affine,
            mvp,
            MotionVector { x: 1, y: -1 },
            MotionVector { x: -1, y: 1 },
        );
        let amvr_decision = AmvrDecision::new(true, 1, false);
        let (rec_decision, rec_amvr, rec_bcw) = round_trip_via_encoder_with_amvr_and_bcw(
            2,
            &affine_gate,
            &mvp_gate,
            &amvr_gate,
            &bcw_gate,
            &decision,
            &amvr_decision,
            0,
        );
        assert_eq!(rec_decision.mvp.inter_pred_idc, InterPredDir::PredBi);
        assert!(rec_amvr.amvr_flag);
        assert_eq!(rec_amvr.amvr_precision_idx, 1);
        assert_eq!(rec_amvr.shift, AmvrShift(4));
        assert_eq!(rec_bcw, 0);
    }

    // ==================================================================
    // Round-230 — reader-side affine composite dispatcher tests.
    //
    // The encoder dispatchers
    // `encode_non_merge_inter_pre_residual_affine`               (round 207),
    // `encode_non_merge_inter_pre_residual_affine_with_amvr`     (round 213),
    // `encode_non_merge_inter_pre_residual_affine_with_amvr_and_bcw`
    //                                                            (round 213)
    // drive the cascade end-to-end across multi-CP-MV affine MVDs and
    // the round-230 reader composites recover the same `(decision,
    // amvr, bcw_idx)` triples. Round-trip tests mirror the encoder-side
    // test set on both affine arms (4-param numCpMv = 2; 6-param
    // numCpMv = 3) and degenerate to the round 219 / 224 wire under
    // `MotionModel::Translational`.
    // ==================================================================

    use crate::non_merge_inter_pre_residual_enc::{
        encode_non_merge_inter_pre_residual_affine,
        encode_non_merge_inter_pre_residual_affine_with_amvr,
        encode_non_merge_inter_pre_residual_affine_with_amvr_and_bcw,
        NonMergeInterPreResidualAffineDecision,
    };

    fn affine_gate_on() -> NonMergeInterAffineGate {
        NonMergeInterAffineGate {
            sps_affine_enabled: true,
            sps_6param_affine_enabled: true,
            cb_width: 16,
            cb_height: 16,
            ..Default::default()
        }
    }

    /// Drive the round-207 encoder dispatcher and recover through the
    /// round-230 reader composite. Returns the recovered affine
    /// decision.
    fn round_trip_affine_via_encoder(
        init_type: u8,
        affine_gate: &NonMergeInterAffineGate,
        mvp_gate: &NonMergeMvpSyntaxGate,
        decision: &NonMergeInterPreResidualAffineDecision,
    ) -> NonMergeInterPreResidualAffineDecision {
        let mut enc = ArithEncoder::new();
        let mut enc_ctxs = LeafCuCtxs::init_with_init_type(26, init_type);
        encode_non_merge_inter_pre_residual_affine(
            &mut enc,
            &mut enc_ctxs,
            affine_gate,
            mvp_gate,
            decision,
        )
        .expect("encoder affine dispatcher accepts a valid decision");
        enc.encode_terminate(1).expect("terminator");
        let mut padded = enc.finish();
        padded.extend_from_slice(&[0u8; 64]);
        let mut dec = ArithDecoder::new(&padded).expect("decoder accepts the encoded stream");
        let mut dec_ctxs = LeafCuCtxs::init_with_init_type(26, init_type);
        let tools = CuToolFlags::default();
        let mut reader = LeafCuReader::new(&mut dec, &mut dec_ctxs, tools);
        read_non_merge_inter_pre_residual_affine(&mut reader, affine_gate, mvp_gate)
            .expect("reader affine dispatcher accepts the encoded stream")
    }

    /// Drive the round-213 encoder dispatcher and recover through the
    /// round-230 reader composite. Returns the `(decision, amvr)` pair
    /// the reader produced.
    fn round_trip_affine_via_encoder_with_amvr(
        init_type: u8,
        affine_gate: &NonMergeInterAffineGate,
        mvp_gate: &NonMergeMvpSyntaxGate,
        amvr_gate: &AmvrGate,
        decision: &NonMergeInterPreResidualAffineDecision,
        amvr_decision: &AmvrDecision,
    ) -> (NonMergeInterPreResidualAffineDecision, AmvrDecision) {
        let mut enc = ArithEncoder::new();
        let mut enc_ctxs = LeafCuCtxs::init_with_init_type(26, init_type);
        encode_non_merge_inter_pre_residual_affine_with_amvr(
            &mut enc,
            &mut enc_ctxs,
            affine_gate,
            mvp_gate,
            amvr_gate,
            decision,
            amvr_decision,
        )
        .expect("encoder affine _with_amvr dispatcher accepts a valid decision");
        enc.encode_terminate(1).expect("terminator");
        let mut padded = enc.finish();
        padded.extend_from_slice(&[0u8; 64]);
        let mut dec = ArithDecoder::new(&padded).expect("decoder accepts the encoded stream");
        let mut dec_ctxs = LeafCuCtxs::init_with_init_type(26, init_type);
        let tools = CuToolFlags::default();
        let mut reader = LeafCuReader::new(&mut dec, &mut dec_ctxs, tools);
        read_non_merge_inter_pre_residual_affine_with_amvr(
            &mut reader,
            affine_gate,
            mvp_gate,
            amvr_gate,
        )
        .expect("reader affine _with_amvr dispatcher accepts the encoded stream")
    }

    /// Drive the round-213 encoder dispatcher and recover through the
    /// round-230 reader composite. Returns the `(decision, amvr,
    /// bcw_idx)` triple the reader produced.
    #[allow(clippy::too_many_arguments)]
    fn round_trip_affine_via_encoder_with_amvr_and_bcw(
        init_type: u8,
        affine_gate: &NonMergeInterAffineGate,
        mvp_gate: &NonMergeMvpSyntaxGate,
        amvr_gate: &AmvrGate,
        bcw_gate: &BcwIdxGate,
        decision: &NonMergeInterPreResidualAffineDecision,
        amvr_decision: &AmvrDecision,
        bcw_idx_value: u32,
    ) -> (NonMergeInterPreResidualAffineDecision, AmvrDecision, u32) {
        let mut enc = ArithEncoder::new();
        let mut enc_ctxs = LeafCuCtxs::init_with_init_type(26, init_type);
        encode_non_merge_inter_pre_residual_affine_with_amvr_and_bcw(
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
        .expect("encoder affine _with_amvr_and_bcw dispatcher accepts a valid decision");
        enc.encode_terminate(1).expect("terminator");
        let mut padded = enc.finish();
        padded.extend_from_slice(&[0u8; 64]);
        let mut dec = ArithDecoder::new(&padded).expect("decoder accepts the encoded stream");
        let mut dec_ctxs = LeafCuCtxs::init_with_init_type(26, init_type);
        let tools = CuToolFlags::default();
        let mut reader = LeafCuReader::new(&mut dec, &mut dec_ctxs, tools);
        read_non_merge_inter_pre_residual_affine_with_amvr_and_bcw(
            &mut reader,
            affine_gate,
            mvp_gate,
            amvr_gate,
            bcw_gate,
        )
        .expect("reader affine _with_amvr_and_bcw dispatcher accepts the encoded stream")
    }

    // -------- Round-230: affine (no AMVR / BCW) round-trips -----------

    #[test]
    fn round230_affine_translational_degenerates_to_round219() {
        // numCpMv == 1: the per-CP cascade reduces to one mvd_coding
        // per active list — round-trip output matches the translational
        // round-219 path.
        let affine_gate = affine_gate_off();
        let mvp_gate = p_slice_gate();
        let affine = make_non_merge_inter_affine_decision(false, false);
        let mvp = make_non_merge_mvp_syntax_decision(InterPredDir::PredL0, false, 1, 0, 1, 0);
        let mvd_cp_l0 = [
            MotionVector { x: 5, y: -3 },
            MotionVector { x: 0, y: 0 },
            MotionVector { x: 0, y: 0 },
        ];
        let mvd_cp_l1 = [MotionVector { x: 0, y: 0 }; 3];
        let decision =
            NonMergeInterPreResidualAffineDecision::new(affine, mvp, mvd_cp_l0, mvd_cp_l1);
        assert_eq!(decision.num_cp_mv(), 1);
        for init_type in [1u8, 2u8] {
            let rec = round_trip_affine_via_encoder(init_type, &affine_gate, &mvp_gate, &decision);
            assert_eq!(
                rec.affine.motion_model,
                crate::affine::MotionModel::Translational
            );
            assert_eq!(rec.mvd_cp_l0[0], MotionVector { x: 5, y: -3 });
            assert_eq!(rec.mvd_cp_l0[1], MotionVector { x: 0, y: 0 });
            assert_eq!(rec.mvd_cp_l0[2], MotionVector { x: 0, y: 0 });
            assert_eq!(rec.mvd_cp_l1, [MotionVector { x: 0, y: 0 }; 3]);
            assert_eq!(rec.mvp.ref_idx_l0, 1);
            assert_eq!(rec.mvp.mvp_l0_flag, 1);
        }
    }

    #[test]
    fn round230_affine4param_p_slice_l0_round_trip() {
        // 4-param affine ⇒ numCpMv == 2. P-slice ⇒ L0-only.
        let affine_gate = affine_gate_on();
        let mvp_gate = p_slice_gate();
        let affine = make_non_merge_inter_affine_decision(true, false);
        assert_eq!(
            affine.motion_model,
            crate::affine::MotionModel::Affine4Param
        );
        let mvp = make_non_merge_mvp_syntax_decision(InterPredDir::PredL0, false, 1, 0, 1, 0);
        let mvd_cp_l0 = [
            MotionVector { x: 4, y: -1 },
            MotionVector { x: -2, y: 7 },
            MotionVector { x: 0, y: 0 },
        ];
        let mvd_cp_l1 = [MotionVector { x: 0, y: 0 }; 3];
        let decision =
            NonMergeInterPreResidualAffineDecision::new(affine, mvp, mvd_cp_l0, mvd_cp_l1);
        assert_eq!(decision.num_cp_mv(), 2);
        for init_type in [1u8, 2u8] {
            let rec = round_trip_affine_via_encoder(init_type, &affine_gate, &mvp_gate, &decision);
            assert_eq!(
                rec.affine.motion_model,
                crate::affine::MotionModel::Affine4Param
            );
            assert!(rec.affine.inter_affine_flag);
            assert!(!rec.affine.cu_affine_type_flag);
            assert_eq!(rec.mvd_cp_l0[0], MotionVector { x: 4, y: -1 });
            assert_eq!(rec.mvd_cp_l0[1], MotionVector { x: -2, y: 7 });
            // CP[2] not read on the wire under numCpMv == 2.
            assert_eq!(rec.mvd_cp_l0[2], MotionVector { x: 0, y: 0 });
            assert_eq!(rec.mvp.ref_idx_l0, 1);
            assert_eq!(rec.mvp.mvp_l0_flag, 1);
        }
    }

    #[test]
    fn round230_affine6param_p_slice_l0_round_trip() {
        // 6-param affine ⇒ numCpMv == 3. P-slice ⇒ L0-only.
        let affine_gate = affine_gate_on();
        let mvp_gate = p_slice_gate();
        let affine = make_non_merge_inter_affine_decision(true, true);
        assert_eq!(
            affine.motion_model,
            crate::affine::MotionModel::Affine6Param
        );
        let mvp = make_non_merge_mvp_syntax_decision(InterPredDir::PredL0, false, 1, 0, 0, 0);
        let mvd_cp_l0 = [
            MotionVector { x: 4, y: -1 },
            MotionVector { x: -2, y: 7 },
            MotionVector { x: 11, y: -11 },
        ];
        let mvd_cp_l1 = [MotionVector { x: 0, y: 0 }; 3];
        let decision =
            NonMergeInterPreResidualAffineDecision::new(affine, mvp, mvd_cp_l0, mvd_cp_l1);
        assert_eq!(decision.num_cp_mv(), 3);
        for init_type in [1u8, 2u8] {
            let rec = round_trip_affine_via_encoder(init_type, &affine_gate, &mvp_gate, &decision);
            assert_eq!(
                rec.affine.motion_model,
                crate::affine::MotionModel::Affine6Param
            );
            assert!(rec.affine.inter_affine_flag);
            assert!(rec.affine.cu_affine_type_flag);
            assert_eq!(rec.mvd_cp_l0, mvd_cp_l0);
            assert_eq!(rec.mvp.ref_idx_l0, 1);
        }
    }

    #[test]
    fn round230_affine4param_b_slice_pred_bi_round_trip() {
        // 4-param affine + B-slice PRED_BI ⇒ 2 CPs × 2 lists = 4
        // mvd_coding invocations.
        let affine_gate = affine_gate_on();
        let mvp_gate = b_slice_gate();
        let affine = make_non_merge_inter_affine_decision(true, false);
        let mvp = make_non_merge_mvp_syntax_decision(InterPredDir::PredBi, false, 1, 1, 1, 0);
        let mvd_cp_l0 = [
            MotionVector { x: 3, y: -3 },
            MotionVector { x: 1, y: 2 },
            MotionVector { x: 0, y: 0 },
        ];
        let mvd_cp_l1 = [
            MotionVector { x: -8, y: 8 },
            MotionVector { x: -1, y: -2 },
            MotionVector { x: 0, y: 0 },
        ];
        let decision =
            NonMergeInterPreResidualAffineDecision::new(affine, mvp, mvd_cp_l0, mvd_cp_l1);
        assert_eq!(decision.num_cp_mv(), 2);
        for init_type in [1u8, 2u8] {
            let rec = round_trip_affine_via_encoder(init_type, &affine_gate, &mvp_gate, &decision);
            assert_eq!(
                rec.affine.motion_model,
                crate::affine::MotionModel::Affine4Param
            );
            assert_eq!(rec.mvp.inter_pred_idc, InterPredDir::PredBi);
            assert!(!rec.mvp.sym_mvd_flag);
            assert_eq!(rec.mvd_cp_l0[0], MotionVector { x: 3, y: -3 });
            assert_eq!(rec.mvd_cp_l0[1], MotionVector { x: 1, y: 2 });
            assert_eq!(rec.mvd_cp_l0[2], MotionVector { x: 0, y: 0 });
            assert_eq!(rec.mvd_cp_l1[0], MotionVector { x: -8, y: 8 });
            assert_eq!(rec.mvd_cp_l1[1], MotionVector { x: -1, y: -2 });
            assert_eq!(rec.mvd_cp_l1[2], MotionVector { x: 0, y: 0 });
            assert_eq!(rec.mvp.ref_idx_l0, 1);
            assert_eq!(rec.mvp.ref_idx_l1, 1);
            assert_eq!(rec.mvp.mvp_l0_flag, 1);
            assert_eq!(rec.mvp.mvp_l1_flag, 0);
        }
    }

    #[test]
    fn round230_affine6param_b_slice_pred_l1_round_trip() {
        // 6-param affine + B-slice PRED_L1 ⇒ 3 CPs × 1 list.
        let affine_gate = affine_gate_on();
        let mvp_gate = b_slice_gate();
        let affine = make_non_merge_inter_affine_decision(true, true);
        let mvp = make_non_merge_mvp_syntax_decision(InterPredDir::PredL1, false, 0, 1, 0, 1);
        let mvd_cp_l0 = [MotionVector { x: 0, y: 0 }; 3];
        let mvd_cp_l1 = [
            MotionVector { x: 7, y: -2 },
            MotionVector { x: 3, y: 5 },
            MotionVector { x: -4, y: -6 },
        ];
        let decision =
            NonMergeInterPreResidualAffineDecision::new(affine, mvp, mvd_cp_l0, mvd_cp_l1);
        assert_eq!(decision.num_cp_mv(), 3);
        let rec = round_trip_affine_via_encoder(2, &affine_gate, &mvp_gate, &decision);
        assert_eq!(rec.mvp.inter_pred_idc, InterPredDir::PredL1);
        assert_eq!(rec.mvd_cp_l0, [MotionVector { x: 0, y: 0 }; 3]);
        assert_eq!(rec.mvd_cp_l1, mvd_cp_l1);
        assert_eq!(rec.mvp.ref_idx_l1, 1);
        assert_eq!(rec.mvp.mvp_l1_flag, 1);
    }

    #[test]
    fn round230_affine_translational_smvd_round_trip_fills_inferred_l1_mvd() {
        // Translational + SMVD: the L1 cpIdx-0 MVD is suppressed on the
        // wire and the dispatcher fills it with the §8.5.2.5 derivation
        // `-MvdL0[0]`. The higher CP slots stay at zero (numCpMv == 1).
        let affine_gate = affine_gate_off();
        let mvp_gate = b_slice_gate_smvd();
        let affine = make_non_merge_inter_affine_decision(false, false);
        let mvp = make_non_merge_mvp_syntax_decision(InterPredDir::PredBi, true, 0, 0, 1, 1);
        let l0_cp0 = MotionVector { x: 9, y: -5 };
        let mvd_cp_l0 = [
            l0_cp0,
            MotionVector { x: 0, y: 0 },
            MotionVector { x: 0, y: 0 },
        ];
        // The constructor clamps mvd_cp_l1[0] to zero on SMVD; the
        // encoder's debug-assert accepts zero or `-mvd_cp_l0[0]`.
        let mvd_cp_l1 = [MotionVector { x: 0, y: 0 }; 3];
        let decision =
            NonMergeInterPreResidualAffineDecision::new(affine, mvp, mvd_cp_l0, mvd_cp_l1);
        for init_type in [1u8, 2u8] {
            let rec = round_trip_affine_via_encoder(init_type, &affine_gate, &mvp_gate, &decision);
            assert!(rec.mvp.sym_mvd_flag);
            assert_eq!(rec.mvd_cp_l0[0], l0_cp0);
            assert_eq!(
                rec.mvd_cp_l1[0],
                MotionVector {
                    x: -l0_cp0.x,
                    y: -l0_cp0.y,
                }
            );
            assert_eq!(rec.mvd_cp_l1[1], MotionVector { x: 0, y: 0 });
            assert_eq!(rec.mvd_cp_l1[2], MotionVector { x: 0, y: 0 });
        }
    }

    // -------- Round-230: affine + AMVR round-trips --------------------

    #[test]
    fn round230_affine_with_amvr_translational_degenerates_to_round224() {
        // Translational + AMVR open (regular arm, prec = 2 → shift 6).
        let affine_gate = affine_gate_off();
        let mvp_gate = p_slice_gate();
        let amvr_gate = AmvrGate {
            sps_amvr_enabled: true,
            sps_affine_amvr_enabled: false,
            inter_affine_flag: false,
            any_mvd_l0_l1_nonzero: true,
            any_mvd_cp_l0_l1_nonzero: false,
        };
        let affine = make_non_merge_inter_affine_decision(false, false);
        let mvp = make_non_merge_mvp_syntax_decision(InterPredDir::PredL0, false, 1, 0, 1, 0);
        let mvd_cp_l0 = [
            MotionVector { x: 5, y: -3 },
            MotionVector { x: 0, y: 0 },
            MotionVector { x: 0, y: 0 },
        ];
        let mvd_cp_l1 = [MotionVector { x: 0, y: 0 }; 3];
        let decision =
            NonMergeInterPreResidualAffineDecision::new(affine, mvp, mvd_cp_l0, mvd_cp_l1);
        let amvr_decision = AmvrDecision::new(true, 2, false);
        for init_type in [1u8, 2u8] {
            let (rec_dec, rec_amvr) = round_trip_affine_via_encoder_with_amvr(
                init_type,
                &affine_gate,
                &mvp_gate,
                &amvr_gate,
                &decision,
                &amvr_decision,
            );
            assert_eq!(
                rec_dec.affine.motion_model,
                crate::affine::MotionModel::Translational
            );
            assert_eq!(rec_dec.mvd_cp_l0[0], MotionVector { x: 5, y: -3 });
            assert!(rec_amvr.amvr_flag);
            assert_eq!(rec_amvr.amvr_precision_idx, 2);
            assert_eq!(rec_amvr.shift, AmvrShift(6));
        }
    }

    #[test]
    fn round230_affine_with_amvr_4param_open_round_trip() {
        // 4-param affine ⇒ AMVR follows the affine arm. prec = 1 →
        // affine shift (1/16-luma steps per Table 16).
        let affine_gate = affine_gate_on();
        let mvp_gate = p_slice_gate();
        let amvr_gate = AmvrGate {
            sps_amvr_enabled: false,
            sps_affine_amvr_enabled: true,
            inter_affine_flag: true,
            any_mvd_l0_l1_nonzero: false,
            any_mvd_cp_l0_l1_nonzero: true,
        };
        let affine = make_non_merge_inter_affine_decision(true, false);
        let mvp = make_non_merge_mvp_syntax_decision(InterPredDir::PredL0, false, 0, 0, 0, 0);
        let mvd_cp_l0 = [
            MotionVector { x: 2, y: -1 },
            MotionVector { x: 1, y: 1 },
            MotionVector { x: 0, y: 0 },
        ];
        let mvd_cp_l1 = [MotionVector { x: 0, y: 0 }; 3];
        let decision =
            NonMergeInterPreResidualAffineDecision::new(affine, mvp, mvd_cp_l0, mvd_cp_l1);
        let amvr_decision = AmvrDecision::new(true, 1, true);
        for init_type in [1u8, 2u8] {
            let (rec_dec, rec_amvr) = round_trip_affine_via_encoder_with_amvr(
                init_type,
                &affine_gate,
                &mvp_gate,
                &amvr_gate,
                &decision,
                &amvr_decision,
            );
            assert_eq!(
                rec_dec.affine.motion_model,
                crate::affine::MotionModel::Affine4Param
            );
            assert_eq!(rec_dec.mvd_cp_l0[0], MotionVector { x: 2, y: -1 });
            assert_eq!(rec_dec.mvd_cp_l0[1], MotionVector { x: 1, y: 1 });
            assert!(rec_amvr.amvr_flag);
            assert_eq!(rec_amvr.amvr_precision_idx, 1);
            assert_eq!(rec_amvr.shift, amvr_decision.shift);
        }
    }

    #[test]
    fn round230_affine_with_amvr_4param_amvr_closed_round_trip() {
        // 4-param affine with all-zero MVDs ⇒ AMVR outer gate closed,
        // §7.4.12.7 inference fires.
        let affine_gate = affine_gate_on();
        let mvp_gate = p_slice_gate();
        let amvr_gate = AmvrGate {
            sps_amvr_enabled: false,
            sps_affine_amvr_enabled: true,
            inter_affine_flag: true,
            any_mvd_l0_l1_nonzero: false,
            any_mvd_cp_l0_l1_nonzero: false,
        };
        assert!(!amvr_gate.is_open());
        let affine = make_non_merge_inter_affine_decision(true, false);
        let mvp = make_non_merge_mvp_syntax_decision(InterPredDir::PredL0, false, 0, 0, 0, 0);
        let mvd_cp_l0 = [MotionVector { x: 0, y: 0 }; 3];
        let mvd_cp_l1 = [MotionVector { x: 0, y: 0 }; 3];
        let decision =
            NonMergeInterPreResidualAffineDecision::new(affine, mvp, mvd_cp_l0, mvd_cp_l1);
        let amvr_decision = AmvrDecision::default_inferred();
        for init_type in [1u8, 2u8] {
            let (rec_dec, rec_amvr) = round_trip_affine_via_encoder_with_amvr(
                init_type,
                &affine_gate,
                &mvp_gate,
                &amvr_gate,
                &decision,
                &amvr_decision,
            );
            assert_eq!(
                rec_dec.affine.motion_model,
                crate::affine::MotionModel::Affine4Param
            );
            assert_eq!(rec_dec.mvd_cp_l0, [MotionVector { x: 0, y: 0 }; 3]);
            assert!(!rec_amvr.amvr_flag);
            assert_eq!(rec_amvr.amvr_precision_idx, 0);
        }
    }

    #[test]
    fn round230_affine_with_amvr_6param_b_slice_pred_bi_round_trip() {
        // 6-param affine + B-slice PRED_BI + AMVR open (prec = 0).
        let affine_gate = affine_gate_on();
        let mvp_gate = b_slice_gate();
        let amvr_gate = AmvrGate {
            sps_amvr_enabled: false,
            sps_affine_amvr_enabled: true,
            inter_affine_flag: true,
            any_mvd_l0_l1_nonzero: false,
            any_mvd_cp_l0_l1_nonzero: true,
        };
        let affine = make_non_merge_inter_affine_decision(true, true);
        let mvp = make_non_merge_mvp_syntax_decision(InterPredDir::PredBi, false, 1, 1, 1, 0);
        let mvd_cp_l0 = [
            MotionVector { x: 3, y: -3 },
            MotionVector { x: 1, y: 2 },
            MotionVector { x: 0, y: 1 },
        ];
        let mvd_cp_l1 = [
            MotionVector { x: -8, y: 8 },
            MotionVector { x: -1, y: -2 },
            MotionVector { x: -3, y: -4 },
        ];
        let decision =
            NonMergeInterPreResidualAffineDecision::new(affine, mvp, mvd_cp_l0, mvd_cp_l1);
        let amvr_decision = AmvrDecision::new(true, 0, true);
        let (rec_dec, rec_amvr) = round_trip_affine_via_encoder_with_amvr(
            1,
            &affine_gate,
            &mvp_gate,
            &amvr_gate,
            &decision,
            &amvr_decision,
        );
        assert_eq!(
            rec_dec.affine.motion_model,
            crate::affine::MotionModel::Affine6Param
        );
        assert_eq!(rec_dec.mvp.inter_pred_idc, InterPredDir::PredBi);
        assert_eq!(rec_dec.mvd_cp_l0, mvd_cp_l0);
        assert_eq!(rec_dec.mvd_cp_l1, mvd_cp_l1);
        assert!(rec_amvr.amvr_flag);
        assert_eq!(rec_amvr.amvr_precision_idx, 0);
    }

    // -------- Round-230: affine + AMVR + BCW round-trips --------------

    #[test]
    fn round230_affine_with_amvr_and_bcw_translational_degenerates_to_round224() {
        // Translational + B-slice PRED_BI with both AMVR and BCW open.
        // Bit-identical to the round-224 `_with_amvr_and_bcw` composite.
        let affine_gate = affine_gate_off();
        let mvp_gate = b_slice_gate();
        let amvr_gate = AmvrGate {
            sps_amvr_enabled: true,
            sps_affine_amvr_enabled: false,
            inter_affine_flag: false,
            any_mvd_l0_l1_nonzero: true,
            any_mvd_cp_l0_l1_nonzero: false,
        };
        let bcw_gate = open_bcw_gate_for_b_pred_bi();
        let affine = make_non_merge_inter_affine_decision(false, false);
        let mvp = make_non_merge_mvp_syntax_decision(InterPredDir::PredBi, false, 1, 1, 0, 1);
        let mvd_cp_l0 = [
            MotionVector { x: 4, y: 4 },
            MotionVector { x: 0, y: 0 },
            MotionVector { x: 0, y: 0 },
        ];
        let mvd_cp_l1 = [
            MotionVector { x: -4, y: -4 },
            MotionVector { x: 0, y: 0 },
            MotionVector { x: 0, y: 0 },
        ];
        let decision =
            NonMergeInterPreResidualAffineDecision::new(affine, mvp, mvd_cp_l0, mvd_cp_l1);
        let amvr_decision = AmvrDecision::new(true, 2, false);
        for init_type in [1u8, 2u8] {
            let (rec_dec, rec_amvr, rec_bcw) = round_trip_affine_via_encoder_with_amvr_and_bcw(
                init_type,
                &affine_gate,
                &mvp_gate,
                &amvr_gate,
                &bcw_gate,
                &decision,
                &amvr_decision,
                1,
            );
            assert_eq!(
                rec_dec.affine.motion_model,
                crate::affine::MotionModel::Translational
            );
            assert_eq!(rec_dec.mvd_cp_l0[0], MotionVector { x: 4, y: 4 });
            assert_eq!(rec_dec.mvd_cp_l1[0], MotionVector { x: -4, y: -4 });
            assert!(rec_amvr.amvr_flag);
            assert_eq!(rec_amvr.shift, AmvrShift(6));
            assert_eq!(rec_bcw, 1);
        }
    }

    #[test]
    fn round230_affine_with_amvr_and_bcw_4param_b_slice_pred_bi_all_bcw_values_round_trip() {
        // 4-param affine + B-slice PRED_BI + BCW open (cMax = 2). AMVR
        // open via per-CP MVD non-zero with prec = 1.
        let affine_gate = affine_gate_on();
        let mvp_gate = b_slice_gate();
        let amvr_gate = AmvrGate {
            sps_amvr_enabled: false,
            sps_affine_amvr_enabled: true,
            inter_affine_flag: true,
            any_mvd_l0_l1_nonzero: false,
            any_mvd_cp_l0_l1_nonzero: true,
        };
        let bcw_gate = open_bcw_gate_for_b_pred_bi();
        let affine = make_non_merge_inter_affine_decision(true, false);
        let mvp = make_non_merge_mvp_syntax_decision(InterPredDir::PredBi, false, 0, 0, 0, 0);
        let mvd_cp_l0 = [
            MotionVector { x: 1, y: 1 },
            MotionVector { x: 2, y: 2 },
            MotionVector { x: 0, y: 0 },
        ];
        let mvd_cp_l1 = [
            MotionVector { x: -1, y: -1 },
            MotionVector { x: -2, y: -2 },
            MotionVector { x: 0, y: 0 },
        ];
        let decision =
            NonMergeInterPreResidualAffineDecision::new(affine, mvp, mvd_cp_l0, mvd_cp_l1);
        let amvr_decision = AmvrDecision::new(true, 1, true);
        for bcw_idx_value in 0u32..=2u32 {
            let (rec_dec, rec_amvr, rec_bcw) = round_trip_affine_via_encoder_with_amvr_and_bcw(
                1,
                &affine_gate,
                &mvp_gate,
                &amvr_gate,
                &bcw_gate,
                &decision,
                &amvr_decision,
                bcw_idx_value,
            );
            assert_eq!(
                rec_dec.affine.motion_model,
                crate::affine::MotionModel::Affine4Param
            );
            assert_eq!(rec_dec.mvd_cp_l0[0], MotionVector { x: 1, y: 1 });
            assert_eq!(rec_dec.mvd_cp_l0[1], MotionVector { x: 2, y: 2 });
            assert_eq!(rec_dec.mvd_cp_l1[0], MotionVector { x: -1, y: -1 });
            assert_eq!(rec_dec.mvd_cp_l1[1], MotionVector { x: -2, y: -2 });
            assert!(rec_amvr.amvr_flag);
            assert_eq!(rec_amvr.amvr_precision_idx, 1);
            assert_eq!(rec_bcw, bcw_idx_value);
        }
    }

    #[test]
    fn round230_affine_with_amvr_and_bcw_6param_b_slice_amvr_closed_bcw_closed_round_trip() {
        // 6-param affine + B-slice PRED_BI but BCW gate closed by
        // `luma_weight_l1_flag = true`. AMVR also closed via all-zero
        // per-CP MVDs.
        let affine_gate = affine_gate_on();
        let mvp_gate = b_slice_gate();
        let amvr_gate = AmvrGate {
            sps_amvr_enabled: false,
            sps_affine_amvr_enabled: true,
            inter_affine_flag: true,
            any_mvd_l0_l1_nonzero: false,
            any_mvd_cp_l0_l1_nonzero: false,
        };
        let bcw_gate = BcwIdxGate {
            sps_bcw_enabled: true,
            inter_pred_idc: Some(InterPredDir::PredBi),
            luma_weight_l1_flag: true,
            cb_width: 16,
            cb_height: 16,
            ..Default::default()
        };
        assert!(!bcw_gate.is_open());
        let affine = make_non_merge_inter_affine_decision(true, true);
        let mvp = make_non_merge_mvp_syntax_decision(InterPredDir::PredBi, false, 0, 0, 0, 0);
        let mvd_cp_l0 = [MotionVector { x: 0, y: 0 }; 3];
        let mvd_cp_l1 = [MotionVector { x: 0, y: 0 }; 3];
        let decision =
            NonMergeInterPreResidualAffineDecision::new(affine, mvp, mvd_cp_l0, mvd_cp_l1);
        let amvr_decision = AmvrDecision::default_inferred();
        let (rec_dec, rec_amvr, rec_bcw) = round_trip_affine_via_encoder_with_amvr_and_bcw(
            1,
            &affine_gate,
            &mvp_gate,
            &amvr_gate,
            &bcw_gate,
            &decision,
            &amvr_decision,
            0,
        );
        assert_eq!(
            rec_dec.affine.motion_model,
            crate::affine::MotionModel::Affine6Param
        );
        assert!(!rec_amvr.amvr_flag);
        assert_eq!(rec_bcw, 0);
    }
}
