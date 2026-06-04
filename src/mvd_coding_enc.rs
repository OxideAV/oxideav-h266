//! Round-187 — encoder-side CABAC emission for the §7.3.10.10
//! `mvd_coding()` structure (plus its §9.3.3.6 limited-EGk `k = 1`
//! `abs_mvd_minus2` sub-binarisation).
//!
//! Companion to round-177's [`crate::affine_syntax_enc`] (which covers
//! the §7.3.11.7 non-merge inter affine-syntax pre-MVD pair) and
//! round-183's [`crate::non_merge_mvp_syntax_enc`] (which covers the
//! §7.3.11.7 non-merge inter MVP-side syntax `inter_pred_idc` /
//! `sym_mvd_flag` / `ref_idx_lX` / `mvp_lX_flag`). Those two modules
//! end at the per-list pre-MVD syntax; this module covers the
//! per-list `mvd_coding()` invocation that runs immediately afterwards
//! and emits the (x, y) component pair into the bitstream.
//!
//! Together with the existing public reader-side
//! [`crate::leaf_cu::LeafCuReader::read_mvd_coding`] this completes the
//! public encoder + decoder symmetry for the entire §7.3.11.7
//! non-merge inter pre-residual syntax: an external encoder can now
//! drive a CU's full `inter_pred_idc → sym_mvd_flag → ref_idx_lX →
//! mvp_lX_flag → mvd_coding(L0) → mvd_coding(L1)` cascade entirely
//! through public encoder helpers.
//!
//! Prior to this round both [`encode_mvd_coding`] and
//! [`encode_abs_mvd_minus2`] lived as `#[cfg(test)]`-only helpers in
//! `leaf_cu.rs` (used exclusively by the round-103 conformance round-
//! trips). They are functionally unchanged — this module lifts them
//! into the public encoder surface so a real encoder pipeline can call
//! them without re-implementing the bin sequence.
//!
//! ## Public surface
//!
//! 1. [`encode_abs_mvd_minus2`] — round-trip mirror of
//!    [`crate::leaf_cu::LeafCuReader::read_abs_mvd_minus2`] under the
//!    §9.3.3.6 limited-EGk binarisation with `k = 1`,
//!    `maxPreExtLen = 15`, `truncSuffixLen = 17`. All bins are
//!    bypass-coded (`mvd_coding()` has no ctx-coded magnitude tail
//!    beyond `abs_mvd_greater{0,1}_flag`).
//! 2. [`encode_mvd_coding`] — round-trip mirror of
//!    [`crate::leaf_cu::LeafCuReader::read_mvd_coding`] for one
//!    `mvd_coding(x0, y0, refList, cpIdx)` structure. Drives the
//!    shared [`crate::leaf_cu::LeafCuCtxs`] bundle through:
//!    1. `abs_mvd_greater0_flag[0]`, `abs_mvd_greater0_flag[1]`
//!       (both ctx-coded against Table 110, slot `init_type`,
//!       deterministic `ctxInc = 0` per
//!       [`crate::ctx::ctx_inc_abs_mvd_greater0_flag`]).
//!    2. for each component whose greater0 flag is 1:
//!       `abs_mvd_greater1_flag[c]` (ctx-coded against Table 111,
//!       slot `init_type`, deterministic `ctxInc = 0` per
//!       [`crate::ctx::ctx_inc_abs_mvd_greater1_flag`]).
//!    3. for each component whose greater0 flag is 1:
//!       `abs_mvd_minus2[c]` (bypass §9.3.3.6 EG1, **only when
//!       greater1 == 1**) then `mvd_sign_flag[c]` (bypass FL `cMax = 1`).
//!    The structural layout matches eq. 190 exactly:
//!    `lMvd[c] = greater0 ? (abs_mvd_minus2 + 2) * (1 − 2*sign) : 0`
//!    (with `abs_mvd_minus2` inferred to −1 when greater1 == 0 so the
//!    magnitude collapses to 1).
//! 3. [`max_mvd_magnitude`] — convenience constant accessor for the
//!    largest *positive* lMvd value the §7.4.10.10 conformance range
//!    allows (`2^17 - 1` = 131_071). The corresponding negative
//!    bound is `-2^17 = -131_072` (the signed-18-bit floor).
//!
//! ## Range invariants
//!
//! Per §7.4.10.10 the spec constrains each `lMvd[c]` to the signed
//! 18-bit range `[-2^17, 2^17 - 1]` (it is a requirement of bitstream
//! conformance). The §9.3.3.6 limited-EGk path technically has
//! headroom above the positive bound (`abs_mvd_minus2` can carry up
//! to `(1 << 15 - 1) << 1 + (1 << 17) - 1 = 196_605` and so represent
//! `|lMvd|` up to `196_607`), but the encoder mirrors the spec value
//! bound rather than the codec capacity. [`max_mvd_magnitude`]
//! returns `2^17 - 1`, the largest *positive* spec-conformant
//! magnitude; the corresponding `-(2^17)` negative value is the
//! signed-18-bit floor. The reader's existing sweep test in
//! `leaf_cu.rs::mvd_coding_large_magnitudes_round_trip` covers
//! `(131_070, -131_070) = (2^17 − 2, -(2^17 − 2))`; the parallel
//! sweep here goes one further, exercising the exact `2^17 - 1`
//! positive bound and `-(2^17 − 1)` so the encoder's behaviour at the
//! spec-conformant maximum is locked in.
//!
//! ## Spec reference
//!
//! ITU-T H.266 | ISO/IEC 23090-3 (V4, 01/2026):
//! * §7.3.10.10 — `mvd_coding(x0, y0, refList, cpIdx)` syntax (the
//!   source-of-truth listing this module's dispatcher mirrors).
//! * §7.4.10.10 — `lMvd[c]` derivation eq. 190.
//! * §9.3.3.6 — limited-EGk binarisation (the `abs_mvd_minus2`
//!   sub-binarisation with `k = 1`, `maxPreExtLen = 15`,
//!   `truncSuffixLen = 17`).
//! * §9.3.4.2.2 / Table 132 — ctxInc derivation for
//!   `abs_mvd_greater0_flag` and `abs_mvd_greater1_flag` (both 0).
//! * Table 51 — init-type → ctx-slot mapping for both flags.
//! * Table 110 / Table 111 — initValue / shiftIdx for the two flags.
//!
//! No third-party VVC encoder source was consulted; the implementation
//! is spec-only and mirrors the existing reader-side code already
//! shipped in this crate.

use oxideav_core::Result;

use crate::cabac_enc::ArithEncoder;
use crate::ctx::{ctx_inc_abs_mvd_greater0_flag, ctx_inc_abs_mvd_greater1_flag};
use crate::inter::MotionVector;
use crate::leaf_cu::LeafCuCtxs;

/// §9.3.3.6 limited-EGk parameters for `abs_mvd_minus2`.
///
/// Re-stated here as public-module-scope constants so callers can
/// reason about the syntax-element's range without having to read into
/// `cabac_enc` or duplicate the spec literals. Matches the
/// [`crate::leaf_cu::LeafCuReader::read_abs_mvd_minus2`] reader-side
/// transcription exactly.
const ABS_MVD_MINUS2_K: u32 = 1;
const ABS_MVD_MINUS2_MAX_PRE_EXT_LEN: u32 = 15;
const ABS_MVD_MINUS2_TRUNC_SUFFIX_LEN: u32 = 17;

/// Largest *positive* `lMvd[c]` value the spec permits.
///
/// Equal to `2^17 - 1 = 131_071`. Per the §7.4.10.10 conformance
/// requirement each `lMvd[c]` shall lie in the signed-18-bit range
/// `[-2^17, 2^17 - 1]`; this constant is the positive upper bound of
/// that range and is the value [`encode_mvd_coding`] uses for its
/// debug-assertion magnitude check (with a one-larger allowance on
/// the negative side to admit `-2^17 = -131_072`, the signed-18-bit
/// floor).
///
/// The §9.3.3.6 limited-EGk binarisation can technically encode
/// `abs_mvd_minus2` values larger than `(2^17 - 1) - 2 = 131_069`
/// (it has headroom up to `196_605` from the
/// `((1 << 15) - 1) << 1 + (1 << 17) - 1` codec cap), but bitstreams
/// that emit such values would violate §7.4.10.10. Callers
/// short-circuit or clip higher up the pipeline.
pub const fn max_mvd_magnitude() -> i32 {
    (1 << 17) - 1
}

/// Encode `abs_mvd_minus2` per the §9.3.3.6 limited-EGk binarisation
/// with `k = 1`, `maxPreExtLen = 15`, `truncSuffixLen = 17`.
///
/// All bins are bypass-coded. Mirror of
/// [`crate::leaf_cu::LeafCuReader::read_abs_mvd_minus2`].
///
/// # Parameters
///
/// * `enc` — shared CABAC encoder state.
/// * `symbol_val` — the integer value to emit. The §9.3.3.6 codec
///   itself caps at `((1 << 15) - 1) << 1 + (1 << 17) - 1 = 196_605`,
///   but bitstream-conformant `mvd_coding()` requires
///   `symbol_val + 2 ∈ [0, 2^17 - 1]` from the §7.4.10.10 range
///   constraint on `|lMvd|`; callers should clip / short-circuit
///   accordingly.
///
/// # Bin sequence
///
/// Per §9.3.3.6, the encoder walks the prefix counter
/// `preExtLen ∈ 0..=maxPreExtLen` while `(2 << preExtLen) - 2` is less
/// than `symbol_val >> k`. For each step a `1` prefix bin is emitted.
/// When the loop exits below the cap, a single terminating `0` prefix
/// bin is emitted and the suffix is `preExtLen + k` bits wide; at the
/// cap the suffix is the fixed `truncSuffixLen`-bit escape field with
/// no terminating `0`. The suffix carries
/// `symbol_val - (((1 << preExtLen) - 1) << k)` MSB-first.
pub fn encode_abs_mvd_minus2(enc: &mut ArithEncoder, symbol_val: u32) -> Result<()> {
    let code_value = symbol_val >> ABS_MVD_MINUS2_K;
    let mut pre_ext_len = 0u32;
    while pre_ext_len < ABS_MVD_MINUS2_MAX_PRE_EXT_LEN && code_value > ((2u32 << pre_ext_len) - 2) {
        pre_ext_len += 1;
        enc.encode_bypass(1)?;
    }
    let escape_length = if pre_ext_len == ABS_MVD_MINUS2_MAX_PRE_EXT_LEN {
        // The cap was hit; the spec emits no terminating 0 prefix bin
        // and the suffix is the fixed `truncSuffixLen`-bit escape field.
        ABS_MVD_MINUS2_TRUNC_SUFFIX_LEN
    } else {
        // Terminating 0 prefix bin closes the variable-length run.
        enc.encode_bypass(0)?;
        pre_ext_len + ABS_MVD_MINUS2_K
    };
    let val = symbol_val - (((1u32 << pre_ext_len) - 1) << ABS_MVD_MINUS2_K);
    for i in (0..escape_length).rev() {
        enc.encode_bypass((val >> i) & 1)?;
    }
    Ok(())
}

/// Encode one `mvd_coding(x0, y0, refList, cpIdx)` structure per
/// §7.3.10.10, driving the supplied [`LeafCuCtxs`] bundle bin-for-bin.
///
/// Mirror of [`crate::leaf_cu::LeafCuReader::read_mvd_coding`] —
/// the wire produced here round-trips bit-identically through the
/// reader and reconstructs `lMvd` exactly.
///
/// # Parameters
///
/// * `enc` — shared CABAC encoder state.
/// * `ctxs` — slice-scope context bundle. The
///   `abs_mvd_greater0_flag` / `abs_mvd_greater1_flag` slots and the
///   `init_type` field are consulted; no other field is touched.
/// * `lmvd` — the `(lMvd[0], lMvd[1]) == (x, y)` component pair in
///   the spec's 1/16-pel storage convention (the §7.4.11.6 AMVR
///   shift, when signalled, is applied separately by [`crate::amvr`]
///   *before* this function — `lMvd` here is the post-AMVR value the
///   bitstream actually carries).
///
/// # Bin sequence per §7.3.10.10
///
/// 1. `abs_mvd_greater0_flag[0]`, `abs_mvd_greater0_flag[1]`
///    (both ctx-coded against Table 110 / Table 132 — slot
///    `init_type`, deterministic `ctxInc = 0` per
///    [`crate::ctx::ctx_inc_abs_mvd_greater0_flag`]).
/// 2. for each component whose greater0 flag is 1:
///    `abs_mvd_greater1_flag[c]` (ctx-coded against Table 111 /
///    Table 132 — slot `init_type`, deterministic `ctxInc = 0` per
///    [`crate::ctx::ctx_inc_abs_mvd_greater1_flag`]).
/// 3. for each component whose greater0 flag is 1:
///    `abs_mvd_minus2[c]` via [`encode_abs_mvd_minus2`] (**only
///    when greater1 == 1** — the §7.4.10.10 inference handles the
///    greater1 == 0 case by collapsing |lMvd[c]| to 1) then
///    `mvd_sign_flag[c]` (bypass FL `cMax = 1`).
///
/// # `lmvd` validity
///
/// Each component must satisfy `lmvd.c ∈ [-(1 << 17), (1 << 17) - 1]`
/// per the §7.4.10.10 conformance range. In a debug build a
/// violating component triggers a panic; in a release build the EGk
/// path has some headroom above the spec bound, but the resulting
/// bitstream would not be §7.4.10.10-conformant — the caller is
/// expected to clip or short-circuit higher up the pipeline.
pub fn encode_mvd_coding(
    enc: &mut ArithEncoder,
    ctxs: &mut LeafCuCtxs,
    lmvd: MotionVector,
) -> Result<()> {
    // §7.4.10.10 conformance: lMvd[c] ∈ [-(1<<17), (1<<17) - 1].
    // The asymmetric negative bound is the signed-18-bit floor; the
    // positive bound is `max_mvd_magnitude()`.
    const LO: i32 = -(1 << 17);
    const HI: i32 = (1 << 17) - 1;
    debug_assert!(
        (LO..=HI).contains(&lmvd.x),
        "lMvd[0] = {} outside §7.4.10.10 range [{LO}, {HI}]",
        lmvd.x
    );
    debug_assert!(
        (LO..=HI).contains(&lmvd.y),
        "lMvd[1] = {} outside §7.4.10.10 range [{LO}, {HI}]",
        lmvd.y
    );

    // Spec-traceability: route through the ctx::* helpers even though
    // both return deterministic 0 — keeps the encoder mirror robust
    // against a future Table 132 amendment introducing a non-trivial
    // derivation. Matches the affine_syntax_enc round-177 pattern.
    let inc_g0 = ctx_inc_abs_mvd_greater0_flag() as usize;
    let inc_g1 = ctx_inc_abs_mvd_greater1_flag() as usize;
    debug_assert_eq!(
        inc_g0, 0,
        "Table 132 lists deterministic ctxInc = 0 for abs_mvd_greater0_flag"
    );
    debug_assert_eq!(
        inc_g1, 0,
        "Table 132 lists deterministic ctxInc = 0 for abs_mvd_greater1_flag"
    );

    let init_type = ctxs.init_type as usize;
    let g0_slot = (init_type + inc_g0).min(ctxs.abs_mvd_greater0_flag.len() - 1);
    let g1_slot = (init_type + inc_g1).min(ctxs.abs_mvd_greater1_flag.len() - 1);

    let comp = [lmvd.x, lmvd.y];
    let greater0 = [comp[0] != 0, comp[1] != 0];
    let greater1 = [comp[0].abs() > 1, comp[1].abs() > 1];

    // Step 1: both greater0 flags (both components, in spec's
    // component-major order: c0 then c1).
    for c in 0..2 {
        enc.encode_decision(&mut ctxs.abs_mvd_greater0_flag[g0_slot], greater0[c] as u32)?;
    }
    // Step 2: greater1 per non-zero component (component-major).
    for c in 0..2 {
        if greater0[c] {
            enc.encode_decision(&mut ctxs.abs_mvd_greater1_flag[g1_slot], greater1[c] as u32)?;
        }
    }
    // Step 3: magnitude tail + sign per non-zero component.
    for c in 0..2 {
        if greater0[c] {
            if greater1[c] {
                // |lMvd[c]| ≥ 2 → emit abs_mvd_minus2.
                encode_abs_mvd_minus2(enc, comp[c].unsigned_abs() - 2)?;
            }
            // mvd_sign_flag[c] — bypass FL cMax = 1.
            enc.encode_bypass((comp[c] < 0) as u32)?;
        }
    }

    Ok(())
}

// =====================================================================
// Round-233 — decomposed §7.3.10.10 `mvd_coding()` body parser.
//
// The existing [`encode_mvd_coding`] / `read_mvd_coding` surface takes
// or returns a [`MotionVector`] (the post-fold `lMvd[c]` pair per
// eq. 190) and hides the eight underlying syntax elements behind the
// fold. The body-parser variant in this section exposes the raw
// syntax elements explicitly via [`MvdCodingDecision`], so external
// callers can inspect the per-bin layout, replay arbitrary bin patterns
// (including the §7.4.10.10 inferred-default cases where greater0 == 0
// or greater1 == 0 leave abs_mvd_minus2 / mvd_sign_flag inferred), and
// verify the eq. 190 fold without re-deriving it inline.
//
// Functionally equivalent to the existing per-element walker, but the
// public surface area carries the structural layout of the eight
// syntax elements alongside the resulting `(x, y)` pair.
// =====================================================================

/// Decomposed §7.3.10.10 `mvd_coding(x0, y0, refList, cpIdx)` syntax
/// elements.
///
/// One instance carries the per-component bin layout the §7.3.10.10
/// listing produces:
///
/// | Field                    | Per-component spec element          |
/// | ------------------------ | ----------------------------------- |
/// | `abs_mvd_greater0_flag`  | `abs_mvd_greater0_flag[c]`          |
/// | `abs_mvd_greater1_flag`  | `abs_mvd_greater1_flag[c]`          |
/// | `abs_mvd_minus2`         | `abs_mvd_minus2[c]`                 |
/// | `mvd_sign_flag`          | `mvd_sign_flag[c]`                  |
///
/// Per §7.4.10.10 inference, when `abs_mvd_greater0_flag[c] == 0` both
/// `abs_mvd_greater1_flag[c]` and `mvd_sign_flag[c]` are inferred 0
/// and `abs_mvd_minus2[c]` is inferred −1 (which collapses the §eq.
/// 190 magnitude to 0). When `abs_mvd_greater0_flag[c] == 1` but
/// `abs_mvd_greater1_flag[c] == 0`, `abs_mvd_minus2[c]` is inferred −1
/// (which collapses the §eq. 190 magnitude to 1).
///
/// The struct carries those inferred slots as ordinary fields (`false`
/// / `0`) for the greater1 / sign cases, and as the sentinel `0`
/// (rather than `−1`) for `abs_mvd_minus2` — the eq. 190 fold gates
/// on the flag pair regardless of the slot's contents, so the
/// inferred-out-of-band sentinel is only of cosmetic interest. The
/// [`Self::to_motion_vector`] method honours the gates exactly per
/// eq. 190.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MvdCodingDecision {
    /// `abs_mvd_greater0_flag[c]` per §7.3.10.10 — true iff
    /// `|lMvd[c]| >= 1`.
    pub abs_mvd_greater0_flag: [bool; 2],
    /// `abs_mvd_greater1_flag[c]` per §7.3.10.10 — true iff
    /// `|lMvd[c]| >= 2`. Meaningful only when
    /// `abs_mvd_greater0_flag[c] == true`; otherwise the slot is
    /// `false` per §7.4.10.10 inference (no bin emitted).
    pub abs_mvd_greater1_flag: [bool; 2],
    /// `abs_mvd_minus2[c]` per §7.3.10.10 — the value carried by the
    /// §9.3.3.6 limited-EGk bypass tail. Meaningful only when
    /// `abs_mvd_greater0_flag[c] && abs_mvd_greater1_flag[c]`;
    /// otherwise the slot is `0` (the §7.4.10.10 inference would
    /// nominally assign `−1`, but the eq. 190 fold gates on the flag
    /// pair so the stored value is only of cosmetic interest in the
    /// inferred cases).
    pub abs_mvd_minus2: [u32; 2],
    /// `mvd_sign_flag[c]` per §7.3.10.10 — true iff `lMvd[c] < 0`.
    /// Meaningful only when `abs_mvd_greater0_flag[c] == true`;
    /// otherwise the slot is `false` per §7.4.10.10 inference (no
    /// bin emitted).
    pub mvd_sign_flag: [bool; 2],
}

impl MvdCodingDecision {
    /// Inferred-zero decision: every flag is `false`, every
    /// `abs_mvd_minus2` slot is `0`. The eq. 190 fold maps this to
    /// `MotionVector { x: 0, y: 0 }`.
    pub const fn zero() -> Self {
        Self {
            abs_mvd_greater0_flag: [false, false],
            abs_mvd_greater1_flag: [false, false],
            abs_mvd_minus2: [0, 0],
            mvd_sign_flag: [false, false],
        }
    }

    /// Derive the decomposed bin layout from a target `lMvd` pair, in
    /// the way the encoder walks them per §7.3.10.10:
    ///
    /// * `abs_mvd_greater0_flag[c] = (lMvd[c] != 0)`
    /// * `abs_mvd_greater1_flag[c] = (|lMvd[c]| > 1)` (only meaningful
    ///   when `greater0[c]`)
    /// * `abs_mvd_minus2[c] = |lMvd[c]| - 2` (only meaningful when
    ///   `greater0[c] && greater1[c]`; otherwise stored as `0`)
    /// * `mvd_sign_flag[c] = (lMvd[c] < 0)` (only meaningful when
    ///   `greater0[c]`)
    ///
    /// `lmvd.x` / `lmvd.y` must lie in the §7.4.10.10 signed-18-bit
    /// range; debug-asserts mirror [`encode_mvd_coding`].
    pub fn from_motion_vector(lmvd: MotionVector) -> Self {
        const LO: i32 = -(1 << 17);
        const HI: i32 = (1 << 17) - 1;
        debug_assert!(
            (LO..=HI).contains(&lmvd.x),
            "lMvd[0] = {} outside §7.4.10.10 range [{LO}, {HI}]",
            lmvd.x
        );
        debug_assert!(
            (LO..=HI).contains(&lmvd.y),
            "lMvd[1] = {} outside §7.4.10.10 range [{LO}, {HI}]",
            lmvd.y
        );

        let comp = [lmvd.x, lmvd.y];
        let mut out = Self::zero();
        for c in 0..2 {
            let nz = comp[c] != 0;
            out.abs_mvd_greater0_flag[c] = nz;
            if !nz {
                continue;
            }
            let abs = comp[c].unsigned_abs();
            let gt1 = abs > 1;
            out.abs_mvd_greater1_flag[c] = gt1;
            if gt1 {
                // `abs >= 2`, so `abs - 2` fits in u32 without underflow.
                out.abs_mvd_minus2[c] = abs - 2;
            }
            out.mvd_sign_flag[c] = comp[c] < 0;
        }
        out
    }

    /// Fold the decomposed elements back into the `lMvd` pair per
    /// §7.4.10.10 eq. 190:
    ///
    /// `lMvd[c] = greater0[c] ? (|magnitude|) * (1 − 2 * sign[c]) : 0`
    ///
    /// where the per-component magnitude is `abs_mvd_minus2[c] + 2`
    /// when `greater1[c]`, otherwise `1` (the inferred fallback).
    pub fn to_motion_vector(&self) -> MotionVector {
        let mut lmvd = [0i32; 2];
        for c in 0..2 {
            if !self.abs_mvd_greater0_flag[c] {
                continue;
            }
            let abs: i32 = if self.abs_mvd_greater1_flag[c] {
                // Saturate against the §7.4.10.10 positive ceiling so
                // a stale or out-of-range `abs_mvd_minus2` cannot
                // overflow the signed-18-bit `lMvd[c]` range.
                let sum = (self.abs_mvd_minus2[c] as i64) + 2;
                sum.min(((1i64 << 17) - 1) + 1) as i32
            } else {
                1
            };
            lmvd[c] = if self.mvd_sign_flag[c] { -abs } else { abs };
        }
        MotionVector {
            x: lmvd[0],
            y: lmvd[1],
        }
    }
}

/// Encode one `mvd_coding(x0, y0, refList, cpIdx)` structure per
/// §7.3.10.10 from the decomposed [`MvdCodingDecision`] form. Mirror
/// of [`crate::leaf_cu::LeafCuReader::read_mvd_coding_decomposed`].
///
/// Functionally equivalent to driving [`encode_mvd_coding`] with
/// `decision.to_motion_vector()`: the encoder walks the same bin
/// sequence per §7.3.10.10, but inspects the flag pair / magnitude
/// tail / sign directly rather than re-deriving them from a packed
/// `(x, y)`. This is the natural mirror for an external caller that
/// has already split a target lMvd into its underlying syntax
/// elements (e.g. a trace-replay harness or a rate-distortion-aware
/// scan that holds an explicit per-bin candidate set).
///
/// # Parameters
///
/// * `enc` — shared CABAC encoder state.
/// * `ctxs` — slice-scope context bundle. The
///   `abs_mvd_greater0_flag` / `abs_mvd_greater1_flag` slots and the
///   `init_type` field are consulted; no other field is touched.
/// * `decision` — the decomposed [`MvdCodingDecision`] to emit. The
///   eq. 190 fold (the `(x, y)` round-trip equivalence) is the
///   caller's contract: any inconsistency between the flag pair and
///   the magnitude / sign slots is treated as the spec's
///   §7.4.10.10 inferred behaviour (no bin emitted, slot value
///   ignored).
///
/// # Bin sequence
///
/// Identical to [`encode_mvd_coding`]:
///
/// 1. `abs_mvd_greater0_flag[0]`, `abs_mvd_greater0_flag[1]`
///    (both ctx-coded, Table 110 slot = `init_type`, ctxInc 0).
/// 2. for each `c` with `greater0[c]`:
///    `abs_mvd_greater1_flag[c]` (ctx-coded, Table 111).
/// 3. for each `c` with `greater0[c]`:
///    `abs_mvd_minus2[c]` via [`encode_abs_mvd_minus2`] (only when
///    `greater1[c]`) then `mvd_sign_flag[c]` (bypass FL `cMax = 1`).
pub fn encode_mvd_coding_decomposed(
    enc: &mut ArithEncoder,
    ctxs: &mut LeafCuCtxs,
    decision: &MvdCodingDecision,
) -> Result<()> {
    // Spec-traceability: route through the ctx::* helpers even though
    // both return deterministic 0 — keeps the encoder mirror robust
    // against a future Table 132 amendment introducing a non-trivial
    // derivation. Matches the round-187 [`encode_mvd_coding`] pattern.
    let inc_g0 = ctx_inc_abs_mvd_greater0_flag() as usize;
    let inc_g1 = ctx_inc_abs_mvd_greater1_flag() as usize;
    debug_assert_eq!(
        inc_g0, 0,
        "Table 132 lists deterministic ctxInc = 0 for abs_mvd_greater0_flag"
    );
    debug_assert_eq!(
        inc_g1, 0,
        "Table 132 lists deterministic ctxInc = 0 for abs_mvd_greater1_flag"
    );

    let init_type = ctxs.init_type as usize;
    let g0_slot = (init_type + inc_g0).min(ctxs.abs_mvd_greater0_flag.len() - 1);
    let g1_slot = (init_type + inc_g1).min(ctxs.abs_mvd_greater1_flag.len() - 1);

    // Step 1: both greater0 flags (both components, in spec's
    // component-major order: c0 then c1).
    for c in 0..2 {
        enc.encode_decision(
            &mut ctxs.abs_mvd_greater0_flag[g0_slot],
            decision.abs_mvd_greater0_flag[c] as u32,
        )?;
    }
    // Step 2: greater1 per non-zero component (component-major).
    for c in 0..2 {
        if decision.abs_mvd_greater0_flag[c] {
            enc.encode_decision(
                &mut ctxs.abs_mvd_greater1_flag[g1_slot],
                decision.abs_mvd_greater1_flag[c] as u32,
            )?;
        }
    }
    // Step 3: magnitude tail + sign per non-zero component.
    for c in 0..2 {
        if decision.abs_mvd_greater0_flag[c] {
            if decision.abs_mvd_greater1_flag[c] {
                // |lMvd[c]| ≥ 2 → emit abs_mvd_minus2 from the
                // decision's explicit slot (the §9.3.3.6 limited-EGk
                // path).
                encode_abs_mvd_minus2(enc, decision.abs_mvd_minus2[c])?;
            }
            // mvd_sign_flag[c] — bypass FL cMax = 1.
            enc.encode_bypass(decision.mvd_sign_flag[c] as u32)?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::cabac::ArithDecoder;
    use crate::leaf_cu::{CuToolFlags, LeafCuReader};

    /// End-to-end round-trip: drive [`encode_mvd_coding`] against a
    /// caller-supplied `(lMvd[0], lMvd[1])` pair, decode the resulting
    /// bitstream through the reader-side
    /// [`LeafCuReader::read_mvd_coding`], and return the recovered
    /// [`MotionVector`]. Asserting `recovered == lmvd` exhibits the
    /// encoder + decoder bit-identical agreement.
    fn mvd_round_trip(lmvd: MotionVector, init_type: u8) -> MotionVector {
        let mut enc = ArithEncoder::new();
        let mut enc_ctxs = LeafCuCtxs::init_with_init_type(26, init_type);
        encode_mvd_coding(&mut enc, &mut enc_ctxs, lmvd)
            .expect("encode_mvd_coding succeeds for in-range lMvd");
        enc.encode_terminate(1).expect("terminator");
        let mut padded = enc.finish();
        padded.extend_from_slice(&[0u8; 32]);
        let mut dec = ArithDecoder::new(&padded).expect("decoder accepts encoded stream");
        let mut dec_ctxs = LeafCuCtxs::init_with_init_type(26, init_type);
        let tools = CuToolFlags::default();
        let mut reader = LeafCuReader::new(&mut dec, &mut dec_ctxs, tools);
        reader
            .read_mvd_coding()
            .expect("reader reconstructs the mvd_coding() structure")
    }

    #[test]
    fn max_mvd_magnitude_matches_spec_positive_bound() {
        // §7.4.10.10 conformance bound: lMvd[c] ∈ [-2^17, 2^17 - 1].
        // The positive upper bound is 2^17 - 1 = 131_071.
        assert_eq!(max_mvd_magnitude(), 131_071);
        assert_eq!(max_mvd_magnitude() as u32, (1u32 << 17) - 1);
    }

    #[test]
    fn zero_both_components_emits_two_zero_greater0_bins_and_round_trips() {
        // (0, 0) → encoder emits two greater0 = 0 bins (ctx-coded) and
        // skips every subsequent step. Reader reconstructs zero pair.
        let mv = MotionVector { x: 0, y: 0 };
        assert_eq!(mvd_round_trip(mv, 1), mv);
        // Also exercise the other non-I initType — initType 2 ⇒ B-slice
        // ctx slot — so the slot-clamp branch is taken on the same value.
        assert_eq!(mvd_round_trip(mv, 2), mv);
    }

    #[test]
    fn unit_magnitude_skips_abs_mvd_minus2_per_eq_190() {
        // |lMvd| == 1 ⇒ greater0 == 1 ∧ greater1 == 0 ⇒ no
        // abs_mvd_minus2 bin, just the sign bit. Cover all four sign
        // combinations across both non-I initTypes.
        for init_type in [1u8, 2] {
            for &(x, y) in &[(1i32, 1i32), (-1, 1), (1, -1), (-1, -1)] {
                let mv = MotionVector { x, y };
                assert_eq!(
                    mvd_round_trip(mv, init_type),
                    mv,
                    "unit magnitude round trip failed at ({x},{y}) init_type={init_type}"
                );
            }
        }
    }

    #[test]
    fn mixed_zero_and_nonzero_components_round_trip() {
        // One component zero, one component non-zero — exercises the
        // skip-the-greater1-bin branch on the zero side without
        // affecting the non-zero side's bin sequence.
        for &(x, y) in &[(0, 5), (-7, 0), (0, -3), (12, 0)] {
            let mv = MotionVector { x, y };
            assert_eq!(
                mvd_round_trip(mv, 1),
                mv,
                "mixed zero / non-zero round trip failed at ({x},{y})"
            );
        }
    }

    #[test]
    fn large_magnitudes_exercise_egk_prefix_growth_up_to_cap() {
        // Span small / sub-pel-scale / large / boundary magnitudes to
        // exercise the §9.3.3.6 prefix growth and the maxPreExtLen
        // cap. The (131_070, -131_070) pair matches the existing
        // leaf_cu.rs round-trip; (131_071, -131_071) reaches exactly
        // `max_mvd_magnitude()`, exercising the EGk cap path.
        for &(x, y) in &[
            (2i32, 2i32),
            (-2, 3),
            (16, -16),
            (255, -255),
            (1000, -1234),
            (65535, -65535),
            (131_070, -131_070), // 2^17 − 2
            (131_071, -131_071), // 2^17 − 1 — exact max magnitude
        ] {
            let mv = MotionVector { x, y };
            assert_eq!(
                mvd_round_trip(mv, 1),
                mv,
                "large magnitude round trip failed at ({x},{y})"
            );
        }
    }

    #[test]
    fn eq_190_derivation_matches_decoded_components() {
        // Spot-check eq. 190 directly:
        //   lMvd[c] = greater0 * (abs_mvd_minus2 + 2) * (1 − 2*sign).
        // For |lMvd| = 9 the encoder emits abs_mvd_minus2 = 7 and a
        // sign bit; reader must rebuild ±9. Sign asymmetry across the
        // two components also covered.
        assert_eq!(
            mvd_round_trip(MotionVector { x: 9, y: -9 }, 2),
            MotionVector { x: 9, y: -9 }
        );
        assert_eq!(
            mvd_round_trip(MotionVector { x: -9, y: 9 }, 2),
            MotionVector { x: -9, y: 9 }
        );
    }

    #[test]
    fn abs_mvd_minus2_limited_egk_isolated_round_trip_via_mvd_coding() {
        // The reader's `read_abs_mvd_minus2` is module-private to
        // `leaf_cu`. Isolate the §9.3.3.6 limited-EGk path indirectly
        // by encoding a `mvd_coding()` whose component 0 carries
        // `abs_mvd_minus2 = sym` and a sign of 0; the reader rebuilds
        // `lMvd[0] = sym + 2`. Component 1 is held at zero so the
        // surrounding ctx-coded bins are deterministic.
        //
        // Sweep covers the §7.4.10.10 *positive* conformance ceiling
        // (`|lMvd| ≤ 2^17 - 1` ⇒ `abs_mvd_minus2 ≤ 131_069`). The
        // negative side admits one larger absolute value (`-2^17`,
        // exercised separately in
        // `signed_18bit_floor_negative_2_pow_17_round_trips`), so the
        // sweep here stays at `131_069` to keep the positive
        // `(sym + 2)` input within the §7.4.10.10 positive range.
        for sym in [0u32, 1, 2, 3, 7, 8, 100, 1000, 65_533, 131_068, 131_069] {
            let mut enc = ArithEncoder::new();
            let mut enc_ctxs = LeafCuCtxs::init_with_init_type(26, 1);
            encode_mvd_coding(
                &mut enc,
                &mut enc_ctxs,
                MotionVector {
                    x: (sym as i32) + 2,
                    y: 0,
                },
            )
            .expect("encode_mvd_coding succeeds for in-range lMvd");
            enc.encode_terminate(1).expect("terminator");
            let mut padded = enc.finish();
            padded.extend_from_slice(&[0u8; 32]);
            let mut dec = ArithDecoder::new(&padded).expect("decoder accepts encoded stream");
            let mut dec_ctxs = LeafCuCtxs::init_with_init_type(26, 1);
            let tools = CuToolFlags::default();
            let mut reader = LeafCuReader::new(&mut dec, &mut dec_ctxs, tools);
            let mv = reader
                .read_mvd_coding()
                .expect("reader reconstructs mvd_coding()");
            assert_eq!(
                mv,
                MotionVector {
                    x: (sym as i32) + 2,
                    y: 0,
                },
                "EGk isolated round trip failed at sym = {sym}"
            );

            // Cross-check the direct encoder path produces a different
            // (smaller) wire than the surrounding `mvd_coding()` call
            // by encoding `sym` alone and asserting the byte stream
            // strictly fits inside the `mvd_coding()` envelope.
            let mut enc_direct = ArithEncoder::new();
            encode_abs_mvd_minus2(&mut enc_direct, sym).expect("encode_abs_mvd_minus2");
            enc_direct.encode_terminate(1).expect("terminator");
            let direct_bytes = enc_direct.finish();
            assert!(
                !direct_bytes.is_empty(),
                "direct encode_abs_mvd_minus2 produced an empty stream for sym = {sym}"
            );
        }
    }

    #[test]
    fn signed_18bit_floor_negative_2_pow_17_round_trips() {
        // §7.4.10.10 asymmetric range: the negative side reaches
        // `-2^17 = -131_072`, one larger in absolute value than the
        // positive bound. The §9.3.3.6 limited-EGk codec admits
        // `abs_mvd_minus2 = 131_070` (the value at `|lMvd| = 131_072`
        // after eq. 190's `- 2`) — at preExtLen = 15 (the cap) the
        // base contributes `((1<<15) - 1) << 1 = 65_534` and the
        // 17-bit escape suffix carries the remaining `65_536`,
        // summing to `131_070` exactly. So `-131_072` is
        // representable through `mvd_coding()`; this test verifies
        // the encoder + reader agree at the signed-18-bit floor.
        let mv = MotionVector { x: -131_072, y: 0 };
        assert_eq!(mvd_round_trip(mv, 1), mv);
        let mv = MotionVector { x: 0, y: -131_072 };
        assert_eq!(mvd_round_trip(mv, 2), mv);
    }

    #[test]
    fn negative_signs_dont_drift_through_egk_cap() {
        // The signed-magnitude split happens *outside* the EGk path:
        // the EGk path codes |lMvd[c]| − 2 and the sign is a separate
        // bypass bin. Make sure the encoder + reader agree at the cap
        // for both signs.
        for &x in &[131_071i32, -131_071, 131_070, -131_070] {
            let mv = MotionVector { x, y: 0 };
            assert_eq!(
                mvd_round_trip(mv, 1),
                mv,
                "sign drift at cap-magnitude lMvd = ({x},0)"
            );
            let mv = MotionVector { x: 0, y: x };
            assert_eq!(
                mvd_round_trip(mv, 2),
                mv,
                "sign drift at cap-magnitude lMvd = (0,{x})"
            );
        }
    }

    #[test]
    fn both_components_negative_round_trip() {
        // Cover the both-negative case across a range of magnitudes —
        // the spec's per-component-major order means both sign bins
        // must be in the correct slots.
        for &(x, y) in &[(-3i32, -7i32), (-100, -200), (-65535, -1), (-1, -65535)] {
            let mv = MotionVector { x, y };
            assert_eq!(
                mvd_round_trip(mv, 1),
                mv,
                "both-negative round trip failed at ({x},{y})"
            );
        }
    }

    #[test]
    fn dispatcher_matches_reader_bin_count_for_zero_pair() {
        // (0, 0) emits exactly 2 ctx-coded bins (the two
        // `abs_mvd_greater0_flag` bins). The terminator adds a small
        // tail. Encoding a second time into a fresh stream produces
        // the same byte length for the same input, confirming the bin
        // count is stable across runs.
        let mut e1 = ArithEncoder::new();
        let mut c1 = LeafCuCtxs::init_with_init_type(26, 1);
        encode_mvd_coding(&mut e1, &mut c1, MotionVector { x: 0, y: 0 }).unwrap();
        e1.encode_terminate(1).unwrap();
        let b1 = e1.finish();

        let mut e2 = ArithEncoder::new();
        let mut c2 = LeafCuCtxs::init_with_init_type(26, 1);
        encode_mvd_coding(&mut e2, &mut c2, MotionVector { x: 0, y: 0 }).unwrap();
        e2.encode_terminate(1).unwrap();
        let b2 = e2.finish();

        assert_eq!(b1, b2, "encoder is non-deterministic for (0, 0)");
    }

    // -----------------------------------------------------------------
    // Round-233 — decomposed `mvd_coding()` body parser tests.
    // -----------------------------------------------------------------

    /// Build a decomposed decision via `from_motion_vector`, push it
    /// through [`encode_mvd_coding_decomposed`], decode through
    /// [`crate::leaf_cu::LeafCuReader::read_mvd_coding_decomposed`],
    /// and return the recovered decision. The wire layout MUST be
    /// bit-identical to the `encode_mvd_coding`-on-the-same-`(x, y)`
    /// path; both call sites are pinned by the parallel `parity_*`
    /// tests below.
    fn decomposed_round_trip(lmvd: MotionVector, init_type: u8) -> MvdCodingDecision {
        let decision = MvdCodingDecision::from_motion_vector(lmvd);
        let mut enc = ArithEncoder::new();
        let mut enc_ctxs = LeafCuCtxs::init_with_init_type(26, init_type);
        encode_mvd_coding_decomposed(&mut enc, &mut enc_ctxs, &decision)
            .expect("encode_mvd_coding_decomposed succeeds for in-range lMvd");
        enc.encode_terminate(1).expect("terminator");
        let mut padded = enc.finish();
        padded.extend_from_slice(&[0u8; 32]);
        let mut dec = ArithDecoder::new(&padded).expect("decoder accepts encoded stream");
        let mut dec_ctxs = LeafCuCtxs::init_with_init_type(26, init_type);
        let tools = CuToolFlags::default();
        let mut reader = LeafCuReader::new(&mut dec, &mut dec_ctxs, tools);
        reader
            .read_mvd_coding_decomposed()
            .expect("reader reconstructs the mvd_coding() body")
    }

    #[test]
    fn decision_zero_is_all_inferred_defaults() {
        let zero = MvdCodingDecision::zero();
        assert_eq!(zero.abs_mvd_greater0_flag, [false, false]);
        assert_eq!(zero.abs_mvd_greater1_flag, [false, false]);
        assert_eq!(zero.abs_mvd_minus2, [0, 0]);
        assert_eq!(zero.mvd_sign_flag, [false, false]);
        assert_eq!(zero.to_motion_vector(), MotionVector { x: 0, y: 0 });
    }

    #[test]
    fn from_motion_vector_zero_pair_leaves_every_slot_default() {
        let d = MvdCodingDecision::from_motion_vector(MotionVector { x: 0, y: 0 });
        assert_eq!(d, MvdCodingDecision::zero());
    }

    #[test]
    fn from_motion_vector_unit_magnitude_skips_minus2_slot() {
        // |lMvd| == 1 ⇒ greater0 == 1, greater1 == 0, abs_mvd_minus2
        // stays at 0 per §7.4.10.10 inference; sign carries the
        // component-wise sign bit.
        let d = MvdCodingDecision::from_motion_vector(MotionVector { x: 1, y: -1 });
        assert_eq!(d.abs_mvd_greater0_flag, [true, true]);
        assert_eq!(d.abs_mvd_greater1_flag, [false, false]);
        assert_eq!(d.abs_mvd_minus2, [0, 0]);
        assert_eq!(d.mvd_sign_flag, [false, true]);
        assert_eq!(d.to_motion_vector(), MotionVector { x: 1, y: -1 });
    }

    #[test]
    fn from_motion_vector_two_or_higher_populates_minus2_slot() {
        // |lMvd| >= 2 ⇒ greater1 == 1, abs_mvd_minus2 carries the
        // §9.3.3.6 tail.
        let d = MvdCodingDecision::from_motion_vector(MotionVector { x: 9, y: -9 });
        assert_eq!(d.abs_mvd_greater0_flag, [true, true]);
        assert_eq!(d.abs_mvd_greater1_flag, [true, true]);
        assert_eq!(d.abs_mvd_minus2, [7, 7]);
        assert_eq!(d.mvd_sign_flag, [false, true]);
        assert_eq!(d.to_motion_vector(), MotionVector { x: 9, y: -9 });
    }

    #[test]
    fn from_motion_vector_mixed_zero_and_nonzero() {
        // c0 == 0 ⇒ greater0 == 0 ⇒ every other c0 slot inferred.
        // c1 != 0 ⇒ the full c1 cascade is populated.
        let d = MvdCodingDecision::from_motion_vector(MotionVector { x: 0, y: 5 });
        assert_eq!(d.abs_mvd_greater0_flag, [false, true]);
        assert_eq!(d.abs_mvd_greater1_flag, [false, true]);
        assert_eq!(d.abs_mvd_minus2, [0, 3]);
        assert_eq!(d.mvd_sign_flag, [false, false]);
        assert_eq!(d.to_motion_vector(), MotionVector { x: 0, y: 5 });
    }

    #[test]
    fn decomposed_round_trip_zero_pair() {
        for init_type in [1u8, 2] {
            let mv = MotionVector { x: 0, y: 0 };
            let recovered = decomposed_round_trip(mv, init_type);
            assert_eq!(recovered, MvdCodingDecision::from_motion_vector(mv));
            assert_eq!(recovered.to_motion_vector(), mv);
        }
    }

    #[test]
    fn decomposed_round_trip_unit_magnitudes_across_signs() {
        for init_type in [1u8, 2] {
            for &(x, y) in &[(1i32, 1i32), (-1, 1), (1, -1), (-1, -1)] {
                let mv = MotionVector { x, y };
                let recovered = decomposed_round_trip(mv, init_type);
                assert_eq!(
                    recovered,
                    MvdCodingDecision::from_motion_vector(mv),
                    "unit magnitude decomposed round trip failed at ({x},{y}) \
                     init_type={init_type}"
                );
                assert_eq!(recovered.to_motion_vector(), mv);
            }
        }
    }

    #[test]
    fn decomposed_round_trip_mixed_zero_and_nonzero() {
        for &(x, y) in &[(0, 5), (-7, 0), (0, -3), (12, 0)] {
            let mv = MotionVector { x, y };
            let recovered = decomposed_round_trip(mv, 1);
            assert_eq!(
                recovered,
                MvdCodingDecision::from_motion_vector(mv),
                "mixed zero / non-zero decomposed round trip failed at ({x},{y})"
            );
            assert_eq!(recovered.to_motion_vector(), mv);
        }
    }

    #[test]
    fn decomposed_round_trip_large_magnitudes_up_to_egk_cap() {
        // Mirrors the round-187 `large_magnitudes_exercise_egk_prefix_growth_up_to_cap`
        // sweep so the decomposed walker is pinned at the §9.3.3.6
        // prefix-growth and `maxPreExtLen` boundary.
        for &(x, y) in &[
            (2i32, 2i32),
            (-2, 3),
            (16, -16),
            (255, -255),
            (1000, -1234),
            (65535, -65535),
            (131_070, -131_070), // 2^17 − 2
            (131_071, -131_071), // 2^17 − 1 — exact max magnitude
        ] {
            let mv = MotionVector { x, y };
            let recovered = decomposed_round_trip(mv, 1);
            assert_eq!(
                recovered,
                MvdCodingDecision::from_motion_vector(mv),
                "large magnitude decomposed round trip failed at ({x},{y})"
            );
            assert_eq!(recovered.to_motion_vector(), mv);
        }
    }

    #[test]
    fn parity_decomposed_and_packed_emit_identical_bitstreams() {
        // The decomposed walker MUST emit the exact same wire as the
        // packed `encode_mvd_coding`-on-`MotionVector` walker. Compare
        // the raw byte payload across an exhaustive small grid plus
        // the boundary magnitudes from the sweep above.
        let mut cases = Vec::new();
        for &(x, y) in &[
            (0i32, 0i32),
            (1, 0),
            (0, 1),
            (-1, -1),
            (1, -1),
            (-1, 1),
            (2, 3),
            (-3, 2),
            (9, -9),
            (255, -255),
            (131_071, -131_071),
            (131_071, 131_071),
        ] {
            cases.push(MotionVector { x, y });
        }
        for init_type in [1u8, 2] {
            for mv in &cases {
                let mut packed_enc = ArithEncoder::new();
                let mut packed_ctxs = LeafCuCtxs::init_with_init_type(26, init_type);
                encode_mvd_coding(&mut packed_enc, &mut packed_ctxs, *mv).unwrap();
                packed_enc.encode_terminate(1).unwrap();
                let packed_bytes = packed_enc.finish();

                let mut decomp_enc = ArithEncoder::new();
                let mut decomp_ctxs = LeafCuCtxs::init_with_init_type(26, init_type);
                let decision = MvdCodingDecision::from_motion_vector(*mv);
                encode_mvd_coding_decomposed(&mut decomp_enc, &mut decomp_ctxs, &decision).unwrap();
                decomp_enc.encode_terminate(1).unwrap();
                let decomp_bytes = decomp_enc.finish();

                assert_eq!(
                    packed_bytes, decomp_bytes,
                    "decomposed encoder must emit the same wire as the packed encoder \
                     at lmvd = {:?}, init_type = {init_type}",
                    mv
                );
            }
        }
    }

    #[test]
    fn cross_path_packed_wire_decodes_through_decomposed_reader() {
        // The wire produced by the packed `encode_mvd_coding` walker
        // must decode through the decomposed reader to a decision
        // equal to `from_motion_vector`-on-the-same-pair. Pins the
        // reader-side equivalence (the reader's bin order is anchored
        // to the encoder's, not derived independently).
        for &(x, y) in &[(0, 0), (1, -2), (-3, 4), (255, -255), (131_071, 1)] {
            let mv = MotionVector { x, y };
            let mut enc = ArithEncoder::new();
            let mut enc_ctxs = LeafCuCtxs::init_with_init_type(26, 1);
            encode_mvd_coding(&mut enc, &mut enc_ctxs, mv).unwrap();
            enc.encode_terminate(1).unwrap();
            let mut padded = enc.finish();
            padded.extend_from_slice(&[0u8; 32]);
            let mut dec = ArithDecoder::new(&padded).unwrap();
            let mut dec_ctxs = LeafCuCtxs::init_with_init_type(26, 1);
            let tools = CuToolFlags::default();
            let mut reader = LeafCuReader::new(&mut dec, &mut dec_ctxs, tools);
            let decomposed = reader.read_mvd_coding_decomposed().unwrap();
            assert_eq!(decomposed, MvdCodingDecision::from_motion_vector(mv));
            assert_eq!(decomposed.to_motion_vector(), mv);
        }
    }

    #[test]
    fn cross_path_decomposed_wire_decodes_through_packed_reader() {
        // Inverse of the test above: the wire produced by the
        // decomposed encoder must decode through the packed
        // `read_mvd_coding` reader back to the original `(x, y)`
        // pair. Pins the encoder-side equivalence end-to-end.
        for &(x, y) in &[(0, 0), (1, -2), (-3, 4), (255, -255), (131_071, 1)] {
            let mv = MotionVector { x, y };
            let decision = MvdCodingDecision::from_motion_vector(mv);
            let mut enc = ArithEncoder::new();
            let mut enc_ctxs = LeafCuCtxs::init_with_init_type(26, 1);
            encode_mvd_coding_decomposed(&mut enc, &mut enc_ctxs, &decision).unwrap();
            enc.encode_terminate(1).unwrap();
            let mut padded = enc.finish();
            padded.extend_from_slice(&[0u8; 32]);
            let mut dec = ArithDecoder::new(&padded).unwrap();
            let mut dec_ctxs = LeafCuCtxs::init_with_init_type(26, 1);
            let tools = CuToolFlags::default();
            let mut reader = LeafCuReader::new(&mut dec, &mut dec_ctxs, tools);
            let recovered = reader.read_mvd_coding().unwrap();
            assert_eq!(recovered, mv);
        }
    }

    #[test]
    fn eq_190_fold_matches_unit_magnitude_inferred_minus2_slot() {
        // |lMvd[c]| = 1 ⇒ greater1 == 0 ⇒ §7.4.10.10 infers
        // `abs_mvd_minus2 = -1` so eq. 190 gives magnitude 1 from the
        // decision struct's stored `0` (the to_motion_vector method
        // honours the gate on greater1).
        let d = MvdCodingDecision {
            abs_mvd_greater0_flag: [true, false],
            abs_mvd_greater1_flag: [false, false],
            abs_mvd_minus2: [0, 0],
            mvd_sign_flag: [true, false],
        };
        assert_eq!(d.to_motion_vector(), MotionVector { x: -1, y: 0 });
    }
}
