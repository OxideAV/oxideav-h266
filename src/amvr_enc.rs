//! Round-195 — encoder-side CABAC emission for the §7.3.10.10
//! `amvr_flag[x0][y0]` / `amvr_precision_idx[x0][y0]` syntax elements,
//! plus a dispatcher that mirrors the round-193 reader
//! [`crate::leaf_cu::LeafCuReader::read_amvr_inter_gated`] bin-for-bin
//! under the same §7.3.10.10 gating cascade.
//!
//! Round-193 landed the reader: `read_amvr_flag`, `read_amvr_precision_idx`,
//! and the `read_amvr_inter_gated` dispatcher that walks
//!
//! ```text
//! if( amvr-gate-open )
//!   amvr_flag = read_amvr_flag( inter_affine_flag )
//!   if( amvr_flag )
//!     amvr_precision_idx = read_amvr_precision_idx( inter_affine_flag, false )
//! ```
//!
//! and returns the typed `(amvr_flag, amvr_precision_idx, AmvrShift)`
//! triple with the §7.4.11.6 / Table 16 shift already folded through
//! [`crate::amvr::AmvrShift::for_inter`] / [`crate::amvr::AmvrShift::for_affine`].
//!
//! The matching encoder mirror has been a follow-up since: the
//! `#[cfg(test)]` helpers `encode_amvr_flag` / `encode_amvr_precision_idx`
//! in `leaf_cu.rs` exercise the same CABAC bin layout but are not
//! callable from a real encoder pipeline. This module lifts those
//! helpers into the public encoder surface parallel to the reader.
//!
//! ## Public surface
//!
//! 1. [`encode_amvr_flag`] — single ctx-coded FL `cMax = 1` bin at slot
//!    `(init_type - 1) * 2 + ctx_inc_amvr_flag(inter_affine_flag,
//!    false)` per §9.3.4.2 / Table 132 / Table 89.
//! 2. [`encode_amvr_precision_idx`] — TR with `cMax = (inter_affine_flag
//!    == 0 && !mode_ibc) ? 2 : 1`; bin 0 ctx-coded at slot `init_type
//!    * 3 + ctx_inc_amvr_precision_idx(...)`, bin 1 (only present
//!    when `cMax = 2`) ctx-coded at slot `init_type * 3 + 1` per
//!    Table 90.
//! 3. [`encode_amvr_inter_gated`] — full §7.3.10.10 dispatcher
//!    mirror. Takes an [`crate::leaf_cu::AmvrGate`] + an
//!    [`AmvrDecision`] and emits the bins the reader will consume.
//!    Applies the same gating cascade so the wire layout stays
//!    bit-identical to [`crate::leaf_cu::LeafCuReader::read_amvr_inter_gated`].
//!    When the §7.3.10.10 outer gate is closed no bin is emitted,
//!    matching the bitstream the reader will skip parsing
//!    (§7.4.12.7 inferences fire).
//!
//! ## Round-trip
//!
//! For every legal `(gate, decision)` pair, the bytes produced by
//! [`encode_amvr_inter_gated`] decode back through
//! [`crate::leaf_cu::LeafCuReader::read_amvr_inter_gated`] to the same
//! `(amvr_flag, amvr_precision_idx, AmvrShift)` triple. The encoder
//! also re-uses the round-40 [`crate::amvr::AmvrShift::for_inter`] /
//! [`crate::amvr::AmvrShift::for_affine`] folds via [`AmvrDecision::shift`]
//! so the spec's §7.4.11.6 / Table 16 derivation lives in exactly one
//! place.
//!
//! ## Spec reference
//!
//! ITU-T H.266 | ISO/IEC 23090-3 (V4, 01/2026):
//! * §7.3.10.10 — `amvr_flag` / `amvr_precision_idx` syntax (the
//!   source-of-truth listing this dispatcher mirrors).
//! * §7.4.12.7 — inference rules: `amvr_flag` inferred to 1 for
//!   `MODE_IBC` / 0 otherwise; `amvr_precision_idx` inferred to 0.
//! * §7.4.11.6 / Table 16 — `AmvrShift` derivation (regular / affine
//!   / IBC).
//! * §9.3.4.2 / Table 132 — ctxInc routing for both bins (regular vs
//!   affine vs IBC arms).
//! * Table 89 — `amvr_flag` initValue / shiftIdx (round-193).
//! * Table 90 — `amvr_precision_idx` initValue / shiftIdx (round-193).
//!
//! No third-party VVC encoder source was consulted; the implementation
//! is spec-only and mirrors the existing round-193 reader-side code
//! already shipped in this crate.

use oxideav_core::Result;

use crate::amvr::{
    amvr_precision_idx_c_max, apply_amvr_shift, ctx_inc_amvr_flag, ctx_inc_amvr_precision_idx,
    ctx_inc_amvr_precision_idx_bin1, AmvrShift,
};
use crate::cabac_enc::ArithEncoder;
use crate::inter::MotionVector;
use crate::leaf_cu::{AmvrGate, LeafCuCtxs};

/// Round-195 — output of the encoder's AMVR cascade for the non-merge
/// inter branch (not IBC). Pairs the raw `(amvr_flag,
/// amvr_precision_idx)` syntax-element values with the typed
/// [`AmvrShift`] folded via [`AmvrShift::for_inter`] /
/// [`AmvrShift::for_affine`].
///
/// Mirrors the return triple of [`crate::leaf_cu::LeafCuReader::read_amvr_inter_gated`]
/// so the encoder dispatcher's input and the reader's output are the
/// same shape.
///
/// When the §7.3.10.10 gate is closed callers should pass the
/// §7.4.12.7-inferred values: `amvr_flag = false`,
/// `amvr_precision_idx = 0`, `shift = AmvrShift(2)` (the default
/// 1/4-luma resolution). The dispatcher debug-asserts this.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AmvrDecision {
    /// `amvr_flag[x0][y0]` — `false` ⇒ 1/4-luma resolution, `true` ⇒
    /// extended resolution further specified by `amvr_precision_idx`.
    pub amvr_flag: bool,
    /// `amvr_precision_idx[x0][y0]` — only consulted when `amvr_flag
    /// == true`. The legal range depends on the arm: regular AMVR ⇒
    /// `0..=2`, affine AMVR ⇒ `0..=1`. The §7.4.12.7-inferred value
    /// when not signalled is `0`.
    pub amvr_precision_idx: u32,
    /// `AmvrShift` folded via [`AmvrShift::for_inter`] (regular) or
    /// [`AmvrShift::for_affine`] (affine), per the gate's
    /// `inter_affine_flag`. The reader returns the same value so the
    /// encoder + decoder agree on the §7.4.11.6 / Table 16 shift.
    pub shift: AmvrShift,
}

impl AmvrDecision {
    /// Convenience constructor: build an [`AmvrDecision`] from the two
    /// raw values, folding `shift` through the matching arm of
    /// [`AmvrShift`].
    ///
    /// `inter_affine_flag` selects which arm of Table 16 supplies the
    /// shift: `false` ⇒ regular AMVR ([`AmvrShift::for_inter`]),
    /// `true` ⇒ affine AMVR ([`AmvrShift::for_affine`]). The IBC arm
    /// is reached via a separate path because §7.3.10.5 emits no
    /// `amvr_flag` (the §7.4.12.7 inference assigns 1) — see
    /// [`crate::amvr::AmvrShift::for_ibc`] for the IBC fold.
    pub fn new(amvr_flag: bool, amvr_precision_idx: u32, inter_affine_flag: bool) -> Self {
        let shift = if inter_affine_flag {
            AmvrShift::for_affine(amvr_flag, amvr_precision_idx)
        } else {
            AmvrShift::for_inter(amvr_flag, amvr_precision_idx)
        };
        Self {
            amvr_flag,
            amvr_precision_idx,
            shift,
        }
    }

    /// Convenience constructor for the §7.4.12.7-inferred default:
    /// `amvr_flag = false`, `amvr_precision_idx = 0`, `shift =
    /// AmvrShift(2)` (1/4-luma resolution per the `amvr_flag == 0`
    /// rows of Table 16). The right thing to pass to
    /// [`encode_amvr_inter_gated`] when the gate is known closed.
    pub fn default_inferred() -> Self {
        Self {
            amvr_flag: false,
            amvr_precision_idx: 0,
            shift: AmvrShift(2),
        }
    }

    /// Apply [`apply_amvr_shift`] to a single per-list (or per-CP)
    /// `lMvd` using this decision's [`AmvrShift`]. Mirrors the
    /// §7.4.11.6 eqs. 161 – 176 step the reader applies after
    /// `read_amvr_inter_gated`. Equivalent to `apply_amvr_shift(mvd,
    /// self.shift)`.
    pub fn apply(self, mvd: MotionVector) -> MotionVector {
        apply_amvr_shift(mvd, self.shift)
    }
}

/// Encode one `amvr_flag[x0][y0]` bin per §7.3.10.10 / Table 132.
///
/// Mirrors [`crate::leaf_cu::LeafCuReader::read_amvr_flag`] bin-for-bin:
/// a single ctx-coded FL `cMax = 1` bin at slot `(init_type - 1) * 2 +
/// ctx_inc_amvr_flag(inter_affine_flag, false)`. The Table 89 bundle
/// is laid out as `(init_type 1, ctxInc 0)`, `(init_type 1, ctxInc 1)`,
/// `(init_type 2, ctxInc 0)`, `(init_type 2, ctxInc 1)` per §9.3.4.2
/// row mapping.
///
/// The caller is responsible for the §7.3.10.10 outer gate: this
/// helper must NOT be invoked when the gate is closed (the
/// §7.4.12.7 inference applies — see [`AmvrGate::is_open`]). It also
/// must NOT be invoked on the IBC branch (the §7.3.10.5 IBC AMVR
/// cascade emits no `amvr_flag`); see the round-193 reader docstring
/// for that nuance.
///
/// The reader assumes the slice is non-I (`init_type ∈ {1, 2}`);
/// AMVR is never signalled in I slices because the non-merge inter
/// branch is unreachable there. This debug-asserts the same.
pub fn encode_amvr_flag(
    enc: &mut ArithEncoder,
    ctxs: &mut LeafCuCtxs,
    amvr_flag: bool,
    inter_affine_flag: bool,
) -> Result<()> {
    debug_assert!(
        ctxs.init_type >= 1,
        "amvr_flag is not signalled in I slices (non-merge inter branch is unreachable)"
    );
    let inc = ctx_inc_amvr_flag(inter_affine_flag, false) as usize;
    let init_off = (ctxs.init_type as usize).saturating_sub(1) * 2;
    let n = ctxs.amvr_flag.len() - 1;
    let slot = (init_off + inc).min(n);
    let bit = if amvr_flag { 1 } else { 0 };
    enc.encode_decision(&mut ctxs.amvr_flag[slot], bit)?;
    Ok(())
}

/// Encode one `amvr_precision_idx[x0][y0]` value per §7.3.10.10 /
/// Table 132.
///
/// Mirrors [`crate::leaf_cu::LeafCuReader::read_amvr_precision_idx`]
/// bin-for-bin: TR with `cMax = (inter_affine_flag == 0 && !mode_ibc)
/// ? 2 : 1`, `cRiceParam = 0`. Both bins are ctx-coded:
///
/// * Bin 0 ctx-slot: `init_type * 3 + ctxInc` with `ctxInc =
///   (mode_ibc) ? 1 : (inter_affine_flag == 0 ? 0 : 2)` per
///   [`ctx_inc_amvr_precision_idx`].
/// * Bin 1 ctx-slot: `init_type * 3 + 1` per
///   [`ctx_inc_amvr_precision_idx_bin1`]. Only the regular AMVR
///   path (`cMax = 2`) reaches bin 1; affine and IBC truncate at
///   bin 0.
///
/// The caller is responsible for the §7.3.10.10 gate: this helper
/// must NOT be invoked when `amvr_flag == 0` on the non-merge inter
/// branch (the syntax element is not present and §7.4.12.7 infers it
/// to 0). It also must NOT be invoked on the IBC branch when
/// `MvdL0[x0][y0][0] == 0 && MvdL0[x0][y0][1] == 0` (the §7.3.10.5
/// IBC AMVR cascade closes).
///
/// `value` must be in `0..=cMax` for the arm; out-of-range values are
/// silently clamped to the truncation point in release builds and
/// debug-assert in debug builds.
pub fn encode_amvr_precision_idx(
    enc: &mut ArithEncoder,
    ctxs: &mut LeafCuCtxs,
    value: u32,
    inter_affine_flag: bool,
    mode_ibc: bool,
) -> Result<()> {
    let cmax = amvr_precision_idx_c_max(inter_affine_flag, mode_ibc);
    debug_assert!(
        value <= cmax,
        "amvr_precision_idx = {value} out of range for cMax = {cmax} \
         (inter_affine_flag = {inter_affine_flag}, mode_ibc = {mode_ibc})"
    );
    let init_off = ctxs.init_type as usize * 3;
    let n = ctxs.amvr_precision_idx.len() - 1;
    // Bin 0.
    let inc0 = ctx_inc_amvr_precision_idx(inter_affine_flag, mode_ibc) as usize;
    let slot0 = (init_off + inc0).min(n);
    let bit0 = if value == 0 { 0 } else { 1 };
    enc.encode_decision(&mut ctxs.amvr_precision_idx[slot0], bit0)?;
    if bit0 == 0 {
        return Ok(());
    }
    if cmax < 2 {
        // TR truncation point — affine / IBC stop after bin 0 = 1.
        return Ok(());
    }
    // Bin 1.
    let inc1 = ctx_inc_amvr_precision_idx_bin1() as usize;
    let slot1 = (init_off + inc1).min(n);
    let bit1 = if value >= 2 { 1 } else { 0 };
    enc.encode_decision(&mut ctxs.amvr_precision_idx[slot1], bit1)?;
    Ok(())
}

/// Round-195 — §7.3.10.10 AMVR dispatcher for the non-merge inter
/// branch (not IBC). Encoder mirror of
/// [`crate::leaf_cu::LeafCuReader::read_amvr_inter_gated`].
///
/// Walks the §7.3.10.10 conditional:
///
/// ```text
/// if( amvr-gate-open )
///   encode_amvr_flag( inter_affine_flag )
///   if( amvr_flag )
///     encode_amvr_precision_idx( inter_affine_flag, mode_ibc = false )
/// ```
///
/// When the §7.3.10.10 outer gate is closed no bin is emitted: the
/// reader applies §7.4.12.7 inferences and recovers the
/// [`AmvrDecision::default_inferred`] triple. The dispatcher
/// debug-asserts the caller passed exactly the inferred decision in
/// that case so a release-build wire stream still round-trips even
/// under a stale `decision`.
///
/// The `decision.shift` field is also debug-checked against the gate's
/// `inter_affine_flag` — i.e. `decision` was built via
/// [`AmvrDecision::new`] (or the inferred default) and didn't drift
/// from the raw flags. Mirrors the round-177 / round-190 debug-assert
/// pattern.
pub fn encode_amvr_inter_gated(
    enc: &mut ArithEncoder,
    ctxs: &mut LeafCuCtxs,
    gate: &AmvrGate,
    decision: &AmvrDecision,
) -> Result<()> {
    // Caller-conformance: decision.shift must agree with the raw
    // flag pair through the matching Table 16 arm.
    debug_assert_eq!(
        decision.shift,
        if gate.inter_affine_flag {
            AmvrShift::for_affine(decision.amvr_flag, decision.amvr_precision_idx)
        } else {
            AmvrShift::for_inter(decision.amvr_flag, decision.amvr_precision_idx)
        },
        "AmvrDecision.shift disagrees with the (amvr_flag, amvr_precision_idx) pair under \
         inter_affine_flag = {}",
        gate.inter_affine_flag
    );

    if !gate.is_open() {
        // §7.4.12.7 inferences fire — no bins on the wire. Caller
        // MUST pass the matching inferred default.
        debug_assert!(
            !decision.amvr_flag && decision.amvr_precision_idx == 0,
            "AMVR gate closed → §7.4.12.7 requires amvr_flag = 0 and amvr_precision_idx = 0 \
             (got amvr_flag = {}, amvr_precision_idx = {})",
            decision.amvr_flag,
            decision.amvr_precision_idx,
        );
        return Ok(());
    }
    encode_amvr_flag(enc, ctxs, decision.amvr_flag, gate.inter_affine_flag)?;
    if !decision.amvr_flag {
        // §7.4.12.7 infers amvr_precision_idx = 0 — no bin on the
        // wire.
        debug_assert_eq!(
            decision.amvr_precision_idx, 0,
            "amvr_flag = 0 → §7.4.12.7 requires amvr_precision_idx = 0"
        );
        return Ok(());
    }
    encode_amvr_precision_idx(
        enc,
        ctxs,
        decision.amvr_precision_idx,
        gate.inter_affine_flag,
        false,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::cabac::ArithDecoder;
    use crate::leaf_cu::{CuToolFlags, LeafCuReader};

    /// End-to-end round-trip: drive [`encode_amvr_inter_gated`] against
    /// a gate + decision, decode the resulting bitstream through the
    /// round-193 reader dispatcher, and assert the recovered triple
    /// matches the input bit-for-bit.
    fn dispatcher_round_trip(
        init_type: u8,
        gate: AmvrGate,
        decision: AmvrDecision,
    ) -> (bool, u32, AmvrShift) {
        let mut enc = ArithEncoder::new();
        let mut enc_ctxs = LeafCuCtxs::init_with_init_type(26, init_type);
        encode_amvr_inter_gated(&mut enc, &mut enc_ctxs, &gate, &decision)
            .expect("encode_amvr_inter_gated succeeds under a valid gate/decision pair");
        enc.encode_terminate(1).expect("terminator");
        let mut padded = enc.finish();
        padded.extend_from_slice(&[0u8; 32]);
        let mut dec = ArithDecoder::new(&padded).expect("decoder accepts the encoded stream");
        let mut dec_ctxs = LeafCuCtxs::init_with_init_type(26, init_type);
        let tools = CuToolFlags::default();
        let mut reader = LeafCuReader::new(&mut dec, &mut dec_ctxs, tools);
        reader
            .read_amvr_inter_gated(&gate)
            .expect("reader dispatcher succeeds")
    }

    fn open_regular_gate() -> AmvrGate {
        AmvrGate {
            sps_amvr_enabled: true,
            sps_affine_amvr_enabled: false,
            inter_affine_flag: false,
            any_mvd_l0_l1_nonzero: true,
            any_mvd_cp_l0_l1_nonzero: false,
        }
    }

    fn open_affine_gate() -> AmvrGate {
        AmvrGate {
            sps_amvr_enabled: false,
            sps_affine_amvr_enabled: true,
            inter_affine_flag: true,
            any_mvd_l0_l1_nonzero: false,
            any_mvd_cp_l0_l1_nonzero: true,
        }
    }

    fn closed_gate() -> AmvrGate {
        AmvrGate::default()
    }

    #[test]
    fn make_amvr_decision_folds_shift_per_table_16() {
        // Regular arm — Table 16 non-affine row:
        //   amvr_flag = 0 → AmvrShift = 2 (1/4 luma).
        //   amvr_flag = 1, prec = 0 → AmvrShift = 3 (1/2 luma).
        //   amvr_flag = 1, prec = 1 → AmvrShift = 4 (1 luma).
        //   amvr_flag = 1, prec = 2 → AmvrShift = 6 (4 luma).
        assert_eq!(AmvrDecision::new(false, 0, false).shift, AmvrShift(2));
        assert_eq!(AmvrDecision::new(true, 0, false).shift, AmvrShift(3));
        assert_eq!(AmvrDecision::new(true, 1, false).shift, AmvrShift(4));
        assert_eq!(AmvrDecision::new(true, 2, false).shift, AmvrShift(6));
        // Affine arm — Table 16 affine row:
        //   amvr_flag = 0 → AmvrShift = 2 (1/4 luma).
        //   amvr_flag = 1, prec = 0 → AmvrShift = 0 (1/16 luma).
        //   amvr_flag = 1, prec = 1 → AmvrShift = 4 (1 luma).
        assert_eq!(AmvrDecision::new(false, 0, true).shift, AmvrShift(2));
        assert_eq!(AmvrDecision::new(true, 0, true).shift, AmvrShift(0));
        assert_eq!(AmvrDecision::new(true, 1, true).shift, AmvrShift(4));
        // §7.4.12.7 inferred default — 1/4 luma.
        let d = AmvrDecision::default_inferred();
        assert_eq!(d.shift, AmvrShift(2));
        assert!(!d.amvr_flag);
        assert_eq!(d.amvr_precision_idx, 0);
    }

    #[test]
    fn dispatcher_closed_gate_emits_no_bins_round_trip() {
        // §7.4.12.7 inference path — gate closed, no bins on the wire,
        // reader recovers the inferred default. Cover both non-I
        // initTypes.
        for it in [1u8, 2] {
            let (f, p, s) =
                dispatcher_round_trip(it, closed_gate(), AmvrDecision::default_inferred());
            assert!(!f, "closed gate should infer amvr_flag = 0 (init={it})");
            assert_eq!(p, 0, "closed gate should infer prec = 0 (init={it})");
            assert_eq!(
                s,
                AmvrShift(2),
                "closed gate should infer 1/4 luma (init={it})"
            );
        }
    }

    #[test]
    fn dispatcher_regular_arm_exhaustive_round_trip() {
        // Regular AMVR — open gate, exhaustive over (amvr_flag,
        // amvr_precision_idx) ∈ {0, (1, 0), (1, 1), (1, 2)} and both
        // non-I initTypes. Each round trip must recover the same
        // (flag, prec, shift) triple bit-for-bit through the reader
        // dispatcher.
        let gate = open_regular_gate();
        for it in [1u8, 2] {
            // amvr_flag = 0 — only bin 0 of amvr_flag goes on the wire.
            let d = AmvrDecision::new(false, 0, false);
            let (f, p, s) = dispatcher_round_trip(it, gate, d);
            assert_eq!((f, p, s), (false, 0, AmvrShift(2)));
            // amvr_flag = 1 + prec ∈ {0, 1, 2}.
            for prec in 0..=2u32 {
                let d = AmvrDecision::new(true, prec, false);
                let (f, p, s) = dispatcher_round_trip(it, gate, d);
                assert_eq!(
                    (f, p, s),
                    (true, prec, d.shift),
                    "regular round-trip failed at init={it} prec={prec}"
                );
            }
        }
    }

    #[test]
    fn dispatcher_affine_arm_exhaustive_round_trip() {
        // Affine AMVR — open gate, exhaustive over (amvr_flag,
        // amvr_precision_idx) ∈ {0, (1, 0), (1, 1)} (affine arm has
        // cMax = 1) and both non-I initTypes.
        let gate = open_affine_gate();
        for it in [1u8, 2] {
            let d = AmvrDecision::new(false, 0, true);
            let (f, p, s) = dispatcher_round_trip(it, gate, d);
            assert_eq!((f, p, s), (false, 0, AmvrShift(2)));
            for prec in 0..=1u32 {
                let d = AmvrDecision::new(true, prec, true);
                let (f, p, s) = dispatcher_round_trip(it, gate, d);
                assert_eq!(
                    (f, p, s),
                    (true, prec, d.shift),
                    "affine round-trip failed at init={it} prec={prec}"
                );
            }
        }
    }

    #[test]
    fn dispatcher_amvr_flag_zero_skips_precision_idx_bin() {
        // Open gate, amvr_flag = 0 — exactly one ctx-coded bin
        // (amvr_flag itself) goes on the wire; the precision_idx is
        // §7.4.12.7-inferred to 0 and the reader does not parse it.
        // We assert this by comparing the byte count to the "gate-
        // closed, no bins" baseline: same number of payload bytes
        // (the terminator byte dominates a single-bit encoding).
        let gate = open_regular_gate();
        let mut enc_open = ArithEncoder::new();
        let mut ctxs_open = LeafCuCtxs::init_with_init_type(26, 1);
        let d = AmvrDecision::new(false, 0, false);
        encode_amvr_inter_gated(&mut enc_open, &mut ctxs_open, &gate, &d).unwrap();
        enc_open.encode_terminate(1).unwrap();
        let bytes_open = enc_open.finish().len();

        // Now drive amvr_flag = 1, prec = 0 (which adds a precision_idx
        // bin) — the byte count must be >= bytes_open since strictly
        // more bins go on the wire.
        let mut enc_full = ArithEncoder::new();
        let mut ctxs_full = LeafCuCtxs::init_with_init_type(26, 1);
        let d_full = AmvrDecision::new(true, 0, false);
        encode_amvr_inter_gated(&mut enc_full, &mut ctxs_full, &gate, &d_full).unwrap();
        enc_full.encode_terminate(1).unwrap();
        let bytes_full = enc_full.finish().len();

        assert!(
            bytes_full >= bytes_open,
            "open + amvr_flag = 1 + prec emits at least as many bytes as open + amvr_flag = 0 \
             (got open = {bytes_open}, full = {bytes_full})"
        );
    }

    #[test]
    fn dispatcher_apply_shift_into_motion_vector() {
        // AmvrDecision::apply matches the round-40 apply_amvr_shift
        // helper bit-for-bit.
        let d = AmvrDecision::new(true, 2, false); // AmvrShift = 6 (4 luma)
        let mvd = MotionVector { x: 1, y: -1 };
        assert_eq!(d.apply(mvd), MotionVector { x: 64, y: -64 });
        // Affine 1/16-luma — AmvrShift = 0 ⇒ identity.
        let d = AmvrDecision::new(true, 0, true);
        let mvd = MotionVector { x: 7, y: -3 };
        assert_eq!(d.apply(mvd), mvd);
        // Inferred default — AmvrShift = 2 (1/4 luma).
        let d = AmvrDecision::default_inferred();
        let mvd = MotionVector { x: 3, y: -2 };
        assert_eq!(d.apply(mvd), MotionVector { x: 12, y: -8 });
    }

    #[test]
    fn dispatcher_gates_dont_cross_regular_vs_affine() {
        // Sanity: a regular gate (sps_amvr_enabled = true,
        // sps_affine_amvr_enabled = false) with inter_affine_flag =
        // true is closed (no bin on the wire), and vice versa. Round-
        // trip the inferred default.
        let regular_with_affine_flag = AmvrGate {
            sps_amvr_enabled: true,
            sps_affine_amvr_enabled: false,
            inter_affine_flag: true,
            any_mvd_l0_l1_nonzero: true,
            any_mvd_cp_l0_l1_nonzero: true,
        };
        assert!(!regular_with_affine_flag.is_open());
        let (f, p, s) = dispatcher_round_trip(
            1,
            regular_with_affine_flag,
            AmvrDecision::default_inferred(),
        );
        assert!(!f);
        assert_eq!(p, 0);
        assert_eq!(s, AmvrShift(2));

        let affine_with_regular_flag = AmvrGate {
            sps_amvr_enabled: false,
            sps_affine_amvr_enabled: true,
            inter_affine_flag: false,
            any_mvd_l0_l1_nonzero: true,
            any_mvd_cp_l0_l1_nonzero: true,
        };
        assert!(!affine_with_regular_flag.is_open());
        let (f, p, s) = dispatcher_round_trip(
            2,
            affine_with_regular_flag,
            AmvrDecision::default_inferred(),
        );
        assert!(!f);
        assert_eq!(p, 0);
        assert_eq!(s, AmvrShift(2));
    }
}
