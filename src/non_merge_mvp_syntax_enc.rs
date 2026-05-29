//! Round-183 — encoder-side CABAC emission for the §7.3.11.7 non-merge
//! inter MVP-side syntax (`inter_pred_idc`, `sym_mvd_flag`,
//! `ref_idx_l0` / `ref_idx_l1`, `mvp_l0_flag` / `mvp_l1_flag`), plus a
//! dispatcher that mirrors the existing reader-side readers
//! ([`crate::leaf_cu::LeafCuReader::read_inter_pred_idc`],
//! [`crate::leaf_cu::LeafCuReader::read_sym_mvd_flag`],
//! [`crate::leaf_cu::LeafCuReader::read_ref_idx_lx`],
//! [`crate::leaf_cu::LeafCuReader::read_mvp_lx_flag`]) bin-for-bin
//! under the §7.3.11.7 gating cascade.
//!
//! Companion to round-177's [`crate::affine_syntax_enc`] — that
//! module covers the §7.3.11.7 non-merge inter *affine*-syntax
//! (`inter_affine_flag` / `cu_affine_type_flag`); this module covers
//! the §7.3.11.7 non-merge inter *MVP*-side syntax that runs whether
//! or not the CU is affine. The two encoder modules together provide
//! the complete public encoder surface for the non-merge inter
//! branch's pre-MVD syntax. The reader-side parallels were landed in
//! round 100 (§7.3.10.8 in earlier spec naming) and stayed
//! `#[cfg(test)]`-only on the encoder side until this round.
//!
//! ## Public surface
//!
//! 1. [`encode_inter_pred_idc`] — Table 131 binarisation against the
//!    per-init-type Table 51 slot block. Two-bin form for
//!    `cbWidth + cbHeight > 12` (`PRED_BI = 1`, `PRED_L0 = 00`,
//!    `PRED_L1 = 01`) with bin 0 ctxInc per
//!    [`crate::ctx::ctx_inc_inter_pred_idc_bin0`] and bin 1 ctxInc per
//!    [`crate::ctx::ctx_inc_inter_pred_idc_bin1`]; one-bin form for
//!    the `== 12` block-size case (`PRED_BI` suppressed,
//!    `PRED_L0 = 0` / `PRED_L1 = 1`).
//! 2. [`encode_sym_mvd_flag`] — single ctx-coded FL `cMax = 1` bin
//!    against the Table 86 slot `init_type - 1` with the deterministic
//!    `ctxInc = 0` of Table 132 ([`crate::ctx::ctx_inc_sym_mvd_flag`]).
//! 3. [`encode_ref_idx_lx`] — Table 127 TR (`cMax =
//!    NumRefIdxActive[X] - 1`, `cRiceParam = 0`). Bin 0 ctxInc 0,
//!    bin 1 ctxInc 1 ([`crate::ctx::ctx_inc_ref_idx_lx`]), bins 2..
//!    bypass-coded; ctx slot block `(init_type - 1) * 2`.
//! 4. [`encode_mvp_lx_flag`] — single ctx-coded FL `cMax = 1` bin
//!    against the Table 132 / Table 51 slot `init_type` with the
//!    deterministic `ctxInc = 0`
//!    ([`crate::ctx::ctx_inc_mvp_lx_flag`]).
//! 5. [`encode_non_merge_mvp_syntax`] — full dispatcher: takes a
//!    [`NonMergeMvpSyntaxGate`] + [`NonMergeMvpSyntaxDecision`] and
//!    walks §7.3.11.7 in spec order (`inter_pred_idc` →
//!    `sym_mvd_flag` → per-list `ref_idx_lX` → per-list `mvp_lX_flag`
//!    for active lists). When a §7.4.12.7 inference holds (gate
//!    closed) no bin is emitted, matching the reader's skip
//!    behaviour.
//!
//! ## Spec reference
//!
//! ITU-T H.266 | ISO/IEC 23090-3 (V4, 01/2026):
//! * §7.3.11.7 — `coding_unit()` non-merge inter MVP-side syntax
//!   (the `inter_pred_idc` / `sym_mvd_flag` / `ref_idx_lX` /
//!   `mvp_lX_flag` source-of-truth listing with the per-list active
//!   selectors that this module's dispatcher mirrors).
//! * §7.4.12.7 — inference rules for the omitted flags
//!   (`inter_pred_idc = PRED_L0` when only L0 is active;
//!   `sym_mvd_flag = 0` when its SPS / slice gate is closed;
//!   `ref_idx_lX = 0` when `NumRefIdxActive[X] == 1`;
//!   `mvp_lX_flag = 0` when the list is not active for the CU).
//! * §9.3.4.2.2 / Table 131 / Table 132 — context derivation rules
//!   for all four syntax elements.
//! * Table 50 — Table 51 init-type → ctx-slot block mapping.
//! * Table 86 — `sym_mvd_flag` initValue / shiftIdx.
//! * Table 127 — `ref_idx_lX` initValue / shiftIdx.
//!
//! No third-party VVC encoder source was consulted; the
//! implementation is spec-only and mirrors the existing reader-side
//! code already shipped in this crate.

use oxideav_core::Result;

use crate::cabac_enc::ArithEncoder;
use crate::ctx::{
    ctx_inc_inter_pred_idc_bin0, ctx_inc_inter_pred_idc_bin1, ctx_inc_mvp_lx_flag,
    ctx_inc_ref_idx_lx, ctx_inc_sym_mvd_flag,
};
use crate::leaf_cu::{InterPredDir, LeafCuCtxs};

/// Round-183 — §7.3.11.7 non-merge inter MVP-side syntax gate.
///
/// Bundles the §7.3.11.7 gating conditions for the four MVP-side
/// syntax elements together with the slice-scope state the
/// derivations consult.
///
/// The struct is intentionally pure data: the §7.3.11.7 control flow
/// is in [`encode_non_merge_mvp_syntax`], which reads these fields
/// without touching any other CTU walker state. This keeps the
/// dispatcher a function of `(gate, decision)` only and makes the
/// encoder mirror trivially testable against the matching reader-side
/// readers.
#[derive(Clone, Copy, Debug, Default)]
pub struct NonMergeMvpSyntaxGate {
    /// `cbWidth` of the current CU in luma samples — feeds bin 0
    /// ctxInc for `inter_pred_idc` and the §7.3.11.7
    /// `(cbWidth + cbHeight)` gate that picks the two-bin / one-bin
    /// binarisation form.
    pub cb_width: u32,
    /// `cbHeight` of the current CU in luma samples — pair to
    /// `cb_width`.
    pub cb_height: u32,
    /// `slice_type` interpreted as "is B-slice" — when `false` the CU
    /// is P-coded, L1 is not active, and §7.4.12.7 infers
    /// `inter_pred_idc = PRED_L0` (no bin emitted), `ref_idx_l1 = 0`,
    /// `mvp_l1_flag = 0`, `sym_mvd_flag = 0`.
    pub b_slice: bool,
    /// `sps_smvd_enabled_flag && !mvd_l1_zero_flag && RefList[0] and
    /// RefList[1] each have a short-term ref with the spec's
    /// `RefPicList[0][i].POC > currPic.POC` / `RefPicList[1][j].POC <
    /// currPic.POC` (or the symmetric reverse) arrangement —
    /// whichever the caller has resolved into a single boolean for
    /// this CU. When `false` §7.4.12.7 infers `sym_mvd_flag = 0`.
    pub sym_mvd_gate_open: bool,
    /// `NumRefIdxActive[0]` for the current slice — `ref_idx_l0` has
    /// `cMax = NumRefIdxActive[0] - 1`; when this is `0` or `1` no
    /// `ref_idx_l0` bins are emitted and §7.4.12.7 infers it as 0.
    pub num_ref_idx_active_l0: u32,
    /// `NumRefIdxActive[1]` for the current slice — `ref_idx_l1` has
    /// `cMax = NumRefIdxActive[1] - 1`; when this is `0` or `1` no
    /// `ref_idx_l1` bins are emitted and §7.4.12.7 infers it as 0.
    pub num_ref_idx_active_l1: u32,
}

impl NonMergeMvpSyntaxGate {
    /// `true` iff the §7.3.11.7 `inter_pred_idc` outer gate opens —
    /// i.e. the slice is a B-slice. When `false` §7.4.12.7 infers
    /// `inter_pred_idc = PRED_L0` and the dispatcher emits zero bins
    /// for this element.
    pub fn inter_pred_idc_gate_open(&self) -> bool {
        self.b_slice
    }

    /// `true` iff the §7.3.11.7 two-bin form is selected for
    /// `inter_pred_idc` (`cbWidth + cbHeight > 12`). When `false` the
    /// one-bin form is used (`PRED_BI` suppressed, `PRED_L0 = 0` /
    /// `PRED_L1 = 1`). Only consulted when
    /// [`Self::inter_pred_idc_gate_open`] returns `true`.
    pub fn inter_pred_idc_two_bin_form(&self) -> bool {
        self.cb_width + self.cb_height > 12
    }

    /// `true` iff the §7.3.11.7 `sym_mvd_flag` gate opens for the
    /// given resolved `inter_pred_idc`. The flag is only signalled
    /// when `inter_pred_idc == PRED_BI` and the SPS / slice gates are
    /// open ([`Self::sym_mvd_gate_open`]). Otherwise §7.4.12.7 infers
    /// `sym_mvd_flag = 0`.
    pub fn sym_mvd_signalled(&self, inter_pred_idc: InterPredDir) -> bool {
        self.sym_mvd_gate_open && inter_pred_idc == InterPredDir::PredBi
    }

    /// `true` iff the L0 list is active for the CU given the resolved
    /// `inter_pred_idc` (`PRED_L0` or `PRED_BI`).
    pub fn l0_active(&self, inter_pred_idc: InterPredDir) -> bool {
        matches!(inter_pred_idc, InterPredDir::PredL0 | InterPredDir::PredBi)
    }

    /// `true` iff the L1 list is active for the CU given the resolved
    /// `inter_pred_idc` (`PRED_L1` or `PRED_BI`).
    pub fn l1_active(&self, inter_pred_idc: InterPredDir) -> bool {
        matches!(inter_pred_idc, InterPredDir::PredL1 | InterPredDir::PredBi)
    }

    /// `true` iff `ref_idx_l0` is signalled — list is active AND
    /// `cMax > 0` (`NumRefIdxActive[0] > 1`). Note: when
    /// `sym_mvd_flag == 1` the §7.3.11.7 spec walk skips `ref_idx_l0`
    /// (the symmetric-MVD path infers `refIdxL0` from the §8.5.2.5
    /// derivation), so the caller threads `sym_mvd_flag` through here
    /// too.
    pub fn ref_idx_l0_signalled(&self, inter_pred_idc: InterPredDir, sym_mvd_flag: bool) -> bool {
        self.l0_active(inter_pred_idc) && self.num_ref_idx_active_l0 > 1 && !sym_mvd_flag
    }

    /// Mirror of [`Self::ref_idx_l0_signalled`] for L1.
    pub fn ref_idx_l1_signalled(&self, inter_pred_idc: InterPredDir, sym_mvd_flag: bool) -> bool {
        self.l1_active(inter_pred_idc) && self.num_ref_idx_active_l1 > 1 && !sym_mvd_flag
    }
}

/// Round-183 — output of the §7.3.11.7 non-merge inter MVP-side
/// syntax dispatcher.
///
/// Carries the four flag values plus the §7.3.11.7-typed
/// `inter_pred_idc`. When a §7.4.12.7 inference holds (the matching
/// gate in [`NonMergeMvpSyntaxGate`] is closed) the caller MUST pass
/// the inferred value (`PRED_L0` / `false` / `0`) so the wire stream
/// round-trips bit-identically through the reader.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NonMergeMvpSyntaxDecision {
    /// `inter_pred_idc[x0][y0]` — the per-CU prediction direction.
    /// Restricted to `PRED_L0` for P-slices (the
    /// [`NonMergeMvpSyntaxGate::inter_pred_idc_gate_open`]-closed
    /// inference).
    pub inter_pred_idc: InterPredDir,
    /// `sym_mvd_flag[x0][y0]` — when `true`, the CU's L0 / L1 MVDs
    /// are symmetric per §8.5.2.5 and `ref_idx_lX` / `mvd_lX` for
    /// list X are derived rather than signalled.
    pub sym_mvd_flag: bool,
    /// `ref_idx_l0[x0][y0]` — only meaningful when L0 is active and
    /// `sym_mvd_flag == 0`. Otherwise must be 0.
    pub ref_idx_l0: u32,
    /// `ref_idx_l1[x0][y0]` — only meaningful when L1 is active and
    /// `sym_mvd_flag == 0`. Otherwise must be 0.
    pub ref_idx_l1: u32,
    /// `mvp_l0_flag[x0][y0]` — AMVP candidate-list index in `0..=1`;
    /// only meaningful when L0 is active.
    pub mvp_l0_flag: u32,
    /// `mvp_l1_flag[x0][y0]` — AMVP candidate-list index in `0..=1`;
    /// only meaningful when L1 is active.
    pub mvp_l1_flag: u32,
}

impl Default for NonMergeMvpSyntaxDecision {
    /// All-zero default: `PRED_L0` direction (matching the
    /// §7.4.12.7 P-slice inference), no symmetric MVD, zero
    /// reference indices and zero MVP-list indices. Suitable as a
    /// neutral base when constructing a decision incrementally.
    fn default() -> Self {
        Self {
            inter_pred_idc: InterPredDir::PredL0,
            sym_mvd_flag: false,
            ref_idx_l0: 0,
            ref_idx_l1: 0,
            mvp_l0_flag: 0,
            mvp_l1_flag: 0,
        }
    }
}

/// Encode one `inter_pred_idc[x0][y0]` per §7.3.11.7 / Table 131.
///
/// Mirrors [`crate::leaf_cu::LeafCuReader::read_inter_pred_idc`]
/// bin-for-bin:
///
/// * `(cbWidth + cbHeight) > 12` — two-bin form. Bin 0 ctxInc per
///   [`crate::ctx::ctx_inc_inter_pred_idc_bin0`] picks `PRED_BI = 1`
///   vs the uni-pred pair; on the uni-pred branch bin 1 ctxInc per
///   [`crate::ctx::ctx_inc_inter_pred_idc_bin1`] picks
///   `PRED_L0 = 0` / `PRED_L1 = 1`.
/// * `(cbWidth + cbHeight) == 12` — single bin, `PRED_BI` suppressed
///   per Table 131, `PRED_L0 = 0` / `PRED_L1 = 1`.
///
/// Ctx slot block is `(init_type - 1) * 6` per Table 51 (only inter
/// slices signal this element).
///
/// # Preconditions
///
/// The caller is responsible for the §7.3.11.7 outer gate (B-slice
/// only — when `slice_type == P` §7.4.12.7 infers
/// `inter_pred_idc = PRED_L0` and this function must not be invoked).
/// Use [`encode_non_merge_mvp_syntax`] for the full dispatcher that
/// respects the gate automatically.
pub fn encode_inter_pred_idc(
    enc: &mut ArithEncoder,
    ctxs: &mut LeafCuCtxs,
    dir: InterPredDir,
    cb_width: u32,
    cb_height: u32,
) -> Result<()> {
    debug_assert!(
        ctxs.init_type >= 1,
        "inter_pred_idc never signalled in I slices"
    );
    let block = (ctxs.init_type as usize).saturating_sub(1) * 6;
    let n = ctxs.inter_pred_idc.len();
    if cb_width + cb_height > 12 {
        // Bin 0 — `PRED_BI` vs uni-pred.
        let inc0 = ctx_inc_inter_pred_idc_bin0(cb_width, cb_height) as usize;
        let slot0 = (block + inc0).min(n - 1);
        let bin0 = if dir == InterPredDir::PredBi { 1 } else { 0 };
        enc.encode_decision(&mut ctxs.inter_pred_idc[slot0], bin0)?;
        if dir != InterPredDir::PredBi {
            // Bin 1 — `PRED_L0` vs `PRED_L1` on the uni-pred branch.
            let inc1 = ctx_inc_inter_pred_idc_bin1() as usize;
            let slot1 = (block + inc1).min(n - 1);
            let bin1 = if dir == InterPredDir::PredL1 { 1 } else { 0 };
            enc.encode_decision(&mut ctxs.inter_pred_idc[slot1], bin1)?;
        }
    } else {
        // (cbWidth + cbHeight) == 12 — single bin, `PRED_BI`
        // suppressed per Table 131.
        debug_assert!(
            dir != InterPredDir::PredBi,
            "PRED_BI is not allowed when (cbWidth + cbHeight) == 12 per Table 131"
        );
        let inc0 = ctx_inc_inter_pred_idc_bin0(cb_width, cb_height) as usize;
        let slot0 = (block + inc0).min(n - 1);
        let bin0 = if dir == InterPredDir::PredL1 { 1 } else { 0 };
        enc.encode_decision(&mut ctxs.inter_pred_idc[slot0], bin0)?;
    }
    Ok(())
}

/// Encode one `sym_mvd_flag[x0][y0]` bin per Table 132.
///
/// Mirrors [`crate::leaf_cu::LeafCuReader::read_sym_mvd_flag`]
/// bin-for-bin:
///
/// * Selects the Table 86 ctx slot via `init_type - 1` (only inter
///   slices signal this element).
/// * `ctxInc = 0` per Table 132 — there is no §9.3.4.2.2 neighbour
///   lookup. The [`crate::ctx::ctx_inc_sym_mvd_flag`] helper exists
///   for spec traceability and is routed through here so a future
///   amendment that introduces a non-trivial derivation is caught in
///   one place.
/// * Emits a single ctx-coded FL `cMax = 1` bin.
///
/// # Preconditions
///
/// The caller is responsible for the §7.3.11.7 outer gate
/// (`sps_smvd_enabled_flag`, B-slice with a short-term L0+L1 ref pair
/// satisfying the §8.5.2.5 POC arrangement, and the resolved
/// `inter_pred_idc == PRED_BI`). When the gate is closed §7.4.12.7
/// infers `sym_mvd_flag = 0` and this function must not be invoked.
/// Use [`encode_non_merge_mvp_syntax`] for the full dispatcher.
pub fn encode_sym_mvd_flag(
    enc: &mut ArithEncoder,
    ctxs: &mut LeafCuCtxs,
    flag: bool,
) -> Result<()> {
    debug_assert!(
        ctxs.init_type >= 1,
        "sym_mvd_flag never signalled in I slices"
    );
    let inc = ctx_inc_sym_mvd_flag() as usize;
    debug_assert_eq!(
        inc, 0,
        "Table 132 lists deterministic ctxInc = 0 for sym_mvd_flag"
    );
    let slot = (ctxs.init_type as usize)
        .saturating_sub(1)
        .min(ctxs.sym_mvd_flag.len() - 1);
    let bit = if flag { 1 } else { 0 };
    enc.encode_decision(&mut ctxs.sym_mvd_flag[slot], bit)
}

/// Encode one `ref_idx_l0[x0][y0]` / `ref_idx_l1[x0][y0]` per Table
/// 127 — TR binarisation with `cMax = NumRefIdxActive[X] - 1`,
/// `cRiceParam = 0`.
///
/// Mirrors [`crate::leaf_cu::LeafCuReader::read_ref_idx_lx`]
/// bin-for-bin:
///
/// * Bin 0 ctxInc = 0, bin 1 ctxInc = 1 (Table 132 via
///   [`crate::ctx::ctx_inc_ref_idx_lx`]). Ctx slot block is
///   `(init_type - 1) * 2` per Table 51.
/// * Bins `2..` are bypass-coded.
/// * When `cMax == 0` (i.e. `num_ref_idx_active == 0` or `1`) no
///   bins are emitted — the value is necessarily 0.
///
/// # Preconditions
///
/// The caller is responsible for the §7.3.11.7 list-active gate
/// (`L0` or `L1` is active per the resolved `inter_pred_idc`) and the
/// `sym_mvd_flag == 0` gate. When either is closed §7.4.12.7 infers
/// `ref_idx_lX = 0` and this function must not be invoked.
pub fn encode_ref_idx_lx(
    enc: &mut ArithEncoder,
    ctxs: &mut LeafCuCtxs,
    value: u32,
    num_ref_idx_active: u32,
) -> Result<()> {
    debug_assert!(
        ctxs.init_type >= 1,
        "ref_idx_lX never signalled in I slices"
    );
    let cmax = num_ref_idx_active.saturating_sub(1);
    if cmax == 0 {
        // `cMax == 0` ⇒ no bins emitted; value is inferred 0. The
        // reader will likewise return 0 without consuming any bins.
        debug_assert_eq!(
            value, 0,
            "ref_idx_lX must be 0 when NumRefIdxActive == 1 (cMax == 0)"
        );
        return Ok(());
    }
    debug_assert!(
        value <= cmax,
        "ref_idx_lX = {value} exceeds cMax = {cmax} for NumRefIdxActive = {num_ref_idx_active}"
    );
    let block = (ctxs.init_type as usize).saturating_sub(1) * 2;
    let n = ctxs.ref_idx_lx.len();
    let mut i = 0u32;
    while i < cmax {
        let bit = if i < value { 1 } else { 0 };
        if i < 2 {
            let inc = ctx_inc_ref_idx_lx(i) as usize;
            let slot = (block + inc).min(n - 1);
            enc.encode_decision(&mut ctxs.ref_idx_lx[slot], bit)?;
        } else {
            enc.encode_bypass(bit)?;
        }
        if bit == 0 {
            break;
        }
        i += 1;
    }
    Ok(())
}

/// Encode one `mvp_l0_flag[x0][y0]` / `mvp_l1_flag[x0][y0]` bin per
/// Table 132.
///
/// Mirrors [`crate::leaf_cu::LeafCuReader::read_mvp_lx_flag`]
/// bin-for-bin:
///
/// * Selects the Table 132 / Table 51 ctx slot via `init_type` (the
///   bundle is shared across the three init types; only inter slices
///   reach this read in practice).
/// * `ctxInc = 0` per Table 132 — there is no §9.3.4.2.2 neighbour
///   lookup. The [`crate::ctx::ctx_inc_mvp_lx_flag`] helper exists
///   for spec traceability and is routed through here.
/// * Emits a single ctx-coded FL `cMax = 1` bin selecting the AMVP
///   candidate-list entry (0 or 1).
///
/// # Preconditions
///
/// The caller is responsible for the §7.3.11.7 list-active gate
/// (the corresponding list is active per the resolved
/// `inter_pred_idc`). When the gate is closed §7.4.12.7 infers
/// `mvp_lX_flag = 0` and this function must not be invoked.
pub fn encode_mvp_lx_flag(enc: &mut ArithEncoder, ctxs: &mut LeafCuCtxs, value: u32) -> Result<()> {
    debug_assert!(
        value <= 1,
        "mvp_lX_flag must be 0 or 1 (FL cMax = 1 per Table 132)"
    );
    let inc = ctx_inc_mvp_lx_flag() as usize;
    debug_assert_eq!(
        inc, 0,
        "Table 132 lists deterministic ctxInc = 0 for mvp_lX_flag"
    );
    let slot = (ctxs.init_type as usize).min(ctxs.mvp_lx_flag.len() - 1);
    enc.encode_decision(&mut ctxs.mvp_lx_flag[slot], value)
}

/// Round-183 — encoder-side dispatcher for the §7.3.11.7 non-merge
/// inter MVP-side syntax.
///
/// Walks §7.3.11.7 in spec order:
///
/// 1. If [`NonMergeMvpSyntaxGate::inter_pred_idc_gate_open`] returns
///    `true` (B-slice), emit one [`encode_inter_pred_idc`] (two-bin
///    or one-bin form per [`NonMergeMvpSyntaxGate::inter_pred_idc_two_bin_form`]).
///    Otherwise emit nothing — the reader will infer
///    `inter_pred_idc = PRED_L0` per §7.4.12.7 and
///    `decision.inter_pred_idc` MUST therefore equal `PRED_L0`.
/// 2. If [`NonMergeMvpSyntaxGate::sym_mvd_signalled`] returns `true`
///    for the (effective) `inter_pred_idc`, emit one
///    [`encode_sym_mvd_flag`] bin. Otherwise emit nothing — the
///    reader will infer `sym_mvd_flag = 0` and `decision.sym_mvd_flag`
///    MUST therefore equal `false`.
/// 3. For each active list (L0, L1), if the list's `ref_idx_lX` is
///    signalled per [`NonMergeMvpSyntaxGate::ref_idx_l0_signalled`] /
///    [`NonMergeMvpSyntaxGate::ref_idx_l1_signalled`] emit the
///    [`encode_ref_idx_lx`] sequence. Otherwise the reader will infer
///    0 and the decision field MUST be 0.
/// 4. For each active list (L0, L1), emit one
///    [`encode_mvp_lx_flag`] bin. Inactive lists emit nothing and the
///    reader infers 0; `decision.mvp_lX_flag` MUST be 0 for inactive
///    lists so the round-trip recovers the same value.
///
/// The reader-side §7.3.11.7 walk lives across multiple
/// [`crate::leaf_cu::LeafCuReader`] entry points already; this
/// dispatcher composes their encoder mirrors into a single
/// callable so a future encoder pipeline has the same single-function
/// surface that round-177's affine-syntax encoder offers for the
/// affine flags.
///
/// # Inferred-flag invariants (debug-only)
///
/// When a gate is closed the caller MUST pass the §7.4.12.7-inferred
/// value (`PRED_L0` / `false` / `0`) for the corresponding decision
/// field. In debug builds violations are caught by `debug_assert!`. In
/// release builds a violating `decision` will still encode 0 bins for
/// the inferred element, but the resulting wire stream will
/// round-trip back to a different `NonMergeMvpSyntaxDecision` than the
/// encoder asked for — same failure mode as round-177's affine-syntax
/// dispatcher.
pub fn encode_non_merge_mvp_syntax(
    enc: &mut ArithEncoder,
    ctxs: &mut LeafCuCtxs,
    gate: &NonMergeMvpSyntaxGate,
    decision: &NonMergeMvpSyntaxDecision,
) -> Result<()> {
    // Step 1 — inter_pred_idc.
    let outer = gate.inter_pred_idc_gate_open();
    if outer {
        // (cbWidth + cbHeight) == 12 single-bin branch forbids
        // `PRED_BI`; defer that check to `encode_inter_pred_idc`'s
        // debug_assert.
        encode_inter_pred_idc(
            enc,
            ctxs,
            decision.inter_pred_idc,
            gate.cb_width,
            gate.cb_height,
        )?;
    } else {
        // P-slice → §7.4.12.7 infers `inter_pred_idc = PRED_L0`. Any
        // other value would have the reader silently disagree with the
        // encoder.
        debug_assert_eq!(
            decision.inter_pred_idc,
            InterPredDir::PredL0,
            "P-slice → §7.4.12.7 requires inter_pred_idc = PRED_L0"
        );
    }
    let effective_inter_pred_idc = if outer {
        decision.inter_pred_idc
    } else {
        InterPredDir::PredL0
    };

    // Step 2 — sym_mvd_flag.
    if gate.sym_mvd_signalled(effective_inter_pred_idc) {
        encode_sym_mvd_flag(enc, ctxs, decision.sym_mvd_flag)?;
    } else {
        debug_assert!(
            !decision.sym_mvd_flag,
            "sym_mvd_flag gate closed → §7.4.12.7 requires sym_mvd_flag = 0"
        );
    }
    let effective_sym_mvd_flag = if gate.sym_mvd_signalled(effective_inter_pred_idc) {
        decision.sym_mvd_flag
    } else {
        false
    };

    // Step 3 — ref_idx_l0.
    if gate.ref_idx_l0_signalled(effective_inter_pred_idc, effective_sym_mvd_flag) {
        encode_ref_idx_lx(enc, ctxs, decision.ref_idx_l0, gate.num_ref_idx_active_l0)?;
    } else {
        debug_assert_eq!(
            decision.ref_idx_l0, 0,
            "ref_idx_l0 not signalled → §7.4.12.7 requires ref_idx_l0 = 0"
        );
    }

    // Step 4 — ref_idx_l1.
    if gate.ref_idx_l1_signalled(effective_inter_pred_idc, effective_sym_mvd_flag) {
        encode_ref_idx_lx(enc, ctxs, decision.ref_idx_l1, gate.num_ref_idx_active_l1)?;
    } else {
        debug_assert_eq!(
            decision.ref_idx_l1, 0,
            "ref_idx_l1 not signalled → §7.4.12.7 requires ref_idx_l1 = 0"
        );
    }

    // (mvd_coding for L0 / L1 sits between the ref-idx pair and the
    // mvp_lX_flags in §7.3.11.7. It is emitted by a separate path —
    // see [`crate::leaf_cu::LeafCuReader::read_mvd_coding`] and its
    // future encoder mirror — so this dispatcher steps across it.)

    // Step 5 — mvp_l0_flag.
    if gate.l0_active(effective_inter_pred_idc) {
        encode_mvp_lx_flag(enc, ctxs, decision.mvp_l0_flag)?;
    } else {
        debug_assert_eq!(
            decision.mvp_l0_flag, 0,
            "L0 inactive → §7.4.12.7 requires mvp_l0_flag = 0"
        );
    }

    // Step 6 — mvp_l1_flag.
    if gate.l1_active(effective_inter_pred_idc) {
        encode_mvp_lx_flag(enc, ctxs, decision.mvp_l1_flag)?;
    } else {
        debug_assert_eq!(
            decision.mvp_l1_flag, 0,
            "L1 inactive → §7.4.12.7 requires mvp_l1_flag = 0"
        );
    }

    Ok(())
}

/// Convenience constructor: build a [`NonMergeMvpSyntaxDecision`] from
/// the five raw fields, zeroing out the L0 / L1 fields whose §7.3.11.7
/// gate is closed for the given `inter_pred_idc` / `sym_mvd_flag`.
///
/// This removes a place where the decision struct could carry stale
/// values for inactive lists (which would `debug_assert!` in
/// [`encode_non_merge_mvp_syntax`]). The L1 zeroing also keeps the
/// struct in lockstep with the reader's behaviour, which only ever
/// writes the corresponding fields when the matching gate was open.
pub fn make_non_merge_mvp_syntax_decision(
    inter_pred_idc: InterPredDir,
    sym_mvd_flag: bool,
    ref_idx_l0: u32,
    ref_idx_l1: u32,
    mvp_l0_flag: u32,
    mvp_l1_flag: u32,
) -> NonMergeMvpSyntaxDecision {
    let l0 = matches!(inter_pred_idc, InterPredDir::PredL0 | InterPredDir::PredBi);
    let l1 = matches!(inter_pred_idc, InterPredDir::PredL1 | InterPredDir::PredBi);
    // `sym_mvd_flag` is only meaningful under bi-pred; clamp it.
    let sym = sym_mvd_flag && inter_pred_idc == InterPredDir::PredBi;
    NonMergeMvpSyntaxDecision {
        inter_pred_idc,
        sym_mvd_flag: sym,
        // When sym_mvd_flag is set the §7.3.11.7 walk skips
        // ref_idx_lX; clamp to 0 to match the reader's recovery.
        ref_idx_l0: if l0 && !sym { ref_idx_l0 } else { 0 },
        ref_idx_l1: if l1 && !sym { ref_idx_l1 } else { 0 },
        mvp_l0_flag: if l0 { mvp_l0_flag } else { 0 },
        mvp_l1_flag: if l1 { mvp_l1_flag } else { 0 },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::cabac::ArithDecoder;
    use crate::leaf_cu::{CuToolFlags, LeafCuReader};

    /// End-to-end round-trip: drive the encoder dispatcher against a
    /// gate + decision, decode the resulting bitstream through the
    /// matching reader entry points in spec order, and assert the
    /// recovered decision matches the input bit-for-bit.
    fn dispatcher_round_trip(
        init_type: u8,
        gate: NonMergeMvpSyntaxGate,
        decision: NonMergeMvpSyntaxDecision,
    ) -> NonMergeMvpSyntaxDecision {
        let mut enc = ArithEncoder::new();
        let mut enc_ctxs = LeafCuCtxs::init_with_init_type(26, init_type);
        encode_non_merge_mvp_syntax(&mut enc, &mut enc_ctxs, &gate, &decision)
            .expect("encode_non_merge_mvp_syntax succeeds under a valid gate/decision pair");
        enc.encode_terminate(1).expect("terminator");
        let mut padded = enc.finish();
        padded.extend_from_slice(&[0u8; 32]);
        let mut dec = ArithDecoder::new(&padded).expect("decoder accepts the encoded stream");
        let mut dec_ctxs = LeafCuCtxs::init_with_init_type(26, init_type);
        let tools = CuToolFlags::default();
        let mut reader = LeafCuReader::new(&mut dec, &mut dec_ctxs, tools);

        // Spec-order walk on the reader side — same shape as
        // `encode_non_merge_mvp_syntax`.
        let inter_pred_idc = if gate.inter_pred_idc_gate_open() {
            reader
                .read_inter_pred_idc(gate.cb_width, gate.cb_height)
                .expect("reader reads inter_pred_idc")
        } else {
            InterPredDir::PredL0
        };
        let sym_mvd_flag = if gate.sym_mvd_signalled(inter_pred_idc) {
            reader
                .read_sym_mvd_flag()
                .expect("reader reads sym_mvd_flag")
        } else {
            false
        };
        let ref_idx_l0 = if gate.ref_idx_l0_signalled(inter_pred_idc, sym_mvd_flag) {
            reader
                .read_ref_idx_lx(gate.num_ref_idx_active_l0)
                .expect("reader reads ref_idx_l0")
        } else {
            0
        };
        let ref_idx_l1 = if gate.ref_idx_l1_signalled(inter_pred_idc, sym_mvd_flag) {
            reader
                .read_ref_idx_lx(gate.num_ref_idx_active_l1)
                .expect("reader reads ref_idx_l1")
        } else {
            0
        };
        let mvp_l0_flag = if gate.l0_active(inter_pred_idc) {
            reader.read_mvp_lx_flag().expect("reader reads mvp_l0_flag")
        } else {
            0
        };
        let mvp_l1_flag = if gate.l1_active(inter_pred_idc) {
            reader.read_mvp_lx_flag().expect("reader reads mvp_l1_flag")
        } else {
            0
        };

        NonMergeMvpSyntaxDecision {
            inter_pred_idc,
            sym_mvd_flag,
            ref_idx_l0,
            ref_idx_l1,
            mvp_l0_flag,
            mvp_l1_flag,
        }
    }

    #[test]
    fn make_decision_clamps_inactive_list_fields() {
        // P-slice (PRED_L0) → L1 fields zeroed; sym_mvd_flag only
        // honoured under PRED_BI; ref_idx_lX zeroed when sym set.
        let d = make_non_merge_mvp_syntax_decision(InterPredDir::PredL0, true, 3, 5, 1, 1);
        assert_eq!(d.inter_pred_idc, InterPredDir::PredL0);
        assert!(
            !d.sym_mvd_flag,
            "sym_mvd_flag only meaningful under PRED_BI"
        );
        assert_eq!(d.ref_idx_l0, 3, "L0 active under PRED_L0");
        assert_eq!(d.ref_idx_l1, 0, "L1 inactive under PRED_L0 — must zero");
        assert_eq!(d.mvp_l0_flag, 1);
        assert_eq!(d.mvp_l1_flag, 0, "L1 inactive under PRED_L0 — must zero");

        let d = make_non_merge_mvp_syntax_decision(InterPredDir::PredL1, false, 3, 5, 1, 1);
        assert_eq!(d.ref_idx_l0, 0, "L0 inactive under PRED_L1");
        assert_eq!(d.ref_idx_l1, 5, "L1 active under PRED_L1");
        assert_eq!(d.mvp_l0_flag, 0);
        assert_eq!(d.mvp_l1_flag, 1);

        let d = make_non_merge_mvp_syntax_decision(InterPredDir::PredBi, true, 3, 5, 1, 1);
        assert!(d.sym_mvd_flag, "sym_mvd_flag honoured under PRED_BI");
        assert_eq!(d.ref_idx_l0, 0, "ref_idx_lX zeroed when sym_mvd_flag set");
        assert_eq!(d.ref_idx_l1, 0, "ref_idx_lX zeroed when sym_mvd_flag set");
        assert_eq!(d.mvp_l0_flag, 1);
        assert_eq!(d.mvp_l1_flag, 1);
    }

    #[test]
    fn p_slice_emits_zero_bins_for_inter_pred_idc() {
        // P-slice (init_type 1, b_slice = false): inter_pred_idc gate
        // closed → no bins. ref_idx_l0 single-active also closed.
        let gate = NonMergeMvpSyntaxGate {
            cb_width: 32,
            cb_height: 32,
            b_slice: false,
            sym_mvd_gate_open: false,
            num_ref_idx_active_l0: 1,
            num_ref_idx_active_l1: 0,
        };
        let decision = NonMergeMvpSyntaxDecision {
            inter_pred_idc: InterPredDir::PredL0,
            sym_mvd_flag: false,
            ref_idx_l0: 0,
            ref_idx_l1: 0,
            mvp_l0_flag: 1,
            mvp_l1_flag: 0,
        };
        let got = dispatcher_round_trip(1, gate, decision);
        assert_eq!(got, decision);
    }

    #[test]
    fn b_slice_two_bin_form_round_trips_all_dirs() {
        // B-slice (init_type 2), 32x32 (sum > 12) → two-bin form.
        // Sweep PRED_L0 / PRED_L1 / PRED_BI.
        for dir in [
            InterPredDir::PredL0,
            InterPredDir::PredL1,
            InterPredDir::PredBi,
        ] {
            let gate = NonMergeMvpSyntaxGate {
                cb_width: 32,
                cb_height: 32,
                b_slice: true,
                sym_mvd_gate_open: false,
                num_ref_idx_active_l0: 4,
                num_ref_idx_active_l1: 4,
            };
            let decision = make_non_merge_mvp_syntax_decision(dir, false, 2, 3, 1, 0);
            let got = dispatcher_round_trip(2, gate, decision);
            assert_eq!(got, decision, "B-slice {dir:?} two-bin form mismatch");
        }
    }

    #[test]
    fn b_slice_one_bin_form_when_sum_is_12() {
        // 4x8 — sum = 12 → PRED_BI is suppressed.
        for dir in [InterPredDir::PredL0, InterPredDir::PredL1] {
            let gate = NonMergeMvpSyntaxGate {
                cb_width: 4,
                cb_height: 8,
                b_slice: true,
                sym_mvd_gate_open: false,
                num_ref_idx_active_l0: 2,
                num_ref_idx_active_l1: 2,
            };
            let decision = make_non_merge_mvp_syntax_decision(dir, false, 1, 1, 1, 1);
            let got = dispatcher_round_trip(2, gate, decision);
            assert_eq!(got, decision, "B-slice {dir:?} one-bin form mismatch");
        }
    }

    #[test]
    fn sym_mvd_signalled_only_under_bi_pred_with_gate_open() {
        // sym_mvd_gate_open = true + PRED_BI → flag is signalled.
        let gate = NonMergeMvpSyntaxGate {
            cb_width: 16,
            cb_height: 16,
            b_slice: true,
            sym_mvd_gate_open: true,
            num_ref_idx_active_l0: 4,
            num_ref_idx_active_l1: 4,
        };
        // sym_mvd_flag = true under PRED_BI: ref_idx_lX is skipped per
        // §7.3.11.7 (sym path infers refIdx), so the decision must zero
        // them — the helper does this for us.
        let decision = make_non_merge_mvp_syntax_decision(
            InterPredDir::PredBi,
            true,
            3, // would be ignored — sym path infers it
            3,
            1,
            0,
        );
        let got = dispatcher_round_trip(2, gate, decision);
        assert!(got.sym_mvd_flag);
        assert_eq!(got.ref_idx_l0, 0, "sym path → ref_idx_l0 inferred 0");
        assert_eq!(got.ref_idx_l1, 0, "sym path → ref_idx_l1 inferred 0");
        assert_eq!(got, decision);

        // sym_mvd_gate_open = true but PRED_L0 → not bi-pred, flag not
        // signalled. The helper clamps `sym` to false.
        let decision = make_non_merge_mvp_syntax_decision(InterPredDir::PredL0, true, 2, 0, 1, 0);
        assert!(!decision.sym_mvd_flag, "helper clamps sym under PRED_L0");
        let got = dispatcher_round_trip(2, gate, decision);
        assert_eq!(got, decision);
    }

    #[test]
    fn ref_idx_skipped_when_num_active_is_one() {
        // num_ref_idx_active_l0 = 1 → cMax = 0 → no bins.
        let gate = NonMergeMvpSyntaxGate {
            cb_width: 16,
            cb_height: 16,
            b_slice: true,
            sym_mvd_gate_open: false,
            num_ref_idx_active_l0: 1,
            num_ref_idx_active_l1: 1,
        };
        let decision = make_non_merge_mvp_syntax_decision(InterPredDir::PredBi, false, 0, 0, 1, 1);
        let got = dispatcher_round_trip(2, gate, decision);
        assert_eq!(got, decision);
    }

    #[test]
    fn ref_idx_truncated_unary_sweep() {
        // num_ref_idx_active_l0 = 4 → cMax = 3 → sweep 0..=3 covering
        // the ctx-coded pair (bins 0/1) and the bypass tail (bin 2)
        // plus the cMax truncation (value 3 stops without a zero).
        for v in 0..=3u32 {
            let gate = NonMergeMvpSyntaxGate {
                cb_width: 16,
                cb_height: 16,
                b_slice: true,
                sym_mvd_gate_open: false,
                num_ref_idx_active_l0: 4,
                num_ref_idx_active_l1: 1,
            };
            let decision =
                make_non_merge_mvp_syntax_decision(InterPredDir::PredL0, false, v, 0, 0, 0);
            let got = dispatcher_round_trip(2, gate, decision);
            assert_eq!(got.ref_idx_l0, v, "ref_idx_l0 round-trip failed at v={v}");
            assert_eq!(got, decision);
        }
    }

    #[test]
    fn mvp_lx_flag_round_trips_both_values() {
        for v in 0..=1u32 {
            let gate = NonMergeMvpSyntaxGate {
                cb_width: 16,
                cb_height: 16,
                b_slice: true,
                sym_mvd_gate_open: false,
                num_ref_idx_active_l0: 2,
                num_ref_idx_active_l1: 2,
            };
            let decision =
                make_non_merge_mvp_syntax_decision(InterPredDir::PredBi, false, 0, 0, v, 1 - v);
            let got = dispatcher_round_trip(2, gate, decision);
            assert_eq!(got, decision);
        }
    }

    #[test]
    fn full_b_slice_bi_pred_round_trip_under_all_gates_open() {
        // The whole §7.3.11.7 MVP-side walk in one shot under
        // sym_mvd_gate_open = false (so ref_idx_lX is signalled).
        let gate = NonMergeMvpSyntaxGate {
            cb_width: 32,
            cb_height: 16,
            b_slice: true,
            sym_mvd_gate_open: false,
            num_ref_idx_active_l0: 4,
            num_ref_idx_active_l1: 4,
        };
        let decision = make_non_merge_mvp_syntax_decision(InterPredDir::PredBi, false, 2, 3, 1, 0);
        let got = dispatcher_round_trip(2, gate, decision);
        assert_eq!(got, decision);
        assert_eq!(got.inter_pred_idc, InterPredDir::PredBi);
        assert_eq!(got.ref_idx_l0, 2);
        assert_eq!(got.ref_idx_l1, 3);
        assert_eq!(got.mvp_l0_flag, 1);
        assert_eq!(got.mvp_l1_flag, 0);
    }

    #[test]
    fn gate_helpers_match_the_dispatcher_branches() {
        // Sanity-check the per-element gate predicates without
        // round-tripping through CABAC.
        let p_gate = NonMergeMvpSyntaxGate {
            cb_width: 16,
            cb_height: 16,
            b_slice: false,
            sym_mvd_gate_open: true, // gate-open meaningless on P-slice
            num_ref_idx_active_l0: 4,
            num_ref_idx_active_l1: 4,
        };
        assert!(!p_gate.inter_pred_idc_gate_open());
        // Under the inferred PRED_L0, L0 is active and L1 is not.
        assert!(p_gate.l0_active(InterPredDir::PredL0));
        assert!(!p_gate.l1_active(InterPredDir::PredL0));
        // sym_mvd_flag isn't signalled because PRED_L0 ≠ PRED_BI.
        assert!(!p_gate.sym_mvd_signalled(InterPredDir::PredL0));

        let b_gate = NonMergeMvpSyntaxGate {
            cb_width: 32,
            cb_height: 32,
            b_slice: true,
            sym_mvd_gate_open: true,
            num_ref_idx_active_l0: 4,
            num_ref_idx_active_l1: 4,
        };
        assert!(b_gate.inter_pred_idc_gate_open());
        assert!(b_gate.inter_pred_idc_two_bin_form());
        assert!(b_gate.sym_mvd_signalled(InterPredDir::PredBi));
        assert!(!b_gate.sym_mvd_signalled(InterPredDir::PredL0));
        assert!(b_gate.l0_active(InterPredDir::PredBi));
        assert!(b_gate.l1_active(InterPredDir::PredBi));
        assert!(b_gate.ref_idx_l0_signalled(InterPredDir::PredBi, false));
        assert!(!b_gate.ref_idx_l0_signalled(InterPredDir::PredBi, true));

        // (cbWidth + cbHeight) == 12 path: 4x8.
        let one_bin_gate = NonMergeMvpSyntaxGate {
            cb_width: 4,
            cb_height: 8,
            b_slice: true,
            sym_mvd_gate_open: false,
            num_ref_idx_active_l0: 2,
            num_ref_idx_active_l1: 2,
        };
        assert!(one_bin_gate.inter_pred_idc_gate_open());
        assert!(!one_bin_gate.inter_pred_idc_two_bin_form());
    }
}
