# Changelog

All notable changes to this crate are recorded here.

## [Unreleased]

### Other

- round 207: encoder-side §7.3.10.5 multi-CP-MV affine MVD dispatcher (`encode_non_merge_inter_pre_residual_affine`) — emits `numCpMv` `mvd_coding()` invocations per active list per §7.3.10.5 listing
- round 201: encoder-side §7.3.10.5 bcw_idx dispatcher (`bcw_idx_enc` module) wired into the non-merge inter pre-residual composite walker as step 11 (after AMVR)
- round 195: encoder-side §7.3.10.10 amvr_flag + amvr_precision_idx dispatcher (`amvr_enc` module) wired into the non-merge inter pre-residual composite walker

## [0.0.7](https://github.com/OxideAV/oxideav-h266/compare/v0.0.6...v0.0.7) - 2026-05-30

### Other

- round 190: encoder-side composite walker for §7.3.11.7 non-merge inter pre-residual
- round 187: encoder-side §7.3.10.10 mvd_coding() dispatcher + §9.3.3.6 limited-EGk abs_mvd_minus2
- round 183: encoder-side §7.3.11.7 non-merge inter MVP-side syntax dispatcher
- round 177: encoder-side §7.3.11.7 non-merge inter affine-syntax dispatcher
- round 164: §7.3.11.7 non-merge inter affine-syntax dispatcher + §8.5.5.2 eq. 160 fold
- round 159: §7.3.11.7 cu_affine_type_flag CABAC reader + Table 85 ctx bundle
- round 152: §7.3.11.7 inter_affine_flag CABAC reader + Table 84 ctx bundle
- round 149: per-CB MergeSubblockFlag/InterAffineFlag neighbour grid fuse
- round 146: §7.3.11.7 merge_data() wire-up of merge_subblock_flag/idx
- round 142: §7.4.3.4 eq. 85 MaxNumSubblockMergeCand derivation
- round 139: §7.3.11.7 merge_subblock_flag + merge_subblock_idx CABAC readers
- §8.5.5.3 CTU-walker fuse — per-sub-block motion fill
- fix §8.5.5.4 tempMv rounding to leftShift=0 (integer-luma units)
- §8.5.5.3/§8.5.5.4 SbTMVP record + availability gate (round 132)
- round-129 §7.3.10.5 bcw_idx gate evaluator + MvField fuse
- round 126 — §7.3.10.5 bcw_idx CABAC reader (Table 91 + Table 132)
- round 120: §8.5.5.7 affine AMVP candidate list (luma CPMV predictors)
- §8.5.2.11 live temporal Col AMVP candidate derivation
- §8.5.2.9 step-5 HMVP RPL-reference-match candidate fill
- §8.5.2.8/§8.5.2.9/§8.5.2.10 AMVP MVP candidate derivation
- §7.3.10.8 non-merge inter MVP-side syntax (inter_pred_idc / sym_mvd_flag / ref_idx_lX / mvp_lX_flag)
- §7.3.10.10 mvd_coding() decode syntax + §9.3.3.14 limited-EGk
- round 100 — §8.5.5.2 steps 3-6 neighbour / corner-selection cascade
- round 94 — §8.5.5.2 subblockMergeCandList insertion order + merge_subblock_idx pick
- round 91 — §8.5.5.5 inherited + §8.5.5.6 constructed affine merge candidates
- round 78 — §8.5.6.4 PROF (Prediction Refinement with Optical Flow)
- round 65 — affine sub-block MC scaffold per §8.5.5.9 + Tables 30 / 31 / 32
- round 64 — Decoder-side Motion Vector Refinement (DMVR) per §8.5.3.2.4 / §8.5.3.2.5
- round 63 (Goal A) — explicit weighted bi-prediction on B-slice encoder + decoder
- round 63 (Goal B) — chroma sub-pel motion compensation on P-slice + B-slice encoder + decoder
- round 62 — multi-reference DPB on P-slice and B-slice encoder + decoder
- round 61 — sub-pel ME on B-slice (½-pel + ¼-pel per L0/L1)
- round 60 — B-slice (bi-prediction) encoder + decoder scaffold
- round 59 — sub-pel motion compensation for P-slice encoder + decoder
- round 58 — inter-frame P-slice encoder + decoder scaffold
- round 57 — MTT TT picker RDO (opt-in, parallel to round 56 BT picker)

### Added

- Round 193 — **§7.3.10.10 `amvr_flag` / `amvr_precision_idx` CABAC
  reader** + Table 89 / Table 90 context bundles in `LeafCuCtxs` +
  outer-gate dispatcher `LeafCuReader::read_amvr_inter_gated` for the
  non-merge inter branch (the IBC branch shares
  `read_amvr_precision_idx` directly because §7.3.10.5 emits no
  `amvr_flag` there — the §7.4.12.7 inference assigns 1 to it). The
  `AmvrGate` struct bundles the §7.3.10.10 outer-gate inputs
  (`sps_amvr_enabled_flag`, `sps_affine_amvr_enabled_flag`, the
  already-decoded `inter_affine_flag`, and the per-arm MVD-non-zero
  reductions over `MvdL0` / `MvdL1` for the regular arm and
  `MvdCpL0` / `MvdCpL1` over the three control points for the affine
  arm); the cascade dispatcher returns
  `(amvr_flag, amvr_precision_idx, AmvrShift)` with the §7.4.11.6 /
  Table 16 shift already folded through
  `AmvrShift::for_inter` / `AmvrShift::for_affine` per
  `inter_affine_flag`. The two new `ContextModel` bundles
  (`amvr_flag`: 4 ctxIdx as 2 ctx slots × 2 non-I initTypes;
  `amvr_precision_idx`: 9 ctxIdx as 3 ctx slots × 3 initTypes
  including I for the IBC path) thread the matching Table 89 / Table
  90 init tables into the new readers. The bin-0 ctxInc for
  `amvr_precision_idx` corrected to match Table 132's verbatim
  `(MODE_IBC) ? 1 : (inter_affine_flag == 0 ? 0 : 2)` (regular → 0,
  IBC → 1, affine → 2); the prior mapping had IBC and affine swapped
  and was unreachable until this round wired up the reader. The bin-1
  ctxInc is the deterministic `1` per Table 132 (only the regular
  AMVR `cMax = 2` path reaches it; affine and IBC truncate at bin 0
  with `cMax = 1`). New `ctx_inc_amvr_precision_idx_bin1` and
  `amvr_precision_idx_c_max` helpers surface the per-bin and
  per-arm derivations for spec traceability. 16 new lib tests pin
  the per-bin round-trip across both AMVR arms × both non-I
  initTypes × every legal precision value × all three initTypes for
  IBC, plus the cascade dispatcher's four control flows (closed
  gate ⇒ inferred-default fallthrough, open + `amvr_flag = 1` regular
  / affine round-trips for the `AmvrShift` fold, and open +
  `amvr_flag = 0` ⇒ precision-idx-not-consumed). Pre-existing
  `apply_amvr_shift` and the round-40 `AmvrShift` Table 16 rows are
  consumed unchanged. Spec ref: ITU-T H.266 (V4, 01/2026) §7.3.10.10 /
  §7.3.11.7 / §7.4.11.6 / §7.4.12.7 / Table 16 / Table 89 / Table 90 /
  Table 132.

- Round 190 — **encoder-side composite walker for the §7.3.11.7
  non-merge inter CU pre-residual syntax.** New
  `non_merge_inter_pre_residual_enc` module composes round-177's
  `affine_syntax_enc`, round-183's `non_merge_mvp_syntax_enc`, and
  round-187's `mvd_coding_enc` into a single dispatcher
  `encode_non_merge_inter_pre_residual` that walks §7.3.11.7 in spec
  order: `inter_affine_flag` → `cu_affine_type_flag` →
  `inter_pred_idc` → `sym_mvd_flag` → `ref_idx_l0` → `ref_idx_l1` →
  `mvd_coding(L0)` → `mvd_coding(L1)` → `mvp_l0_flag` →
  `mvp_l1_flag`. The interleaved `mvd_coding` between `ref_idx_lX`
  and `mvp_lX_flag` matches the §7.3.11.7 listing (the round-183
  dispatcher collapses ref-idx then mvp and explicitly steps across
  `mvd_coding`, so this composite walks the per-element encoder
  helpers directly rather than re-using the MVP-side dispatcher). The
  new `NonMergeInterPreResidualDecision` bundles the round-177 affine
  decision, the round-183 MVP decision, and the per-list lMvd pair;
  its `new(...)` constructor clamps inactive-list MVDs to zero (and
  L1 MVD under `sym_mvd_flag == 1`, where §8.5.2.5 derives
  `MvdL1 = -MvdL0` and the wire carries nothing). The dispatcher
  applies the same §7.4.12.7 inference debug-asserts as the
  underlying round-177 / round-183 dispatchers so a release-mode
  caller can't silently drift from the round-trip-symmetric reader
  path. Scope is translational only (one `mvd_coding` per active
  list); multi-CP-MV affine MVD emission (`numCpMv > 1` ⇒ one
  `mvd_coding` per control point per list), `amvr_flag`, and
  `bcw_idx` remain follow-ups. 14 new lib tests pin the end-to-end
  round-trip across the §7.3.11.7 pre-residual cascade (P-slice
  zero / non-zero mvd round-trips on both non-I initTypes, B-slice
  PRED_L0 / PRED_L1 / PRED_BI round-trips, B-slice SMVD round-trip
  with both the inferred-`-MvdL0` and zero-clamp L1 variants, outer
  affine-gate-closed zero-bin round-trip, (cbW + cbH) == 12 single-
  bin inter_pred_idc form, `num_ref_idx_active == 1` ref_idx
  suppression, §7.4.10.10 positive `|lMvd|` bound, and the
  constructor's three clamp paths).

- Round 187 — **encoder-side mirror of the §7.3.10.10 `mvd_coding()`
  structure plus its §9.3.3.6 limited-EGk `abs_mvd_minus2`
  sub-binarisation.** New `mvd_coding_enc` module lifts the
  `#[cfg(test)]`-only `encode_mvd_coding` / `encode_abs_mvd_minus2`
  helpers in `leaf_cu.rs` (used since round 103 for conformance
  round-trips) into a public encoder surface bin-for-bin parallel to
  [`crate::leaf_cu::LeafCuReader::read_mvd_coding`]. The dispatcher
  walks §7.3.10.10 in spec order: (1) both `abs_mvd_greater0_flag`
  bins ctx-coded against Table 110 / Table 132 (slot `init_type`,
  deterministic `ctxInc = 0`); (2) per non-zero component an
  `abs_mvd_greater1_flag` bin ctx-coded against Table 111 / Table
  132; (3) per non-zero component, when greater1 is set, an
  `abs_mvd_minus2` bypass payload via [`encode_abs_mvd_minus2`]
  followed by an `mvd_sign_flag` bypass bin. The §9.3.3.6 limited-EGk
  sub-binarisation transcribes `k = 1`, `maxPreExtLen = 15`,
  `truncSuffixLen = 17`: the prefix walks `(2 << preExtLen) - 2`
  while `code_value > base`, emitting `1` bins; below the cap a
  terminating `0` closes the run and the suffix is `preExtLen + k`
  bits wide; at the cap the suffix is the fixed 17-bit escape field
  with no terminating `0`. A convenience `pub const fn
  max_mvd_magnitude() -> i32` returns `2^17 - 1`, the §7.4.10.10
  positive conformance bound (the corresponding negative bound is
  `-2^17 = -131_072`, the signed-18-bit floor; this asymmetry is
  baked into the encoder's `debug_assert` range check). Together
  with round-177's `affine_syntax_enc` (`inter_affine_flag` /
  `cu_affine_type_flag`) and round-183's `non_merge_mvp_syntax_enc`
  (`inter_pred_idc` / `sym_mvd_flag` / `ref_idx_lX` / `mvp_lX_flag`)
  this completes the public encoder + decoder symmetry for the
  entire §7.3.11.7 non-merge inter pre-residual syntax: an external
  encoder can now drive a CU's full `inter_pred_idc → sym_mvd_flag →
  ref_idx_lX → mvp_lX_flag → mvd_coding(L0) → mvd_coding(L1)`
  cascade entirely through public encoder helpers. CTU-walker /
  encoder-pipeline call-site that consumes this surface remains the
  next-step follow-up. Tests: 11 new lib cases — `max_mvd_magnitude`
  spec-bound assertion; zero-pair two-bin round-trip on both
  non-I initTypes; unit-magnitude (`|lMvd| == 1`) sign-only
  round-trip exhaustive over the 4-sign 2-initType 8-case truth
  table (verifying eq. 190's `abs_mvd_minus2 = -1` inference path);
  mixed zero / non-zero component round-trips; large-magnitude
  sweep (`(2, 2)` through `(131_071, -131_071)`) exercising the
  §9.3.3.6 prefix-growth path up to the §7.4.10.10 positive ceiling;
  signed-18-bit-floor `-131_072` round-trip locking the negative
  asymmetric bound (`abs_mvd_minus2 = 131_070` at the EGk cap with
  17-bit escape suffix `65_536`); eq. 190 derivation spot-check at
  `(±9, ∓9)`; isolated EGk codec sweep via synthetic
  `mvd_coding()` cascade covering the maxPreExtLen escape boundary
  with cross-check that the direct `encode_abs_mvd_minus2` path
  produces a strictly-smaller wire than its `mvd_coding()`
  envelope; both-components-negative round-trip sweep; sign-drift-
  at-cap-magnitude round-trip across both axes and initTypes; and a
  determinism pin verifying the zero-pair encoder is repeatable
  byte-for-byte across runs. Spec reference: ITU-T H.266 | ISO/IEC
  23090-3 (V4, 01/2026) §7.3.10.10 / §7.4.10.10 (eq. 190) /
  §9.3.3.6 / §9.3.4.2.2 / Tables 51 / 110 / 111 / 132. No
  third-party VVC encoder source consulted.

- Round 183 — **encoder-side mirror of the §7.3.11.7 non-merge
  inter MVP-side syntax.** New `non_merge_mvp_syntax_enc` module
  lifts the four `#[cfg(test)]`-only encoder helpers in
  `leaf_cu.rs` (`encode_inter_pred_idc`, `encode_sym_mvd_flag`,
  `encode_ref_idx_lx`, `encode_mvp_lx_flag`) into a public
  encoder surface bin-for-bin parallel to the matching reader-side
  readers (`LeafCuReader::read_inter_pred_idc` /
  `read_sym_mvd_flag` / `read_ref_idx_lx` / `read_mvp_lx_flag`).
  Table 131 covers the two-bin form for `cbWidth + cbHeight > 12`
  (`PRED_BI = 1`, `PRED_L0 = 00`, `PRED_L1 = 01`) with bin 0
  ctxInc per `ctx_inc_inter_pred_idc_bin0` and bin 1 per
  `ctx_inc_inter_pred_idc_bin1`, the one-bin form for sum == 12
  (`PRED_BI` suppressed). Table 86 covers `sym_mvd_flag` (slot
  `init_type - 1`, deterministic ctxInc 0). Table 127 covers
  `ref_idx_lX` TR `cMax = NumRefIdxActive[X] - 1` with bins 0/1
  ctx-coded (ctxInc 0/1) and bins 2.. bypass against slot block
  `(init_type - 1) * 2`. Table 132 covers `mvp_lX_flag` single
  ctx-coded FL `cMax = 1` bin against slot `init_type`. A
  dispatcher `encode_non_merge_mvp_syntax(enc, ctxs, gate,
  decision)` walks §7.3.11.7 in spec order
  (`inter_pred_idc` → `sym_mvd_flag` → per-list `ref_idx_lX` →
  per-list `mvp_lX_flag` for active lists). When a §7.4.12.7
  inference holds (the corresponding `NonMergeMvpSyntaxGate`
  predicate is closed) the dispatcher emits zero bins, matching
  the reader's skip behaviour. A convenience constructor
  `make_non_merge_mvp_syntax_decision(...)` clamps the L1 /
  sym-path fields to 0 on inactive paths so a decision struct can
  never disagree with the reader-side §7.4.12.7 inferences. The
  module is companion to round-177's `affine_syntax_enc`; together
  they form the complete public encoder surface for the
  §7.3.11.7 non-merge inter pre-MVD syntax. Tests: 10 new lib
  cases — P-slice inter_pred_idc-gate-closed zero-bin round-trip,
  B-slice two-bin form round-trips for all three PRED_L0 /
  PRED_L1 / PRED_BI directions, B-slice one-bin (sum-12) form
  round-trips for both PRED_L0 and PRED_L1, sym_mvd_signalled-only-
  under-PRED_BI-with-gate-open round-trip (verifying ref_idx_lX
  correctly skipped per §7.3.11.7), single-active-list
  (NumRefIdxActive == 1) ref_idx_lX zero-bin round-trip,
  truncated-unary sweep over 0..=3 covering ctx-coded bins 0/1 +
  bypass bin 2 + cMax-truncation, both `mvp_lX_flag` values
  round-trip, full B-slice bi-pred round-trip exercising the entire
  §7.3.11.7 MVP-side cascade end-to-end, decision-helper clamping
  truth table across all three direction modes, and per-element
  gate-predicate sanity check for P-slice + B-slice plus the (4,8)
  one-bin edge case. CTU-walker / encoder-pipeline call-site that
  consumes this surface remains the next-step follow-up. Spec
  reference: ITU-T H.266 | ISO/IEC 23090-3 (V4, 01/2026)
  §7.3.11.7 / §7.4.12.7 / §9.3.4.2.2 / Tables 51 / 86 / 127 /
  131 / 132. No third-party VVC encoder source consulted.

- Round 177 — **encoder-side mirror of the round-164 §7.3.11.7
  non-merge inter affine-syntax dispatcher.** New
  `affine_syntax_enc` module exposes the public CABAC encoder
  helpers for `inter_affine_flag` (Table 84 / §9.3.4.2.2 Table 133
  shared `condL`/`condA` row, ctx slot
  `(init_type - 1) * 3 + ctxInc`) and `cu_affine_type_flag` (Table
  85, deterministic `ctxInc = 0` per Table 133, ctx slot
  `init_type - 1`) plus a full dispatcher
  `encode_non_merge_inter_affine(enc, ctxs, gate, decision)` that
  mirrors `LeafCuReader::read_non_merge_inter_affine` bin-for-bin
  under the same §7.3.11.7 gating cascade. When the outer or inner
  gate is closed the dispatcher emits zero bins, matching the reader's
  §7.4.12.7 inference path; when both gates are open it emits the
  two bins in the spec's order so the wire layout round-trips
  bit-identically through the reader dispatcher. A convenience
  constructor `make_non_merge_inter_affine_decision(inter_affine_flag,
  cu_affine_type_flag)` folds `motion_model` through `derive_motion_model_idc`
  (§8.5.5.2 eq. 160) so callers can't accidentally drift the typed
  enum from the flag pair. The previously `#[cfg(test)]`-only
  encoder helpers in `leaf_cu.rs` remain for the round-152 / -159 /
  -164 reader-side tests; the new public module is the surface a
  real non-merge inter CU encoder will call from outside the crate.
  Tests: 13 new cases — `make_non_merge_inter_affine_decision`
  truth-table per eq. 160 (all four input pairs, including the
  defensive `(false, true)` mapping to `Translational`); outer-gate-
  closed-by-SPS and outer-gate-closed-by-block-size round-trips
  emitting zero bins; outer-gate-open with `inter_affine_flag = 0`
  round-trip (one bin, inner gate stays closed); inner-gate-closed-
  by-SPS one-bin round-trip recovering `MotionModel::Affine4Param`;
  both-gates-open round-trips for `Affine4Param` and `Affine6Param`;
  16-tuple `(left_msb, left_aff, above_msb, above_aff)` neighbour-
  state sweep verifying ctxInc threads through end-to-end; §6.4.4
  unavailable-neighbour masking round-trip across both initTypes
  and all reachable flag pairs; six-case `(gate, decision)` sweep
  across both non-I initTypes; exhaustive 64-case
  `ctx_inc_shared_merge_subblock_inter_affine == ctx_inc_inter_affine_flag`
  identity check (Table 133 shared row); zero-bin emission
  identical-stream pin when outer gate closed; exactly-one-bin
  emission identical-stream pin against direct
  `encode_inter_affine_flag` when inner gate closed. The CTU-walker
  call-site that actually consumes this encoder + the encoder-side
  population of the round-149 affine grid remain follow-ups for the
  broader non-merge inter CU walker.

- Round 164 — **§7.3.11.7 non-merge inter affine-syntax dispatcher +
  §8.5.5.2 eq. 160 fold.** Composes the round-152 `inter_affine_flag`
  reader and the round-159 `cu_affine_type_flag` reader into a single
  entry point that mirrors the spec's `coding_unit()` text on the
  non-merge inter branch, then folds the two decisions into a typed
  `MotionModel` via §8.5.5.2 eq. 160
  `MotionModelIdc = inter_affine_flag + cu_affine_type_flag`.

  The new pieces:
  * `leaf_cu::NonMergeInterAffineGate` — pure-data gate struct
    bundling the §7.3.11.7 gating conditions (`sps_affine_enabled`,
    `sps_6param_affine_enabled`, `cb_width`, `cb_height`) with the
    four neighbour bits the §9.3.4.2.2 / Table 133 ctxInc derivation
    reads. Exposes `outer_affine_gate_open()` (the
    `sps_affine_enabled_flag && cbWidth >= 16 && cbHeight >= 16`
    test) and `inner_6param_gate_open(inter_affine_flag)` (the
    `sps_6param_affine_enabled_flag && inter_affine_flag == 1` test)
    as testable predicates.
  * `leaf_cu::NonMergeInterAffineDecision` — output struct carrying
    the raw `inter_affine_flag` / `cu_affine_type_flag` bit values
    (with §7.4.12.7 inferences applied — any flag whose gate was
    closed comes back `false`) plus the typed `MotionModel` folded
    via eq. 160.
  * `LeafCuReader::read_non_merge_inter_affine(&gate)` — dispatcher
    that reads `inter_affine_flag` only when the outer gate opens,
    reads `cu_affine_type_flag` only when the inner 6-param gate
    opens against the just-decoded `inter_affine_flag`, and returns
    the decision struct. Pure-bitstream: no reconstruction-side
    state is touched and the per-CB grid write is left to the
    caller (the CTU walker, once it brings the non-merge inter
    path online).
  * `affine::MotionModel::from_idc(u8) -> Option<Self>` — inverse of
    `MotionModel::idc()`. Rejects values outside `{0, 1, 2}` per
    Table 15.
  * `affine::derive_motion_model_idc(inter_affine_flag,
    cu_affine_type_flag)` — pure function transcribing §8.5.5.2
    eq. 160 as a typed mapping.

  15 new lib tests pin:
  * the eq. 160 truth table for every
    `(inter_affine_flag, cu_affine_type_flag)` combination including
    the unreachable `(false, true)` corner the parser never produces;
  * `MotionModel::from_idc` round-trip + out-of-range rejection for
    bogus inputs (3, 4, 7, 17, 255);
  * the dispatcher's translational-on-outer-gate-closed paths via
    both the SPS trigger and the four block-size triggers
    (8x32 / 32x8 / 16x8 / 8x16);
  * the inferred 4-param branch when
    `sps_6param_affine_enabled == 0 && inter_affine_flag == 1`
    (the inner gate stays closed and `cu_affine_type_flag` is
    inferred 0);
  * the full 6-param path with both bins driven;
  * the round-trip-both-init-types matrix across all five reachable
    `(sps_6param, inter_affine, cu_affine_type)` tuples for both
    P-slice (initType 1) and B-slice (initType 2);
  * the neighbour-state ctxInc threading through the
    `inter_affine_flag` CABAC ctxInc derivation across the 0 / 1 / 2
    ctxInc levels (the
    `cond{L,A} = MergeSubblockFlag[N] || InterAffineFlag[N]`
    Table 133 row);
  * the §7.4.12.7 inferences against the decision struct
    (gate-closed → `false` / `Translational`);
  * the `outer_affine_gate_open` / `inner_6param_gate_open` gate-test
    helpers in isolation across the AND-decomposition matrix.

  Not yet:
  * **CTU-walker call-site.** The dispatcher is implemented and
    tested in isolation; the non-merge inter CU walker that calls
    it (and then writes `MotionModelIdc` / `inter_affine_flag` /
    `cu_affine_type_flag` into the round-149 affine grids) is the
    next bounded step.
  * **Encoder-side emission.** Round-152 / -159 added per-flag
    encoder mirrors used by tests; an end-to-end encoder dispatcher
    matching this reader (call when the non-merge inter encoder
    path lands) is a separate follow-up.

- Round 159 — **§7.3.11.7 `cu_affine_type_flag[x0][y0]` CABAC reader +
  Table 85 context bundle.** Follow-on to the round-152
  `inter_affine_flag` reader: the second of the two non-merge affine
  syntax elements that drive the §8.5.5.2 `MotionModelIdc` derivation.
  When `sps_6param_affine_enabled_flag == 1` and the parser has just
  decoded `inter_affine_flag == 1`, the §7.3.11.7 inter else-branch
  signals `cu_affine_type_flag` to pick between 4-parameter
  (`flag == 0` → `MotionModelIdc = 1`) and 6-parameter
  (`flag == 1` → `MotionModelIdc = 2`) affine motion via eq. 160
  `MotionModelIdc = inter_affine_flag + cu_affine_type_flag`.

  The new pieces:
  * `SyntaxCtx::CuAffineTypeFlag` — Table 85 (2 ctxIdx, one per non-I
    initType). `CU_AFFINE_TYPE_FLAG_INIT = [35, 35]`,
    `CU_AFFINE_TYPE_FLAG_SHIFT = [4, 4]` transcribed bit-exact from
    the spec. Per Table 51 the indexing rule is `init_type − 1`
    (initType 1 → ctxIdx 0, initType 2 → ctxIdx 1; never signalled in
    I slices nor in the merge branch).
  * `ctx_inc_cu_affine_type_flag()` — deterministic `0` per Table 132
    (the spec entry simply lists "0" — no §9.3.4.2.2 / Table 133
    neighbourhood derivation applies). Wrapped in a helper so the
    reader's call site is spec-traceable and a future Table 133
    amendment that introduces a non-trivial derivation is caught in
    one place.
  * `LeafCuCtxs::cu_affine_type_flag: Vec<ContextModel>` initialised
    from the new table.
  * `LeafCuReader::read_cu_affine_type_flag()` — FL `cMax = 1` single
    ctx-coded bin reader per Table 132, indexed as `init_type - 1`
    against the per-initType pair. Caller is responsible for the
    §7.3.11.7 gate `sps_6param_affine_enabled_flag &&
    inter_affine_flag == 1` (the outer
    `sps_affine_enabled_flag && cbWidth >= 16 && cbHeight >= 16`
    gate is already implicit because `inter_affine_flag` itself is
    gated by it); when the gate is closed §7.4.12.7 infers the flag
    to 0 and the reader must NOT be invoked.
  * No CTU-walker wire-up yet — this round adds the bin-level parser
    only. The live consumer is the same future non-merge affine
    inter CU walk that will also consume the round-152
    `inter_affine_flag`, and the two flags' product feeds eq. 160's
    `MotionModelIdc` write into the round-149 affine grids.
  * 8 new lib tests covering: Table 85 length + bit-exact
    `init` / `shift`, `ctx_inc_cu_affine_type_flag` deterministic `0`,
    encoder-mirror round-trip across both non-I initTypes for both
    flag values, per-initType slot addressability for every legal
    `(init_type)` pair, bundle isolation against the round-152
    `inter_affine_flag` Table 84 state machine (different `initValue`
    / `shiftIdx` rows; driving one bundle leaves the other unchanged),
    `pState` pin against Table 85's identical-row spec (initValue
    `35` / shiftIdx `4` for both ctxIdx), and an independent-stream
    coverage sweep that verifies the reader doesn't depend on a prior
    decision on a different bundle.
  * Test count: 1011 → 1019 (+8 lib tests).

- Round 152 — **§7.3.11.7 `inter_affine_flag[x0][y0]` CABAC reader +
  Table 84 context bundle.** Adds the syntax-element parser that the
  round-149 `inter_affine_grid` neighbour-state cells will eventually
  populate. The new pieces:
  * `SyntaxCtx::InterAffineFlag` — Table 84 (6 ctxIdx, 3 per non-I
    initType). `INTER_AFFINE_FLAG_INIT = [12, 13, 14, 19, 13, 6]`,
    `INTER_AFFINE_FLAG_SHIFT = [4, 0, 0, 4, 0, 0]` transcribed
    bit-exact from the spec.
  * `ctx_inc_inter_affine_flag(...)` — §9.3.4.2.2 / eq. 1551 with the
    Table 133 row whose `condL` / `condA` predicates are identical to
    `merge_subblock_flag` (the spec lists the two syntax elements
    side-by-side). The new helper delegates to
    `ctx_inc_merge_subblock_flag` so the two derivations cannot drift
    apart by accident; a 64-case exhaustive equality test pins this
    invariant.
  * `LeafCuCtxs::inter_affine_flag: Vec<ContextModel>` initialised from
    the new table.
  * `LeafCuReader::read_inter_affine_flag(...)` — FL `cMax = 1` single
    ctx-coded bin reader per Table 132, indexed as
    `(init_type - 1) * 3 + ctxInc` against the per-initType triplet.
    Caller is responsible for the §7.3.11.7 size + sps gate
    `sps_affine_enabled_flag && cbWidth >= 16 && cbHeight >= 16` AND
    the surrounding `general_merge_flag == 0` (the syntax element is
    only present on the non-merge inter branch); when any gate is
    closed §7.4.12.7 infers it to 0.
  * No CTU-walker wire-up yet — this round adds the bin-level parser
    only. The live consumer is the future non-merge affine inter CU
    walk, which will be the one-line drop-in that finally populates
    `inter_affine_grid` with non-default values.
  * 12 new lib tests covering: Table 84 length + bit-exact init/shift,
    `ctx_inc_inter_affine_flag` equality with `ctx_inc_merge_subblock_flag`
    over all 64 input combinations + sentinel values (0/1/2),
    encoder-mirror round-trip across both initTypes with each of the
    three ctxInc cases (no neighbours / one active / both active),
    unavailable-neighbour masking, per-ctx slot addressability for
    every legal `(init_type, ctxInc)` pair, and a bundle-isolation
    test confirming `inter_affine_flag` and `merge_subblock_flag` CABAC
    state machines are disjoint and initialise to different `pState`
    rows.

- Round 149 — **per-CB live `MergeSubblockFlag[x][y]` /
  `InterAffineFlag[x][y]` neighbour grid fuse into
  [`CtuWalker::compute_cu_neighbourhood`]** — the round-146 wire-up
  followup: the §9.3.4.2.2 / Table 133 `cond{L,A} =
  MergeSubblockFlag[{L,A}] || InterAffineFlag[{L,A}]` ctxInc input for
  `read_merge_subblock_flag` was passing the §7.4.12.7 default
  `(false, false)` for both neighbours. Round-149 adds two picture-wide
  4×4 grids (`subblock_merge_grid`, `inter_affine_grid`) sharing the
  existing round-28 §8.5.6.7 intra-grid geometry, four
  `CuNeighbourhood` fields (`left_merge_subblock`,
  `above_merge_subblock`, `left_inter_affine`, `above_inter_affine`),
  and a single-source-of-truth `commit_subblock_neighbour_state(cu,
  info)` helper invoked from both the syntax-only path
  (`decode_ctu_full`) and every CU-completion site in the
  reconstruction path (intra, ISP, regular-merge inter, GPM) so the
  next CU's neighbour query reads the live state. The non-merge
  affine inter walker is not yet parsed by the CTU walker, so the
  `inter_affine_grid` always reads back `false` (matching the
  §7.4.12.7 inference); the field is plumbed end-to-end so the
  eventual affine-inter walker has a one-line drop-in point. The
  `compute_cu_neighbourhood` query samples both grids at the same
  `(xCb − 1, yCb)` / `(xCb, yCb − 1)` 4×4 cells the cu_skip_flag /
  pred_mode ctxInc already use, then passes them straight through to
  the round-146 wire-up — which now feeds them into
  `read_merge_subblock_flag` instead of the hard-coded `(false,
  false)`. 7 new lib tests pin (a) both grids default to false +
  out-of-bounds returns false per §6.4.4 unavailability, (b)
  `write_subblock_merge_block` broadcasts across every 4×4 cell with
  surrounding cells unchanged, (c) `compute_cu_neighbourhood` reads
  the left-of-`(16, 0)` and above-of-`(0, 16)` cells correctly with
  picture-edge unavailability folded in, (d) the `inter_affine_grid`
  plumbs through the same neighbour positions, (e) a CU committed
  with `merge_subblock_flag = 1` flips the per-CB grid so a CU at
  `(16, 0)` reads `left_merge_subblock = true`, (f) an intra CU
  commit clears a stale pre-loaded `true` (the §7.4.12.7 inference
  for non-inter CUs), and (g) un-loaded neighbours read `(false,
  false)` so the pre-r149 stub-call path stays byte-identical on
  every existing fixture.

- Round 146 — **§7.3.11.7 `merge_data()` wire-up of the round-139
  `merge_subblock_flag` / `merge_subblock_idx` readers behind the
  round-142 §7.4.3.4 eq. 85 gate** — the leaf-CU inter path
  (`decode_inter`) now opens with the subblock-merge prologue exactly
  as written in §7.3.11.7 (V4, 01/2026):

  ```text
  if (MaxNumSubblockMergeCand > 0 && cbWidth >= 8 && cbHeight >= 8)
      merge_subblock_flag[x0][y0]
  if (merge_subblock_flag[x0][y0] == 1) {
      if (MaxNumSubblockMergeCand > 1) merge_subblock_idx[x0][y0]
  } else { /* regular / MMVD / CIIP / GPM */ }
  ```

  When the gate is closed (`MaxNumSubblockMergeCand == 0` OR a
  side is `< 8`) the §7.4.12.7 inference `merge_subblock_flag = 0`
  applies and the reader falls through to the regular-merge tree.
  When `merge_subblock_flag == 1` the spec's §7.4.12.7 inference
  `regular_merge_flag = general_merge_flag && !merge_subblock_flag`
  resolves to 0 — the regular / MMVD / CIIP / GPM sub-trees are
  bypassed entirely. The reader still walks the trailing
  `cu_coded_flag` for non-skip CUs (Unsupported when 1, matching the
  pre-r146 merge / CIIP paths).

  New `MergeData` fields: `merge_subblock_flag: bool` +
  `merge_subblock_idx: u32`. New `CuToolFlags` field
  `max_num_subblock_merge_cand: u32` populated by
  `CtuWalker::cu_tool_flags()` from
  `SeqParameterSet::max_num_subblock_merge_cand(ph_temporal_mvp_enabled_flag)`
  — the round-142 source-of-truth scalar — so the CU-side gate stays
  in lockstep with the SPS-side eq.-85 derivation. Per-CB neighbour
  state for the §9.3.4.2.2 / Table 133 ctxInc (`cond{L,A} =
  MergeSubblockFlag[{L,A}] || InterAffineFlag[{L,A}]`) is not yet
  tracked in `CuNeighbourhood`; the wire-up passes the §7.4.12.7
  default `(false, false)` for both neighbours, matching the
  pre-r146 stub-call pattern. 7 new lib tests pin (a) the
  gate-closed path with `cb_width == 4` (no bin consumed), (b) the
  gate-closed path with `MaxNumSubblockMergeCand == 0`, (c) the
  gate-open path with `merge_subblock_flag == 0` (fall-through), (d)
  the gate-open path with `merge_subblock_flag == 1` and
  `merge_subblock_idx == 0` decoding clean (regular / MMVD / CIIP /
  GPM all bypassed, CBFs all 0), (e) the same with
  `merge_subblock_idx == 3` on init_type 2 (cMax = 4 TR ctx-bin +
  bypass tail), (f) the `MaxNumSubblockMergeCand == 1`
  idx-suppression path (cMax = 0, no idx bin on the wire), (g) the
  `CuToolFlags::default()` `max_num_subblock_merge_cand == 0`
  invariant so pre-r146 intra-only tests never accidentally open
  the new gate. The encoder-side subblock-merge emission +
  per-CB live-neighbour grid for the Table 133 ctxInc + the
  cu_coded_flag=1 transform_tree() body for subblock-merge CUs
  remain follow-ups.

- Round 142 — **§7.4.3.4 eq. 85 `MaxNumSubblockMergeCand` derivation**
  — the SPS-side scalar that drives the round-139 `merge_subblock_idx`
  `cMax = MaxNumSubblockMergeCand − 1` truncated-Rice binarisation and
  the §7.3.11.7 size-gate `MaxNumSubblockMergeCand > 0` test. New
  `SeqParameterSet::max_num_subblock_merge_cand(ph_temporal_mvp_enabled_flag)`
  reproduces the eq.-85 derivation verbatim: when `sps_affine_enabled_flag
  == 1` it returns `5 − sps_five_minus_max_num_subblock_merge_cand` and
  ignores the PH input; otherwise it returns the boolean
  `sps_sbtmvp_enabled_flag && ph_temporal_mvp_enabled_flag` (`0` or
  `1`). Both branches clamp into the §7.4.3.4 trailing "0 to 5,
  inclusive" range. Sibling `SeqParameterSet::max_num_merge_cand()`
  exposes the §7.4.3.4 regular-merge derivation
  `6 − sps_six_minus_max_num_merge_cand` (clamped `[1, 6]`) as a public
  scalar so `MaxNumGpmMergeCand` derivation has one source of truth. 6
  new lib tests cover the affine-branch full range (both PH
  polarities), the out-of-range clamp, the non-affine branch truth
  table (sbtmvp & PH-only branch), an exhaustive sweep proving the
  result stays within `[0, 5]`, the `merge_subblock_idx`-driving
  values, and the full `MaxNumMergeCand` `1..=6` mapping plus clamp.
  The §7.3.11.7 `merge_data()` wire-up that calls the round-139
  `read_merge_subblock_flag` / `read_merge_subblock_idx` behind this
  gate + the encoder-side emission remain follow-ups.

- Round 139 — **§7.3.11.7 `merge_subblock_flag` + `merge_subblock_idx`
  CABAC readers** — the live-stream entry into the round-135 SbTMVP
  CTU-walker fuse. New `SyntaxCtx::MergeSubblockFlag` (Table 107: 6
  ctxIdx, 3 per non-I initType, initValue = `[48, 57, 44, 25, 58, 45]`,
  shiftIdx all 4) + `SyntaxCtx::MergeSubblockIdx` (Table 108: 2 ctxIdx,
  one per non-I initType, initValue = `[5, 4]`, shiftIdx all 0). New
  `ctx_inc_merge_subblock_flag(left_msb, left_aff, avail_l, above_msb,
  above_aff, avail_a)` implements §9.3.4.2.2 / eq. 1551 with the Table
  133 merge-side row `cond{L,A} = MergeSubblockFlag[{L,A}] ||
  InterAffineFlag[{L,A}]`, `ctxSetIdx = 0` — yielding `ctxInc ∈
  {0, 1, 2}` after the §6.4.4 availability mask. `ctx_inc_merge_subblock_idx`
  is fixed 0 (only bin 0 ctx-coded). `LeafCuReader::read_merge_subblock_flag`
  consumes one ctx-coded bin against the `(init_type − 1) * 3 + ctxInc`
  slot and returns the §7.3.11.7 flag value (FL `cMax = 1`).
  `LeafCuReader::read_merge_subblock_idx(max_num_subblock_merge_cand)`
  decodes the TR-binarised sub-block-merge-candidate index (`cMax =
  MaxNumSubblockMergeCand − 1`, `cRiceParam = 0`) — bin 0 ctx-coded
  against the Table 108 slot `init_type − 1` with `ctxInc = 0`, bins
  1.. bypass-coded; returns 0 without consuming bits when
  `max_num_subblock_merge_cand ≤ 1` (matching §7.4.12.7's "inferred to
  0" rule). The two new context arrays land on `LeafCuCtxs` and are
  populated by `init_with_init_type`. 16 new lib tests pin the
  Table-133 ctxInc truth table (no-neighbours / one-neighbour /
  both-neighbours / availability-masked), the Tables 107 + 108
  transcription bit-exact, the `read_merge_subblock_flag` round-trips
  on both non-I initTypes with neighbour-driven ctxInc selection, the
  `read_merge_subblock_idx` round-trips at `cMax = 1` and full
  `cMax = 4`, the `MaxNumSubblockMergeCand ≤ 1` suppression branch
  (sentinel-byte non-consumption check), and the value-0 exact-one-
  ctx-bin path. The non-merge inter CU walk fuse that calls these
  readers + populates the round-135 `SbColGrid` from the live ColPic
  + the encoder-side emission for both syntax elements remain
  follow-ups.

- Round 135 — **§8.5.5.3 SbTMVP CTU-walker fuse: per-sub-block motion
  fill.** Builds on the round-132 `SbTmvpRecord` + availability gate to
  land the §8.5.5.3 main-body loop. New `sbtmvp` surface:
  `ColBlockMotion` (the collocated picture's per-8×8-cell `CuPredMode`
  / `predFlagColLX` / `mvLXCol` / `refIdxLXCol` read-back),
  `ColMotionSampler` (the `(xColCb, yColCb) → ColBlockMotion`
  closure), `SbTmvpFuseInputs` (CU origin, `CtbLog2SizeY`, clip
  boundary, slice type, `NoBackwardPredFlag`, the POC operands for
  §8.5.2.12 scaling + the `poc_of_col_ref(listCol, refIdxCol)`
  resolver), `SbColMotion` (per-sub-block `mvLXSbCol` /
  `predFlagLXSbCol` / `refIdxLXSbCol = 0`), and `SbColGrid` (the
  row-major `numSbX × numSbY` fill with an `at(xSbIdx, ySbIdx)`
  accessor). `fill_subblock_motion(record, inputs, col_sampler)`
  iterates the grid: eqs. 720 / 721 sub-block centre → eqs. 722 – 724
  clip → 8×8 snap `(xColCb, yColCb) = ((xColSb >> 3) << 3, …)` →
  §8.5.2.12 (with `sbFlag = 1`) per-list collocated-MV read (the
  `predFlagColLX == 1` LX path, the `NoBackwardPredFlag &&
  predFlagColLY == 1` cross-list LY fallback, §8.5.2.15 integer-pel
  buffer compression, and the eqs. 598 – 605 POC scaling with the
  eq. 600 equal-distance passthrough) → eqs. 725 / 726 substitute the
  record's CU-centre default `ctrMvLX` / `ctrPredFlagLX` when both
  list reads report `predFlagLXSbCol == 0` (the intra / unavailable
  fallback). L1 reads are gated on `sh_slice_type == B`. 7 new lib
  tests pin the uniform-field fill, the all-intra centre fallback, a
  mixed inter/intra grid, the B-slice bi-pred both-lists fill, the
  P-slice L1-suppression (with L0 borrowing L1 via
  `NoBackwardPredFlag`), the eqs. 601 – 605 POC scaling, and the
  `tempMv` collocated-sample offset. The §7.4.6 `merge_subblock_flag`
  reader wire-up for live SbCol selection + the encoder-side SbCol
  emission remain follow-ups.

- Round 132 — **§8.5.5.3 / §8.5.5.4 SbTMVP record + availability gate**
  for the SbCol slot of the §8.5.5.2 sub-block merge candidate list.
  New `sbtmvp` module exposes: `SbTmvpAvailability` capturing the
  §8.5.5.3 first-bullet inputs (`sps_sbtmvp_enabled`,
  `ph_temporal_mvp_enabled`, `cb_width`, `cb_height`,
  `col_pic_present`); `is_sbtmvp_available(g)` implementing the
  first-bullet short-circuit (`true` iff `cbWidth >= 8 && cbHeight
  >= 8 && both flags == 1 && ColPic present`);
  `SbTmvpCenterLoc::derive(xcb, ycb, cb_w, cb_h, ctb_log2_size_y)`
  for §8.5.5.3 eqs. 711 – 714; `SbTmvpGrid::derive(cb_w, cb_h)` for
  eqs. 715 – 718 plus `subblock_centre` for eqs. 720 / 721;
  `derive_temp_mv(...)` for §8.5.5.4 (A1-neighbour `mvL{0,1}A1`
  fallback chain with the `DiffPicOrderCnt(ColPic,
  RefPicList[X][refIdxLXA1]) == 0` POC-match gate and the §8.5.2.14
  `rightShift = 4, leftShift = 0` rounding — `leftShift = 0` yields an
  integer-luma-sample `tempMv`, distinct from the AMVR symmetric-shift
  path);
  `PictureBoundary::{Picture, Subpic}` +
  `clip_col_subblock_location` / `clip_col_centre_location` for
  §8.5.5.3 eqs. 722 – 724 and §8.5.5.4 eqs. 729 – 731; and the
  `SbTmvpRecord` typed record (collocated picture POC, sub-block
  grid geometry, `refIdxLXSbCol = 0` per eq. 719, `tempMv`, the
  walker-populated `(ctrPredFlagLX, ctrMvLX)` slots + the
  `is_sb_col_available()` step-3 final-decision helper). 31 new lib
  tests pin the gate truth table (each close-condition + the
  all-open path + the exact 8×8 boundary), the centre-loc
  derivation, the grid geometry, the tempMv derivation across the
  A1-availability / per-list-match / slice-type / §8.5.2.14
  rounding axes, the Clip3 helpers across the picture / sub-picture
  branches and the CTB-upper / CTB-lower clamps, and the
  `SbTmvpRecord` defaults / `is_sb_col_available` truth table. The
  CTU walker fuse that populates the per-sub-block `mvLXSbCol` /
  `predFlagLXSbCol` arrays from the collocated picture's
  `CuPredMode` + per-4×4 motion field (the §8.5.2.12 collocated-MV
  derivation invoked per sub-block) and the wire-up of the §7.4.6
  `merge_subblock_flag` reader for live SbCol selection remain
  follow-ups.

- Round 129 — **§7.3.10.5 `bcw_idx` gate evaluator + `MvField` fuse**
  (the round-126 reader-side note's call-out). New
  `leaf_cu::BcwIdxGate` packs the spec's seven gate inputs
  (`sps_bcw_enabled, inter_pred_idc, luma_weight_l0/l1_flag,
  chroma_weight_l0/l1_flag, cb_width * cb_height, no_backward_pred_flag`)
  into one pure-data struct; `BcwIdxGate::is_open()` reproduces the
  §7.3.10.5 conditional verbatim (`sps_bcw_enabled_flag &&
  inter_pred_idc == PRED_BI && every per-list weighted-prediction
  flag == 0 && cbWidth * cbHeight >= 256`). New
  `LeafCuReader::read_bcw_idx_gated(gate)` invokes `read_bcw_idx`
  with the threaded `no_backward_pred_flag` when the gate opens and
  returns 0 without consuming any bins when closed (per §7.4.12.5
  "When `bcw_idx[x0][y0]` is not present, it is inferred to be equal
  to 0"). `LeafCuReader::read_bcw_idx_into(gate, &mut MvField)` adds
  the spec's `BcwIdx[x0][y0] = bcw_idx[x0][y0]` array-write step,
  clearing stale `MvField::bcw_idx` to 0 when the gate is closed (the
  CTU walker then broadcasts that value across every covered 4x4
  block, matching the existing per-block default in `crate::ctu` and
  `crate::affine_merge`). 9 new lib tests pin the gate truth table:
  open with all preconditions met, closes on `sps_bcw_enabled =
  false`, closes on uni-pred / `None` `inter_pred_idc`, closes on
  any single weighted-prediction flag set (all four
  individually-checked), the `>= 256` area threshold exhaustively
  swept (16x16, 8x32, 32x8 open; 8x16, 16x8, 8x8, 4x32 close), the
  closed-gate read leaves the bitstream pointer parked at a sentinel
  bypass bit, the open-gate read returns the decoded value, the
  `MvField` writer overwrites a stale value with the decoded one, the
  closed-gate `_into` path clears a stale value to 0 (the §7.4.12.5
  array-slot inference), and the `no_backward_pred_flag = true`
  path threads cMax = 4 through to a value-4 round-trip. The §7.3.10.5
  encoder-side emission for the round-58 / round-60 BCW-RDO winners
  (the matching helper plumbing the chosen `bcw_idx` back into the
  bitstream when the gate is open) and the CTU-walker drop-in that
  fills `BcwIdxGate` from live per-CB state (the round-29
  pred-weight-table walker output + the round-108 `inter_pred_idc` /
  `ref_idx_lX` parse) remain a follow-up.

- Round 126 — **§7.3.10.5 `bcw_idx` CABAC reader** (the last
  unparsed AMVP-side bin in the non-merge inter CU `coding_unit()`
  else-branch). `LeafCuReader::read_bcw_idx(no_backward_pred_flag)`
  decodes the §7.4.12.5 syntax element per Tables 91 / 131 / 132:
  TR binarisation with `cMax = NoBackwardPredFlag ? 4 : 2,
  cRiceParam = 0`; bin 0 is context-coded against the new
  `SyntaxCtx::BcwIdx` slot `init_type - 1` (initValue/shiftIdx
  `(4, 1)` for initType 1, `(5, 1)` for initType 2 — never
  signalled in I slices); bins 1.. are bypass-coded; truncation at
  `cMax - 1` ones (no terminating zero on the maximum-value path).
  The §7.3.10.5 caller is responsible for the gating conditional
  (`sps_bcw_enabled_flag && inter_pred_idc == PRED_BI &&
   luma_weight_lX_flag[refIdxLX] == 0 (X = 0, 1) &&
   chroma_weight_lX_flag[refIdxLX] == 0 (X = 0, 1) &&
   cbWidth * cbHeight >= 256`); when closed the value is inferred 0
  per §7.4.12.5 (matching the existing per-block default in
  `crate::ctu` / `crate::affine_merge`). The returned value flows
  into the round-29 `inter` `bcwWLut[k] = {4, 5, 3, 10, -2}` weight
  lookup once the CTU walker drops the per-CU read into the
  non-merge inter path. New `tables::{BCW_IDX_INIT,
  BCW_IDX_SHIFT}` and `ctx::ctx_inc_bcw_idx(bin_idx)` (Table 132
  fixed 0 for bin 0; debug-asserts no higher bin is asked) plus the
  `LeafCuCtxs::bcw_idx` 2-entry bundle. 6 new lib tests cover an
  encoder-mirror round-trip across all five legal values for both
  `cMax` cases on both non-I initTypes, prove `bcw_idx == 0` reads
  exactly one context bin (zero bypass tail) via a sentinel-byte
  check, and pin the Table 91 transcription + per-initType slot
  isolation. The CTU-walker fuse (the §7.3.10.5 gate evaluator
  populating `(sps_bcw_enabled, inter_pred_idc, luma_weight_*,
  chroma_weight_*, cb_w * cb_h)` and pushing the decoded value into
  `MvField::bcw_idx`) and the matching encoder-side emission for
  the round-58 `encode_p_slice` / `encode_b_slice` BCW-RDO winners
  remain a follow-up.

- Round 120 — **§8.5.5.7 affine AMVP — luma affine control-point
  motion-vector predictor candidate list**. New `affine_amvp` module
  drives the spec's 9-step process end-to-end on pure-data inputs
  (the CTU walker wires `NeighbourAffineQuery` from the live per-CB
  CPMV grid in a follow-up round):
  - `derive_inherited_affine_mvp_candidate(xCb, yCb, cb_w, cb_h, nb,
    ctx, amvr, num_cp_mv, is_ctu_boundary_above)` — the per-neighbour
    inherited CPMVP wrapper around the round-91
    `affine_merge::derive_inherited_affine_cpmvs`. Applies the
    §8.5.5.7 step-4 / step-5 gates: neighbour available + affine
    (`MotionModelIdc > 0`) + list X first then list Y = 1 − X with
    `DiffPicOrderCnt == 0`. Builds the `NeighbourCpmvSource::SameOrLeftCtu`
    or `::AboveCtuBoundary` variant from the neighbour record (the
    CTU-boundary path reconstructs the two bottom-row sub-block MVs from
    the neighbour's CPMVs using a local §8.5.5.9 driver, mirroring what
    the spec's `MvLX[x][y]` sub-block grid would carry). Every emitted
    CP component is AMVR-rounded (§8.5.2.14 with
    `rightShift = leftShift = AmvrShift`).
  - `derive_constructed_affine_mvp_candidate(top_left, top_right,
    bottom_left, num_cp_mv, amvr)` — the §8.5.5.8 inner derivation
    that emits `availableConsFlagLX = 1` plus the per-CP candidate when
    all `numCpMv` corners are available. Each corner is fed by
    `pick_constructed_corner(&[NeighbourAffineQuery], &ctx)` which walks
    the per-corner cascade (B2/B3/A2 for TL, B1/B0 for TR, A1/A0 for
    BL) and picks the first effectively-available neighbour with a
    POC-matching prediction on list X or Y. Unlike the §8.5.5.6
    *merge* constructed path, §8.5.5.8 reads the neighbour's
    sample-anchor MV directly (`MvLX[xNbTL][yNbTL]`) — translational
    neighbours contribute here.
  - `build_affine_mvp_cand_list(AffineMvpListInputs<'_>)` — §8.5.5.7
    steps 1 – 9 assembled in spec order: inherited A → inherited B
    → constructed full → standalone corners (`nbCpIdx = 2, 1, 0`,
    each replicating one corner's MV across every CP) → temporal MV
    (replicated across every CP) → zero-MV pad to exactly
    `MAX_AFFINE_MVP_CAND == 2` entries. No pruning step is specified
    by §8.5.5.7.
  - `select_affine_mvp(list, mvp_lx_flag)` — eq. 840
    `cpMvpLX[cpIdx] = cpMvpListLX[mvp_lX_flag][cpIdx]`.
  - `derive_final_affine_cpmvs(mvp, mvd_cp)` — eqs. 664 – 667 final
    MV folding with the modular 18-bit `(mvpCp + mvdCp) & (2^18 − 1)`
    wrap and the `>= 2^17 ? − 2^18 : ·` two's-complement unwrap, per
    CP. Produces an `AffineCpmvs` ready for the round-65 sub-block
    MV derivation.

  24 new lib tests covering: translational-neighbour rejection,
  unavailable-neighbour rejection, POC-mismatch rejection, 2-cp + 3-cp
  inherited emission, cross-list (`Y = 1 − X`) fallback,
  AMVR-rounding of components (rightShift = leftShift = AmvrShift), the
  §8.5.5.8 per-corner cascade (translational-neighbour acceptance,
  first-available pick, no-hit default), the §8.5.5.8 4-param vs
  6-param all-corners-required gate, full 6-param constructed
  emission, the 9-step list assembly (inherited-only, constructed
  after one inherited, step-7 single-corner order 2 → 1 → 0, step-8
  temporal replication, zero-pad), step-order dominance (inherited
  takes both slots even when constructed / temporal are available),
  `select_affine_mvp` indexing, `derive_final_affine_cpmvs` per-CP
  MVD folding + two's-complement wrap + 6-param, and the CTU-boundary
  inherited path. End-to-end round-trip test runs the full
  inherited → list → select → final-MV-fold chain. Module total 24
  tests; crate 909 (previously 885).

  The CTU-walker fuse (populating `NeighbourAffineQuery` from the live
  per-CB CPMV grid + the §6.4.4 neighbour-availability table) and
  wiring the §8.5.2.11 temporal MV resolver into the per-CU path
  remain follow-ups.

- Round 114 — **§8.5.2.9 step-5 AMVP history-based (HMVP)
  candidate fill** — the RPL-reference-match filter the round-111
  `build_mvp_cand_list` previously consumed pre-injected. New
  `amvp::derive_hmvp_mvp_candidates(table, ctx, AmvrShift,
  slots_remaining)`:
  - Walks `HmvpCandList[i − 1]` for `i = 1..Min(4, NumHmvpCand)` —
    i.e. `entries[0]`, `entries[1]`, … in **oldest-first** order
    (index 0 = oldest), capped at the first 4 entries. This is the
    *opposite* traversal from the §8.5.2.6 *merge* HMVP walk
    (`HmvpCandList[NumHmvpCand − hMvpIdx]`, newest-first, all
    `NumHmvpCand`), and carries no A1/B1 `sameMotion` prune.
  - For each consulted entry and for each RPL `LY` with `Y = X`
    first then `Y = 1 − X`, admits the entry's LY motion vector when
    that entry's `RefIdxLY` references the same reference picture as
    the current CU's `RefPicList[X][refIdxLX]` — established via POC
    equality (`DiffPicOrderCnt == 0`) against the same
    `AmvpRefContext` the spatial scan uses. Because the inner loop is
    over both lists (not the spatial scan's mutually-exclusive
    `If … Otherwise`), a bi-pred entry whose **both** lists reference
    the current picture contributes *twice*.
  - Each admitted MV is §8.5.2.14 AMVR-rounded; the walk halts at the
    caller's `slots_remaining` (= `MAX_MVP_CAND − numCurrMvpCand`),
    the step-5 `until numCurrMvpCand is equal to 2` cap.
  - New private `AmvpRefContext::hmvp_entry_mvs` helper yields the
    per-list matches in `X`-then-`(1 − X)` order.

  10 new lib tests: oldest-first ordering / RPL mismatch drop /
  opposite-list match / bi-pred-contributes-twice / Min(4,N) cap
  excludes the newest entry / slots_remaining cap / zero-slots /
  empty-table / AMVR rounding / end-to-end feed into
  `build_mvp_cand_list`. Module total 30 amvp tests; crate 876.

  Out of scope (next round): the live §8.5.2.11 temporal Col
  invocation behind the §8.5.2.9 step-3 gate, and the CTU-walker fuse
  that drives a complete non-merge inter `coding_unit()`
  reconstruction (wiring the slice's per-list `RefPicList` POC tables
  into the `AmvpRefContext` resolvers, plus the
  `ph_mvd_l1_zero_flag` / `sym_mvd_flag` / affine-AMVP / BCW
  cascade).

- Round 111 — **§8.5.2.8 / §8.5.2.9 / §8.5.2.10 AMVP luma
  motion-vector-prediction candidate-list derivation** — the decode-side
  process that consumes the round-108 `mvp_lX_flag` / `inter_pred_idc` /
  `ref_idx_lX` and the round-103 `mvd_coding()` `lMvd`. New `amvp`
  module:
  - `derive_spatial_mvp_candidates(xcb, ycb, cb_w, cb_h, mvf, ctx)` —
    §8.5.2.10 spatial scan. The A group walks `A0 = (xCb−1, yCb+cbH)` →
    `A1 = (A0, A0_y−1)`; the B group walks `B0 = (xCb+cbW, yCb−1)` →
    `B1 = (xCb+cbW−1, yCb−1)` → `B2 = (xCb−1, yCb−1)`. Each group picks
    the first effectively-available neighbour whose prediction — list X
    first (eqs. 588 / 590), then list `Y = 1 − X` (eqs. 589 / 591) —
    points at a reference picture with `DiffPicOrderCnt == 0` relative
    to the current CU's `RefPicList[X][refIdxLX]`. Unlike §8.5.2.3
    spatial *merge*, AMVP performs **no** POC scaling — a non-zero POC
    difference disqualifies the neighbour outright. POC matching is
    supplied via an `AmvpRefContext` (the current ref POC + two
    `refIdx → poc` resolver closures the CTU walker wires from its
    slice's `RefPicList`).
  - `round_mv_amvr(mv, AmvrShift)` — §8.5.2.14 eqs. 608 – 610 rounding
    with `rightShift = leftShift = AmvrShift` (signed-magnitude
    requantise; identity at shift 0).
  - `build_mvp_cand_list(spatial, col, hmvp)` — §8.5.2.9 steps 3 – 6:
    the step-3 Col gate (Col consulted only when *not* both A and B
    available with **different** MVs), the step-4 list construction
    (eq. 584 — A first, then B when `mvLXA != mvLXB`, then Col when
    `numCurrMvpCand < 2`), the step-5 HMVP fill (caller pre-filtered),
    and the step-6 zero-MV pad to exactly `MAX_MVP_CAND == 2`
    (eqs. 585 – 587).
  - `select_mvp(list, mvp_lx_flag)` — §8.5.2.8 step 2, eq. 583.
  - `derive_final_mv(mvp, raw_mvd, AmvrShift)` — §8.5.2.1 fold
    `mvLX = mvpLX + (mvd << AmvrShift)`, clipped to `[−2^17, 2^17 − 1]`.

  20 new lib tests: A0-first / A0→A1 fall-through / opposite-list
  (eq. 589) fallback / POC-never-matches unavailable / B0→B1→B2 scan /
  intra-neighbour-skipped; §8.5.2.14 rounding at shift 0 / 2 / 4;
  list assembly across both-distinct (Col suppressed) / both-equal (Col
  admitted) / only-A-then-Col / only-B-then-HMVP / all-zero-pad /
  single-spatial-zero-pad / HMVP-clipped-to-2; select-by-flag;
  final-MV fold + 18-bit clip; and an end-to-end spatial→round→list→
  select→fold flow.

  Out of scope (next round): the live §8.5.2.11 temporal Col invocation
  behind the §8.5.2.9 step-3 gate (the collocated derivation itself is
  shared with the merge path's existing §8.5.2.11/§8.5.2.12 code — only
  the AMVP-specific gate is new and lands here; Col is taken as an
  injected optional candidate for now), the §8.5.2.9 step-5 HMVP
  RPL-reference-match filter, and the CTU-walker fuse that drives a
  complete non-merge inter `coding_unit()` reconstruction (also needs
  the `ph_mvd_l1_zero_flag` / `RefIdxSymL{0,1}` cascade around
  `sym_mvd_flag` and the affine-AMVP / BCW branches).

- Round 108 — **§7.3.10.8 non-merge inter MVP-side syntax
  (`inter_pred_idc` / `sym_mvd_flag` / `ref_idx_lX` / `mvp_lX_flag`)** —
  the four AMVP-side CABAC reads that surround the round-103
  `mvd_coding()` in the MODE_INTER `coding_unit()` else-branch. Adds:
  - `LeafCuReader::read_inter_pred_idc(cb_width, cb_height)` — §9.3.3.9 /
    Table 131 binarisation. When `cbWidth + cbHeight > 12` the two-bin
    form gives `PRED_L0 = 00`, `PRED_L1 = 01`, `PRED_BI = 1`; when the
    sum equals 12 `PRED_BI` is suppressed and a single bin gives
    `PRED_L0 = 0` / `PRED_L1 = 1`. Bin 0's ctxInc is
    `7 − ((1 + Log2(cbWidth) + Log2(cbHeight)) >> 1)` (or 5 at sum 12),
    bin 1's ctxInc is fixed 5 (Table 132). Returns a new
    `leaf_cu::InterPredDir` enum (`PredL0` / `PredL1` / `PredBi`,
    `value()` ⇒ the §7.4.12.4 numeric syntax value).
  - `LeafCuReader::read_sym_mvd_flag()` — Table 132 FL `cMax = 1`,
    one ctx-coded bin (`ctxInc = 0`); §7.4.12.4 symmetric-MVD signalling.
  - `LeafCuReader::read_ref_idx_lx(num_ref_idx_active)` — Table 127 TR
    `cMax = NumRefIdxActive[X] − 1`, bin 0 ctxInc 0, bin 1 ctxInc 1,
    bins 2.. bypass (Table 132). `cMax == 0` (single ref) reads no bins.
  - `LeafCuReader::read_mvp_lx_flag()` — Table 132 FL `cMax = 1`, one
    ctx-coded bin (`ctxInc = 0`); the §8.5.2.8 AMVP candidate index.
  - `SyntaxCtx::{InterPredIdc, SymMvdFlag, RefIdxLx, MvpLxFlag}` + Table
    83 / 86 / 87 / 88 `initValue` / `shiftIdx` arrays, wired through
    `LeafCuCtxs` (`inter_pred_idc` / `sym_mvd_flag` / `ref_idx_lx` /
    `mvp_lx_flag` Vec fields) + `init_with_init_type`. Per Table 51 the
    per-initType slot blocks are `(init_type − 1) * 6` (inter_pred_idc),
    `init_type − 1` (sym_mvd_flag), `(init_type − 1) * 2` (ref_idx_lX),
    and `init_type` (mvp_lX_flag).
  - `ctx::{ctx_inc_inter_pred_idc_bin0, ctx_inc_inter_pred_idc_bin1,
    ctx_inc_sym_mvd_flag, ctx_inc_ref_idx_lx, ctx_inc_mvp_lx_flag}`
    per Table 132.

  9 new lib tests (encoder-mirror round-trips driving the same context
  bundle bin-for-bin): inter_pred_idc two-bin form (all three dirs) +
  single-bin form (sum == 12) + bin-0 ctxInc derivation across
  power-of-two sizes; sym_mvd_flag both init types; ref_idx_lX
  single-ref (no bins) + truncated-unary 0..3 across the ctx/bypass
  boundary + two-ref ctx-bin0-only; mvp_lX_flag all three init types.

  Out of scope (next round): fusing these reads into a complete
  non-merge inter `coding_unit()` walk — needs the §8.5.2.8 / §8.5.2.9
  AMVP MVP-candidate list derivation to consume `mvp_lX_flag`, the
  §7.4.11.6 AMVR shift applied to the round-103 `lMvd`, the
  `ph_mvd_l1_zero_flag` / `RefIdxSymL{0,1}` inference cascade around
  `sym_mvd_flag`, and `inter_affine_flag` / `cu_affine_type_flag` /
  `bcw_idx` / `amvr_flag` for the affine-AMVP and BCW branches.

- Round 103 — **§7.3.10.10 `mvd_coding()` decode syntax + §9.3.3.14
  limited-EGk binarisation** — the spec-conformant CABAC parser for one
  motion-vector-difference structure, the first concrete step against
  the "full mvd_coding" lacks-tail item. Adds:
  - `LeafCuReader::read_mvd_coding()` — decodes the §7.3.10.10 bin
    sequence (`abs_mvd_greater0_flag[0/1]` → `abs_mvd_greater1_flag` per
    non-zero component → `abs_mvd_minus2` + `mvd_sign_flag`) and returns
    the `lMvd[0..1]` pair (eq. 190 `lMvd = greater0 *
    (abs_mvd_minus2 + 2) * (1 − 2*sign)`, with `abs_mvd_minus2` inferred
    −1 when greater1 == 0) packed into a `MotionVector`. Exposed `pub`
    so the still-deferred non-merge inter / affine-AMVP CU paths can
    drive the shared CABAC engine through one structure.
  - `LeafCuReader::read_abs_mvd_minus2()` — the §9.3.3.14 / §9.3.3.6
    *limited* k-th order Exp-Golomb decode (`k = 1`, `maxPreExtLen = 15`,
    `truncSuffixLen = 17`), all bins bypass-coded; handles the escape
    boundary at the 15-bit prefix cap (17-bit truncated suffix).
  - `SyntaxCtx::{AbsMvdGreater0Flag, AbsMvdGreater1Flag}` + Table 110 /
    111 `initValue` / `shiftIdx` arrays (`{14, 44, 51}` / `{45, 43, 36}`,
    3 ctxIdx each, one per initType), wired through `LeafCuCtxs`
    (`abs_mvd_greater0_flag` / `abs_mvd_greater1_flag` Vec fields) and
    `init_with_init_type`.
  - `ctx::ctx_inc_abs_mvd_greater0_flag` / `ctx_inc_abs_mvd_greater1_flag`
    (both fixed 0 per Table 132; the per-initType slot selection follows
    Table 51).

  6 new lib tests: zero-both-components; unit-magnitude (greater1 == 0,
  no `abs_mvd_minus2` bin); mixed zero/non-zero per component; large
  magnitudes round-trip up to the spec's max `|lMvd| = 2^17` (encoder
  mirror drives the same context bundle bin-for-bin); eq. 190 sign
  derivation; and a direct §9.3.3.6 limited-EGk codec round-trip across
  the value range including the escape boundary.

  Out of scope (next round): wiring `read_mvd_coding()` into a full
  non-merge inter CU path (needs §8.5.2.8/§8.5.2.9 AMVP MVP-candidate
  derivation, `ref_idx_lX`, `mvp_lX_flag`, and the §7.4.11.6 AMVR shift
  applied to the parsed `lMvd`); the SbTMVP (§8.5.5.3 / §8.5.5.4) SbCol
  candidate; and the §8.5.5.7 affine-AMVP path (`mvd_coding` × numCpMv).

- Round 100 — **§8.5.5.2 steps 3 – 6 neighbour / corner-selection
  cascade** — the precursor round 94 (§8.5.5.2 steps 7 – 9 list
  assembly) and round 91 (§8.5.5.5 / §8.5.5.6 derivations) deferred.
  This cascade answers which neighbour CBs feed the inherited-A /
  inherited-B candidates and which of the seven A0/A1/A2/B0/B1/B2/B3
  corners are available for the constructed candidates. Adds to the
  `affine_merge` module:
  - `AffineNeighbourPositions` + `affine_neighbour_positions(xcb, ycb,
    cb_w, cb_h)` — §8.5.5.2 step 3 eqs. 674 – 680 sample-location
    derivation (`A0 = (xCb−1, yCb+cbH)`, `A1 = (xCb−1, yCb+cbH−1)`,
    `A2 = (xCb−1, yCb)`, `B0 = (xCb+cbW, yCb−1)`, `B1 = (xCb+cbW−1,
    yCb−1)`, `B2 = (xCb−1, yCb−1)`, `B3 = (xCb, yCb−1)`).
  - `NeighbourBlock` — per-position metadata the CTU walker supplies
    from its per-CB grids: the §6.4.4 `available` flag (queried with
    `checkPredModeY = TRUE`, `cIdx = 0`), `MotionModelIdc[xNbN][yNbN]`,
    `(CbPosX/Y[0][·])`, `CbWidth/Height[0][·]`, `PredFlagLX[xNbN][yNbN]`,
    and `BcwIdx[xNbN][yNbN]`.
  - `MotionModelOpt` — a thin `Default`-providing newtype over
    `MotionModel` (so `NeighbourBlock` can `#[derive(Default)]`); its
    `is_affine()` is the step-4 / step-5 `MotionModelIdc > 0` gate.
  - `NeighbourQuery` — the seven step-3 neighbour blocks bundled.
  - `select_inherited_a` (step 4: scan A0 → A1) + `select_inherited_b`
    (step 5: scan B0 → B1 → B2) — return the FIRST effectively-available
    affine neighbour, short-circuiting at the spec's
    `availableFlagN == FALSE` gate; each emits a `ChosenAffineNeighbour
    { available, slot, block }` carrying the originating
    `AffineNeighbourSlot` and the picked block (its `motionModelIdc` ⇒
    `numCpMv`, `cb_pos` ⇒ `(xNb, yNb)`, `cb_w/cb_h` ⇒ `nbW/nbH`, pred
    flags ⇒ eqs. 681 – 684, `bcw_idx` ⇒ `bcwIdxA`/`bcwIdxB`).
  - `parallel_merge_suppressed` / `effective_available` — the step-2 /
    step-4 / step-5 / step-6 "`xCb >> Log2ParMrgLevel == xNbN >>
    Log2ParMrgLevel && yCb >> Log2ParMrgLevel == yNbN >>
    Log2ParMrgLevel` ⇒ `availableN = FALSE`" suppression (eq. 60
    `Log2ParMrgLevel`), folded into a single per-position effective
    availability used by both the inherited scan and the corner flags.
  - `ConstructedCornerAvailability` + `constructed_corner_availability`
    (step 6) — the seven `availableA0/A1/A2/B0/B1/B2/B3` flags the
    §8.5.5.6 constructed derivation consumes. A2 / B3 are queried fresh
    here; the other five reuse the same `effective_available`
    evaluation the inherited scan saw. Corner availability does NOT
    gate on `MotionModelIdc` (a translational-but-available neighbour
    still contributes a corner MV, even though the affine-only
    inherited scan skips it).
  - `AffineNeighbourCascade` + `derive_affine_neighbour_cascade(xcb,
    ycb, cb_w, cb_h, nbrs, log2_par_mrg_level, sps_affine_enabled_flag)`
    — the steps 3 – 6 driver. `sps_affine_enabled_flag == 0`
    short-circuits to no inherited candidates + all corners
    unavailable (the spec's "When sps_affine_enabled_flag is equal to
    1, ..." gate on steps 4 / 5 / 6).

  This is the missing wiring between the round-91 derivations and the
  round-94 list assembly: the cascade's `inherited_a`/`inherited_b`
  blocks feed `derive_inherited_affine_cpmvs`, and its
  `corner_availability` flags feed
  `derive_constructed_affine_merge_candidates`.

  9 new lib tests: eqs. 674 – 680 position derivation; SPS-disable
  short-circuit; inherited-A picks A0-first then falls through to A1;
  inherited-A unavailable with no affine neighbour; inherited-B B0 →
  B1 → B2 scan order; parallel-merge-level same-cell suppression
  (`Log2ParMrgLevel == 6` suppresses, `== 2` does not); corner
  availability includes translational neighbours that the inherited
  scan skips; per-position corner-flag map; and an end-to-end test
  feeding a cascade-chosen inherited-A block's geometry straight into
  `derive_inherited_affine_cpmvs`.

  Out of scope (next round): the SbTMVP (§8.5.5.3 / §8.5.5.4) `SbCol`
  candidate (needs a per-CB collocated 8×8 motion buffer); the
  §8.5.5.7 affine AMVP candidate list (explicit `mvp_lx_flag` +
  per-CPMV `mvd_coding`); and the CTU-walker wire-up that populates
  `NeighbourQuery` from live per-CB grids and invokes the full
  §8.5.5.2 chain (cascade → inherited/constructed derivations → list
  assembly → `merge_subblock_idx` pick → sub-block MV grid).

- Round 94 — **§8.5.5.2 `subblockMergeCandList` insertion order +
  `merge_subblock_idx` pick** layered on top of the round-91 inherited
  + constructed derivation. Adds to the `affine_merge` module:
  - `MAX_SUBBLOCK_MERGE_CAND` const (== 5 per eq. 85 + the
    §7.4.3.4 "in the range of 0 to 5" bullet).
  - `SubblockMergeCandidateKind` enum tagging each slot with the
    spec's symbolic name (`SbCol`, `InheritedA`, `InheritedB`,
    `Const(K)` for K = 1..6, `Zero`).
  - `InheritedAffineCandidate` — `(available, AffineMergeCandidate)`
    pair matching the spec's `availableFlagA / availableFlagB`
    one-bit + the eqs. 681 – 684 fully-formed candidate record.
  - `SubblockMergeListInputs` — bundle of `MaxNumSubblockMergeCand`,
    `sh_slice_type == B` (eq. 690 / 691 zero-cand `L1` flip),
    `availableFlagSbCol`, the inherited-A + inherited-B records, and
    a borrowed `&ConstructedAffineCandidates`.
  - `SubblockMergeList { count, kinds, cands }` — the assembled list
    with parallel kind + payload arrays (length
    `MAX_SUBBLOCK_MERGE_CAND`); `count` is the visible-slot count
    after step-9 padding.
  - `build_subblock_merge_cand_list(inputs) -> SubblockMergeList` —
    §8.5.5.2 steps 7 – 9 verbatim: SbCol → InheritedA → InheritedB →
    Const1..Const6, each guarded by `availableFlagN && i <
    MaxNumSubblockMergeCand`, then zero-MV padding to
    `MaxNumSubblockMergeCand`.
  - `SubblockMergeList::pick(merge_subblock_idx)` — §8.5.5.2 step 10
    array-index lookup returning `(SubblockMergeCandidateKind,
    AffineMergeCandidate)`.
  - `zero_subblock_merge_candidate(slice_type_b)` — eqs. 686 – 695
    zero-cand record (uni-pred refIdxL1 = -1 on P-slice, bi-pred
    refIdxL1 = 0 on B-slice).

  **No pruning / equality pass.** §8.5.5.2 step 7 is a straight
  ordered append with no compare-against-existing branch; the spec
  is silent on dedup for the sub-block merge list (contrast with
  §8.5.2.4 regular-merge pairwise-average, which is a separate
  explicit index pick rather than a compare loop). This module
  therefore appends without dedup; any future spec amendment adding
  CPMV-equality pruning would slot into the same routine.

  12 new lib tests: `max_num_cand == 0` produces empty list;
  `max_num_cand == 3` with no real candidates emits three Zero
  slots; P-slice vs B-slice zero-cand layout (eqs. 690 / 691 flip);
  canonical SbCol → A → B → Const1 → Const2 (5-slot fill) order;
  SbCol-absent shift; A-absent / B-present case; `max_num_cand = 2`
  short-circuit drops B / Const1..6 / Zero; `max_num_cand` clamp to
  `MAX_SUBBLOCK_MERGE_CAND`; `pick` round-trips slot kind + payload
  in order; `pick` rejects out-of-range; end-to-end test fusing
  derived inherited + derived constructed records and confirming the
  `pick`ed payload flows straight into `affine::derive_subblock_mvs`.

  Out of scope (next round): the §8.5.5.2 step-6 corner-selection
  cascade (B2/B3/A2, B1/B0, A1/A0 driven by per-CB neighbour
  `MotionModelIdc` lookup); the SbTMVP (§8.5.5.3 / §8.5.5.4)
  candidate that fills `SbCol`; the affine AMVP (§8.5.5.7) path; the
  CTU-walker wire-up that loads per-CB CPMV storage + invokes this
  helper.

- Round 91 — **§8.5.5.5 inherited + §8.5.5.6 constructed affine merge
  candidate derivation** in the new `affine_merge` module. Feeds the
  round-65 sub-block MV machinery (`affine::derive_subblock_mvs`) and
  the round-78 PROF refinement (`affine::predict_luma_block_affine_prof`)
  with candidate CPMV records produced either from a single neighbour
  CB's CPMV state (the inherited path) or from per-corner triples
  combining up to four spatial+temporal neighbours (the constructed
  path):
  - `AffineCpRecord` — per-corner record carrying `(available,
    predFlagL0/L1, refIdxL0/L1, mvL0/L1, bcwIdx)`. Mirrors the spec's
    `cpMvLXCorner[k]` / `predFlagLXCorner[k]` /
    `refIdxLXCorner[k]` / `bcwIdxCorner[k]` arrays for
    `k ∈ {0, 1, 2, 3}` ↔ {top-left, top-right, bottom-left,
    temporal-BR}.
  - `AffineMergeCandidate` — per-candidate output carrying per-list
    `(predFlag, refIdx, AffineCpmvs)` + `MotionModelIdc` + `bcwIdx`.
  - `NeighbourCpmvSource::AboveCtuBoundary { mv_bottom_left,
    mv_bottom_right }` + `NeighbourCpmvSource::SameOrLeftCtu { cpmvs
    }` — the §8.5.5.5 `isCTUboundary` split. The CTU-boundary branch
    reads the neighbour's bottom-row sub-block MVs per eqs. 736 – 739
    and forces eqs. 746 / 747 4-parameter similarity even when the
    neighbour was 6-parameter; the regular branch reads the neighbour
    CB's stored CPMVs (`CpMvLX[xNb][yNb]`) per eqs. 740 – 743 and
    takes the eqs. 744 / 745 dHorY/dVerY branch only when
    `MotionModelIdc == 2`.
  - `derive_inherited_affine_cpmvs(geom, source, numCpMv)` — §8.5.5.5
    eqs. 748 – 753 inherited CPMV emission at `(xCb, yCb)`,
    `(xCb + cbWidth, yCb)`, and (for `numCpMv == 3`) `(xCb,
    yCb + cbHeight)`. §8.5.2.14 signed-magnitude round with
    `rightShift = 7`, eqs. 754 / 755 `Clip3(-2^17, 2^17 - 1, ·)`
    final clip.
  - `derive_constructed_affine_merge_candidates(cb_w, cb_h, corners,
    flags)` — §8.5.5.6 six-candidate construction. Const1..4 are the
    3-corner 6-parameter triples (gated by
    `sps_6param_affine_enabled_flag`); Const5 is the 4-parameter (CP0,
    CP1) pair from corners {0, 1}; Const6 is the 4-parameter pair from
    corners {0, 2} with the eq. 811 / 812 diagonal-projection top-
    right derivation. Per-list `(predFlag, refIdx)` gates fire
    independently for L0 + L1. bcwIdx inherits from corner 0
    (Const1/2/3/5/6) or corner 1 (Const4) when both lists materialise,
    else 0.
  - `ConstructedAffineCandidates::{available, cands, count}` —
    parallel `[bool; 6]` availability flags + `[AffineMergeCandidate;
    6]` payloads.

  22 new lib tests cover the §8.5.2.14 rounding edge cases, the
  `Clip3` saturation, every Const1..6 assembly path (`(CP0, CP1,
  CP2)` for Const1, `(CP0, CP1, CP3+CP0-CP1)` for Const2, `(CP0,
  CP3+CP0-CP2, CP2)` for Const3, `(CP1+CP2-CP3, CP1, CP2)` for
  Const4, `(CP0, CP1)` for Const5, the eq. 811 / 812 90° rotation
  for Const6), the "missing corner" / "refIdx mismatch" / "predFlag
  mismatch" short-circuits, bipred bcwIdx inheritance (corner 0 for
  Const1/3/5, corner 1 for Const4), `sps_6param` SPS-gate
  suppression, and the §8.5.5.5 inherited paths for translational-
  collapsed neighbour, 4-param shear neighbour (with the eq. 752 /
  753 third-CP emission via the similarity), and the CTU-boundary
  bullet. One integration test feeds a derived inherited CPMV record
  into `affine::derive_subblock_mvs` to confirm shape compatibility
  with the round-65 pipeline.

  Out of scope: the §8.5.5.2 driver that fuses inherited A / inherited
  B / SbTMVP / Const1..6 + zero-MV pad into `subblockMergeCandList`
  (needs per-CB CPMV storage wired into the CTU walker — a separate
  cross-module change); SbTMVP (§8.5.5.3 / §8.5.5.4) — the SbCol
  candidate (needs per-CB collocated 8×8 motion buffer); the §8.5.5.7
  affine AMVP candidate list (explicit `mvp_lx_flag` + per-CPMV
  `mvd_coding`) — same neighbour-availability rules but different
  output shape.

- Round 78 — **§8.5.6.4 PROF (Prediction Refinement with Optical Flow)**
  per-pixel refinement layered on top of the round-65 affine sub-block
  MC. Round 65 left PROF deferred ("requires gradient-based refinement
  out of scope until a later round"); round 78 wires the full
  §8.5.5.9 + §8.5.6.4 pipeline:
  - `affine::cb_prof_flag_lx(cb_w, cb_h, cpmvs, bipred,
    ph_prof_disabled_flag, rpr_constraints_active)` — the §8.5.5.9
    four-bullets disable gate. Returns `false` on any of
    `ph_prof_disabled_flag == 1`, translational-degenerate CPMVs,
    fallback mode triggered, or RPR active.
  - `affine::derive_prof_diff_mv_array(cb_w, cb_h, cpmvs)` — the
    §8.5.5.9 eqs. 880 – 887 per-pixel diffMvLX array. `sbWidth ×
    sbHeight` row-major entries with `posOffsetX = 6 * dHorX + 6 *
    dHorY`, `posOffsetY = 6 * dVerX + 6 * dVerY`, eqs. 885 / 886
    raw values, §8.5.2.14 signed-magnitude rounding with `rightShift
    = 8, leftShift = 0`, eq. 887 `Clip3(-31, 31)`.
  - `affine::predict_luma_subblock_affine_high_precision(dst_x,
    dst_y, sb_w, sb_h, src, mv, filter_set)` — the §8.5.6.4 input
    helper that produces a `(sbW + 2) × (sbH + 2)` `BitDepth + 6`
    precision predSamplesLXL halo'd block (the 1-sample halo on
    every side supplies the eqs. 955 / 956 gradient neighbours).
    Same separable filter as `predict_luma_subblock_affine` but
    leaves the per-sub-block `(v + 32) >> 6` clamp un-applied.
  - `affine::apply_prof_to_subblock(block, diff_mv, bit_depth)` —
    the §8.5.6.4 refinement. Implements eqs. 955 – 959 verbatim:
    `shift1 = 6`, `gradH[x][y] = (predL[x+2][y+1] >> 6) -
    (predL[x][y+1] >> 6)`, `gradV[x][y] = (predL[x+1][y+2] >> 6) -
    (predL[x+1][y] >> 6)`, `dI = gradH * diffMv[0] + gradV *
    diffMv[1]`, `dILimit = 1 << Max(13, BitDepth + 1)`,
    `sbSamples[x][y] = predL[x+1][y+1] + Clip3(-dILimit,
    dILimit - 1, dI)`.
  - `affine::predict_luma_block_affine_prof(...)` — full-CU driver
    composing `derive_subblock_mvs` + the new HP sub-block MC + the
    new PROF refinement + final 8-bit uni-pred clamp. Bit-identically
    short-circuits to `predict_luma_block_affine` when the
    cbProfFlagLX gate reports `false`.
  On the same round-65 horizontal-shear fixture the PROF-on driver
  reaches PSNR_Y = 53.97 dB vs the PROF-off baseline at 52.32 dB
  (+1.65 dB). Translational-degenerate CPMVs and
  `ph_prof_disabled_flag == 1` and `RprConstraintsActiveFlag == 1`
  paths all collapse to byte-identical replay of
  `predict_luma_block_affine` (the spec's "PROF disabled" branches).
  Coverage adds 11 unit tests + 4 integration tests in `tests/
  round78_prof.rs`; total crate-level tests rose from 775 to 803.
  Affine merge / AMVP candidate list construction, affine sub-block
  chroma MC (§8.5.5.9 eqs. 876 – 879 `mvAvgLX`), and the §8.8.3.4
  sub-block boundary deblock propagation remain deferred.

- Round 65 — **Affine sub-block motion compensation scaffold** per VVC
  §8.5.5.9 (eqs. 847 – 887) + Tables 30 / 31 / 32 (affine-mode luma
  1/16-pel interpolation filters). VVC supports 4-parameter and
  6-parameter affine motion in addition to translational motion;
  rounds 21 – 64 covered only the translational path. Round 65 lands
  the affine decoder primitives a future CTU walker will call once
  affine-flagged CUs come online. New `affine` module exposes:
  - `MotionModel` enum — `Translational` / `Affine4Param` /
    `Affine6Param`, the §7.4.10.5 Table 15 `MotionModelIdc` triple.
  - `AffineCpmvs` — control point MV record (2 CPMVs for
    4-parameter, 3 for 6-parameter); constructors `new_4param` /
    `new_6param`; `is_translational` degeneracy detector.
  - `derive_subblock_mvs(cb_w, cb_h, cpmvs, bipred)` — the §8.5.5.9
    sub-block MV array derivation. Implements eqs. 850 – 875: the
    `(mvScaleHor, mvScaleVer)` base, the four affine partials
    `(dHorX, dVerX, dHorY, dVerY)` with the 4-parameter similarity
    constraint `dHorY = -dVerX, dVerY = dHorX` (eqs. 856 / 857), per
    sub-block centre sampling at `(xPosCb, yPosCb) = (2 + 4*sbIdxX,
    2 + 4*sbIdxY)` (eqs. 870 / 871), §8.5.2.14 `>> 7` signed-
    magnitude rounding, and the eqs. 874 / 875 `Clip3(-2^17,
    2^17 - 1, ·)` final clip. Outputs a `SubblockMvGrid` with
    `numSbX = cbW >> 2` × `numSbY = cbH >> 2` per-4×4 MVs.
  - `fallback_mode_triggered(cb_w, cb_h, cpmvs, bipred)` — the
    §8.5.5.9 eqs. 858 – 867 `fallbackModeTriggered` threshold (bxWX4
    * bxHX4 ≤ 225 under bi-pred or per-axis 165 under uni-pred);
    when triggered the per-sub-block grid collapses to a single
    CU-centre MV per eqs. 868 / 869.
  - `AFFINE_LUMA_FILTER_SET_0` / `_1` / `_2` — Tables 30 / 31 / 32
    (the spec's affine-mode luma 1/16-pel interpolation filter
    families). All three coefficient sets sum to 64 per the
    §8.5.6.3 separable-filter normalisation; Set0's row 0 is the
    integer-pel sentinel, Set1 / Set2 row 0 are non-trivial spec
    filter rows used by §8.5.6.3.2's affine-mode integer-position
    path. `AffineLumaFilterSet::table()` returns the static
    coefficient table.
  - `predict_luma_subblock_affine(dst, dst_x, dst_y, sb_w, sb_h, src,
    mv, filter_set)` — separable 8-tap MC helper for one affine
    sub-block. Picture-edge clamping, `shift1 = 0` for BitDepth 8,
    `shift2 = 6`, §8.5.6.6.2 `(v + 32) >> 6 → u8` uni-pred clamp,
    matching the existing translational `inter::predict_luma_block`
    semantics but with caller-selected affine luma filter table.
    Integer-pel + Set0 fast path collapses to a memcpy.
  - `predict_luma_block_affine(dst, dst_x, dst_y, cb_w, cb_h, src,
    cpmvs, filter_set)` — full-CU driver. Walks the 4×4 sub-block
    grid from `derive_subblock_mvs` and dispatches each sub-block
    through `predict_luma_subblock_affine`. The future CTU walker
    will call this for affine inter CUs.
  - **Headline measurement (zoom fixture):** 6-parameter affine
    reconstructs a synthetically zoomed reference at PSNR_Y =
    **53.75 dB** on a 32×32 CU while the best 5×5 translational MV
    search caps at **42.88 dB** — a **+10.87 dB** improvement.
  - **Headline measurement (shear fixture):** 6-parameter affine
    reconstructs a horizontally sheared reference at PSNR_Y =
    **52.32 dB** while the best 7×7 translational MV search caps at
    **34.88 dB** — a **+17.44 dB** improvement.
  - 14 unit tests in `affine::tests` (filter-table normalisation,
    `MotionModel` ↔ `MotionModelIdc` mapping, identity-CPMV
    degeneracy on both 4 / 6-param paths, shear monotonicity in x and
    y, spec-exact centre-sampling values for a pure horizontal-shear
    4-parameter affine, sub-8×8 floor rejection, fallback threshold
    on a small CPMV delta, filter-set table dispatch, integer-pel
    Set0 copy, translational `predict_luma_block_affine` byte-
    identical to translational MC) + 3 integration tests in
    `tests/round65_affine_subblock.rs` (zoom fixture clears +3 dB
    delta, shear fixture clears +3 dB delta + 25 dB floor, identity
    CPMV byte-identical to translational at int-pel).
  - Deferred to later rounds: affine merge / AMVP candidate list
    construction (§8.5.5.6 / §8.5.5.7 / §8.5.5.8), affine sub-block
    chroma MC (eqs. 876 – 879 `mvAvgLX`), PROF (§8.5.5.10), and the
    affine flag propagation into the §8.8.3.4 sub-block boundary
    deblock.

- Round 64 — **Decoder-side Motion Vector Refinement (DMVR)** per VVC
  §8.5.3.2.4 / §8.5.3.2.5. New `dmvr` module exposes:
  - `dmvr_used_flag(...)` — the §8.5.3.2.4 step-1 gating bullet list
    (SPS/PH flags + merge + bi-pred + symmetric-POC bracketing + STRP
    refs + translational motion + non-subblock merge + no
    CIIP/BCW/WP + block size ≥ 8×8 with `cb_w*cb_h >= 128` + luma
    only).
  - `bilateral_matching_sad(predL0, predL1, w, h)` — the §8.5.3.2.5 BM
    cost (full-grid SAD).
  - `refine_mv_pair(mv_l0_init, mv_l1_init, w, h, predictor)` — 2-pass
    refinement: a 5×5 integer-pel search (`(δx,δy) ∈ {-2..2}²`) using
    the spec's opposite-direction pairing `(MV0 + δ, MV1 − δ)`,
    followed by a 3-point parabolic half-pel pass on each axis
    (clamped to ±½-pel ⇒ ±8 in 1/16-pel units; non-convex
    neighbourhoods skip the refinement).
  - `apply_dmvr(...)` — convenience driver that combines the above
    with a `PlanePairPredictor` adapter over the existing
    `crate::inter::predict_luma_block`.
  - On the symmetric-bipred regression fixture (3 luma planes where L0
    and L1 are shifted by ±1 sample around a hidden "truth" plane,
    integer-pel BM optimum at `δ = (-1, 0)`) DMVR converges to exactly
    that delta and the resulting bi-pred MC reaches PSNR_Y = ∞ dB
    (byte-exact reconstruction of truth), starting from a DMVR-off
    baseline of 52.39 dB on the same 16×16 CU — far past the +1.5 dB
    headline target.
  - 12 unit tests in `dmvr::tests` (gating bullets, BM SAD, parabolic
    estimator edge cases, refinement convergence on a synthetic
    shift, refined-vs-baseline SSE) + 5 integration tests in
    `tests/round64_dmvr.rs` (headline PSNR delta, search-range
    constant, three gate-off cases).

- Round 63 (Goal A) — explicit weighted bi-prediction on the B-slice
  encoder + decoder per VVC §8.5.6.5 eq. 994 + §7.4.7.7
  (`pred_weight_table()`). Rounds 60/61 carried only the §8.5.6.4
  default-average bi-pred form `(predL0 + predL1 + 1) >> 1`. Round 63
  (Goal A) lets the encoder estimate slice-level luma weights/offsets
  from the per-list mean luma offsets to the current source frame and
  emit them as a `pred_weight_table` in the slice header; per-CU it
  then probes BOTH the unweighted and weighted forms and picks the
  lower-SSE one, emitting a 1-bit `use_weighted_bi` selector after
  the BI MVD chain. The decoder mirrors this — reads the WP table
  from the slice header (when present) + reads the per-BI-CU
  selector and dispatches to `weighted_bi` (§8.5.6.5 eq. 994 in
  `((p0 * w0 + p1 * w1 + ((o0 + o1 + 1) << log2WD)) >> (log2WD + 1))`
  with 8-bit clip) or `average_bi` accordingly. New `BSliceHeader`
  field `pred_weight_table: Option<PredWeightTable>` (one luma
  record per list — chroma weighting deferred); new public
  `PredWeightTable` struct with `(log2_weight_denom_y, w_l0_y, w_l1_y,
  o_l0_y, o_l1_y)`. Wire format: 1-bit `wp_present_flag` after the
  existing slice-header tail, followed by `ue(log2_weight_denom_y) +
  se(delta_w_l0) + se(o_l0) + se(delta_w_l1) + se(o_l1)` per the
  spec's "delta from default" representation. Backwards-compatible
  with the round-60/61/62 wire bit-for-bit when `wp_present_flag = 0`
  (one extra zero bit per slice header).
  - On a fade fixture (3-frame B-slice with checker-tile content where
    L0 = curr - 20 and L1 = curr - 40 luma offset, so the unweighted
    average sits 30 luma below curr and any uni-pred sits 20-40 luma
    away) the encoder picks weighted-BI on every block, recovers the
    fade exactly, and reaches PSNR_Y = inf (perfect bit-for-bit
    reconstruction since the chosen prediction matches curr exactly
    and the residual is zero on every block). The headline target
    was ≥ 58 dB.
  - The encoder + decoder also gain a `search_range == 0` opt-out for
    sub-pel ME refinement (the round-59/61 8 + 8 fractional-pel probe
    around the integer-pel best). When a caller passes
    `search_range = 0` to `encode_b_slice` / `encode_p_slice`, the
    encoder now skips both the integer-pel full-search AND the
    sub-pel refinement (previously the sub-pel probe still ran). This
    isolates the new weighted-BI test from sub-pel filter ringing on
    sharp tile boundaries and makes `search_range = 0` mean exactly
    "no motion search, MV = MVP". The round-58/59/60/61/62
    regressions all use `search_range = 8` so this is a no-op for
    them.
  - 5 new unit tests in `encoder_inter::tests` (slice header round-
    trip with WP set, default-shape WP equals simple average across
    `log2_wd ∈ 0..=5`, constant-offset weighted-BI recovers target,
    estimator returns `None` when means match, estimator picks
    per-list mean offsets) + 3 new integration tests in
    `tests/round63_weighted_bipred.rs` (fade fixture clears 58 dB,
    no-fade fixture leaves WP off and remains compatible with
    round-60/61/62 wire, decoder byte-identical roundtrip on the
    weighted-BI path).

- Round 63 (Goal B) — chroma sub-pel motion compensation for the
  P-slice and B-slice encoder + decoder
  (`encoder_inter::encode_p_slice_multi_ref` /
  `encoder_inter::encode_b_slice_multi_ref` and the matching
  `decode_*` calls). Rounds 58/60/61 carried luma 8-tap sub-pel MC
  (§8.5.6.3.2 Table 27) but the chroma planes were passed through from
  L0[0] unchanged. Round 63 wires the §8.5.6.3.4 4-tap chroma
  interpolation filter (Table 28) so chroma half-pel + quarter-pel
  positions get spec-correct interpolation, reusing the luma 1/16-pel
  MV (predict_chroma_block applies the 4:2:0 mapping internally —
  chroma 1/32-pel offset is `mv & 31`, integer chroma offset is
  `mv >> 5`). New `encoder_inter::mc_predict_chroma_subpel` /
  `mc_predict_chroma_subpel_bi` helpers wrap the existing
  `crate::inter::predict_chroma_block` with the same `(cx, cy)`-folded
  source-position derivation that `mc_predict_subpel` uses for luma.
  The P-slice, B-slice {L0, L1, BI} dispatches now write
  motion-compensated Cb + Cr into `rec.cb` / `rec.cr` instead of the
  L0[0] pass-through; the decoder mirrors this so encoder + decoder
  chroma are byte-identical. Wire format unchanged (no new bins —
  chroma reuses the per-CU luma MVs already on the wire).
  - On a half-pel translation fixture (band-limited horizontal
    sinusoid in luma + chroma at different frequencies, 64×64 luma
    with 32×32 4:2:0 chroma) the P-slice path reaches PSNR_Cb =
    49.96 dB / PSNR_Cr = 50.39 dB while luma still hits the round-59
    half-pel ceiling at PSNR_Y = 51.57 dB. Both chroma channels clear
    the 45 dB headline target with ~5 dB of headroom. The B-slice
    path on the same fixture (with matched references so BI
    degenerates to uni-pred) hits the same chroma quality.
  - 4 new unit tests in `encoder_inter::tests` (zero-MV chroma MC is
    identity, integer-MV chroma MC translates the block exactly,
    half-pel chroma MC preserves a constant DC plane across all 32×32
    fractional positions, BI chroma helper averages per-list
    predictions per §8.5.6.4) + 5 new integration tests in
    `tests/round63_chroma_subpel.rs` (P-slice half-pel chroma ≥ 45 dB,
    P-slice chroma decoder byte-identical, B-slice half-pel chroma
    ≥ 45 dB, constant-chroma plane preserved through sub-pel,
    integer-pel chroma recovers perfectly).

- Round 62 — multi-reference DPB on the P-slice and B-slice
  encoder + decoder (`encoder_inter::encode_p_slice_multi_ref` /
  `encoder_inter::decode_p_slice_multi_ref` /
  `encoder_inter::encode_b_slice_multi_ref` /
  `encoder_inter::decode_b_slice_multi_ref`). Rounds 58/60/61 carried
  a SINGLE reference per list; round 62 extends each list to hold up
  to `MAX_REF_PICS = 4` pictures (matches the mainstream profile
  constraint in §A.4; the wire-side truncated-unary encoding scales
  to whatever active count the slice header advertises so the
  constant is the encoder/test ceiling, not a wire-format limit). The
  slice header now carries real `num_ref_idx_l0_active_minus1` and
  (for B) `num_ref_idx_l1_active_minus1` values (§7.4.4.2); per-CU
  `ref_idx_l0` / `ref_idx_l1` are emitted as truncated-unary
  (§9.3.3.7 / Table 132 — collapsed to bypass coding for the scaffold,
  matching the round-58/60 mvd "all bypass for the magnitude"
  pattern) with `cMax = num_active - 1`. Encoder ME now iterates every
  candidate reference in each list, runs the round-58 integer-pel SAD
  full search plus round-59 sub-pel refinement against each
  (§8.5.6.3.2 Table 27 8-tap luma filter), and picks the cheapest-SAD
  reference index per list before the existing Lagrangian RDO over
  `{L0, L1, BI}` runs for B-slices. Single-ref `encode_p_slice` /
  `encode_b_slice` / `decode_p_slice` / `decode_b_slice` are now thin
  wrappers over the multi-ref variants; the round-58/60 single-ref
  wire is bit-for-bit unchanged. `PreparedCu::InterPSlice.ref_idx`
  and `PreparedCu::InterBSlice.ref_idx_l{0,1}` now carry the actual
  selected index (previously always 0). Weighted bi-pred and
  chroma sub-pel MC remain deferred.
  - On a 3-frame P-slice fixture (I, P, current) where the current
    frame matches frame 0 better than frame 1, the encoder selects
    `ref_idx=1` (frame 0) on every block and reconstructs to
    PSNR_Y = 58.41 dB (vs. the L0[0]-only path at substantially
    worse PSNR). On a 4-frame B-slice fixture with 2 refs in each
    list (L0=[-2 px, -4 px], L1=[+2 px, +4 px], current = un-shifted)
    the multi-ref-aware RDO splits the translation between L0[0]
    and L1[0] for an exact BI average (PSNR_Y = inf). The round-58
    single-ref 4-px P-slice regression holds at 78.23 dB; a forced
    `ref_idx > 0` fixture (noisy L0[0], perfect L0[1]) round-trips
    byte-identically with PSNR_Y = inf. Multi-ref + sub-pel still
    reaches the round-59 quarter-pel ceiling at 52.39 dB.
  - 3 new unit tests in `encoder_inter::tests` (truncated-unary
    `ref_idx` round-trip across all sizes, single-active-list emits
    no bins, cap clamping) + 6 new integration tests in
    `tests/round62_multi_ref_dpb.rs` (3-frame P-slice prefers
    better ref ≥ 50 dB, 4-ref B-slice ≥ 50 dB, single-ref
    regression ≥ 78 dB, `ref_idx > 0` forced round-trip ≥ 70 dB,
    multi-ref sub-pel ≥ 48 dB, B-slice multi-ref byte-identical
    decoder roundtrip).

- Round 61 — sub-pel motion estimation on the B-slice
  (bi-prediction) encoder + decoder
  (`encoder_inter::encode_b_slice` / `encoder_inter::decode_b_slice`).
  Round 60 added the B-slice scaffold with integer-pel ME only. Round
  61 extends each per-list ME with the same two-stage refinement
  round 59 added to the P-slice path: after the integer-pel SAD full
  search, probe 8 half-pel neighbours through the §8.5.6.3.2 Table 27
  8-tap luma filter (`hpelIfIdx == 0`) via `predict_luma_block`, then
  8 quarter-pel neighbours around the best half-pel candidate. Both
  L0 and L1 lists are refined independently; the RDO over
  `{L0-only, L1-only, BI}` then runs with each list's MV at
  1/16-pel precision. Bi-pred reconstruction is still the §8.5.6.4
  simple average `pred = (predL0 + predL1 + 1) >> 1` (weighted
  bi-pred remains deferred). The decoder is unchanged on the wire
  side — `mc_predict_subpel` is the single per-list MC path and
  already handles sub-pel MVs. Multi-reference DPB (more than one
  picture per list), chroma sub-pel MC, and weighted bi-pred remain
  deferred.
  - On a half-pel B-slice translation fixture with matched
    references (`ref_l0 == ref_l1` so BI degenerates to uni-pred)
    PSNR_Y reaches 51.57 dB (matching the round-59 P-slice
    half-pel ceiling). On a ¼-pel B-slice translation fixture
    PSNR_Y reaches 52.39 dB. On a split-translation fixture
    (L0=+2 px / L1=-2 px with the current frame mid-way) the RDO
    picks BI and reconstructs the current frame essentially exactly
    (PSNR_Y = inf). The round-60 4-px integer-pel B-slice fixture
    still hits 78.23 dB (no regression).
  - 5 new integration tests in `tests/round61_bslice_subpel.rs`
    (half-pel B ≥ 50 dB, quarter-pel B ≥ 48 dB, BI-split-translation
    ≥ 50 dB with byte-identical decode roundtrip, integer-pel
    B-slice regression ≥ 70 dB, sub-pel B-slice decoder
    byte-identical).

- Round 60 — B-slice (bi-prediction) encoder + decoder scaffold
  (`encoder_inter::encode_b_slice` / `encoder_inter::decode_b_slice`).
  The B-slice path mirrors the round 58 / 59 P-slice path but threads
  TWO reference lists (L0 + L1, one picture per list in this round).
  For each 4×4 luma block the encoder runs integer-pel full-search
  SAD against both refs independently (§7.4.7.3 — MVP derivation
  runs per list), forms three candidate predictions (L0-only,
  L1-only, BI), and picks the cheapest via Lagrangian SSE + small
  per-mode bias. Bi-prediction reconstruction is the §8.5.6.4
  simple average `pred = (predL0 + predL1 + 1) >> 1` (weighted
  bi-pred is deferred). The slice header carries
  `num_ref_idx_l0_active_minus1` AND `num_ref_idx_l1_active_minus1`
  per §7.4.4 with `slice_type == B`. On the wire the per-CU
  `inter_pred_idc` per §7.4.7.2 is bypass-coded as `{bi_flag, list}`
  (PRED_L0, PRED_L1, PRED_BI); per-active-list MVDs reuse the
  round-58 EG1 `abs_zero_flag + magnitude + sign` shape. A custom
  `OXAV_VVC_BSLIC` magic-prefixed wire chunk wraps the slice-header
  bit prelude + CABAC payload; `decode_b_slice` is the matching
  in-crate decoder. On a 4-px translation fixture with both refs
  pointing at the same picture the B-slice degenerates to P-slice
  quality (PSNR_Y = 78.23 dB); on a split-translation fixture
  (L0=+2, L1=-2) the RDO picks the better uni-pred path and reaches
  54.15 dB matching the corresponding P-slice. `PreparedCu` gains
  the `InterBSlice` variant alongside `InterPSlice`.
  - Multi-reference DPB (more than one picture per list) is
    deferred; sub-pel ME on the B-slice path is deferred to round
    61 (integer-pel ME only here); chroma sub-pel MC and weighted
    bi-pred remain deferred.
  - 3 new unit tests in `encoder_inter::tests` (B-slice header
    round-trip, `inter_pred_idc` round-trip across all three
    enumerants, `average_bi` matches §8.5.6.4) + 5 new integration
    tests in `tests/round60_bslice_basic.rs` (same-refs degenerate
    to P-slice ≥ 35 dB, split-translation reaches at-least-P-slice
    PSNR, encoder/decoder byte-identical roundtrip, P-slice 78 dB
    regression holds, no-motion zero-residual roundtrip).

- Round 59 — sub-pel motion compensation for the P-slice
  encoder + decoder (`encoder_inter::encode_p_slice` /
  `encoder_inter::decode_p_slice`). The integer-pel SAD full search
  from round 58 is now followed by a two-stage refinement:
  8 half-pel offsets around the integer-pel best, then 8 quarter-pel
  offsets around the half-pel best. All sub-pel candidates are
  evaluated through the spec §8.5.6.3.2 Table 27 8-tap luma
  interpolation filter (`hpelIfIdx == 0`) via the existing
  `crate::inter::predict_luma_block`. The on-wire `MvdLX` magnitudes
  per §7.4.7.2 / §9.3.3.7 now carry 1/16-luma-sample units (¼-pel
  ⇒ |mvd|=4, half-pel ⇒ 8, full-pel ⇒ 16); the CABAC schema is
  unchanged (bypass `abs_zero_flag` + EG1 magnitude + sign). The
  decoder reproduces the encoder's reconstruction byte-for-byte
  even at fractional MVs. On the round-58 4-px integer-pel
  translation fixture, PSNR_Y holds at 78.23 dB (no regression);
  on a band-limited oversampled fixture, ½-pel reconstructs to
  51.6 dB and ¼-pel to 52.4 dB. AMVR, `hpelIfIdx` selection, the
  full 1/16-pel exhaustive search, and chroma sub-pel MC are
  deferred to later rounds (chroma stays pass-through from the
  reference per the round-58 scaffold).
  - 6 new tests in `encoder_inter::tests` (half-pel +
    quarter-pel PSNR floors, integer-pel regression, byte-identical
    decoder roundtrip at sub-pel MVs, sub-pel refinement returns
    integer optimum when SAD already 0, MVD round-trip sub-pel
    magnitudes).
  - 4 new integration tests in `tests/round59_subpel_basic.rs`
    (half-pel ≥ 30 dB, quarter-pel ≥ 30 dB, integer-pel
    regression ≥ 70 dB, sub-pel decoder byte-identical).

- Round 58 — inter-frame P-slice encoder + decoder scaffold
  ([`encoder_inter::encode_p_slice`] / [`encoder_inter::decode_p_slice`]).
  Single-reference DPB (one L0 picture), integer-pel full-search motion
  estimation (`±N` SAD on 4×4 luma blocks per VVC §7.4.10 minimum-PU
  size), spatial MVP picker (left → above → zero per §7.4.7.3),
  per-block CABAC bin emit covering `cu_skip_flag` (§7.4.10),
  `general_merge_flag` (§7.4.10), `inter_pred_idc` (PRED_L0,
  bypass), `ref_idx_l0` (single ref → 0), `mvd_coding(mvd_x, mvd_y)`
  (§7.4.7.2 / §9.3.3.7 — bypass `abs_zero_flag` + EG1 magnitude +
  `mvd_sign_flag`), `tu_y_coded_flag` (§7.4.10), and luma residual
  coefficients via the existing
  `crate::residual_enc::encode_tb_coefficients`. A custom
  `OXAV_VVC_PSLIC` magic-prefixed wire chunk encapsulates the
  slice-header bit prelude (`slice_type` / `slice_pic_order_cnt_lsb`
  per §7.4.4.2.2 / `num_ref_idx_l0_active_minus1` / `slice_qp_delta`)
  + the per-block CABAC payload; `decode_p_slice` is the matching
  in-crate decoder. Achieves PSNR_Y 78.23 dB on the 4-px horizontal
  translation fixture; encoder + decoder roundtrip is bit-identical.
  - Sub-pel / fractional MVs deferred to round 59.
  - Bi-pred and B-slice plumbing deferred (PRED_L0 only).
  - Multi-reference DPB deferred (one L0 picture only).
  - Wire format is in-crate (not Annex B NAL); the full
    `encode_idr_with_residuals_cfg` IDR pipeline does not yet thread
    multi-frame DPB plumbing.
  - 9 new tests in `encoder_inter::tests` + 4 integration tests in
    `tests/round58_pslice_basic.rs` (translation PSNR ≥ 35 dB,
    no-motion roundtrip, byte-identical encoder/decoder roundtrip,
    synthetic moving-square 2-frame fixture).

- Round 57 — MTT TT picker RDO (opt-in via `EncoderConfig::enable_mtt_tt_picker`),
  parallel to round 56's BT picker. For each 64×64 candidate CU the picker
  additionally evaluates `TT_VERT` (1:2:1 three-column 16×64 / 32×64 / 16×64
  split) and `TT_HORZ` (1:2:1 three-row 64×16 / 32×16 / 64×16 split) per VVC
  §7.3.10.4 / §7.4.10.4 and picks the lowest-cost option on
  `cost = SSE_Y + λ·bits` over the union `{leaf, BT_VERT, BT_HORZ,
  TT_VERT, TT_HORZ}` when both flags are on. The chosen sub-CUs route
  through the same `prepare_leaf_cu` / `emit_tu_with_cbf` /
  `accumulate_deblock_cus` / `record_cu_into_map` paths as BT-split
  sub-CUs (new `PreparedCu::TtSplit` variant; `tt_parent_dims` recovers
  the parent dim from any sub-CU's leaf dim × 4 along the split axis).
  The wire-side syntax emit reuses the existing round-55
  `crate::syntax_enc::encode_coding_tree_tt_split` helper, which emits
  `split_cu_flag = 1`, `split_qt_flag = 0`, `mtt_split_cu_vertical_flag`,
  `mtt_split_cu_binary_flag = 0`, then the three child sub-CUs.
  - Encoder-only change; decoder TT syntax was already parseable per
    round 55.
  - Default `enable_mtt_tt_picker = false` reproduces the round-56
    bitstream byte-for-byte.
  - On a 3-stripe vertical fixture with thin (16-col) outer stripes at
    QP 32, the TT picker drops leaf-only SSE from 2576 to 0 (perfect
    reconstruction, vs. PSNR_Y 56.17 dB baseline).
  - 6 new tests in `encoder_pipeline::tests`.

## [0.0.6](https://github.com/OxideAV/oxideav-h266/compare/v0.0.5...v0.0.6) - 2026-05-09

### Other

- round 56 — MTT BT picker RDO (opt-in) + multi-row CU neighbour tracking
- round 55: forced QT split for 128×128 CTBs + MTT BT/TT split syntax
- round 54: wire round-53 helpers into encoder pipeline + SAO merge-flag CABAC emit
- round 53: rustfmt fixup
- round 53: alf_luma_clip_idx joint coeff/clip RDO + chroma SAO merge (opt-in)
- round 52 — coding_tree() / coding_unit() syntax shells + per-CU cu_qp_delta emit
- round 51 — explicit tu_*_coded_flag CABAC emit + chroma APS / CC-ALF APS RDO trade-off
- round 50 — chroma SAO RDO + apply (Cb / Cr) + per-picture CABAC bin cost in the APS-vs-fixed RDO
- round 49 — chroma residual emit (forward DCT + flat quant + CABAC for Cb / Cr)
- round 48 — per-class luma ALF Wiener design + APS-vs-fixed picture-bits trade-off
- round 47 — encoder ALF luma APS design + APS-signalled-set RDO
- round 46 — ALF wired end-to-end through SPS, PH and per-CTU CABAC
- round 45 — ALF per-CTU CABAC syntax emit/decode + ALF APS RBSP emit
- round 44 — encoder primary chroma ALF on/off + alt-filter RDO
- round 43 — CC-ALF integration into the IDR encoder pipeline
- round 42 — encoder CC-ALF per-CTB filter-selection RDO
- round 41 — encoder ALF filter-set RDO over all 16 fixed filter sets
- round 40 — GPM (geometric partitioning) decoder + AMVR helpers + encoder ALF RDO
- drop stale REGISTRARS / with_all_features intra-doc links
- drop needless &src re-borrow at sao_decide_picture call
- drop dead `linkme` dep
- h266 r35: residual emit + SAO encoder + IDR pipeline (PSNR ≥ 30 dB)
- auto-register via oxideav_core::register! macro (linkme distributed slice)

### Added

- **Round 56 — MTT BT picker RDO (opt-in) + multi-row CU neighbour tracking for §9.3.4.2 ctxInc** — closes the two long-deferred encoder + decoder gaps from round 55: (a) the `encode_coding_tree_bt_split` syntax helper landed in r55 was unreachable from `encode_idr_with_residuals` (the picker stayed leaf-or-QT-only); (b) `TreeWalker.nbrs.left/above_avail` were hard-coded `false` regardless of CTU position, biasing every `split_cu_flag` / `split_qt_flag` / `mtt_split_*` ctxInc derivation. **#2 BT picker (opt-in via new `EncoderConfig::enable_mtt_bt_picker`)**: for every full 64×64 luma CU candidate the encoder now evaluates three options — `Leaf` (one 64×64 TB), `BT_VERT` (two 32×64 sub-CUs), `BT_HORZ` (two 64×32 sub-CUs) — and picks the lowest-cost option per `cost = SSE_Y + λ * bits` (`bits` includes the `split_cu_flag` / `split_qt_flag` / `mtt_split_*` syntax bins via a fresh `measure_cu_bits` pass plus the residual coefficient bins). The picker uses `prepare_luma_tb` / `prepare_chroma_tb` extended to take independent `n_tb_w` / `n_tb_h` (was square-only) so non-square TBs run through the existing `forward_dct_ii_2d` (already non-square-capable) + `quantize_tb_flat` (already non-square-capable when both dims are powers of two) pipeline. New `PreparedCu` enum (`Leaf` / `BtSplit { dir, cqt_depth, mtt_depth, sub_a, sub_b }`) replaces the per-CTU flat `Vec<PreparedLumaTb>` so the second-pass CABAC walk can recurse into BT splits via the new `emit_prepared_cu` helper which dispatches to `encode_coding_tree_bt_split` (round-55) for split CUs and `encode_coding_tree_leaf_iframe` for leaves. The picker's `region_sse_y` measures luma SSE inside the candidate region; restoring the chosen option's reconstruction (luma + Cb + Cr) is done from a snapshot taken before each trial. Default `enable_mtt_bt_picker = false` keeps the round-55 wire stream byte-for-byte. **#4 Multi-row CU neighbour tracking**: new `coding_tree::CuNeighbourMap` is a picture-wide grid of CU descriptors (`cb_w`, `cb_h`, `cqt_depth`) at 4-sample granularity. `TreeWalker::with_neighbour_map(map)` (and `with_neighbour_map_rebased(map, base_x, base_y)` for the per-CTU walker which operates in CTU-local coords) wires the map into the §9.3.4.2 ctxInc derivations: `ctx_inc_split_cu_flag` reads `(left_avail, above_avail, cb_height_left, cb_width_above)` from the map (was `(false, false, 0, 0)` round-55), `ctx_inc_split_qt_flag` reads `(cqt_depth_left, cqt_depth_above)`, and `ctx_inc_mtt_split_cu_vertical_flag` reads the same six fields. Each leaf CU's commit inserts its descriptor into the map so look-ups inside the same CTU see siblings emitted earlier in the walk; cross-CTU look-ups see the previous CTU's right-edge / bottom-edge CUs. The decoder side mirrors the encoder: `CtuWalker` carries a `CuNeighbourMap` populated as each CTU's `decode_ctu_partitions` runs through the rebased walker. Encoder pipeline mirrors via the same struct (named `CuStateMap` to keep encoder-only types out of the public `coding_tree` API surface) — `cu_map` is the read-only snapshot the current CTU sees, `cu_map_write` accumulates this CTU's writes, both committed at end-of-CTB so siblings inside the same CTB don't race the spec's neighbour-availability rules. New tests cover (a) `round56_default_config_matches_round55` — both flags off produces a byte-identical bitstream + reconstruction to `encode_idr_with_residuals`; (b) `round56_mtt_bt_picker_runs_and_clears_psnr_floor` — BT picker on a 64×64 gradient runs to completion + clears the round-55 30 dB luma floor; (c) `round56_mtt_bt_picker_improves_sse_on_horizontal_edge` — synthetic 128×128 frame with a horizontal edge at every 32-line midpoint sees the BT_HORZ split picked: SSE_Y goes from 1024 (leaf-only baseline) to 0 (BT picker on), PSNR_Y from 60.17 dB to ∞; (d) `round56_cu_state_map_round_trips_descriptor` — encoder map insert + lookup round-trips a 32×64 CU descriptor; (e) `round56_build_tree_neighbours_packs_descriptors` — picture-edge default + populated-neighbour cases pack the descriptor correctly; (f) `round56_neighbour_state_drives_split_cu_ctx_inc` — `ctx_inc_split_cu_flag` returns 6 for picture-edge defaults vs 8 when both neighbours are smaller, demonstrating the new ctxInc accuracy; (g) `round56_multi_ctb_neighbour_state_runs_to_completion` — 256×128 multi-CTB encoder pipeline run completes + clears 30 dB; (h) `cu_neighbour_map_round_trips_descriptor` — public `CuNeighbourMap::neighbour_state` packs the six-field tuple correctly; (i) `tree_walker_populates_neighbour_map_on_leaf` — `TreeWalker::with_neighbour_map` inserts each emitted leaf into the map; (j) `tree_walker_rebased_inserts_picture_absolute` — rebased variant adds `(base_x, base_y)` to inserts so per-CTU walkers populate the picture-absolute map correctly.
- **Round 55 — forced QT split for 128×128 CTBs + MTT BT/TT split syntax + ctxIdx tables 61 / 62 + §9.3.4.2.3 ctxInc** — closes the long-deferred §7.3.11.4 spec-shape gap from round 52: every 128×128 CTB on the wire now correctly emits `split_cu_flag = 1` + `split_qt_flag = 1` ahead of its four 64×64 sub-CU shells (round 52 silently elided this and treated each TB as a root CU, which violated the spec because `MaxTbSizeY = 64 < CtbSizeY = 128`). New `tables::SyntaxCtx::MttSplitCuVerticalFlag` (Table 61, 15 ctxIdx — 5 per initType per Table 51) + `MttSplitCuBinaryFlag` (Table 62, 12 ctxIdx — 4 per initType) wire the §9.3.2.2 init values into the existing `init_contexts` plumbing. New `ctx::ctx_inc_mtt_split_cu_vertical_flag` implements the §9.3.4.2.3 derivation (asymmetric BT/TT allowance → ctxInc 4 / 3, otherwise the eq. 1553 / 1554 dA / dL aspect-ratio compare against the L / A neighbours), and `ctx::ctx_inc_mtt_split_cu_binary_flag` implements the Table 132 closed form `2 * mtt_split_cu_vertical_flag + (mttDepth <= 1 ? 1 : 0)`. `coding_tree::TreeCtxs` gains `mtt_split_vertical` + `mtt_split_binary` arrays initialised from the new tables; the existing `TreeWalker` MTT split path now reads these bins through their proper contexts (was bypass — a round-49 placeholder). New `syntax_enc::encode_coding_tree_bt_split(enc, ctxs, cb_w, cb_h, nbrs, cqt_depth, mtt_depth, dir, body)` emits the four-flag preamble (`split_cu_flag = 1`, `split_qt_flag = 0`, `mtt_split_cu_vertical_flag` per `MttSplitDir`, `mtt_split_cu_binary_flag = 1`) followed by two equal-size sub-block closure invocations; `encode_coding_tree_tt_split` is the TT skeleton (same preamble but `binary_flag = 0` and three sub-blocks at the 1:2:1 ratio). Decoder duals: `decode_coding_tree_split_qt_flag`, `decode_coding_tree_mtt_split_vertical_flag`, `decode_coding_tree_mtt_split_binary_flag`. `encode_coding_quadtree_split` is now wired into `encode_idr_with_residuals` for every 128×128 CTB whose actual extent fully covers the 128×128 region (the picture-edge non-cover case keeps the round-52 single-leaf path); the body closure receives `(enc, ctxs)` so callers can recurse into `encode_coding_tree_leaf_iframe` per quadrant without the `TreeCtxs` borrow ping-pong. Round-55 encoder RDO picker stays leaf-or-no-split for the picture pipeline (TT picker is round-56+); the BT / TT syntax + parse helpers are unit-tested in isolation. Existing round-51 / round-52 CBF + cu_qp_delta round-trip tests updated to consume the forced QT split flag pair (`split_cu_flag = 1` + `split_qt_flag = 1`) ahead of each 128×128 CTB's four 64×64 leaf-CU shells. New tests cover (a) `ctx::tests::mtt_split_cu_vertical_flag_basic` — every branch of §9.3.4.2.3; (b) `ctx::tests::mtt_split_cu_binary_flag_basic` — Table 132 closed form for the four `(vertical_flag, mtt_depth)` combinations; (c) `tables::tests::mtt_split_context_table_lengths` — Table 61 / 62 transcription length sanity (15 / 12 entries); (d) `syntax_enc::tests::coding_tree_bt_split_vertical_round_trips` + `_horizontal_round_trips` — encoder writes the four-flag BT preamble + 2 sub-blocks, decoder reads each back through the matching helper; (e) `syntax_enc::tests::coding_tree_tt_split_vertical_round_trips` — TT skeleton round-trips with the 1:2:1 sub-block ratios; (f) `round55_forced_qt_128x128_reconstruction_psnr_matches_baseline` — 128×128 gradient at QP=26 reconstructs at PSNR_Y ≥ 30 dB (the round-52 baseline floor); (g) `round55_forced_qt_128x128_qp0_high_psnr` — QP=0 round-trip clears 40 dB on the same fixture, confirming the forced QT split adds no reconstruction divergence.
- **Round 54 — wire round-53 helpers into the encoder pipeline + SAO merge-flag CABAC emit pass** — `EncoderConfig` gains two opt-in flags: `enable_alf_clip_rdo` (runs `design_clip_rdo_for_luma_aps` after the per-class luma ALF design and packs the result through `build_per_class_luma_alf_aps_data_with_clip` so the wire APS carries `alf_luma_clip_flag = 1` + the per-tap `alf_luma_clip_idx[ filtIdx ][ j ]` block when at least one tap wins), and `enable_chroma_sao_merge` (runs `apply_chroma_sao_merge` after the per-CTB chroma SAO RDO and emits `sao_merge_*_flag` CABAC bins ahead of the per-CTB SAO syntax for every CTB the merge map flagged). Both default to `false` so the pipeline is wire-identical to round 53 unless callers opt in. New `encode_idr_with_residuals_cfg(src, qp, cfg)` + `encode_idr_with_qp_picker_cfg(src, slice_qp_y, cfg, picker)` expose the EncoderConfig path; the existing `encode_idr_with_residuals` / `encode_idr_with_qp_picker` thin-wrap through with default flags. New `sao_syntax::encode_sao_ctb(enc, ctxs, cfg, sao_pic, merge_map, rx, ry, left_avail, up_avail)` is the encoder mirror of `decode_sao_ctb` per §7.3.11.3 + §9.3.4.2.1 (Table 124 ctxIdx): it consumes the encoder-RDO-picked `SaoCtbParams` + `SaoMergeChoice` and walks the same syntax order the decoder reads — `sao_merge_left_flag` (ctx 0, when `rx > 0 && leftCtbAvailable`), `sao_merge_up_flag` (ctx 0, when `ry > 0 && !left && upCtbAvailable`), then the per-component `sao_type_idx_*` (bin 0 ctx 0, bin 1 bypass) → `sao_offset_abs[]` (TR bypass) → `sao_offset_sign_flag[]` (FL bypass for BO, inferred for EO) → `sao_band_position[]` (FL5 bypass for BO) / `sao_eo_class_*` (FL2 bypass for EO). `recover_raw_offsets` inverts eq. 153 to recover the raw `(abs, sign)` pair from the bit-depth-scaled `SaoOffsetVal[]` array. Encoder-side SPS now flips `sps_sao_enabled_flag = 1` only when `enable_chroma_sao_merge` is on; the matching PH `ph_sao_chroma_enabled_flag = 1` lands in the `AlfPhChain`. SH parser closes the long-deferred §7.4.8 inference: when `sps_sao_enabled_flag && pps_sao_info_in_ph_flag` the SH does not transmit `sh_sao_*_used_flag` and both values are inferred from the matching `ph_sao_*_enabled_flag` (the `PhState` carries the new `ph_sao_luma_enabled_flag` / `ph_sao_chroma_enabled_flag` fields). New tests cover (a) `round54_default_config_matches_round53` — both flags off produces a byte-identical bitstream + reconstruction to `encode_idr_with_residuals`; (b) `round54_alf_clip_rdo_runs_and_does_not_regress` — high-contrast vertical edge IDR runs through the clip RDO path + clears the round-50 30 dB luma floor; (c) `round54_chroma_sao_merge_runs_and_emits_extra_bits` — chroma noise IDR with the flag on produces a strictly-larger bitstream than the no-SAO baseline (SAO PH bits + per-CTB SAO syntax) + clears the round-50 chroma 57 dB floor; (d) `round54_chroma_sao_merge_fires_on_flat_chroma_row` — 4-CTB-wide flat chroma row exercises the merge-left RDO path; (e) `round54_both_flags_on_runs_to_completion` — both knobs on simultaneously runs to completion + clears 30 dB; (f) `round54_chroma_sao_merge_header_chain_round_trips` — full SPS → PPS → PH → SH parse chain with the new SAO inference. Plus 4 new `sao_syntax` tests covering the encoder/decoder round-trip for `NotApplied`, BO with non-trivial offsets, EO, and a 4-CTB merge-left row, and `recover_raw_offsets_inverts_eq_153_8bit`.
- **Round 53 — alf_luma_clip_idx[] joint coefficient/clip RDO + per-CTB chroma SAO merge-left / merge-above (both opt-in)** — closes the long-deferred (since round 49) per-tap luma ALF clipping gap and the chroma SAO merge-bit-cost gap from round 50, both as additive opt-in modules that don't disturb the existing encoder pipeline. New `alf_aps_design::design_clip_rdo_for_luma_aps(src, rec_pre_alf, coeff, ctb_log2_size_y, bit_depth, chroma_format_idc)` performs greedy per-tap descent over the picture-wide SSE_Y under the §8.8.5.2 / Table 8 clip semantics: starting from the round-48 per-class designed coefficients (which it does NOT redesign), the RDO tries clip indices `{1, 2, 3}` for every (filtIdx, j) tap (25 × 12 × 3 = 900 full-picture replays via the existing decoder `apply_alf` pipeline) and adopts whichever value strictly lowers SSE; subsequent taps see the accumulated improvements through the running APS. `build_per_class_luma_alf_aps_data_with_clip(designed)` packs the resulting `(coeff, clip_idx)` pairs into an `AlfApsData` whose `alf_luma_clip_flag` is true iff at least one tap picked non-zero — when all clips stay at zero the wire format is byte-identical to `build_per_class_luma_alf_aps_data` (round 48 emit path). Synthetic high-contrast fixture (4-pixel vertical stripes 16/240 + LCG noise + hand-crafted strong-low-pass row) measures baseline SSE_Y 15.57M vs with-clip 0.69M (ΔPSNR ≈ +13.5 dB; the linear filter would otherwise over-smooth the edges and the clip RDO bounds the (neighbour - curr) deltas). New `sao_enc::apply_chroma_sao_merge(src, rec_pre_sao, independent_sao_pic, bit_depth, ctb_log2_size_y)` mirrors the §8.8.4.1 merge-bit logic: for every chroma CTB, after the per-CTB-independent RDO has picked its (Cb, Cr) params, the routine compares (a) independent (~10-30 bits depending on type), (b) merge-left from rx-1 (1 merge-flag bit + neighbour SSE), (c) merge-above from ry-1 (2 merge-flag bits + neighbour SSE) under cost = SSE + λ × bits with λ = 8. Returns `(SaoPicture, SaoMergeMap)` — the new `sao::SaoMergeMap` records per-CTB `SaoMergeChoice ∈ {Independent, MergeLeft, MergeAbove}`. New tests cover (a) `clip_rdo_flat_picture_keeps_no_clip` — flat-grey input keeps `alf_luma_clip_flag = false`; (b) `clip_rdo_never_increases_sse` — high-contrast picture's RDO post_clip_sse_y ≤ baseline_sse_y; (c) `clip_rdo_aps_round_trips` — designed clip APS round-trips through `emit_alf_aps_rbsp` + `parse_aps`; (d) `clip_rdo_no_clip_falls_back_to_round48_emit` — empty designed struct emits byte-identical bytes to the round-48 emit path; (e) `clip_rdo_high_contrast_edges_psnr_delta_non_negative` — edge-stripe fixture documents the SSE delta in test stderr; (f) `chroma_sao_merge_flat_picture_fires_on_neighbours` — uniform 4-CTB-wide flat picture has merge fire on > 50% of CTBs; (g) `chroma_sao_merge_single_ctb_no_merge` — single-CTB picture has no neighbours so merge_count = 0; (h) `sao_merge_map_set_get_count` — `SaoMergeMap` round-trips set/get and counts merges. The encoder pipeline is left unchanged for round 53 — these modules are additive opt-in helpers callers can wire in front of `encode_idr_with_residuals` for callers that want the extra RDO without committing to the cost on every encode.
- **Round 52 — `coding_tree_unit()` / `coding_tree()` / `coding_quadtree()` / `coding_unit()` syntax shells + explicit `cu_qp_delta` emit when CBFs are non-zero** — closes the long-deferred §7.3.11 / §7.3.12 / §7.3.13 spec-shape gap (every CU is now wrapped in the `coding_tree() → coding_unit()` syntax envelope ahead of `transform_tree()`) and the constant-QP gap (round-51 hard-coded one QP across the picture; round-52 lets the encoder pipeline pick a per-CU QP and signal the §8.7.1 delta on the wire). New `syntax_enc` module exposes `encode_coding_tree_leaf_iframe(enc, ctxs, cb_w, cb_h, nbrs, body_emit)` which writes one `split_cu_flag = 0` ctx-coded bin (round-52 single-CU-per-CTB scope) ahead of a caller-provided `coding_unit()` body; `encode_coding_quadtree_split` is wired for future rounds (`split_cu_flag = 1` + `split_qt_flag = 1` + per-quadrant recursion); `decode_coding_tree_split_cu_flag` is the matching decoder helper. Encoder pipeline now flips `pps_cu_qp_delta_enabled_flag = 1`, emits the §7.3.2.8 PH gate `ph_cu_qp_delta_subdiv_intra_slice = ue(0)` (CTB-level QG granularity), wraps every prepared TB in the new shell, and emits `cu_qp_delta_abs` + `cu_qp_delta_sign_flag` (via existing `write_cu_qp_delta`) when the CU has any non-zero CBF AND `cu_qp_local != prev_qp_in_qg`. Per-quantisation-group state (`QpDeltaState`) tracks the §7.4.13.2 `IsCuQpDeltaCoded` reset (CTB-aligned QG) and the running `prev_qp_in_qg` baseline. New `encode_idr_with_qp_picker(src, slice_qp_y, qp_picker)` exposes per-CU QP control to callers; `encode_idr_with_residuals` is the constant-QP wrapper. CABAC context init for `tree_ctxs` / `residual_ctxs` / `alf_ctxs` is now bound to the slice QP (was hard-coded to 26). New tests: (a) `syntax_enc::tests::coding_tree_leaf_iframe_round_trips_split_cu_flag` — encoder writes `split_cu_flag = 0` + body, decoder reads back the same; (b) `syntax_enc::tests::coding_quadtree_split_round_trips_split_flags` — `split_cu_flag = 1` + `split_qt_flag = 1` round-trip with per-quadrant invocation; (c) `round52_cu_qp_delta_round_trips_per_cu` — picker assigns QP=22 to the flat-luma quadrants and QP=30 to the textured ones; the textured CU's `cu_qp_delta` decodes to 8 and the cumulative QP reaches 30; (d) `round52_constant_qp_path_round_trips_zero_delta` — `encode_idr_with_residuals` constant-QP path emits a zero-delta wire bin per CU with non-zero CBFs and the decoder reads `delta = 0`. Existing round-51 CBF round-trip tests updated to consume the per-CU `split_cu_flag` shell bin ahead of each TB's CBF triplet.
- **Round 51 — explicit `tu_*_coded_flag` CABAC emit + chroma APS / CC-ALF APS RDO trade-off** — closes the implicit-CBF gap deferred since round 47 and extends round 50's APS-vs-fixed bin-cost RDO to the chroma + CC-ALF APSes. The encoder pipeline's residual emit (`emit_tu_with_cbf`) now writes the §7.3.10 `transform_unit()` header in spec order: `tu_cb_coded_flag` → `tu_cr_coded_flag` → `tu_y_coded_flag` (real CABAC bins via `write_tu_*_coded_flag` / §9.3.4.2.5 + Table 127 ctxIdx) followed by the per-component residual blocks luma → Cb → Cr, gated by the matching CBF (round 50 was implicit-CBF: residual emit only when a non-zero level existed and the decoder side inferred CBF from coeff presence). The decoder side already read explicit CBFs through the round-46 leaf-CU walker so this is a pure encoder-side fix. Chroma + CC-ALF APS RDO trade-off mirrors the round-48 / 50 luma trade-off independently per APS: the chroma APS at id 0 carries the §8.8.5.4 primary-chroma filter bank shared by Cb + Cr (one ship/skip decision); CC-ALF Cb and CC-ALF Cr each pick independently against the §8.8.5.7 cross-component filter bank shared by both via APS id 1 (the APS NAL ships when either component picks). RD compare per APS: `sse_off + lambda * bin_cost_off` vs `sse_on + lambda * (8 * aps_byte_cost_share + bin_cost_on)`. Each candidate `AlfPicture` runs through `estimate_alf_picture_bin_cost` against an `AlfSyntaxConfig` whose `cb_enabled` / `cr_enabled` / `cc_cb_enabled` / `cc_cr_enabled` mirror the trial decision so the per-CTB `alf_ctb_flag[1/2]` + `alf_ctb_cc_*_idc[]` bins are folded in on the right side. The `AlfPhChain` struct (new public encoder API) propagates each per-component PH `ph_alf_*_enabled_flag` decision into `emit_picture_header_nal_with_alf_chain_full`, suppressing the §7.4.3.8 `ph_alf_aps_id_chroma` / `ph_alf_cc_*_aps_id` fields by their spec gates when the corresponding enable flag is 0. Wire NAL order: `[chroma APS], [CC APS], [luma APS], PH` (each gated by its trade-off). New tests: (a) `round51_cbf_round_trip_flat_source_emits_zero_cbfs` walks one CTU's CABAC stream after `decode_alf_picture` and decodes 4 TBs of explicit CBFs (Cb, Cr, luma per §7.3.10) — every CBF must be 0 on a flat-grey 128×128 source where all residuals quantise to zero; (b) `round51_cbf_round_trip_non_flat_source_emits_some_nonzero_cbfs` — high-contrast luma + chroma noise yields at least one TB with a non-zero CBF; (c) `round51_chroma_aps_rdo_ships_aps_on_chroma_noise` — pipeline runs end-to-end and every shipped APS round-trips on a high-frequency chroma checkerboard; (d) `round51_chroma_aps_rdo_skips_aps_on_flat_source` — flat-grey 128×128 source drops every APS NAL (chroma + CC + luma) and the PH carries `ph_alf_{cb,cr,cc_cb,cc_cr}_enabled_flag = 0`.
- **Round 50 — chroma SAO RDO + apply (Cb / Cr) + per-picture CABAC bin cost in the APS-vs-fixed RDO** — composes with the round-49 chroma residual emit (which is what gave the chroma planes real reconstructed samples to feed §8.8.4.2 SAO). The pipeline's `apply_sao` call now flips `chroma_used = true` and passes `sao_chroma = true` to `sao_decide_picture` (which already mirrors the per-CTU BO + EO RDO at the 4:2:0 half-resolution chroma grid). Measured PSNR delta vs the round-49 baseline at QP 26 on the existing structured-chroma 128×128 fixture: PSNR_Cb 57.16 → 59.20 dB (+2.04 dB), PSNR_Cr 56.65 → 57.74 dB (+1.09 dB). Round-50 also closes the rate-cost gap in the round-48 APS-vs-fixed picture-bits trade-off: the previous version compared `sse_with_aps + lambda * 8 * aps_byte_cost` against `sse_fixed_only`, missing the per-CTB CABAC bit deltas for the `alf_use_aps_flag` + `alf_luma_*_filter_idx` syntax elements. The two candidate `AlfPicture`s differ in which family of luma filter index they signal: `alf_luma_fixed_filter_idx` is a 4-bit TB bypass while `alf_luma_prev_filter_idx` (with `cMax = N - 1 = 0` when only one APS is referenced) is suppressed entirely, so the fixed-only branch pays ~5 bits per "luma_on" CTB versus ~1 bit in the with-APS branch. New `estimate_alf_picture_bin_cost(alf_pic, cfg, qp)` measures each candidate against a fresh `ArithEncoder` + freshly-init `AlfCtxs`, returning `8 * bytes_emitted` from the finished arithmetic stream — same accounting the wire stream uses. The trade-off now compares `sse_fixed_only + lambda * bits_fixed_only` against `sse_with_aps + lambda * (8 * aps_byte_cost + bits_with_aps)`, an apples-to-apples SSE + lambda * total_bits comparison. New tests cover (a) `round50_chroma_sao_does_not_regress_chroma_psnr` — Cb / Cr clear the round-50 SAO floor (≥ 58 / 57 dB) on the round-49 structured-chroma fixture, (b) `estimate_alf_bin_cost_disabled_is_zero` — disabled ALF emits at most a CABAC-flush footer (≤ 32 bits), (c) `estimate_alf_bin_cost_grows_with_luma_on_ctbs` — bin cost monotonically grows with the number of "luma_on" CTBs (verifies the helper exercises the per-CTB syntax emit, not always the flush footer).
- **Round 49 — chroma residual emit (forward DCT + flat quant + CABAC for Cb / Cr)** — closes the largest open encoder correctness gap from rounds 35 – 48: chroma planes were direct-copied from the source, which is wrong as soon as the source carries any subsampling / colour-space-conversion artefact the per-component reconstruction loop is supposed to replicate. New `prepare_chroma_tb` mirrors `prepare_luma_tb` for one chroma component (Cb or Cr) in 4:2:0: DC intra prediction at 128, residual = src − 128, `forward_dct_ii_2d`, `quantize_tb_flat`, `dequantize_tb_flat`, `inverse_transform_2d`, `(pred + dequant_residual).clamp(0, 255)` write back. The chroma TB tracks half luma resolution with a §8.7.4 `nTbS ≥ 4` floor (`n_tb_chroma = max(4, n_tb / 2)`). New `dequant::DequantParams::chroma_8bit(n_tb_w, n_tb_h, qp)` is the chroma analogue of `luma_8bit` (caller plumbs the §8.7.1 chroma `Qp′Cb` / `Qp′Cr` derived via `ctu::chroma_qp_identity`). `PreparedLumaTb` is extended with `n_tb_chroma`, `cb_levels`, `cr_levels` so the second-pass CABAC walk can emit them in §7.3.10 component order (luma → Cb → Cr) via the new `emit_chroma_tb_residuals` helper (`c_idx = 1` for Cb, `2` for Cr, implicit-CBF same convention as the luma emit). The first-pass loop now also feeds `tu_cb_coded` / `tu_cr_coded` into the per-CU deblock metadata (any non-zero quantised level → coded). New tests cover (a) `encode_idr_chroma_psnr_clears_30db_at_qp26` — Cb / Cr PSNR each clear 30 dB on a structured chroma gradient at QP 26, (b) `encode_idr_chroma_flat_grey_reconstructs_exactly` — flat-grey 128 chroma round-trips byte-identically (residual is identically zero, every level quantises to 0, dequant returns 0), (c) `prepare_chroma_tb_flat_block_yields_zero_levels` — flat-grey 4×4 input reconstructs to pred and produces an all-zero level array, (d) `prepare_chroma_tb_non_flat_block_yields_nonzero_level` — a 4×4 patch at value 200 vs pred 128 (residual 72) produces at least one non-zero quantised level at QP=26.
- **Round 48 — per-class ALF luma Wiener design + APS-vs-fixed-only picture-bits trade-off RDO** — `alf_aps_design::design_per_class_luma_alf_filters` extends the round-47 single-row Wiener pass to 25 independent §8.8.5.3 per-class regressions. For every interior luma pixel the design pass derives the same `(filtIdx, transposeIdx)` tuple the decoder runs at apply time (via the now-`pub(crate)` `alf::derive_luma_classification`), reorders the 12 raw tap features through the pixel's `transposeIdx` permutation into the spec's canonical coefficient slots, and accumulates the 12×12 outer-product matrix `R[c]` + 12-vector `r[c]` into the bin for `c = filtIdx`. Each class with `≥ 48` design pixels gets its own Gauss-Jordan solve; under-sampled classes inherit the picture-wide round-47 fallback. `build_per_class_luma_alf_aps_data` packs the 25 rows into an `AlfApsData`; the existing `aps_enc::emit_alf_aps_rbsp` row-deduplication step turns equal class rows into a single signalled filter so the wire format collapses to the round-47 single-row encoding when the 25 fits coincide. `encode_idr_with_residuals` now runs the luma RDO twice — once with `alf_decide_and_apply` (fixed-only) and once with `alf_decide_and_apply_with_aps` (APS + 16 fixed) — and ships the APS NAL only when (a) at least one CTB picks the APS slot and (b) the per-picture SSE_Y reduction beats the APS NAL byte cost converted at `lambda * 8` SSE per APS bit (`lambda = 0.85 * 2^((qp - 12) / 3)`). When the trade-off skips the APS the pipeline collapses to round-46 fixed-only behaviour: no luma APS NAL emitted, PH carries `ph_num_alf_aps_ids_luma = 0`, per-CTU walk uses `sh_num_alf_aps_ids_luma = 0`. `encoder::emit_nal_with_alf_aps_chain(EmittedNalKind::PictureHeader, num, first_id)` is a new public method that parameterises the §7.3.2.8 luma APS chain count + bound id; `encoder::emit_nal(EmittedNalKind::PictureHeader)` keeps round-47 behaviour (1 APS at id 2). New tests cover (a) `design_per_class` populates all 25 class rows even when most classes are under-sampled (fallback substitution), (b) `build_per_class_luma_alf_aps_data` packs the 25 rows correctly, (c) a designed per-class APS round-trips through `emit_alf_aps_rbsp` + `parse_aps`, (d) `lambda_for_qp` matches the documented curve at QP 0 / 26 / 51, (e) `total_sse_y` is zero on identical planes and one-pixel-units on off-by-one planes, (f) the encoder pipeline emits 2 APS NALs (chroma + CC-ALF) on flat-grey + correctly carries `ph_num_alf_aps_ids_luma = 0`, (g) the per-class luma APS round-trips cleanly when shipped on a high-contrast checkerboard.
- **Round 47 — encoder-side ALF luma APS design + APS-signalled-set RDO** — new `alf_aps_design` module owns the §8.8.5.2 Wiener-style coefficient design (12-tap diamond against eq. 1449 / 1450 with `alfShiftY = 7`), the 12×12 normal-equations solver (Gauss-Jordan + partial pivoting), and the lattice-quantisation helper that scales real coefficients to the §7.3.2.18 `alf_luma_coeff_abs` / `alf_luma_coeff_sign` integer representation. `build_luma_alf_aps_data` packs the resulting 12 coefficients into an `AlfApsData` with one signalled filter row (`alf_luma_num_filters_signalled_minus1 = 0`) shared across all 25 §8.8.5.3 classes, ready for `aps_enc::emit_alf_aps_rbsp`. New `alf_enc::alf_decide_and_apply_with_aps` extends the round-41 fixed-set RDO to a 17-trial search (16 fixed + 1 APS-signalled) so the per-CTB pick chooses the lower-SSE option among `{off, fixed 0, …, fixed 15, APS slot 0}` and the chosen `(luma_on, luma_filt_set_idx)` pair (with `luma_filt_set_idx = 16` mapping to the APS-signalled set per eq. 1439) is recorded in the returned `AlfPicture`. The IDR pipeline (`encode_idr_with_residuals`) now (a) designs the luma APS from the post-SAO recon, (b) emits a 3rd `PrefixApsNut` (id = 2) ahead of the picture header, (c) flips the §7.3.2.8 PH chain to `ph_num_alf_aps_ids_luma = 1` / `ph_alf_aps_id_luma[0] = 2`, (d) sets `sh_num_alf_aps_ids_luma = 1` in the per-CTU `AlfSyntaxConfig` so the `alf_use_aps_flag = 1` branch resolves end-to-end. New tests cover (a) flat-grey returns near-zero coefficients (singular regression → solver yields no fit), (b) `quantise_coeff` round + saturate semantics, (c) the linear solver returns identity / `None` for the diagonal / singular cases, (d) a designed APS round-trips through `emit_alf_aps_rbsp` + `parse_aps`, (e) noisy-recon fixture produces at least one non-zero quantised tap, (f) `alf_decide_and_apply_with_aps` never increases SSE_Y, (g) the recorded `AlfPicture` self-replays through `apply_alf` with the APS binding, and (h) the encoder pipeline emits 3 ALF APS NALs (ids 0 / 1 / 2) and the parsed PH carries the new luma-APS chain.
- **Round 46 — ALF wired end-to-end through SPS, PH and per-CTU CABAC** — `emit_sps` now flips `sps_alf_enabled_flag = 1` plus `sps_ccalf_enabled_flag = 1`; `emit_picture_header_body` writes the §7.3.2.8 PH-level ALF chain (`ph_alf_enabled_flag = 1`, `ph_num_alf_aps_ids_luma = 0`, `ph_alf_cb_enabled_flag = 1`, `ph_alf_cr_enabled_flag = 1`, `ph_alf_aps_id_chroma = 0`, `ph_alf_cc_cb_enabled_flag = 1`, `ph_alf_cc_cb_aps_id = 1`, `ph_alf_cc_cr_enabled_flag = 1`, `ph_alf_cc_cr_aps_id = 1`) so primary chroma ALF binds to APS id 0 and CC-ALF to APS id 1. `encode_idr_with_residuals` is split into a first-pass reconstruction (DCT + quant + dequant + IDCT, no CABAC) and a second-pass §7.3.11.2 single-stream walk that emits per-CTU ALF bins (`alf_ctb_flag[]` / `alf_use_aps_flag` / `alf_luma_*_idx` / `alf_ctb_filter_alt_idx[]` / `alf_ctb_cc_*_idc[]`) immediately followed by that CTU's residual bins, then `encode_terminate(0)` (or `(1)` on the last CTU) — replacing round-45's "ALF CABAC sub-block ahead of the residual block" scaffold. New `tests/alf_round_trip.rs::encoder_pipeline_ph_alf_chain_round_trip` verifies the SPS + PH chain parses back, and `encoder_pipeline_slice_alf_cabac_inlined_in_ctu_walk` decodes the inlined ALF bins from the IDR slice RBSP through `decode_alf_picture` to confirm the encoder's all-off RDO decision on a flat-grey source.
- **Round 43 — CC-ALF integration into the IDR encoder pipeline** — `encode_idr_with_residuals` now snapshots the pre-luma-ALF `recPictureL` (§8.8.5.7's required source for CC-ALF reads), runs the round-41 luma filter-set RDO, then chains `cc_alf_decide_and_apply` for both Cb and Cr against an in-memory CC-ALF APS produced by the new `build_cc_alf_aps` helper (one signalled vertical-edge filter per chroma component, taps 1 / 2 = `±4` per Fig. 53). The combined `(luma_filt_set_idx, cc_cb_idc, cc_cr_idc)` per-CTB record is now captured for the future bitstream-emit round. New tests cover (a) the synthesised APS satisfies §7.4.3.18 invariants (one row × `ALF_CC_NUM_COEFF` taps per component, both signal flags = 1), (b) the IDR pipeline never increases chroma SSE on a vertical-edge-luma + chroma-misaligned source (the per-CTB RDO inside `cc_alf_decide_and_apply` is monotone-improving), and (c) flat-grey content still reconstructs byte-exact for both chroma planes (eq. 1515 sums to zero on a flat luma plane).
- **Round 42 — encoder CC-ALF per-CTB filter-selection RDO** — new `alf_enc::cc_alf_decide_and_apply` mirrors the round-41 luma per-CTB SSE selection on the cross-component path. Given a bound CC-ALF APS (`AlfApsData::cc_cb_coeff` / `cc_cr_coeff`) plus the pre-luma-ALF `recPictureL` snapshot the §8.8.5.7 second pass requires, the helper picks for each CTB the lower-SSE option among `{idc = 0, 1, …, N}` where `N = cc_{cb,cr}_coeff.len()`, mutates the chosen chroma plane in place, and writes the chosen `cc_cb_idc` / `cc_cr_idc` into a caller-provided `AlfPicture`. Per-component (`CcAlfComponent::{Cb, Cr}`) so the encoder can chain a luma-RDO → Cb-CC-RDO → Cr-CC-RDO sequence on the same `AlfPicture`. Trial reconstructions reuse the decoder `apply_alf` pipeline (`luma_on = false` everywhere, `cb_enabled` or `cr_enabled` gated to the chosen component, CC binding only) so the RDO's commit semantics are byte-identical to a decoder replay of the recorded picture. New tests cover (a) flat-luma keeps chroma unchanged (every CC tap weight is multiplied by zero), (b) RDO never increases SSE on the chosen component, (c) the recorded `apply_pic` decoder-replays to byte-identical chroma samples, (d) the `Cr` pass leaves `Cb` untouched and vice versa, (e) empty-APS / monochrome early-outs, and (f) chained Cb + Cr passes compose without clobbering each other's idc records.
- **Round 41 — encoder ALF filter-set RDO** — `alf_enc::alf_decide_and_apply` now searches the full §7.4.3.18 fixed-filter family. Round 40 picked between `{off, fixed set 0}` per CTB; round 41 picks the lower-SSE_Y option among `{off, fixed set 0, …, fixed set 15}` independently per CTB (16 trial reconstructions of the post-deblock + post-SAO luma plane via the existing decoder `apply_alf` pipeline, then per-CTB min-SSE selection). The chosen `(luma_on, luma_filt_set_idx)` pair is recorded into the returned `AlfPicture` so the future bitstream-emit round can mirror the per-CTB CABAC syntax. New `NUM_FIXED_FILTER_SETS` const (= 16) anchors the loop bound to `ALF_CLASS_TO_FILT_MAP`'s row count. New tests assert (a) multi-set RDO never loses to the round-40 single-set-zero RDO, (b) at least one CTB picks ALF-on across a 256×128 picture with mixed-frequency injected distortion, and (c) the recorded `AlfPicture` decoder-replays to byte-identical luma samples (RDO-commit semantics match `apply_alf`).
- **Round 40 — GPM (Geometric Partitioning Mode) decoder** — §8.5.4 + §8.5.7. New `gpm` module transcribes Tables 36 + 37, derives `(angleIdx, distanceIdx)` from `merge_gpm_partition_idx`, runs the §8.5.4.2 `(m, n)` / `predListFlagN` derivation (eqs. 646 – 655), and applies the §8.5.7.2 per-pixel weighted-blend (eqs. 999 – 1016) at any `(cIdx, BitDepth)`. Leaf-CU parser reads `merge_gpm_partition_idx` (FL `cMax = 63`) / `merge_gpm_idx0` (TR `cMax = MaxNumGpmMergeCand − 1`) / `merge_gpm_idx1` (TR `cMax = MaxNumGpmMergeCand − 2`) on B-slices when the §7.3.11.7 GPM gate opens, and the CTU walker's `reconstruct_leaf_cu_gpm` runs two §8.5.6.3 MC passes followed by the §8.5.7.2 blend on luma + 4:2:0 chroma. End-to-end fixture: `decode_b_slice_gpm_fires_and_decodes` builds a 16×16 GPM CU with two contrasting references and verifies both pre-blend halves surface in the output luma plane.
- **Round 40 — AMVR (Adaptive Motion Vector Resolution) helper module** — §7.4.11.6 + Table 16. New `amvr` module transcribes the spec's `AmvrShift` table (regular / affine / IBC rows), exposes `apply_amvr_shift` (eqs. 161 – 176), and `ctx_inc_amvr_*` for the upcoming `mvd_coding()` parser. New `SyntaxCtx::AmvrFlag` (Table 89) / `SyntaxCtx::AmvrPrecisionIdx` (Table 90) variants with full init transcriptions wire the contexts into the existing `init_contexts` plumbing.
- **Round 40 — encoder ALF on/off RDO** — `alf_enc::alf_decide_and_apply`. Per-CTB SSE_Y compare between (a) the post-SAO `rec` and (b) the same `rec` filtered with §7.4.3.18 *fixed filter set 0* via the existing decoder `apply_alf` pipeline; commits the lower-SSE option per CTB. Wired into `encode_idr_with_residuals` so the encoder pipeline now finishes deblock → SAO → ALF before NAL emit. Fixed-filter-set ALF avoids the APS-bit-emission dependency, keeping the encoder side scoped to the round.
- **Residual CABAC encoder** (`residual_enc`): forward three-pass `encode_tb_coefficients` (sig/gt1/par/gt3/abs_remainder/sign flags per §7.3.10.11), exp-Golomb-k helper, `encode_last_sig_coeff_pos`; round-trips against `decode_tb_coefficients` for levels 1–5, negative levels, scattered positions, and chroma.
- **SAO encoder** (`sao_enc`): per-CTU mode selection (BO vs EO, all four EO directions) with distortion-only RDO; `sao_decide_picture` integrates into the IDR pipeline.
- **Encoder pipeline** (`encoder_pipeline`): `encode_idr_with_residuals` (DC intra + forward DCT-II + flat quant + CABAC emit + dequant + IDCT + deblock + SAO); `psnr_y` measurement helper.
- **Fixed forward DCT-II scaling**: corrected `forward_dct_ii_2d` final normalisation shift from the wrong `7+log2N` formula to the correct `2·log2N−2`, putting FDCT output in the same domain as `dequantize_tb_flat`; PSNR_Y ≥ 30 dB at QP=26 and ≥ 40 dB at QP=0 on a smooth-gradient 64×64 fixture.

## [0.0.5](https://github.com/OxideAV/oxideav-h266/compare/v0.0.4...v0.0.5) - 2026-05-04

### Other

- dedup BDOF u8 / u16 paths through shared kernel
- HBD picture-plane storage + Main10 / Main12 MC + reconstruction
- surface §8.5.6.3 BD+6 precision intermediate for BDOF
- wire BDOF refinement into leaf-CU bipred dispatch — §8.5.5.1 / §8.5.6.5
- BDOF (Bi-Directional Optical Flow) — §8.5.6.5
- pred_weight_table parser — §7.3.8 (PH-carried path)
- BCW (Bi-prediction with CU-level Weights) — §8.5.6.6.2 eq. 981
- MMVD asymmetric POC bi-pred — §8.5.2.7 eqs. 561 – 580
- replace never-match regex with semver_check = false
- migrate to centralized OxideAV/.github reusable workflows
- combined inter-intra prediction (CIIP) — §8.5.6.7 + Tables 92 / 102 / 106
- MMVD merge candidate — §8.5.2.7 (Tables 17 / 18 + 103 / 104 / 105)
- pairwise-average merge candidate — §8.5.2.4 + §8.5.2.2 step 8
- temporal merge candidate (Col) — §8.5.2.11 + §8.5.2.12
- HMVP multi-CU acceptance fixture — §8.5.2.6 + §8.5.2.16
- HMVP merge candidate insertion + table maintenance — §8.5.2.6 + §8.5.2.16
- B-slice merge + bi-pred MC — §8.5.2.2 step 9 + §8.5.6.6.2 eq. 980
- §8.5.6.3 fractional-pel MC — 8-tap luma + 4-tap chroma
- P-slice inter — cu_skip + regular merge + integer-pel MC
- ISP — Intra Sub-Partitions (§8.4.5.1)
- CCLM intra prediction (§8.4.5.2.14)
- round 19 — MIP intra mode (matrix-based intra prediction)
- round 18 — BDPCM intra mode (luma + chroma)
- round 17 — ALF fixed-filter family + CC-ALF apply
- adopt slim VideoFrame shape
- forward-side CABAC engine + forward DCT-II / flat quant
- implement §8.8.5.3 luma classification (filtIdx + transposeIdx)
- implement §8.8.5 adaptive loop filter (luma + chroma apply, ALF APS parser)
- long luma 5/7-tap, chroma strong, §7.3.11.3 SAO CABAC parser
- implement §8.8.4 sample adaptive offset (EO + BO) + size-2 DCT-II
- implement §8.8.3 in-loop deblocking filter (short-tap)
- wire chroma reconstruction (Cb / Cr) + DCT-II 64 (§8.7.4.2)
- implement reconstruct_leaf_cu — intra prediction + inverse transform pipeline (§8.4 / §8.7)
- pin release-plz to patch-only bumps

### Added

- **HBD picture-plane storage + Main10 / Main12 MC + reconstruction —
  round 33.** New [`crate::reconstruct::PicturePlane16`] /
  [`crate::reconstruct::PictureBuffer16`] mirror the legacy `u8`
  [`crate::reconstruct::PicturePlane`] / [`crate::reconstruct::PictureBuffer`]
  but store samples as `u16` and carry an explicit `bit_depth` field
  (8..=16). `PicturePlane16::filled` panics on out-of-range seeds,
  `set` returns `Err` on out-of-bounds writes or values exceeding the
  bit-depth max, and `to_picture_plane_u8` provides the canonical
  `>> (bit_depth - 8)` narrowing for downstream YUV420P sinks. New
  [`crate::reconstruct::reconstruct_tb_into_u16`] is the HBD twin of
  [`crate::reconstruct::reconstruct_tb_into`] — writes the eq. 1426
  `Clip1(pred + res)` value into a `u16` destination plane at the
  supplied `bit_depth` with no narrowing (the legacy `u8` overload
  truncates Main10 / Main12 by 2 / 4 bits). New
  [`crate::inter::predict_luma_block_high_precision_u16`] is the HBD
  twin of [`crate::inter::predict_luma_block_high_precision`]; it
  reads `u16` reference samples via two new HBD filter primitives
  (`luma_h_8tap_u16` / `luma_v_only_8tap_u16`) so the §8.5.6.3
  separable 8-tap luma filter sees the full Main10 / Main12 dynamic
  range rather than the 8-bit-truncated value the existing `&PicturePlane`
  reader exposes. New [`crate::bdof::bdof_refine_into_u16`] mirrors
  [`crate::bdof::bdof_refine_into`] for the HBD output path: same
  §8.5.6.5 algorithmics, but the eq. 977 `pbSamples` clamp lands in a
  `u16` plane at the requested `bit_depth`. The existing 8-bit code
  paths are byte-identical (no signature changes, no behaviour
  changes); the HBD variants are additive and tests cross-pin u8/u16
  parity at `bit_depth == 8`. New tests cover (a) `PicturePlane16`
  filled / set / get / clip-range invariants, (b) Main10 reconstruct
  preserving values > 255, (c) Main10 reconstruct clipping at 1023,
  (d) `to_picture_plane_u8` narrowing 1020 → 255, (e) end-to-end
  Main10 DC-impulse reconstruction (DC-predicted intra TB +
  dequantise + inverse DCT-II + reconstruct landing in a `u16` plane
  with values escaping the 8-bit ceiling), (f) HBD MC u8/u16 parity
  at BD = 8 across 5 sub-pel positions, (g) Main10 integer-pel HP
  lift = `1023 << 4 = 16368` (vs. the legacy 8-bit-truncated `255 <<
  6 = 16320`), (h) Main12 integer-pel HP lift = `4095 << 2 = 16380`,
  (i) BDOF u8/u16 parity at BD = 8 with a non-trivial motion-offset
  predictor pair, (j) Main10 BDOF identical-predictor round-trip
  preserving sample value 900, and (k) bit-depth-mismatch error
  surfacing for both HBD MC and HBD BDOF. The cascade is contained
  to the new HBD entry points; consumer code can opt into HBD by
  switching `PictureBuffer` → `PictureBuffer16` and the corresponding
  `_u16` MC / reconstruct calls without any 8-bit churn.
- **§8.5.6.3 high-precision intermediate surfaced for BDOF — round 32.**
  New [`crate::inter::predict_luma_block_high_precision`] is a
  drop-in parallel of [`crate::inter::predict_luma_block`] that
  returns the spec's `BitDepth + 6` precision i32 intermediate per
  the §8.5.6.3.2 separable filter (the value the §8.5.6.6.x
  composition stages take as input, *before* the per-list clip and
  right-shift). Output buffer is row-major `w × h` `Vec<i32>`. The
  `bit_depth` parameter selects the right shift1 — `Min(4, BD - 8)`
  — so the function works at every supported bit depth (the
  existing 8-bit path was hard-coded to BD = 8). Companion
  [`crate::bdof::build_extended_pred_high_precision`] lifts the
  intermediate into the spec's `(nCbW + 2) × (nCbH + 2)` extended
  layout via pure 1-sample edge replication (no scaling — the input
  already lives in BD + 6 precision). The BDOF dispatch site in
  `reconstruct_leaf_cu_inter` now flows the per-list MC through the
  HP pipeline, making §8.5.6.5 spec-byte-identical at every bit
  depth: BD = 8 keeps a 14-bit gradient pipeline, BD = 10 a 16-bit
  pipeline, BD = 12 an 18-bit pipeline. The round-30 8-bit lifter
  (`build_extended_pred_8bit`) is now `#[deprecated]` and retained
  only for back-compat with pre-round-32 callers; new code should
  use the HP path. New tests exercise (a) integer-pel HP / 8-bit-
  lifter parity (the integer lift `<< (14 - BD)` coincides with the
  8-bit `<< SHIFT1`, so both extended buffers are bit-identical),
  (b) subpel HP / 8-bit-lifter agreement within ≤ 1 LSB at BD = 8
  (the 8-bit lifter discards the sub-LSB precision the HP path
  keeps), (c) precision scaling for BD = 8 / 10 / 12 (`<< 6` /
  `<< 4` / `<< 2` integer-pel lift), (d) HP-path identical-
  predictors no-op (round-30 invariant preserved through the new
  pipeline), and (e) HP buffer length validation. The existing
  integration test
  `decode_b_slice_bdof_refinement_differs_from_bipred_average`
  continues to pass on the HP pipeline.
- **BDOF wiring into leaf-CU bipred dispatch — §8.5.5.1 / §8.5.6.5
  (round 31)** — the round-30 [`bdof`](src/bdof.rs) module is now
  reachable from [`crate::ctu::CtuWalker`]. The leaf-CU bipred branch
  in `reconstruct_leaf_cu_inter` derives `bdofUsedFlag` per the
  §8.5.5.1 bullet list (`ph_bdof_disabled_flag`,
  `sps_bdof_enabled_flag`, both `predFlagL{0,1}`, symmetric POC
  distance computed from `current_poc - poc(L0)` vs `poc(L1) - current_poc`,
  STRP classification of both refs, `MotionModelIdc == 0`, no
  sub-block / sym-MVD / CIIP / BCW / weighted-pred / RPR,
  `cbW * cbH >= 128`, `cIdx == 0`); when the gate is open the bipred
  composition runs the §8.5.6.5 refinement
  ([`bdof_refine_into`]) instead of the eq. 980 default-weighted
  average. The 8-bit per-list MC outputs are lifted into the spec's
  `(nCbW + 2) × (nCbH + 2)` extended layout via the round-30
  [`build_extended_pred_8bit`] bridge — surfacing the §8.5.6.3
  separable filter's `BitDepth + 6` precision intermediate is left
  as a follow-up that would make BDOF spec-byte-identical at all
  bit depths. New picture-header switch [`CtuWalker::set_ph_bdof_disabled`]
  exposes `ph_bdof_disabled_flag` to the slice driver; defaults to
  `true` (BDOF off) so existing tests keep their byte-for-byte
  baselines. New integration test
  `decode_b_slice_bdof_refinement_differs_from_bipred_average`
  builds a 16x8 single-CU B-slice with two short-term references at
  symmetric POC distance (poc 0 / current 1 / poc 2) and asymmetric
  per-list ramps (L1 = L0 with a 1-sample horizontal shift), then
  decodes the same payload twice (BDOF on vs BDOF off) and asserts
  (a) the BDOF-off luma plane equals the byte-exact eq. 980 average
  of the two ramps, (b) the BDOF-on luma plane differs from the
  BDOF-off luma plane (the §8.5.6.5 refinement is observable), and
  (c) the chroma planes are byte-identical between the two runs
  (BDOF only refines `cIdx == 0`).

- **Bi-Directional Optical Flow (BDOF) — §8.5.6.5 (round 30)** — new
  [`bdof`](src/bdof.rs) module implements the per-pixel optical-flow
  refinement layered on top of bi-pred motion compensation.
  [`bdof_refine_into`] is the bit-accurate algorithm: for each `4x4`
  sub-block it computes horizontal/vertical gradients on the two
  list predictors over a `6x6` window (eqs. 962–965), accumulates
  the five correlation sums `sGx2`/`sGy2`/`sGxGy`/`sGxdI`/`sGydI`
  (eqs. 969–973), solves the closed-form `(vx, vy)` refinement
  clipped to `±(2/16)` pel via `mvRefineThres = 1 << 4` (eqs. 974/
  975), and writes the eq. 977 `pbSamples = Clip3(0, (1 << BitDepth)
  − 1, (predL0 + offset4 + predL1 + bdofOffset) >> shift4)` output.
  Inputs are the spec's `(nCbW + 2) × (nCbH + 2)` extended high-
  precision per-list arrays; output writes `nCbW × nCbH` `u8`
  samples into a destination [`reconstruct::PicturePlane`] at
  `(dst_x, dst_y)`. [`bdof_used_flag`] mirrors the §8.5.5.1 bullet
  list one-to-one (`ph_bdof_disabled_flag`, both `predFlagL{0,1}`,
  symmetric POC distance, both refs are STRP, `MotionModelIdc == 0`,
  no `merge_subblock_flag`/`sym_mvd_flag`/`ciip_flag`, `bcwIdx == 0`,
  no luma/chroma weights, `cbWidth >= 8 && cbHeight >= 8 &&
  cbW * cbH >= 128`, no RPR-active flags, `cIdx == 0`).
  [`build_extended_pred_8bit`] convenience helper lifts an existing
  pre-clamped 8-bit per-list prediction into the extended-prediction
  layout (1-sample edge replication + `<< SHIFT1` precision lift) so
  the new module can be exercised against the round-22 fractional-pel
  MC output today; once the §8.5.6.3 separable filter is refactored
  to surface its `BitDepth + 6` precision intermediates, the helper
  becomes optional. 12 new unit tests cover the gating helper, the
  `floor_log2`/`sign` primitives, identical-input no-op behaviour
  (constant + horizontal ramp), non-trivial refinement on a 1-sample-
  shifted L0/L1 pair, edge-replication scaling in the 8-bit lifter,
  and malformed-input rejection. The wiring into the leaf-CU walker
  (replacing the existing default-bipred-average call when
  `bdofUsedFlag` is true) is a follow-up — for now [`bdof_refine_into`]
  is reachable through the public module API.

- **`pred_weight_table()` parsing — §7.3.8** — the picture header
  parser now walks the explicit weighted prediction table when
  `(pps_weighted_pred_flag || pps_weighted_bipred_flag) &&
  pps_wp_info_in_ph_flag` (the in-PH path); the round-28 baseline
  surfaced `Error::Unsupported` here. New top-level
  [`parse_pred_weight_table`](src/picture_header.rs) reads
  `luma_log2_weight_denom` (range-checked to 0..=7), the optional
  `delta_chroma_log2_weight_denom` (range-checked so
  `ChromaLog2WeightDenom ∈ 0..=7`), `num_l0_weights` (1..=15),
  per-i `luma_weight_l0_flag` / `chroma_weight_l0_flag` (skipped when
  `chroma_present == false`), the per-record
  `(delta_luma_weight_l0[i], luma_offset_l0[i])` (each in -128..=127),
  and the chroma quad
  `(delta_chroma_weight_l0[i][j], delta_chroma_offset_l0[i][j])`
  for `j ∈ {Cb, Cr}`. The L1 walk fires when
  `pps_weighted_bipred_flag && pps_wp_info_in_ph_flag &&
  num_ref_entries[1][RplsIdx[1]] > 0`; `num_ref_entries_l1` is
  threaded in from the PH-level RPL. New types
  [`PredWeightTable`](src/picture_header.rs) /
  [`PredWeightTableList`](src/picture_header.rs) /
  [`LumaWeight`](src/picture_header.rs) /
  [`ChromaWeight`](src/picture_header.rs) hold the parsed structure;
  [`derive_luma_weight`](src/picture_header.rs) /
  [`derive_luma_offset`](src/picture_header.rs) implement the §7.4.7
  inferences (`LumaWeightLN[i] = (1 << luma_log2_weight_denom) +
  delta_luma_weight_lN[i]` when the per-i flag is 1, else
  `2^luma_log2_weight_denom`; offset defaults to 0). The
  reconstruction pipeline does not yet apply the parsed weights
  (the §8.5.6.6.3 explicit weighted sample prediction process is a
  future round); this commit covers the parser-side gap so the PH
  now walks end-to-end on bitstreams that signal weighted prediction
  in the PH. The slice-header path (when
  `pps_wp_info_in_ph_flag == 0`) is left as a future increment —
  `parse_pred_weight_table` accepts the flag and pins
  `num_weights = 0` in that mode (slice-header callers will need to
  thread `NumRefIdxActive[N]` separately).
  Test count: 538 unit (was 530, +8 covering the minimal-L0 path,
  explicit luma weight derivation, two-record L0 + one-record L1,
  the L1-suppression-when-no-L1-refs gate, the chroma-skip path
  when `chroma_present == false`, the out-of-range
  `luma_log2_weight_denom > 7` rejection, the
  `num_l0_weights == 0` rejection, and the `num_l0_weights > 15`
  rejection).

- **BCW — Bi-prediction with CU-level Weights (§8.5.6.6.2 eq. 981)** —
  the bi-pred MC path now applies explicit weighted blending when the
  chosen merge candidate carries `bcw_idx > 0` and CIIP is off. New
  [`MvField::bcw_idx`](src/inter.rs) field carries the per-block
  `BcwIdx[x][y]` (the spec's `BcwIdx[xCb][yCb]` array, sampled at
  4x4 luma granularity), inherited automatically by spatial merge
  candidates via the existing motion-field broadcast (eqs. 496 / 501
  / 506). The temporal merge / pairwise-average / zero-MV pad / Col
  candidates pin `bcw_idx = 0` per the spec footnotes (eq. 478, eq.
  483 footnote, §8.5.2.5). New
  [`BCW_W_LUT`](src/inter.rs) ships the spec's
  `bcwWLut[k] = {4, 5, 3, 10, -2}` table; new
  [`bi_pred_avg_8bit_bcw`](src/inter.rs) /
  [`predict_luma_block_bipred_bcw`](src/inter.rs) /
  [`predict_chroma_block_bipred_bcw`](src/inter.rs) implement the
  `pbSamples = Clip1((w0*v0 + w1*v1 + 4) >> 3)` form of eq. 981 (the
  factor-of-64 cancellation of the spec's `(w0*v0*64 + w1*v1*64 +
  256) >> 9` is byte-exact for already-clamped 8-bit predL0 / predL1
  inputs). The CTU walker bi-pred apply path now selects between
  eq. 980 (`bcw_idx == 0` OR `ciip_flag == 1`) and eq. 981 per spec
  gating. The legacy `predict_luma_block_bipred` /
  `predict_chroma_block_bipred` are retained as the `bcw_idx == 0`
  fast-path inside the new BCW helpers. Test count: 530 unit (was
  518, +12 covering the bcw_idx ladder spot-checks at indices 0..=4
  with eq. 981 spec values, the negative-weight `(w0 = -2, w1 = 10)`
  and `(w0 = 10, w1 = -2)` branches, the `Clip1` upper-bound clamp at
  255, the lower-bound clamp at 0 on the negative-blend branch, the
  out-of-range bcw_idx error, the byte-equivalence pin against
  `predict_luma_block_bipred` for the `bcw_idx == 0` path, the
  default-MvField bcw_idx invariant, and the `BCW_W_LUT` value pin).

- **MMVD asymmetric POC bi-pred (§8.5.2.7 eqs. 561 – 580)** — extends
  the round-27 MMVD apply pass with the full §8.5.2.7 bi-pred branch
  ladder. Round-27 only handled the equal-POC-distance shortcut
  (eqs. 557 – 560) and conservatively folded `MmvdOffset` into the L1
  half regardless of POC layout; that lost bit-accuracy on any
  bi-pred MMVD CU with asymmetric short-term references. The new
  [`apply_mmvd_to_base_with_poc`](src/inter.rs) takes
  `(currPocDiffL0, currPocDiffL1, lt_l0, lt_l1)` and dispatches into
  three paths per spec: (1) equal POC distance OR LT-ref shortcut →
  `mMvdL1 = MmvdOffset` (eqs. 557 – 560), (2) opposite-sign POC
  distances → `mMvdL1 = -MmvdOffset` (eqs. 564 / 565), (3) asymmetric
  same-sign short-term refs → scale `MmvdOffset` by the §8.5.2.12
  `distScaleFactor` chain (eqs. 561 – 580 reusing eqs. 601 – 605:
  `td = clip(-128, 127, currPocDiffL0)`, `tx = (16384 + |td|>>1)/td`,
  `distScaleFactor = clip(-4096, 4095, (tb*tx + 32) >> 6)`,
  `mMvdL1[c] = clip(-2^17, 2^17 - 1, (distScaleFactor*MmvdOffset[c] +
  128 - (prod>=0)) >> 8)`). Uni-pred bases collapse to eqs. 581 / 582
  unchanged. The CTU walker MMVD apply site at
  [`reconstruct_leaf_cu_inter`](src/ctu.rs) now pulls per-list POCs
  from `RefPicListN[refIdxLN].poc` against `current_poc` and feeds
  them into the new helper; LT refs are conservatively passed as
  `false` (the LT-ref short-term distinction is not yet plumbed into
  the [`ReferencePicture`] type — the asymmetric branch fires only
  when both refs are short-term, which matches the round-27 fixture's
  layout exactly so that test stays byte-identical). The legacy
  POC-agnostic [`apply_mmvd_to_base`] is retained for the
  unit-test surface but is no longer wired into the pipeline. Test
  count: 518 unit (was 511, +7 covering equal-POC bi, opposite-sign
  bi, asymmetric scaled bi, LT-ref bypass, uni-pred-L0, uni-pred-L1,
  and the degenerate `currPocDiffL0 == 0` guard).

- **Combined Inter-Intra Prediction — CIIP (round-28, §8.5.6.7)** —
  the merge-data sub-tree now lights up the CIIP branch when
  `sps_ciip_enabled_flag == 1`. New `MergeData::ciip_flag` carries the
  parsed (or §7.4.12.7-inferred) syntax. `CuToolFlags` gains
  `ciip_enabled` + `gpm_enabled` so the leaf CU reader sees the SPS
  gates. The §7.3.11.7 `regular_merge_flag` parse is now wired
  end-to-end: when CIIP is enabled, `cu_skip_flag == 0`,
  `cbW * cbH ≥ 64`, and `cbW < 128, cbH < 128`, the bit is read from
  Table 102 (`(init_type − 1) * 2 + ctxInc` with
  `ctxInc = !cu_skip_flag` per Table 132); otherwise it stays inferred
  to 1. When `regular_merge_flag == 0`, the CIIP branch fires per
  §7.4.12.7 (GPM disambiguation deferred — the parse-gated
  ciip_flag bin lands in a future round). Round-28 wires Table 92
  (`cu_coded_flag`) and Table 106 (`ciip_flag`) ctx initialisation
  arrays as new `SyntaxCtx` variants `CuCodedFlag` / `CiipFlag` (Table
  92 `[6, 5, 12] / [4, 4, 4]`, Table 106 `[57, 57] / [1, 1]`); the
  cu_coded_flag read fires for non-skip merge CUs to gate the
  `transform_tree()` body — round-28 still surfaces Unsupported when
  the flag comes back 1, so the acceptance fixture pins
  `cu_coded_flag = 0` (no residual). The §8.5.6.7 weight ladder ships
  as [`ciip_intra_weight`](src/inter.rs) (`(both intra, both not
  intra, exactly one intra) → (3, 1, 2)`) and the eq. 998 combiner
  ships as [`combine_ciip_samples`](src/inter.rs) — both clamp into
  the bit-depth range. The CTU walker now maintains a per-picture
  4x4 intra-coded grid alongside the §7.4.4 motion field (every leaf
  CU writes its `MODE_INTRA / MODE_INTER` bit to the cells it
  covers); CIIP CUs sample the §8.5.6.7 A / B luma neighbours
  `(xCb − 1, yCb − 1 + cbHeight)` and `(xCb − 1 + cbWidth, yCb − 1)`
  off this grid to derive `w`. The reconstruction path computes
  planar `predSamplesIntra` from the partially-reconstructed picture
  buffer (out.luma neighbours + §8.4.5.2.8 substitution → mid-grey
  fallback at picture corners) **before** the regular-merge MC writes
  predSamplesInter into the CU rectangle, then folds the two via eq.
  998. Chroma is composed identically — the spec sets
  `IntraPredModeC = INTRA_PLANAR` for CIIP CUs and reuses the same
  weight `w` (per the eqs. 995 / 996 SubWidthC / SubHeightC scaling
  the chroma A / B neighbours land on the same luma-grid 4x4 cells).
  Acceptance fixture
  [`decode_p_slice_ciip_fires_and_decodes`](tests/reconstruct_pipeline.rs)
  synthesises a P-slice payload at the picture corner: 8x8 CU,
  `cu_skip = 0`, `general_merge = 1`, `regular_merge = 0` (parsed
  through Table 102), `ciip_flag` inferred to 1, `merge_idx = 0`,
  `cu_coded = 0`. With both §8.5.6.7 neighbours out-of-picture
  → `w = 1` → eq. 998 collapses to `(planar128 + 3 * inter + 2) >> 2`;
  the test pins the byte-exact reconstructed luma plus the
  unmodified motion-field MV (CIIP doesn't shift the chosen merge
  candidate). Test count: 511 unit (was 502, +9 covering the
  weight ladder, eq. 998 spot checks at all three weights, equal-
  predictor identity, the bit-depth clamp, and the
  `MergeData::default` CIIP-off invariant) + 24 integration (was 23,
  +1 CIIP acceptance).

- **Merge with Motion Vector Differences — MMVD (round-27, §8.5.2.7)** —
  the merge-data sub-tree now lights up the MMVD branch when
  `sps_mmvd_enabled_flag == 1`. New `MergeData` fields
  `mmvd_merge_flag` / `mmvd_cand_flag` / `mmvd_distance_idx` /
  `mmvd_direction_idx` carry the parsed syntax. Tables 17 + 18 ship as
  `MMVD_DISTANCE_TABLE` (`{1, 2, 4, 8, 16, 32, 64, 128}` in pre-`<<2`
  units → `{1/4, 1/2, 1, 2, 4, 8, 16, 32}` luma after eq. 188),
  `MMVD_DISTANCE_TABLE_FULLPEL` (`4×` the regular table, gated by the
  picture-header `ph_mmvd_fullpel_only_flag`), and `MMVD_SIGN_TABLE`
  (the four cardinal directions `+x / -x / +y / -y` per the spec).
  [`derive_mmvd_offset`](src/inter.rs) emits `MmvdOffset` as a
  `MotionVector` in 1/16-pel units (eqs. 188 / 189: `(MmvdDistance <<
  2) * MmvdSign`), and [`apply_mmvd_to_base`](src/inter.rs) folds the
  offset into a chosen base candidate's per-list MVs — the uni-pred
  case (eqs. 581 / 582) and the equal-POC-distance bi-pred shortcut
  (eqs. 557 – 560) are wired; asymmetric POC bi-pred (eqs. 561 – 580)
  rides alongside future BCW / DMVR work. Tables 103 / 104 / 105 ship
  as new `SyntaxCtx` variants `MmvdMergeFlag` / `MmvdCandFlag` /
  `MmvdDistanceIdx` with the spec initValue / shiftIdx pairs (Table 103
  `[26, 25] / [4, 4]`, Table 104 `[43, 43] / [10, 10]`, Table 105 `[60,
  59] / [0, 0]`). The leaf CU reader gains `read_mmvd_distance_idx`
  (TR `cMax = 7, cRiceParam = 0`: bin0 ctx-coded, bins 1..6 bypass)
  and `read_mmvd_direction_idx` (FL `cMax = 3`: 2 bypass bins MSB
  first); both ctxIdx selectors index by `init_type - 1` (P/B only,
  MMVD is never signalled in I slices). The §7.4.12.7 inference
  `merge_idx == mmvd_cand_flag` is folded into the parser so the
  reconstruction pipeline picks `mergeCandList[merge_idx]` uniformly.
  `CuToolFlags` gains `mmvd_enabled` + `ph_mmvd_fullpel_only`;
  `CtuWalker` gains [`set_ph_mmvd_fullpel_only`](src/ctu.rs) for
  plumbing the picture-header switch. The CTU walker invokes
  `derive_mmvd_offset` + `apply_mmvd_to_base` between merge-list
  selection and motion compensation when `mmvd_merge_flag == 1`, so
  the per-block motion field records the MMVD-corrected MV (not the
  base MV). Acceptance fixture
  [`decode_p_slice_mmvd_fires_and_decodes`](tests/reconstruct_pipeline.rs)
  synthesises a P-slice payload with `cu_skip = 1`, `mmvd_merge_flag =
  1`, `mmvd_cand_flag = 0`, `mmvd_distance_idx = 2` (1-luma offset),
  `mmvd_direction_idx = 0` (`+x`); the chosen base is the §8.5.2.2
  step 9 zero-MV pad, MMVD adds `(+16, 0)` 1/16-pel = `(+1, 0)` int-
  pel, and the test pins the byte-exact `ref[y][x + 1]` translated
  luma plane plus the broadcast `MotionField` carrying the corrected
  MV. Test count: 502 unit (was 492, +10 covering Table 17 fractional
  vs fullpel grids, all four Table 18 directions, eq. 188 `<< 2`
  scaling, the apply-to-base uni / bi-pred branches, and the
  `MergeData::default` MMVD-off invariant) + 23 integration (was 22,
  +1 MMVD acceptance).

- **Pairwise-average merge candidate (round-26, §8.5.2.4)** — the
  §8.5.2.2 step 8 invocation now lands in
  [`build_merge_cand_list`](src/inter.rs) /
  [`build_merge_cand_list_b`](src/inter.rs) between the §8.5.2.6 HMVP
  step and the §8.5.2.5 zero-MV pad. New
  `derive_pairwise_average_candidate` synthesises the avgCand from
  `mergeCandList[0]` and `mergeCandList[1]` per the spec's per-list
  four-way switch (both-active → average, only-p0 → copy p0, only-p1 →
  copy p1, neither → inactive on this list), with the §8.5.2.14
  `rightShift = 1, leftShift = 0` rounding implemented as a signed-
  magnitude divide-by-2-toward-zero in `round_mv_pairwise` (the spec
  rounding is **not** the same as Rust's arithmetic `>> 1` for
  negative sums: `(-1) >> 1 == -1` but `Sign(-1) * (Abs(-1) >> 1) ==
  0`). Trigger gates mirror §8.5.2.2 step 8 exactly:
  `numCurrMergeCand > 1 && numCurrMergeCand < MaxNumMergeCand` (the
  source-pair availability + room-left checks). For P-slices the
  spec's `numRefLists == 1` clause forces `refIdxL1avg = -1` and
  `predFlagL1avg = 0`, which the implementation enforces by skipping
  the L1 derivation. Acceptance fixture
  [`decode_p_slice_pairwise_average_fires_and_decodes`](tests/reconstruct_pipeline.rs)
  exercises a quad-split 16x16 P-slice where four 8x8 cu_skip CUs
  read the §8.5.2.11 Col candidate from a non-uniform reference
  motion field; CU3 (BR @(8,8)) builds a merge list whose pairwise-
  average lands at slot 5 with chosen MV `(-32, 0)` in 1/16-pel units
  (`(B1=(-16,0) + A1=(-48,0)) >> 1`), giving an integer-pel `(-2, 0)`
  translation that the test pins via byte-exact reconstruction of
  the BR quadrant against `mc_copy_block_int` (and via the post-
  decode motion-field assertion that CU3's broadcast MvField carries
  the avgCand MV — distinguishable from the Col / HMVP entries which
  would have surfaced as `(-64, 0)`). Test count: 492 unit (was 481,
  +11 covering the §8.5.2.14 signed-magnitude rounding edge cases,
  the four per-list switch arms, the `>1` and `<Max` gates, and the
  walk-order integration with both `build_merge_cand_list` /
  `build_merge_cand_list_b`) + 22 integration (was 21, +1 acceptance).

- **Temporal merge candidate (round-25, §8.5.2.11 / §8.5.2.12)** — the
  Col candidate now lands at §8.5.2.2 step 5 between the spatial walk
  and the §8.5.2.6 HMVP insertion. New
  [`derive_temporal_merge_candidate`](src/inter.rs) implements the
  bottom-right + centre-fallback collocated-position derivation of
  §8.5.2.11 (eqs. 592 – 597, with `(xColBr >> 3) << 3` 8x8 grid
  rounding and the same-CTB-row gate `yCb >> CtbLog2SizeY == yColBr >>
  CtbLog2SizeY`); intra / IBC / palette `colCb` short-circuit to
  `availableFlagLXCol = 0`. The §8.5.2.12 picked-MV scaling
  (eqs. 598 – 605) lands too: equal-distance / long-term-ref short-
  circuit at eq. 600 (just `Clip3(-2^17, 2^17 - 1, mvCol)`) and the
  general `tx = (16384 + |td|>>1)/td` /
  `distScaleFactor = clip(-4096, 4095, (tb*tx + 32) >> 6)` /
  `mvLXCol = clip(-2^17, 2^17 - 1, (distScaleFactor*mvCol + 128 -
  (prod>=0)) >> 8)` chain. The §8.5.2.15 temporal MV buffer compression
  is folded into the fetch as integer-pel rounding (`(mv >> 4) << 4`).
  [`build_merge_cand_list`] / [`build_merge_cand_list_b`] gain a
  leading `Option<MvField>` "Col" parameter that inserts the candidate
  at the spec-prescribed slot. New [`ReferencePicture`] field
  `motion_field: Option<MotionField>` carries the per-4x4 MV state of
  the collocated picture (`None` for intra-only references —
  derivation falls through to the `availableFlagCol = 0` branch).
  `CtuWalker` gains [`set_temporal_mvp`](src/ctu.rs) for plumbing
  `current_poc` / `ph_temporal_mvp_enabled_flag` /
  `sh_collocated_from_l0_flag` / `sh_collocated_ref_idx`, and a
  private `derive_col_candidate` that wraps the per-list invocation
  (uni-pred for P, fused L0+L1 for B) and feeds the result into the
  merge-list builder. Acceptance fixture
  [`decode_p_slice_temporal_merge_fires_and_decodes`](tests/reconstruct_pipeline.rs)
  decodes a 16x16 P-slice whose single CU has no spatial / no HMVP
  candidates; the chosen `mergeCandList[0]` is the Col MV scaled per
  POC distance, MC translates the reference picture by the scaled
  vector, and the test pins the byte-exact translated luma plane.
  Test count: 481 unit (was 472, +9 covering BR/centre fallback,
  CTB-row crossing, equal-distance + scaled-distance scaling, intra
  collocated rejection, no-MF short-circuit, walk-order spatial → Col
  → HMVP) + 21 integration (was 20, +1 acceptance).

- **HMVP multi-CU acceptance fixture (round-25)** — extends the
  round-24 HMVP wiring with an end-to-end black-box validator that
  drives multiple inter CUs through the merge-mode pull-in path.
  The new
  [`decode_p_slice_quad_split_exercises_hmvp_merge_pull_in`](tests/reconstruct_pipeline.rs)
  integration test synthesises a P-slice CABAC payload via
  [`cabac_enc::ArithEncoder`] for a 16x16 single-CTU picture
  quad-split into four 8x8 cu_skip + merge_idx = 0 leaf CUs
  (split_cu(1) → split_qt(1) at root, then four split_cu(0)
  followed by four (cu_skip(1), merge_idx(0)) pairs in z-order
  matching the decoder's parse-then-syntax bin sequence). After
  decode the test pins three round-24 invariants on the multi-CU
  path: (1) the picture matches the constant-64 reference frame
  byte-exactly (every chosen merge candidate resolves to the
  uni-pred zero-MV / refIdx 0 record across all four CUs, with
  `insert_hmvp_into_merge_list` invoked at least three times during
  the slice — once per CU2/CU3/CU4); (2)
  [`CtuWalker::hmvp_table`] post-decode contains 1..=`MAX_HMVP_CAND`
  entries with the `MvField` carrying `pred_flag_l0 = mode_inter =
  cu_skip_flag = true` / `pred_flag_l1 = false` / `ref_idx_l0 = 0`
  / `mv_l0 = ZERO` (the §8.5.2.16 dedup collapses all four
  identical pushes into a single retained entry); (3) every 4x4
  block of the per-picture motion field carries the chosen MvField,
  proving the broadcast in `reconstruct_leaf_cu_inter` ran for all
  four CUs. Brings the integration test count to 20 (was 19) on
  zero new lib-unit-test additions — the round-24 lib-unit suite
  already covered `insert_hmvp_into_merge_list` directly.

- **HMVP merge candidate insertion + table maintenance (round-24)** —
  the §8.5.2.6 history-based merging-candidate derivation and the
  §8.5.2.16 update process now land in `oxideav_h266::inter`. A new
  [`HmvpTable`](src/inter.rs) per-slice circular buffer of up to 5
  `MvField` records lives on the `CtuWalker` (reset to empty at
  `begin_slice` per the §7.3.11 `NumHmvpCand = 0` slice-start rule;
  for our single-tile fixture the CTU-column-tile-boundary reset
  collapses to the slice-start reset). The signatures of
  [`build_merge_cand_list`] and [`build_merge_cand_list_b`] gain a
  trailing `Option<&HmvpTable>` argument: when `Some`, §8.5.2.2 step
  7 now fires between the spatial walk and the zero-MV pad, invoking
  [`insert_hmvp_into_merge_list`] which walks the HMVP table newest-
  to-oldest (`HmvpCandList[NumHmvpCand − hMvpIdx]` with
  `hMvpIdx = 1..NumHmvpCand`), prunes duplicates against B1 / A1
  only for the two newest entries (`hMvpIdx ≤ 2`), and halts as soon
  as `numCurrMergeCand == MaxNumMergeCand − 1` (last slot reserved
  for the eventual zero-MV pad / pairwise / temporal candidate). The
  §8.5.2.16 update is wired into `CtuWalker::reconstruct_leaf_cu_inter`
  immediately after the per-CU motion-field broadcast: each just-
  decoded inter CU's MvField calls
  [`HmvpTable::update_with`](src/inter.rs), which removes any prior
  duplicate (by `mvf_matches`) and appends the new entry at the
  back, evicting the oldest (front) only when the buffer is at
  capacity *and* no duplicate was found. A new
  [`CtuWalker::hmvp_table`] read accessor exposes the table for
  test inspection. Verified by 12 new lib unit tests
  (`hmvp_table_new_is_empty_and_reset_clears`,
  `hmvp_update_appends_until_capacity`,
  `hmvp_update_evicts_oldest_when_full`,
  `hmvp_update_duplicate_promotes_to_newest`,
  `hmvp_update_duplicate_at_capacity_keeps_oldest`,
  `merge_list_inserts_hmvp_no_pruning_when_no_spatials`,
  `merge_list_hmvp_halts_one_short_of_max`,
  `merge_list_hmvp_prunes_against_b1_when_newest`,
  `merge_list_hmvp_does_not_prune_third_oldest_against_b1`,
  `merge_list_hmvp_skipped_when_spatials_already_max_minus_one`,
  `merge_list_b_inserts_hmvp_then_bipred_pad`,
  `merge_list_with_none_hmvp_is_spatial_only_pad_only`,
  `merge_list_empty_hmvp_skipped`) plus 1 new integration test
  (`decode_p_slice_populates_hmvp_table` — pins both the slice-start
  reset and the post-decode table contents after the all-skip
  P-slice fixture). Out of scope for this round (still surfaces
  `Error::Unsupported` upstream): the §8.5.2.4 pairwise-average
  candidate, the §8.5.2.11 temporal collocated merge candidate (the
  collocated-picture pointer plumbing via `sh_collocated_from_l0` /
  `sh_collocated_ref_idx` lives in r25+), MMVD, CIIP, GPM, subblock
  merge, AMVR, BCW (`bcwIdx != 0`), explicit weighted prediction
  (§8.5.6.6.3), BDOF (§8.5.6.5), DMVR, PROF (§8.5.6.4), dual-tree
  luma / chroma split, and the full inter residual decode
  (`cu_coded_flag` + `transform_tree()`).

- **B-slice merge / regular-merge subset (round-23)** — first bi-pred
  path lands. The CTU walker now accepts B-slices via `begin_slice`
  (the B-slice gate is dropped); a new `set_ref_pic_list_l1` mirror
  installs `RefPicList[1]` alongside the existing
  `set_ref_pic_list_l0`; the `MvField` per-4×4-block record carries
  `mv_l1` / `ref_idx_l1` / `pred_flag_l1` slots in addition to the
  L0 trio so subsequent CUs read both list halves during their merge
  derivation. The §8.5.2.2 step-9 zero-MV padding gains a B-slice
  variant ([`build_merge_cand_list_b`](src/inter.rs)) that emits
  bi-pred zero-MV candidates (`predFlagL0 == predFlagL1 == 1,
  refIdxL0 == refIdxL1 == 0`) per spec; the `mvf_matches` redundancy
  check now compares both list halves so two candidates that share
  L0 but differ on L1 are correctly *not* collapsed. The §8.5.6.6.2
  eq. 980 default-weighted bi-pred composition (BCW disabled,
  weighted-pred disabled) ships as `bi_pred_avg_8bit` —
  `(predL0 + predL1 + 1) >> 1` over per-list 8-bit prediction
  scratch planes, with [`predict_luma_block_bipred`] /
  [`predict_chroma_block_bipred`] driving the per-list §8.5.6.3
  invocations into matching scratch buffers and compositing into the
  final destination. `CtuWalker::reconstruct_leaf_cu_inter` now
  dispatches the L0-only / L1-only / bi-pred trichotomy from the
  chosen merge candidate's `pred_flag_lN` flags. Verified by 10 new
  lib unit tests in `inter::tests` (UNAVAILABLE shape, `mvf_matches`
  L1 distinction, `build_merge_cand_list_b` bi-pred padding,
  `bi_pred_avg_8bit` DC-preservation across all 256 sample values
  + spec-formula spot-checks + bounds rejection,
  `predict_luma_block_bipred` constant-ref invariant + per-list
  byte-equivalence pin against the trusted single-list helpers,
  same chroma 4-tap byte-equivalence pin) plus 2 new integration
  tests in `tests/reconstruct_pipeline.rs`
  (`decode_b_slice_all_skip_bipred_matches_average` synthesises a
  B-slice CABAC payload via [`cabac_enc::ArithEncoder`]
  (split_cu_flag(0) → cu_skip_flag(1) at init_type 2 →
  merge_idx-bin0(0)) and verifies the decoded picture is byte-exactly
  the §8.5.6.6.2 average of two distinct deterministic L0 / L1 ramps;
  `decode_b_slice_writes_bipred_motion_field` verifies the per-CU
  motion-field write-back propagates `pred_flag_l0 == pred_flag_l1
  == true` + `ref_idx_l0 == ref_idx_l1 == 0` onto every 4×4 block).
  Out of scope for this round (still surfaces `Error::Unsupported`
  upstream): the §8.5.2.6 HMVP table, the §8.5.2.4 pairwise-average
  candidate, the §8.5.2.11 temporal collocated candidate, MMVD,
  CIIP, GPM, subblock merge, AMVR, BCW (`bcwIdx != 0`), explicit
  weighted prediction (§8.5.6.6.3), BDOF (§8.5.6.5), DMVR, PROF
  (§8.5.6.4), dual-tree luma / chroma split, and the full inter
  residual decode (`cu_coded_flag` + `transform_tree()`).

- **§8.5.6.3 fractional-pel motion compensation (round-22)** — the
  P-slice MC pipeline now handles sub-pel MVs end to end.
  [`predict_luma_block`](src/inter.rs) and [`predict_chroma_block`]
  (src/inter.rs) replace the round-21 `mc_copy_block_int` dispatch
  inside `ctu::CtuWalker::reconstruct_leaf_cu_inter`. The §8.5.6.3.2
  8-tap separable luma filter (Table 27, `hpelIfIdx == 0` — 16
  fractional positions, coefficients sum to 64) and the §8.5.6.3.4
  4-tap separable chroma filter (Table 33 — 32 fractional positions)
  are wired through a horizontal-then-vertical pipeline per eqs.
  932 – 936 (luma) and 950 – 954 (chroma) with picture-edge
  `Clip3(0, picW - 1, ...)` clamping per eqs. 930 – 931 / 948 – 949.
  The §8.5.6.6.2 default uni-pred clamp (eq. 978: `Clip3(0, 255,
  (predL0 + 32) >> 6)` at BitDepth 8) closes the chain. Integer-pel
  `(xFrac, yFrac) == (0, 0)` collapses byte-identical to the
  round-21 `mc_copy_block_int` path so the all-skip P-slice fixture
  still passes. 13 new unit tests in `inter::tests` (Table 27 / 33
  row-sum normalisation invariants, zero-MV memcpy passthrough, full
  16 × 16 luma + 32 × 32 chroma DC-preservation grids, per-Table-27-
  row spec-formula pins for both xFrac-only and yFrac-only paths,
  the half-pel `hpelIfIdx == 0` row, picture-edge clamping under a
  sub-pel MV, sub-pel self-roundtrip with PSNR = +∞, integer-pel
  cross-check vs `mc_copy_block_int`, and a chroma vertical-only
  ramp). Out of scope for this round (still surface
  `Error::Unsupported` upstream): affine-mode filter tables 30 / 31
  / 32, scaled-reference tables 28 / 29 / 34 / 35, BCW (§8.5.6.6.2
  bcwIdx ≠ 0), explicit weighted prediction (§8.5.6.6.3), BDOF
  (§8.5.6.5), DMVR (§8.5.6.3.2 dmvrFlag clamp), PROF (§8.5.6.4),
  the §8.5.6.3.3 cbProfFlag / bdofFlag integer-fetching fast path,
  RPR (reference-picture resampling), wraparound MC, and the
  subpicture treated-as-pic clamping.

- **P-slice inter coding (round-21 subset)** — first inter path lands
  for the smallest viable scope: `cu_skip_flag` + `general_merge_flag`
  (inferred to 1 when skip) + `merge_data()` regular-merge subtree
  (`merge_idx`). New `inter` module hosts the `MotionVector` /
  `MvField` / `MotionField` per-4×4-block grid, `ReferencePicture`,
  `derive_spatial_merge_candidates` (§8.5.2.3 — B1 → A1 → B0 → A0 →
  B2 walk with the §6.4.4 availability + Log2ParMrgLevel suppression
  + the spec's redundancy checks), `build_merge_cand_list` (§8.5.2.2
  step 5 spatial-only assembly + step 9 zero-MV padding to
  `MaxNumMergeCand`), and `mc_copy_block_int` for §8.5.6 integer-pel
  motion compensation (luma + 4:2:0 chroma) with picture-edge clamping.
  CABAC tables for `cu_skip_flag` (Table 64), `general_merge_flag`
  (Table 82), `regular_merge_flag` (Table 102) and `merge_idx`
  (Table 109) are wired through the new `LeafCuCtxs::init_with_init_type`
  entry point with §9.3.2.2 / Table 51 init-type derivation
  (I → 0; P → 1/2 from `sh_cabac_init_flag`; B → 2/1). The CTU walker
  now accepts P-slices via `begin_slice` (B remains unsupported until
  the bi-pred path lands), drives `set_ref_pic_list_l0` from the
  caller, and `reconstruct_leaf_cu` dispatches into a new
  `reconstruct_leaf_cu_inter` that builds the spatial merge list,
  picks `mergeCandList[merge_idx]`, runs MC, and broadcasts the chosen
  `MvField` across every 4×4 block of the CU so subsequent CUs read
  it during their own merge derivation. 9 new lib unit tests
  (`inter::tests`) + 2 new integration tests
  (`decode_p_slice_all_skip_matches_reference` and
  `decode_p_slice_writes_motion_field`) — the all-skip end-to-end
  test synthesises a CABAC payload via
  [`cabac_enc::ArithEncoder`](src/cabac_enc.rs) (split_cu_flag(0) →
  cu_skip_flag(1) → merge_idx-bin0(0)) and verifies the decoded P
  picture is byte-identical to a hand-painted reference frame. Out
  of scope for this round (= still surfaces `Error::Unsupported`):
  non-merge inter CUs (mvd_coding), non-skip merge CUs (cu_coded_flag
  + residual), MMVD, CIIP, GPM, subblock merge, AMVR, BCW, fractional
  MVs, B-slices, the §8.5.2.6 HMVP table, the §8.5.2.4 pairwise
  average, and the §8.5.2.11 temporal collocated candidate.

- **ISP — Intra Sub-Partitions** — round-20 win. New `isp` module
  implements the §7.4.12.2 / §8.4.5.1 sub-partition derivation:
  `num_intra_subpartitions` covers Table 13 (NoSplit → 1, (4,8) /
  (8,4) → 2, else → 4) and `iter_isp_partitions` materialises the
  per-partition `nW`, `nH`, `nPbW = max(4, nW)`, `pbFactor`,
  `xPartIdx`, `yPartIdx`, `xPartPbIdx` from eqs. 251 – 260. The
  leaf-CU reader gains `decode_transform_unit_isp` which walks each
  subpartition's `transform_unit()` per §7.3.11.10: chroma CBFs are
  read only on the last partition, `tu_y_coded_flag` is read per
  partition with `InferTuCbfLuma` inferring 1 for the last partition
  when every prior CBF was 0. `LeafCuResidual` gains
  `luma_subparts: Vec<LeafCuLumaSubpart>` to carry the per-partition
  geometry + level array. `ctu::CtuWalker::reconstruct_leaf_cu`
  dispatches into `reconstruct_leaf_cu_isp_luma`, which walks the
  subpartitions in spec order and reconstructs each one — predicting
  off the partially-reconstructed plane so partition `i+1` reads
  freshly-written samples from partition `i`. The `pbFactor == 2`
  case (vertical splits with `nW ∈ {1, 2}`) caches the prediction
  window across the paired partitions per eq. 260. Chroma stays
  un-split (eqs. 251 – 254 only fire for `cIdx == 0`). 11 new tests
  (7 in `isp::tests`, 4 in `ctu::tests`) cover the partition
  geometry, ref-dimension derivation, CU walks for HOR/VER splits,
  the small-CU two-partition cases, the `pbFactor == 2` window-share
  path, and a CU with a non-zero residual landing only on the last
  subpartition.

- **CCLM — Cross-Component Linear Model intra prediction** — round-19
  win. `cclm::predict_cclm` runs the §8.4.5.2.14 pipeline end to end:
  the §6.4.4-style neighbour availability collapse (eqs. 359 – 362),
  the §8.4.5.2.14 numIs4N / pickPosN derivation (eq. 364), the
  down-sampled-luma kernels (eqs. 366 – 369; both
  `sps_chroma_vertical_collocated_flag` branches wired), the selected
  neighbour kernels (eqs. 370 – 377 with the `bCTUboundary` 3-tap
  fallback per eq. 373), the 4-point min-max regression (eqs. 386 –
  389), the `(a, b, k)` derivation including the `divSigTable[]` of
  eq. 400 (eqs. 390 – 403), and the eq. 404 `Clip1` predictor. The
  4:2:0 chroma surround is the supported configuration; 4:2:2 / 4:4:4
  branches are wired in the helper but not exercised yet. Wired into
  `ctu::CtuWalker::reconstruct_chroma_plane` so chroma TBs flagged
  with `IntraPredModeC ∈ {81, 82, 83}` reconstruct end to end through
  `INTRA_LT_CCLM` / `INTRA_L_CCLM` / `INTRA_T_CCLM` instead of being
  collapsed to PLANAR. Covered by 13 lib unit tests in `cclm.rs`
  (constant-neighbour identity, mid-grey fallback, mode validation,
  divSigTable spot-check, pickPos derivation x3, 10-bit fallback,
  `(a, b, k)` derivation diff-zero + spot-check, ramp recovery,
  LumaPlane edge clamping, T_CCLM extended-top neighbour) plus 2
  CTU-level smoke tests
  (`reconstruct_leaf_cu_cclm_lt_runs_pipeline` and
  `reconstruct_leaf_cu_cclm_t_picture_edge_writes_mid_grey`).

- **MIP — Matrix-based Intra Prediction** — round-19 win.
  `mip::predict_mip` runs the §8.4.5.2.2 pipeline end to end:
  §8.4.5.2.3 boundary downsampling (rectangular box filter), the
  matrix-vector product against the `mWeight` tables (§8.4.5.2.4), the
  `intra_mip_transposed_flag` block transpose, and §8.4.5.2.5
  upsampling (separable horizontal then vertical) for blocks larger
  than `predSize`. The 30 weight matrices (16 / 8 / 6 modes for
  `mipSizeId = 0 / 1 / 2`, all values in [0, 127]) come straight from
  ITU-T H.266 (V4, 01/2026) Tables 276..305 and live in `mip_tables`.
  Wired into `ctu::CtuWalker::reconstruct_leaf_cu` so leaf CUs flagged
  with `intra_mip_flag` reconstruct end to end instead of surfacing
  `Error::Unsupported`. Covered by 18 lib unit tests (size-id
  derivation, downsampling identity / box-filter, weight-table value
  range, spec-row spot-checks, constant-reference invariants for
  4×4 / 8×8 / 16×16 / 16×8 / 8×16 / 64×64, upsampling identity,
  transpose changes output, output-range clipping, error paths) plus
  2 CTU-level smoke tests
  (`reconstruct_leaf_cu_mip_runs_pipeline` and
  `reconstruct_leaf_cu_mip_transposed_runs_pipeline`).

- **ALF fixed-filter family + CC-ALF apply** — round-17 wins.
  `alf_fixed::ALF_FIX_FILT_COEFF` (64 × 12) and
  `ALF_CLASS_TO_FILT_MAP` (16 × 25) ship the §7.4.3.18 eqs. 90 / 91
  tables, transposed at transcription time so in-code indexing matches
  the spec's symbolic `[i][j]` / `[m][n]` convention. The §8.8.5.2
  `resolve_luma_filter_set` now resolves `AlfCtbFiltSetIdxY < 16` via
  the fixed-filter path (eqs. 1437 / 1438) — previously surfaced as
  "luma off". CC-ALF (§8.8.5.7) runs as a second `apply_alf` pass
  reading the pre-luma-ALF snapshot per eq. 1515; Table 47 yP1 / yP2
  vertical offsets and the eq. 1517 `SubHeightC == 1` row-suppression
  guard land alongside. `AlfApsBinding` gains `cc_cb_aps` / `cc_cr_aps`
  slots + `is_all_off` now considers `cc_*_idc` so CC-ALF-only CTBs
  trigger the apply pass.
- **Forward-side CABAC engine** (`cabac_enc::ArithEncoder`) — round-16
  primitive that mirrors the §9.3.4.3 decoder. `encode_decision` /
  `encode_bypass` / `encode_terminate` + `finish()` produce a byte
  stream that round-trips bit-identically through `cabac::ArithDecoder`,
  with shared `ContextModel` so the §9.3.4.3.2.2 dual-exponential
  probability state stays in lockstep. Validated by 11 self-tests
  including a 4096-bin pseudo-random stress and an integration test
  using the real `tables::SyntaxCtx` bundles.
- **Forward DCT-II + flat quantisation** (`transform_fwd`) — encoder
  duals of `transform::inverse_transform_2d` and
  `dequant::dequantize_tb_flat`. `forward_dct_ii_1d` /
  `forward_dct_ii_2d` apply the 64×64 trMatrix transposed (the
  matrix is orthogonal up to scale per §8.7.4.5);
  `quantize_tb_flat` is the integer dual of eq. 1155.
- §8.8.4 SAO (sample adaptive offset) — Edge Offset (4 classes per
  Table 11) and Band Offset modes per §8.8.4.2 eqs. 1424 – 1435 +
  Table 44. Per-CTB params consumed via the new `sao::SaoPicture` /
  `SaoCtbParams` types and `CtuWalker::set_sao_picture`. Wired into
  `CtuWalker::apply_in_loop_filters` after deblocking.
- DCT-II size-2 inverse transform — re-uses the 64×64 trMatrix entries
  at columns 0 / 32. Unblocks 2×2 / 2×4 / 4×2 chroma TBs (4×4 / 4×8 /
  8×4 luma CUs under 4:2:0); the chroma reconstruction path now runs
  dequant + IDCT for those sizes instead of the previous bypass.

## [0.0.4](https://github.com/OxideAV/oxideav-h266/compare/v0.0.3...v0.0.4) - 2026-04-25

### Other

- drop oxideav-codec/oxideav-container shims, import from oxideav-core

## [0.0.3](https://github.com/OxideAV/oxideav-h266/compare/v0.0.2...v0.0.3) - 2026-04-24

### Other

- spec-exact residual ctxInc + 3-pass walker + §8.7.3 dequant
- forward-bitstream encoder scaffold (VPS/SPS/PPS/PH/IDR)
- CBF reads + last-sig-coeff + sub-block residual walker
- leaf-CU syntax reader + MPM/chroma intra-mode derivation

## [0.0.2](https://github.com/OxideAV/oxideav-h266/compare/v0.0.1...v0.0.2) - 2026-04-24

### Other

- CTU walker scaffold (picture geometry, slice CABAC init, coding_tree dispatch)
- PPS partition block + slice_address derivation
- slice_header tail past deblocking (dep_quant, extension, byte_alignment)
- stateful picture_header_structure() tail parser
- parse header-level ref_pic_lists() block
- stateful slice header — §7.3.7 through sh_qp_delta / deblocking
- PPS tail — cabac/init + chroma QP + deblocking + info_in_ph flags
- RefPicList0 / RefPicList1 construction (§8.3.2)
- SPS subpicture block + HRD timing + VUI + sps_extension tail
- HRD timing parameter parsers (§7.3.5)
- ref_pic_list_struct() — ST / LT / ILRP entries (§7.3.10, §7.4.11)
- SPS tail — dpb_parameters + partition constraints + tool flags
- DCI + OPI RBSP parsers (§7.3.2.1, §7.3.2.2)
- residual decode + reconstruction helpers (§7.4.11.8, §8.7.5)
- angular intra (subset) + reference substitution (§8.4.5.2)
- coding-tree walker (§7.3.11.4) — I-slice subset
- diagonal scan + 4x4 sub-block partitioning (§7.4.11.9, §6.5.2)
- CABAC per-syntax-element initValue / shiftIdx tables
- add 2D inverse transform composition + intra DC / planar
- add ctx module for per-syntax-element CABAC ctxInc derivations
- land DCT-II tables + size 16/32 DST-VII/DCT-VIII kernels
- inverse DST-VII / DCT-VIII 4×4 and 8×8 kernels (§8.7.4)
- CABAC arithmetic decoding engine (§9.3)

- Foundation scaffold: NAL unit framing, RBSP extraction, Exp-Golomb bit
  reader, and the VPS/SPS/PPS/APS + picture/slice header parsers
  (§7.3.2). No CTU reconstruction yet.
