# Changelog

All notable changes to this crate are recorded here.

## [Unreleased]

### Added

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
