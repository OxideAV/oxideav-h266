# oxideav-h266

Pure-Rust **H.266 / VVC** (ITU-T H.266 | ISO/IEC 23090-3) decoder for
oxideav. Zero C dependencies, no FFI, no `*-sys` crates.

This crate currently implements a **foundation / parser scaffold**.
VVC is a very large specification (500+ pages); the current code gives
the rest of the workspace a reliable NAL framing + parameter-set
parsing layer on top of which a CTU walker can be built in later
increments.

Part of the [oxideav](https://github.com/OxideAV/oxideav-workspace)
framework but usable standalone.

## Installation

```toml
[dependencies]
oxideav-core = "0.1"
oxideav-codec = "0.1"
oxideav-h266 = "0.0"
```

## Parse support

* **NAL framing** — Annex B byte-stream (`0x000001` / `0x00000001`
  start codes) and length-prefixed (ISOBMFF `vvc1` / `vvi1`),
  including emulation-prevention byte removal.
* **Exp-Golomb bit reader** (§9.2) — `u(n)`, `ue(v)`, `se(v)`, byte
  alignment, rbsp-trailing-bits probe.
* **Parameter sets**
  * **VPS** (§7.3.2.2) — video parameter set.
  * **SPS** (§7.3.2.3) — profile/tier/level, chroma format, picture
    size, CTU size, transform sizes, and most of the tools-enable
    flag block.
  * **PPS** (§7.3.2.4) — picture parameter set.
  * **APS** (§7.3.2.5) — adaptation parameter set (ALF / LMCS /
    scaling-list type, parameters walked).
* **Picture / slice header** (§7.3.2.7-8) — coarse walk through
  syntax, enough to keep the bit position aligned across NAL
  boundaries during decoder bring-up.
* **Weighted prediction parameters** (§7.3.8) — `pred_weight_table()`
  walked when carried in the PH (round-29; `pps_wp_info_in_ph_flag
  == 1` path); produces `PredWeightTable` with per-record
  `(delta_luma_weight, luma_offset)` plus paired Cb/Cr chroma deltas
  for both L0 and L1. `derive_luma_weight` / `derive_luma_offset`
  implement the §7.4.7 inferences. The reconstruction pipeline does
  not yet apply the parsed weights — the §8.5.6.6.3 explicit
  weighted sample prediction process is a future round.

## Decode support

The reconstruction pipeline is being built incrementally and now covers
the **intra-only single-tile single-slice subset** plus the round-28
**P + B-slice merge / regular-merge subset with HMVP + temporal Col +
§8.5.2.4 pairwise-average candidate + §8.5.2.7 MMVD + §8.5.6.7 CIIP,
§8.5.6.3 fractional-pel MC and §8.5.6.6.2 default-weighted bi-pred**:

* **Intra**: PLANAR / DC / cardinal angular intra (modes 2, 18, 34, 50,
  66) + MIP (§8.4.5.2.2 — all 30 weight matrices) + CCLM (§8.4.5.2.14)
  + BDPCM + ISP (§8.4.5.1, all 4 split types).
* **Inter (round-26)**: P + B-slice `cu_skip_flag` +
  `general_merge_flag` inference + `merge_data()` regular-merge
  subset (`merge_idx`), §8.5.2.3 spatial-merge candidate derivation
  (5-position B1/A1/B0/A0/B2 list with redundancy checks across both
  L0 + L1 records), §8.5.2.2 mergeCandList assembly with §8.5.2.11
  temporal collocated candidate (round-25, bottom-right + centre
  fallback, 8x8 grid rounding, same-CTB-row gate; §8.5.2.12 picked-MV
  scaling per `(tb*tx + 32) >> 6` distScaleFactor with the `mv_col >>
  4 << 4` §8.5.2.15 buffer-compression fold) followed by §8.5.2.6
  HMVP insertion (round-24, per-slice 5-entry table reset at slice
  start per §7.3.11; merge derivation reads newest-to-oldest with the
  spec's two-newest-entry prune against A1/B1; §8.5.2.16 update fires
  after every inter CU), §8.5.2.4 pairwise-average synthetic
  candidate (round-26, derived from `mergeCandList[0]` and
  `mergeCandList[1]` with the §8.5.2.14 `rightShift = 1, leftShift = 0`
  signed-magnitude rounding; per-list switch covering both-active /
  only-p0-active / only-p1-active / neither-active branches; L1 half
  suppressed for P-slices via the `numRefLists == 1` clause), and
  zero-MV padding (uni-pred for P, bi-pred for B), §8.5.6 motion
  compensation including the §8.5.6.3 8-tap luma fractional-sample
  interpolation (Table 27, `hpelIfIdx == 0` family) and §8.5.6.3.4
  4-tap chroma interpolation (Table 33), with the §8.5.6.6.2 default
  uni-pred clamp at 8-bit and the §8.5.6.6.2 eq. 980 default-weighted
  bi-pred composition `(predL0 + predL1 + 1) >> 1` for B-slice bi-
  pred candidates. Single L0 + L1 reference each. **Round-27 lands
  §8.5.2.7 MMVD (Merge with Motion Vector Differences):** the leaf CU
  reader parses `mmvd_merge_flag` (Table 103) / `mmvd_cand_flag`
  (Table 104) / `mmvd_distance_idx` (Table 105 TR `cMax = 7`) /
  `mmvd_direction_idx` (FL `cMax = 3`) when `regular_merge_flag == 1
  && sps_mmvd_enabled_flag == 1`; `derive_mmvd_offset` emits
  `MmvdOffset` per Tables 17 + 18 + eqs. 188 / 189 (regular `{1/4,
  1/2, 1, 2, 4, 8, 16, 32}` luma steps or `ph_mmvd_fullpel_only_flag`-
  scaled `{1, 2, 4, 8, 16, 32, 64, 128}` luma steps) and
  `apply_mmvd_to_base_with_poc` folds it into the chosen base
  candidate's per-list MVs across all four §8.5.2.7 branches: uni-pred
  (eqs. 581 / 582), symmetric bi-pred / equal-POC / LT shortcut
  (eqs. 557 – 560), opposite-sign POC bi-pred (eqs. 564 / 565 — `mMvdL1
  = -MmvdOffset`), and asymmetric same-sign short-term ref bi-pred
  (eqs. 561 – 580 with the §8.5.2.12 `distScaleFactor` chain — `td =
  clip(-128, 127, currPocDiffL0)`, `tx = (16384 + |td|>>1)/td`,
  `distScaleFactor = clip(-4096, 4095, (tb*tx + 32) >> 6)`, then
  `mMvdL1 = clip(-2^17, 2^17-1, (distScaleFactor*MmvdOffset + 128 -
  (prod>=0)) >> 8)`). **Round-28 lands §8.5.6.7 CIIP (Combined Inter-Intra
  Prediction):** when `sps_ciip_enabled_flag = 1` and the §7.3.11.7
  gates open (`cu_skip_flag == 0`, `cbW * cbH ≥ 64`, `cbW < 128`,
  `cbH < 128`), the leaf CU reader parses `regular_merge_flag` (Table
  102) and infers `ciip_flag = 1` per §7.4.12.7 when GPM is off.
  CIIP CUs combine the regular-merge MC predSamplesInter with a
  planar predSamplesIntra drawn from the partially-reconstructed
  picture neighbours via eq. 998
  `(w * intra + (4 − w) * inter + 2) >> 2` with `w ∈ {1, 2, 3}` from
  the §8.5.6.7 A / B luma neighbour intra-coded grid (`(both intra,
  both not intra, one intra) → (3, 1, 2)`). The CTU walker maintains
  a per-picture 4x4 intra-coded grid (read by CIIP, written by every
  leaf CU) so neighbour status is exact across raster scan order.
  Chroma reuses the same `w` and a planar chroma intra prediction
  per the §8.4.3 PLANAR-luma → PLANAR-chroma mapping. **Round-29 lands §8.5.6.6.2 BCW (Bi-prediction with CU-level
  Weights):** the bi-pred composition path now applies eq. 981
  weighted blending when the chosen merge candidate carries
  `bcw_idx > 0` and `ciip_flag == 0`. New `MvField::bcw_idx`
  field carries the per-block `BcwIdx[x][y]` (sampled at 4x4 luma
  granularity, inherited automatically by spatial merge
  candidates per eqs. 496 / 501 / 506). Temporal merge / pairwise
  average / zero-MV pad pin `bcw_idx = 0` per the spec footnotes.
  The `bcwWLut[k] = {4, 5, 3, 10, -2}` table maps `bcw_idx ∈ 1..=4`
  to `w1 ∈ {5, 3, 10, -2}` (and `w0 = 8 - w1 ∈ {3, 5, -2, 10}`);
  `bcw_idx == 0` short-circuits to the eq. 980 default average,
  matching the spec's "If bcwIdx is equal to 0 OR ciip_flag == 1"
  gate.
  **Round-30 lands §8.5.6.5 BDOF (Bi-Directional Optical Flow)** as
  a per-pixel refinement on top of the bi-pred composition; round-31
  wires the §8.5.5.1 `bdofUsedFlag` derivation into the leaf-CU
  bipred dispatch (gated on `sps_bdof_enabled_flag`,
  `ph_bdof_disabled_flag`, both `predFlagL{0,1}`, symmetric POC
  distance, STRP refs, `MotionModelIdc == 0`, no sub-block /
  sym-MVD / CIIP / BCW / weighted-pred, `cbW * cbH >= 128`,
  `cIdx == 0`); when the gate is open the §8.5.6.5 refinement
  runs in place of the eq. 980 average. **Round-32 surfaces the
  §8.5.6.3 `BitDepth + 6` precision intermediate** via
  `predict_luma_block_high_precision` (paired with
  `build_extended_pred_high_precision`) so BDOF gradients now
  operate at the spec's full 14-bit precision (8-bit input),
  16-bit (10-bit input), 18-bit (12-bit input) — the round-30
  8-bit lifter (`build_extended_pred_8bit`) is now deprecated.
  **Round-33 introduces HBD picture-plane storage** —
  `PicturePlane16` / `PictureBuffer16` (parallel to the legacy
  `u8` `PicturePlane` / `PictureBuffer`) carry samples in `u16`
  cells with an explicit `bit_depth` (8..=16) and per-write
  range-clipping. New `_u16` twins of the HP MC
  (`predict_luma_block_high_precision_u16`), the BDOF refinement
  (`bdof_refine_into_u16`), and the per-TB reconstruction
  (`reconstruct_tb_into_u16`) read / write at full Main10 / Main12
  precision instead of 8-bit-truncating reference samples — the
  legacy `u8` paths stay byte-identical and the HBD twins are
  cross-pinned against them at `bit_depth == 8`. The cascade is
  contained to the new HBD entry points; consumer code opts in by
  switching the buffer type + the corresponding `_u16` calls.
  **Round-40 lands §8.5.4 + §8.5.7 GPM (Geometric Partitioning Mode)** —
  the new `gpm` module transcribes Tables 36 + 37, derives
  `(angleIdx, distanceIdx)` from `merge_gpm_partition_idx`, runs the
  §8.5.4.2 `(m, n)` / `predListFlagN` derivation (eqs. 646 – 655),
  and applies the §8.5.7.2 per-pixel weighted blend (eqs. 999 – 1016)
  on luma + 4:2:0 chroma. The leaf-CU parser reads
  `merge_gpm_partition_idx` (FL `cMax = 63`), `merge_gpm_idx0` (TR
  `cMax = MaxNumGpmMergeCand − 1`), `merge_gpm_idx1` (TR `cMax =
  MaxNumGpmMergeCand − 2`) on B-slices when the §7.3.11.7 GPM gate
  opens; the CTU walker runs two §8.5.6.3 MC passes followed by
  the §8.5.7.2 blend. **Round-40 also lands §7.4.11.6 AMVR
  (Adaptive Motion Vector Resolution) helpers** — Table 16
  `AmvrShift` for regular / affine / IBC rows, `apply_amvr_shift`
  (eqs. 161 – 176), `ctx_inc_amvr_*` and Tables 89 / 90 init
  transcriptions, ready for the upcoming `mvd_coding()` parser.
  Affine + scaled-reference filter tables 28 / 29 / 30 / 31 / 32
  / 34 / 35, DMVR, PROF land in later rounds.
  **Round-41 lands the encoder ALF filter-set RDO** — the per-CTB
  trial search now spans all 16 §7.4.3.18 fixed filter sets
  (`AlfCtbFiltSetIdxY ∈ 0..16`) instead of just set 0. For each CTB
  the encoder picks the lower-SSE_Y option among `{off, set 0, …,
  set 15}` and records the choice into the returned `AlfPicture`
  for the future bitstream-emit round. Compute scales linearly
  (16× the round-40 ALF apply cost). APS-signalled luma filter
  sets (`AlfCtbFiltSetIdxY ≥ 16`) remain a follow-up round.
  **Round-42 lands the encoder CC-ALF per-CTB filter-selection
  RDO** — `alf_enc::cc_alf_decide_and_apply` mirrors the round-41
  per-CTB SSE selection on §8.8.5.7. Given a bound CC-ALF APS
  (`cc_cb_coeff` / `cc_cr_coeff`, up to 4 filters per component
  per §7.4.3.18) plus the pre-luma-ALF `recPictureL` snapshot, the
  helper picks for each CTB the lower-SSE option among
  `{idc = 0, 1, …, N}` and records the chosen `cc_cb_idc` /
  `cc_cr_idc` into a caller-provided `AlfPicture`. Per-component
  (`CcAlfComponent::{Cb, Cr}`) so the encoder can chain a luma RDO
  → Cb CC-RDO → Cr CC-RDO on the same picture-level decision
  record.
  **Round-43 wires CC-ALF into the IDR encoder pipeline** —
  `encode_idr_with_residuals` now snapshots the pre-luma-ALF
  `recPictureL`, runs the round-41 luma filter-set RDO, then
  chains `cc_alf_decide_and_apply` for Cb and Cr against an
  in-memory CC-ALF APS (one signalled vertical-edge filter per
  component). The §8.8.5.7 RDO is monotone-improving so the
  pipeline never increases chroma SSE; flat / aligned content is
  a strict no-op. The combined `(luma_filt_set_idx, cc_cb_idc,
  cc_cr_idc)` per-CTB record is now captured for the future
  bitstream-emit round.
* **Transforms**: DCT-II inverse for sizes 2 / 4 / 8 / 16 / 32 / 64;
  DST-VII / DCT-VIII for 4 / 8 / 16; flat-list dequant.
* **CABAC**: full §9.3 arithmetic engine + per-syntax-element initValue
  / shiftIdx tables for everything currently parsed (cu_skip /
  general_merge / regular_merge / merge_idx + round-27 mmvd_merge_flag
  / mmvd_cand_flag / mmvd_distance_idx + round-28 ciip_flag (Table 106)
  / cu_coded_flag (Table 92) with init_type routing for I / P / B
  slices).
* **In-loop filters**: §8.8.3 deblocking, §8.8.4 SAO (Edge + Band
  offset), §8.8.5 ALF including the fixed-filter family + CC-ALF.
* **Non-skip merge with cu_coded_flag == 1 (residual transform_tree),
  full non-merge inter CU reconstruction**: still surface
  `Error::Unsupported`. The §7.3.10.10 `mvd_coding()` parser (round-103)
  and the §7.3.10.8 MVP-side flags `inter_pred_idc` / `sym_mvd_flag` /
  `ref_idx_lX` / `mvp_lX_flag` (round-108) are decoded, and **round-111
  lands the §8.5.2.8 / §8.5.2.9 / §8.5.2.10 AMVP candidate derivation**
  that consumes `mvp_lX_flag` (new `amvp` module). The §8.5.2.10
  spatial scan walks the A group (`A0 = (xCb−1, yCb+cbH)` → `A1`) and
  the B group (`B0 = (xCb+cbW, yCb−1)` → `B1` → `B2`), each picking the
  first effectively-available neighbour whose prediction — list X first
  (eqs. 588 / 590), then list `Y = 1 − X` (eqs. 589 / 591) — points at a
  reference picture with the **same POC** as the current CU's
  `RefPicList[X][refIdxLX]` (`DiffPicOrderCnt == 0`; AMVP does not scale
  across POC distance, unlike spatial *merge*). `round_mv_amvr`
  implements the §8.5.2.14 eqs. 608 – 610 rounding with
  `rightShift = leftShift = AmvrShift`; `build_mvp_cand_list` runs the
  §8.5.2.9 step-3 Col gate (Col consulted only when *not* both A and B
  available with different MVs), the step-4 list assembly (eq. 584), the
  step-5 HMVP fill, and the step-6 zero-MV pad to exactly 2 candidates
  (eqs. 585 – 587); `select_mvp` is eq. 583 (`mvpListLX[mvp_lX_flag]`)
  and `derive_final_mv` folds the AMVR-shifted `mvd` into the predictor
  (`mvLX = mvpLX + mvdLX`, clipped to `[−2^17, 2^17 − 1]`).
  **Round-114 lands the §8.5.2.9 step-5 HMVP RPL-reference filter**
  (`derive_hmvp_mvp_candidates`): it walks `HmvpCandList[i − 1]` for
  `i = 1..Min(4, NumHmvpCand)` — oldest-first, capped at 4 entries, the
  *opposite* traversal from the §8.5.2.6 merge walk and with no A1/B1
  prune — and for each RPL `LY` (`Y = X` then `1 − X`) admits the entry's
  AMVR-rounded LY MV when its `RefIdxLY` references the current CU's
  `RefPicList[X][refIdxLX]` (`DiffPicOrderCnt == 0`); a bi-pred entry
  matching on both lists contributes twice. The
  **Round-117 makes the §8.5.2.11 temporal Col candidate live**
  (`derive_temporal_amvp_candidate`): it applies the §8.5.2.11 first-
  bullet gate (`ph_temporal_mvp_enabled_flag == 0 || cbWidth * cbHeight
  <= 32` ⇒ `availableFlagLXCol = 0`), invokes the shared merge
  §8.5.2.11/§8.5.2.12 collocated walk byte for byte (bottom-right →
  centre fallback, §8.5.2.12 list selection, §8.5.2.15 buffer
  compression, eqs. 598 – 605 POC scaling) via
  `derive_temporal_merge_candidate`, and AMVR-rounds the produced
  `mvLXCol` per the §8.5.2.9 step-3 last bullet — returning the
  `Option<MotionVector>` that `build_mvp_cand_list` already feeds through
  its step-3 suppression gate. The CTU-walker fuse (resolving `ColPic` +
  `RefPicList[X][refIdxLX]` POC from the live non-merge inter path) into
  a full non-merge inter CU walk remains deferred.
  **Round-120 lands the §8.5.5.7 affine AMVP candidate list**
  (new `affine_amvp` module). The 9-step driver assembles
  `cpMvpListLX` with exactly 2 entries: (1) inherited A from the
  A0 → A1 cascade and (2) inherited B from B0 → B1 → B2, both via
  the round-91 §8.5.5.5 derivation
  (`derive_inherited_affine_mvp_candidate` applies the §8.5.5.7
  step-4 / step-5 AMVP gate — neighbour available + affine
  (`MotionModelIdc > 0`) + list-X-then-list-Y POC match — and
  AMVR-rounds every emitted CP component); (3) the §8.5.5.8
  constructed CPMV-predictor candidate
  (`derive_constructed_affine_mvp_candidate` reads per-corner
  neighbour sample MVs at TL (`B2 → B3 → A2`), TR (`B1 → B0`), BL
  (`A1 → A0`), accepting translational neighbours via
  `pick_constructed_corner`, and emits `availableConsFlagLX = 1`
  only when all `numCpMv` corners are available); (4) up to three
  §8.5.5.7 step-7 single-corner standalone candidates (`nbCpIdx =
  2, 1, 0`, each replicating one available corner MV across every
  CP); (5) the §8.5.2.11 temporal MV (replicated across every CP);
  (6) zero-MV pad to exactly 2 entries. `select_affine_mvp` is
  eq. 840 (`cpMvpListLX[mvp_lX_flag]`) and `derive_final_affine_cpmvs`
  folds the per-CP AMVR-shifted `mvdCpLX` into the chosen predictor
  via eqs. 664 – 667 (modular `(mvpCp + mvdCp) & (2^18 − 1)` wrap +
  `≥ 2^17 ? − 2^18 : ·` two's-complement unwrap), per CP. The
  CTU-walker fuse — populating `NeighbourAffineQuery` from the live
  per-CB CPMV grid + the §6.4.4 neighbour-availability table — and
  wiring the §8.5.2.11 temporal MV resolver into the per-CU path
  remain deferred.
  (HMVP — §8.5.2.6 + §8.5.2.16 — landed in round-24; temporal merge
  — §8.5.2.11 + §8.5.2.12 — landed in round-25; pairwise-average
  merge — §8.5.2.4 — landed in round-26; MMVD — §8.5.2.7 — landed in
  round-27; CIIP — §8.5.6.7 — landed in round-28; GPM — §8.5.4 +
  §8.5.7 — landed in round-40; AMVR helpers — §7.4.11.6 — landed in
  round-40.)

## Encoder

An IDR-frame encoder pipeline (`encode_idr_with_residuals_cfg`) builds an
Annex-B bitstream with real coded residuals, deblock, SAO, ALF, CC-ALF, and
optional opt-in tools. Round-56 added the MTT BT picker
(`EncoderConfig::enable_mtt_bt_picker`) which evaluates `{leaf, BT_VERT,
BT_HORZ}` on `cost = SSE_Y + λ·bits` for each 64×64 CU. Round-57
extended this with the MTT TT picker (`EncoderConfig::enable_mtt_tt_picker`)
which adds `TT_VERT` (1:2:1 three-column split, 16×64 / 32×64 / 16×64) and
`TT_HORZ` (1:2:1 three-row split, 64×16 / 64×32 / 64×16) per VVC §7.3.10.4 /
§7.4.10.4. Both flags compose: with both on, the candidate set is
`{leaf, BT_VERT, BT_HORZ, TT_VERT, TT_HORZ}`. On a 3-stripe vertical
fixture with the 1:2:1 ratio at QP 32, the TT picker takes leaf-only
SSE_Y from 2576 to 0 (perfect reconstruction).

**Round-58 lands the inter-frame P-slice encoder + decoder scaffold**
(`encoder_inter::encode_p_slice` / `encoder_inter::decode_p_slice`):
single-reference DPB (one L0 picture), integer-pel full-search motion
estimation (SAD on 4×4 luma blocks, the VVC §7.4.10 minimum-PU size),
spatial MVP picker (left → above → zero per §7.4.7.3), per-block CABAC
bin emit for `cu_skip_flag` / `general_merge_flag` / `inter_pred_idc` /
`ref_idx_l0` / `mvd_coding(mvd_x, mvd_y)` (§7.4.7.2 + §9.3.3.7) /
`tu_y_coded_flag`, and luma residual through the existing
`crate::residual_enc::encode_tb_coefficients` chain. A custom
`OXAV_VVC_PSLIC` magic-prefixed wire chunk wraps the slice-header bit
prelude (`slice_type` / `slice_pic_order_cnt_lsb` per §7.4.4.2.2 +
`num_ref_idx_l0_active_minus1` + `slice_qp_delta`) and the per-block
CABAC payload. On a 4-px horizontal translation fixture the scaffold
hits PSNR_Y 78.23 dB; encoder + decoder roundtrip byte-identical.

**Round-59 extends the P-slice path with sub-pel motion compensation.**
The integer-pel SAD full search is now followed by a two-stage refinement:
8 half-pel offsets (±8 in 1/16-pel units) around the integer-pel best,
then 8 quarter-pel offsets (±4) around the half-pel best. All sub-pel
candidates are evaluated through the spec §8.5.6.3.2 Table 27 8-tap
luma interpolation filter (`hpelIfIdx == 0`) via the existing
`crate::inter::predict_luma_block`. The on-wire `MvdLX` magnitudes now
carry 1/16-luma-sample units; the CABAC schema is unchanged. On a
band-limited oversampled fixture, ½-pel reconstructs to PSNR_Y 51.6 dB
and ¼-pel to 52.4 dB; the round-58 4-px integer-pel fixture holds at
78.23 dB. AMVR, `hpelIfIdx` filter selection, the full 1/16-pel
exhaustive search, chroma sub-pel MC, multi-ref DPB, B-slice, and the
Annex-B NAL integration with the IDR pipeline remain deferred.

**Round-60 adds the B-slice (bi-prediction) encoder + decoder
scaffold** (`encoder_inter::encode_b_slice` /
`encoder_inter::decode_b_slice`). Two reference lists (L0 + L1) with
one picture per list, per-list integer-pel full-search SAD + per-list
spatial MVP picker (§7.4.7.3), per-CU `inter_pred_idc` ∈ {PRED_L0,
PRED_L1, PRED_BI} per §7.4.7.2 chosen by Lagrangian SSE-based RDO over
the three candidate predictions, and §8.5.6.4 simple-average bi-pred
reconstruction `pred = (predL0 + predL1 + 1) >> 1` (weighted bi-pred
deferred). Slice header carries `slice_type == B` plus both
`num_ref_idx_l{0,1}_active_minus1` per §7.4.4. Wire chunk
`OXAV_VVC_BSLIC` mirrors the round-58 in-crate framing pattern. On a
4-px translation fixture with `ref_l0 == ref_l1` the B-slice
degenerates to P-slice quality (PSNR_Y 78.23 dB); on a
split-translation fixture the RDO matches the corresponding P-slice
at 54.15 dB.

**Round-61 extends the B-slice path with sub-pel motion estimation.**
After the integer-pel SAD search, each list (L0 and L1) now goes
through the same two-stage refinement that round 59 added to the
P-slice path: 8 half-pel neighbours probed through the §8.5.6.3.2
Table 27 8-tap luma filter (`hpelIfIdx == 0`), then 8 quarter-pel
neighbours around the best half-pel candidate. The RDO over `{L0,
L1, BI}` runs with each list's MV at 1/16-pel precision; bi-pred
reconstruction is still the §8.5.6.4 simple average. On a half-pel
B-slice translation fixture with matched references PSNR_Y reaches
51.57 dB; quarter-pel reaches 52.39 dB; a split-translation BI
fixture (L0=+2 px / L1=-2 px with the current frame mid-way)
reconstructs essentially exactly. The round-60 integer-pel 4-px
fixture holds at 78.23 dB. Multi-reference DPB (more than one
picture per list), chroma sub-pel MC, and weighted bi-pred remain
deferred.

**Round-62 adds multi-reference DPB to both the P-slice and B-slice
encoder + decoder** (`encoder_inter::encode_p_slice_multi_ref` /
`encoder_inter::encode_b_slice_multi_ref` and the matching decoder
calls). Each list (L0 for P / L0 + L1 for B) now holds up to
`MAX_REF_PICS = 4` pictures matching the §A.4 mainstream-profile
constraint. The encoder ME iterates every candidate reference in
each list, runs the round-58 integer-pel SAD search plus the
round-59 sub-pel refinement against each, and picks the cheapest by
SAD before the {L0, L1, BI} RDO runs. The slice header now
advertises real `num_ref_idx_l0_active_minus1` (and, for B,
`num_ref_idx_l1_active_minus1`) per §7.4.4.2; per-CU `ref_idx_l0`
/ `ref_idx_l1` are emitted as truncated-unary per §9.3.3.7 / Table
132 with `cMax = num_active - 1` (bypass-coded for the scaffold,
collapses to zero bins when the list has only one picture so the
single-ref wire stays bit-for-bit unchanged). On a 3-frame P-slice
fixture where the current frame matches frame 0 better than frame
1, the encoder picks `ref_idx=1` on every block and reaches PSNR_Y
= 58.41 dB; on a 4-ref B-slice fixture the multi-ref-aware RDO
splits the translation for an exact BI reconstruction. The
round-58 4-px P-slice regression holds at 78.23 dB and multi-ref +
sub-pel still reaches 52.39 dB.

**Round-63 (Goal B) wires §8.5.6.3.4 4-tap chroma sub-pel MC** into
both encoder + decoder paths. Rounds 58/60/61 had luma 8-tap sub-pel
MC but chroma planes were passed through from L0[0]. Round 63 calls
the existing `crate::inter::predict_chroma_block` (Table 28 4-tap
filter) at chroma block scale (2×2 chroma per 4×4 luma block in
4:2:0), reusing the per-CU luma 1/16-pel MV — `predict_chroma_block`
applies the 4:2:0 mapping internally (chroma 1/32-pel offset is
`mv & 31`, integer chroma offset is `mv >> 5`). The {L0, L1, BI}
dispatch on the B-slice path picks the chroma prediction matching
the chosen `inter_pred_idc`. Wire format is unchanged. On a band-
limited half-pel translation fixture chroma reaches PSNR_Cb = 49.96
dB / PSNR_Cr = 50.39 dB (≥45 dB target).

**Round-63 (Goal A) lands explicit weighted bi-prediction** per
§8.5.6.5 eq. 994 + §7.4.7.7 `pred_weight_table()`. The encoder
estimates slice-level luma weights/offsets from per-list mean luma
offsets to the current frame and, when those offsets exceed a
heuristic threshold, sets `BSliceHeader.pred_weight_table = Some(...)`
so the decoder will replicate the eq. 994 form
`(p0 * w0 + p1 * w1 + ((o0 + o1 + 1) << log2WD)) >> (log2WD + 1)`
clipped to 8-bit. Per-CU the encoder runs an SSE-based RDO between
the unweighted and weighted form and emits a 1-bit `use_weighted_bi`
selector after the BI MVD chain. On a fade fixture (L0 = curr - 20,
L1 = curr - 40 luma offset on a 32×32 tile checker) the encoder picks
weighted-BI on every block and reconstructs to PSNR_Y = inf (perfect
bit-for-bit reconstruction, ≥ 58 dB headline target). With WP off the
slice header carries one extra zero bit per slice (the
`wp_present_flag = 0`); the wire is otherwise byte-for-byte
compatible with round-62.

**Round-64 wires Decoder-side Motion Vector Refinement (DMVR)** per
§8.5.3.2.4 + §8.5.3.2.5. For bi-pred merge CUs whose two references
bracket the current picture symmetrically in POC space (and none of
BCW / weighted-pred / CIIP / sub-block merge / affine is engaged), the
new `dmvr` module runs a 2-pass refinement around the initial merge MV:
a 5×5 integer-pel search using the spec's opposite-direction pairing
`(MV0 + δ, MV1 − δ)` driven by the §8.5.3.2.5 bilateral-matching SAD,
followed by a per-axis 3-point parabolic half-pel pass clamped to
±½-pel. The refined MV pair feeds the existing `predict_luma_block`
motion compensation. Public API surface: `dmvr_used_flag(...)` (the
12-bullet §8.5.3.2.4 step-1 gate), `bilateral_matching_sad(...)`,
`refine_mv_pair(...)`, and `apply_dmvr(...)`. On a synthetic
symmetric-bipred fixture (refs shifted by ±1 sample around a hidden
"truth" plane, integer-pel BM optimum at `δ = (-1, 0)`) DMVR converges
exactly on that delta and the resulting bi-pred MC reaches PSNR_Y =
inf dB vs a DMVR-off baseline of 52.39 dB — well past the +1.5 dB
headline improvement target.

**Round-65 lands the §8.5.5.9 affine sub-block motion compensation
scaffold** (4-parameter + 6-parameter affine motion models plus
Tables 30 / 31 / 32 affine-mode luma interpolation filters). The new
`affine` module exposes `MotionModel` (the §7.4.10.5 Table 15
`MotionModelIdc` triple), `AffineCpmvs` (the 2 / 3 control point MV
record per `numCpMv = motionModelIdc + 1`), `derive_subblock_mvs(...)`
implementing eqs. 850 – 875 (the `(mvScaleHor, mvScaleVer)` base, the
four partials `(dHorX, dVerX, dHorY, dVerY)` with the 4-parameter
similarity constraint `dHorY = -dVerX, dVerY = dHorX`, sub-block centre
sampling `(xPosCb, yPosCb) = (2 + 4*sbIdxX, 2 + 4*sbIdxY)`, §8.5.2.14
`>> 7` rounding, and the `Clip3(-2^17, 2^17 - 1, ·)` final clip),
`fallback_mode_triggered(...)` per eqs. 858 – 867 (bxWX4 * bxHX4 ≤ 225
under bi-pred or per-axis 165 under uni-pred), `predict_luma_subblock_affine(...)`
(separable 8-tap MC for one sub-block against any of the three
affine luma filter sets), and `predict_luma_block_affine(...)` (full-CU
driver that walks the 4×4 sub-block grid and dispatches each through
the per-sub-block MV). On a zoom fixture (uniform 1.5 % shrink + small
translation on a 32×32 CU) the 6-parameter affine reaches PSNR_Y =
**53.75 dB** vs the best 5×5 translational MV search at 42.88 dB
(**+10.87 dB**); on a horizontal-shear fixture the 6-parameter affine
reaches **52.32 dB** vs translational's 34.88 dB (**+17.44 dB**).
Affine merge / AMVP candidate list construction, affine sub-block
chroma MC (eqs. 876 – 879), PROF (§8.5.5.10), and the §8.8.3.4
sub-block boundary deblock propagation remain deferred.

**Round-78 lands §8.5.6.4 PROF (Prediction Refinement with Optical
Flow)** as a per-pixel refinement on top of the round-65 affine
sub-block MC. The new helpers expose `cb_prof_flag_lx(...)` for the
§8.5.5.9 cbProfFlagLX derivation (four false-conditions gate:
`ph_prof_disabled_flag == 1`, fallback mode triggered, translational-
degenerate CPMVs, `RprConstraintsActiveFlag == 1`),
`derive_prof_diff_mv_array(...)` for the §8.5.5.9 eqs. 880 – 887
per-pixel motion-vector-difference array (`posOffsetX = 6*dHorX +
6*dHorY`, eqs. 885 / 886 raw values, §8.5.2.14 `rightShift = 8`
signed-magnitude rounding, eq. 887 `Clip3(-31, 31)`), the new
`predict_luma_subblock_affine_high_precision(...)` helper that
produces the §8.5.6.4 `(sbW + 2) × (sbH + 2)` `BitDepth + 6` precision
predSamplesLXL halo'd block, `apply_prof_to_subblock(...)` for the
§8.5.6.4 eqs. 955 – 959 refinement (`shift1 = 6` gradient taps,
`dI = gradH*diffMv[0] + gradV*diffMv[1]`, `dILimit = 1 <<
Max(13, BitDepth + 1)`, `Clip3(-dILimit, dILimit - 1, dI)` added to
the centre sample), and the full-CU driver
`predict_luma_block_affine_prof(...)` that composes the §8.5.6.3
sub-block MC + PROF + final 8-bit uni-pred clamp. The driver short-
circuits to a bit-identical replay of `predict_luma_block_affine`
when the cbProfFlagLX gate reports `false`. On the same round-65
horizontal-shear fixture the PROF-on path reaches **PSNR_Y = 53.97
dB** vs the PROF-off baseline at **52.32 dB** (**+1.65 dB**); the
translational-degenerate and `ph_prof_disabled_flag == 1` paths are
byte-identical to the affine-only driver. Affine merge / AMVP
candidate list construction, affine sub-block chroma MC (eqs.
876 – 879), and the §8.8.3.4 sub-block boundary deblock propagation
remain deferred.

**Round-91 lands the §8.5.5.5 inherited + §8.5.5.6 constructed affine
merge candidate derivation** that feeds the round-65 sub-block MV
machinery + round-78 PROF refinement. The new `affine_merge` module
exposes `derive_inherited_affine_cpmvs(geom, source, numCpMv)` for the
§8.5.5.5 from-neighbour CPMV derivation (both the `isCTUboundary`
bullet — eqs. 736 – 739 reading the neighbour's two bottom-row
sub-block MVs with the eq. 746 / 747 4-parameter similarity forced —
and the regular bullet — eqs. 740 – 743 reading the neighbour CB's
stored `CpMvLX` directly, with eqs. 744 / 745 only consulting `cpMv[2]`
when `MotionModelIdc == 2`; eqs. 748 – 753 emit the inherited CPMVs at
the current CU's origin / top-right / bottom-left positions, eqs. 754 /
755 clip to `[-2^17, 2^17 - 1]`, §8.5.2.14 round with `rightShift = 7`)
and `derive_constructed_affine_merge_candidates(cb_w, cb_h, corners,
flags)` for the §8.5.5.6 six-candidate constructed list (corner-record
inputs reflect the §8.5.5.2 step-6 B2/B3/A2 cascade for top-left,
B1/B0 for top-right, A1/A0 for bottom-left, temporal bottom-right; per
candidate the spec's per-list gate `all-three-predFlagLX == 1 ∧
matching-refIdxLX` runs independently for L0 + L1; Const1..4 are the
3-corner 6-parameter triples — Const1 = (CP0, CP1, CP2), Const2 emits
CP[2] = CP3 + CP0 − CP1, Const3 emits CP[1] = CP3 + CP0 − CP2, Const4
emits CP[0] = CP1 + CP2 − CP3, every derived CP clipped to `[-2^17,
2^17 - 1]`; Const5 is the 4-parameter (CP0, CP1) pair; Const6 is the
4-parameter pair from corners {0, 2} with the eq. 811 / 812 diagonal-
projection top-right derivation; `sps_6param_affine_enabled_flag == 0`
suppresses Const1..4; bcwIdx inherits from corner 0 (Const1/2/3/5/6) or
corner 1 (Const4) when both L0 + L1 sides materialise, else 0). 22
new lib tests cover the §8.5.2.14 rounding edge cases, the `Clip3`
saturation, every Const1..6 assembly + the corresponding "missing
corner" / "refIdx mismatch" / "predFlag mismatch" short-circuits, plus
an end-to-end integration test confirming an inherited CPMV record
drops straight into `affine::derive_subblock_mvs`. Both helpers
operate on pure-data inputs so the CTU walker can wire them up once
per-CB CPMV storage lands; the §8.5.5.2 driver that fuses inherited A /
inherited B / Const1..6 + zero-MV pad into `subblockMergeCandList` and
the SbTMVP (§8.5.5.3 / §8.5.5.4) candidate path remain deferred.

**Round-94 layers the §8.5.5.2 `subblockMergeCandList` insertion order +
`merge_subblock_idx` pick on top.** New `build_subblock_merge_cand_list`
+ `SubblockMergeList::pick` helpers in the `affine_merge` module walk
the spec's step-7 listing byte for byte: slot 0 = SbCol when
`availableFlagSbCol`, then inherited A (A0/A1 cascade), then inherited B
(B0/B1/B2 cascade), then Const1..Const6 in numeric order, then zero-MV
padding to `MaxNumSubblockMergeCand`. Each guard is the spec's
`availableFlagN && i < MaxNumSubblockMergeCand` clip. The zero-pad
follows eqs. 686 – 695 exactly — uni-pred `(refIdxL0=0, predFlagL0=1,
refIdxL1=−1, predFlagL1=0)` on P-slice, bi-pred `(refIdxL0=0,
refIdxL1=0, predFlagL0=1, predFlagL1=1)` on B-slice, every CPMV zero,
`motionModelIdc = 1` ⇒ `Affine4Param`, `bcwIdx = 0`. The pick step is
`subblockMergeCandList[ merge_subblock_idx ]` — a single array index
returning the slot's `SubblockMergeCandidateKind` (`SbCol` /
`InheritedA` / `InheritedB` / `Const(K)` / `Zero`) plus the parallel
`AffineMergeCandidate` payload. The list assembly emits **no
pruning/equality pass** — §8.5.5.2 step 7 specifies no dedup step
(contrast with §8.5.2.4 pairwise-average for regular merge, which is a
separate explicit index pick rather than a comparison loop), so this
module appends without dedup; any future amendment introducing a CPMV
prune would slot into the same routine. 12 new lib tests cover the
empty `max_num_cand = 0` edge case, the all-zero-pad fallback, P-slice
vs B-slice zero-cand layout (eq. 690 / 691 flip), the canonical
SbCol→A→B→Const1..Const2 (5-slot fill) order, SbCol-absent shift,
inherited-A-absent / inherited-B-present, `max_num_cand` short-circuit
clipping, `max_num_cand` clamp to `MAX_SUBBLOCK_MERGE_CAND`,
`pick`-by-index round-trip, out-of-range index rejection, and an
end-to-end test confirming a `pick`ed `AffineMergeCandidate` flows
straight into `affine::derive_subblock_mvs`. The §8.5.5.3 / §8.5.5.4
SbCol record itself + the affine AMVP (§8.5.5.7) path + the §8.5.5.2
step-6 corner-selection cascade (B2/B3/A2, B1/B0, A1/A0 driven by
per-CB neighbour availability) remain deferred — the list helper takes
`InheritedAffineCandidate` and `ConstructedAffineCandidates` shaped
inputs so the CTU walker can drop them in once per-CB CPMV storage and
the cascade walker land.

**Round-100 lands the §8.5.5.2 steps 3 – 6 neighbour / corner-selection
cascade** — the missing wiring between the round-91 derivations and the
round-94 list assembly. `derive_affine_neighbour_cascade(xcb, ycb, cb_w,
cb_h, nbrs, log2_par_mrg_level, sps_affine_enabled_flag)` derives the
step-3 sample locations (eqs. 674 – 680: `A0 = (xCb−1, yCb+cbH)`, `A1 =
(xCb−1, yCb+cbH−1)`, `A2 = (xCb−1, yCb)`, `B0 = (xCb+cbW, yCb−1)`, `B1 =
(xCb+cbW−1, yCb−1)`, `B2 = (xCb−1, yCb−1)`, `B3 = (xCb, yCb−1)`), runs
the step-4 A0 → A1 and step-5 B0 → B1 → B2 first-affine scans (each
picks the FIRST effectively-available neighbour with `MotionModelIdc >
0`, short-circuiting at the spec's `availableFlagN == FALSE` gate), and
computes the step-6 seven `availableA0/A1/A2/B0/B1/B2/B3` corner flags
for §8.5.5.6. The §6.4.4 availability the CTU walker supplies per
`NeighbourBlock` is AND-NOT'd with the parallel-merge-level suppression
(`xCb >> Log2ParMrgLevel == xNbN >> Log2ParMrgLevel && yCb >>
Log2ParMrgLevel == yNbN >> Log2ParMrgLevel` ⇒ `availableN = FALSE`, eq.
60). Corner availability does **not** gate on `MotionModelIdc` (a
translational-but-available neighbour still contributes a constructed
corner MV even though the affine-only inherited scan skips it);
`sps_affine_enabled_flag == 0` short-circuits the whole cascade to no
inherited candidates + all corners unavailable. The cascade's chosen
`inherited_a`/`inherited_b` blocks feed `derive_inherited_affine_cpmvs`
and its `corner_availability` flags feed
`derive_constructed_affine_merge_candidates`. 9 new lib tests cover the
eqs. 674 – 680 positions, the SPS-disable short-circuit, the A0-first /
A1-fallthrough / no-affine-neighbour A-side cases, the B0 → B1 → B2 scan
order, parallel-merge-level same-cell suppression (`Log2ParMrgLevel ==
6` suppresses, `== 2` does not), translational-neighbour corner
inclusion, the per-position corner-flag map, and an end-to-end test
feeding a cascade-chosen inherited-A block's geometry into
`derive_inherited_affine_cpmvs`. The SbTMVP (§8.5.5.3 / §8.5.5.4) SbCol
candidate, the §8.5.5.7 affine AMVP path, and the CTU-walker wire-up
that populates `NeighbourQuery` from live per-CB grids remain deferred.

**Round-103 lands the §7.3.10.10 `mvd_coding()` decode syntax** — the
first concrete step against the "full mvd_coding" lacks-tail item.
`LeafCuReader::read_mvd_coding()` (exposed `pub`) decodes one
motion-vector-difference structure bin-for-bin per §7.3.10.10:
`abs_mvd_greater0_flag[0]` then `[1]` (both ctx-coded against the new
Table 110 slot = `init_type`), `abs_mvd_greater1_flag[c]` (Table 111)
for each component whose greater0 flag is 1, then `abs_mvd_minus2[c]`
(only when greater1 is 1) followed by `mvd_sign_flag[c]` (bypass). It
returns the `lMvd[0..1]` pair packed into a `MotionVector` via eq. 190
`lMvd[c] = greater0 * (abs_mvd_minus2 + 2) * (1 − 2*sign)` with
`abs_mvd_minus2` inferred −1 when greater1 is 0 (so the magnitude
collapses to 1). The magnitude tail uses `read_abs_mvd_minus2()`, a
§9.3.3.14 / §9.3.3.6 *limited* k-th order Exp-Golomb decode (`k = 1`,
`maxPreExtLen = 15`, `truncSuffixLen = 17`, all bins bypass) that
handles the 15-bit prefix-cap escape into a 17-bit truncated suffix —
covering the full spec range up to `|lMvd| = 2^17`. 6 new lib tests
exercise an encoder-mirror round-trip (zero components, unit magnitude
that skips `abs_mvd_minus2`, mixed zero/non-zero, large magnitudes to
the 2^17 max, eq. 190 sign derivation) plus a direct limited-EGk codec
round-trip across the value range including the escape boundary. The
parsed `lMvd` is the raw, pre-AMVR difference; wiring it into a full
non-merge inter CU (which needs the §8.5.2.8/§8.5.2.9 AMVP
MVP-candidate derivation, `ref_idx_lX`, `mvp_lX_flag`, and the §7.4.11.6
AMVR shift) plus the SbTMVP SbCol record and the §8.5.5.7 affine-AMVP
path (`mvd_coding` × numCpMv) remain deferred.

**Round-108 lands the §7.3.10.8 non-merge inter MVP-side syntax** — the
four AMVP-side CABAC reads that surround the round-103 `mvd_coding()` in
the MODE_INTER `coding_unit()` else-branch.
`LeafCuReader::read_inter_pred_idc(cb_width, cb_height)` decodes the
§9.3.3.9 / Table 131 binarisation (`PRED_L0 = 00`, `PRED_L1 = 01`,
`PRED_BI = 1` when `cbWidth + cbHeight > 12`; a single bin with
`PRED_BI` suppressed when the sum equals 12) into a new
`leaf_cu::InterPredDir` enum, with bin 0's ctxInc =
`7 - ((1 + Log2(cbWidth) + Log2(cbHeight)) >> 1)` (or 5) and bin 1's
ctxInc fixed 5 per Table 132. `read_sym_mvd_flag()` (Table 86, FL
`cMax = 1`), `read_ref_idx_lx(num_ref_idx_active)` (Table 87, TR
`cMax = NumRefIdxActive[X] - 1`, bins 0/1 ctx-coded then bypass), and
`read_mvp_lx_flag()` (Table 88, FL `cMax = 1`) complete the set. The
new `SyntaxCtx::{InterPredIdc, SymMvdFlag, RefIdxLx, MvpLxFlag}` carry
Tables 83 / 86 / 87 / 88 with the Table 51 per-initType slot blocks
(`(initType-1)*6` / `initType-1` / `(initType-1)*2` / `initType`). 9 new
lib tests round-trip every reader through a bin-for-bin encoder mirror.
Fusing these reads into a complete non-merge inter CU walk (the
§8.5.2.8 AMVP candidate list that consumes `mvp_lX_flag`, the §7.4.11.6
AMVR shift on `lMvd`, the `ph_mvd_l1_zero_flag` / `RefIdxSymL{0,1}`
inference around `sym_mvd_flag`, and the `inter_affine_flag` /
`cu_affine_type_flag` / `bcw_idx` / `amvr_flag` branches) remains
deferred.

**Round-129 lands the §7.3.10.5 `bcw_idx` gate evaluator + `MvField`
fuse** — the round-126 reader-side follow-up call-out. New
`leaf_cu::BcwIdxGate` packs the spec's seven gate inputs
(`sps_bcw_enabled`, `inter_pred_idc`, `luma_weight_l0/l1_flag`,
`chroma_weight_l0/l1_flag`, `cb_width * cb_height`,
`no_backward_pred_flag`) into one pure-data struct;
`BcwIdxGate::is_open()` reproduces the §7.3.10.5 conditional verbatim
(`sps_bcw_enabled_flag && inter_pred_idc == PRED_BI && every per-list
weighted-prediction flag == 0 && cbWidth * cbHeight >= 256`). New
`LeafCuReader::read_bcw_idx_gated(gate)` invokes the round-126
`read_bcw_idx` with the threaded `no_backward_pred_flag` (`cMax = 4`
vs `2`) when the gate opens and returns 0 without consuming any bins
when closed (per §7.4.12.5 "When `bcw_idx[x0][y0]` is not present, it
is inferred to be equal to 0"). `LeafCuReader::read_bcw_idx_into(gate,
&mut MvField)` adds the spec's `BcwIdx[x0][y0] = bcw_idx[x0][y0]`
array-write step, clearing stale `MvField::bcw_idx` to 0 when the
gate is closed (the CTU walker then broadcasts that value across
every covered 4x4 block, matching the existing per-block default in
the `crate::ctu` / `crate::affine_merge` inferred paths). 9 new lib
tests pin the gate truth table: open with all preconditions met,
closes on `sps_bcw_enabled = false`, closes on uni-pred / `None`
`inter_pred_idc`, closes on any single weighted-prediction flag set
(all four individually checked), the `>= 256` area threshold
exhaustively swept (16x16 / 8x32 / 32x8 open; 8x16 / 16x8 / 8x8 /
4x32 close), the closed-gate read leaves the bitstream pointer parked
at a sentinel bypass bit, the open-gate read returns the decoded
value, the `MvField` writer overwrites a stale `bcw_idx = 7` with the
decoded 2, the closed-gate `_into` path clears a stale value to 0,
and the `no_backward_pred_flag = true` path threads `cMax = 4`
through to a value-4 round-trip. The §7.3.10.5 encoder-side emission
for the round-58 / round-60 BCW-RDO winners and the CTU-walker
drop-in that fills `BcwIdxGate` from live per-CB state (round-29
pred-weight-table walker output + round-108 `inter_pred_idc` /
`ref_idx_lX` parse) remain a follow-up.

**Round-126 lands the §7.3.10.5 `bcw_idx` CABAC reader** — the last
unparsed AMVP-side bin in the `coding_unit()` else-branch.
`LeafCuReader::read_bcw_idx(no_backward_pred_flag)` decodes the
§7.4.12.5 syntax element per Tables 91 / 131 / 132: TR with
`cMax = NoBackwardPredFlag ? 4 : 2`, `cRiceParam = 0`; bin 0 is
context-coded against the new `SyntaxCtx::BcwIdx` 2-entry bundle
(slot = `init_type - 1`; Table 91 `initValue/shiftIdx = (4, 1)` for
initType 1, `(5, 1)` for initType 2; never signalled in I slices)
with `ctxInc = 0`; bins 1.. are bypass-coded; the truncation point
at `cMax - 1` ones sends no terminating zero on the max-value path.
The caller is responsible for the §7.3.10.5 gate
(`sps_bcw_enabled_flag && inter_pred_idc == PRED_BI && no per-list
weighted-prediction flags set on the chosen reference indices &&
cbWidth * cbHeight >= 256`); when closed the value is inferred 0 per
§7.4.12.5. The returned `bcw_idx` flows into the round-29
`inter::bcwWLut[k] = {4, 5, 3, 10, -2}` weight lookup once the CTU
walker drops the per-CU read into the non-merge inter path. 6 new
lib tests round-trip every legal value across both `cMax` cases on
both non-I initTypes, prove `bcw_idx == 0` reads exactly one
context bin (zero bypass tail) via a sentinel-byte check, and pin
the Table 91 transcription. The §7.3.10.5 gate evaluator that
pushes the decoded value into `MvField::bcw_idx` and the matching
encoder-side emission for the round-58 / round-60 BCW-RDO winners
remain a follow-up.

**Round-132 lands the §8.5.5.3 / §8.5.5.4 SbTMVP record + availability
gate** — the first SbCol-side concrete data path for the
`subblockMergeCandList` slot 0 the round-94 `affine_merge` driver
already reserves. New `sbtmvp` module exposes: `SbTmvpAvailability`
(the §8.5.5.3 first-bullet inputs: `sps_sbtmvp_enabled`,
`ph_temporal_mvp_enabled`, `cb_width`, `cb_height`,
`col_pic_present`); `is_sbtmvp_available(g)` returning `true` iff all
four spec conditions plus the collocated-picture presence hold
(`cbWidth >= 8 && cbHeight >= 8 && both flags == 1 && ColPic
present`); `SbTmvpCenterLoc::derive(xcb, ycb, cb_w, cb_h,
ctb_log2_size_y)` for §8.5.5.3 eqs. 711 – 714 (`(xCtb, yCtb) = (xCb,
yCb) >> CtbLog2SizeY << CtbLog2SizeY`, `(xCtrCb, yCtrCb) = (xCb +
cbW/2, yCb + cbH/2)`); `SbTmvpGrid::derive(cb_w, cb_h)` for eqs. 715 –
718 (`numSbX = cbW >> 3`, `sbWidth = sbHeight = 8`) plus
`subblock_centre(xcb, ycb, xs_idx, ys_idx)` for eqs. 720 / 721;
`derive_temp_mv(...)` for §8.5.5.4 (zero-init `tempMv`, prefer
`mvL0A1` when `predFlagL0A1 && DiffPicOrderCnt(ColPic,
RefPicList[0][refIdxL0A1]) == 0`, B-slice fallback to `mvL1A1`
similarly, then §8.5.2.14 rounding with `rightShift = 4, leftShift =
0` — note `leftShift = 0` discards the 1/16-pel fraction so the
returned `tempMv` is an integer-luma-sample offset, distinct from the
AMVR `rightShift == leftShift` path); `PictureBoundary::{Picture,
Subpic}` + `clip_col_subblock_location` / `clip_col_centre_location`
for §8.5.5.3 eqs. 722 – 724 and §8.5.5.4 eqs. 729 – 731 (CTB-aligned
lower bound, picture-or-subpic right bound, both branches of
`sps_subpic_treated_as_pic_flag`); and `SbTmvpRecord` carrying
`col_pic_poc`, `centre`, `grid`, `refIdxLXSbCol = 0` (eq. 719),
`temp_mv`, plus the walker-populated `(ctrPredFlagLX, ctrMvLX)` fields
with the `is_sb_col_available() == ctr_pred_flag_l0 ||
ctr_pred_flag_l1` helper for the §8.5.5.3 step-3 final decision. 31
new lib tests pin the gate truth table (each of the five
close-conditions checked individually + the all-open path + the exact
8×8 boundary), the centre-loc derivation (CTU-aligned + non-aligned +
floor-div on odd dims), the grid geometry (8×8 single-subblock case,
32×16 4×2 grid, below-8 zero-grid edge case, sub-block centre
arithmetic), the tempMv derivation (A1-unavailable zero, L0-match pick,
B-slice L1 fallback, P-slice no L1 fallback, predFlag-off branching,
§8.5.2.14 rounding at 1/4-pel / 1/2-pel / 3/4-pel inputs), the Clip3
helpers (in-bounds passthrough, CTB-upper clamp, CTB-lower clamp,
subpic branch), and the `SbTmvpRecord` defaults /
`is_sb_col_available` truth table. The non-merge inter CU walk that
fuses SbTMVP into the live CTU dispatch (sourcing `ColPic`, the A1
query, and the per-sub-block §8.5.2.12 collocated-MV reads for
`mvLXSbCol` / `predFlagLXSbCol`) remains the next milestone.

**Round-135 lands the §8.5.5.3 CTU-walker fuse: per-sub-block motion
fill** — the main-body loop the round-132 record reserved.
`fill_subblock_motion(record, inputs, col_sampler)` iterates the
`numSbX × numSbY` 8×8 grid; for each sub-block it derives the centre
(eqs. 720 / 721), clips `(xSb + tempMv[0], ySb + tempMv[1])` (eqs.
722 – 724), snaps to the 8×8 collocated cell `(xColCb, yColCb) =
((xColSb >> 3) << 3, (yColSb >> 3) << 3)`, samples the collocated
picture's per-cell motion via a `ColMotionSampler` closure, runs
§8.5.2.12 with `sbFlag = 1` per list (the `predFlagColLX == 1` LX
path, the `NoBackwardPredFlag && predFlagColLY == 1` cross-list LY
fallback, §8.5.2.15 integer-pel buffer compression, and the eqs.
598 – 605 POC scaling with the eq. 600 equal-distance passthrough),
and applies eqs. 725 / 726 — when both list reads report
`predFlagLXSbCol == 0` the sub-block inherits the record's CU-centre
default `ctrMvLX` / `ctrPredFlagLX` (the intra / unavailable
fallback). New types: `ColBlockMotion` (the per-cell collocated
read-back), `ColMotionSampler`, `SbTmvpFuseInputs` (CU origin,
`CtbLog2SizeY`, clip boundary, slice type, `NoBackwardPredFlag`, POC
operands + the `poc_of_col_ref(listCol, refIdxCol)` resolver),
`SbColMotion`, and `SbColGrid` (row-major fill + `at(xSbIdx,
ySbIdx)`). L1 reads gate on `sh_slice_type == B`. 7 new lib tests
pin the uniform-field fill, the all-intra centre fallback, a mixed
inter/intra grid, the B-slice bi-pred both-lists fill, the P-slice
L1-suppression with L0 borrowing L1 via `NoBackwardPredFlag`, the
eqs. 601 – 605 POC scaling, and the `tempMv` collocated-sample
offset. The §7.4.6 `merge_subblock_flag` reader wire-up for live
SbCol selection + the encoder-side SbCol emission remain follow-ups.

**Round-146 lands the §7.3.11.7 `merge_data()` wire-up of the
round-139 `merge_subblock_flag` + `merge_subblock_idx` readers
behind the round-142 §7.4.3.4 eq. 85 gate** — the leaf-CU inter
path (`LeafCuReader::decode_inter`) now opens with the
subblock-merge prologue exactly as written in §7.3.11.7
(V4, 01/2026): when `MaxNumSubblockMergeCand > 0 && cbWidth >= 8
&& cbHeight >= 8` the reader takes the new
`read_merge_subblock_flag` ctx bin; when the flag decodes to 1 it
then takes `read_merge_subblock_idx(MaxNumSubblockMergeCand)`
(suppressed to 0 when `MaxNumSubblockMergeCand <= 1`) and short-
circuits the regular / MMVD / CIIP / GPM tree (per §7.4.12.7
`regular_merge_flag = general_merge_flag &&
!merge_subblock_flag`); when the flag decodes to 0 the reader
falls through to the round-21 / 27 / 28 / 40 path. The §7.4.12.7
inference `merge_subblock_flag = 0` covers both gate-closed cases
(`MaxNumSubblockMergeCand == 0`, the §7.4.3.4-eq.-85 zero output
when neither affine nor (SbTMVP × ph_temporal_mvp) is on, OR a
side `< 8`). New `MergeData` fields `merge_subblock_flag: bool` +
`merge_subblock_idx: u32` record the parse. New `CuToolFlags`
field `max_num_subblock_merge_cand: u32` is populated from
`SeqParameterSet::max_num_subblock_merge_cand(ph_temporal_mvp_enabled_flag)`
via `CtuWalker::cu_tool_flags()` so the CU-side gate stays in
lockstep with the SPS-side derivation. Per-CB neighbour state for
the §9.3.4.2.2 / Table 133 ctxInc (`cond{L,A} =
MergeSubblockFlag[{L,A}] || InterAffineFlag[{L,A}]`) is not yet
tracked in `CuNeighbourhood`; the wire-up passes the §7.4.12.7
defaults `(false, false)` for both neighbours, matching the
pre-r146 stub-call pattern. 7 new lib tests pin (a) the
gate-closed-by-size path (`cb_width = 4`, no bin consumed,
flag inferred 0), (b) the gate-closed-by-`max_cand == 0` path
(same outcome), (c) the gate-open path with
`merge_subblock_flag == 0` falling through cleanly, (d) the
gate-open path with `merge_subblock_flag = 1` and
`merge_subblock_idx = 0` decoding clean with regular / MMVD /
CIIP / GPM all bypassed and CBFs all 0, (e) the same with
`merge_subblock_idx = 3` on `init_type 2` exercising the cMax = 4
TR ctx-bin + bypass tail, (f) the `MaxNumSubblockMergeCand = 1`
idx-suppression path (no idx bin on the wire), and (g) the
`CuToolFlags::default()` `max_num_subblock_merge_cand == 0`
invariant so pre-r146 intra-only tests never open the new gate.
Encoder-side subblock-merge emission + per-CB live-neighbour grid
for the Table 133 ctxInc + the `cu_coded_flag = 1`
`transform_tree()` body for subblock-merge CUs remain follow-ups.

**Round-152 lands the §7.3.11.7 `inter_affine_flag[x0][y0]` CABAC
reader + Table 84 context bundle** — the second round-149 follow-up.
The new `SyntaxCtx::InterAffineFlag` ships Table 84's transcription
verbatim (`initValue = [12, 13, 14, 19, 13, 6]`,
`shiftIdx = [4, 0, 0, 4, 0, 0]` — 6 ctxIdx, 3 per non-I initType).
`ctx_inc_inter_affine_flag(...)` derives the ctxInc per §9.3.4.2.2 /
eq. 1551 with the Table 133 row whose `condL` / `condA` predicates are
identical to `merge_subblock_flag` (the spec lists the two syntax
elements side-by-side under a single definition); the new helper
delegates to `ctx_inc_merge_subblock_flag` so the two derivations
cannot drift apart by accident. A new
`LeafCuCtxs::inter_affine_flag: Vec<ContextModel>` bundle is
initialised from Table 84 alongside the round-139
`merge_subblock_flag` bundle, and `LeafCuReader::read_inter_affine_flag`
implements the FL `cMax = 1` single ctx-coded bin reader per Table
132, indexing `(init_type - 1) * 3 + ctxInc` against the per-initType
triplet. The reader contract is documented for the §7.3.11.7 gates the
caller is responsible for: `sps_affine_enabled_flag && cbWidth >= 16
&& cbHeight >= 16` AND the surrounding `general_merge_flag == 0` (the
syntax element only appears on the non-merge inter branch); when any
gate is closed §7.4.12.7 infers it to 0 and the reader must NOT be
invoked. No CTU-walker wire-up yet — this round adds the bin-level
parser only. The live consumer is the future non-merge affine inter
CU walk, which will be the one-line drop-in that finally populates
the round-149 `inter_affine_grid` with non-default values. 12 new lib
tests pin Table 84 length + bit-exact init/shift, an exhaustive 64-case
equality between `ctx_inc_inter_affine_flag` and
`ctx_inc_merge_subblock_flag` over every combination of the six
neighbour-state inputs, the three §9.3.4.2.2 sentinel cases
(no-neighbours → 0, one-active → 1, both-active → 2),
encoder-mirror round-trip across both initTypes for each case,
unavailable-neighbour masking, per-ctx slot addressability for every
legal `(init_type, ctxInc)` pair, and a bundle-isolation test
confirming the `inter_affine_flag` and `merge_subblock_flag` CABAC
state machines are disjoint and initialise to different `pState`
rows. The non-merge affine inter CU walker that would actually parse
`inter_affine_flag` on the wire + populate `inter_affine_grid`
remains the next milestone.

**Round-149 lands the per-CB live `MergeSubblockFlag[x][y]` /
`InterAffineFlag[x][y]` neighbour grid fuse into
`CtuWalker::compute_cu_neighbourhood`** — the first round-146
follow-up. Two picture-wide 4×4 grids (`subblock_merge_grid`,
`inter_affine_grid`) sharing the existing round-28 §8.5.6.7 intra-
grid geometry now drive the §9.3.4.2.2 / Table 133 `cond{L,A} =
MergeSubblockFlag[{L,A}] || InterAffineFlag[{L,A}]` ctxInc input
for `read_merge_subblock_flag`. Four new `CuNeighbourhood` fields
(`left_merge_subblock`, `above_merge_subblock`, `left_inter_affine`,
`above_inter_affine`) replace the pre-r149 hard-coded `(false,
false)` stub in the §7.3.11.7 wire-up. A single-source-of-truth
`commit_subblock_neighbour_state(cu, info)` helper is invoked from
both the syntax-only path (`decode_ctu_full`) and every CU-completion
site in the reconstruction path (intra single-TB, ISP, regular-merge
inter, GPM) so the next CU's neighbour query reads the live state.
`compute_cu_neighbourhood` samples both grids at the same
`(xCb − 1, yCb)` / `(xCb, yCb − 1)` 4×4 cells the round-21
cu_skip_flag / pred_mode ctxInc already use, and the §7.3.11.7
wire-up now feeds the parsed flags straight into
`read_merge_subblock_flag`. The non-merge affine inter path is not
yet parsed by the CTU walker, so the `inter_affine_grid` always
reads back `false` (matching the §7.4.12.7 inference); the field is
plumbed end-to-end so the eventual affine-inter walker is a
one-line drop-in (`write_inter_affine_block` already exists). 7 new
lib tests pin (a) both grids default to false + out-of-bounds
returns false per §6.4.4 unavailability, (b)
`write_subblock_merge_block` broadcasts across every covered 4×4
cell with surrounding cells unchanged, (c)
`compute_cu_neighbourhood` reads the left-of-(16, 0) and
above-of-(0, 16) cells correctly with picture-edge unavailability
folded in, (d) the `inter_affine_grid` plumbs through the same
neighbour positions, (e) a CU committed with
`merge_subblock_flag = 1` flips the per-CB grid so a CU at (16, 0)
reads `left_merge_subblock = true`, (f) an intra CU commit clears a
stale pre-loaded `true` (the §7.4.12.7 inference for non-inter CUs),
and (g) un-loaded neighbours read `(false, false)` so the pre-r149
stub-call path stays byte-identical on every existing fixture.
Encoder-side subblock-merge emission, the `cu_coded_flag = 1`
`transform_tree()` body for subblock-merge CUs, the non-merge affine
inter CU walk that would actually write into `inter_affine_grid`,
and the rest of the non-merge inter CU walk remain follow-ups.

**Round-142 lands the §7.4.3.4 eq. 85 `MaxNumSubblockMergeCand`
derivation** — the SPS-side scalar that drives the round-139
`merge_subblock_idx` `cMax = MaxNumSubblockMergeCand − 1` truncated-Rice
binarisation and the §7.3.11.7 size-gate `MaxNumSubblockMergeCand > 0`
test. New `SeqParameterSet::max_num_subblock_merge_cand(ph_temporal_mvp_enabled_flag)`
reproduces the spec derivation verbatim: when `sps_affine_enabled_flag
== 1` it returns `5 − sps_five_minus_max_num_subblock_merge_cand` and
ignores the PH input; otherwise it returns the boolean
`sps_sbtmvp_enabled_flag && ph_temporal_mvp_enabled_flag` (`0` or `1`).
Both branches clamp into the §7.4.3.4 trailing "0 to 5, inclusive"
range, so even an out-of-range
`sps_five_minus_max_num_subblock_merge_cand = 6+` yields a legal `0`. A
sibling `SeqParameterSet::max_num_merge_cand()` adds the §7.4.3.4
regular-merge derivation `6 − sps_six_minus_max_num_merge_cand` clamped
to `[1, 6]` (the same value the SPS-parse uses inline to gate
`gpm_enabled_flag` / `max_num_merge_cand_minus_max_num_gpm_cand`),
exposed as a public scalar so downstream `MaxNumGpmMergeCand`
derivation has one source of truth. 6 new lib tests cover the affine
branch full range (`five_minus = 0..=5 → 5..=0`, both PH polarities
since PH is ignored), the affine-branch out-of-range clamp (`6 → 0`,
`100 → 0`), the non-affine branch truth table (sbtmvp & PH on → 1,
either off → 0, the SPS-only `ph = false` lower-bound case → 0), an
exhaustive sweep proving the result stays within `[0, 5]` across every
combination of the three inputs, the `merge_subblock_idx`-driving
values (`5/2/1/0` for `five_minus = 0/3/4/5`), and the full
`MaxNumMergeCand` `1..=6` mapping plus clamp. The §7.3.11.7
`merge_data()` wire-up that calls the round-139 `read_merge_subblock_flag`
/ `read_merge_subblock_idx` behind this gate + the encoder-side
emission remain follow-ups.

**Round-139 lands the §7.3.11.7 `merge_subblock_flag` +
`merge_subblock_idx` CABAC readers** — the live-stream entry into the
round-135 SbTMVP / round-94 sub-block-merge-cand-list driver. The
`tables` module gains `SyntaxCtx::MergeSubblockFlag` (Table 107: 6
ctxIdx — 3 per non-I initType, initValue `[48, 57, 44, 25, 58, 45]`,
shiftIdx all 4) and `SyntaxCtx::MergeSubblockIdx` (Table 108: 2
ctxIdx, initValue `[5, 4]`, shiftIdx all 0); the `ctx` module gains
`ctx_inc_merge_subblock_flag(left_msb, left_aff, avail_l, above_msb,
above_aff, avail_a)` implementing §9.3.4.2.2 / eq. 1551 with the
Table 133 merge-side row `cond{L,A} = MergeSubblockFlag[{L,A}] ||
InterAffineFlag[{L,A}]`, `ctxSetIdx = 0` — yielding
`ctxInc ∈ {0, 1, 2}` after the §6.4.4 availability mask folds in.
`LeafCuReader::read_merge_subblock_flag(...)` decodes the FL `cMax = 1`
ctx-coded bin against the `(init_type − 1) * 3 + ctxInc` slot;
`LeafCuReader::read_merge_subblock_idx(max_num_subblock_merge_cand)`
decodes the TR sub-block-merge-candidate index (`cMax =
MaxNumSubblockMergeCand − 1`, `cRiceParam = 0`) — bin 0 ctx-coded
against the Table 108 slot `init_type − 1` with `ctxInc = 0`, bins
1.. bypass-coded; returns 0 without consuming bits when
`max_num_subblock_merge_cand ≤ 1` (the §7.4.12.7 inference for the
single-candidate case). 16 new lib tests pin the Table-133 ctxInc
truth table (no-neighbours / one-active / both-active /
availability-masked), the Tables 107 + 108 initValue / shiftIdx
transcription bit-exact, both reader round-trips on both non-I
initTypes through an encoder mirror (no-neighbour and active-
neighbour ctxInc selection), the `cMax = 1` and full `cMax = 4` TR
ranges of `merge_subblock_idx`, the `MaxNumSubblockMergeCand ≤ 1`
suppression branch (sentinel-byte non-consumption check), the value-0
exact-one-ctx-bin path, and the per-initType slot-addressability
guard for Table 107. The §7.3.11.7 merge_data wire-up that calls
these readers behind the size gate (`MaxNumSubblockMergeCand > 0 &&
cbW >= 8 && cbH >= 8`) + the SPS-side `MaxNumSubblockMergeCand`
derivation per §7.4.3.4 eq. 85 + the encoder-side emission remain
follow-ups.

**Round-159 lands the §7.3.11.7 `cu_affine_type_flag` CABAC reader +
Table 85 context bundle** — the second of the two non-merge affine
syntax elements that drive the §8.5.5.2 `MotionModelIdc` derivation
(round-152 added the first, `inter_affine_flag`). When
`sps_6param_affine_enabled_flag == 1` and the parser has just decoded
`inter_affine_flag == 1`, the §7.3.11.7 inter else-branch signals
`cu_affine_type_flag` to pick between 4-parameter (`0 →
MotionModelIdc = 1`, two CPMVs at corners A / B) and 6-parameter
(`1 → MotionModelIdc = 2`, three CPMVs at corners A / B / C) affine
motion via eq. 160. The `tables` module gains
`SyntaxCtx::CuAffineTypeFlag` (Table 85: 2 ctxIdx — one per non-I
initType — `initValue = [35, 35]`, `shiftIdx = [4, 4]` transcribed
bit-exact); the `ctx` module gains `ctx_inc_cu_affine_type_flag()`
which is the deterministic `0` per Table 132 / Table 133 (the spec
entry simply lists "0" — no §9.3.4.2.2 neighbour-lookup applies, so
the per-initType slot is picked solely via `init_type − 1`).
`LeafCuReader::read_cu_affine_type_flag()` decodes the FL `cMax = 1`
single ctx-coded bin per Table 132 against the matching Table 85
slot; the helper routes through `ctx_inc_cu_affine_type_flag` for
spec traceability so a future Table 133 amendment that introduces a
non-trivial derivation is caught in one place. 8 new lib tests pin
the Table 85 length + bit-exact `init` / `shift`, the
`ctx_inc_cu_affine_type_flag` deterministic-`0` value, encoder-mirror
round-trip across both non-I initTypes for both flag values, the
per-initType slot addressability, bundle isolation against the
round-152 `inter_affine_flag` Table 84 state machine (Table 85's
identical-row pState vs Table 84's diverging row pStates; driving
one bundle does not perturb the other), the Table 85 identical-row
spec pin (initValue `35` / shiftIdx `4` for both ctxIdx), and an
independent-stream coverage sweep proving the reader doesn't depend
on a prior decision on a different bundle. The §7.3.11.7 wire-up
that calls this reader behind the `sps_6param_affine_enabled_flag &&
inter_affine_flag == 1` gate + the encoder-side emission + the
§8.5.5.2 eq. 160 `MotionModelIdc = inter_affine_flag +
cu_affine_type_flag` write into the round-149 affine grid remain
follow-ups.

**Round-164 lands the §7.3.11.7 non-merge inter affine-syntax
dispatcher + §8.5.5.2 eq. 160 fold** — `LeafCuReader::read_non_merge_inter_affine`
composes the round-152 `read_inter_affine_flag` and round-159
`read_cu_affine_type_flag` CABAC readers under the §7.3.11.7
gating cascade and folds the two decisions through
`crate::affine::derive_motion_model_idc` (the spec's
`MotionModelIdc = inter_affine_flag + cu_affine_type_flag`
integer sum) into a typed `MotionModel`. The dispatcher takes a
`NonMergeInterAffineGate` struct that bundles the §7.3.11.7
outer gate (`sps_affine_enabled_flag && cbWidth >= 16 &&
cbHeight >= 16`) + inner 6-param gate
(`sps_6param_affine_enabled_flag && inter_affine_flag == 1`) +
the four neighbour bits the §9.3.4.2.2 / Table 133 ctxInc
derivation reads (left / above `MergeSubblockFlag` /
`InterAffineFlag` + availability), and returns a
`NonMergeInterAffineDecision { inter_affine_flag,
cu_affine_type_flag, motion_model }`. §7.4.12.7 inferences
(value-not-present ⇒ 0) are applied uniformly so the caller can
write the decision into the per-CB grid without branching. The
`affine` module gains `MotionModel::from_idc` (inverse of `idc`,
rejects out-of-range values per Table 15) and
`derive_motion_model_idc` (the spec's eq. 160 truth table as a
standalone pure function). 15 new lib tests pin the eq. 160
truth table for every `(inter_affine_flag, cu_affine_type_flag)`
combination, the `from_idc` round-trip + out-of-range rejection,
the dispatcher's translational-on-outer-gate-closed paths (both
SPS and block-size triggers), the inferred 4-param branch when
`sps_6param_affine_enabled == 0 && inter_affine_flag == 1`, the
full 6-param path with both bins driven, the
round-trip-both-init-types matrix across all five reachable
(sps_6p, inter_affine, cu_affine_type) tuples, the neighbour-
state ctxInc threading through the inter_affine_flag CABAC
ctxInc derivation (the 0 / 1 / 2 ctxInc levels via the
`cond{L,A} = MergeSubblockFlag[N] || InterAffineFlag[N]` Table
133 row), the §7.4.12.7 inferences against the decision struct,
and the `outer_affine_gate_open` / `inner_6param_gate_open`
gate-test helpers in isolation. The CTU walker call-site that
threads the per-CB neighbour state into this dispatcher + the
write of `MotionModelIdc` / `inter_affine_flag` /
`cu_affine_type_flag` into the round-149 affine grid + the
encoder-side emission remain follow-ups for the broader
non-merge inter CU walker.

**Round-177 lands the encoder-side mirror of the round-164
dispatcher** — the new `affine_syntax_enc` module exposes
`encode_inter_affine_flag` (Table 84, ctx slot
`(init_type - 1) * 3 + ctxInc` with the §9.3.4.2.2 / Table 133
shared `condL = MergeSubblockFlag[L] || InterAffineFlag[L]`
neighbour derivation) and `encode_cu_affine_type_flag` (Table 85,
deterministic `ctxInc = 0` per Table 133, slot `init_type - 1`)
plus the dispatcher `encode_non_merge_inter_affine(enc, ctxs, gate,
decision)` that applies the same §7.3.11.7 gating cascade as the
reader: emits one `inter_affine_flag` bin only when the outer gate
(`sps_affine_enabled_flag && cbWidth >= 16 && cbHeight >= 16`)
opens, then emits one `cu_affine_type_flag` bin only when the
inner 6-param gate (`sps_6param_affine_enabled_flag &&
inter_affine_flag == 1`) opens against the effective
`inter_affine_flag`. When a gate is closed the encoder emits zero
bins — matching the reader's §7.4.12.7 inference path — so the
wire layout round-trips bit-identically through
`LeafCuReader::read_non_merge_inter_affine`. A convenience
constructor `make_non_merge_inter_affine_decision(inter_affine_flag,
cu_affine_type_flag)` folds `motion_model` through
`derive_motion_model_idc` (§8.5.5.2 eq. 160) so the encoder side
cannot drift its typed enum from the raw flag pair, and a
spec-traceability re-export
`ctx_inc_shared_merge_subblock_inter_affine` surfaces the shared
Table 133 row both `merge_subblock_flag` and `inter_affine_flag`
consume. 13 new lib tests pin the truth table for
`make_non_merge_inter_affine_decision` over all four flag pairs
(including the unreachable `(false, true)` defensive mapping), the
outer-gate-closed-by-SPS + outer-gate-closed-by-block-size
zero-bin round-trips, the inner-gate-closed (sps_6p = 0) one-bin
round-trip recovering `Affine4Param`, the both-gates-open
round-trips for `Affine4Param` + `Affine6Param`, the 16-tuple
`(left_msb, left_aff, above_msb, above_aff)` neighbour-state
sweep, the §6.4.4 unavailable-neighbour masking round-trip across
both non-I initTypes for every reachable flag pair, a six-case
`(gate, decision)` sweep across both initTypes, the exhaustive
64-case shared-row identity check, and two byte-stream identity
pins (zero-bin emission under closed outer gate equals a
no-dispatch terminator-only stream; one-bin emission under closed
inner gate equals a direct `encode_inter_affine_flag` call). The
CTU-walker / encoder-pipeline call-site that consumes this surface
+ the per-CB affine-grid write at encode time remain follow-ups
for the broader non-merge inter CU encoder.

**Round-183 lands the encoder-side mirror of the §7.3.11.7 non-merge
inter MVP-side syntax** — companion to round-177's
affine-syntax encoder. The new `non_merge_mvp_syntax_enc` module
lifts the four `#[cfg(test)]`-only helpers in `leaf_cu.rs`
(`encode_inter_pred_idc`, `encode_sym_mvd_flag`,
`encode_ref_idx_lx`, `encode_mvp_lx_flag`) into a public encoder
surface that mirrors the matching reader-side readers
(`LeafCuReader::read_inter_pred_idc`, `read_sym_mvd_flag`,
`read_ref_idx_lx`, `read_mvp_lx_flag`) bin-for-bin: Table 131's
two-bin form for `cbWidth + cbHeight > 12` with bin 0 ctxInc per
`ctx_inc_inter_pred_idc_bin0` and bin 1 per
`ctx_inc_inter_pred_idc_bin1`, the one-bin form when sum == 12
(PRED_BI suppressed), Table 86's deterministic-ctxInc-0
`sym_mvd_flag` (slot `init_type - 1`), Table 127's TR `cMax =
NumRefIdxActive[X] - 1` with bins 0/1 ctx-coded (ctxInc 0/1) and
bins 2.. bypass-coded against slot block `(init_type - 1) * 2`,
and Table 132's `mvp_lX_flag` single ctx-coded FL `cMax = 1` bin
against slot `init_type`. A dispatcher
`encode_non_merge_mvp_syntax(enc, ctxs, gate, decision)` walks
§7.3.11.7 in spec order
(`inter_pred_idc` → `sym_mvd_flag` → per-list `ref_idx_lX` →
per-list `mvp_lX_flag` for active lists). The `NonMergeMvpSyntaxGate`
exposes per-element gate predicates
(`inter_pred_idc_gate_open` / `inter_pred_idc_two_bin_form` /
`sym_mvd_signalled` / `l0_active` / `l1_active` /
`ref_idx_l0_signalled` / `ref_idx_l1_signalled`) so callers can
introspect the §7.4.12.7 inference branches without reproducing
the logic. A convenience constructor
`make_non_merge_mvp_syntax_decision(...)` clamps the L1 / sym-path
fields to 0 on inactive paths so the encoder cannot drift away
from the inferences the reader applies on decode. 10 new lib tests
pin the P-slice zero-bin path, the B-slice two-bin form
round-tripping all three PRED_L0 / PRED_L1 / PRED_BI directions,
the B-slice one-bin (sum-12) form round-trips, the
sym_mvd-gate-open path with `ref_idx_lX` correctly skipped, the
single-active-list `ref_idx_lX` zero-bin path, the truncated-unary
sweep over 0..=3 covering ctx + bypass + cMax-truncation bins, the
both-`mvp_lX_flag` values round-trip, a full B-slice bi-pred
round-trip exercising the entire §7.3.11.7 cascade end-to-end, the
decision-helper clamping behaviour across all three direction
modes, and the per-element gate-predicate truth table for both
P-slice and B-slice plus the (4,8) one-bin edge case. CTU-walker /
encoder-pipeline call-site that consumes this surface — alongside
the round-177 affine-syntax dispatcher — remains the next-step
follow-up for the broader non-merge inter CU encoder.

**Round-187 lands the encoder-side mirror of the §7.3.10.10
`mvd_coding()` structure** — the third and final encoder-syntax
module needed to drive a CU's full §7.3.11.7 non-merge inter
pre-residual syntax through public encoder helpers. The new
`mvd_coding_enc` module lifts the previously `#[cfg(test)]`-only
`encode_mvd_coding` / `encode_abs_mvd_minus2` helpers (used since
round 103 for conformance round-trips) into a public surface
parallel to `LeafCuReader::read_mvd_coding`. The dispatcher walks
§7.3.10.10 in spec order: both `abs_mvd_greater0_flag` bins
ctx-coded against Table 110 / Table 132 (slot `init_type`,
deterministic `ctxInc = 0`), per non-zero component an
`abs_mvd_greater1_flag` bin ctx-coded against Table 111 / Table
132, then per non-zero component (when greater1 is set) an
`abs_mvd_minus2` bypass payload via `encode_abs_mvd_minus2`
followed by a single bypass `mvd_sign_flag` bit. The §9.3.3.6
limited-EGk sub-binarisation transcribes `k = 1`,
`maxPreExtLen = 15`, `truncSuffixLen = 17` with the spec's
prefix-then-fixed-width-suffix shape and the cap-no-terminator
escape path. A `pub const fn max_mvd_magnitude() -> i32` returns
`2^17 - 1`, the §7.4.10.10 positive conformance bound; the
encoder's debug-assertion range check also admits `-2^17` (the
signed-18-bit floor), encoded through the EGk cap path. Together
with round-177's `affine_syntax_enc` and round-183's
`non_merge_mvp_syntax_enc` this completes the public encoder +
decoder symmetry for the entire §7.3.11.7 non-merge inter
pre-residual syntax. 11 new lib tests pin the spec-bound assertion,
the zero-pair two-bin round-trip on both non-I initTypes, the
unit-magnitude (`|lMvd| == 1`) sign-only round-trip exhaustive
over the 4-sign × 2-initType truth table, mixed zero / non-zero
component round-trips, large-magnitude sweep up to the §7.4.10.10
positive ceiling, signed-18-bit-floor `-131_072` round-trip
locking the asymmetric negative bound, eq. 190 derivation
spot-check, isolated EGk codec sweep across the maxPreExtLen
escape boundary with cross-check that the direct
`encode_abs_mvd_minus2` path produces a strictly-smaller wire than
its `mvd_coding()` envelope, both-components-negative round-trip
sweep, sign-drift-at-cap-magnitude round-trip across both axes
and initTypes, and a determinism pin verifying the zero-pair
encoder is repeatable byte-for-byte across runs. CTU-walker /
encoder-pipeline call-site that consumes the three encoder-syntax
modules together remains the next-step follow-up for the broader
non-merge inter CU encoder.

**Round-190 lands the encoder-side composite walker for the §7.3.11.7
non-merge inter CU pre-residual syntax** — the new
`non_merge_inter_pre_residual_enc` module composes round-177's
`affine_syntax_enc`, round-183's `non_merge_mvp_syntax_enc`, and
round-187's `mvd_coding_enc` into a single dispatcher
`encode_non_merge_inter_pre_residual` that walks §7.3.11.7 in spec
order: `inter_affine_flag` → `cu_affine_type_flag` →
`inter_pred_idc` → `sym_mvd_flag` → `ref_idx_l0` → `ref_idx_l1` →
`mvd_coding(L0)` → `mvd_coding(L1)` → `mvp_l0_flag` → `mvp_l1_flag`.
The interleaved `mvd_coding` between `ref_idx_lX` and `mvp_lX_flag`
matches the §7.3.11.7 listing (the round-183 dispatcher collapses
ref-idx then mvp and explicitly steps across `mvd_coding`, so this
composite walks the per-element encoder helpers directly rather than
re-using the MVP-side dispatcher). The new
`NonMergeInterPreResidualDecision` bundles the round-177 affine
decision, the round-183 MVP decision, and the per-list lMvd pair;
its `new(...)` constructor clamps inactive-list MVDs to zero (and L1
MVD under `sym_mvd_flag == 1`, where §8.5.2.5 derives
`MvdL1 = -MvdL0` and the wire carries nothing). The dispatcher
applies the same §7.4.12.7 inference debug-asserts as the underlying
round-177 / round-183 dispatchers so a release-mode caller can't
silently drift from the round-trip-symmetric reader path. Scope is
translational only (one `mvd_coding` per active list); multi-CP-MV
affine MVD emission (`numCpMv > 1` ⇒ one `mvd_coding` per control
point per list), `amvr_flag`, and `bcw_idx` remain follow-ups for
the broader non-merge inter CU encoder. 14 new lib tests pin the
end-to-end round-trip across the §7.3.11.7 pre-residual cascade:
P-slice zero-mvd / non-zero-mvd round-trips on both non-I
initTypes, B-slice PRED_L0 / PRED_L1 / PRED_BI round-trips, B-slice
SMVD round-trip exercising the inferred `-MvdL0` debug-assert path,
B-slice SMVD with the zero-clamp L1 variant, the outer affine gate
closed (zero affine bins), the (cbW + cbH) == 12 single-bin
inter_pred_idc form (4x8 CU), the `num_ref_idx_active == 1` ref_idx
suppression path, the §7.4.10.10 positive spec-conformance bound on
|lMvd| (`2^17 - 1`), and the constructor's three clamp paths
(inactive L0 / inactive L1 / sym_mvd L1). CTU-walker /
encoder-pipeline call-site that consumes this dispatcher + the
affine multi-CP / `amvr_flag` / `bcw_idx` extensions remain the
next-step follow-ups.

**Round-193 lands the §7.3.10.10 `amvr_flag` / `amvr_precision_idx`
CABAC reader** (closing the round-40 wiring gap — the round-40
`AmvrShift` / `apply_amvr_shift` Table 16 + eqs. 161-176 work landed
the value-side of AMVR but the syntax-side bin readers were never
written). The new `LeafCuReader::read_amvr_flag` and
`LeafCuReader::read_amvr_precision_idx` mirror the spec's Table 132
binarisations exactly: `amvr_flag` is FL `cMax = 1` ctx-coded at
Table 89 slot `(init_type - 1) * 2 + (inter_affine_flag ? 1 : 0)`;
`amvr_precision_idx` is TR with `cMax = (inter_affine_flag == 0 &&
CuPredMode != MODE_IBC) ? 2 : 1` and `cRiceParam = 0`, with both
bins ctx-coded (bin 0 against Table 90 slot `init_type * 3 +
((MODE_IBC) ? 1 : (inter_affine_flag == 0 ? 0 : 2))`, bin 1 against
`init_type * 3 + 1` — the deterministic Table 132 entry). The
`AmvrGate` struct bundles the §7.3.10.10 outer-gate inputs:
`sps_amvr_enabled_flag` + `sps_affine_amvr_enabled_flag` from the
active SPS, the already-decoded `inter_affine_flag`, and the per-arm
MVD-non-zero reductions over `MvdL0` / `MvdL1` for the regular arm
and `MvdCpL0` / `MvdCpL1` over the three control points for the
affine arm. The cascade dispatcher `read_amvr_inter_gated` walks
`if(gate-open) { amvr_flag = read; if(amvr_flag) precision = read }`
and returns `(amvr_flag, amvr_precision_idx, AmvrShift)` with the
round-40 §7.4.11.6 / Table 16 shift already folded through
`AmvrShift::for_inter` / `AmvrShift::for_affine` per
`inter_affine_flag`. Two new `ContextModel` bundles thread the
matching Table 89 / Table 90 init tables into `LeafCuCtxs`
(`amvr_flag`: 4 ctxIdx as 2 ctx slots × 2 non-I initTypes;
`amvr_precision_idx`: 9 ctxIdx as 3 ctx slots × 3 initTypes
including I for the IBC AMVR path that round 193 deliberately
exposes the per-bin reader for, even though the §7.3.10.10
dispatcher itself is non-IBC only — the IBC branch shares
`read_amvr_precision_idx` directly because §7.3.10.5 emits no
`amvr_flag` there, the §7.4.12.7 inference assigns 1). The bin-0
ctxInc helper for `amvr_precision_idx` corrected to match Table
132's verbatim `(MODE_IBC) ? 1 : (inter_affine_flag == 0 ? 0 : 2)`
mapping (regular → 0, IBC → 1, affine → 2); the prior round-40
mapping had IBC and affine swapped and was unreachable until this
round wired up the reader. New `ctx_inc_amvr_precision_idx_bin1`
and `amvr_precision_idx_c_max` helpers surface the per-bin and
per-arm derivations for spec traceability. 16 new lib tests pin the
per-bin round-trip across both AMVR arms (regular / affine) × both
non-I initTypes × every legal precision value × all three initTypes
for the IBC path, plus the cascade dispatcher's four control flows:
closed gate ⇒ inferred-default fallthrough (no bins consumed),
open + `amvr_flag = 1` regular round-trip with `precision_idx = 2`
folding to `AmvrShift = 6` (4-luma), open + affine `amvr_flag = 1`
round-trip exhaustive over `precision_idx ∈ {0, 1}` folding to
`AmvrShift ∈ {0, 4}` (1/16-luma vs 1-luma), open + `amvr_flag = 0`
⇒ precision-idx-not-consumed (the reader debits exactly one bin
from the bitstream). Two new gate tests confirm the regular and
affine arms don't cross — an affine CU with non-zero regular MVDs
but all-zero CP-MVDs keeps the gate closed, and a regular CU
shielded behind `inter_affine_flag = 1` also stays closed.
Multi-CP-MV affine MVD emission (`numCpMv > 1`), the encoder-side
`amvr_flag` / `amvr_precision_idx` mirror, and the encoder-side
`bcw_idx` mirror remain follow-ups for the broader non-merge inter
CU encoder.

**Round-195 lands the encoder-side §7.3.10.10 `amvr_flag` /
`amvr_precision_idx` mirror** and wires it into the round-190
non-merge inter pre-residual composite walker. The new public
`amvr_enc` module lifts the round-193 `#[cfg(test)]` helpers in
`leaf_cu.rs` into a real encoder surface: `encode_amvr_flag` emits
the FL `cMax = 1` ctx bin at Table 89 slot `(init_type - 1) * 2 +
ctx_inc_amvr_flag(inter_affine_flag, false)`;
`encode_amvr_precision_idx` emits the TR `cMax = (non-affine &&
!mode_ibc) ? 2 : 1` cascade (bin 0 ctx-coded at `init_type * 3 +
ctx_inc_amvr_precision_idx(...)`, bin 1 ctx-coded at `init_type * 3
+ 1`); and the dispatcher `encode_amvr_inter_gated` walks the same
§7.3.10.10 cascade the round-193 reader walks, applying the
§7.4.12.7 inferences (gate closed ⇒ no bins on the wire). A new
`AmvrDecision` carries `(amvr_flag, amvr_precision_idx, AmvrShift)`
with the §7.4.11.6 / Table 16 shift pre-folded through
`AmvrShift::for_inter` or `AmvrShift::for_affine` per the gate's
`inter_affine_flag` — matching the round-193 reader's return triple
exactly. The composite walker
`encode_non_merge_inter_pre_residual_with_amvr` chains the round-190
steps 1–9 with the new round-195 step 10 in §7.3.11.7 spec order
(after `mvp_lX_flag`, before the deferred `bcw_idx` / residual
tree). 9 new lib tests pin the dispatcher: exhaustive `(amvr_flag,
amvr_precision_idx)` round-trip across both arms (regular cMax = 2
⇒ {0, (1, 0), (1, 1), (1, 2)}, affine cMax = 1 ⇒ {0, (1, 0), (1,
1)}) × both non-I initTypes, gate-closed inferred-default
round-trip, regular-vs-affine non-crossing gate tests, and 2 new
composite-walker tests pin the spec ordering (P-slice closed AMVR
gate emits zero AMVR bins + recovers `AmvrShift = 2`; P-slice open
regular AMVR + `prec = 2` recovers `AmvrShift = 6` / 4-luma).
Multi-CP-MV affine MVD emission (`numCpMv > 1`) and the
encoder-side `bcw_idx` mirror remain the final follow-ups for the
full §7.3.11.7 non-merge inter pre-residual CU encoder.

**Round-201 lands the encoder-side §7.3.10.5 `bcw_idx[x0][y0]`
mirror** and wires it into the round-195 non-merge inter pre-residual
composite walker. The new public `bcw_idx_enc` module lifts the
round-126 `#[cfg(test)]` helper in `leaf_cu.rs` into a real encoder
surface: `encode_bcw_idx` emits the TR `cMax = NoBackwardPredFlag ?
4 : 2`, `cRiceParam = 0` cascade (bin 0 ctx-coded against Table 91
slot `init_type - 1` with `ctxInc = 0` per Table 132; bins 1..cMax
bypass-coded). The dispatcher `encode_bcw_idx_gated` walks the same
§7.3.10.5 conditional the round-126 reader walks, applying the
§7.4.12.5 inference (gate closed ⇒ no bin on the wire, reader
recovers `bcw_idx = 0`). The composite walker
`encode_non_merge_inter_pre_residual_with_amvr_and_bcw` chains the
round-195 steps 1–10 with the new round-201 step 11 in §7.3.11.7 /
§7.3.10.5 spec order (after the AMVR cascade, before the deferred
residual / `cu_coded_flag` / `transform_tree()` / `cu_qp_delta`
tail). The new `bcw_idx_c_max` helper surfaces the cMax derivation
(`NoBackwardPredFlag ? 4 : 2`) for spec traceability. 12 new lib
tests pin the dispatcher: `bcw_idx_c_max` Table 132 branches, raw
exhaustive round-trip across both cMax arms (`cMax = 2` ⇒ {0, 1, 2},
`cMax = 4` ⇒ {0, 1, 2, 3, 4} including the value-4 truncation point)
× both non-I initTypes, value-zero single-ctx-bin sanity, and gated
closed-gate inference paths covering every §7.3.10.5 close condition
(default-closed, `sps_bcw_enabled = false`, uni-pred / `None`
`inter_pred_idc`, each of the four luma/chroma weighted-pred flags,
small CU `cb_w * cb_h < 256`). 7 new composite-walker tests pin the
spec ordering: P-slice BCW-closed round-trip, B-slice PRED_BI
exhaustive round-trips across both cMax arms × both non-I initTypes,
B-slice PRED_BI weighted-pred-closed round-trip, B-slice PRED_BI
small-CU-closed round-trip, B-slice uni-pred BCW-closed round-trip,
and an AMVR-open + BCW-open simultaneous round-trip exercising both
gates emitting in sequence. Multi-CP-MV affine MVD emission
(`numCpMv > 1`) is delivered in the following round.

**Round-207 lands the encoder-side §7.3.10.5 multi-CP-MV affine MVD
dispatcher** — the final follow-up for the §7.3.11.7 non-merge inter
pre-residual CU encoder identified by rounds 190 / 195 / 201. The new
public surface
[`encode_non_merge_inter_pre_residual_affine`](src/non_merge_inter_pre_residual_enc.rs)
generalises the round-190 translational dispatcher to emit `numCpMv`
`mvd_coding()` invocations per active list in §7.3.10.5 spec order:
`mvd_coding(L0, cpIdx = 0)` ⇒ `if MotionModelIdc > 0:
mvd_coding(L0, 1)` ⇒ `if MotionModelIdc > 1: mvd_coding(L0, 2)` ⇒
`mvp_l0_flag`, with the symmetric L1 cascade following (where the
§8.5.2.5 SMVD shortcut applies only to `cpIdx == 0`; §7.3.11.7
excludes SMVD on the affine path, debug-asserted in the dispatcher).
The new `NonMergeInterPreResidualAffineDecision` data type carries per-
CP `lMvdCpLX[cpIdx][c]` arrays whose `cpIdx >= numCpMv` slots are
clamped to zero by the constructor (so stale state can't leak past
the §7.3.10.5 per-CP gates); a `num_cp_mv()` accessor surfaces the
§8.5.5.5 `numCpMv = MotionModelIdc + 1` derivation. 10 new round-trip
lib tests pin the dispatcher: translational degenerate-case
bit-identity with the round-190 dispatcher under `numCpMv == 1`
(verified on both P-slice PRED_L0 and B-slice PRED_BI by comparing
encoded bytes from both dispatchers), Affine4Param P-slice L0-only
two-CP round-trip with the third CP slot suppressed, Affine6Param
P-slice L0-only three-CP round-trip, Affine4Param B-slice PRED_BI
bi-pred per-CP round-trip across both lists, Affine6Param B-slice
PRED_BI bi-pred three-CP round-trip across both lists, constructor
clamps on inactive L1 per-CP MVDs, constructor clamps on unused CP
slots (`cpIdx == 2` under Affine4Param), and Affine4Param P-slice
all-zero per-CP MVDs (covering the zero-greater0-flag cascade
through the per-CP loop). The composite dispatchers
`encode_non_merge_inter_pre_residual_with_amvr` and
`encode_non_merge_inter_pre_residual_with_amvr_and_bcw` stay on the
translational scope of round-195 / round-201; folding them with the
new affine cascade is a small follow-up (AMVR and BCW themselves
already accept the per-CP MVD non-zero state via
`amvr_gate.any_mvd_cp_l0_l1_nonzero` and the §7.4.12.5 BCW gates,
respectively).

**Round-213 lands that follow-up** — two new public composite
dispatchers,
[`encode_non_merge_inter_pre_residual_affine_with_amvr`](src/non_merge_inter_pre_residual_enc.rs)
and
[`encode_non_merge_inter_pre_residual_affine_with_amvr_and_bcw`](src/non_merge_inter_pre_residual_enc.rs),
which compose the round-207 per-CP affine cascade with the round-195
§7.3.10.10 AMVR step and the round-201 §7.3.10.5 BCW step in §7.3.11.7
spec order (steps 1–9 affine cascade ⇒ step 10 AMVR ⇒ step 11 BCW).
The AMVR cascade's affine arm is now reachable from the encoder side:
`amvr_gate.any_mvd_cp_l0_l1_nonzero` is computed by the caller over
the active per-CP MVDs (`{0, 1}` for `Affine4Param`, `{0, 1, 2}` for
`Affine6Param`; CPs beyond `numCpMv` are already clamped to zero by
[`NonMergeInterPreResidualAffineDecision::new`]). The BCW cascade
remains gated on the §7.3.10.5 conditions verbatim — `inter_pred_idc
== PRED_BI`, `cb_w * cb_h >= 256`, no luma/chroma weighted-pred flags
— and the affine path's translational degenerate (`numCpMv == 1`)
collapses to the round-201 wire layout exactly. 8 new round-trip lib
tests pin the two dispatchers: (1) translational-degenerate
bit-identity with the round-195 affine+AMVR sibling (P-slice
PRED_L0, regular AMVR open with prec = 2 → AmvrShift = 6, byte-exact
diff over both non-I initTypes); (2) Affine4Param closed-gate
inferred-default AmvrShift = 2 round-trip; (3) Affine4Param
open-gate exhaustive round-trip across the affine arm's `cMax = 1`
precision space (prec ∈ {0, 1} → AmvrShift ∈ {0, 4}); (4)
Affine6Param B-slice PRED_BI three-CP-per-list round-trip with
prec = 1 → AmvrShift = 4; (5) translational-degenerate bit-identity
with the round-201 affine+AMVR+BCW sibling exhaustive over bcw_idx ∈
{0, 1, 2} × both non-I initTypes; (6) Affine4Param B-slice PRED_BI
open-everything round-trip exhaustive over bcw_idx values × both
initTypes; (7) Affine6Param BCW-closed-by-chroma_weight_l0 round-trip
on the affine path; (8) P-slice BCW-closed-by-PRED_L0 round-trip with
the affine AMVR cascade firing. The composite dispatchers complete
the §7.3.11.7 non-merge inter pre-residual CU encoder up to the
residual tree on the full translational + affine grid; the residual
tree / `cu_coded_flag` / `transform_tree()` / `cu_qp_delta` tail
remains the next encoder-side milestone.

**Round-219 lands the reader-side composite walker for the §7.3.11.7
non-merge inter pre-residual cascade** — the long-standing
asymmetry between the round-190 encoder-side dispatcher
(`encode_non_merge_inter_pre_residual`) and its absent reader twin.
The new `non_merge_inter_pre_residual_dec` module exposes
`read_non_merge_inter_pre_residual(reader, &affine_gate, &mvp_gate)`
which walks the per-element reader helpers
(`read_non_merge_inter_affine` → `read_inter_pred_idc` →
`read_sym_mvd_flag` → per-list `read_ref_idx_lx` → per-list
`read_mvd_coding` → per-list `read_mvp_lx_flag`) in §7.3.11.7 spec
order — exactly the sequence the encoder-side dispatcher emits — and
folds the §7.4.12.7 inferences for every gate-closed branch into the
returned `NonMergeInterPreResidualDecision`. P-slice ⇒
`inter_pred_idc = PRED_L0` with zero bins consumed; the §7.3.11.7
SMVD gate fills `mvd_l1 = -mvd_l0` per §8.5.2.5; inactive lists
yield `MotionVector { x: 0, y: 0 }` and zero `ref_idx_lX` /
`mvp_lX_flag` per §7.4.12.7. Scope matches round 190 verbatim:
translational only (one `mvd_coding` per active list,
`numCpMv == 1`); multi-CP-MV affine MVD parsing (round 207 encoder
follow-up), `amvr_flag` / `amvr_precision_idx` (round 195 encoder
follow-up — reader-side `read_amvr_inter_gated` already exists from
round 193), and `bcw_idx` (round 201 encoder follow-up — reader-side
`read_bcw_idx_gated` already exists from round 126) sit after the
pre-residual cascade and are addressable by composing this
dispatcher with those existing per-element reader helpers in spec
order. 13 new lib tests drive every reachable §7.4.12.7-inference
path end-to-end through the encoder dispatcher and back: P-slice
zero-mvd / nonzero-mvd round-trips on both non-I initTypes; B-slice
`PRED_L0` / `PRED_L1` / `PRED_BI` round-trips; the SMVD path with
`mvd_l1 = -mvd_l0` recovery; the outer affine gate closed by SPS
and by `cbWidth < 16` block-size both yielding `Translational`; the
4-param affine gate open but decision `Translational` (one
`inter_affine_flag = 0` bin); `num_ref_idx_active_l1 = 1`
`ref_idx_l1` suppression; the `(cbWidth + cbHeight) == 12` one-bin
`inter_pred_idc` form; `num_ref_idx_active_l0 = 1` `ref_idx_l0`
suppression in P-slice; and the §7.4.10.10 max-magnitude
`±(2^17 − 1)` per-component MVD round-trip on both lists exercising
the §9.3.3.6 limited-EGk cap-escape path. With the round-190 / 195 /
201 / 207 / 213 encoder dispatcher family and now the round-219
reader twin, an end-to-end CU walk (encoder ↔ decoder) over the full
§7.3.11.7 pre-residual cascade collapses to a single function call
on each side. The matching reader-side `_with_amvr` and
`_with_amvr_and_bcw` composites for symmetry with the encoder
family remain the next follow-up.

**Round-224 lands those reader-side composites.** Two new public
entry points in the `non_merge_inter_pre_residual_dec` module —
[`read_non_merge_inter_pre_residual_with_amvr`](src/non_merge_inter_pre_residual_dec.rs)
and
[`read_non_merge_inter_pre_residual_with_amvr_and_bcw`](src/non_merge_inter_pre_residual_dec.rs)
— extend the round-219 reader twin from steps 1–9 to steps 1–10 and
1–11 respectively in §7.3.11.7 spec order, fully mirroring the
round-195 `encode_non_merge_inter_pre_residual_with_amvr` and the
round-201 `encode_non_merge_inter_pre_residual_with_amvr_and_bcw`
encoder dispatchers. The `_with_amvr` variant chains the round-219
pre-residual cascade with the round-193 reader-side
`read_amvr_inter_gated` (§7.3.10.10 outer-gate walk + per-cascade-arm
`amvr_flag` / `amvr_precision_idx` reads + Table 16 `AmvrShift`
fold), returning the pair `(NonMergeInterPreResidualDecision,
AmvrDecision)` where `AmvrDecision` mirrors the round-195
encoder-side input shape (the `(amvr_flag, amvr_precision_idx,
shift)` triple already produced by the reader, repackaged so
encoder→decoder symmetry holds at the type level too). The
`_with_amvr_and_bcw` variant chains that with the round-126
reader-side `read_bcw_idx_gated` (§7.3.10.5 conditional + TR `cMax =
NoBackwardPredFlag ? 4 : 2` walk + §7.4.12.5 inference for the
gate-closed default), returning the triple `(decision, amvr,
bcw_idx)`. Both composites debug-assert the same caller-conformance
contracts the encoder-side dispatchers enforce
(`amvr_gate.inter_affine_flag` must agree with the round-219
dispatcher's decoded affine flag; `bcw_gate.cb_width /
bcw_gate.cb_height` must match the affine gate's; and
`bcw_gate.inter_pred_idc` must match the MVP-side resolved
`inter_pred_idc` per §7.4.12.7 / §7.3.10.5). 11 new lib tests pin
the dispatchers end-to-end against the encoder-side dispatchers:
P-slice AMVR-closed-gate round-trip (no AMVR bins, AmvrShift = 2
inferred), P-slice AMVR-open regular `prec = 2` round-trip
(AmvrShift = 6 / 4-luma), B-slice PRED_BI AMVR-open `prec = 1`
round-trip (AmvrShift = 4 / 1-luma), SMVD path AMVR-closed
round-trip (sym_mvd_flag = 1, §8.5.2.5 `mvd_l1 = -mvd_l0` inferred
and the AMVR cascade emits amvr_flag = 0 on a non-zero MVD), P-slice
BCW-closed-by-PRED_L0 round-trip, B-slice PRED_BI BCW-open
exhaustive over `cMax = 2` × both non-I initTypes, B-slice PRED_BI
BCW-open with `NoBackwardPredFlag = 1` exhaustive over `cMax = 4`
(bcw_idx ∈ {0, 1, 2, 3, 4}), simultaneous AMVR-open + BCW-open
round-trip (both gates emit bins in §7.3.11.7 spec order),
BCW-closed-by-`luma_weight_l0_flag` weighted-pred path, and
BCW-closed-by-small-CU (8 × 8 < 256) with AMVR open. With this round
the reader and encoder dispatcher families are complete mirrors of
each other across the full §7.3.11.7 non-merge inter pre-residual
cascade (steps 1–11 — pre-residual + AMVR + BCW); the residual tree
/ `cu_coded_flag` / `transform_tree()` / `cu_qp_delta` tail remains
the next milestone (the encoder side still needs to ship before the
reader-side composite can land).

**Round-230 mirrors the round-207 / round-213 affine encoder
dispatchers on the reader side.** Three new public entry points in
the `non_merge_inter_pre_residual_dec` module —
[`read_non_merge_inter_pre_residual_affine`](src/non_merge_inter_pre_residual_dec.rs),
[`read_non_merge_inter_pre_residual_affine_with_amvr`](src/non_merge_inter_pre_residual_dec.rs),
and
[`read_non_merge_inter_pre_residual_affine_with_amvr_and_bcw`](src/non_merge_inter_pre_residual_dec.rs)
— generalise the round 219 / 224 reader twins from the translational
case (one `mvd_coding` per active list, `numCpMv == 1`) to the full
§7.3.10.5 non-merge inter affine case (`numCpMv` `mvd_coding`
invocations per active list, in `cpIdx = 0, 1, 2` order). The
dispatchers walk §7.3.11.7 + §7.3.10.5 spec order, fully mirroring
the round-207 `encode_non_merge_inter_pre_residual_affine` /
round-213 `encode_non_merge_inter_pre_residual_affine_with_amvr` /
round-213 `encode_non_merge_inter_pre_residual_affine_with_amvr_and_bcw`
encoder dispatchers. Step 1's affine syntax decode (the round-164
composite reader) drives the §8.5.5.2 eq. 160 `MotionModelIdc`
derivation; `numCpMv = MotionModelIdc + 1` per §8.5.5.5 then gates
the per-CP cascade reads in step 6 (L0) and step 7 (L1). The
§8.5.2.5 `MvdL1[0] = -MvdL0[0]` derivation is folded for the
translational degenerate of the SMVD path (the affine path itself
excludes SMVD per §7.3.11.7). The reader returns
[`NonMergeInterPreResidualAffineDecision`](src/non_merge_inter_pre_residual_enc.rs)
where the per-CP MVD arrays carry the decoded values in slots
`[0..numCpMv]` and zero in slots `[numCpMv..3]`. The `_with_amvr`
variant additionally returns the §7.3.10.10 `AmvrDecision` triple
(`amvr_flag`, `amvr_precision_idx`, Table 16 `AmvrShift`); the
`_with_amvr_and_bcw` variant returns the §7.3.10.5 `bcw_idx` value
on top. 13 new lib tests pin the dispatchers end-to-end against the
round 207 / 213 encoder dispatchers: translational degenerate
(round-trip wire-identical to the round 219 path), 4-param affine
(`numCpMv = 2`) on P-slice L0-only with both non-I initTypes,
6-param affine (`numCpMv = 3`) on P-slice L0-only, 4-param affine on
B-slice PRED_BI exhaustively across L0 + L1 CPs, 6-param affine on
B-slice PRED_L1, translational SMVD path with the `-MvdL0[0]`
inference for the L1 cpIdx-0 slot, the four `_with_amvr` variants
(translational degenerate, 4-param affine-arm open, 4-param
affine-arm AMVR-closed-via-zero-MVDs, 6-param affine-arm B-slice
PRED_BI), and the three `_with_amvr_and_bcw` variants
(translational degenerate, 4-param affine + BCW exhaustive across
`cMax = 2` with AMVR open, 6-param affine with both AMVR and BCW
gates closed). With this round the reader and encoder dispatcher
families are complete mirrors of each other across the full
§7.3.10.5 non-merge inter affine path (steps 1–11 — pre-residual +
AMVR + BCW with per-CP MVD support).

**Round-233 lands a decomposed `mvd_coding()` body parser on top of
the round-187 packed walker.** A new public
[`MvdCodingDecision`](src/mvd_coding_enc.rs) struct exposes the
eight raw §7.3.10.10 syntax elements — `abs_mvd_greater0_flag` × 2,
`abs_mvd_greater1_flag` × 2, `abs_mvd_minus2` × 2, `mvd_sign_flag`
× 2 — instead of the post-eq.-190 fold into a `MotionVector`. The
[`from_motion_vector`](src/mvd_coding_enc.rs) constructor walks the
encoder-side decomposition (`greater0 = lMvd != 0`, `greater1 =
|lMvd| > 1`, `abs_mvd_minus2 = |lMvd| - 2` when meaningful, `sign =
lMvd < 0`); [`to_motion_vector`](src/mvd_coding_enc.rs) re-folds via
eq. 190 honouring the §7.4.10.10 inferred slots (greater1 == 0
collapses the magnitude to 1; greater0 == 0 collapses everything to
zero). Two new public walkers consume / produce the struct directly:
[`encode_mvd_coding_decomposed`](src/mvd_coding_enc.rs) is the
encoder mirror of the new
[`LeafCuReader::read_mvd_coding_decomposed`](src/leaf_cu.rs)
reader; both emit / consume the exact same bin sequence the
round-187 packed `encode_mvd_coding`-on-`MotionVector` walker uses,
so the per-bin layout is now inspectable by external consumers
without re-implementing the §7.3.10.10 bin order. 12 new lib tests
pin the body parser: zero-pair / unit / mixed-zero / large-magnitude
round trips up to the §9.3.3.6 EGk cap at `|lMvd| = 2^17 - 1` for
both non-I initTypes; a `parity_decomposed_and_packed_emit_identical_bitstreams`
test asserts the two encoders produce byte-identical wires across a
12-pair grid plus the EGk-boundary points; cross-path tests pin the
encoder ↔ reader equivalences (packed encoder wire decodes through
the decomposed reader, and decomposed encoder wire decodes through
the packed reader, both back to the original `(x, y)` pair). The
decomposed walker is the natural entry point for a trace-replay
harness or a bin-level rate-distortion scan that holds an explicit
per-bin candidate set rather than a packed `MotionVector`.

## Usage

Registering the codec wires the parser into `oxideav`'s codec
registry:

```rust
use oxideav_codec::CodecRegistry;
let mut codecs = CodecRegistry::new();
oxideav_h266::register(&mut codecs);
```

Parsing parameter sets directly without going through the registry:

```rust
use oxideav_h266::nal::{iter_annex_b, extract_rbsp, NalUnitType};
use oxideav_h266::sps::parse_sps;

let bytes: &[u8] = /* Annex B stream */;
for nal in iter_annex_b(bytes) {
    if nal.header.nal_unit_type == NalUnitType::SpsNut {
        let rbsp = extract_rbsp(nal.payload());
        let sps = parse_sps(&rbsp)?;
        println!(
            "{}x{} {}-bit",
            sps.cropped_width(),
            sps.cropped_height(),
            sps.bit_depth_y(),
        );
    }
}
# Ok::<(), oxideav_core::Error>(())
```

## Spec reference

Parser code references ITU-T H.266 | ISO/IEC 23090-3 (2026-01 edition)
section numbers in the comments. No third-party decoder sources were
consulted — the implementation is spec-only.

## License

MIT — see [LICENSE](LICENSE).
