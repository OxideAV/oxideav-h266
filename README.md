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
  mvd_coding (non-merge inter)**: still surface
  `Error::Unsupported`.
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
