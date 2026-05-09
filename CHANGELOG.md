# Changelog

All notable changes to this crate are recorded here.

## [Unreleased]

### Added

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
