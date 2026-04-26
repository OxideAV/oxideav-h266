# Changelog

All notable changes to this crate are recorded here.

## [Unreleased]

### Added

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
