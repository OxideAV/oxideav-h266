# Changelog

All notable changes to this crate are recorded here.

## [Unreleased]

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
