# oxideav-h266

[![CI](https://github.com/OxideAV/oxideav-h266/actions/workflows/ci.yml/badge.svg)](https://github.com/OxideAV/oxideav-h266/actions/workflows/ci.yml) [![crates.io](https://img.shields.io/crates/v/oxideav-h266.svg)](https://crates.io/crates/oxideav-h266) [![docs.rs](https://docs.rs/oxideav-h266/badge.svg)](https://docs.rs/oxideav-h266) [![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Pure-Rust **H.266 / VVC** (ITU-T H.266 | ISO/IEC 23090-3) codec for
oxideav. Zero C dependencies, no FFI, no `*-sys` crates.

VVC is a very large specification (500+ pages); this crate is an
in-progress, spec-driven implementation. It provides a complete NAL
framing + parameter-set parsing layer, a reconstruction pipeline that
covers the intra tools and — as of r447 — the full inter tool chain
(affine merge + non-merge, SbTMVP, GPM, MMVD, CIIP, SMVD, AMVR incl.
the affine arm, BCW, BDOF, PROF, DMVR) validated tool-by-tool against
black-box fixture streams, and an IDR-frame encoder pipeline. A
complete encoder is still being built up incrementally.

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
  * **VPS** (§7.3.2.2), **SPS** (§7.3.2.3 — profile/tier/level, chroma
    format, picture size, CTU size, transform sizes, tools-enable
    flags), **PPS** (§7.3.2.4). SPS derivations now include the §7.4.3.4
    eq. 41 `NumExtraPhBits` / `NumExtraShBits` counts (from the
    `sps_extra_{ph,sh}_bit_present_flag[i]` runs), the §7.4.3.22 eq. 106
    `Log2TransformRange` (extended-precision `Max(15, Min(20, BitDepth +
    6))` when `sps_extended_precision_flag` is set, threaded through every
    coded-TB dequant + inverse-transform clip), and the §7.4.3.4 eqs.
    54 – 57 `ChromaQpTable[k]` mapping derivation (`ChromaQpTable::build`
    / `map_qp`). The PPS retains the §7.4.10.6 `cu_chroma_qp_offset_list`
    triples; the §7.3.7 slice-header tail consumes the range-extension
    `sh_ts_residual_coding_rice_idx_minus1` / `sh_reverse_last_sig_coeff_flag`.
  * **Tiles / slices layout** (r418) — the §6.5.1 tile-scan
    derivations live in `tile_scan::TileScan`: tile boundary + CTB → 
    tile maps (eqs. 16 – 19), `CtbAddrInSlice` for rectangular slice
    layouts (eqs. 20 – 22, including subpicture-driven and
    multi-slice-per-tile geometries — the PPS slice loop performs the
    eq. 21 uniform replication and the `i += NumSlicesInTile − 1`
    syntax advance), and the §7.4.8 eq. 141 `NumEntryPoints`
    derivation; the slice header parses WPP / tile entry-point
    offsets. **r429 — the slice-data walk is tile/WPP-aware**: the
    CTU walker resolves `CtbAddrInCurrSlice[]` and decodes in
    tile-scan order with per-tile §9.3.2.2 CABAC context
    re-initialization, byte-aligned subset engine re-inits at the
    `end_of_tile_one_bit` / `end_of_subset_one_bit` boundaries
    (§9.3.2.5), the WPP §9.3.2.3/§9.3.2.4 context storage /
    synchronization around each CTU row's first CTU, the §7.3.11.1
    row-in-tile HMVP / IBC resets, and the §6.4.4 "different tile" +
    WPP column-cap availability arms gating every spatial neighbour
    read (intra reference masks, parse-time mode grids, motion /
    affine field scans, split-flag ctxIncs, `qPY_PRED`, the SAO / ALF
    CTU-prefix conditions). **r431 — multi-slice pictures decode
    end-to-end**: `CtuWalker::continue_slice` walks each further
    slice of the picture on the same walker — fresh §9.3.2.2 CABAC
    initialisation at the new slice's SliceQpY, the §7.3.11.1
    first-CTU-in-slice resets (HMVP / IBC / predictor palette /
    `FirstCtbRowInSlice`), the §6.4.4 different-slice availability
    (rectangular slices as region-rect intersection, raster slices
    through the different-tile arm, a per-CTB slice map for the SAO /
    ALF prefix availability), and the
    `pps_loop_filter_across_slices_enabled_flag = 0` deblock / SAO /
    ALF gates (composable with the across-tiles gates). Subpicture
    layouts remain out of scope.
  * **APS** (§7.3.2.5) — ALF / LMCS / scaling-list type. The LMCS APS
    payload (§7.3.2.19) is decoded into a typed `lmcs::LmcsData`, with
    the BitDepth-dependent §7.4.3.19 derivations (`OrgCW`, `lmcsCW`,
    `LmcsPivot`, `ScaleCoeff` / `InvScaleCoeff`, `ChromaScaleCoeff`) and
    the sample-domain forward / inverse luma mapping + chroma residual
    scaling folds — all **live in the reconstruction pipeline** (see
    the LMCS bullet under Decode support).
* **Auxiliary NAL units** — AUD (§7.3.2.10), Filler Data (§7.3.2.13),
  End of Sequence / End of Bitstream (§7.3.2.11 / §7.3.2.12),
  `rbsp_trailing_bits()` / `byte_alignment()` / `rbsp_slice_trailing_bits()`
  validators, and the SEI message + `sei_rbsp()` walkers. The
  `sei_payload()` Annex D bodies specified directly in this Specification
  are decoded into typed structures — the picture timing SEI message
  (§D.4.1, `payloadType == 1`) into `picture_timing::PictureTiming` (the
  full §D.4.1 syntax — the per-sublayer CPB-removal-delay loop with the
  delta / explicit-`minus1` branch, `pt_dpb_output_delay`, the
  `bp_alt_cpb_params_present_flag`-gated NAL / VCL alt-CPB timing block,
  the `bp_du_*_in_pic_timing_sei_flag`-gated DU DPB / CPB blocks with the
  per-DU NAL-count + increment loops, the concatenation flag, and the
  trailing `pt_display_elemental_periods_minus1`; every `u(v)` length and
  branch is driven by a `picture_timing::PtContext` carried from the BP
  message via `BufferingPeriod::pt_context()`), the buffering period SEI
  message
  (§D.3.1, `payloadType == 0`) into `buffering_period::BufferingPeriod`
  (the full §D.3.1 HRD initialisation syntax: NAL / VCL / DU HRD presence,
  the three `bp_*_length_minus1` `u(v)` length fields, concatenation +
  CPB-removal-delay-delta lists, and the per-sublayer per-CPB initial
  CPB removal delay / offset + alt pairs, with the §D.3.2 `u(v)` lengths,
  range checks on `bp_cpb_cnt_minus1` / `bp_num_cpb_removal_delay_deltas_minus1`,
  and the absent-field inferences; `BufferingPeriod::dui_context()` surfaces
  the §D.3.2-derived `DuiContext` a companion DU information message needs),
  the SEI manifest SEI message
  (§D.8.1, `payloadType == 200`) into `sei_manifest::SeiManifest`, the SEI
  prefix indication SEI message (§D.9.1, `payloadType == 201`) into
  `sei_prefix_indication::SeiPrefixIndication` (with the per-indication
  `byte_alignment_bit_equal_to_one` padding verified per §D.9.2), and the
  subpicture level information SEI message (§D.7.1, `payloadType == 203`)
  into `subpic_level_info::SubpicLevelInfo` (the `sli_*` reference-level /
  fraction syntax across sublayers, with `sli_alignment_zero_bit`
  verified per §D.7.2), and the DU information SEI message (§D.5.1,
  `payloadType == 130`) into `decoding_unit_info::DecodingUnitInfo` (the
  `dui_*` CPB-removal-delay / DPB-output-delay syntax, driven by a
  `decoding_unit_info::DuiContext` carrying the §D.3.2 BP-derived gating
  flags and `u(v)` lengths plus the SEI NAL unit `TemporalId`, with the
  §D.2.1 `sei_payload()` trailing bits verified), and the constrained RASL
  encoding indication SEI message (§D.10.1, `payloadType == 207`) into
  `constrained_rasl_encoding_indication::ConstrainedRaslEncodingIndication`
  (an empty-body marker whose presence asserts the four §D.10.2 RASL
  encoding constraints, surfaced via `constraints()`; the §D.2.1
  `sei_payload()` framing is validated against `more_data_in_payload()`);
  the remaining payload bodies (most deferred to Rec. ITU-T H.274) are
  still uninterpreted.
* **Profile / Tier / Level** (§7.3.3.1) — `profile_tier_level()` walked
  end-to-end including the §7.3.3.2 `general_constraints_info()` body
  with every named GCI flag surfaced.
* **Picture / slice header** (§7.3.2.7-8) — walked.
* **Weighted prediction parameters** (§7.3.8) — `pred_weight_table()`
  with the §7.4.7 inferences, plus the §8.5.6.6.3 explicit weighted
  sample prediction process.

## Decode support

The reconstruction pipeline is built incrementally and currently covers
the **intra-only subset across tiles / WPP / multi-slice layouts**
plus a substantial P + B-slice merge subset:

* **Palette mode** (r431 — §7.3.11.6 / §8.4.5.3) — the full
  `palette_coding()` parse (predictor-run reuse loop, eq. 185
  `CurrentPaletteEntries` incl. the LocalDualTreeFlag arm, the
  §9.3.3.13 truncated-binary `palette_idx_idc` with the eq. 186 / 187
  `adjustedRefPaletteIndex` fold, §9.3.4.2.11 run_copy ctxIncs, EG5
  escape values with the SINGLE_TREE chroma sub-sampling gate, in-CU
  `cu_qp_delta` / `cu_chroma_qp_offset`), predictor palette
  maintenance in parse order (eq. 450, the eqs. 444 – 449
  local-dual-tree fold, the eq. 451 chType mirror) with the §9.3.2.1
  per-slice/per-tile resets and §9.3.2.6 / §9.3.2.7 WPP
  storage/synchronization, §8.4.5.3 reconstruction (eq. 438 lookups +
  eqs. 439 – 443 escape dequantisation with the `QpPrimeTsMin`
  floor), the §8.8.3.6.7/.6.8/.6.10 palette-side deblock write
  suppression, and the dual-tree luma / chroma arms.

* **Intra** (r412 — the full §8.4.5.2 pipeline) — PLANAR / DC and the
  complete general angular range (§8.4.5.2.13: Table 24 angles
  −14..=80 with the §8.4.5.2.7 wide-angle remap, eq. 337 invAngle
  side-projection, Table 25 fC/fG 4-tap luma interpolation with the
  Table 23 minDistVerHor filter selection, 2-tap chroma
  interpolation), PDPC (§8.4.5.2.15, all four refL/refT/wT/wL arms,
  luma + chroma), the §8.4.5.2.10 [1 2 1] reference filter, the
  spec-order §8.4.5.2.9 reference substitution, MRL reference lines
  (the parsed `intra_luma_ref_idx` selects the line at
  reconstruction), MIP (§8.4.5.2.2, all 30 weight matrices, eqs.
  263 – 266 reference rows), CCLM (§8.4.5.2.14), BDPCM, and ISP
  (§8.4.5.1, all 4 split types, eqs. 315/316 refW/refH and
  (nPbW)x(nH) prediction windows). Reference-sample availability runs
  the §6.4.4 derivation against per-sample `IsAvailable` masks marked
  in decoding order (§7.4.12.2 eq. 152 / §8.7.5.1 eq. 1212) — a
  reference position in a not-yet-decoded block substitutes instead
  of leaking stale plane bytes (the r409-observed decode-order bug).
* **§7.3.11.4 coding_tree conformance** (r412) — the single-tree
  walker decodes the coding tree in the spec's depth-first
  interleaved order (each leaf's `coding_unit()` bins follow its
  `split_cu_flag == 0` immediately), with the §6.4.1 – §6.4.3
  allowed-split derivations (SplitConstraints from the SPS partition
  constraints) gating both the split-bin presence (§7.4.12.4
  inferences when absent) and the §9.3.4.2.2 ctxIncs. The §7.3.11.2
  ALF CTU prefix (`alf_ctb_flag` family) is consumed in-stream via
  `set_alf_decode`. `finish_slice()` verifies
  `end_of_slice_one_bit == 1`.
* **§7.3.11.11 zero-out** (r412) — a 64-point DCT-II TB codes only
  the low-frequency 32-corner: the coded geometry (last-sig
  binarisation cMax, sub-block grid + scan, pass-1 bin budget) runs
  on `Log2ZoTb = Min(log2Tb, 5)` while the §9.3.4.2.4 last-sig prefix
  ctxInc keeps the full-TB dims, matching the §8.7.4.1 eqs. 1171/1172
  inverse-transform extents.
* **Dual-tree intra** (r391) — an I-slice with
  `sps_qtbtt_dual_tree_intra_flag` decodes through the §7.3.11.2
  `dual_tree_implicit_qt_split` (bin-less quad recursion while
  `cbSize > 64`, picture-bound quadrant gating); per surviving ≤64×64
  node the walker parses + reconstructs one DUAL_TREE_LUMA coding tree
  (luma-only CUs — chroma CBFs / residuals absent per §7.3.11.10) then
  one DUAL_TREE_CHROMA coding tree, with a separate chroma-tree
  neighbour map for the §9.3.4.2 split ctxIncs and the §6.4.1 – §6.4.3
  chroma leaf floor (an 8×8 luma-sample node is an unsplittable 4×4
  chroma CB). Chroma-tree CUs parse `intra_bdpcm_chroma_flag` / dir,
  `cclm_mode_flag` (Table 79 ctx) + `cclm_mode_idx` (Table 80, TR
  cMax = 2 → modes 81 – 83) or `intra_chroma_pred_mode`, the
  chroma-only `transform_unit()`, and the chroma-tree `lfnst_idx`
  (§7.3.11.5 `/ SubWidthC` size gates, chroma-half `lfnstNotTsFlag`);
  the §8.4.3 DM derivation samples a per-picture `IntraPredModeY`
  store at the collocated luma centre (MIP collocated → PLANAR).
  §8.4.4 `CclmEnabled` runs **all** arms — the simple ones plus the
  per-CB 64-grid derivation at CTB ≥ 64 (the walker logs
  `MttSplitMode`; the four 64-node enabling conditions and both
  luma-side suppressions are evaluated against the node's two trees).
  Reconstruction runs luma-only for dual-luma CUs (ISP included) and
  chroma-only for dual-chroma CUs — CCLM reads the node's
  already-reconstructed collocated luma, and §8.7.4.1 eq. 180 applies
  the inverse LFNST to both chroma TBs (eq. 1157 `predModeIntra` with
  CCLM resolved to the collocated luma mode). Chroma-tree
  deblocking-edge records are a documented approximation (luma-tree
  geometry).
* **IBC — intra block copy** (r406, §7.3.11.5 / §8.6) — single-tree
  I-slice IBC decodes end-to-end. Parse: the `cu_skip_flag` /
  `pred_mode_ibc_flag` prologue (Table 65 contexts, the §9.3.4.2.2
  eq. 1551 ctxInc against a parse-time per-4×4 `CuPredMode == MODE_IBC`
  neighbour grid committed in coding order), the §7.3.11.7 IBC merge
  arm (`merge_idx`, TR `cMax = MaxNumIbcMergeCand − 1`, eq. 62), and
  the non-merge arm (`mvd_coding`, `mvp_l0_flag`,
  `amvr_precision_idx` with the Table 16 IBC column — 1-pel / 4-pel).
  Reconstruction: the §8.6.2 block-vector derivation (spatial A1 / B1
  with the §6.4.4 `checkPredModeY` IBC-neighbour gate,
  `HmvpIbcCandList` with the §8.6.2.6 update + the §7.3.11.1
  per-CTU-row reset, zero pad, §8.5.2.14 AMVR rounding of the
  predictor + the eqs. 1092 – 1095 18-bit fold), the §8.6.3 prediction
  from the `IbcVirBuf` virtual buffer (§7.4.3.4 eqs. 45 – 47 geometry,
  §7.4.12.5 eqs. 181 / 182 per-CU invalidation, §8.7.5.1
  eqs. 1207 – 1209 fill after every reconstructed CU, and the
  §8.6.2.1 reference-region conformance constraints enforced) for
  luma + 4:2:0 chroma via the §8.6.2.5 `bvC`, the shared MODE_INTER
  residual tail (without the §8.7.5.2 LMCS forward mapping — an IBC
  prediction copies already-mapped-domain samples), and the
  eqs. 1111 – 1118 bookkeeping (`MvL0 = bvL`, `PredFlagLX = 0`,
  `RefIdxLX = −1`). **r409:** IBC also decodes on **P/B slices** (the
  inter-slice `cu_skip_flag` / `pred_mode_flag` / `pred_mode_ibc_flag`
  prologue with the §7.4.12.5 inferences) and on the **DUAL_TREE_LUMA**
  walk (luma-only §8.6.3 prediction; the sibling chroma tree paints the
  chroma; per-luma-CU `IbcVirBuf` maintenance + parse-grid commits on
  the dual-tree path).
* **`pred_mode_flag` on P/B slices** (r409, §7.3.11.5) — intra CUs
  inside inter slices decode to pixels: the bin is parsed under the
  spec presence condition (with the §9.3.4.2.2 eq. 1552 ctxInc off a
  parse-time `CuPredMode == MODE_INTRA` grid committed in coding
  order, Table 51 / Table 66 context split) and the §7.4.12.5
  inferences (4x4 → intra) apply; a MODE_INTRA CU runs the exact
  I-slice intra branch and reconstructs through the intra pipeline,
  its motion-field cells staying unavailable for later merge / AMVP
  scans. §8.4.2 MPM candidates thread the neighbours' parse-time
  `IntraPredModeY` / `IntraMipFlag` / `CuPredMode` (r409; previously
  every neighbour read PLANAR).
* **Non-merge inter (AMVP) parse in the full-CABAC walker** (r409,
  §7.3.11.5) — a `general_merge_flag == 0` CU parses the whole
  cascade through `decode_picture_into`: the affine pair (Table 133
  merge-side ctxIncs off the live grids), `inter_pred_idc`, the
  §8.3.5-resolved SMVD gate (`RefIdxSymLX` substituted for the
  per-list `ref_idx_lX` under `sym_mvd_flag`), the spec-ordered
  L0 / L1 blocks (per-CP `mvd_coding`, `mvp_lX_flag`, the
  `ph_mvd_l1_zero_flag` arm), the AMVR cascade, `bcw_idx` (§8.3.6
  `NoBackwardPredFlag` cMax), and the trailing `cu_coded_flag` +
  MODE_INTER `transform_unit()` — driving translational P/B AMVP CUs
  to pixels; an affine non-merge CU parses and surfaces a precise
  `Error::Unsupported` at reconstruction. r409 conformance fixes in
  the same area: the composite non-merge dispatchers emitted/read the
  per-list elements in a spec-divergent interleave (bi-pred B-slice
  desync — both sides now walk the L0 block fully first), the
  inter-slice `cu_skip_flag` ctxInc sampled a stale motion field
  instead of the parse-time `CuSkipFlag` grid, and the regular
  §8.5.2.16 HMVP list now resets per CTU row (`NumHmvpCand = 0`,
  §7.3.11.1) alongside the IBC resets.
* **Dependent quantization + sign data hiding** (r387) — the §7.4.12.11
  eq. 198 `QStateTransTable` trellis runs through every regular
  `residual_coding()` read (pass-1 `AbsLevelPass1 & 1` / pass-3
  `AbsLevel & 1` advances, the QState-dependent §9.3.4.2.8 eqs.
  1573 / 1574 sig-coeff ctxInc terms and §9.3.3.12 eq. 1536 `ZeroPos`,
  and the `TransCoeffLevel = (2·AbsLevel − (QState > 1))·(1 − 2·sign)`
  reconstruction re-walked from `startQStateSb`); sign data hiding
  suppresses the first-significant sign bin on `signHiddenFlag`
  sub-blocks and recovers it from the absolute-level-sum parity. The
  §8.7.3 dequant takes the eq. 1143 `bdShift + 1` / eq. 1151
  `(qP + 1)`-scaled levelScale arms from `sh_dep_quant_used_flag`.
* **Explicit scaling lists** (r387) — the §7.3.2.21
  `scaling_list_data()` APS payload is parsed and §7.4.3.20-derived
  into `ScalingMatrixRec` / `ScalingMatrixDcRec`
  (`scaling_list::parse_scaling_list_data`); `CtuWalker::
  set_scaling_list` binds the matrices and every dequant site derives
  the §8.7.3 eq. 1149 / 1150 `m[x][y]` (Table 38 id from
  `(predMode, cIdx, nTbW, nTbH)`, DC substitution for ids > 13, and
  the flat-16 arms for transform-skip / LFNST-disabled / ACT).
* **Inter merge** — P + B-slice skip / merge: §8.5.2.3 spatial-merge
  candidates, §8.5.2.11 / §8.5.2.12 temporal collocated candidate with
  POC scaling and buffer compression, §8.5.2.6 HMVP, §8.5.2.4
  pairwise-average candidate, and zero-MV padding; §8.5.2.7 MMVD;
  §8.5.6.7 CIIP; §8.5.4 / §8.5.7 GPM. Motion compensation uses the
  §8.5.6.3 8-tap luma + 4-tap chroma fractional-sample interpolation
  with default-weighted bi-pred (§8.5.6.6.2), BCW (§8.5.6.6.2 eq. 981),
  and BDOF (§8.5.6.5). **DMVR** (§8.5.1 / §8.5.3) is **live** to pixels
  on the plain bi-pred merge path: the §8.5.1 `dmvrFlag` gate
  (`sps_dmvr_enabled_flag` / `ph_dmvr_disabled_flag`, symmetric STRP POC
  bracket, `!mmvd`, `!ciip`, `bcwIdx == 0`, no weighted-pred,
  `cbW >= 8 && cbH >= 8 && cbW*cbH >= 128`) splits the CU into ≤16×16
  sub-blocks (eqs. 452 – 459), runs the §8.5.3.1 bilateral-matching
  search (with the `minSad >= sbW*sbH` early-out) per sub-block, and
  feeds the refined MVs to the §8.5.6 MC — with §8.5.6.5 BDOF running
  per DMVR sub-block on that sub-block's refined pair when the
  `bdofUsedFlag` derivation also holds (r391). Per the §8.5.1 NOTE the
  unrefined `MvLX` is what the per-picture motion field stores for
  spatial-MVP / deblocking, while every refined unit's `MvDmvrLX` is
  recorded and exported through
  `CtuWalker::motion_field_for_temporal()` so a later picture's
  §8.5.2.11 / §8.5.2.12 collocated derivation sees the refined MVs.
  High-bit-depth (Main10 / Main12) reconstruction
  runs through `u16` picture planes. A non-skip merge / CIIP CU always
  decodes its §7.3.11.10 `transform_unit()` residual — §7.3.11.5 puts
  no `cu_coded_flag` bin on a merge CU; the §7.4.12.5 inference reads
  0 for skip and 1 otherwise (r406 conformance fix: the reader
  previously consumed a spurious bin here) — (MODE_INTER luma-CBF
  condition + chroma CBFs) and adds the
  §8.7.3 dequant + §8.7.4 inverse-DCT-II residual to the MC prediction
  per §8.5.8 + §8.7.5.1 (`recSamples = Clip1(predSamples + resSamples)`).
  **SBT** (§7.4.12.5 / §8.7.4.1): when `cu_sbt_flag == 1` the CU's luma
  residual lives in a single sub-region TU (`sbt_geometry`) and inverse-
  transforms through the Table-40 DST-VII / DCT-VIII kernels; the other
  TU keeps the MC prediction. **Multi-TB tiling** (§7.3.11.4): a CU
  larger than `MaxTbSizeY` (64) splits its luma residual into ≤64×64
  transform blocks (`transform_tree_tiles`, spec recursion order) and
  reconstructs each at its offset. **Joint Cb-Cr** (§7.4.12.11 /
  §8.7.2): when `tu_joint_cbcr_residual_flag == 1` the inter CU's two
  chroma residuals share one coded transform block; the §7.4.12.11
  `TuCResMode` derivation selects the coded component (Cb for modes 1/2,
  Cr for mode 3) and the §8.7.2 `resSamples` derivation (eqs. 1130–1132,
  `cSign = 1 − 2·ph_joint_cbcr_sign_flag`) reconstructs the sibling
  component. **Transform-skip inter residual** (§7.3.11.5 parse gate →
  `residual_ts_coding` → §8.7.4.6 inverse-transform bypass) is **live**
  on the inter luma, Cb, Cr and joint-Cb-Cr paths. **CU-level chroma QP
  offsets** (§7.4.10.6 `CuQpOffsetCb/Cr/CbCr`, indexed from the PPS
  offset lists when `cu_chroma_qp_offset_flag == 1`) feed the §8.7.1
  inter chroma dequant QP, matching the intra path.
* **AMVP** — the §8.5.2.8-10 AMVP candidate derivation (spatial scan,
  HMVP fill, temporal Col, zero-MV pad), the §8.5.5.7 affine AMVP
  candidate list, and AMVR helpers (§7.4.11.6). The §8.5.2.1 non-merge
  inter CU reconstruction is **live**: `reconstruct_leaf_cu_inter_amvp`
  derives the per-list MVP from the candidate list, selects via
  `mvp_lX_flag`, folds the raw `MvdLX` with the per-CU `AmvrShift`
  (`mvLX = mvpLX + (mvdLX << AmvrShift)`), and feeds the shared §8.5.6 /
  §8.5.8 MC + residual tail — driving P/B non-merge CUs to output
  pixels. **Affine MC to pixels** (§8.5.5.3 / §8.5.6):
  `reconstruct_affine_inter_uni` takes a control-point MV set, derives
  the §8.5.5.9 per-4×4-sub-block luma MV grid (eqs. 872 – 875), runs the
  §8.5.6.3.2 affine 6-tap interpolation per sub-block with §8.5.5.8 PROF
  (`predict_luma_block_affine_prof`), and reconstructs 4:2:0 chroma by
  averaging the §8.5.5.3 top-left + bottom-right luma sub-block MVs
  (eqs. 876 – 879) into each chroma 4×4 sub-block before the §8.5.6.3.4
  4-tap chroma MC. `reconstruct_affine_inter_bi` predicts each list's
  affine MC into a CU-sized scratch then forms the §8.5.6.6.2 eq. 980
  default-weighted average over luma + chroma. **Affine-CPMV parse-to-
  pixels** (§8.5.5.5) is **live**: `reconstruct_leaf_cu_inter_affine_amvp`
  takes the parsed affine non-merge decision
  (`NonMergeInterPreResidualAffineDecision`, carrying the per-CP
  `MvdCpLX` arrays the §7.3.11.7 affine branch parses via
  `read_non_merge_inter_pre_residual_affine`), derives `numCpMv =
  MotionModelIdc + 1`, builds the §8.5.5.7 affine CPMVP candidate list
  per active list (`build_affine_mvp_cand_list` + `select_affine_mvp`
  eq. 840), cumulates the per-CP MVDs (§8.5.5.5 eqs. 660 – 663,
  `cumulate_affine_mvd_cp`), folds predictor + cumulative MVD into the
  final CPMVs (eqs. 664 – 667, `derive_final_affine_cpmvs`), and drives
  the uni / bi-pred affine MC — so affine inter CUs now decode from
  parsed CPMV deltas end-to-end. The parsed `bcw_idx` is threaded into
  the bi-pred branch (§8.5.6.6.2 eq. 981 weighted blend for `bcwIdx ∈
  1..=4`) and stored on the affine CB record so a later §8.5.5.2
  inherited-affine-merge scan recovers the weight. **Inherited affine
  CPMVP** (§8.5.5.7
  steps 4 / 5) is now **live**: a per-CB affine CPMV store
  (`inter::AffineCpmvField`, a 4×4-granularity grid of
  `inter::AffineCbRecord` mirroring `CbPosX/Y[0][·]`, `CbWidth/Height[0][·]`,
  `MotionModelIdc[·]`, `MvCpLX`, `PredFlagLX[·]`, `RefIdxLX[·]`) is
  broadcast by every affine AMVP CB across the blocks it covers and
  cleared (to `None`) by translational-inter and intra CBs. The
  §8.5.5.7 A-scan (eqs. 819 / 820 — A0, A1) and B-scan (eqs. 821 – 823
  — B0, B1, B2) sample those neighbour positions, gate them on the
  parallel-merge suppression (eq. 60) + the `DiffPicOrderCnt == 0`
  cross-list test, and feed the §8.5.5.5 inherited-CPMVP derivation
  (with the eqs. 736 – 739 `isCTUboundary` bottom-row path computed from
  `(yNb + nNbH) % CtbSizeY`), so a CU with an affine neighbour now picks
  the inherited predictor (eqs. 824 / 826 / 828 / 830 / 840) instead of
  the step-9 zero-MV pad. The §8.5.5.8 **constructed** CPMVP candidate
  (step 6) + the §8.5.5.7 step-7 single-corner fallback are now **live**:
  `derive_constructed_affine_corners` runs the three per-corner cascades
  (TL = {B2, B3, A2}, TR = {B1, B0}, BL = {A1, A0}) over the **regular
  per-sample motion field** (`MvLX[xNb][yNb]`, eqs. 841 – 846), so
  translational neighbours contribute, with the §6.4.4 availability +
  `PredFlagLX == 1 ∧ DiffPicOrderCnt == 0` gate and the cross-list
  fallback, AMVR-rounded. **Affine sub-block MERGE to pixels** (§8.5.5.2)
  is now **live**: a merge CU with `merge_subblock_flag == 1` routes
  through `reconstruct_leaf_cu_inter_subblock_merge`, which builds the
  §8.5.5.2 sub-block merge list (inherited A / B per §8.5.5.5 from the
  affine CPMV store, constructed Const1..6 per §8.5.5.6 from the regular
  motion field, zero-MV pad), picks the entry at `merge_subblock_idx`,
  and drives the uni / bi-pred affine MC for the picked candidate's
  per-list CPMVs. The §8.5.5.9 per-sub-block MV grid is broadcast into
  the motion field and the affine CB record is stored, so a later CU's
  §8.5.5.7 inherited scan can recover this merge block's CPMVs. The
  **SbTMVP (`SbCol`) sub-block-temporal merge to pixels** (§8.5.5.3) is
  now **live**: `build_sbtmvp_record` runs the §8.5.5.3 gate + §8.5.5.4
  A1 `tempMv` + the CU-centre collocated read (eqs. 729–731 → §8.5.2.12),
  and when `availableFlagSbCol` holds `reconstruct_leaf_cu_sbtmvp` fills
  the §8.5.5.3 per-8×8-sub-block motion grid and runs translational MC
  (uni- or eq. 980 bi-pred) per sub-block from `RefPicList[X][0]`. (BDOF
  is spec-prohibited on affine / sub-block-merge CUs — the §8.5.6.5
  `bdofFlag` derivation requires `MotionModelIdc == 0` and
  `merge_subblock_flag == 0` — so there is no affine-BDOF path to add.)
  **BCW on the affine bi-pred path**
  (§8.5.6.6.2 eq. 981) is now **live**: `reconstruct_affine_inter_bi_bcw`
  routes the luma + chroma composite through the eq. 981 weighted blend
  `Clip1((w0·p0 + w1·p1 + 4) >> 3)` (weights from `BCW_W_LUT`) when the
  picked candidate's `bcwIdx ∈ 1..=4`, and the eq. 980 average otherwise.
  Transform-skip inter residual (`residual_ts_coding`, §7.3.11.12) is
  now **live** on the inter luma + chroma + joint-Cb-Cr paths
  (§7.3.11.5 parse gate → §8.7.4.6 inverse-transform bypass).
* **Transforms** — DCT-II inverse (sizes 2..=64), DST-VII / DCT-VIII
  (4 / 8 / 16), flat-list dequant, the §8.7.2 scaling-and-transformation
  orchestrator with joint Cb-Cr derivation, the §8.7.4.6 inverse
  adaptive colour transform, the §8.7.4.1 – §8.7.4.3 **inverse
  LFNST** (low-frequency non-separable secondary transform) — all 16
  `lowFreqTransMatrix` tables, the Table-41 set selection, the
  intra-path `lfnst_idx` (§7.3.11.5) parse → §8.7.4.1 corner fold →
  pixels for **both square and non-square** TBs (the §8.4.5.2.7
  wide-angle remap is applied internally), and §8.7.4.1 **multiple
  transform selection (MTS)** on the intra luma path — implicit MTS
  (DST-VII for 4..16, eqs. 1167/1168) plus explicit MTS (`mts_idx`
  §7.3.11.5 parse → Table-39 kernel pair), with the eqs. 1169–1172
  non-zero-coefficient extents. **Transform-skip residual coding**
  (§7.3.11.12 `residual_ts_coding()`) is live on the intra luma **and
  chroma** paths: the §7.3.11.5 `transform_skip_flag` parse gate
  (`sps_transform_skip_enabled_flag`, `tbW/tbH <= MaxTsSize`, ISP
  NoSplit for luma, `!cu_sbt_flag`, BDPCM off) routes a TS TB through
  the three-pass TS coefficient walker
  (sb_coded_flag / sig / context-coded coeff_sign / gt1 / par in pass 1,
  the 5-flag gtX pass, the bypass `abs_remainder` + level-prediction
  tail) with the §9.3.4.2.6/.7/.8/.9/.10 TS ctxInc derivations (causal
  left/above neighbourhood, sig 60+, gtx 64+, par 32, sb 4+, `RemCcbs`
  bin budget) and the BDPCM-off level-prediction fold; reconstruction
  bypasses the inverse transform (§8.7.4.6 `res = d`) and LFNST is
  disabled on the TS TB (`lfnstNotTsFlag`).
* **CABAC** — full §9.3 arithmetic engine plus per-syntax-element
  initValue / shiftIdx tables for every element currently parsed.
* **In-loop filters** — §8.8.3 deblocking, §8.8.4 SAO (edge + band),
  §8.8.5 ALF including the fixed-filter family + CC-ALF.
* **LMCS** (§8.7.5.2 / §8.7.5.3 / §8.8.2) is **live** end-to-end: a
  slice with `sh_lmcs_used_flag == 1` (including the §7.4.8 PH-in-SH
  inference) reconstructs its luma in the mapped domain —
  `CtuWalker::set_lmcs` binds the `ph_lmcs_aps_id`-referenced payload,
  every MODE_INTER (non-CIIP) CU forward-maps its MC luma prediction
  (eq. 1213) before the eq. 1214 residual add, a CIIP CU forward-maps
  its inter part inside the §8.5.6.7 eq. 997 blend, intra CUs pass
  through, and the §8.8.1 step-1 picture inverse mapping (§8.8.2.2)
  runs at the head of the in-loop filter stack. Chroma residual
  scaling (§8.7.5.3, gated on `ph_chroma_residual_scale_flag`) derives
  `varScale` from the eq. 1216 `invAvgLuma` neighbour average of the
  sizeY-aligned containing CU (per-4x4 CU-origin grid + availL /
  availT gathers, eq. 1217 mid-grey fallback) and applies the
  eqs. 1219 / 1220 fold on the inter Cb / Cr, inter joint Cb-Cr and
  intra chroma residual paths (`nCurrSw * nCurrSh <= 4` pass-through
  honoured).

## Encoder

An IDR-frame encoder pipeline (`encode_idr_with_residuals_cfg`) builds
an Annex-B bitstream with real coded residuals, deblock, SAO, ALF, and
CC-ALF. Opt-in **palette coding** (`EncoderConfig::palette`, r431):
CUs whose source block draws from ≤ 31 distinct colours palette-code
losslessly (up to 37 with EG5 escape samples quantized at the CU QP),
with the wire-level predictor-reuse split derived from a
decoder-mirrored predictor palette; opt-in **multi-slice layouts**
(r431): `EncoderConfig::slice_per_tile` (rectangular one-slice-per-tile
— the §7.3.2.5 slice loop + `sh_slice_address`),
`EncoderConfig::raster_slice_count` (raster tile runs) and
`EncoderConfig::loop_filter_across_slices`, each slice its own VCL NAL
with full per-slice CABAC / predictor re-initialisation and
slice-local entry points. Plus opt-in partitioning tools: an MTT BT picker
(`EncoderConfig::enable_mtt_bt_picker`, `{leaf, BT_VERT, BT_HORZ}` on
`cost = SSE_Y + λ·bits`) and an MTT TT picker
(`EncoderConfig::enable_mtt_tt_picker`, adding `TT_VERT` / `TT_HORZ`),
and opt-in **LMCS** (`EncoderConfig::lmcs`): the coding loop runs in
the forward-mapped luma domain, the reconstruction is inverse-mapped
(§8.8.1 step 1) before the deblock / SAO / ALF designs, and the wire
carries `sps_lmcs_enabled_flag`, an LMCS APS NAL (§7.3.2.19 payload
via `aps_enc::emit_lmcs_aps_rbsp`), the §7.3.2.8 PH LMCS chain and
`sh_lmcs_used_flag` — all parse-verified against this crate's own
parsers. Opt-in **LMCS chroma residual scaling**
(`EncoderConfig::lmcs_chroma_scaling`, r391): the per-CU `varScale`
is derived decoder-mirrored (§8.7.5.3 eq. 1216 mapped-domain
neighbour-luma average, eq. 1217 fallback), each chroma TB codes the
forward-scaled residual (the exact inverse of the decoder's
eqs. 1219 / 1220 fold), the encoder reconstruction rescales
decoder-exact, and `ph_chroma_residual_scale_flag = 1` goes on the
wire. Opt-in **dependent quantization**
(`EncoderConfig::dep_quant`): every TB is quantised by the greedy
hard-decision TCQ `transform_fwd::quantize_tb_dep_quant` (trellis-
consistent by construction), reconstructed through the §8.7.3
dep-quant arms and emitted with the dep-quant CABAC paths, with
`sps_dep_quant_enabled_flag` + `sh_dep_quant_used_flag` on the wire;
opt-in **sign data hiding** (`EncoderConfig::sign_data_hiding`,
mutually exclusive per §7.3.7): `residual_enc::condition_levels_for_sdh`
parity-conditions every hidden-sign sub-block with a one-step nudge.

r387/r412 wire-conformance sweeps (black-box validated against a
conforming third-party VVC decoder): the emitted stream now carries
the §7.3.11.5 intra-mode cascade (every pipeline CU signals INTRA_DC
through the MPM path + DM chroma), the coding loop predicts real
§8.4.5.2.12 DC (+ PDPC) from the partially-reconstructed planes with
encoder-side eq. 1212 availability masks, `end_of_slice_one_bit` sits
only after the last CTU (§7.3.11.1), the final CABAC flush follows
§9.3.4.3.5 (the last consumed bit is the `rbsp_stop_one_bit`), SAO is
applied to the reconstruction only when actually signalled (and Cr's
decision is constrained to Cb's shared `sao_type_idx_chroma` /
`sao_eo_class_chroma`), the SPS signals `MaxTbSizeY = 64` and — with
the MTT pickers on — a real MTT depth and 64-sample MaxBt/MaxTt, and
all split bins follow the §6.4.1 – §6.4.3 presence/ctxInc rules.
`tests/whole_stream_conformance.rs` pins 11 tool axes (QP sweep,
multi-CTU, chroma SAO merge, MTT BT/TT, LMCS ± chroma scaling,
dep-quant, SDH) decoding BYTE-EXACTLY through this crate's own full
receive path.

r415/r418/r429 closed the external-decode gap completely: **124 of
124** corpus + probe streams (the r412 axes, the r418 extension —
deep-QP 51/57/63, MTT + multi-CTU at QP 45, the 192x128
partial-CTU-column layout —, the r429 tiles/WPP axes below, plus the
~90 single-feature probes of `tests/external_probe_corpus.rs`) decode
byte-exactly through a conforming external decoder invoked black-box.

r429 — **tiles + WPP end-to-end**: `EncoderConfig::{tile_columns,
tile_rows, wpp}` emit real §6.5.1 tile grids (PPS partition block,
one rectangular slice in tile-scan order, per-tile byte-aligned CABAC
subsets with `end_of_tile_one_bit`) and WPP (per-CTU-row subsets with
`end_of_subset_one_bit` + the §9.3.2.3/§9.3.2.4 context
storage/synchronization), with §7.4.8 entry-point offsets on the
wire; the decoder walks the same §7.3.11.1 plan. The
`pps_loop_filter_across_tiles_enabled_flag = 0` arm is live on both
sides (§8.8.3.1 deblock edge exclusion, §8.8.4.2 SAO edgeIdx = 0,
§8.8.5.5/§8.8.5.6 ALF tile-rectangle padding —
`EncoderConfig::loop_filter_across_tiles`). Twelve axes — tiles 2x1 /
2x2 / 3x1 / partial-right-column / 2x2+MTT, WPP 2-row / 3-row,
tiles+WPP combined, 2x2 / 3x1-deep-QP with the across-tiles filters
off, the 2x2 raster-scan slice layout, and the
tiles+WPP+across-off combination — decode byte-exactly through both
this crate's own receive path and the external reference decoder. The r412
sparse-residual divergence resolved into five fixed root causes
(r415): residual ctx-init table transcription drift (Tables
120 – 125), the §7.3.11.2 `alf_use_aps_flag` presence condition, the
§9.3.4.2.4 chroma last-prefix ctxShift exponent, the §7.4.3.7 per-CU
QG wire declaration, and reconstruction-stage conformance (ALF
virtual-boundary classification padding, §7.4.3.4 chroma QP table
identity + §8.7.1 table-mapped chroma QP, §8.8.3.3/§8.8.3.6.10
chroma CTB-row asymmetric deblocking). The r415 remainder (3
streams, 14 – 49 luma samples) root-caused in r418 to the §8.8.3.6.7
weak-filter p1/q1 clip bound (`−(tC >> 1)`, eqs. 1385/1387 — odd-tC
corner); the same round landed the full §8.8.3.6.2 decision (step-6
luma CTB-row rule, asymmetric §8.8.3.6.8 long filters eqs.
1391 – 1394, the §8.8.3.3 either-side ≤ 4 rule, and the step-9
gates). Validation matrix: `tests/WHOLE_STREAM_CORPUS.md`.

r418 also landed the §7.3.11.4 **quantization-group** state machine in
the single-tree walker: `cbSubdiv` / `qgOnY` threading, per-QG
`IsCuQpDeltaCoded` arming (one `cu_qp_delta_abs` per QG, later CUs
inherit `CuQpDeltaVal`), and the §8.7.1 `qPY_PRED` derivation
(per-4x4 QpY map, qPY_A/qPY_B CTB-containment fall-backs,
first-QG-in-CTB-row arm, eq. 1119/1120) — so wires whose PH signals a
`CuQpDeltaSubdiv` smaller than this encoder's per-CU maximum decode
correctly (`CtuWalker::set_cu_qp_delta_subdiv`), pinned by hand-built
foreign-wire CABAC fixtures. The single-tree walker also performs the
§7.4.12.4 **picture-boundary walk** (full-CTB square, inferred splits
at the edges, coded boundary BT, implicit-BT `depthOffset`,
implicit-level `cqtDepth`), so non-CTB-multiple picture layouts
decode end-to-end.

## JVET conformance triage (r434, re-triaged every round since)

The staged JVET FDIS-r1 conformance corpus (56 official bitstreams,
`docs/video/h266/conformance/` in the workspace) now runs through a
stream-level decoder: the new `stream::StreamDecoder` adds the glue an
Annex-B elementary stream needs on top of the CTU walker — parameter-set
/ APS pools, access-unit assembly (PH NAL, PH-in-SH, multi-slice),
§8.3.1 POC, §8.3.2 reference-picture-list resolution against a DPB,
§8.3.3-style retention, CLVSS / RASL handling across concatenated CVSs,
per-slice walker wiring and conformance-crop propagation.
`tests/jvet_conformance.rs` (a no-op without the docs tree) decodes
every stream, hashes each output picture's cropped planes against the
official `.opl` sidecars, and pins a per-stream verdict baseline.

Corpus-driven r434 conformance fixes: full §7.3.7 slice-header parse
for foreign wires (SH-carried `ref_pic_lists()`, `NumRefIdxActive`,
collocated, SH `pred_weight_table()`, full embedded-PH parse, eq. 23
subpicture slice addressing, `sps_rpl1_same_as_rpl0_flag` inference),
the §7.3.11.11 thin-TB sub-block shapes (2x8 / 8x2 / 2x2), intra joint
Cb-Cr parse + §8.7.2 reconstruction on all three intra TU paths, the
dual-tree coding-tree walk rebuilt in the spec's interleaved bin order
with the §7.4.12.4 picture-boundary walk + luma-tree QG arming (and
the §7.3.11.10 chroma-tree `cu_qp_delta` exclusion), and the §7.4.8
deblocking-parameter inheritance chain (PPS β / tC offsets were
previously discarded).

**r437 — the >8-bit reconstruction pipeline is live.** `PicturePlane`
stores `u16` samples at an explicit BitDepth (8..=16); every
BitDepth-specialized arm was generalized to the spec derivations (the
§8.5.6.6.2 closing clamps, §8.5.6.3 MC shift1/lift, §8.5.6.4 PROF,
GPM re-lift, SAO/ALF/deblock/palette/IBC/LMCS clips, deblock β/tC
eqs. 1276 – 1279 / 1345 – 1348, DMVR), and the stream driver decodes
Main10 wires end-to-end (conformance hashing at 2-byte LE per the
corpus convention). Corpus-driven r437 conformance fixes, validated
sample-level against a black-box reference decoder: the §8.4.2 MPM
candidate positions (`A = (xCb − 1, yCb + cbHeight − 1)`, `B =
(xCb + cbWidth − 1, yCb − 1)` — the parse previously sampled the
§9.3.4.2.2 ctxInc cells, the r434 "mode 11 vs 7" root cause), the
§8.7.3 dequant Qp′ domain (`+ QpBdOffset` at every site + the
eq. 1141 / 1144 clips and TS floor in `DequantParams`), the
eqs. 1135 – 1139 JCCR qP selection (joint QP only for TuCResMode 2),
and §8.8.3 chroma deblocking (chroma-tree edge records on dual-tree
slices per §8.8.3.2, and the §8.8.3.1 chroma 8×8-grid exclusion in
chroma-sample units).

**r440 — the "CTC-toolset desync family" (46 streams) is dissolved
and `CodingToolsSets_A_2` is the first fully byte-exact corpus
stream.** The family's root causes, found clause-by-clause with
locally-built single-tool fixture streams (a staged black-box encoder
produces them, an independent black-box decoder is the oracle, and
`examples/decode_dump` diffs our planes sample-level):

* **Publication errata in the 2026-01 (V4) edition**, pinned by
  orthogonality + derivation tests from the Recommendation's own
  printed data: the §8.7.4.3 LFNST matrices print each 16×16
  sub-block transposed (the eq. 1176 multiply indexes them
  accordingly), and — as r443 established — EVERY §8.7.4.5 MTS table
  body is printed the same way (`transMatrix[m][n]` declared with `m`
  the transformed-sample index while the printed rows carry `n`); the
  statics stay byte-for-byte as printed and the eq. 1177 / 1178
  multiplies index them transposed.
* **ISP transform units** re-shaped to spec order (`cu_qp_delta`
  inside the first CBF-positive partition's TU, last-partition chroma
  reads, chroma TS flags), the previously-missing **ISP `lfnst_idx`**
  read, per-partition **implicit-MTS / LFNST reconstruction**, and
  the Table-132 `lfnst_idx` ctxInc tree arm.
* **Missing single-tree `cclm_mode_flag`**, **MIP / BDPCM transform
  units** (both arms returned before their TU reads), and the BDPCM
  §7.4.12.5 transform-skip inference.
* Dual-tree CTB-128: the §8.4.4 64-grid `CqtDepth` threshold
  (`CtbLog2SizeY − 6`, not the node-local 0), the §7.3.11.2
  implicit-QT `cbSubdiv = 2·cqtDepth` seed + CTU-level QG
  declaration, and the previously parsed-but-dropped §7.3.2.8 PH
  partition-constraints override.
* §8.8.3.6.4 chroma deblock QPs (per-TB `Qp′Cb` / `Qp′Cr` /
  `Qp′CbCr` via the ChromaQpTable, eq. 1343), the §8.7.1 dual-tree
  chroma `QpY` (collocated-centre luma CU), and the §8.4.5.2.14 CCLM
  neighbour-array bound (`2·nTbW` / `2·nTbH` — tall T-CCLM TBs picked
  their 4-point model from oversampled positions).

Scorecard (r437 → r440 → r443 → r447 → r449 → r450): 0 P / 2 F / 8 U / 46 E →
1 P / 7 F / 26 U / 22 E → 1 P / 14 F / 33 U / 8 E →
2 P / 17 F / 8 U / 29 E → 14 P / 31 F / 9 U / 2 E →
**34 PASS / 12 FAIL / 9 UNSUPPORTED / 1 ERROR** — twenty streams went
byte-exact in r450 alone (see the r450 block below). In r449
`DMVR_B_4` joined
`CodingToolsSets_A_2` as byte-exact, and the r443 "affine non-merge"
gate (the largest U family) is DISSOLVED. r447's root causes, each
pinned with black-box single-tool fixture matrices that now decode
**byte-exactly end-to-end** (affine, affine-AMVR, SMVD, MMVD, CIIP,
BDOF, PROF, SbTMVP):

* **§7.3.11.5 cascade order** — `inter_pred_idc` precedes the affine
  pair, and `sym_mvd_flag` presence carries `!inter_affine_flag`;
  both dispatcher sides mirrored the same divergence, so round-trips
  passed while every conforming affine-enabled B-slice wire desynced
  (the reason the affine rows sat at UNSUPPORTED: the old
  reconstruction refusal fired before the desync could).
* **§7.4.3.7 PH inferences** — absent `ph_bdof/dmvr/prof_disabled`
  infer `1 − sps_*_enabled`; absent `ph_mvd_l1_zero_flag` infers 1.
* **§8.5.2.12 / §8.5.2.15 temporal motion** — the export resolves
  per-4×4-cell reference POCs so eq. 598 `colPocDiff` is the
  collocated block's own distance (scaling was disabled by an
  equal-distance shortcut); the both-lists arm picks `listCol` per
  `NoBackwardPredFlag` / `sh_collocated_from_l0_flag`; buffer
  compression is the real eqs. 611 – 615 mantissa/exponent rounding.
* **§7.3.11.7 (V4) GPM presence** — the `regular_merge_flag` GPM arm
  has no `cu_skip` / `MaxNumGpmMergeCand` terms (a GPM-eligible SKIP
  CU desynced every GPM wire); GPM reconstruction predicts at the
  CU's picture-absolute origin, blends eq. 1016 on real §8.5.6.3
  intermediates, stores §8.5.7.3 per-4×4 sType motion, and decodes
  its residual.
* **§8.5.6.6.2 high-precision bi-pred** — eqs. 978 – 981 compose the
  BitDepth + 6 per-list arrays on the translational, affine, SbTMVP
  and GPM paths (the legacy final-domain `(a + b + 1) >> 1` average
  double-rounded every fractional bi-pred block).
* **§8.5.6.3.2 / §8.5.6.1** — PROF and BDOF border samples are
  §8.5.6.3.3 integer fetches at the fraction-rounded position; BDOF
  runs per ≤16×16 sub-block (eqs. 888/889); DMVR MC reads are bounded
  by eqs. 926/927 around the unrefined MV.
* **§8.8.3.4 / §8.8.3.5 deblocking** — full boundary-strength cascade
  (motion-difference / CIIP / IBC arms off a per-4×4 grid with
  resolved reference POCs), sub-block interior edges for affine /
  sub-block-merge CUs, per-64-tile CBF maps + interior TB-edge
  filtering for multi-TB CUs.

The r443 rollup below documents the earlier family:

* **A third V4 publication-erratum finding that GENERALIZES the r440
  pair**: every §8.7.4.5 MTS table body is printed transposed
  relative to the declared `transMatrix[m][n]` layout (the eq. 1179
  index ranges and the eq. 1183 / 1184 reflections prove it from the
  printed text alone). The DCT-II path already read the bodies
  transposed; the eq. 1178 DST-VII multiply did not — every DST-VII
  TB at every size reconstructed through the transposed kernel
  (asymptomatic for the symmetric DCT-VIII). The r440-derived
  32-point "rows 16..31" constants dissolved into the same reading.
* **§9.3.2.2 initType columns** — every P/B slice previously
  initialised the coding-tree + residual CABAC context bundles (and
  the leaf reader's intra-family elements) from the I-slice rows,
  desyncing on the first inter picture.
* **§7.3.11.12 TS gtX budget** (once per coefficient, not per flag),
  the **§7.4.12.5 BDPCM transform-skip inference on the
  dual-tree-chroma TU path**, the **§7.3.11.10 `chromaAvailable`
  derivation on the dual-luma IBC path** (+ §8.4.3 INTRA_DC
  collocated substitution for MODE_IBC, Table 82 general_merge
  ctxIdx), and the **§8.5.1 eqs. 456 / 457 DMVR sub-block dims**.
* **§7.3.11.9 multi-TB tiling** — inter AND intra CBs above
  `MaxTbSizeY` parse per-TU and reconstruct per tile (the §8.4.5.1
  eq. 248 – 250 per-transform-block intra prediction).
* **§7.3.11.4 local dual tree (SCIPU)** — `ModeTypeCondition`,
  `non_inter_flag`, DUAL_TREE_LUMA subtrees and the single
  DUAL_TREE_CHROMA leaf in the single-tree walker.

**r449 — every intra picture in the corpus is byte-exact, and the
parse-desync ERROR family is DISSOLVED.** The round's root causes,
each pinned with black-box fixture matrices (10-bit RA wires over
static / noise / moving / screen content) that decode byte-exactly
end-to-end:

* **§8.8.3.2 – §8.8.3.5 deblocking rebuilt as the spec's per-CU edge
  walk** — interior ISP / SBT transform-block edges filter (new
  `DeblockCu::tb_split`), every `maxFilterLength` derives from the
  TRANSFORM-BLOCK dims, edges process in geometric order (the
  §8.8.3.1 same-direction data dependency — the r447 "intra deblock
  tail" root cause), the §8.8.3.4 `edgeTbFlags` overlay and exact
  §8.8.3.5 arm gating (tu-coded arm on TB edges only, motion arms at
  `edgeIdc == 2` only), and the §8.8.3.6.4 chroma `(1, 1)`-length
  edges filter only at `bS == 2`.
* **§8.7.5.3 eq. 1216** — `invAvgLuma` summed native-BitDepth samples
  through a stale `<< (BitDepth − 8)` lift, wrecking LMCS chroma
  residual scaling on every >8-bit wire; **§7.4.3.18** — CC-ALF
  coefficients are MAPPED (`±2^(abs − 1)`), the parse used the raw
  magnitude (and the emit now writes the mapped one, with CC-ALF
  gated on its own §7.3.7 slice flags).
* **§7.3.11.5 / §7.3.11.9 SBT parse** (`cu_sbt_*` with Tables
  93 – 96, the zero-TU inference walk, the §7.3.11.11 SBT 32 → 16
  zero-out, Table-40 kernels under `implicitMtsEnabled`, sub-TU-rect
  chroma reconstruction) — the whole "post-affine-gate parse desync"
  E family (29 rows) was SBT-enabled wires desyncing at their first
  coded eligible inter CU.
* **§8.5.2.1 – §8.5.2.9 merge / AMVP details** — the
  `(cbWidth + cbHeight) == 12` bi→uni collapse, the
  Log2ParMrgLevel-gated HMVP update (level now from the SPS), the
  AMVP HMVP fill counted after the A == B dedup, the eqs. 471 – 474
  18-bit wrap, and the §8.5.2.5 zero-pad `zeroIdx` refIdx counter.
* **§8.5.7.2 eq. 1014** — the GPM blend's `partFlip` arms were
  inverted, mirroring partition A and B across the split line (masked
  whenever both partitions predict near-identical content; exposed at
  picture-edge GPM CUs).
* **§8.5.6.1** — `sbBdofFlag`: BDOF skips per DMVR sub-block when the
  refinement SAD is under `2·sbW·sbH`; and `sym_mvd_flag == 0` is a
  `bdofFlag` bullet (SMVD CUs never run BDOF). **§8.5.6.3.2
  `hpelIfIdx`** — the Table 27 alternative 6-tap half-sample filter
  is live end-to-end (eq. 475 + merge/HMVP/pairwise inheritance).

**r450 — the poc-1-luma inter family COLLAPSES: 14 P → 34 P.** Six
spec fixes, each root-caused from the first diverging pixel against
the corpus (a black-box reference decode supplies per-sample diffs):

* **§8.8.3.3 implicit >MaxTbSizeY transform-block tiling reaches the
  deblocker** — a 128-wide SKIP CU still has TB boundaries at the
  MaxTbSizeY grid (residual or not), so its interior sub-block edge
  is a TB edge taking `maxFilterLength` from the 64-sample TB dims
  (`Min(5, ·)` via §8.8.3.4 eqs. 1231/1232) instead of the length-3
  arm. THIRTEEN streams flipped on this one line of geometry (the
  corpus-wide "±1‥6 long-filter halo" on every large skip / affine /
  SbTMVP CU); the `Tu64CbfMap` now carries `tile_log2` for the
  32-sample tiling when `sps_max_luma_transform_size_64_flag == 0`.
* **§8.5.2.7 MMVD offsets** — the raw `MmvdOffset` lands on the list
  whose reference is FARTHER (ties to L0), and short-term references
  ALWAYS take the eqs. 563 – 578 distScaleFactor chain (opposite-sign
  distances were short-circuited to plain negation — the
  long-term-only arm). `GPM_A_3` + `LFNST_B_4` flipped.
* **§8.5.2.3 raw-availability pruning** — the A0-vs-A1 / B2-vs-A1 /
  B2-vs-B1 redundancy gates key on `availableX` (the §6.4.4 block
  availability) and the neighbour's stored motion, NOT the post-prune
  `availableFlagX`: when A1 was pruned (== B1) a duplicate A0 == B1
  entered every list and Col / HMVP / pairwise shifted one slot.
  `CodingToolsSets_B_2` / `CodingToolsSets_D_2` / `JCCR_C_3` flipped.
* **§8.5.6.1 CIIP chroma gate + §8.8.3.6.4 per-TB joint mode** — the
  CIIP chroma intra blend runs only when `cbWidth / SubWidthC >= 4`
  (2-wide chroma TBs keep the pure inter prediction), and the chroma
  deblock QP picks `Qp′CbCr` per TRANSFORM BLOCK (an SBT CU's
  residual-free sub-TU keeps per-component QPs). `JCCR_A_2` flipped.
* **§8.5.6.3.4 eqs. 944/945 + §8.5.5.9 eqs. 858 – 861** — the DMVR
  chroma bounding block is (w + 3) samples (the patch read one true
  sample past it), and the affine fallback bounding box takes the X
  partials on the width axis / Y partials on the height axis (both
  axes read the Y partials, so wild constructed candidates never
  tripped `fallbackModeTriggered`). `QUANT_A_2` flipped; `RAP_B_1`
  went 28 → 1 divergent and flipped after the tiling fix.
* **§8.5.5.2 steps 3 – 6 gate on `sps_affine_enabled_flag`** (an
  affine-off SbTMVP wire pads with zero-MV candidates — phantom
  constructed candidates ran garbage affine MC), and **§6.4.4
  `checkPredModeY`** in the spatial-merge scan (a MODE_IBC
  neighbour's block-vector record is not an inter merge candidate —
  `IBC_A_2` stops ERRORING and decodes end-to-end).

The 12 remaining FAIL rows diverge late and small (typically one
sub-CU cluster per picture, chroma-first on several — e.g.
`MMVD_A_3` is 21/300 pictures over a single ±1 chroma sample chain);
the 9 UNSUPPORTED rows are subpictures, 4:2:2 / 4:4:4, explicit
weighted prediction, per-slice loop-filter divergence and explicit
§8.8.1 virtual boundaries (`GDR_A_2`'s real gate); the 1 ERROR row
(`LOSSLESS_B_3`) trips an IBC reference-region guard on picture 0
(under triage). `examples/triage_dbg` decodes one corpus stream
against its `.opl` sidecar with optional plane dumps;
`examples/decode_dump` writes POC-ordered YUV for fixture diffing;
`examples/sps_dump` prints a stream's SPS tool-flag set; the
`H266_DBG_*` env family (`_TB`, `_CCLM`, `_CU`, `_IBC`, `_MERGE`,
`_MMVD`, `_DMVR`, `_SBTMVP`, `_AFFAMVP`, `_AFFCU`, `_AFFBI`,
`_AFFUNI`, `_LMCS`, `_SHQP`, `_PIX`, `_RPL`) dumps per-CU pipeline
state, and `H266_DUMP_PREFILTER` writes the pre-filter mapped-domain
reconstruction per picture. The corpus triage remains the priority
pick list.

An inter-frame P-slice and B-slice encoder + decoder scaffold
(`encoder_inter::encode_p_slice` / `encode_b_slice` and their decoders)
provides single / dual-reference DPB, integer-pel full-search motion
estimation with half- and quarter-pel sub-pel refinement through the
§8.5.6.3 interpolation filters, a spatial MVP picker, and per-block
CABAC emit of the inter syntax elements through the residual coefficient
chain. AMVR, `hpelIfIdx` filter selection, chroma sub-pel MC, multi-ref
DPB, and full Annex-B NAL integration with the IDR pipeline are still
being added.

## Usage

Registering the codec wires the decoder into `oxideav`'s codec
registry:

```rust
use oxideav_codec::CodecRegistry;
let mut codecs = CodecRegistry::new();
oxideav_h266::register_codecs(&mut codecs);
```

Or, via the unified `RuntimeContext` entry point:

```rust
let mut ctx = oxideav_core::RuntimeContext::new();
oxideav_h266::register(&mut ctx);
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
