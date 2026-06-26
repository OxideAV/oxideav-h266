# oxideav-h266

Pure-Rust **H.266 / VVC** (ITU-T H.266 | ISO/IEC 23090-3) codec for
oxideav. Zero C dependencies, no FFI, no `*-sys` crates.

VVC is a very large specification (500+ pages); this crate is an
in-progress, spec-driven implementation. It provides a complete NAL
framing + parameter-set parsing layer, a reconstruction pipeline that
covers a growing subset of the intra and inter coding tools, and an
IDR-frame encoder pipeline. Picture reconstruction for the full
non-merge inter path and a complete encoder are still being built up
incrementally.

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
  * **APS** (§7.3.2.5) — ALF / LMCS / scaling-list type. The LMCS APS
    payload (§7.3.2.19) is decoded into a typed `lmcs::LmcsData`, with
    the BitDepth-dependent §7.4.3.19 derivations (`OrgCW`, `lmcsCW`,
    `LmcsPivot`, `ScaleCoeff` / `InvScaleCoeff`, `ChromaScaleCoeff`) and
    the sample-domain forward / inverse luma mapping + chroma residual
    scaling folds.
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
the **intra-only single-tile single-slice subset** plus a substantial
P + B-slice merge subset:

* **Intra** — PLANAR / DC / cardinal angular intra (modes 2, 18, 34,
  50, 66), MIP (§8.4.5.2.2, all 30 weight matrices), CCLM
  (§8.4.5.2.14), BDPCM, and ISP (§8.4.5.1, all 4 split types).
* **Inter merge** — P + B-slice skip / merge: §8.5.2.3 spatial-merge
  candidates, §8.5.2.11 / §8.5.2.12 temporal collocated candidate with
  POC scaling and buffer compression, §8.5.2.6 HMVP, §8.5.2.4
  pairwise-average candidate, and zero-MV padding; §8.5.2.7 MMVD;
  §8.5.6.7 CIIP; §8.5.4 / §8.5.7 GPM. Motion compensation uses the
  §8.5.6.3 8-tap luma + 4-tap chroma fractional-sample interpolation
  with default-weighted bi-pred (§8.5.6.6.2), BCW (§8.5.6.6.2 eq. 981),
  and BDOF (§8.5.6.5). High-bit-depth (Main10 / Main12) reconstruction
  runs through `u16` picture planes. A non-skip merge / CIIP CU whose
  `cu_coded_flag == 1` decodes its §7.3.11.10 `transform_unit()`
  residual (MODE_INTER luma-CBF condition + chroma CBFs) and adds the
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
  parsed CPMV deltas end-to-end. **Inherited affine CPMVP** (§8.5.5.7
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
  §8.5.5.3 SbTMVP (`SbCol`) sub-block-temporal candidate and BCW / BDOF
  on the affine path remain follow-ups.
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

## Encoder

An IDR-frame encoder pipeline (`encode_idr_with_residuals_cfg`) builds
an Annex-B bitstream with real coded residuals, deblock, SAO, ALF, and
CC-ALF, plus opt-in partitioning tools: an MTT BT picker
(`EncoderConfig::enable_mtt_bt_picker`, `{leaf, BT_VERT, BT_HORZ}` on
`cost = SSE_Y + λ·bits`) and an MTT TT picker
(`EncoderConfig::enable_mtt_tt_picker`, adding `TT_VERT` / `TT_HORZ`).

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
