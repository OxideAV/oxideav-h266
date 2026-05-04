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
  No GPM / AMVR yet; affine + scaled-reference filter tables 28 /
  29 / 30 / 31 / 32 / 34 / 35, DMVR, PROF land in later rounds.
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
  mvd_coding (non-merge inter), GPM, AMVR, BCW**: still surface
  `Error::Unsupported`.
  (HMVP — §8.5.2.6 + §8.5.2.16 — landed in round-24; temporal merge
  — §8.5.2.11 + §8.5.2.12 — landed in round-25; pairwise-average
  merge — §8.5.2.4 — landed in round-26; MMVD — §8.5.2.7 — landed in
  round-27; CIIP — §8.5.6.7 — landed in round-28.)

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
