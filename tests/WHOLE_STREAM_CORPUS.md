# Whole-stream decode conformance corpus (r412, externally validated r415/r418, tiles/WPP r429, palette + multi-slice r431)

The corpus streams are generated deterministically by
`tests/whole_stream_conformance.rs` — every test encodes with the
crate's IDR pipeline and asserts the crate's own full receive path
(NAL walk → parameter-set / PH / SH parsers → CABAC CTU walker with
in-stream SAO + ALF prefixes → §8.7 dequant + inverse transforms →
§8.8 in-loop filters) reproduces the encoder reconstruction
**byte-exactly**. All 11 axes hold.

`tests/external_probe_corpus.rs` extends the corpus with ~90
single-feature probe streams (TB-size sweep 8..64, single luma/chroma
coefficients by frequency position, amplitude sweeps, chroma-only
planes, gradient / stripe / checker content, QP sweep, 128x128
full-CTB four-CU walk) for black-box bisection against an external
reference decoder.

r429 adds the **tiles / WPP** axes: the encoder emits real §6.5.1
tile grids (PPS partition block, one rectangular slice covering the
picture in tile-scan order, per-tile byte-aligned CABAC subsets with
`end_of_tile_one_bit` + §9.3.2.2 context re-initialization) and WPP
(`sps_entropy_coding_sync_enabled_flag`, per-CTU-row subsets with
`end_of_subset_one_bit` + the §9.3.2.3/§9.3.2.4 context
storage/synchronization), with §7.4.8 entry-point offsets on the
wire; the decoder walks the same §7.3.11.1 plan with the §6.4.4
different-tile / WPP-column availability gates.

## Generation

```
H266_CORPUS_DIR=<dir> cargo test --test whole_stream_conformance
H266_CORPUS_DIR=<dir> cargo test --test external_probe_corpus
```

writes `<name>.266` (Annex-B) and `<name>.yuv` (decoded planar 4:2:0,
Y then Cb then Cr) per axis. Content is fully deterministic; the
SHA-256 prefixes below were recorded on 2026-07-20 (they change
whenever the encoder's wire evolves — regenerate rather than diff).

## Black-box reference-decoder validation

Each `<name>.266` is decoded to planar 4:2:0 through a conforming
external reference decoder invoked black-box, and the output is
`cmp`'d byte-for-byte against the crate's own `<name>.yuv`.

r431 status: **136 of 136 streams byte-exact** (the 124 r429 streams
+ the 5 palette axes + the 7 multi-slice axes below). The palette axes validated on the first
run except `palette_mixed_256x128`, whose 133-sample divergence
root-caused to the §8.8.3.6.7/.6.8/.6.10 palette-side deblock
suppression (a palette CU's samples are never modified by the
deblocker even when the shared edge filters — nDp/nDq → 0 and the
input-sample substitution); both the decoder and the encoder's own
deblock application now carry the rule.

r429 status: **124 of 124 streams byte-exact** (the 22 historical
corpus axes + the 12 r429 tile/WPP axes — including the
`pps_loop_filter_across_tiles_enabled_flag = 0` axes, the raster-scan
slice layout, and the tiles+WPP+across-off combination — + all ~90
probe streams).

r418 status: **112 of 112 streams byte-exact** (all corpus axes —
including the 8 r418 extension streams below — plus all ~60 probe
streams). The r415 remainder — 3 streams (`qp45`,
`mtt_bt`, `mtt_bt_tt`) with 14 – 49 luma-sample recon-only diffs near
32/64-aligned CU edges — root-caused to the §8.8.3.6.7 weak-filter
p1/q1 clip bound transcribed as `(−tC) >> 1` instead of the spec's
`−(tC >> 1)` (eqs. 1385/1387): the arithmetic shift rounds toward −∞,
over-widening the bound by 1 for odd tC (tC = 13 at QP 45 / bS 2 is
the first corpus point with odd tC AND a binding clip). Root-caused
by an independent §8.8.3.6.2/.6/.7/.8 transcription run sample-exact
against the staged-stage dumps. The same r418 commit lands the rest
of the §8.8.3.6.2 decision faithfully: the step-6 luma CTB-row rule
(EDGE_HOR on a CTB row forces `sidePisLargeBlk = 0` → eq. 1294 caps
`maxFilterLengthP` at 3), the asymmetric §8.8.3.6.8 long filters
(eqs. 1391 – 1394 refMiddle arms + the 3-deep eq. 1401/1402
`fi`/`tCPDi` arrays), the §8.8.3.3 either-side ≤ 4 → both-1 luma
`maxFilterLength` rule, and the step-9 gates (strong-short only when
both lengths > 2; `dEp`/`dEq` only when both > 1).

r415 status was 101 of 104; the r412 "sparse residual" divergence and
the dual-tree/bin-interleave characterization resolved into five
distinct root-cause families, all fixed in r415:

1. residual ctx-init table transcription drift (Tables 120 – 125:
   dropped/duplicated `initValue`/`shiftIdx` entries, ~615 wrong cells);
2. `alf_use_aps_flag` read/written without the
   `sh_num_alf_aps_ids_luma > 0` presence condition (§7.3.11.2);
3. chroma `last_sig_coeff_*_prefix` ctxShift mis-transcribed as
   `2 * log2TbSize >> 3` — the spec's `2` carries the exponent:
   `(1 << log2TbSize) >> 3` (§9.3.4.2.4);
4. `ph_cu_qp_delta_subdiv_intra_slice = 0` on the wire while the
   pipeline arms `cu_qp_delta` per CU (§7.4.3.7 QG mismatch);
5. reconstruction-stage deviations: ALF classification missing the
   §8.8.5.5/§8.8.5.6 virtual-boundary padding, the emitted chroma QP
   table deriving to `QpC = QpY − 1` above its start (§7.4.3.4) while
   dequant assumed identity, and chroma deblocking missing the
   §8.8.3.3 CTB-row `maxFilterLengthP = 1` cap + §8.8.3.6.10
   asymmetric (1,3) filter.

| axis | vs own decoder | vs external reference | stream sha | plane sha |
|------|----------------|---------------|------------|-----------|
| flat_qp26 | byte-exact | byte-exact | 6debac3fbc151682 | 8c8362c09e7c37cf |
| default_qp26 | byte-exact | byte-exact | 6fc38b8dde443083 | e53959ce6e82c01d |
| qp10 / qp17 / qp34 | byte-exact | byte-exact | 62e1216c5c104422 / c5b2cbac3acdb99b / 4c4664e65bcf4681 | 6d656ef15dbc4e1c / c5b026b42c70da6f / e2ace0006bd6586e |
| qp45 | byte-exact | byte-exact (r418) | 502c2535626f2536 | 28e105132000b8ae |
| multi_ctu_256x256 | byte-exact | byte-exact | 11f98182f582f91e | 93d9ef34ff36b5d9 |
| chroma_sao_merge | byte-exact | byte-exact | 6442a2d1fe64f0e8 | b58a0daff3741f39 |
| mtt_bt / mtt_bt_tt | byte-exact | byte-exact (r418) | b9a10b06a87d1122 / 598e470f2d04655c | 13fe347be6c297ad / 9b7e0a8864d97aa5 |
| lmcs / lmcs_chroma_scaling | byte-exact | byte-exact | 81af59718db2c07b / 8a91b058f84df4d3 | 1e164146428d7493 |
| dep_quant / sign_data_hiding | byte-exact | byte-exact | bfc3898b6c9d140b / 0c841ad45810c9a8 | ff8e5a3a0c924e49 / d27ad4ff087635a6 |
| qp51 / qp57 / qp63 (r418) | byte-exact | byte-exact | 471edd9fb6bb7064 / b95e46609b7d6efb / c26ea7b881e008f2 | 2e68ce6e24117003 / 25b4cd760abc6067 / 9c5d7c150f204c9f |
| tiles_2x1_256x128 (r429) | byte-exact | byte-exact | cf10834c0684f582 | 92c536fc19011855 |
| tiles_2x2_256x256 (r429) | byte-exact | byte-exact | 934eeeefa23fa757 | 7115a4ba9591e019 |
| tiles_3x1_384x128_qp34 (r429) | byte-exact | byte-exact | 299485e46dbd68cb | bd8a0bd6c7107648 |
| tiles_2x1_192x128 (r429) | byte-exact | byte-exact | f535d3a2c05c441c | 7e9fbb04c4ec7daf |
| tiles_2x2_mtt_qp30 (r429) | byte-exact | byte-exact | 6a1f04c76cfae5f2 | be7537bce8a3a534 |
| wpp_256x256 (r429) | byte-exact | byte-exact | 1971e4423601d0ad | 93d9ef34ff36b5d9 |
| wpp_128x384_qp34 (r429) | byte-exact | byte-exact | 93cfcd8548651866 | a1262fdd00b70efc |
| tiles_2x1_wpp_256x256 (r429) | byte-exact | byte-exact | 91e8b76ef903ee3b | a36b8894f3f0e6af |
| tiles_2x2_noxlf_256x256 (r429) | byte-exact | byte-exact | a28440cd8459eec8 | da1868ced29dbc99 |
| tiles_3x1_noxlf_qp45 (r429) | byte-exact | byte-exact | 38ef3fc5de5ebf45 | 1b32566fc3457ad3 |
| tiles_2x2_raster_256x256 (r429) | byte-exact | byte-exact | d588efca23ed6aa7 | 7115a4ba9591e019 |
| tiles_2x1_wpp_noxlf_qp34 (r429) | byte-exact | byte-exact | 3188d3d560f8909d | ec4a9d6d0f040232 |
| palette_screen_128x128 (r431) | byte-exact | byte-exact | 69501db44697a1d8 | 5b52c9adf04c5457 |
| palette_mixed_256x128 (r431) | byte-exact | byte-exact | 9ef6ab89c35e823b | 85bc88c14b1b498f |
| palette_escape_64x64_qp30 (r431) | byte-exact | byte-exact | 2c529c7a279d4abe | 7424f81bd738c6b4 |
| palette_tiles_2x2_256x256 (r431) | byte-exact | byte-exact | 44dbffca4fa2413e | 89b53f359d0a972b |
| palette_wpp_256x256 (r431) | byte-exact | byte-exact | 26de0befec4d5ade | 89b53f359d0a972b |
| slices_rect_2x2_256x256 (r431) | byte-exact | byte-exact | 62292d12500e2a33 | 7115a4ba9591e019 |
| slices_rect_3x1_384x128_qp34 (r431) | byte-exact | byte-exact | 4795b3a6250f7922 | bd8a0bd6c7107648 |
| slices_raster_2_256x256 (r431) | byte-exact | byte-exact | 7fe8bd3849eaa102 | 7115a4ba9591e019 |
| slices_rect_noxlf_2x2_256x256 (r431) | byte-exact | byte-exact | b72c2d13c1ee23f9 | da1868ced29dbc99 |
| slices_raster_noxlf_3rows_128x384_qp45 (r431) | byte-exact | byte-exact | 9590aa3993caff82 | f6b5a6d1e6aed23d |
| slices_rect_wpp_128x384 (r431) | byte-exact | byte-exact | 22336d8cdea60f62 | cdf27191c1f2fb6d |
| palette_slices_rect_2x2_256x256 (r431) | byte-exact | byte-exact | 0d06a3ac18eb8b24 | 89b53f359d0a972b |
| mtt_bt_qp45 / mtt_bt_tt_qp45 (r418) | byte-exact | byte-exact | 8daf4a85db40ec28 | 28e105132000b8ae |
| multi_ctu_qp45 / multi_ctu_mtt_qp45 (r418) | byte-exact | byte-exact | b54b2fe36d3200de / 99c0790dfdfdb821 | 052997467c44de0a |
| wide_192x128_qp45 (r418) | byte-exact | byte-exact | d162706ece057f57 | 9de6a3f26d0c8df2 |

r418 extension notes: the deep-QP sweep (51 / 57 / 63) covers the
odd-tC deblock threshold band up to the table maxima; `multi_ctu_qp45`
exercises the §8.8.3.6.2 step-6 luma CTB-row rule externally (interior
CTB row at y = 128 with 64-tall CUs and active long filters);
`wide_192x128_qp45` is the first non-CTB-multiple layout — its 64-wide
right CTB column decodes through the §7.4.12.4 boundary implicit-split
walk. At QP 45 the MTT pickers choose no splits on this content (the
`mtt_*_qp45` streams differ from `qp45` only in the SPS MTT signalling
and the identical reconstruction confirms the tree parse); the QP-30
`mtt_bt` / `mtt_bt_tt` axes remain the MTT-shape coverage.

Probe extension (`external_probe_corpus.rs`): all ~60 probe streams
byte-exact through the reference decoder, including every sparse
single-coefficient case the r412 characterization flagged, all chroma
probes, and the 128x128 four-CU walk.

r429 tile/WPP notes: `tiles_2x2_mtt_qp30` runs the MTT pickers
against the tile-gated split-flag ctxIncs and prediction
availability; `tiles_2x1_192x128` puts a 64-sample-wide tile over the
partial right CTB column (tile boundary + §7.4.12.4 boundary walk
interaction); `tiles_2x1_wpp_256x256` interleaves `end_of_subset` and
`end_of_tile` subsets in one slice. The `*_noxlf_*` axes close the
loop filters at tile boundaries
(`pps_loop_filter_across_tiles_enabled_flag = 0`): deblocking skips
the boundary edges (§8.8.3.1), SAO forces edgeIdx = 0 on cross-tile
neighbour samples (§8.8.4.2), and ALF pads its classification /
filter / CC-ALF fetches at the tile rectangle (§8.8.5.5 / §8.8.5.6);
`tiles_3x1_noxlf_qp45` keeps the deep-QP long deblocking filters
active everywhere except the two tile columns.
`tiles_2x2_raster_256x256` re-signals the 2x2 grid as a raster-scan
slice layout (`pps_rect_slice_flag = 0`, `sh_slice_address` +
`sh_num_tiles_in_slice_minus1` on the wire) — its plane hash equals
`tiles_2x2_256x256`'s because only the layout signalling differs,
pinning that both arms resolve the identical §6.5.1 CTB plan. `wpp_256x256` reconstructs
identically to `multi_ctu_256x256` (matching plane hash) because the
DC-only intra pipeline never reads a reference beyond the §6.4.4 WPP
column cap — the axis therefore validates the WPP wire structure
(subsets, entry points, context storage/sync) rather than the cap's
pixel effect; angular/inter content will exercise the cap once the
encoder can emit it.

r431 palette notes: `palette_screen_128x128` palette-codes every CU
losslessly (≤ 16 colours per 64x64 CU) with predictor reuse across
the four CUs of each 128 CTB and across CTBs;
`palette_mixed_256x128` interleaves palette and transform CUs
(`pred_mode_plt_flag` 1/0 on the wire, palette-side deblock
suppression live on the vertical seam); `palette_escape_64x64_qp30`
carries 34 distinct colours per CU — 31 table entries + EG5 escape
samples quantized at QP 30 (the decoded plane contains two
non-source values, pinning eq. 442 dequant on both sides);
`palette_tiles_2x2_256x256` resets the predictor palette at every
tile start and `palette_wpp_256x256` runs the §9.3.2.6/§9.3.2.7
predictor storage/sync — their plane hashes are EQUAL (the palette
content is lossless, so only the wire structure differs), pinning
that the reset/sync arms produce the identical reconstruction.

r431 multi-slice notes: `slices_rect_2x2_256x256` codes one
rectangular slice per tile (four VCL NALs, per-slice §9.3.2 CABAC
initialisation, `sh_slice_address` on the wire) — its plane hash
equals `tiles_2x2_256x256`'s, pinning that only the slice
structuring differs from the single-slice tile layout;
`slices_rect_3x1_384x128_qp34` matches `tiles_3x1_384x128_qp34` the
same way (deblocking crosses the slice boundaries with
`pps_loop_filter_across_slices_enabled_flag = 1`).
`slices_raster_2_256x256` re-signals the 2x2 grid as a raster layout
with TWO slices of two tiles each (`sh_slice_address` +
`sh_num_tiles_in_slice_minus1` per slice).
`slices_rect_noxlf_2x2_256x256` closes the SLICE boundaries while
across-tiles stays enabled — its plane hash equals
`tiles_2x2_noxlf_256x256`'s, pinning that the slice-map-derived
filter gating reproduces the tile gating on the same geometry.
`slices_raster_noxlf_3rows_128x384_qp45` keeps the deep-QP long
deblocking filters active everywhere except the two slice-row
boundaries. `slices_rect_wpp_128x384` runs WPP subsets inside each
slice with slice-local entry-point lists.
`palette_slices_rect_2x2_256x256` resets the predictor palette at
every slice start (plane hash equal to the palette tile/WPP axes —
lossless content, only the wire structure differs).

No known external divergence remains in the corpus.
