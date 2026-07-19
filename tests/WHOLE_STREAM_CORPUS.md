# Whole-stream decode conformance corpus (r412, externally validated r415/r418)

The corpus streams are generated deterministically by
`tests/whole_stream_conformance.rs` — every test encodes with the
crate's IDR pipeline and asserts the crate's own full receive path
(NAL walk → parameter-set / PH / SH parsers → CABAC CTU walker with
in-stream SAO + ALF prefixes → §8.7 dequant + inverse transforms →
§8.8 in-loop filters) reproduces the encoder reconstruction
**byte-exactly**. All 11 axes hold.

`tests/external_probe_corpus.rs` extends the corpus with ~60
single-feature probe streams (TB-size sweep 8..64, single luma/chroma
coefficients by frequency position, amplitude sweeps, chroma-only
planes, gradient / stripe / checker content, QP sweep, 128x128
full-CTB four-CU walk) for black-box bisection against an external
reference decoder.

## Generation

```
H266_CORPUS_DIR=<dir> cargo test --test whole_stream_conformance
H266_CORPUS_DIR=<dir> cargo test --test external_probe_corpus
```

writes `<name>.266` (Annex-B) and `<name>.yuv` (decoded planar 4:2:0,
Y then Cb then Cr) per axis. Content is fully deterministic; the
SHA-256 prefixes below were recorded on 2026-07-20 (they change
whenever the encoder's wire evolves — regenerate rather than diff).

## Black-box reference-decoder validation (ffmpeg 8.1 `vvc` decoder)

```
ffmpeg -i <name>.266 -f rawvideo -pix_fmt yuv420p <name>.ffmpeg.yuv
cmp <name>.yuv <name>.ffmpeg.yuv
```

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

| axis | vs own decoder | vs ffmpeg vvc | stream sha | plane sha |
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

No known external divergence remains in the corpus.
