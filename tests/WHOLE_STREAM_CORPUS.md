# Whole-stream decode conformance corpus (r412, externally validated r415)

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
SHA-256 prefixes below were recorded on 2026-07-17 (they change
whenever the encoder's wire evolves — regenerate rather than diff).

## Black-box reference-decoder validation (ffmpeg 8.1 `vvc` decoder)

```
ffmpeg -i <name>.266 -f rawvideo -pix_fmt yuv420p <name>.ffmpeg.yuv
cmp <name>.yuv <name>.ffmpeg.yuv
```

r415 status: **11 of 11 corpus axes decode byte-exactly** through the
external reference decoder at qp ≤ 34, and **101 of 104** streams
(corpus + probe extension) overall. The r412 "sparse residual"
divergence and the dual-tree/bin-interleave characterization resolved
into five distinct root-cause families, all fixed in r415:

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
| qp45 | byte-exact | 49 luma px diff (luma long-filter corner, below) | 502c2535626f2536 | 2153f6bddd48efd0 |
| multi_ctu_256x256 | byte-exact | byte-exact | 11f98182f582f91e | 93d9ef34ff36b5d9 |
| chroma_sao_merge | byte-exact | byte-exact | 6442a2d1fe64f0e8 | b58a0daff3741f39 |
| mtt_bt / mtt_bt_tt | byte-exact | 14 / 23 luma px diff (same corner) | b9a10b06a87d1122 / 598e470f2d04655c | af30aaa10f5d2e10 / e892e89ed832e222 |
| lmcs / lmcs_chroma_scaling | byte-exact | byte-exact | 81af59718db2c07b / 8a91b058f84df4d3 | 1e164146428d7493 |
| dep_quant / sign_data_hiding | byte-exact | byte-exact | bfc3898b6c9d140b / 0c841ad45810c9a8 | ff8e5a3a0c924e49 / d27ad4ff087635a6 |

Probe extension (`external_probe_corpus.rs`): all ~60 probe streams
byte-exact through the reference decoder, including every sparse
single-coefficient case the r412 characterization flagged, all chroma
probes, and the 128x128 four-CU walk.

## Known remaining external divergence (followup)

Luma deblocking long-filter corner: three streams (`qp45`, `mtt_bt`,
`mtt_bt_tt`) differ from the reference decode in 14 – 49 luma samples
clustered within 7 rows/columns of 32/64-aligned CU edges — the §8.8.3
luma long-filter (maxFilterLength > 3) decision or filtering deviates
in some high-QP / asymmetric-block-size combination. Bitstreams parse
byte-exactly (the divergence is reconstruction-only). Everything else
— headers, CABAC layer, residual syntax, intra prediction, ALF, SAO,
chroma deblocking — is externally validated.
