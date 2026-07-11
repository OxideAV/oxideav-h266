# r412 whole-stream decode conformance corpus

The corpus streams are generated deterministically by
`tests/whole_stream_conformance.rs` — every test encodes with the
crate's IDR pipeline and asserts the crate's own full receive path
(NAL walk → parameter-set / PH / SH parsers → CABAC CTU walker with
in-stream SAO + ALF prefixes → §8.8 in-loop filters) reproduces the
encoder reconstruction **byte-exactly**. All 11 axes hold.

## Generation

```
H266_CORPUS_DIR=<dir> cargo test --test whole_stream_conformance
```

writes `<name>.266` (Annex-B) and `<name>.yuv` (decoded planar 4:2:0,
Y then Cb then Cr) per axis. Content is fully deterministic; the
SHA-256 prefixes below were recorded on 2026-07-12 (they change
whenever the encoder's wire evolves — regenerate rather than diff).

## Black-box reference-decoder validation (ffmpeg 8.x `vvc` decoder)

```
ffmpeg -i <name>.266 -f rawvideo -pix_fmt yuv420p <name>.ffmpeg.yuv
cmp <name>.yuv <name>.ffmpeg.yuv
```

Status per axis (stream-sha256[0..16] / plane-sha256[0..16]):

| axis | vs own decoder | vs ffmpeg vvc | stream sha | plane sha |
|------|----------------|---------------|------------|-----------|
| flat_qp26 | byte-exact | **byte-exact** | 8a5e0075be2d168b | 8c8362c09e7c37cf |
| default_qp26 | byte-exact | desync (residual corner, below) | 4835a30e54ed8da6 | 536168d3aec5648a |
| qp10 / qp17 / qp34 / qp45 | byte-exact | desync (same corner) | c3c47b22b34f4668 / b431df682dacc863 / 1670ad7606b820e8 / ec65921bbf4d755d | 9c106bc5181373b4 / 0ef00ae7aaa79f77 / 17d1b6d4b7bd5ab2 / c15b11385a1c3bef |
| multi_ctu_256x256 | byte-exact | desync (same corner) | a62bd6f13c8cdd53 | d502282605e2c649 |
| chroma_sao_merge | byte-exact | desync (same corner) | 388b9b82d724de7d | 4ca03151b67c8344 |
| mtt_bt / mtt_bt_tt | byte-exact | desync (same corner) | 1badd2ec6b7dfb60 / 05c3984b19b0bcf7 | 546df2f9c5afa1ca / bf1e7f8598d1a951 |
| lmcs / lmcs_chroma_scaling | byte-exact | desync (same corner) | bee519c254caa93f / 3f1b23fae6f24cbd | 0e4c888754730831 |
| dep_quant / sign_data_hiding | byte-exact | desync (same corner) | f494761acdc14cb5 / 85c76f91f54f707e | 0c4184a351f6acce / f8fb849444825423 |

Additional ffmpeg-validated **byte-exact** probes (r412, same
pipeline, minimal contents):

* 64x64 flat (all planes), any QP;
* single DC coefficient (luma or chroma bump);
* single non-DC coefficient with small remainder (levels ≤ ~41 at the
  probed positions — e.g. (1,0) level 14 / 27 / 32 / 36 / 41);
* alternating-stripe content (coefficients across a full sub-block
  row, large levels, gt1/par/gt3 + escape-EGk remainders);
* dense diagonal-gradient content (`ladbig`, the corpus pattern at 4×
  amplitude).

## Known remaining external divergence (followup)

Sparse-residual 64x64 luma TBs — including every corpus axis whose
content produces them — decode byte-exactly on the crate's own
receive path but desync ffmpeg's `vvc` decoder at (or attributed to)
the last CTU. The boundary is NOT monotone in coefficient level
(single coefficient at (1,0): levels 32/36/41/54/63/72 pass, 45/50/64/91
fail), the bin STRUCTURE of passing and failing cases is identical
(same pass-1 flags, same escape-EGk shape, equal bin counts), and the
crate's own decoder verifies `end_of_slice_one_bit == 1` with the
spec-required "last bit inserted is 1" property on both sets. Repro:

```
# passes                          # fails
basis (1,0) amp 12..18            basis (1,0) amp 20/22, 40
bump / bump_hf / ladbig           lad1 (corpus gradient, amp 2)
```

Everything above the residual layer is externally validated: NAL /
parameter sets / PH / SH (ffmpeg `trace_headers` field-for-field),
ALF CTU prefix, §7.3.11.4 split presence + ctxIncs, intra-mode
cascade, CBFs, cu_qp_delta, §7.3.11.11 zero-out geometry, terminate +
§9.3.4.3.5 flush.
