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
    flags), **PPS** (§7.3.2.4).
  * **APS** (§7.3.2.5) — ALF / LMCS / scaling-list type. The LMCS APS
    payload (§7.3.2.19) is decoded into a typed `lmcs::LmcsData`, with
    the BitDepth-dependent §7.4.3.19 derivations (`OrgCW`, `lmcsCW`,
    `LmcsPivot`, `ScaleCoeff` / `InvScaleCoeff`, `ChromaScaleCoeff`) and
    the sample-domain forward / inverse luma mapping + chroma residual
    scaling folds.
* **Auxiliary NAL units** — AUD (§7.3.2.10), Filler Data (§7.3.2.13),
  End of Sequence / End of Bitstream (§7.3.2.11 / §7.3.2.12),
  `rbsp_trailing_bits()` / `byte_alignment()` / `rbsp_slice_trailing_bits()`
  validators, and the SEI message + `sei_rbsp()` walkers. The first
  per-type `sei_payload()` Annex D body is decoded — the SEI manifest
  SEI message (§D.8.1, `payloadType == 200`) into a typed
  `sei_manifest::SeiManifest`; the remaining payload bodies (most
  deferred to Rec. ITU-T H.274) are still uninterpreted.
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
  runs through `u16` picture planes.
* **AMVP** — the §8.5.2.8-10 AMVP candidate derivation (spatial scan,
  HMVP fill, temporal Col, zero-MV pad), the §8.5.5.7 affine AMVP
  candidate list, and AMVR helpers (§7.4.11.6). The non-merge inter CU
  reconstruction (residual `transform_tree`) and the CTU-walker fuse
  that drives these from the live inter path are still being wired and
  currently surface `Error::Unsupported`.
* **Transforms** — DCT-II inverse (sizes 2..=64), DST-VII / DCT-VIII
  (4 / 8 / 16), flat-list dequant, the §8.7.2 scaling-and-transformation
  orchestrator with joint Cb-Cr derivation, and the §8.7.4.6 inverse
  adaptive colour transform.
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
