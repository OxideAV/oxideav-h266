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

## Decode support

**None.** Reconstruction (CTU walker, intra prediction, transforms,
deblock, ALF, LMCS, SAO-replacement, reference picture lists, etc.)
is out of scope for this foundation pass. The crate's `Decoder`
registration returns `Error::Unsupported` for any packet that would
require pixel output.

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
