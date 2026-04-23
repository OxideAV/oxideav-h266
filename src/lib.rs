//! Pure-Rust VVC / H.266 (ITU-T H.266 | ISO/IEC 23090-3) decoder foundation.
//!
//! This crate currently implements the **parser foundation** only. VVC is a
//! 500+ page spec — we bring in NAL framing, RBSP extraction, an Exp-Golomb
//! bit reader, and (incrementally) the VPS / SPS / PPS / APS + picture
//! header + slice header parsers so the workspace has a reliable surface on
//! top of which a CTU walker can be wired up in later increments. Actual
//! CTU decode, intra / inter prediction, transforms, deblock, ALF, LMCS,
//! SAO-replacement, reference-picture-list construction, etc. are **out of
//! scope** for this pass.
//!
//! ## Crate layout
//!
//! * [`bitreader`] — MSB-first reader, `u(n)` / `ue(v)` / `se(v)` helpers
//!   (§9.2).
//! * [`nal`] — start-code scanner, length-prefix iterator, 2-byte NAL
//!   header parsing (§7.3.1.2), emulation-prevention byte stripping.
//! * [`decoder`] — registry factory; returns `Error::Unsupported` for any
//!   packet that would require pixel output (no CTU reconstruction yet).
//!
//! ## Spec reference
//!
//! All section numbers in this crate refer to **ITU-T H.266 | ISO/IEC
//! 23090-3 (V4, 01/2026)**. The implementation is spec-only; no
//! third-party VVC decoder source was consulted.

#![allow(clippy::too_many_arguments)]

pub mod aps;
pub mod bitreader;
pub mod cabac;
pub mod ctx;
pub mod decoder;
pub mod intra;
pub mod nal;
pub mod picture_header;
pub mod pps;
pub mod ptl;
pub mod slice_header;
pub mod sps;
pub mod transform;
pub mod vps;

use oxideav_codec::{CodecInfo, CodecRegistry};
use oxideav_core::{CodecCapabilities, CodecId, CodecTag};

/// Canonical oxideav codec id for H.266 / VVC.
pub const CODEC_ID_STR: &str = "h266";

/// Register the VVC implementation (currently parser-only) with a codec
/// registry. The registered decoder factory returns an unsupported-error
/// decoder — parameter-set inspection lives in the parser modules for now.
pub fn register(reg: &mut CodecRegistry) {
    let caps = CodecCapabilities::video("h266_sw")
        .with_lossy(true)
        .with_intra_only(false)
        .with_max_size(16384, 16384);
    // ISOBMFF sample-description FourCCs defined in ISO/IEC 14496-15:
    //   `vvc1` — in-band parameter sets
    //   `vvi1` — inband parameter sets, VVC inband picture header
    // `VVC1`/`H266` are additional casings / informal aliases.
    reg.register(
        CodecInfo::new(CodecId::new(CODEC_ID_STR))
            .capabilities(caps)
            .decoder(decoder::make_decoder)
            .tags([
                CodecTag::fourcc(b"vvc1"),
                CodecTag::fourcc(b"vvi1"),
                CodecTag::fourcc(b"VVC1"),
                CodecTag::fourcc(b"H266"),
                CodecTag::fourcc(b"h266"),
            ]),
    );
}
