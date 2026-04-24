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
//! * [`dci`] — Decoding Capability Information parser (§7.3.2.1).
//! * [`opi`] — Operating Point Information parser (§7.3.2.2).
//! * [`ctu`] — CTU walker scaffold: picture → CTU-address scan, per-CTU
//!   geometry + neighbour availability, slice-scope CABAC init, and
//!   coding_tree dispatch that surfaces `Error::Unsupported` for any
//!   construct below the partition tree (reconstruction, in-loop filters).
//! * [`leaf_cu`] — per-CU syntax reads from `coding_unit()` (§7.3.11.5)
//!   plus the §8.4.2 / §8.4.3 luma / chroma intra-mode derivations, now
//!   extended with the `transform_unit()` CBF + `cu_qp_delta` reads and
//!   a single-TB residual walker driver. The module is still "parse +
//!   derive, don't reconstruct": it captures the MPM cascade, MIP flags,
//!   ISP split selector, chroma mode, CBFs, quantised levels and
//!   `LastSignificantCoeff{X,Y}` in a [`leaf_cu::LeafCuInfo`] +
//!   [`leaf_cu::LeafCuResidual`] pair for later reconstruction rounds.
//! * [`residual`] — `tu_y/cb/cr_coded_flag`, `cu_qp_delta_abs`,
//!   `cu_chroma_qp_offset_*`, `last_sig_coeff_{x,y}_{prefix,suffix}`
//!   and the per-sub-block `sb_coded_flag` / `sig_coeff_flag` /
//!   `par_level_flag` / `abs_level_gtx_flag` / `coeff_sign_flag` /
//!   `abs_remainder` reads (§7.3.11.10 / §7.3.11.11). Drives a
//!   coefficient-level array that later rounds will feed into dequant
//!   + inverse transform.
//!
//! ## Spec reference
//!
//! All section numbers in this crate refer to **ITU-T H.266 | ISO/IEC
//! 23090-3 (V4, 01/2026)**. The implementation is spec-only; no
//! third-party VVC decoder source was consulted.

#![allow(clippy::too_many_arguments)]
// The parser-foundation modules (landed in earlier rounds) carry a
// handful of clippy nits — `div_ceil` manual implementations, field
// reassign-after-default patterns, explicit lifetime sugar etc. — that
// do not affect correctness. They are tracked for cleanup but are
// suppressed here so the CTU-walker increment gates on its own diff.
#![allow(clippy::manual_div_ceil)]
#![allow(clippy::field_reassign_with_default)]
#![allow(clippy::needless_lifetimes)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::if_same_then_else)]
#![allow(clippy::manual_memcpy)]
#![allow(clippy::collapsible_if)]
#![allow(clippy::needless_bool)]
#![allow(clippy::doc_lazy_continuation)]
#![allow(clippy::derivable_impls)]
#![allow(clippy::identity_op)]
#![allow(clippy::erasing_op)]
#![allow(clippy::same_item_push)]

pub mod aps;
pub mod bitreader;
pub mod cabac;
pub mod coding_tree;
pub mod ctu;
pub mod ctx;
pub mod dci;
pub mod decoder;
pub mod hrd;
pub mod intra;
pub mod leaf_cu;
pub mod nal;
pub mod opi;
pub mod picture_header;
pub mod pps;
pub mod ptl;
pub mod reconstruct;
pub mod ref_pic_list;
pub mod residual;
pub mod scan;
pub mod slice_header;
pub mod sps;
pub mod tables;
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
