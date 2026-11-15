//! r490 — the decoder's worker budget on real streams.
//!
//! With a budget above one worker the three colour components are
//! deblocked and SAO-filtered side by side and the adaptive loop
//! filter runs its CTB rows in parallel. The in-stream decoded picture
//! hash is the oracle: every picture of a set of staged JVET
//! conformance streams (tiles, slices, subpictures, virtual
//! boundaries, 4:2:2 / 4:4:4 / 4:0:0, dual tree) must verify at every
//! budget. A no-op without the workspace `docs/` tree.

use std::path::PathBuf;

use oxideav_h266::picture_hash::PictureHashStatus;
use oxideav_h266::stream::{DecodedPicture, StreamDecoder};

fn corpus_dir() -> Option<PathBuf> {
    if let Ok(d) = std::env::var("H266_CONFORMANCE_DIR") {
        let p = PathBuf::from(d);
        return p.is_dir().then_some(p);
    }
    let p =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../docs/video/h266/conformance/streams");
    p.is_dir().then_some(p)
}

/// Streams chosen for their loop-filter layouts, all small enough for
/// a debug-build run.
const STREAMS: [&str; 10] = [
    "ALF_C_3",
    "CST_A_4",
    "CodingToolsSets_E_1",
    "DEBLOCKING_E_3",
    "GDR_A_2",
    "SUBPIC_C_1",
    "WP_A_3",
    "10b422_B_5",
    "JCCR_A_2",
    "CCLM_A_2",
];

#[test]
fn picture_hashes_verify_at_every_worker_budget() {
    let Some(dir) = corpus_dir() else {
        eprintln!("jvet corpus not present — skipping");
        return;
    };
    for name in STREAMS {
        let Ok(bs) = std::fs::read(dir.join(format!("{name}.bit"))) else {
            eprintln!("{name}: not staged — skipping");
            continue;
        };
        let mut reference: Option<Vec<(i32, Vec<u16>)>> = None;
        for threads in [1usize, 2, 3, 6] {
            let mut dec = StreamDecoder::new();
            dec.set_threads(threads);
            dec.set_verify_picture_hash(true);
            let mut pictures = Vec::new();
            let mut verified = 0usize;
            let mut sink = |pic: DecodedPicture| {
                assert!(
                    !matches!(pic.hash_status, Some(PictureHashStatus::Mismatch { .. })),
                    "{name} poc {} with {threads} workers: {:?}",
                    pic.poc,
                    pic.hash_status
                );
                if matches!(pic.hash_status, Some(PictureHashStatus::Match(_))) {
                    verified += 1;
                }
                let mut samples = pic.frame.luma.samples.clone();
                samples.extend_from_slice(&pic.frame.cb.samples);
                samples.extend_from_slice(&pic.frame.cr.samples);
                pictures.push((pic.poc, samples));
            };
            dec.decode_annex_b(&bs, &mut sink)
                .unwrap_or_else(|e| panic!("{name} with {threads} workers: {e}"));
            assert_eq!(
                verified,
                pictures.len(),
                "{name} with {threads} workers: every picture carries a hash"
            );
            match &reference {
                None => reference = Some(pictures),
                Some(serial) => assert!(
                    *serial == pictures,
                    "{name}: {threads} workers changed the output"
                ),
            }
        }
    }
}
