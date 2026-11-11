//! r487 — the decoded picture hash SEI message (Rec. ITU-T H.274
//! §8.8) as a conformance oracle: every picture of every staged
//! conformance stream that carries the message must match it — MD5,
//! CRC or checksum, per colour component, over the whole decoded
//! picture. This is independent of the `.opl` sidecars the other
//! harnesses compare against: the expected values travel inside the
//! bitstream.
//!
//! The harness is a no-op without the corpus (CI machines do not
//! carry the docs tree): `$H266_CONFORMANCE_DIR`, else
//! `../../docs/video/h266/conformance/streams`.

use std::collections::BTreeMap;
use std::path::PathBuf;

use oxideav_h266::picture_hash::{PictureHashMethod, PictureHashStatus};
use oxideav_h266::sei_payload::SeiPayload;
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

/// The staged corpus: pictures, pictures carrying a hash message, and
/// which of the three methods the streams use. Updated when the
/// corpus changes.
const EXPECTED_STREAMS: usize = 56;

#[test]
fn every_signalled_picture_hash_matches() {
    let Some(dir) = corpus_dir() else {
        eprintln!("jvet corpus not present — skipping");
        return;
    };
    let mut names: Vec<String> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| {
            let p = e.unwrap().path();
            (p.extension().map(|x| x == "bit") == Some(true))
                .then(|| p.file_stem().unwrap().to_string_lossy().to_string())
        })
        .collect();
    names.sort();
    assert_eq!(names.len(), EXPECTED_STREAMS);

    let mut streams_with_hash = 0usize;
    let mut streams_all_pictures_hashed = 0usize;
    let mut total_pictures = 0usize;
    let mut total_checked = 0usize;
    let mut by_method: BTreeMap<String, usize> = BTreeMap::new();
    let mut failures = Vec::new();
    for name in &names {
        let bs = std::fs::read(dir.join(format!("{name}.bit"))).unwrap();
        let mut dec = StreamDecoder::new();
        dec.set_verify_picture_hash(true);
        let mut pictures = 0usize;
        let mut checked = 0usize;
        let mut with_message = 0usize;
        let mut method: Option<PictureHashMethod> = None;
        let mut sink = |pic: DecodedPicture| {
            pictures += 1;
            if pic
                .sei
                .iter()
                .any(|m| matches!(m, SeiPayload::DecodedPictureHash(_)))
            {
                with_message += 1;
            }
            match pic.hash_status {
                Some(PictureHashStatus::Match(m)) => {
                    checked += 1;
                    method = Some(m);
                }
                Some(other) => failures.push(format!("{name} poc {}: {other:?}", pic.poc)),
                None => {}
            }
        };
        if let Err(e) = dec.decode_annex_b(&bs, &mut sink) {
            failures.push(format!("{name}: decode error {e}"));
        }
        // A message a decoder may not ignore must have been checked.
        if with_message != checked {
            failures.push(format!(
                "{name}: {with_message} pictures carry a hash message, {checked} verified"
            ));
        }
        println!("{name}: {pictures} pictures, {checked} hash-verified ({method:?})");
        total_pictures += pictures;
        total_checked += checked;
        if checked > 0 {
            streams_with_hash += 1;
            *by_method
                .entry(format!("{:?}", method.unwrap()))
                .or_default() += 1;
        }
        if checked == pictures && pictures > 0 {
            streams_all_pictures_hashed += 1;
        }
    }
    println!(
        "== picture hash oracle: {streams_with_hash}/{} streams carry the message \
         ({streams_all_pictures_hashed} on every picture), {total_checked}/{total_pictures} \
         pictures verified, methods {by_method:?}",
        names.len()
    );
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    assert!(streams_with_hash > 0, "no stream carried a picture hash");
}
