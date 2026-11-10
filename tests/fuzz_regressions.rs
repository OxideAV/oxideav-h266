//! Replays of the inputs the scheduled fuzz run crashed on, kept as
//! permanent regressions. Each file under
//! `tests/data/fuzz_regressions/<target>/` is one libFuzzer artifact;
//! the drivers below repeat what the matching `fuzz/fuzz_targets`
//! entry does with it. Every input must come back as `Ok` / `Err` —
//! never a panic.
//!
//! r487 — the artifacts cover out-of-range SPS / PPS / picture-header
//! values that reached the decode unchecked: partition-constraint
//! shifts, chroma QP mapping-table pivots outside `−QpBdOffset..=63`,
//! `sps_six_minus_max_num_merge_cand > 5`, a transform-skip size
//! beyond 32, a conformance window larger than the picture and a
//! wraparound offset beyond the picture width.

use std::path::PathBuf;

use oxideav_core::{CodecId, CodecParameters, DecoderLimits, Error, Packet, TimeBase};

fn artifacts(target: &str) -> Vec<(String, Vec<u8>)> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/fuzz_regressions")
        .join(target);
    let mut out: Vec<(String, Vec<u8>)> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
        .map(|e| {
            let p = e.unwrap().path();
            (
                p.file_name().unwrap().to_string_lossy().into_owned(),
                std::fs::read(&p).unwrap(),
            )
        })
        .collect();
    out.sort();
    assert!(!out.is_empty(), "no artifacts under {}", dir.display());
    out
}

/// The `registry_decode` fuzz entry: `u16` record length, the vvcC
/// record, then the payload as one packet (and its first half again
/// after a reset).
fn drive_registry_decode(data: &[u8]) {
    if data.len() < 2 {
        return;
    }
    let rec_len = usize::from(u16::from_be_bytes([data[0], data[1]]));
    let Some(rec) = data.get(2..2 + rec_len) else {
        return;
    };
    let payload = &data[2 + rec_len..];
    let mut params = CodecParameters::video(CodecId::new("h266"));
    params.extradata = rec.to_vec();
    params.limits = DecoderLimits::default().with_max_pixels_per_frame(512 * 512);
    let Ok(mut dec) = oxideav_h266::decoder::make_decoder(&params) else {
        return;
    };
    let _ = dec.send_packet(&Packet::new(0, TimeBase::new(1, 1), payload.to_vec()));
    while let Ok(_) | Err(Error::NeedMore) = dec.receive_frame() {
        if matches!(dec.receive_frame(), Err(Error::NeedMore)) {
            break;
        }
    }
    let _ = dec.reset();
    let half = &payload[..payload.len() / 2];
    let _ = dec.send_packet(&Packet::new(0, TimeBase::new(1, 1), half.to_vec()));
    let _ = dec.flush();
    while dec.receive_frame().is_ok() {}
}

#[test]
fn registry_decode_artifacts_do_not_panic() {
    for (name, data) in artifacts("registry_decode") {
        eprintln!("registry_decode/{name}");
        drive_registry_decode(&data);
    }
}

/// The parameter-set arm of the `parse_annexb` fuzz entry.
#[test]
fn parse_annexb_artifacts_do_not_panic() {
    use oxideav_h266::nal::{NalUnitType, extract_rbsp, iter_annex_b};
    for (name, data) in artifacts("parse_annexb") {
        eprintln!("parse_annexb/{name}");
        for nal in iter_annex_b(&data) {
            let rbsp = extract_rbsp(nal.payload());
            match nal.header.nal_unit_type {
                NalUnitType::SpsNut => {
                    let _ = oxideav_h266::sps::parse_sps(&rbsp);
                }
                NalUnitType::PpsNut => {
                    let _ = oxideav_h266::pps::parse_pps(&rbsp);
                }
                NalUnitType::VpsNut => {
                    let _ = oxideav_h266::vps::parse_vps(&rbsp);
                }
                NalUnitType::PrefixApsNut | NalUnitType::SuffixApsNut => {
                    let _ = oxideav_h266::aps::parse_aps(&rbsp);
                }
                NalUnitType::PrefixSeiNut | NalUnitType::SuffixSeiNut => {
                    let _ = oxideav_h266::sei_rbsp::parse_sei_rbsp(&rbsp);
                }
                _ => {}
            }
        }
    }
}
