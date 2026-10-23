//! r477 — the JVET conformance corpus THROUGH THE REGISTRY: every
//! staged stream is split into access units (§7.4.2.4.3), each fed as
//! one `Packet` to the `oxideav_core::Decoder` the registry hands
//! out, and the frames it emits — in its own output order, with its
//! own pixel-format labels and plane packing — must match the `.opl`
//! per-picture plane MD5s row for row (the rows are in output
//! order). A second leg hands the whole stream as a single packet.
//! The harness is a no-op without the corpus (CI machines do not
//! carry the docs tree): `$H266_CONFORMANCE_DIR`, else
//! `../../docs/video/h266/conformance/streams`.

use std::path::{Path, PathBuf};

use oxideav_core::{CodecId, CodecParameters, Error, Frame, Packet, PixelFormat, TimeBase};
use oxideav_h266::decoder::split_access_units;

#[path = "common/md5.rs"]
mod md5;

fn corpus_dir() -> Option<PathBuf> {
    if let Ok(d) = std::env::var("H266_CONFORMANCE_DIR") {
        let p = PathBuf::from(d);
        return p.is_dir().then_some(p);
    }
    let p =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../docs/video/h266/conformance/streams");
    p.is_dir().then_some(p)
}

fn stream_names(dir: &Path) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .unwrap()
        .filter_map(|e| {
            let p = e.unwrap().path();
            (p.extension().map(|x| x == "bit") == Some(true))
                .then(|| p.file_stem().unwrap().to_string_lossy().to_string())
        })
        .collect();
    names.sort();
    names
}

/// `.opl` rows: `(poc, [md5 per plane])` in output order.
fn parse_opl(path: &Path) -> Vec<(i32, Vec<String>)> {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let cols: Vec<&str> = l.split(',').map(str::trim).collect();
            (
                cols[1].parse().unwrap(),
                cols[4..].iter().map(|s| s.to_lowercase()).collect(),
            )
        })
        .collect()
}

/// One output frame: `(pts, label, [plane md5])`.
type OutputFrame = (Option<i64>, Option<PixelFormat>, Vec<String>);

/// Registry decoder over the given packets; returns every output
/// frame in output order.
fn decode_packets(packets: Vec<Vec<u8>>) -> Result<Vec<OutputFrame>, Error> {
    let mut ctx = oxideav_core::RuntimeContext::new();
    oxideav_h266::register(&mut ctx);
    let params = CodecParameters::video(CodecId::new("h266"));
    let mut dec = ctx.codecs.first_decoder(&params)?;
    let mut out = Vec::new();
    let drain = |dec: &mut Box<dyn oxideav_core::Decoder>, out: &mut Vec<_>| loop {
        match dec.receive_frame() {
            Ok(Frame::Video(v)) => {
                let hashes: Vec<String> = v
                    .image_planes()
                    .iter()
                    .map(|p| md5::md5_hex(&p.data))
                    .collect();
                out.push((v.pts, dec.output_pixel_format(), hashes));
            }
            Ok(_) => panic!("non-video frame"),
            Err(Error::NeedMore) | Err(Error::Eof) => break,
            Err(e) => panic!("receive_frame: {e}"),
        }
    };
    for (i, data) in packets.into_iter().enumerate() {
        dec.send_packet(&Packet::new(0, TimeBase::new(1, 1), data).with_pts(i as i64))?;
        drain(&mut dec, &mut out);
    }
    dec.flush()?;
    drain(&mut dec, &mut out);
    Ok(out)
}

fn check_stream(dir: &Path, name: &str, whole: bool) -> Result<(), String> {
    let bs = std::fs::read(dir.join(format!("{name}.bit"))).unwrap();
    let opl = parse_opl(&dir.join(format!("{name}.opl")));
    let packets: Vec<Vec<u8>> = if whole {
        vec![bs.clone()]
    } else {
        split_access_units(&bs)
            .iter()
            .map(|au| au.to_vec())
            .collect()
    };
    let n_packets = packets.len();
    let frames = decode_packets(packets).map_err(|e| format!("decode error: {e}"))?;
    if frames.len() != opl.len() {
        return Err(format!(
            "{} output frames != {} .opl rows ({n_packets} packets)",
            frames.len(),
            opl.len()
        ));
    }
    let mut last_pts = None;
    for (i, ((pts, label, hashes), (poc, want))) in frames.iter().zip(&opl).enumerate() {
        if hashes != want {
            return Err(format!(
                "output frame {i} (opl poc {poc}) plane hashes differ: got {hashes:?} want {want:?}"
            ));
        }
        let Some(pf) = label else {
            return Err(format!("output frame {i}: no pixel-format label"));
        };
        let planes = if matches!(
            pf,
            PixelFormat::Gray8
                | PixelFormat::Gray10Le
                | PixelFormat::Gray12Le
                | PixelFormat::Gray16Le
        ) {
            1
        } else {
            3
        };
        if hashes.len() != planes {
            return Err(format!(
                "output frame {i}: label {pf:?} but {} planes",
                hashes.len()
            ));
        }
        // Packet pts ride through in output order: every packet is
        // one access unit, so pts must be strictly increasing when
        // the stream's output order equals its decode order, and
        // never repeat otherwise.
        if let (Some(p), Some(l)) = (pts, last_pts)
            && *p == l
        {
            return Err(format!("output frame {i}: repeated pts {p}"));
        }
        last_pts = *pts;
    }
    Ok(())
}

/// Every corpus stream, one packet per access unit.
#[test]
fn corpus_through_the_registry_per_access_unit() {
    let Some(dir) = corpus_dir() else {
        eprintln!("jvet corpus not present — skipping");
        return;
    };
    let mut failures = Vec::new();
    let names = stream_names(&dir);
    for name in &names {
        let t = std::time::Instant::now();
        match check_stream(&dir, name, false) {
            Ok(()) => println!("{name}: registry PASS ({:.1}s)", t.elapsed().as_secs_f64()),
            Err(e) => {
                println!("{name}: registry FAIL — {e}");
                failures.push(format!("{name}: {e}"));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {} streams failed through the registry:\n{}",
        failures.len(),
        names.len(),
        failures.join("\n")
    );
}

/// A sample of streams as one whole-stream packet each (multi-AU
/// packets must split internally).
#[test]
fn corpus_through_the_registry_whole_stream_packets() {
    let Some(dir) = corpus_dir() else {
        eprintln!("jvet corpus not present — skipping");
        return;
    };
    let names = stream_names(&dir);
    let mut failures = Vec::new();
    for name in names.iter().step_by(7) {
        match check_stream(&dir, name, true) {
            Ok(()) => println!("{name}: whole-packet PASS"),
            Err(e) => failures.push(format!("{name}: {e}")),
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// The registry descriptor is truthful about what the decoder does.
#[test]
fn registry_descriptor_is_truthful() {
    let mut ctx = oxideav_core::RuntimeContext::new();
    oxideav_h266::register(&mut ctx);
    let id = CodecId::new("h266");
    assert!(ctx.codecs.has_decoder(&id));
    let imp = &ctx.codecs.implementations(&id)[0];
    let caps = &imp.caps;
    assert!(caps.decode);
    assert!(caps.lossy && caps.lossless);
    assert_eq!(caps.max_width, Some(16384));
    assert_eq!(caps.max_height, Some(16384));
    assert!(!caps.intra_only);
}
