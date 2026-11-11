//! Decode / encode throughput probe (`harness = false`; plain
//! `std::time`, no bench framework).
//!
//! ```text
//! cargo bench --bench throughput -- decode [filter]   # staged conformance streams
//! cargo bench --bench throughput -- encode            # synthetic stills
//! ```
//!
//! * `decode` runs every `.bit` stream under `$H266_CONFORMANCE_DIR`
//!   (default `../../docs/video/h266/conformance/streams`; a no-op
//!   when absent) through the registry decoder, one packet per access
//!   unit, and prints pictures, median wall time and pictures per
//!   second per stream plus the corpus total. `[filter]` keeps the
//!   streams whose name contains it.
//! * `encode` codes synthetic 4:2:0 stills (lossy at QP 32 and
//!   lossless) through the registry encoder and prints the median
//!   wall time and megapixels per second.
//!
//! Environment: `H266_BENCH_THREADS` (default 1) is handed to the
//! codec as its `ExecutionContext` worker budget; `H266_BENCH_REPS`
//! (default 3) is the number of timed repetitions the median is taken
//! over. Peak memory is read from outside (`/usr/bin/time -l` or
//! `-v` around the bench binary).

use std::path::PathBuf;
use std::time::{Duration, Instant};

use oxideav_core::{
    CodecId, CodecOptions, CodecParameters, Error, ExecutionContext, Frame, Packet, PixelFormat,
    TimeBase, VideoFrame, VideoPlane,
};

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn median(mut v: Vec<Duration>) -> Duration {
    v.sort();
    v[v.len() / 2]
}

fn corpus_dir() -> Option<PathBuf> {
    if let Ok(d) = std::env::var("H266_CONFORMANCE_DIR") {
        let p = PathBuf::from(d);
        return p.is_dir().then_some(p);
    }
    let p =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../docs/video/h266/conformance/streams");
    p.is_dir().then_some(p)
}

/// Decode one stream; returns (pictures, output bytes, FNV-1a of the
/// output planes — a cheap identity check between runs).
fn decode_stream(bs: &[u8], threads: usize) -> (usize, usize, u64) {
    let params = CodecParameters::video(CodecId::new("h266"));
    let mut dec = oxideav_h266::decoder::make_decoder(&params).expect("decoder");
    dec.set_execution_context(&ExecutionContext::with_threads(threads));
    let mut pictures = 0usize;
    let mut bytes = 0usize;
    let mut fnv = 0xcbf2_9ce4_8422_2325u64;
    let mut drain = |dec: &mut dyn oxideav_core::Decoder| {
        loop {
            match dec.receive_frame() {
                Ok(Frame::Video(v)) => {
                    pictures += 1;
                    for p in &v.planes {
                        bytes += p.data.len();
                        for chunk in p.data.chunks(4096) {
                            fnv = (fnv ^ u64::from(chunk[0]) ^ ((chunk.len() as u64) << 8))
                                .wrapping_mul(0x0000_0100_0000_01b3);
                            fnv = (fnv ^ u64::from(chunk[chunk.len() - 1]))
                                .wrapping_mul(0x0000_0100_0000_01b3);
                        }
                    }
                }
                Ok(_) => {}
                Err(Error::NeedMore) | Err(Error::Eof) => break,
                Err(e) => panic!("{e}"),
            }
        }
    };
    for au in oxideav_h266::decoder::split_access_units(bs) {
        dec.send_packet(&Packet::new(0, TimeBase::new(1, 1), au.to_vec()))
            .expect("send_packet");
        drain(dec.as_mut());
    }
    dec.flush().expect("flush");
    drain(dec.as_mut());
    (pictures, bytes, fnv)
}

fn bench_decode(filter: Option<&str>, threads: usize, reps: usize) {
    let Some(dir) = corpus_dir() else {
        println!("conformance corpus not present — nothing to decode");
        return;
    };
    let mut names: Vec<String> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| {
            let p = e.unwrap().path();
            (p.extension().map(|x| x == "bit") == Some(true))
                .then(|| p.file_stem().unwrap().to_string_lossy().to_string())
        })
        .filter(|n| filter.is_none_or(|f| n.contains(f)))
        .collect();
    names.sort();
    println!(
        "decode: {} streams, {threads} worker(s), median of {reps}",
        names.len()
    );
    println!("| stream | pictures | ms | pictures/s | output id |");
    let (mut total_pics, mut total_time) = (0usize, Duration::ZERO);
    for name in &names {
        let bs = std::fs::read(dir.join(format!("{name}.bit"))).unwrap();
        let mut times = Vec::new();
        let mut result = (0, 0, 0);
        for _ in 0..reps {
            let t = Instant::now();
            result = decode_stream(&bs, threads);
            times.push(t.elapsed());
        }
        let t = median(times);
        println!(
            "| {name} | {} | {:.1} | {:.1} | {:016x} |",
            result.0,
            t.as_secs_f64() * 1e3,
            result.0 as f64 / t.as_secs_f64(),
            result.2
        );
        total_pics += result.0;
        total_time += t;
    }
    println!(
        "== decode total: {total_pics} pictures in {:.2} s — {:.1} pictures/s ({threads} worker(s))",
        total_time.as_secs_f64(),
        total_pics as f64 / total_time.as_secs_f64()
    );
}

fn synthetic_frame(w: usize, h: usize) -> VideoFrame {
    let mut state = 0x1234_5678u32;
    let mut rnd = move || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        state
    };
    let mut y = Vec::with_capacity(w * h);
    for yy in 0..h {
        for x in 0..w {
            let band = ((x / 23 + yy / 17) % 5) * 22;
            let edge = if x * 3 + yy * 2 > w + h { 40 } else { 0 };
            y.push((40 + band + edge + (x * 60) / w + (rnd() % 14) as usize).min(255) as u8);
        }
    }
    let (cw, ch) = (w / 2, h / 2);
    let mut cb = Vec::with_capacity(cw * ch);
    let mut cr = Vec::with_capacity(cw * ch);
    for yy in 0..ch {
        for x in 0..cw {
            let l = y[(yy * 2) * w + x * 2] as usize;
            cb.push((70 + l / 2 + (rnd() % 5) as usize).min(255) as u8);
            cr.push((190 - l / 3 + (rnd() % 5) as usize).min(255) as u8);
        }
    }
    VideoFrame {
        pts: Some(0),
        planes: vec![
            VideoPlane { stride: w, data: y },
            VideoPlane {
                stride: cw,
                data: cb,
            },
            VideoPlane {
                stride: cw,
                data: cr,
            },
        ],
    }
}

fn bench_encode(threads: usize, reps: usize) {
    println!("encode: {threads} worker(s), median of {reps}");
    println!("| picture | mode | ms | megapixels/s | bytes |");
    for &(w, h) in &[(416usize, 240usize), (1280, 720)] {
        let frame = synthetic_frame(w, h);
        for (mode, opts) in [
            ("QP 32", CodecOptions::new().set("qp", "32")),
            ("lossless", CodecOptions::new().set("lossless", "true")),
        ] {
            let mut p = CodecParameters::video(CodecId::new("h266"));
            p.width = Some(w as u32);
            p.height = Some(h as u32);
            p.pixel_format = Some(PixelFormat::Yuv420P);
            p.options = opts;
            let mut times = Vec::new();
            let mut bytes = 0;
            for _ in 0..reps {
                let mut enc = oxideav_h266::h266_encoder::make_encoder(&p).expect("encoder");
                enc.set_execution_context(&ExecutionContext::with_threads(threads));
                let t = Instant::now();
                enc.send_frame(&Frame::Video(frame.clone()))
                    .expect("send_frame");
                enc.flush().expect("flush");
                bytes = enc.receive_packet().expect("packet").data.len();
                times.push(t.elapsed());
            }
            let t = median(times);
            println!(
                "| {w}x{h} | {mode} | {:.1} | {:.2} | {bytes} |",
                t.as_secs_f64() * 1e3,
                (w * h) as f64 / 1e6 / t.as_secs_f64()
            );
        }
    }
}

fn main() {
    // `cargo bench` passes `--bench`; everything else is ours.
    let args: Vec<String> = std::env::args()
        .skip(1)
        .filter(|a| !a.starts_with("--"))
        .collect();
    let threads = env_usize("H266_BENCH_THREADS", 1);
    let reps = env_usize("H266_BENCH_REPS", 3).max(1);
    match args.first().map(String::as_str) {
        Some("decode") => bench_decode(args.get(1).map(String::as_str), threads, reps),
        Some("encode") => bench_encode(threads, reps),
        _ => {
            bench_decode(None, threads, reps);
            bench_encode(threads, reps);
        }
    }
}
