//! Bounded mutation smoke: decode random byte-level corruptions of ONE
//! staged conformance stream and report every panic. A robustness
//! aid for freshly generalised parse paths (r456: subpictures, the
//! 4:2:2 / 4:4:4 chroma formats) — `Err` results are the expected
//! outcome of a corrupted stream, panics and hangs are the findings.
//!
//! ```text
//! mutate_smoke <stream-name> [iterations] [seed]
//! ```
//!
//! * `H266_CONFORMANCE_DIR` — corpus dir override (defaults to the
//!   staged `docs/video/h266/conformance/streams`).
//! * `H266_MUTATE_MAX_BYTES` — cap the stream length fed to the decoder
//!   (default 1 MiB) so one iteration stays bounded.

use oxideav_h266::stream::StreamDecoder;

struct XorShift(u64);

impl XorShift {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
}

fn main() {
    let name = std::env::args().nth(1).expect("stream name");
    let iterations: usize = std::env::args()
        .nth(2)
        .map(|s| s.parse().expect("iterations"))
        .unwrap_or(50);
    let seed: u64 = std::env::args()
        .nth(3)
        .map(|s| s.parse().expect("seed"))
        .unwrap_or(0x9E37_79B9_7F4A_7C15);
    let dir = std::env::var("H266_CONFORMANCE_DIR").unwrap_or_else(|_| {
        format!(
            "{}/../../docs/video/h266/conformance/streams",
            env!("CARGO_MANIFEST_DIR")
        )
    });
    let max_bytes: usize = std::env::var("H266_MUTATE_MAX_BYTES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1 << 20);
    let mut base = std::fs::read(format!("{dir}/{name}.bit")).unwrap();
    base.truncate(max_bytes);
    let mut rng = XorShift(seed | 1);
    let mut panics = 0usize;
    let mut errs = 0usize;
    let mut oks = 0usize;
    let started = std::time::Instant::now();
    for it in 0..iterations {
        let mut bs = base.clone();
        // 1..=8 mutations per iteration: byte xor, byte set, or a
        // short truncation — the parse-desync shapes a fuzzer finds
        // first.
        let n = 1 + (rng.next() % 8) as usize;
        for _ in 0..n {
            if bs.is_empty() {
                break;
            }
            let pos = (rng.next() as usize) % bs.len();
            match rng.next() % 3 {
                0 => bs[pos] ^= 1 << (rng.next() % 8),
                1 => bs[pos] = rng.next() as u8,
                _ => bs.truncate(pos.max(64)),
            }
        }
        let result = std::panic::catch_unwind(|| {
            let mut dec = StreamDecoder::new();
            let mut pics = 0usize;
            let r = dec.decode_annex_b(&bs, &mut |_pic| {
                pics += 1;
            });
            (r.is_ok(), pics)
        });
        match result {
            Ok((true, pics)) => {
                oks += 1;
                eprintln!("iter {it}: ok ({pics} pictures)");
            }
            Ok((false, pics)) => {
                errs += 1;
                eprintln!("iter {it}: err after {pics} pictures");
            }
            Err(_) => {
                panics += 1;
                eprintln!("iter {it}: PANIC (seed {seed}, mutations {n})");
            }
        }
    }
    eprintln!(
        "{name}: {iterations} iterations in {:.1}s — {oks} ok / {errs} err / {panics} panic",
        started.elapsed().as_secs_f64()
    );
    if panics > 0 {
        std::process::exit(1);
    }
}
