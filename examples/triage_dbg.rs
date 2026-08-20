//! Conformance debug driver: decode ONE staged JVET conformance
//! stream by name and print per-picture verdicts against its `.opl`
//! sidecar. Development aid for the corpus triage
//! (`tests/jvet_conformance.rs` is the tracked scorecard).
//!
//! Usage: `cargo run --release --example triage_dbg -- <STREAM_NAME>`
//!
//! * `H266_CONFORMANCE_DIR` — corpus dir override (defaults to the
//!   workspace docs staging path).
//! * `H266_DUMP_DIR` — when set, dump each output picture's cropped
//!   planes (8-bit: 1 byte/sample; >8-bit: 2-byte LE) as
//!   `ours_cvs<K>_poc<N>.yuv` for sample-level diffing against a
//!   reference decode.
//! * Combine with `H266_DBG_IGNORE_EOS` / `H266_DBG_CU` (see
//!   `stream.rs` / `ctu.rs`) to inspect desynced pictures.

use oxideav_h266::stream::StreamDecoder;

fn main() {
    let name = std::env::args().nth(1).expect("stream name");
    let dir = std::env::var("H266_CONFORMANCE_DIR").unwrap_or_else(|_| {
        format!(
            "{}/../../docs/video/h266/conformance/streams",
            env!("CARGO_MANIFEST_DIR")
        )
    });
    let bs = std::fs::read(format!("{dir}/{name}.bit")).unwrap();
    let opl_text = std::fs::read_to_string(format!("{dir}/{name}.opl")).unwrap();
    let opl: Vec<(i32, Vec<String>)> = opl_text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let cols: Vec<&str> = l.split(',').map(str::trim).collect();
            (
                cols[1].parse().unwrap(),
                cols[4..].iter().map(|s| s.to_lowercase()).collect(),
            )
        })
        .collect();

    let mut outs: Vec<(u32, i32, Vec<String>)> = Vec::new();
    let mut dec = StreamDecoder::new();
    let r = dec.decode_annex_b(&bs, &mut |pic| {
        let (cx, cy, cw, ch) = pic.crop;
        let wide = pic.bit_depth > 8;
        let plane_md5 = |p: &oxideav_h266::reconstruct::PicturePlane,
                         x0: usize,
                         y0: usize,
                         w: usize,
                         h: usize| {
            let mut row: Vec<u8> = Vec::new();
            for y in y0..y0 + h {
                for &v in &p.samples[y * p.stride + x0..y * p.stride + x0 + w] {
                    if wide {
                        row.extend_from_slice(&v.to_le_bytes());
                    } else {
                        row.push(v as u8);
                    }
                }
            }
            md5_hex(&row)
        };
        let mut hashes = vec![plane_md5(&pic.frame.luma, cx, cy, cw, ch)];
        if pic.chroma_format_idc != 0 {
            hashes.push(plane_md5(&pic.frame.cb, cx / 2, cy / 2, cw / 2, ch / 2));
            hashes.push(plane_md5(&pic.frame.cr, cx / 2, cy / 2, cw / 2, ch / 2));
        }
        if pic.output_flag {
            outs.push((pic.cvs_idx, pic.poc, hashes));
            if let Ok(dir) = std::env::var("H266_DUMP_DIR") {
                let mut raw: Vec<u8> = Vec::new();
                for (p, x0, y0, w, h) in [
                    (&pic.frame.luma, cx, cy, cw, ch),
                    (&pic.frame.cb, cx / 2, cy / 2, cw / 2, ch / 2),
                    (&pic.frame.cr, cx / 2, cy / 2, cw / 2, ch / 2),
                ] {
                    for y in y0..y0 + h {
                        for &v in &p.samples[y * p.stride + x0..y * p.stride + x0 + w] {
                            if wide {
                                raw.extend_from_slice(&v.to_le_bytes());
                            } else {
                                raw.push(v as u8);
                            }
                        }
                    }
                }
                std::fs::write(
                    format!("{dir}/ours_cvs{}_poc{}.yuv", pic.cvs_idx, pic.poc),
                    raw,
                )
                .unwrap();
            }
        }
        eprintln!("decoded poc {} (output {})", pic.poc, pic.output_flag);
    });
    if let Err(e) = r {
        eprintln!("DECODE ERROR after {} pictures: {e}", outs.len());
    }
    outs.sort_by_key(|(c, p, _)| (*c, *p));
    for (i, (_, poc, hashes)) in outs.iter().enumerate() {
        if i >= opl.len() {
            println!("poc {poc}: EXTRA output picture");
            continue;
        }
        let (epoc, eh) = &opl[i];
        let status = if poc == epoc && hashes == eh {
            "OK".to_string()
        } else {
            let planes: Vec<&str> = hashes
                .iter()
                .zip(eh)
                .enumerate()
                .filter(|(_, (a, b))| a != b)
                .map(|(i, _)| ["Y", "Cb", "Cr"][i.min(2)])
                .collect();
            format!("MISMATCH poc {poc} vs {epoc} planes {planes:?}")
        };
        println!("row {i}: {status}");
    }
    println!("{} outputs vs {} opl rows", outs.len(), opl.len());
}

fn md5_hex(data: &[u8]) -> String {
    // Tiny MD5 (same as the test harness).
    const S: [u32; 64] = [
        7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 5, 9, 14, 20, 5, 9, 14, 20, 5,
        9, 14, 20, 5, 9, 14, 20, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 6, 10,
        15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
    ];
    let mut k = [0u32; 64];
    for (i, slot) in k.iter_mut().enumerate() {
        *slot = (((i as f64 + 1.0).sin().abs()) * 4294967296.0) as u32;
    }
    let mut state = [0x6745_2301u32, 0xefcd_ab89, 0x98ba_dcfe, 0x1032_5476];
    let bit_len = (data.len() as u64).wrapping_mul(8);
    let mut buf = data.to_vec();
    buf.push(0x80);
    while buf.len() % 64 != 56 {
        buf.push(0);
    }
    buf.extend_from_slice(&bit_len.to_le_bytes());
    for block in buf.chunks(64) {
        let mut m = [0u32; 16];
        for (i, slot) in m.iter_mut().enumerate() {
            *slot = u32::from_le_bytes(block[i * 4..i * 4 + 4].try_into().unwrap());
        }
        let (mut a, mut b, mut c, mut d) = (state[0], state[1], state[2], state[3]);
        for i in 0..64 {
            let (f, g) = match i / 16 {
                0 => ((b & c) | (!b & d), i),
                1 => ((d & b) | (!d & c), (5 * i + 1) % 16),
                2 => (b ^ c ^ d, (3 * i + 5) % 16),
                _ => (c ^ (b | !d), (7 * i) % 16),
            };
            let tmp = d;
            d = c;
            c = b;
            let sum = a.wrapping_add(f).wrapping_add(k[i]).wrapping_add(m[g]);
            b = b.wrapping_add(sum.rotate_left(S[i]));
            a = tmp;
        }
        state[0] = state[0].wrapping_add(a);
        state[1] = state[1].wrapping_add(b);
        state[2] = state[2].wrapping_add(c);
        state[3] = state[3].wrapping_add(d);
    }
    let mut out = String::new();
    for s in state {
        for byte in s.to_le_bytes() {
            out.push_str(&format!("{byte:02x}"));
        }
    }
    out
}
