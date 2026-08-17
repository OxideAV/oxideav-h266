//! Debug aid: decode an arbitrary Annex-B `.bit` file and dump every
//! output picture's cropped planes as raw YUV (8-bit as 1 byte/sample,
//! deeper depths as 2-byte LE) — companion to `triage_dbg` for
//! locally-built fixtures that have no `.opl` sidecar.
//!
//! Usage: `decode_dump <in.bit> <out.yuv>`

use oxideav_h266::stream::StreamDecoder;

fn main() {
    let mut args = std::env::args().skip(1);
    let input = args.next().expect("input .bit path");
    let output = args.next().expect("output .yuv path");
    let bs = std::fs::read(&input).unwrap();
    // (cvs, poc, raw-planes) — sorted into output order at the end so
    // the dump matches a display-order reference decode byte-for-byte.
    let mut pics: Vec<(u32, i32, Vec<u8>)> = Vec::new();
    let mut n = 0usize;
    let r = StreamDecoder::new().decode_annex_b(&bs, &mut |pic| {
        if !pic.output_flag {
            return;
        }
        n += 1;
        let (cx, cy, cw, ch) = pic.crop;
        let wide = pic.bit_depth > 8;
        let mut planes = vec![(&pic.frame.luma, cx, cy, cw, ch)];
        if pic.chroma_format_idc != 0 {
            planes.push((&pic.frame.cb, cx / 2, cy / 2, cw / 2, ch / 2));
            planes.push((&pic.frame.cr, cx / 2, cy / 2, cw / 2, ch / 2));
        }
        let mut raw: Vec<u8> = Vec::new();
        for (p, x0, y0, w, h) in planes {
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
        // Cheap per-picture fingerprint (FNV-1a over the raw planes)
        // so a display-order mismatch against a reference decode can
        // be localized without external tooling.
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for &b in &raw {
            h ^= b as u64;
            h = h.wrapping_mul(0x1000_0000_01b3);
        }
        eprintln!("pic cvs={} poc={} fnv={h:016x}", pic.cvs_idx, pic.poc);
        pics.push((pic.cvs_idx, pic.poc, raw));
    });
    if let Err(e) = r {
        eprintln!("DECODE ERROR after {n} pictures: {e}");
    }
    pics.sort_by_key(|(cvs, poc, _)| (*cvs, *poc));
    let mut raw: Vec<u8> = Vec::new();
    for (_, _, p) in &pics {
        raw.extend_from_slice(p);
    }
    std::fs::write(&output, raw).unwrap();
    eprintln!("wrote {n} pictures to {output}");
}
