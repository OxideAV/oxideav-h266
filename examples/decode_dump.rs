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
    let mut raw: Vec<u8> = Vec::new();
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
    });
    if let Err(e) = r {
        eprintln!("DECODE ERROR after {n} pictures: {e}");
    }
    std::fs::write(&output, raw).unwrap();
    eprintln!("wrote {n} pictures to {output}");
}
