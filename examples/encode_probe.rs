//! Encoder probe: encode a synthetic `W x H` picture at `QP`, decode it
//! with this crate's decoder and (when a scratch dir is given and
//! `ffmpeg` is installed) with `ffmpeg` as an opaque black-box
//! validator, and compare both against the encoder's own
//! reconstruction. `MTT=1` enables the MTT pickers; `LOSSLESS=1` the
//! lossless (transform-skip) mode; `FMT=0..3` the chroma format and
//! `BD=8|10` the bit depth.
//!
//! `cargo run --release --example encode_probe -- W H [QP] [SCRATCH_DIR]`
use oxideav_h266::encoder::EncoderConfig;
use oxideav_h266::encoder_pipeline::encode_idr_with_residuals_cfg;
use oxideav_h266::reconstruct::PictureBuffer;
use oxideav_h266::stream::StreamDecoder;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let w: usize = args[1].parse().unwrap();
    let h: usize = args[2].parse().unwrap();
    let qp: i32 = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(30);
    let scratch = args.get(4).cloned();
    let fmt: u32 = std::env::var("FMT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);
    let bd: u32 = std::env::var("BD")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8);
    let mid = 1u16 << (bd - 1);
    let sh = bd - 8;
    let mut src = PictureBuffer::with_chroma_format_bd(w, h, fmt, mid, bd);
    // PATTERN=noise (default) is a pseudo-random texture; PATTERN=smooth
    // is a gradient with ripples, a diagonal edge and a few discs —
    // structured content where intra prediction matters.
    let smooth = std::env::var("PATTERN")
        .map(|p| p == "smooth")
        .unwrap_or(false);
    for y in 0..h {
        for x in 0..w {
            let v = if smooth {
                let base = 40 + (x * 150) / w.max(1) + (y * 60) / h.max(1);
                let ripple = ((x * 5 + y * 3) % 17) * 3;
                let edge = if x + y > (w + h) / 2 { 30 } else { 0 };
                let dx = x as i64 - (w / 3) as i64;
                let dy = y as i64 - (h / 3) as i64;
                let disc = if dx * dx + dy * dy < ((w.min(h) / 6) as i64).pow(2) {
                    60
                } else {
                    0
                };
                (base + ripple + edge).saturating_sub(disc).min(255)
            } else {
                (x * 7 + y * 13 + (x * y) % 31) & 0xff
            };
            src.luma.samples[y * w + x] = (v << sh) as u16;
        }
    }
    let cw = src.cb.width;
    for y in 0..src.cb.height {
        for x in 0..cw {
            let (cb, cr) = if smooth {
                (
                    100 + (x * 60) / cw.max(1),
                    160 - (y * 50) / src.cb.height.max(1),
                )
            } else {
                ((x * 5 + y * 3) & 0xff, (x * 11 + y * 2 + 100) & 0xff)
            };
            src.cb.samples[y * cw + x] = (cb << sh) as u16;
            src.cr.samples[y * cw + x] = (cr << sh) as u16;
        }
    }
    let mut cfg = EncoderConfig::new(w as u32, h as u32);
    cfg.chroma_format_idc = fmt as u8;
    cfg.bit_depth = bd as u8;
    if std::env::var_os("MTT").is_some() {
        cfg.enable_mtt_bt_picker = true;
        cfg.enable_mtt_tt_picker = true;
    }
    if std::env::var_os("LOSSLESS").is_some() {
        cfg.lossless = true;
    }
    let (bs, rec) = encode_idr_with_residuals_cfg(&src, qp, cfg).unwrap();
    let psnr = oxideav_h266::encoder_pipeline::psnr_y(&src.luma, &rec.luma).unwrap();
    println!(
        "{w}x{h} fmt{fmt} {bd}-bit qp{qp}: {} bytes, psnr_y {psnr:.2}",
        bs.len()
    );
    let mut dec = StreamDecoder::new();
    let mut got = None;
    match dec.decode_annex_b(&bs, &mut |pic| got = Some(pic)) {
        Err(e) => println!("  OWN DECODE ERROR: {e}"),
        Ok(()) => {
            let pic = got.expect("no picture");
            let mism = cmp(&rec, &pic.frame, w, h, fmt != 0);
            println!(
                "  own decoder: {}x{} crop {:?} mismatches {mism}",
                pic.frame.luma.width, pic.frame.luma.height, pic.crop
            );
        }
    }
    let Some(dir) = scratch else { return };
    let bit = format!("{dir}/probe_{w}x{h}.266");
    let yuv = format!("{dir}/probe_{w}x{h}.yuv");
    std::fs::write(&bit, &bs).unwrap();
    let pix_fmt = match (fmt, bd) {
        (0, 8) => "gray",
        (0, _) => "gray10le",
        (1, 8) => "yuv420p",
        (1, _) => "yuv420p10le",
        (2, 8) => "yuv422p",
        (2, _) => "yuv422p10le",
        (_, 8) => "yuv444p",
        (_, _) => "yuv444p10le",
    };
    let st = std::process::Command::new("ffmpeg")
        .args(["-hide_banner", "-loglevel", "error", "-y", "-i", &bit])
        .args(["-f", "rawvideo", "-pix_fmt", pix_fmt, &yuv])
        .status();
    match st {
        Err(e) => println!("  ffmpeg unavailable: {e}"),
        Ok(s) if !s.success() => println!("  FFMPEG DECODE FAILED: {s}"),
        Ok(_) => {
            let raw = std::fs::read(&yuv).unwrap();
            let data: Vec<u16> = if bd > 8 {
                raw.chunks_exact(2)
                    .map(|c| u16::from_le_bytes([c[0], c[1]]))
                    .collect()
            } else {
                raw.iter().map(|&b| u16::from(b)).collect()
            };
            // Decoder output geometry = conformance-cropped size.
            let (ow, oh) = cfg.output_size();
            let (ow, oh) = (ow as usize, oh as usize);
            let (sw, shh) = cfg.chroma_subsampling();
            let (ocw, och) = if fmt == 0 {
                (0, 0)
            } else {
                (ow / sw as usize, oh / shh as usize)
            };
            let expect = ow * oh + 2 * ocw * och;
            if data.len() != expect {
                println!(
                    "  FFMPEG size {} != expected {expect} ({ow}x{oh})",
                    data.len()
                );
                return;
            }
            let mut mism = 0usize;
            for y in 0..h {
                for x in 0..w {
                    if data[y * ow + x] != rec.luma.samples[y * rec.luma.stride + x] {
                        mism += 1;
                    }
                }
            }
            if fmt != 0 {
                let cbo = ow * oh;
                let cro = cbo + ocw * och;
                for y in 0..rec.cb.height {
                    for x in 0..rec.cb.width {
                        if data[cbo + y * ocw + x] != rec.cb.samples[y * rec.cb.stride + x] {
                            mism += 1;
                        }
                        if data[cro + y * ocw + x] != rec.cr.samples[y * rec.cr.stride + x] {
                            mism += 1;
                        }
                    }
                }
            }
            println!("  ffmpeg: {ow}x{oh} {pix_fmt} mismatches {mism}");
        }
    }
}

fn cmp(rec: &PictureBuffer, dec: &PictureBuffer, w: usize, h: usize, chroma: bool) -> usize {
    let mut mism = 0;
    for y in 0..h {
        for x in 0..w {
            if dec.luma.samples[y * dec.luma.stride + x]
                != rec.luma.samples[y * rec.luma.stride + x]
            {
                mism += 1;
            }
        }
    }
    if !chroma {
        return mism;
    }
    for y in 0..rec.cb.height {
        for x in 0..rec.cb.width {
            if dec.cb.samples[y * dec.cb.stride + x] != rec.cb.samples[y * rec.cb.stride + x] {
                mism += 1;
            }
            if dec.cr.samples[y * dec.cr.stride + x] != rec.cr.samples[y * rec.cr.stride + x] {
                mism += 1;
            }
        }
    }
    mism
}
