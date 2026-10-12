//! Encoder probe: encode a synthetic `W x H` picture at `QP`, decode it
//! with this crate's decoder and (when a scratch dir is given and
//! `ffmpeg` is installed) with `ffmpeg` as an opaque black-box
//! validator, and compare both against the encoder's own
//! reconstruction. `MTT=1` enables the MTT pickers; `LOSSLESS=1` the
//! lossless (transform-skip) mode.
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
    let mut src = PictureBuffer::yuv420_filled(w, h, 128);
    for y in 0..h {
        for x in 0..w {
            src.luma.samples[y * w + x] = ((x * 7 + y * 13 + (x * y) % 31) & 0xff) as u16;
        }
    }
    let cw = src.cb.width;
    for y in 0..src.cb.height {
        for x in 0..cw {
            src.cb.samples[y * cw + x] = ((x * 5 + y * 3) & 0xff) as u16;
            src.cr.samples[y * cw + x] = ((x * 11 + y * 2 + 100) & 0xff) as u16;
        }
    }
    let mut cfg = EncoderConfig::new(w as u32, h as u32);
    if std::env::var_os("MTT").is_some() {
        cfg.enable_mtt_bt_picker = true;
        cfg.enable_mtt_tt_picker = true;
    }
    if std::env::var_os("LOSSLESS").is_some() {
        cfg.lossless = true;
    }
    let (bs, rec) = encode_idr_with_residuals_cfg(&src, qp, cfg).unwrap();
    let psnr = oxideav_h266::encoder_pipeline::psnr_y(&src.luma, &rec.luma).unwrap();
    println!("{w}x{h} qp{qp}: {} bytes, psnr_y {psnr:.2}", bs.len());
    let mut dec = StreamDecoder::new();
    let mut got = None;
    match dec.decode_annex_b(&bs, &mut |pic| got = Some(pic)) {
        Err(e) => println!("  OWN DECODE ERROR: {e}"),
        Ok(()) => {
            let pic = got.expect("no picture");
            let mism = cmp(&rec, &pic.frame, w, h);
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
    let st = std::process::Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            &bit,
            "-f",
            "rawvideo",
            "-pix_fmt",
            "yuv420p",
            &yuv,
        ])
        .status();
    match st {
        Err(e) => println!("  ffmpeg unavailable: {e}"),
        Ok(s) if !s.success() => println!("  FFMPEG DECODE FAILED: {s}"),
        Ok(_) => {
            let data = std::fs::read(&yuv).unwrap();
            // ffmpeg output geometry = conformance-cropped size (even for 4:2:0).
            let ow = (w + 1) & !1;
            let oh = (h + 1) & !1;
            let expect = ow * oh + 2 * (ow / 2) * (oh / 2);
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
                    if data[y * ow + x] as u16 != rec.luma.samples[y * rec.luma.stride + x] {
                        mism += 1;
                    }
                }
            }
            let cbo = ow * oh;
            let cro = cbo + (ow / 2) * (oh / 2);
            for y in 0..rec.cb.height {
                for x in 0..rec.cb.width {
                    if data[cbo + y * (ow / 2) + x] as u16 != rec.cb.samples[y * rec.cb.stride + x]
                    {
                        mism += 1;
                    }
                    if data[cro + y * (ow / 2) + x] as u16 != rec.cr.samples[y * rec.cr.stride + x]
                    {
                        mism += 1;
                    }
                }
            }
            println!("  ffmpeg: {ow}x{oh} mismatches {mism}");
        }
    }
}

fn cmp(rec: &PictureBuffer, dec: &PictureBuffer, w: usize, h: usize) -> usize {
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
