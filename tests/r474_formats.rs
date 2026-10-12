//! r474 — chroma formats (4:0:0 / 4:2:0 / 4:2:2 / 4:4:4) and 10-bit
//! encoding: lossy and lossless streams of every format decode
//! byte-exact (resp. bit-identical to the source) through this crate's
//! decoder, and through a black-box third-party decoder where one is
//! installed.
//!
//! Spec reference: ITU-T H.266 | ISO/IEC 23090-3 (V4, 01/2026).

use oxideav_h266::encoder::EncoderConfig;
use oxideav_h266::encoder_pipeline::{encode_idr_with_residuals_cfg, psnr_y};
use oxideav_h266::reconstruct::{PictureBuffer, PicturePlane};
use oxideav_h266::stream::{DecodedPicture, StreamDecoder};

fn textured_source(w: usize, h: usize, chroma_format_idc: u32, bit_depth: u32) -> PictureBuffer {
    let mid = 1u16 << (bit_depth - 1);
    let sh = bit_depth - 8;
    let mut src = PictureBuffer::with_chroma_format_bd(w, h, chroma_format_idc, mid, bit_depth);
    for y in 0..h {
        for x in 0..w {
            let base = 40 + (x * 150) / w.max(1) + (y * 60) / h.max(1);
            let ripple = ((x * 5 + y * 3) % 17) * 3;
            let edge = if x + y > (w + h) / 2 { 30 } else { 0 };
            // Spread the low bits so 10-bit sources are not just 8-bit << 2.
            let v = ((base + ripple + edge).min(255) << sh) + ((x * 3 + y) % (1 << sh).max(1));
            src.luma.samples[y * w + x] = v as u16;
        }
    }
    let cw = src.cb.width;
    for y in 0..src.cb.height {
        for x in 0..cw {
            src.cb.samples[y * cw + x] =
                (((90 + (x * 7 + y * 2) % 60) << sh) + (x % (1 << sh).max(1))) as u16;
            src.cr.samples[y * cw + x] =
                (((150 - ((x * 3 + y * 5) % 50)) << sh) + (y % (1 << sh).max(1))) as u16;
        }
    }
    src
}

fn plane_eq(a: &PicturePlane, b: &PicturePlane) -> bool {
    (0..a.height).all(|y| {
        a.samples[y * a.stride..y * a.stride + a.width]
            == b.samples[y * b.stride..y * b.stride + a.width]
    })
}

fn decode_once(bs: &[u8]) -> DecodedPicture {
    let mut dec = StreamDecoder::new();
    let mut got = None;
    dec.decode_annex_b(bs, &mut |pic| got = Some(pic))
        .expect("own decoder rejects the stream");
    got.expect("no picture decoded")
}

/// Encode `src` with `cfg`, check the decoder reproduces the encoder
/// reconstruction (and, lossless, the source); returns the stream.
fn round_trip(
    src: &PictureBuffer,
    mut cfg: EncoderConfig,
    qp: i32,
    label: &str,
) -> (Vec<u8>, PictureBuffer) {
    let fmt = u32::from(cfg.chroma_format_idc);
    cfg.width = src.luma.width as u32;
    cfg.height = src.luma.height as u32;
    let (bs, rec) = encode_idr_with_residuals_cfg(src, qp, cfg)
        .unwrap_or_else(|e| panic!("{label}: encode failed: {e}"));
    let pic = decode_once(&bs);
    assert_eq!(
        u32::from(pic.chroma_format_idc),
        fmt,
        "{label}: SPS chroma format"
    );
    assert_eq!(pic.bit_depth, src.bit_depth(), "{label}: SPS bit depth");
    let (ow, oh) = cfg.output_size();
    assert_eq!(
        pic.crop,
        (0, 0, ow as usize, oh as usize),
        "{label}: conformance window"
    );
    assert!(
        plane_eq(&rec.luma, &pic.frame.luma),
        "{label}: decoded luma differs from the encoder reconstruction"
    );
    if fmt != 0 {
        assert!(
            plane_eq(&rec.cb, &pic.frame.cb),
            "{label}: decoded Cb differs"
        );
        assert!(
            plane_eq(&rec.cr, &pic.frame.cr),
            "{label}: decoded Cr differs"
        );
    }
    if cfg.lossless {
        assert!(
            plane_eq(&src.luma, &rec.luma),
            "{label}: lossless luma is not the source"
        );
        if fmt != 0 {
            assert!(
                plane_eq(&src.cb, &rec.cb),
                "{label}: lossless Cb is not the source"
            );
            assert!(
                plane_eq(&src.cr, &rec.cr),
                "{label}: lossless Cr is not the source"
            );
        }
    } else {
        let p = psnr_y(&src.luma, &rec.luma).unwrap();
        assert!(p > 20.0, "{label}: PSNR_Y {p:.2} dB implausibly low");
    }
    (bs, rec)
}

const FORMATS: [u32; 4] = [0, 1, 2, 3];

#[test]
fn every_chroma_format_lossy_8bit_round_trips() {
    for &fmt in &FORMATS {
        for &(w, h) in &[(64usize, 64usize), (128, 128), (100, 100), (33, 17)] {
            let src = textured_source(w, h, fmt, 8);
            let mut cfg = EncoderConfig::new(w as u32, h as u32);
            cfg.chroma_format_idc = fmt as u8;
            round_trip(&src, cfg, 30, &format!("fmt{fmt} 8-bit {w}x{h}"));
        }
    }
}

#[test]
fn every_chroma_format_lossy_10bit_round_trips() {
    for &fmt in &FORMATS {
        for &(w, h) in &[(64usize, 64usize), (100, 100), (33, 17)] {
            let src = textured_source(w, h, fmt, 10);
            let mut cfg = EncoderConfig::new(w as u32, h as u32);
            cfg.chroma_format_idc = fmt as u8;
            for qp in [8, 30, 51] {
                round_trip(&src, cfg, qp, &format!("fmt{fmt} 10-bit {w}x{h} qp{qp}"));
            }
        }
    }
}

#[test]
fn every_chroma_format_lossless_8_and_10bit() {
    for &fmt in &FORMATS {
        for &bd in &[8u32, 10] {
            for &(w, h) in &[(64usize, 64usize), (100, 100), (7, 5)] {
                let src = textured_source(w, h, fmt, bd);
                let mut cfg = EncoderConfig::new(w as u32, h as u32);
                cfg.chroma_format_idc = fmt as u8;
                cfg.lossless = true;
                round_trip(
                    &src,
                    cfg,
                    30,
                    &format!("fmt{fmt} {bd}-bit lossless {w}x{h}"),
                );
            }
        }
    }
}

#[test]
fn mtt_pickers_on_422_and_444() {
    for &fmt in &[2u32, 3] {
        let src = textured_source(130, 70, fmt, 8);
        let mut cfg = EncoderConfig::new(130, 70);
        cfg.chroma_format_idc = fmt as u8;
        cfg.enable_mtt_bt_picker = true;
        cfg.enable_mtt_tt_picker = true;
        round_trip(&src, cfg, 28, &format!("fmt{fmt} MTT 130x70"));
    }
}

#[test]
fn odd_sizes_crop_exactly_without_horizontal_subsampling() {
    // 4:0:0 and 4:4:4 (SubWidthC = SubHeightC = 1) express odd sizes
    // exactly; 4:2:2 crops the width to even but the height exactly.
    let cfg = |fmt: u8| {
        let mut c = EncoderConfig::new(7, 5);
        c.chroma_format_idc = fmt;
        c
    };
    assert_eq!(cfg(0).output_size(), (7, 5));
    assert_eq!(cfg(3).output_size(), (7, 5));
    assert_eq!(cfg(2).output_size(), (8, 5));
    assert_eq!(cfg(1).output_size(), (8, 6));
    for fmt in [0u8, 2, 3] {
        let src = textured_source(7, 5, u32::from(fmt), 8);
        round_trip(&src, cfg(fmt), 30, &format!("fmt{fmt} 7x5"));
    }
}

#[test]
fn config_rejects_unsupported_depths_and_formats() {
    let src = textured_source(16, 16, 1, 8);
    let mut cfg = EncoderConfig::new(16, 16);
    cfg.chroma_format_idc = 4;
    assert!(encode_idr_with_residuals_cfg(&src, 30, cfg).is_err());
    let mut cfg = EncoderConfig::new(16, 16);
    cfg.palette = true;
    cfg.chroma_format_idc = 3;
    assert!(encode_idr_with_residuals_cfg(&textured_source(16, 16, 3, 8), 30, cfg).is_err());
    let src12 = textured_source(16, 16, 1, 12);
    assert!(encode_idr_with_residuals_cfg(&src12, 30, EncoderConfig::new(16, 16)).is_err());
}

/// Black-box validation through `ffmpeg` (opaque process; no-op where
/// the binary is absent): every format / depth, lossy and lossless.
#[test]
fn ffmpeg_black_box_agrees_on_every_format_and_depth() {
    let probe = std::process::Command::new("ffmpeg")
        .arg("-version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status();
    if !matches!(probe, Ok(s) if s.success()) {
        eprintln!("ffmpeg not installed — black-box leg skipped");
        return;
    }
    let dir = std::env::temp_dir().join(format!("oxideav-h266-r474-fmt-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    for &fmt in &FORMATS {
        for &bd in &[8u32, 10] {
            for &lossless in &[false, true] {
                let (w, h) = (100usize, 68usize);
                let src = textured_source(w, h, fmt, bd);
                let mut cfg = EncoderConfig::new(w as u32, h as u32);
                cfg.chroma_format_idc = fmt as u8;
                cfg.lossless = lossless;
                let label = format!("fmt{fmt} {bd}-bit lossless={lossless}");
                let (bs, rec) = round_trip(&src, cfg, 30, &label);
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
                let bit = dir.join(format!("{fmt}_{bd}_{lossless}.266"));
                let yuv = dir.join(format!("{fmt}_{bd}_{lossless}.yuv"));
                std::fs::write(&bit, &bs).unwrap();
                let st = std::process::Command::new("ffmpeg")
                    .args(["-hide_banner", "-loglevel", "error", "-y", "-i"])
                    .arg(&bit)
                    .args(["-f", "rawvideo", "-pix_fmt", pix_fmt])
                    .arg(&yuv)
                    .status()
                    .unwrap();
                assert!(st.success(), "{label}: ffmpeg failed to decode");
                let raw = std::fs::read(&yuv).unwrap();
                let data: Vec<u16> = if bd > 8 {
                    raw.chunks_exact(2)
                        .map(|c| u16::from_le_bytes([c[0], c[1]]))
                        .collect()
                } else {
                    raw.iter().map(|&b| u16::from(b)).collect()
                };
                let (ow, oh) = cfg.output_size();
                let (ow, oh) = (ow as usize, oh as usize);
                let (sw, shh) = cfg.chroma_subsampling();
                let (ocw, och) = if fmt == 0 {
                    (0, 0)
                } else {
                    (ow / sw as usize, oh / shh as usize)
                };
                assert_eq!(
                    data.len(),
                    ow * oh + 2 * ocw * och,
                    "{label}: ffmpeg output geometry"
                );
                let mut n = 0usize;
                for y in 0..h {
                    for x in 0..w {
                        if data[y * ow + x] != rec.luma.samples[y * rec.luma.stride + x] {
                            n += 1;
                        }
                    }
                }
                if fmt != 0 {
                    let cb0 = ow * oh;
                    let cr0 = cb0 + ocw * och;
                    for y in 0..rec.cb.height {
                        for x in 0..rec.cb.width {
                            if data[cb0 + y * ocw + x] != rec.cb.samples[y * rec.cb.stride + x] {
                                n += 1;
                            }
                            if data[cr0 + y * ocw + x] != rec.cr.samples[y * rec.cr.stride + x] {
                                n += 1;
                            }
                        }
                    }
                }
                assert_eq!(
                    n, 0,
                    "{label}: ffmpeg output differs from the encoder reconstruction"
                );
            }
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}
