//! r474 — `EncoderConfig::lossless`: transform-skip coding at `Qp′ =
//! QpPrimeTsMin = 4` with the in-loop filters off decodes bit-identical
//! to the source through this crate's own decoder (and through a
//! black-box third-party decoder where one is installed).
//!
//! Spec reference: ITU-T H.266 | ISO/IEC 23090-3 (V4, 01/2026) —
//! §7.3.11.10 / §7.3.11.12 / §8.7.2 eq. 1129 / §8.7.3.

use oxideav_h266::encoder::EncoderConfig;
use oxideav_h266::encoder_pipeline::encode_idr_with_residuals_cfg;
use oxideav_h266::reconstruct::PictureBuffer;
use oxideav_h266::stream::StreamDecoder;

/// xorshift noise — the worst case for prediction (every level is a
/// full-range residual; exercises the `RemCcbs` budget exhaustion and
/// the bypass tail of `residual_ts_coding()`).
fn noise_source(w: usize, h: usize, seed: u32) -> PictureBuffer {
    let mut s = seed | 1;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        (s & 0xff) as u16
    };
    let mut src = PictureBuffer::yuv420_filled(w, h, 128);
    for v in src.luma.samples.iter_mut() {
        *v = next();
    }
    for v in src.cb.samples.iter_mut() {
        *v = next();
    }
    for v in src.cr.samples.iter_mut() {
        *v = next();
    }
    src
}

/// Smooth gradient + edge — the compressible case.
fn smooth_source(w: usize, h: usize) -> PictureBuffer {
    let mut src = PictureBuffer::yuv420_filled(w, h, 128);
    for y in 0..h {
        for x in 0..w {
            let v = 30 + (x * 160) / w.max(1) + (y * 50) / h.max(1) + usize::from(x + y > w) * 20;
            src.luma.samples[y * w + x] = v.min(255) as u16;
        }
    }
    let cw = src.cb.width;
    for y in 0..src.cb.height {
        for x in 0..cw {
            src.cb.samples[y * cw + x] = (100 + x / 2 + y / 3) as u16;
            src.cr.samples[y * cw + x] = (140 - (x / 3).min(60)) as u16;
        }
    }
    src
}

fn plane_eq(
    a: &oxideav_h266::reconstruct::PicturePlane,
    b: &oxideav_h266::reconstruct::PicturePlane,
) -> bool {
    (0..a.height).all(|y| {
        a.samples[y * a.stride..y * a.stride + a.width]
            == b.samples[y * b.stride..y * b.stride + a.width]
    })
}

fn assert_lossless(src: &PictureBuffer, label: &str) -> Vec<u8> {
    let (w, h) = (src.luma.width, src.luma.height);
    let mut cfg = EncoderConfig::new(w as u32, h as u32);
    cfg.lossless = true;
    // The QP argument is irrelevant in lossless mode (Qp′ is pinned at 4).
    let (bs, rec) = encode_idr_with_residuals_cfg(src, 30, cfg)
        .unwrap_or_else(|e| panic!("{label}: encode failed: {e}"));
    assert!(
        plane_eq(&src.luma, &rec.luma),
        "{label}: encoder luma reconstruction is not the source"
    );
    assert!(
        plane_eq(&src.cb, &rec.cb),
        "{label}: encoder Cb reconstruction is not the source"
    );
    assert!(
        plane_eq(&src.cr, &rec.cr),
        "{label}: encoder Cr reconstruction is not the source"
    );
    let mut dec = StreamDecoder::new();
    let mut got = None;
    dec.decode_annex_b(&bs, &mut |pic| got = Some(pic))
        .unwrap_or_else(|e| panic!("{label}: own decoder rejects the lossless stream: {e}"));
    let pic = got.expect("no picture");
    assert!(
        plane_eq(&src.luma, &pic.frame.luma),
        "{label}: decoded luma differs from the source"
    );
    assert!(
        plane_eq(&src.cb, &pic.frame.cb),
        "{label}: decoded Cb differs from the source"
    );
    assert!(
        plane_eq(&src.cr, &pic.frame.cr),
        "{label}: decoded Cr differs from the source"
    );
    bs
}

#[test]
fn noise_pictures_decode_bit_identical() {
    for &(w, h) in &[
        (8usize, 8usize),
        (64, 64),
        (128, 128),
        (100, 100),
        (7, 5),
        (33, 17),
        (200, 136),
    ] {
        assert_lossless(
            &noise_source(w, h, (w * 131 + h) as u32),
            &format!("noise {w}x{h}"),
        );
    }
}

#[test]
fn smooth_pictures_decode_bit_identical_and_compress() {
    let src = smooth_source(256, 160);
    let bs = assert_lossless(&src, "smooth 256x160");
    // DC-only intra prediction leaves a ramp residual on a gradient, so
    // the lossless stream is only moderately smaller than raw 4:2:0.
    let raw = 256 * 160 * 3 / 2;
    eprintln!("smooth 256x160: lossless {} bytes vs raw {raw}", bs.len());
    assert!(
        bs.len() < raw,
        "smooth 256x160 lossless stream should be under the raw size ({} vs {raw})",
        bs.len()
    );
}

#[test]
fn every_square_size_1_to_40_lossless() {
    for n in 1..=40usize {
        assert_lossless(&noise_source(n, n, n as u32), &format!("noise {n}x{n}"));
    }
}

#[test]
fn lossless_excludes_incompatible_tools() {
    let mut cfg = EncoderConfig::new(64, 64);
    cfg.lossless = true;
    cfg.dep_quant = true;
    assert!(encode_idr_with_residuals_cfg(&smooth_source(64, 64), 30, cfg).is_err());
    let mut cfg = EncoderConfig::new(64, 64);
    cfg.lossless = true;
    cfg.enable_mtt_bt_picker = true;
    assert!(encode_idr_with_residuals_cfg(&smooth_source(64, 64), 30, cfg).is_err());
}

/// Black-box validation through `ffmpeg`'s VVC decoder (opaque process;
/// a no-op where the binary is absent).
#[test]
fn ffmpeg_black_box_decodes_lossless_streams_to_the_source() {
    let probe = std::process::Command::new("ffmpeg")
        .arg("-version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status();
    if !matches!(probe, Ok(s) if s.success()) {
        eprintln!("ffmpeg not installed — black-box leg skipped");
        return;
    }
    let dir = std::env::temp_dir().join(format!("oxideav-h266-r474-ll-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    for &(w, h) in &[(64usize, 64usize), (100, 100), (7, 5), (33, 17)] {
        let src = noise_source(w, h, 7 + w as u32);
        let bs = assert_lossless(&src, &format!("noise {w}x{h}"));
        let bit = dir.join(format!("{w}x{h}.266"));
        let yuv = dir.join(format!("{w}x{h}.yuv"));
        std::fs::write(&bit, &bs).unwrap();
        let st = std::process::Command::new("ffmpeg")
            .args(["-hide_banner", "-loglevel", "error", "-y", "-i"])
            .arg(&bit)
            .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"])
            .arg(&yuv)
            .status()
            .unwrap();
        assert!(
            st.success(),
            "{w}x{h}: ffmpeg failed to decode the lossless stream"
        );
        let data = std::fs::read(&yuv).unwrap();
        let (ow, oh) = ((w + 1) & !1, (h + 1) & !1);
        assert_eq!(data.len(), ow * oh + 2 * (ow / 2) * (oh / 2));
        for y in 0..h {
            for x in 0..w {
                assert_eq!(
                    u16::from(data[y * ow + x]),
                    src.luma.samples[y * w + x],
                    "{w}x{h}: luma ({x},{y})"
                );
            }
        }
        let cb0 = ow * oh;
        let cr0 = cb0 + (ow / 2) * (oh / 2);
        let cw = src.cb.width;
        for y in 0..src.cb.height {
            for x in 0..cw {
                assert_eq!(
                    u16::from(data[cb0 + y * (ow / 2) + x]),
                    src.cb.samples[y * cw + x],
                    "{w}x{h}: cb ({x},{y})"
                );
                assert_eq!(
                    u16::from(data[cr0 + y * (ow / 2) + x]),
                    src.cr.samples[y * cw + x],
                    "{w}x{h}: cr ({x},{y})"
                );
            }
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}
