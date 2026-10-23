//! r474 — every picture geometry encodes to a stream this crate's own
//! decoder reproduces byte-exact (the §7.4.3.4 conformance window +
//! §7.4.12.4 boundary coding tree), and the black-box `ffmpeg` decoder
//! agrees whenever it is installed.
//!
//! Spec reference: ITU-T H.266 | ISO/IEC 23090-3 (V4, 01/2026).

use oxideav_h266::encoder::{CODED_SIZE_ALIGN, EncoderConfig};
use oxideav_h266::encoder_pipeline::encode_idr_with_residuals_cfg;
use oxideav_h266::reconstruct::PictureBuffer;
use oxideav_h266::stream::{DecodedPicture, StreamDecoder};

/// Deterministic textured source: a gradient with a mid-frequency
/// ripple and a diagonal edge so every CU carries residual.
fn synthetic_source(w: usize, h: usize) -> PictureBuffer {
    let mut src = PictureBuffer::yuv420_filled(w, h, 128);
    for y in 0..h {
        for x in 0..w {
            let base = 40 + (x * 150) / w.max(1) + (y * 60) / h.max(1);
            let ripple = ((x * 5 + y * 3) % 17) * 3;
            let edge = if x + y > (w + h) / 2 { 30 } else { 0 };
            src.luma.samples[y * w + x] = (base + ripple + edge).min(255) as u16;
        }
    }
    let cw = src.cb.width;
    for y in 0..src.cb.height {
        for x in 0..cw {
            src.cb.samples[y * cw + x] = (90 + (x * 7 + y * 2) % 60) as u16;
            src.cr.samples[y * cw + x] = (150 - ((x * 3 + y * 5) % 50)) as u16;
        }
    }
    src
}

fn decode_once(bs: &[u8]) -> DecodedPicture {
    let mut dec = StreamDecoder::new();
    let mut got = None;
    dec.decode_annex_b(bs, &mut |pic| got = Some(pic))
        .expect("own decoder rejects the encoder's stream");
    got.expect("the stream decoded to no picture")
}

/// Count sample mismatches between the encoder reconstruction (source
/// geometry) and a decoded picture on the overlapping region.
fn mismatches(rec: &PictureBuffer, dec: &PictureBuffer) -> usize {
    let mut n = 0;
    for y in 0..rec.luma.height {
        for x in 0..rec.luma.width {
            if dec.luma.samples[y * dec.luma.stride + x]
                != rec.luma.samples[y * rec.luma.stride + x]
            {
                n += 1;
            }
        }
    }
    for (r, d) in [(&rec.cb, &dec.cb), (&rec.cr, &dec.cr)] {
        for y in 0..r.height {
            for x in 0..r.width {
                if d.samples[y * d.stride + x] != r.samples[y * r.stride + x] {
                    n += 1;
                }
            }
        }
    }
    n
}

fn round_trip(w: usize, h: usize, qp: i32, cfg: EncoderConfig) -> (Vec<u8>, PictureBuffer) {
    let src = synthetic_source(w, h);
    let (bs, rec) = encode_idr_with_residuals_cfg(&src, qp, cfg)
        .unwrap_or_else(|e| panic!("{w}x{h}: encode failed: {e}"));
    assert_eq!(
        rec.luma.width, w,
        "{w}x{h}: reconstruction keeps the source width"
    );
    assert_eq!(
        rec.luma.height, h,
        "{w}x{h}: reconstruction keeps the source height"
    );
    let pic = decode_once(&bs);
    let coded_w = w.div_ceil(CODED_SIZE_ALIGN as usize) * CODED_SIZE_ALIGN as usize;
    let coded_h = h.div_ceil(CODED_SIZE_ALIGN as usize) * CODED_SIZE_ALIGN as usize;
    assert_eq!(
        (pic.frame.luma.width, pic.frame.luma.height),
        (coded_w, coded_h),
        "{w}x{h}: coded size is the 8-aligned padding"
    );
    // Conformance window: the source size rounded up to the 4:2:0
    // chroma subsampling.
    assert_eq!(
        pic.crop,
        (0, 0, (w + 1) & !1, (h + 1) & !1),
        "{w}x{h}: conformance window"
    );
    assert_eq!(
        mismatches(&rec, &pic.frame),
        0,
        "{w}x{h}: own decoder differs from the encoder reconstruction"
    );
    (bs, rec)
}

#[test]
fn every_square_size_1_to_70_round_trips() {
    for n in 1..=70usize {
        round_trip(n, n, 30, EncoderConfig::new(n as u32, n as u32));
    }
}

#[test]
fn prime_and_awkward_rectangles_round_trip() {
    for &(w, h) in &[
        (7usize, 5usize),
        (5, 7),
        (13, 11),
        (37, 19),
        (19, 37),
        (67, 71),
        (97, 131),
        (131, 97),
        (100, 100),
        (129, 65),
        (200, 136),
        (257, 129),
        (1, 300),
        (300, 1),
    ] {
        round_trip(w, h, 28, EncoderConfig::new(w as u32, h as u32));
    }
}

#[test]
fn boundary_tree_with_mtt_pickers_round_trips() {
    for &(w, h) in &[
        (100usize, 100usize),
        (7, 5),
        (200, 136),
        (130, 70),
        (257, 129),
    ] {
        let mut cfg = EncoderConfig::new(w as u32, h as u32);
        cfg.enable_mtt_bt_picker = true;
        cfg.enable_mtt_tt_picker = true;
        round_trip(w, h, 30, cfg);
    }
}

#[test]
fn tiny_pictures_at_every_qp_band() {
    for qp in [0, 12, 26, 40, 51] {
        round_trip(7, 5, qp, EncoderConfig::new(7, 5));
        round_trip(1, 1, qp, EncoderConfig::new(1, 1));
    }
}

#[test]
fn conformance_window_and_coded_size_derivation() {
    let cfg = EncoderConfig::new(7, 5);
    assert_eq!((cfg.coded_width(), cfg.coded_height()), (8, 8));
    assert_eq!(cfg.conformance_window(), (0, 0, 0, 1));
    assert_eq!(cfg.output_size(), (8, 6));
    let cfg = EncoderConfig::new(100, 100);
    assert_eq!((cfg.coded_width(), cfg.coded_height()), (104, 104));
    assert_eq!(cfg.conformance_window(), (0, 2, 0, 2));
    assert_eq!(cfg.output_size(), (100, 100));
    let cfg = EncoderConfig::new(128, 64);
    assert_eq!(cfg.conformance_window(), (0, 0, 0, 0));
    assert_eq!(cfg.output_size(), (128, 64));
}

/// Black-box validation: when `ffmpeg` is installed, its VVC decoder
/// must reproduce the encoder reconstruction on the cropped output.
/// Opaque process invocation only (no third-party decoder source is
/// consulted); silently a no-op where the binary is absent.
#[test]
fn ffmpeg_black_box_agrees_on_awkward_sizes() {
    let probe = std::process::Command::new("ffmpeg")
        .arg("-version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status();
    if !matches!(probe, Ok(s) if s.success()) {
        eprintln!("ffmpeg not installed — black-box leg skipped");
        return;
    }
    let dir = std::env::temp_dir().join(format!("oxideav-h266-r474-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    for &(w, h) in &[
        (7usize, 5usize),
        (33, 17),
        (100, 100),
        (130, 70),
        (257, 129),
    ] {
        let (bs, rec) = round_trip(w, h, 30, EncoderConfig::new(w as u32, h as u32));
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
        assert!(st.success(), "{w}x{h}: ffmpeg failed to decode the stream");
        let data = std::fs::read(&yuv).unwrap();
        let (ow, oh) = ((w + 1) & !1, (h + 1) & !1);
        assert_eq!(
            data.len(),
            ow * oh + 2 * (ow / 2) * (oh / 2),
            "{w}x{h}: ffmpeg output geometry"
        );
        let mut n = 0usize;
        for y in 0..h {
            for x in 0..w {
                if u16::from(data[y * ow + x]) != rec.luma.samples[y * rec.luma.stride + x] {
                    n += 1;
                }
            }
        }
        let cb0 = ow * oh;
        let cr0 = cb0 + (ow / 2) * (oh / 2);
        for y in 0..rec.cb.height {
            for x in 0..rec.cb.width {
                if u16::from(data[cb0 + y * (ow / 2) + x]) != rec.cb.samples[y * rec.cb.stride + x]
                {
                    n += 1;
                }
                if u16::from(data[cr0 + y * (ow / 2) + x]) != rec.cr.samples[y * rec.cr.stride + x]
                {
                    n += 1;
                }
            }
        }
        assert_eq!(
            n, 0,
            "{w}x{h}: ffmpeg output differs from the encoder reconstruction"
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}
