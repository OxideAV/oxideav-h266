//! r490 — the video encoder (`video_encoder::VideoEncoder`): IDR + P
//! pictures in one Annex B stream.
//!
//! Every stream is decoded by this crate's decoder from scratch and
//! must reproduce, picture for picture, the references the encoder
//! predicted from; PSNR floors make a silent drift a failure too. A
//! black-box decoder binary (an opaque process, skipped where it is
//! not installed) must output the same samples. The rate tests compare
//! the P coding against the all-intra coding of the same pictures.

use oxideav_h266::encoder::EncoderConfig;
use oxideav_h266::reconstruct::{PictureBuffer, PicturePlane};
use oxideav_h266::stream::StreamDecoder;
use oxideav_h266::video_encoder::{PictureKind, VideoEncoder, VideoEncoderConfig};

/// Deterministic noise in `0..n`.
fn hash(a: u32, b: u32, c: u32) -> u32 {
    let mut v = a
        .wrapping_mul(0x9E37_79B1)
        .wrapping_add(b.wrapping_mul(0x85EB_CA77))
        .wrapping_add(c.wrapping_mul(0xC2B2_AE3D));
    v ^= v >> 15;
    v = v.wrapping_mul(0x2C1B_3C6D);
    v ^= v >> 12;
    v
}

fn plane(w: usize, h: usize, bd: u32) -> PicturePlane {
    PicturePlane::filled_bd(w, h, 0, bd)
}

/// A textured background panning by `(pan_x, pan_y)` quarter samples
/// per picture, a block moving the other way, and a little noise.
fn clip_frame(w: usize, h: usize, cf: u32, bd: u32, t: usize, seed: u32) -> PictureBuffer {
    let (sw, sh) = match cf {
        1 => (2, 2),
        2 => (2, 1),
        _ => (1, 1),
    };
    let scale = f64::from(1u32 << (bd - 8));
    let maxv = f64::from((1u32 << bd) - 1);
    let tex = |x: f64, y: f64| -> f64 {
        110.0
            + 50.0 * (x * 0.093).sin() * (y * 0.071).cos()
            + 28.0 * ((x + 1.7 * y) * 0.23).sin()
            + 14.0 * ((x * 0.61).cos() + (y * 0.47).sin())
    };
    let pan_x = 0.75 * t as f64;
    let pan_y = 0.25 * t as f64;
    let (bx, by) = (
        w as f64 * 0.55 - 2.0 * t as f64,
        h as f64 * 0.3 + 1.0 * t as f64,
    );
    let (bw, bh) = (w as f64 * 0.22, h as f64 * 0.28);
    let sample = |x: f64, y: f64| -> (f64, f64, f64) {
        if x >= bx && x < bx + bw && y >= by && y < by + bh {
            let (u, v) = (x - bx, y - by);
            (
                150.0 + 40.0 * ((u * 0.4).sin() * (v * 0.3).cos()),
                90.0 + 0.5 * u,
                170.0 - 0.5 * v,
            )
        } else {
            let (u, v) = (x + pan_x, y + pan_y);
            (
                tex(u, v),
                128.0 + 30.0 * (u * 0.05).sin(),
                128.0 + 30.0 * (v * 0.06).cos(),
            )
        }
    };
    let mut luma = plane(w, h, bd);
    for y in 0..h {
        for x in 0..w {
            let n = f64::from(hash(x as u32, y as u32, seed + t as u32) % 5) - 2.0;
            let v = (sample(x as f64, y as f64).0 + n) * scale;
            luma.samples[y * luma.stride + x] = v.clamp(0.0, maxv) as u16;
        }
    }
    let (cw, ch) = if cf == 0 { (0, 0) } else { (w / sw, h / sh) };
    let mut cb = plane(cw.max(1), ch.max(1), bd);
    let mut cr = plane(cw.max(1), ch.max(1), bd);
    if cf == 0 {
        cb = PicturePlane::filled_bd(w / 2, h / 2, 1 << (bd - 1), bd);
        cr = cb.clone();
    } else {
        for y in 0..ch {
            for x in 0..cw {
                let (_, u, v) = sample((x * sw) as f64 + 0.5, (y * sh) as f64 + 0.5);
                cb.samples[y * cb.stride + x] = (u * scale).clamp(0.0, maxv) as u16;
                cr.samples[y * cr.stride + x] = (v * scale).clamp(0.0, maxv) as u16;
            }
        }
    }
    PictureBuffer { luma, cb, cr }
}

fn picture_cfg(w: usize, h: usize, cf: u32, bd: u32) -> EncoderConfig {
    let mut cfg = EncoderConfig::new(w as u32, h as u32);
    cfg.chroma_format_idc = cf as u8;
    cfg.bit_depth = bd as u8;
    cfg
}

fn psnr(a: &PicturePlane, b: &PicturePlane, bd: u32) -> f64 {
    let mut sse = 0f64;
    for y in 0..a.height {
        for x in 0..a.width {
            let d = f64::from(a.samples[y * a.stride + x]) - f64::from(b.samples[y * b.stride + x]);
            sse += d * d;
        }
    }
    if sse == 0.0 {
        return 99.0;
    }
    let peak = f64::from((1u32 << bd) - 1);
    10.0 * (peak * peak * (a.width * a.height) as f64 / sse).log10()
}

struct Coded {
    stream: Vec<u8>,
    sizes: Vec<usize>,
    kinds: Vec<PictureKind>,
    recs: Vec<PictureBuffer>,
    psnr_y: f64,
    psnr_c: f64,
    skip: u32,
    merge: u32,
    amvp: u32,
    alf_ctbs: u32,
}

impl Coded {
    fn bytes(&self) -> usize {
        self.sizes.iter().sum()
    }
}

/// Encode `frames`, check the reconstruction against the sources and
/// decode the stream from scratch against the encoder's references.
fn encode_and_check(cfg: VideoEncoderConfig, frames: &[PictureBuffer], label: &str) -> Coded {
    let bd = u32::from(cfg.picture.bit_depth);
    let chroma = cfg.picture.chroma_format_idc != 0;
    let mut enc = VideoEncoder::new(cfg).unwrap_or_else(|e| panic!("{label}: {e}"));
    let mut out = Coded {
        stream: Vec::new(),
        sizes: Vec::new(),
        kinds: Vec::new(),
        recs: Vec::new(),
        psnr_y: 0.0,
        psnr_c: 0.0,
        skip: 0,
        merge: 0,
        amvp: 0,
        alf_ctbs: 0,
    };
    for (t, src) in frames.iter().enumerate() {
        let pic = enc
            .encode(src)
            .unwrap_or_else(|e| panic!("{label}: picture {t}: {e}"));
        out.sizes.push(pic.data.len());
        out.kinds.push(pic.kind);
        out.skip += pic.stats.skip_cus;
        out.merge += pic.stats.merge_cus;
        out.amvp += pic.stats.amvp_cus;
        out.alf_ctbs += pic.stats.alf_ctbs;
        out.stream.extend_from_slice(&pic.data);
        let rec = enc.reconstruction().expect("reconstruction");
        assert_eq!(rec.luma.width, src.luma.width, "{label}");
        assert_eq!(rec.luma.height, src.luma.height, "{label}");
        out.psnr_y += psnr(&src.luma, &rec.luma, bd);
        if chroma {
            out.psnr_c += (psnr(&src.cb, &rec.cb, bd) + psnr(&src.cr, &rec.cr, bd)) / 2.0;
        }
        out.recs.push(rec);
    }
    out.psnr_y /= frames.len() as f64;
    out.psnr_c /= frames.len() as f64;

    // A fresh decoder over the whole stream.
    let mut dec = StreamDecoder::new();
    let mut pics = Vec::new();
    dec.decode_annex_b(&out.stream, &mut |p| pics.push(p))
        .unwrap_or_else(|e| panic!("{label}: decode: {e}"));
    assert_eq!(pics.len(), frames.len(), "{label}: picture count");
    for (t, (pic, rec)) in pics.iter().zip(&out.recs).enumerate() {
        let planes: &[(&PicturePlane, &PicturePlane)] = if chroma {
            &[
                (&pic.frame.luma, &rec.luma),
                (&pic.frame.cb, &rec.cb),
                (&pic.frame.cr, &rec.cr),
            ]
        } else {
            &[(&pic.frame.luma, &rec.luma)]
        };
        for (c, (got, want)) in planes.iter().enumerate() {
            for y in 0..want.height {
                assert!(
                    got.samples[y * got.stride..y * got.stride + want.width]
                        == want.samples[y * want.stride..y * want.stride + want.width],
                    "{label}: picture {t} component {c} row {y} differs from the encoder reference"
                );
            }
        }
    }
    out
}

fn clip(w: usize, h: usize, cf: u32, bd: u32, n: usize, seed: u32) -> Vec<PictureBuffer> {
    (0..n).map(|t| clip_frame(w, h, cf, bd, t, seed)).collect()
}

#[test]
fn p_pictures_round_trip_across_sizes_and_qps() {
    for &(w, h, qp) in &[
        (8usize, 8usize, 30),
        (16, 16, 20),
        (64, 64, 40),
        (128, 128, 30),
        (136, 72, 20),
        (200, 130, 40),
        (330, 270, 30),
    ] {
        let frames = clip(w, h, 1, 8, 3, 7);
        let cfg = VideoEncoderConfig::new(picture_cfg(w, h, 1, 8), qp)
            .with_keyint(16)
            .with_scene_cut(false);
        let label = format!("{w}x{h} qp {qp}");
        let coded = encode_and_check(cfg, &frames, &label);
        assert_eq!(coded.kinds[0], PictureKind::Idr, "{label}");
        assert!(
            coded.kinds[1..].iter().all(|k| *k == PictureKind::P),
            "{label}: {:?}",
            coded.kinds
        );
        let floor = match qp {
            20 => 38.0,
            30 => 31.0,
            _ => 25.0,
        };
        assert!(
            coded.psnr_y > floor,
            "{label}: luma PSNR {:.2} under {floor}",
            coded.psnr_y
        );
        assert!(
            coded.psnr_c > floor,
            "{label}: chroma PSNR {:.2} under {floor}",
            coded.psnr_c
        );
    }
}

#[test]
fn p_pictures_round_trip_in_every_chroma_format_and_depth() {
    for &cf in &[0u32, 1, 2, 3] {
        for &bd in &[8u32, 10] {
            let (w, h) = (104usize, 72usize);
            let frames = clip(w, h, cf, bd, 3, 11 + cf);
            let cfg = VideoEncoderConfig::new(picture_cfg(w, h, cf, bd), 28)
                .with_keyint(8)
                .with_scene_cut(false);
            let label = format!("cf {cf} {bd}-bit");
            let coded = encode_and_check(cfg, &frames, &label);
            assert!(
                coded.kinds[1..].iter().all(|k| *k == PictureKind::P),
                "{label}"
            );
            assert!(coded.psnr_y > 32.0, "{label}: luma {:.2}", coded.psnr_y);
            if cf != 0 {
                assert!(coded.psnr_c > 32.0, "{label}: chroma {:.2}", coded.psnr_c);
            }
        }
    }
}

#[test]
fn every_effort_step_is_conformant_and_higher_effort_costs_no_quality() {
    let (w, h) = (192usize, 128usize);
    let frames = clip(w, h, 1, 8, 4, 3);
    let mut rows = Vec::new();
    for effort in 0..=oxideav_h266::video_encoder::MAX_EFFORT {
        let cfg = VideoEncoderConfig::new(picture_cfg(w, h, 1, 8), 30)
            .with_keyint(32)
            .with_effort(effort)
            .with_scene_cut(false);
        let coded = encode_and_check(cfg, &frames, &format!("effort {effort}"));
        eprintln!(
            "effort {effort}: {} B, luma {:.2} dB, chroma {:.2} dB, skip/merge/amvp {}/{}/{}",
            coded.bytes(),
            coded.psnr_y,
            coded.psnr_c,
            coded.skip,
            coded.merge,
            coded.amvp
        );
        rows.push((coded.bytes(), coded.psnr_y));
    }
    // The top of the ladder must not be worse than the bottom on both
    // axes at once.
    let (b0, p0) = rows[0];
    let (b4, p4) = rows[rows.len() - 1];
    assert!(
        (b4 as f64) < b0 as f64 * 1.02 || p4 > p0 + 0.1,
        "effort 4 ({b4} B, {p4:.2} dB) is not better than effort 0 ({b0} B, {p0:.2} dB)"
    );
}

#[test]
fn inter_coding_beats_all_intra_on_moving_content() {
    let (w, h) = (192usize, 128usize);
    let frames = clip(w, h, 1, 8, 6, 5);
    for &qp in &[26, 36] {
        let intra = encode_and_check(
            VideoEncoderConfig::new(picture_cfg(w, h, 1, 8), qp).with_keyint(1),
            &frames,
            &format!("intra qp {qp}"),
        );
        let inter = encode_and_check(
            VideoEncoderConfig::new(picture_cfg(w, h, 1, 8), qp)
                .with_keyint(64)
                .with_scene_cut(false),
            &frames,
            &format!("inter qp {qp}"),
        );
        assert!(intra.kinds.iter().all(|k| *k == PictureKind::Idr));
        eprintln!(
            "qp {qp}: intra {} B {:.2}/{:.2} dB -> inter {} B {:.2}/{:.2} dB ({:+.1} %)",
            intra.bytes(),
            intra.psnr_y,
            intra.psnr_c,
            inter.bytes(),
            inter.psnr_y,
            inter.psnr_c,
            100.0 * (inter.bytes() as f64 / intra.bytes() as f64 - 1.0)
        );
        assert!(
            (inter.bytes() as f64) < intra.bytes() as f64 * 0.6,
            "qp {qp}: inter {} B vs intra {} B",
            inter.bytes(),
            intra.bytes()
        );
        assert!(
            inter.psnr_y > intra.psnr_y - 2.0,
            "qp {qp}: inter luma {:.2} dB vs intra {:.2} dB",
            inter.psnr_y,
            intra.psnr_y
        );
    }
}

/// The adaptive loop filter decision of the P pictures: CTBs elect a
/// fixed filter set, the stream stays conformant (fresh decode against
/// the encoder's references inside `encode_and_check`), the luma PSNR
/// rises at (nearly) the same size, and switching the decision off or
/// coding at effort 0 leaves every P slice unfiltered.
#[test]
fn adaptive_loop_filter_on_p_pictures_raises_luma_psnr() {
    for &(w, h, cf, bd, qp) in &[
        (256usize, 192usize, 1u32, 8u32, 32),
        (200, 136, 3, 10, 36),
        (144, 144, 0, 8, 30),
    ] {
        let frames = clip(w, h, cf, bd, 5, 77);
        let base = VideoEncoderConfig::new(picture_cfg(w, h, cf, bd), qp).with_scene_cut(false);
        let label = format!("{w}x{h} cf{cf} {bd}-bit qp {qp}");
        let on = encode_and_check(base, &frames, &format!("{label} alf"));
        let off = encode_and_check(base.with_alf(false), &frames, &format!("{label} no alf"));
        eprintln!(
            "{label}: no filter {} B {:.2} dB -> filter {} B {:.2} dB ({} CTBs filtered)",
            off.bytes(),
            off.psnr_y,
            on.bytes(),
            on.psnr_y,
            on.alf_ctbs
        );
        assert_eq!(off.alf_ctbs, 0, "{label}");
        assert!(on.alf_ctbs > 0, "{label}: no CTB elected a filter");
        assert!(
            on.psnr_y > off.psnr_y + 0.005,
            "{label}: {:.3} dB with the filter, {:.3} dB without",
            on.psnr_y,
            off.psnr_y
        );
        assert!(
            (on.bytes() as f64) < off.bytes() as f64 * 1.03,
            "{label}: {} B with the filter, {} B without",
            on.bytes(),
            off.bytes()
        );
        let fast = encode_and_check(base.with_effort(0), &frames, &format!("{label} effort 0"));
        assert_eq!(
            fast.alf_ctbs, 0,
            "{label}: effort 0 runs no filter decision"
        );
    }
}

/// The IDR QP offset: the IDR pictures of a sequence with P pictures
/// are coded finer than the P pictures by default, an offset of 0
/// codes them at the configured QP, and both streams are conformant.
#[test]
fn idr_qp_offset_spends_bits_on_the_idr_picture() {
    let (w, h) = (192usize, 128usize);
    let frames = clip(w, h, 1, 8, 6, 5);
    let base = VideoEncoderConfig::new(picture_cfg(w, h, 1, 8), 34).with_scene_cut(false);
    assert_eq!(base.idr_qp_offset, -2);
    let finer = encode_and_check(base, &frames, "idr offset default");
    let flat = encode_and_check(base.with_idr_qp_offset(0), &frames, "idr offset 0");
    assert_eq!(finer.kinds[0], PictureKind::Idr);
    assert!(
        finer.sizes[0] > flat.sizes[0],
        "IDR picture {} B at the default offset, {} B at offset 0",
        finer.sizes[0],
        flat.sizes[0]
    );
    assert!(
        finer.psnr_y > flat.psnr_y,
        "{:.2} dB at the default offset, {:.2} dB at offset 0",
        finer.psnr_y,
        flat.psnr_y
    );
}

#[test]
fn static_content_is_coded_as_skip() {
    let (w, h) = (256usize, 256usize);
    let one = clip_frame(w, h, 1, 8, 0, 9);
    let frames = vec![one.clone(), one.clone(), one];
    let cfg = VideoEncoderConfig::new(picture_cfg(w, h, 1, 8), 30).with_keyint(8);
    let coded = encode_and_check(cfg, &frames, "static");
    // Four 128x128 CTBs, each one skip CU, a handful of bytes.
    for (t, &size) in coded.sizes.iter().enumerate().skip(1) {
        assert!(size < 40, "static picture {t} took {size} bytes");
    }
    assert_eq!(coded.merge + coded.amvp, 0);
}

#[test]
fn keyint_and_scene_changes_place_the_idr_pictures() {
    let (w, h) = (128usize, 96usize);
    let mut frames = clip(w, h, 1, 8, 4, 21);
    // A hard cut to unrelated noise-like content, then motion again.
    for t in 0..4 {
        let mut f = clip_frame(w, h, 1, 8, t, 99);
        for (i, v) in f.luma.samples.iter_mut().enumerate() {
            *v = (hash(i as u32, 17, 5) % 200 + 20) as u16;
        }
        if t > 0 {
            // Pan the noise by whole samples so it stays predictable.
            let base = f.luma.clone();
            for y in 0..h {
                for x in 0..w {
                    let sx = (x + 2 * t) % w;
                    f.luma.samples[y * w + x] = base.samples[y * w + sx];
                }
            }
        }
        frames.push(f);
    }
    let cfg = VideoEncoderConfig::new(picture_cfg(w, h, 1, 8), 30).with_keyint(6);
    let coded = encode_and_check(cfg, &frames, "scene cut");
    assert_eq!(coded.kinds[0], PictureKind::Idr);
    assert_eq!(coded.kinds[4], PictureKind::Idr, "{:?}", coded.kinds);
    assert!(coded.kinds[1..4].iter().all(|k| *k == PictureKind::P));
    // keyint counts from the scene-change IDR.
    let cfg = VideoEncoderConfig::new(picture_cfg(w, h, 1, 8), 30)
        .with_keyint(3)
        .with_scene_cut(false);
    let coded = encode_and_check(cfg, &clip(w, h, 1, 8, 7, 2), "keyint 3");
    let idr: Vec<usize> = coded
        .kinds
        .iter()
        .enumerate()
        .filter(|(_, k)| **k == PictureKind::Idr)
        .map(|(i, _)| i)
        .collect();
    assert_eq!(idr, vec![0, 3, 6]);
}

#[test]
fn unsupported_inter_configurations_are_typed() {
    let mut p = picture_cfg(128, 128, 1, 8);
    p.lossless = true;
    assert!(VideoEncoder::new(VideoEncoderConfig::new(p, 30)).is_err());
    // … and fine as all-intra.
    assert!(VideoEncoder::new(VideoEncoderConfig::new(p, 30).with_keyint(1)).is_ok());
    let mut p = picture_cfg(256, 128, 1, 8);
    p.tile_columns = 2;
    assert!(VideoEncoder::new(VideoEncoderConfig::new(p, 30)).is_err());
    assert!(VideoEncoder::new(VideoEncoderConfig::new(picture_cfg(64, 64, 1, 8), 64)).is_err());
}

/// Black-box validation through an installed decoder binary (opaque
/// process; a no-op where it is absent): it outputs the pictures the
/// encoder predicted from.
#[test]
fn black_box_decoder_agrees_on_inter_streams() {
    let probe = std::process::Command::new("ffmpeg")
        .arg("-version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status();
    if !matches!(probe, Ok(s) if s.success()) {
        eprintln!("black-box decoder not installed — leg skipped");
        return;
    }
    let dir = std::env::temp_dir().join(format!("oxideav-h266-r490-inter-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let cases: &[(usize, usize, u32, u32, i32, u8)] = &[
        (256, 192, 1, 8, 28, 2),
        (200, 130, 1, 8, 34, 4),
        (144, 104, 1, 10, 26, 2),
        (104, 72, 0, 8, 30, 2),
        (104, 72, 2, 8, 30, 2),
        (104, 72, 3, 10, 30, 3),
        (64, 64, 1, 8, 22, 0),
    ];
    for &(w, h, cf, bd, qp, effort) in cases {
        let frames = clip(w, h, cf, bd, 5, 40 + cf + bd);
        let cfg = VideoEncoderConfig::new(picture_cfg(w, h, cf, bd), qp)
            .with_keyint(4)
            .with_effort(effort)
            .with_scene_cut(false);
        let label = format!("{w}x{h} cf {cf} {bd}-bit qp {qp} effort {effort}");
        let coded = encode_and_check(cfg, &frames, &label);
        let pix_fmt = match (cf, bd) {
            (0, 8) => "gray",
            (0, _) => "gray10le",
            (1, 8) => "yuv420p",
            (1, _) => "yuv420p10le",
            (2, 8) => "yuv422p",
            (2, _) => "yuv422p10le",
            (_, 8) => "yuv444p",
            (_, _) => "yuv444p10le",
        };
        let bit = dir.join(format!("{w}_{h}_{cf}_{bd}.266"));
        let yuv = dir.join(format!("{w}_{h}_{cf}_{bd}.yuv"));
        std::fs::write(&bit, &coded.stream).unwrap();
        let st = std::process::Command::new("ffmpeg")
            .args(["-hide_banner", "-loglevel", "error", "-y", "-i"])
            .arg(&bit)
            .args([
                "-fps_mode",
                "passthrough",
                "-f",
                "rawvideo",
                "-pix_fmt",
                pix_fmt,
            ])
            .arg(&yuv)
            .status()
            .unwrap();
        assert!(st.success(), "{label}: black-box decode failed");
        let raw = std::fs::read(&yuv).unwrap();
        let data: Vec<u16> = if bd > 8 {
            raw.as_chunks::<2>()
                .0
                .iter()
                .map(|c| u16::from_le_bytes(*c))
                .collect()
        } else {
            raw.iter().map(|&b| u16::from(b)).collect()
        };
        let mut expect = Vec::with_capacity(data.len());
        for rec in &coded.recs {
            let planes: &[&PicturePlane] = if cf == 0 {
                &[&rec.luma]
            } else {
                &[&rec.luma, &rec.cb, &rec.cr]
            };
            for p in planes {
                for y in 0..p.height {
                    expect.extend_from_slice(&p.samples[y * p.stride..y * p.stride + p.width]);
                }
            }
        }
        assert_eq!(data.len(), expect.len(), "{label}: black-box output size");
        assert!(data == expect, "{label}: black-box output differs");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// Decode a staged conformance stream into source pictures (cropped
/// output); `None` without the docs tree.
fn corpus_clip(name: &str, max: usize) -> Option<(Vec<PictureBuffer>, u32, u32)> {
    let dir = std::env::var("H266_CONFORMANCE_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../docs/video/h266/conformance/streams")
        });
    let bs = std::fs::read(dir.join(format!("{name}.bit"))).ok()?;
    let mut dec = StreamDecoder::new();
    let mut pics = Vec::new();
    dec.decode_annex_b(&bs, &mut |p| pics.push(p)).ok()?;
    pics.sort_by_key(|p| (p.cvs_idx, p.poc));
    let first = pics.first()?;
    let (cf, bd) = (u32::from(first.chroma_format_idc), first.bit_depth);
    let frames = pics
        .into_iter()
        .take(max)
        .map(|p| {
            let (l, t, w, h) = p.crop;
            let (sw, sh) = match cf {
                1 => (2, 2),
                2 => (2, 1),
                _ => (1, 1),
            };
            let cut = |pl: &PicturePlane, l: usize, t: usize, w: usize, h: usize| {
                let mut out = PicturePlane::filled_bd(w, h, 0, pl.bit_depth);
                for y in 0..h {
                    out.samples[y * w..(y + 1) * w].copy_from_slice(
                        &pl.samples[(y + t) * pl.stride + l..(y + t) * pl.stride + l + w],
                    );
                }
                out
            };
            PictureBuffer {
                luma: cut(&p.frame.luma, l, t, w, h),
                cb: cut(&p.frame.cb, l / sw, t / sh, w / sw, h / sh),
                cr: cut(&p.frame.cr, l / sw, t / sh, w / sw, h / sh),
            }
        })
        .collect();
    Some((frames, cf, bd))
}

/// Rate of curve `pts` (bytes, dB; any order) at `psnr`, interpolated
/// in the log-rate domain between the two bracketing points.
fn rate_at(pts: &[(f64, f64)], psnr: f64) -> Option<f64> {
    let mut pts = pts.to_vec();
    pts.sort_by(|a, b| a.1.total_cmp(&b.1));
    for w in pts.windows(2) {
        let ((r0, p0), (r1, p1)) = (w[0], w[1]);
        if psnr >= p0 && psnr <= p1 && p1 > p0 {
            let f = (psnr - p0) / (p1 - p0);
            return Some((r0.ln() + f * (r1.ln() - r0.ln())).exp());
        }
    }
    None
}

/// Measurement: bitrate at equal luma PSNR, P coding against all-intra,
/// on pictures decoded from the staged conformance streams, and the
/// encode speed at the default effort. Runs when the docs tree is
/// present; the assertion is a floor, the numbers go to the README.
#[test]
fn inter_gain_on_decoded_corpus_clips() {
    let full = std::env::var_os("H266_R490_MEASURE").is_some();
    if cfg!(debug_assertions) && !full {
        eprintln!("unoptimised build — measurement skipped (set H266_R490_MEASURE to force it)");
        return;
    }
    let clips: &[(&str, usize)] = if full {
        &[("RAP_B_1", 17), ("8b420_A_2", 9)]
    } else {
        &[("RAP_B_1", 6)]
    };
    for &(name, n) in clips {
        let Some((frames, cf, bd)) = corpus_clip(name, n) else {
            eprintln!("{name}: conformance corpus not available — skipped");
            continue;
        };
        let (w, h) = (frames[0].luma.width, frames[0].luma.height);
        let qps: &[i32] = if full { &[22, 27, 32, 37] } else { &[32] };
        let intra_qps: &[i32] = if full {
            &[18, 22, 27, 32, 37, 42]
        } else {
            &[26, 30, 34, 38]
        };
        let mut intra_curve = Vec::new();
        for &qp in intra_qps {
            let c = encode_and_check(
                VideoEncoderConfig::new(picture_cfg(w, h, cf, bd), qp).with_keyint(1),
                &frames,
                &format!("{name} intra qp {qp}"),
            );
            intra_curve.push((c.bytes() as f64, c.psnr_y));
        }
        let mut savings = Vec::new();
        for &qp in qps {
            let t0 = std::time::Instant::now();
            let c = encode_and_check(
                VideoEncoderConfig::new(picture_cfg(w, h, cf, bd), qp)
                    .with_keyint(64)
                    .with_scene_cut(false),
                &frames,
                &format!("{name} inter qp {qp}"),
            );
            let secs = t0.elapsed().as_secs_f64();
            let at = rate_at(&intra_curve, c.psnr_y);
            let saving = at.map(|r| 100.0 * (c.bytes() as f64 / r - 1.0));
            eprintln!(
                "{name} {w}x{h} x{n} qp {qp}: {} B, {:.2} dB luma / {:.2} dB chroma; all-intra at the \
                 same luma PSNR: {} B ({}); {:.2} s for encode + checks; skip/merge/amvp {}/{}/{}",
                c.bytes(),
                c.psnr_y,
                c.psnr_c,
                at.map(|r| format!("{r:.0}"))
                    .unwrap_or_else(|| "n/a".into()),
                saving
                    .map(|s| format!("{s:+.1} %"))
                    .unwrap_or_else(|| "outside the intra curve".into()),
                secs,
                c.skip,
                c.merge,
                c.amvp
            );
            if let Some(s) = saving {
                savings.push(s);
            }
        }
        assert!(!savings.is_empty(), "{name}: no comparable point");
        let mean = savings.iter().sum::<f64>() / savings.len() as f64;
        eprintln!("{name}: mean bitrate change at equal luma PSNR {mean:+.1} %");
        assert!(mean < -20.0, "{name}: inter coding saves only {mean:+.1} %");
    }
}
