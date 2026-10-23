//! r477 — the registry encoder (`oxideav_core::Encoder`): every
//! declared layout round-trips through the registry encoder and the
//! registry decoder in both framings, the `quality` ladder is
//! monotone and ends lossless, the `vvcC` extradata describes the
//! stream, refusals are typed, and a black-box decoder (where
//! installed) agrees with the registry decoder on the `vvcc` samples
//! re-wrapped as Annex B.

use oxideav_core::{
    CodecId, CodecOptions, CodecParameters, Error, Frame, Packet, PixelFormat, Rational, TimeBase,
    VideoFrame, VideoPlane,
};
use oxideav_h266::h266_encoder::accepted_pixel_formats;
use oxideav_h266::vvcc::VvcDecoderConfigurationRecord;

fn registry() -> oxideav_core::RuntimeContext {
    let mut ctx = oxideav_core::RuntimeContext::new();
    oxideav_h266::register(&mut ctx);
    ctx
}

fn layout(pf: PixelFormat) -> (u32, u32) {
    // (chroma_format_idc, bit depth)
    match pf {
        PixelFormat::Gray8 => (0, 8),
        PixelFormat::Gray10Le => (0, 10),
        PixelFormat::Yuv420P => (1, 8),
        PixelFormat::Yuv420P10Le => (1, 10),
        PixelFormat::Yuv422P => (2, 8),
        PixelFormat::Yuv422P10Le => (2, 10),
        PixelFormat::Yuv444P => (3, 8),
        PixelFormat::Yuv444P10Le => (3, 10),
        other => panic!("not an encoder layout: {other:?}"),
    }
}

/// A textured test picture in `pf` as a framework frame.
fn textured_frame(w: usize, h: usize, pf: PixelFormat, seed: usize) -> VideoFrame {
    let (cf, bd) = layout(pf);
    let wide = bd > 8;
    let sh = bd - 8;
    let put = |v: u32, out: &mut Vec<u8>| {
        if wide {
            out.extend_from_slice(&((v << sh) as u16).to_le_bytes());
        } else {
            out.push(v as u8);
        }
    };
    let mut y = Vec::new();
    for yy in 0..h {
        for x in 0..w {
            let base = 40 + (x * 150) / w.max(1) + (yy * 60) / h.max(1);
            let ripple = ((x * 5 + yy * 3 + seed) % 17) * 3;
            let edge = if x + yy > (w + h) / 2 { 30 } else { 0 };
            let dx = x as i64 - (w / 3) as i64;
            let dy = yy as i64 - (h / 3) as i64;
            let disc = if dx * dx + dy * dy < ((w.min(h) / 6) as i64).pow(2) {
                60
            } else {
                0
            };
            put(
                ((base + ripple + edge).saturating_sub(disc)).min(255) as u32,
                &mut y,
            );
        }
    }
    let bps = if wide { 2 } else { 1 };
    let mut planes = vec![VideoPlane {
        stride: w * bps,
        data: y,
    }];
    if cf != 0 {
        let (sw, shh) = match cf {
            1 => (2, 2),
            2 => (2, 1),
            _ => (1, 1),
        };
        let (cw, ch) = (w / sw, h / shh);
        let mut cb = Vec::new();
        let mut cr = Vec::new();
        for yy in 0..ch {
            for x in 0..cw {
                put((90 + (x * 7 + yy * 2 + seed) % 60) as u32, &mut cb);
                put((150 - ((x * 3 + yy * 5) % 50)) as u32, &mut cr);
            }
        }
        planes.push(VideoPlane {
            stride: cw * bps,
            data: cb,
        });
        planes.push(VideoPlane {
            stride: cw * bps,
            data: cr,
        });
    }
    VideoFrame { pts: None, planes }
}

fn params(w: u32, h: u32, pf: PixelFormat, opts: CodecOptions) -> CodecParameters {
    let mut p = CodecParameters::video(CodecId::new("h266"));
    p.width = Some(w);
    p.height = Some(h);
    p.pixel_format = Some(pf);
    p.frame_rate = Some(Rational::new(30, 1));
    p.options = opts;
    p
}

/// Encode `frames` through the registry; returns the packets and the
/// encoder's output parameters.
fn encode(p: &CodecParameters, frames: &[VideoFrame]) -> (Vec<Packet>, CodecParameters) {
    let ctx = registry();
    let mut enc = ctx.codecs.first_encoder(p).expect("encoder");
    let mut out = Vec::new();
    for f in frames {
        enc.send_frame(&Frame::Video(f.clone()))
            .expect("send_frame");
        while let Ok(pk) = enc.receive_packet() {
            out.push(pk);
        }
    }
    enc.flush().unwrap();
    while let Ok(pk) = enc.receive_packet() {
        out.push(pk);
    }
    (out, enc.output_params().clone())
}

/// Decode packets through the registry decoder built from `outp`.
fn decode(outp: &CodecParameters, packets: &[Packet]) -> (Vec<VideoFrame>, Option<PixelFormat>) {
    let ctx = registry();
    let mut dec = ctx.codecs.first_decoder(outp).expect("decoder");
    let mut frames = Vec::new();
    for pk in packets {
        dec.send_packet(pk).expect("send_packet");
    }
    dec.flush().unwrap();
    loop {
        match dec.receive_frame() {
            Ok(Frame::Video(v)) => frames.push(v),
            Ok(_) => panic!("non-video"),
            Err(Error::NeedMore) | Err(Error::Eof) => break,
            Err(e) => panic!("{e}"),
        }
    }
    let pf = dec.output_pixel_format();
    (frames, pf)
}

fn psnr(a: &[u8], b: &[u8], wide: bool, bd: u32) -> f64 {
    let to: fn(&[u8]) -> Vec<f64> = |d| d.iter().map(|&v| f64::from(v)).collect();
    let (a, b) = if wide {
        let w = |d: &[u8]| {
            d.as_chunks::<2>()
                .0
                .iter()
                .map(|c| f64::from(u16::from_le_bytes([c[0], c[1]])))
                .collect::<Vec<f64>>()
        };
        (w(a), w(b))
    } else {
        (to(a), to(b))
    };
    let mse: f64 = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        / a.len() as f64;
    if mse == 0.0 {
        return f64::INFINITY;
    }
    let peak = f64::from((1u32 << bd) - 1);
    10.0 * (peak * peak / mse).log10()
}

#[test]
fn every_declared_layout_round_trips_in_both_framings() {
    let ctx = registry();
    let imp = &ctx.codecs.implementations(&CodecId::new("h266"))[0];
    assert!(imp.caps.encode);
    assert_eq!(
        imp.caps.accepted_pixel_formats,
        accepted_pixel_formats().to_vec()
    );
    assert!(imp.encoder_options_schema.is_some());
    for &pf in accepted_pixel_formats() {
        let (cf, bd) = layout(pf);
        let wide = bd > 8;
        let (w, h) = (48u32, 32u32);
        for format in ["annexb", "vvcc"] {
            let opts = CodecOptions::new().set("format", format).set("qp", "27");
            let p = params(w, h, pf, opts);
            let f0 = textured_frame(w as usize, h as usize, pf, 0);
            let f1 = textured_frame(w as usize, h as usize, pf, 5);
            let (packets, outp) = encode(&p, &[f0.clone(), f1.clone()]);
            assert_eq!(packets.len(), 2, "{pf:?} {format}");
            assert_eq!(packets[0].pts, Some(0));
            assert_eq!(packets[1].pts, Some(1));
            assert_eq!(packets[1].dts, Some(1));
            assert_eq!(packets[0].duration, Some(1));
            assert!(packets[0].flags.keyframe && packets[1].flags.keyframe);
            assert_eq!(packets[0].time_base, TimeBase::new(1, 30));
            assert_eq!(outp.pixel_format, Some(pf));
            assert_eq!(outp.width, Some(w));
            if format == "vvcc" {
                let rec = VvcDecoderConfigurationRecord::parse(&outp.extradata).unwrap();
                let ptl = rec.ptl.as_ref().unwrap();
                assert_eq!(u32::from(ptl.chroma_format_idc), cf, "{pf:?}");
                assert_eq!(u32::from(ptl.bit_depth_minus8) + 8, bd, "{pf:?}");
                assert_eq!(rec.length_size(), 4);
                assert!(!rec.nal_units_of_type(15).is_empty(), "{pf:?}: SPS array");
                assert!(!rec.nal_units_of_type(16).is_empty(), "{pf:?}: PPS array");
                // Samples carry no parameter sets.
                let nals = oxideav_h266::nal::iter_length_prefixed(&packets[0].data, 4).unwrap();
                assert!(nals.iter().all(|n| !matches!(
                    n.header.nal_unit_type,
                    oxideav_h266::nal::NalUnitType::SpsNut
                        | oxideav_h266::nal::NalUnitType::PpsNut
                        | oxideav_h266::nal::NalUnitType::VpsNut
                )));
            } else {
                assert!(outp.extradata.is_empty());
                assert!(packets[0].data.starts_with(&[0, 0, 0, 1]));
            }
            let (frames, label) = decode(&outp, &packets);
            assert_eq!(frames.len(), 2, "{pf:?} {format}");
            assert_eq!(label, Some(pf), "{pf:?} {format}: output label");
            assert_eq!(frames[0].pts, Some(0));
            assert_eq!(frames[1].pts, Some(1));
            for (f, src) in frames.iter().zip([&f0, &f1]) {
                assert_eq!(f.image_plane_count(), src.image_plane_count());
                let q = psnr(&f.planes[0].data, &src.planes[0].data, wide, bd);
                assert!(q > 33.0, "{pf:?} {format}: luma PSNR {q:.2} dB at QP 27");
                if cf != 0 {
                    let qc = psnr(&f.planes[1].data, &src.planes[1].data, wide, bd);
                    assert!(qc > 33.0, "{pf:?} {format}: Cb PSNR {qc:.2} dB");
                }
            }
        }
    }
}

#[test]
fn quality_ladder_is_monotone_and_100_is_lossless() {
    let pf = PixelFormat::Yuv420P;
    let (w, h) = (64u32, 48u32);
    let src = textured_frame(w as usize, h as usize, pf, 3);
    let mut last_size = 0usize;
    let mut last_psnr = 0.0f64;
    for q in [10u32, 40, 70, 95] {
        let p = params(w, h, pf, CodecOptions::new().set("quality", q.to_string()));
        let (packets, outp) = encode(&p, std::slice::from_ref(&src));
        let size = packets[0].data.len();
        let (frames, _) = decode(&outp, &packets);
        let ps = psnr(&frames[0].planes[0].data, &src.planes[0].data, false, 8);
        // Higher quality: more bytes, higher fidelity.
        assert!(size > last_size, "quality {q}: {size} <= {last_size}");
        assert!(ps > last_psnr, "quality {q}: {ps:.2} <= {last_psnr:.2}");
        last_size = size;
        last_psnr = ps;
    }
    let p = params(w, h, pf, CodecOptions::new().set("quality", "100"));
    let (packets, outp) = encode(&p, std::slice::from_ref(&src));
    let (frames, _) = decode(&outp, &packets);
    for (a, b) in frames[0].planes.iter().zip(&src.planes) {
        assert_eq!(a.data, b.data, "quality=100 is lossless");
    }
    let p = params(w, h, pf, CodecOptions::new().set("lossless", "true"));
    let (packets2, _) = encode(&p, std::slice::from_ref(&src));
    assert_eq!(
        packets2[0].data, packets[0].data,
        "lossless=true == quality=100"
    );
}

#[test]
fn refusals_are_typed() {
    let pf = PixelFormat::Yuv420P;
    let ctx = registry();
    // 12-bit input: unsupported, not invalid.
    let p = params(32, 32, PixelFormat::Yuv420P12Le, CodecOptions::new());
    let e = ctx.codecs.first_encoder(&p).err().expect("12-bit refused");
    assert!(matches!(e, Error::Unsupported(_)), "{e}");
    // Odd 4:2:0 dimensions.
    let p = params(33, 32, pf, CodecOptions::new());
    assert!(ctx.codecs.first_encoder(&p).is_err());
    // 4:4:4 odd dimensions are fine.
    let p = params(33, 17, PixelFormat::Yuv444P, CodecOptions::new());
    assert!(ctx.codecs.first_encoder(&p).is_ok());
    // Unknown option / bad values.
    for (k, v) in [
        ("bogus", "1"),
        ("qp", "64"),
        ("quality", "101"),
        ("format", "mp4"),
        ("tile_columns", "0"),
    ] {
        let p = params(32, 32, pf, CodecOptions::new().set(k, v));
        assert!(ctx.codecs.first_encoder(&p).is_err(), "{k}={v}");
    }
    // Lossless excludes the MTT pickers and dep-quant.
    let p = params(
        32,
        32,
        pf,
        CodecOptions::new()
            .set("lossless", "true")
            .set("mtt", "true"),
    );
    assert!(ctx.codecs.first_encoder(&p).is_err());
    let p = params(
        32,
        32,
        pf,
        CodecOptions::new()
            .set("dep_quant", "true")
            .set("sign_data_hiding", "true"),
    );
    assert!(ctx.codecs.first_encoder(&p).is_err());
    // Wrong plane count at send time.
    let p = params(32, 32, pf, CodecOptions::new());
    let mut enc = ctx.codecs.first_encoder(&p).unwrap();
    let grey = textured_frame(32, 32, PixelFormat::Gray8, 0);
    assert!(enc.send_frame(&Frame::Video(grey)).is_err());
    assert!(matches!(enc.receive_packet(), Err(Error::NeedMore)));
}

#[test]
fn tools_and_tiles_round_trip_through_the_registry() {
    let pf = PixelFormat::Yuv420P;
    let (w, h) = (256u32, 128u32);
    let src = textured_frame(w as usize, h as usize, pf, 9);
    for (opts, floor) in [
        (CodecOptions::new().set("mtt", "true"), 30.0),
        (CodecOptions::new().set("dep_quant", "true"), 30.0),
        (CodecOptions::new().set("sign_data_hiding", "true"), 30.0),
        (
            CodecOptions::new()
                .set("tile_columns", "2")
                .set("wpp", "true")
                .set("format", "vvcc"),
            30.0,
        ),
        // The historical one-CU-per-64x64 layout caps near 25.7 dB on
        // textured content (r474 measurement) — it must still decode.
        (CodecOptions::new().set("rdo", "false"), 24.0),
    ] {
        let p = params(w, h, pf, opts.clone());
        let (packets, outp) = encode(&p, std::slice::from_ref(&src));
        let (frames, _) = decode(&outp, &packets);
        assert_eq!(frames.len(), 1, "{opts:?}");
        let ps = psnr(&frames[0].planes[0].data, &src.planes[0].data, false, 8);
        assert!(ps > floor, "{opts:?}: {ps:.2} dB");
    }
}

/// Black-box validation through `ffmpeg` (opaque process; no-op where
/// the binary is absent): the `vvcc` samples re-wrapped as Annex B
/// with the record's parameter sets decode to the registry decoder's
/// output for every layout.
#[test]
fn black_box_decoder_agrees_on_vvcc_samples() {
    let probe = std::process::Command::new("ffmpeg")
        .arg("-version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status();
    if !matches!(probe, Ok(s) if s.success()) {
        eprintln!("ffmpeg not installed — black-box leg skipped");
        return;
    }
    let dir = std::env::temp_dir().join(format!("oxideav-h266-r477-reg-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    for &pf in accepted_pixel_formats() {
        let (cf, bd) = layout(pf);
        let (w, h) = (64u32, 40u32);
        let src = textured_frame(w as usize, h as usize, pf, 1);
        let p = params(w, h, pf, CodecOptions::new().set("format", "vvcc"));
        let (packets, outp) = encode(&p, std::slice::from_ref(&src));
        let (frames, _) = decode(&outp, &packets);
        // Re-wrap: record parameter sets + the sample's NAL units.
        let rec = VvcDecoderConfigurationRecord::parse(&outp.extradata).unwrap();
        let mut annex_b = Vec::new();
        for nal in rec.nal_units() {
            annex_b.extend_from_slice(&[0, 0, 0, 1]);
            annex_b.extend_from_slice(nal);
        }
        for nal in oxideav_h266::nal::iter_length_prefixed(&packets[0].data, 4).unwrap() {
            annex_b.extend_from_slice(&[0, 0, 0, 1]);
            annex_b.extend_from_slice(nal.raw);
        }
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
        let bit = dir.join(format!("{pf:?}.266"));
        let yuv = dir.join(format!("{pf:?}.yuv"));
        std::fs::write(&bit, &annex_b).unwrap();
        let st = std::process::Command::new("ffmpeg")
            .args(["-hide_banner", "-loglevel", "error", "-y", "-i"])
            .arg(&bit)
            .args(["-f", "rawvideo", "-pix_fmt", pix_fmt])
            .arg(&yuv)
            .status()
            .unwrap();
        assert!(st.success(), "{pf:?}: black-box decode failed");
        let raw = std::fs::read(&yuv).unwrap();
        let ours: Vec<u8> = frames[0]
            .image_planes()
            .iter()
            .flat_map(|p| p.data.iter().copied())
            .collect();
        assert_eq!(
            raw, ours,
            "{pf:?}: black-box output differs from the registry decoder"
        );
    }
}
