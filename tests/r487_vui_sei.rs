//! r487 — VUI (Rec. ITU-T H.274 §7) and SEI (H.274 §8) end to end:
//!
//! * the encoder writes the colour description into the SPS
//!   `vui_payload()`, the decoder parses it and states it in
//!   `output_params` — never on a frame;
//! * a change of shape or colour description between coded video
//!   sequences is announced by one `Frame::StreamParams` event and one
//!   generation bump, immediately before the first frame of the new
//!   shape;
//! * the encoder's decoded picture hash SEI message (MD5 / CRC /
//!   checksum) verifies against this crate's decoder in every chroma
//!   format at 8 and 10 bits, and a wrong hash is reported;
//! * mastering display colour volume, content light level and
//!   alternative transfer characteristics SEI messages reach
//!   `stream_info` and the transfer characteristics of the colour
//!   signal;
//! * a black-box decoder, where installed, decodes the streams with
//!   VUI + hash SEI to the encoder's reconstruction and reports the
//!   signalled colour description.

use oxideav_core::{
    CodecId, CodecOptions, CodecParameters, ColorRange, ColorSignal, Decoder, Error, Frame,
    OutputParamsWatch, Packet, PixelFormat, Rational, TimeBase, VideoFrame, VideoPlane,
};
use oxideav_h266::decoder::H266Decoder;
use oxideav_h266::encoder::EncoderConfig;
use oxideav_h266::encoder_pipeline::encode_idr_with_residuals_cfg;
use oxideav_h266::nal::{NalUnitType, extract_rbsp, iter_annex_b};
use oxideav_h266::picture_hash::{PictureHashMethod, PictureHashStatus};
use oxideav_h266::reconstruct::PictureBuffer;
use oxideav_h266::sei_payload::{
    ContentLightLevelInfo, MasteringDisplayColourVolume, SeiPayload, write_sei_message,
};
use oxideav_h266::sps::parse_sps;
use oxideav_h266::stream::{DecodedPicture, StreamDecoder};
use oxideav_h266::vui::{
    SourceScan, VuiAspectRatio, VuiChromaLoc, VuiColourDescription, VuiParameters,
};

fn source(w: usize, h: usize, cf: u32, bd: u32, seed: u32) -> PictureBuffer {
    let mid = 1u16 << (bd - 1);
    let sh = bd - 8;
    let mut src = PictureBuffer::with_chroma_format_bd(w, h, cf, mid, bd);
    let mut state = seed.wrapping_mul(2654435761).wrapping_add(977);
    let mut rnd = || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        state
    };
    for y in 0..h {
        for x in 0..w {
            let v = 40 + (x * 120) / w + ((y / 5) % 4) * 20 + (rnd() % 24) as usize;
            src.luma.samples[y * src.luma.stride + x] = (v.min(255) << sh) as u16;
        }
    }
    if cf != 0 {
        let cw = src.cb.width;
        for y in 0..src.cb.height {
            for x in 0..cw {
                src.cb.samples[y * src.cb.stride + x] = ((90 + (x * 3 + y) % 70) << sh) as u16;
                src.cr.samples[y * src.cr.stride + x] = ((160 - (x + y * 2) % 60) << sh) as u16;
            }
        }
    }
    src
}

fn decode_all(bs: &[u8], verify: bool) -> Vec<DecodedPicture> {
    let mut dec = StreamDecoder::new();
    dec.set_verify_picture_hash(verify);
    let mut out = Vec::new();
    dec.decode_annex_b(bs, &mut |pic| out.push(pic))
        .expect("own decoder rejects the stream");
    out
}

/// Annex B NAL units of `bs`, start codes included, as owned buffers.
fn nal_units(bs: &[u8]) -> Vec<(NalUnitType, Vec<u8>)> {
    iter_annex_b(bs)
        .map(|n| {
            let mut v = vec![0, 0, 0, 1];
            v.extend_from_slice(n.raw);
            (n.header.nal_unit_type, v)
        })
        .collect()
}

/// A prefix SEI NAL unit (start code included) holding `messages`.
fn prefix_sei_nal(messages: &[(u32, Vec<u8>)]) -> Vec<u8> {
    let mut rbsp = Vec::new();
    for (ty, body) in messages {
        rbsp.extend(write_sei_message(*ty, body));
    }
    rbsp.push(0x80);
    // NAL header: forbidden_zero 0, reserved 0, layer 0 | type 23, tid+1 = 1.
    let mut nal = vec![0, 0, 0, 1, 0x00, (23 << 3) | 1];
    // Emulation prevention.
    let mut zeros = 0;
    for b in rbsp {
        if zeros >= 2 && b <= 3 {
            nal.push(3);
            zeros = 0;
        }
        zeros = if b == 0 { zeros + 1 } else { 0 };
        nal.push(b);
    }
    nal
}

fn full_vui() -> VuiParameters {
    VuiParameters::new()
        .with_progressive_source()
        .with_aspect_ratio(VuiAspectRatio::explicit(16, 11))
        .with_colour_description(VuiColourDescription::new(9, 16, 9, false))
        .with_chroma_loc(VuiChromaLoc::Frame(2))
}

#[test]
fn encoder_vui_reaches_the_sps_and_decodes_identically() {
    for &cf in &[0u32, 1, 2, 3] {
        let (w, h) = (72usize, 40usize);
        let src = source(w, h, cf, 10, cf);
        let mut cfg = EncoderConfig::new(w as u32, h as u32);
        cfg.chroma_format_idc = cf as u8;
        let (plain, rec_plain) = encode_idr_with_residuals_cfg(&src, 30, cfg).unwrap();
        let (with_vui, rec_vui) =
            encode_idr_with_residuals_cfg(&src, 30, cfg.with_vui(full_vui())).unwrap();
        // The VUI changes the SPS only — the picture is the same.
        assert_eq!(rec_plain.luma.samples, rec_vui.luma.samples);
        assert!(with_vui.len() > plain.len());

        let sps_nal = iter_annex_b(&with_vui)
            .find(|n| n.header.nal_unit_type == NalUnitType::SpsNut)
            .unwrap();
        let sps = parse_sps(&extract_rbsp(sps_nal.payload())).unwrap();
        assert!(sps.sps_vui_parameters_present_flag);
        let vui = sps.vui.expect("VUI parsed");
        assert_eq!(vui, full_vui());
        assert_eq!(vui.source_scan(), SourceScan::Progressive);
        assert_eq!(vui.aspect_ratio.unwrap().explicit_ratio(), Some((16, 11)));

        let pics = decode_all(&with_vui, false);
        assert_eq!(pics.len(), 1);
        let info = pics[0].output_info;
        assert_eq!(info.vui, Some(full_vui()));
        assert_eq!(
            info.color_signal(),
            ColorSignal::from_code_points(9, 16, 9, false)
        );
        assert_eq!(
            pics[0].frame.luma.samples,
            decode_all(&plain, false)[0].frame.luma.samples
        );
        // No VUI: nothing is claimed.
        assert_eq!(
            decode_all(&plain, false)[0].output_info.color_signal(),
            ColorSignal::unspecified()
        );
    }
}

#[test]
fn encoder_picture_hash_verifies_in_every_format() {
    for &cf in &[0u32, 1, 2, 3] {
        for &bd in &[8u32, 10] {
            for method in [
                PictureHashMethod::Md5,
                PictureHashMethod::Crc,
                PictureHashMethod::Checksum,
            ] {
                // 70 x 37: padded to 72 x 40 — the hash covers the
                // coded picture, not the cropped one.
                let (w, h) = (70usize, 37usize);
                let src = source(w, h, cf, bd, cf * 7 + bd);
                let mut cfg = EncoderConfig::new(w as u32, h as u32).with_picture_hash(method);
                cfg.chroma_format_idc = cf as u8;
                let label = format!("cf{cf} {bd}-bit {method:?}");
                let (bs, _) = encode_idr_with_residuals_cfg(&src, 28, cfg)
                    .unwrap_or_else(|e| panic!("{label}: {e}"));
                let last = iter_annex_b(&bs).last().unwrap();
                assert_eq!(last.header.nal_unit_type, NalUnitType::SuffixSeiNut);

                let pics = decode_all(&bs, true);
                assert_eq!(pics.len(), 1, "{label}");
                assert_eq!(
                    pics[0].hash_status,
                    Some(PictureHashStatus::Match(method)),
                    "{label}"
                );
                let hashes: Vec<_> = pics[0]
                    .sei
                    .iter()
                    .filter_map(|m| match m {
                        SeiPayload::DecodedPictureHash(h) => Some(h),
                        _ => None,
                    })
                    .collect();
                assert_eq!(hashes.len(), 1, "{label}");
                assert_eq!(hashes[0].components.len(), if cf == 0 { 1 } else { 3 });
                assert_eq!(hashes[0].single_component_flag, cf == 0);
                // Off by default: the message is still handed out.
                let unchecked = decode_all(&bs, false);
                assert_eq!(unchecked[0].hash_status, None);
                assert_eq!(unchecked[0].sei.len(), 1);

                // Flip one bit of the last hash byte: reported, not fatal.
                let mut bad = bs.clone();
                let n = bad.len();
                bad[n - 2] ^= 0x10;
                let pics = decode_all(&bad, true);
                assert!(
                    matches!(
                        pics[0].hash_status,
                        Some(PictureHashStatus::Mismatch { method: m, .. }) if m == method
                    ),
                    "{label}: {:?}",
                    pics[0].hash_status
                );
            }
        }
    }
}

fn frame_of(pic: &PictureBuffer, bd: u32) -> VideoFrame {
    let pack = |p: &oxideav_h266::reconstruct::PicturePlane| {
        let mut data = Vec::new();
        for y in 0..p.height {
            for &v in &p.samples[y * p.stride..y * p.stride + p.width] {
                if bd > 8 {
                    data.extend_from_slice(&v.to_le_bytes());
                } else {
                    data.push(v as u8);
                }
            }
        }
        VideoPlane {
            stride: p.width * if bd > 8 { 2 } else { 1 },
            data,
        }
    };
    VideoFrame {
        pts: None,
        planes: vec![pack(&pic.luma), pack(&pic.cb), pack(&pic.cr)],
    }
}

fn enc_params(
    w: u32,
    h: u32,
    pf: PixelFormat,
    color: ColorSignal,
    opts: &[(&str, &str)],
) -> CodecParameters {
    let mut p = CodecParameters::video(CodecId::new("h266"));
    p.width = Some(w);
    p.height = Some(h);
    p.pixel_format = Some(pf);
    p.frame_rate = Some(Rational::new(25, 1));
    p.color_signal = color;
    let mut o = CodecOptions::new();
    for (k, v) in opts {
        o = o.set(*k, *v);
    }
    p.options = o;
    p
}

fn registry_encode(params: &CodecParameters, frame: &VideoFrame) -> (Vec<u8>, CodecParameters) {
    let mut enc = oxideav_h266::h266_encoder::make_encoder(params).unwrap();
    enc.send_frame(&Frame::Video(frame.clone())).unwrap();
    enc.flush().unwrap();
    let pkt = enc.receive_packet().unwrap();
    (pkt.data, enc.output_params().clone())
}

fn dec_params(opts: &[(&str, &str)]) -> CodecParameters {
    let mut p = CodecParameters::video(CodecId::new("h266"));
    let mut o = CodecOptions::new();
    for (k, v) in opts {
        o = o.set(*k, *v);
    }
    p.options = o;
    p
}

/// The colour signal a registry encoder is configured with comes back
/// from the registry decoder's `output_params` — and from nowhere
/// else: the frames carry planes and a timestamp only.
#[test]
fn colour_signal_round_trips_through_the_registry() {
    let (w, h) = (64usize, 48usize);
    let src = source(w, h, 1, 8, 3);
    let frame = frame_of(&src, 8);
    let bt2020_pq = ColorSignal::from_code_points(9, 16, 9, false);
    let params = enc_params(
        w as u32,
        h as u32,
        PixelFormat::Yuv420P,
        bt2020_pq,
        &[("picture_hash", "md5")],
    );
    let (data, enc_out) = registry_encode(&params, &frame);
    assert_eq!(enc_out.color_signal, bt2020_pq);

    let mut dec = H266Decoder::new(&dec_params(&[("verify_picture_hash", "true")])).unwrap();
    assert!(dec.output_params().is_none());
    dec.send_packet(&Packet::new(0, TimeBase::new(1, 25), data.clone()))
        .unwrap();
    dec.flush().unwrap();
    let Frame::Video(vf) = dec.receive_frame().unwrap() else {
        panic!("first frame is a picture, not an event")
    };
    assert_eq!(vf.planes.len(), 3);
    let out = dec.output_params().unwrap();
    assert_eq!((out.width, out.height), (Some(64), Some(48)));
    assert_eq!(out.pixel_format, Some(PixelFormat::Yuv420P));
    assert_eq!(out.color_signal, bt2020_pq);
    assert_eq!(out.color_signal.range, ColorRange::Limited);
    assert_eq!(dec.output_params_generation(), 0);
    let stats = dec.picture_hash_stats();
    assert_eq!(
        (stats.matched, stats.mismatched, stats.unchecked),
        (1, 0, 0)
    );
    assert!(matches!(dec.receive_frame(), Err(Error::Eof)));

    // A stream without a colour description claims nothing.
    let plain = enc_params(
        w as u32,
        h as u32,
        PixelFormat::Yuv420P,
        ColorSignal::unspecified(),
        &[],
    );
    let (plain_data, plain_out) = registry_encode(&plain, &frame);
    assert_eq!(plain_out.color_signal, ColorSignal::unspecified());
    let mut dec = H266Decoder::new(&dec_params(&[])).unwrap();
    dec.send_packet(&Packet::new(0, TimeBase::new(1, 25), plain_data))
        .unwrap();
    dec.flush().unwrap();
    assert!(matches!(dec.receive_frame(), Ok(Frame::Video(_))));
    assert_eq!(
        dec.output_params().unwrap().color_signal,
        ColorSignal::unspecified()
    );

    // An identity matrix is barred outside 4:4:4: written (and
    // reported) as unspecified, the rest of the description kept.
    let srgb_420 = enc_params(
        w as u32,
        h as u32,
        PixelFormat::Yuv420P,
        ColorSignal::srgb(),
        &[],
    );
    let (_, out) = registry_encode(&srgb_420, &frame);
    assert_eq!(
        out.color_signal,
        ColorSignal::from_code_points(1, 13, 2, true)
    );

    // A wrong hash under `verify_picture_hash` fails the packet; the
    // picture is still delivered.
    let mut bad = data;
    let n = bad.len();
    bad[n - 2] ^= 1;
    let mut dec = H266Decoder::new(&dec_params(&[("verify_picture_hash", "true")])).unwrap();
    let err = dec
        .send_packet(&Packet::new(0, TimeBase::new(1, 25), bad))
        .unwrap_err();
    assert!(matches!(err, Error::InvalidData(_)), "{err:?}");
    dec.flush().unwrap();
    assert!(matches!(dec.receive_frame(), Ok(Frame::Video(_))));
    assert_eq!(dec.picture_hash_stats().mismatched, 1);
}

/// Three coded video sequences in one stream — 64x48 BT.709, then
/// 64x48 BT.2020 PQ, then 96x64 BT.2020 PQ: each change is one
/// `StreamParams` event and one generation bump, immediately before
/// the first frame of the new shape; the unchanged repeat is silent.
#[test]
fn mid_stream_changes_are_announced_once_in_band() {
    let bt709 = ColorSignal::bt709_limited();
    let pq = ColorSignal::from_code_points(9, 16, 9, false);
    let small = frame_of(&source(64, 48, 1, 8, 1), 8);
    let large = frame_of(&source(96, 64, 1, 8, 2), 8);
    let seq = [
        (64u32, 48u32, bt709, &small),
        (64, 48, bt709, &small),
        (64, 48, pq, &small),
        (96, 64, pq, &large),
    ];
    let mut dec = H266Decoder::new(&dec_params(&[])).unwrap();
    let mut stream_params = CodecParameters::video(CodecId::new("h266"));
    let mut watch = OutputParamsWatch::new();
    let mut log: Vec<String> = Vec::new();
    let mut changes = 0;
    for (i, (w, h, color, frame)) in seq.iter().enumerate() {
        let p = enc_params(*w, *h, PixelFormat::Yuv420P, *color, &[]);
        let (data, _) = registry_encode(&p, frame);
        dec.send_packet(&Packet::new(0, TimeBase::new(1, 25), data).with_pts(i as i64))
            .unwrap();
        // Each packet is an IDR access unit: the previous sequence's
        // picture leaves when the next one arrives.
        loop {
            match dec.receive_frame() {
                Ok(Frame::Video(vf)) => {
                    let out = dec.output_params().unwrap();
                    // The description is that of the frame just returned.
                    assert_eq!(
                        vf.planes[0].stride as u32,
                        out.width.unwrap(),
                        "frame {i}: plane width vs output_params"
                    );
                    log.push(format!(
                        "frame {}x{} p{}",
                        out.width.unwrap(),
                        out.height.unwrap(),
                        out.color_signal.primaries.0
                    ));
                    if watch.observe(&dec, &mut stream_params) {
                        changes += 1;
                    }
                }
                Ok(Frame::StreamParams(p)) => {
                    let now = dec.output_params().unwrap();
                    assert_eq!(
                        (p.width, p.height, p.pixel_format, p.color_signal),
                        (now.width, now.height, now.pixel_format, now.color_signal)
                    );
                    log.push(format!(
                        "event {}x{} p{}",
                        p.width.unwrap(),
                        p.height.unwrap(),
                        p.color_signal.primaries.0
                    ));
                }
                Ok(_) => panic!("unexpected frame kind"),
                Err(Error::NeedMore) => break,
                Err(e) => panic!("{e}"),
            }
        }
    }
    dec.flush().unwrap();
    loop {
        match dec.receive_frame() {
            Ok(Frame::Video(_)) => {
                let out = dec.output_params().unwrap();
                log.push(format!(
                    "frame {}x{} p{}",
                    out.width.unwrap(),
                    out.height.unwrap(),
                    out.color_signal.primaries.0
                ));
                if watch.observe(&dec, &mut stream_params) {
                    changes += 1;
                }
            }
            Ok(Frame::StreamParams(p)) => log.push(format!(
                "event {}x{} p{}",
                p.width.unwrap(),
                p.height.unwrap(),
                p.color_signal.primaries.0
            )),
            Ok(_) => panic!("unexpected frame kind"),
            Err(Error::Eof) => break,
            Err(e) => panic!("{e}"),
        }
    }
    assert_eq!(
        log,
        [
            "frame 64x48 p1",
            "frame 64x48 p1",
            "event 64x48 p9",
            "frame 64x48 p9",
            "event 96x64 p9",
            "frame 96x64 p9",
        ]
    );
    assert_eq!(dec.output_params_generation(), 2);
    // First resolution + two changes, as the watch sees them.
    assert_eq!(changes, 3);
    assert_eq!(stream_params.width, Some(96));
    assert_eq!(stream_params.color_signal, pq);
}

/// HDR SEI messages placed before the picture: typed on the picture,
/// persistent in the stream information, and the preferred transfer
/// characteristics replace the VUI's in the colour signal.
#[test]
fn hdr_sei_messages_reach_the_stream_info() {
    let (w, h) = (64usize, 48usize);
    let src = source(w, h, 1, 10, 5);
    let vui =
        VuiParameters::new().with_colour_description(VuiColourDescription::new(9, 14, 9, false));
    let cfg = EncoderConfig::new(w as u32, h as u32).with_vui(vui);
    let (bs, _) = encode_idr_with_residuals_cfg(&src, 30, cfg).unwrap();

    let mdcv = MasteringDisplayColourVolume::new(
        [(8500, 39850), (6550, 2300), (35400, 14600)],
        (15635, 16450),
        10_000_000,
        1,
    );
    let cll = ContentLightLevelInfo::new(1000, 180);
    let sei = prefix_sei_nal(&[
        (137, mdcv.write_payload()),
        (144, cll.write_payload()),
        (147, vec![18]),
        // An unknown type and a truncated known one are skipped.
        (999, vec![1, 2, 3]),
        (137, vec![0; 5]),
    ]);
    // The SEI NAL unit goes before the first VCL NAL unit. A second
    // IDR picture follows without one: it starts a new CLVS, so the
    // persistent state must be gone again.
    let mut stream = Vec::new();
    let mut inserted = false;
    for (ty, nal) in nal_units(&bs) {
        if ty.is_vcl() && !inserted {
            stream.extend_from_slice(&sei);
            inserted = true;
        }
        stream.extend_from_slice(&nal);
    }
    assert!(inserted);
    let first_len = stream.len();
    stream.extend_from_slice(&bs); // second IDR: a new CLVS, no SEI

    let pics = decode_all(&stream, true);
    assert_eq!(pics.len(), 2);
    let p0 = &pics[0];
    assert_eq!(
        p0.sei,
        vec![
            SeiPayload::MasteringDisplayColourVolume(mdcv),
            SeiPayload::ContentLightLevelInfo(cll),
            SeiPayload::AlternativeTransferCharacteristics {
                preferred_transfer_characteristics: 18
            },
            SeiPayload::Other { payload_type: 999 },
        ]
    );
    assert_eq!(p0.output_info.mastering_display, Some(mdcv));
    assert_eq!(p0.output_info.content_light_level, Some(cll));
    assert_eq!(p0.output_info.preferred_transfer_characteristics, Some(18));
    assert_eq!(
        p0.output_info.color_signal(),
        ColorSignal::from_code_points(9, 18, 9, false)
    );
    assert_eq!(p0.hash_status, None);
    // The next CLVS starts clean.
    let p1 = &pics[1];
    assert!(p1.sei.is_empty());
    assert_eq!(p1.output_info.mastering_display, None);
    assert_eq!(
        p1.output_info.color_signal(),
        ColorSignal::from_code_points(9, 14, 9, false)
    );

    // Through the registry: stream_info follows the frame returned,
    // and the transfer change is a stream-parameter change.
    let mut dec = H266Decoder::new(&dec_params(&[])).unwrap();
    assert!(dec.stream_info().is_none());
    dec.send_packet(&Packet::new(
        0,
        TimeBase::new(1, 25),
        stream[..first_len].to_vec(),
    ))
    .unwrap();
    dec.send_packet(&Packet::new(
        0,
        TimeBase::new(1, 25),
        stream[first_len..].to_vec(),
    ))
    .unwrap();
    dec.flush().unwrap();
    assert!(matches!(dec.receive_frame(), Ok(Frame::Video(_))));
    let info = dec.stream_info().unwrap();
    assert_eq!(info.mastering_display, Some(mdcv));
    assert_eq!(
        info.content_light_level.unwrap().max_content_light_level,
        1000
    );
    assert_eq!(dec.output_params().unwrap().color_signal.transfer.0, 18);
    let Ok(Frame::StreamParams(p)) = dec.receive_frame() else {
        panic!("the transfer characteristics changed: an event is due")
    };
    assert_eq!(p.color_signal.transfer.0, 14);
    assert!(matches!(dec.receive_frame(), Ok(Frame::Video(_))));
    assert_eq!(dec.stream_info().unwrap().mastering_display, None);
}

/// A broken VUI costs the colour description, never the picture; SEI
/// NAL units that do not parse are ignored.
#[test]
fn malformed_vui_and_sei_do_not_stop_the_decode() {
    let (w, h) = (64usize, 48usize);
    let src = source(w, h, 1, 8, 9);
    // Rewriting the head of the written payload to claim an explicit
    // sample aspect ratio (vui_aspect_ratio_idc = 255) makes it end
    // inside its own syntax.
    let vui =
        VuiParameters::new().with_colour_description(VuiColourDescription::new(1, 1, 1, false));
    let cfg = EncoderConfig::new(w as u32, h as u32).with_vui(vui);
    let (bs, _) = encode_idr_with_residuals_cfg(&src, 30, cfg).unwrap();
    let good = decode_all(&bs, false);
    let payload = vui.write_payload();

    let mut stream = Vec::new();
    for (ty, mut nal) in nal_units(&bs) {
        if ty == NalUnitType::SpsNut {
            // Find the VUI payload bytes inside the SPS NAL unit:
            // flags 0000, aspect ratio present, constant, idc 0xFF.
            let pos = nal
                .windows(payload.len())
                .rposition(|w| w == payload.as_slice())
                .expect("VUI payload bytes in the SPS");
            nal[pos] = 0x0f;
            nal[pos + 1] = 0xff;
        }
        stream.extend_from_slice(&nal);
        if ty == NalUnitType::PpsNut {
            // Garbage SEI NAL units: empty, cut-off size, no trailing bits.
            stream.extend_from_slice(&[0, 0, 0, 1, 0x00, (23 << 3) | 1]);
            stream.extend_from_slice(&[0, 0, 0, 1, 0x00, (23 << 3) | 1, 137, 200, 1, 2]);
            stream.extend_from_slice(&[0, 0, 0, 1, 0x00, (23 << 3) | 1, 0xff, 0xff, 0xff]);
        }
    }
    let pics = decode_all(&stream, true);
    assert_eq!(pics.len(), 1);
    assert_eq!(pics[0].output_info.vui, None);
    assert_eq!(
        pics[0].output_info.color_signal(),
        ColorSignal::unspecified()
    );
    assert!(pics[0].sei.is_empty());
    assert_eq!(pics[0].frame.luma.samples, good[0].frame.luma.samples);
}

/// Black-box validation (opaque process; a no-op where the binaries
/// are absent): a stream with VUI + hash SEI decodes to the encoder's
/// reconstruction, and the prober reports the colour description the
/// VUI carries.
#[test]
fn black_box_decoder_reads_the_vui_and_accepts_the_hash_sei() {
    let have = |bin: &str| {
        matches!(
            std::process::Command::new(bin)
                .arg("-version")
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .status(),
            Ok(s) if s.success()
        )
    };
    if !have("ffmpeg") || !have("ffprobe") {
        eprintln!("black-box decoder not installed — leg skipped");
        return;
    }
    let dir = std::env::temp_dir().join(format!("oxideav-h266-r487-vui-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let cases = [
        (
            PictureHashMethod::Md5,
            (1u8, 1u8, 1u8, false),
            "bt709",
            "bt709",
            "bt709",
            "tv",
        ),
        (
            PictureHashMethod::Crc,
            (9, 16, 9, false),
            "bt2020",
            "smpte2084",
            "bt2020nc",
            "tv",
        ),
        (
            PictureHashMethod::Checksum,
            (9, 18, 9, true),
            "bt2020",
            "arib-std-b67",
            "bt2020nc",
            "pc",
        ),
    ];
    for (i, (method, (cp, tc, mc, full), want_cp, want_tc, want_mc, want_range)) in
        cases.into_iter().enumerate()
    {
        let (w, h) = (96usize, 64usize);
        let src = source(w, h, 1, 10, 20 + i as u32);
        let vui = VuiParameters::new()
            .with_progressive_source()
            .with_colour_description(VuiColourDescription::new(cp, tc, mc, full));
        let cfg = EncoderConfig::new(w as u32, h as u32)
            .with_vui(vui)
            .with_picture_hash(method);
        let (bs, rec) = encode_idr_with_residuals_cfg(&src, 27, cfg).unwrap();
        let bit = dir.join(format!("{i}.266"));
        let yuv = dir.join(format!("{i}.yuv"));
        std::fs::write(&bit, &bs).unwrap();

        let probe = std::process::Command::new("ffprobe")
            .args(["-v", "error", "-select_streams", "v:0", "-show_entries"])
            .arg("stream=color_range,color_space,color_transfer,color_primaries")
            .args(["-of", "default=noprint_wrappers=1"])
            .arg(&bit)
            .output()
            .unwrap();
        let text = String::from_utf8_lossy(&probe.stdout).to_string();
        for want in [
            format!("color_range={want_range}"),
            format!("color_space={want_mc}"),
            format!("color_transfer={want_tc}"),
            format!("color_primaries={want_cp}"),
        ] {
            assert!(text.contains(&want), "case {i}: want {want} in:\n{text}");
        }

        let out = std::process::Command::new("ffmpeg")
            .args(["-hide_banner", "-loglevel", "warning", "-y", "-i"])
            .arg(&bit)
            .args(["-f", "rawvideo", "-pix_fmt", "yuv420p10le"])
            .arg(&yuv)
            .output()
            .unwrap();
        assert!(out.status.success(), "case {i}: black-box decode failed");
        let log = String::from_utf8_lossy(&out.stderr).to_lowercase();
        assert!(
            !log.contains("mismatch") && !log.contains("incorrect"),
            "case {i}: black-box decoder complained:\n{log}"
        );
        let raw = std::fs::read(&yuv).unwrap();
        let data: Vec<u16> = raw
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| u16::from_le_bytes(*c))
            .collect();
        assert_eq!(data.len(), w * h * 3 / 2);
        let mut expect = Vec::with_capacity(data.len());
        for p in [&rec.luma, &rec.cb, &rec.cr] {
            for y in 0..p.height {
                expect.extend_from_slice(&p.samples[y * p.stride..y * p.stride + p.width]);
            }
        }
        assert!(data == expect, "case {i}: black-box output differs");
    }
    let _ = std::fs::remove_dir_all(&dir);
}
