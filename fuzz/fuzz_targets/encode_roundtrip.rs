//! Encode → decode round-trip fuzz entry: the stills encoder codes a
//! fuzz-shaped picture under fuzz-chosen tools, the registry decoder
//! decodes the Annex B stream and (through a `vvcC` record) the
//! length-prefixed samples, and both outputs must equal the encoder's
//! own reconstruction sample for sample — the invariant every
//! conforming stream of this encoder satisfies. A configuration the
//! encoder refuses (`Err`) is fine; a panic, a decode error or a
//! mismatch is a finding.
//!
//! Input layout: byte 0 → width 1..=32, byte 1 → height 1..=24,
//! byte 2 → bits 0-1 `chroma_format_idc`, bit 2 10-bit, bit 3
//! lossless, bit 4 MTT pickers, bit 5 dependent quantisation, bit 6
//! sign data hiding, bit 7 CU-size RDO; byte 3 → QP (& 63); the rest
//! fills the planes (cycled).
//!
//! r490 — with bit 6 of byte 3 set the same configuration then runs
//! through the video encoder: 2..=4 pictures (bit 7 of byte 3 and bit 5
//! of byte 0), every later picture the plane fill restarted
//! `1 + (byte 1 >> 5)` bytes further into the input (content that
//! moves), IDR distance 3, effort `(byte 0 >> 6) + bit 7 of byte 2`
//! (0..=4). A configuration the video encoder refuses is fine; the
//! registry decoder must output every picture equal to the reference
//! the encoder predicted the next picture from.

#![no_main]

use libfuzzer_sys::fuzz_target;
use oxideav_core::{CodecId, CodecParameters, Decoder, Error, Frame, Packet, TimeBase};
use oxideav_h266::decoder::H266Decoder;
use oxideav_h266::encoder::EncoderConfig;
use oxideav_h266::encoder_pipeline::encode_idr_with_residuals_cfg;
use oxideav_h266::reconstruct::{PictureBuffer, PicturePlane};
use oxideav_h266::video_encoder::{VideoEncoder, VideoEncoderConfig};
use oxideav_h266::vvcc::{annex_b_to_length_prefixed, VvcDecoderConfigurationRecord};

fn plane_bytes(p: &PicturePlane, wide: bool) -> Vec<u8> {
    let mut v = Vec::with_capacity(p.width * p.height * 2);
    for y in 0..p.height {
        for &s in &p.samples[y * p.stride..y * p.stride + p.width] {
            if wide {
                v.extend_from_slice(&s.to_le_bytes());
            } else {
                v.push(s as u8);
            }
        }
    }
    v
}

/// The top-left `w x h` samples of a decoded plane, packed like
/// [`plane_bytes`]. The decoder presents the §7.4.3.4 conformance
/// window — an odd source dimension along a subsampled axis comes
/// back one sample wider / taller (the encoder's `output_size`), so
/// the comparison runs over the source rectangle.
fn decoded_rect(p: &oxideav_core::VideoPlane, w: usize, h: usize, wide: bool) -> Vec<u8> {
    let bps = if wide { 2 } else { 1 };
    let mut v = Vec::with_capacity(w * h * bps);
    for y in 0..h {
        v.extend_from_slice(&p.data[y * p.stride..y * p.stride + w * bps]);
    }
    v
}

/// Compare a decoded frame against the encoder's reconstruction over
/// the source rectangle of every plane (an empty reconstruction plane
/// — a 1-wide 4:2:2 picture has no chroma column — compares as
/// nothing).
fn assert_matches(
    frame: &oxideav_core::VideoFrame,
    rec: &PictureBuffer,
    cf: u32,
    wide: bool,
    what: &str,
) {
    let planes = frame.image_planes();
    let want: Vec<&PicturePlane> = if cf == 0 {
        vec![&rec.luma]
    } else {
        vec![&rec.luma, &rec.cb, &rec.cr]
    };
    assert_eq!(planes.len(), want.len(), "{what}: plane count");
    for (p, r) in planes.iter().zip(want) {
        if r.width == 0 || r.height == 0 {
            continue;
        }
        assert_eq!(
            decoded_rect(p, r.width, r.height, wide),
            plane_bytes(r, wide),
            "{what}: samples differ"
        );
    }
}

fn decode_all(dec: &mut H266Decoder, data: Vec<u8>) -> Vec<oxideav_core::VideoFrame> {
    dec.send_packet(&Packet::new(0, TimeBase::new(1, 1), data))
        .expect("the registry decoder rejects a stream this encoder produced");
    dec.flush().expect("flush");
    let mut out = Vec::new();
    loop {
        match dec.receive_frame() {
            Ok(Frame::Video(v)) => out.push(v),
            Ok(_) => panic!("non-video frame"),
            Err(Error::NeedMore) | Err(Error::Eof) => break,
            Err(e) => panic!("receive_frame: {e}"),
        }
    }
    out
}

fuzz_target!(|data: &[u8]| {
    if data.len() < 4 {
        return;
    }
    let w = usize::from(data[0] & 31) + 1;
    let h = usize::from(data[1] % 24) + 1;
    let flags = data[2];
    let cf = u32::from(flags & 3);
    let bd = if flags & 4 != 0 { 10 } else { 8 };
    let qp = i32::from(data[3] & 63);
    let mut cfg = EncoderConfig::new(w as u32, h as u32);
    cfg.chroma_format_idc = cf as u8;
    cfg.lossless = flags & 8 != 0;
    cfg.enable_mtt_bt_picker = flags & 16 != 0;
    cfg.enable_mtt_tt_picker = flags & 16 != 0;
    cfg.dep_quant = flags & 32 != 0;
    cfg.sign_data_hiding = flags & 64 != 0;
    cfg.rdo_cu_split = flags & 128 != 0;

    let samples = &data[4..];
    let mid = 1u16 << (bd - 1);
    let max = (1u16 << bd) - 1;
    // The picture whose plane fill starts `start` bytes into the input.
    let picture = |start: usize| -> PictureBuffer {
        let mut src = PictureBuffer::with_chroma_format_bd(w, h, cf, mid, bd);
        let mut k = start;
        let mut next = |plane: &mut PicturePlane| {
            for y in 0..plane.height {
                for x in 0..plane.width {
                    let b = if samples.is_empty() {
                        0
                    } else {
                        samples[k % samples.len()]
                    };
                    k += 1;
                    let v = if bd > 8 {
                        (u16::from(b) << 2) | (u16::from(b) >> 6)
                    } else {
                        u16::from(b)
                    };
                    plane.samples[y * plane.stride + x] = v.min(max);
                }
            }
        };
        next(&mut src.luma);
        if cf != 0 {
            next(&mut src.cb);
            next(&mut src.cr);
        }
        src
    };
    let src = picture(0);

    let Ok((annex_b, rec)) = encode_idr_with_residuals_cfg(&src, qp, cfg) else {
        return; // an excluded tool combination
    };
    let wide = bd > 8;

    // Annex B through the registry decoder.
    let params = CodecParameters::video(CodecId::new("h266"));
    let mut dec = H266Decoder::new(&params).expect("decoder");
    let frames = decode_all(&mut dec, annex_b.clone());
    assert_eq!(frames.len(), 1, "one picture in, one picture out");
    assert_matches(
        &frames[0],
        &rec,
        cf,
        wide,
        "Annex B decode != reconstruction",
    );

    // vvcC + length-prefixed sample through the registry decoder.
    let record = VvcDecoderConfigurationRecord::from_annex_b(&annex_b).expect("record");
    let mut params = CodecParameters::video(CodecId::new("h266"));
    params.extradata = record.to_bytes().expect("record bytes");
    let mut dec = H266Decoder::new(&params).expect("decoder with record");
    let sample = annex_b_to_length_prefixed(&annex_b, 4, true).expect("sample");
    let frames = decode_all(&mut dec, sample);
    assert_eq!(frames.len(), 1);
    assert_matches(&frames[0], &rec, cf, wide, "vvcC decode != reconstruction");

    // r490 — IDR + P pictures through the video encoder.
    if data[3] & 64 == 0 {
        return;
    }
    let count = 2 + usize::from(data[3] >> 7) + usize::from((data[0] >> 5) & 1);
    let step = 1 + usize::from(data[1] >> 5);
    let effort = (data[0] >> 6) + (flags >> 7);
    let vcfg = VideoEncoderConfig::new(cfg, qp)
        .with_keyint(3)
        .with_effort(effort);
    let Ok(mut venc) = VideoEncoder::new(vcfg) else {
        return; // lossless: no inter coding
    };
    let mut stream = Vec::new();
    let mut refs = Vec::new();
    for t in 0..count {
        let Ok(pic) = venc.encode(&picture(t * step)) else {
            return; // an excluded tool combination
        };
        stream.extend_from_slice(&pic.data);
        refs.push(venc.reconstruction().expect("a coded picture has a reconstruction"));
    }
    let params = CodecParameters::video(CodecId::new("h266"));
    let mut dec = H266Decoder::new(&params).expect("decoder");
    let frames = decode_all(&mut dec, stream);
    assert_eq!(frames.len(), count, "every coded picture decodes");
    for (frame, rec) in frames.iter().zip(&refs) {
        assert_matches(frame, rec, cf, wide, "inter decode != encoder reference");
    }
});
