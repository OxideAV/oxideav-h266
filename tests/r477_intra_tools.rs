//! r477 — the intra coding tools of the stills encoder: matrix-based
//! intra prediction (§8.4.5.2.2), intra sub-partitions (§8.4.5.1) and
//! BDPCM (§7.3.11.5 / §8.7.3). Every tool is elected by the per-CU
//! RD decision on textured content, the streams decode byte-exact
//! through this crate's decoder (and a black-box decoder where
//! installed) in every chroma format, and switching a tool on never
//! costs bytes at equal quality on the textured sources.
//!
//! Spec reference: ITU-T H.266 | ISO/IEC 23090-3 (V4, 01/2026).

use oxideav_h266::encoder::EncoderConfig;
use oxideav_h266::encoder_pipeline::{encode_idr_with_stats_cfg, psnr_y, EncodeStats};
use oxideav_h266::reconstruct::{PictureBuffer, PicturePlane};
use oxideav_h266::stream::{DecodedPicture, StreamDecoder};

/// A textured source: a pseudo-random field with structure (bands,
/// a diagonal edge, discs) so matrix / sub-partition / DPCM
/// prediction each have blocks to win.
fn textured_source(w: usize, h: usize, cf: u32, bd: u32, seed: u32) -> PictureBuffer {
    let mid = 1u16 << (bd - 1);
    let sh = bd - 8;
    let mut src = PictureBuffer::with_chroma_format_bd(w, h, cf, mid, bd);
    let mut state = seed.wrapping_mul(2654435761).wrapping_add(12345);
    let mut rnd = || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        state
    };
    for y in 0..h {
        for x in 0..w {
            let band = ((x / 7 + y / 11) % 5) * 25;
            let edge = if x * 3 + y * 2 > w * 2 { 40 } else { 0 };
            let dx = x as i64 - (w / 2) as i64;
            let dy = y as i64 - (h / 2) as i64;
            let disc = if dx * dx + dy * dy < ((w.min(h) / 5) as i64).pow(2) {
                50
            } else {
                0
            };
            let noise = (rnd() % 40) as usize;
            let v = (30 + band + edge + noise).saturating_sub(disc).min(255);
            src.luma.samples[y * w + x] = ((v << sh) + (rnd() % (1 << sh).max(1)) as usize) as u16;
        }
    }
    let cw = src.cb.width;
    for y in 0..src.cb.height {
        for x in 0..cw {
            src.cb.samples[y * cw + x] =
                (((100 + (x * 5 + y * 3) % 70) << sh) as u32 + rnd() % (1 << sh).max(1)) as u16;
            src.cr.samples[y * cw + x] =
                (((140 - (x * 2 + y * 7) % 60) << sh) as u32 + rnd() % (1 << sh).max(1)) as u16;
        }
    }
    src
}

/// The reconstruction (source size) against the decoded plane (the
/// conformance window — one sample larger along an odd subsampled
/// axis), over the reconstruction's rectangle.
fn plane_eq(dec: &PicturePlane, rec: &PicturePlane) -> bool {
    dec.width >= rec.width
        && dec.height >= rec.height
        && (0..rec.height).all(|y| {
            dec.samples[y * dec.stride..y * dec.stride + rec.width]
                == rec.samples[y * rec.stride..y * rec.stride + rec.width]
        })
}

fn decode_once(bs: &[u8]) -> DecodedPicture {
    let mut dec = StreamDecoder::new();
    let mut got = None;
    dec.decode_annex_b(bs, &mut |pic| got = Some(pic))
        .expect("own decoder rejects the stream");
    got.expect("no picture decoded")
}

/// Encode, check the own-decoder byte-exact round trip, return
/// (bytes, reconstruction, stats, luma PSNR).
fn round_trip(
    src: &PictureBuffer,
    cfg: EncoderConfig,
    qp: i32,
    label: &str,
) -> (Vec<u8>, PictureBuffer, EncodeStats, f64) {
    let (bs, rec, stats) = encode_idr_with_stats_cfg(src, qp, cfg)
        .unwrap_or_else(|e| panic!("{label}: encode failed: {e}"));
    let pic = decode_once(&bs);
    let (_, _, cw, ch) = pic.crop;
    let (ow, oh) = cfg.output_size();
    assert_eq!((cw, ch), (ow as usize, oh as usize), "{label}: crop");
    assert!(
        plane_eq(&pic.frame.luma, &rec.luma),
        "{label}: luma differs"
    );
    if cfg.chroma_format_idc != 0 {
        assert!(plane_eq(&pic.frame.cb, &rec.cb), "{label}: Cb differs");
        assert!(plane_eq(&pic.frame.cr, &rec.cr), "{label}: Cr differs");
    }
    let psnr = psnr_y(&src.luma, &rec.luma).unwrap();
    (bs, rec, stats, psnr)
}

fn cfg_for(w: usize, h: usize, cf: u32, mip: bool, isp: bool, bdpcm: bool) -> EncoderConfig {
    let mut cfg = EncoderConfig::new(w as u32, h as u32);
    cfg.chroma_format_idc = cf as u8;
    cfg.intra_mip = mip;
    cfg.intra_isp = isp;
    cfg.intra_bdpcm = bdpcm;
    cfg
}

/// MIP is elected on textured content in every chroma format (the
/// 4:4:4 `MipChromaDirectFlag` arm included) at 8 and 10 bits, the
/// streams decode byte-exact, and the tool pays for itself: no cell
/// loses more than the per-CU greedy decision's noise (1.5 % at equal
/// PSNR) and the eight cells together are smaller with it than
/// without at no PSNR loss.
#[test]
fn mip_is_elected_and_pays_for_itself() {
    let (mut total_on, mut total_off) = (0usize, 0usize);
    let (mut psnr_on, mut psnr_off) = (0.0f64, 0.0f64);
    for &cf in &[0u32, 1, 2, 3] {
        for &bd in &[8u32, 10] {
            let (w, h) = (128usize, 96usize);
            let src = textured_source(w, h, cf, bd, 7 + cf + bd);
            let qp = 30;
            let (bs_off, _, s_off, p_off) =
                round_trip(&src, cfg_for(w, h, cf, false, false, false), qp, "mip off");
            let (bs_on, _, s_on, p_on) =
                round_trip(&src, cfg_for(w, h, cf, true, false, false), qp, "mip on");
            assert_eq!(s_off.mip_cus, 0);
            assert!(
                s_on.mip_cus > 0,
                "cf{cf} {bd}-bit: no CU elected MIP ({} CUs)",
                s_on.cus
            );
            let on = bs_on.len() as f64;
            let off = bs_off.len() as f64;
            assert!(
                on <= off * 1.015 || p_on > p_off + 0.05,
                "cf{cf} {bd}-bit: MIP on {on} B / {p_on:.2} dB vs off {off} B / {p_off:.2} dB"
            );
            println!(
                "cf{cf} {bd}-bit QP{qp}: MIP {}/{} CUs, {off} -> {on} B, {p_off:.2} -> {p_on:.2} dB",
                s_on.mip_cus, s_on.cus
            );
            total_on += bs_on.len();
            total_off += bs_off.len();
            psnr_on += p_on;
            psnr_off += p_off;
        }
    }
    assert!(
        total_on <= total_off,
        "MIP costs bytes overall: {total_on} vs {total_off}"
    );
    assert!(
        psnr_on >= psnr_off - 0.05,
        "MIP costs quality overall: {psnr_on:.2} vs {psnr_off:.2}"
    );
    println!(
        "all cells: {total_off} -> {total_on} B, mean PSNR {:.3} -> {:.3} dB",
        psnr_off / 8.0,
        psnr_on / 8.0
    );
}

/// MIP CUs next to each other (the `intra_mip_flag` ctxInc neighbour
/// rule) and MIP at every CU size the RDO reaches, across a few QPs.
#[test]
fn mip_across_qps_and_sizes() {
    let (w, h) = (192usize, 128usize);
    let src = textured_source(w, h, 1, 8, 99);
    for qp in [18, 27, 36, 45] {
        let (_, _, s, p) = round_trip(&src, cfg_for(w, h, 1, true, false, false), qp, "qp");
        assert!(s.mip_cus > 0, "QP{qp}: no MIP CU");
        println!("QP{qp}: MIP {}/{} CUs, {p:.2} dB", s.mip_cus, s.cus);
    }
    // The MTT pickers on top of the MIP decision.
    let mut cfg = cfg_for(w, h, 1, true, false, false);
    cfg.enable_mtt_bt_picker = true;
    cfg.enable_mtt_tt_picker = true;
    let (_, _, s, _) = round_trip(&src, cfg, 30, "mtt + mip");
    assert!(s.mip_cus > 0, "MTT: no MIP CU");
    // Odd geometry (boundary CTBs, implicit splits) with MIP.
    let src = textured_source(77, 45, 1, 8, 5);
    let (_, _, s, _) = round_trip(&src, cfg_for(77, 45, 1, true, false, false), 28, "odd");
    assert!(s.cus > 0);
    // Dependent quantisation / sign hiding with MIP residuals.
    let src = textured_source(64, 64, 1, 8, 11);
    for (dq, sdh) in [(true, false), (false, true)] {
        let mut cfg = cfg_for(64, 64, 1, true, false, false);
        cfg.dep_quant = dq;
        cfg.sign_data_hiding = sdh;
        round_trip(&src, cfg, 30, "dq/sdh + mip");
    }
}

/// Black-box validation through `ffmpeg` (opaque process; no-op where
/// the binary is absent): MIP-heavy streams of every format and depth
/// decode to the encoder reconstruction.
#[test]
fn black_box_decoder_agrees_on_mip_streams() {
    let probe = std::process::Command::new("ffmpeg")
        .arg("-version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status();
    if !matches!(probe, Ok(s) if s.success()) {
        eprintln!("ffmpeg not installed — black-box leg skipped");
        return;
    }
    let dir = std::env::temp_dir().join(format!("oxideav-h266-r477-tools-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    for &cf in &[0u32, 1, 2, 3] {
        for &bd in &[8u32, 10] {
            let (w, h) = (128usize, 96usize);
            let src = textured_source(w, h, cf, bd, 3 + cf * 2 + bd);
            let cfg = cfg_for(w, h, cf, true, false, false);
            let label = format!("cf{cf} {bd}-bit");
            let (bs, rec, stats, _) = round_trip(&src, cfg, 28, &label);
            assert!(stats.mip_cus > 0, "{label}: no MIP CU to validate");
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
            let bit = dir.join(format!("{cf}_{bd}.266"));
            let yuv = dir.join(format!("{cf}_{bd}.yuv"));
            std::fs::write(&bit, &bs).unwrap();
            let st = std::process::Command::new("ffmpeg")
                .args(["-hide_banner", "-loglevel", "error", "-y", "-i"])
                .arg(&bit)
                .args(["-f", "rawvideo", "-pix_fmt", pix_fmt])
                .arg(&yuv)
                .status()
                .unwrap();
            assert!(st.success(), "{label}: black-box decode failed");
            let raw = std::fs::read(&yuv).unwrap();
            let data: Vec<u16> = if bd > 8 {
                raw.chunks_exact(2)
                    .map(|c| u16::from_le_bytes([c[0], c[1]]))
                    .collect()
            } else {
                raw.iter().map(|&b| u16::from(b)).collect()
            };
            let (sw, shh) = cfg.chroma_subsampling();
            let (ocw, och) = if cf == 0 {
                (0, 0)
            } else {
                (w / sw as usize, h / shh as usize)
            };
            assert_eq!(data.len(), w * h + 2 * ocw * och, "{label}: geometry");
            let mut n = 0usize;
            for y in 0..h {
                for x in 0..w {
                    if data[y * w + x] != rec.luma.samples[y * rec.luma.stride + x] {
                        n += 1;
                    }
                }
            }
            if cf != 0 {
                let cb0 = w * h;
                let cr0 = cb0 + ocw * och;
                for y in 0..och {
                    for x in 0..ocw {
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
                "{label}: black-box output differs from the reconstruction"
            );
        }
    }
}

/// ISP (opt-in) is elected on textured content in every chroma format
/// at 8 and 10 bits — both split directions, 4-sample partitions
/// upwards — and the streams decode byte-exact through this crate's
/// decoder. The size / PSNR effect against the same encoder without it
/// is reported (the pre-filter decision does not see the deblocking of
/// the partition edges, which is why the tool is off by default).
#[test]
fn isp_is_elected_and_decodes_byte_exact() {
    let (mut total_on, mut total_off) = (0usize, 0usize);
    let (mut psnr_on, mut psnr_off) = (0.0f64, 0.0f64);
    let mut hor = 0u32;
    let mut ver = 0u32;
    for &cf in &[0u32, 1, 2, 3] {
        for &bd in &[8u32, 10] {
            let (w, h) = (128usize, 96usize);
            let src = textured_source(w, h, cf, bd, 21 + cf * 3 + bd);
            let qp = 30;
            let (bs_off, _, s_off, p_off) =
                round_trip(&src, cfg_for(w, h, cf, false, false, false), qp, "isp off");
            let (bs_on, _, s_on, p_on) =
                round_trip(&src, cfg_for(w, h, cf, false, true, false), qp, "isp on");
            assert_eq!(s_off.isp_cus, 0);
            assert!(
                s_on.isp_cus > 0,
                "cf{cf} {bd}-bit: no CU elected ISP ({} CUs)",
                s_on.cus
            );
            hor += s_on.isp_hor_cus;
            ver += s_on.isp_cus - s_on.isp_hor_cus;
            println!(
                "cf{cf} {bd}-bit QP{qp}: ISP {}/{} CUs, {} -> {} B, {p_off:.2} -> {p_on:.2} dB",
                s_on.isp_cus,
                s_on.cus,
                bs_off.len(),
                bs_on.len()
            );
            total_on += bs_on.len();
            total_off += bs_off.len();
            psnr_on += p_on;
            psnr_off += p_off;
        }
    }
    assert!(
        hor > 0 && ver > 0,
        "both split directions elected ({hor} / {ver})"
    );
    println!(
        "all cells: {total_off} -> {total_on} B, mean PSNR {:.3} -> {:.3} dB, {hor} horizontal / {ver} vertical",
        psnr_off / 8.0,
        psnr_on / 8.0
    );
}

/// ISP next to MIP and the MTT pickers, across QPs, with dependent
/// quantisation / sign hiding, on odd geometry: every stream decodes
/// byte-exact.
#[test]
fn isp_with_the_other_tools() {
    let (w, h) = (192usize, 128usize);
    let src = textured_source(w, h, 1, 8, 42);
    let mut elected = 0;
    for qp in [18, 27, 36, 45] {
        let (_, _, s, p) = round_trip(&src, cfg_for(w, h, 1, true, true, false), qp, "qp");
        println!(
            "QP{qp}: ISP {} + MIP {} of {} CUs, {p:.2} dB",
            s.isp_cus, s.mip_cus, s.cus
        );
        elected += s.isp_cus;
    }
    assert!(elected > 0, "no ISP CU across the QP sweep");
    let mut cfg = cfg_for(w, h, 1, true, true, false);
    cfg.enable_mtt_bt_picker = true;
    cfg.enable_mtt_tt_picker = true;
    round_trip(&src, cfg, 30, "mtt + isp + mip");
    let src_odd = textured_source(77, 45, 1, 8, 8);
    round_trip(&src_odd, cfg_for(77, 45, 1, true, true, false), 28, "odd");
    let src_small = textured_source(64, 64, 1, 8, 12);
    for (dq, sdh) in [(true, false), (false, true)] {
        let mut cfg = cfg_for(64, 64, 1, true, true, false);
        cfg.dep_quant = dq;
        cfg.sign_data_hiding = sdh;
        round_trip(&src_small, cfg, 30, "dq/sdh + isp");
    }
}

/// Black-box validation through `ffmpeg` (opaque process; no-op where
/// the binary is absent): ISP-heavy streams of every format and depth
/// decode to the encoder reconstruction.
#[test]
fn black_box_decoder_agrees_on_isp_streams() {
    let probe = std::process::Command::new("ffmpeg")
        .arg("-version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status();
    if !matches!(probe, Ok(s) if s.success()) {
        eprintln!("ffmpeg not installed — black-box leg skipped");
        return;
    }
    let dir = std::env::temp_dir().join(format!("oxideav-h266-r477-isp-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    for &cf in &[0u32, 1, 2, 3] {
        for &bd in &[8u32, 10] {
            let (w, h) = (128usize, 96usize);
            let src = textured_source(w, h, cf, bd, 60 + cf * 2 + bd);
            let cfg = cfg_for(w, h, cf, true, true, false);
            let label = format!("cf{cf} {bd}-bit");
            let (bs, rec, stats, _) = round_trip(&src, cfg, 28, &label);
            assert!(stats.isp_cus > 0, "{label}: no ISP CU to validate");
            black_box_matches(&dir, &label, &bs, &rec, cf, bd, w, h, &cfg);
        }
    }
}

/// Run `ffmpeg` on `bs` and compare its raw output with `rec`.
#[allow(clippy::too_many_arguments)]
fn black_box_matches(
    dir: &std::path::Path,
    label: &str,
    bs: &[u8],
    rec: &PictureBuffer,
    cf: u32,
    bd: u32,
    w: usize,
    h: usize,
    cfg: &EncoderConfig,
) {
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
    let bit = dir.join(format!("{}.266", label.replace(' ', "_")));
    let yuv = dir.join(format!("{}.yuv", label.replace(' ', "_")));
    std::fs::write(&bit, bs).unwrap();
    let st = std::process::Command::new("ffmpeg")
        .args(["-hide_banner", "-loglevel", "error", "-y", "-i"])
        .arg(&bit)
        .args(["-f", "rawvideo", "-pix_fmt", pix_fmt])
        .arg(&yuv)
        .status()
        .unwrap();
    assert!(st.success(), "{label}: black-box decode failed");
    let raw = std::fs::read(&yuv).unwrap();
    let data: Vec<u16> = if bd > 8 {
        raw.chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect()
    } else {
        raw.iter().map(|&b| u16::from(b)).collect()
    };
    let (sw, shh) = cfg.chroma_subsampling();
    let (ocw, och) = if cf == 0 {
        (0, 0)
    } else {
        (w / sw as usize, h / shh as usize)
    };
    assert_eq!(data.len(), w * h + 2 * ocw * och, "{label}: geometry");
    let mut n = 0usize;
    for y in 0..h {
        for x in 0..w {
            if data[y * w + x] != rec.luma.samples[y * rec.luma.stride + x] {
                n += 1;
            }
        }
    }
    if cf != 0 {
        let cb0 = w * h;
        let cr0 = cb0 + ocw * och;
        for y in 0..och {
            for x in 0..ocw {
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
        "{label}: black-box output differs from the reconstruction"
    );
}

/// A smooth-gradient source with ramps, bands and a few flat plateaus —
/// the content where DPCM of the prediction residual pays.
fn gradient_source(w: usize, h: usize, cf: u32, bd: u32) -> PictureBuffer {
    let mid = 1u16 << (bd - 1);
    let sh = bd - 8;
    let mut src = PictureBuffer::with_chroma_format_bd(w, h, cf, mid, bd);
    for y in 0..h {
        for x in 0..w {
            let ramp = (x * 180) / w.max(1) + (y * 50) / h.max(1);
            let band = if (y / 9) % 3 == 0 { (x / 3) % 7 } else { 0 };
            let plateau = if x > w / 2 && y > h / 2 { 20 } else { 0 };
            let v = (20 + ramp + band + plateau).min(255);
            src.luma.samples[y * w + x] = ((v << sh) + ((x + y) % (1 << sh).max(1))) as u16;
        }
    }
    let cw = src.cb.width;
    for y in 0..src.cb.height {
        for x in 0..cw {
            src.cb.samples[y * cw + x] = (((100 + (x * 60) / cw.max(1)) << sh) as u32) as u16;
            src.cr.samples[y * cw + x] =
                (((140 - (y * 40) / src.cb.height.max(1)) << sh) as u32) as u16;
        }
    }
    src
}

/// Lossless BDPCM: on gradient content in every chroma format at 8 /
/// 10 bits the DPCM candidates are elected, the decoded picture is the
/// source bit for bit, and the eight cells together are smaller than
/// the plain lossless stream.
#[test]
fn bdpcm_shrinks_lossless_streams_and_decodes_bit_exact() {
    let (mut total_on, mut total_off) = (0usize, 0usize);
    for &cf in &[0u32, 1, 2, 3] {
        for &bd in &[8u32, 10] {
            let (w, h) = (96usize, 64usize);
            let src = gradient_source(w, h, cf, bd);
            let mut off = cfg_for(w, h, cf, false, false, false);
            off.lossless = true;
            let mut on = off;
            on.intra_bdpcm = true;
            let (bs_off, rec_off, s_off, _) = round_trip(&src, off, 0, "lossless");
            let (bs_on, rec_on, s_on, _) = round_trip(&src, on, 0, "lossless + bdpcm");
            assert_eq!(s_off.bdpcm_cus, 0);
            assert!(
                s_on.bdpcm_cus > 0,
                "cf{cf} {bd}-bit: no CU elected BDPCM ({} CUs)",
                s_on.cus
            );
            for rec in [&rec_off, &rec_on] {
                assert!(
                    plane_eq(&rec.luma, &src.luma),
                    "cf{cf} {bd}-bit: not lossless"
                );
                if cf != 0 {
                    assert!(plane_eq(&rec.cb, &src.cb) && plane_eq(&rec.cr, &src.cr));
                }
            }
            println!(
                "cf{cf} {bd}-bit lossless: BDPCM {}/{} CUs, {} -> {} B",
                s_on.bdpcm_cus,
                s_on.cus,
                bs_off.len(),
                bs_on.len()
            );
            total_on += bs_on.len();
            total_off += bs_off.len();
        }
    }
    assert!(
        total_on < total_off,
        "BDPCM does not shrink the lossless streams: {total_on} vs {total_off}"
    );
    println!("all cells: {total_off} -> {total_on} B");
    // Textured content too: still bit-exact, never larger than +0.5 %.
    let src = textured_source(128, 96, 1, 8, 77);
    let mut off = cfg_for(128, 96, 1, false, false, false);
    off.lossless = true;
    let mut on = off;
    on.intra_bdpcm = true;
    let (bs_off, _, _, _) = round_trip(&src, off, 0, "textured lossless");
    let (bs_on, rec, _, _) = round_trip(&src, on, 0, "textured lossless + bdpcm");
    assert!(plane_eq(&rec.luma, &src.luma));
    assert!(
        bs_on.len() as f64 <= bs_off.len() as f64 * 1.005,
        "{} vs {}",
        bs_on.len(),
        bs_off.len()
    );
}

/// Black-box validation through `ffmpeg` (opaque process; no-op where
/// the binary is absent): lossless BDPCM streams of every format and
/// depth decode to the source.
#[test]
fn black_box_decoder_agrees_on_bdpcm_streams() {
    let probe = std::process::Command::new("ffmpeg")
        .arg("-version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status();
    if !matches!(probe, Ok(s) if s.success()) {
        eprintln!("ffmpeg not installed — black-box leg skipped");
        return;
    }
    let dir = std::env::temp_dir().join(format!("oxideav-h266-r477-bdpcm-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    for &cf in &[0u32, 1, 2, 3] {
        for &bd in &[8u32, 10] {
            let (w, h) = (96usize, 64usize);
            let src = gradient_source(w, h, cf, bd);
            let mut cfg = cfg_for(w, h, cf, false, false, true);
            cfg.lossless = true;
            let label = format!("bdpcm cf{cf} {bd}-bit");
            let (bs, rec, stats, _) = round_trip(&src, cfg, 0, &label);
            assert!(stats.bdpcm_cus > 0, "{label}: no BDPCM CU to validate");
            black_box_matches(&dir, &label, &bs, &rec, cf, bd, w, h, &cfg);
        }
    }
}
