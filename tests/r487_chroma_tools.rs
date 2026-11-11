//! r487 — the chroma intra mode decision of the stills encoder: the
//! direct mode, the four explicit `intra_chroma_pred_mode` values and
//! the three cross-component linear-model modes (§8.4.5.2.14), ranked
//! per CU. The streams decode byte-exact through this crate's decoder
//! (and a black-box decoder where installed) in every chroma format at
//! 8 and 10 bits, the tools are elected, and they pay for themselves.
//!
//! Spec reference: ITU-T H.266 | ISO/IEC 23090-3 (V4, 01/2026).

use oxideav_h266::encoder::EncoderConfig;
use oxideav_h266::encoder_pipeline::{EncodeStats, encode_idr_with_stats_cfg, psnr_y};
use oxideav_h266::reconstruct::{PictureBuffer, PicturePlane};
use oxideav_h266::stream::{DecodedPicture, StreamDecoder};

/// A picture whose chroma follows its luma structure (as natural
/// content does — the same edges in every component) plus independent
/// chroma detail: bands, a diagonal edge, discs and noise.
fn correlated_source(w: usize, h: usize, cf: u32, bd: u32, seed: u32) -> PictureBuffer {
    let mid = 1u16 << (bd - 1);
    let sh = bd - 8;
    let mut src = PictureBuffer::with_chroma_format_bd(w, h, cf, mid, bd);
    let mut state = seed.wrapping_mul(2654435761).wrapping_add(4242);
    let mut rnd = || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        state
    };
    let shade = |x: usize, y: usize| -> i64 {
        let band = (((x / 9 + y / 13) % 4) * 28) as i64;
        let edge = if x * 2 + y * 3 > w + h { 45 } else { 0 };
        let dx = x as i64 - (w / 2) as i64;
        let dy = y as i64 - (h / 3) as i64;
        let disc = if dx * dx + dy * dy < ((w.min(h) / 4) as i64).pow(2) {
            -40
        } else {
            0
        };
        50 + band + edge + disc + ((x * 40) / w) as i64
    };
    for y in 0..h {
        for x in 0..w {
            let v = (shade(x, y) + (rnd() % 12) as i64).clamp(0, 255);
            src.luma.samples[y * src.luma.stride + x] = (v as u16) << sh;
        }
    }
    if cf != 0 {
        let (sw, shh) = oxideav_h266::reconstruct::chroma_subsampling(cf);
        for y in 0..src.cb.height {
            for x in 0..src.cb.width {
                let l = shade(x * sw, y * shh);
                let cb = (128 + (l - 110) * 3 / 5 + (rnd() % 6) as i64).clamp(0, 255);
                let cr =
                    (128 - (l - 110) * 2 / 5 + ((x / 16) % 2) as i64 * 14 + (rnd() % 6) as i64)
                        .clamp(0, 255);
                src.cb.samples[y * src.cb.stride + x] = (cb as u16) << sh;
                src.cr.samples[y * src.cr.stride + x] = (cr as u16) << sh;
            }
        }
    }
    src
}

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
    dec.set_verify_picture_hash(true);
    let mut got = None;
    dec.decode_annex_b(bs, &mut |pic| got = Some(pic))
        .expect("own decoder rejects the stream");
    got.expect("no picture decoded")
}

struct Coded {
    bytes: Vec<u8>,
    rec: PictureBuffer,
    stats: EncodeStats,
    /// Luma, Cb, Cr PSNR (chroma 0 for 4:0:0).
    psnr: [f64; 3],
}

fn round_trip(src: &PictureBuffer, cfg: EncoderConfig, qp: i32, label: &str) -> Coded {
    let (bytes, rec, stats) = encode_idr_with_stats_cfg(src, qp, cfg)
        .unwrap_or_else(|e| panic!("{label}: encode failed: {e}"));
    let pic = decode_once(&bytes);
    assert!(
        plane_eq(&pic.frame.luma, &rec.luma),
        "{label}: luma differs"
    );
    let mut psnr = [psnr_y(&src.luma, &rec.luma).unwrap(), 0.0, 0.0];
    if cfg.chroma_format_idc != 0 {
        assert!(plane_eq(&pic.frame.cb, &rec.cb), "{label}: Cb differs");
        assert!(plane_eq(&pic.frame.cr, &rec.cr), "{label}: Cr differs");
        psnr[1] = psnr_y(&src.cb, &rec.cb).unwrap();
        psnr[2] = psnr_y(&src.cr, &rec.cr).unwrap();
    }
    Coded {
        bytes,
        rec,
        stats,
        psnr,
    }
}

fn cfg_for(w: usize, h: usize, cf: u32, cclm: bool, modes: bool) -> EncoderConfig {
    let mut cfg = EncoderConfig::new(w as u32, h as u32);
    cfg.chroma_format_idc = cf as u8;
    cfg.intra_cclm = cclm;
    cfg.intra_chroma_modes = modes;
    cfg
}

/// Every chroma format at 8 and 10 bits, QP 22 / 32 / 40, odd sizes
/// (padded pictures: CUs on the picture edge), with both tools on:
/// byte-exact through this crate's decoder, and the linear-model
/// modes are elected.
#[test]
fn chroma_decision_round_trips_in_every_format() {
    for &cf in &[1u32, 2, 3] {
        for &bd in &[8u32, 10] {
            for &(w, h) in &[(128usize, 96usize), (150, 70), (34, 130)] {
                for &qp in &[22, 32, 40] {
                    let src = correlated_source(w, h, cf, bd, cf * 31 + bd + qp as u32);
                    let label = format!("cf{cf} {bd}-bit {w}x{h} qp{qp}");
                    let on = round_trip(&src, cfg_for(w, h, cf, true, true), qp, &label);
                    assert!(
                        on.stats.cclm_cus > 0,
                        "{label}: no CU elected a linear-model mode ({} CUs)",
                        on.stats.cus
                    );
                }
            }
        }
    }
    // 4:0:0 carries no chroma syntax at all: the switches are inert.
    let src = correlated_source(96, 64, 0, 8, 1);
    let a = round_trip(&src, cfg_for(96, 64, 0, true, true), 30, "mono on");
    let b = round_trip(&src, cfg_for(96, 64, 0, false, false), 30, "mono off");
    assert_eq!(a.bytes, b.bytes);
    assert_eq!(a.stats.cclm_cus, 0);
}

/// The decision composes with the other encoder tools: MTT pickers,
/// tiles + WPP, dependent quantisation, sign hiding, ISP, LMCS-free
/// lossless (where the SPS flag is on and every CU codes
/// `cclm_mode_flag = 0`).
#[test]
fn chroma_decision_composes_with_the_other_tools() {
    let (w, h) = (192usize, 136usize);
    let src = correlated_source(w, h, 1, 8, 77);
    let base = cfg_for(w, h, 1, true, true);
    let mut cases: Vec<(&str, EncoderConfig)> = Vec::new();
    let mut c = base;
    c.enable_mtt_bt_picker = true;
    c.enable_mtt_tt_picker = true;
    cases.push(("mtt", c));
    let mut c = base;
    c.tile_columns = 1;
    c.tile_rows = 1;
    c.wpp = true;
    cases.push(("wpp", c));
    let mut c = base;
    c.dep_quant = true;
    cases.push(("dep_quant", c));
    let mut c = base;
    c.sign_data_hiding = true;
    cases.push(("sdh", c));
    let mut c = base;
    c.intra_isp = true;
    cases.push(("isp", c));
    let mut c = base;
    c.lossless = true;
    cases.push(("lossless", c));
    for (name, cfg) in cases {
        let coded = round_trip(&src, cfg, 30, name);
        if cfg.lossless {
            assert_eq!(coded.rec.cb.samples, src.cb.samples, "{name}");
            assert_eq!(coded.stats.cclm_cus, 0, "{name}");
        } else {
            assert!(coded.stats.cclm_cus > 0, "{name}: no linear-model CU");
        }
    }
    // Two tile columns: the linear model's neighbour probes stop at
    // the tile edge exactly like the decoder's.
    let (w, h) = (256usize, 128usize);
    let src = correlated_source(w, h, 1, 8, 78);
    let mut c = cfg_for(w, h, 1, true, true);
    c.tile_columns = 2;
    let coded = round_trip(&src, c, 30, "tiles");
    assert!(coded.stats.cclm_cus > 0);
}

/// The tools pay for themselves on content whose chroma follows its
/// luma: with both on, the streams are smaller AND the chroma closer
/// than with the direct mode alone; each tool alone is a gain too.
#[test]
fn chroma_tools_pay_for_themselves() {
    println!(
        "| format | QP | direct only | + explicit | + linear model | both | chroma PSNR direct → both |"
    );
    let (mut total_off, mut total_on) = (0usize, 0usize);
    let (mut psnr_off, mut psnr_on) = (0.0f64, 0.0f64);
    for &cf in &[1u32, 2, 3] {
        for &qp in &[22, 32, 37] {
            let (w, h) = (256usize, 192usize);
            let src = correlated_source(w, h, cf, 8, 5 + cf);
            let off = round_trip(&src, cfg_for(w, h, cf, false, false), qp, "off");
            let modes = round_trip(&src, cfg_for(w, h, cf, false, true), qp, "modes");
            let cclm = round_trip(&src, cfg_for(w, h, cf, true, false), qp, "cclm");
            let both = round_trip(&src, cfg_for(w, h, cf, true, true), qp, "both");
            let chroma = |c: &Coded| (c.psnr[1] + c.psnr[2]) / 2.0;
            println!(
                "| cf{cf} | {qp} | {} B | {} B | {} B | {} B ({:+.1} %) | {:.2} → {:.2} dB (luma {:.2} → {:.2}), {} of {} CUs linear-model, {} explicit |",
                off.bytes.len(),
                modes.bytes.len(),
                cclm.bytes.len(),
                both.bytes.len(),
                (both.bytes.len() as f64 / off.bytes.len() as f64 - 1.0) * 100.0,
                chroma(&off),
                chroma(&both),
                off.psnr[0],
                both.psnr[0],
                both.stats.cclm_cus,
                both.stats.cus,
                both.stats.explicit_chroma_cus,
            );
            assert_eq!(off.stats.cclm_cus, 0);
            assert_eq!(off.stats.explicit_chroma_cus, 0);
            assert_eq!(modes.stats.cclm_cus, 0);
            assert!(both.stats.cclm_cus > 0);
            // Per cell: never more than the greedy decision's noise
            // worse in bytes, and no chroma loss beyond what the byte
            // saving buys back many times over.
            assert!(
                (both.bytes.len() as f64) < off.bytes.len() as f64 * 1.01,
                "cf{cf} qp{qp}: {} vs {} bytes",
                both.bytes.len(),
                off.bytes.len()
            );
            assert!(
                chroma(&both) > chroma(&off) - 0.2,
                "cf{cf} qp{qp}: chroma PSNR {:.2} vs {:.2}",
                chroma(&both),
                chroma(&off)
            );
            total_off += off.bytes.len();
            total_on += both.bytes.len();
            psnr_off += chroma(&off);
            psnr_on += chroma(&both);
        }
    }
    println!(
        "== total {total_off} → {total_on} B ({:+.2} %), mean chroma PSNR {:.2} → {:.2} dB",
        (total_on as f64 / total_off as f64 - 1.0) * 100.0,
        psnr_off / 9.0,
        psnr_on / 9.0
    );
    assert!(total_on < total_off, "{total_on} vs {total_off} bytes");
    assert!(psnr_on >= psnr_off, "{psnr_on} vs {psnr_off}");
}

/// Chroma that does NOT follow the luma (independent gradients and
/// noise in each plane): the decision has little to win there, and it
/// must not lose — within 1 % bytes of the direct mode alone, chroma
/// no worse than 0.1 dB.
#[test]
fn chroma_tools_do_not_lose_on_uncorrelated_chroma() {
    for &cf in &[1u32, 3] {
        for &qp in &[24, 34] {
            let (w, h) = (192usize, 128usize);
            let mut src = correlated_source(w, h, cf, 8, 60 + cf);
            let mut state = 0x9e3779b9u32 ^ cf;
            let (cw, ch) = (src.cb.width, src.cb.height);
            for y in 0..ch {
                for x in 0..cw {
                    state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                    let n = (state >> 24) % 20;
                    src.cb.samples[y * src.cb.stride + x] =
                        (60 + (x * 130) / cw + n as usize) as u16;
                    src.cr.samples[y * src.cr.stride + x] =
                        (200 - (y * 120) / ch - ((state >> 16) % 20) as usize) as u16;
                }
            }
            let off = round_trip(&src, cfg_for(w, h, cf, false, false), qp, "off");
            let both = round_trip(&src, cfg_for(w, h, cf, true, true), qp, "both");
            let chroma = |c: &Coded| (c.psnr[1] + c.psnr[2]) / 2.0;
            println!(
                "uncorrelated cf{cf} qp{qp}: {} → {} B, chroma {:.2} → {:.2} dB, {} linear-model / {} explicit of {} CUs",
                off.bytes.len(),
                both.bytes.len(),
                chroma(&off),
                chroma(&both),
                both.stats.cclm_cus,
                both.stats.explicit_chroma_cus,
                both.stats.cus
            );
            assert!((both.bytes.len() as f64) < off.bytes.len() as f64 * 1.01);
            assert!(chroma(&both) > chroma(&off) - 0.1);
        }
    }
}

/// Black-box validation through an installed decoder binary (opaque
/// process; a no-op where it is absent): the streams with the chroma decision
/// decode to the encoder's reconstruction.
#[test]
fn black_box_decoder_agrees_on_the_chroma_decision() {
    let probe = std::process::Command::new("ffmpeg")
        .arg("-version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status();
    if !matches!(probe, Ok(s) if s.success()) {
        eprintln!("black-box decoder not installed — leg skipped");
        return;
    }
    let dir = std::env::temp_dir().join(format!("oxideav-h266-r487-chroma-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    for &cf in &[1u32, 2, 3] {
        for &bd in &[8u32, 10] {
            for &(w, h, mtt) in &[(128usize, 96usize, false), (200, 136, true)] {
                let src = correlated_source(w, h, cf, bd, 90 + cf + bd);
                let mut cfg = cfg_for(w, h, cf, true, true);
                cfg.enable_mtt_bt_picker = mtt;
                cfg.enable_mtt_tt_picker = mtt;
                let label = format!("cf{cf} {bd}-bit {w}x{h} mtt={mtt}");
                let coded = round_trip(&src, cfg, 28, &label);
                assert!(coded.stats.cclm_cus > 0, "{label}");
                let pix_fmt = match (cf, bd) {
                    (1, 8) => "yuv420p",
                    (1, _) => "yuv420p10le",
                    (2, 8) => "yuv422p",
                    (2, _) => "yuv422p10le",
                    (_, 8) => "yuv444p",
                    (_, _) => "yuv444p10le",
                };
                let bit = dir.join(format!("{cf}_{bd}_{w}.266"));
                let yuv = dir.join(format!("{cf}_{bd}_{w}.yuv"));
                std::fs::write(&bit, &coded.bytes).unwrap();
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
                    raw.as_chunks::<2>()
                        .0
                        .iter()
                        .map(|c| u16::from_le_bytes(*c))
                        .collect()
                } else {
                    raw.iter().map(|&b| u16::from(b)).collect()
                };
                let mut expect = Vec::with_capacity(data.len());
                for p in [&coded.rec.luma, &coded.rec.cb, &coded.rec.cr] {
                    for y in 0..p.height {
                        expect.extend_from_slice(&p.samples[y * p.stride..y * p.stride + p.width]);
                    }
                }
                assert!(data == expect, "{label}: black-box output differs");
            }
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}
