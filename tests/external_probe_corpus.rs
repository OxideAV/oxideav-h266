//! r415 — external-decoder probe corpus. Every test encodes a
//! deterministic minimal-content stream through the IDR pipeline; with
//! `H266_CORPUS_DIR=<dir>` set, the Annex-B stream and the encoder
//! reconstruction (planar YUV420) are dumped for black-box validation
//! against a conforming reference decoder:
//!
//! ```text
//! H266_CORPUS_DIR=<dir> cargo test --test external_probe_corpus
//! ffmpeg -i <name>.266 -f rawvideo -pix_fmt yuv420p <name>.ref.yuv
//! cmp <name>.yuv <name>.ref.yuv
//! ```
//!
//! The probe axes bisect single features: TB sizes 8..64, single
//! luma/chroma coefficients by position (frequency bases), amplitude
//! sweeps, gradient/stripe content, chroma-only planes, QP sweep, and
//! the 128x128 full-CTB walk. `tests/WHOLE_STREAM_CORPUS.md` records
//! the r415 validation status. These probes bisected the five r415
//! external-conformance root causes (residual ctx-init tables, chroma
//! last-prefix ctxShift, alf_use_aps_flag presence, per-CU QG wire,
//! ALF virtual-boundary classification, chroma QP table, chroma CTB
//! deblocking).

use oxideav_h266::encoder_pipeline::encode_idr_with_residuals;
use oxideav_h266::reconstruct::PictureBuffer;

fn dump(name: &str, bs: &[u8], dec: &PictureBuffer) {
    let Ok(dir) = std::env::var("H266_CORPUS_DIR") else {
        return;
    };
    let base = std::path::Path::new(&dir);
    std::fs::create_dir_all(base).unwrap();
    std::fs::write(base.join(format!("{name}.266")), bs).unwrap();
    let mut yuv = Vec::new();
    yuv.extend_from_slice(&dec.luma.samples);
    yuv.extend_from_slice(&dec.cb.samples);
    yuv.extend_from_slice(&dec.cr.samples);
    std::fs::write(base.join(format!("{name}.yuv")), yuv).unwrap();
}

/// Encode + dump one probe; the encoder reconstruction is the ground
/// truth (own receive path already covered by the conformance suite).
fn probe(name: &str, src: &PictureBuffer, qp: i32) {
    let (bs, rec) = encode_idr_with_residuals(src, qp).unwrap();
    dump(name, &bs, &rec);
}

/// Luma-only diagonal gradient (corpus pattern), flat chroma.
#[test]
fn probe_luma_gradient_flat_chroma() {
    for (tag, amp) in [("lad1", 1usize), ("lad2", 2), ("lad4", 4), ("lad8", 8)] {
        let mut src = PictureBuffer::yuv420_filled(64, 64, 128);
        for y in 0..64 {
            for x in 0..64 {
                src.luma.samples[y * src.luma.stride + x] =
                    (40 + ((x * 3 + y * 2) * amp) % 160) as u8;
            }
        }
        probe(&format!("p_{tag}"), &src, 26);
    }
}

/// Flat luma, chroma ramps only.
#[test]
fn probe_flat_luma_chroma_ramps() {
    let mut src = PictureBuffer::yuv420_filled(64, 64, 128);
    for y in 0..32 {
        for x in 0..32 {
            src.cb.samples[y * src.cb.stride + x] = (96 + (x % 32)) as u8;
            src.cr.samples[y * src.cr.stride + x] = (160 - (y % 32)) as u8;
        }
    }
    probe("p_chroma_only", &src, 26);
}

/// Single luma bump of varying amplitude (sparse-residual family).
#[test]
fn probe_luma_bump_sweep() {
    for amp in [8i32, 12, 16, 20, 24, 32, 40, 48, 64, 96] {
        let mut src = PictureBuffer::yuv420_filled(64, 64, 128);
        for y in 0..8 {
            for x in 0..8 {
                let v = 128 + amp * (((x + y) % 2) as i32 * 2 - 1);
                src.luma.samples[y * src.luma.stride + x] = v.clamp(0, 255) as u8;
            }
        }
        probe(&format!("p_bump{amp}"), &src, 26);
    }
}

/// Horizontal cosine-ish stripe → low-frequency row coefficients.
#[test]
fn probe_luma_stripes() {
    for period in [2usize, 4, 8, 16, 32] {
        let mut src = PictureBuffer::yuv420_filled(64, 64, 128);
        for y in 0..64 {
            for x in 0..64 {
                let v = if (x / period) % 2 == 0 { 148 } else { 108 };
                src.luma.samples[y * src.luma.stride + x] = v;
            }
        }
        probe(&format!("p_stripe{period}"), &src, 26);
    }
}

/// Tiny 16x16 picture — smallest CTU clip; gradient content.
#[test]
fn probe_small_16x16() {
    let mut src = PictureBuffer::yuv420_filled(16, 16, 128);
    for y in 0..16 {
        for x in 0..16 {
            src.luma.samples[y * src.luma.stride + x] = (40 + (x * 3 + y * 2) % 160) as u8;
        }
    }
    probe("p_16x16", &src, 26);
}

/// 32x32 gradient — one 32x32 luma TB, no 64-point zero-out.
#[test]
fn probe_small_32x32() {
    let mut src = PictureBuffer::yuv420_filled(32, 32, 128);
    for y in 0..32 {
        for x in 0..32 {
            src.luma.samples[y * src.luma.stride + x] = (40 + (x * 3 + y * 2) % 160) as u8;
        }
    }
    probe("p_32x32", &src, 26);
}

/// QP sweep on the 64x64 luma gradient with flat chroma.
#[test]
fn probe_gradient_qp_sweep() {
    for qp in [10i32, 17, 34, 45] {
        let mut src = PictureBuffer::yuv420_filled(64, 64, 128);
        for y in 0..64 {
            for x in 0..64 {
                src.luma.samples[y * src.luma.stride + x] = (40 + (x * 3 + y * 2) % 160) as u8;
            }
        }
        probe(&format!("p_gradqp{qp}"), &src, qp);
    }
}

/// Series A — NxN gradient pictures: TB size sweep.
#[test]
fn probe_series_a_tb_size() {
    for n in [8usize, 16, 32, 64] {
        let mut src = PictureBuffer::yuv420_filled(n, n, 128);
        for y in 0..n {
            for x in 0..n {
                src.luma.samples[y * src.luma.stride + x] = (40 + (x * 3 + y * 2) % 160) as u8;
            }
        }
        probe(&format!("pa_n{n}"), &src, 26);
    }
}

/// Series B — 32x32 picture, gradient confined to top-left KxK:
/// fixed TB size, varying last-significant reach.
#[test]
fn probe_series_b_extent() {
    for k in [4usize, 8, 16, 24, 32] {
        let mut src = PictureBuffer::yuv420_filled(32, 32, 128);
        for y in 0..k {
            for x in 0..k {
                src.luma.samples[y * src.luma.stride + x] = (40 + (x * 5 + y * 3) % 160) as u8;
            }
        }
        probe(&format!("pb_k{k}"), &src, 26);
    }
}

/// Series C — 32x32 picture, single horizontal cosine of frequency k:
/// concentrates energy at coefficient (k, 0).
#[test]
fn probe_series_c_freq_x() {
    for k in [1usize, 2, 3, 4, 6, 8, 12, 16, 24, 31] {
        let mut src = PictureBuffer::yuv420_filled(32, 32, 128);
        for y in 0..32 {
            for x in 0..32 {
                let arg = std::f64::consts::PI * (k as f64) * (2.0 * x as f64 + 1.0) / 64.0;
                let v = 128.0 + 24.0 * arg.cos();
                src.luma.samples[y * src.luma.stride + x] = v.round().clamp(0.0, 255.0) as u8;
            }
        }
        probe(&format!("pc_fx{k}"), &src, 26);
    }
}

/// Series D — same but vertical frequency: coefficient (0, k).
#[test]
fn probe_series_d_freq_y() {
    for k in [1usize, 2, 3, 4, 6, 8, 12, 16, 24, 31] {
        let mut src = PictureBuffer::yuv420_filled(32, 32, 128);
        for y in 0..32 {
            for x in 0..32 {
                let arg = std::f64::consts::PI * (k as f64) * (2.0 * y as f64 + 1.0) / 64.0;
                let v = 128.0 + 24.0 * arg.cos();
                src.luma.samples[y * src.luma.stride + x] = v.round().clamp(0.0, 255.0) as u8;
            }
        }
        probe(&format!("pd_fy{k}"), &src, 26);
    }
}

/// Series E — chroma-content family bisection (64x64 picture).
#[test]
fn probe_series_e_chroma() {
    // E1: single Cb DC bump, amplitude sweep.
    for amp in [4i32, 8, 16, 32] {
        let mut src = PictureBuffer::yuv420_filled(64, 64, 128);
        for y in 0..4 {
            for x in 0..4 {
                src.cb.samples[y * src.cb.stride + x] = (128 + amp).clamp(0, 255) as u8;
            }
        }
        probe(&format!("pe_cbdc{amp}"), &src, 26);
    }
    // E2: Cb horizontal ramp confined to top-left KxK.
    for k in [4usize, 8, 16, 32] {
        let mut src = PictureBuffer::yuv420_filled(64, 64, 128);
        for y in 0..k {
            for x in 0..k {
                src.cb.samples[y * src.cb.stride + x] = (96 + x * 2) as u8;
            }
        }
        probe(&format!("pe_cbramp{k}"), &src, 26);
    }
    // E3: Cb-only vs Cr-only vs both (full 32x32 ramps).
    let ramp = |cb: bool, cr: bool| {
        let mut src = PictureBuffer::yuv420_filled(64, 64, 128);
        for y in 0..32 {
            for x in 0..32 {
                if cb {
                    src.cb.samples[y * src.cb.stride + x] = (96 + x) as u8;
                }
                if cr {
                    src.cr.samples[y * src.cr.stride + x] = (160 - y) as u8;
                }
            }
        }
        src
    };
    probe("pe_cbonly", &ramp(true, false), 26);
    probe("pe_cronly", &ramp(false, true), 26);
    probe("pe_cbcr", &ramp(true, true), 26);
    // E4: chroma ramp in a 32x32 picture (16x16 chroma TB).
    let mut src = PictureBuffer::yuv420_filled(32, 32, 128);
    for y in 0..16 {
        for x in 0..16 {
            src.cb.samples[y * src.cb.stride + x] = (96 + x * 2) as u8;
        }
    }
    probe("pe_small_cb", &src, 26);
}

/// Series F — minimal chroma residual: full-plane Cb DC offset.
#[test]
fn probe_series_f_chroma_dc() {
    for off in [2i32, 3, 4, 6, 8, 12] {
        let mut src = PictureBuffer::yuv420_filled(64, 64, 128);
        for y in 0..32 {
            for x in 0..32 {
                src.cb.samples[y * src.cb.stride + x] = (128 + off) as u8;
            }
        }
        probe(&format!("pf_cboff{off}"), &src, 26);
    }
}

/// Series G — single chroma coefficient at (k,0) via cosine basis.
#[test]
fn probe_series_g_chroma_freq() {
    for k in [1usize, 2, 3, 4, 6, 8, 12, 16] {
        let mut src = PictureBuffer::yuv420_filled(64, 64, 128);
        for y in 0..32 {
            for x in 0..32 {
                let arg = std::f64::consts::PI * (k as f64) * (2.0 * x as f64 + 1.0) / 64.0;
                let v = 128.0 + 12.0 * arg.cos();
                src.cb.samples[y * src.cb.stride + x] = v.round().clamp(0.0, 255.0) as u8;
            }
        }
        probe(&format!("pg_cbfx{k}"), &src, 26);
    }
}

/// Series H — discriminator probes: chroma (1,1) basis (sig ctx 36,
/// last-prefix ctx <= 20) vs (2,0)/(0,2)-style content.
#[test]
fn probe_series_h_chroma_disc() {
    let basis = |kx: usize, ky: usize, name: &str| {
        let mut src = PictureBuffer::yuv420_filled(64, 64, 128);
        for y in 0..32 {
            for x in 0..32 {
                let ax = std::f64::consts::PI * (kx as f64) * (2.0 * x as f64 + 1.0) / 64.0;
                let ay = std::f64::consts::PI * (ky as f64) * (2.0 * y as f64 + 1.0) / 64.0;
                let v = 128.0 + 14.0 * ax.cos() * ay.cos();
                src.cb.samples[y * src.cb.stride + x] = v.round().clamp(0.0, 255.0) as u8;
            }
        }
        probe(name, &src, 26);
    };
    basis(1, 1, "ph_cb11");
    basis(0, 2, "ph_cb02");
    basis(2, 0, "ph_cb20");
    basis(0, 1, "ph_cb01");
}

/// Series I — 128x128 feature bisection of the corpus structured source.
#[test]
fn probe_series_i_128() {
    let structured = |luma_grad: bool, blocks: bool, chroma: bool| {
        let mut src = PictureBuffer::yuv420_filled(128, 128, 128);
        for y in 0..128 {
            for x in 0..128 {
                let mut v = 128usize;
                if luma_grad {
                    v = 40 + ((x * 3 + y * 2) % 160);
                }
                if blocks && (x / 16 + y / 16) % 2 == 0 {
                    v += 20;
                }
                src.luma.samples[y * src.luma.stride + x] = v as u8;
            }
        }
        if chroma {
            for y in 0..64 {
                for x in 0..64 {
                    src.cb.samples[y * src.cb.stride + x] = (96 + (x % 64)) as u8;
                    src.cr.samples[y * src.cr.stride + x] = (160 - (y % 64)) as u8;
                }
            }
        }
        src
    };
    probe("pi_grad", &structured(true, false, false), 26);
    probe("pi_gradblocks", &structured(true, true, false), 26);
    probe("pi_chroma", &structured(false, false, true), 26);
    probe("pi_full", &structured(true, true, true), 26);
    // 64x64 with the same combined content (control).
    let mut src = PictureBuffer::yuv420_filled(64, 64, 128);
    for y in 0..64 {
        for x in 0..64 {
            let mut v = 40 + ((x * 3 + y * 2) % 160);
            if (x / 16 + y / 16) % 2 == 0 {
                v += 20;
            }
            src.luma.samples[y * src.luma.stride + x] = v as u8;
        }
    }
    for y in 0..32 {
        for x in 0..32 {
            src.cb.samples[y * src.cb.stride + x] = (96 + x) as u8;
            src.cr.samples[y * src.cr.stride + x] = (160 - y) as u8;
        }
    }
    probe("pi_full64", &src, 26);
}
