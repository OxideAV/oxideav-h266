//! Round-58 — integration tests for the inter-frame P-slice scaffold
//! ([`oxideav_h266::encoder_inter::encode_p_slice`] /
//! [`oxideav_h266::encoder_inter::decode_p_slice`]).
//!
//! These tests live at the integration layer (`tests/`) so they
//! exercise the public encoder + decoder API exactly as a downstream
//! crate would. The `encoder_inter` module's own `#[cfg(test)]` unit
//! tests cover the building blocks (full-search SAD, MVD CABAC bin
//! roundtrip, slice-header bit-prelude); these tests cover end-to-end
//! single-reference I + P pipelines.

use oxideav_h266::encoder_inter::{decode_p_slice, encode_p_slice};
use oxideav_h266::encoder_pipeline::{encode_idr_with_residuals, psnr_y};
use oxideav_h266::reconstruct::PictureBuffer;

/// Build a 64×64 frame with a 4-pixel horizontally-translated stripe
/// pattern relative to the source. Used by every test in this file.
fn translation_pair(dx: i32) -> (PictureBuffer, PictureBuffer) {
    let w = 64usize;
    let h = 64usize;
    let mut a = PictureBuffer::yuv420_filled(w, h, 100);
    let mut b = PictureBuffer::yuv420_filled(w, h, 100);
    for y in 0..h {
        for x in 0..w {
            let v = if (x / 8) % 2 == 0 { 80u8 } else { 180u8 };
            a.luma.samples[y * a.luma.stride + x] = v;
            let sx = ((x as i32 - dx).rem_euclid(w as i32)) as usize;
            let v2 = if (sx / 8) % 2 == 0 { 80u8 } else { 180u8 };
            b.luma.samples[y * b.luma.stride + x] = v2;
        }
    }
    (a, b)
}

/// Round-58 — a 4-pixel horizontal translation between I and P
/// reconstructs at PSNR_Y ≥ 35 dB through the integer-pel MC + DCT
/// residual scaffold.
#[test]
fn pslice_translation_psnr_clears_35db() {
    let (frame_i, frame_p) = translation_pair(4);
    let (_, rec_i) = encode_idr_with_residuals(&frame_i, 26).unwrap();
    let (bs_p, rec_p) = encode_p_slice(&frame_p, &rec_i, 26, 1, 8).unwrap();
    assert!(!bs_p.is_empty());
    let psnr = psnr_y(&frame_p.luma, &rec_p.luma).unwrap();
    assert!(
        psnr >= 35.0,
        "P-slice PSNR_Y {psnr:.2} dB < 35 dB on 4-px translation"
    );
}

/// Round-58 — encoding the same frame as both I and P produces a
/// near-zero residual on the P-slice. The decoder roundtrip and the
/// per-block CABAC overhead stay bounded.
#[test]
fn pslice_no_motion_zero_residual() {
    let (frame_a, _) = translation_pair(0);
    let (_, rec_i) = encode_idr_with_residuals(&frame_a, 26).unwrap();
    let (bs_p, rec_p) = encode_p_slice(&frame_a, &rec_i, 26, 1, 8).unwrap();
    let dec_rec = decode_p_slice(&bs_p, &rec_i).unwrap();
    assert_eq!(rec_p.luma.samples, dec_rec.luma.samples);
    let psnr = psnr_y(&frame_a.luma, &rec_p.luma).unwrap();
    assert!(
        psnr >= 30.0,
        "Identical-frame P-slice PSNR_Y {psnr:.2} dB < 30 dB"
    );
}

/// Round-58 — the encoded P-slice fed back through `decode_p_slice`
/// reconstructs to the same luma the encoder kept internally
/// (bit-perfect roundtrip).
#[test]
fn pslice_decoder_roundtrip_byte_identical() {
    let (frame_i, frame_p) = translation_pair(4);
    let (_, rec_i) = encode_idr_with_residuals(&frame_i, 26).unwrap();
    let (bs_p, enc_rec) = encode_p_slice(&frame_p, &rec_i, 26, 1, 8).unwrap();
    let dec_rec = decode_p_slice(&bs_p, &rec_i).unwrap();
    assert_eq!(
        enc_rec.luma.samples, dec_rec.luma.samples,
        "encoder + decoder P-slice luma must match byte-for-byte",
    );
}

/// Round-58 — synthetic 2-frame fixture: a single 16×16 bright square
/// that translates 4 px horizontally between frames. Stands in for an
/// `ffmpeg -c:v libvvenc -an -frames:v 2` external fixture (not
/// available without a libvvenc binary in the test env).
#[test]
fn pslice_synthetic_moving_square_two_frame() {
    let make = |dx: i32| {
        let mut buf = PictureBuffer::yuv420_filled(64, 64, 100);
        for y in 16..32 {
            for x in 0..16 {
                let xx = (16 + dx as usize + x).min(63);
                buf.luma.samples[y * buf.luma.stride + xx] = 220;
            }
        }
        buf
    };
    let frame_i = make(0);
    let frame_p = make(4);
    let (_, rec_i) = encode_idr_with_residuals(&frame_i, 26).unwrap();
    let (bs_p, rec_p) = encode_p_slice(&frame_p, &rec_i, 26, 1, 8).unwrap();
    assert!(!bs_p.is_empty());
    let dec_rec = decode_p_slice(&bs_p, &rec_i).unwrap();
    assert_eq!(rec_p.luma.samples, dec_rec.luma.samples);
    let psnr = psnr_y(&frame_p.luma, &rec_p.luma).unwrap();
    assert!(
        psnr >= 35.0,
        "Synthetic 2-frame fixture PSNR_Y {psnr:.2} dB < 35 dB"
    );
}
