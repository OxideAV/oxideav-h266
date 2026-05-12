//! Round-59 — integration tests for the sub-pel motion-compensation
//! extension to the P-slice encoder + decoder
//! ([`oxideav_h266::encoder_inter::encode_p_slice`] /
//! [`oxideav_h266::encoder_inter::decode_p_slice`]).
//!
//! These live alongside the round-58 integration tests in
//! `round58_pslice_basic.rs` but specifically target the §8.5.6.3
//! 8-tap luma sub-pel interpolation path and the encoder-side
//! half-pel + quarter-pel refinement step.

use oxideav_h266::encoder_inter::{decode_p_slice, encode_p_slice};
use oxideav_h266::encoder_pipeline::{encode_idr_with_residuals, psnr_y};
use oxideav_h266::reconstruct::PictureBuffer;

/// Build a sub-pel-shifted translation pair by oversampling a smooth
/// brightness signal at 16× the luma resolution and resampling each
/// frame at integer-pel positions in that grid. `dx_q16` is the shift
/// in 1/16-luma-sample units (4 → ¼-pel, 8 → ½-pel).
fn subpel_translation_pair(w: usize, h: usize, dx_q16: i32) -> (PictureBuffer, PictureBuffer) {
    let os = 16usize;
    let big_w = w * os;
    let big = |x_q16: i32| -> u8 {
        // Smooth sinusoid: representable by the spec 8-tap filter to
        // high accuracy at fractional positions.
        let phase = (x_q16 as f64) / (big_w as f64) * (5.0 * std::f64::consts::PI);
        let v = 125.0 + 55.0 * phase.sin();
        v.clamp(0.0, 255.0) as u8
    };
    let mut a = PictureBuffer::yuv420_filled(w, h, 100);
    let mut b = PictureBuffer::yuv420_filled(w, h, 100);
    for y in 0..h {
        for x in 0..w {
            a.luma.samples[y * a.luma.stride + x] = big((x * os) as i32);
            let xq = ((x * os) as i32 - dx_q16).clamp(0, (big_w - 1) as i32);
            b.luma.samples[y * b.luma.stride + x] = big(xq);
        }
    }
    (a, b)
}

/// Round-59 — half-pel translation roundtrips through encode + decode
/// and reconstructs above the round-59 PSNR floor.
#[test]
fn round59_half_pel_translation_psnr_clears_30db() {
    let (frame_i, frame_p) = subpel_translation_pair(64, 64, 8);
    let (_, rec_i) = encode_idr_with_residuals(&frame_i, 26).unwrap();
    let (bs_p, rec_p) = encode_p_slice(&frame_p, &rec_i, 26, 1, 8).unwrap();
    assert!(!bs_p.is_empty());
    let psnr = psnr_y(&frame_p.luma, &rec_p.luma).unwrap();
    assert!(
        psnr >= 30.0,
        "half-pel translation PSNR_Y {psnr:.2} dB < 30 dB"
    );
}

/// Round-59 — quarter-pel translation roundtrips through encode +
/// decode at acceptable PSNR. The spec 8-tap filter at ¼-pel
/// positions in a smoothly band-limited signal is near-exact.
#[test]
fn round59_quarter_pel_translation_psnr_clears_30db() {
    let (frame_i, frame_p) = subpel_translation_pair(64, 64, 4);
    let (_, rec_i) = encode_idr_with_residuals(&frame_i, 26).unwrap();
    let (bs_p, rec_p) = encode_p_slice(&frame_p, &rec_i, 26, 1, 8).unwrap();
    assert!(!bs_p.is_empty());
    let psnr = psnr_y(&frame_p.luma, &rec_p.luma).unwrap();
    assert!(
        psnr >= 30.0,
        "quarter-pel translation PSNR_Y {psnr:.2} dB < 30 dB"
    );
}

/// Round-59 — the integer-pel fixture from round 58 must still reach
/// the high-PSNR ballpark (≥ 70 dB) after sub-pel wiring lands.
#[test]
fn round59_integer_pel_regression_holds() {
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
    let (frame_i, frame_p) = translation_pair(4);
    let (_, rec_i) = encode_idr_with_residuals(&frame_i, 26).unwrap();
    let (bs_p, rec_p) = encode_p_slice(&frame_p, &rec_i, 26, 1, 8).unwrap();
    assert!(!bs_p.is_empty());
    let psnr = psnr_y(&frame_p.luma, &rec_p.luma).unwrap();
    assert!(
        psnr >= 70.0,
        "integer-pel regression: PSNR_Y {psnr:.2} dB < 70 dB"
    );
}

/// Round-59 — the encoded sub-pel bitstream decodes to the same luma
/// the encoder kept internally (bit-perfect roundtrip even at
/// fractional MVs).
#[test]
fn round59_subpel_decoder_byte_identical() {
    let (frame_i, frame_p) = subpel_translation_pair(64, 64, 8);
    let (_, rec_i) = encode_idr_with_residuals(&frame_i, 26).unwrap();
    let (bs_p, enc_rec) = encode_p_slice(&frame_p, &rec_i, 26, 1, 8).unwrap();
    let dec_rec = decode_p_slice(&bs_p, &rec_i).unwrap();
    assert_eq!(
        enc_rec.luma.samples, dec_rec.luma.samples,
        "round-59 encoder + decoder P-slice luma must match byte-for-byte at sub-pel MVs",
    );
}
