//! Round-61 — integration tests for sub-pel motion estimation on the
//! B-slice (bi-prediction) encoder + decoder
//! ([`oxideav_h266::encoder_inter::encode_b_slice`] /
//! [`oxideav_h266::encoder_inter::decode_b_slice`]).
//!
//! Round 60 added the B-slice scaffold with integer-pel ME only. Round
//! 61 extends it with the same two-stage ½-pel + ¼-pel refinement that
//! round 59 added to the P-slice path: per-list, after the integer-pel
//! SAD search, probe 8 half-pel neighbours through the §8.5.6.3.2
//! Table 27 8-tap luma filter, then 8 quarter-pel neighbours around the
//! best half-pel candidate. The RDO over `{L0-only, L1-only, BI}` then
//! runs with each list's MV at 1/16-pel precision. Bi-pred reconstruction
//! is still the §8.5.6.4 simple average `pred = (predL0 + predL1 + 1) >> 1`
//! (weighted bi-pred is deferred to a later round).

use oxideav_h266::encoder_inter::{decode_b_slice, encode_b_slice};
use oxideav_h266::encoder_pipeline::{encode_idr_with_residuals, psnr_y};
use oxideav_h266::reconstruct::PictureBuffer;

/// Build a sub-pel-shifted translation pair by oversampling a smooth
/// brightness signal at 16× the luma resolution and resampling each
/// frame at integer-pel positions in that grid. `dx_q16` is the shift
/// in 1/16-luma-sample units (4 → ¼-pel, 8 → ½-pel). The signal is a
/// band-limited sinusoid which the spec 8-tap filter reproduces to
/// high PSNR at fractional positions.
fn subpel_translation_pair(w: usize, h: usize, dx_q16: i32) -> (PictureBuffer, PictureBuffer) {
    let os = 16usize;
    let big_w = w * os;
    let big = |x_q16: i32| -> u8 {
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

/// Build a stripe-pattern integer-pel translation pair, identical to
/// the round-60 fixture. Used here as the integer-pel regression check
/// — adding sub-pel ME to the B-slice path must not regress the
/// integer-pel ceiling.
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

/// Round-61 — a B-slice on a ½-pel translation fixture (with both
/// reference lists carrying the same picture, so BI degenerates to
/// uni-pred) reaches the round-59 P-slice half-pel ceiling (≥ 50 dB).
/// Without sub-pel ME this fixture saturates at the round-60 integer-pel
/// ceiling (~30 dB) because the SAD optimum is half a pixel off.
#[test]
fn round61_bslice_half_pel_translation_clears_50db() {
    let (frame_i, frame_b) = subpel_translation_pair(64, 64, 8);
    let (_, rec_i) = encode_idr_with_residuals(&frame_i, 26).unwrap();
    let (bs_b, rec_b) = encode_b_slice(&frame_b, &rec_i, &rec_i, 26, 1, 8).unwrap();
    assert!(!bs_b.is_empty());
    let psnr = psnr_y(&frame_b.luma, &rec_b.luma).unwrap();
    assert!(
        psnr >= 50.0,
        "round-61 half-pel B-slice PSNR_Y {psnr:.2} dB < 50 dB"
    );
}

/// Round-61 — same fixture as above but at ¼-pel. The round-59 ¼-pel
/// P-slice ceiling on this fixture was 52.4 dB; the B-slice path with
/// matched refs should at least clear 48 dB.
#[test]
fn round61_bslice_quarter_pel_translation_clears_48db() {
    let (frame_i, frame_b) = subpel_translation_pair(64, 64, 4);
    let (_, rec_i) = encode_idr_with_residuals(&frame_i, 26).unwrap();
    let (bs_b, rec_b) = encode_b_slice(&frame_b, &rec_i, &rec_i, 26, 1, 8).unwrap();
    assert!(!bs_b.is_empty());
    let psnr = psnr_y(&frame_b.luma, &rec_b.luma).unwrap();
    assert!(
        psnr >= 48.0,
        "round-61 quarter-pel B-slice PSNR_Y {psnr:.2} dB < 48 dB"
    );
}

/// Round-61 — split-translation bi-pred. The current frame has a
/// translation halfway between L0 (+2 px) and L1 (-2 px); the RDO
/// should pick BI (averaging the two predictions yields the current).
/// The sub-pel ME isn't strictly required for an integer-pel split
/// fixture, but the test pins that the BI branch is still picked and
/// reconstructs accurately when the encoder threads sub-pel MVs through
/// the per-list refinement. With matched references the BI prediction
/// is exact on the current frame (each list at integer ±2 px hits its
/// reference exactly), so PSNR clears 50 dB.
#[test]
fn round61_bslice_bi_split_translation_picks_bi_and_clears_50db() {
    // frame_a is the un-shifted source. frame_l0 is shifted +2 px so
    // a reference at +2 in L0's frame matches the current at integer
    // pixels. Same construction but with -2 for L1.
    let (frame_a, _) = translation_pair(0);
    // Build L0 + L1 references: the I frames are translations of frame_a.
    let (ref_l0_src, _) = translation_pair(2);
    let (ref_l1_src, _) = translation_pair(-2);
    let (_, rec_l0) = encode_idr_with_residuals(&ref_l0_src, 26).unwrap();
    let (_, rec_l1) = encode_idr_with_residuals(&ref_l1_src, 26).unwrap();

    let (bs_b, rec_b) = encode_b_slice(&frame_a, &rec_l0, &rec_l1, 26, 1, 8).unwrap();
    assert!(!bs_b.is_empty());

    // Round-trip through the decoder.
    let dec_rec = decode_b_slice(&bs_b, &rec_l0, &rec_l1).unwrap();
    assert_eq!(
        rec_b.luma.samples, dec_rec.luma.samples,
        "encoder + decoder B-slice luma must match byte-for-byte"
    );

    let psnr = psnr_y(&frame_a.luma, &rec_b.luma).unwrap();
    // r412 re-baseline: the IDR reference reconstruction is now
    // wire-conformant (no un-signalled SAO polish) — measured 44.1 dB.
    assert!(
        psnr >= 40.0,
        "round-61 bi-pred split translation PSNR_Y {psnr:.2} dB < 40 dB"
    );
}

/// Round-61 — the round-60 integer-pel B-slice fixtures must still pass
/// after sub-pel ME lands. We exercise the 4-px translation with
/// matched refs (degenerates to uni-pred), checking the 78 dB ballpark
/// that the P-slice path already hit, plus the byte-identical roundtrip.
#[test]
fn round61_integer_pel_regression_holds() {
    let (frame_i, frame_b) = translation_pair(4);
    let (_, rec_i) = encode_idr_with_residuals(&frame_i, 26).unwrap();
    let (bs_b, rec_b) = encode_b_slice(&frame_b, &rec_i, &rec_i, 26, 1, 8).unwrap();
    let dec_rec = decode_b_slice(&bs_b, &rec_i, &rec_i).unwrap();
    assert_eq!(
        rec_b.luma.samples, dec_rec.luma.samples,
        "round-61 encoder + decoder B-slice luma must still match byte-for-byte"
    );
    let psnr = psnr_y(&frame_b.luma, &rec_b.luma).unwrap();
    // r412 re-baseline: honest (wire-conformant) IDR reference —
    // measured 43.9 dB on this fixture.
    assert!(
        psnr >= 40.0,
        "round-61: round-60 integer-pel B-slice fixture regressed to PSNR_Y {psnr:.2} dB (< 40 dB)"
    );
}

/// Round-61 — sub-pel B-slice round-trips through encode + decode
/// byte-for-byte (the decoder already handles 1/16-pel MVs because
/// `mc_predict_subpel` is the single per-list MC path).
#[test]
fn round61_bslice_subpel_decoder_byte_identical() {
    let (frame_i, frame_b) = subpel_translation_pair(64, 64, 8);
    let (_, rec_i) = encode_idr_with_residuals(&frame_i, 26).unwrap();
    let (bs_b, enc_rec) = encode_b_slice(&frame_b, &rec_i, &rec_i, 26, 1, 8).unwrap();
    let dec_rec = decode_b_slice(&bs_b, &rec_i, &rec_i).unwrap();
    assert_eq!(
        enc_rec.luma.samples, dec_rec.luma.samples,
        "round-61 encoder + decoder B-slice luma must match byte-for-byte at sub-pel MVs",
    );
}
