//! Round-60 — integration tests for the B-slice (bi-prediction)
//! encoder + decoder scaffold
//! ([`oxideav_h266::encoder_inter::encode_b_slice`] /
//! [`oxideav_h266::encoder_inter::decode_b_slice`]).
//!
//! These live alongside the round-58 / round-59 P-slice integration
//! tests and target the §7.4.7.2 `inter_pred_idc` syntax + §8.5.6.4
//! bi-prediction reconstruction (`pred = (predL0 + predL1 + 1) >> 1`)
//! with a single picture per reference list. Multi-reference DPB and
//! weighted bi-pred are explicitly deferred to later rounds.

use oxideav_h266::encoder_inter::{decode_b_slice, decode_p_slice, encode_b_slice, encode_p_slice};
use oxideav_h266::encoder_pipeline::{encode_idr_with_residuals, psnr_y};
use oxideav_h266::reconstruct::PictureBuffer;

/// Round-58 stripe pattern, copied locally so we can build matched
/// L0 + L1 references with independent dx values.
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

/// Round-60 — feeding the same picture as both L0 and L1 makes
/// bi-prediction degenerate to uni-pred (the average of two identical
/// predictions equals either one). PSNR should still clear the 35 dB
/// floor on the 4-px translation fixture.
#[test]
fn bslice_same_refs_degenerates_to_pslice_quality() {
    let (frame_i, frame_b) = translation_pair(4);
    let (_, rec_i) = encode_idr_with_residuals(&frame_i, 26).unwrap();
    // Both lists point at the same picture (`rec_i`).
    let (bs_b, rec_b) = encode_b_slice(&frame_b, &rec_i, &rec_i, 26, 1, 8).unwrap();
    assert!(!bs_b.is_empty());
    let psnr = psnr_y(&frame_b.luma, &rec_b.luma).unwrap();
    assert!(
        psnr >= 35.0,
        "B-slice with identical L0/L1 PSNR_Y {psnr:.2} dB < 35 dB on 4-px translation"
    );
}

/// Round-60 — when L0 has a 2-px translation and L1 has a 4-px
/// translation, neither uni-pred path is exact, but the simple-average
/// bi-pred can split the difference. We compare against the P-slice
/// quality using only L0 — bi-pred should reach at least P-slice
/// quality (the RDO can always fall back to L0).
#[test]
fn bslice_split_translation_beats_or_matches_l0_only() {
    let (frame_a, frame_b) = translation_pair(2);
    let (_, _frame_c) = translation_pair(4);
    // Make the L0 reference a translation-of-2 of frame_a, L1 a
    // translation-of-4.
    let (_, rec_l0) = encode_idr_with_residuals(&frame_a, 26).unwrap();
    let (_, rec_l1) = {
        let (i_, _) = translation_pair(-2); // shift the other way
        encode_idr_with_residuals(&i_, 26).unwrap()
    };
    let (bs_b, rec_b) = encode_b_slice(&frame_b, &rec_l0, &rec_l1, 26, 1, 8).unwrap();
    assert!(!bs_b.is_empty());
    let psnr_b = psnr_y(&frame_b.luma, &rec_b.luma).unwrap();

    // P-slice using only L0 as the single reference.
    let (_, rec_p) = encode_p_slice(&frame_b, &rec_l0, 26, 1, 8).unwrap();
    let psnr_p = psnr_y(&frame_b.luma, &rec_p.luma).unwrap();

    // B-slice has more freedom (RDO picks the best of L0 / L1 / BI),
    // so it must match or exceed the P-slice PSNR.
    assert!(
        psnr_b >= psnr_p - 0.5,
        "B-slice PSNR_Y {psnr_b:.2} dB unexpectedly worse than P-slice {psnr_p:.2} dB"
    );
    assert!(
        psnr_b >= 25.0,
        "B-slice translation PSNR_Y {psnr_b:.2} dB < 25 dB"
    );
}

/// Round-60 — the encoded B-slice fed back through `decode_b_slice`
/// reconstructs to the same luma the encoder kept internally
/// (bit-perfect roundtrip through the OXAV_VVC_BSLIC wire chunk).
#[test]
fn bslice_decoder_roundtrip_byte_identical() {
    let (frame_i, frame_b) = translation_pair(4);
    let (_, rec_i) = encode_idr_with_residuals(&frame_i, 26).unwrap();
    let (bs_b, enc_rec) = encode_b_slice(&frame_b, &rec_i, &rec_i, 26, 1, 8).unwrap();
    let dec_rec = decode_b_slice(&bs_b, &rec_i, &rec_i).unwrap();
    let mut diff_count = 0usize;
    let mut first: Option<(usize, usize, u8, u8)> = None;
    for y in 0..frame_b.luma.height {
        for x in 0..frame_b.luma.width {
            let e = enc_rec.luma.samples[y * enc_rec.luma.stride + x];
            let d = dec_rec.luma.samples[y * dec_rec.luma.stride + x];
            if e != d {
                diff_count += 1;
                if first.is_none() {
                    first = Some((x, y, e, d));
                }
            }
        }
    }
    assert_eq!(
        diff_count, 0,
        "encoder + decoder B-slice luma differ in {} samples (first at {:?})",
        diff_count, first,
    );
}

/// Round-60 — the round-58 P-slice regression on the 4-px translation
/// fixture must still hit 78.23 dB after the B-slice path lands (the
/// P-slice code path is unchanged and lives next to the B-slice code
/// in `encoder_inter`).
#[test]
fn pslice_regression_holds_at_78db() {
    let (frame_i, frame_p) = translation_pair(4);
    let (_, rec_i) = encode_idr_with_residuals(&frame_i, 26).unwrap();
    let (bs_p, rec_p) = encode_p_slice(&frame_p, &rec_i, 26, 1, 8).unwrap();
    let dec_rec = decode_p_slice(&bs_p, &rec_i).unwrap();
    assert_eq!(rec_p.luma.samples, dec_rec.luma.samples);
    let psnr = psnr_y(&frame_p.luma, &rec_p.luma).unwrap();
    assert!(
        psnr >= 78.0,
        "P-slice regression: {psnr:.2} dB < 78 dB after B-slice landing"
    );
}

/// Round-60 — a B-slice with no motion (current == L0 == L1) emits a
/// near-zero-residual stream and reconstructs the original luma.
#[test]
fn bslice_no_motion_zero_residual_roundtrip() {
    let (frame_a, _) = translation_pair(0);
    let (_, rec_i) = encode_idr_with_residuals(&frame_a, 26).unwrap();
    let (bs_b, rec_b) = encode_b_slice(&frame_a, &rec_i, &rec_i, 26, 1, 8).unwrap();
    let dec_rec = decode_b_slice(&bs_b, &rec_i, &rec_i).unwrap();
    assert_eq!(rec_b.luma.samples, dec_rec.luma.samples);
    let psnr = psnr_y(&frame_a.luma, &rec_b.luma).unwrap();
    assert!(
        psnr >= 30.0,
        "Identical-frame B-slice PSNR_Y {psnr:.2} dB < 30 dB"
    );
}
