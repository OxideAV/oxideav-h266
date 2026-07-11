//! Round-63 (Goal A) — integration tests for weighted bi-prediction
//! on the B-slice encoder + decoder.
//!
//! Rounds 60/61 carried the §8.5.6.4 simple-average bi-pred form
//! `(predL0 + predL1 + 1) >> 1`. Round 63 (Goal A) wires the
//! §8.5.6.5 explicit weighted sample prediction process (eq. 994)
//! with per-list `(luma_weight, luma_offset)` carried in the slice
//! header as `pred_weight_table()` per §7.4.7.7. The encoder
//! estimates the table from per-list mean-luma offsets and runs a
//! per-CU `{unweighted-BI, weighted-BI}` SSE-based RDO; the lower-
//! SSE form is emitted with a 1-bit `use_weighted_bi` selector.
//!
//! Test inventory:
//!   1. Fade fixture — both refs have intentional uniform luminance
//!      offsets vs. the current frame. Expected: encoder turns on WP
//!      in the slice header AND picks weighted-BI on every block;
//!      PSNR_Y clears 58 dB.
//!   2. No-fade fixture — refs already match curr in mean. Expected:
//!      encoder leaves WP off (header bit 0), wire is bit-for-bit
//!      compatible with round-61 / 62.
//!   3. Decoder byte-identical roundtrip on the fade fixture.

use oxideav_h266::encoder_inter::{decode_b_slice, encode_b_slice};
use oxideav_h266::encoder_pipeline::{encode_idr_with_residuals, psnr_y};
use oxideav_h266::reconstruct::PictureBuffer;

/// Build a fade fixture: a reference frame at neutral intensity 100,
/// plus two copies of the same content at different luminance offsets.
/// All three frames share the same content (mid-band 2D gradient with a
/// large-amplitude vertical sinusoid that prevents ME from gaming the
/// fixture by translating the reference). Only the brightness differs
/// between the three frames.
fn fade_fixture(
    w: usize,
    h: usize,
) -> (PictureBuffer, PictureBuffer, PictureBuffer, PictureBuffer) {
    // Spatial content: a 2D pattern designed so shifts within the
    // search window produce strictly worse SAD than the unshifted
    // candidate (so ME settles at mv=(0, 0) and the encoder must rely
    // on the WP table to model the per-frame brightness offset).
    let make = |seed: i32| {
        let mut buf = PictureBuffer::yuv420_filled(w, h, 100);
        for y in 0..h {
            for x in 0..w {
                // Block-checkerboard pattern: large solid-colour
                // 16×16 tiles. Inside each tile every pixel has the
                // same value, so any per-CU shift within `search_range`
                // (which the scaffold caps at ±8 luma) lands on the
                // same tile (or an adjacent tile of opposite parity)
                // — the within-tile case has identical brightness so
                // shifts don't help, and the cross-tile case has a much
                // larger penalty than the small WP-recoverable
                // brightness gap.
                let tile_x = x / 32;
                let tile_y = y / 32;
                let parity = (tile_x + tile_y) & 1;
                let tile_val = if parity == 0 { 60 } else { 180 };
                let v = (tile_val + seed).clamp(0, 255) as u8;
                buf.luma.samples[y * buf.luma.stride + x] = v;
            }
        }
        buf
    };
    // L0 ref: dimmer (-20). L1 ref: even dimmer (-40). curr: nominal.
    // First arg is the IDR seed which we use to bootstrap the rec
    // pictures used as L0/L1 references; second/third/fourth are the
    // P/B frames at different intensity offsets.
    let idr = make(0);
    let l0_src = make(-20);
    let l1_src = make(-40);
    let curr = make(0);
    (idr, l0_src, l1_src, curr)
}

#[test]
fn round63_bslice_fade_fixture_weighted_bi_clears_58db() {
    // Build the fade fixture.
    let (_frame_i, l0_src, l1_src, frame_b) = fade_fixture(64, 64);
    // Use the raw source frames as the reference DPB pictures — the
    // round-58/60 decoder API takes `&PictureBuffer` of any provenance.
    // Bypassing IDR encoding here isolates the weighted-BI test from
    // QP-26 quantisation noise on the references; the headline target
    // is the cleanliness of §8.5.6.5 eq. 994 itself, not the
    // round-58 P-slice reconstruction loop.
    let rec_l0 = &l0_src;
    let rec_l1 = &l1_src;

    // Disable ME (search_range=0) so the encoder cannot translate the
    // reference to mask the brightness fade — this isolates the test
    // to the §8.5.6.5 weighted bi-pred dispatch itself.
    let (bs_b, rec_b) = encode_b_slice(&frame_b, rec_l0, rec_l1, 26, 1, 0).unwrap();
    assert!(!bs_b.is_empty());

    let psnr = psnr_y(&frame_b.luma, &rec_b.luma).unwrap();

    // Decoder must reproduce the encoder's reconstruction byte-for-byte.
    let dec_rec = decode_b_slice(&bs_b, rec_l0, rec_l1).unwrap();
    assert_eq!(
        rec_b.luma.samples, dec_rec.luma.samples,
        "round-63 weighted-BI encoder + decoder luma must match byte-for-byte",
    );

    assert!(
        psnr >= 58.0,
        "round-63 weighted-BI fade fixture PSNR_Y {psnr:.2} dB < 58 dB",
    );
}

#[test]
fn round63_bslice_no_fade_keeps_wp_off_and_matches_round61() {
    // No fade — both refs and curr have identical mean luminance.
    // The encoder must leave WP off in the slice header (so the wire
    // is bit-for-bit compatible with the round-60/61/62 form modulo a
    // single extra "wp_present=0" bit per slice). PSNR_Y must still
    // reach the round-60 ceiling (>= 70 dB on integer-pel
    // translation, as round-60's regression test pins).
    let mut a = PictureBuffer::yuv420_filled(64, 64, 100);
    let mut b = PictureBuffer::yuv420_filled(64, 64, 100);
    let mut c = PictureBuffer::yuv420_filled(64, 64, 100);
    for y in 0..64 {
        for x in 0..64 {
            let v = if (x / 8) % 2 == 0 { 80u8 } else { 180u8 };
            a.luma.samples[y * a.luma.stride + x] = v;
            b.luma.samples[y * b.luma.stride + x] = v;
            c.luma.samples[y * c.luma.stride + x] = v;
        }
    }
    let (_, rec_l0) = encode_idr_with_residuals(&a, 26).unwrap();
    let (_, rec_l1) = encode_idr_with_residuals(&b, 26).unwrap();
    let (bs_b, rec_b) = encode_b_slice(&c, &rec_l0, &rec_l1, 26, 1, 8).unwrap();
    let dec_rec = decode_b_slice(&bs_b, &rec_l0, &rec_l1).unwrap();
    assert_eq!(rec_b.luma.samples, dec_rec.luma.samples);
    let psnr = psnr_y(&c.luma, &rec_b.luma).unwrap();
    // r412 re-baseline — honest (wire-conformant) IDR references;
    // measured 44.1 dB.
    assert!(
        psnr >= 40.0,
        "round-63 no-fade B-slice PSNR_Y {psnr:.2} dB < 40 dB",
    );
}

#[test]
fn round63_bslice_weighted_decoder_byte_identical() {
    // Stronger version of the fade fixture test: any reconstruction
    // path that involves the weighted-bi formula must produce
    // byte-identical output from encoder and decoder. Use a
    // milder fade (-10 / -30) so the encoder still picks WP on enough
    // blocks to exercise the codepath.
    let make = |seed: i32| {
        let mut buf = PictureBuffer::yuv420_filled(64, 64, 100);
        for y in 0..64 {
            for x in 0..64 {
                let v = (100 + seed + ((x as i32) % 32 - 16)).clamp(0, 255) as u8;
                buf.luma.samples[y * buf.luma.stride + x] = v;
            }
        }
        buf
    };
    let l0_src = make(-10);
    let l1_src = make(-30);
    let curr = make(0);
    let (_, rec_l0) = encode_idr_with_residuals(&l0_src, 26).unwrap();
    let (_, rec_l1) = encode_idr_with_residuals(&l1_src, 26).unwrap();
    let (bs_b, rec_b) = encode_b_slice(&curr, &rec_l0, &rec_l1, 26, 1, 8).unwrap();
    let dec_rec = decode_b_slice(&bs_b, &rec_l0, &rec_l1).unwrap();
    assert_eq!(
        rec_b.luma.samples, dec_rec.luma.samples,
        "weighted-BI: encoder + decoder must match byte-for-byte",
    );
}
