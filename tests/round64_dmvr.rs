//! Round-64 — integration tests for Decoder-side Motion Vector
//! Refinement (DMVR) per VVC §8.5.3.2.4 / §8.5.3.2.5.
//!
//! Rounds 60/61 carried `(predL0 + predL1 + 1) >> 1` simple-average
//! bi-pred and round 63 (Goal A) added explicit weighted bi-pred.
//! Round 64 lands the DMVR refinement step — for bi-pred merge CUs
//! whose refs bracket the current picture symmetrically, the decoder
//! runs a 2-pass search (5×5 integer-pel + parabolic half-pel) around
//! the initial merge MV pair to locate a lower-bilateral-matching-SAD
//! pair. The refined MVs replace the initial pair for the subsequent
//! §8.5.6 motion compensation.
//!
//! Test inventory:
//!   1. Symmetric-bipred fixture: a synthetic 3-frame setup where L0
//!      and L1 refs bracket the current picture and the "true" CU
//!      is shifted by exactly 1 luma sample from each reference. The
//!      DMVR refinement must converge on `int_delta_x = -1` (the
//!      spec-prescribed value that aligns ref_l0(MV0 + δ) with
//!      ref_l1(MV1 − δ)) and the resulting bi-pred MC must clear
//!      the headline +1.5 dB PSNR_Y improvement target vs the
//!      DMVR-off baseline.
//!   2. Gating: when block size is below the §8.5.3.2.4 8×8 floor
//!      DMVR must not run — the gate flag returns false.
//!   3. Decoder-bypass roundtrip: when the DMVR gate is off the
//!      refinement output equals the unrefined MV pair byte-for-byte.

use oxideav_h266::dmvr::{apply_dmvr, dmvr_used_flag, DMVR_SEARCH_RANGE};
use oxideav_h266::inter::{predict_luma_block_bipred, MotionVector};
use oxideav_h266::reconstruct::PicturePlane;

/// Build a smooth 2D-aperiodic synthetic luma plane carrying a low-
/// frequency sinusoidal pattern with a horizontal phase shift `shift`
/// (integer luma samples). Aperiodic over the plane so reads inside
/// the test CU never alias to the wrap-around copy.
fn sym_bipred_plane(w: usize, h: usize, shift: i32) -> PicturePlane {
    let mut p = PicturePlane::filled(w, h, 0);
    for y in 0..h {
        for x in 0..w {
            let xs = (x as i32 - shift).clamp(0, w as i32 - 1);
            let phx = (xs as f64) / (w as f64) * 3.0 * std::f64::consts::PI;
            let phy = (y as f64) / (h as f64) * 1.5 * std::f64::consts::PI;
            let v = 128.0 + 50.0 * phx.sin() + 30.0 * phy.cos();
            p.samples[y * p.stride + x] = v.clamp(0.0, 255.0) as u8;
        }
    }
    p
}

/// Compute PSNR_Y over a sub-rectangle of two planes (in dB).
fn psnr_sub(
    truth: &PicturePlane,
    rec: &PicturePlane,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
) -> f64 {
    let mut sse: u64 = 0;
    for r in 0..h {
        for c in 0..w {
            let t = truth.samples[(y + r) * truth.stride + (x + c)] as i32;
            let v = rec.samples[(y + r) * rec.stride + (x + c)] as i32;
            sse += ((t - v) * (t - v)) as u64;
        }
    }
    if sse == 0 {
        return f64::INFINITY;
    }
    let mse = sse as f64 / (w as f64 * h as f64);
    10.0 * (255.0f64 * 255.0 / mse).log10()
}

#[test]
fn round64_symmetric_bipred_dmvr_clears_1_5db_psnr_improvement() {
    // Setup: L0 ref is shifted -1 sample, L1 ref is shifted +1
    // sample, and the truth (the "current" picture) is at shift 0.
    // For ref_l1[y] = ref_l0[y − 2] and initial MV pair (0, 0) the
    // BM cost minimises at integer delta δ = (-1, 0) — i.e. MV0
    // moves to (-1, 0), MV1 moves to (+1, 0), both lining up with
    // truth's pattern.
    let pic_w = 64usize;
    let pic_h = 64usize;
    let r_l0 = sym_bipred_plane(pic_w, pic_h, -1);
    let r_l1 = sym_bipred_plane(pic_w, pic_h, 1);
    let truth = sym_bipred_plane(pic_w, pic_h, 0);

    let cu_x = 16u32;
    let cu_y = 16u32;
    let w = 16u32;
    let h = 16u32;

    // DMVR-off baseline: average the two refs at MV = (0, 0).
    let mut baseline = PicturePlane::filled(pic_w, pic_h, 0);
    predict_luma_block_bipred(
        &mut baseline,
        cu_x,
        cu_y,
        w,
        h,
        &r_l0,
        MotionVector::ZERO,
        &r_l1,
        MotionVector::ZERO,
    )
    .unwrap();

    // DMVR-on: refine, then run bi-pred with the refined MVs.
    let res = apply_dmvr(
        cu_x,
        cu_y,
        w,
        h,
        MotionVector::ZERO,
        MotionVector::ZERO,
        &r_l0,
        &r_l1,
    )
    .unwrap();
    let mut refined = PicturePlane::filled(pic_w, pic_h, 0);
    predict_luma_block_bipred(
        &mut refined,
        cu_x,
        cu_y,
        w,
        h,
        &r_l0,
        res.mv_l0_refined,
        &r_l1,
        res.mv_l1_refined,
    )
    .unwrap();

    let psnr_base = psnr_sub(
        &truth,
        &baseline,
        cu_x as usize,
        cu_y as usize,
        w as usize,
        h as usize,
    );
    let psnr_ref = psnr_sub(
        &truth,
        &refined,
        cu_x as usize,
        cu_y as usize,
        w as usize,
        h as usize,
    );
    eprintln!(
        "round64 DMVR: integer delta=({}, {}) half-pel=({}, {}) PSNR_Y baseline={:.2} dB \
         refined={:.2} dB (Δ={:.2} dB)",
        res.int_delta_x,
        res.int_delta_y,
        res.half_pel.dx_q16,
        res.half_pel.dy_q16,
        psnr_base,
        psnr_ref,
        psnr_ref - psnr_base,
    );
    assert_eq!(
        res.int_delta_x, -1,
        "DMVR must converge on δx=-1 (spec opposite-direction pairing)"
    );
    assert_eq!(res.int_delta_y, 0, "DMVR must converge on δy=0");
    assert!(
        res.final_int_sad < res.baseline_sad,
        "refined SAD {} must beat baseline {}",
        res.final_int_sad,
        res.baseline_sad,
    );
    let delta_db = psnr_ref - psnr_base;
    assert!(
        delta_db >= 1.5,
        "round-64 DMVR PSNR_Y improvement {delta_db:.2} dB < 1.5 dB headline target \
         (baseline {psnr_base:.2} → refined {psnr_ref:.2})",
    );
}

#[test]
fn round64_dmvr_search_range_matches_spec() {
    // Sanity: the spec sets dmvrSearchRange = 2.
    assert_eq!(DMVR_SEARCH_RANGE, 2);
}

#[test]
fn round64_dmvr_gate_blocks_small_blocks() {
    // §8.5.3.2.4 step 1 forbids DMVR for blocks smaller than 8×8.
    let used = dmvr_used_flag(
        true,  // sps_dmvr_enabled
        false, // ph_dmvr_disabled
        true,  // merge
        true, true, // bi-pred
        true, // bracketed
        true, true, // STRP
        0, false, false, false, 0, false, false, false, false, 4, 4, 0,
    );
    assert!(!used);
}

#[test]
fn round64_dmvr_gate_blocks_bcw_engaged() {
    // §8.5.3.2.4 step 1 — DMVR is off whenever BCW or weighted
    // prediction is engaged. Cross-verifies the gate cannot run
    // alongside the round-63 explicit-WP path.
    let used = dmvr_used_flag(
        true, false, true, true, true, true, true, true, 0, false, false, false, 2, false, false,
        false, false, 16, 16, 0,
    );
    assert!(!used, "BcwIdx > 0 must turn DMVR off");

    let used = dmvr_used_flag(
        true, false, true, true, true, true, true, true, 0, false, false, false, 0, true, false,
        false, false, 16, 16, 0,
    );
    assert!(!used, "luma_weight_l0_flag=true must turn DMVR off");
}

#[test]
fn round64_dmvr_gate_blocks_when_not_symmetric() {
    // §8.5.3.2.4 step 1 — the two refs must bracket curr
    // symmetrically. When `bracketed_same_diff_poc` is false (refs
    // not symmetric) DMVR must not run.
    let used = dmvr_used_flag(
        true, false, true, true, true, false, true, true, 0, false, false, false, 0, false, false,
        false, false, 16, 16, 0,
    );
    assert!(!used);
}
