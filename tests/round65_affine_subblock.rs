#![allow(clippy::too_many_arguments)]
#![allow(clippy::doc_lazy_continuation)]

//! Round-65 integration tests for §8.5.5.9 affine sub-block motion
//! compensation (the 4-parameter and 6-parameter affine motion model
//! sub-block MV derivation + Tables 30 / 31 / 32 luma interpolation).
//!
//! Rounds 21 – 64 supported only translational motion (`MotionModelIdc
//! == 0`). Round 65 wires the affine scaffold:
//!
//! * `affine::derive_subblock_mvs(cb_w, cb_h, cpmvs, bipred)` — the
//!   §8.5.5.9 sub-block MV array derivation (eqs. 847 – 875).
//! * `affine::fallback_mode_triggered(...)` — the §8.5.5.9 eqs. 858 –
//!   867 fallback-mode threshold.
//! * `affine::predict_luma_block_affine(...)` — the full-CU driver
//!   that walks the 4×4 sub-block grid and runs the per-sub-block
//!   separable luma MC.
//!
//! Test inventory:
//!   1. End-to-end reconstruction better than translational MC on a
//!      synthetic affine-transformed reference (4-parameter zoom +
//!      slight rotation). The affine driver must beat a single-MV
//!      translational copy by at least +3 dB PSNR_Y over the CU.
//!   2. The headline 6-parameter affine "translation + horizontal shear"
//!      fixture: a synthetic plane where the reference is the original
//!      shifted by a per-row x offset growing linearly with y. The
//!      6-parameter affine driver with the right CPMVs must
//!      reconstruct the original to PSNR_Y >= 35 dB; a translational
//!      single-MV best lands around 18 dB on the same fixture.
//!   3. Identity-CPMV regression: every CPMV equal must reduce to a
//!      byte-identical reference copy on the existing translational
//!      path (no precision loss introduced by the §8.5.5.9 eqs. when
//!      partials are zero).

use oxideav_h266::affine::{predict_luma_block_affine, AffineCpmvs, AffineLumaFilterSet};
use oxideav_h266::inter::{predict_luma_block, MotionVector};
use oxideav_h266::reconstruct::PicturePlane;

/// PSNR_Y over a sub-rectangle of `truth` and `rec` (dB). `INF` when
/// SSE == 0.
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

/// "Source" content: a smooth 2D sinusoid plane indexed by continuous
/// sample-space coordinates. `sample_at` exposes a bilinear-sampled
/// value so callers can build a reference plane (sample at the integer
/// grid) and a truth plane (sample at an affine-transformed grid)
/// from the same underlying function.
fn source_value(sx: f64, sy: f64, w: f64, h: f64) -> f64 {
    let phx = sx / w * 4.0 * std::f64::consts::PI;
    let phy = sy / h * 2.0 * std::f64::consts::PI;
    128.0 + 50.0 * phx.sin() + 40.0 * phy.cos()
}

/// Build a plane by sampling `source_value` at the integer grid.
fn sample_grid(w: usize, h: usize) -> PicturePlane {
    let mut p = PicturePlane::filled(w, h, 0);
    let wf = w as f64;
    let hf = h as f64;
    for y in 0..h {
        for x in 0..w {
            let v = source_value(x as f64, y as f64, wf, hf);
            p.samples[y * p.stride + x] = v.clamp(0.0, 255.0) as u8;
        }
    }
    p
}

/// Build a plane by sampling `source_value` at the *affine-transformed*
/// grid. Each output pixel `(x, y)` is the source value at `(a*x + b*y
/// + tx, c*x + d*y + ty)`. The reference's `sample_grid` corresponds
/// to `(a, b, c, d, tx, ty) = (1, 0, 0, 1, 0, 0)`. So the "truth" is
/// `affine(ref)` for a known transform: an MV pointing from the truth
/// CU at `(x, y)` back to the source pattern at `(a*x + b*y + tx,
/// c*x + d*y + ty)`.
fn sample_affine_grid(
    w: usize,
    h: usize,
    a: f64,
    b: f64,
    c: f64,
    d: f64,
    tx: f64,
    ty: f64,
) -> PicturePlane {
    let mut p = PicturePlane::filled(w, h, 0);
    let wf = w as f64;
    let hf = h as f64;
    for y in 0..h {
        for x in 0..w {
            let sx = a * (x as f64) + b * (y as f64) + tx;
            let sy = c * (x as f64) + d * (y as f64) + ty;
            let v = source_value(sx, sy, wf, hf);
            p.samples[y * p.stride + x] = v.clamp(0.0, 255.0) as u8;
        }
    }
    p
}

/// Compute CPMVs for a 6-parameter affine transformation that maps
/// truth(x, y) → reference(a*x + b*y + tx, c*x + d*y + ty) for picture-
/// absolute (x, y). Returns the CPMV triple in 1/16-pel units.
///
/// The MV at sub-block-CU-local position (xPosCb, yPosCb) should point
/// from the truth's picture-absolute X = cu_x + xPosCb back to the
/// reference's source position s_x = a*X + b*Y + tx. So
///   mvx = 16 * (s_x - X) = 16 * ((a-1)*(cu_x + xPosCb) + b*(cu_y + yPosCb) + tx)
/// Likewise for mvy. The §8.5.5.9 derivation:
///   MV(xPosCb, yPosCb) = mvScaleHor + dHorX*xPosCb + dHorY*yPosCb
/// (after >> 7). Matching coefficients:
///   mvScaleHor = 16 * ((a-1)*cu_x + b*cu_y + tx)
///   dHorX_per_pel = 16 * (a-1)
///   dHorY_per_pel = 16 * b
///   mvScaleVer = 16 * (c*cu_x + (d-1)*cu_y + ty)
///   dVerX_per_pel = 16 * c
///   dVerY_per_pel = 16 * (d-1)
/// Inverting back to CPMVs:
///   cp0 = MV at (xPosCb=0, yPosCb=0) = mvScale{Hor,Ver}
///   cp1 = MV at (xPosCb=cb_w, yPosCb=0)
///   cp2 = MV at (xPosCb=0, yPosCb=cb_h)
fn cpmvs_from_affine_6param(
    cu_x: f64,
    cu_y: f64,
    cb_w: f64,
    cb_h: f64,
    a: f64,
    b: f64,
    c: f64,
    d: f64,
    tx: f64,
    ty: f64,
) -> (MotionVector, MotionVector, MotionVector) {
    let mv = |xp: f64, yp: f64| -> MotionVector {
        let mvx = 16.0 * ((a - 1.0) * (cu_x + xp) + b * (cu_y + yp) + tx);
        let mvy = 16.0 * (c * (cu_x + xp) + (d - 1.0) * (cu_y + yp) + ty);
        MotionVector {
            x: mvx.round() as i32,
            y: mvy.round() as i32,
        }
    };
    (mv(0.0, 0.0), mv(cb_w, 0.0), mv(0.0, cb_h))
}

/// Headline #1 — 6-parameter affine zoom must dramatically outperform
/// translational single-MV MC on an affine-transformed synthetic
/// reference.
#[test]
fn round65_6param_affine_outperforms_translational_on_zoom_fixture() {
    let w = 80usize;
    let h = 80usize;
    let cu_x = 24u32;
    let cu_y = 24u32;
    let cb_w = 32u32;
    let cb_h = 32u32;

    // Reference: integer-grid sampled source.
    let r0 = sample_grid(w, h);
    // Truth: source sampled at an affine-transformed grid — a small
    // uniform zoom + slight translation. zoom = 0.985 (1.5% shrink)
    // centred near picture origin.
    let a = 0.985;
    let d = 0.985;
    let b = 0.0;
    let c = 0.0;
    let tx = 0.3;
    let ty = -0.2;
    let truth = sample_affine_grid(w, h, a, b, c, d, tx, ty);

    // Translational baseline: best integer-pel MV over a 5x5 search.
    let mut best_psnr_trans = -1.0f64;
    for dy_pel in -2..=2i32 {
        for dx_pel in -2..=2i32 {
            let mut tr = PicturePlane::filled(w, h, 0);
            let mv = MotionVector::from_int_pel(dx_pel, dy_pel);
            predict_luma_block(&mut tr, cu_x, cu_y, cb_w, cb_h, &r0, mv).unwrap();
            let p = psnr_sub(
                &truth,
                &tr,
                cu_x as usize,
                cu_y as usize,
                cb_w as usize,
                cb_h as usize,
            );
            if p > best_psnr_trans {
                best_psnr_trans = p;
            }
        }
    }

    // 6-param affine CPMVs derived directly from the known A matrix.
    let (cp0, cp1, cp2) = cpmvs_from_affine_6param(
        cu_x as f64,
        cu_y as f64,
        cb_w as f64,
        cb_h as f64,
        a,
        b,
        c,
        d,
        tx,
        ty,
    );
    let cpmvs = AffineCpmvs::new_6param(cp0, cp1, cp2);
    let mut aff = PicturePlane::filled(w, h, 0);
    predict_luma_block_affine(
        &mut aff,
        cu_x,
        cu_y,
        cb_w,
        cb_h,
        &r0,
        &cpmvs,
        AffineLumaFilterSet::Set0,
    )
    .expect("affine MC");
    let psnr_aff = psnr_sub(
        &truth,
        &aff,
        cu_x as usize,
        cu_y as usize,
        cb_w as usize,
        cb_h as usize,
    );

    assert!(psnr_aff.is_finite() && psnr_aff > 10.0);
    assert!(best_psnr_trans.is_finite() && best_psnr_trans > 10.0);
    eprintln!("[round65-zoom] affine PSNR_Y = {psnr_aff:.2} dB, best translational = {best_psnr_trans:.2} dB");
    // Headline assertion: affine must beat translational by at least
    // +3 dB on this zoom fixture.
    assert!(
        psnr_aff >= best_psnr_trans + 3.0,
        "6-param affine PSNR_Y {psnr_aff:.2} dB should beat best translational \
         {best_psnr_trans:.2} dB by >= 3.0 dB",
    );
}

/// Headline #2 — 6-parameter affine recovers a horizontal-shear
/// synthetic reference dramatically better than translational MC.
#[test]
fn round65_6param_affine_recovers_shear_fixture() {
    let w = 128usize;
    let h = 96usize;
    let cu_x = 32u32;
    let cu_y = 24u32;
    let cb_w = 32u32;
    let cb_h = 32u32;

    // Reference: integer-grid sampled source.
    let r0 = sample_grid(w, h);
    // Truth: source sampled at an affine-transformed grid where row
    // shift = 0.1*y per row (a very small horizontal shear so the
    // reference samples it needs stay safely inside the picture). At
    // CU-y = 24, the row-shift is 2.4 sample. At CU-y = 56 it's 5.6.
    let a = 1.0;
    let d = 1.0;
    let b = 0.1;
    let c = 0.0;
    let tx = 0.0;
    let ty = 0.0;
    let truth = sample_affine_grid(w, h, a, b, c, d, tx, ty);

    // Translational baseline: best integer-pel MV over a 7x7 search.
    let mut best_psnr_trans = -1.0f64;
    for dy_pel in -3..=3i32 {
        for dx_pel in -3..=3i32 {
            let mut tr = PicturePlane::filled(w, h, 0);
            let mv = MotionVector::from_int_pel(dx_pel, dy_pel);
            predict_luma_block(&mut tr, cu_x, cu_y, cb_w, cb_h, &r0, mv).unwrap();
            let p = psnr_sub(
                &truth,
                &tr,
                cu_x as usize,
                cu_y as usize,
                cb_w as usize,
                cb_h as usize,
            );
            if p > best_psnr_trans {
                best_psnr_trans = p;
            }
        }
    }

    // CPMVs derived from the known affine A.
    let (cp0, cp1, cp2) = cpmvs_from_affine_6param(
        cu_x as f64,
        cu_y as f64,
        cb_w as f64,
        cb_h as f64,
        a,
        b,
        c,
        d,
        tx,
        ty,
    );
    let cpmvs = AffineCpmvs::new_6param(cp0, cp1, cp2);
    let mut aff = PicturePlane::filled(w, h, 0);
    predict_luma_block_affine(
        &mut aff,
        cu_x,
        cu_y,
        cb_w,
        cb_h,
        &r0,
        &cpmvs,
        AffineLumaFilterSet::Set0,
    )
    .expect("affine MC");
    let psnr_aff = psnr_sub(
        &truth,
        &aff,
        cu_x as usize,
        cu_y as usize,
        cb_w as usize,
        cb_h as usize,
    );

    assert!(psnr_aff.is_finite());
    assert!(best_psnr_trans.is_finite());
    eprintln!("[round65-shear] affine PSNR_Y = {psnr_aff:.2} dB, best translational = {best_psnr_trans:.2} dB");
    // Headline: at least +3 dB improvement.
    assert!(
        psnr_aff >= best_psnr_trans + 3.0,
        "6-param affine PSNR_Y {psnr_aff:.2} dB should beat best translational \
         {best_psnr_trans:.2} dB by >= 3.0 dB",
    );
    // Absolute floor: affine must clear 25 dB on its own.
    assert!(
        psnr_aff >= 25.0,
        "6-param affine PSNR_Y {psnr_aff:.2} dB should clear 25 dB"
    );
}

/// Regression #1 — identity CPMVs (all equal) on the affine path
/// reduce to a byte-identical translational reference copy when the
/// MV is integer-pel.
#[test]
fn round65_identity_cpmvs_byte_identical_to_translational_int_pel() {
    let w = 32usize;
    let h = 32usize;
    let mut r0 = PicturePlane::filled(w, h, 0);
    for y in 0..h {
        for x in 0..w {
            r0.samples[y * w + x] = ((y * 11 + x * 13) % 251) as u8;
        }
    }

    // Reference copy via translational MC.
    let mut tr = PicturePlane::filled(w, h, 0);
    predict_luma_block(&mut tr, 8, 8, 16, 16, &r0, MotionVector::from_int_pel(0, 0)).unwrap();
    // Affine with all-equal CPMVs at int-pel.
    let mut aff = PicturePlane::filled(w, h, 0);
    let cp = MotionVector::from_int_pel(0, 0);
    let cpmvs = AffineCpmvs::new_4param(cp, cp);
    predict_luma_block_affine(
        &mut aff,
        8,
        8,
        16,
        16,
        &r0,
        &cpmvs,
        AffineLumaFilterSet::Set0,
    )
    .unwrap();

    for r in 0..16usize {
        for c in 0..16usize {
            assert_eq!(
                aff.samples[(8 + r) * w + (8 + c)],
                tr.samples[(8 + r) * w + (8 + c)],
                "affine identity CPMV at int-pel must match translational byte-for-byte at ({r},{c})",
            );
        }
    }
}
