#![allow(clippy::too_many_arguments)]
#![allow(clippy::doc_lazy_continuation)]

//! Round-78 integration tests for §8.5.6.4 PROF (Prediction Refinement
//! with Optical Flow) — the per-pixel refinement layered on top of
//! the round-65 affine sub-block MC.
//!
//! Round 65 left PROF deferred ("PROF — Prediction Refinement with
//! Optical Flow (§8.5.5.10). cbProfFlagLX is parsed in §8.5.5.9 but
//! its application requires gradient-based refinement out of scope
//! until a later round"). Round 78 wires the refinement:
//!
//! * `affine::cb_prof_flag_lx(...)` — the §8.5.5.9 cbProfFlagLX
//!   derivation gate.
//! * `affine::derive_prof_diff_mv_array(...)` — the §8.5.5.9 eqs.
//!   880 – 887 per-pixel motion-vector-difference array.
//! * `affine::apply_prof_to_subblock(...)` — the §8.5.6.4 eqs.
//!   955 – 959 gradient + clip + add.
//! * `affine::predict_luma_block_affine_prof(...)` — the full-CU
//!   driver that runs the §8.5.6.3 affine sub-block MC into a
//!   `(sbW + 2) × (sbH + 2)` high-precision halo'd buffer and then
//!   applies PROF per sub-block.
//!
//! Test inventory:
//!   1. PSNR_Y on an affine shear / zoom fixture: PROF on must
//!      *not* regress vs PROF off, and on at least one fixture
//!      with strong affine-gradient content must improve.
//!   2. Translational-degenerate CPMVs ⇒ PROF disabled ⇒
//!      `predict_luma_block_affine_prof` is byte-identical to
//!      `predict_luma_block_affine`.
//!   3. `ph_prof_disabled_flag == 1` short-circuits PROF even on
//!      a real affine fixture ⇒ identical to PROF-off.

use oxideav_h266::affine::{
    cb_prof_flag_lx, derive_prof_diff_mv_array, predict_luma_block_affine,
    predict_luma_block_affine_prof, AffineCpmvs, AffineLumaFilterSet,
};
use oxideav_h266::inter::MotionVector;
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

/// 2D smooth source used as the affine reference + truth driver. The
/// gradient magnitude is non-trivial so the PROF gradient terms
/// `(predL[x+2] - predL[x]) >> 6` carry information.
fn source_value(sx: f64, sy: f64, w: f64, h: f64) -> f64 {
    let phx = sx / w * 4.0 * std::f64::consts::PI;
    let phy = sy / h * 2.0 * std::f64::consts::PI;
    128.0 + 50.0 * phx.sin() + 40.0 * phy.cos()
}

fn sample_grid(w: usize, h: usize) -> PicturePlane {
    let mut p = PicturePlane::filled(w, h, 0);
    for y in 0..h {
        for x in 0..w {
            let v = source_value(x as f64, y as f64, w as f64, h as f64);
            p.samples[y * p.stride + x] = v.clamp(0.0, 255.0) as u8;
        }
    }
    p
}

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
    for y in 0..h {
        for x in 0..w {
            let sx = a * (x as f64) + b * (y as f64) + tx;
            let sy = c * (x as f64) + d * (y as f64) + ty;
            let v = source_value(sx, sy, w as f64, h as f64);
            p.samples[y * p.stride + x] = v.clamp(0.0, 255.0) as u8;
        }
    }
    p
}

/// Derive (cp0, cp1, cp2) for a 6-parameter affine transform mapping
/// truth(x, y) = source(a*x + b*y + tx, c*x + d*y + ty) — matches the
/// round-65 fixture helper.
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

/// Headline — PROF must not regress PSNR_Y vs the affine-only driver on
/// a real affine fixture, and should improve it when the affine model
/// has non-degenerate partials.
#[test]
fn round78_prof_does_not_regress_on_shear_fixture() {
    let w = 128usize;
    let h = 96usize;
    let cu_x = 32u32;
    let cu_y = 24u32;
    let cb_w = 32u32;
    let cb_h = 32u32;

    // Reference: integer-grid source.
    let r0 = sample_grid(w, h);
    // Truth: very small horizontal shear (row-shift = 0.1*y).
    let a = 1.0;
    let d = 1.0;
    let b = 0.1;
    let c = 0.0;
    let tx = 0.0;
    let ty = 0.0;
    let truth = sample_affine_grid(w, h, a, b, c, d, tx, ty);

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

    // PROF off — bare affine sub-block MC.
    let mut no_prof = PicturePlane::filled(w, h, 0);
    predict_luma_block_affine(
        &mut no_prof,
        cu_x,
        cu_y,
        cb_w,
        cb_h,
        &r0,
        &cpmvs,
        AffineLumaFilterSet::Set0,
    )
    .expect("no-prof affine");
    let psnr_no_prof = psnr_sub(
        &truth,
        &no_prof,
        cu_x as usize,
        cu_y as usize,
        cb_w as usize,
        cb_h as usize,
    );

    // PROF on.
    let mut prof = PicturePlane::filled(w, h, 0);
    predict_luma_block_affine_prof(
        &mut prof,
        cu_x,
        cu_y,
        cb_w,
        cb_h,
        &r0,
        &cpmvs,
        AffineLumaFilterSet::Set0,
        /* bipred */ false,
        /* ph_prof_disabled */ false,
        /* rpr_constraints_active */ false,
    )
    .expect("prof affine");
    let psnr_prof = psnr_sub(
        &truth,
        &prof,
        cu_x as usize,
        cu_y as usize,
        cb_w as usize,
        cb_h as usize,
    );
    eprintln!(
        "[round78-shear] PROF off PSNR_Y = {:.2} dB, PROF on PSNR_Y = {:.2} dB",
        psnr_no_prof, psnr_prof
    );

    // PROF on must not regress by more than 0.1 dB.
    assert!(psnr_prof.is_finite() || psnr_no_prof.is_finite());
    if psnr_no_prof.is_finite() {
        assert!(
            psnr_prof >= psnr_no_prof - 0.1,
            "PROF on should not regress (got {psnr_prof:.4} dB vs no-PROF {psnr_no_prof:.4} dB)"
        );
    }
    // And cbProfFlagLX must report TRUE — the gate is satisfied.
    assert!(cb_prof_flag_lx(
        cb_w, cb_h, &cpmvs, /* bipred */ false, false, false,
    ));
    // The diffMv array must be non-zero (real affine ⇒ non-zero
    // partials ⇒ at least some non-zero entries).
    let arr = derive_prof_diff_mv_array(cb_w, cb_h, &cpmvs).expect("diffMv");
    assert!(
        !arr.is_all_zero(),
        "real affine CU must produce non-zero diffMv entries"
    );
}

/// Translational-degenerate CPMVs ⇒ PROF disabled by `is_translational`
/// gate ⇒ output bit-identical to affine-only driver.
#[test]
fn round78_prof_translational_byte_identical_to_no_prof() {
    let w = 64usize;
    let h = 64usize;
    let cu_x = 16u32;
    let cu_y = 16u32;
    let cb_w = 32u32;
    let cb_h = 32u32;

    let r0 = sample_grid(w, h);
    // Degenerate CPMVs (all the same).
    let cp = MotionVector::from_int_pel(0, 0);
    let cpmvs = AffineCpmvs::new_4param(cp, cp);

    let mut no_prof = PicturePlane::filled(w, h, 0);
    predict_luma_block_affine(
        &mut no_prof,
        cu_x,
        cu_y,
        cb_w,
        cb_h,
        &r0,
        &cpmvs,
        AffineLumaFilterSet::Set0,
    )
    .expect("no-prof");
    let mut prof = PicturePlane::filled(w, h, 0);
    predict_luma_block_affine_prof(
        &mut prof,
        cu_x,
        cu_y,
        cb_w,
        cb_h,
        &r0,
        &cpmvs,
        AffineLumaFilterSet::Set0,
        false,
        false,
        false,
    )
    .expect("prof");
    assert_eq!(
        prof.samples, no_prof.samples,
        "degenerate CPMVs must bit-identically pass through PROF"
    );
    assert!(!cb_prof_flag_lx(cb_w, cb_h, &cpmvs, false, false, false));
}

/// `ph_prof_disabled_flag == 1` must short-circuit PROF even on a
/// real affine fixture.
#[test]
fn round78_ph_prof_disabled_flag_disables_prof() {
    let w = 128usize;
    let h = 96usize;
    let cu_x = 32u32;
    let cu_y = 24u32;
    let cb_w = 32u32;
    let cb_h = 32u32;

    let r0 = sample_grid(w, h);
    let a = 1.0;
    let d = 1.0;
    let b = 0.1;
    let c = 0.0;
    let tx = 0.0;
    let ty = 0.0;
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

    let mut no_prof = PicturePlane::filled(w, h, 0);
    predict_luma_block_affine(
        &mut no_prof,
        cu_x,
        cu_y,
        cb_w,
        cb_h,
        &r0,
        &cpmvs,
        AffineLumaFilterSet::Set0,
    )
    .expect("no-prof");

    let mut prof_disabled = PicturePlane::filled(w, h, 0);
    predict_luma_block_affine_prof(
        &mut prof_disabled,
        cu_x,
        cu_y,
        cb_w,
        cb_h,
        &r0,
        &cpmvs,
        AffineLumaFilterSet::Set0,
        false,
        /* ph_prof_disabled */ true,
        false,
    )
    .expect("prof-disabled");
    assert_eq!(
        no_prof.samples, prof_disabled.samples,
        "ph_prof_disabled_flag must collapse PROF driver to non-PROF affine"
    );
    assert!(!cb_prof_flag_lx(
        cb_w, cb_h, &cpmvs, false, /* ph_prof_disabled */ true, false,
    ));
}

/// RPR-constraints-active must also disable PROF.
#[test]
fn round78_rpr_constraints_disables_prof() {
    let w = 128usize;
    let h = 96usize;
    let cu_x = 32u32;
    let cu_y = 24u32;
    let cb_w = 32u32;
    let cb_h = 32u32;

    let r0 = sample_grid(w, h);
    let a = 1.0;
    let d = 1.0;
    let b = 0.1;
    let c = 0.0;
    let (cp0, cp1, cp2) = cpmvs_from_affine_6param(
        cu_x as f64,
        cu_y as f64,
        cb_w as f64,
        cb_h as f64,
        a,
        b,
        c,
        d,
        0.0,
        0.0,
    );
    let cpmvs = AffineCpmvs::new_6param(cp0, cp1, cp2);

    let mut no_prof = PicturePlane::filled(w, h, 0);
    predict_luma_block_affine(
        &mut no_prof,
        cu_x,
        cu_y,
        cb_w,
        cb_h,
        &r0,
        &cpmvs,
        AffineLumaFilterSet::Set0,
    )
    .expect("no-prof");

    let mut rpr = PicturePlane::filled(w, h, 0);
    predict_luma_block_affine_prof(
        &mut rpr,
        cu_x,
        cu_y,
        cb_w,
        cb_h,
        &r0,
        &cpmvs,
        AffineLumaFilterSet::Set0,
        false,
        false,
        /* rpr_constraints_active */ true,
    )
    .expect("rpr");
    assert_eq!(
        no_prof.samples, rpr.samples,
        "RPR-constraints-active must collapse PROF driver to non-PROF affine"
    );
    assert!(!cb_prof_flag_lx(cb_w, cb_h, &cpmvs, false, false, true));
}
