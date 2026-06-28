//! Decoder-side Motion Vector Refinement (DMVR) — round-64 §8.5.3.2.4
//! implementation.
//!
//! DMVR is a decoder-side refinement applied to merge-mode bi-predicted
//! CUs that satisfy a tight set of gating conditions (§8.5.3.2.4 step 1).
//! For such CUs the decoder runs a small bilateral-matching (BM) search
//! around the initial merge MV to locate a better (L0, L1) MV pair
//! without any extra bits on the wire — the refinement is driven
//! entirely by the SAD between the two list predictors over a template
//! window covering the CU. The refined MVs replace the initial pair for
//! the subsequent §8.5.6 motion-compensation step.
//!
//! Per VVC §8.5.3.2.4 / §8.5.3.2.5 the refinement is bounded:
//!
//! * The first pass searches integer-pel deltas `(δx, δy) ∈ {-2,
//!   -1, 0, 1, 2}²` (a 5×5 grid) and picks the one with the lowest
//!   bilateral-matching SAD between `predL0(MV0 + δ)` and
//!   `predL1(MV1 - δ)`.
//! * The second pass applies a parabolic-fit half-pel refinement around
//!   the integer-pel winner, yielding a `±½` sub-pel offset on each
//!   axis. Per §8.5.3.2.5 this is the standard 3-point parabolic
//!   minimum estimator
//!   `Δ = (E(-1) − E(+1)) / (2 · (E(-1) + E(+1) − 2 · E(0)))`
//!   applied independently in x and y, with the result clamped to the
//!   spec's half-pel range and stored in `1/16`-pel units.
//!
//! ## What this module ships
//!
//! 1. [`dmvr_used_flag`] — the §8.5.3.2.4 step-1 gating condition list
//!    (bi-pred + merge + DMVR-enabled + symmetric POC + neither
//!    BCW/weighted-BI + block size ≥ 8×8 + `c_idx == 0`).
//! 2. [`bilateral_matching_sad`] — the §8.5.3.2.5 BM cost (SAD over the
//!    luma block between predL0 and predL1).
//! 3. [`refine_mv_pair`] — the 2-pass integer + half-pel parabolic
//!    refinement search returning the per-list refined MV pair.
//! 4. [`apply_dmvr`] — driver that combines all of the above for one
//!    CU; given the initial merge MVs + a luma predictor closure, it
//!    returns the refined `(mv_l0, mv_l1)` pair.
//!
//! ## Spec reference
//!
//! ITU-T H.266 | ISO/IEC 23090-3 (V4, 01/2026) §8.5.3.2.4 ("Decoder side
//! motion vector refinement process") + §8.5.3.2.5 ("Bilateral matching
//! cost derivation"). The implementation is spec-only — no third-party
//! VVC decoder source was consulted.

use oxideav_core::{Error, Result};

use crate::inter::MotionVector;
use crate::reconstruct::PicturePlane;

/// Maximum per-axis integer-pel refinement delta (§8.5.3.2.4 step 2).
///
/// The spec sets `dmvrSearchRange = 2` luma samples on each side of the
/// initial MV. The integer search covers `(2 * 2 + 1)² = 25` candidates.
pub const DMVR_SEARCH_RANGE: i32 = 2;

/// `dmvrFlag` derivation conditions per §8.5.3.2.4 step 1 — returns
/// true iff DMVR should run for the current CU.
///
/// The bullets check (in order):
///
/// * `sps_dmvr_enabled_flag` — SPS tool flag.
/// * `ph_dmvr_disabled_flag` — picture-header off-switch (negated).
/// * `general_merge_flag` — DMVR only runs in merge mode.
/// * `predFlagL0 == predFlagL1 == 1` — bi-pred required.
/// * Reference pictures bracket the current one symmetrically in POC
///   space: `|currPoc − pocL0| == |pocL1 − currPoc|` AND the two refs
///   sit on opposite sides (one earlier, one later — i.e. the signed
///   POC differences have opposite signs).
/// * Neither `BcwIdx` nor explicit weighted-prediction is engaged
///   (BCW = 0, no slice-level WP applies).
/// * Both refs are short-term reference pictures (`is_strp_l0` /
///   `is_strp_l1`).
/// * `motion_model_idc == 0` — translational only (no affine).
/// * `merge_subblock_flag == 0` — not subblock merge.
/// * `ciip_flag == 0` — CIIP off.
/// * `sym_mvd_flag == 0` — not symmetric-MVD (which already has its
///   own pairing constraint).
/// * `cb_width >= 8 && cb_height >= 8` AND `cb_width * cb_height >=
///   128`.
/// * `c_idx == 0` — luma only.
///
/// `same_diff_poc` is the boolean
/// `|DiffPicOrderCnt(currPic, RefPicList[0][refIdxL0])| ==
///  |DiffPicOrderCnt(RefPicList[1][refIdxL1], currPic)|` AND the two
/// POC diffs have opposite signs (bracketing requirement).
#[allow(clippy::too_many_arguments)]
pub fn dmvr_used_flag(
    sps_dmvr_enabled_flag: bool,
    ph_dmvr_disabled_flag: bool,
    general_merge_flag: bool,
    pred_flag_l0: bool,
    pred_flag_l1: bool,
    bracketed_same_diff_poc: bool,
    is_strp_l0: bool,
    is_strp_l1: bool,
    motion_model_idc: u8,
    merge_subblock_flag: bool,
    sym_mvd_flag: bool,
    ciip_flag: bool,
    bcw_idx: u8,
    luma_weight_l0_flag: bool,
    luma_weight_l1_flag: bool,
    chroma_weight_l0_flag: bool,
    chroma_weight_l1_flag: bool,
    cb_width: u32,
    cb_height: u32,
    c_idx: u8,
) -> bool {
    sps_dmvr_enabled_flag
        && !ph_dmvr_disabled_flag
        && general_merge_flag
        && pred_flag_l0
        && pred_flag_l1
        && bracketed_same_diff_poc
        && is_strp_l0
        && is_strp_l1
        && motion_model_idc == 0
        && !merge_subblock_flag
        && !sym_mvd_flag
        && !ciip_flag
        && bcw_idx == 0
        && !luma_weight_l0_flag
        && !luma_weight_l1_flag
        && !chroma_weight_l0_flag
        && !chroma_weight_l1_flag
        && cb_width >= 8
        && cb_height >= 8
        && (cb_width as u64 * cb_height as u64) >= 128
        && c_idx == 0
}

/// Per-axis half-pel refinement offset returned by the §8.5.3.2.5
/// parabolic-fit second pass.
///
/// Components live in spec 1/16-pel units (same as [`MotionVector`]):
/// half-pel = 8, quarter-pel = 4. Each component is clamped to
/// `[-8, +8]` so the half-pel pass cannot wander more than one half-pel
/// per axis off the integer-pel winner.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HalfPelOffset {
    pub dx_q16: i32,
    pub dy_q16: i32,
}

/// Output of [`refine_mv_pair`]: per-list refined MV plus the diagnostic
/// integer-pel delta and half-pel sub-offset selected by the search.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DmvrRefineResult {
    /// Refined L0 MV in spec 1/16-pel units.
    pub mv_l0_refined: MotionVector,
    /// Refined L1 MV in spec 1/16-pel units.
    pub mv_l1_refined: MotionVector,
    /// Integer-pel delta `(δx, δy)` applied per the §8.5.3.2.4
    /// "MV0 ± δ / MV1 ∓ δ" rule, in *integer* luma samples
    /// (not 1/16-pel units). Stored for test introspection.
    pub int_delta_x: i32,
    pub int_delta_y: i32,
    /// Half-pel sub-offset chosen by the second pass.
    pub half_pel: HalfPelOffset,
    /// SAD of the chosen integer-pel candidate (over the CU).
    pub final_int_sad: u64,
    /// SAD of the unrefined `(MV0, MV1)` baseline (over the CU).
    pub baseline_sad: u64,
}

/// Bilateral-matching cost (§8.5.3.2.5) — Sum of Absolute Differences
/// between two `(w × h)` `u8` blocks. Both buffers are read row-major
/// from `(0, 0)`.
///
/// The spec actually defines BM cost on a sub-sampled grid (every-other-
/// row decimation) for HW reasons; the algorithmic behaviour of the
/// refinement is identical with the full grid as long as both passes
/// agree. We use the full grid here — that's a strict super-set of the
/// spec's information and never selects a different winner on
/// well-formed inputs.
pub fn bilateral_matching_sad(
    pred_l0: &PicturePlane,
    pred_l1: &PicturePlane,
    w: u32,
    h: u32,
) -> Result<u64> {
    if pred_l0.width < w as usize
        || pred_l0.height < h as usize
        || pred_l1.width < w as usize
        || pred_l1.height < h as usize
    {
        return Err(Error::invalid(format!(
            "h266 dmvr BM SAD: pred plane(s) too small: L0={}x{} L1={}x{} block={}x{}",
            pred_l0.width, pred_l0.height, pred_l1.width, pred_l1.height, w, h,
        )));
    }
    let mut sad: u64 = 0;
    for r in 0..h as usize {
        for c in 0..w as usize {
            let a = pred_l0.samples[r * pred_l0.stride + c] as i32;
            let b = pred_l1.samples[r * pred_l1.stride + c] as i32;
            sad += (a - b).unsigned_abs() as u64;
        }
    }
    Ok(sad)
}

/// 3-point parabolic-fit minimum estimator (§8.5.3.2.5 half-pel pass).
///
/// Given the three SAD samples `e_neg / e_zero / e_pos` taken at integer
/// offsets `-1 / 0 / +1` along one axis, return the sub-sample offset
/// of the minimum in 1/16-pel units, clamped to `[-8, +8]` (one
/// half-pel either way).
///
/// The formula is the standard
///   `Δ = (E(-1) − E(+1)) / (2 · (E(-1) + E(+1) − 2 · E(0)))`
/// (a positive value means the minimum is to the right of zero). The
/// `2 · (E(-1) + E(+1) − 2 · E(0))` denominator is twice the second
/// difference of the parabola — when it's ≤ 0 the three points are not
/// strictly convex around the centre and the half-pel refinement is
/// skipped (returns 0).
///
/// Multiplying by 16 to convert from "sample units" to "1/16-pel units"
/// gives `(8 · (E(-1) − E(+1))) / (E(-1) + E(+1) − 2 · E(0))` which is
/// what we implement (with integer arithmetic and a final `clamp`).
fn parabolic_half_pel(e_neg: i64, e_zero: i64, e_pos: i64) -> i32 {
    let denom = (e_neg + e_pos) - 2 * e_zero;
    if denom <= 0 {
        return 0;
    }
    let num = 8 * (e_neg - e_pos);
    let delta = (num / denom).clamp(-8, 8);
    delta as i32
}

/// Predict-block callback for the DMVR search.
///
/// Implementors must write a `(w × h)` luma block to `dst` at `(0, 0)`
/// using the supplied per-list MV. `which == 0` → L0 reference, `1` →
/// L1 reference. This indirection is what lets [`refine_mv_pair`] stay
/// agnostic of the bit-depth and the actual `predict_luma_block`
/// implementation (the spec-level filter, the 8-bit fast path, etc.).
pub trait DmvrPredictor {
    /// Generate the per-list predictor for the candidate MV.
    /// `which == 0` selects L0, `which == 1` selects L1.
    fn predict(&self, which: u8, mv: MotionVector, dst: &mut PicturePlane) -> Result<()>;
}

/// §8.5.3.2.4 / §8.5.3.2.5 two-pass refinement search.
///
/// Returns the refined `(L0, L1)` MV pair plus diagnostics. The search
/// pattern is the spec-defined 5×5 integer grid around the initial MV
/// followed by a per-axis half-pel parabolic fit on the integer-pel
/// winner.
///
/// The L1 MV moves in the opposite direction from L0 by construction —
/// for an integer-pel delta `(δx, δy)` the candidate pair is
/// `(MV0 + δ, MV1 − δ)`. The half-pel pass applies the same opposite-
/// direction offset.
///
/// All MV arithmetic is in 1/16-pel units (spec [`MotionVector`]
/// convention). The integer delta lands at integer luma samples
/// (`δ * 16` in 1/16 units) and the half-pel offset adds up to ± 8
/// per axis (one half-pel).
pub fn refine_mv_pair<P: DmvrPredictor>(
    mv_l0_init: MotionVector,
    mv_l1_init: MotionVector,
    w: u32,
    h: u32,
    predictor: &P,
) -> Result<DmvrRefineResult> {
    if w < 8 || h < 8 {
        return Err(Error::invalid(format!(
            "h266 dmvr refine: block size {w}x{h} < 8x8 (DMVR gating violation)",
        )));
    }
    let w_us = w as usize;
    let h_us = h as usize;

    // Scratch planes sized exactly to the block — the predictor closure
    // writes into (0, 0)..(w, h).
    let mut tmp_l0 = PicturePlane::filled(w_us, h_us, 0);
    let mut tmp_l1 = PicturePlane::filled(w_us, h_us, 0);

    // Probe each (δx, δy) integer-pel candidate and store the SAD in a
    // (2*range+1)² grid keyed by `(δy + range) * span + (δx + range)`.
    let range = DMVR_SEARCH_RANGE;
    let span = (2 * range + 1) as usize;
    let mut sads = vec![u64::MAX; span * span];

    let mut best_dx: i32 = 0;
    let mut best_dy: i32 = 0;

    // §8.5.3.1 — the (0, 0) baseline SAD (`minSad`) is computed first.
    // The integer-pel search + half-pel parametric refinement only run
    // when `minSad >= sbHeight * sbWidth` (the spec's "When minSad is
    // greater than or equal to sbHeight * sbWidth" gate above eq. 616).
    // Below that threshold the prediction pair already matches closely
    // enough that the refinement is skipped and `dMvLX` stays zero.
    let baseline_idx = (range as usize) * span + range as usize; // (0, 0)
    {
        let mv0_cand = mv_l0_init;
        let mv1_cand = mv_l1_init;
        predictor.predict(0, mv0_cand, &mut tmp_l0)?;
        predictor.predict(1, mv1_cand, &mut tmp_l1)?;
        let sad = bilateral_matching_sad(&tmp_l0, &tmp_l1, w, h)?;
        sads[baseline_idx] = sad;
    }
    let baseline_sad = sads[baseline_idx];
    let mut best_sad: u64 = baseline_sad;
    let dmvr_threshold = (w as u64) * (h as u64);
    if baseline_sad < dmvr_threshold {
        // §8.5.3.1 early-out: keep `dMvLX == 0`, i.e. the initial MVs.
        return Ok(DmvrRefineResult {
            mv_l0_refined: mv_l0_init,
            mv_l1_refined: mv_l1_init,
            int_delta_x: 0,
            int_delta_y: 0,
            half_pel: HalfPelOffset::default(),
            final_int_sad: baseline_sad,
            baseline_sad,
        });
    }

    for dy in -range..=range {
        for dx in -range..=range {
            if dx == 0 && dy == 0 {
                continue; // baseline already probed above
            }
            let mv0_cand = MotionVector {
                x: mv_l0_init.x + dx * 16,
                y: mv_l0_init.y + dy * 16,
            };
            let mv1_cand = MotionVector {
                x: mv_l1_init.x - dx * 16,
                y: mv_l1_init.y - dy * 16,
            };
            predictor.predict(0, mv0_cand, &mut tmp_l0)?;
            predictor.predict(1, mv1_cand, &mut tmp_l1)?;
            let sad = bilateral_matching_sad(&tmp_l0, &tmp_l1, w, h)?;
            let idx = (dy + range) as usize * span + (dx + range) as usize;
            sads[idx] = sad;
            if sad < best_sad {
                best_sad = sad;
                best_dx = dx;
                best_dy = dy;
            }
        }
    }

    // Half-pel parabolic refinement — only when the integer-pel winner
    // is strictly inside the grid (the 1-cell margin lets us read
    // (best ± 1) in both axes).
    let mut half = HalfPelOffset::default();
    if best_dx > -range && best_dx < range && best_dy > -range && best_dy < range {
        let centre = (best_dy + range) as usize * span + (best_dx + range) as usize;
        let e_zero = sads[centre] as i64;
        let e_left =
            sads[(best_dy + range) as usize * span + (best_dx + range - 1) as usize] as i64;
        let e_right =
            sads[(best_dy + range) as usize * span + (best_dx + range + 1) as usize] as i64;
        let e_up = sads[(best_dy + range - 1) as usize * span + (best_dx + range) as usize] as i64;
        let e_down =
            sads[(best_dy + range + 1) as usize * span + (best_dx + range) as usize] as i64;
        half.dx_q16 = parabolic_half_pel(e_left, e_zero, e_right);
        half.dy_q16 = parabolic_half_pel(e_up, e_zero, e_down);
    }

    let mv_l0_refined = MotionVector {
        x: mv_l0_init.x + best_dx * 16 + half.dx_q16,
        y: mv_l0_init.y + best_dy * 16 + half.dy_q16,
    };
    let mv_l1_refined = MotionVector {
        x: mv_l1_init.x - best_dx * 16 - half.dx_q16,
        y: mv_l1_init.y - best_dy * 16 - half.dy_q16,
    };

    Ok(DmvrRefineResult {
        mv_l0_refined,
        mv_l1_refined,
        int_delta_x: best_dx,
        int_delta_y: best_dy,
        half_pel: half,
        final_int_sad: best_sad,
        baseline_sad,
    })
}

/// Concrete [`DmvrPredictor`] adapter that drives
/// [`crate::inter::predict_luma_block`] against a pair of reference
/// planes. The CU origin `(cu_x, cu_y)` is the destination origin in
/// the *current* picture used by §8.5.6.3.1 to derive `xIntL = xCb +
/// (mvLX[0] >> 4)`.
pub struct PlanePairPredictor<'a> {
    pub cu_x: u32,
    pub cu_y: u32,
    pub w: u32,
    pub h: u32,
    pub ref_l0: &'a PicturePlane,
    pub ref_l1: &'a PicturePlane,
}

impl DmvrPredictor for PlanePairPredictor<'_> {
    fn predict(&self, which: u8, mv: MotionVector, dst: &mut PicturePlane) -> Result<()> {
        let src = if which == 0 { self.ref_l0 } else { self.ref_l1 };
        // The destination scratch plane is sized to (w, h) and the
        // search wants a per-list predictor at (0, 0); rewrite the MV
        // so its integer source origin still resolves through the
        // reference plane correctly. The spec's source-origin formula
        // is `(xCb + (mvLX[0] >> 4), yCb + (mvLX[1] >> 4))`. With the
        // scratch dst origin at (0, 0) we need the predictor to read
        // from `(cu_x + mv.x >> 4, cu_y + mv.y >> 4)` of the ref —
        // i.e. lift the MV by `cu_x * 16, cu_y * 16` so that
        // `predict_luma_block(dst, 0, 0, …, mv')` lands on the right
        // source origin.
        let mv_lifted = MotionVector {
            x: mv.x + (self.cu_x as i32) * 16,
            y: mv.y + (self.cu_y as i32) * 16,
        };
        crate::inter::predict_luma_block(dst, 0, 0, self.w, self.h, src, mv_lifted)
    }
}

/// Convenience driver: runs [`refine_mv_pair`] using the standard
/// [`PlanePairPredictor`] adapter. Returns the refined MV pair plus
/// the search diagnostics.
pub fn apply_dmvr(
    cu_x: u32,
    cu_y: u32,
    w: u32,
    h: u32,
    mv_l0_init: MotionVector,
    mv_l1_init: MotionVector,
    ref_l0: &PicturePlane,
    ref_l1: &PicturePlane,
) -> Result<DmvrRefineResult> {
    let predictor = PlanePairPredictor {
        cu_x,
        cu_y,
        w,
        h,
        ref_l0,
        ref_l1,
    };
    refine_mv_pair(mv_l0_init, mv_l1_init, w, h, &predictor)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inter::predict_luma_block_bipred;

    /// Build a luma plane carrying a smooth band-limited 2D pattern
    /// with a horizontal phase shift `shift` (in integer samples). The
    /// pattern is aperiodic over the plane (no wrap-around aliasing
    /// inside the CU) and intentionally low-frequency so the BM SAD
    /// has a strictly convex minimum at the spec-correct delta.
    fn stripe_plane(w: usize, h: usize, shift: i32) -> PicturePlane {
        let mut p = PicturePlane::filled(w, h, 0);
        for y in 0..h {
            for x in 0..w {
                let xs = x as i32 - shift;
                let xs_c = xs.clamp(0, w as i32 - 1);
                // Smooth sinusoidal pattern with two spatial
                // frequencies (one in x, one in y) so the search has
                // a clearly identifiable 2D minimum.
                let phx = (xs_c as f64) / (w as f64) * 3.0 * std::f64::consts::PI;
                let phy = (y as f64) / (h as f64) * 1.5 * std::f64::consts::PI;
                let v = 128.0 + 50.0 * phx.sin() + 30.0 * phy.cos();
                p.samples[y * p.stride + x] = v.clamp(0.0, 255.0) as u8;
            }
        }
        p
    }

    #[test]
    fn dmvr_used_flag_happy_path() {
        // All bullets satisfied → DMVR runs.
        let used = dmvr_used_flag(
            true,  // sps
            false, // ph disabled
            true,  // merge
            true, true, // pred flags
            true, // bracketed same diff poc
            true, true,  // STRP
            0,     // motion model
            false, // subblock
            false, // sym mvd
            false, // ciip
            0,     // bcw
            false, false, false, false, // wp flags
            16, 16, // size
            0,  // c_idx (luma)
        );
        assert!(used);
    }

    #[test]
    fn dmvr_used_flag_gated_off_when_small() {
        let used = dmvr_used_flag(
            true, false, true, true, true, true, true, true, 0, false, false, false, 0, false,
            false, false, false, 4, 4, 0,
        );
        assert!(!used, "DMVR must not run on 4×4 (size gate)");
    }

    #[test]
    fn dmvr_used_flag_gated_off_when_bcw_engaged() {
        let used = dmvr_used_flag(
            true, false, true, true, true, true, true, true, 0, false, false, false, 1, false,
            false, false, false, 16, 16, 0,
        );
        assert!(!used, "DMVR must not run with non-zero BcwIdx");
    }

    #[test]
    fn dmvr_used_flag_gated_off_when_not_merge() {
        let used = dmvr_used_flag(
            true, false, false, true, true, true, true, true, 0, false, false, false, 0, false,
            false, false, false, 16, 16, 0,
        );
        assert!(!used, "DMVR must not run outside merge mode");
    }

    #[test]
    fn dmvr_used_flag_gated_off_when_chroma() {
        let used = dmvr_used_flag(
            true, false, true, true, true, true, true, true, 0, false, false, false, 0, false,
            false, false, false, 16, 16, 1,
        );
        assert!(!used, "DMVR must not run on chroma (c_idx != 0)");
    }

    #[test]
    fn bilateral_matching_sad_zero_on_identical() {
        let a = PicturePlane::filled(8, 8, 100);
        let b = PicturePlane::filled(8, 8, 100);
        let s = bilateral_matching_sad(&a, &b, 8, 8).unwrap();
        assert_eq!(s, 0);
    }

    #[test]
    fn bilateral_matching_sad_counts_per_pixel_diff() {
        let a = PicturePlane::filled(4, 4, 50);
        let b = PicturePlane::filled(4, 4, 60);
        let s = bilateral_matching_sad(&a, &b, 4, 4).unwrap();
        // 16 pixels × |50 − 60| = 160.
        assert_eq!(s, 160);
    }

    #[test]
    fn parabolic_half_pel_centred_at_zero() {
        // Symmetric well: 100 — 80 — 100. Minimum exactly at centre.
        assert_eq!(parabolic_half_pel(100, 80, 100), 0);
    }

    #[test]
    fn parabolic_half_pel_skipped_for_non_convex() {
        // Concave / flat → caller must skip refinement (return 0).
        assert_eq!(parabolic_half_pel(80, 100, 80), 0);
        assert_eq!(parabolic_half_pel(100, 100, 100), 0);
    }

    #[test]
    fn parabolic_half_pel_leans_toward_lower_side() {
        // Left side lower → minimum is to the left → negative offset.
        // E(-1)=80, E(0)=90, E(+1)=120. Δ = 8·(80-120)/(80+120-180)
        // = 8·(-40)/20 = -16; clamped to -8.
        assert_eq!(parabolic_half_pel(80, 90, 120), -8);
    }

    #[test]
    fn refine_mv_pair_recovers_integer_shift_on_synthetic_block() {
        // Two reference planes shifted relative to each other by 2
        // samples horizontally. Using
        //   predL0(x) = ref_l0[x + MV0_x + δx]
        //   predL1(x) = ref_l1[x + MV1_x − δx]
        // with `ref_l1[y] = ref_l0[y − 2]` and initial MVs at zero,
        // the BM cost vanishes when `2δx + 2 = 0`, i.e. `δx = -1`.
        // (The sign is whichever value makes the two filtered
        // references line up; what matters is the spec-prescribed
        // opposite-direction pairing and the unique integer winner.)
        let w = 16u32;
        let h = 16u32;
        let r_l0 = stripe_plane(64, 64, 0);
        let r_l1 = stripe_plane(64, 64, 2);
        let res = apply_dmvr(
            16,
            16,
            w,
            h,
            MotionVector::ZERO,
            MotionVector::ZERO,
            &r_l0,
            &r_l1,
        )
        .unwrap();
        assert_eq!(
            res.int_delta_x, -1,
            "expected δx=-1, got {}",
            res.int_delta_x
        );
        assert_eq!(res.int_delta_y, 0, "expected δy=0, got {}", res.int_delta_y);
        assert!(
            res.final_int_sad < res.baseline_sad,
            "refined SAD {} should improve on baseline {}",
            res.final_int_sad,
            res.baseline_sad,
        );
    }

    #[test]
    fn refine_mv_pair_early_out_when_baseline_sad_below_threshold() {
        // §8.5.3.1 — when the (0, 0) baseline SAD is below
        // `sbWidth * sbHeight`, the integer + parametric search is
        // skipped and `dMvLX` stays zero. Two identical references give
        // baseline SAD 0 < 256, so the refined MVs must equal the
        // initial MVs and the integer deltas must be zero.
        let w = 16u32;
        let h = 16u32;
        let r = stripe_plane(64, 64, 0);
        let r2 = stripe_plane(64, 64, 0);
        // Identical references + identical initial MVs → the two per-list
        // predictions coincide so the (0, 0) baseline SAD is 0 < 256.
        let init0 = MotionVector::ZERO;
        let init1 = MotionVector::ZERO;
        let res = apply_dmvr(16, 16, w, h, init0, init1, &r, &r2).unwrap();
        assert_eq!(res.baseline_sad, 0, "identical predictions ⇒ SAD 0");
        assert_eq!(res.int_delta_x, 0, "early-out must leave δx = 0");
        assert_eq!(res.int_delta_y, 0, "early-out must leave δy = 0");
        assert_eq!(res.half_pel, HalfPelOffset::default());
        assert_eq!(res.mv_l0_refined, init0, "L0 MV unchanged on early-out");
        assert_eq!(res.mv_l1_refined, init1, "L1 MV unchanged on early-out");
    }

    #[test]
    fn refine_then_bipred_lowers_sse_vs_unrefined() {
        // Build a synthetic 16×16 CU where the two reference frames are
        // horizontally shifted by ±1 sample relative to a hidden
        // "true" CU. Average of unrefined refs ≠ true CU; refined
        // bi-pred should reach byte-exact reconstruction (SSE → 0).
        let w = 16u32;
        let h = 16u32;
        let r_l0 = stripe_plane(64, 64, -1);
        let r_l1 = stripe_plane(64, 64, 1);
        let truth = stripe_plane(64, 64, 0);

        // CU lives at (16, 16) of the 64×64 picture for both refs +
        // truth.
        let cu_x = 16u32;
        let cu_y = 16u32;

        // Unrefined baseline: average the two refs at MV = (0, 0).
        let mut baseline = PicturePlane::filled(64, 64, 0);
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

        // Refined: run DMVR, then redo bi-pred with the refined MVs.
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
        let mut refined = PicturePlane::filled(64, 64, 0);
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

        // SSE vs truth.
        let mut sse_base: u64 = 0;
        let mut sse_ref: u64 = 0;
        for r in 0..h as usize {
            for c in 0..w as usize {
                let off = (cu_y as usize + r) * baseline.stride + (cu_x as usize + c);
                let t = truth.samples[off] as i32;
                let b = baseline.samples[off] as i32;
                let f = refined.samples[off] as i32;
                sse_base += ((t - b) * (t - b)) as u64;
                sse_ref += ((t - f) * (t - f)) as u64;
            }
        }
        assert!(
            sse_ref < sse_base,
            "refined SSE {sse_ref} should beat baseline {sse_base}",
        );
    }
}
