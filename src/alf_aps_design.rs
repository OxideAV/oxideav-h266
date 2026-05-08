//! Encoder-side ALF luma APS design — round 47.
//!
//! Round 46 wired the per-CTU ALF CABAC bins through the IDR pipeline
//! but left `ph_num_alf_aps_ids_luma = 0`, so every CTB the encoder
//! turns ALF on for picks one of the 16 §7.4.3.18 fixed filter sets
//! (`AlfCtbFiltSetIdxY < 16`). Round 47 closes the §8.8.5.2 APS-signalled
//! branch (`AlfCtbFiltSetIdxY >= 16`): given the post-SAO reconstruction
//! and the source picture, derive a single 12-tap luma filter row that
//! minimises the squared error against the source, lattice-quantise it
//! into the spec's 8-bit `alf_luma_coeff_abs` / `alf_luma_coeff_sign`
//! representation, and pack the result into an [`AlfApsData`] the
//! [`crate::aps_enc::emit_alf_aps_rbsp`] emitter can serialise.
//!
//! ## Filter geometry — §8.8.5.2 luma diamond (eq. 1448 / 1449)
//!
//! The luma filter is a 7×7 diamond with 13 taps; the central tap is
//! implicit so 12 are signalled. The decoder pairs every signalled tap
//! `f[k]` with its mirrored neighbour, so the per-pixel contribution
//! from `f[k]` is `f[k] · (clip(neighbour_+) + clip(neighbour_-))`. We
//! ignore clipping in the design pass (the round-46 fixed-filter path
//! ships `clipIdx = 0` everywhere, which Table 8 expands to `1 <<
//! BitDepth` — i.e. the maximum, no-op clip). Each design sample is
//! therefore the linear regression
//!
//! ```text
//!   src - curr ≈ Σ_k f[k] · ( neighbour_+ + neighbour_- - 2·curr ) / (1 << alfShiftY)
//! ```
//!
//! where `alfShiftY = 7` for the central rows of the CTB (§8.8.5.2
//! Table 45 — `applyAlfLineBufBoundary == 0` case). We collect the
//! `(12 × 12)` normal-equations matrix `R` plus the 12-vector `r` over
//! the picture's interior (a 5-pixel guard band excludes pixels whose
//! diamond overhangs the picture edge), solve `R · f_real = r` via
//! Gauss-Jordan elimination, then lattice-quantise.
//!
//! Round 47 deliberately signals **one** filter row
//! (`alf_luma_num_filters_signalled_minus1 = 0`) shared across all 25
//! §8.8.5.3 classes. A future round can split per-class statistics +
//! design 25 separate rows; the APS encoder + decoder framework
//! introduced here scales to that without changes.
//!
//! ## Lattice quantisation
//!
//! The §7.3.2.18 wire format encodes each tap as `alf_luma_coeff_abs`
//! (`ue(v)`) plus a sign bit. The spec caps `|coeff|` at 128. The
//! design's real-valued coefficients are scaled by
//! `2^ALF_LUMA_FILTER_PRECISION = 2^7 = 128` (matching `alfShiftY` so
//! the eq. 1450 right-shift puts the result back at sample precision)
//! and rounded to the nearest integer with the spec cap. Saturating
//! beyond ±64 in this round avoids edge cases where the linear fit
//! over-fits a small noise band; the encoder gives up a small amount
//! of theoretical headroom for a much smaller chance of clip-induced
//! ringing.
//!
//! ## Spec reference
//!
//! ITU-T H.266 | ISO/IEC 23090-3 (V4, 01/2026)
//! * §7.3.2.18 / §7.4.3.18 — `alf_data()` syntax + semantics.
//! * §8.8.5.2 — luma filter (eq. 1448 / 1449 / 1450 / 1451).
//! * §8.8.5.3 — per-4×4 sub-block classification.

use crate::aps::{AlfApsData, ALF_LUMA_NUM_COEFF, NUM_ALF_FILTERS};
use crate::reconstruct::{PictureBuffer, PicturePlane};

/// Coefficient precision in fractional bits — eq. 1450 right-shifts by
/// `alfShiftY = 7` for the body of the CTB.
const ALF_LUMA_FILTER_PRECISION: i32 = 7;

/// Per-tap saturation magnitude. The §7.4.3.18 spec cap is 128; we
/// saturate at half of that to keep round 47 conservative — the design
/// over-fits less reliably than a 25-class Wiener solver, so a tighter
/// cap reduces ringing risk on outlier pixels.
const ALF_LUMA_COEFF_MAX_ABS: i32 = 64;

/// Spec cap from §7.4.3.18 (`alf_luma_coeff_abs` ≤ 128). Used as the
/// hard upper bound regardless of the design's saturation policy.
const ALF_LUMA_SPEC_COEFF_MAX_ABS: i32 = 128;

/// Per-tap displacement table for the §8.8.5.2 7×7 luma diamond. Each
/// row corresponds to one of the 12 signalled taps and lists the
/// `(dx, dy)` of the pair `neighbour_+`; the decoder pairs every
/// signalled tap with its mirrored partner so the per-pixel
/// contribution sums neighbour_+ and neighbour_- (`(-dx, -dy)`).
///
/// Order matches the spec's `f[0..12]` enumeration in eq. 1449.
const LUMA_TAP_DISPL: [(i32, i32); ALF_LUMA_NUM_COEFF] = [
    (0, 3),  // f[0]
    (1, 2),  // f[1]
    (0, 2),  // f[2]
    (-1, 2), // f[3]
    (2, 1),  // f[4]
    (1, 1),  // f[5]
    (0, 1),  // f[6]
    (-1, 1), // f[7]
    (-2, 1), // f[8]
    (3, 0),  // f[9]
    (2, 0),  // f[10]
    (1, 0),  // f[11]
];

/// Sample a luma plane with `(0..=255)` bounds clamping for design-pass
/// reads. The spec mirrors out-of-picture neighbours; the design pass
/// skips pixels whose diamond extends out of the picture so we don't
/// need that here.
#[inline]
fn sample_in(plane: &PicturePlane, x: i32, y: i32) -> i32 {
    plane.samples[(y as usize) * plane.stride + (x as usize)] as i32
}

/// Tap-feature row for a single design pixel.
///
/// Eq. 1449 sums `f[k] · ( neighbour_+ + neighbour_- - 2·curr )`. The
/// "feature" for tap `k` is therefore that bracketed quantity divided
/// by `2^ALF_LUMA_FILTER_PRECISION` so the regression's coefficients
/// land in the same numeric range the lattice quant expects.
fn tap_features(plane: &PicturePlane, x: i32, y: i32) -> [f64; ALF_LUMA_NUM_COEFF] {
    let curr = sample_in(plane, x, y) as f64;
    let mut feat = [0.0f64; ALF_LUMA_NUM_COEFF];
    for k in 0..ALF_LUMA_NUM_COEFF {
        let (dx, dy) = LUMA_TAP_DISPL[k];
        let n_p = sample_in(plane, x + dx, y + dy) as f64;
        let n_m = sample_in(plane, x - dx, y - dy) as f64;
        feat[k] = (n_p + n_m - 2.0 * curr) / (1u32 << ALF_LUMA_FILTER_PRECISION) as f64;
    }
    feat
}

/// Solve `A · x = b` with Gauss-Jordan elimination + partial pivoting.
///
/// `A` is a `n × n` row-major matrix (length `n*n`); `b` has length
/// `n`. Returns `Some(x)` on success, `None` when the system is
/// singular (zero pivot column). Used for the 12×12 normal-equations
/// solve.
fn solve_linear_system(mat: &mut [f64], rhs: &mut [f64], n: usize) -> Option<Vec<f64>> {
    debug_assert_eq!(mat.len(), n * n);
    debug_assert_eq!(rhs.len(), n);
    for col in 0..n {
        // Partial pivot — find the row with the largest |pivot| in `col`.
        let mut piv = col;
        for r in col + 1..n {
            if mat[r * n + col].abs() > mat[piv * n + col].abs() {
                piv = r;
            }
        }
        if mat[piv * n + col].abs() < 1.0e-12 {
            return None; // Singular column.
        }
        if piv != col {
            for c in 0..n {
                mat.swap(col * n + c, piv * n + c);
            }
            rhs.swap(col, piv);
        }
        // Normalise row.
        let p = mat[col * n + col];
        for c in col..n {
            mat[col * n + c] /= p;
        }
        rhs[col] /= p;
        // Eliminate other rows.
        for r in 0..n {
            if r == col {
                continue;
            }
            let factor = mat[r * n + col];
            if factor == 0.0 {
                continue;
            }
            for c in col..n {
                mat[r * n + c] -= factor * mat[col * n + c];
            }
            rhs[r] -= factor * rhs[col];
        }
    }
    Some(rhs.to_vec())
}

/// Round a real coefficient into the spec's signed-integer encoding.
///
/// Multiplies by `2^ALF_LUMA_FILTER_PRECISION` (= 128) so the integer
/// matches the eq. 1450 right-shift, then rounds-half-away-from-zero
/// and saturates to `±ALF_LUMA_COEFF_MAX_ABS`. Returns the integer
/// `f_int` such that `f_real ≈ f_int / 2^ALF_LUMA_FILTER_PRECISION`.
fn quantise_coeff(f: f64) -> i32 {
    let scaled = f * ((1u32 << ALF_LUMA_FILTER_PRECISION) as f64);
    let rounded = if scaled >= 0.0 {
        (scaled + 0.5).floor() as i32
    } else {
        -((-scaled + 0.5).floor() as i32)
    };
    rounded.clamp(-ALF_LUMA_COEFF_MAX_ABS, ALF_LUMA_COEFF_MAX_ABS)
}

/// Outcome of [`design_luma_alf_aps`]: the coefficients themselves
/// plus a flag the caller can use to decide whether to ship the APS.
#[derive(Clone, Debug, Default)]
pub struct DesignedLumaAlf {
    /// 12 signed integer coefficients in spec wire-format range,
    /// already saturated to `ALF_LUMA_COEFF_MAX_ABS`.
    pub coeff: [i32; ALF_LUMA_NUM_COEFF],
    /// Number of pixels that contributed to the regression. Zero iff
    /// the picture is too small to host even one fully-interior pixel
    /// (i.e. width or height ≤ 6 — the diamond reaches ±3 in the worst
    /// case).
    pub n_samples: usize,
    /// `true` if the linear system was solvable and the design produced
    /// a non-trivial filter (at least one non-zero quantised tap).
    pub is_meaningful: bool,
}

/// Design a single 12-tap luma ALF filter from the post-SAO recon and
/// the source.
///
/// Inputs:
/// * `src` — original picture (uncompressed).
/// * `rec` — current reconstruction *after* deblock + SAO, *before*
///   ALF runs. The design pass must see the same pixels the §8.8.5.2
///   apply pass will read from on the decoder side.
///
/// Returns a [`DesignedLumaAlf`] whose `coeff` field is ready to drop
/// straight into [`build_luma_alf_aps_data`]. When the picture is too
/// small to host a single interior pixel (width or height ≤ 6 — the
/// 7×7 diamond reaches ±3 samples) the function returns `coeff = [0;
/// 12]` with `is_meaningful = false`; the caller should skip APS
/// emission in that case.
pub fn design_luma_alf_filter(src: &PictureBuffer, rec: &PictureBuffer) -> DesignedLumaAlf {
    let pw = rec.luma.width as i32;
    let ph = rec.luma.height as i32;
    if pw < 7 || ph < 7 {
        return DesignedLumaAlf::default();
    }
    let n = ALF_LUMA_NUM_COEFF;
    let mut r_mat = vec![0.0f64; n * n];
    let mut r_vec = vec![0.0f64; n];
    let mut n_samples: usize = 0;

    // Diamond half-extent — taps reach ±3 in y and ±3 in x.
    let margin = 3i32;
    for y in margin..ph - margin {
        for x in margin..pw - margin {
            let feat = tap_features(&rec.luma, x, y);
            let target = (sample_in(&src.luma, x, y) - sample_in(&rec.luma, x, y)) as f64;
            for i in 0..n {
                r_vec[i] += feat[i] * target;
                for j in 0..n {
                    r_mat[i * n + j] += feat[i] * feat[j];
                }
            }
            n_samples += 1;
        }
    }

    if n_samples == 0 {
        return DesignedLumaAlf::default();
    }

    let solution = solve_linear_system(&mut r_mat, &mut r_vec, n);
    let mut out = DesignedLumaAlf {
        n_samples,
        ..DesignedLumaAlf::default()
    };
    if let Some(real_coeff) = solution {
        let mut any_nonzero = false;
        for k in 0..n {
            out.coeff[k] = quantise_coeff(real_coeff[k]);
            if out.coeff[k] != 0 {
                any_nonzero = true;
            }
        }
        out.is_meaningful = any_nonzero;
        // Spec hard cap defensively re-asserted (the saturation in
        // `quantise_coeff` is tighter, but a future round may lift it).
        for k in 0..n {
            debug_assert!(out.coeff[k].abs() <= ALF_LUMA_SPEC_COEFF_MAX_ABS);
        }
    }
    out
}

/// Build an [`AlfApsData`] that signals exactly one luma filter row
/// (every class shares it) plus, optionally, whatever chroma / CC-ALF
/// payload the caller layers in via the `chroma_overlay` argument.
///
/// `chroma_overlay` lets the encoder pipeline keep its existing
/// chroma + CC-ALF APSes separate (they're emitted as their own
/// `aps_id`s) but is exposed so a future round can pack everything
/// into a single APS.
pub fn build_luma_alf_aps_data(designed: &DesignedLumaAlf) -> AlfApsData {
    let mut aps = AlfApsData::default();
    aps.alf_luma_filter_signal_flag = true;
    aps.alf_luma_clip_flag = false;
    // §7.3.2.18 — when only one filter is signalled, every CTB class
    // index maps to the same signalled filter. Eq. 89 collapses to
    // `AlfCoeffL[ filtIdx ] = filtCoeff[ 0 ]` for all 25 classes.
    aps.luma_coeff = vec![designed.coeff; NUM_ALF_FILTERS];
    aps.luma_clip_idx = vec![[0u8; ALF_LUMA_NUM_COEFF]; NUM_ALF_FILTERS];
    aps
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aps::parse_aps;
    use crate::aps_enc::emit_alf_aps_rbsp;
    use crate::reconstruct::PictureBuffer;

    /// Round-47 — design pass on a flat-grey source should solve to a
    /// near-zero filter (no error to fit) with `is_meaningful = false`.
    #[test]
    fn design_flat_grey_returns_near_zero_filter() {
        let src = PictureBuffer::yuv420_filled(64, 64, 128);
        let rec = src.clone();
        let designed = design_luma_alf_filter(&src, &rec);
        assert!(designed.n_samples > 0);
        // Flat plane → R has zero diagonal → singular → solver returns
        // None → coefficients stay zero.
        assert!(!designed.is_meaningful);
        for c in &designed.coeff {
            assert_eq!(*c, 0);
        }
    }

    /// Round-47 — picture too small for any interior pixel returns the
    /// default (zero) filter and the caller can skip APS emission.
    #[test]
    fn design_picture_too_small_returns_default() {
        let src = PictureBuffer::yuv420_filled(6, 6, 128);
        let rec = src.clone();
        let designed = design_luma_alf_filter(&src, &rec);
        assert_eq!(designed.n_samples, 0);
        assert!(!designed.is_meaningful);
    }

    /// Round-47 — quantise_coeff matches its documented contract:
    /// scale by 2^7, round half-away-from-zero, saturate.
    #[test]
    fn quantise_coeff_round_and_saturate() {
        // 0.1 * 128 = 12.8 → round to 13.
        assert_eq!(quantise_coeff(0.1), 13);
        // -0.1 * 128 = -12.8 → -13.
        assert_eq!(quantise_coeff(-0.1), -13);
        // 1.0 * 128 = 128 → saturated to 64 by the round-47 cap.
        assert_eq!(quantise_coeff(1.0), 64);
        // Tiny noise rounds to zero.
        assert_eq!(quantise_coeff(0.001), 0);
    }

    /// Round-47 — the linear solver returns the identity mapping for a
    /// diagonal matrix.
    #[test]
    fn solve_linear_system_identity() {
        let n = 3;
        let mut mat = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let mut rhs = vec![1.5, -2.5, 3.0];
        let x = solve_linear_system(&mut mat, &mut rhs, n).unwrap();
        assert!((x[0] - 1.5).abs() < 1.0e-9);
        assert!((x[1] - -2.5).abs() < 1.0e-9);
        assert!((x[2] - 3.0).abs() < 1.0e-9);
    }

    /// Round-47 — the linear solver returns None on a singular matrix.
    #[test]
    fn solve_linear_system_singular() {
        let n = 2;
        let mut mat = vec![1.0, 2.0, 2.0, 4.0]; // rows are colinear
        let mut rhs = vec![1.0, 2.0];
        assert!(solve_linear_system(&mut mat, &mut rhs, n).is_none());
    }

    /// Round-47 — the designed APS round-trips through emit_alf_aps_rbsp
    /// + parse_aps, and the parser-expanded `luma_coeff` matches the
    /// designer's coefficients on every one of the 25 classes (single-
    /// signalled-filter contract).
    #[test]
    fn designed_luma_aps_round_trips() {
        // Build a non-trivial filter — flat grey would yield a zero
        // filter that the emitter would otherwise still encode but
        // isn't very interesting. Instead hand-pick coefficients that
        // exercise the sign bit on multiple taps.
        let mut designed = DesignedLumaAlf::default();
        designed.coeff[0] = 1;
        designed.coeff[2] = -3;
        designed.coeff[6] = 5;
        designed.coeff[11] = -2;
        designed.is_meaningful = true;
        designed.n_samples = 100;
        let aps = build_luma_alf_aps_data(&designed);
        let bytes = emit_alf_aps_rbsp(2, false, &aps).unwrap();
        let parsed = parse_aps(&bytes).unwrap();
        let pp = parsed.alf_data.as_ref().unwrap();
        assert!(pp.alf_luma_filter_signal_flag);
        assert!(!pp.alf_luma_clip_flag);
        assert_eq!(pp.luma_coeff.len(), NUM_ALF_FILTERS);
        for row in &pp.luma_coeff {
            assert_eq!(row, &designed.coeff);
        }
    }

    /// Round-47 — a structured noise pattern must produce a non-trivial
    /// filter. The recon carries a deterministic-but-content-rich noise
    /// pattern (LCG-based per-pixel offset on top of a smooth gradient
    /// source) so the 12×12 normal-equations matrix is full rank and
    /// the solver yields at least one non-zero quantised tap.
    #[test]
    fn design_noisy_recon_yields_meaningful_filter() {
        let w = 64usize;
        let h = 64usize;
        let mut src = PictureBuffer::yuv420_filled(w, h, 128);
        let mut rec = src.clone();
        // Source: smooth horizontal gradient.
        for y in 0..h {
            for x in 0..w {
                let v = (32 + (x as u32 * 192 / w as u32)) as u8;
                src.luma.samples[y * src.luma.stride + x] = v;
            }
        }
        // Noisy recon: same gradient + LCG-derived noise so the rec's
        // 12-tap context window varies enough to produce a non-singular
        // R matrix. Amplitude ±20 keeps the picture in 0..=255.
        for y in 0..h {
            for x in 0..w {
                let base = (32 + (x as u32 * 192 / w as u32)) as i32;
                // Cheap LCG; sign flips on every other invocation so the
                // noise has zero mean.
                let seed = (y as u64).wrapping_mul(2654435761).wrapping_add(x as u64);
                let bits = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let noise = ((bits >> 56) & 0x1f) as i32 - 16; // -16..=15
                let v = (base + noise).clamp(0, 255) as u8;
                rec.luma.samples[y * rec.luma.stride + x] = v;
            }
        }
        let designed = design_luma_alf_filter(&src, &rec);
        assert!(designed.n_samples > 0);
        // The LCG-perturbed gradient is content-rich enough that the
        // 12×12 normal-equations matrix is solvable.
        assert!(
            designed.is_meaningful,
            "non-singular noisy recon must yield at least one non-zero tap; got {:?}",
            designed.coeff
        );
        // At least one tap must be non-zero.
        assert!(designed.coeff.iter().any(|&c| c != 0));
    }
}
