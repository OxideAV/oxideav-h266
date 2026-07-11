//! VVC intra prediction primitives (§8.4.5.2).
//!
//! Scope of this drop: the mode-agnostic sample-level predictors for
//! the two non-angular modes, `INTRA_PLANAR` (§8.4.5.2.11, eqs. 330 –
//! 332) and `INTRA_DC` (§8.4.5.2.12, eqs. 333 – 336). The angular
//! modes (§8.4.5.2.13) depend on a large reference-sample filter /
//! substitution / wide-angle-remapping pipeline (§8.4.5.2.7 –
//! §8.4.5.2.10) and are deferred to a follow-up increment.
//!
//! Neighbour samples are provided as a slice laid out in the order the
//! spec uses:
//!
//! * `left[y]` for y = 0..nTbH (i.e. nTbH+1 samples, the extra row is
//!   `p[-1][nTbH]` used by planar's eq. 330 right-bottom corner).
//! * `above[x]` for x = 0..nTbW (nTbW+1 samples, `p[nTbW][-1]` is the
//!   above-right corner for planar's eq. 331).
//! * `top_left` for `p[-1][-1]`.
//!
//! DC only uses the first nTbW / nTbH entries of each side. Planar
//! reads the full extended arrays.
//!
//! All sample values are stored as `i16` (they fit in `[0, 1<<BitDepth)`
//! for BitDepth ≤ 15 which covers all VVC profiles).

use oxideav_core::{Error, Result};

/// Reference-sample view for intra prediction. `above[x]` stores
/// `p[x][-1-refIdx]`, `left[y]` stores `p[-1-refIdx][y]`, and
/// `top_left` stores `p[-1][-1]` (used by angular modes; DC / planar
/// ignore it).
#[derive(Clone, Debug)]
pub struct IntraRefs<'a> {
    /// Length = `n_tb_w + 1` — covers `p[0..nTbW][-1]`.
    pub above: &'a [i16],
    /// Length = `n_tb_h + 1` — covers `p[-1][0..nTbH]`.
    pub left: &'a [i16],
    /// `p[-1][-1]`.
    pub top_left: i16,
}

/// INTRA_PLANAR (§8.4.5.2.11).
///
/// `pred[y * n_tb_w + x]` is the row-major output array.
pub fn predict_planar(n_tb_w: usize, n_tb_h: usize, refs: &IntraRefs<'_>) -> Result<Vec<i16>> {
    if refs.above.len() < n_tb_w + 1 || refs.left.len() < n_tb_h + 1 {
        return Err(Error::invalid(
            "h266 intra planar: reference arrays too short",
        ));
    }
    if !n_tb_w.is_power_of_two() || !n_tb_h.is_power_of_two() {
        return Err(Error::invalid(
            "h266 intra planar: nTbW / nTbH must be a power of two",
        ));
    }
    let log2_w = n_tb_w.trailing_zeros();
    let log2_h = n_tb_h.trailing_zeros();
    let mut pred = vec![0i16; n_tb_w * n_tb_h];
    let p_tr = refs.above[n_tb_w] as i32; // p[nTbW][-1]
    let p_bl = refs.left[n_tb_h] as i32; // p[-1][nTbH]
    let round = (n_tb_w * n_tb_h) as i32;
    let shift = log2_w + log2_h + 1;
    for y in 0..n_tb_h {
        for x in 0..n_tb_w {
            let p_a = refs.above[x] as i32; // p[x][-1]
            let p_l = refs.left[y] as i32; // p[-1][y]
                                           // eq. 330 / 331
            let pred_v = ((n_tb_h as i32 - 1 - y as i32) * p_a + (y as i32 + 1) * p_bl) << log2_w;
            let pred_h = ((n_tb_w as i32 - 1 - x as i32) * p_l + (x as i32 + 1) * p_tr) << log2_h;
            // eq. 332
            let p = (pred_v + pred_h + round) >> shift;
            pred[y * n_tb_w + x] = p as i16;
        }
    }
    Ok(pred)
}

/// INTRA_DC (§8.4.5.2.12) with `refIdx = 0`.
///
/// `pred[y * n_tb_w + x]` is the row-major output array, all samples
/// set to the same dcVal.
pub fn predict_dc(n_tb_w: usize, n_tb_h: usize, refs: &IntraRefs<'_>) -> Result<Vec<i16>> {
    if refs.above.len() < n_tb_w || refs.left.len() < n_tb_h {
        return Err(Error::invalid("h266 intra DC: reference arrays too short"));
    }
    if !n_tb_w.is_power_of_two() || !n_tb_h.is_power_of_two() {
        return Err(Error::invalid(
            "h266 intra DC: nTbW / nTbH must be a power of two",
        ));
    }
    let log2_w = n_tb_w.trailing_zeros();
    let log2_h = n_tb_h.trailing_zeros();
    let sum_w: i32 = refs.above[..n_tb_w].iter().map(|&v| v as i32).sum();
    let sum_h: i32 = refs.left[..n_tb_h].iter().map(|&v| v as i32).sum();
    let dc_val = if n_tb_w == n_tb_h {
        // eq. 333
        (sum_w + sum_h + n_tb_w as i32) >> (log2_w + 1)
    } else if n_tb_w > n_tb_h {
        // eq. 334
        (sum_w + (n_tb_w as i32 >> 1)) >> log2_w
    } else {
        // eq. 335
        (sum_h + (n_tb_h as i32 >> 1)) >> log2_h
    };
    let dc = dc_val as i16;
    Ok(vec![dc; n_tb_w * n_tb_h])
}

/// Fill missing reference samples per §8.4.5.2.8.
///
/// Spec-exact substitution: the top-left, above, and left arrays are
/// padded from the nearest available sample. In this foundation
/// implementation the caller passes pre-populated arrays with the
/// "missing" mask; we scan for the first available sample and
/// replicate from there. When none are available, replicate
/// `1 << (BitDepth - 1)` (mid-grey).
///
/// * `above[x]`, `left[y]` — current reference-sample values (in-place).
/// * `above_avail[x]`, `left_avail[y]` — `true` if the sample exists.
/// * `top_left_avail` — availability of `p[-1][-1]`.
/// * `mid` — fallback sample = `1 << (BitDepth - 1)`.
pub fn substitute_references(
    above: &mut [i16],
    left: &mut [i16],
    top_left: &mut i16,
    above_avail: &[bool],
    left_avail: &[bool],
    top_left_avail: bool,
    mid: i16,
) {
    let any_above = above_avail.iter().any(|&b| b);
    let any_left = left_avail.iter().any(|&b| b);
    // First available sample (scan order: top-left → above → left).
    let mut first: Option<i16> = None;
    if top_left_avail {
        first = Some(*top_left);
    } else if any_above {
        for (i, &av) in above_avail.iter().enumerate() {
            if av {
                first = Some(above[i]);
                break;
            }
        }
    } else if any_left {
        for (i, &av) in left_avail.iter().enumerate() {
            if av {
                first = Some(left[i]);
                break;
            }
        }
    }
    let fallback = first.unwrap_or(mid);
    if !top_left_avail {
        *top_left = fallback;
    }
    for (i, &av) in above_avail.iter().enumerate() {
        if !av {
            above[i] = fallback;
        }
    }
    for (i, &av) in left_avail.iter().enumerate() {
        if !av {
            left[i] = fallback;
        }
    }
}

/// Angular intra prediction — minimal subset (§8.4.5.2.13).
///
/// Only the five cardinal / diagonal modes are supported in this
/// foundation increment:
///
/// * Mode 2 — bottom-left diagonal.
/// * Mode 18 — pure horizontal.
/// * Mode 34 — top-left diagonal (45° down-right from the above
///   reference).
/// * Mode 50 — pure vertical.
/// * Mode 66 — top-right diagonal.
///
/// The general interpolation pipeline (wide-angle remap, reference
/// filter, 4-tap cubic / Gaussian filter) is not yet implemented —
/// this subset uses nearest-neighbour sampling so it can produce
/// plausible output for sanity tests.
pub fn predict_angular(
    n_tb_w: usize,
    n_tb_h: usize,
    mode: u32,
    refs: &IntraRefs<'_>,
) -> Result<Vec<i16>> {
    let w = n_tb_w;
    let h = n_tb_h;
    match mode {
        18 => {
            // Horizontal: each row takes its left-reference sample.
            if refs.left.len() < h {
                return Err(Error::invalid("h266 intra horizontal: left too short"));
            }
            let mut out = vec![0i16; w * h];
            for y in 0..h {
                let v = refs.left[y];
                for x in 0..w {
                    out[y * w + x] = v;
                }
            }
            Ok(out)
        }
        50 => {
            // Vertical: each column takes its above-reference sample.
            if refs.above.len() < w {
                return Err(Error::invalid("h266 intra vertical: above too short"));
            }
            let mut out = vec![0i16; w * h];
            for y in 0..h {
                for x in 0..w {
                    out[y * w + x] = refs.above[x];
                }
            }
            Ok(out)
        }
        2 => {
            // Bottom-left diagonal: pred[y][x] = left[y + x + 1] with
            // clamp to the last available left sample.
            if refs.left.len() < h {
                return Err(Error::invalid("h266 intra mode 2: left too short"));
            }
            let mut out = vec![0i16; w * h];
            let max_i = refs.left.len() - 1;
            for y in 0..h {
                for x in 0..w {
                    let idx = core::cmp::min(y + x + 1, max_i);
                    out[y * w + x] = refs.left[idx];
                }
            }
            Ok(out)
        }
        66 => {
            // Top-right diagonal: pred[y][x] = above[y + x + 1].
            if refs.above.len() < w {
                return Err(Error::invalid("h266 intra mode 66: above too short"));
            }
            let mut out = vec![0i16; w * h];
            let max_i = refs.above.len() - 1;
            for y in 0..h {
                for x in 0..w {
                    let idx = core::cmp::min(y + x + 1, max_i);
                    out[y * w + x] = refs.above[idx];
                }
            }
            Ok(out)
        }
        34 => {
            // Top-left diagonal: pred[y][x] selects from the main
            // reference running along the top-left diagonal —
            // above[x - y - 1] when x > y else left[y - x - 1]; the
            // corner sample `p[-1][-1]` covers x == y.
            let mut out = vec![0i16; w * h];
            for y in 0..h {
                for x in 0..w {
                    let v = if x > y {
                        refs.above[x - y - 1]
                    } else if y > x {
                        refs.left[y - x - 1]
                    } else {
                        refs.top_left
                    };
                    out[y * w + x] = v;
                }
            }
            Ok(out)
        }
        other => Err(Error::unsupported(format!(
            "h266 intra: angular mode {other} not in minimal subset (2/18/34/50/66)"
        ))),
    }
}

// ---------------------------------------------------------------------------
// Full §8.4.5.2 intra sample prediction pipeline (r412).
//
// The historical helpers above (`predict_planar` / `predict_dc` /
// `predict_angular` / `substitute_references`) are the foundation
// subset kept for source compatibility; the reconstruction paths now
// route through the spec-complete machinery below:
//
// * §8.4.5.2.7  wide-angle intra prediction mode mapping,
// * §8.4.5.2.8  reference sample availability marking (per-sample
//   §6.4.4 availability supplied by the caller as a fetch closure),
// * §8.4.5.2.9  reference sample substitution (spec scan order),
// * §8.4.5.2.10 reference sample filtering ([1 2 1]),
// * §8.4.5.2.11/.12 PLANAR / DC over the full reference set,
// * §8.4.5.2.13 general angular prediction (Table 24 angles, eq. 337
//   invAngle, Table 25 fC/fG 4-tap luma filters, 2-tap chroma),
// * §8.4.5.2.15 position-dependent prediction sample filtering (PDPC).
// ---------------------------------------------------------------------------

/// Table 24 — `intraPredAngle` for `predModeIntra ∈ −14..=80`
/// (index = mode + 14; entries at modes 0 / 1 are unused placeholders).
const INTRA_PRED_ANGLE: [i32; 95] = [
    512, 341, 256, 171, 128, 102, 86, 73, 64, 57, 51, 45, 39, 35, // -14..-1
    0, 0, // 0, 1 (PLANAR / DC — never angular)
    32, 29, 26, 23, 20, 18, 16, 14, 12, 10, 8, 6, 4, 3, 2, 1, // 2..17
    0, // 18
    -1, -2, -3, -4, -6, -8, -10, -12, -14, -16, -18, -20, -23, -26, -29, // 19..33
    -32, // 34
    -29, -26, -23, -20, -18, -16, -14, -12, -10, -8, -6, -4, -3, -2, -1, // 35..49
    0,  // 50
    1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 20, 23, 26, 29, // 51..65
    32, // 66
    35, 39, 45, 51, 57, 64, 73, 86, 102, 128, 171, 256, 341, 512, // 67..80
];

/// Table 24 lookup — `intraPredAngle` for a (possibly wide-angle
/// remapped) angular mode in `−14..=80`.
pub fn intra_pred_angle(mode: i32) -> Result<i32> {
    if !(-14..=80).contains(&mode) || mode == 0 || mode == 1 {
        return Err(Error::invalid(format!(
            "h266 intra: angular mode {mode} outside Table 24 range"
        )));
    }
    Ok(INTRA_PRED_ANGLE[(mode + 14) as usize])
}

/// Eq. 337 — `invAngle = Round( 512 * 32 / intraPredAngle )` with the
/// spec `Round( x ) = Sign( x ) * Floor( Abs( x ) + 0.5 )`.
pub fn inv_angle(intra_pred_angle: i32) -> i32 {
    debug_assert_ne!(intra_pred_angle, 0);
    let a = intra_pred_angle.unsigned_abs() as i64;
    let v = ((16384 * 2 + a) / (2 * a)) as i32;
    if intra_pred_angle < 0 {
        -v
    } else {
        v
    }
}

/// Table 25 — `fC[ phase ][ j ]` interpolation filter coefficients.
const INTRA_FC: [[i32; 4]; 32] = [
    [0, 64, 0, 0],
    [-1, 63, 2, 0],
    [-2, 62, 4, 0],
    [-2, 60, 7, -1],
    [-2, 58, 10, -2],
    [-3, 57, 12, -2],
    [-4, 56, 14, -2],
    [-4, 55, 15, -2],
    [-4, 54, 16, -2],
    [-5, 53, 18, -2],
    [-6, 52, 20, -2],
    [-6, 49, 24, -3],
    [-6, 46, 28, -4],
    [-5, 44, 29, -4],
    [-4, 42, 30, -4],
    [-4, 39, 33, -4],
    [-4, 36, 36, -4],
    [-4, 33, 39, -4],
    [-4, 30, 42, -4],
    [-4, 29, 44, -5],
    [-4, 28, 46, -6],
    [-3, 24, 49, -6],
    [-2, 20, 52, -6],
    [-2, 18, 53, -5],
    [-2, 16, 54, -4],
    [-2, 15, 55, -4],
    [-2, 14, 56, -4],
    [-2, 12, 57, -3],
    [-2, 10, 58, -2],
    [-1, 7, 60, -2],
    [0, 4, 62, -2],
    [0, 2, 63, -1],
];

/// Table 25 — `fG[ phase ][ j ]` interpolation filter coefficients.
const INTRA_FG: [[i32; 4]; 32] = [
    [16, 32, 16, 0],
    [16, 32, 16, 0],
    [15, 31, 17, 1],
    [15, 31, 17, 1],
    [14, 30, 18, 2],
    [14, 30, 18, 2],
    [13, 29, 19, 3],
    [13, 29, 19, 3],
    [12, 28, 20, 4],
    [12, 28, 20, 4],
    [11, 27, 21, 5],
    [11, 27, 21, 5],
    [10, 26, 22, 6],
    [10, 26, 22, 6],
    [9, 25, 23, 7],
    [9, 25, 23, 7],
    [8, 24, 24, 8],
    [8, 24, 24, 8],
    [7, 23, 25, 9],
    [7, 23, 25, 9],
    [6, 22, 26, 10],
    [6, 22, 26, 10],
    [5, 21, 27, 11],
    [5, 21, 27, 11],
    [4, 20, 28, 12],
    [4, 20, 28, 12],
    [3, 19, 29, 13],
    [3, 19, 29, 13],
    [2, 18, 30, 14],
    [2, 18, 30, 14],
    [1, 17, 31, 15],
    [1, 17, 31, 15],
];

/// §8.4.5.2.7 — wide angle intra prediction mode mapping.
///
/// `n_w` / `n_h` are the eq. 318 – 321 dimensions (the transform block
/// for `ISP_NO_SPLIT` or chroma, the coding block for ISP luma). Only
/// angular modes (`2..=66`) are remapped; PLANAR / DC pass through.
/// The output ranges over `−14..=80`.
pub fn wide_angle_remap(mode: u32, n_w: usize, n_h: usize) -> i32 {
    let m = mode as i32;
    if !(2..=66).contains(&m) || n_w == n_h {
        return m;
    }
    let wh_ratio = (n_w.trailing_zeros() as i32 - n_h.trailing_zeros() as i32).abs();
    if n_w > n_h {
        let bound = if wh_ratio > 1 { 8 + 2 * wh_ratio } else { 8 };
        if m >= 2 && m < bound {
            return m + 65;
        }
    } else {
        let bound = if wh_ratio > 1 { 60 - 2 * wh_ratio } else { 60 };
        if m <= 66 && m > bound {
            return m - 67;
        }
    }
    m
}

/// §8.4.5.2.6 — `refFilterFlag` derivation: 1 for the integer-slope
/// mode set `{0, −14, −12, −10, −6, 2, 34, 66, 72, 76, 78, 80}`.
pub fn ref_filter_flag(mode: i32) -> bool {
    matches!(
        mode,
        0 | -14 | -12 | -10 | -6 | 2 | 34 | 66 | 72 | 76 | 78 | 80
    )
}

/// Full §8.4.5.2.8 reference-sample bundle for one TB.
///
/// Spec layout: the left column `p[ −1 − refIdx ][ y ]` with
/// `y = −1 − refIdx..refH − 1` and the top row `p[ x ][ −1 − refIdx ]`
/// with `x = −refIdx..refW − 1`. Stored as
///
/// * `left[ i ] = p[ −1 − refIdx ][ −1 − refIdx + i ]`,
///   `i = 0..=refH + refIdx` (index 0 is the corner),
/// * `top[ j ] = p[ −refIdx + j ][ −1 − refIdx ]`,
///   `j = 0..refW + refIdx`.
#[derive(Clone, Debug)]
pub struct RefSamples {
    pub ref_idx: usize,
    pub ref_w: usize,
    pub ref_h: usize,
    left: Vec<i16>,
    top: Vec<i16>,
}

impl RefSamples {
    /// §8.4.5.2.8 + §8.4.5.2.9 — build the reference bundle.
    ///
    /// `fetch(dx, dy)` returns the reconstructed sample at the
    /// TB-relative position `(dx, dy)` when the covering block is
    /// available per §6.4.4 (already decoded, same picture / slice /
    /// tile), `None` otherwise. The substitution process fills every
    /// unavailable position in the spec scan order; a fully
    /// unavailable set becomes `1 << (BitDepth − 1)`.
    pub fn build<F>(fetch: F, ref_idx: usize, ref_w: usize, ref_h: usize, bit_depth: u32) -> Self
    where
        F: Fn(i32, i32) -> Option<i16>,
    {
        let r = ref_idx as i32;
        let left_len = ref_h + ref_idx + 1;
        let top_len = ref_w + ref_idx;
        let mut left: Vec<Option<i16>> = Vec::with_capacity(left_len);
        // left[i] = p[-1-refIdx][-1-refIdx + i]
        for i in 0..left_len {
            left.push(fetch(-1 - r, -1 - r + i as i32));
        }
        // top[j] = p[-refIdx + j][-1-refIdx]
        let mut top: Vec<Option<i16>> = Vec::with_capacity(top_len);
        for j in 0..top_len {
            top.push(fetch(-r + j as i32, -1 - r));
        }

        // §8.4.5.2.9 substitution.
        let any_avail = left.iter().chain(top.iter()).any(|s| s.is_some());
        let mid = 1i16 << (bit_depth - 1);
        let mut left_v = vec![mid; left_len];
        let mut top_v = vec![mid; top_len];
        if any_avail {
            // Step 1 — when p[-1-refIdx][refH-1] (left[left_len-1]) is
            // missing, scan from there up the left column to the
            // corner, then across the top row, for the first
            // available sample.
            if left[left_len - 1].is_none() {
                let mut found = None;
                for i in (0..left_len).rev() {
                    if let Some(v) = left[i] {
                        found = Some(v);
                        break;
                    }
                }
                if found.is_none() {
                    for j in 0..top_len {
                        if let Some(v) = top[j] {
                            found = Some(v);
                            break;
                        }
                    }
                }
                // `any_avail` guarantees a hit.
                left[left_len - 1] = found;
            }
            // Step 2 — fill the left column bottom-up from the sample
            // below (y from refH-2 down to -1-refIdx).
            left_v[left_len - 1] = left[left_len - 1].unwrap_or(mid);
            for i in (0..left_len - 1).rev() {
                left_v[i] = left[i].unwrap_or(left_v[i + 1]);
            }
            // Step 3 — fill the top row left-to-right; the leftmost
            // position substitutes from the corner p[-1-refIdx][-1-refIdx].
            let mut prev = left_v[0];
            for j in 0..top_len {
                top_v[j] = top[j].unwrap_or(prev);
                prev = top_v[j];
            }
        }
        Self {
            ref_idx,
            ref_w,
            ref_h,
            left: left_v,
            top: top_v,
        }
    }

    /// Spec-coordinate accessor `p[ x ][ y ]` over the L-shape.
    #[inline]
    pub fn p(&self, x: i32, y: i32) -> Result<i16> {
        let r = self.ref_idx as i32;
        if x == -1 - r {
            let i = y + 1 + r;
            if i >= 0 && (i as usize) < self.left.len() {
                return Ok(self.left[i as usize]);
            }
        } else if y == -1 - r {
            let j = x + r;
            if j >= 0 && (j as usize) < self.top.len() {
                return Ok(self.top[j as usize]);
            }
        }
        Err(Error::invalid(format!(
            "h266 intra refs: p[{x}][{y}] outside the reference L-shape \
             (refIdx {} refW {} refH {})",
            self.ref_idx, self.ref_w, self.ref_h
        )))
    }

    /// §8.4.5.2.10 — reference sample filtering. Returns the filtered
    /// array `p` when `filterFlag` holds (refIdx 0, `nTbW * nTbH > 32`,
    /// luma, `ISP_NO_SPLIT`, `refFilterFlag == 1`), otherwise a clone.
    pub fn filtered(
        &self,
        n_tb_w: usize,
        n_tb_h: usize,
        c_idx: u32,
        isp_no_split: bool,
        ref_filter_flag: bool,
    ) -> Self {
        let filter = self.ref_idx == 0
            && n_tb_w * n_tb_h > 32
            && c_idx == 0
            && isp_no_split
            && ref_filter_flag;
        if !filter {
            return self.clone();
        }
        let ref_h = self.ref_h;
        let ref_w = self.ref_w;
        let mut out = self.clone();
        // Eq. 325 — corner.
        out.left[0] =
            ((self.left[1] as i32 + 2 * self.left[0] as i32 + self.top[0] as i32 + 2) >> 2) as i16;
        // Eq. 326 — left column y = 0..refH-2 (left[i], i = y+1).
        for y in 0..ref_h.saturating_sub(1) {
            let i = y + 1;
            out.left[i] =
                ((self.left[i + 1] as i32 + 2 * self.left[i] as i32 + self.left[i - 1] as i32 + 2)
                    >> 2) as i16;
        }
        // Eq. 327 — bottom-most left sample copied.
        out.left[ref_h] = self.left[ref_h];
        // Eq. 328 — top row x = 0..refW-2; p[x-1][-1] at x == 0 is the
        // corner.
        for x in 0..ref_w.saturating_sub(1) {
            let pm1 = if x == 0 {
                self.left[0]
            } else {
                self.top[x - 1]
            };
            out.top[x] =
                ((pm1 as i32 + 2 * self.top[x] as i32 + self.top[x + 1] as i32 + 2) >> 2) as i16;
        }
        // Eq. 329 — right-most top sample copied.
        out.top[ref_w - 1] = self.top[ref_w - 1];
        out
    }
}

/// §8.4.5.2.13 — general angular intra prediction, all modes
/// `−14..=80` (post wide-angle remap), any `refIdx`, luma 4-tap
/// (Table 25) or chroma 2-tap interpolation.
///
/// `pred[ y * n_tb_w + x ]` row-major output.
#[allow(clippy::too_many_arguments)]
pub fn predict_angular_spec(
    mode: i32,
    ref_idx: usize,
    n_tb_w: usize,
    n_tb_h: usize,
    ref_w: usize,
    ref_h: usize,
    ref_filter_flag: bool,
    isp_no_split: bool,
    c_idx: u32,
    bit_depth: u32,
    refs: &RefSamples,
) -> Result<Vec<i16>> {
    let angle = intra_pred_angle(mode)?;
    let r = ref_idx as i32;
    let max_pix = (1i32 << bit_depth) - 1;

    // §8.4.5.2.13 filterFlag (interpolation filter selection — fG vs
    // fC), distinct from the §8.4.5.2.10 reference filter.
    let filter_flag = if ref_filter_flag || ref_idx != 0 || !isp_no_split {
        false
    } else {
        let n_tb_s = (n_tb_w.trailing_zeros() + n_tb_h.trailing_zeros()) >> 1;
        // Table 23 — intraHorVerDistThres[nTbS] for nTbS = 2..6.
        const THRES: [i32; 7] = [0, 0, 24, 14, 2, 0, 0];
        let min_dist_ver_hor = (mode - 50).abs().min((mode - 18).abs());
        min_dist_ver_hor > THRES[(n_tb_s as usize).min(6)]
    };

    let vertical = mode >= 34;
    // Swap roles for the horizontal family (mirrored derivation with
    // x ↔ y, nTbW ↔ nTbH, refW ↔ refH).
    let (n_main, n_side, ref_main) = if vertical {
        (n_tb_w, n_tb_h, ref_w)
    } else {
        (n_tb_h, n_tb_w, ref_h)
    };

    // Reference array ref[x] with x = -n_side .. ref_main + refIdx +
    // extras; stored with offset n_side.
    let extra = if angle >= 0 {
        (1usize.max(n_main / n_side)) * ref_idx + 2
    } else {
        0
    };
    let hi = ref_main + ref_idx + extra;
    let offset = n_side;
    let mut r_arr = vec![0i16; offset + hi + 1];
    // Main segment (eqs. 338 / 348): x = 0..=n_main + refIdx + 1.
    for x in 0..=(n_main + ref_idx + 1) {
        let v = if vertical {
            refs.p(-1 - r + x as i32, -1 - r)?
        } else {
            refs.p(-1 - r, -1 - r + x as i32)?
        };
        r_arr[offset + x] = v;
    }
    if angle < 0 {
        // Eqs. 339 / 349 — side-projected extension.
        let ia = inv_angle(angle);
        for x in -(n_side as i32)..0 {
            let proj = ((x * ia + 256) >> 9).min(n_side as i32);
            let v = if vertical {
                refs.p(-1 - r, -1 - r + proj)?
            } else {
                refs.p(-1 - r + proj, -1 - r)?
            };
            r_arr[(offset as i32 + x) as usize] = v;
        }
    } else {
        // Eqs. 340 / 350 — straight continuation.
        for x in (n_main + 2 + ref_idx)..=(ref_main + ref_idx) {
            let v = if vertical {
                refs.p(-1 - r + x as i32, -1 - r)?
            } else {
                refs.p(-1 - r, -1 - r + x as i32)?
            };
            r_arr[offset + x] = v;
        }
        // Eqs. 341 / 351 — clamp padding.
        let pad = if vertical {
            refs.p(ref_main as i32 - 1, -1 - r)?
        } else {
            refs.p(-1 - r, ref_main as i32 - 1)?
        };
        for x in 1..=extra {
            r_arr[offset + ref_main + ref_idx + x] = pad;
        }
    }

    let get = |idx: i32| -> Result<i16> {
        let k = offset as i32 + idx;
        if k < 0 || k as usize >= r_arr.len() {
            return Err(Error::invalid(format!(
                "h266 intra angular: ref[{idx}] outside the derived range \
                 (mode {mode} nTbW {n_tb_w} nTbH {n_tb_h} refIdx {ref_idx})"
            )));
        }
        Ok(r_arr[k as usize])
    };

    let mut pred = vec![0i16; n_tb_w * n_tb_h];
    for side in 0..n_side {
        // eqs. 342/343 (vertical: side = y) / eqs. 352/353 (horizontal:
        // side = x).
        let t = (side as i32 + 1 + r) * angle;
        let i_idx = (t >> 5) + r;
        let i_fact = (t & 31) as usize;
        for main in 0..n_main {
            let v = if c_idx == 0 {
                let ft = if filter_flag {
                    &INTRA_FG[i_fact]
                } else {
                    &INTRA_FC[i_fact]
                };
                let mut acc = 32i32;
                for (i, &c) in ft.iter().enumerate() {
                    acc += c * get(main as i32 + i_idx + i as i32)? as i32;
                }
                (acc >> 6).clamp(0, max_pix) as i16
            } else if i_fact != 0 {
                // Eqs. 346 / 356 — 2-tap chroma interpolation.
                let a = get(main as i32 + i_idx + 1)? as i32;
                let b = get(main as i32 + i_idx + 2)? as i32;
                (((32 - i_fact as i32) * a + i_fact as i32 * b + 16) >> 5) as i16
            } else {
                // Eqs. 347 / 357.
                get(main as i32 + i_idx + 1)?
            };
            let (x, y) = if vertical { (main, side) } else { (side, main) };
            pred[y * n_tb_w + x] = v;
        }
    }
    Ok(pred)
}

/// §8.4.5.2.6 — PDPC applicability (invocation conditions on the
/// wide-angle-remapped mode; the caller supplies the BDPCM gate).
pub fn pdpc_applies(mode: i32, n_tb_w: usize, n_tb_h: usize, ref_idx: usize, bdpcm: bool) -> bool {
    n_tb_w >= 4
        && n_tb_h >= 4
        && ref_idx == 0
        && !bdpcm
        && (mode == 0 || mode == 1 || mode <= 18 || (50..81).contains(&mode))
}

/// §8.4.5.2.15 — position-dependent intra prediction sample filtering.
///
/// `refs` is the (filtered) reference array with `ref_idx == 0`;
/// `mode` is the wide-angle-remapped intra mode. Modifies `pred`
/// in place.
pub fn apply_pdpc(
    mode: i32,
    n_tb_w: usize,
    n_tb_h: usize,
    refs: &RefSamples,
    pred: &mut [i16],
    bit_depth: u32,
) -> Result<()> {
    let max_pix = (1i32 << bit_depth) - 1;
    let log2_w = n_tb_w.trailing_zeros() as i32;
    let log2_h = n_tb_h.trailing_zeros() as i32;
    // nScale.
    let n_scale = if mode > 50 {
        let ia = inv_angle(intra_pred_angle(mode)?);
        let f = 31 - (3 * ia - 2).leading_zeros() as i32; // Floor(Log2(3*invAngle-2))
        2.min(log2_h - f + 8)
    } else if mode < 18 && mode != 0 && mode != 1 {
        let ia = inv_angle(intra_pred_angle(mode)?);
        let f = 31 - (3 * ia - 2).leading_zeros() as i32;
        2.min(log2_w - f + 8)
    } else {
        (log2_w + log2_h - 2) >> 2
    };

    for y in 0..n_tb_h {
        for x in 0..n_tb_w {
            let cur = pred[y * n_tb_w + x] as i32;
            let (ref_l, ref_t, w_l, w_t): (i32, i32, i32, i32) = if mode == 0 || mode == 1 {
                // Eqs. 406 – 409.
                (
                    refs.p(-1, y as i32)? as i32,
                    refs.p(x as i32, -1)? as i32,
                    32 >> (((x as i32) << 1) >> n_scale),
                    32 >> (((y as i32) << 1) >> n_scale),
                )
            } else if mode == 18 || mode == 50 {
                // Eqs. 410 – 413.
                let corner = refs.p(-1, -1)? as i32;
                (
                    refs.p(-1, y as i32)? as i32 - corner + cur,
                    refs.p(x as i32, -1)? as i32 - corner + cur,
                    if mode == 50 {
                        32 >> (((x as i32) << 1) >> n_scale)
                    } else {
                        0
                    },
                    if mode == 18 {
                        32 >> (((y as i32) << 1) >> n_scale)
                    } else {
                        0
                    },
                )
            } else if mode < 18 && n_scale >= 0 {
                // Eqs. 414 – 418 (mainRef = top row).
                let ia = inv_angle(intra_pred_angle(mode)?);
                let dx_int = ((y as i32 + 1) * ia + 256) >> 9;
                let dx = x as i32 + dx_int;
                let ref_t = if (y as i32) < (3 << n_scale) && dx < refs.ref_w as i32 {
                    refs.p(dx, -1)? as i32
                } else {
                    0
                };
                (0, ref_t, 0, 32 >> (((y as i32) << 1) >> n_scale))
            } else if mode > 50 && n_scale >= 0 {
                // Eqs. 419 – 423 (sideRef = left column).
                let ia = inv_angle(intra_pred_angle(mode)?);
                let dy_int = ((x as i32 + 1) * ia + 256) >> 9;
                let dy = y as i32 + dy_int;
                let ref_l = if (x as i32) < (3 << n_scale) && dy < refs.ref_h as i32 {
                    refs.p(-1, dy)? as i32
                } else {
                    0
                };
                (ref_l, 0, 32 >> (((x as i32) << 1) >> n_scale), 0)
            } else {
                (0, 0, 0, 0)
            };
            // Eq. 424.
            let v = (ref_l * w_l + ref_t * w_t + (64 - w_l - w_t) * cur + 32) >> 6;
            pred[y * n_tb_w + x] = v.clamp(0, max_pix) as i16;
        }
    }
    Ok(())
}

/// Inputs to the §8.4.5.2.6 general intra sample prediction
/// orchestrator [`predict_intra`].
#[derive(Clone, Copy, Debug)]
pub struct IntraPredParams {
    /// Parsed intra prediction mode (pre wide-angle remap): 0 = PLANAR,
    /// 1 = DC, 2..=66 angular. CCLM / MIP dispatch elsewhere.
    pub mode: u32,
    pub n_tb_w: usize,
    pub n_tb_h: usize,
    /// Coding-block dimensions (only consulted for ISP luma TBs).
    pub n_cb_w: usize,
    pub n_cb_h: usize,
    pub c_idx: u32,
    /// `IntraSubPartitionsSplitType == ISP_NO_SPLIT` (always true for
    /// chroma).
    pub isp_no_split: bool,
    /// §7.4.12.2 `IntraLumaRefLineIdx` (0 unless MRL; always 0 for
    /// chroma).
    pub ref_idx: usize,
    /// `BdpcmFlag` for the PDPC gate (BDPCM TBs bypass this process
    /// entirely in practice, but the gate is kept for completeness).
    pub bdpcm: bool,
    pub bit_depth: u32,
}

/// §8.4.5.2.6 — general intra sample prediction: reference generation
/// (availability marking via `fetch`, substitution, filtering), the
/// PLANAR / DC / angular dispatch, and PDPC.
///
/// `fetch(dx, dy)` returns the reconstructed sample at TB-relative
/// `(dx, dy)` when available per §6.4.4, else `None`.
pub fn predict_intra<F>(params: &IntraPredParams, fetch: F) -> Result<Vec<i16>>
where
    F: Fn(i32, i32) -> Option<i16>,
{
    let p = params;
    if p.n_tb_w == 0 || p.n_tb_h == 0 || !p.n_tb_w.is_power_of_two() || !p.n_tb_h.is_power_of_two()
    {
        return Err(Error::invalid(format!(
            "h266 intra: nTbW {} x nTbH {} must be non-zero powers of two",
            p.n_tb_w, p.n_tb_h
        )));
    }
    // Eqs. 313 – 316.
    let (ref_w, ref_h) = if p.isp_no_split || p.c_idx != 0 {
        (p.n_tb_w * 2, p.n_tb_h * 2)
    } else {
        (p.n_cb_w + p.n_tb_w, p.n_cb_h + p.n_tb_h)
    };
    // §8.4.5.2.7 — eqs. 318 – 321 dimensions feed the remap.
    let (n_w, n_h) = if p.isp_no_split || p.c_idx != 0 {
        (p.n_tb_w, p.n_tb_h)
    } else {
        (p.n_cb_w, p.n_cb_h)
    };
    let mode = wide_angle_remap(p.mode, n_w, n_h);
    let rff = ref_filter_flag(mode);

    // §8.4.5.2.8 / §8.4.5.2.9.
    let unfilt = RefSamples::build(fetch, p.ref_idx, ref_w, ref_h, p.bit_depth);
    // §8.4.5.2.10.
    let refs = unfilt.filtered(p.n_tb_w, p.n_tb_h, p.c_idx, p.isp_no_split, rff);

    let mut pred = match mode {
        0 => {
            // §8.4.5.2.11 PLANAR (refIdx is always 0 — §7.4.12.2 infers
            // intra_luma_not_planar_flag = 1 when intra_luma_ref_idx != 0).
            let above: Vec<i16> = (0..=p.n_tb_w as i32)
                .map(|x| refs.p(x, -1))
                .collect::<Result<_>>()?;
            let left: Vec<i16> = (0..=p.n_tb_h as i32)
                .map(|y| refs.p(-1, y))
                .collect::<Result<_>>()?;
            let view = IntraRefs {
                above: &above,
                left: &left,
                top_left: refs.p(-1, -1)?,
            };
            predict_planar(p.n_tb_w, p.n_tb_h, &view)?
        }
        1 => {
            // §8.4.5.2.12 DC over p[·][−1 − refIdx] / p[−1 − refIdx][·].
            let r = p.ref_idx as i32;
            let above: Vec<i16> = (0..p.n_tb_w as i32)
                .map(|x| refs.p(x, -1 - r))
                .collect::<Result<_>>()?;
            let left: Vec<i16> = (0..p.n_tb_h as i32)
                .map(|y| refs.p(-1 - r, y))
                .collect::<Result<_>>()?;
            let view = IntraRefs {
                above: &above,
                left: &left,
                top_left: refs.p(-1 - r, -1 - r)?,
            };
            predict_dc(p.n_tb_w, p.n_tb_h, &view)?
        }
        _ => predict_angular_spec(
            mode,
            p.ref_idx,
            p.n_tb_w,
            p.n_tb_h,
            ref_w,
            ref_h,
            rff,
            p.isp_no_split,
            p.c_idx,
            p.bit_depth,
            &refs,
        )?,
    };

    if pdpc_applies(mode, p.n_tb_w, p.n_tb_h, p.ref_idx, p.bdpcm) {
        apply_pdpc(mode, p.n_tb_w, p.n_tb_h, &refs, &mut pred, p.bit_depth)?;
    }
    Ok(pred)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// DC of a square block with constant neighbours = the constant.
    #[test]
    fn dc_constant_neighbours_square() {
        let above = vec![100i16; 5];
        let left = vec![100i16; 5];
        let refs = IntraRefs {
            above: &above,
            left: &left,
            top_left: 100,
        };
        let p = predict_dc(4, 4, &refs).unwrap();
        assert_eq!(p, vec![100; 16]);
    }

    /// DC of a 4x8 (height > width) block falls back to eq. 335 (the
    /// average over the left column only, + half rounding).
    #[test]
    fn dc_tall_block_uses_left_only() {
        let above = vec![10i16; 5];
        let left = vec![200i16; 9];
        let refs = IntraRefs {
            above: &above,
            left: &left,
            top_left: 0,
        };
        let p = predict_dc(4, 8, &refs).unwrap();
        // sum_h = 8 * 200 = 1600; dcVal = (1600 + 4) >> 3 = 1604 >> 3 = 200
        assert_eq!(p[0], 200);
    }

    /// DC of a 8x4 (width > height) block uses eq. 334 (above only).
    #[test]
    fn dc_wide_block_uses_above_only() {
        let above = vec![50i16; 9];
        let left = vec![0i16; 5];
        let refs = IntraRefs {
            above: &above,
            left: &left,
            top_left: 0,
        };
        let p = predict_dc(8, 4, &refs).unwrap();
        // sum_w = 400; dcVal = (400 + 4) >> 3 = 404 >> 3 = 50
        assert_eq!(p[0], 50);
    }

    /// Planar with uniform reference samples reproduces the constant.
    /// All above / left / corners equal to 128 → every predSample = 128.
    #[test]
    fn planar_constant_neighbours() {
        let above = vec![128i16; 5];
        let left = vec![128i16; 5];
        let refs = IntraRefs {
            above: &above,
            left: &left,
            top_left: 128,
        };
        let p = predict_planar(4, 4, &refs).unwrap();
        assert_eq!(p, vec![128; 16]);
    }

    /// Planar diagonal ramp: above and left both linearly increase; the
    /// prediction should interpolate between them. Spot-check that the
    /// corner samples match eqs. 330-332.
    #[test]
    fn planar_ramp_top_left_corner() {
        // above = [10, 20, 30, 40, 50], left = [10, 20, 30, 40, 50].
        let above: Vec<i16> = (1..=5).map(|i| i as i16 * 10).collect();
        let left = above.clone();
        let refs = IntraRefs {
            above: &above,
            left: &left,
            top_left: 10,
        };
        let p = predict_planar(4, 4, &refs).unwrap();
        // predSamples[0][0]:
        //   predV = (3 * 10 + 1 * 50) << 2 = 80 << 2 = 320
        //   predH = (3 * 10 + 1 * 50) << 2 = 320
        //   pred = (320 + 320 + 16) >> 5 = 656 >> 5 = 20
        assert_eq!(p[0], 20);
    }

    /// Reference-array-length validation.
    #[test]
    fn dc_rejects_short_references() {
        let above = vec![0i16; 2]; // too short for nTbW=4
        let left = vec![0i16; 4];
        let refs = IntraRefs {
            above: &above,
            left: &left,
            top_left: 0,
        };
        assert!(predict_dc(4, 4, &refs).is_err());
    }

    #[test]
    fn planar_rejects_short_references() {
        // Planar needs nTbW+1 above samples; pass only nTbW.
        let above = vec![0i16; 4];
        let left = vec![0i16; 5];
        let refs = IntraRefs {
            above: &above,
            left: &left,
            top_left: 0,
        };
        assert!(predict_planar(4, 4, &refs).is_err());
    }

    /// Mode 18 (horizontal): each row equals the left[y] value.
    #[test]
    fn angular_mode_18_horizontal_replicates_rows() {
        let above = vec![0i16; 5];
        let left = vec![10, 20, 30, 40, 50];
        let refs = IntraRefs {
            above: &above,
            left: &left,
            top_left: 0,
        };
        let p = predict_angular(4, 4, 18, &refs).unwrap();
        for y in 0..4 {
            for x in 0..4 {
                assert_eq!(p[y * 4 + x], left[y]);
            }
        }
    }

    /// Mode 50 (vertical): each column equals the above[x] value.
    #[test]
    fn angular_mode_50_vertical_replicates_cols() {
        let above = vec![10, 20, 30, 40, 50];
        let left = vec![0i16; 5];
        let refs = IntraRefs {
            above: &above,
            left: &left,
            top_left: 0,
        };
        let p = predict_angular(4, 4, 50, &refs).unwrap();
        for y in 0..4 {
            for x in 0..4 {
                assert_eq!(p[y * 4 + x], above[x]);
            }
        }
    }

    /// Mode 2 (bottom-left diagonal): pred[y][x] = left[y + x + 1].
    #[test]
    fn angular_mode_2_bottom_left_diagonal() {
        let above = vec![0i16; 5];
        let left: Vec<i16> = (0..9).map(|i| i * 10).collect();
        let refs = IntraRefs {
            above: &above,
            left: &left,
            top_left: 0,
        };
        let p = predict_angular(4, 4, 2, &refs).unwrap();
        assert_eq!(p[0], 10); // y=0, x=0 → left[1] = 10
        assert_eq!(p[3], 40); // y=0, x=3 → left[4] = 40
        assert_eq!(p[12], 40); // y=3, x=0 → left[4] = 40
    }

    /// Mode 66 (top-right diagonal): pred[y][x] = above[y + x + 1].
    #[test]
    fn angular_mode_66_top_right_diagonal() {
        let above: Vec<i16> = (0..9).map(|i| i * 10).collect();
        let left = vec![0i16; 5];
        let refs = IntraRefs {
            above: &above,
            left: &left,
            top_left: 0,
        };
        let p = predict_angular(4, 4, 66, &refs).unwrap();
        assert_eq!(p[0], 10);
        assert_eq!(p[3], 40);
        assert_eq!(p[12], 40);
    }

    /// Mode 34 (top-left diagonal): diagonal axis = top-left;
    /// x == y cells take the corner; x > y comes from above; x < y
    /// comes from left.
    #[test]
    fn angular_mode_34_top_left_diagonal() {
        let above: Vec<i16> = (1..=5).map(|i| i * 10).collect();
        let left: Vec<i16> = (1..=5).map(|i| i * 100).collect();
        let refs = IntraRefs {
            above: &above,
            left: &left,
            top_left: 7,
        };
        let p = predict_angular(4, 4, 34, &refs).unwrap();
        // Diagonal cells (x == y) all get the corner.
        assert_eq!(p[0 * 4 + 0], 7);
        assert_eq!(p[1 * 4 + 1], 7);
        // x > y → above[x - y - 1].
        assert_eq!(p[0 * 4 + 1], above[0]);
        assert_eq!(p[0 * 4 + 3], above[2]);
        // y > x → left[y - x - 1].
        assert_eq!(p[1 * 4 + 0], left[0]);
        assert_eq!(p[3 * 4 + 0], left[2]);
    }

    /// Reference substitution: all unavailable → fallback to mid-grey.
    #[test]
    fn substitute_all_unavailable_fallback_to_mid() {
        let mut above = [0i16; 5];
        let mut left = [0i16; 5];
        let mut top_left = 0i16;
        let above_avail = [false; 5];
        let left_avail = [false; 5];
        substitute_references(
            &mut above,
            &mut left,
            &mut top_left,
            &above_avail,
            &left_avail,
            false,
            512,
        );
        assert_eq!(top_left, 512);
        assert!(above.iter().all(|&v| v == 512));
        assert!(left.iter().all(|&v| v == 512));
    }

    /// Reference substitution: one available sample replicates.
    #[test]
    fn substitute_single_available_replicates() {
        let mut above = [0i16; 5];
        let mut left = [0i16; 5];
        let mut top_left = 0i16;
        above[2] = 100;
        let mut above_avail = [false; 5];
        above_avail[2] = true;
        let left_avail = [false; 5];
        substitute_references(
            &mut above,
            &mut left,
            &mut top_left,
            &above_avail,
            &left_avail,
            false,
            512,
        );
        assert_eq!(top_left, 100);
        assert_eq!(above[0], 100);
        assert_eq!(above[2], 100); // already-available sample kept.
        assert_eq!(above[4], 100);
        assert!(left.iter().all(|&v| v == 100));
    }

    // ---- r412: full §8.4.5.2 pipeline ---------------------------------

    /// Table 24 spot checks — cardinal / diagonal / wide-angle entries.
    #[test]
    fn table24_angles() {
        assert_eq!(intra_pred_angle(2).unwrap(), 32);
        assert_eq!(intra_pred_angle(18).unwrap(), 0);
        assert_eq!(intra_pred_angle(34).unwrap(), -32);
        assert_eq!(intra_pred_angle(50).unwrap(), 0);
        assert_eq!(intra_pred_angle(66).unwrap(), 32);
        assert_eq!(intra_pred_angle(-14).unwrap(), 512);
        assert_eq!(intra_pred_angle(-1).unwrap(), 35);
        assert_eq!(intra_pred_angle(35).unwrap(), -29);
        assert_eq!(intra_pred_angle(80).unwrap(), 512);
        assert_eq!(intra_pred_angle(63).unwrap(), 23);
        assert!(intra_pred_angle(0).is_err());
        assert!(intra_pred_angle(81).is_err());
    }

    /// Eq. 337 — invAngle = Round(16384 / angle).
    #[test]
    fn inv_angle_rounding() {
        assert_eq!(inv_angle(32), 512);
        assert_eq!(inv_angle(-32), -512);
        assert_eq!(inv_angle(512), 32);
        // 16384 / 3 = 5461.33 → 5461.
        assert_eq!(inv_angle(3), 5461);
        // 16384 / 45 = 364.09 → 364.
        assert_eq!(inv_angle(45), 364);
        // 16384 / 29 = 564.96 → 565.
        assert_eq!(inv_angle(29), 565);
    }

    /// §8.4.5.2.7 — square blocks pass through; wide blocks remap the
    /// low modes up (+65), tall blocks remap the high modes down (−67).
    #[test]
    fn wide_angle_remap_cases() {
        for m in 2..=66 {
            assert_eq!(wide_angle_remap(m, 8, 8), m as i32);
        }
        // 16x4: whRatio = 2 → bound = 12; modes 2..11 remap.
        assert_eq!(wide_angle_remap(2, 16, 4), 67);
        assert_eq!(wide_angle_remap(11, 16, 4), 76);
        assert_eq!(wide_angle_remap(12, 16, 4), 12);
        // 8x4: whRatio = 1 → bound = 8; modes 2..7 remap.
        assert_eq!(wide_angle_remap(7, 8, 4), 72);
        assert_eq!(wide_angle_remap(8, 8, 4), 8);
        // 4x16: whRatio = 2 → bound = 56; modes 57..66 remap.
        assert_eq!(wide_angle_remap(66, 4, 16), -1);
        assert_eq!(wide_angle_remap(57, 4, 16), -10);
        assert_eq!(wide_angle_remap(56, 4, 16), 56);
        // PLANAR / DC never remap.
        assert_eq!(wide_angle_remap(0, 16, 4), 0);
        assert_eq!(wide_angle_remap(1, 16, 4), 1);
    }

    /// §8.4.5.2.9 — all-unavailable references become mid-grey.
    #[test]
    fn ref_build_all_unavailable_mid_grey() {
        let refs = RefSamples::build(|_, _| None, 0, 8, 8, 8);
        for y in -1..8 {
            assert_eq!(refs.p(-1, y).unwrap(), 128);
        }
        for x in 0..8 {
            assert_eq!(refs.p(x, -1).unwrap(), 128);
        }
    }

    /// §8.4.5.2.9 — the spec scan starts at p[−1][refH−1]; a missing
    /// bottom-left run is filled from the first available sample
    /// scanning up the left column then across the top row.
    #[test]
    fn ref_build_substitution_scan_order() {
        // Left column available only for y = 0..3 (value 40); top row
        // available (value 90). refW = refH = 8 (4x4 TB).
        let refs = RefSamples::build(
            |x, y| {
                if x == -1 && (0..4).contains(&y) {
                    Some(40)
                } else if y == -1 && x >= 0 {
                    Some(90)
                } else {
                    None
                }
            },
            0,
            8,
            8,
            8,
        );
        // Bottom-left (y = 7) missing → step 1 scans up: first
        // available is y = 3 (40). Step 2 propagates 40 upward through
        // the missing y = 4..7 run — wait: step 2 fills top-down from
        // the sample *below*, so y = 6..4 copy from below (40), and
        // the corner (missing) copies from y = 0 (40).
        assert_eq!(refs.p(-1, 7).unwrap(), 40);
        assert_eq!(refs.p(-1, 5).unwrap(), 40);
        assert_eq!(refs.p(-1, 0).unwrap(), 40);
        assert_eq!(refs.p(-1, -1).unwrap(), 40); // corner from below
        assert_eq!(refs.p(0, -1).unwrap(), 90); // top row kept
        assert_eq!(refs.p(7, -1).unwrap(), 90);
    }

    /// §8.4.5.2.10 — the [1 2 1] filter fires only for refFilterFlag
    /// modes on >32-sample luma TBs.
    #[test]
    fn ref_filter_121() {
        // Step ramp on the top row; constant left column.
        let refs = RefSamples::build(
            |x, y| {
                if y == -1 && x >= 0 {
                    Some(((x as i16) % 2) * 100) // 0,100,0,100,...
                } else if x == -1 {
                    Some(50)
                } else {
                    None
                }
            },
            0,
            16,
            16,
            8,
        );
        let f = refs.filtered(8, 8, 0, true, true);
        // Interior top sample x = 1: (p[0] + 2*p[1] + p[2] + 2) >> 2 =
        // (0 + 200 + 0 + 2) >> 2 = 50.
        assert_eq!(f.p(1, -1).unwrap(), 50);
        // Corner: (p[-1][0] + 2*p[-1][-1] + p[0][-1] + 2) >> 2 =
        // (50 + 100 + 0 + 2) >> 2 = 38.
        assert_eq!(f.p(-1, -1).unwrap(), 38);
        // Right-most top sample copied unfiltered.
        assert_eq!(f.p(15, -1).unwrap(), refs.p(15, -1).unwrap());
        // Chroma / small-TB / non-refFilter modes: unchanged.
        let nf = refs.filtered(4, 4, 0, true, true);
        assert_eq!(nf.p(1, -1).unwrap(), 100);
        let nf2 = refs.filtered(8, 8, 1, true, true);
        assert_eq!(nf2.p(1, -1).unwrap(), 100);
        let nf3 = refs.filtered(8, 8, 0, true, false);
        assert_eq!(nf3.p(1, -1).unwrap(), 100);
    }

    /// §8.4.5.2.13 — integer-slope mode 66 reduces to the diagonal
    /// copy `pred[y][x] = p[x + y + 1][−1]` (fC[0] = {0,64,0,0}).
    #[test]
    fn angular_spec_mode66_matches_diagonal() {
        let top: Vec<i16> = (0..16).map(|i| (i * 10) as i16).collect();
        let refs = RefSamples::build(
            |x, y| {
                if y == -1 && x >= 0 {
                    Some(top[x as usize])
                } else if x == -1 {
                    Some(7)
                } else {
                    None
                }
            },
            0,
            8,
            8,
            8,
        );
        let p = predict_angular_spec(66, 0, 4, 4, 8, 8, true, true, 0, 8, &refs).unwrap();
        for y in 0..4usize {
            for x in 0..4usize {
                assert_eq!(p[y * 4 + x], top[x + y + 1], "({x},{y})");
            }
        }
    }

    /// §8.4.5.2.13 — mode 34 (angle −32) projects the left column onto
    /// the negative main-reference indices via invAngle = −512.
    #[test]
    fn angular_spec_mode34_projects_left() {
        let refs = RefSamples::build(
            |x, y| {
                if y == -1 && x >= 0 {
                    Some(10 * (x as i16 + 1))
                } else if x == -1 && y == -1 {
                    Some(7)
                } else if x == -1 {
                    Some(50 * (y as i16 + 1))
                } else {
                    None
                }
            },
            0,
            8,
            8,
            8,
        );
        let p = predict_angular_spec(34, 0, 4, 4, 8, 8, true, true, 0, 8, &refs).unwrap();
        // pred[y][x] = ref[x − y]; ref[0] = corner, ref[−k] = left[k−1],
        // ref[k] = top[k−1].
        assert_eq!(p[0], 7); // (0,0) → corner
        assert_eq!(p[1], 10); // (1,0) → top[0]
        assert_eq!(p[4], 50); // (0,1) → left[0]
        assert_eq!(p[4 * 3], 150); // (0,3) → left[2]
    }

    /// §8.4.5.2.13 — fractional luma interpolation through fC (mode 63,
    /// angle 23): hand-computed first sample.
    #[test]
    fn angular_spec_fractional_fc() {
        let refs = RefSamples::build(
            |x, y| {
                if y == -1 && x >= 0 {
                    Some(64)
                } else if x == -1 && y == -1 {
                    Some(0)
                } else if x == -1 {
                    Some(0)
                } else {
                    None
                }
            },
            0,
            8,
            8,
            8,
        );
        // 4x4, mode 63: nTbS = 2, minDistVerHor = 13 <= 24 → fC.
        // y = 0: iIdx = 0, iFact = 23; fC[23] = {−2, 18, 53, −5}.
        // pred[0][0] = (−2*p[−1][−1] + 18*top[0] + 53*top[1] − 5*top[2]
        //              + 32) >> 6 = (0 + 64*(18 + 53 − 5) + 32) >> 6 = 66.
        let p = predict_angular_spec(63, 0, 4, 4, 8, 8, false, true, 0, 8, &refs).unwrap();
        assert_eq!(p[0], 66);
    }

    /// §8.4.5.2.13 — chroma 2-tap interpolation (eq. 346).
    #[test]
    fn angular_spec_chroma_two_tap() {
        let refs = RefSamples::build(
            |x, y| {
                if y == -1 && x >= 0 {
                    Some(if x == 0 { 0 } else { 32 })
                } else if x == -1 {
                    Some(0)
                } else {
                    None
                }
            },
            0,
            8,
            8,
            8,
        );
        // mode 63 (angle 23), y = 0: iIdx = 0, iFact = 23.
        // pred[0][0] = ((32−23)*top[0] + 23*top[1] + 16) >> 5
        //            = (0 + 736 + 16) >> 5 = 23.
        let p = predict_angular_spec(63, 0, 4, 4, 8, 8, false, true, 1, 8, &refs).unwrap();
        assert_eq!(p[0], 23);
    }

    /// §8.4.5.2.15 — PDPC on a vertical (mode 50) prediction:
    /// hand-computed nScale-0 case.
    #[test]
    fn pdpc_vertical_hand_case() {
        let left = [10i16, 20, 30, 40];
        let refs = RefSamples::build(
            |x, y| {
                if x == -1 && y == -1 {
                    Some(50)
                } else if x == -1 && y >= 0 {
                    Some(left.get(y as usize).copied().unwrap_or(40))
                } else if y == -1 {
                    Some(100)
                } else {
                    None
                }
            },
            0,
            8,
            8,
            8,
        );
        let mut pred = vec![100i16; 16];
        // 4x4 mode 50: nScale = (2 + 2 − 2) >> 2 = 0; wL[x] = 32 >> 2x.
        // (0,0): refL = 10 − 50 + 100 = 60, wL = 32 →
        //        (60*32 + 32*100 + 32) >> 6 = 80.
        apply_pdpc(50, 4, 4, &refs, &mut pred, 8).unwrap();
        assert_eq!(pred[0], 80);
        // (3,0): wL = 32 >> 6 = 0 → unchanged.
        assert_eq!(pred[3], 100);
        // (0,1): refL = 20 − 50 + 100 = 70 → (70*32 + 3200 + 32) >> 6 = 85.
        assert_eq!(pred[4], 85);
    }

    /// §8.4.5.2.15 — PDPC leaves a uniform DC prediction with uniform
    /// references untouched.
    #[test]
    fn pdpc_dc_uniform_fixed_point() {
        let refs = RefSamples::build(|_, _| Some(100), 0, 8, 8, 8);
        let mut pred = vec![100i16; 16];
        apply_pdpc(1, 4, 4, &refs, &mut pred, 8).unwrap();
        assert!(pred.iter().all(|&v| v == 100));
    }

    /// §8.4.5.2.6 gate — PDPC needs a ≥4×4 TB, refIdx 0, no BDPCM, and
    /// the PLANAR / DC / ≤18 / ≥50 mode classes.
    #[test]
    fn pdpc_applicability_gate() {
        assert!(pdpc_applies(0, 4, 4, 0, false));
        assert!(pdpc_applies(1, 8, 8, 0, false));
        assert!(pdpc_applies(18, 4, 4, 0, false));
        assert!(pdpc_applies(-3, 4, 4, 0, false));
        assert!(pdpc_applies(50, 4, 4, 0, false));
        assert!(pdpc_applies(80, 4, 4, 0, false));
        assert!(!pdpc_applies(30, 4, 4, 0, false)); // diagonal class
        assert!(!pdpc_applies(1, 2, 4, 0, false)); // too narrow
        assert!(!pdpc_applies(1, 4, 4, 1, false)); // MRL
        assert!(!pdpc_applies(1, 4, 4, 0, true)); // BDPCM
        assert!(!pdpc_applies(81, 4, 4, 0, false)); // CCLM
    }

    /// §8.4.5.2.6 orchestrator — a DC TB with only the top row decoded
    /// substitutes the left column from the top and lands on the top
    /// average.
    #[test]
    fn predict_intra_dc_left_unavailable() {
        let params = IntraPredParams {
            mode: 1,
            n_tb_w: 4,
            n_tb_h: 4,
            n_cb_w: 4,
            n_cb_h: 4,
            c_idx: 0,
            isp_no_split: true,
            ref_idx: 0,
            bdpcm: false,
            bit_depth: 8,
        };
        let pred = predict_intra(
            &params,
            |x, y| {
                if y == -1 && x >= 0 {
                    Some(80)
                } else {
                    None
                }
            },
        )
        .unwrap();
        // DC = 80 everywhere; PDPC over uniform refs keeps it.
        assert!(pred.iter().all(|&v| v == 80));
    }

    /// §8.4.5.2.6 orchestrator — MRL (refIdx 2) DC reads the offset
    /// reference line and skips PDPC.
    #[test]
    fn predict_intra_dc_mrl_line() {
        let params = IntraPredParams {
            mode: 1,
            n_tb_w: 4,
            n_tb_h: 4,
            n_cb_w: 4,
            n_cb_h: 4,
            c_idx: 0,
            isp_no_split: true,
            ref_idx: 2,
            bdpcm: false,
            bit_depth: 8,
        };
        let pred = predict_intra(&params, |x, y| {
            if y == -3 && x >= -2 {
                Some(60) // reference line 2 (top)
            } else if y == -1 && x >= 0 {
                Some(200) // line 0 must NOT be read for DC refs
            } else if x == -3 {
                Some(60)
            } else if x == -1 {
                Some(200)
            } else {
                None
            }
        })
        .unwrap();
        assert!(pred.iter().all(|&v| v == 60));
    }

    /// Unsupported angular mode surfaces an error.
    #[test]
    fn unsupported_angular_mode_errors() {
        let above = vec![0i16; 5];
        let left = vec![0i16; 5];
        let refs = IntraRefs {
            above: &above,
            left: &left,
            top_left: 0,
        };
        assert!(predict_angular(4, 4, 10, &refs).is_err());
    }
}
