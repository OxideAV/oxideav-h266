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
pub fn predict_planar(
    n_tb_w: usize,
    n_tb_h: usize,
    refs: &IntraRefs<'_>,
) -> Result<Vec<i16>> {
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
            let pred_v = ((n_tb_h as i32 - 1 - y as i32) * p_a + (y as i32 + 1) * p_bl)
                << log2_w;
            let pred_h = ((n_tb_w as i32 - 1 - x as i32) * p_l + (x as i32 + 1) * p_tr)
                << log2_h;
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
pub fn predict_dc(
    n_tb_w: usize,
    n_tb_h: usize,
    refs: &IntraRefs<'_>,
) -> Result<Vec<i16>> {
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
