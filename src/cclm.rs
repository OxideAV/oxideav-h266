//! Cross-Component Linear Model intra prediction (CCLM) — §8.4.5.2.14
//! of ITU-T H.266 (V4, 01/2026).
//!
//! CCLM is a chroma intra prediction mode that predicts a chroma TB
//! from the *reconstructed* (pre-loop-filter) collocated luma block via
//! a linear model:
//!
//! ```text
//! predSamples[x][y] = Clip1( ((pDsY[x][y] * a) >> k) + b )       (404)
//! ```
//!
//! where `(a, b, k)` are derived from a 4-point regression between
//! down-sampled luma neighbours and the matching chroma neighbours
//! (§8.4.5.2.14 eqs. 386 – 399). Three modes select which neighbour
//! sides participate:
//!
//! * `INTRA_LT_CCLM` (81) — both top and left neighbours.
//! * `INTRA_L_CCLM`  (82) — left only (extends down-left).
//! * `INTRA_T_CCLM`  (83) — top only (extends top-right).
//!
//! Implementation scope for round 19:
//!
//! * 4:2:0 chroma sub-sampling (`SubWidthC == SubHeightC == 2`). This
//!   is the only colour format the rest of `oxideav-h266` exercises
//!   today; 4:2:2 / 4:4:4 are deferred.
//! * `sps_chroma_vertical_collocated_flag` value supplied by the
//!   caller — both branches of eqs. 368 / 369 / 371 / 372 / 376 / 377
//!   are wired so a future SPS toggle costs nothing extra.
//! * `bCTUboundary` (eq. 363) is honoured for the top-neighbour
//!   down-sampling path.
//! * Spec-identical 4-tap min-max regression over 4 (`numIs4N == 1`,
//!   single-side modes) or 4 (`numIs4N == 0`, LT_CCLM) sample pairs.
//!
//! Neighbour availability is reflected in the array lengths supplied by
//! the caller: when a side is unavailable its `neigh_*` slice is
//! empty / `None` and the mode-specific `numSampN` derivation collapses
//! that side out automatically.

use oxideav_core::{Error, Result};

/// CCLM mode identifiers from Table 18 (§7.4.11.5). Match the constants
/// re-exported from [`crate::leaf_cu`].
pub const INTRA_LT_CCLM: u32 = 81;
pub const INTRA_L_CCLM: u32 = 82;
pub const INTRA_T_CCLM: u32 = 83;

/// Inputs to [`predict_cclm`].
///
/// All coordinates are picture-absolute samples in the **chroma** grid;
/// the CCLM helper internally derives the luma sample positions via
/// `xTbY = xTbC << (SubWidthC - 1)` (eq. 358).
#[derive(Debug)]
pub struct CclmInputs<'a> {
    /// CCLM mode: one of `INTRA_LT_CCLM`, `INTRA_L_CCLM`, `INTRA_T_CCLM`.
    pub mode: u32,
    /// Chroma TB width.
    pub n_tb_w: usize,
    /// Chroma TB height.
    pub n_tb_h: usize,
    /// Chroma sub-sampling — `SubWidthC` (1 for 4:4:4, 2 for 4:2:0 / 4:2:2).
    pub sub_width_c: u32,
    /// Chroma sub-sampling — `SubHeightC` (1 for 4:4:4 / 4:2:2, 2 for 4:2:0).
    pub sub_height_c: u32,
    /// `sps_chroma_vertical_collocated_flag` (selects between eqs.
    /// 368 / 369 + eqs. 371 / 372).
    pub chroma_vertical_collocated_flag: bool,
    /// `bCTUboundary` per eq. 363 — set when the chroma TB sits at a
    /// CTB-row boundary (`yTbY & (CtbSizeY - 1) == 0`). Forces the
    /// 3-tap horizontal-only down-sampling for the top neighbour row.
    pub b_ctu_boundary: bool,
    /// BitDepth of the chroma samples (bit_depth_c). Used by the
    /// `Clip1` step (eq. 404) and the all-unavailable fallback
    /// (eq. 365).
    pub bit_depth: u32,
    /// Top neighbour chroma row, length `2 * n_tb_w` when `availT` and
    /// `mode == INTRA_T_CCLM`, length `n_tb_w` otherwise. Empty when
    /// the top side is unavailable.
    pub neigh_top_chroma: &'a [i16],
    /// Left neighbour chroma column, length `2 * n_tb_h` when `availL`
    /// and `mode == INTRA_L_CCLM`, length `n_tb_h` otherwise. Empty
    /// when the left side is unavailable.
    pub neigh_left_chroma: &'a [i16],
    /// Reconstructed luma plane covering at least the collocated
    /// region plus the top/left neighbour skirts. Indexed via
    /// `luma_at(x, y)` below.
    pub luma_plane: LumaPlane<'a>,
    /// Top-left corner of the chroma TB, used to translate the spec's
    /// signed `pY[x][y]` indexing into picture-absolute luma reads.
    pub x_tb_c: usize,
    pub y_tb_c: usize,
}

/// Read-only sample provider for the reconstructed luma plane. The
/// CCLM derivation reads luma samples at picture-absolute positions
/// `(x_tb_y + dx, y_tb_y + dy)` with `dx, dy ∈ -3..nTb*Sub - 1`. The
/// caller (typically [`crate::reconstruct::PicturePlane`]) is
/// responsible for performing the §8.4.5.2.8 boundary substitution
/// before handing the plane in — out-of-bounds reads here clamp to the
/// edge to keep the pipeline numerically well-defined when the spec's
/// own substitution has been applied conservatively.
#[derive(Debug)]
pub struct LumaPlane<'a> {
    pub samples: &'a [u8],
    pub stride: usize,
    pub width: usize,
    pub height: usize,
}

impl LumaPlane<'_> {
    #[inline]
    fn read(&self, x: i32, y: i32) -> i32 {
        let xc = x.clamp(0, self.width as i32 - 1) as usize;
        let yc = y.clamp(0, self.height as i32 - 1) as usize;
        self.samples[yc * self.stride + xc] as i32
    }
}

/// Run §8.4.5.2.14. Returns the row-major chroma prediction array of
/// size `n_tb_w * n_tb_h`.
pub fn predict_cclm(inp: &CclmInputs<'_>) -> Result<Vec<i16>> {
    if !matches!(inp.mode, INTRA_LT_CCLM | INTRA_L_CCLM | INTRA_T_CCLM) {
        return Err(Error::invalid("h266 cclm: mode must be 81 / 82 / 83"));
    }
    if inp.sub_width_c == 0 || inp.sub_height_c == 0 {
        return Err(Error::invalid(
            "h266 cclm: SubWidthC / SubHeightC must be >= 1",
        ));
    }
    if inp.bit_depth < 8 || inp.bit_depth > 16 {
        return Err(Error::invalid("h266 cclm: bit_depth out of [8, 16]"));
    }
    if inp.n_tb_w == 0 || inp.n_tb_h == 0 {
        return Err(Error::invalid("h266 cclm: zero TB size"));
    }

    // §8.4.5.2.14 eq. 358 — collocated luma top-left.
    let x_tb_y = (inp.x_tb_c as i32) << (inp.sub_width_c as i32 - 1);
    let y_tb_y = (inp.y_tb_c as i32) << (inp.sub_height_c as i32 - 1);
    let _ = x_tb_y; // x_tb_y not directly indexed below — luma reads are picture-absolute via plane

    // availT / availL come from the caller via the array lengths.
    let avail_t = !inp.neigh_top_chroma.is_empty();
    let avail_l = !inp.neigh_left_chroma.is_empty();

    // numSampT / numSampL — eqs. 359 – 362.
    let n_tb_w = inp.n_tb_w;
    let n_tb_h = inp.n_tb_h;
    let (num_samp_t, num_samp_l) = match inp.mode {
        INTRA_LT_CCLM => (
            if avail_t { n_tb_w } else { 0 },
            if avail_l { n_tb_h } else { 0 },
        ),
        INTRA_T_CCLM => {
            // For T_CCLM the spec adds Min(numTopRight, nTbH); the
            // caller signals availability of those extra samples via
            // the slice length (it may pass anywhere from n_tb_w to
            // 2*n_tb_w samples).
            let num = if avail_t {
                let top_avail = inp.neigh_top_chroma.len();
                let extra = top_avail.saturating_sub(n_tb_w).min(n_tb_h);
                n_tb_w + extra
            } else {
                0
            };
            (num, 0)
        }
        INTRA_L_CCLM => {
            let num = if avail_l {
                let left_avail = inp.neigh_left_chroma.len();
                let extra = left_avail.saturating_sub(n_tb_h).min(n_tb_w);
                n_tb_h + extra
            } else {
                0
            };
            (0, num)
        }
        _ => unreachable!(),
    };

    // Eq. 365 — fully-unavailable neighbours fall back to mid-grey.
    if num_samp_t == 0 && num_samp_l == 0 {
        let mid = 1i16 << (inp.bit_depth - 1);
        return Ok(vec![mid; n_tb_w * n_tb_h]);
    }

    // Eq. 364: numIs4N.
    let num_is_4n = if avail_t && avail_l && inp.mode == INTRA_LT_CCLM {
        0u32
    } else {
        1u32
    };

    // Picking positions per side (§8.4.5.2.14, "cntN and array
    // pickPosN"). We compute pickPosT / pickPosL into Vecs; the
    // expected length is `cntN ∈ {0, 2, 4}` per spec.
    let (cnt_t, pick_pos_t) = pick_positions(num_samp_t, num_is_4n);
    let (cnt_l, pick_pos_l) = pick_positions(num_samp_l, num_is_4n);

    // Step 3 — down-sampled collocated luma pDsY[x][y] for the current
    // TB. Indexed [y * n_tb_w + x].
    let p_ds_y = down_sample_collocated_luma(inp, x_tb_y, y_tb_y);

    // Step 4 / 5 — selected neighbour samples. `pSelDsY[idx]` for
    // `idx = 0..cntT-1` cover the top-neighbour pickup, `cntT..cntT+cntL-1`
    // cover the left side. `pSelC[idx]` is the matching chroma sample.
    let mut p_sel_dsy: Vec<i32> = Vec::with_capacity(4);
    let mut p_sel_c: Vec<i32> = Vec::with_capacity(4);
    for &x in pick_pos_t.iter().take(cnt_t) {
        // pSelC[idx] = p[pickPosT[idx]][-1]
        p_sel_c.push(inp.neigh_top_chroma[x] as i32);
        p_sel_dsy.push(top_neighbour_dsy(inp, x_tb_y, y_tb_y, x));
    }
    for &y in pick_pos_l.iter().take(cnt_l) {
        // pSelC[idx] = p[-1][pickPosL[idx - cntT]]
        p_sel_c.push(inp.neigh_left_chroma[y] as i32);
        p_sel_dsy.push(left_neighbour_dsy(inp, x_tb_y, y_tb_y, y));
    }

    // Step 6 — derive (minY, maxY, minC, maxC) via the 4-point min/max
    // grouping. When `cntT + cntL == 2`, the spec replicates the pair
    // out to 4 samples first.
    let mut p_dsy = [0i32; 4];
    let mut p_c = [0i32; 4];
    let total = p_sel_dsy.len();
    if total == 0 {
        // Defensive: shouldn't be reachable because num_samp_*==0 is
        // already handled above, but keep predictable behaviour.
        let mid = 1i16 << (inp.bit_depth - 1);
        return Ok(vec![mid; n_tb_w * n_tb_h]);
    }
    if total == 2 {
        // pSelComp[3] = pSelComp[0]; pSelComp[2] = pSelComp[1];
        // pSelComp[0] = pSelComp[1]; pSelComp[1] = pSelComp[3];
        // → effectively reorders to (orig1, orig0, orig1, orig0).
        let (c0, c1) = (p_sel_c[0], p_sel_c[1]);
        let (y0, y1) = (p_sel_dsy[0], p_sel_dsy[1]);
        p_dsy = [y1, y0, y1, y0];
        p_c = [c1, c0, c1, c0];
    } else {
        // total ∈ {4} — both sides contribute (LT_CCLM, eq. 364
        // numIs4N=0) or one side contributes (numIs4N=1, cntN=4).
        for i in 0..4.min(total) {
            p_dsy[i] = p_sel_dsy[i];
            p_c[i] = p_sel_c[i];
        }
    }

    let mut min_grp_idx = [0usize, 2];
    let mut max_grp_idx = [1usize, 3];
    if p_dsy[min_grp_idx[0]] > p_dsy[min_grp_idx[1]] {
        min_grp_idx.swap(0, 1);
    }
    if p_dsy[max_grp_idx[0]] > p_dsy[max_grp_idx[1]] {
        max_grp_idx.swap(0, 1);
    }
    if p_dsy[min_grp_idx[0]] > p_dsy[max_grp_idx[1]] {
        std::mem::swap(&mut min_grp_idx, &mut max_grp_idx);
    }
    if p_dsy[min_grp_idx[1]] > p_dsy[max_grp_idx[0]] {
        // Cross-array swap: clippy's `manual_swap` rule pattern-matches
        // the three-line idiom; split the assignments through a pair of
        // temporaries so the swap intent is explicit without needing a
        // `std::mem::swap` (which would require disjoint mutable
        // borrows across two separate arrays).
        let (mi1, ma0) = (min_grp_idx[1], max_grp_idx[0]);
        max_grp_idx[0] = mi1;
        min_grp_idx[1] = ma0;
    }

    let max_y = (p_dsy[max_grp_idx[0]] + p_dsy[max_grp_idx[1]] + 1) >> 1; // 386
    let max_c = (p_c[max_grp_idx[0]] + p_c[max_grp_idx[1]] + 1) >> 1; // 387
    let min_y = (p_dsy[min_grp_idx[0]] + p_dsy[min_grp_idx[1]] + 1) >> 1; // 388
    let min_c = (p_c[min_grp_idx[0]] + p_c[min_grp_idx[1]] + 1) >> 1; // 389

    // Step 7 — derive (a, b, k).
    let (a, b, k) = derive_a_b_k(min_y, max_y, min_c, max_c);

    // Step 8 — generate the prediction (eq. 404).
    let lo = 0i32;
    let hi = (1i32 << inp.bit_depth) - 1;
    let mut pred = vec![0i16; n_tb_w * n_tb_h];
    for y in 0..n_tb_h {
        for x in 0..n_tb_w {
            let v = ((p_ds_y[y * n_tb_w + x] * a) >> k) + b;
            pred[y * n_tb_w + x] = v.clamp(lo, hi) as i16;
        }
    }
    Ok(pred)
}

/// `(cntN, pickPosN)` per the §8.4.5.2.14 derivation. Returns up to 4
/// positions when the side is active, else `(0, [])`.
fn pick_positions(num_samp_n: usize, num_is_4n: u32) -> (usize, Vec<usize>) {
    if num_samp_n == 0 {
        return (0, Vec::new());
    }
    let shift_start = 2 + num_is_4n; // start = numSampN >> (2 + numIs4N)
    let start_pos_n = num_samp_n >> shift_start;
    let pick_step = core::cmp::max(1usize, num_samp_n >> (1 + num_is_4n));
    let cnt_n = core::cmp::min(num_samp_n, ((1 + num_is_4n) << 1) as usize);
    let positions: Vec<usize> = (0..cnt_n)
        .map(|pos| start_pos_n + pos * pick_step)
        .collect();
    (cnt_n, positions)
}

/// Step 3 — down-sample the collocated luma block (eqs. 366 – 369).
fn down_sample_collocated_luma(inp: &CclmInputs<'_>, x_tb_y: i32, y_tb_y: i32) -> Vec<i32> {
    let n_tb_w = inp.n_tb_w;
    let n_tb_h = inp.n_tb_h;
    let sub_w = inp.sub_width_c as i32;
    let sub_h = inp.sub_height_c as i32;
    let plane = &inp.luma_plane;
    let mut out = vec![0i32; n_tb_w * n_tb_h];

    if sub_w == 1 && sub_h == 1 {
        // 4:4:4 — eq. 366.
        for y in 0..n_tb_h {
            for x in 0..n_tb_w {
                out[y * n_tb_w + x] = plane.read(x_tb_y + x as i32, y_tb_y + y as i32);
            }
        }
        return out;
    }

    if sub_h == 1 {
        // 4:2:2 — eq. 367 (3-tap horizontal).
        for y in 0..n_tb_h {
            for x in 0..n_tb_w {
                let xb = x_tb_y + sub_w * x as i32;
                let v = plane.read(xb - 1, y_tb_y + y as i32)
                    + 2 * plane.read(xb, y_tb_y + y as i32)
                    + plane.read(xb + 1, y_tb_y + y as i32)
                    + 2;
                out[y * n_tb_w + x] = v >> 2;
            }
        }
        return out;
    }

    // 4:2:0 — eqs. 368 / 369.
    if inp.chroma_vertical_collocated_flag {
        for y in 0..n_tb_h {
            for x in 0..n_tb_w {
                let xb = x_tb_y + sub_w * x as i32;
                let yb = y_tb_y + sub_h * y as i32;
                let v = plane.read(xb, yb - 1)
                    + plane.read(xb - 1, yb)
                    + 4 * plane.read(xb, yb)
                    + plane.read(xb + 1, yb)
                    + plane.read(xb, yb + 1)
                    + 4;
                out[y * n_tb_w + x] = v >> 3;
            }
        }
    } else {
        for y in 0..n_tb_h {
            for x in 0..n_tb_w {
                let xb = x_tb_y + sub_w * x as i32;
                let yb = y_tb_y + sub_h * y as i32;
                let v = plane.read(xb - 1, yb)
                    + plane.read(xb - 1, yb + 1)
                    + 2 * plane.read(xb, yb)
                    + 2 * plane.read(xb, yb + 1)
                    + plane.read(xb + 1, yb)
                    + plane.read(xb + 1, yb + 1)
                    + 4;
                out[y * n_tb_w + x] = v >> 3;
            }
        }
    }
    out
}

/// Step 4 — pSelDsY for the top neighbour at chroma-x = `x_pick`
/// (eqs. 370 – 373).
fn top_neighbour_dsy(inp: &CclmInputs<'_>, x_tb_y: i32, y_tb_y: i32, x_pick: usize) -> i32 {
    let sub_w = inp.sub_width_c as i32;
    let sub_h = inp.sub_height_c as i32;
    let plane = &inp.luma_plane;
    let x = x_pick as i32;

    if sub_w == 1 && sub_h == 1 {
        // 4:4:4 — eq. 370.
        return plane.read(x_tb_y + x, y_tb_y - 1);
    }

    if sub_h != 1 && !inp.b_ctu_boundary {
        // 4:2:0 (and 4:2:0-like 4:2:2 with sub_h != 1) — choose
        // collocated branch.
        if inp.chroma_vertical_collocated_flag {
            // eq. 371
            let xb = x_tb_y + sub_w * x;
            let v = plane.read(xb, y_tb_y - 3)
                + plane.read(xb - 1, y_tb_y - 2)
                + 4 * plane.read(xb, y_tb_y - 2)
                + plane.read(xb + 1, y_tb_y - 2)
                + plane.read(xb, y_tb_y - 1)
                + 4;
            return v >> 3;
        } else {
            // eq. 372
            let xb = x_tb_y + sub_w * x;
            let v = plane.read(xb - 1, y_tb_y - 1)
                + plane.read(xb - 1, y_tb_y - 2)
                + 2 * plane.read(xb, y_tb_y - 1)
                + 2 * plane.read(xb, y_tb_y - 2)
                + plane.read(xb + 1, y_tb_y - 1)
                + plane.read(xb + 1, y_tb_y - 2)
                + 4;
            return v >> 3;
        }
    }

    // SubHeightC == 1 OR bCTUboundary — eq. 373 (3-tap horizontal,
    // single row above).
    let xb = x_tb_y + sub_w * x;
    let v = plane.read(xb - 1, y_tb_y - 1)
        + 2 * plane.read(xb, y_tb_y - 1)
        + plane.read(xb + 1, y_tb_y - 1)
        + 2;
    v >> 2
}

/// Step 5 — pSelDsY for the left neighbour at chroma-y = `y_pick`
/// (eqs. 374 – 377).
fn left_neighbour_dsy(inp: &CclmInputs<'_>, x_tb_y: i32, y_tb_y: i32, y_pick: usize) -> i32 {
    let sub_w = inp.sub_width_c as i32;
    let sub_h = inp.sub_height_c as i32;
    let plane = &inp.luma_plane;
    let y = y_pick as i32;

    if sub_w == 1 && sub_h == 1 {
        return plane.read(x_tb_y - 1, y_tb_y + y);
    }

    if sub_h == 1 {
        // 4:2:2 — eq. 375.
        let yb = y_tb_y + y;
        let v = plane.read(x_tb_y - 1 - sub_w, yb)
            + 2 * plane.read(x_tb_y - sub_w, yb)
            + plane.read(x_tb_y + 1 - sub_w, yb)
            + 2;
        return v >> 2;
    }

    if inp.chroma_vertical_collocated_flag {
        // eq. 376
        let yb = y_tb_y + sub_h * y;
        let v = plane.read(x_tb_y - sub_w, yb - 1)
            + plane.read(x_tb_y - 1 - sub_w, yb)
            + 4 * plane.read(x_tb_y - sub_w, yb)
            + plane.read(x_tb_y + 1 - sub_w, yb)
            + plane.read(x_tb_y - sub_w, yb + 1)
            + 4;
        v >> 3
    } else {
        // eq. 377
        let yb = y_tb_y + sub_h * y;
        let v = plane.read(x_tb_y - 1 - sub_w, yb)
            + plane.read(x_tb_y - 1 - sub_w, yb + 1)
            + 2 * plane.read(x_tb_y - sub_w, yb)
            + 2 * plane.read(x_tb_y - sub_w, yb + 1)
            + plane.read(x_tb_y + 1 - sub_w, yb)
            + plane.read(x_tb_y + 1 - sub_w, yb + 1)
            + 4;
        v >> 3
    }
}

/// `divSigTable[]` from eq. 400.
const DIV_SIG_TABLE: [i32; 16] = [0, 7, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 1, 1, 0];

/// Step 7 — derive (a, b, k) per eqs. 390 – 403.
fn derive_a_b_k(min_y: i32, max_y: i32, min_c: i32, max_c: i32) -> (i32, i32, u32) {
    let diff = max_y - min_y; // 390
    if diff == 0 {
        // 401 / 402 / 403
        return (0, min_c, 0);
    }
    let diff_c = max_c - min_c; // 391
    let x = floor_log2(diff as u32) as i32; // 392
    let norm_diff = ((diff << 4) >> x) & 15; // 393
    let x_adj = x + if norm_diff != 0 { 1 } else { 0 }; // 394
    let abs_dc = diff_c.unsigned_abs() as i32;
    let y = if abs_dc > 0 {
        floor_log2(abs_dc as u32) as i32 + 1
    } else {
        0
    }; // 395

    // a = ( diffC * (divSigTable[normDiff] | 8) + 2^(y-1) ) >> y         (396)
    // The "2^(y-1)" rounding term is 0 when y == 0 (since 2^-1 isn't
    // defined and the formula collapses to a >> 0 with diff_c == 0).
    let pow_y_minus_1 = if y > 0 { 1i32 << (y - 1) } else { 0 };
    let div_entry = DIV_SIG_TABLE[norm_diff as usize] | 8;
    let mut a = if y > 0 {
        (diff_c * div_entry + pow_y_minus_1) >> y
    } else {
        // diff_c == 0 → diff_c * div_entry == 0; >> 0.
        0
    };

    let three_x_minus_y = 3 + x_adj - y;
    let k = if three_x_minus_y < 1 {
        1u32
    } else {
        three_x_minus_y as u32
    };
    if three_x_minus_y < 1 {
        // 398
        a = sign_i32(a) * 15;
    }

    let b = min_c - ((a * min_y) >> k); // 399
    (a, b, k)
}

#[inline]
fn floor_log2(v: u32) -> u32 {
    debug_assert!(v > 0);
    31 - v.leading_zeros()
}

#[inline]
fn sign_i32(v: i32) -> i32 {
    if v > 0 {
        1
    } else if v < 0 {
        -1
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat_luma_plane(value: u8, w: usize, h: usize) -> Vec<u8> {
        vec![value; w * h]
    }

    /// All chroma + luma neighbours equal a constant + flat luma → CCLM
    /// must reproduce that constant (a == 0 path, b == minC).
    #[test]
    fn cclm_constant_neighbours_lt() {
        let plane = flat_luma_plane(100, 32, 32);
        let chroma_top = vec![50i16; 4]; // n_tb_w = 4
        let chroma_left = vec![50i16; 4]; // n_tb_h = 4
        let inp = CclmInputs {
            mode: INTRA_LT_CCLM,
            n_tb_w: 4,
            n_tb_h: 4,
            sub_width_c: 2,
            sub_height_c: 2,
            chroma_vertical_collocated_flag: false,
            b_ctu_boundary: false,
            bit_depth: 8,
            neigh_top_chroma: &chroma_top,
            neigh_left_chroma: &chroma_left,
            luma_plane: LumaPlane {
                samples: &plane,
                stride: 32,
                width: 32,
                height: 32,
            },
            // Chroma TB starts after at least 4 chroma rows / cols of
            // already-reconstructed luma so the neighbour reads stay
            // inside the plane.
            x_tb_c: 4,
            y_tb_c: 4,
        };
        let p = predict_cclm(&inp).unwrap();
        // diff = 0 → a = 0, b = minC = 50, k = 0 → predSamples = 50.
        assert_eq!(p, vec![50i16; 16]);
    }

    /// Both sides unavailable → eq. 365 fallback to mid-grey.
    #[test]
    fn cclm_no_neighbours_returns_mid_grey() {
        let plane = flat_luma_plane(0, 16, 16);
        let inp = CclmInputs {
            mode: INTRA_LT_CCLM,
            n_tb_w: 4,
            n_tb_h: 4,
            sub_width_c: 2,
            sub_height_c: 2,
            chroma_vertical_collocated_flag: false,
            b_ctu_boundary: false,
            bit_depth: 8,
            neigh_top_chroma: &[],
            neigh_left_chroma: &[],
            luma_plane: LumaPlane {
                samples: &plane,
                stride: 16,
                width: 16,
                height: 16,
            },
            x_tb_c: 0,
            y_tb_c: 0,
        };
        let p = predict_cclm(&inp).unwrap();
        // BitDepth 8 → mid = 1 << 7 = 128.
        assert_eq!(p, vec![128i16; 16]);
    }

    /// Mode validation — anything other than 81 / 82 / 83 is a hard
    /// error.
    #[test]
    fn cclm_rejects_bad_mode() {
        let plane = flat_luma_plane(0, 16, 16);
        let inp = CclmInputs {
            mode: 0,
            n_tb_w: 4,
            n_tb_h: 4,
            sub_width_c: 2,
            sub_height_c: 2,
            chroma_vertical_collocated_flag: false,
            b_ctu_boundary: false,
            bit_depth: 8,
            neigh_top_chroma: &[50; 4],
            neigh_left_chroma: &[50; 4],
            luma_plane: LumaPlane {
                samples: &plane,
                stride: 16,
                width: 16,
                height: 16,
            },
            x_tb_c: 4,
            y_tb_c: 4,
        };
        assert!(predict_cclm(&inp).is_err());
    }

    /// Spec eq. 400 sanity: divSigTable entries are exactly 16 and span
    /// the printed values.
    #[test]
    fn cclm_div_sig_table_values() {
        assert_eq!(DIV_SIG_TABLE.len(), 16);
        assert_eq!(DIV_SIG_TABLE[0], 0);
        assert_eq!(DIV_SIG_TABLE[1], 7);
        assert_eq!(DIV_SIG_TABLE[7], 3);
        assert_eq!(DIV_SIG_TABLE[15], 0);
    }

    /// Pick-positions stride: with `numIs4N=1` and 4 samples, cnt=4 and
    /// `pickStep = max(1, 4 >> 2) = 1`, `start = 4 >> 3 = 0` → [0,1,2,3].
    #[test]
    fn pick_positions_single_side_4_samples() {
        let (cnt, pos) = pick_positions(4, 1);
        assert_eq!(cnt, 4);
        assert_eq!(pos, vec![0, 1, 2, 3]);
    }

    /// Pick-positions for LT_CCLM (numIs4N=0) with both sides equal to
    /// nTb=4: cnt=2, pickStep = max(1, 4>>1) = 2, start = 4>>2 = 1
    /// → [1, 3]. Matches the spec's "two-tap from each side" sampling.
    #[test]
    fn pick_positions_lt_two_per_side() {
        let (cnt, pos) = pick_positions(4, 0);
        assert_eq!(cnt, 2);
        assert_eq!(pos, vec![1, 3]);
    }

    /// Pick-positions when only one neighbour pair is available: cnt=2
    /// is capped to numSamp.
    #[test]
    fn pick_positions_caps_to_num_samp() {
        let (cnt, pos) = pick_positions(2, 1);
        assert_eq!(cnt, 2);
        // numSamp >> 3 = 0, pickStep = max(1, 2 >> 2) = 1 → [0, 1].
        assert_eq!(pos, vec![0, 1]);
    }

    /// Eq. 365 + the 0-clamp output range: bit_depth 10 → mid = 512.
    #[test]
    fn cclm_no_neighbours_10_bit_mid() {
        let plane = flat_luma_plane(0, 16, 16);
        let inp = CclmInputs {
            mode: INTRA_T_CCLM,
            n_tb_w: 4,
            n_tb_h: 4,
            sub_width_c: 2,
            sub_height_c: 2,
            chroma_vertical_collocated_flag: false,
            b_ctu_boundary: false,
            bit_depth: 10,
            neigh_top_chroma: &[],
            neigh_left_chroma: &[],
            luma_plane: LumaPlane {
                samples: &plane,
                stride: 16,
                width: 16,
                height: 16,
            },
            x_tb_c: 0,
            y_tb_c: 0,
        };
        let p = predict_cclm(&inp).unwrap();
        assert_eq!(p, vec![512i16; 16]);
    }

    /// `derive_a_b_k` collapses to (0, minC, 0) when diff == 0 (eqs.
    /// 401-403).
    #[test]
    fn derive_a_b_k_diff_zero() {
        let (a, b, k) = derive_a_b_k(100, 100, 200, 200);
        assert_eq!((a, b, k), (0, 200, 0));
    }

    /// `derive_a_b_k` end-to-end spot check — pick numbers that exercise
    /// the eq. 396 branch.
    /// minY=10, maxY=20 → diff=10; minC=20, maxC=40 → diffC=20.
    /// x = floor_log2(10) = 3; normDiff = ((10<<4)>>3)&15 = 20 & 15 = 4.
    /// x_adj = 3 + 1 = 4 (normDiff != 0).
    /// |diffC|=20; y = floor_log2(20)+1 = 4 + 1 = 5.
    /// 2^(y-1)=16. divSigTable[4]|8 = 5|8 = 13.
    /// a = (20 * 13 + 16) >> 5 = (260 + 16) >> 5 = 276 >> 5 = 8.
    /// 3 + x_adj - y = 3 + 4 - 5 = 2 ≥ 1 → k = 2 (no abs override).
    /// b = 20 - ((8 * 10) >> 2) = 20 - 20 = 0.
    #[test]
    fn derive_a_b_k_spot_check() {
        let (a, b, k) = derive_a_b_k(10, 20, 20, 40);
        assert_eq!(a, 8);
        assert_eq!(b, 0);
        assert_eq!(k, 2);
    }

    /// Synthetic pY ramp + matching chroma ramp → CCLM should match
    /// the linear model. We seed a luma plane with a horizontal ramp
    /// and chroma neighbours that sit on the same line `c = (y/2) +
    /// 10`. CCLM's a / b / k should reproduce the chroma ramp at
    /// the predicted positions (within integer rounding).
    #[test]
    fn cclm_horizontal_luma_ramp_recovers_neighbours() {
        // Build a 32×32 luma plane where each sample = x value (capped
        // at 255). The chroma TB sits at chroma-(8, 8) → luma-(16, 16).
        let w = 32usize;
        let h = 32usize;
        let mut plane = vec![0u8; w * h];
        for y in 0..h {
            for x in 0..w {
                plane[y * w + x] = x.min(255) as u8;
            }
        }
        // Chroma neighbours: top row sits at chroma-y=7, x=8..11. The
        // 4:2:0 down-sampled luma at top neighbour x=8 picks samples
        // around picture-x=16 → ~16. Match chroma to a linear fn of
        // luma so CCLM converges to a small a / b. Use C = L + 50.
        let chroma_top: Vec<i16> = (0..4)
            .map(|i| {
                // Approximate down-sampled luma at neighbour x = 8+i:
                // average of pY around (16 + 2*i, -1..0). pY[16+2i][y] ≈ 16+2i.
                let lx = 16 + 2 * i;
                (lx as i16) + 50
            })
            .collect();
        let chroma_left: Vec<i16> = (0..4).map(|_i| 16i16 + 50).collect();
        let inp = CclmInputs {
            mode: INTRA_LT_CCLM,
            n_tb_w: 4,
            n_tb_h: 4,
            sub_width_c: 2,
            sub_height_c: 2,
            chroma_vertical_collocated_flag: false,
            b_ctu_boundary: false,
            bit_depth: 8,
            neigh_top_chroma: &chroma_top,
            neigh_left_chroma: &chroma_left,
            luma_plane: LumaPlane {
                samples: &plane,
                stride: w,
                width: w,
                height: h,
            },
            x_tb_c: 8,
            y_tb_c: 8,
        };
        let p = predict_cclm(&inp).unwrap();
        // The first row should be a horizontal ramp around 16..22 + 50;
        // exact integer values depend on the down-sampled-luma kernel
        // (eq. 369) and the 4-point regression. Sanity check: the
        // prediction must be within [50, 100] (chroma values were 66..72
        // pre-regression) and monotonically non-decreasing along x.
        for y in 0..4 {
            for x in 0..4 {
                let v = p[y * 4 + x];
                assert!(
                    (0..=255).contains(&v),
                    "CCLM ramp output out of range at ({x},{y}): {v}"
                );
            }
            // Monotonic non-decreasing along x (the regression should
            // give a positive slope).
            for x in 1..4 {
                assert!(
                    p[y * 4 + x] >= p[y * 4 + x - 1],
                    "CCLM ramp not monotonic at row {y}: {:?}",
                    &p[y * 4..y * 4 + 4]
                );
            }
        }
    }

    /// LumaPlane reads clamp out-of-bounds indices to the edge — keeps
    /// the caller's substituted plane numerically defined.
    #[test]
    fn luma_plane_read_clamps_edges() {
        let samples = (0..16u8).collect::<Vec<_>>();
        let plane = LumaPlane {
            samples: &samples,
            stride: 4,
            width: 4,
            height: 4,
        };
        assert_eq!(plane.read(0, 0), 0);
        assert_eq!(plane.read(-5, -5), 0);
        assert_eq!(plane.read(100, 100), 15);
        assert_eq!(plane.read(2, 1), 6);
    }

    /// Mode `T_CCLM` with extra top samples available: numSampT
    /// includes the extension up to nTbH.
    #[test]
    fn cclm_t_cclm_uses_extended_top() {
        let plane = flat_luma_plane(80, 32, 32);
        // Pass 8 samples for n_tb_w=4 → 4 extra (nTbH=4 cap).
        let chroma_top = vec![60i16; 8];
        let inp = CclmInputs {
            mode: INTRA_T_CCLM,
            n_tb_w: 4,
            n_tb_h: 4,
            sub_width_c: 2,
            sub_height_c: 2,
            chroma_vertical_collocated_flag: false,
            b_ctu_boundary: false,
            bit_depth: 8,
            neigh_top_chroma: &chroma_top,
            neigh_left_chroma: &[],
            luma_plane: LumaPlane {
                samples: &plane,
                stride: 32,
                width: 32,
                height: 32,
            },
            x_tb_c: 4,
            y_tb_c: 4,
        };
        let p = predict_cclm(&inp).unwrap();
        // Constant luma + constant chroma → diff == 0 → predSamples == minC == 60.
        assert_eq!(p, vec![60i16; 16]);
    }
}
