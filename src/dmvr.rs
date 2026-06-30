//! Decoder-side Motion Vector Refinement (DMVR) — §8.5.3.
//!
//! DMVR is a decoder-side refinement applied to merge-mode bi-predicted
//! CUs (or their ≤16×16 sub-blocks) that satisfy a tight set of gating
//! conditions (§8.5.1 `dmvrFlag` derivation). For such CUs the decoder
//! runs a small bilateral-matching (BM) search around the initial merge
//! MV to locate a better (L0, L1) MV pair without any extra bits on the
//! wire — the refinement is driven entirely by the SAD between the two
//! list predictors over a window covering the sub-block. The refined MVs
//! replace the initial pair for the subsequent §8.5.6 motion-
//! compensation step.
//!
//! ## Spec-exact pipeline (§8.5.3)
//!
//! 1. **§8.5.3.2.1 / §8.5.3.2.2 bilinear interpolation** — for each list
//!    a single `(sbWidth + 2·srRange) × (sbHeight + 2·srRange)` array of
//!    prediction samples is built ONCE using the 2-tap *bilinear* luma
//!    filter (Table 26), NOT the 8-tap MC filter. The `srRange = 2`
//!    border on each side is exactly what the integer search shifts the
//!    SAD window across, so no per-candidate re-interpolation is needed.
//! 2. **§8.5.3.3 SAD** — the cost between the two list arrays at an
//!    integer offset `(dX, dY)` reads `pL0[x+2+dX][2y+2+dY]` against
//!    `pL1[x+2−dX][2y+2−dY]` over `x = 0..sbW−1`, `y = 0..sbH/2−1`
//!    (the spec sub-samples even rows). The `(0, 0)` baseline gets the
//!    `sad −= sad >> 2` bias (eq. 641) so the search has to clear a
//!    higher bar before it displaces the unrefined MV.
//! 3. **§8.5.3.4 array entry selection** — the integer winner is the
//!    first strict-`<` improvement over `minSad` in the spec scan order
//!    (`dY` outer, `dX` inner, both `−2..2`).
//! 4. **§8.5.3.5 parametric refinement** — when the integer winner is
//!    not on the `±2` border (`|intOffX| != 2 && |intOffY| != 2`), the
//!    half-pel offset is derived per-axis from the 3-point SAD parabola
//!    by the exact §8.5.3.5.2 integer-division pseudo-code (bit-by-bit
//!    division, `sadMinus == sadCenter → −8`, `sadPlus == sadCenter →
//!    +8`), bounded to `[−8, +8]` 1/16-pel units.
//! 5. The L1 delta is the negation of the L0 delta (eqs. 618 / 619):
//!    `dMvL1 = −dMvL0`.
//!
//! ## Spec reference
//!
//! ITU-T H.266 | ISO/IEC 23090-3 (V4, 01/2026) §8.5.3.1 ("General") +
//! §8.5.3.2 ("Fractional sample bilinear interpolation process") +
//! §8.5.3.3 ("Sum of absolute differences calculation process") +
//! §8.5.3.4 ("Array entry selection process") + §8.5.3.5 ("Parametric
//! motion vector refinement process"). The implementation is spec-only —
//! no third-party VVC decoder source was consulted.

use oxideav_core::{Error, Result};

use crate::inter::MotionVector;
use crate::reconstruct::PicturePlane;

/// Maximum per-axis integer-pel refinement delta (§8.5.3.1 `srRange`).
///
/// The spec sets `srRange = 2` luma samples on each side of the initial
/// MV. The integer search covers `(2 · 2 + 1)² = 25` candidates.
pub const DMVR_SEARCH_RANGE: i32 = 2;

/// §8.5.3.2.2 Table 26 — luma bilinear interpolation filter coefficients
/// `fbL[p][0..1]` for each 1/16 fractional sample position `p` in
/// `0..=15`. Position `0` (integer) is the trivial `{16, 0}` so the
/// 2-tap weighted sum at frac 0 reduces to `refPic << shift1` after the
/// `>> shift1` (see §8.5.3.2.2 eq. 635 special-case); we keep it here so
/// the generic two-tap path is uniform.
const FB_L: [[i32; 2]; 16] = [
    [16, 0],
    [15, 1],
    [14, 2],
    [13, 3],
    [12, 4],
    [11, 5],
    [10, 6],
    [9, 7],
    [8, 8],
    [7, 9],
    [6, 10],
    [5, 11],
    [4, 12],
    [3, 13],
    [2, 14],
    [1, 15],
];

/// `dmvrFlag` derivation conditions per the §8.5.1 inter-prediction gate
/// — returns true iff DMVR should run for the current CU.
///
/// The bullets check (in order):
///
/// * `sps_dmvr_enabled_flag` — SPS tool flag.
/// * `ph_dmvr_disabled_flag` — picture-header off-switch (negated).
/// * `general_merge_flag` — DMVR only runs in merge mode.
/// * `predFlagL0 == predFlagL1 == 1` — bi-pred required.
/// * Reference pictures bracket the current one symmetrically in POC
///   space: `DiffPicOrderCnt(currPic, refL0) == DiffPicOrderCnt(refL1,
///   currPic)` (this is the caller-supplied `bracketed_same_diff_poc`).
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

/// Per-axis half-pel refinement offset returned by the §8.5.3.5
/// parametric pass.
///
/// Components live in spec 1/16-pel units (same as [`MotionVector`]):
/// half-pel = 8, quarter-pel = 4. Each component is bounded to
/// `[-8, +8]` by §8.5.3.5.2 so the parametric pass cannot wander more
/// than one half-pel per axis off the integer-pel winner.
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
    /// Integer-pel delta `(intOffX, intOffY)` selected by §8.5.3.4, in
    /// *integer* luma samples (not 1/16-pel units). Stored for test
    /// introspection.
    pub int_delta_x: i32,
    pub int_delta_y: i32,
    /// Half-pel sub-offset chosen by the §8.5.3.5 parametric pass.
    pub half_pel: HalfPelOffset,
    /// `dmvrSad` — the minimum SAD of the chosen integer-pel candidate.
    pub final_int_sad: u64,
    /// `minSad` — the §8.5.3.1 `(0, 0)` baseline SAD (after the eq. 641
    /// `sad −= sad >> 2` bias).
    pub baseline_sad: u64,
}

/// §8.5.3.2.2 luma sample bilinear interpolation for one fractional
/// sample. `int_x`/`int_y` are the full-sample base location, `frac_x`/
/// `frac_y` the 1/16 fractional offsets (`0..=15`). Reference samples
/// are read from `ref_pic` with edge clamping (the §8.5.3.2.2
/// `Clip3(0, picW−1, …)` / `Clip3(0, picH−1, …)` border handling for the
/// non-subpicture, non-wraparound case). The returned value is in the
/// §8.5.3.2.2 internal precision (BitDepth-8 here, so `shift3 = 2`).
#[inline]
fn bilinear_sample(
    ref_pic: &PicturePlane,
    int_x: i32,
    int_y: i32,
    frac_x: usize,
    frac_y: usize,
    bit_depth: u32,
) -> i32 {
    // §8.5.3.2.2 eqs. 624 – 630.
    let shift1 = bit_depth as i32 - 6;
    let offset1 = 1 << (shift1 - 1);
    let shift2 = 4;
    let offset2 = 1 << (shift2 - 1);
    let shift3 = 10 - bit_depth as i32;
    let shift4 = bit_depth as i32 - 10;
    let offset4 = if shift4 > 0 { 1 << (shift4 - 1) } else { 0 };

    let pw = ref_pic.width as i32;
    let ph = ref_pic.height as i32;
    // §8.5.3.2.2 eqs. 633 / 634 (no subpic, no wraparound): edge clamp.
    let at = |xi: i32, yi: i32| -> i32 {
        let xc = xi.clamp(0, pw - 1) as usize;
        let yc = yi.clamp(0, ph - 1) as usize;
        ref_pic.samples[yc * ref_pic.stride + xc] as i32
    };

    if frac_x == 0 && frac_y == 0 {
        // eq. 635.
        let v = at(int_x, int_y);
        if bit_depth <= 10 {
            v << shift3
        } else {
            (v + offset4) >> shift4
        }
    } else if frac_x != 0 && frac_y == 0 {
        // eq. 636.
        let f = FB_L[frac_x];
        let s = f[0] * at(int_x, int_y) + f[1] * at(int_x + 1, int_y);
        (s + offset1) >> shift1
    } else if frac_x == 0 && frac_y != 0 {
        // eq. 637.
        let f = FB_L[frac_y];
        let s = f[0] * at(int_x, int_y) + f[1] * at(int_x, int_y + 1);
        (s + offset1) >> shift1
    } else {
        // eqs. 638 / 639 — separable horizontal-then-vertical.
        let fx = FB_L[frac_x];
        let fy = FB_L[frac_y];
        let mut temp = [0i32; 2];
        for (n, t) in temp.iter_mut().enumerate() {
            let yi = int_y + n as i32;
            let s = fx[0] * at(int_x, yi) + fx[1] * at(int_x + 1, yi);
            *t = (s + offset1) >> shift1;
        }
        let s = fy[0] * temp[0] + fy[1] * temp[1];
        (s + offset2) >> shift2
    }
}

/// §8.5.3.2.1 — build the `predWidth × predHeight` bilinear prediction
/// array for one list, where `predWidth = sbWidth + 2·srRange` and
/// `predHeight = sbHeight + 2·srRange`. The array is laid out row-major
/// (`pred[yL * predWidth + xL]`) in §8.5.3.2.2 internal precision.
///
/// `sb_x` / `sb_y` is the sub-block origin in the *current* picture
/// (`(xSb, ySb)` in eqs. 620 / 621). The integer/fractional split of the
/// MV (eqs. 620 – 623) and the `− srRange` border shift are folded into
/// the per-sample base location.
fn bilinear_pred_array(
    ref_pic: &PicturePlane,
    sb_x: i32,
    sb_y: i32,
    pred_w: usize,
    pred_h: usize,
    mv: MotionVector,
    sr_range: i32,
    bit_depth: u32,
) -> Vec<i32> {
    // eqs. 622 / 623.
    let frac_x = (mv.x & 15) as usize;
    let frac_y = (mv.y & 15) as usize;
    // Integer part of (xSb + mv>>4 − srRange); the per-sample xL/yL add
    // on top (eqs. 620 / 621).
    let base_x = sb_x + (mv.x >> 4) - sr_range;
    let base_y = sb_y + (mv.y >> 4) - sr_range;

    let mut out = vec![0i32; pred_w * pred_h];
    for yl in 0..pred_h {
        for xl in 0..pred_w {
            let int_x = base_x + xl as i32;
            let int_y = base_y + yl as i32;
            out[yl * pred_w + xl] =
                bilinear_sample(ref_pic, int_x, int_y, frac_x, frac_y, bit_depth);
        }
    }
    out
}

/// §8.5.3.3 — sum of absolute differences between the two list prediction
/// arrays at integer offset `(dX, dY)`. `pred_w` is the padded array
/// width (`sbW + 2·srRange`); the `+2` constants in the spec indexing are
/// the `srRange = 2` border. The SAD sub-samples even rows (`2·y`) per
/// eq. 640, and the `(0, 0)` candidate gets the eq. 641 `sad −= sad >> 2`
/// bias.
fn sad_at_offset(
    p_l0: &[i32],
    p_l1: &[i32],
    pred_w: usize,
    sb_w: usize,
    sb_h: usize,
    dx: i32,
    dy: i32,
) -> u64 {
    debug_assert_eq!(DMVR_SEARCH_RANGE, 2);
    let mut sad: u64 = 0;
    let half_h = sb_h / 2;
    for y in 0..half_h {
        let row0 = ((2 * y) as i32 + 2 + dy) as usize;
        let row1 = ((2 * y) as i32 + 2 - dy) as usize;
        for x in 0..sb_w {
            let c0 = (x as i32 + 2 + dx) as usize;
            let c1 = (x as i32 + 2 - dx) as usize;
            let a = p_l0[row0 * pred_w + c0];
            let b = p_l1[row1 * pred_w + c1];
            sad += (a - b).unsigned_abs() as u64;
        }
    }
    if dx == 0 && dy == 0 {
        sad -= sad >> 2;
    }
    sad
}

/// §8.5.3.5.2 — derivation process for delta motion vector component
/// offset. Given the three SAD samples at integer offsets `−1 / 0 / +1`
/// along one axis (`sad_minus / sad_center / sad_plus`), return the
/// sub-sample correction `dMvC` in 1/16-pel units, bounded to `[−8, +8]`.
///
/// This is the exact spec pseudo-code (eq. 645): a bit-by-bit integer
/// division of `(sadMinus − sadPlus) << 4` by `((sadMinus + sadPlus) −
/// (sadCenter << 1)) << 3`, with the early `denom == 0 → 0`,
/// `sadMinus == sadCenter → −8`, and `sadPlus == sadCenter → +8` exits.
fn delta_mv_component(sad_minus: i64, sad_center: i64, sad_plus: i64) -> i32 {
    let mut denom = ((sad_minus + sad_plus) - (sad_center << 1)) << 3;
    if denom == 0 {
        return 0;
    }
    if sad_minus == sad_center {
        return -8;
    }
    if sad_plus == sad_center {
        return 8;
    }
    let mut num = (sad_minus - sad_plus) << 4;
    let mut sign_num = false;
    if num < 0 {
        num = -num;
        sign_num = true;
    }
    let mut quotient: i64 = 0;
    let mut counter = 3;
    while counter > 0 {
        counter -= 1;
        quotient <<= 1;
        if num >= denom {
            num -= denom;
            quotient += 1;
        }
        denom >>= 1;
    }
    let q = if sign_num { -quotient } else { quotient };
    q as i32
}

/// Bilateral-matching cost (§8.5.3.3) for two equally-sized luma blocks
/// read from `(0, 0)` of each plane. This is the *un-windowed*, full-grid
/// SAD retained for callers (and tests) that want a plain block-vs-block
/// cost without the §8.5.3 padded-array indexing. It does NOT apply the
/// even-row sub-sampling or the `(0, 0)` bias — those belong to the
/// search's §8.5.3.3 path ([`sad_at_offset`]).
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

/// §8.5.3.1 spec-exact refinement for one sub-block.
///
/// `sb_x` / `sb_y` is the sub-block origin in the current picture; `w` /
/// `h` is the sub-block size (the DMVR gate guarantees `w, h >= 8` for a
/// CU, but per-16×16-sub-block callers pass the sub-block dimensions).
/// `ref_l0` / `ref_l1` are the L0 / L1 reference luma planes. `bit_depth`
/// drives the §8.5.3.2.2 shifts (8 for the `u8` plane path).
///
/// Returns the refined `(L0, L1)` MV pair plus diagnostics. Per eqs.
/// 618 / 619 the L1 delta is the exact negation of the L0 delta.
#[allow(clippy::too_many_arguments)]
pub fn refine_mv_pair_bd(
    sb_x: i32,
    sb_y: i32,
    w: u32,
    h: u32,
    mv_l0_init: MotionVector,
    mv_l1_init: MotionVector,
    ref_l0: &PicturePlane,
    ref_l1: &PicturePlane,
    bit_depth: u32,
) -> Result<DmvrRefineResult> {
    if w < 8 || h < 8 {
        return Err(Error::invalid(format!(
            "h266 dmvr refine: block size {w}x{h} < 8x8 (DMVR gating violation)",
        )));
    }
    let sr = DMVR_SEARCH_RANGE;
    let sb_w = w as usize;
    let sb_h = h as usize;
    let pred_w = sb_w + 2 * sr as usize;
    let pred_h = sb_h + 2 * sr as usize;

    // §8.5.3.1 — build the per-list bilinear prediction arrays ONCE.
    let p_l0 = bilinear_pred_array(
        ref_l0, sb_x, sb_y, pred_w, pred_h, mv_l0_init, sr, bit_depth,
    );
    let p_l1 = bilinear_pred_array(
        ref_l1, sb_x, sb_y, pred_w, pred_h, mv_l1_init, sr, bit_depth,
    );

    // §8.5.3.1 — minSad at the (0, 0) offset (with the eq. 641 bias).
    let min_sad = sad_at_offset(&p_l0, &p_l1, pred_w, sb_w, sb_h, 0, 0);
    let mut dmvr_sad = min_sad;

    let mut int_off_x = 0i32;
    let mut int_off_y = 0i32;
    let mut half = HalfPelOffset::default();

    // §8.5.3.1 — "When minSad is greater than or equal to sbHeight *
    // sbWidth" the integer + parametric search runs; otherwise dMvLX
    // stays zero.
    if min_sad >= (w as u64) * (h as u64) {
        // §8.5.3.3 — fill sadArray[dX+2][dY+2] for dX, dY = −2..2.
        let span = (2 * sr + 1) as usize;
        let mut sad_array = vec![0u64; span * span];
        for dy in -sr..=sr {
            for dx in -sr..=sr {
                let s = sad_at_offset(&p_l0, &p_l1, pred_w, sb_w, sb_h, dx, dy);
                sad_array[(dy + sr) as usize * span + (dx + sr) as usize] = s;
            }
        }

        // §8.5.3.4 — array entry selection (dY outer, dX inner, strict <).
        let mut sel_min = min_sad;
        for dy in -sr..=sr {
            for dx in -sr..=sr {
                let s = sad_array[(dy + sr) as usize * span + (dx + sr) as usize];
                if s < sel_min {
                    sel_min = s;
                    int_off_x = dx;
                    int_off_y = dy;
                }
            }
        }
        dmvr_sad = sel_min;

        // §8.5.3.1 — subPelFlag gate: not on the ±2 border in either axis.
        let sub_pel = int_off_x.abs() != 2 && int_off_y.abs() != 2;

        // §8.5.3.5 — parametric refinement around the integer winner.
        if sub_pel {
            let idx = |dx: i32, dy: i32| -> i64 {
                sad_array[(dy + sr) as usize * span + (dx + sr) as usize] as i64
            };
            let center = idx(int_off_x, int_off_y);
            // §8.5.3.5.1 — X axis: sadArray[0][1], [1][1], [2][1] for
            // the 3×3 window centred at the winner.
            let dmv_x = delta_mv_component(
                idx(int_off_x - 1, int_off_y),
                center,
                idx(int_off_x + 1, int_off_y),
            );
            let dmv_y = delta_mv_component(
                idx(int_off_x, int_off_y - 1),
                center,
                idx(int_off_x, int_off_y + 1),
            );
            half.dx_q16 = dmv_x;
            half.dy_q16 = dmv_y;
        }
    }

    // §8.5.3.1 eqs. 616 / 617 + §8.5.3.5 eqs. 643 / 644.
    let dmv_l0_x = 16 * int_off_x + half.dx_q16;
    let dmv_l0_y = 16 * int_off_y + half.dy_q16;
    // eqs. 618 / 619 — L1 delta is the negation of L0.
    let mv_l0_refined = MotionVector {
        x: mv_l0_init.x + dmv_l0_x,
        y: mv_l0_init.y + dmv_l0_y,
    };
    let mv_l1_refined = MotionVector {
        x: mv_l1_init.x - dmv_l0_x,
        y: mv_l1_init.y - dmv_l0_y,
    };

    Ok(DmvrRefineResult {
        mv_l0_refined,
        mv_l1_refined,
        int_delta_x: int_off_x,
        int_delta_y: int_off_y,
        half_pel: half,
        final_int_sad: dmvr_sad,
        baseline_sad: min_sad,
    })
}

/// §8.5.3.1 spec-exact refinement at BitDepth 8 (the `u8` plane path).
///
/// Thin wrapper over [`refine_mv_pair_bd`] with `bit_depth = 8`. The
/// `(cu_x, cu_y)` origin is the sub-block top-left in the current picture
/// (`(xSb, ySb)` in eqs. 620 / 621).
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
    refine_mv_pair_bd(
        cu_x as i32,
        cu_y as i32,
        w,
        h,
        mv_l0_init,
        mv_l1_init,
        ref_l0,
        ref_l1,
        8,
    )
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
        let used = dmvr_used_flag(
            true, false, true, true, true, true, true, true, 0, false, false, false, 0, false,
            false, false, false, 16, 16, 0,
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
        assert_eq!(s, 160);
    }

    #[test]
    fn bilinear_sample_integer_position_is_left_shift() {
        // BitDepth 8 → shift3 = 2. Integer-position sample = v << 2.
        let mut p = PicturePlane::filled(4, 4, 0);
        p.samples[1 * p.stride + 1] = 100;
        let v = bilinear_sample(&p, 1, 1, 0, 0, 8);
        assert_eq!(v, 100 << 2);
    }

    #[test]
    fn bilinear_sample_half_pel_x_is_average() {
        // frac_x = 8 → fbL = {8, 8}. shift1 = 2, offset1 = 2.
        // (8*a + 8*b + 2) >> 2 for a=40, b=80 → (320+640+2)>>2 = 240.
        let mut p = PicturePlane::filled(4, 4, 0);
        p.samples[0] = 40;
        p.samples[1] = 80;
        let v = bilinear_sample(&p, 0, 0, 8, 0, 8);
        assert_eq!(v, (8 * 40 + 8 * 80 + 2) >> 2);
    }

    #[test]
    fn delta_mv_component_symmetric_well_is_zero() {
        // Symmetric → numerator 0 → dMvC 0.
        assert_eq!(delta_mv_component(100, 80, 100), 0);
    }

    #[test]
    fn delta_mv_component_flat_denominator_zero() {
        // denom == 0 → 0.
        assert_eq!(delta_mv_component(100, 100, 100), 0);
    }

    #[test]
    fn delta_mv_component_edge_equalities() {
        // sadMinus == sadCenter → −8.
        assert_eq!(delta_mv_component(80, 80, 120), -8);
        // sadPlus == sadCenter → +8.
        assert_eq!(delta_mv_component(120, 80, 80), 8);
    }

    #[test]
    fn delta_mv_component_division_is_bounded() {
        // Lower on the minus side → minimum to the left → negative.
        let d = delta_mv_component(80, 70, 120);
        assert!((-8..=0).contains(&d), "dMvC {d} out of [-8, 0]");
    }

    #[test]
    fn refine_mv_pair_recovers_integer_shift_on_synthetic_block() {
        // Two reference planes shifted by 2 samples horizontally. With
        // the spec opposite-direction pairing the BM cost has a unique
        // integer winner. The bilinear search must land on it.
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
        // §8.5.3.1 — when minSad < sbWidth*sbHeight the search is skipped
        // and dMvLX stays zero. Identical references give baseline SAD 0.
        let w = 16u32;
        let h = 16u32;
        let r = stripe_plane(64, 64, 0);
        let r2 = stripe_plane(64, 64, 0);
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
    fn refine_mv_pair_l1_delta_is_negation_of_l0() {
        // Whatever the search picks, dMvL1 == −dMvL0 (eqs. 618 / 619).
        let w = 16u32;
        let h = 16u32;
        let r_l0 = stripe_plane(64, 64, 0);
        let r_l1 = stripe_plane(64, 64, 3);
        let init0 = MotionVector::from_int_pel(1, 0);
        let init1 = MotionVector::from_int_pel(-1, 0);
        let res = apply_dmvr(16, 16, w, h, init0, init1, &r_l0, &r_l1).unwrap();
        let dl0 = (res.mv_l0_refined.x - init0.x, res.mv_l0_refined.y - init0.y);
        let dl1 = (res.mv_l1_refined.x - init1.x, res.mv_l1_refined.y - init1.y);
        assert_eq!(dl1.0, -dl0.0, "dMvL1.x must be −dMvL0.x");
        assert_eq!(dl1.1, -dl0.1, "dMvL1.y must be −dMvL0.y");
    }

    #[test]
    fn refine_then_bipred_lowers_sse_vs_unrefined() {
        // Synthetic 16×16 CU; the two reference frames are shifted by
        // ±1 sample. Refined bi-pred should beat unrefined SSE.
        let w = 16u32;
        let h = 16u32;
        let r_l0 = stripe_plane(64, 64, -1);
        let r_l1 = stripe_plane(64, 64, 1);
        let truth = stripe_plane(64, 64, 0);

        let cu_x = 16u32;
        let cu_y = 16u32;

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
            sse_ref <= sse_base,
            "refined SSE {sse_ref} should not exceed baseline {sse_base}",
        );
    }
}
