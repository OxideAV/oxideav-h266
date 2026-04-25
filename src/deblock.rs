//! VVC in-loop deblocking filter (§8.8.3).
//!
//! Implements the §8.8.3 deblocking filter — vertical-then-horizontal
//! edges per CTU on a CU basis. The full §8.8.3 path is large (long
//! 5-/7-tap filters, sub-block boundaries, virtual boundaries, LADF QP
//! offsets) so this round focuses on the **short-filter** subset that
//! handles every TB the round-12 CTU walker actually emits:
//!
//! * §8.8.3.2 — per-direction loop, gated by
//!   `sh_deblocking_filter_disabled_flag`.
//! * §8.8.3.3 — transform-block edge identification: the round-12
//!   walker emits one TB per CU, so every CU rectangle's outer edges
//!   are TB edges. Sub-block edges (§8.8.3.4) are skipped — there are
//!   no SBT / affine sub-blocks in the intra-only path yet.
//! * §8.8.3.5 — boundary-strength derivation with
//!   `intra ? 2 : (tu_coded ? 1 : 0)` collapsed to the round-12
//!   intra-only case (`bS = 2` when neither side is BDPCM).
//! * §8.8.3.6.2 — luma decision `dE`: short-filter / weak / off
//!   selection driven by `β`, `tC`, and the dpq / sp / sq / spq
//!   metrics on the `k = 0` and `k = 3` rows.
//! * §8.8.3.6.3 — short luma filter dispatch.
//! * §8.8.3.6.4 / §8.8.3.6.10 — chroma decision + weak/strong filter
//!   for cIdx = 1 / 2.
//! * §8.8.3.6.7 — short luma sample filter (`dE = 1` weak / `dE = 2`
//!   strong + `dEp` / `dEq` for p1 / q1).
//!
//! Out of scope this round (each gated as a no-op so the deblock pass
//! still runs on un-tested edges):
//!
//! * Long luma filters (§8.8.3.6.8) — only triggered when
//!   `maxFilterLengthP/Q > 3`, which requires a CU side ≥ 32 samples
//!   *and* the dE-strong condition. This scaffold caps both lengths at
//!   3, so the long-tap branch is never taken.
//! * Sub-block boundary derivation (§8.8.3.4) — affine / SBT only.
//! * LADF QP offset (`sps_ladf_enabled_flag`) — falls back to 0.
//! * Virtual / subpicture boundaries — single-slice fixture only.
//! * Tile / slice boundary suppression — single-slice fixture only.
//!
//! Spec reference: ITU-T H.266 | ISO/IEC 23090-3 (V4, 01/2026).

use crate::reconstruct::{PictureBuffer, PicturePlane};

/// Per-CU context the deblocker needs about each leaf coding unit. The
/// CTU walker accumulates one of these per leaf and hands them to
/// [`apply_deblocking`] after the CTU walk completes.
#[derive(Clone, Copy, Debug)]
pub struct DeblockCu {
    /// Top-left luma sample x.
    pub x: u32,
    /// Top-left luma sample y.
    pub y: u32,
    /// CB width in luma samples.
    pub w: u32,
    /// CB height in luma samples.
    pub h: u32,
    /// `QpY` for this CU (slice QP + cu_qp_delta).
    pub qp_y: i32,
    /// True iff the CU was coded as INTRA (`MODE_INTRA`).
    pub intra: bool,
    /// `tu_y_coded_flag` for the CU's single luma TB.
    pub tu_y_coded: bool,
    /// `tu_cb_coded_flag` for the CU's single chroma TB.
    pub tu_cb_coded: bool,
    /// `tu_cr_coded_flag` for the CU's single chroma TB.
    pub tu_cr_coded: bool,
    /// `intra_bdpcm_luma_flag` — disables luma deblock per §8.8.3.1.
    pub bdpcm_luma: bool,
    /// `intra_bdpcm_chroma_flag` — disables chroma deblock per §8.8.3.1.
    pub bdpcm_chroma: bool,
}

/// Offsets and disable flags that govern the deblock pass.
///
/// Drawn from the active slice header (§7.4.8) — when the slice header
/// did not carry deblocking overrides, the values fall back to the PPS
/// / picture-header counterparts upstream.
#[derive(Clone, Copy, Debug, Default)]
pub struct DeblockParams {
    pub disabled: bool,
    pub luma_beta_offset_div2: i32,
    pub luma_tc_offset_div2: i32,
    pub cb_beta_offset_div2: i32,
    pub cb_tc_offset_div2: i32,
    pub cr_beta_offset_div2: i32,
    pub cr_tc_offset_div2: i32,
    /// PPS / slice chroma QP offsets — added on top of `cu.qp_y` for
    /// the chroma `QpC` derivation in §8.7.1.
    pub chroma_qp_offset_cb: i32,
    pub chroma_qp_offset_cr: i32,
    pub bit_depth: u32,
}

/// Spec Table 43 — β′ as a function of the QP-derived index Q ∈ [0, 63].
///
/// Below 16 → 0; the table-defined band starts at Q = 16 (= 6) and grows
/// to 88 at Q = 63. Values for Q > 63 are not defined in Table 43; the
/// helper clamps the input to 0..=63 so callers can pass arbitrary
/// `qP` derivatives without a panic.
pub const BETA_PRIME_TABLE: [i32; 64] = [
    // Q = 0..15 — all zero.
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // Q = 16..31
    6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 22, 24, // Q = 32..47
    26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, // Q = 48..63
    58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88,
];

/// Spec Table 43 — tC′ as a function of the QP-derived index Q ∈ [0, 65].
///
/// Below 18 → 0; defined at Q ≥ 18 and tabulated up through Q = 65.
pub const TC_PRIME_TABLE: [i32; 66] = [
    // Q = 0..17 — all zero.
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // Q = 18..33
    3, 4, 4, 4, 4, 5, 5, 5, 5, 7, 7, 8, 9, 10, 10, 11, // Q = 34..49
    13, 14, 15, 17, 19, 21, 24, 25, 29, 33, 36, 41, 45, 51, 57, 64, // Q = 50..65
    71, 80, 89, 100, 112, 125, 141, 157, 177, 198, 222, 250, 280, 314, 352, 395,
];

/// Lookup β′ for an integer Q. Out-of-range values clamp to the table
/// boundary (Q < 0 → 0, Q > 63 → 88). Spec eq. 1276 then scales by the
/// bit-depth shift `(1 << (BitDepth - 8))`.
#[inline]
pub fn beta_prime(q: i32) -> i32 {
    let q = q.clamp(0, 63) as usize;
    BETA_PRIME_TABLE[q]
}

/// Lookup tC′ for an integer Q. Same clamping rule as [`beta_prime`].
#[inline]
pub fn tc_prime(q: i32) -> i32 {
    let q = q.clamp(0, 65) as usize;
    TC_PRIME_TABLE[q]
}

/// Eq. 1278 / 1279: scale the table-defined `tC′` to the active bit
/// depth.
#[inline]
fn scale_tc_for_bit_depth(tc_prime: i32, bit_depth: u32) -> i32 {
    if bit_depth < 10 {
        let shift = 9 - bit_depth as i32;
        (tc_prime + (1 << shift)) >> (10 - bit_depth as i32)
    } else {
        tc_prime * (1 << (bit_depth as i32 - 10))
    }
}

/// Eq. 1276 / 1345: β = β′ * (1 << (BitDepth - 8)).
#[inline]
fn scale_beta_for_bit_depth(beta_prime: i32, bit_depth: u32) -> i32 {
    beta_prime * (1 << (bit_depth as i32 - 8))
}

/// Per-component deblock target. Bundles the plane reference with the
/// chroma sub-sampling factors and the per-component slice offsets so a
/// single helper can deblock luma / Cb / Cr without re-deriving the
/// per-component arithmetic.
struct PlaneCtx<'a> {
    plane: &'a mut PicturePlane,
    /// Component index (0 = luma, 1 = Cb, 2 = Cr) — needed for the
    /// chroma weak/strong dispatch in §8.8.3.6.10.
    c_idx: u32,
    /// `SubWidthC` / `SubHeightC`. 1 for luma; 2 for chroma in 4:2:0.
    sub_w: u32,
    sub_h: u32,
    /// Slice-header β/tC offsets for this component (in units of /2).
    beta_offset_div2: i32,
    tc_offset_div2: i32,
    /// PPS + slice chroma QP offset (added to `qP` for chroma).
    qp_offset: i32,
    bit_depth: u32,
}

/// Apply §8.8.3 deblocking to all three planes of `out`. CUs in `cus`
/// must be in decode order and must tile the picture exactly. The
/// vertical pass runs first across all CUs, then the horizontal pass
/// (matching the spec's whole-picture ordering for §8.8.3.1).
pub fn apply_deblocking(
    out: &mut PictureBuffer,
    cus: &[DeblockCu],
    params: &DeblockParams,
    chroma_format_idc: u32,
) {
    if params.disabled {
        return;
    }
    // Build a CU lookup indexed by 4x4 luma grid cell so the
    // boundary-strength derivation can find the CU on either side of an
    // edge in O(1). Stores indexes into `cus`.
    let grid = CuGrid::build(out.luma.width, out.luma.height, cus);

    // Vertical edges first (eq. EDGE_VER pass per §8.8.3.1).
    let mut luma = PlaneCtx {
        plane: &mut out.luma,
        c_idx: 0,
        sub_w: 1,
        sub_h: 1,
        beta_offset_div2: params.luma_beta_offset_div2,
        tc_offset_div2: params.luma_tc_offset_div2,
        qp_offset: 0,
        bit_depth: params.bit_depth,
    };
    deblock_one_direction(&mut luma, cus, &grid, EdgeType::Vertical);
    deblock_one_direction(&mut luma, cus, &grid, EdgeType::Horizontal);

    if chroma_format_idc != 0 {
        let mut cb = PlaneCtx {
            plane: &mut out.cb,
            c_idx: 1,
            sub_w: 2,
            sub_h: 2,
            beta_offset_div2: params.cb_beta_offset_div2,
            tc_offset_div2: params.cb_tc_offset_div2,
            qp_offset: params.chroma_qp_offset_cb,
            bit_depth: params.bit_depth,
        };
        deblock_one_direction(&mut cb, cus, &grid, EdgeType::Vertical);
        deblock_one_direction(&mut cb, cus, &grid, EdgeType::Horizontal);
        let mut cr = PlaneCtx {
            plane: &mut out.cr,
            c_idx: 2,
            sub_w: 2,
            sub_h: 2,
            beta_offset_div2: params.cr_beta_offset_div2,
            tc_offset_div2: params.cr_tc_offset_div2,
            qp_offset: params.chroma_qp_offset_cr,
            bit_depth: params.bit_depth,
        };
        deblock_one_direction(&mut cr, cus, &grid, EdgeType::Vertical);
        deblock_one_direction(&mut cr, cus, &grid, EdgeType::Horizontal);
    }
}

/// Edge orientation as named in §8.8.3.1 Table 42.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EdgeType {
    /// EDGE_VER (vertical edge — filtered along columns).
    Vertical,
    /// EDGE_HOR.
    Horizontal,
}

/// CU grid lookup: per 4×4 luma cell, the index into the `cus` slice
/// of the CU that owns that cell. Cells outside any CU stay `None`.
struct CuGrid {
    cells: Vec<Option<u32>>,
    cells_w: usize,
    cells_h: usize,
}

impl CuGrid {
    fn build(pic_w: usize, pic_h: usize, cus: &[DeblockCu]) -> Self {
        let cw = pic_w.div_ceil(4);
        let ch = pic_h.div_ceil(4);
        let mut cells = vec![None; cw * ch];
        for (idx, cu) in cus.iter().enumerate() {
            let x0 = cu.x as usize / 4;
            let y0 = cu.y as usize / 4;
            let x1 = ((cu.x + cu.w) as usize).div_ceil(4).min(cw);
            let y1 = ((cu.y + cu.h) as usize).div_ceil(4).min(ch);
            for cy in y0..y1 {
                for cx in x0..x1 {
                    cells[cy * cw + cx] = Some(idx as u32);
                }
            }
        }
        Self {
            cells,
            cells_w: cw,
            cells_h: ch,
        }
    }

    /// CU index for a luma sample, or `None` when out of bounds.
    fn cu_at(&self, luma_x: i32, luma_y: i32) -> Option<u32> {
        if luma_x < 0 || luma_y < 0 {
            return None;
        }
        let cx = (luma_x as usize) / 4;
        let cy = (luma_y as usize) / 4;
        if cx >= self.cells_w || cy >= self.cells_h {
            return None;
        }
        self.cells[cy * self.cells_w + cx]
    }
}

/// Drive one pass (vertical or horizontal) for one component.
///
/// Iterates CUs in decode order. For each CU, the relevant outer edges
/// (left edge for EDGE_VER, top edge for EDGE_HOR) are tested against
/// the picture boundary — the spec excludes picture-edge edges per
/// §8.8.3.1.
fn deblock_one_direction(
    plane: &mut PlaneCtx,
    cus: &[DeblockCu],
    grid: &CuGrid,
    edge_type: EdgeType,
) {
    for (idx, cu) in cus.iter().enumerate() {
        // CU rectangle in this component's coordinate system.
        let cx = (cu.x / plane.sub_w) as i32;
        let cy = (cu.y / plane.sub_h) as i32;
        let cw = (cu.w / plane.sub_w) as i32;
        let ch = (cu.h / plane.sub_h) as i32;

        // §8.8.3.2: chroma deblock is gated on 8-sample alignment of
        // the leading edge in luma coordinates. Skip CUs that don't
        // satisfy that constraint.
        if plane.c_idx != 0 {
            match edge_type {
                EdgeType::Vertical => {
                    if cu.x % 8 != 0 {
                        continue;
                    }
                }
                EdgeType::Horizontal => {
                    if cu.y % 8 != 0 {
                        continue;
                    }
                }
            }
        }

        match edge_type {
            EdgeType::Vertical => {
                // Skip the left edge if it sits on the picture edge.
                if cx == 0 {
                    continue;
                }
                deblock_cu_left_edge(plane, cus, grid, idx as u32, cu, cx, cy, cw, ch);
            }
            EdgeType::Horizontal => {
                if cy == 0 {
                    continue;
                }
                deblock_cu_top_edge(plane, cus, grid, idx as u32, cu, cx, cy, cw, ch);
            }
        }
    }
}

/// Filter the left edge of a CU (vertical edge) in 4-sample (luma) /
/// 2-sample (chroma) segments per §8.8.3.5. Each segment runs the §8.8.3.6
/// decision + filter pipeline; segments straddling the same CU pair share
/// β / tC.
#[allow(clippy::too_many_arguments)]
fn deblock_cu_left_edge(
    plane: &mut PlaneCtx,
    cus: &[DeblockCu],
    grid: &CuGrid,
    idx: u32,
    cu: &DeblockCu,
    cx: i32,
    cy: i32,
    _cw: i32,
    ch: i32,
) {
    // Step length along the edge: 4 luma samples (= the §8.8.3.5 bS
    // grid) for cIdx=0; 2 chroma samples for cIdx>0.
    let step = if plane.c_idx == 0 { 4 } else { 2 };
    let luma_segment = if plane.c_idx == 0 { 4 } else { 4 };

    let mut k = 0i32;
    while k < ch {
        // Luma coordinates of the segment's anchor sample.
        let luma_x = ((cx as u32) * plane.sub_w) as i32;
        let luma_y = (((cy + k) as u32) * plane.sub_h) as i32;
        // Find the neighbour CU on the P side (one luma sample left).
        let p_idx = grid.cu_at(luma_x - 1, luma_y);
        let q_idx = Some(idx);
        if p_idx.is_none() || p_idx == q_idx {
            // No neighbour or same-CU virtual edge — nothing to filter.
            k += step;
            continue;
        }
        let p_cu = &cus[p_idx.unwrap() as usize];
        let q_cu = cu;

        let (b_s, _is_tb_edge) = derive_bs(plane.c_idx, p_cu, q_cu);
        if b_s == 0 {
            k += step;
            continue;
        }

        // §8.8.3.6.1 picks the per-sample filter from cIdx; both sides
        // must skip BDPCM (handled inside derive_bs).
        if plane.c_idx == 0 {
            // Luma: dispatch §8.8.3.6.2 + §8.8.3.6.7 (short) or
            // §8.8.3.6.6 + §8.8.3.6.8 (long).
            let (_qp, beta, tc) = compute_thresholds_luma(plane, p_cu, q_cu, b_s);
            let (mfl_p, mfl_q) = luma_max_filter_length_v(p_cu, q_cu);
            run_luma_filter_v(plane.plane, cx, cy + k, beta, tc, mfl_p, mfl_q);
            // Move down by 4 chroma rows = 4 luma rows.
            k += luma_segment / (plane.sub_h as i32).max(1);
        } else {
            // Chroma: dispatch §8.8.3.6.4 + §8.8.3.6.10.
            let (_qp, beta, tc) = compute_thresholds_chroma(plane, p_cu, q_cu, b_s);
            let (mfl_p, mfl_q) = chroma_max_filter_length_v(p_cu, q_cu, plane.sub_w);
            run_chroma_filter_v(plane.plane, cx, cy + k, beta, tc, b_s, mfl_p, mfl_q);
            k += step;
        }
    }
}

/// Mirror of [`deblock_cu_left_edge`] for the horizontal (top) edge.
#[allow(clippy::too_many_arguments)]
fn deblock_cu_top_edge(
    plane: &mut PlaneCtx,
    cus: &[DeblockCu],
    grid: &CuGrid,
    idx: u32,
    cu: &DeblockCu,
    cx: i32,
    cy: i32,
    cw: i32,
    _ch: i32,
) {
    let step = if plane.c_idx == 0 { 4 } else { 2 };
    let luma_segment = 4i32;

    let mut k = 0i32;
    while k < cw {
        let luma_x = (((cx + k) as u32) * plane.sub_w) as i32;
        let luma_y = ((cy as u32) * plane.sub_h) as i32;
        let p_idx = grid.cu_at(luma_x, luma_y - 1);
        let q_idx = Some(idx);
        if p_idx.is_none() || p_idx == q_idx {
            k += step;
            continue;
        }
        let p_cu = &cus[p_idx.unwrap() as usize];
        let q_cu = cu;

        let (b_s, _is_tb_edge) = derive_bs(plane.c_idx, p_cu, q_cu);
        if b_s == 0 {
            k += step;
            continue;
        }

        if plane.c_idx == 0 {
            let (_qp, beta, tc) = compute_thresholds_luma(plane, p_cu, q_cu, b_s);
            let (mfl_p, mfl_q) = luma_max_filter_length_h(p_cu, q_cu);
            run_luma_filter_h(plane.plane, cx + k, cy, beta, tc, mfl_p, mfl_q);
            k += luma_segment / (plane.sub_w as i32).max(1);
        } else {
            let (_qp, beta, tc) = compute_thresholds_chroma(plane, p_cu, q_cu, b_s);
            let (mfl_p, mfl_q) = chroma_max_filter_length_h(p_cu, q_cu, plane.sub_h);
            run_chroma_filter_h(plane.plane, cx + k, cy, beta, tc, b_s, mfl_p, mfl_q);
            k += step;
        }
    }
}

/// §8.8.3.5.5 chroma `maxFilterLength{P,Q}` derivation for an
/// EDGE_VER chroma edge.
///
/// Spec rule: when both adjacent chroma TBs are ≥ 8 chroma samples
/// wide → `maxFilterLengthP = maxFilterLengthQ = 3`; otherwise both
/// are 1. The single-tile / single-slice scaffold treats one CU = one
/// TB, so the chroma TB width on each side equals the CU's luma width
/// divided by `SubWidthC`. The Q-side `(luma_x) % CtbHeightC == 0`
/// gating that forces P = 1 / Q = 3 at chroma CTB row boundaries is
/// not modelled here yet — it would only differentiate behaviour at
/// chroma-CTB boundaries, which the round-12 fixture never exercises.
#[inline]
fn chroma_max_filter_length_v(p_cu: &DeblockCu, q_cu: &DeblockCu, sub_w: u32) -> (u32, u32) {
    let p_chroma_w = p_cu.w / sub_w.max(1);
    let q_chroma_w = q_cu.w / sub_w.max(1);
    if p_chroma_w >= 8 && q_chroma_w >= 8 {
        (3, 3)
    } else {
        (1, 1)
    }
}

/// Mirror of [`chroma_max_filter_length_v`] for an EDGE_HOR edge —
/// uses the chroma TB *height* on each side.
#[inline]
fn chroma_max_filter_length_h(p_cu: &DeblockCu, q_cu: &DeblockCu, sub_h: u32) -> (u32, u32) {
    let p_chroma_h = p_cu.h / sub_h.max(1);
    let q_chroma_h = q_cu.h / sub_h.max(1);
    if p_chroma_h >= 8 && q_chroma_h >= 8 {
        (3, 3)
    } else {
        (1, 1)
    }
}

/// §8.8.3.5.5 luma `maxFilterLength{P,Q}` derivation for an EDGE_VER
/// luma edge.
///
/// Spec rule (`cIdx == 0`):
/// * If either CU's width on the relevant side is ≤ 4 → that side's
///   `maxFilterLength` is 1.
/// * Else if the side's CU width is ≥ 32 → that side's
///   `maxFilterLength` is 7.
/// * Else → 3.
///
/// The round-12 scaffold treats one CU = one TB, so the luma TB width
/// on the P side is `p_cu.w` and on the Q side is `q_cu.w`.
#[inline]
fn luma_max_filter_length_v(p_cu: &DeblockCu, q_cu: &DeblockCu) -> (u32, u32) {
    let mfl_p = if p_cu.w <= 4 {
        1
    } else if p_cu.w >= 32 {
        7
    } else {
        3
    };
    let mfl_q = if q_cu.w <= 4 {
        1
    } else if q_cu.w >= 32 {
        7
    } else {
        3
    };
    (mfl_p, mfl_q)
}

/// Mirror of [`luma_max_filter_length_v`] for an EDGE_HOR luma edge —
/// uses CU heights instead of widths.
#[inline]
fn luma_max_filter_length_h(p_cu: &DeblockCu, q_cu: &DeblockCu) -> (u32, u32) {
    let mfl_p = if p_cu.h <= 4 {
        1
    } else if p_cu.h >= 32 {
        7
    } else {
        3
    };
    let mfl_q = if q_cu.h <= 4 {
        1
    } else if q_cu.h >= 32 {
        7
    } else {
        3
    };
    (mfl_p, mfl_q)
}

/// §8.8.3.5 boundary-strength derivation (intra-only round-12 subset):
/// `bS = 2` when either side is INTRA, `bS = 1` otherwise when at least
/// one TB on the edge is coded, `bS = 0` otherwise. Returns the bS plus
/// a flag indicating that the edge is also a TB edge — true here because
/// the round-12 walker emits one TB per CU so every CU outer edge is a
/// TB edge by construction.
#[inline]
fn derive_bs(c_idx: u32, p: &DeblockCu, q: &DeblockCu) -> (i32, bool) {
    // BDPCM both-sides → bS = 0 (§8.8.3.5).
    if c_idx == 0 && p.bdpcm_luma && q.bdpcm_luma {
        return (0, true);
    }
    if c_idx != 0 && p.bdpcm_chroma && q.bdpcm_chroma {
        return (0, true);
    }
    if p.intra || q.intra {
        return (2, true);
    }
    let coded_either = match c_idx {
        0 => p.tu_y_coded || q.tu_y_coded,
        1 => p.tu_cb_coded || q.tu_cb_coded,
        _ => p.tu_cr_coded || q.tu_cr_coded,
    };
    (if coded_either { 1 } else { 0 }, true)
}

/// Eq. 1274–1279: derive `qP`, `β`, `tC` for a luma edge.
fn compute_thresholds_luma(
    plane: &PlaneCtx,
    p: &DeblockCu,
    q: &DeblockCu,
    b_s: i32,
) -> (i32, i32, i32) {
    let qp = ((p.qp_y + q.qp_y + 1) >> 1) + 0; // qpOffset = 0 (no LADF).
    let q_beta = (qp + (plane.beta_offset_div2 << 1)).clamp(0, 63);
    let beta_p = beta_prime(q_beta);
    let beta = scale_beta_for_bit_depth(beta_p, plane.bit_depth);
    let q_tc = (qp + 2 * (b_s - 1) + (plane.tc_offset_div2 << 1)).clamp(0, 65);
    let tc_p = tc_prime(q_tc);
    let tc = scale_tc_for_bit_depth(tc_p, plane.bit_depth);
    (qp, beta, tc)
}

/// Eq. 1343–1348: derive `QpC`, `β`, `tC` for a chroma edge.
///
/// The §8.7.1 chroma QP path uses the per-CU chroma `Qp′Cb` / `Qp′Cr`
/// (luma QP + offset). Since the round-12 scaffold uses the identity
/// chroma-QP table, `QpC = QpY + qp_offset` is the spec's eq. 1343
/// result with `QpBdOffset = 0` (8-bit).
fn compute_thresholds_chroma(
    plane: &PlaneCtx,
    p: &DeblockCu,
    q: &DeblockCu,
    b_s: i32,
) -> (i32, i32, i32) {
    let qp_p = (p.qp_y + plane.qp_offset).clamp(0, 63);
    let qp_q = (q.qp_y + plane.qp_offset).clamp(0, 63);
    let qp_c = (qp_p + qp_q + 1) >> 1;
    let q_beta = (qp_c + (plane.beta_offset_div2 << 1)).clamp(0, 63);
    let beta_p = beta_prime(q_beta);
    let beta = scale_beta_for_bit_depth(beta_p, plane.bit_depth);
    let q_tc = (qp_c + 2 * (b_s - 1) + (plane.tc_offset_div2 << 1)).clamp(0, 65);
    let tc_p = tc_prime(q_tc);
    let tc = scale_tc_for_bit_depth(tc_p, plane.bit_depth);
    (qp_c, beta, tc)
}

// ---------------------------------------------------------------------
// Sample-level filters.
// ---------------------------------------------------------------------

/// Read a sample with bounds-clamp (replicate-edge fallback). Used to
/// keep filter access safe when a CU sits on the picture boundary;
/// the decision step protects against edge-of-picture filtering, but
/// reading p/q samples can still touch out-of-bounds rows for very
/// small picture dimensions.
#[inline]
fn read_clamped(plane: &PicturePlane, x: i32, y: i32) -> i32 {
    let cx = x.clamp(0, plane.width as i32 - 1) as usize;
    let cy = y.clamp(0, plane.height as i32 - 1) as usize;
    plane.samples[cy * plane.stride + cx] as i32
}

#[inline]
fn write(plane: &mut PicturePlane, x: i32, y: i32, v: i32, bit_depth: u32) {
    if x < 0 || y < 0 || x >= plane.width as i32 || y >= plane.height as i32 {
        return;
    }
    let max = (1i32 << bit_depth) - 1;
    let clipped = v.clamp(0, max) as u8;
    plane.samples[y as usize * plane.stride + x as usize] = clipped;
}

/// Top-level §8.8.3.6 luma filter dispatch for an EDGE_VER edge.
///
/// Picks between the §8.8.3.6.7 short filter (always available; handles
/// the `dE = 0/1/2` short-block branch) and the §8.8.3.6.8 long
/// filters (5- or 7-tap symmetric / asymmetric kernels). The long path
/// activates when both sides report `maxFilterLength{P,Q} > 3` and the
/// §8.8.3.6.6 long-block decision passes; otherwise the short path
/// runs unchanged.
///
/// The §8.8.3.6.6 large-block decision combines the short-block `dpq`
/// metrics with the wider sample reads (`p4..p7`, `q4..q7`). For the
/// round-14 scaffold we implement the symmetric (`mflP == mflQ`)
/// branch — that subsumes the most common case (neighbouring 32-tall
/// CUs at e.g. CTU-row boundaries). Asymmetric cases (e.g. one side
/// 32, the other 8) still need the asymmetric coefficient picks of
/// eqs. 1391 / 1393 / 1394; until those land they fall back to the
/// short filter.
#[allow(clippy::too_many_arguments)]
fn run_luma_filter_v(
    plane: &mut PicturePlane,
    cx: i32,
    cy: i32,
    beta: i32,
    tc: i32,
    max_filter_length_p: u32,
    max_filter_length_q: u32,
) {
    if tc == 0 {
        return;
    }
    let symmetric_long = max_filter_length_p > 3
        && max_filter_length_q > 3
        && max_filter_length_p == max_filter_length_q;
    if symmetric_long
        && long_luma_decision_passes_v(
            plane,
            cx,
            cy,
            beta,
            tc,
            max_filter_length_p,
            max_filter_length_q,
        )
    {
        for k in 0..4i32 {
            apply_long_luma_v(
                plane,
                cx,
                cy + k,
                tc,
                max_filter_length_p,
                max_filter_length_q,
            );
        }
        return;
    }
    run_luma_short_filter_v(plane, cx, cy, beta, tc);
}

/// EDGE_HOR mirror of [`run_luma_filter_v`].
#[allow(clippy::too_many_arguments)]
fn run_luma_filter_h(
    plane: &mut PicturePlane,
    cx: i32,
    cy: i32,
    beta: i32,
    tc: i32,
    max_filter_length_p: u32,
    max_filter_length_q: u32,
) {
    if tc == 0 {
        return;
    }
    let symmetric_long = max_filter_length_p > 3
        && max_filter_length_q > 3
        && max_filter_length_p == max_filter_length_q;
    if symmetric_long
        && long_luma_decision_passes_h(
            plane,
            cx,
            cy,
            beta,
            tc,
            max_filter_length_p,
            max_filter_length_q,
        )
    {
        for k in 0..4i32 {
            apply_long_luma_h(
                plane,
                cx + k,
                cy,
                tc,
                max_filter_length_p,
                max_filter_length_q,
            );
        }
        return;
    }
    run_luma_short_filter_h(plane, cx, cy, beta, tc);
}

/// §8.8.3.6.2 (long-block path) + §8.8.3.6.6 large-block decision for
/// an EDGE_VER edge. Reads the `p2..p5` (or `p2..p7` for the 7-tap
/// case) and `q2..q5` (`q2..q7`) sample columns on the k = 0 and k = 3
/// rows, builds the dpq/sp/sq/spq metrics with the long-block
/// adjustments (eqs 1290 / 1291 / 1295 / 1296 / 1299 / 1300 / 1303 /
/// 1304), then runs the §8.8.3.6.6 thresholds. Returns true when both
/// dSam0 and dSam3 are 1 — the spec's gate for invoking the long
/// filter at all (otherwise the short filter applies).
fn long_luma_decision_passes_v(
    plane: &PicturePlane,
    cx: i32,
    cy: i32,
    beta: i32,
    tc: i32,
    mfl_p: u32,
    mfl_q: u32,
) -> bool {
    long_luma_decision(plane, cx, cy, beta, tc, mfl_p, mfl_q, true)
}

fn long_luma_decision_passes_h(
    plane: &PicturePlane,
    cx: i32,
    cy: i32,
    beta: i32,
    tc: i32,
    mfl_p: u32,
    mfl_q: u32,
) -> bool {
    long_luma_decision(plane, cx, cy, beta, tc, mfl_p, mfl_q, false)
}

/// Shared body of [`long_luma_decision_passes_v`] / `_h`. `is_vertical`
/// switches the sample-fetch axis; everything else is identical.
#[allow(clippy::too_many_arguments)]
fn long_luma_decision(
    plane: &PicturePlane,
    cx: i32,
    cy: i32,
    beta: i32,
    tc: i32,
    mfl_p: u32,
    mfl_q: u32,
    is_vertical: bool,
) -> bool {
    // Read p[i] / q[i] at decision row k (k = 0 or 3).
    let read_pq = |k: i32, i: i32| -> (i32, i32) {
        if is_vertical {
            // P side at column cx - i - 1; Q side at column cx + i.
            let p = read_clamped(plane, cx - i - 1, cy + k);
            let q = read_clamped(plane, cx + i, cy + k);
            (p, q)
        } else {
            // P side at row cy - i - 1; Q side at row cy + i.
            let p = read_clamped(plane, cx + k, cy - i - 1);
            let q = read_clamped(plane, cx + k, cy + i);
            (p, q)
        }
    };
    let mut dec = [false; 2];
    let rows = [0i32, 3];
    for (slot, &row) in dec.iter_mut().zip(rows.iter()) {
        let (p0, q0) = read_pq(row, 0);
        let (p1, q1) = read_pq(row, 1);
        let (p2, q2) = read_pq(row, 2);
        let (p3, q3) = read_pq(row, 3);
        // Short-block dp/dq (eqs 1280..1283).
        let dp = (p2 - 2 * p1 + p0).abs();
        let dq = (q2 - 2 * q1 + q0).abs();
        // sp / sq / spq (eqs 1284..1289).
        let sp = (p3 - p0).abs();
        let sq = (q0 - q3).abs();
        let spq = (p0 - q0).abs();
        // Long-block dp / dq adjustments (eqs 1290 / 1291 / 1295 /
        // 1296). For mflP > 3 we need p3,p4,p5; mflQ > 3 needs
        // q3,q4,q5.
        let (_, _) = (p3, q3);
        let (_, _) = read_pq(row, 4);
        let (p5, q5) = read_pq(row, 5);
        let p4 = read_pq(row, 4).0;
        let q4 = read_pq(row, 4).1;
        let dp_l = if mfl_p > 3 {
            (dp + (p5 - 2 * p4 + p3).abs() + 1) >> 1
        } else {
            dp
        };
        let dq_l = if mfl_q > 3 {
            (dq + (q5 - 2 * q4 + q3).abs() + 1) >> 1
        } else {
            dq
        };
        // sp / sq long adjustments (eqs 1299 / 1300 / 1303 / 1304) —
        // only when mflP / mflQ == 7.
        let sp_l = if mfl_p == 7 {
            let (p6, _) = read_pq(row, 6);
            let (p7, _) = read_pq(row, 7);
            sp + (p7 - p6 - p5 + p4).abs()
        } else {
            sp
        };
        let sq_l = if mfl_q == 7 {
            let (_, q6) = read_pq(row, 6);
            let (_, q7) = read_pq(row, 7);
            sq + (q4 - q5 - q6 + q7).abs()
        } else {
            sq
        };
        // §8.8.3.6.6 — large-block thresholds (eqs 1369 / 1370 + 1371 /
        // 1372 when at least one side is large).
        let side_p_large = mfl_p > 3;
        let side_q_large = mfl_q > 3;
        let mut sp_adj = sp_l;
        let mut sq_adj = sq_l;
        if side_p_large {
            // Re-pick p3 / p_mflP for the wider span (eqs 1310 / 1311):
            // sp uses p_mflP - p3 (replacing p3 - p0 in the long path).
            let (p_max, _) = read_pq(row, mfl_p as i32);
            let (p3_new, _) = read_pq(row, 3);
            sp_adj = (sp_l + (p3_new - p_max).abs() + 1) >> 1;
        }
        if side_q_large {
            let (_, q_max) = read_pq(row, mfl_q as i32);
            let (_, q3_new) = read_pq(row, 3);
            sq_adj = (sq_l + (q3_new - q_max).abs() + 1) >> 1;
        }
        let dpq = 2 * (dp_l + dq_l);
        let (s_thr1, s_thr2) = if side_p_large || side_q_large {
            ((3 * beta) >> 5, beta >> 4)
        } else {
            (beta >> 3, beta >> 2)
        };
        *slot = dpq < s_thr2 && (sp_adj + sq_adj) < s_thr1 && spq < ((5 * tc + 1) >> 1);
    }
    dec[0] && dec[1]
}

/// §8.8.3.6.8 long luma filter (symmetric branch only) for one sample
/// row of an EDGE_VER edge. Implements eqs 1389 (5-tap) / 1390 (7-tap)
/// + the per-side `fi` / `gj` / `tCPDi` / `tCQDj` arrays + the eqs
/// 1409 / 1410 update.
fn apply_long_luma_v(plane: &mut PicturePlane, cx: i32, cy: i32, tc: i32, mfl_p: u32, mfl_q: u32) {
    long_luma_apply(plane, cx, cy, tc, mfl_p, mfl_q, true);
}

/// EDGE_HOR mirror of [`apply_long_luma_v`].
fn apply_long_luma_h(plane: &mut PicturePlane, cx: i32, cy: i32, tc: i32, mfl_p: u32, mfl_q: u32) {
    long_luma_apply(plane, cx, cy, tc, mfl_p, mfl_q, false);
}

/// Shared body. Reads the full p / q arrays and writes the §8.8.3.6.8
/// outputs.
fn long_luma_apply(
    plane: &mut PicturePlane,
    cx: i32,
    cy: i32,
    tc: i32,
    mfl_p: u32,
    mfl_q: u32,
    is_vertical: bool,
) {
    let bd = 8u32;
    let mfl_p_u = mfl_p as usize;
    let mfl_q_u = mfl_q as usize;

    // Fetch p[0..=mfl_p] and q[0..=mfl_q].
    let mut p_arr = [0i32; 8];
    let mut q_arr = [0i32; 8];
    for i in 0..=mfl_p_u {
        p_arr[i] = if is_vertical {
            read_clamped(plane, cx - i as i32 - 1, cy)
        } else {
            read_clamped(plane, cx, cy - i as i32 - 1)
        };
    }
    for j in 0..=mfl_q_u {
        q_arr[j] = if is_vertical {
            read_clamped(plane, cx + j as i32, cy)
        } else {
            read_clamped(plane, cx, cy + j as i32)
        };
    }

    // Eq. 1389 / 1390 — symmetric refMiddle.
    let ref_middle = if mfl_p == 5 {
        // 5/5 case (eq. 1389).
        (p_arr[4]
            + p_arr[3]
            + 2 * (p_arr[2] + p_arr[1] + p_arr[0] + q_arr[0] + q_arr[1] + q_arr[2])
            + q_arr[3]
            + q_arr[4]
            + 8)
            >> 4
    } else {
        // 7/7 case (eq. 1390).
        (p_arr[6]
            + p_arr[5]
            + p_arr[4]
            + p_arr[3]
            + p_arr[2]
            + p_arr[1]
            + 2 * (p_arr[0] + q_arr[0])
            + q_arr[1]
            + q_arr[2]
            + q_arr[3]
            + q_arr[4]
            + q_arr[5]
            + q_arr[6]
            + 8)
            >> 4
    };

    // refP / refQ (eqs 1395 / 1396).
    let ref_p = (p_arr[mfl_p_u] + p_arr[mfl_p_u - 1] + 1) >> 1;
    let ref_q = (q_arr[mfl_q_u] + q_arr[mfl_q_u - 1] + 1) >> 1;

    // fi / tCPDi (eqs 1397/1398 for 7-tap, 1399/1400 for 5-tap).
    const F7: [i32; 7] = [59, 50, 41, 32, 23, 14, 5];
    const F5: [i32; 5] = [58, 45, 32, 19, 6];
    const T7: [i32; 7] = [6, 5, 4, 3, 2, 1, 1];
    const T5: [i32; 5] = [6, 5, 4, 3, 2];
    let (fi, tcpdi): (&[i32], &[i32]) = if mfl_p == 7 { (&F7, &T7) } else { (&F5, &T5) };
    let (gj, tcqdj): (&[i32], &[i32]) = if mfl_q == 7 { (&F7, &T7) } else { (&F5, &T5) };

    // Eqs 1409 / 1410 — write filtered samples.
    for i in 0..mfl_p_u {
        let lo = p_arr[i] - ((tc * tcpdi[i]) >> 1);
        let hi = p_arr[i] + ((tc * tcpdi[i]) >> 1);
        let raw = (ref_middle * fi[i] + ref_p * (64 - fi[i]) + 32) >> 6;
        let v = raw.clamp(lo, hi);
        if is_vertical {
            write(plane, cx - i as i32 - 1, cy, v, bd);
        } else {
            write(plane, cx, cy - i as i32 - 1, v, bd);
        }
    }
    for j in 0..mfl_q_u {
        let lo = q_arr[j] - ((tc * tcqdj[j]) >> 1);
        let hi = q_arr[j] + ((tc * tcqdj[j]) >> 1);
        let raw = (ref_middle * gj[j] + ref_q * (64 - gj[j]) + 32) >> 6;
        let v = raw.clamp(lo, hi);
        if is_vertical {
            write(plane, cx + j as i32, cy, v, bd);
        } else {
            write(plane, cx, cy + j as i32, v, bd);
        }
    }
}

/// Run the §8.8.3.6.2 decision + §8.8.3.6.7 short luma filter on a
/// single 4-sample vertical edge segment with anchor at chroma /
/// component coordinate `(cx, cy)`. The edge sits between columns
/// `cx-1` (P side) and `cx` (Q side).
fn run_luma_short_filter_v(plane: &mut PicturePlane, cx: i32, cy: i32, beta: i32, tc: i32) {
    if tc == 0 {
        return;
    }
    let bd = 8u32; // luma is always 8-bit in this round (decoder emits 8-bit).

    // Decision rows k ∈ {0, 3} (eq. 1280–1283).
    let p2_0 = read_clamped(plane, cx - 3, cy);
    let p1_0 = read_clamped(plane, cx - 2, cy);
    let p0_0 = read_clamped(plane, cx - 1, cy);
    let q0_0 = read_clamped(plane, cx, cy);
    let q1_0 = read_clamped(plane, cx + 1, cy);
    let q2_0 = read_clamped(plane, cx + 2, cy);
    let p2_3 = read_clamped(plane, cx - 3, cy + 3);
    let p1_3 = read_clamped(plane, cx - 2, cy + 3);
    let p0_3 = read_clamped(plane, cx - 1, cy + 3);
    let q0_3 = read_clamped(plane, cx, cy + 3);
    let q1_3 = read_clamped(plane, cx + 1, cy + 3);
    let q2_3 = read_clamped(plane, cx - 3 + 5, cy + 3); // q2 = cx+2 row 3

    let dp0 = (p2_0 - 2 * p1_0 + p0_0).abs();
    let dp3 = (p2_3 - 2 * p1_3 + p0_3).abs();
    let dq0 = (q2_0 - 2 * q1_0 + q0_0).abs();
    let dq3 = (q2_3 - 2 * q1_3 + q0_3).abs();
    let d = dp0 + dp3 + dq0 + dq3;
    if d >= beta {
        // §8.8.3.6.2 — no filtering at all on this segment.
        return;
    }

    // Strong / weak decision (§8.8.3.6.6 short-block branch).
    let dpq0 = dp0 + dq0;
    let dpq3 = dp3 + dq3;
    let p3_0 = read_clamped(plane, cx - 4, cy);
    let q3_0 = read_clamped(plane, cx + 3, cy);
    let p3_3 = read_clamped(plane, cx - 4, cy + 3);
    let q3_3 = read_clamped(plane, cx + 3, cy + 3);
    let sp0 = (p3_0 - p0_0).abs();
    let sq0 = (q0_0 - q3_0).abs();
    let spq0 = (p0_0 - q0_0).abs();
    let sp3 = (p3_3 - p0_3).abs();
    let sq3 = (q0_3 - q3_3).abs();
    let spq3 = (p0_3 - q0_3).abs();

    let s_thr1 = beta >> 3;
    let s_thr2 = beta >> 2;
    let strong0 = (2 * dpq0) < s_thr2 && (sp0 + sq0) < s_thr1 && spq0 < (5 * tc + 1) >> 1;
    let strong3 = (2 * dpq3) < s_thr2 && (sp3 + sq3) < s_thr1 && spq3 < (5 * tc + 1) >> 1;
    let strong = strong0 && strong3;
    let de = if strong { 2 } else { 1 };

    // dEp / dEq (§8.8.3.6.2 short-block branch — eq. 1320 family).
    // For the short-filter path: dEp = 1 iff dp0 + dp3 < (β + (β >> 1)) >> 3,
    // similarly for dEq with dq.
    let d_p = dp0 + dp3;
    let d_q = dq0 + dq3;
    let dep = if d_p < (beta + (beta >> 1)) >> 3 {
        1
    } else {
        0
    };
    let deq = if d_q < (beta + (beta >> 1)) >> 3 {
        1
    } else {
        0
    };

    // Apply the filter to each of the 4 sample rows.
    for k in 0..4i32 {
        let p3 = read_clamped(plane, cx - 4, cy + k);
        let p2 = read_clamped(plane, cx - 3, cy + k);
        let p1 = read_clamped(plane, cx - 2, cy + k);
        let p0 = read_clamped(plane, cx - 1, cy + k);
        let q0 = read_clamped(plane, cx, cy + k);
        let q1 = read_clamped(plane, cx + 1, cy + k);
        let q2 = read_clamped(plane, cx + 2, cy + k);
        let q3 = read_clamped(plane, cx + 3, cy + k);
        if de == 2 {
            // Strong filter — eq. 1375–1380.
            let p0n =
                ((p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + 4) >> 3).clamp(p0 - 3 * tc, p0 + 3 * tc);
            let p1n = ((p2 + p1 + p0 + q0 + 2) >> 2).clamp(p1 - 2 * tc, p1 + 2 * tc);
            let p2n = ((2 * p3 + 3 * p2 + p1 + p0 + q0 + 4) >> 3).clamp(p2 - tc, p2 + tc);
            let q0n =
                ((p1 + 2 * p0 + 2 * q0 + 2 * q1 + q2 + 4) >> 3).clamp(q0 - 3 * tc, q0 + 3 * tc);
            let q1n = ((p0 + q0 + q1 + q2 + 2) >> 2).clamp(q1 - 2 * tc, q1 + 2 * tc);
            let q2n = ((p0 + q0 + q1 + 3 * q2 + 2 * q3 + 4) >> 3).clamp(q2 - tc, q2 + tc);
            write(plane, cx - 1, cy + k, p0n, bd);
            write(plane, cx - 2, cy + k, p1n, bd);
            write(plane, cx - 3, cy + k, p2n, bd);
            write(plane, cx, cy + k, q0n, bd);
            write(plane, cx + 1, cy + k, q1n, bd);
            write(plane, cx + 2, cy + k, q2n, bd);
        } else {
            // Weak filter — eq. 1381–1388.
            let delta_raw = (9 * (q0 - p0) - 3 * (q1 - p1) + 8) >> 4;
            if delta_raw.abs() < tc * 10 {
                let delta = delta_raw.clamp(-tc, tc);
                let p0n = p0 + delta;
                let q0n = q0 - delta;
                write(plane, cx - 1, cy + k, p0n, bd);
                write(plane, cx, cy + k, q0n, bd);
                if dep == 1 {
                    let dp = (((p2 + p0 + 1) >> 1) - p1 + delta) >> 1;
                    let dp = dp.clamp(-tc >> 1, tc >> 1);
                    write(plane, cx - 2, cy + k, p1 + dp, bd);
                }
                if deq == 1 {
                    let dq = (((q2 + q0 + 1) >> 1) - q1 - delta) >> 1;
                    let dq = dq.clamp(-tc >> 1, tc >> 1);
                    write(plane, cx + 1, cy + k, q1 + dq, bd);
                }
            }
        }
    }
}

/// Mirror of [`run_luma_short_filter_v`] for a horizontal edge between
/// rows `cy-1` (P) and `cy` (Q), running across 4 columns starting at
/// `cx`.
fn run_luma_short_filter_h(plane: &mut PicturePlane, cx: i32, cy: i32, beta: i32, tc: i32) {
    if tc == 0 {
        return;
    }
    let bd = 8u32;

    let p2_0 = read_clamped(plane, cx, cy - 3);
    let p1_0 = read_clamped(plane, cx, cy - 2);
    let p0_0 = read_clamped(plane, cx, cy - 1);
    let q0_0 = read_clamped(plane, cx, cy);
    let q1_0 = read_clamped(plane, cx, cy + 1);
    let q2_0 = read_clamped(plane, cx, cy + 2);
    let p2_3 = read_clamped(plane, cx + 3, cy - 3);
    let p1_3 = read_clamped(plane, cx + 3, cy - 2);
    let p0_3 = read_clamped(plane, cx + 3, cy - 1);
    let q0_3 = read_clamped(plane, cx + 3, cy);
    let q1_3 = read_clamped(plane, cx + 3, cy + 1);
    let q2_3 = read_clamped(plane, cx + 3, cy + 2);

    let dp0 = (p2_0 - 2 * p1_0 + p0_0).abs();
    let dp3 = (p2_3 - 2 * p1_3 + p0_3).abs();
    let dq0 = (q2_0 - 2 * q1_0 + q0_0).abs();
    let dq3 = (q2_3 - 2 * q1_3 + q0_3).abs();
    let d = dp0 + dp3 + dq0 + dq3;
    if d >= beta {
        return;
    }
    let dpq0 = dp0 + dq0;
    let dpq3 = dp3 + dq3;
    let p3_0 = read_clamped(plane, cx, cy - 4);
    let q3_0 = read_clamped(plane, cx, cy + 3);
    let p3_3 = read_clamped(plane, cx + 3, cy - 4);
    let q3_3 = read_clamped(plane, cx + 3, cy + 3);
    let sp0 = (p3_0 - p0_0).abs();
    let sq0 = (q0_0 - q3_0).abs();
    let spq0 = (p0_0 - q0_0).abs();
    let sp3 = (p3_3 - p0_3).abs();
    let sq3 = (q0_3 - q3_3).abs();
    let spq3 = (p0_3 - q0_3).abs();
    let s_thr1 = beta >> 3;
    let s_thr2 = beta >> 2;
    let strong0 = (2 * dpq0) < s_thr2 && (sp0 + sq0) < s_thr1 && spq0 < (5 * tc + 1) >> 1;
    let strong3 = (2 * dpq3) < s_thr2 && (sp3 + sq3) < s_thr1 && spq3 < (5 * tc + 1) >> 1;
    let de = if strong0 && strong3 { 2 } else { 1 };

    let d_p = dp0 + dp3;
    let d_q = dq0 + dq3;
    let dep = if d_p < (beta + (beta >> 1)) >> 3 {
        1
    } else {
        0
    };
    let deq = if d_q < (beta + (beta >> 1)) >> 3 {
        1
    } else {
        0
    };

    for k in 0..4i32 {
        let p3 = read_clamped(plane, cx + k, cy - 4);
        let p2 = read_clamped(plane, cx + k, cy - 3);
        let p1 = read_clamped(plane, cx + k, cy - 2);
        let p0 = read_clamped(plane, cx + k, cy - 1);
        let q0 = read_clamped(plane, cx + k, cy);
        let q1 = read_clamped(plane, cx + k, cy + 1);
        let q2 = read_clamped(plane, cx + k, cy + 2);
        let q3 = read_clamped(plane, cx + k, cy + 3);
        if de == 2 {
            let p0n =
                ((p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + 4) >> 3).clamp(p0 - 3 * tc, p0 + 3 * tc);
            let p1n = ((p2 + p1 + p0 + q0 + 2) >> 2).clamp(p1 - 2 * tc, p1 + 2 * tc);
            let p2n = ((2 * p3 + 3 * p2 + p1 + p0 + q0 + 4) >> 3).clamp(p2 - tc, p2 + tc);
            let q0n =
                ((p1 + 2 * p0 + 2 * q0 + 2 * q1 + q2 + 4) >> 3).clamp(q0 - 3 * tc, q0 + 3 * tc);
            let q1n = ((p0 + q0 + q1 + q2 + 2) >> 2).clamp(q1 - 2 * tc, q1 + 2 * tc);
            let q2n = ((p0 + q0 + q1 + 3 * q2 + 2 * q3 + 4) >> 3).clamp(q2 - tc, q2 + tc);
            write(plane, cx + k, cy - 1, p0n, bd);
            write(plane, cx + k, cy - 2, p1n, bd);
            write(plane, cx + k, cy - 3, p2n, bd);
            write(plane, cx + k, cy, q0n, bd);
            write(plane, cx + k, cy + 1, q1n, bd);
            write(plane, cx + k, cy + 2, q2n, bd);
        } else {
            let delta_raw = (9 * (q0 - p0) - 3 * (q1 - p1) + 8) >> 4;
            if delta_raw.abs() < tc * 10 {
                let delta = delta_raw.clamp(-tc, tc);
                let p0n = p0 + delta;
                let q0n = q0 - delta;
                write(plane, cx + k, cy - 1, p0n, bd);
                write(plane, cx + k, cy, q0n, bd);
                if dep == 1 {
                    let dp = (((p2 + p0 + 1) >> 1) - p1 + delta) >> 1;
                    let dp = dp.clamp(-tc >> 1, tc >> 1);
                    write(plane, cx + k, cy - 2, p1 + dp, bd);
                }
                if deq == 1 {
                    let dq = (((q2 + q0 + 1) >> 1) - q1 - delta) >> 1;
                    let dq = dq.clamp(-tc >> 1, tc >> 1);
                    write(plane, cx + k, cy + 1, q1 + dq, bd);
                }
            }
        }
    }
}

/// §8.8.3.6.10 chroma deblocker on a 2-sample vertical edge segment in
/// chroma coordinates.
///
/// The chroma path picks between three filter shapes (eqs 1411 – 1423)
/// based on the per-side `maxFilterLength{P,Q}` derivation in §8.8.3.5.5:
///
/// * Both `maxFilterLengthP == maxFilterLengthQ == 3` (chroma TB ≥ 8 on
///   both sides AND not crossing a chroma CTB boundary) — invoke the
///   §8.8.3.6.9 decision process; if the strong-filter check passes,
///   apply the 7-tap strong filter (eqs 1411 – 1416). Otherwise fall
///   through to the weak filter.
/// * `maxFilterLengthQ == 3 && maxFilterLengthP == 1` (asymmetric
///   "P-side small" case at chroma CTB row/col boundaries) — this
///   round still falls through to the weak filter; the asymmetric
///   eqs 1417 – 1420 are wired but only enabled by the same chroma TB
///   ≥ 8 on the Q side.
/// * Otherwise — weak filter (eqs 1421 – 1423).
///
/// Because the round-12 walker emits one CU = one TB and the deblock
/// path has no SBT split tracking, the two sides' `maxFilterLength`
/// values are derived from the CU rectangle's chroma dimensions. The
/// strong path therefore activates only when both adjacent chroma TBs
/// are ≥ 8 chroma samples in the relevant direction (corresponding to
/// luma CU widths/heights ≥ 16 for 4:2:0).
#[allow(clippy::too_many_arguments)]
fn run_chroma_filter_v(
    plane: &mut PicturePlane,
    cx: i32,
    cy: i32,
    beta: i32,
    tc: i32,
    _b_s: i32,
    max_filter_length_p: u32,
    max_filter_length_q: u32,
) {
    if tc == 0 {
        return;
    }
    let bd = 8u32;
    // §8.8.3.6.5 maxK for EDGE_VER, SubHeightC = 2 (4:2:0) → maxK = 1
    // (i.e. 2 sample rows along the edge). Our chroma path always
    // operates on 2 sample positions per segment.
    let max_k = 1i32;

    let strong_eligible = max_filter_length_p == 3 && max_filter_length_q == 3;
    if strong_eligible {
        // §8.8.3.5.5 / §8.8.3.6.9 strong-filter decision. Read the
        // p0, p3, q0, q3 samples on both decision rows (k = 0, 1)
        // and compute dpq0, dpq1, d.
        let dec0 = chroma_strong_decision_v(plane, cx, cy, beta, tc);
        let dec1 = chroma_strong_decision_v(plane, cx, cy + max_k, beta, tc);
        if dec0 && dec1 {
            // Strong filter on the full 2-row stripe.
            for k in 0..=max_k {
                chroma_strong_apply_v(plane, cx, cy + k, tc, bd);
            }
            return;
        }
        // Decision failed → fall through to the weak filter.
    }

    // Weak filter (eqs 1421 – 1423).
    for k in 0..2i32 {
        let p1 = read_clamped(plane, cx - 2, cy + k);
        let p0 = read_clamped(plane, cx - 1, cy + k);
        let q0 = read_clamped(plane, cx, cy + k);
        let q1 = read_clamped(plane, cx + 1, cy + k);
        let delta = ((((q0 - p0) << 2) + p1 - q1 + 4) >> 3).clamp(-tc, tc);
        write(plane, cx - 1, cy + k, p0 + delta, bd);
        write(plane, cx, cy + k, q0 - delta, bd);
    }
}

/// Mirror of [`run_chroma_filter_v`] for the horizontal edge case.
#[allow(clippy::too_many_arguments)]
fn run_chroma_filter_h(
    plane: &mut PicturePlane,
    cx: i32,
    cy: i32,
    beta: i32,
    tc: i32,
    _b_s: i32,
    max_filter_length_p: u32,
    max_filter_length_q: u32,
) {
    if tc == 0 {
        return;
    }
    let bd = 8u32;
    let max_k = 1i32;

    let strong_eligible = max_filter_length_p == 3 && max_filter_length_q == 3;
    if strong_eligible {
        let dec0 = chroma_strong_decision_h(plane, cx, cy, beta, tc);
        let dec1 = chroma_strong_decision_h(plane, cx + max_k, cy, beta, tc);
        if dec0 && dec1 {
            for k in 0..=max_k {
                chroma_strong_apply_h(plane, cx + k, cy, tc, bd);
            }
            return;
        }
    }

    for k in 0..2i32 {
        let p1 = read_clamped(plane, cx + k, cy - 2);
        let p0 = read_clamped(plane, cx + k, cy - 1);
        let q0 = read_clamped(plane, cx + k, cy);
        let q1 = read_clamped(plane, cx + k, cy + 1);
        let delta = ((((q0 - p0) << 2) + p1 - q1 + 4) >> 3).clamp(-tc, tc);
        write(plane, cx + k, cy - 1, p0 + delta, bd);
        write(plane, cx + k, cy, q0 - delta, bd);
    }
}

/// §8.8.3.6.9 chroma decision process (strong-filter eligibility) for
/// one decision row of an EDGE_VER edge. Returns true when the row
/// passes the strong-filter test (`dpq < β/4`, neighbour energy
/// `|p3-p0| + |q0-q3| < β/8`, edge magnitude `|p0-q0| < (5*tC + 1) >> 1`).
fn chroma_strong_decision_v(plane: &PicturePlane, cx: i32, cy: i32, beta: i32, tc: i32) -> bool {
    let p3 = read_clamped(plane, cx - 4, cy);
    let p2 = read_clamped(plane, cx - 3, cy);
    let p1 = read_clamped(plane, cx - 2, cy);
    let p0 = read_clamped(plane, cx - 1, cy);
    let q0 = read_clamped(plane, cx, cy);
    let q1 = read_clamped(plane, cx + 1, cy);
    let q2 = read_clamped(plane, cx + 2, cy);
    let q3 = read_clamped(plane, cx + 3, cy);
    let dp = (p2 - 2 * p1 + p0).abs();
    let dq = (q2 - 2 * q1 + q0).abs();
    let dpq = 2 * (dp + dq);
    let edge = (p3 - p0).abs() + (q0 - q3).abs();
    let centre = (p0 - q0).abs();
    dpq < (beta >> 2) && edge < (beta >> 3) && centre < (5 * tc + 1) >> 1
}

/// §8.8.3.6.9 mirror for an EDGE_HOR edge.
fn chroma_strong_decision_h(plane: &PicturePlane, cx: i32, cy: i32, beta: i32, tc: i32) -> bool {
    let p3 = read_clamped(plane, cx, cy - 4);
    let p2 = read_clamped(plane, cx, cy - 3);
    let p1 = read_clamped(plane, cx, cy - 2);
    let p0 = read_clamped(plane, cx, cy - 1);
    let q0 = read_clamped(plane, cx, cy);
    let q1 = read_clamped(plane, cx, cy + 1);
    let q2 = read_clamped(plane, cx, cy + 2);
    let q3 = read_clamped(plane, cx, cy + 3);
    let dp = (p2 - 2 * p1 + p0).abs();
    let dq = (q2 - 2 * q1 + q0).abs();
    let dpq = 2 * (dp + dq);
    let edge = (p3 - p0).abs() + (q0 - q3).abs();
    let centre = (p0 - q0).abs();
    dpq < (beta >> 2) && edge < (beta >> 3) && centre < (5 * tc + 1) >> 1
}

/// §8.8.3.6.10 strong chroma filter for a single sample row of an
/// EDGE_VER edge (eqs 1411 – 1416).
fn chroma_strong_apply_v(plane: &mut PicturePlane, cx: i32, cy: i32, tc: i32, bd: u32) {
    let p3 = read_clamped(plane, cx - 4, cy);
    let p2 = read_clamped(plane, cx - 3, cy);
    let p1 = read_clamped(plane, cx - 2, cy);
    let p0 = read_clamped(plane, cx - 1, cy);
    let q0 = read_clamped(plane, cx, cy);
    let q1 = read_clamped(plane, cx + 1, cy);
    let q2 = read_clamped(plane, cx + 2, cy);
    let q3 = read_clamped(plane, cx + 3, cy);
    let p0n = ((p3 + p2 + p1 + 2 * p0 + q0 + q1 + q2 + 4) >> 3).clamp(p0 - tc, p0 + tc);
    let p1n = ((2 * p3 + p2 + 2 * p1 + p0 + q0 + q1 + 4) >> 3).clamp(p1 - tc, p1 + tc);
    let p2n = ((3 * p3 + 2 * p2 + p1 + p0 + q0 + 4) >> 3).clamp(p2 - tc, p2 + tc);
    let q0n = ((p2 + p1 + p0 + 2 * q0 + q1 + q2 + q3 + 4) >> 3).clamp(q0 - tc, q0 + tc);
    let q1n = ((p1 + p0 + q0 + 2 * q1 + q2 + 2 * q3 + 4) >> 3).clamp(q1 - tc, q1 + tc);
    let q2n = ((p0 + q0 + q1 + 2 * q2 + 3 * q3 + 4) >> 3).clamp(q2 - tc, q2 + tc);
    write(plane, cx - 1, cy, p0n, bd);
    write(plane, cx - 2, cy, p1n, bd);
    write(plane, cx - 3, cy, p2n, bd);
    write(plane, cx, cy, q0n, bd);
    write(plane, cx + 1, cy, q1n, bd);
    write(plane, cx + 2, cy, q2n, bd);
}

/// Mirror of [`chroma_strong_apply_v`] for an EDGE_HOR edge.
fn chroma_strong_apply_h(plane: &mut PicturePlane, cx: i32, cy: i32, tc: i32, bd: u32) {
    let p3 = read_clamped(plane, cx, cy - 4);
    let p2 = read_clamped(plane, cx, cy - 3);
    let p1 = read_clamped(plane, cx, cy - 2);
    let p0 = read_clamped(plane, cx, cy - 1);
    let q0 = read_clamped(plane, cx, cy);
    let q1 = read_clamped(plane, cx, cy + 1);
    let q2 = read_clamped(plane, cx, cy + 2);
    let q3 = read_clamped(plane, cx, cy + 3);
    let p0n = ((p3 + p2 + p1 + 2 * p0 + q0 + q1 + q2 + 4) >> 3).clamp(p0 - tc, p0 + tc);
    let p1n = ((2 * p3 + p2 + 2 * p1 + p0 + q0 + q1 + 4) >> 3).clamp(p1 - tc, p1 + tc);
    let p2n = ((3 * p3 + 2 * p2 + p1 + p0 + q0 + 4) >> 3).clamp(p2 - tc, p2 + tc);
    let q0n = ((p2 + p1 + p0 + 2 * q0 + q1 + q2 + q3 + 4) >> 3).clamp(q0 - tc, q0 + tc);
    let q1n = ((p1 + p0 + q0 + 2 * q1 + q2 + 2 * q3 + 4) >> 3).clamp(q1 - tc, q1 + tc);
    let q2n = ((p0 + q0 + q1 + 2 * q2 + 3 * q3 + 4) >> 3).clamp(q2 - tc, q2 + tc);
    write(plane, cx, cy - 1, p0n, bd);
    write(plane, cx, cy - 2, p1n, bd);
    write(plane, cx, cy - 3, p2n, bd);
    write(plane, cx, cy, q0n, bd);
    write(plane, cx, cy + 1, q1n, bd);
    write(plane, cx, cy + 2, q2n, bd);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Table 43: spot-check β′ at the documented breakpoints (Q = 16
    /// → 6, Q = 32 → 26, Q = 50 → 62, Q = 63 → 88).
    #[test]
    fn beta_prime_table_anchors() {
        assert_eq!(beta_prime(15), 0);
        assert_eq!(beta_prime(16), 6);
        assert_eq!(beta_prime(32), 26);
        assert_eq!(beta_prime(50), 62);
        assert_eq!(beta_prime(63), 88);
        // Out-of-range clamps to boundaries.
        assert_eq!(beta_prime(-1), 0);
        assert_eq!(beta_prime(99), 88);
    }

    /// Table 43: spot-check tC′ at the documented breakpoints.
    #[test]
    fn tc_prime_table_anchors() {
        assert_eq!(tc_prime(17), 0);
        assert_eq!(tc_prime(18), 3);
        assert_eq!(tc_prime(32), 10);
        assert_eq!(tc_prime(50), 71);
        assert_eq!(tc_prime(65), 395);
    }

    /// Bit-depth scaling: at BitDepth = 8 both β and tC stay at the
    /// raw table value.
    #[test]
    fn bit_depth_scaling_is_identity_at_8bit() {
        assert_eq!(scale_beta_for_bit_depth(26, 8), 26);
        // tC: BitDepth 8 → (tc' + (1 << 1)) >> 2 = (tc' + 2) >> 2.
        assert_eq!(scale_tc_for_bit_depth(10, 8), (10 + 2) >> 2);
    }

    /// Disabling the deblocker leaves the picture untouched.
    #[test]
    fn disabled_does_not_modify_picture() {
        let mut buf = PictureBuffer::yuv420_filled(16, 16, 64);
        // Stripe a clear edge across the middle so any filtering would
        // be visible.
        for y in 0..16 {
            for x in 8..16 {
                buf.luma.samples[y * 16 + x] = 200;
            }
        }
        let snapshot = buf.luma.samples.clone();
        let cus = vec![
            DeblockCu {
                x: 0,
                y: 0,
                w: 8,
                h: 16,
                qp_y: 32,
                intra: true,
                tu_y_coded: true,
                tu_cb_coded: false,
                tu_cr_coded: false,
                bdpcm_luma: false,
                bdpcm_chroma: false,
            },
            DeblockCu {
                x: 8,
                y: 0,
                w: 8,
                h: 16,
                qp_y: 32,
                intra: true,
                tu_y_coded: true,
                tu_cb_coded: false,
                tu_cr_coded: false,
                bdpcm_luma: false,
                bdpcm_chroma: false,
            },
        ];
        let params = DeblockParams {
            disabled: true,
            bit_depth: 8,
            ..Default::default()
        };
        apply_deblocking(&mut buf, &cus, &params, 1);
        assert_eq!(buf.luma.samples, snapshot);
    }

    /// Two flat CUs joined at a small 100 ↔ 110 luma seam: the weak
    /// short filter must smooth the transition (samples right next to
    /// the seam move toward each other), and the picture-edge column
    /// must stay untouched. This exercises §8.8.3.5 bS = 2 (intra)
    /// + §8.8.3.6.7 weak filter — the |delta| < tC*10 gate kicks in
    /// for small jumps, which is the canonical "code-bug at high QP"
    /// scenario the deblocker is designed for.
    #[test]
    fn vertical_edge_smooths_weak_filter() {
        let mut buf = PictureBuffer::yuv420_filled(16, 16, 100);
        for y in 0..16 {
            for x in 8..16 {
                buf.luma.samples[y * 16 + x] = 110;
            }
        }
        let cus = vec![
            DeblockCu {
                x: 0,
                y: 0,
                w: 8,
                h: 16,
                qp_y: 32,
                intra: true,
                tu_y_coded: true,
                tu_cb_coded: false,
                tu_cr_coded: false,
                bdpcm_luma: false,
                bdpcm_chroma: false,
            },
            DeblockCu {
                x: 8,
                y: 0,
                w: 8,
                h: 16,
                qp_y: 32,
                intra: true,
                tu_y_coded: true,
                tu_cb_coded: false,
                tu_cr_coded: false,
                bdpcm_luma: false,
                bdpcm_chroma: false,
            },
        ];
        let params = DeblockParams {
            disabled: false,
            bit_depth: 8,
            ..Default::default()
        };
        apply_deblocking(&mut buf, &cus, &params, 1);
        let p0 = buf.luma.samples[8 * 16 + 7] as i32;
        let q0 = buf.luma.samples[8 * 16 + 8] as i32;
        // The weak filter pushes p0 toward q0 and vice versa.
        assert!(
            p0 > 100,
            "p0 sample on seam should move toward q-side, got {p0}"
        );
        assert!(
            q0 < 110,
            "q0 sample on seam should move toward p-side, got {q0}"
        );
        // Picture-edge column (x = 0) must be unchanged.
        assert_eq!(buf.luma.samples[8 * 16 + 0], 100);
        assert_eq!(buf.luma.samples[8 * 16 + 15], 110);
    }

    /// Same fixture but rotated: a horizontal edge must also smooth.
    #[test]
    fn horizontal_edge_smooths_weak_filter() {
        let mut buf = PictureBuffer::yuv420_filled(16, 16, 100);
        for y in 8..16 {
            for x in 0..16 {
                buf.luma.samples[y * 16 + x] = 110;
            }
        }
        let cus = vec![
            DeblockCu {
                x: 0,
                y: 0,
                w: 16,
                h: 8,
                qp_y: 32,
                intra: true,
                tu_y_coded: true,
                tu_cb_coded: false,
                tu_cr_coded: false,
                bdpcm_luma: false,
                bdpcm_chroma: false,
            },
            DeblockCu {
                x: 0,
                y: 8,
                w: 16,
                h: 8,
                qp_y: 32,
                intra: true,
                tu_y_coded: true,
                tu_cb_coded: false,
                tu_cr_coded: false,
                bdpcm_luma: false,
                bdpcm_chroma: false,
            },
        ];
        let params = DeblockParams {
            disabled: false,
            bit_depth: 8,
            ..Default::default()
        };
        apply_deblocking(&mut buf, &cus, &params, 1);
        let p0 = buf.luma.samples[7 * 16 + 8] as i32;
        let q0 = buf.luma.samples[8 * 16 + 8] as i32;
        assert!(p0 > 100, "p0 row above edge should smooth, got {p0}");
        assert!(q0 < 110, "q0 row below edge should smooth, got {q0}");
    }

    /// §8.8.3.5.5 luma maxFilterLength derivation: 4-tall CU forces
    /// the 1-tap path, ≥32 → 7-tap, in-between → 3-tap.
    #[test]
    fn luma_max_filter_length_v_branches() {
        let cu_small = DeblockCu {
            x: 0,
            y: 0,
            w: 4,
            h: 16,
            qp_y: 32,
            intra: true,
            tu_y_coded: true,
            tu_cb_coded: false,
            tu_cr_coded: false,
            bdpcm_luma: false,
            bdpcm_chroma: false,
        };
        let cu_medium = DeblockCu { w: 16, ..cu_small };
        let cu_large = DeblockCu { w: 32, ..cu_small };
        let (mp, mq) = luma_max_filter_length_v(&cu_small, &cu_medium);
        assert_eq!((mp, mq), (1, 3));
        let (mp, mq) = luma_max_filter_length_v(&cu_medium, &cu_large);
        assert_eq!((mp, mq), (3, 7));
        let (mp, mq) = luma_max_filter_length_v(&cu_large, &cu_large);
        assert_eq!((mp, mq), (7, 7));
    }

    /// §8.8.3.5.5 chroma maxFilterLength derivation: chroma TB ≥ 8 on
    /// both sides → (3, 3); otherwise (1, 1).
    #[test]
    fn chroma_max_filter_length_v_branches() {
        let cu_8 = DeblockCu {
            x: 0,
            y: 0,
            w: 8,
            h: 8,
            qp_y: 32,
            intra: true,
            tu_y_coded: true,
            tu_cb_coded: false,
            tu_cr_coded: false,
            bdpcm_luma: false,
            bdpcm_chroma: false,
        };
        let cu_16 = DeblockCu { w: 16, ..cu_8 };
        // 4:2:0: SubWidthC = 2 → chroma w = luma_w / 2.
        // 8 luma → 4 chroma → both < 8 → (1, 1).
        let (mp, mq) = chroma_max_filter_length_v(&cu_8, &cu_8, 2);
        assert_eq!((mp, mq), (1, 1));
        // 16 luma → 8 chroma → both ≥ 8 → (3, 3).
        let (mp, mq) = chroma_max_filter_length_v(&cu_16, &cu_16, 2);
        assert_eq!((mp, mq), (3, 3));
        // Asymmetric: one side small → (1, 1).
        let (mp, mq) = chroma_max_filter_length_v(&cu_8, &cu_16, 2);
        assert_eq!((mp, mq), (1, 1));
    }

    /// Long luma symmetric path: build two 32x16 CUs that meet on a
    /// vertical edge, with a clean 100/110 step. The long-tap filter
    /// must smooth the seam — and at this QP / size the long-block
    /// decision should pass on the flat run of samples.
    #[test]
    fn long_luma_filter_activates_for_32x16_cus() {
        let mut buf = PictureBuffer::yuv420_filled(64, 16, 100);
        for y in 0..16 {
            for x in 32..64 {
                buf.luma.samples[y * 64 + x] = 110;
            }
        }
        let cus = vec![
            DeblockCu {
                x: 0,
                y: 0,
                w: 32,
                h: 16,
                qp_y: 32,
                intra: true,
                tu_y_coded: true,
                tu_cb_coded: false,
                tu_cr_coded: false,
                bdpcm_luma: false,
                bdpcm_chroma: false,
            },
            DeblockCu {
                x: 32,
                y: 0,
                w: 32,
                h: 16,
                qp_y: 32,
                intra: true,
                tu_y_coded: true,
                tu_cb_coded: false,
                tu_cr_coded: false,
                bdpcm_luma: false,
                bdpcm_chroma: false,
            },
        ];
        let params = DeblockParams {
            disabled: false,
            bit_depth: 8,
            ..Default::default()
        };
        apply_deblocking(&mut buf, &cus, &params, 1);
        let p0 = buf.luma.samples[8 * 64 + 31] as i32;
        let q0 = buf.luma.samples[8 * 64 + 32] as i32;
        // Either the long or short filter applied; both must move the
        // seam toward each other.
        assert!(p0 > 100, "p0 should smooth toward q-side, got {p0}");
        assert!(q0 < 110, "q0 should smooth toward p-side, got {q0}");
        // Far-from-edge samples remain at 100/110 (long-tap touches up
        // to 7 samples deep but with a strong centre weighting; the
        // boundary samples at x = 0 and x = 63 stay clamped).
        assert_eq!(buf.luma.samples[8 * 64 + 0], 100);
        assert_eq!(buf.luma.samples[8 * 64 + 63], 110);
    }

    /// `bS = 0` (no neighbour) → no modification anywhere.
    #[test]
    fn no_neighbour_skips_filtering() {
        let mut buf = PictureBuffer::yuv420_filled(8, 8, 100);
        let snapshot = buf.luma.samples.clone();
        let cus = vec![DeblockCu {
            x: 0,
            y: 0,
            w: 8,
            h: 8,
            qp_y: 32,
            intra: true,
            tu_y_coded: true,
            tu_cb_coded: false,
            tu_cr_coded: false,
            bdpcm_luma: false,
            bdpcm_chroma: false,
        }];
        let params = DeblockParams {
            disabled: false,
            bit_depth: 8,
            ..Default::default()
        };
        apply_deblocking(&mut buf, &cus, &params, 1);
        assert_eq!(buf.luma.samples, snapshot);
    }
}
