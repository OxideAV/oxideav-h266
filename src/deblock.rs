//! VVC in-loop deblocking filter (§8.8.3).
//!
//! Implements the §8.8.3 deblocking filter — vertical-then-horizontal
//! edges per CTU on a CU basis:
//!
//! * §8.8.3.2 — per-direction loop, gated by
//!   `sh_deblocking_filter_disabled_flag`.
//! * §8.8.3.3 — transform-block edge identification: the walker emits
//!   one TB per CU, so every CU rectangle's outer edges are TB edges.
//!   The luma `maxFilterLength{P,Q}` derivation (either side ≤ 4 →
//!   both 1; per side ≥ 32 → 7; else 3) and the chroma derivation
//!   (both ≥ 8 → 3, with the CTB-row P-cap) live here. Sub-block
//!   edges (§8.8.3.4) are skipped — no SBT / affine sub-block edge
//!   records yet.
//! * §8.8.3.5 — boundary-strength derivation with
//!   `intra ? 2 : (tu_coded ? 1 : 0)`.
//! * §8.8.3.6.2 — the full luma decision (steps 1 – 9): short-block
//!   dE = 0/1/2 selection, the large-block dSam derivation with the
//!   eqs. 1290 – 1309 metrics, and the step-6 luma CTB-row rule
//!   (EDGE_HOR on `yEdge % CtbSizeY == 0` forces `sidePisLargeBlk = 0`
//!   → eq. 1294 caps `maxFilterLengthP` at 3, turning the boundary
//!   into an asymmetric long filter).
//! * §8.8.3.6.3 — luma filter dispatch (short for dE = 1/2, long for
//!   dE = 3).
//! * §8.8.3.6.4 / §8.8.3.6.9 / §8.8.3.6.10 — chroma decision +
//!   weak/strong filter for cIdx = 1 / 2, including the asymmetric
//!   (1, 3) CTB-row variant.
//! * §8.8.3.6.6 — the per-sample decision with both threshold sets
//!   (eqs. 1369 – 1374).
//! * §8.8.3.6.7 — short luma sample filter (`dE = 1` weak / `dE = 2`
//!   strong + `dEp` / `dEq` for p1 / q1).
//! * §8.8.3.6.8 — long luma sample filter: all refMiddle arms
//!   (symmetric eqs. 1389/1390 **and** asymmetric eqs. 1391 – 1394)
//!   with the 7-/5-/3-deep `fi`/`tCPDi` arrays (eqs. 1397 – 1408).
//!
//! Out of scope (each gated as a no-op so the deblock pass still runs
//! on un-tested edges):
//!
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
    /// `CtbLog2SizeY` — the §8.8.3.3 chroma `maxFilterLength` derivation
    /// caps the P side to 1 on horizontal edges that coincide with a
    /// chroma-CTB row boundary (r415).
    pub ctb_log2_size_y: u32,
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
    /// `CtbSizeY` in luma samples (for the §8.8.3.3 chroma CTB-row rule).
    ctb_size_y: u32,
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
        ctb_size_y: 1 << params.ctb_log2_size_y,
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
            ctb_size_y: 1 << params.ctb_log2_size_y,
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
            ctb_size_y: 1 << params.ctb_log2_size_y,
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
            // §8.8.3.6.6 + §8.8.3.6.8 (long). EDGE_VER never triggers
            // the §8.8.3.6.2 step-6 CTB-row rule.
            let (_qp, beta, tc) = compute_thresholds_luma(plane, p_cu, q_cu, b_s);
            let (mfl_p, mfl_q) = luma_max_filter_length_v(p_cu, q_cu);
            run_luma_filter(plane.plane, cx, cy + k, beta, tc, mfl_p, mfl_q, true, false);
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
            // §8.8.3.6.2 step 6 — an EDGE_HOR edge on a luma CTB row
            // boundary suppresses sidePisLargeBlk (the P side would
            // reach into the CTB line buffer above).
            let ctb_row_edge = plane.ctb_size_y > 0 && (cy as u32) % plane.ctb_size_y == 0;
            run_luma_filter(
                plane.plane,
                cx + k,
                cy,
                beta,
                tc,
                mfl_p,
                mfl_q,
                false,
                ctb_row_edge,
            );
            k += luma_segment / (plane.sub_w as i32).max(1);
        } else {
            let (_qp, beta, tc) = compute_thresholds_chroma(plane, p_cu, q_cu, b_s);
            let ctb_h_c = plane.ctb_size_y / plane.sub_h.max(1);
            let (mfl_p, mfl_q) = chroma_max_filter_length_h(p_cu, q_cu, plane.sub_h, cy, ctb_h_c);
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
/// uses the chroma TB *height* on each side. §8.8.3.3 (r415): when the
/// horizontal edge coincides with a chroma CTB row boundary
/// (`(yCb + y) % CtbHeightC == 0`), the P side reaches into the CTB row
/// above (the decoder's line buffer), so `maxFilterLengthP` is capped
/// at 1 while `maxFilterLengthQ` stays 3.
#[inline]
fn chroma_max_filter_length_h(
    p_cu: &DeblockCu,
    q_cu: &DeblockCu,
    sub_h: u32,
    edge_chroma_y: i32,
    ctb_h_c: u32,
) -> (u32, u32) {
    let p_chroma_h = p_cu.h / sub_h.max(1);
    let q_chroma_h = q_cu.h / sub_h.max(1);
    if p_chroma_h >= 8 && q_chroma_h >= 8 {
        if ctb_h_c > 0 && (edge_chroma_y as u32) % ctb_h_c == 0 {
            (1, 3)
        } else {
            (3, 3)
        }
    } else {
        (1, 1)
    }
}

/// §8.8.3.3 luma `maxFilterLength{P,Q}` derivation for an EDGE_VER
/// luma edge (`cIdx == 0`):
///
/// * If **either** adjacent TB's width is ≤ 4 → **both**
///   `maxFilterLengthQ` and `maxFilterLengthP` are 1 (the spec's Q and
///   P rules each test both sides).
/// * Otherwise each side derives independently: TB width ≥ 32 on that
///   side → 7, else 3.
///
/// The walker treats one CU = one TB (multi-TB tiling only occurs
/// above 64 samples, where every tile is ≥ 32), so the luma TB width
/// on the P side is `p_cu.w` and on the Q side is `q_cu.w`.
#[inline]
fn luma_max_filter_length_v(p_cu: &DeblockCu, q_cu: &DeblockCu) -> (u32, u32) {
    if p_cu.w <= 4 || q_cu.w <= 4 {
        return (1, 1);
    }
    let mfl_p = if p_cu.w >= 32 { 7 } else { 3 };
    let mfl_q = if q_cu.w >= 32 { 7 } else { 3 };
    (mfl_p, mfl_q)
}

/// Mirror of [`luma_max_filter_length_v`] for an EDGE_HOR luma edge —
/// uses CU heights instead of widths.
#[inline]
fn luma_max_filter_length_h(p_cu: &DeblockCu, q_cu: &DeblockCu) -> (u32, u32) {
    if p_cu.h <= 4 || q_cu.h <= 4 {
        return (1, 1);
    }
    let mfl_p = if p_cu.h >= 32 { 7 } else { 3 };
    let mfl_q = if q_cu.h >= 32 { 7 } else { 3 };
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

/// The §8.8.3.6.2 outputs for one 4-sample luma edge segment: the
/// filtering decision `dE` (0 = off, 1 = weak, 2 = strong-short,
/// 3 = long), the `dEp` / `dEq` p1/q1-filtering decisions, and the
/// (possibly modified) `maxFilterLength{P,Q}` the §8.8.3.6.3 filter
/// dispatch consumes.
struct LumaEdgeDecision {
    d_e: u32,
    d_ep: u32,
    d_eq: u32,
    mfl_p: u32,
    mfl_q: u32,
}

/// §8.8.3.6.2 decision process for one luma edge segment (steps 1–9),
/// shared between EDGE_VER (`is_vertical`) and EDGE_HOR. `(cx, cy)`
/// is the segment anchor: the edge runs between columns `cx-1 | cx`
/// (vertical) or rows `cy-1 | cy` (horizontal). `ctb_row_edge` is the
/// step-6 input — true iff the edge is EDGE_HOR and
/// `(yCb + yBl) % CtbSizeY == 0` (the P side would reach into the CTB
/// row above, i.e. the decoder's line buffer, so `sidePisLargeBlk` is
/// forced to 0 and eq. 1294 caps `maxFilterLengthP` at 3).
#[allow(clippy::too_many_arguments)]
fn luma_edge_decision(
    plane: &PicturePlane,
    cx: i32,
    cy: i32,
    beta: i32,
    tc: i32,
    mut mfl_p: u32,
    mut mfl_q: u32,
    is_vertical: bool,
    ctb_row_edge: bool,
) -> LumaEdgeDecision {
    // Sample fetch: p_i,k / q_j,k per eqs. 1268–1271.
    let read_p = |i: i32, k: i32| -> i32 {
        if is_vertical {
            read_clamped(plane, cx - i - 1, cy + k)
        } else {
            read_clamped(plane, cx + k, cy - i - 1)
        }
    };
    let read_q = |j: i32, k: i32| -> i32 {
        if is_vertical {
            read_clamped(plane, cx + j, cy + k)
        } else {
            read_clamped(plane, cx + k, cy + j)
        }
    };

    // Step 1 — eqs. 1280–1283.
    let dp0 = (read_p(2, 0) - 2 * read_p(1, 0) + read_p(0, 0)).abs();
    let dp3 = (read_p(2, 3) - 2 * read_p(1, 3) + read_p(0, 3)).abs();
    let dq0 = (read_q(2, 0) - 2 * read_q(1, 0) + read_q(0, 0)).abs();
    let dq3 = (read_q(2, 3) - 2 * read_q(1, 3) + read_q(0, 3)).abs();

    // Step 2 — eqs. 1284–1289 (defined when both mfl >= 3; the values
    // are only consumed on branches the spec gates accordingly).
    let sp0 = (read_p(3, 0) - read_p(0, 0)).abs();
    let sq0 = (read_q(0, 0) - read_q(3, 0)).abs();
    let spq0 = (read_p(0, 0) - read_q(0, 0)).abs();
    let sp3 = (read_p(3, 3) - read_p(0, 3)).abs();
    let sq3 = (read_q(0, 3) - read_q(3, 3)).abs();
    let spq3 = (read_p(0, 3) - read_q(0, 3)).abs();

    // Steps 3–6 — side*isLargeBlk, including the EDGE_HOR CTB-row
    // suppression (step 6).
    let mut side_p_large = mfl_p > 3;
    let side_q_large = mfl_q > 3;
    if ctb_row_edge {
        side_p_large = false;
    }

    // Steps 7–8 — the large-block decision (dSam0 / dSam3).
    let mut d_sam0 = false;
    let mut d_sam3 = false;
    if side_p_large || side_q_large {
        // 8.a — eqs. 1290–1294. When the P side is not large the spec
        // caps maxFilterLengthP at 3 (eq. 1294) — this is what turns a
        // 7-deep P side into the asymmetric (3, 7) long filter at a
        // luma CTB row boundary.
        let (dp0_l, dp3_l) = if side_p_large {
            (
                (dp0 + (read_p(5, 0) - 2 * read_p(4, 0) + read_p(3, 0)).abs() + 1) >> 1,
                (dp3 + (read_p(5, 3) - 2 * read_p(4, 3) + read_p(3, 3)).abs() + 1) >> 1,
            )
        } else {
            mfl_p = 3;
            (dp0, dp3)
        };
        // 8.b — eqs. 1295–1298.
        let (dq0_l, dq3_l) = if side_q_large {
            (
                (dq0 + (read_q(5, 0) - 2 * read_q(4, 0) + read_q(3, 0)).abs() + 1) >> 1,
                (dq3 + (read_q(5, 3) - 2 * read_q(4, 3) + read_q(3, 3)).abs() + 1) >> 1,
            )
        } else {
            (dq0, dq3)
        };
        // 8.c — eqs. 1299–1302 (uses the *modified* maxFilterLengthP).
        let (sp0_l, sp3_l) = if mfl_p == 7 {
            (
                sp0 + (read_p(7, 0) - read_p(6, 0) - read_p(5, 0) + read_p(4, 0)).abs(),
                sp3 + (read_p(7, 3) - read_p(6, 3) - read_p(5, 3) + read_p(4, 3)).abs(),
            )
        } else {
            (sp0, sp3)
        };
        // 8.d — eqs. 1303–1306.
        let (sq0_l, sq3_l) = if mfl_q == 7 {
            (
                sq0 + (read_q(4, 0) - read_q(5, 0) - read_q(6, 0) + read_q(7, 0)).abs(),
                sq3 + (read_q(4, 3) - read_q(5, 3) - read_q(6, 3) + read_q(7, 3)).abs(),
            )
        } else {
            (sq0, sq3)
        };
        // 8.e — eqs. 1307–1309.
        let dl = (dp0_l + dq0_l) + (dp3_l + dq3_l);
        // 8.f — per-row §8.8.3.6.6 invocations with the eqs. 1310–1317
        // p0/p3/q0/q3 picks (0 on a non-large side).
        if dl < beta {
            let row_decision = |k: i32, dpq_l: i32, sp_l: i32, sq_l: i32, spq: i32| -> bool {
                // §8.8.3.6.6 eqs. 1369/1370 — the sp/sq widening uses
                // the picked p3 = p_3,k / p0 = p_mflP,k samples.
                let sp_in = if side_p_large {
                    (sp_l + (read_p(3, k) - read_p(mfl_p as i32, k)).abs() + 1) >> 1
                } else {
                    sp_l
                };
                let sq_in = if side_q_large {
                    (sq_l + (read_q(3, k) - read_q(mfl_q as i32, k)).abs() + 1) >> 1
                } else {
                    sq_l
                };
                // eqs. 1371/1372 — large-block thresholds (at least one
                // side is large on this branch).
                let s_thr1 = (3 * beta) >> 5;
                let s_thr2 = beta >> 4;
                2 * dpq_l < s_thr2 && sp_in + sq_in < s_thr1 && spq < (5 * tc + 1) >> 1
            };
            d_sam0 = row_decision(0, dp0_l + dq0_l, sp0_l, sq0_l, spq0);
            d_sam3 = row_decision(3, dp3_l + dq3_l, sp3_l, sq3_l, spq3);
        }
    }

    // Step 9 — final dE / dEp / dEq.
    if d_sam0 && d_sam3 {
        return LumaEdgeDecision {
            d_e: 3,
            d_ep: 1,
            d_eq: 1,
            mfl_p,
            mfl_q,
        };
    }
    // 9.a — eqs. 1318–1322 (short-block metrics).
    let dpq0 = dp0 + dq0;
    let dpq3 = dp3 + dq3;
    let dp = dp0 + dp3;
    let dq = dq0 + dq3;
    let d = dpq0 + dpq3;
    // 9.b — reset.
    let mut d_e = 0;
    let mut d_ep = 0;
    let mut d_eq = 0;
    // 9.c — the strong-short decision runs only when d < β and both
    // maxFilterLengths exceed 2 (§8.8.3.6.6 with all of p0/p3/q0/q3 =
    // 0 and the small-block eqs. 1373/1374 thresholds).
    let mut s_sam0 = false;
    let mut s_sam3 = false;
    if d < beta && mfl_p > 2 && mfl_q > 2 {
        let s_thr1 = beta >> 3;
        let s_thr2 = beta >> 2;
        let short_row = |dpq: i32, sp: i32, sq: i32, spq: i32| -> bool {
            2 * dpq < s_thr2 && sp + sq < s_thr1 && spq < (5 * tc + 1) >> 1
        };
        s_sam0 = short_row(dpq0, sp0, sq0, spq0);
        s_sam3 = short_row(dpq3, sp3, sq3, spq3);
    }
    // 9.d — dE/dEp/dEq + the final maxFilterLength folds.
    if d < beta {
        d_e = 1;
        if s_sam0 && s_sam3 {
            d_e = 2;
            mfl_p = 3;
            mfl_q = 3;
        }
        if mfl_p > 1 && mfl_q > 1 {
            if dp < (beta + (beta >> 1)) >> 3 {
                d_ep = 1;
            }
            if dq < (beta + (beta >> 1)) >> 3 {
                d_eq = 1;
            }
        }
        if d_e == 1 {
            mfl_p = 1 + d_ep;
            mfl_q = 1 + d_eq;
        }
    }
    LumaEdgeDecision {
        d_e,
        d_ep,
        d_eq,
        mfl_p,
        mfl_q,
    }
}

/// §8.8.3.6.1 / §8.8.3.6.3 luma dispatch for one 4-sample edge segment:
/// run the §8.8.3.6.2 decision, then apply the §8.8.3.6.7 short filter
/// (dE = 1/2) or the §8.8.3.6.8 long filter (dE = 3) to each of the 4
/// sample lines. `dE = 0` leaves the segment untouched.
#[allow(clippy::too_many_arguments)]
fn run_luma_filter(
    plane: &mut PicturePlane,
    cx: i32,
    cy: i32,
    beta: i32,
    tc: i32,
    mfl_p: u32,
    mfl_q: u32,
    is_vertical: bool,
    ctb_row_edge: bool,
) {
    if tc == 0 {
        // Every filter arm clips its update into [x − c·tC, x + c·tC];
        // tC = 0 makes the whole segment a no-op.
        return;
    }
    let dec = luma_edge_decision(
        plane,
        cx,
        cy,
        beta,
        tc,
        mfl_p,
        mfl_q,
        is_vertical,
        ctb_row_edge,
    );
    match dec.d_e {
        0 => {}
        3 => {
            for k in 0..4i32 {
                if is_vertical {
                    long_luma_apply(plane, cx, cy + k, tc, dec.mfl_p, dec.mfl_q, true);
                } else {
                    long_luma_apply(plane, cx + k, cy, tc, dec.mfl_p, dec.mfl_q, false);
                }
            }
        }
        _ => {
            for k in 0..4i32 {
                if is_vertical {
                    short_luma_apply(plane, cx, cy + k, tc, dec.d_e, dec.d_ep, dec.d_eq, true);
                } else {
                    short_luma_apply(plane, cx + k, cy, tc, dec.d_e, dec.d_ep, dec.d_eq, false);
                }
            }
        }
    }
}

/// §8.8.3.6.8 long luma filter for one sample line. Implements the
/// refMiddle derivation for every `maxFilterLength{P,Q}` combination —
/// symmetric eqs. 1389 (5/5) and 1390 (7/7) plus the asymmetric
/// eqs. 1391 (7/5, 5/7), 1392 (5/3, 3/5), 1393 (P=3, Q=7) and 1394
/// (P=7, Q=3) — the eqs. 1395/1396 refP/refQ, the per-side
/// `fi`/`gj`/`tCPDi`/`tCQDj` arrays (eqs. 1397–1408, including the
/// 3-deep {53, 32, 11} / {6, 4, 2} pair), and the eqs. 1409/1410
/// clipped updates.
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
    let mut p = [0i32; 8];
    let mut q = [0i32; 8];
    for (i, slot) in p.iter_mut().enumerate().take(mfl_p_u + 1) {
        *slot = if is_vertical {
            read_clamped(plane, cx - i as i32 - 1, cy)
        } else {
            read_clamped(plane, cx, cy - i as i32 - 1)
        };
    }
    for (j, slot) in q.iter_mut().enumerate().take(mfl_q_u + 1) {
        *slot = if is_vertical {
            read_clamped(plane, cx + j as i32, cy)
        } else {
            read_clamped(plane, cx, cy + j as i32)
        };
    }

    // refMiddle — eqs. 1389–1394 keyed on the (P, Q) length pair.
    let ref_middle = match (mfl_p, mfl_q) {
        // eq. 1389 — 5/5.
        (5, 5) => {
            (p[4] + p[3] + 2 * (p[2] + p[1] + p[0] + q[0] + q[1] + q[2]) + q[3] + q[4] + 8) >> 4
        }
        // eq. 1390 — equal lengths other than 5 (7/7 and 3/3).
        (a, b) if a == b => {
            (p[6]
                + p[5]
                + p[4]
                + p[3]
                + p[2]
                + p[1]
                + 2 * (p[0] + q[0])
                + q[1]
                + q[2]
                + q[3]
                + q[4]
                + q[5]
                + q[6]
                + 8)
                >> 4
        }
        // eq. 1391 — {7,5} / {5,7}.
        (7, 5) | (5, 7) => {
            (p[5]
                + p[4]
                + p[3]
                + p[2]
                + 2 * (p[1] + p[0] + q[0] + q[1])
                + q[2]
                + q[3]
                + q[4]
                + q[5]
                + 8)
                >> 4
        }
        // eq. 1392 — {5,3} / {3,5}.
        (5, 3) | (3, 5) => (p[3] + p[2] + p[1] + p[0] + q[0] + q[1] + q[2] + q[3] + 4) >> 3,
        // eq. 1393 — P = 3, Q = 7.
        (3, 7) => {
            (2 * (p[2] + p[1] + p[0] + q[0])
                + p[0]
                + p[1]
                + q[1]
                + q[2]
                + q[3]
                + q[4]
                + q[5]
                + q[6]
                + 8)
                >> 4
        }
        // eq. 1394 — P = 7, Q = 3 (the remaining arm).
        _ => {
            (p[6]
                + p[5]
                + p[4]
                + p[3]
                + p[2]
                + p[1]
                + 2 * (q[2] + q[1] + q[0] + p[0])
                + q[0]
                + q[1]
                + 8)
                >> 4
        }
    };

    // refP / refQ (eqs. 1395 / 1396).
    let ref_p = (p[mfl_p_u] + p[mfl_p_u - 1] + 1) >> 1;
    let ref_q = (q[mfl_q_u] + q[mfl_q_u - 1] + 1) >> 1;

    // fi / tCPDi and gj / tCQDj (eqs. 1397–1408).
    const F7: [i32; 7] = [59, 50, 41, 32, 23, 14, 5];
    const F5: [i32; 5] = [58, 45, 32, 19, 6];
    const F3: [i32; 3] = [53, 32, 11];
    const T7: [i32; 7] = [6, 5, 4, 3, 2, 1, 1];
    const T5: [i32; 5] = [6, 5, 4, 3, 2];
    const T3: [i32; 3] = [6, 4, 2];
    let (fi, tcpdi): (&[i32], &[i32]) = match mfl_p {
        7 => (&F7, &T7),
        5 => (&F5, &T5),
        _ => (&F3, &T3),
    };
    let (gj, tcqdj): (&[i32], &[i32]) = match mfl_q {
        7 => (&F7, &T7),
        5 => (&F5, &T5),
        _ => (&F3, &T3),
    };

    // Eqs. 1409 / 1410 — write filtered samples.
    for i in 0..mfl_p_u {
        let lo = p[i] - ((tc * tcpdi[i]) >> 1);
        let hi = p[i] + ((tc * tcpdi[i]) >> 1);
        let v = ((ref_middle * fi[i] + ref_p * (64 - fi[i]) + 32) >> 6).clamp(lo, hi);
        if is_vertical {
            write(plane, cx - i as i32 - 1, cy, v, bd);
        } else {
            write(plane, cx, cy - i as i32 - 1, v, bd);
        }
    }
    for j in 0..mfl_q_u {
        let lo = q[j] - ((tc * tcqdj[j]) >> 1);
        let hi = q[j] + ((tc * tcqdj[j]) >> 1);
        let v = ((ref_middle * gj[j] + ref_q * (64 - gj[j]) + 32) >> 6).clamp(lo, hi);
        if is_vertical {
            write(plane, cx + j as i32, cy, v, bd);
        } else {
            write(plane, cx, cy + j as i32, v, bd);
        }
    }
}

/// §8.8.3.6.7 short luma filter for one sample line: strong filtering
/// (eqs. 1375–1380) when `dE == 2`, weak filtering (eqs. 1381–1388,
/// with the `|Δ| < tC·10` gate and the `dEp` / `dEq` p1/q1 arms)
/// otherwise.
#[allow(clippy::too_many_arguments)]
fn short_luma_apply(
    plane: &mut PicturePlane,
    cx: i32,
    cy: i32,
    tc: i32,
    d_e: u32,
    d_ep: u32,
    d_eq: u32,
    is_vertical: bool,
) {
    let bd = 8u32;
    let read_p = |i: i32| -> i32 {
        if is_vertical {
            read_clamped(plane, cx - i - 1, cy)
        } else {
            read_clamped(plane, cx, cy - i - 1)
        }
    };
    let read_q = |j: i32| -> i32 {
        if is_vertical {
            read_clamped(plane, cx + j, cy)
        } else {
            read_clamped(plane, cx, cy + j)
        }
    };
    let p3 = read_p(3);
    let p2 = read_p(2);
    let p1 = read_p(1);
    let p0 = read_p(0);
    let q0 = read_q(0);
    let q1 = read_q(1);
    let q2 = read_q(2);
    let q3 = read_q(3);
    let write_p = |i: i32, v: i32, plane: &mut PicturePlane| {
        if is_vertical {
            write(plane, cx - i - 1, cy, v, bd);
        } else {
            write(plane, cx, cy - i - 1, v, bd);
        }
    };
    if d_e == 2 {
        // Strong filter — eqs. 1375–1380.
        let p0n = ((p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + 4) >> 3).clamp(p0 - 3 * tc, p0 + 3 * tc);
        let p1n = ((p2 + p1 + p0 + q0 + 2) >> 2).clamp(p1 - 2 * tc, p1 + 2 * tc);
        let p2n = ((2 * p3 + 3 * p2 + p1 + p0 + q0 + 4) >> 3).clamp(p2 - tc, p2 + tc);
        let q0n = ((p1 + 2 * p0 + 2 * q0 + 2 * q1 + q2 + 4) >> 3).clamp(q0 - 3 * tc, q0 + 3 * tc);
        let q1n = ((p0 + q0 + q1 + q2 + 2) >> 2).clamp(q1 - 2 * tc, q1 + 2 * tc);
        let q2n = ((p0 + q0 + q1 + 3 * q2 + 2 * q3 + 4) >> 3).clamp(q2 - tc, q2 + tc);
        write_p(0, p0n, plane);
        write_p(1, p1n, plane);
        write_p(2, p2n, plane);
        if is_vertical {
            write(plane, cx, cy, q0n, bd);
            write(plane, cx + 1, cy, q1n, bd);
            write(plane, cx + 2, cy, q2n, bd);
        } else {
            write(plane, cx, cy, q0n, bd);
            write(plane, cx, cy + 1, q1n, bd);
            write(plane, cx, cy + 2, q2n, bd);
        }
    } else {
        // Weak filter — eqs. 1381–1388.
        let delta_raw = (9 * (q0 - p0) - 3 * (q1 - p1) + 8) >> 4;
        if delta_raw.abs() < tc * 10 {
            let delta = delta_raw.clamp(-tc, tc);
            write_p(0, p0 + delta, plane);
            if is_vertical {
                write(plane, cx, cy, q0 - delta, bd);
            } else {
                write(plane, cx, cy, q0 - delta, bd);
            }
            if d_ep == 1 {
                // Eq. 1385 — the clip bound is −(tC >> 1), NOT
                // (−tC) >> 1: the two differ by 1 for odd tC (the
                // arithmetic right-shift rounds toward −∞).
                let dp = ((((p2 + p0 + 1) >> 1) - p1 + delta) >> 1).clamp(-(tc >> 1), tc >> 1);
                write_p(1, p1 + dp, plane);
            }
            if d_eq == 1 {
                // Eq. 1387 — same −(tC >> 1) bound.
                let dq = ((((q2 + q0 + 1) >> 1) - q1 - delta) >> 1).clamp(-(tc >> 1), tc >> 1);
                if is_vertical {
                    write(plane, cx + 1, cy, q1 + dq, bd);
                } else {
                    write(plane, cx, cy + 1, q1 + dq, bd);
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

    // §8.8.3.6.4: with both lengths 1 and bS != 2 the edge is left
    // untouched entirely.
    if max_filter_length_p == 1 && max_filter_length_q == 1 && _b_s != 2 {
        return;
    }
    // §8.8.3.6.4 / §8.8.3.6.10 — the strong path runs when
    // maxFilterLengthQ == 3; the P side is either 3 (symmetric
    // eqs. 1411-1416) or 1 (the r415 chroma-CTB-row asymmetric
    // variant, eqs. 1417-1420, with the decision's p3/p2 := p1
    // substitution).
    let strong_eligible =
        max_filter_length_q == 3 && (max_filter_length_p == 3 || max_filter_length_p == 1);
    if strong_eligible {
        let short_p = max_filter_length_p == 1;
        let dec0 = chroma_strong_decision_h(plane, cx, cy, beta, tc, short_p);
        let dec1 = chroma_strong_decision_h(plane, cx + max_k, cy, beta, tc, short_p);
        if dec0 && dec1 {
            for k in 0..=max_k {
                chroma_strong_apply_h(plane, cx + k, cy, tc, bd, short_p);
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

/// §8.8.3.6.9 mirror for an EDGE_HOR edge. With `short_p` (the r415
/// maxFilterLengthP == 1 CTB-row case) §8.8.3.6.4 step 2 substitutes
/// `p3 = p2 = p1` before the decision math.
fn chroma_strong_decision_h(
    plane: &PicturePlane,
    cx: i32,
    cy: i32,
    beta: i32,
    tc: i32,
    short_p: bool,
) -> bool {
    let p1 = read_clamped(plane, cx, cy - 2);
    let p3 = if short_p {
        p1
    } else {
        read_clamped(plane, cx, cy - 4)
    };
    let p2 = if short_p {
        p1
    } else {
        read_clamped(plane, cx, cy - 3)
    };
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

/// Mirror of [`chroma_strong_apply_v`] for an EDGE_HOR edge. With
/// `short_p` the §8.8.3.6.10 asymmetric (P = 1, Q = 3) filter applies
/// (eqs. 1417 - 1420): only p0 is modified on the P side.
fn chroma_strong_apply_h(
    plane: &mut PicturePlane,
    cx: i32,
    cy: i32,
    tc: i32,
    bd: u32,
    short_p: bool,
) {
    let p1 = read_clamped(plane, cx, cy - 2);
    let p0 = read_clamped(plane, cx, cy - 1);
    let q0 = read_clamped(plane, cx, cy);
    let q1 = read_clamped(plane, cx, cy + 1);
    let q2 = read_clamped(plane, cx, cy + 2);
    let q3 = read_clamped(plane, cx, cy + 3);
    if short_p {
        let p0n = ((3 * p1 + 2 * p0 + q0 + q1 + q2 + 4) >> 3).clamp(p0 - tc, p0 + tc);
        let q0n = ((2 * p1 + p0 + 2 * q0 + q1 + q2 + q3 + 4) >> 3).clamp(q0 - tc, q0 + tc);
        let q1n = ((p1 + p0 + q0 + 2 * q1 + q2 + 2 * q3 + 4) >> 3).clamp(q1 - tc, q1 + tc);
        let q2n = ((p0 + q0 + q1 + 2 * q2 + 3 * q3 + 4) >> 3).clamp(q2 - tc, q2 + tc);
        write(plane, cx, cy - 1, p0n, bd);
        write(plane, cx, cy, q0n, bd);
        write(plane, cx, cy + 1, q1n, bd);
        write(plane, cx, cy + 2, q2n, bd);
        return;
    }
    let p3 = read_clamped(plane, cx, cy - 4);
    let p2 = read_clamped(plane, cx, cy - 3);
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

    /// §8.8.3.3 luma maxFilterLength derivation: **either** side ≤ 4
    /// forces both to 1; ≥32 → 7 per side; in-between → 3.
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
        // A ≤4-wide TB on either side caps BOTH sides at 1 (the spec's
        // Q and P rules each test both adjacent TB widths).
        let (mp, mq) = luma_max_filter_length_v(&cu_small, &cu_medium);
        assert_eq!((mp, mq), (1, 1));
        let (mp, mq) = luma_max_filter_length_v(&cu_large, &cu_small);
        assert_eq!((mp, mq), (1, 1));
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

    /// Convenience: an intra CU record with coded luma.
    fn cu(x: u32, y: u32, w: u32, h: u32, qp_y: i32) -> DeblockCu {
        DeblockCu {
            x,
            y,
            w,
            h,
            qp_y,
            intra: true,
            tu_y_coded: true,
            tu_cb_coded: false,
            tu_cr_coded: false,
            bdpcm_luma: false,
            bdpcm_chroma: false,
        }
    }

    /// Eqs. 1385 / 1387 — the weak-filter p1/q1 clip bound is
    /// `−(tC >> 1)`, not `(−tC) >> 1`: for odd tC the arithmetic shift
    /// rounds toward −∞ and over-widens the bound by 1 (the r418
    /// qp45-corpus divergence root cause). QP 45 / bS 2 gives tC = 13;
    /// a −140→100 step with flat sides drives dE = 1 with dEp = dEq =
    /// 1 and a p1 delta that saturates the clip: p1′ must be
    /// 140 − 6 = 134 (the buggy bound gave 133).
    #[test]
    fn weak_filter_p1_clip_bound_is_neg_of_half_tc() {
        let mut buf = PictureBuffer::yuv420_filled(32, 8, 128);
        for y in 0..8 {
            for x in 0..16 {
                buf.luma.samples[y * 32 + x] = 140;
            }
            for x in 16..32 {
                buf.luma.samples[y * 32 + x] = 100;
            }
        }
        let cus = vec![cu(0, 0, 16, 8, 45), cu(16, 0, 16, 8, 45)];
        let params = DeblockParams {
            disabled: false,
            bit_depth: 8,
            ctb_log2_size_y: 7,
            ..Default::default()
        };
        apply_deblocking(&mut buf, &cus, &params, 1);
        // Spec values (β = 52, tC = 13, Δ = −13): p0′ = 127, q0′ = 113,
        // p1′ = 140 + Clip3(−6, 6, −7) = 134, q1′ = 106.
        assert_eq!(buf.luma.samples[2 * 32 + 15], 127, "p0'");
        assert_eq!(buf.luma.samples[2 * 32 + 16], 113, "q0'");
        assert_eq!(buf.luma.samples[2 * 32 + 14], 134, "p1' (clip −(tC>>1))");
        assert_eq!(buf.luma.samples[2 * 32 + 17], 106, "q1'");
    }

    /// §8.8.3.6.8 asymmetric long filter, maxFilterLengthP = 7 /
    /// maxFilterLengthQ = 3 (eq. 1394): a 32-wide P CU against a
    /// 16-wide Q CU on a vertical edge with a flat 100→104 step passes
    /// the §8.8.3.6.6 large-block decision and must run the asymmetric
    /// kernel — 7 filtered P columns, 3 filtered Q columns. The
    /// pre-r418 fallback ran the short filter here.
    #[test]
    fn asymmetric_long_filter_7_3_vertical() {
        let mut buf = PictureBuffer::yuv420_filled(48, 8, 128);
        for y in 0..8 {
            for x in 0..32 {
                buf.luma.samples[y * 48 + x] = 100;
            }
            for x in 32..48 {
                buf.luma.samples[y * 48 + x] = 104;
            }
        }
        let cus = vec![cu(0, 0, 32, 8, 45), cu(32, 0, 16, 8, 45)];
        let params = DeblockParams {
            disabled: false,
            bit_depth: 8,
            ctb_log2_size_y: 7,
            ..Default::default()
        };
        apply_deblocking(&mut buf, &cus, &params, 1);
        // refMiddle (eq. 1394) = 102; p′ = [102,102,101,101,101,100,100]
        // (i = 0..6 at columns 31..25), q′ = [102,103,104] (32..34).
        let row = &buf.luma.samples[3 * 48..3 * 48 + 48];
        assert_eq!(&row[25..32], &[100, 100, 101, 101, 101, 102, 102]);
        assert_eq!(&row[32..36], &[102, 103, 104, 104]);
        // p7 and beyond untouched.
        assert_eq!(row[24], 100);
    }

    /// §8.8.3.6.2 step 6 — an EDGE_HOR edge on a luma CTB row boundary
    /// forces `sidePisLargeBlk = 0`, so eq. 1294 caps
    /// `maxFilterLengthP` at 3 and the long filter runs the asymmetric
    /// (P = 3, Q = 7) kernel of eq. 1393: only 3 rows above the CTB
    /// boundary are filtered (the line-buffer constraint), 7 below.
    #[test]
    fn ctb_row_boundary_caps_luma_p_side_at_3() {
        // CTB size 64 (ctb_log2_size_y = 6): the 8x128 picture holds
        // two stacked 8x64 CUs whose shared edge y = 64 IS a CTB row.
        let mut buf = PictureBuffer::yuv420_filled(8, 128, 128);
        for y in 0..64 {
            for x in 0..8 {
                buf.luma.samples[y * 8 + x] = 100;
            }
        }
        for y in 64..128 {
            for x in 0..8 {
                buf.luma.samples[y * 8 + x] = 104;
            }
        }
        let cus = vec![cu(0, 0, 8, 64, 45), cu(0, 64, 8, 64, 45)];
        let params = DeblockParams {
            disabled: false,
            bit_depth: 8,
            ctb_log2_size_y: 6,
            ..Default::default()
        };
        apply_deblocking(&mut buf, &cus, &params, 1);
        // refMiddle (eq. 1393) = 102; p′ = [102, 101, 100] on rows
        // 63/62/61, q′ = [102,102,103,103,103,104,104] on rows 64..70.
        let col = |y: usize| buf.luma.samples[y * 8 + 2];
        assert_eq!(
            [col(61), col(62), col(63)],
            [100, 101, 102],
            "3 filtered P rows"
        );
        assert_eq!(
            [
                col(64),
                col(65),
                col(66),
                col(67),
                col(68),
                col(69),
                col(70)
            ],
            [102, 102, 103, 103, 103, 104, 104],
            "7 filtered Q rows"
        );
        // Step 6: rows 57..=60 (p3..p6 of a symmetric 7/7 filter) must
        // stay untouched — the P side may not reach past 3 samples
        // above a CTB row boundary.
        for y in 57..=60 {
            assert_eq!(col(y), 100, "row {y} above the CTB row must be untouched");
        }
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
