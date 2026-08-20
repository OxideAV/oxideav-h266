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
    /// r431 — `pred_mode_plt_flag` of the CU. §8.8.3.6.7 (nDp/nDq →
    /// 0), §8.8.3.6.8 and §8.8.3.6.10 substitute the input samples for
    /// every filtered sample on a palette-coded side: the deblocker
    /// reads across the edge but never modifies palette samples.
    pub plt: bool,
    /// r440 — §8.8.3.6.4 per-component chroma QPs for the eq. 1343
    /// average: `[QpCb, QpCr, QpCbCr]` of the TB, each the
    /// ChromaQpTable-mapped `QpY` plus the PPS / SH / CU additive
    /// offsets (i.e. `Qp′X − QpBdOffset`). The sentinel
    /// `[i32::MIN; 3]` selects the legacy identity arm
    /// (`QpY + plane qp_offset`) used by the encoder-side designs and
    /// harnesses, whose SPS derives an identity chroma-QP table.
    pub qp_c: [i32; 3],
    /// r440 — `TuCResMode == 2` for the CU's chroma TB: §8.8.3.6.4
    /// reads `Qp′CbCr` for both components.
    pub joint_cbcr2: bool,
    /// r447 — `ciip_flag` of the CU: §8.8.3.5 forces `bS = 2` when
    /// either side of the edge is a CIIP-coded block.
    pub ciip: bool,
    /// r447 — the CU is MODE_IBC: drives the §8.8.3.5
    /// prediction-mode-difference and block-vector-difference `bS = 1`
    /// arms (the BV lives in the motion grid's L0 slot per the
    /// eqs. 1111 – 1118 bookkeeping).
    pub ibc: bool,
    /// r447 — `Some((NumSbX, NumSbY))` for an affine
    /// (`inter_affine_flag == 1`) or sub-block-merge
    /// (`merge_subblock_flag == 1`) CU (§8.5.5.9 grid dims). Drives the
    /// §8.8.3.4 sub-block boundary derivation: internal edges every
    /// `Max(8, nCb/numSb)` samples plus the eqs. 1229 / 1230 /
    /// 1240 / 1241 maxFilterLength caps at the CU edge. `None` for
    /// every other CU kind.
    pub num_sb: Option<(u32, u32)>,
    /// r447 — §7.3.11.9 multi-TB tiling map for a CU above
    /// `MaxTbSizeY = 64`: per-64-luma-tile coded flags, so the
    /// §8.8.3.5 tu-coded arm and the §8.8.3.2 interior
    /// transform-block edges see the PER-TB flags instead of a
    /// CU-wide OR. `None` for single-TB CUs (the plain `tu_*_coded`
    /// fields apply).
    pub tu64: Option<Tu64CbfMap>,
    /// r449 — the CU's transform tree is 1-D partitioned into 2..=4
    /// transform blocks (§8.4.5.1 ISP subpartitions or §7.4.12.5 SBT
    /// sub-TUs). §8.8.3.3 derives interior TB edges (`edgeIdc = 1`)
    /// at the 4-grid boundaries of these TBs and takes every
    /// `maxFilterLength` from the TRANSFORM-BLOCK dims, not the CU
    /// dims. `None` for one-TB-per-CU records (the `tu64` map covers
    /// the §7.3.11.9 >MaxTbSizeY tiling separately).
    pub tb_split: Option<TbSplit>,
}

/// r449 — a 1-D interior transform-block split of a CU (§8.4.5.1 ISP
/// subpartitions / §7.4.12.5 SBT sub-TUs) for the §8.8.3.3 transform
/// block boundary derivation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TbSplit {
    /// `true`: the boundaries are vertical lines (x offsets within the
    /// CU — a VER_SPLIT ISP / vertical SBT); `false`: horizontal.
    pub vertical: bool,
    /// Number of interior boundaries (1..=3).
    pub n_bounds: u8,
    /// Interior boundary offsets within the CU along the split axis,
    /// ascending, in luma samples (entries beyond `n_bounds` unused).
    pub bounds: [u32; 3],
    /// Per-TB `tu_y_coded_flag` (index 0 = first TB; `n_bounds + 1`
    /// entries used).
    pub y_coded: [bool; 4],
    /// Per-TB `tu_cb_coded_flag + tu_joint_cbcr_residual_flag` fold —
    /// only meaningful when `luma_only == false` (SBT).
    pub cb_coded: [bool; 4],
    /// Per-TB `tu_cr_coded_flag + tu_joint_cbcr_residual_flag` fold.
    pub cr_coded: [bool; 4],
    /// `true` for ISP: only the luma TB is split (§8.4.5.1 — the
    /// chroma TBs span the whole CU), so chroma edges / flags read the
    /// CU-level fields.
    pub luma_only: bool,
}

/// r447 — per-64-luma-tile CBF flags of a §7.3.11.9 multi-TB CU
/// (at most 2×2 tiles at the 128-sample CTU ceiling). Indexed
/// `[tile_y][tile_x]` with `tile = offset / 64` in CU-relative luma
/// samples. `y` is `tu_y_coded_flag`; `cb` / `cr` fold the per-TU
/// chroma CBFs (`tu_cb_coded` / `tu_cr_coded`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Tu64CbfMap {
    pub y: [[bool; 2]; 2],
    pub cb: [[bool; 2]; 2],
    pub cr: [[bool; 2]; 2],
}

impl DeblockCu {
    /// Per-component coded flag of the transform block containing the
    /// luma-coordinate sample — the per-64-tile flag for a multi-TB
    /// CU, the per-sub-TB flag for a [`TbSplit`] CU, the CU-level flag
    /// otherwise.
    fn tu_coded_at(&self, c_idx: u32, luma_x: i32, luma_y: i32) -> bool {
        if let Some(map) = &self.tu64 {
            let lx = (luma_x - self.x as i32).clamp(0, self.w as i32 - 1) as u32;
            let ly = (luma_y - self.y as i32).clamp(0, self.h as i32 - 1) as u32;
            let tx = ((lx / 64) as usize).min(1);
            let ty = ((ly / 64) as usize).min(1);
            return match c_idx {
                0 => map.y[ty][tx],
                1 => map.cb[ty][tx],
                _ => map.cr[ty][tx],
            };
        }
        if let Some(ts) = &self.tb_split {
            if c_idx == 0 || !ts.luma_only {
                let rel = if ts.vertical {
                    (luma_x - self.x as i32).clamp(0, self.w as i32 - 1) as u32
                } else {
                    (luma_y - self.y as i32).clamp(0, self.h as i32 - 1) as u32
                };
                let idx = ts.tb_index(rel);
                return match c_idx {
                    0 => ts.y_coded[idx],
                    1 => ts.cb_coded[idx],
                    _ => ts.cr_coded[idx],
                };
            }
        }
        match c_idx {
            0 => self.tu_y_coded,
            1 => self.tu_cb_coded,
            _ => self.tu_cr_coded,
        }
    }

    /// §8.8.3.3 — length (along the given axis) of the LUMA transform
    /// block containing the luma sample `(luma_x, luma_y)`. One TB per
    /// CU unless the CU tiles above `MaxTbSizeY` (`tu64`) or carries a
    /// [`TbSplit`].
    fn luma_tb_len(&self, luma_x: i32, luma_y: i32, vertical: bool) -> u32 {
        let dim = if vertical { self.w } else { self.h };
        let rel = if vertical {
            (luma_x - self.x as i32).clamp(0, self.w as i32 - 1) as u32
        } else {
            (luma_y - self.y as i32).clamp(0, self.h as i32 - 1) as u32
        };
        if self.tu64.is_some() && dim > 64 {
            return (dim - (rel / 64) * 64).min(64);
        }
        if let Some(ts) = &self.tb_split {
            if ts.vertical == vertical {
                let (start, end) = ts.tb_extent(rel, dim);
                return end - start;
            }
        }
        dim
    }

    /// §8.8.3.3 — length of the CHROMA transform block containing the
    /// sample, in CHROMA samples along the given axis. `sub` is the
    /// subsampling factor for that axis. ISP (`luma_only`) splits do
    /// not partition the chroma TB.
    fn chroma_tb_len(&self, luma_x: i32, luma_y: i32, vertical: bool, sub: u32) -> u32 {
        let dim = if vertical { self.w } else { self.h };
        let rel = if vertical {
            (luma_x - self.x as i32).clamp(0, self.w as i32 - 1) as u32
        } else {
            (luma_y - self.y as i32).clamp(0, self.h as i32 - 1) as u32
        };
        let luma_len = if self.tu64.is_some() && dim > 64 {
            (dim - (rel / 64) * 64).min(64)
        } else if let Some(ts) = &self.tb_split {
            if ts.vertical == vertical && !ts.luma_only {
                let (start, end) = ts.tb_extent(rel, dim);
                end - start
            } else {
                dim
            }
        } else {
            dim
        };
        luma_len / sub.max(1)
    }

    /// Is there a luma TB boundary at the CU-relative offset `off`
    /// (> 0) along the given axis?
    fn is_luma_tb_boundary(&self, off: u32, vertical: bool) -> bool {
        if off == 0 {
            return false;
        }
        let dim = if vertical { self.w } else { self.h };
        if off >= dim {
            return false;
        }
        if self.tu64.is_some() && dim > 64 {
            return off % 64 == 0;
        }
        if let Some(ts) = &self.tb_split {
            if ts.vertical == vertical {
                return ts.bounds[..ts.n_bounds as usize].contains(&off);
            }
        }
        false
    }

    /// Is there a chroma TB boundary at the CU-relative LUMA offset
    /// `off` (> 0) along the given axis?
    fn is_chroma_tb_boundary(&self, off: u32, vertical: bool) -> bool {
        if off == 0 {
            return false;
        }
        let dim = if vertical { self.w } else { self.h };
        if off >= dim {
            return false;
        }
        if self.tu64.is_some() && dim > 64 {
            return off % 64 == 0;
        }
        if let Some(ts) = &self.tb_split {
            if ts.vertical == vertical && !ts.luma_only {
                return ts.bounds[..ts.n_bounds as usize].contains(&off);
            }
        }
        false
    }
}

impl TbSplit {
    /// Index of the TB containing the CU-relative offset `rel` along
    /// the split axis.
    fn tb_index(&self, rel: u32) -> usize {
        let mut idx = 0usize;
        for i in 0..self.n_bounds as usize {
            if rel >= self.bounds[i] {
                idx = i + 1;
            }
        }
        idx
    }

    /// `(start, end)` of the TB containing `rel`, CU-relative.
    fn tb_extent(&self, rel: u32, dim: u32) -> (u32, u32) {
        let mut start = 0u32;
        let mut end = dim;
        for i in 0..self.n_bounds as usize {
            let b = self.bounds[i];
            if rel < b {
                end = b;
                break;
            }
            start = b;
        }
        (start, end)
    }
}

/// r447 — per-4×4-luma-cell motion snapshot the §8.8.3.5 boundary
/// strength derivation reads: prediction flags, the *resolved reference
/// POCs* (NOTE 1 — reference identity, not list/index position), and
/// the (unrefined, per the §8.5.1 NOTE) motion vectors in 1/16-luma
/// units. For a MODE_IBC cell the block vector sits in `mv_l0` with
/// both pred flags 0 (eqs. 1111 – 1118).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DeblockMvCell {
    pub pred_l0: bool,
    pub pred_l1: bool,
    pub poc_l0: i32,
    pub poc_l1: i32,
    pub mv_l0: (i32, i32),
    pub mv_l1: (i32, i32),
}

/// r447 — picture-wide grid of [`DeblockMvCell`]s at 4×4 luma
/// granularity. Built by the CTU walker from its per-picture motion
/// field after the last CTU reconstructs; `None` cells (outside any
/// inter CU) read as the all-zero default, which the §8.8.3.5 arms
/// never consult because the intra / mode rules fire first.
pub struct DeblockMvGrid {
    pub cells: Vec<DeblockMvCell>,
    pub cells_w: usize,
    pub cells_h: usize,
}

impl DeblockMvGrid {
    /// Cell covering the luma sample `(x, y)`; default cell when out
    /// of bounds.
    pub fn at_luma(&self, x: i32, y: i32) -> DeblockMvCell {
        if x < 0 || y < 0 {
            return DeblockMvCell::default();
        }
        let cx = (x as usize) / 4;
        let cy = (y as usize) / 4;
        if cx >= self.cells_w || cy >= self.cells_h {
            return DeblockMvCell::default();
        }
        self.cells[cy * self.cells_w + cx]
    }
}

/// Sentinel for [`DeblockCu::qp_c`] — take the legacy
/// `QpY + qp_offset` identity arm.
pub const DEBLOCK_QP_C_LEGACY: [i32; 3] = [i32::MIN; 3];

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
    apply_deblocking_clipped(out, cus, None, None, params, chroma_format_idc, &[], &[])
}

/// r429 — [`apply_deblocking`] with the §8.8.3.1 tile-boundary edge
/// exclusion: vertical edges whose luma x coincides with an entry of
/// `no_filter_cols`, and horizontal edges whose luma y coincides with
/// an entry of `no_filter_rows`, are not filtered (the
/// `pps_loop_filter_across_tiles_enabled_flag == 0` /
/// `pps_loop_filter_across_slices_enabled_flag == 0` arms; callers
/// pass the interior tile boundary positions).
#[allow(clippy::too_many_arguments)]
pub fn apply_deblocking_clipped(
    out: &mut PictureBuffer,
    cus: &[DeblockCu],
    chroma_cus: Option<&[DeblockCu]>,
    mv_grid: Option<&DeblockMvGrid>,
    params: &DeblockParams,
    chroma_format_idc: u32,
    no_filter_cols: &[u32],
    no_filter_rows: &[u32],
) {
    if std::env::var_os("H266_DBG_DBLK_CHROMA2").is_some() {
        eprintln!(
            "DBLK apply: cus={} chroma_cus={:?} cfi={chroma_format_idc} disabled={}",
            cus.len(),
            chroma_cus.map(|c| c.len()),
            params.disabled
        );
    }
    if params.disabled {
        return;
    }
    // Build a CU lookup indexed by 4x4 luma grid cell so the
    // boundary-strength derivation can find the CU on either side of an
    // edge in O(1). Stores indexes into `cus`.
    let grid = CuGrid::build(out.luma.width, out.luma.height, cus);
    // §8.8.3.2 — on a dual-tree slice the chroma edges derive from the
    // CHROMA coding tree's CB geometry, not the luma tree's. Callers
    // pass the chroma-tree records via `chroma_cus`; `None` (single
    // tree) shares the luma records.
    let chroma_grid = chroma_cus.map(|cc| CuGrid::build(out.luma.width, out.luma.height, cc));

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
    deblock_one_direction(
        &mut luma,
        cus,
        &grid,
        mv_grid,
        EdgeType::Vertical,
        no_filter_cols,
    );
    deblock_one_direction(
        &mut luma,
        cus,
        &grid,
        mv_grid,
        EdgeType::Horizontal,
        no_filter_rows,
    );

    if chroma_format_idc != 0 {
        let (c_cus, c_grid) = match (chroma_cus, chroma_grid.as_ref()) {
            (Some(cc), Some(cg)) => (cc, cg),
            _ => (cus, &grid),
        };
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
        deblock_one_direction(
            &mut cb,
            c_cus,
            c_grid,
            None,
            EdgeType::Vertical,
            no_filter_cols,
        );
        deblock_one_direction(
            &mut cb,
            c_cus,
            c_grid,
            None,
            EdgeType::Horizontal,
            no_filter_rows,
        );
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
        deblock_one_direction(
            &mut cr,
            c_cus,
            c_grid,
            None,
            EdgeType::Vertical,
            no_filter_cols,
        );
        deblock_one_direction(
            &mut cr,
            c_cus,
            c_grid,
            None,
            EdgeType::Horizontal,
            no_filter_rows,
        );
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
/// r449 — CUs iterate in decode order and each CU walks ALL of its
/// edges (leading CU edge first, then the interior transform-block /
/// sub-block edges in ascending geometric order), per the §8.8.3.1
/// ordering: "the vertical edges of the coding blocks in a coding
/// unit are filtered starting with the edge on the left-hand side of
/// the coding blocks proceeding through the edges towards the
/// right-hand side of the coding blocks in their geometrical order".
/// The filtering is sequential and in place — a decision or filter
/// read at one edge sees the samples already modified by the edges
/// processed before it (the same-direction data dependency the NOTE
/// in §8.8.3.1 describes), which is observable whenever edges sit 4
/// or 8 samples apart.
fn deblock_one_direction(
    plane: &mut PlaneCtx,
    cus: &[DeblockCu],
    grid: &CuGrid,
    mv_grid: Option<&DeblockMvGrid>,
    edge_type: EdgeType,
    no_filter: &[u32],
) {
    for (idx, cu) in cus.iter().enumerate() {
        deblock_cu_dir(
            plane, cus, grid, mv_grid, idx as u32, cu, edge_type, no_filter,
        );
    }
}

/// r449 — the §8.8.3.2 per-coding-block edge walk for one direction,
/// composing the §8.8.3.3 transform-block boundary derivation, the
/// §8.8.3.4 sub-block boundary derivation, the §8.8.3.5 boundary
/// strength and the §8.8.3.6 edge filtering, edge positions ascending.
#[allow(clippy::too_many_arguments)]
fn deblock_cu_dir(
    plane: &mut PlaneCtx,
    cus: &[DeblockCu],
    grid: &CuGrid,
    mv_grid: Option<&DeblockMvGrid>,
    idx: u32,
    cu: &DeblockCu,
    edge_type: EdgeType,
    no_filter: &[u32],
) {
    let vertical = edge_type == EdgeType::Vertical;
    let c_idx = plane.c_idx;
    // §8.8.3.2 — chroma coding blocks only walk when the leading edge
    // sits on the chroma 8-grid.
    if c_idx != 0 {
        let lead_c = if vertical {
            cu.x / plane.sub_w
        } else {
            cu.y / plane.sub_h
        };
        if lead_c % 8 != 0 {
            return;
        }
    }
    // Axis dims in luma samples: `n_cb` along the edge normal, `n_other`
    // along the edge.
    let (n_cb, n_other) = if vertical { (cu.w, cu.h) } else { (cu.h, cu.w) };
    let lead_luma = if vertical { cu.x } else { cu.y };
    // §8.8.3.2 step 1 — filterEdgeFlag: picture boundary plus the
    // r429 tile / slice boundary exclusions (`no_filter` carries the
    // excluded luma positions).
    let filter_edge = lead_luma != 0 && !no_filter.contains(&lead_luma);
    // Edge-position step along the normal (luma units): the luma
    // 4-grid, or the chroma 8-grid scaled to luma.
    let grid_step = if c_idx == 0 {
        4u32
    } else {
        8 * if vertical { plane.sub_w } else { plane.sub_h }
    };
    // Segment step along the edge (luma units): 4 luma rows / 2 chroma
    // rows per §8.8.3.5 eqs. 1254 / 1258.
    let seg_step = if c_idx == 0 {
        4u32
    } else {
        2 * if vertical { plane.sub_h } else { plane.sub_w }
    };
    // §8.8.3.4 sub-block geometry (luma only; numSb = 1 otherwise).
    let num_sb_axis = if c_idx == 0 {
        cu.num_sb
            .map(|(nx, ny)| if vertical { nx } else { ny })
            .unwrap_or(1)
            .max(1)
    } else {
        1
    };
    let sb = 8u32.max(n_cb / num_sb_axis);
    let n8 = n_cb / 8;
    // Largest xEdge index of the §8.8.3.4 loop:
    // Min( Max( 1, nCb / 8 ) − 1, numSb − 1 ).
    let last_sb_edge = (n8.max(1) - 1).min(num_sb_axis - 1);
    // Chroma CTB height (chroma samples) for the §8.8.3.3 CTB-row arm.
    let ctb_h_c = plane.ctb_size_y / plane.sub_h.max(1);

    let mut off = 0u32;
    while off < n_cb {
        // §8.8.3.3 — transform-block edge?
        let is_tb = if off == 0 {
            filter_edge
        } else if c_idx == 0 {
            cu.is_luma_tb_boundary(off, vertical)
        } else {
            cu.is_chroma_tb_boundary(off, vertical)
        };
        // §8.8.3.4 — sub-block edge (edgeIdc = 2)? Every luma CU marks
        // its own leading edge (xEdge = 0); sub-block CUs additionally
        // mark interior edges at multiples of sbW.
        let is_sb_edge =
            c_idx == 0 && off % sb == 0 && off / sb <= last_sb_edge && (off > 0 || filter_edge);
        if !(is_tb || is_sb_edge) {
            off += grid_step;
            continue;
        }

        let mut k = 0u32;
        while k < n_other {
            // Luma coordinates of q0 / p0 for this segment.
            let (qx, qy) = if vertical {
                ((cu.x + off) as i32, (cu.y + k) as i32)
            } else {
                ((cu.x + k) as i32, (cu.y + off) as i32)
            };
            let (px, py) = if vertical { (qx - 1, qy) } else { (qx, qy - 1) };
            // The CU records on each side (cross-CU on the leading edge).
            let p_idx = if off == 0 {
                grid.cu_at(px, py)
            } else {
                Some(idx)
            };
            let Some(p_idx) = p_idx else {
                k += seg_step;
                continue;
            };
            if off == 0 && p_idx == idx {
                k += seg_step;
                continue;
            }
            let p_cu = &cus[p_idx as usize];
            let q_cu = cu;

            let b_s = derive_bs(
                c_idx,
                p_cu,
                q_cu,
                mv_grid,
                (px, py),
                (qx, qy),
                is_tb,
                is_sb_edge,
            );
            if c_idx != 0 && std::env::var_os("H266_DBG_DBLK_CHROMA2").is_some() {
                eprintln!(
                    "DBLKC2 c={c_idx} {} q=({qx},{qy}) off={off} is_tb={is_tb} bS={b_s} p_intra={} q_intra={} p_cu=({},{})",
                    if vertical { "V" } else { "H" },
                    p_cu.intra,
                    q_cu.intra,
                    p_cu.x,
                    p_cu.y,
                );
            }
            if b_s == 0 {
                k += seg_step;
                continue;
            }

            if c_idx == 0 {
                // §8.8.3.3 — maxFilterLength from the adjacent
                // TRANSFORM-BLOCK dims (0-initialized when the edge is
                // not a TB edge; the §8.8.3.4 overlay then assigns).
                let (mut mfl_p, mut mfl_q) = if is_tb {
                    let wq = q_cu.luma_tb_len(qx, qy, vertical);
                    let wp = p_cu.luma_tb_len(px, py, vertical);
                    if wp <= 4 || wq <= 4 {
                        (1u32, 1u32)
                    } else {
                        (if wp >= 32 { 7 } else { 3 }, if wq >= 32 { 7 } else { 3 })
                    }
                } else {
                    (0, 0)
                };
                // §8.8.3.4 — the sub-block overlay.
                if is_sb_edge {
                    if off == 0 {
                        // eqs. 1229 / 1230 (P side capped when the
                        // neighbour is affine / sub-block-merge coded).
                        if num_sb_axis > 1 {
                            mfl_q = mfl_q.min(5);
                        }
                        if p_cu.num_sb.is_some() {
                            mfl_p = mfl_p.min(5);
                        }
                    } else {
                        // Interior sub-block edge — the cascade reads
                        // edgeTbFlags (the §8.8.3.3 edgeIdc snapshot).
                        let tb_at = |z: i64| -> bool {
                            if z < 0 || z as u32 >= n_cb {
                                false
                            } else if z == 0 {
                                filter_edge
                            } else {
                                cu.is_luma_tb_boundary(z as u32, vertical)
                            }
                        };
                        if is_tb {
                            // eqs. 1231 / 1232.
                            mfl_p = mfl_p.min(5);
                            mfl_q = mfl_q.min(5);
                        } else if tb_at(off as i64 - 4) || tb_at(off as i64 + 4) {
                            // eqs. 1233 / 1234.
                            mfl_p = 1;
                            mfl_q = 1;
                        } else if off / sb == 1
                            || off / sb == n8.max(1) - 1
                            || tb_at(off as i64 - sb as i64)
                            || tb_at(off as i64 + sb as i64)
                        {
                            // eqs. 1235 / 1236.
                            mfl_p = 2;
                            mfl_q = 2;
                        } else {
                            // eqs. 1237 / 1238.
                            mfl_p = 3;
                            mfl_q = 3;
                        }
                    }
                }
                // §8.8.3.6.2 step 6 — an EDGE_HOR edge on a luma CTB
                // row boundary suppresses sidePisLargeBlk (only the
                // CU's own top edge can sit on a CTB row).
                let ctb_row_edge =
                    !vertical && off == 0 && plane.ctb_size_y > 0 && cu.y % plane.ctb_size_y == 0;
                let (_qp, beta, tc) = compute_thresholds_luma(plane, p_cu, q_cu, b_s);
                run_luma_filter(
                    plane.plane,
                    qx,
                    qy,
                    beta,
                    tc,
                    mfl_p,
                    mfl_q,
                    vertical,
                    ctb_row_edge,
                    p_cu.plt,
                    q_cu.plt,
                );
            } else {
                // §8.8.3.3 chroma arm — both adjacent chroma TBs ≥ 8
                // chroma samples → (3, 3) (P capped to 1 on a chroma
                // CTB row for EDGE_HOR), else (1, 1).
                let sub_axis = if vertical { plane.sub_w } else { plane.sub_h };
                let wq = q_cu.chroma_tb_len(qx, qy, vertical, sub_axis);
                let wp = p_cu.chroma_tb_len(px, py, vertical, sub_axis);
                let (mfl_p, mfl_q) = if wp >= 8 && wq >= 8 {
                    let ctb_row =
                        !vertical && off == 0 && ctb_h_c > 0 && (cu.y / plane.sub_h) % ctb_h_c == 0;
                    if ctb_row {
                        (1, 3)
                    } else {
                        (3, 3)
                    }
                } else {
                    (1, 1)
                };
                let (_qp, beta, tc) = compute_thresholds_chroma(plane, p_cu, q_cu, b_s);
                let ex = qx / plane.sub_w as i32;
                let ey = qy / plane.sub_h as i32;
                if std::env::var_os("H266_DBG_DBLK_CHROMA").is_some() {
                    eprintln!(
                        "DBLKC c={} {} edge ({ex},{ey}) bS={b_s} mfl=({mfl_p},{mfl_q}) beta={beta} tc={tc} cu=({},{}) off={off}",
                        c_idx,
                        if vertical { "V" } else { "H" },
                        cu.x,
                        cu.y,
                    );
                }
                if vertical {
                    run_chroma_filter_v(
                        plane.plane,
                        ex,
                        ey,
                        beta,
                        tc,
                        b_s,
                        mfl_p,
                        mfl_q,
                        p_cu.plt,
                        q_cu.plt,
                    );
                } else {
                    run_chroma_filter_h(
                        plane.plane,
                        ex,
                        ey,
                        beta,
                        tc,
                        b_s,
                        mfl_p,
                        mfl_q,
                        p_cu.plt,
                        q_cu.plt,
                    );
                }
            }
            k += seg_step;
        }
        off += grid_step;
    }
}

/// r447 — the §8.8.3.5 motion-difference `bS = 1` cascade for a luma
/// edge between two inter sub-blocks (both sides MODE_INTER; the
/// intra / CIIP / IBC / mode-difference arms are the caller's).
///
/// Reference identity is by resolved POC (NOTE 1); "number of motion
/// vectors" is `PredFlagL0 + PredFlagL1` (NOTE 2); the threshold is 8
/// in 1/16-luma units (half a luma sample) on either component.
fn bs_motion_rule(p: DeblockMvCell, q: DeblockMvCell) -> bool {
    let np = u32::from(p.pred_l0) + u32::from(p.pred_l1);
    let nq = u32::from(q.pred_l0) + u32::from(q.pred_l1);
    // Different number of motion vectors.
    if np != nq {
        return true;
    }
    if np == 0 {
        return false;
    }
    let diff_ge8 =
        |a: (i32, i32), b: (i32, i32)| -> bool { (a.0 - b.0).abs() >= 8 || (a.1 - b.1).abs() >= 8 };
    if np == 1 {
        let (p_poc, p_mv) = if p.pred_l0 {
            (p.poc_l0, p.mv_l0)
        } else {
            (p.poc_l1, p.mv_l1)
        };
        let (q_poc, q_mv) = if q.pred_l0 {
            (q.poc_l0, q.mv_l0)
        } else {
            (q.poc_l1, q.mv_l1)
        };
        // Different reference pictures (by identity, list-agnostic).
        if p_poc != q_poc {
            return true;
        }
        return diff_ge8(p_mv, q_mv);
    }
    // np == nq == 2. Reference multiset comparison first.
    let same_ordered = p.poc_l0 == q.poc_l0 && p.poc_l1 == q.poc_l1;
    let same_swapped = p.poc_l0 == q.poc_l1 && p.poc_l1 == q.poc_l0;
    if !(same_ordered || same_swapped) {
        return true;
    }
    if p.poc_l0 != p.poc_l1 {
        // Two different reference pictures on each side — compare the
        // MVs per reference picture.
        if same_ordered && (diff_ge8(p.mv_l0, q.mv_l0) || diff_ge8(p.mv_l1, q.mv_l1)) {
            return true;
        }
        if !same_ordered && (diff_ge8(p.mv_l0, q.mv_l1) || diff_ge8(p.mv_l1, q.mv_l0)) {
            return true;
        }
        false
    } else {
        // Both MVs of each side point at the SAME reference picture:
        // bS = 1 only when BOTH the list-aligned and the cross-list
        // comparisons see a ≥ half-luma difference.
        let straight = diff_ge8(p.mv_l0, q.mv_l0) || diff_ge8(p.mv_l1, q.mv_l1);
        let crossed = diff_ge8(p.mv_l0, q.mv_l1) || diff_ge8(p.mv_l1, q.mv_l0);
        straight && crossed
    }
}

/// §8.8.3.5 boundary-strength derivation for one edge segment.
///
/// r449 — the tu-coded arm applies only when the edge "is also a
/// transform block edge" (`is_tb_edge`), and the prediction-mode /
/// IBC-BV / motion-difference arms only when `edgeIdc == 2`
/// (`edge_idc2` — CU boundaries and §8.8.3.4 sub-block edges).
#[inline]
#[allow(clippy::too_many_arguments)]
fn derive_bs(
    c_idx: u32,
    p: &DeblockCu,
    q: &DeblockCu,
    mv_grid: Option<&DeblockMvGrid>,
    p_luma: (i32, i32),
    q_luma: (i32, i32),
    is_tb_edge: bool,
    edge_idc2: bool,
) -> i32 {
    // BDPCM both-sides → bS = 0 (§8.8.3.5).
    if c_idx == 0 && p.bdpcm_luma && q.bdpcm_luma {
        return 0;
    }
    if c_idx != 0 && p.bdpcm_chroma && q.bdpcm_chroma {
        return 0;
    }
    if p.intra || q.intra {
        return 2;
    }
    // r447 — §8.8.3.5: a CIIP-coded block on either side → bS = 2.
    if p.ciip || q.ciip {
        return 2;
    }
    // The tu-coded arm applies only when the edge is also a
    // transform-block edge. r447: for a §7.3.11.9 multi-TB CU (and,
    // r449, a [`TbSplit`] CU) the flag is the PER-TB one adjacent to
    // the edge, not a CU-wide OR.
    if is_tb_edge {
        let coded_either =
            p.tu_coded_at(c_idx, p_luma.0, p_luma.1) || q.tu_coded_at(c_idx, q_luma.0, q_luma.1);
        if coded_either {
            return 1;
        }
    }
    // r447 — the remaining §8.8.3.5 arms are luma-only and gated on
    // `edgeIdc == 2` (CU boundaries and sub-block edges).
    if c_idx != 0 || !edge_idc2 {
        return 0;
    }
    // Prediction-mode difference (MODE_IBC vs MODE_INTER; intra was
    // handled above).
    if p.ibc != q.ibc {
        return 1;
    }
    if p.ibc && q.ibc {
        // Both IBC: block-vector difference ≥ 8 in 1/16-luma units.
        // The BV lives in the motion grid's L0 slot (eq. 1111).
        if let Some(g) = mv_grid {
            let pb = g.at_luma(p_luma.0, p_luma.1).mv_l0;
            let qb = g.at_luma(q_luma.0, q_luma.1).mv_l0;
            if (pb.0 - qb.0).abs() >= 8 || (pb.1 - qb.1).abs() >= 8 {
                return 1;
            }
        }
        return 0;
    }
    // Both MODE_INTER: reference / motion-vector difference rules.
    if let Some(g) = mv_grid {
        if bs_motion_rule(g.at_luma(p_luma.0, p_luma.1), g.at_luma(q_luma.0, q_luma.1)) {
            return 1;
        }
    }
    0
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
    // §8.8.3.6.4 — QpP / QpQ are the per-component chroma QPs of the
    // TBs containing p0,0 / q0,0 (`Qp′CbCr` when the TB's
    // `TuCResMode == 2`, else `Qp′Cb` / `Qp′Cr` by cIdx), taken in the
    // `− QpBdOffset` domain per eq. 1343. Records carrying the legacy
    // sentinel fall back to the identity `QpY + qp_offset` arm.
    let pick = |c: &DeblockCu| -> i32 {
        if c.qp_c[0] == i32::MIN {
            (c.qp_y + plane.qp_offset).clamp(0, 63)
        } else if c.joint_cbcr2 {
            c.qp_c[2]
        } else {
            c.qp_c[(plane.c_idx as usize).saturating_sub(1).min(1)]
        }
    };
    let qp_p = pick(p);
    let qp_q = pick(q);
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
    let clipped = v.clamp(0, max) as u16;
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
    plt_p: bool,
    plt_q: bool,
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
                    long_luma_apply(
                        plane,
                        cx,
                        cy + k,
                        tc,
                        dec.mfl_p,
                        dec.mfl_q,
                        true,
                        plt_p,
                        plt_q,
                    );
                } else {
                    long_luma_apply(
                        plane,
                        cx + k,
                        cy,
                        tc,
                        dec.mfl_p,
                        dec.mfl_q,
                        false,
                        plt_p,
                        plt_q,
                    );
                }
            }
        }
        _ => {
            for k in 0..4i32 {
                if is_vertical {
                    short_luma_apply(
                        plane,
                        cx,
                        cy + k,
                        tc,
                        dec.d_e,
                        dec.d_ep,
                        dec.d_eq,
                        true,
                        plt_p,
                        plt_q,
                    );
                } else {
                    short_luma_apply(
                        plane,
                        cx + k,
                        cy,
                        tc,
                        dec.d_e,
                        dec.d_ep,
                        dec.d_eq,
                        false,
                        plt_p,
                        plt_q,
                    );
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
#[allow(clippy::too_many_arguments)]
fn long_luma_apply(
    plane: &mut PicturePlane,
    cx: i32,
    cy: i32,
    tc: i32,
    mfl_p: u32,
    mfl_q: u32,
    is_vertical: bool,
    plt_p: bool,
    plt_q: bool,
) {
    let bd = plane.bit_depth;
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

    // Eqs. 1409 / 1410 — write filtered samples. §8.8.3.6.8: a
    // palette-coded side keeps its input samples (the substitution
    // realised as a write skip; the cross-edge reads are unchanged).
    for i in 0..mfl_p_u {
        if plt_p {
            break;
        }
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
        if plt_q {
            break;
        }
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
    plt_p: bool,
    plt_q: bool,
) {
    let bd = plane.bit_depth;
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
        // Strong filter — eqs. 1375–1380. §8.8.3.6.7: nDp / nDq drop
        // to 0 when the containing CU is palette-coded.
        let p0n = ((p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + 4) >> 3).clamp(p0 - 3 * tc, p0 + 3 * tc);
        let p1n = ((p2 + p1 + p0 + q0 + 2) >> 2).clamp(p1 - 2 * tc, p1 + 2 * tc);
        let p2n = ((2 * p3 + 3 * p2 + p1 + p0 + q0 + 4) >> 3).clamp(p2 - tc, p2 + tc);
        let q0n = ((p1 + 2 * p0 + 2 * q0 + 2 * q1 + q2 + 4) >> 3).clamp(q0 - 3 * tc, q0 + 3 * tc);
        let q1n = ((p0 + q0 + q1 + q2 + 2) >> 2).clamp(q1 - 2 * tc, q1 + 2 * tc);
        let q2n = ((p0 + q0 + q1 + 3 * q2 + 2 * q3 + 4) >> 3).clamp(q2 - tc, q2 + tc);
        if !plt_p {
            write_p(0, p0n, plane);
            write_p(1, p1n, plane);
            write_p(2, p2n, plane);
        }
        if !plt_q {
            if is_vertical {
                write(plane, cx, cy, q0n, bd);
                write(plane, cx + 1, cy, q1n, bd);
                write(plane, cx + 2, cy, q2n, bd);
            } else {
                write(plane, cx, cy, q0n, bd);
                write(plane, cx, cy + 1, q1n, bd);
                write(plane, cx, cy + 2, q2n, bd);
            }
        }
    } else {
        // Weak filter — eqs. 1381–1388.
        let delta_raw = (9 * (q0 - p0) - 3 * (q1 - p1) + 8) >> 4;
        if delta_raw.abs() < tc * 10 {
            let delta = delta_raw.clamp(-tc, tc);
            if !plt_p {
                write_p(0, p0 + delta, plane);
            }
            if !plt_q {
                write(plane, cx, cy, q0 - delta, bd);
            }
            if d_ep == 1 && !plt_p {
                // Eq. 1385 — the clip bound is −(tC >> 1), NOT
                // (−tC) >> 1: the two differ by 1 for odd tC (the
                // arithmetic right-shift rounds toward −∞).
                let dp = ((((p2 + p0 + 1) >> 1) - p1 + delta) >> 1).clamp(-(tc >> 1), tc >> 1);
                write_p(1, p1 + dp, plane);
            }
            if d_eq == 1 && !plt_q {
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
    b_s: i32,
    max_filter_length_p: u32,
    max_filter_length_q: u32,
    plt_p: bool,
    plt_q: bool,
) {
    if tc == 0 {
        return;
    }
    // §8.8.3.6.4 — "When both maxFilterLengthP and maxFilterLengthQ
    // are equal to 1 and bS is not equal to 2, maxFilterLengthP and
    // maxFilterLengthQ are both set equal to 0" — and §8.8.3.6.1
    // step 2 only invokes the filter when maxFilterLengthQ > 0, so a
    // (1, 1) edge filters ONLY at bS = 2.
    if max_filter_length_p == 1 && max_filter_length_q == 1 && b_s != 2 {
        return;
    }
    let bd = plane.bit_depth;
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
                chroma_strong_apply_v(plane, cx, cy + k, tc, bd, plt_p, plt_q);
            }
            return;
        }
        // Decision failed → fall through to the weak filter.
    }

    // Weak filter (eqs 1421 – 1423) — §8.8.3.6.10 palette
    // substitution realised as per-side write skips.
    for k in 0..2i32 {
        let p1 = read_clamped(plane, cx - 2, cy + k);
        let p0 = read_clamped(plane, cx - 1, cy + k);
        let q0 = read_clamped(plane, cx, cy + k);
        let q1 = read_clamped(plane, cx + 1, cy + k);
        let delta = ((((q0 - p0) << 2) + p1 - q1 + 4) >> 3).clamp(-tc, tc);
        if !plt_p {
            write(plane, cx - 1, cy + k, p0 + delta, bd);
        }
        if !plt_q {
            write(plane, cx, cy + k, q0 - delta, bd);
        }
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
    b_s: i32,
    max_filter_length_p: u32,
    max_filter_length_q: u32,
    plt_p: bool,
    plt_q: bool,
) {
    if tc == 0 {
        return;
    }
    // §8.8.3.6.4 — "When both maxFilterLengthP and maxFilterLengthQ
    // are equal to 1 and bS is not equal to 2, maxFilterLengthP and
    // maxFilterLengthQ are both set equal to 0" — and §8.8.3.6.1
    // step 2 only invokes the filter when maxFilterLengthQ > 0, so a
    // (1, 1) edge filters ONLY at bS = 2.
    if max_filter_length_p == 1 && max_filter_length_q == 1 && b_s != 2 {
        return;
    }
    let bd = plane.bit_depth;
    let max_k = 1i32;

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
                chroma_strong_apply_h(plane, cx + k, cy, tc, bd, short_p, plt_p, plt_q);
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
        if !plt_p {
            write(plane, cx + k, cy - 1, p0 + delta, bd);
        }
        if !plt_q {
            write(plane, cx + k, cy, q0 - delta, bd);
        }
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
fn chroma_strong_apply_v(
    plane: &mut PicturePlane,
    cx: i32,
    cy: i32,
    tc: i32,
    bd: u32,
    plt_p: bool,
    plt_q: bool,
) {
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
    if !plt_p {
        write(plane, cx - 1, cy, p0n, bd);
        write(plane, cx - 2, cy, p1n, bd);
        write(plane, cx - 3, cy, p2n, bd);
    }
    if !plt_q {
        write(plane, cx, cy, q0n, bd);
        write(plane, cx + 1, cy, q1n, bd);
        write(plane, cx + 2, cy, q2n, bd);
    }
}

/// Mirror of [`chroma_strong_apply_v`] for an EDGE_HOR edge. With
/// `short_p` the §8.8.3.6.10 asymmetric (P = 1, Q = 3) filter applies
/// (eqs. 1417 - 1420): only p0 is modified on the P side.
#[allow(clippy::too_many_arguments)]
fn chroma_strong_apply_h(
    plane: &mut PicturePlane,
    cx: i32,
    cy: i32,
    tc: i32,
    bd: u32,
    short_p: bool,
    plt_p: bool,
    plt_q: bool,
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
        if !plt_p {
            write(plane, cx, cy - 1, p0n, bd);
        }
        if !plt_q {
            write(plane, cx, cy, q0n, bd);
            write(plane, cx, cy + 1, q1n, bd);
            write(plane, cx, cy + 2, q2n, bd);
        }
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
    if !plt_p {
        write(plane, cx, cy - 1, p0n, bd);
        write(plane, cx, cy - 2, p1n, bd);
        write(plane, cx, cy - 3, p2n, bd);
    }
    if !plt_q {
        write(plane, cx, cy, q0n, bd);
        write(plane, cx, cy + 1, q1n, bd);
        write(plane, cx, cy + 2, q2n, bd);
    }
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

    /// r437 — eqs. 1276 / 1278 / 1279 at 10-bit: β scales by
    /// `<< (BitDepth − 8)` and tC passes through at BitDepth 10
    /// (`tC' * (1 << (10 − 10)) = tC'`); at 12-bit tC gains `<< 2`.
    #[test]
    fn bit_depth_scaling_10_and_12_bit() {
        assert_eq!(scale_beta_for_bit_depth(26, 10), 26 << 2);
        assert_eq!(scale_tc_for_bit_depth(10, 10), 10);
        assert_eq!(scale_beta_for_bit_depth(26, 12), 26 << 4);
        assert_eq!(scale_tc_for_bit_depth(10, 12), 10 << 2);
        // BitDepth 9 rounds per eq. 1278: (tc' + 1) >> 1.
        assert_eq!(scale_tc_for_bit_depth(9, 9), (9 + 1) >> 1);
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
                qp_c: crate::deblock::DEBLOCK_QP_C_LEGACY,
                joint_cbcr2: false,
                plt: false,
                ciip: false,
                ibc: false,
                num_sb: None,
                tu64: None,
                tb_split: None,
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
                qp_c: crate::deblock::DEBLOCK_QP_C_LEGACY,
                joint_cbcr2: false,
                plt: false,
                ciip: false,
                ibc: false,
                num_sb: None,
                tu64: None,
                tb_split: None,
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
                qp_c: crate::deblock::DEBLOCK_QP_C_LEGACY,
                joint_cbcr2: false,
                plt: false,
                ciip: false,
                ibc: false,
                num_sb: None,
                tu64: None,
                tb_split: None,
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
                qp_c: crate::deblock::DEBLOCK_QP_C_LEGACY,
                joint_cbcr2: false,
                plt: false,
                ciip: false,
                ibc: false,
                num_sb: None,
                tu64: None,
                tb_split: None,
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
                qp_c: crate::deblock::DEBLOCK_QP_C_LEGACY,
                joint_cbcr2: false,
                plt: false,
                ciip: false,
                ibc: false,
                num_sb: None,
                tu64: None,
                tb_split: None,
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
                qp_c: crate::deblock::DEBLOCK_QP_C_LEGACY,
                joint_cbcr2: false,
                plt: false,
                ciip: false,
                ibc: false,
                num_sb: None,
                tu64: None,
                tb_split: None,
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

    /// r449 — §8.8.3.3 transform-block geometry lookups: a one-TB CU
    /// reports its own dims; a [`TbSplit`] CU reports the sub-TB
    /// extents and boundaries; ISP (`luma_only`) splits leave the
    /// chroma TB whole.
    #[test]
    fn tb_split_geometry_lookups() {
        let base = DeblockCu {
            x: 32,
            y: 16,
            w: 32,
            h: 16,
            qp_y: 32,
            intra: true,
            tu_y_coded: true,
            tu_cb_coded: false,
            tu_cr_coded: false,
            bdpcm_luma: false,
            bdpcm_chroma: false,
            qp_c: crate::deblock::DEBLOCK_QP_C_LEGACY,
            joint_cbcr2: false,
            plt: false,
            ciip: false,
            ibc: false,
            num_sb: None,
            tu64: None,
            tb_split: None,
        };
        // One TB per CU: full dims, no interior boundaries.
        assert_eq!(base.luma_tb_len(40, 20, true), 32);
        assert_eq!(base.luma_tb_len(40, 20, false), 16);
        assert!(!base.is_luma_tb_boundary(8, true));
        // Vertical ISP of the 32-wide CU into 4 8-wide partitions.
        let isp = DeblockCu {
            tb_split: Some(TbSplit {
                vertical: true,
                n_bounds: 3,
                bounds: [8, 16, 24],
                y_coded: [true; 4],
                cb_coded: [false; 4],
                cr_coded: [false; 4],
                luma_only: true,
            }),
            ..base
        };
        assert_eq!(isp.luma_tb_len(40, 20, true), 8);
        assert_eq!(isp.luma_tb_len(63, 20, true), 8);
        // Perpendicular extent is unaffected by a vertical split.
        assert_eq!(isp.luma_tb_len(40, 20, false), 16);
        assert!(isp.is_luma_tb_boundary(8, true));
        assert!(isp.is_luma_tb_boundary(16, true));
        assert!(!isp.is_luma_tb_boundary(4, true));
        assert!(!isp.is_luma_tb_boundary(8, false));
        // ISP does not split the chroma TB (§8.4.5.1).
        assert!(!isp.is_chroma_tb_boundary(16, true));
        assert_eq!(isp.chroma_tb_len(40, 20, true, 2), 16);
        // SBT half split (not luma-only): chroma splits too.
        let sbt = DeblockCu {
            intra: false,
            tb_split: Some(TbSplit {
                vertical: true,
                n_bounds: 1,
                bounds: [16, 0, 0],
                y_coded: [true, false, false, false],
                cb_coded: [true, false, false, false],
                cr_coded: [false; 4],
                luma_only: false,
            }),
            ..base
        };
        assert!(sbt.is_chroma_tb_boundary(16, true));
        assert_eq!(sbt.chroma_tb_len(34, 20, true, 2), 8);
        // Per-TB coded flags: the residual TB is the first sub-TU.
        assert!(sbt.tu_coded_at(0, 40, 20));
        assert!(!sbt.tu_coded_at(0, 50, 20));
        assert!(sbt.tu_coded_at(1, 40, 20));
        assert!(!sbt.tu_coded_at(1, 50, 20));
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
                qp_c: crate::deblock::DEBLOCK_QP_C_LEGACY,
                joint_cbcr2: false,
                plt: false,
                ciip: false,
                ibc: false,
                num_sb: None,
                tu64: None,
                tb_split: None,
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
                qp_c: crate::deblock::DEBLOCK_QP_C_LEGACY,
                joint_cbcr2: false,
                plt: false,
                ciip: false,
                ibc: false,
                num_sb: None,
                tu64: None,
                tb_split: None,
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
            qp_c: crate::deblock::DEBLOCK_QP_C_LEGACY,
            joint_cbcr2: false,
            plt: false,
            ciip: false,
            ibc: false,
            num_sb: None,
            tu64: None,
            tb_split: None,
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
            qp_c: crate::deblock::DEBLOCK_QP_C_LEGACY,
            joint_cbcr2: false,
            plt: false,
            ciip: false,
            ibc: false,
            num_sb: None,
            tu64: None,
            tb_split: None,
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
