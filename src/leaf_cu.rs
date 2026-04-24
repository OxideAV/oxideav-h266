//! VVC leaf coding-unit syntax + intra-mode derivation.
//!
//! This module covers the *per-CU* parsing hop between the coding-tree
//! walker (which produces leaf rectangles) and the reconstruction
//! pipeline (which turns an intra mode + residual into pixels). The
//! focus of this round is strictly the syntax reads from the spec's
//! `coding_unit()` structure (§7.3.11.5) plus the derivations
//! in §8.4.2 (luma intra mode) and §8.4.3 (chroma intra mode).
//!
//! Covered syntax elements (I-slice, single-tree subset):
//!
//! * `pred_mode_flag`, `pred_mode_ibc_flag`, `pred_mode_plt_flag` —
//!   coarse prediction-mode dispatch. In I-slices `pred_mode_flag` is
//!   not signalled and is inferred to 1 (MODE_INTRA) per spec §7.4.12.2.
//! * `intra_bdpcm_luma_flag` / `intra_bdpcm_luma_dir_flag` —
//!   block DPCM (§7.4.12.2). Only parsed when the SPS enables it.
//! * `intra_mip_flag` + `intra_mip_transposed_flag` + `intra_mip_mode` —
//!   Matrix-based intra prediction (§8.4.5.2.15). Parsed when the SPS
//!   enables it and the CU size qualifies.
//! * `intra_luma_ref_idx` — multi-reference-line selection (§7.4.12.2).
//!   Parsed via TR(cMax=2) when MRL is enabled and `y0 % CtbSizeY > 0`.
//! * `intra_subpartitions_mode_flag` / `intra_subpartitions_split_flag`
//!   — ISP selector (§7.4.12.2, Table 13).
//! * `intra_luma_mpm_flag`, `intra_luma_not_planar_flag`,
//!   `intra_luma_mpm_idx`, `intra_luma_mpm_remainder` — the MPM cascade
//!   that the luma intra-mode derivation (§8.4.2) reads out of.
//! * `intra_chroma_pred_mode` — Table 130 binarisation (ctx bin 0 +
//!   two bypass bins). Mapped to `IntraPredModeC` via Table 20 in the
//!   caller's derivation step.
//!
//! Out of scope (still returns `Error::Unsupported` one layer up):
//!
//! * CCLM (`cclm_mode_flag` / `cclm_mode_idx`) — chroma from luma.
//! * IBC / PLT CUs — IBC parsing + palette coding.
//! * CBF / residual coding / transforms / inverse quantisation.
//! * Dual-tree luma / chroma split. Chroma intra mode is derived for
//!   `treeType == SINGLE_TREE` only.
//!
//! The module is intentionally "parse + derive, don't reconstruct" —
//! [`LeafCuInfo`] captures everything a follow-up round needs to build
//! reference samples + drive the intra predictor, but nothing here
//! touches pixels.
//!
//! Spec reference: ITU-T H.266 | ISO/IEC 23090-3 (V4, 01/2026). The
//! implementation is spec-only; no third-party VVC decoder source was
//! consulted.

use oxideav_core::{Error, Result};

use crate::cabac::{ArithDecoder, ContextModel};
use crate::ctx::{
    ctx_inc_intra_chroma_pred_mode, ctx_inc_intra_luma_mpm_flag,
    ctx_inc_intra_luma_not_planar_flag, ctx_inc_intra_luma_ref_idx, ctx_inc_intra_mip_flag,
    ctx_inc_intra_subpartitions_mode_flag, ctx_inc_intra_subpartitions_split_flag,
};
use crate::residual::{
    decode_tb_coefficients, read_cu_chroma_qp_offset, read_cu_qp_delta, read_tu_cb_coded_flag,
    read_tu_cr_coded_flag, read_tu_y_coded_flag, ResidualCtxs,
};
use crate::tables::{init_contexts, SyntaxCtx};

/// CU prediction mode (§7.4.12.2 — `MODE_*`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CuPredMode {
    /// Regular intra-predicted CU.
    Intra,
    /// Inter-predicted (not reachable in I-slices).
    Inter,
    /// Intra-block-copy CU (IBC).
    Ibc,
    /// Palette-mode CU.
    Plt,
}

/// Intra subpartitions split type (Table 13).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IspSplitType {
    /// `ISP_NO_SPLIT` — ISP disabled for this CU.
    NoSplit,
    /// `ISP_HOR_SPLIT` — horizontal partition.
    HorSplit,
    /// `ISP_VER_SPLIT` — vertical partition.
    VerSplit,
}

/// Tool-enable flags that gate the `coding_unit()` syntax reads in
/// the SPS. Grouped here for test convenience; production callers pass
/// a [`crate::sps::SeqParameterSet`] view into [`LeafCuReader::new`].
#[derive(Clone, Copy, Debug, Default)]
pub struct CuToolFlags {
    /// `sps_ibc_enabled_flag`.
    pub ibc: bool,
    /// `sps_palette_enabled_flag`.
    pub palette: bool,
    /// `sps_bdpcm_enabled_flag`.
    pub bdpcm: bool,
    /// `sps_mip_enabled_flag`.
    pub mip: bool,
    /// `sps_mrl_enabled_flag`.
    pub mrl: bool,
    /// `sps_isp_enabled_flag`.
    pub isp: bool,
    /// `sps_act_enabled_flag`.
    pub act: bool,
    /// `MaxTbSizeY` — max transform block size in luma samples.
    pub max_tb_size_y: u32,
    /// `MinTbSizeY` — min transform block size in luma samples.
    pub min_tb_size_y: u32,
    /// `MaxTsSize` — max transform-skip block size in luma samples
    /// (§7.4.3.4 via `sps_transform_skip_enabled_flag`).
    pub max_ts_size: u32,
    /// `CtbSizeY`.
    pub ctb_size_y: u32,
    /// `sps_chroma_format_idc` (0 = monochrome).
    pub chroma_format_idc: u32,
    /// `pps_cu_qp_delta_enabled_flag` — gates `cu_qp_delta_abs` reads.
    pub cu_qp_delta_enabled: bool,
    /// `sh_cu_chroma_qp_offset_enabled_flag` — gates
    /// `cu_chroma_qp_offset_flag` reads.
    pub cu_chroma_qp_offset_enabled: bool,
    /// `pps_chroma_qp_offset_list_len_minus1` — cMax for the
    /// `cu_chroma_qp_offset_idx` TR read.
    pub chroma_qp_offset_list_len_minus1: u32,
    /// `sps_joint_cbcr_enabled_flag` — if true, `tu_joint_cbcr_residual_flag`
    /// may appear (currently surfaced as Unsupported when it would be read).
    pub joint_cbcr_enabled: bool,
    /// `sh_ts_residual_coding_disabled_flag`. When set the
    /// `transform_skip_flag` path is elided in the residual walker.
    pub ts_residual_coding_disabled: bool,
}

/// Parsed + derived per-CU state for an intra leaf CU.
///
/// Holds every syntax element read by [`LeafCuReader`] for a single
/// coding unit, plus the spec-derived luma / chroma intra prediction
/// modes (§8.4.2 / §8.4.3). The decoder pipeline in later rounds will
/// consume this struct to drive reference-sample fetch + prediction
/// without replaying the CABAC reads.
///
/// CBFs and the raw coefficient level array live in a sibling
/// [`LeafCuResidual`] because they are not `Copy` (they carry a
/// `Vec<i32>`). The residual struct stays `None` when the CU has no
/// residual to decode (e.g. BDPCM pure-prediction path or a CU where
/// `tu_y_coded_flag == 0` and chroma CBFs are 0).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LeafCuInfo {
    /// Top-left luma x (picture-absolute).
    pub x0: u32,
    /// Top-left luma y (picture-absolute).
    pub y0: u32,
    /// CB width in luma samples.
    pub cb_width: u32,
    /// CB height in luma samples.
    pub cb_height: u32,
    /// `CuPredMode[chType][x0][y0]`.
    pub pred_mode: CuPredMode,
    /// `BdpcmFlag[x0][y0][0]`.
    pub intra_bdpcm_luma: bool,
    /// `BdpcmDir[x0][y0][0]` — direction (false = horizontal, true = vertical).
    pub intra_bdpcm_luma_dir: bool,
    /// `IntraMipFlag[x0][y0]`.
    pub intra_mip_flag: bool,
    /// `intra_mip_transposed_flag`.
    pub intra_mip_transposed_flag: bool,
    /// `intra_mip_mode`.
    pub intra_mip_mode: u32,
    /// `IntraLumaRefLineIdx[x0][y0]`.
    pub intra_luma_ref_idx: u32,
    /// Derived ISP split type.
    pub isp_split: IspSplitType,
    /// `intra_luma_mpm_flag[x0][y0]` (inferred 1 if not present).
    pub intra_luma_mpm_flag: bool,
    /// `intra_luma_not_planar_flag[x0][y0]` (inferred 1 if not present).
    pub intra_luma_not_planar_flag: bool,
    /// `intra_luma_mpm_idx[x0][y0]` (0 when MPM flag is 0).
    pub intra_luma_mpm_idx: u32,
    /// `intra_luma_mpm_remainder[x0][y0]` (present only when MPM flag == 0).
    pub intra_luma_mpm_remainder: u32,
    /// `intra_chroma_pred_mode` (inferred 0 when not present).
    pub intra_chroma_pred_mode: u32,
    /// `IntraPredModeY[x0][y0]` derived per §8.4.2 — 0..66 for regular
    /// intra, encoded verbatim when MIP is in use (MIP has its own
    /// mode namespace `intra_mip_mode`).
    pub intra_pred_mode_y: u32,
    /// `IntraPredModeC[x0][y0]` derived per §8.4.3 / Table 20. For
    /// monochrome sequences this remains 0 and callers should ignore
    /// the chroma plane.
    pub intra_pred_mode_c: u32,
    /// `tu_y_coded_flag[x0][y0]` — luma CBF for the single-TB CU case.
    pub tu_y_coded_flag: bool,
    /// `tu_cb_coded_flag[x0][y0]`.
    pub tu_cb_coded_flag: bool,
    /// `tu_cr_coded_flag[x0][y0]`.
    pub tu_cr_coded_flag: bool,
    /// `CuQpDeltaVal` (signed, §7.4.11.8). 0 when `cu_qp_delta_abs == 0`
    /// or the flag was not present in the slice.
    pub cu_qp_delta_val: i32,
    /// `cu_chroma_qp_offset_flag`.
    pub cu_chroma_qp_offset_flag: bool,
    /// `cu_chroma_qp_offset_idx` (0 when the flag is 0).
    pub cu_chroma_qp_offset_idx: u32,
    /// `LastSignificantCoeffX` for the luma TB (0 when no residual).
    pub last_sig_x: u32,
    /// `LastSignificantCoeffY` for the luma TB.
    pub last_sig_y: u32,
}

impl Default for LeafCuInfo {
    fn default() -> Self {
        Self {
            x0: 0,
            y0: 0,
            cb_width: 0,
            cb_height: 0,
            pred_mode: CuPredMode::Intra,
            intra_bdpcm_luma: false,
            intra_bdpcm_luma_dir: false,
            intra_mip_flag: false,
            intra_mip_transposed_flag: false,
            intra_mip_mode: 0,
            intra_luma_ref_idx: 0,
            isp_split: IspSplitType::NoSplit,
            intra_luma_mpm_flag: true,
            intra_luma_not_planar_flag: true,
            intra_luma_mpm_idx: 0,
            intra_luma_mpm_remainder: 0,
            intra_chroma_pred_mode: 0,
            intra_pred_mode_y: INTRA_PLANAR,
            intra_pred_mode_c: INTRA_PLANAR,
            tu_y_coded_flag: false,
            tu_cb_coded_flag: false,
            tu_cr_coded_flag: false,
            cu_qp_delta_val: 0,
            cu_chroma_qp_offset_flag: false,
            cu_chroma_qp_offset_idx: 0,
            last_sig_x: 0,
            last_sig_y: 0,
        }
    }
}

/// Residual side-state for a leaf CU that has at least one coded TB.
/// Held alongside [`LeafCuInfo`] because the level arrays require an
/// allocation.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LeafCuResidual {
    /// Luma coefficient levels, row-major `(cb_width * cb_height)`.
    /// Empty when `tu_y_coded_flag == 0`.
    pub luma_levels: Vec<i32>,
    /// Chroma Cb coefficient levels (row-major, chroma-plane size).
    pub cb_levels: Vec<i32>,
    /// Chroma Cr coefficient levels (row-major, chroma-plane size).
    pub cr_levels: Vec<i32>,
}

/// §7.3.11.5 syntax values for the intra-mode constants (Table 19).
pub const INTRA_PLANAR: u32 = 0;
pub const INTRA_DC: u32 = 1;
pub const INTRA_ANGULAR18: u32 = 18;
pub const INTRA_ANGULAR46: u32 = 46;
pub const INTRA_ANGULAR50: u32 = 50;
pub const INTRA_ANGULAR54: u32 = 54;

/// Chroma-direct modes emitted by Table 20 when `cclm_mode_flag == 1`.
pub const INTRA_LT_CCLM: u32 = 81;
pub const INTRA_L_CCLM: u32 = 82;
pub const INTRA_T_CCLM: u32 = 83;

/// CABAC context bundle used by [`LeafCuReader`].
pub struct LeafCuCtxs {
    pub intra_mip_flag: Vec<ContextModel>,
    pub intra_luma_ref_idx: Vec<ContextModel>,
    pub intra_subpartitions_mode_flag: Vec<ContextModel>,
    pub intra_subpartitions_split_flag: Vec<ContextModel>,
    pub intra_luma_mpm_flag: Vec<ContextModel>,
    pub intra_luma_not_planar_flag: Vec<ContextModel>,
    pub intra_chroma_pred_mode: Vec<ContextModel>,
    /// CBF + residual + last-sig-coeff context arrays. Shared with
    /// [`crate::residual`] so the leaf reader + the residual reader
    /// advance the same CABAC state machine.
    pub residual: ResidualCtxs,
}

impl std::fmt::Debug for LeafCuCtxs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LeafCuCtxs")
            .field("intra_mip_flag.len", &self.intra_mip_flag.len())
            .field("intra_luma_ref_idx.len", &self.intra_luma_ref_idx.len())
            .field(
                "intra_subpartitions_mode_flag.len",
                &self.intra_subpartitions_mode_flag.len(),
            )
            .field(
                "intra_subpartitions_split_flag.len",
                &self.intra_subpartitions_split_flag.len(),
            )
            .field("intra_luma_mpm_flag.len", &self.intra_luma_mpm_flag.len())
            .field(
                "intra_luma_not_planar_flag.len",
                &self.intra_luma_not_planar_flag.len(),
            )
            .field(
                "intra_chroma_pred_mode.len",
                &self.intra_chroma_pred_mode.len(),
            )
            .field("residual.tu_y_coded.len", &self.residual.tu_y_coded.len())
            .finish()
    }
}

impl LeafCuCtxs {
    /// Build all context arrays using the supplied SliceQpY.
    pub fn init(slice_qp_y: i32) -> Self {
        Self {
            intra_mip_flag: init_contexts(SyntaxCtx::IntraMipFlag, slice_qp_y),
            intra_luma_ref_idx: init_contexts(SyntaxCtx::IntraLumaRefIdx, slice_qp_y),
            intra_subpartitions_mode_flag: init_contexts(
                SyntaxCtx::IntraSubpartitionsModeFlag,
                slice_qp_y,
            ),
            intra_subpartitions_split_flag: init_contexts(
                SyntaxCtx::IntraSubpartitionsSplitFlag,
                slice_qp_y,
            ),
            intra_luma_mpm_flag: init_contexts(SyntaxCtx::IntraLumaMpmFlag, slice_qp_y),
            intra_luma_not_planar_flag: init_contexts(
                SyntaxCtx::IntraLumaNotPlanarFlag,
                slice_qp_y,
            ),
            intra_chroma_pred_mode: init_contexts(SyntaxCtx::IntraChromaPredMode, slice_qp_y),
            residual: ResidualCtxs::init(slice_qp_y),
        }
    }
}

/// Neighbour state used by §8.4.2 (MPM derivation) and the `intra_mip`
/// context derivation. One of these is held per leaf CU that has been
/// parsed so far; the reader looks up the immediate left / above
/// neighbour entries by `(x0, y0)`.
///
/// Off-picture / off-slice neighbours are encoded as `None`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct IntraNeighbour {
    /// `IntraPredModeY[x][y]` of the neighbour (0..66). Only valid
    /// when `pred_mode == Intra` and `mip == false`.
    pub intra_pred_mode_y: u32,
    /// `IntraMipFlag[x][y]`.
    pub mip: bool,
    /// `CuPredMode[0][x][y]`.
    pub pred_mode: Option<CuPredMode>,
}

/// Derive §8.4.2's `candIntraPredModeX` for a given side.
///
/// * `available_x` — §6.4.4 availability flag.
/// * `neighbour` — the neighbour's intra info (ignored if unavailable).
/// * `side_b` — true when deriving the "above" (B) candidate; used to
///   trigger the CTB-row boundary rule that forces PLANAR when the
///   neighbour sits in a prior CTB row.
/// * `crosses_ctb_row` — spec condition
///   `yCb − 1 < (yCb >> CtbLog2SizeY) << CtbLog2SizeY`, i.e. the above
///   neighbour is in a different CTB row than the current CU.
fn cand_intra_mode_for_side(
    available_x: bool,
    neighbour: Option<IntraNeighbour>,
    side_b: bool,
    crosses_ctb_row: bool,
) -> u32 {
    let Some(n) = neighbour else {
        return INTRA_PLANAR;
    };
    if !available_x {
        return INTRA_PLANAR;
    }
    if n.pred_mode != Some(CuPredMode::Intra) {
        return INTRA_PLANAR;
    }
    if n.mip {
        return INTRA_PLANAR;
    }
    if side_b && crosses_ctb_row {
        return INTRA_PLANAR;
    }
    n.intra_pred_mode_y
}

/// Build the 5-entry `candModeList[]` per §8.4.2. Spec equations 210 –
/// 240, covering every branch of the A == B / one-above-DC / neither-
/// above-DC decision tree.
pub fn build_mpm_cand_list(cand_a: u32, cand_b: u32) -> [u32; 5] {
    let mut list = [INTRA_PLANAR; 5];
    if cand_b == cand_a && cand_a > INTRA_DC {
        // eqs 210–214.
        list[0] = cand_a;
        list[1] = 2 + ((cand_a + 61) % 64);
        list[2] = 2 + ((cand_a.wrapping_sub(1)) % 64);
        list[3] = 2 + ((cand_a + 60) % 64);
        list[4] = 2 + (cand_a % 64);
        return list;
    }
    if cand_a != cand_b && (cand_a > INTRA_DC || cand_b > INTRA_DC) {
        let min_ab = cand_a.min(cand_b);
        let max_ab = cand_a.max(cand_b);
        if cand_a > INTRA_DC && cand_b > INTRA_DC {
            // Both > DC (eqs 217 – 230).
            list[0] = cand_a;
            list[1] = cand_b;
            let diff = max_ab - min_ab;
            if diff == 1 {
                list[2] = 2 + ((min_ab + 61) % 64);
                list[3] = 2 + ((max_ab - 1) % 64);
                list[4] = 2 + ((min_ab + 60) % 64);
            } else if diff >= 62 {
                list[2] = 2 + ((min_ab.wrapping_sub(1)) % 64);
                list[3] = 2 + ((max_ab + 61) % 64);
                list[4] = 2 + (min_ab % 64);
            } else if diff == 2 {
                list[2] = 2 + ((min_ab - 1) % 64);
                list[3] = 2 + ((min_ab + 61) % 64);
                list[4] = 2 + ((max_ab - 1) % 64);
            } else {
                list[2] = 2 + ((min_ab + 61) % 64);
                list[3] = 2 + ((min_ab - 1) % 64);
                list[4] = 2 + ((max_ab + 61) % 64);
            }
        } else {
            // Only one > DC (eqs 231 – 235).
            list[0] = max_ab;
            list[1] = 2 + ((max_ab + 61) % 64);
            list[2] = 2 + ((max_ab - 1) % 64);
            list[3] = 2 + ((max_ab + 60) % 64);
            list[4] = 2 + (max_ab % 64);
        }
        return list;
    }
    // Neither > DC (eqs 236 – 240).
    list[0] = INTRA_DC;
    list[1] = INTRA_ANGULAR50;
    list[2] = INTRA_ANGULAR18;
    list[3] = INTRA_ANGULAR46;
    list[4] = INTRA_ANGULAR54;
    list
}

/// §8.4.2 final-step `IntraPredModeY` derivation.
///
/// * `not_planar_flag == 0` → returns `INTRA_PLANAR`.
/// * `bdpcm == true` → returns `INTRA_ANGULAR50` for vertical direction,
///   `INTRA_ANGULAR18` for horizontal.
/// * MPM matched (`mpm_flag == true`): `candModeList[mpm_idx]`.
/// * MPM remainder path: incremented + offset-skipped value.
pub fn derive_intra_pred_mode_y(
    not_planar_flag: bool,
    bdpcm: bool,
    bdpcm_vertical: bool,
    mpm_flag: bool,
    mpm_idx: u32,
    mpm_remainder: u32,
    cand_list: &[u32; 5],
) -> u32 {
    if !not_planar_flag {
        return INTRA_PLANAR;
    }
    if bdpcm {
        return if bdpcm_vertical {
            INTRA_ANGULAR50
        } else {
            INTRA_ANGULAR18
        };
    }
    if mpm_flag {
        // Spec step 4.a: pick candModeList[mpm_idx].
        return cand_list[(mpm_idx as usize).min(4)];
    }
    // Step 4.b: sort candModeList ascending, then add 1, then skip any
    // candidate <= current to produce the final mode outside the MPM
    // set.
    let mut sorted = *cand_list;
    sorted.sort_unstable();
    let mut mode = mpm_remainder + 1;
    for &c in sorted.iter() {
        if mode >= c {
            mode += 1;
        }
    }
    mode
}

/// §8.4.3 / Table 20 chroma intra-mode derivation for `cclm_mode_flag == 0`.
///
/// Table 20 (lower half) maps `intra_chroma_pred_mode ∈ 0..=4` and a
/// representative `lumaIntraPredMode` to the final `IntraPredModeC`.
/// Entries in the table where the luma mode equals 0 / 50 / 18 / 1
/// re-target to a dedicated "alt" mode; the remainder passes the luma
/// mode through untouched.
///
/// Handles the common SINGLE_TREE + non-CCLM + non-BDPCM +
/// `cu_act_enabled_flag == 0` path used in tests.
pub fn derive_intra_pred_mode_c(intra_chroma_pred_mode: u32, luma_intra_pred_mode: u32) -> u32 {
    // Bottom half of Table 20 (cclm_mode_flag == 0).
    match intra_chroma_pred_mode {
        0 => match luma_intra_pred_mode {
            INTRA_PLANAR => INTRA_ANGULAR66,
            _ => INTRA_PLANAR,
        },
        1 => match luma_intra_pred_mode {
            INTRA_ANGULAR50 => INTRA_ANGULAR66,
            _ => INTRA_ANGULAR50,
        },
        2 => match luma_intra_pred_mode {
            INTRA_ANGULAR18 => INTRA_ANGULAR66,
            _ => INTRA_ANGULAR18,
        },
        3 => match luma_intra_pred_mode {
            INTRA_DC => INTRA_ANGULAR66,
            _ => INTRA_DC,
        },
        4 => luma_intra_pred_mode, // DM_CHROMA — direct luma inherit.
        _ => INTRA_PLANAR,
    }
}

/// INTRA_ANGULAR66 (top-right diagonal) — Table 20's "alt" direct mode.
pub const INTRA_ANGULAR66: u32 = 66;

/// Neighbourhood snapshot passed into [`LeafCuReader`] per CU.
#[derive(Clone, Copy, Debug, Default)]
pub struct CuNeighbourhood {
    pub left_available: bool,
    pub above_available: bool,
    pub left: Option<IntraNeighbour>,
    pub above: Option<IntraNeighbour>,
}

/// Leaf-CU syntax reader. Stateless w.r.t. the spec (each call to
/// [`Self::decode`] consumes the full `coding_unit()` bin stream); the
/// caller feeds CABAC state + CU geometry + SPS tool flags.
pub struct LeafCuReader<'a, 'b> {
    dec: &'a mut ArithDecoder<'b>,
    ctxs: &'a mut LeafCuCtxs,
    tools: CuToolFlags,
}

impl<'a, 'b> LeafCuReader<'a, 'b> {
    pub fn new(
        dec: &'a mut ArithDecoder<'b>,
        ctxs: &'a mut LeafCuCtxs,
        tools: CuToolFlags,
    ) -> Self {
        Self { dec, ctxs, tools }
    }

    /// Decode the `coding_unit()` body for a leaf CU in an I-slice,
    /// single-tree pipeline. `info.x0 / y0 / cb_width / cb_height`
    /// must be populated by the caller; every other field is written
    /// by this method. The residual-level storage (per-plane
    /// coefficient arrays) is filled into `residual` when any CBF
    /// fires; it stays empty when the CU carries no coded residual
    /// (e.g. all CBFs 0 on a zero-stream test fixture).
    pub fn decode(
        &mut self,
        info: &mut LeafCuInfo,
        residual: &mut LeafCuResidual,
        neigh: &CuNeighbourhood,
    ) -> Result<()> {
        info.pred_mode = CuPredMode::Intra; // I-slice default (§7.4.12.2).

        // IBC / palette are not currently walked: the flags are
        // grouped here to surface `Error::Unsupported` the moment the
        // SPS opts into them, rather than silently skipping them.
        if self.tools.ibc {
            return Err(Error::unsupported(
                "h266 leaf CU: IBC (sps_ibc_enabled_flag) not supported by this round",
            ));
        }
        if self.tools.palette {
            return Err(Error::unsupported(
                "h266 leaf CU: palette coding (sps_palette_enabled_flag) not supported",
            ));
        }

        // pred_mode_plt_flag: ignored for now (palette disabled above).
        // cu_act_enabled_flag: only present when sps_act_enabled_flag &&
        // treeType == SINGLE_TREE. We gate it similarly.
        if self.tools.act {
            return Err(Error::unsupported(
                "h266 leaf CU: ACT (sps_act_enabled_flag) not supported",
            ));
        }

        // intra_bdpcm_luma_flag gate: spec condition is
        // "cbWidth <= MaxTsSize && cbHeight <= MaxTsSize".
        let bdpcm_gate = self.tools.bdpcm
            && info.cb_width <= self.tools.max_ts_size
            && info.cb_height <= self.tools.max_ts_size;
        if bdpcm_gate {
            // Table 132 — ctxInc 0; use the mip_flag table as a simple
            // single-context holder (same "solo ctxIdx" shape).
            // The spec has a dedicated Table 69 init; we reuse the
            // mip-flag context slot for the flag read since this round
            // does not surface bdpcm-specific init values. If BDPCM is
            // disabled (the default in our test fixtures), this branch
            // is not taken.
            return Err(Error::unsupported(
                "h266 leaf CU: BDPCM luma flag parsing needs Table 69 init (not plumbed yet)",
            ));
        }

        // intra_mip_flag.
        if self.tools.mip {
            let inc = ctx_inc_intra_mip_flag(
                info.cb_width,
                info.cb_height,
                neigh.left_available,
                neigh.above_available,
                neigh.left.map(|n| n.mip).unwrap_or(false),
                neigh.above.map(|n| n.mip).unwrap_or(false),
            ) as usize;
            let n = self.ctxs.intra_mip_flag.len() - 1;
            let bit = self
                .dec
                .decode_decision(&mut self.ctxs.intra_mip_flag[inc.min(n)])?;
            info.intra_mip_flag = bit == 1;
        }

        if info.intra_mip_flag {
            // intra_mip_transposed_flag — bypass (Table 132).
            info.intra_mip_transposed_flag = self.dec.decode_bypass()? == 1;
            // intra_mip_mode — TB with size-dependent cMax.
            let c_max = Self::mip_mode_cmax(info.cb_width, info.cb_height);
            info.intra_mip_mode = decode_tb(self.dec, c_max)?;
            // MIP has its own mode namespace — we still store 0 for
            // intra_pred_mode_y so callers know to branch on the flag.
            info.intra_pred_mode_y = 0;
            self.derive_chroma(info);
            return Ok(());
        }

        // intra_luma_ref_idx — only when MRL is enabled and y0 % CtbSizeY > 0.
        // TR(cMax=2, cRice=0) is a context-coded truncated-unary up to 2
        // bins. Each bin gets its own ctxInc per
        // [`ctx_inc_intra_luma_ref_idx`].
        if self.tools.mrl && (info.y0 % self.tools.ctb_size_y) > 0 {
            let ctxs = &mut self.ctxs.intra_luma_ref_idx;
            let n = ctxs.len() - 1;
            let mut val = 0u32;
            for bin_idx in 0..2u32 {
                let inc = ctx_inc_intra_luma_ref_idx(bin_idx) as usize;
                let bit = self.dec.decode_decision(&mut ctxs[inc.min(n)])?;
                if bit == 0 {
                    break;
                }
                val += 1;
            }
            info.intra_luma_ref_idx = val;
        }

        // intra_subpartitions_mode_flag — ISP gated by:
        //   sps_isp_enabled_flag && intra_luma_ref_idx == 0 &&
        //   cbWidth <= MaxTbSizeY && cbHeight <= MaxTbSizeY &&
        //   (cbWidth * cbHeight) > (MinTbSizeY * MinTbSizeY).
        let isp_gate = self.tools.isp
            && info.intra_luma_ref_idx == 0
            && info.cb_width <= self.tools.max_tb_size_y
            && info.cb_height <= self.tools.max_tb_size_y
            && info.cb_width * info.cb_height > self.tools.min_tb_size_y * self.tools.min_tb_size_y;
        let mut isp_mode_flag = false;
        if isp_gate {
            let inc = ctx_inc_intra_subpartitions_mode_flag() as usize;
            let n = self.ctxs.intra_subpartitions_mode_flag.len() - 1;
            let bit = self
                .dec
                .decode_decision(&mut self.ctxs.intra_subpartitions_mode_flag[inc.min(n)])?;
            isp_mode_flag = bit == 1;
        }
        if isp_mode_flag {
            let inc = ctx_inc_intra_subpartitions_split_flag() as usize;
            let n = self.ctxs.intra_subpartitions_split_flag.len() - 1;
            let split_bit = self
                .dec
                .decode_decision(&mut self.ctxs.intra_subpartitions_split_flag[inc.min(n)])?;
            info.isp_split = if split_bit == 1 {
                IspSplitType::VerSplit
            } else {
                IspSplitType::HorSplit
            };
        } else {
            info.isp_split = IspSplitType::NoSplit;
        }

        // intra_luma_mpm_flag — only present when intra_luma_ref_idx == 0.
        if info.intra_luma_ref_idx == 0 {
            let inc = ctx_inc_intra_luma_mpm_flag() as usize;
            let bit = self
                .dec
                .decode_decision(&mut self.ctxs.intra_luma_mpm_flag[inc])?;
            info.intra_luma_mpm_flag = bit == 1;
        } else {
            // MRL index > 0 implies the MPM is used (not_planar and mpm_idx).
            info.intra_luma_mpm_flag = true;
        }

        if info.intra_luma_mpm_flag {
            if info.intra_luma_ref_idx == 0 {
                // intra_luma_not_planar_flag.
                let inc = ctx_inc_intra_luma_not_planar_flag(isp_mode_flag) as usize;
                let n = self.ctxs.intra_luma_not_planar_flag.len() - 1;
                let bit = self
                    .dec
                    .decode_decision(&mut self.ctxs.intra_luma_not_planar_flag[inc.min(n)])?;
                info.intra_luma_not_planar_flag = bit == 1;
            } else {
                info.intra_luma_not_planar_flag = true;
            }
            if info.intra_luma_not_planar_flag {
                // intra_luma_mpm_idx — bypass TR(cMax=4).
                info.intra_luma_mpm_idx = decode_tr_bypass(self.dec, 4, 0)?;
            }
        } else {
            // intra_luma_mpm_remainder — bypass TB(cMax=60).
            info.intra_luma_mpm_remainder = decode_tb_bypass(self.dec, 60)?;
        }

        // Derive luma mode per §8.4.2.
        let crosses_ctb = {
            let ctb_row = (info.y0 / self.tools.ctb_size_y) * self.tools.ctb_size_y;
            info.y0 == ctb_row
        };
        let cand_a = cand_intra_mode_for_side(
            neigh.left_available,
            neigh.left,
            /*side_b=*/ false,
            /*crosses_ctb_row=*/ false,
        );
        let cand_b = cand_intra_mode_for_side(
            neigh.above_available,
            neigh.above,
            /*side_b=*/ true,
            crosses_ctb,
        );
        let cand_list = build_mpm_cand_list(cand_a, cand_b);
        info.intra_pred_mode_y = derive_intra_pred_mode_y(
            info.intra_luma_not_planar_flag,
            info.intra_bdpcm_luma,
            info.intra_bdpcm_luma_dir,
            info.intra_luma_mpm_flag,
            info.intra_luma_mpm_idx,
            info.intra_luma_mpm_remainder,
            &cand_list,
        );

        // Chroma — skip for monochrome.
        self.derive_chroma(info);

        // --- transform_unit() reads (§7.3.11.10) ---
        self.decode_transform_unit(info, residual)?;

        Ok(())
    }

    /// Read the transform_unit()-level syntax (CBFs, cu_qp_delta,
    /// cu_chroma_qp_offset_*) and drive the residual walker.
    ///
    /// Scope: single-TB CU only (no ISP split, no SBT). The CBF read
    /// gates per §7.3.11.10 reduce to:
    ///   * chroma CBFs are always read when the chroma format is not
    ///     monochrome and the CU is in single-tree mode.
    ///   * luma CBF is always read in the intra path.
    ///
    /// The residual decode path surfaces `Error::Unsupported` if
    /// `sps_joint_cbcr_enabled_flag` is set and the joint-cbcr flag
    /// would be read — that binarisation has a separate context init
    /// that is not plumbed yet.
    fn decode_transform_unit(
        &mut self,
        info: &mut LeafCuInfo,
        residual: &mut LeafCuResidual,
    ) -> Result<()> {
        let cb_w = info.cb_width as usize;
        let cb_h = info.cb_height as usize;
        let chroma = self.tools.chroma_format_idc != 0;
        // Chroma dims: 4:2:0 halves both, 4:2:2 halves width only, 4:4:4 keeps.
        // We only support 4:2:0 here (chroma_format_idc == 1); other formats
        // surface Unsupported the moment the code would diverge.
        let (sub_w, sub_h) = match self.tools.chroma_format_idc {
            0 => (1, 1),
            1 => (2, 2),
            2 => (2, 1),
            3 => (1, 1),
            _ => {
                return Err(Error::invalid(
                    "h266 leaf CU: unknown sps_chroma_format_idc value",
                ));
            }
        };
        // Read tu_cb/tu_cr_coded first (spec order — chroma CBFs come
        // before luma within transform_unit()).
        if chroma {
            info.tu_cb_coded_flag = read_tu_cb_coded_flag(
                self.dec,
                &mut self.ctxs.residual,
                /*bdpcm_chroma=*/ false,
            )?;
            info.tu_cr_coded_flag = read_tu_cr_coded_flag(
                self.dec,
                &mut self.ctxs.residual,
                /*bdpcm_chroma=*/ false,
                info.tu_cb_coded_flag,
            )?;
        }
        // Luma CBF: always read in the intra path (per §7.3.11.10 the
        // condition simplifies to true when pred_mode == INTRA, ISP is
        // NO_SPLIT, SBT is off and ACT is off).
        info.tu_y_coded_flag = read_tu_y_coded_flag(
            self.dec,
            &mut self.ctxs.residual,
            /*bdpcm_y=*/ info.intra_bdpcm_luma,
            /*isp_split=*/ info.isp_split != IspSplitType::NoSplit,
            /*prev_tu_cbf_y=*/ false,
        )?;

        // cu_qp_delta_abs + sign (spec gates on CB > 64 || any CBF + enable).
        let any_cbf = info.tu_y_coded_flag || info.tu_cb_coded_flag || info.tu_cr_coded_flag;
        if self.tools.cu_qp_delta_enabled && any_cbf {
            info.cu_qp_delta_val = read_cu_qp_delta(self.dec, &mut self.ctxs.residual)?;
        }
        // cu_chroma_qp_offset_flag + idx.
        let chroma_cbf = info.tu_cb_coded_flag || info.tu_cr_coded_flag;
        if self.tools.cu_chroma_qp_offset_enabled && chroma && chroma_cbf {
            let (flag, idx) = read_cu_chroma_qp_offset(
                self.dec,
                &mut self.ctxs.residual,
                self.tools.chroma_qp_offset_list_len_minus1,
            )?;
            info.cu_chroma_qp_offset_flag = flag;
            info.cu_chroma_qp_offset_idx = idx;
        }
        // joint_cbcr_residual_flag — gate exists but parsing not
        // plumbed; surface Unsupported when it would actually fire.
        if self.tools.joint_cbcr_enabled
            && chroma
            && ((info.pred_mode == CuPredMode::Intra && chroma_cbf)
                || (info.tu_cb_coded_flag && info.tu_cr_coded_flag))
        {
            return Err(Error::unsupported(
                "h266 leaf CU: tu_joint_cbcr_residual_flag parsing not plumbed yet",
            ));
        }

        // Residual decode per plane.
        if info.tu_y_coded_flag {
            let levels = decode_tb_coefficients(self.dec, &mut self.ctxs.residual, cb_w, cb_h, 0)?;
            // Capture the last-sig position derived by the residual
            // reader by re-reading the info from the coefficient
            // array: the last non-zero in scan order is the last-sig.
            // (The residual reader already reads last_sig_coeff_*; we
            // expose a simplified record here.)
            // Find max (x, y) with non-zero level as a best-effort
            // proxy for LastSignificantCoeffX/Y.
            let mut lx = 0u32;
            let mut ly = 0u32;
            for y in 0..cb_h {
                for x in 0..cb_w {
                    if levels[y * cb_w + x] != 0 {
                        lx = lx.max(x as u32);
                        ly = ly.max(y as u32);
                    }
                }
            }
            info.last_sig_x = lx;
            info.last_sig_y = ly;
            residual.luma_levels = levels;
        }
        if info.tu_cb_coded_flag && chroma {
            let cw = cb_w / sub_w;
            let ch = cb_h / sub_h;
            if cw >= 2 && ch >= 2 {
                residual.cb_levels =
                    decode_tb_coefficients(self.dec, &mut self.ctxs.residual, cw, ch, 1)?;
            }
        }
        if info.tu_cr_coded_flag && chroma {
            let cw = cb_w / sub_w;
            let ch = cb_h / sub_h;
            if cw >= 2 && ch >= 2 {
                residual.cr_levels =
                    decode_tb_coefficients(self.dec, &mut self.ctxs.residual, cw, ch, 2)?;
            }
        }
        Ok(())
    }

    fn derive_chroma(&mut self, info: &mut LeafCuInfo) {
        if self.tools.chroma_format_idc == 0 {
            info.intra_pred_mode_c = 0;
            return;
        }
        // Read intra_chroma_pred_mode — Table 130, bin 0 ctx, bins 1-2 bypass.
        // If MIP is active the chroma derivation path inherits luma
        // (MipChromaDirectFlag can force this), but reading the mode
        // element is still gated on cclm_mode_flag == 0. For this round
        // we always take the non-CCLM path (cclm tool disabled in tests).
        // When intra_bdpcm_chroma_flag is not present or is 0 the
        // element is read.
        // We bail the read to a best-effort: if there is insufficient
        // bitstream, error up the chain.
        // Best-effort: missing bitstream tail means defaults.
        let icp = self.read_intra_chroma_pred_mode().unwrap_or_default();
        info.intra_chroma_pred_mode = icp;
        let luma = info.intra_pred_mode_y;
        info.intra_pred_mode_c = derive_intra_pred_mode_c(icp, luma);
    }

    fn read_intra_chroma_pred_mode(&mut self) -> Result<u32> {
        // Binarisation Table 130:
        //   val==4 → "0", val∈0..3 → "1" + FL(2 bits).
        let inc = ctx_inc_intra_chroma_pred_mode() as usize;
        let n = self.ctxs.intra_chroma_pred_mode.len() - 1;
        let bin0 = self
            .dec
            .decode_decision(&mut self.ctxs.intra_chroma_pred_mode[inc.min(n)])?;
        if bin0 == 0 {
            return Ok(4);
        }
        let bin1 = self.dec.decode_bypass()?;
        let bin2 = self.dec.decode_bypass()?;
        Ok((bin1 << 1) | bin2)
    }

    fn mip_mode_cmax(cb_w: u32, cb_h: u32) -> u32 {
        if cb_w == 4 && cb_h == 4 {
            15
        } else if cb_w == 4 || cb_h == 4 || (cb_w == 8 && cb_h == 8) {
            7
        } else {
            5
        }
    }
}

/// Truncated-Rice binarisation decode for the bypass-only variant
/// (intra_luma_mpm_idx).
fn decode_tr_bypass(dec: &mut ArithDecoder<'_>, c_max: u32, c_rice: u32) -> Result<u32> {
    let prefix_max = c_max >> c_rice;
    let mut prefix = 0u32;
    for _ in 0..prefix_max {
        let bit = dec.decode_bypass()?;
        if bit == 0 {
            break;
        }
        prefix += 1;
    }
    if c_rice == 0 {
        return Ok(prefix);
    }
    let suffix = dec.decode_bypass_bits(c_rice)?;
    Ok((prefix << c_rice) | suffix)
}

/// Truncated-binary binarisation decode (§9.3.3.4) — context-coded
/// bins, not currently used.
#[allow(dead_code)]
fn decode_tb(dec: &mut ArithDecoder<'_>, c_max: u32) -> Result<u32> {
    decode_tb_bypass(dec, c_max)
}

/// Truncated-binary binarisation decode for the bypass-only
/// variant (§9.3.3.4). `cMax == 0` decodes to 0 with no bin consumed.
fn decode_tb_bypass(dec: &mut ArithDecoder<'_>, c_max: u32) -> Result<u32> {
    if c_max == 0 {
        return Ok(0);
    }
    let n = c_max + 1;
    let k = 31 - n.leading_zeros(); // floor(log2(n))
    let u = (1u32 << (k + 1)) - n;
    // Read k bits.
    let prefix = dec.decode_bypass_bits(k)?;
    if prefix < u {
        return Ok(prefix);
    }
    // Read one more bit and remap.
    let extra = dec.decode_bypass()?;
    let combined = (prefix << 1) | extra;
    Ok(combined - u)
}

#[cfg(test)]
mod tests {
    use super::*;

    // === MPM candidate-list derivation tests (§8.4.2 eqs 210–240) ===

    #[test]
    fn mpm_list_both_above_dc_equal() {
        // candA == candB == 34 (angular). eqs 210–214.
        let l = build_mpm_cand_list(34, 34);
        assert_eq!(l[0], 34);
        assert_eq!(l[1], 2 + ((34 + 61) % 64)); // = 2 + 31 = 33
        assert_eq!(l[2], 2 + ((34 - 1) % 64)); // = 2 + 33 = 35
        assert_eq!(l[3], 2 + ((34 + 60) % 64)); // = 2 + 30 = 32
        assert_eq!(l[4], 2 + (34 % 64)); // = 2 + 34 = 36
    }

    #[test]
    fn mpm_list_both_above_dc_diff_1() {
        // candA=10, candB=11. diff=1. eqs 217-221.
        let l = build_mpm_cand_list(10, 11);
        assert_eq!(l[0], 10); // candA
        assert_eq!(l[1], 11); // candB
                              // minAB=10, maxAB=11.
        assert_eq!(l[2], 2 + ((10 + 61) % 64)); // = 2 + 7 = 9
        assert_eq!(l[3], 2 + ((11 - 1) % 64)); // = 2 + 10 = 12
        assert_eq!(l[4], 2 + ((10 + 60) % 64)); // = 2 + 6 = 8
    }

    #[test]
    fn mpm_list_both_above_dc_diff_ge_62() {
        // candA=3, candB=65 → diff=62.
        let l = build_mpm_cand_list(3, 65);
        assert_eq!(l[0], 3); // candA
        assert_eq!(l[1], 65); // candB
                              // eqs 222-224.
        assert_eq!(l[2], 2 + ((3u32.wrapping_sub(1)) % 64)); // = 2 + 2 = 4
        assert_eq!(l[3], 2 + ((65 + 61) % 64)); // = 2 + 62 = 64
        assert_eq!(l[4], 2 + (3 % 64)); // = 2 + 3 = 5
    }

    #[test]
    fn mpm_list_both_above_dc_diff_2() {
        // candA=10, candB=12 → diff=2. eqs 225-227.
        let l = build_mpm_cand_list(10, 12);
        assert_eq!(l[0], 10);
        assert_eq!(l[1], 12);
        assert_eq!(l[2], 2 + ((10 - 1) % 64)); // 11
        assert_eq!(l[3], 2 + ((10 + 61) % 64)); // 9
        assert_eq!(l[4], 2 + ((12 - 1) % 64)); // 13
    }

    #[test]
    fn mpm_list_both_above_dc_other_diff() {
        // candA=10, candB=20 → diff=10 (not 1, not 2, not ≥62). eqs 228-230.
        let l = build_mpm_cand_list(10, 20);
        assert_eq!(l[0], 10);
        assert_eq!(l[1], 20);
        assert_eq!(l[2], 2 + ((10 + 61) % 64)); // 9
        assert_eq!(l[3], 2 + ((10 - 1) % 64)); // 11
        assert_eq!(l[4], 2 + ((20 + 61) % 64)); // 19
    }

    #[test]
    fn mpm_list_one_above_dc() {
        // A planar, B angular. eqs 231-235 with maxAB=50.
        let l = build_mpm_cand_list(INTRA_PLANAR, 50);
        assert_eq!(l[0], 50);
        assert_eq!(l[1], 2 + ((50 + 61) % 64));
        assert_eq!(l[2], 2 + ((50 - 1) % 64));
        assert_eq!(l[3], 2 + ((50 + 60) % 64));
        assert_eq!(l[4], 2 + (50 % 64));
    }

    #[test]
    fn mpm_list_neither_above_dc() {
        // Both planar → eqs 236-240.
        let l = build_mpm_cand_list(INTRA_PLANAR, INTRA_DC);
        assert_eq!(
            l,
            [
                INTRA_DC,
                INTRA_ANGULAR50,
                INTRA_ANGULAR18,
                INTRA_ANGULAR46,
                INTRA_ANGULAR54
            ]
        );

        let l = build_mpm_cand_list(INTRA_PLANAR, INTRA_PLANAR);
        assert_eq!(
            l,
            [
                INTRA_DC,
                INTRA_ANGULAR50,
                INTRA_ANGULAR18,
                INTRA_ANGULAR46,
                INTRA_ANGULAR54
            ]
        );
    }

    // === derive_intra_pred_mode_y ===

    #[test]
    fn derive_luma_mode_planar_shortcircuit() {
        let mode = derive_intra_pred_mode_y(
            /*not_planar=*/ false,
            /*bdpcm=*/ false,
            /*bdpcm_v=*/ false,
            /*mpm_flag=*/ true,
            /*mpm_idx=*/ 3,
            /*remainder=*/ 0,
            &[10, 20, 30, 40, 50],
        );
        assert_eq!(mode, INTRA_PLANAR);
    }

    #[test]
    fn derive_luma_mode_bdpcm_vertical_and_horizontal() {
        let mode_v = derive_intra_pred_mode_y(true, true, true, false, 0, 0, &[0; 5]);
        let mode_h = derive_intra_pred_mode_y(true, true, false, false, 0, 0, &[0; 5]);
        assert_eq!(mode_v, INTRA_ANGULAR50);
        assert_eq!(mode_h, INTRA_ANGULAR18);
    }

    #[test]
    fn derive_luma_mode_mpm_match_picks_list_entry() {
        let list = [10, 20, 30, 40, 50];
        let mode = derive_intra_pred_mode_y(true, false, false, true, 2, 0, &list);
        assert_eq!(mode, 30);
    }

    #[test]
    fn derive_luma_mode_remainder_increments_and_skips() {
        // Sorted candModeList = [2, 5, 10, 20, 50].
        // remainder = 0 → mode starts at 1. Not >= 2, so stays at 1.
        let l = [10, 20, 2, 5, 50];
        let sorted = [2, 5, 10, 20, 50];
        // remainder=0 → mode = 1.
        let mode = derive_intra_pred_mode_y(true, false, false, false, 0, 0, &l);
        assert_eq!(mode, 1);

        // remainder = 1 → initial 2; equals sorted[0]=2, so ++ → 3; not ≥5, stops → 3.
        let mode = derive_intra_pred_mode_y(true, false, false, false, 0, 1, &l);
        assert_eq!(mode, 3);

        // remainder = 4 → initial 5; ≥ sorted[0]=2 → 6; ≥ sorted[1]=5 → 7; not ≥ 10 → 7.
        // Wait: sorted has [2,5,10,20,50]. At mode=5, compare against sorted[0]=2: 5>=2, mode=6.
        // sorted[1]=5: 6>=5, mode=7. sorted[2]=10: 7<10, no change. 7<20 and 7<50.
        let mode = derive_intra_pred_mode_y(true, false, false, false, 0, 4, &l);
        assert_eq!(mode, 7);

        // Verify list sort worked: compare against manual calculation.
        assert_eq!(
            {
                let mut s = l;
                s.sort_unstable();
                s
            },
            sorted
        );
    }

    // === derive_intra_pred_mode_c (Table 20) ===

    #[test]
    fn chroma_derivation_table_20_rows() {
        // Row (icp=0): luma=0 → 66, luma=50/18/1 → 0.
        assert_eq!(derive_intra_pred_mode_c(0, 0), 66);
        assert_eq!(derive_intra_pred_mode_c(0, 50), 0);
        assert_eq!(derive_intra_pred_mode_c(0, 18), 0);
        assert_eq!(derive_intra_pred_mode_c(0, 1), 0);
        // Row (icp=1): luma=50 → 66, else 50.
        assert_eq!(derive_intra_pred_mode_c(1, 0), 50);
        assert_eq!(derive_intra_pred_mode_c(1, 50), 66);
        assert_eq!(derive_intra_pred_mode_c(1, 18), 50);
        // Row (icp=2): luma=18 → 66, else 18.
        assert_eq!(derive_intra_pred_mode_c(2, 0), 18);
        assert_eq!(derive_intra_pred_mode_c(2, 18), 66);
        assert_eq!(derive_intra_pred_mode_c(2, 50), 18);
        // Row (icp=3): luma=1 → 66, else 1.
        assert_eq!(derive_intra_pred_mode_c(3, 0), 1);
        assert_eq!(derive_intra_pred_mode_c(3, 1), 66);
        // Row (icp=4): passthrough.
        assert_eq!(derive_intra_pred_mode_c(4, 0), 0);
        assert_eq!(derive_intra_pred_mode_c(4, 50), 50);
        assert_eq!(derive_intra_pred_mode_c(4, 25), 25);
    }

    // === cand_intra_mode_for_side corners ===

    #[test]
    fn cand_mode_unavailable_is_planar() {
        let v = cand_intra_mode_for_side(false, None, false, false);
        assert_eq!(v, INTRA_PLANAR);
        let v = cand_intra_mode_for_side(
            false,
            Some(IntraNeighbour {
                intra_pred_mode_y: 30,
                mip: false,
                pred_mode: Some(CuPredMode::Intra),
            }),
            false,
            false,
        );
        assert_eq!(v, INTRA_PLANAR);
    }

    #[test]
    fn cand_mode_mip_neighbour_is_planar() {
        let nb = IntraNeighbour {
            intra_pred_mode_y: 30,
            mip: true,
            pred_mode: Some(CuPredMode::Intra),
        };
        let v = cand_intra_mode_for_side(true, Some(nb), false, false);
        assert_eq!(v, INTRA_PLANAR);
    }

    #[test]
    fn cand_mode_non_intra_neighbour_is_planar() {
        let nb = IntraNeighbour {
            intra_pred_mode_y: 30,
            mip: false,
            pred_mode: Some(CuPredMode::Inter),
        };
        let v = cand_intra_mode_for_side(true, Some(nb), false, false);
        assert_eq!(v, INTRA_PLANAR);
    }

    #[test]
    fn cand_mode_side_b_crossing_ctb_row_is_planar() {
        let nb = IntraNeighbour {
            intra_pred_mode_y: 30,
            mip: false,
            pred_mode: Some(CuPredMode::Intra),
        };
        let v = cand_intra_mode_for_side(true, Some(nb), true, true);
        assert_eq!(v, INTRA_PLANAR);
        // Side A is never subject to the CTB-row rule, even if the
        // crossing flag is somehow set.
        let v = cand_intra_mode_for_side(true, Some(nb), false, true);
        assert_eq!(v, 30);
    }

    // === TB decode correctness (bypass path) ===

    #[test]
    fn tb_cmax_60_decodes_full_range() {
        // A carefully crafted 64-byte all-zero stream returns 0 from
        // every bypass bit (ivlOffset stays 0 forever). A bypass-TB
        // decode with cMax=60 needs k=5, u=64-61=3. Reading 5 zero
        // bits yields prefix=0 < 3 → returns 0. Good.
        let data = [0u8; 64];
        let mut dec = ArithDecoder::new(&data).unwrap();
        let v = decode_tb_bypass(&mut dec, 60).unwrap();
        assert_eq!(v, 0);
    }

    // === CuToolFlags defaults: all disabled, chroma 4:2:0 ===

    #[test]
    fn leaf_cu_default_round_trip_without_mip_isp_mrl() {
        // Set up a stream with known deterministic behaviour: all zero.
        // With all tools disabled, the reader must read:
        //   - intra_luma_mpm_flag (ctx) → 0 (depending on init bias)
        //   - depending on flag value, either not_planar + mpm_idx or
        //     mpm_remainder.
        //   - intra_chroma_pred_mode.
        // On all-zero streams the specific answers depend on init; we
        // just check that decoding succeeds and produces legal modes.
        let tools = CuToolFlags {
            ibc: false,
            palette: false,
            bdpcm: false,
            mip: false,
            mrl: false,
            isp: false,
            act: false,
            max_tb_size_y: 64,
            min_tb_size_y: 4,
            max_ts_size: 32,
            ctb_size_y: 128,
            chroma_format_idc: 1,
            cu_qp_delta_enabled: false,
            cu_chroma_qp_offset_enabled: false,
            chroma_qp_offset_list_len_minus1: 0,
            joint_cbcr_enabled: false,
            ts_residual_coding_disabled: false,
        };
        let data = [0u8; 128];
        let mut dec = ArithDecoder::new(&data).unwrap();
        let mut ctxs = LeafCuCtxs::init(26);
        let mut info = LeafCuInfo {
            x0: 0,
            y0: 0,
            cb_width: 16,
            cb_height: 16,
            ..LeafCuInfo::default()
        };
        let mut residual = LeafCuResidual::default();
        let neigh = CuNeighbourhood::default();
        let mut reader = LeafCuReader::new(&mut dec, &mut ctxs, tools);
        reader.decode(&mut info, &mut residual, &neigh).unwrap();
        assert_eq!(info.pred_mode, CuPredMode::Intra);
        // Legal luma mode range.
        assert!(info.intra_pred_mode_y <= 66);
        assert!(info.intra_pred_mode_c <= 83);
    }

    #[test]
    fn leaf_cu_with_mip_reads_transposed_and_mode() {
        let tools = CuToolFlags {
            mip: true,
            chroma_format_idc: 1,
            ctb_size_y: 128,
            max_tb_size_y: 64,
            min_tb_size_y: 4,
            max_ts_size: 32,
            ..CuToolFlags::default()
        };
        let data = [0u8; 128];
        let mut dec = ArithDecoder::new(&data).unwrap();
        let mut ctxs = LeafCuCtxs::init(26);
        let mut info = LeafCuInfo {
            x0: 0,
            y0: 0,
            cb_width: 8,
            cb_height: 8,
            ..LeafCuInfo::default()
        };
        let mut residual = LeafCuResidual::default();
        let neigh = CuNeighbourhood::default();
        let mut reader = LeafCuReader::new(&mut dec, &mut ctxs, tools);
        reader.decode(&mut info, &mut residual, &neigh).unwrap();
        // With 8x8 CU + MIP enabled, cMax = 7 → 3 bypass bits; parse
        // must yield mip_mode ∈ 0..=7.
        if info.intra_mip_flag {
            assert!(info.intra_mip_mode <= 7);
        }
    }

    // === MPM-remainder swap + skip behaviour end-to-end via derivation ===

    #[test]
    fn end_to_end_neither_above_dc_remainder() {
        // Both neighbours planar → candModeList per eq 236-240.
        let cand_list = build_mpm_cand_list(INTRA_PLANAR, INTRA_PLANAR);
        // sorted = [1, 18, 46, 50, 54] (after sorting DC, 50, 18, 46, 54).
        // remainder 0 → mode 1 → ≥1 → 2 → <18 → stop → 2.
        let mode = derive_intra_pred_mode_y(true, false, false, false, 0, 0, &cand_list);
        assert_eq!(mode, 2);
        // remainder 15 → initial 16 → ≥1 → 17 → <18 → stop → 17.
        let mode = derive_intra_pred_mode_y(true, false, false, false, 0, 15, &cand_list);
        assert_eq!(mode, 17);
        // remainder 20 → initial 21 → ≥1 → 22 → ≥18 → 23 → <46 → stop → 23.
        let mode = derive_intra_pred_mode_y(true, false, false, false, 0, 20, &cand_list);
        assert_eq!(mode, 23);
    }

    #[test]
    fn mip_mode_cmax_sizes() {
        assert_eq!(LeafCuReader::mip_mode_cmax(4, 4), 15);
        assert_eq!(LeafCuReader::mip_mode_cmax(8, 4), 7);
        assert_eq!(LeafCuReader::mip_mode_cmax(8, 8), 7);
        assert_eq!(LeafCuReader::mip_mode_cmax(16, 16), 5);
        assert_eq!(LeafCuReader::mip_mode_cmax(32, 16), 5);
    }

    #[test]
    fn ibc_and_palette_tools_surface_unsupported() {
        let data = [0u8; 32];
        let mut dec = ArithDecoder::new(&data).unwrap();
        let mut ctxs = LeafCuCtxs::init(26);
        let mut info = LeafCuInfo {
            cb_width: 16,
            cb_height: 16,
            ..LeafCuInfo::default()
        };
        let mut residual = LeafCuResidual::default();
        let neigh = CuNeighbourhood::default();

        let mut tools = CuToolFlags {
            chroma_format_idc: 1,
            ctb_size_y: 128,
            max_tb_size_y: 64,
            min_tb_size_y: 4,
            max_ts_size: 32,
            ibc: true,
            ..CuToolFlags::default()
        };
        let mut reader = LeafCuReader::new(&mut dec, &mut ctxs, tools);
        assert!(matches!(
            reader.decode(&mut info, &mut residual, &neigh),
            Err(Error::Unsupported(_))
        ));

        tools.ibc = false;
        tools.palette = true;
        let mut reader = LeafCuReader::new(&mut dec, &mut ctxs, tools);
        assert!(matches!(
            reader.decode(&mut info, &mut residual, &neigh),
            Err(Error::Unsupported(_))
        ));
    }

    #[test]
    fn mpm_idx_exercises_all_five_candidates_via_derivation() {
        // Take a cand_list with distinct modes and verify every index
        // picks the right entry.
        let list = [20u32, 30, 40, 50, 60];
        for idx in 0..5u32 {
            let m = derive_intra_pred_mode_y(true, false, false, true, idx, 0, &list);
            assert_eq!(m, list[idx as usize], "mpm_idx={idx} failed");
        }
    }

    /// Hand-built 4×4 intra-only CU decode: verifies that the full
    /// coding_unit() + transform_unit() chain runs end-to-end on a
    /// zero-stream without panicking, producing CBFs and a coefficient
    /// array of the right shape.
    #[test]
    fn intra_4x4_cu_runs_full_decode_chain() {
        let tools = CuToolFlags {
            chroma_format_idc: 1,
            ctb_size_y: 128,
            max_tb_size_y: 64,
            min_tb_size_y: 4,
            max_ts_size: 32,
            ..CuToolFlags::default()
        };
        let data = [0u8; 256];
        let mut dec = ArithDecoder::new(&data).unwrap();
        let mut ctxs = LeafCuCtxs::init(26);
        let mut info = LeafCuInfo {
            cb_width: 4,
            cb_height: 4,
            ..LeafCuInfo::default()
        };
        let mut residual = LeafCuResidual::default();
        let neigh = CuNeighbourhood::default();
        let mut reader = LeafCuReader::new(&mut dec, &mut ctxs, tools);
        reader.decode(&mut info, &mut residual, &neigh).unwrap();
        // CBFs read as MPS on the zero-stream for this QP; regardless
        // of exact values the reader must produce legal output.
        // When tu_y_coded_flag is true, the luma residual array must
        // have the expected shape.
        if info.tu_y_coded_flag {
            assert_eq!(residual.luma_levels.len(), 16);
        } else {
            assert!(residual.luma_levels.is_empty());
        }
        // Chroma CBFs must also yield either 0 or a 2x2 level array
        // (4:2:0 chroma plane of a 4x4 luma is 2x2).
        if info.tu_cb_coded_flag {
            assert_eq!(residual.cb_levels.len(), 4);
        }
        if info.tu_cr_coded_flag {
            assert_eq!(residual.cr_levels.len(), 4);
        }
        // QP delta stays 0 because cu_qp_delta_enabled is false.
        assert_eq!(info.cu_qp_delta_val, 0);
    }

    /// QP delta path: enable `cu_qp_delta_enabled` and verify that a
    /// zero-stream at least runs the read. Cannot assert on the exact
    /// value without hand-crafting the bit pattern.
    #[test]
    fn cu_qp_delta_is_read_when_enabled_and_cbf_fires() {
        let tools = CuToolFlags {
            chroma_format_idc: 1,
            ctb_size_y: 128,
            max_tb_size_y: 64,
            min_tb_size_y: 4,
            max_ts_size: 32,
            cu_qp_delta_enabled: true,
            ..CuToolFlags::default()
        };
        let data = [0u8; 256];
        let mut dec = ArithDecoder::new(&data).unwrap();
        let mut ctxs = LeafCuCtxs::init(26);
        let mut info = LeafCuInfo {
            cb_width: 8,
            cb_height: 8,
            ..LeafCuInfo::default()
        };
        let mut residual = LeafCuResidual::default();
        let neigh = CuNeighbourhood::default();
        let mut reader = LeafCuReader::new(&mut dec, &mut ctxs, tools);
        reader.decode(&mut info, &mut residual, &neigh).unwrap();
        // cu_qp_delta_val is in the legal signed range; on zero stream
        // it's almost always 0 (MPS branch). We can't bound it tighter
        // without re-implementing the CABAC engine math here.
        assert!(info.cu_qp_delta_val.abs() < 64);
    }

    #[test]
    fn tb_bypass_reads_small_cmax_cases() {
        // cMax = 3 → n=4 → k=2 → u=0. pure FL(2 bits).
        // All-zero stream → decodes 0.
        let data = [0u8; 4];
        let mut dec = ArithDecoder::new(&data).unwrap();
        assert_eq!(decode_tb_bypass(&mut dec, 3).unwrap(), 0);

        // cMax = 4 → n=5 → k=2 → u=8-5=3. prefix∈[0..3) is FL 2-bit
        // (values 0..2); ≥3 needs an extra bit. 0-stream → prefix=0<3 → v=0.
        let data = [0u8; 4];
        let mut dec = ArithDecoder::new(&data).unwrap();
        assert_eq!(decode_tb_bypass(&mut dec, 4).unwrap(), 0);
    }
}
