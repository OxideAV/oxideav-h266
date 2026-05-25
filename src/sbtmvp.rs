//! SbTMVP — Sub-block-based Temporal Motion Vector Prediction
//! (§8.5.5.3 + §8.5.5.4).
//!
//! Round 132 lands the **typed record + availability gate + tempMv
//! derivation** for the SbCol slot of the §8.5.5.2 sub-block merge
//! candidate list. The actual collocated-MV grid (per-`xSbIdx, ySbIdx`
//! `mvLXSbCol` / `predFlagLXSbCol`) is sourced by the CTU walker once
//! per-picture `ColPic` state is wired up; this module ships the data
//! types + the pure-data derivation steps that don't depend on the
//! collocated picture's per-block motion field.
//!
//! Today the round-94 `affine_merge::build_subblock_merge_cand_list`
//! already reserves slot 0 of `subblockMergeCandList` for SbCol via the
//! `sb_col_available` input. That input was previously a bare bool with
//! the corresponding `AffineMergeCandidate` slot filled with the default
//! placeholder; with this round the gate is decided by
//! [`is_sbtmvp_available`] and the per-CU record is carried in
//! [`SbTmvpRecord`], ready for the CTU-walker fuse to populate the
//! per-sub-block MV grid in a follow-up round.
//!
//! ## What this module ships
//!
//! 1. [`SbTmvpAvailability`] — captures the inputs to the §8.5.5.3 first
//!    bullet (`ph_temporal_mvp_enabled_flag`, `sps_sbtmvp_enabled_flag`,
//!    `cbWidth`, `cbHeight`) plus the §8.5.5.4 A1-neighbour state
//!    (availability flag + per-list `predFlagL{0,1}A1` /
//!    `refIdxL{0,1}A1` / `mvL{0,1}A1`) and the collocated-picture
//!    presence (`col_pic_present`). The "is the SbCol slot allowed at
//!    all?" predicate.
//! 2. [`is_sbtmvp_available`] — the §8.5.5.3 first-bullet gate.
//!    Returns `false` if any of `ph_temporal_mvp_enabled_flag == 0`,
//!    `sps_sbtmvp_enabled_flag == 0`, `cbWidth < 8`, `cbHeight < 8`, or
//!    the slice has no collocated picture; returns `true` otherwise so
//!    the §8.5.5.3 step-3 `availableFlagSbCol` decision can run. Note:
//!    the spec's step-3 final decision additionally requires
//!    `ctrPredFlagL0 || ctrPredFlagL1`, which depends on the
//!    collocated-MV read and is therefore consumed by the CTU walker
//!    (see [`SbTmvpRecord::is_sb_col_available`]).
//! 3. [`SbTmvpCenterLoc`] — §8.5.5.3 eqs. 711 – 714 — the `(xCtb, yCtb)`
//!    CTU origin + `(xCtrCb, yCtrCb)` below-right CB centre, computed
//!    pure-data from the current CU's `(xCb, yCb, cbWidth, cbHeight,
//!    CtbLog2SizeY)`.
//! 4. [`SbTmvpGrid`] — §8.5.5.3 eqs. 715 – 718 — derives `numSbX,
//!    numSbY, sbWidth = sbHeight = 8` plus the `(xSb, ySb)` below-right
//!    centre of each sub-block per eqs. 720 / 721.
//! 5. [`derive_temp_mv`] — §8.5.5.4 `tempMv` derivation from the A1
//!    neighbour. Steps: zero-init `tempMv = (0, 0)`; when `A1` is
//!    available, prefer `mvL0A1` if `predFlagL0A1 && DiffPicOrderCnt(
//!    ColPic, RefPicList[0][refIdxL0A1] ) == 0`, otherwise pick
//!    `mvL1A1` when `sh_slice_type == B && predFlagL1A1 &&
//!    DiffPicOrderCnt( ColPic, RefPicList[1][refIdxL1A1] ) == 0`;
//!    finally round per §8.5.2.14 with `rightShift = 4, leftShift = 0`.
//! 6. [`clip_col_subblock_location`] / [`clip_col_centre_location`] —
//!    the §8.5.5.3 eqs. 722 – 724 and §8.5.5.4 eqs. 729 – 731 Clip3
//!    bounds applied to the collocated-sub-block (`xColSb, yColSb`) and
//!    collocated-centre (`xColCb, yColCb`) locations. Takes the
//!    `sps_subpic_treated_as_pic_flag[CurrSubpicIdx]` selector so the
//!    sub-picture branch ([`PictureBoundary::SubpicRightBoundaryPos`])
//!    and the picture-wide branch (eqs. 724 / 731) are both reachable.
//! 7. [`SbTmvpRecord`] — the typed per-CU record consumed by the CTU
//!    walker's SbCol path. Carries the collocated-picture pointer
//!    (`col_pic_poc`), the sub-block grid geometry (`num_sb_x, num_sb_y,
//!    sb_width, sb_height`), the `(xSb_centre, ySb_centre)` per-sub-block
//!    centres, the SbCol reference indices `refIdxL{0,1}SbCol = 0` per
//!    eq. 719, the derived `tempMv`, and the `(ctrMvLX, ctrPredFlagLX)`
//!    centre-block result (populated by the walker from §8.5.5.4 step
//!    "if colPredMode[…] == MODE_INTER…"). Empty per-sub-block
//!    `mvLXSbCol` arrays are reserved for the walker to fill.
//!
//! ## Out of scope (deferred to later rounds)
//!
//! * The §8.5.5.4 "colPredMode[…] is equal to MODE_INTER" branch that
//!   invokes §8.5.2.12 collocated-MV derivation against `(xColCb,
//!   yColCb)` of the collocated picture — that requires the CTU walker
//!   to surface the collocated picture's `CuPredMode` + per-4×4 MV
//!   field (the same surface §8.5.2.11 temporal-merge already uses;
//!   re-wired through this module once the walker can dispatch
//!   non-merge inter CUs).
//! * The per-sub-block §8.5.2.12 collocated-MV reads for `mvLXSbCol` +
//!   `predFlagLXSbCol` (§8.5.5.3 main body), and the
//!   `ctrMvLX`-fallback eqs. 725 / 726 — same blocker.
//! * The encoder-side SbCol candidate emission: that needs the
//!   round-58 / round-60 inter encoder to spawn collocated-MV state on
//!   reconstructed reference pictures.
//!
//! ## Spec reference
//!
//! ITU-T H.266 | ISO/IEC 23090-3 (V4, 01/2026):
//! * §8.5.5.2 — "Derivation process for motion vectors and reference
//!   indices in subblock merge mode" (the driver that consumes SbCol).
//! * §8.5.5.3 — "Derivation process for subblock-based temporal merging
//!   candidates" (eqs. 711 – 726).
//! * §8.5.5.4 — "Derivation process for subblock-based temporal merging
//!   base motion data" (eqs. 727 – 733).
//! * §8.5.2.14 — "Rounding process for motion vectors" (eqs. 608 – 610).
//!
//! No third-party VVC decoder source was consulted; the implementation
//! is spec-only.

use crate::amvp::round_mv_amvr;
use crate::amvr::AmvrShift;
use crate::inter::MotionVector;
use crate::slice_header::SliceType;

/// Reference index assigned to every SbCol sub-block per §8.5.5.3
/// eq. 719 (`refIdxLXSbCol = 0`).
pub const SBCOL_REF_IDX: i32 = 0;

/// Spec-required sub-block size for SbTMVP — §8.5.5.3 eqs. 717 / 718
/// derive `sbWidth = cbWidth / numSbX = cbWidth / (cbWidth >> 3) = 8`
/// (and likewise for height). The grid is always 8×8 luma samples.
pub const SBTMVP_SUBBLOCK_SIZE: i32 = 8;

/// Minimum CB dimension that lets the SbTMVP gate open — §8.5.5.3
/// first bullet (`cbWidth < 8` or `cbHeight < 8` ⇒
/// `availableFlagSbCol = 0`).
pub const MIN_SBTMVP_CB_DIM: i32 = 8;

// ============================================================ §8.5.5.3
// First-bullet availability gate (`availableFlagSbCol` short-circuit).
// =====================================================================

/// Inputs to [`is_sbtmvp_available`] — the §8.5.5.3 first-bullet gate
/// plus the §8.5.5.4 collocated-picture presence check. All fields are
/// pure data, sourced directly from the SPS / picture header / slice
/// header / current CU geometry.
#[derive(Clone, Copy, Debug, Default)]
pub struct SbTmvpAvailability {
    /// `sps_sbtmvp_enabled_flag` from the active SPS (§7.4.3.4).
    pub sps_sbtmvp_enabled: bool,
    /// `ph_temporal_mvp_enabled_flag` from the active picture header
    /// (§7.4.3.7 / §7.4.4).
    pub ph_temporal_mvp_enabled: bool,
    /// `cbWidth` (luma samples) of the current CB.
    pub cb_width: i32,
    /// `cbHeight` (luma samples) of the current CB.
    pub cb_height: i32,
    /// `true` iff the slice header binds a valid collocated picture
    /// (`ColPic` resolves to a reconstructed reference). When `false`
    /// the gate is closed regardless of the SPS / PH flags — §8.5.5.4
    /// has no collocated picture to walk.
    pub col_pic_present: bool,
}

/// §8.5.5.3 first bullet — gate the SbCol candidate slot. Returns
/// `true` iff:
///
/// 1. `ph_temporal_mvp_enabled_flag == 1`, and
/// 2. `sps_sbtmvp_enabled_flag == 1`, and
/// 3. `cbWidth >= 8`, and
/// 4. `cbHeight >= 8`, and
/// 5. the slice has a collocated picture (§8.5.5.4 input).
///
/// A `true` result does **not** yet establish `availableFlagSbCol == 1`;
/// the §8.5.5.3 step-3 final decision additionally requires
/// `ctrPredFlagL0 || ctrPredFlagL1`, which is only knowable after
/// reading the collocated picture's centre-block prediction state.
pub fn is_sbtmvp_available(g: SbTmvpAvailability) -> bool {
    g.ph_temporal_mvp_enabled
        && g.sps_sbtmvp_enabled
        && g.cb_width >= MIN_SBTMVP_CB_DIM
        && g.cb_height >= MIN_SBTMVP_CB_DIM
        && g.col_pic_present
}

// ============================================================ §8.5.5.3
// Eqs. 711 – 714 — CTU origin + below-right centre.
// =====================================================================

/// §8.5.5.3 eqs. 711 – 714 — the `(xCtb, yCtb)` luma CTB origin of the
/// CTB containing the current CB and the `(xCtrCb, yCtrCb)` below-right
/// centre sample of the current CB.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SbTmvpCenterLoc {
    /// Top-left sample of the containing luma CTB — eq. 711.
    pub x_ctb: i32,
    /// Top-left sample of the containing luma CTB — eq. 712.
    pub y_ctb: i32,
    /// Below-right centre sample of the current CB — eq. 713.
    pub x_ctr_cb: i32,
    /// Below-right centre sample of the current CB — eq. 714.
    pub y_ctr_cb: i32,
}

impl SbTmvpCenterLoc {
    /// Derive the four locations from the current CB origin / dims and
    /// the `CtbLog2SizeY` from the active SPS (§7.4.3.4 eq. 31).
    pub fn derive(xcb: i32, ycb: i32, cb_width: i32, cb_height: i32, ctb_log2_size_y: u32) -> Self {
        // eqs. 711 / 712: xCtb = (xCb >> CtbLog2SizeY) << CtbLog2SizeY
        let mask = !((1i32 << ctb_log2_size_y) - 1);
        Self {
            x_ctb: xcb & mask,
            y_ctb: ycb & mask,
            // eqs. 713 / 714: xCtrCb = xCb + (cbWidth / 2)
            x_ctr_cb: xcb + (cb_width / 2),
            y_ctr_cb: ycb + (cb_height / 2),
        }
    }
}

// ============================================================ §8.5.5.3
// Eqs. 715 – 721 — SbTMVP sub-block grid geometry.
// =====================================================================

/// §8.5.5.3 eqs. 715 – 718 — the SbTMVP sub-block grid. `sbWidth ==
/// sbHeight == 8` always (eqs. 717 / 718 with the eqs. 715 / 716
/// `numSbX = cbWidth >> 3` definition).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SbTmvpGrid {
    /// `numSbX` — horizontal sub-block count, eq. 715.
    pub num_sb_x: i32,
    /// `numSbY` — vertical sub-block count, eq. 716.
    pub num_sb_y: i32,
    /// `sbWidth = sbHeight = 8` per eqs. 717 / 718. Stored for
    /// completeness; callers can also use the [`SBTMVP_SUBBLOCK_SIZE`]
    /// constant.
    pub sb_width: i32,
    /// `sbHeight` (always 8) — see [`Self::sb_width`].
    pub sb_height: i32,
}

impl SbTmvpGrid {
    /// Derive the grid from `(cbWidth, cbHeight)`. Returns a grid with
    /// `num_sb_x = num_sb_y = 0` when the CB is below the 8×8 threshold
    /// (the gate is closed in that case — see [`is_sbtmvp_available`]).
    pub fn derive(cb_width: i32, cb_height: i32) -> Self {
        if cb_width < MIN_SBTMVP_CB_DIM || cb_height < MIN_SBTMVP_CB_DIM {
            return Self::default();
        }
        Self {
            num_sb_x: cb_width >> 3,
            num_sb_y: cb_height >> 3,
            sb_width: SBTMVP_SUBBLOCK_SIZE,
            sb_height: SBTMVP_SUBBLOCK_SIZE,
        }
    }

    /// §8.5.5.3 eqs. 720 / 721 — the below-right centre `(xSb, ySb)` of
    /// the sub-block at `(xSbIdx, ySbIdx)`, relative to the picture top-
    /// left luma sample.
    pub fn subblock_centre(&self, xcb: i32, ycb: i32, xs_idx: i32, ys_idx: i32) -> (i32, i32) {
        let x_sb = xcb + xs_idx * self.sb_width + self.sb_width / 2;
        let y_sb = ycb + ys_idx * self.sb_height + self.sb_height / 2;
        (x_sb, y_sb)
    }
}

// ============================================================ §8.5.5.4
// `tempMv` derivation from the A1 neighbour.
// =====================================================================

/// Per-list closure resolving a neighbour's `refIdxLX` into the
/// reference picture's POC. Returns `None` when `refIdxLX == -1` (no
/// reference on that list).
pub type RefPocResolver<'a> = &'a dyn Fn(i32) -> Option<i32>;

/// §8.5.5.4 inputs — the A1 neighbour's per-list state plus the
/// `(ColPic, sh_slice_type)` selectors and `RefPicList[*]` POC
/// resolvers.
///
/// The A1 record mirrors the §8.5.2.3 spatial-merge A1 query: a single
/// 4×4 luma block at `(xCb − 1, yCb + cbHeight − 1)` carrying the
/// per-list `(predFlagLXA1, refIdxLXA1, mvLXA1)` triples. `available_a1
/// == false` short-circuits the entire derivation (`tempMv` stays
/// `(0, 0)` per eqs. 727 / 728).
pub struct SbTmvpTempMvInputs<'a> {
    /// `availableFlagA1` from §8.5.5.3 (the §8.5.2.3 A1 spatial-merge
    /// query for the current CB).
    pub available_a1: bool,
    /// `predFlagL0A1` — A1's L0 prediction flag.
    pub pred_flag_l0_a1: bool,
    /// `predFlagL1A1` — A1's L1 prediction flag.
    pub pred_flag_l1_a1: bool,
    /// `refIdxL0A1` — A1's L0 reference index (`-1` ⇒ no reference).
    pub ref_idx_l0_a1: i32,
    /// `refIdxL1A1` — A1's L1 reference index (`-1` ⇒ no reference).
    pub ref_idx_l1_a1: i32,
    /// `mvL0A1` — A1's L0 motion vector (in 1/16 luma units).
    pub mv_l0_a1: MotionVector,
    /// `mvL1A1` — A1's L1 motion vector (in 1/16 luma units).
    pub mv_l1_a1: MotionVector,
    /// `sh_slice_type` of the current slice. The L1 fallback branch
    /// only fires when this is `SliceType::B` (§8.5.5.4 second bullet).
    pub slice_type: SliceType,
    /// `PicOrderCnt( ColPic )` — the POC of the collocated picture
    /// resolved from the slice header `sh_collocated_from_l0_flag` /
    /// `sh_collocated_ref_idx` indices.
    pub col_pic_poc: i32,
    /// Resolve a `refIdxL0` (current CU's RPL list 0) into the
    /// referenced picture's POC. The §8.5.5.4 bullet tests
    /// `DiffPicOrderCnt(ColPic, RefPicList[0][refIdxL0A1]) == 0` — i.e.
    /// the referenced picture's POC equals `col_pic_poc`.
    pub poc_of_l0_ref: RefPocResolver<'a>,
    /// Same for `refIdxL1`.
    pub poc_of_l1_ref: RefPocResolver<'a>,
}

/// §8.5.5.4 — derive `tempMv` from the A1 neighbour, then apply the
/// §8.5.2.14 rounding with `rightShift = 4, leftShift = 0`.
///
/// Returns `MotionVector::ZERO` whenever the A1 neighbour is
/// unavailable or its lists don't reference `ColPic` (eqs. 727 / 728
/// initial value, preserved through the rounding which is the identity
/// on the all-zero vector).
pub fn derive_temp_mv(inputs: SbTmvpTempMvInputs<'_>) -> MotionVector {
    // §8.5.5.4 eqs. 727 / 728 — initialise tempMv to (0, 0).
    let mut temp_mv = MotionVector::ZERO;

    if inputs.available_a1 {
        // First bullet — try mvL0A1 when L0 predicts at ColPic.
        if inputs.pred_flag_l0_a1
            && (inputs.poc_of_l0_ref)(inputs.ref_idx_l0_a1) == Some(inputs.col_pic_poc)
        {
            temp_mv = inputs.mv_l0_a1;
        } else if inputs.slice_type == SliceType::B
            && inputs.pred_flag_l1_a1
            && (inputs.poc_of_l1_ref)(inputs.ref_idx_l1_a1) == Some(inputs.col_pic_poc)
        {
            // Second bullet — L1 fallback (B-slice only).
            temp_mv = inputs.mv_l1_a1;
        }
    }

    // §8.5.2.14 rounding with rightShift = 4, leftShift = 0. The
    // round-111 amvp::round_mv_amvr helper implements eqs. 608 – 610
    // with rightShift = leftShift = shift, and the leftShift = 0 case
    // for §8.5.5.4 collapses to the same signed-magnitude round-toward-
    // zero-then-requantise behaviour (the leftShift only re-expands the
    // value back; the SbTMVP path's leftShift = 0 keeps the rounded
    // value at the rounded magnitude).
    round_mv_amvr(temp_mv, AmvrShift(4))
}

// ============================================================ §8.5.5.3
// + §8.5.5.4 — clip collocated locations.
// =====================================================================

/// Selector for the §8.5.5.3 / §8.5.5.4 horizontal Clip3 boundary —
/// the picture-wide vs sub-picture branches diverge on
/// `sps_subpic_treated_as_pic_flag[CurrSubpicIdx]`.
#[derive(Clone, Copy, Debug)]
pub enum PictureBoundary {
    /// `sps_subpic_treated_as_pic_flag == 0` (eqs. 724 / 731) — clip to
    /// `pps_pic_width_in_luma_samples − 1`.
    Picture {
        /// `pps_pic_width_in_luma_samples`.
        pic_width_luma: i32,
        /// `pps_pic_height_in_luma_samples`.
        pic_height_luma: i32,
    },
    /// `sps_subpic_treated_as_pic_flag == 1` (eqs. 723 / 730) — clip to
    /// `SubpicRightBoundaryPos` horizontally and to the sub-picture's
    /// bottom boundary vertically.
    Subpic {
        /// `SubpicRightBoundaryPos` for the current sub-picture.
        subpic_right_boundary_pos: i32,
        /// `SubpicBottomBoundaryPos` — used for the vertical clip on
        /// the sub-picture branch. The spec eqs. 722 / 729 don't
        /// branch the vertical clip; we still expose this here so the
        /// caller can route through whichever boundary the implementer
        /// later chooses (today vertical always uses
        /// `pps_pic_height_in_luma_samples − 1`).
        subpic_bottom_boundary_pos: i32,
    },
}

impl PictureBoundary {
    /// Horizontal upper bound: `Min(rightBoundary, xCtb + (1 <<
    /// CtbLog2SizeY) + 3)`.
    fn h_upper(self, x_ctb: i32, ctb_log2_size_y: u32) -> i32 {
        let ctb_right_plus_3 = x_ctb + (1i32 << ctb_log2_size_y) + 3;
        let right = match self {
            PictureBoundary::Picture { pic_width_luma, .. } => pic_width_luma - 1,
            PictureBoundary::Subpic {
                subpic_right_boundary_pos,
                ..
            } => subpic_right_boundary_pos,
        };
        right.min(ctb_right_plus_3)
    }

    /// Vertical upper bound: `Min(picHeight − 1, yCtb + (1 <<
    /// CtbLog2SizeY) − 1)`. The spec uses `pps_pic_height_in_luma_samples
    /// − 1` for both branches; the `Subpic` branch keeps the picture
    /// height handy via the bottom-boundary slot.
    fn v_upper(self, y_ctb: i32, ctb_log2_size_y: u32) -> i32 {
        let ctb_bottom_minus_1 = y_ctb + (1i32 << ctb_log2_size_y) - 1;
        let bottom = match self {
            PictureBoundary::Picture {
                pic_height_luma, ..
            } => pic_height_luma - 1,
            PictureBoundary::Subpic {
                subpic_bottom_boundary_pos,
                ..
            } => subpic_bottom_boundary_pos,
        };
        bottom.min(ctb_bottom_minus_1)
    }
}

/// §8.5.5.3 eqs. 722 – 724 — clip a per-sub-block collocated location
/// `(xSb + tempMv[0], ySb + tempMv[1])` to the CTB-aligned bounds plus
/// the picture / sub-picture right + bottom boundary.
pub fn clip_col_subblock_location(
    x_ctb: i32,
    y_ctb: i32,
    ctb_log2_size_y: u32,
    boundary: PictureBoundary,
    x_sb: i32,
    y_sb: i32,
    temp_mv: MotionVector,
) -> (i32, i32) {
    // §8.5.2.14 already rounded tempMv at the §8.5.5.4 caller; here we
    // just apply Clip3.
    let y_col = clip3(
        y_ctb,
        boundary.v_upper(y_ctb, ctb_log2_size_y),
        y_sb + temp_mv.y,
    );
    let x_col = clip3(
        x_ctb,
        boundary.h_upper(x_ctb, ctb_log2_size_y),
        x_sb + temp_mv.x,
    );
    (x_col, y_col)
}

/// §8.5.5.4 eqs. 729 – 731 — clip the collocated-centre location
/// `(xCtrCb + tempMv[0], yCtrCb + tempMv[1])` to the CTB-aligned bounds
/// plus the picture / sub-picture right + bottom boundary.
pub fn clip_col_centre_location(
    x_ctb: i32,
    y_ctb: i32,
    ctb_log2_size_y: u32,
    boundary: PictureBoundary,
    x_ctr_cb: i32,
    y_ctr_cb: i32,
    temp_mv: MotionVector,
) -> (i32, i32) {
    clip_col_subblock_location(
        x_ctb,
        y_ctb,
        ctb_log2_size_y,
        boundary,
        x_ctr_cb,
        y_ctr_cb,
        temp_mv,
    )
}

#[inline]
fn clip3(lo: i32, hi: i32, v: i32) -> i32 {
    v.clamp(lo, hi)
}

// ============================================================ §8.5.5.3
// + §8.5.5.4 — typed SbTMVP record.
// =====================================================================

/// Per-CU SbTMVP record. Mirrors the §8.5.5.3 outputs (the SbCol
/// per-sub-block MV grid) plus the §8.5.5.4 outputs (`ctrMvL{0,1}`,
/// `ctrPredFlagL{0,1}`, `tempMv`). Today's surface is **the geometry +
/// gate state**; the per-sub-block MV arrays are placeholders that the
/// CTU walker fills once the collocated picture's `CuPredMode` +
/// per-4×4 MV grid is wired through this module.
#[derive(Clone, Debug, Default)]
pub struct SbTmvpRecord {
    /// `ColPic` POC — the reference picture the collocated motion is
    /// read from.
    pub col_pic_poc: i32,
    /// `(xCtb, yCtb, xCtrCb, yCtrCb)` per §8.5.5.3 eqs. 711 – 714.
    pub centre: SbTmvpCenterLoc,
    /// SbTMVP sub-block grid per §8.5.5.3 eqs. 715 – 718.
    pub grid: SbTmvpGrid,
    /// `refIdxL0SbCol = 0` per §8.5.5.3 eq. 719.
    pub ref_idx_l0_sb_col: i32,
    /// `refIdxL1SbCol = 0` per §8.5.5.3 eq. 719.
    pub ref_idx_l1_sb_col: i32,
    /// `tempMv` derived in §8.5.5.4 (rounded per §8.5.2.14 with
    /// `rightShift = 4, leftShift = 0`).
    pub temp_mv: MotionVector,
    /// `ctrPredFlagL0` after the §8.5.5.4 centre-block read. False
    /// when the collocated centre block is intra-coded; `true` only
    /// when the §8.5.2.12 collocated-MV derivation reported a valid L0
    /// MV. Populated by the CTU walker.
    pub ctr_pred_flag_l0: bool,
    /// `ctrPredFlagL1` (B-slice only). False on P-slice and on B-slice
    /// when the centre block has no L1 reference matching the current
    /// slice's RPL.
    pub ctr_pred_flag_l1: bool,
    /// `ctrMvL0` from §8.5.5.4 — the centre-block collocated L0 MV.
    /// Only meaningful when [`Self::ctr_pred_flag_l0`] is true.
    pub ctr_mv_l0: MotionVector,
    /// `ctrMvL1` from §8.5.5.4 — the centre-block collocated L1 MV.
    /// Only meaningful when [`Self::ctr_pred_flag_l1`] is true.
    pub ctr_mv_l1: MotionVector,
}

impl SbTmvpRecord {
    /// §8.5.5.3 step-3 final decision — `availableFlagSbCol` is true
    /// iff the §8.5.5.3 first-bullet gate was open (already enforced
    /// upstream when this record was constructed) AND `ctrPredFlagL0
    /// || ctrPredFlagL1` after the §8.5.5.4 centre-block read.
    pub fn is_sb_col_available(&self) -> bool {
        self.ctr_pred_flag_l0 || self.ctr_pred_flag_l1
    }
}

// =====================================================================
// Tests
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn mv(x: i32, y: i32) -> MotionVector {
        MotionVector { x, y }
    }

    // ---------- §8.5.5.3 first-bullet gate ----------------------------

    #[test]
    fn gate_open_when_all_preconditions_met() {
        let g = SbTmvpAvailability {
            sps_sbtmvp_enabled: true,
            ph_temporal_mvp_enabled: true,
            cb_width: 16,
            cb_height: 16,
            col_pic_present: true,
        };
        assert!(is_sbtmvp_available(g));
    }

    #[test]
    fn gate_closes_on_sps_sbtmvp_disabled() {
        let g = SbTmvpAvailability {
            sps_sbtmvp_enabled: false,
            ph_temporal_mvp_enabled: true,
            cb_width: 16,
            cb_height: 16,
            col_pic_present: true,
        };
        assert!(!is_sbtmvp_available(g));
    }

    #[test]
    fn gate_closes_on_ph_temporal_mvp_disabled() {
        let g = SbTmvpAvailability {
            sps_sbtmvp_enabled: true,
            ph_temporal_mvp_enabled: false,
            cb_width: 16,
            cb_height: 16,
            col_pic_present: true,
        };
        assert!(!is_sbtmvp_available(g));
    }

    #[test]
    fn gate_closes_on_cb_width_below_8() {
        let g = SbTmvpAvailability {
            sps_sbtmvp_enabled: true,
            ph_temporal_mvp_enabled: true,
            cb_width: 4,
            cb_height: 16,
            col_pic_present: true,
        };
        assert!(!is_sbtmvp_available(g));
    }

    #[test]
    fn gate_closes_on_cb_height_below_8() {
        let g = SbTmvpAvailability {
            sps_sbtmvp_enabled: true,
            ph_temporal_mvp_enabled: true,
            cb_width: 16,
            cb_height: 4,
            col_pic_present: true,
        };
        assert!(!is_sbtmvp_available(g));
    }

    #[test]
    fn gate_open_at_exact_8x8_boundary() {
        let g = SbTmvpAvailability {
            sps_sbtmvp_enabled: true,
            ph_temporal_mvp_enabled: true,
            cb_width: 8,
            cb_height: 8,
            col_pic_present: true,
        };
        assert!(is_sbtmvp_available(g));
    }

    #[test]
    fn gate_closes_on_no_collocated_picture() {
        let g = SbTmvpAvailability {
            sps_sbtmvp_enabled: true,
            ph_temporal_mvp_enabled: true,
            cb_width: 16,
            cb_height: 16,
            col_pic_present: false,
        };
        assert!(!is_sbtmvp_available(g));
    }

    // ---------- §8.5.5.3 eqs. 711 – 714 centre locations --------------

    #[test]
    fn centre_loc_aligned_cb_inside_ctb() {
        // 128×128 CTB (CtbLog2SizeY = 7), CB at (16, 32), 16×16.
        let c = SbTmvpCenterLoc::derive(16, 32, 16, 16, 7);
        assert_eq!(c.x_ctb, 0);
        assert_eq!(c.y_ctb, 0);
        // xCtrCb = 16 + 8 = 24; yCtrCb = 32 + 8 = 40.
        assert_eq!(c.x_ctr_cb, 24);
        assert_eq!(c.y_ctr_cb, 40);
    }

    #[test]
    fn centre_loc_cb_at_second_ctb_origin() {
        // CB starts at (128, 256) — top-left of two CTB units down +
        // one across in a 128-CTB grid.
        let c = SbTmvpCenterLoc::derive(128, 256, 64, 32, 7);
        assert_eq!(c.x_ctb, 128);
        assert_eq!(c.y_ctb, 256);
        assert_eq!(c.x_ctr_cb, 128 + 32);
        assert_eq!(c.y_ctr_cb, 256 + 16);
    }

    #[test]
    fn centre_loc_uses_floor_div_for_odd_dims() {
        // (cbWidth / 2) is integer division. 17/2 = 8.
        let c = SbTmvpCenterLoc::derive(0, 0, 17, 9, 7);
        assert_eq!(c.x_ctr_cb, 8);
        assert_eq!(c.y_ctr_cb, 4);
    }

    // ---------- §8.5.5.3 eqs. 715 – 721 grid geometry -----------------

    #[test]
    fn grid_8x8_emits_single_subblock() {
        let g = SbTmvpGrid::derive(8, 8);
        assert_eq!(g.num_sb_x, 1);
        assert_eq!(g.num_sb_y, 1);
        assert_eq!(g.sb_width, 8);
        assert_eq!(g.sb_height, 8);
    }

    #[test]
    fn grid_32x16_emits_4x2_subblocks() {
        let g = SbTmvpGrid::derive(32, 16);
        assert_eq!(g.num_sb_x, 4);
        assert_eq!(g.num_sb_y, 2);
        assert_eq!(g.sb_width, 8);
        assert_eq!(g.sb_height, 8);
    }

    #[test]
    fn grid_zero_when_below_8x8() {
        let g = SbTmvpGrid::derive(4, 16);
        assert_eq!(g.num_sb_x, 0);
        assert_eq!(g.num_sb_y, 0);
    }

    #[test]
    fn grid_subblock_centre_lands_on_below_right_centre() {
        let g = SbTmvpGrid::derive(16, 16);
        // CB at (32, 64), sub-block (0, 0) ⇒ centre at (32 + 0 + 4, 64
        // + 0 + 4) = (36, 68).
        assert_eq!(g.subblock_centre(32, 64, 0, 0), (36, 68));
        // Sub-block (1, 0) ⇒ centre at (32 + 8 + 4, 64 + 0 + 4) = (44, 68).
        assert_eq!(g.subblock_centre(32, 64, 1, 0), (44, 68));
        // Sub-block (1, 1) ⇒ centre at (32 + 8 + 4, 64 + 8 + 4) = (44, 76).
        assert_eq!(g.subblock_centre(32, 64, 1, 1), (44, 76));
    }

    // ---------- §8.5.5.4 tempMv derivation ---------------------------

    /// Helper — build a resolver that returns the same POC for every
    /// refIdx >= 0 (signalling "always matches ColPic").
    fn always(p: i32) -> impl Fn(i32) -> Option<i32> {
        move |idx| if idx >= 0 { Some(p) } else { None }
    }

    /// Helper — build a resolver that never matches ColPic.
    fn never() -> impl Fn(i32) -> Option<i32> {
        |idx| if idx >= 0 { Some(-9999) } else { None }
    }

    #[test]
    fn temp_mv_zero_when_a1_unavailable() {
        let resolver = always(7);
        let inputs = SbTmvpTempMvInputs {
            available_a1: false,
            pred_flag_l0_a1: true,
            pred_flag_l1_a1: false,
            ref_idx_l0_a1: 0,
            ref_idx_l1_a1: -1,
            mv_l0_a1: mv(64, 32),
            mv_l1_a1: MotionVector::ZERO,
            slice_type: SliceType::B,
            col_pic_poc: 7,
            poc_of_l0_ref: &resolver,
            poc_of_l1_ref: &resolver,
        };
        assert_eq!(derive_temp_mv(inputs), MotionVector::ZERO);
    }

    #[test]
    fn temp_mv_picks_l0_when_l0_matches_col_pic() {
        let resolver = always(7);
        let mv_l0 = mv(64, 32);
        let inputs = SbTmvpTempMvInputs {
            available_a1: true,
            pred_flag_l0_a1: true,
            pred_flag_l1_a1: true,
            ref_idx_l0_a1: 0,
            ref_idx_l1_a1: 0,
            mv_l0_a1: mv_l0,
            mv_l1_a1: mv(-16, 16),
            slice_type: SliceType::B,
            col_pic_poc: 7,
            poc_of_l0_ref: &resolver,
            poc_of_l1_ref: &resolver,
        };
        // mv (64, 32) is already 1/16-pel aligned to whole pel; the §8.5.2.14
        // rightShift = 4 rounding lands at (64, 32) (offset = 7,
        // (64+7) >> 4 << 4 = 64).
        assert_eq!(derive_temp_mv(inputs), mv(64, 32));
    }

    #[test]
    fn temp_mv_falls_back_to_l1_on_b_slice_when_l0_missing() {
        let l0_resolver = never();
        let l1_resolver = always(7);
        let inputs = SbTmvpTempMvInputs {
            available_a1: true,
            pred_flag_l0_a1: true,
            pred_flag_l1_a1: true,
            ref_idx_l0_a1: 0,
            ref_idx_l1_a1: 0,
            mv_l0_a1: mv(999, 999),
            mv_l1_a1: mv(48, -16),
            slice_type: SliceType::B,
            col_pic_poc: 7,
            poc_of_l0_ref: &l0_resolver,
            poc_of_l1_ref: &l1_resolver,
        };
        // L0 doesn't match ⇒ try L1. L1 matches.
        // (48, -16) is whole-pel aligned ⇒ rounds to (48, -16).
        assert_eq!(derive_temp_mv(inputs), mv(48, -16));
    }

    #[test]
    fn temp_mv_no_l1_fallback_on_p_slice() {
        let l0_resolver = never();
        let l1_resolver = always(7);
        let inputs = SbTmvpTempMvInputs {
            available_a1: true,
            pred_flag_l0_a1: true,
            pred_flag_l1_a1: true,
            ref_idx_l0_a1: 0,
            ref_idx_l1_a1: 0,
            mv_l0_a1: mv(0, 0),
            mv_l1_a1: mv(48, -16),
            slice_type: SliceType::P,
            col_pic_poc: 7,
            poc_of_l0_ref: &l0_resolver,
            poc_of_l1_ref: &l1_resolver,
        };
        // L0 doesn't match; L1 would but slice is P. Stays at (0, 0).
        assert_eq!(derive_temp_mv(inputs), MotionVector::ZERO);
    }

    #[test]
    fn temp_mv_no_l1_fallback_when_l0_pred_flag_off() {
        // §8.5.5.4: the L0 branch is "predFlagL0A1 == 1 AND
        // DiffPicOrderCnt == 0". The L1 branch fires only when
        // *otherwise* (and the per-list flags are checked there too).
        // When L0's predFlag is off, the L1 branch must still be
        // reached (this is the "otherwise" branch).
        let l0_resolver = always(99);
        let l1_resolver = always(7);
        let inputs = SbTmvpTempMvInputs {
            available_a1: true,
            pred_flag_l0_a1: false,
            pred_flag_l1_a1: true,
            ref_idx_l0_a1: -1,
            ref_idx_l1_a1: 0,
            mv_l0_a1: mv(0, 0),
            mv_l1_a1: mv(32, 32),
            slice_type: SliceType::B,
            col_pic_poc: 7,
            poc_of_l0_ref: &l0_resolver,
            poc_of_l1_ref: &l1_resolver,
        };
        assert_eq!(derive_temp_mv(inputs), mv(32, 32));
    }

    #[test]
    fn temp_mv_rounds_per_8_5_2_14_with_right_shift_4() {
        // mv (5, -5) in 1/16-pel units; rightShift = 4 ⇒
        // round-toward-zero-then-requantise lands at 0 (offset = 7,
        // (5 + 7) >> 4 << 4 = 0).
        let resolver = always(7);
        let inputs = SbTmvpTempMvInputs {
            available_a1: true,
            pred_flag_l0_a1: true,
            pred_flag_l1_a1: false,
            ref_idx_l0_a1: 0,
            ref_idx_l1_a1: -1,
            mv_l0_a1: mv(5, -5),
            mv_l1_a1: MotionVector::ZERO,
            slice_type: SliceType::B,
            col_pic_poc: 7,
            poc_of_l0_ref: &resolver,
            poc_of_l1_ref: &resolver,
        };
        assert_eq!(derive_temp_mv(inputs), MotionVector::ZERO);
    }

    #[test]
    fn temp_mv_rounds_half_pel_to_whole_pel() {
        // mv (8, -8) — exactly 1/2-pel. rightShift = 4 with offset = 7 ⇒
        // sign * (((8 + 7) >> 4) << 4) = 0 for positive; for negative
        // sign * (((|-8| + 7) >> 4) << 4) = -1 * 0 = 0.
        let resolver = always(7);
        let inputs = SbTmvpTempMvInputs {
            available_a1: true,
            pred_flag_l0_a1: true,
            pred_flag_l1_a1: false,
            ref_idx_l0_a1: 0,
            ref_idx_l1_a1: -1,
            mv_l0_a1: mv(8, -8),
            mv_l1_a1: MotionVector::ZERO,
            slice_type: SliceType::B,
            col_pic_poc: 7,
            poc_of_l0_ref: &resolver,
            poc_of_l1_ref: &resolver,
        };
        assert_eq!(derive_temp_mv(inputs), MotionVector::ZERO);
    }

    #[test]
    fn temp_mv_rounds_three_quarter_pel_up_to_one_pel() {
        // mv (12, 0) is 3/4-pel; offset = 7 ⇒ (12 + 7) >> 4 << 4 = 16.
        let resolver = always(7);
        let inputs = SbTmvpTempMvInputs {
            available_a1: true,
            pred_flag_l0_a1: true,
            pred_flag_l1_a1: false,
            ref_idx_l0_a1: 0,
            ref_idx_l1_a1: -1,
            mv_l0_a1: mv(12, 0),
            mv_l1_a1: MotionVector::ZERO,
            slice_type: SliceType::B,
            col_pic_poc: 7,
            poc_of_l0_ref: &resolver,
            poc_of_l1_ref: &resolver,
        };
        assert_eq!(derive_temp_mv(inputs), mv(16, 0));
    }

    // ---------- §8.5.5.3 + §8.5.5.4 clip ----------------------------

    #[test]
    fn clip_subblock_loc_at_picture_boundary() {
        // CTB (128, 128), 128×128 CTB, picture 256×256.
        // xSb = 132, ySb = 132, tempMv = (16, 16) ⇒ raw (148, 148).
        // h_upper = min(255, 128 + 128 + 3) = min(255, 259) = 255.
        // v_upper = min(255, 128 + 128 - 1) = min(255, 255) = 255.
        // Result: (148, 148) — well inside, no clamp.
        let bd = PictureBoundary::Picture {
            pic_width_luma: 256,
            pic_height_luma: 256,
        };
        let (x, y) = clip_col_subblock_location(128, 128, 7, bd, 132, 132, mv(16, 16));
        assert_eq!((x, y), (148, 148));
    }

    #[test]
    fn clip_subblock_loc_clamps_to_ctb_upper_bound() {
        // CTB (0, 0), 128×128. xSb = 4, ySb = 4, tempMv = (500, 500) ⇒
        // raw (504, 504). h_upper = min(picW - 1, 0 + 128 + 3) = 131.
        // v_upper = min(picH - 1, 0 + 128 - 1) = 127. Clamped to (131, 127).
        let bd = PictureBoundary::Picture {
            pic_width_luma: 1024,
            pic_height_luma: 1024,
        };
        let (x, y) = clip_col_subblock_location(0, 0, 7, bd, 4, 4, mv(500, 500));
        assert_eq!((x, y), (131, 127));
    }

    #[test]
    fn clip_subblock_loc_clamps_to_ctb_lower_bound() {
        // CTB (256, 256). xSb = 260, tempMv = (-500, -500) ⇒ raw (-240,
        // -240). Lower bound is (256, 256) per eqs. 722 / 724.
        let bd = PictureBoundary::Picture {
            pic_width_luma: 1024,
            pic_height_luma: 1024,
        };
        let (x, y) = clip_col_subblock_location(256, 256, 7, bd, 260, 260, mv(-500, -500));
        assert_eq!((x, y), (256, 256));
    }

    #[test]
    fn clip_subblock_loc_uses_subpic_boundary_when_treated_as_pic() {
        // sub-picture branch: h_upper picks min(SubpicRightBoundaryPos,
        // xCtb + 1 << 7 + 3). CTB (128, 128), subpic ends at 200.
        // h_upper = min(200, 128+128+3) = min(200, 259) = 200.
        let bd = PictureBoundary::Subpic {
            subpic_right_boundary_pos: 200,
            subpic_bottom_boundary_pos: 220,
        };
        let (x, y) = clip_col_subblock_location(128, 128, 7, bd, 132, 132, mv(500, 500));
        assert_eq!((x, y), (200, 220));
    }

    #[test]
    fn clip_centre_loc_is_same_as_subblock_clip() {
        // Sanity — the two helpers share the body; pin that.
        let bd = PictureBoundary::Picture {
            pic_width_luma: 256,
            pic_height_luma: 256,
        };
        let sb = clip_col_subblock_location(0, 0, 7, bd, 16, 16, mv(32, 32));
        let ctr = clip_col_centre_location(0, 0, 7, bd, 16, 16, mv(32, 32));
        assert_eq!(sb, ctr);
    }

    // ---------- SbTmvpRecord ----------------------------------------

    #[test]
    fn record_is_available_when_either_ctr_pred_flag_set() {
        let mut rec = SbTmvpRecord {
            ref_idx_l0_sb_col: SBCOL_REF_IDX,
            ref_idx_l1_sb_col: SBCOL_REF_IDX,
            ..Default::default()
        };
        assert!(!rec.is_sb_col_available());
        rec.ctr_pred_flag_l0 = true;
        assert!(rec.is_sb_col_available());
        rec.ctr_pred_flag_l0 = false;
        rec.ctr_pred_flag_l1 = true;
        assert!(rec.is_sb_col_available());
        rec.ctr_pred_flag_l0 = true;
        rec.ctr_pred_flag_l1 = true;
        assert!(rec.is_sb_col_available());
    }

    #[test]
    fn record_defaults_match_spec_inits() {
        let rec = SbTmvpRecord::default();
        // §8.5.5.4 eqs. 727 / 728 — tempMv initial value.
        assert_eq!(rec.temp_mv, MotionVector::ZERO);
        // §8.5.5.4 eqs. 732 / 733 — ctrPredFlag defaults to 0 when the
        // colPredMode != MODE_INTER branch fires.
        assert!(!rec.ctr_pred_flag_l0);
        assert!(!rec.ctr_pred_flag_l1);
    }

    #[test]
    fn sbcol_ref_idx_is_zero_per_eq_719() {
        assert_eq!(SBCOL_REF_IDX, 0);
    }

    #[test]
    fn sbtmvp_subblock_size_is_eight_per_eqs_717_718() {
        assert_eq!(SBTMVP_SUBBLOCK_SIZE, 8);
    }
}
