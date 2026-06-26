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
/// **Units note.** Unlike the AMVR `rightShift == leftShift` rounding
/// used elsewhere in §8.5.2, the SbTMVP rounding has `leftShift = 0`,
/// so the eq. 609 / 610 `<< leftShift` term does **not** re-expand the
/// shifted value. The returned `tempMv` is therefore an **integer-luma-
/// sample** offset (1/16-pel precision discarded), which matches the
/// way §8.5.5.3 eqs. 722 – 724 add it directly to integer pixel
/// coordinates (`xSb + tempMv[0]`).
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

    // §8.5.2.14 rounding with rightShift = 4, leftShift = 0 (eqs. 608 –
    // 610).
    MotionVector {
        x: round_mv_component(temp_mv.x, 4, 0),
        y: round_mv_component(temp_mv.y, 4, 0),
    }
}

/// §8.5.2.14 eqs. 608 – 610 — signed-magnitude motion-vector rounding
/// with explicit `rightShift` / `leftShift`. Distinct from the
/// `amvp::round_mv_amvr` helper, which pins `rightShift == leftShift`;
/// the §8.5.5.4 SbTMVP path needs `leftShift = 0`.
fn round_mv_component(v: i32, right_shift: u32, left_shift: u32) -> i32 {
    // eq. 608.
    let offset = if right_shift == 0 {
        0
    } else {
        (1i32 << (right_shift - 1)) - 1
    };
    // eqs. 609 / 610: Sign(v) * (((Abs(v) + offset) >> rightShift) << leftShift).
    let sign = v.signum();
    sign * (((v.abs() + offset) >> right_shift) << left_shift)
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

// ============================================================ §8.5.5.3
// + §8.5.2.12 — CTU-walker fuse: per-sub-block motion fill.
// =====================================================================
//
// Round 135 lands the §8.5.5.3 main-body loop that the round-132 record
// reserved: for each 8×8 sub-block of the CU, derive the collocated
// location `(xColSb, yColSb)` (centre + tempMv, clipped per eqs.
// 722 – 724), snap it to the 8×8 grid `(xColCb, yColCb)`, read the
// collocated picture's per-4×4 `CuPredMode` + `MvLX` / `predFlagLX`
// motion field at that grid cell, run the §8.5.2.12 collocated-MV
// derivation (with `sbFlag = 1`), POC-scale per eqs. 598 – 605, and fill
// the per-sub-block `mvLXSbCol` / `predFlagLXSbCol` arrays. When a
// sub-block's collocated block is intra/unavailable (both list reads
// report `predFlagLXSbCol == 0`), eqs. 725 / 726 substitute the
// CU-centre default motion `ctrMvLX` / `ctrPredFlagLX` carried on the
// record.

/// The §8.5.2.12 collocated-block motion record at one 8×8 grid cell of
/// `ColPic`. Mirrors the spec arrays `CuPredMode[0]`, `PredFlagLX`,
/// `MvDmvrLX`, `RefIdxLX` of the collocated picture, sampled at the
/// `(xColCb, yColCb)` location §8.5.5.3 snaps the per-sub-block
/// collocated coordinate to.
///
/// `mode_inter == false` models the §8.5.2.12 first bullet —
/// "`colCb` is coded in an intra, IBC, or palette prediction mode" —
/// which forces `availableFlagLXCol = 0` for both lists (and therefore
/// the §8.5.5.3 eqs. 725 / 726 centre fallback).
#[derive(Clone, Copy, Debug, Default)]
pub struct ColBlockMotion {
    /// `CuPredMode[0][xColCb][yColCb] == MODE_INTER` of the collocated
    /// picture. `false` ⇒ intra / IBC / palette ⇒ no collocated MV.
    pub mode_inter: bool,
    /// `predFlagColL0[xColCb][yColCb]`.
    pub pred_flag_l0: bool,
    /// `predFlagColL1[xColCb][yColCb]`.
    pub pred_flag_l1: bool,
    /// `mvL0Col[xColCb][yColCb]` (1/16-pel; pre-§8.5.2.15 compression).
    pub mv_l0: MotionVector,
    /// `mvL1Col[xColCb][yColCb]` (1/16-pel; pre-§8.5.2.15 compression).
    pub mv_l1: MotionVector,
    /// `refIdxL0Col[xColCb][yColCb]` — index into the collocated
    /// slice's RPL[0]. `-1` ⇒ no L0 reference.
    pub ref_idx_l0: i32,
    /// `refIdxL1Col[xColCb][yColCb]` — index into the collocated
    /// slice's RPL[1]. `-1` ⇒ no L1 reference.
    pub ref_idx_l1: i32,
}

/// A read-back of the collocated picture's per-4×4 motion field at an
/// 8×8-snapped `(xColCb, yColCb)` luma location. Returns the default
/// (`mode_inter == false` ⇒ unavailable) when the coordinate is outside
/// the collocated picture or no CU wrote there.
pub type ColMotionSampler<'a> = &'a dyn Fn(i32, i32) -> ColBlockMotion;

/// Per-list POC operands + `NoBackwardPredFlag` for the §8.5.2.12
/// scaling. The same operands apply to every sub-block of one CU (the
/// current picture POC and the per-list current-RPL[0] reference POC are
/// fixed at `refIdxLXSbCol = 0`), so they live on the CU-scoped input
/// bundle rather than being recomputed per sub-block.
pub struct SbTmvpFuseInputs<'a> {
    /// CU origin `(xCb, yCb)` — picture-absolute luma top-left.
    pub xcb: i32,
    /// CU origin Y.
    pub ycb: i32,
    /// `CtbLog2SizeY` from the active SPS (§7.4.3.4 eq. 31).
    pub ctb_log2_size_y: u32,
    /// The §8.5.5.3 eqs. 722 – 724 clip boundary selector.
    pub boundary: PictureBoundary,
    /// `sh_slice_type` of the current slice. The L1 sub-block read +
    /// the `NoBackwardPredFlag` LY fallback only fire on `SliceType::B`.
    pub slice_type: SliceType,
    /// `NoBackwardPredFlag` (§8.5.2.1) — `1` when every active reference
    /// has POC ≤ the current picture POC. Drives the §8.5.2.12 sbFlag=1
    /// `predFlagColLX == 0` cross-list (LY) fallback.
    pub no_backward_pred: bool,
    /// `PicOrderCnt( ColPic )` — POC of the collocated picture.
    pub col_pic_poc: i32,
    /// `PicOrderCnt( currPic )` — current picture POC.
    pub curr_pic_poc: i32,
    /// `DiffPicOrderCnt( currPic, RefPicList[0][0] )` operand:
    /// `PicOrderCnt( RefPicList[0][refIdxL0SbCol=0] )`. Used for
    /// `currPocDiff` (eq. 599) on the L0 sub-block read.
    pub curr_ref_poc_l0: i32,
    /// Same for L1 (`RefPicList[1][refIdxL1SbCol=0]`). Only consulted on
    /// B-slices.
    pub curr_ref_poc_l1: i32,
    /// Resolve the collocated block's `refIdxCol` (on the collocated
    /// slice's RPL `listCol`) into the referenced picture's POC. Used
    /// for `colPocDiff` (eq. 598). `listCol` is passed as the first
    /// argument (0 or 1), `refIdxCol` as the second. Returns `None`
    /// when the reference cannot be resolved (treated as
    /// `availableFlagLXCol = 0`).
    pub poc_of_col_ref: &'a dyn Fn(i32, i32) -> Option<i32>,
}

/// Per-sub-block SbCol motion — the §8.5.5.3 outputs
/// `mvLXSbCol[xSbIdx][ySbIdx]` / `predFlagLXSbCol[xSbIdx][ySbIdx]` plus
/// the fixed `refIdxLXSbCol = 0` (eq. 719).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SbColMotion {
    /// `mvL0SbCol[xSbIdx][ySbIdx]`.
    pub mv_l0: MotionVector,
    /// `predFlagL0SbCol[xSbIdx][ySbIdx]`.
    pub pred_flag_l0: bool,
    /// `mvL1SbCol[xSbIdx][ySbIdx]` (B-slice only).
    pub mv_l1: MotionVector,
    /// `predFlagL1SbCol[xSbIdx][ySbIdx]` (B-slice only).
    pub pred_flag_l1: bool,
    /// `refIdxL0SbCol = 0` per eq. 719 (only meaningful when
    /// `pred_flag_l0`).
    pub ref_idx_l0: i32,
    /// `refIdxL1SbCol = 0` per eq. 719 (only meaningful when
    /// `pred_flag_l1`).
    pub ref_idx_l1: i32,
}

/// The CU's filled SbCol sub-block grid, indexed `[ySbIdx * numSbX +
/// xSbIdx]` (row-major). The geometry (`num_sb_x`, `num_sb_y`) is copied
/// from the record's grid so callers can iterate without re-deriving it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SbColGrid {
    /// `numSbX` (eq. 715).
    pub num_sb_x: i32,
    /// `numSbY` (eq. 716).
    pub num_sb_y: i32,
    /// Row-major per-sub-block motion; `len() == num_sb_x * num_sb_y`.
    pub sub_blocks: Vec<SbColMotion>,
}

impl SbColGrid {
    /// Fetch the sub-block at `(xSbIdx, ySbIdx)` — panics on an
    /// out-of-range index (callers iterate within the derived grid).
    pub fn at(&self, xs_idx: i32, ys_idx: i32) -> SbColMotion {
        let idx = (ys_idx * self.num_sb_x + xs_idx) as usize;
        self.sub_blocks[idx]
    }
}

/// §8.5.2.12 with `sbFlag = 1` — derive one list's (`X`) collocated MV
/// at the 8×8-snapped `(xColCb, yColCb)` cell already sampled into
/// `col`. Returns `(mvLXCol, availableFlagLXCol)`.
///
/// `x` selects the list this invocation is for (0 or 1).
/// `curr_ref_poc` is `PicOrderCnt( RefPicList[X][refIdxLX=0] )` — the
/// `currPocDiff` operand (eq. 599). The §8.5.2.15 buffer compression is
/// applied to `mvCol` before scaling, matching the round-25 temporal
/// merge convention of integer-pel rounding (`mv >> 4 << 4`) which is
/// sample-exact for the integer-pel test vectors.
/// Public wrapper over [`derive_collocated_mv_subblock`] so the CTU
/// walker can run the §8.5.2.12 (sbFlag = 1) collocated-MV derivation
/// for the §8.5.5.4 CU-centre block read (eqs. 729 – 731) outside this
/// module. `x` selects the list (0 or 1).
pub fn derive_collocated_mv_subblock_pub(
    col: ColBlockMotion,
    x: i32,
    curr_ref_poc: i32,
    inputs: &SbTmvpFuseInputs<'_>,
) -> (MotionVector, bool) {
    derive_collocated_mv_subblock(col, x, curr_ref_poc, inputs)
}

fn derive_collocated_mv_subblock(
    col: ColBlockMotion,
    x: i32,
    curr_ref_poc: i32,
    inputs: &SbTmvpFuseInputs<'_>,
) -> (MotionVector, bool) {
    // §8.5.2.12 first bullet — intra / IBC / palette colCb.
    if !col.mode_inter {
        return (MotionVector::ZERO, false);
    }

    // §8.5.2.12, sbFlag == 1 branch — per-list selection.
    let (pred_flag_lx, mv_lx, ref_idx_lx) = if x == 0 {
        (col.pred_flag_l0, col.mv_l0, col.ref_idx_l0)
    } else {
        (col.pred_flag_l1, col.mv_l1, col.ref_idx_l1)
    };
    let (pred_flag_ly, mv_ly, ref_idx_ly) = if x == 0 {
        (col.pred_flag_l1, col.mv_l1, col.ref_idx_l1)
    } else {
        (col.pred_flag_l0, col.mv_l0, col.ref_idx_l0)
    };

    // Pick (mvCol, refIdxCol, listCol).
    let (mv_col, ref_idx_col, list_col) = if pred_flag_lx {
        // predFlagColLX == 1 → use LX.
        (mv_lx, ref_idx_lx, x)
    } else if inputs.no_backward_pred && pred_flag_ly {
        // predFlagColLX == 0, NoBackwardPredFlag == 1, predFlagColLY == 1
        // → use LY (Y = 1 − X).
        (mv_ly, ref_idx_ly, 1 - x)
    } else {
        // predFlagColLX == 0 and the LY fallback gate is closed →
        // availableFlagLXCol = 0.
        return (MotionVector::ZERO, false);
    };

    // §8.5.2.15 buffer compression — integer-pel rounding (matches the
    // §8.5.2.11 round-25 temporal-merge convention; sample-exact for the
    // integer-pel vectors the fuse tests exercise).
    let mv_col = MotionVector {
        x: (mv_col.x >> 4) << 4,
        y: (mv_col.y >> 4) << 4,
    };

    // §8.5.2.12 eqs. 598 / 599 — POC distances.
    let col_ref_poc = match (inputs.poc_of_col_ref)(list_col, ref_idx_col) {
        Some(p) => p,
        // Cannot resolve the collocated reference's POC → treat as
        // availableFlagLXCol = 0.
        None => return (MotionVector::ZERO, false),
    };
    let col_poc_diff = inputs.col_pic_poc.wrapping_sub(col_ref_poc);
    let curr_poc_diff = inputs.curr_pic_poc.wrapping_sub(curr_ref_poc);

    if col_poc_diff == 0 {
        // Degenerate scaling (division by zero in eq. 601).
        return (MotionVector::ZERO, false);
    }

    // Eq. 600 short-circuit when the POC distances are equal (also the
    // long-term-reference path, which the fuse models as equal-distance).
    let scaled = if col_poc_diff == curr_poc_diff {
        MotionVector {
            x: mv_col.x.clamp(-131072, 131071),
            y: mv_col.y.clamp(-131072, 131071),
        }
    } else {
        // Eqs. 601 – 605.
        let td = col_poc_diff.clamp(-128, 127);
        let tb = curr_poc_diff.clamp(-128, 127);
        let abs_td = td.unsigned_abs() as i32;
        let tx = (16384 + (abs_td >> 1)) / td;
        let dist_scale_factor = ((tb * tx + 32) >> 6).clamp(-4096, 4095);
        let scale = |c: i32| -> i32 {
            let prod = dist_scale_factor * c;
            let bias: i32 = if prod >= 0 { 1 } else { 0 };
            ((prod + 128 - bias) >> 8).clamp(-131072, 131071)
        };
        MotionVector {
            x: scale(mv_col.x),
            y: scale(mv_col.y),
        }
    };

    (scaled, true)
}

/// §8.5.5.3 main body — the CTU-walker fuse. Iterates the record's
/// `numSbX × numSbY` 8×8 sub-block grid, deriving each sub-block's
/// collocated motion per eqs. 720 – 726.
///
/// For each `(xSbIdx, ySbIdx)`:
/// 1. Eqs. 720 / 721 — `(xSb, ySb)` below-right centre.
/// 2. Eqs. 722 – 724 — clip `(xSb + tempMv[0], ySb + tempMv[1])` to the
///    CTB-aligned bounds → `(xColSb, yColSb)`.
/// 3. Snap to the 8×8 grid — `(xColCb, yColCb) = ((xColSb >> 3) << 3,
///    (yColSb >> 3) << 3)`.
/// 4. Sample the collocated motion field at `(xColCb, yColCb)`.
/// 5. §8.5.2.12 (sbFlag = 1) for L0, and for L1 when the slice is B,
///    filling `mvLXSbCol` / `predFlagLXSbCol`.
/// 6. Eqs. 725 / 726 — when **both** list reads report
///    `predFlagLXSbCol == 0`, substitute the record's CU-centre default
///    `ctrMvLX` / `ctrPredFlagLX`.
///
/// The record must already carry the §8.5.5.4 centre-block result
/// (`ctr_mv_l{0,1}`, `ctr_pred_flag_l{0,1}`) and the derived `temp_mv`;
/// this fuse does not re-run §8.5.5.4.
pub fn fill_subblock_motion(
    record: &SbTmvpRecord,
    inputs: &SbTmvpFuseInputs<'_>,
    col_sampler: ColMotionSampler<'_>,
) -> SbColGrid {
    let grid = record.grid;
    let num_sb_x = grid.num_sb_x;
    let num_sb_y = grid.num_sb_y;
    let mut sub_blocks = Vec::with_capacity((num_sb_x.max(0) * num_sb_y.max(0)) as usize);

    let is_b = inputs.slice_type == SliceType::B;

    for ys_idx in 0..num_sb_y {
        for xs_idx in 0..num_sb_x {
            // Eqs. 720 / 721.
            let (x_sb, y_sb) = grid.subblock_centre(inputs.xcb, inputs.ycb, xs_idx, ys_idx);

            // Eqs. 722 – 724.
            let (x_col_sb, y_col_sb) = clip_col_subblock_location(
                record.centre.x_ctb,
                record.centre.y_ctb,
                inputs.ctb_log2_size_y,
                inputs.boundary,
                x_sb,
                y_sb,
                record.temp_mv,
            );

            // Snap to the 8×8 collocated grid.
            let x_col_cb = (x_col_sb >> 3) << 3;
            let y_col_cb = (y_col_sb >> 3) << 3;
            let col = col_sampler(x_col_cb, y_col_cb);

            // §8.5.2.12 (sbFlag = 1) — L0.
            let (mv_l0, avail_l0) =
                derive_collocated_mv_subblock(col, 0, inputs.curr_ref_poc_l0, inputs);
            // L1 only on B-slices.
            let (mv_l1, avail_l1) = if is_b {
                derive_collocated_mv_subblock(col, 1, inputs.curr_ref_poc_l1, inputs)
            } else {
                (MotionVector::ZERO, false)
            };

            let mut sb = SbColMotion {
                mv_l0,
                pred_flag_l0: avail_l0,
                mv_l1,
                pred_flag_l1: avail_l1,
                ref_idx_l0: if avail_l0 { SBCOL_REF_IDX } else { -1 },
                ref_idx_l1: if avail_l1 { SBCOL_REF_IDX } else { -1 },
            };

            // Eqs. 725 / 726 — centre-default fallback when both list
            // reads are unavailable.
            if !sb.pred_flag_l0 && !sb.pred_flag_l1 {
                sb.mv_l0 = record.ctr_mv_l0;
                sb.pred_flag_l0 = record.ctr_pred_flag_l0;
                sb.ref_idx_l0 = if record.ctr_pred_flag_l0 {
                    SBCOL_REF_IDX
                } else {
                    -1
                };
                sb.mv_l1 = record.ctr_mv_l1;
                sb.pred_flag_l1 = record.ctr_pred_flag_l1;
                sb.ref_idx_l1 = if record.ctr_pred_flag_l1 {
                    SBCOL_REF_IDX
                } else {
                    -1
                };
            }

            sub_blocks.push(sb);
        }
    }

    SbColGrid {
        num_sb_x,
        num_sb_y,
        sub_blocks,
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
        // mv (64, 32) in 1/16-pel; §8.5.2.14 rightShift = 4, leftShift =
        // 0 collapses to integer-luma units: (64 + 7) >> 4 = 4,
        // (32 + 7) >> 4 = 2.
        assert_eq!(derive_temp_mv(inputs), mv(4, 2));
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
        // (48, -16) in 1/16-pel ⇒ integer-luma (48 + 7) >> 4 = 3,
        // -((16 + 7) >> 4) = -1.
        assert_eq!(derive_temp_mv(inputs), mv(3, -1));
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
        // (32, 32) in 1/16-pel ⇒ integer-luma (32 + 7) >> 4 = 2.
        assert_eq!(derive_temp_mv(inputs), mv(2, 2));
    }

    #[test]
    fn temp_mv_rounds_per_8_5_2_14_with_right_shift_4() {
        // mv (5, -5) in 1/16-pel units; rightShift = 4, leftShift = 0 ⇒
        // integer-luma (5 + 7) >> 4 = 0.
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
        // mv (8, -8) — exactly 1/2-pel. rightShift = 4, leftShift = 0,
        // offset = 7 ⇒ ((8 + 7) >> 4) = 0 both components (integer-luma).
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
        // mv (12, 0) is 3/4-pel; offset = 7 ⇒ (12 + 7) >> 4 = 1
        // integer-luma sample (leftShift = 0).
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
        assert_eq!(derive_temp_mv(inputs), mv(1, 0));
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

    // ---------- §8.5.5.3 main-body fuse ------------------------------

    /// Build a record covering a 16×16 CU at `(xCb, yCb)` with a 2×2
    /// sub-block grid, a known `tempMv`, and a centre-default motion.
    fn fuse_record(xcb: i32, ycb: i32, temp_mv: MotionVector) -> SbTmvpRecord {
        SbTmvpRecord {
            col_pic_poc: 8,
            centre: SbTmvpCenterLoc::derive(xcb, ycb, 16, 16, 7),
            grid: SbTmvpGrid::derive(16, 16),
            ref_idx_l0_sb_col: SBCOL_REF_IDX,
            ref_idx_l1_sb_col: SBCOL_REF_IDX,
            temp_mv,
            // Centre default: uni-pred L0, used by the eqs. 725 / 726
            // fallback only.
            ctr_pred_flag_l0: true,
            ctr_pred_flag_l1: false,
            ctr_mv_l0: mv(64, -32),
            ctr_mv_l1: MotionVector::ZERO,
        }
    }

    fn pic_boundary() -> PictureBoundary {
        PictureBoundary::Picture {
            pic_width_luma: 1024,
            pic_height_luma: 1024,
        }
    }

    /// L0-only collocated block (MODE_INTER, L0 active) at every cell.
    fn col_inter_l0(mv_l0: MotionVector) -> ColBlockMotion {
        ColBlockMotion {
            mode_inter: true,
            pred_flag_l0: true,
            pred_flag_l1: false,
            mv_l0,
            mv_l1: MotionVector::ZERO,
            ref_idx_l0: 0,
            ref_idx_l1: -1,
        }
    }

    fn fuse_inputs<'a>(
        xcb: i32,
        ycb: i32,
        slice_type: SliceType,
        poc_of_col_ref: &'a dyn Fn(i32, i32) -> Option<i32>,
    ) -> SbTmvpFuseInputs<'a> {
        SbTmvpFuseInputs {
            xcb,
            ycb,
            ctb_log2_size_y: 7,
            boundary: pic_boundary(),
            slice_type,
            no_backward_pred: true,
            col_pic_poc: 8,
            curr_pic_poc: 12,
            // Equal POC distances ⇒ eq. 600 passthrough (no scaling):
            // colPocDiff = 8 − 4 = 4 (resolver returns 4 below), and
            // currPocDiff = 12 − 8 = 4.
            curr_ref_poc_l0: 8,
            curr_ref_poc_l1: 8,
            poc_of_col_ref,
        }
    }

    #[test]
    fn fuse_fills_each_subblock_from_collocated_field() {
        // 16×16 CU ⇒ 2×2 sub-blocks. Collocated field is uniform L0
        // MODE_INTER with integer-pel MV (32, 16) at every cell; tempMv
        // = 0. With equal POC distances the §8.5.2.12 scaling collapses
        // to passthrough.
        let resolver = |_list: i32, _idx: i32| Some(4);
        let rec = fuse_record(0, 0, MotionVector::ZERO);
        let inputs = fuse_inputs(0, 0, SliceType::P, &resolver);
        let col_mv = MotionVector::from_int_pel(2, 1); // (32, 16) in 1/16.
        let sampler = |_x: i32, _y: i32| col_inter_l0(col_mv);

        let g = fill_subblock_motion(&rec, &inputs, &sampler);
        assert_eq!(g.num_sb_x, 2);
        assert_eq!(g.num_sb_y, 2);
        assert_eq!(g.sub_blocks.len(), 4);
        for sb in &g.sub_blocks {
            assert!(sb.pred_flag_l0);
            assert!(!sb.pred_flag_l1);
            assert_eq!(sb.mv_l0, col_mv);
            assert_eq!(sb.ref_idx_l0, SBCOL_REF_IDX);
            assert_eq!(sb.ref_idx_l1, -1);
        }
    }

    #[test]
    fn fuse_intra_subblock_falls_back_to_centre_default() {
        // Every collocated cell is intra (mode_inter == false) ⇒ each
        // sub-block read reports predFlagLXSbCol == 0 for both lists ⇒
        // eqs. 725 / 726 substitute the record's ctrMvL0 / ctrPredFlagL0.
        let resolver = |_list: i32, _idx: i32| Some(4);
        let rec = fuse_record(0, 0, MotionVector::ZERO);
        let inputs = fuse_inputs(0, 0, SliceType::P, &resolver);
        let sampler = |_x: i32, _y: i32| ColBlockMotion {
            mode_inter: false,
            ..Default::default()
        };

        let g = fill_subblock_motion(&rec, &inputs, &sampler);
        assert_eq!(g.sub_blocks.len(), 4);
        for sb in &g.sub_blocks {
            // Centre default is uni-pred L0 (64, -32).
            assert!(sb.pred_flag_l0);
            assert!(!sb.pred_flag_l1);
            assert_eq!(sb.mv_l0, mv(64, -32));
            assert_eq!(sb.ref_idx_l0, SBCOL_REF_IDX);
            assert_eq!(sb.ref_idx_l1, -1);
        }
    }

    #[test]
    fn fuse_mixed_field_fills_inter_and_falls_back_intra() {
        // Sub-block (0, 0) collocated cell is inter; the other three are
        // intra. tempMv = 0, CU at (0, 0), 2×2 grid. Sub-block (0, 0)
        // centre = (4, 4) ⇒ xColCb/yColCb snap to (0, 0). Mark cell (0,
        // 0) inter and the rest intra.
        let resolver = |_list: i32, _idx: i32| Some(4);
        let rec = fuse_record(0, 0, MotionVector::ZERO);
        let inputs = fuse_inputs(0, 0, SliceType::P, &resolver);
        let inter_mv = MotionVector::from_int_pel(1, -1);
        let sampler = |x: i32, y: i32| {
            if x == 0 && y == 0 {
                col_inter_l0(inter_mv)
            } else {
                ColBlockMotion {
                    mode_inter: false,
                    ..Default::default()
                }
            }
        };

        let g = fill_subblock_motion(&rec, &inputs, &sampler);
        // (0, 0): inter read.
        let sb00 = g.at(0, 0);
        assert!(sb00.pred_flag_l0);
        assert_eq!(sb00.mv_l0, inter_mv);
        // (1, 0), (0, 1), (1, 1): intra ⇒ centre fallback.
        for (xi, yi) in [(1, 0), (0, 1), (1, 1)] {
            let sb = g.at(xi, yi);
            assert!(sb.pred_flag_l0);
            assert_eq!(sb.mv_l0, mv(64, -32));
        }
    }

    #[test]
    fn fuse_b_slice_fills_both_lists() {
        // Bi-pred collocated cell on a B-slice ⇒ both lists fill.
        let resolver = |_list: i32, _idx: i32| Some(4);
        let mut rec = fuse_record(0, 0, MotionVector::ZERO);
        rec.ctr_pred_flag_l1 = true;
        rec.ctr_mv_l1 = mv(16, 16);
        let inputs = fuse_inputs(0, 0, SliceType::B, &resolver);
        let mv0 = MotionVector::from_int_pel(2, 0);
        let mv1 = MotionVector::from_int_pel(-2, 0);
        let sampler = |_x: i32, _y: i32| ColBlockMotion {
            mode_inter: true,
            pred_flag_l0: true,
            pred_flag_l1: true,
            mv_l0: mv0,
            mv_l1: mv1,
            ref_idx_l0: 0,
            ref_idx_l1: 0,
        };

        let g = fill_subblock_motion(&rec, &inputs, &sampler);
        for sb in &g.sub_blocks {
            assert!(sb.pred_flag_l0);
            assert!(sb.pred_flag_l1);
            assert_eq!(sb.mv_l0, mv0);
            assert_eq!(sb.mv_l1, mv1);
            assert_eq!(sb.ref_idx_l0, SBCOL_REF_IDX);
            assert_eq!(sb.ref_idx_l1, SBCOL_REF_IDX);
        }
    }

    #[test]
    fn fuse_p_slice_never_reads_l1() {
        // On a P-slice the L1 read is skipped even if the collocated
        // cell carries L1 — predFlagL1SbCol stays false. The cell is L1-
        // only, so the L0 read is unavailable; NoBackwardPredFlag lets
        // L0 borrow L1 (LY fallback) — but only the L0 slot is filled.
        let resolver = |_list: i32, _idx: i32| Some(4);
        let rec = fuse_record(0, 0, MotionVector::ZERO);
        let inputs = fuse_inputs(0, 0, SliceType::P, &resolver);
        let mv1 = MotionVector::from_int_pel(3, 3);
        let sampler = |_x: i32, _y: i32| ColBlockMotion {
            mode_inter: true,
            pred_flag_l0: false,
            pred_flag_l1: true,
            mv_l0: MotionVector::ZERO,
            mv_l1: mv1,
            ref_idx_l0: -1,
            ref_idx_l1: 0,
        };

        let g = fill_subblock_motion(&rec, &inputs, &sampler);
        for sb in &g.sub_blocks {
            // L0 borrowed L1 via NoBackwardPredFlag.
            assert!(sb.pred_flag_l0);
            assert_eq!(sb.mv_l0, mv1);
            // P-slice never lights L1.
            assert!(!sb.pred_flag_l1);
        }
    }

    #[test]
    fn fuse_poc_scaling_applied_when_distances_differ() {
        // colPocDiff != currPocDiff ⇒ eqs. 601 – 605 scaling. ColPic POC
        // = 8, col ref POC = 4 ⇒ colPocDiff = 4. currPic POC = 12, curr
        // ref POC = 4 ⇒ currPocDiff = 8. td = 4, tb = 8, tx = (16384 +
        // 2) / 4 = 4096, distScaleFactor = (8 * 4096 + 32) >> 6 =
        // 32800 >> 6 = 512. mvCol = (16, 0) (1 int-pel). scaled.x =
        // (512 * 16 + 128 - 1) >> 8 = (8192 + 127) >> 8 = 8319 >> 8 = 32.
        let resolver = |_list: i32, _idx: i32| Some(4);
        let rec = fuse_record(0, 0, MotionVector::ZERO);
        let mut inputs = fuse_inputs(0, 0, SliceType::P, &resolver);
        inputs.curr_ref_poc_l0 = 4;
        let col_mv = MotionVector::from_int_pel(1, 0); // (16, 0).
        let sampler = |_x: i32, _y: i32| col_inter_l0(col_mv);

        let g = fill_subblock_motion(&rec, &inputs, &sampler);
        for sb in &g.sub_blocks {
            assert!(sb.pred_flag_l0);
            assert_eq!(sb.mv_l0, mv(32, 0));
        }
    }

    #[test]
    fn fuse_tempmv_shifts_collocated_sample_position() {
        // tempMv is already-rounded integer-luma (§8.5.5.4 leftShift =
        // 0): tempMv = (8, 0). CU at (0, 0), 2×2 grid. Sub-block (0, 0)
        // centre = (4, 4); + tempMv (8, 0) ⇒ (12, 4); snap ⇒ (8, 0).
        // Sub-block (1, 0) centre = (12, 4); + (8, 0) ⇒ (20, 4); snap ⇒
        // (16, 0). The sampler distinguishes columns by xColCb so we can
        // confirm the tempMv offset moved the read.
        let resolver = |_list: i32, _idx: i32| Some(4);
        let rec = fuse_record(0, 0, MotionVector { x: 8, y: 0 });
        let inputs = fuse_inputs(0, 0, SliceType::P, &resolver);
        let left_mv = MotionVector::from_int_pel(1, 0);
        let right_mv = MotionVector::from_int_pel(2, 0);
        let sampler = move |x: i32, _y: i32| {
            if x == 8 {
                col_inter_l0(left_mv)
            } else if x == 16 {
                col_inter_l0(right_mv)
            } else {
                ColBlockMotion {
                    mode_inter: false,
                    ..Default::default()
                }
            }
        };

        let g = fill_subblock_motion(&rec, &inputs, &sampler);
        // Column 0 sampled at xColCb = 8 ⇒ left_mv.
        assert_eq!(g.at(0, 0).mv_l0, left_mv);
        assert_eq!(g.at(0, 1).mv_l0, left_mv);
        // Column 1 sampled at xColCb = 16 ⇒ right_mv.
        assert_eq!(g.at(1, 0).mv_l0, right_mv);
        assert_eq!(g.at(1, 1).mv_l0, right_mv);
    }
}
