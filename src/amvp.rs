//! VVC advanced motion vector prediction (AMVP) — §8.5.2.8 / §8.5.2.9 /
//! §8.5.2.10 luma MVP candidate-list derivation.
//!
//! This is the decode-side process that consumes the round-108
//! [`crate::leaf_cu`] non-merge inter MVP-side syntax (`inter_pred_idc`
//! / `mvp_lX_flag` / `ref_idx_lX`) plus the round-103 `mvd_coding()`
//! [`crate::inter::MotionVector`] and produces the final per-list MV
//! `mvLX = mvpLX + mvdLX` (§8.5.2.1 eqs. 504 – 507, with the §7.4.11.6
//! AMVR shift folded into `mvdLX`).
//!
//! ## What lands here
//!
//! * [`derive_spatial_mvp_candidates`] — §8.5.2.10. The A-side scan
//!   (`A0 = (xCb−1, yCb+cbH)` → `A1 = (xNbA0, yNbA0−1)`) and the B-side
//!   scan (`B0 = (xCb+cbW, yCb−1)` → `B1 = (xCb+cbW−1, yCb−1)` →
//!   `B2 = (xCb−1, yCb−1)`), each picking the first effectively-
//!   available neighbour whose prediction (list X first, then list
//!   `Y = 1 − X`) points at a reference picture with the **same POC**
//!   as the current CU's `RefPicList[X][refIdxLX]` — the spec's
//!   `DiffPicOrderCnt( … ) == 0` gate (eqs. 588 – 591). Unlike
//!   §8.5.2.3 spatial *merge*, AMVP does **not** scale across POC
//!   distance — a neighbour only contributes when the POC difference
//!   is exactly 0.
//! * [`round_mv_amvr`] — §8.5.2.14 rounding (eqs. 608 – 610) with
//!   `rightShift = leftShift = AmvrShift`, applied to each available
//!   spatial / temporal candidate so the predictor lands on the same
//!   granularity grid as the AMVR-shifted MVD.
//! * [`derive_hmvp_mvp_candidates`] — §8.5.2.9 step 5. The history-based
//!   AMVP fill: walk `HmvpCandList[i − 1]` for `i = 1..Min(4,
//!   NumHmvpCand)` (oldest-first, capped at 4 — distinct from the
//!   §8.5.2.6 merge walk), and for each RPL `LY` (`Y = X` then `1 − X`)
//!   admit + AMVR-round the entry's LY MV when its `RefIdxLY` references
//!   the current CU's reference picture (`DiffPicOrderCnt == 0`).
//! * [`build_mvp_cand_list`] — §8.5.2.9 steps 3 – 6: the §8.5.2.9 step-3
//!   Col gate (Col is only consulted when *not* both A and B are
//!   available with **different** MVs), step-4 list construction
//!   (eq. 584), the step-5 HMVP fill (the `hmvp` slice produced by
//!   [`derive_hmvp_mvp_candidates`]), and the step-6 zero-MV pad to
//!   exactly 2 candidates (eqs. 585 – 587).
//! * [`select_mvp`] — §8.5.2.8 step 2 (eq. 583): `mvpLX =
//!   mvpListLX[mvp_lX_flag]`.
//! * [`derive_final_mv`] — §8.5.2.1: fold the AMVR-shifted `mvd` into
//!   the chosen predictor.
//!
//! ## Scope note (temporal MVP)
//!
//! The §8.5.2.11 temporal collocated derivation itself already exists
//! for the *merge* path ([`crate::inter::derive_temporal_merge_candidate`]).
//! Rather than duplicate that ~150-line collocated walk, this module
//! takes the AMVP temporal Col candidate as an **injected** optional
//! `(mv, available)` produced by that machinery (the AMVP and merge
//! collocated derivations share §8.5.2.11 / §8.5.2.12 byte for byte —
//! only the §8.5.2.9 step-3 *gate* deciding whether to invoke it is
//! AMVP-specific, and that gate lives here). Wiring the live §8.5.2.11
//! invocation behind that gate into the CTU walker (which needs the
//! collocated picture + `ph_temporal_mvp_enabled_flag` plumbed through
//! the non-merge inter path) is the remaining follow-up. The §8.5.2.9
//! step-5 HMVP RPL-reference-match fill now lands here in
//! [`derive_hmvp_mvp_candidates`]; the caller wires its slice's
//! per-list `refIdx → POC` resolvers through the same [`AmvpRefContext`]
//! the spatial scan uses.
//!
//! Spec reference: ITU-T H.266 | ISO/IEC 23090-3 (V4, 01/2026). The
//! implementation is spec-only; no third-party VVC decoder source was
//! consulted.

use crate::amvr::AmvrShift;
use crate::inter::{HmvpTable, MotionField, MotionVector, MvField};

/// Maximum AMVP candidate-list length (§8.5.2.9 — the list is padded to
/// exactly 2 entries).
pub const MAX_MVP_CAND: usize = 2;

/// §8.5.2.10 spatial AMVP scan result for one neighbour group (A or B).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SpatialMvpCandidate {
    /// `availableFlagLXN` (N = A or B).
    pub available: bool,
    /// `mvLXN` in 1/16-luma units (only meaningful when `available`).
    pub mv: MotionVector,
}

/// Reference-list selector for the AMVP derivation. `X` is the list the
/// current CU is predicting (0 or 1); the neighbour scan also consults
/// the opposite list `Y = 1 − X` as the §8.5.2.10 fallback.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RefList {
    L0,
    L1,
}

impl RefList {
    /// `1 − X` — the opposite list consulted as the §8.5.2.10 fallback.
    pub fn other(self) -> RefList {
        match self {
            RefList::L0 => RefList::L1,
            RefList::L1 => RefList::L0,
        }
    }
}

/// The per-CU AMVP inputs that select which reference picture the
/// predictor must point at.
///
/// `current_ref_poc` is `PicOrderCnt( RefPicList[X][refIdxLX] )` for the
/// current CU — the POC a neighbour's reference must match (i.e.
/// `DiffPicOrderCnt == 0`) to contribute. The closures resolve a
/// neighbour's per-list reference index into its reference picture's POC
/// (so the caller wires its slice's `RefPicList[*]` POC table); they
/// return `None` when the neighbour's `refIdx` is the `-1` "no
/// reference" sentinel.
pub struct AmvpRefContext<'a> {
    /// `X` — the list the current CU predicts.
    pub list: RefList,
    /// POC of `RefPicList[X][refIdxLX]` for the current CU.
    pub current_ref_poc: i32,
    /// Resolve a neighbour's L0 reference index → that reference's POC.
    pub poc_of_l0_ref: &'a dyn Fn(i32) -> Option<i32>,
    /// Resolve a neighbour's L1 reference index → that reference's POC.
    pub poc_of_l1_ref: &'a dyn Fn(i32) -> Option<i32>,
}

impl AmvpRefContext<'_> {
    /// Resolve the POC of the picture a neighbour's `(predFlag, refIdx,
    /// mv)` on list `which` points at, returning `(poc, mv)` when the
    /// neighbour predicts on that list, else `None`.
    fn neighbour_list_match(&self, nb: &MvField, which: RefList) -> Option<(i32, MotionVector)> {
        match which {
            RefList::L0 if nb.pred_flag_l0 => {
                (self.poc_of_l0_ref)(nb.ref_idx_l0).map(|poc| (poc, nb.mv_l0))
            }
            RefList::L1 if nb.pred_flag_l1 => {
                (self.poc_of_l1_ref)(nb.ref_idx_l1).map(|poc| (poc, nb.mv_l1))
            }
            _ => None,
        }
    }

    /// §8.5.2.10 per-neighbour contribution: list X first (eq. 588 /
    /// 590), then list `Y = 1 − X` (eq. 589 / 591). A neighbour
    /// contributes only when the picked reference's POC equals the
    /// current CU's `current_ref_poc` (`DiffPicOrderCnt == 0`). The
    /// spatial scan picks the *first* matching list and stops (eqs. 588 –
    /// 591 are mutually-exclusive `If … Otherwise when …` branches).
    fn neighbour_mv(&self, nb: &MvField) -> Option<MotionVector> {
        // Try list X.
        if let Some((poc, mv)) = self.neighbour_list_match(nb, self.list) {
            if poc == self.current_ref_poc {
                return Some(mv);
            }
        }
        // Then list Y = 1 − X.
        if let Some((poc, mv)) = self.neighbour_list_match(nb, self.list.other()) {
            if poc == self.current_ref_poc {
                return Some(mv);
            }
        }
        None
    }

    /// §8.5.2.9 step-5 per-HMVP-entry contributions. Unlike the spatial
    /// scan (which is a mutually-exclusive `If … Otherwise` and picks one
    /// list), step 5 loops "for each RPL LY with Y equal to X or
    /// (1 − X)" — so an entry whose **both** lists reference the current
    /// CU's reference picture contributes *twice*. Yields the matching
    /// list MVs in `X`-then-`(1 − X)` order; the caller AMVR-rounds and
    /// applies the `until numCurrMvpCand == 2` cap.
    fn hmvp_entry_mvs(&self, e: &MvField) -> impl Iterator<Item = MotionVector> {
        let mut found: [Option<MotionVector>; 2] = [None, None];
        if let Some((poc, mv)) = self.neighbour_list_match(e, self.list) {
            if poc == self.current_ref_poc {
                found[0] = Some(mv);
            }
        }
        if let Some((poc, mv)) = self.neighbour_list_match(e, self.list.other()) {
            if poc == self.current_ref_poc {
                found[1] = Some(mv);
            }
        }
        found.into_iter().flatten()
    }
}

/// §8.5.2.10 — derive the two spatial AMVP candidates `(mvLXA,
/// availableFlagLXA)` and `(mvLXB, availableFlagLXB)`.
///
/// Inputs mirror §8.5.2.10:
/// * `xcb / ycb` — top-left luma sample of the current CB.
/// * `cb_w / cb_h` — CB dimensions in luma samples.
/// * `mvf` — the per-picture [`MotionField`] (its per-block `available`
///   flag stands in for the §6.4.4 neighbour-availability derivation —
///   a block is available iff some prior CU wrote it).
/// * `ctx` — the per-list reference context (POC matching).
///
/// Returns `[A, B]`.
pub fn derive_spatial_mvp_candidates(
    xcb: i32,
    ycb: i32,
    cb_w: i32,
    cb_h: i32,
    mvf: &MotionField,
    ctx: &AmvpRefContext<'_>,
) -> [SpatialMvpCandidate; 2] {
    // ---- A-side: A0 = (xCb − 1, yCb + cbHeight), A1 = (A0 − 1 in y) --
    // §8.5.2.10 step 5 walks ( xNbAk, yNbAk ) from A0 to A1 and picks
    // the FIRST available neighbour that satisfies the POC gate.
    let a = {
        let positions = [
            (xcb - 1, ycb + cb_h),     // A0
            (xcb - 1, ycb + cb_h - 1), // A1
        ];
        scan_spatial_group(&positions, mvf, ctx)
    };

    // ---- B-side: B0 = (xCb + cbWidth, yCb − 1), B1 = (… − 1 in x),
    //              B2 = (xCb − 1, yCb − 1). ---------------------------
    let b = {
        let positions = [
            (xcb + cb_w, ycb - 1),     // B0
            (xcb + cb_w - 1, ycb - 1), // B1
            (xcb - 1, ycb - 1),        // B2
        ];
        scan_spatial_group(&positions, mvf, ctx)
    };

    [a, b]
}

/// Scan one ordered group of neighbour positions, returning the first
/// effectively-available POC-matching contribution (§8.5.2.10 steps 5).
fn scan_spatial_group(
    positions: &[(i32, i32)],
    mvf: &MotionField,
    ctx: &AmvpRefContext<'_>,
) -> SpatialMvpCandidate {
    for &(x, y) in positions {
        let nb = mvf.get_at_luma(x, y);
        // §6.4.4 availability stand-in: the block must have been
        // written by a prior CU (and be inter — intra blocks carry
        // pred_flag_l0 == pred_flag_l1 == false so they never match).
        if !nb.available {
            continue;
        }
        if let Some(mv) = ctx.neighbour_mv(&nb) {
            return SpatialMvpCandidate {
                available: true,
                mv,
            };
        }
    }
    SpatialMvpCandidate::default()
}

/// §8.5.2.9 step 5 — derive the history-based (HMVP) AMVP candidates.
///
/// This is the RPL-reference-match filter the round-111
/// [`build_mvp_cand_list`] previously consumed pre-filtered. The spec
/// walk is distinct from the §8.5.2.6 *merge* HMVP walk in three ways:
///
/// * **Index order.** §8.5.2.9 step 5 reads `HmvpCandList[i − 1]` for
///   `i = 1..Min(4, NumHmvpCand)` — i.e. `HmvpCandList[0]`,
///   `HmvpCandList[1]`, … in **oldest-first** order (index 0 = oldest),
///   the opposite of the merge path's `HmvpCandList[NumHmvpCand −
///   hMvpIdx]` newest-first walk.
/// * **Bound.** Only the first `Min(4, NumHmvpCand)` entries are
///   consulted (the merge path walks all `NumHmvpCand`).
/// * **No A1/B1 pruning.** The merge path's `sameMotion` prune is absent
///   here; the only admission test is the RPL-reference match.
///
/// For each consulted HMVP entry and for each RPL `LY` with `Y = X`
/// first then `Y = 1 − X`, the entry contributes when the reference
/// picture corresponding to that entry's `RefIdxLY` in RPL `LY` is the
/// same reference picture as the current CU's `RefPicList[X][refIdxLX]`
/// — established here, as throughout §8.5.2, via POC equality against
/// `ctx.current_ref_poc`. A contributing entry's LY motion vector is
/// AMVR-rounded (§8.5.2.14) and appended.
///
/// The walk halts as soon as `slots_remaining` admissions have been
/// produced (the caller passes `MAX_MVP_CAND − numCurrMvpCand`, the
/// step-5 `until numCurrMvpCand is equal to 2` cap). Returns the rounded
/// MVs in admission order, ready to splice into [`build_mvp_cand_list`]
/// via its `hmvp` slice.
pub fn derive_hmvp_mvp_candidates(
    table: &HmvpTable,
    ctx: &AmvpRefContext<'_>,
    amvr: AmvrShift,
    slots_remaining: usize,
) -> Vec<MotionVector> {
    let mut out = Vec::with_capacity(slots_remaining);
    if slots_remaining == 0 {
        return out;
    }
    // i = 1..Min( 4, NumHmvpCand ) → HmvpCandList[ i − 1 ] = entries[0],
    // entries[1], … (oldest-first), capped at 4 entries.
    let limit = table.entries.len().min(4);
    'outer: for entry in table.entries.iter().take(limit) {
        // For each RPL LY with Y = X first, then Y = 1 − X. Both lists
        // may contribute (the step-5 inner loop is over LY); the "until
        // numCurrMvpCand is equal to 2" cap is `slots_remaining`.
        for mv in ctx.hmvp_entry_mvs(entry) {
            out.push(round_mv_amvr(mv, amvr));
            if out.len() >= slots_remaining {
                break 'outer;
            }
        }
    }
    out
}

/// §8.5.2.14 motion-vector rounding (eqs. 608 – 610) with
/// `rightShift = leftShift = AmvrShift`. Signed-magnitude round-toward-
/// zero-then-requantise: when `AmvrShift == 0` this is the identity.
pub fn round_mv_amvr(mv: MotionVector, amvr: AmvrShift) -> MotionVector {
    let s = amvr.value();
    MotionVector {
        x: round_component(mv.x, s),
        y: round_component(mv.y, s),
    }
}

fn round_component(v: i32, shift: u32) -> i32 {
    if shift == 0 {
        return v;
    }
    // offset = (1 << (rightShift − 1)) − 1
    let offset = (1i32 << (shift - 1)) - 1;
    let sign = v.signum();
    sign * (((v.abs() + offset) >> shift) << shift)
}

/// §8.5.2.9 step-3 + step-4 + step-5 + step-6 — assemble the AMVP
/// candidate list and pad to exactly [`MAX_MVP_CAND`] entries.
///
/// * `spatial` — `[A, B]` from [`derive_spatial_mvp_candidates`],
///   already AMVR-rounded by the caller (§8.5.2.9 step 2).
/// * `col` — the §8.5.2.11 temporal collocated candidate already
///   AMVR-rounded (`None` ⇒ `availableFlagLXCol == 0`). The §8.5.2.9
///   step-3 gate (Col only consulted when *not* both A and B available
///   with **different** MVs) is applied here.
/// * `hmvp` — the §8.5.2.9 step-5 history candidates from
///   [`derive_hmvp_mvp_candidates`] (already RPL-reference-filtered and
///   AMVR-rounded), in admission order. Consumed until the list reaches
///   2.
///
/// Returns the list (length exactly [`MAX_MVP_CAND`]) — eqs. 584 – 587.
pub fn build_mvp_cand_list(
    spatial: [SpatialMvpCandidate; 2],
    col: Option<MotionVector>,
    hmvp: &[MotionVector],
) -> [MotionVector; MAX_MVP_CAND] {
    let [a, b] = spatial;
    let mut list = [MotionVector::ZERO; MAX_MVP_CAND];
    let mut n = 0usize;

    // §8.5.2.9 step 4 — eq. 584.
    if a.available {
        list[n] = a.mv;
        n += 1;
        if b.available && a.mv != b.mv {
            list[n] = b.mv;
            n += 1;
        }
    } else if b.available {
        list[n] = b.mv;
        n += 1;
    }

    // §8.5.2.9 step 3 gate + step-4 last line — Col is consulted only
    // when NOT (both A and B available with mvLXA != mvLXB). When both
    // are available with the same MV, or one/none available, Col may
    // contribute. The step-4 insertion guards on `numCurrMvpCand < 2`.
    let col_suppressed_by_ab = a.available && b.available && a.mv != b.mv;
    if n < MAX_MVP_CAND && !col_suppressed_by_ab {
        if let Some(mvcol) = col {
            list[n] = mvcol;
            n += 1;
        }
    }

    // §8.5.2.9 step 5 — HMVP fill (caller pre-filtered).
    for &mv in hmvp {
        if n >= MAX_MVP_CAND {
            break;
        }
        list[n] = mv;
        n += 1;
    }

    // §8.5.2.9 step 6 — zero-MV pad (eqs. 585 – 587).
    while n < MAX_MVP_CAND {
        list[n] = MotionVector::ZERO;
        n += 1;
    }

    list
}

/// §8.5.2.8 step 2 — eq. 583: `mvpLX = mvpListLX[ mvp_lX_flag ]`.
pub fn select_mvp(list: &[MotionVector; MAX_MVP_CAND], mvp_lx_flag: u32) -> MotionVector {
    let idx = (mvp_lx_flag as usize).min(MAX_MVP_CAND - 1);
    list[idx]
}

/// §8.5.2.1 — fold the AMVR-shifted MVD into the chosen predictor:
/// `mvLX = mvpLX + mvdLX`. `mvd` here is the **raw** parsed `lMvd`
/// (round-103, pre-AMVR); the AMVR shift (eqs. 161 – 176) is applied
/// here so the caller passes the raw value plus the per-CU
/// [`AmvrShift`].
///
/// Components are clipped to the §8.5.2.1 / eq. 600 range
/// `[−2^17, 2^17 − 1]` (the 18-bit MV storage range).
pub fn derive_final_mv(mvp: MotionVector, raw_mvd: MotionVector, amvr: AmvrShift) -> MotionVector {
    let s = amvr.value();
    let shifted = MotionVector {
        x: raw_mvd.x << s,
        y: raw_mvd.y << s,
    };
    MotionVector {
        x: (mvp.x + shifted.x).clamp(-131072, 131071),
        y: (mvp.y + shifted.y).clamp(-131072, 131071),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a uni-pred L0 inter MvField at a given (mv, refIdx).
    fn l0_block(mvx: i32, mvy: i32, ref_idx: i32) -> MvField {
        MvField {
            mv_l0: MotionVector { x: mvx, y: mvy },
            ref_idx_l0: ref_idx,
            pred_flag_l0: true,
            mode_inter: true,
            available: true,
            ..MvField::UNAVAILABLE
        }
    }

    /// Build a uni-pred L1 inter MvField.
    fn l1_block(mvx: i32, mvy: i32, ref_idx: i32) -> MvField {
        MvField {
            mv_l1: MotionVector { x: mvx, y: mvy },
            ref_idx_l1: ref_idx,
            pred_flag_l1: true,
            mode_inter: true,
            available: true,
            ..MvField::UNAVAILABLE
        }
    }

    /// Build a bi-pred inter MvField with independent L0 / L1 MVs + refs.
    fn bi_block(l0: (i32, i32, i32), l1: (i32, i32, i32)) -> MvField {
        MvField {
            mv_l0: MotionVector { x: l0.0, y: l0.1 },
            ref_idx_l0: l0.2,
            pred_flag_l0: true,
            mv_l1: MotionVector { x: l1.0, y: l1.1 },
            ref_idx_l1: l1.2,
            pred_flag_l1: true,
            mode_inter: true,
            available: true,
            ..MvField::UNAVAILABLE
        }
    }

    /// A trivial single-reference POC table: refIdx 0 → POC 0, refIdx 1
    /// → POC 8; anything else (incl. −1) → None.
    fn poc_table(ref_idx: i32) -> Option<i32> {
        match ref_idx {
            0 => Some(0),
            1 => Some(8),
            _ => None,
        }
    }

    fn ctx_l0(current_ref_poc: i32) -> AmvpRefContext<'static> {
        AmvpRefContext {
            list: RefList::L0,
            current_ref_poc,
            poc_of_l0_ref: &poc_table,
            poc_of_l1_ref: &poc_table,
        }
    }

    // ---- §8.5.2.10 spatial scan ------------------------------------

    #[test]
    fn spatial_a_picks_a0_first() {
        // 64x64 picture; current CB at (16, 16), 8x8.
        let mut mvf = MotionField::new(64, 64);
        // A0 = (15, 24), A1 = (15, 23). Both L0 ref 0 (POC 0). A0 wins.
        mvf.write_block(12, 24, 4, 4, l0_block(32, 0, 0)); // covers (15,24)
        mvf.write_block(12, 20, 4, 4, l0_block(48, 0, 0)); // covers (15,23)
        let ctx = ctx_l0(0);
        let [a, _b] = derive_spatial_mvp_candidates(16, 16, 8, 8, &mvf, &ctx);
        assert!(a.available);
        assert_eq!(a.mv, MotionVector { x: 32, y: 0 });
    }

    #[test]
    fn spatial_a_falls_through_a0_to_a1() {
        let mut mvf = MotionField::new(64, 64);
        // A0 = (15, 24): L0 ref 1 (POC 8) — mismatches current_ref_poc 0.
        mvf.write_block(12, 24, 4, 4, l0_block(32, 0, 1));
        // A1 = (15, 23): L0 ref 0 (POC 0) — matches → contributes.
        mvf.write_block(12, 20, 4, 4, l0_block(48, 0, 0));
        let ctx = ctx_l0(0);
        let [a, _b] = derive_spatial_mvp_candidates(16, 16, 8, 8, &mvf, &ctx);
        assert!(a.available);
        assert_eq!(a.mv, MotionVector { x: 48, y: 0 });
    }

    #[test]
    fn spatial_uses_opposite_list_fallback() {
        let mut mvf = MotionField::new(64, 64);
        // A0 predicts on L1 (ref 0 → POC 0). Current list is L0, ref POC
        // 0 → opposite-list (eq. 589) fallback matches.
        mvf.write_block(12, 24, 4, 4, l1_block(64, 16, 0));
        let ctx = ctx_l0(0);
        let [a, _b] = derive_spatial_mvp_candidates(16, 16, 8, 8, &mvf, &ctx);
        assert!(a.available);
        assert_eq!(a.mv, MotionVector { x: 64, y: 16 });
    }

    #[test]
    fn spatial_unavailable_when_poc_never_matches() {
        let mut mvf = MotionField::new(64, 64);
        // All neighbours point at refIdx 1 (POC 8); current_ref_poc is 0.
        mvf.write_block(12, 24, 4, 4, l0_block(32, 0, 1));
        mvf.write_block(12, 20, 4, 4, l0_block(48, 0, 1));
        let ctx = ctx_l0(0);
        let [a, _b] = derive_spatial_mvp_candidates(16, 16, 8, 8, &mvf, &ctx);
        assert!(!a.available);
    }

    #[test]
    fn spatial_b_scans_b0_b1_b2() {
        let mut mvf = MotionField::new(64, 64);
        // current CB (16,16) 8x8. B0 = (24, 15) out of any prior write,
        // B1 = (23, 15) ditto, B2 = (15, 15) written → B picks B2.
        mvf.write_block(12, 12, 4, 4, l0_block(16, 16, 0)); // covers (15,15)
        let ctx = ctx_l0(0);
        let [_a, b] = derive_spatial_mvp_candidates(16, 16, 8, 8, &mvf, &ctx);
        assert!(b.available);
        assert_eq!(b.mv, MotionVector { x: 16, y: 16 });
    }

    #[test]
    fn spatial_intra_neighbour_does_not_contribute() {
        let mut mvf = MotionField::new(64, 64);
        // An "intra" block: available but no pred flags set.
        let intra = MvField {
            available: true,
            mode_inter: false,
            ..MvField::UNAVAILABLE
        };
        mvf.write_block(12, 24, 4, 4, intra);
        let ctx = ctx_l0(0);
        let [a, _b] = derive_spatial_mvp_candidates(16, 16, 8, 8, &mvf, &ctx);
        assert!(!a.available);
    }

    // ---- §8.5.2.9 step-5 HMVP fill ---------------------------------

    fn hmvp_table(entries: &[MvField]) -> HmvpTable {
        let mut t = HmvpTable::new();
        for &e in entries {
            t.update_with(e);
        }
        t
    }

    #[test]
    fn hmvp_oldest_first_order() {
        // §8.5.2.9 step 5 walks HmvpCandList[i−1] for i=1.. — oldest
        // first (entries[0]), the OPPOSITE of the merge path. Push three
        // distinct L0 ref-0 (POC 0) entries; the oldest pushed must be
        // emitted first.
        let table = hmvp_table(&[
            l0_block(10, 0, 0), // oldest
            l0_block(20, 0, 0),
            l0_block(30, 0, 0), // newest
        ]);
        let ctx = ctx_l0(0);
        let out = derive_hmvp_mvp_candidates(&table, &ctx, AmvrShift(0), 2);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0], MotionVector { x: 10, y: 0 }); // oldest first
        assert_eq!(out[1], MotionVector { x: 20, y: 0 });
    }

    #[test]
    fn hmvp_rpl_reference_filter_drops_mismatch() {
        // Two entries: first ref 1 (POC 8 ≠ current 0) → dropped; second
        // ref 0 (POC 0) → admitted.
        let table = hmvp_table(&[l0_block(99, 0, 1), l0_block(7, 7, 0)]);
        let ctx = ctx_l0(0);
        let out = derive_hmvp_mvp_candidates(&table, &ctx, AmvrShift(0), 2);
        assert_eq!(out, vec![MotionVector { x: 7, y: 7 }]);
    }

    #[test]
    fn hmvp_opposite_list_match() {
        // Entry predicts only on L1 (ref 0 → POC 0). Current list L0,
        // POC 0 → the Y = 1 − X branch admits the L1 MV.
        let table = hmvp_table(&[l1_block(44, 4, 0)]);
        let ctx = ctx_l0(0);
        let out = derive_hmvp_mvp_candidates(&table, &ctx, AmvrShift(0), 2);
        assert_eq!(out, vec![MotionVector { x: 44, y: 4 }]);
    }

    #[test]
    fn hmvp_bipred_entry_contributes_twice() {
        // A bi-pred entry whose BOTH lists reference POC 0 contributes
        // its L0 MV (X branch) then its L1 MV (Y branch) — distinct from
        // the spatial scan, which picks only the first matching list.
        let table = hmvp_table(&[bi_block((11, 0, 0), (0, 22, 0))]);
        let ctx = ctx_l0(0);
        let out = derive_hmvp_mvp_candidates(&table, &ctx, AmvrShift(0), 2);
        assert_eq!(
            out,
            vec![MotionVector { x: 11, y: 0 }, MotionVector { x: 0, y: 22 }]
        );
    }

    #[test]
    fn hmvp_capped_at_four_entries() {
        // Five entries (capacity), all matching ref 0. Step 5 reads only
        // Min(4, NumHmvpCand) = 4 — the newest pushed (entries[4]) is
        // never consulted even though slots remain unbounded here.
        let table = hmvp_table(&[
            l0_block(1, 0, 0),
            l0_block(2, 0, 0),
            l0_block(3, 0, 0),
            l0_block(4, 0, 0),
            l0_block(5, 0, 0), // newest — outside the i=1..Min(4,N) window
        ]);
        let ctx = ctx_l0(0);
        // Ask for 5 slots so the cap, not slots_remaining, is the bound.
        let out = derive_hmvp_mvp_candidates(&table, &ctx, AmvrShift(0), 5);
        assert_eq!(out.len(), 4);
        assert_eq!(out[3], MotionVector { x: 4, y: 0 });
        assert!(!out.contains(&MotionVector { x: 5, y: 0 }));
    }

    #[test]
    fn hmvp_respects_slots_remaining_cap() {
        // numCurrMvpCand already at 1 → one slot left. Stop after one
        // admission even with multiple matches.
        let table = hmvp_table(&[l0_block(8, 0, 0), l0_block(9, 0, 0)]);
        let ctx = ctx_l0(0);
        let out = derive_hmvp_mvp_candidates(&table, &ctx, AmvrShift(0), 1);
        assert_eq!(out, vec![MotionVector { x: 8, y: 0 }]);
    }

    #[test]
    fn hmvp_zero_slots_emits_nothing() {
        let table = hmvp_table(&[l0_block(8, 0, 0)]);
        let ctx = ctx_l0(0);
        assert!(derive_hmvp_mvp_candidates(&table, &ctx, AmvrShift(0), 0).is_empty());
    }

    #[test]
    fn hmvp_empty_table_emits_nothing() {
        let table = HmvpTable::new();
        let ctx = ctx_l0(0);
        assert!(derive_hmvp_mvp_candidates(&table, &ctx, AmvrShift(0), 2).is_empty());
    }

    #[test]
    fn hmvp_applies_amvr_rounding() {
        // Entry MV (37, -19) at quarter-pel (shift 2) rounds to (36,-20),
        // matching round_mv_amvr.
        let table = hmvp_table(&[l0_block(37, -19, 0)]);
        let ctx = ctx_l0(0);
        let out = derive_hmvp_mvp_candidates(&table, &ctx, AmvrShift(2), 2);
        assert_eq!(out, vec![MotionVector { x: 36, y: -20 }]);
    }

    #[test]
    fn hmvp_feeds_build_mvp_cand_list() {
        // End-to-end step 4 → step 5: only A available (one spatial), so
        // build_mvp_cand_list has one HMVP slot to fill from the derived
        // candidates.
        let a = SpatialMvpCandidate {
            available: true,
            mv: MotionVector { x: 5, y: 5 },
        };
        let table = hmvp_table(&[l0_block(40, 0, 0)]);
        let ctx = ctx_l0(0);
        let hmvp = derive_hmvp_mvp_candidates(&table, &ctx, AmvrShift(0), MAX_MVP_CAND - 1);
        let list = build_mvp_cand_list([a, SpatialMvpCandidate::default()], None, &hmvp);
        assert_eq!(list[0], MotionVector { x: 5, y: 5 });
        assert_eq!(list[1], MotionVector { x: 40, y: 0 });
    }

    // ---- §8.5.2.14 rounding ----------------------------------------

    #[test]
    fn round_mv_amvr_identity_at_shift_zero() {
        let mv = MotionVector { x: 37, y: -19 };
        assert_eq!(round_mv_amvr(mv, AmvrShift(0)), mv);
    }

    #[test]
    fn round_mv_amvr_quarter_pel() {
        // shift 2: offset = (1<<1)−1 = 1. |37|+1 = 38 >> 2 = 9 << 2 = 36.
        // |19|+1 = 20 >> 2 = 5 << 2 = 20, sign negative → −20.
        let mv = MotionVector { x: 37, y: -19 };
        assert_eq!(
            round_mv_amvr(mv, AmvrShift(2)),
            MotionVector { x: 36, y: -20 }
        );
    }

    #[test]
    fn round_mv_amvr_one_luma() {
        // shift 4: offset = (1<<3)−1 = 7. |16|+7=23>>4=1<<4=16.
        // |33|+7=40>>4=2<<4=32.
        let mv = MotionVector { x: 16, y: 33 };
        assert_eq!(
            round_mv_amvr(mv, AmvrShift(4)),
            MotionVector { x: 16, y: 32 }
        );
    }

    // ---- §8.5.2.9 list construction --------------------------------

    #[test]
    fn list_both_spatial_distinct() {
        let a = SpatialMvpCandidate {
            available: true,
            mv: MotionVector { x: 16, y: 0 },
        };
        let b = SpatialMvpCandidate {
            available: true,
            mv: MotionVector { x: 0, y: 16 },
        };
        let list = build_mvp_cand_list([a, b], Some(MotionVector { x: 99, y: 99 }), &[]);
        // Both spatial distinct → Col suppressed, list = [A, B].
        assert_eq!(list[0], MotionVector { x: 16, y: 0 });
        assert_eq!(list[1], MotionVector { x: 0, y: 16 });
    }

    #[test]
    fn list_both_spatial_equal_admits_col() {
        let a = SpatialMvpCandidate {
            available: true,
            mv: MotionVector { x: 16, y: 0 },
        };
        let b = SpatialMvpCandidate {
            available: true,
            mv: MotionVector { x: 16, y: 0 },
        };
        // §8.5.2.9 step 4: A added; B suppressed (mvLXA == mvLXB) so
        // numCurrMvpCand == 1; step 3 gate not tripped (MVs equal) so Col
        // contributes.
        let list = build_mvp_cand_list([a, b], Some(MotionVector { x: 5, y: 5 }), &[]);
        assert_eq!(list[0], MotionVector { x: 16, y: 0 });
        assert_eq!(list[1], MotionVector { x: 5, y: 5 });
    }

    #[test]
    fn list_only_a_then_col() {
        let a = SpatialMvpCandidate {
            available: true,
            mv: MotionVector { x: 8, y: 8 },
        };
        let b = SpatialMvpCandidate::default();
        let list = build_mvp_cand_list([a, b], Some(MotionVector { x: -4, y: 4 }), &[]);
        assert_eq!(list[0], MotionVector { x: 8, y: 8 });
        assert_eq!(list[1], MotionVector { x: -4, y: 4 });
    }

    #[test]
    fn list_only_b_no_col_uses_hmvp() {
        let a = SpatialMvpCandidate::default();
        let b = SpatialMvpCandidate {
            available: true,
            mv: MotionVector { x: 2, y: 2 },
        };
        let list = build_mvp_cand_list([a, b], None, &[MotionVector { x: 7, y: 7 }]);
        assert_eq!(list[0], MotionVector { x: 2, y: 2 });
        assert_eq!(list[1], MotionVector { x: 7, y: 7 });
    }

    #[test]
    fn list_zero_pad_when_empty() {
        let list = build_mvp_cand_list(
            [
                SpatialMvpCandidate::default(),
                SpatialMvpCandidate::default(),
            ],
            None,
            &[],
        );
        assert_eq!(list[0], MotionVector::ZERO);
        assert_eq!(list[1], MotionVector::ZERO);
    }

    #[test]
    fn list_single_spatial_zero_pad_second() {
        let a = SpatialMvpCandidate {
            available: true,
            mv: MotionVector { x: 12, y: -12 },
        };
        let list = build_mvp_cand_list([a, SpatialMvpCandidate::default()], None, &[]);
        assert_eq!(list[0], MotionVector { x: 12, y: -12 });
        assert_eq!(list[1], MotionVector::ZERO);
    }

    #[test]
    fn list_hmvp_clipped_to_two() {
        // Empty spatial + Col absent; two HMVP entries should fill but
        // never exceed 2.
        let list = build_mvp_cand_list(
            [
                SpatialMvpCandidate::default(),
                SpatialMvpCandidate::default(),
            ],
            None,
            &[
                MotionVector { x: 1, y: 1 },
                MotionVector { x: 2, y: 2 },
                MotionVector { x: 3, y: 3 },
            ],
        );
        assert_eq!(list[0], MotionVector { x: 1, y: 1 });
        assert_eq!(list[1], MotionVector { x: 2, y: 2 });
    }

    // ---- §8.5.2.8 select + §8.5.2.1 fold ---------------------------

    #[test]
    fn select_mvp_picks_by_flag() {
        let list = [MotionVector { x: 1, y: 1 }, MotionVector { x: 2, y: 2 }];
        assert_eq!(select_mvp(&list, 0), MotionVector { x: 1, y: 1 });
        assert_eq!(select_mvp(&list, 1), MotionVector { x: 2, y: 2 });
    }

    #[test]
    fn derive_final_mv_folds_mvd_with_amvr() {
        // mvp = (16, 0); raw mvd = (3, -1) at quarter-pel (shift 2) →
        // (12, -4); final = (28, -4).
        let mvp = MotionVector { x: 16, y: 0 };
        let raw = MotionVector { x: 3, y: -1 };
        let out = derive_final_mv(mvp, raw, AmvrShift(2));
        assert_eq!(out, MotionVector { x: 28, y: -4 });
    }

    #[test]
    fn derive_final_mv_clips_to_18bit_range() {
        let mvp = MotionVector {
            x: 131000,
            y: -131000,
        };
        let raw = MotionVector { x: 1000, y: -1000 };
        let out = derive_final_mv(mvp, raw, AmvrShift(4)); // shift 4 → *16
        assert_eq!(out.x, 131071);
        assert_eq!(out.y, -131072);
    }

    // ---- end-to-end §8.5.2.8 ---------------------------------------

    #[test]
    fn end_to_end_spatial_to_final_mv() {
        // Realistic flow: derive spatial A/B, round, build list, select,
        // fold the decoded mvd.
        let mut mvf = MotionField::new(64, 64);
        mvf.write_block(12, 24, 4, 4, l0_block(33, 0, 0)); // A0 at (15,24)
        mvf.write_block(12, 12, 4, 4, l0_block(0, 17, 0)); // B2 at (15,15)
        let ctx = ctx_l0(0);
        let spatial = derive_spatial_mvp_candidates(16, 16, 8, 8, &mvf, &ctx);
        // AMVR round to quarter-pel (shift 2): A 33→32, B 17→16.
        let amvr = AmvrShift(2);
        let rounded = [
            SpatialMvpCandidate {
                available: spatial[0].available,
                mv: round_mv_amvr(spatial[0].mv, amvr),
            },
            SpatialMvpCandidate {
                available: spatial[1].available,
                mv: round_mv_amvr(spatial[1].mv, amvr),
            },
        ];
        assert_eq!(rounded[0].mv, MotionVector { x: 32, y: 0 });
        assert_eq!(rounded[1].mv, MotionVector { x: 0, y: 16 });
        let list = build_mvp_cand_list(rounded, None, &[]);
        // mvp_l0_flag == 1 picks B.
        let mvp = select_mvp(&list, 1);
        assert_eq!(mvp, MotionVector { x: 0, y: 16 });
        // decoded raw mvd (2, 2) at quarter-pel → (8, 8); final (8, 24).
        let final_mv = derive_final_mv(mvp, MotionVector { x: 2, y: 2 }, amvr);
        assert_eq!(final_mv, MotionVector { x: 8, y: 24 });
    }
}
