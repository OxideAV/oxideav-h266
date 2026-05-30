//! VVC Adaptive Motion Vector Resolution (AMVR) — §7.3.11.5 +
//! §7.4.11.6 (semantics) + Table 16 (`AmvrShift`).
//!
//! AMVR lets an AMVP / affine-AMVP CU encode its MV differences at one
//! of several granularities — `1/16`-luma, `1/4`-luma, `1/2`-luma,
//! `1`-luma, `4`-luma. The two-bit selector `(amvr_flag,
//! amvr_precision_idx)` together with the `inter_affine_flag` and
//! `CuPredMode == MODE_IBC` slots produce the spec's `AmvrShift`
//! per [`Table 16`]; eqs. 161 – 176 then left-shift the parsed
//! `MvdL{0,1}` (or `MvdCpL{0,1}`) by `AmvrShift` to land the actual
//! MV difference in 1/16-luma units.
//!
//! ## Round-40 scope
//!
//! * [`AmvrShift::for_inter`] — the `inter_affine_flag == 0 &&
//!   CuPredMode != MODE_IBC` row of Table 16 (the only one our
//!   non-affine, non-IBC scaffold currently exercises).
//! * [`AmvrShift::for_affine`] — affine-AMVR (`inter_affine_flag == 1
//!   && CuPredMode != MODE_IBC`) — wired so the affine round (later)
//!   can plug straight in.
//! * [`AmvrShift::for_ibc`] — the IBC row (`CuPredMode == MODE_IBC`).
//! * [`apply_amvr_shift`] — eqs. 161 / 162 / 163 / 164 — the
//!   non-affine MVD shift.
//!
//! The leaf-CU parser surfaces `amvr_flag` / `amvr_precision_idx` via
//! the [`crate::leaf_cu`] module (along with the rest of `mvd_coding()`)
//! once the non-merge inter path lands. Round-40 wires the table look-
//! ups + the per-CU shift helper so the parser can plug in immediately.
//!
//! Spec reference: ITU-T H.266 | ISO/IEC 23090-3 (V4, 01/2026).

use crate::inter::MotionVector;

/// AMVR shift selector — the opaque per-CU integer that left-shifts
/// `Mvd*` per eqs. 161 – 176.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AmvrShift(pub u32);

impl AmvrShift {
    /// Table 16 — non-affine / non-IBC row.
    ///
    /// ```text
    /// amvr_flag  amvr_precision_idx | AmvrShift  | Resolution
    /// 0          -                  | 2          | 1/4 luma
    /// 1          0                  | 3          | 1/2 luma
    /// 1          1                  | 4          | 1 luma
    /// 1          2                  | 6          | 4 luma
    /// ```
    pub fn for_inter(amvr_flag: bool, amvr_precision_idx: u32) -> Self {
        if !amvr_flag {
            return Self(2);
        }
        match amvr_precision_idx {
            0 => Self(3),
            1 => Self(4),
            2 => Self(6),
            _ => Self(2), // spec-illegal; fall back to default 1/4-luma
        }
    }

    /// Table 16 — affine row (`inter_affine_flag == 1`).
    ///
    /// ```text
    /// amvr_flag  amvr_precision_idx | AmvrShift  | Resolution
    /// 0          -                  | 2          | 1/4 luma
    /// 1          0                  | 0          | 1/16 luma
    /// 1          1                  | 4          | 1 luma
    /// ```
    ///
    /// Note `amvr_precision_idx == 2` is reserved (the affine column of
    /// Table 16 contains "-"); we surface the default to stay total.
    pub fn for_affine(amvr_flag: bool, amvr_precision_idx: u32) -> Self {
        if !amvr_flag {
            return Self(2);
        }
        match amvr_precision_idx {
            0 => Self(0),
            1 => Self(4),
            _ => Self(2),
        }
    }

    /// Table 16 — IBC row (`CuPredMode == MODE_IBC`). When IBC is
    /// active `amvr_flag` is inferred to 1; the precision idx then
    /// picks integer / 4-luma resolution.
    ///
    /// ```text
    /// amvr_flag  amvr_precision_idx | AmvrShift  | Resolution
    /// 0          -                  | -          | (illegal — AMVR is on for IBC)
    /// 1          0                  | 4          | 1 luma
    /// 1          1                  | 6          | 4 luma
    /// ```
    pub fn for_ibc(amvr_precision_idx: u32) -> Self {
        match amvr_precision_idx {
            0 => Self(4),
            1 => Self(6),
            _ => Self(4),
        }
    }

    /// Numeric value used by eqs. 161 / 162 / 163 / 164 (non-affine)
    /// or 165 / 166 / 167 / 168 (affine — six MVDs per CP).
    pub fn value(self) -> u32 {
        self.0
    }
}

/// §7.4.11.6 eqs. 161 / 162 / 163 / 164 — apply the AmvrShift to a
/// non-affine MV difference. Each call re-quantises `mvd` from the
/// parsed `1/4`-luma (or precision-selected) magnitude into the
/// canonical `1/16`-luma [`MotionVector`] space.
///
/// Note the spec wording "left-shift" applies to the **integer**
/// magnitude of the MVD before it is added to `MvpLN[ x0 ][ y0 ][ * ]`.
/// Our [`MotionVector`] cells already carry `1/16`-luma units, so
/// callers should construct the unshifted MVD with the spec's
/// pre-shift granularity (e.g. `mvd_lN_x_minus1 + 1` for the
/// `amvr_flag == 0` quarter-pel case) and let this helper land it in
/// the canonical units.
#[inline]
pub fn apply_amvr_shift(mvd: MotionVector, amvr_shift: AmvrShift) -> MotionVector {
    let s = amvr_shift.value();
    MotionVector {
        x: mvd.x << s,
        y: mvd.y << s,
    }
}

/// §9.3.4.2 / Table 132 ctxInc helper for `amvr_flag`. The Table 132
/// row reads `inter_affine_flag[ ][ ] ? 1 : 0` (no IBC column — when
/// `CuPredMode == MODE_IBC` the §7.4.12.7 inference fires and the
/// flag is set to 1 without being parsed; see [`Self::for_ibc`] for
/// the corresponding shift-table row). Returns `0` for the regular
/// AMVR path and `1` for affine-AMVR. Callers that pass `mode_ibc =
/// true` get the inactive sentinel `2`, which is never used to index
/// the Table 89 bundle because IBC skips the bin entirely.
#[inline]
pub fn ctx_inc_amvr_flag(inter_affine_flag: bool, mode_ibc: bool) -> u32 {
    if mode_ibc {
        2
    } else if inter_affine_flag {
        1
    } else {
        0
    }
}

/// §9.3.4.2 / Table 132 ctxInc helper for `amvr_precision_idx` bin 0.
/// Spec: `( CuPredMode == MODE_IBC ) ? 1 : ( inter_affine_flag == 0 ?
/// 0 : 2 )`. The three returns map to Table 90's per-initType row of
/// three slots (regular = 0, IBC = 1, affine = 2). Bin 1 (when
/// present — only the regular AMVR path has `cMax = 2`, never IBC nor
/// affine) is also ctx-coded with the deterministic `ctxInc = 1`,
/// surfaced by [`ctx_inc_amvr_precision_idx_bin1`] for spec
/// traceability.
#[inline]
pub fn ctx_inc_amvr_precision_idx(inter_affine_flag: bool, mode_ibc: bool) -> u32 {
    if mode_ibc {
        1
    } else if inter_affine_flag {
        2
    } else {
        0
    }
}

/// §9.3.4.2 / Table 132 ctxInc helper for `amvr_precision_idx` bin 1.
/// Per the Table 132 row bin 1 is ctx-coded with the deterministic
/// constant `1` — independent of `inter_affine_flag` / IBC / neighbour
/// state. Bin 1 is only present on the regular AMVR path
/// (`cMax = 2`); the affine / IBC rows truncate at bin 0 with
/// `cMax = 1`.
#[inline]
pub fn ctx_inc_amvr_precision_idx_bin1() -> u32 {
    1
}

/// Per §7.4.11.6 `amvr_precision_idx`'s `cMax` is
/// `( inter_affine_flag == 0 && CuPredMode != MODE_IBC ) ? 2 : 1`.
/// Returns `2` for the regular AMVR path (truncated-unary covers
/// values 0 / 1 / 2 — i.e. 1/2 / 1 / 4 luma — at AmvrShift positions
/// 3 / 4 / 6), `1` for affine and IBC (truncated-unary covers values
/// 0 / 1 only — affine 1/16 vs 1 luma, IBC 1 vs 4 luma). The TR
/// binarisation per Table 132 uses `cRiceParam = 0`.
#[inline]
pub fn amvr_precision_idx_c_max(inter_affine_flag: bool, mode_ibc: bool) -> u32 {
    if !inter_affine_flag && !mode_ibc {
        2
    } else {
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn amvr_default_is_quarter_pel() {
        // amvr_flag == 0 → AmvrShift = 2 (1/4 luma) for every row.
        assert_eq!(AmvrShift::for_inter(false, 0), AmvrShift(2));
        assert_eq!(AmvrShift::for_inter(false, 1), AmvrShift(2));
        assert_eq!(AmvrShift::for_affine(false, 0), AmvrShift(2));
    }

    #[test]
    fn amvr_inter_table_16() {
        // amvr_flag = 1, precision 0 → AmvrShift = 3 (1/2 luma).
        assert_eq!(AmvrShift::for_inter(true, 0), AmvrShift(3));
        // precision 1 → AmvrShift = 4 (1 luma).
        assert_eq!(AmvrShift::for_inter(true, 1), AmvrShift(4));
        // precision 2 → AmvrShift = 6 (4 luma).
        assert_eq!(AmvrShift::for_inter(true, 2), AmvrShift(6));
    }

    #[test]
    fn amvr_affine_table_16() {
        // affine + precision 0 → AmvrShift = 0 (1/16 luma).
        assert_eq!(AmvrShift::for_affine(true, 0), AmvrShift(0));
        // affine + precision 1 → AmvrShift = 4 (1 luma).
        assert_eq!(AmvrShift::for_affine(true, 1), AmvrShift(4));
    }

    #[test]
    fn amvr_ibc_table_16() {
        // IBC + precision 0 → AmvrShift = 4 (1 luma).
        assert_eq!(AmvrShift::for_ibc(0), AmvrShift(4));
        // IBC + precision 1 → AmvrShift = 6 (4 luma).
        assert_eq!(AmvrShift::for_ibc(1), AmvrShift(6));
    }

    #[test]
    fn apply_amvr_shift_folds_into_motion_vector() {
        // Quarter-pel MVD of (3, -2) → (12, -8) after shift = 2.
        let mvd = MotionVector { x: 3, y: -2 };
        let out = apply_amvr_shift(mvd, AmvrShift(2));
        assert_eq!(out, MotionVector { x: 12, y: -8 });
        // Integer-pel MVD of (1, 1) → (16, 16) after shift = 4 (1 luma → 1/16 luma).
        let out = apply_amvr_shift(MotionVector { x: 1, y: 1 }, AmvrShift(4));
        assert_eq!(out, MotionVector { x: 16, y: 16 });
        // 4-luma MVD of (1, 0) → (64, 0) after shift = 6.
        let out = apply_amvr_shift(MotionVector { x: 1, y: 0 }, AmvrShift(6));
        assert_eq!(out, MotionVector { x: 64, y: 0 });
    }

    #[test]
    fn ctx_inc_amvr_flag_routing() {
        // Table 132 row: `inter_affine_flag ? 1 : 0`. IBC is the
        // inactive sentinel (the flag is inferred to 1 and never
        // parsed when MODE_IBC fires).
        assert_eq!(ctx_inc_amvr_flag(false, false), 0);
        assert_eq!(ctx_inc_amvr_flag(true, false), 1);
        assert_eq!(ctx_inc_amvr_flag(false, true), 2);
    }

    #[test]
    fn ctx_inc_amvr_precision_idx_bin0_routing() {
        // Table 132 row: `(MODE_IBC) ? 1 : (inter_affine_flag == 0 ?
        // 0 : 2)`. Regular = 0, IBC = 1, affine = 2.
        assert_eq!(ctx_inc_amvr_precision_idx(false, false), 0);
        assert_eq!(ctx_inc_amvr_precision_idx(false, true), 1);
        assert_eq!(ctx_inc_amvr_precision_idx(true, false), 2);
        // IBC dominates affine (MODE_IBC and inter_affine_flag = 1 is
        // illegal per the spec but the helper stays total).
        assert_eq!(ctx_inc_amvr_precision_idx(true, true), 1);
    }

    #[test]
    fn ctx_inc_amvr_precision_idx_bin1_is_one() {
        // Bin 1 is deterministic per Table 132 — never neighbour-aware.
        assert_eq!(ctx_inc_amvr_precision_idx_bin1(), 1);
    }

    #[test]
    fn amvr_precision_idx_c_max_routing() {
        // Regular AMVR — cMax = 2 (three legal values 0 / 1 / 2 →
        // 1/2 / 1 / 4 luma).
        assert_eq!(amvr_precision_idx_c_max(false, false), 2);
        // Affine-AMVR — cMax = 1 (two legal values 0 / 1 → 1/16 / 1
        // luma).
        assert_eq!(amvr_precision_idx_c_max(true, false), 1);
        // IBC-AMVR — cMax = 1 (two legal values 0 / 1 → 1 / 4 luma).
        assert_eq!(amvr_precision_idx_c_max(false, true), 1);
        // IBC + affine — illegal but the helper stays total at cMax = 1.
        assert_eq!(amvr_precision_idx_c_max(true, true), 1);
    }
}
