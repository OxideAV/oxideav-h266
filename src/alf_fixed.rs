//! VVC ALF fixed-filter coefficient + class-to-filter-set mapping
//! tables (§7.4.3.18, eqs. 90 / 91).
//!
//! These two tables are the "free" fixed-filter family used for
//! `AlfCtbFiltSetIdxY < 16` (eq. 1437): the per-CTB filter is picked by
//! `AlfFixFiltCoeff[ AlfClassToFiltMap[ i ][ filtIdx ] ][ j ]` where:
//!
//! * `i` = `AlfCtbFiltSetIdxY` ∈ 0..15 — selects one of 16 fixed-filter
//!   "sets" (rows of `AlfClassToFiltMap`).
//! * `filtIdx` = §8.8.5.3 classification result ∈ 0..24 — selects one of
//!   25 filter "classes" within the chosen set.
//! * The resulting row index ∈ 0..63 selects one of 64 stored 12-tap
//!   coefficient vectors in `AlfFixFiltCoeff`.
//!
//! ## Spec table layout vs. this module
//!
//! The spec PDF prints `AlfFixFiltCoeff[][]` as **12 displayed rows × 64
//! columns** (each displayed row holds the j-th coefficient across all
//! 64 row-indices i). We transpose at transcription time so the in-code
//! shape `ALF_FIX_FILT_COEFF[i][j]` matches the spec's symbolic
//! `AlfFixFiltCoeff[i][j]`.
//!
//! Similarly `AlfClassToFiltMap[][]` is printed as **25 displayed rows ×
//! 16 columns** (rows = class index, columns = set index). We transpose
//! to `ALF_CLASS_TO_FILT_MAP[set][class]` to match the spec's symbolic
//! `AlfClassToFiltMap[m][n]` indexing.
//!
//! Spec reference: ITU-T H.266 | ISO/IEC 23090-3 (V4, 01/2026), §7.4.3.18.

/// `AlfFixFiltCoeff[i][j]` — 64 fixed-filter rows of 12 signalled-tap
/// coefficients each (eq. 90).
pub const ALF_FIX_FILT_COEFF: [[i8; 12]; 64] = [
    [0, 0, 2, -3, 1, -4, 1, 7, -1, 1, -1, 5],
    [0, 0, 0, 0, 0, -1, 0, 1, 0, 0, -1, 2],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1],
    [2, 2, -7, -3, 0, -5, 13, 22, 12, -3, -3, 17],
    [-1, 0, 6, -8, 1, -5, 1, 23, 0, 2, -5, 10],
    [0, 0, -1, -1, 0, -1, 2, 1, 0, 0, -1, 4],
    [0, 0, 3, -11, 1, 0, -1, 35, 5, 2, -9, 9],
    [0, 0, 8, -8, -2, -7, 4, 4, 2, 1, -1, 25],
    [0, 0, 1, -1, 0, -3, 1, 3, -1, 1, -1, 3],
    [0, 0, 3, -3, 0, -6, 5, -1, 2, 1, -4, 21],
    [-7, 1, 5, 4, -3, 5, 11, 13, 12, -8, 11, 12],
    [-5, -3, 6, -2, -3, 8, 14, 15, 2, -7, 11, 16],
    [2, -1, -6, -5, -2, -2, 20, 14, -4, 0, -3, 25],
    [3, 1, -8, -4, 0, -8, 22, 5, -3, 2, -10, 29],
    [2, 1, -7, -1, 2, -11, 23, -5, 0, 2, -10, 29],
    [-6, -3, 8, 9, -4, 8, 9, 7, 14, -2, 8, 9],
    [2, 1, -4, -7, 0, -8, 17, 22, 1, -1, -4, 23],
    [3, 0, -5, -7, 0, -7, 15, 18, -5, 0, -5, 27],
    [2, 0, 0, -7, 1, -10, 13, 13, -4, 2, -7, 24],
    [3, 3, -13, 4, -2, -5, 9, 21, 25, -2, -3, 12],
    [-5, -2, 7, -3, -7, 9, 8, 9, 16, -2, 15, 12],
    [0, -1, 0, -7, -5, 4, 11, 11, 8, -6, 12, 21],
    [3, -2, -3, -8, -4, -1, 16, 15, -2, -3, 3, 26],
    [2, 1, -5, -4, -1, -8, 16, 4, -2, 1, -7, 33],
    [2, 1, -4, -2, 1, -10, 17, -2, 0, 2, -11, 33],
    [1, -2, 7, -15, -16, 10, 8, 8, 20, 11, 14, 11],
    [2, 2, 3, -13, -13, 4, 8, 12, 2, -3, 16, 24],
    [1, 4, 0, -7, -8, -4, 9, 9, -2, -2, 8, 29],
    [1, 1, 2, -4, -1, -6, 6, 3, -1, -1, -3, 30],
    [-7, 3, 2, 10, -2, 3, 7, 11, 19, -7, 8, 10],
    [0, -2, -5, -3, -2, 4, 20, 15, -1, -3, -1, 22],
    [3, -1, -8, -4, -1, -4, 22, 8, -4, 2, -8, 28],
    [0, 3, -14, 3, 0, 1, 19, 17, 8, -3, -7, 20],
    [0, 2, -1, -8, 3, -6, 5, 21, 1, 1, -9, 13],
    [-4, -2, 8, 20, -2, 2, 3, 5, 21, 4, 6, 1],
    [2, -2, -3, -9, -4, 2, 14, 16, 3, -6, 8, 24],
    [2, 1, 5, -16, -7, 2, 3, 11, 15, -3, 11, 22],
    [1, 2, 3, -11, -2, -5, 4, 8, 9, -3, -2, 26],
    [0, -1, 10, -9, -1, -8, 2, 3, 4, 0, 0, 29],
    [1, 2, 0, -5, 1, -9, 9, 3, 0, 1, -7, 20],
    [-2, 8, -6, -4, 3, -9, -8, 45, 14, 2, -13, 7],
    [1, -1, 16, -19, -8, -4, -3, 2, 19, 0, 4, 30],
    [1, 1, -3, 0, 2, -11, 15, -5, 1, 2, -9, 24],
    [0, 1, -2, 0, 1, -4, 4, 0, 0, 1, -4, 7],
    [0, 1, 2, -5, 1, -6, 4, 10, -2, 1, -4, 10],
    [3, 0, -3, -6, -2, -6, 14, 8, -1, -1, -3, 31],
    [0, 1, 0, -2, 1, -6, 5, 1, 0, 1, -5, 13],
    [3, 1, 9, -19, -21, 9, 7, 6, 13, 5, 15, 21],
    [2, 4, 3, -12, -13, 1, 7, 8, 3, 0, 12, 26],
    [3, 1, -8, -2, 0, -6, 18, 2, -2, 3, -10, 23],
    [1, 1, -4, -1, 1, -5, 8, 1, -1, 2, -5, 10],
    [0, 1, -1, 0, 0, -2, 2, 0, 0, 1, -2, 3],
    [1, 1, -2, -7, 1, -7, 14, 18, 0, 0, -7, 21],
    [0, 1, 0, -2, 0, -7, 8, 1, -2, 0, -3, 24],
    [0, 1, 1, -2, 2, -10, 10, 0, -2, 1, -7, 23],
    [0, 2, 2, -11, 2, -4, -3, 39, 7, 1, -10, 9],
    [1, 0, 13, -16, -5, -6, -1, 8, 6, 0, 6, 29],
    [1, 3, 1, -6, -4, -7, 9, 6, -3, -2, 3, 33],
    [4, 0, -17, -1, -1, 5, 26, 8, -2, 3, -15, 30],
    [0, 1, -2, 0, 2, -8, 12, -6, 1, 1, -6, 16],
    [0, 0, 0, -1, 1, -4, 4, 0, 0, 0, -3, 11],
    [0, 1, 2, -8, 2, -6, 5, 15, 0, 2, -7, 9],
    [1, -1, 12, -15, -7, -2, 3, 6, 6, -1, 7, 30],
];

/// `AlfClassToFiltMap[m][n]` — 16 sets × 25 classes → row index into
/// `ALF_FIX_FILT_COEFF` (eq. 91).
pub const ALF_CLASS_TO_FILT_MAP: [[u8; 25]; 16] = [
    [
        8, 2, 2, 2, 3, 4, 53, 9, 9, 52, 4, 4, 5, 9, 2, 8, 10, 9, 1, 3, 39, 39, 10, 9, 52,
    ],
    [
        11, 12, 13, 14, 15, 30, 11, 17, 18, 19, 16, 20, 20, 4, 53, 21, 22, 23, 14, 25, 26, 26, 27,
        28, 10,
    ],
    [
        16, 12, 31, 32, 14, 16, 30, 33, 53, 34, 35, 16, 20, 4, 7, 16, 21, 36, 18, 19, 21, 26, 37,
        38, 39,
    ],
    [
        35, 11, 13, 14, 43, 35, 16, 4, 34, 62, 35, 35, 30, 56, 7, 35, 21, 38, 24, 40, 16, 21, 48,
        57, 39,
    ],
    [
        11, 31, 32, 43, 44, 16, 4, 17, 34, 45, 30, 20, 20, 7, 5, 21, 22, 46, 40, 47, 26, 48, 63,
        58, 10,
    ],
    [
        12, 13, 50, 51, 52, 11, 17, 53, 45, 9, 30, 4, 53, 19, 0, 22, 23, 25, 43, 44, 37, 27, 28,
        10, 55,
    ],
    [
        30, 33, 62, 51, 44, 20, 41, 56, 34, 45, 20, 41, 41, 56, 5, 30, 56, 38, 40, 47, 11, 37, 42,
        57, 8,
    ],
    [
        35, 11, 23, 32, 14, 35, 20, 4, 17, 18, 21, 20, 20, 20, 4, 16, 21, 36, 46, 25, 41, 26, 48,
        49, 58,
    ],
    [
        12, 31, 59, 59, 3, 33, 33, 59, 59, 52, 4, 33, 17, 59, 55, 22, 36, 59, 59, 60, 22, 36, 59,
        25, 55,
    ],
    [
        31, 25, 15, 60, 60, 22, 17, 19, 55, 55, 20, 20, 53, 19, 55, 22, 46, 25, 43, 60, 37, 28, 10,
        55, 52,
    ],
    [
        12, 31, 32, 50, 51, 11, 33, 53, 19, 45, 16, 4, 4, 53, 5, 22, 36, 18, 25, 43, 26, 27, 27,
        28, 10,
    ],
    [
        5, 2, 44, 52, 3, 4, 53, 45, 9, 3, 4, 56, 5, 0, 2, 5, 10, 47, 52, 3, 63, 39, 10, 9, 52,
    ],
    [
        12, 34, 44, 44, 3, 56, 56, 62, 45, 9, 56, 56, 7, 5, 0, 22, 38, 40, 47, 52, 48, 57, 39, 10,
        9,
    ],
    [
        35, 11, 23, 14, 51, 35, 20, 41, 56, 62, 16, 20, 41, 56, 7, 16, 21, 38, 24, 40, 26, 26, 42,
        57, 39,
    ],
    [
        33, 34, 51, 51, 52, 41, 41, 34, 62, 0, 41, 41, 56, 7, 5, 56, 38, 38, 40, 44, 37, 42, 57,
        39, 10,
    ],
    [
        16, 31, 32, 15, 60, 30, 4, 17, 19, 25, 22, 20, 4, 53, 19, 21, 22, 46, 25, 55, 26, 48, 63,
        58, 55,
    ],
];

/// Resolve the per-class coefficient row from a `(set_idx, class_idx)`
/// pair. Wraps eqs. 90 / 91 / 1437 — applies `AlfClassToFiltMap[set][class]`
/// and returns the matching `AlfFixFiltCoeff[row]` 12-tap row.
///
/// Returns `None` when `set_idx >= 16` (i.e. caller should resolve via
/// the APS-signalled path) or when either index is out of range.
#[inline]
pub fn fixed_filter_coeff_row(set_idx: u8, class_idx: u8) -> Option<&'static [i8; 12]> {
    if (set_idx as usize) >= ALF_CLASS_TO_FILT_MAP.len() {
        return None;
    }
    if (class_idx as usize) >= ALF_CLASS_TO_FILT_MAP[0].len() {
        return None;
    }
    let row_idx = ALF_CLASS_TO_FILT_MAP[set_idx as usize][class_idx as usize] as usize;
    if row_idx >= ALF_FIX_FILT_COEFF.len() {
        return None;
    }
    Some(&ALF_FIX_FILT_COEFF[row_idx])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fix_filt_coeff_shape() {
        // The spec (§7.4.3.18 eq. 90) declares i = 0..63, j = 0..11.
        assert_eq!(ALF_FIX_FILT_COEFF.len(), 64);
        assert_eq!(ALF_FIX_FILT_COEFF[0].len(), 12);
    }

    #[test]
    fn class_to_filt_map_shape() {
        // The spec (§7.4.3.18 eq. 91) declares m = 0..15, n = 0..24.
        assert_eq!(ALF_CLASS_TO_FILT_MAP.len(), 16);
        assert_eq!(ALF_CLASS_TO_FILT_MAP[0].len(), 25);
    }

    #[test]
    fn class_to_filt_map_row_indices_are_in_range() {
        // Every entry must point to a valid row of ALF_FIX_FILT_COEFF.
        for set in 0..ALF_CLASS_TO_FILT_MAP.len() {
            for class in 0..ALF_CLASS_TO_FILT_MAP[0].len() {
                let r = ALF_CLASS_TO_FILT_MAP[set][class] as usize;
                assert!(r < 64, "set={set} class={class} row={r} out of range",);
            }
        }
    }

    #[test]
    fn fixed_filter_coeff_row_returns_some_for_valid_inputs() {
        let row = fixed_filter_coeff_row(0, 0);
        assert!(row.is_some());
        assert_eq!(row.unwrap().len(), 12);
    }

    #[test]
    fn fixed_filter_coeff_row_rejects_oob() {
        assert!(fixed_filter_coeff_row(16, 0).is_none());
        assert!(fixed_filter_coeff_row(0, 25).is_none());
    }

    /// Spot-check a known mapping: spec table row 0 (set 0) class 0 → 8;
    /// `ALF_FIX_FILT_COEFF[8]` equals `[0, 0, 8, -8, -2, -7, 4, 4, 2, 1,
    /// -1, 25]` per eq. 90's transposed transcription.
    #[test]
    fn spot_check_set0_class0() {
        let row = fixed_filter_coeff_row(0, 0).unwrap();
        assert_eq!(row, &[0, 0, 8, -8, -2, -7, 4, 4, 2, 1, -1, 25]);
    }

    /// Spot-check: set 15 class 24 → ALF_CLASS_TO_FILT_MAP[15][24] = 55;
    /// `ALF_FIX_FILT_COEFF[55]` equals `[0, 1, 1, -2, 2, -10, 10, 0, -2,
    /// 1, -7, 23]`.
    #[test]
    fn spot_check_set15_class24() {
        assert_eq!(ALF_CLASS_TO_FILT_MAP[15][24], 55);
        let row = fixed_filter_coeff_row(15, 24).unwrap();
        assert_eq!(row, &ALF_FIX_FILT_COEFF[55]);
    }
}
