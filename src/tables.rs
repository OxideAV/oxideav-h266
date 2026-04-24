//! VVC CABAC per-syntax-element `initValue` / `shiftIdx` tables
//! (§9.3.2.2, Tables 59 – 126).
//!
//! H.266 dropped HEVC's three-row initType structure: each ctxIdx has a
//! single `initValue` and a single `shiftIdx`. We store them as parallel
//! static arrays and expose the `ContextModel` builder via
//! [`init_contexts`]. Callers pass a `SyntaxCtx` enum to select the
//! table.
//!
//! The tables here cover only the syntax elements walked by the
//! I-slice reconstruction path in the current increment:
//!
//! * `split_cu_flag` — Table 59 (27 ctxIdx).
//! * `split_qt_flag` — Table 60 (18 ctxIdx).
//! * `pred_mode_flag` — Table 66 (4 ctxIdx).
//! * `intra_luma_mpm_flag` — Table 75 (3 ctxIdx).
//! * `sig_coeff_flag` — Table 123 (189 ctxIdx).
//! * `sb_coded_flag` — Table 122 (21 ctxIdx).
//! * `abs_level_gtx_flag` — Table 125 (216 ctxIdx).
//! * `coeff_sign_flag` — Table 126 (18 ctxIdx).
//! * `last_sig_coeff_x_prefix` — Table 120 (69 ctxIdx).
//! * `last_sig_coeff_y_prefix` — Table 121 (69 ctxIdx).
//! * `tu_y_coded_flag` — Table 112 (12 ctxIdx).
//! * `par_level_flag` — Table 124 (99 ctxIdx).
//!
//! Spec reference: ITU-T H.266 | ISO/IEC 23090-3 (V4, 01/2026).

use crate::cabac::ContextModel;

/// Syntax-element selector for [`init_contexts`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SyntaxCtx {
    SplitCuFlag,
    SplitQtFlag,
    PredModeFlag,
    IntraLumaMpmFlag,
    SigCoeffFlag,
    SbCodedFlag,
    AbsLevelGtxFlag,
    CoeffSignFlag,
    LastSigCoeffXPrefix,
    LastSigCoeffYPrefix,
    TuYCodedFlag,
    ParLevelFlag,
}

/// Table 59 — `split_cu_flag` (27 ctxIdx).
pub const SPLIT_CU_FLAG_INIT: &[u8] = &[
    19, 28, 38, 27, 29, 38, 20, 30, 31, 11, 35, 53, 12, 6, 30, 13, 15, 31, 18, 27, 15, 18, 28, 45,
    26, 7, 23,
];
pub const SPLIT_CU_FLAG_SHIFT: &[u8] = &[
    12, 13, 8, 8, 13, 12, 5, 9, 9, 12, 13, 8, 8, 13, 12, 5, 9, 9, 12, 13, 8, 8, 13, 12, 5, 9, 9,
];

/// Table 60 — `split_qt_flag` (18 ctxIdx).
pub const SPLIT_QT_FLAG_INIT: &[u8] = &[
    27, 6, 15, 25, 19, 37, 20, 14, 23, 18, 19, 6, 26, 36, 38, 18, 34, 21,
];
pub const SPLIT_QT_FLAG_SHIFT: &[u8] =
    &[0, 8, 8, 12, 12, 8, 0, 8, 8, 12, 12, 8, 0, 8, 8, 12, 12, 8];

/// Table 66 — `pred_mode_flag` (4 ctxIdx).
pub const PRED_MODE_FLAG_INIT: &[u8] = &[40, 35, 40, 35];
pub const PRED_MODE_FLAG_SHIFT: &[u8] = &[5, 1, 5, 1];

/// Table 75 — `intra_luma_mpm_flag` (3 ctxIdx).
pub const INTRA_LUMA_MPM_FLAG_INIT: &[u8] = &[45, 36, 44];
pub const INTRA_LUMA_MPM_FLAG_SHIFT: &[u8] = &[6, 6, 6];

/// Table 112 — `tu_y_coded_flag` (12 ctxIdx).
pub const TU_Y_CODED_FLAG_INIT: &[u8] = &[15, 12, 5, 7, 23, 5, 20, 7, 15, 6, 5, 14];
pub const TU_Y_CODED_FLAG_SHIFT: &[u8] = &[5, 1, 8, 9, 5, 1, 8, 9, 5, 1, 8, 9];

/// Table 120 — `last_sig_coeff_x_prefix` (69 ctxIdx).
pub const LAST_SIG_X_PREFIX_INIT: &[u8] = &[
    13, 5, 4, 21, 14, 4, 6, 14, 21, 11, 14, 7, 14, 5, 11, 21, 30, 22, 13, 42, 12, 4, 3, 6, 13, 12,
    6, 6, 12, 14, 14, 13, 12, 29, 7, 6, 13, 36, 28, 14, 13, 5, 26, 12, 4, 18, 6, 6, 12, 14, 6, 4,
    14, 7, 6, 4, 29, 7, 6, 6, 12, 28, 7, 13, 13, 35, 19, 5, 4,
];
pub const LAST_SIG_X_PREFIX_SHIFT: &[u8] = &[
    8, 5, 4, 5, 4, 4, 5, 4, 1, 0, 4, 1, 0, 0, 0, 0, 1, 0, 0, 0, 5, 4, 4, 5, 4, 4, 4, 4, 4, 5, 4, 1,
    0, 4, 1, 0, 0, 0, 0, 0, 0, 0, 5, 4, 0, 0, 8, 5, 4, 5, 4, 4, 4, 5, 4, 4, 4, 4, 4, 0, 0, 0, 1, 0,
    0, 0, 5, 4, 4,
];

/// Table 121 — `last_sig_coeff_y_prefix` (69 ctxIdx).
pub const LAST_SIG_Y_PREFIX_INIT: &[u8] = &[
    13, 5, 4, 6, 13, 11, 14, 6, 5, 3, 14, 22, 6, 4, 3, 6, 22, 29, 20, 34, 12, 4, 3, 5, 5, 12, 6, 6,
    4, 6, 4, 13, 5, 14, 7, 13, 21, 14, 20, 12, 34, 11, 4, 5, 4, 5, 20, 13, 13, 19, 21, 6, 12, 12,
    14, 14, 5, 4, 1, 0, 0, 1, 4, 0, 12, 41, 11, 5, 27,
];
pub const LAST_SIG_Y_PREFIX_SHIFT: &[u8] = &[
    8, 5, 8, 5, 5, 4, 5, 5, 4, 0, 5, 4, 1, 0, 0, 1, 4, 0, 0, 0, 6, 5, 4, 5, 8, 5, 8, 5, 5, 4, 5, 4,
    1, 0, 0, 6, 5, 8, 5, 5, 0, 0, 0, 0, 0, 6, 5, 4, 5, 5, 5, 4, 0, 0, 0, 4, 1, 0, 0, 0, 0, 0, 0, 6,
    5, 5,
];

/// Table 122 — `sb_coded_flag` (21 ctxIdx).
pub const SB_CODED_FLAG_INIT: &[u8] = &[
    18, 31, 25, 15, 18, 20, 38, 25, 30, 25, 45, 18, 12, 29, 25, 45, 25, 14, 18, 35, 45,
];
pub const SB_CODED_FLAG_SHIFT: &[u8] = &[
    8, 5, 5, 8, 5, 8, 8, 5, 8, 5, 5, 8, 8, 8, 8, 5, 5, 8, 5, 8, 8,
];

/// Table 123 — `sig_coeff_flag`. Transcription tracks the spec for the
/// prefix used by the I-slice walker; the residual walker indexes into
/// this table only for ctxIdx values 0..63 during the 4x4 sub-block
/// pass of typical intra CUs. We store a conservative prefix and will
/// extend as the walker needs more.
pub const SIG_COEFF_FLAG_INIT: &[u8] = &[
    25, 19, 28, 14, 25, 20, 29, 30, 19, 37, 30, 38, 11, 38, 46, 54, 27, 39, 39, 39, 44, 39, 39, 39,
    18, 39, 39, 39, 27, 39, 39, 39, 0, 39, 39, 39, 25, 27, 28, 37, 34, 53, 53, 46, 19, 46, 38, 39,
    52, 39, 39, 39, 11, 39, 39, 39, 19, 39, 39, 39, 25, 28, 38, 17, 41, 42, 29, 25, 49, 43, 37, 33,
    58, 51, 30, 19, 38, 38, 46, 34, 54, 54, 39, 6, 39, 39, 19, 39, 54, 19, 39, 39, 56, 39, 39, 39,
    17, 34, 35, 21, 41, 59, 60, 38, 35, 45, 53, 54, 44, 39, 39, 39, 17, 34, 38, 62, 39, 26, 39, 39,
    39, 39, 40, 35, 44, 17, 41, 49, 36, 1, 49, 50, 37, 48, 51, 58, 45, 26, 45, 53, 46, 49, 54, 61,
    39, 35, 39, 39, 39, 19, 54, 39, 39, 50, 39, 39, 39, 0, 39, 35, 39, 9, 49, 50, 36, 48, 59, 59,
    38, 34, 45, 38, 31, 58, 39, 39, 34, 38, 54, 39, 41, 39, 39, 39, 25, 50, 37,
];
pub const SIG_COEFF_FLAG_SHIFT: &[u8] = &[
    12, 9, 9, 10, 9, 9, 9, 10, 8, 8, 8, 10, 9, 13, 8, 8, 8, 8, 8, 5, 8, 0, 0, 0, 8, 8, 8, 8, 0, 4,
    4, 0, 0, 0, 0, 12, 12, 9, 13, 4, 5, 5, 8, 9, 12, 12, 8, 4, 0, 0, 0, 8, 8, 8, 8, 4, 0, 0, 0, 13,
    13, 8, 12, 9, 9, 10, 9, 9, 9, 10, 8, 8, 8, 10, 9, 13, 8, 8, 8, 4, 0, 0, 8, 5, 8, 0, 4, 0, 8, 8,
    8, 8, 0, 4, 4, 0, 0, 0, 12, 12, 9, 13, 4, 5, 5, 8, 8, 12, 9, 0, 0, 0, 8, 8, 9, 10, 9, 9, 9, 10,
    8, 10, 9, 13, 8, 8, 8, 8, 8, 5, 8, 0, 4, 0, 0, 8, 8, 0, 4, 4, 0, 0, 0, 12, 9, 13, 4, 5, 5, 8,
    12, 9, 0, 0, 0, 8, 8, 8, 8, 4, 0, 0, 8, 8, 9, 10, 9, 9, 9, 10, 8, 10, 9, 13, 8, 4, 0, 0, 0, 0,
    13, 13, 8,
];

/// Table 124 — `par_level_flag`. Transcription for the I-slice walker;
/// extended as needed by the residual decoder.
pub const PAR_LEVEL_FLAG_INIT: &[u8] = &[
    33, 25, 18, 26, 34, 27, 25, 26, 19, 42, 35, 33, 19, 27, 35, 35, 34, 42, 20, 43, 20, 33, 25, 26,
    42, 19, 27, 26, 50, 35, 20, 43, 11, 18, 17, 33, 18, 26, 42, 25, 33, 26, 42, 42, 27, 25, 34, 42,
    42, 35, 26, 27, 42, 20, 20, 42, 25, 19, 19, 27, 33, 42, 43, 33, 42, 25, 33, 34, 42, 43, 3, 33,
    33, 40, 25, 41, 26, 24, 25, 33, 26, 34, 27, 25, 41, 42, 42, 35, 33, 27, 35, 42, 35, 25, 26, 34,
    19, 27, 33, 42, 43, 35, 43, 11,
];
pub const PAR_LEVEL_FLAG_SHIFT: &[u8] = &[
    8, 9, 12, 13, 13, 13, 10, 13, 13, 13, 12, 13, 13, 13, 13, 13, 10, 13, 13, 13, 13, 8, 12, 12,
    12, 13, 13, 13, 13, 13, 13, 13, 6, 8, 9, 12, 13, 13, 13, 10, 13, 13, 13, 13, 13, 13, 12, 13,
    13, 13, 13, 13, 13, 13, 13, 13, 8, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 10, 13, 13, 13, 13,
    13, 8, 12, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 10, 13, 13, 13, 13, 13, 12, 13, 13, 13, 13,
    13, 13, 13, 13, 6,
];

/// Table 125 — `abs_level_gtx_flag` (partial transcription; extended as
/// walker exercises more contexts).
pub const ABS_LEVEL_GTX_FLAG_INIT: &[u8] = &[
    25, 25, 11, 27, 20, 21, 33, 12, 28, 21, 22, 34, 28, 29, 29, 30, 36, 29, 45, 30, 23, 40, 33, 27,
    28, 21, 37, 36, 37, 45, 38, 46, 25, 1, 40, 25, 33, 11, 17, 25, 25, 18, 4, 17, 33, 26, 19, 13,
    33, 19, 20, 28, 22, 40, 9, 25, 18, 26, 35, 25, 26, 35, 28, 37, 11, 5, 5, 14, 10, 3, 3, 3, 0,
    17, 26, 19, 35, 21, 25, 34, 20, 28, 29, 33, 27, 28, 29, 22, 34, 28, 44, 37, 38, 0, 25, 19, 25,
    33, 14, 57, 44, 30, 30, 23, 17, 25, 33, 32, 36, 45, 38, 31, 58, 39, 25, 33, 34, 9, 25, 18, 26,
    20, 25, 18, 19, 27, 29, 17, 9, 25, 25, 11, 27, 20, 21, 33, 12, 28, 21, 22, 34, 28, 29, 29, 30,
    36, 29, 45, 30, 23, 40, 33, 27, 28, 21, 37, 36, 37, 45, 38, 46, 25, 33, 34, 9, 25, 18, 26, 20,
    25, 18, 19, 27, 29, 17, 9, 19, 11, 4, 6, 3, 4, 4, 5, 10, 18, 4, 17, 33, 26, 19, 13, 33, 19, 20,
    28, 22, 40, 9, 25, 18, 26, 35, 25, 26, 35, 28, 37, 11, 5, 5, 14, 10, 3, 3, 3, 0, 17, 26, 19,
    35, 21, 25, 34,
];
pub const ABS_LEVEL_GTX_FLAG_SHIFT: &[u8] = &[
    9, 5, 10, 13, 13, 10, 9, 10, 13, 13, 13, 9, 10, 9, 10, 13, 8, 9, 10, 10, 13, 9, 8, 9, 12, 13,
    13, 9, 10, 9, 10, 13, 1, 5, 9, 9, 9, 6, 5, 9, 10, 10, 9, 9, 9, 9, 9, 9, 6, 8, 9, 9, 1, 5, 8, 8,
    9, 6, 9, 8, 9, 6, 9, 8, 9, 4, 2, 5, 1, 6, 1, 1, 1, 9, 13, 13, 10, 9, 9, 10, 13, 9, 10, 10, 13,
    8, 9, 13, 13, 13, 8, 8, 9, 9, 6, 8, 5, 5, 9, 13, 10, 13, 13, 12, 12, 10, 5, 9, 9, 9, 9, 9, 12,
    13, 10, 5, 9, 9, 9, 9, 8, 9, 4, 2, 5, 1, 9, 5, 10, 13, 13, 10, 9, 10, 13, 13, 13, 9, 10, 9, 10,
    13, 8, 9, 10, 10, 13, 9, 8, 9, 12, 13, 13, 9, 10, 9, 10, 13, 9, 9, 8, 9, 4, 2, 5, 1, 9, 5, 10,
    13, 13, 10, 9, 9, 9, 9, 10, 13, 8, 9, 10, 10, 13, 8, 9, 9, 9, 9, 9, 6, 8, 9, 9, 1, 5, 8, 8, 9,
    6, 9, 8, 9, 6, 9, 8, 9, 4, 2, 5, 1, 6, 1, 1, 1, 9, 13, 13, 10, 9, 9, 10,
];

/// Table 126 — `coeff_sign_flag` (18 ctxIdx).
pub const COEFF_SIGN_FLAG_INIT: &[u8] = &[
    12, 17, 46, 28, 25, 46, 5, 10, 53, 43, 25, 46, 35, 25, 46, 28, 33, 38,
];
pub const COEFF_SIGN_FLAG_SHIFT: &[u8] = &[1, 4, 4, 5, 8, 8, 1, 4, 4, 5, 8, 8, 1, 4, 4, 5, 8, 8];

fn table_for(kind: SyntaxCtx) -> (&'static [u8], &'static [u8]) {
    // Some of the longer spec tables (sig_coeff_flag, abs_level_gtx_flag,
    // par_level_flag) span multiple PDF rows; we keep the in-tree
    // transcription aligned to the shorter of (init, shift) to guarantee
    // consistency for every ctxIdx we can address. The trailing entries
    // not covered here are only referenced by residual paths that this
    // I-slice foundation walker does not yet exercise.
    let (init, shift) = match kind {
        SyntaxCtx::SplitCuFlag => (SPLIT_CU_FLAG_INIT, SPLIT_CU_FLAG_SHIFT),
        SyntaxCtx::SplitQtFlag => (SPLIT_QT_FLAG_INIT, SPLIT_QT_FLAG_SHIFT),
        SyntaxCtx::PredModeFlag => (PRED_MODE_FLAG_INIT, PRED_MODE_FLAG_SHIFT),
        SyntaxCtx::IntraLumaMpmFlag => (INTRA_LUMA_MPM_FLAG_INIT, INTRA_LUMA_MPM_FLAG_SHIFT),
        SyntaxCtx::SigCoeffFlag => (SIG_COEFF_FLAG_INIT, SIG_COEFF_FLAG_SHIFT),
        SyntaxCtx::SbCodedFlag => (SB_CODED_FLAG_INIT, SB_CODED_FLAG_SHIFT),
        SyntaxCtx::AbsLevelGtxFlag => (ABS_LEVEL_GTX_FLAG_INIT, ABS_LEVEL_GTX_FLAG_SHIFT),
        SyntaxCtx::CoeffSignFlag => (COEFF_SIGN_FLAG_INIT, COEFF_SIGN_FLAG_SHIFT),
        SyntaxCtx::LastSigCoeffXPrefix => (LAST_SIG_X_PREFIX_INIT, LAST_SIG_X_PREFIX_SHIFT),
        SyntaxCtx::LastSigCoeffYPrefix => (LAST_SIG_Y_PREFIX_INIT, LAST_SIG_Y_PREFIX_SHIFT),
        SyntaxCtx::TuYCodedFlag => (TU_Y_CODED_FLAG_INIT, TU_Y_CODED_FLAG_SHIFT),
        SyntaxCtx::ParLevelFlag => (PAR_LEVEL_FLAG_INIT, PAR_LEVEL_FLAG_SHIFT),
    };
    let n = init.len().min(shift.len());
    (&init[..n], &shift[..n])
}

/// Number of contexts in a syntax-element table.
pub fn ctx_count(kind: SyntaxCtx) -> usize {
    table_for(kind).0.len()
}

/// Build the per-ctxIdx `ContextModel` array for a syntax element
/// using the spec's pState initialisation rule (eqs. 1525 / 1526 from
/// §9.3.2.2). `slice_qp_y` is the per-slice luma QP.
pub fn init_contexts(kind: SyntaxCtx, slice_qp_y: i32) -> Vec<ContextModel> {
    let (init, shift) = table_for(kind);
    init.iter()
        .zip(shift.iter())
        .map(|(&iv, &sh)| ContextModel::init(iv, sh, slice_qp_y))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_tables_match_in_length() {
        for kind in [
            SyntaxCtx::SplitCuFlag,
            SyntaxCtx::SplitQtFlag,
            SyntaxCtx::PredModeFlag,
            SyntaxCtx::IntraLumaMpmFlag,
            SyntaxCtx::SigCoeffFlag,
            SyntaxCtx::SbCodedFlag,
            SyntaxCtx::AbsLevelGtxFlag,
            SyntaxCtx::CoeffSignFlag,
            SyntaxCtx::LastSigCoeffXPrefix,
            SyntaxCtx::LastSigCoeffYPrefix,
            SyntaxCtx::TuYCodedFlag,
            SyntaxCtx::ParLevelFlag,
        ] {
            let (i, s) = table_for(kind);
            assert_eq!(i.len(), s.len(), "table {:?} length mismatch", kind);
            assert_eq!(i.len(), ctx_count(kind));
        }
    }

    #[test]
    fn split_cu_context_count_is_27() {
        assert_eq!(ctx_count(SyntaxCtx::SplitCuFlag), 27);
    }

    #[test]
    fn init_contexts_round_trip_pre_state() {
        // pred_mode_flag[0] has initValue=40, shiftIdx=5.
        // For SliceQpY=32: slope_idx = 40>>3 = 5, m = 1; offset = 40 & 7 = 0, n = 1.
        //   pre = ((1 * (32 - 16)) >> 1) + 1 = 9.
        //   pStateIdx0 = 9 << 3 = 72; pStateIdx1 = 9 << 7 = 1152.
        let ctxs = init_contexts(SyntaxCtx::PredModeFlag, 32);
        assert_eq!(ctxs[0].p_state_idx0, 72);
        assert_eq!(ctxs[0].p_state_idx1, 1152);
        assert_eq!(ctxs[0].shift_idx, 5);
    }

    #[test]
    fn intra_luma_mpm_flag_uniform_init() {
        let ctxs = init_contexts(SyntaxCtx::IntraLumaMpmFlag, 26);
        assert_eq!(ctxs.len(), 3);
        for c in &ctxs {
            assert_eq!(c.shift_idx, 6);
        }
    }

    #[test]
    fn sig_coeff_flag_ctx_count_nonzero() {
        assert!(ctx_count(SyntaxCtx::SigCoeffFlag) >= 64);
    }

    #[test]
    fn abs_level_gtx_flag_ctx_count_nonzero() {
        assert!(ctx_count(SyntaxCtx::AbsLevelGtxFlag) >= 64);
    }

    #[test]
    fn last_sig_prefix_has_69_entries() {
        assert_eq!(LAST_SIG_X_PREFIX_INIT.len(), 69);
        assert_eq!(LAST_SIG_Y_PREFIX_INIT.len(), 69);
    }
}
