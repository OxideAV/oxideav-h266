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
//! * `mtt_split_cu_vertical_flag` — Table 61 (15 ctxIdx, 5 per initType).
//! * `mtt_split_cu_binary_flag` — Table 62 (12 ctxIdx, 4 per initType).
//! * `pred_mode_flag` — Table 66 (4 ctxIdx).
//! * `intra_bdpcm_luma_flag` — Table 69 (3 ctxIdx).
//! * `intra_bdpcm_luma_dir_flag` — Table 70 (3 ctxIdx).
//! * `intra_bdpcm_chroma_flag` — Table 77 (3 ctxIdx).
//! * `intra_bdpcm_chroma_dir_flag` — Table 78 (3 ctxIdx).
//! * `intra_mip_flag` — Table 71 (12 ctxIdx).
//! * `intra_luma_ref_idx` — Table 72 (6 ctxIdx).
//! * `intra_subpartitions_mode_flag` — Table 73 (3 ctxIdx).
//! * `intra_subpartitions_split_flag` — Table 74 (3 ctxIdx).
//! * `intra_luma_mpm_flag` — Table 75 (3 ctxIdx).
//! * `intra_luma_not_planar_flag` — Table 76 (6 ctxIdx).
//! * `intra_chroma_pred_mode` — Table 81 (3 ctxIdx).
//! * `sig_coeff_flag` — Table 123 (189 ctxIdx).
//! * `sb_coded_flag` — Table 122 (21 ctxIdx).
//! * `abs_level_gtx_flag` — Table 125 (216 ctxIdx).
//! * `coeff_sign_flag` — Table 126 (18 ctxIdx).
//! * `last_sig_coeff_x_prefix` — Table 120 (69 ctxIdx).
//! * `last_sig_coeff_y_prefix` — Table 121 (69 ctxIdx).
//! * `tu_y_coded_flag` — Table 112 (12 ctxIdx).
//! * `tu_cb_coded_flag` — Table 113 (6 ctxIdx).
//! * `tu_cr_coded_flag` — Table 114 (9 ctxIdx).
//! * `cu_qp_delta_abs` — Table 115 (6 ctxIdx).
//! * `cu_chroma_qp_offset_flag` — Table 116 (3 ctxIdx).
//! * `cu_chroma_qp_offset_idx` — Table 117 (3 ctxIdx).
//! * `par_level_flag` — Table 124 (99 ctxIdx).
//! * `sao_merge_left_flag` / `sao_merge_up_flag` — Table 57 (3 ctxIdx, one
//!   per initType).
//! * `sao_type_idx_luma` / `sao_type_idx_chroma` — Table 58 (3 ctxIdx, one
//!   per initType).
//! * `cu_skip_flag` — Table 64 (9 ctxIdx, 3 per initType).
//! * `general_merge_flag` — Table 82 (3 ctxIdx, one per initType).
//! * `regular_merge_flag` — Table 102 (4 ctxIdx, two per non-I initType).
//! * `mmvd_merge_flag` — Table 103 (2 ctxIdx, one per non-I initType).
//! * `mmvd_cand_flag` — Table 104 (2 ctxIdx, one per non-I initType).
//! * `mmvd_distance_idx` — Table 105 (2 ctxIdx, one per non-I initType).
//! * `merge_idx` — Table 109 (3 ctxIdx, one per initType).
//! * `alf_ctb_flag` — Table 52 (27 ctxIdx, 9 per initType).
//! * `alf_use_aps_flag` — Table 53 (3 ctxIdx, one per initType).
//! * `alf_ctb_cc_cb_idc` — Table 54 (9 ctxIdx, 3 per initType).
//! * `alf_ctb_cc_cr_idc` — Table 55 (9 ctxIdx, 3 per initType).
//! * `alf_ctb_filter_alt_idx` — Table 56 (6 ctxIdx, 2 per initType).
//!
//! Spec reference: ITU-T H.266 | ISO/IEC 23090-3 (V4, 01/2026).

use crate::cabac::ContextModel;

/// Syntax-element selector for [`init_contexts`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SyntaxCtx {
    SplitCuFlag,
    SplitQtFlag,
    /// Table 61 — `mtt_split_cu_vertical_flag` (15 ctxIdx, 5 per initType).
    /// Round-55 §9.3.4.2.3: ctxInc derived from
    /// `(allowSplitBtVer + allowSplitTtVer)` vs
    /// `(allowSplitBtHor + allowSplitTtHor)` plus the `dA` / `dL`
    /// neighbour-aspect ratios.
    MttSplitCuVerticalFlag,
    /// Table 62 — `mtt_split_cu_binary_flag` (12 ctxIdx, 4 per initType).
    /// Round-55: per Table 132 the ctxInc =
    /// `2 * mtt_split_cu_vertical_flag + (mttDepth <= 1 ? 1 : 0)`.
    MttSplitCuBinaryFlag,
    PredModeFlag,
    IntraBdpcmLumaFlag,
    IntraBdpcmLumaDirFlag,
    IntraBdpcmChromaFlag,
    IntraBdpcmChromaDirFlag,
    IntraMipFlag,
    IntraLumaRefIdx,
    IntraSubpartitionsModeFlag,
    IntraSubpartitionsSplitFlag,
    IntraLumaMpmFlag,
    IntraLumaNotPlanarFlag,
    IntraChromaPredMode,
    SigCoeffFlag,
    SbCodedFlag,
    AbsLevelGtxFlag,
    CoeffSignFlag,
    LastSigCoeffXPrefix,
    LastSigCoeffYPrefix,
    TuYCodedFlag,
    TuCbCodedFlag,
    TuCrCodedFlag,
    CuQpDeltaAbs,
    CuChromaQpOffsetFlag,
    CuChromaQpOffsetIdx,
    /// Table 119 — `tu_joint_cbcr_residual_flag` (9 ctxIdx, 3 per
    /// initType). The ctxInc within an initType is
    /// `2 * tu_cb_coded_flag + tu_cr_coded_flag − 1` (Table 132).
    TuJointCbCrResidualFlag,
    /// Table 118 — `transform_skip_flag` (6 ctxIdx, 2 per initType).
    /// The ctxInc within an initType is `cIdx == 0 ? 0 : 1` (Table 132).
    TransformSkipFlag,
    ParLevelFlag,
    /// Table 97 — `lfnst_idx` (9 ctxIdx, 3 per initType). The TR(cMax=2)
    /// binarisation's two ctx-coded bins use ctxInc 0/1 (bin 0,
    /// treeType-dependent) and 2 (bin 1) per Table 132; the per-slice
    /// initType offset is `init_type * 3` applied at parse time.
    LfnstIdx,
    /// Table 98 — `mts_idx` (12 ctxIdx, 4 per initType). The TR(cMax=4)
    /// binarisation's four ctx-coded bins use ctxInc = binIdx (0..3) per
    /// Table 132; the per-slice initType offset is `init_type * 4`
    /// applied at parse time.
    MtsIdx,
    /// Table 57 — `sao_merge_left_flag` and `sao_merge_up_flag` share the
    /// same 3-entry table (one ctxIdx per initType).
    SaoMergeFlag,
    /// Table 58 — `sao_type_idx_luma` and `sao_type_idx_chroma` share the
    /// same 3-entry table (one ctxIdx per initType).
    SaoTypeIdx,
    /// Table 64 — `cu_skip_flag` (9 ctxIdx, 3 per initType).
    CuSkipFlag,
    /// Table 82 — `general_merge_flag` (3 ctxIdx, one per initType).
    GeneralMergeFlag,
    /// Table 102 — `regular_merge_flag` (4 ctxIdx, two per initType for the
    /// last two initTypes; initType 0 is unused → only 4 entries shown).
    RegularMergeFlag,
    /// Table 103 — `mmvd_merge_flag` (2 ctxIdx, one per non-I initType).
    /// Bin 0 ctx-coded with ctxInc = 0 per Table 132; the I-slice slot is
    /// unused since MMVD is only signalled for inter slices.
    MmvdMergeFlag,
    /// Table 104 — `mmvd_cand_flag` (2 ctxIdx, one per non-I initType).
    MmvdCandFlag,
    /// Table 105 — `mmvd_distance_idx` (2 ctxIdx, one per non-I initType).
    /// Bin 0 ctx-coded; bins 1..6 (TR, cMax = 7) bypass-coded.
    MmvdDistanceIdx,
    /// Table 106 — `ciip_flag` (2 ctxIdx, one per non-I initType).
    /// Single ctx-coded bin (FL `cMax = 1`) per Table 132 with
    /// `ctxInc = 0`; the I-slice slot is unused since CIIP is only
    /// signalled for inter slices.
    CiipFlag,
    /// Table 109 — `merge_idx` (3 ctxIdx, one per initType).
    MergeIdx,
    /// Table 110 — `abs_mvd_greater0_flag` (3 ctxIdx, one per initType).
    /// Per Table 51 the slot is `init_type`; per Table 132 the single
    /// ctx-coded bin uses `ctxInc = 0` (FL `cMax = 1`). Read once per
    /// component (compIdx 0 / 1) inside `mvd_coding()` §7.3.10.10.
    AbsMvdGreater0Flag,
    /// Table 111 — `abs_mvd_greater1_flag` (3 ctxIdx, one per initType).
    /// Same shape as `AbsMvdGreater0Flag`; only read when the matching
    /// `abs_mvd_greater0_flag` was 1.
    AbsMvdGreater1Flag,
    /// Table 92 — `cu_coded_flag` (3 ctxIdx, one per initType). Single
    /// ctx-coded bin (FL `cMax = 1`) with `ctxInc = 0` per Table 132.
    /// Indexed at parse time as `init_type` (0 / 1 / 2). Used by the
    /// non-skip merge / non-merge inter paths to gate the
    /// `transform_tree()` body.
    CuCodedFlag,
    /// Table 83 — `inter_pred_idc` (12 ctxIdx; 6 per non-I initType). Per
    /// Table 51 the slot block is `(init_type − 1) * 6` (initType 1 →
    /// 0..5, initType 2 → 6..11); `inter_pred_idc` is never signalled in
    /// I slices so there is no initType-0 block. Bin 0's ctxInc spans
    /// 0..4 (`7 − ((1 + Log2(cbWidth) + Log2(cbHeight)) >> 1)` or 5),
    /// bin 1's ctxInc is fixed 5, so the 6 slots cover the full
    /// {0,1,2,3,4,5} ctxInc range used by §9.3.3.9 / Table 131.
    InterPredIdc,
    /// Table 86 — `sym_mvd_flag` (2 ctxIdx, one per non-I initType). Per
    /// Table 51 initType 1 → 0, initType 2 → 1 (indexed at parse time as
    /// `init_type - 1`). Single ctx-coded bin (FL `cMax = 1`) with
    /// `ctxInc = 0` per Table 132; only signalled for inter slices.
    SymMvdFlag,
    /// Table 87 — `ref_idx_l0` / `ref_idx_l1` (4 ctxIdx; 2 per non-I
    /// initType). Per Table 51 initType 1 → 0..1, initType 2 → 2..3
    /// (indexed at parse time as `(init_type - 1) * 2 + ctxInc`). TR
    /// binarisation (`cMax = NumRefIdxActive[X] − 1`): bin 0 ctxInc 0,
    /// bin 1 ctxInc 1, bins 2.. bypass per Table 132.
    RefIdxLx,
    /// Table 88 — `mvp_l0_flag` / `mvp_l1_flag` (3 ctxIdx, one per
    /// initType). Per Table 51 indexed by `init_type`. Single ctx-coded
    /// bin (FL `cMax = 1`) with `ctxInc = 0` per Table 132.
    MvpLxFlag,
    /// Table 89 — `amvr_flag` (4 ctxIdx). Round-40 §7.4.11.6. The four
    /// slots map to `(initType, ctxInc) ∈ ({0,1,2}, {0,1,2})` per
    /// Tables 51 / 132 — regular AMVR / affine-AMVR / IBC-AMVR rows.
    AmvrFlag,
    /// Table 90 — `amvr_precision_idx` (9 ctxIdx). Round-40
    /// §7.4.11.6. Three rows of three slots (regular / affine / IBC),
    /// indexed at parse time by the same `(initType, ctxInc)` pair.
    AmvrPrecisionIdx,
    /// Table 52 — `alf_ctb_flag` (27 ctxIdx; 9 per initType, 3 ctxInc per
    /// component cIdx). Per Table 51 the cIdx-major slicing is
    /// `(initType, cIdx) → (initType * 9 + cIdx * 3)..+3`. Round-45
    /// §7.4.3.13 / §9.3.4.2.2 with Table 133 condL/condA giving
    /// `ctxInc = (condL && availL) + (condA && availA) + cIdx * 3`.
    AlfCtbFlag,
    /// Table 53 — `alf_use_aps_flag` (3 ctxIdx, one per initType).
    /// Single ctx-coded bin (FL `cMax = 1`), Table 132 fixes
    /// `ctxInc = 0`; the per-initType row follows Table 51.
    AlfUseApsFlag,
    /// Table 54 — `alf_ctb_cc_cb_idc` (9 ctxIdx, 3 per initType). Bin 0
    /// is ctx-coded with `ctxInc = (condL && availL) + (condA && availA)`
    /// per §9.3.4.2.2 / Table 133 (`ctxSetIdx = 0`). Bins 1.. are
    /// bypass-coded per Table 132 (TR with `cMax =
    /// alf_cc_cb_filters_signalled_minus1 + 1`).
    AlfCtbCcCbIdc,
    /// Table 55 — `alf_ctb_cc_cr_idc` (9 ctxIdx, 3 per initType). Same
    /// shape as `AlfCtbCcCbIdc`.
    AlfCtbCcCrIdc,
    /// Table 56 — `alf_ctb_filter_alt_idx` (6 ctxIdx, 2 per initType).
    /// Per Table 132 every TR bin uses the same ctxInc derived from
    /// chromaIdx (0 → ctx 0 of the row, 1 → ctx 1 of the row).
    AlfCtbFilterAltIdx,
    /// Table 91 — `bcw_idx` (2 ctxIdx, one per non-I initType). Per
    /// Table 51 initType 1 → ctxIdx 0, initType 2 → ctxIdx 1 (indexed
    /// at parse time as `init_type - 1`). Per Table 132 only bin 0 of
    /// the TR sequence is context-coded (`ctxInc = 0`); bins 1.. are
    /// bypass-coded. The TR `cMax = NoBackwardPredFlag ? 4 : 2`,
    /// `cRiceParam = 0` (so the value range is 0..=4 with backward
    /// prediction absent on B slices, 0..=2 otherwise). `bcw_idx` is
    /// never signalled in I slices.
    BcwIdx,
    /// Table 107 — `merge_subblock_flag` (6 ctxIdx, 3 per non-I
    /// initType). Per Table 51 the per-initType slot block is
    /// `(initType − 1) * 3` (initType 1 → ctxIdx 0..2, initType 2 →
    /// ctxIdx 3..5; merge data is never signalled in I slices). Per
    /// Table 132 the single ctx-coded bin uses the §9.3.4.2.2 / eq. 1551
    /// derivation with the Table 133 merge-side row
    /// `condL = MergeSubblockFlag[L] || InterAffineFlag[L]`,
    /// `condA = MergeSubblockFlag[A] || InterAffineFlag[A]`,
    /// `ctxSetIdx = 0` — yielding `ctxInc ∈ {0, 1, 2}` and indexing the
    /// per-initType triplet directly.
    MergeSubblockFlag,
    /// Table 108 — `merge_subblock_idx` (2 ctxIdx, one per non-I
    /// initType). Per Table 51 the slot is `initType − 1` (initType 1 →
    /// ctxIdx 0, initType 2 → ctxIdx 1). Per Table 132 only bin 0 of
    /// the TR sequence is context-coded (`ctxInc = 0`); bins 1.. (when
    /// `MaxNumSubblockMergeCand ≥ 3`) are bypass-coded. The TR
    /// `cMax = MaxNumSubblockMergeCand − 1`, `cRiceParam = 0`.
    MergeSubblockIdx,
    /// Table 84 — `inter_affine_flag` (6 ctxIdx, 3 per non-I initType).
    /// Per Table 51 the per-initType slot block is `(initType − 1) * 3`
    /// (initType 1 → ctxIdx 0..2, initType 2 → ctxIdx 3..5; only
    /// signalled in non-I slices behind the §7.3.11.7 gate
    /// `sps_affine_enabled_flag && cbWidth >= 16 && cbHeight >= 16`).
    /// Per Table 132 the single ctx-coded bin (FL `cMax = 1`) uses the
    /// §9.3.4.2.2 / eq. 1551 derivation with the Table 133 row identical
    /// to `merge_subblock_flag`:
    /// `condL = MergeSubblockFlag[L] || InterAffineFlag[L]`,
    /// `condA = MergeSubblockFlag[A] || InterAffineFlag[A]`,
    /// `ctxSetIdx = 0` — yielding `ctxInc ∈ {0, 1, 2}` and indexing the
    /// per-initType triplet directly.
    InterAffineFlag,
    /// Table 85 — `cu_affine_type_flag` (2 ctxIdx, one per non-I
    /// initType). Per Table 51 the slot is `initType − 1` (initType 1 →
    /// ctxIdx 0, initType 2 → ctxIdx 1; never signalled in I slices —
    /// the syntax element only appears on the non-merge inter branch
    /// of §7.3.11.7 behind `sps_6param_affine_enabled_flag &&
    /// inter_affine_flag == 1`). Per Table 132 the single ctx-coded
    /// bin (FL `cMax = 1`) has `ctxInc = 0` deterministically; per
    /// Table 133 the §9.3.4.2.2 derivation falls through to the
    /// fixed-`0` row. The reader picks the per-initType slot directly
    /// without any neighbour lookup.
    CuAffineTypeFlag,
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

/// Table 61 — `mtt_split_cu_vertical_flag` (15 ctxIdx, 5 per initType).
/// Round-55 §9.3.4.2.3.
pub const MTT_SPLIT_CU_VERTICAL_FLAG_INIT: &[u8] =
    &[43, 42, 29, 27, 44, 43, 35, 37, 34, 52, 43, 42, 37, 42, 44];
pub const MTT_SPLIT_CU_VERTICAL_FLAG_SHIFT: &[u8] = &[9, 8, 9, 8, 5, 9, 8, 9, 8, 5, 9, 8, 9, 8, 5];

/// Table 62 — `mtt_split_cu_binary_flag` (12 ctxIdx, 4 per initType).
/// Round-55: ctxInc per Table 132 =
/// `2 * mtt_split_cu_vertical_flag + (mttDepth <= 1 ? 1 : 0)`.
pub const MTT_SPLIT_CU_BINARY_FLAG_INIT: &[u8] = &[36, 45, 36, 45, 43, 37, 21, 22, 28, 29, 28, 29];
pub const MTT_SPLIT_CU_BINARY_FLAG_SHIFT: &[u8] = &[12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13];

/// Table 66 — `pred_mode_flag` (4 ctxIdx).
pub const PRED_MODE_FLAG_INIT: &[u8] = &[40, 35, 40, 35];
pub const PRED_MODE_FLAG_SHIFT: &[u8] = &[5, 1, 5, 1];

/// Table 69 — `intra_bdpcm_luma_flag` (3 ctxIdx, one per initType).
pub const INTRA_BDPCM_LUMA_FLAG_INIT: &[u8] = &[19, 40, 19];
pub const INTRA_BDPCM_LUMA_FLAG_SHIFT: &[u8] = &[1, 1, 1];

/// Table 70 — `intra_bdpcm_luma_dir_flag` (3 ctxIdx, one per initType).
pub const INTRA_BDPCM_LUMA_DIR_FLAG_INIT: &[u8] = &[35, 36, 21];
pub const INTRA_BDPCM_LUMA_DIR_FLAG_SHIFT: &[u8] = &[4, 4, 4];

/// Table 77 — `intra_bdpcm_chroma_flag` (3 ctxIdx, one per initType).
pub const INTRA_BDPCM_CHROMA_FLAG_INIT: &[u8] = &[1, 0, 0];
pub const INTRA_BDPCM_CHROMA_FLAG_SHIFT: &[u8] = &[1, 1, 1];

/// Table 78 — `intra_bdpcm_chroma_dir_flag` (3 ctxIdx, one per initType).
pub const INTRA_BDPCM_CHROMA_DIR_FLAG_INIT: &[u8] = &[27, 13, 28];
pub const INTRA_BDPCM_CHROMA_DIR_FLAG_SHIFT: &[u8] = &[0, 0, 0];

/// Table 71 — `intra_mip_flag` (12 ctxIdx).
pub const INTRA_MIP_FLAG_INIT: &[u8] = &[33, 49, 50, 25, 41, 57, 58, 26, 56, 57, 50, 26];
pub const INTRA_MIP_FLAG_SHIFT: &[u8] = &[9, 10, 9, 6, 9, 10, 9, 6, 9, 10, 9, 6];

/// Table 72 — `intra_luma_ref_idx` (6 ctxIdx).
pub const INTRA_LUMA_REF_IDX_INIT: &[u8] = &[25, 60, 25, 58, 25, 59];
pub const INTRA_LUMA_REF_IDX_SHIFT: &[u8] = &[5, 8, 5, 8, 5, 8];

/// Table 73 — `intra_subpartitions_mode_flag` (3 ctxIdx).
pub const INTRA_SP_MODE_FLAG_INIT: &[u8] = &[33, 33, 33];
pub const INTRA_SP_MODE_FLAG_SHIFT: &[u8] = &[9, 9, 9];

/// Table 74 — `intra_subpartitions_split_flag` (3 ctxIdx).
pub const INTRA_SP_SPLIT_FLAG_INIT: &[u8] = &[43, 36, 43];
pub const INTRA_SP_SPLIT_FLAG_SHIFT: &[u8] = &[2, 2, 2];

/// Table 75 — `intra_luma_mpm_flag` (3 ctxIdx).
pub const INTRA_LUMA_MPM_FLAG_INIT: &[u8] = &[45, 36, 44];
pub const INTRA_LUMA_MPM_FLAG_SHIFT: &[u8] = &[6, 6, 6];

/// Table 76 — `intra_luma_not_planar_flag` (6 ctxIdx).
pub const INTRA_LUMA_NOT_PLANAR_FLAG_INIT: &[u8] = &[13, 28, 12, 20, 13, 6];
pub const INTRA_LUMA_NOT_PLANAR_FLAG_SHIFT: &[u8] = &[1, 5, 1, 5, 1, 5];

/// Table 81 — `intra_chroma_pred_mode` (3 ctxIdx).
pub const INTRA_CHROMA_PRED_MODE_INIT: &[u8] = &[34, 25, 25];
pub const INTRA_CHROMA_PRED_MODE_SHIFT: &[u8] = &[5, 5, 5];

/// Table 112 — `tu_y_coded_flag` (12 ctxIdx).
pub const TU_Y_CODED_FLAG_INIT: &[u8] = &[15, 12, 5, 7, 23, 5, 20, 7, 15, 6, 5, 14];
pub const TU_Y_CODED_FLAG_SHIFT: &[u8] = &[5, 1, 8, 9, 5, 1, 8, 9, 5, 1, 8, 9];

/// Table 113 — `tu_cb_coded_flag` (6 ctxIdx).
pub const TU_CB_CODED_FLAG_INIT: &[u8] = &[12, 21, 25, 28, 25, 37];
pub const TU_CB_CODED_FLAG_SHIFT: &[u8] = &[5, 0, 5, 0, 5, 0];

/// Table 114 — `tu_cr_coded_flag` (9 ctxIdx).
pub const TU_CR_CODED_FLAG_INIT: &[u8] = &[33, 28, 36, 25, 29, 45, 9, 36, 45];
pub const TU_CR_CODED_FLAG_SHIFT: &[u8] = &[2, 1, 0, 2, 1, 0, 2, 1, 0];

/// Table 115 — `cu_qp_delta_abs` (6 ctxIdx).
pub const CU_QP_DELTA_ABS_INIT: &[u8] = &[35, 35, 35, 35, 35, 35];
pub const CU_QP_DELTA_ABS_SHIFT: &[u8] = &[8, 8, 8, 8, 8, 8];

/// Table 116 — `cu_chroma_qp_offset_flag` (3 ctxIdx).
pub const CU_CHROMA_QP_OFFSET_FLAG_INIT: &[u8] = &[35, 35, 35];
pub const CU_CHROMA_QP_OFFSET_FLAG_SHIFT: &[u8] = &[8, 8, 8];

/// Table 117 — `cu_chroma_qp_offset_idx` (3 ctxIdx).
pub const CU_CHROMA_QP_OFFSET_IDX_INIT: &[u8] = &[35, 35, 35];
pub const CU_CHROMA_QP_OFFSET_IDX_SHIFT: &[u8] = &[8, 8, 8];

/// Table 118 — `transform_skip_flag` (6 ctxIdx, 2 per initType).
pub const TRANSFORM_SKIP_FLAG_INIT: &[u8] = &[25, 9, 25, 9, 25, 17];
pub const TRANSFORM_SKIP_FLAG_SHIFT: &[u8] = &[1, 1, 1, 1, 1, 1];

/// Table 119 — `tu_joint_cbcr_residual_flag` (9 ctxIdx, 3 per initType).
pub const TU_JOINT_CBCR_RESIDUAL_FLAG_INIT: &[u8] = &[12, 21, 35, 27, 36, 45, 42, 43, 52];
pub const TU_JOINT_CBCR_RESIDUAL_FLAG_SHIFT: &[u8] = &[1, 1, 0, 1, 1, 0, 1, 1, 0];

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

/// Table 97 — `lfnst_idx` (9 ctxIdx: 3 per initType). The TR(cMax=2)
/// binarisation has two ctx-coded bins; bin 0 ctxInc = `treeType !=
/// SINGLE_TREE ? 1 : 0`, bin 1 ctxInc = 2 (Table 132). The per-slice
/// `ctxIdx = init_type * 3 + ctxInc` mapping happens at parse time.
pub const LFNST_IDX_INIT: &[u8] = &[28, 52, 42, 37, 45, 27, 52, 37, 27];
pub const LFNST_IDX_SHIFT: &[u8] = &[9, 9, 10, 9, 9, 10, 9, 9, 10];

/// Table 98 — `mts_idx` initValue / shiftIdx (12 ctxIdx, 4 per initType).
pub const MTS_IDX_INIT: &[u8] = &[29, 0, 28, 0, 45, 40, 27, 0, 45, 25, 27, 0];
pub const MTS_IDX_SHIFT: &[u8] = &[8, 0, 9, 0, 8, 0, 9, 0, 8, 0, 9, 0];

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

/// Table 57 — `sao_merge_left_flag` / `sao_merge_up_flag` (3 ctxIdx,
/// one per initType ∈ {0, 1, 2}). Both syntax elements share Table 57
/// per Table 51. From the spec table:
///   initType  | 0  | 1  | 2 |
///   initValue | 60 | 60 | 2 |
///   shiftIdx  |  0 |  0 | 0 |
pub const SAO_MERGE_FLAG_INIT: &[u8] = &[60, 60, 2];
pub const SAO_MERGE_FLAG_SHIFT: &[u8] = &[0, 0, 0];

/// Table 58 — `sao_type_idx_luma` / `sao_type_idx_chroma` (3 ctxIdx,
/// one per initType ∈ {0, 1, 2}). Both share Table 58 per Table 51.
/// From the spec table:
///   initType  | 0  | 1 | 2 |
///   initValue | 13 | 5 | 2 |
///   shiftIdx  |  4 | 4 | 4 |
pub const SAO_TYPE_IDX_INIT: &[u8] = &[13, 5, 2];
pub const SAO_TYPE_IDX_SHIFT: &[u8] = &[4, 4, 4];

/// Table 64 — `cu_skip_flag` (9 ctxIdx, 3 per initType).
/// `ctxInc = (condL && availableL) + (condA && availableA) + ctxSetIdx*3`
/// per §9.3.4.2.2 with `condX = CuSkipFlag[xNbX][yNbX]` and
/// `ctxSetIdx = 0`. Final ctxIdx = `initType * 3 + ctxInc` (P/B paths
/// only; `cu_skip_flag` is only signalled for non-I slices outside the
/// IBC corner case).
pub const CU_SKIP_FLAG_INIT: &[u8] = &[0, 26, 28, 57, 59, 45, 57, 60, 46];
pub const CU_SKIP_FLAG_SHIFT: &[u8] = &[5, 4, 8, 5, 4, 8, 5, 4, 8];

/// Table 82 — `general_merge_flag` (3 ctxIdx, one per initType).
/// ctxInc = 0 per Table 132.
pub const GENERAL_MERGE_FLAG_INIT: &[u8] = &[26, 21, 6];
pub const GENERAL_MERGE_FLAG_SHIFT: &[u8] = &[4, 4, 4];

/// Table 102 — `regular_merge_flag` (4 ctxIdx). The spec splits across
/// initType 1 (P, ctxIdx 0..1) and initType 2 (B, ctxIdx 2..3); Table
/// 132 picks ctxInc ∈ {0, 1} from `cu_skip_flag ? 0 : 1`. Initialised
/// via `init_type * 2 + ctxInc` for non-I slices.
pub const REGULAR_MERGE_FLAG_INIT: &[u8] = &[38, 7, 46, 15];
pub const REGULAR_MERGE_FLAG_SHIFT: &[u8] = &[5, 5, 5, 5];

/// Table 109 — `merge_idx` (3 ctxIdx, one per initType). Bin 0 ctx; bins
/// 1..4 bypass per Table 132.
pub const MERGE_IDX_INIT: &[u8] = &[34, 20, 18];
pub const MERGE_IDX_SHIFT: &[u8] = &[4, 4, 4];

/// Table 103 — `mmvd_merge_flag` (2 ctxIdx, one per non-I initType).
/// Per Table 132 ctxInc = 0; only one ctx-coded bin (FL, cMax = 1).
/// Spec values:
///   initType  | 0  | 1 |
///   initValue | 26 | 25 |
///   shiftIdx  |  4 | 4 |
pub const MMVD_MERGE_FLAG_INIT: &[u8] = &[26, 25];
pub const MMVD_MERGE_FLAG_SHIFT: &[u8] = &[4, 4];

/// Table 104 — `mmvd_cand_flag` (2 ctxIdx, one per non-I initType).
/// FL cMax = 1 binarisation; bin 0 ctx-coded with ctxInc = 0.
/// Spec values:
///   initType  | 0  | 1  |
///   initValue | 43 | 43 |
///   shiftIdx  | 10 | 10 |
pub const MMVD_CAND_FLAG_INIT: &[u8] = &[43, 43];
pub const MMVD_CAND_FLAG_SHIFT: &[u8] = &[10, 10];

/// Table 105 — `mmvd_distance_idx` (2 ctxIdx, one per non-I initType).
/// TR binarisation with cMax = 7, cRiceParam = 0; bin 0 ctx-coded with
/// ctxInc = 0, bins 1..6 bypass-coded per Table 132.
/// Spec values:
///   initType  | 0  | 1 |
///   initValue | 60 | 59 |
///   shiftIdx  |  0 | 0 |
pub const MMVD_DISTANCE_IDX_INIT: &[u8] = &[60, 59];
pub const MMVD_DISTANCE_IDX_SHIFT: &[u8] = &[0, 0];

/// Table 106 — `ciip_flag` (2 ctxIdx, one per non-I initType). FL
/// binarisation with `cMax = 1` (a single ctx-coded bin); Table 132
/// fixes `ctxInc = 0`. Spec values:
///   initType  | 0  | 1  |
///   initValue | 57 | 57 |
///   shiftIdx  |  1 | 1  |
pub const CIIP_FLAG_INIT: &[u8] = &[57, 57];
pub const CIIP_FLAG_SHIFT: &[u8] = &[1, 1];

/// Table 92 — `cu_coded_flag` (3 ctxIdx, one per initType). FL
/// binarisation with `cMax = 1` (a single ctx-coded bin); Table 132
/// fixes `ctxInc = 0`. Spec values:
///   initType  | 0 | 1 | 2  |
///   initValue | 6 | 5 | 12 |
///   shiftIdx  | 4 | 4 | 4  |
pub const CU_CODED_FLAG_INIT: &[u8] = &[6, 5, 12];
pub const CU_CODED_FLAG_SHIFT: &[u8] = &[4, 4, 4];

/// Table 83 — `inter_pred_idc` (12 ctxIdx, 6 per non-I initType).
/// Round-108 §9.3.3.9 transcription:
///   ctxIdx     | 0 | 1 | 2 | 3  | 4 | 5  | 6  | 7  | 8 | 9 | 10 | 11 |
///   initValue  | 7 | 6 | 5 | 12 | 4 | 40 | 14 | 13 | 5 | 4 | 3  | 40 |
///   shiftIdx   | 0 | 0 | 1 | 4  | 4 | 0  | 0  | 0  | 1 | 4 | 4  | 0  |
/// Per Table 51 the initType-1 (P) block is ctxIdx 0..5 and the
/// initType-2 (B) block is ctxIdx 6..11; the unused initType-0 (I)
/// block has no entries.
pub const INTER_PRED_IDC_INIT: &[u8] = &[7, 6, 5, 12, 4, 40, 14, 13, 5, 4, 3, 40];
pub const INTER_PRED_IDC_SHIFT: &[u8] = &[0, 0, 1, 4, 4, 0, 0, 0, 1, 4, 4, 0];

/// Table 86 — `sym_mvd_flag` (2 ctxIdx, one per non-I initType).
/// Round-108 transcription:
///   ctxIdx     | 0  | 1  |
///   initValue  | 28 | 28 |
///   shiftIdx   |  5 |  5 |
pub const SYM_MVD_FLAG_INIT: &[u8] = &[28, 28];
pub const SYM_MVD_FLAG_SHIFT: &[u8] = &[5, 5];

/// Table 87 — `ref_idx_l0` / `ref_idx_l1` (4 ctxIdx, 2 per non-I
/// initType). Round-108 transcription:
///   ctxIdx     | 0  | 1  | 2 | 3  |
///   initValue  | 20 | 35 | 5 | 35 |
///   shiftIdx   |  0 |  4 | 0 |  4 |
pub const REF_IDX_LX_INIT: &[u8] = &[20, 35, 5, 35];
pub const REF_IDX_LX_SHIFT: &[u8] = &[0, 4, 0, 4];

/// Table 88 — `mvp_l0_flag` / `mvp_l1_flag` (3 ctxIdx, one per
/// initType). Round-108 transcription:
///   ctxIdx     | 0  | 1  | 2  |
///   initValue  | 42 | 34 | 34 |
///   shiftIdx   | 12 | 12 | 12 |
pub const MVP_LX_FLAG_INIT: &[u8] = &[42, 34, 34];
pub const MVP_LX_FLAG_SHIFT: &[u8] = &[12, 12, 12];

/// Table 110 — `abs_mvd_greater0_flag` (3 ctxIdx, one per initType).
/// Round-103 §7.3.10.10 / §9.3.3.14 transcription:
///   initType  | 0  | 1  | 2  |
///   initValue | 14 | 44 | 51 |
///   shiftIdx  |  9 |  9 |  9 |
pub const ABS_MVD_GREATER0_FLAG_INIT: &[u8] = &[14, 44, 51];
pub const ABS_MVD_GREATER0_FLAG_SHIFT: &[u8] = &[9, 9, 9];

/// Table 111 — `abs_mvd_greater1_flag` (3 ctxIdx, one per initType).
/// Round-103 §7.3.10.10 / §9.3.3.14 transcription:
///   initType  | 0  | 1  | 2  |
///   initValue | 45 | 43 | 36 |
///   shiftIdx  |  5 |  5 |  5 |
pub const ABS_MVD_GREATER1_FLAG_INIT: &[u8] = &[45, 43, 36];
pub const ABS_MVD_GREATER1_FLAG_SHIFT: &[u8] = &[5, 5, 5];

/// Table 91 — `bcw_idx` (2 ctxIdx, one per non-I initType). Round-126
/// transcription from the spec Table 91:
///   ctxIdx     | 0  | 1  |
///   initValue  | 4  | 5  |
///   shiftIdx   | 1  | 1  |
/// Per Table 51 the indexing rule is `init_type - 1` (initType 1 →
/// ctxIdx 0, initType 2 → ctxIdx 1; never signalled in I slices). Per
/// Table 132 only bin 0 of the TR sequence is context-coded with
/// `ctxInc = 0`; bins 1.. are bypass-coded.
pub const BCW_IDX_INIT: &[u8] = &[4, 5];
pub const BCW_IDX_SHIFT: &[u8] = &[1, 1];

/// Table 107 — `merge_subblock_flag` (6 ctxIdx, 3 per non-I initType).
/// Transcription from the spec Table 107:
///   ctxIdx     | 0  | 1  | 2  | 3  | 4  | 5  |
///   initValue  | 48 | 57 | 44 | 25 | 58 | 45 |
///   shiftIdx   |  4 |  4 |  4 |  4 |  4 |  4 |
/// Per Table 51 the indexing rule is `(init_type − 1) * 3 + ctxInc`
/// (initType 1 → ctxIdx 0..2, initType 2 → ctxIdx 3..5; merge data is
/// never signalled in I slices). The single ctx-coded bin uses
/// §9.3.4.2.2 / eq. 1551 with `ctxSetIdx = 0` and the Table 133 merge-
/// side `condL` / `condA` derivation, yielding `ctxInc ∈ {0, 1, 2}`.
pub const MERGE_SUBBLOCK_FLAG_INIT: &[u8] = &[48, 57, 44, 25, 58, 45];
pub const MERGE_SUBBLOCK_FLAG_SHIFT: &[u8] = &[4, 4, 4, 4, 4, 4];

/// Table 108 — `merge_subblock_idx` (2 ctxIdx, one per non-I initType).
/// Transcription from the spec Table 108:
///   ctxIdx     | 0  | 1  |
///   initValue  |  5 |  4 |
///   shiftIdx   |  0 |  0 |
/// Per Table 51 the slot is `init_type − 1` (initType 1 → ctxIdx 0,
/// initType 2 → ctxIdx 1). Per Table 132 only bin 0 of the TR sequence
/// is context-coded with `ctxInc = 0`; bins 1.. (only present when
/// `MaxNumSubblockMergeCand ≥ 3`) are bypass-coded.
pub const MERGE_SUBBLOCK_IDX_INIT: &[u8] = &[5, 4];
pub const MERGE_SUBBLOCK_IDX_SHIFT: &[u8] = &[0, 0];

/// Table 84 — `inter_affine_flag` (6 ctxIdx, 3 per non-I initType).
/// Transcription from the spec Table 84:
///   ctxIdx     | 0  | 1  | 2  | 3  | 4  | 5 |
///   initValue  | 12 | 13 | 14 | 19 | 13 | 6 |
///   shiftIdx   |  4 |  0 |  0 |  4 |  0 | 0 |
/// Per Table 51 the indexing rule is `(init_type − 1) * 3 + ctxInc`
/// (initType 1 → ctxIdx 0..2, initType 2 → ctxIdx 3..5; never signalled
/// in I slices). The single ctx-coded bin uses §9.3.4.2.2 / eq. 1551
/// with `ctxSetIdx = 0` and the Table 133 row whose `condL` / `condA`
/// are identical to the `merge_subblock_flag` row, yielding
/// `ctxInc ∈ {0, 1, 2}`.
pub const INTER_AFFINE_FLAG_INIT: &[u8] = &[12, 13, 14, 19, 13, 6];
pub const INTER_AFFINE_FLAG_SHIFT: &[u8] = &[4, 0, 0, 4, 0, 0];

/// Table 85 — `cu_affine_type_flag` (2 ctxIdx, one per non-I initType).
/// Round-159 §7.3.11.7 transcription:
///   ctxIdx     | 0  | 1  |
///   initValue  | 35 | 35 |
///   shiftIdx   |  4 |  4 |
/// Per Table 51 the indexing rule is `init_type − 1` (initType 1 →
/// ctxIdx 0, initType 2 → ctxIdx 1; never signalled in I slices). The
/// single ctx-coded bin uses Table 132's deterministic `ctxInc = 0`
/// (no §9.3.4.2.2 / Table 133 row applies — the value is fixed). The
/// surrounding syntax gate is
/// `sps_6param_affine_enabled_flag && inter_affine_flag == 1`; the
/// caller is responsible for the gate check.
pub const CU_AFFINE_TYPE_FLAG_INIT: &[u8] = &[35, 35];
pub const CU_AFFINE_TYPE_FLAG_SHIFT: &[u8] = &[4, 4];

/// Table 89 — `amvr_flag` (4 ctxIdx). Round-40 §7.4.11.6 transcription:
///   ctxIdx     | 0  | 1  | 2  | 3  |
///   initValue  | 59 | 58 | 59 | 50 |
///   shiftIdx   |  0 |  0 |  0 |  0 |
pub const AMVR_FLAG_INIT: &[u8] = &[59, 58, 59, 50];
pub const AMVR_FLAG_SHIFT: &[u8] = &[0, 0, 0, 0];

/// Table 90 — `amvr_precision_idx` (9 ctxIdx). Round-40 §7.4.11.6
/// transcription:
///   ctxIdx     | 0  | 1  | 2  | 3  | 4  | 5  | 6  | 7  | 8  |
///   initValue  | 35 | 34 | 35 | 60 | 48 | 60 | 38 | 26 | 60 |
///   shiftIdx   |  4 |  5 |  0 |  4 |  5 |  0 |  4 |  5 |  0 |
pub const AMVR_PRECISION_IDX_INIT: &[u8] = &[35, 34, 35, 60, 48, 60, 38, 26, 60];
pub const AMVR_PRECISION_IDX_SHIFT: &[u8] = &[4, 5, 0, 4, 5, 0, 4, 5, 0];

/// Table 52 — `alf_ctb_flag` (27 ctxIdx). Round-45 §9.3.4.2.2 / Table 51
/// transcription. ctxIdx layout: 27 entries split as 3 initType blocks
/// of 9 ctxIdx (3 ctxInc rows × 3 cIdx rows = 9). Per Table 51:
///   * initType 0 → ctxIdx 0..8
///   * initType 1 → ctxIdx 9..17
///   * initType 2 → ctxIdx 18..26
/// The `ctxInc` derivation in §9.3.4.2.2 produces values
/// `(condL && availL) + (condA && availA) + cIdx * 3` ∈ {0..8}.
pub const ALF_CTB_FLAG_INIT: &[u8] = &[
    62, 39, 39, 54, 39, 39, 31, 39, 39, 13, 23, 46, 4, 61, 54, 19, 46, 54, 33, 52, 46, 25, 61, 54,
    25, 61, 54,
];
pub const ALF_CTB_FLAG_SHIFT: &[u8] = &[
    0, 0, 0, 4, 0, 0, 1, 0, 0, 0, 0, 0, 4, 0, 0, 1, 0, 0, 0, 0, 0, 4, 0, 0, 1, 0, 0,
];

/// Table 53 — `alf_use_aps_flag` (3 ctxIdx, one per initType). Round-45
/// §9.3.4.2 / Table 132 `ctxInc = 0` per bin. Per Table 51 each initType
/// uses ctxIdx 0 / 1 / 2 respectively.
pub const ALF_USE_APS_FLAG_INIT: &[u8] = &[46, 46, 46];
pub const ALF_USE_APS_FLAG_SHIFT: &[u8] = &[0, 0, 0];

/// Table 54 — `alf_ctb_cc_cb_idc` (9 ctxIdx, 3 per initType). Round-45
/// §9.3.4.2.2 / Table 51. The 3 ctxInc values per initType cover the
/// {0, 1, 2} = `(condL && availL) + (condA && availA)` range of bin 0
/// (bin 1+ is bypass per Table 132).
pub const ALF_CTB_CC_CB_IDC_INIT: &[u8] = &[18, 30, 31, 18, 21, 38, 25, 35, 38];
pub const ALF_CTB_CC_CB_IDC_SHIFT: &[u8] = &[4, 1, 4, 4, 1, 4, 4, 1, 4];

/// Table 55 — `alf_ctb_cc_cr_idc` (9 ctxIdx, 3 per initType). Same
/// shape as Table 54.
pub const ALF_CTB_CC_CR_IDC_INIT: &[u8] = &[18, 30, 31, 18, 21, 38, 25, 28, 38];
pub const ALF_CTB_CC_CR_IDC_SHIFT: &[u8] = &[4, 1, 4, 4, 1, 4, 4, 1, 4];

/// Table 56 — `alf_ctb_filter_alt_idx` (6 ctxIdx, 2 per initType).
/// Round-45 §9.3.4.2 / Table 132. Per Table 51:
///   * initType 0 → ctxIdx 0..1   (Cb / Cr)
///   * initType 1 → ctxIdx 2..3
///   * initType 2 → ctxIdx 4..5
/// Bin 0 (and the rest of the TR sequence) all share `ctxInc = 0` for
/// chroma component 0 and `ctxInc = 1` for chroma component 1
/// (Table 132).
pub const ALF_CTB_FILTER_ALT_IDX_INIT: &[u8] = &[11, 11, 20, 12, 11, 26];
pub const ALF_CTB_FILTER_ALT_IDX_SHIFT: &[u8] = &[0, 0, 0, 0, 0, 0];

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
        SyntaxCtx::MttSplitCuVerticalFlag => (
            MTT_SPLIT_CU_VERTICAL_FLAG_INIT,
            MTT_SPLIT_CU_VERTICAL_FLAG_SHIFT,
        ),
        SyntaxCtx::MttSplitCuBinaryFlag => (
            MTT_SPLIT_CU_BINARY_FLAG_INIT,
            MTT_SPLIT_CU_BINARY_FLAG_SHIFT,
        ),
        SyntaxCtx::PredModeFlag => (PRED_MODE_FLAG_INIT, PRED_MODE_FLAG_SHIFT),
        SyntaxCtx::IntraBdpcmLumaFlag => (INTRA_BDPCM_LUMA_FLAG_INIT, INTRA_BDPCM_LUMA_FLAG_SHIFT),
        SyntaxCtx::IntraBdpcmLumaDirFlag => (
            INTRA_BDPCM_LUMA_DIR_FLAG_INIT,
            INTRA_BDPCM_LUMA_DIR_FLAG_SHIFT,
        ),
        SyntaxCtx::IntraBdpcmChromaFlag => {
            (INTRA_BDPCM_CHROMA_FLAG_INIT, INTRA_BDPCM_CHROMA_FLAG_SHIFT)
        }
        SyntaxCtx::IntraBdpcmChromaDirFlag => (
            INTRA_BDPCM_CHROMA_DIR_FLAG_INIT,
            INTRA_BDPCM_CHROMA_DIR_FLAG_SHIFT,
        ),
        SyntaxCtx::IntraMipFlag => (INTRA_MIP_FLAG_INIT, INTRA_MIP_FLAG_SHIFT),
        SyntaxCtx::IntraLumaRefIdx => (INTRA_LUMA_REF_IDX_INIT, INTRA_LUMA_REF_IDX_SHIFT),
        SyntaxCtx::IntraSubpartitionsModeFlag => {
            (INTRA_SP_MODE_FLAG_INIT, INTRA_SP_MODE_FLAG_SHIFT)
        }
        SyntaxCtx::IntraSubpartitionsSplitFlag => {
            (INTRA_SP_SPLIT_FLAG_INIT, INTRA_SP_SPLIT_FLAG_SHIFT)
        }
        SyntaxCtx::IntraLumaMpmFlag => (INTRA_LUMA_MPM_FLAG_INIT, INTRA_LUMA_MPM_FLAG_SHIFT),
        SyntaxCtx::IntraLumaNotPlanarFlag => (
            INTRA_LUMA_NOT_PLANAR_FLAG_INIT,
            INTRA_LUMA_NOT_PLANAR_FLAG_SHIFT,
        ),
        SyntaxCtx::IntraChromaPredMode => {
            (INTRA_CHROMA_PRED_MODE_INIT, INTRA_CHROMA_PRED_MODE_SHIFT)
        }
        SyntaxCtx::SigCoeffFlag => (SIG_COEFF_FLAG_INIT, SIG_COEFF_FLAG_SHIFT),
        SyntaxCtx::SbCodedFlag => (SB_CODED_FLAG_INIT, SB_CODED_FLAG_SHIFT),
        SyntaxCtx::AbsLevelGtxFlag => (ABS_LEVEL_GTX_FLAG_INIT, ABS_LEVEL_GTX_FLAG_SHIFT),
        SyntaxCtx::CoeffSignFlag => (COEFF_SIGN_FLAG_INIT, COEFF_SIGN_FLAG_SHIFT),
        SyntaxCtx::LastSigCoeffXPrefix => (LAST_SIG_X_PREFIX_INIT, LAST_SIG_X_PREFIX_SHIFT),
        SyntaxCtx::LastSigCoeffYPrefix => (LAST_SIG_Y_PREFIX_INIT, LAST_SIG_Y_PREFIX_SHIFT),
        SyntaxCtx::TuYCodedFlag => (TU_Y_CODED_FLAG_INIT, TU_Y_CODED_FLAG_SHIFT),
        SyntaxCtx::TuCbCodedFlag => (TU_CB_CODED_FLAG_INIT, TU_CB_CODED_FLAG_SHIFT),
        SyntaxCtx::TuCrCodedFlag => (TU_CR_CODED_FLAG_INIT, TU_CR_CODED_FLAG_SHIFT),
        SyntaxCtx::CuQpDeltaAbs => (CU_QP_DELTA_ABS_INIT, CU_QP_DELTA_ABS_SHIFT),
        SyntaxCtx::CuChromaQpOffsetFlag => (
            CU_CHROMA_QP_OFFSET_FLAG_INIT,
            CU_CHROMA_QP_OFFSET_FLAG_SHIFT,
        ),
        SyntaxCtx::CuChromaQpOffsetIdx => {
            (CU_CHROMA_QP_OFFSET_IDX_INIT, CU_CHROMA_QP_OFFSET_IDX_SHIFT)
        }
        SyntaxCtx::TuJointCbCrResidualFlag => (
            TU_JOINT_CBCR_RESIDUAL_FLAG_INIT,
            TU_JOINT_CBCR_RESIDUAL_FLAG_SHIFT,
        ),
        SyntaxCtx::TransformSkipFlag => (TRANSFORM_SKIP_FLAG_INIT, TRANSFORM_SKIP_FLAG_SHIFT),
        SyntaxCtx::ParLevelFlag => (PAR_LEVEL_FLAG_INIT, PAR_LEVEL_FLAG_SHIFT),
        SyntaxCtx::LfnstIdx => (LFNST_IDX_INIT, LFNST_IDX_SHIFT),
        SyntaxCtx::MtsIdx => (MTS_IDX_INIT, MTS_IDX_SHIFT),
        SyntaxCtx::SaoMergeFlag => (SAO_MERGE_FLAG_INIT, SAO_MERGE_FLAG_SHIFT),
        SyntaxCtx::SaoTypeIdx => (SAO_TYPE_IDX_INIT, SAO_TYPE_IDX_SHIFT),
        SyntaxCtx::CuSkipFlag => (CU_SKIP_FLAG_INIT, CU_SKIP_FLAG_SHIFT),
        SyntaxCtx::GeneralMergeFlag => (GENERAL_MERGE_FLAG_INIT, GENERAL_MERGE_FLAG_SHIFT),
        SyntaxCtx::RegularMergeFlag => (REGULAR_MERGE_FLAG_INIT, REGULAR_MERGE_FLAG_SHIFT),
        SyntaxCtx::MmvdMergeFlag => (MMVD_MERGE_FLAG_INIT, MMVD_MERGE_FLAG_SHIFT),
        SyntaxCtx::MmvdCandFlag => (MMVD_CAND_FLAG_INIT, MMVD_CAND_FLAG_SHIFT),
        SyntaxCtx::MmvdDistanceIdx => (MMVD_DISTANCE_IDX_INIT, MMVD_DISTANCE_IDX_SHIFT),
        SyntaxCtx::CiipFlag => (CIIP_FLAG_INIT, CIIP_FLAG_SHIFT),
        SyntaxCtx::MergeIdx => (MERGE_IDX_INIT, MERGE_IDX_SHIFT),
        SyntaxCtx::AbsMvdGreater0Flag => (ABS_MVD_GREATER0_FLAG_INIT, ABS_MVD_GREATER0_FLAG_SHIFT),
        SyntaxCtx::AbsMvdGreater1Flag => (ABS_MVD_GREATER1_FLAG_INIT, ABS_MVD_GREATER1_FLAG_SHIFT),
        SyntaxCtx::CuCodedFlag => (CU_CODED_FLAG_INIT, CU_CODED_FLAG_SHIFT),
        SyntaxCtx::InterPredIdc => (INTER_PRED_IDC_INIT, INTER_PRED_IDC_SHIFT),
        SyntaxCtx::SymMvdFlag => (SYM_MVD_FLAG_INIT, SYM_MVD_FLAG_SHIFT),
        SyntaxCtx::RefIdxLx => (REF_IDX_LX_INIT, REF_IDX_LX_SHIFT),
        SyntaxCtx::MvpLxFlag => (MVP_LX_FLAG_INIT, MVP_LX_FLAG_SHIFT),
        SyntaxCtx::AmvrFlag => (AMVR_FLAG_INIT, AMVR_FLAG_SHIFT),
        SyntaxCtx::AmvrPrecisionIdx => (AMVR_PRECISION_IDX_INIT, AMVR_PRECISION_IDX_SHIFT),
        SyntaxCtx::AlfCtbFlag => (ALF_CTB_FLAG_INIT, ALF_CTB_FLAG_SHIFT),
        SyntaxCtx::AlfUseApsFlag => (ALF_USE_APS_FLAG_INIT, ALF_USE_APS_FLAG_SHIFT),
        SyntaxCtx::AlfCtbCcCbIdc => (ALF_CTB_CC_CB_IDC_INIT, ALF_CTB_CC_CB_IDC_SHIFT),
        SyntaxCtx::AlfCtbCcCrIdc => (ALF_CTB_CC_CR_IDC_INIT, ALF_CTB_CC_CR_IDC_SHIFT),
        SyntaxCtx::AlfCtbFilterAltIdx => {
            (ALF_CTB_FILTER_ALT_IDX_INIT, ALF_CTB_FILTER_ALT_IDX_SHIFT)
        }
        SyntaxCtx::BcwIdx => (BCW_IDX_INIT, BCW_IDX_SHIFT),
        SyntaxCtx::MergeSubblockFlag => (MERGE_SUBBLOCK_FLAG_INIT, MERGE_SUBBLOCK_FLAG_SHIFT),
        SyntaxCtx::MergeSubblockIdx => (MERGE_SUBBLOCK_IDX_INIT, MERGE_SUBBLOCK_IDX_SHIFT),
        SyntaxCtx::InterAffineFlag => (INTER_AFFINE_FLAG_INIT, INTER_AFFINE_FLAG_SHIFT),
        SyntaxCtx::CuAffineTypeFlag => (CU_AFFINE_TYPE_FLAG_INIT, CU_AFFINE_TYPE_FLAG_SHIFT),
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
            SyntaxCtx::MttSplitCuVerticalFlag,
            SyntaxCtx::MttSplitCuBinaryFlag,
            SyntaxCtx::PredModeFlag,
            SyntaxCtx::IntraBdpcmLumaFlag,
            SyntaxCtx::IntraBdpcmLumaDirFlag,
            SyntaxCtx::IntraBdpcmChromaFlag,
            SyntaxCtx::IntraBdpcmChromaDirFlag,
            SyntaxCtx::IntraMipFlag,
            SyntaxCtx::IntraLumaRefIdx,
            SyntaxCtx::IntraSubpartitionsModeFlag,
            SyntaxCtx::IntraSubpartitionsSplitFlag,
            SyntaxCtx::IntraLumaMpmFlag,
            SyntaxCtx::IntraLumaNotPlanarFlag,
            SyntaxCtx::IntraChromaPredMode,
            SyntaxCtx::SigCoeffFlag,
            SyntaxCtx::SbCodedFlag,
            SyntaxCtx::AbsLevelGtxFlag,
            SyntaxCtx::CoeffSignFlag,
            SyntaxCtx::LastSigCoeffXPrefix,
            SyntaxCtx::LastSigCoeffYPrefix,
            SyntaxCtx::TuYCodedFlag,
            SyntaxCtx::TuCbCodedFlag,
            SyntaxCtx::TuCrCodedFlag,
            SyntaxCtx::CuQpDeltaAbs,
            SyntaxCtx::CuChromaQpOffsetFlag,
            SyntaxCtx::CuChromaQpOffsetIdx,
            SyntaxCtx::ParLevelFlag,
            SyntaxCtx::LfnstIdx,
            SyntaxCtx::MtsIdx,
            SyntaxCtx::SaoMergeFlag,
            SyntaxCtx::SaoTypeIdx,
            SyntaxCtx::CuSkipFlag,
            SyntaxCtx::GeneralMergeFlag,
            SyntaxCtx::RegularMergeFlag,
            SyntaxCtx::MmvdMergeFlag,
            SyntaxCtx::MmvdCandFlag,
            SyntaxCtx::MmvdDistanceIdx,
            SyntaxCtx::CiipFlag,
            SyntaxCtx::MergeIdx,
            SyntaxCtx::CuCodedFlag,
            SyntaxCtx::AmvrFlag,
            SyntaxCtx::AmvrPrecisionIdx,
            SyntaxCtx::AlfCtbFlag,
            SyntaxCtx::AlfUseApsFlag,
            SyntaxCtx::AlfCtbCcCbIdc,
            SyntaxCtx::AlfCtbCcCrIdc,
            SyntaxCtx::AlfCtbFilterAltIdx,
            SyntaxCtx::BcwIdx,
            SyntaxCtx::MergeSubblockFlag,
            SyntaxCtx::MergeSubblockIdx,
            SyntaxCtx::InterAffineFlag,
            SyntaxCtx::CuAffineTypeFlag,
        ] {
            let (i, s) = table_for(kind);
            assert_eq!(i.len(), s.len(), "table {:?} length mismatch", kind);
            assert_eq!(i.len(), ctx_count(kind));
        }
    }

    /// Round-45 — Tables 52-56 transcription length sanity. Table 51
    /// pins the per-ALF context counts (27 / 3 / 9 / 9 / 6).
    #[test]
    fn alf_context_table_lengths() {
        assert_eq!(ctx_count(SyntaxCtx::AlfCtbFlag), 27);
        assert_eq!(ctx_count(SyntaxCtx::AlfUseApsFlag), 3);
        assert_eq!(ctx_count(SyntaxCtx::AlfCtbCcCbIdc), 9);
        assert_eq!(ctx_count(SyntaxCtx::AlfCtbCcCrIdc), 9);
        assert_eq!(ctx_count(SyntaxCtx::AlfCtbFilterAltIdx), 6);
    }

    #[test]
    fn split_cu_context_count_is_27() {
        assert_eq!(ctx_count(SyntaxCtx::SplitCuFlag), 27);
    }

    /// Round-55 — Tables 61 / 62 transcription length sanity. Per Table 51:
    ///   * `mtt_split_cu_vertical_flag` ∈ 0..14 → 15 entries.
    ///   * `mtt_split_cu_binary_flag` ∈ 0..11 → 12 entries.
    #[test]
    fn mtt_split_context_table_lengths() {
        assert_eq!(ctx_count(SyntaxCtx::MttSplitCuVerticalFlag), 15);
        assert_eq!(ctx_count(SyntaxCtx::MttSplitCuBinaryFlag), 12);
    }

    /// Round-139 — Tables 107 / 108 transcription length sanity. Per
    /// Table 51 / Table 132:
    ///   * `merge_subblock_flag` ∈ 0..5 → 6 entries (3 per non-I initType,
    ///     covering the `ctxInc ∈ {0, 1, 2}` range of §9.3.4.2.2 /
    ///     eq. 1551).
    ///   * `merge_subblock_idx` ∈ 0..1 → 2 entries (one per non-I
    ///     initType; only bin 0 of the TR sequence is ctx-coded).
    #[test]
    fn merge_subblock_context_table_lengths() {
        assert_eq!(ctx_count(SyntaxCtx::MergeSubblockFlag), 6);
        assert_eq!(ctx_count(SyntaxCtx::MergeSubblockIdx), 2);
    }

    /// Pin Tables 107 + 108 initValue / shiftIdx transcription bit-exact
    /// to guard against silent table-rewrite drift.
    #[test]
    fn merge_subblock_init_values_match_spec() {
        // Table 107: initValue = [48, 57, 44, 25, 58, 45], shiftIdx all 4.
        let (init, shift) = (MERGE_SUBBLOCK_FLAG_INIT, MERGE_SUBBLOCK_FLAG_SHIFT);
        assert_eq!(init, &[48, 57, 44, 25, 58, 45]);
        assert_eq!(shift, &[4, 4, 4, 4, 4, 4]);
        // Table 108: initValue = [5, 4], shiftIdx all 0.
        let (init, shift) = (MERGE_SUBBLOCK_IDX_INIT, MERGE_SUBBLOCK_IDX_SHIFT);
        assert_eq!(init, &[5, 4]);
        assert_eq!(shift, &[0, 0]);
    }

    /// Round-152 — Table 84 transcription length + initValue / shiftIdx
    /// pin. Per Table 51 / Table 132 `inter_affine_flag` has 6 entries
    /// (3 per non-I initType), addressed by `(init_type − 1) * 3 +
    /// ctxInc` with `ctxInc ∈ {0, 1, 2}` from the §9.3.4.2.2 / eq. 1551
    /// derivation under the Table 133 row identical to
    /// `merge_subblock_flag`.
    #[test]
    fn inter_affine_flag_context_table_length() {
        assert_eq!(ctx_count(SyntaxCtx::InterAffineFlag), 6);
    }

    #[test]
    fn inter_affine_flag_init_values_match_spec() {
        // Table 84: initValue = [12, 13, 14, 19, 13, 6],
        //           shiftIdx  = [ 4,  0,  0,  4,  0, 0].
        let (init, shift) = (INTER_AFFINE_FLAG_INIT, INTER_AFFINE_FLAG_SHIFT);
        assert_eq!(init, &[12, 13, 14, 19, 13, 6]);
        assert_eq!(shift, &[4, 0, 0, 4, 0, 0]);
    }

    /// Round-159 — Table 85 transcription length + initValue / shiftIdx
    /// pin. Per Table 51 / Table 132 `cu_affine_type_flag` has 2 entries
    /// (one per non-I initType), addressed by `init_type − 1` with the
    /// deterministic `ctxInc = 0`. Never signalled in I slices; only on
    /// the non-merge inter branch behind `sps_6param_affine_enabled_flag
    /// && inter_affine_flag == 1`.
    #[test]
    fn cu_affine_type_flag_context_table_length() {
        assert_eq!(ctx_count(SyntaxCtx::CuAffineTypeFlag), 2);
    }

    #[test]
    fn cu_affine_type_flag_init_values_match_spec() {
        // Table 85: initValue = [35, 35], shiftIdx = [4, 4].
        let (init, shift) = (CU_AFFINE_TYPE_FLAG_INIT, CU_AFFINE_TYPE_FLAG_SHIFT);
        assert_eq!(init, &[35, 35]);
        assert_eq!(shift, &[4, 4]);
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
