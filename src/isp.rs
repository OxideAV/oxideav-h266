//! Intra Sub-Partitions (ISP) — geometry helpers per §8.4.5.1.
//!
//! ISP splits an intra-predicted luma coding unit into either two or
//! four equal-sized transform-block subpartitions. Each subpartition is
//! decoded sequentially (predict → dequant → IDCT → reconstruct) so a
//! later subpartition can reference the freshly-reconstructed samples
//! of the earlier ones along the split direction.
//!
//! # Spec coverage
//!
//! * §7.4.12.2 — `intra_subpartitions_mode_flag`,
//!   `intra_subpartitions_split_flag` semantics. Table 13 maps the
//!   split flag onto `IntraSubPartitionsSplitType` ∈
//!   {`ISP_NO_SPLIT`, `ISP_HOR_SPLIT`, `ISP_VER_SPLIT`}.
//! * §7.4.12.2 — `NumIntraSubPartitions` derivation:
//!     - 1 if `IntraSubPartitionsSplitType == ISP_NO_SPLIT`,
//!     - 2 if `(cbWidth, cbHeight)` is `(4, 8)` or `(8, 4)`,
//!     - 4 otherwise.
//! * §7.3.11.9 — `transform_tree()` walker emits one
//!   `transform_unit()` call per partition (offsets `trafoWidth * partIdx`
//!   for vertical splits, `trafoHeight * partIdx` for horizontal).
//! * §8.4.5.1 — eqs. 251 – 260 derive the per-partition `nW`, `nH`,
//!   `nPbW = max(4, nW)`, `pbFactor`, `xPartIdx`, `yPartIdx`,
//!   `xPartPbIdx`. The intra-prediction sample process is invoked
//!   only when `xPartPbIdx == 0`; otherwise the previous prediction is
//!   reused (the prediction window covers `pbFactor` consecutive
//!   subpartitions when `nW < 4`).
//!
//! Spec PDF: ITU-T H.266 (V4) (01/2026), pp. 171 – 172 (syntax),
//! pp. 204 – 205 (eqs. 251 – 260).

use crate::leaf_cu::IspSplitType;

/// `NumIntraSubPartitions` per §7.4.12.2.
///
/// Returns 1 for `IspSplitType::NoSplit`, 2 for the two "edge" CU
/// shapes (4×8 and 8×4), and 4 otherwise.
pub fn num_intra_subpartitions(split: IspSplitType, cb_width: u32, cb_height: u32) -> u32 {
    match split {
        IspSplitType::NoSplit => 1,
        _ => {
            if (cb_width == 4 && cb_height == 8) || (cb_width == 8 && cb_height == 4) {
                2
            } else {
                4
            }
        }
    }
}

/// Geometry for a single ISP subpartition, as derived by §8.4.5.1
/// eqs. 251 – 260.
///
/// Coordinates are CB-relative (i.e. the CU's top-left corner is the
/// origin). Add the CU's `(x0, y0)` to obtain picture-absolute
/// coordinates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IspPartition {
    /// `i` — partition index in walk order.
    pub idx: u32,
    /// `xPartIdx` — `i * xPartInc`. 0 for horizontal split.
    pub x_part_idx: u32,
    /// `yPartIdx` — `i * yPartInc`. 0 for vertical split.
    pub y_part_idx: u32,
    /// `nW` — sub-TB width in luma samples (eq. 251).
    pub n_w: u32,
    /// `nH` — sub-TB height in luma samples (eq. 252).
    pub n_h: u32,
    /// `nPbW = max(4, nW)` — prediction-block width (eq. 255).
    /// For vertical splits with `nW ∈ {1, 2}` this exceeds `nW`,
    /// meaning a single prediction window covers `pbFactor`
    /// consecutive subpartitions.
    pub n_pb_w: u32,
    /// `pbFactor = nPbW / nW` (eq. 256). Always 1 for horizontal
    /// splits and for vertical splits with `nW >= 4`.
    pub pb_factor: u32,
    /// `xPartPbIdx = xPartIdx % pbFactor` (eq. 260). The intra
    /// sample-prediction process is invoked only when this is 0;
    /// otherwise the previous prediction is reused.
    pub x_part_pb_idx: u32,
    /// CB-relative top-left luma sample of this subpartition.
    /// Equal to `(x_part_idx * n_w, y_part_idx * n_h)`.
    pub x_offset: u32,
    /// CB-relative top-left luma sample of this subpartition.
    pub y_offset: u32,
}

/// Walk an ISP-split CU's subpartitions in spec order. Returns an
/// empty vector when the split type is `NoSplit`.
///
/// `cb_width` / `cb_height` are the luma CB dimensions. `cIdx` is
/// fixed to 0 (luma) here — chroma subpartitioning is not invoked
/// in single-tree mode (per eqs. 251 – 254 the splitting only fires
/// when `cIdx == 0`).
pub fn iter_isp_partitions(
    split: IspSplitType,
    cb_width: u32,
    cb_height: u32,
) -> Vec<IspPartition> {
    let num = num_intra_subpartitions(split, cb_width, cb_height);
    if num == 1 || matches!(split, IspSplitType::NoSplit) {
        return Vec::new();
    }
    let (n_w, n_h, x_inc, y_inc) = match split {
        IspSplitType::HorSplit => (cb_width, cb_height / num, 0u32, 1u32),
        IspSplitType::VerSplit => (cb_width / num, cb_height, 1u32, 0u32),
        IspSplitType::NoSplit => unreachable!(),
    };
    let n_pb_w = n_w.max(4);
    let pb_factor = n_pb_w / n_w;
    let mut out = Vec::with_capacity(num as usize);
    for i in 0..num {
        let x_part_idx = i * x_inc;
        let y_part_idx = i * y_inc;
        out.push(IspPartition {
            idx: i,
            x_part_idx,
            y_part_idx,
            n_w,
            n_h,
            n_pb_w,
            pb_factor,
            x_part_pb_idx: x_part_idx % pb_factor,
            x_offset: x_part_idx * n_w,
            y_offset: y_part_idx * n_h,
        });
    }
    out
}

/// `refW` / `refH` for the ISP-aware reference-sample fetch
/// (§8.4.5.2.1, eqs. 313 – 316).
///
/// * For ISP_NO_SPLIT (or chroma): `refW = nTbW * 2`, `refH = nTbH * 2`.
/// * For ISP-split luma: `refW = nCbW + nTbW`, `refH = nCbH + nTbH`.
///
/// The current decoder does not yet consume the extended reference
/// width/height for any specific predictor (the cardinal-angular
/// helpers all read at most `nTbW + 1` / `nTbH + 1` samples), so this
/// helper is exposed for callers that want the spec value verbatim.
pub fn ref_dimensions(
    split: IspSplitType,
    c_idx: u32,
    n_cb_w: u32,
    n_cb_h: u32,
    n_tb_w: u32,
    n_tb_h: u32,
) -> (u32, u32) {
    if matches!(split, IspSplitType::NoSplit) || c_idx != 0 {
        (n_tb_w * 2, n_tb_h * 2)
    } else {
        (n_cb_w + n_tb_w, n_cb_h + n_tb_h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `NumIntraSubPartitions` matches the spec table verbatim:
    /// NoSplit → 1, (4,8)/(8,4) → 2, everything else → 4.
    #[test]
    fn num_subpartitions_table() {
        assert_eq!(num_intra_subpartitions(IspSplitType::NoSplit, 16, 16), 1);
        assert_eq!(num_intra_subpartitions(IspSplitType::HorSplit, 4, 8), 2);
        assert_eq!(num_intra_subpartitions(IspSplitType::VerSplit, 4, 8), 2);
        assert_eq!(num_intra_subpartitions(IspSplitType::HorSplit, 8, 4), 2);
        assert_eq!(num_intra_subpartitions(IspSplitType::VerSplit, 8, 4), 2);
        assert_eq!(num_intra_subpartitions(IspSplitType::HorSplit, 8, 8), 4);
        assert_eq!(num_intra_subpartitions(IspSplitType::VerSplit, 16, 16), 4);
        assert_eq!(num_intra_subpartitions(IspSplitType::HorSplit, 32, 64), 4);
    }

    /// 16x16 horizontal split → 4 subpartitions of 16x4 stacked
    /// vertically. `xPartPbIdx` is always 0 because `nW = 16 >= 4`.
    #[test]
    fn hor_split_16x16_emits_four_16x4_partitions() {
        let parts = iter_isp_partitions(IspSplitType::HorSplit, 16, 16);
        assert_eq!(parts.len(), 4);
        for (i, p) in parts.iter().enumerate() {
            assert_eq!(p.n_w, 16);
            assert_eq!(p.n_h, 4);
            assert_eq!(p.n_pb_w, 16);
            assert_eq!(p.pb_factor, 1);
            assert_eq!(p.x_offset, 0);
            assert_eq!(p.y_offset, (i as u32) * 4);
            assert_eq!(p.x_part_pb_idx, 0);
        }
    }

    /// 16x16 vertical split → 4 subpartitions of 4x16 side by side.
    /// `nW = 4 == nPbW` so `pbFactor == 1` and prediction runs once
    /// per partition.
    #[test]
    fn ver_split_16x16_emits_four_4x16_partitions() {
        let parts = iter_isp_partitions(IspSplitType::VerSplit, 16, 16);
        assert_eq!(parts.len(), 4);
        for (i, p) in parts.iter().enumerate() {
            assert_eq!(p.n_w, 4);
            assert_eq!(p.n_h, 16);
            assert_eq!(p.n_pb_w, 4);
            assert_eq!(p.pb_factor, 1);
            assert_eq!(p.x_offset, (i as u32) * 4);
            assert_eq!(p.y_offset, 0);
            assert_eq!(p.x_part_pb_idx, 0);
        }
    }

    /// 8x8 vertical split → 4 subpartitions of 2x8. `nPbW = 4`,
    /// `pbFactor = 2`, so partitions 0 and 1 share a prediction
    /// (`xPartPbIdx == 0` for i=0, == 1 for i=1).
    #[test]
    fn ver_split_8x8_creates_pb_factor_of_two() {
        let parts = iter_isp_partitions(IspSplitType::VerSplit, 8, 8);
        assert_eq!(parts.len(), 4);
        for p in &parts {
            assert_eq!(p.n_w, 2);
            assert_eq!(p.n_h, 8);
            assert_eq!(p.n_pb_w, 4);
            assert_eq!(p.pb_factor, 2);
        }
        assert_eq!(parts[0].x_part_pb_idx, 0);
        assert_eq!(parts[1].x_part_pb_idx, 1);
        assert_eq!(parts[2].x_part_pb_idx, 0);
        assert_eq!(parts[3].x_part_pb_idx, 1);
    }

    /// (4, 8) and (8, 4) only emit 2 partitions — the spec's special
    /// case to avoid sub-TBs smaller than 16 samples.
    #[test]
    fn small_cu_yields_two_partitions() {
        let parts_h = iter_isp_partitions(IspSplitType::HorSplit, 4, 8);
        assert_eq!(parts_h.len(), 2);
        assert_eq!(parts_h[0].n_w, 4);
        assert_eq!(parts_h[0].n_h, 4);
        assert_eq!(parts_h[1].y_offset, 4);

        let parts_v = iter_isp_partitions(IspSplitType::VerSplit, 8, 4);
        assert_eq!(parts_v.len(), 2);
        assert_eq!(parts_v[0].n_w, 4);
        assert_eq!(parts_v[0].n_h, 4);
        assert_eq!(parts_v[1].x_offset, 4);
    }

    /// NoSplit produces an empty walk. Callers handle the NoSplit
    /// case directly and never iterate over the (1-entry) partition
    /// list.
    #[test]
    fn no_split_emits_empty_walk() {
        let parts = iter_isp_partitions(IspSplitType::NoSplit, 16, 16);
        assert!(parts.is_empty());
    }

    /// `refW` / `refH` for ISP-split luma sums the CU and TB sides
    /// (eqs. 315 – 316). Chroma + NoSplit fall back to 2x dims
    /// (eqs. 313 – 314).
    #[test]
    fn ref_dimensions_match_spec() {
        // ISP_NO_SPLIT: refW = 2 * nTbW.
        assert_eq!(
            ref_dimensions(IspSplitType::NoSplit, 0, 16, 16, 8, 8),
            (16, 16)
        );
        // chroma path falls back to 2 * nTbW even with ISP set on luma.
        assert_eq!(
            ref_dimensions(IspSplitType::HorSplit, 1, 16, 16, 8, 4),
            (16, 8)
        );
        // ISP_HOR_SPLIT luma: refW = nCbW + nTbW = 16 + 16 = 32,
        // refH = nCbH + nTbH = 16 + 4 = 20.
        assert_eq!(
            ref_dimensions(IspSplitType::HorSplit, 0, 16, 16, 16, 4),
            (32, 20)
        );
        // ISP_VER_SPLIT luma: refW = nCbW + nTbW = 16 + 4 = 20,
        // refH = nCbH + nTbH = 16 + 16 = 32.
        assert_eq!(
            ref_dimensions(IspSplitType::VerSplit, 0, 16, 16, 4, 16),
            (20, 32)
        );
    }
}
