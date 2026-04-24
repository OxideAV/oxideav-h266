//! VVC Picture Parameter Set parser (§7.3.2.5).
//!
//! Foundation scope: identifier + picture dimension fields, conformance
//! and scaling windows, the partition-gating flags. The tile/slice
//! layout sub-syntax (under `!pps_no_pic_partition_flag`) references
//! derived variables (`NumTilesInPic`, `SliceTopLeftTileIdx[]`, …) whose
//! population requires full SPS context; we refuse to walk it here and
//! surface an `Unsupported` error when the PPS signals per-picture
//! tiling. A single-tile / single-slice PPS is the common case for
//! conformance fixtures.

use oxideav_core::{Error, Result};

use crate::bitreader::BitReader;

/// Derived tile + slice partitioning layout (§6.5.1 + §7.3.2.5).
///
/// Populated when `pps_no_pic_partition_flag == 0`. The derived
/// quantities listed here are the minimum set the slice-header
/// parser consults to resolve `sh_slice_address` / `sh_num_tiles_in_slice_minus1`
/// emission and widths (§7.4.8). The full `CtbAddrInSlice[][]` matrix
/// — needed only by a CTU-level walker — is intentionally left out.
#[derive(Clone, Debug, Default)]
pub struct PicPartition {
    /// PPS-level CTU size. §7.4.3.5 requires this to equal the SPS's
    /// `sps_log2_ctu_size_minus5`.
    pub log2_ctu_size_minus5: u8,
    /// Raw `pps_tile_column_width_minus1[i]` values as transmitted (not
    /// expanded by the §6.5.1 uniform-replication).
    pub explicit_col_widths: Vec<u32>,
    pub explicit_row_heights: Vec<u32>,
    /// `ColWidthVal[i]` / `RowHeightVal[j]` after §6.5.1 expansion
    /// (`equations 14 / 15`). Each value is in CTB units.
    pub col_width_ctbs: Vec<u32>,
    pub row_height_ctbs: Vec<u32>,
    /// `NumTileColumns` + `NumTileRows` (§6.5.1). `num_tiles_in_pic`
    /// equals their product.
    pub num_tile_columns: u32,
    pub num_tile_rows: u32,
    pub num_tiles_in_pic: u32,
    pub pps_loop_filter_across_tiles_enabled_flag: bool,
    /// `pps_num_slices_in_pic_minus1 + 1` when `pps_rect_slice_flag`
    /// and the explicit loop ran. Equals `num_subpics` when
    /// `pps_single_slice_per_subpic_flag == 1`.
    pub num_slices_in_pic: u32,
    /// `SliceTopLeftTileIdx[i]` for the non-single-slice rectangular
    /// layout. Length equals `num_slices_in_pic` when populated.
    pub slice_top_left_tile_idx: Vec<u32>,
    /// `NumSlicesInSubpic[i]` per §6.5.1. Indexed by subpic idx.
    pub num_slices_in_subpic: Vec<u32>,
    /// `pps_tile_idx_delta_present_flag`.
    pub tile_idx_delta_present_flag: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct ConformanceWindow {
    pub left_offset: u32,
    pub right_offset: u32,
    pub top_offset: u32,
    pub bottom_offset: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct ScalingWindow {
    pub left_offset: i32,
    pub right_offset: i32,
    pub top_offset: i32,
    pub bottom_offset: i32,
}

#[derive(Clone, Debug)]
pub struct PicParameterSet {
    pub pps_pic_parameter_set_id: u8,
    pub pps_seq_parameter_set_id: u8,
    pub pps_mixed_nalu_types_in_pic_flag: bool,
    pub pps_pic_width_in_luma_samples: u32,
    pub pps_pic_height_in_luma_samples: u32,
    pub conformance_window: Option<ConformanceWindow>,
    pub scaling_window: Option<ScalingWindow>,
    pub pps_output_flag_present_flag: bool,
    pub pps_no_pic_partition_flag: bool,
    pub pps_subpic_id_mapping_present_flag: bool,
    /// `pps_rect_slice_flag` — inferred to `true` when the partitioning
    /// block is skipped (§7.4.3.5 inference: defaults to 1 when not
    /// transmitted). `pps_no_pic_partition_flag == 1` also forces
    /// `NumTilesInPic = 1` / `pps_num_slices_in_pic_minus1 = 0`.
    pub pps_rect_slice_flag: bool,
    pub pps_single_slice_per_subpic_flag: bool,
    pub pps_loop_filter_across_slices_enabled_flag: bool,
    pub pps_cabac_init_present_flag: bool,
    pub pps_num_ref_idx_default_active_minus1: [u32; 2],
    pub pps_rpl1_idx_present_flag: bool,
    pub pps_weighted_pred_flag: bool,
    pub pps_weighted_bipred_flag: bool,
    pub pps_ref_wraparound_enabled_flag: bool,
    pub pps_pic_width_minus_wraparound_offset: u32,
    pub pps_init_qp_minus26: i32,
    pub pps_cu_qp_delta_enabled_flag: bool,
    pub pps_chroma_tool_offsets_present_flag: bool,
    pub pps_cb_qp_offset: i32,
    pub pps_cr_qp_offset: i32,
    pub pps_joint_cbcr_qp_offset_present_flag: bool,
    pub pps_joint_cbcr_qp_offset_value: i32,
    pub pps_slice_chroma_qp_offsets_present_flag: bool,
    pub pps_cu_chroma_qp_offset_list_enabled_flag: bool,
    pub pps_deblocking_filter_control_present_flag: bool,
    pub pps_deblocking_filter_override_enabled_flag: bool,
    pub pps_deblocking_filter_disabled_flag: bool,
    pub pps_dbf_info_in_ph_flag: bool,
    /// Present only when `pps_no_pic_partition_flag == 0`; inferred to
    /// `true` otherwise (§7.4.3.5 / §7.3.2.5: the "in PH" flags default
    /// to 1 when not signalled, forcing the slice header to skip the
    /// corresponding per-slice fields).
    pub pps_rpl_info_in_ph_flag: bool,
    pub pps_sao_info_in_ph_flag: bool,
    pub pps_alf_info_in_ph_flag: bool,
    pub pps_wp_info_in_ph_flag: bool,
    pub pps_qp_delta_info_in_ph_flag: bool,
    pub pps_picture_header_extension_present_flag: bool,
    pub pps_slice_header_extension_present_flag: bool,
    pub pps_extension_flag: bool,
    /// Derived partition layout. `None` when
    /// `pps_no_pic_partition_flag == 1` (single tile, single slice).
    pub partition: Option<PicPartition>,
}

/// Parse a PPS NAL RBSP payload (the bytes after the 2-byte NAL header,
/// already stripped of emulation-prevention bytes).
pub fn parse_pps(rbsp: &[u8]) -> Result<PicParameterSet> {
    let mut br = BitReader::new(rbsp);
    let pps_pic_parameter_set_id = br.u(6)? as u8;
    let pps_seq_parameter_set_id = br.u(4)? as u8;
    let pps_mixed_nalu_types_in_pic_flag = br.u1()? == 1;
    let pps_pic_width_in_luma_samples = br.ue()?;
    let pps_pic_height_in_luma_samples = br.ue()?;
    if pps_pic_width_in_luma_samples == 0
        || pps_pic_height_in_luma_samples == 0
        || pps_pic_width_in_luma_samples > 16384
        || pps_pic_height_in_luma_samples > 16384
    {
        return Err(Error::invalid(format!(
            "h266 PPS: implausible picture size {pps_pic_width_in_luma_samples}x{pps_pic_height_in_luma_samples}"
        )));
    }
    let pps_conformance_window_flag = br.u1()? == 1;
    let conformance_window = if pps_conformance_window_flag {
        Some(ConformanceWindow {
            left_offset: br.ue()?,
            right_offset: br.ue()?,
            top_offset: br.ue()?,
            bottom_offset: br.ue()?,
        })
    } else {
        None
    };
    let pps_scaling_window_explicit_signalling_flag = br.u1()? == 1;
    let scaling_window = if pps_scaling_window_explicit_signalling_flag {
        Some(ScalingWindow {
            left_offset: br.se()?,
            right_offset: br.se()?,
            top_offset: br.se()?,
            bottom_offset: br.se()?,
        })
    } else {
        None
    };
    let pps_output_flag_present_flag = br.u1()? == 1;
    let pps_no_pic_partition_flag = br.u1()? == 1;
    let pps_subpic_id_mapping_present_flag = br.u1()? == 1;
    if pps_subpic_id_mapping_present_flag {
        return Err(Error::unsupported(
            "h266 PPS: pps_subpic_id_mapping_present_flag = 1 (subpicture streams not yet supported)",
        ));
    }

    // Partition-block state. `pps_rect_slice_flag` and
    // `pps_single_slice_per_subpic_flag` default to "1" when the block
    // is skipped (§7.4.3.5). When the block is emitted, both are
    // overwritten by the signalled values.
    let mut pps_rect_slice_flag = true;
    let mut pps_single_slice_per_subpic_flag = true;
    let mut pps_loop_filter_across_slices_enabled_flag = false;
    let mut partition: Option<PicPartition> = None;

    if !pps_no_pic_partition_flag {
        // §7.3.2.5 partition block. We need PicWidthInCtbsY /
        // PicHeightInCtbsY to bound the uniform-replication loops and
        // the sh_slice_address width.
        let log2_ctu_size_minus5 = br.u(2)? as u8;
        let ctb_log2 = log2_ctu_size_minus5 as u32 + 5;
        let ctb_size = 1u32 << ctb_log2;
        let pic_width_ctbs =
            (pps_pic_width_in_luma_samples + ctb_size - 1) / ctb_size;
        let pic_height_ctbs =
            (pps_pic_height_in_luma_samples + ctb_size - 1) / ctb_size;
        if pic_width_ctbs == 0 || pic_height_ctbs == 0 {
            return Err(Error::invalid(
                "h266 PPS: partition block with zero CTB-count picture dimensions",
            ));
        }

        let num_exp_tile_columns_minus1 = br.ue()?;
        let num_exp_tile_rows_minus1 = br.ue()?;
        if num_exp_tile_columns_minus1 >= pic_width_ctbs
            || num_exp_tile_rows_minus1 >= pic_height_ctbs
        {
            return Err(Error::invalid(format!(
                "h266 PPS: num_exp_tile_columns/rows out of range ({num_exp_tile_columns_minus1}, {num_exp_tile_rows_minus1})"
            )));
        }

        let mut explicit_col_widths: Vec<u32> =
            Vec::with_capacity(num_exp_tile_columns_minus1 as usize + 1);
        for _ in 0..=num_exp_tile_columns_minus1 {
            let w = br.ue()?;
            if w >= pic_width_ctbs {
                return Err(Error::invalid(format!(
                    "h266 PPS: pps_tile_column_width_minus1 out of range ({w})"
                )));
            }
            explicit_col_widths.push(w + 1);
        }
        let mut explicit_row_heights: Vec<u32> =
            Vec::with_capacity(num_exp_tile_rows_minus1 as usize + 1);
        for _ in 0..=num_exp_tile_rows_minus1 {
            let h = br.ue()?;
            if h >= pic_height_ctbs {
                return Err(Error::invalid(format!(
                    "h266 PPS: pps_tile_row_height_minus1 out of range ({h})"
                )));
            }
            explicit_row_heights.push(h + 1);
        }

        // §6.5.1 equations 14 / 15: repeat the last explicit width /
        // height until the picture is covered.
        let col_width_ctbs =
            derive_tile_sizes(&explicit_col_widths, pic_width_ctbs);
        let row_height_ctbs =
            derive_tile_sizes(&explicit_row_heights, pic_height_ctbs);
        let num_tile_columns = col_width_ctbs.len() as u32;
        let num_tile_rows = row_height_ctbs.len() as u32;
        let num_tiles_in_pic = num_tile_columns.saturating_mul(num_tile_rows);

        let mut pps_loop_filter_across_tiles_enabled_flag = false;
        if num_tiles_in_pic > 1 {
            pps_loop_filter_across_tiles_enabled_flag = br.u1()? == 1;
            pps_rect_slice_flag = br.u1()? == 1;
        }
        let mut pps_tile_idx_delta_present_flag = false;
        let mut slice_top_left_tile_idx: Vec<u32> = Vec::new();
        let mut num_slices_in_pic: u32 = 1;

        if pps_rect_slice_flag {
            pps_single_slice_per_subpic_flag = br.u1()? == 1;
        }
        if pps_rect_slice_flag && !pps_single_slice_per_subpic_flag {
            let pps_num_slices_in_pic_minus1 = br.ue()?;
            // Conformance bound: NumSlicesInPic ≤ MaxSlicesPerAu (600
            // for the highest defined level). Guard against runaway.
            if pps_num_slices_in_pic_minus1 > 2047 {
                return Err(Error::invalid(format!(
                    "h266 PPS: pps_num_slices_in_pic_minus1 out of range ({pps_num_slices_in_pic_minus1})"
                )));
            }
            num_slices_in_pic = pps_num_slices_in_pic_minus1 + 1;
            if num_slices_in_pic > 1 {
                pps_tile_idx_delta_present_flag = br.u1()? == 1;
            }
            // We walk the per-slice fields enough to advance the
            // bitreader past them — the full `CtbAddrInSlice[][]`
            // derivation is out of scope for this round. The only
            // derived variable we track is `SliceTopLeftTileIdx[i]`.
            let mut tile_idx: i64 = 0;
            slice_top_left_tile_idx.reserve(num_slices_in_pic as usize);
            let num_tile_cols_i = num_tile_columns as i64;
            let mut slice_widths: Vec<u32> = Vec::with_capacity(num_slices_in_pic as usize);
            let mut slice_heights: Vec<u32> = Vec::with_capacity(num_slices_in_pic as usize);
            let mut i: u32 = 0;
            while i < pps_num_slices_in_pic_minus1 {
                if tile_idx < 0 || tile_idx >= num_tiles_in_pic as i64 {
                    return Err(Error::invalid(format!(
                        "h266 PPS: slice tile idx {tile_idx} out of range"
                    )));
                }
                slice_top_left_tile_idx.push(tile_idx as u32);
                let tile_x = (tile_idx % num_tile_cols_i) as u32;
                let tile_y = (tile_idx / num_tile_cols_i) as u32;
                let mut slice_w: u32 = 1;
                let mut slice_h: u32 = 1;
                let last_col = tile_x == num_tile_columns - 1;
                let last_row = tile_y == num_tile_rows - 1;
                if !last_col {
                    slice_w = br.ue()? + 1;
                }
                if !last_row
                    && (pps_tile_idx_delta_present_flag || tile_x == 0)
                {
                    slice_h = br.ue()? + 1;
                }
                slice_widths.push(slice_w);
                slice_heights.push(slice_h);
                if slice_w == 1
                    && slice_h == 1
                    && row_height_ctbs[tile_y as usize] > 1
                {
                    let pps_num_exp_slices_in_tile = br.ue()?;
                    if pps_num_exp_slices_in_tile > row_height_ctbs[tile_y as usize] {
                        return Err(Error::invalid(format!(
                            "h266 PPS: pps_num_exp_slices_in_tile out of range ({pps_num_exp_slices_in_tile})"
                        )));
                    }
                    for _ in 0..pps_num_exp_slices_in_tile {
                        let _h = br.ue()?;
                    }
                    // §6.5.1 derivation of NumSlicesInTile would
                    // advance `i` by `NumSlicesInTile - 1` here; for
                    // this scaffold we treat each explicit entry as a
                    // single slice in the loop (consumers doing full
                    // CTU-level walk can recompute the NumSlicesInTile
                    // from the stored pps_num_exp_slices_in_tile).
                }
                if pps_tile_idx_delta_present_flag && i < pps_num_slices_in_pic_minus1 {
                    let delta = br.se()?;
                    tile_idx += delta as i64;
                } else {
                    tile_idx += slice_w as i64;
                    if tile_idx % num_tile_cols_i == 0 {
                        tile_idx += (slice_h as i64 - 1) * num_tile_cols_i;
                    }
                }
                i += 1;
            }
            // Last slice gets the remaining top-left tile.
            if tile_idx < 0 || tile_idx >= num_tiles_in_pic as i64 {
                return Err(Error::invalid(format!(
                    "h266 PPS: last slice tile idx {tile_idx} out of range"
                )));
            }
            slice_top_left_tile_idx.push(tile_idx as u32);
        }
        if !pps_rect_slice_flag
            || pps_single_slice_per_subpic_flag
            || num_slices_in_pic > 1
        {
            pps_loop_filter_across_slices_enabled_flag = br.u1()? == 1;
        }

        // NumSlicesInSubpic[]: §6.5.1 equation 23. When the SPS has no
        // subpic info, there is exactly one subpic and every slice is
        // mapped to it. Full subpic-aware derivation lives in a future
        // increment.
        let num_slices_in_subpic = if pps_single_slice_per_subpic_flag {
            // Each subpic has exactly one slice.
            vec![1u32]
        } else {
            vec![num_slices_in_pic]
        };

        partition = Some(PicPartition {
            log2_ctu_size_minus5,
            explicit_col_widths,
            explicit_row_heights,
            col_width_ctbs,
            row_height_ctbs,
            num_tile_columns,
            num_tile_rows,
            num_tiles_in_pic,
            pps_loop_filter_across_tiles_enabled_flag,
            num_slices_in_pic,
            slice_top_left_tile_idx,
            num_slices_in_subpic,
            tile_idx_delta_present_flag: pps_tile_idx_delta_present_flag,
        });
    }

    let pps_cabac_init_present_flag = br.u1()? == 1;
    let pps_num_ref_idx_default_active_minus1 = [br.ue()?, br.ue()?];
    let pps_rpl1_idx_present_flag = br.u1()? == 1;
    let pps_weighted_pred_flag = br.u1()? == 1;
    let pps_weighted_bipred_flag = br.u1()? == 1;
    let pps_ref_wraparound_enabled_flag = br.u1()? == 1;
    let pps_pic_width_minus_wraparound_offset = if pps_ref_wraparound_enabled_flag {
        br.ue()?
    } else {
        0
    };
    let pps_init_qp_minus26 = br.se()?;
    let pps_cu_qp_delta_enabled_flag = br.u1()? == 1;
    let pps_chroma_tool_offsets_present_flag = br.u1()? == 1;
    let mut pps_cb_qp_offset: i32 = 0;
    let mut pps_cr_qp_offset: i32 = 0;
    let mut pps_joint_cbcr_qp_offset_present_flag = false;
    let mut pps_joint_cbcr_qp_offset_value: i32 = 0;
    let mut pps_slice_chroma_qp_offsets_present_flag = false;
    let mut pps_cu_chroma_qp_offset_list_enabled_flag = false;
    if pps_chroma_tool_offsets_present_flag {
        pps_cb_qp_offset = br.se()?;
        pps_cr_qp_offset = br.se()?;
        pps_joint_cbcr_qp_offset_present_flag = br.u1()? == 1;
        if pps_joint_cbcr_qp_offset_present_flag {
            pps_joint_cbcr_qp_offset_value = br.se()?;
        }
        pps_slice_chroma_qp_offsets_present_flag = br.u1()? == 1;
        pps_cu_chroma_qp_offset_list_enabled_flag = br.u1()? == 1;
        if pps_cu_chroma_qp_offset_list_enabled_flag {
            let len_minus1 = br.ue()?;
            // Walk the pairs / triples — we don't retain them in this
            // pass (only the gate matters for slice-header decoding).
            if len_minus1 > 64 {
                return Err(Error::invalid(format!(
                    "h266 PPS: pps_chroma_qp_offset_list_len_minus1 out of range ({len_minus1})"
                )));
            }
            for _ in 0..=len_minus1 {
                let _ = br.se()?; // pps_cb_qp_offset_list[i]
                let _ = br.se()?; // pps_cr_qp_offset_list[i]
                if pps_joint_cbcr_qp_offset_present_flag {
                    let _ = br.se()?; // pps_joint_cbcr_qp_offset_list[i]
                }
            }
        }
    }
    let pps_deblocking_filter_control_present_flag = br.u1()? == 1;
    let mut pps_deblocking_filter_override_enabled_flag = false;
    let mut pps_deblocking_filter_disabled_flag = false;
    let mut pps_dbf_info_in_ph_flag = true; // inferred when not present
    if pps_deblocking_filter_control_present_flag {
        pps_deblocking_filter_override_enabled_flag = br.u1()? == 1;
        pps_deblocking_filter_disabled_flag = br.u1()? == 1;
        // §7.3.2.5: pps_dbf_info_in_ph_flag is transmitted only when
        // `!pps_no_pic_partition_flag && pps_deblocking_filter_override_enabled_flag`.
        // Otherwise it's inferred to 1 (so the slice header skips the
        // deblocking-override branch).
        if !pps_no_pic_partition_flag && pps_deblocking_filter_override_enabled_flag {
            pps_dbf_info_in_ph_flag = br.u1()? == 1;
        }
        if !pps_deblocking_filter_disabled_flag {
            let _ = br.se()?; // pps_luma_beta_offset_div2
            let _ = br.se()?; // pps_luma_tc_offset_div2
            if pps_chroma_tool_offsets_present_flag {
                let _ = br.se()?;
                let _ = br.se()?;
                let _ = br.se()?;
                let _ = br.se()?;
            }
        }
    }

    // `if (!pps_no_pic_partition_flag)` block: the five in_ph_flag
    // members are either signalled or inferred (§7.4.3.5).
    let mut pps_rpl_info_in_ph_flag = true;
    let mut pps_sao_info_in_ph_flag = true;
    let mut pps_alf_info_in_ph_flag = true;
    let mut pps_wp_info_in_ph_flag = true;
    let mut pps_qp_delta_info_in_ph_flag = true;
    if !pps_no_pic_partition_flag {
        pps_rpl_info_in_ph_flag = br.u1()? == 1;
        pps_sao_info_in_ph_flag = br.u1()? == 1;
        pps_alf_info_in_ph_flag = br.u1()? == 1;
        if (pps_weighted_pred_flag || pps_weighted_bipred_flag) && pps_rpl_info_in_ph_flag {
            pps_wp_info_in_ph_flag = br.u1()? == 1;
        } else {
            // §7.4.3.5: inferred to 0 when pps_rpl_info_in_ph_flag == 0
            // (weighted-pred table must follow the per-slice RPL).
            pps_wp_info_in_ph_flag = pps_rpl_info_in_ph_flag;
        }
        pps_qp_delta_info_in_ph_flag = br.u1()? == 1;
    }

    let pps_picture_header_extension_present_flag = br.u1()? == 1;
    let pps_slice_header_extension_present_flag = br.u1()? == 1;
    let pps_extension_flag = br.u1()? == 1;
    if pps_extension_flag {
        // Consume the extension-data bits until the stop bit (§7.3.2.5).
        while br.has_more_rbsp_data() {
            br.u1()?;
        }
    }

    Ok(PicParameterSet {
        pps_pic_parameter_set_id,
        pps_seq_parameter_set_id,
        pps_mixed_nalu_types_in_pic_flag,
        pps_pic_width_in_luma_samples,
        pps_pic_height_in_luma_samples,
        conformance_window,
        scaling_window,
        pps_output_flag_present_flag,
        pps_no_pic_partition_flag,
        pps_subpic_id_mapping_present_flag,
        pps_rect_slice_flag,
        pps_single_slice_per_subpic_flag,
        pps_loop_filter_across_slices_enabled_flag,
        pps_cabac_init_present_flag,
        pps_num_ref_idx_default_active_minus1,
        pps_rpl1_idx_present_flag,
        pps_weighted_pred_flag,
        pps_weighted_bipred_flag,
        pps_ref_wraparound_enabled_flag,
        pps_pic_width_minus_wraparound_offset,
        pps_init_qp_minus26,
        pps_cu_qp_delta_enabled_flag,
        pps_chroma_tool_offsets_present_flag,
        pps_cb_qp_offset,
        pps_cr_qp_offset,
        pps_joint_cbcr_qp_offset_present_flag,
        pps_joint_cbcr_qp_offset_value,
        pps_slice_chroma_qp_offsets_present_flag,
        pps_cu_chroma_qp_offset_list_enabled_flag,
        pps_deblocking_filter_control_present_flag,
        pps_deblocking_filter_override_enabled_flag,
        pps_deblocking_filter_disabled_flag,
        pps_dbf_info_in_ph_flag,
        pps_rpl_info_in_ph_flag,
        pps_sao_info_in_ph_flag,
        pps_alf_info_in_ph_flag,
        pps_wp_info_in_ph_flag,
        pps_qp_delta_info_in_ph_flag,
        pps_picture_header_extension_present_flag,
        pps_slice_header_extension_present_flag,
        pps_extension_flag,
        partition,
    })
}

/// Derive `ColWidthVal[]` / `RowHeightVal[]` per §6.5.1 equations 14 /
/// 15 by repeating the last explicit size until the remaining CTBs
/// are covered, then trimming the tail.
fn derive_tile_sizes(explicit: &[u32], pic_dim_in_ctbs: u32) -> Vec<u32> {
    let mut out: Vec<u32> = Vec::with_capacity(explicit.len() + 1);
    let mut remaining = pic_dim_in_ctbs;
    for &size in explicit {
        if size == 0 || size > remaining {
            out.push(remaining);
            return out;
        }
        out.push(size);
        remaining -= size;
    }
    let last = *explicit.last().unwrap_or(&1);
    if last > 0 {
        while remaining >= last {
            out.push(last);
            remaining -= last;
        }
    }
    if remaining > 0 {
        out.push(remaining);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn push_u(bits: &mut Vec<u8>, v: u64, n: u32) {
        for i in (0..n).rev() {
            bits.push(((v >> i) & 1) as u8);
        }
    }

    fn pack_bits(bits: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity((bits.len() + 7) / 8);
        let mut cur = 0u8;
        for (i, &bit) in bits.iter().enumerate() {
            cur |= bit << (7 - (i % 8));
            if i % 8 == 7 {
                out.push(cur);
                cur = 0;
            }
        }
        if bits.len() % 8 != 0 {
            out.push(cur);
        }
        out
    }

    fn push_ue(bits: &mut Vec<u8>, value: u32) {
        let code_num = value as u64 + 1;
        let mut zeros: u32 = 0;
        while (1u64 << (zeros + 1)) <= code_num {
            zeros += 1;
        }
        for _ in 0..zeros {
            bits.push(0);
        }
        push_u(bits, code_num, zeros + 1);
    }

    fn push_se(bits: &mut Vec<u8>, value: i32) {
        let code = if value <= 0 {
            (-(value as i64) * 2) as u32
        } else {
            (value as i64 * 2 - 1) as u32
        };
        push_ue(bits, code);
    }

    /// Minimal single-slice 320x240 PPS (no partitioning, no subpic map).
    #[test]
    fn minimal_pps_roundtrip() {
        let mut bits: Vec<u8> = Vec::new();
        push_u(&mut bits, 0, 6); // pps_id
        push_u(&mut bits, 0, 4); // sps_id
        push_u(&mut bits, 0, 1); // mixed_nalu_types
        push_ue(&mut bits, 320);
        push_ue(&mut bits, 240);
        push_u(&mut bits, 0, 1); // conformance_window_flag
        push_u(&mut bits, 0, 1); // scaling_window
        push_u(&mut bits, 0, 1); // output_flag_present
        push_u(&mut bits, 1, 1); // no_pic_partition = 1 (foundation path)
        push_u(&mut bits, 0, 1); // subpic_id_mapping_present = 0
                                 // Cabac + ref-idx defaults + rpl1 flag.
        push_u(&mut bits, 0, 1); // cabac_init_present
        push_ue(&mut bits, 0); // num_ref_idx_default_active_minus1[0]
        push_ue(&mut bits, 0); // num_ref_idx_default_active_minus1[1]
        push_u(&mut bits, 0, 1); // rpl1_idx_present
        push_u(&mut bits, 0, 1); // weighted_pred
        push_u(&mut bits, 0, 1); // weighted_bipred
        push_u(&mut bits, 0, 1); // ref_wraparound
        push_se(&mut bits, 0); // init_qp_minus26
        push_u(&mut bits, 0, 1); // cu_qp_delta_enabled
        push_u(&mut bits, 0, 1); // chroma_tool_offsets_present
        push_u(&mut bits, 0, 1); // deblocking_filter_control_present
                                 // partition block skipped. Extension-flag tail.
        push_u(&mut bits, 0, 1); // picture_header_ext_present
        push_u(&mut bits, 0, 1); // slice_header_ext_present
        push_u(&mut bits, 0, 1); // pps_extension_flag
        let bytes = pack_bits(&bits);
        let pps = parse_pps(&bytes).unwrap();
        assert_eq!(pps.pps_pic_parameter_set_id, 0);
        assert_eq!(pps.pps_seq_parameter_set_id, 0);
        assert_eq!(pps.pps_pic_width_in_luma_samples, 320);
        assert_eq!(pps.pps_pic_height_in_luma_samples, 240);
        assert!(pps.pps_no_pic_partition_flag);
        assert!(!pps.pps_subpic_id_mapping_present_flag);
        // The info_in_ph flags must be inferred to true when the
        // partition block was skipped.
        assert!(pps.pps_rpl_info_in_ph_flag);
        assert!(pps.pps_sao_info_in_ph_flag);
        assert!(pps.pps_alf_info_in_ph_flag);
        assert!(pps.pps_qp_delta_info_in_ph_flag);
        assert_eq!(pps.pps_init_qp_minus26, 0);
    }

    /// 256x128 picture (2 tile cols x 1 tile row of 128x128 CTBs),
    /// single-slice-per-subpic inferred. Exercises the partition block
    /// under `pps_no_pic_partition_flag = 0` without requiring a subpic
    /// layout.
    #[test]
    fn partitioned_pps_two_tile_cols_single_slice() {
        let mut bits: Vec<u8> = Vec::new();
        push_u(&mut bits, 0, 6); // pps_id
        push_u(&mut bits, 0, 4); // sps_id
        push_u(&mut bits, 0, 1); // mixed_nalu_types
        push_ue(&mut bits, 256);
        push_ue(&mut bits, 128);
        push_u(&mut bits, 0, 1); // conformance_window_flag
        push_u(&mut bits, 0, 1); // scaling_window
        push_u(&mut bits, 0, 1); // output_flag_present
        push_u(&mut bits, 0, 1); // no_pic_partition = 0
        push_u(&mut bits, 0, 1); // subpic_id_mapping_present = 0
                                 // --- Partition block ---
                                 // CtbSize = 128 (log2_ctu_size_minus5 = 2).
                                 // PicWidthInCtbsY = ceil(256/128) = 2.
                                 // PicHeightInCtbsY = ceil(128/128) = 1.
        push_u(&mut bits, 2, 2); // pps_log2_ctu_size_minus5
        push_ue(&mut bits, 1); // pps_num_exp_tile_columns_minus1 = 1 → 2 exp widths
        push_ue(&mut bits, 0); // pps_num_exp_tile_rows_minus1 = 0 → 1 exp height
        push_ue(&mut bits, 0); // pps_tile_column_width_minus1[0] = 0 → width 1
        push_ue(&mut bits, 0); // pps_tile_column_width_minus1[1] = 0 → width 1
        push_ue(&mut bits, 0); // pps_tile_row_height_minus1[0] = 0 → height 1
                                // NumTileColumns = 2, NumTileRows = 1 → NumTilesInPic = 2.
                                // (NumTilesInPic > 1) branch:
        push_u(&mut bits, 1, 1); // pps_loop_filter_across_tiles_enabled_flag
        push_u(&mut bits, 1, 1); // pps_rect_slice_flag
                                 // (pps_rect_slice_flag) branch:
        push_u(&mut bits, 1, 1); // pps_single_slice_per_subpic_flag = 1
                                 // !pps_rect_slice_flag || pps_single_slice_per_subpic || num_slices>0 → loop_filter_across_slices:
        push_u(&mut bits, 0, 1); // pps_loop_filter_across_slices_enabled_flag
                                 // --- End partition block ---
        push_u(&mut bits, 0, 1); // cabac_init_present
        push_ue(&mut bits, 0); // num_ref_idx_default_active_minus1[0]
        push_ue(&mut bits, 0); // num_ref_idx_default_active_minus1[1]
        push_u(&mut bits, 0, 1); // rpl1_idx_present
        push_u(&mut bits, 0, 1); // weighted_pred
        push_u(&mut bits, 0, 1); // weighted_bipred
        push_u(&mut bits, 0, 1); // ref_wraparound
        push_se(&mut bits, 0); // init_qp_minus26
        push_u(&mut bits, 0, 1); // cu_qp_delta_enabled
        push_u(&mut bits, 0, 1); // chroma_tool_offsets_present
        push_u(&mut bits, 0, 1); // deblocking_filter_control_present
                                 // Now the (!pps_no_pic_partition_flag) block: five in_ph_flag.
        push_u(&mut bits, 1, 1); // pps_rpl_info_in_ph_flag
        push_u(&mut bits, 1, 1); // pps_sao_info_in_ph_flag
        push_u(&mut bits, 1, 1); // pps_alf_info_in_ph_flag
                                 // wp_info_in_ph not emitted because weighted_pred/bipred = 0.
        push_u(&mut bits, 1, 1); // pps_qp_delta_info_in_ph_flag
        push_u(&mut bits, 0, 1); // picture_header_ext_present
        push_u(&mut bits, 0, 1); // slice_header_ext_present
        push_u(&mut bits, 0, 1); // pps_extension_flag
        let bytes = pack_bits(&bits);
        let pps = parse_pps(&bytes).unwrap();
        assert!(!pps.pps_no_pic_partition_flag);
        let p = pps.partition.as_ref().unwrap();
        assert_eq!(p.num_tile_columns, 2);
        assert_eq!(p.num_tile_rows, 1);
        assert_eq!(p.num_tiles_in_pic, 2);
        assert_eq!(p.col_width_ctbs, vec![1, 1]);
        assert_eq!(p.row_height_ctbs, vec![1]);
        assert!(pps.pps_rect_slice_flag);
        assert!(pps.pps_single_slice_per_subpic_flag);
        assert!(pps.pps_rpl_info_in_ph_flag);
        assert!(pps.pps_sao_info_in_ph_flag);
        assert!(pps.pps_alf_info_in_ph_flag);
        assert!(pps.pps_qp_delta_info_in_ph_flag);
    }

    #[test]
    fn derive_tile_sizes_repeats_last_and_rounds_tail() {
        // Picture 5 CTBs wide, explicit widths [1, 2]. Expected:
        //   [1, 2, 2 (repeated uniform)], remaining 0 → [1, 2, 2].
        let v = derive_tile_sizes(&[1, 2], 5);
        assert_eq!(v, vec![1, 2, 2]);
        // Picture 7 CTBs, explicit [1, 2] → uniform = 2, repeats → [1, 2, 2, 2].
        let v = derive_tile_sizes(&[1, 2], 7);
        assert_eq!(v, vec![1, 2, 2, 2]);
        // Tail that doesn't fit a full uniform: [2, 3], pic = 8 → [2, 3, 3].
        let v = derive_tile_sizes(&[2, 3], 8);
        assert_eq!(v, vec![2, 3, 3]);
    }
}
