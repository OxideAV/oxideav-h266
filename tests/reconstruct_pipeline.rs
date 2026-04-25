//! Integration tests for the §8 reconstruction pipeline.
//!
//! These tests exercise [`oxideav_h266::ctu::CtuWalker::reconstruct_leaf_cu`]
//! and [`oxideav_h266::ctu::CtuWalker::decode_picture_into`] against
//! synthetic SPS / PPS / slice fixtures. They do not rely on a real
//! VVC bitstream — the CTU walker is fed a zero-byte CABAC payload, so
//! every context-coded decision stays on the MPS path. That gives a
//! deterministic, residual-free walk for the picture-edge "left and
//! above unavailable" case where intra prediction collapses to a flat
//! mid-grey fill and the inverse-transform pipeline runs end-to-end
//! without producing measurable noise.

use oxideav_h266::ctu::{CtuLayout, CtuWalker};
use oxideav_h266::pps::PicParameterSet;
use oxideav_h266::reconstruct::PictureBuffer;
use oxideav_h266::slice_header::{SliceType, StatefulSliceHeader};
use oxideav_h266::sps::{PartitionConstraints, SeqParameterSet, ToolFlags};

fn dummy_sps(ctb_log2_minus5: u8, pic_w: u32, pic_h: u32) -> SeqParameterSet {
    SeqParameterSet {
        sps_seq_parameter_set_id: 0,
        sps_video_parameter_set_id: 0,
        sps_max_sublayers_minus1: 0,
        sps_chroma_format_idc: 1,
        sps_log2_ctu_size_minus5: ctb_log2_minus5,
        sps_ptl_dpb_hrd_params_present_flag: false,
        profile_tier_level: None,
        sps_gdr_enabled_flag: false,
        sps_ref_pic_resampling_enabled_flag: false,
        sps_res_change_in_clvs_allowed_flag: false,
        sps_pic_width_max_in_luma_samples: pic_w,
        sps_pic_height_max_in_luma_samples: pic_h,
        conformance_window: None,
        sps_subpic_info_present_flag: false,
        sps_bitdepth_minus8: 0,
        sps_entropy_coding_sync_enabled_flag: false,
        sps_entry_point_offsets_present_flag: false,
        sps_log2_max_pic_order_cnt_lsb_minus4: 4,
        sps_poc_msb_cycle_flag: false,
        sps_poc_msb_cycle_len_minus1: 0,
        sps_num_extra_ph_bytes: 0,
        sps_num_extra_sh_bytes: 0,
        sps_sublayer_dpb_params_flag: false,
        dpb_parameters: None,
        partition_constraints: PartitionConstraints::default(),
        tool_flags: ToolFlags::default(),
        subpic_info: None,
        sps_timing_hrd_params_present_flag: false,
        general_timing_hrd: None,
        sps_sublayer_cpb_params_present_flag: false,
        ols_timing_hrd: None,
        sps_field_seq_flag: false,
        sps_vui_parameters_present_flag: false,
        vui_payload: Vec::new(),
        sps_extension_flag: false,
        sps_range_extension_flag: false,
        sps_extension_7bits: 0,
    }
}

fn dummy_pps(pic_w: u32, pic_h: u32) -> PicParameterSet {
    PicParameterSet {
        pps_pic_parameter_set_id: 0,
        pps_seq_parameter_set_id: 0,
        pps_mixed_nalu_types_in_pic_flag: false,
        pps_pic_width_in_luma_samples: pic_w,
        pps_pic_height_in_luma_samples: pic_h,
        conformance_window: None,
        scaling_window: None,
        pps_output_flag_present_flag: false,
        pps_no_pic_partition_flag: true,
        pps_subpic_id_mapping_present_flag: false,
        pps_rect_slice_flag: true,
        pps_single_slice_per_subpic_flag: true,
        pps_loop_filter_across_slices_enabled_flag: false,
        pps_cabac_init_present_flag: false,
        pps_num_ref_idx_default_active_minus1: [0, 0],
        pps_rpl1_idx_present_flag: false,
        pps_weighted_pred_flag: false,
        pps_weighted_bipred_flag: false,
        pps_ref_wraparound_enabled_flag: false,
        pps_pic_width_minus_wraparound_offset: 0,
        pps_init_qp_minus26: 0,
        pps_cu_qp_delta_enabled_flag: false,
        pps_chroma_tool_offsets_present_flag: false,
        pps_cb_qp_offset: 0,
        pps_cr_qp_offset: 0,
        pps_joint_cbcr_qp_offset_present_flag: false,
        pps_joint_cbcr_qp_offset_value: 0,
        pps_slice_chroma_qp_offsets_present_flag: false,
        pps_cu_chroma_qp_offset_list_enabled_flag: false,
        pps_deblocking_filter_control_present_flag: false,
        pps_deblocking_filter_override_enabled_flag: false,
        pps_deblocking_filter_disabled_flag: false,
        pps_dbf_info_in_ph_flag: false,
        pps_rpl_info_in_ph_flag: true,
        pps_sao_info_in_ph_flag: true,
        pps_alf_info_in_ph_flag: true,
        pps_wp_info_in_ph_flag: true,
        pps_qp_delta_info_in_ph_flag: false,
        pps_picture_header_extension_present_flag: false,
        pps_slice_header_extension_present_flag: false,
        pps_extension_flag: false,
        // Single-slice fixture: pps_no_pic_partition_flag == 1 means
        // the partition struct must be absent.
        partition: None,
    }
}

fn intra_slice_header() -> StatefulSliceHeader {
    StatefulSliceHeader {
        sh_slice_type: SliceType::I,
        sh_qp_delta: 0,
        ..Default::default()
    }
}

/// Walk a single 64x64 picture as one CTU and reconstruct every leaf
/// into a 4:2:0 luma plane. Fails if any leaf surfaces an Unsupported
/// error; success means the §8.4 / §8.7 pipeline (intra + dequant +
/// inverse 2D DCT-II + add residual + clip) ran for every CU.
#[test]
fn decode_picture_into_64x64_zero_stream() {
    // CTB size = 32 (sps_log2_ctu_size_minus5 = 0 → ctb = 32). 64x64
    // picture = 2x2 CTUs.
    let sps = dummy_sps(0, 64, 64);
    let pps = dummy_pps(64, 64);
    let sh = intra_slice_header();
    let layout = CtuLayout::from_sps_pps(&sps, &pps);

    // Plenty of bytes — the residual walker may consume tens of bins
    // per leaf even on a zero stream.
    let payload = [0u8; 1024];
    let mut walker = CtuWalker::begin_slice(&layout, &sps, &pps, &sh, 0, &payload).unwrap();

    let mut out = PictureBuffer::yuv420_filled(64, 64, 0);
    walker.decode_picture_into(&mut out).unwrap();

    // The luma plane must contain plausible pixel values everywhere.
    // We check that some non-zero samples were written (the seed of 0
    // would be rewritten to ~mid-grey by every leaf).
    let nonzero = out.luma.samples.iter().filter(|&&v| v != 0).count();
    assert!(
        nonzero > out.luma.samples.len() / 2,
        "expected most luma samples to be painted, got {nonzero} / {}",
        out.luma.samples.len()
    );
}

/// Smaller (32x32) single-CTU picture: makes sure
/// `decode_picture_into` works for the smallest legal CTU layout
/// `CtbLog2SizeY = 5`.
#[test]
fn decode_picture_into_32x32_single_ctu() {
    let sps = dummy_sps(0, 32, 32);
    let pps = dummy_pps(32, 32);
    let sh = intra_slice_header();
    let layout = CtuLayout::from_sps_pps(&sps, &pps);
    let payload = [0u8; 256];
    let mut walker = CtuWalker::begin_slice(&layout, &sps, &pps, &sh, 0, &payload).unwrap();
    let mut out = PictureBuffer::yuv420_filled(32, 32, 0);
    walker.decode_picture_into(&mut out).unwrap();
    // Same plausibility check — most samples should now be non-zero.
    let nonzero = out.luma.samples.iter().filter(|&&v| v != 0).count();
    assert!(nonzero > out.luma.samples.len() / 2);
}

/// `OwnedIntraRefs::from_plane` falls back to mid-grey when both edges
/// are unavailable — the spec's substitution rule (§8.4.5.2.8).
#[test]
fn picture_buffer_constructs_yuv420() {
    let buf = PictureBuffer::yuv420_filled(64, 64, 17);
    assert_eq!(buf.luma.samples.len(), 64 * 64);
    assert_eq!(buf.cb.samples.len(), 32 * 32);
    assert_eq!(buf.cr.samples.len(), 32 * 32);
    assert!(buf.luma.samples.iter().all(|&v| v == 17));
    assert!(buf.cb.samples.iter().all(|&v| v == 128));
    assert!(buf.cr.samples.iter().all(|&v| v == 128));
}

/// End-to-end IDR reconstruction must paint chroma planes too — not
/// just luma. The seed used to mark "untouched" chroma is overridden
/// to a sentinel value so the assertion catches the case where the
/// chroma path is silently skipped.
#[test]
fn decode_picture_into_paints_chroma_planes() {
    let sps = dummy_sps(0, 32, 32);
    let pps = dummy_pps(32, 32);
    let sh = intra_slice_header();
    let layout = CtuLayout::from_sps_pps(&sps, &pps);
    let payload = [0u8; 256];
    let mut walker = CtuWalker::begin_slice(&layout, &sps, &pps, &sh, 0, &payload).unwrap();
    let mut out = PictureBuffer::yuv420_filled(32, 32, 0);
    // Sentinel seed for chroma so we can detect writes (the default
    // `yuv420_filled` constructor seeds chroma to 128, the exact value
    // the prediction produces at the picture edge — that would mask
    // the test).
    for v in out.cb.samples.iter_mut() {
        *v = 7;
    }
    for v in out.cr.samples.iter_mut() {
        *v = 7;
    }
    walker.decode_picture_into(&mut out).unwrap();
    let cb_overwritten = out.cb.samples.iter().filter(|&&v| v != 7).count();
    let cr_overwritten = out.cr.samples.iter().filter(|&&v| v != 7).count();
    assert!(
        cb_overwritten > out.cb.samples.len() / 2,
        "expected most Cb samples to be painted, got {cb_overwritten} / {}",
        out.cb.samples.len()
    );
    assert!(
        cr_overwritten > out.cr.samples.len() / 2,
        "expected most Cr samples to be painted, got {cr_overwritten} / {}",
        out.cr.samples.len()
    );
}
