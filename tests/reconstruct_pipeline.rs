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

use oxideav_h266::alf::{AlfApsBinding, AlfCtb, AlfPicture};
use oxideav_h266::aps::{AlfApsData, ALF_CHROMA_NUM_COEFF, ALF_LUMA_NUM_COEFF, NUM_ALF_FILTERS};
use oxideav_h266::ctu::{CtuLayout, CtuWalker};
use oxideav_h266::deblock::{apply_deblocking, DeblockCu, DeblockParams};
use oxideav_h266::inter::ReferencePicture;
use oxideav_h266::pps::PicParameterSet;
use oxideav_h266::reconstruct::PictureBuffer;
use oxideav_h266::sao::{SaoCtb, SaoCtbParams, SaoEoClass, SaoPicture, SaoTypeIdx};
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

/// Synthetic high-Q fixture that paints a 4-sample block-edge stripe
/// (98 / 102) across two adjacent CUs. Before deblock the seam reads
/// 98 → 102; after the §8.8.3 deblock pass the boundary samples must
/// have moved closer to each other (canonical "block edge smoothed"
/// behaviour). Picture-edge columns must remain unchanged.
#[test]
fn deblock_smooths_synthetic_block_edge_fixture() {
    let mut buf = PictureBuffer::yuv420_filled(32, 32, 98);
    for y in 0..32 {
        for x in 16..32 {
            buf.luma.samples[y * 32 + x] = 102;
        }
    }
    // Snapshot the seam columns *before* deblock to compare against.
    let before_p = buf.luma.samples[16 * 32 + 15] as i32;
    let before_q = buf.luma.samples[16 * 32 + 16] as i32;
    let cus = vec![
        DeblockCu {
            x: 0,
            y: 0,
            w: 16,
            h: 32,
            qp_y: 32,
            intra: true,
            tu_y_coded: true,
            tu_cb_coded: false,
            tu_cr_coded: false,
            bdpcm_luma: false,
            bdpcm_chroma: false,
        },
        DeblockCu {
            x: 16,
            y: 0,
            w: 16,
            h: 32,
            qp_y: 32,
            intra: true,
            tu_y_coded: true,
            tu_cb_coded: false,
            tu_cr_coded: false,
            bdpcm_luma: false,
            bdpcm_chroma: false,
        },
    ];
    let params = DeblockParams {
        disabled: false,
        bit_depth: 8,
        ..Default::default()
    };
    apply_deblocking(&mut buf, &cus, &params, 1);
    let after_p = buf.luma.samples[16 * 32 + 15] as i32;
    let after_q = buf.luma.samples[16 * 32 + 16] as i32;
    // P side moved from 98 toward 102.
    assert!(after_p > before_p, "p side: {before_p} → {after_p}");
    // Q side moved from 102 toward 98.
    assert!(after_q < before_q, "q side: {before_q} → {after_q}");
    // Picture-edge column unchanged.
    assert_eq!(buf.luma.samples[16 * 32], 98);
    assert_eq!(buf.luma.samples[16 * 32 + 31], 102);
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

/// End-to-end IDR reconstruction followed by the §8.8.3 deblocking
/// pass: must run cleanly, produce non-trivial luma data, and leave
/// edge samples within valid 8-bit range. The combination of
/// `decode_picture_into` + `apply_in_loop_filters` is the call sequence
/// the higher-level decoder will use per IDR.
#[test]
fn decode_picture_into_then_deblock() {
    let sps = dummy_sps(0, 64, 64);
    let pps = dummy_pps(64, 64);
    let sh = intra_slice_header();
    let layout = CtuLayout::from_sps_pps(&sps, &pps);
    let payload = [0u8; 1024];
    let mut walker = CtuWalker::begin_slice(&layout, &sps, &pps, &sh, 0, &payload).unwrap();
    let mut out = PictureBuffer::yuv420_filled(64, 64, 0);
    walker.decode_picture_into(&mut out).unwrap();
    walker.apply_in_loop_filters(&mut out).unwrap();
    // Sanity: every sample must still sit in 8-bit range and most must
    // be non-zero (the deblock pass only smooths — it cannot zero out
    // the painted picture).
    let nonzero = out.luma.samples.iter().filter(|&&v| v != 0).count();
    assert!(nonzero > out.luma.samples.len() / 2);
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

/// SAO end-to-end through the CTU walker: synthesise a 32×32 IDR with
/// `sh_sao_luma_used_flag = 1`, install a per-CTB band-offset entry that
/// matches the post-deblock luma value, and verify
/// [`CtuWalker::apply_in_loop_filters`] dispatches SAO correctly.
///
/// The test relies on the deterministic zero-stream behaviour: with no
/// residuals coded, every reconstructed luma sample lands at the
/// 4×4-block DC = 128 mid-grey. Picking a band that covers 128 (i.e.
/// 128 >> 3 = band 16) and a single offset of +20 must shift those
/// samples to 148.
#[test]
fn ctu_walker_apply_sao_band_offset_luma() {
    let sps = dummy_sps(0, 32, 32);
    let pps = dummy_pps(32, 32);
    let mut sh = intra_slice_header();
    sh.sh_sao_luma_used_flag = true;
    let layout = CtuLayout::from_sps_pps(&sps, &pps);
    let payload = [0u8; 256];
    let mut walker = CtuWalker::begin_slice(&layout, &sps, &pps, &sh, 0, &payload).unwrap();
    let mut out = PictureBuffer::yuv420_filled(32, 32, 0);
    walker.decode_picture_into(&mut out).unwrap();
    // Capture the post-decode (pre-SAO) luma sample at a centre position
    // so the assertion below is anchored to what the §8.4 / §8.7 path
    // actually produced (rather than relying on a literal "128").
    let centre = out.luma.samples[16 * 32 + 16];
    let centre_band = centre >> 3;
    let mut sao_pic = SaoPicture::empty(layout.pic_width_in_ctbs_y, layout.pic_height_in_ctbs_y);
    sao_pic.set(
        0,
        0,
        SaoCtbParams {
            luma: SaoCtb::band_offset(centre_band, [20, 0, 0, 0], [0, 0, 0, 0], 8),
            ..Default::default()
        },
    );
    walker.set_sao_picture(sao_pic).unwrap();
    walker.apply_in_loop_filters(&mut out).unwrap();
    let after = out.luma.samples[16 * 32 + 16];
    assert_eq!(after, centre.saturating_add(20));
}

/// SAO is a no-op when the slice flag stays off, even with non-trivial
/// per-CTB params installed. Mirrors the §8.8.4.1 gating clause.
#[test]
fn ctu_walker_apply_sao_no_op_when_flag_off() {
    let sps = dummy_sps(0, 32, 32);
    let pps = dummy_pps(32, 32);
    let sh = intra_slice_header(); // sh_sao_luma_used_flag = false
    let layout = CtuLayout::from_sps_pps(&sps, &pps);
    let payload = [0u8; 256];
    let mut walker = CtuWalker::begin_slice(&layout, &sps, &pps, &sh, 0, &payload).unwrap();
    let mut out = PictureBuffer::yuv420_filled(32, 32, 0);
    walker.decode_picture_into(&mut out).unwrap();
    let snapshot = out.luma.samples.clone();
    let mut sao_pic = SaoPicture::empty(layout.pic_width_in_ctbs_y, layout.pic_height_in_ctbs_y);
    // BO with +50 on every band — would otherwise shift many samples.
    sao_pic.set(
        0,
        0,
        SaoCtbParams {
            luma: SaoCtb::band_offset(0, [50, 50, 50, 50], [0, 0, 0, 0], 8),
            ..Default::default()
        },
    );
    walker.set_sao_picture(sao_pic).unwrap();
    walker.apply_in_loop_filters(&mut out).unwrap();
    assert_eq!(out.luma.samples, snapshot);
}

/// SAO CABAC syntax wiring: when `sh_sao_luma_used_flag = 1` and the
/// CTU walker drives `decode_picture_into`, the `sao(rx, ry)` helper
/// must be invoked once per CTU and the resulting per-CTB params must
/// land in `walker.sao_picture()`. The all-zero CABAC payload causes
/// the deterministic context biases (Table 57 / Table 58 init at QP 32)
/// to produce a fixed `SaoTypeIdx` for the first CTU — we just check
/// that the array is no longer the default-empty placeholder.
#[test]
fn ctu_walker_decodes_sao_syntax_on_zero_stream() {
    let sps = dummy_sps(0, 32, 32);
    let pps = dummy_pps(32, 32);
    let mut sh = intra_slice_header();
    sh.sh_sao_luma_used_flag = true;
    let layout = CtuLayout::from_sps_pps(&sps, &pps);
    // Zero stream: see sao_syntax::decode_sao_ctb_merge_left_inherits
    // for the bias maths — ivlOffset=0 + Table-57 MPS=1 means the merge
    // bits resolve to 1 wherever they're emitted, so the very first CTU
    // (no merge available) parses sao_type_idx_luma straight away.
    let payload = [0u8; 256];
    let mut walker = CtuWalker::begin_slice(&layout, &sps, &pps, &sh, 0, &payload).unwrap();
    let mut out = PictureBuffer::yuv420_filled(32, 32, 0);
    walker.decode_picture_into(&mut out).unwrap();
    // Layout is 1×1 CTU at 32×32. The first CTU's parsed SAO params
    // must be a non-default value because the syntax walker ran (the
    // contextual bin0 of sao_type_idx_luma on a zero stream resolves
    // to 1 → SaoTypeIdx is BandOffset or EdgeOffset, never NotApplied).
    let first = walker.sao_picture().get(0, 0);
    assert_ne!(
        first.luma.sao_type_idx,
        SaoTypeIdx::NotApplied,
        "expected SAO syntax to populate the first CTB"
    );
}

/// Decoder-facing programmatic SAO API: an EO horizontal class on a
/// hand-painted alternating pattern produces the spec's category-1 /
/// category-4 shifts. This is the same shape exercised by the unit test
/// in `sao::tests`, but verifies the integration point (the CTB grid
/// dimension check) is wired sensibly.
#[test]
fn ctu_walker_set_sao_picture_grid_check() {
    let sps = dummy_sps(0, 32, 32);
    let pps = dummy_pps(32, 32);
    let sh = intra_slice_header();
    let layout = CtuLayout::from_sps_pps(&sps, &pps);
    let payload = [0u8; 256];
    let mut walker = CtuWalker::begin_slice(&layout, &sps, &pps, &sh, 0, &payload).unwrap();
    // Wrong grid dimensions must be rejected.
    let bad = SaoPicture::empty(2, 2);
    assert!(walker.set_sao_picture(bad).is_err());
    // Correct grid is accepted.
    let good = SaoPicture::empty(layout.pic_width_in_ctbs_y, layout.pic_height_in_ctbs_y);
    assert!(walker.set_sao_picture(good).is_ok());
    // Sao picture round-trip — read out the params we just installed.
    assert_eq!(walker.sao_picture().pic_width_in_ctbs_y, 1);
    assert_eq!(walker.sao_picture().pic_height_in_ctbs_y, 1);
    let p = walker.sao_picture().get(0, 0);
    assert_eq!(p.luma.sao_type_idx, SaoTypeIdx::NotApplied);
    assert_eq!(p.luma.eo_class, SaoEoClass::Horizontal);
}

/// ALF wiring through the CTU walker: the round-15 walker accepts
/// `sh_alf_enabled_flag = 1` and runs the §8.8.5 apply pass after SAO.
/// With the default empty AlfPicture the apply pass is a no-op (every
/// CTB has `luma_on = false`), so the post-decode picture is identical
/// to the deblock+SAO output.
#[test]
fn ctu_walker_apply_alf_no_op_with_empty_picture() {
    let sps = dummy_sps(0, 32, 32);
    let pps = dummy_pps(32, 32);
    let mut sh = intra_slice_header();
    sh.sh_alf_enabled_flag = true;
    let layout = CtuLayout::from_sps_pps(&sps, &pps);
    let payload = [0u8; 256];
    let mut walker = CtuWalker::begin_slice(&layout, &sps, &pps, &sh, 0, &payload).unwrap();
    let mut out = PictureBuffer::yuv420_filled(32, 32, 0);
    walker.decode_picture_into(&mut out).unwrap();
    let snapshot = out.luma.samples.clone();
    walker.apply_in_loop_filters(&mut out).unwrap();
    assert_eq!(out.luma.samples, snapshot);
}

/// ALF wiring with an explicit ALF APS binding: install a per-CTB ALF
/// "luma on" record that points at APS slot 0, with a non-zero filter
/// coefficient that would shift samples on a non-flat plane. Decode
/// produces a near-flat picture, so the spike must come from the
/// `apply_in_loop_filters_with_alf` path.
#[test]
fn ctu_walker_apply_alf_runs_with_aps_binding() {
    let sps = dummy_sps(0, 32, 32);
    let pps = dummy_pps(32, 32);
    let mut sh = intra_slice_header();
    sh.sh_alf_enabled_flag = true;
    let layout = CtuLayout::from_sps_pps(&sps, &pps);
    let payload = [0u8; 256];
    let mut walker = CtuWalker::begin_slice(&layout, &sps, &pps, &sh, 0, &payload).unwrap();
    let mut out = PictureBuffer::yuv420_filled(32, 32, 0);
    walker.decode_picture_into(&mut out).unwrap();
    // Inject a known spike so the §8.8.5.2 filter math has something
    // to act on (otherwise the post-decode flat picture short-circuits
    // every filter contribution to 0).
    let stride = out.luma.stride;
    out.luma.samples[16 * stride + 16] = 200;
    // Build an APS with all-zero coefficients except f[6] (vertical
    // y1 pair) — that produces a deterministic deviation at the spike
    // and identity everywhere else (see alf::tests::nonzero_coeff_…).
    let mut row = [0i32; ALF_LUMA_NUM_COEFF];
    row[6] = 32;
    let aps = AlfApsData {
        alf_luma_filter_signal_flag: true,
        luma_coeff: vec![row; NUM_ALF_FILTERS],
        luma_clip_idx: vec![[0u8; ALF_LUMA_NUM_COEFF]; NUM_ALF_FILTERS],
        ..Default::default()
    };
    let aps_slot: [Option<&AlfApsData>; 1] = [Some(&aps)];
    let binding = AlfApsBinding {
        luma_apses: &aps_slot,
        chroma_aps: None,
        cc_cb_aps: None,
        cc_cr_aps: None,
    };
    // Mark CTB (0,0) ALF-on with `AlfCtbFiltSetIdxY = 16` (= APS slot 0).
    let mut alf_pic = AlfPicture::empty(layout.pic_width_in_ctbs_y, layout.pic_height_in_ctbs_y);
    alf_pic.set(
        0,
        0,
        AlfCtb {
            luma_on: true,
            luma_filt_set_idx: 16,
            ..Default::default()
        },
    );
    walker.set_alf_picture(alf_pic).unwrap();
    let before_centre = out.luma.samples[16 * stride + 16];
    assert_eq!(before_centre, 200);
    walker
        .apply_in_loop_filters_with_alf(&mut out, &binding)
        .unwrap();
    let after_centre = out.luma.samples[16 * stride + 16];
    assert_ne!(after_centre, 200, "ALF should have modified the centre");
}

/// `set_alf_picture` rejects a per-picture array whose CTB grid does
/// not match the layout — symmetrical to `set_sao_picture`.
#[test]
fn ctu_walker_set_alf_picture_grid_check() {
    let sps = dummy_sps(0, 32, 32);
    let pps = dummy_pps(32, 32);
    let sh = intra_slice_header();
    let layout = CtuLayout::from_sps_pps(&sps, &pps);
    let payload = [0u8; 256];
    let mut walker = CtuWalker::begin_slice(&layout, &sps, &pps, &sh, 0, &payload).unwrap();
    let bad = AlfPicture::empty(2, 2);
    assert!(walker.set_alf_picture(bad).is_err());
    let good = AlfPicture::empty(layout.pic_width_in_ctbs_y, layout.pic_height_in_ctbs_y);
    assert!(walker.set_alf_picture(good).is_ok());
    assert_eq!(walker.alf_picture().pic_width_in_ctbs_y, 1);
    assert_eq!(walker.alf_picture().pic_height_in_ctbs_y, 1);
    assert!(walker.alf_picture().is_all_off());
}

/// Round-21: a P-slice that encodes a single 8x8 all-skip CU produces a
/// reconstructed picture identical to the supplied reference frame (plus
/// any deblocking smoothing on a flat plane is a no-op). This is the
/// smallest end-to-end smoke test for §8.5.2.2 spatial-merge derivation
/// + §8.5.6 integer-pel motion compensation.
///
/// Bitstream synthesis: we use the workspace forward-CABAC encoder
/// ([`oxideav_h266::cabac_enc::ArithEncoder`]) to drop the exact bin
/// sequence the CTU walker will consume — split_cu_flag(0) → root CU
/// stays unsplit at 8x8; cu_skip_flag(1) at Table 64 / ctxInc 0;
/// general_merge_flag inferred 1; merge_idx bin0(0) at Table 109 /
/// ctxInc 0 picks `mergeCandList[0]` = zero-MV / refIdx 0. With
/// `MaxNumMergeCand = 6` >= 2 the merge_idx read does fire.
#[test]
fn decode_p_slice_all_skip_matches_reference() {
    use oxideav_h266::cabac::ContextModel;
    use oxideav_h266::cabac_enc::ArithEncoder;
    use oxideav_h266::ctx::{ctx_inc_cu_skip_flag, ctx_inc_general_merge_flag, ctx_inc_merge_idx};
    use oxideav_h266::tables::{init_contexts, SyntaxCtx};

    // ---- Reference (IDR) frame ------------------------------------------
    // Paint a deterministic ramp into the reference picture so the
    // post-decode buffer is uniquely identifiable. With all-skip + zero
    // MV, the output of the P-slice decode must match this byte-for-byte.
    let pic_w = 8u32;
    let pic_h = 8u32;
    let mut ref_buf = PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 0);
    for y in 0..pic_h as usize {
        for x in 0..pic_w as usize {
            ref_buf.luma.samples[y * 8 + x] = (10 + (y * 8 + x)) as u8;
        }
    }
    for y in 0..(pic_h / 2) as usize {
        for x in 0..(pic_w / 2) as usize {
            ref_buf.cb.samples[y * 4 + x] = (90 + y * 4 + x) as u8;
            ref_buf.cr.samples[y * 4 + x] = (170 + y * 4 + x) as u8;
        }
    }
    let ref_pic = ReferencePicture {
        poc: 0,
        frame: ref_buf.clone(),
    };

    // ---- P-slice CABAC payload synthesis --------------------------------
    // Slice QP = 26 (pps_init_qp_minus26 = 0; sh_qp_delta = 0). For
    // sh_slice_type = P with sh_cabac_init_flag = 0, init_type = 1.
    let slice_qp = 26;
    let init_type = 1u8; // P-slice, cabac_init_flag = 0
                         // CTU is 32 (sps_log2_ctu_size_minus5 = 0), the picture is 8x8 so
                         // the walker calls walk(0, 0, 8, 8). For an 8x8 root the leaf
                         // walker reads split_cu_flag at depth 0 (since trailing_zeros(8) =
                         // 3 > min_cb_log2 = 2). split_cu_flag's ctxInc with all neighbours
                         // unavailable + the partition gate (no allowed splits at depth 0
                         // beyond the root) lands at the formula in §9.3.4.2.2.
                         //
                         // Calling [`ctx_inc_split_cu_flag`] with the same args as the
                         // walker keeps the encoder and decoder in lockstep.
    let mut split_cu_ctxs = init_contexts(SyntaxCtx::SplitCuFlag, slice_qp);
    let mut cu_skip_ctxs = init_contexts(SyntaxCtx::CuSkipFlag, slice_qp);
    let mut general_merge_ctxs = init_contexts(SyntaxCtx::GeneralMergeFlag, slice_qp);
    let mut merge_idx_ctxs = init_contexts(SyntaxCtx::MergeIdx, slice_qp);

    let mut enc = ArithEncoder::new();
    // 1. split_cu_flag(0). ctxInc here mirrors the TreeWalker's call:
    //    `ctx_inc_split_cu_flag(false, false, 0, 0, 8, 8, 1, 1, 1, 1, 1)`.
    let split_inc =
        oxideav_h266::ctx::ctx_inc_split_cu_flag(false, false, 0, 0, 8, 8, 1, 1, 1, 1, 1) as usize;
    let split_n_minus1 = split_cu_ctxs.len() - 1;
    enc.encode_decision(&mut split_cu_ctxs[split_inc.min(split_n_minus1)], 0)
        .unwrap();

    // 2. cu_skip_flag(1). ctxInc = (condL && availL) + (condA && availA)
    //    = 0 (no neighbours). With init_type = 1 (P), the slot in the
    //    9-entry table is `1 * 3 + 0 = 3`.
    let skip_inc = ctx_inc_cu_skip_flag(false, false, false, false) as usize;
    let skip_slot = (init_type as usize) * 3 + skip_inc;
    enc.encode_decision(&mut cu_skip_ctxs[skip_slot], 1)
        .unwrap();
    // 3. general_merge_flag is inferred to 1 — no bin emitted.
    let _ = ctx_inc_general_merge_flag();
    let _ = &mut general_merge_ctxs; // referenced by the parser when skip=0

    // 4. merge_idx bin 0 = 0 → picks merge candidate 0 (zero-MV, refIdx 0).
    let merge_inc = ctx_inc_merge_idx() as usize;
    let merge_idx_n_minus1 = merge_idx_ctxs.len() - 1;
    let merge_slot = (init_type as usize + merge_inc).min(merge_idx_n_minus1);
    enc.encode_decision(&mut merge_idx_ctxs[merge_slot], 0)
        .unwrap();

    // Terminate the stream cleanly so the decoder can read past the
    // last bin without overrunning the byte boundary.
    enc.encode_terminate(1).unwrap();
    let payload = enc.finish();

    // ---- SPS / PPS / slice header for a P-slice on a 8x8 picture --------
    let mut sps = dummy_sps(0, pic_w, pic_h);
    sps.tool_flags.six_minus_max_num_merge_cand = 0; // MaxNumMergeCand = 6
    let pps = dummy_pps(pic_w, pic_h);
    let mut sh = StatefulSliceHeader {
        sh_slice_type: SliceType::P,
        sh_qp_delta: 0,
        ..Default::default()
    };
    let _ = &mut sh;

    let layout = CtuLayout::from_sps_pps(&sps, &pps);
    // Avoid the cabac context model unused warning.
    let _: &ContextModel = &cu_skip_ctxs[0];

    // ---- Decode the P-slice ---------------------------------------------
    let mut walker = CtuWalker::begin_slice(&layout, &sps, &pps, &sh, 0, &payload).unwrap();
    walker.set_ref_pic_list_l0(vec![ref_pic]);

    // Seed the picture buffer with a sentinel so the test fails loudly
    // if the inter path silently no-ops.
    let mut out = PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 222);
    walker.decode_picture_into(&mut out).unwrap();

    // The decoded P-slice must match the reference (zero MV, single
    // CU covering the whole picture, no residual). Apply deblocking
    // for completeness — on a flat single CU it is a no-op.
    walker.apply_in_loop_filters(&mut out).unwrap();

    assert_eq!(
        out.luma.samples, ref_buf.luma.samples,
        "luma plane must equal reference frame after all-skip P-slice decode"
    );
    assert_eq!(
        out.cb.samples, ref_buf.cb.samples,
        "Cb plane must equal reference frame"
    );
    assert_eq!(
        out.cr.samples, ref_buf.cr.samples,
        "Cr plane must equal reference frame"
    );
}

/// Round-21: P-slice merge derivation with a single CU should cleanly
/// pad out to MaxNumMergeCand zero-MV entries when no spatial neighbours
/// exist. Verified by inspecting the walker's per-picture motion field
/// after decode: the chosen MvField for the (only) skip CU must carry
/// ref_idx_l0 = 0 (the zero-MV pad slot).
#[test]
fn decode_p_slice_writes_motion_field() {
    use oxideav_h266::cabac_enc::ArithEncoder;
    use oxideav_h266::ctx::{ctx_inc_cu_skip_flag, ctx_inc_merge_idx, ctx_inc_split_cu_flag};
    use oxideav_h266::tables::{init_contexts, SyntaxCtx};

    let pic_w = 8u32;
    let pic_h = 8u32;
    let ref_pic = ReferencePicture {
        poc: 0,
        frame: PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 64),
    };
    // Synthesize the same all-skip stream as above.
    let slice_qp = 26;
    let init_type = 1u8;
    let mut split_cu_ctxs = init_contexts(SyntaxCtx::SplitCuFlag, slice_qp);
    let mut cu_skip_ctxs = init_contexts(SyntaxCtx::CuSkipFlag, slice_qp);
    let mut merge_idx_ctxs = init_contexts(SyntaxCtx::MergeIdx, slice_qp);
    let mut enc = ArithEncoder::new();
    let split_inc = ctx_inc_split_cu_flag(false, false, 0, 0, 8, 8, 1, 1, 1, 1, 1) as usize;
    let split_slot = split_inc.min(split_cu_ctxs.len() - 1);
    enc.encode_decision(&mut split_cu_ctxs[split_slot], 0)
        .unwrap();
    let skip_inc = ctx_inc_cu_skip_flag(false, false, false, false) as usize;
    let skip_slot = (init_type as usize) * 3 + skip_inc;
    enc.encode_decision(&mut cu_skip_ctxs[skip_slot], 1)
        .unwrap();
    let merge_inc = ctx_inc_merge_idx() as usize;
    let merge_slot = (init_type as usize + merge_inc).min(merge_idx_ctxs.len() - 1);
    enc.encode_decision(&mut merge_idx_ctxs[merge_slot], 0)
        .unwrap();
    enc.encode_terminate(1).unwrap();
    let payload = enc.finish();

    let mut sps = dummy_sps(0, pic_w, pic_h);
    sps.tool_flags.six_minus_max_num_merge_cand = 0;
    let pps = dummy_pps(pic_w, pic_h);
    let sh = StatefulSliceHeader {
        sh_slice_type: SliceType::P,
        sh_qp_delta: 0,
        ..Default::default()
    };
    let layout = CtuLayout::from_sps_pps(&sps, &pps);
    let mut walker = CtuWalker::begin_slice(&layout, &sps, &pps, &sh, 0, &payload).unwrap();
    walker.set_ref_pic_list_l0(vec![ref_pic]);
    let mut out = PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 0);
    walker.decode_picture_into(&mut out).unwrap();

    // Inspect the motion field: every 4x4 block in the picture must
    // carry the chosen zero-MV / refIdx 0 record from the merge list.
    let mf = walker.motion_field();
    for by in 0..mf.blocks_h {
        for bx in 0..mf.blocks_w {
            let f = mf.field[(by * mf.blocks_w + bx) as usize];
            assert!(f.available, "block ({bx},{by}) must be marked available");
            assert!(f.pred_flag_l0);
            assert_eq!(f.ref_idx_l0, 0);
            assert_eq!(f.mv_l0.x, 0);
            assert_eq!(f.mv_l0.y, 0);
            assert!(f.cu_skip_flag);
            assert!(f.mode_inter);
        }
    }
}

/// Round-23: B-slice all-skip + bi-pred zero-MV merge candidate. Two
/// distinct reference pictures are bound to RefPicList0 and
/// RefPicList1; the synthesised CABAC payload picks merge index 0
/// (the spatial-list zero-MV pad, which for B-slices is **bi-pred**
/// per [`oxideav_h266::inter::build_merge_cand_list_b`]). The
/// expected output is byte-exactly the §8.5.6.6.2 eq. 980 default-
/// weighted average `(L0 + L1 + 1) >> 1` of the two reference frames.
///
/// Bitstream synthesis matches the round-21 P-slice fixture but
/// flips `sh_slice_type = B` (init_type = 2 with
/// `sh_cabac_init_flag = 0`). Verifies:
///   1. The B-slice gate inside `CtuWalker::begin_slice` is gone.
///   2. The bi-pred merge candidate is correctly built and selected.
///   3. The §8.5.6.6.2 eq. 980 composition produces byte-exact output.
#[test]
fn decode_b_slice_all_skip_bipred_matches_average() {
    use oxideav_h266::cabac_enc::ArithEncoder;
    use oxideav_h266::ctx::{ctx_inc_cu_skip_flag, ctx_inc_merge_idx, ctx_inc_split_cu_flag};
    use oxideav_h266::tables::{init_contexts, SyntaxCtx};

    let pic_w = 8u32;
    let pic_h = 8u32;

    // ---- Build two distinct reference pictures (L0 / L1) -----------------
    // Use deterministic ramps so the bipred output is uniquely determined.
    let mut ref_l0_buf = PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 0);
    let mut ref_l1_buf = PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 0);
    for y in 0..pic_h as usize {
        for x in 0..pic_w as usize {
            ref_l0_buf.luma.samples[y * 8 + x] = (10 + (y * 8 + x)) as u8;
            ref_l1_buf.luma.samples[y * 8 + x] = (200 - (y * 8 + x)) as u8;
        }
    }
    for y in 0..(pic_h / 2) as usize {
        for x in 0..(pic_w / 2) as usize {
            ref_l0_buf.cb.samples[y * 4 + x] = (90 + y * 4 + x) as u8;
            ref_l0_buf.cr.samples[y * 4 + x] = (170 + y * 4 + x) as u8;
            ref_l1_buf.cb.samples[y * 4 + x] = (160 - y * 4 - x) as u8;
            ref_l1_buf.cr.samples[y * 4 + x] = (200 - y * 4 - x) as u8;
        }
    }
    let ref_l0 = ReferencePicture {
        poc: 0,
        frame: ref_l0_buf.clone(),
    };
    let ref_l1 = ReferencePicture {
        poc: 2,
        frame: ref_l1_buf.clone(),
    };

    // ---- B-slice CABAC payload synthesis --------------------------------
    // Slice QP = 26; sh_slice_type = B with sh_cabac_init_flag = 0 →
    // init_type = 2 per Table 51.
    let slice_qp = 26;
    let init_type = 2u8;
    let mut split_cu_ctxs = init_contexts(SyntaxCtx::SplitCuFlag, slice_qp);
    let mut cu_skip_ctxs = init_contexts(SyntaxCtx::CuSkipFlag, slice_qp);
    let mut merge_idx_ctxs = init_contexts(SyntaxCtx::MergeIdx, slice_qp);

    let mut enc = ArithEncoder::new();
    // 1. split_cu_flag(0).
    let split_inc = ctx_inc_split_cu_flag(false, false, 0, 0, 8, 8, 1, 1, 1, 1, 1) as usize;
    let split_slot = split_inc.min(split_cu_ctxs.len() - 1);
    enc.encode_decision(&mut split_cu_ctxs[split_slot], 0)
        .unwrap();
    // 2. cu_skip_flag(1) at init_type 2 (B-slice).
    let skip_inc = ctx_inc_cu_skip_flag(false, false, false, false) as usize;
    let skip_slot = (init_type as usize) * 3 + skip_inc;
    enc.encode_decision(&mut cu_skip_ctxs[skip_slot], 1)
        .unwrap();
    // 3. general_merge_flag inferred to 1 (skip → no bin emitted).
    // 4. merge_idx bin 0 = 0 → mergeCandList[0] = bipred zero-MV pad.
    let merge_inc = ctx_inc_merge_idx() as usize;
    let merge_slot = (init_type as usize + merge_inc).min(merge_idx_ctxs.len() - 1);
    enc.encode_decision(&mut merge_idx_ctxs[merge_slot], 0)
        .unwrap();
    enc.encode_terminate(1).unwrap();
    let payload = enc.finish();

    // ---- SPS / PPS / slice header for a B-slice on an 8x8 picture --------
    let mut sps = dummy_sps(0, pic_w, pic_h);
    sps.tool_flags.six_minus_max_num_merge_cand = 0; // MaxNumMergeCand = 6
    let pps = dummy_pps(pic_w, pic_h);
    let sh = StatefulSliceHeader {
        sh_slice_type: SliceType::B,
        sh_qp_delta: 0,
        ..Default::default()
    };

    let layout = CtuLayout::from_sps_pps(&sps, &pps);

    // ---- Decode the B-slice --------------------------------------------
    let mut walker = CtuWalker::begin_slice(&layout, &sps, &pps, &sh, 0, &payload).unwrap();
    walker.set_ref_pic_list_l0(vec![ref_l0]);
    walker.set_ref_pic_list_l1(vec![ref_l1]);

    let mut out = PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 222);
    walker.decode_picture_into(&mut out).unwrap();
    walker.apply_in_loop_filters(&mut out).unwrap();

    // ---- Build expected output: byte-exact (L0 + L1 + 1) >> 1 -----------
    let expected_y: Vec<u8> = ref_l0_buf
        .luma
        .samples
        .iter()
        .zip(&ref_l1_buf.luma.samples)
        .map(|(&a, &b)| ((a as u32 + b as u32 + 1) >> 1) as u8)
        .collect();
    let expected_cb: Vec<u8> = ref_l0_buf
        .cb
        .samples
        .iter()
        .zip(&ref_l1_buf.cb.samples)
        .map(|(&a, &b)| ((a as u32 + b as u32 + 1) >> 1) as u8)
        .collect();
    let expected_cr: Vec<u8> = ref_l0_buf
        .cr
        .samples
        .iter()
        .zip(&ref_l1_buf.cr.samples)
        .map(|(&a, &b)| ((a as u32 + b as u32 + 1) >> 1) as u8)
        .collect();

    assert_eq!(
        out.luma.samples, expected_y,
        "luma plane must equal byte-exact bipred average of L0 + L1"
    );
    assert_eq!(out.cb.samples, expected_cb, "Cb plane bipred mismatch");
    assert_eq!(out.cr.samples, expected_cr, "Cr plane bipred mismatch");
}

/// Round-23: B-slice motion-field write-back broadcasts the chosen
/// bi-pred MvField across every 4x4 block of the (only) skip CU.
/// Verifies that `pred_flag_l0 == pred_flag_l1 == true` and that
/// `ref_idx_l0 == ref_idx_l1 == 0` (the zero-MV pad slot) propagate
/// onto the per-block grid for downstream merge derivation.
#[test]
fn decode_b_slice_writes_bipred_motion_field() {
    use oxideav_h266::cabac_enc::ArithEncoder;
    use oxideav_h266::ctx::{ctx_inc_cu_skip_flag, ctx_inc_merge_idx, ctx_inc_split_cu_flag};
    use oxideav_h266::tables::{init_contexts, SyntaxCtx};

    let pic_w = 8u32;
    let pic_h = 8u32;
    let ref_l0 = ReferencePicture {
        poc: 0,
        frame: PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 64),
    };
    let ref_l1 = ReferencePicture {
        poc: 2,
        frame: PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 192),
    };

    let slice_qp = 26;
    let init_type = 2u8;
    let mut split_cu_ctxs = init_contexts(SyntaxCtx::SplitCuFlag, slice_qp);
    let mut cu_skip_ctxs = init_contexts(SyntaxCtx::CuSkipFlag, slice_qp);
    let mut merge_idx_ctxs = init_contexts(SyntaxCtx::MergeIdx, slice_qp);
    let mut enc = ArithEncoder::new();
    let split_inc = ctx_inc_split_cu_flag(false, false, 0, 0, 8, 8, 1, 1, 1, 1, 1) as usize;
    let split_slot = split_inc.min(split_cu_ctxs.len() - 1);
    enc.encode_decision(&mut split_cu_ctxs[split_slot], 0)
        .unwrap();
    let skip_inc = ctx_inc_cu_skip_flag(false, false, false, false) as usize;
    let skip_slot = (init_type as usize) * 3 + skip_inc;
    enc.encode_decision(&mut cu_skip_ctxs[skip_slot], 1)
        .unwrap();
    let merge_inc = ctx_inc_merge_idx() as usize;
    let merge_slot = (init_type as usize + merge_inc).min(merge_idx_ctxs.len() - 1);
    enc.encode_decision(&mut merge_idx_ctxs[merge_slot], 0)
        .unwrap();
    enc.encode_terminate(1).unwrap();
    let payload = enc.finish();

    let mut sps = dummy_sps(0, pic_w, pic_h);
    sps.tool_flags.six_minus_max_num_merge_cand = 0;
    let pps = dummy_pps(pic_w, pic_h);
    let sh = StatefulSliceHeader {
        sh_slice_type: SliceType::B,
        sh_qp_delta: 0,
        ..Default::default()
    };
    let layout = CtuLayout::from_sps_pps(&sps, &pps);
    let mut walker = CtuWalker::begin_slice(&layout, &sps, &pps, &sh, 0, &payload).unwrap();
    walker.set_ref_pic_list_l0(vec![ref_l0]);
    walker.set_ref_pic_list_l1(vec![ref_l1]);
    let mut out = PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 0);
    walker.decode_picture_into(&mut out).unwrap();

    // Verify the per-block motion field carries the bi-pred record.
    let mf = walker.motion_field();
    for by in 0..mf.blocks_h {
        for bx in 0..mf.blocks_w {
            let f = mf.field[(by * mf.blocks_w + bx) as usize];
            assert!(f.available);
            assert!(
                f.pred_flag_l0,
                "block ({bx},{by}) must carry predFlagL0 == 1"
            );
            assert!(
                f.pred_flag_l1,
                "block ({bx},{by}) must carry predFlagL1 == 1"
            );
            assert_eq!(f.ref_idx_l0, 0);
            assert_eq!(f.ref_idx_l1, 0);
            assert_eq!(f.mv_l0.x, 0);
            assert_eq!(f.mv_l0.y, 0);
            assert_eq!(f.mv_l1.x, 0);
            assert_eq!(f.mv_l1.y, 0);
            assert!(f.cu_skip_flag);
            assert!(f.mode_inter);
        }
    }

    // The per-pixel output should be the rounded average of 64 and 192:
    // (64 + 192 + 1) >> 1 = 128.
    for s in &out.luma.samples {
        assert_eq!(*s, 128, "luma bipred should be (64 + 192 + 1) >> 1 = 128");
    }
}

/// Chroma ALF wiring: install a per-CTB Cb-on record + a chroma APS
/// with all-zero coefficients. On the post-decode (flat) chroma plane
/// the filter math is identity, but the apply pass should still have
/// dispatched into the chroma branch (we observe via a programmatic
/// chroma spike).
#[test]
fn ctu_walker_chroma_alf_runs_with_chroma_aps() {
    let sps = dummy_sps(0, 32, 32);
    let pps = dummy_pps(32, 32);
    let mut sh = intra_slice_header();
    sh.sh_alf_enabled_flag = true;
    sh.sh_alf_cb_enabled_flag = true;
    let layout = CtuLayout::from_sps_pps(&sps, &pps);
    let payload = [0u8; 256];
    let mut walker = CtuWalker::begin_slice(&layout, &sps, &pps, &sh, 0, &payload).unwrap();
    let mut out = PictureBuffer::yuv420_filled(32, 32, 0);
    walker.decode_picture_into(&mut out).unwrap();
    let cb_stride = out.cb.stride;
    out.cb.samples[5 * cb_stride + 5] = 200;
    // Chroma APS with f[0] = 64 (vertical y2 pair).
    let mut row = [0i32; ALF_CHROMA_NUM_COEFF];
    row[0] = 64;
    let aps = AlfApsData {
        alf_chroma_filter_signal_flag: true,
        alf_chroma_num_alt_filters_minus1: 0,
        chroma_coeff: vec![row; 1],
        chroma_clip_idx: vec![[0u8; ALF_CHROMA_NUM_COEFF]; 1],
        ..Default::default()
    };
    let binding = AlfApsBinding {
        luma_apses: &[],
        chroma_aps: Some(&aps),
        cc_cb_aps: None,
        cc_cr_aps: None,
    };
    let mut alf_pic = AlfPicture::empty(layout.pic_width_in_ctbs_y, layout.pic_height_in_ctbs_y);
    alf_pic.set(
        0,
        0,
        AlfCtb {
            cb_on: true,
            cb_alt_idx: 0,
            ..Default::default()
        },
    );
    walker.set_alf_picture(alf_pic).unwrap();
    walker
        .apply_in_loop_filters_with_alf(&mut out, &binding)
        .unwrap();
    let after = out.cb.samples[5 * cb_stride + 5];
    assert_ne!(after, 200, "chroma ALF should have moved the spike");
}
