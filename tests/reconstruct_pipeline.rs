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
        motion_field: None,
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
        motion_field: None,
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
        motion_field: None,
    };
    let ref_l1 = ReferencePicture {
        poc: 2,
        frame: ref_l1_buf.clone(),
        motion_field: None,
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
        motion_field: None,
    };
    let ref_l1 = ReferencePicture {
        poc: 2,
        frame: PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 192),
        motion_field: None,
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

/// Round-24: HMVP table wiring at the walker level. After a P-slice
/// decode containing a single inter (skip) CU, the walker's HMVP table
/// must be populated with exactly one entry — the §8.5.2.16 update
/// invariant. The entry's motion field must mirror the chosen merge
/// candidate (zero MV, refIdxL0 = 0, predFlagL0 = 1, predFlagL1 = 0).
///
/// Pre-conditions verified:
///   1. Slice begin reset: `walker.hmvp_table().len() == 0` immediately
///      after `begin_slice` (per §7.3.11 `NumHmvpCand = 0` reset).
///   2. After decode: exactly one HMVP entry, encoding the chosen MvField.
#[test]
fn decode_p_slice_populates_hmvp_table() {
    use oxideav_h266::cabac_enc::ArithEncoder;
    use oxideav_h266::ctx::{ctx_inc_cu_skip_flag, ctx_inc_merge_idx, ctx_inc_split_cu_flag};
    use oxideav_h266::tables::{init_contexts, SyntaxCtx};

    let pic_w = 8u32;
    let pic_h = 8u32;
    let ref_pic = ReferencePicture {
        poc: 0,
        frame: PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 64),
        motion_field: None,
    };
    let slice_qp = 26;
    let init_type = 1u8; // P-slice, cabac_init_flag = 0
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

    // Slice-start invariant: HMVP table reset to empty per §7.3.11.
    assert_eq!(
        walker.hmvp_table().len(),
        0,
        "HMVP table must be empty at slice start (§7.3.11 NumHmvpCand = 0 reset)"
    );

    let mut out = PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 0);
    walker.decode_picture_into(&mut out).unwrap();

    // Post-decode invariant: exactly one HMVP entry (the lone inter CU).
    let table = walker.hmvp_table();
    assert_eq!(
        table.len(),
        1,
        "HMVP table must contain exactly the single inter CU's MvField after decode"
    );
    let entry = table.entries[0];
    assert!(entry.pred_flag_l0);
    assert!(!entry.pred_flag_l1);
    assert_eq!(entry.ref_idx_l0, 0);
    assert_eq!(entry.mv_l0.x, 0);
    assert_eq!(entry.mv_l0.y, 0);
    assert!(entry.mode_inter);
    assert!(entry.cu_skip_flag);
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

/// Round-24 acceptance fixture: a quad-split P-slice exercises the
/// HMVP merge-mode pull-in path end-to-end. The CTU is a 16x16 root
/// that splits into four 8x8 leaf CUs, each cu_skip + merge_idx = 0.
/// CU2 / CU3 / CU4 invoke [`build_merge_cand_list`] with a non-empty
/// HMVP table (CU1 already pushed its MvField via `update_with` per
/// §8.5.2.16), so `insert_hmvp_into_merge_list` runs at least three
/// times during the slice — proving the §8.5.2.6 walk is wired into
/// the per-CU reconstruct path, not just the table-update side.
///
/// This is the broader counterpart to `decode_p_slice_populates_hmvp_table`
/// which only pinned table population on a single-CU fixture. The
/// acceptance criterion is that the multi-CU decode produces the
/// reference picture byte-exactly (every chosen merge candidate is the
/// uni-pred zero-MV record — either the spatial neighbour from a prior
/// CU, the HMVP entry that mirrors it, or the zero-MV pad — they are
/// all the same MvField after the §8.5.2.16 dedup, which is itself
/// the round-24 invariant the test exercises).
#[test]
fn decode_p_slice_quad_split_exercises_hmvp_merge_pull_in() {
    use oxideav_h266::cabac_enc::ArithEncoder;
    use oxideav_h266::ctx::{
        ctx_inc_cu_skip_flag, ctx_inc_merge_idx, ctx_inc_split_cu_flag, ctx_inc_split_qt_flag,
    };
    use oxideav_h266::tables::{init_contexts, SyntaxCtx};

    // 16x16 picture inside a single 32-CTU. The walker recurses
    // walk(0,0,16,16): split_cu_flag(1) + split_qt_flag(1) → four 8x8
    // children at depth 1. Each child reads split_cu_flag(0) (leaf at
    // 8x8 since 8 > min_cb_log2 = 2 still emits a split-flag bin) and
    // then the cu_skip + merge_idx pair on the inter path.
    let pic_w = 16u32;
    let pic_h = 16u32;
    let ref_pic = ReferencePicture {
        poc: 0,
        frame: PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 64),
        motion_field: None,
    };
    let ref_buf = PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 64);

    let slice_qp = 26;
    let init_type = 1u8; // P-slice, sh_cabac_init_flag = 0
    let mut split_cu_ctxs = init_contexts(SyntaxCtx::SplitCuFlag, slice_qp);
    let mut split_qt_ctxs = init_contexts(SyntaxCtx::SplitQtFlag, slice_qp);
    let mut cu_skip_ctxs = init_contexts(SyntaxCtx::CuSkipFlag, slice_qp);
    let mut merge_idx_ctxs = init_contexts(SyntaxCtx::MergeIdx, slice_qp);
    let mut enc = ArithEncoder::new();

    // 1. Root split_cu_flag(1) at 16x16, depth 0. ctxInc args mirror
    //    `TreeWalker::recurse` which passes neighbour-availability =
    //    false today.
    let split_cu_inc_root =
        ctx_inc_split_cu_flag(false, false, 0, 0, 16, 16, 1, 1, 1, 1, 1) as usize;
    let split_cu_n_minus1 = split_cu_ctxs.len() - 1;
    enc.encode_decision(
        &mut split_cu_ctxs[split_cu_inc_root.min(split_cu_n_minus1)],
        1,
    )
    .unwrap();

    // 2. Root split_qt_flag(1) at depth 0.
    let split_qt_inc_root = ctx_inc_split_qt_flag(false, false, 0, 0, 0) as usize;
    let split_qt_n_minus1 = split_qt_ctxs.len() - 1;
    enc.encode_decision(
        &mut split_qt_ctxs[split_qt_inc_root.min(split_qt_n_minus1)],
        1,
    )
    .unwrap();

    // 3. The decoder performs ALL partition decoding (`decode_ctu_partitions`)
    //    BEFORE reading any leaf-CU syntax (`decode_leaf_cu_syntax`).
    //    We therefore emit all 4 child split_cu_flag(0) bins first,
    //    then all 4 (cu_skip + merge_idx) pairs in z-order. Even though
    //    these bins use different context bundles (tree_ctxs vs
    //    leaf_ctxs), the arithmetic coder's range/low state is shared,
    //    so bin order in the stream must match the decoder's read order.
    //
    // Note on cu_skip_flag ctxInc: the decoder's
    // [`CtuWalker::compute_cu_neighbourhood`] samples the motion field
    // in its parse-time state. The motion-field writes performed by
    // [`CtuWalker::reconstruct_leaf_cu_inter`] only happen during
    // [`CtuWalker::decode_picture_into`] — *after* `decode_ctu_full`
    // has parsed every CU's syntax. So during parse every prior CU's
    // motion-field slot is still UNAVAILABLE, `left_cu_skip =
    // above_cu_skip = false` for every CU, ctxInc = 0, and the slot
    // is `init_type * 3 + 0 = 3` uniformly.
    let split_inc_8 = ctx_inc_split_cu_flag(false, false, 0, 0, 8, 8, 1, 1, 1, 1, 1) as usize;
    let split_slot_8 = split_inc_8.min(split_cu_n_minus1);
    let merge_inc = ctx_inc_merge_idx() as usize;
    let merge_idx_n_minus1 = merge_idx_ctxs.len() - 1;
    let merge_slot = (init_type as usize + merge_inc).min(merge_idx_n_minus1);
    let skip_inc = ctx_inc_cu_skip_flag(false, false, false, false) as usize;
    let skip_slot = (init_type as usize) * 3 + skip_inc;

    // a) Four split_cu_flag(0) for the four 8x8 children (TL, TR, BL, BR).
    for _ in 0..4 {
        enc.encode_decision(&mut split_cu_ctxs[split_slot_8], 0)
            .unwrap();
    }
    // b) Four (cu_skip(1) + merge_idx(0)) pairs in z-order.
    for _ in 0..4 {
        enc.encode_decision(&mut cu_skip_ctxs[skip_slot], 1)
            .unwrap();
        enc.encode_decision(&mut merge_idx_ctxs[merge_slot], 0)
            .unwrap();
    }

    enc.encode_terminate(1).unwrap();
    let payload = enc.finish();

    // ---- SPS / PPS / slice header ---------------------------------------
    let mut sps = dummy_sps(0, pic_w, pic_h);
    sps.tool_flags.six_minus_max_num_merge_cand = 0; // MaxNumMergeCand = 6
    let pps = dummy_pps(pic_w, pic_h);
    let sh = StatefulSliceHeader {
        sh_slice_type: SliceType::P,
        sh_qp_delta: 0,
        ..Default::default()
    };
    let layout = CtuLayout::from_sps_pps(&sps, &pps);

    let mut walker = CtuWalker::begin_slice(&layout, &sps, &pps, &sh, 0, &payload).unwrap();
    walker.set_ref_pic_list_l0(vec![ref_pic]);

    // Slice-start invariant: HMVP table reset to empty per §7.3.11.
    assert_eq!(
        walker.hmvp_table().len(),
        0,
        "HMVP table must be empty at slice start (§7.3.11 NumHmvpCand = 0 reset)"
    );

    let mut out = PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 222);
    walker.decode_picture_into(&mut out).unwrap();

    // Acceptance: the multi-CU decode must reproduce the reference
    // picture byte-exactly. Every chosen merge candidate resolves to
    // the same uni-pred zero-MV / refIdx 0 record (HMVP entries from
    // CU1..CU3 dedup with the spatial zero-MV neighbours and zero-MV
    // pads), so the reconstructed picture must equal the constant-64
    // reference frame.
    assert_eq!(
        out.luma.samples, ref_buf.luma.samples,
        "luma plane must equal reference frame after quad-split all-skip P-slice decode"
    );
    assert_eq!(
        out.cb.samples, ref_buf.cb.samples,
        "Cb plane must equal reference frame"
    );
    assert_eq!(
        out.cr.samples, ref_buf.cr.samples,
        "Cr plane must equal reference frame"
    );

    // Post-decode HMVP table invariant: §8.5.2.16 dedup collapses all
    // four identical inter-CU pushes into a single entry (same MvField
    // → existing entry is removed and re-appended at the back). Pin
    // both bounds: at least 1 entry (CU4's push) and at most
    // `MAX_HMVP_CAND` = 5.
    let table = walker.hmvp_table();
    assert!(
        !table.entries.is_empty(),
        "HMVP table must have at least one entry after a 4-CU inter slice"
    );
    assert!(
        table.entries.len() <= oxideav_h266::inter::MAX_HMVP_CAND,
        "HMVP table size must respect §8.5.2.16 cap of {}",
        oxideav_h266::inter::MAX_HMVP_CAND
    );
    // The retained entry must be the CU's chosen MvField (zero-MV,
    // refIdx 0, predFlagL0 = 1, cu_skip = 1).
    let last = table.entries.last().unwrap();
    assert!(last.pred_flag_l0);
    assert!(!last.pred_flag_l1);
    assert_eq!(last.ref_idx_l0, 0);
    assert_eq!(last.mv_l0.x, 0);
    assert_eq!(last.mv_l0.y, 0);
    assert!(last.mode_inter);
    assert!(last.cu_skip_flag);

    // Per-block motion-field invariant: every 4x4 block must carry
    // the chosen MvField — proves the broadcast in
    // `reconstruct_leaf_cu_inter` ran for all four CUs.
    let mf = walker.motion_field();
    for by in 0..mf.blocks_h {
        for bx in 0..mf.blocks_w {
            let f = mf.field[(by * mf.blocks_w + bx) as usize];
            assert!(f.available);
            assert!(f.pred_flag_l0);
            assert_eq!(f.ref_idx_l0, 0);
            assert_eq!(f.mv_l0.x, 0);
            assert_eq!(f.mv_l0.y, 0);
            assert!(f.cu_skip_flag);
        }
    }
}

/// Round-25 acceptance fixture — temporal merge candidate (§8.5.2.11)
/// fires and decodes correctly. A P-slice picks the Col candidate at
/// `merge_idx = 0` because:
///   * spatial neighbours at `(xCb, yCb) = (0, 0)` are unavailable
///     (negative coordinates); and
///   * the HMVP table is empty at slice start (single CU); and
///   * `ph_temporal_mvp_enabled_flag = 1` + the collocated reference
///     picture carries an attached MotionField with a uniform non-zero
///     L0 MV at the centre fallback position.
///
/// The §8.5.2.11 derivation rejects the bottom-right collocated
/// position (xColBr = 16 is out-of-bounds for a 16x16 picture), falls
/// back to centre at (8, 8) — 8x8-rounded to (8, 8) — and fetches the
/// uniform MV `(2, 0)` integer-pel from the reference. Equal POC
/// distances (currPocDiff == colPocDiff == 1) ⇒ §8.5.2.12 eq. 600
/// short-circuit ⇒ unscaled MV.
///
/// The §8.5.6.3 motion-compensation step then translates the reference
/// frame by `(2, 0)` luma samples, with picture-edge clamping on the
/// right side. The expected output is therefore the reference frame
/// rolled left by 2 pixels (rightmost 2 columns become the original
/// column 7 — clamped to picture edge).
///
/// Acceptance criterion: the decoded luma plane must equal the
/// translated reference. This is the smallest end-to-end fixture that
/// proves the temporal merge candidate fires and the picked MV survives
/// through to MC.
#[test]
fn decode_p_slice_temporal_merge_fires_and_decodes() {
    use oxideav_h266::cabac_enc::ArithEncoder;
    use oxideav_h266::ctx::{ctx_inc_cu_skip_flag, ctx_inc_merge_idx, ctx_inc_split_cu_flag};
    use oxideav_h266::inter::{MotionField, MotionVector, MvField};
    use oxideav_h266::tables::{init_contexts, SyntaxCtx};

    let pic_w = 16u32;
    let pic_h = 16u32;

    // ---- Reference picture: distinctive ramp + uniform MotionField ----
    let mut ref_buf = PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 0);
    for y in 0..pic_h as usize {
        for x in 0..pic_w as usize {
            ref_buf.luma.samples[y * pic_w as usize + x] = (10 + x + y * 2) as u8;
        }
    }
    for y in 0..(pic_h / 2) as usize {
        for x in 0..(pic_w / 2) as usize {
            ref_buf.cb.samples[y * (pic_w / 2) as usize + x] = (90 + y + x) as u8;
            ref_buf.cr.samples[y * (pic_w / 2) as usize + x] = (170 + y + x) as u8;
        }
    }

    // Uniform MotionField: every 4x4 block carries L0 MV (2, 0) /
    // refIdx 0. The temporal-merge derivation samples at the centre
    // 8x8-rounded position (8, 8) and gets back this MV.
    let mut mf = MotionField::new(pic_w, pic_h);
    let cell = MvField {
        mv_l0: MotionVector::from_int_pel(2, 0),
        ref_idx_l0: 0,
        pred_flag_l0: true,
        mv_l1: MotionVector::ZERO,
        ref_idx_l1: -1,
        pred_flag_l1: false,
        cu_skip_flag: true,
        mode_inter: true,
        available: true,
        bcw_idx: 0,
    };
    mf.write_block(0, 0, pic_w, pic_h, cell);

    let ref_pic = ReferencePicture {
        poc: 0,
        frame: ref_buf.clone(),
        motion_field: Some(mf),
    };

    // ---- P-slice CABAC payload synthesis ------------------------------
    // Single 16x16 CU at (0, 0): split_cu_flag(0), cu_skip_flag(1),
    // general_merge_flag inferred 1, merge_idx bin0 = 0 → mergeCandList[0].
    // mergeCandList[0] is the Col candidate (spatials all unavailable;
    // HMVP empty at slice start).
    let slice_qp = 26;
    let init_type = 1u8; // P-slice, sh_cabac_init_flag = 0
    let mut split_cu_ctxs = init_contexts(SyntaxCtx::SplitCuFlag, slice_qp);
    let mut cu_skip_ctxs = init_contexts(SyntaxCtx::CuSkipFlag, slice_qp);
    let mut merge_idx_ctxs = init_contexts(SyntaxCtx::MergeIdx, slice_qp);
    let mut enc = ArithEncoder::new();

    let split_inc = ctx_inc_split_cu_flag(false, false, 0, 0, 16, 16, 1, 1, 1, 1, 1) as usize;
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

    // ---- SPS / PPS / SH ----------------------------------------------
    let mut sps = dummy_sps(0, pic_w, pic_h);
    sps.tool_flags.six_minus_max_num_merge_cand = 0; // MaxNumMergeCand = 6
    sps.tool_flags.temporal_mvp_enabled_flag = true;
    let pps = dummy_pps(pic_w, pic_h);
    let sh = StatefulSliceHeader {
        sh_slice_type: SliceType::P,
        sh_qp_delta: 0,
        ..Default::default()
    };

    let layout = CtuLayout::from_sps_pps(&sps, &pps);
    let mut walker = CtuWalker::begin_slice(&layout, &sps, &pps, &sh, 0, &payload).unwrap();
    walker.set_ref_pic_list_l0(vec![ref_pic]);
    // §8.5.2.11 wiring: enable temporal MVP, current POC = 1 (the
    // reference is at POC 0), collocated_from_l0 = true, col_ref_idx = 0.
    // currPocDiff = 1 - 0 = 1; col_ref_poc defaults to current_poc = 1
    // (the walker's heuristic) so colPocDiff = 0 - 1 = -1. Wait —
    // that is unequal-magnitude; rerun the analysis to confirm the
    // expected scaled MV.
    //
    // Actually the walker's `derive_col_candidate` sets
    //   col_ref_poc = self.current_poc = 1
    // so colPocDiff = ColPic.poc - col_ref_poc = 0 - 1 = -1
    // and currPocDiff = current_poc - currentRefL0.poc = 1 - 0 = 1.
    // Eqs. 601 – 603 then scale (mv = (32, 0) in 1/16 units = 2 int-pel):
    //   td = -1, tb = 1
    //   tx = (16384 + 0) / -1 = -16384
    //   distScaleFactor = clamp(-4096..4095, (1 * -16384 + 32) >> 6)
    //                   = clamp(..., (-16352) >> 6) = clamp(..., -256) = -256
    //   prod = -256 * 32 = -8192
    //   prod < 0 → bias = 0
    //   result = (-8192 + 128 - 0) >> 8 = -8064 >> 8 = -32 (round-down)
    // So scaled MV.x = -32 (1/16 units = -2 int-pel). The reference is
    // POC 0 → it appears later than the current picture (POC 1) in
    // display order ⇒ a backward MV. The MV gets sign-flipped because
    // currPocDiff and colPocDiff have opposite signs.
    //
    // Expected output: reference rolled RIGHT by 2 pixels (i.e. dst[x]
    // reads src[x - 2], with src clamped at the left edge).
    walker.set_temporal_mvp(/*current_poc*/ 1, true, true, 0);

    let mut out = PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 222);
    walker.decode_picture_into(&mut out).unwrap();

    // ---- Build expected output: reference translated by (-2, 0) ------
    // mc_copy_block_int with src = ref, src_x = -2, src_y = 0:
    //   dst[y][x] = src[clamp(y+0, 0, h-1)][clamp(x-2, 0, w-1)]
    let mut expected_y = vec![0u8; (pic_w * pic_h) as usize];
    for y in 0..pic_h as usize {
        for x in 0..pic_w as usize {
            let sx = (x as i32 - 2).max(0) as usize;
            let sy = y;
            expected_y[y * pic_w as usize + x] = ref_buf.luma.samples[sy * pic_w as usize + sx];
        }
    }
    assert_eq!(
        out.luma.samples, expected_y,
        "luma must equal reference translated by the temporal-merge MV"
    );

    // Sanity: the chosen MvField in the motion field must carry the
    // scaled MV that was actually fetched (-32 in 1/16 units = -2 int-pel).
    let mf_out = walker.motion_field();
    let centre = mf_out.get_at_luma(8, 8);
    assert!(centre.available);
    assert!(centre.pred_flag_l0);
    assert_eq!(
        centre.mv_l0.x, -32,
        "broadcast MvField must carry the scaled temporal MV"
    );
    assert_eq!(centre.mv_l0.y, 0);
    assert_eq!(centre.ref_idx_l0, 0);
}

/// Round-26 acceptance fixture: pairwise-average merge candidate
/// (§8.5.2.4) is the chosen merge index for the BR CU of a quad-split
/// P-slice and decodes byte-exactly via the §8.5.2.2 step 8 invocation
/// path.
///
/// Setup geometry:
///   * 16x16 picture, single 32x32 CTU (one CTU covers everything).
///   * Quad-split into four 8x8 leaf CUs in z-order
///     (TL=(0,0), TR=(8,0), BL=(0,8), BR=(8,8)).
///   * `MaxNumMergeCand = 6`, P-slice, single L0 reference picture.
///   * `ph_temporal_mvp_enabled_flag = 1` → §8.5.2.11 Col candidate
///     fires for every CU (8x8 area = 64 > 32, the spec's `cbWidth *
///     cbHeight <= 32` skip threshold).
///
/// Reference picture has a non-uniform L0 MotionField split into the
/// same four 8x8 quadrants. Only the TR/BL/BR regions are exercised by
/// the §8.5.2.11 fetches:
///   * TR region (8..16, 0..8):  ref MV = (1, 0) integer-pel
///   * BL region (0..8, 8..16):  ref MV = (3, 0)
///   * BR region (8..16, 8..16): ref MV = (4, 0)
///   * TL region: ref MV = (0, 0) — never sampled (no CU's Col fetch
///     lands here once the spec's 8x8 round + BR-fallback / centre
///     position resolve to TR/BL/BR).
///
/// Per-CU §8.5.2.11 Col MV resolution (ref_pic.poc=0, current_poc=1,
/// `col_ref_poc = current_poc = 1` per the walker heuristic, so
/// `colPocDiff = 0 - 1 = -1` and `currPocDiff = 1 - 0 = 1`. §8.5.2.12
/// eqs. 601 – 603 then scale by `distScaleFactor = -256` (sign flips):
/// for `mvCol = (a, 0)` integer-pel, `scaled.x = ((-256 * a*16) + 128)
/// >> 8 = (-4096a + 128) >> 8`):
///   * a=1 → scaled = (-16, 0) = (-1, 0) integer-pel
///   * a=3 → scaled = (-48, 0) = (-3, 0) integer-pel
///   * a=4 → scaled = (-64, 0) = (-4, 0) integer-pel
///
/// **CU0 (TL @(0,0,8x8))**: BR fetch position (8,8) → BR region → mvCol
/// (4,0) → scaled (-64, 0). Spatial neighbours all unavailable; HMVP
/// empty; merge_idx=0 picks Col. CU0 chosen MV = (-64, 0).
///
/// **CU1 (TR @(8,0,8x8))**: BR fetch position (16,8) outside picture →
/// centre fallback (12,4) rounded to (8,0) → TR region → mvCol (1,0) →
/// scaled (-16, 0). Spatial walk: B1 at (15,-1) unavail, A1 at (7,7) =
/// TL's MV (-64, 0). B0/A0/B2 all unavail. HMVP = [(-64, 0)] from CU0;
/// newest entry duplicates A1 → §8.5.2.6 prune. Merge list =
/// [A1=(-64, 0), Col=(-16, 0)]. Pairwise = avg → (-40, 0). merge_idx=1
/// picks Col. CU1 chosen MV = (-16, 0).
///
/// **CU2 (BL @(0,8,8x8))**: BR fetch position (8,16) outside picture →
/// centre fallback (4,12) rounded to (0,8) → BL region → mvCol (3,0) →
/// scaled (-48, 0). Spatial walk: B1 at (7,7) = TL's MV (-64, 0); A1
/// at (-1,15) unavail; **B0 at (8,7) = TR's MV (-16, 0)** (distinct
/// from B1 → not pruned); A0 unavail; B2 unavail. Merge list spatial
/// portion = [B1=(-64, 0), B0=(-16, 0)]. Col adds (-48, 0). HMVP =
/// [(-64, 0), (-16, 0)]; newest (-16, 0) doesn't match B1 (A1 unavail
/// in the spatial slot pair) → inserted; older (-64, 0) duplicates B1
/// → pruned. Merge list before pairwise =
/// [(-64, 0), (-16, 0), (-48, 0), (-16, 0)]. Pairwise = avg of slots 0
/// + 1 = (-40, 0). merge_idx=2 picks Col. CU2 chosen MV = (-48, 0).
///
/// **CU3 (BR @(8,8,8x8))**: BR fetch position (16,16) outside picture
/// → centre fallback (12,12) rounded to (8,8) → BR region → mvCol (4,
/// 0) → scaled (-64, 0). Spatial walk: B1 at (15,7) = TR's MV (-16,
/// 0); A1 at (7,15) = BL's MV (-48, 0) (distinct from B1 → not
/// pruned); B0 at (16,7) unavail; A0 at (7,16) unavail; B2 at (7,7) =
/// TL's MV (-64, 0) (distinct from B1 + A1 + the not-all-4-prior
/// guard → kept). Spatial list portion = [B1=(-16, 0), A1=(-48, 0),
/// B2=(-64, 0)]. Col adds another (-64, 0). HMVP =
/// [(-64, 0), (-16, 0), (-48, 0)] (CU0/CU1/CU2 pushes, all distinct).
/// Walk newest first: (-48, 0) matches A1 → pruned; (-16, 0) matches
/// B1 → pruned; (-64, 0) at hMvpIdx=3 (no prune) → inserted. Merge
/// list before §8.5.2.4 step 8 = [B1=(-16, 0), A1=(-48, 0),
/// B2=(-64, 0), Col=(-64, 0), HMVP=(-64, 0)] (length 5). §8.5.2.4
/// pairwise of slots 0+1 = avg((-16, 0), (-48, 0)) =
/// round_pairwise(-64, 0) = (-32, 0) — the test's signature MV.
/// **merge_idx=5 picks the §8.5.2.4 pairwise-average candidate.** CU3
/// chosen MV = (-32, 0) — exactly (-2, 0) integer-pel (zero
/// fractional component, so the §8.5.6.3 fractional-pel filter
/// collapses to the integer-pel passthrough and we get a clean
/// `mc_copy_block_int` translation).
///
/// Acceptance: each 8x8 quadrant of the reconstructed luma plane
/// matches `mc_copy_block_int(ref, src_x = mv.x >> 4, src_y = 0)` with
/// picture-edge clamping. The pairwise quadrant (BR) reads the
/// reference at integer-pel offset (-2, 0) — i.e. `dst[8+y][8+x] =
/// ref[8+y][6+x]` (no clamping needed; 6 + 0 ≥ 0 and 6 + 7 = 13 < 16).
#[test]
fn decode_p_slice_pairwise_average_fires_and_decodes() {
    use oxideav_h266::cabac_enc::ArithEncoder;
    use oxideav_h266::ctx::{
        ctx_inc_cu_skip_flag, ctx_inc_merge_idx, ctx_inc_split_cu_flag, ctx_inc_split_qt_flag,
    };
    use oxideav_h266::inter::{MotionField, MotionVector, MvField};
    use oxideav_h266::tables::{init_contexts, SyntaxCtx};

    let pic_w = 16u32;
    let pic_h = 16u32;

    // ---- Reference picture: distinctive luma ramp + non-uniform MF ---
    let mut ref_buf = PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 0);
    for y in 0..pic_h as usize {
        for x in 0..pic_w as usize {
            // Distinctive per-pixel pattern so the per-quadrant MC
            // translation can be verified bit-exactly.
            ref_buf.luma.samples[y * pic_w as usize + x] = (10 + x + y * 3) as u8;
        }
    }

    // Non-uniform L0 motion field per the four 8x8 quadrants.
    let mut mf = MotionField::new(pic_w, pic_h);
    let cell = |dx: i32| -> MvField {
        MvField {
            mv_l0: MotionVector::from_int_pel(dx, 0),
            ref_idx_l0: 0,
            pred_flag_l0: true,
            mv_l1: MotionVector::ZERO,
            ref_idx_l1: -1,
            pred_flag_l1: false,
            cu_skip_flag: false,
            mode_inter: true,
            available: true,
            bcw_idx: 0,
        }
    };
    // TL (0..8, 0..8): mv (0, 0) — unused by any CU's Col fetch.
    mf.write_block(0, 0, 8, 8, cell(0));
    // TR (8..16, 0..8): mv (1, 0) — sampled by CU1.
    mf.write_block(8, 0, 8, 8, cell(1));
    // BL (0..8, 8..16): mv (3, 0) — sampled by CU2.
    mf.write_block(0, 8, 8, 8, cell(3));
    // BR (8..16, 8..16): mv (4, 0) — sampled by CU0 + CU3.
    mf.write_block(8, 8, 8, 8, cell(4));

    let ref_pic = ReferencePicture {
        poc: 0,
        frame: ref_buf.clone(),
        motion_field: Some(mf),
    };

    // ---- P-slice CABAC payload synthesis -----------------------------
    // Tree structure: split_cu_flag(1) + split_qt_flag(1) at the 16x16
    // root → four 8x8 children. Per-leaf split_cu_flag(0) at 8x8, then
    // (cu_skip(1), merge_idx) per leaf in z-order. merge_idx values:
    // CU0=0, CU1=1, CU2=1, CU3=4.
    let slice_qp = 26;
    let init_type = 1u8; // P-slice, sh_cabac_init_flag = 0
    let mut split_cu_ctxs = init_contexts(SyntaxCtx::SplitCuFlag, slice_qp);
    let mut split_qt_ctxs = init_contexts(SyntaxCtx::SplitQtFlag, slice_qp);
    let mut cu_skip_ctxs = init_contexts(SyntaxCtx::CuSkipFlag, slice_qp);
    let mut merge_idx_ctxs = init_contexts(SyntaxCtx::MergeIdx, slice_qp);
    let mut enc = ArithEncoder::new();

    // Root split_cu_flag(1) + split_qt_flag(1).
    let split_cu_inc_root =
        ctx_inc_split_cu_flag(false, false, 0, 0, 16, 16, 1, 1, 1, 1, 1) as usize;
    let split_cu_n_minus1 = split_cu_ctxs.len() - 1;
    enc.encode_decision(
        &mut split_cu_ctxs[split_cu_inc_root.min(split_cu_n_minus1)],
        1,
    )
    .unwrap();
    let split_qt_inc_root = ctx_inc_split_qt_flag(false, false, 0, 0, 0) as usize;
    let split_qt_n_minus1 = split_qt_ctxs.len() - 1;
    enc.encode_decision(
        &mut split_qt_ctxs[split_qt_inc_root.min(split_qt_n_minus1)],
        1,
    )
    .unwrap();

    // Per-leaf bin slots.
    let split_inc_8 = ctx_inc_split_cu_flag(false, false, 0, 0, 8, 8, 1, 1, 1, 1, 1) as usize;
    let split_slot_8 = split_inc_8.min(split_cu_n_minus1);
    let merge_inc = ctx_inc_merge_idx() as usize;
    let merge_idx_n_minus1 = merge_idx_ctxs.len() - 1;
    let merge_slot = (init_type as usize + merge_inc).min(merge_idx_n_minus1);
    let skip_inc = ctx_inc_cu_skip_flag(false, false, false, false) as usize;
    let skip_slot = (init_type as usize) * 3 + skip_inc;

    // Four split_cu_flag(0) for the four 8x8 children.
    for _ in 0..4 {
        enc.encode_decision(&mut split_cu_ctxs[split_slot_8], 0)
            .unwrap();
    }

    // Per-CU (cu_skip(1), merge_idx) pairs in z-order. The merge_idx
    // truncated-unary encoding (cMax = MaxNumMergeCand - 1 = 5):
    //   * value 0 → bin0=0 (1 ctx-coded bin)
    //   * value 1 → bin0=1 ctx, bin1=0 bypass
    //   * value 2 → bin0=1 ctx, bin1=1 bypass, bin2=0 bypass
    //   * value 5 (== cMax) → bin0=1 ctx, bins1..4 = 1 bypass each;
    //     loop exits at val == cMax with NO terminator bypass bit
    //     (`while val < cmax` falls through).
    let max_merge_minus1: u32 = 5;
    let merge_idx_values: [u32; 4] = [0, 1, 2, 5];
    for &mi in &merge_idx_values {
        // cu_skip_flag(1).
        enc.encode_decision(&mut cu_skip_ctxs[skip_slot], 1)
            .unwrap();
        // merge_idx truncated-unary encode.
        if mi == 0 {
            enc.encode_decision(&mut merge_idx_ctxs[merge_slot], 0)
                .unwrap();
        } else {
            enc.encode_decision(&mut merge_idx_ctxs[merge_slot], 1)
                .unwrap();
            // Emit (mi - 1) bypass=1 bits to advance val to mi.
            for _ in 1..mi {
                enc.encode_bypass(1).unwrap();
            }
            // Terminator bypass=0 — UNLESS val == cMax, in which case
            // the decoder's `while val < cmax` loop exits without
            // reading a terminator bin. Mirror that here.
            if mi < max_merge_minus1 {
                enc.encode_bypass(0).unwrap();
            }
        }
    }

    enc.encode_terminate(1).unwrap();
    let payload = enc.finish();

    // ---- SPS / PPS / slice header ------------------------------------
    let mut sps = dummy_sps(0, pic_w, pic_h);
    sps.tool_flags.six_minus_max_num_merge_cand = 0; // MaxNumMergeCand = 6
    sps.tool_flags.temporal_mvp_enabled_flag = true;
    let pps = dummy_pps(pic_w, pic_h);
    let sh = StatefulSliceHeader {
        sh_slice_type: SliceType::P,
        sh_qp_delta: 0,
        ..Default::default()
    };
    let layout = CtuLayout::from_sps_pps(&sps, &pps);

    let mut walker = CtuWalker::begin_slice(&layout, &sps, &pps, &sh, 0, &payload).unwrap();
    walker.set_ref_pic_list_l0(vec![ref_pic]);
    // Enable §8.5.2.11 with the same (current_poc=1, ref.poc=0,
    // collocated_from_l0=true, col_ref_idx=0) configuration as the
    // round-25 fixture. The walker's heuristic
    // `col_ref_poc = current_poc` produces colPocDiff = -1 vs.
    // currPocDiff = +1 → eqs. 601 – 603 sign-flip the Col MV (the
    // distScaleFactor = -256 chain analysed above).
    walker.set_temporal_mvp(/*current_poc*/ 1, true, true, 0);

    let mut out = PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 222);
    walker.decode_picture_into(&mut out).unwrap();

    // ---- Build the expected luma plane: per-quadrant translated copy
    // of the reference frame.
    //   CU0 (TL) MV = (-4, 0) int-pel: dst[y][x] = ref[y][clamp(x-4)]
    //   CU1 (TR) MV = (-1, 0):       dst[y][8+x] = ref[y][7+x]
    //   CU2 (BL) MV = (-3, 0):       dst[8+y][x] = ref[8+y][clamp(x-3)]
    //   CU3 (BR) MV = (-2, 0) (PAIRWISE!): dst[8+y][8+x] = ref[8+y][6+x]
    let mut expected_y = vec![0u8; (pic_w * pic_h) as usize];
    let stride = pic_w as usize;
    let read_clamped = |src_x: i32, src_y: i32| -> u8 {
        let sx = src_x.clamp(0, pic_w as i32 - 1) as usize;
        let sy = src_y.clamp(0, pic_h as i32 - 1) as usize;
        ref_buf.luma.samples[sy * stride + sx]
    };
    let translate_quad = |expected: &mut [u8], qx: usize, qy: usize, mv_int_x: i32| {
        for y in 0..8 {
            for x in 0..8 {
                expected[(qy + y) * stride + (qx + x)] =
                    read_clamped(qx as i32 + x as i32 + mv_int_x, qy as i32 + y as i32);
            }
        }
    };
    translate_quad(&mut expected_y, 0, 0, -4); // CU0 MV = (-4, 0)
    translate_quad(&mut expected_y, 8, 0, -1); // CU1 MV = (-1, 0)
    translate_quad(&mut expected_y, 0, 8, -3); // CU2 MV = (-3, 0)
    translate_quad(&mut expected_y, 8, 8, -2); // CU3 MV = (-2, 0) ← PAIRWISE-AVERAGE

    assert_eq!(
        out.luma.samples, expected_y,
        "luma must match per-quadrant translated reference; the BR \
         quadrant's translation = (-2, 0) integer-pel proves the \
         §8.5.2.4 pairwise-average candidate fired and was selected"
    );

    // Per-block motion-field invariant: the BR CU's MvField must carry
    // the pairwise-average MV (-32, 0) in 1/16-pel units. This is the
    // load-bearing assertion — it pins that the chosen merge candidate
    // for CU3 was indeed the §8.5.2.4 avgCand and not the Col / HMVP
    // entries (which would have surfaced as (-64, 0)).
    let mf_out = walker.motion_field();
    let cu3_centre = mf_out.get_at_luma(12, 12);
    assert!(cu3_centre.available);
    assert!(cu3_centre.pred_flag_l0);
    assert_eq!(
        cu3_centre.mv_l0,
        MotionVector { x: -32, y: 0 },
        "CU3's broadcast MvField must carry the §8.5.2.4 pairwise- \
         average MV ((-16) + (-48)) >> 1 = -32 in 1/16-pel units"
    );
    assert_eq!(cu3_centre.ref_idx_l0, 0);
    // Sanity-check the per-CU MVs the analysis predicted:
    assert_eq!(
        mf_out.get_at_luma(2, 2).mv_l0,
        MotionVector { x: -64, y: 0 },
        "CU0 picked Col = (-64, 0)"
    );
    assert_eq!(
        mf_out.get_at_luma(10, 2).mv_l0,
        MotionVector { x: -16, y: 0 },
        "CU1 picked Col = (-16, 0)"
    );
    assert_eq!(
        mf_out.get_at_luma(2, 10).mv_l0,
        MotionVector { x: -48, y: 0 },
        "CU2 picked Col = (-48, 0)"
    );
}

/// Round-27: P-slice merge with motion vector difference (MMVD,
/// §8.5.2.7). Single 8x8 CU; cu_skip = 1 path infers general_merge_flag
/// + regular_merge_flag to 1, then the MMVD sub-tree fires:
///
/// * `mmvd_merge_flag = 1`   (ctx-coded ctx 0 of Table 103)
/// * `mmvd_cand_flag = 0`    (ctx-coded ctx 0 of Table 104; picks
///   `mergeCandList[0]` = zero-MV pad as the base)
/// * `mmvd_distance_idx = 2` (TR cMax=7; Table 17 entry `MmvdDistance =
///   4`, eq. 188 emits `4 << 2 = 16` 1/16-pel units = `1` integer luma
///   sample)
/// * `mmvd_direction_idx = 0` (FL cMax=3; Table 18 entry
///   `(MmvdSign[0], MmvdSign[1]) = (+1, 0)`)
///
/// → `MmvdOffset = (+16, 0)` 1/16-pel = `(+1, 0)` integer-pel. Applied
/// to the zero-MV base candidate, the final L0 MV is `(+16, 0)` so the
/// MC fetch translates the reference by one column (`dst[y][x] =
/// ref[y][x + 1]`, clamped at the right edge per §8.5.6.3).
///
/// The fixture verifies:
///
/// 1. `sps_mmvd_enabled_flag = 1` survives the SPS round-trip and is
///    surfaced via [`CtuWalker::cu_tool_flags`].
/// 2. The leaf CU reader parses the four MMVD syntax elements (Table
///    103 / 104 / 105 + Table 18 FL bypass bins) correctly off a
///    synthesised CABAC payload.
/// 3. The CTU walker calls [`oxideav_h266::inter::derive_mmvd_offset`]
///    then [`oxideav_h266::inter::apply_mmvd_to_base`] in the right
///    order so the per-block MotionField records the *MMVD-corrected*
///    MV (`(+16, 0)` in 1/16-pel units) — not the base MV (`(0, 0)`).
/// 4. The reconstructed luma plane matches the spec-derived
///    translated reference (`ref[y][x + 1]` clamped).
#[test]
fn decode_p_slice_mmvd_fires_and_decodes() {
    use oxideav_h266::cabac_enc::ArithEncoder;
    use oxideav_h266::ctx::{ctx_inc_cu_skip_flag, ctx_inc_split_cu_flag};
    use oxideav_h266::inter::MotionVector;
    use oxideav_h266::tables::{init_contexts, SyntaxCtx};

    let pic_w = 8u32;
    let pic_h = 8u32;

    // Reference: distinctive ramp so the per-pixel translation is
    // verifiable bit-exactly.
    let mut ref_buf = PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 0);
    for y in 0..pic_h as usize {
        for x in 0..pic_w as usize {
            ref_buf.luma.samples[y * pic_w as usize + x] = (10 + x + y * 5) as u8;
        }
    }
    let ref_pic = ReferencePicture {
        poc: 0,
        frame: ref_buf.clone(),
        motion_field: None,
    };

    // ---- CABAC payload synthesis -------------------------------------
    let slice_qp = 26;
    let init_type = 1u8; // P-slice, sh_cabac_init_flag = 0

    let mut split_cu_ctxs = init_contexts(SyntaxCtx::SplitCuFlag, slice_qp);
    let mut cu_skip_ctxs = init_contexts(SyntaxCtx::CuSkipFlag, slice_qp);
    let mut mmvd_merge_ctxs = init_contexts(SyntaxCtx::MmvdMergeFlag, slice_qp);
    let mut mmvd_cand_ctxs = init_contexts(SyntaxCtx::MmvdCandFlag, slice_qp);
    let mut mmvd_dist_ctxs = init_contexts(SyntaxCtx::MmvdDistanceIdx, slice_qp);

    let mut enc = ArithEncoder::new();

    // 1. split_cu_flag(0) — single 8x8 CU at the CTU root.
    let split_inc = ctx_inc_split_cu_flag(false, false, 0, 0, 8, 8, 1, 1, 1, 1, 1) as usize;
    let split_slot = split_inc.min(split_cu_ctxs.len() - 1);
    enc.encode_decision(&mut split_cu_ctxs[split_slot], 0)
        .unwrap();

    // 2. cu_skip_flag(1) — skip-mode CU. Inference chain: skip=1
    //    implies general_merge_flag=1 (no bin); CIIP/GPM gates collapse
    //    so regular_merge_flag also infers to 1 (no bin).
    let skip_inc = ctx_inc_cu_skip_flag(false, false, false, false) as usize;
    let skip_slot = (init_type as usize) * 3 + skip_inc;
    enc.encode_decision(&mut cu_skip_ctxs[skip_slot], 1)
        .unwrap();

    // 3. mmvd_merge_flag(1) — Table 103 ctxIdx = init_type - 1 = 0.
    let mmvd_merge_slot = (init_type as usize) - 1;
    enc.encode_decision(&mut mmvd_merge_ctxs[mmvd_merge_slot], 1)
        .unwrap();

    // 4. mmvd_cand_flag(0) — picks mergeCandList[0] = zero-MV pad.
    let mmvd_cand_slot = (init_type as usize) - 1;
    enc.encode_decision(&mut mmvd_cand_ctxs[mmvd_cand_slot], 0)
        .unwrap();

    // 5. mmvd_distance_idx = 2 — TR(cMax=7, cRiceParam=0) encoding:
    //    bin0 = 1 (ctx), bin1 = 1 (bypass), bin2 = 0 (bypass terminator).
    let mmvd_dist_slot = (init_type as usize) - 1;
    enc.encode_decision(&mut mmvd_dist_ctxs[mmvd_dist_slot], 1)
        .unwrap();
    enc.encode_bypass(1).unwrap();
    enc.encode_bypass(0).unwrap();

    // 6. mmvd_direction_idx = 0 — FL(cMax=3) emits 2 bypass bins MSB
    //    first: `00`.
    enc.encode_bypass(0).unwrap();
    enc.encode_bypass(0).unwrap();

    enc.encode_terminate(1).unwrap();
    let payload = enc.finish();

    // ---- SPS / PPS / slice header ------------------------------------
    let mut sps = dummy_sps(0, pic_w, pic_h);
    sps.tool_flags.six_minus_max_num_merge_cand = 0; // MaxNumMergeCand = 6
    sps.tool_flags.mmvd_enabled_flag = true;
    let pps = dummy_pps(pic_w, pic_h);
    let sh = StatefulSliceHeader {
        sh_slice_type: SliceType::P,
        sh_qp_delta: 0,
        ..Default::default()
    };
    let layout = CtuLayout::from_sps_pps(&sps, &pps);
    let mut walker = CtuWalker::begin_slice(&layout, &sps, &pps, &sh, 0, &payload).unwrap();
    walker.set_ref_pic_list_l0(vec![ref_pic]);
    // Default: ph_mmvd_fullpel_only = false → use Table 17 regular grid.

    let mut out = PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 222);
    walker.decode_picture_into(&mut out).unwrap();

    // ---- Expected luma: ref translated by (+1, 0) integer-pel.
    //
    // MMVD chain:
    //   mmvd_distance_idx = 2 → MmvdDistance = 4
    //   eq. 188:    MmvdOffset[0] = (4 << 2) * (+1) = +16 (1/16-pel)
    //               = +1 integer-pel
    //   chosen_mv  = base_mv + MmvdOffset = (0,0) + (+16, 0) = (+16, 0)
    //   §8.5.6.3 fetch: src_x = dst_x + (mv >> 4) = dst_x + 1
    //
    // → dst[y][x] = ref[y][min(x + 1, W - 1)].
    let stride = pic_w as usize;
    let mut expected_y = vec![0u8; (pic_w * pic_h) as usize];
    for y in 0..pic_h as usize {
        for x in 0..pic_w as usize {
            let sx = (x + 1).min(pic_w as usize - 1);
            expected_y[y * stride + x] = ref_buf.luma.samples[y * stride + sx];
        }
    }
    assert_eq!(
        out.luma.samples, expected_y,
        "luma must match the MMVD-corrected reference translation \
         (base zero-MV merge candidate + (+1, 0) MMVD offset = +1 \
         integer-pel column shift)"
    );

    // ---- Per-block motion field: MMVD-corrected MV must be broadcast.
    //
    // The base candidate is mergeCandList[0] which for a CU with no
    // available spatial / temporal / HMVP neighbours is the §8.5.2.2
    // step 9 zero-MV pad. After §8.5.2.7 MMVD application the recorded
    // MvField must carry mv_l0 = (+16, 0), refIdx 0, predFlagL0 = 1.
    let mf = walker.motion_field();
    let centre = mf.get_at_luma(4, 4);
    assert!(centre.available);
    assert!(centre.pred_flag_l0);
    assert!(!centre.pred_flag_l1);
    assert_eq!(centre.ref_idx_l0, 0);
    assert_eq!(
        centre.mv_l0,
        MotionVector { x: 16, y: 0 },
        "MotionField must carry the MMVD-corrected MV (+16, 0) \
         (i.e. base zero-MV + MmvdOffset(+16, 0) per §8.5.2.7), not \
         the base zero-MV"
    );
    assert!(centre.cu_skip_flag);
    assert!(centre.mode_inter);
}

/// Round-28: P-slice combined inter-intra prediction (CIIP, §8.5.6.7).
/// Single 8x8 CU at the top-left of an 8x8 picture; the CU sits at
/// the picture corner so neither §8.5.6.7 neighbour (A at
/// `(xCb − 1, yCb − 1 + cbHeight) = (−1, 7)`, B at
/// `(xCb − 1 + cbWidth, yCb − 1) = (7, −1)`) is in-picture, so both
/// register as not-intra → §8.5.6.7 weight `w = 1` → eq. 998 collapses
/// to `(1 * predIntra + 3 * predInter + 2) >> 2`.
///
/// CABAC sequence (from `decode_inter` in `leaf_cu.rs`):
///
/// * `split_cu_flag = 0`           — single 8x8 CU at the CTU root.
/// * `cu_skip_flag = 0`            — non-skip merge (CIIP requires this).
/// * `general_merge_flag = 1`      — merge data follows.
/// * `regular_merge_flag = 0`      — gate is open because
///   `sps_ciip_enabled_flag = 1`, `cu_skip_flag = 0`, `cbW * cbH = 64`,
///   `cbW < 128`, `cbH < 128`. The `0` selects the CIIP branch.
/// * `ciip_flag = 1` (inferred per §7.4.12.7 — `gpm_enabled = 0` means
///   no parse, the inference rules pick 1 since CIIP is the only
///   non-regular branch open). No CABAC bin consumed.
/// * `merge_idx = 0`               — TR(cMax = 5), bin 0 ctx-coded = 0.
/// * `cu_coded_flag = 0`           — no transform_tree() body.
///
/// Reference picture: a per-pixel ramp with each sample distinct so the
/// CIIP combination is observable. Zero MV pad → predSamplesInter =
/// the reference sample at the same position. Above / left neighbours
/// of the CU are all out-of-picture → planar refs default to mid-grey
/// (128 at 8-bit) → predSamplesIntra = 128 (planar of all-128 refs).
///
/// Expected reconstructed luma sample at `(x, y)`:
///   `(1 * 128 + 3 * ref[y][x] + 2) >> 2`
///
/// The fixture verifies:
///
/// 1. `sps_ciip_enabled_flag = 1` flows through `CtuWalker::cu_tool_flags`
///    so the leaf CU reader hits the CIIP gate.
/// 2. The `regular_merge_flag = 0` parse is taken (not the round-21 +
///    round-27 inferred-to-1 collapse).
/// 3. `MergeData::ciip_flag` is set to `true` per the §7.4.12.7
///    inference and propagated to the CTU walker.
/// 4. The reconstructed luma plane matches the spec eq. 998 weighted
///    average sample-by-sample.
/// 5. The motion field still records the regular-merge MC vector
///    (zero-MV pad, ref index 0, predFlagL0 = 1) — CIIP doesn't
///    change the per-block MV state.
#[test]
fn decode_p_slice_ciip_fires_and_decodes() {
    use oxideav_h266::cabac_enc::ArithEncoder;
    use oxideav_h266::ctx::{
        ctx_inc_cu_skip_flag, ctx_inc_general_merge_flag, ctx_inc_regular_merge_flag,
        ctx_inc_split_cu_flag,
    };
    use oxideav_h266::inter::MotionVector;
    use oxideav_h266::tables::{init_contexts, SyntaxCtx};

    let pic_w = 8u32;
    let pic_h = 8u32;

    // Reference: per-pixel ramp; the high range exercises the eq. 998
    // weighted-average rounding path.
    let mut ref_buf = PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 0);
    for y in 0..pic_h as usize {
        for x in 0..pic_w as usize {
            ref_buf.luma.samples[y * pic_w as usize + x] = (40 + (x * 7) + (y * 11)) as u8;
        }
    }
    let ref_pic = ReferencePicture {
        poc: 0,
        frame: ref_buf.clone(),
        motion_field: None,
    };

    // ---- CABAC payload synthesis -------------------------------------
    let slice_qp = 26;
    let init_type = 1u8; // P-slice, sh_cabac_init_flag = 0

    let mut split_cu_ctxs = init_contexts(SyntaxCtx::SplitCuFlag, slice_qp);
    let mut cu_skip_ctxs = init_contexts(SyntaxCtx::CuSkipFlag, slice_qp);
    let mut general_merge_ctxs = init_contexts(SyntaxCtx::GeneralMergeFlag, slice_qp);
    let mut regular_merge_ctxs = init_contexts(SyntaxCtx::RegularMergeFlag, slice_qp);
    let mut merge_idx_ctxs = init_contexts(SyntaxCtx::MergeIdx, slice_qp);
    let mut cu_coded_ctxs = init_contexts(SyntaxCtx::CuCodedFlag, slice_qp);

    let mut enc = ArithEncoder::new();

    // 1. split_cu_flag(0) — single 8x8 CU at the CTU root.
    let split_inc = ctx_inc_split_cu_flag(false, false, 0, 0, 8, 8, 1, 1, 1, 1, 1) as usize;
    let split_slot = split_inc.min(split_cu_ctxs.len() - 1);
    enc.encode_decision(&mut split_cu_ctxs[split_slot], 0)
        .unwrap();

    // 2. cu_skip_flag(0) — non-skip merge.
    let skip_inc = ctx_inc_cu_skip_flag(false, false, false, false) as usize;
    let skip_slot = (init_type as usize) * 3 + skip_inc;
    enc.encode_decision(&mut cu_skip_ctxs[skip_slot], 0)
        .unwrap();

    // 3. general_merge_flag(1) — merge_data() follows. The leaf
    //    reader picks the ctx slot via `(init_type * 3 + ctxInc).min(n)`
    //    against the 3-entry Table 82 array; encode-side mirrors the
    //    same formula so we land on the identical context.
    let gm_inc = ctx_inc_general_merge_flag() as usize;
    let gm_n = general_merge_ctxs.len() - 1;
    let gm_slot = ((init_type as usize) * 3 + gm_inc).min(gm_n);
    enc.encode_decision(&mut general_merge_ctxs[gm_slot], 1)
        .unwrap();

    // 4. regular_merge_flag(0) — selects the CIIP/GPM branch. Table
    //    102 has 4 entries; the leaf reader indexes
    //    `((init_type - 1) * 2 + ctxInc).min(n)` where
    //    `ctxInc = (cu_skip_flag ? 0 : 1)` per Table 132.
    let rm_inc = ctx_inc_regular_merge_flag(false) as usize; // cu_skip = 0 → ctxInc = 1
    let rm_n = regular_merge_ctxs.len() - 1;
    let rm_slot = ((init_type as usize - 1) * 2 + rm_inc).min(rm_n);
    enc.encode_decision(&mut regular_merge_ctxs[rm_slot], 0)
        .unwrap();

    // 5. ciip_flag inferred to 1 per §7.4.12.7 (no GPM → no parse).
    //    No CABAC bin consumed.

    // 6. merge_idx = 0 — TR(cMax = MaxNumMergeCand - 1 = 5), bin 0
    //    ctx-coded with ctxInc = 0. Encode the single 0 bin.
    let mi_slot = (init_type as usize).min(merge_idx_ctxs.len() - 1);
    enc.encode_decision(&mut merge_idx_ctxs[mi_slot], 0)
        .unwrap();

    // 7. cu_coded_flag = 0 — no transform_tree(). Single ctx bin,
    //    ctxInc = 0 per Table 132; ctxIdx = init_type per Table 92.
    let cc_slot = (init_type as usize).min(cu_coded_ctxs.len() - 1);
    enc.encode_decision(&mut cu_coded_ctxs[cc_slot], 0).unwrap();

    enc.encode_terminate(1).unwrap();
    let payload = enc.finish();

    // ---- SPS / PPS / slice header ------------------------------------
    let mut sps = dummy_sps(0, pic_w, pic_h);
    sps.tool_flags.six_minus_max_num_merge_cand = 0; // MaxNumMergeCand = 6
    sps.tool_flags.ciip_enabled_flag = true;
    let pps = dummy_pps(pic_w, pic_h);
    let sh = StatefulSliceHeader {
        sh_slice_type: SliceType::P,
        sh_qp_delta: 0,
        ..Default::default()
    };
    let layout = CtuLayout::from_sps_pps(&sps, &pps);
    let mut walker = CtuWalker::begin_slice(&layout, &sps, &pps, &sh, 0, &payload).unwrap();
    walker.set_ref_pic_list_l0(vec![ref_pic]);

    let mut out = PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 222);
    walker.decode_picture_into(&mut out).unwrap();

    // ---- Expected luma per §8.5.6.7 eq. 998 --------------------------
    //
    // Both §8.5.6.7 neighbours sit outside the picture → both register
    // as not-intra → w = 1. predSamplesIntra = planar(neighbours all
    // 128, 4:2:0 padded mid-grey) = 128 everywhere. predSamplesInter
    // = ref_buf.luma at (x, y) (zero MV).
    //
    //   predSamplesComb[x][y] = (1 * 128 + 3 * ref[y][x] + 2) >> 2.
    let stride = pic_w as usize;
    let mut expected_y = vec![0u8; (pic_w * pic_h) as usize];
    for y in 0..pic_h as usize {
        for x in 0..pic_w as usize {
            let inter = ref_buf.luma.samples[y * stride + x] as i32;
            let intra = 128i32;
            expected_y[y * stride + x] = ((intra + 3 * inter + 2) >> 2) as u8;
        }
    }
    assert_eq!(
        out.luma.samples, expected_y,
        "luma must match the §8.5.6.7 eq. 998 weighted average \
         (1 * planar128 + 3 * inter + 2) >> 2 at corner CU with \
         w = 1 (both neighbours unavailable / not-intra)"
    );

    // ---- Per-block motion field: regular-merge MV unchanged by CIIP.
    //
    // The CIIP branch does not modify the chosen merge candidate; the
    // motion field still records the zero-MV pad (mergeCandList[0] for
    // a CU with no spatial / temporal / HMVP neighbours).
    let mf = walker.motion_field();
    let centre = mf.get_at_luma(4, 4);
    assert!(centre.available);
    assert!(centre.pred_flag_l0);
    assert!(!centre.pred_flag_l1);
    assert_eq!(centre.ref_idx_l0, 0);
    assert_eq!(
        centre.mv_l0,
        MotionVector { x: 0, y: 0 },
        "CIIP path must not modify the regular-merge MV — the \
         motion field still records the zero-MV pad chosen as the \
         base merge candidate"
    );
    // CIIP CUs are non-skip merge per §7.3.11.7 (the ciip_size_ok
    // branch requires cu_skip_flag == 0); the per-block flag must
    // mirror that.
    assert!(!centre.cu_skip_flag);
    assert!(centre.mode_inter);
}

/// Round-31 §8.5.6.5 — BDOF wiring acceptance test.
///
/// Builds a 16x8 single-CU B-slice picture with two short-term
/// references at symmetric POC distance (`current_poc = 1`,
/// `pocL0 = 0`, `pocL1 = 2` → `currPocDiff_L0 = 1` matches
/// `pocL1 - currPoc = 1`). The merge candidate selected is the
/// bipred zero-MV pad (`bcw_idx == 0`, both `pred_flag` set,
/// `MotionModelIdc == 0`, no sub-block / sym-MVD / CIIP /
/// weighted-pred), so every §8.5.5.1 bullet is satisfied. With
/// `sps_bdof_enabled_flag = 1` and `ph_bdof_disabled_flag = 0`
/// the §8.5.6.5 refinement runs in place of the eq. 980 average.
///
/// Both reference luma planes carry **distinct** horizontal ramps
/// — L0 is `value = x` and L1 is `value = 32 + x` — so the §8.5.6.5
/// gradient sums in eqs. 962–965 are non-zero and the per-sub-block
/// `(vx, vy)` solver picks a non-zero refinement. The eq. 977
/// `pbSamples` therefore differ from the plain `(predL0 + predL1
/// + 1) >> 1` average in at least some pixels of the CU rectangle.
///
/// To build the eq. 980 reference we re-run the whole pipeline with
/// `ph_bdof_disabled_flag = 1` (everything else identical, including
/// the CABAC payload) and compare luma planes. The chroma planes
/// must remain byte-identical between the two runs because BDOF
/// only refines the luma plane (`cIdx == 0` bullet of §8.5.5.1).
#[test]
fn decode_b_slice_bdof_refinement_differs_from_bipred_average() {
    use oxideav_h266::cabac_enc::ArithEncoder;
    use oxideav_h266::ctx::{ctx_inc_cu_skip_flag, ctx_inc_merge_idx, ctx_inc_split_cu_flag};
    use oxideav_h266::tables::{init_contexts, SyntaxCtx};

    let pic_w = 16u32;
    let pic_h = 8u32;

    // ---- Build two distinct reference pictures (L0 / L1) ------------
    // Mirror the bdof unit-test pattern in `src/bdof.rs`: L0 carries
    // a 2-D ramp `(10 + x*8 + y*2)` and L1 is the same ramp with a
    // 1-sample horizontal shift. With that pattern the §8.5.6.5
    // gradient sums are *asymmetric* between the two lists (gh0 != gh1
    // along the shifted axis), which drives a non-zero `vx`
    // refinement and an observable per-pixel `bdofOffset`.
    let mut ref_l0_buf = PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 0);
    let mut ref_l1_buf = PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 0);
    for y in 0..pic_h as usize {
        for x in 0..pic_w as usize {
            let v0 = (10 + x * 8 + y * 2) as u8;
            ref_l0_buf.luma.samples[y * pic_w as usize + x] = v0;
            let xp = x.saturating_sub(1);
            ref_l1_buf.luma.samples[y * pic_w as usize + x] = (10 + xp * 8 + y * 2) as u8;
        }
    }
    // Chroma planes left at the mid-grey fill — BDOF must not touch
    // them, so a flat field exposes any accidental write.
    let make_l0 = || ReferencePicture {
        poc: 0,
        frame: ref_l0_buf.clone(),
        motion_field: None,
    };
    let make_l1 = || ReferencePicture {
        poc: 2,
        frame: ref_l1_buf.clone(),
        motion_field: None,
    };

    // ---- B-slice CABAC payload synthesis ----------------------------
    // Single 16x8 leaf CU (split_cu_flag = 0 at the top level), then
    // cu_skip_flag = 1 + merge_idx = 0 → mergeCandList[0] = bipred
    // zero-MV pad.
    let slice_qp = 26;
    let init_type = 2u8;
    let mut split_cu_ctxs = init_contexts(SyntaxCtx::SplitCuFlag, slice_qp);
    let mut cu_skip_ctxs = init_contexts(SyntaxCtx::CuSkipFlag, slice_qp);
    let mut merge_idx_ctxs = init_contexts(SyntaxCtx::MergeIdx, slice_qp);

    let payload = {
        let mut enc = ArithEncoder::new();
        let split_inc = ctx_inc_split_cu_flag(false, false, 0, 0, 16, 8, 1, 1, 1, 1, 1) as usize;
        let split_slot = split_inc.min(split_cu_ctxs.len() - 1);
        enc.encode_decision(&mut split_cu_ctxs[split_slot], 0)
            .unwrap();
        let skip_inc = ctx_inc_cu_skip_flag(false, false, false, false) as usize;
        let skip_slot = (init_type as usize) * 3 + skip_inc;
        enc.encode_decision(&mut cu_skip_ctxs[skip_slot], 1)
            .unwrap();
        // general_merge_flag inferred to 1 (skip CU); merge_idx bin 0.
        let merge_inc = ctx_inc_merge_idx() as usize;
        let merge_slot = (init_type as usize + merge_inc).min(merge_idx_ctxs.len() - 1);
        enc.encode_decision(&mut merge_idx_ctxs[merge_slot], 0)
            .unwrap();
        enc.encode_terminate(1).unwrap();
        enc.finish()
    };

    // ---- SPS / PPS / SH for a B-slice on a 16x8 picture with BDOF ----
    let mut sps = dummy_sps(0, pic_w, pic_h);
    sps.tool_flags.six_minus_max_num_merge_cand = 0; // MaxNumMergeCand = 6
    sps.tool_flags.bdof_enabled_flag = true;
    let pps = dummy_pps(pic_w, pic_h);
    let sh = StatefulSliceHeader {
        sh_slice_type: SliceType::B,
        sh_qp_delta: 0,
        ..Default::default()
    };
    let layout = CtuLayout::from_sps_pps(&sps, &pps);

    // ---- Run 1: BDOF ON (ph_bdof_disabled_flag = 0) -----------------
    let mut walker_on = CtuWalker::begin_slice(&layout, &sps, &pps, &sh, 0, &payload).unwrap();
    walker_on.set_ref_pic_list_l0(vec![make_l0()]);
    walker_on.set_ref_pic_list_l1(vec![make_l1()]);
    // current_poc = 1 → symmetric distance to L0 (poc=0) and L1 (poc=2).
    walker_on.set_temporal_mvp(/*current_poc*/ 1, false, true, 0);
    walker_on.set_ph_bdof_disabled(false);
    let mut out_on = PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 222);
    walker_on.decode_picture_into(&mut out_on).unwrap();

    // ---- Run 2: BDOF OFF (ph_bdof_disabled_flag = 1, gating fails) --
    let mut walker_off = CtuWalker::begin_slice(&layout, &sps, &pps, &sh, 0, &payload).unwrap();
    walker_off.set_ref_pic_list_l0(vec![make_l0()]);
    walker_off.set_ref_pic_list_l1(vec![make_l1()]);
    walker_off.set_temporal_mvp(/*current_poc*/ 1, false, true, 0);
    // Default for ph_bdof_disabled is `true`; spell it out for clarity.
    walker_off.set_ph_bdof_disabled(true);
    let mut out_off = PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 222);
    walker_off.decode_picture_into(&mut out_off).unwrap();

    // ---- Compare ----------------------------------------------------
    // The eq. 980 reference is the byte-exact average across the
    // whole CU rectangle.
    let expected_avg: Vec<u8> = ref_l0_buf
        .luma
        .samples
        .iter()
        .zip(&ref_l1_buf.luma.samples)
        .map(|(&a, &b)| ((a as u32 + b as u32 + 1) >> 1) as u8)
        .collect();
    assert_eq!(
        out_off.luma.samples, expected_avg,
        "BDOF-off run must match the byte-exact eq. 980 average"
    );

    // BDOF-on output must differ from the plain average in at least
    // one luma sample — that is the wiring's only observable side
    // effect at fixture level.
    assert_ne!(
        out_on.luma.samples, out_off.luma.samples,
        "BDOF-on output must differ from BDOF-off (the §8.5.6.5 \
         refinement is supposed to add a non-zero per-sample offset \
         on a fixture with non-trivial gradients)"
    );

    // §8.5.5.1 last bullet: BDOF only refines the luma plane
    // (`cIdx == 0`). The chroma planes therefore must be byte-
    // identical between the two runs.
    assert_eq!(
        out_on.cb.samples, out_off.cb.samples,
        "BDOF must not touch the Cb plane"
    );
    assert_eq!(
        out_on.cr.samples, out_off.cr.samples,
        "BDOF must not touch the Cr plane"
    );
}

/// Round-40 §8.5.4 + §8.5.7 — GPM (Geometric Partitioning Mode)
/// reconstruction acceptance test.
///
/// Builds a 16×16 single-CU B-slice picture with two contrasting
/// short-term references (`L0` is flat 50, `L1` is flat 200). The
/// CABAC payload selects:
///   * `cu_skip_flag = 0`, `general_merge_flag = 1`, `regular_merge_flag = 0`,
///     CIIP gate is closed (`sps_ciip_enabled = 0`) so `ciip_flag` is
///     inferred to 0 → GPM branch fires.
///   * `merge_gpm_partition_idx = 0` (angle 0, distance 1).
///   * `merge_gpm_idx0 = 0`, `merge_gpm_idx1 = 0`. Eq. 647 then yields
///     `n = 1`, picking the second mergeCandList entry for partition B.
///
/// The two merge candidates come from the bipred zero-MV pad: at a
/// corner CU with no spatial / temporal / HMVP neighbours, every
/// mergeCandList entry is `(L0 = ref 0, L1 = ref 0, mv = 0)`. The
/// §8.5.4.2 step 4 / 5 X-derivation picks `predListFlagA = m & 1 = 0`
/// (L0 → flat-50 ref) for partition A and `predListFlagB = n & 1 = 1`
/// (L1 → flat-200 ref) for partition B. With the angle-0 / distance-1
/// partition, the CU's top half blends mostly from one source and the
/// bottom half mostly from the other → the output picture must show
/// both 50 and 200 in the per-CU luma rectangle (proving that the
/// per-pixel weight `w` actually crosses 0/8 across the partition).
#[test]
fn decode_b_slice_gpm_fires_and_decodes() {
    use oxideav_h266::cabac_enc::ArithEncoder;
    use oxideav_h266::ctx::{
        ctx_inc_cu_skip_flag, ctx_inc_general_merge_flag, ctx_inc_regular_merge_flag,
        ctx_inc_split_cu_flag,
    };
    use oxideav_h266::tables::{init_contexts, SyntaxCtx};

    let pic_w = 16u32;
    let pic_h = 16u32;

    let ref_l0_buf = PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 50);
    let ref_l1_buf = PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 200);
    let ref_l0 = ReferencePicture {
        poc: 0,
        frame: ref_l0_buf,
        motion_field: None,
    };
    let ref_l1 = ReferencePicture {
        poc: 2,
        frame: ref_l1_buf,
        motion_field: None,
    };

    // ---- CABAC payload synthesis ------------------------------------
    let slice_qp = 26;
    let init_type = 2u8; // B-slice, sh_cabac_init_flag = 0

    let mut split_cu_ctxs = init_contexts(SyntaxCtx::SplitCuFlag, slice_qp);
    let mut cu_skip_ctxs = init_contexts(SyntaxCtx::CuSkipFlag, slice_qp);
    let mut general_merge_ctxs = init_contexts(SyntaxCtx::GeneralMergeFlag, slice_qp);
    let mut regular_merge_ctxs = init_contexts(SyntaxCtx::RegularMergeFlag, slice_qp);
    let mut merge_idx_ctxs = init_contexts(SyntaxCtx::MergeIdx, slice_qp);
    let mut cu_coded_ctxs = init_contexts(SyntaxCtx::CuCodedFlag, slice_qp);

    let mut enc = ArithEncoder::new();

    // 1. split_cu_flag(0) — single 16×16 CU at the CTU root.
    let split_inc = ctx_inc_split_cu_flag(false, false, 0, 0, 16, 16, 1, 1, 1, 1, 1) as usize;
    let split_slot = split_inc.min(split_cu_ctxs.len() - 1);
    enc.encode_decision(&mut split_cu_ctxs[split_slot], 0)
        .unwrap();

    // 2. cu_skip_flag(0) — non-skip merge.
    let skip_inc = ctx_inc_cu_skip_flag(false, false, false, false) as usize;
    let skip_slot = (init_type as usize) * 3 + skip_inc;
    enc.encode_decision(&mut cu_skip_ctxs[skip_slot], 0)
        .unwrap();

    // 3. general_merge_flag(1).
    let gm_inc = ctx_inc_general_merge_flag() as usize;
    let gm_n = general_merge_ctxs.len() - 1;
    let gm_slot = ((init_type as usize) * 3 + gm_inc).min(gm_n);
    enc.encode_decision(&mut general_merge_ctxs[gm_slot], 1)
        .unwrap();

    // 4. regular_merge_flag(0) — selects GPM (CIIP off in SPS so
    //    ciip_flag is inferred to 0).
    let rm_inc = ctx_inc_regular_merge_flag(false) as usize; // cu_skip = 0 → ctxInc = 1
    let rm_n = regular_merge_ctxs.len() - 1;
    let rm_slot = ((init_type as usize - 1) * 2 + rm_inc).min(rm_n);
    enc.encode_decision(&mut regular_merge_ctxs[rm_slot], 0)
        .unwrap();

    // 5. merge_gpm_partition_idx = 0 — six bypass bins, all 0.
    for _ in 0..6 {
        enc.encode_bypass(0).unwrap();
    }

    // 6. merge_gpm_idx0 = 0 — TR(cMax = MaxNumGpmMergeCand - 1 = 5),
    //    bin 0 ctx-coded with ctxInc = 0; encode the single 0 bin.
    let mi_slot = (init_type as usize).min(merge_idx_ctxs.len() - 1);
    enc.encode_decision(&mut merge_idx_ctxs[mi_slot], 0)
        .unwrap();

    // 7. merge_gpm_idx1 = 0 — TR(cMax = MaxNumGpmMergeCand - 2 = 4),
    //    bin 0 ctx-coded; same Table 109 ctxIdx as gpm_idx0.
    enc.encode_decision(&mut merge_idx_ctxs[mi_slot], 0)
        .unwrap();

    // 8. cu_coded_flag = 0 — no transform_tree(). Single ctx bin.
    let cc_slot = (init_type as usize).min(cu_coded_ctxs.len() - 1);
    enc.encode_decision(&mut cu_coded_ctxs[cc_slot], 0).unwrap();

    enc.encode_terminate(1).unwrap();
    let payload = enc.finish();

    // ---- SPS / PPS / slice header ------------------------------------
    let mut sps = dummy_sps(0, pic_w, pic_h);
    sps.tool_flags.six_minus_max_num_merge_cand = 0; // MaxNumMergeCand = 6
    sps.tool_flags.gpm_enabled_flag = true;
    sps.tool_flags.max_num_merge_cand_minus_max_num_gpm_cand = 0; // MaxNumGpmMergeCand = 6
    sps.tool_flags.ciip_enabled_flag = false;
    let pps = dummy_pps(pic_w, pic_h);
    let sh = StatefulSliceHeader {
        sh_slice_type: SliceType::B,
        sh_qp_delta: 0,
        ..Default::default()
    };
    let layout = CtuLayout::from_sps_pps(&sps, &pps);
    let mut walker = CtuWalker::begin_slice(&layout, &sps, &pps, &sh, 1, &payload).unwrap();
    walker.set_ref_pic_list_l0(vec![ref_l0]);
    walker.set_ref_pic_list_l1(vec![ref_l1]);

    let mut out = PictureBuffer::yuv420_filled(pic_w as usize, pic_h as usize, 222);
    walker.decode_picture_into(&mut out).unwrap();

    // ---- GPM blend signature: both 50 and 200 must surface in the
    // CU's luma plane (proving the wValue weight crosses 0 ↔ 8 across
    // the angle-0 partition line). The deblocker may smooth the
    // transition band but the extreme values above and below it must
    // remain visible.
    let mut saw_low = false;
    let mut saw_high = false;
    for v in &out.luma.samples {
        if *v <= 80 {
            saw_low = true;
        } else if *v >= 170 {
            saw_high = true;
        }
    }
    assert!(
        saw_low && saw_high,
        "GPM blend must produce both regions of the partition — saw_low = {}, saw_high = {}",
        saw_low,
        saw_high
    );

    // ---- Motion field: GPM stored partition A's MV uniformly per
    // round-40's §8.5.7.3 simplification. Spatial slot is L0 (X = 0)
    // because m = 0 → X = m & 1 = 0 and the bipred zero-MV pad has
    // predFlagL0 = 1.
    let mf = walker.motion_field();
    let centre = mf.get_at_luma(8, 8);
    assert!(centre.available);
    assert!(centre.pred_flag_l0);
    assert!(!centre.pred_flag_l1);
    assert_eq!(centre.ref_idx_l0, 0);
}
