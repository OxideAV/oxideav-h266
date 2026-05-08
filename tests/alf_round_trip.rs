//! Round-45 — integration tests for the ALF bitstream emit / decode
//! round-trip.
//!
//! Covers two scenarios:
//!
//! 1. Per-picture ALF CABAC syntax: encode a hand-crafted `AlfPicture`
//!    via `alf_syntax::encode_alf_picture`, decode it back through
//!    `alf_syntax::decode_alf_picture`, and verify every CTB's
//!    `(luma_on / cb_on / cr_on / luma_filt_set_idx / *_alt_idx /
//!     cc_*_idc)` matches.
//! 2. ALF APS NAL emission: encode a synthesised chroma + CC-ALF APS
//!    via `aps_enc::emit_alf_aps_rbsp`, wrap into a NAL, then walk
//!    the produced Annex-B stream with the existing `nal::iter_annex_b`
//!    + `aps::parse_aps` machinery to verify the §7.3.2.18
//!      transcription is invertible.

use oxideav_h266::alf::{AlfCtb, AlfPicture};
use oxideav_h266::alf_syntax::{decode_alf_picture, encode_alf_picture, AlfCtxs, AlfSyntaxConfig};
use oxideav_h266::aps::{parse_aps, ApsParamsType, ALF_CC_NUM_COEFF, ALF_CHROMA_NUM_COEFF};
use oxideav_h266::aps_enc::emit_alf_aps_rbsp;
use oxideav_h266::cabac::ArithDecoder;
use oxideav_h266::cabac_enc::ArithEncoder;
use oxideav_h266::nal::{extract_rbsp, iter_annex_b, NalUnitType};
use oxideav_h266::slice_header::SliceType;

fn pad(bytes: Vec<u8>) -> Vec<u8> {
    let mut out = bytes;
    out.extend_from_slice(&[0u8; 32]);
    out
}

fn encoder_pipeline_cfg() -> AlfSyntaxConfig {
    AlfSyntaxConfig {
        alf_enabled: true,
        cb_enabled: true,
        cr_enabled: true,
        cc_cb_enabled: true,
        cc_cr_enabled: true,
        sh_num_alf_aps_ids_luma: 0,
        alf_chroma_num_alt_filters_minus1: 1,
        alf_cc_cb_filters_signalled_minus1: 1,
        alf_cc_cr_filters_signalled_minus1: 1,
        chroma_format_idc: 1,
        slice_type: SliceType::I,
        sh_cabac_init_flag: false,
    }
}

/// Round-45 — encode a 4×3 grid of mixed ALF decisions, decode it back,
/// every CTB matches.
#[test]
fn alf_picture_round_trip_4x3() {
    let cfg = encoder_pipeline_cfg();

    let mut alf_pic = AlfPicture::empty(4, 3);
    // CTB(0,0): luma off, Cb on with alt 0, CC-Cb idc 1.
    alf_pic.set(
        0,
        0,
        AlfCtb {
            cb_on: true,
            cb_alt_idx: 0,
            cc_cb_idc: 1,
            ..AlfCtb::default()
        },
    );
    // CTB(1,0): luma on with fixed-filter idx 11, Cb on with alt 1.
    alf_pic.set(
        1,
        0,
        AlfCtb {
            luma_on: true,
            luma_filt_set_idx: 11,
            cb_on: true,
            cb_alt_idx: 1,
            ..AlfCtb::default()
        },
    );
    // CTB(2,0): all off (default).
    // CTB(3,0): luma on idx 0, Cr on alt 1, CC-Cr idc 2.
    alf_pic.set(
        3,
        0,
        AlfCtb {
            luma_on: true,
            luma_filt_set_idx: 0,
            cr_on: true,
            cr_alt_idx: 1,
            cc_cr_idc: 2,
            ..AlfCtb::default()
        },
    );
    // Row 1: every CTB luma on with idx 7.
    for rx in 0..4 {
        alf_pic.set(
            rx,
            1,
            AlfCtb {
                luma_on: true,
                luma_filt_set_idx: 7,
                ..AlfCtb::default()
            },
        );
    }
    // Row 2: alternating chroma decisions to exercise neighbour-context
    // ctxInc derivation across rows.
    alf_pic.set(
        0,
        2,
        AlfCtb {
            cb_on: true,
            cr_on: true,
            cb_alt_idx: 0,
            cr_alt_idx: 1,
            cc_cb_idc: 1,
            cc_cr_idc: 1,
            ..AlfCtb::default()
        },
    );
    alf_pic.set(1, 2, AlfCtb::default());
    alf_pic.set(
        2,
        2,
        AlfCtb {
            cb_on: true,
            cb_alt_idx: 1,
            cc_cb_idc: 2,
            ..AlfCtb::default()
        },
    );
    alf_pic.set(
        3,
        2,
        AlfCtb {
            luma_on: true,
            luma_filt_set_idx: 15,
            ..AlfCtb::default()
        },
    );

    // Encode.
    let mut enc = ArithEncoder::new();
    let mut enc_ctxs = AlfCtxs::init(26);
    encode_alf_picture(&mut enc, &mut enc_ctxs, &cfg, &alf_pic).unwrap();
    let bytes = pad(enc.finish());

    // Decode.
    let mut dec = ArithDecoder::new(&bytes).unwrap();
    let mut dec_ctxs = AlfCtxs::init(26);
    let mut dec_pic = AlfPicture::empty(4, 3);
    decode_alf_picture(&mut dec, &mut dec_ctxs, &cfg, &mut dec_pic).unwrap();

    // Verify every CTB.
    for ry in 0..3u32 {
        for rx in 0..4u32 {
            let e = alf_pic.get(rx, ry);
            let d = dec_pic.get(rx, ry);
            assert_eq!(d.luma_on, e.luma_on, "luma_on at ({rx},{ry})");
            assert_eq!(d.cb_on, e.cb_on, "cb_on at ({rx},{ry})");
            assert_eq!(d.cr_on, e.cr_on, "cr_on at ({rx},{ry})");
            if e.luma_on {
                assert_eq!(
                    d.luma_filt_set_idx, e.luma_filt_set_idx,
                    "luma_filt_set_idx at ({rx},{ry})"
                );
            }
            if e.cb_on {
                assert_eq!(d.cb_alt_idx, e.cb_alt_idx, "cb_alt_idx at ({rx},{ry})");
            }
            if e.cr_on {
                assert_eq!(d.cr_alt_idx, e.cr_alt_idx, "cr_alt_idx at ({rx},{ry})");
            }
            assert_eq!(d.cc_cb_idc, e.cc_cb_idc, "cc_cb_idc at ({rx},{ry})");
            assert_eq!(d.cc_cr_idc, e.cc_cr_idc, "cc_cr_idc at ({rx},{ry})");
        }
    }
}

/// Round-45 — wrapping the emitted ALF APS RBSP into a NAL header
/// + emulation-prevention pass + Annex-B prefix yields a stream the
///   existing parser walks back to the same `AlfApsData`.
#[test]
fn alf_aps_nal_round_trip() {
    use oxideav_h266::aps::AlfApsData;
    use oxideav_h266::encoder::{insert_emulation_prevention, nal_header_bytes};

    // Build a chroma + CC-ALF APS.
    let mut row_chroma = [0i32; ALF_CHROMA_NUM_COEFF];
    row_chroma[0] = 1;
    row_chroma[1] = -2;
    row_chroma[2] = 3;
    row_chroma[5] = -4;
    let mut row_cc_cb = [0i32; ALF_CC_NUM_COEFF];
    row_cc_cb[1] = 2;
    row_cc_cb[2] = -2;
    let mut row_cc_cr = [0i32; ALF_CC_NUM_COEFF];
    row_cc_cr[3] = 5;
    row_cc_cr[6] = -6;

    let alf = AlfApsData {
        alf_chroma_filter_signal_flag: true,
        alf_chroma_num_alt_filters_minus1: 0,
        chroma_coeff: vec![row_chroma],
        chroma_clip_idx: vec![[0u8; ALF_CHROMA_NUM_COEFF]],
        alf_cc_cb_filter_signal_flag: true,
        alf_cc_cr_filter_signal_flag: true,
        cc_cb_coeff: vec![row_cc_cb],
        cc_cr_coeff: vec![row_cc_cr],
        ..AlfApsData::default()
    };

    // Emit RBSP + wrap into a complete Annex-B NAL byte stream.
    let rbsp = emit_alf_aps_rbsp(2, true, &alf).unwrap();
    let mut nal = Vec::new();
    nal.extend_from_slice(&nal_header_bytes(NalUnitType::PrefixApsNut, 0, 1));
    nal.extend_from_slice(&insert_emulation_prevention(&rbsp));
    let mut annex_b = vec![0x00, 0x00, 0x00, 0x01];
    annex_b.extend_from_slice(&nal);

    // Walk through the iter_annex_b path and parse via the existing APS
    // parser.
    let nals: Vec<_> = iter_annex_b(&annex_b).collect();
    assert_eq!(nals.len(), 1);
    assert_eq!(nals[0].header.nal_unit_type, NalUnitType::PrefixApsNut);
    let parsed_rbsp = extract_rbsp(nals[0].payload());
    let parsed = parse_aps(&parsed_rbsp).expect("parse_aps must succeed on emitted ALF APS NAL");
    assert_eq!(parsed.aps_params_type, ApsParamsType::Alf);
    assert_eq!(parsed.aps_adaptation_parameter_set_id, 2);
    let p = parsed.alf_data.as_ref().expect("alf_data");
    assert!(p.alf_chroma_filter_signal_flag);
    assert!(p.alf_cc_cb_filter_signal_flag);
    assert!(p.alf_cc_cr_filter_signal_flag);
    assert_eq!(p.chroma_coeff[0], row_chroma);
    assert_eq!(p.cc_cb_coeff[0], row_cc_cb);
    assert_eq!(p.cc_cr_coeff[0], row_cc_cr);
}

/// Round-45 — the full encoder pipeline must include two ALF APS NALs
/// (chroma + CC-ALF) that parse cleanly through `parse_aps`.
#[test]
fn encoder_pipeline_emits_alf_aps_nals() {
    use oxideav_h266::encoder_pipeline::encode_idr_with_residuals;
    use oxideav_h266::reconstruct::PictureBuffer;

    let src = PictureBuffer::yuv420_filled(64, 64, 128);
    let (bs, _) = encode_idr_with_residuals(&src, 26).unwrap();

    // Walk all NALs, count APS NALs and confirm each parses as an ALF
    // APS with the round-45 wire layout.
    let mut alf_aps_count = 0u32;
    let mut seen_chroma_signal = false;
    let mut seen_cc_signal = false;
    for nal in iter_annex_b(&bs) {
        if nal.header.nal_unit_type != NalUnitType::PrefixApsNut {
            continue;
        }
        alf_aps_count += 1;
        let rbsp = extract_rbsp(nal.payload());
        let aps = parse_aps(&rbsp).expect("APS NAL must parse");
        assert_eq!(aps.aps_params_type, ApsParamsType::Alf);
        let p = aps.alf_data.as_ref().expect("ALF APS payload");
        if p.alf_chroma_filter_signal_flag {
            seen_chroma_signal = true;
        }
        if p.alf_cc_cb_filter_signal_flag || p.alf_cc_cr_filter_signal_flag {
            seen_cc_signal = true;
        }
    }
    assert_eq!(alf_aps_count, 2, "pipeline must emit 2 ALF APS NALs");
    assert!(seen_chroma_signal, "one APS must carry primary chroma ALF");
    assert!(seen_cc_signal, "one APS must carry CC-ALF");
}

/// Round-46 — the encoder pipeline now flips `sps_alf_enabled_flag` +
/// `sps_ccalf_enabled_flag` on, and emits the §7.3.2.8 PH-level ALF
/// chain with primary chroma bound to APS id 0 and CC-ALF bound to
/// APS id 1. Parsing the bitstream must surface every flag.
#[test]
fn encoder_pipeline_ph_alf_chain_round_trip() {
    use oxideav_h266::encoder_pipeline::encode_idr_with_residuals;
    use oxideav_h266::picture_header::parse_picture_header_stateful;
    use oxideav_h266::pps::parse_pps;
    use oxideav_h266::reconstruct::PictureBuffer;
    use oxideav_h266::sps::parse_sps;

    let src = PictureBuffer::yuv420_filled(128, 128, 100);
    let (bs, _) = encode_idr_with_residuals(&src, 26).unwrap();

    let nals: Vec<_> = iter_annex_b(&bs).collect();
    // Identify SPS / PPS / PH for the stateful parsers.
    let sps_nal = nals
        .iter()
        .find(|n| n.header.nal_unit_type == NalUnitType::SpsNut)
        .expect("SPS");
    let pps_nal = nals
        .iter()
        .find(|n| n.header.nal_unit_type == NalUnitType::PpsNut)
        .expect("PPS");
    let ph_nal = nals
        .iter()
        .find(|n| n.header.nal_unit_type == NalUnitType::PhNut)
        .expect("PH");

    let sps = parse_sps(&extract_rbsp(sps_nal.payload())).expect("SPS parse");
    assert!(
        sps.tool_flags.alf_enabled_flag,
        "sps_alf_enabled_flag must be 1"
    );
    assert!(
        sps.tool_flags.ccalf_enabled_flag,
        "sps_ccalf_enabled_flag must be 1"
    );

    let pps = parse_pps(&extract_rbsp(pps_nal.payload())).expect("PPS parse");
    assert!(
        pps.pps_alf_info_in_ph_flag,
        "pps_no_pic_partition_flag = 1 → pps_alf_info_in_ph_flag = 1 (inferred)"
    );

    let ph = parse_picture_header_stateful(&extract_rbsp(ph_nal.payload()), &sps, &pps)
        .expect("PH parse");
    assert!(ph.ph_alf_enabled_flag, "ph_alf_enabled_flag must be 1");
    assert_eq!(ph.ph_num_alf_aps_ids_luma, 0);
    assert!(ph.ph_alf_cb_enabled_flag);
    assert!(ph.ph_alf_cr_enabled_flag);
    assert_eq!(
        ph.ph_alf_aps_id_chroma, 0,
        "primary chroma ALF must bind to APS id 0"
    );
    assert!(ph.ph_alf_cc_cb_enabled_flag);
    assert_eq!(ph.ph_alf_cc_cb_aps_id, 1, "CC-ALF Cb must bind to APS id 1");
    assert!(ph.ph_alf_cc_cr_enabled_flag);
    assert_eq!(ph.ph_alf_cc_cr_aps_id, 1, "CC-ALF Cr must bind to APS id 1");
}

/// Round-46 — slice-header parse: with `pps_alf_info_in_ph_flag = 1`
/// (inferred from `pps_no_pic_partition_flag = 1`), the slice-header
/// ALF block is suppressed but the PH-level ALF state propagates to
/// the slice-header `PhState`. The IDR slice RBSP's CABAC tail must
/// then decode through `decode_alf_picture` followed by the per-CTU
/// `encode_terminate(1)` exit condition without choking on the ALF
/// bins inlined ahead of the residual bins.
#[test]
fn encoder_pipeline_slice_alf_cabac_inlined_in_ctu_walk() {
    use oxideav_h266::alf::AlfPicture;
    use oxideav_h266::alf_syntax::{decode_alf_picture, AlfCtxs, AlfSyntaxConfig};
    use oxideav_h266::cabac::ArithDecoder;
    use oxideav_h266::encoder_pipeline::encode_idr_with_residuals;
    use oxideav_h266::picture_header::parse_picture_header_stateful;
    use oxideav_h266::pps::parse_pps;
    use oxideav_h266::reconstruct::PictureBuffer;
    use oxideav_h266::slice_header::{parse_slice_header_stateful, PhState};
    use oxideav_h266::sps::parse_sps;

    // 128×128 IDR — 1×1 CTU grid (CTB = 128).
    let src = PictureBuffer::yuv420_filled(128, 128, 128);
    let (bs, _) = encode_idr_with_residuals(&src, 26).unwrap();

    let nals: Vec<_> = iter_annex_b(&bs).collect();
    let sps_rbsp = extract_rbsp(
        nals.iter()
            .find(|n| n.header.nal_unit_type == NalUnitType::SpsNut)
            .unwrap()
            .payload(),
    );
    let sps = parse_sps(&sps_rbsp).unwrap();
    let pps_rbsp = extract_rbsp(
        nals.iter()
            .find(|n| n.header.nal_unit_type == NalUnitType::PpsNut)
            .unwrap()
            .payload(),
    );
    let pps = parse_pps(&pps_rbsp).unwrap();
    let ph_rbsp = extract_rbsp(
        nals.iter()
            .find(|n| n.header.nal_unit_type == NalUnitType::PhNut)
            .unwrap()
            .payload(),
    );
    let ph = parse_picture_header_stateful(&ph_rbsp, &sps, &pps).unwrap();

    // PH ALF state propagates into the slice-header parser via PhState.
    let ph_state = PhState {
        ph_inter_slice_allowed_flag: ph.ph_inter_slice_allowed_flag,
        ph_intra_slice_allowed_flag: ph.ph_intra_slice_allowed_flag,
        ph_alf_enabled_flag: ph.ph_alf_enabled_flag,
        ph_lmcs_enabled_flag: ph.ph_lmcs_enabled_flag,
        ph_explicit_scaling_list_enabled_flag: ph.ph_explicit_scaling_list_enabled_flag,
        ph_temporal_mvp_enabled_flag: ph.ph_temporal_mvp_enabled_flag,
        num_extra_sh_bits: 0,
        nal_unit_type: NalUnitType::IdrNLp,
    };
    assert!(ph_state.ph_alf_enabled_flag);

    let slice_nal = nals
        .iter()
        .find(|n| n.header.nal_unit_type == NalUnitType::IdrNLp)
        .expect("IDR slice");
    let slice_rbsp = extract_rbsp(slice_nal.payload());
    let sh = parse_slice_header_stateful(&slice_rbsp, &sps, &pps, &ph_state).expect("slice header");

    // pps_alf_info_in_ph_flag = 1 → no SH-level ALF block; SH default.
    assert!(
        !sh.sh_alf_enabled_flag,
        "PH-carried ALF must suppress SH ALF block"
    );

    // The CABAC payload starts after the byte-aligned SH; the first
    // CABAC bin is the §7.3.11.2 alf_ctb_flag[0] for CTU(0,0).
    let cabac_bytes = sh.trailing_bits.clone();
    assert!(!cabac_bytes.is_empty(), "slice RBSP must carry CABAC bytes");

    // Decode the inlined ALF bins for the 1×1 CTU grid.
    let cfg = AlfSyntaxConfig {
        alf_enabled: true,
        cb_enabled: ph.ph_alf_cb_enabled_flag,
        cr_enabled: ph.ph_alf_cr_enabled_flag,
        cc_cb_enabled: ph.ph_alf_cc_cb_enabled_flag,
        cc_cr_enabled: ph.ph_alf_cc_cr_enabled_flag,
        sh_num_alf_aps_ids_luma: ph.ph_num_alf_aps_ids_luma,
        // The chroma APS the encoder ships carries one alt filter
        // (`alf_chroma_num_alt_filters_minus1 = 0`) so the per-CTB
        // alt-idx fields are not transmitted; same for CC-ALF
        // (`alf_cc_*_filters_signalled_minus1 = 0` → cMax = 1).
        alf_chroma_num_alt_filters_minus1: 0,
        alf_cc_cb_filters_signalled_minus1: 0,
        alf_cc_cr_filters_signalled_minus1: 0,
        chroma_format_idc: sps.sps_chroma_format_idc as u32,
        slice_type: SliceType::I,
        sh_cabac_init_flag: false,
    };
    let padded = pad(cabac_bytes);
    let mut dec = ArithDecoder::new(&padded).unwrap();
    let mut ctxs = AlfCtxs::init(26);
    let mut alf_pic = AlfPicture::empty(1, 1);
    decode_alf_picture(&mut dec, &mut ctxs, &cfg, &mut alf_pic)
        .expect("ALF CABAC inlined in CTU walk must decode through decode_alf_picture");

    // Flat-grey input → ALF RDO leaves every per-CTB flag off (no
    // SSE wins are possible on a constant plane). The decoded
    // alf_pic must therefore mirror the encoder's all-off decision.
    let ctb = alf_pic.get(0, 0);
    assert!(!ctb.luma_on, "luma ALF off on flat source");
    assert!(!ctb.cb_on, "Cb ALF off on flat source");
    assert!(!ctb.cr_on, "Cr ALF off on flat source");
    assert_eq!(ctb.cc_cb_idc, 0, "CC-ALF Cb idc 0 on flat source");
    assert_eq!(ctb.cc_cr_idc, 0, "CC-ALF Cr idc 0 on flat source");
}
