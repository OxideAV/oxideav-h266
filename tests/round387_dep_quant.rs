//! Round-387 — dependent-quantization encoder knob (§7.4.12.11 eq. 198
//! trellis) end-to-end through the IDR pipeline:
//!
//! * `EncoderConfig::dep_quant = true` quantises every TB with the
//!   greedy hard-decision TCQ (`quantize_tb_dep_quant`), reconstructs
//!   through the §8.7.3 dep-quant dequant arms and emits the residual
//!   bins with `RcOpts { dep_quant: true, .. }`.
//! * The wire signals `sps_dep_quant_enabled_flag = 1` and
//!   `sh_dep_quant_used_flag = 1` (§7.3.7 — and the SDH gate collapses).
//!
//! Spec reference: ITU-T H.266 | ISO/IEC 23090-3 (V4, 01/2026).

use oxideav_h266::encoder::EncoderConfig;
use oxideav_h266::encoder_pipeline::{encode_idr_with_residuals_cfg, psnr_y};
use oxideav_h266::nal::{extract_rbsp, iter_annex_b, NalUnitType};
use oxideav_h266::picture_header::parse_picture_header_stateful;
use oxideav_h266::pps::parse_pps;
use oxideav_h266::reconstruct::PictureBuffer;
use oxideav_h266::slice_header::{parse_slice_header_stateful, PhState};
use oxideav_h266::sps::parse_sps;

fn gradient_source(n: usize) -> PictureBuffer {
    let mut src = PictureBuffer::yuv420_filled(n, n, 128);
    for y in 0..n {
        for x in 0..n {
            src.luma.samples[y * src.luma.stride + x] =
                ((x * 3 + y * 2 + (x * y) / 7) % 200 + 20) as u8;
        }
    }
    for y in 0..n / 2 {
        for x in 0..n / 2 {
            src.cb.samples[y * src.cb.stride + x] = (100 + x * 2) as u8;
            src.cr.samples[y * src.cr.stride + x] = (150 - y) as u8;
        }
    }
    src
}

/// The dep-quant stream signals the switch in the SPS + slice header,
/// and the SDH / TS gates collapse per §7.3.7.
#[test]
fn dep_quant_pipeline_signals_sps_and_sh_flags() {
    let src = gradient_source(64);
    let mut cfg = EncoderConfig::new(64, 64);
    cfg.dep_quant = true;
    let (bs, _) = encode_idr_with_residuals_cfg(&src, 30, cfg).unwrap();

    let nals: Vec<_> = iter_annex_b(&bs).collect();
    let sps = parse_sps(&extract_rbsp(
        nals.iter()
            .find(|n| n.header.nal_unit_type == NalUnitType::SpsNut)
            .unwrap()
            .payload(),
    ))
    .unwrap();
    assert!(
        sps.tool_flags.dep_quant_enabled_flag,
        "sps_dep_quant_enabled_flag must be 1"
    );
    assert!(!sps.tool_flags.sign_data_hiding_enabled_flag);

    let pps = parse_pps(&extract_rbsp(
        nals.iter()
            .find(|n| n.header.nal_unit_type == NalUnitType::PpsNut)
            .unwrap()
            .payload(),
    ))
    .unwrap();
    let ph = parse_picture_header_stateful(
        &extract_rbsp(
            nals.iter()
                .find(|n| n.header.nal_unit_type == NalUnitType::PhNut)
                .unwrap()
                .payload(),
        ),
        &sps,
        &pps,
    )
    .unwrap();
    let ph_state = PhState {
        ph_inter_slice_allowed_flag: ph.ph_inter_slice_allowed_flag,
        ph_intra_slice_allowed_flag: ph.ph_intra_slice_allowed_flag,
        ph_alf_enabled_flag: ph.ph_alf_enabled_flag,
        ph_lmcs_enabled_flag: ph.ph_lmcs_enabled_flag,
        ph_explicit_scaling_list_enabled_flag: ph.ph_explicit_scaling_list_enabled_flag,
        ph_temporal_mvp_enabled_flag: ph.ph_temporal_mvp_enabled_flag,
        ph_sao_luma_enabled_flag: ph.ph_sao_luma_enabled_flag,
        ph_sao_chroma_enabled_flag: ph.ph_sao_chroma_enabled_flag,
        num_extra_sh_bits: 0,
        nal_unit_type: NalUnitType::IdrNLp,
    };
    let slice_nal = nals
        .iter()
        .find(|n| n.header.nal_unit_type == NalUnitType::IdrNLp)
        .expect("IDR slice");
    let sh = parse_slice_header_stateful(&extract_rbsp(slice_nal.payload()), &sps, &pps, &ph_state)
        .expect("SH parse");
    assert!(
        sh.sh_dep_quant_used_flag,
        "sh_dep_quant_used_flag must be 1"
    );
    assert!(
        !sh.sh_sign_data_hiding_used_flag,
        "SDH gate collapses when dep_quant is used (§7.3.7)"
    );
}

/// dep-quant reconstruction quality: at the same QP the trellis
/// quantizer's finer effective step must not fall behind the flat
/// quantizer's PSNR floor.
#[test]
fn dep_quant_pipeline_psnr_holds_the_floor() {
    let src = gradient_source(64);

    let (_, rec_flat) =
        encode_idr_with_residuals_cfg(&src, 26, EncoderConfig::new(64, 64)).unwrap();
    let psnr_flat = psnr_y(&src.luma, &rec_flat.luma).unwrap();

    let mut cfg = EncoderConfig::new(64, 64);
    cfg.dep_quant = true;
    let (_, rec_dq) = encode_idr_with_residuals_cfg(&src, 26, cfg).unwrap();
    let psnr_dq = psnr_y(&src.luma, &rec_dq.luma).unwrap();

    assert!(
        psnr_dq >= 30.0,
        "dep-quant reconstruction PSNR_Y {psnr_dq:.2} dB < 30 dB floor"
    );
    // The greedy TCQ runs at half the quantisation step of the flat
    // ladder ((qP + 1)-scaled levelScale + bdShift + 1), so its
    // distortion at equal QP must not be materially worse.
    assert!(
        psnr_dq + 1.0 >= psnr_flat,
        "dep-quant PSNR_Y {psnr_dq:.2} dB fell behind flat {psnr_flat:.2} dB"
    );
}

/// A dep-quant stream and a flat stream at the same QP differ on the
/// wire (the switch actually reaches the residual bins).
#[test]
fn dep_quant_changes_the_bitstream() {
    let src = gradient_source(64);
    let (bs_flat, _) = encode_idr_with_residuals_cfg(&src, 30, EncoderConfig::new(64, 64)).unwrap();
    let mut cfg = EncoderConfig::new(64, 64);
    cfg.dep_quant = true;
    let (bs_dq, _) = encode_idr_with_residuals_cfg(&src, 30, cfg).unwrap();
    assert_ne!(bs_flat, bs_dq);
}

/// Round-387 — sign-data-hiding encoder knob: the pipeline
/// parity-conditions each sub-block (§7.3.11.11 signHiddenFlag) and
/// signals `sps_sign_data_hiding_enabled_flag` + the SH bit.
#[test]
fn sdh_pipeline_signals_flags_and_holds_psnr() {
    let src = gradient_source(64);
    let mut cfg = EncoderConfig::new(64, 64);
    cfg.sign_data_hiding = true;
    let (bs, rec) = encode_idr_with_residuals_cfg(&src, 26, cfg).unwrap();

    let psnr = psnr_y(&src.luma, &rec.luma).unwrap();
    assert!(
        psnr >= 30.0,
        "SDH reconstruction PSNR_Y {psnr:.2} dB < 30 dB"
    );

    let nals: Vec<_> = iter_annex_b(&bs).collect();
    let sps = parse_sps(&extract_rbsp(
        nals.iter()
            .find(|n| n.header.nal_unit_type == NalUnitType::SpsNut)
            .unwrap()
            .payload(),
    ))
    .unwrap();
    assert!(sps.tool_flags.sign_data_hiding_enabled_flag);
    assert!(!sps.tool_flags.dep_quant_enabled_flag);

    let pps = parse_pps(&extract_rbsp(
        nals.iter()
            .find(|n| n.header.nal_unit_type == NalUnitType::PpsNut)
            .unwrap()
            .payload(),
    ))
    .unwrap();
    let ph = parse_picture_header_stateful(
        &extract_rbsp(
            nals.iter()
                .find(|n| n.header.nal_unit_type == NalUnitType::PhNut)
                .unwrap()
                .payload(),
        ),
        &sps,
        &pps,
    )
    .unwrap();
    let ph_state = PhState {
        ph_inter_slice_allowed_flag: ph.ph_inter_slice_allowed_flag,
        ph_intra_slice_allowed_flag: ph.ph_intra_slice_allowed_flag,
        ph_alf_enabled_flag: ph.ph_alf_enabled_flag,
        ph_lmcs_enabled_flag: ph.ph_lmcs_enabled_flag,
        ph_explicit_scaling_list_enabled_flag: ph.ph_explicit_scaling_list_enabled_flag,
        ph_temporal_mvp_enabled_flag: ph.ph_temporal_mvp_enabled_flag,
        ph_sao_luma_enabled_flag: ph.ph_sao_luma_enabled_flag,
        ph_sao_chroma_enabled_flag: ph.ph_sao_chroma_enabled_flag,
        num_extra_sh_bits: 0,
        nal_unit_type: NalUnitType::IdrNLp,
    };
    let slice_nal = nals
        .iter()
        .find(|n| n.header.nal_unit_type == NalUnitType::IdrNLp)
        .expect("IDR slice");
    let sh = parse_slice_header_stateful(&extract_rbsp(slice_nal.payload()), &sps, &pps, &ph_state)
        .expect("SH parse");
    assert!(!sh.sh_dep_quant_used_flag);
    assert!(sh.sh_sign_data_hiding_used_flag);
}

/// dep_quant + sign_data_hiding on the same config is rejected (§7.3.7).
#[test]
fn dep_quant_and_sdh_config_mutually_exclusive() {
    let src = gradient_source(64);
    let mut cfg = EncoderConfig::new(64, 64);
    cfg.dep_quant = true;
    cfg.sign_data_hiding = true;
    assert!(encode_idr_with_residuals_cfg(&src, 26, cfg).is_err());
}
