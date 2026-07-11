//! r412 — whole-stream decode conformance corpus.
//!
//! Every Annex-B stream the IDR encoder pipeline emits must decode
//! **byte-exactly** to the encoder's own reconstruction through the
//! full receive path: NAL walk → SPS / PPS / APS / PH / SH parsers →
//! the §9.3 CABAC CTU walker (SAO + ALF CTU prefixes in-stream,
//! §7.3.11 coding trees, §8.4 intra prediction with the §6.4.4
//! decode-order reference availability, §8.7 dequant + inverse
//! transforms) → the §8.8 in-loop filter chain (LMCS inverse map,
//! deblocking, SAO, ALF + CC-ALF).
//!
//! The corpus spans the tool axes the encoder can produce: QP sweep,
//! multi-CTU layouts (the §7.3.11.1 `end_of_slice_one_bit` placement),
//! chroma SAO, the MTT BT / TT pickers, LMCS (+ chroma residual
//! scaling), dependent quantization, and sign data hiding.
//!
//! Fixture generation commands and stream/plane SHA-256 hashes are
//! recorded in `tests/WHOLE_STREAM_CORPUS.md`, together with the
//! black-box reference-decoder validation notes.

use oxideav_h266::alf::{AlfApsBinding, AlfPicture};
use oxideav_h266::alf_syntax::AlfSyntaxConfig;
use oxideav_h266::aps::{parse_aps, ApsParamsType};
use oxideav_h266::ctu::{CtuLayout, CtuWalker};
use oxideav_h266::encoder::EncoderConfig;
use oxideav_h266::encoder_pipeline::{encode_idr_with_residuals, encode_idr_with_residuals_cfg};
use oxideav_h266::nal::{extract_rbsp, iter_annex_b, NalUnitType};
use oxideav_h266::picture_header::parse_picture_header_stateful;
use oxideav_h266::pps::parse_pps;
use oxideav_h266::reconstruct::PictureBuffer;
use oxideav_h266::slice_header::{parse_slice_header_stateful, PhState};
use oxideav_h266::sps::parse_sps;

/// Decode one single-slice IDR Annex-B stream end-to-end through the
/// crate's own parsers + CTU walker + in-loop filters.
fn decode_whole_stream(bs: &[u8]) -> PictureBuffer {
    let nals: Vec<_> = iter_annex_b(bs).collect();
    let find = |t: NalUnitType| {
        nals.iter()
            .find(|n| n.header.nal_unit_type == t)
            .unwrap_or_else(|| panic!("stream must carry a {t:?} NAL"))
    };
    let sps = parse_sps(&extract_rbsp(find(NalUnitType::SpsNut).payload())).expect("SPS parses");
    let pps = parse_pps(&extract_rbsp(find(NalUnitType::PpsNut).payload())).expect("PPS parses");

    // APS pool — ALF payloads keyed by id, plus the LMCS payload.
    let mut alf_apses = std::collections::HashMap::new();
    let mut lmcs_aps = None;
    for nal in nals
        .iter()
        .filter(|n| n.header.nal_unit_type == NalUnitType::PrefixApsNut)
    {
        let aps = parse_aps(&extract_rbsp(nal.payload())).expect("APS parses");
        match aps.aps_params_type {
            ApsParamsType::Alf => {
                alf_apses.insert(
                    aps.aps_adaptation_parameter_set_id,
                    aps.alf_data.clone().expect("ALF APS payload"),
                );
            }
            ApsParamsType::Lmcs => {
                lmcs_aps = Some(aps.lmcs_data.expect("LMCS APS payload"));
            }
            _ => {}
        }
    }

    let ph = parse_picture_header_stateful(
        &extract_rbsp(find(NalUnitType::PhNut).payload()),
        &sps,
        &pps,
    )
    .expect("PH parses");
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
    let slice_rbsp = extract_rbsp(find(NalUnitType::IdrNLp).payload());
    let sh = parse_slice_header_stateful(&slice_rbsp, &sps, &pps, &ph_state).expect("SH parses");

    let layout = CtuLayout::from_sps_pps(&sps, &pps);
    let mut cabac = sh.trailing_bits.clone();
    cabac.extend_from_slice(&[0u8; 64]);
    let mut walker =
        CtuWalker::begin_slice(&layout, &sps, &pps, &sh, 0, &cabac).expect("begin_slice");

    // §7.3.11.2 — enable the in-stream ALF CTU-prefix decode with the
    // APS-derived binarisation widths.
    if sh.sh_alf_enabled_flag {
        let chroma_aps = alf_apses.get(&sh.sh_alf_aps_id_chroma);
        let cc_cb_aps = alf_apses.get(&sh.sh_alf_cc_cb_aps_id);
        let cc_cr_aps = alf_apses.get(&sh.sh_alf_cc_cr_aps_id);
        walker.set_alf_decode(AlfSyntaxConfig {
            alf_enabled: true,
            cb_enabled: sh.sh_alf_cb_enabled_flag,
            cr_enabled: sh.sh_alf_cr_enabled_flag,
            cc_cb_enabled: sh.sh_alf_cc_cb_enabled_flag,
            cc_cr_enabled: sh.sh_alf_cc_cr_enabled_flag,
            sh_num_alf_aps_ids_luma: sh.sh_num_alf_aps_ids_luma,
            alf_chroma_num_alt_filters_minus1: chroma_aps
                .map(|a| a.alf_chroma_num_alt_filters_minus1 as u8)
                .unwrap_or(0),
            alf_cc_cb_filters_signalled_minus1: cc_cb_aps
                .map(|a| a.cc_cb_coeff.len().saturating_sub(1) as u8)
                .unwrap_or(0),
            alf_cc_cr_filters_signalled_minus1: cc_cr_aps
                .map(|a| a.cc_cr_coeff.len().saturating_sub(1) as u8)
                .unwrap_or(0),
            chroma_format_idc: sps.sps_chroma_format_idc as u32,
            slice_type: sh.sh_slice_type,
            sh_cabac_init_flag: sh.sh_cabac_init_flag,
        });
    }
    if sh.sh_lmcs_used_flag {
        walker
            .set_lmcs(
                lmcs_aps.as_ref().expect("LMCS APS shipped"),
                ph.ph_chroma_residual_scale_flag,
            )
            .expect("LMCS derives");
    }

    let w = sps.cropped_width() as usize;
    let h = sps.cropped_height() as usize;
    let mut out = PictureBuffer::yuv420_filled(w, h, 0);
    walker
        .decode_picture_into(&mut out)
        .expect("decode_picture_into");
    // §7.3.11.1 — the stream must end on end_of_slice_one_bit == 1.
    walker.finish_slice().expect("end_of_slice_one_bit");

    // §8.8 in-loop filters with the SH-referenced ALF APS bindings.
    let luma_slots: Vec<Option<&oxideav_h266::aps::AlfApsData>> = sh
        .sh_alf_aps_id_luma
        .iter()
        .map(|id| alf_apses.get(id))
        .collect();
    let binding = AlfApsBinding {
        luma_apses: &luma_slots,
        chroma_aps: if sh.sh_alf_cb_enabled_flag || sh.sh_alf_cr_enabled_flag {
            alf_apses.get(&sh.sh_alf_aps_id_chroma)
        } else {
            None
        },
        cc_cb_aps: if sh.sh_alf_cc_cb_enabled_flag {
            alf_apses.get(&sh.sh_alf_cc_cb_aps_id)
        } else {
            None
        },
        cc_cr_aps: if sh.sh_alf_cc_cr_enabled_flag {
            alf_apses.get(&sh.sh_alf_cc_cr_aps_id)
        } else {
            None
        },
    };
    walker
        .apply_in_loop_filters_with_alf(&mut out, &binding)
        .expect("in-loop filters");
    let _ = AlfPicture::empty(1, 1); // keep the import used on all paths
    out
}

/// Assert the decoded planes byte-match the encoder's reconstruction.
fn assert_byte_exact(dec: &PictureBuffer, rec: &PictureBuffer, tag: &str) {
    for (name, a, b, w) in [
        ("luma", &dec.luma, &rec.luma, dec.luma.width),
        ("cb", &dec.cb, &rec.cb, dec.cb.width),
        ("cr", &dec.cr, &rec.cr, dec.cr.width),
    ] {
        if a.samples != b.samples {
            let n = a
                .samples
                .iter()
                .zip(&b.samples)
                .filter(|(x, y)| x != y)
                .count();
            let first = a
                .samples
                .iter()
                .zip(&b.samples)
                .position(|(x, y)| x != y)
                .unwrap();
            let mut xmod = [0usize; 8];
            let mut ymod = [0usize; 8];
            let mut maxd = 0i32;
            for (i, (x, y)) in a.samples.iter().zip(&b.samples).enumerate() {
                if x != y {
                    xmod[(i % w) % 8] += 1;
                    ymod[(i / w) % 8] += 1;
                    maxd = maxd.max((*x as i32 - *y as i32).abs());
                }
            }
            panic!(
                "{tag}: {name} diverged — {n} samples differ; first at ({}, {}) dec {} rec {} maxd {} xmod8 {:?} ymod8 {:?}",
                first % w,
                first / w,
                a.samples[first],
                b.samples[first],
                maxd,
                xmod,
                ymod
            );
        }
    }
}

/// Deterministic structured test content: luma diagonal gradient +
/// block pattern, chroma ramps.
fn structured_source(w: usize, h: usize) -> PictureBuffer {
    let mut src = PictureBuffer::yuv420_filled(w, h, 128);
    for y in 0..h {
        for x in 0..w {
            let v = 40
                + ((x * 3 + y * 2) % 160) as u8
                + if (x / 16 + y / 16) % 2 == 0 { 20 } else { 0 };
            src.luma.samples[y * src.luma.stride + x] = v;
        }
    }
    for y in 0..h / 2 {
        for x in 0..w / 2 {
            src.cb.samples[y * src.cb.stride + x] = (96 + (x % 64)) as u8;
            src.cr.samples[y * src.cr.stride + x] = (160 - (y % 64)) as u8;
        }
    }
    src
}

/// Optional corpus dump for external black-box validation: when
/// `H266_CORPUS_DIR` is set, write the Annex-B stream and the decoded
/// planes (planar YUV 4:2:0, luma then Cb then Cr) under that
/// directory. `tests/WHOLE_STREAM_CORPUS.md` records the validation
/// commands + SHA-256 hashes.
fn dump_corpus(name: &str, bs: &[u8], dec: &PictureBuffer) {
    let Ok(dir) = std::env::var("H266_CORPUS_DIR") else {
        return;
    };
    let base = std::path::Path::new(&dir);
    std::fs::create_dir_all(base).expect("corpus dir");
    std::fs::write(base.join(format!("{name}.266")), bs).expect("write stream");
    let mut yuv =
        Vec::with_capacity(dec.luma.samples.len() + dec.cb.samples.len() + dec.cr.samples.len());
    yuv.extend_from_slice(&dec.luma.samples);
    yuv.extend_from_slice(&dec.cb.samples);
    yuv.extend_from_slice(&dec.cr.samples);
    std::fs::write(base.join(format!("{name}.yuv")), yuv).expect("write planes");
}

#[test]
fn whole_stream_default_qp26() {
    let src = structured_source(128, 128);
    let (bs, rec) = encode_idr_with_residuals(&src, 26).unwrap();
    let dec = decode_whole_stream(&bs);
    assert_byte_exact(&dec, &rec, "default qp26");
    dump_corpus("default_qp26", &bs, &dec);
}

#[test]
fn whole_stream_qp_sweep() {
    let src = structured_source(128, 128);
    for qp in [10, 17, 34, 45] {
        let (bs, rec) = encode_idr_with_residuals(&src, qp).unwrap();
        let dec = decode_whole_stream(&bs);
        assert_byte_exact(&dec, &rec, &format!("qp {qp}"));
        dump_corpus(&format!("qp{qp}"), &bs, &dec);
    }
}

#[test]
fn whole_stream_flat_source() {
    let src = PictureBuffer::yuv420_filled(128, 128, 128);
    let (bs, rec) = encode_idr_with_residuals(&src, 26).unwrap();
    let dec = decode_whole_stream(&bs);
    assert_byte_exact(&dec, &rec, "flat");
    dump_corpus("flat_qp26", &bs, &dec);
}

/// Multi-CTU-row / multi-CTU-column layout — pins the §7.3.11.1
/// `end_of_slice_one_bit` placement fix (the pre-r412 pipeline emitted
/// a spec-divergent terminate bin after every CTU).
#[test]
fn whole_stream_multi_ctu_256x256() {
    let src = structured_source(256, 256);
    let (bs, rec) = encode_idr_with_residuals(&src, 26).unwrap();
    let dec = decode_whole_stream(&bs);
    assert_byte_exact(&dec, &rec, "256x256");
    dump_corpus("multi_ctu_256x256", &bs, &dec);
}

#[test]
fn whole_stream_chroma_sao_merge() {
    let src = structured_source(256, 128);
    let mut cfg = EncoderConfig::new(256, 128);
    cfg.enable_chroma_sao_merge = true;
    let (bs, rec) = encode_idr_with_residuals_cfg(&src, 26, cfg).unwrap();
    let dec = decode_whole_stream(&bs);
    assert_byte_exact(&dec, &rec, "chroma SAO merge");
    dump_corpus("chroma_sao_merge", &bs, &dec);
}

#[test]
fn whole_stream_mtt_bt_picker() {
    let src = structured_source(128, 128);
    let mut cfg = EncoderConfig::new(128, 128);
    cfg.enable_mtt_bt_picker = true;
    let (bs, rec) = encode_idr_with_residuals_cfg(&src, 30, cfg).unwrap();
    let dec = decode_whole_stream(&bs);
    assert_byte_exact(&dec, &rec, "MTT BT");
    dump_corpus("mtt_bt", &bs, &dec);
}

#[test]
fn whole_stream_mtt_tt_picker() {
    let src = structured_source(128, 128);
    let mut cfg = EncoderConfig::new(128, 128);
    cfg.enable_mtt_bt_picker = true;
    cfg.enable_mtt_tt_picker = true;
    let (bs, rec) = encode_idr_with_residuals_cfg(&src, 30, cfg).unwrap();
    let dec = decode_whole_stream(&bs);
    assert_byte_exact(&dec, &rec, "MTT BT+TT");
    dump_corpus("mtt_bt_tt", &bs, &dec);
}

/// The LMCS payload the encoder ships (mirrors the crate's LMCS
/// encoder-path fixtures: one shrunk bin at index 0).
fn lmcs_payload() -> oxideav_h266::lmcs::LmcsData {
    let mut abs_cw = [0u32; oxideav_h266::lmcs::LMCS_NUM_BINS];
    let mut sign_cw = [false; oxideav_h266::lmcs::LMCS_NUM_BINS];
    abs_cw[0] = 8;
    sign_cw[0] = true;
    oxideav_h266::lmcs::LmcsData {
        lmcs_min_bin_idx: 0,
        lmcs_delta_max_bin_idx: 0,
        lmcs_delta_cw_prec_minus1: 3,
        lmcs_delta_abs_cw: abs_cw,
        lmcs_delta_sign_cw_flag: sign_cw,
        lmcs_delta_abs_crs: 0,
        lmcs_delta_sign_crs_flag: false,
    }
}

#[test]
fn whole_stream_lmcs() {
    let src = structured_source(128, 128);
    let mut cfg = EncoderConfig::new(128, 128);
    cfg.lmcs = Some(lmcs_payload());
    let (bs, rec) = encode_idr_with_residuals_cfg(&src, 26, cfg).unwrap();
    let dec = decode_whole_stream(&bs);
    assert_byte_exact(&dec, &rec, "LMCS");
    dump_corpus("lmcs", &bs, &dec);
}

#[test]
fn whole_stream_lmcs_chroma_scaling() {
    let src = structured_source(128, 128);
    let mut cfg = EncoderConfig::new(128, 128);
    cfg.lmcs = Some(lmcs_payload());
    cfg.lmcs_chroma_scaling = true;
    let (bs, rec) = encode_idr_with_residuals_cfg(&src, 26, cfg).unwrap();
    let dec = decode_whole_stream(&bs);
    assert_byte_exact(&dec, &rec, "LMCS + chroma scaling");
    dump_corpus("lmcs_chroma_scaling", &bs, &dec);
}

#[test]
fn whole_stream_dep_quant() {
    let src = structured_source(128, 128);
    let mut cfg = EncoderConfig::new(128, 128);
    cfg.dep_quant = true;
    let (bs, rec) = encode_idr_with_residuals_cfg(&src, 26, cfg).unwrap();
    let dec = decode_whole_stream(&bs);
    assert_byte_exact(&dec, &rec, "dep-quant");
    dump_corpus("dep_quant", &bs, &dec);
}

#[test]
fn whole_stream_sign_data_hiding() {
    let src = structured_source(128, 128);
    let mut cfg = EncoderConfig::new(128, 128);
    cfg.sign_data_hiding = true;
    let (bs, rec) = encode_idr_with_residuals_cfg(&src, 26, cfg).unwrap();
    let dec = decode_whole_stream(&bs);
    assert_byte_exact(&dec, &rec, "sign data hiding");
    dump_corpus("sign_data_hiding", &bs, &dec);
}
