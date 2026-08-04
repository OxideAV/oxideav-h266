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

/// Decode one IDR Annex-B stream (single- or multi-slice) end-to-end
/// through the crate's own parsers + CTU walker + in-loop filters.
/// Multi-slice pictures walk slice 0 via `begin_slice` and every
/// further slice via `continue_slice` (r431), with the §8.8 filter
/// pass running once after the last slice.
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
        ..Default::default()
    };
    // r431 — every VCL NAL of the picture, in decode order.
    let slice_headers: Vec<_> = nals
        .iter()
        .filter(|n| n.header.nal_unit_type == NalUnitType::IdrNLp)
        .map(|n| {
            parse_slice_header_stateful(&extract_rbsp(n.payload()), &sps, &pps, &ph_state)
                .expect("SH parses")
        })
        .collect();
    assert!(!slice_headers.is_empty(), "stream carries no VCL NAL");
    let cabacs: Vec<Vec<u8>> = slice_headers
        .iter()
        .map(|sh| {
            let mut c = sh.trailing_bits.clone();
            c.extend_from_slice(&[0u8; 64]);
            c
        })
        .collect();
    let sh = &slice_headers[0];

    let layout = CtuLayout::from_sps_pps(&sps, &pps);
    let mut walker =
        CtuWalker::begin_slice(&layout, &sps, &pps, sh, 0, &cabacs[0]).expect("begin_slice");

    // r418 — §7.4.3.7 eq. 123: CuQpDeltaSubdiv for an I-slice is the
    // PH-signalled `ph_cu_qp_delta_subdiv_intra_slice`. Driving the
    // walker's quantization-group declarations from the wire value
    // (instead of the per-CU-arming default) exercises the §7.3.11.4
    // QG logic on every corpus stream.
    if pps.pps_cu_qp_delta_enabled_flag {
        walker.set_cu_qp_delta_subdiv(ph.ph_cu_qp_delta_subdiv_intra_slice);
    }

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
    for (i, sh_i) in slice_headers.iter().enumerate() {
        if i > 0 {
            walker
                .continue_slice(sh_i, 0, &cabacs[i])
                .expect("continue_slice");
        }
        walker
            .decode_picture_into(&mut out)
            .expect("decode_picture_into");
        // §7.3.11.1 — each slice ends on end_of_slice_one_bit == 1.
        walker.finish_slice().expect("end_of_slice_one_bit");
    }

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
                + ((x * 3 + y * 2) % 160) as u16
                + if (x / 16 + y / 16) % 2 == 0 { 20 } else { 0 };
            src.luma.samples[y * src.luma.stride + x] = v;
        }
    }
    for y in 0..h / 2 {
        for x in 0..w / 2 {
            src.cb.samples[y * src.cb.stride + x] = (96 + (x % 64)) as u16;
            src.cr.samples[y * src.cr.stride + x] = (160 - (y % 64)) as u16;
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
    let mut yuv: Vec<u8> =
        Vec::with_capacity(dec.luma.samples.len() + dec.cb.samples.len() + dec.cr.samples.len());
    for plane in [&dec.luma, &dec.cb, &dec.cr] {
        yuv.extend(plane.samples.iter().map(|&v| v as u8));
    }
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

/// r418 — deep-QP deblocking stress: QP 51 / 57 / 63 on the
/// structured source. QP ≥ 45 puts the §8.8.3 luma thresholds in the
/// odd-tC band (tC′ 80 / 157 / 395 → tC 20 / 39 / 99 at 8-bit) where
/// the eqs. 1385/1387 −(tC >> 1) weak-filter clip and the long-filter
/// clip products exercise rounding corners the qp ≤ 45 sweep misses.
#[test]
fn whole_stream_deep_qp_sweep() {
    let src = structured_source(128, 128);
    for qp in [51, 57, 63] {
        let (bs, rec) = encode_idr_with_residuals(&src, qp).unwrap();
        let dec = decode_whole_stream(&bs);
        assert_byte_exact(&dec, &rec, &format!("deep qp {qp}"));
        dump_corpus(&format!("qp{qp}"), &bs, &dec);
    }
}

/// r418 — MTT shapes at high QP: the BT and BT+TT pickers at QP 45
/// combine asymmetric block pairings (16-vs-64 style edges → the
/// §8.8.3.6.8 eqs. 1391 – 1394 asymmetric long filters) with the odd-tC
/// threshold band. The r415 mtt streams ran at QP 30 only.
#[test]
fn whole_stream_mtt_high_qp() {
    let src = structured_source(128, 128);
    let mut cfg = EncoderConfig::new(128, 128);
    cfg.enable_mtt_bt_picker = true;
    let (bs, rec) = encode_idr_with_residuals_cfg(&src, 45, cfg).unwrap();
    let dec = decode_whole_stream(&bs);
    assert_byte_exact(&dec, &rec, "MTT BT qp45");
    dump_corpus("mtt_bt_qp45", &bs, &dec);

    let src = structured_source(128, 128);
    let mut cfg = EncoderConfig::new(128, 128);
    cfg.enable_mtt_bt_picker = true;
    cfg.enable_mtt_tt_picker = true;
    let (bs, rec) = encode_idr_with_residuals_cfg(&src, 45, cfg).unwrap();
    let dec = decode_whole_stream(&bs);
    assert_byte_exact(&dec, &rec, "MTT BT+TT qp45");
    dump_corpus("mtt_bt_tt_qp45", &bs, &dec);
}

/// r418 — multi-CTU at high QP with MTT: a 256x256 picture has an
/// interior CTB row at y = 128, so ≥32-tall CUs above it hit the
/// §8.8.3.6.2 step-6 luma CTB-row rule (sidePisLargeBlk = 0 →
/// eq. 1294 asymmetric (3, 7) long filter) — a path the single-CTU
/// corpus never reaches externally. QP 45 keeps the filters active.
#[test]
fn whole_stream_multi_ctu_mtt_qp45() {
    let src = structured_source(256, 256);
    let mut cfg = EncoderConfig::new(256, 256);
    cfg.enable_mtt_bt_picker = true;
    cfg.enable_mtt_tt_picker = true;
    let (bs, rec) = encode_idr_with_residuals_cfg(&src, 45, cfg).unwrap();
    let dec = decode_whole_stream(&bs);
    assert_byte_exact(&dec, &rec, "256x256 MTT qp45");
    dump_corpus("multi_ctu_mtt_qp45", &bs, &dec);
}

/// r418 — plain multi-CTU at QP 45: CTB-row deblocking (luma step-6 +
/// the chroma CTB-row (1, 3) cap) with 64x64 CUs, no MTT.
#[test]
fn whole_stream_multi_ctu_qp45() {
    let src = structured_source(256, 256);
    let (bs, rec) = encode_idr_with_residuals(&src, 45).unwrap();
    let dec = decode_whole_stream(&bs);
    assert_byte_exact(&dec, &rec, "256x256 qp45");
    dump_corpus("multi_ctu_qp45", &bs, &dec);
}

/// r418 — non-square multi-CTU layout with a partial-width CTU column
/// (192 = 128 + 64): the right CTU column is 64 wide, exercising the
/// boundary implicit splits + edge-clipped deblock/SAO/ALF at high QP.
#[test]
fn whole_stream_192x128_qp45() {
    let src = structured_source(192, 128);
    let (bs, rec) = encode_idr_with_residuals(&src, 45).unwrap();
    let dec = decode_whole_stream(&bs);
    assert_byte_exact(&dec, &rec, "192x128 qp45");
    dump_corpus("wide_192x128_qp45", &bs, &dec);
}

/// r429 — tiles: the picture splits into a §6.5.1 tile grid, the PPS
/// carries the partition block, the slice data walks tile-scan order
/// with per-tile CABAC subsets (end_of_tile_one_bit + byte alignment +
/// §9.3.2.2 context re-init), the SH carries the §7.4.8 entry-point
/// offsets, and the §6.4.4 "different tile" arm gates both sides'
/// prediction availability.
#[test]
fn whole_stream_tiles_2x1() {
    let src = structured_source(256, 128);
    let mut cfg = EncoderConfig::new(256, 128);
    cfg.tile_columns = 2;
    let (bs, rec) = encode_idr_with_residuals_cfg(&src, 26, cfg).unwrap();
    let dec = decode_whole_stream(&bs);
    assert_byte_exact(&dec, &rec, "tiles 2x1");
    dump_corpus("tiles_2x1_256x128", &bs, &dec);
}

/// r429 — 2x2 tile grid on a 256x256 picture (one CTB per tile): four
/// subsets, three entry points, every interior tile boundary active in
/// both directions.
#[test]
fn whole_stream_tiles_2x2() {
    let src = structured_source(256, 256);
    let mut cfg = EncoderConfig::new(256, 256);
    cfg.tile_columns = 2;
    cfg.tile_rows = 2;
    let (bs, rec) = encode_idr_with_residuals_cfg(&src, 26, cfg).unwrap();
    let dec = decode_whole_stream(&bs);
    assert_byte_exact(&dec, &rec, "tiles 2x2");
    dump_corpus("tiles_2x2_256x256", &bs, &dec);
}

/// r429 — 3x1 tile columns at QP 34 on 384x128: three vertical tiles,
/// two entry points, deblocking across the tile boundaries
/// (pps_loop_filter_across_tiles_enabled_flag = 1).
#[test]
fn whole_stream_tiles_3x1_qp34() {
    let src = structured_source(384, 128);
    let mut cfg = EncoderConfig::new(384, 128);
    cfg.tile_columns = 3;
    let (bs, rec) = encode_idr_with_residuals_cfg(&src, 34, cfg).unwrap();
    let dec = decode_whole_stream(&bs);
    assert_byte_exact(&dec, &rec, "tiles 3x1 qp34");
    dump_corpus("tiles_3x1_384x128_qp34", &bs, &dec);
}

/// r429 — tiles with a partial-width right CTB column (192 = 128 +
/// 64): tile 1 is 64 samples wide, so the tile boundary and the
/// picture-boundary implicit-split walk interact.
#[test]
fn whole_stream_tiles_2x1_partial_column() {
    let src = structured_source(192, 128);
    let mut cfg = EncoderConfig::new(192, 128);
    cfg.tile_columns = 2;
    let (bs, rec) = encode_idr_with_residuals_cfg(&src, 30, cfg).unwrap();
    let dec = decode_whole_stream(&bs);
    assert_byte_exact(&dec, &rec, "tiles 2x1 partial column");
    dump_corpus("tiles_2x1_192x128", &bs, &dec);
}

/// r429 — tiles + the MTT BT/TT pickers: mixed-size CUs against the
/// tile-gated split-flag ctxIncs and prediction availability.
#[test]
fn whole_stream_tiles_2x2_mtt() {
    let src = structured_source(256, 256);
    let mut cfg = EncoderConfig::new(256, 256);
    cfg.tile_columns = 2;
    cfg.tile_rows = 2;
    cfg.enable_mtt_bt_picker = true;
    cfg.enable_mtt_tt_picker = true;
    let (bs, rec) = encode_idr_with_residuals_cfg(&src, 30, cfg).unwrap();
    let dec = decode_whole_stream(&bs);
    assert_byte_exact(&dec, &rec, "tiles 2x2 + MTT");
    dump_corpus("tiles_2x2_mtt_qp30", &bs, &dec);
}

/// r429 — the RASTER-scan slice layout (`pps_rect_slice_flag = 0`):
/// the slice header carries `sh_slice_address` (tile 0) +
/// `sh_num_tiles_in_slice_minus1`, and the decoder resolves the CTB
/// plan through the §6.5.1 raster tile-run arm instead of
/// `CtbAddrInSlice`.
#[test]
fn whole_stream_tiles_2x2_raster_layout() {
    let src = structured_source(256, 256);
    let mut cfg = EncoderConfig::new(256, 256);
    cfg.tile_columns = 2;
    cfg.tile_rows = 2;
    cfg.raster_slice_layout = true;
    let (bs, rec) = encode_idr_with_residuals_cfg(&src, 26, cfg).unwrap();
    let dec = decode_whole_stream(&bs);
    assert_byte_exact(&dec, &rec, "tiles 2x2 raster layout");
    dump_corpus("tiles_2x2_raster_256x256", &bs, &dec);
}

/// r429 — tiles with `pps_loop_filter_across_tiles_enabled_flag = 0`:
/// deblocking skips the tile-boundary edges (§8.8.3.1), SAO forces
/// edgeIdx = 0 on cross-tile neighbour samples (§8.8.4.2), and ALF
/// pads its classification / filter / CC-ALF fetches at the tile
/// rectangle (§8.8.5.5 / §8.8.5.6) — on both the encoder
/// reconstruction and the decode path.
#[test]
fn whole_stream_tiles_2x2_no_cross_tile_filters() {
    let src = structured_source(256, 256);
    let mut cfg = EncoderConfig::new(256, 256);
    cfg.tile_columns = 2;
    cfg.tile_rows = 2;
    cfg.loop_filter_across_tiles = false;
    let (bs, rec) = encode_idr_with_residuals_cfg(&src, 26, cfg).unwrap();
    let dec = decode_whole_stream(&bs);
    assert_byte_exact(&dec, &rec, "tiles 2x2 across-tiles off");
    dump_corpus("tiles_2x2_noxlf_256x256", &bs, &dec);
}

/// r429 — across-tiles-off at deep QP with three tile columns: strong
/// deblocking everywhere EXCEPT the two tile boundaries, ALF padding
/// live on every interior boundary.
#[test]
fn whole_stream_tiles_3x1_no_cross_tile_filters_qp45() {
    let src = structured_source(384, 128);
    let mut cfg = EncoderConfig::new(384, 128);
    cfg.tile_columns = 3;
    cfg.loop_filter_across_tiles = false;
    let (bs, rec) = encode_idr_with_residuals_cfg(&src, 45, cfg).unwrap();
    let dec = decode_whole_stream(&bs);
    assert_byte_exact(&dec, &rec, "tiles 3x1 across-tiles off qp45");
    dump_corpus("tiles_3x1_noxlf_qp45", &bs, &dec);
}

/// r429 — WPP (sps_entropy_coding_sync_enabled_flag): every CTU row is
/// a byte-aligned subset (end_of_subset_one_bit), the §9.3.2.3/.4
/// context storage/synchronization runs around each row's first CTU,
/// and the §6.4.4 WPP column arm caps prediction availability.
#[test]
fn whole_stream_wpp_256x256() {
    let src = structured_source(256, 256);
    let mut cfg = EncoderConfig::new(256, 256);
    cfg.wpp = true;
    let (bs, rec) = encode_idr_with_residuals_cfg(&src, 26, cfg).unwrap();
    let dec = decode_whole_stream(&bs);
    assert_byte_exact(&dec, &rec, "wpp 256x256");
    dump_corpus("wpp_256x256", &bs, &dec);
}

/// r429 — WPP over three CTU rows at QP 34 (two sync generations: row
/// 2 restores state stored after row 1's first CTU, which itself ran
/// on state restored from row 0).
#[test]
fn whole_stream_wpp_three_rows() {
    let src = structured_source(128, 384);
    let mut cfg = EncoderConfig::new(128, 384);
    cfg.wpp = true;
    let (bs, rec) = encode_idr_with_residuals_cfg(&src, 34, cfg).unwrap();
    let dec = decode_whole_stream(&bs);
    assert_byte_exact(&dec, &rec, "wpp 128x384");
    dump_corpus("wpp_128x384_qp34", &bs, &dec);
}

/// r429 — everything combined: tiles + WPP + across-tiles filters
/// off at QP 34 (end_of_subset / end_of_tile interleave, WPP context
/// storage/sync across per-tile re-inits, §6.4.4 tile rect + column
/// cap, and the §8.8 boundary gates all in one slice).
#[test]
fn whole_stream_tiles_wpp_noxlf_combined() {
    let src = structured_source(256, 256);
    let mut cfg = EncoderConfig::new(256, 256);
    cfg.tile_columns = 2;
    cfg.wpp = true;
    cfg.loop_filter_across_tiles = false;
    let (bs, rec) = encode_idr_with_residuals_cfg(&src, 34, cfg).unwrap();
    let dec = decode_whole_stream(&bs);
    assert_byte_exact(&dec, &rec, "tiles + wpp + across-tiles off");
    dump_corpus("tiles_2x1_wpp_noxlf_qp34", &bs, &dec);
}

/// r429 — tiles + WPP combined (§7.3.11.1: end_of_subset fires at CTU
/// row ends inside a tile, end_of_tile at tile ends; the WPP storage /
/// sync and the per-tile context re-init interleave).
#[test]
fn whole_stream_tiles_2x1_wpp() {
    let src = structured_source(256, 256);
    let mut cfg = EncoderConfig::new(256, 256);
    cfg.tile_columns = 2;
    cfg.wpp = true;
    let (bs, rec) = encode_idr_with_residuals_cfg(&src, 26, cfg).unwrap();
    let dec = decode_whole_stream(&bs);
    assert_byte_exact(&dec, &rec, "tiles 2x1 + wpp");
    dump_corpus("tiles_2x1_wpp_256x256", &bs, &dec);
}

/// r431 — screen-content source: flat colour cells aligned to the
/// 4:2:0 chroma grid. `cell` is the square cell size in luma samples
/// (even, ≥ 2); `ncolors` bounds the per-64x64-region colour count so
/// the palette gate (≤ 31 distinct triples per CU) is controllable.
fn screen_source(w: usize, h: usize, cell: usize, ncolors: usize) -> PictureBuffer {
    let mut src = PictureBuffer::yuv420_filled(w, h, 128);
    // Deterministic colour list — distinct (Y, Cb, Cr) triples.
    let color = |i: usize| -> (u16, u16, u16) {
        (
            (17 + (i * 41) % 224) as u16,
            (32 + (i * 29) % 192) as u16,
            (24 + (i * 53) % 200) as u16,
        )
    };
    for cy in 0..h.div_ceil(cell) {
        for cx in 0..w.div_ceil(cell) {
            // Cell colour keyed inside the containing 64x64 region so
            // every CU sees at most `ncolors` distinct triples.
            let (rx, ry) = (cx * cell / 64, cy * cell / 64);
            let key = (cx * 7 + cy * 13 + rx + ry * 3) % ncolors;
            let (yv, cbv, crv) = color(key);
            for dy in 0..cell.min(h - cy * cell) {
                for dx in 0..cell.min(w - cx * cell) {
                    let (px, py) = (cx * cell + dx, cy * cell + dy);
                    src.luma.samples[py * src.luma.stride + px] = yv;
                    if px % 2 == 0 && py % 2 == 0 {
                        src.cb.samples[(py / 2) * src.cb.stride + px / 2] = cbv;
                        src.cr.samples[(py / 2) * src.cr.stride + px / 2] = crv;
                    }
                }
            }
        }
    }
    src
}

/// r431 — pure screen content: every CU palette-codes losslessly (≤ 16
/// colours per 64x64 CU), exercising predictor reuse across the four
/// CUs of each 128 CTB and across CTBs.
#[test]
fn whole_stream_palette_screen() {
    let src = screen_source(128, 128, 8, 16);
    let mut cfg = EncoderConfig::new(128, 128);
    cfg.palette = true;
    let (bs, rec) = encode_idr_with_residuals_cfg(&src, 26, cfg).unwrap();
    let dec = decode_whole_stream(&bs);
    assert_byte_exact(&dec, &rec, "palette screen");
    dump_corpus("palette_screen_128x128", &bs, &dec);
}

/// r431 — mixed content: the left half is flat colour cells (palette
/// CUs), the right half is the corpus gradient (transform CUs carrying
/// `pred_mode_plt_flag = 0`); the two CU kinds interleave inside the
/// picture and the predictor palette must survive the transform CUs
/// unchanged.
#[test]
fn whole_stream_palette_mixed_content() {
    let grad = structured_source(256, 128);
    let cells = screen_source(256, 128, 4, 12);
    let mut src = grad;
    for y in 0..128 {
        for x in 0..128 {
            src.luma.samples[y * src.luma.stride + x] =
                cells.luma.samples[y * cells.luma.stride + x];
        }
    }
    for y in 0..64 {
        for x in 0..64 {
            src.cb.samples[y * src.cb.stride + x] = cells.cb.samples[y * cells.cb.stride + x];
            src.cr.samples[y * src.cr.stride + x] = cells.cr.samples[y * cells.cr.stride + x];
        }
    }
    let mut cfg = EncoderConfig::new(256, 128);
    cfg.palette = true;
    let (bs, rec) = encode_idr_with_residuals_cfg(&src, 26, cfg).unwrap();
    let dec = decode_whole_stream(&bs);
    assert_byte_exact(&dec, &rec, "palette mixed");
    dump_corpus("palette_mixed_256x128", &bs, &dec);
}

/// r431 — escape samples: 34 distinct colours per 64x64 CU exceed
/// `maxNumPaletteEntries` (31), so the three least frequent become
/// EG5-coded escapes quantized at the CU QP (eq. 442 dequant on both
/// sides).
#[test]
fn whole_stream_palette_escape() {
    let src = screen_source(64, 64, 4, 34);
    let mut cfg = EncoderConfig::new(64, 64);
    cfg.palette = true;
    let (bs, rec) = encode_idr_with_residuals_cfg(&src, 30, cfg).unwrap();
    let dec = decode_whole_stream(&bs);
    assert_byte_exact(&dec, &rec, "palette escape");
    dump_corpus("palette_escape_64x64_qp30", &bs, &dec);
}

/// r431 — palette + 2x2 tile grid: the predictor palette resets at
/// every tile start (§9.3.2.1) on both sides, and each tile's first
/// palette CU re-signals its entries from scratch.
#[test]
fn whole_stream_palette_tiles_2x2() {
    let src = screen_source(256, 256, 8, 20);
    let mut cfg = EncoderConfig::new(256, 256);
    cfg.palette = true;
    cfg.tile_columns = 2;
    cfg.tile_rows = 2;
    let (bs, rec) = encode_idr_with_residuals_cfg(&src, 26, cfg).unwrap();
    let dec = decode_whole_stream(&bs);
    assert_byte_exact(&dec, &rec, "palette tiles 2x2");
    dump_corpus("palette_tiles_2x2_256x256", &bs, &dec);
}

/// r431 — palette + WPP: the predictor palette stores after each CTU
/// row's first CTU and synchronizes at the next row start (§9.3.2.6 /
/// §9.3.2.7), alongside the context tables.
#[test]
fn whole_stream_palette_wpp() {
    let src = screen_source(256, 256, 8, 20);
    let mut cfg = EncoderConfig::new(256, 256);
    cfg.palette = true;
    cfg.wpp = true;
    let (bs, rec) = encode_idr_with_residuals_cfg(&src, 26, cfg).unwrap();
    let dec = decode_whole_stream(&bs);
    assert_byte_exact(&dec, &rec, "palette wpp");
    dump_corpus("palette_wpp_256x256", &bs, &dec);
}

/// r431 — rectangular multi-slice: one slice per tile on a 2x2 grid.
/// Four VCL NALs, per-slice §9.3.2 CABAC initialisation +
/// `sh_slice_address`, HMVP / predictor resets at each slice start,
/// and the §8.8 filters crossing the slice boundaries
/// (`pps_loop_filter_across_slices_enabled_flag = 1`).
#[test]
fn whole_stream_slices_rect_per_tile_2x2() {
    let src = structured_source(256, 256);
    let mut cfg = EncoderConfig::new(256, 256);
    cfg.tile_columns = 2;
    cfg.tile_rows = 2;
    cfg.slice_per_tile = true;
    let (bs, rec) = encode_idr_with_residuals_cfg(&src, 26, cfg).unwrap();
    let dec = decode_whole_stream(&bs);
    assert_byte_exact(&dec, &rec, "slices rect per tile 2x2");
    dump_corpus("slices_rect_2x2_256x256", &bs, &dec);
}

/// r431 — rectangular multi-slice at QP 34: stronger deblocking
/// activity across the slice boundaries while each slice runs its own
/// CABAC state.
#[test]
fn whole_stream_slices_rect_per_tile_3x1_qp34() {
    let src = structured_source(384, 128);
    let mut cfg = EncoderConfig::new(384, 128);
    cfg.tile_columns = 3;
    cfg.slice_per_tile = true;
    let (bs, rec) = encode_idr_with_residuals_cfg(&src, 34, cfg).unwrap();
    let dec = decode_whole_stream(&bs);
    assert_byte_exact(&dec, &rec, "slices rect 3x1 qp34");
    dump_corpus("slices_rect_3x1_384x128_qp34", &bs, &dec);
}

/// r431 — raster-scan multi-slice: a 2x2 tile grid split into two
/// slices of two tiles each (`pps_rect_slice_flag = 0`,
/// `sh_slice_address` = first tile index + tile run length).
#[test]
fn whole_stream_slices_raster_2_of_2x2() {
    let src = structured_source(256, 256);
    let mut cfg = EncoderConfig::new(256, 256);
    cfg.tile_columns = 2;
    cfg.tile_rows = 2;
    cfg.raster_slice_count = 2;
    let (bs, rec) = encode_idr_with_residuals_cfg(&src, 26, cfg).unwrap();
    let dec = decode_whole_stream(&bs);
    assert_byte_exact(&dec, &rec, "slices raster 2 of 2x2");
    dump_corpus("slices_raster_2_256x256", &bs, &dec);
}

/// r431 — `pps_loop_filter_across_slices_enabled_flag = 0` on the
/// rectangular per-tile layout while the across-TILES flag stays 1:
/// the filters must gate on the slice map alone (§8.8.3.1 edge
/// exclusion, §8.8.4.2 edgeIdx = 0, §8.8.5.5/§8.8.5.6 padding).
#[test]
fn whole_stream_slices_rect_noxlf_2x2() {
    let src = structured_source(256, 256);
    let mut cfg = EncoderConfig::new(256, 256);
    cfg.tile_columns = 2;
    cfg.tile_rows = 2;
    cfg.slice_per_tile = true;
    cfg.loop_filter_across_slices = false;
    let (bs, rec) = encode_idr_with_residuals_cfg(&src, 26, cfg).unwrap();
    let dec = decode_whole_stream(&bs);
    assert_byte_exact(&dec, &rec, "slices rect noxlf 2x2");
    dump_corpus("slices_rect_noxlf_2x2_256x256", &bs, &dec);
}

/// r431 — raster multi-slice with closed slice boundaries at deep QP:
/// three tile rows, three slices, the long deblocking filters active
/// everywhere except the two slice-row boundaries.
#[test]
fn whole_stream_slices_raster_noxlf_3rows_qp45() {
    let src = structured_source(128, 384);
    let mut cfg = EncoderConfig::new(128, 384);
    cfg.tile_rows = 3;
    cfg.raster_slice_count = 3;
    cfg.loop_filter_across_slices = false;
    let (bs, rec) = encode_idr_with_residuals_cfg(&src, 45, cfg).unwrap();
    let dec = decode_whole_stream(&bs);
    assert_byte_exact(&dec, &rec, "slices raster noxlf 3 rows qp45");
    dump_corpus("slices_raster_noxlf_3rows_128x384_qp45", &bs, &dec);
}

/// r431 — multi-slice + WPP: each slice-tile codes its CTU rows as
/// byte-aligned subsets with the §9.3.2.3/§9.3.2.4 storage/sync, and
/// the per-slice `sh_entry_point_offset` lists stay slice-local.
#[test]
fn whole_stream_slices_rect_wpp() {
    let src = structured_source(128, 384);
    let mut cfg = EncoderConfig::new(128, 384);
    cfg.tile_rows = 3;
    cfg.slice_per_tile = true;
    cfg.wpp = true;
    let (bs, rec) = encode_idr_with_residuals_cfg(&src, 26, cfg).unwrap();
    let dec = decode_whole_stream(&bs);
    assert_byte_exact(&dec, &rec, "slices rect wpp");
    dump_corpus("slices_rect_wpp_128x384", &bs, &dec);
}

/// r431 — palette + rectangular multi-slice: the predictor palette
/// resets at every slice start on both sides (§9.3.2.1 rides the
/// slice-initialisation arm, not just the tile arm).
#[test]
fn whole_stream_palette_slices_rect_2x2() {
    let src = screen_source(256, 256, 8, 20);
    let mut cfg = EncoderConfig::new(256, 256);
    cfg.palette = true;
    cfg.tile_columns = 2;
    cfg.tile_rows = 2;
    cfg.slice_per_tile = true;
    let (bs, rec) = encode_idr_with_residuals_cfg(&src, 26, cfg).unwrap();
    let dec = decode_whole_stream(&bs);
    assert_byte_exact(&dec, &rec, "palette slices rect 2x2");
    dump_corpus("palette_slices_rect_2x2_256x256", &bs, &dec);
}
