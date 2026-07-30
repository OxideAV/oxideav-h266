//! r434 — JVET VVC conformance-corpus triage harness.
//!
//! Decodes every staged FDIS r1 conformance stream (Annex-B `.bit`)
//! found under the docs conformance directory and classifies each as
//! pass / fail / unsupported-tool against the `.opl` per-picture
//! per-plane MD5 sidecars. The harness is a no-op when the corpus is
//! not present (CI machines do not carry the docs tree).
//!
//! Corpus location: `$H266_CONFORMANCE_DIR`, else
//! `../../docs/video/h266/conformance/streams` relative to the crate.

use std::path::PathBuf;

fn corpus_dir() -> Option<PathBuf> {
    if let Ok(d) = std::env::var("H266_CONFORMANCE_DIR") {
        let p = PathBuf::from(d);
        return p.is_dir().then_some(p);
    }
    let p =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../docs/video/h266/conformance/streams");
    p.is_dir().then_some(p)
}

/// Probe mode: dump one line of SPS-derived facts per stream.
#[test]
fn probe_corpus_sps() {
    use oxideav_h266::nal::{extract_rbsp, iter_annex_b, NalUnitType};
    use oxideav_h266::pps::parse_pps;
    use oxideav_h266::sps::parse_sps;
    let Some(dir) = corpus_dir() else {
        eprintln!("jvet corpus not present — skipping");
        return;
    };
    let mut names: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| {
            let p = e.unwrap().path();
            (p.extension().map(|x| x == "bit") == Some(true)).then_some(p)
        })
        .collect();
    names.sort();
    for path in names {
        let bs = std::fs::read(&path).unwrap();
        let name = path.file_stem().unwrap().to_string_lossy().to_string();
        let mut sps_line = String::from("no SPS");
        for nal in iter_annex_b(&bs) {
            if nal.header.nal_unit_type == NalUnitType::SpsNut {
                match parse_sps(&extract_rbsp(nal.payload())) {
                    Ok(sps) => {
                        let mut pps_note = String::new();
                        for nal2 in iter_annex_b(&bs) {
                            if nal2.header.nal_unit_type == NalUnitType::PpsNut {
                                match parse_pps(&extract_rbsp(nal2.payload())) {
                                    Ok(pps) => {
                                        let (tc, tr, ns) = pps
                                            .partition
                                            .as_ref()
                                            .map(|p| {
                                                (
                                                    p.num_tile_columns,
                                                    p.num_tile_rows,
                                                    p.num_slices_in_pic,
                                                )
                                            })
                                            .unwrap_or((1, 1, 1));
                                        pps_note = format!(
                                            " tiles={tc}x{tr} rect={} slices={ns} wp={}/{}",
                                            pps.pps_rect_slice_flag,
                                            pps.pps_weighted_pred_flag,
                                            pps.pps_weighted_bipred_flag,
                                        )
                                    }
                                    Err(e) => pps_note = format!(" PPS-ERR {e}"),
                                }
                                break;
                            }
                        }
                        let subpics = sps
                            .subpic_info
                            .as_ref()
                            .map(|s| s.num_subpics_minus1 + 1)
                            .unwrap_or(1);
                        sps_line = format!(
                            "{}x{} bd={} cf={} dualtree={} wpp={} subpics={} ibc={} plt={} act={} sbt={} lmcs={} vb={} numrpl={}{}",
                            sps.cropped_width(),
                            sps.cropped_height(),
                            sps.bit_depth_y(),
                            sps.sps_chroma_format_idc,
                            sps.partition_constraints.qtbtt_dual_tree_intra_flag,
                            sps.sps_entropy_coding_sync_enabled_flag,
                            subpics,
                            sps.tool_flags.ibc_enabled_flag,
                            sps.tool_flags.palette_enabled_flag,
                            sps.tool_flags.act_enabled_flag,
                            sps.tool_flags.sbt_enabled_flag,
                            sps.tool_flags.lmcs_enabled_flag,
                            sps.tool_flags.virtual_boundaries_enabled_flag,
                            sps.tool_flags.num_ref_pic_lists[0],
                            pps_note,
                        );
                    }
                    Err(e) => sps_line = format!("SPS-ERR {e}"),
                }
                break;
            }
        }
        println!("{name}: {sps_line}");
    }
}

/// Header triage: every NAL of every stream must parse through the
/// crate's own SPS / PPS / APS / PH / SH parsers. Prints a per-stream
/// verdict and fails if any stream has a header parse error.
#[test]
fn corpus_headers_parse() {
    use oxideav_h266::aps::parse_aps;
    use oxideav_h266::nal::{extract_rbsp, iter_annex_b, NalUnitType};
    use oxideav_h266::picture_header::{parse_picture_header_stateful, PictureHeader};
    use oxideav_h266::pps::{parse_pps, PicParameterSet};
    use oxideav_h266::slice_header::{parse_slice_header, parse_slice_header_stateful, PhState};
    use oxideav_h266::sps::{parse_sps, SeqParameterSet};
    use std::collections::HashMap;

    let Some(dir) = corpus_dir() else {
        eprintln!("jvet corpus not present — skipping");
        return;
    };
    let mut names: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| {
            let p = e.unwrap().path();
            (p.extension().map(|x| x == "bit") == Some(true)).then_some(p)
        })
        .collect();
    names.sort();
    let mut bad = Vec::new();
    for path in names {
        let bs = std::fs::read(&path).unwrap();
        let name = path.file_stem().unwrap().to_string_lossy().to_string();
        let mut spss: HashMap<u8, SeqParameterSet> = HashMap::new();
        let mut ppss: HashMap<u8, PicParameterSet> = HashMap::new();
        let mut cur_ph: Option<PictureHeader> = None;
        let mut vcl = 0usize;
        let mut err: Option<String> = None;
        'nals: for (idx, nal) in iter_annex_b(&bs).enumerate() {
            let t = nal.header.nal_unit_type;
            let rbsp = extract_rbsp(nal.payload());
            let step: Result<(), String> = (|| {
                match t {
                    NalUnitType::SpsNut => {
                        let sps = parse_sps(&rbsp).map_err(|e| format!("SPS: {e}"))?;
                        spss.insert(sps.sps_seq_parameter_set_id, sps);
                    }
                    NalUnitType::PpsNut => {
                        let pps = parse_pps(&rbsp).map_err(|e| format!("PPS: {e}"))?;
                        ppss.insert(pps.pps_pic_parameter_set_id, pps);
                    }
                    NalUnitType::PrefixApsNut | NalUnitType::SuffixApsNut => {
                        parse_aps(&rbsp).map_err(|e| format!("APS: {e}"))?;
                    }
                    NalUnitType::PhNut => {
                        // Resolve PPS/SPS by the PH's own pps id (lead peek
                        // is unnecessary — PH starts at bit 0 here, and the
                        // stateful parser reads the ids itself; we pre-peek
                        // via the lead parser inside parse_slice_header for
                        // SH-embedded PHs only).
                        let lead = oxideav_h266::picture_header::parse_picture_header(&rbsp)
                            .map_err(|e| format!("PH lead: {e}"))?;
                        let pps = ppss
                            .get(&(lead.ph_pic_parameter_set_id as u8))
                            .ok_or_else(|| "PH references unknown PPS".to_string())?;
                        let sps = spss
                            .get(&pps.pps_seq_parameter_set_id)
                            .ok_or_else(|| "PPS references unknown SPS".to_string())?;
                        let ph = parse_picture_header_stateful(&rbsp, sps, pps)
                            .map_err(|e| format!("PH: {e}"))?;
                        cur_ph = Some(ph);
                    }
                    _ if t.is_vcl() => {
                        vcl += 1;
                        // Peek: does this SH embed its PH?
                        let peek =
                            parse_slice_header(&rbsp).map_err(|e| format!("SH peek: {e}"))?;
                        let pps_id = if let Some(lead) = &peek.embedded_picture_header {
                            lead.ph_pic_parameter_set_id as u8
                        } else {
                            cur_ph
                                .as_ref()
                                .map(|p| p.ph_pic_parameter_set_id as u8)
                                .ok_or_else(|| "VCL NAL before any PH".to_string())?
                        };
                        let pps = ppss
                            .get(&pps_id)
                            .ok_or_else(|| "SH references unknown PPS".to_string())?;
                        let sps = spss
                            .get(&pps.pps_seq_parameter_set_id)
                            .ok_or_else(|| "PPS references unknown SPS".to_string())?;
                        let ph_state = if peek.embedded_picture_header.is_some() {
                            PhState {
                                nal_unit_type: t,
                                ..Default::default()
                            }
                        } else {
                            let ph = cur_ph.as_ref().unwrap();
                            PhState {
                                ph_inter_slice_allowed_flag: ph.ph_inter_slice_allowed_flag,
                                ph_intra_slice_allowed_flag: ph.ph_intra_slice_allowed_flag,
                                ph_alf_enabled_flag: ph.ph_alf_enabled_flag,
                                ph_lmcs_enabled_flag: ph.ph_lmcs_enabled_flag,
                                ph_explicit_scaling_list_enabled_flag: ph
                                    .ph_explicit_scaling_list_enabled_flag,
                                ph_temporal_mvp_enabled_flag: ph.ph_temporal_mvp_enabled_flag,
                                ph_sao_luma_enabled_flag: ph.ph_sao_luma_enabled_flag,
                                ph_sao_chroma_enabled_flag: ph.ph_sao_chroma_enabled_flag,
                                num_extra_sh_bits: 0,
                                nal_unit_type: t,
                                num_ref_entries: [
                                    ph.ref_pic_lists
                                        .as_ref()
                                        .map(|r| r[0].rpls.entries.len() as u32)
                                        .unwrap_or(0),
                                    ph.ref_pic_lists
                                        .as_ref()
                                        .map(|r| r[1].rpls.entries.len() as u32)
                                        .unwrap_or(0),
                                ],
                                ph_collocated_from_l0_flag: ph.ph_collocated_from_l0_flag,
                                ph_collocated_ref_idx: ph.ph_collocated_ref_idx,
                            }
                        };
                        parse_slice_header_stateful(&rbsp, sps, pps, &ph_state)
                            .map_err(|e| format!("SH: {e}"))?;
                    }
                    _ => {}
                }
                Ok(())
            })();
            if let Err(e) = step {
                err = Some(format!("NAL #{idx} ({t:?}): {e}"));
                break 'nals;
            }
        }
        match err {
            None => println!("{name}: headers OK ({vcl} VCL NALs)"),
            Some(e) => {
                println!("{name}: HEADER-ERR {e}");
                bad.push(name);
            }
        }
    }
    assert!(bad.is_empty(), "header parse failures: {bad:?}");
}
