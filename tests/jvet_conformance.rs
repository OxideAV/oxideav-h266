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

// ---------------------------------------------------------------------
// Decode triage — StreamDecoder vs the .opl per-picture plane MD5s.
// ---------------------------------------------------------------------

/// Minimal RFC-1321-shape MD5 (the conformance sidecars are MD5-based;
/// no new dependency). The 64 round constants are derived at run time
/// as `floor(|sin(i + 1)| * 2^32)`.
mod md5 {
    const S: [u32; 64] = [
        7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, //
        5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, //
        4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, //
        6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
    ];

    fn k_table() -> [u32; 64] {
        let mut k = [0u32; 64];
        for (i, slot) in k.iter_mut().enumerate() {
            *slot = (((i as f64 + 1.0).sin().abs()) * 4294967296.0) as u32;
        }
        k
    }

    pub struct Md5 {
        state: [u32; 4],
        buf: Vec<u8>,
        len: u64,
        k: [u32; 64],
    }

    impl Md5 {
        pub fn new() -> Self {
            Self {
                state: [0x6745_2301, 0xefcd_ab89, 0x98ba_dcfe, 0x1032_5476],
                buf: Vec::new(),
                len: 0,
                k: k_table(),
            }
        }

        pub fn update(&mut self, data: &[u8]) {
            self.len = self.len.wrapping_add(data.len() as u64);
            self.buf.extend_from_slice(data);
            let blocks = self.buf.len() / 64;
            for b in 0..blocks {
                let block: [u8; 64] = self.buf[b * 64..b * 64 + 64].try_into().unwrap();
                self.process(&block);
            }
            self.buf.drain(..blocks * 64);
        }

        fn process(&mut self, block: &[u8; 64]) {
            let mut m = [0u32; 16];
            for (i, slot) in m.iter_mut().enumerate() {
                *slot = u32::from_le_bytes(block[i * 4..i * 4 + 4].try_into().unwrap());
            }
            let (mut a, mut b, mut c, mut d) =
                (self.state[0], self.state[1], self.state[2], self.state[3]);
            #[allow(clippy::needless_range_loop)] // i indexes S, K and drives g
            for i in 0..64 {
                let (f, g) = match i / 16 {
                    0 => ((b & c) | (!b & d), i),
                    1 => ((d & b) | (!d & c), (5 * i + 1) % 16),
                    2 => (b ^ c ^ d, (3 * i + 5) % 16),
                    _ => (c ^ (b | !d), (7 * i) % 16),
                };
                let tmp = d;
                d = c;
                c = b;
                let sum = a.wrapping_add(f).wrapping_add(self.k[i]).wrapping_add(m[g]);
                b = b.wrapping_add(sum.rotate_left(S[i]));
                a = tmp;
            }
            self.state[0] = self.state[0].wrapping_add(a);
            self.state[1] = self.state[1].wrapping_add(b);
            self.state[2] = self.state[2].wrapping_add(c);
            self.state[3] = self.state[3].wrapping_add(d);
        }

        pub fn finish_hex(mut self) -> String {
            let bit_len = self.len.wrapping_mul(8);
            self.update(&[0x80]);
            while self.buf.len() % 64 != 56 {
                self.buf.push(0);
            }
            let mut tail = self.buf.split_off(0);
            tail.extend_from_slice(&bit_len.to_le_bytes());
            for b in 0..tail.len() / 64 {
                let block: [u8; 64] = tail[b * 64..b * 64 + 64].try_into().unwrap();
                self.process(&block);
            }
            let mut out = String::with_capacity(32);
            for s in self.state {
                for byte in s.to_le_bytes() {
                    out.push_str(&format!("{byte:02x}"));
                }
            }
            out
        }
    }

    pub fn md5_hex(data: &[u8]) -> String {
        let mut h = Md5::new();
        h.update(data);
        h.finish_hex()
    }

    #[test]
    fn known_vectors() {
        assert_eq!(md5_hex(b""), "d41d8cd98f00b204e9800998ecf8427e");
        assert_eq!(md5_hex(b"abc"), "900150983cd24fb0d6963f7d28e17f72");
        assert_eq!(
            md5_hex(b"The quick brown fox jumps over the lazy dog"),
            "9e107d9d372bb6826bd81d3542a419d6"
        );
        // Multi-block + 56-byte-boundary shapes.
        let long = vec![0xa5u8; 200];
        let mut h = Md5::new();
        h.update(&long[..3]);
        h.update(&long[3..]);
        assert_eq!(h.finish_hex(), md5_hex(&long));
    }
}

/// One `.opl` row.
#[derive(Debug)]
struct OplRow {
    poc: i32,
    hashes: Vec<String>,
}

fn parse_opl(path: &std::path::Path) -> Vec<OplRow> {
    let text = std::fs::read_to_string(path).unwrap();
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let cols: Vec<&str> = l.split(',').map(str::trim).collect();
            OplRow {
                poc: cols[1].parse().unwrap(),
                hashes: cols[4..].iter().map(|s| s.to_lowercase()).collect(),
            }
        })
        .collect()
}

/// Cropped-plane MD5s for one decoded picture. Per the conformance
/// convention, 8-bit streams hash one byte per sample and >8-bit
/// streams hash two bytes per sample, little-endian.
fn picture_hashes(pic: &oxideav_h266::stream::DecodedPicture) -> Vec<String> {
    let (cx, cy, cw, ch) = pic.crop;
    let wide = pic.bit_depth > 8;
    let plane_md5 =
        |p: &oxideav_h266::reconstruct::PicturePlane, x0: usize, y0: usize, w: usize, h: usize| {
            let mut hasher = md5::Md5::new();
            let mut row: Vec<u8> = Vec::with_capacity(w * 2);
            for y in y0..y0 + h {
                row.clear();
                for &v in &p.samples[y * p.stride + x0..y * p.stride + x0 + w] {
                    if wide {
                        row.extend_from_slice(&v.to_le_bytes());
                    } else {
                        row.push(v as u8);
                    }
                }
                hasher.update(&row);
            }
            hasher.finish_hex()
        };
    let mut out = vec![plane_md5(&pic.frame.luma, cx, cy, cw, ch)];
    if pic.chroma_format_idc != 0 {
        out.push(plane_md5(&pic.frame.cb, cx / 2, cy / 2, cw / 2, ch / 2));
        out.push(plane_md5(&pic.frame.cr, cx / 2, cy / 2, cw / 2, ch / 2));
    }
    out
}

#[derive(Debug, PartialEq, Eq, Clone)]
enum Verdict {
    /// Every output picture's every plane hash matches its `.opl` row.
    Pass,
    /// Decoded end-to-end but at least one plane hash (or the output
    /// picture count) differs. Payload: short diagnosis.
    Fail(String),
    /// The stream needs a tool the pipeline does not implement yet.
    Unsupported(String),
    /// Hard decode error (parse failure, missing reference, panic).
    Error(String),
}

fn triage_stream(dir: &std::path::Path, name: &str) -> Verdict {
    use oxideav_h266::stream::StreamDecoder;
    let bs = std::fs::read(dir.join(format!("{name}.bit"))).unwrap();
    let opl = parse_opl(&dir.join(format!("{name}.opl")));

    let outputs: std::sync::Mutex<Vec<(u32, i32, Vec<String>)>> = std::sync::Mutex::new(Vec::new());
    let decode = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut dec = StreamDecoder::new();
        dec.decode_annex_b(&bs, &mut |pic| {
            if pic.output_flag {
                let hashes = picture_hashes(&pic);
                outputs.lock().unwrap().push((pic.cvs_idx, pic.poc, hashes));
            }
        })
    }));
    match decode {
        Err(payload) => {
            let msg = payload
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_else(|| "opaque panic".into());
            return Verdict::Error(format!("panic: {msg}"));
        }
        Ok(Err(e)) => {
            let msg = format!("{e}");
            return if msg.contains("nsupported") {
                Verdict::Unsupported(msg)
            } else {
                Verdict::Error(msg)
            };
        }
        Ok(Ok(())) => {}
    }

    // Output order: ascending POC within each CVS, CVSs in decode
    // order (each conformance stream decodes a CVS completely before
    // the next begins).
    let mut outs = outputs.into_inner().unwrap();
    outs.sort_by_key(|(cvs, poc, _)| (*cvs, *poc));

    if outs.len() != opl.len() {
        return Verdict::Fail(format!(
            "output picture count {} != .opl rows {}",
            outs.len(),
            opl.len()
        ));
    }
    let mut bad_rows = 0usize;
    let mut first_bad: Option<String> = None;
    for ((_, poc, hashes), row) in outs.iter().zip(&opl) {
        let row_ok = *poc == row.poc && hashes == &row.hashes;
        if !row_ok {
            bad_rows += 1;
            if first_bad.is_none() {
                let plane = if *poc != row.poc {
                    format!("poc {} != opl poc {}", poc, row.poc)
                } else {
                    let idx = hashes
                        .iter()
                        .zip(&row.hashes)
                        .position(|(a, b)| a != b)
                        .map(|i| ["Y", "Cb", "Cr"][i.min(2)])
                        .unwrap_or("count");
                    format!("poc {poc} plane {idx}")
                };
                first_bad = Some(plane);
            }
        }
    }
    if bad_rows == 0 {
        Verdict::Pass
    } else {
        Verdict::Fail(format!(
            "{bad_rows}/{} pictures diverge; first at {}",
            opl.len(),
            first_bad.unwrap()
        ))
    }
}

/// Full-corpus decode triage. Prints the scorecard; asserts the
/// verdict of every stream matches the tracked baseline below so any
/// regression (or unrecorded improvement) fails loudly.
///
/// `H266_CONFORMANCE_FILTER` (comma-separated substrings) restricts
/// the run to matching stream names — a development aid; the baseline
/// ratchet only applies to the streams actually run.
#[test]
fn corpus_decode_triage() {
    let Some(dir) = corpus_dir() else {
        eprintln!("jvet corpus not present — skipping");
        return;
    };
    let filter: Option<Vec<String>> = std::env::var("H266_CONFORMANCE_FILTER")
        .ok()
        .map(|f| f.split(',').map(|s| s.trim().to_string()).collect());
    let mut names: Vec<String> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| {
            let p = e.unwrap().path();
            (p.extension().map(|x| x == "bit") == Some(true))
                .then(|| p.file_stem().unwrap().to_string_lossy().to_string())
        })
        .filter(|n| {
            filter
                .as_ref()
                .map(|f| f.iter().any(|s| n.contains(s.as_str())))
                .unwrap_or(true)
        })
        .collect();
    names.sort();
    let mut counts = [0usize; 4];
    let mut lines = Vec::new();
    for name in &names {
        let v = triage_stream(&dir, name);
        let (idx, tag) = match &v {
            Verdict::Pass => (0, "PASS".to_string()),
            Verdict::Fail(m) => (1, format!("FAIL {m}")),
            Verdict::Unsupported(m) => (2, format!("UNSUPPORTED {m}")),
            Verdict::Error(m) => (3, format!("ERROR {m}")),
        };
        counts[idx] += 1;
        println!("{name}: {tag}");
        lines.push((name.clone(), v));
    }
    println!(
        "== scorecard: {} pass / {} fail / {} unsupported / {} error (of {})",
        counts[0],
        counts[1],
        counts[2],
        counts[3],
        names.len()
    );

    // Ratchet against the tracked baseline.
    let expected: std::collections::HashMap<&str, char> =
        EXPECTED_VERDICTS.iter().copied().collect();
    let mut drift = Vec::new();
    for (name, v) in &lines {
        let tag = match v {
            Verdict::Pass => 'P',
            Verdict::Fail(_) => 'F',
            Verdict::Unsupported(_) => 'U',
            Verdict::Error(_) => 'E',
        };
        match expected.get(name.as_str()) {
            Some(&e) if e == tag => {}
            Some(&e) => drift.push(format!("{name}: expected {e}, got {tag} ({v:?})")),
            None => drift.push(format!("{name}: not in EXPECTED_VERDICTS (got {tag})")),
        }
    }
    assert!(
        drift.is_empty(),
        "triage verdicts drifted from the tracked baseline:\n{}",
        drift.join("\n")
    );
}

/// Tracked triage baseline — one row per stream, updated alongside
/// every family fix so both regressions AND unrecorded improvements
/// fail the harness. Classes: P = pass (byte-exact vs .opl),
/// F = fail (decodes, diverges), U = unsupported tool, E = error.
const EXPECTED_VERDICTS: &[(&str, char)] = &[
    // r449: EVERY intra picture in the corpus is byte-exact. Three
    // root causes closed the former "data-dependent intra deblock
    // tail" + the poc-0 chroma family: (1) the §8.8.3.2 – §8.8.3.5
    // deblock driver was rebuilt as the spec's per-CU edge walk —
    // interior ISP / SBT transform-block edges now filter, every
    // maxFilterLength derives from TRANSFORM-BLOCK dims, and edges
    // process in geometric order (the §8.8.3.1 same-direction data
    // dependency); (2) the §8.7.5.3 eq. 1216 invAvgLuma summed
    // native-BitDepth samples through a stale `<< (BitDepth − 8)`
    // lift, wrecking LMCS chroma residual scaling on every >8-bit
    // wire; (3) §7.4.3.18 CC-ALF coefficients are MAPPED
    // (±2^(abs − 1)) — the parse used the raw magnitude. The
    // remaining F rows all first diverge on their first INTER
    // picture (poc 1 plane Y — under triage); the E rows are the
    // post-affine-gate parse desync family; the U rows are
    // subpictures / 4:2:2 / 4:4:4 / explicit WP / per-slice
    // loop-filter divergence.
    ("10b422_B_5", 'U'),          // 4:2:2 / 4:4:4 chroma formats
    ("8b400_A_2", 'P'),           // byte-exact (r450 implicit MaxTbSizeY deblock tiling)
    ("8b420_A_2", 'P'),           // byte-exact (r452 §8.5.2.1 collapsed-list motion clear)
    ("8b420_B_2", 'P'),           // byte-exact (r450 implicit MaxTbSizeY deblock tiling)
    ("8b444_A_2", 'U'),           // 4:2:2 / 4:4:4 chroma formats
    ("AFF_A_2", 'P'),             // byte-exact (r450 implicit MaxTbSizeY deblock tiling)
    ("ALF_A_3", 'P'),             // byte-exact (r449 sbBdofFlag)
    ("ALF_B_3", 'P'),             // byte-exact (r449 sbBdofFlag)
    ("ALF_C_3", 'P'),             // byte-exact (r449)
    ("AMVR_A_3", 'P'),            // byte-exact (r452 §8.5.2.1 collapsed-list motion clear)
    ("BDOF_A_4", 'P'),            // byte-exact (r452 §8.5.2.1 collapsed-list motion clear)
    ("BDPCM_A_2", 'P'),           // byte-exact (r449)
    ("CCLM_A_2", 'P'),            // byte-exact (r449)
    ("CST_A_4", 'P'),             // byte-exact (r449)
    ("CodingToolsSets_A_2", 'P'), // byte-exact
    ("CodingToolsSets_B_2", 'P'), // byte-exact (r450 §8.5.2.3 raw-availability pruning)
    ("CodingToolsSets_C_2", 'P'), // byte-exact (r449)
    ("CodingToolsSets_D_2", 'P'), // byte-exact (r450 §8.5.5.2 affine gate + §8.5.2.3 pruning)
    ("CodingToolsSets_E_1", 'U'), // subpictures
    ("DEBLOCKING_A_3", 'F'),      // 9/17 pictures diverge (first poc 8 plane Cb)
    ("DEBLOCKING_C_3", 'P'),      // byte-exact (r450 implicit MaxTbSizeY deblock tiling)
    ("DEBLOCKING_E_3", 'P'),      // byte-exact (r450 implicit MaxTbSizeY deblock tiling)
    ("DMVR_A_3", 'U'),            // explicit weighted prediction
    ("DMVR_B_4", 'P'),            // byte-exact (r447 §8.5.6.6.2 HP bi-pred composition)
    ("GDR_A_2", 'U'),             // explicit §8.8.1 virtual boundaries
    ("GPM_A_3", 'P'),             // byte-exact (r450 §8.5.2.7 MMVD offset scaling)
    ("IBC_A_2", 'F'),             // 12/17 pictures diverge (first poc 5 plane Cb)
    ("ISP_A_3", 'P'),             // byte-exact (r449)
    ("JCCR_A_2", 'P'),            // byte-exact (r450 CIIP chroma gate + per-TB joint deblock QP)
    ("JCCR_C_3", 'P'),            // byte-exact (r450 §8.5.2.3 raw-availability pruning)
    ("LFNST_A_4", 'P'),           // byte-exact (r449)
    ("LFNST_B_4", 'P'),           // byte-exact (r450 §8.5.2.7 MMVD offset scaling)
    ("LMCS_A_3", 'U'),            // per-slice loop-filter parameter divergence
    ("LMCS_B_2", 'U'),            // subpictures
    ("LMCS_C_1", 'P'),            // byte-exact (r450 implicit MaxTbSizeY deblock tiling)
    ("LOSSLESS_B_3", 'E'),        // IBC BV derivation (under triage)
    ("MERGE_A_2", 'P'),           // byte-exact (r452 §8.5.2.1 collapsed-list motion clear)
    ("MIP_A_3", 'P'),             // byte-exact (r449)
    ("MIP_B_3", 'P'),             // byte-exact (r450 implicit MaxTbSizeY deblock tiling)
    ("MMVD_A_3", 'F'),            // 21/300 pictures diverge (first poc 33 plane Cb)
    ("MTS_A_4", 'P'),             // byte-exact (r449)
    ("PROF_B_3", 'P'),            // byte-exact (r450 implicit MaxTbSizeY deblock tiling)
    ("QUANT_A_2", 'P'),           // byte-exact (r450 chroma DMVR bounding window)
    ("QUANT_B_2", 'P'),           // byte-exact (r450 implicit MaxTbSizeY deblock tiling)
    ("RAP_A_1", 'P'),             // byte-exact (r449)
    ("RAP_B_1", 'P'),             // byte-exact (r450 implicit MaxTbSizeY deblock tiling)
    ("SAO_A_3", 'P'),             // byte-exact (r450 implicit MaxTbSizeY deblock tiling)
    ("SBT_A_2", 'P'),             // byte-exact (r452 §8.5.2.1 collapsed-list motion clear)
    ("SLICES_A_3", 'F'),          // 24/25 pictures diverge (first poc 0 plane Cb)
    ("SMVD_A_2", 'P'),            // byte-exact (r450 implicit MaxTbSizeY deblock tiling)
    ("SUBPIC_C_1", 'U'),          // subpictures
    ("SbTMVP_A_3", 'P'),          // byte-exact (r450 implicit MaxTbSizeY deblock tiling)
    ("TILE_A_2", 'P'),            // byte-exact (r452 §8.5.2.1 collapsed-list motion clear)
    ("TMVP_A_3", 'P'),            // byte-exact (r452 §8.5.2.1 collapsed-list motion clear)
    ("WPP_A_3", 'P'),             // byte-exact (r452 §8.5.2.1 collapsed-list motion clear)
    ("WP_A_3", 'U'),              // explicit weighted prediction
];

/// CI-runnable (no corpus needed): the stream driver must decode this
/// crate's own encoder output byte-exactly through the same glue the
/// conformance triage uses.
#[test]
fn stream_decoder_decodes_own_idr_stream() {
    use oxideav_h266::encoder_pipeline::encode_idr_with_residuals;
    use oxideav_h266::reconstruct::PictureBuffer;
    use oxideav_h266::stream::StreamDecoder;

    let mut src = PictureBuffer::yuv420_filled(128, 128, 128);
    for y in 0..128 {
        for x in 0..128 {
            let v = 40 + ((x * 3 + y * 2) % 160) + if (x / 16 + y / 16) % 2 == 0 { 20 } else { 0 };
            src.luma.samples[y * src.luma.stride + x] = v as u16;
        }
    }
    for y in 0..64 {
        for x in 0..64 {
            src.cb.samples[y * src.cb.stride + x] = (96 + x) as u16;
            src.cr.samples[y * src.cr.stride + x] = (160 - y) as u16;
        }
    }
    let (bs, rec) = encode_idr_with_residuals(&src, 26).unwrap();
    let mut frames = Vec::new();
    StreamDecoder::new()
        .decode_annex_b(&bs, &mut |p| frames.push(p))
        .unwrap();
    assert_eq!(frames.len(), 1, "one coded picture");
    let f = &frames[0];
    assert_eq!(f.poc, 0);
    assert!(f.output_flag);
    assert_eq!(f.crop, (0, 0, 128, 128));
    assert_eq!(f.frame.luma.samples, rec.luma.samples, "luma diverged");
    assert_eq!(f.frame.cb.samples, rec.cb.samples, "cb diverged");
    assert_eq!(f.frame.cr.samples, rec.cr.samples, "cr diverged");
}
