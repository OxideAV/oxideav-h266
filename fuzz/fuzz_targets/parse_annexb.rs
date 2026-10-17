//! Annex B walk + every header parser fuzz entry.
//!
//! Treats the raw fuzz input as an Annex B byte stream (§B.2): walks
//! every NAL unit, parses the §7.3.1.2 two-byte header, strips the
//! §7.4.1 emulation-prevention bytes and dispatches the RBSP to the
//! matching parser — DCI (§7.3.2.1), OPI (§7.3.2.2), VPS (§7.3.2.3),
//! SPS (§7.3.2.4), PPS (§7.3.2.5), APS (§7.3.2.6), picture header
//! (§7.3.2.8, lead + the stateful parse against the pooled SPS / PPS),
//! AUD (§7.3.2.10), EOS / EOB / filler (§7.3.2.11 – §7.3.2.13), SEI
//! RBSP (§7.3.2.9) with the context-free payload bodies (buffering
//! period, SEI manifest, SEI prefix indication, subpicture level
//! info), and every VCL NAL unit through the slice-header lead and —
//! once an SPS + PPS pair is pooled — the stateful §7.3.7 parse
//! (PH-in-SH or against the last pooled picture header).
//!
//! Every path must return `Ok` / `Err` — no panics, no aborts, no
//! unbounded allocation — regardless of input.

#![no_main]

use std::collections::HashMap;

use libfuzzer_sys::fuzz_target;
use oxideav_h266::nal::{extract_rbsp, iter_annex_b, NalUnitType};
use oxideav_h266::picture_header::{
    parse_picture_header, parse_picture_header_stateful, PictureHeader,
};
use oxideav_h266::pps::{parse_pps, PicParameterSet};
use oxideav_h266::slice_header::{parse_slice_header, parse_slice_header_stateful, PhState};
use oxideav_h266::sps::{parse_sps, SeqParameterSet};

fuzz_target!(|data: &[u8]| {
    let mut spss: HashMap<u8, SeqParameterSet> = HashMap::new();
    let mut ppss: HashMap<u8, PicParameterSet> = HashMap::new();
    let mut ph: Option<PictureHeader> = None;
    for nal in iter_annex_b(data) {
        let t = nal.header.nal_unit_type;
        let rbsp = extract_rbsp(nal.payload());
        match t {
            NalUnitType::DciNut => {
                let _ = oxideav_h266::dci::parse_dci(&rbsp);
            }
            NalUnitType::OpiNut => {
                let _ = oxideav_h266::opi::parse_opi(&rbsp);
            }
            NalUnitType::VpsNut => {
                let _ = oxideav_h266::vps::parse_vps(&rbsp);
            }
            NalUnitType::SpsNut => {
                if let Ok(sps) = parse_sps(&rbsp) {
                    // Keep the pool bounded: a hostile stream may name
                    // every id.
                    spss.insert(sps.sps_seq_parameter_set_id, sps);
                }
            }
            NalUnitType::PpsNut => {
                if let Ok(pps) = parse_pps(&rbsp) {
                    ppss.insert(pps.pps_pic_parameter_set_id, pps);
                }
            }
            NalUnitType::PrefixApsNut | NalUnitType::SuffixApsNut => {
                let _ = oxideav_h266::aps::parse_aps(&rbsp);
            }
            NalUnitType::PhNut => {
                if let Ok(lead) = parse_picture_header(&rbsp) {
                    if let Some(pps) = ppss.get(&(lead.ph_pic_parameter_set_id as u8)) {
                        if let Some(sps) = spss.get(&pps.pps_seq_parameter_set_id) {
                            ph = parse_picture_header_stateful(&rbsp, sps, pps).ok();
                        }
                    }
                }
            }
            NalUnitType::AudNut => {
                let _ = oxideav_h266::aud::parse_access_unit_delimiter(&rbsp);
            }
            NalUnitType::EosNut => {
                let _ = oxideav_h266::end_of_seq::parse_end_of_seq(&rbsp);
            }
            NalUnitType::EobNut => {
                let _ = oxideav_h266::end_of_bitstream::parse_end_of_bitstream(&rbsp);
            }
            NalUnitType::FdNut => {
                let _ = oxideav_h266::filler_data::parse_filler_data(&rbsp);
            }
            NalUnitType::PrefixSeiNut | NalUnitType::SuffixSeiNut => {
                if let Ok(sei) = oxideav_h266::sei_rbsp::parse_sei_rbsp(&rbsp) {
                    for m in &sei.messages {
                        match m.payload_type {
                            0 => {
                                let _ = oxideav_h266::buffering_period::parse_buffering_period(
                                    m.payload,
                                );
                            }
                            200 => {
                                let _ = oxideav_h266::sei_manifest::parse_sei_manifest(m.payload);
                            }
                            201 => {
                                let _ = oxideav_h266::sei_prefix_indication::parse_sei_prefix_indication(m.payload);
                            }
                            203 => {
                                let _ = oxideav_h266::subpic_level_info::parse_subpic_level_info(
                                    m.payload,
                                );
                            }
                            _ => {}
                        }
                    }
                }
            }
            _ if t.is_vcl() => {
                let Ok(lead) = parse_slice_header(&rbsp) else {
                    continue;
                };
                let pps_id = match lead.embedded_picture_header.as_ref() {
                    Some(l) => l.ph_pic_parameter_set_id,
                    None => match ph.as_ref() {
                        Some(p) => p.ph_pic_parameter_set_id,
                        None => continue,
                    },
                };
                let Some(pps) = ppss.get(&(pps_id as u8)) else {
                    continue;
                };
                let Some(sps) = spss.get(&pps.pps_seq_parameter_set_id) else {
                    continue;
                };
                let ph_state = PhState {
                    nal_unit_type: t,
                    num_extra_sh_bits: sps.num_extra_sh_bits,
                    ..Default::default()
                };
                let _ = parse_slice_header_stateful(&rbsp, sps, pps, &ph_state);
            }
            _ => {}
        }
    }
});
