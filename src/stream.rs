//! r434 — whole-bitstream decode driver.
//!
//! Everything below the picture level (CTU walk, reconstruction,
//! in-loop filters) has lived in [`crate::ctu`] for many rounds; what
//! was missing was the stream-level glue an Annex-B *elementary
//! stream* needs: parameter-set pools, access-unit assembly, §8.3.1
//! picture-order-count derivation, §8.3.2 reference-picture-list
//! resolution against a decoded-picture buffer, §8.3.3-style DPB
//! retention, CLVSS / RASL handling across concatenated CVSs, and the
//! per-slice walker wiring (ALF / LMCS / scaling-list APS bindings,
//! temporal-MVP collocated selection, the PH inter-tool switches).
//!
//! [`StreamDecoder::decode_annex_b`] drives a caller-supplied sink
//! with every decoded picture in decode order; each carries its CVS
//! index, `PicOrderCntVal`, output flag and conformance-crop window,
//! so output-order sequencing (ascending POC within a CVS for these
//! single-AU-per-picture streams) is a caller-side sort.
//!
//! Current fail-fast gates (surfaced as `Error::Unsupported` with a
//! precise reason, so a conformance triage can classify streams):
//! bit depths above 8 (the reconstruction planes are `u8`), 4:2:2 /
//! 4:4:4 chroma formats, subpicture layouts, explicit weighted
//! prediction, and §8.8.1 explicit virtual boundaries.

use std::collections::HashMap;

use oxideav_core::{Error, Result};

use crate::alf::AlfApsBinding;
use crate::alf_syntax::AlfSyntaxConfig;
use crate::aps::{parse_aps, AlfApsData, ApsParamsType};
use crate::ctu::{CtuLayout, CtuWalker};
use crate::inter::{MotionField, ReferencePicture};
use crate::lmcs::LmcsData;
use crate::nal::{extract_rbsp, iter_annex_b, NalUnitType};
use crate::picture_header::{parse_picture_header, parse_picture_header_stateful, PictureHeader};
use crate::pps::{parse_pps, PicParameterSet};
use crate::reconstruct::PictureBuffer;
use crate::ref_pic_list::{
    build_ref_pic_list, BuiltRefPicEntry, HeaderRefPicList, RefPicKind, RefPicListInputs,
};
use crate::scaling_list::ScalingListData;
use crate::slice_header::{parse_slice_header_stateful, PhState, SliceType, StatefulSliceHeader};
use crate::sps::{parse_sps, SeqParameterSet};

/// One decoded picture, in decode order.
pub struct DecodedPicture {
    /// 0-based index of the CVS this picture belongs to (increments at
    /// every CLVSS picture — IDR, or CRA/GDR starting the sequence).
    pub cvs_idx: u32,
    /// `PicOrderCntVal` (§8.3.1).
    pub poc: i32,
    /// `ph_pic_output_flag` — `false` pictures are decoded for
    /// reference only and never presented.
    pub output_flag: bool,
    /// Full (uncropped) reconstruction. Chroma planes are meaningful
    /// only for `chroma_format_idc == 1`.
    pub frame: PictureBuffer,
    /// Conformance-crop window in luma samples: `(left, top, width,
    /// height)`.
    pub crop: (usize, usize, usize, usize),
    /// `sps_chroma_format_idc`.
    pub chroma_format_idc: u8,
    /// Luma/chroma bit depth (8 for the current pipeline).
    pub bit_depth: u32,
}

/// A reference picture retained in the decoded-picture buffer.
struct DpbEntry {
    poc: i32,
    frame: PictureBuffer,
    motion: Option<MotionField>,
}

/// One picture's NAL bundle while its slices are being collected.
struct PictureUnit {
    nal_type: NalUnitType,
    temporal_id: u8,
    ph: PictureHeader,
    /// Parsed slice headers in decode order.
    slices: Vec<StatefulSliceHeader>,
}

/// Annex-B VVC elementary-stream decoder (stream-level glue over the
/// CTU walker).
#[derive(Default)]
pub struct StreamDecoder {
    spss: HashMap<u8, SeqParameterSet>,
    ppss: HashMap<u8, PicParameterSet>,
    alf_apss: HashMap<u8, AlfApsData>,
    lmcs_apss: HashMap<u8, LmcsData>,
    sl_apss: HashMap<u8, ScalingListData>,
    dpb: Vec<DpbEntry>,
    /// (PicOrderCntLsb, PicOrderCntMsb) of `prevTid0Pic` (§8.3.1).
    prev_tid0: Option<(u32, i32)>,
    /// True when the next IRAP/GDR picture starts a new CLVS (stream
    /// start / after end-of-sequence).
    clvss_pending: bool,
    /// True while RASL pictures associated with the last IRAP must be
    /// discarded (the IRAP was a CLVSS CRA — its leading pictures
    /// reference pictures from before the random-access point).
    discard_rasl: bool,
    cvs_idx: u32,
    started: bool,
}

impl StreamDecoder {
    pub fn new() -> Self {
        Self {
            clvss_pending: true,
            ..Default::default()
        }
    }

    /// Decode a whole Annex-B elementary stream, feeding every decoded
    /// picture (in decode order) to `sink`.
    pub fn decode_annex_b(
        &mut self,
        bs: &[u8],
        sink: &mut dyn FnMut(DecodedPicture),
    ) -> Result<()> {
        let mut pending_ph: Option<PictureHeader> = None;
        let mut cur: Option<PictureUnit> = None;

        for nal in iter_annex_b(bs) {
            let t = nal.header.nal_unit_type;
            match t {
                NalUnitType::SpsNut | NalUnitType::PpsNut | NalUnitType::PrefixApsNut => {
                    // A prefix non-VCL NAL opens a new access unit —
                    // finish the picture in flight before the pools
                    // change under it.
                    if let Some(pic) = cur.take() {
                        self.decode_picture(pic, sink)?;
                    }
                    let rbsp = extract_rbsp(nal.payload());
                    match t {
                        NalUnitType::SpsNut => {
                            let sps = parse_sps(&rbsp)?;
                            self.spss.insert(sps.sps_seq_parameter_set_id, sps);
                        }
                        NalUnitType::PpsNut => {
                            let pps = parse_pps(&rbsp)?;
                            self.ppss.insert(pps.pps_pic_parameter_set_id, pps);
                        }
                        _ => self.stash_aps(&rbsp)?,
                    }
                }
                NalUnitType::SuffixApsNut => {
                    if let Some(pic) = cur.take() {
                        self.decode_picture(pic, sink)?;
                    }
                    let rbsp = extract_rbsp(nal.payload());
                    self.stash_aps(&rbsp)?;
                }
                NalUnitType::PhNut => {
                    if let Some(pic) = cur.take() {
                        self.decode_picture(pic, sink)?;
                    }
                    let rbsp = extract_rbsp(nal.payload());
                    let lead = parse_picture_header(&rbsp)?;
                    let (sps, pps) = self.resolve_ps(lead.ph_pic_parameter_set_id)?;
                    pending_ph = Some(parse_picture_header_stateful(&rbsp, sps, pps)?);
                }
                NalUnitType::EosNut | NalUnitType::EobNut => {
                    if let Some(pic) = cur.take() {
                        self.decode_picture(pic, sink)?;
                    }
                    self.clvss_pending = true;
                }
                _ if t.is_vcl() => {
                    let rbsp = extract_rbsp(nal.payload());
                    let temporal_id = nal.header.nuh_temporal_id_plus1.saturating_sub(1);
                    // Peek the first bit: PH-in-SH starts a new picture.
                    let ph_in_sh = rbsp.first().map(|b| b & 0x80 != 0).unwrap_or(false);
                    if ph_in_sh {
                        if let Some(pic) = cur.take() {
                            self.decode_picture(pic, sink)?;
                        }
                        let lead = crate::slice_header::parse_slice_header(&rbsp)?;
                        let pps_id = lead
                            .embedded_picture_header
                            .as_ref()
                            .map(|l| l.ph_pic_parameter_set_id)
                            .ok_or_else(|| Error::invalid("h266 stream: PH-in-SH without PH"))?;
                        let (sps, pps) = self.resolve_ps(pps_id)?;
                        let ph_state = PhState {
                            nal_unit_type: t,
                            num_extra_sh_bits: sps.num_extra_sh_bits,
                            ..Default::default()
                        };
                        let sh = parse_slice_header_stateful(&rbsp, sps, pps, &ph_state)?;
                        let ph =
                            sh.embedded_ph.as_deref().cloned().ok_or_else(|| {
                                Error::invalid("h266 stream: embedded PH missing")
                            })?;
                        cur = Some(PictureUnit {
                            nal_type: t,
                            temporal_id,
                            ph,
                            slices: vec![sh],
                        });
                    } else {
                        // Separate PH: either the pending PhNut opens a
                        // new picture, or this slice continues the
                        // current one.
                        if let Some(ph) = pending_ph.take() {
                            if let Some(pic) = cur.take() {
                                self.decode_picture(pic, sink)?;
                            }
                            let sh = self.parse_slice_with_ph(&rbsp, &ph, t)?;
                            cur = Some(PictureUnit {
                                nal_type: t,
                                temporal_id,
                                ph,
                                slices: vec![sh],
                            });
                        } else {
                            let pic = cur.as_mut().ok_or_else(|| {
                                Error::invalid("h266 stream: VCL NAL before any picture header")
                            })?;
                            let ph = pic.ph.clone();
                            let sh = self.parse_slice_with_ph(&rbsp, &ph, t)?;
                            cur.as_mut().unwrap().slices.push(sh);
                        }
                    }
                }
                _ => {} // AUD / SEI / filler / reserved — no decode state.
            }
        }
        if let Some(pic) = cur.take() {
            self.decode_picture(pic, sink)?;
        }
        Ok(())
    }

    fn stash_aps(&mut self, rbsp: &[u8]) -> Result<()> {
        let aps = parse_aps(rbsp)?;
        match aps.aps_params_type {
            ApsParamsType::Alf => {
                if let Some(d) = aps.alf_data {
                    self.alf_apss.insert(aps.aps_adaptation_parameter_set_id, d);
                }
            }
            ApsParamsType::Lmcs => {
                if let Some(d) = aps.lmcs_data {
                    self.lmcs_apss
                        .insert(aps.aps_adaptation_parameter_set_id, d);
                }
            }
            ApsParamsType::Scaling => {
                if let Some(d) = aps.scaling_data {
                    self.sl_apss.insert(aps.aps_adaptation_parameter_set_id, d);
                }
            }
            ApsParamsType::Reserved(_) => {}
        }
        Ok(())
    }

    fn resolve_ps(&self, pps_id: u32) -> Result<(&SeqParameterSet, &PicParameterSet)> {
        let pps = self
            .ppss
            .get(&(pps_id as u8))
            .ok_or_else(|| Error::invalid(format!("h266 stream: unknown PPS id {pps_id}")))?;
        let sps = self
            .spss
            .get(&pps.pps_seq_parameter_set_id)
            .ok_or_else(|| {
                Error::invalid(format!(
                    "h266 stream: unknown SPS id {}",
                    pps.pps_seq_parameter_set_id
                ))
            })?;
        Ok((sps, pps))
    }

    fn parse_slice_with_ph(
        &self,
        rbsp: &[u8],
        ph: &PictureHeader,
        nal_type: NalUnitType,
    ) -> Result<StatefulSliceHeader> {
        let (sps, pps) = self.resolve_ps(ph.ph_pic_parameter_set_id)?;
        let ph_state = PhState {
            ph_inter_slice_allowed_flag: ph.ph_inter_slice_allowed_flag,
            ph_intra_slice_allowed_flag: ph.ph_intra_slice_allowed_flag,
            ph_alf_enabled_flag: ph.ph_alf_enabled_flag,
            ph_lmcs_enabled_flag: ph.ph_lmcs_enabled_flag,
            ph_explicit_scaling_list_enabled_flag: ph.ph_explicit_scaling_list_enabled_flag,
            ph_temporal_mvp_enabled_flag: ph.ph_temporal_mvp_enabled_flag,
            ph_sao_luma_enabled_flag: ph.ph_sao_luma_enabled_flag,
            ph_sao_chroma_enabled_flag: ph.ph_sao_chroma_enabled_flag,
            num_extra_sh_bits: sps.num_extra_sh_bits,
            nal_unit_type: nal_type,
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
        };
        parse_slice_header_stateful(rbsp, sps, pps, &ph_state)
    }

    /// §8.3.1 — PicOrderCntVal.
    fn derive_poc(
        &self,
        sps: &SeqParameterSet,
        ph: &PictureHeader,
        is_clvss: bool,
        nal_type: NalUnitType,
        temporal_id: u8,
    ) -> i32 {
        let lsb_bits = sps.sps_log2_max_pic_order_cnt_lsb_minus4 as u32 + 4;
        let max_lsb = 1i32 << lsb_bits;
        let lsb = ph.ph_pic_order_cnt_lsb as i32;
        let msb = if ph.ph_poc_msb_cycle_present_flag {
            ph.ph_poc_msb_cycle_val as i32 * max_lsb
        } else if is_clvss {
            0
        } else if let Some((prev_lsb, prev_msb)) = self.prev_tid0 {
            let prev_lsb = prev_lsb as i32;
            if lsb < prev_lsb && prev_lsb - lsb >= max_lsb / 2 {
                prev_msb + max_lsb
            } else if lsb > prev_lsb && lsb - prev_lsb > max_lsb / 2 {
                prev_msb - max_lsb
            } else {
                prev_msb
            }
        } else {
            0
        };
        let _ = (nal_type, temporal_id);
        msb + lsb
    }

    /// §8.3.2 — resolve one list against the DPB into walker-ready
    /// reference pictures. `active` truncates to `NumRefIdxActive`.
    fn resolve_list(
        &self,
        built: &[BuiltRefPicEntry],
        active: usize,
        max_poc_lsb: i32,
    ) -> Result<Vec<ReferencePicture>> {
        let mut out = Vec::new();
        for entry in built.iter().take(active) {
            let found =
                match entry.kind {
                    RefPicKind::ShortTerm => self.dpb.iter().find(|d| d.poc == entry.poc),
                    RefPicKind::LongTerm => {
                        if entry.is_poc_lsb_only {
                            self.dpb.iter().find(|d| {
                                d.poc & (max_poc_lsb - 1) == entry.poc & (max_poc_lsb - 1)
                            })
                        } else {
                            self.dpb.iter().find(|d| d.poc == entry.poc)
                        }
                    }
                    RefPicKind::InterLayer => return Err(Error::unsupported(
                        "h266 stream: inter-layer reference pictures (multilayer) not supported",
                    )),
                };
            let found = found.ok_or_else(|| {
                Error::invalid(format!(
                    "h266 stream: reference POC {} not in DPB (have {:?})",
                    entry.poc,
                    self.dpb.iter().map(|d| d.poc).collect::<Vec<_>>()
                ))
            })?;
            out.push(ReferencePicture {
                poc: found.poc,
                frame: found.frame.clone(),
                motion_field: found.motion.clone(),
            });
        }
        Ok(out)
    }

    fn decode_picture(
        &mut self,
        mut pic: PictureUnit,
        sink: &mut dyn FnMut(DecodedPicture),
    ) -> Result<()> {
        // §7.4.8 — when the PPS routes deblocking parameters through a
        // separate PH NAL, absent slice-level parameters inherit the
        // PH-carried values (the SH parser handles the embedded-PH and
        // PPS arms itself).
        if let Some(pps) = self.ppss.get(&(pic.ph.ph_pic_parameter_set_id as u8)) {
            if pps.pps_dbf_info_in_ph_flag && pic.ph.deblocking.present_flag {
                let d = pic.ph.deblocking;
                for sh in &mut pic.slices {
                    if !sh.sh_deblocking_params_present_flag && sh.embedded_ph.is_none() {
                        sh.sh_deblocking_filter_disabled_flag = d.filter_disabled_flag;
                        sh.sh_luma_beta_offset_div2 = d.luma_beta_offset_div2;
                        sh.sh_luma_tc_offset_div2 = d.luma_tc_offset_div2;
                        sh.sh_cb_beta_offset_div2 = d.cb_beta_offset_div2;
                        sh.sh_cb_tc_offset_div2 = d.cb_tc_offset_div2;
                        sh.sh_cr_beta_offset_div2 = d.cr_beta_offset_div2;
                        sh.sh_cr_tc_offset_div2 = d.cr_tc_offset_div2;
                    }
                }
            }
        }
        let ph = &pic.ph;
        let (sps, pps) = self.resolve_ps(ph.ph_pic_parameter_set_id)?;
        let sps = sps.clone();
        let pps = pps.clone();

        // --- fail-fast tool gates -----------------------------------
        if sps.bit_depth_y() > 8 {
            return Err(Error::unsupported(format!(
                "h266 stream: {}-bit reconstruction not supported (u8 pipeline)",
                sps.bit_depth_y()
            )));
        }
        if sps.sps_chroma_format_idc >= 2 {
            return Err(Error::unsupported(format!(
                "h266 stream: chroma format idc {} (4:2:2/4:4:4) not supported",
                sps.sps_chroma_format_idc
            )));
        }
        if sps.tool_flags.virtual_boundaries_enabled_flag
            && (sps.tool_flags.virtual_boundaries.is_some()
                || ph.ph_virtual_boundaries_present_flag)
        {
            return Err(Error::unsupported(
                "h266 stream: explicit §8.8.1 virtual boundaries not supported",
            ));
        }

        let is_irap = matches!(
            pic.nal_type,
            NalUnitType::IdrWRadl | NalUnitType::IdrNLp | NalUnitType::CraNut
        );
        let is_idr = matches!(pic.nal_type, NalUnitType::IdrWRadl | NalUnitType::IdrNLp);
        let is_clvss = (is_irap || pic.nal_type == NalUnitType::GdrNut)
            && (is_idr || self.clvss_pending || !self.started);
        self.started = true;
        if is_clvss {
            // New CLVS: prior pictures are gone (their POC domain
            // ends). Output ordering restarts with the new CVS index.
            self.dpb.clear();
            self.prev_tid0 = None;
            self.cvs_idx += 1;
            self.clvss_pending = false;
            self.discard_rasl = is_irap && !is_idr; // CLVSS CRA → drop its RASLs
        } else if is_irap || pic.nal_type == NalUnitType::GdrNut {
            self.discard_rasl = false;
        }

        let poc = self.derive_poc(&sps, ph, is_clvss, pic.nal_type, pic.temporal_id);
        let max_poc_lsb = 1i32 << (sps.sps_log2_max_pic_order_cnt_lsb_minus4 as u32 + 4);

        // RASL pictures of a CLVSS CRA reference pre-RAP pictures that
        // were never decoded — the spec discards them.
        if pic.nal_type == NalUnitType::RaslNut && self.discard_rasl {
            return Ok(());
        }

        // §8.3.1 — prevTid0Pic update happens for the *decoded*
        // picture below (after RASL discard), before the next call.

        // --- per-slice RPL resolution + DPB retention ---------------
        // Every slice re-derives its lists; slice 0's union drives the
        // §8.3.3-style retention sweep after the picture decodes.
        let header_rpls_of = |sh: &StatefulSliceHeader| -> Option<[HeaderRefPicList; 2]> {
            sh.sh_ref_pic_lists
                .clone()
                .or_else(|| ph.ref_pic_lists.clone())
        };

        let build_lists = |sh: &StatefulSliceHeader| -> Result<[Vec<BuiltRefPicEntry>; 2]> {
            match header_rpls_of(sh) {
                Some(rpls) => {
                    let mut out: [Vec<BuiltRefPicEntry>; 2] = [Vec::new(), Vec::new()];
                    for i in 0..2 {
                        out[i] = build_ref_pic_list(&RefPicListInputs {
                            current_poc: poc,
                            max_poc_lsb: max_poc_lsb as u32,
                            rpls: &rpls[i].rpls,
                            lt_info: &rpls[i].lt_info,
                        })?;
                    }
                    Ok(out)
                }
                None => Ok([Vec::new(), Vec::new()]),
            }
        };

        // --- decode ----------------------------------------------------
        let layout = CtuLayout::from_sps_pps(&sps, &pps);
        let sh0 = &pic.slices[0];

        // Explicit weighted prediction has no walker path yet.
        for sh in &pic.slices {
            if sh.sh_pred_weight_table.is_some()
                || (pps.pps_wp_info_in_ph_flag
                    && ((pps.pps_weighted_pred_flag && sh.sh_slice_type == SliceType::P)
                        || (pps.pps_weighted_bipred_flag && sh.sh_slice_type == SliceType::B)))
            {
                return Err(Error::unsupported(
                    "h266 stream: explicit weighted prediction not supported",
                ));
            }
        }

        let cabacs: Vec<Vec<u8>> = pic
            .slices
            .iter()
            .map(|sh| {
                let mut c = sh.trailing_bits.clone();
                c.extend_from_slice(&[0u8; 64]);
                c
            })
            .collect();

        let mut walker =
            CtuWalker::begin_slice(&layout, &sps, &pps, sh0, ph.ph_qp_delta, &cabacs[0])?;

        // PH-level switches.
        if pps.pps_cu_qp_delta_enabled_flag {
            walker.set_cu_qp_delta_subdiv(if sh0.sh_slice_type == SliceType::I {
                ph.ph_cu_qp_delta_subdiv_intra_slice
            } else {
                ph.ph_cu_qp_delta_subdiv_inter_slice
            });
        }
        walker.set_ph_mvd_l1_zero(ph.ph_mvd_l1_zero_flag);
        walker.set_ph_bdof_disabled(ph.ph_bdof_disabled_flag);
        walker.set_ph_dmvr_disabled(ph.ph_dmvr_disabled_flag);
        walker.set_ph_prof_disabled(ph.ph_prof_disabled_flag);
        walker.set_ph_mmvd_fullpel_only(ph.ph_mmvd_fullpel_only_flag);
        walker.set_ph_joint_cbcr_sign(ph.ph_joint_cbcr_sign_flag);

        // ALF CTU-prefix decode config (per continue_slice contract the
        // ALF selections must agree across the picture's slices).
        if sh0.sh_alf_enabled_flag {
            let chroma_aps = self.alf_apss.get(&sh0.sh_alf_aps_id_chroma);
            let cc_cb_aps = self.alf_apss.get(&sh0.sh_alf_cc_cb_aps_id);
            let cc_cr_aps = self.alf_apss.get(&sh0.sh_alf_cc_cr_aps_id);
            walker.set_alf_decode(AlfSyntaxConfig {
                alf_enabled: true,
                cb_enabled: sh0.sh_alf_cb_enabled_flag,
                cr_enabled: sh0.sh_alf_cr_enabled_flag,
                cc_cb_enabled: sh0.sh_alf_cc_cb_enabled_flag,
                cc_cr_enabled: sh0.sh_alf_cc_cr_enabled_flag,
                sh_num_alf_aps_ids_luma: sh0.sh_num_alf_aps_ids_luma,
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
                slice_type: sh0.sh_slice_type,
                sh_cabac_init_flag: sh0.sh_cabac_init_flag,
            });
        }
        if sh0.sh_lmcs_used_flag {
            let data = self.lmcs_apss.get(&ph.ph_lmcs_aps_id).ok_or_else(|| {
                Error::invalid(format!(
                    "h266 stream: LMCS APS id {} not seen",
                    ph.ph_lmcs_aps_id
                ))
            })?;
            walker.set_lmcs(data, ph.ph_chroma_residual_scale_flag)?;
        }
        if sh0.sh_explicit_scaling_list_used_flag {
            let data = self
                .sl_apss
                .get(&ph.ph_scaling_list_aps_id)
                .ok_or_else(|| {
                    Error::invalid(format!(
                        "h266 stream: scaling-list APS id {} not seen",
                        ph.ph_scaling_list_aps_id
                    ))
                })?;
            walker.set_scaling_list(data);
        }

        let w = pps.pps_pic_width_in_luma_samples as usize;
        let h = pps.pps_pic_height_in_luma_samples as usize;
        let mut out = PictureBuffer::yuv420_filled(w, h, 0);

        // Union of all RPL entries as (poc, lsb_only) match specs —
        // the §8.3.3 retention sweep must match the same way the
        // resolver does (an LTRP named by LSB only still retains).
        let mut union_refs: Vec<(i32, bool)> = Vec::new();
        for (i, sh) in pic.slices.iter().enumerate() {
            if i > 0 {
                walker.continue_slice(sh, ph.ph_qp_delta, &cabacs[i])?;
            }
            // Per-slice reference lists + collocated selection.
            let built = build_lists(sh)?;
            for l in &built {
                for e in l {
                    let key = (e.poc, e.is_poc_lsb_only);
                    if !union_refs.contains(&key) {
                        union_refs.push(key);
                    }
                }
            }
            if sh.sh_slice_type != SliceType::I {
                let l0 =
                    self.resolve_list(&built[0], sh.num_ref_idx_active[0] as usize, max_poc_lsb)?;
                let l1 =
                    self.resolve_list(&built[1], sh.num_ref_idx_active[1] as usize, max_poc_lsb)?;
                walker.set_ref_pic_list_l0(l0);
                walker.set_ref_pic_list_l1(l1);
                walker.set_temporal_mvp(
                    poc,
                    ph.ph_temporal_mvp_enabled_flag,
                    sh.sh_collocated_from_l0_flag,
                    sh.sh_collocated_ref_idx,
                );
            } else {
                walker.set_ref_pic_list_l0(Vec::new());
                walker.set_ref_pic_list_l1(Vec::new());
                walker.set_temporal_mvp(poc, false, true, 0);
            }
            walker.decode_picture_into(&mut out)?;
            walker.finish_slice()?;
        }

        // §8.8 in-loop filters (ALF APS bindings from the last slice —
        // continue_slice enforces cross-slice agreement).
        let sh_last = pic.slices.last().unwrap();
        let luma_slots: Vec<Option<&AlfApsData>> = sh_last
            .sh_alf_aps_id_luma
            .iter()
            .map(|id| self.alf_apss.get(id))
            .collect();
        let binding = AlfApsBinding {
            luma_apses: &luma_slots,
            chroma_aps: if sh_last.sh_alf_cb_enabled_flag || sh_last.sh_alf_cr_enabled_flag {
                self.alf_apss.get(&sh_last.sh_alf_aps_id_chroma)
            } else {
                None
            },
            cc_cb_aps: if sh_last.sh_alf_cc_cb_enabled_flag {
                self.alf_apss.get(&sh_last.sh_alf_cc_cb_aps_id)
            } else {
                None
            },
            cc_cr_aps: if sh_last.sh_alf_cc_cr_enabled_flag {
                self.alf_apss.get(&sh_last.sh_alf_cc_cr_aps_id)
            } else {
                None
            },
        };
        walker.apply_in_loop_filters_with_alf(&mut out, &binding)?;

        let motion = walker.motion_field_for_temporal();
        drop(walker);

        // §8.3.3-style retention: keep only pictures the just-decoded
        // picture's RPL union still names (the spec marks everything
        // else "unused for reference" when the lists are built).
        if !union_refs.is_empty() {
            self.dpb.retain(|d| {
                union_refs.iter().any(|&(p, lsb_only)| {
                    if lsb_only {
                        d.poc & (max_poc_lsb - 1) == p & (max_poc_lsb - 1)
                    } else {
                        d.poc == p
                    }
                })
            });
        }

        // Store the current picture for future reference.
        self.dpb.push(DpbEntry {
            poc,
            frame: out.clone(),
            motion: Some(motion),
        });

        // prevTid0Pic (§8.3.1).
        if pic.temporal_id == 0
            && !matches!(pic.nal_type, NalUnitType::RaslNut | NalUnitType::RadlNut)
        {
            let lsb = ph.ph_pic_order_cnt_lsb;
            let msb = poc - lsb as i32;
            self.prev_tid0 = Some((lsb, msb));
        }

        // Conformance crop (§7.4.3.5: the PPS window applies when
        // present; a PPS at the SPS max size inherits the SPS window).
        let (sub_w, sub_h) = match sps.sps_chroma_format_idc {
            1 => (2usize, 2usize),
            2 => (2, 1),
            _ => (1, 1),
        };
        let win = pps
            .conformance_window
            .map(|c| (c.left_offset, c.top_offset, c.right_offset, c.bottom_offset))
            .or_else(|| {
                sps.conformance_window
                    .map(|c| (c.left_offset, c.top_offset, c.right_offset, c.bottom_offset))
            });
        let (left, top, right, bottom) = win
            .map(|(l, t, r, b)| {
                (
                    l as usize * sub_w,
                    t as usize * sub_h,
                    r as usize * sub_w,
                    b as usize * sub_h,
                )
            })
            .unwrap_or((0, 0, 0, 0));
        let crop_w = w - left - right;
        let crop_h = h - top - bottom;

        sink(DecodedPicture {
            cvs_idx: self.cvs_idx.saturating_sub(1),
            poc,
            output_flag: ph.ph_pic_output_flag,
            frame: out,
            crop: (left, top, crop_w, crop_h),
            chroma_format_idc: sps.sps_chroma_format_idc,
            bit_depth: sps.bit_depth_y(),
        });
        Ok(())
    }
}
