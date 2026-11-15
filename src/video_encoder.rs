//! r490 — video encoder: IDR pictures followed by inter (P) pictures
//! in one conforming Annex B stream.
//!
//! [`VideoEncoder`] codes a sequence in a low-delay `I P P P …`
//! structure: every `keyint`-th picture (and every picture a P coding
//! does not pay for — a scene change) is an IDR coded by the stills
//! pipeline ([`crate::encoder_pipeline`]), the pictures between them
//! are single-slice P pictures that predict from the previous picture
//! (`ref_pic_lists()` with one short-term entry).
//!
//! # P pictures
//!
//! * **Coding tree** (§7.3.11.4) — a quad tree from the 128x128 CTB
//!   down to 8x8, the split decided per node by `SSE + λ · bits`
//!   against the best leaf of the node; nodes crossing the coded
//!   picture edge take the §7.4.12.4 inferred split.
//! * **Coding units** (§7.3.11.5 / §7.3.11.7) — `cu_skip_flag` +
//!   `merge_idx`; `general_merge_flag` + `merge_idx` + residual; or the
//!   non-merge cascade (`mvd_coding()`, `mvp_l0_flag`, `cu_coded_flag`)
//!   + residual. A 128x128 CU is only ever a skip CU.
//! * **Predictors** — the merge list (§8.5.2.2: spatial candidates,
//!   the temporal candidate of §8.5.2.11 from the previous picture's
//!   motion field as the decoder stored it, history candidates,
//!   pairwise average, zero padding) and the motion
//!   vector predictor list (§8.5.2.8 – §8.5.2.10) are the decoder's own
//!   derivations ([`crate::inter`], [`crate::amvp`]) run against a
//!   motion field and a history table the encoder maintains in coding
//!   order, so every index on the wire selects the vector the encoder
//!   meant.
//! * **Motion search** — per CU: the merge candidates, the two
//!   predictors, the zero vector and the vector of the enclosing node
//!   seed a step-halving pattern search on integer positions (SAD +
//!   `√λ ·` vector bits), refined to half and quarter sample positions
//!   through the §8.5.6.3 interpolation.
//! * **Residual** (§7.3.11.10) — one luma and two chroma DCT-II
//!   transform blocks per CU, quantised with the zero bin widened to a
//!   quarter step above half a step ([`INTER_ROUNDING`]);
//!   `cu_qp_delta_abs = 0` once per CTU (one quantization group per
//!   CTU on inter slices).
//!
//! * **Adaptive loop filter** (§8.8.5) — the access unit is decoded
//!   once without the filter; against that deblocked picture every
//!   luma CTB elects one of the 16 fixed filter sets or no filter
//!   (`alf_ctb_flag`, `alf_luma_fixed_filter_idx`, no APS), and when a
//!   CTB elects one the slice is emitted again with the filter
//!   signalled.
//!
//! The reference picture of the next P picture is obtained by decoding
//! the access unit just written with [`crate::stream::StreamDecoder`]:
//! deblocking and every rounding rule are the decoder's, so encoder and
//! decoder references agree by construction and the encoder only has to
//! mirror the *syntax* state (contexts, neighbour-derived `ctxInc`s,
//! candidate lists).
//!
//! Not in the P coding: intra CUs inside a P slice (a picture that
//! would need them is coded as an IDR instead), bi-prediction,
//! adaptive motion vector resolution, the
//! sub-block / affine / geometric tools, SAO, APS-signalled / chroma /
//! cross-component ALF filters on P slices, and tiles / WPP / multiple
//! slices.

use std::sync::Arc;

use oxideav_core::{Error, Result};

use crate::alf::AlfPicture;
use crate::alf_syntax::{AlfCtxs, AlfNeighbours, AlfSyntaxConfig};
use crate::amvp::{
    AmvpRefContext, AmvpTemporalInputs, MAX_MVP_CAND, RefList, SpatialMvpCandidate,
    build_mvp_cand_list, derive_hmvp_mvp_candidates, derive_spatial_mvp_candidates,
    derive_temporal_amvp_candidate, round_mv_amvr,
};
use crate::amvr::AmvrShift;
use crate::cabac_enc::ArithEncoder;
use crate::coding_tree::{SplitConstraints, TreeCtxs, TreeType};
use crate::dequant::{DequantParams, dequantize_tb_flat};
use crate::encoder::{AlfPhChain, EncoderConfig, VvcEncoder, annex_b_wrap};
use crate::encoder_pipeline::{
    build_nal, chroma_qp_identity_bd, crop_picture, encode_idr_with_residuals_cfg, lambda_for_qp,
    pad_source_to_coded,
};
use crate::inter::{
    HmvpTable, MotionField, MotionVector, MvField, ReferencePicture, TemporalMergeInputs,
    build_merge_cand_list, derive_spatial_merge_candidates, derive_temporal_merge_candidate,
    predict_chroma_block, predict_luma_block,
};
use crate::leaf_cu::LeafCuCtxs;
use crate::nal::NalUnitType;
use crate::reconstruct::{PictureBuffer, PicturePlane};
use crate::residual::{RcOpts, ResidualCtxs};
use crate::residual_enc::{
    encode_tb_coefficients_opts, write_cu_qp_delta, write_tu_cb_coded_flag, write_tu_cr_coded_flag,
    write_tu_y_coded_flag,
};
use crate::slice_header::SliceType;
use crate::stream::StreamDecoder;
use crate::syntax_enc::TreeNeighbours;
use crate::transform::{TrType, inverse_transform_2d};
use crate::transform_fwd::{forward_dct_ii_2d_low, quantize_tb_flat_rounding};

/// Highest value of [`VideoEncoderConfig::effort`].
pub const MAX_EFFORT: u8 = 4;
/// Default of [`VideoEncoderConfig::effort`].
pub const DEFAULT_EFFORT: u8 = 2;
/// Default of [`VideoEncoderConfig::idr_qp_offset`].
pub const DEFAULT_IDR_QP_OFFSET: i32 = -2;
/// Default of [`VideoEncoderConfig::keyint`].
pub const DEFAULT_KEYINT: u32 = 64;

/// `MaxNumMergeCand` of the emitted SPS
/// (`sps_six_minus_max_num_merge_cand = 0`).
const MAX_NUM_MERGE_CAND: u32 = 6;
/// `Log2ParMrgLevel` of the emitted SPS.
const LOG2_PAR_MRG_LEVEL: u32 = 2;
/// CTB size of the emitted SPS.
const CTB_SIZE: usize = 128;
/// Largest CU that carries a residual (`MaxTbSizeY`).
const MAX_RESIDUAL_CU: usize = 64;
/// Rounding point of the inter residual quantiser, in 1/64 of a
/// quantisation step (32 would round to nearest).
const INTER_ROUNDING: i64 = 16;

/// Configuration of a [`VideoEncoder`].
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct VideoEncoderConfig {
    /// Geometry, chroma format, bit depth and the intra tool set of
    /// the IDR pictures.
    pub picture: EncoderConfig,
    /// Slice QP of every picture (0..=63).
    pub qp: i32,
    /// Distance between IDR pictures; 1 codes every picture as an IDR.
    pub keyint: u32,
    /// Search / decision effort of the P pictures, `0..=`[`MAX_EFFORT`]
    /// (see [`VideoEncoder`] for the ladder).
    pub effort: u8,
    /// Code a picture as an IDR when its P coding is larger than the
    /// last IDR picture (a scene change).
    pub scene_cut: bool,
    /// Decide the adaptive loop filter of the P pictures (effort 1 and
    /// above; effort 0 never runs it).
    pub alf: bool,
    /// QP of the IDR pictures relative to `qp` when the sequence has
    /// P pictures (`keyint > 1`): every P picture predicts, directly
    /// or through its predecessors, from the IDR picture, so bits
    /// spent there are shared by the pictures that follow.
    pub idr_qp_offset: i32,
}

impl VideoEncoderConfig {
    /// `picture` at slice QP `qp`, [`DEFAULT_KEYINT`],
    /// [`DEFAULT_EFFORT`], scene-change detection on.
    pub fn new(picture: EncoderConfig, qp: i32) -> Self {
        Self {
            picture,
            qp,
            keyint: DEFAULT_KEYINT,
            effort: DEFAULT_EFFORT,
            scene_cut: true,
            alf: true,
            idr_qp_offset: DEFAULT_IDR_QP_OFFSET,
        }
    }

    /// Builder: IDR distance (`0` is treated as 1).
    pub fn with_keyint(mut self, keyint: u32) -> Self {
        self.keyint = keyint.max(1);
        self
    }

    /// Builder: effort, clamped to [`MAX_EFFORT`].
    pub fn with_effort(mut self, effort: u8) -> Self {
        self.effort = effort.min(MAX_EFFORT);
        self
    }

    /// Builder: scene-change detection.
    pub fn with_scene_cut(mut self, on: bool) -> Self {
        self.scene_cut = on;
        self
    }

    /// Builder: QP offset of the IDR pictures (clamped to −12..=12).
    pub fn with_idr_qp_offset(mut self, offset: i32) -> Self {
        self.idr_qp_offset = offset.clamp(-12, 12);
        self
    }

    /// Builder: the adaptive loop filter decision of the P pictures.
    pub fn with_alf(mut self, on: bool) -> Self {
        self.alf = on;
        self
    }
}

/// How a picture was coded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum PictureKind {
    /// IDR picture (a random access point).
    Idr,
    /// P picture predicting from the previous picture.
    P,
}

/// Per-picture counters of a P coding.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct InterStats {
    /// Skip CUs.
    pub skip_cus: u32,
    /// Merge CUs with a residual.
    pub merge_cus: u32,
    /// Non-merge (explicit motion vector) CUs.
    pub amvp_cus: u32,
    /// CTBs whose luma is filtered by the adaptive loop filter.
    pub alf_ctbs: u32,
}

/// One coded access unit.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct EncodedPicture {
    /// Annex B access unit (start codes; the parameter sets precede
    /// every IDR picture).
    pub data: Vec<u8>,
    /// How the picture was coded.
    pub kind: PictureKind,
    /// `PicOrderCntVal` of the picture inside its coded video sequence.
    pub poc: i32,
    /// CU counters (all zero for an IDR picture).
    pub stats: InterStats,
}

/// The search / decision parameters behind one effort step.
#[derive(Clone, Copy, Debug)]
struct EffortParams {
    /// Integer search: largest pattern step (a power of two, samples).
    max_step: i32,
    /// Integer search: moves allowed per step size.
    max_moves: u32,
    /// Fractional refinement: 0 none, 1 half sample, 2 quarter sample.
    subpel: u8,
    /// Smallest CU.
    min_cu: usize,
    /// Test the split of a node whose best leaf is a residual-free
    /// skip CU.
    split_below_skip: bool,
    /// Eight-neighbour (instead of four-neighbour) integer pattern.
    eight_point: bool,
    /// Decide the adaptive loop filter of the picture (a second emit
    /// and decode of the access unit).
    alf: bool,
}

impl EffortParams {
    fn of(effort: u8) -> Self {
        match effort {
            0 => Self {
                max_step: 2,
                max_moves: 2,
                subpel: 0,
                min_cu: 16,
                split_below_skip: false,
                eight_point: false,
                alf: false,
            },
            1 => Self {
                max_step: 4,
                max_moves: 4,
                subpel: 1,
                min_cu: 16,
                split_below_skip: false,
                eight_point: false,
                alf: true,
            },
            2 => Self {
                max_step: 8,
                max_moves: 6,
                subpel: 2,
                min_cu: 8,
                split_below_skip: false,
                eight_point: false,
                alf: true,
            },
            3 => Self {
                max_step: 16,
                max_moves: 12,
                subpel: 2,
                min_cu: 8,
                split_below_skip: false,
                eight_point: true,
                alf: true,
            },
            _ => Self {
                max_step: 32,
                max_moves: 24,
                subpel: 2,
                min_cu: 8,
                split_below_skip: true,
                eight_point: true,
                alf: true,
            },
        }
    }
}

/// IDR + P picture encoder.
///
/// Effort ladder of the P pictures (`effort`, default
/// [`DEFAULT_EFFORT`]):
///
/// | effort | integer search | fractional | smallest CU | loop filter decision |
/// | --- | --- | --- | --- | --- |
/// | 0 | steps 2, 1 (4 points) | none | 16x16 | — |
/// | 1 | steps 4 … 1 (4 points) | half sample | 16x16 | ALF |
/// | 2 | steps 8 … 1 (4 points) | quarter sample | 8x8 | ALF |
/// | 3 | steps 16 … 1 (8 points) | quarter sample | 8x8 | ALF |
/// | 4 | steps 32 … 1 (8 points), split tested under skip CUs | quarter sample | 8x8 | ALF |
pub struct VideoEncoder {
    cfg: VideoEncoderConfig,
    vvc: VvcEncoder,
    dec: StreamDecoder,
    /// Decoded previous picture at the coded geometry.
    reference: Option<Arc<PictureBuffer>>,
    /// The motion field the decoder stored with the previous picture
    /// (the collocated picture of the next P picture).
    reference_motion: Option<Arc<MotionField>>,
    since_idr: u32,
    last_idr_bytes: usize,
}

impl VideoEncoder {
    /// Validate `cfg` and build the encoder. A configuration with
    /// `keyint > 1` must be lossy, single-tile, single-slice, without
    /// WPP and without palette coding (typed `Unsupported` otherwise).
    pub fn new(mut cfg: VideoEncoderConfig) -> Result<Self> {
        cfg.keyint = cfg.keyint.max(1);
        cfg.effort = cfg.effort.min(MAX_EFFORT);
        if !(0..=63).contains(&cfg.qp) {
            return Err(Error::invalid(format!(
                "h266 video encoder: qp {} outside 0..=63",
                cfg.qp
            )));
        }
        if cfg.keyint > 1 {
            let p = &cfg.picture;
            if p.lossless
                || p.palette
                || p.wpp
                || p.tile_columns > 1
                || p.tile_rows > 1
                || p.slice_per_tile
                || p.raster_slice_layout
                || p.raster_slice_count > 1
            {
                return Err(Error::unsupported(
                    "h266 video encoder: inter pictures (keyint > 1) need a lossy single-tile \
                     single-slice configuration without WPP and palette coding",
                ));
            }
            cfg.picture.inter_pictures = true;
        }
        let vvc = VvcEncoder::new(cfg.picture)?;
        Ok(Self {
            cfg,
            vvc,
            dec: StreamDecoder::new(),
            reference: None,
            reference_motion: None,
            since_idr: 0,
            last_idr_bytes: 0,
        })
    }

    /// The configuration in force (`picture.inter_pictures` set when
    /// `keyint > 1`).
    pub fn config(&self) -> &VideoEncoderConfig {
        &self.cfg
    }

    /// The decoded previous picture, cropped to the source size (what
    /// a decoder outputs for it).
    pub fn reconstruction(&self) -> Option<PictureBuffer> {
        let r = self.reference.as_ref()?;
        let (w, h) = self.cfg.picture.output_size();
        let (sw, sh) = self.cfg.picture.chroma_subsampling();
        Some(crop_picture(
            r,
            w as usize,
            h as usize,
            (w / sw) as usize,
            (h / sh) as usize,
        ))
    }

    /// Make the next picture an IDR.
    pub fn force_idr(&mut self) {
        self.reference = None;
    }

    /// Encode the next picture of the sequence (`src` at the
    /// configured source size and bit depth).
    pub fn encode(&mut self, src: &PictureBuffer) -> Result<EncodedPicture> {
        let p = self.cfg.picture;
        if src.luma.width != p.width as usize || src.luma.height != p.height as usize {
            return Err(Error::invalid(format!(
                "h266 video encoder: source is {}x{}, configured {}x{}",
                src.luma.width, src.luma.height, p.width, p.height
            )));
        }
        if self.cfg.keyint > 1 && self.since_idr < self.cfg.keyint {
            if let Some(reference) = self.reference.clone() {
                let (sw, sh) = p.chroma_subsampling();
                let padded = pad_source_to_coded(
                    src,
                    p.coded_width() as usize,
                    p.coded_height() as usize,
                    sw as usize,
                    sh as usize,
                );
                let coded_src = padded.as_ref().unwrap_or(src);
                let poc = self.since_idr as i32;
                let (coded, mut stats) = self.encode_p(coded_src, &reference, poc)?;
                let au = self.p_access_unit(poc, &coded.plain, None);
                let scene_change = self.cfg.scene_cut
                    && self.last_idr_bytes != 0
                    && au.len() > self.last_idr_bytes;
                if !scene_change {
                    let data = self.commit_p(au, coded_src, poc, &coded, &mut stats)?;
                    self.since_idr += 1;
                    return Ok(EncodedPicture {
                        data,
                        kind: PictureKind::P,
                        poc,
                        stats,
                    });
                }
            }
        }
        let idr_qp = if self.cfg.keyint > 1 {
            (self.cfg.qp + self.cfg.idr_qp_offset).clamp(0, 63)
        } else {
            self.cfg.qp
        };
        let (au, _rec) = encode_idr_with_residuals_cfg(src, idr_qp, p)?;
        self.last_idr_bytes = au.len();
        // The pipeline's own hash message (if any) already closes the
        // access unit; `commit` only decodes it.
        let data = self.commit_decoded(au, false)?;
        self.since_idr = 1;
        Ok(EncodedPicture {
            data,
            kind: PictureKind::Idr,
            poc: 0,
            stats: InterStats::default(),
        })
    }

    /// Decode `au` with the crate's decoder; the decoded picture is
    /// the next reference. With `append_hash` the configured decoded
    /// picture hash SEI message is appended (P pictures — the hash is
    /// of the decoder's reconstruction).
    fn commit_decoded(&mut self, mut au: Vec<u8>, append_hash: bool) -> Result<Vec<u8>> {
        let frame = self.decode_au(&au)?;
        if append_hash {
            self.append_hash(&mut au, &frame);
        }
        self.reference = Some(Arc::new(frame));
        Ok(au)
    }

    /// Decode one access unit with the crate's decoder.
    fn decode_au(&mut self, au: &[u8]) -> Result<PictureBuffer> {
        let mut frame: Option<PictureBuffer> = None;
        self.dec.decode_annex_b(au, &mut |pic| {
            frame = Some(pic.frame);
        })?;
        self.reference_motion = self.dec.newest_picture_motion();
        frame.ok_or_else(|| {
            Error::invalid("h266 video encoder: the coded access unit decoded to no picture")
        })
    }

    /// Append the configured decoded picture hash SEI message of
    /// `frame` to `au`.
    fn append_hash(&self, au: &mut Vec<u8>, frame: &PictureBuffer) {
        let p = &self.cfg.picture;
        if let Some(method) = p.picture_hash {
            let hash = crate::picture_hash::picture_hash(
                frame,
                p.coded_width() as usize,
                p.coded_height() as usize,
                p.chroma_format_idc,
                u32::from(p.bit_depth),
                method,
            );
            let mut sei_rbsp = crate::sei_payload::write_sei_message(
                crate::sei_payload::payload_type::DECODED_PICTURE_HASH,
                &hash.write_payload(),
            );
            sei_rbsp.push(0x80); // rbsp_trailing_bits()
            annex_b_wrap(&build_nal(NalUnitType::SuffixSeiNut, 0, 1, &sei_rbsp), au);
        }
    }

    /// Decide one P picture and emit its slice data without the
    /// adaptive loop filter.
    fn encode_p(
        &self,
        src: &PictureBuffer,
        reference: &Arc<PictureBuffer>,
        poc: i32,
    ) -> Result<(CodedPicture, InterStats)> {
        let p = self.cfg.picture;
        let (sw, sh) = p.chroma_subsampling();
        crate::inter::set_chroma_subsampling(sw, sh);
        crate::inter::set_ref_subpic_clip(None);
        crate::inter::set_ref_wraparound(0);
        // The collocated picture: the previous picture with the motion
        // field the decoder stored for it.
        let col = ReferencePicture {
            poc: poc - 1,
            frame: reference.clone(),
            motion_field: self.reference_motion.clone(),
        };
        let mut pe = PictureCoder::new(&self.cfg, src, reference, col, poc);
        let nodes = pe.decide_picture()?;
        let plain = pe.emit_picture(&nodes, None)?;
        Ok((
            CodedPicture {
                nodes,
                plain,
                w: pe.w,
                h: pe.h,
            },
            pe.stats,
        ))
    }

    /// PH NAL + the P slice NAL around `slice_data`.
    fn p_access_unit(&self, poc: i32, slice_data: &[u8], alf: Option<&AlfPhChain>) -> Vec<u8> {
        let mut au = Vec::new();
        annex_b_wrap(&self.vvc.emit_inter_picture_header_nal(poc as u32), &mut au);
        let mut slice_rbsp = self
            .vvc
            .emit_p_slice_header_bytes_with_alf(self.cfg.qp - 26, alf);
        slice_rbsp.extend_from_slice(slice_data);
        annex_b_wrap(
            &build_nal(NalUnitType::TrailNut, 0, 1, &slice_rbsp),
            &mut au,
        );
        au
    }

    /// Commit a P picture: decode the access unit (the decoded picture
    /// is deblocked, nothing else), and — when the effort asks for it —
    /// decide the adaptive loop filter against that picture, emit the
    /// access unit again with the filter signalled and decode that one
    /// as the reference.
    ///
    /// The second access unit replaces the first in the stream; the
    /// crate's decoder is handed both, which it takes as two pictures
    /// with the same picture order count: the second one resolves its
    /// reference (the previous picture) exactly as the first did and
    /// displaces the first from the reference pictures.
    fn commit_p(
        &mut self,
        au: Vec<u8>,
        src: &PictureBuffer,
        poc: i32,
        coded: &CodedPicture,
        stats: &mut InterStats,
    ) -> Result<Vec<u8>> {
        let deblocked = self.decode_au(&au)?;
        let p = self.cfg.picture;
        let mut out = au;
        let mut frame = deblocked;
        if self.cfg.alf && EffortParams::of(self.cfg.effort).alf {
            let bd = u32::from(p.bit_depth);
            let cf = u32::from(p.chroma_format_idc);
            let mut filtered = frame.clone();
            let alf_pic = crate::alf_enc::alf_decide_and_apply_clipped(
                src,
                &mut filtered,
                CTB_SIZE.trailing_zeros(),
                bd,
                cf,
                None,
            );
            let mut on = 0u32;
            for ry in 0..alf_pic.pic_height_in_ctbs_y {
                for rx in 0..alf_pic.pic_width_in_ctbs_y {
                    on += u32::from(alf_pic.get(rx, ry).luma_on);
                }
            }
            if on > 0 {
                // Fixed filter sets only: no APS, luma only.
                let chain = AlfPhChain {
                    num_alf_aps_ids_luma: 0,
                    ph_alf_cb_enabled_flag: false,
                    ph_alf_cr_enabled_flag: false,
                    ph_alf_cc_cb_enabled_flag: false,
                    ph_alf_cc_cr_enabled_flag: false,
                    ..AlfPhChain::default_full()
                };
                let syntax = AlfSyntaxConfig {
                    alf_enabled: true,
                    cb_enabled: false,
                    cr_enabled: false,
                    cc_cb_enabled: false,
                    cc_cr_enabled: false,
                    sh_num_alf_aps_ids_luma: 0,
                    alf_chroma_num_alt_filters_minus1: 0,
                    alf_cc_cb_filters_signalled_minus1: 0,
                    alf_cc_cr_filters_signalled_minus1: 0,
                    chroma_format_idc: cf,
                    slice_type: SliceType::P,
                    sh_cabac_init_flag: false,
                };
                let slice_data = emit_slice_data(&self.cfg, coded, Some((&alf_pic, &syntax)))?;
                let au2 = self.p_access_unit(poc, &slice_data, Some(&chain));
                frame = self.decode_au(&au2)?;
                out = au2;
                stats.alf_ctbs = on;
            }
        }
        self.append_hash(&mut out, &frame);
        self.reference = Some(Arc::new(frame));
        Ok(out)
    }
}

/// A decided P picture: its coding trees and the slice data without
/// the adaptive loop filter.
struct CodedPicture {
    /// `(x, y, tree)` of every CTB in coding order.
    nodes: Vec<(usize, usize, Node)>,
    plain: Vec<u8>,
    w: usize,
    h: usize,
}

// ---------------------------------------------------------------------
// P picture coding
// ---------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CuKind {
    Skip { merge_idx: u32 },
    Merge { merge_idx: u32 },
    Amvp { mvp_flag: u32, mvd: MotionVector },
}

#[derive(Clone, Debug)]
struct CuChoice {
    kind: CuKind,
    mvf: MvField,
    /// Quantised levels; empty = not coded (CBF 0).
    y: Vec<i32>,
    cb: Vec<i32>,
    cr: Vec<i32>,
}

#[derive(Clone, Debug)]
enum Node {
    Leaf(CuChoice),
    /// Quadrants in z-order; `None` lies outside the coded picture.
    Split(Box<[Option<Node>; 4]>),
}

/// Parse-order neighbour state the `ctxInc` derivations read.
#[derive(Clone, Copy, Debug)]
struct CuCell {
    cb_w: u32,
    cb_h: u32,
    cqt_depth: u32,
    skip: bool,
}

struct CuGrid {
    w4: usize,
    h4: usize,
    cells: Vec<Option<CuCell>>,
}

impl CuGrid {
    fn new(w: usize, h: usize) -> Self {
        let (w4, h4) = (w.div_ceil(4), h.div_ceil(4));
        Self {
            w4,
            h4,
            cells: vec![None; w4 * h4],
        }
    }

    fn get(&self, x: i64, y: i64) -> Option<CuCell> {
        if x < 0 || y < 0 {
            return None;
        }
        let (cx, cy) = (x as usize / 4, y as usize / 4);
        if cx >= self.w4 || cy >= self.h4 {
            return None;
        }
        self.cells[cy * self.w4 + cx]
    }

    fn insert(&mut self, x: usize, y: usize, size: usize, cell: CuCell) {
        for cy in y / 4..((y + size) / 4).min(self.h4) {
            for cx in x / 4..((x + size) / 4).min(self.w4) {
                self.cells[cy * self.w4 + cx] = Some(cell);
            }
        }
    }

    fn tree_neighbours(&self, x: usize, y: usize) -> TreeNeighbours {
        let left = self.get(x as i64 - 1, y as i64);
        let above = self.get(x as i64, y as i64 - 1);
        TreeNeighbours {
            left_avail: left.is_some(),
            above_avail: above.is_some(),
            cb_height_left: left.map(|d| d.cb_h).unwrap_or(0),
            cb_width_above: above.map(|d| d.cb_w).unwrap_or(0),
            cqt_depth_left: left.map(|d| d.cqt_depth).unwrap_or(0),
            cqt_depth_above: above.map(|d| d.cqt_depth).unwrap_or(0),
        }
    }
}

/// Fractional-bit estimates of the CU-level flags (the residual and
/// the vector difference are measured, these are not).
mod flag_bits {
    pub const SKIP_ON: f64 = 0.6;
    pub const SKIP_OFF: f64 = 1.6;
    pub const PRED_MODE: f64 = 0.2;
    pub const MERGE_ON: f64 = 1.0;
    pub const MERGE_OFF: f64 = 1.0;
    pub const MVP_FLAG: f64 = 1.0;
    pub const CODED_ON: f64 = 0.7;
    pub const CODED_OFF: f64 = 1.4;
    pub const SPLIT: f64 = 1.0;
}

fn merge_idx_bits(idx: u32) -> f64 {
    // bin 0 context coded, the rest of the truncated-unary bypass.
    if idx == 0 {
        0.6
    } else {
        1.4 + f64::from(idx.min(MAX_NUM_MERGE_CAND - 2))
    }
}

/// Bits of one `mvd_coding()` component of raw magnitude `v`
/// (greater0 / greater1 flags, the EG1 remainder, the sign).
fn mvd_component_bits(v: i32) -> f64 {
    let a = v.unsigned_abs();
    match a {
        0 => 1.0,
        1 => 3.0,
        _ => {
            // abs_mvd_minus2 in EG1: 2 * floor(log2(a - 2 + 2)) + 1 bins.
            let k = 31 - a.leading_zeros();
            3.0 + f64::from(2 * k + 1)
        }
    }
}

fn mvd_bits(mv: MotionVector, mvp: MotionVector) -> f64 {
    mvd_component_bits((mv.x - mvp.x) >> 2) + mvd_component_bits((mv.y - mvp.y) >> 2)
}

fn any_nonzero(levels: &[i32]) -> bool {
    levels.iter().any(|&l| l != 0)
}

/// The §7.3.11.10 `transform_unit()` of an inter CU that has one
/// (`cu_coded_flag` = 1, explicit or inferred): chroma CBFs, the luma
/// CBF when a chroma CBF is set (inferred 1 otherwise), `cu_qp_delta`
/// when the quantization group has not coded one, then the residuals.
#[allow(clippy::too_many_arguments)]
fn emit_inter_tu(
    enc: &mut ArithEncoder,
    rctx: &mut ResidualCtxs,
    size: usize,
    c_w: usize,
    c_h: usize,
    chroma: bool,
    cu: &CuChoice,
    qp_delta_pending: &mut bool,
) -> Result<()> {
    let cbf_y = any_nonzero(&cu.y);
    let cbf_cb = chroma && any_nonzero(&cu.cb);
    let cbf_cr = chroma && any_nonzero(&cu.cr);
    if chroma {
        write_tu_cb_coded_flag(enc, rctx, cbf_cb, false)?;
        write_tu_cr_coded_flag(enc, rctx, cbf_cr, false, cbf_cb)?;
    }
    if cbf_cb || cbf_cr {
        write_tu_y_coded_flag(enc, rctx, cbf_y, false, false, false)?;
    } else if !cbf_y {
        return Err(Error::invalid(
            "h266 video encoder: a transform unit without any coded block",
        ));
    }
    if *qp_delta_pending {
        write_cu_qp_delta(enc, rctx, 0)?;
        *qp_delta_pending = false;
    }
    let rc = RcOpts::default();
    if cbf_y {
        encode_tb_coefficients_opts(enc, rctx, size, size, 0, &cu.y, rc)?;
    }
    if cbf_cb {
        encode_tb_coefficients_opts(enc, rctx, c_w, c_h, 1, &cu.cb, rc)?;
    }
    if cbf_cr {
        encode_tb_coefficients_opts(enc, rctx, c_w, c_h, 2, &cu.cr, rc)?;
    }
    Ok(())
}

struct PictureCoder<'a> {
    src: &'a PictureBuffer,
    reference: &'a PictureBuffer,
    w: usize,
    h: usize,
    bd: u32,
    chroma: bool,
    sub_w: usize,
    sub_h: usize,
    qp: i32,
    qp_c: i32,
    lambda: f64,
    sqrt_lambda: f64,
    params: EffortParams,
    mf: MotionField,
    hmvp: HmvpTable,
    /// The collocated picture (§8.5.2.11) and the current
    /// `PicOrderCntVal`.
    col: ReferencePicture,
    poc: i32,
    /// Prediction scratch at the picture geometry (the interpolation
    /// writes a block at its picture position).
    pred: PictureBuffer,
    /// Contexts residual bit estimates start from.
    est_ctxs: ResidualCtxs,
    stats: InterStats,
}

/// A coded candidate of one CU: its rate-distortion cost and choice.
struct Trial {
    cost: f64,
    choice: CuChoice,
    /// A skip CU, or a CU whose residual quantised to nothing.
    residual_free: bool,
}

impl<'a> PictureCoder<'a> {
    fn new(
        cfg: &VideoEncoderConfig,
        src: &'a PictureBuffer,
        reference: &'a PictureBuffer,
        col: ReferencePicture,
        poc: i32,
    ) -> Self {
        let p = &cfg.picture;
        let (sw, sh) = p.chroma_subsampling();
        let w = p.coded_width() as usize;
        let h = p.coded_height() as usize;
        let bd = u32::from(p.bit_depth);
        let lambda = lambda_for_qp(cfg.qp);
        Self {
            src,
            reference,
            w,
            h,
            bd,
            chroma: p.chroma_format_idc != 0,
            sub_w: sw as usize,
            sub_h: sh as usize,
            qp: cfg.qp,
            qp_c: chroma_qp_identity_bd(cfg.qp, bd),
            lambda,
            sqrt_lambda: lambda.sqrt(),
            params: EffortParams::of(cfg.effort),
            mf: MotionField::new(w as u32, h as u32),
            hmvp: HmvpTable::new(),
            col,
            poc,
            pred: reference.clone(),
            est_ctxs: ResidualCtxs::init_with_init_type(cfg.qp, 1),
            stats: InterStats::default(),
        }
    }

    /// Decide every CTU; returns `(x, y, tree)` per CTB in coding
    /// order.
    fn decide_picture(&mut self) -> Result<Vec<(usize, usize, Node)>> {
        let mut nodes = Vec::new();
        for ctb_y in (0..self.h).step_by(CTB_SIZE) {
            // §7.3.11.1 — the history table empties at the first CTB
            // of every CTB row.
            self.hmvp.reset();
            for ctb_x in (0..self.w).step_by(CTB_SIZE) {
                let (node, _) = self.decide(ctb_x, ctb_y, CTB_SIZE, MotionVector::ZERO)?;
                if let Some(node) = node {
                    nodes.push((ctb_x, ctb_y, node));
                }
            }
        }
        Ok(nodes)
    }

    /// The slice data of the decided picture (see [`emit_trees`]).
    fn emit_picture(
        &self,
        nodes: &[(usize, usize, Node)],
        alf: Option<(&AlfPicture, &AlfSyntaxConfig)>,
    ) -> Result<Vec<u8>> {
        emit_trees(
            nodes,
            self.w,
            self.h,
            self.sub_w,
            self.sub_h,
            self.chroma,
            self.qp,
            alf,
        )
    }

    // ---- decision ---------------------------------------------------

    /// Decide the coding tree node at `(x, y)`; the motion field and
    /// the history table are left as the decoder will have them after
    /// the node. Returns `None` for a node outside the picture.
    fn decide(
        &mut self,
        x: usize,
        y: usize,
        size: usize,
        hint: MotionVector,
    ) -> Result<(Option<Node>, f64)> {
        if x >= self.w || y >= self.h {
            return Ok((None, 0.0));
        }
        let in_pic = x + size <= self.w && y + size <= self.h;
        let mut leaf: Option<Trial> = None;
        if in_pic {
            leaf = Some(self.best_leaf(x, y, size, hint)?);
        }
        let try_split = match &leaf {
            None => true,
            Some(t) => {
                size > self.params.min_cu
                    && (size > MAX_RESIDUAL_CU
                        || !t.residual_free
                        || self.params.split_below_skip)
                    // A 128x128 skip CU that predicts exactly is final.
                    && !(size > MAX_RESIDUAL_CU && t.cost <= self.lambda * 2.0)
            }
        };
        if !try_split {
            let t = leaf.expect("a node that is not split has a leaf");
            self.apply(x, y, size, &t.choice);
            return Ok((Some(Node::Leaf(t.choice)), t.cost));
        }
        // Snapshot what the quadrants will overwrite.
        let hmvp_before = self.hmvp.clone();
        let mf_before = self.snapshot_mf(x, y, size);
        let stats_before = self.stats;
        let child_hint = leaf.as_ref().map(|t| t.choice.mvf.mv_l0).unwrap_or(hint);
        let half = size / 2;
        let mut children: [Option<Node>; 4] = [None, None, None, None];
        let mut split_cost = self.lambda * flag_bits::SPLIT;
        for (q, child) in children.iter_mut().enumerate() {
            let (qx, qy) = (x + (q % 2) * half, y + (q / 2) * half);
            let (node, cost) = self.decide(qx, qy, half, child_hint)?;
            *child = node;
            split_cost += cost;
        }
        if let Some(t) = leaf {
            if t.cost <= split_cost {
                self.hmvp = hmvp_before;
                self.restore_mf(x, y, size, &mf_before);
                self.stats = stats_before;
                self.apply(x, y, size, &t.choice);
                return Ok((Some(Node::Leaf(t.choice)), t.cost));
            }
        }
        Ok((Some(Node::Split(Box::new(children))), split_cost))
    }

    fn snapshot_mf(&self, x: usize, y: usize, size: usize) -> Vec<MvField> {
        let bw = self.mf.blocks_w as usize;
        let (x0, y0) = (x / 4, y / 4);
        let x1 = ((x + size) / 4).min(bw);
        let y1 = ((y + size) / 4).min(self.mf.blocks_h as usize);
        let mut out = Vec::with_capacity((x1 - x0) * (y1 - y0));
        for by in y0..y1 {
            out.extend_from_slice(&self.mf.field[by * bw + x0..by * bw + x1]);
        }
        out
    }

    fn restore_mf(&mut self, x: usize, y: usize, size: usize, snap: &[MvField]) {
        let bw = self.mf.blocks_w as usize;
        let (x0, y0) = (x / 4, y / 4);
        let x1 = ((x + size) / 4).min(bw);
        let y1 = ((y + size) / 4).min(self.mf.blocks_h as usize);
        let n = x1 - x0;
        for (row, by) in (y0..y1).enumerate() {
            self.mf.field[by * bw + x0..by * bw + x1]
                .copy_from_slice(&snap[row * n..(row + 1) * n]);
        }
    }

    /// Commit a leaf: the motion field cells and the §8.5.2.16 history
    /// update the decoder performs after the CU.
    fn apply(&mut self, x: usize, y: usize, size: usize, choice: &CuChoice) {
        self.mf
            .write_block(x as u32, y as u32, size as u32, size as u32, choice.mvf);
        // Log2ParMrgLevel = 2: every CU completes its merge region.
        self.hmvp.update_with(choice.mvf);
        match choice.kind {
            CuKind::Skip { .. } => self.stats.skip_cus += 1,
            CuKind::Merge { .. } => self.stats.merge_cus += 1,
            CuKind::Amvp { .. } => self.stats.amvp_cus += 1,
        }
    }

    fn uni_field(mv: MotionVector, skip: bool) -> MvField {
        MvField {
            mv_l0: mv,
            ref_idx_l0: 0,
            pred_flag_l0: true,
            mv_l1: MotionVector::ZERO,
            ref_idx_l1: -1,
            pred_flag_l1: false,
            cu_skip_flag: skip,
            mode_inter: true,
            available: true,
            bcw_idx: 0,
            hpel_if_idx: 0,
        }
    }

    /// The best leaf CU of the node: skip, merge + residual, or an
    /// explicit vector (with or without residual).
    fn best_leaf(&mut self, x: usize, y: usize, size: usize, hint: MotionVector) -> Result<Trial> {
        let (xi, yi, si) = (x as i32, y as i32, size as i32);
        let spatial = derive_spatial_merge_candidates(xi, yi, si, si, &self.mf, LOG2_PAR_MRG_LEVEL);
        // §8.5.2.11 — the temporal candidate (refIdxL0Col = 0), from
        // the bottom-right or the centre collocated block.
        let col = derive_temporal_merge_candidate(&TemporalMergeInputs {
            xcb: xi,
            ycb: yi,
            cb_w: si,
            cb_h: si,
            pic_w: self.w as i32,
            pic_h: self.h as i32,
            ctb_log2_size_y: CTB_SIZE.trailing_zeros(),
            current_poc: self.poc,
            current_ref_poc: self.poc - 1,
            col_pic: &self.col,
            col_ref_poc: self.poc,
            x: 0,
            no_backward_pred: true,
            collocated_from_l0: true,
        });
        let mlist = build_merge_cand_list(&spatial, MAX_NUM_MERGE_CAND, col, Some(&self.hmvp), 1);

        // Rank the distinct merge vectors by luma SAD + index bits.
        let mut best_merge: Option<(u32, f64)> = None;
        for (i, cand) in mlist.iter().enumerate() {
            if !cand.pred_flag_l0 || cand.ref_idx_l0 != 0 || cand.pred_flag_l1 {
                continue;
            }
            if mlist[..i]
                .iter()
                .any(|c| c.pred_flag_l0 && c.mv_l0 == cand.mv_l0)
            {
                continue;
            }
            let sad = self.luma_sad(x, y, size, cand.mv_l0)?;
            let cost = sad as f64 + self.sqrt_lambda * merge_idx_bits(i as u32);
            if best_merge.is_none_or(|(_, c)| cost < c) {
                best_merge = Some((i as u32, cost));
            }
        }
        let (merge_idx, _) = best_merge.ok_or_else(|| {
            Error::invalid("h266 video encoder: merge list without a P candidate")
        })?;
        let merge_mv = mlist[merge_idx as usize].mv_l0;
        let idx_bits = merge_idx_bits(merge_idx);

        // Skip: the prediction is the reconstruction.
        let (py, pcb, pcr) = self.predict_cu(x, y, size, merge_mv)?;
        let skip_sse = self.pred_sse(x, y, size, &py, &pcb, &pcr);
        let mut skip_field = mlist[merge_idx as usize];
        skip_field.cu_skip_flag = true;
        let mut best = Trial {
            cost: skip_sse as f64 + self.lambda * (flag_bits::SKIP_ON + idx_bits),
            choice: CuChoice {
                kind: CuKind::Skip { merge_idx },
                mvf: skip_field,
                y: Vec::new(),
                cb: Vec::new(),
                cr: Vec::new(),
            },
            residual_free: true,
        };
        // A skip CU that already costs less than the flags of any
        // other mode is final.
        if size > MAX_RESIDUAL_CU || best.cost <= self.lambda * 4.0 {
            return Ok(best);
        }

        // Merge + residual.
        let (ly, lcb, lcr, sse) = self.code_residual(x, y, size, &py, &pcb, &pcr)?;
        let merge_has_residual = any_nonzero(&ly) || any_nonzero(&lcb) || any_nonzero(&lcr);
        if merge_has_residual {
            let mut field = mlist[merge_idx as usize];
            field.cu_skip_flag = false;
            let choice = CuChoice {
                kind: CuKind::Merge { merge_idx },
                mvf: field,
                y: ly,
                cb: lcb,
                cr: lcr,
            };
            let bits = flag_bits::SKIP_OFF
                + flag_bits::PRED_MODE
                + flag_bits::MERGE_ON
                + idx_bits
                + self.tu_bits(size, &choice)?;
            let cost = sse as f64 + self.lambda * bits;
            if cost < best.cost {
                best = Trial {
                    cost,
                    choice,
                    residual_free: false,
                };
            }
        } else {
            // The merge prediction already quantises to nothing: an
            // explicit vector cannot pay for its bits.
            return Ok(best);
        }

        // Explicit vector.
        let mvp = self.mvp_list(xi, yi, si);
        let mut seeds: Vec<MotionVector> = vec![MotionVector::ZERO, hint, mvp[0], mvp[1]];
        seeds.extend(mlist.iter().filter(|c| c.pred_flag_l0).map(|c| c.mv_l0));
        let mv = self.motion_search(x, y, size, &seeds, &mvp)?;
        if mv != merge_mv {
            let mvp_flag = u32::from(mvd_bits(mv, mvp[1]) < mvd_bits(mv, mvp[0]));
            let pred_mv = mvp[mvp_flag as usize];
            let mvd = MotionVector {
                x: (mv.x - pred_mv.x) >> 2,
                y: (mv.y - pred_mv.y) >> 2,
            };
            let (py, pcb, pcr) = self.predict_cu(x, y, size, mv)?;
            let (ly, lcb, lcr, sse) = self.code_residual(x, y, size, &py, &pcb, &pcr)?;
            let coded = any_nonzero(&ly) || any_nonzero(&lcb) || any_nonzero(&lcr);
            let head_bits = flag_bits::SKIP_OFF
                + flag_bits::PRED_MODE
                + flag_bits::MERGE_OFF
                + mvd_bits(mv, pred_mv)
                + flag_bits::MVP_FLAG;
            let choice = CuChoice {
                kind: CuKind::Amvp { mvp_flag, mvd },
                mvf: Self::uni_field(mv, false),
                y: if coded { ly } else { Vec::new() },
                cb: if coded { lcb } else { Vec::new() },
                cr: if coded { lcr } else { Vec::new() },
            };
            let bits = if coded {
                head_bits + flag_bits::CODED_ON + self.tu_bits(size, &choice)?
            } else {
                head_bits + flag_bits::CODED_OFF
            };
            let cost = sse as f64 + self.lambda * bits;
            if cost < best.cost {
                best = Trial {
                    cost,
                    choice,
                    residual_free: !coded,
                };
            }
        }
        Ok(best)
    }

    /// §8.5.2.8 `mvpListL0` for the reference picture (refIdx 0), as
    /// the decoder builds it for a CU without AMVR.
    fn mvp_list(&self, x: i32, y: i32, size: i32) -> [MotionVector; MAX_MVP_CAND] {
        let amvr = AmvrShift(2);
        // One reference picture: every neighbour that predicts from
        // list 0 refers to it. The POC values only need to be
        // consistent (current = 1, reference = 0).
        let ref_poc = self.poc - 1;
        let poc_l0 = |idx: i32| -> Option<i32> { (idx == 0).then_some(ref_poc) };
        let poc_l1 = |_: i32| -> Option<i32> { None };
        let ctx = AmvpRefContext {
            list: RefList::L0,
            current_ref_poc: ref_poc,
            poc_of_l0_ref: &poc_l0,
            poc_of_l1_ref: &poc_l1,
        };
        let raw = derive_spatial_mvp_candidates(x, y, size, size, &self.mf, &ctx);
        let spatial = [
            SpatialMvpCandidate {
                available: raw[0].available,
                mv: round_mv_amvr(raw[0].mv, amvr),
            },
            SpatialMvpCandidate {
                available: raw[1].available,
                mv: round_mv_amvr(raw[1].mv, amvr),
            },
        ];
        let n_spatial = if spatial[0].available {
            1 + usize::from(spatial[1].available && spatial[0].mv != spatial[1].mv)
        } else {
            usize::from(spatial[1].available)
        };
        // §8.5.2.11 — the temporal candidate, consulted unless both
        // spatial candidates are present and differ (§8.5.2.9 step 3).
        let col = derive_temporal_amvp_candidate(
            &AmvpTemporalInputs {
                xcb: x,
                ycb: y,
                cb_w: size,
                cb_h: size,
                pic_w: self.w as i32,
                pic_h: self.h as i32,
                ctb_log2_size_y: CTB_SIZE.trailing_zeros(),
                current_poc: self.poc,
                current_ref_poc: ref_poc,
                col_ref_poc: self.poc,
                ph_temporal_mvp_enabled: true,
                x: 0,
                no_backward_pred: true,
                collocated_from_l0: true,
            },
            &self.col,
            amvr,
        );
        let col_suppressed =
            spatial[0].available && spatial[1].available && spatial[0].mv != spatial[1].mv;
        let n_pre_hmvp = (n_spatial + usize::from(col.is_some() && !col_suppressed)).min(2);
        let hmvp = derive_hmvp_mvp_candidates(&self.hmvp, &ctx, amvr, 2 - n_pre_hmvp);
        build_mvp_cand_list(spatial, col, &hmvp)
    }

    // ---- prediction / distortion -----------------------------------

    /// Clamp a vector so the predicted block stays within a margin of
    /// the reference picture (keeps every vector far inside the
    /// §7.4.10.10 range).
    fn clamp_mv(&self, x: usize, y: usize, size: usize, mv: MotionVector) -> MotionVector {
        const MARGIN: i32 = 16;
        let lo_x = -((x + size) as i32 + MARGIN) * 16;
        let hi_x = ((self.w - x) as i32 + MARGIN) * 16;
        let lo_y = -((y + size) as i32 + MARGIN) * 16;
        let hi_y = ((self.h - y) as i32 + MARGIN) * 16;
        MotionVector {
            x: mv.x.clamp(lo_x, hi_x),
            y: mv.y.clamp(lo_y, hi_y),
        }
    }

    /// Luma SAD of the block predicted with `mv` (any precision).
    fn luma_sad(&mut self, x: usize, y: usize, size: usize, mv: MotionVector) -> Result<u64> {
        if mv.x & 15 == 0 && mv.y & 15 == 0 {
            return Ok(self.luma_sad_int(x, y, size, mv.x >> 4, mv.y >> 4));
        }
        predict_luma_block(
            &mut self.pred.luma,
            x as u32,
            y as u32,
            size as u32,
            size as u32,
            &self.reference.luma,
            mv,
        )?;
        let (s, p) = (&self.src.luma, &self.pred.luma);
        let mut sad = 0u64;
        for row in y..y + size {
            let a = &s.samples[row * s.stride + x..row * s.stride + x + size];
            let b = &p.samples[row * p.stride + x..row * p.stride + x + size];
            sad += a
                .iter()
                .zip(b)
                .map(|(&u, &v)| u64::from(u.abs_diff(v)))
                .sum::<u64>();
        }
        Ok(sad)
    }

    /// Luma SAD against the reference displaced by whole samples (an
    /// integer vector predicts by copying, reference positions clamped
    /// to the picture as §8.5.6.3.2 does).
    fn luma_sad_int(&self, x: usize, y: usize, size: usize, dx: i32, dy: i32) -> u64 {
        let (s, r) = (&self.src.luma, &self.reference.luma);
        let (rw, rh) = (r.width as i32, r.height as i32);
        let rx0 = x as i32 + dx;
        let inside_x = rx0 >= 0 && rx0 + size as i32 <= rw;
        let mut sad = 0u64;
        for row in 0..size {
            let ry = (y as i32 + row as i32 + dy).clamp(0, rh - 1) as usize;
            let a = &s.samples[(y + row) * s.stride + x..(y + row) * s.stride + x + size];
            if inside_x {
                let b =
                    &r.samples[ry * r.stride + rx0 as usize..ry * r.stride + rx0 as usize + size];
                sad += a
                    .iter()
                    .zip(b)
                    .map(|(&u, &v)| u64::from(u.abs_diff(v)))
                    .sum::<u64>();
            } else {
                for (col, &u) in a.iter().enumerate() {
                    let rx = (rx0 + col as i32).clamp(0, rw - 1) as usize;
                    sad += u64::from(u.abs_diff(r.samples[ry * r.stride + rx]));
                }
            }
        }
        sad
    }

    /// The §8.5.6 uni-prediction of the CU: luma and both chroma
    /// blocks as the decoder forms them.
    fn predict_cu(
        &mut self,
        x: usize,
        y: usize,
        size: usize,
        mv: MotionVector,
    ) -> Result<(Vec<i16>, Vec<i16>, Vec<i16>)> {
        predict_luma_block(
            &mut self.pred.luma,
            x as u32,
            y as u32,
            size as u32,
            size as u32,
            &self.reference.luma,
            mv,
        )?;
        let grab = |p: &PicturePlane, x: usize, y: usize, w: usize, h: usize| -> Vec<i16> {
            let mut out = Vec::with_capacity(w * h);
            for row in y..y + h {
                out.extend(
                    p.samples[row * p.stride + x..row * p.stride + x + w]
                        .iter()
                        .map(|&v| v as i16),
                );
            }
            out
        };
        let py = grab(&self.pred.luma, x, y, size, size);
        if !self.chroma {
            return Ok((py, Vec::new(), Vec::new()));
        }
        let (cx, cy) = (x / self.sub_w, y / self.sub_h);
        let (cw, ch) = (size / self.sub_w, size / self.sub_h);
        predict_chroma_block(
            &mut self.pred.cb,
            cx as u32,
            cy as u32,
            cw as u32,
            ch as u32,
            &self.reference.cb,
            mv,
        )?;
        predict_chroma_block(
            &mut self.pred.cr,
            cx as u32,
            cy as u32,
            cw as u32,
            ch as u32,
            &self.reference.cr,
            mv,
        )?;
        let pcb = grab(&self.pred.cb, cx, cy, cw, ch);
        let pcr = grab(&self.pred.cr, cx, cy, cw, ch);
        Ok((py, pcb, pcr))
    }

    fn block_sse(src: &PicturePlane, x: usize, y: usize, w: usize, h: usize, pred: &[i16]) -> u64 {
        let mut sse = 0u64;
        for row in 0..h {
            let a = &src.samples[(y + row) * src.stride + x..(y + row) * src.stride + x + w];
            let b = &pred[row * w..(row + 1) * w];
            sse += a
                .iter()
                .zip(b)
                .map(|(&u, &v)| {
                    let d = i64::from(u) - i64::from(v);
                    (d * d) as u64
                })
                .sum::<u64>();
        }
        sse
    }

    /// SSE of the prediction alone over the three components.
    fn pred_sse(
        &self,
        x: usize,
        y: usize,
        size: usize,
        py: &[i16],
        pcb: &[i16],
        pcr: &[i16],
    ) -> u64 {
        let mut sse = Self::block_sse(&self.src.luma, x, y, size, size, py);
        if self.chroma {
            let (cx, cy) = (x / self.sub_w, y / self.sub_h);
            let (cw, ch) = (size / self.sub_w, size / self.sub_h);
            sse += Self::block_sse(&self.src.cb, cx, cy, cw, ch, pcb);
            sse += Self::block_sse(&self.src.cr, cx, cy, cw, ch, pcr);
        }
        sse
    }

    /// Transform, quantise (zero bin widened to [`INTER_ROUNDING`]) and
    /// reconstruct one block against `pred`: the levels and the SSE of
    /// the reconstruction `Clip1( pred + res )` (§8.7.5.1).
    #[allow(clippy::too_many_arguments)]
    fn code_block(
        &self,
        src: &PicturePlane,
        x: usize,
        y: usize,
        w: usize,
        h: usize,
        pred: &[i16],
        qp: i32,
    ) -> Result<(Vec<i32>, u64)> {
        let bd = self.bd;
        let qp_prime = qp + 6 * (bd as i32 - 8);
        let max_v = (1i32 << bd) - 1;
        let mut residual = vec![0i32; w * h];
        for row in 0..h {
            let s = &src.samples[(y + row) * src.stride + x..(y + row) * src.stride + x + w];
            let p = &pred[row * w..(row + 1) * w];
            for ((r, &sv), &pv) in residual[row * w..(row + 1) * w].iter_mut().zip(s).zip(p) {
                *r = i32::from(sv) - i32::from(pv);
            }
        }
        let pred_sse = || -> u64 {
            residual
                .iter()
                .map(|&r| (i64::from(r) * i64::from(r)) as u64)
                .sum()
        };
        // §7.3.11.11 — a 64-point transform codes the low 32.
        let coeffs = forward_dct_ii_2d_low(w, h, &residual, bd, 32, 32)?;
        let levels = quantize_tb_flat_rounding(
            &coeffs,
            w as u32,
            h as u32,
            qp_prime,
            bd,
            15,
            INTER_ROUNDING,
        )?;
        if !any_nonzero(&levels) {
            return Ok((levels, pred_sse()));
        }
        let mut params = DequantParams::luma_8bit(w as u32, h as u32, qp_prime);
        params.bit_depth = bd;
        let dq = dequantize_tb_flat(&levels, &params)?;
        let rec_res = inverse_transform_2d(w, h, w, h, TrType::DctII, TrType::DctII, &dq, bd, 15)?;
        let mut sse = 0u64;
        for row in 0..h {
            let s = &src.samples[(y + row) * src.stride + x..(y + row) * src.stride + x + w];
            for col in 0..w {
                let v = (i32::from(pred[row * w + col]) + rec_res[row * w + col]).clamp(0, max_v);
                let d = i64::from(v) - i64::from(s[col]);
                sse += (d * d) as u64;
            }
        }
        Ok((levels, sse))
    }

    /// Transform + quantise the residual of the three components
    /// against the given prediction. Returns the levels and the SSE of
    /// the reconstruction.
    #[allow(clippy::type_complexity)]
    fn code_residual(
        &self,
        x: usize,
        y: usize,
        size: usize,
        py: &[i16],
        pcb: &[i16],
        pcr: &[i16],
    ) -> Result<(Vec<i32>, Vec<i32>, Vec<i32>, u64)> {
        let (ly, mut sse) = self.code_block(&self.src.luma, x, y, size, size, py, self.qp)?;
        if !self.chroma {
            return Ok((ly, Vec::new(), Vec::new(), sse));
        }
        let (cx, cy) = (x / self.sub_w, y / self.sub_h);
        let (cw, ch) = (size / self.sub_w, size / self.sub_h);
        let (lcb, s_cb) = self.code_block(&self.src.cb, cx, cy, cw, ch, pcb, self.qp_c)?;
        let (lcr, s_cr) = self.code_block(&self.src.cr, cx, cy, cw, ch, pcr, self.qp_c)?;
        sse += s_cb + s_cr;
        Ok((ly, lcb, lcr, sse))
    }

    /// Bits of the CU's `transform_unit()` from the slice-initial
    /// residual contexts.
    fn tu_bits(&self, size: usize, cu: &CuChoice) -> Result<f64> {
        let mut enc = ArithEncoder::new();
        let mut rctx = self.est_ctxs.clone();
        let mut pending = false;
        emit_inter_tu(
            &mut enc,
            &mut rctx,
            size,
            size / self.sub_w,
            size / self.sub_h,
            self.chroma,
            cu,
            &mut pending,
        )?;
        Ok(enc.estimated_bits())
    }

    // ---- motion search ---------------------------------------------

    /// Search the quarter-sample vector of the block: the best seed,
    /// a step-halving integer pattern search, then half and quarter
    /// sample refinement.
    fn motion_search(
        &mut self,
        x: usize,
        y: usize,
        size: usize,
        seeds: &[MotionVector],
        mvp: &[MotionVector; MAX_MVP_CAND],
    ) -> Result<MotionVector> {
        let sqrt_lambda = self.sqrt_lambda;
        let rate = |mv: MotionVector| -> f64 {
            sqrt_lambda * mvd_bits(mv, mvp[0]).min(mvd_bits(mv, mvp[1]))
        };
        // Integer stage.
        let mut tried: Vec<MotionVector> = Vec::with_capacity(seeds.len());
        let mut best = MotionVector::ZERO;
        let mut best_cost = f64::INFINITY;
        for &seed in seeds {
            let int = MotionVector {
                x: (seed.x + 8) >> 4 << 4,
                y: (seed.y + 8) >> 4 << 4,
            };
            let int = self.clamp_mv(x, y, size, int);
            let int = MotionVector {
                x: int.x >> 4 << 4,
                y: int.y >> 4 << 4,
            };
            if tried.contains(&int) {
                continue;
            }
            tried.push(int);
            let cost = self.luma_sad_int(x, y, size, int.x >> 4, int.y >> 4) as f64 + rate(int);
            if cost < best_cost {
                best_cost = cost;
                best = int;
            }
        }
        const FOUR: [(i32, i32); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];
        const EIGHT: [(i32, i32); 8] = [
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1, 1),
            (-1, 1),
            (1, -1),
            (-1, -1),
        ];
        let pattern: &[(i32, i32)] = if self.params.eight_point {
            &EIGHT
        } else {
            &FOUR
        };
        let mut step = self.params.max_step;
        while step >= 1 {
            for _ in 0..self.params.max_moves {
                let centre = best;
                for &(dx, dy) in pattern {
                    let cand = MotionVector {
                        x: centre.x + dx * step * 16,
                        y: centre.y + dy * step * 16,
                    };
                    if self.clamp_mv(x, y, size, cand) != cand {
                        continue;
                    }
                    let cost =
                        self.luma_sad_int(x, y, size, cand.x >> 4, cand.y >> 4) as f64 + rate(cand);
                    if cost < best_cost {
                        best_cost = cost;
                        best = cand;
                    }
                }
                if best == centre {
                    break;
                }
            }
            step /= 2;
        }
        // Fractional stage: half (8/16) then quarter (4/16) sample.
        for (level, frac) in [(1u8, 8i32), (2, 4)] {
            if self.params.subpel < level {
                break;
            }
            let centre = best;
            for &(dx, dy) in &EIGHT {
                let cand = MotionVector {
                    x: centre.x + dx * frac,
                    y: centre.y + dy * frac,
                };
                let cost = self.luma_sad(x, y, size, cand)? as f64 + rate(cand);
                if cost < best_cost {
                    best_cost = cost;
                    best = cand;
                }
            }
        }
        Ok(best)
    }
}

// ---------------------------------------------------------------------
// Emission
// ---------------------------------------------------------------------

/// [`emit_trees`] for a [`CodedPicture`] under `cfg`.
fn emit_slice_data(
    cfg: &VideoEncoderConfig,
    coded: &CodedPicture,
    alf: Option<(&AlfPicture, &AlfSyntaxConfig)>,
) -> Result<Vec<u8>> {
    let p = &cfg.picture;
    let (sw, sh) = p.chroma_subsampling();
    emit_trees(
        &coded.nodes,
        coded.w,
        coded.h,
        sw as usize,
        sh as usize,
        p.chroma_format_idc != 0,
        cfg.qp,
        alf,
    )
}

/// §7.3.11.1 `slice_data()` of the single P slice: per CTU the
/// §7.3.11.2 ALF prefix (when the slice enables the filter) and the
/// coding tree, then `end_of_slice_one_bit`.
#[allow(clippy::too_many_arguments)]
fn emit_trees(
    nodes: &[(usize, usize, Node)],
    w: usize,
    h: usize,
    sub_w: usize,
    sub_h: usize,
    chroma: bool,
    qp: i32,
    alf: Option<(&AlfPicture, &AlfSyntaxConfig)>,
) -> Result<Vec<u8>> {
    let mut enc = ArithEncoder::new();
    let mut tree_ctxs = TreeCtxs::init_with_init_type(qp, 1);
    let mut leaf_ctxs = LeafCuCtxs::init_with_init_type(qp, 1);
    let mut alf_ctxs = AlfCtxs::init(qp);
    let mut grid = CuGrid::new(w, h);
    let constraints = SplitConstraints {
        min_qt_log2: 2,
        max_mtt_depth: 0,
        max_bt_size: 4,
        max_tt_size: 4,
        min_cb_log2: 2,
        pic_w: w as u32,
        pic_h: h as u32,
        sub_w: sub_w as u32,
        sub_h: sub_h as u32,
    };
    for (ctb_x, ctb_y, node) in nodes {
        let (ctb_x, ctb_y) = (*ctb_x, *ctb_y);
        if let Some((alf_pic, syntax)) = alf {
            // One slice, one tile: the left / above CTBs are available
            // whenever they exist.
            let (rx, ry) = ((ctb_x / CTB_SIZE) as u32, (ctb_y / CTB_SIZE) as u32);
            crate::alf_syntax::encode_alf_ctu(
                &mut enc,
                &mut alf_ctxs,
                syntax,
                alf_pic,
                rx,
                ry,
                AlfNeighbours {
                    left_avail: rx > 0,
                    up_avail: ry > 0,
                },
            )?;
        }
        let mut em = Emitter {
            leaf_ctxs: &mut leaf_ctxs,
            grid: &mut grid,
            constraints: &constraints,
            w,
            h,
            sub_w,
            sub_h,
            chroma,
            qp_delta_pending: true,
        };
        em.emit_node(&mut enc, &mut tree_ctxs, node, ctb_x, ctb_y, CTB_SIZE, 0)?;
    }
    enc.encode_terminate(1)?; // end_of_slice_one_bit
    Ok(enc.finish())
}

struct Emitter<'a> {
    leaf_ctxs: &'a mut LeafCuCtxs,
    grid: &'a mut CuGrid,
    constraints: &'a SplitConstraints,
    w: usize,
    h: usize,
    sub_w: usize,
    sub_h: usize,
    chroma: bool,
    /// `!IsCuQpDeltaCoded` of the CTU's quantization group.
    qp_delta_pending: bool,
}

impl Emitter<'_> {
    #[allow(clippy::too_many_arguments)]
    fn emit_node(
        &mut self,
        enc: &mut ArithEncoder,
        tree_ctxs: &mut TreeCtxs,
        node: &Node,
        x: usize,
        y: usize,
        size: usize,
        cqt_depth: u32,
    ) -> Result<()> {
        let nbrs = self.grid.tree_neighbours(x, y);
        let allows = self.constraints.allows(
            x as u32,
            y as u32,
            size as u32,
            size as u32,
            0,
            TreeType::SingleTree,
            None,
            0,
        );
        match node {
            Node::Leaf(cu) => {
                crate::syntax_enc::encode_coding_tree_leaf_iframe(
                    enc,
                    tree_ctxs,
                    size as u32,
                    size as u32,
                    nbrs,
                    allows,
                    |e| self.emit_cu(e, cu, x, y, size),
                )?;
                self.grid.insert(
                    x,
                    y,
                    size,
                    CuCell {
                        cb_w: size as u32,
                        cb_h: size as u32,
                        cqt_depth,
                        skip: matches!(cu.kind, CuKind::Skip { .. }),
                    },
                );
                Ok(())
            }
            Node::Split(children) => {
                if !allows.qt {
                    return Err(Error::invalid(format!(
                        "h266 video encoder: node ({x},{y}) size {size} cannot quad-split"
                    )));
                }
                let half = size / 2;
                let in_pic = x + size <= self.w && y + size <= self.h;
                let mut quadrant = |e: &mut ArithEncoder, c: &mut TreeCtxs, q: u32| -> Result<()> {
                    let q = q as usize;
                    match &children[q] {
                        Some(child) => self.emit_node(
                            e,
                            c,
                            child,
                            x + (q % 2) * half,
                            y + (q / 2) * half,
                            half,
                            cqt_depth + 1,
                        ),
                        None => Ok(()),
                    }
                };
                if in_pic {
                    crate::syntax_enc::encode_coding_quadtree_split(
                        enc,
                        tree_ctxs,
                        size as u32,
                        size as u32,
                        nbrs,
                        cqt_depth,
                        allows,
                        |e, c, q, _w, _h| quadrant(e, c, q),
                    )
                } else {
                    crate::syntax_enc::encode_coding_quadtree_implicit_split(
                        enc,
                        tree_ctxs,
                        nbrs,
                        cqt_depth,
                        allows,
                        |e, c, q| quadrant(e, c, q),
                    )
                }
            }
        }
    }

    /// §7.3.11.5 `coding_unit()` of an inter CU on a P slice
    /// (initType 1).
    fn emit_cu(
        &mut self,
        enc: &mut ArithEncoder,
        cu: &CuChoice,
        x: usize,
        y: usize,
        size: usize,
    ) -> Result<()> {
        let ctxs = &mut *self.leaf_ctxs;
        let init_type = ctxs.init_type as usize;
        // cu_skip_flag — ctxInc = the skip flags of the left and above
        // neighbours (§9.3.4.2.2), three contexts per initType.
        let left_skip = self
            .grid
            .get(x as i64 - 1, y as i64)
            .is_some_and(|c| c.skip);
        let above_skip = self
            .grid
            .get(x as i64, y as i64 - 1)
            .is_some_and(|c| c.skip);
        let inc = usize::from(left_skip) + usize::from(above_skip);
        let n = ctxs.cu_skip_flag.len() - 1;
        let skip = matches!(cu.kind, CuKind::Skip { .. });
        enc.encode_decision(
            &mut ctxs.cu_skip_flag[(init_type * 3 + inc).min(n)],
            u32::from(skip),
        )?;
        if !skip {
            // pred_mode_flag = 0 (MODE_INTER); no neighbour is intra,
            // so ctxInc = 0. Two contexts per inter initType.
            let n = ctxs.pred_mode_flag.len() - 1;
            let slot = (init_type.saturating_sub(1) * 2).min(n);
            enc.encode_decision(&mut ctxs.pred_mode_flag[slot], 0)?;
            // general_merge_flag — one context per initType.
            let n = ctxs.general_merge_flag.len() - 1;
            let merge = matches!(cu.kind, CuKind::Merge { .. });
            enc.encode_decision(
                &mut ctxs.general_merge_flag[init_type.min(n)],
                u32::from(merge),
            )?;
        }
        match cu.kind {
            CuKind::Skip { merge_idx } | CuKind::Merge { merge_idx } => {
                // merge_idx — TR, cMax = MaxNumMergeCand − 1; bin 0
                // context coded, the rest bypass.
                let c_max = MAX_NUM_MERGE_CAND - 1;
                let n = ctxs.merge_idx.len() - 1;
                enc.encode_decision(
                    &mut ctxs.merge_idx[init_type.min(n)],
                    u32::from(merge_idx > 0),
                )?;
                if merge_idx > 0 {
                    for v in 1..c_max {
                        if v < merge_idx {
                            enc.encode_bypass(1)?;
                        } else {
                            enc.encode_bypass(0)?;
                            break;
                        }
                    }
                }
            }
            CuKind::Amvp { mvp_flag, mvd } => {
                // One reference picture, no affine / SMVD / AMVR / BCW:
                // mvd_coding( x0, y0, 0, 0 ) then mvp_l0_flag.
                crate::mvd_coding_enc::encode_mvd_coding(enc, ctxs, mvd)?;
                crate::non_merge_mvp_syntax_enc::encode_mvp_lx_flag(enc, ctxs, mvp_flag)?;
                let coded = !cu.y.is_empty() || !cu.cb.is_empty() || !cu.cr.is_empty();
                let n = ctxs.cu_coded_flag.len() - 1;
                enc.encode_decision(&mut ctxs.cu_coded_flag[init_type.min(n)], u32::from(coded))?;
            }
        }
        let has_tu = match cu.kind {
            CuKind::Skip { .. } => false,
            CuKind::Merge { .. } => true,
            CuKind::Amvp { .. } => !cu.y.is_empty() || !cu.cb.is_empty() || !cu.cr.is_empty(),
        };
        if has_tu {
            emit_inter_tu(
                enc,
                &mut ctxs.residual,
                size,
                size / self.sub_w,
                size / self.sub_h,
                self.chroma,
                cu,
                &mut self.qp_delta_pending,
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn moving_source(w: usize, h: usize, t: usize) -> PictureBuffer {
        let mut pic = PictureBuffer::yuv420_filled(w, h, 0);
        for y in 0..h {
            for x in 0..w {
                let xx = x + 2 * t;
                let v = 96.0
                    + 60.0 * ((xx as f64) * 0.11).sin() * ((y as f64) * 0.07).cos()
                    + 30.0 * (((xx + y) as f64) * 0.31).sin();
                pic.luma.samples[y * pic.luma.stride + x] = v.clamp(0.0, 255.0) as u16;
            }
        }
        for y in 0..h / 2 {
            for x in 0..w / 2 {
                let xx = x + t;
                pic.cb.samples[y * pic.cb.stride + x] =
                    (128.0 + 40.0 * ((xx as f64) * 0.09).sin()) as u16;
                pic.cr.samples[y * pic.cr.stride + x] =
                    (128.0 + 40.0 * ((y as f64) * 0.13).cos()) as u16;
            }
        }
        pic
    }

    fn psnr(a: &PicturePlane, b: &PicturePlane) -> f64 {
        let mut sse = 0f64;
        for y in 0..a.height {
            for x in 0..a.width {
                let d =
                    f64::from(a.samples[y * a.stride + x]) - f64::from(b.samples[y * b.stride + x]);
                sse += d * d;
            }
        }
        if sse == 0.0 {
            return 99.0;
        }
        10.0 * (255.0f64 * 255.0 * (a.width * a.height) as f64 / sse).log10()
    }

    #[test]
    fn p_pictures_decode_and_track_the_source() {
        let (w, h) = (192usize, 136usize);
        let cfg =
            VideoEncoderConfig::new(EncoderConfig::new(w as u32, h as u32), 30).with_keyint(8);
        let mut enc = VideoEncoder::new(cfg).unwrap();
        let mut stream = Vec::new();
        let mut sources = Vec::new();
        let mut kinds = Vec::new();
        for t in 0..5 {
            let src = moving_source(w, h, t);
            let pic = enc.encode(&src).unwrap();
            kinds.push(pic.kind);
            stream.extend_from_slice(&pic.data);
            // The encoder's reference is what a fresh decoder outputs.
            let rec = enc.reconstruction().unwrap();
            assert!(psnr(&src.luma, &rec.luma) > 30.0, "picture {t} luma PSNR");
            assert!(psnr(&src.cb, &rec.cb) > 30.0, "picture {t} Cb PSNR");
            sources.push((src, rec));
        }
        assert_eq!(kinds[0], PictureKind::Idr);
        assert!(kinds[1..].iter().all(|k| *k == PictureKind::P), "{kinds:?}");
        // A fresh decoder over the whole stream reproduces every
        // reference the encoder used.
        let mut dec = StreamDecoder::new();
        let mut out = Vec::new();
        dec.decode_annex_b(&stream, &mut |p| out.push(p)).unwrap();
        assert_eq!(out.len(), sources.len());
        for (t, (pic, (_, rec))) in out.iter().zip(&sources).enumerate() {
            assert_eq!(pic.poc, t as i32);
            assert_eq!(
                pic.frame.luma.samples[..],
                rec.luma.samples[..],
                "picture {t}"
            );
            assert_eq!(pic.frame.cb.samples[..], rec.cb.samples[..], "picture {t}");
            assert_eq!(pic.frame.cr.samples[..], rec.cr.samples[..], "picture {t}");
        }
    }
}
