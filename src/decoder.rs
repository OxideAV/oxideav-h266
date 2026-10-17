//! r477 — the framework (`oxideav_core::Decoder`) face of the VVC
//! decoder: a streaming, packet-driven wrapper over
//! [`crate::stream::StreamDecoder`].
//!
//! * **Framing.** Packets are Annex B byte streams (start codes) or
//!   ISO/IEC 14496-15 §4.2.3 length-prefixed samples. With a
//!   `VvcDecoderConfigurationRecord` in `CodecParameters::extradata`
//!   ([`crate::vvcc`]) the record fixes the length-prefix size and its
//!   parameter-set arrays are pushed before the first sample; without
//!   one, every packet is sniffed (a leading start code means Annex B,
//!   otherwise a 4-byte length-prefixed walk must cover the packet
//!   exactly).
//! * **Access units.** A packet holds one or more access units; the
//!   picture a packet ends with is decoded at the end of the packet
//!   (§7.4.2.4.3 — the samples a container hands over are whole
//!   access units), so slices of one picture never need to be split
//!   across packets.
//! * **Output order.** Decoded pictures with `ph_pic_output_flag = 1`
//!   wait in a reorder set and leave in `PicOrderCntVal` order under
//!   the §C.5.2.2 bumping rules of the SPS `dpb_parameters()`: a
//!   picture is output when more than `dpb_max_num_reorder_pics`
//!   pictures wait, when one of them has waited
//!   `MaxLatencyPictures` pictures, at a CLVS boundary (every picture
//!   of the previous CVS leaves first) and on `flush`.
//! * **Timestamps.** A picture inherits the `pts` of the packet that
//!   completed it. Without one, a presentation time is derived from
//!   the POC: in `num_units_in_tick / time_scale` ticks when the SPS
//!   carries `general_timing_hrd_parameters()`, else in pictures
//!   ([`H266Decoder::output_time_base`]), monotonic across CVSs.
//! * **Labels.** The output pixel format follows the stream — 4:0:0 /
//!   4:2:0 / 4:2:2 / 4:4:4 at 8, 10, 12 or 16 bits (`Gray8` …
//!   `Yuv444P16Le`); an SPS at another bit depth is refused with a
//!   typed `Unsupported` before any picture decodes (the core
//!   `PixelFormat` set has no label for it).
//! * **Limits.** `CodecParameters::limits` and the registry's
//!   16384 × 16384 ceiling are checked against every SPS / PPS before
//!   the parameter set is accepted — no plane is allocated for a
//!   hostile header; every malformed input returns an error, never a
//!   panic.

use std::collections::VecDeque;

use oxideav_core::Decoder;
use oxideav_core::{
    CodecId, CodecParameters, DecoderLimits, Error, Frame, Packet, PixelFormat, Rational, Result,
    TimeBase, VideoFrame, VideoPlane,
};

use crate::nal::{extract_rbsp, iter_annex_b, iter_length_prefixed, NalHeader, NalUnitType};
use crate::reconstruct::PicturePlane;
use crate::sps::{parse_sps, SeqParameterSet};
use crate::stream::{DecodedPicture, OutputInfo, StreamDecoder};
use crate::vvcc::VvcDecoderConfigurationRecord;
use crate::CODEC_ID_STR;

/// Largest picture the registry decoder accepts, per side (the
/// `CodecCapabilities::max_size` it declares).
pub const MAX_DIMENSION: u32 = 16384;

/// Build the registry decoder (`CodecInfo::decoder`).
pub fn make_decoder(params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    Ok(Box::new(H266Decoder::new(params)?))
}

/// The pixel-format label of a `chroma_format_idc` / bit-depth pair,
/// or a typed refusal when the core format set has none.
pub fn output_pixel_format_for(chroma_format_idc: u8, bit_depth: u32) -> Result<PixelFormat> {
    use PixelFormat as P;
    let pf = match (chroma_format_idc, bit_depth) {
        (0, 8) => P::Gray8,
        (0, 10) => P::Gray10Le,
        (0, 12) => P::Gray12Le,
        (0, 16) => P::Gray16Le,
        (1, 8) => P::Yuv420P,
        (1, 10) => P::Yuv420P10Le,
        (1, 12) => P::Yuv420P12Le,
        (1, 16) => P::Yuv420P16Le,
        (2, 8) => P::Yuv422P,
        (2, 10) => P::Yuv422P10Le,
        (2, 12) => P::Yuv422P12Le,
        (2, 16) => P::Yuv422P16Le,
        (3, 8) => P::Yuv444P,
        (3, 10) => P::Yuv444P10Le,
        (3, 12) => P::Yuv444P12Le,
        (3, 16) => P::Yuv444P16Le,
        (cf @ 0..=3, bd) => {
            return Err(Error::unsupported(format!(
                "h266: {bd}-bit {} output has no pixel-format label (8 / 10 / 12 / 16-bit are \
                 labelled)",
                chroma_format_name(cf)
            )))
        }
        (cf, _) => {
            return Err(Error::invalid(format!(
                "h266: sps_chroma_format_idc {cf} is out of range"
            )))
        }
    };
    Ok(pf)
}

fn chroma_format_name(cf: u8) -> &'static str {
    match cf {
        0 => "4:0:0",
        1 => "4:2:0",
        2 => "4:2:2",
        _ => "4:4:4",
    }
}

/// How the packets are framed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Framing {
    /// Sniff every packet: start code → Annex B, else 4-byte lengths.
    Auto,
    /// `vvcC`-declared length prefixes of this many bytes.
    LengthPrefixed(u8),
}

/// A decoded picture waiting for its output turn.
struct Pending {
    cvs: u32,
    poc: i32,
    /// Packet `pts` the picture inherited, if any.
    packet_pts: Option<i64>,
    frame: VideoFrame,
    /// §C.5.2.2 `PicLatencyCount`.
    latency: u32,
}

/// The registry decoder. Build it with [`make_decoder`] /
/// [`H266Decoder::new`]; drive it through the [`Decoder`] trait.
pub struct H266Decoder {
    codec_id: CodecId,
    inner: StreamDecoder,
    framing: Framing,
    limits: DecoderLimits,
    /// Pictures decoded but not yet output (reorder set).
    pending: Vec<Pending>,
    /// Frames ready for `receive_frame`, in output order.
    ready: VecDeque<VideoFrame>,
    /// Bumping bounds of the current CVS.
    info: OutputInfo,
    /// `pts` of the packet being decoded, consumed by the first
    /// picture it completes.
    packet_pts: Option<i64>,
    /// Synthetic presentation clock: the next pts to hand out and the
    /// `(cvs, poc)` anchor it maps to.
    clock_next: i64,
    clock_anchor: Option<(u32, i32, i64)>,
    last_pts: Option<i64>,
    /// The label of the stream's output.
    output_pf: Option<PixelFormat>,
    output_dims: Option<(u32, u32)>,
    time_base: Option<TimeBase>,
    eos: bool,
}

impl H266Decoder {
    /// Build from stream parameters: `extradata` (optional) is an
    /// ISO/IEC 14496-15 `VvcDecoderConfigurationRecord`; `limits`
    /// caps the pictures the decoder will allocate.
    pub fn new(params: &CodecParameters) -> Result<Self> {
        let mut dec = Self {
            codec_id: CodecId::new(CODEC_ID_STR),
            inner: StreamDecoder::new(),
            framing: Framing::Auto,
            limits: params.limits,
            pending: Vec::new(),
            ready: VecDeque::new(),
            info: OutputInfo::default(),
            packet_pts: None,
            clock_next: 0,
            clock_anchor: None,
            last_pts: None,
            output_pf: None,
            output_dims: None,
            time_base: None,
            eos: false,
        };
        let side_cap = u64::from(MAX_DIMENSION) * u64::from(MAX_DIMENSION);
        dec.inner
            .set_max_pixels(Some(params.limits.max_pixels_per_frame.min(side_cap)));
        if !params.extradata.is_empty() {
            let rec = VvcDecoderConfigurationRecord::parse(&params.extradata)?;
            dec.framing = Framing::LengthPrefixed(rec.length_size());
            if let Some(p) = &rec.ptl {
                // Label the output from the record before any SPS
                // (a container may ask before the first sample).
                dec.output_pf = Some(output_pixel_format_for(
                    p.chroma_format_idc,
                    u32::from(p.bit_depth_minus8) + 8,
                )?);
                dec.output_dims = Some((
                    u32::from(p.max_picture_width),
                    u32::from(p.max_picture_height),
                ));
            }
            let mut sink = |_: DecodedPicture| {};
            for nal in rec.nal_units() {
                dec.push_nal(nal, &mut sink)?;
            }
            dec.inner.finish_picture(&mut sink)?;
        }
        Ok(dec)
    }

    /// The time base the synthetic presentation timestamps are in
    /// (pictures without a packet `pts`): `num_units_in_tick /
    /// time_scale` once an SPS with timing information has been seen,
    /// `1/1` ("one tick per POC step") otherwise.
    pub fn output_time_base(&self) -> TimeBase {
        self.time_base.unwrap_or(TimeBase::new(1, 1))
    }

    /// `time_scale / num_units_in_tick` of the active SPS, when it
    /// signals timing (`None` otherwise).
    pub fn output_frame_rate(&self) -> Option<Rational> {
        self.time_base.map(|tb| Rational::new(tb.den(), tb.num()))
    }

    /// The cropped output picture size, once known (from the `vvcC`
    /// record or the first SPS).
    pub fn output_dimensions(&self) -> Option<(u32, u32)> {
        self.output_dims
    }

    /// Pre-flight an SPS before the stream decoder stores it: the
    /// output must be labellable and the declared maxima must fit the
    /// limits (pixels and bytes) — refused before any allocation.
    fn vet_sps(&mut self, sps: &SeqParameterSet) -> Result<()> {
        let pf = output_pixel_format_for(sps.sps_chroma_format_idc, sps.bit_depth_y())?;
        let w = u64::from(sps.sps_pic_width_max_in_luma_samples);
        let h = u64::from(sps.sps_pic_height_max_in_luma_samples);
        if w > u64::from(MAX_DIMENSION) || h > u64::from(MAX_DIMENSION) {
            return Err(Error::resource_exhausted(format!(
                "h266: SPS maximum picture {w}x{h} exceeds the {MAX_DIMENSION}x{MAX_DIMENSION} \
                 decoder ceiling"
            )));
        }
        let area = w.saturating_mul(h);
        if area > self.limits.max_pixels_per_frame {
            return Err(Error::resource_exhausted(format!(
                "h266: SPS maximum picture {w}x{h} ({area} samples) exceeds the {}-sample limit",
                self.limits.max_pixels_per_frame
            )));
        }
        // Reconstruction planes are u16; chroma adds 2 / (SubWidthC ·
        // SubHeightC) of the luma area. The DPB keeps up to
        // max_dec_pic_buffering copies — bound the single picture
        // here (the per-frame cap) and the DPB fan-out by the
        // same cap times the declared buffering.
        let chroma_factor_x2 = match sps.sps_chroma_format_idc {
            0 => 0u64,
            1 => 1,
            2 => 2,
            _ => 4,
        };
        let bytes = area.saturating_mul(2).saturating_mul(2 + chroma_factor_x2) / 2;
        if bytes > self.limits.max_alloc_bytes_per_frame {
            return Err(Error::resource_exhausted(format!(
                "h266: a {w}x{h} {} picture needs {bytes} bytes of reconstruction, above the \
                 {}-byte per-frame limit",
                chroma_format_name(sps.sps_chroma_format_idc),
                self.limits.max_alloc_bytes_per_frame
            )));
        }
        self.output_pf = Some(pf);
        self.output_dims = Some((sps.cropped_width(), sps.cropped_height()));
        if let Some(t) = sps.general_timing_hrd.as_ref() {
            if t.num_units_in_tick > 0 && t.time_scale > 0 {
                let tb = TimeBase::new(i64::from(t.num_units_in_tick), i64::from(t.time_scale));
                if self.time_base != Some(tb) {
                    // A new tick length: keep the clock monotonic by
                    // re-anchoring at the next tick.
                    self.clock_anchor = None;
                    self.time_base = Some(tb);
                }
            }
        }
        Ok(())
    }

    /// Push one NAL unit (header + payload) through the vetting layer
    /// into the stream decoder.
    fn push_nal(&mut self, raw: &[u8], sink: &mut dyn FnMut(DecodedPicture)) -> Result<()> {
        let header = NalHeader::parse(raw)?;
        if header.nal_unit_type == NalUnitType::SpsNut {
            let sps = parse_sps(&extract_rbsp(&raw[2..]))?;
            self.vet_sps(&sps)?;
        }
        self.inner.push_nal(raw, sink)
    }

    /// Decode every NAL unit of `data` under the configured framing.
    fn decode_packet_data(&mut self, data: &[u8]) -> Result<()> {
        let mut out: Vec<DecodedPicture> = Vec::new();
        let result = (|| -> Result<()> {
            let mut sink = |pic: DecodedPicture| out.push(pic);
            match self.framing {
                Framing::LengthPrefixed(n) => {
                    let nals = iter_length_prefixed(data, n)?;
                    let raws: Vec<&[u8]> = nals.iter().map(|n| n.raw).collect();
                    for raw in raws {
                        self.push_nal(raw, &mut sink)?;
                    }
                }
                Framing::Auto => {
                    if starts_with_start_code(data) {
                        let raws: Vec<&[u8]> = iter_annex_b(data).map(|n| n.raw).collect();
                        for raw in raws {
                            self.push_nal(raw, &mut sink)?;
                        }
                    } else {
                        let nals = iter_length_prefixed(data, 4).map_err(|e| {
                            Error::invalid(format!(
                                "h266: packet is neither an Annex B byte stream nor 4-byte \
                                 length-prefixed sample data ({e})"
                            ))
                        })?;
                        let raws: Vec<&[u8]> = nals.iter().map(|n| n.raw).collect();
                        for raw in raws {
                            self.push_nal(raw, &mut sink)?;
                        }
                    }
                }
            }
            self.inner.finish_picture(&mut sink)
        })();
        // Pictures decoded before an error still go out (the error
        // itself is reported to the caller).
        for pic in out {
            self.admit(pic)?;
        }
        result
    }

    /// Take a decoded picture into the reorder set and run the
    /// bumping rules.
    fn admit(&mut self, pic: DecodedPicture) -> Result<()> {
        if let Some(info) = self.inner.last_output_info() {
            self.info = info;
        }
        // CLVS boundary: everything of the previous CVS leaves first.
        if let Some(prev) = self.pending.first().map(|p| p.cvs) {
            if prev != pic.cvs_idx {
                self.bump_all();
            }
        }
        if !pic.output_flag {
            return Ok(());
        }
        let frame = picture_to_video_frame(&pic);
        for p in &mut self.pending {
            p.latency += 1;
        }
        self.pending.push(Pending {
            cvs: pic.cvs_idx,
            poc: pic.poc,
            packet_pts: self.packet_pts.take(),
            frame,
            latency: 0,
        });
        self.bump_by_rules();
        Ok(())
    }

    /// §C.5.2.2 — output while more pictures wait than the SPS allows
    /// to be reordered, or one has waited `MaxLatencyPictures`.
    fn bump_by_rules(&mut self) {
        let reorder = self.info.max_num_reorder_pics as usize;
        let max_latency = if self.info.max_latency_increase_plus1 != 0 {
            Some(self.info.max_num_reorder_pics + self.info.max_latency_increase_plus1 - 1)
        } else {
            None
        };
        loop {
            let over_reorder = self.pending.len() > reorder;
            let over_latency = max_latency
                .map(|m| self.pending.iter().any(|p| p.latency >= m))
                .unwrap_or(false);
            if !(over_reorder || over_latency) || self.pending.is_empty() {
                break;
            }
            self.bump_one();
        }
    }

    /// Output the smallest-POC waiting picture.
    fn bump_one(&mut self) {
        let Some(idx) = self
            .pending
            .iter()
            .enumerate()
            .min_by_key(|(_, p)| p.poc)
            .map(|(i, _)| i)
        else {
            return;
        };
        let p = self.pending.remove(idx);
        let mut frame = p.frame;
        frame.pts = Some(self.stamp(p.cvs, p.poc, p.packet_pts));
        self.ready.push_back(frame);
    }

    /// Output every waiting picture in POC order.
    fn bump_all(&mut self) {
        while !self.pending.is_empty() {
            self.bump_one();
        }
    }

    /// The presentation timestamp of an output picture: the packet's
    /// when it had one, else the POC-derived clock (monotonic, one
    /// tick per POC step within a CVS, one tick across CVS joins).
    fn stamp(&mut self, cvs: u32, poc: i32, packet_pts: Option<i64>) -> i64 {
        let synthetic = match self.clock_anchor {
            Some((a_cvs, a_poc, a_pts)) if a_cvs == cvs => {
                a_pts.saturating_add(i64::from(poc).saturating_sub(i64::from(a_poc)))
            }
            _ => {
                self.clock_anchor = Some((cvs, poc, self.clock_next));
                self.clock_next
            }
        };
        // Monotonic even for non-conforming POC order.
        let synthetic = match self.last_pts {
            Some(last) if synthetic <= last && packet_pts.is_none() => last + 1,
            _ => synthetic,
        };
        let pts = packet_pts.unwrap_or(synthetic);
        self.last_pts = Some(pts);
        self.clock_next = synthetic.saturating_add(1);
        pts
    }
}

fn starts_with_start_code(data: &[u8]) -> bool {
    data.starts_with(&[0, 0, 1]) || data.starts_with(&[0, 0, 0, 1])
}

/// Crop a reconstruction to its conformance window and pack it as a
/// [`VideoFrame`]: 8-bit streams as one byte per sample, deeper
/// streams as little-endian 16-bit words (the `*Le` labels).
fn picture_to_video_frame(pic: &DecodedPicture) -> VideoFrame {
    let (cx, cy, cw, ch) = pic.crop;
    let wide = pic.bit_depth > 8;
    let (sw, sh) = crate::reconstruct::chroma_subsampling(u32::from(pic.chroma_format_idc));
    let mut planes = Vec::with_capacity(3);
    planes.push(pack_plane(&pic.frame.luma, cx, cy, cw, ch, wide));
    if pic.chroma_format_idc != 0 {
        planes.push(pack_plane(
            &pic.frame.cb,
            cx / sw,
            cy / sh,
            cw / sw,
            ch / sh,
            wide,
        ));
        planes.push(pack_plane(
            &pic.frame.cr,
            cx / sw,
            cy / sh,
            cw / sw,
            ch / sh,
            wide,
        ));
    }
    VideoFrame { pts: None, planes }
}

fn pack_plane(
    p: &PicturePlane,
    x0: usize,
    y0: usize,
    w: usize,
    h: usize,
    wide: bool,
) -> VideoPlane {
    let w = w.min(p.width.saturating_sub(x0));
    let h = h.min(p.height.saturating_sub(y0));
    let bps = if wide { 2 } else { 1 };
    let stride = w * bps;
    let mut data = Vec::with_capacity(stride * h);
    for y in 0..h {
        let row = &p.samples[(y0 + y) * p.stride + x0..(y0 + y) * p.stride + x0 + w];
        if wide {
            for &v in row {
                data.extend_from_slice(&v.to_le_bytes());
            }
        } else {
            data.extend(row.iter().map(|&v| v.min(255) as u8));
        }
    }
    VideoPlane { stride, data }
}

impl Decoder for H266Decoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn send_packet(&mut self, packet: &Packet) -> Result<()> {
        self.eos = false;
        self.packet_pts = packet.pts;
        let r = self.decode_packet_data(&packet.data);
        self.packet_pts = None;
        r
    }

    fn receive_frame(&mut self) -> Result<Frame> {
        if let Some(f) = self.ready.pop_front() {
            return Ok(Frame::Video(f));
        }
        if self.eos {
            Err(Error::Eof)
        } else {
            Err(Error::NeedMore)
        }
    }

    fn flush(&mut self) -> Result<()> {
        let mut out: Vec<DecodedPicture> = Vec::new();
        let r = self
            .inner
            .end_of_sequence(&mut |pic: DecodedPicture| out.push(pic));
        for pic in out {
            self.admit(pic)?;
        }
        self.bump_all();
        self.eos = true;
        r
    }

    fn reset(&mut self) -> Result<()> {
        self.inner.reset_pictures();
        self.pending.clear();
        self.ready.clear();
        self.packet_pts = None;
        self.clock_anchor = None;
        self.eos = false;
        Ok(())
    }

    fn output_pixel_format(&self) -> Option<PixelFormat> {
        self.output_pf
    }
}

/// Historical name of the registry decoder type (the 0.0.x
/// placeholder that rejected every packet). The real decoder lives
/// under [`H266Decoder`]; this alias keeps the name resolvable.
#[deprecated(since = "0.0.11", note = "use `H266Decoder`")]
pub type FoundationDecoder = H266Decoder;

/// Split an Annex B elementary stream into access units (§7.4.2.4.3,
/// single-layer form): a new access unit starts at the first AUD /
/// OPI / DCI / VPS / SPS / PPS / prefix APS / PH / prefix SEI /
/// reserved-26 / unspecified-28..29 NAL unit that follows a VCL NAL
/// unit, and at a VCL NAL unit with `sh_picture_header_in_slice_header_flag`
/// that follows another VCL NAL unit. Each returned slice begins with
/// its start code. Trailing EOS / EOB units stay with the access unit
/// they close.
pub fn split_access_units(annex_b: &[u8]) -> Vec<&[u8]> {
    let base = annex_b.as_ptr() as usize;
    let mut starts: Vec<usize> = Vec::new();
    let mut seen_vcl = false;
    for nal in iter_annex_b(annex_b) {
        let off = nal.raw.as_ptr() as usize - base;
        // The start code precedes `raw`: 4 bytes when a zero byte
        // sits before the 3-byte code.
        let sc = if off >= 4 && annex_b[off - 4] == 0 {
            4
        } else {
            3
        };
        let begin = off - sc;
        if starts.is_empty() {
            starts.push(begin);
        }
        let t = nal.header.nal_unit_type;
        let t_u8 = t.as_u8();
        let starts_au = matches!(
            t,
            NalUnitType::AudNut
                | NalUnitType::OpiNut
                | NalUnitType::DciNut
                | NalUnitType::VpsNut
                | NalUnitType::SpsNut
                | NalUnitType::PpsNut
                | NalUnitType::PrefixApsNut
                | NalUnitType::PhNut
                | NalUnitType::PrefixSeiNut
        ) || t_u8 == 26
            || t_u8 == 28
            || t_u8 == 29;
        if t.is_vcl() {
            let ph_in_sh = nal
                .payload()
                .first()
                .map(|b| b & 0x80 != 0)
                .unwrap_or(false);
            if seen_vcl && ph_in_sh && starts.last() != Some(&begin) {
                starts.push(begin);
            }
            seen_vcl = true;
        } else if starts_au && seen_vcl {
            starts.push(begin);
            seen_vcl = false;
        }
    }
    let mut out = Vec::with_capacity(starts.len());
    for (i, &s) in starts.iter().enumerate() {
        let e = starts.get(i + 1).copied().unwrap_or(annex_b.len());
        out.push(&annex_b[s..e]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reconstruct::PictureBuffer;

    fn source(w: usize, h: usize, cf: u32, bd: u32) -> PictureBuffer {
        let mid = 1u16 << (bd - 1);
        let mut src = PictureBuffer::with_chroma_format_bd(w, h, cf, mid, bd);
        let sh = bd - 8;
        for y in 0..h {
            for x in 0..w {
                src.luma.samples[y * w + x] = ((((x * 7 + y * 13) % 200) + 20) << sh) as u16;
            }
        }
        for y in 0..src.cb.height {
            for x in 0..src.cb.width {
                src.cb.samples[y * src.cb.width + x] = (((x * 3 + 90) % 256) << sh) as u16;
                src.cr.samples[y * src.cb.width + x] = (((y * 5 + 120) % 256) << sh) as u16;
            }
        }
        src
    }

    fn encode(src: &PictureBuffer, cf: u32) -> (Vec<u8>, PictureBuffer) {
        let mut cfg =
            crate::encoder::EncoderConfig::new(src.luma.width as u32, src.luma.height as u32);
        cfg.chroma_format_idc = cf as u8;
        crate::encoder_pipeline::encode_idr_with_residuals_cfg(src, 28, cfg).expect("encode")
    }

    fn drain(dec: &mut dyn Decoder) -> Vec<VideoFrame> {
        let mut out = Vec::new();
        loop {
            match dec.receive_frame() {
                Ok(Frame::Video(v)) => out.push(v),
                Ok(_) => panic!("non-video frame"),
                Err(Error::NeedMore) | Err(Error::Eof) => break,
                Err(e) => panic!("receive_frame: {e}"),
            }
        }
        out
    }

    fn plane_bytes(p: &PicturePlane, wide: bool) -> Vec<u8> {
        let mut v = Vec::new();
        for y in 0..p.height {
            for &s in &p.samples[y * p.stride..y * p.stride + p.width] {
                if wide {
                    v.extend_from_slice(&s.to_le_bytes());
                } else {
                    v.push(s as u8);
                }
            }
        }
        v
    }

    #[test]
    fn annex_b_packet_decodes_to_the_direct_reconstruction() {
        let src = source(40, 24, 1, 8);
        let (bs, rec) = encode(&src, 1);
        let params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
        let mut dec = make_decoder(&params).unwrap();
        dec.send_packet(&Packet::new(0, TimeBase::new(1, 25), bs.clone()).with_pts(7))
            .unwrap();
        dec.flush().unwrap();
        let frames = drain(dec.as_mut());
        assert_eq!(frames.len(), 1);
        let f = &frames[0];
        assert_eq!(f.pts, Some(7));
        assert_eq!(f.image_plane_count(), 3);
        assert_eq!(f.planes[0].stride, 40);
        assert_eq!(f.planes[0].data, plane_bytes(&rec.luma, false));
        assert_eq!(f.planes[1].data, plane_bytes(&rec.cb, false));
        assert_eq!(f.planes[2].data, plane_bytes(&rec.cr, false));
        assert_eq!(dec.output_pixel_format(), Some(PixelFormat::Yuv420P));
        assert!(matches!(dec.receive_frame(), Err(Error::Eof)));
    }

    #[test]
    fn vvcc_framed_packet_decodes_identically() {
        let src = source(33, 17, 3, 10);
        let (bs, rec) = encode(&src, 3);
        let record = VvcDecoderConfigurationRecord::from_annex_b(&bs).unwrap();
        let mut params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
        params.extradata = record.to_bytes().unwrap();
        let mut dec = H266Decoder::new(&params).unwrap();
        assert_eq!(dec.output_pixel_format(), Some(PixelFormat::Yuv444P10Le));
        // The record's SPS array labels and sizes the output before
        // the first sample: the cropped 33x17, not the coded 40x24.
        assert_eq!(dec.output_dimensions(), Some((33, 17)));
        let sample = crate::vvcc::annex_b_to_length_prefixed(&bs, 4, true).unwrap();
        dec.send_packet(&Packet::new(0, TimeBase::new(1, 1), sample))
            .unwrap();
        dec.flush().unwrap();
        let frames = drain(&mut dec);
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].planes[0].data, plane_bytes(&rec.luma, true));
        assert_eq!(frames[0].planes[1].data, plane_bytes(&rec.cb, true));
        // Synthetic pts: the first picture sits at tick 0.
        assert_eq!(frames[0].pts, Some(0));
        assert_eq!(dec.output_time_base(), TimeBase::new(1, 1));
    }

    #[test]
    fn monochrome_is_labelled_grey_with_one_plane() {
        let src = source(16, 16, 0, 8);
        let (bs, rec) = encode(&src, 0);
        let params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
        let mut dec = H266Decoder::new(&params).unwrap();
        dec.send_packet(&Packet::new(0, TimeBase::new(1, 1), bs))
            .unwrap();
        assert_eq!(dec.output_pixel_format(), Some(PixelFormat::Gray8));
        dec.flush().unwrap();
        let frames = drain(&mut dec);
        assert_eq!(frames[0].image_plane_count(), 1);
        assert_eq!(frames[0].planes[0].data, plane_bytes(&rec.luma, false));
    }

    #[test]
    fn limits_refuse_before_allocation() {
        let src = source(64, 64, 1, 8);
        let (bs, _) = encode(&src, 1);
        let mut params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
        params.limits = DecoderLimits::default().with_max_pixels_per_frame(1000);
        let mut dec = H266Decoder::new(&params).unwrap();
        let err = dec
            .send_packet(&Packet::new(0, TimeBase::new(1, 1), bs))
            .unwrap_err();
        assert!(err.is_resource_exhausted(), "{err}");
        assert!(drain(&mut dec).is_empty());
    }

    #[test]
    fn hostile_packets_never_panic() {
        let src = source(24, 16, 1, 8);
        let (bs, _) = encode(&src, 1);
        let params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
        // Truncations, byte flips and garbage all return.
        for cut in (0..bs.len()).step_by(7) {
            let mut dec = H266Decoder::new(&params).unwrap();
            let _ = dec.send_packet(&Packet::new(0, TimeBase::new(1, 1), bs[..cut].to_vec()));
            let _ = dec.flush();
            let _ = drain(&mut dec);
        }
        for i in (0..bs.len()).step_by(5) {
            let mut bad = bs.clone();
            bad[i] ^= 0x5A;
            let mut dec = H266Decoder::new(&params).unwrap();
            let _ = dec.send_packet(&Packet::new(0, TimeBase::new(1, 1), bad));
            let _ = dec.flush();
            let _ = drain(&mut dec);
        }
        let mut dec = H266Decoder::new(&params).unwrap();
        assert!(dec
            .send_packet(&Packet::new(0, TimeBase::new(1, 1), vec![0x12, 0x34, 0x56]))
            .is_err());
        assert!(dec
            .send_packet(&Packet::new(0, TimeBase::new(1, 1), Vec::new()))
            .is_ok());
        // A hostile record.
        let mut params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
        params.extradata = vec![0xFF; 40];
        assert!(H266Decoder::new(&params).is_err());
    }

    #[test]
    fn two_pictures_in_two_packets_keep_their_pts_and_reset_restarts() {
        let src = source(32, 16, 1, 8);
        let (bs, rec) = encode(&src, 1);
        let params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
        let mut dec = H266Decoder::new(&params).unwrap();
        dec.send_packet(&Packet::new(0, TimeBase::new(1, 1), bs.clone()).with_pts(100))
            .unwrap();
        dec.send_packet(&Packet::new(0, TimeBase::new(1, 1), bs.clone()).with_pts(200))
            .unwrap();
        let frames = drain(&mut dec);
        // Intra-only, no reorder → both pictures are out already.
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0].pts, Some(100));
        assert_eq!(frames[1].pts, Some(200));
        assert_eq!(frames[1].planes[0].data, plane_bytes(&rec.luma, false));
        dec.reset().unwrap();
        dec.send_packet(&Packet::new(0, TimeBase::new(1, 1), bs))
            .unwrap();
        dec.flush().unwrap();
        let frames = drain(&mut dec);
        assert_eq!(frames.len(), 1);
        assert!(matches!(dec.receive_frame(), Err(Error::Eof)));
    }

    #[test]
    fn access_unit_splitter_finds_every_picture() {
        let src = source(24, 24, 1, 8);
        let (bs, _) = encode(&src, 1);
        let mut two = bs.clone();
        two.extend_from_slice(&bs);
        let aus = split_access_units(&two);
        assert_eq!(aus.len(), 2);
        assert_eq!(aus[0], &bs[..]);
        assert_eq!(aus[1], &bs[..]);
        assert!(split_access_units(&[]).is_empty());
    }

    #[test]
    fn unlabelled_bit_depths_are_typed_refusals() {
        assert!(output_pixel_format_for(1, 9).is_err());
        assert!(output_pixel_format_for(1, 14).is_err());
        assert_eq!(
            output_pixel_format_for(2, 12).unwrap(),
            PixelFormat::Yuv422P12Le
        );
        assert_eq!(
            output_pixel_format_for(0, 16).unwrap(),
            PixelFormat::Gray16Le
        );
        assert!(output_pixel_format_for(4, 8).is_err());
    }
}
