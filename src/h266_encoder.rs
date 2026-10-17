//! r477 — the VVC stills encoder exposed through the
//! `oxideav_core::Encoder` trait so pipelines and muxers can drive it
//! from [`CodecParameters`].
//!
//! Every input frame is coded as one IDR access unit by
//! [`crate::encoder_pipeline::encode_idr_with_residuals_cfg`] (the
//! intra encoder: intra decision + CU-size RDO, deblocking, SAO, ALF,
//! optional MTT / dependent quantisation / sign hiding / LMCS / tiles
//! / WPP / palette, lossless transform-skip coding), so the stream is
//! all-intra: one packet out per frame in, `pts == dts`, every packet
//! a keyframe, `flush` has nothing to drain.
//!
//! Layouts: `Gray8` / `Gray10Le` (coded as 4:0:0), `Yuv420P` /
//! `Yuv420P10Le`, `Yuv422P` / `Yuv422P10Le`, `Yuv444P` /
//! `Yuv444P10Le` — the depth of the input is the coded depth
//! (`sps_bitdepth_minus8`). 12-bit input is a typed refusal (the
//! coding loop runs at 8 – 10 bits); the registry descriptor lists
//! the accepted set so the gateway ladder converts ahead of us.
//!
//! Quality: `qp` (0..=63, default 30) or `quality` (framework 0..=100;
//! 100 = lossless when the layout admits it, 0..=99 → QP 51..=2 on a
//! straight line), `lossless=true` directly.
//!
//! Framing: `format=annexb` (default — start codes, in-band VPS /
//! SPS / PPS on every access unit, empty `extradata`) or
//! `format=vvcc` (4-byte length-prefixed samples without the DCI / VPS
//! / SPS / PPS units; `output_params().extradata` holds the ISO/IEC
//! 14496-15 §11.2.4 `VvcDecoderConfigurationRecord` built from the
//! session's parameter sets — the `vvc1` sample-entry / HEIF `vvcC`
//! shape).

use std::collections::VecDeque;

use oxideav_core::options::{
    parse_options, CodecOptionsStruct, OptionField, OptionKind, OptionValue,
};
use oxideav_core::{
    CodecId, CodecParameters, Encoder as CoreEncoder, Error, Frame, Packet, PixelFormat, Result,
    TimeBase, VideoPlane,
};

use crate::encoder::EncoderConfig;
use crate::encoder_pipeline::encode_idr_with_residuals_cfg;
use crate::nal::{iter_annex_b, NalUnitType};
use crate::reconstruct::PictureBuffer;
use crate::vvcc::{annex_b_to_length_prefixed, VvcDecoderConfigurationRecord};

/// Typed option struct for the registry encoder. See the module docs
/// for the semantics of each knob.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct H266EncoderOptions {
    /// Slice QP (0..=63). Default 30. Ignored under `lossless` or a
    /// `quality` of 100.
    pub qp: u32,
    /// Framework quality 0..=100 (`-1` = unset, use `qp`): 100 is
    /// lossless, 0..=99 map to QP 51..=2.
    pub quality: i32,
    /// Lossless coding (transform-skip residuals at Qp′ =
    /// QpPrimeTsMin; the reconstruction equals the source).
    pub lossless: bool,
    /// Packet framing: `annexb` or `vvcc`.
    pub format: String,
    /// CU-size rate-distortion decision (default true).
    pub rdo: bool,
    /// MTT binary + ternary split pickers (default false; excluded
    /// under `lossless`).
    pub mtt: bool,
    /// Dependent quantisation (default false; excludes
    /// `sign_data_hiding` and `lossless`).
    pub dep_quant: bool,
    /// Sign data hiding (default false; excludes `dep_quant` and
    /// `lossless`).
    pub sign_data_hiding: bool,
    /// Tile columns (>= 1).
    pub tile_columns: u32,
    /// Tile rows (>= 1).
    pub tile_rows: u32,
    /// Wavefront parallel processing entry points (default false).
    pub wpp: bool,
    /// Palette mode for screen content (8-bit 4:2:0 only).
    pub palette: bool,
    /// Matrix-based intra prediction candidates in the intra decision
    /// (default true).
    pub mip: bool,
    /// Intra sub-partitions in the intra decision (default false —
    /// conformant but a net loss on textured content with the current
    /// pre-filter decision; see `EncoderConfig::intra_isp`).
    pub isp: bool,
    /// Block-based delta pulse code modulation in the intra decision
    /// (default true).
    pub bdpcm: bool,
}

impl H266EncoderOptions {
    /// The defaults (see each field).
    pub fn new() -> Self {
        Self::default()
    }
}

impl Default for H266EncoderOptions {
    fn default() -> Self {
        Self {
            qp: 30,
            quality: -1,
            lossless: false,
            format: "annexb".to_string(),
            rdo: true,
            mtt: false,
            dep_quant: false,
            sign_data_hiding: false,
            tile_columns: 1,
            tile_rows: 1,
            wpp: false,
            palette: false,
            mip: true,
            isp: false,
            bdpcm: true,
        }
    }
}

impl CodecOptionsStruct for H266EncoderOptions {
    const SCHEMA: &'static [OptionField] = &[
        OptionField {
            name: "qp",
            kind: OptionKind::U32,
            default: OptionValue::U32(30),
            help: "slice QP (0..=63); ignored under lossless / quality=100",
        },
        OptionField {
            name: "quality",
            kind: OptionKind::I32,
            default: OptionValue::I32(-1),
            help: "framework quality 0..=100 (-1: use qp); 100 = lossless, 0..=99 -> QP 51..=2",
        },
        OptionField {
            name: "lossless",
            kind: OptionKind::Bool,
            default: OptionValue::Bool(false),
            help: "lossless coding (transform-skip residuals at Qp' = QpPrimeTsMin)",
        },
        OptionField {
            name: "format",
            kind: OptionKind::Enum(&["annexb", "vvcc"]),
            default: OptionValue::String(String::new()),
            help: "packet framing: annexb (start codes, in-band VPS/SPS/PPS) or vvcc (4-byte length prefixes, extradata = VvcDecoderConfigurationRecord)",
        },
        OptionField {
            name: "rdo",
            kind: OptionKind::Bool,
            default: OptionValue::Bool(true),
            help: "CU-size rate-distortion decision (64x64 node vs explicit quad split)",
        },
        OptionField {
            name: "mtt",
            kind: OptionKind::Bool,
            default: OptionValue::Bool(false),
            help: "MTT binary + ternary split pickers (not under lossless)",
        },
        OptionField {
            name: "dep_quant",
            kind: OptionKind::Bool,
            default: OptionValue::Bool(false),
            help: "dependent quantisation (excludes sign_data_hiding and lossless)",
        },
        OptionField {
            name: "sign_data_hiding",
            kind: OptionKind::Bool,
            default: OptionValue::Bool(false),
            help: "sign data hiding (excludes dep_quant and lossless)",
        },
        OptionField {
            name: "tile_columns",
            kind: OptionKind::U32,
            default: OptionValue::U32(1),
            help: "tile columns (>= 1; every column at least one 128-sample CTB wide)",
        },
        OptionField {
            name: "tile_rows",
            kind: OptionKind::U32,
            default: OptionValue::U32(1),
            help: "tile rows (>= 1; every row at least one 128-sample CTB tall)",
        },
        OptionField {
            name: "wpp",
            kind: OptionKind::Bool,
            default: OptionValue::Bool(false),
            help: "wavefront parallel processing (per-CTU-row entry points)",
        },
        OptionField {
            name: "palette",
            kind: OptionKind::Bool,
            default: OptionValue::Bool(false),
            help: "palette mode for screen content (8-bit 4:2:0 only; not with mtt)",
        },
        OptionField {
            name: "mip",
            kind: OptionKind::Bool,
            default: OptionValue::Bool(true),
            help: "matrix-based intra prediction candidates in the intra decision",
        },
        OptionField {
            name: "isp",
            kind: OptionKind::Bool,
            default: OptionValue::Bool(false),
            help: "intra sub-partition candidates in the intra decision (opt-in: conformant, a net loss on textured content with the pre-filter decision)",
        },
        OptionField {
            name: "bdpcm",
            kind: OptionKind::Bool,
            default: OptionValue::Bool(true),
            help: "block-based delta pulse code modulation candidates in the intra decision",
        },
    ];

    fn apply(&mut self, key: &str, value: &OptionValue) -> Result<()> {
        match key {
            "qp" => self.qp = value.as_u32()?,
            "quality" => self.quality = value.as_i32()?,
            "lossless" => self.lossless = value.as_bool()?,
            "format" => self.format = value.as_str()?.to_string(),
            "rdo" => self.rdo = value.as_bool()?,
            "mtt" => self.mtt = value.as_bool()?,
            "dep_quant" => self.dep_quant = value.as_bool()?,
            "sign_data_hiding" => self.sign_data_hiding = value.as_bool()?,
            "tile_columns" => self.tile_columns = value.as_u32()?,
            "tile_rows" => self.tile_rows = value.as_u32()?,
            "wpp" => self.wpp = value.as_bool()?,
            "palette" => self.palette = value.as_bool()?,
            "mip" => self.mip = value.as_bool()?,
            "isp" => self.isp = value.as_bool()?,
            "bdpcm" => self.bdpcm = value.as_bool()?,
            _ => unreachable!("guarded by SCHEMA"),
        }
        Ok(())
    }
}

/// Pixel formats the encoder accepts as input, in the order the
/// registry descriptor lists them. Every layout is coded natively
/// (grey as 4:0:0) at the input's bit depth.
pub fn accepted_pixel_formats() -> &'static [PixelFormat] {
    &[
        PixelFormat::Yuv420P,
        PixelFormat::Yuv422P,
        PixelFormat::Yuv444P,
        PixelFormat::Gray8,
        PixelFormat::Yuv420P10Le,
        PixelFormat::Yuv422P10Le,
        PixelFormat::Yuv444P10Le,
        PixelFormat::Gray10Le,
    ]
}

/// Map the framework `quality` (0..=100) to a slice QP: 100 is
/// lossless (`None`), 0..=99 run QP 51 → 2 on a straight line.
pub fn quality_to_qp(quality: u32) -> Option<i32> {
    if quality >= 100 {
        return None;
    }
    // qp = 51 − q · 49 / 99, rounded to nearest.
    let q = i64::from(quality);
    Some((51 - (q * 49 + 49) / 99) as i32)
}

/// Input sample layout derived from the pixel format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Layout {
    chroma_format_idc: u32,
    bit_depth: u32,
}

impl Layout {
    fn of(pf: PixelFormat) -> Result<Self> {
        use PixelFormat as P;
        let (cf, bd) = match pf {
            P::Gray8 => (0, 8),
            P::Gray10Le => (0, 10),
            P::Yuv420P => (1, 8),
            P::Yuv420P10Le => (1, 10),
            P::Yuv422P => (2, 8),
            P::Yuv422P10Le => (2, 10),
            P::Yuv444P => (3, 8),
            P::Yuv444P10Le => (3, 10),
            P::Gray12Le
            | P::Yuv420P12Le
            | P::Yuv422P12Le
            | P::Yuv444P12Le
            | P::Gray16Le
            | P::Yuv420P16Le
            | P::Yuv422P16Le
            | P::Yuv444P16Le => {
                return Err(Error::unsupported(format!(
                    "h266 encoder: {pf:?} input is deeper than the 8 – 10-bit coding loop (feed \
                     an 8- or 10-bit layout)"
                )))
            }
            other => {
                return Err(Error::unsupported(format!(
                    "h266 encoder input must be one of {:?}, got {other:?}",
                    accepted_pixel_formats()
                )))
            }
        };
        Ok(Self {
            chroma_format_idc: cf,
            bit_depth: bd,
        })
    }

    fn wide(self) -> bool {
        self.bit_depth > 8
    }

    fn grey(self) -> bool {
        self.chroma_format_idc == 0
    }

    fn subsampling(self) -> (usize, usize) {
        crate::reconstruct::chroma_subsampling(self.chroma_format_idc)
    }
}

/// The registry encoder.
pub struct H266CodecEncoder {
    codec_id: CodecId,
    output_params: CodecParameters,
    config: EncoderConfig,
    layout: Layout,
    qp: i32,
    time_base: TimeBase,
    width: usize,
    height: usize,
    frame_index: i64,
    /// `format=vvcc`: strip the parameter sets, length-prefix the rest.
    vvcc: bool,
    queue: VecDeque<Packet>,
}

impl H266CodecEncoder {
    /// Build an encoder from parameters + option bag. Errors on
    /// missing / invalid geometry, unknown options, a pixel format
    /// outside [`accepted_pixel_formats`] (12 / 16-bit layouts are a
    /// typed `Unsupported`), odd dimensions along a subsampled axis,
    /// and option combinations the coding loop excludes.
    pub fn new(params: &CodecParameters) -> Result<Self> {
        let opts: H266EncoderOptions = parse_options(&params.options)?;
        let width = params
            .width
            .ok_or_else(|| Error::invalid("h266 encoder requires width"))?;
        let height = params
            .height
            .ok_or_else(|| Error::invalid("h266 encoder requires height"))?;
        if width == 0 || height == 0 {
            return Err(Error::invalid("h266 encoder requires non-zero dimensions"));
        }
        let pf = params.pixel_format.unwrap_or(PixelFormat::Yuv420P);
        let layout = Layout::of(pf)?;
        let (sw, sh) = layout.subsampling();
        if width as usize % sw != 0 || height as usize % sh != 0 {
            return Err(Error::invalid(format!(
                "h266 encoder: {pf:?} input needs dimensions that are multiples of the chroma \
                 subsampling ({sw}x{sh}), got {width}x{height} — the §7.4.3.4 conformance window \
                 is in chroma units"
            )));
        }
        if opts.qp > 63 {
            return Err(Error::invalid(format!("qp {} out of 0..=63", opts.qp)));
        }
        if opts.quality > 100 {
            return Err(Error::invalid(format!(
                "quality {} out of 0..=100",
                opts.quality
            )));
        }
        let vvcc = match opts.format.as_str() {
            "" | "annexb" => false,
            "vvcc" => true,
            other => return Err(Error::invalid(format!("unknown format '{other}'"))),
        };
        // Quality → QP / lossless.
        let (lossless, qp) = if opts.lossless {
            (true, 0)
        } else if opts.quality >= 0 {
            match quality_to_qp(opts.quality as u32) {
                None => (true, 0),
                Some(qp) => (false, qp),
            }
        } else {
            (false, opts.qp as i32)
        };

        let mut config = EncoderConfig::new(width, height);
        config.bit_depth = layout.bit_depth as u8;
        config.chroma_format_idc = layout.chroma_format_idc as u8;
        config.lossless = lossless;
        config.rdo_cu_split = opts.rdo;
        config.enable_mtt_bt_picker = opts.mtt;
        config.enable_mtt_tt_picker = opts.mtt;
        config.dep_quant = opts.dep_quant;
        config.sign_data_hiding = opts.sign_data_hiding;
        if opts.tile_columns == 0 || opts.tile_rows == 0 {
            return Err(Error::invalid("tile_columns / tile_rows must be >= 1"));
        }
        if opts.tile_columns > 255 || opts.tile_rows > 255 {
            return Err(Error::invalid("tile_columns / tile_rows must be <= 255"));
        }
        config.tile_columns = opts.tile_columns as u8;
        config.tile_rows = opts.tile_rows as u8;
        config.wpp = opts.wpp;
        config.palette = opts.palette;
        config.intra_mip = opts.mip;
        config.intra_isp = opts.isp;
        config.intra_bdpcm = opts.bdpcm;
        // Validate up front (geometry, tile grid, tool exclusions).
        let probe = crate::encoder::VvcEncoder::new(config)?;

        let (fps_num, fps_den) = match params.frame_rate {
            Some(r) if r.num > 0 && r.den > 0 => (r.num, r.den),
            _ => (25, 1),
        };
        let time_base = TimeBase::new(fps_den, fps_num);

        let mut output_params = CodecParameters::video(CodecId::new(crate::CODEC_ID_STR));
        output_params.width = Some(width);
        output_params.height = Some(height);
        output_params.pixel_format = Some(pf);
        output_params.frame_rate = params.frame_rate;
        if vvcc {
            // The record from the session's parameter sets (the same
            // VPS / SPS / PPS every access unit carries in-band).
            let mut ps = Vec::new();
            for kind in [
                crate::encoder::EmittedNalKind::Vps,
                crate::encoder::EmittedNalKind::Sps,
                crate::encoder::EmittedNalKind::Pps,
            ] {
                let nal = probe.emit_nal(kind)?;
                crate::encoder::annex_b_wrap(&nal, &mut ps);
            }
            output_params.extradata =
                VvcDecoderConfigurationRecord::from_annex_b(&ps)?.to_bytes()?;
        }
        Ok(Self {
            codec_id: CodecId::new(crate::CODEC_ID_STR),
            output_params,
            config,
            layout,
            qp,
            time_base,
            width: width as usize,
            height: height as usize,
            frame_index: 0,
            vvcc,
            queue: VecDeque::new(),
        })
    }

    /// The slice QP the lossy pictures are coded at.
    pub fn qp(&self) -> i32 {
        self.qp
    }

    /// The encoder configuration every picture is coded with.
    pub fn config(&self) -> &EncoderConfig {
        &self.config
    }

    /// Copy one input plane into a `PicturePlane` of `w × h` samples.
    fn fill_plane(
        src: &VideoPlane,
        w: usize,
        h: usize,
        wide: bool,
        dst: &mut crate::reconstruct::PicturePlane,
        what: &str,
    ) -> Result<()> {
        let bps = if wide { 2 } else { 1 };
        let need = w * bps;
        if src.stride < need || src.data.len() < src.stride * (h - 1) + need {
            return Err(Error::invalid(format!(
                "h266 encoder: {what} plane too small for {w}x{h} ({} bytes/row, {} bytes)",
                src.stride,
                src.data.len()
            )));
        }
        let max = (1u32 << dst.bit_depth) - 1;
        for y in 0..h {
            let row = &src.data[y * src.stride..y * src.stride + need];
            let out = &mut dst.samples[y * dst.stride..y * dst.stride + w];
            if wide {
                for (o, px) in out.iter_mut().zip(row.chunks_exact(2)) {
                    let v = u16::from_le_bytes([px[0], px[1]]);
                    *o = (u32::from(v).min(max)) as u16;
                }
            } else {
                for (o, &b) in out.iter_mut().zip(row) {
                    *o = u16::from(b);
                }
            }
        }
        Ok(())
    }

    fn frame_to_picture(&self, planes: &[VideoPlane]) -> Result<PictureBuffer> {
        let need = if self.layout.grey() { 1 } else { 3 };
        if planes.len() < need {
            return Err(Error::invalid(format!(
                "h266 encoder expects {need} plane(s) for the configured layout, got {}",
                planes.len()
            )));
        }
        let bd = self.layout.bit_depth;
        let mut pic = PictureBuffer::with_chroma_format_bd(
            self.width,
            self.height,
            self.layout.chroma_format_idc,
            1u16 << (bd - 1),
            bd,
        );
        let wide = self.layout.wide();
        Self::fill_plane(
            &planes[0],
            self.width,
            self.height,
            wide,
            &mut pic.luma,
            "luma",
        )?;
        if !self.layout.grey() {
            let (sw, sh) = self.layout.subsampling();
            let (cw, ch) = (self.width / sw, self.height / sh);
            Self::fill_plane(&planes[1], cw, ch, wide, &mut pic.cb, "Cb")?;
            Self::fill_plane(&planes[2], cw, ch, wide, &mut pic.cr, "Cr")?;
        }
        Ok(pic)
    }
}

impl CoreEncoder for H266CodecEncoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn output_params(&self) -> &CodecParameters {
        &self.output_params
    }

    fn send_frame(&mut self, frame: &Frame) -> Result<()> {
        let vf = match frame {
            Frame::Video(vf) => vf,
            _ => return Err(Error::invalid("h266 encoder accepts video frames only")),
        };
        let src = self.frame_to_picture(vf.image_planes())?;
        let (annex_b, _rec) = encode_idr_with_residuals_cfg(&src, self.qp, self.config)?;
        let data = if self.vvcc {
            // The in-band parameter sets must be the ones the record
            // carries; the record is rebuilt if a picture ever disagrees
            // (it never does for a fixed configuration).
            let rec = VvcDecoderConfigurationRecord::parse(&self.output_params.extradata)?;
            let mut same = true;
            for nal in iter_annex_b(&annex_b) {
                let t = nal.header.nal_unit_type;
                if matches!(
                    t,
                    NalUnitType::VpsNut | NalUnitType::SpsNut | NalUnitType::PpsNut
                ) && !rec
                    .nal_units_of_type(t.as_u8())
                    .iter()
                    .any(|u| u == nal.raw)
                {
                    same = false;
                }
            }
            if !same {
                self.output_params.extradata =
                    VvcDecoderConfigurationRecord::from_annex_b(&annex_b)?.to_bytes()?;
            }
            annex_b_to_length_prefixed(&annex_b, 4, true)?
        } else {
            annex_b
        };
        let pts = vf.pts.unwrap_or(self.frame_index);
        self.queue.push_back(
            Packet::new(0, self.time_base, data)
                .with_pts(pts)
                .with_dts(pts)
                .with_duration(1)
                .with_keyframe(true),
        );
        self.frame_index += 1;
        Ok(())
    }

    fn receive_packet(&mut self) -> Result<Packet> {
        self.queue.pop_front().ok_or(Error::NeedMore)
    }

    fn flush(&mut self) -> Result<()> {
        Ok(())
    }
}

/// Registry factory (`CodecInfo::encoder`).
pub fn make_encoder(params: &CodecParameters) -> Result<Box<dyn CoreEncoder>> {
    Ok(Box::new(H266CodecEncoder::new(params)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quality_ladder_is_monotone_and_ends_lossless() {
        assert_eq!(quality_to_qp(0), Some(51));
        assert_eq!(quality_to_qp(99), Some(2));
        assert_eq!(quality_to_qp(100), None);
        let mut last = 52;
        for q in 0..100 {
            let qp = quality_to_qp(q).unwrap();
            assert!(qp <= last, "q={q}: {qp} > {last}");
            assert!((2..=51).contains(&qp));
            last = qp;
        }
        assert_eq!(quality_to_qp(50), Some(26));
    }

    #[test]
    fn layout_refusals_are_typed() {
        assert!(Layout::of(PixelFormat::Yuv420P12Le).is_err());
        assert!(Layout::of(PixelFormat::Rgb24).is_err());
        assert_eq!(
            Layout::of(PixelFormat::Gray10Le).unwrap(),
            Layout {
                chroma_format_idc: 0,
                bit_depth: 10
            }
        );
        for pf in accepted_pixel_formats() {
            assert!(Layout::of(*pf).is_ok());
        }
    }
}
