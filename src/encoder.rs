//! VVC / H.266 forward-bitstream encoder scaffold (§7 — inverse of the
//! parser modules).
//!
//! Scope — round 8 lean goal: emit an Annex-B byte stream that the
//! decoder front-end in this crate can parse end-to-end without
//! errors. The encoder is **not** a real compressor; it writes one
//! coded video sequence consisting of:
//!
//! 1. a VPS NAL with a single-layer, single-sublayer PTL,
//! 2. an SPS NAL for the target resolution (8-bit 4:2:0, CTB = 128,
//!    tool flags all 0),
//! 3. a PPS NAL with `pps_no_pic_partition_flag = 1` (single-tile,
//!    single-slice),
//! 4. a standalone PH_NUT NAL carrying the
//!    `picture_header_structure()` (§7.3.2.7) — emitted out-of-band so
//!    our existing [`crate::picture_header::parse_picture_header_stateful`]
//!    gets a byte-aligned PH RBSP to walk,
//! 5. an IDR_N_LP slice NAL with `sh_picture_header_in_slice_header_flag
//!    = 0` (PH lives in the PH_NUT above) and an empty coded slice
//!    body.
//!
//! The bit-level layout mirrors the field-by-field order walked by
//! [`crate::vps::parse_vps`], [`crate::sps::parse_sps`],
//! [`crate::pps::parse_pps`], [`crate::picture_header::parse_picture_header_stateful`]
//! and [`crate::slice_header::parse_slice_header_stateful`]. Anything
//! further (residual syntax, CABAC, deblock, SAO, ALF, reconstruction)
//! is out of scope for this round.
//!
//! ## Limitations
//!
//! * The emitted slice payload is a "structural skeleton": the IDR
//!   slice ends right after `byte_alignment()`. A fully conformant
//!   decoder that tries to reconstruct pixels would see zero CTUs'
//!   worth of CABAC-coded data. Our decoder front-end, which stops at
//!   parse-level, is happy.
//! * Tool flags are hard-coded to the simplest legal combination:
//!   SAO / ALF / LMCS / dual-tree / etc. all disabled.
//! * No residual / CABAC / transform / reconstruction. The goal is
//!   parse-valid, not playable.

use oxideav_core::{Error, Result};

use crate::nal::NalUnitType;

/// MSB-first bit writer over a growing `Vec<u8>`. Mirrors the reader
/// order used by [`crate::bitreader::BitReader`] so encoder tests can
/// round-trip the same byte layouts the parser tests exercise.
pub struct BitWriter {
    data: Vec<u8>,
    /// Number of valid bits inside the partial tail byte (0..8). When
    /// `bit_pos == 0`, the tail byte is fully packed and the next
    /// `write` starts a fresh byte.
    bit_pos: u8,
}

impl Default for BitWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl BitWriter {
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            bit_pos: 0,
        }
    }

    pub fn with_capacity(n: usize) -> Self {
        Self {
            data: Vec::with_capacity(n),
            bit_pos: 0,
        }
    }

    /// Number of bits already written.
    pub fn bit_len(&self) -> u64 {
        if self.bit_pos == 0 {
            self.data.len() as u64 * 8
        } else {
            (self.data.len() as u64 - 1) * 8 + self.bit_pos as u64
        }
    }

    pub fn is_byte_aligned(&self) -> bool {
        self.bit_pos == 0
    }

    /// Finalise and return the packed byte vector. Any partial tail
    /// byte is kept as-is; callers that need byte alignment must
    /// invoke [`BitWriter::byte_align_zeros`] (or a trailing-bits
    /// helper) first.
    pub fn into_bytes(self) -> Vec<u8> {
        self.data
    }

    /// Write `n` bits (0..=32) of `value`, MSB-first.
    pub fn write_bits(&mut self, value: u32, n: u32) {
        debug_assert!(n <= 32);
        if n == 0 {
            return;
        }
        for i in (0..n).rev() {
            self.write_bit(((value >> i) & 1) as u8);
        }
    }

    /// Write a single bit (0 or 1).
    pub fn write_bit(&mut self, bit: u8) {
        if self.bit_pos == 0 {
            self.data.push(0);
            let shift = 7;
            *self.data.last_mut().unwrap() |= (bit & 1) << shift;
            self.bit_pos = 1;
        } else {
            let shift = 7 - self.bit_pos;
            *self.data.last_mut().unwrap() |= (bit & 1) << shift;
            self.bit_pos += 1;
            if self.bit_pos == 8 {
                self.bit_pos = 0;
            }
        }
    }

    /// Write `n` bits of `value`, MSB-first, allowing 33..=64 bits.
    pub fn write_bits_u64(&mut self, value: u64, n: u32) {
        debug_assert!(n <= 64);
        if n == 0 {
            return;
        }
        if n <= 32 {
            self.write_bits(value as u32, n);
            return;
        }
        let hi = (value >> 32) as u32;
        self.write_bits(hi, n - 32);
        self.write_bits(value as u32, 32);
    }

    /// Unsigned Exp-Golomb `ue(v)` (§9.2).
    pub fn write_ue(&mut self, value: u32) {
        // codeNum = value + 1, length = 2*floor(log2(codeNum)) + 1.
        let code_num = (value as u64) + 1;
        let mut zeros: u32 = 0;
        while (1u64 << (zeros + 1)) <= code_num {
            zeros += 1;
        }
        // Emit `zeros` leading zeros, then the (zeros+1)-bit code.
        self.write_bits(0, zeros);
        self.write_bits_u64(code_num, zeros + 1);
    }

    /// Signed Exp-Golomb `se(v)` (§9.2).
    pub fn write_se(&mut self, value: i32) {
        let code = if value <= 0 {
            (-(value as i64) as u64) * 2
        } else {
            (value as i64) as u64 * 2 - 1
        };
        self.write_ue(code as u32);
    }

    /// Append zero bits until the writer is byte-aligned. Useful for
    /// `alignment_zero_bit()` pads (§7.3.2.1 etc.).
    pub fn byte_align_zeros(&mut self) {
        while !self.is_byte_aligned() {
            self.write_bit(0);
        }
    }

    /// Append an `rbsp_trailing_bits()` sequence (§7.3.2.17): a single
    /// "1" stop bit followed by zero bits until byte alignment.
    pub fn rbsp_trailing_bits(&mut self) {
        self.write_bit(1);
        while !self.is_byte_aligned() {
            self.write_bit(0);
        }
    }

    /// Append a `byte_alignment()` sequence (§7.3.2.17). Same shape as
    /// `rbsp_trailing_bits()` — the two names in the spec differ only
    /// by which higher-level structure requests them.
    pub fn byte_alignment(&mut self) {
        self.rbsp_trailing_bits();
    }
}

/// Insert VVC emulation-prevention bytes into a raw RBSP payload
/// (§7.4.1, inverse of [`crate::nal::extract_rbsp`]).
///
/// The rule is: any sequence `0x00 0x00 0x00`, `0x00 0x00 0x01`, or
/// `0x00 0x00 0x02` inside the NAL payload gets a `0x03` inserted
/// after the two zeros so that decoders scanning for a start code do
/// not mis-trigger. `0x00 0x00 0x03` also gets the insertion so the
/// decoder can distinguish real `0x03` bytes from emulation-prevention
/// ones (§7.4.1).
pub fn insert_emulation_prevention(rbsp: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(rbsp.len() + rbsp.len() / 256 + 4);
    let mut i = 0;
    while i < rbsp.len() {
        if i + 2 < rbsp.len()
            && rbsp[i] == 0
            && rbsp[i + 1] == 0
            && matches!(rbsp[i + 2], 0x00..=0x03)
        {
            out.push(0);
            out.push(0);
            out.push(0x03);
            i += 2;
        } else {
            out.push(rbsp[i]);
            i += 1;
        }
    }
    out
}

/// Build the 2-byte NAL header for a given `(nal_unit_type, layer_id,
/// temporal_id_plus1)` triple (§7.3.1.2).
pub fn nal_header_bytes(nut: NalUnitType, layer_id: u8, temporal_id_plus1: u8) -> [u8; 2] {
    // b0 = [F=0:1][R=0:1][layer_id:6], b1 = [nal_unit_type:5][tid+1:3].
    let b0 = layer_id & 0x3F;
    let b1 = ((nut.as_u8() & 0x1F) << 3) | (temporal_id_plus1 & 0x07);
    [b0, b1]
}

/// Wrap an emulation-prevented NAL payload with a 4-byte Annex B start
/// code prefix. Used to concatenate NAL units into a single
/// byte-stream.
pub fn annex_b_wrap(nal_payload: &[u8], out: &mut Vec<u8>) {
    out.extend_from_slice(&[0x00, 0x00, 0x00, 0x01]);
    out.extend_from_slice(nal_payload);
}

/// Encoder configuration. The fields are deliberately spartan — only
/// the bits the bitstream layout branches on are exposed.
///
/// Round-54 — adds two opt-in encoder-pipeline knobs:
///
/// * [`Self::enable_alf_clip_rdo`] — when `true`, the per-class luma
///   ALF design pass is followed by [`crate::alf_aps_design::
///   design_clip_rdo_for_luma_aps`], which performs a greedy per-tap
///   descent over `alf_luma_clip_idx[ filtIdx ][ j ]`. The resulting
///   per-tap clip indices flow into `AlfApsData` via
///   [`crate::alf_aps_design::build_per_class_luma_alf_aps_data_with_clip`]
///   and the wire APS carries `alf_luma_clip_flag = 1` plus the per-tap
///   block. Default `false` until validated on a broader corpus.
///
/// * [`Self::enable_chroma_sao_merge`] — when `true`, the per-CTB chroma
///   SAO RDO is followed by [`crate::sao_enc::apply_chroma_sao_merge`],
///   which considers `sao_merge_left_flag` / `sao_merge_up_flag` for
///   every chroma CTB and rewrites the `SaoPicture` chroma slots to
///   inherit from the chosen neighbour when that lowers the §8.8.4.1
///   rate-distortion cost. The matching merge bits are emitted in the
///   second CABAC pass via [`crate::sao_syntax::encode_sao_ctb`] (round-54
///   added the encoder-side mirror of [`crate::sao_syntax::decode_sao_ctb`]).
///   Default `false`. When `true`, the encoder also flips on
///   `sps_sao_enabled_flag` + the matching `ph_sao_*_enabled_flag` bits
///   so the SAO bins land on the wire instead of being internal-only.
#[derive(Clone, Copy, Debug)]
pub struct EncoderConfig {
    pub width: u32,
    pub height: u32,
    /// Bit depth per component. Only 8-bit (luma = chroma) is emitted
    /// by this scaffold.
    pub bit_depth: u8,
    /// Chroma subsampling. Only 4:2:0 (`chroma_format_idc = 1`) is
    /// supported by this scaffold.
    pub chroma_format_idc: u8,
    /// Round-54 — opt-in: run the per-tap luma ALF clip-index RDO after
    /// the per-class coefficient design. Default `false`.
    pub enable_alf_clip_rdo: bool,
    /// Round-54 — opt-in: run [`crate::sao_enc::apply_chroma_sao_merge`]
    /// after the per-CTB chroma SAO RDO and emit `sao_merge_*_flag`
    /// CABAC bins on the wire. When `true`, `sps_sao_enabled_flag` is
    /// signalled 1 + the picture header carries `ph_sao_chroma_enabled_flag
    /// = 1` (luma SAO stays per the round-50 internal-only path; the
    /// merge map's `Independent`/`MergeLeft`/`MergeAbove` choices apply
    /// only to the chroma slots). Default `false`.
    pub enable_chroma_sao_merge: bool,
    /// Round-56 — opt-in: enable the multi-type-tree binary-split (MTT
    /// BT) picker in the encoder pipeline. For each leaf CU candidate
    /// the picker also evaluates `BT_VERT` (split into two stacked
    /// half-width CUs) and `BT_HORZ` (split into two stacked half-height
    /// CUs) and picks the lowest-cost option per `cost = SSE + λ·bits`
    /// (rate-distortion). Default `false` preserves the round-55 leaf-or-
    /// QT-only behaviour. The picker uses
    /// [`crate::syntax_enc::encode_coding_tree_bt_split`] for the syntax
    /// emit; the recursive sub-CU bodies follow the same
    /// `transform_tree()` / `cu_qp_delta` path as leaf CUs.
    pub enable_mtt_bt_picker: bool,
    /// Round-57 — opt-in: enable the multi-type-tree ternary-split (MTT
    /// TT) picker. Parallel to [`Self::enable_mtt_bt_picker`]; the
    /// picker additionally evaluates `TT_VERT` (1:2:1 three-column split)
    /// and `TT_HORZ` (1:2:1 three-row split) per VVC §7.3.10.4 /
    /// §7.4.10.4. The chosen option minimises `cost = SSE + λ·bits` over
    /// the union `{leaf, BT_VERT, BT_HORZ, TT_VERT, TT_HORZ}` when both
    /// flags are on; with only the TT flag the candidate set collapses
    /// to `{leaf, TT_VERT, TT_HORZ}`. Default `false` preserves the
    /// round-56 behaviour.
    pub enable_mtt_tt_picker: bool,
    /// Round-384 — opt-in LMCS (§7.4.3.19 luma mapping with chroma
    /// scaling). When `Some`, the pipeline forward-maps the source luma
    /// before intra coding (the coding loop runs in the mapped domain),
    /// emits `sps_lmcs_enabled_flag = 1`, ships the payload as an LMCS
    /// APS NAL (id 0), signals `ph_lmcs_enabled_flag = 1` /
    /// `ph_lmcs_aps_id = 0` / `ph_chroma_residual_scale_flag = 0` in
    /// the picture header plus `sh_lmcs_used_flag = 1` in the slice
    /// header, and inverse-maps the reconstruction (§8.8.1 step 1)
    /// before the deblock / SAO / ALF chain. Default `None`.
    pub lmcs: Option<crate::lmcs::LmcsData>,
}

impl EncoderConfig {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            bit_depth: 8,
            chroma_format_idc: 1,
            enable_alf_clip_rdo: false,
            enable_chroma_sao_merge: false,
            enable_mtt_bt_picker: false,
            enable_mtt_tt_picker: false,
            lmcs: None,
        }
    }

    fn validate(&self) -> Result<()> {
        if self.width == 0 || self.height == 0 {
            return Err(Error::invalid(
                "h266 encoder: picture dimensions must be non-zero",
            ));
        }
        if self.width > 16384 || self.height > 16384 {
            return Err(Error::invalid(format!(
                "h266 encoder: picture {}x{} exceeds level-independent 16384 cap",
                self.width, self.height
            )));
        }
        if self.bit_depth != 8 {
            return Err(Error::unsupported(
                "h266 encoder: only 8-bit is supported by the scaffold",
            ));
        }
        if self.chroma_format_idc != 1 {
            return Err(Error::unsupported(
                "h266 encoder: only 4:2:0 (chroma_format_idc = 1) is supported",
            ));
        }
        Ok(())
    }
}

/// Minimal VVC encoder. Stateless for this scaffold — every call to
/// [`VvcEncoder::encode_idr_frame`] produces a complete single-frame
/// bitstream (VPS + SPS + PPS + PH + IDR slice).
#[derive(Clone, Copy, Debug)]
pub struct VvcEncoder {
    config: EncoderConfig,
}

impl VvcEncoder {
    pub fn new(config: EncoderConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self { config })
    }

    pub fn config(&self) -> &EncoderConfig {
        &self.config
    }

    /// Emit a complete Annex B byte stream for a single IDR frame.
    ///
    /// The `frame` is accepted for API parity with other encoders in
    /// the workspace, but its pixel data is **not** encoded — this
    /// scaffold only emits the syntax skeleton. Frame dimensions live
    /// on the stream's [`CodecParameters`](oxideav_core::CodecParameters)
    /// and on this encoder's [`EncoderConfig`] — callers are responsible
    /// for keeping those in sync.
    pub fn encode_idr_frame(&self, _frame: &oxideav_core::VideoFrame) -> Result<Vec<u8>> {
        let mut bitstream = Vec::new();
        let vps = self.emit_vps()?;
        annex_b_wrap(&vps, &mut bitstream);
        let sps = self.emit_sps()?;
        annex_b_wrap(&sps, &mut bitstream);
        let pps = self.emit_pps()?;
        annex_b_wrap(&pps, &mut bitstream);
        let ph = self.emit_picture_header_nal()?;
        annex_b_wrap(&ph, &mut bitstream);
        let slice = self.emit_idr_slice()?;
        annex_b_wrap(&slice, &mut bitstream);
        Ok(bitstream)
    }

    /// Emit a single NAL in isolation — for tests that want to
    /// exercise one parser pass at a time.
    pub fn emit_nal(&self, kind: EmittedNalKind) -> Result<Vec<u8>> {
        match kind {
            EmittedNalKind::Vps => self.emit_vps(),
            EmittedNalKind::Sps => self.emit_sps(),
            EmittedNalKind::Pps => self.emit_pps(),
            EmittedNalKind::PictureHeader => self.emit_picture_header_nal(),
            EmittedNalKind::IdrSlice => self.emit_idr_slice(),
        }
    }

    /// Round-387 — emit a NAL with a parameterised §7.3.7 luma ALF APS
    /// chain. Since the §7.4.3.5 `pps_alf_info_in_ph_flag = 0`
    /// inference fix, the ALF chain lives in the SLICE HEADER: the
    /// `IdrSlice` kind reads the parameters; the `PictureHeader` kind
    /// no longer carries an ALF chain at all.
    pub fn emit_nal_with_alf_aps_chain(
        &self,
        kind: EmittedNalKind,
        num_alf_aps_ids_luma: u8,
        first_aps_id: u8,
    ) -> Result<Vec<u8>> {
        match kind {
            EmittedNalKind::IdrSlice => {
                let chain = AlfPhChain {
                    num_alf_aps_ids_luma,
                    luma_aps_id: first_aps_id,
                    ..AlfPhChain::default_full()
                };
                let header = self.emit_idr_slice_header_bytes(&chain, 0);
                Ok(Self::wrap_nal(NalUnitType::IdrNLp, 0, 1, &header))
            }
            _ => self.emit_nal(kind),
        }
    }

    // ------------------------------------------------------------------
    // VPS (§7.3.2.3)
    // ------------------------------------------------------------------

    fn emit_vps(&self) -> Result<Vec<u8>> {
        let mut bw = BitWriter::new();
        // vps_video_parameter_set_id must be > 0 (§7.4.3.3). Use 1.
        bw.write_bits(1, 4); // vps_video_parameter_set_id
        bw.write_bits(0, 6); // vps_max_layers_minus1
        bw.write_bits(0, 3); // vps_max_sublayers_minus1
                             // Single-layer + single-sublayer → neither
                             // `vps_default_ptl_dpb_hrd_max_tid_flag` nor
                             // `vps_all_independent_layers_flag` is transmitted.
                             //
                             // vps_layer_id[0] for the only layer.
        bw.write_bits(0, 6);
        // vps_max_layers_minus1 == 0 → skip the `each_layer_is_an_ols`
        // / `vps_ols_mode_idc` / output-layer-flags block entirely —
        // including `vps_num_ptls_minus1`, which §7.3.2.3 places INSIDE
        // the `vps_max_layers_minus1 > 0` block (§7.4.3.3 infers it to
        // 0 when absent). r387: the emitter used to write the u(8)
        // unconditionally, shifting the whole PTL by one byte on every
        // single-layer VPS (caught by a conforming third-party decoder
        // rejecting the `gci_alignment_zero_bit`).
        //
        // vps_pt_present_flag[0] is implicit (always 1 for i = 0).
        // With vps_default_ptl_dpb_hrd_max_tid_flag inferred to 1 and
        // vps_max_sublayers_minus1 = 0, no per-PTL max-tid bits are
        // transmitted.
        //
        // vps_ptl_alignment_zero_bit(s) — pad to byte boundary.
        while !bw.is_byte_aligned() {
            bw.write_bit(0);
        }
        // profile_tier_level(profileTierPresentFlag = 1, MaxNumSubLayersMinus1 = 0).
        bw.write_bits(1, 7); // general_profile_idc = 1 (Main 10)
        bw.write_bit(0); // general_tier_flag = 0
        bw.write_bits(0x5A, 8); // general_level_idc = 0x5A
        bw.write_bit(1); // ptl_frame_only_constraint_flag = 1
        bw.write_bit(0); // ptl_multilayer_enabled_flag = 0
                         // gci_present_flag = 0 → no GCI body.
        bw.write_bit(0);
        // Byte-align GCI (§7.3.3.2 tail).
        while !bw.is_byte_aligned() {
            bw.write_bit(0);
        }
        // MaxNumSubLayersMinus1 = 0 → no sublayer_level_present / level_idc.
        // ptl_num_sub_profiles = 0.
        bw.write_bits(0, 8);
        // No sub-profiles → no general_sub_profile_idc entries.

        // Single-layer tail (§7.3.2.3): TotalNumOlss = 1 with
        // vps_num_ptls_minus1 = 0 → no vps_ols_ptl_idx;
        // vps_each_layer_is_an_ols_flag infers to 1 → the whole
        // DPB / HRD block (including vps_timing_hrd_params_present_flag)
        // is skipped. Only vps_extension_flag remains before the
        // trailing bits — r387: previously omitted, so a conforming
        // decoder consumed our rbsp_stop_one_bit as the extension flag.
        bw.write_bit(0); // vps_extension_flag = 0
        bw.rbsp_trailing_bits();

        let rbsp = bw.into_bytes();
        Ok(Self::wrap_nal(NalUnitType::VpsNut, 0, 1, &rbsp))
    }

    // ------------------------------------------------------------------
    // SPS (§7.3.2.4)
    // ------------------------------------------------------------------

    fn emit_sps(&self) -> Result<Vec<u8>> {
        let mut bw = BitWriter::new();
        bw.write_bits(0, 4); // sps_seq_parameter_set_id
        bw.write_bits(0, 4); // sps_video_parameter_set_id
        bw.write_bits(0, 3); // sps_max_sublayers_minus1
        bw.write_bits(self.config.chroma_format_idc as u32, 2);
        // sps_log2_ctu_size_minus5 = 2 → CTB = 128.
        bw.write_bits(2, 2);
        // sps_ptl_dpb_hrd_params_present_flag = 0 → skip PTL+DPB+HRD.
        bw.write_bit(0);
        bw.write_bit(0); // sps_gdr_enabled_flag = 0
        bw.write_bit(0); // sps_ref_pic_resampling_enabled_flag = 0
        bw.write_ue(self.config.width);
        bw.write_ue(self.config.height);
        bw.write_bit(0); // sps_conformance_window_flag
        bw.write_bit(0); // sps_subpic_info_present_flag
        bw.write_ue((self.config.bit_depth - 8) as u32); // sps_bitdepth_minus8 = 0
        bw.write_bit(0); // sps_entropy_coding_sync_enabled_flag
        bw.write_bit(0); // sps_entry_point_offsets_present_flag
        bw.write_bits(4, 4); // sps_log2_max_pic_order_cnt_lsb_minus4 = 4 → 8-bit POC LSB
        bw.write_bit(0); // sps_poc_msb_cycle_flag
        bw.write_bits(0, 2); // sps_num_extra_ph_bytes = 0
        bw.write_bits(0, 2); // sps_num_extra_sh_bytes = 0

        // ---- Partition constraints (§7.3.2.4 tail) ----
        bw.write_ue(0); // log2_min_luma_cb_size_minus2
        bw.write_bit(0); // partition_constraints_override_enabled
        bw.write_ue(0); // log2_diff_min_qt_min_cb_intra_luma
        bw.write_ue(0); // max_mtt_depth_intra_luma = 0 → skip bt/tt
        bw.write_bit(0); // qtbtt_dual_tree_intra = 0 → skip chroma block
        bw.write_ue(0); // log2_diff_min_qt_min_cb_inter
        bw.write_ue(0); // max_mtt_depth_inter = 0 → skip bt/tt
        bw.write_bit(0); // max_luma_transform_size_64_flag

        // ---- Tool flags (§7.3.2.4 tail) ----
        bw.write_bit(0); // transform_skip_enabled
        bw.write_bit(0); // mts_enabled
        bw.write_bit(0); // lfnst_enabled
                         // chroma_format_idc != 0 → joint_cbcr + same_qp_table + QP tables.
        bw.write_bit(0); // joint_cbcr_enabled
        bw.write_bit(1); // same_qp_table_for_chroma → 1 QP table
        bw.write_se(0); // qp_table_start_minus26
        bw.write_ue(0); // num_points_minus1 = 0
        bw.write_ue(0); // delta_qp_in_val_minus1[0][0]
        bw.write_ue(0); // delta_qp_diff_val[0][0]
                        // Round-54 — `sps_sao_enabled_flag` is gated by the encoder
                        // config knob `enable_chroma_sao_merge`. Off → wire matches
                        // round-53 (SAO is internal-only on the encoder reconstruction).
                        // On → the per-CTB SAO syntax (and round-54's
                        // `sao_merge_*_flag`) is emitted by the second-pass CABAC walk
                        // and the PH carries `ph_sao_chroma_enabled_flag = 1`.
        bw.write_bit(self.config.enable_chroma_sao_merge as u8); // sao_enabled
                                                                 // Round-46 — flip `sps_alf_enabled_flag` on so the
                                                                 // PH-level ALF chain (`ph_alf_*`) becomes the
                                                                 // decoder-visible gate for the per-CTU ALF CABAC
                                                                 // emit. With `sps_chroma_format_idc != 0`,
                                                                 // §7.3.2.4 also emits `sps_ccalf_enabled_flag`;
                                                                 // we set both to 1.
        bw.write_bit(1); // alf_enabled
        bw.write_bit(1); // ccalf_enabled (gated by alf_enabled && chroma_format_idc != 0)
                         // Round-384 — sps_lmcs_enabled_flag mirrors the
                         // EncoderConfig::lmcs knob; the PH then carries the
                         // §7.3.2.8 LMCS chain and an LMCS APS NAL ships.
        bw.write_bit(if self.config.lmcs.is_some() { 1 } else { 0 }); // lmcs_enabled
        bw.write_bit(0); // weighted_pred
        bw.write_bit(0); // weighted_bipred
        bw.write_bit(0); // long_term_ref_pics
                         // sps_video_parameter_set_id = 0 → no inter_layer_prediction_enabled
        bw.write_bit(0); // idr_rpl_present
        bw.write_bit(0); // rpl1_same_as_rpl0 → 2 loops
        bw.write_ue(0); // num_ref_pic_lists[0] = 0
        bw.write_ue(0); // num_ref_pic_lists[1] = 0
        bw.write_bit(0); // ref_wraparound
        bw.write_bit(0); // temporal_mvp
        bw.write_bit(0); // amvr
        bw.write_bit(0); // bdof
        bw.write_bit(0); // smvd
        bw.write_bit(0); // dmvr
        bw.write_bit(0); // mmvd
        bw.write_ue(0); // six_minus_max_num_merge_cand → MaxNumMergeCand = 6
        bw.write_bit(0); // sbt
        bw.write_bit(0); // affine
        bw.write_bit(0); // bcw
        bw.write_bit(0); // ciip
                         // MaxNumMergeCand = 6 >= 2 → gpm_enabled.
        bw.write_bit(0); // gpm_enabled = 0
        bw.write_ue(0); // log2_parallel_merge_level_minus2
        bw.write_bit(0); // isp
        bw.write_bit(0); // mrl
        bw.write_bit(0); // mip
        bw.write_bit(0); // cclm
                         // chroma_format_idc == 1 → chroma-collocated flags.
        bw.write_bit(0); // chroma_horizontal_collocated
        bw.write_bit(0); // chroma_vertical_collocated
        bw.write_bit(0); // palette
        bw.write_bit(0); // ibc
        bw.write_bit(0); // ladf
        bw.write_bit(0); // explicit_scaling_list
        bw.write_bit(0); // dep_quant
        bw.write_bit(0); // sign_data_hiding
        bw.write_bit(0); // virtual_boundaries_enabled

        bw.write_bit(0); // sps_field_seq_flag
        bw.write_bit(0); // sps_vui_parameters_present_flag
        bw.write_bit(0); // sps_extension_flag

        bw.rbsp_trailing_bits();

        let rbsp = bw.into_bytes();
        Ok(Self::wrap_nal(NalUnitType::SpsNut, 0, 1, &rbsp))
    }

    // ------------------------------------------------------------------
    // PPS (§7.3.2.5)
    // ------------------------------------------------------------------

    fn emit_pps(&self) -> Result<Vec<u8>> {
        let mut bw = BitWriter::new();
        bw.write_bits(0, 6); // pps_pic_parameter_set_id
        bw.write_bits(0, 4); // pps_seq_parameter_set_id
        bw.write_bit(0); // pps_mixed_nalu_types_in_pic_flag
        bw.write_ue(self.config.width);
        bw.write_ue(self.config.height);
        bw.write_bit(0); // pps_conformance_window_flag
        bw.write_bit(0); // pps_scaling_window_explicit_signalling_flag
        bw.write_bit(0); // pps_output_flag_present_flag
        bw.write_bit(1); // pps_no_pic_partition_flag = 1
        bw.write_bit(0); // pps_subpic_id_mapping_present_flag
        bw.write_bit(0); // pps_cabac_init_present_flag
        bw.write_ue(0); // pps_num_ref_idx_default_active_minus1[0]
        bw.write_ue(0); // pps_num_ref_idx_default_active_minus1[1]
        bw.write_bit(0); // pps_rpl1_idx_present_flag
        bw.write_bit(0); // pps_weighted_pred_flag
        bw.write_bit(0); // pps_weighted_bipred_flag
        bw.write_bit(0); // pps_ref_wraparound_enabled_flag
        bw.write_se(0); // pps_init_qp_minus26 → QP 26
                        // Round-52 — enable per-CU QP delta signalling so the encoder
                        // pipeline can emit `cu_qp_delta_abs` + `cu_qp_delta_sign_flag`
                        // on CUs whose CBFs are non-zero (§7.3.13 + §8.7.1). The
                        // decoder already respects `cu_qp_delta_val` in
                        // `CtuWalker::reconstruct_leaf_cu` (luma dequant) and the
                        // §8.8.3 deblocker.
        bw.write_bit(1); // pps_cu_qp_delta_enabled_flag
        bw.write_bit(0); // pps_chroma_tool_offsets_present_flag
        bw.write_bit(0); // pps_deblocking_filter_control_present_flag
        bw.write_bit(0); // pps_picture_header_extension_present_flag
        bw.write_bit(0); // pps_slice_header_extension_present_flag
        bw.write_bit(0); // pps_extension_flag

        bw.rbsp_trailing_bits();

        let rbsp = bw.into_bytes();
        Ok(Self::wrap_nal(NalUnitType::PpsNut, 0, 1, &rbsp))
    }

    // ------------------------------------------------------------------
    // Picture Header NAL (§7.3.2.7) + IDR slice (§7.3.7)
    // ------------------------------------------------------------------

    /// Emit the §7.3.2.7 PH NAL. Since the §7.4.3.5
    /// `pps_*_info_in_ph_flag = 0` inference fix (r387), the PH of the
    /// no-partition profile is minimal: the ALF chain / ref-pic lists /
    /// qp-delta / SAO / deblocking gates all live in the slice header
    /// (§7.3.7) — see [`Self::emit_idr_slice_header_bytes`].
    pub fn emit_picture_header_nal(&self) -> Result<Vec<u8>> {
        let mut bw = BitWriter::new();
        self.emit_picture_header_body(&mut bw);
        bw.rbsp_trailing_bits();
        let rbsp = bw.into_bytes();
        Ok(Self::wrap_nal(NalUnitType::PhNut, 0, 1, &rbsp))
    }

    /// Emit the §7.3.2.8 `picture_header_structure()` body into `bw`.
    ///
    /// Gates that are closed by our SPS / PPS profile (and therefore
    /// emit nothing): ALF chain (`pps_alf_info_in_ph_flag = 0` — §7.4.3.5
    /// inference from `pps_no_pic_partition_flag = 1`), explicit scaling
    /// lists / virtual boundaries / POC MSB (SPS gates 0), output flag
    /// (`pps_output_flag_present_flag = 0`), ref_pic_lists
    /// (`pps_rpl_info_in_ph_flag = 0`), partition-constraints override
    /// (SPS gate 0), `ph_qp_delta` (`pps_qp_delta_info_in_ph_flag = 0` —
    /// the slice header carries `sh_qp_delta`), SAO
    /// (`pps_sao_info_in_ph_flag = 0`) and deblocking
    /// (`pps_dbf_info_in_ph_flag = 0`).
    fn emit_picture_header_body(&self, bw: &mut BitWriter) {
        // IRAP IDR intra-only picture.
        bw.write_bit(1); // ph_gdr_or_irap_pic_flag = 1
        bw.write_bit(0); // ph_non_ref_pic_flag = 0
        bw.write_bit(0); // ph_gdr_pic_flag = 0
        bw.write_bit(0); // ph_inter_slice_allowed_flag = 0
        bw.write_ue(0); // ph_pic_parameter_set_id = 0
                        // ph_pic_order_cnt_lsb — 8 bits.
        bw.write_bits(0, 8);
        // Round-384 — §7.3.2.8 LMCS chain, gated by
        // sps_lmcs_enabled_flag (== config.lmcs.is_some()). The
        // pipeline ships the LMCS APS at id 0 and keeps
        // ph_chroma_residual_scale_flag = 0 (the mapped-domain coding
        // loop only remaps luma).
        if self.config.lmcs.is_some() {
            bw.write_bit(1); // ph_lmcs_enabled_flag = 1
            bw.write_bits(0, 2); // ph_lmcs_aps_id = 0
            if self.config.chroma_format_idc != 0 {
                bw.write_bit(0); // ph_chroma_residual_scale_flag = 0
            }
        }
        // ph_intra_slice_allowed_flag == 1 (inferred) arm:
        // `if( pps_cu_qp_delta_enabled_flag )` — the round-52 PPS sets
        // it, and the pipeline runs with QG = CTB granularity.
        bw.write_ue(0); // ph_cu_qp_delta_subdiv_intra_slice = 0
    }

    /// Round-387 — emit the §7.3.7 IDR slice-header bits (through
    /// `byte_alignment()`), ready to be prefixed to the CABAC payload.
    ///
    /// Carries everything the §7.4.3.5 `pps_*_info_in_ph_flag = 0`
    /// inferences move out of the PH: the ALF chain, `sh_lmcs_used_flag`,
    /// `sh_qp_delta` (`SliceQpY = 26 + pps_init_qp_minus26 + sh_qp_delta`,
    /// so callers pass `slice_qp_y − 26`) and the SAO used flags.
    pub fn emit_idr_slice_header_bytes(&self, chain: &AlfPhChain, sh_qp_delta: i32) -> Vec<u8> {
        let mut bw = BitWriter::new();
        // sh_picture_header_in_slice_header_flag = 0 (separate PH NAL).
        bw.write_bit(0);
        // Single-slice no-partition profile: no sh_subpic_id /
        // sh_slice_address / sh_extra_bit / tile count. With
        // ph_inter_slice_allowed_flag = 0, sh_slice_type is inferred I.
        //
        // IDR nal_unit_type → sh_no_output_of_prior_pics_flag.
        bw.write_bit(0);
        // §7.3.7 ALF chain — sps_alf_enabled_flag && !pps_alf_info_in_ph_flag.
        bw.write_bit(1); // sh_alf_enabled_flag = 1
        bw.write_bits(chain.num_alf_aps_ids_luma as u32, 3);
        for _ in 0..chain.num_alf_aps_ids_luma {
            bw.write_bits(chain.luma_aps_id as u32, 3); // sh_alf_aps_id_luma[i]
        }
        if self.config.chroma_format_idc != 0 {
            bw.write_bit(chain.ph_alf_cb_enabled_flag as u8);
            bw.write_bit(chain.ph_alf_cr_enabled_flag as u8);
        }
        if chain.ph_alf_cb_enabled_flag || chain.ph_alf_cr_enabled_flag {
            bw.write_bits(chain.ph_alf_aps_id_chroma as u32, 3);
        }
        // sps_ccalf_enabled_flag = 1 in our SPS.
        bw.write_bit(chain.ph_alf_cc_cb_enabled_flag as u8);
        if chain.ph_alf_cc_cb_enabled_flag {
            bw.write_bits(chain.ph_alf_cc_cb_aps_id as u32, 3);
        }
        bw.write_bit(chain.ph_alf_cc_cr_enabled_flag as u8);
        if chain.ph_alf_cc_cr_enabled_flag {
            bw.write_bits(chain.ph_alf_cc_cr_aps_id as u32, 3);
        }
        // sh_lmcs_used_flag — ph_lmcs_enabled_flag && separate PH.
        if self.config.lmcs.is_some() {
            bw.write_bit(1);
        }
        // ref_pic_lists() skipped: IDR without sps_idr_rpl_present_flag.
        // I-slice → no cabac_init / collocated / pred_weight_table.
        //
        // !pps_qp_delta_info_in_ph_flag → sh_qp_delta.
        bw.write_se(sh_qp_delta);
        // No slice chroma QP offsets / cu-chroma-offset list (PPS gates 0).
        // SAO — sps_sao_enabled_flag && !pps_sao_info_in_ph_flag.
        if self.config.enable_chroma_sao_merge {
            bw.write_bit(chain.ph_sao_luma_enabled_flag as u8);
            if self.config.chroma_format_idc != 0 {
                bw.write_bit(chain.ph_sao_chroma_enabled_flag as u8);
            }
        }
        // Deblocking override (PPS gate 0), dep-quant / SDH / TS /
        // reverse-last-sig (SPS gates 0), SH extension (PPS gate 0):
        // nothing to emit.
        //
        // byte_alignment() — stop bit + zero pad; CABAC starts after.
        bw.byte_alignment();
        bw.into_bytes()
    }

    fn emit_idr_slice(&self) -> Result<Vec<u8>> {
        // Header-only slice stub for parser tests: the default ALF
        // chain matches the historical PH default (one luma APS at
        // id 2, chroma + CC-ALF on), sh_qp_delta = 0.
        let rbsp = self.emit_idr_slice_header_bytes(&AlfPhChain::default_full(), 0);
        Ok(Self::wrap_nal(NalUnitType::IdrNLp, 0, 1, &rbsp))
    }

    /// Wrap a raw RBSP payload (already terminated with
    /// `rbsp_trailing_bits()` by the caller) into a complete NAL-body
    /// byte vector: 2-byte NAL header + emulation-prevented RBSP.
    fn wrap_nal(nut: NalUnitType, layer_id: u8, temporal_id_plus1: u8, rbsp: &[u8]) -> Vec<u8> {
        let mut body = Vec::with_capacity(rbsp.len() + 2);
        let hdr = nal_header_bytes(nut, layer_id, temporal_id_plus1);
        body.extend_from_slice(&hdr);
        let ep = insert_emulation_prevention(rbsp);
        body.extend_from_slice(&ep);
        body
    }
}

/// Which single NAL to emit via [`VvcEncoder::emit_nal`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EmittedNalKind {
    Vps,
    Sps,
    Pps,
    PictureHeader,
    IdrSlice,
}

/// Round-51 — fully-parameterised ALF chain for the encoder emit.
///
/// The encoder pipeline runs an APS-vs-fixed RDO trade-off per APS
/// independently (round-51 extends round-48's luma trade-off to chroma
/// and CC-ALF). This struct propagates each per-APS ship/skip decision
/// into the bitstream emit by carrying every gate the ALF-chain syntax
/// reads when the enable flag is 1.
///
/// r387 — since the §7.4.3.5 `pps_alf_info_in_ph_flag = 0` inference
/// fix, the chain is emitted in the **slice header** (§7.3.7 `sh_alf_*`)
/// rather than the PH; the historical `ph_alf_*` field names are kept
/// (the values are the same syntax elements under their SH spelling).
#[derive(Clone, Copy, Debug)]
pub struct AlfPhChain {
    /// `ph_num_alf_aps_ids_luma` — 0 when the round-48 trade-off skipped
    /// the luma APS NAL, 1 when it shipped.
    pub num_alf_aps_ids_luma: u8,
    /// `ph_alf_aps_id_luma[i]` repeated for each `i ∈ 0..num_alf_aps_ids_luma`.
    /// The pipeline ships the luma APS at id 2.
    pub luma_aps_id: u8,
    /// `ph_alf_cb_enabled_flag`.
    pub ph_alf_cb_enabled_flag: bool,
    /// `ph_alf_cr_enabled_flag`.
    pub ph_alf_cr_enabled_flag: bool,
    /// `ph_alf_aps_id_chroma` — emitted only when at least one of the Cb /
    /// Cr enable flags is set. The pipeline binds primary chroma ALF to
    /// APS id 0.
    pub ph_alf_aps_id_chroma: u8,
    /// `ph_alf_cc_cb_enabled_flag`.
    pub ph_alf_cc_cb_enabled_flag: bool,
    /// `ph_alf_cc_cb_aps_id` — pipeline binds CC-ALF Cb to APS id 1 when
    /// shipped.
    pub ph_alf_cc_cb_aps_id: u8,
    /// `ph_alf_cc_cr_enabled_flag`.
    pub ph_alf_cc_cr_enabled_flag: bool,
    /// `ph_alf_cc_cr_aps_id` — pipeline binds CC-ALF Cr to APS id 1 when
    /// shipped.
    pub ph_alf_cc_cr_aps_id: u8,
    /// Round-54 — `ph_sao_luma_enabled_flag`. Only emitted when
    /// `sps_sao_enabled_flag = 1` (round-54 ties this to
    /// `EncoderConfig::enable_chroma_sao_merge`). Present in the PH per
    /// §7.4.3.7 because `pps_sao_info_in_ph_flag = 1` is inferred from
    /// the round-35 PPS's `pps_no_pic_partition_flag = 1`.
    pub ph_sao_luma_enabled_flag: bool,
    /// Round-54 — `ph_sao_chroma_enabled_flag`. Same gate as luma; the
    /// round-54 chroma SAO merge path always sets this `true` so the
    /// per-CTB `sao_type_idx_chroma` + the new `sao_merge_*_flag` bins
    /// flow through the wire stream.
    pub ph_sao_chroma_enabled_flag: bool,
}

impl AlfPhChain {
    /// The historical default chain: one luma APS at id 2, primary
    /// chroma ALF at id 0, CC-ALF Cb / Cr at id 1, SAO off.
    pub fn default_full() -> Self {
        Self {
            num_alf_aps_ids_luma: 1,
            luma_aps_id: 2,
            ph_alf_cb_enabled_flag: true,
            ph_alf_cr_enabled_flag: true,
            ph_alf_aps_id_chroma: 0,
            ph_alf_cc_cb_enabled_flag: true,
            ph_alf_cc_cb_aps_id: 1,
            ph_alf_cc_cr_enabled_flag: true,
            ph_alf_cc_cr_aps_id: 1,
            ph_sao_luma_enabled_flag: false,
            ph_sao_chroma_enabled_flag: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitreader::BitReader;
    use crate::nal::{extract_rbsp, iter_annex_b, NalHeader};
    use crate::picture_header::parse_picture_header_stateful;
    use crate::pps::parse_pps;
    use crate::slice_header::{parse_slice_header_stateful, PhState};
    use crate::sps::parse_sps;
    use crate::vps::parse_vps;
    use oxideav_core::{VideoFrame, VideoPlane};

    // ---- BitWriter / EP tests ----

    #[test]
    fn bitwriter_packs_msb_first() {
        let mut bw = BitWriter::new();
        bw.write_bit(1);
        bw.write_bits(0b011, 3);
        bw.write_bits(0xAA, 8);
        bw.byte_align_zeros();
        let out = bw.into_bytes();
        assert_eq!(out, vec![0xBA, 0xA0]);
    }

    #[test]
    fn bitwriter_ue_roundtrips_through_parser() {
        let mut bw = BitWriter::new();
        for v in [0u32, 1, 2, 3, 15, 16, 255, 1023] {
            bw.write_ue(v);
        }
        bw.byte_align_zeros();
        let bytes = bw.into_bytes();
        let mut br = BitReader::new(&bytes);
        for v in [0u32, 1, 2, 3, 15, 16, 255, 1023] {
            assert_eq!(br.ue().unwrap(), v);
        }
    }

    #[test]
    fn bitwriter_se_roundtrips_through_parser() {
        let mut bw = BitWriter::new();
        for v in [0i32, 1, -1, 2, -2, 7, -7, 42, -42] {
            bw.write_se(v);
        }
        bw.byte_align_zeros();
        let bytes = bw.into_bytes();
        let mut br = BitReader::new(&bytes);
        for v in [0i32, 1, -1, 2, -2, 7, -7, 42, -42] {
            assert_eq!(br.se().unwrap(), v);
        }
    }

    #[test]
    fn emulation_prevention_inserts_0x03() {
        let input = [0x00, 0x00, 0x00, 0xAB, 0x00, 0x00, 0x01];
        let ep = insert_emulation_prevention(&input);
        assert_eq!(
            ep,
            vec![0x00, 0x00, 0x03, 0x00, 0xAB, 0x00, 0x00, 0x03, 0x01]
        );
        assert_eq!(extract_rbsp(&ep), input.to_vec());
    }

    #[test]
    fn emulation_prevention_passthrough() {
        let input = [0x00, 0xAB, 0x01, 0x00, 0x05];
        let ep = insert_emulation_prevention(&input);
        assert_eq!(ep, input.to_vec());
    }

    // ---- Encoder structural tests ----

    fn dummy_frame(width: u32, height: u32) -> VideoFrame {
        let y_stride = width as usize;
        let c_stride = (width / 2) as usize;
        VideoFrame {
            pts: None,
            planes: vec![
                VideoPlane {
                    stride: y_stride,
                    data: vec![0x80u8; y_stride * height as usize],
                },
                VideoPlane {
                    stride: c_stride,
                    data: vec![0x80u8; c_stride * (height / 2) as usize],
                },
                VideoPlane {
                    stride: c_stride,
                    data: vec![0x80u8; c_stride * (height / 2) as usize],
                },
            ],
        }
    }

    #[test]
    fn encoder_rejects_bad_config() {
        assert!(VvcEncoder::new(EncoderConfig::new(0, 240)).is_err());
        assert!(VvcEncoder::new(EncoderConfig::new(320, 0)).is_err());
        assert!(VvcEncoder::new(EncoderConfig {
            bit_depth: 10,
            ..EncoderConfig::new(320, 240)
        })
        .is_err());
        assert!(VvcEncoder::new(EncoderConfig {
            chroma_format_idc: 3,
            ..EncoderConfig::new(320, 240)
        })
        .is_err());
    }

    #[test]
    fn encoder_ignores_frame_dimensions() {
        // Slim VideoFrame no longer carries dimensions — the encoder
        // can no longer validate them, so it accepts whatever planes
        // the caller hands over (dimensions live on the stream's
        // CodecParameters / `EncoderConfig`).
        let enc = VvcEncoder::new(EncoderConfig::new(320, 240)).unwrap();
        let frame = dummy_frame(160, 120);
        assert!(enc.encode_idr_frame(&frame).is_ok());
    }

    #[test]
    fn vps_roundtrips_through_parser() {
        let enc = VvcEncoder::new(EncoderConfig::new(320, 240)).unwrap();
        let vps_nal = enc.emit_nal(EmittedNalKind::Vps).unwrap();
        let hdr = NalHeader::parse(&vps_nal).unwrap();
        assert_eq!(hdr.nal_unit_type, NalUnitType::VpsNut);
        let rbsp = extract_rbsp(&vps_nal[2..]);
        let vps = parse_vps(&rbsp).expect("VPS must parse");
        assert_eq!(vps.vps_video_parameter_set_id, 1);
        assert_eq!(vps.vps_max_layers_minus1, 0);
        assert_eq!(vps.vps_max_sublayers_minus1, 0);
        assert_eq!(vps.profile_tier_levels.len(), 1);
        assert_eq!(vps.profile_tier_levels[0].general_profile_idc, 1);
        assert_eq!(vps.profile_tier_levels[0].general_level_idc, 0x5A);
    }

    #[test]
    fn sps_roundtrips_through_parser() {
        let enc = VvcEncoder::new(EncoderConfig::new(320, 240)).unwrap();
        let sps_nal = enc.emit_nal(EmittedNalKind::Sps).unwrap();
        let hdr = NalHeader::parse(&sps_nal).unwrap();
        assert_eq!(hdr.nal_unit_type, NalUnitType::SpsNut);
        let rbsp = extract_rbsp(&sps_nal[2..]);
        let sps = parse_sps(&rbsp).expect("SPS must parse");
        assert_eq!(sps.sps_seq_parameter_set_id, 0);
        assert_eq!(sps.sps_video_parameter_set_id, 0);
        assert_eq!(sps.sps_chroma_format_idc, 1);
        assert_eq!(sps.sps_pic_width_max_in_luma_samples, 320);
        assert_eq!(sps.sps_pic_height_max_in_luma_samples, 240);
        assert_eq!(sps.bit_depth_y(), 8);
        assert_eq!(sps.ctb_size(), 128);
        assert!(!sps.tool_flags.sao_enabled_flag);
        // Round-46 — `alf_enabled_flag` + `ccalf_enabled_flag` are now
        // emitted as 1 by the SPS so the PH-level ALF chain (and per-CTU
        // ALF CABAC) becomes part of the decoder-visible bitstream.
        assert!(sps.tool_flags.alf_enabled_flag);
        assert!(sps.tool_flags.ccalf_enabled_flag);
        assert!(!sps.tool_flags.lmcs_enabled_flag);
        assert!(!sps.tool_flags.transform_skip_enabled_flag);
        assert!(!sps.tool_flags.mts_enabled_flag);
        assert!(!sps.tool_flags.lfnst_enabled_flag);
    }

    #[test]
    fn pps_roundtrips_through_parser() {
        let enc = VvcEncoder::new(EncoderConfig::new(320, 240)).unwrap();
        let pps_nal = enc.emit_nal(EmittedNalKind::Pps).unwrap();
        let hdr = NalHeader::parse(&pps_nal).unwrap();
        assert_eq!(hdr.nal_unit_type, NalUnitType::PpsNut);
        let rbsp = extract_rbsp(&pps_nal[2..]);
        let pps = parse_pps(&rbsp).expect("PPS must parse");
        assert_eq!(pps.pps_pic_parameter_set_id, 0);
        assert_eq!(pps.pps_seq_parameter_set_id, 0);
        assert!(pps.pps_no_pic_partition_flag);
        assert_eq!(pps.pps_pic_width_in_luma_samples, 320);
        assert_eq!(pps.pps_pic_height_in_luma_samples, 240);
        // §7.4.3.5 — with pps_no_pic_partition_flag == 1 every absent
        // pps_*_info_in_ph_flag infers to 0: RPL / SAO / ALF / QP-delta
        // information lives in the slice header.
        assert!(!pps.pps_rpl_info_in_ph_flag);
        assert!(!pps.pps_sao_info_in_ph_flag);
        assert!(!pps.pps_alf_info_in_ph_flag);
        assert!(!pps.pps_qp_delta_info_in_ph_flag);
        assert!(!pps.pps_dbf_info_in_ph_flag);
        assert_eq!(pps.pps_init_qp_minus26, 0);
    }

    #[test]
    fn picture_header_roundtrips_through_parser() {
        let enc = VvcEncoder::new(EncoderConfig::new(320, 240)).unwrap();
        let sps_rbsp = extract_rbsp(&enc.emit_nal(EmittedNalKind::Sps).unwrap()[2..]);
        let sps = parse_sps(&sps_rbsp).unwrap();
        let pps_rbsp = extract_rbsp(&enc.emit_nal(EmittedNalKind::Pps).unwrap()[2..]);
        let pps = parse_pps(&pps_rbsp).unwrap();
        let ph_nal = enc.emit_nal(EmittedNalKind::PictureHeader).unwrap();
        let hdr = NalHeader::parse(&ph_nal).unwrap();
        assert_eq!(hdr.nal_unit_type, NalUnitType::PhNut);
        let ph_rbsp = extract_rbsp(&ph_nal[2..]);
        let ph = parse_picture_header_stateful(&ph_rbsp, &sps, &pps).expect("PH must parse");
        assert!(ph.ph_gdr_or_irap_pic_flag);
        assert!(!ph.ph_inter_slice_allowed_flag);
        assert!(ph.ph_intra_slice_allowed_flag);
        assert_eq!(ph.ph_pic_parameter_set_id, 0);
        assert_eq!(ph.ph_pic_order_cnt_lsb, 0);
        assert_eq!(ph.ph_qp_delta, 0);
        // r387 — §7.4.3.5 pps_alf_info_in_ph_flag infers to 0 for the
        // no-partition PPS, so the PH carries NO ALF chain (it lives in
        // the §7.3.7 slice header — see the IdrSlice test below).
        assert!(!ph.ph_alf_enabled_flag);
        assert_eq!(ph.ph_num_alf_aps_ids_luma, 0);
        assert!(!ph.ph_lmcs_enabled_flag);

        // The §7.3.7 slice header now carries the ALF chain: one luma
        // APS at id 2, primary chroma at id 0, CC-ALF Cb / Cr at id 1
        // (the AlfPhChain::default_full layout).
        let slice_nal = enc.emit_nal(EmittedNalKind::IdrSlice).unwrap();
        let slice_rbsp = extract_rbsp(&slice_nal[2..]);
        let ph_state = PhState {
            ph_inter_slice_allowed_flag: false,
            ph_intra_slice_allowed_flag: true,
            ph_alf_enabled_flag: false,
            ph_lmcs_enabled_flag: false,
            ph_explicit_scaling_list_enabled_flag: false,
            ph_temporal_mvp_enabled_flag: false,
            ph_sao_luma_enabled_flag: false,
            ph_sao_chroma_enabled_flag: false,
            num_extra_sh_bits: 0,
            nal_unit_type: NalUnitType::IdrNLp,
        };
        let sh = parse_slice_header_stateful(&slice_rbsp, &sps, &pps, &ph_state)
            .expect("slice header must parse");
        assert!(sh.sh_alf_enabled_flag);
        assert_eq!(sh.sh_num_alf_aps_ids_luma, 1);
        assert_eq!(sh.sh_alf_aps_id_luma, vec![2]);
        assert!(sh.sh_alf_cb_enabled_flag);
        assert!(sh.sh_alf_cr_enabled_flag);
        assert_eq!(sh.sh_alf_aps_id_chroma, 0);
        assert!(sh.sh_alf_cc_cb_enabled_flag);
        assert_eq!(sh.sh_alf_cc_cb_aps_id, 1);
        assert!(sh.sh_alf_cc_cr_enabled_flag);
        assert_eq!(sh.sh_alf_cc_cr_aps_id, 1);
        assert_eq!(sh.sh_qp_delta, 0);
    }

    #[test]
    fn idr_slice_parses_as_vcl_idr() {
        let enc = VvcEncoder::new(EncoderConfig::new(320, 240)).unwrap();
        let nal = enc.emit_nal(EmittedNalKind::IdrSlice).unwrap();
        let hdr = NalHeader::parse(&nal).unwrap();
        assert_eq!(hdr.nal_unit_type, NalUnitType::IdrNLp);
        assert!(hdr.nal_unit_type.is_vcl());
        assert!(hdr.nal_unit_type.is_irap());
    }

    #[test]
    fn full_idr_bitstream_roundtrips() {
        let enc = VvcEncoder::new(EncoderConfig::new(320, 240)).unwrap();
        let frame = dummy_frame(320, 240);
        let bs = enc.encode_idr_frame(&frame).unwrap();

        // `VvcEncoder::encode_idr_frame` is the round-35 minimal path —
        // it does NOT walk through `encode_idr_with_residuals` and so
        // does not emit the round-45 chroma+CC-ALF APSes or the
        // round-47 luma APS. Only VPS/SPS/PPS/PH/slice land here.
        let nals: Vec<_> = iter_annex_b(&bs).collect();
        assert_eq!(nals.len(), 5);
        assert_eq!(nals[0].header.nal_unit_type, NalUnitType::VpsNut);
        assert_eq!(nals[1].header.nal_unit_type, NalUnitType::SpsNut);
        assert_eq!(nals[2].header.nal_unit_type, NalUnitType::PpsNut);
        assert_eq!(nals[3].header.nal_unit_type, NalUnitType::PhNut);
        assert_eq!(nals[4].header.nal_unit_type, NalUnitType::IdrNLp);

        let vps_rbsp = extract_rbsp(nals[0].payload());
        parse_vps(&vps_rbsp).expect("VPS round-trip");

        let sps_rbsp = extract_rbsp(nals[1].payload());
        let sps = parse_sps(&sps_rbsp).expect("SPS round-trip");

        let pps_rbsp = extract_rbsp(nals[2].payload());
        let pps = parse_pps(&pps_rbsp).expect("PPS round-trip");

        let ph_rbsp = extract_rbsp(nals[3].payload());
        let ph = parse_picture_header_stateful(&ph_rbsp, &sps, &pps).expect("PH round-trip");
        assert!(ph.ph_gdr_or_irap_pic_flag);
        assert!(!ph.ph_inter_slice_allowed_flag);

        let slice_rbsp = extract_rbsp(nals[4].payload());
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
        let sh = parse_slice_header_stateful(&slice_rbsp, &sps, &pps, &ph_state)
            .expect("slice header round-trip");
        assert!(!sh.sh_picture_header_in_slice_header_flag);
        assert!(sh.embedded_picture_header.is_none());
        assert!(sh.trailing_bits.is_empty());
    }
}
