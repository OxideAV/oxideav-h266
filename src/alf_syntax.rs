//! VVC ALF per-CTU CABAC syntax — §7.3.11.2 `coding_tree_unit()` ALF
//! block.
//!
//! Decodes / encodes the `alf_ctb_flag[]`, `alf_use_aps_flag`,
//! `alf_luma_prev_filter_idx` / `alf_luma_fixed_filter_idx`,
//! `alf_ctb_filter_alt_idx[]`, `alf_ctb_cc_cb_idc[]`, and
//! `alf_ctb_cc_cr_idc[]` syntax elements that gate the §8.8.5 ALF apply
//! pass on a per-CTB basis.
//!
//! ## Binarisation summary (Table 127)
//!
//! | Syntax element                           | Binarisation | cMax                                                |
//! |------------------------------------------|--------------|-----------------------------------------------------|
//! | `alf_ctb_flag[ ][ ][ ]`                  | FL           | 1                                                   |
//! | `alf_use_aps_flag`                       | FL           | 1                                                   |
//! | `alf_luma_fixed_filter_idx`              | TB (bypass)  | 15                                                  |
//! | `alf_luma_prev_filter_idx`               | TB (bypass)  | `sh_num_alf_aps_ids_luma − 1`                       |
//! | `alf_ctb_filter_alt_idx[ ][ ][ ]`        | TR           | `alf_chroma_num_alt_filters_minus1`                 |
//! | `alf_ctb_cc_cb_idc[ ][ ]`                | TR           | `alf_cc_cb_filters_signalled_minus1 + 1`            |
//! | `alf_ctb_cc_cr_idc[ ][ ]`                | TR           | `alf_cc_cr_filters_signalled_minus1 + 1`            |
//!
//! ## ctxInc assignments (Table 132)
//!
//! * `alf_ctb_flag[cIdx]` bin 0 — `ctxInc ∈ 0..8` per §9.3.4.2.2 with
//!   `(condL, condA) = alf_ctb_flag[cIdx][ctbLx/Ax][ctbLy/Ay]` and
//!   `ctxSetIdx = cIdx`.
//! * `alf_use_aps_flag` — single context `ctxInc = 0`.
//! * `alf_luma_fixed_filter_idx` / `alf_luma_prev_filter_idx` — entirely
//!   bypass (TB binarisation).
//! * `alf_ctb_filter_alt_idx[chromaIdx]` — every TR bin uses the same
//!   `ctxInc = chromaIdx` (0 for Cb, 1 for Cr).
//! * `alf_ctb_cc_cb_idc` bin 0 — `ctxInc ∈ 0..2` per §9.3.4.2.2 with
//!   `(condL, condA) = alf_ctb_cc_cb_idc[ctbLx/Ax][ctbLy/Ay] != 0`,
//!   `ctxSetIdx = 0`. Bins 1.. are bypass.
//! * `alf_ctb_cc_cr_idc` bin 0 — same shape as `alf_ctb_cc_cb_idc`.
//!
//! Spec reference: ITU-T H.266 | ISO/IEC 23090-3 (V4, 01/2026)
//! §7.3.11.2 + §9.3.4.2.1 (Table 132) + §9.3.4.2.2 + Tables 52..56.

use oxideav_core::{Error, Result};

use crate::alf::AlfPicture;
use crate::cabac::{ArithDecoder, ContextModel};
use crate::cabac_enc::ArithEncoder;
use crate::slice_header::SliceType;
use crate::tables::{init_contexts, SyntaxCtx};

/// CABAC context bundle used by the per-CTU ALF syntax parser /
/// emitter. Each field is the full per-syntax-element ctxIdx array
/// from Tables 52..56.
pub struct AlfCtxs {
    /// Table 52 — `alf_ctb_flag` (27 ctxIdx, 3 per cIdx per initType).
    pub ctb_flag: Vec<ContextModel>,
    /// Table 53 — `alf_use_aps_flag` (3 ctxIdx, one per initType).
    pub use_aps_flag: Vec<ContextModel>,
    /// Table 54 — `alf_ctb_cc_cb_idc` (9 ctxIdx, 3 per initType).
    pub cc_cb_idc: Vec<ContextModel>,
    /// Table 55 — `alf_ctb_cc_cr_idc` (9 ctxIdx, 3 per initType).
    pub cc_cr_idc: Vec<ContextModel>,
    /// Table 56 — `alf_ctb_filter_alt_idx` (6 ctxIdx, 2 per initType).
    pub filter_alt_idx: Vec<ContextModel>,
}

impl std::fmt::Debug for AlfCtxs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AlfCtxs")
            .field("ctb_flag.len", &self.ctb_flag.len())
            .field("use_aps_flag.len", &self.use_aps_flag.len())
            .field("cc_cb_idc.len", &self.cc_cb_idc.len())
            .field("cc_cr_idc.len", &self.cc_cr_idc.len())
            .field("filter_alt_idx.len", &self.filter_alt_idx.len())
            .finish()
    }
}

impl AlfCtxs {
    /// Build the ALF context arrays for the given slice QP.
    pub fn init(slice_qp_y: i32) -> Self {
        Self {
            ctb_flag: init_contexts(SyntaxCtx::AlfCtbFlag, slice_qp_y),
            use_aps_flag: init_contexts(SyntaxCtx::AlfUseApsFlag, slice_qp_y),
            cc_cb_idc: init_contexts(SyntaxCtx::AlfCtbCcCbIdc, slice_qp_y),
            cc_cr_idc: init_contexts(SyntaxCtx::AlfCtbCcCrIdc, slice_qp_y),
            filter_alt_idx: init_contexts(SyntaxCtx::AlfCtbFilterAltIdx, slice_qp_y),
        }
    }
}

/// Per-slice configuration the ALF syntax parser/emitter needs.
#[derive(Clone, Copy, Debug)]
pub struct AlfSyntaxConfig {
    /// `sh_alf_enabled_flag` — gates the entire luma + chroma ALF block.
    pub alf_enabled: bool,
    /// `sh_alf_cb_enabled_flag`.
    pub cb_enabled: bool,
    /// `sh_alf_cr_enabled_flag`.
    pub cr_enabled: bool,
    /// `sh_alf_cc_cb_enabled_flag`.
    pub cc_cb_enabled: bool,
    /// `sh_alf_cc_cr_enabled_flag`.
    pub cc_cr_enabled: bool,
    /// `sh_num_alf_aps_ids_luma` — width input for `alf_luma_prev_filter_idx`.
    pub sh_num_alf_aps_ids_luma: u8,
    /// `alf_chroma_num_alt_filters_minus1` (taken from the bound chroma
    /// APS, or 0 when chroma ALF is off / monochrome).
    pub alf_chroma_num_alt_filters_minus1: u8,
    /// `alf_cc_cb_filters_signalled_minus1` (taken from the bound
    /// CC-Cb APS).
    pub alf_cc_cb_filters_signalled_minus1: u8,
    /// `alf_cc_cr_filters_signalled_minus1`.
    pub alf_cc_cr_filters_signalled_minus1: u8,
    /// `sps_chroma_format_idc` — 0 (monochrome), 1 (4:2:0), 2 (4:2:2),
    /// 3 (4:4:4).
    pub chroma_format_idc: u32,
    /// Slice type — selects the initType row for Tables 52..56 via
    /// eq. 1527.
    pub slice_type: SliceType,
    /// `sh_cabac_init_flag` — flips initType for P/B slices per
    /// eq. 1527.
    pub sh_cabac_init_flag: bool,
}

impl AlfSyntaxConfig {
    /// Eq. 1527 — derive initType from slice type + `sh_cabac_init_flag`.
    pub fn init_type(&self) -> usize {
        match self.slice_type {
            SliceType::I => 0,
            SliceType::P => {
                if self.sh_cabac_init_flag {
                    2
                } else {
                    1
                }
            }
            SliceType::B => {
                if self.sh_cabac_init_flag {
                    1
                } else {
                    2
                }
            }
        }
    }
}

/// Neighbour availability for the §9.3.4.2.2 `alf_ctb_*` ctxInc
/// derivation.
///
/// Per §9.3.4.2.2 the locations `(ctbLx, ctbLy) = (ctbX - 1, ctbY)` and
/// `(ctbAx, ctbAy) = (ctbX, ctbY - 1)` are used; the `availableL` /
/// `availableA` flags are the §6.4.4 neighbour-availability outputs.
/// For the foundation single-slice / single-tile pipeline these reduce
/// to "neighbour exists in the picture" (i.e. `ctbX > 0` for L,
/// `ctbY > 0` for A).
#[derive(Clone, Copy, Debug, Default)]
pub struct AlfNeighbours {
    pub left_avail: bool,
    pub up_avail: bool,
}

/// Decode the ALF block of `coding_tree_unit()` for one CTB. Updates
/// `alf_pic` at position `(rx, ry)` with the parsed flags / indices.
///
/// `nbrs` carries the picture-edge neighbour-availability bits the
/// §9.3.4.2.2 ctxInc derivation requires.
pub fn decode_alf_ctu(
    dec: &mut ArithDecoder<'_>,
    ctxs: &mut AlfCtxs,
    cfg: &AlfSyntaxConfig,
    alf_pic: &mut AlfPicture,
    rx: u32,
    ry: u32,
    nbrs: AlfNeighbours,
) -> Result<()> {
    if !cfg.alf_enabled {
        return Ok(());
    }
    let init_type = cfg.init_type();

    let mut ctb = alf_pic.get(rx, ry);

    // -- alf_ctb_flag[0] (luma) -------------------------------------
    // §9.3.4.2.2 / Table 133: ctxInc = (condL && availL) + (condA &&
    // availA) + cIdx*3, with condL/condA = alf_ctb_flag[cIdx][L/A].
    let cond_l_y = nbrs.left_avail && rx > 0 && alf_pic.get(rx - 1, ry).luma_on;
    let cond_a_y = nbrs.up_avail && ry > 0 && alf_pic.get(rx, ry - 1).luma_on;
    let inc_y = cond_l_y as usize + cond_a_y as usize; // cIdx = 0
    let idx_y = init_type * 9 + inc_y;
    let luma_on = dec.decode_decision(&mut ctxs.ctb_flag[idx_y])? == 1;
    ctb.luma_on = luma_on;

    if luma_on {
        // alf_use_aps_flag — single ctx-coded bin per Table 132, but
        // only PRESENT when `sh_num_alf_aps_ids_luma > 0` (§7.3.11.2).
        // With zero luma APS ids the flag is inferred 0 and the
        // fixed-filter branch is taken directly. r415: the
        // unconditional read here was a matched encoder/decoder
        // deviation — every stream whose ALF RDO turned `alf_ctb_flag`
        // on carried a spurious bin that desynced conforming external
        // decoders (the r412 "sparse residual" corner).
        let use_aps = if cfg.sh_num_alf_aps_ids_luma > 0 {
            let use_aps_idx = init_type.min(ctxs.use_aps_flag.len() - 1);
            dec.decode_decision(&mut ctxs.use_aps_flag[use_aps_idx])? == 1
        } else {
            false
        };

        if use_aps {
            // alf_luma_prev_filter_idx — TB bypass, cMax =
            // sh_num_alf_aps_ids_luma − 1. When that is 0 the field is
            // absent and prev_idx is inferred to 0 (eq. 1437 / §7.4.12.4).
            let c_max = cfg.sh_num_alf_aps_ids_luma.saturating_sub(1) as u32;
            let prev_idx = if c_max == 0 {
                0
            } else {
                decode_tb_bypass(dec, c_max)?
            };
            // AlfCtbFiltSetIdxY = 16 + prev_idx (eq. 1439).
            ctb.luma_filt_set_idx = 16u8.saturating_add(prev_idx as u8);
        } else {
            // alf_luma_fixed_filter_idx — TB bypass cMax = 15.
            let fixed_idx = decode_tb_bypass(dec, 15)?;
            ctb.luma_filt_set_idx = fixed_idx as u8;
        }
    }

    // -- alf_ctb_flag[1] (Cb) ---------------------------------------
    if cfg.chroma_format_idc != 0 && cfg.cb_enabled {
        let cond_l_cb = nbrs.left_avail && rx > 0 && alf_pic.get(rx - 1, ry).cb_on;
        let cond_a_cb = nbrs.up_avail && ry > 0 && alf_pic.get(rx, ry - 1).cb_on;
        let inc_cb = cond_l_cb as usize + cond_a_cb as usize + 3; // cIdx = 1
        let idx_cb = init_type * 9 + inc_cb;
        let cb_on = dec.decode_decision(&mut ctxs.ctb_flag[idx_cb])? == 1;
        ctb.cb_on = cb_on;
        if cb_on && cfg.alf_chroma_num_alt_filters_minus1 > 0 {
            let alt_idx_ctx = init_type * 2; // chromaIdx = 0
            let alt = decode_alf_alt_idx_tr(
                dec,
                &mut ctxs.filter_alt_idx[alt_idx_ctx],
                cfg.alf_chroma_num_alt_filters_minus1 as u32,
            )?;
            ctb.cb_alt_idx = alt as u8;
        }
    }

    // -- alf_ctb_flag[2] (Cr) ---------------------------------------
    if cfg.chroma_format_idc != 0 && cfg.cr_enabled {
        let cond_l_cr = nbrs.left_avail && rx > 0 && alf_pic.get(rx - 1, ry).cr_on;
        let cond_a_cr = nbrs.up_avail && ry > 0 && alf_pic.get(rx, ry - 1).cr_on;
        let inc_cr = cond_l_cr as usize + cond_a_cr as usize + 6; // cIdx = 2
        let idx_cr = init_type * 9 + inc_cr;
        let cr_on = dec.decode_decision(&mut ctxs.ctb_flag[idx_cr])? == 1;
        ctb.cr_on = cr_on;
        if cr_on && cfg.alf_chroma_num_alt_filters_minus1 > 0 {
            let alt_idx_ctx = init_type * 2 + 1; // chromaIdx = 1
            let alt = decode_alf_alt_idx_tr(
                dec,
                &mut ctxs.filter_alt_idx[alt_idx_ctx],
                cfg.alf_chroma_num_alt_filters_minus1 as u32,
            )?;
            ctb.cr_alt_idx = alt as u8;
        }
    }

    // -- alf_ctb_cc_cb_idc / alf_ctb_cc_cr_idc ----------------------
    if cfg.chroma_format_idc != 0 && cfg.cc_cb_enabled {
        let cond_l = nbrs.left_avail && rx > 0 && alf_pic.get(rx - 1, ry).cc_cb_idc != 0;
        let cond_a = nbrs.up_avail && ry > 0 && alf_pic.get(rx, ry - 1).cc_cb_idc != 0;
        let inc = cond_l as usize + cond_a as usize; // ctxSetIdx = 0
        let ctx_off = init_type * 3 + inc;
        let c_max = (cfg.alf_cc_cb_filters_signalled_minus1 as u32).saturating_add(1);
        let cb_idc = decode_alf_cc_idc_tr(dec, &mut ctxs.cc_cb_idc[ctx_off], c_max)?;
        ctb.cc_cb_idc = cb_idc as u8;
    }
    if cfg.chroma_format_idc != 0 && cfg.cc_cr_enabled {
        let cond_l = nbrs.left_avail && rx > 0 && alf_pic.get(rx - 1, ry).cc_cr_idc != 0;
        let cond_a = nbrs.up_avail && ry > 0 && alf_pic.get(rx, ry - 1).cc_cr_idc != 0;
        let inc = cond_l as usize + cond_a as usize;
        let ctx_off = init_type * 3 + inc;
        let c_max = (cfg.alf_cc_cr_filters_signalled_minus1 as u32).saturating_add(1);
        let cr_idc = decode_alf_cc_idc_tr(dec, &mut ctxs.cc_cr_idc[ctx_off], c_max)?;
        ctb.cc_cr_idc = cr_idc as u8;
    }

    alf_pic.set(rx, ry, ctb);
    Ok(())
}

/// Encode the ALF block of `coding_tree_unit()` for one CTB. Reads from
/// `alf_pic` (assumed populated by the encoder's RDO pass) and emits
/// the matching CABAC bins via `enc`.
pub fn encode_alf_ctu(
    enc: &mut ArithEncoder,
    ctxs: &mut AlfCtxs,
    cfg: &AlfSyntaxConfig,
    alf_pic: &AlfPicture,
    rx: u32,
    ry: u32,
    nbrs: AlfNeighbours,
) -> Result<()> {
    if !cfg.alf_enabled {
        return Ok(());
    }
    let init_type = cfg.init_type();
    let ctb = alf_pic.get(rx, ry);

    // -- alf_ctb_flag[0] (luma) -------------------------------------
    let cond_l_y = nbrs.left_avail && rx > 0 && alf_pic.get(rx - 1, ry).luma_on;
    let cond_a_y = nbrs.up_avail && ry > 0 && alf_pic.get(rx, ry - 1).luma_on;
    let inc_y = cond_l_y as usize + cond_a_y as usize;
    let idx_y = init_type * 9 + inc_y;
    enc.encode_decision(&mut ctxs.ctb_flag[idx_y], ctb.luma_on as u32)?;

    if ctb.luma_on {
        let use_aps = ctb.luma_filt_set_idx >= 16;
        // §7.3.11.2 — the alf_use_aps_flag bin exists only when
        // `sh_num_alf_aps_ids_luma > 0`; with zero luma APS ids the
        // decoder infers 0, so an APS-referencing CTB is unencodable.
        if cfg.sh_num_alf_aps_ids_luma > 0 {
            let use_aps_idx = init_type.min(ctxs.use_aps_flag.len() - 1);
            enc.encode_decision(&mut ctxs.use_aps_flag[use_aps_idx], use_aps as u32)?;
        } else if use_aps {
            return Err(Error::invalid(
                "alf_syntax: luma_filt_set_idx >= 16 (APS filter set) requires sh_num_alf_aps_ids_luma > 0 (§7.3.11.2)",
            ));
        }
        if use_aps {
            let c_max = cfg.sh_num_alf_aps_ids_luma.saturating_sub(1) as u32;
            let prev_idx = (ctb.luma_filt_set_idx as u32).saturating_sub(16);
            if c_max > 0 {
                if prev_idx > c_max {
                    return Err(Error::invalid(format!(
                        "alf_syntax: prev_filter_idx={prev_idx} > cMax={c_max}"
                    )));
                }
                encode_tb_bypass(enc, prev_idx, c_max)?;
            }
        } else {
            let fixed_idx = ctb.luma_filt_set_idx as u32;
            if fixed_idx > 15 {
                return Err(Error::invalid(format!(
                    "alf_syntax: fixed_filter_idx={fixed_idx} > 15"
                )));
            }
            encode_tb_bypass(enc, fixed_idx, 15)?;
        }
    }

    // -- alf_ctb_flag[1] (Cb) ---------------------------------------
    if cfg.chroma_format_idc != 0 && cfg.cb_enabled {
        let cond_l_cb = nbrs.left_avail && rx > 0 && alf_pic.get(rx - 1, ry).cb_on;
        let cond_a_cb = nbrs.up_avail && ry > 0 && alf_pic.get(rx, ry - 1).cb_on;
        let inc_cb = cond_l_cb as usize + cond_a_cb as usize + 3;
        let idx_cb = init_type * 9 + inc_cb;
        enc.encode_decision(&mut ctxs.ctb_flag[idx_cb], ctb.cb_on as u32)?;
        if ctb.cb_on && cfg.alf_chroma_num_alt_filters_minus1 > 0 {
            let alt_idx_ctx = init_type * 2;
            encode_alf_alt_idx_tr(
                enc,
                &mut ctxs.filter_alt_idx[alt_idx_ctx],
                ctb.cb_alt_idx as u32,
                cfg.alf_chroma_num_alt_filters_minus1 as u32,
            )?;
        }
    }

    // -- alf_ctb_flag[2] (Cr) ---------------------------------------
    if cfg.chroma_format_idc != 0 && cfg.cr_enabled {
        let cond_l_cr = nbrs.left_avail && rx > 0 && alf_pic.get(rx - 1, ry).cr_on;
        let cond_a_cr = nbrs.up_avail && ry > 0 && alf_pic.get(rx, ry - 1).cr_on;
        let inc_cr = cond_l_cr as usize + cond_a_cr as usize + 6;
        let idx_cr = init_type * 9 + inc_cr;
        enc.encode_decision(&mut ctxs.ctb_flag[idx_cr], ctb.cr_on as u32)?;
        if ctb.cr_on && cfg.alf_chroma_num_alt_filters_minus1 > 0 {
            let alt_idx_ctx = init_type * 2 + 1;
            encode_alf_alt_idx_tr(
                enc,
                &mut ctxs.filter_alt_idx[alt_idx_ctx],
                ctb.cr_alt_idx as u32,
                cfg.alf_chroma_num_alt_filters_minus1 as u32,
            )?;
        }
    }

    // -- alf_ctb_cc_cb_idc / alf_ctb_cc_cr_idc ----------------------
    if cfg.chroma_format_idc != 0 && cfg.cc_cb_enabled {
        let cond_l = nbrs.left_avail && rx > 0 && alf_pic.get(rx - 1, ry).cc_cb_idc != 0;
        let cond_a = nbrs.up_avail && ry > 0 && alf_pic.get(rx, ry - 1).cc_cb_idc != 0;
        let inc = cond_l as usize + cond_a as usize;
        let ctx_off = init_type * 3 + inc;
        let c_max = (cfg.alf_cc_cb_filters_signalled_minus1 as u32).saturating_add(1);
        encode_alf_cc_idc_tr(
            enc,
            &mut ctxs.cc_cb_idc[ctx_off],
            ctb.cc_cb_idc as u32,
            c_max,
        )?;
    }
    if cfg.chroma_format_idc != 0 && cfg.cc_cr_enabled {
        let cond_l = nbrs.left_avail && rx > 0 && alf_pic.get(rx - 1, ry).cc_cr_idc != 0;
        let cond_a = nbrs.up_avail && ry > 0 && alf_pic.get(rx, ry - 1).cc_cr_idc != 0;
        let inc = cond_l as usize + cond_a as usize;
        let ctx_off = init_type * 3 + inc;
        let c_max = (cfg.alf_cc_cr_filters_signalled_minus1 as u32).saturating_add(1);
        encode_alf_cc_idc_tr(
            enc,
            &mut ctxs.cc_cr_idc[ctx_off],
            ctb.cc_cr_idc as u32,
            c_max,
        )?;
    }
    Ok(())
}

/// Encode every CTB of `alf_pic` in raster order through `enc`. The
/// neighbour-availability flags are derived from the picture-internal
/// CTB grid (left available iff `rx > 0`, up available iff `ry > 0`),
/// matching the foundation single-slice / single-tile assumption.
///
/// This is a thin wrapper over [`encode_alf_ctu`] used by the encoder
/// pipeline + the round-trip integration tests.
pub fn encode_alf_picture(
    enc: &mut ArithEncoder,
    ctxs: &mut AlfCtxs,
    cfg: &AlfSyntaxConfig,
    alf_pic: &AlfPicture,
) -> Result<()> {
    if !cfg.alf_enabled {
        return Ok(());
    }
    let w = alf_pic.pic_width_in_ctbs_y;
    let h = alf_pic.pic_height_in_ctbs_y;
    for ry in 0..h {
        for rx in 0..w {
            let nbrs = AlfNeighbours {
                left_avail: rx > 0,
                up_avail: ry > 0,
            };
            encode_alf_ctu(enc, ctxs, cfg, alf_pic, rx, ry, nbrs)?;
        }
    }
    Ok(())
}

/// Decode every CTB of an `AlfPicture` in raster order. Mirror of
/// [`encode_alf_picture`].
pub fn decode_alf_picture(
    dec: &mut ArithDecoder<'_>,
    ctxs: &mut AlfCtxs,
    cfg: &AlfSyntaxConfig,
    alf_pic: &mut AlfPicture,
) -> Result<()> {
    if !cfg.alf_enabled {
        return Ok(());
    }
    let w = alf_pic.pic_width_in_ctbs_y;
    let h = alf_pic.pic_height_in_ctbs_y;
    for ry in 0..h {
        for rx in 0..w {
            let nbrs = AlfNeighbours {
                left_avail: rx > 0,
                up_avail: ry > 0,
            };
            decode_alf_ctu(dec, ctxs, cfg, alf_pic, rx, ry, nbrs)?;
        }
    }
    Ok(())
}

/// `alf_ctb_filter_alt_idx` decode — TR(`cMax = num_alt_minus1`,
/// cRice = 0). Per Table 132 every TR bin shares the same ctxIdx.
fn decode_alf_alt_idx_tr(
    dec: &mut ArithDecoder<'_>,
    ctx: &mut ContextModel,
    c_max: u32,
) -> Result<u32> {
    if c_max == 0 {
        return Ok(0);
    }
    let mut v = 0u32;
    while v < c_max {
        let b = dec.decode_decision(ctx)?;
        if b == 0 {
            return Ok(v);
        }
        v += 1;
    }
    Ok(c_max)
}

/// Encode the TR bin sequence for `alf_ctb_filter_alt_idx`.
fn encode_alf_alt_idx_tr(
    enc: &mut ArithEncoder,
    ctx: &mut ContextModel,
    value: u32,
    c_max: u32,
) -> Result<()> {
    if c_max == 0 {
        return Ok(());
    }
    if value > c_max {
        return Err(Error::invalid(format!(
            "alf_syntax: alt_idx value {value} > cMax {c_max}"
        )));
    }
    // Emit `value` MPS=1 bins, then a 0 if `value < c_max`.
    for _ in 0..value {
        enc.encode_decision(ctx, 1)?;
    }
    if value < c_max {
        enc.encode_decision(ctx, 0)?;
    }
    Ok(())
}

/// `alf_ctb_cc_cb_idc` / `alf_ctb_cc_cr_idc` decode — TR with bin 0
/// context-coded (ctx 0..2 per `ctxSetIdx = 0` row of Table 132) and
/// bins 1.. bypass-coded.
fn decode_alf_cc_idc_tr(
    dec: &mut ArithDecoder<'_>,
    ctx: &mut ContextModel,
    c_max: u32,
) -> Result<u32> {
    if c_max == 0 {
        return Ok(0);
    }
    let bin0 = dec.decode_decision(ctx)?;
    if bin0 == 0 {
        return Ok(0);
    }
    let mut v = 1u32;
    while v < c_max {
        let b = dec.decode_bypass()?;
        if b == 0 {
            return Ok(v);
        }
        v += 1;
    }
    Ok(c_max)
}

/// Encode the TR bin sequence for `alf_ctb_cc_*_idc`.
fn encode_alf_cc_idc_tr(
    enc: &mut ArithEncoder,
    ctx: &mut ContextModel,
    value: u32,
    c_max: u32,
) -> Result<()> {
    if c_max == 0 {
        if value != 0 {
            return Err(Error::invalid(format!(
                "alf_syntax: cc_idc {value} but cMax = 0"
            )));
        }
        return Ok(());
    }
    if value > c_max {
        return Err(Error::invalid(format!(
            "alf_syntax: cc_idc {value} > cMax {c_max}"
        )));
    }
    if value == 0 {
        enc.encode_decision(ctx, 0)?;
        return Ok(());
    }
    enc.encode_decision(ctx, 1)?;
    // value in 1..=cMax. Emit (value-1) MPS=1 bypass bins, then 0 if
    // value < cMax.
    for _ in 1..value {
        enc.encode_bypass(1)?;
    }
    if value < c_max {
        enc.encode_bypass(0)?;
    }
    Ok(())
}

/// TB binarisation decode (§9.3.3.6) with `cRice = 0`. The TB tree
/// binarises 0..=cMax into a prefix of `floor(log2(cMax+1))` or
/// `ceil(log2(cMax+1))` bypass bins; values < `(1 << k) - cMax - 1` use
/// the short prefix and the rest use the long. For ALF this is the
/// path used by `alf_luma_fixed_filter_idx` (cMax = 15 → fixed 4-bit
/// FL) and `alf_luma_prev_filter_idx`.
fn decode_tb_bypass(dec: &mut ArithDecoder<'_>, c_max: u32) -> Result<u32> {
    if c_max == 0 {
        return Ok(0);
    }
    // §9.3.3.6: k = ceil(log2(cMax + 1)), u = (1 << k) - cMax - 1.
    let k = ceil_log2(c_max as u64 + 1);
    let u = (1u32 << k) - c_max - 1;
    if k == 0 {
        return Ok(0);
    }
    // Short prefix is k-1 bits; if value < u → short, else long (k bits)
    // and value = (short_value << 1 | bit) - u.
    let short = dec.decode_bypass_bits(k - 1)?;
    if short < u {
        return Ok(short);
    }
    let extra = dec.decode_bypass()?;
    Ok((short << 1 | extra) - u)
}

/// TB binarisation encode (§9.3.3.6) with `cRice = 0`. Mirror inverse
/// of `decode_tb_bypass`.
fn encode_tb_bypass(enc: &mut ArithEncoder, value: u32, c_max: u32) -> Result<()> {
    if c_max == 0 {
        if value != 0 {
            return Err(Error::invalid(format!(
                "alf_syntax: tb value {value} but cMax = 0"
            )));
        }
        return Ok(());
    }
    if value > c_max {
        return Err(Error::invalid(format!(
            "alf_syntax: tb value {value} > cMax {c_max}"
        )));
    }
    let k = ceil_log2(c_max as u64 + 1);
    if k == 0 {
        return Ok(());
    }
    let u = (1u32 << k) - c_max - 1;
    if value < u {
        // Short: k-1 bits
        for i in (0..k - 1).rev() {
            enc.encode_bypass((value >> i) & 1)?;
        }
    } else {
        // Long: k bits = (value + u)
        let coded = value + u;
        for i in (0..k).rev() {
            enc.encode_bypass((coded >> i) & 1)?;
        }
    }
    Ok(())
}

/// `Ceil(Log2(x))` for `x >= 1` — local copy used by the TB binariser.
fn ceil_log2(x: u64) -> u32 {
    debug_assert!(x >= 1);
    if x == 1 {
        0
    } else {
        64 - (x - 1).leading_zeros()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alf::AlfCtb;
    use crate::cabac::ArithDecoder;
    use crate::cabac_enc::ArithEncoder;

    fn pad(bytes: Vec<u8>) -> Vec<u8> {
        let mut out = bytes;
        out.extend_from_slice(&[0u8; 32]);
        out
    }

    fn cfg_default() -> AlfSyntaxConfig {
        AlfSyntaxConfig {
            alf_enabled: true,
            cb_enabled: true,
            cr_enabled: true,
            cc_cb_enabled: true,
            cc_cr_enabled: true,
            sh_num_alf_aps_ids_luma: 1,
            alf_chroma_num_alt_filters_minus1: 0,
            alf_cc_cb_filters_signalled_minus1: 0,
            alf_cc_cr_filters_signalled_minus1: 0,
            chroma_format_idc: 1,
            slice_type: SliceType::I,
            sh_cabac_init_flag: false,
        }
    }

    /// Round-45 — `AlfCtxs::init` allocates each table at its full
    /// per-syntax-element ctxIdx count.
    #[test]
    fn alf_ctxs_init_sizes() {
        let c = AlfCtxs::init(26);
        assert_eq!(c.ctb_flag.len(), 27);
        assert_eq!(c.use_aps_flag.len(), 3);
        assert_eq!(c.cc_cb_idc.len(), 9);
        assert_eq!(c.cc_cr_idc.len(), 9);
        assert_eq!(c.filter_alt_idx.len(), 6);
    }

    /// Round-45 — `cfg_default` selects initType 0 for I slices
    /// regardless of `sh_cabac_init_flag` (eq. 1527).
    #[test]
    fn init_type_derivation() {
        let mut cfg = cfg_default();
        assert_eq!(cfg.init_type(), 0);
        cfg.sh_cabac_init_flag = true;
        assert_eq!(cfg.init_type(), 0);

        cfg.slice_type = SliceType::P;
        cfg.sh_cabac_init_flag = false;
        assert_eq!(cfg.init_type(), 1);
        cfg.sh_cabac_init_flag = true;
        assert_eq!(cfg.init_type(), 2);

        cfg.slice_type = SliceType::B;
        cfg.sh_cabac_init_flag = false;
        assert_eq!(cfg.init_type(), 2);
        cfg.sh_cabac_init_flag = true;
        assert_eq!(cfg.init_type(), 1);
    }

    /// Round-45 — encoding a single all-off CTB and decoding it back
    /// must reproduce the same `AlfCtb`. This exercises the
    /// ctx-coded `alf_ctb_flag[0/1/2] = 0` path with no neighbours.
    #[test]
    fn ctu_round_trip_all_off() {
        let cfg = cfg_default();
        let mut enc = ArithEncoder::new();
        let mut enc_ctxs = AlfCtxs::init(26);
        let alf_pic = AlfPicture::empty(1, 1);
        // All zeros — the default AlfPicture is already off.
        encode_alf_ctu(
            &mut enc,
            &mut enc_ctxs,
            &cfg,
            &alf_pic,
            0,
            0,
            AlfNeighbours::default(),
        )
        .unwrap();
        let bytes = pad(enc.finish());

        let mut dec = ArithDecoder::new(&bytes).unwrap();
        let mut dec_ctxs = AlfCtxs::init(26);
        let mut dec_pic = AlfPicture::empty(1, 1);
        decode_alf_ctu(
            &mut dec,
            &mut dec_ctxs,
            &cfg,
            &mut dec_pic,
            0,
            0,
            AlfNeighbours::default(),
        )
        .unwrap();
        let dec_ctb = dec_pic.get(0, 0);
        let enc_ctb = alf_pic.get(0, 0);
        assert_eq!(dec_ctb.luma_on, enc_ctb.luma_on);
        assert_eq!(dec_ctb.cb_on, enc_ctb.cb_on);
        assert_eq!(dec_ctb.cr_on, enc_ctb.cr_on);
        assert_eq!(dec_ctb.cc_cb_idc, enc_ctb.cc_cb_idc);
        assert_eq!(dec_ctb.cc_cr_idc, enc_ctb.cc_cr_idc);
    }

    /// Round-45 — encode a CTB with luma ON / fixed filter idx 7 / Cb ON
    /// + alt 0 / CC-Cb idc 1 / CC-Cr idc 0 and decode it back.
    #[test]
    fn ctu_round_trip_mixed_components() {
        let mut cfg = cfg_default();
        cfg.alf_chroma_num_alt_filters_minus1 = 1;
        cfg.alf_cc_cb_filters_signalled_minus1 = 1;
        cfg.alf_cc_cr_filters_signalled_minus1 = 1;
        let mut enc = ArithEncoder::new();
        let mut enc_ctxs = AlfCtxs::init(26);
        let mut alf_pic = AlfPicture::empty(1, 1);
        let ctb = AlfCtb {
            luma_on: true,
            cb_on: true,
            cr_on: false,
            luma_filt_set_idx: 7, // fixed filter idx 7
            cb_alt_idx: 1,
            cr_alt_idx: 0,
            cc_cb_idc: 1,
            cc_cr_idc: 0,
        };
        alf_pic.set(0, 0, ctb);
        encode_alf_ctu(
            &mut enc,
            &mut enc_ctxs,
            &cfg,
            &alf_pic,
            0,
            0,
            AlfNeighbours::default(),
        )
        .unwrap();
        let bytes = pad(enc.finish());

        let mut dec = ArithDecoder::new(&bytes).unwrap();
        let mut dec_ctxs = AlfCtxs::init(26);
        let mut dec_pic = AlfPicture::empty(1, 1);
        decode_alf_ctu(
            &mut dec,
            &mut dec_ctxs,
            &cfg,
            &mut dec_pic,
            0,
            0,
            AlfNeighbours::default(),
        )
        .unwrap();
        let dec_ctb = dec_pic.get(0, 0);
        assert_eq!(dec_ctb.luma_on, ctb.luma_on);
        assert_eq!(dec_ctb.cb_on, ctb.cb_on);
        assert_eq!(dec_ctb.cr_on, ctb.cr_on);
        assert_eq!(dec_ctb.luma_filt_set_idx, ctb.luma_filt_set_idx);
        assert_eq!(dec_ctb.cb_alt_idx, ctb.cb_alt_idx);
        assert_eq!(dec_ctb.cc_cb_idc, ctb.cc_cb_idc);
        assert_eq!(dec_ctb.cc_cr_idc, ctb.cc_cr_idc);
    }

    /// Round-45 — encode a CTB using an APS-signalled filter set
    /// (`luma_filt_set_idx = 16 + prev_idx`) and round-trip it. With
    /// `sh_num_alf_aps_ids_luma = 2` the prev_idx field is one TB bin
    /// (cMax = 1).
    #[test]
    fn ctu_round_trip_aps_filter() {
        let mut cfg = cfg_default();
        cfg.sh_num_alf_aps_ids_luma = 2;
        cfg.cb_enabled = false;
        cfg.cr_enabled = false;
        cfg.cc_cb_enabled = false;
        cfg.cc_cr_enabled = false;
        let mut enc = ArithEncoder::new();
        let mut enc_ctxs = AlfCtxs::init(26);
        let mut alf_pic = AlfPicture::empty(1, 1);
        let ctb = AlfCtb {
            luma_on: true,
            luma_filt_set_idx: 17, // 16 + prev_idx = 1
            ..AlfCtb::default()
        };
        alf_pic.set(0, 0, ctb);
        encode_alf_ctu(
            &mut enc,
            &mut enc_ctxs,
            &cfg,
            &alf_pic,
            0,
            0,
            AlfNeighbours::default(),
        )
        .unwrap();
        let bytes = pad(enc.finish());

        let mut dec = ArithDecoder::new(&bytes).unwrap();
        let mut dec_ctxs = AlfCtxs::init(26);
        let mut dec_pic = AlfPicture::empty(1, 1);
        decode_alf_ctu(
            &mut dec,
            &mut dec_ctxs,
            &cfg,
            &mut dec_pic,
            0,
            0,
            AlfNeighbours::default(),
        )
        .unwrap();
        let dec_ctb = dec_pic.get(0, 0);
        assert!(dec_ctb.luma_on);
        assert_eq!(dec_ctb.luma_filt_set_idx, 17);
    }

    /// Round-45 — encode/decode multiple CTBs in raster order so the
    /// neighbour-availability ctxInc derivation gets exercised. The
    /// second CTB has `left_avail = true`; the ctxInc for its luma
    /// flag depends on the first CTB's `luma_on`.
    #[test]
    fn ctu_round_trip_multi_ctb_with_neighbours() {
        let cfg = cfg_default();
        let mut enc = ArithEncoder::new();
        let mut enc_ctxs = AlfCtxs::init(26);
        let mut enc_pic = AlfPicture::empty(2, 1);
        // CTB(0,0): luma on, fixed filter idx 0, all chroma off.
        enc_pic.set(
            0,
            0,
            AlfCtb {
                luma_on: true,
                luma_filt_set_idx: 0,
                ..AlfCtb::default()
            },
        );
        // CTB(1,0): luma off — exercises the cond_l = true ctxInc path.
        enc_pic.set(1, 0, AlfCtb::default());

        let nbrs0 = AlfNeighbours {
            left_avail: false,
            up_avail: false,
        };
        let nbrs1 = AlfNeighbours {
            left_avail: true,
            up_avail: false,
        };
        encode_alf_ctu(&mut enc, &mut enc_ctxs, &cfg, &enc_pic, 0, 0, nbrs0).unwrap();
        encode_alf_ctu(&mut enc, &mut enc_ctxs, &cfg, &enc_pic, 1, 0, nbrs1).unwrap();
        let bytes = pad(enc.finish());

        let mut dec = ArithDecoder::new(&bytes).unwrap();
        let mut dec_ctxs = AlfCtxs::init(26);
        let mut dec_pic = AlfPicture::empty(2, 1);
        decode_alf_ctu(&mut dec, &mut dec_ctxs, &cfg, &mut dec_pic, 0, 0, nbrs0).unwrap();
        decode_alf_ctu(&mut dec, &mut dec_ctxs, &cfg, &mut dec_pic, 1, 0, nbrs1).unwrap();

        assert!(dec_pic.get(0, 0).luma_on);
        assert!(!dec_pic.get(1, 0).luma_on);
    }

    /// Round-45 — `decode_alf_ctu` with `cfg.alf_enabled == false` must
    /// be a no-op (no bins consumed, alf_pic untouched).
    #[test]
    fn decode_no_op_when_alf_disabled() {
        let mut cfg = cfg_default();
        cfg.alf_enabled = false;
        let bytes = vec![0u8; 32];
        let mut dec = ArithDecoder::new(&bytes).unwrap();
        let mut ctxs = AlfCtxs::init(26);
        // The decode call takes &mut alf_pic, but with alf_enabled=false
        // it returns immediately without writing — clippy still flags
        // the binding as needing `mut` because of the &mut argument.
        #[allow(unused_mut)]
        let mut alf_pic = AlfPicture::empty(1, 1);
        decode_alf_ctu(
            &mut dec,
            &mut ctxs,
            &cfg,
            &mut alf_pic,
            0,
            0,
            AlfNeighbours::default(),
        )
        .unwrap();
        // Default off.
        let ctb = alf_pic.get(0, 0);
        assert!(!ctb.luma_on);
        assert!(!ctb.cb_on);
        assert!(!ctb.cr_on);
    }

    /// Round-45 — TB bypass round-trip across a representative range.
    /// Targets the `alf_luma_fixed_filter_idx` (cMax = 15) decode/encode
    /// pair specifically.
    #[test]
    fn tb_bypass_round_trip_cmax_15() {
        for v in 0..=15u32 {
            let mut enc = ArithEncoder::new();
            encode_tb_bypass(&mut enc, v, 15).unwrap();
            let bytes = pad(enc.finish());
            let mut dec = ArithDecoder::new(&bytes).unwrap();
            let got = decode_tb_bypass(&mut dec, 15).unwrap();
            assert_eq!(got, v, "TB(cMax=15) round-trip failed at v={v}");
        }
    }

    /// Round-45 — TB bypass round-trip with cMax = 3 (non-power-of-two
    /// boundary). cMax = 3 → k = 2, u = 1 → values 0 use 1-bit prefix,
    /// values 1..=3 use 2-bit prefix.
    #[test]
    fn tb_bypass_round_trip_cmax_3() {
        for v in 0..=3u32 {
            let mut enc = ArithEncoder::new();
            encode_tb_bypass(&mut enc, v, 3).unwrap();
            let bytes = pad(enc.finish());
            let mut dec = ArithDecoder::new(&bytes).unwrap();
            let got = decode_tb_bypass(&mut dec, 3).unwrap();
            assert_eq!(got, v, "TB(cMax=3) round-trip failed at v={v}");
        }
    }

    /// Round-45 — full-picture round-trip on a 3x2 grid with mixed
    /// luma / chroma / CC-ALF decisions. Exercises the picture walker
    /// in [`encode_alf_picture`] / [`decode_alf_picture`] including
    /// the inter-CTB neighbour-availability ctxInc derivation.
    #[test]
    fn picture_round_trip_3x2_grid() {
        let mut cfg = cfg_default();
        cfg.alf_chroma_num_alt_filters_minus1 = 1;
        cfg.alf_cc_cb_filters_signalled_minus1 = 1;
        cfg.alf_cc_cr_filters_signalled_minus1 = 1;

        let mut alf_pic = AlfPicture::empty(3, 2);
        // Mix various decisions across the grid.
        alf_pic.set(
            0,
            0,
            AlfCtb {
                luma_on: true,
                cb_on: true,
                luma_filt_set_idx: 5,
                cb_alt_idx: 0,
                cc_cb_idc: 1,
                ..AlfCtb::default()
            },
        );
        alf_pic.set(
            1,
            0,
            AlfCtb {
                luma_on: true,
                cr_on: true,
                luma_filt_set_idx: 12,
                cr_alt_idx: 1,
                cc_cr_idc: 2,
                ..AlfCtb::default()
            },
        );
        alf_pic.set(2, 0, AlfCtb::default()); // all off
        alf_pic.set(
            0,
            1,
            AlfCtb {
                luma_on: true,
                cb_on: true,
                cr_on: true,
                luma_filt_set_idx: 0,
                cb_alt_idx: 1,
                cr_alt_idx: 0,
                cc_cb_idc: 0,
                cc_cr_idc: 1,
            },
        );
        alf_pic.set(
            1,
            1,
            AlfCtb {
                luma_on: false,
                cb_on: true,
                cb_alt_idx: 0,
                cc_cb_idc: 2,
                ..AlfCtb::default()
            },
        );
        alf_pic.set(
            2,
            1,
            AlfCtb {
                luma_on: true,
                luma_filt_set_idx: 15,
                ..AlfCtb::default()
            },
        );

        let mut enc = ArithEncoder::new();
        let mut enc_ctxs = AlfCtxs::init(26);
        encode_alf_picture(&mut enc, &mut enc_ctxs, &cfg, &alf_pic).unwrap();
        let bytes = pad(enc.finish());

        let mut dec = ArithDecoder::new(&bytes).unwrap();
        let mut dec_ctxs = AlfCtxs::init(26);
        let mut dec_pic = AlfPicture::empty(3, 2);
        decode_alf_picture(&mut dec, &mut dec_ctxs, &cfg, &mut dec_pic).unwrap();

        for ry in 0..2 {
            for rx in 0..3 {
                let e = alf_pic.get(rx, ry);
                let d = dec_pic.get(rx, ry);
                assert_eq!(d.luma_on, e.luma_on, "luma_on at ({rx},{ry})");
                assert_eq!(d.cb_on, e.cb_on, "cb_on at ({rx},{ry})");
                assert_eq!(d.cr_on, e.cr_on, "cr_on at ({rx},{ry})");
                if e.luma_on {
                    assert_eq!(
                        d.luma_filt_set_idx, e.luma_filt_set_idx,
                        "luma_filt_set_idx at ({rx},{ry})"
                    );
                }
                if e.cb_on {
                    assert_eq!(d.cb_alt_idx, e.cb_alt_idx);
                }
                if e.cr_on {
                    assert_eq!(d.cr_alt_idx, e.cr_alt_idx);
                }
                assert_eq!(d.cc_cb_idc, e.cc_cb_idc);
                assert_eq!(d.cc_cr_idc, e.cc_cr_idc);
            }
        }
    }

    /// Round-45 — `encode_alf_picture` is a no-op when `alf_enabled` is
    /// false (zero-sized output, decoder picks up nothing).
    #[test]
    fn picture_no_op_when_disabled() {
        let mut cfg = cfg_default();
        cfg.alf_enabled = false;
        let mut enc = ArithEncoder::new();
        let mut ctxs = AlfCtxs::init(26);
        let alf_pic = AlfPicture::empty(2, 2);
        encode_alf_picture(&mut enc, &mut ctxs, &cfg, &alf_pic).unwrap();
        assert_eq!(enc.committed_bits(), 0);
    }

    /// Round-45 — encoding a `cMax = 0` TB (e.g. `alf_luma_prev_filter_idx`
    /// when `sh_num_alf_aps_ids_luma == 1`) emits no bins, decoding
    /// returns 0.
    #[test]
    fn tb_bypass_cmax_zero_emits_no_bins() {
        let mut enc = ArithEncoder::new();
        encode_tb_bypass(&mut enc, 0, 0).unwrap();
        // No commits, so finish should still produce the init prefix
        // bits but no payload.
        let _ = enc.finish();
        let bytes = vec![0u8; 32];
        let mut dec = ArithDecoder::new(&bytes).unwrap();
        let v = decode_tb_bypass(&mut dec, 0).unwrap();
        assert_eq!(v, 0);
    }
}
