//! §7.3.11.6 `palette_coding()` — parse, emission mirror, and the
//! §8.4.5.3 palette-mode decoding process.
//!
//! Palette coding replaces the whole prediction + transform pipeline
//! of a CU with a small colour table: the CU signals which entries of
//! a running *predictor palette* it reuses, an optional list of newly
//! signalled entries, then a traverse-scanned index map that names one
//! table entry (or an escape sample) per position. The pieces here:
//!
//! * [`traverse_scan`] — the §6.5.3 horizontal / vertical traverse
//!   scan order arrays (eqs. 25 / 26).
//! * [`PalettePredictor`] — `PredictorPaletteSize[chType]` +
//!   `PredictorPaletteEntries[cIdx][i]` (§7.4.12.6). Reset at the
//!   first CTU of a slice or tile and on the WPP no-above arm
//!   (§9.3.2.1); stored / synchronized around WPP CTU rows
//!   (§9.3.2.6 / §9.3.2.7).
//! * [`read_palette_coding`] — the full §7.3.11.6 syntax walk: the
//!   predictor-run reuse loop, `num_signalled_palette_entries` +
//!   `new_palette_entries` (eq. 185 `CurrentPaletteEntries`),
//!   `palette_escape_val_present_flag`, `palette_transpose_flag`, the
//!   in-CU `cu_qp_delta` / `cu_chroma_qp_offset` reads, and the
//!   16-position subset loop (run_copy / copy_above passes, the
//!   §9.3.3.13 truncated-binary `palette_idx_idc` with the eq. 186 /
//!   187 `adjustedRefPaletteIndex` fold, and the EG5
//!   `palette_escape_val` pass).
//! * [`write_palette_coding`] — the encoder mirror. It derives the
//!   run structure from the target index map with the same state
//!   machine the parser runs, so any legal map round-trips bit-exactly.
//! * [`PaletteCu::reconstruct_component`] — §8.4.5.3 sample
//!   reconstruction for one colour component (eq. 438 table lookup,
//!   eqs. 439 – 443 escape dequantisation).
//! * [`update_predictor`] — the §8.4.5.3 predictor maintenance
//!   (eq. 450, the eq. 444 – 449 local-dual-tree fold, eq. 451).
//!
//! Spec reference: ITU-T H.266 | ISO/IEC 23090-3 (V4, 01/2026). The
//! implementation is spec-only; no third-party VVC decoder source was
//! consulted.

use oxideav_core::{Error, Result};

use crate::cabac::{ArithDecoder, ContextModel};
use crate::cabac_enc::ArithEncoder;
use crate::coding_tree::TreeType;
use crate::ctx::{
    ctx_inc_copy_above_palette_indices_flag, ctx_inc_palette_transpose_flag, ctx_inc_run_copy_flag,
};
use crate::residual::{
    decode_exp_golomb_k, read_cu_chroma_qp_offset, read_cu_qp_delta, ResidualCtxs,
};
use crate::residual_enc::{encode_exp_golomb_k, write_cu_chroma_qp_offset, write_cu_qp_delta};
use crate::tables::{init_contexts, SyntaxCtx};

/// `maxNumPalettePredictorSize` for SINGLE_TREE (§8.4.5.3 eq. 427) —
/// also the joint size used by the local-dual-tree fold (eq. 449).
pub const MAX_PREDICTOR_SINGLE: usize = 63;
/// `maxNumPalettePredictorSize` for a dual-tree component pass
/// (§8.4.5.3 eqs. 430 / 433).
pub const MAX_PREDICTOR_DUAL: usize = 31;
/// `maxNumPaletteEntries` for SINGLE_TREE (§7.3.11.6).
pub const MAX_ENTRIES_SINGLE: usize = 31;
/// `maxNumPaletteEntries` for either dual-tree pass (§7.3.11.6).
pub const MAX_ENTRIES_DUAL: usize = 15;

/// §6.5.3 — horizontal (`transpose == false`, eq. 25) or vertical
/// (`transpose == true`, eq. 26) traverse scan order for a
/// `bw x bh` block. Entry `[s]` is the `(x, y)` block-local position
/// of scan index `s`.
pub fn traverse_scan(bw: usize, bh: usize, transpose: bool) -> Vec<(u32, u32)> {
    let mut out = Vec::with_capacity(bw * bh);
    if !transpose {
        for y in 0..bh {
            if y % 2 == 0 {
                for x in 0..bw {
                    out.push((x as u32, y as u32));
                }
            } else {
                for x in (0..bw).rev() {
                    out.push((x as u32, y as u32));
                }
            }
        }
    } else {
        for x in 0..bw {
            if x % 2 == 0 {
                for y in 0..bh {
                    out.push((x as u32, y as u32));
                }
            } else {
                for y in (0..bh).rev() {
                    out.push((x as u32, y as u32));
                }
            }
        }
    }
    out
}

/// §7.4.12.6 — the running predictor palette.
///
/// `size[chType]` is `PredictorPaletteSize[chType]` (chType 0 keys the
/// luma-led table used by SINGLE_TREE / DUAL_TREE_LUMA, chType 1 the
/// chroma-led table used by DUAL_TREE_CHROMA); `entries[cIdx][i]` is
/// `PredictorPaletteEntries[cIdx][i]`. The three component rows are
/// shared — a dual-tree walk updates disjoint rows for the two trees.
#[derive(Clone, Debug)]
pub struct PalettePredictor {
    /// `PredictorPaletteSize[chType]`, chType ∈ {0, 1}.
    pub size: [usize; 2],
    /// `PredictorPaletteEntries[cIdx][i]`, cIdx ∈ {0, 1, 2}.
    pub entries: [[u16; MAX_PREDICTOR_SINGLE]; 3],
}

impl Default for PalettePredictor {
    fn default() -> Self {
        Self::new()
    }
}

impl PalettePredictor {
    /// Empty predictor (`PredictorPaletteSize[chType] = 0` — the
    /// §9.3.2.1 initialization at the first CTU of a slice / tile).
    pub fn new() -> Self {
        Self {
            size: [0; 2],
            entries: [[0; MAX_PREDICTOR_SINGLE]; 3],
        }
    }

    /// §9.3.2.1 — re-initialize both `PredictorPaletteSize[chType]`
    /// entries to 0.
    pub fn reset(&mut self) {
        self.size = [0; 2];
    }
}

/// CABAC context bundle for the palette syntax elements. `Clone`
/// supports the §9.3.2.3 / §9.3.2.4 WPP context storage paths.
#[derive(Clone, Debug)]
pub struct PaletteCtxs {
    /// Table 67 — `pred_mode_plt_flag` (one ctx per initType).
    pub pred_mode_plt_flag: Vec<ContextModel>,
    /// Table 100 — `palette_transpose_flag` (one ctx per initType).
    pub palette_transpose_flag: Vec<ContextModel>,
    /// Table 99 — `copy_above_palette_indices_flag` (one ctx per
    /// initType).
    pub copy_above_palette_indices_flag: Vec<ContextModel>,
    /// Table 101 — `run_copy_flag` (8 ctxs per initType, ctxInc per
    /// §9.3.4.2.11).
    pub run_copy_flag: Vec<ContextModel>,
    /// §9.3.2.2 / Table 51 initType (0 = I, 1 / 2 = P / B).
    pub init_type: u8,
}

impl PaletteCtxs {
    /// Build the palette context arrays for a slice.
    pub fn init(slice_qp_y: i32, init_type: u8) -> Self {
        Self {
            pred_mode_plt_flag: init_contexts(SyntaxCtx::PredModePltFlag, slice_qp_y),
            palette_transpose_flag: init_contexts(SyntaxCtx::PaletteTransposeFlag, slice_qp_y),
            copy_above_palette_indices_flag: init_contexts(
                SyntaxCtx::CopyAbovePaletteIndicesFlag,
                slice_qp_y,
            ),
            run_copy_flag: init_contexts(SyntaxCtx::RunCopyFlag, slice_qp_y),
            init_type,
        }
    }
}

/// A parsed palette CU bundled with the invocation parameters it was
/// parsed under — carried on [`crate::leaf_cu::LeafCuResidual`] so the
/// reconstruction pass can re-derive `startComp` / `numComps` and the
/// escape sub-sampling without replaying the CABAC reads.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PaletteCuInfo {
    /// The `palette_coding()` invocation parameters.
    pub params: PaletteParams,
    /// The parsed CU payload.
    pub cu: PaletteCu,
}

/// Invocation parameters for `palette_coding( x0, y0, cbWidth,
/// cbHeight, treeType )`. `bw` / `bh` are the block dimensions at the
/// invocation resolution — luma samples for SINGLE_TREE /
/// DUAL_TREE_LUMA, chroma samples (`cbWidth / SubWidthC`) for
/// DUAL_TREE_CHROMA (§7.3.11.5).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PaletteParams {
    /// `treeType`.
    pub tree: TreeType,
    /// `sps_chroma_format_idc` (0 = monochrome, 1 = 4:2:0).
    pub chroma_format_idc: u32,
    /// `BitDepth`.
    pub bit_depth: u32,
    /// Block width at the invocation resolution.
    pub bw: u32,
    /// Block height at the invocation resolution.
    pub bh: u32,
    /// Eq. 184 `LocalDualTreeFlag` — 1 when `treeType != SINGLE_TREE`
    /// and the slice is not an I slice with
    /// `sps_qtbtt_dual_tree_intra_flag == 1`.
    pub local_dual_tree: bool,
    /// `pps_cu_qp_delta_enabled_flag`.
    pub cu_qp_delta_enabled: bool,
    /// `IsCuQpDeltaCoded` carried in from the quantization-group state.
    pub cu_qp_delta_already_coded: bool,
    /// `sh_cu_chroma_qp_offset_enabled_flag`.
    pub cu_chroma_qp_offset_enabled: bool,
    /// `IsCuChromaQpOffsetCoded` carried in from the CU state.
    pub cu_chroma_qp_offset_already_coded: bool,
    /// `pps_chroma_qp_offset_list_len_minus1`.
    pub chroma_qp_offset_list_len_minus1: u32,
}

impl PaletteParams {
    /// `startComp` (§7.3.11.6).
    pub fn start_comp(&self) -> usize {
        if self.tree == TreeType::DualTreeChroma {
            1
        } else {
            0
        }
    }

    /// `numComps` (§7.3.11.6).
    pub fn num_comps(&self) -> usize {
        match self.tree {
            TreeType::SingleTree => {
                if self.chroma_format_idc == 0 {
                    1
                } else {
                    3
                }
            }
            TreeType::DualTreeChroma => 2,
            TreeType::DualTreeLuma => 1,
        }
    }

    /// `maxNumPaletteEntries` (§7.3.11.6).
    pub fn max_entries(&self) -> usize {
        if self.tree == TreeType::SingleTree {
            MAX_ENTRIES_SINGLE
        } else {
            MAX_ENTRIES_DUAL
        }
    }

    /// (SubWidthC, SubHeightC) — only consulted on the SINGLE_TREE
    /// escape sub-sampling gate and the §8.4.5.3 chroma pass.
    pub fn sub_wh(&self) -> (u32, u32) {
        match self.chroma_format_idc {
            1 => (2, 2),
            2 => (2, 1),
            _ => (1, 1),
        }
    }
}

/// One parsed (or planned) palette CU: everything §8.4.5.3 needs to
/// reconstruct the block and update the predictor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PaletteCu {
    /// `palette_transpose_flag`.
    pub transpose: bool,
    /// `palette_escape_val_present_flag` (inferred 1 when
    /// `CurrentPaletteSize[startComp] == 0`).
    pub escape_present: bool,
    /// `NumPredictedPaletteEntries`.
    pub num_predicted: usize,
    /// `num_signalled_palette_entries`.
    pub num_signalled: usize,
    /// `PalettePredictorEntryReuseFlags[i]` for the predictor the CU
    /// was parsed against (length = `PredictorPaletteSize[startComp]`).
    pub reuse_flags: Vec<bool>,
    /// `CurrentPaletteEntries[cIdx][i]` (eq. 185). Rows outside
    /// `startComp..startComp+numComps` are only populated for the
    /// predictor-reuse span when `LocalDualTreeFlag == 1`.
    pub entries: [[u16; MAX_ENTRIES_SINGLE]; 3],
    /// `PaletteIndexMap`, row-major `bw x bh` at the invocation
    /// resolution.
    pub index_map: Vec<u8>,
    /// `PaletteEscapeVal[cIdx]`, row-major `bw x bh`; positions that
    /// are not escape-coded (or not coded for a sub-sampled chroma
    /// component) stay 0.
    pub escape_vals: [Vec<u16>; 3],
    /// Signed `CuQpDeltaVal` when this CU carried `cu_qp_delta_abs`.
    pub cu_qp_delta_val: i32,
    /// True iff this CU read `cu_qp_delta_abs` (sets
    /// `IsCuQpDeltaCoded`).
    pub cu_qp_delta_read: bool,
    /// `cu_chroma_qp_offset_flag` (false when not present).
    pub cu_chroma_qp_offset_flag: bool,
    /// `cu_chroma_qp_offset_idx`.
    pub cu_chroma_qp_offset_idx: u32,
    /// True iff this CU read `cu_chroma_qp_offset_flag` (sets
    /// `IsCuChromaQpOffsetCoded`).
    pub cu_chroma_qp_offset_read: bool,
}

impl PaletteCu {
    /// `CurrentPaletteSize[startComp]` (eq. 183).
    pub fn current_size(&self) -> usize {
        self.num_predicted + self.num_signalled
    }

    /// `MaxPaletteIndex` (§7.4.12.6).
    pub fn max_palette_index(&self) -> usize {
        self.current_size() + usize::from(self.escape_present) - 1
    }

    /// §8.4.5.3 — reconstruct one colour component.
    ///
    /// * `c_idx` — the colour component (absolute: 0 = luma).
    /// * `n_cb_w` / `n_cb_h` — the component block size (§8.4.1 passes
    ///   `cbWidth / SubWidthC` for the chroma components).
    /// * `qp` — the component's `Qp′` (eqs. 439 – 441 apply the
    ///   `QpPrimeTsMin` floor here).
    /// * `qp_prime_ts_min` — `QpPrimeTsMin` (§7.4.3.4).
    /// * `write` — sink called as `(x, y, sample)` in component
    ///   coordinates.
    ///
    /// The index map is sampled at `x * nSubWidth` (eqs. 434 – 437):
    /// `nSubWidth / nSubHeight` are `SubWidthC / SubHeightC` only for
    /// a chroma component of a `startComp == 0` table (§8.4.5.3).
    pub fn reconstruct_component<F: FnMut(u32, u32, u16)>(
        &self,
        p: &PaletteParams,
        c_idx: usize,
        n_cb_w: u32,
        n_cb_h: u32,
        qp: i32,
        qp_prime_ts_min: i32,
        mut write: F,
    ) {
        let (sub_w, sub_h) = if c_idx > 0 && p.start_comp() == 0 {
            p.sub_wh()
        } else {
            (1, 1)
        };
        let max_idx = self.max_palette_index();
        let max_val = ((1u32 << p.bit_depth) - 1) as i32;
        // Eq. 442 levelScale.
        const LEVEL_SCALE: [i32; 6] = [40, 45, 51, 57, 64, 72];
        let q = qp.max(qp_prime_ts_min);
        let ls = LEVEL_SCALE[(q % 6) as usize];
        let shift = q / 6;
        for y in 0..n_cb_h {
            for x in 0..n_cb_w {
                let xl = (x * sub_w) as usize;
                let yl = (y * sub_h) as usize;
                let map_idx = yl * (p.bw as usize) + xl;
                let idx = self.index_map[map_idx] as usize;
                let is_escape = self.escape_present && idx == max_idx;
                let sample = if !is_escape {
                    // Eq. 438.
                    i32::from(self.entries[c_idx][idx])
                } else {
                    // Eqs. 439 – 443.
                    let v = i32::from(self.escape_vals[c_idx][map_idx]);
                    let tmp = ((v * ls) << shift).wrapping_add(32) >> 6;
                    tmp.clamp(0, max_val)
                };
                write(x, y, sample.clamp(0, max_val) as u16);
            }
        }
    }
}

/// Truncated-binary read (§9.3.3.4) — bypass bins.
fn read_tb(dec: &mut ArithDecoder<'_>, c_max: u32) -> Result<u32> {
    if c_max == 0 {
        return Ok(0);
    }
    let n = c_max + 1;
    let k = 31 - n.leading_zeros(); // Floor(Log2(n)), eq. 1539
    let u = (1u32 << (k + 1)) - n;
    let first = if k > 0 { dec.decode_bypass_bits(k)? } else { 0 };
    if first < u {
        Ok(first)
    } else {
        let ext = dec.decode_bypass()?;
        Ok(((first << 1) | ext) - u)
    }
}

/// Truncated-binary write (§9.3.3.4 inverse).
fn write_tb(enc: &mut ArithEncoder, val: u32, c_max: u32) -> Result<()> {
    if c_max == 0 {
        return Ok(());
    }
    let n = c_max + 1;
    let k = 31 - n.leading_zeros();
    let u = (1u32 << (k + 1)) - n;
    let (bits, v) = if val < u { (k, val) } else { (k + 1, val + u) };
    for b in (0..bits).rev() {
        enc.encode_bypass((v >> b) & 1)?;
    }
    Ok(())
}

/// Eq. 185 — build `CurrentPaletteEntries` from the predictor +
/// signalled entries. `reuse[i]` must be sized
/// `PredictorPaletteSize[startComp]`; `new_entries[cIdx]` carries
/// `num_signalled` values for each coded component row.
fn build_current_entries(
    pred: &PalettePredictor,
    reuse: &[bool],
    new_entries: &[Vec<u16>; 3],
    num_signalled: usize,
    p: &PaletteParams,
) -> [[u16; MAX_ENTRIES_SINGLE]; 3] {
    let start = p.start_comp();
    let ncomp = p.num_comps();
    let mut entries = [[0u16; MAX_ENTRIES_SINGLE]; 3];
    let mut n = 0usize;
    for (i, &r) in reuse.iter().enumerate() {
        if !r {
            continue;
        }
        let (c0, c1) = if p.local_dual_tree {
            (0usize, 3usize)
        } else {
            (start, start + ncomp)
        };
        for (c, row) in entries.iter_mut().enumerate().take(c1).skip(c0) {
            row[n] = pred.entries[c][i];
        }
        n += 1;
    }
    for c in start..start + ncomp {
        for i in 0..num_signalled {
            entries[c][n + i] = new_entries[c][i];
        }
    }
    entries
}

/// §7.3.11.6 — parse one `palette_coding()` body.
///
/// The caller has already consumed `pred_mode_plt_flag`. `pred` is the
/// running predictor for the slice (not modified here — predictor
/// maintenance is a decoding process, see [`update_predictor`]).
pub fn read_palette_coding(
    dec: &mut ArithDecoder<'_>,
    pctx: &mut PaletteCtxs,
    rctx: &mut ResidualCtxs,
    pred: &PalettePredictor,
    p: &PaletteParams,
) -> Result<PaletteCu> {
    let start = p.start_comp();
    let ncomp = p.num_comps();
    let max_entries = p.max_entries();
    let pred_size = pred.size[start];
    let bw = p.bw as usize;
    let bh = p.bh as usize;
    let npos = bw * bh;

    // ---- predictor reuse runs ---------------------------------------
    let mut reuse = vec![false; pred_size];
    let mut num_predicted = 0usize;
    let mut finished = false;
    let mut idx = 0usize;
    while idx < pred_size && !finished && num_predicted < max_entries {
        let run = decode_exp_golomb_k(dec, 0)? as usize;
        if run != 1 {
            if run > 1 {
                // Conformance: run <= PredictorPaletteSize - idx.
                if run > pred_size - idx {
                    return Err(Error::invalid(
                        "h266 palette_coding: palette_predictor_run exceeds the predictor",
                    ));
                }
                idx += run - 1;
            }
            reuse[idx] = true;
            num_predicted += 1;
        } else {
            finished = true;
        }
        idx += 1;
    }

    // ---- num_signalled_palette_entries + new_palette_entries --------
    let num_signalled = if num_predicted < max_entries {
        let v = decode_exp_golomb_k(dec, 0)? as usize;
        if v > max_entries - num_predicted {
            return Err(Error::invalid(
                "h266 palette_coding: num_signalled_palette_entries exceeds \
                 maxNumPaletteEntries",
            ));
        }
        v
    } else {
        0
    };
    let mut new_entries: [Vec<u16>; 3] = [vec![], vec![], vec![]];
    for row in new_entries.iter_mut().take(start + ncomp).skip(start) {
        for _ in 0..num_signalled {
            row.push(dec.decode_bypass_bits(p.bit_depth)? as u16);
        }
    }
    let entries = build_current_entries(pred, &reuse, &new_entries, num_signalled, p);
    let current_size = num_predicted + num_signalled;

    // ---- escape flag + transpose + QP side elements ------------------
    let escape_present = if current_size > 0 {
        dec.decode_bypass()? == 1
    } else {
        true // §7.4.12.6 inference
    };
    let max_palette_index = current_size + usize::from(escape_present) - 1;
    let mut adjust = 0usize;
    let transpose = if max_palette_index > 0 {
        let inc = ctx_inc_palette_transpose_flag() as usize + pctx.init_type as usize;
        let n = pctx.palette_transpose_flag.len() - 1;
        dec.decode_decision(&mut pctx.palette_transpose_flag[inc.min(n)])? == 1
    } else {
        false
    };

    let mut cu = PaletteCu {
        transpose,
        escape_present,
        num_predicted,
        num_signalled,
        reuse_flags: reuse,
        entries,
        index_map: vec![0u8; npos],
        escape_vals: [vec![0u16; npos], vec![0u16; npos], vec![0u16; npos]],
        cu_qp_delta_val: 0,
        cu_qp_delta_read: false,
        cu_chroma_qp_offset_flag: false,
        cu_chroma_qp_offset_idx: 0,
        cu_chroma_qp_offset_read: false,
    };

    if p.tree != TreeType::DualTreeChroma
        && escape_present
        && p.cu_qp_delta_enabled
        && !p.cu_qp_delta_already_coded
    {
        cu.cu_qp_delta_val = read_cu_qp_delta(dec, rctx)?;
        cu.cu_qp_delta_read = true;
    }
    if p.tree != TreeType::DualTreeLuma
        && escape_present
        && p.cu_chroma_qp_offset_enabled
        && !p.cu_chroma_qp_offset_already_coded
    {
        let (flag, idxv) = read_cu_chroma_qp_offset(dec, rctx, p.chroma_qp_offset_list_len_minus1)?;
        cu.cu_chroma_qp_offset_flag = flag;
        cu.cu_chroma_qp_offset_idx = idxv;
        cu.cu_chroma_qp_offset_read = true;
    }

    // ---- index map subsets -------------------------------------------
    let scan = traverse_scan(bw, bh, transpose);
    let (sub_w, sub_h) = p.sub_wh();
    let mut run_copy_map = vec![false; npos];
    let mut copy_above = vec![false; npos];
    let mut prev_run_pos = 0usize;
    let mut prev_run_type = false;
    let mut curr_palette_index = 0u32;
    let n_subsets = npos.div_ceil(16);
    for subset in 0..n_subsets {
        let min_sub = subset * 16;
        let max_sub = (min_sub + 16).min(npos);
        // Pass 1 — run_copy_flag / copy_above_palette_indices_flag.
        for s in min_sub..max_sub {
            let (xc, yc) = scan[s];
            let map_at = |sp: usize| -> usize {
                let (px, py) = scan[sp];
                py as usize * bw + px as usize
            };
            let cur = yc as usize * bw + xc as usize;
            let run_copy = if max_palette_index > 0 && s > 0 {
                let bin_dist = (s - prev_run_pos - 1) as u32;
                let inc = ctx_inc_run_copy_flag(prev_run_type, bin_dist) as usize
                    + pctx.init_type as usize * 8;
                let n = pctx.run_copy_flag.len() - 1;
                let b = dec.decode_decision(&mut pctx.run_copy_flag[inc.min(n)])? == 1;
                run_copy_map[cur] = b;
                b
            } else {
                false
            };
            copy_above[cur] = false;
            if max_palette_index > 0 && !run_copy {
                let row_gate = (!transpose && yc > 0) || (transpose && xc > 0);
                let prev_copy_above = s > 0 && copy_above[map_at(s - 1)];
                if row_gate && s > 0 && !prev_copy_above {
                    let inc = ctx_inc_copy_above_palette_indices_flag() as usize
                        + pctx.init_type as usize;
                    let n = pctx.copy_above_palette_indices_flag.len() - 1;
                    copy_above[cur] = dec
                        .decode_decision(&mut pctx.copy_above_palette_indices_flag[inc.min(n)])?
                        == 1;
                }
                prev_run_type = copy_above[cur];
                prev_run_pos = s;
            } else if s > 0 {
                copy_above[cur] = copy_above[map_at(s - 1)];
            }
        }
        // Pass 2 — palette_idx_idc + PaletteIndexMap.
        for s in min_sub..max_sub {
            let (xc, yc) = scan[s];
            let cur = yc as usize * bw + xc as usize;
            if max_palette_index > 0 && !run_copy_map[cur] && !copy_above[cur] {
                let c_max = (max_palette_index - adjust) as u32;
                let raw = if c_max > 0 { read_tb(dec, c_max)? } else { 0 };
                adjust = 1;
                // Eqs. 186 / 187 — fold adjustedRefPaletteIndex.
                let adjusted_ref = adjusted_ref_palette_index(
                    &scan,
                    &cu.index_map,
                    &copy_above,
                    s,
                    bw,
                    transpose,
                    max_palette_index,
                );
                curr_palette_index = if raw >= adjusted_ref { raw + 1 } else { raw };
            }
            if !copy_above[cur] {
                cu.index_map[cur] = curr_palette_index as u8;
            } else if !transpose {
                cu.index_map[cur] = cu.index_map[cur - bw];
            } else {
                cu.index_map[cur] = cu.index_map[cur - 1];
            }
        }
        // Pass 3 — palette_escape_val.
        if escape_present {
            for c in start..start + ncomp {
                for s in min_sub..max_sub {
                    let (xc, yc) = scan[s];
                    let cur = yc as usize * bw + xc as usize;
                    let sub_sampled = p.tree == TreeType::SingleTree
                        && c != 0
                        && (xc % sub_w != 0 || yc % sub_h != 0);
                    if sub_sampled {
                        continue;
                    }
                    if cu.index_map[cur] as usize == max_palette_index {
                        let v = decode_exp_golomb_k(dec, 5)?;
                        if v >= (1u32 << p.bit_depth) {
                            // `H266_DBG_LENIENT` — triage aid: clamp a
                            // desynced escape value instead of aborting so
                            // the picture can still be diffed.
                            if std::env::var_os("H266_DBG_LENIENT").is_none() {
                                return Err(Error::invalid(
                                    "h266 palette_coding: palette_escape_val exceeds \
                                     (1 << BitDepth) - 1",
                                ));
                            }
                        }
                        let v = v.min((1u32 << p.bit_depth) - 1);
                        cu.escape_vals[c][cur] = v as u16;
                    }
                }
            }
        }
    }
    Ok(cu)
}

/// Eq. 186 — `adjustedRefPaletteIndex` at scan position `s`.
fn adjusted_ref_palette_index(
    scan: &[(u32, u32)],
    index_map: &[u8],
    copy_above: &[bool],
    s: usize,
    bw: usize,
    transpose: bool,
    max_palette_index: usize,
) -> u32 {
    if s == 0 {
        return (max_palette_index + 1) as u32;
    }
    let (px, py) = scan[s - 1];
    let prev = py as usize * bw + px as usize;
    if !copy_above[prev] {
        u32::from(index_map[prev])
    } else {
        let (xc, yc) = scan[s];
        let cur = yc as usize * bw + xc as usize;
        if !transpose {
            u32::from(index_map[cur - bw])
        } else {
            u32::from(index_map[cur - 1])
        }
    }
}

/// Encoder-side plan for one palette CU: the desired palette
/// composition and per-position content. The run structure is derived
/// by [`write_palette_coding`] with the same state machine the parser
/// runs.
#[derive(Clone, Debug)]
pub struct PalettePlan {
    /// Predictor entries to reuse (length must equal
    /// `PredictorPaletteSize[startComp]`).
    pub reuse: Vec<bool>,
    /// Newly signalled entries per component row
    /// (`startComp..startComp+numComps` rows must each carry
    /// `num_signalled` values).
    pub new_entries: [Vec<u16>; 3],
    /// `palette_escape_val_present_flag` to emit.
    pub escape_present: bool,
    /// `palette_transpose_flag` to emit.
    pub transpose: bool,
    /// Target `PaletteIndexMap` (row-major `bw x bh`).
    pub index_map: Vec<u8>,
    /// Quantized escape values per component row (row-major; consulted
    /// only at escape positions).
    pub escape_vals: [Vec<u16>; 3],
    /// `cu_qp_delta` to emit when the gate opens (signed value).
    pub cu_qp_delta: i32,
    /// `cu_chroma_qp_offset_flag` / idx to emit when the gate opens.
    pub cu_chroma_qp_offset_flag: bool,
    pub cu_chroma_qp_offset_idx: u32,
}

/// §7.3.11.6 — emit one `palette_coding()` body (the exact mirror of
/// [`read_palette_coding`]). Returns the [`PaletteCu`] the decoder
/// will parse, so the caller can run the shared §8.4.5.3
/// reconstruction + predictor maintenance.
pub fn write_palette_coding(
    enc: &mut ArithEncoder,
    pctx: &mut PaletteCtxs,
    rctx: &mut ResidualCtxs,
    pred: &PalettePredictor,
    p: &PaletteParams,
    plan: &PalettePlan,
) -> Result<PaletteCu> {
    let start = p.start_comp();
    let ncomp = p.num_comps();
    let max_entries = p.max_entries();
    let pred_size = pred.size[start];
    if plan.reuse.len() != pred_size {
        return Err(Error::invalid(
            "h266 palette emit: reuse mask length != PredictorPaletteSize",
        ));
    }
    let bw = p.bw as usize;
    let bh = p.bh as usize;
    let npos = bw * bh;
    if plan.index_map.len() != npos {
        return Err(Error::invalid("h266 palette emit: index map size mismatch"));
    }
    let num_predicted = plan.reuse.iter().filter(|&&r| r).count();
    let num_signalled = plan.new_entries[start].len();
    if num_predicted + num_signalled > max_entries || num_predicted + num_signalled == 0 {
        return Err(Error::invalid(
            "h266 palette emit: illegal CurrentPaletteSize",
        ));
    }

    // ---- predictor reuse runs ----------------------------------------
    // The syntax codes, per reused entry, the distance from the prior
    // reused entry (+1 past the first — `palette_predictor_run` counts
    // the zeros before the next 1); a trailing run of exactly 1 stops
    // the loop early when reuse ends before the predictor does.
    {
        let mut prev = None::<usize>;
        let mut emitted = 0usize;
        for (i, &r) in plan.reuse.iter().enumerate() {
            if !r {
                continue;
            }
            let zeros = match prev {
                None => i,
                Some(pi) => i - pi - 1,
            };
            // run == 0 → reuse at the current scan index; run > 1 →
            // skip run − 1. A gap of `zeros` needs run = zeros == 0 ?
            // 0 : zeros + 1 (run == 1 would terminate).
            let run = if zeros == 0 { 0 } else { zeros + 1 };
            encode_exp_golomb_k(enc, run as u32, 0)?;
            prev = Some(i);
            emitted += 1;
        }
        // Terminating run (palette_predictor_run == 1) — only when the
        // parser's loop would still be live: more predictor entries
        // remain after the last reused one and the entry cap is not
        // reached.
        let next_idx = prev.map(|v| v + 1).unwrap_or(0);
        if next_idx < pred_size && emitted < max_entries {
            encode_exp_golomb_k(enc, 1, 0)?;
        }
    }

    // ---- num_signalled + new entries -----------------------------------
    if num_predicted < max_entries {
        encode_exp_golomb_k(enc, num_signalled as u32, 0)?;
    } else if num_signalled != 0 {
        return Err(Error::invalid(
            "h266 palette emit: cannot signal entries at maxNumPaletteEntries",
        ));
    }
    for c in start..start + ncomp {
        if plan.new_entries[c].len() != num_signalled {
            return Err(Error::invalid(
                "h266 palette emit: new_entries component row size mismatch",
            ));
        }
        for &v in &plan.new_entries[c] {
            if u32::from(v) >= (1u32 << p.bit_depth) {
                return Err(Error::invalid("h266 palette emit: entry exceeds BitDepth"));
            }
            for b in (0..p.bit_depth).rev() {
                enc.encode_bypass((u32::from(v) >> b) & 1)?;
            }
        }
    }
    let entries = build_current_entries(pred, &plan.reuse, &plan.new_entries, num_signalled, p);
    let current_size = num_predicted + num_signalled;

    // ---- escape flag + transpose + QP side elements ---------------------
    // current_size > 0 is guaranteed above.
    enc.encode_bypass(u32::from(plan.escape_present))?;
    let max_palette_index = current_size + usize::from(plan.escape_present) - 1;
    let mut adjust = 0usize;
    if max_palette_index > 0 {
        let inc = ctx_inc_palette_transpose_flag() as usize + pctx.init_type as usize;
        let n = pctx.palette_transpose_flag.len() - 1;
        enc.encode_decision(
            &mut pctx.palette_transpose_flag[inc.min(n)],
            u32::from(plan.transpose),
        )?;
    } else if plan.transpose {
        return Err(Error::invalid(
            "h266 palette emit: transpose requires MaxPaletteIndex > 0",
        ));
    }

    let mut cu = PaletteCu {
        transpose: plan.transpose,
        escape_present: plan.escape_present,
        num_predicted,
        num_signalled,
        reuse_flags: plan.reuse.clone(),
        entries,
        index_map: plan.index_map.clone(),
        escape_vals: [vec![0u16; npos], vec![0u16; npos], vec![0u16; npos]],
        cu_qp_delta_val: 0,
        cu_qp_delta_read: false,
        cu_chroma_qp_offset_flag: false,
        cu_chroma_qp_offset_idx: 0,
        cu_chroma_qp_offset_read: false,
    };

    if p.tree != TreeType::DualTreeChroma
        && plan.escape_present
        && p.cu_qp_delta_enabled
        && !p.cu_qp_delta_already_coded
    {
        write_cu_qp_delta(enc, rctx, plan.cu_qp_delta)?;
        cu.cu_qp_delta_val = plan.cu_qp_delta;
        cu.cu_qp_delta_read = true;
    }
    if p.tree != TreeType::DualTreeLuma
        && plan.escape_present
        && p.cu_chroma_qp_offset_enabled
        && !p.cu_chroma_qp_offset_already_coded
    {
        write_cu_chroma_qp_offset(
            enc,
            rctx,
            plan.cu_chroma_qp_offset_flag,
            plan.cu_chroma_qp_offset_idx,
            p.chroma_qp_offset_list_len_minus1,
        )?;
        cu.cu_chroma_qp_offset_flag = plan.cu_chroma_qp_offset_flag;
        cu.cu_chroma_qp_offset_idx = plan.cu_chroma_qp_offset_idx;
        cu.cu_chroma_qp_offset_read = true;
    }

    // ---- index map subsets ----------------------------------------------
    let scan = traverse_scan(bw, bh, plan.transpose);
    let (sub_w, sub_h) = p.sub_wh();
    let map = &plan.index_map;
    let above_of = |cur: usize| -> Option<usize> {
        if !plan.transpose {
            (cur >= bw).then(|| cur - bw)
        } else {
            (cur % bw > 0).then(|| cur - 1)
        }
    };
    let mut copy_above = vec![false; npos];
    let mut run_copy_map = vec![false; npos];
    let mut prev_run_pos = 0usize;
    let mut prev_run_type = false;
    let n_subsets = npos.div_ceil(16);
    let mut sim_map = vec![0u8; npos]; // decoder-visible map replay
    let mut curr_palette_index = 0u32;
    for subset in 0..n_subsets {
        let min_sub = subset * 16;
        let max_sub = (min_sub + 16).min(npos);
        // Pass 1 — choose + emit run_copy / copy_above.
        for s in min_sub..max_sub {
            let (xc, yc) = scan[s];
            let cur = yc as usize * bw + xc as usize;
            let prev = if s > 0 {
                let (px, py) = scan[s - 1];
                Some(py as usize * bw + px as usize)
            } else {
                None
            };
            // What would continuing the previous run reproduce?
            let continuation_legal = match prev {
                Some(pi) if max_palette_index > 0 => {
                    if copy_above[pi] {
                        // Continuing a COPY_ABOVE run: legal when an
                        // above neighbour exists and matches.
                        above_of(cur).is_some_and(|a| map[a] == map[cur])
                    } else {
                        // Continuing an INDEX run: same index as prev.
                        map[pi] == map[cur]
                    }
                }
                _ => false,
            };
            let row_gate = (!plan.transpose && yc > 0) || (plan.transpose && xc > 0);
            let prev_copy_above = prev.map(|pi| copy_above[pi]).unwrap_or(false);
            let copy_above_flag_present = row_gate && s > 0 && !prev_copy_above;
            let copy_above_legal =
                copy_above_flag_present && above_of(cur).is_some_and(|a| map[a] == map[cur]);
            let run_copy = continuation_legal;
            if max_palette_index > 0 && s > 0 {
                let bin_dist = (s - prev_run_pos - 1) as u32;
                let inc = ctx_inc_run_copy_flag(prev_run_type, bin_dist) as usize
                    + pctx.init_type as usize * 8;
                let n = pctx.run_copy_flag.len() - 1;
                enc.encode_decision(&mut pctx.run_copy_flag[inc.min(n)], u32::from(run_copy))?;
                run_copy_map[cur] = run_copy;
            }
            copy_above[cur] = false;
            if max_palette_index > 0 && !run_copy {
                if copy_above_flag_present {
                    let inc = ctx_inc_copy_above_palette_indices_flag() as usize
                        + pctx.init_type as usize;
                    let n = pctx.copy_above_palette_indices_flag.len() - 1;
                    enc.encode_decision(
                        &mut pctx.copy_above_palette_indices_flag[inc.min(n)],
                        u32::from(copy_above_legal),
                    )?;
                    copy_above[cur] = copy_above_legal;
                }
                prev_run_type = copy_above[cur];
                prev_run_pos = s;
            } else if s > 0 {
                let (px, py) = scan[s - 1];
                copy_above[cur] = copy_above[py as usize * bw + px as usize];
            }
        }
        // Pass 2 — palette_idx_idc.
        for s in min_sub..max_sub {
            let (xc, yc) = scan[s];
            let cur = yc as usize * bw + xc as usize;
            if max_palette_index > 0 && !run_copy_map[cur] && !copy_above[cur] {
                let target = u32::from(map[cur]);
                let adjusted_ref = adjusted_ref_palette_index(
                    &scan,
                    &sim_map,
                    &copy_above,
                    s,
                    bw,
                    plan.transpose,
                    max_palette_index,
                );
                if target == adjusted_ref {
                    return Err(Error::invalid(
                        "h266 palette emit: explicit index equals adjustedRefPaletteIndex \
                         (the run structure must absorb it)",
                    ));
                }
                let raw = if target > adjusted_ref {
                    target - 1
                } else {
                    target
                };
                let c_max = (max_palette_index - adjust) as u32;
                if raw > c_max {
                    return Err(Error::invalid(
                        "h266 palette emit: palette_idx_idc exceeds its cMax",
                    ));
                }
                if c_max > 0 {
                    write_tb(enc, raw, c_max)?;
                }
                adjust = 1;
                curr_palette_index = target;
            }
            if !copy_above[cur] {
                sim_map[cur] = curr_palette_index as u8;
            } else if !plan.transpose {
                sim_map[cur] = sim_map[cur - bw];
            } else {
                sim_map[cur] = sim_map[cur - 1];
            }
            if sim_map[cur] != map[cur] {
                return Err(Error::invalid(
                    "h266 palette emit: replay diverged from the target index map",
                ));
            }
        }
        // Pass 3 — palette_escape_val.
        if plan.escape_present {
            for c in start..start + ncomp {
                for s in min_sub..max_sub {
                    let (xc, yc) = scan[s];
                    let cur = yc as usize * bw + xc as usize;
                    let sub_sampled = p.tree == TreeType::SingleTree
                        && c != 0
                        && (xc % sub_w != 0 || yc % sub_h != 0);
                    if sub_sampled {
                        continue;
                    }
                    if map[cur] as usize == max_palette_index {
                        let v = plan.escape_vals[c].get(cur).copied().ok_or_else(|| {
                            Error::invalid("h266 palette emit: escape value row too short")
                        })?;
                        encode_exp_golomb_k(enc, u32::from(v), 5)?;
                        cu.escape_vals[c][cur] = v;
                    }
                }
            }
        }
    }
    Ok(cu)
}

/// §8.4.5.3 — predictor palette maintenance after a palette CU
/// completes.
///
/// * `dual_tree_structure` — true when the CTU walk actually splits
///   into separate luma / chroma trees (I slice with
///   `sps_qtbtt_dual_tree_intra_flag == 1`); eq. 451 mirrors
///   `PredictorPaletteSize[1]` from the joint update whenever this is
///   false.
///
/// A DUAL_TREE_CHROMA CU under `LocalDualTreeFlag == 1` performs no
/// update (none of the §8.4.5.3 trigger conditions hold).
pub fn update_predictor(
    pred: &mut PalettePredictor,
    cu: &PaletteCu,
    p: &PaletteParams,
    dual_tree_structure: bool,
) {
    if p.tree == TreeType::DualTreeChroma && p.local_dual_tree {
        return;
    }
    // Post-trigger overrides (eqs. 444 – 449).
    let (start, ncomp, max_pred) = if p.local_dual_tree {
        // DUAL_TREE_LUMA with LocalDualTreeFlag: the newly signalled
        // entries acquire mid-grey chroma rows (eqs. 444 / 445) and
        // the update runs jointly over all three components.
        (0usize, 3usize, MAX_PREDICTOR_SINGLE)
    } else {
        (
            p.start_comp(),
            p.num_comps(),
            if p.tree == TreeType::SingleTree {
                MAX_PREDICTOR_SINGLE
            } else {
                MAX_PREDICTOR_DUAL
            },
        )
    };
    let mut current = cu.entries;
    if p.local_dual_tree && p.tree == TreeType::DualTreeLuma {
        let mid = 1u16 << (p.bit_depth - 1);
        for i in 0..cu.num_signalled {
            current[1][cu.num_predicted + i] = mid;
            current[2][cu.num_predicted + i] = mid;
        }
    }
    let current_size = cu.current_size();
    // Eq. 450.
    let mut new_entries = [[0u16; MAX_PREDICTOR_SINGLE]; 3];
    for (c, row) in new_entries
        .iter_mut()
        .enumerate()
        .take(start + ncomp)
        .skip(start)
    {
        for i in 0..current_size {
            row[i] = current[c][i];
        }
    }
    let mut new_size = current_size;
    let old_start = if p.tree == TreeType::DualTreeChroma {
        1
    } else {
        0
    };
    let old_size = pred.size[old_start];
    for i in 0..old_size {
        if new_size >= max_pred {
            break;
        }
        let reused = cu.reuse_flags.get(i).copied().unwrap_or(false);
        if !reused {
            for (c, row) in new_entries
                .iter_mut()
                .enumerate()
                .take(start + ncomp)
                .skip(start)
            {
                row[new_size] = pred.entries[c][i];
            }
            new_size += 1;
        }
    }
    for c in start..start + ncomp {
        pred.entries[c][..new_size].copy_from_slice(&new_entries[c][..new_size]);
    }
    pred.size[if p.tree == TreeType::DualTreeChroma {
        1
    } else {
        0
    }] = new_size;
    // Eq. 451 — non-dual-tree structures keep the chType 1 size in
    // lock-step with the joint predictor.
    if !dual_tree_structure {
        pred.size[1] = new_size;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cabac::ArithDecoder;
    use crate::cabac_enc::ArithEncoder;

    fn params_single(bw: u32, bh: u32) -> PaletteParams {
        PaletteParams {
            tree: TreeType::SingleTree,
            chroma_format_idc: 1,
            bit_depth: 8,
            bw,
            bh,
            local_dual_tree: false,
            cu_qp_delta_enabled: false,
            cu_qp_delta_already_coded: false,
            cu_chroma_qp_offset_enabled: false,
            cu_chroma_qp_offset_already_coded: false,
            chroma_qp_offset_list_len_minus1: 0,
        }
    }

    fn roundtrip(p: &PaletteParams, pred: &PalettePredictor, plan: &PalettePlan) -> PaletteCu {
        let mut enc = ArithEncoder::new();
        let mut pctx_e = PaletteCtxs::init(26, 0);
        let mut rctx_e = ResidualCtxs::init(26);
        let emitted =
            write_palette_coding(&mut enc, &mut pctx_e, &mut rctx_e, pred, p, plan).expect("emit");
        enc.encode_terminate(1).expect("terminate");
        let bs = enc.finish();
        let mut dec = ArithDecoder::new(&bs).expect("dec");
        let mut pctx_d = PaletteCtxs::init(26, 0);
        let mut rctx_d = ResidualCtxs::init(26);
        let parsed =
            read_palette_coding(&mut dec, &mut pctx_d, &mut rctx_d, pred, p).expect("parse");
        assert_eq!(parsed.index_map, emitted.index_map, "index map mismatch");
        assert_eq!(parsed.transpose, emitted.transpose);
        assert_eq!(parsed.escape_present, emitted.escape_present);
        assert_eq!(parsed.num_predicted, emitted.num_predicted);
        assert_eq!(parsed.num_signalled, emitted.num_signalled);
        assert_eq!(parsed.entries, emitted.entries);
        assert_eq!(parsed.escape_vals, emitted.escape_vals);
        assert_eq!(dec.decode_terminate().expect("term bin"), 1);
        parsed
    }

    fn plan_from_map(map: Vec<u8>, entries: &[(u16, u16, u16)]) -> PalettePlan {
        PalettePlan {
            reuse: vec![],
            new_entries: [
                entries.iter().map(|e| e.0).collect(),
                entries.iter().map(|e| e.1).collect(),
                entries.iter().map(|e| e.2).collect(),
            ],
            escape_present: false,
            transpose: false,
            index_map: map,
            escape_vals: [vec![], vec![], vec![]],
            cu_qp_delta: 0,
            cu_chroma_qp_offset_flag: false,
            cu_chroma_qp_offset_idx: 0,
        }
    }

    #[test]
    fn traverse_scan_snakes_rows_and_columns() {
        // 4x2 horizontal: row 0 L→R, row 1 R→L (eq. 25).
        let h = traverse_scan(4, 2, false);
        assert_eq!(
            h,
            vec![
                (0, 0),
                (1, 0),
                (2, 0),
                (3, 0),
                (3, 1),
                (2, 1),
                (1, 1),
                (0, 1)
            ]
        );
        // 2x4 vertical: column 0 T→B, column 1 B→T (eq. 26).
        let v = traverse_scan(2, 4, true);
        assert_eq!(
            v,
            vec![
                (0, 0),
                (0, 1),
                (0, 2),
                (0, 3),
                (1, 3),
                (1, 2),
                (1, 1),
                (1, 0)
            ]
        );
    }

    #[test]
    fn tb_binarization_round_trips_all_values() {
        // §9.3.3.4 at a non-power-of-two cMax: n = 6, k = 2, u = 2.
        for c_max in 1u32..=32 {
            for v in 0..=c_max {
                let mut enc = ArithEncoder::new();
                write_tb(&mut enc, v, c_max).unwrap();
                enc.encode_terminate(1).unwrap();
                let bs = enc.finish();
                let mut dec = ArithDecoder::new(&bs).unwrap();
                assert_eq!(read_tb(&mut dec, c_max).unwrap(), v, "cMax {c_max} v {v}");
            }
        }
    }

    #[test]
    fn two_colour_checker_round_trips() {
        // 8x8 two-colour vertical stripes — INDEX runs + COPY_ABOVE rows.
        let mut map = vec![0u8; 64];
        for y in 0..8 {
            for x in 0..8 {
                map[y * 8 + x] = (x / 2 % 2) as u8;
            }
        }
        let p = params_single(8, 8);
        let pred = PalettePredictor::new();
        let plan = plan_from_map(map, &[(200, 100, 50), (30, 140, 220)]);
        let cu = roundtrip(&p, &pred, &plan);
        assert_eq!(cu.current_size(), 2);
        assert_eq!(cu.max_palette_index(), 1);
    }

    #[test]
    fn transpose_scan_round_trips() {
        // Horizontal stripes favour the vertical traverse scan.
        let mut map = vec![0u8; 64];
        for y in 0..8 {
            for x in 0..8 {
                map[y * 8 + x] = (y % 4) as u8;
            }
        }
        let p = params_single(8, 8);
        let pred = PalettePredictor::new();
        let mut plan = plan_from_map(
            map,
            &[(10, 20, 30), (40, 50, 60), (70, 80, 90), (100, 110, 120)],
        );
        plan.transpose = true;
        roundtrip(&p, &pred, &plan);
    }

    #[test]
    fn escape_positions_round_trip_with_subsampled_chroma() {
        // 3 palette colours + escapes on a diagonal; SINGLE_TREE 4:2:0
        // codes chroma escapes only at even/even positions.
        let bw = 8usize;
        let mut map = vec![0u8; 64];
        for y in 0..8 {
            for x in 0..8 {
                map[y * 8 + x] = if x == y { 3 } else { (x % 3) as u8 };
            }
        }
        let mut escape_vals = [vec![0u16; 64], vec![0u16; 64], vec![0u16; 64]];
        for y in 0..8usize {
            let cur = y * bw + y;
            escape_vals[0][cur] = (40 + y) as u16;
            if y % 2 == 0 {
                escape_vals[1][cur] = (10 + y) as u16;
                escape_vals[2][cur] = (20 + y) as u16;
            }
        }
        let p = params_single(8, 8);
        let pred = PalettePredictor::new();
        let mut plan = plan_from_map(map, &[(1, 2, 3), (4, 5, 6), (7, 8, 9)]);
        plan.escape_present = true;
        plan.escape_vals = escape_vals;
        let cu = roundtrip(&p, &pred, &plan);
        assert_eq!(cu.max_palette_index(), 3);
        // Escape dequant at qP 4 (< QpPrimeTsMin floor exercises the Max).
        let mut luma = [0u16; 64];
        cu.reconstruct_component(&p, 0, 8, 8, 4, 4, |x, y, s| {
            luma[y as usize * 8 + x as usize] = s;
        });
        // Eq. 442 at qP = 4: (v * 64 + 32) >> 6 = v (levelScale[4] = 64).
        assert_eq!(luma[0], 40); // escape (0,0): coded value 40
        assert_eq!(luma[9], 41); // escape (1,1): 40 + 1
        assert_eq!(luma[1], 4); // entry (x % 3 == 1) luma
    }

    #[test]
    fn predictor_reuse_and_update_round_trip() {
        // Seed a predictor, reuse a scattered subset, verify eq. 450.
        let mut pred = PalettePredictor::new();
        pred.size = [5, 5];
        for i in 0..5 {
            pred.entries[0][i] = (i as u16) * 10;
            pred.entries[1][i] = (i as u16) * 10 + 1;
            pred.entries[2][i] = (i as u16) * 10 + 2;
        }
        let mut map = vec![0u8; 64];
        for (i, m) in map.iter_mut().enumerate() {
            *m = (i % 3) as u8;
        }
        let p = params_single(8, 8);
        let plan = PalettePlan {
            reuse: vec![true, false, true, false, false],
            new_entries: [vec![99], vec![98], vec![97]],
            escape_present: false,
            transpose: false,
            index_map: map,
            escape_vals: [vec![], vec![], vec![]],
            cu_qp_delta: 0,
            cu_chroma_qp_offset_flag: false,
            cu_chroma_qp_offset_idx: 0,
        };
        let cu = roundtrip(&p, &pred, &plan);
        // CurrentPaletteEntries: predictor 0, predictor 2, new 99.
        assert_eq!(cu.entries[0][0], 0);
        assert_eq!(cu.entries[0][1], 20);
        assert_eq!(cu.entries[0][2], 99);
        assert_eq!(cu.entries[1][2], 98);
        update_predictor(&mut pred, &cu, &p, false);
        // Eq. 450: current palette first, then unused predictor
        // entries 1, 3, 4.
        assert_eq!(pred.size, [6, 6]);
        assert_eq!(pred.entries[0][..6], [0, 20, 99, 10, 30, 40]);
        assert_eq!(pred.entries[1][..6], [1, 21, 98, 11, 31, 41]);
        assert_eq!(pred.entries[2][..6], [2, 22, 97, 12, 32, 42]);
    }

    #[test]
    fn predictor_update_caps_at_max_size() {
        let mut pred = PalettePredictor::new();
        pred.size = [63, 63];
        for i in 0..63 {
            pred.entries[0][i] = i as u16;
            pred.entries[1][i] = i as u16;
            pred.entries[2][i] = i as u16;
        }
        let p = params_single(8, 8);
        let plan = PalettePlan {
            reuse: {
                let mut r = vec![false; 63];
                r[62] = true;
                r
            },
            new_entries: [vec![200], vec![200], vec![200]],
            escape_present: false,
            transpose: false,
            index_map: vec![0u8; 64],
            escape_vals: [vec![], vec![], vec![]],
            cu_qp_delta: 0,
            cu_chroma_qp_offset_flag: false,
            cu_chroma_qp_offset_idx: 0,
        };
        let cu = roundtrip(&p, &pred, &plan);
        update_predictor(&mut pred, &cu, &p, false);
        assert_eq!(pred.size[0], 63);
        assert_eq!(pred.entries[0][0], 62);
        assert_eq!(pred.entries[0][1], 200);
        // Unused entries 0.. fill the remainder, capped at 63.
        assert_eq!(pred.entries[0][2], 0);
        assert_eq!(pred.entries[0][62], 60);
    }

    #[test]
    fn all_escape_block_when_palette_empty() {
        // CurrentPaletteSize == 0: escape flag inferred 1, no index
        // bins, everything escape-coded (MaxPaletteIndex == 0).
        let p = params_single(4, 4);
        let pred = PalettePredictor::new();
        let mut enc = ArithEncoder::new();
        let mut pctx = PaletteCtxs::init(26, 0);
        let rctx = ResidualCtxs::init(26);
        // Emit by hand: no predictor runs (size 0), num_signalled = 0,
        // escape flag not coded (size 0 → inferred), no transpose, no
        // run flags — just 16 luma + 4+4 chroma EG5 values.
        encode_exp_golomb_k(&mut enc, 0, 0).unwrap(); // num_signalled
        let mut expect = [vec![0u16; 16], vec![0u16; 16], vec![0u16; 16]];
        for s in 0..16 {
            let (x, y) = traverse_scan(4, 4, false)[s];
            let cur = (y * 4 + x) as usize;
            expect[0][cur] = (s + 1) as u16;
            encode_exp_golomb_k(&mut enc, (s + 1) as u32, 5).unwrap();
        }
        for c in 1..3usize {
            for s in 0..16 {
                let (x, y) = traverse_scan(4, 4, false)[s];
                if x % 2 != 0 || y % 2 != 0 {
                    continue;
                }
                let cur = (y * 4 + x) as usize;
                expect[c][cur] = (100 + c) as u16;
                encode_exp_golomb_k(&mut enc, (100 + c) as u32, 5).unwrap();
            }
        }
        enc.encode_terminate(1).unwrap();
        let bs = enc.finish();
        let mut dec = ArithDecoder::new(&bs).unwrap();
        let mut rctx_d = ResidualCtxs::init(26);
        let cu = read_palette_coding(&mut dec, &mut pctx, &mut rctx_d, &pred, &p).expect("parse");
        assert!(cu.escape_present);
        assert_eq!(cu.current_size(), 0);
        assert_eq!(cu.max_palette_index(), 0);
        assert_eq!(cu.escape_vals, expect);
        let _ = rctx;
    }

    #[test]
    fn dual_tree_chroma_palette_round_trips() {
        // DUAL_TREE_CHROMA: two components, chroma-resolution block,
        // escapes coded for both components at every escape position.
        let p = PaletteParams {
            tree: TreeType::DualTreeChroma,
            chroma_format_idc: 1,
            bit_depth: 8,
            bw: 8,
            bh: 8,
            local_dual_tree: false,
            cu_qp_delta_enabled: false,
            cu_qp_delta_already_coded: false,
            cu_chroma_qp_offset_enabled: false,
            cu_chroma_qp_offset_already_coded: false,
            chroma_qp_offset_list_len_minus1: 0,
        };
        let mut pred = PalettePredictor::new();
        let mut map = vec![0u8; 64];
        for (i, m) in map.iter_mut().enumerate() {
            *m = ((i / 8) % 2) as u8;
        }
        let plan = PalettePlan {
            reuse: vec![],
            new_entries: [vec![], vec![60, 190], vec![190, 60]],
            escape_present: false,
            transpose: false,
            index_map: map,
            escape_vals: [vec![], vec![], vec![]],
            cu_qp_delta: 0,
            cu_chroma_qp_offset_flag: false,
            cu_chroma_qp_offset_idx: 0,
        };
        let cu = roundtrip(&p, &pred, &plan);
        assert_eq!(cu.entries[1][0], 60);
        assert_eq!(cu.entries[2][1], 60);
        update_predictor(&mut pred, &cu, &p, true);
        // DUAL_TREE_CHROMA updates chType 1 only.
        assert_eq!(pred.size, [0, 2]);
        assert_eq!(pred.entries[1][..2], [60, 190]);
    }

    #[test]
    fn single_index_block_codes_no_index_bins() {
        // One palette entry, no escape: MaxPaletteIndex == 0 — the
        // whole index map is inferred (no run/copy/idc bins at all).
        let p = params_single(4, 4);
        let pred = PalettePredictor::new();
        let plan = plan_from_map(vec![0u8; 16], &[(77, 88, 99)]);
        let cu = roundtrip(&p, &pred, &plan);
        assert_eq!(cu.max_palette_index(), 0);
        assert!(cu.index_map.iter().all(|&v| v == 0));
    }

    #[test]
    fn escape_dequant_matches_eq442() {
        // qP = 13: levelScale[1] = 45, shift = 2.
        let p = params_single(4, 4);
        let mut map = vec![0u8; 16];
        map[0] = 1; // escape (entry count 1 + escape → maxIdx 1)
        let mut plan = plan_from_map(map, &[(50, 60, 70)]);
        plan.escape_present = true;
        let mut ev = [vec![0u16; 16], vec![0u16; 16], vec![0u16; 16]];
        ev[0][0] = 7;
        ev[1][0] = 3;
        ev[2][0] = 5;
        plan.escape_vals = ev;
        let pred = PalettePredictor::new();
        let cu = roundtrip(&p, &pred, &plan);
        let mut luma = [0u16; 16];
        cu.reconstruct_component(&p, 0, 4, 4, 13, 4, |x, y, s| {
            luma[y as usize * 4 + x as usize] = s;
        });
        // ((7 * 45) << 2) + 32 >> 6 = (1260 + 32) >> 6 = 20.
        assert_eq!(luma[0], 20);
        assert_eq!(luma[1], 50);
        // Chroma at (0,0): ((3 * 45) << 2) + 32 >> 6 = 572 >> 6 = 8.
        let mut cb = [0u16; 4];
        cu.reconstruct_component(&p, 1, 2, 2, 13, 4, |x, y, s| {
            cb[y as usize * 2 + x as usize] = s;
        });
        assert_eq!(cb[0], 8);
        assert_eq!(cb[1], 60);
    }
}
