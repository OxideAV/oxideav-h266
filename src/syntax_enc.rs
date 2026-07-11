//! VVC `coding_tree_unit()` / `coding_tree()` / `coding_quadtree()` /
//! `coding_unit()` syntax shells — encoder side (round 52).
//!
//! Per ITU-T H.266 §7.3.11 the bitstream layout for one CTU is
//!
//! ```text
//! coding_tree_unit() {
//!     // (alf_ctb_flag[], sao_*, sao_merge_*: emitted by their own
//!     //  encoders before this shell is invoked.)
//!     coding_tree(x0, y0, log2CbSize, qgOnY=1, qgOnC=1, ...)
//! }
//! coding_tree(...) {
//!     split_cu_flag
//!     if (split_cu_flag) {
//!         split_qt_flag
//!         if (split_qt_flag) coding_quadtree(...)
//!         else { mtt_split_cu_vertical_flag, mtt_split_cu_binary_flag,
//!                multi_type_tree(...) }
//!     } else {
//!         coding_unit(...)
//!     }
//! }
//! coding_unit(...) {
//!     // I-slice intra-only scope: cu_skip_flag, pred_mode_flag NOT
//!     // signalled (they are derived to 0 / MODE_INTRA per §7.4.11.5).
//!     // Intra-mode cascade + transform_tree() are emitted inline. The
//!     // round-52 shell defers the intra-mode cascade to the body
//!     // (round-44 emits the implicit DC fallback inside the existing
//!     // `emit_tu_with_cbf`); the `cu_qp_delta` emit is also handled
//!     // there since its gate (`pps_cu_qp_delta_enabled_flag` + any
//!     // CBF non-zero) lives inside transform_unit().
//! }
//! ```
//!
//! ## Round-52 scope
//!
//! Round-52 wires the **shell calls** without implementing the recursive
//! splitting cases. The encoder pipeline today emits each TB block (up to
//! 64×64 luma) as a single CU, so the shell collapses to:
//!
//! * `encode_coding_tree_unit_intra_iframe()` — outer wrapper that emits
//!   `split_cu_flag = 0` + `coding_unit_intra_iframe()` for the single
//!   CU. The ALF / SAO bins remain the responsibility of the caller
//!   (already wired by the encoder pipeline).
//! * `encode_coding_unit_intra_iframe()` — emits the I-slice intra-only
//!   `coding_unit()` body which today consists of just the
//!   `transform_tree()` content (via the supplied `body_emit` closure).
//!
//! The recursive split path (`split_cu_flag = 1` → `split_qt_flag` →
//! `coding_quadtree()`) is implemented for future rounds via
//! [`encode_coding_tree_split_qt_iframe`] but unused by the round-52
//! pipeline.
//!
//! Decoder counterparts are exposed for round-trip tests:
//! [`decode_coding_tree_split_cu_flag`] reads the `split_cu_flag` bin
//! using the matching ctxInc.
//!
//! Spec reference: ITU-T H.266 | ISO/IEC 23090-3 (V4, 01/2026).

use oxideav_core::Result;

use crate::cabac::ArithDecoder;
use crate::cabac_enc::ArithEncoder;
use crate::coding_tree::{SplitAllows, TreeCtxs};
use crate::ctx::{
    ctx_inc_mtt_split_cu_binary_flag, ctx_inc_mtt_split_cu_vertical_flag, ctx_inc_split_cu_flag,
    ctx_inc_split_qt_flag,
};

/// Encode the I-slice `coding_tree_unit()` body for a single 64×64 CU per
/// CTU (round-52 scope: no actual splitting). Emits one `split_cu_flag = 0`
/// bin then invokes `body_emit` to write the `coding_unit()` body
/// (transform_tree → transform_unit → residual).
///
/// Per §7.3.11.4 the `split_cu_flag` is signalled at the root of every
/// CU that is *eligible* to split (size > MinCbSize and at least one
/// allowed split mode). For a 64×64 CU at the maximum CB size with all
/// split modes allowed the bin must be coded; we emit `0` to indicate
/// the CU is a leaf.
///
/// `nbrs` describe the L/A neighbour availability for the §9.3.4.2.2
/// `condL` / `condA` ctxInc derivation. For a single-CU CTU the root
/// neighbours are the picture edges → both `false`.
///
/// # Arguments
/// * `cb_w` / `cb_h` — current CB dimensions in luma samples (64×64 in
///   the round-52 scope).
/// * `nbrs` — left / above neighbour availability + their CB sizes (used
///   only by the §9.3.4.2.2 ctxInc — not consumed when neighbours are
///   unavailable).
/// * `body_emit` — closure that emits the `coding_unit()` body (CBFs +
///   `cu_qp_delta` + residual). The shell takes a closure so the caller
///   can keep its own ctxs / shared state across CUs.
pub fn encode_coding_tree_leaf_iframe<F>(
    enc: &mut ArithEncoder,
    ctxs: &mut TreeCtxs,
    cb_w: u32,
    cb_h: u32,
    nbrs: TreeNeighbours,
    allows: SplitAllows,
    body_emit: F,
) -> Result<()>
where
    F: FnOnce(&mut ArithEncoder) -> Result<()>,
{
    // Emit split_cu_flag = 0 (leaf CU) — r412: present ONLY when at
    // least one §6.4.1 – §6.4.3 split family is allowed at this node
    // (§7.3.11.4; when absent the decoder infers 0), and the
    // §9.3.4.2.2 ctxInc consumes the REAL allow flags.
    if allows.any() {
        let inc = ctx_inc_split_cu_flag(
            nbrs.left_avail,
            nbrs.above_avail,
            nbrs.cb_height_left,
            nbrs.cb_width_above,
            cb_w,
            cb_h,
            allows.bt_ver as u32,
            allows.bt_hor as u32,
            allows.tt_ver as u32,
            allows.tt_hor as u32,
            allows.qt as u32,
        ) as usize;
        let n = ctxs.split_cu.len() - 1;
        enc.encode_decision(&mut ctxs.split_cu[inc.min(n)], 0)?;
    }

    // coding_unit() body — caller-provided. In I-slice intra-only scope
    // this is just transform_tree() (cu_skip_flag / pred_mode_flag are
    // not signalled in I-slice per §7.3.11.5).
    body_emit(enc)
}

/// Encode the `split_qt_flag` bin per §7.3.11.4 / §9.3.4.2.2. Used by the
/// recursive split path; round-52 encoder pipeline does not exercise
/// this directly but the helper is exposed for future rounds.
pub fn encode_split_qt_flag(
    enc: &mut ArithEncoder,
    ctxs: &mut TreeCtxs,
    split: bool,
    nbrs: TreeNeighbours,
    cqt_depth: u32,
) -> Result<()> {
    let inc = ctx_inc_split_qt_flag(
        nbrs.left_avail,
        nbrs.above_avail,
        nbrs.cqt_depth_left,
        nbrs.cqt_depth_above,
        cqt_depth,
    ) as usize;
    let n = ctxs.split_qt.len() - 1;
    enc.encode_decision(&mut ctxs.split_qt[inc.min(n)], split as u32)
}

/// Encode a `coding_quadtree()` shell that emits a forced 4-way QT
/// split: `split_cu_flag = 1` + `split_qt_flag = 1` + 4× recursion via
/// `body_emit_quadrant(enc, ctxs, quadrant_idx, sub_w, sub_h)`.
///
/// Used when the current CU exceeds MaxBtSize (e.g. a 128×128 CTB
/// requires a forced QT split because MaxCbSizeY = 64 in our profile).
/// Round-55 encoder pipeline wires this for 128×128 CTBs: the emitted
/// four 64×64 sub-CU shells follow inside `body_emit_quadrant`. The
/// closure receives `(enc, ctxs)` so the per-quadrant body can call
/// back into [`encode_coding_tree_leaf_iframe`] without needing to
/// share the `TreeCtxs` borrow with the caller.
pub fn encode_coding_quadtree_split<F>(
    enc: &mut ArithEncoder,
    ctxs: &mut TreeCtxs,
    cb_w: u32,
    cb_h: u32,
    nbrs: TreeNeighbours,
    cqt_depth: u32,
    allows: SplitAllows,
    mut body_emit_quadrant: F,
) -> Result<()>
where
    F: FnMut(&mut ArithEncoder, &mut TreeCtxs, u32, u32, u32) -> Result<()>,
{
    // split_cu_flag = 1 — real §6.4.1 – §6.4.3 allow flags feed the
    // §9.3.4.2.2 ctxInc (r412).
    let inc = ctx_inc_split_cu_flag(
        nbrs.left_avail,
        nbrs.above_avail,
        nbrs.cb_height_left,
        nbrs.cb_width_above,
        cb_w,
        cb_h,
        allows.bt_ver as u32,
        allows.bt_hor as u32,
        allows.tt_ver as u32,
        allows.tt_hor as u32,
        allows.qt as u32,
    ) as usize;
    let n = ctxs.split_cu.len() - 1;
    enc.encode_decision(&mut ctxs.split_cu[inc.min(n)], 1)?;
    // split_qt_flag = 1 — §7.3.11.4: the bin is coded only when a QT
    // split AND at least one MTT split are both allowed; otherwise a
    // conforming decoder infers it (r412 — the pre-r412 emitter always
    // wrote the bin, desyncing QT-only nodes).
    if allows.qt && allows.any_mtt() {
        encode_split_qt_flag(enc, ctxs, true, nbrs, cqt_depth)?;
    }
    let hw = cb_w / 2;
    let hh = cb_h / 2;
    for q in 0..4u32 {
        body_emit_quadrant(enc, ctxs, q, hw, hh)?;
    }
    Ok(())
}

/// Round-55 — direction selector for an MTT split.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MttSplitDir {
    /// Vertical split (`mtt_split_cu_vertical_flag = 1`).
    Vertical,
    /// Horizontal split (`mtt_split_cu_vertical_flag = 0`).
    Horizontal,
}

impl MttSplitDir {
    fn vertical_flag(self) -> u32 {
        match self {
            MttSplitDir::Vertical => 1,
            MttSplitDir::Horizontal => 0,
        }
    }
}

/// Round-55 — encode a `coding_tree()` MTT binary-tree (BT) split:
///   * `split_cu_flag = 1`
///   * `split_qt_flag = 0`
///   * `mtt_split_cu_vertical_flag` per `dir`
///   * `mtt_split_cu_binary_flag = 1`
///   * 2× recursion via `body_emit_subblock(idx, sub_w, sub_h)` (idx ∈ {0, 1}).
///
/// `mtt_depth` is the current MTT depth (= 0 for the first MTT split
/// inside a quadtree leaf). Used by the §9.3.4.2.1 / Table 132 ctxInc
/// derivation for the binary flag.
///
/// All four allow-split-* flags are passed `1` (no allowance restriction
/// in the round-55 scope where every dir is available); §9.3.4.2.3
/// collapses this to ctxInc 0 because vert_sum == hor_sum and the
/// neighbour-aspect ratio is degenerate for picture-edge CUs.
pub fn encode_coding_tree_bt_split<F>(
    enc: &mut ArithEncoder,
    ctxs: &mut TreeCtxs,
    cb_w: u32,
    cb_h: u32,
    nbrs: TreeNeighbours,
    cqt_depth: u32,
    mtt_depth: u32,
    dir: MttSplitDir,
    allows: SplitAllows,
    mut body_emit_subblock: F,
) -> Result<()>
where
    F: FnMut(&mut ArithEncoder, &mut TreeCtxs, u32, u32, u32) -> Result<()>,
{
    // split_cu_flag = 1 (r412 — real allow flags in the ctxInc).
    let inc = ctx_inc_split_cu_flag(
        nbrs.left_avail,
        nbrs.above_avail,
        nbrs.cb_height_left,
        nbrs.cb_width_above,
        cb_w,
        cb_h,
        allows.bt_ver as u32,
        allows.bt_hor as u32,
        allows.tt_ver as u32,
        allows.tt_hor as u32,
        allows.qt as u32,
    ) as usize;
    let n = ctxs.split_cu.len() - 1;
    enc.encode_decision(&mut ctxs.split_cu[inc.min(n)], 1)?;
    // split_qt_flag = 0 — coded only when QT and MTT are both allowed
    // (§7.3.11.4; r412).
    if allows.qt && allows.any_mtt() {
        encode_split_qt_flag(enc, ctxs, false, nbrs, cqt_depth)?;
    }
    let mtt_v = dir.vertical_flag();
    // mtt_split_cu_vertical_flag — coded only when both directions are
    // possible (§7.3.11.4; r412).
    if (allows.bt_hor || allows.tt_hor) && (allows.bt_ver || allows.tt_ver) {
        let mtt_v_inc = ctx_inc_mtt_split_cu_vertical_flag(
            nbrs.left_avail,
            nbrs.above_avail,
            nbrs.cb_height_left,
            nbrs.cb_width_above,
            cb_w,
            cb_h,
            allows.bt_ver as u32,
            allows.bt_hor as u32,
            allows.tt_ver as u32,
            allows.tt_hor as u32,
        ) as usize;
        let mtt_v_n = ctxs.mtt_split_vertical.len() - 1;
        enc.encode_decision(&mut ctxs.mtt_split_vertical[mtt_v_inc.min(mtt_v_n)], mtt_v)?;
    }
    // mtt_split_cu_binary_flag = 1 (BT) — coded only when both the
    // binary and ternary split are allowed in the chosen direction.
    let bin_coded = if mtt_v == 1 {
        allows.bt_ver && allows.tt_ver
    } else {
        allows.bt_hor && allows.tt_hor
    };
    if bin_coded {
        let mtt_b_inc = ctx_inc_mtt_split_cu_binary_flag(mtt_v, mtt_depth) as usize;
        let mtt_b_n = ctxs.mtt_split_binary.len() - 1;
        enc.encode_decision(&mut ctxs.mtt_split_binary[mtt_b_inc.min(mtt_b_n)], 1)?;
    }

    // Two sub-blocks of equal size.
    let (sub_w, sub_h) = match dir {
        MttSplitDir::Vertical => (cb_w / 2, cb_h),
        MttSplitDir::Horizontal => (cb_w, cb_h / 2),
    };
    body_emit_subblock(enc, ctxs, 0, sub_w, sub_h)?;
    body_emit_subblock(enc, ctxs, 1, sub_w, sub_h)?;
    Ok(())
}

/// Round-55 — encode a `coding_tree()` MTT ternary-tree (TT) split:
///   * `split_cu_flag = 1`
///   * `split_qt_flag = 0`
///   * `mtt_split_cu_vertical_flag` per `dir`
///   * `mtt_split_cu_binary_flag = 0`
///   * 3× recursion via `body_emit_subblock(idx, sub_w, sub_h)` with the
///     1:2:1 ratio (idx ∈ {0, 1, 2}).
///
/// **Round-55 scope: skeleton only.** The encoder pipeline RDO picker
/// stays leaf-or-BT for now (TT picker is round-56+), but the syntax /
/// parse path is ready so future rounds can flip the picker.
pub fn encode_coding_tree_tt_split<F>(
    enc: &mut ArithEncoder,
    ctxs: &mut TreeCtxs,
    cb_w: u32,
    cb_h: u32,
    nbrs: TreeNeighbours,
    cqt_depth: u32,
    mtt_depth: u32,
    dir: MttSplitDir,
    allows: SplitAllows,
    mut body_emit_subblock: F,
) -> Result<()>
where
    F: FnMut(&mut ArithEncoder, &mut TreeCtxs, u32, u32, u32) -> Result<()>,
{
    // split_cu_flag = 1 (r412 — real allow flags in the ctxInc).
    let inc = ctx_inc_split_cu_flag(
        nbrs.left_avail,
        nbrs.above_avail,
        nbrs.cb_height_left,
        nbrs.cb_width_above,
        cb_w,
        cb_h,
        allows.bt_ver as u32,
        allows.bt_hor as u32,
        allows.tt_ver as u32,
        allows.tt_hor as u32,
        allows.qt as u32,
    ) as usize;
    let n = ctxs.split_cu.len() - 1;
    enc.encode_decision(&mut ctxs.split_cu[inc.min(n)], 1)?;
    // split_qt_flag = 0 — §7.3.11.4 presence (r412).
    if allows.qt && allows.any_mtt() {
        encode_split_qt_flag(enc, ctxs, false, nbrs, cqt_depth)?;
    }
    let mtt_v = dir.vertical_flag();
    // mtt_split_cu_vertical_flag — §7.3.11.4 presence (r412).
    if (allows.bt_hor || allows.tt_hor) && (allows.bt_ver || allows.tt_ver) {
        let mtt_v_inc = ctx_inc_mtt_split_cu_vertical_flag(
            nbrs.left_avail,
            nbrs.above_avail,
            nbrs.cb_height_left,
            nbrs.cb_width_above,
            cb_w,
            cb_h,
            allows.bt_ver as u32,
            allows.bt_hor as u32,
            allows.tt_ver as u32,
            allows.tt_hor as u32,
        ) as usize;
        let mtt_v_n = ctxs.mtt_split_vertical.len() - 1;
        enc.encode_decision(&mut ctxs.mtt_split_vertical[mtt_v_inc.min(mtt_v_n)], mtt_v)?;
    }
    // mtt_split_cu_binary_flag = 0 (TT) — §7.3.11.4 presence (r412).
    let bin_coded = if mtt_v == 1 {
        allows.bt_ver && allows.tt_ver
    } else {
        allows.bt_hor && allows.tt_hor
    };
    if bin_coded {
        let mtt_b_inc = ctx_inc_mtt_split_cu_binary_flag(mtt_v, mtt_depth) as usize;
        let mtt_b_n = ctxs.mtt_split_binary.len() - 1;
        enc.encode_decision(&mut ctxs.mtt_split_binary[mtt_b_inc.min(mtt_b_n)], 0)?;
    }

    // Three sub-blocks with the 1:2:1 ratio.
    let (sub0_dim, sub1_dim, sub2_dim) = match dir {
        MttSplitDir::Vertical => {
            let q = cb_w / 4;
            ((q, cb_h), (q * 2, cb_h), (q, cb_h))
        }
        MttSplitDir::Horizontal => {
            let q = cb_h / 4;
            ((cb_w, q), (cb_w, q * 2), (cb_w, q))
        }
    };
    body_emit_subblock(enc, ctxs, 0, sub0_dim.0, sub0_dim.1)?;
    body_emit_subblock(enc, ctxs, 1, sub1_dim.0, sub1_dim.1)?;
    body_emit_subblock(enc, ctxs, 2, sub2_dim.0, sub2_dim.1)?;
    Ok(())
}

/// Read `split_cu_flag` per §9.3.4.2.2 — encoder dual is
/// [`encode_coding_tree_leaf_iframe`]. Round-trip used by round-52 tests.
pub fn decode_coding_tree_split_cu_flag(
    dec: &mut ArithDecoder<'_>,
    ctxs: &mut TreeCtxs,
    cb_w: u32,
    cb_h: u32,
    nbrs: TreeNeighbours,
    allows: SplitAllows,
) -> Result<u32> {
    let inc = ctx_inc_split_cu_flag(
        nbrs.left_avail,
        nbrs.above_avail,
        nbrs.cb_height_left,
        nbrs.cb_width_above,
        cb_w,
        cb_h,
        allows.bt_ver as u32,
        allows.bt_hor as u32,
        allows.tt_ver as u32,
        allows.tt_hor as u32,
        allows.qt as u32,
    ) as usize;
    let n = ctxs.split_cu.len() - 1;
    dec.decode_decision(&mut ctxs.split_cu[inc.min(n)])
}

/// Round-55 — read `split_qt_flag` per §9.3.4.2.2. Decoder dual to
/// [`encode_split_qt_flag`].
pub fn decode_coding_tree_split_qt_flag(
    dec: &mut ArithDecoder<'_>,
    ctxs: &mut TreeCtxs,
    nbrs: TreeNeighbours,
    cqt_depth: u32,
) -> Result<u32> {
    let inc = ctx_inc_split_qt_flag(
        nbrs.left_avail,
        nbrs.above_avail,
        nbrs.cqt_depth_left,
        nbrs.cqt_depth_above,
        cqt_depth,
    ) as usize;
    let n = ctxs.split_qt.len() - 1;
    dec.decode_decision(&mut ctxs.split_qt[inc.min(n)])
}

/// Round-55 — read `mtt_split_cu_vertical_flag` per §9.3.4.2.3. Decoder
/// dual to the corresponding emit inside
/// [`encode_coding_tree_bt_split`] / [`encode_coding_tree_tt_split`].
pub fn decode_coding_tree_mtt_split_vertical_flag(
    dec: &mut ArithDecoder<'_>,
    ctxs: &mut TreeCtxs,
    cb_w: u32,
    cb_h: u32,
    nbrs: TreeNeighbours,
) -> Result<u32> {
    let inc = ctx_inc_mtt_split_cu_vertical_flag(
        nbrs.left_avail,
        nbrs.above_avail,
        nbrs.cb_height_left,
        nbrs.cb_width_above,
        cb_w,
        cb_h,
        1,
        1,
        1,
        1,
    ) as usize;
    let n = ctxs.mtt_split_vertical.len() - 1;
    dec.decode_decision(&mut ctxs.mtt_split_vertical[inc.min(n)])
}

/// Round-55 — read `mtt_split_cu_binary_flag` per §9.3.4.2.1 / Table 132.
/// Decoder dual to the corresponding emit inside
/// [`encode_coding_tree_bt_split`] / [`encode_coding_tree_tt_split`].
pub fn decode_coding_tree_mtt_split_binary_flag(
    dec: &mut ArithDecoder<'_>,
    ctxs: &mut TreeCtxs,
    mtt_split_cu_vertical_flag: u32,
    mtt_depth: u32,
) -> Result<u32> {
    let inc = ctx_inc_mtt_split_cu_binary_flag(mtt_split_cu_vertical_flag, mtt_depth) as usize;
    let n = ctxs.mtt_split_binary.len() - 1;
    dec.decode_decision(&mut ctxs.mtt_split_binary[inc.min(n)])
}

/// Neighbour availability + size info used by the §9.3.4.2.2 ctxInc
/// derivations for `split_cu_flag` and `split_qt_flag`.
#[derive(Clone, Copy, Debug, Default)]
pub struct TreeNeighbours {
    /// Left neighbour available (false at picture / slice edges).
    pub left_avail: bool,
    /// Above neighbour available.
    pub above_avail: bool,
    /// `cbHeight` of the left neighbour (ignored when `left_avail`
    /// is false). 0 for root CUs at the picture edge.
    pub cb_height_left: u32,
    /// `cbWidth` of the above neighbour (ignored when `above_avail`
    /// is false).
    pub cb_width_above: u32,
    /// `CqtDepth` of the left neighbour (ignored when `left_avail`
    /// is false). 0 for root CUs at the picture edge.
    pub cqt_depth_left: u32,
    /// `CqtDepth` of the above neighbour.
    pub cqt_depth_above: u32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cabac::ArithDecoder;

    /// `encode_coding_tree_leaf_iframe` followed by
    /// `decode_coding_tree_split_cu_flag` round-trips the `split_cu_flag = 0`
    /// bin and lets the body closure run to completion.
    #[test]
    fn coding_tree_leaf_iframe_round_trips_split_cu_flag() {
        let mut enc = ArithEncoder::new();
        let mut enc_ctxs = TreeCtxs::init(26);
        let mut body_called = false;
        encode_coding_tree_leaf_iframe(
            &mut enc,
            &mut enc_ctxs,
            64,
            64,
            TreeNeighbours::default(),
            SplitAllows {
                qt: true,
                bt_ver: true,
                bt_hor: true,
                tt_ver: true,
                tt_hor: true,
            },
            |e| {
                body_called = true;
                // Body is empty — emit one bypass bin so the test sees the
                // closure actually ran.
                e.encode_bypass(1)?;
                Ok(())
            },
        )
        .unwrap();
        assert!(body_called, "body_emit closure was not invoked");
        enc.encode_terminate(1).unwrap();
        let bytes = enc.finish();

        let mut dec = ArithDecoder::new(&bytes).unwrap();
        let mut dec_ctxs = TreeCtxs::init(26);
        let split = decode_coding_tree_split_cu_flag(
            &mut dec,
            &mut dec_ctxs,
            64,
            64,
            TreeNeighbours::default(),
            SplitAllows {
                qt: true,
                bt_ver: true,
                bt_hor: true,
                tt_ver: true,
                tt_hor: true,
            },
        )
        .unwrap();
        assert_eq!(split, 0, "expected split_cu_flag = 0");
        let bypass = dec.decode_bypass().unwrap();
        assert_eq!(bypass, 1, "body bypass bin must round-trip");
    }

    /// `encode_coding_quadtree_split` emits split_cu_flag=1 + split_qt_flag=1
    /// followed by the per-quadrant closure invocations. The round-trip
    /// reads the same two bins back via the matching decoder helpers.
    #[test]
    fn coding_quadtree_split_round_trips_split_flags() {
        let mut enc = ArithEncoder::new();
        let mut enc_ctxs = TreeCtxs::init(26);
        let mut quadrants = Vec::<u32>::new();
        encode_coding_quadtree_split(
            &mut enc,
            &mut enc_ctxs,
            128,
            128,
            TreeNeighbours::default(),
            0,
            SplitAllows {
                qt: true,
                bt_ver: true,
                bt_hor: true,
                tt_ver: true,
                tt_hor: true,
            },
            |_e, _ctxs, q, _w, _h| {
                quadrants.push(q);
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(quadrants, vec![0, 1, 2, 3]);
        enc.encode_terminate(1).unwrap();
        let bytes = enc.finish();

        let mut dec = ArithDecoder::new(&bytes).unwrap();
        let mut dec_ctxs = TreeCtxs::init(26);
        let split_cu = decode_coding_tree_split_cu_flag(
            &mut dec,
            &mut dec_ctxs,
            128,
            128,
            TreeNeighbours::default(),
            SplitAllows {
                qt: true,
                bt_ver: true,
                bt_hor: true,
                tt_ver: true,
                tt_hor: true,
            },
        )
        .unwrap();
        assert_eq!(split_cu, 1);
        // split_qt_flag — read with the matching ctxInc.
        let inc = ctx_inc_split_qt_flag(false, false, 0, 0, 0) as usize;
        let n = dec_ctxs.split_qt.len() - 1;
        let split_qt = dec
            .decode_decision(&mut dec_ctxs.split_qt[inc.min(n)])
            .unwrap();
        assert_eq!(split_qt, 1);
    }

    /// Round-55 — `encode_coding_tree_bt_split` emits the four-flag
    /// preamble (`split_cu_flag = 1`, `split_qt_flag = 0`,
    /// `mtt_split_cu_vertical_flag = 1`, `mtt_split_cu_binary_flag = 1`)
    /// for a vertical BT, then the two sub-block closures (idx 0 and 1)
    /// each see equal half-width sub-blocks. The round-trip reads each
    /// flag back through its matching decoder helper.
    #[test]
    fn coding_tree_bt_split_vertical_round_trips() {
        let mut enc = ArithEncoder::new();
        let mut enc_ctxs = TreeCtxs::init(26);
        let mut subblocks = Vec::<(u32, u32, u32)>::new();
        encode_coding_tree_bt_split(
            &mut enc,
            &mut enc_ctxs,
            64,
            64,
            TreeNeighbours::default(),
            0,
            0,
            MttSplitDir::Vertical,
            SplitAllows {
                qt: true,
                bt_ver: true,
                bt_hor: true,
                tt_ver: true,
                tt_hor: true,
            },
            |_e, _ctxs, idx, w, h| {
                subblocks.push((idx, w, h));
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(subblocks, vec![(0, 32, 64), (1, 32, 64)]);
        enc.encode_terminate(1).unwrap();
        let bytes = enc.finish();

        let mut dec = ArithDecoder::new(&bytes).unwrap();
        let mut dec_ctxs = TreeCtxs::init(26);
        let split_cu = decode_coding_tree_split_cu_flag(
            &mut dec,
            &mut dec_ctxs,
            64,
            64,
            TreeNeighbours::default(),
            SplitAllows {
                qt: true,
                bt_ver: true,
                bt_hor: true,
                tt_ver: true,
                tt_hor: true,
            },
        )
        .unwrap();
        assert_eq!(split_cu, 1);
        let split_qt =
            decode_coding_tree_split_qt_flag(&mut dec, &mut dec_ctxs, TreeNeighbours::default(), 0)
                .unwrap();
        assert_eq!(split_qt, 0);
        let mtt_v = decode_coding_tree_mtt_split_vertical_flag(
            &mut dec,
            &mut dec_ctxs,
            64,
            64,
            TreeNeighbours::default(),
        )
        .unwrap();
        assert_eq!(mtt_v, 1);
        let mtt_b =
            decode_coding_tree_mtt_split_binary_flag(&mut dec, &mut dec_ctxs, mtt_v, 0).unwrap();
        assert_eq!(mtt_b, 1);
    }

    /// Round-55 — `encode_coding_tree_bt_split` for a horizontal BT
    /// emits `mtt_split_cu_vertical_flag = 0` and the two sub-blocks
    /// each have full width and half height.
    #[test]
    fn coding_tree_bt_split_horizontal_round_trips() {
        let mut enc = ArithEncoder::new();
        let mut enc_ctxs = TreeCtxs::init(26);
        let mut subblocks = Vec::<(u32, u32, u32)>::new();
        encode_coding_tree_bt_split(
            &mut enc,
            &mut enc_ctxs,
            64,
            64,
            TreeNeighbours::default(),
            0,
            1,
            MttSplitDir::Horizontal,
            SplitAllows {
                qt: true,
                bt_ver: true,
                bt_hor: true,
                tt_ver: true,
                tt_hor: true,
            },
            |_e, _ctxs, idx, w, h| {
                subblocks.push((idx, w, h));
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(subblocks, vec![(0, 64, 32), (1, 64, 32)]);
        enc.encode_terminate(1).unwrap();
        let bytes = enc.finish();

        let mut dec = ArithDecoder::new(&bytes).unwrap();
        let mut dec_ctxs = TreeCtxs::init(26);
        decode_coding_tree_split_cu_flag(
            &mut dec,
            &mut dec_ctxs,
            64,
            64,
            TreeNeighbours::default(),
            SplitAllows {
                qt: true,
                bt_ver: true,
                bt_hor: true,
                tt_ver: true,
                tt_hor: true,
            },
        )
        .unwrap();
        decode_coding_tree_split_qt_flag(&mut dec, &mut dec_ctxs, TreeNeighbours::default(), 0)
            .unwrap();
        let mtt_v = decode_coding_tree_mtt_split_vertical_flag(
            &mut dec,
            &mut dec_ctxs,
            64,
            64,
            TreeNeighbours::default(),
        )
        .unwrap();
        assert_eq!(mtt_v, 0);
        let mtt_b =
            decode_coding_tree_mtt_split_binary_flag(&mut dec, &mut dec_ctxs, mtt_v, 1).unwrap();
        assert_eq!(mtt_b, 1);
    }

    /// Round-55 — `encode_coding_tree_tt_split` (skeleton): emits
    /// `mtt_split_cu_binary_flag = 0` and the three sub-blocks have the
    /// 1:2:1 ratio.
    #[test]
    fn coding_tree_tt_split_vertical_round_trips() {
        let mut enc = ArithEncoder::new();
        let mut enc_ctxs = TreeCtxs::init(26);
        let mut subblocks = Vec::<(u32, u32, u32)>::new();
        encode_coding_tree_tt_split(
            &mut enc,
            &mut enc_ctxs,
            64,
            64,
            TreeNeighbours::default(),
            0,
            0,
            MttSplitDir::Vertical,
            SplitAllows {
                qt: true,
                bt_ver: true,
                bt_hor: true,
                tt_ver: true,
                tt_hor: true,
            },
            |_e, _ctxs, idx, w, h| {
                subblocks.push((idx, w, h));
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(subblocks, vec![(0, 16, 64), (1, 32, 64), (2, 16, 64)]);
        enc.encode_terminate(1).unwrap();
        let bytes = enc.finish();

        let mut dec = ArithDecoder::new(&bytes).unwrap();
        let mut dec_ctxs = TreeCtxs::init(26);
        decode_coding_tree_split_cu_flag(
            &mut dec,
            &mut dec_ctxs,
            64,
            64,
            TreeNeighbours::default(),
            SplitAllows {
                qt: true,
                bt_ver: true,
                bt_hor: true,
                tt_ver: true,
                tt_hor: true,
            },
        )
        .unwrap();
        decode_coding_tree_split_qt_flag(&mut dec, &mut dec_ctxs, TreeNeighbours::default(), 0)
            .unwrap();
        let mtt_v = decode_coding_tree_mtt_split_vertical_flag(
            &mut dec,
            &mut dec_ctxs,
            64,
            64,
            TreeNeighbours::default(),
        )
        .unwrap();
        assert_eq!(mtt_v, 1);
        let mtt_b =
            decode_coding_tree_mtt_split_binary_flag(&mut dec, &mut dec_ctxs, mtt_v, 0).unwrap();
        assert_eq!(mtt_b, 0, "TT split must signal binary_flag = 0");
    }
}
