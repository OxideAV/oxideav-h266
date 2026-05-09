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
use crate::coding_tree::TreeCtxs;
use crate::ctx::{ctx_inc_split_cu_flag, ctx_inc_split_qt_flag};

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
    body_emit: F,
) -> Result<()>
where
    F: FnOnce(&mut ArithEncoder) -> Result<()>,
{
    // Emit split_cu_flag = 0 (leaf CU). Per §9.3.4.2.2 eq. 1551 the
    // ctxInc reads two neighbour conditions — `cbHeightLeft < cbHeight`
    // and `cbWidthAbove < cbWidth` — plus the partition-allowance
    // `ctxSetIdx`. For a root-CU at the max size all splits are allowed
    // (allow_*_flag = 1); the picture-edge case has both availabilities
    // false so condL / condA collapse to false.
    let inc = ctx_inc_split_cu_flag(
        nbrs.left_avail,
        nbrs.above_avail,
        nbrs.cb_height_left,
        nbrs.cb_width_above,
        cb_w,
        cb_h,
        /*allow_split_bt_ver=*/ 1,
        /*allow_split_bt_hor=*/ 1,
        /*allow_split_tt_ver=*/ 1,
        /*allow_split_tt_hor=*/ 1,
        /*allow_split_qt=*/ 1,
    ) as usize;
    let n = ctxs.split_cu.len() - 1;
    enc.encode_decision(&mut ctxs.split_cu[inc.min(n)], 0)?;

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
/// `body_emit_quadrant(quadrant_idx, x0, y0, sub_w, sub_h)`.
///
/// Used when the current CU exceeds MaxBtSize (e.g. a 128×128 CTB
/// requires a forced QT split because MaxCbSizeY = 64 in our profile).
/// Round-52 encoder pipeline keeps to single 64×64 CUs and does not
/// invoke this; exposed for future rounds.
pub fn encode_coding_quadtree_split<F>(
    enc: &mut ArithEncoder,
    ctxs: &mut TreeCtxs,
    cb_w: u32,
    cb_h: u32,
    nbrs: TreeNeighbours,
    cqt_depth: u32,
    mut body_emit_quadrant: F,
) -> Result<()>
where
    F: FnMut(&mut ArithEncoder, u32, u32, u32) -> Result<()>,
{
    // split_cu_flag = 1.
    let inc = ctx_inc_split_cu_flag(
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
        1,
    ) as usize;
    let n = ctxs.split_cu.len() - 1;
    enc.encode_decision(&mut ctxs.split_cu[inc.min(n)], 1)?;
    // split_qt_flag = 1.
    encode_split_qt_flag(enc, ctxs, true, nbrs, cqt_depth)?;
    let hw = cb_w / 2;
    let hh = cb_h / 2;
    for q in 0..4u32 {
        body_emit_quadrant(enc, q, hw, hh)?;
    }
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
) -> Result<u32> {
    let inc = ctx_inc_split_cu_flag(
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
        1,
    ) as usize;
    let n = ctxs.split_cu.len() - 1;
    dec.decode_decision(&mut ctxs.split_cu[inc.min(n)])
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
            |_e, q, _w, _h| {
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
}
