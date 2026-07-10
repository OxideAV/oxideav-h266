//! §8.6 — decoding processes for coding units coded in IBC (intra
//! block copy) prediction mode.
//!
//! IBC predicts a block from the *current* picture's already-
//! reconstructed (pre-in-loop-filter) samples, addressed through the
//! `IbcVirBuf` virtual buffer — a `IbcBufWidthY × CtbSizeY` rolling
//! window of the current CTU row (§7.4.3.4 eqs. 45 – 47). This module
//! holds the spec machinery that is independent of the CTU walker:
//!
//! * [`HmvpIbcList`] — the history-based block-vector predictor list
//!   (`HmvpIbcCandList`, §8.6.2.4 / §8.6.2.6).
//! * [`build_bv_cand_list`] — the §8.6.2.2 block-vector candidate
//!   list (spatial A1 / B1 → HMVP → zero pad).
//! * [`fold_bv_with_bvd`] — the §8.6.2.1 eqs. 1092 – 1095 18-bit
//!   wrap-around predictor + difference fold.
//! * [`derive_chroma_bv`] — §8.6.2.5 eqs. 1103 / 1104.
//! * [`IbcVirtualBuffer`] — `IbcVirBuf[cIdx]` storage with the
//!   §7.4.12.5 eqs. 181 / 182 invalidation, the §8.7.5.1
//!   eqs. 1207 – 1209 reconstructed-sample fill, and the §8.6.3
//!   eqs. 1105 – 1110 prediction reads.
//!
//! The CTU walker (`crate::ctu`) owns when these run: the
//! `NumHmvpIbcCand = 0` / `ResetIbcBuf = 1` resets fire at the start
//! of every CTU row (§7.3.11.1 — `CtbAddrX == CtbToTileColBd[..]`,
//! which for the single-tile pictures this crate walks is column 0),
//! the per-CU invalidations fire at each `coding_unit()`, and the
//! buffer fill happens after each CU's §8.7.5.1 reconstruction.

use crate::inter::MotionVector;
use oxideav_core::{Error, Result};

/// §8.6.2.6 — the HMVP IBC candidate list holds at most 5 entries.
pub const HMVP_IBC_CAP: usize = 5;

/// §7.4.3.4 eq. 45 — `IbcBufWidthY = 256 * 128 / CtbSizeY`.
#[inline]
pub fn ibc_buf_width_y(ctb_size_y: u32) -> u32 {
    256 * 128 / ctb_size_y
}

/// §8.6.1 eq. 1089 — `IsGt4by4 = ( cbWidth * cbHeight ) > 16`.
#[inline]
pub fn is_gt_4by4(cb_width: u32, cb_height: u32) -> bool {
    cb_width * cb_height > 16
}

/// §8.6.2.6 — `HmvpIbcCandList`, the per-slice history-based
/// block-vector predictor list. Entries are stored oldest-first
/// (index 0 = oldest), mirroring the spec's `HmvpIbcCandList[0..
/// NumHmvpIbcCand − 1]` ordering (the §8.6.2.4 consumer walks it
/// newest-first via `NumHmvpIbcCand − hMvpIdx`).
#[derive(Clone, Debug, Default)]
pub struct HmvpIbcList {
    cands: Vec<MotionVector>,
}

impl HmvpIbcList {
    pub fn new() -> Self {
        Self::default()
    }

    /// §7.3.11.1 — `NumHmvpIbcCand = 0` at the start of every CTU row
    /// (single-tile layout) / tile-column.
    pub fn reset(&mut self) {
        self.cands.clear();
    }

    /// `NumHmvpIbcCand`.
    pub fn len(&self) -> usize {
        self.cands.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cands.is_empty()
    }

    /// `HmvpIbcCandList[idx]` (0 = oldest).
    pub fn get(&self, idx: usize) -> Option<MotionVector> {
        self.cands.get(idx).copied()
    }

    /// §8.6.2.6 — the updating process for the history-based
    /// block-vector predictor candidate list. Invoked with the CU's
    /// final `bvL` when `IsGt4by4` holds (§8.6.1 / §8.6.2.1).
    pub fn update(&mut self, bv: MotionVector) {
        // Steps 1 / 2 — find an identical candidate.
        if let Some(remove_idx) = self.cands.iter().position(|c| *c == bv) {
            // Step 3, first arm — shift down past removeIdx, append.
            self.cands.remove(remove_idx);
            self.cands.push(bv);
        } else if self.cands.len() == HMVP_IBC_CAP {
            // Step 3, first arm with removeIdx = 0 (the spec sets
            // removeIdx = 0 when no identical candidate exists and the
            // list is full — the initial value from step 1).
            self.cands.remove(0);
            self.cands.push(bv);
        } else {
            // Step 3, second arm — plain append.
            self.cands.push(bv);
        }
    }
}

/// §8.6.2.2 / §8.6.2.3 / §8.6.2.4 — build the block-vector candidate
/// list `bvCandList` for the current IBC CU.
///
/// * `a1` / `b1` — the spatial candidates from `( xCb − 1,
///   yCb + cbHeight − 1 )` and `( xCb + cbWidth − 1, yCb − 1 )`,
///   already gated by the §6.4.4 availability derivation with
///   `checkPredModeY = TRUE` (the neighbour must itself be an IBC CU);
///   `None` when unavailable. The §8.6.2.3 B1 pruning (drop B1 when it
///   equals A1) is applied here.
/// * `hmvp` — the slice's `HmvpIbcCandList`.
/// * `is_gt4by4` — §8.6.1 eq. 1089; when `false` the spatial scan is
///   skipped entirely and `numCurrCand` starts at 0 (§8.6.2.2 steps
///   1 – 3) and the §8.6.2.4 `sameMotion` pruning never fires.
/// * `max_num_ibc_merge_cand` — §7.4.3.4 eq. 62.
///
/// The returned list always has exactly `max_num_ibc_merge_cand`
/// entries (§8.6.2.2 step 5 zero-pads).
pub fn build_bv_cand_list(
    a1: Option<MotionVector>,
    b1: Option<MotionVector>,
    hmvp: &HmvpIbcList,
    is_gt4by4: bool,
    max_num_ibc_merge_cand: u32,
) -> Vec<MotionVector> {
    let max = max_num_ibc_merge_cand as usize;
    let mut list: Vec<MotionVector> = Vec::with_capacity(max);
    let (a1, b1) = if is_gt4by4 {
        // §8.6.2.3 — availableFlagB1 = 0 when A1 is available and
        // carries the same block vector.
        let b1_pruned = match (a1, b1) {
            (Some(a), Some(b)) if a == b => None,
            _ => b1,
        };
        (a1, b1_pruned)
    } else {
        (None, None)
    };
    if let Some(a) = a1 {
        list.push(a);
    }
    if let Some(b) = b1 {
        list.push(b);
    }
    // §8.6.2.4 — HMVP fill, newest-first (`NumHmvpIbcCand − hMvpIdx`
    // for hMvpIdx = 1..NumHmvpIbcCand). The `sameMotion` pruning only
    // applies to the *first* (newest) HMVP entry, only against A1 /
    // B1, and only when IsGt4by4.
    let n = hmvp.len();
    for h_mvp_idx in 1..=n {
        if list.len() >= max {
            break;
        }
        let cand = hmvp
            .get(n - h_mvp_idx)
            .expect("index in 0..n is always present");
        let same_motion = is_gt4by4
            && h_mvp_idx == 1
            && (a1.map(|a| a == cand).unwrap_or(false) || b1.map(|b| b == cand).unwrap_or(false));
        if !same_motion {
            list.push(cand);
        }
    }
    // Step 5 — zero-BV pad.
    while list.len() < max {
        list.push(MotionVector::ZERO);
    }
    list
}

/// §8.6.2.1 step 3 — fold the (already §8.5.2.14-rounded) predictor
/// `bvL` with the block-vector difference `bvd` (the CU's `MvdL0`
/// after the §7.4.12.7 eq. 161 / 162 `<< AmvrShift` scaling) through
/// the eqs. 1092 – 1095 18-bit wrap-around:
///
/// ```text
/// u[c]   = ( bvL[c] + bvd[c] ) & ( 2^18 − 1 )
/// bvL[c] = ( u[c] >= 2^17 ) ? ( u[c] − 2^18 ) : u[c]
/// ```
pub fn fold_bv_with_bvd(bv_pred: MotionVector, bvd: MotionVector) -> MotionVector {
    #[inline]
    fn fold(p: i32, d: i32) -> i32 {
        let u = (p.wrapping_add(d)) & ((1 << 18) - 1);
        if u >= (1 << 17) {
            u - (1 << 18)
        } else {
            u
        }
    }
    MotionVector {
        x: fold(bv_pred.x, bvd.x),
        y: fold(bv_pred.y, bvd.y),
    }
}

/// §8.6.2.5 — derivation process for chroma block vectors
/// (eqs. 1103 / 1104):
///
/// ```text
/// bvC[0] = ( bvL[0] >> ( 3 + SubWidthC ) ) * 32
/// bvC[1] = ( bvL[1] >> ( 3 + SubHeightC ) ) * 32
/// ```
///
/// `bvL` is in 1/16 luma-sample units; the output is in 1/32
/// chroma-sample units. The spec `>>` is an arithmetic shift, which
/// Rust's `>>` on `i32` matches.
pub fn derive_chroma_bv(bv_l: MotionVector, sub_width_c: u32, sub_height_c: u32) -> MotionVector {
    MotionVector {
        x: (bv_l.x >> (3 + sub_width_c)) * 32,
        y: (bv_l.y >> (3 + sub_height_c)) * 32,
    }
}

/// `IbcVirBuf[cIdx]` — the IBC virtual reference buffer.
///
/// Dimensions per §7.4.3.4: luma `IbcBufWidthY × CtbSizeY`, chroma
/// `IbcBufWidthC × ( CtbSizeY / SubHeightC )`. Luma entries carry the
/// `−1` invalid marker of §7.4.12.5 eqs. 181 / 182 (the spec only
/// invalidates plane 0; the §8.6.2.1 bitstream-conformance constraint
/// is checked against the luma plane and covers chroma implicitly).
#[derive(Clone, Debug)]
pub struct IbcVirtualBuffer {
    ctb_size_y: u32,
    sub_width_c: u32,
    sub_height_c: u32,
    /// `IbcBufWidthY` (eq. 45).
    buf_w_y: u32,
    /// Luma plane, row-major `buf_w_y × ctb_size_y`; `−1` = invalid.
    luma: Vec<i32>,
    /// Cb / Cr planes, row-major `(buf_w_y / SubWidthC) ×
    /// (ctb_size_y / SubHeightC)`. Empty for monochrome.
    cb: Vec<i32>,
    cr: Vec<i32>,
    /// `ResetIbcBuf` (§7.3.11.1 sets it at the start of a CTU row;
    /// §7.4.12.5 eq. 181 consumes it at the next `coding_unit()`).
    reset_pending: bool,
}

impl IbcVirtualBuffer {
    /// Build the buffer for a stream's `CtbSizeY` / chroma format.
    /// `chroma_format_idc == 0` (monochrome) allocates no chroma
    /// planes; 1 = 4:2:0, 2 = 4:2:2, 3 = 4:4:4 follow Table 2.
    pub fn new(ctb_size_y: u32, chroma_format_idc: u32) -> Result<Self> {
        if !(32..=128).contains(&ctb_size_y) || !ctb_size_y.is_power_of_two() {
            return Err(Error::invalid(format!(
                "h266 IBC: CtbSizeY {ctb_size_y} out of range for the IbcVirBuf derivation"
            )));
        }
        let (sub_w, sub_h) = match chroma_format_idc {
            0 => (1u32, 1u32),
            1 => (2, 2),
            2 => (2, 1),
            3 => (1, 1),
            other => {
                return Err(Error::invalid(format!(
                    "h266 IBC: unknown sps_chroma_format_idc {other}"
                )));
            }
        };
        let buf_w_y = ibc_buf_width_y(ctb_size_y);
        let luma = vec![-1i32; (buf_w_y * ctb_size_y) as usize];
        let (cb, cr) = if chroma_format_idc == 0 {
            (Vec::new(), Vec::new())
        } else {
            let n = ((buf_w_y / sub_w) * (ctb_size_y / sub_h)) as usize;
            (vec![0i32; n], vec![0i32; n])
        };
        Ok(Self {
            ctb_size_y,
            sub_width_c: sub_w,
            sub_height_c: sub_h,
            buf_w_y,
            luma,
            cb,
            cr,
            reset_pending: true,
        })
    }

    /// `IbcBufWidthY`.
    pub fn buf_width_y(&self) -> u32 {
        self.buf_w_y
    }

    /// §7.3.11.1 — `ResetIbcBuf = 1` (start of a CTU row / tile
    /// column).
    pub fn mark_reset(&mut self) {
        self.reset_pending = true;
    }

    /// §7.4.12.5 per-`coding_unit()` maintenance:
    ///
    /// * eq. 181 — when `ResetIbcBuf == 1`, every luma entry becomes
    ///   `−1` and the flag clears.
    /// * eq. 182 — when `x0 % VSize == 0 && y0 % VSize == 0`, the
    ///   half-buffer-ahead region covering this VPDU's footprint is
    ///   invalidated: for `x = x0..x0 + Max(cbWidth, VSize) − 1`,
    ///   `y = y0..y0 + Max(cbHeight, VSize) − 1`,
    ///   `IbcVirBuf[0][ (x + (IbcBufWidthY >> 1)) % IbcBufWidthY ]
    ///   [ y % CtbSizeY ] = −1`.
    ///
    /// `vsize` is the §7.4.3.4 eq. 47 `VSize = Min(64, CtbSizeY)`.
    pub fn on_cu_start(&mut self, x0: u32, y0: u32, cb_width: u32, cb_height: u32, vsize: u32) {
        if self.reset_pending {
            self.luma.fill(-1);
            self.reset_pending = false;
        }
        if x0 % vsize == 0 && y0 % vsize == 0 {
            let w = cb_width.max(vsize);
            let h = cb_height.max(vsize);
            for y in y0..y0 + h {
                let yv = (y % self.ctb_size_y) as usize;
                for x in x0..x0 + w {
                    let xv = ((x + (self.buf_w_y >> 1)) % self.buf_w_y) as usize;
                    self.luma[yv * self.buf_w_y as usize + xv] = -1;
                }
            }
        }
    }

    /// §8.7.5.1 eqs. 1207 – 1209 — store a just-reconstructed block
    /// into the buffer. `(x_curr, y_curr)` and the dimensions are in
    /// the plane's own sample units (luma samples for `c_idx == 0`,
    /// chroma samples otherwise); `get(dx, dy)` returns the
    /// reconstructed sample at `(x_curr + dx, y_curr + dy)`.
    pub fn store_region(
        &mut self,
        c_idx: u32,
        x_curr: u32,
        y_curr: u32,
        w: u32,
        h: u32,
        get: impl Fn(u32, u32) -> i32,
    ) {
        let (buf_w, buf_h) = self.plane_dims(c_idx);
        if buf_w == 0 {
            return;
        }
        let plane: &mut Vec<i32> = match c_idx {
            0 => &mut self.luma,
            1 => &mut self.cb,
            _ => &mut self.cr,
        };
        for j in 0..h {
            let yv = ((y_curr + j) % buf_h) as usize;
            for i in 0..w {
                let xv = ((x_curr + i) % buf_w) as usize;
                plane[yv * buf_w as usize + xv] = get(i, j);
            }
        }
    }

    /// §8.6.3 eqs. 1105 – 1107 — luma prediction read for the sample
    /// at picture position `(x, y)` with block vector `bv` (1/16
    /// luma-sample units). Errors when the referenced entry carries
    /// the `−1` invalid marker (a §8.6.2.1 bitstream-conformance
    /// violation).
    pub fn luma_at(&self, x: u32, y: u32, bv: MotionVector) -> Result<i32> {
        let xv = ((x as i32 + (bv.x >> 4)) & (self.buf_w_y as i32 - 1)) as usize;
        let yv = ((y as i32 + (bv.y >> 4)) & (self.ctb_size_y as i32 - 1)) as usize;
        let v = self.luma[yv * self.buf_w_y as usize + xv];
        if v < 0 {
            return Err(Error::invalid(format!(
                "h266 IBC: block vector ({}, {}) references an invalid IbcVirBuf luma entry at \
                 buffer ({xv}, {yv}) — §8.6.2.1 bitstream-conformance violation",
                bv.x, bv.y
            )));
        }
        Ok(v)
    }

    /// §8.6.3 eqs. 1108 – 1110 — chroma prediction read for the
    /// chroma-plane sample at `(x, y)` (chroma-sample units) with the
    /// *luma* block vector `bv` (the spec indexes the chroma buffer
    /// with `bv >> (3 + SubWidthC / SubHeightC)`, i.e. the integer
    /// chroma displacement of the luma BV).
    pub fn chroma_at(&self, c_idx: u32, x: u32, y: u32, bv: MotionVector) -> Result<i32> {
        let (buf_w, buf_h) = self.plane_dims(c_idx);
        if buf_w == 0 {
            return Err(Error::invalid(
                "h266 IBC: chroma read on a monochrome IbcVirBuf",
            ));
        }
        let xv = ((x as i32 + (bv.x >> (3 + self.sub_width_c))) & (buf_w as i32 - 1)) as usize;
        let yv = ((y as i32 + (bv.y >> (3 + self.sub_height_c))) & (buf_h as i32 - 1)) as usize;
        let plane = if c_idx == 1 { &self.cb } else { &self.cr };
        Ok(plane[yv * buf_w as usize + xv])
    }

    /// §8.6.2.1 — the two bitstream-conformance constraints on a
    /// final luma block vector `bvL`:
    ///
    /// * `CtbSizeY >= ( ( yCb + ( bvL[1] >> 4 ) ) & ( CtbSizeY − 1 ) )
    ///   + cbHeight` — the reference block does not straddle a CTU-row
    ///   boundary vertically.
    /// * every referenced `IbcVirBuf[0]` entry is valid (not `−1`).
    pub fn bv_conformance_ok(
        &self,
        x_cb: u32,
        y_cb: u32,
        cb_width: u32,
        cb_height: u32,
        bv: MotionVector,
    ) -> bool {
        let wrapped_top = (y_cb as i32 + (bv.y >> 4)) & (self.ctb_size_y as i32 - 1);
        if (self.ctb_size_y as i32) < wrapped_top + cb_height as i32 {
            return false;
        }
        for y in y_cb..y_cb + cb_height {
            for x in x_cb..x_cb + cb_width {
                if self.luma_at(x, y, bv).is_err() {
                    return false;
                }
            }
        }
        true
    }

    fn plane_dims(&self, c_idx: u32) -> (u32, u32) {
        if c_idx == 0 {
            (self.buf_w_y, self.ctb_size_y)
        } else if self.cb.is_empty() {
            (0, 0)
        } else {
            (
                self.buf_w_y / self.sub_width_c,
                self.ctb_size_y / self.sub_height_c,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mv(x: i32, y: i32) -> MotionVector {
        MotionVector { x, y }
    }

    // ---- §7.4.3.4 derivations ---------------------------------------

    #[test]
    fn ibc_buf_width_eq_45() {
        assert_eq!(ibc_buf_width_y(128), 256);
        assert_eq!(ibc_buf_width_y(64), 512);
        assert_eq!(ibc_buf_width_y(32), 1024);
    }

    #[test]
    fn is_gt_4by4_eq_1089() {
        assert!(!is_gt_4by4(4, 4));
        assert!(is_gt_4by4(8, 4));
        assert!(is_gt_4by4(4, 8));
        assert!(is_gt_4by4(8, 8));
    }

    // ---- §8.6.2.6 HMVP list ------------------------------------------

    #[test]
    fn hmvp_appends_until_cap_then_drops_oldest() {
        let mut h = HmvpIbcList::new();
        for i in 0..5 {
            h.update(mv(i, 0));
        }
        assert_eq!(h.len(), 5);
        // Full list + new candidate → removeIdx = 0 path (drop oldest).
        h.update(mv(100, 0));
        assert_eq!(h.len(), 5);
        assert_eq!(h.get(0), Some(mv(1, 0)));
        assert_eq!(h.get(4), Some(mv(100, 0)));
    }

    #[test]
    fn hmvp_identical_candidate_moves_to_newest() {
        let mut h = HmvpIbcList::new();
        h.update(mv(1, 1));
        h.update(mv(2, 2));
        h.update(mv(3, 3));
        // Re-insert the oldest — it must move to the newest slot
        // without growing the list.
        h.update(mv(1, 1));
        assert_eq!(h.len(), 3);
        assert_eq!(h.get(0), Some(mv(2, 2)));
        assert_eq!(h.get(1), Some(mv(3, 3)));
        assert_eq!(h.get(2), Some(mv(1, 1)));
    }

    #[test]
    fn hmvp_reset_clears() {
        let mut h = HmvpIbcList::new();
        h.update(mv(1, 1));
        h.reset();
        assert!(h.is_empty());
    }

    // ---- §8.6.2.2 candidate list -------------------------------------

    #[test]
    fn bv_cand_list_spatial_then_hmvp_then_zero_pad() {
        let mut h = HmvpIbcList::new();
        h.update(mv(-64, 0)); // oldest
        h.update(mv(-32, 0)); // newest
        let list = build_bv_cand_list(Some(mv(-16, 0)), Some(mv(0, -16)), &h, true, 6);
        assert_eq!(list.len(), 6);
        assert_eq!(list[0], mv(-16, 0)); // A1
        assert_eq!(list[1], mv(0, -16)); // B1
        assert_eq!(list[2], mv(-32, 0)); // HMVP newest-first
        assert_eq!(list[3], mv(-64, 0));
        assert_eq!(list[4], MotionVector::ZERO); // zero pad
        assert_eq!(list[5], MotionVector::ZERO);
    }

    #[test]
    fn bv_cand_list_prunes_b1_equal_to_a1() {
        let h = HmvpIbcList::new();
        let list = build_bv_cand_list(Some(mv(-16, 0)), Some(mv(-16, 0)), &h, true, 3);
        assert_eq!(list[0], mv(-16, 0));
        // B1 pruned → slot 1 is the zero pad.
        assert_eq!(list[1], MotionVector::ZERO);
    }

    #[test]
    fn bv_cand_list_hmvp_first_entry_same_motion_pruned() {
        let mut h = HmvpIbcList::new();
        h.update(mv(-64, 0));
        h.update(mv(-16, 0)); // newest == A1 → sameMotion prune
        let list = build_bv_cand_list(Some(mv(-16, 0)), None, &h, true, 4);
        assert_eq!(list[0], mv(-16, 0)); // A1
        assert_eq!(list[1], mv(-64, 0)); // hMvpIdx=2 NOT pruned
        assert_eq!(list[2], MotionVector::ZERO);
    }

    #[test]
    fn bv_cand_list_hmvp_second_entry_never_same_motion_pruned() {
        // The sameMotion prune only applies to hMvpIdx == 1 — an older
        // duplicate still enters the list.
        let mut h = HmvpIbcList::new();
        h.update(mv(-16, 0)); // oldest == A1 but hMvpIdx = 2
        h.update(mv(-64, 0)); // newest, distinct
        let list = build_bv_cand_list(Some(mv(-16, 0)), None, &h, true, 4);
        assert_eq!(list[0], mv(-16, 0));
        assert_eq!(list[1], mv(-64, 0));
        assert_eq!(list[2], mv(-16, 0)); // duplicate allowed at hMvpIdx 2
    }

    #[test]
    fn bv_cand_list_not_gt4by4_skips_spatials() {
        let mut h = HmvpIbcList::new();
        h.update(mv(-32, 0));
        // Spatials present but IsGt4by4 == FALSE → ignored; HMVP
        // pruning also disabled.
        let list = build_bv_cand_list(Some(mv(-16, 0)), Some(mv(0, -16)), &h, false, 2);
        assert_eq!(list[0], mv(-32, 0)); // HMVP directly
        assert_eq!(list[1], MotionVector::ZERO);
    }

    // ---- §8.6.2.1 fold + §8.6.2.5 chroma BV ---------------------------

    #[test]
    fn fold_bv_plain_addition() {
        assert_eq!(fold_bv_with_bvd(mv(-256, 16), mv(16, -32)), mv(-240, -16));
    }

    #[test]
    fn fold_bv_wraps_at_18_bits() {
        // 2^17 - 1 + 1 wraps to -2^17.
        assert_eq!(
            fold_bv_with_bvd(mv((1 << 17) - 1, 0), mv(1, 0)),
            mv(-(1 << 17), 0)
        );
        // -2^17 - 1 wraps to 2^17 - 1.
        assert_eq!(
            fold_bv_with_bvd(mv(-(1 << 17), 0), mv(-1, 0)),
            mv((1 << 17) - 1, 0)
        );
    }

    #[test]
    fn chroma_bv_420_eq_1103_1104() {
        // 4:2:0 → SubWidthC = SubHeightC = 2 → bvC = (bvL >> 5) * 32.
        assert_eq!(derive_chroma_bv(mv(-256, -64), 2, 2), mv(-256, -64));
        // Arithmetic shift on negatives rounds toward -inf.
        assert_eq!(derive_chroma_bv(mv(-40, 33), 2, 2), mv(-64, 32));
    }

    // ---- IbcVirBuf ----------------------------------------------------

    #[test]
    fn virbuf_starts_invalid_and_reset_clears_luma() {
        let mut b = IbcVirtualBuffer::new(64, 1).unwrap();
        assert!(b.luma_at(0, 0, MotionVector::ZERO).is_err());
        b.on_cu_start(0, 0, 16, 16, 64);
        // Reset applied then the (0,0) VPDU invalidation targets the
        // half-buffer-ahead region — (0,0) itself stays invalid until
        // something is stored.
        assert!(b.luma_at(0, 0, MotionVector::ZERO).is_err());
        b.store_region(0, 0, 0, 16, 16, |dx, dy| (dx + dy) as i32);
        assert_eq!(b.luma_at(0, 0, MotionVector::ZERO).unwrap(), 0);
        assert_eq!(b.luma_at(15, 15, MotionVector::ZERO).unwrap(), 30);
    }

    #[test]
    fn virbuf_store_then_read_via_negative_bv() {
        let mut b = IbcVirtualBuffer::new(64, 1).unwrap();
        b.on_cu_start(0, 0, 64, 64, 64);
        b.store_region(0, 0, 0, 64, 64, |dx, dy| (dy * 64 + dx) as i32);
        // Read the block at (16, 16) shifted 16 left in integer units:
        // bv = (-16 << 4, 0).
        let bv = mv(-16 << 4, 0);
        assert_eq!(b.luma_at(16, 16, bv).unwrap(), (16 * 64) as i32);
        assert_eq!(b.luma_at(31, 31, bv).unwrap(), (31 * 64 + 15) as i32);
    }

    #[test]
    fn virbuf_vpdu_invalidation_eq_182() {
        let ctb = 64u32;
        let mut b = IbcVirtualBuffer::new(ctb, 1).unwrap();
        b.on_cu_start(0, 0, 64, 64, 64);
        b.store_region(0, 0, 0, 64, 64, |_, _| 7);
        // Simulate the next CTU (x0 = 64): the eq. 182 invalidation at
        // its (64, 0) VPDU wipes the region (64 + 256) % 512 = 320..384
        // — not the freshly stored 0..64 column.
        b.on_cu_start(64, 0, 64, 64, 64);
        assert_eq!(b.luma_at(0, 0, MotionVector::ZERO).unwrap(), 7);
        // Store the second CTU and hop back one CTU with a BV — the
        // §8.6.3 wrap finds the first CTU's samples.
        b.store_region(0, 64, 0, 64, 64, |_, _| 9);
        let bv = mv(-64 << 4, 0);
        assert_eq!(b.luma_at(64, 0, bv).unwrap(), 7);
        assert_eq!(b.luma_at(64, 0, MotionVector::ZERO).unwrap(), 9);
    }

    #[test]
    fn virbuf_conformance_rejects_row_straddle() {
        let mut b = IbcVirtualBuffer::new(64, 1).unwrap();
        b.on_cu_start(0, 0, 64, 64, 64);
        b.store_region(0, 0, 0, 64, 64, |_, _| 1);
        // A BV pointing 8 up from y=32 with height 40 straddles the
        // wrapped row bound: ((32 - 8) & 63) + 40 = 64 → OK edge;
        // height 48 → 72 > 64 → reject.
        assert!(b.bv_conformance_ok(32, 32, 8, 32, mv(-256, -128)));
        assert!(!b.bv_conformance_ok(32, 32, 8, 48, mv(-256, -128)));
    }

    #[test]
    fn virbuf_conformance_rejects_invalid_region() {
        let mut b = IbcVirtualBuffer::new(64, 1).unwrap();
        b.on_cu_start(0, 0, 32, 32, 64);
        b.store_region(0, 0, 0, 32, 32, |_, _| 1);
        // Reference fully inside the stored 32×32 → OK.
        assert!(b.bv_conformance_ok(16, 16, 16, 16, mv(-16 << 4, -16 << 4)));
        // Reference reaching into never-stored (still -1) samples → reject.
        assert!(!b.bv_conformance_ok(48, 0, 16, 16, mv(-8 << 4, 0)));
    }

    #[test]
    fn virbuf_chroma_read_420() {
        let mut b = IbcVirtualBuffer::new(64, 1).unwrap();
        b.on_cu_start(0, 0, 64, 64, 64);
        b.store_region(1, 0, 0, 32, 32, |dx, dy| (dy * 32 + dx) as i32 + 1000);
        b.store_region(2, 0, 0, 32, 32, |dx, dy| (dy * 32 + dx) as i32 + 2000);
        // Luma BV (-16 px, 0) → chroma displacement -8 chroma px.
        let bv = mv(-16 << 4, 0);
        assert_eq!(b.chroma_at(1, 8, 0, bv).unwrap(), 1000);
        assert_eq!(b.chroma_at(2, 9, 1, bv).unwrap(), 2000 + 32 + 1);
    }

    #[test]
    fn virbuf_rejects_bad_geometry() {
        assert!(IbcVirtualBuffer::new(16, 1).is_err());
        assert!(IbcVirtualBuffer::new(96, 1).is_err());
        assert!(IbcVirtualBuffer::new(64, 9).is_err());
    }

    #[test]
    fn virbuf_monochrome_has_no_chroma_planes() {
        let b = IbcVirtualBuffer::new(64, 0).unwrap();
        assert!(b.chroma_at(1, 0, 0, MotionVector::ZERO).is_err());
    }
}
