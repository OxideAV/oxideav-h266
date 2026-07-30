//! VVC coefficient scan orders and sub-block partitioning
//! (§7.3.11.11 + §6.5.2).
//!
//! VVC uses a **diagonal up-right scan** both at the sub-block level
//! and within each sub-block. The sub-block SHAPE follows the
//! §7.3.11.11 `log2SbW` / `log2SbH` derivation: 4×4 for regular TBs,
//! 2×2 when both TB dims are small, and stretched (2×8 / 8×2 / 1×16 /
//! 16×1) for thin TBs with 16+ coefficients (r434 — previously
//! hardcoded to 4×4-with-clipping, which desynced every thin chroma
//! TB of the JVET conformance streams).
//!
//! Primitives landed:
//!
//! * [`diag_scan_order`] — returns the (x, y) scan positions for a
//!   `w × h` block in diagonal up-right order. Used both for the
//!   within-sub-block scan and the sub-block-level scan.
//! * [`sb_log2_dims`] / [`sb_coeff_dims`] / [`num_sb_coeff`] — the
//!   §7.3.11.11 sub-block shape.
//! * [`sb_grid`] — returns the `(numSbW, numSbH)` sub-block grid for
//!   a TB of size `(n_tb_w, n_tb_h)`.
//! * [`sb_scan_positions`] — returns the list of sub-block origins
//!   `(xSb, ySb)` in diagonal-scan order.
//! * [`coeff_scan_positions`] — combines the two: emits every
//!   `(xC, yC)` in the TB in the spec's full residual-coding scan
//!   order.
//!
//! Spec reference: ITU-T H.266 | ISO/IEC 23090-3 (V4, 01/2026).

/// Diagonal up-right scan order for a `w × h` rectangle. For each
/// anti-diagonal `k = x + y = 0..w + h - 2`, yield `(x, k-x)` for
/// valid coordinates inside the rectangle, with x starting from
/// `max(0, k - h + 1)`.
pub fn diag_scan_order(w: usize, h: usize) -> Vec<(u32, u32)> {
    let mut out = Vec::with_capacity(w * h);
    for k in 0..(w + h - 1) {
        let x_start = k.saturating_sub(h - 1);
        let x_end = core::cmp::min(k, w - 1);
        for x in x_start..=x_end {
            let y = k - x;
            out.push((x as u32, y as u32));
        }
    }
    out
}

/// §7.3.11.11 — `log2SbW` / `log2SbH` for a TB of
/// `(log2_w, log2_h)` (Zo dims). Square 4×4 sub-blocks for regular
/// TBs; 2×2 when both dims are small; a thin TB (one dim < 4) with 16+
/// coefficients keeps `numSbCoeff == 16` by stretching the other
/// sub-block dimension (2×8 / 8×2 / 1×16 / 16×1):
///
/// ```text
/// log2SbW = ( Min( Log2ZoTbWidth, Log2ZoTbHeight ) < 2 ? 1 : 2 )
/// log2SbH = log2SbW
/// if( Log2ZoTbWidth + Log2ZoTbHeight > 3 )
///   if( Log2ZoTbWidth < 2 )      { log2SbW = Log2ZoTbWidth;  log2SbH = 4 − log2SbW }
///   else if( Log2ZoTbHeight < 2 ){ log2SbH = Log2ZoTbHeight; log2SbW = 4 − log2SbH }
/// ```
pub fn sb_log2_dims(log2_w: u32, log2_h: u32) -> (u32, u32) {
    let mut log2_sb_w = if log2_w.min(log2_h) < 2 { 1 } else { 2 };
    let mut log2_sb_h = log2_sb_w;
    if log2_w + log2_h > 3 {
        if log2_w < 2 {
            log2_sb_w = log2_w;
            log2_sb_h = 4 - log2_sb_w;
        } else if log2_h < 2 {
            log2_sb_h = log2_h;
            log2_sb_w = 4 - log2_sb_h;
        }
    }
    (log2_sb_w, log2_sb_h)
}

/// Width and height of one coefficient sub-block for the given TB
/// dims (§7.3.11.11 `1 << log2SbW` / `1 << log2SbH`). `(4, 4)` for
/// regular TBs, `(2, 2)` for small ones, stretched shapes (2×8, 8×2,
/// …) for thin TBs with 16+ coefficients.
pub fn sb_coeff_dims(n_tb_w: usize, n_tb_h: usize) -> (usize, usize) {
    let (lw, lh) = sb_log2_dims((n_tb_w.max(1)).ilog2(), (n_tb_h.max(1)).ilog2());
    (1usize << lw, 1usize << lh)
}

/// `numSbCoeff = 1 << (log2SbW + log2SbH)` — coefficients per
/// sub-block (uniform across the TB: the sub-block tiles the TB
/// exactly).
pub fn num_sb_coeff(n_tb_w: usize, n_tb_h: usize) -> usize {
    let (w, h) = sb_coeff_dims(n_tb_w, n_tb_h);
    w * h
}

/// Compute the number of sub-blocks horizontally / vertically in a TB
/// of size `(n_tb_w, n_tb_h)` under the §7.3.11.11 sub-block shape.
pub fn sb_grid(n_tb_w: usize, n_tb_h: usize) -> (usize, usize) {
    let (sb_w, sb_h) = sb_coeff_dims(n_tb_w, n_tb_h);
    ((n_tb_w / sb_w).max(1), (n_tb_h / sb_h).max(1))
}

/// Sub-block scan positions in diagonal order, expressed as
/// `(xSb, ySb)` sample-space origins relative to the TB top-left.
pub fn sb_scan_positions(n_tb_w: usize, n_tb_h: usize) -> Vec<(u32, u32)> {
    let (num_sb_w, num_sb_h) = sb_grid(n_tb_w, n_tb_h);
    let (sb_w, sb_h) = sb_coeff_dims(n_tb_w, n_tb_h);
    diag_scan_order(num_sb_w, num_sb_h)
        .into_iter()
        .map(|(sx, sy)| (sx * sb_w as u32, sy * sb_h as u32))
        .collect()
}

/// Emit all `(xC, yC)` scan positions for a full TB in the spec's
/// composed residual-coding order: sub-blocks in diagonal order, then
/// within each sub-block another diagonal scan over the 4×4 grid.
pub fn coeff_scan_positions(n_tb_w: usize, n_tb_h: usize) -> Vec<(u32, u32)> {
    let (sb_w, sb_h) = sb_coeff_dims(n_tb_w, n_tb_h);
    let within_sb = diag_scan_order(sb_w, sb_h);
    let mut out = Vec::with_capacity(n_tb_w * n_tb_h);
    for (sx, sy) in sb_scan_positions(n_tb_w, n_tb_h) {
        for &(dx, dy) in &within_sb {
            out.push((sx + dx, sy + dy));
        }
    }
    debug_assert_eq!(out.len(), n_tb_w * n_tb_h);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Diagonal scan for a 2×2 block: (0,0), (0,1)/(1,0), (1,1).
    /// Within a single anti-diagonal, positions are ordered with
    /// x increasing.
    #[test]
    fn diag_scan_2x2() {
        let s = diag_scan_order(2, 2);
        assert_eq!(s, vec![(0, 0), (0, 1), (1, 0), (1, 1)]);
    }

    /// Diagonal scan for 4×4 covers all 16 positions in the correct
    /// diagonal order.
    #[test]
    fn diag_scan_4x4_count_and_first_few() {
        let s = diag_scan_order(4, 4);
        assert_eq!(s.len(), 16);
        assert_eq!(s[0], (0, 0));
        assert_eq!(s[1], (0, 1));
        assert_eq!(s[2], (1, 0));
        assert_eq!(s[3], (0, 2));
        assert_eq!(s[4], (1, 1));
        assert_eq!(s[5], (2, 0));
        assert_eq!(s[15], (3, 3));
    }

    /// Diagonal scan for a non-square rectangle (4×2).
    /// k=0: (0,0). k=1: (0,1),(1,0). k=2: (1,1),(2,0). k=3: (2,1),(3,0).
    /// k=4: (3,1). Total 8 = 4*2.
    #[test]
    fn diag_scan_4x2() {
        let s = diag_scan_order(4, 2);
        assert_eq!(s.len(), 8);
        assert_eq!(s[0], (0, 0));
        assert_eq!(s[1], (0, 1));
        assert_eq!(s[2], (1, 0));
        assert_eq!(s[3], (1, 1));
        assert_eq!(s[4], (2, 0));
        assert_eq!(s[5], (2, 1));
        assert_eq!(s[6], (3, 0));
        assert_eq!(s[7], (3, 1));
    }

    #[test]
    fn sb_grid_8x8_is_2x2() {
        assert_eq!(sb_grid(8, 8), (2, 2));
    }

    /// §7.3.11.11 thin-TB sub-block shapes (r434): a TB with one dim
    /// < 4 and 16+ coefficients stretches the other sub-block dim so
    /// numSbCoeff stays 16; small TBs use 2x2 sub-blocks.
    #[test]
    fn sb_dims_thin_tbs() {
        assert_eq!(sb_coeff_dims(2, 8), (2, 8));
        assert_eq!(sb_grid(2, 8), (1, 1));
        assert_eq!(sb_coeff_dims(2, 16), (2, 8));
        assert_eq!(sb_grid(2, 16), (1, 2));
        assert_eq!(sb_coeff_dims(16, 2), (8, 2));
        assert_eq!(sb_grid(16, 2), (2, 1));
        assert_eq!(sb_coeff_dims(2, 4), (2, 2));
        assert_eq!(sb_grid(2, 4), (1, 2));
        assert_eq!(sb_coeff_dims(4, 2), (2, 2));
        assert_eq!(sb_grid(4, 2), (2, 1));
        assert_eq!(sb_coeff_dims(2, 2), (2, 2));
        assert_eq!(sb_grid(2, 2), (1, 1));
        // Regular TBs keep 4x4 sub-blocks.
        assert_eq!(sb_coeff_dims(4, 4), (4, 4));
        assert_eq!(sb_coeff_dims(32, 4), (4, 4));
        assert_eq!(num_sb_coeff(16, 2), 16);
        assert_eq!(num_sb_coeff(2, 4), 4);
    }

    /// The composed scan covers a 16x2 TB as two 8x2 sub-blocks of 16
    /// coefficients each — every position once, sub-blocks contiguous.
    #[test]
    fn coeff_scan_16x2_sub_block_shape() {
        let s = coeff_scan_positions(16, 2);
        assert_eq!(s.len(), 32);
        // First sub-block spans x 0..8 only.
        assert!(s[..16].iter().all(|&(x, _)| x < 8));
        assert!(s[16..].iter().all(|&(x, _)| x >= 8));
        let mut seen = std::collections::HashSet::new();
        for &(x, y) in &s {
            assert!(seen.insert((x, y)));
        }
    }

    #[test]
    fn sb_grid_16x8_is_4x2() {
        assert_eq!(sb_grid(16, 8), (4, 2));
    }

    /// 4×4 TB has exactly one sub-block at origin (0,0).
    #[test]
    fn sb_scan_positions_4x4_single_origin() {
        let p = sb_scan_positions(4, 4);
        assert_eq!(p, vec![(0, 0)]);
    }

    /// 8×8 TB has four sub-blocks in diagonal order:
    /// (0,0), (0,4), (4,0), (4,4) — i.e. sub-block grid (0,0),(0,1),(1,0),(1,1)
    /// scaled by 4.
    #[test]
    fn sb_scan_positions_8x8() {
        let p = sb_scan_positions(8, 8);
        assert_eq!(p, vec![(0, 0), (0, 4), (4, 0), (4, 4)]);
    }

    /// Full coefficient scan composition for a 4×4 TB is exactly the
    /// 4×4 diagonal scan.
    #[test]
    fn coeff_scan_4x4() {
        let scan = coeff_scan_positions(4, 4);
        let inner = diag_scan_order(4, 4);
        assert_eq!(scan.len(), inner.len());
        for (a, b) in scan.iter().zip(&inner) {
            assert_eq!(a, b);
        }
    }

    /// 8×8 TB: 4 sub-blocks × 16 coeffs = 64. First 16 = sub-block (0,0).
    #[test]
    fn coeff_scan_8x8_totals_64() {
        let scan = coeff_scan_positions(8, 8);
        assert_eq!(scan.len(), 64);
        assert_eq!(scan[0], (0, 0));
        assert_eq!(scan[15], (3, 3));
        // Next sub-block is at (0, 4): first entry (0, 4).
        assert_eq!(scan[16], (0, 4));
    }
}
