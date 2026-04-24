//! VVC coefficient scan orders and 4×4 sub-block partitioning
//! (§7.4.11.9 + §6.5.2).
//!
//! VVC uses a **diagonal up-right scan** both at the sub-block level
//! (4×4 sub-blocks inside the TB) and within each 4×4 sub-block. The
//! scan orders are generated procedurally per TB dimensions.
//!
//! Primitives landed:
//!
//! * [`diag_scan_order`] — returns the (x, y) scan positions for a
//!   `w × h` block in diagonal up-right order. Used both for the 4×4
//!   within-sub-block scan and the sub-block-level scan.
//! * [`sb_grid`] — returns the `(numSbW, numSbH)` sub-block grid for
//!   a TB of size `(log2_w, log2_h)`, with the 4×4 sub-block size
//!   produced by the spec's eq. 68 / 69.
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

/// Compute the number of 4×4 sub-blocks horizontally / vertically in a
/// TB of size `(n_tb_w, n_tb_h)`. Degenerate sizes 1 / 2 collapse the
/// sub-block in that dimension (see §7.4.11.9 note: a 2×N TB has 1×N/4
/// sub-blocks of "size" 2×4).
pub fn sb_grid(n_tb_w: usize, n_tb_h: usize) -> (usize, usize) {
    let num_sb_w = ((n_tb_w + 3) / 4).max(1);
    let num_sb_h = ((n_tb_h + 3) / 4).max(1);
    (num_sb_w, num_sb_h)
}

/// Sub-block scan positions in diagonal order, expressed as
/// `(xSb, ySb)` sample-space origins relative to the TB top-left.
pub fn sb_scan_positions(n_tb_w: usize, n_tb_h: usize) -> Vec<(u32, u32)> {
    let (num_sb_w, num_sb_h) = sb_grid(n_tb_w, n_tb_h);
    diag_scan_order(num_sb_w, num_sb_h)
        .into_iter()
        .map(|(sx, sy)| (sx * 4, sy * 4))
        .collect()
}

/// Width and height of a 4×4 sub-block in coefficient space for the
/// given TB size. Returns `(sb_w, sb_h)` — always `(4, 4)` for
/// non-degenerate TBs; smaller when the TB collapses to a thin strip.
pub fn sb_coeff_dims(n_tb_w: usize, n_tb_h: usize) -> (usize, usize) {
    let sb_w = core::cmp::min(4, n_tb_w);
    let sb_h = core::cmp::min(4, n_tb_h);
    (sb_w, sb_h)
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
            let xc = sx + dx;
            let yc = sy + dy;
            if (xc as usize) < n_tb_w && (yc as usize) < n_tb_h {
                out.push((xc, yc));
            }
        }
    }
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
