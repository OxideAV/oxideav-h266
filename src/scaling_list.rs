//! VVC scaling-list APS payload — §7.3.2.21 `scaling_list_data()`
//! parse + §7.4.3.20 `ScalingMatrixRec` / `ScalingMatrixDcRec`
//! derivation.
//!
//! The 28 scaling matrices are indexed by the §8.7.3 Table 38 `id`
//! ([`crate::dequant::scaling_matrix_id`]): ids 0..2 are 2×2
//! (inter-chroma only), 2..8 are 4×4, 8..28 are 8×8; ids ≥ 14
//! additionally carry an explicit DC that eq. 1150 substitutes at
//! `m[0][0]`; ids > 25 (the 64-point luma matrices) skip the
//! bottom-right 4×4 quadrant of their 8×8 pattern on the wire (the
//! §8.7.3 NOTE zero-out keeps those samples unused).
//!
//! Derivation (§7.4.3.20):
//!
//! * `scaling_list_copy_mode_flag[id]` — infer 1 when absent (the
//!   `aps_chroma_present_flag == 0` chroma ids).
//! * prediction source `scalingMatrixPred` / `scalingMatrixDcPred`:
//!   all-8 when copy = pred = 0 (explicit deltas ride on 8), all-16
//!   when `scaling_list_pred_id_delta == 0` (the flat default), else
//!   `ScalingMatrixRec[refId]` with `refId = id − delta` (eq. 102) and
//!   the DC from `ScalingMatrixDcRec[refId − 14]` when `refId > 13`
//!   else `scalingMatrixPred[0][0]`.
//! * eq. 104: `ScalingMatrixDcRec[id − 14] = (dcPred + dc_coef) & 255`.
//! * eq. 105: `ScalingMatrixRec[id][x][y] = (pred[x][y] +
//!   ScalingList[id][k]) & 255` over the matrix's own diagonal scan.
//!
//! Spec reference: ITU-T H.266 | ISO/IEC 23090-3 (V4, 01/2026).

use oxideav_core::{Error, Result};

use crate::bitreader::BitReader;
use crate::scan::diag_scan_order;

/// Number of scaling matrices (§7.3.2.21 loop bound).
pub const NUM_SCALING_IDS: usize = 28;

/// eq. 103 — `matrixSize` for a scaling-matrix `id`.
#[inline]
pub fn matrix_size(id: usize) -> usize {
    if id < 2 {
        2
    } else if id < 8 {
        4
    } else {
        8
    }
}

/// Fully-derived §7.4.3.20 scaling matrices, ready for the §8.7.3
/// eq. 1149 / 1150 dequant lookup.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScalingListData {
    /// `ScalingMatrixRec[id]` — row-major `matrixSize²` entries
    /// (`m[y * size + x]`), one vec per id in 0..28.
    pub scaling_matrix_rec: Vec<Vec<u8>>,
    /// `ScalingMatrixDcRec[id − 14]` for id in 14..28 (eq. 1150).
    pub scaling_matrix_dc_rec: [u8; 14],
}

impl ScalingListData {
    /// The all-16 flat set — equivalent to the §8.7.3 `m[x][y] = 16`
    /// arm, useful as a neutral binding in tests.
    pub fn flat16() -> Self {
        Self {
            scaling_matrix_rec: (0..NUM_SCALING_IDS)
                .map(|id| vec![16u8; matrix_size(id) * matrix_size(id)])
                .collect(),
            scaling_matrix_dc_rec: [16u8; 14],
        }
    }
}

/// Parse the §7.3.2.21 `scaling_list_data()` payload and run the
/// §7.4.3.20 derivations. `aps_chroma_present_flag` gates which ids
/// are transmitted (absent ids take the copy-mode / delta-0 inference,
/// i.e. the flat-16 default).
pub fn parse_scaling_list_data(
    br: &mut BitReader<'_>,
    aps_chroma_present_flag: bool,
) -> Result<ScalingListData> {
    // The parse-side zero-out check uses the 8×8 diagonal scan
    // (`DiagScanOrder[3][3]`) regardless of matrixSize; the id > 25
    // condition can only fire for 8×8 matrices anyway.
    let scan8 = diag_scan_order(8, 8);

    let mut rec: Vec<Vec<u8>> = Vec::with_capacity(NUM_SCALING_IDS);
    let mut dc_rec = [0u8; 14];

    for id in 0..NUM_SCALING_IDS {
        let size = matrix_size(id);
        let n = size * size;

        // Presence condition (§7.3.2.21).
        let present = aps_chroma_present_flag || id % 3 == 2 || id == 27;

        // Inferred defaults: copy = 1, pred = 0, pred_id_delta = 0.
        let mut copy_mode = true;
        let mut pred_mode = false;
        let mut pred_id_delta = 0u32;
        let mut dc_coef: i32 = 0;
        // `ScalingList[id][i]` accumulator (§7.3.2.21); all-zero under
        // copy mode.
        let mut scaling_list = vec![0i32; n];

        if present {
            copy_mode = br.u1()? == 1;
            if !copy_mode {
                pred_mode = br.u1()? == 1;
            }
            if (copy_mode || pred_mode) && id != 0 && id != 2 && id != 8 {
                pred_id_delta = br.ue()?;
                // eq. 101 — maxIdDelta range check.
                let max_id_delta = if id < 2 {
                    id
                } else if id < 8 {
                    id - 2
                } else {
                    id - 8
                } as u32;
                if pred_id_delta > max_id_delta {
                    return Err(Error::invalid(format!(
                        "h266 scaling list: scaling_list_pred_id_delta[{id}] = {pred_id_delta} \
                         exceeds maxIdDelta = {max_id_delta}"
                    )));
                }
            }
            if !copy_mode {
                let mut next_coef: i32 = 0;
                if id > 13 {
                    let d = br.se()?;
                    if !(-128..=127).contains(&d) {
                        return Err(Error::invalid(format!(
                            "h266 scaling list: scaling_list_dc_coef[{}] = {d} out of range",
                            id - 14
                        )));
                    }
                    dc_coef = d;
                    next_coef += d;
                }
                for (i, sl) in scaling_list.iter_mut().enumerate() {
                    let (x, y) = scan8[i];
                    if !(id > 25 && x >= 4 && y >= 4) {
                        let d = br.se()?;
                        if !(-128..=127).contains(&d) {
                            return Err(Error::invalid(format!(
                                "h266 scaling list: scaling_list_delta_coef[{id}][{i}] = {d} \
                                 out of range"
                            )));
                        }
                        next_coef += d;
                    }
                    *sl = next_coef;
                }
            } else if id > 13 {
                // Copy mode: dc_coef inferred 0 (the DC predicts through).
                dc_coef = 0;
            }
        }

        // §7.4.3.20 — prediction source.
        let (pred, dc_pred): (Vec<i32>, i32) = if !copy_mode && !pred_mode {
            (vec![8i32; n], 8)
        } else if pred_id_delta == 0 {
            (vec![16i32; n], 16)
        } else {
            let ref_id = id - pred_id_delta as usize; // eq. 102
            let ref_rec = &rec[ref_id];
            // The reference matrix has the same matrixSize: ids within
            // one size class only reference each other (eq. 101 bounds
            // delta below the class base).
            let pred: Vec<i32> = ref_rec.iter().map(|&v| v as i32).collect();
            let dc_pred = if ref_id > 13 {
                dc_rec[ref_id - 14] as i32
            } else {
                pred[0]
            };
            (pred, dc_pred)
        };

        // eq. 104 — DC reconstruction.
        if id > 13 {
            let dc = (dc_pred + dc_coef) & 255;
            if dc == 0 {
                return Err(Error::invalid(format!(
                    "h266 scaling list: ScalingMatrixDcRec[{}] must be > 0",
                    id - 14
                )));
            }
            dc_rec[id - 14] = dc as u8;
        }

        // eq. 105 — matrix reconstruction over the matrix's own scan.
        let scan = diag_scan_order(size, size);
        let mut m = vec![0u8; n];
        for (k, &(x, y)) in scan.iter().enumerate() {
            let v = (pred[(y as usize) * size + (x as usize)] + scaling_list[k]) & 255;
            if v == 0 {
                return Err(Error::invalid(format!(
                    "h266 scaling list: ScalingMatrixRec[{id}][{x}][{y}] must be > 0"
                )));
            }
            m[(y as usize) * size + (x as usize)] = v as u8;
        }
        rec.push(m);
    }

    Ok(ScalingListData {
        scaling_matrix_rec: rec,
        scaling_matrix_dc_rec: dc_rec,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: bit-level writer for hand-built scaling_list_data()
    /// payloads (u(1) / ue(v) / se(v)).
    struct Bw {
        bits: Vec<u8>,
    }
    impl Bw {
        fn new() -> Self {
            Self { bits: Vec::new() }
        }
        fn u1(&mut self, b: u8) {
            self.bits.push(b & 1);
        }
        fn ue(&mut self, v: u32) {
            let vp1 = v + 1;
            let n = 32 - vp1.leading_zeros();
            for _ in 0..(n - 1) {
                self.bits.push(0);
            }
            for i in (0..n).rev() {
                self.bits.push(((vp1 >> i) & 1) as u8);
            }
        }
        fn se(&mut self, v: i32) {
            let code = if v <= 0 {
                (-2 * v) as u32
            } else {
                (2 * v - 1) as u32
            };
            self.ue(code);
        }
        fn bytes(mut self) -> Vec<u8> {
            self.bits.push(1); // stop-ish pad
            while self.bits.len() % 8 != 0 {
                self.bits.push(0);
            }
            self.bits
                .chunks(8)
                .map(|c| c.iter().fold(0u8, |a, &b| (a << 1) | b))
                .collect()
        }
    }

    /// All-copy-mode payload (every transmitted id: copy = 1,
    /// delta = 0 where present) → every matrix flat 16, every DC 16.
    #[test]
    fn all_copy_mode_yields_flat_16() {
        let mut bw = Bw::new();
        for id in 0..NUM_SCALING_IDS {
            bw.u1(1); // scaling_list_copy_mode_flag = 1
            if id != 0 && id != 2 && id != 8 {
                bw.ue(0); // scaling_list_pred_id_delta = 0
            }
        }
        let bytes = bw.bytes();
        let mut br = BitReader::new(&bytes);
        let data = parse_scaling_list_data(&mut br, true).unwrap();
        assert_eq!(data, ScalingListData::flat16());
    }

    /// `aps_chroma_present_flag = 0`: only ids ≡ 2 (mod 3) and 27 are
    /// transmitted; the absent ids infer copy/delta-0 → flat 16.
    #[test]
    fn monochrome_transmits_luma_ids_only() {
        let mut bw = Bw::new();
        for id in 0..NUM_SCALING_IDS {
            if id % 3 == 2 || id == 27 {
                bw.u1(1);
                if id != 2 && id != 8 {
                    bw.ue(0);
                }
            }
        }
        let bytes = bw.bytes();
        let mut br = BitReader::new(&bytes);
        let data = parse_scaling_list_data(&mut br, false).unwrap();
        assert_eq!(data, ScalingListData::flat16());
    }

    /// Explicit mode with copy = pred = 0: the deltas ride on the all-8
    /// base (§7.4.3.20 first arm). id 0 is a 2×2 matrix: deltas
    /// +8, +2, −1, +1 over the diagonal scan (0,0),(0,1),(1,0),(1,1)
    /// accumulate to 16, 18, 17, 18 on the 8-base.
    #[test]
    fn explicit_mode_rides_on_eight_base() {
        let mut bw = Bw::new();
        // id 0 — explicit.
        bw.u1(0); // copy = 0
        bw.u1(0); // pred = 0
        bw.se(8);
        bw.se(2);
        bw.se(-1);
        bw.se(1);
        // ids 1..28 — copy mode, delta 0.
        for id in 1..NUM_SCALING_IDS {
            bw.u1(1);
            if id != 2 && id != 8 {
                bw.ue(0);
            }
        }
        let bytes = bw.bytes();
        let mut br = BitReader::new(&bytes);
        let data = parse_scaling_list_data(&mut br, true).unwrap();
        // Diag scan of 2×2: (0,0),(0,1),(1,0),(1,1) → row-major
        // m = [16, 17, 18, 18].
        assert_eq!(data.scaling_matrix_rec[0], vec![16u8, 17, 18, 18]);
        assert_eq!(data.scaling_matrix_rec[1], vec![16u8; 4]);
    }

    /// Copy mode with pred_id_delta > 0 clones the reference matrix
    /// (eq. 102), including the DC chain for ids > 13.
    #[test]
    fn copy_mode_with_delta_clones_reference() {
        let mut bw = Bw::new();
        for id in 0..NUM_SCALING_IDS {
            if id == 14 {
                // Explicit: DC = 16 + 4, matrix = 8-base + 12 then flat.
                bw.u1(0); // copy = 0
                bw.u1(0); // pred = 0
                bw.se(20); // dc_coef → nextCoef 20
                bw.se(-4); // first coef: 20 − 4 = 16 → rec 24
                for _ in 1..64 {
                    bw.se(0); // stay at 16 → rec 24 everywhere
                }
            } else if id == 15 {
                // Copy of id 14 (delta 1).
                bw.u1(1);
                bw.ue(1);
            } else {
                bw.u1(1);
                if id != 0 && id != 2 && id != 8 {
                    bw.ue(0);
                }
            }
        }
        let bytes = bw.bytes();
        let mut br = BitReader::new(&bytes);
        let data = parse_scaling_list_data(&mut br, true).unwrap();
        assert_eq!(data.scaling_matrix_rec[14], vec![24u8; 64]);
        // eq. 104 — DC: dcPred 8 (copy=pred=0 arm) + 20 = 28.
        assert_eq!(data.scaling_matrix_dc_rec[0], 28);
        // id 15 copies both the matrix and (via eq. 104 with dc_coef
        // inferred 0) the DC.
        assert_eq!(data.scaling_matrix_rec[15], vec![24u8; 64]);
        assert_eq!(data.scaling_matrix_dc_rec[1], 28);
    }

    /// ids > 25 skip the bottom-right 4×4 quadrant on the wire; the
    /// accumulator carries the last coefficient through the skipped
    /// positions.
    #[test]
    fn id_gt_25_skips_bottom_right_quadrant() {
        let mut bw = Bw::new();
        for id in 0..NUM_SCALING_IDS {
            if id == 27 {
                bw.u1(0); // copy = 0
                bw.u1(0); // pred = 0
                bw.se(10); // dc_coef
                           // 64 scan positions; 16 of them (x>=4 && y>=4) skipped.
                let scan = diag_scan_order(8, 8);
                let mut wrote_first = false;
                for &(x, y) in &scan {
                    if x >= 4 && y >= 4 {
                        continue;
                    }
                    if !wrote_first {
                        bw.se(-2); // nextCoef 10 − 2 = 8 → rec 16
                        wrote_first = true;
                    } else {
                        bw.se(0);
                    }
                }
            } else {
                bw.u1(1);
                if id != 0 && id != 2 && id != 8 {
                    bw.ue(0);
                }
            }
        }
        let bytes = bw.bytes();
        let mut br = BitReader::new(&bytes);
        let data = parse_scaling_list_data(&mut br, true).unwrap();
        // Every position (transmitted or skipped) accumulated 8 on the
        // 8-base → 16.
        assert_eq!(data.scaling_matrix_rec[27], vec![16u8; 64]);
        // DC: 8 + 10 = 18.
        assert_eq!(data.scaling_matrix_dc_rec[13], 18);
    }
}
