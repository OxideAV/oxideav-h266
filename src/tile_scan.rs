//! §6.5.1 CTB raster / tile scanning derivations.
//!
//! [`TileScan::derive`] computes, from a parsed SPS + PPS pair, the
//! §6.5.1 tile geometry and slice → CTB maps the slice-data walkers
//! and the §7.4.8 `NumEntryPoints` derivation consume:
//!
//! * `TileColBdVal[]` / `TileRowBdVal[]` (eqs. 16 / 17),
//! * `CtbToTileColBd[]` / `ctbToTileColIdx[]` and the row mirrors
//!   (eqs. 18 / 19),
//! * `CtbAddrInSlice[i][j]` + `NumCtusInSlice[i]` for rectangular
//!   slice layouts (`AddCtbsToSlice`, eq. 22) — the
//!   single-slice-per-subpic arms (with and without SPS subpicture
//!   info, eq. 20) and the explicit rectangular-slice arm (eq. 21,
//!   replayed from the per-slice geometry the PPS parser derived),
//! * the §7.4.8 eq. 141 `NumEntryPoints` count (tile transitions +
//!   the WPP per-CTU-row arm under
//!   `sps_entropy_coding_sync_enabled_flag`).
//!
//! For raster-scan slice layouts (`pps_rect_slice_flag == 0`) the CTB
//! list of a slice is a slice-header-time derivation —
//! [`TileScan::raster_slice_ctbs`] takes the parsed
//! `sh_slice_address` (a tile index) + `sh_num_tiles_in_slice_minus1`.
//!
//! Spec reference: ITU-T H.266 | ISO/IEC 23090-3 (V4, 01/2026).

use crate::pps::PicParameterSet;
use crate::sps::SeqParameterSet;
use oxideav_core::{Error, Result};

/// Derived §6.5.1 tile / slice scan state for one SPS + PPS pair.
#[derive(Clone, Debug)]
pub struct TileScan {
    /// `PicWidthInCtbsY` / `PicHeightInCtbsY`.
    pub pic_width_in_ctbs: u32,
    pub pic_height_in_ctbs: u32,
    /// `NumTileColumns` / `NumTileRows`.
    pub num_tile_columns: u32,
    pub num_tile_rows: u32,
    /// `ColWidthVal[i]` / `RowHeightVal[j]` in CTBs.
    pub col_width_val: Vec<u32>,
    pub row_height_val: Vec<u32>,
    /// eqs. 16 / 17 — tile boundary positions (len = columns/rows + 1).
    pub tile_col_bd_val: Vec<u32>,
    pub tile_row_bd_val: Vec<u32>,
    /// eq. 18 — per `ctbAddrX` (len = PicWidthInCtbsY + 1): the left
    /// tile-column boundary and tile-column index.
    pub ctb_to_tile_col_bd: Vec<u32>,
    pub ctb_to_tile_col_idx: Vec<u32>,
    /// eq. 19 — the row mirrors (len = PicHeightInCtbsY + 1).
    pub ctb_to_tile_row_bd: Vec<u32>,
    pub ctb_to_tile_row_idx: Vec<u32>,
    /// `CtbAddrInSlice[i][j]` (picture raster-scan CTB addresses, in
    /// slice decoding order) for rectangular slice layouts. Empty for
    /// raster-scan layouts (derive per slice header via
    /// [`Self::raster_slice_ctbs`]).
    pub ctb_addr_in_slice: Vec<Vec<u32>>,
    /// True iff the PPS uses rectangular slices (or has no partition
    /// block at all — the single-slice degenerate case).
    pub rect_slices: bool,
}

impl TileScan {
    /// Run the §6.5.1 derivations for a parsed SPS + PPS pair.
    pub fn derive(sps: &SeqParameterSet, pps: &PicParameterSet) -> Result<Self> {
        let ctb_log2 = sps.sps_log2_ctu_size_minus5 as u32 + 5;
        let ctb_size = 1u32 << ctb_log2;
        let pic_w_ctbs = pps.pps_pic_width_in_luma_samples.div_ceil(ctb_size);
        let pic_h_ctbs = pps.pps_pic_height_in_luma_samples.div_ceil(ctb_size);

        // Tile geometry: from the PPS partition block; a PPS with
        // `pps_no_pic_partition_flag` is a single tile covering the
        // picture (§7.4.3.5).
        let (col_width_val, row_height_val) = match &pps.partition {
            Some(p) => (p.col_width_ctbs.clone(), p.row_height_ctbs.clone()),
            None => (vec![pic_w_ctbs], vec![pic_h_ctbs]),
        };
        let num_tile_columns = col_width_val.len() as u32;
        let num_tile_rows = row_height_val.len() as u32;
        if col_width_val.iter().sum::<u32>() != pic_w_ctbs
            || row_height_val.iter().sum::<u32>() != pic_h_ctbs
        {
            return Err(Error::invalid(
                "h266 tile scan: tile geometry does not cover the picture",
            ));
        }

        // eqs. 16 / 17.
        let mut tile_col_bd_val = Vec::with_capacity(num_tile_columns as usize + 1);
        tile_col_bd_val.push(0u32);
        for w in &col_width_val {
            tile_col_bd_val.push(tile_col_bd_val.last().unwrap() + w);
        }
        let mut tile_row_bd_val = Vec::with_capacity(num_tile_rows as usize + 1);
        tile_row_bd_val.push(0u32);
        for h in &row_height_val {
            tile_row_bd_val.push(tile_row_bd_val.last().unwrap() + h);
        }

        // eqs. 18 / 19.
        let mut ctb_to_tile_col_bd = Vec::with_capacity(pic_w_ctbs as usize + 1);
        let mut ctb_to_tile_col_idx = Vec::with_capacity(pic_w_ctbs as usize + 1);
        let mut tile_x = 0usize;
        for ctb_addr_x in 0..=pic_w_ctbs {
            if ctb_addr_x == tile_col_bd_val[tile_x + 1] {
                tile_x += 1;
            }
            ctb_to_tile_col_bd.push(tile_col_bd_val[tile_x]);
            ctb_to_tile_col_idx.push(tile_x as u32);
        }
        let mut ctb_to_tile_row_bd = Vec::with_capacity(pic_h_ctbs as usize + 1);
        let mut ctb_to_tile_row_idx = Vec::with_capacity(pic_h_ctbs as usize + 1);
        let mut tile_y = 0usize;
        for ctb_addr_y in 0..=pic_h_ctbs {
            if ctb_addr_y == tile_row_bd_val[tile_y + 1] {
                tile_y += 1;
            }
            ctb_to_tile_row_bd.push(tile_row_bd_val[tile_y]);
            ctb_to_tile_row_idx.push(tile_y as u32);
        }

        let mut scan = TileScan {
            pic_width_in_ctbs: pic_w_ctbs,
            pic_height_in_ctbs: pic_h_ctbs,
            num_tile_columns,
            num_tile_rows,
            col_width_val,
            row_height_val,
            tile_col_bd_val,
            tile_row_bd_val,
            ctb_to_tile_col_bd,
            ctb_to_tile_col_idx,
            ctb_to_tile_row_bd,
            ctb_to_tile_row_idx,
            ctb_addr_in_slice: Vec::new(),
            rect_slices: pps.pps_rect_slice_flag || pps.partition.is_none(),
        };

        // CtbAddrInSlice — rectangular layouts only.
        if scan.rect_slices {
            scan.derive_rect_slices(sps, pps)?;
        }
        Ok(scan)
    }

    /// `AddCtbsToSlice` (eq. 22): append the CTB rectangle
    /// `[start_x, stop_x) × [start_y, stop_y)` (CTB units) to slice
    /// `slice_idx` in raster order.
    fn add_ctbs_to_slice(&mut self, slice_idx: usize, sx: u32, ex: u32, sy: u32, ey: u32) {
        while self.ctb_addr_in_slice.len() <= slice_idx {
            self.ctb_addr_in_slice.push(Vec::new());
        }
        for ctb_y in sy..ey {
            for ctb_x in sx..ex {
                self.ctb_addr_in_slice[slice_idx].push(ctb_y * self.pic_width_in_ctbs + ctb_x);
            }
        }
    }

    fn derive_rect_slices(&mut self, sps: &SeqParameterSet, pps: &PicParameterSet) -> Result<()> {
        let Some(part) = &pps.partition else {
            // No partition block: one slice = one tile = the picture.
            self.add_ctbs_to_slice(0, 0, self.pic_width_in_ctbs, 0, self.pic_height_in_ctbs);
            return Ok(());
        };
        if pps.pps_single_slice_per_subpic_flag {
            if !sps.sps_subpic_info_present_flag {
                // One slice covering the whole picture, CTBs in TILE
                // scan order (the spec's j/i tile loops).
                for j in 0..self.num_tile_rows as usize {
                    for i in 0..self.num_tile_columns as usize {
                        let (sx, ex) = (self.tile_col_bd_val[i], self.tile_col_bd_val[i + 1]);
                        let (sy, ey) = (self.tile_row_bd_val[j], self.tile_row_bd_val[j + 1]);
                        self.add_ctbs_to_slice(0, sx, ex, sy, ey);
                    }
                }
                return Ok(());
            }
            // One slice per subpicture (eq. 20 + the AddCtbsToSlice
            // arms). Requires the SPS subpicture layout.
            let info = sps.subpic_info.as_ref().ok_or_else(|| {
                Error::invalid("h266 tile scan: sps_subpic_info_present_flag without subpic info")
            })?;
            for (i, sp) in info.subpics.iter().enumerate() {
                let left_x = sp.ctu_top_left_x;
                let right_x = left_x + sp.width_minus1;
                let top_y = sp.ctu_top_left_y;
                let bottom_y = top_y + sp.height_minus1;
                let width_in_tiles = self.ctb_to_tile_col_idx[right_x as usize] + 1
                    - self.ctb_to_tile_col_idx[left_x as usize];
                let height_in_tiles = self.ctb_to_tile_row_idx[bottom_y as usize] + 1
                    - self.ctb_to_tile_row_idx[top_y as usize];
                let top_tile_row = self.ctb_to_tile_row_idx[top_y as usize] as usize;
                let less_than_one_tile = height_in_tiles == 1
                    && sp.height_minus1 + 1 < self.row_height_val[top_tile_row];
                if less_than_one_tile {
                    // CTU rows inside one tile.
                    self.add_ctbs_to_slice(
                        i,
                        left_x,
                        left_x + sp.width_minus1 + 1,
                        top_y,
                        top_y + sp.height_minus1 + 1,
                    );
                } else {
                    let tile_x = self.ctb_to_tile_col_idx[left_x as usize] as usize;
                    let tile_y = top_tile_row;
                    for j in 0..height_in_tiles as usize {
                        for k in 0..width_in_tiles as usize {
                            let (sx, ex) = (
                                self.tile_col_bd_val[tile_x + k],
                                self.tile_col_bd_val[tile_x + k + 1],
                            );
                            let (sy, ey) = (
                                self.tile_row_bd_val[tile_y + j],
                                self.tile_row_bd_val[tile_y + j + 1],
                            );
                            self.add_ctbs_to_slice(i, sx, ex, sy, ey);
                        }
                    }
                }
            }
            return Ok(());
        }
        // Explicit rectangular slices — replay eq. 21 from the
        // per-slice geometry the PPS parser derived (top-left tile,
        // width/height in tiles, and for single-tile sub-slices the
        // CTU-row height + row offset inside the tile).
        let n = part.num_slices_in_pic as usize;
        if part.slice_top_left_tile_idx.len() != n
            || part.slice_width_in_tiles.len() != n
            || part.slice_height_in_tiles.len() != n
            || part.slice_height_in_ctus.len() != n
            || part.slice_ctb_row_offset_in_tile.len() != n
        {
            return Err(Error::invalid(
                "h266 tile scan: PPS per-slice geometry arrays inconsistent",
            ));
        }
        for i in 0..n {
            let tile_idx = part.slice_top_left_tile_idx[i];
            let tile_x = (tile_idx % self.num_tile_columns) as usize;
            let tile_y = (tile_idx / self.num_tile_columns) as usize;
            let w_tiles = part.slice_width_in_tiles[i] as usize;
            let h_tiles = part.slice_height_in_tiles[i] as usize;
            if tile_x + w_tiles > self.num_tile_columns as usize
                || tile_y + h_tiles > self.num_tile_rows as usize
            {
                return Err(Error::invalid(
                    "h266 tile scan: slice tile rectangle out of range",
                ));
            }
            let sub_tile = w_tiles == 1
                && h_tiles == 1
                && part.slice_height_in_ctus[i] < self.row_height_val[tile_y];
            if sub_tile {
                let sy = self.tile_row_bd_val[tile_y] + part.slice_ctb_row_offset_in_tile[i];
                let ey = sy + part.slice_height_in_ctus[i];
                if ey > self.tile_row_bd_val[tile_y + 1] {
                    return Err(Error::invalid(
                        "h266 tile scan: sub-tile slice overflows its tile",
                    ));
                }
                self.add_ctbs_to_slice(
                    i,
                    self.tile_col_bd_val[tile_x],
                    self.tile_col_bd_val[tile_x + 1],
                    sy,
                    ey,
                );
            } else {
                for j in 0..h_tiles {
                    for k in 0..w_tiles {
                        let (sx, ex) = (
                            self.tile_col_bd_val[tile_x + k],
                            self.tile_col_bd_val[tile_x + k + 1],
                        );
                        let (sy, ey) = (
                            self.tile_row_bd_val[tile_y + j],
                            self.tile_row_bd_val[tile_y + j + 1],
                        );
                        self.add_ctbs_to_slice(i, sx, ex, sy, ey);
                    }
                }
            }
        }
        Ok(())
    }

    /// Raster-scan slice layout (`pps_rect_slice_flag == 0`): the CTB
    /// list of the slice starting at tile `first_tile_idx` and
    /// covering `num_tiles` complete tiles in tile scan order
    /// (`sh_slice_address` / `sh_num_tiles_in_slice_minus1 + 1`).
    pub fn raster_slice_ctbs(&self, first_tile_idx: u32, num_tiles: u32) -> Result<Vec<u32>> {
        let total = self.num_tile_columns * self.num_tile_rows;
        if first_tile_idx + num_tiles > total {
            return Err(Error::invalid(format!(
                "h266 tile scan: raster slice tiles {first_tile_idx}+{num_tiles} exceed {total}"
            )));
        }
        let mut ctbs = Vec::new();
        for t in first_tile_idx..first_tile_idx + num_tiles {
            let tx = (t % self.num_tile_columns) as usize;
            let ty = (t / self.num_tile_columns) as usize;
            for ctb_y in self.tile_row_bd_val[ty]..self.tile_row_bd_val[ty + 1] {
                for ctb_x in self.tile_col_bd_val[tx]..self.tile_col_bd_val[tx + 1] {
                    ctbs.push(ctb_y * self.pic_width_in_ctbs + ctb_x);
                }
            }
        }
        Ok(ctbs)
    }

    /// §7.4.8 eq. 141 — `NumEntryPoints` for a slice whose CTBs (in
    /// decoding order, picture raster addresses) are `slice_ctbs`.
    /// Counts tile transitions plus, with
    /// `sps_entropy_coding_sync_enabled_flag`, every CTU-row change.
    /// The caller applies the `sps_entry_point_offsets_present_flag`
    /// gate.
    pub fn num_entry_points(&self, slice_ctbs: &[u32], entropy_sync: bool) -> u32 {
        let mut n = 0u32;
        for i in 1..slice_ctbs.len() {
            let x = (slice_ctbs[i] % self.pic_width_in_ctbs) as usize;
            let y = (slice_ctbs[i] / self.pic_width_in_ctbs) as usize;
            let px = (slice_ctbs[i - 1] % self.pic_width_in_ctbs) as usize;
            let py = (slice_ctbs[i - 1] / self.pic_width_in_ctbs) as usize;
            if self.ctb_to_tile_row_bd[y] != self.ctb_to_tile_row_bd[py]
                || self.ctb_to_tile_col_bd[x] != self.ctb_to_tile_col_bd[px]
                || (y != py && entropy_sync)
            {
                n += 1;
            }
        }
        n
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Synthetic 2-column x 2-row tile grid on an 8x4-CTB picture
    /// (columns 5 + 3 CTBs, rows 3 + 1): eqs. 16 – 19 anchors.
    fn two_by_two(pps: &mut PicParameterSet) {
        use crate::pps::PicPartition;
        pps.pps_no_pic_partition_flag = false;
        pps.pps_rect_slice_flag = true;
        pps.pps_single_slice_per_subpic_flag = true;
        pps.partition = Some(PicPartition {
            log2_ctu_size_minus5: 2,
            explicit_col_widths: vec![5],
            explicit_row_heights: vec![3],
            col_width_ctbs: vec![5, 3],
            row_height_ctbs: vec![3, 1],
            num_tile_columns: 2,
            num_tile_rows: 2,
            num_tiles_in_pic: 4,
            pps_loop_filter_across_tiles_enabled_flag: false,
            num_slices_in_pic: 1,
            slice_top_left_tile_idx: Vec::new(),
            num_slices_in_subpic: vec![1],
            tile_idx_delta_present_flag: false,
            slice_width_in_tiles: Vec::new(),
            slice_height_in_tiles: Vec::new(),
            slice_height_in_ctus: Vec::new(),
            slice_ctb_row_offset_in_tile: Vec::new(),
        });
    }

    fn sps_pps_8x4_ctbs() -> (SeqParameterSet, PicParameterSet) {
        // 128-CTB picture 1024x512 = 8x4 CTBs.
        let sps = crate::sps::test_minimal_sps(2, 1024, 512);
        let mut pps = crate::pps::test_minimal_pps(1024, 512);
        two_by_two(&mut pps);
        (sps, pps)
    }

    #[test]
    fn tile_boundary_maps_match_eqs_16_to_19() {
        let (sps, pps) = sps_pps_8x4_ctbs();
        let scan = TileScan::derive(&sps, &pps).unwrap();
        assert_eq!(scan.pic_width_in_ctbs, 8);
        assert_eq!(scan.pic_height_in_ctbs, 4);
        assert_eq!(scan.tile_col_bd_val, vec![0, 5, 8]);
        assert_eq!(scan.tile_row_bd_val, vec![0, 3, 4]);
        // eq. 18: ctbAddrX 0..4 → bd 0 idx 0; 5..7 → bd 5 idx 1; the
        // one-past sentinel entry (8) advances to the closing
        // boundary (the spec's literal loop increments tileX there).
        assert_eq!(scan.ctb_to_tile_col_bd, vec![0, 0, 0, 0, 0, 5, 5, 5, 8]);
        assert_eq!(scan.ctb_to_tile_col_idx, vec![0, 0, 0, 0, 0, 1, 1, 1, 2]);
        assert_eq!(scan.ctb_to_tile_row_bd, vec![0, 0, 0, 3, 4]);
        assert_eq!(scan.ctb_to_tile_row_idx, vec![0, 0, 0, 1, 2]);
    }

    /// Single slice per (absent) subpic: the slice's CTBs come in TILE
    /// scan order, not picture raster order.
    #[test]
    fn single_slice_ctbs_follow_tile_scan_order() {
        let (sps, pps) = sps_pps_8x4_ctbs();
        let scan = TileScan::derive(&sps, &pps).unwrap();
        assert_eq!(scan.ctb_addr_in_slice.len(), 1);
        let ctbs = &scan.ctb_addr_in_slice[0];
        assert_eq!(ctbs.len(), 32);
        // Tile (0,0) is 5x3: rows 0..3, columns 0..5 first.
        assert_eq!(&ctbs[0..5], &[0, 1, 2, 3, 4]);
        assert_eq!(&ctbs[5..10], &[8, 9, 10, 11, 12]);
        // After 15 CTBs the scan enters tile (1,0) at column 5.
        assert_eq!(ctbs[15], 5);
        // Every CTB appears exactly once.
        let mut seen = ctbs.clone();
        seen.sort_unstable();
        assert_eq!(seen, (0..32).collect::<Vec<u32>>());
    }

    /// eq. 141 — tiles only: entry points at every tile transition
    /// (4 tiles → 3 entry points); with WPP every CTU-row change
    /// inside a tile adds one more.
    #[test]
    fn num_entry_points_counts_tile_and_wpp_transitions() {
        let (sps, pps) = sps_pps_8x4_ctbs();
        let scan = TileScan::derive(&sps, &pps).unwrap();
        let ctbs = scan.ctb_addr_in_slice[0].clone();
        assert_eq!(scan.num_entry_points(&ctbs, false), 3);
        // WPP: tile (0,0) has 3 rows (2 extra), tile (1,0) 3 rows (2
        // extra), bottom tiles 1 row each → 3 + 4 = 7.
        assert_eq!(scan.num_entry_points(&ctbs, true), 7);
    }

    /// eq. 20 — single-slice-per-subpic with SPS subpicture info: two
    /// side-by-side subpictures (5 + 3 CTB columns matching the tile
    /// columns) each become one slice covering their tile column
    /// (whole tiles, so the eq. 20 tiles-in-subpic loops run); a third
    /// derivation with a subpicture shorter than its tile row takes
    /// the `subpicHeightLessThanOneTileFlag` CTU-row arm.
    #[test]
    fn single_slice_per_subpic_arms() {
        use crate::sps::{SubpicEntry, SubpicInfo};
        let (mut sps, pps) = sps_pps_8x4_ctbs();
        sps.sps_subpic_info_present_flag = true;
        sps.subpic_info = Some(SubpicInfo {
            num_subpics_minus1: 1,
            subpics: vec![
                SubpicEntry {
                    ctu_top_left_x: 0,
                    ctu_top_left_y: 0,
                    width_minus1: 4,
                    height_minus1: 3,
                    ..Default::default()
                },
                SubpicEntry {
                    ctu_top_left_x: 5,
                    ctu_top_left_y: 0,
                    width_minus1: 2,
                    height_minus1: 3,
                    ..Default::default()
                },
            ],
            ..Default::default()
        });
        let scan = TileScan::derive(&sps, &pps).unwrap();
        assert_eq!(scan.ctb_addr_in_slice.len(), 2);
        // Subpic 0: tile column 0 (5 wide), both tile rows = 20 CTBs,
        // in tile order (tile (0,0) then tile (0,1)).
        assert_eq!(scan.ctb_addr_in_slice[0].len(), 20);
        assert_eq!(scan.ctb_addr_in_slice[0][0], 0);
        // Subpic 1: tile column 1 (3 wide) = 12 CTBs starting at x=5.
        assert_eq!(scan.ctb_addr_in_slice[1].len(), 12);
        assert_eq!(scan.ctb_addr_in_slice[1][0], 5);
        // Together they cover the picture exactly once.
        let mut all: Vec<u32> = scan.ctb_addr_in_slice.iter().flatten().copied().collect();
        all.sort_unstable();
        assert_eq!(all, (0..32).collect::<Vec<u32>>());

        // Sub-tile-height subpicture: 2 CTU rows of the 3-row top-left
        // tile → the CTU-row arm (rows 0..2, columns 0..5).
        let (mut sps2, pps2) = sps_pps_8x4_ctbs();
        sps2.sps_subpic_info_present_flag = true;
        sps2.subpic_info = Some(SubpicInfo {
            num_subpics_minus1: 0,
            subpics: vec![SubpicEntry {
                ctu_top_left_x: 0,
                ctu_top_left_y: 0,
                width_minus1: 4,
                height_minus1: 1,
                ..Default::default()
            }],
            ..Default::default()
        });
        let scan2 = TileScan::derive(&sps2, &pps2).unwrap();
        assert_eq!(scan2.ctb_addr_in_slice[0].len(), 10);
        assert_eq!(&scan2.ctb_addr_in_slice[0][..5], &[0, 1, 2, 3, 4]);
        assert_eq!(&scan2.ctb_addr_in_slice[0][5..], &[8, 9, 10, 11, 12]);
    }

    /// Explicit rectangular slices from PPS-derived geometry: a 2x2
    /// tile grid split into a 2x1-tile top slice, one whole-tile
    /// slice, and a bottom-right tile subdivided into two CTU-row
    /// slices — CtbAddrInSlice must cover the picture once with the
    /// eq. 21/22 rectangles.
    #[test]
    fn explicit_rect_slices_cover_picture() {
        let (sps, mut pps) = sps_pps_8x4_ctbs();
        pps.pps_single_slice_per_subpic_flag = false;
        {
            let p = pps.partition.as_mut().unwrap();
            p.num_slices_in_pic = 4;
            p.slice_top_left_tile_idx = vec![0, 2, 3, 3];
            p.slice_width_in_tiles = vec![2, 1, 1, 1];
            p.slice_height_in_tiles = vec![1, 1, 1, 1];
            // Slice 0: both top tiles (3 CTU rows); slice 1: tile
            // (0,1) whole (1 row); slices 2/3: tile (1,1)... its row
            // is only 1 CTB tall, so use whole-tile heights instead:
            // make slice 2 cover the whole tile and slice 3 empty is
            // illegal — instead subdivide the TOP-RIGHT tile: swap
            // layout: slice 1 = tile 1 (3 rows) split 1+2 handled as
            // slices 1/2, slice 3 = bottom row (2 tiles wide).
            p.slice_top_left_tile_idx = vec![0, 1, 1, 2];
            p.slice_width_in_tiles = vec![1, 1, 1, 2];
            p.slice_height_in_tiles = vec![1, 1, 1, 1];
            p.slice_height_in_ctus = vec![3, 1, 2, 1];
            p.slice_ctb_row_offset_in_tile = vec![0, 0, 1, 0];
        }
        let scan = TileScan::derive(&sps, &pps).unwrap();
        assert_eq!(scan.ctb_addr_in_slice.len(), 4);
        // Slice 0: whole tile (0,0) = 5x3 = 15 CTBs.
        assert_eq!(scan.ctb_addr_in_slice[0].len(), 15);
        // Slice 1: first CTU row of tile (1,0) = 3 CTBs at x 5..8.
        assert_eq!(scan.ctb_addr_in_slice[1], vec![5, 6, 7]);
        // Slice 2: remaining 2 CTU rows of tile (1,0).
        assert_eq!(scan.ctb_addr_in_slice[2], vec![13, 14, 15, 21, 22, 23]);
        // Slice 3: the bottom tile row across both tiles (tile order).
        assert_eq!(scan.ctb_addr_in_slice[3].len(), 8);
        assert_eq!(scan.ctb_addr_in_slice[3][0], 24);
        let mut all: Vec<u32> = scan.ctb_addr_in_slice.iter().flatten().copied().collect();
        all.sort_unstable();
        assert_eq!(all, (0..32).collect::<Vec<u32>>());
    }

    /// Raster-scan slice helper: two tiles starting at tile 1 cover
    /// tile (1,0) then tile (0,1) in tile-index order.
    #[test]
    fn raster_slice_ctbs_walks_tiles_in_index_order() {
        let (sps, mut pps) = sps_pps_8x4_ctbs();
        pps.pps_rect_slice_flag = false;
        let scan = TileScan::derive(&sps, &pps).unwrap();
        let ctbs = scan.raster_slice_ctbs(1, 2).unwrap();
        // Tile 1 = (1,0): 3 rows x 3 cols = 9 CTBs, first is (5,0).
        assert_eq!(ctbs.len(), 9 + 5);
        assert_eq!(ctbs[0], 5);
        // Tile 2 = (0,1): 1 row x 5 cols, first is (0,3) = 24.
        assert_eq!(ctbs[9], 24);
        assert!(scan.raster_slice_ctbs(3, 2).is_err());
    }
}
