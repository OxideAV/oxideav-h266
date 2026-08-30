//! Debug aid: dump the full SPS tool-flag set of one conformance
//! stream, followed by every PPS's partition / loop-filter / chroma-QP
//! controls (development aid for the corpus triage; the corpus dir
//! comes from `H266_CONFORMANCE_DIR` or the workspace docs staging
//! path).

use oxideav_h266::nal::{extract_rbsp, iter_annex_b, NalUnitType};
use oxideav_h266::pps::parse_pps;
use oxideav_h266::sps::parse_sps;

fn main() {
    let name = std::env::args().nth(1).expect("stream name");
    let dir = std::env::var("H266_CONFORMANCE_DIR").unwrap_or_else(|_| {
        format!(
            "{}/../../docs/video/h266/conformance/streams",
            env!("CARGO_MANIFEST_DIR")
        )
    });
    let bs = std::fs::read(format!("{dir}/{name}.bit")).unwrap();
    let mut seen_sps = false;
    let mut seen_pps = std::collections::BTreeSet::new();
    for nal in iter_annex_b(&bs) {
        match nal.header.nal_unit_type {
            NalUnitType::SpsNut if !seen_sps => {
                seen_sps = true;
                let sps = parse_sps(&extract_rbsp(nal.payload())).unwrap();
                println!(
                    "{name}: ctu_size={} {:#?}",
                    1u32 << (sps.sps_log2_ctu_size_minus5 + 5),
                    sps.tool_flags
                );
                println!("partition: {:#?}", sps.partition_constraints);
            }
            NalUnitType::PpsNut => {
                let pps = parse_pps(&extract_rbsp(nal.payload())).unwrap();
                if !seen_pps.insert(pps.pps_pic_parameter_set_id) {
                    continue;
                }
                println!(
                    "PPS {}: no_partition={} rect_slice={} lf_across_slices={} \
                     cb_qp_offset={} cr_qp_offset={} slice_chroma_qp_offsets_present={} \
                     cu_chroma_qp_offset_list_enabled={} cb_list={:?} cr_list={:?} \
                     dbf_disabled={} beta/tc={} {} cb {} {} cr {} {} sao_in_ph={} alf_in_ph={} \
                     dbf_in_ph={} qp_delta_in_ph={} cu_qp_delta_enabled={} init_qp_minus26={}",
                    pps.pps_pic_parameter_set_id,
                    pps.pps_no_pic_partition_flag,
                    pps.pps_rect_slice_flag,
                    pps.pps_loop_filter_across_slices_enabled_flag,
                    pps.pps_cb_qp_offset,
                    pps.pps_cr_qp_offset,
                    pps.pps_slice_chroma_qp_offsets_present_flag,
                    pps.pps_cu_chroma_qp_offset_list_enabled_flag,
                    pps.pps_cb_qp_offset_list,
                    pps.pps_cr_qp_offset_list,
                    pps.pps_deblocking_filter_disabled_flag,
                    pps.pps_luma_beta_offset_div2,
                    pps.pps_luma_tc_offset_div2,
                    pps.pps_cb_beta_offset_div2,
                    pps.pps_cb_tc_offset_div2,
                    pps.pps_cr_beta_offset_div2,
                    pps.pps_cr_tc_offset_div2,
                    pps.pps_sao_info_in_ph_flag,
                    pps.pps_alf_info_in_ph_flag,
                    pps.pps_dbf_info_in_ph_flag,
                    pps.pps_qp_delta_info_in_ph_flag,
                    pps.pps_cu_qp_delta_enabled_flag,
                    pps.pps_init_qp_minus26,
                );
                if let Some(part) = &pps.partition {
                    println!(
                        "  tiles {}x{} cols={:?} rows={:?} lf_across_tiles={} slices={} \
                         slice_w_tiles={:?} slice_h_tiles={:?}",
                        part.num_tile_columns,
                        part.num_tile_rows,
                        part.col_width_ctbs,
                        part.row_height_ctbs,
                        part.pps_loop_filter_across_tiles_enabled_flag,
                        part.num_slices_in_pic,
                        part.slice_width_in_tiles,
                        part.slice_height_in_tiles,
                    );
                }
            }
            _ => {}
        }
    }
}
