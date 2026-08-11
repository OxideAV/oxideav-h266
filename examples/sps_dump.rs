//! Debug aid: dump the full SPS tool-flag set of one conformance
//! stream (development aid for the corpus triage; the corpus dir comes
//! from `H266_CONFORMANCE_DIR` or the workspace docs staging path).

use oxideav_h266::nal::{extract_rbsp, iter_annex_b, NalUnitType};
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
    for nal in iter_annex_b(&bs) {
        if nal.header.nal_unit_type == NalUnitType::SpsNut {
            let sps = parse_sps(&extract_rbsp(nal.payload())).unwrap();
            println!(
                "{name}: ctu_size={} {:#?}",
                1u32 << (sps.sps_log2_ctu_size_minus5 + 5),
                sps.tool_flags
            );
            println!("partition: {:#?}", sps.partition_constraints);
            return;
        }
    }
}
