//! `vvcC` inspection tool: build an ISO/IEC 14496-15 §11.2.4
//! `VvcDecoderConfigurationRecord` from an Annex B stream and re-frame
//! the stream as length-prefixed sample data, or dump a record.
//!
//! `cargo run --example vvcc_tool -- build IN.266 OUT.vvcC OUT.sample`
//! `cargo run --example vvcc_tool -- dump IN.vvcC`
use oxideav_h266::vvcc::{annex_b_to_length_prefixed, VvcDecoderConfigurationRecord};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("build") if args.len() == 5 => {
            let bs = std::fs::read(&args[2]).expect("read the Annex B stream");
            let rec = VvcDecoderConfigurationRecord::from_annex_b(&bs).expect("build the record");
            std::fs::write(&args[3], rec.to_bytes().expect("serialise")).expect("write vvcC");
            let sample =
                annex_b_to_length_prefixed(&bs, rec.length_size(), true).expect("re-frame");
            std::fs::write(&args[4], sample).expect("write sample");
            println!("{rec:#?}");
        }
        Some("dump") if args.len() == 3 => {
            let bytes = std::fs::read(&args[2]).expect("read the record");
            let rec = VvcDecoderConfigurationRecord::parse(&bytes).expect("parse the record");
            println!("{rec:#?}");
        }
        _ => {
            eprintln!("usage: vvcc_tool build IN.266 OUT.vvcC OUT.sample | dump IN.vvcC");
            std::process::exit(2);
        }
    }
}
