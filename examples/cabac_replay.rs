//! CABAC triage: replay an encoder bin log (`H266_ENC_DBG_BINS=1` on an
//! `encode_probe` run, stderr captured) through a fresh
//! `ArithEncoder` / `ArithDecoder` pair, check the regenerated code
//! word against the slice payload of the produced stream, and report
//! the first bin the decoder disagrees on.
//!
//! `cargo run --release --example cabac_replay -- <bin-log> <stream.266>`
use oxideav_h266::cabac::{ArithDecoder, ContextModel};
use oxideav_h266::cabac_enc::ArithEncoder;
use oxideav_h266::nal::{NalUnitType, extract_rbsp, iter_annex_b};
use std::collections::HashMap;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let log = std::fs::read_to_string(&args[1]).unwrap();
    let stream = std::fs::read(&args[2]).unwrap();
    // ops: (kind, ctx_key, bin)
    let mut ops: Vec<(u8, usize, u32)> = Vec::new();
    let mut keys: HashMap<String, usize> = HashMap::new();
    let mut init: Vec<ContextModel> = Vec::new();
    for line in log.lines() {
        if let Some(rest) = line.strip_prefix("E BIN ctx=(") {
            // (p0,p1) -> bin @addr sN
            let (st, rest) = rest.split_once(") -> ").unwrap();
            let (p0, p1) = st.split_once(',').unwrap();
            let mut it = rest.split_whitespace();
            let bin: u32 = it.next().unwrap().parse().unwrap();
            let addr = it.next().unwrap().to_string();
            let shift: u8 = it.next().unwrap()[1..].parse().unwrap();
            let n = keys.len();
            let key = *keys.entry(addr).or_insert(n);
            if key == init.len() {
                init.push(ContextModel {
                    p_state_idx0: p0.parse().unwrap(),
                    p_state_idx1: p1.parse().unwrap(),
                    shift_idx: shift,
                });
            }
            ops.push((0, key, bin));
        } else if let Some(rest) = line.strip_prefix("E BYP -> ") {
            ops.push((1, 0, rest.trim().parse().unwrap()));
        }
    }
    println!("{} ops, {} contexts", ops.len(), init.len());
    let mut ectx = init.clone();
    let mut enc = ArithEncoder::new();
    for &(k, c, b) in &ops {
        if k == 0 {
            enc.encode_decision(&mut ectx[c], b).unwrap();
        } else {
            enc.encode_bypass(b).unwrap();
        }
    }
    enc.encode_terminate(1).unwrap();
    let bytes = enc.finish();
    println!("replay cabac bytes: {}", bytes.len());
    // Compare with the stream's slice payload tail.
    let mut found = false;
    for nal in iter_annex_b(&stream) {
        if nal.header.nal_unit_type == NalUnitType::IdrNLp {
            let rbsp = extract_rbsp(nal.payload());
            println!("slice rbsp bytes: {}", rbsp.len());
            if rbsp.ends_with(&bytes) {
                println!("stream slice payload ENDS WITH the replayed cabac bytes (identical)");
            } else {
                // find longest common suffix / first differing byte from the start of cabac
                let off = rbsp.len() - bytes.len();
                let mut first = None;
                for i in 0..bytes.len() {
                    if rbsp[off + i] != bytes[i] {
                        first = Some(i);
                        break;
                    }
                }
                println!(
                    "stream payload DIFFERS from replay; first differing cabac byte {:?} of {}",
                    first,
                    bytes.len()
                );
            }
            found = true;
        }
    }
    assert!(found);
    // Decode the replay.
    let mut padded = bytes.clone();
    padded.extend_from_slice(&[0u8; 32]);
    let mut dec = ArithDecoder::new(&padded).unwrap();
    let mut dctx = init.clone();
    for (i, &(k, c, b)) in ops.iter().enumerate() {
        let got = if k == 0 {
            dec.decode_decision(&mut dctx[c]).unwrap()
        } else {
            dec.decode_bypass().unwrap()
        };
        if got != b {
            println!("REPLAY MISMATCH at op {i} (kind {k}): enc {b} dec {got}");
            return;
        }
    }
    println!(
        "replay decodes all {} bins; terminate = {}",
        ops.len(),
        dec.decode_terminate().unwrap()
    );
}
