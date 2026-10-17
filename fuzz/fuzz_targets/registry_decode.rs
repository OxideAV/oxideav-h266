//! Registry decoder fuzz entry — the carriage a container hands the
//! `oxideav_core::Decoder`.
//!
//! Input layout: a big-endian `u16` byte count, then that many bytes
//! of `VvcDecoderConfigurationRecord` (ISO/IEC 14496-15 §11.2.4 — the
//! parameter-set arrays and the NAL length-prefix size), then the
//! coded payload. A zero count means no record: the payload is an
//! Annex B byte stream or 4-byte length-prefixed sample data, sniffed
//! per packet. The record becomes `CodecParameters::extradata`, the
//! payload one `Packet`, and the decoder is driven through
//! `make_decoder` → `send_packet` → `flush` → `receive_frame` — the
//! whole §7 / §8 / §9 decode (parameter sets, picture header, slice
//! data, CABAC, residuals, prediction, loop filters, POC output
//! order) runs on hostile parameter sets and slice payloads under a
//! 512 × 512 picture limit.
//!
//! Every path must return `Ok` / `Err` — no panics, no aborts, no
//! unbounded allocation — regardless of input.

#![no_main]

use libfuzzer_sys::fuzz_target;
use oxideav_core::{CodecId, CodecParameters, DecoderLimits, Error, Packet, TimeBase};

fuzz_target!(|data: &[u8]| {
    if data.len() < 2 {
        return;
    }
    let rec_len = usize::from(u16::from_be_bytes([data[0], data[1]]));
    let Some(rec) = data.get(2..2 + rec_len) else {
        return;
    };
    let payload = &data[2 + rec_len..];

    let mut params = CodecParameters::video(CodecId::new("h266"));
    params.extradata = rec.to_vec();
    params.limits = DecoderLimits::default().with_max_pixels_per_frame(512 * 512);
    let Ok(mut dec) = oxideav_h266::decoder::make_decoder(&params) else {
        return;
    };
    // Two packets: the payload, then its first half again (a seek-like
    // re-feed after `reset`), so the parameter-set pools and the DPB
    // state are exercised across packets.
    let _ = dec.send_packet(&Packet::new(0, TimeBase::new(1, 1), payload.to_vec()));
    while let Ok(_) | Err(Error::NeedMore) = dec.receive_frame() {
        if matches!(dec.receive_frame(), Err(Error::NeedMore)) {
            break;
        }
    }
    let _ = dec.reset();
    let half = &payload[..payload.len() / 2];
    let _ = dec.send_packet(&Packet::new(0, TimeBase::new(1, 1), half.to_vec()));
    let _ = dec.flush();
    while let Ok(_) = dec.receive_frame() {}
});
