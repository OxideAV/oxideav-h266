//! VUI + SEI payload fuzz entry (Rec. ITU-T H.274 §7 / §8 as H.266
//! carries them).
//!
//! Input layout: one selector byte, then the body.
//!
//! * selector 0 — the body is a `vui_payload()`:
//!   `VuiParameters::parse_payload`, and for an accepted payload the
//!   write → parse identity (the writer emits exactly what the parser
//!   understood).
//! * selector 1 — the body is an SEI RBSP: every message goes through
//!   `parse_sei_payload` as a prefix and as a suffix message; an
//!   accepted decoded picture hash is re-serialised and re-parsed.
//! * any other selector — the next two bytes are a `payloadType`
//!   (big-endian `u16`), the rest one message body, parsed both ways.
//!
//! Every path must return `Ok` / `Err` — no panics, no unbounded
//! allocation — regardless of input.

#![no_main]

use libfuzzer_sys::fuzz_target;
use oxideav_h266::sei_payload::{parse_sei_payload, SeiPayload};
use oxideav_h266::vui::VuiParameters;

fn check_message(payload_type: u32, body: &[u8]) {
    let _ = parse_sei_payload(payload_type, body, false);
    if let Ok(SeiPayload::DecodedPictureHash(h)) = parse_sei_payload(payload_type, body, true) {
        let again = parse_sei_payload(132, &h.write_payload(), true)
            .expect("a written picture hash parses");
        assert_eq!(again, SeiPayload::DecodedPictureHash(h));
    }
}

fuzz_target!(|data: &[u8]| {
    let Some((&selector, body)) = data.split_first() else {
        return;
    };
    match selector {
        0 => {
            if let Ok(vui) = VuiParameters::parse_payload(body) {
                let written = vui.write_payload();
                let again = VuiParameters::parse_payload(&written)
                    .expect("a written VUI payload parses");
                assert_eq!(again, vui);
            }
        }
        1 => {
            if let Ok(sei) = oxideav_h266::sei_rbsp::parse_sei_rbsp(body) {
                for m in &sei.messages {
                    check_message(m.payload_type, m.payload);
                }
            }
        }
        _ => {
            if body.len() >= 2 {
                let payload_type = u32::from(u16::from_be_bytes([body[0], body[1]]));
                check_message(payload_type, &body[2..]);
            }
        }
    }
});
