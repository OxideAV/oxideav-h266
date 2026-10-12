//! VVC CABAC arithmetic **encoder** — the forward (encode) side of the
//! recursive interval subdivision specified in §9.3.4.3.
//!
//! The H.266 spec only defines the *decoding* process. The encoder
//! side is the unique inverse: produce a bit stream such that
//! [`crate::cabac::ArithDecoder`] reads back the same `(bin, context
//! state, range)` trajectory.
//!
//! * Encode side of `DecodeDecision` → [`ArithEncoder::encode_decision`]
//! * Encode side of `DecodeBypass` → [`ArithEncoder::encode_bypass`]
//! * Encode side of `DecodeTerminate` → [`ArithEncoder::encode_terminate`]
//!   (a `1` flushes the code word, ending on the `rbsp_stop_one_bit`)
//! * [`ArithEncoder::finish`] byte-aligns (and, without a terminate,
//!   writes the register so a decoder lands inside the live interval)
//! * [`ArithEncoder::estimated_bits`] — fractional rate estimate for
//!   encoder decisions.
//!
//! # Implementation notes
//!
//! The math is the classical recursive-interval-subdivision arithmetic
//! encoder dual of the §9.3.4.3 decoder: a 10-bit interval lower bound
//! `low` (bit 9 is the carry position) next to the decoder's
//! `ivlCurrRange`, renormalised one bit at a time. A window bit whose
//! value is not yet determined — the interval straddles the midpoint —
//! is deferred as an *outstanding* bit and written, as the complement of
//! the bit that finally resolves it, once a later bin settles the
//! interval below (`0 1…1`) or carries it above (`1 0…0`) the midpoint.
//!
//! r474 — the pre-r474 engine instead kept every undetermined bit
//! inside a `u64 low`; the transform-skip tail of a repeating residual
//! (bins 1,1,0,0,… at `range = 310`) kept the interval straddling for
//! hundreds of bins, the window overflowed 64 bits and the slice tail
//! was corrupt (lossless 4:4:4 / 4:2:2 pictures failed to decode). The
//! bounded register has no such limit.
//!
//! No third-party VVC encoder source was consulted; the encoder is the
//! mechanical inverse of [`crate::cabac::ArithDecoder`]'s decode
//! functions.
//!
//! Spec reference: ITU-T H.266 | ISO/IEC 23090-3 (V4, 01/2026), §9.3.4.3.

use oxideav_core::Result;

use crate::cabac::ContextModel;

/// CABAC arithmetic encoder — the exact dual of [`ArithDecoder`]
/// (§9.3.4.3): the same `ivlCurrRange` evolution per decision /
/// bypass / terminate bin, so the byte stream it emits decodes
/// bin-for-bin.
///
/// r474 — the interval lower bound lives in a bounded 10-bit `low`
/// register with the classic *outstanding-bit* carry resolution: a bit
/// whose value is not yet determined (the interval straddles the
/// midpoint of the current window) is counted in `bits_outstanding`
/// and emitted once a later bin resolves it — as `0 1…1` when the
/// interval settles below the midpoint, as `1 0…0` when it carries
/// above. The pre-r474 engine kept every undetermined bit inside a
/// `u64 low`; a periodic bypass pattern (the transform-skip tail of a
/// repeating residual) can keep the interval straddling for hundreds
/// of bins, and the window overflowed 64 bits and corrupted the slice
/// tail (lossless 4:4:4 / 4:2:2 pictures failed to decode).
///
/// `precision` counts the window bits consumed so far (9 at init, +1
/// per renormalisation shift or bypass bin) for the fractional
/// [`Self::estimated_bits`] rate estimate.
pub struct ArithEncoder {
    /// 10-bit interval lower bound (bit 9 is the carry position).
    low: u32,
    /// `ivlCurrRange`, in [256, 510] after renormalisation.
    range: u32,
    /// Determined-but-unwritten bits awaiting carry resolution.
    bits_outstanding: u32,
    /// The very first PutBit is the virtual carry position above the
    /// decoder's 9-bit init window and is not written.
    first_bit: bool,
    /// Window bits consumed (rate estimate).
    precision: u32,
    /// Bits actually written to `data`.
    committed: u32,
    /// Output bytes (MSB-first packed).
    data: Vec<u8>,
    /// Bits packed into the partial tail of `data` (0..8).
    bit_pos: u8,
    finished: bool,
    terminated: bool,
}

impl Default for ArithEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl ArithEncoder {
    pub fn new() -> Self {
        Self {
            low: 0,
            range: 510,
            bits_outstanding: 0,
            first_bit: true,
            precision: 9,
            committed: 0,
            data: Vec::new(),
            bit_pos: 0,
            finished: false,
            terminated: false,
        }
    }

    pub fn is_terminated(&self) -> bool {
        self.terminated
    }

    /// Encode one decision bin, mirroring
    /// [`ArithDecoder::decode_decision`]: the LPS sub-interval from the
    /// context's state and `ivlCurrRange >> 5`, MPS / LPS selection,
    /// state update, renormalisation.
    pub fn encode_decision(&mut self, ctx: &mut ContextModel, bin: u32) -> Result<()> {
        debug_assert!(!self.finished);
        debug_assert!(bin <= 1);
        // r474 triage aid — mirrors the decoder's `H266_DBG_BINS` trace
        // (armed by the pipeline under `H266_ENC_DBG_BINS`).
        if crate::cabac::bin_trace_on() {
            eprintln!(
                "E BIN ctx=({},{}) -> {bin} @{:p} s{}",
                ctx.p_state_idx0, ctx.p_state_idx1, ctx as *const ContextModel, ctx.shift_idx
            );
        }
        let q_range_idx = self.range >> 5;
        let (val_mps, ivl_lps_range) = ctx.lps_range(q_range_idx);
        let ivl_mps_range = self.range - ivl_lps_range;
        if bin == val_mps {
            self.range = ivl_mps_range;
        } else {
            self.low += ivl_mps_range;
            self.range = ivl_lps_range;
        }
        ctx.update(bin);
        self.renormalize();
        Ok(())
    }

    /// Encode a bypass bin (§9.3.4.3.4 dual): the window shifts by one
    /// bit, the interval moves up by `range` for a 1.
    pub fn encode_bypass(&mut self, bin: u32) -> Result<()> {
        if crate::cabac::bin_trace_on() {
            eprintln!("E BYP -> {bin}");
        }
        debug_assert!(!self.finished);
        debug_assert!(bin <= 1);
        self.low <<= 1;
        if bin == 1 {
            self.low += self.range;
        }
        self.precision += 1;
        if self.low >= 1024 {
            self.put_bit(1);
            self.low -= 1024;
        } else if self.low < 512 {
            self.put_bit(0);
        } else {
            self.low -= 512;
            self.bits_outstanding += 1;
        }
        Ok(())
    }

    /// Encode a terminate bin (§9.3.4.3.5 dual). A `1` closes the
    /// arithmetic code word: the flush writes the interval's identifying
    /// bits with the final `1` being the `rbsp_stop_one_bit` the decoder
    /// consumes as its last bit.
    pub fn encode_terminate(&mut self, bin: u32) -> Result<()> {
        debug_assert!(!self.finished);
        debug_assert!(bin <= 1);
        self.range -= 2;
        if bin == 0 {
            self.renormalize();
        } else {
            self.terminated = true;
            self.low += self.range;
            // Flush: range = 2 renormalises through seven shifts (low's
            // bottom seven bits become zero), then the top register bit
            // and the two bits below it (the last forced to 1).
            self.range = 2;
            self.renormalize();
            self.put_bit((self.low >> 9) & 1);
            self.write_raw_bit(((self.low >> 8) & 1) as u8);
            self.write_raw_bit(1);
        }
        Ok(())
    }

    /// Renormalisation (decision / terminate paths): while `range <
    /// 256`, resolve or defer the window's top bit and double both
    /// registers.
    fn renormalize(&mut self) {
        while self.range < 256 {
            if self.low < 256 {
                self.put_bit(0);
            } else if self.low >= 512 {
                self.low -= 512;
                self.put_bit(1);
            } else {
                self.low -= 256;
                self.bits_outstanding += 1;
            }
            self.range <<= 1;
            self.low <<= 1;
            self.precision += 1;
        }
    }

    /// Emit a resolved bit followed by the outstanding bits it
    /// determines (their complement); the first call is the virtual
    /// carry position and writes nothing.
    fn put_bit(&mut self, bit: u32) {
        if self.first_bit {
            self.first_bit = false;
        } else {
            self.write_raw_bit(bit as u8);
        }
        while self.bits_outstanding > 0 {
            self.write_raw_bit((1 - bit) as u8);
            self.bits_outstanding -= 1;
        }
    }

    /// Append a single bit (MSB-first) to the byte stream.
    fn write_raw_bit(&mut self, bit: u8) {
        if self.bit_pos == 0 {
            self.data.push(0);
        }
        let shift = 7 - self.bit_pos;
        *self.data.last_mut().unwrap() |= (bit & 1) << shift;
        self.bit_pos += 1;
        if self.bit_pos == 8 {
            self.bit_pos = 0;
        }
        self.committed += 1;
    }

    /// Finalise the stream and return the bytes. After
    /// [`Self::encode_terminate`]`(1)` the code word is already complete
    /// (the flush ended on the stop bit) and only the byte alignment
    /// remains; without a terminate (rate probes, unit tests) the
    /// register's ten bits are written verbatim so a decoder lands
    /// inside the live interval.
    pub fn finish(&mut self) -> Vec<u8> {
        if self.finished {
            return std::mem::take(&mut self.data);
        }
        self.finished = true;
        if !self.terminated {
            self.put_bit((self.low >> 9) & 1);
            for i in (0..9).rev() {
                self.write_raw_bit(((self.low >> i) & 1) as u8);
            }
        }
        // Byte-align with zeros.
        while self.bit_pos != 0 {
            self.write_raw_bit_no_count(0);
        }
        std::mem::take(&mut self.data)
    }

    /// Append a bit but do not increment `committed` (byte-alignment
    /// tail in `finish`).
    fn write_raw_bit_no_count(&mut self, bit: u8) {
        if self.bit_pos == 0 {
            self.data.push(0);
        }
        let shift = 7 - self.bit_pos;
        *self.data.last_mut().unwrap() |= (bit & 1) << shift;
        self.bit_pos += 1;
        if self.bit_pos == 8 {
            self.bit_pos = 0;
        }
    }

    /// Number of bits written so far (excludes the byte-alignment
    /// tail). For tests.
    pub fn committed_bits(&self) -> u64 {
        self.committed as u64
    }

    /// r474 triage — `(low, range, precision, outstanding)`.
    #[doc(hidden)]
    pub fn debug_state(&self) -> (u64, u64, u32, u32) {
        (
            u64::from(self.low),
            u64::from(self.range),
            self.precision,
            self.bits_outstanding,
        )
    }

    /// r474 — fractional information content of everything coded so
    /// far: the live interval has width `range / 2^precision`, so the
    /// code length is `precision − log2(range)` bits (written bits plus
    /// the outstanding / partially-determined tail). Encoder-side rate
    /// estimation for mode / split decisions; the wire length after
    /// [`Self::finish`] rounds this up to whole bytes.
    pub fn estimated_bits(&self) -> f64 {
        f64::from(self.precision) - f64::from(self.range.max(1)).log2()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cabac::{ArithDecoder, ContextModel};

    /// r474 regression — a periodic bypass pattern can keep the coding
    /// interval straddling the window midpoint for hundreds of bins
    /// (the transform-skip tail of a repeating residual did exactly
    /// that at `range = 310` with the bins 1,1,0,0,…). The pre-r474
    /// engine accumulated every undetermined bit in a `u64` and
    /// overflowed; the outstanding-bit engine must round-trip any such
    /// run. Several decision prefixes vary the range the run starts at.
    #[test]
    fn periodic_bypass_runs_keep_straddling_without_overflow() {
        let patterns: [&[u32]; 6] = [
            &[1, 1, 0, 0],
            &[0, 0, 1, 1],
            &[1, 0],
            &[0, 1],
            &[1, 0, 0, 1],
            &[1, 1, 1, 0, 0, 0],
        ];
        for prefix_len in 0..64u32 {
            for (pi, pat) in patterns.iter().enumerate() {
                let mut ctx = ContextModel::init(0x29, 7, 30);
                let mut dctx = ctx;
                let mut enc = ArithEncoder::new();
                let mut bins: Vec<(bool, u32)> = Vec::new();
                for i in 0..prefix_len {
                    let b = u32::from(i % 5 == 0);
                    enc.encode_decision(&mut ctx, b).unwrap();
                    bins.push((false, b));
                }
                for i in 0..3000usize {
                    let b = pat[i % pat.len()];
                    enc.encode_bypass(b).unwrap();
                    bins.push((true, b));
                }
                enc.encode_terminate(1).unwrap();
                let mut bytes = enc.finish();
                bytes.extend_from_slice(&[0u8; 16]);
                let mut dec = ArithDecoder::new(&bytes).unwrap();
                for (i, &(bypass, b)) in bins.iter().enumerate() {
                    let got = if bypass {
                        dec.decode_bypass().unwrap()
                    } else {
                        dec.decode_decision(&mut dctx).unwrap()
                    };
                    assert_eq!(got, b, "prefix {prefix_len} pattern {pi} bin {i}");
                }
                assert_eq!(
                    dec.decode_terminate().unwrap(),
                    1,
                    "prefix {prefix_len} pattern {pi}"
                );
            }
        }
    }

    /// r474 — engine stress: long pseudo-random mixes of decision bins
    /// (a few contexts, skewed values) and bypass runs of up to ~120
    /// bins (the lossless `residual_ts_coding()` tails), encoded then
    /// decoded bin-for-bin.
    #[test]
    fn stress_mixed_decision_and_long_bypass_runs() {
        let mut seed = 0xC0FF_EE11u32;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed
        };
        for round in 0..6 {
            let n = 60_000 + round * 7_000;
            let mut ectx: Vec<ContextModel> = (0..8)
                .map(|i| ContextModel::init(20 + i * 5, 4, 30))
                .collect();
            let mut dctx = ectx.clone();
            let mut enc = ArithEncoder::new();
            let mut ops: Vec<(u8, u8, u32)> = Vec::with_capacity(n); // (kind, ctx, bin)
            let mut run = 0u32;
            while ops.len() < n {
                let r = next();
                if run > 0 {
                    run -= 1;
                    let bin = (r >> 3) & 1;
                    enc.encode_bypass(bin).unwrap();
                    ops.push((1, 0, bin));
                } else if r % 7 == 0 {
                    run = (r >> 8) % 120;
                } else {
                    let c = ((r >> 4) % 8) as u8;
                    // Skewed: mostly 0 for even contexts, mostly 1 for odd.
                    let bin = if (r >> 12) % 10 < 8 {
                        (c & 1) as u32
                    } else {
                        1 - (c & 1) as u32
                    };
                    enc.encode_decision(&mut ectx[c as usize], bin).unwrap();
                    ops.push((0, c, bin));
                }
            }
            enc.encode_terminate(1).unwrap();
            let mut bytes = enc.finish();
            bytes.extend_from_slice(&[0u8; 32]);
            let mut dec = ArithDecoder::new(&bytes).unwrap();
            for (i, &(kind, c, bin)) in ops.iter().enumerate() {
                let got = if kind == 0 {
                    dec.decode_decision(&mut dctx[c as usize]).unwrap()
                } else {
                    dec.decode_bypass().unwrap()
                };
                assert_eq!(got, bin, "round {round}: bin {i} (kind {kind}) mismatch");
            }
            assert_eq!(
                dec.decode_terminate().unwrap(),
                1,
                "round {round}: terminate"
            );
        }
    }

    fn pad(bytes: Vec<u8>) -> Vec<u8> {
        let mut out = bytes;
        out.extend_from_slice(&[0u8; 32]);
        out
    }

    #[test]
    fn bypass_round_trip_simple() {
        let bins = [0u32, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1];
        let mut enc = ArithEncoder::new();
        for &b in &bins {
            enc.encode_bypass(b).unwrap();
        }
        let bytes = pad(enc.finish());
        let mut dec = ArithDecoder::new(&bytes).unwrap();
        for (i, &b) in bins.iter().enumerate() {
            let got = dec.decode_bypass().unwrap();
            assert_eq!(got, b, "bypass mismatch at bit {i}");
        }
    }

    #[test]
    fn decision_round_trip_single_ctx() {
        let bins = [1u32, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1];
        let mut enc = ArithEncoder::new();
        let mut enc_ctx = ContextModel::init(0x10, 0, 32);
        for &b in &bins {
            enc.encode_decision(&mut enc_ctx, b).unwrap();
        }
        let final_enc_state = (enc_ctx.p_state_idx0, enc_ctx.p_state_idx1);
        let bytes = pad(enc.finish());
        let mut dec = ArithDecoder::new(&bytes).unwrap();
        let mut dec_ctx = ContextModel::init(0x10, 0, 32);
        for (i, &b) in bins.iter().enumerate() {
            let got = dec.decode_decision(&mut dec_ctx).unwrap();
            assert_eq!(got, b, "decision mismatch at bin {i}");
        }
        // Final context state must match — encoder and decoder both
        // walked the same bins through the same `update` math.
        assert_eq!(
            (dec_ctx.p_state_idx0, dec_ctx.p_state_idx1),
            final_enc_state,
            "final context state must match"
        );
    }

    #[test]
    fn long_mps_run() {
        let mut enc = ArithEncoder::new();
        let mut enc_ctx = ContextModel::init(0x3F, 0, 63);
        let n_bins = 64;
        for _ in 0..n_bins {
            enc.encode_decision(&mut enc_ctx, 1).unwrap();
        }
        let bytes = pad(enc.finish());
        let mut dec = ArithDecoder::new(&bytes).unwrap();
        let mut dec_ctx = ContextModel::init(0x3F, 0, 63);
        for i in 0..n_bins {
            let got = dec.decode_decision(&mut dec_ctx).unwrap();
            assert_eq!(got, 1, "expected MPS=1 at bin {i}");
        }
    }

    #[test]
    fn mixed_decision_bypass_round_trip() {
        let pattern: &[(bool, u32)] = &[
            (false, 1),
            (false, 0),
            (false, 1),
            (false, 1),
            (true, 0),
            (true, 1),
            (true, 0),
            (false, 0),
            (false, 1),
            (false, 1),
            (false, 0),
            (true, 1),
            (true, 1),
            (true, 0),
            (false, 1),
            (false, 0),
            (false, 1),
            (false, 1),
            (true, 0),
            (true, 0),
            (true, 1),
        ];
        let mut enc = ArithEncoder::new();
        let mut enc_ctx = ContextModel::init(0x18, 0, 30);
        for &(bypass, b) in pattern {
            if bypass {
                enc.encode_bypass(b).unwrap();
            } else {
                enc.encode_decision(&mut enc_ctx, b).unwrap();
            }
        }
        let bytes = pad(enc.finish());
        let mut dec = ArithDecoder::new(&bytes).unwrap();
        let mut dec_ctx = ContextModel::init(0x18, 0, 30);
        for (i, &(bypass, b)) in pattern.iter().enumerate() {
            let got = if bypass {
                dec.decode_bypass().unwrap()
            } else {
                dec.decode_decision(&mut dec_ctx).unwrap()
            };
            assert_eq!(got, b, "mismatch at step {i} (bypass={bypass})");
        }
    }

    #[test]
    fn terminate_zero_then_continue() {
        let mut enc = ArithEncoder::new();
        let mut ctx = ContextModel::init(0x20, 0, 26);
        enc.encode_decision(&mut ctx, 1).unwrap();
        enc.encode_decision(&mut ctx, 0).unwrap();
        enc.encode_terminate(0).unwrap();
        enc.encode_decision(&mut ctx, 1).unwrap();
        enc.encode_decision(&mut ctx, 1).unwrap();
        let bytes = pad(enc.finish());
        let mut dec = ArithDecoder::new(&bytes).unwrap();
        let mut dctx = ContextModel::init(0x20, 0, 26);
        assert_eq!(dec.decode_decision(&mut dctx).unwrap(), 1);
        assert_eq!(dec.decode_decision(&mut dctx).unwrap(), 0);
        assert_eq!(dec.decode_terminate().unwrap(), 0);
        assert!(!dec.is_terminated());
        assert_eq!(dec.decode_decision(&mut dctx).unwrap(), 1);
        assert_eq!(dec.decode_decision(&mut dctx).unwrap(), 1);
    }

    #[test]
    fn terminate_one_end_of_slice() {
        let mut enc = ArithEncoder::new();
        let mut ctx = ContextModel::init(0x20, 0, 26);
        enc.encode_decision(&mut ctx, 1).unwrap();
        enc.encode_decision(&mut ctx, 0).unwrap();
        enc.encode_decision(&mut ctx, 1).unwrap();
        enc.encode_terminate(1).unwrap();
        assert!(enc.is_terminated());
        let bytes = pad(enc.finish());
        let mut dec = ArithDecoder::new(&bytes).unwrap();
        let mut dctx = ContextModel::init(0x20, 0, 26);
        assert_eq!(dec.decode_decision(&mut dctx).unwrap(), 1);
        assert_eq!(dec.decode_decision(&mut dctx).unwrap(), 0);
        assert_eq!(dec.decode_decision(&mut dctx).unwrap(), 1);
        assert_eq!(dec.decode_terminate().unwrap(), 1);
        assert!(dec.is_terminated());
    }

    #[test]
    fn finish_emits_bits() {
        let mut enc = ArithEncoder::new();
        let mut ctx = ContextModel::init(0x18, 0, 30);
        for _ in 0..40 {
            enc.encode_decision(&mut ctx, 0).unwrap();
        }
        let bytes = enc.finish();
        assert!(!bytes.is_empty());
    }

    /// Encoding zero ops then finish: the decoder's init read should
    /// still be valid (less than 510). With `low = 0, range = 510`,
    /// any 9-bit prefix in [0, 510) works; we emit `low = 0` so the
    /// init read sees ivl_offset = 0.
    #[test]
    fn finish_with_no_ops() {
        let mut enc = ArithEncoder::new();
        let bytes = enc.finish();
        // Need at least 9 bits (init read).
        assert!(bytes.len() >= 2);
        let mut padded = bytes.clone();
        padded.extend_from_slice(&[0u8; 16]);
        // Decoder init must succeed.
        let _dec = ArithDecoder::new(&padded).unwrap();
    }

    /// Pseudo-random stress test: 4096 bins encoded across a mix of
    /// decision (with several distinct contexts) + bypass + occasional
    /// terminate(0) markers. Round-trip the entire stream and verify
    /// every bin matches.
    #[test]
    fn long_pseudorandom_stream_round_trips() {
        // Tiny linear-congruential PRNG for reproducibility.
        struct Lcg(u64);
        impl Lcg {
            fn next(&mut self) -> u32 {
                self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
                (self.0 >> 33) as u32
            }
        }
        let mut rng = Lcg(0xDEAD_BEEF_CAFE_BABE);

        // Build the operation list deterministically.
        enum Op {
            Decision(usize, u32),
            Bypass(u32),
            Terminate0,
        }
        let mut ops: Vec<Op> = Vec::with_capacity(4096);
        for _ in 0..4096 {
            let r = rng.next() % 16;
            match r {
                // 25% bypass.
                0..=3 => ops.push(Op::Bypass(rng.next() & 1)),
                // 6.25% terminate(0).
                4 => ops.push(Op::Terminate0),
                // 68.75% decision across 4 different contexts.
                _ => {
                    let ctx_id = (rng.next() & 3) as usize;
                    let bin = rng.next() & 1;
                    ops.push(Op::Decision(ctx_id, bin));
                }
            }
        }

        // Encode.
        let mut enc = ArithEncoder::new();
        let mut enc_ctxs = [
            ContextModel::init(0x10, 0, 30),
            ContextModel::init(0x18, 0, 30),
            ContextModel::init(0x24, 0, 30),
            ContextModel::init(0x3F, 0, 30),
        ];
        for op in &ops {
            match op {
                Op::Decision(c, b) => enc.encode_decision(&mut enc_ctxs[*c], *b).unwrap(),
                Op::Bypass(b) => enc.encode_bypass(*b).unwrap(),
                Op::Terminate0 => enc.encode_terminate(0).unwrap(),
            }
        }
        let bytes = pad(enc.finish());

        // Decode.
        let mut dec = ArithDecoder::new(&bytes).unwrap();
        let mut dec_ctxs = [
            ContextModel::init(0x10, 0, 30),
            ContextModel::init(0x18, 0, 30),
            ContextModel::init(0x24, 0, 30),
            ContextModel::init(0x3F, 0, 30),
        ];
        for (i, op) in ops.iter().enumerate() {
            match op {
                Op::Decision(c, b) => {
                    let got = dec.decode_decision(&mut dec_ctxs[*c]).unwrap();
                    assert_eq!(got, *b, "decision mismatch at op {i} (ctx {c})");
                }
                Op::Bypass(b) => {
                    let got = dec.decode_bypass().unwrap();
                    assert_eq!(got, *b, "bypass mismatch at op {i}");
                }
                Op::Terminate0 => {
                    let got = dec.decode_terminate().unwrap();
                    assert_eq!(got, 0, "terminate mismatch at op {i}");
                }
            }
        }
    }

    /// Encode a pure-MPS=0 long run from a strongly-biased context.
    /// Symmetric to [`long_mps_run`] but on the other branch — guards
    /// the LPS-path renormalisation as well, since with MPS=0 the
    /// `bin == val_mps` branch matches and `low` stays at 0 throughout.
    #[test]
    fn long_mps_zero_run() {
        let mut enc = ArithEncoder::new();
        let mut enc_ctx = ContextModel::init(0x00, 0, 0);
        let n_bins = 64;
        for _ in 0..n_bins {
            enc.encode_decision(&mut enc_ctx, 0).unwrap();
        }
        let bytes = pad(enc.finish());
        let mut dec = ArithDecoder::new(&bytes).unwrap();
        let mut dec_ctx = ContextModel::init(0x00, 0, 0);
        for i in 0..n_bins {
            let got = dec.decode_decision(&mut dec_ctx).unwrap();
            assert_eq!(got, 0, "expected MPS=0 at bin {i}");
        }
    }

    /// Integration: encode a synthetic residual-flag sequence using
    /// the actual `tables::init_contexts` bundles, then decode it back
    /// with the matching contexts. This mirrors the shape of the
    /// per-TB residual emit pipeline that future rounds will build —
    /// proving the CABAC encoder primitive plugs into the existing
    /// `SyntaxCtx` machinery without further glue.
    #[test]
    fn integration_synthetic_cbf_and_sig_flags() {
        use crate::tables::{init_contexts, SyntaxCtx};

        let slice_qp_y = 26;
        let mut enc_tu_y = init_contexts(SyntaxCtx::TuYCodedFlag, slice_qp_y);
        let mut enc_sig = init_contexts(SyntaxCtx::SigCoeffFlag, slice_qp_y);

        // Synthetic op list: tu_y_coded_flag = 1 (CBF set), then 8
        // sig_coeff_flag bins from the §9.3.4.2.8 ctxInc range. We pick
        // ctxIdx 0 throughout for simplicity.
        let cbf_bin = 1u32;
        let sig_bins = [1u32, 0, 0, 1, 1, 0, 1, 0];

        let mut enc = ArithEncoder::new();
        // Choose a representative ctxIdx (0) for both — the test only
        // needs encoder/decoder to walk the same indexed contexts.
        let cbf_idx = 0;
        let sig_idx = 0;
        enc.encode_decision(&mut enc_tu_y[cbf_idx], cbf_bin)
            .unwrap();
        for &b in &sig_bins {
            enc.encode_decision(&mut enc_sig[sig_idx], b).unwrap();
        }
        let bytes = pad(enc.finish());

        // Decode with a fresh pair of context arrays.
        let mut dec_tu_y = init_contexts(SyntaxCtx::TuYCodedFlag, slice_qp_y);
        let mut dec_sig = init_contexts(SyntaxCtx::SigCoeffFlag, slice_qp_y);
        let mut dec = ArithDecoder::new(&bytes).unwrap();
        let got_cbf = dec.decode_decision(&mut dec_tu_y[cbf_idx]).unwrap();
        assert_eq!(got_cbf, cbf_bin);
        for (i, &b) in sig_bins.iter().enumerate() {
            let got = dec.decode_decision(&mut dec_sig[sig_idx]).unwrap();
            assert_eq!(got, b, "sig_coeff_flag mismatch at i={i}");
        }
    }
}
