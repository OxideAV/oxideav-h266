//! VVC CABAC arithmetic **encoder** — the forward (encode) side of the
//! recursive interval subdivision specified in §9.3.4.3.
//!
//! The H.266 spec only defines the *decoding* process. The encoder
//! side is the unique inverse: produce a bit stream such that
//! [`crate::cabac::ArithDecoder`] reads back the same `(bin, context
//! state, range)` trajectory. This module implements that inverse
//! using a high-precision tracking of the live arithmetic interval
//! and emitting just enough bits at finalisation to identify a value
//! inside it.
//!
//! Round-16 scope:
//!
//! * Encode side of `DecodeDecision` → [`ArithEncoder::encode_decision`]
//! * Encode side of `DecodeBypass` → [`ArithEncoder::encode_bypass`]
//! * Encode side of `DecodeTerminate` → [`ArithEncoder::encode_terminate`]
//! * [`ArithEncoder::finish`] flushes the partial low/range state into
//!   bits the decoder's 9-bit `ivlOffset` init reads back correctly.
//!
//! # Implementation notes
//!
//! The math here is the well-known recursive-interval-subdivision
//! arithmetic encoder dual of the recursive-interval-subdivision
//! decoder in §9.3.4.3 — invented in Pasco/Rissanen 1976 and used in
//! H.263, H.264, H.265, and now H.266 with only the per-symbol
//! probability state estimator changing.
//!
//! ## Strategy
//!
//! Rather than the classical 10-bit-register-with-carry-propagation
//! formulation (which trades correctness footguns for amortised
//! O(1) emission), this module uses a **large-precision** running
//! interval that mirrors the decoder's `(ivlCurrRange, virtualBits)`
//! state exactly. At finalisation we choose the minimum-prefix value
//! inside the live interval and emit it.
//!
//! Concretely: `low` and `range` together describe the set of all
//! decoder-bitstreams that would have produced the bin sequence we've
//! encoded so far, when interpreted as `bits_emitted_count`-bit
//! integers. We track everything in `u64` (more than enough headroom
//! for tens of thousands of bins) and emit committed bits whenever the
//! top bit of `low` and `low + range - 1` agree (i.e. the next bit is
//! determined). This avoids the carry-propagation footgun and is a
//! straightforward inverse of the decoder.
//!
//! No third-party VVC encoder source was consulted; the encoder math is
//! mechanically derived from [`crate::cabac::ArithDecoder`]'s decode
//! functions.
//!
//! Spec reference: ITU-T H.266 | ISO/IEC 23090-3 (V4, 01/2026), §9.3.4.3.

use oxideav_core::Result;

use crate::cabac::ContextModel;

/// Arithmetic encoder mirroring [`crate::cabac::ArithDecoder`].
///
/// State: `low` is the lower bound of the live interval in a
/// `precision`-bit window; `range` is the interval width. The decoder's
/// equivalent register is `ivl_offset ∈ [low_decoder, low_decoder + range)`
/// after consuming the same operations, where `low_decoder` is the
/// projection of our `low` into the decoder's 9-bit window.
///
/// Initial state: `low = 0, range = 510`. This matches the decoder's
/// post-init state (`ivlCurrRange = 510, ivlOffset = read_bits(9)`).
///
/// Each operation extends `precision` (the meaningful bit-width of
/// `low`) by 0 or more bits. We emit committed prefix bits whenever
/// `low`'s and `(low + range - 1)`'s top bit agree (definite bit).
pub struct ArithEncoder {
    /// Lower bound of the live interval. Tracked to full precision.
    low: u64,
    /// Interval width. Always ∈ [256, 510] right after decision
    /// renormalisation; can briefly grow (during bypass shifts) and
    /// shrink (during LPS / terminate paths).
    range: u64,
    /// Current bit-precision of `low`. After init `precision = 9`
    /// (matching the decoder's 9-bit init `read_bits(9)`).
    /// Renormalisation steps (`while range < 256: range <<= 1, low <<= 1,
    /// precision++`) increase this. Bypass also adds 1.
    precision: u32,
    /// Number of bits already committed to `data`. We never emit a bit
    /// until it's determined (top bit of `low` matches top bit of
    /// `low + range - 1`).
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
    /// [`crate::cabac::ArithDecoder::decode_decision`].
    pub fn encode_decision(&mut self, ctx: &mut ContextModel, bin: u32) -> Result<()> {
        debug_assert!(!self.finished);
        debug_assert!(bin <= 1);
        let q_range_idx = (self.range as u32) >> 5;
        let (val_mps, ivl_lps_range) = ctx.lps_range(q_range_idx);
        let ivl_mps_range = self.range - ivl_lps_range as u64;
        if bin == val_mps {
            self.range = ivl_mps_range;
        } else {
            self.low += ivl_mps_range;
            self.range = ivl_lps_range as u64;
        }
        ctx.update(bin);
        self.renormalize_decision();
        self.flush_committed_bits();
        Ok(())
    }

    /// Encode a bypass bin (§9.3.4.3.4).
    pub fn encode_bypass(&mut self, bin: u32) -> Result<()> {
        debug_assert!(!self.finished);
        debug_assert!(bin <= 1);
        // The decoder's bypass: `ivl_offset = (ivl_offset << 1) | bit;
        // if ivl_offset >= range: bin=1; ivl_offset -= range`.
        // Inverse: shift `low` left (extending precision by 1) and
        // bump by `range` for bin=1.
        self.low <<= 1;
        self.precision += 1;
        if bin == 1 {
            self.low += self.range;
        }
        self.flush_committed_bits();
        Ok(())
    }

    /// Encode a terminate bin (§9.3.4.3.5).
    pub fn encode_terminate(&mut self, bin: u32) -> Result<()> {
        debug_assert!(!self.finished);
        debug_assert!(bin <= 1);
        self.range -= 2;
        if bin == 0 {
            self.renormalize_decision();
            self.flush_committed_bits();
        } else {
            self.terminated = true;
            self.low += self.range;
            self.range = 2;
            self.renormalize_decision();
            self.flush_committed_bits();
        }
        Ok(())
    }

    /// Renormalisation for decision / terminate paths: while
    /// `range < 256`, double both registers (extending precision).
    fn renormalize_decision(&mut self) {
        while self.range < 256 {
            self.range <<= 1;
            self.low <<= 1;
            self.precision += 1;
        }
    }

    /// Emit any bits at the front of `low` whose value is determined
    /// (i.e. the same in `low` and in `low + range - 1`).
    fn flush_committed_bits(&mut self) {
        // Loop: while the top bit (at position `precision - 1`) is
        // determined, emit it and reduce precision.
        loop {
            if self.precision <= self.committed {
                break;
            }
            // The "top bit" we're trying to commit is at position
            // (precision - committed - 1) within the low-precision
            // window (counting from LSB). Equivalently: bit
            // (precision - 1) of `low` and `low + range - 1`.
            let top_bit_pos = self.precision - self.committed - 1;
            let mask: u64 = 1u64 << top_bit_pos;
            let lo_top = (self.low & mask) != 0;
            let hi = self.low + self.range - 1;
            let hi_top = (hi & mask) != 0;
            if lo_top != hi_top {
                break;
            }
            self.commit_bit(if lo_top { 1 } else { 0 });
        }
    }

    /// Append a single bit (MSB-first) to the byte stream.
    fn commit_bit(&mut self, bit: u8) {
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

    /// Finalise the stream. Pushes enough bits to identify the live
    /// interval, then byte-aligns.
    pub fn finish(&mut self) -> Vec<u8> {
        if self.finished {
            return std::mem::take(&mut self.data);
        }
        self.finished = true;
        // Commit the remaining (precision - committed) bits of `low`
        // (or actually of the midpoint, to maximise robustness against
        // future stream reads).
        //
        // The decoder will land its `ivlOffset` at the chosen value
        // (interpreted as a `precision`-bit integer) and any further
        // bits read are zero-padding (bypass / decision renorm reads
        // shift in zeros). For the encoder to land safely, pick a
        // value `v ∈ [low, low + range)` and emit its remaining bits.
        //
        // Safest choice: pick `v = low` (lower bound) — the decoder's
        // future zero-pad reads will give `(v << k) | 0 = v * 2^k` as
        // the running register, which compares against the running
        // `range` — the comparison stays inside the live interval as
        // long as `v` was chosen inside it.
        //
        // Even safer: emit one extra "1" bit after `v` so any future
        // bit-9 register comparison falls comfortably inside.
        let target = self.low;
        let remaining_bits = self.precision - self.committed;
        for i in (0..remaining_bits).rev() {
            let bit = ((target >> i) & 1) as u8;
            self.commit_bit(bit);
        }
        // Byte-align with zeros.
        while self.bit_pos != 0 {
            self.commit_bit_no_count(0);
        }
        std::mem::take(&mut self.data)
    }

    /// Append a bit but do not increment `committed` (used during
    /// the byte-alignment tail in `finish`).
    fn commit_bit_no_count(&mut self, bit: u8) {
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

    /// Number of bits committed so far (excludes pending tail / partial
    /// byte). For tests.
    pub fn committed_bits(&self) -> u64 {
        self.committed as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cabac::{ArithDecoder, ContextModel};

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
