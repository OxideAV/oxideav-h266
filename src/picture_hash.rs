//! Decoded picture hash — the three methods of the decoded picture
//! hash SEI message (Rec. ITU-T H.274 §8.8.2): MD5, CRC and checksum
//! over each colour component of a decoded picture.
//!
//! The picture data of a component is its samples in raster order,
//! one byte per sample at 8 bits and two bytes (low byte first) above
//! (H.274 eq. 33). [`component_hash`] computes one component;
//! [`picture_hash`] the one or three components of a
//! [`PictureBuffer`]; [`verify`] compares against a received message.
//!
//! The MD5 here is the crate's own (the same routine the conformance
//! harnesses have always checked their per-plane digests with); the
//! CRC and the checksum are the H.274 eq. 35 / eq. 36 loops.

use crate::reconstruct::{PictureBuffer, PicturePlane};
use crate::sei_payload::{DecodedPictureHash, PictureHashValue};

/// Per-round left-rotation amounts of MD5.
const SHIFTS: [u32; 64] = [
    7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, //
    5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, //
    4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, //
    6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
];

/// The 64 additive constants, `floor(|sin(i + 1)| · 2^32)`.
fn sine_table() -> &'static [u32; 64] {
    static TABLE: std::sync::OnceLock<[u32; 64]> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| {
        let mut k = [0u32; 64];
        for (i, slot) in k.iter_mut().enumerate() {
            *slot = (((i as f64 + 1.0).sin().abs()) * 4294967296.0) as u32;
        }
        k
    })
}

/// Streaming MD5.
#[derive(Clone)]
pub struct Md5 {
    state: [u32; 4],
    block: [u8; 64],
    filled: usize,
    len: u64,
}

impl Default for Md5 {
    fn default() -> Self {
        Self::new()
    }
}

impl Md5 {
    pub fn new() -> Self {
        Self {
            state: [0x6745_2301, 0xefcd_ab89, 0x98ba_dcfe, 0x1032_5476],
            block: [0; 64],
            filled: 0,
            len: 0,
        }
    }

    pub fn update(&mut self, mut data: &[u8]) {
        self.len = self.len.wrapping_add(data.len() as u64);
        if self.filled > 0 {
            let take = (64 - self.filled).min(data.len());
            self.block[self.filled..self.filled + take].copy_from_slice(&data[..take]);
            self.filled += take;
            data = &data[take..];
            if self.filled < 64 {
                return;
            }
            let block = self.block;
            self.process(&block);
            self.filled = 0;
        }
        let (blocks, rest) = data.as_chunks::<64>();
        for block in blocks {
            self.process(block);
        }
        self.block[..rest.len()].copy_from_slice(rest);
        self.filled = rest.len();
    }

    fn process(&mut self, block: &[u8; 64]) {
        let k = sine_table();
        let mut m = [0u32; 16];
        for (slot, word) in m.iter_mut().zip(block.as_chunks::<4>().0) {
            *slot = u32::from_le_bytes(*word);
        }
        let [mut a, mut b, mut c, mut d] = self.state;
        for i in 0..64 {
            let (f, g) = match i / 16 {
                0 => ((b & c) | (!b & d), i),
                1 => ((d & b) | (!d & c), (5 * i + 1) % 16),
                2 => (b ^ c ^ d, (3 * i + 5) % 16),
                _ => (c ^ (b | !d), (7 * i) % 16),
            };
            let sum = a.wrapping_add(f).wrapping_add(k[i]).wrapping_add(m[g]);
            a = d;
            d = c;
            c = b;
            b = b.wrapping_add(sum.rotate_left(SHIFTS[i]));
        }
        self.state[0] = self.state[0].wrapping_add(a);
        self.state[1] = self.state[1].wrapping_add(b);
        self.state[2] = self.state[2].wrapping_add(c);
        self.state[3] = self.state[3].wrapping_add(d);
    }

    /// The 16-byte digest.
    pub fn finish(mut self) -> [u8; 16] {
        let bit_len = self.len.wrapping_mul(8);
        self.update(&[0x80]);
        while self.filled != 56 {
            self.update(&[0]);
        }
        self.update(&bit_len.to_le_bytes());
        debug_assert_eq!(self.filled, 0);
        let mut out = [0u8; 16];
        for (dst, word) in out.as_chunks_mut::<4>().0.iter_mut().zip(self.state) {
            *dst = word.to_le_bytes();
        }
        out
    }

    /// The digest as 32 lowercase hex digits.
    pub fn finish_hex(self) -> String {
        hex(&self.finish())
    }
}

/// MD5 of one buffer.
pub fn md5(data: &[u8]) -> [u8; 16] {
    let mut h = Md5::new();
    h.update(data);
    h.finish()
}

/// Lowercase hex.
pub fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write;
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        let _ = write!(out, "{b:02x}");
    }
    out
}

/// The hash method of a decoded picture hash SEI message
/// (`dph_sei_hash_type`, H.274 Table 9).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum PictureHashMethod {
    /// `dph_sei_hash_type = 0`.
    Md5,
    /// `dph_sei_hash_type = 1`.
    Crc,
    /// `dph_sei_hash_type = 2`.
    Checksum,
}

impl PictureHashMethod {
    /// The method of a `dph_sei_hash_type` code; `None` for the
    /// reserved values (a decoder ignores such a message).
    pub fn from_hash_type(hash_type: u8) -> Option<Self> {
        match hash_type {
            0 => Some(Self::Md5),
            1 => Some(Self::Crc),
            2 => Some(Self::Checksum),
            _ => None,
        }
    }

    /// The `dph_sei_hash_type` code.
    pub fn hash_type(self) -> u8 {
        match self {
            Self::Md5 => 0,
            Self::Crc => 1,
            Self::Checksum => 2,
        }
    }
}

/// H.274 eq. 35 — the 16-bit CRC over the picture data followed by
/// two zero bytes, generator 0x1021, initial value 0xFFFF, most
/// significant bit first.
#[derive(Clone, Copy)]
struct Crc16(u16);

impl Crc16 {
    fn new() -> Self {
        Self(0xffff)
    }

    #[inline]
    fn push(&mut self, byte: u8) {
        let mut crc = self.0;
        for bit in (0..8).rev() {
            let msb = crc >> 15;
            crc = ((crc << 1) | u16::from(byte >> bit & 1)) ^ (msb * 0x1021);
        }
        self.0 = crc;
    }

    fn finish(mut self) -> u16 {
        self.push(0);
        self.push(0);
        self.0
    }
}

/// Hash one colour component (`width × height` visible samples of
/// `plane`) with `method` at `bit_depth`.
pub fn component_hash(
    plane: &PicturePlane,
    width: usize,
    height: usize,
    bit_depth: u32,
    method: PictureHashMethod,
) -> PictureHashValue {
    let width = width.min(plane.width);
    let height = height.min(plane.height);
    let wide = bit_depth > 8;
    let rows = (0..height).map(|y| &plane.samples[y * plane.stride..y * plane.stride + width]);
    match method {
        PictureHashMethod::Md5 => {
            let mut h = Md5::new();
            let mut bytes = Vec::with_capacity(width * if wide { 2 } else { 1 });
            for row in rows {
                bytes.clear();
                if wide {
                    for &v in row {
                        bytes.extend_from_slice(&v.to_le_bytes());
                    }
                } else {
                    bytes.extend(row.iter().map(|&v| v as u8));
                }
                h.update(&bytes);
            }
            PictureHashValue::Md5(h.finish())
        }
        PictureHashMethod::Crc => {
            let mut crc = Crc16::new();
            for row in rows {
                for &v in row {
                    crc.push(v as u8);
                    if wide {
                        crc.push((v >> 8) as u8);
                    }
                }
            }
            PictureHashValue::Crc(crc.finish())
        }
        PictureHashMethod::Checksum => {
            // H.274 eq. 36.
            let mut sum = 0u32;
            for (y, row) in rows.enumerate() {
                let y_mask = (y & 0xff) ^ (y >> 8);
                for (x, &v) in row.iter().enumerate() {
                    let mask = ((x & 0xff) ^ (x >> 8) ^ y_mask) as u32;
                    sum = sum.wrapping_add(u32::from(v & 0xff) ^ mask);
                    if wide {
                        sum = sum.wrapping_add(u32::from(v >> 8) ^ mask);
                    }
                }
            }
            PictureHashValue::Checksum(sum)
        }
    }
}

/// The decoded picture hash of `pic`: one component for
/// `chroma_format_idc = 0`, three otherwise. `width` / `height` are
/// the picture size in luma samples (the full decoded picture, before
/// the conformance crop).
pub fn picture_hash(
    pic: &PictureBuffer,
    width: usize,
    height: usize,
    chroma_format_idc: u8,
    bit_depth: u32,
    method: PictureHashMethod,
) -> DecodedPictureHash {
    let mut components = vec![component_hash(&pic.luma, width, height, bit_depth, method)];
    if chroma_format_idc != 0 {
        let (sw, sh) = crate::reconstruct::chroma_subsampling(u32::from(chroma_format_idc));
        for plane in [&pic.cb, &pic.cr] {
            components.push(component_hash(
                plane,
                width / sw,
                height / sh,
                bit_depth,
                method,
            ));
        }
    }
    DecodedPictureHash {
        hash_type: method.hash_type(),
        single_component_flag: chroma_format_idc == 0,
        components,
    }
}

/// Outcome of checking a picture against its hash SEI message.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum PictureHashStatus {
    /// Every signalled component matched.
    Match(PictureHashMethod),
    /// The component with this index (0 = luma) did not match.
    Mismatch {
        method: PictureHashMethod,
        component: u8,
    },
}

impl PictureHashStatus {
    /// `true` for [`Self::Match`].
    pub fn is_match(&self) -> bool {
        matches!(self, Self::Match(_))
    }
}

/// Check `pic` against a received message. `None` when the message
/// uses a reserved hash type or its component count does not fit the
/// chroma format (H.274: such a message is ignored).
pub fn verify(
    message: &DecodedPictureHash,
    pic: &PictureBuffer,
    width: usize,
    height: usize,
    chroma_format_idc: u8,
    bit_depth: u32,
) -> Option<PictureHashStatus> {
    let method = PictureHashMethod::from_hash_type(message.hash_type)?;
    let expected_components = if chroma_format_idc == 0 { 1 } else { 3 };
    if message.components.len() != expected_components {
        return None;
    }
    let ours = picture_hash(pic, width, height, chroma_format_idc, bit_depth, method);
    for (i, (a, b)) in ours.components.iter().zip(&message.components).enumerate() {
        if a != b {
            return Some(PictureHashStatus::Mismatch {
                method,
                component: i as u8,
            });
        }
    }
    Some(PictureHashStatus::Match(method))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn md5_known_vectors() {
        assert_eq!(hex(&md5(b"")), "d41d8cd98f00b204e9800998ecf8427e");
        assert_eq!(hex(&md5(b"abc")), "900150983cd24fb0d6963f7d28e17f72");
        assert_eq!(
            hex(&md5(b"The quick brown fox jumps over the lazy dog")),
            "9e107d9d372bb6826bd81d3542a419d6"
        );
        // Split updates across the block boundary agree with one shot.
        let data: Vec<u8> = (0..1000u32).map(|i| (i * 7 + 3) as u8).collect();
        for split in [0, 1, 55, 56, 63, 64, 65, 128, 999, 1000] {
            let mut h = Md5::new();
            h.update(&data[..split]);
            h.update(&data[split..]);
            assert_eq!(h.finish(), md5(&data), "split {split}");
        }
    }

    /// H.274 eq. 35 run literally, bit by bit over the data plus two
    /// zero bytes.
    fn crc_literal(data: &[u8]) -> u16 {
        let mut padded = data.to_vec();
        padded.extend_from_slice(&[0, 0]);
        let mut crc: u32 = 0xffff;
        for bit_idx in 0..padded.len() * 8 {
            let data_byte = u32::from(padded[bit_idx >> 3]);
            let crc_msb = (crc >> 15) & 1;
            let bit_val = (data_byte >> (7 - (bit_idx & 7))) & 1;
            crc = (((crc << 1) + bit_val) & 0xffff) ^ (crc_msb * 0x1021);
        }
        crc as u16
    }

    fn plane(w: usize, h: usize, bd: u32, seed: u32) -> PicturePlane {
        let mut p = PicturePlane::filled_bd(w, h, 0, bd);
        let mut s = seed;
        for v in &mut p.samples {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            *v = ((s >> 9) & ((1 << bd) - 1)) as u16;
        }
        p
    }

    #[test]
    fn crc_and_checksum_follow_the_text() {
        for (w, h, bd) in [
            (16usize, 8usize, 8u32),
            (300, 5, 8),
            (20, 270, 10),
            (7, 3, 12),
        ] {
            let p = plane(w, h, bd, (w * h) as u32 + bd);
            let mut data = Vec::new();
            let mut sum = 0u32;
            for y in 0..h {
                for x in 0..w {
                    let v = p.samples[y * p.stride + x];
                    data.push(v as u8);
                    let mask = ((x & 0xff) ^ (y & 0xff) ^ (x >> 8) ^ (y >> 8)) as u32;
                    sum = sum.wrapping_add(u32::from(v & 0xff) ^ mask);
                    if bd > 8 {
                        data.push((v >> 8) as u8);
                        sum = sum.wrapping_add(u32::from(v >> 8) ^ mask);
                    }
                }
            }
            assert_eq!(
                component_hash(&p, w, h, bd, PictureHashMethod::Crc),
                PictureHashValue::Crc(crc_literal(&data))
            );
            assert_eq!(
                component_hash(&p, w, h, bd, PictureHashMethod::Checksum),
                PictureHashValue::Checksum(sum)
            );
            assert_eq!(
                component_hash(&p, w, h, bd, PictureHashMethod::Md5),
                PictureHashValue::Md5(md5(&data))
            );
        }
    }

    #[test]
    fn verify_reports_the_first_wrong_component() {
        let pic = PictureBuffer {
            luma: plane(16, 16, 8, 1),
            cb: plane(8, 8, 8, 2),
            cr: plane(8, 8, 8, 3),
        };
        for method in [
            PictureHashMethod::Md5,
            PictureHashMethod::Crc,
            PictureHashMethod::Checksum,
        ] {
            let msg = picture_hash(&pic, 16, 16, 1, 8, method);
            assert_eq!(msg.components.len(), 3);
            assert_eq!(
                verify(&msg, &pic, 16, 16, 1, 8),
                Some(PictureHashStatus::Match(method))
            );
            let mut other = pic.clone();
            other.cr.samples[5] ^= 1;
            assert_eq!(
                verify(&msg, &other, 16, 16, 1, 8),
                Some(PictureHashStatus::Mismatch {
                    method,
                    component: 2
                })
            );
            // A three-component message does not describe a 4:0:0 picture.
            assert_eq!(verify(&msg, &pic, 16, 16, 0, 8), None);
        }
        let mut reserved = picture_hash(&pic, 16, 16, 1, 8, PictureHashMethod::Md5);
        reserved.hash_type = 3;
        assert_eq!(verify(&reserved, &pic, 16, 16, 1, 8), None);
    }
}
