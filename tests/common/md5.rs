//! Minimal RFC-1321-shape MD5 shared by the conformance harnesses
//! (the `.opl` sidecars are MD5-based; no new dependency). The 64
//! round constants are derived at run time as
//! `floor(|sin(i + 1)| * 2^32)`.
#![allow(dead_code)]

const S: [u32; 64] = [
    7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, //
    5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, //
    4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, //
    6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
];

fn k_table() -> [u32; 64] {
    let mut k = [0u32; 64];
    for (i, slot) in k.iter_mut().enumerate() {
        *slot = (((i as f64 + 1.0).sin().abs()) * 4294967296.0) as u32;
    }
    k
}

pub struct Md5 {
    state: [u32; 4],
    buf: Vec<u8>,
    len: u64,
    k: [u32; 64],
}

impl Md5 {
    pub fn new() -> Self {
        Self {
            state: [0x6745_2301, 0xefcd_ab89, 0x98ba_dcfe, 0x1032_5476],
            buf: Vec::new(),
            len: 0,
            k: k_table(),
        }
    }

    pub fn update(&mut self, data: &[u8]) {
        self.len = self.len.wrapping_add(data.len() as u64);
        self.buf.extend_from_slice(data);
        let blocks = self.buf.len() / 64;
        for b in 0..blocks {
            let block: [u8; 64] = self.buf[b * 64..b * 64 + 64].try_into().unwrap();
            self.process(&block);
        }
        self.buf.drain(..blocks * 64);
    }

    fn process(&mut self, block: &[u8; 64]) {
        let mut m = [0u32; 16];
        for (i, slot) in m.iter_mut().enumerate() {
            *slot = u32::from_le_bytes(block[i * 4..i * 4 + 4].try_into().unwrap());
        }
        let (mut a, mut b, mut c, mut d) =
            (self.state[0], self.state[1], self.state[2], self.state[3]);
        #[allow(clippy::needless_range_loop)] // i indexes S, K and drives g
        for i in 0..64 {
            let (f, g) = match i / 16 {
                0 => ((b & c) | (!b & d), i),
                1 => ((d & b) | (!d & c), (5 * i + 1) % 16),
                2 => (b ^ c ^ d, (3 * i + 5) % 16),
                _ => (c ^ (b | !d), (7 * i) % 16),
            };
            let tmp = d;
            d = c;
            c = b;
            let sum = a.wrapping_add(f).wrapping_add(self.k[i]).wrapping_add(m[g]);
            b = b.wrapping_add(sum.rotate_left(S[i]));
            a = tmp;
        }
        self.state[0] = self.state[0].wrapping_add(a);
        self.state[1] = self.state[1].wrapping_add(b);
        self.state[2] = self.state[2].wrapping_add(c);
        self.state[3] = self.state[3].wrapping_add(d);
    }

    pub fn finish_hex(mut self) -> String {
        let bit_len = self.len.wrapping_mul(8);
        self.update(&[0x80]);
        while self.buf.len() % 64 != 56 {
            self.buf.push(0);
        }
        let mut tail = self.buf.split_off(0);
        tail.extend_from_slice(&bit_len.to_le_bytes());
        for b in 0..tail.len() / 64 {
            let block: [u8; 64] = tail[b * 64..b * 64 + 64].try_into().unwrap();
            self.process(&block);
        }
        let mut out = String::with_capacity(32);
        for s in self.state {
            for byte in s.to_le_bytes() {
                out.push_str(&format!("{byte:02x}"));
            }
        }
        out
    }
}

pub fn md5_hex(data: &[u8]) -> String {
    let mut h = Md5::new();
    h.update(data);
    h.finish_hex()
}

#[test]
fn known_vectors() {
    assert_eq!(md5_hex(b""), "d41d8cd98f00b204e9800998ecf8427e");
    assert_eq!(md5_hex(b"abc"), "900150983cd24fb0d6963f7d28e17f72");
    assert_eq!(
        md5_hex(b"The quick brown fox jumps over the lazy dog"),
        "9e107d9d372bb6826bd81d3542a419d6"
    );
    // Multi-block + 56-byte-boundary shapes.
    let long = vec![0xa5u8; 200];
    let mut h = Md5::new();
    h.update(&long[..3]);
    h.update(&long[3..]);
    assert_eq!(h.finish_hex(), md5_hex(&long));
}
