//! VVC inverse transform kernels (§8.7.4).
//!
//! Kernels landed:
//!
//! * DST-VII (`trType = 1`): sizes 4, 8, 16, 32 (§8.7.4.5 eqs. 1185–1191).
//! * DCT-VIII (`trType = 2`): sizes 4, 8, 16, 32 (eqs. 1192–1198).
//! * DCT-II (`trType = 0`): sizes 4, 8, 16, 32 (eqs. 1179–1184). The
//!   spec defines a single 64×64 base matrix via two 16-column
//!   sub-tables (`transMatrixCol0to15`, `transMatrixCol16to31`) plus
//!   two symmetry relations. We transcribe the two sub-tables verbatim
//!   and derive the remaining 32 rows via the symmetry rules at build
//!   time. The per-size inverse transform samples the full 64×64
//!   matrix at column stride `2^(6 − log2(N))` (eq. 1177).
//!
//! Size 64 (DCT-II) is not needed for intra decode of the fixtures we
//! target and is deferred.
//!
//! The exposed primitive is [`one_d_transform`], which implements
//! eq. 1178 (DST-VII / DCT-VIII) and eq. 1177 (DCT-II) from §8.7.4.4:
//!
//! ```text
//!   DCT-II :  y[i] = sum_{j=0}^{nonZeroS-1}
//!                      transMatrix[i][j * (64 >> log2(nTbS))] * x[j]
//!   DST-VII /
//!   DCT-VIII: y[i] = sum_{j=0}^{nonZeroS-1} transMatrix[i][j] * x[j]
//! ```
//!
//! Matrix entries are exactly as printed in the spec (§8.7.4.5) with
//! sign-extended 8-bit values fitting in `i16`. Inputs are coefficient
//! arrays in `i32` (they may carry up to 15-bit magnitudes after
//! de-quantisation, plus room for the accumulation headroom).
//!
//! Composition into the full 2D inverse transform (`clause 8.7.4.1`)
//! — including the mid-shift of 7 and final shift to
//! `(20 - BitDepth)` — lives alongside the CU/TU walker rather than in
//! this primitive.

use oxideav_core::{Error, Result};

/// Transform kernel identifier per §8.7.4.5.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum TrType {
    DctII = 0,
    DstVII = 1,
    DctVIII = 2,
}

/// 4-point DST-VII matrix (eq. 1185).
#[rustfmt::skip]
pub const DST_VII_4: [[i16; 4]; 4] = [
    [ 29,  55,  74,  84],
    [ 74,  74,   0, -74],
    [ 84, -29, -74,  55],
    [ 55, -84,  74, -29],
];

/// 8-point DST-VII matrix (eq. 1186).
#[rustfmt::skip]
pub const DST_VII_8: [[i16; 8]; 8] = [
    [17,  32,  46,  60,  71,  78,  85,  86],
    [46,  78,  86,  71,  32, -17, -60, -85],
    [71,  85,  32, -46, -86, -60,  17,  78],
    [85,  46, -60, -78,  17,  86,  32, -71],
    [86, -17, -85,  32,  78, -46, -71,  60],
    [78, -71, -17,  85, -60, -32,  86, -46],
    [60, -86,  71, -17, -46,  85, -78,  32],
    [32, -60,  78, -86,  85, -71,  46, -17],
];

/// 4-point DCT-VIII matrix (eq. 1192).
#[rustfmt::skip]
pub const DCT_VIII_4: [[i16; 4]; 4] = [
    [84,  74,  55,  29],
    [74,   0, -74, -74],
    [55, -74, -29,  84],
    [29, -74,  84, -55],
];

/// 8-point DCT-VIII matrix (eq. 1193).
#[rustfmt::skip]
pub const DCT_VIII_8: [[i16; 8]; 8] = [
    [86,  85,  78,  71,  60,  46,  32,  17],
    [85,  60,  17, -32, -71, -86, -78, -46],
    [78,  17, -60, -86, -46,  32,  85,  71],
    [71, -32, -86, -17,  78,  60, -46, -85],
    [60, -71, -46,  78,  32, -85, -17,  86],
    [46, -86,  32,  60, -85,  17,  71, -78],
    [32, -78,  85, -46, -17,  71, -86,  60],
    [17, -46,  71, -85,  86, -78,  60, -32],
];

/// 16-point DST-VII matrix (eq. 1187).
#[rustfmt::skip]
pub const DST_VII_16: [[i16; 16]; 16] = [
    [   8,  17,  25,  33,  40,  48,  55,  62,  68,  73,  77,  81,  85,  87,  88,  88],
    [  25,  48,  68,  81,  88,  88,  81,  68,  48,  25,   0, -25, -48, -68, -81, -88],
    [  40,  73,  88,  85,  62,  25, -17, -55, -81, -88, -77, -48,  -8,  33,  68,  87],
    [  55,  87,  81,  40, -17, -68, -88, -73, -25,  33,  77,  88,  62,   8, -48, -85],
    [  68,  88,  48, -25, -81, -81, -25,  48,  88,  68,   0, -68, -88, -48,  25,  81],
    [  77,  77,   0, -77, -77,   0,  77,  77,   0, -77, -77,   0,  77,  77,   0, -77],
    [  85,  55, -48, -87,  -8,  81,  62, -40, -88, -17,  77,  68, -33, -88, -25,  73],
    [  88,  25, -81, -48,  68,  68, -48, -81,  25,  88,   0, -88, -25,  81,  48, -68],
    [  88,  -8, -88,  17,  87, -25, -85,  33,  81, -40, -77,  48,  73, -55, -68,  62],
    [  87, -40, -68,  73,  33, -88,   8,  85, -48, -62,  77,  25, -88,  17,  81, -55],
    [  81, -68, -25,  88, -48, -48,  88, -25, -68,  81,   0, -81,  68,  25, -88,  48],
    [  73, -85,  25,  55, -88,  48,  33, -87,  68,   8, -77,  81, -17, -62,  88, -40],
    [  62, -88,  68,  -8, -55,  88, -73,  17,  48, -87,  77, -25, -40,  85, -81,  33],
    [  48, -81,  88, -68,  25,  25, -68,  88, -81,  48,   0, -48,  81, -88,  68, -25],
    [  33, -62,  81, -88,  85, -68,  40,  -8, -25,  55, -77,  88, -87,  73, -48,  17],
    [  17, -33,  48, -62,  73, -81,  87, -88,  88, -85,  77, -68,  55, -40,  25,  -8],
];

/// 32-point DST-VII matrix columns 0..15 (eq. 1189).
#[rustfmt::skip]
pub const DST_VII_32_COL_0_15: [[i16; 16]; 16] = [
    [   4,   9,  13,  17,  21,  26,  30,  34,  38,  42,  46,  50,  53,  56,  60,  63],
    [  13,  26,  38,  50,  60,  68,  77,  82,  86,  89,  90,  88,  85,  80,  74,  66],
    [  21,  42,  60,  74,  84,  89,  89,  84,  74,  60,  42,  21,   0, -21, -42, -60],
    [  30,  56,  77,  87,  89,  80,  63,  38,   9, -21, -50, -72, -85, -90, -84, -68],
    [  38,  68,  86,  88,  74,  46,   9, -30, -63, -84, -90, -78, -53, -17,  21,  56],
    [  46,  78,  90,  77,  42,  -4, -50, -80, -90, -74, -38,   9,  53,  82,  89,  72],
    [  53,  85,  85,  53,   0, -53, -85, -85, -53,   0,  53,  85,  85,  53,   0, -53],
    [  60,  89,  74,  21, -42, -84, -84, -42,  21,  74,  89,  60,   0, -60, -89, -74],
    [  66,  90,  56, -13, -74, -87, -46,  26,  80,  84,  34, -38, -85, -78, -21,  50],
    [  72,  86,  34, -46, -89, -63,  13,  78,  82,  21, -56, -90, -53,  26,  84,  77],
    [  77,  80,   9, -72, -84, -17,  66,  86,  26, -60, -88, -34,  53,  90,  42, -46],
    [  80,  72, -17, -86, -60,  34,  90,  46, -50, -89, -30,  63,  85,  13, -74, -78],
    [  84,  60, -42, -89, -21,  74,  74, -21, -89, -42,  60,  84,   0, -84, -60,  42],
    [  86,  46, -63, -78,  21,  90,  26, -77, -66,  42,  87,   4, -85, -50,  60,  80],
    [  88,  30, -78, -56,  60,  77, -34, -87,   4,  89,  26, -80, -53,  63,  74, -38],
    [  90,  13, -87, -26,  84,  38, -78, -50,  72,  60, -63, -68,  53,  77, -42, -82],
];

/// 32-point DST-VII matrix columns 16..31 (eq. 1191).
#[rustfmt::skip]
pub const DST_VII_32_COL_16_31: [[i16; 16]; 16] = [
    [  66,  68,  72,  74,  77,  78,  80,  82,  84,  85,  86,  87,  88,  89,  90,  90],
    [  56,  46,  34,  21,   9,  -4, -17, -30, -42, -53, -63, -72, -78, -84, -87, -90],
    [ -74, -84, -89, -89, -84, -74, -60, -42, -21,   0,  21,  42,  60,  74,  84,  89],
    [ -46, -17,  13,  42,  66,  82,  90,  86,  74,  53,  26,  -4, -34, -60, -78, -88],
    [  80,  90,  82,  60,  26, -13, -50, -77, -89, -85, -66, -34,   4,  42,  72,  87],
    [  34, -13, -56, -84, -88, -68, -30,  17,  60,  85,  87,  66,  26, -21, -63, -86],
    [ -85, -85, -53,   0,  53,  85,  85,  53,   0, -53, -85, -85, -53,   0,  53,  85],
    [ -21,  42,  84,  84,  42, -21, -74, -89, -60,   0,  60,  89,  74,  21, -42, -84],
    [  88,  72,   9, -60, -90, -63,   4,  68,  89,  53, -17, -77, -86, -42,  30,  82],
    [   9, -66, -88, -42,  38,  87,  68,  -4, -74, -85, -30,  50,  90,  60, -17, -80],
    [ -90, -50,  38,  89,  56, -30, -87, -63,  21,  85,  68, -13, -82, -74,   4,  78],
    [   4,  82,  68, -21, -87, -56,  38,  90,  42, -53, -88, -26,  66,  84,   9, -77],
    [  89,  21, -74, -74,  21,  89,  42, -60, -84,   0,  84,  60, -42, -89, -21,  74],
    [ -17, -90, -30,  74,  68, -38, -88,  -9,  84,  53, -56, -82,  13,  89,  34, -72],
    [ -86,   9,  90,  21, -82, -50,  66,  72, -42, -85,  13,  90,  17, -84, -46,  68],
    [  30,  86, -17, -89,   4,  90,   9, -88, -21,  85,  34, -80, -46,  74,  56, -66],
];

/// 16-point DCT-VIII matrix (eq. 1194).
#[rustfmt::skip]
pub const DCT_VIII_16: [[i16; 16]; 16] = [
    [  88,  88,  87,  85,  81,  77,  73,  68,  62,  55,  48,  40,  33,  25,  17,   8],
    [  88,  81,  68,  48,  25,   0, -25, -48, -68, -81, -88, -88, -81, -68, -48, -25],
    [  87,  68,  33,  -8, -48, -77, -88, -81, -55, -17,  25,  62,  85,  88,  73,  40],
    [  85,  48,  -8, -62, -88, -77, -33,  25,  73,  88,  68,  17, -40, -81, -87, -55],
    [  81,  25, -48, -88, -68,   0,  68,  88,  48, -25, -81, -81, -25,  48,  88,  68],
    [  77,   0, -77, -77,   0,  77,  77,   0, -77, -77,   0,  77,  77,   0, -77, -77],
    [  73, -25, -88, -33,  68,  77, -17, -88, -40,  62,  81,  -8, -87, -48,  55,  85],
    [  68, -48, -81,  25,  88,   0, -88, -25,  81,  48, -68, -68,  48,  81, -25, -88],
    [  62, -68, -55,  73,  48, -77, -40,  81,  33, -85, -25,  87,  17, -88,  -8,  88],
    [  55, -81, -17,  88, -25, -77,  62,  48, -85,  -8,  88, -33, -73,  68,  40, -87],
    [  48, -88,  25,  68, -81,   0,  81, -68, -25,  88, -48, -48,  88, -25, -68,  81],
    [  40, -88,  62,  17, -81,  77,  -8, -68,  87, -33, -48,  88, -55, -25,  85, -73],
    [  33, -81,  85, -40, -25,  77, -87,  48,  17, -73,  88, -55,  -8,  68, -88,  62],
    [  25, -68,  88, -81,  48,   0, -48,  81, -88,  68, -25, -25,  68, -88,  81, -48],
    [  17, -48,  73, -87,  88, -77,  55, -25,  -8,  40, -68,  85, -88,  81, -62,  33],
    [   8, -25,  40, -55,  68, -77,  85, -88,  88, -87,  81, -73,  62, -48,  33, -17],
];

/// 32-point DCT-VIII matrix columns 0..15 (eq. 1196).
#[rustfmt::skip]
pub const DCT_VIII_32_COL_0_15: [[i16; 16]; 16] = [
    [  90,  90,  89,  88,  87,  86,  85,  84,  82,  80,  78,  77,  74,  72,  68,  66],
    [  90,  87,  84,  78,  72,  63,  53,  42,  30,  17,   4,  -9, -21, -34, -46, -56],
    [  89,  84,  74,  60,  42,  21,   0, -21, -42, -60, -74, -84, -89, -89, -84, -74],
    [  88,  78,  60,  34,   4, -26, -53, -74, -86, -90, -82, -66, -42, -13,  17,  46],
    [  87,  72,  42,   4, -34, -66, -85, -89, -77, -50, -13,  26,  60,  82,  90,  80],
    [  86,  63,  21, -26, -66, -87, -85, -60, -17,  30,  68,  88,  84,  56,  13, -34],
    [  85,  53,   0, -53, -85, -85, -53,   0,  53,  85,  85,  53,   0, -53, -85, -85],
    [  84,  42, -21, -74, -89, -60,   0,  60,  89,  74,  21, -42, -84, -84, -42,  21],
    [  82,  30, -42, -86, -77, -17,  53,  89,  68,   4, -63, -90, -60,   9,  72,  88],
    [  80,  17, -60, -90, -50,  30,  85,  74,   4, -68, -87, -38,  42,  88,  66,  -9],
    [  78,   4, -74, -82, -13,  68,  85,  21, -63, -87, -30,  56,  89,  38, -50, -90],
    [  77,  -9, -84, -66,  26,  88,  53, -42, -90, -38,  56,  87,  21, -68, -82,  -4],
    [  74, -21, -89, -42,  60,  84,   0, -84, -60,  42,  89,  21, -74, -74,  21,  89],
    [  72, -34, -89, -13,  82,  56, -53, -84,   9,  88,  38, -68, -74,  30,  90,  17],
    [  68, -46, -84,  17,  90,  13, -85, -42,  72,  66, -50, -82,  21,  90,   9, -86],
    [  66, -56, -74,  46,  80, -34, -85,  21,  88,  -9, -90,  -4,  89,  17, -86, -30],
];

/// 32-point DCT-VIII matrix columns 16..31 (eq. 1198).
#[rustfmt::skip]
pub const DCT_VIII_32_COL_16_31: [[i16; 16]; 16] = [
    [  63,  60,  56,  53,  50,  46,  42,  38,  34,  30,  26,  21,  17,  13,   9,   4],
    [ -66, -74, -80, -85, -88, -90, -89, -86, -82, -77, -68, -60, -50, -38, -26, -13],
    [ -60, -42, -21,   0,  21,  42,  60,  74,  84,  89,  89,  84,  74,  60,  42,  21],
    [  68,  84,  90,  85,  72,  50,  21,  -9, -38, -63, -80, -89, -87, -77, -56, -30],
    [  56,  21, -17, -53, -78, -90, -84, -63, -30,   9,  46,  74,  88,  86,  68,  38],
    [ -72, -89, -82, -53,  -9,  38,  74,  90,  80,  50,   4, -42, -77, -90, -78, -46],
    [ -53,   0,  53,  85,  85,  53,   0, -53, -85, -85, -53,   0,  53,  85,  85,  53],
    [  74,  89,  60,   0, -60, -89, -74, -21,  42,  84,  84,  42, -21, -74, -89, -60],
    [  50, -21, -78, -85, -38,  34,  84,  80,  26, -46, -87, -74, -13,  56,  90,  66],
    [ -77, -84, -26,  53,  90,  56, -21, -82, -78, -13,  63,  89,  46, -34, -86, -72],
    [ -46,  42,  90,  53, -34, -88, -60,  26,  86,  66, -17, -84, -72,   9,  80,  77],
    [  78,  74, -13, -85, -63,  30,  89,  50, -46, -90, -34,  60,  86,  17, -72, -80],
    [  42, -60, -84,   0,  84,  60, -42, -89, -21,  74,  74, -21, -89, -42,  60,  84],
    [ -80, -60,  50,  85,  -4, -87, -42,  66,  77, -26, -90, -21,  78,  63, -46, -86],
    [ -38,  74,  63, -53, -80,  26,  89,   4, -87, -34,  77,  60, -56, -78,  30,  88],
    [  82,  42, -77, -53,  68,  63, -60, -72,  50,  78, -38, -84,  26,  87, -13, -90],
];

/// DCT-II base 64x16 column block rows 0..63, columns 0..15 (eq. 1180).
#[rustfmt::skip]
pub const DCT_II_COL_0_15: [[i16; 16]; 64] = [
    [  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64],
    [  91,  90,  90,  90,  88,  87,  86,  84,  83,  81,  79,  77,  73,  71,  69,  65],
    [  90,  90,  88,  85,  82,  78,  73,  67,  61,  54,  46,  38,  31,  22,  13,   4],
    [  90,  88,  84,  79,  71,  62,  52,  41,  28,  15,   2, -11, -24, -37, -48, -59],
    [  90,  87,  80,  70,  57,  43,  25,   9,  -9, -25, -43, -57, -70, -80, -87, -90],
    [  90,  84,  73,  59,  41,  20,  -2, -24, -44, -62, -77, -86, -90, -90, -83, -71],
    [  90,  82,  67,  46,  22,  -4, -31, -54, -73, -85, -90, -88, -78, -61, -38, -13],
    [  90,  79,  59,  33,   2, -28, -56, -77, -88, -90, -81, -62, -37,  -7,  24,  52],
    [  89,  75,  50,  18, -18, -50, -75, -89, -89, -75, -50, -18,  18,  50,  75,  89],
    [  88,  71,  41,   2, -37, -69, -87, -90, -73, -44,  -7,  33,  65,  86,  90,  77],
    [  88,  67,  31, -13, -54, -82, -90, -78, -46,  -4,  38,  73,  90,  85,  61,  22],
    [  87,  62,  20, -28, -69, -90, -84, -56, -11,  37,  73,  90,  81,  48,   2, -44],
    [  87,  57,   9, -43, -80, -90, -70, -25,  25,  70,  90,  80,  43,  -9, -57, -87],
    [  86,  52,  -2, -56, -87, -84, -48,   7,  59,  88,  83,  44, -11, -62, -90, -81],
    [  85,  46, -13, -67, -90, -73, -22,  38,  82,  88,  54,  -4, -61, -90, -78, -31],
    [  84,  41, -24, -77, -90, -56,   7,  65,  91,  69,  11, -52, -88, -79, -28,  37],
    [  83,  36, -36, -83, -83, -36,  36,  83,  83,  36, -36, -83, -83, -36,  36,  83],
    [  83,  28, -44, -88, -73, -11,  59,  91,  62,  -7, -71, -90, -48,  24,  81,  84],
    [  82,  22, -54, -90, -61,  13,  78,  85,  31, -46, -90, -67,   4,  73,  88,  38],
    [  81,  15, -62, -90, -44,  37,  88,  69,  -7, -77, -84, -24,  56,  91,  52, -28],
    [  80,   9, -70, -87, -25,  57,  90,  43, -43, -90, -57,  25,  87,  70,  -9, -80],
    [  79,   2, -77, -81,  -7,  73,  83,  11, -71, -84, -15,  69,  86,  20, -65, -87],
    [  78,  -4, -82, -73,  13,  85,  67, -22, -88, -61,  31,  90,  54, -38, -90, -46],
    [  77, -11, -86, -62,  33,  90,  44, -52, -90, -24,  69,  83,   2, -81, -71,  20],
    [  75, -18, -89, -50,  50,  89,  18, -75, -75,  18,  89,  50, -50, -89, -18,  75],
    [  73, -24, -90, -37,  65,  81, -11, -88, -48,  56,  86,   2, -84, -59,  44,  90],
    [  73, -31, -90, -22,  78,  67, -38, -90, -13,  82,  61, -46, -88,  -4,  85,  54],
    [  71, -37, -90,  -7,  86,  48, -62, -79,  24,  91,  20, -81, -59,  52,  84, -11],
    [  70, -43, -87,   9,  90,  25, -80, -57,  57,  80, -25, -90,  -9,  87,  43, -70],
    [  69, -48, -83,  24,  90,   2, -90, -28,  81,  52, -65, -71,  44,  84, -20, -90],
    [  67, -54, -78,  38,  85, -22, -90,   4,  90,  13, -88, -31,  82,  46, -73, -61],
    [  65, -59, -71,  52,  77, -44, -81,  37,  84, -28, -87,  20,  90, -11, -90,   2],
    [  64, -64, -64,  64,  64, -64, -64,  64,  64, -64, -64,  64,  64, -64, -64,  64],
    [  62, -69, -56,  73,  48, -79, -41,  83,  33, -86, -24,  88,  15, -90,  -7,  91],
    [  61, -73, -46,  82,  31, -88, -13,  90,  -4, -90,  22,  85, -38, -78,  54,  67],
    [  59, -77, -37,  87,  11, -91,  15,  86, -41, -73,  62,  56, -79, -33,  88,   7],
    [  57, -80, -25,  90,  -9, -87,  43,  70, -70, -43,  87,   9, -90,  25,  80, -57],
    [  56, -83, -15,  90, -28, -77,  65,  44, -87,  -2,  88, -41, -69,  73,  33, -90],
    [  54, -85,  -4,  88, -46, -61,  82,  13, -90,  38,  67, -78, -22,  90, -31, -73],
    [  52, -87,   7,  83, -62, -41,  90, -20, -77,  71,  28, -91,  33,  69, -79, -15],
    [  50, -89,  18,  75, -75, -18,  89, -50, -50,  89, -18, -75,  75,  18, -89,  50],
    [  48, -90,  28,  65, -84,   7,  79, -73, -15,  87, -59, -37,  91, -41, -56,  88],
    [  46, -90,  38,  54, -90,  31,  61, -88,  22,  67, -85,  13,  73, -82,   4,  78],
    [  44, -91,  48,  41, -90,  52,  37, -90,  56,  33, -90,  59,  28, -88,  62,  24],
    [  43, -90,  57,  25, -87,  70,   9, -80,  80,  -9, -70,  87, -25, -57,  90, -43],
    [  41, -90,  65,  11, -79,  83, -20, -59,  90, -48, -33,  87, -71,  -2,  73, -86],
    [  38, -88,  73,  -4, -67,  90, -46, -31,  85, -78,  13,  61, -90,  54,  22, -82],
    [  37, -86,  79, -20, -52,  90, -69,   2,  65, -90,  56,  15, -77,  87, -41, -33],
    [  36, -83,  83, -36, -36,  83, -83,  36,  36, -83,  83, -36, -36,  83, -83,  36],
    [  33, -81,  87, -48, -15,  71, -90,  62,  -2, -59,  90, -73,  20,  44, -86,  83],
    [  31, -78,  90, -61,   4,  54, -88,  82, -38, -22,  73, -90,  67, -13, -46,  85],
    [  28, -73,  91, -71,  24,  33, -77,  90, -69,  20,  37, -79,  90, -65,  15,  41],
    [  25, -70,  90, -80,  43,   9, -57,  87, -87,  57,  -9, -43,  80, -90,  70, -25],
    [  24, -65,  88, -86,  59, -15, -33,  71, -90,  83, -52,   7,  41, -77,  91, -79],
    [  22, -61,  85, -90,  73, -38,  -4,  46, -78,  90, -82,  54, -13, -31,  67, -88],
    [  20, -56,  81, -91,  83, -59,  24,  15, -52,  79, -90,  84, -62,  28,  11, -48],
    [  18, -50,  75, -89,  89, -75,  50, -18, -18,  50, -75,  89, -89,  75, -50,  18],
    [  15, -44,  69, -84,  91, -86,  71, -48,  20,  11, -41,  65, -83,  90, -87,  73],
    [  13, -38,  61, -78,  88, -90,  85, -73,  54, -31,   4,  22, -46,  67, -82,  90],
    [  11, -33,  52, -69,  81, -88,  91, -87,  79, -65,  48, -28,   7,  15, -37,  56],
    [   9, -25,  43, -57,  70, -80,  87, -90,  90, -87,  80, -70,  57, -43,  25,  -9],
    [   7, -20,  33, -44,  56, -65,  73, -81,  86, -90,  91, -90,  87, -83,  77, -69],
    [   4, -13,  22, -31,  38, -46,  54, -61,  67, -73,  78, -82,  85, -88,  90, -90],
    [   2,  -7,  11, -15,  20, -24,  28, -33,  37, -41,  44, -48,  52, -56,  59, -62],
];

/// DCT-II base 64x16 column block rows 0..63, columns 16..31 (eq. 1182).
#[rustfmt::skip]
pub const DCT_II_COL_16_31: [[i16; 16]; 64] = [
    [  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64],
    [  62,  59,  56,  52,  48,  44,  41,  37,  33,  28,  24,  20,  15,  11,   7,   2],
    [  -4, -13, -22, -31, -38, -46, -54, -61, -67, -73, -78, -82, -85, -88, -90, -90],
    [ -69, -77, -83, -87, -90, -91, -90, -86, -81, -73, -65, -56, -44, -33, -20,  -7],
    [ -90, -87, -80, -70, -57, -43, -25,  -9,   9,  25,  43,  57,  70,  80,  87,  90],
    [ -56, -37, -15,   7,  28,  48,  65,  79,  87,  91,  88,  81,  69,  52,  33,  11],
    [  13,  38,  61,  78,  88,  90,  85,  73,  54,  31,   4, -22, -46, -67, -82, -90],
    [  73,  87,  90,  83,  65,  41,  11, -20, -48, -71, -86, -91, -84, -69, -44, -15],
    [  89,  75,  50,  18, -18, -50, -75, -89, -89, -75, -50, -18,  18,  50,  75,  89],
    [  48,  11, -28, -62, -84, -90, -79, -52, -15,  24,  59,  83,  91,  81,  56,  20],
    [ -22, -61, -85, -90, -73, -38,   4,  46,  78,  90,  82,  54,  13, -31, -67, -88],
    [ -79, -91, -77, -41,   7,  52,  83,  90,  71,  33, -15, -59, -86, -88, -65, -24],
    [ -87, -57,  -9,  43,  80,  90,  70,  25, -25, -70, -90, -80, -43,   9,  57,  87],
    [ -41,  15,  65,  90,  79,  37, -20, -69, -90, -77, -33,  24,  71,  91,  73,  28],
    [  31,  78,  90,  61,   4, -54, -88, -82, -38,  22,  73,  90,  67,  13, -46, -85],
    [  83,  86,  44, -20, -73, -90, -59,   2,  62,  90,  71,  15, -48, -87, -81, -33],
    [  83,  36, -36, -83, -83, -36,  36,  83,  83,  36, -36, -83, -83, -36,  36,  83],
    [  33, -41, -87, -77, -15,  56,  90,  65,  -2, -69, -90, -52,  20,  79,  86,  37],
    [ -38, -88, -73,  -4,  67,  90,  46, -31, -85, -78, -13,  61,  90,  54, -22, -82],
    [ -86, -73,  -2,  71,  87,  33, -48, -90, -59,  20,  83,  79,  11, -65, -90, -41],
    [ -80,  -9,  70,  87,  25, -57, -90, -43,  43,  90,  57, -25, -87, -70,   9,  80],
    [ -24,  62,  88,  28, -59, -90, -33,  56,  90,  37, -52, -90, -41,  48,  91,  44],
    [  46,  90,  38, -54, -90, -31,  61,  88,  22, -67, -85, -13,  73,  82,   4, -78],
    [  88,  56, -41, -91, -37,  59,  87,  15, -73, -79,   7,  84,  65, -28, -90, -48],
    [  75, -18, -89, -50,  50,  89,  18, -75, -75,  18,  89,  50, -50, -89, -18,  75],
    [  15, -79, -69,  33,  91,  28, -71, -77,  20,  90,  41, -62, -83,   7,  87,  52],
    [ -54, -85,   4,  88,  46, -61, -82,  13,  90,  38, -67, -78,  22,  90,  31, -73],
    [ -90, -33,  73,  69, -41, -88,  -2,  87,  44, -65, -77,  28,  90,  15, -83, -56],
    [ -70,  43,  87,  -9, -90, -25,  80,  57, -57, -80,  25,  90,   9, -87, -43,  70],
    [  -7,  88,  33, -79, -56,  62,  73, -41, -86,  15,  91,  11, -87, -37,  77,  59],
    [  61,  73, -46, -82,  31,  88, -13, -90,  -4,  90,  22, -85, -38,  78,  54, -67],
    [  91,   7, -90, -15,  88,  24, -86, -33,  83,  41, -79, -48,  73,  56, -69, -62],
    [  64, -64, -64,  64,  64, -64, -64,  64,  64, -64, -64,  64,  64, -64, -64,  64],
    [  -2, -90,  11,  90, -20, -87,  28,  84, -37, -81,  44,  77, -52, -71,  59,  65],
    [ -67, -54,  78,  38, -85, -22,  90,   4, -90,  13,  88, -31, -82,  46,  73, -61],
    [ -90,  20,  84, -44, -71,  65,  52, -81, -28,  90,   2, -90,  24,  83, -48, -69],
    [ -57,  80,  25, -90,   9,  87, -43, -70,  70,  43, -87,  -9,  90, -25, -80,  57],
    [  11,  84, -52, -59,  81,  20, -91,  24,  79, -62, -48,  86,   7, -90,  37,  71],
    [  73,  31, -90,  22,  78, -67, -38,  90, -13, -82,  61,  46, -88,   4,  85, -54],
    [  90, -44, -59,  84,   2, -86,  56,  48, -88,  11,  81, -65, -37,  90, -24, -73],
    [  50, -89,  18,  75, -75, -18,  89, -50, -50,  89, -18, -75,  75,  18, -89,  50],
    [ -20, -71,  81,   2, -83,  69,  24, -90,  52,  44, -90,  33,  62, -86,  11,  77],
    [ -78,  -4,  82, -73, -13,  85, -67, -22,  88, -61, -31,  90, -54, -38,  90, -46],
    [ -87,  65,  20, -86,  69,  15, -84,  71,  11, -83,  73,   7, -81,  77,   2, -79],
    [ -43,  90, -57, -25,  87, -70,  -9,  80, -80,   9,  70, -87,  25,  57, -90,  43],
    [  28,  52, -91,  56,  24, -84,  77,  -7, -69,  88, -37, -44,  90, -62, -15,  81],
    [  82, -22, -54,  90, -61, -13,  78, -85,  31,  46, -90,  67,   4, -73,  88, -38],
    [  84, -81,  24,  48, -90,  71,  -7, -62,  91, -59, -11,  73, -88,  44,  28, -83],
    [  36, -83,  83, -36, -36,  83, -83,  36,  36, -83,  83, -36, -36,  83, -83,  36],
    [ -37, -28,  79, -88,  52,  11, -69,  91, -65,   7,  56, -90,  77, -24, -41,  84],
    [ -85,  46,  13, -67,  90, -73,  22,  38, -82,  88, -54,  -4,  61, -90,  78, -31],
    [ -81,  90, -62,  11,  44, -83,  88, -59,   7,  48, -84,  87, -56,   2,  52, -86],
    [ -25,  70, -90,  80, -43,  -9,  57, -87,  87, -57,   9,  43, -80,  90, -70,  25],
    [  44,   2, -48,  81, -90,  73, -37, -11,  56, -84,  90, -69,  28,  20, -62,  87],
    [  88, -67,  31,  13, -54,  82, -90,  78, -46,   4,  38, -73,  90, -85,  61, -22],
    [  77, -90,  86, -65,  33,   7, -44,  73, -90,  87, -69,  37,   2, -41,  71, -88],
    [  18, -50,  75, -89,  89, -75,  50, -18, -18,  50, -75,  89, -89,  75, -50,  18],
    [ -52,  24,   7, -37,  62, -81,  90, -88,  77, -56,  28,   2, -33,  59, -79,  90],
    [ -90,  82, -67,  46, -22,  -4,  31, -54,  73, -85,  90, -88,  78, -61,  38, -13],
    [ -71,  83, -90,  90, -86,  77, -62,  44, -24,   2,  20, -41,  59, -73,  84, -90],
    [  -9,  25, -43,  57, -70,  80, -87,  90, -90,  87, -80,  70, -57,  43, -25,   9],
    [  59, -48,  37, -24,  11,   2, -15,  28, -41,  52, -62,  71, -79,  84, -88,  90],
    [  90, -90,  88, -85,  82, -78,  73, -67,  61, -54,  46, -38,  31, -22,  13,  -4],
    [  65, -69,  71, -73,  77, -79,  81, -83,  84, -86,  87, -88,  90, -90,  90, -91],
];

/// Full 64×64 DCT-II transMatrix entry at row `m`, column `n`
/// (§8.7.4.5 eqs. 1179 / 1181 / 1183 / 1184). `m,n ∈ 0..63`.
#[inline]
fn dct_ii_entry(m: usize, n: usize) -> i32 {
    debug_assert!(m < 64 && n < 64);
    match m {
        0..=15 => DCT_II_COL_0_15[n][m] as i32,
        16..=31 => DCT_II_COL_16_31[n][m - 16] as i32,
        32..=47 => {
            // Eq. 1183: transMatrix[m][n] = sign * transMatrixCol16to31[47 - m][n]
            // for m = 32..47. In our storage DCT_II_COL_16_31[n][p] where
            // p = spec's first index (0..15), p = 47 - m.
            let sign = if n & 1 == 1 { -1 } else { 1 };
            sign * DCT_II_COL_16_31[n][47 - m] as i32
        }
        48..=63 => {
            // Eq. 1184: sign ^ col0to15[63 - m][n].
            let sign = if n & 1 == 1 { -1 } else { 1 };
            sign * DCT_II_COL_0_15[n][63 - m] as i32
        }
        _ => unreachable!(),
    }
}

/// Apply the inverse DCT-II transform (eq. 1177) at size `n_tb_s` ∈
/// {4,8,16,32}. `non_zero_s` allows skipping tail-zero columns.
fn apply_dct_ii(n_tb_s: usize, non_zero_s: usize, x: &[i32]) -> Vec<i32> {
    // Column stride per eq. 1177: `j * (64 >> log2(N))`.
    let stride = 64usize >> log2_exact(n_tb_s);
    let mut y = vec![0i32; n_tb_s];
    for (i, y_i) in y.iter_mut().enumerate() {
        let mut acc: i64 = 0;
        for j in 0..non_zero_s {
            let col = j * stride;
            acc += dct_ii_entry(i, col) as i64 * x[j] as i64;
        }
        *y_i = acc as i32;
    }
    y
}

/// `log2(n)` for `n` a power of two in `{4, 8, 16, 32, 64}`.
#[inline]
fn log2_exact(n: usize) -> u32 {
    debug_assert!(n.is_power_of_two());
    n.trailing_zeros()
}

/// Apply one of the inverse transforms.
///
/// * `tr_type` — kernel selector.
/// * `n_tb_s` — transform size (4, 8, 16 or 32; 64 only for DCT-II once
///   landed, currently deferred).
/// * `non_zero_s` — horizontal size of non-zero coefficients; inputs
///   beyond this range are ignored by the sum.
/// * `x` — input coefficients (len ≥ `non_zero_s`).
///
/// Returns an owned `Vec<i32>` of length `n_tb_s`, storing `y[i]` per
/// eq. 1177 (DCT-II) or eq. 1178 (DST-VII / DCT-VIII).
pub fn one_d_transform(
    tr_type: TrType,
    n_tb_s: usize,
    non_zero_s: usize,
    x: &[i32],
) -> Result<Vec<i32>> {
    if non_zero_s > n_tb_s {
        return Err(Error::invalid(format!(
            "h266 transform: nonZeroS={non_zero_s} > nTbS={n_tb_s}"
        )));
    }
    if x.len() < non_zero_s {
        return Err(Error::invalid(format!(
            "h266 transform: input slice {} shorter than nonZeroS {non_zero_s}",
            x.len()
        )));
    }
    match (tr_type, n_tb_s) {
        // DST-VII
        (TrType::DstVII, 4) => Ok(apply_matrix(&DST_VII_4, non_zero_s, x)),
        (TrType::DstVII, 8) => Ok(apply_matrix(&DST_VII_8, non_zero_s, x)),
        (TrType::DstVII, 16) => Ok(apply_matrix(&DST_VII_16, non_zero_s, x)),
        (TrType::DstVII, 32) => Ok(apply_dst_vii_32(non_zero_s, x)),
        // DCT-VIII
        (TrType::DctVIII, 4) => Ok(apply_matrix(&DCT_VIII_4, non_zero_s, x)),
        (TrType::DctVIII, 8) => Ok(apply_matrix(&DCT_VIII_8, non_zero_s, x)),
        (TrType::DctVIII, 16) => Ok(apply_matrix(&DCT_VIII_16, non_zero_s, x)),
        (TrType::DctVIII, 32) => Ok(apply_dct_viii_32(non_zero_s, x)),
        // DCT-II
        (TrType::DctII, 4) | (TrType::DctII, 8) | (TrType::DctII, 16) | (TrType::DctII, 32) => {
            Ok(apply_dct_ii(n_tb_s, non_zero_s, x))
        }
        (TrType::DctII, 64) => Err(Error::unsupported(
            "h266 transform: DCT-II size 64 not yet implemented",
        )),
        _ => Err(Error::unsupported(format!(
            "h266 transform: unsupported (trType={:?}, nTbS={n_tb_s})",
            tr_type
        ))),
    }
}

fn apply_matrix<const N: usize>(matrix: &[[i16; N]; N], non_zero_s: usize, x: &[i32]) -> Vec<i32> {
    let mut y = vec![0i32; N];
    for (i, row) in matrix.iter().enumerate() {
        let mut acc: i64 = 0;
        for j in 0..non_zero_s {
            acc += row[j] as i64 * x[j] as i64;
        }
        y[i] = acc as i32;
    }
    y
}

/// Table 39 — (trTypeHor, trTypeVer) from `mts_idx` (§8.7.4.1).
pub fn mts_idx_to_tr_types(mts_idx: u8) -> Option<(TrType, TrType)> {
    let map = |t: u8| match t {
        0 => Some(TrType::DctII),
        1 => Some(TrType::DstVII),
        2 => Some(TrType::DctVIII),
        _ => None,
    };
    match mts_idx {
        0 => Some((TrType::DctII, TrType::DctII)),
        1 => Some((map(1)?, map(1)?)),
        2 => Some((map(2)?, map(1)?)),
        3 => Some((map(1)?, map(2)?)),
        4 => Some((map(2)?, map(2)?)),
        _ => None,
    }
}

/// Implicit-MTS kernel selection (eqs. 1167 / 1168 of §8.7.4.1).
/// Returns `(trTypeHor, trTypeVer)` based purely on TB width / height.
/// Used when `implicitMtsEnabled == 1 && cu_sbt_flag == 0`.
pub fn implicit_mts_tr_types(n_tb_w: u32, n_tb_h: u32) -> (TrType, TrType) {
    let tr_h = if (4..=16).contains(&n_tb_w) {
        TrType::DstVII
    } else {
        TrType::DctII
    };
    let tr_v = if (4..=16).contains(&n_tb_h) {
        TrType::DstVII
    } else {
        TrType::DctII
    };
    (tr_h, tr_v)
}

/// Spec-exact clamp helper — CoeffMin / CoeffMax per §7.4.11.9
/// (Log2TransformRange = 15 with default config, giving
/// `[-(1<<15), (1<<15)-1]`).
#[inline]
fn clip_coeff(v: i32) -> i32 {
    v.clamp(COEFF_MIN, COEFF_MAX)
}

/// `CoeffMin` per §7.4.11.9 at `Log2TransformRange = 15`.
pub const COEFF_MIN: i32 = -(1 << 15);
/// `CoeffMax` per §7.4.11.9 at `Log2TransformRange = 15`.
pub const COEFF_MAX: i32 = (1 << 15) - 1;

/// Apply the separable 2D inverse transform (§8.7.4.1) with the spec's
/// mid-shift of 7 (eq. 1173) and final bdShift (eqs. 1174 / 1175).
///
/// Inputs:
/// * `n_tb_w`, `n_tb_h` — transform-block dimensions (each ∈ {4, 8, 16,
///   32} for this increment).
/// * `non_zero_w`, `non_zero_h` — non-zero coefficient ranges (eqs.
///   1171 / 1172; caller computes from `trType`).
/// * `tr_type_hor`, `tr_type_ver` — kernel selectors.
/// * `d` — row-major TB of scaled transform coefficients, length
///   `n_tb_w * n_tb_h`.
/// * `bit_depth` — BitDepth of the current component (the spec's
///   `BitDepth` from §7.4.3.4).
/// * `log2_transform_range` — spec's `Log2TransformRange` (default 15).
///
/// Output: row-major residual array of the same dimensions.
///
/// Handles the degenerate `nTbH == 1` / `nTbW == 1` cases per §8.7.4.1
/// steps 3 / 5.
pub fn inverse_transform_2d(
    n_tb_w: usize,
    n_tb_h: usize,
    non_zero_w: usize,
    non_zero_h: usize,
    tr_type_hor: TrType,
    tr_type_ver: TrType,
    d: &[i32],
    bit_depth: u32,
    log2_transform_range: u32,
) -> Result<Vec<i32>> {
    if d.len() != n_tb_w * n_tb_h {
        return Err(Error::invalid(format!(
            "h266 2D xfm: input array length {} != {}x{}",
            d.len(),
            n_tb_w,
            n_tb_h
        )));
    }
    // Step 1 — vertical 1D transform for each non-zero column when
    // nTbH > 1. Result is `e[x][y]`, held as a row-major
    // `n_tb_w * n_tb_h` buffer.
    let e = if n_tb_h > 1 {
        let mut e = vec![0i32; n_tb_w * n_tb_h];
        for x in 0..non_zero_w {
            // Gather d[x][y] for y = 0..non_zero_h - 1.
            let mut col = vec![0i32; non_zero_h];
            for y in 0..non_zero_h {
                col[y] = d[y * n_tb_w + x];
            }
            let out = one_d_transform(tr_type_ver, n_tb_h, non_zero_h, &col)?;
            for y in 0..n_tb_h {
                e[y * n_tb_w + x] = out[y];
            }
        }
        e
    } else {
        // nTbH == 1: e is not used; g[x][0] = d[x][0] per step 3.
        Vec::new()
    };
    // Step 2 — mid-shift of 7 applied to e[x][y] to get g[x][y] (nTbH &
    // nTbW both > 1). Clip to [CoeffMin, CoeffMax].
    let g = if n_tb_h > 1 && n_tb_w > 1 {
        let mut g = vec![0i32; non_zero_w * n_tb_h];
        for y in 0..n_tb_h {
            for x in 0..non_zero_w {
                let e_xy = e[y * n_tb_w + x];
                g[y * non_zero_w + x] = clip_coeff((e_xy + 64) >> 7);
            }
        }
        g
    } else if n_tb_h == 1 {
        // Step 3 — g[x][0] = d[x][0].
        let mut g = vec![0i32; non_zero_w];
        for x in 0..non_zero_w {
            g[x] = d[x];
        }
        g
    } else {
        // n_tb_w == 1: r[0][y] = e[0][y] per step 5. Put this into g so
        // the step-4 branch is a no-op.
        let mut g = vec![0i32; n_tb_h];
        for y in 0..n_tb_h {
            g[y] = e[y * n_tb_w];
        }
        g
    };
    // Step 4 / 5 — horizontal 1D transform per row, or passthrough
    // when nTbW == 1.
    let r = if n_tb_w > 1 {
        let mut r = vec![0i32; n_tb_w * n_tb_h];
        let stride = if n_tb_h > 1 { non_zero_w } else { non_zero_w };
        for y in 0..n_tb_h {
            let row_slice = if n_tb_h > 1 {
                &g[y * stride..y * stride + stride]
            } else {
                // nTbH == 1: single row g[0..non_zero_w].
                &g[..non_zero_w]
            };
            let out = one_d_transform(tr_type_hor, n_tb_w, non_zero_w, row_slice)?;
            for x in 0..n_tb_w {
                r[y * n_tb_w + x] = out[x];
            }
        }
        r
    } else {
        // nTbW == 1: r[0][y] = e[0][y] — we stored that into g above.
        let mut r = vec![0i32; n_tb_h];
        for y in 0..n_tb_h {
            r[y] = g[y];
        }
        r
    };
    // Step 6 — bdShift + rounding.
    let bd_shift = if n_tb_h > 1 && n_tb_w > 1 {
        5 + log2_transform_range - bit_depth
    } else {
        6 + log2_transform_range - bit_depth
    };
    let round = 1i32 << (bd_shift - 1);
    let mut res = vec![0i32; n_tb_w * n_tb_h];
    for i in 0..(n_tb_w * n_tb_h) {
        res[i] = (r[i] + round) >> bd_shift;
    }
    Ok(res)
}

/// Apply the 32-point DST-VII using the two 16-column sub-tables
/// (eqs. 1188 / 1190: rows 0..15 from COL_0_15, rows 16..31 from
/// COL_16_31, both indexed by `[row][n]` with n=0..15 — i.e. the
/// coefficient range is the first 16 columns; the rest of the 32-col
/// row is zero per the sub-table encoding).
///
/// Spec text (eqs. 1188 / 1190): `transMatrix[m][n]` with `n=0..15`
/// comes from the two sub-tables; the n in 16..31 is implicit in the
/// table text as zero.
///
/// NOTE: this matches the spec verbatim — the 32-point DST-VII is
/// restricted to nonZeroS ≤ 16 by the spec, and the decoder is
/// expected to zero-out positions 16..31 upstream. We enforce
/// `nonZeroS ≤ 16` here.
fn apply_dst_vii_32(non_zero_s: usize, x: &[i32]) -> Vec<i32> {
    let non_zero_s = non_zero_s.min(16);
    let mut y = vec![0i32; 32];
    for (m, y_m) in y.iter_mut().enumerate() {
        let row: &[i16; 16] = if m < 16 {
            &DST_VII_32_COL_0_15[m]
        } else {
            &DST_VII_32_COL_16_31[m - 16]
        };
        let mut acc: i64 = 0;
        for j in 0..non_zero_s {
            acc += row[j] as i64 * x[j] as i64;
        }
        *y_m = acc as i32;
    }
    y
}

fn apply_dct_viii_32(non_zero_s: usize, x: &[i32]) -> Vec<i32> {
    let non_zero_s = non_zero_s.min(16);
    let mut y = vec![0i32; 32];
    for (m, y_m) in y.iter_mut().enumerate() {
        let row: &[i16; 16] = if m < 16 {
            &DCT_VIII_32_COL_0_15[m]
        } else {
            &DCT_VIII_32_COL_16_31[m - 16]
        };
        let mut acc: i64 = 0;
        for j in 0..non_zero_s {
            acc += row[j] as i64 * x[j] as i64;
        }
        *y_m = acc as i32;
    }
    y
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A DC-only input (`x[0]` non-zero, others zero) through DST-VII
    /// extracts the first column of the matrix, scaled by `x[0]`.
    #[test]
    fn dst_vii_4_dc_input_returns_first_column() {
        let x = [100, 0, 0, 0];
        let y = one_d_transform(TrType::DstVII, 4, 1, &x).unwrap();
        assert_eq!(y, vec![29 * 100, 74 * 100, 84 * 100, 55 * 100]);
    }

    /// Same for DCT-VIII 4×4.
    #[test]
    fn dct_viii_4_dc_input_returns_first_column() {
        let x = [50, 0, 0, 0];
        let y = one_d_transform(TrType::DctVIII, 4, 1, &x).unwrap();
        assert_eq!(y, vec![84 * 50, 74 * 50, 55 * 50, 29 * 50]);
    }

    /// Feeding an impulse in coefficient position k returns the k-th
    /// column of the matrix. Loop over all 8 positions of DST-VII 8.
    #[test]
    fn dst_vii_8_impulse_responses_match_columns() {
        for k in 0..8 {
            let mut x = [0i32; 8];
            x[k] = 1;
            let y = one_d_transform(TrType::DstVII, 8, 8, &x).unwrap();
            let expected: Vec<i32> = DST_VII_8.iter().map(|row| row[k] as i32).collect();
            assert_eq!(y, expected, "DST-VII-8 column {k} mismatch");
        }
    }

    #[test]
    fn dst_vii_is_linear() {
        let a = [3, -7, 12, 5];
        let b = [-2, 1, 8, -4];
        let sum: Vec<i32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();
        let ta = one_d_transform(TrType::DstVII, 4, 4, &a).unwrap();
        let tb = one_d_transform(TrType::DstVII, 4, 4, &b).unwrap();
        let tsum = one_d_transform(TrType::DstVII, 4, 4, &sum).unwrap();
        for i in 0..4 {
            assert_eq!(tsum[i], ta[i] + tb[i], "linearity fails at row {i}");
        }
    }

    #[test]
    fn non_zero_s_clips_input() {
        let x_full = [10, 20, 30, 40];
        let _y_full = one_d_transform(TrType::DstVII, 4, 4, &x_full).unwrap();
        let y_trunc = one_d_transform(TrType::DstVII, 4, 2, &x_full).unwrap();
        let expected: Vec<i32> = (0..4)
            .map(|i| (DST_VII_4[i][0] as i32 * 10) + (DST_VII_4[i][1] as i32 * 20))
            .collect();
        assert_eq!(y_trunc, expected);
    }

    #[test]
    fn non_zero_s_out_of_range_is_rejected() {
        let x = [1, 2, 3, 4, 5];
        assert!(one_d_transform(TrType::DstVII, 4, 5, &x).is_err());
    }

    /// DCT-II DC column: `transMatrix[m][0] = 64` for m = 0..15 per the
    /// top row of transMatrixCol0to15 (eq. 1180). That row is `{64, 64,
    /// ..., 64}` in the spec — i.e. the first element of each `m`-row
    /// of the logical matrix. In our storage that's
    /// `DCT_II_COL_0_15[0][m]` for m=0..15.
    #[test]
    fn dct_ii_dc_column_first_half_is_64() {
        for m in 0..16 {
            assert_eq!(dct_ii_entry(m, 0), 64, "m={m}");
        }
    }

    /// For size 4, stride=16. The 4×4 inverse-transform submatrix uses
    /// rows 0..3 and columns {0, 16, 32, 48}. Column 0 (applied to the
    /// DC coefficient) is `{64, 64, 64, 64}` for rows 0..3 — the spec's
    /// first-row entries of transMatrixCol0to15.
    #[test]
    fn dct_ii_size4_dc_col_is_64_64_64_64() {
        assert_eq!(dct_ii_entry(0, 0), 64);
        assert_eq!(dct_ii_entry(1, 0), 64);
        assert_eq!(dct_ii_entry(2, 0), 64);
        assert_eq!(dct_ii_entry(3, 0), 64);
    }

    /// The full 64×64 matrix should exhibit the symmetry from eqs.
    /// 1183 / 1184. Spot-check a few rows.
    #[test]
    fn dct_ii_symmetry() {
        // Eq. 1184: row 48 = (n & 1 ? -1 : 1) * transMatrixCol0to15[63 - 48][n]
        //                  = sign * DCT_II_COL_0_15[n][15]. Row n=0, col 15 = 64.
        assert_eq!(dct_ii_entry(48, 0), 64);
        // n=1, col 15 of DCT_II_COL_0_15[1] = 65 (last value in row 1).
        assert_eq!(dct_ii_entry(48, 1), -65);
        // Eq. 1183: row 32 = sign * transMatrixCol16to31[47 - 32][n]
        //                  = sign * DCT_II_COL_16_31[n][15].
        // At n=0: DCT_II_COL_16_31[0][15] = 64 → transMatrix[32][0] = +64.
        assert_eq!(dct_ii_entry(32, 0), 64);
        // At n=1: DCT_II_COL_16_31[1][15] = 2 → transMatrix[32][1] = -2.
        assert_eq!(dct_ii_entry(32, 1), -2);
        // At m=47, 47-m=0 → DCT_II_COL_16_31[n][0] (the DC column of the
        // second half). At n=0: value=64. At n=1: value=62.
        assert_eq!(dct_ii_entry(47, 0), 64);
        assert_eq!(dct_ii_entry(47, 1), -62);
        // At m=63, 63-m=0 → DCT_II_COL_0_15[n][0] (DC column of first half).
        // At n=0: 64. At n=1: -91. At n=2: 90.
        assert_eq!(dct_ii_entry(63, 0), 64);
        assert_eq!(dct_ii_entry(63, 1), -91);
        assert_eq!(dct_ii_entry(63, 2), 90);
    }

    /// DCT-II size-4 impulse: x = [1,0,0,0] feeds the DC coefficient,
    /// which maps to the first column of transMatrix — all 64s (see
    /// eq. 1180 top row).
    #[test]
    fn dct_ii_size4_dc_impulse() {
        let x = [1, 0, 0, 0];
        let y = one_d_transform(TrType::DctII, 4, 1, &x).unwrap();
        assert_eq!(y, vec![64, 64, 64, 64]);
    }

    /// DCT-II size-4 second-coefficient impulse: `x = [0, 1, 0, 0]`
    /// with stride=16 accesses `transMatrix[i][16]` for i=0..3. The
    /// spec's `transMatrixCol0to15[m][16] = DCT_II_COL_0_15[16][m]`
    /// which is PDF row 16 (starts with 83): `[83, 36, -36, -83]`.
    #[test]
    fn dct_ii_size4_ac_impulse() {
        let x = [0, 1, 0, 0];
        let y = one_d_transform(TrType::DctII, 4, 2, &x).unwrap();
        assert_eq!(y, vec![83, 36, -36, -83]);
    }

    /// DCT-II linearity.
    #[test]
    fn dct_ii_is_linear() {
        let a = [3, -7, 12, 5];
        let b = [-2, 1, 8, -4];
        let sum: Vec<i32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();
        let ta = one_d_transform(TrType::DctII, 4, 4, &a).unwrap();
        let tb = one_d_transform(TrType::DctII, 4, 4, &b).unwrap();
        let tsum = one_d_transform(TrType::DctII, 4, 4, &sum).unwrap();
        for i in 0..4 {
            assert_eq!(tsum[i], ta[i] + tb[i]);
        }
    }

    /// DST-VII / DCT-VIII at size 16 / 32 now land.
    #[test]
    fn dst_vii_16_dc_impulse() {
        let mut x = [0i32; 16];
        x[0] = 1;
        let y = one_d_transform(TrType::DstVII, 16, 1, &x).unwrap();
        let expected: Vec<i32> = DST_VII_16.iter().map(|row| row[0] as i32).collect();
        assert_eq!(y, expected);
    }

    #[test]
    fn dct_viii_16_dc_impulse() {
        let mut x = [0i32; 16];
        x[0] = 1;
        let y = one_d_transform(TrType::DctVIII, 16, 1, &x).unwrap();
        let expected: Vec<i32> = DCT_VIII_16.iter().map(|row| row[0] as i32).collect();
        assert_eq!(y, expected);
    }

    /// Size 64 DCT-II still unsupported.
    #[test]
    fn dct_ii_64_is_unsupported() {
        let x = vec![0i32; 64];
        assert!(one_d_transform(TrType::DctII, 64, 64, &x).is_err());
    }

    /// MTS table 39 lookup. Only index 0 maps (DCT-II, DCT-II); others
    /// pick DST-VII / DCT-VIII per the spec table.
    #[test]
    fn mts_idx_table_39() {
        assert_eq!(
            mts_idx_to_tr_types(0),
            Some((TrType::DctII, TrType::DctII))
        );
        assert_eq!(
            mts_idx_to_tr_types(1),
            Some((TrType::DstVII, TrType::DstVII))
        );
        assert_eq!(
            mts_idx_to_tr_types(2),
            Some((TrType::DctVIII, TrType::DstVII))
        );
        assert_eq!(
            mts_idx_to_tr_types(3),
            Some((TrType::DstVII, TrType::DctVIII))
        );
        assert_eq!(
            mts_idx_to_tr_types(4),
            Some((TrType::DctVIII, TrType::DctVIII))
        );
        assert_eq!(mts_idx_to_tr_types(5), None);
    }

    /// Implicit MTS picks DST-VII for dimensions in 4..=16 and DCT-II
    /// otherwise.
    #[test]
    fn implicit_mts_dimension_thresholds() {
        assert_eq!(
            implicit_mts_tr_types(4, 4),
            (TrType::DstVII, TrType::DstVII)
        );
        assert_eq!(
            implicit_mts_tr_types(16, 16),
            (TrType::DstVII, TrType::DstVII)
        );
        assert_eq!(
            implicit_mts_tr_types(32, 32),
            (TrType::DctII, TrType::DctII)
        );
        assert_eq!(
            implicit_mts_tr_types(16, 32),
            (TrType::DstVII, TrType::DctII)
        );
    }

    /// 2D inverse transform DC round-trip: a single DC coefficient in
    /// the top-left corner produces a constant residual (up to the
    /// mid-shift + bdShift rounding). Use DCT-II at N=4 where
    /// transMatrix[*][0] = 64 so the 1D transforms yield 64*k in every
    /// row/col, then the composition adds 64 / shift >> 7 and
    /// 1<<(bdShift-1) / bdShift shifts.
    #[test]
    fn inverse_2d_dct_ii_dc_impulse() {
        // d[0][0] = 100, rest zero. nTbW = nTbH = 4.
        let mut d = vec![0i32; 16];
        d[0] = 100;
        // bitDepth = 10, Log2TransformRange = 15.
        let res = inverse_transform_2d(
            4,
            4,
            1, // non_zero_w: only DC column has non-zero
            1, // non_zero_h: only DC row has non-zero
            TrType::DctII,
            TrType::DctII,
            &d,
            10,
            15,
        )
        .unwrap();
        // Manual derivation:
        //   Step 1: vertical DCT-II on column x=0 with y=0..0. nonZeroH=1, nTbH=4.
        //     e[x=0][y] = DCT_II[y][0] * d[0][0] = 64 * 100 = 6400 for y=0..3.
        //   Step 2: g[0][y] = ((6400 + 64) >> 7) = (6464 >> 7) = 50.
        //   Step 4: horizontal DCT-II on row y with g[0..0][y] (nonZeroW=1).
        //     r[x][y] = DCT_II[x][0] * g[0][y] = 64 * 50 = 3200 for x=0..3.
        //   Step 6: bdShift = 5 + 15 - 10 = 10; round = 512.
        //     res[x][y] = (3200 + 512) >> 10 = 3712 >> 10 = 3.
        for v in &res {
            assert_eq!(*v, 3);
        }
    }

    #[test]
    fn inverse_2d_rejects_bad_input() {
        let d = vec![0i32; 10]; // wrong size for 4x4
        assert!(inverse_transform_2d(
            4,
            4,
            4,
            4,
            TrType::DctII,
            TrType::DctII,
            &d,
            10,
            15
        )
        .is_err());
    }
}
