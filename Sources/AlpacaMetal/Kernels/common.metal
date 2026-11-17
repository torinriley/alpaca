// Copyright (c) 2026 Torin Etheridge. MIT License (see LICENSE).
// Author: Torin Etheridge · 2026-10-09

#include <metal_stdlib>
using namespace metal;

// alpaca Metal kernels. Conventions shared by every kernel file:
//  * Activations are float32, row-major [tokens, width].
//  * Weights are [out, in] row-major in their GGUF on-disk layout (f16 / q8_0 / q4_0 blocks), read in place.
//  * One simdgroup (32 lanes) reduces one dot product; all accumulation is float32.
//  * Dispatches are encoded on one serial compute encoder, so each kernel observes the previous kernel's writes.
// Block layouts (see DType.swift):
//  q8_0: half d; int8 qs[32]           = 34 bytes, value = d * qs[j]
//  q4_0: half d; uint8 qs[16]          = 18 bytes, value = d * ((qs[j] & 0xF) - 8) for element j,
//                                                           d * ((qs[j] >> 4) - 8)  for element j + 16

constant constexpr uint SIMD_WIDTH = 32;
constant constexpr uint Q_BLOCK = 32;
constant constexpr uint Q8_BYTES = 34;
constant constexpr uint Q4_BYTES = 18;
// Simdgroups per threadgroup in the mat-vec kernels: each simdgroup produces one output row.
constant constexpr uint MV_SIMDGROUPS = 4;

// Finite stand-in for -infinity in running-max initialisation (Metal fast-math does not guarantee inf semantics).
constant constexpr float NEG_BIG_PF = -1.0e30f;
constant constexpr float INFINITY_F = 3.0e38f;   // finite stand-in for +inf (fast-math safe): 'below every logit' sentinel

// ---------------------------------------------------------------------------------------------------------------------
// Generic weight formats beyond f16 / q8_0 / q4_0. Every format is read through one primitive: dq8<FMT>(row, u, a, b)
// dequantises the 8 consecutive elements [8u, 8u+8) of a weight row into a (first four) and b (last four). All 8 elements of a
// unit share one scale / sub-block in every format, so a lane needs one scale decode per unit. The kernels in
// quant_generic.metal, decode_fused.metal and matmul_small.metal are written once against this primitive.
// Layouts: see Docs/FORMATS.md (ports of ggml's dequantize_row_*; validated against llama.cpp output).
// ---------------------------------------------------------------------------------------------------------------------
constant constexpr int FMT_Q4_1 = 3, FMT_Q5_0 = 4, FMT_Q5_1 = 5, FMT_Q2_K = 6, FMT_Q3_K = 7, FMT_Q4_K = 8, FMT_Q5_K = 9, FMT_Q6_K = 10, FMT_BF16 = 11;

template <int FMT> inline uint fmt_block_elems() { return FMT >= FMT_Q2_K && FMT <= FMT_Q6_K ? 256u : (FMT == FMT_BF16 ? 1u : 32u); }
template <int FMT> inline uint fmt_block_bytes() {
    return FMT == FMT_Q4_1 ? 20u : FMT == FMT_Q5_0 ? 22u : FMT == FMT_Q5_1 ? 24u : FMT == FMT_Q2_K ? 84u : FMT == FMT_Q3_K ? 110u
         : FMT == FMT_Q4_K ? 144u : FMT == FMT_Q5_K ? 176u : FMT == FMT_Q6_K ? 210u : 2u;
}
template <int FMT> inline ulong fmt_row_bytes(uint K) { return (ulong)(K / fmt_block_elems<FMT>()) * fmt_block_bytes<FMT>(); }

inline uint ld_u32(device const uchar* p) { return uint(p[0]) | (uint(p[1]) << 8) | (uint(p[2]) << 16) | (uint(p[3]) << 24); }
inline float ld_half(device const uchar* p) { return float(as_type<half>(ushort(uint(p[0]) | (uint(p[1]) << 8)))); }
// 8 consecutive bytes as two uchar4.
inline void ld_bytes8(device const uchar* p, thread uchar4& lo, thread uchar4& hi) {
    lo = uchar4(*(device const packed_uchar4*)p); hi = uchar4(*(device const packed_uchar4*)(p + 4));
}
// Low (hi == false) or high nibbles of 8 bytes as floats.
inline void nibbles8(device const uchar* p, bool hi, thread float4& a, thread float4& b) {
    uchar4 l, h; ld_bytes8(p, l, h);
    a = float4(hi ? (l >> 4) : (l & 0x0F)); b = float4(hi ? (h >> 4) : (h & 0x0F));
}
// 6-bit (scale, min) pair j of the 12-byte K-quant scale array.
inline float2 k4_scale_min(device const uchar* q, uint j) {
    return j < 4 ? float2(float(q[j] & 63), float(q[j + 4] & 63))
                 : float2(float((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4)), float((q[j + 4] >> 4) | ((q[j] >> 6) << 4)));
}

template <int FMT>
inline void dq8(device const uchar* row, uint u, thread float4& a, thread float4& b) {
    if (FMT == FMT_BF16) {
        device const packed_ushort4* p = (device const packed_ushort4*)(row + (ulong)u * 16);
        ushort4 x = ushort4(p[0]), y = ushort4(p[1]);
        a = as_type<float4>(uint4(x) << 16); b = as_type<float4>(uint4(y) << 16);
    } else if (FMT == FMT_Q4_1 || FMT == FMT_Q5_0 || FMT == FMT_Q5_1) {
        const uint blk = u >> 2, j = u & 3;
        device const uchar* p = row + (ulong)blk * fmt_block_bytes<FMT>();
        const bool hi = j >= 2;
        if (FMT == FMT_Q4_1) {
            float d = ld_half(p), m = ld_half(p + 2);
            float4 x, y; nibbles8(p + 4 + (j & 1) * 8, hi, x, y);
            a = d * x + m; b = d * y + m;
        } else {
            const bool one = FMT == FMT_Q5_1;
            float d = ld_half(p), m = one ? ld_half(p + 2) : 0.0f;
            uint qh = ld_u32(p + (one ? 4 : 2));
            float4 x, y; nibbles8(p + (one ? 8 : 6) + (j & 1) * 8, hi, x, y);
            uint4 hb0 = (uint4(qh >> (j * 8)) >> uint4(0, 1, 2, 3)) & 1u, hb1 = (uint4(qh >> (j * 8 + 4)) >> uint4(0, 1, 2, 3)) & 1u;
            x += float4(hb0) * 16.0f; y += float4(hb1) * 16.0f;
            if (one) { a = d * x + m; b = d * y + m; } else { a = d * (x - 16.0f); b = d * (y - 16.0f); }
        }
    } else {
        const uint blk = u >> 5, e0 = (u & 31) * 8;
        device const uchar* p = row + (ulong)blk * fmt_block_bytes<FMT>();
        if (FMT == FMT_Q2_K || FMT == FMT_Q3_K) {
            const uint half_ = e0 >> 7, r = e0 & 127, j = r >> 5, s = r & 31, si = half_ * 8 + j * 2 + (s >= 16 ? 1 : 0);
            uchar4 l, h; ld_bytes8(p + (FMT == FMT_Q2_K ? 16 : 32) + half_ * 32 + s, l, h);
            float4 q0 = float4((l >> (2 * j)) & 3), q1 = float4((h >> (2 * j)) & 3);
            if (FMT == FMT_Q2_K) {
                float d = ld_half(p + 80), dmin = ld_half(p + 82);
                uint sc = p[si];
                float dl = d * float(sc & 0xF), ml = dmin * float(sc >> 4);
                a = dl * q0 - ml; b = dl * q1 - ml;
            } else {
                float d = ld_half(p + 108);
                uint low = si < 8 ? (p[96 + si] & 0xF) : (p[96 + si - 8] >> 4);
                uint high = (p[104 + (si & 3)] >> (2 * (si >> 2))) & 3;
                float dl = d * float(int(low | (high << 4)) - 32);
                uchar4 hl, hh; ld_bytes8(p + s, hl, hh);
                const uint hbit = half_ * 4 + j;
                float4 c0 = float4((hl >> hbit) & 1), c1 = float4((hh >> hbit) & 1);
                a = dl * (q0 - 4.0f * (1.0f - c0)); b = dl * (q1 - 4.0f * (1.0f - c1));
            }
        } else if (FMT == FMT_Q4_K || FMT == FMT_Q5_K) {
            const uint c = e0 >> 6, r = e0 & 63, l0 = r & 31;
            const bool hi = r >= 32;
            float d = ld_half(p), dmin = ld_half(p + 2);
            float2 smn = k4_scale_min(p + 4, 2 * c + (hi ? 1 : 0));
            float4 x, y; nibbles8(p + (FMT == FMT_Q4_K ? 16 : 48) + c * 32 + l0, hi, x, y);
            if (FMT == FMT_Q5_K) {
                uchar4 hl, hh; ld_bytes8(p + 16 + l0, hl, hh);
                const uint hbit = 2 * c + (hi ? 1 : 0);
                x += 16.0f * float4((hl >> hbit) & 1); y += 16.0f * float4((hh >> hbit) & 1);
            }
            float dl = d * smn.x, ml = dmin * smn.y;
            a = dl * x - ml; b = dl * y - ml;
        } else {   // FMT_Q6_K
            const uint half_ = e0 >> 7, r = e0 & 127, g = r >> 5, l0 = r & 31;
            float d = ld_half(p + 208);
            float sc = float(((device const char*)(p + 192))[half_ * 8 + (l0 >> 4) + 2 * g]);
            float4 x, y; nibbles8(p + half_ * 64 + l0 + (g & 1) * 32, g >= 2, x, y);
            uchar4 hl, hh; ld_bytes8(p + 128 + half_ * 32 + l0, hl, hh);
            float4 h0 = float4((hl >> (2 * g)) & 3), h1 = float4((hh >> (2 * g)) & 3);
            a = (d * sc) * (x + 16.0f * h0 - 32.0f); b = (d * sc) * (y + 16.0f * h1 - 32.0f);
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Q4_K / Q6_K single-row dot products for decode (one simdgroup per output row). 8 lanes cooperate on one 256-element super-block, so
// four super-blocks are in flight per simdgroup; each lane owns 32 consecutive elements (one sub-block), decodes its scale once and loads
// its packed bytes as vectors, instead of repeating the scale decode for every 8 elements as dq8 does. XP is the activation pointer type
// (threadgroup for the fused kernels, device for the plain mat-vec).
// ---------------------------------------------------------------------------------------------------------------------
inline float4 xld4(threadgroup const float* p, uint i) { return *(threadgroup const float4*)(p + i * 4); }
inline float4 xld4(device const float* p, uint i) { return *(device const float4*)(p + i * 4); }

template <int FMT, typename XP>
inline float kq_row_dot(device const uchar* row, XP xs, uint K, uint lane) {
    float acc = 0.0f;
    if (FMT == FMT_Q4_K) {
        // 8 lanes per super-block, one 32-element sub-block each (scale decoded once, 32 packed bytes loaded as two 16-byte vectors):
        // sum((d*sc*n - dmin*m) * x) = d*sc*sum(n*x) - dmin*m*sum(x). Four super-blocks are in flight per simdgroup.
        const uint t = lane & 7, nsb = K >> 8;
        const uint shift = (t & 1) * 4, c = t >> 1;
        for (uint sb = lane >> 3; sb < nsb; sb += 4) {
            device const uchar* p = row + (ulong)sb * 144;
            const float d = ld_half(p), dmin = ld_half(p + 2);
            const float2 smn = k4_scale_min(p + 4, t);
            device const packed_uint4* q = (device const packed_uint4*)(p + 16 + c * 32);
            XP xb = xs + sb * 256 + t * 32;
            float s = 0.0f; float4 sx = float4(0.0f);
            for (uint k = 0; k < 2; k++) {
                uint4 w = uint4(q[k]);
                for (uint e = 0; e < 4; e++) {
                    float4 f = float4(as_type<uchar4>(w[e]) >> shift & 0x0F), xv = xld4(xb, k * 4 + e);
                    s += dot(f, xv); sx += xv;
                }
            }
            acc += d * smn.x * s - dmin * smn.y * (sx.x + sx.y + sx.z + sx.w);
        }
    } else {
        // 8 lanes per super-block: (half, group) selects 32 elements = 32 ql bytes + 32 qh bytes and two int8 scales.
        const uint t = lane & 7, nsb = K >> 8, h = t >> 2, g = t & 3;
        const uint nshift = g >= 2 ? 4 : 0, hshift = 2 * g;
        for (uint sb = lane >> 3; sb < nsb; sb += 4) {
            device const uchar* p = row + (ulong)sb * 210;
            const float d = ld_half(p + 208);
            device const char* sc = (device const char*)(p + 192) + h * 8 + 2 * g;
            device const packed_uchar4* ql = (device const packed_uchar4*)(p + h * 64 + (g & 1) * 32);
            device const packed_uchar4* qh = (device const packed_uchar4*)(p + 128 + h * 32);
            XP xb = xs + sb * 256 + h * 128 + g * 32;
            float a0 = 0.0f, a1 = 0.0f;
            for (uint k = 0; k < 8; k++) {
                uchar4 lo = (uchar4(ql[k]) >> nshift) & 0x0F, hb = (uchar4(qh[k]) >> hshift) & 3;
                float part = dot(float4(lo | (hb << 4)) - 32.0f, xld4(xb, k));
                if (k < 4) a0 += part; else a1 += part;
            }
            acc += d * (float(sc[0]) * a0 + float(sc[1]) * a1);
        }
    }
    return acc;
}
