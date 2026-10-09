// Copyright (c) 2026 Torin Etheridge. MIT License (see LICENSE).
// Author: Torin Etheridge · 2026-10-09

#include "common.metal"

// Fused single-token (decode) projection kernels. A decode step is memory- and launch-latency-bound, so each fused
// kernel removes whole dispatches (and the global-memory round trips between them) instead of adding arithmetic.
//
//   mode flags (DecParams.flags):
//     FLAG_NORM  : the input row is RMS-normalised on the fly:  xn[k] = x[k] * rsqrt(mean(x²) + eps) * g[k]
//                  (the row is staged in threadgroup memory once per threadgroup, so the separate rmsnorm dispatch disappears)
//     FLAG_SILU  : two weight matrices share the row index: out[n] = silu(W0[n]·x) * (W1[n]·x)  (SwiGLU gate+up in one pass)
//     FLAG_ADD   : out += result instead of out = result (fused residual connection)
//   up to three weight matrices are laid out back to back in the row space (Q, K, V projections): global row r selects
//   matrix 0 for r < n0, matrix 1 for r < n0 + n1, else matrix 2, and writes to the matching output buffer.
//
// One simdgroup produces one output row; 4 simdgroups per threadgroup. Lanes stride over 8-element (f16) or
// half-block (q8_0 / q4_0: 16 elements) units exactly as in matvec.metal, accumulation in float32.
// Requires K <= DEC_MAX_K (the staged row must fit in threadgroup memory) and K % 32 == 0.

constant constexpr uint FLAG_NORM = 1;
constant constexpr uint FLAG_SILU = 2;
constant constexpr uint FLAG_ADD = 4;
constant constexpr uint DEC_MAX_K = 4096;

struct DecParams {
    uint K;
    uint n0, n1, n2;       // rows of each matrix (n1 = n2 = 0 for single-matrix use; in SILU mode n0 rows of both W0 and W1)
    uint flags;
    float eps;
};

template <int FMT>
inline float dec_row_dot(device const uchar* row, threadgroup const float* xs, uint K, uint lane) {
    float acc = 0.0f;
    if (FMT == 0) {                                   // f16
        device const half4* w4 = (device const half4*)row;
        threadgroup const float4* x4 = (threadgroup const float4*)xs;
        for (uint u = lane; u < K / 8; u += SIMD_WIDTH)
            acc += dot(float4(w4[2 * u]), x4[2 * u]) + dot(float4(w4[2 * u + 1]), x4[2 * u + 1]);
    } else if (FMT == 1) {                            // q8_0
        for (uint u = lane; u < (K / Q_BLOCK) * 2; u += SIMD_WIDTH) {
            uint b = u >> 1, h = u & 1;
            device const uchar* p = row + (ulong)b * Q8_BYTES;
            float d = float(*(device const half*)p);
            device const packed_char4* q = (device const packed_char4*)(p + 2 + h * 16);
            threadgroup const float4* x4 = (threadgroup const float4*)(xs + b * Q_BLOCK + h * 16);
            acc += d * (dot(float4(q[0]), x4[0]) + dot(float4(q[1]), x4[1]) + dot(float4(q[2]), x4[2]) + dot(float4(q[3]), x4[3]));
        }
    } else {                                          // q4_0
        for (uint u = lane; u < (K / Q_BLOCK) * 2; u += SIMD_WIDTH) {
            uint b = u >> 1, h = u & 1;
            device const uchar* p = row + (ulong)b * Q4_BYTES;
            float d = float(*(device const half*)p);
            device const uchar* qs = p + 2 + h * 8;
            float4 lo0, lo1, hi0, hi1;
            for (uint j = 0; j < 4; j++) {
                lo0[j] = float(int(qs[j] & 0x0F) - 8);     hi0[j] = float(int(qs[j] >> 4) - 8);
                lo1[j] = float(int(qs[j + 4] & 0x0F) - 8); hi1[j] = float(int(qs[j + 4] >> 4) - 8);
            }
            threadgroup const float4* xl = (threadgroup const float4*)(xs + b * Q_BLOCK + h * 8);
            threadgroup const float4* xh = (threadgroup const float4*)(xs + b * Q_BLOCK + 16 + h * 8);
            acc += d * (dot(lo0, xl[0]) + dot(lo1, xl[1]) + dot(hi0, xh[0]) + dot(hi1, xh[1]));
        }
    }
    return acc;
}

template <int FMT>
inline uint dec_row_bytes(uint K) { return FMT == 0 ? K * 2 : (FMT == 1 ? (K / Q_BLOCK) * Q8_BYTES : (K / Q_BLOCK) * Q4_BYTES); }

template <int FMT>
inline void dec_impl(device const uchar* W0, device const uchar* W1, device const uchar* W2,
                     device const float* X, device const float* G,
                     device float* Y0, device float* Y1, device float* Y2,
                     constant DecParams& P, uint3 tgid, uint tid, uint lane, uint sg, threadgroup float* xs, threadgroup float* red) {
    const uint K = P.K;
    // Stage the (optionally RMS-normalised) input row in threadgroup memory.
    float ss = 0.0f;
    for (uint i = tid; i < K; i += 4 * SIMD_WIDTH) { float v = X[i]; xs[i] = v; ss += v * v; }
    if (P.flags & FLAG_NORM) {
        ss = simd_sum(ss);
        if (lane == 0) red[sg] = ss;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float total = red[0] + red[1] + red[2] + red[3];
        float scale = 1.0f / precise::sqrt(total / float(K) + P.eps);
        for (uint i = tid; i < K; i += 4 * SIMD_WIDTH) xs[i] = xs[i] * scale * G[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint r = tgid.x * 4 + sg;
    const bool silu = (P.flags & FLAG_SILU) != 0;
    const uint total = silu ? P.n0 : P.n0 + P.n1 + P.n2;
    if (r >= total) return;
    const uint rb = dec_row_bytes<FMT>(K);

    if (silu) {
        float g = simd_sum(dec_row_dot<FMT>(W0 + (ulong)r * rb, xs, K, lane));
        float u = simd_sum(dec_row_dot<FMT>(W1 + (ulong)r * rb, xs, K, lane));
        if (lane == 0) Y0[r] = (g / (1.0f + precise::exp(-g))) * u;
        return;
    }
    device const uchar* W = W0; device float* Y = Y0; uint local = r;
    if (r >= P.n0 + P.n1) { W = W2; Y = Y2; local = r - P.n0 - P.n1; }
    else if (r >= P.n0) { W = W1; Y = Y1; local = r - P.n0; }
    float s = simd_sum(dec_row_dot<FMT>(W + (ulong)local * rb, xs, K, lane));
    if (lane == 0) Y[local] = ((P.flags & FLAG_ADD) ? Y[local] : 0.0f) + s;
}

#define DEC_KERNEL(NAME, FMT)                                                                                  \
    kernel void NAME(device const uchar* W0 [[buffer(0)]], device const uchar* W1 [[buffer(1)]],               \
                     device const uchar* W2 [[buffer(2)]], device const float* X [[buffer(3)]],                \
                     device const float* G [[buffer(4)]], device float* Y0 [[buffer(5)]],                       \
                     device float* Y1 [[buffer(6)]], device float* Y2 [[buffer(7)]],                            \
                     constant DecParams& P [[buffer(8)]], uint3 tgid [[threadgroup_position_in_grid]],          \
                     uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],         \
                     uint sg [[simdgroup_index_in_threadgroup]], threadgroup float* xs [[threadgroup(0)]]) {    \
        threadgroup float red[4];                                                                               \
        dec_impl<FMT>(W0, W1, W2, X, G, Y0, Y1, Y2, P, tgid, tid, lane, sg, xs, red);                           \
    }
DEC_KERNEL(dec_f16, 0)
DEC_KERNEL(dec_q8_0, 1)
DEC_KERNEL(dec_q4_0, 2)

// RoPE of q (in place) and RoPE + cache append of k, v in a single dispatch (decode: one token).
// Grid: (headDim/2, heads + kvHeads, tokens). y < heads rotates q head y; otherwise processes KV head y - heads.
template <typename KV>
inline void rope_qkv_impl(device float* q, device const float* k, device const float* v, device KV* kCache, device KV* vCache,
                          device const float* freqs, uint heads, uint kvHeads, uint headDim, uint startPos, uint3 gid) {
    uint pair = gid.x, y = gid.y, t = gid.z;
    if (pair >= headDim / 2) return;
    float angle = float(startPos + t) * freqs[pair];
    float c = precise::cos(angle), s = precise::sin(angle);
    if (y < heads) {
        device float* p = q + ((ulong)t * heads + y) * headDim + 2 * pair;
        float a = p[0], b = p[1];
        p[0] = a * c - b * s; p[1] = a * s + b * c;
    } else {
        uint h = y - heads;
        ulong src = ((ulong)t * kvHeads + h) * headDim + 2 * pair;
        ulong dst = ((ulong)(startPos + t) * kvHeads + h) * headDim + 2 * pair;
        float a = k[src], b = k[src + 1];
        kCache[dst] = KV(a * c - b * s); kCache[dst + 1] = KV(a * s + b * c);
        vCache[dst] = KV(v[src]); vCache[dst + 1] = KV(v[src + 1]);
    }
}
#define ROPE_QKV_KERNEL(NAME, KV)                                                                              \
    kernel void NAME(device float* q [[buffer(0)]], device const float* k [[buffer(1)]], device const float* v [[buffer(2)]], \
                     device KV* kCache [[buffer(3)]], device KV* vCache [[buffer(4)]], device const float* freqs [[buffer(5)]], \
                     constant uint& heads [[buffer(6)]], constant uint& kvHeads [[buffer(7)]], constant uint& headDim [[buffer(8)]], \
                     constant uint& startPos [[buffer(9)]], uint3 gid [[thread_position_in_grid]]) {            \
        rope_qkv_impl<KV>(q, k, v, kCache, vCache, freqs, heads, kvHeads, headDim, startPos, gid);              \
    }
ROPE_QKV_KERNEL(rope_qkv_store_kv16, half)
ROPE_QKV_KERNEL(rope_qkv_store_kv32, float)
