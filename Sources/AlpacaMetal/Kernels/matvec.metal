// Copyright (c) 2026 Torin Etheridge. MIT License (see LICENSE).
// Author: Torin Etheridge · 2026-10-09

#include "common.metal"

// Linear layer  Y[t, n] = sum_k X[t, k] * W[n, k]   for W in f16 / q8_0 / q4_0.
//
// One simdgroup computes one output row n for TB tokens at once, so each weight element is read from memory
// once per TB tokens (TB = 1 for decode, 8 for prefill chunks). Lanes stride across the row in "units" of
// 8 elements (f16) or half a quantisation block (q8_0, q4_0: 16 elements, 8 packed bytes); partial sums are
// reduced with simd_sum. Quantised weights are never expanded to memory: they are dequantised in registers.
// With acc != 0 the result is added to Y instead of overwriting it (fused residual connection).
// Grid: threadgroups = (ceil(N / MV_SIMDGROUPS), ceil(T / TB)), 32 * MV_SIMDGROUPS threads per threadgroup.

template <uint TB>
inline void mv_f16_impl(device const uchar* W, device const float* X, device float* Y,
                        uint K, uint N, uint T, uint acc_, uint3 tg, uint lane, uint sg) {
    uint n = tg.x * MV_SIMDGROUPS + sg;
    if (n >= N) return;
    uint t0 = tg.y * TB;
    device const half4* w4 = (device const half4*)(W + (ulong)n * K * 2);
    float acc[TB];
    for (uint t = 0; t < TB; t++) acc[t] = 0.0f;
    for (uint u = lane; u < K / 8; u += SIMD_WIDTH) {
        float4 wa = float4(w4[2 * u]), wb = float4(w4[2 * u + 1]);
        for (uint t = 0; t < TB; t++) {
            if (t0 + t >= T) break;
            device const float4* x4 = (device const float4*)(X + (ulong)(t0 + t) * K);
            acc[t] += dot(wa, x4[2 * u]) + dot(wb, x4[2 * u + 1]);
        }
    }
    for (uint t = 0; t < TB; t++) {
        float s = simd_sum(acc[t]);
        if (lane == 0 && t0 + t < T) { device float* yp = Y + (ulong)(t0 + t) * N + n; *yp = (acc_ ? *yp : 0.0f) + s; }
    }
}

template <uint TB>
inline void mv_q8_0_impl(device const uchar* W, device const float* X, device float* Y,
                         uint K, uint N, uint T, uint acc_, uint3 tg, uint lane, uint sg) {
    uint n = tg.x * MV_SIMDGROUPS + sg;
    if (n >= N) return;
    uint t0 = tg.y * TB;
    device const uchar* row = W + (ulong)n * (K / Q_BLOCK) * Q8_BYTES;
    float acc[TB];
    for (uint t = 0; t < TB; t++) acc[t] = 0.0f;
    for (uint u = lane; u < (K / Q_BLOCK) * 2; u += SIMD_WIDTH) {
        uint b = u >> 1, h = u & 1;
        device const uchar* p = row + (ulong)b * Q8_BYTES;
        float d = float(*(device const half*)p);
        device const packed_char4* q = (device const packed_char4*)(p + 2 + h * 16);
        float4 w0 = float4(q[0]), w1 = float4(q[1]), w2 = float4(q[2]), w3 = float4(q[3]);
        for (uint t = 0; t < TB; t++) {
            if (t0 + t >= T) break;
            device const float4* x4 = (device const float4*)(X + (ulong)(t0 + t) * K + b * Q_BLOCK + h * 16);
            float s = dot(w0, x4[0]) + dot(w1, x4[1]) + dot(w2, x4[2]) + dot(w3, x4[3]);
            acc[t] += d * s;
        }
    }
    for (uint t = 0; t < TB; t++) {
        float s = simd_sum(acc[t]);
        if (lane == 0 && t0 + t < T) { device float* yp = Y + (ulong)(t0 + t) * N + n; *yp = (acc_ ? *yp : 0.0f) + s; }
    }
}

template <uint TB>
inline void mv_q4_0_impl(device const uchar* W, device const float* X, device float* Y,
                         uint K, uint N, uint T, uint acc_, uint3 tg, uint lane, uint sg) {
    uint n = tg.x * MV_SIMDGROUPS + sg;
    if (n >= N) return;
    uint t0 = tg.y * TB;
    device const uchar* row = W + (ulong)n * (K / Q_BLOCK) * Q4_BYTES;
    float acc[TB];
    for (uint t = 0; t < TB; t++) acc[t] = 0.0f;
    for (uint u = lane; u < (K / Q_BLOCK) * 2; u += SIMD_WIDTH) {
        uint b = u >> 1, h = u & 1;
        device const uchar* p = row + (ulong)b * Q4_BYTES;
        float d = float(*(device const half*)p);
        // This lane's 8 packed bytes: low nibbles are elements [8h, 8h+8), high nibbles elements [16+8h, 16+8h+8).
        device const uchar* qs = p + 2 + h * 8;
        float4 lo0, lo1, hi0, hi1;
        for (uint j = 0; j < 4; j++) {
            lo0[j] = float(int(qs[j] & 0x0F) - 8);     hi0[j] = float(int(qs[j] >> 4) - 8);
            lo1[j] = float(int(qs[j + 4] & 0x0F) - 8); hi1[j] = float(int(qs[j + 4] >> 4) - 8);
        }
        for (uint t = 0; t < TB; t++) {
            if (t0 + t >= T) break;
            device const float4* xl = (device const float4*)(X + (ulong)(t0 + t) * K + b * Q_BLOCK + h * 8);
            device const float4* xh = (device const float4*)(X + (ulong)(t0 + t) * K + b * Q_BLOCK + 16 + h * 8);
            float s = dot(lo0, xl[0]) + dot(lo1, xl[1]) + dot(hi0, xh[0]) + dot(hi1, xh[1]);
            acc[t] += d * s;
        }
    }
    for (uint t = 0; t < TB; t++) {
        float s = simd_sum(acc[t]);
        if (lane == 0 && t0 + t < T) { device float* yp = Y + (ulong)(t0 + t) * N + n; *yp = (acc_ ? *yp : 0.0f) + s; }
    }
}

#define MV_KERNEL(FMT, TBV)                                                                              \
    kernel void mv_##FMT##_tb##TBV(device const uchar* W [[buffer(0)]], device const float* X [[buffer(1)]], \
                                   device float* Y [[buffer(2)]], constant uint& K [[buffer(3)]],            \
                                   constant uint& N [[buffer(4)]], constant uint& T [[buffer(5)]],           \
                                   constant uint& acc [[buffer(6)]],                                          \
                                   uint3 tg [[threadgroup_position_in_grid]],                                \
                                   uint lane [[thread_index_in_simdgroup]],                                  \
                                   uint sg [[simdgroup_index_in_threadgroup]]) {                             \
        mv_##FMT##_impl<TBV>(W, X, Y, K, N, T, acc, tg, lane, sg);                                                \
    }

MV_KERNEL(f16, 1)   MV_KERNEL(f16, 8)
MV_KERNEL(q8_0, 1)  MV_KERNEL(q8_0, 8)
MV_KERNEL(q4_0, 1)  MV_KERNEL(q4_0, 8)
