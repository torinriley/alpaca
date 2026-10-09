// Copyright (c) 2026 Torin Etheridge. MIT License (see LICENSE).
// Author: Torin Etheridge · 2026-10-09

#include "common.metal"

// Rotary embedding. Pair p of head h at token t rotates (x[2p], x[2p+1]) by angle (start + t) * freqs[p],
// where freqs[p] = theta^(-2p/headDim) is computed on the host in double precision.
// Grid: (headDim/2, heads, tokens).
kernel void rope_f32(device float* x [[buffer(0)]], device const float* freqs [[buffer(1)]],
                     constant uint& heads [[buffer(2)]], constant uint& headDim [[buffer(3)]],
                     constant uint& startPos [[buffer(4)]], uint3 gid [[thread_position_in_grid]]) {
    uint pair = gid.x, h = gid.y, t = gid.z;
    if (pair >= headDim / 2) return;
    float angle = float(startPos + t) * freqs[pair];
    float c = precise::cos(angle), s = precise::sin(angle);
    device float* p = x + ((ulong)t * heads + h) * headDim + 2 * pair;
    float a = p[0], b = p[1];
    p[0] = a * c - b * s;
    p[1] = a * s + b * c;
}

// Causal grouped-query attention with online softmax (no score storage, any context length).
// Threadgroup = ATT_SIMDGROUPS simdgroups; one threadgroup per (head, token). Query token t sits at absolute
// position startPos + t and attends to cache rows 0...startPos+t. Each simdgroup walks every ATT_SIMDGROUPS-th
// position keeping a running (max m, denominator l, weighted value sum acc); the partials are merged at the end.
// Lane i owns elements i, i+32, ... of the head dimension (headDim must be a multiple of 32, at most 256).
// Output: ctx[t, head * headDim + i].  Grid: threadgroups = (heads, tokens).
constant constexpr uint ATT_SIMDGROUPS = 16;
constant constexpr uint ATT_MAX_EPL = 8;   // elements per lane: headDim / 32
// Running-max initial value. A finite sentinel (not -inf) because Metal's fast-math mode does not guarantee inf handling.
constant constexpr float NEG_BIG = -1.0e30f;

template <typename KV>
inline void attention_impl(device const float* q, device const KV* kCache, device const KV* vCache, device float* ctx,
                           uint heads, uint kvHeads, uint headDim, uint startPos,
                           uint3 tg, uint lane, uint sg, uint tid,
                           threadgroup float* tgAcc, threadgroup float* tgM, threadgroup float* tgL) {
    const uint h = tg.x, t = tg.y;
    const uint kvh = h / (heads / kvHeads);
    const uint epl = headDim / SIMD_WIDTH;
    const uint last = startPos + t;                    // inclusive causal limit
    const float scale = 1.0f / sqrt(float(headDim));

    float qv[ATT_MAX_EPL], acc[ATT_MAX_EPL];
    device const float* qrow = q + ((ulong)t * heads + h) * headDim;
    for (uint e = 0; e < epl; e++) { qv[e] = qrow[lane + e * SIMD_WIDTH] * scale; acc[e] = 0.0f; }
    float m = NEG_BIG, l = 0.0f;

    for (uint p = sg; p <= last; p += ATT_SIMDGROUPS) {
        ulong base = ((ulong)p * kvHeads + kvh) * headDim;
        float s = 0.0f;
        for (uint e = 0; e < epl; e++) s += qv[e] * float(kCache[base + lane + e * SIMD_WIDTH]);
        s = simd_sum(s);
        float mNew = max(m, s);
        float corr = precise::exp(m - mNew), w = precise::exp(s - mNew);
        l = l * corr + w;
        for (uint e = 0; e < epl; e++) acc[e] = acc[e] * corr + w * float(vCache[base + lane + e * SIMD_WIDTH]);
        m = mNew;
    }
    for (uint e = 0; e < epl; e++) tgAcc[sg * 256 + lane + e * SIMD_WIDTH] = acc[e];
    if (lane == 0) { tgM[sg] = m; tgL[sg] = l; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float M = NEG_BIG;
    for (uint i = 0; i < ATT_SIMDGROUPS; i++) M = max(M, tgM[i]);
    float L = 0.0f, f[ATT_SIMDGROUPS];
    for (uint i = 0; i < ATT_SIMDGROUPS; i++) { f[i] = precise::exp(tgM[i] - M); L += tgL[i] * f[i]; }
    for (uint d = tid; d < headDim; d += ATT_SIMDGROUPS * SIMD_WIDTH) {
        float o = 0.0f;
        for (uint i = 0; i < ATT_SIMDGROUPS; i++) o += tgAcc[i * 256 + d] * f[i];
        ctx[((ulong)t * heads + h) * headDim + d] = o / L;
    }
}

#define ATTENTION_KERNEL(NAME, KV)                                                                          \
    kernel void NAME(device const float* q [[buffer(0)]], device const KV* kCache [[buffer(1)]],            \
                     device const KV* vCache [[buffer(2)]], device float* ctx [[buffer(3)]],                 \
                     constant uint& heads [[buffer(4)]], constant uint& kvHeads [[buffer(5)]],               \
                     constant uint& headDim [[buffer(6)]], constant uint& startPos [[buffer(7)]],            \
                     uint3 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],     \
                     uint sg [[simdgroup_index_in_threadgroup]], uint3 tidv [[thread_position_in_threadgroup]]) { \
        threadgroup float tgAcc[ATT_SIMDGROUPS * 256];                                                       \
        threadgroup float tgM[ATT_SIMDGROUPS], tgL[ATT_SIMDGROUPS];                                          \
        attention_impl<KV>(q, kCache, vCache, ctx, heads, kvHeads, headDim, startPos, tg, lane, sg, tidv.x, tgAcc, tgM, tgL); \
    }
ATTENTION_KERNEL(attention_kv16, half)
ATTENTION_KERNEL(attention_kv32, float)
