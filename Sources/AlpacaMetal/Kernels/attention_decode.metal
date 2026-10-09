// Copyright (c) 2026 Torin Etheridge. MIT License (see LICENSE).
// Author: Torin Etheridge · 2026-10-09

#include "common.metal"

// Split-K causal attention for single-token decode (f16 KV cache), written for this project.
//
// Why: with one query token a layer has only `heads` units of independent work, which cannot occupy the GPU, and the
// previous kernel re-read each KV row once per query head that shares it (GQA group of G heads).
//
//   Phase 1  attn_decode_split_*  grid = (splits, kvHeads), 4 simdgroups per threadgroup.
//     * One threadgroup handles one KV head over `splitLen` consecutive positions and ALL G query heads that share
//       it, so every K/V row is read from memory exactly once.
//     * Within a simdgroup, a quad of 4 lanes owns one position (lane i holds elements 4i..4i+3 of each 16-element
//       chunk, 16B..32B contiguous loads), so 8 positions advance per iteration and the dot product needs only two
//       shuffle steps instead of a 32-lane reduction.
//     * Each quad keeps an online-softmax state (running max m, denominator l, weighted V sum) per query head;
//       states are merged across quads with shuffles, then across simdgroups through threadgroup memory.
//   Phase 2  attn_decode_merge  grid = (heads): combines the per-split partial states with the usual max/denominator
//     rescaling and writes the normalised context vector.
//
// Partial-state layout: [kvHeads][splits][G][HD + 2] floats: HD accumulator values, then m, then l.

constant constexpr uint DEC_SIMDGROUPS = 4;
constant constexpr uint DEC_MAX_G = 8;       // query heads per KV head supported

template <uint HD>
inline void attn_decode_split_impl(device const float* q, device const half* kc, device const half* vc, device float* partial,
                                   uint heads, uint kvHeads, uint L, uint splitLen, uint nSplits,
                                   uint2 tg, uint lane, uint sg, uint tid, threadgroup float* tgBuf) {
    constexpr uint CH = HD / 16;                         // 16-element chunks per row; each lane handles 4 elements per chunk
    const uint G = heads / kvHeads, kvh = tg.y, split = tg.x;
    const uint quad = lane >> 2, qi = lane & 3;
    const uint kvStride = kvHeads * HD;
    const uint pBegin = split * splitLen, pEnd = min(L, pBegin + splitLen);
    const float scale = 1.0f / sqrt(float(HD));

    float qv[DEC_MAX_G][CH * 4], acc[DEC_MAX_G][CH * 4], m[DEC_MAX_G], l[DEC_MAX_G];
    for (uint g = 0; g < G; g++) {
        device const float* qrow = q + (ulong)(kvh * G + g) * HD;
        for (uint j = 0; j < CH; j++) for (uint e = 0; e < 4; e++) { qv[g][j * 4 + e] = qrow[j * 16 + qi * 4 + e] * scale; acc[g][j * 4 + e] = 0.0f; }
        m[g] = NEG_BIG_PF; l[g] = 0.0f;
    }

    for (uint p = pBegin + sg * 8 + quad; p < pEnd; p += DEC_SIMDGROUPS * 8) {
        device const half* krow = kc + (ulong)p * kvStride + kvh * HD + qi * 4;
        device const half* vrow = vc + (ulong)p * kvStride + kvh * HD + qi * 4;
        float kk[CH * 4], vv[CH * 4];
        for (uint j = 0; j < CH; j++) {
            half4 k4 = *(device const half4*)(krow + j * 16), v4 = *(device const half4*)(vrow + j * 16);
            for (uint e = 0; e < 4; e++) { kk[j * 4 + e] = float(k4[e]); vv[j * 4 + e] = float(v4[e]); }
        }
        for (uint g = 0; g < G; g++) {
            float s = 0.0f;
            for (uint i = 0; i < CH * 4; i++) s += qv[g][i] * kk[i];
            s += simd_shuffle_xor(s, 1);
            s += simd_shuffle_xor(s, 2);
            float mn = max(m[g], s);
            float cr = precise::exp(m[g] - mn), w = precise::exp(s - mn);
            l[g] = l[g] * cr + w;
            for (uint i = 0; i < CH * 4; i++) acc[g][i] = acc[g][i] * cr + w * vv[i];
            m[g] = mn;
        }
    }

    // Merge the 8 quads of this simdgroup (lanes with equal qi) by butterfly over lane bits 2,3,4.
    for (uint off = 4; off < 32; off <<= 1) {
        for (uint g = 0; g < G; g++) {
            float m2 = simd_shuffle_xor(m[g], off), l2 = simd_shuffle_xor(l[g], off);
            float mn = max(m[g], m2);
            float c1 = precise::exp(m[g] - mn), c2 = precise::exp(m2 - mn);
            for (uint i = 0; i < CH * 4; i++) acc[g][i] = acc[g][i] * c1 + simd_shuffle_xor(acc[g][i], off) * c2;
            l[g] = l[g] * c1 + l2 * c2;
            m[g] = mn;
        }
    }
    // Simdgroup states -> threadgroup memory: per simdgroup, per g: HD values then m, l.
    const uint stride = HD + 2;
    if (quad == 0) {
        for (uint g = 0; g < G; g++) {
            threadgroup float* dst = tgBuf + (sg * G + g) * stride;
            for (uint j = 0; j < CH; j++) for (uint e = 0; e < 4; e++) dst[j * 16 + qi * 4 + e] = acc[g][j * 4 + e];
            if (qi == 0) { dst[HD] = m[g]; dst[HD + 1] = l[g]; }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint idx = tid; idx < G * stride; idx += DEC_SIMDGROUPS * SIMD_WIDTH) {
        uint g = idx / stride, d = idx % stride;
        float M = NEG_BIG_PF;
        for (uint s2 = 0; s2 < DEC_SIMDGROUPS; s2++) M = max(M, tgBuf[(s2 * G + g) * stride + HD]);
        float out = 0.0f;
        if (d < HD) {
            for (uint s2 = 0; s2 < DEC_SIMDGROUPS; s2++) out += tgBuf[(s2 * G + g) * stride + d] * precise::exp(tgBuf[(s2 * G + g) * stride + HD] - M);
        } else if (d == HD) {
            out = M;
        } else {
            for (uint s2 = 0; s2 < DEC_SIMDGROUPS; s2++) out += tgBuf[(s2 * G + g) * stride + HD + 1] * precise::exp(tgBuf[(s2 * G + g) * stride + HD] - M);
        }
        partial[(((ulong)kvh * nSplits + split) * G + g) * stride + d] = out;
    }
}

#define ATTN_DECODE_SPLIT(NAME, HDV)                                                                           \
    kernel void NAME(device const float* q [[buffer(0)]], device const half* kc [[buffer(1)]],                   \
                     device const half* vc [[buffer(2)]], device float* partial [[buffer(3)]],                   \
                     constant uint& heads [[buffer(4)]], constant uint& kvHeads [[buffer(5)]],                   \
                     constant uint& L [[buffer(6)]], constant uint& splitLen [[buffer(7)]],                      \
                     constant uint& nSplits [[buffer(8)]],                                                       \
                     uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],         \
                     uint sg [[simdgroup_index_in_threadgroup]], uint tid [[thread_index_in_threadgroup]]) {     \
        threadgroup float tgBuf[DEC_SIMDGROUPS * DEC_MAX_G * (HDV + 2)];                                         \
        attn_decode_split_impl<HDV>(q, kc, vc, partial, heads, kvHeads, L, splitLen, nSplits, tg, lane, sg, tid, tgBuf); \
    }
ATTN_DECODE_SPLIT(attn_decode_split_hd64, 64)
ATTN_DECODE_SPLIT(attn_decode_split_hd128, 128)

// Combine the split partials of one query head. Grid: (heads), HD threads.
kernel void attn_decode_merge(device const float* partial [[buffer(0)]], device float* ctx [[buffer(1)]],
                              constant uint& heads [[buffer(2)]], constant uint& kvHeads [[buffer(3)]],
                              constant uint& headDim [[buffer(4)]], constant uint& nSplits [[buffer(5)]],
                              uint h [[threadgroup_position_in_grid]], uint d [[thread_index_in_threadgroup]]) {
    const uint G = heads / kvHeads, kvh = h / G, g = h % G, stride = headDim + 2;
    device const float* base = partial + ((ulong)kvh * nSplits * G + g) * stride;      // split s at + s * G * stride
    float M = NEG_BIG_PF;
    for (uint s = 0; s < nSplits; s++) M = max(M, base[(ulong)s * G * stride + headDim]);
    float num = 0.0f, den = 0.0f;
    for (uint s = 0; s < nSplits; s++) {
        float f = precise::exp(base[(ulong)s * G * stride + headDim] - M);
        num += base[(ulong)s * G * stride + d] * f;
        den += base[(ulong)s * G * stride + headDim + 1] * f;
    }
    ctx[(ulong)h * headDim + d] = num / den;
}
