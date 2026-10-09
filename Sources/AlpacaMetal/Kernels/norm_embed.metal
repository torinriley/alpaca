// Copyright (c) 2026 Torin Etheridge. MIT License (see LICENSE).
// Author: Torin Etheridge · 2026-10-09

#include "common.metal"

// RMSNorm: out[r, :] = x[r, :] * rsqrt(mean(x[r, :]^2) + eps) * w.  One threadgroup per row; float32 sum of squares.
kernel void rmsnorm_f32(device const float* x [[buffer(0)]], device const float* w [[buffer(1)]],
                        device float* out [[buffer(2)]], constant uint& dim [[buffer(3)]],
                        constant float& eps [[buffer(4)]],
                        uint row [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
                        uint tgSize [[threads_per_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                        uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partial[32];
    device const float* xr = x + (ulong)row * dim;
    float ss = 0.0f;
    for (uint i = tid; i < dim; i += tgSize) ss += xr[i] * xr[i];
    ss = simd_sum(ss);
    if (lane == 0) partial[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float v = lane < (tgSize / SIMD_WIDTH) ? partial[lane] : 0.0f;
        v = simd_sum(v);
        if (lane == 0) partial[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float scale = 1.0f / precise::sqrt(partial[0] / float(dim) + eps);
    device float* orow = out + (ulong)row * dim;
    for (uint i = tid; i < dim; i += tgSize) orow[i] = xr[i] * scale * w[i];
}

// Embedding lookup: out[t, e] = dequant(table[tokens[t], e]). Grid: (dim, tokens). Token ids are validated on the host.
kernel void embed_f16(device const int* tokens [[buffer(0)]], device const half* table [[buffer(1)]],
                      device float* out [[buffer(2)]], constant uint& dim [[buffer(3)]],
                      uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= dim) return;
    out[(ulong)gid.y * dim + gid.x] = float(table[(ulong)tokens[gid.y] * dim + gid.x]);
}

kernel void embed_q8_0(device const int* tokens [[buffer(0)]], device const uchar* table [[buffer(1)]],
                       device float* out [[buffer(2)]], constant uint& dim [[buffer(3)]],
                       uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= dim) return;
    uint blk = gid.x / Q_BLOCK, j = gid.x % Q_BLOCK;
    device const uchar* p = table + ((ulong)tokens[gid.y] * (dim / Q_BLOCK) + blk) * Q8_BYTES;
    float d = float(*(device const half*)p);
    out[(ulong)gid.y * dim + gid.x] = d * float(((device const char*)(p + 2))[j]);
}

kernel void embed_q4_0(device const int* tokens [[buffer(0)]], device const uchar* table [[buffer(1)]],
                       device float* out [[buffer(2)]], constant uint& dim [[buffer(3)]],
                       uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= dim) return;
    uint blk = gid.x / Q_BLOCK, j = gid.x % Q_BLOCK;
    device const uchar* p = table + ((ulong)tokens[gid.y] * (dim / Q_BLOCK) + blk) * Q4_BYTES;
    float d = float(*(device const half*)p);
    uchar q = p[2 + (j & 15)];
    int nib = (j < 16) ? (q & 0x0F) : (q >> 4);
    out[(ulong)gid.y * dim + gid.x] = d * float(nib - 8);
}

// Greedy sampling on the GPU: index of the largest logit (lowest index on ties; NaN never wins). The winner is written to
// `chainToken` (the next step's input) and to `emitted[slot]` (read by the CPU). One threadgroup of 1024 threads.
kernel void argmax_logits(device const float* logits [[buffer(0)]], constant uint& n [[buffer(1)]],
                          device int* chainToken [[buffer(2)]], device int* emitted [[buffer(3)]], constant uint& slot [[buffer(4)]],
                          uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                          uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float bestV[32];
    threadgroup uint bestI[32];
    float v = -INFINITY_F; uint idx = 0xFFFFFFFFu;
    for (uint i = tid; i < n; i += 1024) {
        float x = logits[i];
        if (x > v) { v = x; idx = i; }                     // strictly greater: the lowest index wins within a thread
    }
    // Simdgroup: max value, then the smallest index among lanes holding it.
    float sv = simd_max(v);
    uint si = simd_min(v == sv ? idx : 0xFFFFFFFFu);
    if (lane == 0) { bestV[sg] = sv; bestI[sg] = si; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float v2 = bestV[lane]; uint i2 = bestI[lane];
        float m = simd_max(v2);
        uint w = simd_min(v2 == m ? i2 : 0xFFFFFFFFu);
        if (lane == 0) {
            int token = (w == 0xFFFFFFFFu) ? 0 : int(w);
            chainToken[0] = token;
            emitted[slot] = token;
        }
    }
}
