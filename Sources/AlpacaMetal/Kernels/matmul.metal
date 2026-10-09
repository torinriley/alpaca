// Copyright (c) 2026 Torin Etheridge. MIT License (see LICENSE).
// Author: Torin Etheridge · 2026-10-09

#include "common.metal"

// Prefill GEMM  Y[t, n] = sum_k X[t, k] * W[n, k]  using 8x8 simdgroup matrix multiply-accumulate (float32).
//
// Threadgroup (128 threads = 4 simdgroups) computes a 32-token x 64-row output tile, stepping over K in chunks of
// 32 (one quantisation block). Each step cooperatively dequantises a 64x32 weight tile and loads a 32x32
// activation tile into threadgroup memory; each simdgroup then owns a 32x16 sub-tile (4x2 accumulators of 8x8).
// Weights are dequantised in registers on the way into threadgroup memory — the full matrix is never expanded.
// Edges (N % 64, T % 32) are zero-filled on load and masked on store. Requires K % 32 == 0.
// Grid: threadgroups = (ceil(N / 64), ceil(T / 32)).

constant constexpr uint MM_TOKENS = 32;
constant constexpr uint MM_ROWS = 64;
constant constexpr uint MM_K = 32;

// Loads 16 consecutive dequantised weights (this thread's slice of the 64x32 tile) into sw[row][...].
inline void load_w_f16(device const uchar* W, uint K, uint n, uint N, uint k0, uint half_, threadgroup float* dst) {
    if (n >= N) { for (uint j = 0; j < 16; j++) dst[j] = 0.0f; return; }
    device const half* p = (device const half*)(W + ((ulong)n * K + k0 + half_ * 16) * 2);
    for (uint j = 0; j < 16; j++) dst[j] = float(p[j]);
}

inline void load_w_q8_0(device const uchar* W, uint K, uint n, uint N, uint k0, uint half_, threadgroup float* dst) {
    if (n >= N) { for (uint j = 0; j < 16; j++) dst[j] = 0.0f; return; }
    device const uchar* blk = W + ((ulong)n * (K / Q_BLOCK) + k0 / Q_BLOCK) * Q8_BYTES;
    float d = float(*(device const half*)blk);
    device const char* q = (device const char*)(blk + 2 + half_ * 16);
    for (uint j = 0; j < 16; j++) dst[j] = d * float(q[j]);
}

// Q4_0: this thread's 16 outputs are elements [8h, 8h+8) (low nibbles) and [16+8h, 16+8h+8) (high nibbles) of the
// block, written to their natural positions, so dst here points at the row start rather than a 16-element slice.
inline void load_w_q4_0(device const uchar* W, uint K, uint n, uint N, uint k0, uint half_, threadgroup float* rowBase) {
    if (n >= N) { for (uint j = 0; j < 8; j++) { rowBase[half_ * 8 + j] = 0.0f; rowBase[16 + half_ * 8 + j] = 0.0f; } return; }
    device const uchar* blk = W + ((ulong)n * (K / Q_BLOCK) + k0 / Q_BLOCK) * Q4_BYTES;
    float d = float(*(device const half*)blk);
    device const uchar* q = blk + 2 + half_ * 8;
    for (uint j = 0; j < 8; j++) {
        rowBase[half_ * 8 + j] = d * float(int(q[j] & 0x0F) - 8);
        rowBase[16 + half_ * 8 + j] = d * float(int(q[j] >> 4) - 8);
    }
}

#define MM_KERNEL(FMT, LOAD_EXPR)                                                                                \
    kernel void mm_##FMT(device const uchar* W [[buffer(0)]], device const float* X [[buffer(1)]],                \
                         device float* Y [[buffer(2)]], constant uint& K [[buffer(3)]],                           \
                         constant uint& N [[buffer(4)]], constant uint& T [[buffer(5)]],                          \
                         constant uint& addToOutput [[buffer(6)]],                                                        \
                         uint3 tg [[threadgroup_position_in_grid]], uint tid3 [[thread_index_in_threadgroup]],    \
                         uint sg [[simdgroup_index_in_threadgroup]]) {                                            \
        threadgroup float sw[MM_ROWS * MM_K];     /* [row][k] */                                                  \
        threadgroup float sx[MM_TOKENS * MM_K];   /* [token][k] */                                               \
        const uint n0 = tg.x * MM_ROWS, t0 = tg.y * MM_TOKENS;                                                    \
        const uint tid = tid3;                                                                                    \
        const uint wrow = tid / 2, whalf = tid % 2;            /* weight loader: row and 16-element half */       \
        const uint xtok = tid / 4, xq = (tid % 4) * 8;         /* activation loader: token and 8-element slice */ \
        simdgroup_float8x8 acc[4][2];                                                                             \
        for (uint i = 0; i < 4; i++) for (uint j = 0; j < 2; j++) acc[i][j] = simdgroup_float8x8(0.0f);           \
        for (uint k0 = 0; k0 < K; k0 += MM_K) {                                                                   \
            LOAD_EXPR;                                                                                            \
            {                                                                                                     \
                uint t = t0 + xtok;                                                                               \
                for (uint j = 0; j < 8; j++) sx[xtok * MM_K + xq + j] = (t < T) ? X[(ulong)t * K + k0 + xq + j] : 0.0f; \
            }                                                                                                     \
            threadgroup_barrier(mem_flags::mem_threadgroup);                                                      \
            for (uint kk = 0; kk < MM_K; kk += 8) {                                                               \
                simdgroup_float8x8 a[4], b[2];                                                                    \
                for (uint i = 0; i < 4; i++) simdgroup_load(a[i], sx + (i * 8) * MM_K + kk, MM_K);                 \
                for (uint j = 0; j < 2; j++) simdgroup_load(b[j], sw + (sg * 16 + j * 8) * MM_K + kk, MM_K, ulong2(0, 0), true); \
                for (uint i = 0; i < 4; i++) for (uint j = 0; j < 2; j++) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]); \
            }                                                                                                     \
            threadgroup_barrier(mem_flags::mem_threadgroup);                                                      \
        }                                                                                                         \
        /* Stage the 32x64 result tile in threadgroup memory (reusing sw: 64*32 floats) and write with edge masks. */ \
        threadgroup float* out = sw;                                                                              \
        for (uint i = 0; i < 4; i++) for (uint j = 0; j < 2; j++)                                                 \
            simdgroup_store(acc[i][j], out + (i * 8) * MM_ROWS + sg * 16 + j * 8, MM_ROWS);                       \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                                          \
        for (uint idx = tid; idx < MM_TOKENS * MM_ROWS; idx += 128) {                                             \
            uint tt = idx / MM_ROWS, nn = idx % MM_ROWS;                                                          \
            if (t0 + tt < T && n0 + nn < N) { device float* yp = Y + (ulong)(t0 + tt) * N + n0 + nn; *yp = (addToOutput ? *yp : 0.0f) + out[idx]; }                        \
        }                                                                                                         \
    }

MM_KERNEL(f16,  load_w_f16(W, K, n0 + wrow, N, k0, whalf, sw + wrow * MM_K + whalf * 16))
MM_KERNEL(q8_0, load_w_q8_0(W, K, n0 + wrow, N, k0, whalf, sw + wrow * MM_K + whalf * 16))
MM_KERNEL(q4_0, load_w_q4_0(W, K, n0 + wrow, N, k0, whalf, sw + wrow * MM_K))
