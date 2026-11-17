// Copyright (c) 2026 Torin Etheridge. MIT License (see LICENSE).

#include "common.metal"

// Small-batch linear layer  Y[t, n] (+)= sum_k X[t, k] * W[n, k]  for 2...16 tokens (mm8_*: up to 8, mm16_*: up to 16) (speculative verification, short prompts).
//
// The row-per-simdgroup mat-vec kernels re-do every weight FMA once per token on the vector ALUs, which makes an 8-token pass ~3x
// slower than one token even though it reads the weights once. Here one simdgroup owns 8 output rows and multiplies an
// 8-token x 8-k activation tile by an 8-k x 8-row weight tile on the matrix units (float32 operands and accumulators, so the
// arithmetic matches the mat-vec kernels up to summation order).
//
// Per K step of 32 (one quantisation block): each lane dequantises 8 consecutive weights of one row into a float tile in
// threadgroup memory (4 lanes per row, 8 rows) and copies 8 activations of one token into a second tile (rows t >= T are zero-filled,
// so nothing past the end of the activation buffer is ever read), then 4 matrix multiply-accumulates consume the tiles.
// The SM_SG simdgroups of a threadgroup share the same 8 rows and split K between them (interleaved blocks), then add their partial
// tiles through threadgroup memory: small-N projections (N = 576 for a 135M model) would otherwise occupy only a few simdgroups.
// Grid: threadgroups = N / 8, 32 * SM_SG threads each. Requires K % 32 == 0 and N % 8 == 0.

constant constexpr uint SM_SG = 4;
constant constexpr uint SM_STRIDE = 36;      // tile row stride in floats (32 + 4 padding against bank conflicts)

inline void sm_load_f16(device const uchar* row, uint b, uint j, threadgroup float* dst) {
    device const half4* p = (device const half4*)(row + ((ulong)b * 32 + j * 8) * 2);
    half4 a = p[0], c = p[1];
    dst[0] = a.x; dst[1] = a.y; dst[2] = a.z; dst[3] = a.w; dst[4] = c.x; dst[5] = c.y; dst[6] = c.z; dst[7] = c.w;
}
inline void sm_load_q8_0(device const uchar* row, uint b, uint j, threadgroup float* dst) {
    device const uchar* p = row + (ulong)b * Q8_BYTES;
    float d = float(*(device const half*)p);
    device const packed_char4* q = (device const packed_char4*)(p + 2 + j * 8);
    float4 a = float4(q[0]) * d, c = float4(q[1]) * d;
    dst[0] = a.x; dst[1] = a.y; dst[2] = a.z; dst[3] = a.w; dst[4] = c.x; dst[5] = c.y; dst[6] = c.z; dst[7] = c.w;
}
// Quarter j of a q4_0 block: elements [8j, 8j+8): low nibbles of bytes 8j.. for j < 2, high nibbles of bytes 8(j-2).. for j >= 2.
inline void sm_load_q4_0(device const uchar* row, uint b, uint j, threadgroup float* dst) {
    device const uchar* p = row + (ulong)b * Q4_BYTES;
    float d = float(*(device const half*)p);
    device const uchar* q = p + 2 + (j & 1) * 8;
    const bool hi = j >= 2;
    for (uint e = 0; e < 8; e++) dst[e] = d * float(int(hi ? (q[e] >> 4) : (q[e] & 0x0F)) - 8);
}

// Every other format: this lane's 8-element unit through dq8.
template <int FMT>
inline void sm_load_gen(device const uchar* row, uint b, uint j, threadgroup float* dst) {
    float4 lo, hi;
    dq8<FMT>(row, b * 4 + j, lo, hi);
    dst[0] = lo.x; dst[1] = lo.y; dst[2] = lo.z; dst[3] = lo.w; dst[4] = hi.x; dst[5] = hi.y; dst[6] = hi.z; dst[7] = hi.w;
}

template <int FMT, uint NT>
inline void mm_small_impl(device const uchar* W, device const float* X, device float* Y, uint K, uint N, uint T, uint acc,
                          uint tg, uint lane, uint sg, uint tid, threadgroup float* tiles, threadgroup float* outTile) {
    const uint n0 = tg * 8;
    const uint r = lane >> 2, j = lane & 3;                                      // row within the tile, quarter of the block
    const ulong rowBytes = FMT == 0 ? K * 2 : (FMT == 1 ? (K / Q_BLOCK) * Q8_BYTES : (FMT == 2 ? (K / Q_BLOCK) * Q4_BYTES : fmt_row_bytes<FMT>(K)));
    device const uchar* row = W + (ulong)(n0 + r) * rowBytes;
    threadgroup float* tile = tiles + sg * (8 + 8 * NT) * SM_STRIDE;
    threadgroup float* xt = tile + 8 * SM_STRIDE;
    simdgroup_float8x8 C[NT];
    for (uint tt = 0; tt < NT; tt++) C[tt] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint b = sg; b < K / 32; b += SM_SG) {
        threadgroup float* dst = tile + r * SM_STRIDE + j * 8;
        if (FMT == 0) sm_load_f16(row, b, j, dst);
        else if (FMT == 1) sm_load_q8_0(row, b, j, dst);
        else if (FMT == 2) sm_load_q4_0(row, b, j, dst);
        else sm_load_gen<FMT>(row, b, j, dst);
        for (uint tt = 0; tt < NT; tt++) {
            threadgroup float* xd = xt + (tt * 8 + r) * SM_STRIDE + j * 8;
            const uint tok = tt * 8 + r;
            if (tok < T) {
                device const float4* xs = (device const float4*)(X + (ulong)tok * K + b * 32 + j * 8);
                float4 u = xs[0], v = xs[1];
                xd[0] = u.x; xd[1] = u.y; xd[2] = u.z; xd[3] = u.w; xd[4] = v.x; xd[5] = v.y; xd[6] = v.z; xd[7] = v.w;
            } else {
                for (uint e = 0; e < 8; e++) xd[e] = 0.0f;
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < 4; kk++) {
            simdgroup_float8x8 B;
            simdgroup_load(B, tile + kk * 8, SM_STRIDE, ulong2(0, 0), true);     // B[k][row] = tile[row][k]
            for (uint tt = 0; tt < NT; tt++) {
                simdgroup_float8x8 A;
                simdgroup_load(A, xt + tt * 8 * SM_STRIDE + kk * 8, SM_STRIDE);  // A[t][k]
                simdgroup_multiply_accumulate(C[tt], A, B, C[tt]);
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint tt = 0; tt < NT; tt++) simdgroup_store(C[tt], outTile + (sg * NT + tt) * 64, 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint idx = tid; idx < NT * 64; idx += 32 * SM_SG) {
        const uint tt = idx >> 6, i = idx & 63;
        float v = 0.0f;
        for (uint s2 = 0; s2 < SM_SG; s2++) v += outTile[(s2 * NT + tt) * 64 + i];
        const uint t = tt * 8 + (i >> 3), c = i & 7;
        if (t < T) { device float* yp = Y + (ulong)t * N + n0 + c; *yp = (acc ? *yp : 0.0f) + v; }
    }
}

#define MM_SMALL_KERNEL(NAME, FMT, NTV)                                                                         \
    kernel void NAME(device const uchar* W [[buffer(0)]], device const float* X [[buffer(1)]],                 \
                     device float* Y [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]], \
                     constant uint& T [[buffer(5)]], constant uint& acc [[buffer(6)]],                         \
                     uint tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],        \
                     uint sg [[simdgroup_index_in_threadgroup]], uint tid [[thread_index_in_threadgroup]]) {   \
        threadgroup float tiles[SM_SG * (8 + 8 * NTV) * SM_STRIDE];                                             \
        threadgroup float outTile[SM_SG * NTV * 64];                                                            \
        mm_small_impl<FMT, NTV>(W, X, Y, K, N, T, acc, tg, lane, sg, tid, tiles, outTile);                      \
    }
#define MM_SMALL_FORMAT(SUFFIX, FMT)                 \
    MM_SMALL_KERNEL(mm8_##SUFFIX, FMT, 1)            \
    MM_SMALL_KERNEL(mm16_##SUFFIX, FMT, 2)
MM_SMALL_FORMAT(f16, 0)
MM_SMALL_FORMAT(q8_0, 1)
MM_SMALL_FORMAT(q4_0, 2)
MM_SMALL_FORMAT(q4_1, 3)
MM_SMALL_FORMAT(q5_0, 4)
MM_SMALL_FORMAT(q5_1, 5)
MM_SMALL_FORMAT(q2_k, 6)
MM_SMALL_FORMAT(q3_k, 7)
MM_SMALL_FORMAT(q4_k, 8)
MM_SMALL_FORMAT(q5_k, 9)
MM_SMALL_FORMAT(q6_k, 10)
MM_SMALL_FORMAT(bf16, 11)
