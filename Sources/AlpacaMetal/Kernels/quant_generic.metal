// Copyright (c) 2026 Torin Etheridge. MIT License (see LICENSE).

#include "common.metal"

// Kernels for the additional weight formats (Q4_1, Q5_0, Q5_1, Q2_K ... Q6_K, BF16), all built on dq8<FMT> from common.metal:
//   mv_<fmt>_tb1 / _tb8   row-per-simdgroup mat-vec, same signature and grid as matvec.metal
//   mm_<fmt>              32-token x 64-row simdgroup-matrix GEMM (exact float32 operands), same grid as matmul.metal
//   dequant_<fmt>_to_half expand one matrix to half for the tensor-op GEMM, grid (K / 8, N)
//   embed_<fmt>           embedding lookup, grid (dim, tokens)
// (decode_fused.metal and matmul_small.metal instantiate dq8 directly.)

template <int FMT, uint TB>
inline void mv_gen_impl(device const uchar* W, device const float* X, device float* Y, uint K, uint N, uint T, uint acc_,
                        uint3 tg, uint lane, uint sg) {
    uint n = tg.x * MV_SIMDGROUPS + sg;
    if (n >= N) return;
    uint t0 = tg.y * TB;
    device const uchar* row = W + (ulong)n * fmt_row_bytes<FMT>(K);
    float acc[TB];
    for (uint t = 0; t < TB; t++) acc[t] = 0.0f;
    for (uint u = lane; u < K / 8; u += SIMD_WIDTH) {
        float4 wa, wb;
        dq8<FMT>(row, u, wa, wb);
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

#define MV_GEN_KERNEL(NAME, FMT, TBV)                                                                          \
    kernel void NAME(device const uchar* W [[buffer(0)]], device const float* X [[buffer(1)]],                \
                     device float* Y [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]], \
                     constant uint& T [[buffer(5)]], constant uint& acc [[buffer(6)]],                         \
                     uint3 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],       \
                     uint sg [[simdgroup_index_in_threadgroup]]) {                                              \
        mv_gen_impl<FMT, TBV>(W, X, Y, K, N, T, acc, tg, lane, sg);                                             \
    }

// Loads 16 consecutive dequantised weights of row n (elements [k0 + 16*half_, +16)) into dst.
template <int FMT>
inline void load_w_gen(device const uchar* W, uint K, uint n, uint N, uint k0, uint half_, threadgroup float* dst) {
    if (n >= N) { for (uint j = 0; j < 16; j++) dst[j] = 0.0f; return; }
    device const uchar* row = W + (ulong)n * fmt_row_bytes<FMT>(K);
    const uint u = (k0 + half_ * 16) >> 3;
    float4 a, b;
    dq8<FMT>(row, u, a, b);
    dst[0] = a.x; dst[1] = a.y; dst[2] = a.z; dst[3] = a.w; dst[4] = b.x; dst[5] = b.y; dst[6] = b.z; dst[7] = b.w;
    dq8<FMT>(row, u + 1, a, b);
    dst[8] = a.x; dst[9] = a.y; dst[10] = a.z; dst[11] = a.w; dst[12] = b.x; dst[13] = b.y; dst[14] = b.z; dst[15] = b.w;
}

// Same body as matmul.metal's MM_KERNEL, instantiated for the generic loader.
#define MM_GEN_KERNEL(NAME, FMT)                                                                                  \
    kernel void NAME(device const uchar* W [[buffer(0)]], device const float* X [[buffer(1)]],                    \
                     device float* Y [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]], \
                     constant uint& T [[buffer(5)]], constant uint& addToOutput [[buffer(6)]],                    \
                     uint3 tg [[threadgroup_position_in_grid]], uint tid3 [[thread_index_in_threadgroup]],        \
                     uint sg [[simdgroup_index_in_threadgroup]]) {                                                \
        threadgroup float sw[MM_ROWS * MM_K];                                                                     \
        threadgroup float sx[MM_TOKENS * MM_K];                                                                   \
        const uint n0 = tg.x * MM_ROWS, t0 = tg.y * MM_TOKENS;                                                    \
        const uint tid = tid3;                                                                                    \
        const uint wrow = tid / 2, whalf = tid % 2;                                                               \
        const uint xtok = tid / 4, xq = (tid % 4) * 8;                                                            \
        simdgroup_float8x8 acc[4][2];                                                                             \
        for (uint i = 0; i < 4; i++) for (uint j = 0; j < 2; j++) acc[i][j] = simdgroup_float8x8(0.0f);           \
        for (uint k0 = 0; k0 < K; k0 += MM_K) {                                                                   \
            load_w_gen<FMT>(W, K, n0 + wrow, N, k0, whalf, sw + wrow * MM_K + whalf * 16);                        \
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
        threadgroup float* out = sw;                                                                              \
        for (uint i = 0; i < 4; i++) for (uint j = 0; j < 2; j++)                                                 \
            simdgroup_store(acc[i][j], out + (i * 8) * MM_ROWS + sg * 16 + j * 8, MM_ROWS);                       \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                                          \
        for (uint idx = tid; idx < MM_TOKENS * MM_ROWS; idx += 128) {                                             \
            uint tt = idx / MM_ROWS, nn = idx % MM_ROWS;                                                          \
            if (t0 + tt < T && n0 + nn < N) { device float* yp = Y + (ulong)(t0 + tt) * N + n0 + nn; *yp = (addToOutput ? *yp : 0.0f) + out[idx]; } \
        }                                                                                                         \
    }

template <int FMT>
inline void dequant_gen_impl(device const uchar* W, device half* out, uint K, uint2 gid) {
    if (gid.x >= K / 8) return;
    float4 a, b;
    dq8<FMT>(W + (ulong)gid.y * fmt_row_bytes<FMT>(K), gid.x, a, b);
    device half4* o = (device half4*)(out + (ulong)gid.y * K + gid.x * 8);
    o[0] = half4(a); o[1] = half4(b);
}
#define DEQUANT_GEN_KERNEL(NAME, FMT)                                                                             \
    kernel void NAME(device const uchar* W [[buffer(0)]], device half* out [[buffer(1)]],                        \
                     constant uint& K [[buffer(2)]], uint2 gid [[thread_position_in_grid]]) { dequant_gen_impl<FMT>(W, out, K, gid); }

template <int FMT>
inline void embed_gen_impl(device const int* tokens, device const uchar* table, device float* out, uint dim, uint2 gid) {
    if (gid.x >= dim) return;
    float4 a, b;
    dq8<FMT>(table + (ulong)tokens[gid.y] * fmt_row_bytes<FMT>(dim), gid.x >> 3, a, b);
    uint i = gid.x & 7;
    out[(ulong)gid.y * dim + gid.x] = i < 4 ? a[i] : b[i - 4];
}
#define EMBED_GEN_KERNEL(NAME, FMT)                                                                               \
    kernel void NAME(device const int* tokens [[buffer(0)]], device const uchar* table [[buffer(1)]],            \
                     device float* out [[buffer(2)]], constant uint& dim [[buffer(3)]], uint2 gid [[thread_position_in_grid]]) { \
        embed_gen_impl<FMT>(tokens, table, out, dim, gid);                                                        \
    }

#define GEN_FORMAT(NAME, FMT)                       \
    MV_GEN_KERNEL(mv_##NAME##_tb1, FMT, 1)          \
    MV_GEN_KERNEL(mv_##NAME##_tb8, FMT, 8)          \
    MM_GEN_KERNEL(mm_##NAME, FMT)                   \
    DEQUANT_GEN_KERNEL(dequant_##NAME##_to_half, FMT) \
    EMBED_GEN_KERNEL(embed_##NAME, FMT)

GEN_FORMAT(q4_1, FMT_Q4_1)
GEN_FORMAT(q5_0, FMT_Q5_0)
GEN_FORMAT(q5_1, FMT_Q5_1)
GEN_FORMAT(q2_k, FMT_Q2_K)
GEN_FORMAT(q3_k, FMT_Q3_K)
GEN_FORMAT(q4_k, FMT_Q4_K)
GEN_FORMAT(q5_k, FMT_Q5_K)
GEN_FORMAT(q6_k, FMT_Q6_K)
GEN_FORMAT(bf16, FMT_BF16)
