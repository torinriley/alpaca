// Copyright (c) 2026 Torin Etheridge. MIT License (see LICENSE).

// Tensor-op GEMM that dequantises its weights on the fly. The original path expands a whole quantised matrix to half in device memory
// (2 bytes/weight written, then read again by the GEMM): on a 1B K-quant model that round trip was ~45% of prefill time. Here each
// threadgroup walks K in chunks of 64: it dequantises a 64-row x 64-element weight tile straight into threadgroup memory (dq8 from
// common.metal, which is prepended to this library), then runs the tensor-op multiply-accumulate on that tile. Nothing is written to device
// memory but the result.
//
// Grid and threadgroup shape match gemm_tensor_half_w_*: threadgroups = (ceil(T / TT), ceil(N / 64)), 128 threads (4 simdgroups).
// Requires K % 64 == 0. Weights: any format dq8 supports (q8_0, q4_0, q4_1, q5_0, q5_1, q2_k ... q6_k).
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

constant constexpr int DQ_TR = 64;     // weight rows per threadgroup
constant constexpr int DQ_KC = 64;     // K elements per chunk

// Dequantises rows [n0, n0 + 64) x elements [k0, k0 + 64) of W into tile[row][k] (half).
// Q4_K / Q6_K: a 64-element chunk is exactly two whole 32-element sub-blocks, so two threads per row fill 32 elements each with a single scale
// decode and vector loads. Every other format goes through dq8 in 8-element units.
template <int FMT>
static inline void dq_fill_tile(device const uchar* W, uint K, uint N, uint n0, uint k0, uint tid, threadgroup half* tile) {
    const ulong rowBytes = fmt_row_bytes<FMT>(K);
    if (FMT == FMT_Q4_K || FMT == FMT_Q6_K) {
        const uint r = tid >> 1, part = tid & 1;
        threadgroup half4* o = (threadgroup half4*)(tile + r * DQ_KC + part * 32);
        if (n0 + r >= N) { for (uint i = 0; i < 8; i++) o[i] = half4(0.0h); return; }
        device const uchar* p = W + (ulong)(n0 + r) * rowBytes + (ulong)(k0 >> 8) * (FMT == FMT_Q4_K ? 144 : 210);
        const uint off = k0 & 255;
        if (FMT == FMT_Q4_K) {
            const uint c = off >> 6;
            const float d = ld_half(p), dmin = ld_half(p + 2);
            const float2 smn = k4_scale_min(p + 4, 2 * c + part);
            const float dl = d * smn.x, ml = dmin * smn.y;
            device const packed_uchar4* q = (device const packed_uchar4*)(p + 16 + c * 32);
            const uint shift = part * 4;
            for (uint i = 0; i < 8; i++) o[i] = half4(dl * float4((uchar4(q[i]) >> shift) & 0x0F) - ml);
        } else {
            const uint h = off >> 7, hi = (off >> 6) & 1;
            const float d = ld_half(p + 208);
            device const char* sc = (device const char*)(p + 192) + h * 8 + hi * 4 + part * 2;
            const float s0 = d * float(sc[0]), s1 = d * float(sc[1]);
            device const packed_uchar4* ql = (device const packed_uchar4*)(p + h * 64 + part * 32);
            device const packed_uchar4* qh = (device const packed_uchar4*)(p + 128 + h * 32);
            const uint nshift = hi * 4, hshift = hi * 4 + part * 2;
            for (uint i = 0; i < 8; i++) {
                uchar4 lo = (uchar4(ql[i]) >> nshift) & 0x0F, hb = (uchar4(qh[i]) >> hshift) & 3;
                o[i] = half4((i < 4 ? s0 : s1) * (float4(lo | (hb << 4)) - 32.0f));
            }
        }
        return;
    }
    for (uint idx = tid; idx < DQ_TR * (DQ_KC / 8); idx += 128) {
        const uint r = idx / (DQ_KC / 8), u = idx % (DQ_KC / 8);
        float4 a = float4(0.0f), b = float4(0.0f);
        if (n0 + r < N) dq8<FMT>(W + (ulong)(n0 + r) * rowBytes, (k0 >> 3) + u, a, b);
        threadgroup half4* o = (threadgroup half4*)(tile + r * DQ_KC + u * 8);
        o[0] = half4(a); o[1] = half4(b);
    }
}

template <int FMT, int TT, bool ACC>
static inline void gemm_dq_body(device const uchar* W, device half* X, device float* Y, uint K, uint N, uint T, uint2 tg, uint tid,
                                threadgroup half* tile) {
    tensor<device half, dextents<int32_t, 2>, tensor_inline> A(X, dextents<int32_t, 2>(K, T));
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> B(tile, dextents<int32_t, 2>(DQ_KC, DQ_TR));
    constexpr auto desc = matmul2d_descriptor(TT, DQ_TR, DQ_KC, false, true, true, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<4>> op;
    const uint n0 = tg.y * DQ_TR, t0 = tg.x * TT;
    auto mA = A.template slice<DQ_KC, dynamic_extent>(0, t0);
    auto cC = op.template get_destination_cooperative_tensor<decltype(mA), decltype(B), float>();
    #pragma clang loop unroll(full)
    for (uint16_t i = 0; i < cC.get_capacity(); ++i) { if (cC.is_valid_element(i)) cC[i] = 0.0f; }
    for (uint k0 = 0; k0 < K; k0 += DQ_KC) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        dq_fill_tile<FMT>(W, K, N, n0, k0, tid, tile);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto mAk = A.template slice<DQ_KC, dynamic_extent>(k0, t0);
        op.run(mAk, B, cC);
    }
    #pragma clang loop unroll(full)
    for (uint16_t i = 0; i < cC.get_capacity(); ++i) {
        if (cC.is_valid_element(i)) {
            auto ids = cC.get_multidimensional_index(i);
            const uint col = n0 + uint(ids[0]), row = t0 + uint(ids[1]);
            if (col < N && row < T) { device float* yp = Y + (ulong)row * N + col; *yp = (ACC ? *yp : 0.0f) + cC[i]; }
        }
    }
}

// Y = silu(X . Wgate^T) * (X . Wup^T) as half, both weight tiles dequantised per chunk.
template <int FMT, int TT>
static inline void gateup_dq_body(device const uchar* Wg, device const uchar* Wu, device half* X, device half* Y, uint K, uint N, uint T,
                                  uint2 tg, uint tid, threadgroup half* tileG, threadgroup half* tileU) {
    tensor<device half, dextents<int32_t, 2>, tensor_inline> A(X, dextents<int32_t, 2>(K, T));
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> Bg(tileG, dextents<int32_t, 2>(DQ_KC, DQ_TR));
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> Bu(tileU, dextents<int32_t, 2>(DQ_KC, DQ_TR));
    constexpr auto desc = matmul2d_descriptor(TT, DQ_TR, DQ_KC, false, true, true, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<4>> op;
    const uint n0 = tg.y * DQ_TR, t0 = tg.x * TT;
    auto mA = A.template slice<DQ_KC, dynamic_extent>(0, t0);
    auto cG = op.template get_destination_cooperative_tensor<decltype(mA), decltype(Bg), float>();
    auto cU = op.template get_destination_cooperative_tensor<decltype(mA), decltype(Bu), float>();
    #pragma clang loop unroll(full)
    for (uint16_t i = 0; i < cG.get_capacity(); ++i) { if (cG.is_valid_element(i)) { cG[i] = 0.0f; cU[i] = 0.0f; } }
    for (uint k0 = 0; k0 < K; k0 += DQ_KC) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        dq_fill_tile<FMT>(Wg, K, N, n0, k0, tid, tileG);
        dq_fill_tile<FMT>(Wu, K, N, n0, k0, tid, tileU);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto mAk = A.template slice<DQ_KC, dynamic_extent>(k0, t0);
        op.run(mAk, Bg, cG);
        op.run(mAk, Bu, cU);
    }
    #pragma clang loop unroll(full)
    for (uint16_t i = 0; i < cG.get_capacity(); ++i) {
        if (cG.is_valid_element(i)) {
            const float g = cG[i];
            const float r = (g / (1.0f + precise::exp(-g))) * cU[i];
            auto ids = cG.get_multidimensional_index(i);
            const uint col = n0 + uint(ids[0]), row = t0 + uint(ids[1]);
            if (col < N && row < T) Y[(ulong)row * N + col] = half(r);
        }
    }
}

#define GEMM_DQ_KERNEL(NAME, FMT, TT, ACC)                                                                       \
    kernel void NAME(device const uchar* W [[buffer(0)]], device half* X [[buffer(1)]], device float* Y [[buffer(2)]], \
                     constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]], constant uint& T [[buffer(5)]],   \
                     uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {            \
        threadgroup half tile[DQ_TR * DQ_KC];                                                                    \
        gemm_dq_body<FMT, TT, ACC>(W, X, Y, K, N, T, tg, tid, tile);                                             \
    }
#define GATEUP_DQ_KERNEL(NAME, FMT, TT)                                                                          \
    kernel void NAME(device const uchar* Wg [[buffer(0)]], device const uchar* Wu [[buffer(1)]], device half* X [[buffer(2)]], \
                     device half* Y [[buffer(3)]], constant uint& K [[buffer(4)]], constant uint& N [[buffer(5)]],    \
                     constant uint& T [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],                       \
                     uint tid [[thread_index_in_threadgroup]]) {                                                      \
        threadgroup half tileG[DQ_TR * DQ_KC];                                                                   \
        threadgroup half tileU[DQ_TR * DQ_KC];                                                                   \
        gateup_dq_body<FMT, TT>(Wg, Wu, X, Y, K, N, T, tg, tid, tileG, tileU);                                   \
    }
#define DQ_FORMAT(SUFFIX, FMT)                                  \
    GEMM_DQ_KERNEL(gemm_dq_##SUFFIX##_t64, FMT, 64, false)      \
    GEMM_DQ_KERNEL(gemm_dq_##SUFFIX##_t128, FMT, 128, false)    \
    GEMM_DQ_KERNEL(gemm_dq_##SUFFIX##_acc_t64, FMT, 64, true)   \
    GEMM_DQ_KERNEL(gemm_dq_##SUFFIX##_acc_t128, FMT, 128, true) \
    GATEUP_DQ_KERNEL(gateup_dq_##SUFFIX##_t64, FMT, 64)         \
    GATEUP_DQ_KERNEL(gateup_dq_##SUFFIX##_t128, FMT, 128)
DQ_FORMAT(q8_0, 1)
DQ_FORMAT(q4_0, 2)
DQ_FORMAT(q4_1, 3)
DQ_FORMAT(q5_0, 4)
DQ_FORMAT(q5_1, 5)
DQ_FORMAT(q2_k, 6)
DQ_FORMAT(q3_k, 7)
DQ_FORMAT(q4_k, 8)
DQ_FORMAT(q5_k, 9)
DQ_FORMAT(q6_k, 10)
