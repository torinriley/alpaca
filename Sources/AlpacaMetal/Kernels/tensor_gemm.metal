// Copyright (c) 2026 Torin Etheridge. MIT License (see LICENSE).
// Author: Torin Etheridge · 2026-10-09

// Metal 4 tensor-op GEMM for prefill (compiled as its own library with language version 4.0; Apple10-family GPUs).
//
//   Y[t, n] = sum_k X[t, k] * W[n, k]      X half [T, K], W half [N, K] row-major, Y float32 (the SwiGLU kernel writes half)
//
// Uses MetalPerformancePrimitives `matmul2d` with relaxed precision, which runs on the GPU's matrix hardware with
// half-precision operands and float32 accumulation (activations arrive as half, so no conversion happens inside the operation and A traffic is halved: measured 1.2-2.4x faster than float32 activations at identical results). Quantised
// weights are first expanded one matrix at a time into a reusable half scratch buffer by the `dequant_*` kernels
// below, so the GEMM itself only ever sees half weights; the scratch is a single matrix, never the whole model.
// Dequantisation to half rounds d*q to 11 significant bits (relative 2^-11), the dominant extra error of this path.
//
// Grid: threadgroups = (ceil(N / 64), ceil(T / 64)), 128 threads (4 simdgroups). Edge tiles are handled by the
// tensor slices' built-in bounds checks.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

constant constexpr int TG_TOKENS = 64;
constant constexpr int TG_ROWS = 64;

kernel void gemm_tensor_half_w(device half* W [[buffer(0)]], device half* X [[buffer(1)]], device float* Y [[buffer(2)]],
                               constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]], constant uint& T [[buffer(5)]],
                               uint2 tg [[threadgroup_position_in_grid]]) {
    tensor<device half, dextents<int32_t, 2>, tensor_inline> A(X, dextents<int32_t, 2>(K, T));
    tensor<device half, dextents<int32_t, 2>, tensor_inline> B(W, dextents<int32_t, 2>(K, N));
    tensor<device float, dextents<int32_t, 2>, tensor_inline> C(Y, dextents<int32_t, 2>(N, T));
    constexpr auto desc = matmul2d_descriptor(TG_TOKENS, TG_ROWS, static_cast<int>(dynamic_extent), false, true, true);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto mA = A.slice(0, tg.y * TG_TOKENS);
    auto mB = B.slice(0, tg.x * TG_ROWS);
    auto mC = C.slice(tg.x * TG_ROWS, tg.y * TG_TOKENS);
    op.run(mA, mB, mC);
}

// Same product accumulated into an existing Y (Y += X·Wᵀ): fuses the residual connection into the projection,
// removing a separate add kernel and the temporary output buffer.
kernel void gemm_tensor_half_w_acc(device half* W [[buffer(0)]], device half* X [[buffer(1)]], device float* Y [[buffer(2)]],
                                   constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]], constant uint& T [[buffer(5)]],
                                   uint2 tg [[threadgroup_position_in_grid]]) {
    tensor<device half, dextents<int32_t, 2>, tensor_inline> A(X, dextents<int32_t, 2>(K, T));
    tensor<device half, dextents<int32_t, 2>, tensor_inline> B(W, dextents<int32_t, 2>(K, N));
    tensor<device float, dextents<int32_t, 2>, tensor_inline> C(Y, dextents<int32_t, 2>(N, T));
    constexpr auto desc = matmul2d_descriptor(TG_TOKENS, TG_ROWS, static_cast<int>(dynamic_extent), false, true, true,
                                              matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto mA = A.slice(0, tg.y * TG_TOKENS);
    auto mB = B.slice(0, tg.x * TG_ROWS);
    auto mC = C.slice(tg.x * TG_ROWS, tg.y * TG_TOKENS);
    op.run(mA, mB, mC);
}

// SwiGLU feed-forward input in one pass:  Y = silu(X·Wgateᵀ) ⊙ (X·Wupᵀ).
// Both products of a tile stay in registers (two cooperative tensors with identical layout), so neither the gate nor the up
// activations are ever written to memory and no separate silu·mul pass runs.
kernel void gemm_tensor_gateup_silu(device half* Wg [[buffer(0)]], device half* Wu [[buffer(1)]], device half* X [[buffer(2)]],
                                    device half* Y [[buffer(3)]], constant uint& K [[buffer(4)]], constant uint& N [[buffer(5)]],
                                    constant uint& T [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]]) {
    tensor<device half, dextents<int32_t, 2>, tensor_inline> A(X, dextents<int32_t, 2>(K, T));
    tensor<device half, dextents<int32_t, 2>, tensor_inline> Bg(Wg, dextents<int32_t, 2>(K, N));
    tensor<device half, dextents<int32_t, 2>, tensor_inline> Bu(Wu, dextents<int32_t, 2>(K, N));
    constexpr auto desc = matmul2d_descriptor(TG_TOKENS, TG_ROWS, static_cast<int>(dynamic_extent), false, true, true);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto mA = A.slice(0, tg.y * TG_TOKENS);
    auto mBg = Bg.slice(0, tg.x * TG_ROWS);
    auto mBu = Bu.slice(0, tg.x * TG_ROWS);
    auto cG = op.template get_destination_cooperative_tensor<decltype(mA), decltype(mBg), float>();
    auto cU = op.template get_destination_cooperative_tensor<decltype(mA), decltype(mBu), float>();
    op.run(mA, mBg, cG);
    op.run(mA, mBu, cU);
    #pragma clang loop unroll(full)
    for (uint16_t i = 0; i < cG.get_capacity(); ++i) {
        if (cG.is_valid_element(i)) {
            float g = cG[i];
            float r = (g / (1.0f + precise::exp(-g))) * cU[i];
            // Local (column, row) of this element inside the 64 x 64 output tile; the cooperative tensor has no
            // float -> half store, so write converted elements directly (bounds-checked for edge tiles).
            auto ids = cG.get_multidimensional_index(i);
            uint col = tg.x * TG_ROWS + uint(ids[0]), row = tg.y * TG_TOKENS + uint(ids[1]);
            if (col < N && row < T) Y[(ulong)row * N + col] = half(r);
        }
    }
}

// Expand quantised weight rows to half, 4 threads per 32-element block with vector loads/stores (196 GB/s measured vs 73 GB/s for
// one thread per block). Grid: (K / 32 * 4, N).
kernel void dequant_q8_0_to_half(device const uchar* W [[buffer(0)]], device half* out [[buffer(1)]],
                                 constant uint& K [[buffer(2)]], uint2 gid [[thread_position_in_grid]]) {
    const uint nb = K / 32, b = gid.x >> 2, part = gid.x & 3;       // part: elements [8*part, 8*part + 8) of the block
    if (b >= nb) return;
    device const uchar* blk = W + ((ulong)gid.y * nb + b) * 34;
    const float d = float(*(device const half*)blk);
    device const packed_char4* q = (device const packed_char4*)(blk + 2 + part * 8);
    device half4* o = (device half4*)(out + (ulong)gid.y * K + b * 32 + part * 8);
    o[0] = half4(d * float4(q[0]));
    o[1] = half4(d * float4(q[1]));
}

kernel void dequant_q4_0_to_half(device const uchar* W [[buffer(0)]], device half* out [[buffer(1)]],
                                 constant uint& K [[buffer(2)]], uint2 gid [[thread_position_in_grid]]) {
    const uint nb = K / 32, b = gid.x >> 2, part = gid.x & 3;       // part: bytes [4*part, 4*part + 4): low nibbles -> elements 4p.., high -> 16 + 4p..
    if (b >= nb) return;
    device const uchar* blk = W + ((ulong)gid.y * nb + b) * 18;
    const float d = float(*(device const half*)blk);
    device const packed_uchar4* qs = (device const packed_uchar4*)(blk + 2 + part * 4);
    const uchar4 q = uchar4(*qs);
    device half* base = out + (ulong)gid.y * K + b * 32;
    *(device half4*)(base + part * 4) = half4(d * (float4(q & 0x0F) - 8.0f));
    *(device half4*)(base + 16 + part * 4) = half4(d * (float4(q >> 4) - 8.0f));
}
