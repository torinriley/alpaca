// Copyright (c) 2026 Torin Etheridge. MIT License (see LICENSE).
// Author: Torin Etheridge · 2026-10-09

// Metal 4 tensor-op GEMM for prefill (compiled as its own library with language version 4.0; Apple10-family GPUs).
//
//   Y[t, n] = sum_k X[t, k] * W[n, k]      X, Y float32;  W half  [N, K] row-major
//
// Uses MetalPerformancePrimitives `matmul2d` with relaxed precision, which runs on the GPU's matrix hardware with
// half-precision operands and float32 accumulation (activations are rounded to half inside the operation). Quantised
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

kernel void gemm_tensor_half_w(device half* W [[buffer(0)]], device float* X [[buffer(1)]], device float* Y [[buffer(2)]],
                               constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]], constant uint& T [[buffer(5)]],
                               uint2 tg [[threadgroup_position_in_grid]]) {
    tensor<device float, dextents<int32_t, 2>, tensor_inline> A(X, dextents<int32_t, 2>(K, T));
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
kernel void gemm_tensor_half_w_acc(device half* W [[buffer(0)]], device float* X [[buffer(1)]], device float* Y [[buffer(2)]],
                                   constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]], constant uint& T [[buffer(5)]],
                                   uint2 tg [[threadgroup_position_in_grid]]) {
    tensor<device float, dextents<int32_t, 2>, tensor_inline> A(X, dextents<int32_t, 2>(K, T));
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
kernel void gemm_tensor_gateup_silu(device half* Wg [[buffer(0)]], device half* Wu [[buffer(1)]], device float* X [[buffer(2)]],
                                    device float* Y [[buffer(3)]], constant uint& K [[buffer(4)]], constant uint& N [[buffer(5)]],
                                    constant uint& T [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]]) {
    tensor<device float, dextents<int32_t, 2>, tensor_inline> A(X, dextents<int32_t, 2>(K, T));
    tensor<device half, dextents<int32_t, 2>, tensor_inline> Bg(Wg, dextents<int32_t, 2>(K, N));
    tensor<device half, dextents<int32_t, 2>, tensor_inline> Bu(Wu, dextents<int32_t, 2>(K, N));
    tensor<device float, dextents<int32_t, 2>, tensor_inline> C(Y, dextents<int32_t, 2>(N, T));
    constexpr auto desc = matmul2d_descriptor(TG_TOKENS, TG_ROWS, static_cast<int>(dynamic_extent), false, true, true);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto mA = A.slice(0, tg.y * TG_TOKENS);
    auto mBg = Bg.slice(0, tg.x * TG_ROWS);
    auto mBu = Bu.slice(0, tg.x * TG_ROWS);
    auto mC = C.slice(tg.x * TG_ROWS, tg.y * TG_TOKENS);
    auto cG = op.template get_destination_cooperative_tensor<decltype(mA), decltype(mBg), float>();
    auto cU = op.template get_destination_cooperative_tensor<decltype(mA), decltype(mBu), float>();
    op.run(mA, mBg, cG);
    op.run(mA, mBu, cU);
    #pragma clang loop unroll(full)
    for (uint16_t i = 0; i < cG.get_capacity(); ++i) {
        if (cG.is_valid_element(i)) {
            float g = cG[i];
            cG[i] = (g / (1.0f + precise::exp(-g))) * cU[i];
        }
    }
    cG.store(mC);
}

// Expand quantised weight rows to half. One thread per 32-element block. Grid: (K / 32, N).
kernel void dequant_q8_0_to_half(device const uchar* W [[buffer(0)]], device half* out [[buffer(1)]],
                                 constant uint& K [[buffer(2)]], uint2 gid [[thread_position_in_grid]]) {
    const uint nb = K / 32;
    if (gid.x >= nb) return;
    device const uchar* blk = W + ((ulong)gid.y * nb + gid.x) * 34;
    const float d = float(*(device const half*)blk);
    device const char* q = (device const char*)(blk + 2);
    device half* o = out + (ulong)gid.y * K + gid.x * 32;
    for (uint j = 0; j < 32; j++) o[j] = half(d * float(q[j]));
}

kernel void dequant_q4_0_to_half(device const uchar* W [[buffer(0)]], device half* out [[buffer(1)]],
                                 constant uint& K [[buffer(2)]], uint2 gid [[thread_position_in_grid]]) {
    const uint nb = K / 32;
    if (gid.x >= nb) return;
    device const uchar* blk = W + ((ulong)gid.y * nb + gid.x) * 18;
    const float d = float(*(device const half*)blk);
    device const uchar* q = blk + 2;
    device half* o = out + (ulong)gid.y * K + gid.x * 32;
    for (uint j = 0; j < 16; j++) {
        o[j] = half(d * float(int(q[j] & 0x0F) - 8));
        o[j + 16] = half(d * float(int(q[j] >> 4) - 8));
    }
}
