// Copyright (c) 2026 Torin Etheridge. MIT License (see LICENSE).
// Author: Torin Etheridge · 2026-10-09

// Causal prefill attention on the GPU's matrix hardware (Metal 4 tensor ops; compiled in the tensor library, Apple10 GPUs).
//
// One threadgroup (2 simdgroups, 64 threads) handles a 32-query tile of one head and walks the KV cache in blocks of 64 positions:
//   S = Q·Kᵀ        tensor-op GEMM, Q and K read straight from device memory (K/V are strided views of the cache), S -> threadgroup
//   softmax         two lanes per query row on S in threadgroup memory: online max / denominator, writes P as half and the
//                   per-row rescale factor
//   O += P·V        tensor-op GEMM accumulating into a cooperative tensor that stays in registers for the whole loop;
//                   before each product O is rescaled by the row factors (cooperative-tensor element access by row index)
// No K/V staging: the tensor operation loads its operands itself, which is what made this 1.9x faster than the simdgroup-matrix
// kernel (0.86 vs 1.61 ms per layer at 2048 tokens, 9 heads, head dim 64). Q is supplied as half; P, K and V are half; products
// accumulate in float32 (relaxed-precision tensor ops). Edge tiles are handled by the tensor slices' bounds checks.
// Tile shape was chosen by sweeping (queries per tile, block length, simdgroups) at 4096 tokens: 32 x 64 x 2 simdgroups ran at 1.5 ms per
// layer-call against 3.1 ms for 64 x 64 x 4 (the smaller threadgroup lets more of them be resident per core).
// Grid: threadgroups = (ceil(tokens / 32), heads), 64 threads. Head dim 64 or 128.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

constant constexpr int AT_BQ = 32;     // query rows per threadgroup (2 simdgroups = 64 threads)
constant constexpr int AT_BK = 64;     // KV positions per block
constant constexpr float AT_NEG = -1.0e30f;

template <int HD, bool HALF_OUT, typename OUT>
inline void attn_tensor_impl(device half* qh, device half* kc, device half* vc, device OUT* out,
                             uint heads, uint kvHeads, uint startPos, uint T, uint2 tg, uint tid,
                             threadgroup float* sS, threadgroup half* sP, threadgroup float* sCorr) {
    const uint h = tg.y, kvh = h / (heads / kvHeads), kvStride = kvHeads * HD;
    const uint q0 = tg.x * AT_BQ;
    const int rowsLeft = int(T) - int(q0);
    const int positions = int(startPos + T);                         // cached positions visible to this call

    array<int32_t, 2> qStride = {1, int32_t(heads * HD)};
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tQ(qh + (ulong)q0 * heads * HD + h * HD, dextents<int32_t, 2>(HD, rowsLeft), qStride);
    array<int32_t, 2> kvStrides = {1, int32_t(kvStride)};
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tK(kc + kvh * HD, dextents<int32_t, 2>(HD, positions), kvStrides);
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tV(vc + kvh * HD, dextents<int32_t, 2>(HD, positions), kvStrides);
    tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline> tS(sS, dextents<int32_t, 2>(AT_BK, AT_BQ));
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tP(sP, dextents<int32_t, 2>(AT_BK, AT_BQ));

    constexpr auto descS = matmul2d_descriptor(AT_BQ, AT_BK, HD, false, true, true);
    matmul2d<descS, execution_simdgroups<2>> opS;
    constexpr auto descO = matmul2d_descriptor(AT_BQ, HD, AT_BK, false, false, true, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<descO, execution_simdgroups<2>> opO;
    auto cO = opO.template get_destination_cooperative_tensor<decltype(tP), decltype(tV), float>();
    #pragma clang loop unroll(full)
    for (uint16_t i = 0; i < cO.get_capacity(); ++i) if (cO.is_valid_element(i)) cO[i] = 0.0f;

    const uint myRow = tid >> 1, myHalf = tid & 1;                   // lane pair (2r, 2r+1) owns query row r: 32 columns each
    float mRow = AT_NEG, lRow = 0.0f;
    const uint maxPos = startPos + min(q0 + AT_BQ - 1, T - 1);       // causal limit of the last row of the tile
    const float scale = 1.0f / sqrt(float(HD));
    const uint qpos = startPos + q0 + myRow;

    for (uint p0 = 0; p0 <= maxPos; p0 += AT_BK) {
        auto mK = tK.slice(0, p0);
        auto mV = tV.slice(0, p0);
        opS.run(tQ, mK, tS);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
            threadgroup float* Srow = sS + myRow * AT_BK + myHalf * (AT_BK / 2);
            float v[AT_BK / 2];
            float mx = AT_NEG;
            for (uint j = 0; j < AT_BK / 2; j++) {
                bool ok = (p0 + myHalf * (AT_BK / 2) + j) <= qpos;
                v[j] = ok ? Srow[j] * scale : AT_NEG;
                mx = max(mx, v[j]);
            }
            mx = max(mx, simd_shuffle_xor(mx, 1));
            const float mn = max(mRow, mx);
            float sum = 0.0f;
            threadgroup half* Prow = sP + myRow * AT_BK + myHalf * (AT_BK / 2);
            for (uint j = 0; j < AT_BK / 2; j++) {
                bool ok = (p0 + myHalf * (AT_BK / 2) + j) <= qpos;
                float w = ok ? fast::exp(v[j] - mn) : 0.0f;
                Prow[j] = half(w);
                sum += w;
            }
            sum += simd_shuffle_xor(sum, 1);
            const float cr = fast::exp(mRow - mn);
            lRow = lRow * cr + sum;
            mRow = mn;
            if (myHalf == 0) sCorr[myRow] = cr;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma clang loop unroll(full)
        for (uint16_t i = 0; i < cO.get_capacity(); ++i) {
            if (cO.is_valid_element(i)) { auto ids = cO.get_multidimensional_index(i); cO[i] *= sCorr[ids[1]]; }
        }
        opO.run(tP, mV, cO);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (myHalf == 0) sCorr[myRow] = 1.0f / lRow;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    #pragma clang loop unroll(full)
    for (uint16_t i = 0; i < cO.get_capacity(); ++i) {
        if (cO.is_valid_element(i)) {
            auto ids = cO.get_multidimensional_index(i);
            const float r = cO[i] * sCorr[ids[1]];
            if constexpr (HALF_OUT) {                    // no float -> half cooperative store: write converted elements directly
                const uint col = uint(ids[0]), row = q0 + uint(ids[1]);
                if (row < T) out[(ulong)row * heads * HD + h * HD + col] = OUT(r);
            } else {
                cO[i] = r;
            }
        }
    }
    if constexpr (!HALF_OUT) {
        tensor<device OUT, dextents<int32_t, 2>, tensor_inline> tOut(out, dextents<int32_t, 2>(heads * HD, T));
        auto mOut = tOut.slice(h * HD, q0);
        cO.store(mOut);
    }
}

#define ATTN_TENSOR_KERNEL(NAME, HDV, HALFOUT, OUTT)                                                            \
    kernel void NAME(device half* qh [[buffer(0)]], device half* kc [[buffer(1)]], device half* vc [[buffer(2)]], \
                     device OUTT* out [[buffer(3)]], constant uint& heads [[buffer(4)]], constant uint& kvHeads [[buffer(5)]], \
                     constant uint& startPos [[buffer(6)]], constant uint& T [[buffer(7)]],                     \
                     uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {      \
        threadgroup float sS[AT_BQ * AT_BK];                                                                    \
        threadgroup half sP[AT_BQ * AT_BK];                                                                     \
        threadgroup float sCorr[AT_BQ];                                                                         \
        attn_tensor_impl<HDV, HALFOUT, OUTT>(qh, kc, vc, out, heads, kvHeads, startPos, T, tg, tid, sS, sP, sCorr); \
    }
ATTN_TENSOR_KERNEL(attn_tensor_hd64, 64, false, float)
ATTN_TENSOR_KERNEL(attn_tensor_hd128, 128, false, float)
ATTN_TENSOR_KERNEL(attn_tensor_hd64_oh, 64, true, half)
ATTN_TENSOR_KERNEL(attn_tensor_hd128_oh, 128, true, half)
