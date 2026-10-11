// Copyright (c) 2026 Torin Etheridge. MIT License (see LICENSE).
// Author: Torin Etheridge · 2026-10-09

#include "common.metal"

// Tiled causal attention for prefill (FlashAttention-style, written for this project).
//
// One threadgroup = 4 simdgroups handles a 64-token query tile of one head; each simdgroup owns 16 query rows
// (two 8x8 MMA row-tiles). The KV cache is walked in blocks of 32 positions: the whole threadgroup cooperatively
// stages the block's K and V rows in threadgroup memory ONCE, then every simdgroup reads them from there (4x less
// cache traffic than per-simdgroup loads). Blocks entirely above the causal diagonal of a simdgroup's rows are skipped
// by that simdgroup (but it still joins the barriers). Per block, per simdgroup:
//   S = Q·Kᵀ        (8 MMA tiles of 8x8, K-dim = head dim, half operands, float32 accumulate)
//   online softmax  (lane = KV column; running row max m and denominator l; P written back as half)
//   O += P·V        (MMA)
// The output accumulator never leaves registers: when a row's running max grows, O is rescaled by multiplying with a
// diagonal matrix diag(corr) through an MMA (no threadgroup round trip); at the end O is scaled by diag(1/l) the same way
// and stored straight to device memory.
//
// Precision: Q, P (the exponentiated scores), K and V are half before the MMA, accumulation is float32.
// KV cache layout: [positions + 32 padding rows, kvHeads, HD] half; padding is zero so a block that straddles the end
// reads finite values, which the causal test masks. Grid: threadgroups = (ceil(tokens / 64), heads), 128 threads.
// HD in {64, 128}.

constant constexpr uint PF_BLOCK = 32;        // KV positions per block
constant constexpr uint PF_SIMDGROUPS = 4;
constant constexpr uint PF_ROWS_PER_SG = 16;  // two 8-row MMA tiles
constant constexpr uint PF_TILE = PF_SIMDGROUPS * PF_ROWS_PER_SG;   // query tokens per threadgroup

template <uint HD, typename OUT>
inline void attn_prefill_impl(device const float* q, device const half* kc, device const half* vc, device OUT* out,
                              uint heads, uint kvHeads, uint startPos, uint T,
                              uint2 tg, uint lane, uint sg, uint tid,
                              threadgroup half* sK, threadgroup half* sV,
                              threadgroup float* sS, threadgroup half* sP, threadgroup float* sD) {
    const uint h = tg.y;
    const uint kvh = h / (heads / kvHeads);
    const uint kvStride = kvHeads * HD;
    const uint tileRow0 = tg.x * PF_TILE;
    const uint row0 = tileRow0 + sg * PF_ROWS_PER_SG;                       // this simdgroup's first query token
    threadgroup float* S = sS + sg * (PF_ROWS_PER_SG * PF_BLOCK);
    threadgroup half* P = sP + sg * (PF_ROWS_PER_SG * PF_BLOCK);
    threadgroup float* D = sD + sg * 128;                                   // two 8x8 diagonal matrices (64 floats each)
    device const half* kbase = kc + kvh * HD;
    device const half* vbase = vc + kvh * HD;

    // Q rows -> half, staged per 8-row tile through this simdgroup's S region (unused until the first block), then into registers.
    simdgroup_half8x8 qf[2][HD / 8];
    for (uint i = 0; i < 2; i++) for (uint ks = 0; ks < HD / 8; ks++) qf[i][ks] = simdgroup_half8x8(0.0h);
    for (uint tile = 0; tile < 2; tile++) {
        threadgroup half* Qt = (threadgroup half*)S;                        // 8 * HD halves = 1 KB (HD=64) / 2 KB (HD=128) <= S (2 KB)
        for (uint i = lane; i < 8 * HD; i += SIMD_WIDTH) {
            uint r = i / HD, d = i % HD, t = row0 + tile * 8 + r;
            Qt[i] = t < T ? half(q[((ulong)t * heads + h) * HD + d]) : half(0.0h);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (uint ks = 0; ks < HD / 8; ks++) simdgroup_load(qf[tile][ks], Qt + ks * 8, HD);
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    simdgroup_float8x8 o[2][HD / 8];
    for (uint i = 0; i < 2; i++) for (uint d = 0; d < HD / 8; d++) o[i][d] = simdgroup_float8x8(0.0f);
    // Softmax state lives with the lane pair that owns the row: lane = 2*row + half.
    const uint myRow = lane >> 1, myHalf = lane & 1;
    const uint qpos = startPos + row0 + myRow;
    float mRow = NEG_BIG_PF, lRow = 0.0f;
    for (uint i = lane; i < 128; i += SIMD_WIDTH) D[i] = 0.0f;
    simdgroup_barrier(mem_flags::mem_threadgroup);

    const uint tgLastRow = min(tileRow0 + PF_TILE - 1, T - 1);
    const uint tgMaxPos = startPos + tgLastRow;                              // causal limit of the whole threadgroup
    const uint sgLastRow = min(row0 + PF_ROWS_PER_SG - 1, T - 1);
    const uint sgMaxPos = startPos + sgLastRow;
    const bool sgActive = row0 < T;
    const float scale = 1.0f / sqrt(float(HD));

    for (uint p0 = 0; p0 <= tgMaxPos; p0 += PF_BLOCK) {
        // Cooperative staging of K and V rows p0 ..< p0+32 (half4 loads).
        for (uint i = tid; i < PF_BLOCK * HD / 4; i += PF_SIMDGROUPS * SIMD_WIDTH) {
            uint r = (i * 4) / HD, c = (i * 4) % HD;
            *(threadgroup half4*)(sK + r * HD + c) = *(device const half4*)(kbase + (ulong)(p0 + r) * kvStride + c);
            *(threadgroup half4*)(sV + r * HD + c) = *(device const half4*)(vbase + (ulong)(p0 + r) * kvStride + c);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sgActive && p0 <= sgMaxPos) {
            bool rescale[2];
            simdgroup_float8x8 s[2][4];
            for (uint i = 0; i < 2; i++) for (uint c = 0; c < 4; c++) s[i][c] = simdgroup_float8x8(0.0f);
            for (uint ks = 0; ks < HD / 8; ks++) {
                for (uint c = 0; c < 4; c++) {
                    simdgroup_half8x8 kb;
                    simdgroup_load(kb, sK + (8 * c) * HD + ks * 8, HD, ulong2(0, 0), true);
                    simdgroup_multiply_accumulate(s[0][c], qf[0][ks], kb, s[0][c]);
                    simdgroup_multiply_accumulate(s[1][c], qf[1][ks], kb, s[1][c]);
                }
            }
            for (uint i = 0; i < 2; i++)
                for (uint c = 0; c < 4; c++) simdgroup_store(s[i][c], S + (i * 8) * PF_BLOCK + 8 * c, PF_BLOCK);
            simdgroup_barrier(mem_flags::mem_threadgroup);

            // Online softmax, two lanes per query row (16 KV columns each): no cross-lane reductions beyond one pair-exchange
            // for the row max and one for the row sum. Lane pair r owns row r's running (m, l).
            {
                threadgroup float* Srow = S + myRow * PF_BLOCK + myHalf * 16;
                float v[16];
                float mx = NEG_BIG_PF;
                const bool diagonal = (p0 + PF_BLOCK - 1) > (startPos + row0);   // some column of this block may lie above the diagonal
                for (uint j = 0; j < 16; j++) {
                    float x = Srow[j] * scale;
                    bool ok = !diagonal || (p0 + myHalf * 16 + j) <= qpos;
                    v[j] = ok ? x : NEG_BIG_PF;
                    mx = max(mx, v[j]);
                }
                mx = max(mx, simd_shuffle_xor(mx, 1));
                const float mn = max(mRow, mx);
                float sum = 0.0f;
                threadgroup half* Prow = P + myRow * PF_BLOCK + myHalf * 16;
                for (uint j = 0; j < 16; j++) {
                    bool ok = !diagonal || (p0 + myHalf * 16 + j) <= qpos;
                    float w = ok ? fast::exp(v[j] - mn) : 0.0f;
                    Prow[j] = half(w);
                    sum += w;
                }
                sum += simd_shuffle_xor(sum, 1);
                const float cr = fast::exp(mRow - mn);
                lRow = lRow * cr + sum;
                mRow = mn;
                if (myHalf == 0) D[(myRow / 8) * 64 + (myRow % 8) * 9] = cr;     // diag(corr) per 8-row tile
                const bool any = simd_any(cr != 1.0f);
                rescale[0] = any; rescale[1] = any;
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (uint i = 0; i < 2; i++) {
                if (rescale[i]) {
                    simdgroup_float8x8 dm;
                    simdgroup_load(dm, D + i * 64, 8);
                    for (uint d = 0; d < HD / 8; d++) {
                        simdgroup_float8x8 z = simdgroup_float8x8(0.0f);
                        simdgroup_multiply_accumulate(o[i][d], dm, o[i][d], z);   // o = diag(corr) · o
                    }
                }
            }

            for (uint c = 0; c < 4; c++) {
                simdgroup_half8x8 pf0, pf1;
                simdgroup_load(pf0, P + 8 * c, PF_BLOCK);
                simdgroup_load(pf1, P + 8 * PF_BLOCK + 8 * c, PF_BLOCK);
                for (uint d = 0; d < HD / 8; d++) {
                    simdgroup_half8x8 vb;
                    simdgroup_load(vb, sV + (8 * c) * HD + d * 8, HD);
                    simdgroup_multiply_accumulate(o[0][d], pf0, vb, o[0][d]);
                    simdgroup_multiply_accumulate(o[1][d], pf1, vb, o[1][d]);
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);                       // sK/sV are overwritten by the next block
    }

    if (!sgActive) return;
    // Normalise: o = diag(1/l) · o, then store (edge tiles go through S so rows >= T are never written).
    if (myHalf == 0) D[(myRow / 8) * 64 + (myRow % 8) * 9] = 1.0f / lRow;
    simdgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float* O = S;                                                   // reuse as an 8 x 8 staging tile
    for (uint i = 0; i < 2; i++) {
        simdgroup_float8x8 dm;
        simdgroup_load(dm, D + i * 64, 8);
        for (uint d = 0; d < HD / 8; d++) {
            simdgroup_float8x8 z = simdgroup_float8x8(0.0f), res;
            simdgroup_multiply_accumulate(res, dm, o[i][d], z);
            // Stage through an 8x8 threadgroup tile so rows >= T are never written and the output may be converted to OUT.
            simdgroup_store(res, O, 8);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (uint e = lane; e < 64; e += SIMD_WIDTH) {
                uint r = e / 8, c = e % 8, t = row0 + i * 8 + r;
                if (t < T) out[((ulong)t * heads + h) * HD + d * 8 + c] = OUT(O[e]);
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

#define ATTN_PREFILL_KERNEL(NAME, HDV, OUTT)                                                                    \
    kernel void NAME(device const float* q [[buffer(0)]], device const half* kc [[buffer(1)]],                  \
                     device const half* vc [[buffer(2)]], device OUTT* out [[buffer(3)]],                       \
                     constant uint& heads [[buffer(4)]], constant uint& kvHeads [[buffer(5)]],                  \
                     constant uint& startPos [[buffer(6)]], constant uint& T [[buffer(7)]],                     \
                     uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],        \
                     uint sg [[simdgroup_index_in_threadgroup]], uint tid [[thread_index_in_threadgroup]]) {     \
        threadgroup half sK[PF_BLOCK * HDV];                                                                    \
        threadgroup half sV[PF_BLOCK * HDV];                                                                    \
        threadgroup float sS[PF_SIMDGROUPS * PF_ROWS_PER_SG * PF_BLOCK];                                        \
        threadgroup half sP[PF_SIMDGROUPS * PF_ROWS_PER_SG * PF_BLOCK];                                         \
        threadgroup float sD[PF_SIMDGROUPS * 128];                                                              \
        attn_prefill_impl<HDV, OUTT>(q, kc, vc, out, heads, kvHeads, startPos, T, tg, lane, sg, tid, sK, sV, sS, sP, sD); \
    }
ATTN_PREFILL_KERNEL(attn_prefill_hd64_kv16, 64, float)
ATTN_PREFILL_KERNEL(attn_prefill_hd128_kv16, 128, float)
ATTN_PREFILL_KERNEL(attn_prefill_hd64_kv16_oh, 64, half)      // half output: feeds the tensor-op GEMM directly
ATTN_PREFILL_KERNEL(attn_prefill_hd128_kv16_oh, 128, half)
