// Copyright (c) 2026 Torin Etheridge. MIT License (see LICENSE).
// Author: Torin Etheridge · 2026-10-09

#include "common.metal"

// out[i] = a[i] + b[i]            (a and out may alias: residual add)
kernel void add_f32(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]],
                    device float* out [[buffer(2)]], constant uint& n [[buffer(3)]],
                    uint i [[thread_position_in_grid]]) {
    if (i < n) out[i] = a[i] + b[i];
}

// out[i] = a[i] * b[i]
kernel void mul_f32(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]],
                    device float* out [[buffer(2)]], constant uint& n [[buffer(3)]],
                    uint i [[thread_position_in_grid]]) {
    if (i < n) out[i] = a[i] * b[i];
}

// out[i] = silu(x[i]) = x / (1 + exp(-x))
kernel void silu_f32(device const float* x [[buffer(0)]], device float* out [[buffer(1)]],
                     constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < n) { float v = x[i]; out[i] = v / (1.0f + precise::exp(-v)); }
}

// out[i] = silu(gate[i]) * up[i]   (SwiGLU gate; fuses silu and mul to save one pass over memory)
kernel void silu_mul_f32(device const float* gate [[buffer(0)]], device const float* up [[buffer(1)]],
                         device float* out [[buffer(2)]], constant uint& n [[buffer(3)]],
                         uint i [[thread_position_in_grid]]) {
    if (i < n) { float g = gate[i]; out[i] = (g / (1.0f + precise::exp(-g))) * up[i]; }
}
