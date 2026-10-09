// Copyright (c) 2026 Torin Etheridge. MIT License (see LICENSE).
// Author: Torin Etheridge · 2026-10-09

#include <metal_stdlib>
using namespace metal;

// alpaca Metal kernels. Conventions shared by every kernel file:
//  * Activations are float32, row-major [tokens, width].
//  * Weights are [out, in] row-major in their GGUF on-disk layout (f16 / q8_0 / q4_0 blocks), read in place.
//  * One simdgroup (32 lanes) reduces one dot product; all accumulation is float32.
//  * Dispatches are encoded on one serial compute encoder, so each kernel observes the previous kernel's writes.
// Block layouts (see DType.swift):
//  q8_0: half d; int8 qs[32]           = 34 bytes, value = d * qs[j]
//  q4_0: half d; uint8 qs[16]          = 18 bytes, value = d * ((qs[j] & 0xF) - 8) for element j,
//                                                           d * ((qs[j] >> 4) - 8)  for element j + 16

constant constexpr uint SIMD_WIDTH = 32;
constant constexpr uint Q_BLOCK = 32;
constant constexpr uint Q8_BYTES = 34;
constant constexpr uint Q4_BYTES = 18;
// Simdgroups per threadgroup in the mat-vec kernels: each simdgroup produces one output row.
constant constexpr uint MV_SIMDGROUPS = 4;

// Finite stand-in for -infinity in running-max initialisation (Metal fast-math does not guarantee inf semantics).
constant constexpr float NEG_BIG_PF = -1.0e30f;
constant constexpr float INFINITY_F = 3.0e38f;   // finite stand-in for +inf (fast-math safe): 'below every logit' sentinel
