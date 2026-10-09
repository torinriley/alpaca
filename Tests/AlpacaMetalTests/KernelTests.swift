// Copyright (c) 2026 Torin Etheridge. MIT License (see LICENSE).
// Author: Torin Etheridge · 2026-10-09

import XCTest
import AlpacaCore
@testable import AlpacaMetal

/// GPU kernels vs the CPU reference implementations that were validated against numpy/PyTorch/gguf.
///
/// Tolerances: both sides use float32 accumulation but different summation orders (32-lane tree reduction
/// vs sequential). Error is bounded by ≈ K·u·Σ|aᵢbᵢ| with u = 2^-24; for K ≤ 1536 and unit-variance inputs
/// that is ≲ 1e-4 worst case, ~1e-5 typical. Bounds below are 2e-4 absolute + 1e-4 relative for dot products,
/// 1e-6/1e-5 for elementwise/normalisation, 2e-6 for attention with f32 K/V, and (the CPU reference is fed the same f16-rounded K/V) 2e-5 for attention with f16 K/V.
final class KernelTests: XCTestCase {
    var ops: MetalOps!

    override func setUpWithError() throws {
        try XCTSkipUnless(MetalContext.isAvailable, "no Metal device")
        ops = MetalOps(context: try MetalContext())
    }

    func rand(_ shape: [Int], seed: UInt64, scale: Float = 1) throws -> Tensor {
        var g = SplitMix(seed: seed)
        let n = shape.reduce(1, *)
        return try Tensor((0..<n).map { _ in g.nextFloat() * scale }, shape: shape)
    }

    func compare(_ gpu: Tensor, _ cpu: Tensor, atol: Double, rtol: Double, _ label: String) throws {
        let a = try gpu.toFloatArray(), e = try cpu.toFloatArray().map(Double.init)
        let m = ErrorMetrics(actual: a, expected: e)
        print("[metrics] GPU \(label): \(m)")
        for (i, (x, y)) in zip(a, e).enumerated() where abs(Double(x) - y) > atol + rtol * abs(y) {
            return XCTFail("\(label)[\(i)] gpu \(x) cpu \(y) (atol \(atol), rtol \(rtol))")
        }
    }

    func testElementwise() throws {
        let a = try rand([7, 37], seed: 1, scale: 3), b = try rand([7, 37], seed: 2, scale: 3)
        try compare(ops.add(a, b), CPUOps.add(a, b), atol: 1e-6, rtol: 1e-6, "add")
        try compare(ops.mul(a, b), CPUOps.mul(a, b), atol: 1e-6, rtol: 1e-6, "mul")
        try compare(ops.silu(a), CPUOps.silu(a), atol: 1e-6, rtol: 1e-5, "silu")
        try compare(ops.siluMul(gate: a, up: b), CPUOps.mul(CPUOps.silu(a), b), atol: 1e-5, rtol: 1e-5, "silu*mul")
    }

    func testRMSNorm() throws {
        for dim in [32, 576, 1536, 2048] {
            let x = try rand([5, dim], seed: 3, scale: 2), w = try rand([dim], seed: 4)
            try compare(ops.rmsNorm(x, weight: w, eps: 1e-5), CPUOps.rmsNorm(x, weight: w, eps: 1e-5), atol: 1e-5, rtol: 1e-5, "rmsnorm dim \(dim)")
        }
    }

    func testLinearAllFormatsAndBatchSizes() throws {
        // N not a multiple of 4 (threadgroup rows), T not a multiple of 8 (token block): exercises both edge guards.
        for (k, n) in [(64, 10), (576, 193), (1536, 576)] {
            let w32 = try rand([n, k], seed: 5, scale: 0.3)
            let q8 = try quantize(w32.toFloatArray(), to: .q8_0).reshaped([n, k])
            let q4 = try quantize(w32.toFloatArray(), to: .q4_0).reshaped([n, k])
            let weights: [(String, Tensor)] = [("f16", try w32.converted(to: .float16)), ("q8_0", q8), ("q4_0", q4)]
            for (name, w) in weights {
                for t in [1, 3, 8, 11, 16, 33, 70] {
                    let x = try rand([t, k], seed: 6 + UInt64(t))
                    try compare(ops.linear(x, weight: w), CPUOps.linear(x, weight: w), atol: 2e-4, rtol: 1e-4, "linear \(name) K=\(k) N=\(n) T=\(t)")
                }
            }
        }
    }

    func testRoPE() throws {
        for (start, theta) in [(0, Float(10000)), (7, 100000), (1500, 100000)] {
            let x = try rand([4, 9, 64], seed: 9)
            // Angle = position * freq in float32; error ≈ 1.2e-7 * angle (≤ ~1e-4 rad at position 1500).
            try compare(ops.rope(x, startPosition: start, theta: theta), CPUOps.rope(x, startPosition: start, theta: theta), atol: 2e-4, rtol: 0, "rope start \(start)")
        }
    }

    func testAttentionCausalGQA() throws {
        for (tokens, start, heads, kv, hd) in [(1, 0, 4, 2, 64), (1, 100, 9, 3, 64), (5, 0, 9, 3, 64), (7, 20, 8, 2, 128), (3, 1, 6, 6, 32)] where tokens < 8 || hd == 32 {
            let cap = start + tokens + 3
            let q = try rand([tokens, heads, hd], seed: 11), k = try rand([cap, kv, hd], seed: 12), v = try rand([cap, kv, hd], seed: 13)
            let cpuScores = try CPUOps.attentionScores(q: q, keys: k.converted(to: .float16).converted(to: .float32), length: start + tokens, startPosition: start)
            let cpu = try CPUOps.attentionApply(probs: CPUOps.softmax(cpuScores), values: v.converted(to: .float16).converted(to: .float32))
            try compare(ops.attention(q: q, keys: k, values: v, startPosition: start), cpu, atol: 2e-5, rtol: 1e-4,
                        "attention f16-KV T=\(tokens) start=\(start) H=\(heads)/\(kv) hd=\(hd)")
            let cpu32 = try CPUOps.attentionApply(probs: CPUOps.softmax(CPUOps.attentionScores(q: q, keys: k, length: start + tokens, startPosition: start)), values: v)
            try compare(ops.attention(q: q, keys: k, values: v, startPosition: start, kv: .float32), cpu32, atol: 2e-6, rtol: 1e-5,
                        "attention f32-KV T=\(tokens) start=\(start) H=\(heads)/\(kv) hd=\(hd)")
        }
    }

    func testEmbeddingAllFormats() throws {
        let table = try rand([50, 96], seed: 21)
        let tokens: [Int32] = [0, 49, 7, 7]
        for (name, t) in [("f16", try table.converted(to: .float16)),
                          ("q8_0", try quantize(table.toFloatArray(), to: .q8_0).reshaped([50, 96])),
                          ("q4_0", try quantize(table.toFloatArray(), to: .q4_0).reshaped([50, 96]))] {
            try compare(ops.embed(tokens: tokens, table: t), CPUOps.embed(tokens: tokens, table: t), atol: 1e-6, rtol: 1e-6, "embed \(name)")
        }
        XCTAssertThrowsError(try ops.embed(tokens: [50], table: table.converted(to: .float16)))
    }
}

struct SplitMix {
    var state: UInt64
    init(seed: UInt64) { state = seed }
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
    /// Uniform in [-1, 1).
    mutating func nextFloat() -> Float { Float(Double(next() >> 11) / Double(1 << 53)) * 2 - 1 }
}

/// The tiled prefill attention kernel (half MMA operands, float32 accumulation, online softmax).
///
/// Tolerance basis: Q, the exponentiated scores P and K/V are rounded to half (relative 2^-11 = 4.9e-4 each) before
/// the matrix multiply; each output is a convex combination of V rows (|V| ≤ 1 here), so the absolute error is bounded by
/// roughly 3 × 2^-11 × max|V| ≈ 1.5e-3 in the worst case, ~1e-4 typical. Compared against a float64-accurate CPU
/// reference fed the same half-rounded K/V; the Q and P roundings are what remain.
final class PrefillAttentionTests: XCTestCase {
    /// Split-K decode attention (one token): context lengths around split boundaries (splitLen 256), GQA groups 1…8, both head dims.
    func testSplitDecodeAttentionMatchesCPUReference() throws {
        try XCTSkipUnless(MetalContext.isAvailable, "no Metal device")
        let ops = MetalOps(context: try MetalContext())
        let cases: [(start: Int, heads: Int, kv: Int, hd: Int)] = [
            (0, 9, 3, 64), (1, 9, 3, 64), (7, 9, 3, 64), (255, 9, 3, 64), (256, 9, 3, 64), (257, 9, 3, 64), (511, 9, 3, 64),
            (1000, 9, 3, 64), (2047, 9, 3, 64), (300, 8, 2, 128), (700, 8, 8, 64), (90, 32, 4, 128), (500, 4, 1, 64),
        ]
        for c in cases {
            let cap = c.start + 1
            var g = SplitMix(seed: UInt64(c.start * 7 + c.heads))
            func r(_ shape: [Int]) throws -> Tensor { try Tensor((0..<shape.reduce(1, *)).map { _ in g.nextFloat() }, shape: shape) }
            let q = try r([1, c.heads, c.hd]), k = try r([cap, c.kv, c.hd]), v = try r([cap, c.kv, c.hd])
            let k16 = try k.converted(to: .float16).converted(to: .float32), v16 = try v.converted(to: .float16).converted(to: .float32)
            let cpu = try CPUOps.attentionApply(probs: CPUOps.softmax(CPUOps.attentionScores(q: q, keys: k16, length: cap, startPosition: c.start)), values: v16)
            let gpu = try ops.attention(q: q, keys: k, values: v, startPosition: c.start)
            let m = ErrorMetrics(actual: try gpu.toFloatArray(), expected: try cpu.toFloatArray().map(Double.init))
            print("[metrics] GPU split decode attention ctx=\(cap) H=\(c.heads)/\(c.kv) hd=\(c.hd): \(m)")
            // Same float32 arithmetic as the reference apart from summation order and the split merge: tight bound.
            XCTAssertLessThan(m.maxAbs, 2e-6, "\(c)")
        }
    }

    func testTiledAttentionMatchesCPUReference() throws {
        try XCTSkipUnless(MetalContext.isAvailable, "no Metal device")
        let ops = MetalOps(context: try MetalContext())
        // Cover: partial query tiles (T % 32, T % 8), multi-block causal diagonals, start offsets (chunked prefill),
        // GQA ratios 1/2/3/4, both head dims, and a start such that the last block straddles the cache end.
        let cases: [(t: Int, start: Int, heads: Int, kv: Int, hd: Int)] = [
            (8, 0, 9, 3, 64), (31, 0, 9, 3, 64), (32, 0, 9, 3, 64), (33, 0, 9, 3, 64), (100, 0, 9, 3, 64),
            (64, 200, 9, 3, 64), (45, 7, 8, 2, 128), (128, 128, 4, 4, 128), (17, 3, 6, 3, 64), (200, 0, 8, 2, 64),
        ]
        for c in cases {
            let cap = c.start + c.t
            var g = SplitMix(seed: UInt64(c.t * 131 + c.start))
            func r(_ shape: [Int], _ sc: Float) throws -> Tensor { try Tensor((0..<shape.reduce(1, *)).map { _ in g.nextFloat() * sc }, shape: shape) }
            let q = try r([c.t, c.heads, c.hd], 1), k = try r([cap, c.kv, c.hd], 1), v = try r([cap, c.kv, c.hd], 1)
            let k16 = try k.converted(to: .float16).converted(to: .float32), v16 = try v.converted(to: .float16).converted(to: .float32)
            let cpu = try CPUOps.attentionApply(probs: CPUOps.softmax(CPUOps.attentionScores(q: q, keys: k16, length: cap, startPosition: c.start)), values: v16)
            let gpu = try ops.attention(q: q, keys: k, values: v, startPosition: c.start)
            let m = ErrorMetrics(actual: try gpu.toFloatArray(), expected: try cpu.toFloatArray().map(Double.init))
            print("[metrics] GPU tiled attention T=\(c.t) start=\(c.start) H=\(c.heads)/\(c.kv) hd=\(c.hd): \(m)")
            XCTAssertLessThan(m.maxAbs, 1.5e-3, "\(c)")
            XCTAssertTrue(try gpu.toFloatArray().allSatisfy { $0.isFinite })
        }
    }
}

/// `.fast` GEMM (Metal 4 tensor ops, relaxed precision): half-precision operands, float32 accumulation.
///
/// Tolerance basis (per output element, worst case, first order): rounding x to half perturbs each product by ≤ 2^-11
/// relative, rounding the (dequantised) weight by ≤ 2^-11 relative, so |error| ≤ 2^-10 · Σₖ |xₖ·wₖ| plus float32
/// accumulation noise (≲ K·2^-24·Σ|xw|, negligible here). The test asserts that bound element-wise, and reports the
/// observed ratio (error / bound) to show how much of it is used.
final class FastGEMMTests: XCTestCase {
    func testFastGEMMWithinAnalyticBound() throws {
        try XCTSkipUnless(MetalContext.isAvailable, "no Metal device")
        let context = try MetalContext()
        try XCTSkipUnless(context.supportsTensorGEMM, "tensor-op GEMM unavailable on this GPU/OS")
        let ops = MetalOps(context: context)
        var worstRatio = 0.0
        #if DEBUG
        let shapes = [(64, 10), (576, 193)], batches = [33, 70]        // the O(T·N·K) reference loops are slow unoptimised
        #else
        let shapes = [(64, 10), (576, 193), (1536, 576), (576, 1536)], batches = [32, 33, 70, 130, 300]
        #endif
        for (k, n) in shapes {
            var g = SplitMix(seed: UInt64(k * 31 + n))
            func r(_ shape: [Int], _ s: Float) throws -> Tensor { try Tensor((0..<shape.reduce(1, *)).map { _ in g.nextFloat() * s }, shape: shape) }
            let w32 = try r([n, k], 0.3)
            let weights: [(String, Tensor)] = [("f16", try w32.converted(to: .float16)),
                ("q8_0", try quantize(w32.toFloatArray(), to: .q8_0).reshaped([n, k])), ("q4_0", try quantize(w32.toFloatArray(), to: .q4_0).reshaped([n, k]))]
            for (name, w) in weights {
                for t in batches {
                    let x = try r([t, k], 1)
                    let gpu = try ops.linear(x, weight: w, precision: .fast).toFloatArray()
                    let cpu = try CPUOps.linear(x, weight: w).toFloatArray()
                    let xv = try x.toFloatArray(), wv = try w.toFloatArray()
                    var maxErr = 0.0
                    for ti in 0..<t { for ni in 0..<n {
                        var absSum = 0.0
                        for ki in 0..<k { absSum += abs(Double(xv[ti * k + ki]) * Double(wv[ni * k + ki])) }
                        let err = abs(Double(gpu[ti * n + ni]) - Double(cpu[ti * n + ni]))
                        let bound = absSum / 1024 + 1e-5
                        worstRatio = max(worstRatio, err / bound); maxErr = max(maxErr, err)
                        if err > bound { return XCTFail("fast GEMM \(name) K=\(k) N=\(n) T=\(t) [\(ti),\(ni)]: err \(err) > bound \(bound)") }
                    } }
                    print("[metrics] GPU fast GEMM \(name) K=\(k) N=\(n) T=\(t): max|Δ| \(maxErr)")
                }
            }
        }
        print("[metrics] fast GEMM worst error / analytic bound = \(worstRatio)")
    }

    func testFusedGateUpSiluMatchesReference() throws {
        try XCTSkipUnless(MetalContext.isAvailable, "no Metal device")
        let context = try MetalContext()
        let ops = MetalOps(context: context)
        var g = SplitMix(seed: 77)
        func r(_ shape: [Int], _ s: Float) throws -> Tensor { try Tensor((0..<shape.reduce(1, *)).map { _ in g.nextFloat() * s }, shape: shape) }
        let (k, n) = (576, 200)                                   // n not a multiple of 64: edge tiles
        for t in [3, 16, 40, 100] {
            let x = try r([t, k], 1)
            let wg32 = try r([n, k], 0.3), wu32 = try r([n, k], 0.3)
            let pairs: [(Tensor, Tensor)] = [(try wg32.converted(to: .float16), try wu32.converted(to: .float16)),
                (try quantize(wg32.toFloatArray(), to: .q8_0).reshaped([n, k]), try quantize(wu32.toFloatArray(), to: .q8_0).reshaped([n, k])),
                (try quantize(wg32.toFloatArray(), to: .q4_0).reshaped([n, k]), try quantize(wu32.toFloatArray(), to: .q4_0).reshaped([n, k]))]
            for (wg, wu) in pairs {
                let cpu = try CPUOps.mul(CPUOps.silu(CPUOps.linear(x, weight: wg)), CPUOps.linear(x, weight: wu))
                for precision in [GEMMPrecision.exact, .fast] where precision == .exact || context.supportsTensorGEMM {
                    let gpu = try ops.gateUpSilu(x, gate: wg, up: wu, precision: precision)
                    let m = ErrorMetrics(actual: try gpu.toFloatArray(), expected: try cpu.toFloatArray().map(Double.init))
                    print("[metrics] GPU gate+up+silu T=\(t) \(wg.dtype) \(precision): \(m)")
                    // exact: float32 arithmetic. fast: half operands (2^-11 each) through silu and a product of two such sums,
                    // so the error scales with the output magnitude: bound relative to the peak (observed worst 8.3e-4).
                    if precision == .exact || t < 16 { XCTAssertLessThan(m.maxAbs, 1e-3) } else { XCTAssertLessThan(m.maxRelativeToPeak, 3e-3) }
                }
            }
        }
    }

    func testFusedResidualProjection() throws {
        try XCTSkipUnless(MetalContext.isAvailable, "no Metal device")
        let context = try MetalContext()
        try XCTSkipUnless(context.supportsTensorGEMM, "tensor-op GEMM unavailable on this GPU/OS")
        let ops = MetalOps(context: context)
        var g = SplitMix(seed: 5)
        func r(_ shape: [Int], _ s: Float) throws -> Tensor { try Tensor((0..<shape.reduce(1, *)).map { _ in g.nextFloat() * s }, shape: shape) }
        let (k, n) = (576, 576)
        for t in [1, 5, 8, 31, 40, 100] {
            let x = try r([t, k], 1), res = try r([t, n], 5)
            for w in [try r([n, k], 0.3).converted(to: .float16), try quantize(r([n, k], 0.3).toFloatArray(), to: .q8_0).reshaped([n, k]),
                      try quantize(r([n, k], 0.3).toFloatArray(), to: .q4_0).reshaped([n, k])] {
                let cpu = try CPUOps.add(res, CPUOps.linear(x, weight: w))
                for precision in [GEMMPrecision.exact, .fast] {
                    let gpu = try ops.linearAdd(x, weight: w, residual: res, precision: precision)
                    let m = ErrorMetrics(actual: try gpu.toFloatArray(), expected: try cpu.toFloatArray().map(Double.init))
                    print("[metrics] GPU linear+residual T=\(t) \(w.dtype) \(precision): \(m)")
                    XCTAssertLessThan(m.maxAbs, precision == .exact || t < 16 ? 2e-4 : 2e-2)
                }
            }
        }
    }
}
