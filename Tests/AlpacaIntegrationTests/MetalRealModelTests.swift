// Copyright (c) 2026 Torin Etheridge. MIT License (see LICENSE).
// Author: Torin Etheridge · 2026-10-09

import XCTest
import AlpacaCore
import AlpacaModels
import AlpacaMetal

/// Level 5/6 — real SmolLM2-135M-Instruct on the Metal backend vs Hugging Face float32 on the same GGUF weights.
///
/// The default GPU configuration trades a little precision for speed (f16 KV cache, half-precision GEMM operands during prefill,
/// half Q/P in tiled attention); each is isolated by its own test below. Observed values are in Docs/NUMERICAL_VALIDATION.md.
final class MetalRealModelTests: XCTestCase {
    override func setUpWithError() throws { try requireOptimizedBuild() }
    func check(quant: String, logitsAtol: Double) throws {
        try XCTSkipUnless(MetalContext.isAvailable, "no Metal device")
        let loaded = try LlamaLoader.load(url: ModelFiles.url(quant))
        let context = try MetalContext()
        let model = try MetalLlamaModel(context: context, config: loaded.config, weights: loaded.weights,
                                        mappedRegion: loaded.file.mappedMemory, keepAlive: loaded.file.mappingOwner)
        print("[metal] \(context.deviceName): \(model.zeroCopyWeightBytes) weight bytes zero-copy, \(model.copiedWeightBytes) copied")
        for ref in try Reference.load(quant) {
            let session = try model.makeSession(capacity: 256)
            let t0 = Date()
            let logits = try session.forward(tokens: ref.ids)
            let wall = Date().timeIntervalSince(t0)
            let m = ErrorMetrics(actual: logits, expected: ref.lastLogits)
            print("[metrics] GPU \(quant) prefill(\(ref.ids.count) tok, gpu \(String(format: "%.1f", session.lastGPUSeconds * 1000)) ms, wall \(String(format: "%.1f", wall * 1000)) ms) vs HF-on-same-weights: \(m)")
            // Error scales with the logit magnitude (half-precision GEMM operands, f16 KV cache, half Q/P in tiled attention), so the
            // bound is relative to the logit peak: observed worst 1.5e-3 (all formats, all prompts) → bound 3e-3.
            XCTAssertLessThan(m.maxRelativeToPeak, 3e-3)
            XCTAssertEqual(argmax(logits), argmax(ref.lastLogits))

            var next = argmax(logits), matched = 0
            for step in 0..<ref.generated.count {
                if next != ref.generated[step] {
                    XCTAssertLessThan(ref.margins[step], 8 * logitsAtol, "\(quant): GPU token \(step) diverged with clear margin \(ref.margins[step])")
                    break
                }
                matched += 1
                next = argmax(try session.forward(tokens: [next]))
            }
            print("[generate] GPU \(quant) \(matched)/\(ref.generated.count) tokens identical to reference greedy continuation")
        }
    }

    func testF16() throws { try check(quant: "f16", logitsAtol: 5e-2) }
    func testQ8_0() throws { try check(quant: "Q8_0", logitsAtol: 5e-2) }
    func testQ4_0() throws { try check(quant: "Q4_0", logitsAtol: 5e-2) }

    /// GPU and CPU backends must agree token-by-token through the same cache lifecycle (chunked prefill, decode, reset).
    func testGPUMatchesCPUBackendAcrossChunksAndReset() throws {
        try XCTSkipUnless(MetalContext.isAvailable, "no Metal device")
        let loaded = try LlamaLoader.load(url: ModelFiles.url("Q8_0"))
        let cpu = try LlamaCPUModel(config: loaded.config, weights: loaded.weights)
        let gpuModel = try MetalLlamaModel(context: MetalContext(), config: loaded.config, weights: loaded.weights,
                                           mappedRegion: loaded.file.mappedMemory, keepAlive: loaded.file.mappingOwner)
        let prompt = try Reference.load("Q8_0")[2].ids               // 90 tokens: 3 chunks at maxBatch 32
        // f32 KV and exact (float32) GEMMs on both sides: any difference is arithmetic (summation order, fast-math transcendental).
        let gpu = try gpuModel.makeSession(capacity: 128, maxBatch: 32, kvPrecision: .float32, gemmPrecision: .exact)
        let cache = try KVCache(config: loaded.config, capacity: 128)
        let g = try gpu.forward(tokens: prompt), c = try cpu.forward(tokens: prompt, cache: cache).toFloatArray()
        let m = ErrorMetrics(actual: g, expected: c)
        print("[metrics] GPU vs CPU chunked prefill: \(m)")
        XCTAssertLessThan(m.maxAbs, 5e-4)
        XCTAssertEqual(argmax(g), argmax(c))
        let g2 = try gpu.forward(tokens: [argmax(g)]), c2 = try cpu.forward(tokens: [argmax(c)], cache: cache).toFloatArray()
        XCTAssertLessThan(ErrorMetrics(actual: g2, expected: c2).maxAbs, 5e-4)
        XCTAssertEqual(gpu.length, prompt.count + 1)

        gpu.reset()
        let again = try gpu.forward(tokens: prompt)
        XCTAssertEqual(again, g, "reset must reproduce the first run bit-for-bit (deterministic kernels)")
        XCTAssertThrowsError(try gpu.forward(tokens: Array(repeating: 1, count: 200)))   // exceeds capacity
        XCTAssertThrowsError(try gpu.forward(tokens: [999_999]))
        XCTAssertThrowsError(try gpu.forward(tokens: []))
    }

    /// GPU-resident greedy decoding (`decodeGreedy`, several command buffers in flight) must produce exactly the tokens of
    /// the synchronous path (`forward` + CPU arg-max), stop cleanly mid-chain, respect capacity, and leave the session usable.
    func testGPUGreedyChainMatchesSynchronousDecoding() throws {
        try XCTSkipUnless(MetalContext.isAvailable, "no Metal device")
        let loaded = try LlamaLoader.load(url: ModelFiles.url("Q8_0"))
        let model = try MetalLlamaModel(context: MetalContext(), config: loaded.config, weights: loaded.weights,
                                        mappedRegion: loaded.file.mappedMemory, keepAlive: loaded.file.mappingOwner)
        let prompt = try Reference.load("Q8_0")[1].ids
        let sync = try model.makeSession(capacity: 128)
        var logits = try sync.forward(tokens: prompt)
        var expected: [Int32] = []
        for _ in 0..<40 { let t = argmax(logits); expected.append(t); logits = try sync.forward(tokens: [t]) }

        let chain = try model.makeSession(capacity: 128)
        let first = argmax(try chain.forward(tokens: prompt))
        XCTAssertEqual(first, expected[0])
        var got: [Int32] = []
        let fed = try chain.decodeGreedy(startToken: first, maxSteps: 39) { got.append($0); return true }
        XCTAssertEqual([first] + got, expected, "chained GPU arg-max must reproduce the synchronous token sequence")
        XCTAssertEqual(fed, 39)
        XCTAssertEqual(chain.length, prompt.count + 39)

        // Early stop: the caller halts after 5 tokens while up to `depth` extra steps are in flight; length counts only
        // tokens whose input was really consumed, and the session continues correctly afterwards.
        let stop = try model.makeSession(capacity: 128)
        _ = try stop.forward(tokens: prompt)
        var seen: [Int32] = []
        let fed2 = try stop.decodeGreedy(startToken: first, maxSteps: 30) { seen.append($0); return seen.count < 5 }
        XCTAssertEqual(seen, Array(expected[1...5]))
        XCTAssertEqual(fed2, 5)
        XCTAssertEqual(stop.length, prompt.count + 5)
        // Feeding the next token synchronously now must continue the same greedy sequence.
        let resumed = try stop.forward(tokens: [expected[5]])
        XCTAssertEqual(argmax(resumed), expected[6])

        // Capacity: a session that can feed only 3 more tokens stops after 3 steps.
        let tight = try model.makeSession(capacity: prompt.count + 3)
        _ = try tight.forward(tokens: prompt)
        var n = 0
        let fed3 = try tight.decodeGreedy(startToken: first, maxSteps: 100) { _ in n += 1; return true }
        XCTAssertEqual(fed3, 3); XCTAssertEqual(n, 3); XCTAssertEqual(tight.length, prompt.count + 3)
        XCTAssertEqual(try tight.decodeGreedy(startToken: first, maxSteps: 5) { _ in true }, 0)
        XCTAssertThrowsError(try tight.decodeGreedy(startToken: 999_999, maxSteps: 1) { _ in true })
    }

    /// `decode(token:)` returns a no-copy view of the same logits `forward` returns.
    func testDecodeViewMatchesForward() throws {
        try XCTSkipUnless(MetalContext.isAvailable, "no Metal device")
        let loaded = try LlamaLoader.load(url: ModelFiles.url("Q8_0"))
        let model = try MetalLlamaModel(context: MetalContext(), config: loaded.config, weights: loaded.weights,
                                        mappedRegion: loaded.file.mappedMemory, keepAlive: loaded.file.mappingOwner)
        let prompt = try Reference.load("Q8_0")[0].ids
        let a = try model.makeSession(capacity: 64), b = try model.makeSession(capacity: 64)
        var la = try a.forward(tokens: prompt); _ = try b.forward(tokens: prompt)
        for _ in 0..<6 {
            let t = argmax(la)
            la = try a.forward(tokens: [t])
            XCTAssertEqual(Array(try b.decode(token: t)), la)
        }
    }

    /// Isolates the cost of the `.fast` prefill GEMM: same GPU, same prompt, only the projection precision differs.
    func testFastPrefillGEMMErrorIsBoundedAndTokensAgree() throws {
        try XCTSkipUnless(MetalContext.isAvailable, "no Metal device")
        let context = try MetalContext()
        try XCTSkipUnless(context.supportsTensorGEMM, "tensor-op GEMM unavailable on this GPU/OS")
        for quant in ["f16", "Q8_0", "Q4_0"] {
            let loaded = try LlamaLoader.load(url: ModelFiles.url(quant))
            let model = try MetalLlamaModel(context: context, config: loaded.config, weights: loaded.weights,
                                            mappedRegion: loaded.file.mappedMemory, keepAlive: loaded.file.mappingOwner)
            for ref in try Reference.load(quant) {
                let exact = try model.makeSession(capacity: 256, kvPrecision: .float16, gemmPrecision: .exact).forward(tokens: ref.ids)
                let fast = try model.makeSession(capacity: 256, kvPrecision: .float16, gemmPrecision: .fast).forward(tokens: ref.ids)
                let m = ErrorMetrics(actual: fast, expected: exact)
                print("[metrics] GPU \(quant) fast-vs-exact prefill GEMM (\(ref.ids.count) tok) logits: \(m)")
                // Half-precision operands (relative 2^-11 ≈ 4.9e-4 per operand) compounded over 30 layers: the error scales with the
                // logit magnitude, so the bound is relative to the logit peak. Observed worst 1.4e-3 → bound 3e-3 (about 2x).
                XCTAssertLessThan(m.maxRelativeToPeak, 3e-3)
                XCTAssertEqual(argmax(fast), argmax(exact))
            }
        }
    }

    /// Isolates the cost of the f16 KV cache: same GPU, same prompt, only the cache precision differs.
    func testFloat16KVCacheErrorIsBoundedAndTokensAgree() throws {
        try XCTSkipUnless(MetalContext.isAvailable, "no Metal device")
        let loaded = try LlamaLoader.load(url: ModelFiles.url("Q8_0"))
        let model = try MetalLlamaModel(context: MetalContext(), config: loaded.config, weights: loaded.weights,
                                        mappedRegion: loaded.file.mappedMemory, keepAlive: loaded.file.mappingOwner)
        let prompt = try Reference.load("Q8_0")[2].ids
        let a = try model.makeSession(capacity: 128, kvPrecision: .float32, gemmPrecision: .exact).forward(tokens: prompt)
        let b = try model.makeSession(capacity: 128, kvPrecision: .float16, gemmPrecision: .exact).forward(tokens: prompt)
        let m = ErrorMetrics(actual: b, expected: a)
        print("[metrics] GPU f16-KV vs f32-KV logits (same weights, same kernels): \(m)")
        XCTAssertLessThan(m.maxAbs, 3e-2)
        XCTAssertEqual(argmax(a), argmax(b))
    }
}
