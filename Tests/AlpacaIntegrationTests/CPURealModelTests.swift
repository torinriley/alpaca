// Copyright (c) 2026 Torin Etheridge. MIT License (see LICENSE).
// Author: Torin Etheridge · 2026-10-09

import XCTest
import AlpacaCore
import AlpacaModels

/// Level 5 — real SmolLM2-135M-Instruct execution on the CPU reference backend, compared with Hugging Face
/// float32 running on the *same GGUF weights* (so quantisation error is excluded; only arithmetic differs).
///
/// Run in release for speed:  swift test -c release --filter AlpacaIntegrationTests
/// Tolerances (logits are O(10)): all three files compare identical dequantised weights, so only float32
/// accumulation-order differences over 30 layers remain. Observed max |Δlogit| (Docs/NUMERICAL_VALIDATION.md):
/// F16 5e-5, Q8_0 7e-5, Q4_0 1.6e-3. Q4_0 is larger because three Q4_1 tensors are rounded to F16 at load
/// (relative 2^-11 per weight). Bounds below are ~10x observed so they detect regressions rather than hide them.
final class CPURealModelTests: XCTestCase {
    override func setUpWithError() throws { try requireOptimizedBuild() }
    func check(quant: String, logitsAtol: Double) throws {
        let loaded = try LlamaLoader.load(url: ModelFiles.url(quant))
        let model = try LlamaCPUModel(config: loaded.config, weights: loaded.weights)
        for note in loaded.report.notes { print("[load] \(note)") }
        for ref in try Reference.load(quant) {
            let cache = try KVCache(config: loaded.config, capacity: 256)
            let started = Date()
            let logits = try model.forward(tokens: ref.ids, cache: cache).toFloatArray()
            let m = ErrorMetrics(actual: logits, expected: ref.lastLogits)
            print("[metrics] \(quant) prefill(\(ref.ids.count) tok, \(String(format: "%.2f", Date().timeIntervalSince(started)))s) last-token logits vs HF-on-same-weights: \(m)")
            XCTAssertLessThan(m.maxAbs, logitsAtol, "\(quant) \(ref.prompt.prefix(20))")
            XCTAssertEqual(argmax(logits), argmax(ref.lastLogits))

            // Greedy decode through the KV cache must reproduce the reference continuation wherever the
            // reference's top-1 margin exceeds the numerical noise floor.
            var next = argmax(logits)
            var out: [Int32] = []
            for step in 0..<ref.generated.count {
                if next != ref.generated[step] {
                    XCTAssertLessThan(ref.margins[step], 20 * logitsAtol, "\(quant): token \(step) diverged with a clear margin \(ref.margins[step])")
                    break
                }
                out.append(next)
                next = argmax(try model.forward(tokens: [next], cache: cache).toFloatArray())
            }
            print("[generate] \(quant) \(out.count)/\(ref.generated.count) tokens identical to reference greedy continuation")
        }
    }

    func testF16() throws { try check(quant: "f16", logitsAtol: 5e-4) }
    func testQ8_0() throws { try check(quant: "Q8_0", logitsAtol: 7e-4) }
    func testQ4_0() throws { try check(quant: "Q4_0", logitsAtol: 1.6e-2) }
}
