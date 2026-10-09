// Copyright (c) 2026 Torin Etheridge. MIT License (see LICENSE).
// Author: Torin Etheridge · 2026-10-09

import XCTest
@testable import AlpacaCore

/// Levels 3–4: full forward pass vs a float64 Hugging Face reference, and cached vs uncached decoding.
///
/// Tolerance: the model runs in Float32 with 2 layers and weights ~N(0, 0.15²); logits are O(1).
/// Accumulated rounding error is a small multiple of u·depth·√width ≈ 1e-6; we allow 2e-5 absolute.
final class TransformerTests: XCTestCase {
    let atol = 2e-5

    struct Case { let config: LlamaConfig; var weights: LlamaWeights; let tokens: [Int32]; let logits: [Double] }

    func load(_ name: String, gguf: Bool) throws -> Case {
        let all = try Fixtures.load("tiny_llama")
        let c = all.dict(name)
        let cc = c.dict("config")
        let config = LlamaConfig(
            vocabSize: cc.int("vocab"), hiddenSize: cc.int("hidden"), layerCount: cc.int("layers"),
            headCount: cc.int("heads"), kvHeadCount: cc.int("kvHeads"), headDim: cc.int("headDim"),
            feedForwardSize: cc.int("ffn"), contextLength: cc.int("context"),
            rmsNormEps: Float(cc["eps"] as! Double), ropeTheta: Float(cc["theta"] as! Double), ropeInterleaved: gguf)
        let w = c.dict(gguf ? "weights_gguf" : "weights_hf")
        func t(_ n: String) throws -> Tensor { let e = w.dict(n); return try Tensor(e.floats("data"), shape: e.ints("shape")) }
        let layers = try (0..<config.layerCount).map { i in
            try LlamaLayerWeights(
                attnNorm: t("blk.\(i).attn_norm"), wq: t("blk.\(i).attn_q"), wk: t("blk.\(i).attn_k"), wv: t("blk.\(i).attn_v"),
                wo: t("blk.\(i).attn_output"), ffnNorm: t("blk.\(i).ffn_norm"), wGate: t("blk.\(i).ffn_gate"),
                wUp: t("blk.\(i).ffn_up"), wDown: t("blk.\(i).ffn_down"))
        }
        let weights = try LlamaWeights(tokenEmbedding: t("token_embd"), layers: layers, outputNorm: t("output_norm"),
                                       output: (cc["tied"] as! Bool) ? nil : t("output"))
        return Case(config: config, weights: weights, tokens: c.ints("tokens").map(Int32.init), logits: c.doubles("logits"))
    }

    func forEachVariant(_ body: (String, Case) throws -> Void) throws {
        for name in ["gqa_untied", "mha_tied"] {
            for gguf in [false, true] { try body("\(name)/\(gguf ? "gguf-interleaved" : "hf-split")", load(name, gguf: gguf)) }
        }
    }

    func testPrefillLogitsMatchPyTorchFloat64() throws {
        try forEachVariant { label, c in
            let model = try LlamaCPUModel(config: c.config, weights: c.weights)
            let cache = try KVCache(config: c.config)
            let logits = try model.forward(tokens: c.tokens, cache: cache, logits: .all)
            XCTAssertEqual(logits.shape, [c.tokens.count, c.config.vocabSize])
            try assertClose(logits, c.logits, atol: atol, "prefill logits \(label)")
            // Greedy token agreement at every position.
            let got = try logits.toFloatArray(), v = c.config.vocabSize
            for p in 0..<c.tokens.count {
                let a = got[(p * v)..<((p + 1) * v)].enumerated().max { $0.element < $1.element }!.offset
                let r = c.logits[(p * v)..<((p + 1) * v)].enumerated().max { $0.element < $1.element }!.offset
                XCTAssertEqual(a, r, "argmax at position \(p) (\(label))")
            }
        }
    }

    func testIncrementalDecodeMatchesUncachedForward() throws {
        try forEachVariant { label, c in
            let model = try LlamaCPUModel(config: c.config, weights: c.weights)
            let v = c.config.vocabSize
            // Token-by-token decode through the KV cache.
            let cache = try KVCache(config: c.config)
            var stepLogits: [Float] = []
            for t in c.tokens { stepLogits += try model.forward(tokens: [t], cache: cache).toFloatArray() }
            try assertClose(Tensor(stepLogits, shape: [c.tokens.count, v]), c.logits, atol: atol, "incremental vs reference \(label)")
            // Split prefill (4 + 3) must equal single prefill.
            let split = try KVCache(config: c.config)
            let first = try model.forward(tokens: Array(c.tokens[0..<4]), cache: split, logits: .all).toFloatArray()
            let second = try model.forward(tokens: Array(c.tokens[4...]), cache: split, logits: .all).toFloatArray()
            let full = try model.forward(tokens: c.tokens, cache: KVCache(config: c.config), logits: .all).toFloatArray()
            let m = ErrorMetrics(actual: first + second, expected: full)
            print("[metrics] cached-vs-uncached \(label): \(m)")
            XCTAssertLessThan(m.maxAbs, 5e-6)   // same arithmetic order per row; only attention-sum grouping differs
            XCTAssertEqual(cache.length, c.tokens.count)
        }
    }

    func testCacheBoundsAndReset() throws {
        let c = try load("gqa_untied", gguf: true)
        let model = try LlamaCPUModel(config: c.config, weights: c.weights)
        let cache = try KVCache(config: c.config, capacity: 8)
        _ = try model.forward(tokens: c.tokens, cache: cache)                 // 7 of 8 used
        XCTAssertThrowsError(try model.forward(tokens: [1, 2], cache: cache)) // would exceed capacity
        XCTAssertEqual(cache.length, 7, "failed forward must not advance the cache")
        _ = try model.forward(tokens: [1], cache: cache)
        XCTAssertThrowsError(try model.forward(tokens: [1], cache: cache))
        XCTAssertThrowsError(try cache.write(layer: 0, position: 3, keys: Tensor(zeros: [1, 2, 8]), values: Tensor(zeros: [1, 2, 8])))
        // After reset the same prompt reproduces identical logits (stale rows are never read).
        cache.reset()
        let a = try model.forward(tokens: c.tokens, cache: cache).toFloatArray()
        let b = try model.forward(tokens: c.tokens, cache: KVCache(config: c.config, capacity: 8)).toFloatArray()
        XCTAssertEqual(a, b)
        XCTAssertThrowsError(try model.forward(tokens: [], cache: cache))
        XCTAssertThrowsError(try model.forward(tokens: [64], cache: try KVCache(config: c.config)))
    }

    func testWeightValidation() throws {
        var c = try load("gqa_untied", gguf: true)
        c.weights.layers[0].wk = try Tensor(zeros: [8, 32])
        XCTAssertThrowsError(try LlamaCPUModel(config: c.config, weights: c.weights))
        var bad = c.config; bad.kvHeadCount = 3
        XCTAssertThrowsError(try bad.validate())
    }
}
