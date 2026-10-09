// Copyright (c) 2026 Torin Etheridge. MIT License (see LICENSE).
// Author: Torin Etheridge · 2026-10-09

import XCTest
@testable import AlpacaCore

final class SamplerTests: XCTestCase {
    let logits: [Float] = [1.0, 3.0, 2.0, -1.0, 3.0, 0.5]

    func draws(_ p: SamplingParameters, n: Int = 20_000) throws -> [Int: Int] {
        var s = try Sampler(p)
        var counts: [Int: Int] = [:]
        for _ in 0..<n { counts[Int(try s.sample(logits: logits)), default: 0] += 1 }
        return counts
    }

    func testGreedyPicksFirstMaximum() throws {
        var s = try Sampler(.greedy)
        XCTAssertEqual(try s.sample(logits: logits), 1)       // tie between 1 and 4 → lowest index
    }

    func testSeededSamplingIsReproducibleAndSeedsDiffer() throws {
        let p = SamplingParameters(temperature: 1, seed: 42)
        var a = try Sampler(p), b = try Sampler(p), c = try Sampler(SamplingParameters(temperature: 1, seed: 43))
        let sa = try (0..<200).map { _ in try a.sample(logits: logits) }
        let sb = try (0..<200).map { _ in try b.sample(logits: logits) }
        let sc = try (0..<200).map { _ in try c.sample(logits: logits) }
        XCTAssertEqual(sa, sb)
        XCTAssertNotEqual(sa, sc)
    }

    /// Empirical frequencies must match softmax(logits / T). With n = 20000 the standard error of a probability p
    /// is sqrt(p(1-p)/n) ≤ 0.0035, so a 0.015 tolerance is > 4 sigma.
    func testTemperatureDistributionMatchesSoftmax() throws {
        for temperature: Float in [0.5, 1, 2] {
            let counts = try draws(SamplingParameters(temperature: temperature, seed: 7))
            let w = logits.map { exp(Double($0) / Double(temperature)) }, z = w.reduce(0, +)
            for i in logits.indices {
                XCTAssertEqual(Double(counts[i] ?? 0) / 20_000, w[i] / z, accuracy: 0.015, "T=\(temperature) token \(i)")
            }
        }
    }

    func testTopKRestrictsSupportAndRenormalises() throws {
        let counts = try draws(SamplingParameters(temperature: 1, topK: 2, seed: 1))
        XCTAssertEqual(Set(counts.keys), [1, 4])                       // the two tied maxima
        XCTAssertEqual(Double(counts[1]!) / 20_000, 0.5, accuracy: 0.015)
        XCTAssertEqual(Set(try draws(SamplingParameters(temperature: 5, topK: 1, seed: 1)).keys), [1])
    }

    func testTopPKeepsSmallestNucleus() throws {
        // softmax(T=1): tokens 1,4 ≈ 0.389 each, token 2 ≈ 0.143 → cumulative 0.389, 0.778, 0.921.
        // topP 0.3 keeps {1}; 0.6 keeps {1,4}; 0.8 also needs token 2.
        XCTAssertEqual(Set(try draws(SamplingParameters(temperature: 1, topP: 0.3, seed: 3)).keys), [1])
        XCTAssertEqual(Set(try draws(SamplingParameters(temperature: 1, topP: 0.6, seed: 3)).keys), [1, 4])
        XCTAssertEqual(Set(try draws(SamplingParameters(temperature: 1, topP: 0.8, seed: 3)).keys), [1, 2, 4])
        XCTAssertEqual(Set(try draws(SamplingParameters(temperature: 1, topP: 1, seed: 3)).keys), Set(0..<6))
    }

    func testInvalidInputsAreRejected() throws {
        XCTAssertThrowsError(try Sampler(SamplingParameters(temperature: -1)))
        XCTAssertThrowsError(try Sampler(SamplingParameters(temperature: .nan)))
        XCTAssertThrowsError(try Sampler(SamplingParameters(topK: -1)))
        XCTAssertThrowsError(try Sampler(SamplingParameters(topP: 0)))
        XCTAssertThrowsError(try Sampler(SamplingParameters(topP: 1.5)))
        var s = try Sampler(.greedy)
        XCTAssertThrowsError(try s.sample(logits: []))
        XCTAssertThrowsError(try s.sample(logits: [1, .nan]))
    }

    func testExtremeLogitsDoNotOverflow() throws {
        var s = try Sampler(SamplingParameters(temperature: 0.1, seed: 1))
        XCTAssertEqual(try s.sample(logits: [1e30, 0, -1e30]), 0)
    }

    func testMemoryEstimateAndBudget() {
        let c = LlamaConfig(vocabSize: 49152, hiddenSize: 576, layerCount: 30, headCount: 9, kvHeadCount: 3, feedForwardSize: 1536, contextLength: 8192, rmsNormEps: 1e-5, ropeTheta: 1e5)
        let e = MemoryEstimate(weightBytes: 270_885_952, config: c, contextLength: 2048, kvBytesPerElement: 2, prefillBatch: 128)
        XCTAssertEqual(e.kvCacheBytes, 2 * 30 * 2048 * 192 * 2)
        XCTAssertEqual(e.totalBytes, e.weightBytes + e.kvCacheBytes + e.scratchBytes + e.runtimeOverheadBytes)
        XCTAssertGreaterThan(MemoryBudget.defaultBytes(), 0)
    }
}
