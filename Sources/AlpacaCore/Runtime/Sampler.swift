// Copyright (c) 2026 Torin Etheridge. MIT License (see LICENSE).
// Author: Torin Etheridge · 2026-10-09

import Foundation

public struct SamplingParameters: Sendable, Equatable {
    /// 0 selects greedy decoding (argmax, lowest index wins ties).
    public var temperature: Float
    /// Keep only the k most likely tokens; 0 disables.
    public var topK: Int
    /// Nucleus sampling: keep the smallest set of tokens whose probability mass reaches p; 1 disables.
    public var topP: Float
    /// Seed for the sampling RNG. nil draws a fresh seed.
    public var seed: UInt64?

    public init(temperature: Float = 0.8, topK: Int = 0, topP: Float = 1, seed: UInt64? = nil) {
        self.temperature = temperature; self.topK = topK; self.topP = topP; self.seed = seed
    }

    public static let greedy = SamplingParameters(temperature: 0)

    public func validate() throws {
        guard temperature.isFinite, temperature >= 0 else { throw ConfigError("temperature must be finite and >= 0") }
        guard topK >= 0 else { throw ConfigError("topK must be >= 0") }
        guard topP > 0, topP <= 1 else { throw ConfigError("topP must be in (0, 1]") }
    }
}

/// SplitMix64: tiny, fast, statistically solid for sampling, and fully reproducible from a 64-bit seed.
public struct SplitMix64: RandomNumberGenerator, Sendable {
    private var state: UInt64
    public init(seed: UInt64) { state = seed }
    public mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}

/// Picks the next token from logits. Order of operations: top-k filter → temperature softmax (float64) →
/// top-p filter → renormalise → inverse-CDF draw. A sampler owns its RNG, so equal seeds give equal streams.
public struct Sampler: Sendable {
    public let parameters: SamplingParameters
    private var rng: SplitMix64

    public init(_ parameters: SamplingParameters) throws {
        try parameters.validate()
        self.parameters = parameters
        self.rng = SplitMix64(seed: parameters.seed ?? UInt64.random(in: .min ... .max))
    }

    public mutating func sample(logits: [Float]) throws -> Int32 {
        try logits.withUnsafeBufferPointer { try sample(logits: $0) }
    }

    /// Samples straight from a logits buffer (e.g. GPU shared memory) without copying it.
    public mutating func sample(logits: UnsafeBufferPointer<Float>) throws -> Int32 {
        guard !logits.isEmpty else { throw ConfigError("cannot sample from empty logits") }
        if parameters.temperature == 0 { return try Self.argmax(logits) }
        guard !logits.contains(where: { $0.isNaN }) else { throw ConfigError("logits contain NaN") }

        var candidates = Array(logits.indices)
        // Descending by logit; index breaks ties so the order is deterministic.
        let k = parameters.topK > 0 ? min(parameters.topK, logits.count) : logits.count
        if k < logits.count {
            candidates.sort { logits[$0] != logits[$1] ? logits[$0] > logits[$1] : $0 < $1 }
            candidates.removeSubrange(k...)
        } else if parameters.topP < 1 {
            candidates.sort { logits[$0] != logits[$1] ? logits[$0] > logits[$1] : $0 < $1 }
        }
        let t = Double(parameters.temperature)
        let top = candidates.map { Double(logits[$0]) }.max()!
        var probs = candidates.map { exp((Double(logits[$0]) - top) / t) }
        var sum = probs.reduce(0, +)
        if parameters.topP < 1 {
            var cumulative = 0.0, keep = probs.count
            for (i, p) in probs.enumerated() {
                cumulative += p / sum
                if cumulative >= Double(parameters.topP) { keep = i + 1; break }
            }
            candidates.removeSubrange(keep...); probs.removeSubrange(keep...)
            sum = probs.reduce(0, +)
        }
        let u = Double.random(in: 0..<1, using: &rng) * sum
        var acc = 0.0
        for (i, p) in probs.enumerated() {
            acc += p
            if u < acc { return Int32(candidates[i]) }
        }
        return Int32(candidates[candidates.count - 1])
    }

    public static func argmax(_ logits: [Float]) -> Int32 {
        logits.withUnsafeBufferPointer { (try? argmax($0)) ?? 0 }
    }

    /// Index of the largest logit (lowest index wins ties). Throws on NaN, which would otherwise select silently.
    public static func argmax(_ logits: UnsafeBufferPointer<Float>) throws -> Int32 {
        var best = 0
        for i in 0..<logits.count {
            let v = logits[i]
            if v.isNaN { throw ConfigError("logits contain NaN") }
            if v > logits[best] { best = i }
        }
        return Int32(best)
    }
}
