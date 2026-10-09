// Copyright (c) 2026 Torin Etheridge. MIT License (see LICENSE).
// Author: Torin Etheridge · 2026-10-09

import Foundation
import AlpacaCore
import AlpacaMetal

public typealias GEMMPrecision = AlpacaMetal.GEMMPrecision
public typealias KVPrecision = AlpacaMetal.KVPrecision

/// Which compute backend executes the transformer.
public enum Backend: Sendable, Equatable {
    /// Metal when a device exists, otherwise the CPU reference backend.
    case automatic
    case metal
    /// Deterministic but slow reference implementation; intended for testing and machines without Metal.
    case cpu
}

public struct LoadOptions: Sendable {
    public var backend: Backend = .automatic
    /// Maximum positions per generation (KV cache size). nil → min(model context, 4096).
    public var contextLength: Int? = nil
    /// Upper bound on estimated memory (weights + KV cache + scratch + runtime). nil → `MemoryBudget.defaultBytes()`.
    public var memoryBudgetBytes: Int? = nil
    /// GPU KV cache precision. `.float16` halves cache memory and adds ~2^-11 relative rounding per cached value.
    public var kvPrecision: KVPrecision = .float16
    /// Tokens processed per prefill chunk (bounds scratch memory).
    public var prefillBatchSize: Int = 512
    /// `.fast` runs batched projections on the GPU's matrix hardware where available (half-precision operands, float32
    /// accumulation, ~2.5× faster prefill on M5); `.exact` keeps float32 operands. Decode is unaffected.
    public var prefillPrecision: GEMMPrecision = .fast
    public init() {}
}

public struct GenerationConfiguration: Sendable {
    public var maxTokens: Int
    /// 0 → greedy.
    public var temperature: Float
    public var topK: Int
    public var topP: Float
    /// Fixed seed → reproducible output for identical model, prompt and backend.
    public var seed: UInt64?
    /// Stop (without emitting) when the model produces its end-of-sequence token or any of `stopTokens`.
    public var stopOnEndOfSequence: Bool
    public var stopTokens: Set<Int32>
    /// Recognise literal special-token strings such as `<|im_start|>` in the prompt.
    public var parseSpecialTokens: Bool
    /// Prepend BOS if the model's metadata asks for it.
    public var addBeginningOfSequence: Bool

    public init(maxTokens: Int = 256, temperature: Float = 0.8, topK: Int = 0, topP: Float = 1, seed: UInt64? = nil,
                stopOnEndOfSequence: Bool = true, stopTokens: Set<Int32> = [], parseSpecialTokens: Bool = true,
                addBeginningOfSequence: Bool = true) {
        self.maxTokens = maxTokens; self.temperature = temperature; self.topK = topK; self.topP = topP; self.seed = seed
        self.stopOnEndOfSequence = stopOnEndOfSequence; self.stopTokens = stopTokens
        self.parseSpecialTokens = parseSpecialTokens; self.addBeginningOfSequence = addBeginningOfSequence
    }

    var sampling: SamplingParameters { SamplingParameters(temperature: temperature, topK: topK, topP: topP, seed: seed) }
}
