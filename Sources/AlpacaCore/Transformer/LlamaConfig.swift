// Copyright (c) 2026 Torin Etheridge. MIT License (see LICENSE).
// Author: Torin Etheridge · 2026-10-09

/// Architecture hyper-parameters of a Llama-style decoder-only transformer.
///
/// The exact variant supported (see Docs/MODEL_SUPPORT.md): pre-norm RMSNorm, GQA attention with RoPE
/// (no scaling), SwiGLU feed-forward, no biases, final RMSNorm, optionally tied input/output embeddings.
public struct LlamaConfig: Sendable, Equatable {
    public var vocabSize: Int
    public var hiddenSize: Int
    public var layerCount: Int
    public var headCount: Int
    public var kvHeadCount: Int
    public var headDim: Int
    public var feedForwardSize: Int
    public var contextLength: Int
    public var rmsNormEps: Float
    public var ropeTheta: Float
    /// true: rotate adjacent pairs (GGUF/llama.cpp weight layout). false: split halves (Hugging Face layout).
    public var ropeInterleaved: Bool
    /// Per-pair divisors applied to the RoPE frequencies (`headDim / 2` values): angleᵢ = position · theta^(-2i/headDim) / factorᵢ.
    /// Llama 3.1 / 3.2 ship their long-context ("llama3") scaling as this table (`rope_freqs.weight` in GGUF). nil = no scaling.
    public var ropeFreqFactors: [Float]?

    public init(
        vocabSize: Int, hiddenSize: Int, layerCount: Int, headCount: Int, kvHeadCount: Int, headDim: Int? = nil,
        feedForwardSize: Int, contextLength: Int, rmsNormEps: Float, ropeTheta: Float, ropeInterleaved: Bool = true, ropeFreqFactors: [Float]? = nil
    ) {
        self.ropeFreqFactors = ropeFreqFactors
        self.vocabSize = vocabSize; self.hiddenSize = hiddenSize; self.layerCount = layerCount
        self.headCount = headCount; self.kvHeadCount = kvHeadCount
        self.headDim = headDim ?? (headCount > 0 ? hiddenSize / headCount : 0)
        self.feedForwardSize = feedForwardSize; self.contextLength = contextLength
        self.rmsNormEps = rmsNormEps; self.ropeTheta = ropeTheta; self.ropeInterleaved = ropeInterleaved
    }

    public var queryWidth: Int { headCount * headDim }
    public var kvWidth: Int { kvHeadCount * headDim }

    /// Rejects configurations the engine cannot execute correctly.
    public func validate() throws {
        func fail(_ m: String) -> ConfigError { ConfigError(m) }
        guard vocabSize > 0, hiddenSize > 0, layerCount > 0, headCount > 0, kvHeadCount > 0, headDim > 0,
            feedForwardSize > 0, contextLength > 0 else { throw fail("all dimensions must be positive: \(self)") }
        guard headCount % kvHeadCount == 0 else { throw fail("headCount \(headCount) is not a multiple of kvHeadCount \(kvHeadCount)") }
        guard headDim % 2 == 0 else { throw fail("RoPE needs an even headDim, got \(headDim)") }
        if let f = ropeFreqFactors, f.count != headDim / 2 || f.contains(where: { !($0 > 0) || !$0.isFinite }) {
            throw fail("ropeFreqFactors must hold \(headDim / 2) positive finite values")
        }
        guard rmsNormEps > 0, ropeTheta > 0 else { throw fail("rmsNormEps and ropeTheta must be positive") }
        // Overflow guards for sizes derived from untrusted metadata.
        for (name, v) in [("vocabSize", vocabSize), ("hiddenSize", hiddenSize), ("feedForwardSize", feedForwardSize),
                          ("layerCount", layerCount), ("contextLength", contextLength)] where v > 1 << 28 {
            throw fail("\(name) \(v) is implausibly large")
        }
    }
}

public struct ConfigError: Error, CustomStringConvertible {
    public let description: String
    public init(_ d: String) { description = d }
}
