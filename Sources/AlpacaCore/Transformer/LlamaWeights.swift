// Copyright (c) 2026 Torin Etheridge. MIT License (see LICENSE).
// Author: Torin Etheridge · 2026-10-09

/// Weights of one decoder layer. Projection matrices are [out, in] (y = x·Wᵀ); norms are f32 [hidden].
public struct LlamaLayerWeights: Sendable {
    public var attnNorm: Tensor
    public var wq: Tensor, wk: Tensor, wv: Tensor, wo: Tensor
    public var ffnNorm: Tensor
    public var wGate: Tensor, wUp: Tensor, wDown: Tensor

    public init(attnNorm: Tensor, wq: Tensor, wk: Tensor, wv: Tensor, wo: Tensor,
                ffnNorm: Tensor, wGate: Tensor, wUp: Tensor, wDown: Tensor) {
        self.attnNorm = attnNorm; self.wq = wq; self.wk = wk; self.wv = wv; self.wo = wo
        self.ffnNorm = ffnNorm; self.wGate = wGate; self.wUp = wUp; self.wDown = wDown
    }
}

public struct LlamaWeights: Sendable {
    /// [vocab, hidden]; any dtype.
    public var tokenEmbedding: Tensor
    public var layers: [LlamaLayerWeights]
    public var outputNorm: Tensor
    /// [vocab, hidden]. nil means tied to `tokenEmbedding`.
    public var output: Tensor?

    public init(tokenEmbedding: Tensor, layers: [LlamaLayerWeights], outputNorm: Tensor, output: Tensor?) {
        self.tokenEmbedding = tokenEmbedding; self.layers = layers; self.outputNorm = outputNorm; self.output = output
    }

    public var outputProjection: Tensor { output ?? tokenEmbedding }

    /// Verifies every tensor has the shape and a supported dtype for `config`.
    public func validate(against c: LlamaConfig) throws {
        func check(_ t: Tensor, _ shape: [Int], _ name: String, norm: Bool = false) throws {
            guard t.shape == shape else { throw ConfigError("\(name): expected shape \(shape), found \(t.shape)") }
            guard t.isContiguous else { throw ConfigError("\(name): weights must be contiguous") }
            if norm, t.dtype != .float32 { throw ConfigError("\(name): norm weights must be f32, found \(t.dtype)") }
        }
        try check(tokenEmbedding, [c.vocabSize, c.hiddenSize], "token_embd")
        try check(outputNorm, [c.hiddenSize], "output_norm", norm: true)
        if let output { try check(output, [c.vocabSize, c.hiddenSize], "output") }
        guard layers.count == c.layerCount else { throw ConfigError("expected \(c.layerCount) layers, found \(layers.count)") }
        for (i, l) in layers.enumerated() {
            try check(l.attnNorm, [c.hiddenSize], "blk.\(i).attn_norm", norm: true)
            try check(l.wq, [c.queryWidth, c.hiddenSize], "blk.\(i).attn_q")
            try check(l.wk, [c.kvWidth, c.hiddenSize], "blk.\(i).attn_k")
            try check(l.wv, [c.kvWidth, c.hiddenSize], "blk.\(i).attn_v")
            try check(l.wo, [c.hiddenSize, c.queryWidth], "blk.\(i).attn_output")
            try check(l.ffnNorm, [c.hiddenSize], "blk.\(i).ffn_norm", norm: true)
            try check(l.wGate, [c.feedForwardSize, c.hiddenSize], "blk.\(i).ffn_gate")
            try check(l.wUp, [c.feedForwardSize, c.hiddenSize], "blk.\(i).ffn_up")
            try check(l.wDown, [c.hiddenSize, c.feedForwardSize], "blk.\(i).ffn_down")
        }
    }

    /// Total bytes of weight storage referenced (shared tied embeddings counted once).
    public var byteCount: Int {
        var total = 0
        func size(_ t: Tensor) -> Int { (t.elementCount / t.dtype.blockElements) * t.dtype.blockBytes }
        total += size(tokenEmbedding) + size(outputNorm) + (output.map(size) ?? 0)
        for l in layers {
            total += [l.attnNorm, l.wq, l.wk, l.wv, l.wo, l.ffnNorm, l.wGate, l.wUp, l.wDown].reduce(0) { $0 + size($1) }
        }
        return total
    }
}
