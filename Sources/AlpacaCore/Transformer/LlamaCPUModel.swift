// Copyright (c) 2026 Torin Etheridge. MIT License (see LICENSE).
// Author: Torin Etheridge · 2026-10-09

import Foundation

/// How many positions' logits a forward pass returns.
public enum LogitsSelection: Sendable { case all, last }

/// CPU reference executor for the Llama decoder. Correctness-first; used for validation and as the
/// no-Metal fallback. Handles both prefill (many tokens) and decode (one token) with the same code path:
/// `tokens` are appended to `cache` at positions `cache.length ..< cache.length + tokens.count`.
public final class LlamaCPUModel: @unchecked Sendable {
    public let config: LlamaConfig
    public let weights: LlamaWeights

    public init(config: LlamaConfig, weights: LlamaWeights) throws {
        try config.validate()
        try weights.validate(against: config)
        self.config = config; self.weights = weights
    }

    /// Runs the transformer on `tokens`, updating `cache`, and returns f32 logits [n, vocab]
    /// (n = tokens.count for `.all`, 1 for `.last`).
    public func forward(tokens: [Int32], cache: KVCache, logits selection: LogitsSelection = .last) throws -> Tensor {
        let c = config
        guard !tokens.isEmpty else { throw TensorError.shapeMismatch("forward called with no tokens") }
        guard cache.layerCount == c.layerCount, cache.kvHeadCount == c.kvHeadCount, cache.headDim == c.headDim else {
            throw TensorError.shapeMismatch("KV cache geometry does not match model")
        }
        let start = cache.length, n = tokens.count
        guard start + n <= cache.capacity else {
            throw TensorError.outOfBounds("context exhausted: \(start) cached + \(n) new > capacity \(cache.capacity)")
        }

        var x = try CPUOps.embed(tokens: tokens, table: weights.tokenEmbedding)           // [n, hidden]
        for (li, l) in weights.layers.enumerated() {
            // Attention block: x += Wo · attn(norm(x))
            let h = try CPUOps.rmsNorm(x, weight: l.attnNorm, eps: c.rmsNormEps)
            var q = try CPUOps.linear(h, weight: l.wq).reshaped([n, c.headCount, c.headDim])
            var k = try CPUOps.linear(h, weight: l.wk).reshaped([n, c.kvHeadCount, c.headDim])
            let v = try CPUOps.linear(h, weight: l.wv).reshaped([n, c.kvHeadCount, c.headDim])
            q = try CPUOps.rope(q, startPosition: start, theta: c.ropeTheta, interleaved: c.ropeInterleaved)
            k = try CPUOps.rope(k, startPosition: start, theta: c.ropeTheta, interleaved: c.ropeInterleaved)
            try cache.write(layer: li, position: start, keys: k, values: v)
            let scores = try CPUOps.attentionScores(q: q, keys: cache.keys[li], length: start + n, startPosition: start)
            let probs = try CPUOps.softmax(scores)
            let context = try CPUOps.attentionApply(probs: probs, values: cache.values[li]).reshaped([n, c.queryWidth])
            x = try CPUOps.add(x, CPUOps.linear(context, weight: l.wo))

            // Feed-forward block (SwiGLU): x += Wdown · (silu(Wgate·h) ⊙ Wup·h)
            let f = try CPUOps.rmsNorm(x, weight: l.ffnNorm, eps: c.rmsNormEps)
            let gate = try CPUOps.silu(CPUOps.linear(f, weight: l.wGate))
            let up = try CPUOps.linear(f, weight: l.wUp)
            x = try CPUOps.add(x, CPUOps.linear(CPUOps.mul(gate, up), weight: l.wDown))
        }
        try cache.commit(length: start + n)

        var final = try CPUOps.rmsNorm(x, weight: weights.outputNorm, eps: c.rmsNormEps)
        if case .last = selection, n > 1 { final = try final.slice(axis: 0, (n - 1)..<n) }
        return try CPUOps.linear(final.contiguous(), weight: weights.outputProjection)
    }
}
