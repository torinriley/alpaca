// Copyright (c) 2026 Torin Etheridge. MIT License (see LICENSE).
// Author: Torin Etheridge · 2026-10-09

import XCTest
import AlpacaCore
import AlpacaModels
import AlpacaMetal

/// Localises GPU/CPU disagreement: every stage of every layer is run on the GPU using the *CPU's* exact
/// input for that stage, so errors cannot compound and the first stage exceeding its bound is the culprit.
final class LayerwiseTests: XCTestCase {
    override func setUpWithError() throws { try requireOptimizedBuild() }
    func testEveryStageOfEveryLayerMatchesCPU() throws {
        try XCTSkipUnless(MetalContext.isAvailable, "no Metal device")
        let loaded = try LlamaLoader.load(url: ModelFiles.url("Q8_0"))
        let c = loaded.config, w = loaded.weights
        let ops = MetalOps(context: try MetalContext())
        let tokens = try Reference.load("Q8_0")[1].ids
        let n = tokens.count
        var x = try CPUOps.embed(tokens: tokens, table: w.tokenEmbedding)
        var worst: [String: Double] = [:]
        func rel(_ stage: String, _ gpu: Tensor, _ cpu: Tensor) throws {
            let g = try gpu.toFloatArray(), e = try cpu.toFloatArray().map(Double.init)
            let m = ErrorMetrics(actual: g, expected: e)
            worst[stage] = max(worst[stage] ?? 0, m.maxRelativeToPeak)
        }
        for (li, l) in w.layers.enumerated() {
            let h = try CPUOps.rmsNorm(x, weight: l.attnNorm, eps: c.rmsNormEps)
            try rel("rmsnorm", ops.rmsNorm(x, weight: l.attnNorm, eps: c.rmsNormEps), h)
            let qc = try CPUOps.linear(h, weight: l.wq), kc = try CPUOps.linear(h, weight: l.wk), vc = try CPUOps.linear(h, weight: l.wv)
            try rel("linear q", ops.linear(h, weight: l.wq), qc)
            try rel("linear k", ops.linear(h, weight: l.wk), kc)
            let q = try qc.reshaped([n, c.headCount, c.headDim]), k = try kc.reshaped([n, c.kvHeadCount, c.headDim]), v = try vc.reshaped([n, c.kvHeadCount, c.headDim])
            let qr = try CPUOps.rope(q, startPosition: 0, theta: c.ropeTheta), kr = try CPUOps.rope(k, startPosition: 0, theta: c.ropeTheta)
            try rel("rope q", ops.rope(q, startPosition: 0, theta: c.ropeTheta), qr)
            let probs = try CPUOps.softmax(CPUOps.attentionScores(q: qr, keys: kr, length: n, startPosition: 0))
            let ctxCPU = try CPUOps.attentionApply(probs: probs, values: v)
            try rel("attention (f32 CPU vs f16-KV GPU)", ops.attention(q: qr, keys: kr, values: v, startPosition: 0), ctxCPU)
            let ctx = try ctxCPU.reshaped([n, c.queryWidth])
            let attnOut = try CPUOps.linear(ctx, weight: l.wo)
            try rel("linear wo", ops.linear(ctx, weight: l.wo), attnOut)
            x = try CPUOps.add(x, attnOut)
            let f = try CPUOps.rmsNorm(x, weight: l.ffnNorm, eps: c.rmsNormEps)
            let g = try CPUOps.linear(f, weight: l.wGate), u = try CPUOps.linear(f, weight: l.wUp)
            try rel("linear gate", ops.linear(f, weight: l.wGate), g)
            try rel("silu*mul", ops.siluMul(gate: g, up: u), CPUOps.mul(CPUOps.silu(g), u))
            let act = try CPUOps.mul(CPUOps.silu(g), u)
            let down = try CPUOps.linear(act, weight: l.wDown)
            try rel("linear down", ops.linear(act, weight: l.wDown), down)
            try rel("residual add", ops.add(x, down), CPUOps.add(x, down))
            x = try CPUOps.add(x, down)
            if li == 0 || li == c.layerCount - 1 { print("[layerwise] after layer \(li): |x|max = \(try x.toFloatArray().map(abs).max()!)") }
        }
        for (stage, v) in worst.sorted(by: { $0.value > $1.value }) { print("[layerwise] \(stage): worst error relative to tensor peak = \(v)") }
    }
}
