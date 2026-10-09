// Copyright (c) 2026 Torin Etheridge. MIT License (see LICENSE).
// Author: Torin Etheridge · 2026-10-09

import Foundation
import Metal
import AlpacaCore

/// Tensor-in / tensor-out wrappers around the GPU kernels. Each call uploads its operands, runs the kernel
/// and reads the result back, so every kernel can be validated against `CPUOps` in isolation.
/// (The transformer path in `MetalLlamaSession` encodes the same kernels into one command buffer without round trips.)
public final class MetalOps: @unchecked Sendable {
    public let context: MetalContext
    public init(context: MetalContext) { self.context = context }

    /// Runs `body` on a fresh serial compute encoder and waits. Returns GPU execution time in seconds.
    @discardableResult
    func run(precision: GEMMPrecision = .exact, scratch: MTLBuffer? = nil, scratch2: MTLBuffer? = nil, _ body: (KernelEncoder) throws -> Void) throws -> Double {
        guard let cmd = context.queue.makeCommandBuffer(), let enc = cmd.makeComputeCommandEncoder() else {
            throw MetalError.commandFailed("could not create command buffer")
        }
        var kernels = KernelEncoder(context: context, encoder: enc)
        kernels.precision = precision; kernels.dequantScratch = scratch; kernels.dequantScratch2 = scratch2
        try body(kernels)
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
        if cmd.status != .completed { throw MetalError.commandFailed(cmd.error.map { "\($0)" } ?? "status \(cmd.status.rawValue)") }
        return cmd.gpuEndTime - cmd.gpuStartTime
    }

    func buffer(_ t: Tensor) throws -> MTLBuffer {
        guard t.isContiguous else { throw MetalError.invalidArgument("tensor must be contiguous") }
        let bytes = (t.elementCount / t.dtype.blockElements) * t.dtype.blockBytes
        return try context.makeBuffer(copying: t.basePointer, length: bytes)
    }

    func read(_ b: MTLBuffer, shape: [Int]) throws -> Tensor {
        let t = try Tensor(zeros: shape)
        memcpy(t.storage.pointer, b.contents(), t.elementCount * 4)
        return t
    }

    /// Half-precision copy of a float tensor, as the tensor-op GEMM expects its activations.
    func halfBuffer(_ t: Tensor) throws -> MTLBuffer {
        let h = try t.converted(to: .float16)
        return try context.makeBuffer(copying: h.basePointer, length: h.elementCount * 2)
    }

    func readHalf(_ b: MTLBuffer, shape: [Int]) throws -> Tensor {
        let h = try Tensor(zeros: shape, dtype: .float16)
        memcpy(h.storage.pointer, b.contents(), h.elementCount * 2)
        return try h.converted(to: .float32)
    }

    func eligibleForTensorGEMM(_ w: GPUMatrix, tokens: Int, precision: GEMMPrecision, scratch: MTLBuffer?) -> Bool {
        KernelEncoder.canUseTensorGEMM(w, tokens: tokens, precision: precision, context: context, scratch: scratch)
    }

    func requireF32(_ ts: Tensor...) throws {
        for t in ts where t.dtype != .float32 || !t.isContiguous { throw MetalError.invalidArgument("operands must be contiguous f32") }
    }

    public func add(_ a: Tensor, _ b: Tensor) throws -> Tensor {
        try requireF32(a, b)
        guard a.shape == b.shape else { throw MetalError.invalidArgument("shape mismatch \(a.shape) vs \(b.shape)") }
        let out = try context.makeBuffer(length: a.elementCount * 4)
        let (ba, bb) = (try buffer(a), try buffer(b))
        try run { try $0.add(ba, bb, out: out, count: a.elementCount) }
        return try read(out, shape: a.shape)
    }

    public func mul(_ a: Tensor, _ b: Tensor) throws -> Tensor {
        try requireF32(a, b)
        guard a.shape == b.shape else { throw MetalError.invalidArgument("shape mismatch \(a.shape) vs \(b.shape)") }
        let out = try context.makeBuffer(length: a.elementCount * 4)
        let (ba, bb) = (try buffer(a), try buffer(b))
        try run { try $0.mul(ba, bb, out: out, count: a.elementCount) }
        return try read(out, shape: a.shape)
    }

    public func silu(_ x: Tensor) throws -> Tensor {
        try requireF32(x)
        let out = try context.makeBuffer(length: x.elementCount * 4)
        let bx = try buffer(x)
        try run { try $0.silu(bx, out: out, count: x.elementCount) }
        return try read(out, shape: x.shape)
    }

    public func siluMul(gate: Tensor, up: Tensor) throws -> Tensor {
        try requireF32(gate, up)
        guard gate.shape == up.shape else { throw MetalError.invalidArgument("shape mismatch") }
        let out = try context.makeBuffer(length: gate.elementCount * 4)
        let (g, u) = (try buffer(gate), try buffer(up))
        try run { try $0.siluMul(gate: g, up: u, out: out, count: gate.elementCount) }
        return try read(out, shape: gate.shape)
    }

    public func rmsNorm(_ x: Tensor, weight: Tensor, eps: Float) throws -> Tensor {
        try requireF32(x, weight)
        guard let d = x.shape.last, weight.shape == [d], eps > 0 else { throw MetalError.invalidArgument("rmsNorm shapes \(x.shape) / \(weight.shape)") }
        let out = try context.makeBuffer(length: x.elementCount * 4)
        let (bx, bw) = (try buffer(x), try buffer(weight))
        try run { try $0.rmsNorm(bx, weight: bw, out: out, rows: x.elementCount / d, dim: d, eps: eps) }
        return try read(out, shape: x.shape)
    }

    /// y[M,N] = x[M,K] · W[N,K]ᵀ with W in f16 / q8_0 / q4_0.
    public func linear(_ x: Tensor, weight w: Tensor, precision: GEMMPrecision = .exact) throws -> Tensor {
        try requireF32(x)
        guard x.rank == 2, w.rank == 2, x.shape[1] == w.shape[1] else { throw MetalError.invalidArgument("linear shapes \(x.shape) / \(w.shape)") }
        guard w.dtype != .float32, w.isContiguous else { throw MetalError.unsupported("weight dtype \(w.dtype)") }
        let (m, k, n) = (x.shape[0], x.shape[1], w.shape[0])
        guard k % (w.dtype.isQuantized ? 32 : 8) == 0 else { throw MetalError.invalidArgument("K=\(k) not a multiple of the kernel unit") }
        let out = try context.makeBuffer(length: m * n * 4)
        let bx = try buffer(x), bw = try buffer(w)
        let gw = GPUMatrix(buffer: bw, offset: 0, dtype: w.dtype, rows: n, cols: k)
        let scratch = (precision == .fast && w.dtype.isQuantized) ? try context.makeBuffer(length: n * k * 2) : nil
        if eligibleForTensorGEMM(gw, tokens: m, precision: precision, scratch: scratch) {
            let xh = try halfBuffer(x)
            try run(precision: precision, scratch: scratch) { try $0.projectHalf(gw, xHalf: xh, y: out, tokens: m, accumulate: false) }
        } else {
            try run(precision: precision, scratch: scratch) { try $0.linear(gw, x: bx, y: out, tokens: m) }
        }
        return try read(out, shape: [m, n])
    }

    /// residual += x · Wᵀ (the fused projection+residual used by the transformer). Returns the updated residual.
    public func linearAdd(_ x: Tensor, weight w: Tensor, residual: Tensor, precision: GEMMPrecision = .exact) throws -> Tensor {
        try requireF32(x, residual)
        guard x.rank == 2, w.rank == 2, residual.shape == [x.shape[0], w.shape[0]], x.shape[1] == w.shape[1] else { throw MetalError.invalidArgument("linearAdd shapes") }
        let (m, k, n) = (x.shape[0], x.shape[1], w.shape[0])
        let bx = try buffer(x), bw = try buffer(w), res = try buffer(residual)
        let scratch = w.dtype.isQuantized ? try context.makeBuffer(length: n * k * 2) : nil
        let gw = GPUMatrix(buffer: bw, offset: 0, dtype: w.dtype, rows: n, cols: k)
        if eligibleForTensorGEMM(gw, tokens: m, precision: precision, scratch: scratch) {
            let xh = try halfBuffer(x)
            try run(precision: precision, scratch: scratch) { try $0.projectHalf(gw, xHalf: xh, y: res, tokens: m, accumulate: true) }
        } else {
            try run(precision: precision, scratch: scratch) { try $0.linearAdd(gw, x: bx, residual: res, tokens: m) }
        }
        return try read(res, shape: [m, n])
    }

    /// silu(x·Wgᵀ) ⊙ (x·Wuᵀ) — the SwiGLU feed-forward input (fused into one kernel on the tensor-op path).
    public func gateUpSilu(_ x: Tensor, gate wg: Tensor, up wu: Tensor, precision: GEMMPrecision = .exact) throws -> Tensor {
        try requireF32(x)
        guard x.rank == 2, wg.rank == 2, wg.shape == wu.shape, wg.dtype == wu.dtype, x.shape[1] == wg.shape[1], wg.dtype != .float32 else {
            throw MetalError.invalidArgument("gateUpSilu shapes/dtypes")
        }
        let (m, k, n) = (x.shape[0], x.shape[1], wg.shape[0])
        let bx = try buffer(x), bg = try buffer(wg), bu = try buffer(wu)
        let out = try context.makeBuffer(length: m * n * 4), up = try context.makeBuffer(length: m * n * 4)
        let s1 = wg.dtype.isQuantized ? try context.makeBuffer(length: n * k * 2) : nil
        let s2 = wg.dtype.isQuantized ? try context.makeBuffer(length: n * k * 2) : nil
        let gg = GPUMatrix(buffer: bg, offset: 0, dtype: wg.dtype, rows: n, cols: k), gu = GPUMatrix(buffer: bu, offset: 0, dtype: wu.dtype, rows: n, cols: k)
        if wg.dtype == wu.dtype, eligibleForTensorGEMM(gg, tokens: m, precision: precision, scratch: s1) {
            let xh = try halfBuffer(x), oh = try context.makeBuffer(length: m * n * 2)
            try run(precision: precision, scratch: s1, scratch2: s2) { try $0.gateUpSiluHalf(gg, gu, xHalf: xh, outHalf: oh, tokens: m) }
            return try readHalf(oh, shape: [m, n])
        }
        try run(precision: precision, scratch: s1, scratch2: s2) { try $0.gateUpSilu(gg, gu, x: bx, out: out, scratchUp: up, tokens: m) }
        return try read(out, shape: [m, n])
    }

    public func rope(_ x: Tensor, startPosition: Int, theta: Float) throws -> Tensor {
        try requireF32(x)
        guard x.rank == 3, x.shape[2] % 2 == 0 else { throw MetalError.invalidArgument("rope shape \(x.shape)") }
        let b = try buffer(x)
        let f = try context.makeBuffer(copying: ropeFrequencies(headDim: x.shape[2], theta: theta), length: x.shape[2] / 2 * 4)
        try run { try $0.rope(b, freqs: f, heads: x.shape[1], headDim: x.shape[2], startPosition: startPosition, tokens: x.shape[0]) }
        return try read(b, shape: x.shape)
    }

    /// Full causal GQA attention as one kernel (online softmax). Keys/values are f32 on the way in and are
    /// rounded to f16 for the cache when `kv == .float16`, exactly as in the model path.
    /// q: [tokens, heads, headDim]; keys/values: [capacity, kvHeads, headDim]. Returns [tokens, heads, headDim].
    public func attention(q: Tensor, keys: Tensor, values: Tensor, startPosition: Int, kv: KVPrecision = .float16) throws -> Tensor {
        try requireF32(q, keys, values)
        guard q.rank == 3, keys.shape == values.shape, keys.rank == 3, keys.shape[2] == q.shape[2], q.shape[1] % keys.shape[1] == 0,
              q.shape[2] % 32 == 0, q.shape[2] <= 256, startPosition + q.shape[0] <= keys.shape[0]
        else { throw MetalError.invalidArgument("attention shapes q \(q.shape) k \(keys.shape)") }
        let toCache = { (t: Tensor) throws -> MTLBuffer in
            if kv == .float32 { return try self.buffer(t) }
            let h = try t.converted(to: .float16)
            let b = try self.context.makeBuffer(length: h.elementCount * 2 + KernelEncoder.kvPaddingRows * t.shape[1] * t.shape[2] * 2)
            memset(b.contents(), 0, b.length)
            memcpy(b.contents(), h.basePointer, h.elementCount * 2)
            return b
        }
        let kb = try toCache(keys), vb = try toCache(values), qb = try buffer(q)
        let out = try context.makeBuffer(length: q.elementCount * 4)
        let scratch = try context.makeBuffer(length: KernelEncoder.decodeScratchBytes(heads: q.shape[1], kvHeads: keys.shape[1], headDim: q.shape[2], capacity: keys.shape[0]))
        try run {
            try $0.attention(q: qb, kCache: kb, vCache: vb, out: out, heads: q.shape[1], kvHeads: keys.shape[1],
                             headDim: q.shape[2], startPosition: startPosition, tokens: q.shape[0], kv: kv, decodeScratch: scratch)
        }
        return try read(out, shape: q.shape)
    }

    public func embed(tokens: [Int32], table: Tensor) throws -> Tensor {
        guard table.rank == 2, table.dtype != .float32, table.isContiguous else { throw MetalError.invalidArgument("embedding table") }
        let vocab = table.shape[0]
        guard tokens.allSatisfy({ $0 >= 0 && Int($0) < vocab }) else { throw MetalError.invalidArgument("token id outside vocabulary") }
        let tb = try tokens.withUnsafeBytes { try context.makeBuffer(copying: $0.baseAddress!, length: $0.count) }
        let tab = try buffer(table)
        let out = try context.makeBuffer(length: tokens.count * table.shape[1] * 4)
        let gm = GPUMatrix(buffer: tab, offset: 0, dtype: table.dtype, rows: vocab, cols: table.shape[1])
        try run { try $0.embed(tokens: tb, tokensOffset: 0, table: gm, out: out, count: tokens.count) }
        return try read(out, shape: [tokens.count, table.shape[1]])
    }
}
