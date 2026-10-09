// Copyright (c) 2026 Torin Etheridge. MIT License (see LICENSE).
// Author: Torin Etheridge · 2026-10-09

import Foundation

/// Deterministic CPU reference implementations of the operations used by a Llama decoder.
///
/// All operations validate dtype/shape/layout and throw `TensorError`. Accumulation is in Float32
/// (RMSNorm sum-of-squares, softmax denominator and dot products too) unless noted.
/// Mathematical definitions are in Docs/NUMERICAL_VALIDATION.md.
public enum CPUOps {

    // MARK: Helpers

    static func requireF32(_ tensors: Tensor..., name: String) throws {
        for t in tensors where t.dtype != .float32 {
            throw TensorError.dtypeMismatch("\(name) requires f32 operands, got \(t.dtype)")
        }
    }

    static func requireContiguous(_ tensors: Tensor..., name: String) throws {
        for t in tensors where !t.isContiguous {
            throw TensorError.unsupportedLayout("\(name) requires contiguous operands")
        }
    }

    // MARK: Elementwise

    /// out[i] = a[i] + b[i]
    public static func add(_ a: Tensor, _ b: Tensor) throws -> Tensor {
        try binary(a, b, name: "add") { $0 + $1 }
    }

    /// out[i] = a[i] * b[i]
    public static func mul(_ a: Tensor, _ b: Tensor) throws -> Tensor {
        try binary(a, b, name: "mul") { $0 * $1 }
    }

    private static func binary(_ a: Tensor, _ b: Tensor, name: String, _ f: (Float, Float) -> Float) throws -> Tensor {
        try requireF32(a, b, name: name)
        guard a.shape == b.shape else { throw TensorError.shapeMismatch("\(name): \(a.shape) vs \(b.shape)") }
        let x = try a.toFloatArray(), y = try b.toFloatArray()
        var r = [Float](repeating: 0, count: x.count)
        for i in 0..<x.count { r[i] = f(x[i], y[i]) }
        return try Tensor(r, shape: a.shape)
    }

    /// SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
    public static func silu(_ x: Tensor) throws -> Tensor {
        try requireF32(x, name: "silu")
        let v = try x.toFloatArray().map { $0 / (1 + exp(-$0)) }
        return try Tensor(v, shape: x.shape)
    }

    // MARK: Matrix products

    /// C[M,N] = A[M,K] · B[K,N]. Both f32.
    public static func matmul(_ a: Tensor, _ b: Tensor) throws -> Tensor {
        try requireF32(a, b, name: "matmul")
        guard a.rank == 2, b.rank == 2 else { throw TensorError.shapeMismatch("matmul needs rank-2 operands") }
        let (m, k) = (a.shape[0], a.shape[1]), n = b.shape[1]
        guard b.shape[0] == k else { throw TensorError.shapeMismatch("matmul: \(a.shape) x \(b.shape)") }
        let av = try a.toFloatArray(), bv = try b.toFloatArray()
        var c = [Float](repeating: 0, count: m * n)
        for i in 0..<m {
            for p in 0..<k {
                let aip = av[i * k + p]
                for j in 0..<n { c[i * n + j] += aip * bv[p * n + j] }
            }
        }
        return try Tensor(c, shape: [m, n])
    }

    /// y[M,N] = x[M,K] · W[N,K]ᵀ  (Llama weight convention: W is [out, in]).
    /// `weight` may be f32, f16, q8_0 or q4_0 and must be contiguous; `x` is f32.
    /// With M == 1 this is the matrix-vector product used during decode.
    public static func linear(_ x: Tensor, weight w: Tensor) throws -> Tensor {
        try requireF32(x, name: "linear")
        try requireContiguous(x, w, name: "linear")
        guard x.rank == 2, w.rank == 2, x.shape[1] == w.shape[1] else {
            throw TensorError.shapeMismatch("linear: x \(x.shape), weight \(w.shape)")
        }
        let (m, k, n) = (x.shape[0], x.shape[1], w.shape[0])
        let out = try Tensor(zeros: [m, n])
        let xp = x.storage.pointer.assumingMemoryBound(to: Float.self) + x.offset
        let op = out.storage.pointer.assumingMemoryBound(to: Float.self)
        let wbase = w.basePointer
        let rowBytes = (k / w.dtype.blockElements) * w.dtype.blockBytes
        let dtype = w.dtype
        let shared = SharedPointers(xp, op, wbase)
        let rowsPerChunk = 16
        let chunks = (n + rowsPerChunk - 1) / rowsPerChunk
        for t in 0..<m {
            DispatchQueue.concurrentPerform(iterations: chunks) { chunk in
                let x = shared.x + t * k
                for r in (chunk * rowsPerChunk)..<min(n, (chunk + 1) * rowsPerChunk) {
                    let row = shared.w + r * rowBytes
                    shared.out[t * n + r] = dot(dtype, row, x, k)
                }
            }
        }
        return out
    }

    /// Dot product of one weight row (any dtype) with an f32 vector, accumulated in Float32.
    @inline(__always)
    static func dot(_ dtype: DType, _ row: UnsafeRawPointer, _ x: UnsafePointer<Float>, _ k: Int) -> Float {
        var acc: Float = 0
        switch dtype {
        case .float32:
            let w = row.assumingMemoryBound(to: Float.self)
            for i in 0..<k { acc += w[i] * x[i] }
        case .float16:
            let w = row.assumingMemoryBound(to: Float16.self)
            for i in 0..<k { acc += Float(w[i]) * x[i] }
        case .q8_0:
            for b in 0..<(k / 32) {
                let p = row + b * 34
                let d = Float(p.loadUnaligned(as: Float16.self))
                let q = (p + 2).assumingMemoryBound(to: Int8.self)
                var s: Float = 0
                for j in 0..<32 { s += Float(q[j]) * x[b * 32 + j] }
                acc += d * s
            }
        case .q4_0:
            for b in 0..<(k / 32) {
                let p = row + b * 18
                let d = Float(p.loadUnaligned(as: Float16.self))
                let q = (p + 2).assumingMemoryBound(to: UInt8.self)
                var s: Float = 0
                for j in 0..<16 {
                    s += Float(Int(q[j] & 0x0F) - 8) * x[b * 32 + j]
                    s += Float(Int(q[j] >> 4) - 8) * x[b * 32 + j + 16]
                }
                acc += d * s
            }
        }
        return acc
    }

    // MARK: Normalisation / softmax

    /// RMSNorm over the last dimension: y = x / sqrt(mean(x²) + eps) * weight. Sum of squares in Float32.
    public static func rmsNorm(_ x: Tensor, weight: Tensor, eps: Float) throws -> Tensor {
        try requireF32(x, weight, name: "rmsNorm")
        guard let d = x.shape.last, weight.shape == [d] else {
            throw TensorError.shapeMismatch("rmsNorm: x \(x.shape), weight \(weight.shape)")
        }
        guard eps > 0 else { throw TensorError.invalidShape("rmsNorm eps must be positive") }
        let xv = try x.toFloatArray(), wv = try weight.toFloatArray()
        var out = [Float](repeating: 0, count: xv.count)
        for r in 0..<(xv.count / max(d, 1)) {
            var ss: Float = 0
            for i in 0..<d { ss += xv[r * d + i] * xv[r * d + i] }
            let scale = 1 / (ss / Float(d) + eps).squareRoot()
            for i in 0..<d { out[r * d + i] = xv[r * d + i] * scale * wv[i] }
        }
        return try Tensor(out, shape: x.shape)
    }

    /// Numerically stable softmax over the last dimension: exp(x - max) / Σ exp(x - max).
    public static func softmax(_ x: Tensor) throws -> Tensor {
        try requireF32(x, name: "softmax")
        guard let d = x.shape.last, d > 0 else { throw TensorError.shapeMismatch("softmax over empty dimension") }
        var v = try x.toFloatArray()
        for r in 0..<(v.count / d) {
            let base = r * d
            var mx = -Float.infinity
            for i in 0..<d { mx = max(mx, v[base + i]) }
            var sum: Float = 0
            for i in 0..<d { v[base + i] = exp(v[base + i] - mx); sum += v[base + i] }
            for i in 0..<d { v[base + i] /= sum }
        }
        return try Tensor(v, shape: x.shape)
    }

    // MARK: RoPE

    /// Rotary position embedding on x[tokens, heads, headDim], token t at absolute position `startPosition + t`.
    ///
    /// For pair index i in 0..<headDim/2: θᵢ = theta^(-2i/headDim), angle = position·θᵢ.
    /// `interleaved == true` rotates adjacent pairs (x[2i], x[2i+1]) — the layout GGUF Llama files use.
    /// `interleaved == false` rotates split halves (x[i], x[i+headDim/2]) — the Hugging Face "rotate_half" layout.
    public static func rope(
        _ x: Tensor, startPosition: Int, theta: Float, interleaved: Bool = true
    ) throws -> Tensor {
        try requireF32(x, name: "rope")
        guard x.rank == 3, x.shape[2] % 2 == 0, x.shape[2] > 0 else {
            throw TensorError.shapeMismatch("rope expects [tokens, heads, even headDim], got \(x.shape)")
        }
        guard startPosition >= 0, theta > 0 else { throw TensorError.invalidShape("rope: startPosition \(startPosition), theta \(theta)") }
        let (tokens, heads, hd) = (x.shape[0], x.shape[1], x.shape[2])
        var v = try x.toFloatArray()
        let half = hd / 2
        for t in 0..<tokens {
            let pos = Double(startPosition + t)
            for i in 0..<half {
                // Frequencies and angles in Double, rounded once to Float: removes trig range-reduction noise from the reference.
                let freq = pow(Double(theta), -2.0 * Double(i) / Double(hd))
                let angle = pos * freq
                let c = Float(cos(angle)), s = Float(sin(angle))
                for h in 0..<heads {
                    let base = (t * heads + h) * hd
                    let (ia, ib) = interleaved ? (base + 2 * i, base + 2 * i + 1) : (base + i, base + i + half)
                    let a = v[ia], b = v[ib]
                    v[ia] = a * c - b * s
                    v[ib] = a * s + b * c
                }
            }
        }
        return try Tensor(v, shape: x.shape)
    }

    // MARK: Attention

    /// Scaled attention scores with causal masking for grouped-query attention.
    ///
    /// q: [tokens, nHeads, headDim]; keys: cache laid out [capacity, nKVHeads, headDim] with `length` valid rows
    /// (only those rows are read; the cache is never copied).
    /// Query token t sits at absolute position `startPosition + t` and may attend to cache rows 0...startPosition+t.
    /// Query head h reads KV head h / (nHeads / nKVHeads). Returns [nHeads, tokens, length] where masked entries are -inf.
    public static func attentionScores(
        q: Tensor, keys: Tensor, length: Int, startPosition: Int
    ) throws -> Tensor {
        try requireF32(q, keys, name: "attentionScores")
        try requireContiguous(q, keys, name: "attentionScores")
        guard q.rank == 3, keys.rank == 3, q.shape[2] == keys.shape[2], keys.shape[1] > 0, q.shape[1] % keys.shape[1] == 0
        else { throw TensorError.shapeMismatch("attentionScores: q \(q.shape), keys \(keys.shape)") }
        guard length >= 0, length <= keys.shape[0], startPosition + q.shape[0] <= length else {
            throw TensorError.outOfBounds("attention length \(length), start \(startPosition), tokens \(q.shape[0]), capacity \(keys.shape[0])")
        }
        let (tokens, nHeads, hd) = (q.shape[0], q.shape[1], q.shape[2])
        let nKV = keys.shape[1], group = nHeads / nKV
        let scale = 1 / Float(hd).squareRoot()
        let qv = q.storage.pointer.assumingMemoryBound(to: Float.self) + q.offset
        let kv = keys.storage.pointer.assumingMemoryBound(to: Float.self) + keys.offset
        var s = [Float](repeating: -.infinity, count: nHeads * tokens * length)
        for h in 0..<nHeads {
            for t in 0..<tokens {
                for p in 0...(startPosition + t) {
                    var acc: Float = 0
                    let qb = (t * nHeads + h) * hd, kb = (p * nKV + h / group) * hd
                    for i in 0..<hd { acc += qv[qb + i] * kv[kb + i] }
                    s[(h * tokens + t) * length + p] = acc * scale
                }
            }
        }
        return try Tensor(s, shape: [nHeads, tokens, length])
    }

    /// out[t, h, :] = Σ_p probs[h, t, p] · values[p, h/group, :]. probs: [nHeads, tokens, length].
    public static func attentionApply(probs: Tensor, values: Tensor) throws -> Tensor {
        try requireF32(probs, values, name: "attentionApply")
        guard probs.rank == 3, values.rank == 3, probs.shape[2] <= values.shape[0],
            values.shape[1] > 0, probs.shape[0] % values.shape[1] == 0
        else { throw TensorError.shapeMismatch("attentionApply: probs \(probs.shape), values \(values.shape)") }
        let (nHeads, tokens, length) = (probs.shape[0], probs.shape[1], probs.shape[2])
        let nKV = values.shape[1], hd = values.shape[2], group = nHeads / nKV
        try requireContiguous(probs, values, name: "attentionApply")
        let pv = probs.storage.pointer.assumingMemoryBound(to: Float.self) + probs.offset
        let vv = values.storage.pointer.assumingMemoryBound(to: Float.self) + values.offset
        var out = [Float](repeating: 0, count: tokens * nHeads * hd)
        for h in 0..<nHeads {
            for t in 0..<tokens {
                for p in 0..<length {
                    let w = pv[(h * tokens + t) * length + p]
                    if w == 0 { continue }
                    let vb = (p * nKV + h / group) * hd, ob = (t * nHeads + h) * hd
                    for i in 0..<hd { out[ob + i] += w * vv[vb + i] }
                }
            }
        }
        return try Tensor(out, shape: [tokens, nHeads, hd])
    }

    // MARK: Embedding

    /// Rows of `table` ([vocab, dim], any dtype) selected by `tokens`, as f32 [tokens.count, dim].
    public static func embed(tokens: [Int32], table: Tensor) throws -> Tensor {
        guard table.rank == 2, table.isContiguous else { throw TensorError.unsupportedLayout("embedding table must be contiguous rank-2") }
        let (vocab, dim) = (table.shape[0], table.shape[1])
        let out = try Tensor(zeros: [tokens.count, dim])
        let rowBytes = (dim / table.dtype.blockElements) * table.dtype.blockBytes
        let dst = out.storage.pointer.assumingMemoryBound(to: Float.self)
        for (i, tok) in tokens.enumerated() {
            guard tok >= 0, Int(tok) < vocab else { throw TensorError.outOfBounds("token id \(tok) outside vocabulary of \(vocab)") }
            dequantize(table.dtype, source: table.basePointer + Int(tok) * rowBytes, elements: dim, into: dst + i * dim)
        }
        return out
    }
}

/// Raw pointers shared across `concurrentPerform` iterations that write disjoint output rows.
struct SharedPointers: @unchecked Sendable {
    let x: UnsafePointer<Float>
    let out: UnsafeMutablePointer<Float>
    let w: UnsafeRawPointer
    init(_ x: UnsafePointer<Float>, _ out: UnsafeMutablePointer<Float>, _ w: UnsafeRawPointer) {
        self.x = x; self.out = out; self.w = w
    }
}
