// Copyright (c) 2026 Torin Etheridge. MIT License (see LICENSE).
// Author: Torin Etheridge · 2026-10-09

import Foundation
import Metal
import AlpacaCore

/// Storage precision of the GPU KV cache.
public enum KVPrecision: Sendable {
    /// Half the memory and bandwidth; adds ~2^-11 relative rounding to every cached key and value.
    case float16
    /// Bit-faithful to the CPU reference cache; twice the memory.
    case float32

    var kernelSuffix: String { self == .float16 ? "kv16" : "kv32" }
    public var bytesPerElement: Int { self == .float16 ? 2 : 4 }
}

/// Parameter block of the fused decode kernels (matches `DecParams` in decode_fused.metal).
struct DecodeParams {
    var k: UInt32
    var n0: UInt32, n1: UInt32, n2: UInt32
    var flags: UInt32
    var eps: Float
}

/// A weight matrix [rows, cols] resident in a Metal buffer in its on-disk format.
public struct GPUMatrix: @unchecked Sendable {
    public let buffer: MTLBuffer
    public let offset: Int
    public let dtype: DType
    public let rows: Int
    public let cols: Int
}

/// Encodes alpaca's kernels onto a compute encoder. All kernels run on one serial encoder, so ordering
/// between dependent dispatches is implicit.
/// Precision/speed trade-off of the batched (prefill) projections.
public enum GEMMPrecision: Sendable {
    /// float32 simdgroup-matrix kernels; products and sums in float32 (weights exact as stored).
    case exact
    /// Matrix-hardware (Metal 4 tensor-op) GEMM where available: operands rounded to half (relative 2^-11 each),
    /// float32 accumulation; quantised weights are expanded to half through a reusable scratch matrix.
    /// Falls back to `.exact` on GPUs/OS versions without the capability.
    case fast
}

struct KernelEncoder {
    let context: MetalContext
    let encoder: MTLComputeCommandEncoder
    var precision: GEMMPrecision = .exact
    /// Half-precision scratch for one expanded weight matrix (needed by `.fast` with quantised weights).
    var dequantScratch: MTLBuffer?
    /// Second half-precision scratch, so gate and up weights can both be expanded for the fused SwiGLU GEMM.
    var dequantScratch2: MTLBuffer?

    /// Tokens handled per threadgroup row in the prefill mat-vec variant.
    static let prefillTokenBlock = 8
    /// Batches at least this large use the tiled simdgroup-matrix GEMM instead of the batched mat-vec kernel.
    nonisolated(unsafe) static var gemmMinTokens = 32
    /// Batches at least this large use the tensor-op GEMM when `.fast` precision is selected and supported.
    nonisolated(unsafe) static var tensorGEMMMinTokens = Tuning.int("ALPACA_TENSOR_MIN_TOKENS", 16)
    /// Batches at least this large use the tiled simdgroup-matrix attention kernel (f16 KV, head dim 64 or 128).
    nonisolated(unsafe) static var prefillAttentionMinTokens = 8
    /// Zero rows appended to every KV buffer so the tiled attention kernel can read whole 32-position blocks.
    static let kvPaddingRows = 32
    /// Decode attention aims for about this many splits per KV head (parallelism vs merge cost; swept in Docs/PERFORMANCE.md).
    nonisolated(unsafe) static var decodeTargetSplits = Tuning.int("ALPACA_DECODE_TARGET", 8)
    /// Smallest number of KV positions one decode-attention threadgroup handles.
    nonisolated(unsafe) static var decodeMinSplitLength = Tuning.int("ALPACA_DECODE_MINSPLIT", 128)

    /// Positions per split for a context of `length`: ceil(length / target), rounded up to 32, at least the minimum.
    static func decodeSplitLength(forContext length: Int) -> Int {
        let raw = (length + decodeTargetSplits - 1) / decodeTargetSplits
        return max(decodeMinSplitLength, (raw + 31) / 32 * 32)
    }
    /// Contexts shorter than this use the single-kernel attention (the extra merge dispatch is not worth it).
    nonisolated(unsafe) static var decodeSplitMinContext = Tuning.int("ALPACA_DECODE_MINCTX", 64)

    /// Bytes of partial-state scratch the split decode attention needs for contexts up to `capacity` positions.
    static func decodeScratchBytes(heads: Int, kvHeads: Int, headDim: Int, capacity: Int) -> Int {
        // Worst case is the smallest split length.
        let splits = (capacity + decodeMinSplitLength - 1) / decodeMinSplitLength
        return kvHeads * splits * (heads / kvHeads) * (headDim + 2) * 4
    }
    /// Simdgroups per attention threadgroup; must equal ATT_SIMDGROUPS in attention.metal.
    static let attentionSimdgroups = 16

    fileprivate func set<T: BitwiseCopyable>(_ value: T, _ index: Int) {
        var v = value
        encoder.setBytes(&v, length: MemoryLayout<T>.size, index: index)
    }

    private func dispatch1D(_ pipeline: MTLComputePipelineState, count: Int) {
        encoder.setComputePipelineState(pipeline)
        let tg = min(256, pipeline.maxTotalThreadsPerThreadgroup)
        encoder.dispatchThreads(MTLSize(width: count, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
    }

    func add(_ a: MTLBuffer, _ b: MTLBuffer, out: MTLBuffer, count: Int) throws {
        let p = try context.pipeline("add_f32")
        encoder.setComputePipelineState(p)
        encoder.setBuffer(a, offset: 0, index: 0); encoder.setBuffer(b, offset: 0, index: 1); encoder.setBuffer(out, offset: 0, index: 2)
        set(UInt32(count), 3)
        dispatch1D(p, count: count)
    }

    func mul(_ a: MTLBuffer, _ b: MTLBuffer, out: MTLBuffer, count: Int) throws {
        let p = try context.pipeline("mul_f32")
        encoder.setComputePipelineState(p)
        encoder.setBuffer(a, offset: 0, index: 0); encoder.setBuffer(b, offset: 0, index: 1); encoder.setBuffer(out, offset: 0, index: 2)
        set(UInt32(count), 3)
        dispatch1D(p, count: count)
    }

    func silu(_ x: MTLBuffer, out: MTLBuffer, count: Int) throws {
        let p = try context.pipeline("silu_f32")
        encoder.setComputePipelineState(p)
        encoder.setBuffer(x, offset: 0, index: 0); encoder.setBuffer(out, offset: 0, index: 1)
        set(UInt32(count), 2)
        dispatch1D(p, count: count)
    }

    func siluMul(gate: MTLBuffer, up: MTLBuffer, out: MTLBuffer, count: Int) throws {
        let p = try context.pipeline("silu_mul_f32")
        encoder.setComputePipelineState(p)
        encoder.setBuffer(gate, offset: 0, index: 0); encoder.setBuffer(up, offset: 0, index: 1); encoder.setBuffer(out, offset: 0, index: 2)
        set(UInt32(count), 3)
        dispatch1D(p, count: count)
    }

    func rmsNorm(_ x: MTLBuffer, xOffset: Int = 0, weight: MTLBuffer, weightOffset: Int = 0,
                 out: MTLBuffer, outOffset: Int = 0, rows: Int, dim: Int, eps: Float) throws {
        let p = try context.pipeline("rmsnorm_f32")
        encoder.setComputePipelineState(p)
        encoder.setBuffer(x, offset: xOffset, index: 0); encoder.setBuffer(weight, offset: weightOffset, index: 1)
        encoder.setBuffer(out, offset: outOffset, index: 2)
        set(UInt32(dim), 3); set(eps, 4)
        let tg = dim <= 512 ? 128 : 256
        encoder.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
    }

    func embed(tokens: MTLBuffer, tokensOffset: Int, table: GPUMatrix, out: MTLBuffer, count: Int) throws {
        let name: String
        switch table.dtype {
        case .float16: name = "embed_f16"
        case .q8_0: name = "embed_q8_0"
        case .q4_0: name = "embed_q4_0"
        case .float32: throw MetalError.unsupported("f32 embedding tables")
        }
        let p = try context.pipeline(name)
        encoder.setComputePipelineState(p)
        encoder.setBuffer(tokens, offset: tokensOffset, index: 0)
        encoder.setBuffer(table.buffer, offset: table.offset, index: 1)
        encoder.setBuffer(out, offset: 0, index: 2)
        set(UInt32(table.cols), 3)
        encoder.dispatchThreads(MTLSize(width: table.cols, height: count, depth: 1), threadsPerThreadgroup: MTLSize(width: min(table.cols, 64), height: 1, depth: 1))
    }

    /// Whether the half-activation tensor-op GEMM can run `w` for a batch of `tokens` (capability, size, scratch).
    static func canUseTensorGEMM(_ w: GPUMatrix, tokens: Int, precision: GEMMPrecision, context: MetalContext, scratch: MTLBuffer?) -> Bool {
        precision == .fast && context.supportsTensorGEMM && tokens >= tensorGEMMMinTokens && w.cols % 32 == 0
            && (w.dtype == .float16 || (w.dtype.isQuantized && (scratch.map { $0.length >= w.rows * w.cols * 2 } ?? false)))
    }

    func canUseTensorGEMM(_ w: GPUMatrix, tokens: Int) -> Bool {
        Self.canUseTensorGEMM(w, tokens: tokens, precision: precision, context: context, scratch: dequantScratch)
    }

    /// Expands a quantised matrix into `target` (half) or returns its own buffer when already half.
    private func expandToHalf(_ w: GPUMatrix, into target: MTLBuffer?) throws -> (MTLBuffer, Int) {
        guard w.dtype != .float16 else { return (w.buffer, w.offset) }
        let name = w.dtype == .q8_0 ? "dequant_q8_0_to_half" : "dequant_q4_0_to_half"
        let dp = try context.pipeline(name)
        encoder.setComputePipelineState(dp)
        encoder.setBuffer(w.buffer, offset: w.offset, index: 0); encoder.setBuffer(target!, offset: 0, index: 1)
        set(UInt32(w.cols), 2)
        encoder.dispatchThreads(MTLSize(width: w.cols / 32 * 4, height: w.rows, depth: 1), threadsPerThreadgroup: MTLSize(width: min(w.cols / 32 * 4, 32), height: 8, depth: 1))
        return (target!, 0)
    }

    /// Y[tokens, rows] (+)= Xhalf[tokens, cols] · Wᵀ on the tensor-op GEMM. `x` holds half activations; quantised weights are
    /// expanded into the scratch first. Call only when `canUseTensorGEMM(w, tokens:)` holds.
    func projectHalf(_ w: GPUMatrix, xHalf x: MTLBuffer, y: MTLBuffer, tokens: Int, accumulate: Bool) throws {
        let weights = try expandToHalf(w, into: dequantScratch)
        let gp = try context.pipeline(accumulate ? "gemm_tensor_half_w_acc" : "gemm_tensor_half_w")
        encoder.setComputePipelineState(gp)
        encoder.setBuffer(weights.0, offset: weights.1, index: 0)
        encoder.setBuffer(x, offset: 0, index: 1)
        encoder.setBuffer(y, offset: 0, index: 2)
        set(UInt32(w.cols), 3); set(UInt32(w.rows), 4); set(UInt32(tokens), 5)
        encoder.dispatchThreadgroups(MTLSize(width: (w.rows + 63) / 64, height: (tokens + 63) / 64, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    }

    /// RMSNorm writing half output (the activation format of `projectHalf`).
    func rmsNormToHalf(_ x: MTLBuffer, weight: GPUMatrix, out: MTLBuffer, rows: Int, dim: Int, eps: Float) throws {
        let p = try context.pipeline("rmsnorm_f32_to_half")
        encoder.setComputePipelineState(p)
        encoder.setBuffer(x, offset: 0, index: 0); encoder.setBuffer(weight.buffer, offset: weight.offset, index: 1)
        encoder.setBuffer(out, offset: 0, index: 2)
        set(UInt32(dim), 3); set(eps, 4)
        encoder.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: dim <= 512 ? 128 : 256, height: 1, depth: 1))
    }

    func convertToHalf(_ x: MTLBuffer, out: MTLBuffer, count: Int) throws {
        let p = try context.pipeline("convert_f32_to_half")
        encoder.setComputePipelineState(p)
        encoder.setBuffer(x, offset: 0, index: 0); encoder.setBuffer(out, offset: 0, index: 1)
        set(UInt32(count), 2)
        dispatch1D(p, count: count)
    }

    /// residual[tokens, rows] += X[tokens, cols] · Wᵀ in one dispatch (float32 kernels; the product is accumulated in place).
    func linearAdd(_ w: GPUMatrix, x: MTLBuffer, residual: MTLBuffer, tokens: Int) throws {
        try linear(w, x: x, y: residual, tokens: tokens, accumulate: true)
    }

    /// out = silu(X·Wgᵀ) ⊙ (X·Wuᵀ), float32 kernels (two projections and the elementwise gate).
    func gateUpSilu(_ wg: GPUMatrix, _ wu: GPUMatrix, x: MTLBuffer, out: MTLBuffer, scratchUp: MTLBuffer, tokens: Int) throws {
        try linear(wg, x: x, y: out, tokens: tokens)
        try linear(wu, x: x, y: scratchUp, tokens: tokens)
        try siluMul(gate: out, up: scratchUp, out: out, count: tokens * wg.rows)
    }

    /// outHalf = half(silu(Xhalf·Wgᵀ) ⊙ (Xhalf·Wuᵀ)) in one tensor-op kernel: both products stay in registers.
    func gateUpSiluHalf(_ wg: GPUMatrix, _ wu: GPUMatrix, xHalf x: MTLBuffer, outHalf out: MTLBuffer, tokens: Int) throws {
        let g = try expandToHalf(wg, into: dequantScratch)
        let u = try expandToHalf(wu, into: dequantScratch2)
        let p = try context.pipeline("gemm_tensor_gateup_silu")
        encoder.setComputePipelineState(p)
        encoder.setBuffer(g.0, offset: g.1, index: 0); encoder.setBuffer(u.0, offset: u.1, index: 1)
        encoder.setBuffer(x, offset: 0, index: 2); encoder.setBuffer(out, offset: 0, index: 3)
        set(UInt32(wg.cols), 4); set(UInt32(wg.rows), 5); set(UInt32(tokens), 6)
        encoder.dispatchThreadgroups(MTLSize(width: (wg.rows + 63) / 64, height: (tokens + 63) / 64, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    }

    /// Y[tokens, rows] = X[tokens, cols] · Wᵀ
    func linear(_ w: GPUMatrix, x: MTLBuffer, xOffset: Int = 0, y: MTLBuffer, yOffset: Int = 0, tokens: Int, accumulate: Bool = false) throws {
        let fmt: String
        switch w.dtype {
        case .float16: fmt = "f16"
        case .q8_0: fmt = "q8_0"
        case .q4_0: fmt = "q4_0"
        case .float32: throw MetalError.unsupported("f32 weight matrices on Metal")
        }
        if tokens >= Self.gemmMinTokens && w.cols % 32 == 0 {
            let p = try context.pipeline("mm_\(fmt)")
            encoder.setComputePipelineState(p)
            encoder.setBuffer(w.buffer, offset: w.offset, index: 0)
            encoder.setBuffer(x, offset: xOffset, index: 1)
            encoder.setBuffer(y, offset: yOffset, index: 2)
            set(UInt32(w.cols), 3); set(UInt32(w.rows), 4); set(UInt32(tokens), 5); set(UInt32(accumulate ? 1 : 0), 6)
            encoder.dispatchThreadgroups(MTLSize(width: (w.rows + 63) / 64, height: (tokens + 31) / 32, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            return
        }
        let tb = tokens == 1 ? 1 : Self.prefillTokenBlock
        let p = try context.pipeline("mv_\(fmt)_tb\(tb)")
        encoder.setComputePipelineState(p)
        encoder.setBuffer(w.buffer, offset: w.offset, index: 0)
        encoder.setBuffer(x, offset: xOffset, index: 1)
        encoder.setBuffer(y, offset: yOffset, index: 2)
        set(UInt32(w.cols), 3); set(UInt32(w.rows), 4); set(UInt32(tokens), 5); set(UInt32(accumulate ? 1 : 0), 6)
        let simdgroups = 4
        encoder.dispatchThreadgroups(
            MTLSize(width: (w.rows + simdgroups - 1) / simdgroups, height: (tokens + tb - 1) / tb, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32 * simdgroups, height: 1, depth: 1))
    }

    /// Causal prefill attention on the tensor-op hardware (f16 KV cache, head dim 64/128). `qHalf` is the half copy of the queries;
    /// the context vector is written as half (`outputHalf`) for the following tensor-op projection, or as float32.
    func attentionTensor(qHalf: MTLBuffer, kCache: MTLBuffer, vCache: MTLBuffer, out: MTLBuffer, outputHalf: Bool,
                         heads: Int, kvHeads: Int, headDim: Int, startPosition: Int, tokens: Int) throws {
        let p = try context.pipeline("attn_tensor_hd\(headDim)" + (outputHalf ? "_oh" : ""))
        encoder.setComputePipelineState(p)
        encoder.setBuffer(qHalf, offset: 0, index: 0); encoder.setBuffer(kCache, offset: 0, index: 1)
        encoder.setBuffer(vCache, offset: 0, index: 2); encoder.setBuffer(out, offset: 0, index: 3)
        set(UInt32(heads), 4); set(UInt32(kvHeads), 5); set(UInt32(startPosition), 6); set(UInt32(tokens), 7)
        encoder.dispatchThreadgroups(MTLSize(width: (tokens + 63) / 64, height: heads, depth: 1), threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    }

    static func canUseTensorAttention(context: MetalContext, precision: GEMMPrecision, kv: KVPrecision, headDim: Int, tokens: Int) -> Bool {
        precision == .fast && context.supportsTensorGEMM && kv == .float16 && (headDim == 64 || headDim == 128) && tokens >= tensorGEMMMinTokens
    }

    func rope(_ x: MTLBuffer, freqs: MTLBuffer, heads: Int, headDim: Int, startPosition: Int, tokens: Int) throws {
        let p = try context.pipeline("rope_f32")
        encoder.setComputePipelineState(p)
        encoder.setBuffer(x, offset: 0, index: 0); encoder.setBuffer(freqs, offset: 0, index: 1)
        set(UInt32(heads), 2); set(UInt32(headDim), 3); set(UInt32(startPosition), 4)
        encoder.dispatchThreads(MTLSize(width: headDim / 2, height: heads, depth: tokens), threadsPerThreadgroup: MTLSize(width: headDim / 2, height: 1, depth: 1))
    }

    func attention(q: MTLBuffer, kCache: MTLBuffer, vCache: MTLBuffer, out: MTLBuffer,
                   heads: Int, kvHeads: Int, headDim: Int, startPosition: Int, tokens: Int, kv: KVPrecision,
                   decodeScratch: MTLBuffer? = nil, outputHalf: Bool = false) throws {
        if kv == .float16 && tokens == 1 && startPosition + 1 >= Self.decodeSplitMinContext && (headDim == 64 || headDim == 128) && heads % kvHeads == 0 && heads / kvHeads <= 8,
           let scratch = decodeScratch {
            let length = startPosition + 1
            let splitLen = Self.decodeSplitLength(forContext: length)
            let nSplits = (length + splitLen - 1) / splitLen
            precondition(Self.decodeScratchBytes(heads: heads, kvHeads: kvHeads, headDim: headDim, capacity: length) <= scratch.length, "decode scratch too small")
            let p1 = try context.pipeline("attn_decode_split_hd\(headDim)")
            encoder.setComputePipelineState(p1)
            encoder.setBuffer(q, offset: 0, index: 0); encoder.setBuffer(kCache, offset: 0, index: 1)
            encoder.setBuffer(vCache, offset: 0, index: 2); encoder.setBuffer(scratch, offset: 0, index: 3)
            set(UInt32(heads), 4); set(UInt32(kvHeads), 5); set(UInt32(length), 6); set(UInt32(splitLen), 7); set(UInt32(nSplits), 8)
            encoder.dispatchThreadgroups(MTLSize(width: nSplits, height: kvHeads, depth: 1), threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            let p2 = try context.pipeline("attn_decode_merge")
            encoder.setComputePipelineState(p2)
            encoder.setBuffer(scratch, offset: 0, index: 0); encoder.setBuffer(out, offset: 0, index: 1)
            set(UInt32(heads), 2); set(UInt32(kvHeads), 3); set(UInt32(headDim), 4); set(UInt32(nSplits), 5)
            encoder.dispatchThreadgroups(MTLSize(width: heads, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: headDim, height: 1, depth: 1))
            return
        }
        if kv == .float16 && tokens >= Self.prefillAttentionMinTokens && (headDim == 64 || headDim == 128) {
            let p = try context.pipeline("attn_prefill_hd\(headDim)_kv16" + (outputHalf ? "_oh" : ""))
            encoder.setComputePipelineState(p)
            encoder.setBuffer(q, offset: 0, index: 0); encoder.setBuffer(kCache, offset: 0, index: 1)
            encoder.setBuffer(vCache, offset: 0, index: 2); encoder.setBuffer(out, offset: 0, index: 3)
            set(UInt32(heads), 4); set(UInt32(kvHeads), 5); set(UInt32(startPosition), 6); set(UInt32(tokens), 7)
            encoder.dispatchThreadgroups(MTLSize(width: (tokens + 63) / 64, height: heads, depth: 1), threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            return
        }
        let p = try context.pipeline("attention_\(kv.kernelSuffix)")
        encoder.setComputePipelineState(p)
        encoder.setBuffer(q, offset: 0, index: 0); encoder.setBuffer(kCache, offset: 0, index: 1)
        encoder.setBuffer(vCache, offset: 0, index: 2); encoder.setBuffer(out, offset: 0, index: 3)
        set(UInt32(heads), 4); set(UInt32(kvHeads), 5); set(UInt32(headDim), 6); set(UInt32(startPosition), 7)
        encoder.dispatchThreadgroups(MTLSize(width: heads, height: tokens, depth: 1), threadsPerThreadgroup: MTLSize(width: 32 * Self.attentionSimdgroups, height: 1, depth: 1))
    }
}

extension KernelEncoder {
    /// GPU greedy sampling: arg-max of `logits` → `chainToken` (next input) and `emitted[slot]`.
    func argmax(logits: MTLBuffer, count: Int, chainToken: MTLBuffer, emitted: MTLBuffer, slot: Int) throws {
        let p = try context.pipeline("argmax_logits")
        encoder.setComputePipelineState(p)
        encoder.setBuffer(logits, offset: 0, index: 0)
        set(UInt32(count), 1)
        encoder.setBuffer(chainToken, offset: 0, index: 2); encoder.setBuffer(emitted, offset: 0, index: 3)
        set(UInt32(slot), 4)
        encoder.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1))
    }

    /// Fused decode kernels handle a single token, K ≤ 4096, and need every weight of one launch in the same format.
    static let decodeMaxK = 4096
    nonisolated(unsafe) static var fusedDecodeEnabled = Tuning.int("ALPACA_DISABLE_FUSED_DECODE", 0) == 0

    static func canFuseDecode(_ mats: [GPUMatrix], tokens: Int) -> Bool {
        guard fusedDecodeEnabled, tokens == 1, let first = mats.first, first.dtype != .float32 else { return false }
        return mats.allSatisfy { $0.dtype == first.dtype && $0.cols == first.cols } && first.cols <= decodeMaxK && first.cols % 32 == 0
            && (first.dtype.isQuantized || first.cols % 8 == 0)
    }

    private func dispatchDecode(_ mats: [GPUMatrix], x: MTLBuffer, xOffset: Int, norm: (weight: GPUMatrix, eps: Float)?,
                                outputs: [(MTLBuffer, Int)], flags: UInt32, rows: [Int]) {
        let fmt = mats[0].dtype == .float16 ? "dec_f16" : (mats[0].dtype == .q8_0 ? "dec_q8_0" : "dec_q4_0")
        let pipeline = try! context.pipeline(fmt)
        encoder.setComputePipelineState(pipeline)
        for i in 0..<3 { let m = mats[min(i, mats.count - 1)]; encoder.setBuffer(m.buffer, offset: m.offset, index: i) }
        encoder.setBuffer(x, offset: xOffset, index: 3)
        if let norm { encoder.setBuffer(norm.weight.buffer, offset: norm.weight.offset, index: 4) } else { encoder.setBuffer(x, offset: xOffset, index: 4) }
        for i in 0..<3 { let o = outputs[min(i, outputs.count - 1)]; encoder.setBuffer(o.0, offset: o.1, index: 5 + i) }
        var p = DecodeParams(k: UInt32(mats[0].cols), n0: UInt32(rows[0]), n1: UInt32(rows.count > 1 ? rows[1] : 0), n2: UInt32(rows.count > 2 ? rows[2] : 0),
                             flags: flags | (norm != nil ? 1 : 0), eps: norm?.eps ?? 0)
        encoder.setBytes(&p, length: MemoryLayout<DecodeParams>.size, index: 8)
        encoder.setThreadgroupMemoryLength((mats[0].cols * 4 + 15) / 16 * 16, index: 0)   // the staged input row, sized to K
        let total = (flags & 2) != 0 ? rows[0] : rows.reduce(0, +)
        encoder.dispatchThreadgroups(MTLSize(width: (total + 3) / 4, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    }

    /// out (+)= W · norm?(x) for one token.
    func decodeProjection(_ w: GPUMatrix, x: MTLBuffer, xOffset: Int = 0, norm: (weight: GPUMatrix, eps: Float)? = nil,
                          out: MTLBuffer, outOffset: Int = 0, add: Bool = false) {
        dispatchDecode([w], x: x, xOffset: xOffset, norm: norm, outputs: [(out, outOffset)], flags: add ? 4 : 0, rows: [w.rows])
    }

    /// q, k, v = Wq·xn, Wk·xn, Wv·xn in one dispatch (xn = RMS-normalised x).
    func decodeQKV(_ wq: GPUMatrix, _ wk: GPUMatrix, _ wv: GPUMatrix, x: MTLBuffer, norm: (weight: GPUMatrix, eps: Float), q: MTLBuffer, k: MTLBuffer, v: MTLBuffer) {
        dispatchDecode([wq, wk, wv], x: x, xOffset: 0, norm: norm, outputs: [(q, 0), (k, 0), (v, 0)], flags: 0, rows: [wq.rows, wk.rows, wv.rows])
    }

    /// out = silu(Wgate·xn) ⊙ (Wup·xn) in one dispatch.
    func decodeGateUp(_ wg: GPUMatrix, _ wu: GPUMatrix, x: MTLBuffer, norm: (weight: GPUMatrix, eps: Float), out: MTLBuffer) {
        dispatchDecode([wg, wu], x: x, xOffset: 0, norm: norm, outputs: [(out, 0)], flags: 2, rows: [wg.rows])
    }

    func ropeQKVStore(q: MTLBuffer, k: MTLBuffer, v: MTLBuffer, kCache: MTLBuffer, vCache: MTLBuffer, freqs: MTLBuffer,
                      heads: Int, kvHeads: Int, headDim: Int, startPosition: Int, tokens: Int, kv: KVPrecision) throws {
        let p = try context.pipeline("rope_qkv_store_\(kv.kernelSuffix)")
        encoder.setComputePipelineState(p)
        encoder.setBuffer(q, offset: 0, index: 0); encoder.setBuffer(k, offset: 0, index: 1); encoder.setBuffer(v, offset: 0, index: 2)
        encoder.setBuffer(kCache, offset: 0, index: 3); encoder.setBuffer(vCache, offset: 0, index: 4); encoder.setBuffer(freqs, offset: 0, index: 5)
        set(UInt32(heads), 6); set(UInt32(kvHeads), 7); set(UInt32(headDim), 8); set(UInt32(startPosition), 9)
        encoder.dispatchThreads(MTLSize(width: headDim / 2, height: heads + kvHeads, depth: tokens), threadsPerThreadgroup: MTLSize(width: headDim / 2, height: 1, depth: 1))
    }
}

/// RoPE frequency table θ^(-2i/headDim), computed in double precision and rounded once to Float.
func ropeFrequencies(headDim: Int, theta: Float) -> [Float] {
    (0..<(headDim / 2)).map { Float(pow(Double(theta), -2.0 * Double($0) / Double(headDim))) }
}

/// Developer tuning knobs read once from the environment (used for kernel parameter sweeps; defaults are the shipped values).
enum Tuning {
    static func int(_ name: String, _ fallback: Int) -> Int {
        ProcessInfo.processInfo.environment[name].flatMap(Int.init) ?? fallback
    }
}
