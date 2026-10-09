// Copyright (c) 2026 Torin Etheridge. MIT License (see LICENSE).
// Author: Torin Etheridge · 2026-10-09

import Foundation
import Metal
import AlpacaCore

/// Llama weights resident on the GPU, shared (read-only) by any number of `MetalLlamaSession`s.
///
/// Memory: when the weights come from a memory-mapped GGUF file the whole file is wrapped as a single
/// no-copy `MTLBuffer` (unified memory: the GPU reads the page cache directly, so resident weight memory ≈ file size
/// and nothing is duplicated). Tensors that were converted at load time (e.g. Q4_1 → F16) live in their own buffers.
public final class MetalLlamaModel: @unchecked Sendable {
    public let context: MetalContext
    public let config: LlamaConfig

    struct Layer {
        let attnNorm: GPUMatrix, wq: GPUMatrix, wk: GPUMatrix, wv: GPUMatrix, wo: GPUMatrix
        let ffnNorm: GPUMatrix, wGate: GPUMatrix, wUp: GPUMatrix, wDown: GPUMatrix
    }
    let embedding: GPUMatrix
    let layers: [Layer]
    let outputNorm: GPUMatrix
    let output: GPUMatrix
    let ropeFreqs: MTLBuffer

    /// Bytes of weight data referenced by GPU buffers without copying.
    public let zeroCopyWeightBytes: Int
    /// Bytes copied into dedicated GPU buffers.
    public let copiedWeightBytes: Int
    private let keepAlive: AnyObject?

    /// - Parameters:
    ///   - mappedRegion: base/length of a read-only file mapping that backs some or all of `weights`.
    ///   - keepAlive: object that must outlive the GPU buffers (the mapping owner).
    public init(
        context: MetalContext, config: LlamaConfig, weights: LlamaWeights,
        mappedRegion: (base: UnsafeRawPointer, length: Int)? = nil, keepAlive: AnyObject? = nil
    ) throws {
        try config.validate()
        try weights.validate(against: config)
        guard config.headDim % 32 == 0, config.headDim <= 256 else {
            throw MetalError.unsupported("head dimension \(config.headDim): the attention kernel needs a multiple of 32 up to 256")
        }
        self.context = context; self.config = config; self.keepAlive = keepAlive

        // Wrap the file mapping once (page-aligned base; length rounded up to a whole page, all of which is mapped).
        var fileBuffer: MTLBuffer?
        if let r = mappedRegion {
            let page = Int(getpagesize())
            let length = (r.length + page - 1) / page * page
            if length <= context.device.maxBufferLength {
                fileBuffer = context.device.makeBuffer(bytesNoCopy: UnsafeMutableRawPointer(mutating: r.base), length: length,
                                                       options: .storageModeShared, deallocator: nil)
            }
        }
        var zeroCopy = 0, copied = 0
        func upload(_ t: Tensor) throws -> GPUMatrix {
            let rows = t.rank == 2 ? t.shape[0] : 1, cols = t.shape[t.rank - 1]
            let bytes = (t.elementCount / t.dtype.blockElements) * t.dtype.blockBytes
            if let r = mappedRegion, let fb = fileBuffer {
                let p = UnsafeRawPointer(t.basePointer)
                if p >= r.base, p + bytes <= r.base + r.length {
                    zeroCopy += bytes
                    return GPUMatrix(buffer: fb, offset: p - r.base, dtype: t.dtype, rows: rows, cols: cols)
                }
            }
            copied += bytes
            return GPUMatrix(buffer: try context.makeBuffer(copying: t.basePointer, length: bytes), offset: 0, dtype: t.dtype, rows: rows, cols: cols)
        }
        func matrix(_ t: Tensor) throws -> GPUMatrix {
            guard t.dtype != .float32 else { throw MetalError.unsupported("f32 projection matrices (use f16, q8_0 or q4_0)") }
            guard t.shape[1] % (t.dtype.isQuantized ? 32 : 8) == 0 else { throw MetalError.unsupported("input width \(t.shape[1]) is not a multiple of the kernel unit") }
            return try upload(t)
        }
        embedding = try matrix(weights.tokenEmbedding)
        output = weights.output != nil ? try matrix(weights.outputProjection) : embedding
        outputNorm = try upload(weights.outputNorm)
        layers = try weights.layers.map { l in
            Layer(attnNorm: try upload(l.attnNorm), wq: try matrix(l.wq), wk: try matrix(l.wk), wv: try matrix(l.wv), wo: try matrix(l.wo),
                  ffnNorm: try upload(l.ffnNorm), wGate: try matrix(l.wGate), wUp: try matrix(l.wUp), wDown: try matrix(l.wDown))
        }
        let freqs = ropeFrequencies(headDim: config.headDim, theta: config.ropeTheta)
        ropeFreqs = try context.makeBuffer(copying: freqs, length: freqs.count * 4)
        zeroCopyWeightBytes = zeroCopy; copiedWeightBytes = copied
    }

    /// Bytes a session with `capacity` positions and `maxBatch` prefill chunk allocates (KV cache + scratch).
    public static func sessionByteCount(config c: LlamaConfig, capacity: Int, maxBatch: Int, kvPrecision: KVPrecision = .float16) -> Int {
        let kv = 2 * c.layerCount * capacity * c.kvWidth * kvPrecision.bytesPerElement
        let scratch = 4 * maxBatch * (2 * c.hiddenSize + 2 * c.queryWidth + 2 * c.kvWidth + 2 * c.feedForwardSize) + 4 * c.vocabSize + 4 * capacity
        return kv + scratch
    }

    /// Bytes of half-precision scratch needed to expand the largest quantised per-layer projection (0 if none is quantised).
    /// The output projection / embedding table is excluded: it is only ever multiplied one token at a time (mat-vec).
    var maxQuantizedMatrixHalfBytes: Int {
        let all = layers.flatMap { [$0.wq, $0.wk, $0.wv, $0.wo, $0.wGate, $0.wUp, $0.wDown] }
        return all.filter { $0.dtype.isQuantized }.map { $0.rows * $0.cols * 2 }.max() ?? 0
    }

    public func makeSession(capacity: Int? = nil, maxBatch: Int = 512, kvPrecision: KVPrecision = .float16,
                            gemmPrecision: GEMMPrecision = .fast) throws -> MetalLlamaSession {
        try MetalLlamaSession(model: self, capacity: min(capacity ?? config.contextLength, config.contextLength), maxBatch: maxBatch,
                              kvPrecision: kvPrecision, gemmPrecision: gemmPrecision)
    }
}

/// One generation's mutable state: KV cache (f16 by default) + scratch buffers. Sessions never share mutable memory, so
/// concurrent generations on one `MetalLlamaModel` cannot corrupt each other. A session itself is single-flight:
/// a second `forward` while one is running throws instead of racing.
public final class MetalLlamaSession: @unchecked Sendable {
    public let model: MetalLlamaModel
    public let capacity: Int
    public let maxBatch: Int
    public let kvPrecision: KVPrecision
    public private(set) var length = 0
    /// GPU execution time (seconds) of the last `forward`, from command-buffer timestamps.
    public private(set) var lastGPUSeconds: Double = 0
    /// CPU time (seconds) spent encoding the last `forward`'s commands, before commit.
    public private(set) var lastEncodeSeconds: Double = 0
    /// When true, every labelled stage runs in its own command buffer so GPU time can be attributed per stage
    /// (`lastProfile`). This serialises and slows execution; use for analysis only.
    public var profilingEnabled = false
    /// GPU seconds per stage label and dispatch count for the last profiled `forward`.
    public private(set) var lastProfile: [String: (seconds: Double, dispatches: Int)] = [:]

    private let kCache: [MTLBuffer], vCache: [MTLBuffer]
    private let x: MTLBuffer, xn: MTLBuffer, q: MTLBuffer, k: MTLBuffer, v: MTLBuffer, ctx: MTLBuffer
    private let gate: MTLBuffer, up: MTLBuffer, logits: MTLBuffer, tokenBuffer: MTLBuffer
    /// Half-precision activations for the tensor-op GEMM path (nil when that path is unavailable).
    private let xnHalf: MTLBuffer?, ctxHalf: MTLBuffer?, gateHalf: MTLBuffer?
    private let halfPathAvailable: Bool
    private let decodeScratch: MTLBuffer
    private let busy = NSLock()
    /// Own command queue per session: greedy chaining keeps several command buffers in flight, and independent queues keep
    /// concurrent generations from queueing behind each other.
    private let queue: MTLCommandQueue
    /// Greedy chaining: the argmax kernel writes the next input token to `chainToken` and records every step's token in
    /// `chainEmitted[step]`.
    private let chainToken: MTLBuffer
    private let chainEmitted: MTLBuffer
    public let gemmPrecision: GEMMPrecision
    private let dequantScratch: MTLBuffer?
    private let dequantScratch2: MTLBuffer?

    init(model: MetalLlamaModel, capacity: Int, maxBatch: Int, kvPrecision: KVPrecision, gemmPrecision: GEMMPrecision) throws {
        guard capacity > 0, maxBatch > 0 else { throw MetalError.invalidArgument("capacity and maxBatch must be positive") }
        let c = model.config, ctx0 = model.context
        self.model = model; self.capacity = capacity; self.maxBatch = maxBatch; self.kvPrecision = kvPrecision; self.gemmPrecision = gemmPrecision
        let scratchBytes = model.maxQuantizedMatrixHalfBytes
        let wantScratch = gemmPrecision == .fast && ctx0.supportsTensorGEMM && scratchBytes > 0
        dequantScratch = wantScratch ? try ctx0.makeBuffer(length: scratchBytes) : nil
        dequantScratch2 = wantScratch ? try ctx0.makeBuffer(length: scratchBytes) : nil
        let kvBytes = (capacity + KernelEncoder.kvPaddingRows) * c.kvWidth * kvPrecision.bytesPerElement
        kCache = try (0..<c.layerCount).map { _ in try ctx0.makeBuffer(length: kvBytes) }
        vCache = try (0..<c.layerCount).map { _ in try ctx0.makeBuffer(length: kvBytes) }
        x = try ctx0.makeBuffer(length: maxBatch * c.hiddenSize * 4)
        xn = try ctx0.makeBuffer(length: maxBatch * c.hiddenSize * 4)
        q = try ctx0.makeBuffer(length: maxBatch * c.queryWidth * 4)
        k = try ctx0.makeBuffer(length: maxBatch * c.kvWidth * 4)
        v = try ctx0.makeBuffer(length: maxBatch * c.kvWidth * 4)
        ctx = try ctx0.makeBuffer(length: maxBatch * c.queryWidth * 4)
        gate = try ctx0.makeBuffer(length: maxBatch * c.feedForwardSize * 4)
        up = try ctx0.makeBuffer(length: maxBatch * c.feedForwardSize * 4)
        logits = try ctx0.makeBuffer(length: c.vocabSize * 4)
        tokenBuffer = try ctx0.makeBuffer(length: capacity * 4)
        // The half-activation prefill path needs every per-layer projection to be tensor-op eligible and (for quantised
        // weights) the dequantisation scratch to exist.
        let eligible = gemmPrecision == .fast && ctx0.supportsTensorGEMM
            && model.layers.allSatisfy { l in [l.wq, l.wk, l.wv, l.wo, l.wGate, l.wUp, l.wDown].allSatisfy { $0.cols % 32 == 0 && ($0.dtype == .float16 || $0.dtype.isQuantized) } }
            && (scratchBytes == 0 || dequantScratch != nil)
        halfPathAvailable = eligible
        xnHalf = eligible ? try ctx0.makeBuffer(length: maxBatch * c.hiddenSize * 2) : nil
        ctxHalf = eligible ? try ctx0.makeBuffer(length: maxBatch * c.queryWidth * 2) : nil
        gateHalf = eligible ? try ctx0.makeBuffer(length: maxBatch * c.feedForwardSize * 2) : nil
        chainToken = try ctx0.makeBuffer(length: 16)
        chainEmitted = try ctx0.makeBuffer(length: (capacity + 8) * 4)
        guard let q = ctx0.device.makeCommandQueue() else { throw MetalError.commandFailed("could not create command queue") }
        queue = q
        decodeScratch = try ctx0.makeBuffer(length: KernelEncoder.decodeScratchBytes(heads: c.headCount, kvHeads: c.kvHeadCount, headDim: c.headDim, capacity: capacity))
    }

    public func reset() { busy.lock(); length = 0; busy.unlock() }

    /// Appends one token and returns a view of the next-token logits, valid until the next call on this session
    /// (no copy: the view aliases the GPU's shared-memory logits buffer).
    public func decode(token: Int32) throws -> UnsafeBufferPointer<Float> {
        _ = try forward(tokens: [token])
        return UnsafeBufferPointer(start: logits.contents().assumingMemoryBound(to: Float.self), count: model.config.vocabSize)
    }

    /// Greedy decoding with the sampling loop on the GPU. Feeds `startToken` at position `length`, picks the arg-max of each
    /// step's logits on the GPU (lowest index wins ties, like `Sampler.argmax`) and feeds it straight into the next step.
    /// Up to `depth` command buffers are in flight, so the GPU never waits for the CPU between tokens (a CPU round trip per
    /// token costs 6–14% of decode time). `emit` receives each newly produced token in order and returns false to stop;
    /// steps already in flight when it stops are drained and their results discarded (their KV rows lie beyond `length`).
    /// Returns the number of tokens fed, by which `length` has advanced.
    public func decodeGreedy(startToken: Int32, maxSteps: Int, depth: Int = 3, emit: (Int32) -> Bool) throws -> Int {
        guard busy.try() else { throw MetalError.invalidArgument("session is already executing a forward pass") }
        defer { busy.unlock() }
        let c = model.config
        guard startToken >= 0, Int(startToken) < c.vocabSize else { throw MetalError.invalidArgument("token id outside vocabulary of \(c.vocabSize)") }
        let budget = min(maxSteps, capacity - length)
        guard budget > 0 else { return 0 }
        chainToken.contents().storeBytes(of: startToken, as: Int32.self)
        var inflight: [MTLCommandBuffer] = []
        var submitted = 0, fed = 0
        var gpuSeconds = 0.0
        func submit() throws {
            let provider = try EncoderProvider(context: model.context, queue: queue, profile: false, precision: gemmPrecision, scratch: dequantScratch, scratch2: dequantScratch2)
            try encodeChunk(provider, tokens: chainToken, tokenOffset: 0, count: 1, startPosition: length + submitted, computeLogits: true)
            try provider.stage("sample").argmax(logits: logits, count: c.vocabSize, chainToken: chainToken, emitted: chainEmitted, slot: submitted)
            inflight.append(try provider.commitWithoutWaiting())
            submitted += 1
        }
        func drain() { for cmd in inflight { cmd.waitUntilCompleted() }; inflight.removeAll() }
        do {
            while submitted < min(depth, budget) { try submit() }
            while !inflight.isEmpty {
                let cmd = inflight.removeFirst()
                cmd.waitUntilCompleted()
                guard cmd.status == .completed else { drain(); throw MetalError.commandFailed(cmd.error.map { "\($0)" } ?? "status \(cmd.status.rawValue)") }
                gpuSeconds += cmd.gpuEndTime - cmd.gpuStartTime
                let token = chainEmitted.contents().load(fromByteOffset: fed * 4, as: Int32.self)
                fed += 1
                if !emit(token) { break }
                if submitted < budget { try submit() }
            }
        } catch { drain(); length += fed; throw error }
        drain()
        length += fed
        lastGPUSeconds = gpuSeconds
        return fed
    }

    /// Appends `tokens` at positions `length..<length+tokens.count` and returns the logits for the last token.
    /// Prompts longer than `maxBatch` are processed in chunks inside a single command buffer.
    public func forward(tokens: [Int32]) throws -> [Float] {
        guard busy.try() else { throw MetalError.invalidArgument("session is already executing a forward pass") }
        defer { busy.unlock() }
        let c = model.config
        guard !tokens.isEmpty else { throw MetalError.invalidArgument("forward called with no tokens") }
        guard length + tokens.count <= capacity else {
            throw MetalError.invalidArgument("context exhausted: \(length) cached + \(tokens.count) new > capacity \(capacity)")
        }
        guard tokens.allSatisfy({ $0 >= 0 && Int($0) < c.vocabSize }) else { throw MetalError.invalidArgument("token id outside vocabulary of \(c.vocabSize)") }
        tokens.withUnsafeBytes { _ = memcpy(tokenBuffer.contents(), $0.baseAddress!, $0.count) }

        // The tiled attention kernel reads whole 32-position blocks, so it may touch up to 31 rows past the last new
        // token. Those rows are masked, but a NaN there would still poison the MMA (0 × NaN), so zero them first.
        // (Rows never written since allocation are zero pages; rows from earlier use hold finite values.)
        let rowBytes = c.kvWidth * kvPrecision.bytesPerElement, tail = length + tokens.count
        for b in kCache + vCache { memset(b.contents() + tail * rowBytes, 0, KernelEncoder.kvPaddingRows * rowBytes) }
        let provider = try EncoderProvider(context: model.context, queue: queue, profile: profilingEnabled, precision: gemmPrecision, scratch: dequantScratch, scratch2: dequantScratch2)
        let encodeStart = Date()
        var chunkStart = 0
        while chunkStart < tokens.count {
            let n = min(maxBatch, tokens.count - chunkStart)
            try encodeChunk(provider, tokens: tokenBuffer, tokenOffset: chunkStart, count: n, startPosition: length + chunkStart,
                            computeLogits: chunkStart + n == tokens.count)
            chunkStart += n
        }
        lastEncodeSeconds = Date().timeIntervalSince(encodeStart)
        let result = try provider.finish()
        lastGPUSeconds = result.gpuSeconds
        lastProfile = result.stages
        length += tokens.count
        return Array(UnsafeBufferPointer(start: logits.contents().assumingMemoryBound(to: Float.self), count: c.vocabSize))
    }

    private func encodeChunk(_ p: EncoderProvider, tokens tokenSource: MTLBuffer, tokenOffset: Int, count n: Int, startPosition: Int, computeLogits: Bool) throws {
        let c = model.config, h = c.hiddenSize
        try p.stage("embed").embed(tokens: tokenSource, tokensOffset: tokenOffset * 4, table: model.embedding, out: x, count: n)
        // Chunks of at least `tensorGEMMMinTokens` tokens keep GEMM inputs in half precision end to end (norm, attention and
        // SwiGLU write half), halving activation traffic into the tensor-op GEMMs.
        let halfPath = halfPathAvailable && n >= KernelEncoder.tensorGEMMMinTokens
        let tiledAttention = kvPrecision == .float16 && n >= KernelEncoder.prefillAttentionMinTokens && (c.headDim == 64 || c.headDim == 128)
        for (i, l) in model.layers.enumerated() {
            if halfPath, let xnH = xnHalf, let ctxH = ctxHalf, let gateH = gateHalf {
                try p.stage("rmsnorm").rmsNormToHalf(x, weight: l.attnNorm, out: xnH, rows: n, dim: h, eps: c.rmsNormEps)
                let qkv = p.stage("qkv projections")
                try qkv.projectHalf(l.wq, xHalf: xnH, y: q, tokens: n, accumulate: false)
                try qkv.projectHalf(l.wk, xHalf: xnH, y: k, tokens: n, accumulate: false)
                try qkv.projectHalf(l.wv, xHalf: xnH, y: v, tokens: n, accumulate: false)
                try p.stage("rope + kv store").ropeQKVStore(q: q, k: k, v: v, kCache: kCache[i], vCache: vCache[i], freqs: model.ropeFreqs,
                                                            heads: c.headCount, kvHeads: c.kvHeadCount, headDim: c.headDim,
                                                            startPosition: startPosition, tokens: n, kv: kvPrecision)
                let att = p.stage("attention")
                if tiledAttention {
                    try att.attention(q: q, kCache: kCache[i], vCache: vCache[i], out: ctxH, heads: c.headCount, kvHeads: c.kvHeadCount,
                                      headDim: c.headDim, startPosition: startPosition, tokens: n, kv: kvPrecision, outputHalf: true)
                } else {
                    try att.attention(q: q, kCache: kCache[i], vCache: vCache[i], out: ctx, heads: c.headCount, kvHeads: c.kvHeadCount,
                                      headDim: c.headDim, startPosition: startPosition, tokens: n, kv: kvPrecision, decodeScratch: decodeScratch)
                    try att.convertToHalf(ctx, out: ctxH, count: n * c.queryWidth)
                }
                try p.stage("attn out projection (+residual)").projectHalf(l.wo, xHalf: ctxH, y: x, tokens: n, accumulate: true)
                try p.stage("rmsnorm").rmsNormToHalf(x, weight: l.ffnNorm, out: xnH, rows: n, dim: h, eps: c.rmsNormEps)
                try p.stage("ffn gate+up+silu").gateUpSiluHalf(l.wGate, l.wUp, xHalf: xnH, outHalf: gateH, tokens: n)
                try p.stage("ffn down (+residual)").projectHalf(l.wDown, xHalf: gateH, y: x, tokens: n, accumulate: true)
                continue
            }
            if [[l.wq, l.wk, l.wv], [l.wo], [l.wGate, l.wUp], [l.wDown]].allSatisfy({ KernelEncoder.canFuseDecode($0, tokens: n) }) {
                // Single-token path: 6 dispatches per layer instead of 13 (norms, residuals, SiLU and RoPE folded into neighbours).
                p.stage("qkv (fused norm)").decodeQKV(l.wq, l.wk, l.wv, x: x, norm: (l.attnNorm, c.rmsNormEps), q: q, k: k, v: v)
                try p.stage("rope + kv store").ropeQKVStore(q: q, k: k, v: v, kCache: kCache[i], vCache: vCache[i], freqs: model.ropeFreqs,
                                                            heads: c.headCount, kvHeads: c.kvHeadCount, headDim: c.headDim,
                                                            startPosition: startPosition, tokens: 1, kv: kvPrecision)
                try p.stage("attention").attention(q: q, kCache: kCache[i], vCache: vCache[i], out: ctx, heads: c.headCount, kvHeads: c.kvHeadCount,
                                                   headDim: c.headDim, startPosition: startPosition, tokens: 1, kv: kvPrecision, decodeScratch: decodeScratch)
                p.stage("attn out projection (+residual)").decodeProjection(l.wo, x: ctx, out: x, add: true)
                p.stage("ffn gate+up+silu (fused norm)").decodeGateUp(l.wGate, l.wUp, x: x, norm: (l.ffnNorm, c.rmsNormEps), out: gate)
                p.stage("ffn down (+residual)").decodeProjection(l.wDown, x: gate, out: x, add: true)
                continue
            }
            try p.stage("rmsnorm").rmsNorm(x, weight: l.attnNorm.buffer, weightOffset: l.attnNorm.offset, out: xn, rows: n, dim: h, eps: c.rmsNormEps)
            let qkv = p.stage("qkv projections")
            try qkv.linear(l.wq, x: xn, y: q, tokens: n)
            try qkv.linear(l.wk, x: xn, y: k, tokens: n)
            try qkv.linear(l.wv, x: xn, y: v, tokens: n)
            try p.stage("rope + kv store").ropeQKVStore(q: q, k: k, v: v, kCache: kCache[i], vCache: vCache[i], freqs: model.ropeFreqs,
                                                        heads: c.headCount, kvHeads: c.kvHeadCount, headDim: c.headDim,
                                                        startPosition: startPosition, tokens: n, kv: kvPrecision)
            try p.stage("attention").attention(q: q, kCache: kCache[i], vCache: vCache[i], out: ctx, heads: c.headCount, kvHeads: c.kvHeadCount,
                                               headDim: c.headDim, startPosition: startPosition, tokens: n, kv: kvPrecision, decodeScratch: decodeScratch)
            try p.stage("attn out projection (+residual)").linearAdd(l.wo, x: ctx, residual: x, tokens: n)

            try p.stage("rmsnorm").rmsNorm(x, weight: l.ffnNorm.buffer, weightOffset: l.ffnNorm.offset, out: xn, rows: n, dim: h, eps: c.rmsNormEps)
            try p.stage("ffn gate+up+silu").gateUpSilu(l.wGate, l.wUp, x: xn, out: gate, scratchUp: up, tokens: n)
            try p.stage("ffn down (+residual)").linearAdd(l.wDown, x: gate, residual: x, tokens: n)
        }
        if computeLogits {
            // Only the final position's logits are needed: normalise and project that single row.
            let fin = p.stage("final norm + logits")
            if KernelEncoder.canFuseDecode([model.output], tokens: 1) {
                fin.decodeProjection(model.output, x: x, xOffset: (n - 1) * h * 4, norm: (model.outputNorm, c.rmsNormEps), out: logits)
            } else {
                try fin.rmsNorm(x, xOffset: (n - 1) * h * 4, weight: model.outputNorm.buffer, weightOffset: model.outputNorm.offset,
                                out: xn, rows: 1, dim: h, eps: c.rmsNormEps)
                try fin.linear(model.output, x: xn, y: logits, tokens: 1)
            }
        }
    }
}

/// Hands out compute encoders. Normally every stage shares one encoder in one command buffer. In profiling mode
/// each stage change starts a new encoder in the *same* command buffer with GPU timestamp samples at its start and end
/// (stage-boundary counter sampling), so per-stage GPU time is measured without commit/wait gaps. Encoder boundaries
/// still serialise work and add a small launch cost, so the stage sum slightly exceeds an unprofiled run.
final class EncoderProvider {
    let context: MetalContext
    let queue: MTLCommandQueue
    let profile: Bool
    private let cmd: MTLCommandBuffer
    private var enc: MTLComputeCommandEncoder?
    private var kernels: KernelEncoder?
    private var currentLabel = ""
    private var invocations = 0
    private var segments: [(label: String, invocations: Int)] = []
    private var sampleBuffer: MTLCounterSampleBuffer?
    private static let maxSegments = 2048   // sample buffer limit: 32 KiB = 4096 timestamps

    private let precision: GEMMPrecision
    private let scratch: MTLBuffer?
    private let scratch2: MTLBuffer?
    private func makeKernels(_ e: MTLComputeCommandEncoder) -> KernelEncoder {
        var k = KernelEncoder(context: context, encoder: e)
        k.precision = precision; k.dequantScratch = scratch; k.dequantScratch2 = scratch2
        return k
    }

    init(context: MetalContext, queue: MTLCommandQueue, profile: Bool, precision: GEMMPrecision, scratch: MTLBuffer?, scratch2: MTLBuffer? = nil) throws {
        self.context = context; self.profile = profile; self.precision = precision; self.scratch = scratch; self.scratch2 = scratch2
        self.queue = queue
        guard let c = queue.makeCommandBuffer() else { throw MetalError.commandFailed("could not create command buffer") }
        cmd = c
        if profile {
            guard context.device.supportsCounterSampling(.atStageBoundary),
                  let set = context.device.counterSets?.first(where: { $0.name == MTLCommonCounterSet.timestamp.rawValue })
            else { throw MetalError.unsupported("GPU stage-boundary timestamp sampling") }
            let d = MTLCounterSampleBufferDescriptor()
            d.counterSet = set; d.storageMode = .shared; d.sampleCount = 2 * Self.maxSegments
            sampleBuffer = try context.device.makeCounterSampleBuffer(descriptor: d)
        } else {
            guard let e = c.makeComputeCommandEncoder() else { throw MetalError.commandFailed("could not create encoder") }
            enc = e; kernels = makeKernels(e)
        }
    }

    func stage(_ label: String) -> KernelEncoder {
        guard profile else { return kernels! }
        invocations += 1
        if label != currentLabel || kernels == nil {
            endSegment()
            guard segments.count < Self.maxSegments else { return kernels ?? makeKernels(enc!) }
            let desc = MTLComputePassDescriptor()
            let i = segments.count
            desc.sampleBufferAttachments[0].sampleBuffer = sampleBuffer
            desc.sampleBufferAttachments[0].startOfEncoderSampleIndex = 2 * i
            desc.sampleBufferAttachments[0].endOfEncoderSampleIndex = 2 * i + 1
            let e = cmd.makeComputeCommandEncoder(descriptor: desc)!
            enc = e; kernels = makeKernels(e)
            currentLabel = label; invocations = 1
        }
        return kernels!
    }

    private func endSegment() {
        guard profile, let e = enc, !currentLabel.isEmpty else { return }
        e.endEncoding(); enc = nil; kernels = nil
        segments.append((currentLabel, invocations))
        currentLabel = ""
    }

    /// Ends encoding and commits the (non-profiled) command buffer without waiting for it.
    func commitWithoutWaiting() throws -> MTLCommandBuffer {
        precondition(!profile)
        enc?.endEncoding(); enc = nil
        cmd.commit()
        return cmd
    }

    func finish() throws -> (gpuSeconds: Double, stages: [String: (seconds: Double, dispatches: Int)]) {
        endSegment()
        if !profile { enc?.endEncoding() }
        cmd.commit(); cmd.waitUntilCompleted()
        guard cmd.status == .completed else { throw MetalError.commandFailed(cmd.error.map { "\($0)" } ?? "status \(cmd.status.rawValue)") }
        let total = cmd.gpuEndTime - cmd.gpuStartTime
        guard profile, let sb = sampleBuffer, !segments.isEmpty else { return (total, [:]) }
        guard let data = try sb.resolveCounterRange(0..<(2 * segments.count)) else { throw MetalError.commandFailed("could not resolve GPU timestamps") }
        let ticks = data.withUnsafeBytes { Array($0.bindMemory(to: MTLCounterResultTimestamp.self)).map { $0.timestamp } }
        // Calibrate ticks → seconds from this command buffer's own span (first start … last end).
        let span = Double(ticks[ticks.count - 1] &- ticks[0])
        let perTick = span > 0 ? total / span : 0
        var stages: [String: (seconds: Double, dispatches: Int)] = [:]
        for (i, seg) in segments.enumerated() {
            let t = Double(ticks[2 * i + 1] &- ticks[2 * i]) * perTick
            let prev = stages[seg.label] ?? (0, 0)
            stages[seg.label] = (prev.seconds + t, prev.dispatches + seg.invocations)
        }
        return (total, stages)
    }
}
