// Copyright (c) 2026 Torin Etheridge. MIT License (see LICENSE).
// Author: Torin Etheridge · 2026-10-09

import Foundation
import AlpacaCore
import AlpacaMetal
import AlpacaModels
import AlpacaTokenizers

/// A loaded language model. Immutable weights are shared; every `generate` call gets its own isolated session
/// (KV cache + scratch), so concurrent generations can never corrupt each other.
public final class LanguageModel: @unchecked Sendable {
    public struct Info: Sendable {
        public let name: String
        public let architecture: String
        public let parameterBytes: Int
        public let layerCount: Int
        public let hiddenSize: Int
        public let vocabularySize: Int
        public let trainedContextLength: Int
        public let contextLength: Int
        public let backend: Backend
        public let estimatedMemory: MemoryEstimate
        /// Notes from loading (e.g. tensors converted from unsupported-in-place formats).
        public let loadNotes: [String]
    }

    public let info: Info
    let config: LlamaConfig
    let tokenizer: BPETokenizer
    let options: LoadOptions
    private let loaded: LoadedLlama
    private var cpu: LlamaCPUModel?
    private var gpu: MetalLlamaModel?
    private let lock = NSLock()
    private var activeGenerations: [UUID: GenerationState] = [:]
    private var unloaded = false
    /// Finished sessions kept for reuse: a fresh session costs buffer allocation and first-touch page faults on every KV
    /// cache and scratch buffer (several ms of time-to-first-token). At most `maxIdleSessions` are retained.
    private var idleSessions: [any InferenceSession] = []
    private static let maxIdleSessions = 1
    private static let greedyChainDisabled = ProcessInfo.processInfo.environment["ALPACA_DISABLE_GREEDY_CHAIN"] != nil

    /// Loads and validates a GGUF model off the calling actor. Fails with a descriptive `AlpacaError` before
    /// allocating GPU/KV memory if the estimated footprint exceeds the budget.
    public static func load(from url: URL, options: LoadOptions = LoadOptions()) async throws -> LanguageModel {
        try await Task.detached(priority: .userInitiated) { try LanguageModel(url: url, options: options) }.value
    }

    init(url: URL, options: LoadOptions) throws {
        if options.prefillBatchSize < 1 { throw AlpacaError.invalidConfiguration("prefillBatchSize must be positive") }
        let loaded: LoadedLlama, tokenizer: BPETokenizer
        do {
            loaded = try LlamaLoader.load(url: url)
            tokenizer = try loaded.file.makeTokenizer()
        } catch { throw AlpacaError.modelLoadFailed(url, underlying: error) }
        guard tokenizer.vocabularySize == loaded.config.vocabSize || tokenizer.vocabularySize <= loaded.config.vocabSize else {
            throw AlpacaError.modelLoadFailed(url, underlying: ConfigError("tokenizer has \(tokenizer.vocabularySize) tokens but the model's vocabulary is \(loaded.config.vocabSize)"))
        }
        let config = loaded.config
        let context = options.contextLength ?? min(config.contextLength, 4096)
        guard context > 0, context <= config.contextLength else {
            throw AlpacaError.invalidConfiguration("contextLength \(context) must be in 1...\(config.contextLength)")
        }

        var backend = options.backend
        if backend == .automatic { backend = MetalContext.isAvailable ? .metal : .cpu }
        if backend == .metal && !MetalContext.isAvailable { throw AlpacaError.backendUnavailable("no Metal device found") }

        let kvBytes = backend == .metal ? options.kvPrecision.bytesPerElement : 4
        // `.fast` prefill expands the largest quantised matrix to half in two per-session scratch buffers (gate and up).
        let w = loaded.weights
        let quantised = w.layers.flatMap { [$0.wq, $0.wk, $0.wv, $0.wo, $0.wGate, $0.wUp, $0.wDown] }   // output projection is mat-vec only
        let dequantScratch = (backend == .metal && options.prefillPrecision == .fast)
            ? 2 * (quantised.filter { $0.dtype.isQuantized }.map { $0.elementCount * 2 }.max() ?? 0) : 0
        let estimate = MemoryEstimate(weightBytes: loaded.report.mappedWeightBytes + loaded.report.convertedWeightBytes, config: config,
                                      contextLength: context, kvBytesPerElement: kvBytes, prefillBatch: options.prefillBatchSize,
                                      extraScratchBytes: dequantScratch)
        let budget = options.memoryBudgetBytes ?? MemoryBudget.defaultBytes()
        guard estimate.totalBytes <= budget else {
            throw AlpacaError.insufficientMemory(requiredBytes: estimate.totalBytes, budgetBytes: budget, breakdown: estimate.description)
        }

        do {
            switch backend {
            case .metal:
                gpu = try MetalLlamaModel(context: MetalContext(), config: config, weights: loaded.weights,
                                          mappedRegion: loaded.file.mappedMemory, keepAlive: loaded.file.mappingOwner)
            default:
                cpu = try LlamaCPUModel(config: config, weights: loaded.weights)
            }
        } catch { throw AlpacaError.modelLoadFailed(url, underlying: error) }

        self.loaded = loaded; self.tokenizer = tokenizer; self.config = config
        var resolved = options; resolved.contextLength = context
        self.options = resolved
        info = Info(name: loaded.file.string("general.name") ?? url.deletingPathExtension().lastPathComponent,
                    architecture: "llama", parameterBytes: estimate.weightBytes, layerCount: config.layerCount,
                    hiddenSize: config.hiddenSize, vocabularySize: config.vocabSize, trainedContextLength: config.contextLength,
                    contextLength: context, backend: backend, estimatedMemory: estimate, loadNotes: loaded.report.notes)
    }

    // MARK: Tokenizer access

    public func tokenize(_ text: String, addBeginningOfSequence: Bool = false, parseSpecialTokens: Bool = true) throws -> [Int32] {
        do { return try tokenizer.encode(text, addBOS: addBeginningOfSequence, parseSpecial: parseSpecialTokens) }
        catch { throw AlpacaError.tokenizationFailed("\(error)") }
    }

    public func detokenize(_ tokens: [Int32]) throws -> String {
        do { return try tokenizer.decode(tokens) } catch { throw AlpacaError.tokenizationFailed("\(error)") }
    }

    // MARK: Lifecycle

    /// Releases weights and GPU buffers once running generations finish. New `generate` calls fail with `.modelUnloaded`.
    public func unload() {
        lock.lock(); unloaded = true
        let running = Array(activeGenerations.values)
        cpu = nil; gpu = nil; idleSessions.removeAll()
        lock.unlock()
        running.forEach { $0.cancel() }
    }

    /// Frees memory held for reuse by finished generations (idle KV caches and scratch). Call on memory warnings or when
    /// the app moves to the background; the next generation then pays the one-time session allocation again.
    public func trimMemory() {
        lock.lock(); idleSessions.removeAll(); lock.unlock()
    }

    /// Cancels every running generation. Call from `applicationDidReceiveMemoryWarning` / when entering the background.
    public func cancelAllGenerations() {
        lock.lock(); let running = Array(activeGenerations.values); lock.unlock()
        running.forEach { $0.cancel() }
    }

    public var activeGenerationCount: Int { lock.lock(); defer { lock.unlock() }; return activeGenerations.count }

    // MARK: Profiling

    /// Profiles one forward pass of `newTokens` tokens appended after `contextBefore` cached tokens (Metal only).
    /// Returns the median over `repetitions`. Use `newTokens: 1` for a decode step, a large value for prefill.
    public func profileForward(newTokens: Int, contextBefore: Int, repetitions: Int = 5) throws -> ForwardProfile {
        lock.lock(); let gpu = gpu; lock.unlock()
        guard let gpu else { throw AlpacaError.backendUnavailable("profiling needs the Metal backend") }
        guard newTokens > 0, contextBefore >= 0, contextBefore + newTokens <= options.contextLength! else {
            throw AlpacaError.invalidConfiguration("contextBefore + newTokens must fit in the context of \(options.contextLength!)")
        }
        let corpus = try tokenize(String(repeating: "The quick brown fox jumps over the lazy dog while the transformer predicts the next token. ", count: (contextBefore + newTokens) / 8 + 20))
        let ids = Array(corpus.prefix(contextBefore + newTokens))
        func median(_ xs: [Double]) -> Double { xs.sorted()[xs.count / 2] }
        var perStage: [String: [Double]] = [:], calls: [String: Int] = [:]
        var profiled: [Double] = [], plainGPU: [Double] = [], encode: [Double] = [], wall: [Double] = []
        do {
            let session = try gpu.makeSession(capacity: options.contextLength, maxBatch: options.prefillBatchSize, kvPrecision: options.kvPrecision, gemmPrecision: options.prefillPrecision)
            for rep in 0..<(repetitions + 1) {          // first repetition is a warm-up
                session.reset()
                if contextBefore > 0 { _ = try session.forward(tokens: Array(ids.prefix(contextBefore))) }
                session.profilingEnabled = false
                let t0 = Date()
                session.reset(); if contextBefore > 0 { _ = try session.forward(tokens: Array(ids.prefix(contextBefore))) }
                let w0 = Date()
                _ = try session.forward(tokens: Array(ids.suffix(newTokens)))
                let w = Date().timeIntervalSince(w0)
                _ = t0
                let g = session.lastGPUSeconds, en = session.lastEncodeSeconds
                session.reset(); if contextBefore > 0 { _ = try session.forward(tokens: Array(ids.prefix(contextBefore))) }
                session.profilingEnabled = true
                _ = try session.forward(tokens: Array(ids.suffix(newTokens)))
                session.profilingEnabled = false
                if rep == 0 { continue }
                plainGPU.append(g * 1000); encode.append(en * 1000); wall.append(w * 1000)
                profiled.append(session.lastProfile.values.map(\.seconds).reduce(0, +) * 1000)
                for (k, v) in session.lastProfile { perStage[k, default: []].append(v.seconds * 1000); calls[k] = v.dispatches }
            }
        } catch { throw AlpacaError.inferenceFailed("\(error)") }
        let total = perStage.values.map(median).reduce(0, +)
        let stages = perStage.map { StageProfile(name: $0.key, milliseconds: median($0.value), share: median($0.value) / max(total, 1e-9), invocations: calls[$0.key] ?? 0) }
            .sorted { $0.milliseconds > $1.milliseconds }
        return ForwardProfile(newTokens: newTokens, contextBefore: contextBefore, stages: stages, profiledGPUMilliseconds: median(profiled),
                              unprofiledGPUMilliseconds: median(plainGPU), encodeMilliseconds: median(encode), wallMilliseconds: median(wall))
    }

    // MARK: Generation

    /// Streams generated tokens for `prompt`. Argument validation errors are thrown by the stream on first iteration.
    public func generate(prompt: String, configuration: GenerationConfiguration = GenerationConfiguration()) -> GenerationStream {
        start(configuration: configuration) { [self] in
            try tokenize(prompt, addBeginningOfSequence: configuration.addBeginningOfSequence && tokenizer.addsBOSByDefault,
                         parseSpecialTokens: configuration.parseSpecialTokens)
        }
    }

    /// Streams generated tokens continuing an already tokenized prompt (used by benchmarks and chat front-ends).
    public func generate(promptTokens: [Int32], configuration: GenerationConfiguration = GenerationConfiguration()) -> GenerationStream {
        start(configuration: configuration) { promptTokens }
    }

    private func start(configuration: GenerationConfiguration, tokens: @escaping @Sendable () throws -> [Int32]) -> GenerationStream {
        let state = GenerationState()
        let id = UUID()
        let stream = AsyncThrowingStream<Token, Error> { continuation in
            let work: @Sendable () -> Void = { [self] in
                defer { finish(id) }
                do {
                    try run(tokens: tokens, configuration: configuration, state: state) { continuation.yield($0) }
                    continuation.finish()
                } catch { continuation.finish(throwing: error) }
            }
            continuation.onTermination = { _ in state.cancel() }
            lock.lock()
            if unloaded { lock.unlock(); continuation.finish(throwing: AlpacaError.modelUnloaded); return }
            activeGenerations[id] = state
            lock.unlock()
            // A dedicated thread per generation keeps blocking GPU waits off Swift's cooperative pool.
            Thread.detachNewThread { work() }
        }
        return GenerationStream(stream: stream, state: state)
    }

    private func finish(_ id: UUID) { lock.lock(); activeGenerations[id] = nil; lock.unlock() }

    private func recycle(_ session: any InferenceSession) {
        lock.lock(); defer { lock.unlock() }
        if !unloaded && idleSessions.count < Self.maxIdleSessions { session.reset(); idleSessions.append(session) }
    }

    private func makeSession() throws -> any InferenceSession {
        lock.lock()
        if !unloaded, let idle = idleSessions.popLast() { lock.unlock(); return idle }
        let cpu = cpu, gpu = gpu, unloaded = unloaded
        lock.unlock()
        if unloaded { throw AlpacaError.modelUnloaded }
        do {
            if let gpu {
                return MetalSessionAdapter(try gpu.makeSession(capacity: options.contextLength, maxBatch: options.prefillBatchSize, kvPrecision: options.kvPrecision, gemmPrecision: options.prefillPrecision))
            }
            if let cpu { return CPUSessionAdapter(model: cpu, cache: try KVCache(config: config, capacity: options.contextLength)) }
        } catch { throw AlpacaError.inferenceFailed("\(error)") }
        throw AlpacaError.modelUnloaded
    }

    private func run(tokens: () throws -> [Int32], configuration c: GenerationConfiguration, state: GenerationState, emit: (Token) -> Void) throws {
        guard c.maxTokens >= 0 else { throw AlpacaError.invalidConfiguration("maxTokens must be >= 0") }
        var sampler: Sampler
        do { sampler = try Sampler(c.sampling) } catch { throw AlpacaError.invalidConfiguration("\(error)") }
        let promptTokens = try tokens()
        guard promptTokens.allSatisfy({ $0 >= 0 && Int($0) < config.vocabSize }) else { throw AlpacaError.invalidConfiguration("prompt token id outside the vocabulary") }
        guard !promptTokens.isEmpty else { throw AlpacaError.invalidConfiguration("the prompt produced no tokens (empty prompt and no BOS token)") }
        let capacity = options.contextLength!
        guard promptTokens.count <= capacity else { throw AlpacaError.contextExhausted(promptTokens: promptTokens.count, capacity: capacity) }
        state.update { $0.promptTokens = promptTokens.count }
        if c.maxTokens == 0 { state.update { $0.finishReason = .maxTokens }; return }

        let session = try makeSession()
        var healthy = false                      // a session that threw is dropped, not reused
        defer { if healthy { recycle(session) } }
        var decoder = tokenizer.makeStreamDecoder()
        let start = Date()
        var firstTokenTime: Date?, logits: [Float]
        do { logits = try session.forward(tokens: promptTokens) } catch { throw AlpacaError.inferenceFailed("\(error)") }
        state.update { $0.prefillSeconds = Date().timeIntervalSince(start) }

        var stops = c.stopTokens
        if c.stopOnEndOfSequence, let eos = tokenizer.eosTokenID { stops.insert(eos) }
        var generated = 0
        var reason = FinishReason.maxTokens
        // One step of bookkeeping per produced token; returns false when generation must stop.
        func accept(_ token: Int32) -> Bool {
            if state.isCancelled { reason = .cancelled; return false }
            if stops.contains(token) { reason = .endOfSequence; return false }
            if firstTokenTime == nil {
                firstTokenTime = Date()
                state.update { $0.timeToFirstToken = firstTokenTime!.timeIntervalSince(start) }
            }
            let text = (try? decoder.append(token)) ?? ""
            emit(Token(id: token, text: text))
            generated += 1
            state.update { $0.generatedTokens = generated; $0.decodeSeconds = Date().timeIntervalSince(firstTokenTime!) }
            if generated >= c.maxTokens { reason = .maxTokens; return false }
            return true
        }

        var next: Int32
        do { next = try sampler.sample(logits: logits) } catch { throw AlpacaError.inferenceFailed("\(error)") }
        var running = accept(next)
        if running, c.temperature == 0, !Self.greedyChainDisabled {
            // Greedy: the sampling loop runs on the GPU with several steps in flight; the CPU only streams tokens out.
            var stoppedByCaller = false
            let fed = try session.decodeGreedy(startToken: next, maxSteps: c.maxTokens - generated) { token in
                let keepGoing = accept(token)
                if !keepGoing { stoppedByCaller = true }
                return keepGoing
            }
            if fed != nil { running = false; if !stoppedByCaller { reason = .contextFull } }
        }
        while running {
            if session.length >= session.capacity { reason = .contextFull; break }
            // Feed the token back and sample the next one straight from the logits buffer (no copy).
            do { next = try session.decode(token: next) { try sampler.sample(logits: $0) } }
            catch { throw AlpacaError.inferenceFailed("\(error)") }
            running = accept(next)
        }
        let tail = decoder.finish()
        if !tail.isEmpty { emit(Token(id: -1, text: tail)) }
        state.update { $0.finishReason = reason; $0.gpuSeconds = session.gpuSeconds }
        healthy = true
    }
}

// MARK: - Backend adapters

protocol InferenceSession: AnyObject {
    var length: Int { get }
    var capacity: Int { get }
    /// Accumulated GPU execution time of this session's forward passes (0 on the CPU backend).
    var gpuSeconds: Double { get }
    func forward(tokens: [Int32]) throws -> [Float]
    /// Appends one token and hands the next-token logits to `body` (valid only inside the closure).
    func decode<R>(token: Int32, _ body: (UnsafeBufferPointer<Float>) throws -> R) throws -> R
    /// GPU-resident greedy decoding (Metal only): nil if the backend cannot do it. See `MetalLlamaSession.decodeGreedy`.
    func decodeGreedy(startToken: Int32, maxSteps: Int, emit: (Int32) -> Bool) throws -> Int?
    func reset()
}

final class MetalSessionAdapter: InferenceSession {
    let session: MetalLlamaSession
    init(_ s: MetalLlamaSession) { session = s }
    var length: Int { session.length }
    var capacity: Int { session.capacity }
    func reset() { session.reset(); gpuSeconds = 0 }
    private(set) var gpuSeconds = 0.0
    func forward(tokens: [Int32]) throws -> [Float] {
        defer { gpuSeconds += session.lastGPUSeconds }
        return try session.forward(tokens: tokens)
    }
    func decode<R>(token: Int32, _ body: (UnsafeBufferPointer<Float>) throws -> R) throws -> R {
        defer { gpuSeconds += session.lastGPUSeconds }
        return try body(try session.decode(token: token))
    }
    func decodeGreedy(startToken: Int32, maxSteps: Int, emit: (Int32) -> Bool) throws -> Int? {
        defer { gpuSeconds += session.lastGPUSeconds }
        return try session.decodeGreedy(startToken: startToken, maxSteps: maxSteps, emit: emit)
    }
}

final class CPUSessionAdapter: InferenceSession {
    let model: LlamaCPUModel
    let cache: KVCache
    init(model: LlamaCPUModel, cache: KVCache) { self.model = model; self.cache = cache }
    var length: Int { cache.length }
    var capacity: Int { cache.capacity }
    var gpuSeconds: Double { 0 }
    func reset() { cache.reset() }
    func forward(tokens: [Int32]) throws -> [Float] { try model.forward(tokens: tokens, cache: cache).toFloatArray() }
    func decode<R>(token: Int32, _ body: (UnsafeBufferPointer<Float>) throws -> R) throws -> R {
        let logits = try forward(tokens: [token])
        return try logits.withUnsafeBufferPointer(body)
    }
    func decodeGreedy(startToken: Int32, maxSteps: Int, emit: (Int32) -> Bool) throws -> Int? { nil }
}
