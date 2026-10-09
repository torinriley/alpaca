// Copyright (c) 2026 Torin Etheridge. MIT License (see LICENSE).
// Author: Torin Etheridge · 2026-10-09

import Foundation

/// One generated token. `text` is the newly completed UTF-8 text: it can be empty while a multi-byte
/// character is still incomplete, and a single token can complete text begun by earlier tokens.
public struct Token: Sendable {
    public let id: Int32
    public let text: String
}

public enum FinishReason: Sendable, Equatable {
    case endOfSequence
    case maxTokens
    case contextFull
    case cancelled
}

public struct GenerationSummary: Sendable {
    public var promptTokens = 0
    public var generatedTokens = 0
    public var finishReason: FinishReason = .maxTokens
    /// Seconds spent processing the prompt (until the logits for the first generated token exist).
    public var prefillSeconds = 0.0
    /// Seconds from the first generated token to the last (decode phase only).
    public var decodeSeconds = 0.0
    /// Wall time from the start of `generate` to the first emitted token.
    public var timeToFirstToken = 0.0
    /// GPU execution time summed over all forward passes (command-buffer timestamps); 0 on the CPU backend.
    public var gpuSeconds = 0.0
    public var prefillTokensPerSecond: Double { prefillSeconds > 0 ? Double(promptTokens) / prefillSeconds : 0 }
    /// Decode throughput over tokens 2…n (token 1 is produced by prefill).
    public var decodeTokensPerSecond: Double { decodeSeconds > 0 && generatedTokens > 1 ? Double(generatedTokens - 1) / decodeSeconds : 0 }
}

/// Async sequence of tokens. Cancel by cancelling the consuming task or by calling `cancel()`; either releases the
/// session's KV cache and GPU buffers promptly. `summary` is complete once iteration ends.
public final class GenerationStream: AsyncSequence, @unchecked Sendable {
    public typealias Element = Token
    private let stream: AsyncThrowingStream<Token, Error>
    private let state: GenerationState

    init(stream: AsyncThrowingStream<Token, Error>, state: GenerationState) { self.stream = stream; self.state = state }

    public func makeAsyncIterator() -> AsyncThrowingStream<Token, Error>.AsyncIterator { stream.makeAsyncIterator() }
    public func cancel() { state.cancel() }
    public var summary: GenerationSummary { state.summary }
    /// Convenience: concatenated text of the whole generation.
    public func text() async throws -> String {
        var out = ""
        for try await t in self { out += t.text }
        return out
    }
}

final class GenerationState: @unchecked Sendable {
    private let lock = NSLock()
    private var _cancelled = false
    private var _summary = GenerationSummary()
    var onCancel: (@Sendable () -> Void)?

    var isCancelled: Bool { lock.lock(); defer { lock.unlock() }; return _cancelled }
    var summary: GenerationSummary { lock.lock(); defer { lock.unlock() }; return _summary }
    func update(_ body: (inout GenerationSummary) -> Void) { lock.lock(); body(&_summary); lock.unlock() }
    func cancel() { lock.lock(); _cancelled = true; lock.unlock() }
}

/// GPU time attributed to one pipeline stage during a profiled forward pass.
public struct StageProfile: Sendable {
    public let name: String
    public let milliseconds: Double
    public let share: Double
    public let invocations: Int
}

/// Where one forward pass spent its time. Stage times come from GPU timestamps taken at encoder boundaries; separate
/// encoders serialise work and add launch cost, so the stage sum can exceed `unprofiledGPUMilliseconds`, which is the true figure.
public struct ForwardProfile: Sendable {
    public let newTokens: Int
    public let contextBefore: Int
    public let stages: [StageProfile]
    public let profiledGPUMilliseconds: Double
    public let unprofiledGPUMilliseconds: Double
    /// CPU time spent encoding commands (before commit) in an unprofiled pass.
    public let encodeMilliseconds: Double
    /// Wall time of an unprofiled pass (encode + GPU + completion latency).
    public let wallMilliseconds: Double
}
