// Copyright (c) 2026 Torin Etheridge. MIT License (see LICENSE).
// Author: Torin Etheridge · 2026-10-09

import Foundation

/// Itemised memory a loaded model plus one generation session is expected to need.
public struct MemoryEstimate: Sendable, CustomStringConvertible {
    /// Weights resident for the model's lifetime (file-backed pages for mapped GGUF files).
    public var weightBytes: Int
    /// KV cache for `contextLength` positions.
    public var kvCacheBytes: Int
    /// Activation scratch buffers for one session (prefill chunk × layer widths, logits).
    public var scratchBytes: Int
    /// Fixed allowance for tokenizer tables, pipelines, command buffers and allocator slack.
    public var runtimeOverheadBytes: Int

    public var totalBytes: Int { weightBytes + kvCacheBytes + scratchBytes + runtimeOverheadBytes }

    /// Runtime allowance: vocabulary strings and merge tables for ~50k–150k tokens (tens of MB) plus Metal state.
    /// Conservative fixed figure; measured process footprint is reported by the CLI for comparison.
    public static let defaultRuntimeOverheadBytes = 96 << 20

    public init(weightBytes: Int, config: LlamaConfig, contextLength: Int, kvBytesPerElement: Int, prefillBatch: Int,
                extraScratchBytes: Int = 0, runtimeOverheadBytes: Int = MemoryEstimate.defaultRuntimeOverheadBytes) {
        self.weightBytes = weightBytes
        self.kvCacheBytes = 2 * config.layerCount * contextLength * config.kvWidth * kvBytesPerElement
        self.scratchBytes = 4 * prefillBatch * (3 * config.hiddenSize + 2 * config.queryWidth + 2 * config.kvWidth + 2 * config.feedForwardSize)
            + 4 * config.vocabSize + 4 * contextLength + extraScratchBytes
        self.runtimeOverheadBytes = runtimeOverheadBytes
    }

    public var description: String {
        func mb(_ b: Int) -> String { String(format: "%.1f MiB", Double(b) / 1_048_576) }
        return "weights \(mb(weightBytes)) + KV cache \(mb(kvCacheBytes)) + scratch \(mb(scratchBytes)) + runtime \(mb(runtimeOverheadBytes)) = \(mb(totalBytes))"
    }
}

public enum MemoryBudget {
    /// A conservative default budget: 60% of physical memory on macOS; on iOS the memory the OS says this
    /// process may still allocate before jetsam (`os_proc_available_memory`), with the same 60% cap on physical RAM.
    public static func defaultBytes() -> Int {
        let physical = Int(ProcessInfo.processInfo.physicalMemory)
        #if os(iOS)
        let available = Int(os_proc_available_memory())
        if available > 0 { return min(available, physical * 6 / 10) }
        #endif
        return physical * 6 / 10
    }
}
