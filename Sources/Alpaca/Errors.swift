// Copyright (c) 2026 Torin Etheridge. MIT License (see LICENSE).
// Author: Torin Etheridge · 2026-10-09

import Foundation

public enum AlpacaError: Error, CustomStringConvertible {
    case modelLoadFailed(URL, underlying: any Error)
    case insufficientMemory(requiredBytes: Int, budgetBytes: Int, breakdown: String)
    case backendUnavailable(String)
    case invalidConfiguration(String)
    case contextExhausted(promptTokens: Int, capacity: Int)
    case modelUnloaded
    case tokenizationFailed(String)
    case inferenceFailed(String)

    public var description: String {
        switch self {
        case .modelLoadFailed(let url, let e): return "could not load \(url.lastPathComponent): \(e)"
        case .insufficientMemory(let need, let budget, let b):
            return "model needs about \(need >> 20) MiB but the budget is \(budget >> 20) MiB (\(b)); lower contextLength, use a smaller quantization, or raise memoryBudgetBytes"
        case .backendUnavailable(let m): return "backend unavailable: \(m)"
        case .invalidConfiguration(let m): return "invalid configuration: \(m)"
        case .contextExhausted(let p, let c): return "prompt of \(p) tokens does not fit in a context of \(c)"
        case .modelUnloaded: return "the model has been unloaded"
        case .tokenizationFailed(let m): return "tokenization failed: \(m)"
        case .inferenceFailed(let m): return "inference failed: \(m)"
        }
    }
}
