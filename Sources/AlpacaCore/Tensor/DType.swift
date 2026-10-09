// Copyright (c) 2026 Torin Etheridge. MIT License (see LICENSE).
// Author: Torin Etheridge · 2026-10-09

/// Element storage formats understood by alpaca.
///
/// `float32` and `float16` are element-addressable. `q8_0` and `q4_0` are GGUF block formats:
/// elements are grouped into blocks of `blockElements` values that share one scale.
public enum DType: Sendable, Hashable, CustomStringConvertible {
    case float32
    case float16
    /// GGUF Q8_0: per 32 elements, `Float16 d` followed by 32 `Int8 q`; value = d * q. 34 bytes/block.
    case q8_0
    /// GGUF Q4_0: per 32 elements, `Float16 d` followed by 16 bytes of packed nibbles.
    /// Byte j holds element j in its low nibble and element j+16 in its high nibble;
    /// value = d * (nibble - 8). 18 bytes/block.
    case q4_0

    /// Number of logical elements per storage block.
    public var blockElements: Int {
        switch self {
        case .float32, .float16: return 1
        case .q8_0, .q4_0: return quantBlockElements
        }
    }

    /// Bytes per storage block.
    public var blockBytes: Int {
        switch self {
        case .float32: return 4
        case .float16: return 2
        case .q8_0: return 34
        case .q4_0: return 18
        }
    }

    public var isQuantized: Bool { blockElements > 1 }

    public var description: String {
        switch self {
        case .float32: return "f32"
        case .float16: return "f16"
        case .q8_0: return "q8_0"
        case .q4_0: return "q4_0"
        }
    }

    /// Bytes needed for `count` elements, or nil if `count` is not a whole number of blocks or overflows.
    public func byteCount(elements count: Int) -> Int? {
        guard count >= 0, count % blockElements == 0 else { return nil }
        let (bytes, overflow) = (count / blockElements).multipliedReportingOverflow(by: blockBytes)
        return overflow ? nil : bytes
    }
}

/// Elements per GGUF Q8_0 / Q4_0 block (QK8_0 / QK4_0).
public let quantBlockElements = 32
