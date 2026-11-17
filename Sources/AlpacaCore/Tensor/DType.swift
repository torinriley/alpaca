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
    /// GGUF Q4_1 (32 elements, 20 B: half d, half m, 16 nibble bytes laid out like Q4_0): value = d * nibble + m.
    case q4_1
    /// GGUF Q5_0 (32 elements, 22 B: half d, uint32 qh, 16 nibble bytes): value = d * (5-bit - 16).
    case q5_0
    /// GGUF Q5_1 (32 elements, 24 B: half d, half m, uint32 qh, 16 nibble bytes): value = d * 5-bit + m.
    case q5_1
    /// GGUF K-quants: 256-element super-blocks (see `Docs/FORMATS.md` for the byte layouts).
    case q2_K, q3_K, q4_K, q5_K, q6_K
    /// bfloat16 (element-addressable, 2 bytes).
    case bfloat16

    /// Number of logical elements per storage block.
    public var blockElements: Int {
        switch self {
        case .float32, .float16, .bfloat16: return 1
        case .q8_0, .q4_0, .q4_1, .q5_0, .q5_1: return quantBlockElements
        case .q2_K, .q3_K, .q4_K, .q5_K, .q6_K: return kQuantBlockElements
        }
    }

    /// Bytes per storage block.
    public var blockBytes: Int {
        switch self {
        case .float32: return 4
        case .float16, .bfloat16: return 2
        case .q8_0: return 34
        case .q4_0: return 18
        case .q4_1: return 20
        case .q5_0: return 22
        case .q5_1: return 24
        case .q2_K: return 84
        case .q3_K: return 110
        case .q4_K: return 144
        case .q5_K: return 176
        case .q6_K: return 210
        }
    }

    public var isQuantized: Bool { blockElements > 1 }

    public var description: String {
        switch self {
        case .float32: return "f32"
        case .float16: return "f16"
        case .q8_0: return "q8_0"
        case .q4_0: return "q4_0"
        case .q4_1: return "q4_1"
        case .q5_0: return "q5_0"
        case .q5_1: return "q5_1"
        case .q2_K: return "q2_k"
        case .q3_K: return "q3_k"
        case .q4_K: return "q4_k"
        case .q5_K: return "q5_k"
        case .q6_K: return "q6_k"
        case .bfloat16: return "bf16"
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
/// Elements per GGUF K-quant super-block (QK_K).
public let kQuantBlockElements = 256
