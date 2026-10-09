// Copyright (c) 2026 Torin Etheridge. MIT License (see LICENSE).
// Author: Torin Etheridge · 2026-10-09

import AlpacaCore

/// GGML tensor storage types that can appear in a GGUF file. Only sizes are needed to validate the
/// container; the loader decides which types it can execute.
public enum GGMLType: UInt32, Sendable {
    case f32 = 0, f16 = 1, q4_0 = 2, q4_1 = 3, q5_0 = 6, q5_1 = 7, q8_0 = 8, q8_1 = 9
    case q2_K = 10, q3_K = 11, q4_K = 12, q5_K = 13, q6_K = 14, q8_K = 15
    case bf16 = 30

    /// (elements per block, bytes per block)
    public var blockLayout: (elements: Int, bytes: Int) {
        switch self {
        case .f32: return (1, 4)
        case .f16, .bf16: return (1, 2)
        case .q4_0: return (32, 18)
        case .q4_1: return (32, 20)
        case .q5_0: return (32, 22)
        case .q5_1: return (32, 24)
        case .q8_0: return (32, 34)
        case .q8_1: return (32, 36)
        case .q2_K: return (256, 84)
        case .q3_K: return (256, 110)
        case .q4_K: return (256, 144)
        case .q5_K: return (256, 176)
        case .q6_K: return (256, 210)
        case .q8_K: return (256, 292)
        }
    }

    /// The AlpacaCore dtype that executes this type directly, if any.
    public var nativeDType: DType? {
        switch self {
        case .f32: return .float32
        case .f16: return .float16
        case .q8_0: return .q8_0
        case .q4_0: return .q4_0
        default: return nil
        }
    }
}

public enum GGUFValue: Sendable, Equatable {
    case uint(UInt64)
    case int(Int64)
    case float(Double)
    case bool(Bool)
    case string(String)
    case array([GGUFValue])

    public var asInt: Int? {
        switch self {
        case .uint(let v): return Int(exactly: v)
        case .int(let v): return Int(exactly: v)
        default: return nil
        }
    }
    public var asDouble: Double? {
        switch self {
        case .float(let v): return v
        case .uint(let v): return Double(v)
        case .int(let v): return Double(v)
        default: return nil
        }
    }
    public var asString: String? { if case .string(let s) = self { return s } else { return nil } }
    public var asBool: Bool? { if case .bool(let b) = self { return b } else { return nil } }
    public var asArray: [GGUFValue]? { if case .array(let a) = self { return a } else { return nil } }
}

public struct GGUFTensorInfo: Sendable {
    public let name: String
    /// Dimensions in GGUF order: dims[0] is the fastest-varying (row length).
    public let dims: [Int]
    public let type: GGMLType
    /// Absolute byte offset of the tensor data within the file.
    public let fileOffset: Int
    public let byteCount: Int
    public var elementCount: Int { dims.reduce(1, *) }
    /// Row-major shape as AlpacaCore expects it (outermost dimension first).
    public var shape: [Int] { dims.reversed() }
}
