// Copyright (c) 2026 Torin Etheridge. MIT License (see LICENSE).
// Author: Torin Etheridge · 2026-10-09

import Foundation

/// Minimal GGUF writer for tests. Deliberately independent of the parser under test.
struct GGUFBuilder {
    enum Value {
        case u8(UInt8), i8(Int8), u16(UInt16), i16(Int16), u32(UInt32), i32(Int32), f32(Float), bool(Bool)
        case str(String), u64(UInt64), i64(Int64), f64(Double)
        case array(elementType: UInt32, [Value])

        var typeID: UInt32 {
            switch self {
            case .u8: return 0; case .i8: return 1; case .u16: return 2; case .i16: return 3; case .u32: return 4
            case .i32: return 5; case .f32: return 6; case .bool: return 7; case .str: return 8; case .array: return 9
            case .u64: return 10; case .i64: return 11; case .f64: return 12
            }
        }
    }
    struct Tensor { var name: String; var dims: [UInt64]; var type: UInt32; var data: Data; var offsetOverride: UInt64? = nil }

    var version: UInt32 = 3
    var metadata: [(String, Value)] = []
    var tensors: [Tensor] = []
    var alignment = 32

    static func le<T: FixedWidthInteger>(_ v: T) -> Data { withUnsafeBytes(of: v.littleEndian) { Data($0) } }
    static func string(_ s: String) -> Data { le(UInt64(s.utf8.count)) + Data(s.utf8) }
    static func encode(_ v: Value) -> Data {
        switch v {
        case .u8(let x): return le(x); case .i8(let x): return le(x); case .u16(let x): return le(x); case .i16(let x): return le(x)
        case .u32(let x): return le(x); case .i32(let x): return le(x); case .u64(let x): return le(x); case .i64(let x): return le(x)
        case .f32(let x): return le(x.bitPattern); case .f64(let x): return le(x.bitPattern)
        case .bool(let b): return le(UInt8(b ? 1 : 0))
        case .str(let s): return string(s)
        case .array(let t, let items): return le(t) + le(UInt64(items.count)) + items.reduce(Data()) { $0 + encode($1) }
        }
    }

    func build() -> Data {
        var d = Data("GGUF".utf8) + Self.le(version) + Self.le(UInt64(tensors.count)) + Self.le(UInt64(metadata.count))
        var md = metadata
        if alignment != 32 { md.append(("general.alignment", .u32(UInt32(alignment)))) }
        d = Data("GGUF".utf8) + Self.le(version) + Self.le(UInt64(tensors.count)) + Self.le(UInt64(md.count))
        for (k, v) in md { d += Self.string(k) + Self.le(v.typeID) + Self.encode(v) }
        var offset: UInt64 = 0
        var payload = Data()
        for t in tensors {
            d += Self.string(t.name) + Self.le(UInt32(t.dims.count))
            for x in t.dims { d += Self.le(x) }
            d += Self.le(t.type) + Self.le(t.offsetOverride ?? offset)
            payload += t.data
            let pad = (alignment - payload.count % alignment) % alignment
            payload += Data(count: pad)
            offset = UInt64(payload.count)
        }
        d += Data(count: (alignment - d.count % alignment) % alignment)
        return d + payload
    }
}

extension GGUFBuilder.Tensor {
    static func f32(_ name: String, dims: [UInt64], values: [Float]) -> Self {
        let data = values.reduce(Data()) { $0 + GGUFBuilder.le($1.bitPattern) }
        return .init(name: name, dims: dims, type: 0, data: data)
    }
}
