// Copyright (c) 2026 Torin Etheridge. MIT License (see LICENSE).
// Author: Torin Etheridge · 2026-10-09

/// Bounds-checked little-endian cursor over an immutable byte range.
struct ByteReader {
    let base: UnsafeRawPointer
    let count: Int
    private(set) var position: Int = 0

    init(base: UnsafeRawPointer, count: Int, position: Int = 0) { self.base = base; self.count = count; self.position = position }

    var remaining: Int { count - position }

    mutating func read<T: FixedWidthInteger>(_ type: T.Type, _ what: String) throws -> T {
        let size = MemoryLayout<T>.size
        guard remaining >= size else { throw GGUFError.truncated(what: what, offset: position) }
        let v = base.loadUnaligned(fromByteOffset: position, as: T.self)
        position += size
        return T(littleEndian: v)
    }

    mutating func readFloat32(_ what: String) throws -> Float { Float(bitPattern: try read(UInt32.self, what)) }
    mutating func readFloat64(_ what: String) throws -> Double { Double(bitPattern: try read(UInt64.self, what)) }

    mutating func skip(_ n: Int, _ what: String) throws {
        guard n >= 0, remaining >= n else { throw GGUFError.truncated(what: what, offset: position) }
        position += n
    }

    /// GGUF string: u64 length + UTF-8 bytes (no terminator). Invalid UTF-8 is repaired, not trusted to be valid.
    mutating func readString(_ what: String, maxLength: Int) throws -> String {
        let len = try read(UInt64.self, "\(what) length")
        guard len <= UInt64(maxLength) else { throw GGUFError.limitExceeded("\(what) length \(len) > \(maxLength)") }
        let n = Int(len)
        guard remaining >= n else { throw GGUFError.truncated(what: what, offset: position) }
        let s = String(decoding: UnsafeRawBufferPointer(start: base + position, count: n), as: UTF8.self)
        position += n
        return s
    }
}
