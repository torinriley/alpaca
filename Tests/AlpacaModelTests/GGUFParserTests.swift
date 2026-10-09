// Copyright (c) 2026 Torin Etheridge. MIT License (see LICENSE).
// Author: Torin Etheridge · 2026-10-09

import XCTest
@testable import AlpacaModels
import AlpacaCore

final class GGUFParserTests: XCTestCase {
    func sample() -> GGUFBuilder {
        var b = GGUFBuilder()
        b.metadata = [
            ("general.architecture", .str("llama")), ("n.u8", .u8(200)), ("n.i8", .i8(-5)), ("n.u16", .u16(65000)), ("n.i16", .i16(-300)),
            ("n.u32", .u32(4_000_000_000)), ("n.i32", .i32(-7)), ("n.f32", .f32(1.5)), ("n.bool", .bool(true)),
            ("n.u64", .u64(1 << 40)), ("n.i64", .i64(-(1 << 40))), ("n.f64", .f64(2.25)),
            ("arr.str", .array(elementType: 8, [.str("a"), .str("bc")])), ("arr.u32", .array(elementType: 4, [.u32(1), .u32(2), .u32(3)])),
        ]
        b.tensors = [
            .f32("a", dims: [4, 2], values: Array(1...8).map(Float.init)),
            .f32("b", dims: [3], values: [9, 10, 11]),
        ]
        return b
    }

    func testParsesWellFormedFile() throws {
        let f = try GGUFFile(data: sample().build())
        XCTAssertEqual(f.version, 3)
        XCTAssertEqual(f.string("general.architecture"), "llama")
        XCTAssertEqual(f.metadata["n.u8"]?.asInt, 200)
        XCTAssertEqual(f.metadata["n.i8"]?.asInt, -5)
        XCTAssertEqual(f.metadata["n.u32"]?.asInt, 4_000_000_000)
        XCTAssertEqual(f.metadata["n.i64"]?.asInt, -(1 << 40))
        XCTAssertEqual(f.metadata["n.f32"]?.asDouble, 1.5)
        XCTAssertEqual(f.metadata["n.bool"]?.asBool, true)
        XCTAssertEqual(f.metadata["arr.str"]?.asArray?.compactMap(\.asString), ["a", "bc"])
        XCTAssertEqual(f.tensors.count, 2)
        // GGUF dims are fastest-first: dims [4,2] is a row-major [2,4] tensor.
        let a = try f.tensor(named: "a")
        XCTAssertEqual(a.shape, [2, 4])
        XCTAssertEqual(try a.toFloatArray(), Array(1...8).map(Float.init))
        XCTAssertEqual(try f.tensor(named: "b").toFloatArray(), [9, 10, 11])
        XCTAssertThrowsError(try f.tensor(named: "missing")) { XCTAssertEqual($0 as? GGUFError, .missingTensor("missing")) }
    }

    func testCustomAlignment() throws {
        var b = sample(); b.alignment = 64
        let f = try GGUFFile(data: b.build())
        XCTAssertEqual(f.alignment, 64)
        XCTAssertEqual(f.dataOffset % 64, 0)
        XCTAssertEqual(try f.tensor(named: "b").toFloatArray(), [9, 10, 11])
    }

    func expectError(_ data: Data, _ matches: (GGUFError) -> Bool, _ label: String, file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertThrowsError(try GGUFFile(data: data), label, file: file, line: line) { e in
            guard let g = e as? GGUFError else { return XCTFail("\(label): non-GGUF error \(e)", file: file, line: line) }
            XCTAssertTrue(matches(g), "\(label): got \(g)", file: file, line: line)
        }
    }

    func testRejectsBadHeaders() {
        var d = sample().build()
        expectError(Data(), { $0 == .invalidMagic }, "empty")
        d.replaceSubrange(0..<4, with: Data("GGML".utf8))
        expectError(d, { $0 == .invalidMagic }, "magic")
        for v: UInt32 in [0, 1, 4, 99] {
            var b = sample(); b.version = v
            expectError(b.build(), { if case .unsupportedVersion(v) = $0 { return true } else { return false } }, "version \(v)")
        }
        var be = sample().build()
        be.replaceSubrange(4..<8, with: Data([0, 0, 0, 3]))
        expectError(be, { $0 == .bigEndianNotSupported }, "big endian")
    }

    func testEveryTruncationIsRejectedCleanly() throws {
        let full = sample().build()
        // Every prefix that cuts into the header, directory or any tensor's bytes must be rejected.
        // (Cutting only the alignment padding after the last tensor is legitimately still a valid file.)
        let parsed = try GGUFFile(data: full)
        let required = parsed.tensors.map { $0.fileOffset + $0.byteCount }.max()!
        for n in 0..<required {
            XCTAssertThrowsError(try GGUFFile(data: full.prefix(n)), "prefix \(n) of \(full.count) parsed")
        }
        XCTAssertNoThrow(try GGUFFile(data: full))
    }

    func testRejectsMalformedMetadataAndTensors() {
        // Huge declared counts / lengths must fail on size checks before any allocation.
        var d = sample().build()
        d.replaceSubrange(8..<16, with: GGUFBuilder.le(UInt64.max))
        expectError(d, { if case .limitExceeded = $0 { return true } else { return false } }, "tensor count max")
        d = sample().build(); d.replaceSubrange(16..<24, with: GGUFBuilder.le(UInt64(1 << 40)))
        expectError(d, { if case .limitExceeded = $0 { return true } else { return false } }, "kv count huge")
        d = sample().build(); d.replaceSubrange(16..<24, with: GGUFBuilder.le(UInt64(60_000)))
        expectError(d, { if case .truncated = $0 { return true } else { return false } }, "kv count beyond file")

        var b = GGUFBuilder(); b.metadata = [("k", .array(elementType: 4, [.u32(1)]))]
        var bytes = b.build()
        // Patch the array length (after magic, version, counts, key, type, element type) to something enormous.
        let keyEnd = 24 + 8 + 1 + 4 + 4
        bytes.replaceSubrange(keyEnd..<(keyEnd + 8), with: GGUFBuilder.le(UInt64(1 << 23)))
        expectError(bytes, { if case .truncated = $0 { return true } else { return false } }, "array length beyond file")
        bytes.replaceSubrange(keyEnd..<(keyEnd + 8), with: GGUFBuilder.le(UInt64.max))
        expectError(bytes, { if case .limitExceeded = $0 { return true } else { return false } }, "array length u64 max")

        b = GGUFBuilder(); b.metadata = [("k", .str("x"))]
        bytes = b.build()
        bytes.replaceSubrange(37..<45, with: GGUFBuilder.le(UInt64.max)) // string length of value
        expectError(bytes, { if case .limitExceeded = $0 { return true } else { return false } }, "string length u64 max")

        b = GGUFBuilder(); b.metadata = [("k", .bool(true)), ("k", .bool(false))]
        expectError(b.build(), { if case .malformed = $0 { return true } else { return false } }, "duplicate metadata key")
        b = GGUFBuilder(); b.metadata = [("general.alignment", .u32(48))]
        expectError(b.build(), { if case .malformed = $0 { return true } else { return false } }, "non power-of-two alignment")
    }

    func testRejectsInvalidTensorRecords() {
        func one(_ mutate: (inout GGUFBuilder.Tensor) -> Void) -> Data {
            var b = GGUFBuilder(); var t = GGUFBuilder.Tensor.f32("w", dims: [8], values: Array(repeating: 1, count: 8)); mutate(&t); b.tensors = [t]; return b.build()
        }
        let malformed: (GGUFError) -> Bool = { if case .malformed = $0 { return true } else { return false } }
        expectError(one { $0.dims = [0] }, malformed, "zero dimension")
        expectError(one { $0.dims = [] }, malformed, "no dimensions")
        expectError(one { $0.dims = [1, 1, 1, 1, 8] }, malformed, "five dimensions")
        expectError(one { $0.dims = [1 << 62, 1 << 62] }, malformed, "dimension overflow")
        expectError(one { $0.dims = [UInt64.max] }, malformed, "dimension u64 max")
        expectError(one { $0.type = 99 }, { if case .unsupportedTensorType(99, "w") = $0 { return true } else { return false } }, "unknown type")
        expectError(one { $0.type = 8 }, malformed, "q8_0 row not multiple of 32")
        expectError(one { $0.offsetOverride = 1 << 40 }, { if case .tensorOutOfRange = $0 { return true } else { return false } }, "offset beyond file")
        expectError(one { $0.offsetOverride = UInt64.max - 31 }, { if case .tensorOutOfRange = $0 { return true } else { return false } }, "offset u64 max")
        expectError(one { $0.offsetOverride = 5 }, malformed, "unaligned offset")
        expectError(one { $0.dims = [1 << 20] }, { if case .tensorOutOfRange = $0 { return true } else { return false } }, "tensor larger than file")

        var b = GGUFBuilder()
        b.tensors = [.f32("w", dims: [4], values: [1, 2, 3, 4]), .f32("w", dims: [4], values: [1, 2, 3, 4])]
        expectError(b.build(), { $0 == .duplicateTensor("w") }, "duplicate tensor")
        var o = GGUFBuilder()
        var second = GGUFBuilder.Tensor.f32("y", dims: [16], values: Array(repeating: 0, count: 16)); second.offsetOverride = 32
        var first = GGUFBuilder.Tensor.f32("x", dims: [16], values: Array(repeating: 0, count: 16)); first.offsetOverride = 0
        o.tensors = [first, second]   // x occupies bytes 0..64, y starts at 32
        expectError(o.build(), { if case .tensorsOverlap = $0 { return true } else { return false } }, "overlapping tensors")
    }

    func testMissingFileReportsCleanError() {
        XCTAssertThrowsError(try GGUFFile(url: URL(fileURLWithPath: "/nonexistent/model.gguf"))) { e in
            guard case GGUFError.cannotOpen = e else { return XCTFail("\(e)") }
        }
    }

    /// Seeded mutation fuzzing: random byte flips, insertions, truncations. The parser must always either
    /// throw `GGUFError` or return a file whose every tensor lies inside the mapping — never trap.
    func testMutationFuzzing() throws {
        let base = sample().build()
        var rng = SplitMix64(seed: 0xA17ACA)
        var accepted = 0, rejected = 0
        for _ in 0..<20_000 {
            var d = base
            for _ in 0..<(1 + Int(rng.next() % 4)) {
                switch rng.next() % 4 {
                case 0: d[Int(rng.next() % UInt64(d.count))] = UInt8(truncatingIfNeeded: rng.next())
                case 1: d[Int(rng.next() % UInt64(d.count))] ^= UInt8(1 << (rng.next() % 8))
                case 2: d = d.prefix(Int(rng.next() % UInt64(d.count + 1)))
                default: let at = Int(rng.next() % UInt64(d.count + 1)); d.insert(contentsOf: [UInt8(truncatingIfNeeded: rng.next())], at: at)
                }
                if d.isEmpty { d = Data([0]) }
            }
            do {
                let f = try GGUFFile(data: d)
                accepted += 1
                for t in f.tensors {
                    XCTAssertLessThanOrEqual(t.fileOffset + t.byteCount, f.fileSize)
                    _ = f.rawBytes(of: t).reduce(0) { $0 &+ Int($1) }
                    if t.type == .f32 { _ = try f.tensor(named: t.name) }
                }
            } catch is GGUFError { rejected += 1 }
        }
        print("[fuzz] 20000 mutated files: \(accepted) accepted (all tensor ranges verified in-bounds), \(rejected) rejected with GGUFError")
    }
}

struct SplitMix64 {
    var state: UInt64
    init(seed: UInt64) { state = seed }
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}
