// Copyright (c) 2026 Torin Etheridge. MIT License (see LICENSE).
// Author: Torin Etheridge · 2026-10-09

import Foundation
import AlpacaCore

/// A memory-mapped, validated GGUF container.
///
/// Parsing validates the header, metadata and tensor directory without trusting any length field:
/// every count is bounded by the bytes actually remaining, every size is overflow-checked, and every
/// tensor range is verified to lie inside the file, be aligned, and not overlap another tensor.
public final class GGUFFile: @unchecked Sendable {
    public struct Limits: Sendable {
        public var maxMetadataEntries = 1 << 16
        public var maxTensors = 1 << 16
        public var maxStringLength = 1 << 24          // 16 MiB (chat templates are the largest legitimate strings)
        public var maxArrayLength = 1 << 24
        public var maxArrayDepth = 2
        public var maxTensorDims = 4
        public init() {}
    }

    public let version: UInt32
    public let metadata: [String: GGUFValue]
    public let tensors: [GGUFTensorInfo]
    public let alignment: Int
    public let dataOffset: Int
    public let fileSize: Int
    private let tensorIndex: [String: Int]
    private let mapping: MappedRegion

    /// Maps and validates the file at `url`.
    public convenience init(url: URL, limits: Limits = Limits()) throws {
        let region = try MappedRegion(url: url)
        try self.init(region: region, limits: limits)
    }

    /// Validates an in-memory copy (used by tests and fuzzing).
    public convenience init(data: Data, limits: Limits = Limits()) throws {
        try self.init(region: MappedRegion(copying: data), limits: limits)
    }

    init(region: MappedRegion, limits: Limits) throws {
        var r = ByteReader(base: region.pointer, count: region.count)
        guard region.count >= 4, try r.read(UInt32.self, "magic") == 0x4655_4747 else { throw GGUFError.invalidMagic }  // "GGUF"
        let version = try r.read(UInt32.self, "version")
        if version & 0xFFFF == 0 && version != 0 { throw GGUFError.bigEndianNotSupported }
        guard version == 2 || version == 3 else { throw GGUFError.unsupportedVersion(version) }
        let tensorCount = try r.read(UInt64.self, "tensor count")
        let kvCount = try r.read(UInt64.self, "metadata count")
        guard kvCount <= UInt64(limits.maxMetadataEntries) else { throw GGUFError.limitExceeded("metadata entries \(kvCount)") }
        guard tensorCount <= UInt64(limits.maxTensors) else { throw GGUFError.limitExceeded("tensor count \(tensorCount)") }
        // Each metadata entry needs >= 13 bytes and each tensor record >= 24: reject counts the file cannot hold.
        guard Int(kvCount) <= r.remaining / 13, Int(tensorCount) <= r.remaining / 24 else {
            throw GGUFError.truncated(what: "directory", offset: r.position)
        }

        var metadata: [String: GGUFValue] = [:]
        for _ in 0..<Int(kvCount) {
            let key = try r.readString("metadata key", maxLength: 1 << 12)
            let typeID = try r.read(UInt32.self, "value type for \(key)")
            guard metadata[key] == nil else { throw GGUFError.malformed("duplicate metadata key '\(key)'") }
            metadata[key] = try GGUFFile.readValue(&r, typeID: typeID, key: key, limits: limits, depth: 0)
        }

        var alignment = 32
        if let a = metadata["general.alignment"] {
            guard let v = a.asInt, v > 0, v & (v - 1) == 0, v <= 1 << 20 else { throw GGUFError.malformed("invalid general.alignment") }
            alignment = v
        }

        // Tensor directory.
        struct Raw { let name: String; let dims: [Int]; let type: GGMLType; let relOffset: UInt64; let bytes: Int }
        var raws: [Raw] = []
        var seen = Set<String>()
        for _ in 0..<Int(tensorCount) {
            let name = try r.readString("tensor name", maxLength: 1 << 12)
            guard seen.insert(name).inserted else { throw GGUFError.duplicateTensor(name) }
            let nDims = try r.read(UInt32.self, "dimension count of \(name)")
            guard nDims >= 1, nDims <= UInt32(limits.maxTensorDims) else { throw GGUFError.malformed("tensor '\(name)' has \(nDims) dimensions") }
            var dims: [Int] = []
            var elements = 1
            for _ in 0..<Int(nDims) {
                let d = try r.read(UInt64.self, "dimension of \(name)")
                guard d > 0, d <= UInt64(Int32.max) else { throw GGUFError.malformed("tensor '\(name)' has invalid dimension \(d)") }
                let (p, o) = elements.multipliedReportingOverflow(by: Int(d))
                if o { throw GGUFError.malformed("tensor '\(name)' element count overflows") }
                elements = p
                dims.append(Int(d))
            }
            let typeID = try r.read(UInt32.self, "type of \(name)")
            guard let type = GGMLType(rawValue: typeID) else { throw GGUFError.unsupportedTensorType(id: typeID, tensor: name) }
            let layout = type.blockLayout
            guard dims[0] % layout.elements == 0 else {
                throw GGUFError.malformed("tensor '\(name)': row length \(dims[0]) is not a multiple of block size \(layout.elements)")
            }
            let (bytes, o) = (elements / layout.elements).multipliedReportingOverflow(by: layout.bytes)
            if o { throw GGUFError.malformed("tensor '\(name)' byte size overflows") }
            let offset = try r.read(UInt64.self, "offset of \(name)")
            raws.append(Raw(name: name, dims: dims, type: type, relOffset: offset, bytes: bytes))
        }

        let (aligned, ovf) = (r.position + alignment - 1).addingReportingOverflow(0)
        guard !ovf else { throw GGUFError.malformed("directory size overflows") }
        let dataOffset = aligned / alignment * alignment
        guard dataOffset <= region.count else { throw GGUFError.truncated(what: "tensor data", offset: r.position) }

        var infos: [GGUFTensorInfo] = []
        for raw in raws {
            guard raw.relOffset % UInt64(alignment) == 0 else { throw GGUFError.malformed("tensor '\(raw.name)' offset \(raw.relOffset) is not \(alignment)-byte aligned") }
            guard raw.relOffset <= UInt64(region.count - dataOffset) else { throw GGUFError.tensorOutOfRange(tensor: raw.name) }
            let start = dataOffset + Int(raw.relOffset)
            let (end, o) = start.addingReportingOverflow(raw.bytes)
            guard !o, end <= region.count else { throw GGUFError.tensorOutOfRange(tensor: raw.name) }
            infos.append(GGUFTensorInfo(name: raw.name, dims: raw.dims, type: raw.type, fileOffset: start, byteCount: raw.bytes))
        }
        let sorted = infos.sorted { $0.fileOffset < $1.fileOffset }
        for (a, b) in zip(sorted, sorted.dropFirst()) where a.fileOffset + a.byteCount > b.fileOffset {
            throw GGUFError.tensorsOverlap(a.name, b.name)
        }

        self.version = version; self.metadata = metadata; self.tensors = infos
        self.alignment = alignment; self.dataOffset = dataOffset; self.fileSize = region.count
        self.tensorIndex = Dictionary(uniqueKeysWithValues: infos.enumerated().map { ($1.name, $0) })
        self.mapping = region
    }

    private static func readValue(_ r: inout ByteReader, typeID: UInt32, key: String, limits: Limits, depth: Int) throws -> GGUFValue {
        switch typeID {
        case 0: return .uint(UInt64(try r.read(UInt8.self, key)))
        case 1: return .int(Int64(try r.read(Int8.self, key)))
        case 2: return .uint(UInt64(try r.read(UInt16.self, key)))
        case 3: return .int(Int64(try r.read(Int16.self, key)))
        case 4: return .uint(UInt64(try r.read(UInt32.self, key)))
        case 5: return .int(Int64(try r.read(Int32.self, key)))
        case 6: return .float(Double(try r.readFloat32(key)))
        case 7:
            let b = try r.read(UInt8.self, key)
            guard b <= 1 else { throw GGUFError.malformed("bool '\(key)' has value \(b)") }
            return .bool(b == 1)
        case 8: return .string(try r.readString(key, maxLength: limits.maxStringLength))
        case 9:
            guard depth < limits.maxArrayDepth else { throw GGUFError.limitExceeded("array nesting in '\(key)'") }
            let elemType = try r.read(UInt32.self, "array type of \(key)")
            let n = try r.read(UInt64.self, "array length of \(key)")
            guard n <= UInt64(limits.maxArrayLength) else { throw GGUFError.limitExceeded("array '\(key)' length \(n)") }
            // Every element occupies at least one byte (strings: 8), so the file bounds the real count.
            guard Int(n) <= r.remaining else { throw GGUFError.truncated(what: "array '\(key)'", offset: r.position) }
            var items: [GGUFValue] = []
            items.reserveCapacity(Int(n))
            for _ in 0..<Int(n) { items.append(try readValue(&r, typeID: elemType, key: key, limits: limits, depth: depth + 1)) }
            return .array(items)
        case 10: return .uint(try r.read(UInt64.self, key))
        case 11: return .int(try r.read(Int64.self, key))
        case 12: return .float(try r.readFloat64(key))
        default: throw GGUFError.malformed("metadata '\(key)' has unknown value type \(typeID)")
        }
    }

    // MARK: Access

    public func info(for name: String) -> GGUFTensorInfo? { tensorIndex[name].map { tensors[$0] } }

    /// A tensor over the mapped file bytes in their on-disk dtype — no copy. Only natively supported types qualify.
    public func tensor(named name: String) throws -> Tensor {
        guard let info = info(for: name) else { throw GGUFError.missingTensor(name) }
        guard let dtype = info.type.nativeDType else { throw GGUFError.unsupportedTensorType(id: info.type.rawValue, tensor: name) }
        let storage = TensorStorage(borrowing: mapping.pointer.advanced(by: info.fileOffset), byteCount: info.byteCount, keepAlive: mapping)
        do { return try Tensor(storage: storage, dtype: dtype, shape: info.shape) } catch { throw GGUFError.malformed("tensor '\(name)': \(error)") }
    }

    /// Raw bytes of a tensor (any type), valid while this file object is alive.
    public func rawBytes(of info: GGUFTensorInfo) -> UnsafeRawBufferPointer {
        UnsafeRawBufferPointer(start: UnsafeRawPointer(mapping.pointer) + info.fileOffset, count: info.byteCount)
    }

    /// The read-only mapping backing every mapped tensor: lets a GPU backend wrap the whole file as one
    /// no-copy buffer instead of copying each tensor.
    public var mappedMemory: (base: UnsafeRawPointer, length: Int) { (UnsafeRawPointer(mapping.pointer), mapping.count) }

    /// An object that keeps the mapping alive (retain it for as long as raw pointers into the file are in use).
    public var mappingOwner: AnyObject { mapping }

    public func string(_ key: String) -> String? { metadata[key]?.asString }
    public func int(_ key: String) -> Int? { metadata[key]?.asInt }
    public func double(_ key: String) -> Double? { metadata[key]?.asDouble }
}

/// Read-only memory mapping (or private copy) of a file. Unmaps on deinit.
final class MappedRegion: @unchecked Sendable {
    let pointer: UnsafeMutableRawPointer
    let count: Int
    private let mapped: Bool

    init(url: URL) throws {
        let path = url.path
        let fd = open(path, O_RDONLY)
        guard fd >= 0 else { throw GGUFError.cannotOpen(path: path, reason: String(cString: strerror(errno))) }
        defer { close(fd) }
        var st = stat()
        guard fstat(fd, &st) == 0 else { throw GGUFError.cannotOpen(path: path, reason: String(cString: strerror(errno))) }
        guard st.st_size > 0 else { throw GGUFError.truncated(what: "header", offset: 0) }
        guard let size = Int(exactly: st.st_size) else { throw GGUFError.limitExceeded("file too large") }
        let p = mmap(nil, size, PROT_READ, MAP_PRIVATE, fd, 0)
        guard let p, p != MAP_FAILED else { throw GGUFError.cannotOpen(path: path, reason: "mmap failed: \(String(cString: strerror(errno)))") }
        pointer = p; count = size; mapped = true
    }

    init(copying data: Data) throws {
        let size = max(data.count, 1)
        guard let p = malloc(size) else { throw GGUFError.limitExceeded("allocation of \(size) bytes failed") }
        data.withUnsafeBytes { if let b = $0.baseAddress { memcpy(p, b, data.count) } }
        pointer = p; count = data.count; mapped = false
    }

    deinit { if mapped { munmap(pointer, count) } else { free(pointer) } }
}
