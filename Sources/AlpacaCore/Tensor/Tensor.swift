// Copyright (c) 2026 Torin Etheridge. MIT License (see LICENSE).
// Author: Torin Etheridge · 2026-10-09

import Foundation

/// A strided view over `TensorStorage`.
///
/// - `shape` and `strides` are in elements; `offset` is in elements from the start of storage.
/// - Quantised dtypes (`q8_0`, `q4_0`) must be contiguous with a last dimension divisible by the block size.
/// - Views (`reshaped`, `transposed`, `slice`) share storage; writes through one are visible through the others.
public struct Tensor: @unchecked Sendable {
    public let storage: TensorStorage
    public let dtype: DType
    public let shape: [Int]
    public let strides: [Int]
    public let offset: Int

    public var rank: Int { shape.count }
    public var elementCount: Int { shape.reduce(1, *) }

    /// Row-major strides for `shape`.
    public static func contiguousStrides(for shape: [Int]) -> [Int] {
        var strides = [Int](repeating: 1, count: shape.count)
        var running = 1
        for i in stride(from: shape.count - 1, through: 0, by: -1) {
            strides[i] = running
            running *= max(shape[i], 1)
        }
        return strides
    }

    /// Creates a view and validates that every addressable element lies inside `storage`.
    public init(storage: TensorStorage, dtype: DType, shape: [Int], strides: [Int]? = nil, offset: Int = 0) throws {
        guard shape.allSatisfy({ $0 >= 0 }) else { throw TensorError.invalidShape("negative dimension in \(shape)") }
        let strides = strides ?? Tensor.contiguousStrides(for: shape)
        guard strides.count == shape.count else {
            throw TensorError.invalidShape("strides \(strides) do not match rank of shape \(shape)")
        }
        guard offset >= 0 else { throw TensorError.outOfBounds("negative offset \(offset)") }

        // Highest addressed element index (overflow-checked).
        var maxIndex = offset
        var empty = false
        for (n, s) in zip(shape, strides) {
            if n == 0 { empty = true; break }
            guard s >= 0 else { throw TensorError.unsupportedLayout("negative stride \(s)") }
            let (span, o1) = (n - 1).multipliedReportingOverflow(by: s)
            let (sum, o2) = maxIndex.addingReportingOverflow(span)
            if o1 || o2 { throw TensorError.outOfBounds("tensor extent overflows") }
            maxIndex = sum
        }
        if !empty {
            if dtype.isQuantized {
                guard strides == Tensor.contiguousStrides(for: shape), offset % dtype.blockElements == 0,
                    let last = shape.last, last % dtype.blockElements == 0
                else {
                    throw TensorError.unsupportedLayout(
                        "\(dtype) tensors must be contiguous with last dim divisible by \(dtype.blockElements)")
                }
            }
            guard let needed = dtype.byteCount(elements: ((maxIndex + 1 + dtype.blockElements - 1) / dtype.blockElements) * dtype.blockElements),
                needed <= storage.byteCount
            else {
                throw TensorError.outOfBounds("shape \(shape) strides \(strides) offset \(offset) exceeds \(storage.byteCount) bytes of \(dtype) storage")
            }
        }
        self.storage = storage
        self.dtype = dtype
        self.shape = shape
        self.strides = strides
        self.offset = offset
    }

    /// Allocates a zero-filled contiguous tensor.
    public init(zeros shape: [Int], dtype: DType = .float32) throws {
        let count = try Tensor.checkedElementCount(shape)
        guard let bytes = dtype.byteCount(elements: count) else {
            throw TensorError.invalidShape("\(count) elements is not a whole number of \(dtype) blocks")
        }
        try self.init(storage: TensorStorage(byteCount: bytes), dtype: dtype, shape: shape)
    }

    public init(_ values: [Float], shape: [Int]) throws {
        let count = try Tensor.checkedElementCount(shape)
        guard count == values.count else {
            throw TensorError.shapeMismatch("\(values.count) values for shape \(shape)")
        }
        try self.init(zeros: shape, dtype: .float32)
        values.withUnsafeBytes { src in
            if let base = src.baseAddress, src.count > 0 { memcpy(storage.pointer, base, src.count) }
        }
    }

    static func checkedElementCount(_ shape: [Int]) throws -> Int {
        var total = 1
        for d in shape {
            guard d >= 0 else { throw TensorError.invalidShape("negative dimension in \(shape)") }
            let (p, o) = total.multipliedReportingOverflow(by: d)
            if o { throw TensorError.invalidShape("element count overflows for \(shape)") }
            total = p
        }
        return total
    }

    // MARK: Layout

    public var isContiguous: Bool {
        var expected = 1
        for i in stride(from: rank - 1, through: 0, by: -1) where shape[i] != 1 {
            if strides[i] != expected { return false }
            expected *= shape[i]
        }
        return true
    }

    // MARK: Views

    /// Reinterprets a contiguous tensor with a new shape of equal element count.
    public func reshaped(_ newShape: [Int]) throws -> Tensor {
        guard isContiguous else { throw TensorError.unsupportedLayout("reshape requires a contiguous tensor") }
        guard try Tensor.checkedElementCount(newShape) == elementCount else {
            throw TensorError.shapeMismatch("cannot reshape \(shape) to \(newShape)")
        }
        return try Tensor(storage: storage, dtype: dtype, shape: newShape, offset: offset)
    }

    /// Swaps two dimensions without copying.
    public func transposed(_ a: Int, _ b: Int) throws -> Tensor {
        guard a >= 0, b >= 0, a < rank, b < rank else { throw TensorError.invalidShape("transpose axes \(a),\(b) for rank \(rank)") }
        guard !dtype.isQuantized else { throw TensorError.unsupportedLayout("cannot transpose \(dtype) tensors") }
        var s = shape, st = strides
        s.swapAt(a, b); st.swapAt(a, b)
        return try Tensor(storage: storage, dtype: dtype, shape: s, strides: st, offset: offset)
    }

    /// A view of `range` along `axis`.
    public func slice(axis: Int, _ range: Range<Int>) throws -> Tensor {
        guard axis >= 0, axis < rank else { throw TensorError.invalidShape("axis \(axis) for rank \(rank)") }
        guard range.lowerBound >= 0, range.upperBound <= shape[axis] else {
            throw TensorError.outOfBounds("slice \(range) on axis \(axis) of size \(shape[axis])")
        }
        var s = shape
        s[axis] = range.count
        return try Tensor(storage: storage, dtype: dtype, shape: s, strides: strides, offset: offset + range.lowerBound * strides[axis])
    }

    /// Returns `self` if already contiguous, otherwise a contiguous float copy.
    public func contiguous() throws -> Tensor {
        if isContiguous { return self }
        let out = try Tensor(zeros: shape, dtype: dtype)
        try copyElements(into: out)
        return out
    }

    // MARK: Typed access

    /// Calls `body` with the float32 elements of this tensor starting at `offset`. Requires `.float32`.
    public func withFloat32<R>(_ body: (UnsafeMutablePointer<Float>) throws -> R) throws -> R {
        guard dtype == .float32 else { throw TensorError.dtypeMismatch("expected f32, got \(dtype)") }
        return try body(storage.pointer.assumingMemoryBound(to: Float.self) + offset)
    }

    public func withFloat16<R>(_ body: (UnsafeMutablePointer<Float16>) throws -> R) throws -> R {
        guard dtype == .float16 else { throw TensorError.dtypeMismatch("expected f16, got \(dtype)") }
        return try body(storage.pointer.assumingMemoryBound(to: Float16.self) + offset)
    }

    /// Raw pointer to the first byte of the first element. Quantised tensors are addressed in whole blocks.
    public var basePointer: UnsafeMutableRawPointer {
        storage.pointer + (offset / dtype.blockElements) * dtype.blockBytes
    }

    /// Row-major flattened copy as Float, converting from f16 or dequantising where needed.
    public func toFloatArray() throws -> [Float] {
        let count = elementCount
        var result = [Float](repeating: 0, count: count)
        if count == 0 { return result }
        if dtype.isQuantized {
            let rowLen = shape[rank - 1]
            let rows = count / rowLen
            result.withUnsafeMutableBufferPointer { dst in
                dequantize(dtype, source: basePointer, elements: count, into: dst.baseAddress!)
            }
            _ = rows
            return result
        }
        var index = [Int](repeating: 0, count: rank)
        let f32 = storage.pointer.assumingMemoryBound(to: Float.self)
        let f16 = storage.pointer.assumingMemoryBound(to: Float16.self)
        for i in 0..<count {
            var linear = offset
            for d in 0..<rank { linear += index[d] * strides[d] }
            result[i] = dtype == .float32 ? f32[linear] : Float(f16[linear])
            var d = rank - 1
            while d >= 0 {
                index[d] += 1
                if index[d] < shape[d] { break }
                index[d] = 0; d -= 1
            }
        }
        return result
    }

    /// Element-wise copy between float tensors of identical shape (any strides).
    public func copyElements(into dst: Tensor) throws {
        guard shape == dst.shape else { throw TensorError.shapeMismatch("copy \(shape) -> \(dst.shape)") }
        guard !dtype.isQuantized, !dst.dtype.isQuantized else { throw TensorError.unsupportedLayout("copyElements on quantised tensor") }
        let values = try toFloatArray()
        try dst.assign(values)
    }

    /// Writes row-major `values` into this (possibly strided) float tensor.
    public func assign(_ values: [Float]) throws {
        guard values.count == elementCount else { throw TensorError.shapeMismatch("\(values.count) values for shape \(shape)") }
        guard !dtype.isQuantized else { throw TensorError.unsupportedLayout("assign into \(dtype)") }
        if elementCount == 0 { return }
        var index = [Int](repeating: 0, count: rank)
        let f32 = storage.pointer.assumingMemoryBound(to: Float.self)
        let f16 = storage.pointer.assumingMemoryBound(to: Float16.self)
        for i in 0..<elementCount {
            var linear = offset
            for d in 0..<rank { linear += index[d] * strides[d] }
            if dtype == .float32 { f32[linear] = values[i] } else { f16[linear] = Float16(values[i]) }
            var d = rank - 1
            while d >= 0 {
                index[d] += 1
                if index[d] < shape[d] { break }
                index[d] = 0; d -= 1
            }
        }
    }

    /// Copy converted to another float dtype (contiguous).
    public func converted(to target: DType) throws -> Tensor {
        guard !target.isQuantized else { throw TensorError.unsupportedLayout("use quantize() for \(target)") }
        let out = try Tensor(zeros: shape, dtype: target)
        try out.assign(try toFloatArray())
        return out
    }
}
