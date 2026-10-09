// Copyright (c) 2026 Torin Etheridge. MIT License (see LICENSE).
// Author: Torin Etheridge · 2026-10-09

/// Contiguous per-layer key/value cache for one generation sequence.
///
/// Layout per layer: [capacity, kvHeadCount, headDim], float32, row = absolute token position.
/// Rows are written in place; attention reads rows `0..<length` directly (no copies). GQA needs no special
/// storage: query head h reads KV head h / (headCount / kvHeadCount).
/// A `KVCache` is a mutable, single-owner object: each generation session owns its own cache.
public final class KVCache: @unchecked Sendable {
    public let layerCount: Int
    public let capacity: Int
    public let kvHeadCount: Int
    public let headDim: Int
    public private(set) var length: Int = 0
    public let keys: [Tensor]
    public let values: [Tensor]
    /// When true, written rows are rounded through Float16 (storage stays f32). This reproduces the numerical
    /// effect of the Metal backend's f16 cache so the two backends can be compared without precision noise.
    public let roundsToFloat16: Bool

    public init(layerCount: Int, capacity: Int, kvHeadCount: Int, headDim: Int, roundsToFloat16: Bool = false) throws {
        self.roundsToFloat16 = roundsToFloat16
        guard layerCount > 0, capacity > 0, kvHeadCount > 0, headDim > 0 else { throw TensorError.invalidShape("KV cache dimensions must be positive") }
        self.layerCount = layerCount; self.capacity = capacity; self.kvHeadCount = kvHeadCount; self.headDim = headDim
        let shape = [capacity, kvHeadCount, headDim]
        keys = try (0..<layerCount).map { _ in try Tensor(zeros: shape) }
        values = try (0..<layerCount).map { _ in try Tensor(zeros: shape) }
    }

    public convenience init(config: LlamaConfig, capacity: Int? = nil, roundsToFloat16: Bool = false) throws {
        try self.init(layerCount: config.layerCount, capacity: min(capacity ?? config.contextLength, config.contextLength),
                      kvHeadCount: config.kvHeadCount, headDim: config.headDim, roundsToFloat16: roundsToFloat16)
    }

    /// Total bytes held by the cache (keys + values, all layers).
    public var byteCount: Int { 2 * layerCount * capacity * kvHeadCount * headDim * 4 }

    public static func byteCount(config: LlamaConfig, capacity: Int) -> Int {
        2 * config.layerCount * capacity * config.kvWidth * 4
    }

    /// Writes `count` rows of k/v for `layer` starting at absolute `position`. Does not change `length`.
    public func write(layer: Int, position: Int, keys k: Tensor, values v: Tensor) throws {
        guard layer >= 0, layer < layerCount else { throw TensorError.outOfBounds("KV layer \(layer)") }
        guard k.shape == v.shape, k.rank == 3, k.shape[1] == kvHeadCount, k.shape[2] == headDim,
            k.dtype == .float32, v.dtype == .float32, k.isContiguous, v.isContiguous
        else { throw TensorError.shapeMismatch("KV write expects contiguous f32 [n, \(kvHeadCount), \(headDim)], got k \(k.shape) v \(v.shape)") }
        let n = k.shape[0]
        guard position >= 0, position == length, position + n <= capacity else {
            throw TensorError.outOfBounds("KV write of \(n) rows at position \(position) (length \(length), capacity \(capacity))")
        }
        let rowBytes = kvHeadCount * headDim * 4
        memcpy(keys[layer].storage.pointer + position * rowBytes, k.basePointer, n * rowBytes)
        memcpy(values[layer].storage.pointer + position * rowBytes, v.basePointer, n * rowBytes)
        if roundsToFloat16 {
            let count = n * kvHeadCount * headDim, first = position * kvHeadCount * headDim
            for tensor in [keys[layer], values[layer]] {
                let p = tensor.storage.pointer.assumingMemoryBound(to: Float.self) + first
                for i in 0..<count { p[i] = Float(Float16(p[i])) }
            }
        }
    }

    /// Marks rows up to `newLength` valid once every layer has written them.
    public func commit(length newLength: Int) throws {
        guard newLength >= length, newLength <= capacity else { throw TensorError.outOfBounds("KV commit to \(newLength) (length \(length), capacity \(capacity))") }
        length = newLength
    }

    /// Forgets all cached positions. Storage is retained and reused; stale rows are never read because
    /// attention is bounded by `length`.
    public func reset() { length = 0 }
}

import Foundation
