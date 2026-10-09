// Copyright (c) 2026 Torin Etheridge. MIT License (see LICENSE).
// Author: Torin Etheridge · 2026-10-09

import Foundation

/// Owns (or borrows) a contiguous block of bytes backing one or more tensors.
///
/// Allocated storage is 64-byte aligned (cache line / Metal buffer friendly) and zero-initialised.
/// Borrowed storage (e.g. a memory-mapped model file) keeps its `owner` alive and is never freed here.
/// Tensors hold a strong reference to their storage, so views can never outlive the bytes they read.
public final class TensorStorage: @unchecked Sendable {
    public let pointer: UnsafeMutableRawPointer
    public let byteCount: Int
    private let ownsMemory: Bool
    private let keepAlive: AnyObject?

    public static let alignment = 64

    /// Allocates zeroed storage. Throws instead of trapping when the allocation cannot be satisfied.
    public init(byteCount: Int) throws {
        guard byteCount >= 0 else { throw TensorError.allocationFailed(bytes: byteCount) }
        let size = max(byteCount, 1)
        var raw: UnsafeMutableRawPointer?
        guard posix_memalign(&raw, Self.alignment, size) == 0, let raw else {
            throw TensorError.allocationFailed(bytes: byteCount)
        }
        memset(raw, 0, size)
        self.pointer = raw
        self.byteCount = byteCount
        self.ownsMemory = true
        self.keepAlive = nil
    }

    /// Wraps externally owned memory. `keepAlive` (e.g. a mapped-file object) is retained for the lifetime of the storage.
    public init(borrowing pointer: UnsafeMutableRawPointer, byteCount: Int, keepAlive: AnyObject?) {
        self.pointer = pointer
        self.byteCount = byteCount
        self.ownsMemory = false
        self.keepAlive = keepAlive
    }

    deinit {
        if ownsMemory { free(pointer) }
    }
}
