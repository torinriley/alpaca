// Copyright (c) 2026 Torin Etheridge. MIT License (see LICENSE).
// Author: Torin Etheridge · 2026-10-09

/// Errors raised by tensor construction and CPU operations. Every operation validates shape,
/// dtype and layout up front and throws instead of reading out of bounds.
public enum TensorError: Error, Equatable, CustomStringConvertible {
    case invalidShape(String)
    case dtypeMismatch(String)
    case shapeMismatch(String)
    case unsupportedLayout(String)
    case outOfBounds(String)
    case allocationFailed(bytes: Int)

    public var description: String {
        switch self {
        case .invalidShape(let m): return "invalid shape: \(m)"
        case .dtypeMismatch(let m): return "dtype mismatch: \(m)"
        case .shapeMismatch(let m): return "shape mismatch: \(m)"
        case .unsupportedLayout(let m): return "unsupported layout: \(m)"
        case .outOfBounds(let m): return "out of bounds: \(m)"
        case .allocationFailed(let b): return "failed to allocate \(b) bytes"
        }
    }
}
