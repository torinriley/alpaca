// Copyright (c) 2026 Torin Etheridge. MIT License (see LICENSE).
// Author: Torin Etheridge · 2026-10-09

/// Every way a GGUF file can be rejected. Model files are untrusted input: parsing never traps.
public enum GGUFError: Error, Equatable, CustomStringConvertible {
    case cannotOpen(path: String, reason: String)
    case invalidMagic
    case unsupportedVersion(UInt32)
    case bigEndianNotSupported
    case truncated(what: String, offset: Int)
    case malformed(String)
    case limitExceeded(String)
    case unsupportedTensorType(id: UInt32, tensor: String)
    case duplicateTensor(String)
    case missingMetadata(String)
    case missingTensor(String)
    case unsupportedModel(String)
    case tensorOutOfRange(tensor: String)
    case tensorsOverlap(String, String)

    public var description: String {
        switch self {
        case .cannotOpen(let p, let r): return "cannot open \(p): \(r)"
        case .invalidMagic: return "not a GGUF file (bad magic)"
        case .unsupportedVersion(let v): return "unsupported GGUF version \(v) (supported: 2, 3)"
        case .bigEndianNotSupported: return "big-endian GGUF files are not supported"
        case .truncated(let w, let o): return "file truncated while reading \(w) at offset \(o)"
        case .malformed(let m): return "malformed GGUF: \(m)"
        case .limitExceeded(let m): return "GGUF limit exceeded: \(m)"
        case .unsupportedTensorType(let id, let t): return "tensor '\(t)' uses unsupported GGML type id \(id)"
        case .duplicateTensor(let n): return "duplicate tensor '\(n)'"
        case .missingMetadata(let k): return "required metadata key '\(k)' is missing or has the wrong type"
        case .missingTensor(let n): return "required tensor '\(n)' is missing"
        case .unsupportedModel(let m): return "unsupported model: \(m)"
        case .tensorOutOfRange(let t): return "tensor '\(t)' lies outside the file"
        case .tensorsOverlap(let a, let b): return "tensors '\(a)' and '\(b)' overlap"
        }
    }
}
