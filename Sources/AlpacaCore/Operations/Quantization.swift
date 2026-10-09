// Copyright (c) 2026 Torin Etheridge. MIT License (see LICENSE).
// Author: Torin Etheridge · 2026-10-09

import Foundation

// GGUF block formats. See DType for the byte layouts.

/// Dequantises `elements` values (a whole number of blocks) starting at `source` into float32.
public func dequantize(_ dtype: DType, source: UnsafeRawPointer, elements: Int, into dst: UnsafeMutablePointer<Float>) {
    let blocks = elements / quantBlockElements
    switch dtype {
    case .q8_0:
        for b in 0..<blocks {
            let p = source + b * 34
            let d = Float(p.loadUnaligned(as: Float16.self))
            let q = (p + 2).assumingMemoryBound(to: Int8.self)
            for j in 0..<32 { dst[b * 32 + j] = d * Float(q[j]) }
        }
    case .q4_0:
        for b in 0..<blocks {
            let p = source + b * 18
            let d = Float(p.loadUnaligned(as: Float16.self))
            let q = (p + 2).assumingMemoryBound(to: UInt8.self)
            for j in 0..<16 {
                dst[b * 32 + j] = d * Float(Int(q[j] & 0x0F) - 8)
                dst[b * 32 + j + 16] = d * Float(Int(q[j] >> 4) - 8)
            }
        }
    case .float32:
        memcpy(dst, source, elements * 4)
    case .float16:
        let h = source.assumingMemoryBound(to: Float16.self)
        for i in 0..<elements { dst[i] = Float(h[i]) }
    }
}

/// Reference quantiser following ggml's `quantize_row_q8_0_ref` / `quantize_row_q4_0_ref`.
/// Used to build test weights and to quantise tensors for CPU-only experiments; real models ship pre-quantised.
public func quantize(_ values: [Float], to dtype: DType) throws -> Tensor {
    guard dtype == .q8_0 || dtype == .q4_0 else { throw TensorError.dtypeMismatch("quantize target must be q8_0 or q4_0") }
    guard values.count % quantBlockElements == 0 else {
        throw TensorError.invalidShape("\(values.count) values is not a multiple of \(quantBlockElements)")
    }
    let blocks = values.count / quantBlockElements
    let storage = try TensorStorage(byteCount: blocks * dtype.blockBytes)
    for b in 0..<blocks {
        let x = values[(b * 32)..<(b * 32 + 32)]
        let p = storage.pointer + b * dtype.blockBytes
        if dtype == .q8_0 {
            let amax = x.reduce(0) { max($0, abs($1)) }
            let d = amax / 127
            let id: Float = d != 0 ? 1 / d : 0
            p.storeBytes(of: Float16(d), as: Float16.self)
            let q = (p + 2).assumingMemoryBound(to: Int8.self)
            for (j, v) in x.enumerated() { q[j] = Int8((v * id).rounded(.toNearestOrAwayFromZero)) }
        } else {
            // The element with the largest magnitude keeps its sign: d = max / -8 maps it to nibble 0.
            var maxv: Float = 0, amax: Float = 0
            for v in x where abs(v) > amax { amax = abs(v); maxv = v }
            let d = maxv / -8
            let id: Float = d != 0 ? 1 / d : 0
            p.storeBytes(of: Float16(d), as: Float16.self)
            let q = (p + 2).assumingMemoryBound(to: UInt8.self)
            let xs = Array(x)
            for j in 0..<16 {
                let lo = min(15, Int(xs[j] * id + 8.5))
                let hi = min(15, Int(xs[j + 16] * id + 8.5))
                q[j] = UInt8(lo) | (UInt8(hi) << 4)
            }
        }
    }
    return try Tensor(storage: storage, dtype: dtype, shape: [values.count])
}
