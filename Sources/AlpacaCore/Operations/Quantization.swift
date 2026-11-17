// Copyright (c) 2026 Torin Etheridge. MIT License (see LICENSE).
// Author: Torin Etheridge · 2026-10-09

import Foundation

// GGUF block formats. See DType for the byte layouts.

/// Dequantises `elements` values (a whole number of blocks) starting at `source` into float32.
public func dequantize(_ dtype: DType, source: UnsafeRawPointer, elements: Int, into dst: UnsafeMutablePointer<Float>) {
    let blocks = elements / quantBlockElements
    switch dtype {
    case .q4_1, .q5_0, .q5_1, .q2_K, .q3_K, .q4_K, .q5_K, .q6_K:
        let n = dtype.blockElements, bytes = dtype.blockBytes
        for b in 0..<(elements / n) { dequantizeBlock(dtype, source + b * bytes, into: dst + b * n) }
    case .bfloat16:
        let h = source.assumingMemoryBound(to: UInt16.self)
        for i in 0..<elements { dst[i] = Float(bitPattern: UInt32(h[i]) << 16) }
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


// MARK: - Reference dequantisation of the additional GGUF formats (ports of ggml's dequantize_row_* — the layouts are documented in Docs/FORMATS.md)

@inline(__always) private func le16(_ p: UnsafeRawPointer, _ o: Int) -> Float { Float((p + o).loadUnaligned(as: Float16.self)) }
@inline(__always) private func le32(_ p: UnsafeRawPointer, _ o: Int) -> UInt32 { (p + o).loadUnaligned(as: UInt32.self) }

/// Scale and min of 6-bit sub-block `j` (0..<8) in the 12-byte packed array used by Q4_K / Q5_K (ggml `get_scale_min_k4`).
@inline(__always) func kQuantScaleMin(_ j: Int, _ q: UnsafePointer<UInt8>) -> (Float, Float) {
    if j < 4 { return (Float(q[j] & 63), Float(q[j + 4] & 63)) }
    return (Float((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4)), Float((q[j + 4] >> 4) | ((q[j] >> 6) << 4)))
}

/// Dequantises one block of `dtype` (one of the formats above) into `blockElements` floats.
func dequantizeBlock(_ dtype: DType, _ p: UnsafeRawPointer, into y: UnsafeMutablePointer<Float>) {
    let u8 = p.assumingMemoryBound(to: UInt8.self)
    switch dtype {
    case .q4_1:
        let d = le16(p, 0), m = le16(p, 2)
        for j in 0..<16 { y[j] = d * Float(u8[4 + j] & 0x0F) + m; y[j + 16] = d * Float(u8[4 + j] >> 4) + m }
    case .q5_0:
        let d = le16(p, 0), qh = le32(p, 2)
        for j in 0..<16 {
            let x0 = Int(u8[6 + j] & 0x0F) | Int(((qh >> UInt32(j)) & 1) << 4)
            let x1 = Int(u8[6 + j] >> 4) | Int(((qh >> UInt32(j + 16)) & 1) << 4)
            y[j] = d * Float(x0 - 16); y[j + 16] = d * Float(x1 - 16)
        }
    case .q5_1:
        let d = le16(p, 0), m = le16(p, 2), qh = le32(p, 4)
        for j in 0..<16 {
            let x0 = Int(u8[8 + j] & 0x0F) | Int(((qh >> UInt32(j)) & 1) << 4)
            let x1 = Int(u8[8 + j] >> 4) | Int(((qh >> UInt32(j + 16)) & 1) << 4)
            y[j] = d * Float(x0) + m; y[j + 16] = d * Float(x1) + m
        }
    case .q2_K:   // scales[16] qs[64] d dmin
        let d = le16(p, 80), dmin = le16(p, 82)
        for e in 0..<256 {
            let half = e / 128, r = e % 128, j = r / 32, s = r % 32
            let sc = u8[half * 8 + j * 2 + (s >= 16 ? 1 : 0)]
            let q = Int((u8[16 + half * 32 + s] >> UInt8(2 * j)) & 3)
            y[e] = d * Float(sc & 0xF) * Float(q) - dmin * Float(sc >> 4)
        }
    case .q3_K:   // hmask[32] qs[64] scales[12] d
        let d = le16(p, 108)
        var sc = [Int](repeating: 0, count: 16)
        for i in 0..<16 {
            let low = i < 8 ? Int(u8[96 + i] & 0xF) : Int(u8[96 + i - 8] >> 4)
            let high = Int((u8[96 + 8 + (i % 4)] >> UInt8(2 * (i / 4))) & 3)
            sc[i] = (low | (high << 4)) - 32
        }
        for e in 0..<256 {
            let half = e / 128, r = e % 128, j = r / 32, s = r % 32
            let q = Int((u8[32 + half * 32 + s] >> UInt8(2 * j)) & 3)
            let hbit = (u8[s] >> UInt8(half * 4 + j)) & 1
            y[e] = d * Float(sc[half * 8 + j * 2 + (s >= 16 ? 1 : 0)]) * Float(q - (hbit != 0 ? 0 : 4))
        }
    case .q4_K:   // d dmin scales[12] qs[128]
        let d = le16(p, 0), dmin = le16(p, 2)
        let scales = u8 + 4, qs = u8 + 16
        for c in 0..<4 {
            let (sc1, m1) = kQuantScaleMin(2 * c, scales), (sc2, m2) = kQuantScaleMin(2 * c + 1, scales)
            for l in 0..<32 {
                y[c * 64 + l] = d * sc1 * Float(qs[c * 32 + l] & 0xF) - dmin * m1
                y[c * 64 + 32 + l] = d * sc2 * Float(qs[c * 32 + l] >> 4) - dmin * m2
            }
        }
    case .q5_K:   // d dmin scales[12] qh[32] qs[128]
        let d = le16(p, 0), dmin = le16(p, 2)
        let scales = u8 + 4, qh = u8 + 16, qs = u8 + 48
        for c in 0..<4 {
            let (sc1, m1) = kQuantScaleMin(2 * c, scales), (sc2, m2) = kQuantScaleMin(2 * c + 1, scales)
            for l in 0..<32 {
                let h1 = (qh[l] >> UInt8(2 * c)) & 1, h2 = (qh[l] >> UInt8(2 * c + 1)) & 1
                y[c * 64 + l] = d * sc1 * Float(Int(qs[c * 32 + l] & 0xF) + Int(h1) * 16) - dmin * m1
                y[c * 64 + 32 + l] = d * sc2 * Float(Int(qs[c * 32 + l] >> 4) + Int(h2) * 16) - dmin * m2
            }
        }
    case .q6_K:   // ql[128] qh[64] scales[16] (int8) d
        let d = le16(p, 208)
        let sc = (p + 192).assumingMemoryBound(to: Int8.self)
        for e in 0..<256 {
            let half = e / 128, r = e % 128, g = r / 32, l = r % 32
            let ql = u8[half * 64 + l + (g & 1) * 32], qh = u8[128 + half * 32 + l]
            let low = Int(g >= 2 ? ql >> 4 : ql & 0xF), high = Int((qh >> UInt8(2 * g)) & 3)
            y[e] = d * Float(sc[half * 8 + l / 16 + 2 * g]) * Float((low | (high << 4)) - 32)
        }
    default: fatalError("dequantizeBlock: \(dtype) is not a block format handled here")
    }
}
