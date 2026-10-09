// Copyright (c) 2026 Torin Etheridge. MIT License (see LICENSE).
// Author: Torin Etheridge · 2026-10-09

import XCTest
@testable import AlpacaCore

/// Level 1 — mathematical primitives against numpy/PyTorch/gguf references (Scripts/generate_op_fixtures.py).
///
/// Tolerance rationale: references are float64; alpaca accumulates in Float32 (unit roundoff u = 2^-24 ≈ 6e-8).
/// A length-K dot product has worst-case error ≈ K·u·Σ|aᵢbᵢ|; the tolerances below are that bound rounded
/// up for the operand magnitudes in the fixtures (|x| ≲ 4), not tuned to observed error.
final class OperationTests: XCTestCase {
    var ops: [String: Any] { get throws { try Fixtures.load("ops") } }

    func testAddMul() throws {
        let a = try ops.dict("add"), m = try ops.dict("mul")
        let shape = a.ints("shape")
        let sum = try CPUOps.add(Tensor(a.floats("a"), shape: shape), Tensor(a.floats("b"), shape: shape))
        try assertClose(sum, a.doubles("expected"), atol: 2e-7, rtol: 1e-6, "add")
        let prod = try CPUOps.mul(Tensor(m.floats("a"), shape: shape), Tensor(m.floats("b"), shape: shape))
        try assertClose(prod, m.doubles("expected"), atol: 2e-7, rtol: 1e-6, "mul")
    }

    func testSiLU() throws {
        let c = try ops.dict("silu")
        try assertClose(CPUOps.silu(Tensor(c.floats("x"), shape: c.ints("shape"))), c.doubles("expected"), atol: 1e-6, rtol: 1e-6, "silu")
    }

    func testMatmul() throws {
        let c = try ops.dict("matmul")
        let r = try CPUOps.matmul(Tensor(c.floats("a"), shape: [c.int("m"), c.int("k")]), Tensor(c.floats("b"), shape: [c.int("k"), c.int("n")]))
        try assertClose(r, c.doubles("expected"), atol: 2e-6, rtol: 1e-5, "matmul")
    }

    func testLinearF32AndF16() throws {
        let c = try ops.dict("linear")
        let x = try Tensor(c.floats("x"), shape: [c.int("m"), c.int("k")])
        let w = try Tensor(c.floats("w"), shape: [c.int("n"), c.int("k")])
        try assertClose(CPUOps.linear(x, weight: w), c.doubles("expected"), atol: 5e-6, rtol: 1e-5, "linear f32")
        // f16 weights: reference uses the f16-rounded weights, so only accumulation error remains.
        try assertClose(CPUOps.linear(x, weight: w.converted(to: .float16)), c.doubles("expected_f16"), atol: 5e-6, rtol: 1e-5, "linear f16")
        // Matrix-vector (decode) path: single row.
        let row = try x.slice(axis: 0, 0..<1)
        let y = try CPUOps.linear(row, weight: w)
        try assertClose(y, Array(c.doubles("expected")[0..<c.int("n")]), atol: 5e-6, rtol: 1e-5, "matvec f32")
    }

    func testRMSNorm() throws {
        let c = try ops.dict("rmsnorm")
        let d = c.ints("shape")[1]
        let r = try CPUOps.rmsNorm(Tensor(c.floats("x"), shape: c.ints("shape")), weight: Tensor(c.floats("w"), shape: [d]), eps: Float(c["eps"] as! Double))
        try assertClose(r, c.doubles("expected"), atol: 2e-6, rtol: 2e-6, "rmsnorm")
    }

    func testSoftmax() throws {
        let c = try ops.dict("softmax")
        let r = try CPUOps.softmax(Tensor(c.floats("x"), shape: c.ints("shape")))
        try assertClose(r, c.doubles("expected"), atol: 1e-6, rtol: 1e-5, "softmax")
        let rows = try r.toFloatArray().chunks(of: 11).map { $0.reduce(0, +) }
        for s in rows { XCTAssertEqual(s, 1, accuracy: 1e-6) }
    }

    func testRoPEBothConventions() throws {
        let c = try ops.dict("rope")
        let x = try Tensor(c.floats("x"), shape: c.ints("shape"))
        let theta = Float(c["theta"] as! Double)
        // cos/sin are rounded to Float32 once: error ≤ |x|·u·few ≈ 1e-6 for |x| ≲ 4.
        try assertClose(CPUOps.rope(x, startPosition: c.int("start"), theta: theta, interleaved: true), c.doubles("expected_interleaved"), atol: 2e-6, "rope interleaved")
        try assertClose(CPUOps.rope(x, startPosition: c.int("start"), theta: theta, interleaved: false), c.doubles("expected_split"), atol: 2e-6, "rope split-half")
    }

    func testCausalGQAAttention() throws {
        let c = try ops.dict("attention")
        let (t, h, kv, hd, cap) = (c.int("tokens"), c.int("heads"), c.int("kvHeads"), c.int("headDim"), c.int("capacity"))
        let q = try Tensor(c.floats("q"), shape: [t, h, hd])
        let k = try Tensor(c.floats("k"), shape: [cap, kv, hd]), v = try Tensor(c.floats("v"), shape: [cap, kv, hd])
        let scores = try CPUOps.attentionScores(q: q, keys: k, length: c.int("length"), startPosition: c.int("start"))
        // Causal mask: token i (position start+i) must not see positions > start+i.
        let s = try scores.toFloatArray(), len = c.int("length")
        for hh in 0..<h { for tt in 0..<t { for p in 0..<len {
            let masked = p > c.int("start") + tt
            XCTAssertEqual(s[(hh * t + tt) * len + p] == -.infinity, masked)
        } } }
        let out = try CPUOps.attentionApply(probs: CPUOps.softmax(scores), values: v)
        try assertClose(out, c.doubles("expected"), atol: 3e-6, rtol: 1e-5, "attention")
    }

    func testQuantizedFormatsAgainstGGUFReference() throws {
        for name in ["q8_0", "q4_0"] {
            let c = try ops.dict(name)
            let dtype: DType = name == "q8_0" ? .q8_0 : .q4_0
            let raw = Data(base64Encoded: c["blocks"] as! String)!
            let (k, n, m) = (c.int("k"), c.int("n"), c.int("m"))
            XCTAssertEqual(raw.count, dtype.byteCount(elements: n * k))
            let storage = try TensorStorage(byteCount: raw.count)
            raw.withUnsafeBytes { memcpy(storage.pointer, $0.baseAddress!, raw.count) }
            let w = try Tensor(storage: storage, dtype: dtype, shape: [n, k])
            // Block decoding must reproduce the reference dequantisation (products of an f16 scale and small ints, exact in f32).
            try assertClose(Tensor(w.toFloatArray(), shape: [n, k]), c.doubles("dequantized"), atol: 1e-6, rtol: 1e-6, "\(name) dequantize")
            // Quantised mat-vec must equal float mat-vec on the dequantised weights (arithmetic error only).
            let x = try Tensor(c.floats("x"), shape: [m, k])
            try assertClose(CPUOps.linear(x, weight: w), c.doubles("expected"), atol: 2e-5, rtol: 1e-5, "\(name) linear vs dequantized reference")
            // alpaca's reference quantiser must produce byte-identical blocks to the gguf package.
            let mine = try quantize(c.floats("original"), to: dtype)
            let theirs = Data(bytes: mine.storage.pointer, count: mine.storage.byteCount)
            XCTAssertEqual(theirs, raw, "\(name) quantiser output differs from gguf reference")
        }
    }

    func testEmbedding() throws {
        let c = try ops.dict("embedding")
        let t = try Tensor(c.floats("table"), shape: [c.int("vocab"), c.int("dim")])
        try assertClose(CPUOps.embed(tokens: c.ints("tokens").map(Int32.init), table: t), c.doubles("expected"), atol: 0, "embedding")
        XCTAssertThrowsError(try CPUOps.embed(tokens: [12], table: t))
        XCTAssertThrowsError(try CPUOps.embed(tokens: [-1], table: t))
    }

    // MARK: Validation

    func testShapeAndDTypeValidation() throws {
        let a = try Tensor(zeros: [2, 3]), b = try Tensor(zeros: [3, 2])
        XCTAssertThrowsError(try CPUOps.add(a, b))
        XCTAssertThrowsError(try CPUOps.matmul(a, a))
        let h = try Tensor(zeros: [2, 3], dtype: .float16)
        XCTAssertThrowsError(try CPUOps.add(a, h))
        XCTAssertThrowsError(try CPUOps.rmsNorm(a, weight: Tensor(zeros: [4]), eps: 1e-5))
        XCTAssertThrowsError(try CPUOps.rmsNorm(a, weight: Tensor(zeros: [3]), eps: 0))
        XCTAssertThrowsError(try CPUOps.rope(Tensor(zeros: [1, 1, 3]), startPosition: 0, theta: 1e4))
        XCTAssertThrowsError(try Tensor(zeros: [5], dtype: .q8_0))      // not a whole block
        XCTAssertThrowsError(try Tensor(zeros: [-1]))
        XCTAssertThrowsError(try Tensor([1, 2, 3], shape: [2, 2]))
    }

    func testViewsShareStorageAndAreBoundsChecked() throws {
        let t = try Tensor([1, 2, 3, 4, 5, 6], shape: [2, 3])
        let tr = try t.transposed(0, 1)
        XCTAssertFalse(tr.isContiguous)
        XCTAssertEqual(try tr.toFloatArray(), [1, 4, 2, 5, 3, 6])
        XCTAssertEqual(try t.slice(axis: 1, 1..<3).toFloatArray(), [2, 3, 5, 6])
        XCTAssertThrowsError(try t.slice(axis: 1, 2..<4))
        XCTAssertThrowsError(try t.reshaped([4]))
        XCTAssertThrowsError(try tr.reshaped([6]))
        XCTAssertThrowsError(try Tensor(storage: t.storage, dtype: .float32, shape: [7]))
        try t.reshaped([3, 2]).withFloat32 { $0[0] = 9 }
        XCTAssertEqual(try t.toFloatArray()[0], 9)
        // Non-contiguous operands to ops that need contiguity are rejected, not misread.
        XCTAssertThrowsError(try CPUOps.linear(tr, weight: Tensor(zeros: [4, 2])))
    }
}

extension Array {
    func chunks(of n: Int) -> [[Element]] { stride(from: 0, to: count, by: n).map { Array(self[$0..<Swift.min($0 + n, count)]) } }
}
