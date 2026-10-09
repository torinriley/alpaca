// Copyright (c) 2026 Torin Etheridge. MIT License (see LICENSE).
// Author: Torin Etheridge · 2026-10-09

import Foundation

/// Error statistics between a computed result and a reference.
public struct ErrorMetrics: Sendable, CustomStringConvertible {
    public let count: Int
    public let maxAbs: Double
    public let meanAbs: Double
    public let rms: Double
    /// max |a-e| / max(|e|) over the reference — a scale-aware error that is meaningful when entries are near zero.
    public let maxRelativeToPeak: Double

    public init(actual: [Float], expected: [Double]) {
        precondition(actual.count == expected.count, "length mismatch")
        count = actual.count
        var mx = 0.0, sum = 0.0, sq = 0.0, peak = 0.0
        for (a, e) in zip(actual, expected) {
            let d = abs(Double(a) - e)
            mx = max(mx, d); sum += d; sq += d * d; peak = max(peak, abs(e))
        }
        maxAbs = mx
        meanAbs = count > 0 ? sum / Double(count) : 0
        rms = count > 0 ? (sq / Double(count)).squareRoot() : 0
        maxRelativeToPeak = peak > 0 ? mx / peak : mx
    }

    public init(actual: [Float], expected: [Float]) {
        self.init(actual: actual, expected: expected.map(Double.init))
    }

    public var description: String {
        String(format: "n=%d maxAbs=%.3e meanAbs=%.3e rms=%.3e maxRel(peak)=%.3e", count, maxAbs, meanAbs, rms, maxRelativeToPeak)
    }
}
