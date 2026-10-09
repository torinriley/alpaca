// Copyright (c) 2026 Torin Etheridge. MIT License (see LICENSE).
// Author: Torin Etheridge · 2026-10-09

import Foundation
import XCTest
@testable import AlpacaCore

enum Fixtures {
    static func load(_ name: String) throws -> [String: Any] {
        let url = try XCTUnwrap(Bundle.module.url(forResource: "Fixtures/\(name)", withExtension: "json"))
        return try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
    }
}

extension Dictionary where Key == String, Value == Any {
    func floats(_ key: String) -> [Float] { (self[key] as! [Double]).map(Float.init) }
    func doubles(_ key: String) -> [Double] { self[key] as! [Double] }
    func int(_ key: String) -> Int { (self[key] as! NSNumber).intValue }
    func dict(_ key: String) -> [String: Any] { self[key] as! [String: Any] }
    func ints(_ key: String) -> [Int] { (self[key] as! [NSNumber]).map(\.intValue) }
}

/// Asserts |actual-expected| <= atol + rtol*|expected| elementwise and logs the error metrics.
func assertClose(_ t: Tensor, _ expected: [Double], atol: Double, rtol: Double = 0, _ label: String,
                 file: StaticString = #filePath, line: UInt = #line) throws {
    let actual = try t.toFloatArray()
    XCTAssertEqual(actual.count, expected.count, "\(label): length", file: file, line: line)
    let m = ErrorMetrics(actual: actual, expected: expected)
    print("[metrics] \(label): \(m)  (atol=\(atol) rtol=\(rtol))")
    for (i, (a, e)) in zip(actual, expected).enumerated() where abs(Double(a) - e) > atol + rtol * abs(e) {
        XCTFail("\(label)[\(i)]: \(a) vs \(e) exceeds atol \(atol) rtol \(rtol)", file: file, line: line)
        return
    }
}
