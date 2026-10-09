// Copyright (c) 2026 Torin Etheridge. MIT License (see LICENSE).
// Author: Torin Etheridge · 2026-10-09

import Foundation
import XCTest
import AlpacaCore
import AlpacaModels

enum ModelFiles {
    static var directory: URL {
        let dir = ProcessInfo.processInfo.environment["ALPACA_MODEL_DIR"]
            ?? URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
                .appendingPathComponent("Models/gguf").path
        return URL(fileURLWithPath: dir)
    }

    static func url(_ quant: String) throws -> URL {
        let url = directory.appendingPathComponent("SmolLM2-135M-Instruct-\(quant).gguf")
        try XCTSkipUnless(FileManager.default.fileExists(atPath: url.path), "model not found at \(url.path); see Docs/DEVELOPMENT.md")
        return url
    }
}

struct ReferenceCase {
    let prompt: String
    let ids: [Int32]
    let lastLogits: [Float]
    let generated: [Int32]
    let margins: [Double]
}

enum Reference {
    static func load(_ key: String) throws -> [ReferenceCase] {
        let url = try XCTUnwrap(Bundle.module.url(forResource: "Fixtures/reference", withExtension: "json"))
        let root = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: [[String: Any]]])
        return try XCTUnwrap(root[key]).map { e in
            let raw = Data(base64Encoded: e["last_logits"] as! String)!
            let logits = raw.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
            return ReferenceCase(
                prompt: e["prompt"] as! String, ids: (e["ids"] as! [NSNumber]).map { Int32(truncating: $0) },
                lastLogits: logits, generated: (e["generated"] as! [NSNumber]).map { Int32(truncating: $0) },
                margins: (e["margins"] as! [NSNumber]).map(\.doubleValue))
        }
    }
}

func argmax(_ v: [Float]) -> Int32 { Int32(v.enumerated().max { $0.element < $1.element }!.offset) }

/// Real-model tests run the 135M-parameter CPU reference, which is impractically slow unoptimised.
func requireOptimizedBuild() throws {
    #if DEBUG
    throw XCTSkip("real-model tests need an optimised build: swift test -c release -Xswiftc -enable-testing")
    #endif
}

enum LanguageModelTestSupport {
    static var metalAvailable: Bool { MetalContextProbe.available }
}
import Metal
enum MetalContextProbe { static var available: Bool { MTLCreateSystemDefaultDevice() != nil } }
