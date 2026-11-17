// Copyright (c) 2026 Torin Etheridge. MIT License (see LICENSE).
// Author: Torin Etheridge · 2026-10-09

import XCTest
import AlpacaModels
@testable import AlpacaTokenizers

/// Token-id equality against Hugging Face `tokenizers` for 191 strings (Scripts/generate_reference.py):
/// English, whitespace variants, punctuation, numbers, Unicode, emoji, multilingual text, empty input,
/// special tokens, and seeded random strings.
final class TokenizerTests: XCTestCase {
    static func modelURL(_ file: String) -> URL {
        let dir = ProcessInfo.processInfo.environment["ALPACA_MODEL_DIR"]
            ?? URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
                .appendingPathComponent("Models/gguf").path
        return URL(fileURLWithPath: dir).appendingPathComponent(file)
    }

    func makeTokenizer() throws -> BPETokenizer {
        let url = Self.modelURL("SmolLM2-135M-Instruct-f16.gguf")
        try XCTSkipUnless(FileManager.default.fileExists(atPath: url.path), "model not found at \(url.path); see Docs/DEVELOPMENT.md")
        return try GGUFFile(url: url).makeTokenizer()
    }

    func testTokenIDsMatchReferenceTokenizer() throws {
        let tok = try makeTokenizer()
        let fixture = try XCTUnwrap(Bundle.module.url(forResource: "Fixtures/corpus", withExtension: "json"))
        let corpus = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: fixture)) as? [[String: Any]])
        XCTAssertGreaterThan(corpus.count, 150)
        var failures = 0
        for entry in corpus {
            let text = entry["text"] as! String
            let expected = (entry["ids"] as! [NSNumber]).map { Int32(truncating: $0) }
            let got = try tok.encode(text, addBOS: false)
            if got != expected {
                failures += 1
                XCTFail("ids differ for \(String(reflecting: text)):\n  got      \(got)\n  expected \(expected)")
            }
        }
        print("[tokenizer] \(corpus.count - failures)/\(corpus.count) corpus entries match reference token ids")
    }

    func testRoundTripAndStreamingDecode() throws {
        let tok = try makeTokenizer()
        for text in ["Hello, world!", "  spaces  ", "日本語のテキスト 🧑‍🚀 café", "line\n\nbreaks\t", "1234 5678"] {
            let ids = try tok.encode(text)
            XCTAssertEqual(try tok.decode(ids), text)
            // Streaming: one token at a time must reassemble the exact text, never emitting broken UTF-8.
            var dec = tok.makeStreamDecoder()
            var out = ""
            for id in ids {
                let piece = try dec.append(id)
                XCTAssertFalse(piece.contains("\u{FFFD}"), "partial UTF-8 leaked for \(text)")
                out += piece
            }
            out += dec.finish()
            XCTAssertEqual(out, text)
        }
    }

    func testSpecialTokenHandling() throws {
        let tok = try makeTokenizer()
        XCTAssertEqual(try tok.encode("<|im_start|>"), [1])
        XCTAssertEqual(try tok.encode("<|im_end|>"), [2])
        XCTAssertNotEqual(try tok.encode("<|im_end|>", parseSpecial: false), [2], "literal text when special parsing is off")
        XCTAssertEqual(try tok.encode(""), [])
        XCTAssertEqual(try tok.encode("", addBOS: true), [1], "explicit BOS request on empty string")
        XCTAssertEqual(try tok.decode([1, 2]), "")
        XCTAssertEqual(try tok.decode([1, 2], skipSpecial: false), "<|im_start|><|im_end|>")
        XCTAssertThrowsError(try tok.decode([999_999]))
        XCTAssertThrowsError(try tok.decode([-1]))
    }

    func testUnsupportedPreTokenizerIsRejected() {
        XCTAssertThrowsError(try BPETokenizer(tokens: ["a"], merges: [], preTokenizer: "llama-bpe", bosTokenID: nil, eosTokenID: nil, addsBOSByDefault: false))
        XCTAssertThrowsError(try BPETokenizer(tokens: ["a"], merges: ["bad"], preTokenizer: "gpt2", bosTokenID: nil, eosTokenID: nil, addsBOSByDefault: false))
        XCTAssertThrowsError(try BPETokenizer(tokens: [], merges: [], preTokenizer: "gpt2", bosTokenID: nil, eosTokenID: nil, addsBOSByDefault: false))
    }
}

/// Llama 3 ("llama-bpe") pre-tokenization, checked on a synthetic byte-level vocabulary (no model file needed).
final class LlamaBPEPreTokenizerTests: XCTestCase {
    func makeTokenizer(pre: String) throws -> BPETokenizer {
        let tokens = (0..<256).map { String(ByteLevel.byteToScalar[UInt8($0)]!) }
        return try BPETokenizer(tokens: tokens, merges: [], preTokenizer: pre, bosTokenID: nil, eosTokenID: nil, addsBOSByDefault: false)
    }

    func testDigitsInGroupsOfThreeContractionsAndNewlines() throws {
        let t = try makeTokenizer(pre: "llama-bpe")
        XCTAssertEqual(t.preTokenize("It's 12345 DON'T\n\nok"), ["It", "'s", " ", "123", "45", " DON", "'T", "\n\n", "ok"])
        XCTAssertEqual(t.preTokenize("x=1.5;\n"), ["x", "=", "1", ".", "5", ";\n"])
    }

    func testGPT2PatternUnchanged() throws {
        let t = try makeTokenizer(pre: "gpt2")
        XCTAssertEqual(t.preTokenize("It's 12345"), ["It", "'s", " 12345"])
        XCTAssertThrowsError(try makeTokenizer(pre: "no-such-pre-tokenizer"))
    }
}
