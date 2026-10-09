// Copyright (c) 2026 Torin Etheridge. MIT License (see LICENSE).
// Author: Torin Etheridge · 2026-10-09

import Foundation

public enum TokenizerError: Error, Equatable, CustomStringConvertible {
    case invalidVocabulary(String)
    case unsupportedPreTokenizer(String)
    case unknownTokenID(Int32)
    public var description: String {
        switch self {
        case .invalidVocabulary(let m): return "invalid tokenizer vocabulary: \(m)"
        case .unsupportedPreTokenizer(let p): return "unsupported pre-tokenizer '\(p)' (supported: gpt2, smollm)"
        case .unknownTokenID(let id): return "token id \(id) is outside the vocabulary"
        }
    }
}

/// Byte-level BPE tokenizer (GPT-2 family) driven by GGUF-style vocabulary data.
///
/// Encoding pipeline, matching the Hugging Face `tokenizer.json` of the supported models:
/// 1. split out literal special tokens (token types control/user-defined) when `parseSpecial` is true;
/// 2. pre-tokenize each remaining span — for `smollm`, isolate every decimal digit, then apply the GPT-2 regex;
/// 3. map each piece's UTF-8 bytes through the GPT-2 byte→unicode table;
/// 4. repeatedly merge the adjacent pair with the lowest merge rank;
/// 5. look the resulting symbols up in the vocabulary.
public final class BPETokenizer: @unchecked Sendable {
    public let vocabularySize: Int
    public let bosTokenID: Int32?
    public let eosTokenID: Int32?
    public let addsBOSByDefault: Bool

    private let tokens: [String]
    private let tokenTypes: [Int32]
    private let tokenToID: [String: Int32]
    private let mergeRanks: [MergeKey: Int]
    private let specialTokens: [(text: String, id: Int32)]   // longest first
    private let isolateDigits: Bool
    private let wordRegex: NSRegularExpression
    private let cacheLock = NSLock()
    nonisolated(unsafe) private var wordCache: [String: [Int32]] = [:]

    struct MergeKey: Hashable { let a: String; let b: String }

    /// GGUF `token_type` values.
    public enum TokenType: Int32 { case normal = 1, unknown = 2, control = 3, userDefined = 4, unused = 5, byte = 6 }

    /// GPT-2 contraction / word / number / punctuation / whitespace pre-tokenization regex.
    static let gpt2Pattern = #"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"#

    public init(
        tokens: [String], merges: [String], tokenTypes: [Int32]? = nil, preTokenizer: String,
        bosTokenID: Int32?, eosTokenID: Int32?, addsBOSByDefault: Bool
    ) throws {
        guard !tokens.isEmpty, tokens.count < Int(Int32.max) else { throw TokenizerError.invalidVocabulary("empty or oversized vocabulary") }
        if let t = tokenTypes, t.count != tokens.count { throw TokenizerError.invalidVocabulary("token_type count \(t.count) != token count \(tokens.count)") }
        switch preTokenizer {
        case "gpt2": isolateDigits = false
        case "smollm": isolateDigits = true
        default: throw TokenizerError.unsupportedPreTokenizer(preTokenizer)
        }
        for id in [bosTokenID, eosTokenID].compactMap({ $0 }) where id < 0 || Int(id) >= tokens.count {
            throw TokenizerError.invalidVocabulary("special token id \(id) outside vocabulary")
        }
        self.tokens = tokens
        self.tokenTypes = tokenTypes ?? [Int32](repeating: TokenType.normal.rawValue, count: tokens.count)
        var map: [String: Int32] = [:]
        map.reserveCapacity(tokens.count)
        for (i, t) in tokens.enumerated() where map[t] == nil { map[t] = Int32(i) }   // first id wins on duplicates
        tokenToID = map
        var ranks: [MergeKey: Int] = [:]
        ranks.reserveCapacity(merges.count)
        for (rank, m) in merges.enumerated() {
            let parts = m.split(separator: " ", maxSplits: 1, omittingEmptySubsequences: false)
            guard parts.count == 2 else { throw TokenizerError.invalidVocabulary("merge rule \(rank) '\(m)' is not 'left right'") }
            let key = MergeKey(a: String(parts[0]), b: String(parts[1]))
            if ranks[key] == nil { ranks[key] = rank }
        }
        mergeRanks = ranks
        var specials: [(String, Int32)] = []
        for (i, t) in tokens.enumerated() {
            let type = self.tokenTypes[i]
            if type == TokenType.control.rawValue || type == TokenType.userDefined.rawValue, !t.isEmpty { specials.append((t, Int32(i))) }
        }
        specialTokens = specials.sorted { $0.0.utf8.count > $1.0.utf8.count }.map { (text: $0.0, id: $0.1) }
        vocabularySize = tokens.count
        self.bosTokenID = bosTokenID; self.eosTokenID = eosTokenID; self.addsBOSByDefault = addsBOSByDefault
        wordRegex = try NSRegularExpression(pattern: Self.gpt2Pattern)
    }

    // MARK: Encode

    public func encode(_ text: String, addBOS: Bool? = nil, parseSpecial: Bool = true) throws -> [Int32] {
        var ids: [Int32] = []
        if addBOS ?? addsBOSByDefault, let bos = bosTokenID { ids.append(bos) }
        for span in splitSpecial(text, enabled: parseSpecial) {
            switch span {
            case .special(let id): ids.append(id)
            case .text(let s): for piece in preTokenize(s) { ids += try encodeWord(piece) }
            }
        }
        return ids
    }

    enum Span { case text(String), special(Int32) }

    func splitSpecial(_ text: String, enabled: Bool) -> [Span] {
        guard enabled, !specialTokens.isEmpty, !text.isEmpty else { return text.isEmpty ? [] : [.text(text)] }
        var spans: [Span] = []
        var pending = text.startIndex
        var searchFrom = text.startIndex
        while searchFrom < text.endIndex {
            // Earliest match wins; ties go to the longest token (specialTokens is sorted longest first).
            var best: (range: Range<String.Index>, id: Int32)?
            for s in specialTokens {
                if let r = text.range(of: s.text, options: .literal, range: searchFrom..<text.endIndex) {
                    if best == nil || r.lowerBound < best!.range.lowerBound { best = (r, s.id) }
                }
            }
            guard let found = best else { break }
            if pending < found.range.lowerBound { spans.append(.text(String(text[pending..<found.range.lowerBound]))) }
            spans.append(.special(found.id))
            pending = found.range.upperBound
            searchFrom = pending
        }
        if pending < text.endIndex { spans.append(.text(String(text[pending...]))) }
        return spans
    }

    /// Splits `text` into the pieces that BPE runs on independently.
    func preTokenize(_ text: String) -> [String] {
        var segments: [String] = []
        if isolateDigits {
            var current = ""
            for ch in text.unicodeScalars {
                // The `tokenizers` Digits pre-tokenizer isolates every Unicode numeric character (categories Nd, Nl, No).
                if [.decimalNumber, .letterNumber, .otherNumber].contains(ch.properties.generalCategory) {
                    if !current.isEmpty { segments.append(current); current = "" }
                    segments.append(String(ch))
                } else {
                    current.unicodeScalars.append(ch)
                }
            }
            if !current.isEmpty { segments.append(current) }
        } else if !text.isEmpty {
            segments = [text]
        }
        var pieces: [String] = []
        for seg in segments {
            let ns = seg as NSString
            var last = 0
            wordRegex.enumerateMatches(in: seg, range: NSRange(location: 0, length: ns.length)) { m, _, _ in
                guard let r = m?.range else { return }
                if r.location > last { pieces.append(ns.substring(with: NSRange(location: last, length: r.location - last))) }
                pieces.append(ns.substring(with: r))
                last = r.location + r.length
            }
            if last < ns.length { pieces.append(ns.substring(from: last)) }
        }
        return pieces
    }

    func encodeWord(_ word: String) throws -> [Int32] {
        cacheLock.lock()
        if let hit = wordCache[word] { cacheLock.unlock(); return hit }
        cacheLock.unlock()

        var symbols: [String] = word.utf8.map { String(ByteLevel.byteToScalar[$0]!) }
        while symbols.count > 1 {
            var bestRank = Int.max, bestIndex = -1
            for i in 0..<(symbols.count - 1) {
                if let r = mergeRanks[MergeKey(a: symbols[i], b: symbols[i + 1])], r < bestRank { bestRank = r; bestIndex = i }
            }
            if bestIndex < 0 { break }
            symbols[bestIndex] += symbols[bestIndex + 1]
            symbols.remove(at: bestIndex + 1)
        }
        var ids: [Int32] = []
        for s in symbols {
            if let id = tokenToID[s] { ids.append(id); continue }
            // A symbol missing from the vocabulary: fall back to its individual byte tokens.
            for scalar in s.unicodeScalars {
                guard let id = tokenToID[String(scalar)] else { throw TokenizerError.invalidVocabulary("no token for byte symbol U+\(String(scalar.value, radix: 16))") }
                ids.append(id)
            }
        }
        cacheLock.lock(); wordCache[word] = ids; cacheLock.unlock()
        return ids
    }

    // MARK: Decode

    public func isSpecial(_ id: Int32) -> Bool {
        guard id >= 0, Int(id) < tokens.count else { return false }
        let t = tokenTypes[Int(id)]
        return t == TokenType.control.rawValue || t == TokenType.userDefined.rawValue
    }

    /// Raw bytes represented by token `id`.
    public func bytes(for id: Int32, skipSpecial: Bool = true) throws -> [UInt8] {
        guard id >= 0, Int(id) < tokens.count else { throw TokenizerError.unknownTokenID(id) }
        if isSpecial(id) { return skipSpecial ? [] : Array(tokens[Int(id)].utf8) }
        var out: [UInt8] = []
        for scalar in tokens[Int(id)].unicodeScalars {
            if let b = ByteLevel.scalarToByte[scalar] { out.append(b) } else { out += Array(String(scalar).utf8) }
        }
        return out
    }

    public func decode(_ ids: [Int32], skipSpecial: Bool = true) throws -> String {
        var bytes: [UInt8] = []
        for id in ids { bytes += try self.bytes(for: id, skipSpecial: skipSpecial) }
        return String(decoding: bytes, as: UTF8.self)
    }

    /// Incremental decoder: feed token ids, receive text as soon as complete UTF-8 sequences are available.
    public func makeStreamDecoder(skipSpecial: Bool = true) -> StreamDecoder { StreamDecoder(tokenizer: self, skipSpecial: skipSpecial) }

    public struct StreamDecoder {
        let tokenizer: BPETokenizer
        let skipSpecial: Bool
        private var pending: [UInt8] = []

        init(tokenizer: BPETokenizer, skipSpecial: Bool) { self.tokenizer = tokenizer; self.skipSpecial = skipSpecial }

        /// Appends `id`; returns the newly completed text (possibly empty if a multi-byte character is still incomplete).
        public mutating func append(_ id: Int32) throws -> String {
            pending += try tokenizer.bytes(for: id, skipSpecial: skipSpecial)
            let complete = Self.completePrefixLength(pending)
            guard complete > 0 else { return "" }
            let text = String(decoding: pending[0..<complete], as: UTF8.self)
            pending.removeFirst(complete)
            return text
        }

        /// Flushes any trailing incomplete bytes (as replacement characters).
        public mutating func finish() -> String {
            defer { pending.removeAll() }
            return pending.isEmpty ? "" : String(decoding: pending, as: UTF8.self)
        }

        /// Length of the longest prefix that does not end in the middle of a UTF-8 sequence.
        static func completePrefixLength(_ b: [UInt8]) -> Int {
            var i = b.count - 1, back = 0
            while i >= 0, back < 4 {
                let byte = b[i]
                if byte & 0xC0 == 0x80 { i -= 1; back += 1; continue }          // continuation byte
                let need = byte >= 0xF0 ? 4 : byte >= 0xE0 ? 3 : byte >= 0xC0 ? 2 : 1
                return (back + 1 < need) ? i : b.count                          // lead byte with missing continuations
            }
            return b.count
        }
    }
}
