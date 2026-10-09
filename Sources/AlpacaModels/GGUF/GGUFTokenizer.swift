// Copyright (c) 2026 Torin Etheridge. MIT License (see LICENSE).
// Author: Torin Etheridge · 2026-10-09

import AlpacaTokenizers

extension GGUFFile {
    /// Builds the byte-level BPE tokenizer described by `tokenizer.ggml.*` metadata.
    public func makeTokenizer() throws -> BPETokenizer {
        guard let model = string("tokenizer.ggml.model") else { throw GGUFError.missingMetadata("tokenizer.ggml.model") }
        guard model == "gpt2" else { throw GGUFError.unsupportedModel("tokenizer.ggml.model '\(model)' (only 'gpt2' byte-level BPE is supported)") }
        guard let tokenValues = metadata["tokenizer.ggml.tokens"]?.asArray else { throw GGUFError.missingMetadata("tokenizer.ggml.tokens") }
        guard let mergeValues = metadata["tokenizer.ggml.merges"]?.asArray else { throw GGUFError.missingMetadata("tokenizer.ggml.merges") }
        let tokens = try tokenValues.map { v -> String in
            guard let s = v.asString else { throw GGUFError.malformed("tokenizer.ggml.tokens contains a non-string") }
            return s
        }
        let merges = try mergeValues.map { v -> String in
            guard let s = v.asString else { throw GGUFError.malformed("tokenizer.ggml.merges contains a non-string") }
            return s
        }
        let types = try metadata["tokenizer.ggml.token_type"]?.asArray?.map { v -> Int32 in
            guard let i = v.asInt, let t = Int32(exactly: i) else { throw GGUFError.malformed("invalid token_type entry") }
            return t
        }
        func id(_ key: String) -> Int32? { int(key).flatMap { Int32(exactly: $0) } }
        do {
            return try BPETokenizer(
                tokens: tokens, merges: merges, tokenTypes: types, preTokenizer: string("tokenizer.ggml.pre") ?? "gpt2",
                bosTokenID: id("tokenizer.ggml.bos_token_id"), eosTokenID: id("tokenizer.ggml.eos_token_id"),
                addsBOSByDefault: metadata["tokenizer.ggml.add_bos_token"]?.asBool ?? false)
        } catch let e as TokenizerError {
            throw GGUFError.unsupportedModel("\(e)")
        }
    }
}
