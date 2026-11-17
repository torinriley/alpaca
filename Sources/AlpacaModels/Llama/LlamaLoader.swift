// Copyright (c) 2026 Torin Etheridge. MIT License (see LICENSE).
// Author: Torin Etheridge · 2026-10-09

import Foundation
import AlpacaCore

/// What happened while mapping a GGUF file onto Llama weights.
public struct LoadReport: Sendable {
    /// Bytes of weights referenced directly from the memory-mapped file (no copy).
    public var mappedWeightBytes = 0
    /// Bytes of weights materialised in anonymous memory (converted tensors).
    public var convertedWeightBytes = 0
    /// Human-readable notes for every non-trivial decision (e.g. Q4_1 → f16 conversion).
    public var notes: [String] = []
}

public struct LoadedLlama: Sendable {
    public let config: LlamaConfig
    public let weights: LlamaWeights
    public let file: GGUFFile
    public let report: LoadReport
}

/// Maps a GGUF file with `general.architecture == "llama"` to `LlamaConfig` + `LlamaWeights`.
///
/// Supported on-disk tensor types, all used in place from the mapping (no conversion, no copy): F32 (norms), F16, BF16,
/// Q8_0, Q5_0, Q5_1, Q4_0, Q4_1 and the K-quants Q2_K, Q3_K, Q4_K, Q5_K, Q6_K. Anything else (e.g. IQ-quants) is rejected with
/// `GGUFError.unsupportedTensorType`.
/// RoPE: GGUF Llama weights use the interleaved (adjacent-pair) layout; Llama 3.x frequency scaling is applied from `rope_freqs.weight`, other scaling types are rejected.
public enum LlamaLoader {
    public static func load(url: URL) throws -> LoadedLlama {
        try load(file: GGUFFile(url: url))
    }

    public static func load(file: GGUFFile) throws -> LoadedLlama {
        guard file.string("general.architecture") == "llama" else {
            throw GGUFError.unsupportedModel("general.architecture is '\(file.string("general.architecture") ?? "<missing>")', only 'llama' is supported")
        }
        func req(_ key: String) throws -> Int {
            guard let v = file.int(key) else { throw GGUFError.missingMetadata(key) }
            return v
        }
        let heads = try req("llama.attention.head_count")
        let hidden = try req("llama.embedding_length")
        guard heads > 0 else { throw GGUFError.malformed("llama.attention.head_count is 0") }
        let headDim = file.int("llama.attention.key_length") ?? (hidden / heads)
        if let valueLen = file.int("llama.attention.value_length"), valueLen != headDim {
            throw GGUFError.unsupportedModel("attention.value_length \(valueLen) != key_length \(headDim)")
        }
        if let rd = file.int("llama.rope.dimension_count"), rd != headDim {
            throw GGUFError.unsupportedModel("partial rotary embedding (rope dimension_count \(rd) != head dim \(headDim)) is not supported")
        }
        if let s = file.string("llama.rope.scaling.type"), s != "none" {
            throw GGUFError.unsupportedModel("RoPE scaling '\(s)' is not supported (Llama 3.x scaling is read from the rope_freqs.weight tensor)")
        }
        // Llama 3.1 / 3.2 "llama3" scaling: the converter bakes it into a per-pair factor table (rope_freqs.weight); angle = pos · θᵢ / factorᵢ.
        var freqFactors: [Float]?
        if let info = file.info(for: "rope_freqs.weight") {
            guard info.type == .f32, info.elementCount == headDim / 2 else {
                throw GGUFError.unsupportedModel("rope_freqs.weight must be \(headDim / 2) F32 values, found \(info.elementCount) \(info.type)")
            }
            freqFactors = try file.tensor(named: "rope_freqs.weight").toFloatArray()
        }
        if file.metadata["llama.expert_count"] != nil, (file.int("llama.expert_count") ?? 0) > 0 {
            throw GGUFError.unsupportedModel("mixture-of-experts models are not supported")
        }
        guard let embd = file.info(for: "token_embd.weight") else { throw GGUFError.missingTensor("token_embd.weight") }
        guard embd.dims.count == 2 else { throw GGUFError.malformed("token_embd.weight must be 2-D") }

        let config = LlamaConfig(
            vocabSize: embd.dims[1], hiddenSize: hidden, layerCount: try req("llama.block_count"),
            headCount: heads, kvHeadCount: file.int("llama.attention.head_count_kv") ?? heads, headDim: headDim,
            feedForwardSize: try req("llama.feed_forward_length"), contextLength: try req("llama.context_length"),
            rmsNormEps: Float(file.double("llama.attention.layer_norm_rms_epsilon") ?? 1e-5),
            ropeTheta: Float(file.double("llama.rope.freq_base") ?? 10000), ropeInterleaved: true, ropeFreqFactors: freqFactors)
        do { try config.validate() } catch { throw GGUFError.unsupportedModel("\(error)") }

        var report = LoadReport()
        func weight(_ name: String, norm: Bool = false) throws -> Tensor {
            guard let info = file.info(for: name) else { throw GGUFError.missingTensor(name) }
            if norm {
                guard info.type == .f32 else { throw GGUFError.unsupportedModel("norm tensor '\(name)' is \(info.type), expected F32") }
            }
            if info.type.nativeDType != nil {
                report.mappedWeightBytes += info.byteCount
                return try file.tensor(named: name)
            }
            throw GGUFError.unsupportedTensorType(id: info.type.rawValue, tensor: name)
        }

        var layers: [LlamaLayerWeights] = []
        for i in 0..<config.layerCount {
            let p = "blk.\(i)."
            layers.append(LlamaLayerWeights(
                attnNorm: try weight(p + "attn_norm.weight", norm: true),
                wq: try weight(p + "attn_q.weight"), wk: try weight(p + "attn_k.weight"),
                wv: try weight(p + "attn_v.weight"), wo: try weight(p + "attn_output.weight"),
                ffnNorm: try weight(p + "ffn_norm.weight", norm: true),
                wGate: try weight(p + "ffn_gate.weight"), wUp: try weight(p + "ffn_up.weight"),
                wDown: try weight(p + "ffn_down.weight")))
        }
        let weights = LlamaWeights(
            tokenEmbedding: try weight("token_embd.weight"), layers: layers,
            outputNorm: try weight("output_norm.weight", norm: true),
            output: file.info(for: "output.weight") != nil ? try weight("output.weight") : nil)
        do { try weights.validate(against: config) } catch { throw GGUFError.malformed("\(error)") }
        return LoadedLlama(config: config, weights: weights, file: file, report: report)
    }
}
