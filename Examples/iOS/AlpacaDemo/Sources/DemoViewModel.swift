// Copyright (c) 2026 Torin Etheridge. MIT License (see LICENSE).
// Author: Torin Etheridge · 2026-10-09

import Foundation
import Alpaca
import os

/// Minimal state holder: model selection → load → streaming generation with cancel and diagnostics.
@MainActor
@Observable
final class DemoViewModel {
    var modelName = "No model selected"
    var status = "Choose a .gguf model file."
    var output = ""
    var prompt = "Explain how transformers work in two sentences."
    var isLoading = false
    var isGenerating = false
    var decodeRate = 0.0
    var prefillRate = 0.0
    var residentMiB = 0
    var availableMiB = 0

    private var model: LanguageModel?
    private var stream: GenerationStream?
    private var task: Task<Void, Never>?
    private var securityScopedURL: URL?

    var canGenerate: Bool { model != nil && !isGenerating && !prompt.isEmpty }

    func select(_ url: URL) {
        stopAccessing()
        _ = url.startAccessingSecurityScopedResource()
        securityScopedURL = url
        modelName = url.lastPathComponent
        Task { await load(url) }
    }

    private func load(_ url: URL) async {
        cancel()
        model?.unload(); model = nil
        isLoading = true; status = "Loading…"
        do {
            // The budget check runs before any large allocation; failures are reported, not crashed on.
            let m = try await LanguageModel.load(from: url)
            model = m
            status = "Loaded on \(m.info.backend) — \(m.info.estimatedMemory)"
        } catch {
            status = "Load failed: \(error)"
        }
        isLoading = false
        updateMemory()
    }

    func generate() {
        guard let model else { return }
        output = ""; isGenerating = true; status = "Generating…"
        let chat = "<|im_start|>user\n\(prompt)<|im_end|>\n<|im_start|>assistant\n"   // ChatML (SmolLM2); adapt per model
        let s = model.generate(prompt: chat, configuration: .init(maxTokens: 256, temperature: 0.7, topP: 0.95))
        stream = s
        task = Task {
            do {
                for try await token in s {
                    output += token.text
                    decodeRate = s.summary.decodeTokensPerSecond
                    updateMemory()
                }
                let sum = s.summary
                prefillRate = sum.prefillTokensPerSecond; decodeRate = sum.decodeTokensPerSecond
                status = "Done (\(sum.finishReason)) — \(sum.generatedTokens) tokens"
            } catch {
                status = "Generation failed: \(error)"
            }
            isGenerating = false
            updateMemory()
        }
    }

    func cancel() {
        stream?.cancel()
        task?.cancel()
    }

    /// Call when the app leaves the foreground: GPU work is not allowed in the background and memory is reclaimed first.
    func appMovedToBackground() {
        model?.cancelAllGenerations()
        model?.trimMemory()          // drop the idle KV cache / scratch kept for fast restarts
    }

    func updateMemory() {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size / MemoryLayout<integer_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count) }
        }
        if kr == KERN_SUCCESS { residentMiB = Int(info.resident_size) >> 20 }
        availableMiB = Int(os_proc_available_memory()) >> 20
    }

    private func stopAccessing() { securityScopedURL?.stopAccessingSecurityScopedResource(); securityScopedURL = nil }
}
