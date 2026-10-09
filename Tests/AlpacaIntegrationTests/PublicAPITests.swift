// Copyright (c) 2026 Torin Etheridge. MIT License (see LICENSE).
// Author: Torin Etheridge · 2026-10-09

import XCTest
import AlpacaCore
@testable import Alpaca

func greedy(_ n: Int) -> GenerationConfiguration { GenerationConfiguration(maxTokens: n, temperature: 0) }

func collect(_ s: GenerationStream) async throws -> [Token] { var t: [Token] = []; for try await x in s { t.append(x) }; return t }

/// Level 4/5 through the public API: streaming, determinism, cancellation, isolation, budgets and failures.
final class PublicAPITests: XCTestCase {
    override func setUpWithError() throws { try requireOptimizedBuild() }

    func load(_ quant: String = "Q8_0", backend: Backend = .metal, _ tweak: (inout LoadOptions) -> Void = { _ in }) async throws -> LanguageModel {
        try XCTSkipUnless(backend == .cpu || LanguageModelTestSupport.metalAvailable, "no Metal device")
        var o = LoadOptions(); o.backend = backend; o.contextLength = 512; tweak(&o)
        return try await LanguageModel.load(from: ModelFiles.url(quant), options: o)
    }

    func testGreedyStreamMatchesReferenceOnBothBackends() async throws {
        let ref = try Reference.load("Q8_0")[0]
        for backend in [Backend.metal, .cpu] {
            let model = try await load(backend: backend)
            let stream = model.generate(prompt: ref.prompt, configuration: greedy(24))
            let tokens = try await collect(stream)
            XCTAssertEqual(tokens.map(\.id), ref.generated, "\(backend)")
            XCTAssertEqual(tokens.map(\.text).joined(), try model.detokenize(ref.generated))
            let s = stream.summary
            print("[api] \(backend): prefill \(s.promptTokens) tok @ \(Int(s.prefillTokensPerSecond)) tok/s, decode \(s.generatedTokens) tok @ \(Int(s.decodeTokensPerSecond)) tok/s, ttft \(Int(s.timeToFirstToken * 1000)) ms")
            XCTAssertEqual(s.finishReason, .maxTokens)
            XCTAssertEqual(s.generatedTokens, 24)
        }
    }

    func testSeededSamplingIsDeterministicAndSeedsDiffer() async throws {
        let model = try await load()
        func run(_ seed: UInt64) async throws -> [Int32] {
            let c = GenerationConfiguration(maxTokens: 30, temperature: 1.0, topK: 40, topP: 0.95, seed: seed)
            return try await collect(model.generate(prompt: "Once upon a time", configuration: c)).map(\.id)
        }
        let a = try await run(1), b = try await run(1), c = try await run(2)
        XCTAssertEqual(a, b)
        XCTAssertNotEqual(a, c)
    }

    func testStopsAtEndOfSequence() async throws {
        let model = try await load()
        let prompt = "<|im_start|>user\nSay hi.<|im_end|>\n<|im_start|>assistant\n"
        let stream = model.generate(prompt: prompt, configuration: greedy(200))
        let tokens = try await collect(stream)
        XCTAssertEqual(stream.summary.finishReason, .endOfSequence)
        XCTAssertLessThan(tokens.count, 200)
        XCTAssertFalse(tokens.contains { $0.id == 2 }, "EOS must not be emitted")
    }

    func testCancellationStopsGenerationAndReleasesSession() async throws {
        let model = try await load()
        let stream = model.generate(prompt: "Count forever:", configuration: greedy(100_000))
        var n = 0
        for try await _ in stream { n += 1; if n == 5 { stream.cancel() } }
        XCTAssertEqual(stream.summary.finishReason, .cancelled)
        XCTAssertLessThan(stream.summary.generatedTokens, 200)
        try await waitUntilIdle(model)

        // Cancelling the consuming Task (structured cancellation) must also stop the producer.
        let consumer = Task {
            let s = model.generate(prompt: "Count forever:", configuration: greedy(100_000))
            for try await _ in s { if Task.isCancelled { break } }
            return s
        }
        try await Task.sleep(nanoseconds: 200_000_000)
        consumer.cancel()
        _ = try? await consumer.value
        try await waitUntilIdle(model)

        let again = try await collect(model.generate(prompt: "Hello", configuration: greedy(5)))
        XCTAssertEqual(again.count, 5, "model remains usable after cancellations")
    }

    func waitUntilIdle(_ model: LanguageModel) async throws {
        for _ in 0..<100 { if model.activeGenerationCount == 0 { return }; try await Task.sleep(nanoseconds: 50_000_000) }
        XCTFail("generation did not terminate after cancellation")
    }

    func testConcurrentGenerationsAreIsolated() async throws {
        let model = try await load()
        let prompts = ["The capital of France is", "def fibonacci(n):", "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n", "1, 2, 3, 4,"]
        var solo: [[Int32]] = []
        for p in prompts { solo.append(try await collect(model.generate(prompt: p, configuration: greedy(40))).map(\.id)) }
        let together = try await withThrowingTaskGroup(of: (Int, [Int32]).self) { group in
            for (i, p) in prompts.enumerated() { group.addTask { (i, try await collect(model.generate(prompt: p, configuration: greedy(40))).map(\.id)) } }
            var out = [[Int32]](repeating: [], count: prompts.count)
            for try await (i, ids) in group { out[i] = ids }
            return out
        }
        XCTAssertEqual(together, solo, "concurrent greedy generations must equal their solo runs")
    }

    func testContextLimits() async throws {
        let model = try await load { $0.contextLength = 16 }
        let long = String(repeating: "word ", count: 40)
        do { _ = try await collect(model.generate(prompt: long)); XCTFail("expected contextExhausted") }
        catch AlpacaError.contextExhausted(let p, let c) { XCTAssertGreaterThan(p, c); XCTAssertEqual(c, 16) }
        let stream = model.generate(prompt: "Once upon a time there was", configuration: greedy(1000))
        let tokens = try await collect(stream)
        XCTAssertEqual(stream.summary.finishReason, .contextFull)
        // All 16 positions get cached; the logits after the last cached position still yield one more token (never fed back).
        XCTAssertEqual(stream.summary.promptTokens + tokens.count, 16 + 1)
    }

    func testInvalidRequestsFailCleanly() async throws {
        let model = try await load()
        do { _ = try await collect(model.generate(prompt: "")); XCTFail() } catch AlpacaError.invalidConfiguration { }
        do { _ = try await collect(model.generate(prompt: "x", configuration: GenerationConfiguration(temperature: -1))); XCTFail() } catch AlpacaError.invalidConfiguration { }
        do { _ = try await collect(model.generate(prompt: "x", configuration: GenerationConfiguration(maxTokens: -1))); XCTFail() } catch AlpacaError.invalidConfiguration { }
        let zero = model.generate(prompt: "hello", configuration: GenerationConfiguration(maxTokens: 0))
        let none = try await collect(zero)
        XCTAssertTrue(none.isEmpty)
    }

    func testMemoryBudgetRefusesBeforeAllocating() async throws {
        do { _ = try await load { $0.memoryBudgetBytes = 64 << 20 }; XCTFail("expected insufficientMemory") }
        catch AlpacaError.insufficientMemory(let need, let budget, let detail) {
            XCTAssertGreaterThan(need, budget); XCTAssertTrue(detail.contains("weights"))
        }
        do { _ = try await load { $0.contextLength = 1_000_000 }; XCTFail() } catch AlpacaError.invalidConfiguration { }
    }

    func testUnloadAndLifecycle() async throws {
        let model = try await load()
        let running = model.generate(prompt: "Count forever:", configuration: greedy(100_000))
        var it = running.makeAsyncIterator()
        _ = try await it.next()
        model.unload()
        while try await it.next() != nil { }
        XCTAssertEqual(running.summary.finishReason, .cancelled)
        do { _ = try await collect(model.generate(prompt: "hi")); XCTFail() } catch AlpacaError.modelUnloaded { }
    }

    func testRepeatedLoadingDoesNotLeak() async throws {
        func rss() -> Int { var info = mach_task_basic_info(); var c = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size / 4)
            _ = withUnsafeMutablePointer(to: &info) { $0.withMemoryRebound(to: integer_t.self, capacity: 1) { task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &c) } }
            return Int(info.resident_size) }
        var sizes: [Int] = []
        for _ in 0..<6 {
            let m = try await load()
            _ = try await collect(m.generate(prompt: "Hello", configuration: greedy(3)))
            m.unload()
            sizes.append(rss())
        }
        print("[api] resident size after each of 6 load/generate/unload cycles (MiB): \(sizes.map { $0 >> 20 })")
        XCTAssertLessThan(sizes.last! - sizes[1], 150 << 20, "resident memory must not grow with repeated loads")
    }

    func testCorruptAndMissingModelsReportErrors() async throws {
        let src = try ModelFiles.url("Q8_0")
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("alpaca-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let data = try Data(contentsOf: src, options: .mappedIfSafe)
        for (name, bytes) in [("truncated", data.prefix(data.count / 2)), ("header-only", data.prefix(4096)), ("garbage", Data(repeating: 7, count: 5000))] {
            let url = dir.appendingPathComponent("\(name).gguf")
            try bytes.write(to: url)
            do { _ = try await LanguageModel.load(from: url, options: { var o = LoadOptions(); o.backend = .cpu; return o }()); XCTFail("\(name) loaded") }
            catch AlpacaError.modelLoadFailed { }
        }
        do { _ = try await LanguageModel.load(from: dir.appendingPathComponent("nope.gguf")); XCTFail() } catch AlpacaError.modelLoadFailed { }
    }
}
