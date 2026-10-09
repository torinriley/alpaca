// Copyright (c) 2026 Torin Etheridge. MIT License (see LICENSE).
// Author: Torin Etheridge · 2026-10-09

import Foundation
import Alpaca

// alpaca — command line front end.
//   alpaca info     <model.gguf>
//   alpaca generate <model.gguf> --prompt "…" [--chat] [--max-tokens N] [--temperature T] [--top-k K] [--top-p P] [--seed S] [--backend metal|cpu] [--context N] [--stats]
//   alpaca bench    <model.gguf> [--backend metal|cpu] [--prompt-lengths 16,128,512] [--decode 64] [--runs 5] [--context-sweep 16,512,2048] [--json out.json]

struct CLIError: Error, CustomStringConvertible { let description: String }

struct Args {
    var positional: [String] = []
    var options: [String: String] = [:]
    var flags: Set<String> = []
    init(_ argv: [String], valueOptions: Set<String>) throws {
        var i = 0
        while i < argv.count {
            let a = argv[i]
            if a.hasPrefix("--") {
                let key = String(a.dropFirst(2))
                if valueOptions.contains(key) {
                    guard i + 1 < argv.count else { throw CLIError(description: "--\(key) needs a value") }
                    options[key] = argv[i + 1]; i += 2; continue
                }
                flags.insert(key)
            } else { positional.append(a) }
            i += 1
        }
    }
    func int(_ k: String) throws -> Int? { try options[k].map { guard let v = Int($0) else { throw CLIError(description: "--\(k) expects an integer") }; return v } }
    func float(_ k: String) throws -> Float? { try options[k].map { guard let v = Float($0) else { throw CLIError(description: "--\(k) expects a number") }; return v } }
    func intList(_ k: String, default d: [Int]) throws -> [Int] {
        guard let s = options[k] else { return d }
        return try s.split(separator: ",").map { guard let v = Int($0) else { throw CLIError(description: "--\(k) expects comma-separated integers") }; return v }
    }
}

func backend(_ a: Args) throws -> Backend {
    switch a.options["backend"] ?? "automatic" {
    case "metal": return .metal
    case "cpu": return .cpu
    case "automatic", "auto": return .automatic
    default: throw CLIError(description: "--backend must be metal, cpu or automatic")
    }
}

func modelURL(_ a: Args) throws -> URL {
    guard let p = a.positional.dropFirst().first else { throw CLIError(description: "missing model path") }
    return URL(fileURLWithPath: p)
}

func loadOptions(_ a: Args) throws -> LoadOptions {
    var o = LoadOptions()
    o.backend = try backend(a)
    o.contextLength = try a.int("context")
    if a.options["kv"] == "f32" { o.kvPrecision = .float32 }
    if a.flags.contains("exact-gemm") { o.prefillPrecision = .exact }
    if let b = try a.int("prefill-batch") { o.prefillBatchSize = b }
    if let mb = try a.int("memory-budget-mb") { o.memoryBudgetBytes = mb << 20 }
    return o
}

func sysctlString(_ name: String) -> String {
    var size = 0
    sysctlbyname(name, nil, &size, nil, 0)
    guard size > 0 else { return "unknown" }
    var buf = [CChar](repeating: 0, count: size)
    sysctlbyname(name, &buf, &size, nil, 0)
    return String(decoding: buf.prefix(while: { $0 != 0 }).map { UInt8(bitPattern: $0) }, as: UTF8.self)
}

func peakRSSBytes() -> Int { var u = rusage(); getrusage(RUSAGE_SELF, &u); return Int(u.ru_maxrss) }

/// Peak and current physical footprint (dirty + compressed + GPU-wired memory the kernel charges to this process):
/// the number iOS jetsam looks at. Unlike RSS it includes Metal buffers, and file-backed mapped weights are not double counted.
func physicalFootprint() -> (current: Int, peak: Int) {
    var info = rusage_info_v4()
    let rc = withUnsafeMutablePointer(to: &info) { $0.withMemoryRebound(to: rusage_info_t?.self, capacity: 1) { proc_pid_rusage(getpid(), RUSAGE_INFO_V4, $0) } }
    return rc == 0 ? (Int(info.ri_phys_footprint), Int(info.ri_lifetime_max_phys_footprint)) : (0, 0)
}
func cpuSeconds() -> Double {
    var u = rusage(); getrusage(RUSAGE_SELF, &u)
    return Double(u.ru_utime.tv_sec + u.ru_stime.tv_sec) + Double(u.ru_utime.tv_usec + u.ru_stime.tv_usec) / 1e6
}
func thermalName() -> String {
    switch ProcessInfo.processInfo.thermalState { case .nominal: "nominal"; case .fair: "fair"; case .serious: "serious"; case .critical: "critical"; @unknown default: "unknown" }
}

func stats(_ xs: [Double]) -> (median: Double, min: Double, max: Double, stdev: Double) {
    guard !xs.isEmpty else { return (0, 0, 0, 0) }
    let s = xs.sorted(), mean = xs.reduce(0, +) / Double(xs.count)
    let med = s.count % 2 == 1 ? s[s.count / 2] : (s[s.count / 2 - 1] + s[s.count / 2]) / 2
    let var_ = xs.count > 1 ? xs.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Double(xs.count - 1) : 0
    return (med, s.first!, s.last!, var_.squareRoot())
}

func runInfo(_ a: Args) async throws {
    let url = try modelURL(a)
    let m = try await LanguageModel.load(from: url, options: try loadOptions(a))
    let i = m.info
    print("name:            \(i.name)\narchitecture:    \(i.architecture)\nlayers:          \(i.layerCount)\nhidden size:     \(i.hiddenSize)\nvocabulary:      \(i.vocabularySize)")
    print("context:         \(i.contextLength) (trained \(i.trainedContextLength))\nbackend:         \(i.backend)\nweights:         \(i.parameterBytes >> 20) MiB\nestimated memory: \(i.estimatedMemory)")
    for n in i.loadNotes { print("note: \(n)") }
}

func runGenerate(_ a: Args) async throws {
    let url = try modelURL(a)
    guard var prompt = a.options["prompt"] else { throw CLIError(description: "--prompt is required") }
    if a.flags.contains("chat") { prompt = "<|im_start|>user\n\(prompt)<|im_end|>\n<|im_start|>assistant\n" }
    let model = try await LanguageModel.load(from: url, options: try loadOptions(a))
    var c = GenerationConfiguration(maxTokens: try a.int("max-tokens") ?? 256, temperature: try a.float("temperature") ?? 0.8)
    c.topK = try a.int("top-k") ?? 0
    c.topP = try a.float("top-p") ?? 1
    c.seed = try a.int("seed").map { UInt64($0) }
    let stream = model.generate(prompt: prompt, configuration: c)
    let task = Task { for try await t in stream { FileHandle.standardOutput.write(Data(t.text.utf8)) } }
    signal(SIGINT, SIG_IGN)
    let src = DispatchSource.makeSignalSource(signal: SIGINT, queue: .main)
    src.setEventHandler { stream.cancel() }
    src.resume()
    try await task.value
    print()
    if a.flags.contains("stats") {
        let s = stream.summary
        FileHandle.standardError.write(Data(String(format: "[%@] prompt %d tok, %d generated (%@) | prefill %.1f tok/s | decode %.1f tok/s | ttft %.0f ms\n",
            "\(model.info.backend)", s.promptTokens, s.generatedTokens, "\(s.finishReason)", s.prefillTokensPerSecond, s.decodeTokensPerSecond, s.timeToFirstToken * 1000).utf8))
    }
}

func runProfile(_ a: Args) async throws {
    let url = try modelURL(a)
    var options = try loadOptions(a)
    let tokens = try a.int("tokens") ?? 1, ctxBefore = try a.int("context-before") ?? 0
    options.contextLength = max(options.contextLength ?? 0, tokens + ctxBefore + 8)
    let model = try await LanguageModel.load(from: url, options: options)
    let p = try model.profileForward(newTokens: tokens, contextBefore: ctxBefore, repetitions: try a.int("runs") ?? 5)
    print("profile: \(tokens) new tokens after \(ctxBefore) cached | unprofiled GPU \(String(format: "%.2f", p.unprofiledGPUMilliseconds)) ms, encode (CPU) \(String(format: "%.2f", p.encodeMilliseconds)) ms, wall \(String(format: "%.2f", p.wallMilliseconds)) ms")
    print("per-stage GPU time (separate command buffers; sum \(String(format: "%.2f", p.profiledGPUMilliseconds)) ms):")
    for s in p.stages { print(String(format: "  %-22@ %8.3f ms  %5.1f%%", s.name as NSString, s.milliseconds, s.share * 100)) }
}

func runBench(_ a: Args) async throws {
    let url = try modelURL(a)
    let runs = try a.int("runs") ?? 5, decodeN = try a.int("decode") ?? 64
    let promptLengths = try a.intList("prompt-lengths", default: [16, 128, 512])
    let sweep = try a.intList("context-sweep", default: [16, 512, 2048])
    var options = try loadOptions(a)
    options.contextLength = max((promptLengths + sweep).max()! + decodeN + 8, options.contextLength ?? 0)

    var report: [String: Any] = [
        "hardware": ["model": sysctlString("hw.model"), "chip": sysctlString("machdep.cpu.brand_string"), "memoryGiB": Int(ProcessInfo.processInfo.physicalMemory >> 30),
                     "cores": ProcessInfo.processInfo.activeProcessorCount],
        "os": ProcessInfo.processInfo.operatingSystemVersionString,
        "model": url.lastPathComponent, "modelBytes": (try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? Int) ?? 0,
        "backend": "\(options.backend)", "kvPrecision": options.kvPrecision == .float16 ? "f16" : "f32",
        "thermalBefore": thermalName(), "runs": runs, "decodeTokens": decodeN,
    ]

    // Cold start (first load in this process; includes shader compilation and page-cache faults) vs warm start.
    let t0 = Date()
    let model = try await LanguageModel.load(from: url, options: options)
    let cold = Date().timeIntervalSince(t0)
    let t1 = Date()
    let warmModel = try await LanguageModel.load(from: url, options: options)
    let warm = Date().timeIntervalSince(t1)
    warmModel.unload()
    report["loadSecondsCold"] = cold; report["loadSecondsWarm"] = warm
    report["resolvedBackend"] = "\(model.info.backend)"
    print("model \(url.lastPathComponent) | backend \(model.info.backend) | load cold \(String(format: "%.3f", cold)) s, warm \(String(format: "%.3f", warm)) s")

    let corpus = String(repeating: "The quick brown fox jumps over the lazy dog while the transformer predicts the next token. ", count: 400)
    let base = try model.tokenize(corpus)
    func tokens(_ n: Int) -> [Int32] { Array(base.prefix(n)) }
    func measure(prompt n: Int, decode: Int) async throws -> GenerationSummary {
        var c = GenerationConfiguration(maxTokens: decode, temperature: 0)
        c.stopOnEndOfSequence = false
        let s = model.generate(promptTokens: tokens(n), configuration: c)
        for try await _ in s {}
        return s.summary
    }
    // Warm-up run (not reported): primes pipelines and caches.
    _ = try await measure(prompt: 16, decode: 8)

    var rows: [[String: Any]] = []
    print("\nprefill / decode by prompt length (median of \(runs); decode \(decodeN) tokens)")
    print("prompt  prefill tok/s (min–max)        decode tok/s (min–max)         ttft ms   gpu ms/run")
    let cpu0 = cpuSeconds(), wall0 = Date()
    for n in promptLengths + sweep.filter({ !promptLengths.contains($0) }) {
        var pre: [Double] = [], dec: [Double] = [], ttft: [Double] = [], gpu: [Double] = []
        for _ in 0..<runs {
            let s = try await measure(prompt: n, decode: decodeN)
            pre.append(s.prefillTokensPerSecond); dec.append(s.decodeTokensPerSecond); ttft.append(s.timeToFirstToken * 1000); gpu.append(s.gpuSeconds * 1000)
        }
        let p = stats(pre), d = stats(dec), t = stats(ttft), g = stats(gpu)
        print(String(format: "%5d   %9.1f (%.1f–%.1f, sd %.1f)   %9.1f (%.1f–%.1f, sd %.1f)   %7.1f   %7.1f", n, p.median, p.min, p.max, p.stdev, d.median, d.min, d.max, d.stdev, t.median, g.median))
        rows.append(["promptTokens": n, "prefillTokensPerSecond": ["median": p.median, "min": p.min, "max": p.max, "stdev": p.stdev],
                     "decodeTokensPerSecond": ["median": d.median, "min": d.min, "max": d.max, "stdev": d.stdev],
                     "timeToFirstTokenMs": ["median": t.median, "min": t.min, "max": t.max], "gpuMs": ["median": g.median]])
    }
    let wall = Date().timeIntervalSince(wall0).magnitude
    report["results"] = rows
    report["peakResidentMiB"] = peakRSSBytes() >> 20
    report["peakPhysicalFootprintMiB"] = physicalFootprint().peak >> 20
    report["currentPhysicalFootprintMiB"] = physicalFootprint().current >> 20
    report["processCPUUtilizationDuringBench"] = (cpuSeconds() - cpu0) / wall   // CPU-seconds per wall-second (1.0 = one core busy)
    report["thermalAfter"] = thermalName()
    print(String(format: "\npeak footprint %d MiB (resident %d MiB) | process CPU use %.2f cores | thermal %@ → %@", physicalFootprint().peak >> 20, peakRSSBytes() >> 20, (cpuSeconds() - cpu0) / wall, thermalName(), thermalName()))
    if let out = a.options["json"] {
        try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys]).write(to: URL(fileURLWithPath: out))
        print("wrote \(out)")
    }
}

do {
    let argv = Array(CommandLine.arguments.dropFirst())
    guard let cmd = argv.first else { throw CLIError(description: "usage: alpaca <info|generate|bench> <model.gguf> [options]") }
    let a = try Args(argv, valueOptions: ["prompt", "max-tokens", "temperature", "top-k", "top-p", "seed", "backend", "context", "kv", "memory-budget-mb",
                                          "prompt-lengths", "decode", "runs", "context-sweep", "json", "tokens", "context-before", "prefill-batch"])
    switch cmd {
    case "info": try await runInfo(a)
    case "generate": try await runGenerate(a)
    case "bench": try await runBench(a)
    case "profile": try await runProfile(a)
    default: throw CLIError(description: "unknown command '\(cmd)'")
    }
} catch {
    FileHandle.standardError.write(Data("error: \(error)\n".utf8))
    exit(1)
}
