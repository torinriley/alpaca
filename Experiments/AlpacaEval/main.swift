// Offline evaluation harness for the paper: paired next-token scoring of one model under several GPU precision configurations.
// For each sampled position p the window tokens[p-L..<p] is prefilled (so the *prefill* kernels produce the logits) and the
// logits of the last window token score tokens[p]. Usage:
//   alpaca-eval <model.gguf> <text.txt> --positions 3000 --window 256 --configs ref,gemm,kv16,default --out out.json
import Foundation
import AlpacaCore
import AlpacaMetal
import AlpacaModels
import AlpacaTokenizers

struct Cfg { let name: String; let kv: KVPrecision; let gemm: GEMMPrecision }
let all: [String: Cfg] = [
    "ref": Cfg(name: "ref", kv: .float32, gemm: .exact),        // most precise available configuration
    "gemm": Cfg(name: "gemm", kv: .float32, gemm: .fast),       // only the half-precision prefill GEMM
    "kv16": Cfg(name: "kv16", kv: .float16, gemm: .exact),      // only the f16 KV cache
    "default": Cfg(name: "default", kv: .float16, gemm: .fast), // shipped default
]
let args = Array(CommandLine.arguments.dropFirst())
func opt(_ k: String, _ d: String) -> String { if let i = args.firstIndex(of: "--\(k)"), i + 1 < args.count { return args[i + 1] }; return d }
guard args.count >= 2 else { print("usage: alpaca-eval model.gguf text.txt [--positions N] [--window L] [--configs a,b] [--out f.json]"); exit(2) }
let positions = Int(opt("positions", "3000"))!, window = Int(opt("window", "256"))!
let names = opt("configs", "ref,gemm,kv16,default").split(separator: ",").map(String.init)
let cfgs = names.map { all[$0]! }
let out = opt("out", "eval.json")

let loaded = try LlamaLoader.load(url: URL(fileURLWithPath: args[0]))
let tok = try loaded.file.makeTokenizer()
let text = try String(contentsOfFile: args[1], encoding: .utf8)
let t0 = Date()
let ids = try tok.encode(text, addBOS: false, parseSpecial: false)
print("tokens: \(ids.count) (tokenised in \(String(format: "%.1f", Date().timeIntervalSince(t0))) s)")
let ctx = try MetalContext()
let model = try MetalLlamaModel(context: ctx, config: loaded.config, weights: loaded.weights,
                                mappedRegion: loaded.file.mappedMemory, keepAlive: loaded.file.mappingOwner)
let sessions = try cfgs.map { try model.makeSession(capacity: window, maxBatch: 512, kvPrecision: $0.kv, gemmPrecision: $0.gemm) }
let V = loaded.config.vocabSize
let span = ids.count - 1 - window
let ps = (0..<positions).map { window + $0 * span / positions }

func logSoftmax(_ x: [Float]) -> [Float] {
    var m = -Float.infinity; for v in x { m = max(m, v) }
    var s = 0.0; for v in x { s += exp(Double(v - m)) }
    let lse = m + Float(log(s)); return x.map { $0 - lse }
}
var nll = [[Double]](repeating: [], count: cfgs.count), top1 = [[Int]](repeating: [], count: cfgs.count), kl = [[Double]](repeating: [], count: cfgs.count)
var ms = [Double](repeating: 0, count: cfgs.count)
let start = Date()
for (n, p) in ps.enumerated() {
    let w = Array(ids[(p - window)..<p]), target = Int(ids[p])
    var refLP: [Float] = []
    for (i, s) in sessions.enumerated() {
        s.reset()
        let t = Date()
        let logits = try s.forward(tokens: w)
        ms[i] += Date().timeIntervalSince(t) * 1000
        let lp = logSoftmax(logits)
        nll[i].append(-Double(lp[target]))
        top1[i].append(lp.indices.max { lp[$0] < lp[$1] }!)
        if i == 0 { refLP = lp; kl[i].append(0) } else {
            var k = 0.0; for j in 0..<V { let pr = exp(Double(refLP[j])); if pr > 1e-12 { k += pr * Double(refLP[j] - lp[j]) } }
            kl[i].append(k)
        }
    }
    if n % 500 == 0 { print("  \(n)/\(positions)  \(String(format: "%.0f", Date().timeIntervalSince(start))) s") }
}
var result: [String: Any] = ["model": args[0], "window": window, "positions": positions, "vocab": V, "tokens": ids.count, "device": ctx.deviceName]
for (i, c) in cfgs.enumerated() {
    result[c.name] = ["nll": nll[i], "top1": top1[i], "kl": kl[i], "prefillMsPerWindow": ms[i] / Double(positions)] as [String: Any]
    print("\(c.name): ppl \(String(format: "%.3f", exp(nll[i].reduce(0, +) / Double(positions)))) meanKL \(String(format: "%.2e", kl[i].reduce(0, +) / Double(positions)))")
}
try JSONSerialization.data(withJSONObject: result).write(to: URL(fileURLWithPath: out))
