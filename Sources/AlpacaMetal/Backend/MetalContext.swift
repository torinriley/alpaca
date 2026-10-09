// Copyright (c) 2026 Torin Etheridge. MIT License (see LICENSE).
// Author: Torin Etheridge · 2026-10-09

import Foundation
import Metal

public enum MetalError: Error, CustomStringConvertible {
    case noDevice
    case shaderSourceMissing(String)
    case shaderCompilation(String)
    case missingFunction(String)
    case pipelineCreation(String, String)
    case bufferAllocation(bytes: Int)
    case commandFailed(String)
    case unsupported(String)
    case invalidArgument(String)

    public var description: String {
        switch self {
        case .noDevice: return "no Metal device is available"
        case .shaderSourceMissing(let m): return "Metal shader sources not found: \(m)"
        case .shaderCompilation(let m): return "Metal shader compilation failed: \(m)"
        case .missingFunction(let n): return "Metal function '\(n)' not found in library"
        case .pipelineCreation(let n, let m): return "could not create pipeline '\(n)': \(m)"
        case .bufferAllocation(let b): return "failed to allocate a \(b)-byte Metal buffer"
        case .commandFailed(let m): return "Metal command buffer failed: \(m)"
        case .unsupported(let m): return "unsupported on the Metal backend: \(m)"
        case .invalidArgument(let m): return "invalid argument: \(m)"
        }
    }
}

/// Owns the Metal device, command queue, compiled shader library and pipeline cache.
///
/// Shaders ship as `.metal` source files (resource directory `Kernels`) and are compiled at first use with
/// `makeLibrary(source:)`, which works identically under `swift build`, Xcode and on iOS without a build plugin.
public final class MetalContext: @unchecked Sendable {
    public let device: MTLDevice
    public let queue: MTLCommandQueue
    let library: MTLLibrary
    /// Metal 4 tensor-op kernels (matrix-hardware GEMM). nil when the device family or OS cannot compile them.
    let tensorLibrary: MTLLibrary?
    private var pipelines: [String: MTLComputePipelineState] = [:]
    private let lock = NSLock()

    public static var isAvailable: Bool { MTLCreateSystemDefaultDevice() != nil }

    public init(device: MTLDevice? = nil) throws {
        guard let device = device ?? MTLCreateSystemDefaultDevice() else { throw MetalError.noDevice }
        guard let queue = device.makeCommandQueue() else { throw MetalError.commandFailed("could not create command queue") }
        self.device = device
        self.queue = queue
        self.library = try Self.compileLibrary(device: device)
        self.tensorLibrary = Self.compileTensorLibrary(device: device)
    }

    public var deviceName: String { device.name }

    static func compileLibrary(device: MTLDevice) throws -> MTLLibrary {
        guard let dir = Bundle.module.url(forResource: "Kernels", withExtension: nil) else {
            throw MetalError.shaderSourceMissing("Kernels resource directory")
        }
        let files = try FileManager.default.contentsOfDirectory(atPath: dir.path).filter { $0.hasSuffix(".metal") }.sorted()
        guard files.contains("common.metal") else { throw MetalError.shaderSourceMissing("common.metal") }
        // makeLibrary(source:) has no include search path: inline `common.metal` once and strip local #includes.
        var source = ""
        // tensor_*.metal need language version 4.0 and MetalPerformancePrimitives: compiled separately (compileTensorLibrary).
        for name in ["common.metal"] + files.filter({ $0 != "common.metal" && !$0.hasPrefix("tensor_") }) {
            let text = try String(contentsOf: dir.appendingPathComponent(name), encoding: .utf8)
            source += "\n// ---- \(name) ----\n"
            source += text.split(separator: "\n", omittingEmptySubsequences: false)
                .filter { !$0.hasPrefix("#include \"") }
                .joined(separator: "\n")
        }
        let options = MTLCompileOptions()
        options.languageVersion = .version3_1
        do { return try device.makeLibrary(source: source, options: options) }
        catch { throw MetalError.shaderCompilation("\(error)") }
    }

    /// Compiles `tensor_*.metal` with Metal language 4.0 on GPUs with matrix-multiply hardware in the shader cores
    /// (Apple10 family: M5 / A19). Returns nil elsewhere or if the OS toolchain cannot compile them; callers fall back
    /// to the simdgroup-matrix kernels. Set ALPACA_DISABLE_TENSOR_OPS=1 to force the fallback.
    static func compileTensorLibrary(device: MTLDevice) -> MTLLibrary? {
        guard ProcessInfo.processInfo.environment["ALPACA_DISABLE_TENSOR_OPS"] == nil,
              device.supportsFamily(.apple10), #available(macOS 26.0, iOS 26.0, *),
              let dir = Bundle.module.url(forResource: "Kernels", withExtension: nil),
              let files = try? FileManager.default.contentsOfDirectory(atPath: dir.path).filter({ $0.hasPrefix("tensor_") && $0.hasSuffix(".metal") }).sorted(),
              !files.isEmpty
        else { return nil }
        let source = files.compactMap { try? String(contentsOf: dir.appendingPathComponent($0), encoding: .utf8) }.joined(separator: "\n")
        let options = MTLCompileOptions()
        options.languageVersion = .version4_0
        return try? device.makeLibrary(source: source, options: options)
    }

    public var supportsTensorGEMM: Bool { tensorLibrary != nil }

    public func pipeline(_ name: String) throws -> MTLComputePipelineState {
        lock.lock(); defer { lock.unlock() }
        if let p = pipelines[name] { return p }
        guard let fn = library.makeFunction(name: name) ?? tensorLibrary?.makeFunction(name: name) else { throw MetalError.missingFunction(name) }
        do {
            let p = try device.makeComputePipelineState(function: fn)
            pipelines[name] = p
            return p
        } catch { throw MetalError.pipelineCreation(name, "\(error)") }
    }

    public func makeBuffer(length: Int) throws -> MTLBuffer {
        guard let b = device.makeBuffer(length: max(length, 16), options: .storageModeShared) else {
            throw MetalError.bufferAllocation(bytes: length)
        }
        return b
    }

    public func makeBuffer(copying pointer: UnsafeRawPointer, length: Int) throws -> MTLBuffer {
        guard let b = device.makeBuffer(bytes: pointer, length: max(length, 16), options: .storageModeShared) else {
            throw MetalError.bufferAllocation(bytes: length)
        }
        return b
    }
}
