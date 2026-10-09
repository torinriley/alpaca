// swift-tools-version: 6.0
// Copyright (c) 2026 Torin Etheridge. MIT License (see LICENSE).
// Author: Torin Etheridge · 2026-10-09
import PackageDescription

let package = Package(
    name: "alpaca.swift",
    platforms: [.macOS(.v14), .iOS(.v17)],
    products: [
        .library(name: "Alpaca", targets: ["Alpaca"]),
        .executable(name: "alpaca", targets: ["AlpacaCLI"]),
    ],
    targets: [
        // Tensors, CPU reference operations, KV cache, Llama forward pass. Foundation + Accelerate-free.
        .target(name: "AlpacaCore"),
        // Metal device/pipeline management and kernels (MSL sources are compiled at load time).
        .target(
            name: "AlpacaMetal",
            dependencies: ["AlpacaCore"],
            resources: [.copy("Kernels")]
        ),
        // GGUF container parsing and Llama weight mapping.
        .target(name: "AlpacaModels", dependencies: ["AlpacaCore", "AlpacaTokenizers"]),
        // Byte-level BPE tokenizer built from GGUF metadata.
        .target(name: "AlpacaTokenizers"),
        // Public API.
        .target(
            name: "Alpaca",
            dependencies: ["AlpacaCore", "AlpacaMetal", "AlpacaModels", "AlpacaTokenizers"]
        ),
        .executableTarget(name: "AlpacaCLI", dependencies: ["Alpaca"]),

        .testTarget(
            name: "AlpacaCoreTests", dependencies: ["AlpacaCore"],
            resources: [.copy("Fixtures")]),
        .testTarget(name: "AlpacaMetalTests", dependencies: ["AlpacaMetal", "AlpacaCore"]),
        .testTarget(name: "AlpacaModelTests", dependencies: ["AlpacaModels", "AlpacaCore"]),
        .testTarget(
            name: "AlpacaTokenizerTests", dependencies: ["AlpacaTokenizers", "AlpacaModels"],
            resources: [.copy("Fixtures")]),
        .testTarget(
            name: "AlpacaIntegrationTests",
            dependencies: ["Alpaca", "AlpacaCore", "AlpacaModels", "AlpacaTokenizers", "AlpacaMetal"],
            resources: [.copy("Fixtures")]),
    ],
    swiftLanguageModes: [.v6]
)
