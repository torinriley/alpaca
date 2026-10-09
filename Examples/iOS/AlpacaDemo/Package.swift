// swift-tools-version: 6.0
// Copyright (c) 2026 Torin Etheridge. MIT License (see LICENSE).
// Author: Torin Etheridge · 2026-10-09
import PackageDescription
import AppleProductTypes

// Developer reference app. Open this folder in Xcode, pick an iOS device or simulator, and run.
let package = Package(
    name: "AlpacaDemo",
    platforms: [.iOS(.v17)],
    products: [
        .iOSApplication(
            name: "AlpacaDemo",
            targets: ["AlpacaDemo"],
            bundleIdentifier: "dev.alpaca.demo",
            displayVersion: "0.1",
            bundleVersion: "1",
            supportedDeviceFamilies: [.phone, .pad],
            supportedInterfaceOrientations: [.portrait, .landscapeRight, .landscapeLeft]
        )
    ],
    dependencies: [.package(path: "../../..")],
    targets: [
        .executableTarget(
            name: "AlpacaDemo",
            dependencies: [.product(name: "Alpaca", package: "Alpaca")],
            path: "Sources"
        )
    ]
)
