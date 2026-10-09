// Copyright (c) 2026 Torin Etheridge. MIT License (see LICENSE).
// Author: Torin Etheridge · 2026-10-09

import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @State private var vm = DemoViewModel()
    @State private var picking = false
    @Environment(\.scenePhase) private var scenePhase

    var body: some View {
        NavigationStack {
            VStack(alignment: .leading, spacing: 12) {
                HStack {
                    Button("Select model…") { picking = true }
                    Text(vm.modelName).font(.footnote).lineLimit(1).truncationMode(.middle)
                    if vm.isLoading { ProgressView() }
                }
                TextField("Prompt", text: $vm.prompt, axis: .vertical).textFieldStyle(.roundedBorder).lineLimit(1...4)
                HStack {
                    Button("Generate") { vm.generate() }.disabled(!vm.canGenerate).buttonStyle(.borderedProminent)
                    Button("Stop") { vm.cancel() }.disabled(!vm.isGenerating)
                }
                ScrollView { Text(vm.output).frame(maxWidth: .infinity, alignment: .leading).textSelection(.enabled) }
                    .frame(maxHeight: .infinity)
                Divider()
                VStack(alignment: .leading, spacing: 2) {
                    Text(vm.status).font(.footnote)
                    Text(String(format: "prefill %.0f tok/s · decode %.0f tok/s", vm.prefillRate, vm.decodeRate)).font(.caption.monospacedDigit())
                    Text("resident \(vm.residentMiB) MiB · available to this process \(vm.availableMiB) MiB").font(.caption.monospacedDigit())
                }.foregroundStyle(.secondary)
            }
            .padding()
            .navigationTitle("alpaca.swift")
            .fileImporter(isPresented: $picking, allowedContentTypes: [UTType(filenameExtension: "gguf") ?? .data]) { result in
                if case .success(let url) = result { vm.select(url) }
            }
            .onChange(of: scenePhase) { _, phase in if phase != .active { vm.appMovedToBackground() } }
        }
    }
}
