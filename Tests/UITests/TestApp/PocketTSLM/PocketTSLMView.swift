//
// This source file is part of the PocketTSLM open-source project
//
// SPDX-FileCopyrightText: 2026 Schmiedmayer Lab and the project authors (see CONTRIBUTORS.md)
//
// SPDX-License-Identifier: MIT
//

import SpeziChat
import SpeziHealthKit
import SwiftUI
import OSLog

/// Chat-style front end for PocketTSLM. On launch it loads the base model through MLX; once
/// ready, the task sheet offers the sample runs and, under *Evaluations and benchmarks*, the
/// paper's evaluations and benchmarks. Tapping one runs the operation on-device and appends
/// the constructed prompt as the sender bubble and the model's generated answer as the reply bubble.
struct PocketTSLMView: View {
    private static let logger = Logger(subsystem: "PocketTSLM", category: "PocketTSLMView")

    @Environment(HealthDataInterpreter.self) private var interpreter
    @Environment(HealthKit.self) private var healthKit

    @State private var chat: Chat = []
    @State private var didSetup = false
    @State private var isRunning = false
    @State private var errorMessage = ""
    @State private var actionSheetDetent: PresentationDetent = Self.collapsedDetent
    @State private var showsEvaluations = false

    private static let collapsedDetent = PresentationDetent.height(72)

    var body: some View {
        NavigationStack {
            Group {
                if interpreter.loaded {
                    chatInterface
                } else {
                    loadingView
                }
            }
            .navigationTitle("PocketTSLM")
            .toolbar {
                if interpreter.loaded && !chat.isEmpty {
                    ToolbarItem(placement: .topBarTrailing) {
                        Button { chat.removeAll() } label: {
                            Image(systemName: "arrow.counterclockwise")
                        }
                        .disabled(isRunning)
                        .accessibilityLabel(Text("Clear"))
                    }
                }
            }
        }
        .task {
            guard !didSetup else { return }
            didSetup = true
            if ProcessInfo.processInfo.environment["XCODE_RUNNING_FOR_PREVIEWS"] == "1" { return }
            do {
                try await interpreter.setup()
            } catch {
                Self.logger.error("setup() failed: \(error.localizedDescription, privacy: .public)")
                errorMessage = error.localizedDescription
            }
        }
    }

    private var chatInterface: some View {
        VStack(spacing: 0) {
            if chat.isEmpty {
                emptyState
            } else {
                MessagesView(chat, hideMessages: .custom(hiddenMessageTypes: []))
            }
        }
        // Keep the last message clear of the collapsed action sheet.
        .safeAreaPadding(.bottom, 72)
        .sheet(isPresented: .constant(true)) {
            actionSheet
                .presentationDetents([Self.collapsedDetent, .medium, .large], selection: $actionSheetDetent)
                .presentationDragIndicator(.visible)
                .presentationBackgroundInteraction(.enabled(upThrough: .medium))
                .presentationContentInteraction(.scrolls)
                .interactiveDismissDisabled()
        }
    }

    private var emptyState: some View {
        VStack(spacing: 10) {
            Spacer()
            Image(systemName: "waveform.path.ecg")
                .font(.system(size: 44))
                .foregroundStyle(.tint)
            Text("On-device time-series LLM")
                .font(.headline)
            Text("Open Tasks below to run one on this device.")
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
            Spacer()
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding()
    }

    private var actionSheet: some View {
        ScrollView {
            VStack(spacing: 12) {
                actionSheetHeader
                actionButtons
            }
            .padding(.horizontal)
            .padding(.top, 20)
        }
        .scrollBounceBehavior(.basedOnSize)
    }

    /// Visible in the collapsed detent; tapping it expands the sheet.
    private var actionSheetHeader: some View {
        Button {
            withAnimation {
                actionSheetDetent = actionSheetDetent == Self.collapsedDetent ? .medium : Self.collapsedDetent
            }
        } label: {
            HStack {
                Text("Tasks").font(.headline)
                Spacer()
                if isRunning {
                    ProgressView().controlSize(.small)
                    Text("Running on-device…").font(.caption).foregroundStyle(.secondary)
                } else {
                    Image(systemName: actionSheetDetent == Self.collapsedDetent ? "chevron.up" : "chevron.down")
                        .foregroundStyle(.secondary)
                }
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }

    private var actionButtons: some View {
        VStack(spacing: 8) {
            HStack(spacing: 8) {
                actionButton("EEG sample", systemImage: "brain.head.profile") { await interpreter.eegSample() }
                actionButton("ECG sample", systemImage: "heart.text.square") { await interpreter.ecgSample() }
            }
            HStack(spacing: 8) {
                actionButton("My Watch ECG", systemImage: "applewatch") { await interpreter.ecgFromHealthKit(healthKit: healthKit) }
            }
            // The paper's evaluations and benchmarks, which write JSON reports to Documents.
            DisclosureGroup("Evaluations and benchmarks", isExpanded: $showsEvaluations) {
                VStack(spacing: 8) {
                    HStack(spacing: 8) {
                        actionButton("EEG eval (20)", systemImage: "checklist") { await interpreter.eegAccuracyEval() }
                        actionButton("ECG eval (20)", systemImage: "checklist.checked") { await interpreter.ecgAccuracyEval() }
                    }
                    HStack(spacing: 8) {
                        actionButton("EEG bench", systemImage: "chart.bar") { await interpreter.eegBenchmark() }
                        actionButton("ECG bench", systemImage: "chart.bar.doc.horizontal") { await interpreter.ecgBenchmark() }
                    }
                    HStack(spacing: 8) {
                        actionButton("EEG var (5×)", systemImage: "chart.line.uptrend.xyaxis") { await interpreter.eegRepeatedBenchmark() }
                        actionButton("ECG var (5×)", systemImage: "chart.line.uptrend.xyaxis") { await interpreter.ecgRepeatedBenchmark() }
                    }
                }
                .padding(.top, 8)
            }
            .padding(.top, 4)
        }
    }

    private func actionButton(
        _ title: String,
        systemImage: String,
        _ operation: @escaping () async -> ChatTurn
    ) -> some View {
        Button {
            run(operation)
        } label: {
            Label(title, systemImage: systemImage)
                .frame(maxWidth: .infinity)
        }
        .buttonStyle(.borderedProminent)
        .disabled(isRunning)
    }

    private func run(_ operation: @escaping () async -> ChatTurn) {
        isRunning = true
        // Collapse so the chat stays visible while the task runs.
        withAnimation { actionSheetDetent = Self.collapsedDetent }
        Task {
            let turn = await operation()
            await MainActor.run {
                chat.append(ChatEntity(role: .user, content: turn.prompt))
                chat.append(ChatEntity(role: .assistant, content: turn.response))
                isRunning = false
            }
        }
    }

    private var loadingView: some View {
        VStack(spacing: 12) {
            Spacer()
            if !errorMessage.isEmpty {
                Image(systemName: "exclamationmark.triangle")
                    .font(.largeTitle)
                    .foregroundStyle(.orange)
                Text("Setup failed").font(.headline)
                Text(errorMessage)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal)
            } else {
                ProgressView()
                Text(interpreter.loadingStage.rawValue).font(.subheadline)
                if !interpreter.loadingDetail.isEmpty {
                    Text(interpreter.loadingDetail)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .multilineTextAlignment(.center)
                        .padding(.horizontal)
                }
            }
            Spacer()
        }
    }
}
