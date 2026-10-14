//
// This source file is part of the PocketTSLM open-source project
//
// SPDX-FileCopyrightText: 2026 Schmiedmayer Lab and the project authors (see CONTRIBUTORS.md)
//
// SPDX-License-Identifier: MIT
//

import Foundation
import Hub
import OSLog
import PocketTSLM
import Spezi
import MLX
import MLXLMCommon
import MLXLLM
import SpeziHealthKit

@Observable
class HealthDataInterpreter: DefaultInitializable, Module, EnvironmentAccessible {
    @ObservationIgnored private let logger = Logger(subsystem: "PocketTSLM", category: "HealthDataInterpreter")

    enum LoadingStage: String {
        case idle = "Idle"
        case downloadingModel = "Downloading the base model"
        case loadingModel = "Loading the base model"
        case ready = "Ready"
        case failed = "Failed"
    }

    private(set) var loaded = false
    private(set) var loadingStage: LoadingStage = .idle
    private(set) var loadingDetail: String = ""

    @ObservationIgnored @Dependency(OpenTSLMInferenceService.self) private var openTSLMInferenceService: OpenTSLMInferenceService
    @ObservationIgnored @Dependency(HealthKitECGFetcher.self) private var healthKitECGFetcher: HealthKitECGFetcher

    /// The base model the checkpoints run on, loaded once and shared by all tasks.
    @ObservationIgnored private var baseModel: ModelContainer?

    required init() { }

    func setup() async throws {
        // MLX requires a real Metal GPU, which the iOS Simulator does not provide, so force the
        // CPU backend there. Must happen before any MLX compute. Slow, but simulator runs are
        // diagnostic only.
        #if targetEnvironment(simulator)
        Device.setDefault(device: Device(.cpu))
        logger.info("setup(): iOS Simulator detected — forcing MLX CPU backend")
        #endif

        // Make the MLX factory build our embedding-capable Llama for "llama"/"mistral"
        // model types, so the base model can be primed with OpenTSLM soft-prompt
        // embeddings. Idempotent; must run before any LLMModelFactory.shared.loadContainer.
        registerModelTypes()

        do {
            let configuration = try openTSLMInferenceService.baseModelConfiguration()
            logger.info("setup(): loading \(configuration.name, privacy: .public)")
            await MainActor.run {
                loadingStage = .loadingModel
                loadingDetail = configuration.name
            }
            let start = Date()
            // The first launch downloads the base model into Documents/huggingface, which the
            // system does not purge. meta-llama/Llama-3.2-1B is gated: swift-transformers reads
            // HF_TOKEN from the environment, so set it in a local, unshared Xcode scheme.
            let container = try await LLMModelFactory.shared.loadContainer(
                hub: HubApi(), configuration: configuration
            ) { progress in
                Task { @MainActor in
                    guard !self.loaded else {
                        return
                    }
                    self.loadingStage = .downloadingModel
                    self.loadingDetail = "\(Int(progress.fractionCompleted * 100))% of ~2.5 GB"
                }
            }
            baseModel = container
            let duration = Date().timeIntervalSince(start)
            logger.info("setup(): loaded in \(duration, privacy: .public)s")
            await MainActor.run {
                loaded = true
                loadingStage = .ready
                loadingDetail = String(format: "Loaded in %.1fs", duration)
            }
        } catch {
            logger.error("setup(): \(String(reflecting: error), privacy: .public)")
            await MainActor.run {
                loadingStage = .failed
                loadingDetail = error.localizedDescription
            }
            throw error
        }
    }

    /// EEG/sleep single-sample inference — returns the constructed prompt and the model's
    /// generated answer for the chat transcript.
    func eegSample() async -> ChatTurn {
        guard let baseModel else { return ChatTurn(prompt: "EEG sample", response: "The base model is not loaded.") }
        do {
            let interaction = try await openTSLMInferenceService.sleepSampleInteraction(base: baseModel)
            return ChatTurn(prompt: interaction.prompt, response: interaction.response)
        } catch {
            return ChatTurn(prompt: "EEG sample", response: "Failed: \(error.localizedDescription)")
        }
    }

    /// 12-lead ECG single-sample inference.
    func ecgSample() async -> ChatTurn {
        guard let baseModel else { return ChatTurn(prompt: "ECG sample", response: "The base model is not loaded.") }
        do {
            let interaction = try await openTSLMInferenceService.ecgSampleInteraction(base: baseModel)
            return ChatTurn(prompt: interaction.prompt, response: interaction.response)
        } catch {
            return ChatTurn(prompt: "ECG sample", response: "Failed: \(error.localizedDescription)")
        }
    }

    /// On-device accuracy eval over the pinned 20-row Sleep-EDF test-split export.
    /// Comparable sample-for-sample with the Python reference runs.
    func eegAccuracyEval() async -> ChatTurn {
        guard let baseModel else { return ChatTurn(prompt: "EEG eval", response: "The base model is not loaded.") }
        do {
            let summary = try await openTSLMInferenceService.runSleepAccuracyEval(
                sampleCount: 20, maxTokens: 200, base: baseModel
            )
            return ChatTurn(prompt: "Run the Sleep-EDF accuracy eval (20 pinned test samples).", response: summary)
        } catch {
            return ChatTurn(prompt: "EEG eval", response: "Failed: \(error.localizedDescription)")
        }
    }

    /// On-device accuracy eval over the bundled 12-lead ECG-QA CoT test rows.
    func ecgAccuracyEval() async -> ChatTurn {
        guard let baseModel else { return ChatTurn(prompt: "ECG eval", response: "The base model is not loaded.") }
        do {
            let summary = try await openTSLMInferenceService.runECGAccuracyEval(
                sampleCount: 20, maxTokens: 400, base: baseModel
            )
            return ChatTurn(prompt: "Run the ECG-QA accuracy eval (20 bundled test samples).", response: summary)
        } catch {
            return ChatTurn(prompt: "ECG eval", response: "Failed: \(error.localizedDescription)")
        }
    }

    /// Runs the model on the most recent Apple Watch ECG read from HealthKit — the only
    /// path here that uses real, prospectively recorded physiological data rather than a
    /// bundled research sample. Nothing leaves the device.
    func ecgFromHealthKit(healthKit: HealthKit) async -> ChatTurn {
        guard let baseModel else { return ChatTurn(prompt: "HealthKit ECG", response: "The base model is not loaded.") }
        do {
            try await healthKitECGFetcher.askForAuthorization(healthKit)
            guard let ecg = try await healthKitECGFetcher.fetchLatestElectrocardiogram(healthKit),
                  !ecg.voltages.isEmpty else {
                return ChatTurn(
                    prompt: "Analyze my latest Apple Watch ECG.",
                    response: "No ECG recording with voltage data was found. Record one with the ECG app on your Apple Watch, then try again."
                )
            }
            let recording = "voltages=\(ecg.voltages.count) freq=\(ecg.samplingFrequency ?? -1) class=\(ecg.classification)"
            logger.info("ecgFromHealthKit: \(recording, privacy: .public)")

            let analysis = try await openTSLMInferenceService.runECGInference(
                voltages: ecg.voltages,
                samplingFrequency: ecg.samplingFrequency ?? 512.0,
                base: baseModel
            )
            let recorded = ecg.startDate.formatted(date: .abbreviated, time: .shortened)
            let heartRate = ecg.averageHeartRate.map { " · \(Int($0)) bpm" } ?? ""
            return ChatTurn(
                prompt: "Analyze my Apple Watch ECG recorded \(recorded)\(heartRate) (\(ecg.voltages.count) samples, "
                    + "\(Int(ecg.samplingFrequency ?? 512)) Hz, classification: \(ecg.classification)).",
                response: analysis
            )
        } catch {
            return ChatTurn(prompt: "HealthKit ECG", response: "Failed: \(error.localizedDescription)")
        }
    }

    /// Repeated §4 sweeps for run-to-run variance (mean ± SD of TTFT and peak memory).
    func eegRepeatedBenchmark() async -> ChatTurn {
        await repeatedBenchmark(ecg: false, label: "EEG")
    }

    /// ECG variant of the repeated-variance benchmark.
    func ecgRepeatedBenchmark() async -> ChatTurn {
        await repeatedBenchmark(ecg: true, label: "ECG")
    }

    private func repeatedBenchmark(ecg: Bool, label: String) async -> ChatTurn {
        guard let baseModel else { return ChatTurn(prompt: "\(label) variance", response: "The base model is not loaded.") }
        do {
            let summary = try await openTSLMInferenceService.runRepeatedContextEconomics(
                ecg: ecg, repeats: 5, base: baseModel
            )
            return ChatTurn(prompt: "Run the \(label) context-economics sweep 5× and report variance.", response: summary)
        } catch {
            return ChatTurn(prompt: "\(label) variance", response: "Failed: \(error.localizedDescription)")
        }
    }

    /// EEG (sleep) §4 context-economics benchmark — full JSON also written to Documents.
    func eegBenchmark() async -> ChatTurn {
        guard let baseModel else { return ChatTurn(prompt: "EEG benchmark", response: "The base model is not loaded.") }
        do {
            let summary = try await openTSLMInferenceService.runContextEconomicsBenchmark(
                maxTokens: 80, maxPromptPositions: 4300, base: baseModel
            )
            return ChatTurn(prompt: "Run the EEG (sleep) context-economics benchmark.", response: summary)
        } catch {
            return ChatTurn(prompt: "EEG benchmark", response: "Failed: \(error.localizedDescription)")
        }
    }

    /// 12-lead ECG §4 context-economics benchmark — full JSON also written to Documents.
    func ecgBenchmark() async -> ChatTurn {
        guard let baseModel else { return ChatTurn(prompt: "ECG benchmark", response: "The base model is not loaded.") }
        do {
            let summary = try await openTSLMInferenceService.runContextEconomicsBenchmarkECG(
                maxTokens: 80, maxPromptPositions: 4300, base: baseModel
            )
            return ChatTurn(prompt: "Run the 12-lead ECG context-economics benchmark.", response: summary)
        } catch {
            return ChatTurn(prompt: "ECG benchmark", response: "Failed: \(error.localizedDescription)")
        }
    }
}

/// A single chat turn: the prompt shown in the sender bubble and the model's response
/// shown in the reply bubble.
struct ChatTurn {
    let prompt: String
    let response: String
}
