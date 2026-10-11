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
import Spezi
import MLX
import MLXLMCommon
import MLXLLM
import SpeziLLM
import SpeziHealthKit
import SpeziLLMLocal
import SpeziLLMLocalDownload

@Observable
class HealthDataInterpreter: DefaultInitializable, Module, EnvironmentAccessible {
    @ObservationIgnored private let logger = Logger(subsystem: "PocketTSLM", category: "HealthDataInterpreter")

    enum LoadingStage: String {
        case idle = "Idle"
        case stagingModel = "Staging local model files"
        case downloadingModel = "Downloading model weights"
        case configuringParameters = "Configuring LLM parameters"
        case creatingSession = "Creating LLM session"
        case openingSession = "Opening LLM session (loading weights)"
        case ready = "Ready"
        case failed = "Failed"
    }

    private(set) var loaded = false
    private(set) var loadingStage: LoadingStage = .idle
    private(set) var loadingDetail: String = ""
    
    @ObservationIgnored @Dependency(LLMRunner.self) private var llmRunner: LLMRunner
    @ObservationIgnored @Dependency(OpenTSLMInferenceService.self) private var openTSLMInferenceService: OpenTSLMInferenceService
    @ObservationIgnored @Dependency(HealthKitECGFetcher.self) private var healthKitECGFetcher: HealthKitECGFetcher
    
    @ObservationIgnored private var functionCallParameters: LLMLocalParameters?
    @ObservationIgnored private var functionCallSamplingParameters: LLMLocalSamplingParameters?
    @ObservationIgnored private var defaultParameters: LLMLocalParameters?
    @ObservationIgnored private var defaultSamplingParameters: LLMLocalSamplingParameters?
    @ObservationIgnored private var sharedSession: LLMLocalSession?
    @ObservationIgnored private let requiredModelFiles = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "model.safetensors"
    ]

    /// The subset shipped in the app bundle. `LLMLocalDownloadManager` only fetches
    /// `["*.safetensors", "config.json"]` from the Hub, so the tokenizer files have to come
    /// from the bundle — which means a bundle qualifies as a staging source without the
    /// weights present. Requiring `model.safetensors` here made `resolveLocalModelSourceDirectory`
    /// return nil, so the tokenizer was never staged and setup failed.
    @ObservationIgnored private let bundledModelFiles = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json"
    ]
    
    required init() { }
    
    func setup() async throws {
        logger.info("setup(): starting. modelID=\(Constants.llmModelName, privacy: .public) destination=\(Constants.llmLocalModelDirectory.path, privacy: .public)")

        // MLX requires a real Metal GPU, which the iOS Simulator does not provide, so
        // force the CPU backend there. This also runs in the Mac's full RAM (no 6 GB
        // process cap), enabling the otherwise-OOM cap=1000 ECG run for reference parity.
        // Must happen before any MLX compute. Slow, but simulator runs are diagnostic only.
        #if targetEnvironment(simulator)
        Device.setDefault(device: Device(.cpu))
        logger.info("setup(): iOS Simulator detected — forcing MLX CPU backend")
        #endif

        // Make the MLX factory build our embedding-capable Llama for "llama"/"mistral"
        // model types, so the session's model can be primed with OpenTSLM soft-prompt
        // embeddings. Idempotent; must run before any LLMModelFactory.shared.loadContainer.
        EmbeddingLlamaModelRegistration.register()

        await MainActor.run {
            loadingStage = .stagingModel
            loadingDetail = Constants.llmModelName
        }

        if Constants.skipLLMLoad {
            logger.info("setup(): skipping Llama staging/load (HEALTHYLLM_SKIP_LLM_LOAD=1)")
            await MainActor.run {
                loaded = true
                loadingStage = .ready
                loadingDetail = "Encoder-only (Llama skipped)"
            }
            return
        }

        do {
            try await stageLocalModelIfNeeded()
        try await downloadModelWeightsIfNeeded()
            try removeNonBaseWeightSafetensorsFromModelDirectory()
        } catch {
            logger.error("setup(): stageLocalModelIfNeeded threw: \(error.localizedDescription, privacy: .public)")
            await MainActor.run {
                loadingStage = .failed
                loadingDetail = "Staging failed: \(error.localizedDescription)"
            }
            throw error
        }

        await MainActor.run {
            loadingStage = .configuringParameters
        }
        logger.info("setup(): staging complete; configuring parameters")

        let chatTemplate: String? = Constants.useCustomChatTemplate ? Constants.llmModelChatTemplate : nil
        logger.info("setup(): chatTemplate=\(chatTemplate == nil ? "tokenizer-default" : "custom-jinja", privacy: .public)")

        functionCallParameters = .init(
            maxOutputLength: 32,
            chatTemplate: chatTemplate
        )
        functionCallSamplingParameters = .init(
            topP: 1.0,
            temperature: 0.001,
            penaltyRepeat: 1.3
        )

        defaultParameters = .init(
            maxOutputLength: Constants.llmDefaultMaxOutputLength,
            chatTemplate: chatTemplate
        )
        defaultSamplingParameters = .init(
            topP: 1.0,
            temperature: 0.7,
            penaltyRepeat: 1.2
        )
        guard let defaultParameters else {
            logger.error("setup(): defaultParameters unexpectedly nil after assignment")
            await MainActor.run {
                loadingStage = .failed
                loadingDetail = "defaultParameters nil"
            }
            return
        }

        await MainActor.run {
            loadingStage = .creatingSession
        }
        logger.info("setup(): creating LLMLocalSchema and session")

        let schema = LLMLocalSchema(
            model: .custom(id: Constants.llmModelName),
            parameters: defaultParameters,
            samplingParameters: defaultSamplingParameters ?? .init(),
            injectIntoContext: true
        )

        sharedSession = llmRunner.callAsFunction(with: schema)
        guard let sharedSession else {
            logger.error("setup(): llmRunner.callAsFunction returned nil session")
            await MainActor.run {
                loadingStage = .failed
                loadingDetail = "llmRunner returned nil session"
            }
            return
        }

        await MainActor.run {
            loadingStage = .openingSession
            loadingDetail = "Loading weights — this can take a while on first launch"
        }
        let setupStart = Date()
        let fileManager = FileManager.default
        let modelDirectory = Constants.llmLocalModelDirectory

        if hasRequiredModelFiles(in: modelDirectory, fileManager: fileManager) {
            logger.info("setup(): loading MLX container from local directory (bf16 base weights only)")
            do {
                let container = try await LLMModelFactory.shared.loadContainer(
                    configuration: ModelConfiguration(directory: modelDirectory)
                )
                await MainActor.run {
                    sharedSession.modelContainer = container
                    sharedSession.state = .ready
                }
                let setupDuration = Date().timeIntervalSince(setupStart)
                logger.info("setup(): direct loadContainer succeeded in \(setupDuration, privacy: .public)s")
                await MainActor.run {
                    loaded = true
                    loadingStage = .ready
                    loadingDetail = String(format: "Loaded in %.1fs", setupDuration)
                }
                return
            } catch {
                logger.error("setup(): direct loadContainer failed: \(String(reflecting: error), privacy: .public)")
            }
        }

        logger.info("setup(): calling sharedSession.setup() — Hub snapshot fallback")

        do {
            try await sharedSession.setup()
        } catch {
            logger.error("setup(): sharedSession.setup() threw after \(Date().timeIntervalSince(setupStart), privacy: .public)s: \(error.localizedDescription, privacy: .public)")
            await MainActor.run {
                loadingStage = .failed
                loadingDetail = "Session setup failed: \(error.localizedDescription)"
            }
            throw HealthDataInterpreterError.modelNotLoaded
        }

        let setupDuration = Date().timeIntervalSince(setupStart)
        logger.info("setup(): sharedSession.setup() completed in \(setupDuration, privacy: .public)s")

        await MainActor.run {
            loaded = true
            loadingStage = .ready
            loadingDetail = String(format: "Loaded in %.1fs", setupDuration)
        }
    }

    /// Downloads the base LLM weights from the Hugging Face Hub when they are not already
    /// present. `LLMLocalDownloadManager` fetches `["*.safetensors", "config.json"]` into
    /// `HubApi.shared.localRepoLocation(repo)`, which is the same directory as
    /// ``Constants/llmLocalModelDirectory``. `meta-llama/Llama-3.2-1B` is gated, so `HubApi`
    /// needs an `HF_TOKEN` in the process environment (set it in your user-specific,
    /// unshared Xcode run scheme).
    private func downloadModelWeightsIfNeeded() async throws {
        let fileManager = FileManager.default
        let destinationURL = Constants.llmLocalModelDirectory
        if fileManager.fileExists(atPath: destinationURL.appendingPathComponent("model.safetensors").path) {
            logger.info("downloadModelWeightsIfNeeded: weights already present — skipping download")
            return
        }

        logger.info("downloadModelWeightsIfNeeded: downloading \(Constants.llmModelName, privacy: .public) from the Hub")
        loadingStage = .downloadingModel
        let manager = await LLMLocalDownloadManager(model: .custom(id: Constants.llmModelName))
        await manager.startDownload()

        while true {
            let state = await manager.state
            switch state {
            case .downloaded:
                logger.info("downloadModelWeightsIfNeeded: download complete")
                loadingDetail = ""
                return
            case .error(let error):
                logger.error("downloadModelWeightsIfNeeded: failed: \(error.localizedDescription, privacy: .public)")
                throw error
            case .downloading(let progress):
                loadingDetail = "\(Int(progress.fractionCompleted * 100))% of ~2.5 GB"
            case .idle:
                break
            }
            try await Task.sleep(for: .seconds(1))
        }
    }

    private func stageLocalModelIfNeeded() async throws {
        let fileManager = FileManager.default
        let destinationURL = Constants.llmLocalModelDirectory
        logger.info("stageLocalModelIfNeeded: destination=\(destinationURL.path, privacy: .public)")

        do {
            try fileManager.createDirectory(at: destinationURL, withIntermediateDirectories: true)
        } catch {
            logger.error("Failed creating local model destination directory: \(error.localizedDescription, privacy: .public)")
            return
        }

        try stageOpenTSLMLoRACheckpointIfNeeded()

        if hasRequiredModelFiles(in: destinationURL, fileManager: fileManager) {
            logger.info("stageLocalModelIfNeeded: destination already has all required files (\(self.requiredModelFiles.joined(separator: ", "), privacy: .public)) — skipping copy")
            try removeNonBaseWeightSafetensorsFromModelDirectory()
            return
        }

        let missing = requiredModelFiles.filter { fileName in
            !fileManager.fileExists(atPath: destinationURL.appendingPathComponent(fileName).path)
        }
        logger.info("stageLocalModelIfNeeded: destination is missing files: \(missing.joined(separator: ", "), privacy: .public)")

        guard let sourceURL = resolveLocalModelSourceDirectory() else {
            logger.warning("stageLocalModelIfNeeded: no local model source directory found. The app will rely on the download flow / Hub cache. Required files still missing at destination.")
            return
        }
        logger.info("stageLocalModelIfNeeded: copying from \(sourceURL.path, privacy: .public)")

        do {
            try copyRequiredModelFiles(from: sourceURL, to: destinationURL)
            try removeNonBaseWeightSafetensorsFromModelDirectory()

            if hasRequiredModelFiles(in: destinationURL, fileManager: fileManager) {
                logger.info("stageLocalModelIfNeeded: staged local model from \(sourceURL.path, privacy: .public) to \(destinationURL.path, privacy: .public)")
            } else {
                let stillMissing = requiredModelFiles.filter { fileName in
                    !fileManager.fileExists(atPath: destinationURL.appendingPathComponent(fileName).path)
                }
                logger.error("stageLocalModelIfNeeded: staging finished but required files still missing: \(stillMissing.joined(separator: ", "), privacy: .public)")
            }
        } catch {
            logger.error("stageLocalModelIfNeeded: failed: \(error.localizedDescription, privacy: .public)")
            throw error
        }
    }

    /// Stage LoRA under ``Constants/openTSLMDocumentsDirectory`` — never into the Llama HF folder (MLX would load it as base weights).
    private func stageOpenTSLMLoRACheckpointIfNeeded() throws {
        let fileManager = FileManager.default
        let destinationDirectory = Constants.openTSLMDocumentsDirectory
        let destinationLoRA = destinationDirectory
            .appendingPathComponent("\(Constants.openTSLMLoRACheckpointName).safetensors")

        if fileManager.fileExists(atPath: destinationLoRA.path) {
            return
        }

        guard let sourceLoRA = resolveLoRACheckpointSource() else {
            if Constants.requireLoRACheckpoint {
                throw NSError(
                    domain: "HealthDataInterpreter",
                    code: 10,
                    userInfo: [NSLocalizedDescriptionKey: "HEALTHYLLM_REQUIRE_LORA=1 but no LoRA checkpoint was found. Set HEALTHYLLM_OPEN_TSLM_LORA_CHECKPOINT or bundle \(Constants.openTSLMLoRACheckpointName).safetensors under OpenTSLM/."]
                )
            }
            return
        }

        try fileManager.createDirectory(at: destinationDirectory, withIntermediateDirectories: true)
        try fileManager.copyItem(at: sourceLoRA, to: destinationLoRA)
        logger.info("LoRA checkpoint staged at \(destinationLoRA.path, privacy: .public)")
    }

    private func resolveLoRACheckpointSource() -> URL? {
        let fileManager = FileManager.default
        var candidates: [URL] = []

        if !Constants.openTSLMLoRACheckpointPath.isEmpty {
            candidates.append(URL(fileURLWithPath: Constants.openTSLMLoRACheckpointPath))
        }

        if let bundled = Bundle.main.url(
            forResource: Constants.openTSLMLoRACheckpointName,
            withExtension: "safetensors",
            subdirectory: Constants.openTSLMBundleSubdirectory
        ) {
            candidates.append(bundled)
        }

        if let bundleRoot = Bundle.main.resourceURL {
            candidates.append(
                bundleRoot
                    .appendingPathComponent(Constants.openTSLMBundleSubdirectory, isDirectory: true)
                    .appendingPathComponent("\(Constants.openTSLMLoRACheckpointName).safetensors")
            )
        }

        return candidates.first { fileManager.fileExists(atPath: $0.path) }
    }

    /// Remove adapter / OpenTSLM safetensors from the Llama model directory so ``loadContainer`` only sees base weights.
    private func removeNonBaseWeightSafetensorsFromModelDirectory() throws {
        let fileManager = FileManager.default
        let modelDirectory = Constants.llmLocalModelDirectory
        guard let items = try? fileManager.contentsOfDirectory(at: modelDirectory, includingPropertiesForKeys: nil) else {
            return
        }

        for item in items where item.pathExtension == "safetensors" {
            let name = item.lastPathComponent
            let isBaseWeight = name == "model.safetensors"
                || (name.hasPrefix("model-") && name.hasSuffix(".safetensors"))
            guard !isBaseWeight else {
                continue
            }
            try fileManager.removeItem(at: item)
            logger.info("Removed non-base safetensors from model dir: \(name, privacy: .public)")
        }
    }

    /// Copy only Llama base checkpoint files — never OpenTSLM encoder/projector/LoRA weights.
    private func copyRequiredModelFiles(from sourceURL: URL, to destinationURL: URL) throws {
        let fileManager = FileManager.default

        for fileName in requiredModelFiles {
            let sourceFile = sourceURL.appendingPathComponent(fileName)
            let destinationFile = destinationURL.appendingPathComponent(fileName)
            guard fileManager.fileExists(atPath: sourceFile.path) else {
                continue
            }
            if fileManager.fileExists(atPath: destinationFile.path) {
                continue
            }
            try fileManager.copyItem(at: sourceFile, to: destinationFile)
        }

        let sourceItems = try fileManager.contentsOfDirectory(at: sourceURL, includingPropertiesForKeys: nil)
        for item in sourceItems where item.pathExtension == "safetensors" {
            let name = item.lastPathComponent
            let isBaseWeight = name == "model.safetensors"
                || (name.hasPrefix("model-") && name.hasSuffix(".safetensors"))
            guard isBaseWeight else {
                continue
            }
            let destinationFile = destinationURL.appendingPathComponent(name)
            if !fileManager.fileExists(atPath: destinationFile.path) {
                try fileManager.copyItem(at: item, to: destinationFile)
            }
        }
    }

    private func hasRequiredModelFiles(in directoryURL: URL, fileManager: FileManager) -> Bool {
        requiredModelFiles.allSatisfy { fileName in
            fileManager.fileExists(atPath: directoryURL.appendingPathComponent(fileName).path)
        }
    }

    private func resolveLocalModelSourceDirectory() -> URL? {
        let fileManager = FileManager.default

        if let overridePath = Constants.localModelSourcePathOverride,
           let overrideURL = existingDirectoryURL(at: overridePath, fileManager: fileManager) {
            return overrideURL
        }

        if let bundledLocalModelURL = Bundle.main.resourceURL {
            let bundledDirectory = bundledLocalModelURL.appendingPathComponent(Constants.localModelBundleSubdirectory, isDirectory: true)
            if fileManager.fileExists(atPath: bundledDirectory.path) {
                return bundledDirectory
            }

            // Xcode synced folders flatten `Supporting Files/LocalLLM/*` to the bundle root,
            // so there is no `LocalLLM` subdirectory in the built app.
            let bundledRootFiles = bundledModelFiles.allSatisfy { fileName in
                fileManager.fileExists(atPath: bundledLocalModelURL.appendingPathComponent(fileName).path)
            }

            if bundledRootFiles {
                return bundledLocalModelURL
            }
        }

        // Fallback: detect a downloaded model snapshot from Hugging Face cache.
        let sanitizedRepoID = Constants.llmModelName.replacingOccurrences(of: "/", with: "--")
        let hostSnapshotsPath = "\(Constants.hostHuggingFaceCacheRoot)/models--\(sanitizedRepoID)/snapshots"
        let hostSnapshotsURL = URL(fileURLWithPath: hostSnapshotsPath, isDirectory: true)

        if let hostSnapshot = newestSnapshotDirectory(in: hostSnapshotsURL, fileManager: fileManager) {
            return hostSnapshot
        }

        let snapshotsPath = "~/.cache/huggingface/hub/models--\(sanitizedRepoID)/snapshots"
        let snapshotsURL = URL(fileURLWithPath: NSString(string: snapshotsPath).expandingTildeInPath, isDirectory: true)
        return newestSnapshotDirectory(in: snapshotsURL, fileManager: fileManager)
    }

    private func newestSnapshotDirectory(in snapshotsURL: URL, fileManager: FileManager) -> URL? {
        guard fileManager.fileExists(atPath: snapshotsURL.path) else {
            return nil
        }

        let directoryContents = try? fileManager.contentsOfDirectory(
            at: snapshotsURL,
            includingPropertiesForKeys: [.contentModificationDateKey],
            options: [.skipsHiddenFiles]
        )

        guard let directoryContents else {
            return nil
        }

        var newestURL: URL?
        var newestDate = Date.distantPast

        for url in directoryContents {
            var isDirectory: ObjCBool = false
            guard fileManager.fileExists(atPath: url.path, isDirectory: &isDirectory), isDirectory.boolValue else {
                continue
            }
            let modified = (try? url.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
            if modified > newestDate {
                newestDate = modified
                newestURL = url
            }
        }

        return newestURL
    }

    private func existingDirectoryURL(at rawPath: String, fileManager: FileManager) -> URL? {
        let expandedPath = NSString(string: rawPath).expandingTildeInPath
        let url = URL(fileURLWithPath: expandedPath, isDirectory: true)
        var isDirectory: ObjCBool = false

        guard fileManager.fileExists(atPath: url.path, isDirectory: &isDirectory), isDirectory.boolValue else {
            return nil
        }

        return url
    }

    private func copyDirectoryContents(from sourceURL: URL, to destinationURL: URL) throws {
        let fileManager = FileManager.default
        let items = try fileManager.contentsOfDirectory(at: sourceURL, includingPropertiesForKeys: nil)

        for item in items {
            let destinationItem = destinationURL.appendingPathComponent(item.lastPathComponent)
            var isDirectory: ObjCBool = false
            let exists = fileManager.fileExists(atPath: item.path, isDirectory: &isDirectory)

            guard exists else {
                continue
            }

            if isDirectory.boolValue {
                try fileManager.createDirectory(at: destinationItem, withIntermediateDirectories: true)
                try copyDirectoryContents(from: item, to: destinationItem)
            } else if !fileManager.fileExists(atPath: destinationItem.path) {
                try fileManager.copyItem(at: item, to: destinationItem)
            }
        }
    }


    /// EEG/sleep single-sample inference — returns the constructed prompt and the model's
    /// generated answer for the chat transcript.
    func eegSample() async -> ChatTurn {
        guard let sharedSession else { return ChatTurn(prompt: "EEG sample", response: "LLM session not loaded.") }
        do {
            let interaction = try await openTSLMInferenceService.sleepSampleInteraction(llmRunner: llmRunner, llmSession: sharedSession)
            return ChatTurn(prompt: interaction.prompt, response: interaction.response)
        } catch {
            return ChatTurn(prompt: "EEG sample", response: "Failed: \(error.localizedDescription)")
        }
    }

    /// 12-lead ECG single-sample inference.
    func ecgSample() async -> ChatTurn {
        guard let sharedSession else { return ChatTurn(prompt: "ECG sample", response: "LLM session not loaded.") }
        do {
            let interaction = try await openTSLMInferenceService.ecgSampleInteraction(llmRunner: llmRunner, llmSession: sharedSession)
            return ChatTurn(prompt: interaction.prompt, response: interaction.response)
        } catch {
            return ChatTurn(prompt: "ECG sample", response: "Failed: \(error.localizedDescription)")
        }
    }

    /// On-device accuracy eval over the pinned 20-row Sleep-EDF test-split export.
    /// Comparable sample-for-sample with the Python reference runs.
    func eegAccuracyEval() async -> ChatTurn {
        guard let sharedSession else { return ChatTurn(prompt: "EEG eval", response: "LLM session not loaded.") }
        do {
            let summary = try await openTSLMInferenceService.runSleepAccuracyEval(
                sampleCount: 20, maxTokens: 200, llmRunner: llmRunner, llmSession: sharedSession
            )
            return ChatTurn(prompt: "Run the Sleep-EDF accuracy eval (20 pinned test samples).", response: summary)
        } catch {
            return ChatTurn(prompt: "EEG eval", response: "Failed: \(error.localizedDescription)")
        }
    }

    /// On-device accuracy eval over the bundled 12-lead ECG-QA CoT test rows.
    func ecgAccuracyEval() async -> ChatTurn {
        guard let sharedSession else { return ChatTurn(prompt: "ECG eval", response: "LLM session not loaded.") }
        do {
            let summary = try await openTSLMInferenceService.runECGAccuracyEval(
                sampleCount: 20, maxTokens: 400, llmRunner: llmRunner, llmSession: sharedSession
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
        guard let sharedSession else { return ChatTurn(prompt: "HealthKit ECG", response: "LLM session not loaded.") }
        do {
            try await healthKitECGFetcher.askForAuthorization(healthKit)
            guard let ecg = try await healthKitECGFetcher.fetchLatestElectrocardiogram(healthKit),
                  !ecg.voltages.isEmpty else {
                return ChatTurn(
                    prompt: "Analyze my latest Apple Watch ECG.",
                    response: "No ECG recording with voltage data was found. Record one with the ECG app on your Apple Watch, then try again."
                )
            }
            logger.info("ecgFromHealthKit: voltages=\(ecg.voltages.count, privacy: .public) freq=\(ecg.samplingFrequency ?? -1, privacy: .public) class=\(ecg.classification, privacy: .public)")

            let analysis = try await openTSLMInferenceService.runECGInference(
                voltages: ecg.voltages,
                samplingFrequency: ecg.samplingFrequency ?? 512.0,
                classification: ecg.classification,
                symptomsStatus: ecg.symptomsStatus,
                averageHeartRate: ecg.averageHeartRate,
                llmRunner: llmRunner,
                llmSession: sharedSession
            )
            let recorded = ecg.startDate.formatted(date: .abbreviated, time: .shortened)
            let heartRate = ecg.averageHeartRate.map { " · \(Int($0)) bpm" } ?? ""
            return ChatTurn(
                prompt: "Analyze my Apple Watch ECG recorded \(recorded)\(heartRate) (\(ecg.voltages.count) samples, \(Int(ecg.samplingFrequency ?? 512)) Hz, classification: \(ecg.classification)).",
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
        guard let sharedSession else { return ChatTurn(prompt: "\(label) variance", response: "LLM session not loaded.") }
        do {
            let summary = try await openTSLMInferenceService.runRepeatedContextEconomics(
                ecg: ecg, repeats: 5, llmRunner: llmRunner, llmSession: sharedSession
            )
            return ChatTurn(prompt: "Run the \(label) context-economics sweep 5× and report variance.", response: summary)
        } catch {
            return ChatTurn(prompt: "\(label) variance", response: "Failed: \(error.localizedDescription)")
        }
    }

    /// EEG (sleep) §4 context-economics benchmark — full JSON also written to Documents.
    func eegBenchmark() async -> ChatTurn {
        guard let sharedSession else { return ChatTurn(prompt: "EEG benchmark", response: "LLM session not loaded.") }
        do {
            let summary = try await openTSLMInferenceService.runContextEconomicsBenchmark(
                maxTokens: 80, maxPromptPositions: 4300, llmRunner: llmRunner, llmSession: sharedSession
            )
            return ChatTurn(prompt: "Run the EEG (sleep) context-economics benchmark.", response: summary)
        } catch {
            return ChatTurn(prompt: "EEG benchmark", response: "Failed: \(error.localizedDescription)")
        }
    }

    /// 12-lead ECG §4 context-economics benchmark — full JSON also written to Documents.
    func ecgBenchmark() async -> ChatTurn {
        guard let sharedSession else { return ChatTurn(prompt: "ECG benchmark", response: "LLM session not loaded.") }
        do {
            let summary = try await openTSLMInferenceService.runContextEconomicsBenchmarkECG(
                maxTokens: 80, maxPromptPositions: 4300, llmRunner: llmRunner, llmSession: sharedSession
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
