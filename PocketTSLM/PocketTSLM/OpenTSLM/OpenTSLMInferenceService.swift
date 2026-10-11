//
// This source file is part of the PocketTSLM open-source project
//
// SPDX-FileCopyrightText: 2026 Schmiedmayer Lab and the project authors (see CONTRIBUTORS.md)
//
// SPDX-License-Identifier: MIT
//

import Foundation
import MLX
import OSLog
import Spezi
import SpeziLLM
import SpeziLLMLocal

@Observable
class OpenTSLMInferenceService: DefaultInitializable, Module, EnvironmentAccessible {
    @ObservationIgnored private let logger = Logger(subsystem: "HealthyLLM", category: "OpenTSLMInferenceService")

    required init() { }

    func runSleepSampleInference(
        split: SleepEDFDataset.Split = .test,
        sampleIndex: Int = 0,
        llmRunner: LLMRunner? = nil,
        llmSession: LLMLocalSession? = nil
    ) async throws -> String {
        guard let csvURL = resolveAssetURL(
            overridePath: Constants.openTSLMSleepCSVPath,
            bundledName: Constants.openTSLMSleepCSVName,
            fileExtension: "csv"
        ) else {
            throw NSError(domain: "OpenTSLMInferenceService", code: 1, userInfo: [NSLocalizedDescriptionKey: "Missing sleep CSV in bundle/OpenTSLM or override path"]) 
        }

        guard let encoderURL = resolveAssetURL(
            overridePath: Constants.openTSLMEncoderCheckpointPath,
            bundledName: Constants.openTSLMEncoderCheckpointName,
            fileExtension: "safetensors"
        ) else {
            throw NSError(domain: "OpenTSLMInferenceService", code: 2, userInfo: [NSLocalizedDescriptionKey: "Missing encoder checkpoint in bundle/OpenTSLM or override path"])
        }

        guard let projectorURL = resolveAssetURL(
            overridePath: Constants.openTSLMProjectorCheckpointPath,
            bundledName: Constants.openTSLMProjectorCheckpointName,
            fileExtension: "safetensors"
        ) else {
            throw NSError(domain: "OpenTSLMInferenceService", code: 3, userInfo: [NSLocalizedDescriptionKey: "Missing projector checkpoint in bundle/OpenTSLM or override path"])
        }

        logger.info("OpenTSLM assets: csv=\(csvURL.path), encoder=\(encoderURL.path), projector=\(projectorURL.path)")

        // Read one raw CSV row (no split) so this matches the Python reference exactly
        // for single-sample parity checks. sampleIndex selects the raw row.
        let safeIndex = max(sampleIndex, 0)
        let sample = cappedSample(try SleepEDFDataset.rawSample(csvURL: csvURL, rowIndex: safeIndex), limit: Constants.openTSLMSleepMaxTimeSeriesLength)

        let projected: [MLXArray]
        do {
            let pipeline = OpenTSLMSPPipeline(hiddenSize: 2048)
            try pipeline.loadWeights(encoderURL: encoderURL, projectorURL: projectorURL)
            projected = pipeline.projectSample(sample)
        }

        guard let first = projected.first else {
            throw NSError(domain: "OpenTSLMInferenceService", code: 5, userInfo: [NSLocalizedDescriptionKey: "Projection returned no tensors"])
        }

        eval(first)
        GPU.clearCache()
        logger.info("OpenTSLM encoder projection done: split=\(split.rawValue), sample=\(safeIndex)")

        if let llmRunner = llmRunner, let llmSession = llmSession {
            let loraApplied = await applyLoRAIfAvailable(on: llmSession)

            GPU.clearCache()

            if Constants.openTSLMRunSampleLLMGeneration {
                let openTSLMLLM = OpenTSLMLLM(llmRunner: llmRunner, session: llmSession)
                let generatedText = try await openTSLMLLM.generate(
                    prePrompt: sample.prePrompt,
                    timeSeriesText: sample.timeSeriesText,
                    timeSeriesEmbeddings: projected,
                    postPrompt: sample.postPrompt,
                    maxTokens: 200
                )

                let outputText = """
                **LLM-Generated Analysis:**

                \(generatedText)

                ---
                Ground-truth label: \(sample.label)
                Ground-truth answer: \(sample.answer)
                """

                return formatSampleReport(
                    title: "Sleep-EDF sample inference with LLM generation",
                    prePrompt: sample.prePrompt,
                    timeSeriesText: sample.timeSeriesText,
                    postPrompt: sample.postPrompt,
                    label: sample.label,
                    answer: outputText,
                    extraLines: openTSLMSampleExtraLines(
                        split: split,
                        sampleIndex: safeIndex,
                        sample: sample,
                        embeddingsShape: first.shape,
                        loraApplied: loraApplied
                    )
                )
            }

            let outputText = """
            **OpenTSLM encoder + LoRA (on-device decode skipped to avoid OOM)**

            Projected time-series embeddings: \(first.shape)
            LoRA applied to Llama: \(loraApplied ? "yes" : "no")
            Ground-truth label: \(sample.label)
            Ground-truth answer: \(sample.answer)

            Set `HEALTHYLLM_OPEN_TSLM_RUN_LLM=1` in the scheme to attempt a short LLM decode (may exceed device memory).
            """

            return formatSampleReport(
                title: "Sleep-EDF sample inference (encoder + LoRA)",
                prePrompt: sample.prePrompt,
                timeSeriesText: sample.timeSeriesText,
                postPrompt: sample.postPrompt,
                label: sample.label,
                answer: outputText,
                extraLines: openTSLMSampleExtraLines(
                    split: split,
                    sampleIndex: safeIndex,
                    sample: sample,
                    embeddingsShape: first.shape,
                    loraApplied: loraApplied
                )
            )
        } else {
            // Fallback: just describe the embeddings
            let outputText = """
            **Embeddings Computed Successfully**

            The time series embeddings were computed (\(first.shape)) but no LLM session was provided for generation.

            To enable LLM generation with embeddings:
            1. Ensure the HealthDataInterpreter is initialized with a valid LLM session
            2. Pass the llmRunner and llmSession parameters to this method

            Ground-truth label: \(sample.label)
            Ground-truth answer: \(sample.answer)
            """

            return formatSampleReport(
                title: "Sleep-EDF sample inference (embeddings only)",
                prePrompt: sample.prePrompt,
                timeSeriesText: sample.timeSeriesText,
                postPrompt: sample.postPrompt,
                label: sample.label,
                answer: outputText,
                extraLines: [
                    "split: \(split.rawValue)",
                    "sample_index: \(safeIndex)",
                    "series_count: \(sample.timeSeries.count)",
                    "embeddings_shape: \(first.shape)",
                    "llm_integration: no",
                ]
            )
        }
    }

    func runECGSampleInference(
        split: ECGQACoTDataset.Split = .test,
        sampleIndex: Int = 0,
        llmRunner: LLMRunner? = nil,
        llmSession: LLMLocalSession? = nil
    ) async throws -> String {
        let loaded = try loadECGQACoTSample(split: split, sampleIndex: sampleIndex)
        let sample = cappedSample(loaded.sample)
        var metadata = loaded.metadata
        metadata = ECGQACoTSampleMetadata(
            source: metadata.source,
            loader: metadata.loader,
            split: metadata.split,
            sampleIndex: metadata.sampleIndex,
            ecgId: metadata.ecgId,
            templateId: metadata.templateId,
            questionType: metadata.questionType,
            seriesCount: sample.timeSeries.count,
            samplesPerLead: sample.timeSeries.first?.count ?? 0
        )

        // Load ECG encoder + projector (per-task checkpoints — sleep weights would
        // produce a meaningless projection for ECG inputs).
        let projected = try loadPipelineAndProject(
            sample,
            encoderName: Constants.openTSLMECGEncoderCheckpointName,
            projectorName: Constants.openTSLMECGProjectorCheckpointName
        )
        guard let first = projected.first else {
            throw NSError(domain: "OpenTSLMInferenceService", code: 5, userInfo: [NSLocalizedDescriptionKey: "Projection returned no tensors"])
        }

        GPU.clearCache()

        guard Constants.openTSLMRunSampleLLMGeneration,
              let llmRunner,
              let llmSession
        else {
            let outputText = """
            **OpenTSLM ECG encoder (Llama decode skipped)**

            Projected time-series embeddings: \(first.shape)
            Dataset answer: \(sample.answer)

            Set `HEALTHYLLM_OPEN_TSLM_RUN_LLM=1` (and do not set `HEALTHYLLM_SKIP_LLM_LOAD=1`) to run Llama decode — likely OOM on physical iPhone.
            """

            return formatSampleReport(
                title: "ECG-QA CoT sample inference (encoder only)",
                prePrompt: sample.prePrompt,
                timeSeriesText: sample.timeSeriesText,
                postPrompt: sample.postPrompt,
                label: sample.label,
                answer: outputText,
                extraLines: ecgQACoTSampleExtraLines(
                    metadata: metadata,
                    embeddingsShape: first.shape,
                    loraApplied: false
                ) + [
                    "llm_model: \(Constants.llmModelName)",
                    "llm_integration: encoder-only",
                    "llama_loaded: \(Constants.skipLLMLoad ? "no" : "yes")",
                ]
            )
        }

        // Llama loaded + generation requested
        let loraApplied = await applyLoRAIfAvailable(
            on: llmSession,
            checkpointName: Constants.openTSLMECGLoRACheckpointName
        )
        GPU.clearCache()

        let openTSLMLLM = OpenTSLMLLM(llmRunner: llmRunner, session: llmSession)
        let generatedText = try await openTSLMLLM.generate(
            prePrompt: sample.prePrompt,
            timeSeriesText: sample.timeSeriesText,
            timeSeriesEmbeddings: projected,
            postPrompt: sample.postPrompt,
            maxTokens: 200
        )

        let outputText = """
        **LLM-Generated ECG Analysis:**

        \(generatedText)

        ---
        Dataset answer: \(sample.answer)
        """

        return formatSampleReport(
            title: "ECG-QA CoT sample inference with LLM generation",
            prePrompt: sample.prePrompt,
            timeSeriesText: sample.timeSeriesText,
            postPrompt: sample.postPrompt,
            label: sample.label,
            answer: outputText,
            extraLines: ecgQACoTSampleExtraLines(
                metadata: metadata,
                embeddingsShape: first.shape,
                loraApplied: loraApplied
            ) + [
                "llm_model: \(Constants.llmModelName)",
                "llm_integration: generate",
            ]
        )
    }

    /// A single prompt → response interaction for the chat UI: the constructed text prompt
    /// (pre-prompt + per-lead time-series descriptions + post-prompt) and the model's
    /// generated answer, returned separately so they can occupy the sender and reply bubbles.
    struct SampleInteraction {
        let prompt: String
        let response: String
        let groundTruth: String
    }

    /// EEG/sleep single-sample interaction (fixed raw CSV row): projects, applies the SP
    /// LoRA, and runs the LLM decode, returning the prompt and generated answer separately.
    func sleepSampleInteraction(
        sampleIndex: Int = 0,
        llmRunner: LLMRunner,
        llmSession: LLMLocalSession
    ) async throws -> SampleInteraction {
        guard let csvURL = resolveAssetURL(
            overridePath: Constants.openTSLMSleepCSVPath,
            bundledName: Constants.openTSLMSleepCSVName,
            fileExtension: "csv"
        ) else {
            throw NSError(domain: "OpenTSLMInferenceService", code: 1, userInfo: [NSLocalizedDescriptionKey: "Missing sleep CSV in bundle/OpenTSLM or override path"])
        }
        let sample = cappedSample(
            try SleepEDFDataset.rawSample(csvURL: csvURL, rowIndex: max(sampleIndex, 0)),
            limit: Constants.openTSLMSleepMaxTimeSeriesLength
        )
        let projected = try loadPipelineAndProject(sample)
        GPU.clearCache()
        _ = await applyLoRAIfAvailable(on: llmSession)
        GPU.clearCache()
        let openTSLMLLM = OpenTSLMLLM(llmRunner: llmRunner, session: llmSession)
        let generatedText = try await openTSLMLLM.generate(
            prePrompt: sample.prePrompt,
            timeSeriesText: sample.timeSeriesText,
            timeSeriesEmbeddings: projected,
            postPrompt: sample.postPrompt,
            maxTokens: 200
        )
        return SampleInteraction(prompt: promptText(for: sample), response: generatedText, groundTruth: sample.answer)
    }

    /// 12-lead ECG-QA single-sample interaction (bundled sample) using the ECG-specific
    /// encoder/projector/LoRA checkpoints.
    func ecgSampleInteraction(
        sampleIndex: Int = 0,
        llmRunner: LLMRunner,
        llmSession: LLMLocalSession
    ) async throws -> SampleInteraction {
        let loaded = try loadECGQACoTSample(split: .test, sampleIndex: sampleIndex)
        let sample = cappedSample(loaded.sample)
        let projected = try loadPipelineAndProject(
            sample,
            encoderName: Constants.openTSLMECGEncoderCheckpointName,
            projectorName: Constants.openTSLMECGProjectorCheckpointName
        )
        GPU.clearCache()
        _ = await applyLoRAIfAvailable(on: llmSession, checkpointName: Constants.openTSLMECGLoRACheckpointName)
        GPU.clearCache()
        let openTSLMLLM = OpenTSLMLLM(llmRunner: llmRunner, session: llmSession)
        let generatedText = try await openTSLMLLM.generate(
            prePrompt: sample.prePrompt,
            timeSeriesText: sample.timeSeriesText,
            timeSeriesEmbeddings: projected,
            postPrompt: sample.postPrompt,
            maxTokens: 200
        )
        return SampleInteraction(prompt: promptText(for: sample), response: generatedText, groundTruth: sample.answer)
    }

    /// The displayable text prompt for a sample (pre-prompt, per-lead time-series
    /// descriptions, post-prompt). The numeric series itself is injected as embeddings.
    private func promptText(for sample: OpenTSLMSPSample) -> String {
        ([sample.prePrompt] + sample.timeSeriesText + [sample.postPrompt]).joined(separator: "\n")
    }

    /// Runs OpenTSLM-SP on a real (e.g. HealthKit) ECG recording and returns the model's
    /// analysis text. Unlike ``runECGSampleInference`` (a debug command using a hardcoded/JSON
    /// sample and emitting a verbose report), this uses the supplied recording, requires a live
    /// LLM session, and returns a clean user-facing answer.
    func runECGInference(
        voltages: [Double],
        samplingFrequency: Double,
        classification: String?,
        symptomsStatus: String?,
        averageHeartRate: Double?,
        llmRunner: LLMRunner?,
        llmSession: LLMLocalSession?
    ) async throws -> String {
        guard let llmRunner, let llmSession else {
            throw NSError(
                domain: "OpenTSLMInferenceService", code: 7,
                userInfo: [NSLocalizedDescriptionKey: "An LLM session is required to analyze the ECG."])
        }
        guard !voltages.isEmpty else {
            throw NSError(
                domain: "OpenTSLMInferenceService", code: 8,
                userInfo: [NSLocalizedDescriptionKey: "The ECG recording has no voltage samples."])
        }

        logger.info("runECGInference: start voltages=\(voltages.count, privacy: .public) freq=\(samplingFrequency, privacy: .public)")
        let ecg = ECGSample(
            source: .healthkit,
            samplingFrequency: samplingFrequency,
            classification: classification,
            symptomsStatus: symptomsStatus,
            averageHeartRate: averageHeartRate,
            voltages: voltages
        )
        let sample = cappedSample(makeOpenTSLMSample(from: ecg))
        let projected = try loadPipelineAndProject(
            sample,
            encoderName: Constants.openTSLMECGEncoderCheckpointName,
            projectorName: Constants.openTSLMECGProjectorCheckpointName
        )
        logger.info("runECGInference: projected series=\(projected.count, privacy: .public) shape0=\(projected.first?.shape ?? [], privacy: .public)")

        let loraApplied = await applyLoRAIfAvailable(
            on: llmSession,
            checkpointName: Constants.openTSLMECGLoRACheckpointName
        )
        GPU.clearCache()
        logger.info("runECGInference: loraApplied=\(loraApplied, privacy: .public); calling generate")

        let openTSLMLLM = OpenTSLMLLM(llmRunner: llmRunner, session: llmSession)
        let analysis = try await openTSLMLLM.generate(
            prePrompt: sample.prePrompt,
            timeSeriesText: sample.timeSeriesText,
            timeSeriesEmbeddings: projected,
            postPrompt: sample.postPrompt,
            maxTokens: 200
        )
        logger.info("runECGInference: generate returned \(analysis.count, privacy: .public) chars")
        let trimmed = analysis.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? "The ECG model did not produce an analysis." : trimmed
    }

    // MARK: - Repeated context-economics runs (variance)

    /// Per-(length, encoding) summary over repeated sweeps. Token counts are analytic and must
    /// be identical across runs; \gls{TTFT} and peak memory are the wall-clock quantities whose
    /// run-to-run spread this experiment exists to report.
    struct RepeatedArmStats: Encodable {
        let encoding: String
        let seriesLength: Int
        let runs: Int
        let skipped: Bool
        let promptPositions: Int
        let timeSeriesPositions: Int
        let positionsStable: Bool
        let prefillSecondsMean: Double
        let prefillSecondsSD: Double
        let prefillSecondsMin: Double
        let prefillSecondsMax: Double
        let peakMemoryMiBMean: Double
        let peakMemoryMiBSD: Double
        let peakMemoryMiBMin: Double
        let peakMemoryMiBMax: Double
        /// Per-run values in run order. Thermal throttling produces a monotone drift rather
        /// than random scatter, which a mean±SD summary hides — so keep the raw series.
        let prefillSecondsPerRun: [Double]
        let peakMemoryMiBPerRun: [Double]
    }

    struct RepeatedReport: Encodable {
        let modality: String
        let repeats: Int
        let lengths: [Int]
        let maxTokens: Int
        let maxPromptPositions: Int
        let loraApplied: Bool
        /// `ProcessInfo.thermalState` before each sweep, in run order. Sustained inference
        /// drives this from `nominal` upward, and the device throttles accordingly.
        let thermalStatePerRun: [String]
        let lowPowerModePerRun: [Bool]
        let arms: [RepeatedArmStats]
    }

    private static func thermalStateName(_ state: ProcessInfo.ThermalState) -> String {
        switch state {
        case .nominal: return "nominal"
        case .fair: return "fair"
        case .serious: return "serious"
        case .critical: return "critical"
        @unknown default: return "unknown"
        }
    }

    private static func meanSD(_ values: [Double]) -> (mean: Double, sd: Double, min: Double, max: Double) {
        guard !values.isEmpty else { return (0, 0, 0, 0) }
        let mean = values.reduce(0, +) / Double(values.count)
        guard values.count > 1 else { return (mean, 0, mean, mean) }
        // Sample standard deviation (n-1): we are estimating spread, not describing a population.
        let variance = values.reduce(0) { $0 + ($1 - mean) * ($1 - mean) } / Double(values.count - 1)
        return (mean, variance.squareRoot(), values.min() ?? mean, values.max() ?? mean)
    }

    /// Repeats the §4 context-economics sweep `repeats` times on one modality and reports
    /// mean ± SD of \gls{TTFT} and peak memory per length and encoding. The pipeline, \gls{LORA},
    /// and LLM session are loaded once and shared across runs, so the spread reflects run-to-run
    /// variation of the measurement itself rather than setup cost.
    func runRepeatedContextEconomics(
        ecg: Bool,
        repeats: Int = 5,
        lengths: [Int]? = nil,
        maxTokens: Int = 80,
        contextWindow: Int = 8192,
        maxPromptPositions: Int = 4300,
        llmRunner: LLMRunner,
        llmSession: LLMLocalSession?
    ) async throws -> String {
        guard let llmSession else {
            throw NSError(domain: "OpenTSLMInferenceService", code: 6,
                          userInfo: [NSLocalizedDescriptionKey: "An LLM session is required."])
        }
        let sweep = lengths ?? (ecg ? [100, 250, 500, 1000] : [100, 250, 500, 1000, 1500])

        let encoderName = ecg ? Constants.openTSLMECGEncoderCheckpointName : Constants.openTSLMEncoderCheckpointName
        let projectorName = ecg ? Constants.openTSLMECGProjectorCheckpointName : Constants.openTSLMProjectorCheckpointName
        guard let encoderURL = resolveAssetURL(
            overridePath: Constants.openTSLMEncoderCheckpointPath, bundledName: encoderName, fileExtension: "safetensors"
        ), let projectorURL = resolveAssetURL(
            overridePath: Constants.openTSLMProjectorCheckpointPath, bundledName: projectorName, fileExtension: "safetensors"
        ) else {
            throw NSError(domain: "OpenTSLMInferenceService", code: 2,
                          userInfo: [NSLocalizedDescriptionKey: "Missing encoder/projector checkpoint"])
        }

        let sample: OpenTSLMSPSample
        if ecg {
            sample = cappedSample(try loadECGQACoTSample(split: .test, sampleIndex: 0).sample)
        } else {
            guard let csvURL = resolveAssetURL(
                overridePath: Constants.openTSLMSleepCSVPath,
                bundledName: Constants.openTSLMSleepEvalCSVName,
                fileExtension: "csv"
            ) else {
                throw NSError(domain: "OpenTSLMInferenceService", code: 1,
                              userInfo: [NSLocalizedDescriptionKey: "Missing pinned sleep CSV"])
            }
            sample = cappedSample(try SleepEDFDataset.rawSample(csvURL: csvURL, rowIndex: 0),
                                  limit: Constants.openTSLMSleepMaxTimeSeriesLength)
        }

        let pipeline = OpenTSLMSPPipeline(hiddenSize: 2048)
        try pipeline.loadWeights(encoderURL: encoderURL, projectorURL: projectorURL)
        let loraApplied = await applyLoRAIfAvailable(
            on: llmSession,
            checkpointName: ecg ? Constants.openTSLMECGLoRACheckpointName : Constants.openTSLMLoRACheckpointName
        )
        GPU.clearCache()

        let benchmark = OpenTSLMBenchmark(pipeline: pipeline, llmRunner: llmRunner, session: llmSession)
        var reports: [OpenTSLMBenchmark.Report] = []
        var thermalStates: [String] = []
        var lowPowerModes: [Bool] = []
        for run in 0 ..< max(repeats, 1) {
            let thermal = Self.thermalStateName(ProcessInfo.processInfo.thermalState)
            let lowPower = ProcessInfo.processInfo.isLowPowerModeEnabled
            thermalStates.append(thermal)
            lowPowerModes.append(lowPower)
            logger.info("repeated benchmark: run \(run + 1, privacy: .public) starting, thermalState=\(thermal, privacy: .public) lowPower=\(lowPower, privacy: .public)")
            let report = try await benchmark.run(
                sample: sample, lengths: sweep, maxTokens: maxTokens,
                contextWindow: contextWindow, maxPromptPositions: maxPromptPositions,
                loraApplied: loraApplied)
            reports.append(report)
            GPU.clearCache()
            logger.info("repeated benchmark (\(ecg ? "ecg" : "eeg", privacy: .public)): run \(run + 1, privacy: .public)/\(repeats, privacy: .public) done")
        }

        var arms: [RepeatedArmStats] = []
        for (rowIndex, length) in sweep.enumerated() {
            for encoding in ["soft_prompt", "text_serialized"] {
                let samples: [OpenTSLMBenchmark.ArmStats] = reports.compactMap { report in
                    guard rowIndex < report.rows.count else { return nil }
                    let row = report.rows[rowIndex]
                    return encoding == "soft_prompt" ? row.softPrompt : row.textSerialized
                }
                guard let first = samples.first else { continue }
                let ran = samples.filter { !$0.didOOM }
                let prefill = Self.meanSD(ran.map(\.prefillSeconds))
                let memory = Self.meanSD(ran.map { Double($0.peakMemoryBytes) / 1_048_576 })
                arms.append(RepeatedArmStats(
                    encoding: encoding,
                    seriesLength: first.seriesLength,
                    runs: ran.count,
                    skipped: ran.isEmpty,
                    promptPositions: first.promptPositions,
                    timeSeriesPositions: first.timeSeriesPositions,
                    positionsStable: samples.allSatisfy { $0.promptPositions == first.promptPositions },
                    prefillSecondsMean: prefill.mean, prefillSecondsSD: prefill.sd,
                    prefillSecondsMin: prefill.min, prefillSecondsMax: prefill.max,
                    peakMemoryMiBMean: memory.mean, peakMemoryMiBSD: memory.sd,
                    peakMemoryMiBMin: memory.min, peakMemoryMiBMax: memory.max,
                    prefillSecondsPerRun: ran.map { ($0.prefillSeconds * 1000).rounded() / 1000 },
                    peakMemoryMiBPerRun: ran.map { (Double($0.peakMemoryBytes) / 1_048_576).rounded() }))
                _ = length
            }
        }

        let report = RepeatedReport(
            modality: ecg ? "ecg" : "eeg", repeats: reports.count, lengths: sweep,
            maxTokens: maxTokens, maxPromptPositions: maxPromptPositions,
            loraApplied: loraApplied, thermalStatePerRun: thermalStates,
            lowPowerModePerRun: lowPowerModes, arms: arms)

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        if let json = try? encoder.encode(report),
           let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first {
            let name = ecg ? "opentslm_repeated_ecg.json" : "opentslm_repeated_eeg.json"
            try? json.write(to: docs.appendingPathComponent(name))
            logger.info("repeated benchmark: wrote \(name, privacy: .public)")
        }

        var lines = ["Repeated context-economics (\(ecg ? "12-lead ECG" : "single-channel EEG"), \(reports.count) runs, greedy, max \(maxTokens) new tokens)",
                     "positions are analytic; TTFT and peak memory are mean +/- SD over runs",
                     "thermal: \(thermalStates.joined(separator: " -> "))",
                     "low power: \(lowPowerModes.map { $0 ? "on" : "off" }.joined(separator: " -> "))", ""]
        lines.append("    L  encoding         pos     TTFT (s)            peak (MiB)")
        for arm in arms {
            let enc = arm.encoding == "soft_prompt" ? "soft" : "text"
            if arm.skipped {
                lines.append(String(format: "%5d  %-8s %7d     n/a (exceeds budget)", arm.seriesLength, (enc as NSString).utf8String!, arm.promptPositions))
                continue
            }
            lines.append(String(format: "%5d  %-8s %7d  %6.2f +/- %.2f   %7.0f +/- %.0f",
                                arm.seriesLength, (enc as NSString).utf8String!, arm.promptPositions,
                                arm.prefillSecondsMean, arm.prefillSecondsSD,
                                arm.peakMemoryMiBMean, arm.peakMemoryMiBSD))
        }
        // Thermal drift shows up as a monotone first->last change, not as scatter.
        if let longest = arms.last(where: { $0.encoding == "soft_prompt" && !$0.skipped }),
           let first = longest.prefillSecondsPerRun.first,
           let last = longest.prefillSecondsPerRun.last, first > 0 {
            lines.append("")
            lines.append(String(format: "soft-prompt TTFT at L=%d across runs: %@  (last/first = %.2fx)",
                                longest.seriesLength,
                                longest.prefillSecondsPerRun.map { String(format: "%.2f", $0) }.joined(separator: ", "),
                                last / first))
        }
        if let unstable = arms.first(where: { !$0.positionsStable }) {
            lines.append("")
            lines.append("WARNING: prompt positions varied across runs at L=\(unstable.seriesLength) (\(unstable.encoding)) — token counts should be deterministic")
        }
        return lines.joined(separator: "\n")
    }

    // MARK: - Sleep accuracy eval

    /// One scored sample of the pinned Sleep-EDF test-split eval.
    struct SleepEvalRow: Encodable {
        let index: Int
        let label: String
        let groundTruth: String?
        let predicted: String?
        let correct: Bool
        let seconds: Double
        let output: String
    }

    /// Full eval result, written to the app's Documents directory.
    struct SleepEvalReport: Encodable {
        let sampleCount: Int
        let maxNewTokens: Int
        let seriesLength: Int
        let loraApplied: Bool
        let parsed: Int
        let correct: Int
        let accuracy: Double
        let rows: [SleepEvalRow]
    }

    /// Extracts the final `Answer: <label>` from generated text, normalized for comparison
    /// (special tokens stripped, whitespace collapsed, trailing period dropped, lowercased).
    static func finalAnswer(in text: String) -> String? {
        // NB: the pipes must be escaped — an unescaped `<|...|>` is regex *alternation*,
        // whose `[^|]*` branch matches the empty string and loops forever.
        let cleaned = text.replacingOccurrences(
            of: "<\\|[^|]*\\|>", with: "", options: .regularExpression)
        guard let marker = cleaned.range(of: "Answer:", options: [.backwards, .caseInsensitive]) else {
            return nil
        }
        let tail = cleaned[marker.upperBound...]
        let line = tail.prefix { $0 != "\n" && $0 != "." }
        let collapsed = line
            .split(whereSeparator: { $0.isWhitespace })
            .joined(separator: " ")
            .lowercased()
        return collapsed.isEmpty ? nil : collapsed
    }

    /// Runs the on-device accuracy eval over the pinned 20-row Sleep-EDF test-split export.
    /// Row `i` here is the same sample as index `i` in the Python reference's test split, so
    /// the per-sample predictions are directly comparable to the PyTorch/MLX reference runs.
    /// Writes `opentslm_sleep_eval.json` to Documents and returns a short summary.
    func runSleepAccuracyEval(
        sampleCount: Int = 20,
        maxTokens: Int = 200,
        llmRunner: LLMRunner,
        llmSession: LLMLocalSession?
    ) async throws -> String {
        guard let llmSession else {
            throw NSError(domain: "OpenTSLMInferenceService", code: 6,
                          userInfo: [NSLocalizedDescriptionKey: "An LLM session is required to run the eval."])
        }
        guard let csvURL = resolveAssetURL(
            overridePath: Constants.openTSLMSleepCSVPath,
            bundledName: Constants.openTSLMSleepEvalCSVName,
            fileExtension: "csv"
        ) else {
            throw NSError(domain: "OpenTSLMInferenceService", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "Missing \(Constants.openTSLMSleepEvalCSVName).csv in bundle/OpenTSLM"])
        }
        guard let encoderURL = resolveAssetURL(
            overridePath: Constants.openTSLMEncoderCheckpointPath,
            bundledName: Constants.openTSLMEncoderCheckpointName,
            fileExtension: "safetensors"
        ), let projectorURL = resolveAssetURL(
            overridePath: Constants.openTSLMProjectorCheckpointPath,
            bundledName: Constants.openTSLMProjectorCheckpointName,
            fileExtension: "safetensors"
        ) else {
            throw NSError(domain: "OpenTSLMInferenceService", code: 2,
                          userInfo: [NSLocalizedDescriptionKey: "Missing sleep encoder/projector checkpoint"])
        }

        // Load the pipeline and apply the LoRA once, then reuse across all samples.
        let pipeline = OpenTSLMSPPipeline(hiddenSize: 2048)
        try pipeline.loadWeights(encoderURL: encoderURL, projectorURL: projectorURL)
        let loraApplied = await applyLoRAIfAvailable(on: llmSession)
        GPU.clearCache()

        let openTSLMLLM = OpenTSLMLLM(llmRunner: llmRunner, session: llmSession)
        var rows: [SleepEvalRow] = []
        var correct = 0
        var parsed = 0
        var seriesLength = 0

        for index in 0 ..< max(sampleCount, 0) {
            let sample = cappedSample(
                try SleepEDFDataset.rawSample(csvURL: csvURL, rowIndex: index),
                limit: Constants.openTSLMSleepMaxTimeSeriesLength
            )
            seriesLength = sample.timeSeries.first?.count ?? 0
            let projected = pipeline.projectSample(sample)
            GPU.clearCache()

            let start = DispatchTime.now()
            let output = try await openTSLMLLM.generate(
                prePrompt: sample.prePrompt,
                timeSeriesText: sample.timeSeriesText,
                timeSeriesEmbeddings: projected,
                postPrompt: sample.postPrompt,
                maxTokens: maxTokens
            )
            let seconds = Double(DispatchTime.now().uptimeNanoseconds - start.uptimeNanoseconds) / 1_000_000_000
            GPU.clearCache()

            let groundTruth = Self.finalAnswer(in: sample.answer)
            let predicted = Self.finalAnswer(in: output)
            let isCorrect = predicted != nil && groundTruth != nil && predicted == groundTruth
            if predicted != nil { parsed += 1 }
            if isCorrect { correct += 1 }

            rows.append(SleepEvalRow(
                index: index, label: sample.label, groundTruth: groundTruth,
                predicted: predicted, correct: isCorrect,
                seconds: (seconds * 100).rounded() / 100, output: output))
            logger.info("sleep eval [\(index + 1, privacy: .public)/\(sampleCount, privacy: .public)] gt=\(groundTruth ?? "nil", privacy: .public) pred=\(predicted ?? "nil", privacy: .public) \(isCorrect ? "OK" : "X", privacy: .public)")
        }

        let total = max(rows.count, 1)
        let report = SleepEvalReport(
            sampleCount: rows.count, maxNewTokens: maxTokens, seriesLength: seriesLength,
            loraApplied: loraApplied, parsed: parsed, correct: correct,
            accuracy: (Double(correct) / Double(total) * 10_000).rounded() / 10_000, rows: rows)

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        if let json = try? encoder.encode(report),
           let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first {
            let url = docs.appendingPathComponent("opentslm_sleep_eval.json")
            try? json.write(to: url)
            logger.info("sleep eval: wrote \(url.path, privacy: .public)")
        }

        var lines = ["Sleep-EDF accuracy eval (pinned test-split rows 0..<\(rows.count), L=\(seriesLength), greedy, max \(maxTokens) new tokens)",
                     "LoRA applied: \(loraApplied ? "yes" : "no")", ""]
        for row in rows {
            let gt = (row.groundTruth ?? "nil").padding(toLength: 16, withPad: " ", startingAt: 0)
            let pred = (row.predicted ?? "nil").padding(toLength: 16, withPad: " ", startingAt: 0)
            lines.append("\(row.index)  \(gt) -> \(pred) \(row.correct ? "OK" : "X")")
        }
        lines.append("")
        lines.append("accuracy: \(correct)/\(rows.count)   parsed: \(parsed)/\(rows.count)")
        return lines.joined(separator: "\n")
    }

    /// ECG variant of the accuracy eval: iterates the bundled ECG-QA CoT test rows, scores the
    /// generated `Answer:` against the row's ground truth, and writes `opentslm_ecg_eval.json`.
    func runECGAccuracyEval(
        sampleCount: Int = 20,
        // OpenTSLM's own prediction scripts use 300-400 (and its curriculum eval loop 2000);
        // 200 truncated ECG rationales before they reached `Answer:`, scoring them as wrong.
        maxTokens: Int = 400,
        llmRunner: LLMRunner,
        llmSession: LLMLocalSession?
    ) async throws -> String {
        guard let llmSession else {
            throw NSError(domain: "OpenTSLMInferenceService", code: 6,
                          userInfo: [NSLocalizedDescriptionKey: "An LLM session is required to run the eval."])
        }
        guard let encoderURL = resolveAssetURL(
            overridePath: Constants.openTSLMEncoderCheckpointPath,
            bundledName: Constants.openTSLMECGEncoderCheckpointName,
            fileExtension: "safetensors"
        ), let projectorURL = resolveAssetURL(
            overridePath: Constants.openTSLMProjectorCheckpointPath,
            bundledName: Constants.openTSLMECGProjectorCheckpointName,
            fileExtension: "safetensors"
        ) else {
            throw NSError(domain: "OpenTSLMInferenceService", code: 2,
                          userInfo: [NSLocalizedDescriptionKey: "Missing ECG encoder/projector checkpoint"])
        }

        let pipeline = OpenTSLMSPPipeline(hiddenSize: 2048)
        try pipeline.loadWeights(encoderURL: encoderURL, projectorURL: projectorURL)
        let loraApplied = await applyLoRAIfAvailable(on: llmSession, checkpointName: Constants.openTSLMECGLoRACheckpointName)
        GPU.clearCache()

        let openTSLMLLM = OpenTSLMLLM(llmRunner: llmRunner, session: llmSession)
        var rows: [SleepEvalRow] = []
        var correct = 0
        var parsed = 0
        var seriesLength = 0

        for index in 0 ..< max(sampleCount, 0) {
            let loaded = try loadECGQACoTSample(split: .test, sampleIndex: index)
            guard loaded.metadata.loader != "formatted_json" else {
                throw NSError(domain: "OpenTSLMInferenceService", code: 16, userInfo: [NSLocalizedDescriptionKey:
                    "ECG eval fell back to the single legacy JSON sample — the bundled ECG-QA CSV/waveforms were not found, so every row would be identical."])
            }
            let sample = cappedSample(loaded.sample)
            seriesLength = sample.timeSeries.first?.count ?? 0
            let projected = pipeline.projectSample(sample)
            GPU.clearCache()

            let start = DispatchTime.now()
            let output = try await openTSLMLLM.generate(
                prePrompt: sample.prePrompt,
                timeSeriesText: sample.timeSeriesText,
                timeSeriesEmbeddings: projected,
                postPrompt: sample.postPrompt,
                maxTokens: maxTokens
            )
            let seconds = Double(DispatchTime.now().uptimeNanoseconds - start.uptimeNanoseconds) / 1_000_000_000
            GPU.clearCache()

            // ECG ground truth is the row's `correct_answer` (sample.label), not a rationale tail.
            let groundTruth = sample.label.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            let predicted = Self.finalAnswer(in: output)
            let isCorrect = predicted != nil && predicted == groundTruth
            if predicted != nil { parsed += 1 }
            if isCorrect { correct += 1 }

            rows.append(SleepEvalRow(
                index: index, label: sample.label, groundTruth: groundTruth,
                predicted: predicted, correct: isCorrect,
                seconds: (seconds * 100).rounded() / 100, output: output))
            logger.info("ecg eval [\(index + 1, privacy: .public)/\(sampleCount, privacy: .public)] gt=\(groundTruth, privacy: .public) pred=\(predicted ?? "nil", privacy: .public) \(isCorrect ? "OK" : "X", privacy: .public)")
        }

        let total = max(rows.count, 1)
        let report = SleepEvalReport(
            sampleCount: rows.count, maxNewTokens: maxTokens, seriesLength: seriesLength,
            loraApplied: loraApplied, parsed: parsed, correct: correct,
            accuracy: (Double(correct) / Double(total) * 10_000).rounded() / 10_000, rows: rows)

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        if let json = try? encoder.encode(report),
           let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first {
            let url = docs.appendingPathComponent("opentslm_ecg_eval.json")
            try? json.write(to: url)
            logger.info("ecg eval: wrote \(url.path, privacy: .public)")
        }

        var lines = ["ECG-QA accuracy eval (bundled test rows 0..<\(rows.count), L=\(seriesLength), greedy, max \(maxTokens) new tokens)",
                     "LoRA applied: \(loraApplied ? "yes" : "no")", ""]
        for row in rows {
            let gt = (row.groundTruth ?? "nil").padding(toLength: 22, withPad: " ", startingAt: 0)
            let pred = (row.predicted ?? "nil").padding(toLength: 22, withPad: " ", startingAt: 0)
            lines.append("\(row.index)  \(gt) -> \(pred) \(row.correct ? "OK" : "X")")
        }
        lines.append("")
        lines.append("accuracy: \(correct)/\(rows.count)   parsed: \(parsed)/\(rows.count)")
        return lines.joined(separator: "\n")
    }

    /// Runs the §4 context-economics benchmark: the same Sleep-EDF sample and the
    /// same on-device backbone, encoded as soft prompts vs. OpenTSLM text
    /// serialization, swept over series length. Reports prompt positions, end-to-end
    /// latency, decode tokens/s, peak GPU memory, and the context-window ceiling for
    /// each encoding; writes the full JSON to Documents and returns a summary string.
    func runContextEconomicsBenchmark(
        split: SleepEDFDataset.Split = .test,
        sampleIndex: Int = 0,
        lengths: [Int] = [100, 250, 500, 1000, 1500],
        maxTokens: Int = 200,
        contextWindow: Int = 8192,
        maxPromptPositions: Int = 2048,
        llmRunner: LLMRunner,
        llmSession: LLMLocalSession?
    ) async throws -> String {
        guard let llmSession else {
            throw NSError(domain: "OpenTSLMInferenceService", code: 6, userInfo: [NSLocalizedDescriptionKey: "An LLM session is required to run the benchmark."])
        }
        guard let csvURL = resolveAssetURL(
            overridePath: Constants.openTSLMSleepCSVPath,
            bundledName: Constants.openTSLMSleepCSVName,
            fileExtension: "csv"
        ) else {
            throw NSError(domain: "OpenTSLMInferenceService", code: 1, userInfo: [NSLocalizedDescriptionKey: "Missing sleep CSV in bundle/OpenTSLM or override path"])
        }
        guard let encoderURL = resolveAssetURL(
            overridePath: Constants.openTSLMEncoderCheckpointPath,
            bundledName: Constants.openTSLMEncoderCheckpointName,
            fileExtension: "safetensors"
        ) else {
            throw NSError(domain: "OpenTSLMInferenceService", code: 2, userInfo: [NSLocalizedDescriptionKey: "Missing encoder checkpoint in bundle/OpenTSLM or override path"])
        }
        guard let projectorURL = resolveAssetURL(
            overridePath: Constants.openTSLMProjectorCheckpointPath,
            bundledName: Constants.openTSLMProjectorCheckpointName,
            fileExtension: "safetensors"
        ) else {
            throw NSError(domain: "OpenTSLMInferenceService", code: 3, userInfo: [NSLocalizedDescriptionKey: "Missing projector checkpoint in bundle/OpenTSLM or override path"])
        }

        // Use a fixed raw CSV row (no split) — the parity-verified sample whose generation
        // matches the Python reference — so the benchmark's decoded text is coherent and
        // deterministic. Cost metrics (positions/prefill/memory) are sample-independent.
        let safeIndex = max(sampleIndex, 0)
        let sample = try SleepEDFDataset.rawSample(csvURL: csvURL, rowIndex: safeIndex)

        let pipeline = OpenTSLMSPPipeline(hiddenSize: 2048)
        try pipeline.loadWeights(encoderURL: encoderURL, projectorURL: projectorURL)

        // Apply the SP LoRA to the shared backbone (changes generated content, not
        // prefill/throughput/memory — see OpenTSLMBenchmark header).
        let loraApplied = await applyLoRAIfAvailable(on: llmSession)
        GPU.clearCache()

        let benchmark = OpenTSLMBenchmark(pipeline: pipeline, llmRunner: llmRunner, session: llmSession)
        let report = try await benchmark.run(
            sample: sample,
            lengths: lengths,
            maxTokens: maxTokens,
            contextWindow: contextWindow,
            maxPromptPositions: maxPromptPositions,
            loraApplied: loraApplied)

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        if let json = try? encoder.encode(report) {
            if let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first {
                let url = docs.appendingPathComponent("opentslm_context_economics.json")
                try? json.write(to: url)
                logger.info("benchmark: wrote \(url.path, privacy: .public)")
            }
            print("=== opentslm_context_economics.json ===\n\(String(decoding: json, as: UTF8.self))\n=== end opentslm_context_economics.json ===")
        }

        return report.formattedSummary(split: "raw-row", sampleIndex: safeIndex)
    }

    /// ECG variant of the §4 benchmark: the 12-lead ECG-QA CoT sample swept over per-lead
    /// length, same soft-prompt vs. text-serialized comparison on the ECG-tuned backbone.
    /// 12 leads use ~12× the positions of the 1-lead sleep case, so the soft-prompt arm
    /// relies on ``MLXEmbeddingGenerator``'s chunked prefill to avoid OOM at longer lengths.
    func runContextEconomicsBenchmarkECG(
        split: ECGQACoTDataset.Split = .test,
        sampleIndex: Int = 0,
        lengths: [Int] = [100, 250, 500, 1000],
        maxTokens: Int = 200,
        contextWindow: Int = 8192,
        maxPromptPositions: Int = 4096,
        llmRunner: LLMRunner,
        llmSession: LLMLocalSession?
    ) async throws -> String {
        guard let llmSession else {
            throw NSError(domain: "OpenTSLMInferenceService", code: 6, userInfo: [NSLocalizedDescriptionKey: "An LLM session is required to run the benchmark."])
        }
        guard let encoderURL = resolveAssetURL(
            overridePath: Constants.openTSLMEncoderCheckpointPath,
            bundledName: Constants.openTSLMECGEncoderCheckpointName,
            fileExtension: "safetensors"
        ) else {
            throw NSError(domain: "OpenTSLMInferenceService", code: 2, userInfo: [NSLocalizedDescriptionKey: "Missing ECG encoder checkpoint in bundle/OpenTSLM or override path"])
        }
        guard let projectorURL = resolveAssetURL(
            overridePath: Constants.openTSLMProjectorCheckpointPath,
            bundledName: Constants.openTSLMECGProjectorCheckpointName,
            fileExtension: "safetensors"
        ) else {
            throw NSError(domain: "OpenTSLMInferenceService", code: 3, userInfo: [NSLocalizedDescriptionKey: "Missing ECG projector checkpoint in bundle/OpenTSLM or override path"])
        }

        let loaded = try loadECGQACoTSample(split: split, sampleIndex: sampleIndex)
        let sample = loaded.sample

        // Encoder built with default maxPatches; loadWeights replaces pos_embed with the
        // ECG checkpoint's (1024), so the default pipeline works for ECG too.
        let pipeline = OpenTSLMSPPipeline(hiddenSize: 2048)
        try pipeline.loadWeights(encoderURL: encoderURL, projectorURL: projectorURL)

        let loraApplied = await applyLoRAIfAvailable(on: llmSession, checkpointName: Constants.openTSLMECGLoRACheckpointName)
        GPU.clearCache()

        let benchmark = OpenTSLMBenchmark(pipeline: pipeline, llmRunner: llmRunner, session: llmSession)
        let report = try await benchmark.run(
            sample: sample,
            lengths: lengths,
            maxTokens: maxTokens,
            contextWindow: contextWindow,
            maxPromptPositions: maxPromptPositions,
            loraApplied: loraApplied)

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        if let json = try? encoder.encode(report) {
            if let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first {
                let url = docs.appendingPathComponent("opentslm_context_economics_ecg.json")
                try? json.write(to: url)
                logger.info("benchmark(ecg): wrote \(url.path, privacy: .public)")
            }
            // Dump full JSON to the Xcode console so all fields (prefill latency, tokens,
            // output text, exact bytes) are copyable without pulling the file off-device.
            print("=== opentslm_context_economics_ecg.json ===\n\(String(decoding: json, as: UTF8.self))\n=== end opentslm_context_economics_ecg.json ===")
        }

        return report.formattedSummary(split: "ecg-\(split.rawValue)", sampleIndex: sampleIndex)
    }

    /// Resolves the encoder/projector checkpoints, loads the SP pipeline, and projects the
    /// sample's time series to LLM-hidden-size embeddings.
    /// Defaults to the sleep checkpoints; the ECG paths pass the `.ecg` names.
    private func loadPipelineAndProject(
        _ sample: OpenTSLMSPSample,
        encoderName: String = Constants.openTSLMEncoderCheckpointName,
        projectorName: String = Constants.openTSLMProjectorCheckpointName
    ) throws -> [MLXArray] {
        guard let encoderURL = resolveAssetURL(
            overridePath: Constants.openTSLMEncoderCheckpointPath,
            bundledName: encoderName,
            fileExtension: "safetensors"
        ) else {
            throw NSError(domain: "OpenTSLMInferenceService", code: 2, userInfo: [NSLocalizedDescriptionKey: "Missing encoder checkpoint '\(encoderName)' in bundle/OpenTSLM or override path"])
        }
        guard let projectorURL = resolveAssetURL(
            overridePath: Constants.openTSLMProjectorCheckpointPath,
            bundledName: projectorName,
            fileExtension: "safetensors"
        ) else {
            throw NSError(domain: "OpenTSLMInferenceService", code: 3, userInfo: [NSLocalizedDescriptionKey: "Missing projector checkpoint '\(projectorName)' in bundle/OpenTSLM or override path"])
        }

        let pipeline = OpenTSLMSPPipeline(hiddenSize: 2048)
        try pipeline.loadWeights(encoderURL: encoderURL, projectorURL: projectorURL)
        let projected = pipeline.projectSample(sample)
        guard let first = projected.first else {
            throw NSError(domain: "OpenTSLMInferenceService", code: 5, userInfo: [NSLocalizedDescriptionKey: "Projection returned no tensors"])
        }
        eval(first)
        GPU.clearCache()
        return projected
    }

    /// Defaults to the sleep LoRA checkpoint; the ECG paths pass `Constants.openTSLMECGLoRACheckpointName`.
    private func applyLoRAIfAvailable(
        on llmSession: LLMLocalSession,
        checkpointName: String = Constants.openTSLMLoRACheckpointName
    ) async -> Bool {
        do {
            try await OpenTSLMLoRA.applyIfNeeded(on: llmSession, checkpointName: checkpointName)
            return true
        } catch {
            logger.warning("OpenTSLM LoRA apply failed: \(error.localizedDescription, privacy: .public)")
            return false
        }
    }

    private func resolveAssetURL(overridePath: String, bundledName: String, fileExtension: String) -> URL? {
        let fileManager = FileManager.default

        if !overridePath.isEmpty {
            let expandedPath = NSString(string: overridePath).expandingTildeInPath
            let overrideURL = URL(fileURLWithPath: expandedPath)
            if fileManager.fileExists(atPath: overrideURL.path) {
                return overrideURL
            }
        }

        if let bundledURL = Bundle.main.url(
            forResource: bundledName,
            withExtension: fileExtension,
            subdirectory: Constants.openTSLMBundleSubdirectory
        ) {
            return bundledURL
        }

        return Bundle.main.url(forResource: bundledName, withExtension: fileExtension)
    }

    private func resolveLocalModelDirectory() -> URL? {
        let fileManager = FileManager.default

        // Check for override path
        if let overridePath = Constants.localModelSourcePathOverride,
           let overrideURL = existingDirectoryURL(at: overridePath, fileManager: fileManager) {
            return overrideURL
        }

        // Check bundled model directory
        if let bundledLocalModelURL = Bundle.main.resourceURL {
            let bundledDirectory = bundledLocalModelURL.appendingPathComponent(Constants.localModelBundleSubdirectory, isDirectory: true)
            if fileManager.fileExists(atPath: bundledDirectory.path) {
                return bundledDirectory
            }

            // Check if model files are directly in bundle root
            let requiredModelFiles = ["config.json", "tokenizer.json", "model.safetensors"]
            let bundledRootFiles = requiredModelFiles.allSatisfy { fileName in
                fileManager.fileExists(atPath: bundledLocalModelURL.appendingPathComponent(fileName).path)
            }

            if bundledRootFiles {
                return bundledLocalModelURL
            }
        }

        // Fallback: detect a downloaded model snapshot from Hugging Face cache
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

        return directoryContents?
            .filter { url in
                var isDirectory: ObjCBool = false
                return fileManager.fileExists(atPath: url.path, isDirectory: &isDirectory) && isDirectory.boolValue
            }
            .sorted(by: { lhs, rhs in
                let lhsDate = (try? lhs.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
                let rhsDate = (try? rhs.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
                return lhsDate > rhsDate
            })
            .first
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

    private func cappedSample(_ sample: OpenTSLMSPSample, limit: Int = Constants.openTSLMMaxTimeSeriesLength) -> OpenTSLMSPSample {
        let cap = limit
        guard cap > 0 else { return sample }

        let cappedSeries = sample.timeSeries.map { series in
            series.count > cap ? Array(series.prefix(cap)) : series
        }
        guard cappedSeries != sample.timeSeries else { return sample }

        return OpenTSLMSPSample(
            prePrompt: sample.prePrompt,
            timeSeriesText: sample.timeSeriesText,
            timeSeries: cappedSeries,
            postPrompt: sample.postPrompt,
            label: sample.label,
            answer: sample.answer
        )
    }

    private func openTSLMSampleExtraLines(
        split: SleepEDFDataset.Split,
        sampleIndex: Int,
        sample: OpenTSLMSPSample,
        embeddingsShape: [Int],
        loraApplied: Bool
    ) -> [String] {
        [
            "split: \(split.rawValue)",
            "sample_index: \(sampleIndex)",
            "series_count: \(sample.timeSeries.count)",
            "max_series_length: \(Constants.openTSLMSleepMaxTimeSeriesLength)",
            "embeddings_shape: \(embeddingsShape)",
            "lora_checkpoint_found: \(OpenTSLMLoRA.resolveLoRAURL() != nil)",
            "lora_applied: \(loraApplied ? "yes" : "no")",
            "llm_model: \(Constants.llmModelName)",
            "llm_integration: \(Constants.openTSLMRunSampleLLMGeneration ? "generate" : "encoder+lora-only")",
        ]
    }

    private func ecgQACoTSampleExtraLines(
        metadata: ECGQACoTSampleMetadata,
        embeddingsShape: [Int],
        loraApplied: Bool
    ) -> [String] {
        [
            "source: \(metadata.source)",
            "loader: \(metadata.loader)",
            "split: \(metadata.split)",
            "sample_index: \(metadata.sampleIndex)",
            "ecg_id: \(metadata.ecgId)",
            "template_id: \(metadata.templateId)",
            "question_type: \(metadata.questionType)",
            "series_count: \(metadata.seriesCount)",
            "samples_per_lead: \(metadata.samplesPerLead)",
            "max_series_length: \(Constants.openTSLMMaxTimeSeriesLength)",
            "skip_llm_load: \(Constants.skipLLMLoad ? "yes" : "no")",
            "run_llm: \(Constants.openTSLMRunSampleLLMGeneration ? "yes" : "no")",
            "embeddings_shape: \(embeddingsShape)",
            "lora_checkpoint_found: \(OpenTSLMLoRA.resolveLoRAURL(checkpointName: Constants.openTSLMECGLoRACheckpointName) != nil)",
            "lora_applied: \(loraApplied ? "yes" : "no")",
        ]
    }

    private struct LoadedECGQACoTSample {
        let sample: OpenTSLMSPSample
        let metadata: ECGQACoTSampleMetadata
    }

    private struct ECGQACoTSampleMetadata {
        let source: String
        let loader: String
        let split: String
        let sampleIndex: Int
        let ecgId: String
        let templateId: String
        let questionType: String
        let seriesCount: Int
        let samplesPerLead: Int
    }

    /// Loads one ECG-QA CoT row via ``ECGQACoTDataset`` (CSV metadata + PTB-XL waveform sidecar).
    private func loadECGQACoTSample(
        split: ECGQACoTDataset.Split,
        sampleIndex: Int
    ) throws -> LoadedECGQACoTSample {
        if let csvURL = resolveECGCoTCSVURL(split: split),
           let waveformsDirectory = resolveECGWaveformsDirectoryURL() {
            let dataset = try ECGQACoTDataset(
                csvURL: csvURL,
                waveformsDirectory: waveformsDirectory,
                templateAnswersURL: resolveECGTemplateAnswersURL(),
                split: split,
                maxRows: max(sampleIndex + 1, 1)
            )
            guard dataset.count > 0 else {
                throw NSError(domain: "OpenTSLMInferenceService", code: 15, userInfo: [NSLocalizedDescriptionKey: "ECG-QA CoT CSV has no rows"])
            }

            let safeIndex = min(max(sampleIndex, 0), dataset.count - 1)
            let sample = try dataset.sample(at: safeIndex)
            let row = dataset.rowMetadata(at: safeIndex)

            return LoadedECGQACoTSample(
                sample: sample,
                metadata: ECGQACoTSampleMetadata(
                    source: "ecg_qa_cot",
                    loader: "ECGQACoTDataset",
                    split: row.split.rawValue,
                    sampleIndex: safeIndex,
                    ecgId: String(row.ecgId),
                    templateId: String(row.templateId),
                    questionType: row.questionType,
                    seriesCount: sample.timeSeries.count,
                    samplesPerLead: sample.timeSeries.first?.count ?? 0
                )
            )
        }

        // Legacy fallback: monolithic JSON exported by inference_ecg.py --export-json.
        let legacy = try loadECGQACoTFormattedSample()
        return LoadedECGQACoTSample(
            sample: legacy.sample,
            metadata: ECGQACoTSampleMetadata(
                source: legacy.info.source,
                loader: "formatted_json",
                split: legacy.info.split,
                sampleIndex: legacy.info.sampleIndex,
                ecgId: legacy.info.ecgId,
                templateId: legacy.info.templateId,
                questionType: "unknown",
                seriesCount: legacy.info.seriesCount,
                samplesPerLead: legacy.info.samplesPerLead
            )
        )
    }

    private func resolveECGCoTCSVURL(split: ECGQACoTDataset.Split) -> URL? {
        if !Constants.openTSLMECGCoTCSVPath.isEmpty {
            let expandedPath = NSString(string: Constants.openTSLMECGCoTCSVPath).expandingTildeInPath
            let overrideURL = URL(fileURLWithPath: expandedPath)
            if FileManager.default.fileExists(atPath: overrideURL.path) {
                return overrideURL
            }
        }

        if split == .test {
            return resolveAssetURL(
                overridePath: "",
                bundledName: Constants.openTSLMECGCoTTestCSVName,
                fileExtension: "csv"
            )
        }

        return resolveAssetURL(
            overridePath: "",
            bundledName: split.csvBaseName,
            fileExtension: "csv"
        )
    }

    private func resolveECGWaveformsDirectoryURL() -> URL? {
        if !Constants.openTSLMECGWaveformsPath.isEmpty {
            let expandedPath = NSString(string: Constants.openTSLMECGWaveformsPath).expandingTildeInPath
            var isDirectory: ObjCBool = false
            if FileManager.default.fileExists(atPath: expandedPath, isDirectory: &isDirectory), isDirectory.boolValue {
                return URL(fileURLWithPath: expandedPath, isDirectory: true)
            }
        }

        if let bundledURL = Bundle.main.url(
            forResource: Constants.openTSLMECGWaveformsDirectoryName,
            withExtension: nil,
            subdirectory: Constants.openTSLMBundleSubdirectory
        ) {
            return bundledURL
        }

        if let named = Bundle.main.url(forResource: Constants.openTSLMECGWaveformsDirectoryName, withExtension: nil) {
            return named
        }

        // Xcode synced folders flatten `Supporting Files/OpenTSLM/ecg_qa_waveforms/*.json` to the
        // bundle root, so the named directory does not exist in the built app. Fall back to the
        // resource root when it contains the `<ecg_id>.json` waveform files.
        if let resourceURL = Bundle.main.resourceURL,
           let entries = try? FileManager.default.contentsOfDirectory(atPath: resourceURL.path),
           entries.contains(where: { $0.hasSuffix(".json") && (Int($0.dropLast(5)) != nil) }) {
            return resourceURL
        }

        return nil
    }

    private func resolveECGTemplateAnswersURL() -> URL? {
        return resolveAssetURL(
            overridePath: "",
            bundledName: Constants.openTSLMECGTemplateAnswersName,
            fileExtension: "json"
        )
    }

    private struct LoadedECGQACoTFormattedSample {
        let sample: OpenTSLMSPSample
        let info: ECGQACoTFormattedSampleInfo
    }

    private struct ECGQACoTFormattedSampleInfo {
        let source: String
        let split: String
        let sampleIndex: Int
        let ecgId: String
        let templateId: String
        let seriesCount: Int
        let samplesPerLead: Int
    }

    /// Legacy formatted JSON export from ``inference_ecg.py --export-json``.
    private func loadECGQACoTFormattedSample() throws -> LoadedECGQACoTFormattedSample {
        guard let url = resolveECGQACoTSampleURL() else {
            throw NSError(
                domain: "OpenTSLMInferenceService",
                code: 9,
                userInfo: [
                    NSLocalizedDescriptionKey:
                        "Missing ECG-QA CoT sample JSON. Bundle \(Constants.openTSLMECGQACoTSampleName).json "
                        + "or set HEALTHYLLM_OPEN_TSLM_ECG_JSON to the file exported by inference_ecg.py --export-json.",
                ]
            )
        }

        let data = try Data(contentsOf: url)
        let decoded = try JSONSerialization.jsonObject(with: data)
        guard let object = decoded as? [String: Any] else {
            throw NSError(domain: "OpenTSLMInferenceService", code: 10, userInfo: [NSLocalizedDescriptionKey: "ECG-QA JSON root must be an object"])
        }

        guard let prePrompt = object["pre_prompt"] as? String,
              let postPrompt = object["post_prompt"] as? String,
              let timeSeriesText = object["time_series_text"] as? [String],
              let label = object["label"] as? String,
              let answer = object["answer"] as? String
        else {
            throw NSError(
                domain: "OpenTSLMInferenceService",
                code: 11,
                userInfo: [NSLocalizedDescriptionKey: "ECG-QA JSON missing pre_prompt/post_prompt/time_series_text/label/answer"]
            )
        }

        guard let rawSeries = object["time_series"] as? [[Any]], !rawSeries.isEmpty else {
            throw NSError(domain: "OpenTSLMInferenceService", code: 12, userInfo: [NSLocalizedDescriptionKey: "ECG-QA JSON missing time_series"])
        }

        let timeSeries: [[Float]] = try rawSeries.map { lead in
            guard !lead.isEmpty else {
                throw NSError(domain: "OpenTSLMInferenceService", code: 13, userInfo: [NSLocalizedDescriptionKey: "ECG-QA JSON has empty lead"])
            }
            return try lead.map { value in
                if let f = value as? Float { return f }
                if let d = value as? Double { return Float(d) }
                if let i = value as? Int { return Float(i) }
                throw NSError(domain: "OpenTSLMInferenceService", code: 14, userInfo: [NSLocalizedDescriptionKey: "Invalid time_series value"])
            }
        }

        let samplesPerLead = timeSeries.first?.count ?? 0
        let info = ECGQACoTFormattedSampleInfo(
            source: object["source"] as? String ?? "ecg_qa_cot",
            split: object["split"] as? String ?? "unknown",
            sampleIndex: object["sample_idx"] as? Int ?? -1,
            ecgId: String(describing: object["ecg_id"] ?? "unknown"),
            templateId: String(describing: object["template_id"] ?? "unknown"),
            seriesCount: timeSeries.count,
            samplesPerLead: samplesPerLead
        )

        let sample = OpenTSLMSPSample(
            prePrompt: prePrompt,
            timeSeriesText: timeSeriesText,
            timeSeries: timeSeries,
            postPrompt: postPrompt,
            label: label,
            answer: answer
        )
        return LoadedECGQACoTFormattedSample(sample: sample, info: info)
    }

    private func resolveECGQACoTSampleURL() -> URL? {
        if let override = resolveECGJSONURL() {
            return override
        }

        return resolveAssetURL(
            overridePath: "",
            bundledName: Constants.openTSLMECGQACoTSampleName,
            fileExtension: "json"
        )
    }

    private func loadECGSample() throws -> ECGSample {
        if let url = resolveECGJSONURL() {
            return try ECGSample.load(from: url)
        }

        return ECGSample.hardcoded(sampleLength: Constants.hardcodedECGSampleLength)
    }

    private func resolveECGJSONURL() -> URL? {
        let overridePath = Constants.openTSLMECGJSONPath
        guard !overridePath.isEmpty else {
            return nil
        }

        let fileManager = FileManager.default
        let expandedPath = NSString(string: overridePath).expandingTildeInPath
        let overrideURL = URL(fileURLWithPath: expandedPath)
        guard fileManager.fileExists(atPath: overrideURL.path) else {
            return nil
        }

        return overrideURL
    }

    // PTB-XL / ECG-QA standard 12-lead order — must match what the encoder was
    // trained to receive in the per-lead text labels.
    private static let ecgLeadNames = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

    private static let ecgTargetSamplingRate: Double = 100  // Hz — matches ECG-QA training (`[::5]` from 500 Hz)
    /// Window length per lead. Capped to the global series-length budget so we don't
    /// generate samples we'd only truncate; ECG-QA training was on 10 s windows but
    /// 12 × 1000 over-runs the iOS 6 GB process limit at prefill time.
    private static var ecgSamplesPerLead: Int { Constants.openTSLMMaxTimeSeriesLength }

    private func makeOpenTSLMSample(from ecg: ECGSample) -> OpenTSLMSPSample {
        // Downsample to 100 Hz and window to 1000 samples (10 s) to match ECG-QA training.
        let downsampled = Self.downsampleAndWindow(
            ecg.voltages,
            inputRate: ecg.samplingFrequency,
            outputRate: Self.ecgTargetSamplingRate,
            count: Self.ecgSamplesPerLead
        )
        let stats = Self.statistics(of: downsampled)
        let normalizedLead = downsampled.map { Float(($0 - stats.mean) / stats.std) }

        // Apple Watch is hardware-single-lead — replicate Lead I across all 12 channels
        // so the encoder receives the 12-series shape it was trained on. The per-lead
        // mean/std in each label are therefore identical, by construction; that's an
        // acknowledged distribution shift vs. real PTB-XL multi-lead recordings.
        var timeSeries: [[Float]] = []
        timeSeries.reserveCapacity(Self.ecgLeadNames.count)
        var timeSeriesText: [String] = []
        timeSeriesText.reserveCapacity(Self.ecgLeadNames.count)
        for name in Self.ecgLeadNames {
            timeSeries.append(normalizedLead)
            timeSeriesText.append("This is ECG Lead \(name), it has mean \(String(format: "%.4f", stats.mean)) and std \(String(format: "%.4f", stats.std)):")
        }

        return OpenTSLMSPSample(
            prePrompt: ECGQACoTDataset.prePrompt(
                clinicalContext: ecg.clinicalContext,
                question: ecg.question
            ),
            timeSeriesText: timeSeriesText,
            timeSeries: timeSeries,
            postPrompt: ECGQACoTDataset.postPrompt(possibleAnswers: ecg.possibleAnswers),
            label: ecg.classification ?? "unknown",
            answer: ecg.summary
        )
    }

    /// Verbatim from OpenTSLM `src/time_series_datasets/ecg_qa/ECGQACoTQADataset.py:_get_pre_prompt`,
    /// with the runtime-supplied `clinical_context` and `question` substituted.

    /// Verbatim from `ECGQACoTQADataset.py:_get_post_prompt`. Two branches:
    ///   - if `possibleAnswers` is non-empty → the templated branch, which lists the closed
    ///     answer set the model was trained to pick from.
    ///   - otherwise → the open-ended branch.
    /// Both end with literal `"Answer: ` (open quote + trailing space, no closing quote) —
    /// the model continues from there with its rationale and concludes `Answer: <label>`.

    /// Nearest-neighbor resampling to `count` samples at `outputRate`, starting at t=0.
    /// Pads with zeros if the source is shorter than the requested window.
    private static func downsampleAndWindow(
        _ voltages: [Double],
        inputRate: Double,
        outputRate: Double,
        count: Int
    ) -> [Double] {
        guard !voltages.isEmpty, inputRate > 0, outputRate > 0, count > 0 else {
            return Array(repeating: 0, count: max(count, 0))
        }
        let step = inputRate / outputRate
        var out: [Double] = []
        out.reserveCapacity(count)
        for i in 0 ..< count {
            let srcIdx = Int(Double(i) * step)
            out.append(srcIdx < voltages.count ? voltages[srcIdx] : 0)
        }
        return out
    }

    private static func statistics(of values: [Double]) -> (mean: Double, std: Double) {
        guard !values.isEmpty else { return (0, 1) }
        let mean = values.reduce(0, +) / Double(values.count)
        let variance = values.reduce(0) { partial, value in
            let d = value - mean
            return partial + d * d
        } / Double(values.count)
        return (mean, max(sqrt(variance), 1e-6))
    }

    private func formatSampleReport(
        title: String,
        prePrompt: String,
        timeSeriesText: [String],
        postPrompt: String,
        label: String,
        answer: String,
        extraLines: [String] = []
    ) -> String {
        var lines: [String] = [
            title,
            "",
            "pre_prompt:",
            prePrompt.trimmingCharacters(in: .whitespacesAndNewlines),
            "",
            "time_series_text:",
        ]

        lines.append(contentsOf: timeSeriesText.map { "- \($0)" })
        lines.append("")
        lines.append("post_prompt:")
        lines.append(postPrompt.trimmingCharacters(in: .whitespacesAndNewlines))

        if !extraLines.isEmpty {
            lines.append("")
            lines.append(contentsOf: extraLines)
        }

        lines.append("")
        lines.append("label: \(label)")
        lines.append("answer: \(answer.isEmpty ? "No Data" : answer)")

        return lines.joined(separator: "\n")
    }

    private static func zNormalize(_ values: [Double]) -> [Double] {
        guard !values.isEmpty else {
            return []
        }

        let mean = values.reduce(0, +) / Double(values.count)
        let variance = values.reduce(0) { partial, value in
            let delta = value - mean
            return partial + delta * delta
        } / Double(values.count)
        let standardDeviation = max(sqrt(variance), 1e-6)
        return values.map { ($0 - mean) / standardDeviation }
    }
}

private enum ECGSampleSource: String {
    case hardcoded
    case healthkit
    case healthkitJSON = "healthkit_json"
    case jsonOverride = "json_override"
}

private struct ECGSample {
    let source: ECGSampleSource
    let samplingFrequency: Double
    let classification: String?
    let symptomsStatus: String?
    let averageHeartRate: Double?
    let voltages: [Double]
    // ECG-QA-style framing for the model's pre/post prompts. Defaulted so existing
    // call sites that don't supply per-sample fields still compile.
    // The default mirrors the most common ECG-QA *closed-set* template (yes/no
    // about a single condition) — ECG-QA was trained heavily on these and greedy
    // decoding picks from the list much more reliably than from an open question.
    var clinicalContext: String = "Single-lead ambulatory ECG recording (Lead I equivalent, replicated across the standard 12 leads)."
    var question: String = "Is the cardiac rhythm shown in this ECG consistent with atrial fibrillation?"
    var possibleAnswers: [String] = ["yes", "no"]

    var sourceDescription: String {
        source.rawValue
    }

    var summary: String {
        let classificationText = classification ?? "unknown"
        let symptomsText = symptomsStatus ?? "unknown"
        let heartRateText = averageHeartRate.map { String(format: "%.1f", $0) } ?? "unknown"
        return "classification=\(classificationText), symptoms_status=\(symptomsText), average_heart_rate=\(heartRateText)"
    }

    static func hardcoded(sampleLength: Int = 1024, samplingFrequency: Double = 256) -> ECGSample {
        let voltages: [Double] = (0 ..< sampleLength).map { index in
            let t = Double(index) / samplingFrequency

            let base = 0.025 * sin(2.0 * .pi * 1.2 * t)
            let pWave = 0.010 * sin(2.0 * .pi * 4.0 * t)
            let qrsPhase = t.truncatingRemainder(dividingBy: 0.86)
            let qrs = qrsPhase < 0.018 ? 0.72 * exp(-pow((qrsPhase - 0.006) * 120.0, 2.0)) : 0.0
            let tWave = 0.040 * exp(-pow((qrsPhase - 0.24) * 14.0, 2.0))
            return base + pWave + qrs + tWave
        }

        return ECGSample(
            source: .hardcoded,
            samplingFrequency: samplingFrequency,
            classification: "sinusRhythm_sample",
            symptomsStatus: "notSet_sample",
            averageHeartRate: 70.0,
            voltages: voltages
        )
    }

    static func load(from url: URL) throws -> ECGSample {
        let data = try Data(contentsOf: url)
        let decoded = try JSONSerialization.jsonObject(with: data)

        guard let object = decoded as? [String: Any],
              let voltages = object["voltages"] as? [Double],
              !voltages.isEmpty
        else {
            throw NSError(
                domain: "OpenTSLMInferenceService",
                code: 6,
                userInfo: [NSLocalizedDescriptionKey: "Unsupported ECG JSON format"]
            )
        }

        let samplingFrequency = object["samplingFrequency"] as? Double
            ?? object["sampling_frequency"] as? Double
            ?? 256.0
        let classification = object["classification"] as? String
        let symptomsStatus = object["symptomsStatus"] as? String
            ?? object["symptoms_status"] as? String
        let averageHeartRate = object["averageHeartRate"] as? Double
            ?? object["average_heart_rate"] as? Double

        let source: ECGSampleSource
        if object["source"] as? String == ECGSampleSource.healthkitJSON.rawValue {
            source = .healthkitJSON
        } else {
            source = .jsonOverride
        }

        return ECGSample(
            source: source,
            samplingFrequency: samplingFrequency,
            classification: classification,
            symptomsStatus: symptomsStatus,
            averageHeartRate: averageHeartRate,
            voltages: voltages
        )
    }
}
