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
import PocketTSLM
import Spezi
import SpeziLLM
import SpeziLLMLocal

@Observable
class OpenTSLMInferenceService: DefaultInitializable, Module, EnvironmentAccessible {
    @ObservationIgnored private let logger = Logger(subsystem: "HealthyLLM", category: "OpenTSLMInferenceService")

    required init() { }

    /// A single prompt → response interaction for the chat UI: the constructed text prompt
    /// (pre-prompt + per-lead time-series descriptions + post-prompt) and the model's
    /// generated answer, returned separately so they can occupy the sender and reply bubbles.
    struct SampleInteraction {
        let prompt: String
        let response: String
        let groundTruth: String
    }

    /// EEG/sleep single-sample interaction (fixed raw CSV row) with the sleep checkpoint,
    /// returning the prompt and generated answer separately.
    func sleepSampleInteraction(
        sampleIndex: Int = 0,
        llmRunner: LLMRunner,
        llmSession: LLMLocalSession
    ) async throws -> SampleInteraction {
        let model = try await bundledModel(BundledAssets.sleepCheckpoint, on: llmSession)
        return try await interaction(with: model, on: prepared(BundledAssets.sleepRow(sampleIndex, of: "sleep_cot.csv"), for: model))
    }

    /// 12-lead ECG-QA single-sample interaction (bundled test row) with the ECG checkpoint.
    func ecgSampleInteraction(
        sampleIndex: Int = 0,
        llmRunner: LLMRunner,
        llmSession: LLMLocalSession
    ) async throws -> SampleInteraction {
        let model = try await bundledModel(BundledAssets.ecgCheckpoint, on: llmSession)
        return try await interaction(with: model, on: prepared(BundledAssets.ecgRow(sampleIndex), for: model))
    }

    private func interaction(with model: TSLM, on sample: OpenTSLMSPSample) async throws -> SampleInteraction {
        let response = try await model.generate(sample, maxTokens: 200)
        return SampleInteraction(prompt: promptText(for: sample), response: response, groundTruth: sample.answer)
    }

    /// The displayable text prompt for a sample (pre-prompt, per-lead time-series
    /// descriptions, post-prompt). The numeric series itself is injected as embeddings.
    private func promptText(for sample: OpenTSLMSPSample) -> String {
        ([sample.prePrompt] + sample.timeSeriesText + [sample.postPrompt]).joined(separator: "\n")
    }

    /// Runs OpenTSLM-SP on a real (e.g. HealthKit) ECG recording and returns the model's
    /// analysis text. Unlike the bundled-sample commands, this uses the supplied recording,
    /// requires a live LLM session, and returns a clean user-facing answer.
    func runECGInference(
        voltages: [Double],
        samplingFrequency: Double,
        classification: String?,
        symptomsStatus: String?,
        averageHeartRate: Double?,
        llmRunner: LLMRunner?,
        llmSession: LLMLocalSession?
    ) async throws -> String {
        guard !voltages.isEmpty else {
            throw NSError(
                domain: "OpenTSLMInferenceService", code: 8,
                userInfo: [NSLocalizedDescriptionKey: "The ECG recording has no voltage samples."])
        }

        logger.info("runECGInference: start voltages=\(voltages.count, privacy: .public) freq=\(samplingFrequency, privacy: .public)")
        let model = try await bundledModel(BundledAssets.ecgCheckpoint, on: llmSession)
        // Apple Watch records one lead; it is copied to the 12 leads the checkpoint expects, which
        // is a distribution shift from PTB-XL. The question mirrors ECG-QA's most common
        // closed-set template, which greedy decoding answers far more reliably than an open one.
        let input = TSLMInput(
            series: ECG.twelveLeads(fromLeadI: voltages, samplingRate: samplingFrequency),
            context: [
                "clinicalContext": "Single-lead ambulatory ECG recording (Lead I equivalent, replicated across the standard 12 leads).",
                "question": "Is the cardiac rhythm shown in this ECG consistent with atrial fibrillation?"
            ],
            answerOptions: ["yes", "no"]
        )
        let analysis = try await model.generate(input, maxTokens: 200)
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
        let sweep = lengths ?? (ecg ? [100, 250, 500, 1000] : [100, 250, 500, 1000, 1500])
        let model = try await bundledModel(ecg ? BundledAssets.ecgCheckpoint : BundledAssets.sleepCheckpoint, on: llmSession)
        let row = try ecg ? BundledAssets.ecgRow(0) : BundledAssets.sleepRow(0, of: "sleep_cot_test20.csv")
        let sample = try prepared(row, for: model)

        let benchmark = OpenTSLMBenchmark(model: model)
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
                contextWindow: contextWindow, maxPromptPositions: maxPromptPositions)
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
            loraApplied: true, thermalStatePerRun: thermalStates,
            lowPowerModePerRun: lowPowerModes, arms: arms)

        writeToDocuments(report, name: ecg ? "opentslm_repeated_ecg.json" : "opentslm_repeated_eeg.json")

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

    // MARK: - Accuracy evals

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
        let model = try await bundledModel(BundledAssets.sleepCheckpoint, on: llmSession)
        let samples = try SleepEDFDataset.evaluationSamples(
            csvURL: BundledAssets.evaluationFile("sleep_cot_test20.csv"),
            count: sampleCount
        )
        let report = try await Evaluation.run(model, on: samples, maxTokens: maxTokens) { row in
            self.logRow(row, of: samples.count, eval: "sleep")
        }
        writeToDocuments(report, name: "opentslm_sleep_eval.json")
        return summary(
            of: report,
            title: "Sleep-EDF accuracy eval (pinned test-split rows 0..<\(report.sampleCount), L=\(report.seriesLength), greedy, max \(maxTokens) new tokens)",
            columnWidth: 16
        )
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
        let model = try await bundledModel(BundledAssets.ecgCheckpoint, on: llmSession)
        let dataset = try BundledAssets.ecgQADataset()
        let samples = try (0 ..< min(sampleCount, dataset.count)).map(dataset.evaluationSample(at:))
        let report = try await Evaluation.run(model, on: samples, maxTokens: maxTokens) { row in
            self.logRow(row, of: samples.count, eval: "ecg")
        }
        writeToDocuments(report, name: "opentslm_ecg_eval.json")
        return summary(
            of: report,
            title: "ECG-QA accuracy eval (bundled test rows 0..<\(report.sampleCount), L=\(report.seriesLength), greedy, max \(maxTokens) new tokens)",
            columnWidth: 22
        )
    }

    private func logRow(_ row: Evaluation.Row, of count: Int, eval: String) {
        logger.info("\(eval, privacy: .public) eval [\(row.index + 1, privacy: .public)/\(count, privacy: .public)] gt=\(row.groundTruth ?? "nil", privacy: .public) pred=\(row.predicted ?? "nil", privacy: .public) \(row.correct ? "OK" : "X", privacy: .public)")
    }

    /// Writes `report` to Documents as pretty-printed JSON with sorted keys. `echo` also prints it
    /// to the console, so all fields are copyable without pulling the file off the device.
    private func writeToDocuments(_ report: some Encodable, name: String, echo: Bool = false) {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        guard let json = try? encoder.encode(report) else {
            return
        }
        if let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first {
            let url = documents.appendingPathComponent(name)
            try? json.write(to: url)
            logger.info("wrote \(url.path, privacy: .public)")
        }
        if echo {
            print("=== \(name) ===\n\(String(decoding: json, as: UTF8.self))\n=== end \(name) ===")
        }
    }

    private func summary(of report: Evaluation.Report, title: String, columnWidth: Int) -> String {
        var lines = [title, "LoRA applied: \(report.loraApplied ? "yes" : "no")", ""]
        for row in report.rows {
            let groundTruth = (row.groundTruth ?? "nil").padding(toLength: columnWidth, withPad: " ", startingAt: 0)
            let predicted = (row.predicted ?? "nil").padding(toLength: columnWidth, withPad: " ", startingAt: 0)
            lines.append("\(row.index)  \(groundTruth) -> \(predicted) \(row.correct ? "OK" : "X")")
        }
        lines.append("")
        lines.append("accuracy: \(report.correct)/\(report.rows.count)   parsed: \(report.parsed)/\(report.rows.count)")
        return lines.joined(separator: "\n")
    }

    /// Runs the §4 context-economics benchmark: the same Sleep-EDF sample and the
    /// same on-device backbone, encoded as soft prompts vs. OpenTSLM text
    /// serialization, swept over series length. Reports prompt positions, end-to-end
    /// latency, decode tokens/s, peak GPU memory, and the context-window ceiling for
    /// each encoding; writes the full JSON to Documents and returns a summary string.
    func runContextEconomicsBenchmark(
        sampleIndex: Int = 0,
        lengths: [Int] = [100, 250, 500, 1000, 1500],
        maxTokens: Int = 200,
        contextWindow: Int = 8192,
        maxPromptPositions: Int = 2048,
        llmRunner: LLMRunner,
        llmSession: LLMLocalSession?
    ) async throws -> String {
        let model = try await bundledModel(BundledAssets.sleepCheckpoint, on: llmSession)
        // Use a fixed raw CSV row (no split) — the parity-verified sample whose generation
        // matches the Python reference — so the benchmark's decoded text is coherent and
        // deterministic. Cost metrics (positions/prefill/memory) are sample-independent.
        let sample = try prepared(BundledAssets.sleepRow(sampleIndex, of: "sleep_cot.csv"), for: model)
        let report = try await OpenTSLMBenchmark(model: model).run(
            sample: sample,
            lengths: lengths,
            maxTokens: maxTokens,
            contextWindow: contextWindow,
            maxPromptPositions: maxPromptPositions)
        writeToDocuments(report, name: "opentslm_context_economics.json", echo: true)
        return report.formattedSummary(split: "raw-row", sampleIndex: max(sampleIndex, 0))
    }

    /// ECG variant of the §4 benchmark: the 12-lead ECG-QA CoT sample swept over per-lead
    /// length, same soft-prompt vs. text-serialized comparison on the ECG-tuned backbone.
    /// 12 leads use ~12× the positions of the 1-lead sleep case, so the soft-prompt arm
    /// relies on ``MLXEmbeddingGenerator``'s chunked prefill to avoid OOM at longer lengths.
    func runContextEconomicsBenchmarkECG(
        sampleIndex: Int = 0,
        lengths: [Int] = [100, 250, 500, 1000],
        maxTokens: Int = 200,
        contextWindow: Int = 8192,
        maxPromptPositions: Int = 4096,
        llmRunner: LLMRunner,
        llmSession: LLMLocalSession?
    ) async throws -> String {
        let model = try await bundledModel(BundledAssets.ecgCheckpoint, on: llmSession)
        let sample = try prepared(BundledAssets.ecgRow(sampleIndex), for: model)
        let report = try await OpenTSLMBenchmark(model: model).run(
            sample: sample,
            lengths: lengths,
            maxTokens: maxTokens,
            contextWindow: contextWindow,
            maxPromptPositions: maxPromptPositions)
        writeToDocuments(report, name: "opentslm_context_economics_ecg.json", echo: true)
        return report.formattedSummary(split: "ecg-test", sampleIndex: sampleIndex)
    }

    // MARK: - Bundled models and data

    /// The checkpoint folders and evaluation data the app bundles; `scripts/fetch_assets.sh` restores them.
    private enum BundledAssets {
        static let sleepCheckpoint = "llama-3.2-1b-sleep-sp-mlx"
        static let ecgCheckpoint = "llama-3.2-1b-ecg-sp-mlx"

        static func item(_ path: String) throws -> URL {
            guard let url = Bundle.main.resourceURL?.appendingPathComponent(path),
                  FileManager.default.fileExists(atPath: url.path) else {
                throw NSError(
                    domain: "OpenTSLMInferenceService", code: 1,
                    userInfo: [NSLocalizedDescriptionKey: "\(path) is not bundled; run scripts/fetch_assets.sh"])
            }
            return url
        }

        static func evaluationFile(_ name: String) throws -> URL {
            try item("EvalData/\(name)")
        }

        static func ecgQADataset() throws -> ECGQACoTDataset {
            try ECGQACoTDataset(
                csvURL: evaluationFile("ecg_qa_cot_test.csv"),
                waveformsDirectory: evaluationFile("ecg_qa_waveforms"),
                templateAnswersURL: evaluationFile("ecg_qa_template_answers.json")
            )
        }

        /// Row `index` of a bundled Sleep-EDF export, or its last row if the export is shorter.
        static func sleepRow(_ index: Int, of csvName: String) throws -> EvaluationSample {
            let rows = try SleepEDFDataset.evaluationSamples(csvURL: evaluationFile(csvName), count: max(index, 0) + 1)
            guard let row = rows.last else {
                throw NSError(
                    domain: "OpenTSLMInferenceService", code: 15,
                    userInfo: [NSLocalizedDescriptionKey: "\(csvName) has no rows"])
            }
            return row
        }

        /// Row `index` of the bundled ECG-QA test export, or its last row if the export is shorter.
        static func ecgRow(_ index: Int) throws -> EvaluationSample {
            let dataset = try ecgQADataset()
            guard dataset.count > 0 else {
                throw NSError(
                    domain: "OpenTSLMInferenceService", code: 15,
                    userInfo: [NSLocalizedDescriptionKey: "ecg_qa_cot_test.csv has no rows"])
            }
            return try dataset.evaluationSample(at: min(max(index, 0), dataset.count - 1))
        }
    }

    /// The bundled checkpoint `name` on the session's base model.
    private func bundledModel(_ name: String, on session: LLMLocalSession?) async throws -> TSLM {
        guard let session, let base = await MainActor.run(body: { session.modelContainer }) else {
            throw NSError(
                domain: "OpenTSLMInferenceService", code: 6,
                userInfo: [NSLocalizedDescriptionKey: "An LLM session is required."])
        }
        return try TSLM(base: base, checkpoint: TSLMCheckpoint(directory: BundledAssets.item("Checkpoints/\(name)")))
    }

    /// `row` as `model`'s input, carrying the row's label and rationale.
    private func prepared(_ row: EvaluationSample, for model: TSLM) throws -> OpenTSLMSPSample {
        try model.checkpoint.makeSample(row.input, label: row.label, answer: row.rationale)
    }
}
