//
// This source file is part of the PocketTSLM open-source project
//
// SPDX-FileCopyrightText: 2026 Schmiedmayer Lab and the project authors (see CONTRIBUTORS.md)
//
// SPDX-License-Identifier: MIT
//

import Foundation
import MLX
import MLXLMCommon
import OSLog
import PocketTSLM
import Spezi

@Observable
class OpenTSLMInferenceService: DefaultInitializable, Module, EnvironmentAccessible {
    @ObservationIgnored let logger = Logger(subsystem: "PocketTSLM", category: "OpenTSLMInferenceService")

    required init() { }

    /// A single prompt → response interaction for the chat UI: the constructed text prompt
    /// (pre-prompt + per-lead time-series descriptions + post-prompt) and the model's
    /// generated answer, returned separately so they can occupy the sender and reply bubbles.
    struct SampleInteraction {
        let prompt: String
        let response: String
    }

    /// EEG/sleep single-sample interaction (fixed raw CSV row) with the sleep checkpoint,
    /// returning the prompt and generated answer separately.
    func sleepSampleInteraction(
        sampleIndex: Int = 0,
        base: ModelContainer?
    ) async throws -> SampleInteraction {
        let model = try bundledModel(BundledAssets.sleepCheckpoint, on: base)
        return try await interaction(with: model, on: prepared(BundledAssets.sleepRow(sampleIndex, of: "sleep_cot.csv"), for: model))
    }

    /// 12-lead ECG-QA single-sample interaction (bundled test row) with the ECG checkpoint.
    func ecgSampleInteraction(
        sampleIndex: Int = 0,
        base: ModelContainer?
    ) async throws -> SampleInteraction {
        let model = try bundledModel(BundledAssets.ecgCheckpoint, on: base)
        return try await interaction(with: model, on: prepared(BundledAssets.ecgRow(sampleIndex), for: model))
    }

    private func interaction(with model: TSLM, on prompt: TSLMPrompt) async throws -> SampleInteraction {
        let response = try await model.generate(prompt, maxTokens: 200).text
        return SampleInteraction(prompt: prompt.text, response: response)
    }

    /// Runs OpenTSLM-SP on a real (e.g. HealthKit) ECG recording and returns the model's
    /// analysis text. Unlike the bundled-sample commands, this uses the supplied recording,
    /// requires a live LLM session, and returns a clean user-facing answer.
    func runECGInference(
        voltages: [Double],
        samplingFrequency: Double,
        base: ModelContainer?
    ) async throws -> String {
        guard !voltages.isEmpty else {
            throw NSError(
                domain: "OpenTSLMInferenceService", code: 8,
                userInfo: [NSLocalizedDescriptionKey: "The ECG recording has no voltage samples."])
        }

        logger.info("runECGInference: start voltages=\(voltages.count, privacy: .public) freq=\(samplingFrequency, privacy: .public)")
        let model = try bundledModel(BundledAssets.ecgCheckpoint, on: base)
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

    // MARK: - Accuracy evals

    /// Runs the on-device accuracy eval over the pinned 20-row Sleep-EDF test-split export.
    /// Row `i` here is the same sample as index `i` in the Python reference's test split, so
    /// the per-sample predictions are directly comparable to the PyTorch/MLX reference runs.
    /// Writes `opentslm_sleep_eval.json` to Documents and returns a short summary.
    func runSleepAccuracyEval(
        sampleCount: Int = 20,
        maxTokens: Int = 200,
        base: ModelContainer?
    ) async throws -> String {
        let model = try bundledModel(BundledAssets.sleepCheckpoint, on: base)
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
            title: "Sleep-EDF accuracy eval (pinned test-split rows 0..<\(report.sampleCount), L=\(report.seriesLength), "
                + "greedy, max \(maxTokens) new tokens)",
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
        base: ModelContainer?
    ) async throws -> String {
        let model = try bundledModel(BundledAssets.ecgCheckpoint, on: base)
        let dataset = try BundledAssets.ecgQADataset()
        let samples = try (0 ..< min(sampleCount, dataset.count)).map(dataset.evaluationSample(at:))
        let report = try await Evaluation.run(model, on: samples, maxTokens: maxTokens) { row in
            self.logRow(row, of: samples.count, eval: "ecg")
        }
        writeToDocuments(report, name: "opentslm_ecg_eval.json")
        return summary(
            of: report,
            title: "ECG-QA accuracy eval (bundled test rows 0..<\(report.sampleCount), L=\(report.seriesLength), "
                + "greedy, max \(maxTokens) new tokens)",
            columnWidth: 22
        )
    }

    private func logRow(_ row: Evaluation.Row, of count: Int, eval: String) {
        let verdict = row.correct ? "OK" : "X"
        let line = "\(eval) eval [\(row.index + 1)/\(count)] gt=\(row.groundTruth ?? "nil") pred=\(row.predicted ?? "nil") \(verdict)"
        logger.info("\(line, privacy: .public)")
    }

    /// Writes `report` to Documents as pretty-printed JSON with sorted keys. `echo` also prints it
    /// to the console, so all fields are copyable without pulling the file off the device.
    func writeToDocuments(_ report: some Encodable, name: String, echo: Bool = false) {
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
            print("=== \(name) ===\n\(String(bytes: json, encoding: .utf8) ?? "")\n=== end \(name) ===")
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
        base: ModelContainer?
    ) async throws -> String {
        let model = try bundledModel(BundledAssets.sleepCheckpoint, on: base)
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
    /// relies on the package's chunked prefill to avoid OOM at longer lengths.
    func runContextEconomicsBenchmarkECG(
        sampleIndex: Int = 0,
        lengths: [Int] = [100, 250, 500, 1000],
        maxTokens: Int = 200,
        contextWindow: Int = 8192,
        maxPromptPositions: Int = 4096,
        base: ModelContainer?
    ) async throws -> String {
        let model = try bundledModel(BundledAssets.ecgCheckpoint, on: base)
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
    enum BundledAssets {
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

    /// The base model the bundled checkpoints were converted for, at its pinned revision.
    func baseModelConfiguration() throws -> ModelConfiguration {
        let checkpoint = try TSLMCheckpoint(directory: BundledAssets.item("Checkpoints/\(BundledAssets.sleepCheckpoint)"))
        let baseModel = checkpoint.configuration.baseModel
        return ModelConfiguration(id: baseModel.id, revision: baseModel.revision ?? "main")
    }

    /// The bundled checkpoint `name` on the base model.
    func bundledModel(_ name: String, on base: ModelContainer?) throws -> TSLM {
        guard let base else {
            throw NSError(
                domain: "OpenTSLMInferenceService", code: 6,
                userInfo: [NSLocalizedDescriptionKey: "The base model is not loaded."])
        }
        return try TSLM(base: base, checkpoint: TSLMCheckpoint(directory: BundledAssets.item("Checkpoints/\(name)")))
    }

    /// `row` as `model`'s prompt.
    func prepared(_ row: EvaluationSample, for model: TSLM) throws -> TSLMPrompt {
        try model.checkpoint.prompt(for: row.input)
    }
}
