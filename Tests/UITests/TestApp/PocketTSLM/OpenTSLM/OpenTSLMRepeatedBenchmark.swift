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
import PocketTSLM

// MARK: - Repeated context-economics runs (variance)

extension OpenTSLMInferenceService {
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

    /// Mean, sample standard deviation, minimum and maximum of a series of measurements.
    private struct Spread {
        let mean: Double
        let standardDeviation: Double
        let min: Double
        let max: Double

        init(_ values: [Double]) {
            guard !values.isEmpty else {
                (mean, standardDeviation, min, max) = (0, 0, 0, 0)
                return
            }
            let mean = values.reduce(0, +) / Double(values.count)
            guard values.count > 1 else {
                (self.mean, standardDeviation, min, max) = (mean, 0, mean, mean)
                return
            }
            // Sample standard deviation (n-1): we are estimating spread, not describing a population.
            let variance = values.reduce(0) { $0 + ($1 - mean) * ($1 - mean) } / Double(values.count - 1)
            (self.mean, standardDeviation, min, max) = (mean, variance.squareRoot(), values.min() ?? mean, values.max() ?? mean)
        }
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
        base: ModelContainer?
    ) async throws -> String {
        let sweep = lengths ?? (ecg ? [100, 250, 500, 1000] : [100, 250, 500, 1000, 1500])
        let model = try bundledModel(ecg ? BundledAssets.ecgCheckpoint : BundledAssets.sleepCheckpoint, on: base)
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
            let state = "thermalState=\(thermal) lowPower=\(lowPower)"
            logger.info("repeated benchmark: run \(run + 1, privacy: .public) starting, \(state, privacy: .public)")
            let report = try await benchmark.run(
                sample: sample, lengths: sweep, maxTokens: maxTokens,
                contextWindow: contextWindow, maxPromptPositions: maxPromptPositions
            )
            reports.append(report)
            GPU.clearCache()
            let modality = ecg ? "ecg" : "eeg"
            logger.info("repeated benchmark (\(modality, privacy: .public)): run \(run + 1, privacy: .public)/\(repeats, privacy: .public) done")
        }

        let report = RepeatedReport(
            modality: ecg ? "ecg" : "eeg", repeats: reports.count, lengths: sweep,
            maxTokens: maxTokens, maxPromptPositions: maxPromptPositions,
            loraApplied: true, thermalStatePerRun: thermalStates,
            lowPowerModePerRun: lowPowerModes, arms: Self.arms(of: reports, lengthCount: sweep.count))
        writeToDocuments(report, name: ecg ? "opentslm_repeated_ecg.json" : "opentslm_repeated_eeg.json")
        return Self.summary(of: report, modality: ecg ? "12-lead ECG" : "single-channel EEG")
    }

    /// One entry per length and encoding, over the runs that were not skipped.
    private static func arms(of reports: [OpenTSLMBenchmark.Report], lengthCount: Int) -> [RepeatedArmStats] {
        var arms: [RepeatedArmStats] = []
        for rowIndex in 0 ..< lengthCount {
            for encoding in ["soft_prompt", "text_serialized"] {
                let samples: [OpenTSLMBenchmark.ArmStats] = reports.compactMap { report in
                    guard rowIndex < report.rows.count else {
                        return nil
                    }
                    let row = report.rows[rowIndex]
                    return encoding == "soft_prompt" ? row.softPrompt : row.textSerialized
                }
                guard let first = samples.first else {
                    continue
                }
                let ran = samples.filter { !$0.didOOM }
                let prefill = Spread(ran.map(\.prefillSeconds))
                let memory = Spread(ran.map { Double($0.peakMemoryBytes) / 1_048_576 })
                arms.append(RepeatedArmStats(
                    encoding: encoding,
                    seriesLength: first.seriesLength,
                    runs: ran.count,
                    skipped: ran.isEmpty,
                    promptPositions: first.promptPositions,
                    timeSeriesPositions: first.timeSeriesPositions,
                    positionsStable: samples.allSatisfy { $0.promptPositions == first.promptPositions },
                    prefillSecondsMean: prefill.mean, prefillSecondsSD: prefill.standardDeviation,
                    prefillSecondsMin: prefill.min, prefillSecondsMax: prefill.max,
                    peakMemoryMiBMean: memory.mean, peakMemoryMiBSD: memory.standardDeviation,
                    peakMemoryMiBMin: memory.min, peakMemoryMiBMax: memory.max,
                    prefillSecondsPerRun: ran.map { ($0.prefillSeconds * 1000).rounded() / 1000 },
                    peakMemoryMiBPerRun: ran.map { (Double($0.peakMemoryBytes) / 1_048_576).rounded() }))
            }
        }
        return arms
    }

    /// The text summary shown in the chat.
    private static func summary(of report: RepeatedReport, modality: String) -> String {
        var lines = [
            "Repeated context-economics (\(modality), \(report.repeats) runs, greedy, max \(report.maxTokens) new tokens)",
            "positions are analytic; TTFT and peak memory are mean +/- SD over runs",
            "thermal: \(report.thermalStatePerRun.joined(separator: " -> "))",
            "low power: \(report.lowPowerModePerRun.map { $0 ? "on" : "off" }.joined(separator: " -> "))",
            ""
        ]
        lines.append("    L  encoding         pos     TTFT (s)            peak (MiB)")
        for arm in report.arms {
            let enc = arm.encoding == "soft_prompt" ? "soft" : "text"
            if arm.skipped {
                lines.append(String(
                    format: "%5d  %-8s %7d     n/a (exceeds budget)", arm.seriesLength, (enc as NSString).utf8String!, arm.promptPositions
                ))
                continue
            }
            lines.append(String(format: "%5d  %-8s %7d  %6.2f +/- %.2f   %7.0f +/- %.0f",
                                arm.seriesLength, (enc as NSString).utf8String!, arm.promptPositions,
                                arm.prefillSecondsMean, arm.prefillSecondsSD,
                                arm.peakMemoryMiBMean, arm.peakMemoryMiBSD))
        }
        // Thermal drift shows up as a monotone first->last change, not as scatter.
        if let longest = report.arms.last(where: { $0.encoding == "soft_prompt" && !$0.skipped }),
           let first = longest.prefillSecondsPerRun.first,
           let last = longest.prefillSecondsPerRun.last, first > 0 {
            lines.append("")
            lines.append(String(format: "soft-prompt TTFT at L=%d across runs: %@  (last/first = %.2fx)",
                                longest.seriesLength,
                                longest.prefillSecondsPerRun.map { String(format: "%.2f", $0) }.joined(separator: ", "),
                                last / first))
        }
        if let unstable = report.arms.first(where: { !$0.positionsStable }) {
            lines.append("")
            lines.append(
                "WARNING: prompt positions varied across runs at L=\(unstable.seriesLength) (\(unstable.encoding)) "
                    + "— token counts should be deterministic"
            )
        }
        return lines.joined(separator: "\n")
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
}
