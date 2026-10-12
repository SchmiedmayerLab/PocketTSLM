//
// This source file is part of the PocketTSLM open-source project
//
// SPDX-FileCopyrightText: 2026 Schmiedmayer Lab and the project authors (see CONTRIBUTORS.md)
//
// SPDX-License-Identifier: MIT
//

import Foundation

/// One row of an evaluation dataset.
public struct EvaluationSample: Sendable {
    public let input: TSLMInput
    /// The dataset's label, e.g. `W` or `yes`.
    public let label: String
    /// The dataset's reference rationale.
    public let rationale: String
    /// The normalised answer a correct generation ends with, or `nil` if the row has none.
    public let expectedAnswer: String?

    public init(input: TSLMInput, label: String, rationale: String, expectedAnswer: String?) {
        self.input = input
        self.label = label
        self.rationale = rationale
        self.expectedAnswer = expectedAnswer
    }
}

/// The paper's accuracy evaluation: greedy answers to dataset rows, scored by their final answer.
public enum Evaluation {
    /// One evaluated row.
    public struct Row: Codable, Sendable {
        public let index: Int
        public let label: String
        public let groundTruth: String?
        public let predicted: String?
        public let correct: Bool
        public let seconds: Double
        public let output: String
    }

    /// A whole run, in the format of the paper app's `opentslm_sleep_eval.json` and `opentslm_ecg_eval.json`.
    public struct Report: Codable, Sendable {
        public let sampleCount: Int
        public let maxNewTokens: Int
        public let seriesLength: Int
        public let loraApplied: Bool
        public let parsed: Int
        public let correct: Int
        public let accuracy: Double
        public let rows: [Row]

        /// Pretty-printed JSON with sorted keys, as the paper app wrote it.
        public func json() throws -> Data {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            return try encoder.encode(self)
        }
    }

    /// The answer after the last `cue`, up to the end of its line or sentence, with special tokens
    /// removed, whitespace collapsed and lowercased; `nil` if the text has none.
    public static func finalAnswer(in text: String, cue: String = "Answer:") -> String? {
        // The pipes must be escaped: an unescaped `<|...|>` is regex alternation, whose `[^|]*`
        // branch matches the empty string and loops forever.
        let cleaned = text.replacingOccurrences(of: "<\\|[^|]*\\|>", with: "", options: .regularExpression)
        guard let marker = cleaned.range(of: cue, options: [.backwards, .caseInsensitive]) else {
            return nil
        }
        let line = cleaned[marker.upperBound...].prefix { $0 != "\n" && $0 != "." }
        let collapsed = line.split(whereSeparator: { $0.isWhitespace }).joined(separator: " ").lowercased()
        return collapsed.isEmpty ? nil : collapsed
    }

    /// Answers `samples` in order and scores each answer against the sample's expected answer.
    /// - Parameters:
    ///   - maxTokens: Defaults to the checkpoint's `generation.maxTokens`.
    ///   - onRow: Called after each sample, e.g. to log progress.
    public static func run(
        _ model: TSLM,
        on samples: [EvaluationSample],
        maxTokens: Int? = nil,
        onRow: ((Row) -> Void)? = nil
    ) async throws -> Report {
        let maxTokens = maxTokens ?? model.checkpoint.configuration.generation.maxTokens
        let cue = model.checkpoint.configuration.prompt.answerCue
        var rows: [Row] = []
        var seriesLength = 0

        for (index, sample) in samples.enumerated() {
            let prompt = try model.checkpoint.prompt(for: sample.input)
            seriesLength = prompt.series.first?.count ?? 0
            let start = DispatchTime.now()
            let output = try await model.generate(prompt, maxTokens: maxTokens)
            let seconds = Double(DispatchTime.now().uptimeNanoseconds - start.uptimeNanoseconds) / 1_000_000_000

            let predicted = finalAnswer(in: output, cue: cue)
            let row = Row(
                index: index,
                label: sample.label,
                groundTruth: sample.expectedAnswer,
                predicted: predicted,
                correct: predicted != nil && predicted == sample.expectedAnswer,
                seconds: (seconds * 100).rounded() / 100,
                output: output
            )
            rows.append(row)
            onRow?(row)
        }

        let correct = rows.filter(\.correct).count
        return Report(
            sampleCount: rows.count,
            maxNewTokens: maxTokens,
            seriesLength: seriesLength,
            loraApplied: true,
            parsed: rows.filter { $0.predicted != nil }.count,
            correct: correct,
            accuracy: (Double(correct) / Double(max(rows.count, 1)) * 10_000).rounded() / 10_000,
            rows: rows
        )
    }
}
