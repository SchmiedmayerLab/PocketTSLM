//
// This source file is part of the PocketTSLM open-source project
//
// SPDX-FileCopyrightText: 2026 Schmiedmayer Lab and the project authors (see CONTRIBUTORS.md)
//
// SPDX-License-Identifier: MIT
//

import Foundation

/// Reads Sleep-EDF chain-of-thought exports of OpenTSLM's `SleepEDFCoTQADataset`.
public enum SleepEDFDataset {
    /// Reads the first `count` rows of a Sleep-EDF CoT export (columns `time_series`, `label`,
    /// `rationale`), with no train/validation/test split.
    public static func evaluationSamples(csvURL: URL, count: Int) throws -> [EvaluationSample] {
        try readRows(from: csvURL, maxRows: max(count, 1)).prefix(count).map { row in
            EvaluationSample(
                input: TSLMInput(series: [TimeSeries(samples: try parseTimeSeries(row.timeSeries))]),
                label: row.label,
                rationale: row.rationale,
                expectedAnswer: Evaluation.finalAnswer(in: row.rationale)
            )
        }
    }
}

private extension SleepEDFDataset {

    struct Row {
        let timeSeries: String
        let label: String
        let rationale: String
    }

    static func readRows(from csvURL: URL, maxRows: Int) throws -> [Row] {
        let records = try CSV.records(at: csvURL)
        guard let header = records.first else {
            throw NSError(domain: "SleepEDFDataset", code: 1, userInfo: [NSLocalizedDescriptionKey: "CSV has no header"])
        }

        let columnIndex = Dictionary(uniqueKeysWithValues: header.enumerated().map { ($1, $0) })
        guard let timeSeriesIdx = columnIndex["time_series"],
              let labelIdx = columnIndex["label"]
        else {
            throw NSError(domain: "SleepEDFDataset", code: 2, userInfo: [NSLocalizedDescriptionKey: "CSV must include 'time_series' and 'label' columns"])
        }

        let rationaleIdx = columnIndex["rationale"]

        var rows: [Row] = []
        rows.reserveCapacity(maxRows)

        for record in records.dropFirst() where rows.count < maxRows {
            guard !record.isEmpty else { continue }
            guard timeSeriesIdx < record.count, labelIdx < record.count else { continue }
            let rationale = (rationaleIdx != nil && rationaleIdx! < record.count) ? record[rationaleIdx!] : ""
            rows.append(Row(timeSeries: record[timeSeriesIdx], label: record[labelIdx], rationale: rationale))
        }

        return rows
    }

    static func parseTimeSeries(_ raw: String) throws -> [Float] {
        guard let data = raw.data(using: .utf8) else {
            throw NSError(domain: "SleepEDFDataset", code: 3, userInfo: [NSLocalizedDescriptionKey: "Invalid UTF-8 in time_series"])
        }

        let parsed = try JSONSerialization.jsonObject(with: data)

        if let one = parsed as? [Double] {
            return one.map(Float.init)
        }

        if let two = parsed as? [[Double]], let first = two.first {
            return first.map(Float.init)
        }

        throw NSError(domain: "SleepEDFDataset", code: 4, userInfo: [NSLocalizedDescriptionKey: "Unsupported time_series JSON shape"])
    }
}
