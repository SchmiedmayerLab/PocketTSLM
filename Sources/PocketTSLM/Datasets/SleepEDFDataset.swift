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
        // Bundled sleep_cot.csv is ~300MB — read only a prefix on device.
        let handle = try FileHandle(forReadingFrom: csvURL)
        let prefix = try handle.read(upToCount: 20 * 1024 * 1024) ?? Data()
        let csvPrefix = String(decoding: prefix, as: UTF8.self)
        let safePrefix = csvPrefix.prefix(upTo: csvPrefix.lastIndex(of: "\n") ?? csvPrefix.endIndex)

        let records = parseCSV(String(safePrefix))
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

    static func parseCSV(_ csv: String) -> [[String]] {
        var records: [[String]] = []
        var row: [String] = []
        var field = ""
        var insideQuotes = false

        var i = csv.startIndex
        while i < csv.endIndex {
            let ch = csv[i]
            if ch == "\"" {
                let nextIndex = csv.index(after: i)
                if insideQuotes && nextIndex < csv.endIndex && csv[nextIndex] == "\"" {
                    field.append("\"")
                    i = csv.index(after: nextIndex)
                    continue
                }
                insideQuotes.toggle()
                i = nextIndex
                continue
            }

            if ch == "," && !insideQuotes {
                row.append(field)
                field.removeAll(keepingCapacity: true)
                i = csv.index(after: i)
                continue
            }

            // NB: Swift treats "\r\n" as a single Character (grapheme cluster), so a CRLF file
            // matches neither "\n" nor "\r" and the whole file collapses into one record.
            if (ch == "\n" || ch == "\r" || ch == "\r\n") && !insideQuotes {
                row.append(field)
                field.removeAll(keepingCapacity: true)
                if !row.isEmpty {
                    records.append(row)
                }
                row.removeAll(keepingCapacity: true)

                if ch == "\r" {
                    let nextIndex = csv.index(after: i)
                    if nextIndex < csv.endIndex && csv[nextIndex] == "\n" {
                        i = csv.index(after: nextIndex)
                        continue
                    }
                }

                i = csv.index(after: i)
                continue
            }

            field.append(ch)
            i = csv.index(after: i)
        }

        if !field.isEmpty || !row.isEmpty {
            row.append(field)
            records.append(row)
        }

        return records
    }
}
