//
// This source file is part of the PocketTSLM open-source project
//
// SPDX-FileCopyrightText: 2026 Schmiedmayer Lab and the project authors (see CONTRIBUTORS.md)
//
// SPDX-License-Identifier: MIT
//

import Foundation

/// Reads ECG-QA chain-of-thought exports of OpenTSLM's `ECGQACoTQADataset`: CSV rows (question,
/// answer, template, ECG id, rationale, clinical context) and, per ECG, a waveform JSON file with
/// the leads and descriptions that OpenTSLM's loader produced.
public final class ECGQACoTDataset {
    private struct Row {
        let ecgId: Int
        let templateId: Int
        let question: String
        let clinicalContext: String
        let correctAnswer: String
        let rationale: String
    }

    private struct WaveformEntry: Decodable {
        let timeSeries: [[Float]]
        let timeSeriesText: [String]

        enum CodingKeys: String, CodingKey {
            case timeSeries = "time_series"
            case timeSeriesText = "time_series_text"
        }
    }

    private let rows: [Row]
    private let waveformsDirectory: URL
    private var waveformCache: [String: WaveformEntry] = [:]
    private let templateAnswers: [Int: [String]]

    public init(
        csvURL: URL,
        waveformsDirectory: URL,
        templateAnswersURL: URL? = nil,
        maxRows: Int = 5000
    ) throws {
        self.waveformsDirectory = waveformsDirectory
        self.rows = try Self.readRows(from: csvURL, maxRows: maxRows)
        if let templateAnswersURL {
            self.templateAnswers = try Self.readTemplateAnswers(from: templateAnswersURL)
        } else {
            self.templateAnswers = [:]
        }
    }

    public var count: Int { rows.count }

    /// The row at `index` as an evaluation sample: the exported leads with the descriptions the
    /// OpenTSLM loader wrote, and the row's values for the ECG-QA prompt's `{clinicalContext}`,
    /// `{question}` and answer options.
    public func evaluationSample(at index: Int) throws -> EvaluationSample {
        precondition(index >= 0 && index < rows.count, "sample index out of range")
        let row = rows[index]
        let waveform = try loadWaveform(for: row.ecgId)
        let series = zip(waveform.timeSeries, waveform.timeSeriesText).map { values, description in
            TimeSeries(values: values, mean: 0, std: 0, description: description)
        }
        return EvaluationSample(
            input: TSLMInput(
                series: series,
                context: ["clinicalContext": row.clinicalContext, "question": row.question],
                answerOptions: templateAnswers[row.templateId] ?? []
            ),
            label: row.correctAnswer,
            rationale: row.rationale,
            expectedAnswer: row.correctAnswer.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        )
    }

    private func loadWaveform(for ecgId: Int) throws -> WaveformEntry {
        let key = String(ecgId)
        if let cached = waveformCache[key] {
            return cached
        }

        guard let entry = loadWaveformFile(for: key) else {
            throw NSError(
                domain: "ECGQACoTDataset",
                code: 5,
                userInfo: [
                    NSLocalizedDescriptionKey:
                        "Missing PTB-XL waveform for ecg_id \(ecgId) under \(waveformsDirectory.path). "
                        + "Run scripts/export_ecg_qa_ios_assets.py to regenerate loader waveforms.",
                ]
            )
        }
        waveformCache[key] = entry
        return entry
    }

    private func loadWaveformFile(for ecgId: String) -> WaveformEntry? {
        let url = waveformsDirectory.appendingPathComponent("\(ecgId).json")
        guard FileManager.default.fileExists(atPath: url.path),
              let data = try? Data(contentsOf: url),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let rawSeries = object["time_series"] as? [[Any]],
              let rawText = object["time_series_text"] as? [String]
        else {
            return nil
        }

        guard let series = try? rawSeries.map({ lead in
            try lead.map { value -> Float in
                if let f = value as? Float { return f }
                if let d = value as? Double { return Float(d) }
                if let i = value as? Int { return Float(i) }
                throw NSError(domain: "ECGQACoTDataset", code: 7, userInfo: [NSLocalizedDescriptionKey: "Invalid waveform value"])
            }
        }) else {
            return nil
        }

        return WaveformEntry(timeSeries: series, timeSeriesText: rawText)
    }

    private static func readTemplateAnswers(from url: URL) throws -> [Int: [String]] {
        let data = try Data(contentsOf: url)
        let decoded = try JSONSerialization.jsonObject(with: data)
        guard let object = decoded as? [String: Any],
              let rawTemplates = object["template_answers"] as? [String: [Any]]
        else {
            return [:]
        }

        var answers: [Int: [String]] = [:]
        for (templateId, values) in rawTemplates {
            guard let intId = Int(templateId) else { continue }
            answers[intId] = values.compactMap { $0 as? String }
        }
        return answers
    }

    private static func readRows(from csvURL: URL, maxRows: Int) throws -> [Row] {
        guard maxRows > 0 else {
            throw NSError(domain: "ECGQACoTDataset", code: 8, userInfo: [NSLocalizedDescriptionKey: "maxRows must be > 0"])
        }

        let handle = try FileHandle(forReadingFrom: csvURL)
        let prefix = try handle.read(upToCount: 20 * 1024 * 1024) ?? Data()
        let csvPrefix = String(decoding: prefix, as: UTF8.self)
        let safePrefix = csvPrefix.prefix(upTo: csvPrefix.lastIndex(of: "\n") ?? csvPrefix.endIndex)

        let records = parseCSV(String(safePrefix))
        guard let header = records.first else {
            throw NSError(domain: "ECGQACoTDataset", code: 1, userInfo: [NSLocalizedDescriptionKey: "CSV has no header"])
        }

        let columnIndex = Dictionary(uniqueKeysWithValues: header.enumerated().map { ($1, $0) })
        guard let questionIdx = columnIndex["question"],
              let answerIdx = columnIndex["answer"],
              let templateIdx = columnIndex["template_id"],
              let ecgIdIdx = columnIndex["ecg_id"],
              let rationaleIdx = columnIndex["rationale"]
        else {
            throw NSError(domain: "ECGQACoTDataset", code: 2, userInfo: [NSLocalizedDescriptionKey: "CSV missing required ECG-QA CoT columns"])
        }

        let clinicalIdx = columnIndex["clinical_context"]

        var rows: [Row] = []
        rows.reserveCapacity(maxRows)

        for record in records.dropFirst() where rows.count < maxRows {
            guard !record.isEmpty else { continue }
            guard questionIdx < record.count,
                  answerIdx < record.count,
                  templateIdx < record.count,
                  ecgIdIdx < record.count,
                  rationaleIdx < record.count
            else { continue }

            let ecgId = try parseECGId(record[ecgIdIdx])
            let templateId = Int(record[templateIdx].trimmingCharacters(in: .whitespacesAndNewlines)) ?? 0
            let clinicalContext = (clinicalIdx != nil && clinicalIdx! < record.count && !record[clinicalIdx!].isEmpty)
                ? record[clinicalIdx!]
                : "12-lead ECG recording."

            rows.append(
                Row(
                    ecgId: ecgId,
                    templateId: templateId,
                    question: record[questionIdx],
                    clinicalContext: clinicalContext,
                    correctAnswer: record[answerIdx],
                    rationale: record[rationaleIdx]
                )
            )
        }

        guard !rows.isEmpty else {
            throw NSError(domain: "ECGQACoTDataset", code: 3, userInfo: [NSLocalizedDescriptionKey: "CSV has no ECG-QA CoT rows"])
        }

        return rows
    }

    private static func parseECGId(_ raw: String) throws -> Int {
        let cleaned = raw.trimmingCharacters(in: .whitespacesAndNewlines).trimmingCharacters(in: CharacterSet(charactersIn: "[]"))
        guard let value = Int(cleaned) else {
            throw NSError(domain: "ECGQACoTDataset", code: 4, userInfo: [NSLocalizedDescriptionKey: "Invalid ecg_id '\(raw)'"])
        }
        return value
    }

    private static func parseCSV(_ csv: String) -> [[String]] {
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
