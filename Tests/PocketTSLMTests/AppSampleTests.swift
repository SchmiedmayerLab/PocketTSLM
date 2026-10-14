//
// This source file is part of the PocketTSLM open-source project
//
// SPDX-FileCopyrightText: 2026 Schmiedmayer Lab and the project authors (see CONTRIBUTORS.md)
//
// SPDX-License-Identifier: MIT
//

import CryptoKit
@testable import PocketTSLM
import XCTest

/// Checks that the package builds exactly the model inputs the paper app built. The reference,
/// `Fixtures/app_samples.json`, comes from running the snapshot app's own code on synthetic data
/// (`scripts/capture_app_samples.sh`); these tests rebuild the same data.
final class AppSampleTests: XCTestCase {
    struct Captured: Decodable {
        let name: String
        let prePrompt: String
        let timeSeriesText: [String]
        let postPrompt: String
        let label: String
        let answer: String
        let seriesLengths: [Int]
        let seriesSHA256: [String]
    }

    /// The generator of `scripts/capture_app_samples.sh`.
    struct Synthetic {
        var state: UInt64

        mutating func next() -> UInt64 {
            state = 6364136223846793005 &* state &+ 1442695040888963407
            return state
        }

        mutating func unit() -> Float {
            Float(next() >> 40) / Float(1 << 23) - 1
        }

        static func series(count: Int, seed: UInt64, scale: Float, offset: Float) -> [Float] {
            var generator = Synthetic(state: seed)
            return (0 ..< count).map { _ in offset + scale * generator.unit() }
        }
    }

    static let fixtures = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("Fixtures")

    static func captured(_ name: String) throws -> Captured {
        let data = try Data(contentsOf: fixtures.appendingPathComponent("app_samples.json"))
        return try XCTUnwrap(JSONDecoder().decode([Captured].self, from: data).first { $0.name == name })
    }

    static func checkpoint(_ task: String) throws -> TSLMCheckpoint {
        try TSLMCheckpoint(directory: fixtures.appendingPathComponent("Checkpoints/\(task)"))
    }

    static func sha256(_ values: [Float]) -> String {
        let bytes = values.flatMap { withUnsafeBytes(of: $0.bitPattern.littleEndian, Array.init) }
        return SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined()
    }

    func assertMatches(_ prompt: TSLMPrompt, _ expected: Captured, file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(prompt.pre, expected.prePrompt, expected.name, file: file, line: line)
        XCTAssertEqual(prompt.seriesDescriptions, expected.timeSeriesText, expected.name, file: file, line: line)
        XCTAssertEqual(prompt.post, expected.postPrompt, expected.name, file: file, line: line)
        XCTAssertEqual(prompt.series.map(\.count), expected.seriesLengths, expected.name, file: file, line: line)
        XCTAssertEqual(prompt.series.map(Self.sha256), expected.seriesSHA256, expected.name, file: file, line: line)
    }

    /// The label and rationale a reader passes on, which the app carried in its sample.
    func assertMatches(_ sample: EvaluationSample, _ expected: Captured, file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(sample.label, expected.label, expected.name, file: file, line: line)
        XCTAssertEqual(sample.rationale.trimmingCharacters(in: .whitespacesAndNewlines), expected.answer, expected.name, file: file, line: line)
    }

    func testSleepEDFSamplesMatchTheApp() throws {
        struct Row {
            let count: Int
            let seed: UInt64
            let scale: Float
            let offset: Float
        }
        let rows = [Row(count: 1500, seed: 1, scale: 2e-5, offset: -1e-6), Row(count: 1600, seed: 2, scale: 5e-5, offset: 3e-6)]
        let checkpoint = try Self.checkpoint("sleep-edf")
        for (index, row) in rows.enumerated() {
            let series = TimeSeries(samples: Synthetic.series(count: row.count, seed: row.seed, scale: row.scale, offset: row.offset))
            assertMatches(try checkpoint.prompt(for: TSLMInput(series: [series])), try Self.captured("sleep-\(index)"))
        }
    }

    func testECGQASamplesMatchTheApp() throws {
        struct Row {
            let ecgID: Int
            let question: String
            let context: String
            let options: [String]
        }
        let rows = [
            Row(
                ecgID: 101,
                question: "Does this ECG show sinus rhythm?",
                context: "A 63-year-old patient with palpitations.",
                options: ["yes", "no", "not sure"]
            ),
            Row(ecgID: 202, question: "What is the heart rhythm?", context: "12-lead ECG recording.", options: [])
        ]
        let checkpoint = try Self.checkpoint("ecg-qa")
        for (index, row) in rows.enumerated() {
            let expected = try Self.captured("ecg-qa-\(index)")
            // ECG-QA exports carry normalised leads with descriptions the OpenTSLM loader wrote.
            let series = expected.timeSeriesText.indices.map { lead in
                TimeSeries(
                    values: Synthetic.series(count: 1000, seed: UInt64(row.ecgID * 100 + lead), scale: 1.5, offset: 0),
                    mean: 0, std: 0, description: expected.timeSeriesText[lead]
                )
            }
            let input = TSLMInput(
                series: series,
                context: ["clinicalContext": row.context, "question": row.question],
                answerOptions: row.options
            )
            assertMatches(try checkpoint.prompt(for: input), expected)
        }
    }

    func testWatchECGSamplesMatchTheApp() throws {
        let checkpoint = try Self.checkpoint("ecg-qa")
        // The paper app's prompt values for a recording from HealthKit.
        let context = [
            "clinicalContext": "Single-lead ambulatory ECG recording (Lead I equivalent, replicated across the standard 12 leads).",
            "question": "Is the cardiac rhythm shown in this ECG consistent with atrial fibrillation?"
        ]
        for (index, (count, seed)) in [(15360, UInt64(20)), (2048, UInt64(21))].enumerated() {
            let voltages = Synthetic.series(count: count, seed: seed, scale: 8e-4, offset: 1e-4).map(Double.init)
            let input = TSLMInput(
                series: ECG.twelveLeads(fromLeadI: voltages, samplingRate: 512),
                context: context,
                answerOptions: ["yes", "no"]
            )
            assertMatches(try checkpoint.prompt(for: input), try Self.captured("watch-\(index)"))
        }
    }

    /// Writes the synthetic exports of `scripts/capture_app_samples.sh`, which the app's readers read there.
    static func writeExports(to folder: URL) throws {
        func number(_ value: Float) -> String {
            String(format: "%.17g", Double(value))
        }
        var csv = "id,time_series,label,prompt,rationale\n"
        for (index, row) in [(1500, UInt64(1), Float(2e-5), Float(-1e-6), "W", "The signal shows fast, low-amplitude activity. Answer: Wake"),
                             (1600, 2, 5e-5, 3e-6, "N2", "Spindles and K-complexes appear throughout. Answer: Non-REM stage 2")].enumerated() {
            let series = Synthetic.series(count: row.0, seed: row.1, scale: row.2, offset: row.3)
            csv += "row\(index),\"[[\(series.map(number).joined(separator: ", "))]]\",\(row.4),,\"\(row.5)\"\n"
        }
        try csv.write(to: folder.appendingPathComponent("sleep.csv"), atomically: true, encoding: .utf8)

        let waveforms = folder.appendingPathComponent("waveforms")
        try FileManager.default.createDirectory(at: waveforms, withIntermediateDirectories: true)
        for ecgID in [101, 202] {
            let series = ECG.leadNames.indices.map { Synthetic.series(count: 1000, seed: UInt64(ecgID * 100 + $0), scale: 1.5, offset: 0) }
            let text = ECG.leadNames.indices.map { lead in
                "This is ECG Lead \(ECG.leadNames[lead]), it has mean \(String(format: "%.4f", Double(lead) * 0.0123 - 0.05)) "
                    + "and std \(String(format: "%.4f", 0.1 + Double(lead) * 0.01)):"
            }
            let json = "{\"time_series\": [\(series.map { "[\($0.map(number).joined(separator: ", "))]" }.joined(separator: ", "))], "
                + "\"time_series_text\": [\(text.map { "\"\($0)\"" }.joined(separator: ", "))]}"
            try json.write(to: waveforms.appendingPathComponent("\(ecgID).json"), atomically: true, encoding: .utf8)
        }
        try """
        question,answer,template_id,question_type,ecg_id,rationale,clinical_context
        Does this ECG show sinus rhythm?,yes,5,single-verify,[101],"Regular rhythm with upright P waves. Answer: yes",\
        A 63-year-old patient with palpitations.
        What is the heart rhythm?,sinus rhythm,9,single-query,202,"Normal P waves precede each QRS complex. Answer: sinus rhythm",

        """.write(to: folder.appendingPathComponent("ecg.csv"), atomically: true, encoding: .utf8)
        try #"{"template_answers": {"5": ["yes", "no", "not sure"]}}"#
            .write(to: folder.appendingPathComponent("templates.json"), atomically: true, encoding: .utf8)
    }

    func testReadersMatchTheApp() throws {
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }
        try Self.writeExports(to: folder)

        let sleep = try SleepEDFDataset.evaluationSamples(csvURL: folder.appendingPathComponent("sleep.csv"), count: 2)
        XCTAssertEqual(sleep.map(\.expectedAnswer), ["wake", "non-rem stage 2"])
        for (index, sample) in sleep.enumerated() {
            assertMatches(try Self.checkpoint("sleep-edf").prompt(for: sample.input), try Self.captured("sleep-\(index)"))
            assertMatches(sample, try Self.captured("sleep-\(index)"))
        }

        let ecg = try ECGQACoTDataset(
            csvURL: folder.appendingPathComponent("ecg.csv"),
            waveformsDirectory: folder.appendingPathComponent("waveforms"),
            templateAnswersURL: folder.appendingPathComponent("templates.json")
        )
        for index in 0 ..< ecg.count {
            let sample = try ecg.evaluationSample(at: index)
            XCTAssertEqual(sample.expectedAnswer, ["yes", "sinus rhythm"][index])
            assertMatches(try Self.checkpoint("ecg-qa").prompt(for: sample.input), try Self.captured("ecg-qa-\(index)"))
            assertMatches(sample, try Self.captured("ecg-qa-\(index)"))
        }
    }

    func testMissingTemplateValuesThrow() throws {
        let input = TSLMInput(series: [TimeSeries(samples: [1, 2, 3])], context: ["question": "Is this normal?"])
        XCTAssertThrowsError(try Self.checkpoint("ecg-qa").prompt(for: input)) { error in
            guard case TSLMError.missingTemplateValue("clinicalContext") = error else {
                return XCTFail("unexpected error \(error)")
            }
        }
    }
}
