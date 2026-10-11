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

    func assertMatches(_ sample: OpenTSLMSPSample, _ expected: Captured, file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(sample.prePrompt, expected.prePrompt, expected.name, file: file, line: line)
        XCTAssertEqual(sample.timeSeriesText, expected.timeSeriesText, expected.name, file: file, line: line)
        XCTAssertEqual(sample.postPrompt, expected.postPrompt, expected.name, file: file, line: line)
        XCTAssertEqual(sample.label, expected.label, expected.name, file: file, line: line)
        XCTAssertEqual(sample.answer, expected.answer, expected.name, file: file, line: line)
        XCTAssertEqual(sample.timeSeries.map(\.count), expected.seriesLengths, expected.name, file: file, line: line)
        XCTAssertEqual(sample.timeSeries.map(Self.sha256), expected.seriesSHA256, expected.name, file: file, line: line)
    }

    func testSleepEDFSamplesMatchTheApp() throws {
        let rows: [(count: Int, seed: UInt64, scale: Float, offset: Float, label: String, rationale: String)] = [
            (1500, 1, 2e-5, -1e-6, "W", "The signal shows fast, low-amplitude activity. Answer: Wake"),
            (1600, 2, 5e-5, 3e-6, "N2", "Spindles and K-complexes appear throughout. Answer: Non-REM stage 2")
        ]
        let checkpoint = try Self.checkpoint("sleep-edf")
        for (index, row) in rows.enumerated() {
            let series = TimeSeries(samples: Synthetic.series(count: row.count, seed: row.seed, scale: row.scale, offset: row.offset))
            let sample = try checkpoint.makeSample(TSLMInput(series: [series]), label: row.label, answer: row.rationale)
            assertMatches(sample, try Self.captured("sleep-\(index)"))
        }
    }

    func testECGQASamplesMatchTheApp() throws {
        let rows: [(ecgID: Int, question: String, context: String, options: [String], label: String, rationale: String)] = [
            (101, "Does this ECG show sinus rhythm?", "A 63-year-old patient with palpitations.", ["yes", "no", "not sure"],
             "yes", "Regular rhythm with upright P waves. Answer: yes"),
            (202, "What is the heart rhythm?", "12-lead ECG recording.", [],
             "sinus rhythm", "Normal P waves precede each QRS complex. Answer: sinus rhythm")
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
            assertMatches(try checkpoint.makeSample(input, label: row.label, answer: row.rationale), expected)
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
            let sample = try checkpoint.makeSample(
                input,
                label: "sinus rhythm",
                answer: "classification=sinus rhythm, symptoms_status=none, average_heart_rate=72.0"
            )
            assertMatches(sample, try Self.captured("watch-\(index)"))
        }
    }

    func testMissingTemplateValuesThrow() throws {
        let input = TSLMInput(series: [TimeSeries(samples: [1, 2, 3])], context: ["question": "Is this normal?"])
        XCTAssertThrowsError(try Self.checkpoint("ecg-qa").makeSample(input)) { error in
            guard case TSLMError.missingTemplateValue("clinicalContext") = error else {
                return XCTFail("unexpected error \(error)")
            }
        }
    }
}
