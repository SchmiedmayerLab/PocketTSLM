//
// This source file is part of the PocketTSLM open-source project
//
// SPDX-FileCopyrightText: 2026 Schmiedmayer Lab and the project authors (see CONTRIBUTORS.md)
//
// SPDX-License-Identifier: MIT
//

@testable import PocketTSLM
import XCTest

final class EvaluationTests: XCTestCase {
    func testFinalAnswer() {
        XCTAssertEqual(Evaluation.finalAnswer(in: "Low-amplitude, fast activity. Answer: Wake"), "wake")
        XCTAssertEqual(Evaluation.finalAnswer(in: "Spindles appear. Answer: Non-REM stage 2."), "non-rem stage 2")
        XCTAssertEqual(Evaluation.finalAnswer(in: "Answer: no. Later: answer:  YES <|end_of_text|>"), "yes")
        XCTAssertEqual(Evaluation.finalAnswer(in: "Answer: sinus\n rhythm"), "sinus")
        XCTAssertNil(Evaluation.finalAnswer(in: "The rationale never reaches a verdict"))
        XCTAssertNil(Evaluation.finalAnswer(in: "Answer: \n"))
    }

    func testReportJSONMatchesTheAppFormat() throws {
        let row = Evaluation.Row(index: 0, label: "W", groundTruth: "wake", predicted: nil, correct: false, seconds: 1.25, output: "…")
        let report = Evaluation.Report(
            sampleCount: 1, maxNewTokens: 200, seriesLength: 1500, loraApplied: true,
            parsed: 0, correct: 0, accuracy: 0, rows: [row]
        )
        let json = try XCTUnwrap(String(data: report.json(), encoding: .utf8))
        let keys = ["accuracy", "correct", "loraApplied", "maxNewTokens", "parsed", "rows", "sampleCount", "seriesLength"]
        let positions = keys.compactMap { json.range(of: "\"\($0)\"")?.lowerBound }
        XCTAssertEqual(positions.count, keys.count)
        XCTAssertEqual(positions, positions.sorted(), "keys must be sorted, as the app wrote them")
    }
}
