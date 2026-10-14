//
// This source file is part of the PocketTSLM open-source project
//
// SPDX-FileCopyrightText: 2026 Schmiedmayer Lab and the project authors (see CONTRIBUTORS.md)
//
// SPDX-License-Identifier: MIT
//

import XCTest

@testable import PocketTSLM

/// Verifies the Swift serializer is byte-identical to OpenTSLM's Python
/// `QADataset` formatter, so token counts in the §4 baseline are comparable to
/// the published reference.
final class TimeSeriesTextSerializerTests: XCTestCase {

    func testSingleValueFormat() {
        XCTAssertEqual(TimeSeriesTextSerializer.format(0.12), "\"012\"")
        XCTAssertEqual(TimeSeriesTextSerializer.format(-1.453), "\"-145\"")
        XCTAssertEqual(TimeSeriesTextSerializer.format(2.3), "\"230\"")
        XCTAssertEqual(TimeSeriesTextSerializer.format(0.0), "\"000\"")
        XCTAssertEqual(TimeSeriesTextSerializer.format(-0.05), "\"-005\"")
        XCTAssertEqual(TimeSeriesTextSerializer.format(12.7), "\"1270\"")
    }

    func testSeriesMatchesPythonReference() {
        let series: [Float] = [0.12, -1.453, 2.3, 0.0, -0.05, 12.7]
        // Exact output of OpenTSLM's np.array2string(...).removeprefix("[").removesuffix("]")
        let expected = "\"012\" \"-145\" \"230\" \"000\" \"-005\" \"1270\""
        XCTAssertEqual(TimeSeriesTextSerializer.serialize(series), expected)
    }

    func testRoundingToTwoDecimals() {
        // 0.125 -> 0.12 (round-half-to-even), 0.135 -> 0.14 ; both drop the dot.
        XCTAssertEqual(TimeSeriesTextSerializer.format(0.129), "\"013\"")
        XCTAssertEqual(TimeSeriesTextSerializer.format(1.999), "\"200\"")
    }
}
