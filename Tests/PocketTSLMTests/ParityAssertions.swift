//
// This source file is part of the PocketTSLM open-source project
//
// SPDX-FileCopyrightText: 2026 Schmiedmayer Lab and the project authors (see CONTRIBUTORS.md)
//
// SPDX-License-Identifier: MIT
//

import MLX
import XCTest

extension XCTestCase {
    /// Skips a test that runs MLX on the iOS Simulator: MLX needs a Metal GPU, which the simulator
    /// does not provide, so MLX crashes the test process there.
    func skipWithoutMetalGPU() throws {
        #if targetEnvironment(simulator)
        throw XCTSkip("MLX needs a Metal GPU, which the iOS Simulator does not provide")
        #endif
    }

    /// Asserts that `output` matches a Python reference output element-wise within `maxTolerance`,
    /// and on average within `meanTolerance`.
    func assertParity(
        _ output: MLXArray,
        _ reference: MLXArray,
        _ label: String,
        maxTolerance: Float = 1e-4,
        meanTolerance: Float = 1e-5,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertEqual(output.shape, reference.shape, "\(label): shape", file: file, line: line)
        guard output.shape == reference.shape else {
            return
        }
        let difference = (output - reference).abs()
        let maxDifference = difference.max().item(Float.self)
        let meanDifference = difference.mean().item(Float.self)
        print("\(label) parity: max_diff=\(maxDifference) mean_diff=\(meanDifference)")
        XCTAssertLessThan(maxDifference, maxTolerance, "\(label): max abs diff", file: file, line: line)
        XCTAssertLessThan(meanDifference, meanTolerance, "\(label): mean abs diff", file: file, line: line)
    }
}
