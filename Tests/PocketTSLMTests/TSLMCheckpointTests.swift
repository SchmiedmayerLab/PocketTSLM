//
// This source file is part of the PocketTSLM open-source project
//
// SPDX-FileCopyrightText: 2026 Schmiedmayer Lab and the project authors (see CONTRIBUTORS.md)
//
// SPDX-License-Identifier: MIT
//

@testable import PocketTSLM
import XCTest

final class TSLMCheckpointTests: XCTestCase {
    static let checkpoints = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .appendingPathComponent("Fixtures/Checkpoints")

    func testReadsConverterOutput() throws {
        let ecg = try TSLMCheckpoint(directory: Self.checkpoints.appendingPathComponent("ecg-qa")).configuration
        XCTAssertEqual(ecg.task, "ecg-qa")
        XCTAssertEqual(ecg.baseModel.id, "meta-llama/Llama-3.2-1B")
        XCTAssertEqual(ecg.baseModel.hiddenSize, 2048)
        XCTAssertEqual(ecg.encoder.maxPatches, 1024)
        XCTAssertEqual(ecg.lora.rank, 16)
        XCTAssertEqual(ecg.lora.alpha, 32)
        XCTAssertEqual(ecg.lora.keys.count, 7)
        XCTAssertEqual(ecg.series.names?.count, 12)
        XCTAssertEqual(ecg.series.maxLength, 1000)
        XCTAssertEqual(ecg.generation.maxTokens, 400)
        XCTAssertNotNil(ecg.prompt.postWithoutOptions)

        let sleep = try TSLMCheckpoint(directory: Self.checkpoints.appendingPathComponent("sleep-edf")).configuration
        XCTAssertEqual(sleep.task, "sleep-edf")
        XCTAssertNil(sleep.series.names)
        XCTAssertEqual(sleep.series.maxLength, 1500)
        XCTAssertEqual(sleep.generation.maxTokens, 200)
    }

    func testWeightLocations() throws {
        let directory = Self.checkpoints.appendingPathComponent("ecg-qa")
        let checkpoint = try TSLMCheckpoint(directory: directory)
        XCTAssertEqual(checkpoint.encoderURL, directory.appendingPathComponent("encoder.safetensors"))
        XCTAssertEqual(checkpoint.loraURL.lastPathComponent, "lora.safetensors")
    }

    func testRejectsOtherFormatsAndUnsafeStatFormats() throws {
        let source = try Data(contentsOf: Self.checkpoints.appendingPathComponent("ecg-qa/config.json"))
        for (old, new) in [("pockettslm-checkpoint/1", "pockettslm-checkpoint/2"), ("\"%.4f\"", "\"%@\"")] {
            let folder = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: folder) }
            let changed = String(decoding: source, as: UTF8.self).replacingOccurrences(of: old, with: new)
            try Data(changed.utf8).write(to: folder.appendingPathComponent("config.json"))
            XCTAssertThrowsError(try TSLMCheckpoint(directory: folder))
        }
    }
}
