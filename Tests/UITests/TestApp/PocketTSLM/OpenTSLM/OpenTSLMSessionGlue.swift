//
// This source file is part of the PocketTSLM open-source project
//
// SPDX-FileCopyrightText: 2026 Schmiedmayer Lab and the project authors (see CONTRIBUTORS.md)
//
// SPDX-License-Identifier: MIT
//

import Foundation
import OSLog
import PocketTSLM
import SpeziLLM
import SpeziLLMLocal

// Spezi glue for the package's container-based generation and LoRA code: the app finds the
// adapter files and the loaded model through its `LLMLocalSession`.

extension OpenTSLMLLM {
    convenience init(llmRunner _: LLMRunner, session: LLMLocalSession) {
        self.init { await MainActor.run { session.modelContainer } }
    }
}

extension OpenTSLMLoRA {
    private static let logger = Logger(subsystem: "PocketTSLM", category: "OpenTSLMLoRA")

    static func resolveLoRAURL(checkpointName: String = Constants.openTSLMLoRACheckpointName) -> URL? {
        let fileManager = FileManager.default
        var candidates: [URL] = []

        // The env-var override always wins (regardless of task) — that's the explicit
        // "use this exact LoRA file" escape hatch.
        if !Constants.openTSLMLoRACheckpointPath.isEmpty {
            let expanded = NSString(string: Constants.openTSLMLoRACheckpointPath).expandingTildeInPath
            candidates.append(URL(fileURLWithPath: expanded))
        }

        let stagedOpenTSLM = Constants.openTSLMDocumentsDirectory
        candidates.append(stagedOpenTSLM.appendingPathComponent("\(checkpointName).safetensors"))
        candidates.append(stagedOpenTSLM.appendingPathComponent("adapter_model.safetensors"))

        if let bundled = Bundle.main.url(
            forResource: checkpointName,
            withExtension: "safetensors",
            subdirectory: Constants.openTSLMBundleSubdirectory
        ) {
            candidates.append(bundled)
        }

        // Filesystem-synchronized groups in Xcode 16 install supporting-file
        // resources at the bundle root, not under the source-tree subdirectory.
        // Fall back to a root lookup so the bundled LoRA is actually found.
        if let bundledRoot = Bundle.main.url(
            forResource: checkpointName,
            withExtension: "safetensors"
        ) {
            candidates.append(bundledRoot)
        }
        if let bundledAdapter = Bundle.main.url(
            forResource: "adapter_model",
            withExtension: "safetensors"
        ) {
            candidates.append(bundledAdapter)
        }

        if let bundleRoot = Bundle.main.resourceURL {
            let bundledOpenTSLM = bundleRoot.appendingPathComponent(Constants.openTSLMBundleSubdirectory, isDirectory: true)
            candidates.append(bundledOpenTSLM.appendingPathComponent("\(checkpointName).safetensors"))
            candidates.append(bundledOpenTSLM.appendingPathComponent("adapter_model.safetensors"))
        }

        return candidates.first { fileManager.fileExists(atPath: $0.path) }
    }

    /// Applies the task's LoRA checkpoint to the session's model. Switching tasks swaps the
    /// adapter weights in place; see ``OpenTSLMLoRA/applyIfNeeded(to:checkpointURL:)``.
    static func applyIfNeeded(
        on session: LLMLocalSession,
        checkpointName: String = Constants.openTSLMLoRACheckpointName
    ) async throws {
        guard let container = await MainActor.run(body: { session.modelContainer }) else {
            throw NSError(
                domain: "OpenTSLMLoRA",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "LLM session is not ready"]
            )
        }

        guard let checkpointURL = resolveLoRAURL(checkpointName: checkpointName) else {
            logger.warning("OpenTSLM LoRA checkpoint '\(checkpointName, privacy: .public)' not found — running base Llama (set POCKETTSLM_OPEN_TSLM_LORA_CHECKPOINT or bundle \(checkpointName).safetensors).")
            if Constants.requireLoRACheckpoint {
                throw NSError(
                    domain: "OpenTSLMLoRA",
                    code: 2,
                    userInfo: [NSLocalizedDescriptionKey: "OpenTSLM LoRA checkpoint '\(checkpointName)' not found"]
                )
            }
            return
        }

        try await applyIfNeeded(to: container, checkpointURL: checkpointURL)
    }
}
