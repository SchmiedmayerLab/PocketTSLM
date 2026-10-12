//
// This source file is part of the PocketTSLM open-source project
//
// SPDX-FileCopyrightText: 2026 Schmiedmayer Lab and the project authors (see CONTRIBUTORS.md)
//
// SPDX-License-Identifier: MIT
//

import Foundation
import MLX
import MLXLMCommon

/// A time-series language model: a loaded Llama base model plus one OpenTSLM-SP checkpoint.
///
/// Several `TSLM`s can share one base model, for example one per task. Each generation
/// activates its own LoRA adapter first, which swaps the adapter weights in place.
public final class TSLM {
    public let checkpoint: TSLMCheckpoint
    /// The checkpoint's time-series encoder and projector.
    public let pipeline: OpenTSLMSPPipeline
    /// The base model, loaded after ``registerModelTypes()``.
    public let base: ModelContainer

    /// Loads the checkpoint's encoder and projector; its LoRA adapter is applied to `base` on first use.
    public init(base: ModelContainer, checkpoint: TSLMCheckpoint) throws {
        let configuration = checkpoint.configuration
        pipeline = OpenTSLMSPPipeline(
            hiddenSize: configuration.baseModel.hiddenSize,
            patchSize: configuration.encoder.patchSize,
            maxPatches: configuration.encoder.maxPatches
        )
        try pipeline.loadWeights(encoderURL: checkpoint.encoderURL, projectorURL: checkpoint.projectorURL)
        self.base = base
        self.checkpoint = checkpoint
    }

    /// Answers an input with greedy decoding.
    /// - Parameter maxTokens: Defaults to the checkpoint's `generation.maxTokens`.
    public func generate(_ input: TSLMInput, maxTokens: Int? = nil) async throws -> String {
        try await generate(checkpoint.prompt(for: input), maxTokens: maxTokens)
    }

    /// Answers a prompt that ``TSLMCheckpoint/prompt(for:)`` built, with greedy decoding.
    public func generate(_ prompt: TSLMPrompt, maxTokens: Int? = nil) async throws -> String {
        let projected = pipeline.projectTimeSeries(prompt.series)
        GPU.clearCache()
        try await activateAdapter()
        GPU.clearCache()
        let base = base
        return try await OpenTSLMLLM { base }.generate(
            prePrompt: prompt.pre,
            timeSeriesText: prompt.seriesDescriptions,
            timeSeriesEmbeddings: projected,
            postPrompt: prompt.post,
            maxTokens: maxTokens ?? checkpoint.configuration.generation.maxTokens
        )
    }

    /// Makes this checkpoint's LoRA adapter the active one on the base model.
    public func activateAdapter() async throws {
        let lora = checkpoint.configuration.lora
        let shape = OpenTSLMLoRA.Shape(rank: lora.rank, scale: lora.alpha / Float(lora.rank), keys: Set(lora.keys))
        try await OpenTSLMLoRA.applyIfNeeded(to: base, checkpointURL: checkpoint.loraURL, shape: shape)
    }
}
