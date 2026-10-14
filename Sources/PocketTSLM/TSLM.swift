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
    let pipeline: OpenTSLMSPPipeline
    /// The base model, loaded after ``registerModelTypes()``.
    let base: ModelContainer

    /// Loads the checkpoint's encoder and projector; its LoRA adapter is applied to `base` on first use.
    public init(base: ModelContainer, checkpoint: TSLMCheckpoint) throws {
        pipeline = try OpenTSLMSPPipeline(checkpoint: checkpoint)
        self.base = base
        self.checkpoint = checkpoint
    }

    /// Answers an input with greedy decoding.
    /// - Parameter maxTokens: Defaults to the checkpoint's `generation.maxTokens`.
    public func generate(_ input: TSLMInput, maxTokens: Int? = nil) async throws -> String {
        try await generate(checkpoint.prompt(for: input), maxTokens: maxTokens).text
    }

    /// Answers a prompt that ``TSLMCheckpoint/prompt(for:)`` built, with greedy decoding, and
    /// measures the run.
    /// - Parameters:
    ///   - encoding: How the series reach the model; ``SeriesEncoding/text`` is the paper's baseline.
    ///   - maxTokens: Defaults to the checkpoint's `generation.maxTokens`.
    ///   - maxPromptPositions: Throws ``TSLMError/promptTooLong(positions:seriesPositions:limit:)``
    ///     instead of starting a longer prefill.
    public func generate(
        _ prompt: TSLMPrompt,
        seriesAs encoding: SeriesEncoding = .softPrompt,
        maxTokens: Int? = nil,
        maxPromptPositions: Int? = nil
    ) async throws -> TSLMGeneration {
        let maxTokens = maxTokens ?? checkpoint.configuration.generation.maxTokens
        let projected = encoding == .softPrompt ? pipeline.projectTimeSeries(prompt.series) : nil
        GPU.clearCache()
        let lora = checkpoint.configuration.lora
        let loraURL = checkpoint.loraURL
        let shape = OpenTSLMLoRA.Shape(rank: lora.rank, scale: lora.alpha / Float(lora.rank), keys: Set(lora.keys))
        // The adapter is activated in the same `perform` as the generation, so a `TSLM` sharing the
        // base model cannot swap in its own adapter in between.
        return try await base.perform { context in
            guard let model = context.model as? EmbeddingLlamaModel else {
                throw TSLMError.unsupportedBaseModel(String(describing: type(of: context.model)))
            }
            try OpenTSLMLoRA.applyIfNeeded(to: model, checkpointURL: loraURL, shape: shape)
            GPU.clearCache()
            return try model.generate(
                prompt,
                projected: projected,
                tokenizer: context.tokenizer,
                maxTokens: maxTokens,
                maxPromptPositions: maxPromptPositions
            )
        }
    }
}
