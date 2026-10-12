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
import Tokenizers

/// How the series of a prompt reach the model.
public enum SeriesEncoding: String, Sendable {
    /// Encoder and projector outputs placed between the text embeddings: OpenTSLM-SP's soft prompt.
    case softPrompt
    /// OpenTSLM's text serialisation, read as ordinary tokens: the paper's §4 baseline.
    case text
}

/// A generated answer and what it cost.
public struct TSLMGeneration: Sendable {
    public let text: String
    /// Generated tokens, the stop token included.
    public let tokenCount: Int
    /// Positions the model attends over before the first generated token.
    public let promptPositions: Int
    /// The part of ``promptPositions`` that the series take.
    public let seriesPositions: Int
    /// From the start of the prefill to the first generated token.
    public let prefillSeconds: Double
    /// From the start of the prefill to the last generated token.
    public let totalSeconds: Double
    /// MLX's peak memory from the start of the prefill.
    public let peakMemoryBytes: Int
}

extension EmbeddingLlamaModel {
    /// Answers `prompt` with greedy decoding, as OpenTSLM-MLX's `opentslm_sp.py` does: argmax with no
    /// temperature, top-p or repetition penalty, until a Llama 3 stop token or `maxTokens`.
    /// - Parameter projected: The projected series for ``SeriesEncoding/softPrompt``.
    func generate(
        _ prompt: TSLMPrompt,
        projected: [MLXArray],
        encoding: SeriesEncoding,
        tokenizer: Tokenizer,
        maxTokens: Int,
        maxPromptPositions: Int?
    ) throws -> TSLMGeneration {
        let (embedding, seriesPositions) = promptEmbedding(prompt, projected: projected, encoding: encoding, tokenizer: tokenizer)
        let promptPositions = embedding.dim(1)
        // Checked before `eval`, so a skipped prompt is never materialised: on iOS, running out of
        // memory ends the app instead of throwing.
        if let maxPromptPositions, promptPositions > maxPromptPositions {
            throw TSLMError.promptTooLong(positions: promptPositions, seriesPositions: seriesPositions, limit: maxPromptPositions)
        }
        eval(embedding)

        // Llama 3 ends with `<|end_of_text|>` or `<|eot_id|>`; the reference stops at either.
        var stopTokens = Set(tokenizer.eosTokenId.map { [$0] } ?? [])
        for marker in ["<|eot_id|>", "<|end_of_text|>"] {
            let ids = tokenizer.encode(text: marker, addSpecialTokens: false)
            if ids.count == 1 {
                stopTokens.insert(ids[0])
            }
        }

        GPU.resetPeakMemory()
        let start = DispatchTime.now()
        func elapsed() -> Double {
            Double(DispatchTime.now().uptimeNanoseconds - start.uptimeNanoseconds) / 1_000_000_000
        }

        // The prefill runs in chunks of 512 positions into the KV cache. The logits are the same as
        // in one pass, but peak memory stays bounded: a 12-lead ECG prompt has over 3000 positions.
        let cache = newCache(parameters: nil)
        var logits = MLXArray([])
        for offset in stride(from: 0, to: promptPositions, by: 512) {
            logits = self(nil, cache: cache, inputEmbedding: embedding[0 ..< 1, offset ..< min(offset + 512, promptPositions)])
            eval(logits)
            GPU.clearCache()
        }

        var tokens: [Int] = []
        var prefillSeconds = 0.0
        while tokens.count < maxTokens {
            // `item` evaluates, so the timings measure computation rather than graph building.
            let token = argMax(logits[0 ..< 1, (logits.dim(1) - 1)...], axis: -1).item(Int.self)
            if tokens.isEmpty {
                prefillSeconds = elapsed()
            }
            tokens.append(token)
            if stopTokens.contains(token) {
                break
            }
            logits = self(MLXArray([Int32(token)], [1, 1]), cache: cache, inputEmbedding: nil)
        }

        return TSLMGeneration(
            text: tokenizer.decode(tokens: tokens, skipSpecialTokens: true),
            tokenCount: tokens.count,
            promptPositions: promptPositions,
            seriesPositions: seriesPositions,
            prefillSeconds: prefillSeconds,
            totalSeconds: elapsed(),
            peakMemoryBytes: GPU.peakMemory
        )
    }

    /// The input sequence `[1, positions, hidden]`: the pre-prompt, each series description followed
    /// by its series, and the post-prompt. Each text segment starts with BOS, because the reference
    /// tokenises the segments separately with special tokens; serialised series are inline data
    /// without BOS.
    private func promptEmbedding(
        _ prompt: TSLMPrompt,
        projected: [MLXArray],
        encoding: SeriesEncoding,
        tokenizer: Tokenizer
    ) -> (embedding: MLXArray, seriesPositions: Int) {
        func embed(_ text: String, addSpecialTokens: Bool = true) -> MLXArray {
            let ids = tokenizer.encode(text: text, addSpecialTokens: addSpecialTokens)
            return tokenEmbeddings(MLXArray(ids.map { Int32($0) }, [1, ids.count]))[0]
        }

        let pre = embed(prompt.pre)
        var segments = [pre]
        var seriesPositions = 0
        for (index, description) in prompt.seriesDescriptions.enumerated() {
            segments.append(embed(description))
            let series = switch encoding {
            case .softPrompt: projected[index].asType(pre.dtype)
            case .text: embed(TimeSeriesTextSerializer.serialize(prompt.series[index]), addSpecialTokens: false)
            }
            segments.append(series)
            seriesPositions += series.dim(0)
        }
        segments.append(embed(prompt.post))

        let sequence = concatenated(segments, axis: 0)
        return (sequence.expandedDimensions(axis: 0), seriesPositions)
    }
}
