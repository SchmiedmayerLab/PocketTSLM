//
// MLXEmbeddingGenerator.swift
// Custom MLX generation loop that accepts pre-computed embeddings (inputs_embeds)
//
// Note: This is a best-effort implementation that targets the MLX/MLXLLM API surface.
// If signatures differ in your installed MLX version, adapt calls accordingly.
//

import Foundation
import MLX

public enum MLXEmbeddingGeneratorError: Error {
    case modelDoesNotSupportEmbeddingGeneration
    case generationFailed(String)
}

/// A small protocol describing the minimum functionality needed for embedding-primed generation.
/// This keeps the implementation decoupled from a specific MLXLLM version.
public protocol EmbeddingPrimedLanguageModel {
    associatedtype Cache

    func makeCache() -> Cache
    func callAsFunction(_ inputs: MLXArray?, cache: Cache, inputEmbedding: MLXArray?) throws -> MLXArray
}

/// Minimal helper implementing a generation loop that uses pre-computed `inputs_embeds`.
/// The model supplies logits; token sampling is injected so this stays compatible with the
/// exact MLX version available in the app target.
public final class MLXEmbeddingGenerator<Model: EmbeddingPrimedLanguageModel> {
    private let model: Model
    private let eosTokenIds: Set<Int>
    private let tokenSampler: (MLXArray) throws -> Int
    private let decodeTokens: ([Int]) throws -> String

    public init(
        model: Model,
        eosTokenIds: Set<Int>,
        tokenSampler: @escaping (MLXArray) throws -> Int,
        decodeTokens: @escaping ([Int]) throws -> String
    ) {
        self.model = model
        self.eosTokenIds = eosTokenIds
        self.tokenSampler = tokenSampler
        self.decodeTokens = decodeTokens
    }

    /// Result of an instrumented generation run — used by the §4 benchmark.
    public struct Measured {
        /// Decoded output text.
        public let text: String
        /// Generated token ids (length = number of decode steps, EOS included).
        public let tokenIds: [Int]
        /// Wall-clock from start to the first sampled token (prefill + first forward).
        public let prefillSeconds: Double
        /// Wall-clock from start to completion (prefill + full decode).
        public let totalSeconds: Double
    }

    /// Identical decode loop to ``generate(inputsEmbeds:maxNewTokens:temperature:)``
    /// but returns token ids and timing so callers can report end-to-end latency
    /// and decode throughput. The token sampler forces a synchronous `eval` each
    /// step (via `.item`), so the captured wall-clock brackets real compute rather
    /// than MLX's lazy graph building.
    public func generateMeasured(
        inputsEmbeds: MLXArray,
        maxNewTokens: Int = 128,
        temperature: Float = 1.0,
        prefillStepSize: Int = 512
    ) throws -> Measured {
        let start = DispatchTime.now()
        var prefillSeconds = 0.0

        let cache = model.makeCache()
        // Chunked prefill (see generate(_:)): feed the prompt in fixed slices into the shared
        // KV cache so peak activation memory stays bounded — required for long multi-lead
        // (e.g. 12-lead ECG) soft prompts that would otherwise OOM at single-shot prefill.
        // Identical logits to single-shot; prefillSeconds still brackets the whole prefill.
        let promptLength = inputsEmbeds.dim(1)
        var logits: MLXArray = MLXArray([])
        var offset = 0
        while offset < promptLength {
            let end = min(offset + max(prefillStepSize, 1), promptLength)
            logits = try model.callAsFunction(nil, cache: cache, inputEmbedding: inputsEmbeds[0 ..< 1, offset ..< end])
            eval(logits)
            GPU.clearCache()
            offset = end
        }
        var generatedTokenIds: [Int] = []

        for step in 0 ..< maxNewTokens {
            guard logits.ndim >= 2 else {
                throw MLXEmbeddingGeneratorError.generationFailed("Unexpected logits ndim: \(logits.ndim)")
            }
            let seqLen = Int(logits.dim(1))
            guard seqLen > 0 else {
                throw MLXEmbeddingGeneratorError.generationFailed("Logits sequence length is zero")
            }

            let lastLogits = logits[0 ..< 1, seqLen - 1 ..< seqLen] / Float(temperature)
            let nextToken = try tokenSampler(lastLogits)   // forces eval via `.item`
            if step == 0 {
                prefillSeconds = Double(DispatchTime.now().uptimeNanoseconds - start.uptimeNanoseconds) / 1_000_000_000
            }
            generatedTokenIds.append(nextToken)

            if eosTokenIds.contains(nextToken) {
                break
            }

            let inputIds = MLXArray([Int32(nextToken)], [1, 1])
            logits = try model.callAsFunction(inputIds, cache: cache, inputEmbedding: nil)
        }

        let totalSeconds = Double(DispatchTime.now().uptimeNanoseconds - start.uptimeNanoseconds) / 1_000_000_000
        let text = try decodeTokens(generatedTokenIds)
        return Measured(text: text, tokenIds: generatedTokenIds, prefillSeconds: prefillSeconds, totalSeconds: totalSeconds)
    }

    /// Generate text starting from pre-computed embeddings.
    public func generate(
        inputsEmbeds: MLXArray,
        maxNewTokens: Int = 128,
        temperature: Float = 1.0,
        prefillStepSize: Int = 512
    ) throws -> String {
        let cache = model.makeCache()

        // Prefill in chunks rather than one forward over the whole sequence. The 12-lead
        // ECG soft prompt is ~3000+ tokens; prefilling it all at once allocates attention/
        // activation buffers for the full length and blows the iOS process limit. Feeding
        // fixed-size chunks into the shared KV cache bounds peak memory to one chunk while
        // producing identical logits (same math, just split). Only the final chunk's logits
        // are needed to start decoding.
        let promptLength = inputsEmbeds.dim(1)
        var logits: MLXArray = MLXArray([])
        var offset = 0
        while offset < promptLength {
            let end = min(offset + max(prefillStepSize, 1), promptLength)
            let chunk = inputsEmbeds[0 ..< 1, offset ..< end]
            logits = try model.callAsFunction(nil, cache: cache, inputEmbedding: chunk)
            eval(logits)          // force the forward (and cache update) for this chunk…
            GPU.clearCache()      // …then reclaim freed buffers before the next chunk
            offset = end
        }
        var generatedTokenIds: [Int] = []

        for _ in 0..<maxNewTokens {
            guard logits.ndim >= 2 else {
                throw MLXEmbeddingGeneratorError.generationFailed("Unexpected logits ndim: \(logits.ndim)")
            }

            let seqLen = Int(logits.dim(1))
            guard seqLen > 0 else {
                throw MLXEmbeddingGeneratorError.generationFailed("Logits sequence length is zero")
            }

            let lastLogitsSlice = logits[0 ..< 1, seqLen - 1 ..< seqLen]
            let lastLogits = lastLogitsSlice / Float(temperature)
            let nextToken = try tokenSampler(lastLogits)
            generatedTokenIds.append(nextToken)

            if eosTokenIds.contains(nextToken) {
                break
            }

            // Integer index array — the embedding lookup requires integer ids,
            // not the float array produced by `MLXArray(converting:)`.
            let inputIds = MLXArray([Int32(nextToken)], [1, 1])
            logits = try model.callAsFunction(inputIds, cache: cache, inputEmbedding: nil)
        }

        return try decodeTokens(generatedTokenIds)
    }
}
