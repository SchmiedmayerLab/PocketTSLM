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
import OSLog
import PocketTSLM
import Tokenizers

/// Context-economics benchmark for the paper's §4 headline comparison: the same
/// raw series and the same on-device backbone, encoded two ways —
/// representation-level **soft prompts** (encoder → projector → `inputs_embeds`)
/// vs. prompt-level **text serialization** (OpenTSLM's `"012"` format fed as
/// ordinary tokens) — swept over series length.
///
/// Both arms decode through the *exact same* ``MLXEmbeddingGenerator`` (greedy
/// argmax, same stop tokens) so any difference in latency, throughput, or memory
/// is attributable to the encoding alone, not the decoder. The only structural
/// difference is the time-series region: projected embeddings vs. serialized text
/// tokens. The surrounding text (pre-prompt, per-series label, post-prompt) is
/// tokenized identically for both, so prompt-position differences isolate the
/// time-series cost.
///
/// Confounders handled / noted:
///  - Same backbone weights and decoder for both arms; greedy decoding for
///    determinism. The checkpoint's LoRA is shared by both — it changes
///    generated *content*, not prefill/throughput/memory, so the headline
///    metrics are unaffected (documented in the report header).
///  - Soft prompts cost sequence *positions* but zero vocabulary tokens; the text
///    arm costs BPE tokens. We report positions for both (the currency that drives
///    prefill, memory, and the context ceiling) plus the time-series sub-count.
///  - MLX lazy eval: the generator forces a synchronous eval per token, so
///    wall-clock brackets real compute.
public final class OpenTSLMBenchmark {

    /// Per-encoding measurements at one series length.
    public struct ArmStats: Encodable {
        public let encoding: String           // "soft_prompt" | "text_serialized"
        public let seriesLength: Int
        public let promptPositions: Int        // total positions the LLM attends over
        public let timeSeriesPositions: Int    // positions used by the time-series region
        public let generatedTokens: Int
        public let prefillSeconds: Double       // start → first token
        public let totalSeconds: Double         // start → done (end-to-end)
        public let decodeTokensPerSecond: Double
        public let peakMemoryBytes: Int
        public let didOOM: Bool                  // exceeded budget or generation threw
        public let outputText: String
    }

    /// Both arms compared at one series length.
    public struct LengthComparison: Encodable {
        public let seriesLength: Int
        public let softPrompt: ArmStats
        public let textSerialized: ArmStats
        /// text time-series positions / soft-prompt positions (>1 = soft prompt wins).
        public let positionCompression: Double
    }

    /// Full benchmark output.
    public struct Report: Encodable {
        public let contextWindow: Int
        public let loraApplied: Bool
        public let rows: [LengthComparison]
        /// Max raw series length that fits `contextWindow` under each encoding,
        /// extrapolated from the marginal positions-per-sample at the longest length.
        public let textMaxSeriesLength: Int
        public let softPromptMaxSeriesLength: Int
    }

    /// Where the time-series region of the prompt comes from for an arm.
    private enum TimeSeriesRegion {
        case embeddings([MLXArray])   // projected soft prompts
        case serializedText           // OpenTSLM "012" text tokens
    }

    private let logger = Logger(subsystem: "PocketTSLM", category: "OpenTSLMBenchmark")
    private let model: TSLM

    public init(model: TSLM) {
        self.model = model
    }

    /// Run both encodings across a series-length sweep.
    /// - Parameters:
    ///   - maxPromptPositions: on-device memory guard. Prompts longer than this are
    ///     *skipped* (recorded as `didOOM`) rather than attempted, because the long
    ///     text-serialized prefill's KV-cache + activations can trigger an
    ///     un-catchable jetsam termination. The text arm crosses this first; the
    ///     soft-prompt arm (~L/4 positions) stays well under it — which is the §4
    ///     point. Raise it to probe the true device ceiling (at OOM risk).
    ///   - contextWindow: the backbone's theoretical context length, used only for
    ///     the extrapolated context-ceiling figures (not the runtime guard).
    public func run(
        sample: OpenTSLMSPSample,
        lengths: [Int],
        maxTokens: Int = 200,
        contextWindow: Int = 8192,
        maxPromptPositions: Int = 2048
    ) async throws -> Report {
        try await model.activateAdapter()
        GPU.clearCache()
        let container = model.base

        var rows: [LengthComparison] = []
        for length in lengths {
            let truncated = Self.truncate(sample, to: length)
            // Soft-prompt embeddings are produced outside the decoder container,
            // mirroring OpenTSLMInferenceService's projection step (projectSample
            // already forces an eval internally).
            let projected = model.pipeline.projectSample(truncated)

            let comparison: LengthComparison = try await container.perform { context in
                guard let model = context.model as? EmbeddingLlamaModel else {
                    throw NSError(
                        domain: "OpenTSLMBenchmark", code: 2,
                        userInfo: [NSLocalizedDescriptionKey:
                            "Loaded model is not EmbeddingLlamaModel — ensure registerModelTypes() ran."])
                }
                let tokenizer = context.tokenizer

                let sp = try self.runArm(
                    encoding: "soft_prompt", sample: truncated, region: .embeddings(projected),
                    model: model, tokenizer: tokenizer, maxTokens: maxTokens, maxPromptPositions: maxPromptPositions)
                GPU.clearCache()
                let txt = try self.runArm(
                    encoding: "text_serialized", sample: truncated, region: .serializedText,
                    model: model, tokenizer: tokenizer, maxTokens: maxTokens, maxPromptPositions: maxPromptPositions)
                GPU.clearCache()

                let compression = sp.timeSeriesPositions > 0
                    ? Double(txt.timeSeriesPositions) / Double(sp.timeSeriesPositions)
                    : 0   // 0 = undefined; keeps the report JSON-encodable (no NaN)
                return LengthComparison(
                    seriesLength: truncated.timeSeries.first?.count ?? length,
                    softPrompt: sp, textSerialized: txt, positionCompression: compression)
            }
            rows.append(comparison)
            self.logger.info("benchmark L=\(comparison.seriesLength, privacy: .public): text=\(comparison.textSerialized.promptPositions, privacy: .public) pos, sp=\(comparison.softPrompt.promptPositions, privacy: .public) pos, compression=\(comparison.positionCompression, privacy: .public)x")
            GPU.clearCache()
        }

        let (textMax, spMax) = Self.contextCeiling(rows: rows, contextWindow: contextWindow)
        return Report(
            contextWindow: contextWindow, loraApplied: true, rows: rows,
            textMaxSeriesLength: textMax, softPromptMaxSeriesLength: spMax)
    }

    // MARK: - One arm

    private func runArm(
        encoding: String,
        sample: OpenTSLMSPSample,
        region: TimeSeriesRegion,
        model: EmbeddingLlamaModel,
        tokenizer: Tokenizer,
        maxTokens: Int,
        maxPromptPositions: Int
    ) throws -> ArmStats {
        let seriesLength = sample.timeSeries.first?.count ?? 0

        // Embed a text segment to `[T, hidden]` via the model's own token table.
        // Pre/label/post add special tokens (BOS) to match OpenTSLMLLM / the Python
        // reference; the serialized numbers are inline data (no extra BOS) so the
        // text scaffold stays identical to the soft-prompt arm.
        func textEmbedding(_ text: String, addSpecialTokens: Bool) -> MLXArray {
            let ids = tokenizer.encode(text: text, addSpecialTokens: addSpecialTokens)
            let idArray = MLXArray(ids.map { Int32($0) }, [1, ids.count])
            return model.tokenEmbeddings(idArray)[0]
        }

        let pre = textEmbedding(sample.prePrompt, addSpecialTokens: true)
        let dtype = pre.dtype
        var segments: [MLXArray] = [pre]
        var timeSeriesPositions = 0

        switch region {
        case .embeddings(let embeddings):
            for (text, embedding) in zip(sample.timeSeriesText, embeddings) {
                segments.append(textEmbedding(text, addSpecialTokens: true))
                let typed = embedding.asType(dtype)
                segments.append(typed)
                timeSeriesPositions += typed.dim(0)
            }
        case .serializedText:
            for (text, series) in zip(sample.timeSeriesText, sample.timeSeries) {
                segments.append(textEmbedding(text, addSpecialTokens: true))
                let serialized = TimeSeriesTextSerializer.serialize(series)
                let ids = tokenizer.encode(text: serialized, addSpecialTokens: false)
                timeSeriesPositions += ids.count
                let idArray = MLXArray(ids.map { Int32($0) }, [1, ids.count])
                segments.append(model.tokenEmbeddings(idArray)[0].asType(dtype))
            }
        }
        segments.append(textEmbedding(sample.postPrompt, addSpecialTokens: true))

        let sequence = concatenated(segments, axis: 0)
        let inputsEmbeds = sequence.reshaped([1, sequence.dim(0), sequence.dim(1)])
        let promptPositions = inputsEmbeds.dim(1)

        // Memory guard: don't attempt a prefill beyond the on-device budget. A jetsam
        // OOM termination can't be caught, so we must skip rather than catch. We return
        // here *before* `eval`, so the (lazy) embedding graph above is never
        // materialized for a skipped arm — no large allocation occurs.
        if promptPositions > maxPromptPositions {
            return ArmStats(
                encoding: encoding, seriesLength: seriesLength, promptPositions: promptPositions,
                timeSeriesPositions: timeSeriesPositions, generatedTokens: 0, prefillSeconds: 0,
                totalSeconds: 0, decodeTokensPerSecond: 0, peakMemoryBytes: 0, didOOM: true,
                outputText: "[skipped: \(promptPositions) positions exceed on-device budget \(maxPromptPositions) — would risk an OOM termination]")
        }
        eval(inputsEmbeds)

        // Stop tokens identical to OpenTSLMLLM (Llama-3.2 has multiple).
        var stopTokens: Set<Int> = []
        if let primary = tokenizer.eosTokenId { stopTokens.insert(primary) }
        for marker in ["<|eot_id|>", "<|end_of_text|>"] {
            let ids = tokenizer.encode(text: marker, addSpecialTokens: false)
            if ids.count == 1 { stopTokens.insert(ids[0]) }
        }

        let generator = MLXEmbeddingGenerator(
            model: model,
            eosTokenIds: stopTokens,
            tokenSampler: { logits in argMax(logits, axis: -1).item(Int.self) },
            decodeTokens: { ids in tokenizer.decode(tokens: ids, skipSpecialTokens: true) })

        GPU.resetPeakMemory()
        do {
            let measured = try generator.generateMeasured(inputsEmbeds: inputsEmbeds, maxNewTokens: maxTokens)
            let peak = GPU.peakMemory
            let decodeSeconds = max(measured.totalSeconds - measured.prefillSeconds, 1e-9)
            let tps = measured.tokenIds.count > 1
                ? Double(measured.tokenIds.count - 1) / decodeSeconds
                : 0
            return ArmStats(
                encoding: encoding, seriesLength: seriesLength, promptPositions: promptPositions,
                timeSeriesPositions: timeSeriesPositions, generatedTokens: measured.tokenIds.count,
                prefillSeconds: measured.prefillSeconds, totalSeconds: measured.totalSeconds,
                decodeTokensPerSecond: tps, peakMemoryBytes: peak, didOOM: false,
                outputText: measured.text)
        } catch {
            logger.error("benchmark arm \(encoding, privacy: .public) failed at \(promptPositions, privacy: .public) positions: \(error.localizedDescription, privacy: .public)")
            return ArmStats(
                encoding: encoding, seriesLength: seriesLength, promptPositions: promptPositions,
                timeSeriesPositions: timeSeriesPositions, generatedTokens: 0, prefillSeconds: 0,
                totalSeconds: 0, decodeTokensPerSecond: 0, peakMemoryBytes: GPU.peakMemory,
                didOOM: true, outputText: "[generation failed: \(error.localizedDescription)]")
        }
    }

    // MARK: - Helpers

    private static func truncate(_ sample: OpenTSLMSPSample, to length: Int) -> OpenTSLMSPSample {
        OpenTSLMSPSample(
            prePrompt: sample.prePrompt,
            timeSeriesText: sample.timeSeriesText,
            timeSeries: sample.timeSeries.map { Array($0.prefix(length)) },
            postPrompt: sample.postPrompt,
            label: sample.label,
            answer: sample.answer)
    }

    /// Extrapolate the max raw series length that fits `contextWindow` under each
    /// encoding, using the longest measured length's scaffold + marginal rate.
    private static func contextCeiling(rows: [LengthComparison], contextWindow: Int) -> (text: Int, softPrompt: Int) {
        guard let longest = rows.max(by: { $0.seriesLength < $1.seriesLength }), longest.seriesLength > 0 else {
            return (0, 0)
        }
        let length = Double(longest.seriesLength)
        func maxLength(_ arm: ArmStats) -> Int {
            let scaffold = Double(arm.promptPositions - arm.timeSeriesPositions)
            let ratePerSample = Double(arm.timeSeriesPositions) / length
            guard ratePerSample > 0 else { return 0 }
            return max(0, Int((Double(contextWindow) - scaffold) / ratePerSample))
        }
        return (maxLength(longest.textSerialized), maxLength(longest.softPrompt))
    }
}

extension OpenTSLMBenchmark.Report {
    /// Human-readable summary table for logs / debug views.
    public func formattedSummary(split: String, sampleIndex: Int) -> String {
        var lines: [String] = []
        lines.append("OpenTSLM context-economics benchmark (split=\(split), sample=\(sampleIndex))")
        lines.append("backbone shared; greedy decode; LoRA applied to both arms: \(loraApplied ? "yes" : "no")")
        lines.append("")
        lines.append("  L | text_pos  sp_pos | text_ts  sp_ts  compress | text_tot(s) sp_tot(s) | text_tps sp_tps | text_peakMB sp_peakMB")
        for row in rows {
            let t = row.textSerialized
            let s = row.softPrompt
            func mb(_ bytes: Int) -> String { String(format: "%.0f", Double(bytes) / 1_048_576) }
            let numeric = String(
                format: "%5d | %8d %7d | %7d %6d %8.1fx | %10.2f %9.2f | %8.1f %7.1f",
                row.seriesLength, t.promptPositions, s.promptPositions,
                t.timeSeriesPositions, s.timeSeriesPositions, row.positionCompression,
                t.totalSeconds, s.totalSeconds, t.decodeTokensPerSecond, s.decodeTokensPerSecond)
            lines.append("\(numeric) | \(mb(t.peakMemoryBytes)) \(mb(s.peakMemoryBytes))")
            if t.didOOM { lines.append("        text arm: \(t.outputText)") }
            if s.didOOM { lines.append("        soft-prompt arm: \(s.outputText)") }
        }
        lines.append("")
        lines.append("Context ceiling @ \(contextWindow) positions: "
            + "text fits ~\(textMaxSeriesLength) samples, soft-prompt fits ~\(softPromptMaxSeriesLength) samples"
            + (textMaxSeriesLength > 0 ? String(format: " (%.1fx longer)", Double(softPromptMaxSeriesLength) / Double(textMaxSeriesLength)) : ""))
        return lines.joined(separator: "\n")
    }
}
