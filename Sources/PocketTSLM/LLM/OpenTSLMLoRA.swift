//
// This source file is part of the PocketTSLM open-source project
//
// SPDX-FileCopyrightText: 2026 Schmiedmayer Lab and the project authors (see CONTRIBUTORS.md)
//
// SPDX-License-Identifier: MIT
//

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import OSLog

/// Loads OpenTSLM LoRA adapters into an ``EmbeddingLlamaModel`` (MLX ``LoRALinear``).
///
/// Mirrors [OpenTSLMMLX](https://github.com/SchmiedmayerLab/OpenTSLMMLX) ``_apply_lora``:
/// ``linear_to_lora_layers`` on all transformer projections, rank 16, alpha 32 (scale 2), PEFT transpose.
public enum OpenTSLMLoRA {

    private static let logger = Logger(subsystem: "PocketTSLM", category: "OpenTSLMLoRA")

    /// Tracks which checkpoint URL is currently loaded into each model instance.
    /// Allows task switching (ECG ↔ sleep) by re-loading adapter weights without
    /// repeating the one-time Linear→LoRALinear layer conversion.
    private static var appliedModelCheckpoints: [ObjectIdentifier: URL] = [:]

    /// Rank, scale and target projections of the adapters. The defaults are OpenTSLM-SP's:
    /// rank 16, alpha 32 (scale 2), all seven projections.
    public struct Shape: Sendable {
        public var rank: Int
        public var scale: Float
        public var keys: Set<String>

        public init(
            rank: Int = 16,
            scale: Float = 2,
            keys: Set<String> = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        ) {
            self.rank = rank
            self.scale = scale
            self.keys = keys
        }
    }

    /// Makes `checkpointURL` the active adapter of the container's ``EmbeddingLlamaModel``.
    ///
    /// The first call converts the projections to `LoRALinear` once; a later call with another
    /// checkpoint only swaps the adapter weights, so one loaded base model serves several tasks.
    public static func applyIfNeeded(
        to container: ModelContainer,
        checkpointURL: URL,
        shape: Shape = Shape()
    ) async throws {
        try await container.perform { context in
            guard let llama = context.model as? EmbeddingLlamaModel else {
                return
            }
            try applyIfNeeded(to: llama, checkpointURL: checkpointURL, shape: shape)
        }
    }

    static func applyIfNeeded(to llama: EmbeddingLlamaModel, checkpointURL: URL, shape: Shape = Shape()) throws {
        let modelID = ObjectIdentifier(llama)

        if appliedModelCheckpoints[modelID] == checkpointURL {
            return
        }

        if appliedModelCheckpoints[modelID] == nil {
            convertOpenTSLMLoRALayers(on: llama, shape: shape)
        }

        let rawWeights = try loadArrays(url: checkpointURL)
        let sanitized = sanitizeCheckpointKeys(rawWeights)

        let trainableKeys = Set(llama.trainableParameters().flattened().map(\.0))
        let filtered = sanitized.filter { trainableKeys.contains($0.key) }
        let parameters = ModuleParameters.unflattened(filtered)
        try llama.update(parameters: parameters, verify: .noUnusedKeys)
        eval(llama)

        appliedModelCheckpoints[modelID] = checkpointURL

        logger.info("OpenTSLM LoRA applied from \(checkpointURL.lastPathComponent, privacy: .public)")
    }

    /// Convert every attention/MLP linear in all transformer layers (matches ``linear_to_lora_layers`` in OpenTSLMMLX).
    private static func convertOpenTSLMLoRALayers(on llama: EmbeddingLlamaModel, shape: Shape) {
        llama.freeze()
        let layers = openTSMLLoRALinearLayers(on: llama, keys: shape.keys)
        for (layer, keys) in layers {
            var update = ModuleChildren()
            let children = layer.children()
            for key in keys {
                guard let item = children[key], case .value(let child) = item, let linear = child as? Linear else {
                    continue
                }
                let (outputDimensions, inputDimensions) = linear.shape
                update[key] = .value(
                    LoRALinear(
                        inputDimensions,
                        outputDimensions,
                        rank: shape.rank,
                        scale: shape.scale,
                        linear: linear
                    )
                )
            }
            if !update.isEmpty {
                layer.update(modules: update)
            }
        }
    }

    private static func openTSMLLoRALinearLayers(on llama: EmbeddingLlamaModel, keys: Set<String>) -> LoRALinearLayers {
        var groups: [String: [String]] = [:]
        for (path, module) in llama.namedModules() {
            guard module is Linear, !path.contains("lora_") else {
                continue
            }
            guard path.hasPrefix("model.layers."),
                  let suffix = path.split(separator: ".").last.map(String.init),
                  keys.contains(suffix)
            else {
                continue
            }
            let parentPath = path.split(separator: ".").dropLast().joined(separator: ".")
            groups[parentPath, default: []].append(suffix)
        }

        let named = Dictionary(uniqueKeysWithValues: llama.namedModules())
        return groups.keys.sorted().compactMap { parentPath in
            guard let parent = named[parentPath], let keys = groups[parentPath] else {
                return nil
            }
            return (parent, keys.sorted())
        }
    }

    /// Remap PEFT keys to MLX ``LoRALinear`` names (with transpose). MLX-native keys pass through unchanged.
    static func sanitizeCheckpointKeys(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var result: [String: MLXArray] = [:]

        for (key, value) in weights {
            if let mapped = mapPEFTKey(key, value: value) {
                result[mapped.key] = mapped.value
            } else if key.contains("lora_a") || key.contains("lora_b") {
                result[key] = value
            }
        }

        return result
    }

    private static func mapPEFTKey(_ key: String, value: MLXArray) -> (key: String, value: MLXArray)? {
        let pattern = #"^base_model\.model\.(.+)\.(lora_[AB])\.default\.weight$"#
        guard let regex = try? NSRegularExpression(pattern: pattern),
              let match = regex.firstMatch(in: key, range: NSRange(key.startIndex..., in: key)),
              match.numberOfRanges == 3,
              let layerRange = Range(match.range(at: 1), in: key),
              let abRange = Range(match.range(at: 2), in: key)
        else {
            return nil
        }

        let layerPath = String(key[layerRange])
        let ab = String(key[abRange]).lowercased()
        let mlxKey = layerPath.hasPrefix("model.") ? "\(layerPath).\(ab)" : "model.\(layerPath).\(ab)"
        return (mlxKey, value.transposed())
    }
}
