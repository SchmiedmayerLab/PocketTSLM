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
import os

/// Loads OpenTSLM LoRA adapters into an ``EmbeddingLlamaModel`` (MLX ``LoRALinear``).
///
/// Mirrors [OpenTSLMMLX](https://github.com/SchmiedmayerLab/OpenTSLMMLX) ``_apply_lora``:
/// ``linear_to_lora_layers`` on all transformer projections, rank 16, alpha 32 (scale 2), PEFT transpose.
enum OpenTSLMLoRA {

    private static let logger = Logger(subsystem: "PocketTSLM", category: "OpenTSLMLoRA")

    /// The adapter each base model carries and its shape. The layers are converted to `LoRALinear`
    /// once, so switching tasks (ECG ↔ sleep) only swaps the adapter weights.
    private static let applied = OSAllocatedUnfairLock(initialState: [ObjectIdentifier: (checkpoint: URL, shape: Shape)]())

    /// Rank, scale and target projections of the adapters.
    struct Shape: Equatable, Sendable {
        var rank: Int
        var scale: Float
        var keys: Set<String>
    }

    /// Makes `checkpointURL` the active adapter of `llama`. Call it inside the base model's
    /// `ModelContainer.perform`, which serialises the calls for one model.
    static func applyIfNeeded(to llama: EmbeddingLlamaModel, checkpointURL: URL, shape: Shape) throws {
        let modelID = ObjectIdentifier(llama)
        let current = applied.withLock { $0[modelID] }
        if current?.checkpoint == checkpointURL {
            return
        }
        if let current, current.shape != shape {
            throw TSLMError.invalidCheckpoint(
                "its LoRA adapter (rank \(shape.rank)) differs from the one already on the base model (rank \(current.shape.rank))")
        }
        if current == nil {
            convertOpenTSLMLoRALayers(on: llama, shape: shape)
        }

        let rawWeights = try loadArrays(url: checkpointURL)
        let sanitized = sanitizeCheckpointKeys(rawWeights)

        let trainableKeys = Set(llama.trainableParameters().flattened().map(\.0))
        let filtered = sanitized.filter { trainableKeys.contains($0.key) }
        let parameters = ModuleParameters.unflattened(filtered)
        try llama.update(parameters: parameters, verify: .noUnusedKeys)
        eval(llama)

        applied.withLock { $0[modelID] = (checkpointURL, shape) }
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
