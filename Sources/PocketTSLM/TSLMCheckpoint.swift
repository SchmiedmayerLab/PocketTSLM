//
// This source file is part of the PocketTSLM open-source project
//
// SPDX-FileCopyrightText: 2026 Schmiedmayer Lab and the project authors (see CONTRIBUTORS.md)
//
// SPDX-License-Identifier: MIT
//

import Foundation

/// A converted OpenTSLM-SP checkpoint: a folder with `config.json` and the encoder, projector and
/// LoRA weights, as `tools/convert_checkpoint.py` writes it.
public struct TSLMCheckpoint: Sendable {
    /// The `config.json` of a checkpoint folder, format `pockettslm-checkpoint/1`.
    public struct Configuration: Codable, Sendable {
        public struct BaseModel: Codable, Sendable {
            /// Hugging Face repository of the base model, e.g. `meta-llama/Llama-3.2-1B`.
            public let id: String
            public let revision: String?
            public let weightsSHA256: String?
            public let hiddenSize: Int
        }

        public struct Encoder: Codable, Sendable {
            public let patchSize: Int
            public let embedDim: Int
            public let maxPatches: Int
        }

        public struct LoRA: Codable, Sendable {
            public let rank: Int
            public let alpha: Float
            /// The projections that carry adapters, e.g. `q_proj`.
            public let keys: [String]
        }

        public struct Series: Codable, Sendable {
            /// Always `z-score`: each series is normalised by its own mean and standard deviation.
            public let normalization: String
            /// Lower bound for the standard deviation used to normalise a series.
            public let minStd: Float
            /// Samples per series the model was trained on; longer series are cut to this length.
            public let maxLength: Int
            public let samplingRate: Double?
            /// Default series names, in order, for templates that use `{name}`.
            public let names: [String]?
            /// Template with `{name}`, `{mean}` and `{std}` that introduces each series.
            public let description: String
            /// `printf` format for the mean and standard deviation, e.g. `%.4f`.
            public let statFormat: String
        }

        public struct Prompt: Codable, Sendable {
            /// Text before the series. `{placeholder}` fields are filled from the input's context.
            public let pre: String
            /// Text after the series; `{answerOptions}` receives the joined answer options.
            public let post: String
            /// Replaces `post` when an input has no answer options.
            public let postWithoutOptions: String?
            public let answerOptionSeparator: String?
            /// Marker in front of the final answer, e.g. `Answer:`.
            public let answerCue: String
        }

        public struct Generation: Codable, Sendable {
            /// Default maximum number of generated tokens.
            public let maxTokens: Int
        }

        public let format: String
        public let task: String
        public let baseModel: BaseModel
        public let encoder: Encoder
        public let lora: LoRA
        public let series: Series
        public let prompt: Prompt
        public let generation: Generation
    }

    /// The `format` value this version of the package reads.
    public static let format = "pockettslm-checkpoint/1"

    public let directory: URL
    public let configuration: Configuration

    public var encoderURL: URL { directory.appendingPathComponent("encoder.safetensors") }
    public var projectorURL: URL { directory.appendingPathComponent("projector.safetensors") }
    public var loraURL: URL { directory.appendingPathComponent("lora.safetensors") }

    /// Reads `config.json` from a checkpoint folder.
    public init(directory: URL) throws {
        let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let configuration = try JSONDecoder().decode(Configuration.self, from: data)
        guard configuration.format == Self.format else {
            throw TSLMError.unsupportedCheckpointFormat(configuration.format)
        }
        guard configuration.series.statFormat.range(of: #"^%\.[0-9]+f$"#, options: .regularExpression) != nil else {
            throw TSLMError.invalidCheckpoint("statFormat must look like %.4f, not \(configuration.series.statFormat)")
        }
        self.directory = directory
        self.configuration = configuration
    }
}

/// Errors that PocketTSLM throws.
public enum TSLMError: LocalizedError {
    case unsupportedCheckpointFormat(String)
    case invalidCheckpoint(String)
    /// A prompt template uses a `{placeholder}` the input does not provide.
    case missingTemplateValue(String)
    /// The loaded base model was not built by ``registerModelTypes()``.
    case unsupportedBaseModel(String)

    public var errorDescription: String? {
        switch self {
        case .unsupportedCheckpointFormat(let format):
            "Unsupported checkpoint format \(format); this version of PocketTSLM reads \(TSLMCheckpoint.format)."
        case .invalidCheckpoint(let reason):
            "Invalid checkpoint: \(reason)"
        case .missingTemplateValue(let name):
            "The prompt template needs a value for {\(name)}."
        case .unsupportedBaseModel(let type):
            "The base model is \(type), not EmbeddingLlamaModel; call registerModelTypes() before loading it."
        }
    }
}
