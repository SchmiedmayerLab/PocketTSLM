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

extension TSLMCheckpoint {
    /// Builds what the model sees for `input`: the filled pre- and post-prompts, one description per
    /// series, and the normalised series cut to the checkpoint's `maxLength`.
    public func prompt(for input: TSLMInput) throws -> TSLMPrompt {
        let prompt = configuration.prompt
        let series = configuration.series

        var values = input.context
        var postTemplate = prompt.post
        if input.answerOptions.isEmpty, let postWithoutOptions = prompt.postWithoutOptions {
            postTemplate = postWithoutOptions
        } else {
            values["answerOptions"] = input.answerOptions.joined(separator: prompt.answerOptionSeparator ?? ", ")
        }

        let descriptions = try input.series.enumerated().map { index, timeSeries in
            if let description = timeSeries.description {
                return description
            }
            var fields = [
                "mean": String(format: series.statFormat, timeSeries.mean),
                "std": String(format: series.statFormat, timeSeries.std)
            ]
            if let name = timeSeries.name {
                fields["name"] = name
            } else if let names = series.names, index < names.count {
                fields["name"] = names[index]
            }
            return try Self.fill(series.description, with: fields)
        }

        return TSLMPrompt(
            pre: try Self.fill(prompt.pre, with: values),
            seriesDescriptions: descriptions,
            series: input.series.map { Array($0.values.prefix(series.maxLength)) },
            post: try Self.fill(postTemplate, with: values)
        )
    }

    /// Replaces each `{name}` in `template` with `values[name]` in one pass, so values are never re-expanded.
    static func fill(_ template: String, with values: [String: String]) throws -> String {
        let placeholder = try NSRegularExpression(pattern: #"\{([A-Za-z][A-Za-z0-9]*)\}"#)
        var result = ""
        var rest = template.startIndex
        for match in placeholder.matches(in: template, range: NSRange(template.startIndex..., in: template)) {
            guard let whole = Range(match.range, in: template), let key = Range(match.range(at: 1), in: template) else {
                continue
            }
            guard let value = values[String(template[key])] else {
                throw TSLMError.missingTemplateValue(String(template[key]))
            }
            result += template[rest ..< whole.lowerBound] + value
            rest = whole.upperBound
        }
        return result + template[rest...]
    }
}

/// What the model sees for one input: the text before the series, a description in front of each
/// series, the normalised series, and the text after them.
public struct TSLMPrompt: Sendable {
    public var pre: String
    public var seriesDescriptions: [String]
    public var series: [[Float]]
    public var post: String

    /// The texts are trimmed like OpenTSLM's `QADataset._format_sample` does before tokenising:
    /// the checkpoints were trained on stripped text, and a stray newline or trailing space derails
    /// greedy decoding (Sleep-EDF: 45% instead of 85%).
    public init(pre: String, seriesDescriptions: [String], series: [[Float]], post: String) {
        self.pre = pre.trimmingCharacters(in: .whitespacesAndNewlines)
        self.seriesDescriptions = seriesDescriptions
        self.series = series
        self.post = post.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// The prompt as text, with each series shown by its description only.
    public var text: String {
        ([pre] + seriesDescriptions + [post]).joined(separator: "\n")
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
    /// A prompt has more positions than the `maxPromptPositions` of a generation.
    case promptTooLong(positions: Int, seriesPositions: Int, limit: Int)

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
        case let .promptTooLong(positions, _, limit):
            "The prompt has \(positions) positions, more than the limit of \(limit)."
        }
    }
}
