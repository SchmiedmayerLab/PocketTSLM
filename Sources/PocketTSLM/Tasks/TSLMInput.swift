//
// This source file is part of the PocketTSLM open-source project
//
// SPDX-FileCopyrightText: 2026 Schmiedmayer Lab and the project authors (see CONTRIBUTORS.md)
//
// SPDX-License-Identifier: MIT
//

import Foundation

/// One time series of a ``TSLMInput``.
public struct TimeSeries: Sendable {
    /// Fills `{name}` in the series description, e.g. an ECG lead. Defaults to the checkpoint's names.
    public var name: String?
    /// z-normalised values, which the encoder receives.
    public var values: [Float]
    /// Mean of the raw series, which the description reports.
    public var mean: Double
    /// Standard deviation of the raw series, which the description reports.
    public var std: Double
    /// Replaces the checkpoint's description template, e.g. with a description a dataset export computed.
    public var description: String?

    /// Normalises raw samples like OpenTSLM: z-score with the population standard deviation, floored
    /// at `minStd`. It computes in `Float`, exactly as the paper app's Sleep-EDF path does.
    public init(name: String? = nil, samples: [Float], minStd: Float = 1e-6) {
        let count = Float(max(samples.count, 1))
        let mean = samples.reduce(0, +) / count
        let variance = samples.reduce(0) { partial, value in
            let d = value - mean
            return partial + d * d
        } / count
        let std = max(sqrt(variance), minStd)
        self.init(name: name, values: samples.map { ($0 - mean) / std }, mean: Double(mean), std: Double(std))
    }

    /// A series that was normalised elsewhere.
    public init(name: String? = nil, values: [Float], mean: Double, std: Double, description: String? = nil) {
        self.name = name
        self.values = values
        self.mean = mean
        self.std = std
        self.description = description
    }
}

/// The input of a time-series language model: series plus the values its prompt templates need.
public struct TSLMInput: Sendable {
    public var series: [TimeSeries]
    /// Values for the `{placeholder}` fields of the prompts, e.g. `question` and `clinicalContext`.
    public var context: [String: String]
    /// Answer options the post-prompt lists; leave empty for an open question.
    public var answerOptions: [String]

    public init(series: [TimeSeries], context: [String: String] = [:], answerOptions: [String] = []) {
        self.series = series
        self.context = context
        self.answerOptions = answerOptions
    }
}

extension TSLMCheckpoint {
    /// Builds what the model sees: the filled pre- and post-prompts, one description per series, and
    /// the normalised series cut to the checkpoint's `maxLength`. `label` and `answer` are carried
    /// along for evaluations.
    public func makeSample(_ input: TSLMInput, label: String = "", answer: String = "") throws -> OpenTSLMSPSample {
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

        return OpenTSLMSPSample(
            prePrompt: try Self.fill(prompt.pre, with: values),
            timeSeriesText: descriptions,
            timeSeries: input.series.map { Array($0.values.prefix(series.maxLength)) },
            postPrompt: try Self.fill(postTemplate, with: values),
            label: label,
            answer: answer
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
