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
