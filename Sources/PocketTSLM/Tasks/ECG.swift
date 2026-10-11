//
// This source file is part of the PocketTSLM open-source project
//
// SPDX-FileCopyrightText: 2026 Schmiedmayer Lab and the project authors (see CONTRIBUTORS.md)
//
// SPDX-License-Identifier: MIT
//

import Foundation

/// Prepares single-lead ECG recordings, such as Apple Watch's, for the ECG-QA checkpoint.
public enum ECG {
    /// The 12-lead order of PTB-XL and ECG-QA.
    public static let leadNames = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

    /// Builds the 12-lead input the ECG-QA checkpoint was trained on from one lead.
    ///
    /// The recording is resampled to `targetRate` by nearest neighbour from t = 0 and cut to
    /// `length` samples, padded with zeros when it is shorter. Its statistics and z-normalisation
    /// are computed in `Double`, and the lead is copied to all 12 leads, so every description shows
    /// the same mean and standard deviation, in the recording's unit (Apple Watch: volts).
    public static func twelveLeads(
        fromLeadI voltages: [Double],
        samplingRate: Double,
        targetRate: Double = 100,
        length: Int = 1000
    ) -> [TimeSeries] {
        let window = resample(voltages, from: samplingRate, to: targetRate, count: length)
        let (mean, std) = statistics(of: window)
        let values = window.map { Float(($0 - mean) / std) }
        return leadNames.map { TimeSeries(name: $0, values: values, mean: mean, std: std) }
    }

    static func resample(_ values: [Double], from inputRate: Double, to outputRate: Double, count: Int) -> [Double] {
        guard !values.isEmpty, inputRate > 0, outputRate > 0, count > 0 else {
            return Array(repeating: 0, count: max(count, 0))
        }
        let step = inputRate / outputRate
        return (0 ..< count).map { index in
            let source = Int(Double(index) * step)
            return source < values.count ? values[source] : 0
        }
    }

    static func statistics(of values: [Double]) -> (mean: Double, std: Double) {
        guard !values.isEmpty else {
            return (0, 1)
        }
        let mean = values.reduce(0, +) / Double(values.count)
        let variance = values.reduce(0) { partial, value in
            let d = value - mean
            return partial + d * d
        } / Double(values.count)
        return (mean, max(sqrt(variance), 1e-6))
    }
}
