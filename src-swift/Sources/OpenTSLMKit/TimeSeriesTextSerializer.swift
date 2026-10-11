//
// This source file is part of the PocketTSLM open-source project
//
// SPDX-FileCopyrightText: 2026 Schmiedmayer Lab and the project authors (see CONTRIBUTORS.md)
//
// SPDX-License-Identifier: MIT
//

import Foundation

/// Serializes a time series into the exact text format OpenTSLM uses for its
/// text-serialized (prompt-level fusion) baseline.
///
/// This is the §4 "context economics" counterpart to soft prompts: instead of
/// encoding samples into projected embeddings, each value is rendered as text and
/// fed to the same backbone as ordinary tokens. The format mirrors OpenTSLM's
/// `QADataset` fallback formatter byte-for-byte:
///
/// ```python
/// np.array2string(ts, separator=" ",
///     formatter={"all": lambda x: f'"{x:.2f}"'.replace(".", "")})
/// ```
///
/// i.e. round to two decimals, drop the decimal point, wrap in double quotes,
/// space-separate. Examples (verified against the Python reference):
/// `0.12 -> "012"`, `-1.453 -> "-145"`, `2.3 -> "230"`, `0.0 -> "000"`,
/// `-0.05 -> "-005"`, `12.7 -> "1270"`.
public enum TimeSeriesTextSerializer {

    /// Serialize one value the way OpenTSLM's formatter does.
    public static func format(_ value: Float) -> String {
        // `%.2f` rounds half-to-even like Python's `:.2f`; ties are vanishingly
        // unlikely on z-normalized float data, so the two agree in practice.
        let digits = String(format: "%.2f", value).replacingOccurrences(of: ".", with: "")
        return "\"\(digits)\""
    }

    /// Serialize a 1D series to a single space-separated string.
    public static func serialize(_ series: [Float]) -> String {
        series.map(format).joined(separator: " ")
    }
}
