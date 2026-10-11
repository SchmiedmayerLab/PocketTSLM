#!/usr/bin/env bash
#
# This source file is part of the PocketTSLM open-source project
#
# SPDX-FileCopyrightText: 2026 Schmiedmayer Lab and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#

# Captures the model inputs the paper app builds, as a reference for the package's prompt
# assembly: it runs the snapshot commit's own Sleep-EDF reader, ECG-QA reader and Apple Watch ECG
# path on deterministic synthetic data and writes Tests/PocketTSLMTests/Fixtures/app_samples.json.
#
# Usage: scripts/capture_app_samples.sh

set -euo pipefail

snapshot=14c4426
repo_root="$(cd "$(dirname "$0")/.." && pwd)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
app=PocketTSLM/PocketTSLM/OpenTSLM

git -C "$repo_root" show "$snapshot:$app/SleepEDFDataset.swift" > "$work/SleepEDFDataset.swift"
git -C "$repo_root" show "$snapshot:$app/ECGQACoTDataset.swift" > "$work/ECGQACoTDataset.swift"
# OpenTSLMSPSample is the part of the pipeline file that needs no MLX.
git -C "$repo_root" show "$snapshot:$app/OpenTSLMSPPipeline.swift" | sed -n '1p;4,34p' > "$work/OpenTSLMSPSample.swift"
# The Watch ECG path and the series cap are private members of the app's inference service.
{
    echo 'import Foundation'
    echo 'enum Constants { static let openTSLMMaxTimeSeriesLength = 1000 }'
    echo 'struct WatchPath {'
    git -C "$repo_root" show "$snapshot:$app/OpenTSLMInferenceService.swift" | sed -n '1177,1194p;1500,1545p;1557,1586p'
    echo '}'
    git -C "$repo_root" show "$snapshot:$app/OpenTSLMInferenceService.swift" | sed -n '1638,1734p'
} | sed -e 's/private func /func /' -e 's/private static /static /' -e 's/^private enum /enum /' -e 's/^private struct /struct /' \
    > "$work/WatchPath.swift"

cat > "$work/main.swift" <<'EOF'
import CryptoKit
import Foundation

/// The same generator as `AppSampleTests.Synthetic`, so the tests can rebuild every input.
struct Synthetic {
    var state: UInt64

    mutating func next() -> UInt64 {
        state = 6364136223846793005 &* state &+ 1442695040888963407
        return state
    }

    /// Uniform in [-1, 1), from the top 24 bits.
    mutating func unit() -> Float {
        Float(next() >> 40) / Float(1 << 23) - 1
    }

    static func series(count: Int, seed: UInt64, scale: Float, offset: Float) -> [Float] {
        var generator = Synthetic(state: seed)
        return (0 ..< count).map { _ in offset + scale * generator.unit() }
    }
}

struct Captured: Encodable {
    let name: String
    let prePrompt: String
    let timeSeriesText: [String]
    let postPrompt: String
    let label: String
    let answer: String
    let seriesLengths: [Int]
    let seriesSHA256: [String]

    init(_ name: String, _ sample: OpenTSLMSPSample) {
        self.name = name
        prePrompt = sample.prePrompt
        timeSeriesText = sample.timeSeriesText
        postPrompt = sample.postPrompt
        label = sample.label
        answer = sample.answer
        seriesLengths = sample.timeSeries.map(\.count)
        seriesSHA256 = sample.timeSeries.map { values in
            let bytes = values.flatMap { withUnsafeBytes(of: $0.bitPattern.littleEndian, Array.init) }
            return SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined()
        }
    }
}

/// Writes a value so that parsing it as Double and converting to Float gives it back exactly.
func number(_ value: Float) -> String {
    String(format: "%.17g", Double(value))
}

let work = URL(fileURLWithPath: CommandLine.arguments[1])
let watch = WatchPath()
var cases: [Captured] = []

// Sleep-EDF: the second row is longer than the 1500-sample cap of the app's sleep eval.
let sleepRows: [(count: Int, seed: UInt64, scale: Float, offset: Float, label: String, rationale: String)] = [
    (1500, 1, 2e-5, -1e-6, "W", "The signal shows fast, low-amplitude activity. Answer: Wake"),
    (1600, 2, 5e-5, 3e-6, "N2", "Spindles and K-complexes appear throughout. Answer: Non-REM stage 2")
]
var csv = "id,time_series,label,prompt,rationale\n"
for (index, row) in sleepRows.enumerated() {
    let series = Synthetic.series(count: row.count, seed: row.seed, scale: row.scale, offset: row.offset)
    csv += "row\(index),\"[[\(series.map(number).joined(separator: ", "))]]\",\(row.label),,\"\(row.rationale)\"\n"
}
let sleepCSV = work.appendingPathComponent("sleep.csv")
try csv.write(to: sleepCSV, atomically: true, encoding: .utf8)
for index in sleepRows.indices {
    let sample = watch.cappedSample(try SleepEDFDataset.rawSample(csvURL: sleepCSV, rowIndex: index), limit: 1500)
    cases.append(Captured("sleep-\(index)", sample))
}

// ECG-QA: the second row's template has no answer options and the row has no clinical context.
let leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
let waveforms = work.appendingPathComponent("waveforms")
try FileManager.default.createDirectory(at: waveforms, withIntermediateDirectories: true)
for ecgID in [101, 202] {
    let series = leads.indices.map { Synthetic.series(count: 1000, seed: UInt64(ecgID * 100 + $0), scale: 1.5, offset: 0) }
    let text = leads.indices.map { lead in
        "This is ECG Lead \(leads[lead]), it has mean \(String(format: "%.4f", Double(lead) * 0.0123 - 0.05)) and std \(String(format: "%.4f", 0.1 + Double(lead) * 0.01)):"
    }
    let json = "{\"time_series\": [\(series.map { "[\($0.map(number).joined(separator: ", "))]" }.joined(separator: ", "))], "
        + "\"time_series_text\": [\(text.map { "\"\($0)\"" }.joined(separator: ", "))]}"
    try json.write(to: waveforms.appendingPathComponent("\(ecgID).json"), atomically: true, encoding: .utf8)
}
let ecgCSV = work.appendingPathComponent("ecg.csv")
try """
question,answer,template_id,question_type,ecg_id,rationale,clinical_context
Does this ECG show sinus rhythm?,yes,5,single-verify,[101],"Regular rhythm with upright P waves. Answer: yes",A 63-year-old patient with palpitations.
What is the heart rhythm?,sinus rhythm,9,single-query,202,"Normal P waves precede each QRS complex. Answer: sinus rhythm",

""".write(to: ecgCSV, atomically: true, encoding: .utf8)
let templates = work.appendingPathComponent("templates.json")
try #"{"template_answers": {"5": ["yes", "no", "not sure"]}}"#.write(to: templates, atomically: true, encoding: .utf8)
let ecgQA = try ECGQACoTDataset(csvURL: ecgCSV, waveformsDirectory: waveforms, templateAnswersURL: templates)
for index in 0 ..< ecgQA.count {
    cases.append(Captured("ecg-qa-\(index)", watch.cappedSample(try ecgQA.sample(at: index))))
}

// Apple Watch ECG at 512 Hz: 30 s, and 4 s, which the app pads with zeros to 10 s.
for (index, (count, seed)) in [(15360, UInt64(20)), (2048, UInt64(21))].enumerated() {
    let voltages = Synthetic.series(count: count, seed: seed, scale: 8e-4, offset: 1e-4).map(Double.init)
    let ecg = ECGSample(
        source: .healthkit, samplingFrequency: 512, classification: "sinus rhythm",
        symptomsStatus: "none", averageHeartRate: 72, voltages: voltages
    )
    cases.append(Captured("watch-\(index)", watch.cappedSample(watch.makeOpenTSLMSample(from: ecg))))
}

let encoder = JSONEncoder()
encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
FileHandle.standardOutput.write(try encoder.encode(cases))
FileHandle.standardOutput.write(Data("\n".utf8))
EOF

xcrun swiftc -O -o "$work/capture" "$work"/*.swift
"$work/capture" "$work" > "$repo_root/Tests/PocketTSLMTests/Fixtures/app_samples.json"
echo "Wrote Tests/PocketTSLMTests/Fixtures/app_samples.json"
