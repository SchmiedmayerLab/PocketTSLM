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
import PocketTSLM

let usage = """
    Usage: pockettslm-cli eval --base <dir> --checkpoint <dir> --data <dir> [options]

    Runs a checkpoint's accuracy evaluation as the demo app does and writes the same report JSON.

      --base <dir>        Llama-3.2-1B folder with config.json, the tokenizer files and model.safetensors
      --checkpoint <dir>  PocketTSLM checkpoint folder, e.g. Tests/UITests/Checkpoints/llama-3.2-1b-sleep-sp-mlx
      --data <dir>        Evaluation data, e.g. Tests/UITests/EvalData (scripts/fetch_assets.sh fills both)
      --count <n>         Rows to evaluate (default: 20)
      --max-tokens <n>    New tokens per answer (default: the checkpoint's)
      --output <file>     Where to write the report (default: standard output)
      --cpu               Run on the CPU instead of the GPU
    """

struct UsageError: Error, CustomStringConvertible {
    let description: String
}

struct EvalOptions {
    var base: URL?
    var checkpoint: URL?
    var data: URL?
    var count = 20
    var maxTokens: Int?
    var output: URL?
    var cpu = false

    init(_ arguments: ArraySlice<String>) throws {
        var arguments = arguments
        func value(of option: String) throws -> String {
            guard let value = arguments.popFirst() else {
                throw UsageError(description: "\(option) needs a value")
            }
            return value
        }
        func number(of option: String) throws -> Int {
            guard let number = Int(try value(of: option)), number > 0 else {
                throw UsageError(description: "\(option) needs a positive number")
            }
            return number
        }
        while let option = arguments.popFirst() {
            switch option {
            case "--base": base = URL(fileURLWithPath: try value(of: option))
            case "--checkpoint": checkpoint = URL(fileURLWithPath: try value(of: option))
            case "--data": data = URL(fileURLWithPath: try value(of: option))
            case "--count": count = try number(of: option)
            case "--max-tokens": maxTokens = try number(of: option)
            case "--output": output = URL(fileURLWithPath: try value(of: option))
            case "--cpu": cpu = true
            default: throw UsageError(description: "unknown option \(option)")
            }
        }
    }
}

/// The rows the demo app evaluates for `task`, from the evaluation data folder.
func evaluationSamples(task: String, data: URL, count: Int) throws -> [EvaluationSample] {
    switch task {
    case "sleep-edf":
        return try SleepEDFDataset.evaluationSamples(csvURL: data.appendingPathComponent("sleep_cot_test20.csv"), count: count)
    case "ecg-qa":
        let dataset = try ECGQACoTDataset(
            csvURL: data.appendingPathComponent("ecg_qa_cot_test.csv"),
            waveformsDirectory: data.appendingPathComponent("ecg_qa_waveforms"),
            templateAnswersURL: data.appendingPathComponent("ecg_qa_template_answers.json")
        )
        return try (0 ..< min(count, dataset.count)).map(dataset.evaluationSample(at:))
    default:
        throw UsageError(description: "no evaluation data for task \(task)")
    }
}

func printError(_ text: String) {
    FileHandle.standardError.write(Data((text + "\n").utf8))
}

func evaluate(_ options: EvalOptions) async throws {
    guard let base = options.base, let checkpointURL = options.checkpoint, let data = options.data else {
        throw UsageError(description: "--base, --checkpoint and --data are required")
    }
    let checkpoint = try TSLMCheckpoint(directory: checkpointURL)
    let samples = try evaluationSamples(task: checkpoint.configuration.task, data: data, count: options.count)

    registerModelTypes()
    let report = try await Device.withDefaultDevice(options.cpu ? .cpu : .gpu) {
        let container = try await LLMModelFactory.shared.loadContainer(configuration: ModelConfiguration(directory: base))
        let model = try TSLM(base: container, checkpoint: checkpoint)
        return try await Evaluation.run(model, on: samples, maxTokens: options.maxTokens) { row in
            let verdict = row.correct ? "OK" : "X"
            printError("[\(row.index + 1)/\(samples.count)] gt=\(row.groundTruth ?? "nil") pred=\(row.predicted ?? "nil") \(verdict) \(row.seconds)s")
        }
    }
    printError("accuracy: \(report.correct)/\(report.rows.count)   parsed: \(report.parsed)/\(report.rows.count)")

    let json = try report.json()
    if let output = options.output {
        try json.write(to: output)
    } else {
        FileHandle.standardOutput.write(json + Data("\n".utf8))
    }
}

let arguments = CommandLine.arguments.dropFirst()
if arguments.contains("--help") || arguments.contains("-h") {
    print(usage)
    exit(0)
}
do {
    guard arguments.first == "eval" else {
        throw UsageError(description: arguments.first.map { "unknown command \($0)" } ?? "missing command")
    }
    try await evaluate(EvalOptions(arguments.dropFirst()))
} catch let error as UsageError {
    printError("error: \(error)\n\n\(usage)")
    exit(2)
} catch {
    printError("error: \(error.localizedDescription)")
    exit(1)
}
