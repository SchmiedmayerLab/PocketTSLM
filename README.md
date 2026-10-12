<!--

This source file is part of the PocketTSLM open-source project

SPDX-FileCopyrightText: 2023 Schmiedmayer Lab and the project authors (see CONTRIBUTORS.md)

SPDX-License-Identifier: MIT

-->

# PocketTSLM

**PocketTSLM** runs a soft-prompt time-series language model (OpenTSLM-SP) fully on-device with MLX Swift: the encoder, the projector and the LLM all execute on an iPhone, so physiological time series (ECG, EEG) can be interpreted in natural language without sending health data off the device. This repository holds the PocketTSLM Swift package, a demo app with the paper's accuracy evaluations and benchmarks, a command-line eval runner for the Mac, and the Python↔Swift parity tests.

The repository contains code only. Model weights and evaluation data are restored with `scripts/fetch_assets.sh`.

## Repository layout

- `Package.swift` and `Sources/PocketTSLM/`: the Swift package, with checkpoint loading, prompt assembly, the encoder and projector, the embedding-primed Llama, generation, the dataset readers and the accuracy evaluation.
- `Sources/pockettslm-cli/`: `pockettslm-cli eval`, which runs the accuracy evaluations on a Mac.
- `Tests/PocketTSLMTests/`: unit and parity tests, their fixtures and the fixture generators.
- `Tests/UITests/`: the demo app (`UITests.xcodeproj`, target `TestApp`, shown as "PocketTSLM Demo"), with on-device inference, accuracy evaluations and benchmarks.
- `baselines/`: evaluation outputs that later changes must reproduce.
- `benchmark-results/`: reference benchmark outputs from our run.
- `scripts/`: asset restore and dataset export helpers.
- `tools/`: the checkpoint converter and its task presets.

## Requirements

- macOS with Xcode 26 and an Apple-silicon iPhone, for the latency and memory numbers. MLX needs a real Metal GPU; in the simulator the app falls back to the CPU, which is only good for diagnostics.
- Access to [`meta-llama/Llama-3.2-1B`](https://huggingface.co/meta-llama/Llama-3.2-1B), which is gated. The app downloads it on first launch, at the revision the checkpoints pin.

## Using the package

Register the patched Llama before the first model load, load the base model and a checkpoint, and answer an input:

```swift
import MLXLLM
import MLXLMCommon
import PocketTSLM

registerModelTypes()
let base = try await LLMModelFactory.shared.loadContainer(configuration: ModelConfiguration(directory: llamaFolder))
let model = try TSLM(base: base, checkpoint: TSLMCheckpoint(directory: checkpointFolder))
let answer = try await model.generate(TSLMInput(
    series: ECG.twelveLeads(fromLeadI: voltages, samplingRate: 512),
    context: [
        "clinicalContext": "Single-lead ambulatory ECG recording (Lead I equivalent, replicated across the standard 12 leads).",
        "question": "Is the cardiac rhythm shown in this ECG consistent with atrial fibrillation?"
    ],
    answerOptions: ["yes", "no"]
))
```

The checkpoint's `config.json` holds the task's prompt templates, whose placeholders `context` fills. A second checkpoint on the same base model swaps its LoRA adapter in. The package pins mlx-swift 0.21.3, mlx-swift-examples at revision `0c50f71e` and swift-transformers 0.1.20; because of the revision pin, apps add it by local path for now.

## Model files and data

The checkpoints are conversions of the OpenTSLM-SP releases on Hugging Face, `OpenTSLM/llama-3.2-1b-sleep-sp` and `OpenTSLM/llama-3.2-1b-ecg-sp`. `tools/convert_checkpoint.py` writes a checkpoint folder with `encoder.safetensors`, `projector.safetensors` and `lora.safetensors` (byte for byte the weights of the paper's app), a `config.json` with the task's prompts, and a model card:

```bash
uv run tools/convert_checkpoint.py --input model_checkpoint.pt --out <folder>/llama-3.2-1b-ecg-sp-mlx --task ecg-qa
```

The demo app bundles the checkpoints from `Tests/UITests/Checkpoints/` and the evaluation data from `Tests/UITests/EvalData/`, which git ignores. Restore them with

```bash
scripts/fetch_assets.sh --checkpoints <folder with llama-3.2-1b-{sleep,ecg}-sp-mlx> <source>
```

where `<source>` is laid out like the paper artifact's `PocketTSLM/Supporting Files`, with the Sleep-EDF exports (`sleep_cot.csv`, `sleep_cot_test20.csv`) and the ECG-QA exports (`ecg_qa_cot_test.csv`, `ecg_qa_template_answers.json`, `ecg_qa_waveforms/`) in `OpenTSLM/`. The script checks every file against `scripts/assets.sha256`.

## Running the app

1. Open `Tests/UITests/UITests.xcodeproj` and select the `TestApp` scheme.
2. Choose your development team under *Signing & Capabilities*. Do not commit it.
3. Duplicate the scheme, untick *Shared*, and add the environment variable `HF_TOKEN` with a Hugging Face token whose account has accepted the Llama 3.2 license. Never commit the token. The first launch downloads the 2.5 GB base model into the app's Documents folder.
4. Run on the iPhone.

The task sheet offers single-sample runs (*EEG sample*, *ECG sample*, *My Watch ECG*). Under *Evaluations and benchmarks* it has the accuracy evaluations on 20 test samples each (*EEG eval (20)*, *ECG eval (20)*), the context-economics benchmarks (*EEG bench*, *ECG bench*) and repeated runs (*EEG var (5×)*, *ECG var (5×)*). Evaluations and benchmarks write JSON reports to the app's Documents folder: `opentslm_sleep_eval.json` and `opentslm_ecg_eval.json` with every generated output, `opentslm_context_economics[_ecg].json`, and `opentslm_repeated_{eeg,ecg}.json`.

## Evaluations on the Mac

`pockettslm-cli eval` runs a checkpoint's accuracy evaluation as the app does and writes the same report JSON. Build it with Xcode, which compiles MLX's Metal shaders, and run the binary from the build folder:

```bash
xcodebuild -scheme pockettslm-cli -destination 'platform=macOS,arch=arm64,name=My Mac' build
pockettslm-cli eval --base <Llama-3.2-1B folder> --checkpoint Tests/UITests/Checkpoints/llama-3.2-1b-sleep-sp-mlx \
    --data Tests/UITests/EvalData --output opentslm_sleep_eval.json
```

`baselines/macos-mlx-0.21.3/` holds the outputs of both evaluations, and its README shows how to compare a run with them.

## Reproducing the paper

**Parity tests (§3).** Component-level Python↔Swift parity (encoder, projector, fused pipeline, soft-prompt interleaver, RoPE, tokenizer, text serializer) against committed fixtures, no Python needed. Run them through Xcode, which compiles MLX's Metal shaders; `swift test` on the command line cannot:

```bash
xcodebuild test -scheme PocketTSLM-Package -destination 'platform=macOS,arch=arm64,name=My Mac'
```

- The encoder and projector tests use small modules with seeded random weights.
- The tokenizer test reads Meta's tokenizer from the Hugging Face cache (`~/.cache/huggingface/hub`, the revision the checkpoints pin) or from the folder in `POCKETTSLM_TOKENIZER_DIR`, and is skipped without it.
- The real-weight ECG tests run the converted ECG checkpoint and only run with `POCKETTSLM_INTEGRATION=1`, which `xcodebuild` passes on as `TEST_RUNNER_POCKETTSLM_INTEGRATION=1`. They take the checkpoint from `POCKETTSLM_ECG_CHECKPOINT`, else from `Tests/UITests/Checkpoints/`, else from Hugging Face.
- `Tests/PocketTSLMTests/generate_parity_fixtures.py` regenerates the encoder, projector and real-weight fixtures from the MLX Python reference.

Tolerances are max 1e-4 / mean 1e-5 (1e-3 for RoPE). The greedy decode itself runs in the app (`OpenTSLMLLM`); the tests cover the pipeline that feeds it.

**Benchmark (§4).** Run the app on an iPhone and tap **EEG bench** / **ECG bench**: each sweeps time-series tokens, time-to-first-token, and peak memory, and writes `opentslm_context_economics[_ecg].json` to the app's Documents folder. Our reference outputs (iPhone Air, iOS 26.4.1) are in `benchmark-results/`.

## License

MIT, see `LICENSE.md`.
