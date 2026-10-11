<!--

This source file is part of the PocketTSLM open-source project

SPDX-FileCopyrightText: 2023 Schmiedmayer Lab and the project authors (see CONTRIBUTORS.md)

SPDX-License-Identifier: MIT

-->

# PocketTSLM

**PocketTSLM** ports a soft-prompt time-series language model (OpenTSLM-SP) to run fully on-device with MLX Swift: the encoder, the projector and the LLM all execute on an iPhone, so physiological time series (ECG, EEG) can be interpreted in natural language without sending health data off the device. This repository holds the iOS app, the Python↔Swift numerical-parity tests, and the on-device context-economics benchmark reported in the paper.

The repository contains code only. Model weights and evaluation data are restored with `scripts/fetch_assets.sh`.

## Repository layout

- `PocketTSLM/`: the iOS app, with on-device inference, accuracy evaluations and benchmarks.
- `src-swift/`: the `OpenTSLMKit` Swift package (encoder, projector, interleaver, serializer) and its parity tests.
- `benchmark-results/`: reference benchmark outputs from our run.
- `scripts/`: asset restore and dataset export helpers.
- `tools/`: the checkpoint converter.

## Requirements

- macOS with Xcode 26 and an Apple-silicon iPhone, for the latency and memory numbers. MLX needs a real Metal GPU; in the simulator the app falls back to the CPU, which is only good for diagnostics.
- Access to [`meta-llama/Llama-3.2-1B`](https://huggingface.co/meta-llama/Llama-3.2-1B), which is gated, unless you bundle the base weights (see below).

## Model files and data

The app reads its model files and evaluation data from `PocketTSLM/Supporting Files/LocalLLM/` and `PocketTSLM/Supporting Files/OpenTSLM/`, which git ignores. Restore them from a folder laid out like the paper artifact's `PocketTSLM/Supporting Files`:

```bash
scripts/fetch_assets.sh [--with-base-weights] <folder>
```

The script copies
- the Llama-3.2-1B config and tokenizer files into `LocalLLM/`,
- the converted sleep and ECG checkpoints, `mlx-checkpoint{,.ecg}.{encoder,projector,lora}.safetensors`, into `OpenTSLM/`,
- the evaluation data into `OpenTSLM/`: `sleep_cot.csv` and `sleep_cot_test20.csv` (Sleep-EDF), `ecg_qa_cot_test.csv`, `ecg_qa_cot_test_0.json`, `ecg_qa_template_answers.json` and `ecg_qa_waveforms/` (ECG-QA),

and checks every file against `scripts/assets.sha256`. With `--with-base-weights` it also copies the 2.5 GB `LocalLLM/model.safetensors`, which then ships inside the app. Without it, the app downloads the base weights from Hugging Face on first launch.

The checkpoints are conversions of the OpenTSLM-SP releases on Hugging Face, `OpenTSLM/llama-3.2-1b-sleep-sp` and `OpenTSLM/llama-3.2-1b-ecg-sp`. `tools/convert_checkpoint.py` writes the same weights byte for byte, as `encoder.safetensors`, `projector.safetensors` and `lora.safetensors`, together with a `config.json` and a model card:

```bash
uv run tools/convert_checkpoint.py --input model_checkpoint.pt --out <folder> --task ecg-qa
```

## Running the app

1. Open `PocketTSLM.xcodeproj` and select the `PocketTSLM` scheme.
2. Choose your development team under *Signing & Capabilities*. Do not commit it.
3. Unless you bundled the base weights, duplicate the scheme, untick *Shared*, and add the environment variable `HF_TOKEN` with a Hugging Face token whose account has accepted the Llama 3.2 license. Never commit the token.
4. Run on the iPhone.

The task sheet offers single-sample runs (*EEG sample*, *ECG sample*, *My Watch ECG*), accuracy evaluations on 20 test samples each (*EEG eval (20)*, *ECG eval (20)*), the context-economics benchmarks (*EEG bench*, *ECG bench*) and repeated runs (*EEG var (5×)*, *ECG var (5×)*). Evaluations and benchmarks write JSON reports to the app's Documents folder: `opentslm_sleep_eval.json` and `opentslm_ecg_eval.json` with every generated output, `opentslm_context_economics[_ecg].json`, and `opentslm_repeated_{eeg,ecg}.json`.

The app resolves mlx-swift 0.21.3, mlx-swift-examples at revision `0c50f71e` and swift-transformers 0.1.20 (`PocketTSLM.xcodeproj/project.xcworkspace/xcshareddata/swiftpm/Package.resolved`).

## Reproducing the paper

**Parity tests (§3).** Component-level Python↔Swift parity (encoder, projector, fused pipeline, soft-prompt interleaver, RoPE, tokenizer, text serializer) against fixtures, no Python needed. Run them through Xcode, which compiles MLX's Metal shaders; `swift test` on the command line cannot:

```bash
cd src-swift
xcodebuild test -scheme OpenTSLMKit-Package -destination 'platform=macOS,arch=arm64,name=My Mac'
```

Tests whose fixtures are missing are skipped:
- the tokenizer test reads Meta's tokenizer from `PocketTSLM/Supporting Files/LocalLLM/` (restored by `scripts/fetch_assets.sh`) or from the folder in `POCKETTSLM_TOKENIZER_DIR`;
- the encoder test needs `encoder_weights.safetensors` (9 MB) from the paper artifact in `src-swift/Tests/OpenTSLMKitTests/Fixtures/`;
- the real-weight ECG tests need the fixtures that `generate_real_parity_fixtures.py` writes.

Tolerances are max 1e-4 / mean 1e-5 (1e-3 for RoPE). In this snapshot the RoPE test's comparison with the default base frequency misses its bound slightly (1.02e-3). The greedy decode itself runs in the app (`OpenTSLMLLM`); the tests cover the pipeline that feeds it.

**Benchmark (§4).** Run the app on an iPhone and tap **EEG bench** / **ECG bench**: each sweeps time-series tokens, time-to-first-token, and peak memory, and writes `opentslm_context_economics[_ecg].json` to the app's Documents folder. Our reference outputs (iPhone Air, iOS 26.4.1) are in `benchmark-results/`.

## License

MIT, see `LICENSE.md`.
