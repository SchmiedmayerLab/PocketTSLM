<!--

This source file is part of the PocketTSLM open-source project

SPDX-FileCopyrightText: 2026 Schmiedmayer Lab and the project authors (see CONTRIBUTORS.md)

SPDX-License-Identifier: MIT

-->

# Mac baselines on MLX Swift 0.21.3

The demo app's two accuracy evaluations, run on a Mac by `pockettslm-cli` at commit `b9b43aa`. The stack is MLX Swift 0.21.3, mlx-swift-examples `0c50f71e` and swift-transformers 0.1.20. Later changes must reproduce the generated text; only `seconds` varies between runs. Mac and iPhone outputs can differ, so compare these with Mac runs only.

| File | Checkpoint | Rows | Correct | Parsed |
|---|---|---|---|---|
| `opentslm_sleep_eval.json` | `llama-3.2-1b-sleep-sp-mlx` | `sleep_cot_test20.csv`, rows 0–19 | 17/20 | 20/20 |
| `opentslm_ecg_eval.json` | `llama-3.2-1b-ecg-sp-mlx` | `ecg_qa_cot_test.csv`, rows 0–19 | 10/20 | 20/20 |

Recorded on a MacBook Pro with an Apple M1 Pro, macOS 15.7.7 and Xcode 26.0, on the GPU, from a Debug build. A rerun gave identical text. The base model is `meta-llama/Llama-3.2-1B` at revision `4e20de362430cd3b72f300e6b0f18e50e7166e08`.

To reproduce, build the CLI with Xcode, which compiles MLX's Metal shaders, then run it from the repository root:

```sh
xcodebuild -scheme pockettslm-cli -destination 'platform=macOS,arch=arm64,name=My Mac' build
CLI=~/Library/Developer/Xcode/DerivedData/<PocketTSLM-…>/Build/Products/Debug/pockettslm-cli
BASE=~/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08
for task in sleep ecg; do
    $CLI eval --base $BASE --checkpoint Tests/UITests/Checkpoints/llama-3.2-1b-$task-sp-mlx \
        --data Tests/UITests/EvalData --output opentslm_${task}_eval.json
done
```

`scripts/fetch_assets.sh` fills `Tests/UITests/Checkpoints` and `Tests/UITests/EvalData`. To compare a run with its baseline, ignoring the timings:

```sh
diff <(jq 'del(.rows[].seconds)' baselines/macos-mlx-0.21.3/opentslm_sleep_eval.json) \
     <(jq 'del(.rows[].seconds)' opentslm_sleep_eval.json)
```
