<!--

This source file is part of the PocketTSLM open-source project

SPDX-FileCopyrightText: 2026 Schmiedmayer Lab and the project authors (see CONTRIBUTORS.md)

SPDX-License-Identifier: MIT

-->

# Mac baselines on MLX Swift 0.29.1

The demo app's two accuracy evaluations on Grove's MLX stack: mlx-swift 0.29.1, mlx-swift-examples 2.29.1 and swift-transformers 1.0.0. `pockettslm-cli` ran them at commit `b33fd96`, on the same Mac, base model, checkpoints and data as [`../macos-mlx-0.21.3/`](../macos-mlx-0.21.3/README.md), which also shows how to reproduce and compare runs.

| File | Correct | Parsed | Rows with the same text as on 0.21.3 |
|---|---|---|---|
| `opentslm_sleep_eval.json` | 17/20 | 20/20 | 13/20 |
| `opentslm_ecg_eval.json` | 10/20 | 20/20 | 9/20 |

No answer changed. In the other rows, greedy decoding picks a different token partway through the rationale, and the text diverges from there.

The cause is MLX itself. With the Phase 4 copy of the patched Llama, which predates the move to mlx-swift-examples 2.29.1, this stack gives exactly the text of this baseline in all 40 rows. So neither the new Llama copy nor any other package change alters the outputs; the float32 kernels of MLX 0.21 and 0.29 differ slightly, neither consistently closer to exact. RoPE at position 3500, for example, is 2.7e-4 (default frequencies) and 4.4e-4 (Llama 3 frequencies) from the exact value on 0.21, and 5.3e-4 and 2.2e-4 on 0.29 (`RoPEParityTests`).
