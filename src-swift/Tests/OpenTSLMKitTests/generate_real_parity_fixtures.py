#!/usr/bin/env python3
"""Generate Python↔Swift parity fixtures using the REAL on-device ECG weights and the
REAL ECG-QA CoT sample (test idx 0) — the exact pipeline that diverged between the iOS
Swift port and the pytorch/mlx-python reference.

For each component we dump the shared weights plus the reference input/output that the
mlx-python implementation produces. The Swift tests load the same weights + input and
assert their output matches, localizing where the port diverges.

Run from the repo with a python that has mlx:
    POCKETTSLM_MLX_REFERENCE_SRC=<OpenTSLMMLX checkout>/src \
        python3 src-swift/Tests/OpenTSLMKitTests/generate_real_parity_fixtures.py

POCKETTSLM_OPENTSLM_ASSETS overrides the folder with the ECG checkpoint and sample; it defaults
to the app's Supporting Files/OpenTSLM, which scripts/fetch_assets.sh fills.
"""

import os
import sys
import json
import numpy as np
import mlx.core as mx
from safetensors.numpy import load_file, save_file

MLX_IOS = os.environ["POCKETTSLM_MLX_REFERENCE_SRC"]
sys.path.insert(0, MLX_IOS)
from ts_encoder import TransformerCNNEncoder  # noqa: E402
from ts_projector import MLPProjector  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
BUNDLE = os.environ.get("POCKETTSLM_OPENTSLM_ASSETS", f"{REPO}/PocketTSLM/Supporting Files/OpenTSLM")
SAMPLE = f"{BUNDLE}/ecg_qa_cot_test_0.json"
FIXTURES = os.path.dirname(os.path.abspath(__file__)) + "/Fixtures"
CAP = 1000  # in-distribution window where iOS="no" diverged from reference="yes"


def load_into(module, path):
    w = {k: mx.array(v) for k, v in load_file(path).items()}
    module.load_weights(list(w.items()))
    return load_file(path)  # numpy, for re-saving as the shared Swift weights fixture


def main():
    os.makedirs(FIXTURES, exist_ok=True)

    # Real ECG input: 12 leads, capped, exactly as the app feeds the encoder.
    d = json.load(open(SAMPLE))
    leads = np.array([lead[:CAP] for lead in d["time_series"]], dtype=np.float32)  # [12, CAP]

    # --- Encoder (max_patches = bundled pos_embed length) ---
    enc_w = load_file(f"{BUNDLE}/mlx-checkpoint.ecg.encoder.safetensors")
    max_patches = enc_w["pos_embed"].shape[1]
    encoder = TransformerCNNEncoder(max_patches=max_patches)
    encoder.load_weights([(k, mx.array(v)) for k, v in enc_w.items()])

    enc_out = encoder(mx.array(leads))
    mx.eval(enc_out)
    enc_out_np = np.array(enc_out)

    # --- Projector ---
    proj_w = load_file(f"{BUNDLE}/mlx-checkpoint.ecg.projector.safetensors")
    projector = MLPProjector(input_dim=128, output_dim=2048)
    projector.load_weights([(k, mx.array(v)) for k, v in proj_w.items()])

    proj_out = projector(enc_out)
    mx.eval(proj_out)
    proj_out_np = np.array(proj_out)

    # --- Save shared weights + IO fixtures (float32) ---
    save_file(enc_w, f"{FIXTURES}/real_encoder_weights.safetensors")
    save_file(proj_w, f"{FIXTURES}/real_projector_weights.safetensors")
    save_file({"input": leads, "output": enc_out_np},
              f"{FIXTURES}/real_encoder_io.safetensors")
    save_file({"input": enc_out_np, "output": proj_out_np},
              f"{FIXTURES}/real_projector_io.safetensors")
    # Pipeline fixture: raw leads in, projected embeddings out (encoder→projector).
    save_file({"input": leads, "output": proj_out_np},
              f"{FIXTURES}/real_pipeline_io.safetensors")

    print(f"max_patches={max_patches} cap={CAP}")
    print(f"encoder:   in {leads.shape} -> out {enc_out_np.shape}")
    print(f"projector: in {enc_out_np.shape} -> out {proj_out_np.shape}")
    print(f"wrote fixtures to {FIXTURES}")


if __name__ == "__main__":
    main()
