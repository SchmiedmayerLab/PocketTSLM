#
# This source file is part of the PocketTSLM open-source project
#
# SPDX-FileCopyrightText: 2026 Schmiedmayer Lab and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#
# /// script
# requires-python = "==3.12.*"
# dependencies = ["mlx==0.21.1", "numpy", "safetensors"]
# ///

"""Writes the encoder and projector parity fixtures of the Swift tests with the MLX Python reference.

The reference is the `src` folder of an OpenTSLMMLX checkout. mlx 0.21.1 is the release closest to
the MLX core (0.21.0) of mlx-swift 0.21.3. From the repository root:

    export POCKETTSLM_MLX_REFERENCE_SRC=<OpenTSLMMLX checkout>/src
    uv run Tests/PocketTSLMTests/generate_parity_fixtures.py tiny
    uv run Tests/PocketTSLMTests/generate_parity_fixtures.py real \\
        --checkpoint Tests/UITests/Checkpoints/llama-3.2-1b-ecg-sp-mlx --data Tests/UITests/EvalData

`tiny` gives a small encoder and projector seeded random weights and writes them with an input and
the reference outputs (`tiny_*.safetensors`, a few kB) for the offline tests. `real` runs the ECG
checkpoint on the first ECG-QA test row and writes the input and a strided subsample of the outputs
(`real_ecg_parity.safetensors`). The integration tests load the checkpoint themselves, so no real
weights are committed.
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten
from safetensors.numpy import save_file

# Import the reference without writing bytecode into its checkout.
sys.dont_write_bytecode = True
sys.path.insert(0, os.environ["POCKETTSLM_MLX_REFERENCE_SRC"])
from ts_encoder import TransformerCNNEncoder  # noqa: E402
from ts_projector import MLPProjector  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "Fixtures"
# Strides of the subsample of the real outputs, [lead, patch, feature]; `RealECGParityTests` uses the same.
ENCODED_STRIDES = (1, 10, 8)
PROJECTED_STRIDES = (1, 10, 64)


def save(arrays, name):
    # Contiguous copies: safetensors 0.8.0 writes the raw buffer of non-contiguous arrays.
    arrays = {key: np.ascontiguousarray(value, dtype=np.float32) for key, value in arrays.items()}
    save_file(arrays, str(FIXTURES / name))
    print(f"wrote {name}: " + ", ".join(f"{key} {list(value.shape)}" for key, value in arrays.items()))


def randomize(module, rng):
    """Replaces every parameter, `pos_embed` included, with seeded normal values."""
    weights = {
        key: (rng.standard_normal(value.shape) * 0.5).astype(np.float32)
        for key, value in tree_flatten(module.parameters())
    }
    module.load_weights([(key, mx.array(value)) for key, value in weights.items()])
    return weights


def tiny(_):
    rng = np.random.default_rng(0)
    # `TransformerCNNEncoderTests` and `MLPProjectorTests` build the same modules.
    encoder = TransformerCNNEncoder(
        output_dim=16, transformer_input_dim=16, num_heads=2, num_layers=2, patch_size=4, ff_dim=32, max_patches=16
    )
    projector = MLPProjector(input_dim=16, output_dim=24)
    save(randomize(encoder, rng), "tiny_encoder.safetensors")
    save(randomize(projector, rng), "tiny_projector.safetensors")

    series = rng.standard_normal((2, 64)).astype(np.float32)
    encoded = encoder(mx.array(series))
    projected = projector(encoded)
    save({"input": series, "encoded": np.array(encoded), "projected": np.array(projected)}, "tiny_io.safetensors")


def real(arguments):
    with open(arguments.data / "ecg_qa_cot_test.csv", newline="", encoding="utf-8") as file:
        row = next(csv.DictReader(file))
    waveform = json.loads((arguments.data / "ecg_qa_waveforms" / f"{row['ecg_id'].strip(' []')}.json").read_text())

    config = json.loads((arguments.checkpoint / "config.json").read_text())
    leads = np.array([lead[: config["series"]["maxLength"]] for lead in waveform["time_series"]], dtype=np.float32)
    dimensions = config["encoder"]["embedDim"]
    encoder = TransformerCNNEncoder(
        output_dim=dimensions,
        transformer_input_dim=dimensions,
        patch_size=config["encoder"]["patchSize"],
        max_patches=config["encoder"]["maxPatches"],
    )
    encoder.load_weights(str(arguments.checkpoint / "encoder.safetensors"))
    projector = MLPProjector(input_dim=dimensions, output_dim=config["baseModel"]["hiddenSize"])
    projector.load_weights(str(arguments.checkpoint / "projector.safetensors"))

    encoded = encoder(mx.array(leads))
    projected = projector(encoded)

    def subsample(array, strides):
        return np.array(array)[:: strides[0], :: strides[1], :: strides[2]]

    save(
        {
            "input": leads,
            "encoded": subsample(encoded, ENCODED_STRIDES),
            "projected": subsample(projected, PROJECTED_STRIDES),
        },
        "real_ecg_parity.safetensors",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(required=True)
    commands.add_parser("tiny", help="seeded random weights, for the offline tests").set_defaults(run=tiny)
    command = commands.add_parser("real", help="the ECG checkpoint on the first ECG-QA test row")
    command.add_argument("--checkpoint", type=Path, required=True, help="converted ECG checkpoint folder")
    command.add_argument("--data", type=Path, required=True, help="folder with the ECG-QA test export")
    command.set_defaults(run=real)
    arguments = parser.parse_args()

    mx.set_default_device(mx.cpu)
    arguments.run(arguments)


if __name__ == "__main__":
    main()
