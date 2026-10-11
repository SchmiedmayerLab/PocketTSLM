#
# This source file is part of the PocketTSLM open-source project
#
# SPDX-FileCopyrightText: 2026 Schmiedmayer Lab and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#
# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "safetensors", "numpy"]
# ///

"""Convert an OpenTSLM-SP checkpoint into a PocketTSLM checkpoint folder.

The input is a PyTorch `model_checkpoint.pt` as published by OpenTSLM on Hugging Face, for
example `OpenTSLM/llama-3.2-1b-ecg-sp`. The output folder holds

    encoder.safetensors    TransformerCNNEncoder weights in MLX layout
    projector.safetensors  MLPProjector weights in MLX layout
    lora.safetensors       PEFT LoRA adapters in MLX `LoRALinear` layout

Usage:
    uv run tools/convert_checkpoint.py --input model_checkpoint.pt --out <folder>

The checkpoint is read with `torch.load(weights_only=True)`. Its PEFT config objects are
restored into stand-in classes, so no code from the pickle runs and PEFT is not needed.

Merged from the app's `scripts/convert_opentslm_checkpoint.py` and Nikhil Krishnaswamy's
`convert_checkpoint.py` for OpenTSLMMLX.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import load_file, save_file

# The classes OpenTSLM pickles reference besides tensors and containers.
PEFT_CLASSES = [
    "peft.tuners.lora.config.LoraConfig",
    "peft.tuners.lora.config.LoraRuntimeConfig",
    "peft.utils.peft_types.PeftType",
    "peft.utils.peft_types.TaskType",
]


class _PickledObject:
    """Stand-in for a PEFT class: keeps the pickled state and runs no code."""

    def __init__(self, *args):
        self.args = args

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self.state = state


def load_checkpoint(path: Path) -> dict:
    stand_ins = [(type(name.rsplit(".", 1)[1], (_PickledObject,), {}), name) for name in PEFT_CLASSES]
    with torch.serialization.safe_globals(stand_ins):
        return torch.load(path, map_location="cpu", weights_only=True)


def _array(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().float().numpy()


def convert_encoder(state: dict) -> dict[str, np.ndarray]:
    """TransformerCNNEncoder: PyTorch state dict → MLX names and layouts."""
    out = {
        # Conv1d: PyTorch [C_out, C_in, K] → MLX [C_out, K, C_in]
        "patch_embed.weight": _array(state["patch_embed.weight"]).transpose(0, 2, 1),
        "pos_embed": _array(state["pos_embed"]),
        "input_norm.weight": _array(state["input_norm.weight"]),
        "input_norm.bias": _array(state["input_norm.bias"]),
    }
    num_layers = sum(1 for key in state if key.startswith("encoder.layers.") and key.endswith(".norm1.weight"))
    if num_layers == 0:
        raise ValueError("no encoder layers found in encoder_state")

    for i in range(num_layers):
        source, target = f"encoder.layers.{i}", f"layers.{i}"
        # PyTorch packs Q, K and V into one in_proj; MLX keeps three projections.
        in_weight = _array(state[f"{source}.self_attn.in_proj_weight"])
        in_bias = _array(state[f"{source}.self_attn.in_proj_bias"])
        d = in_weight.shape[1]
        for j, name in enumerate(("query_proj", "key_proj", "value_proj")):
            out[f"{target}.self_attn.{name}.weight"] = in_weight[j * d:(j + 1) * d]
            out[f"{target}.self_attn.{name}.bias"] = in_bias[j * d:(j + 1) * d]
        for name in ("self_attn.out_proj", "linear1", "linear2", "norm1", "norm2"):
            for parameter in ("weight", "bias"):
                out[f"{target}.{name}.{parameter}"] = _array(state[f"{source}.{name}.{parameter}"])
    return out


def convert_projector(state: dict) -> dict[str, np.ndarray]:
    """MLPProjector (LayerNorm, then Linear): PyTorch Sequential indices → MLX names."""
    return {
        "norm.weight": _array(state["projector.0.weight"]),
        "norm.bias": _array(state["projector.0.bias"]),
        "linear.weight": _array(state["projector.1.weight"]),
        "linear.bias": _array(state["projector.1.bias"]),
    }


_LORA_KEY = re.compile(r"^base_model\.model\.(.+)\.(lora_[AB])\.default\.weight$")


def convert_lora(state: dict) -> dict[str, np.ndarray]:
    """PEFT adapters → MLX `LoRALinear` names and layouts.

    PEFT stores `base_model.model.<path>.lora_A.default.weight` as [r, in] and `lora_B` as
    [out, r]. MLX wants `<path>.lora_a` as [in, r] and `<path>.lora_b` as [r, out].
    """
    out = {}
    for key, tensor in state.items():
        match = _LORA_KEY.match(key)
        if not match:
            continue
        path, matrix = match.group(1), match.group(2).lower()
        if not path.startswith("model."):
            path = f"model.{path}"
        out[f"{path}.{matrix}"] = _array(tensor).T
    if not out:
        raise ValueError("no PEFT LoRA tensors found in lora_state")
    return out


def save(tensors: dict[str, np.ndarray], path: Path) -> None:
    """Write row-major copies and read them back.

    safetensors 0.8.0 writes the raw buffer of a non-contiguous array, which silently scrambles
    transposed weights such as the LoRA matrices; 0.7.0 and earlier write them correctly.
    """
    tensors = {key: np.ascontiguousarray(value) for key, value in tensors.items()}
    save_file(tensors, str(path))
    written = load_file(str(path))
    if written.keys() != tensors.keys() or any(not np.array_equal(written[key], tensors[key]) for key in tensors):
        raise RuntimeError(f"{path} does not read back as written")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="OpenTSLM-SP model_checkpoint.pt")
    parser.add_argument("--out", type=Path, required=True, help="output folder")
    args = parser.parse_args()

    checkpoint = load_checkpoint(args.input)
    if not checkpoint.get("lora_state"):
        parser.error("the checkpoint has no LoRA adapters")
    encoder = convert_encoder(checkpoint["encoder_state"])
    projector = convert_projector(checkpoint["projector_state"])
    lora = convert_lora(checkpoint["lora_state"])

    args.out.mkdir(parents=True, exist_ok=True)
    save(encoder, args.out / "encoder.safetensors")
    save(projector, args.out / "projector.safetensors")
    save(lora, args.out / "lora.safetensors")
    print(f"Wrote {args.out}: {len(encoder)} encoder, {len(projector)} projector and {len(lora)} LoRA tensors")


if __name__ == "__main__":
    main()
