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
    config.json            checkpoint description, format "pockettslm-checkpoint/1"
    README.md              model card

Usage:
    uv run tools/convert_checkpoint.py --input model_checkpoint.pt --out <folder> --task ecg-qa \
        [--source-repo OpenTSLM/llama-3.2-1b-ecg-sp --source-revision <commit>]

The checkpoint is read with `torch.load(weights_only=True)`. Its PEFT config objects are
restored into stand-in classes, so no code from the pickle runs and PEFT is not needed.

Merged from the app's `scripts/convert_opentslm_checkpoint.py` and Nikhil Krishnaswamy's
`convert_checkpoint.py` for OpenTSLMMLX.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import load_file, save_file

FORMAT = "pockettslm-checkpoint/1"

# Base models with their pinned Hugging Face revision and the SHA-256 of `model.safetensors`.
BASE_MODELS = {
    "meta-llama/Llama-3.2-1B": {
        "revision": "4e20de362430cd3b72f300e6b0f18e50e7166e08",
        "weightsSHA256": "68a2e4be76fa709455a60272fba8e512c02d81c46e6c671cc9449e374fd6809a",
    },
}

# Llama's projections in module order, so config.json lists LoRA targets in a stable order.
LLAMA_PROJECTIONS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

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


def lora_settings(config: dict, lora: dict[str, np.ndarray]) -> dict:
    """Rank, alpha and target modules from the PEFT config stored in the checkpoint."""
    if config.get("use_rslora") or config.get("use_dora") or config.get("rank_pattern") \
            or config.get("alpha_pattern") or config.get("bias", "none") != "none":
        raise ValueError("only plain LoRA is supported: no rsLoRA, DoRA, bias or per-module ranks")
    rank = int(config["r"])
    ranks = {matrix.shape[1] for key, matrix in lora.items() if key.endswith(".lora_a")}
    if ranks != {rank}:
        raise ValueError(f"LoRA rank {rank} in the config does not match the tensors: {sorted(ranks)}")
    modules = set(config["target_modules"])
    keys = [name for name in LLAMA_PROJECTIONS if name in modules] + sorted(modules - set(LLAMA_PROJECTIONS))
    return {"rank": rank, "alpha": config["lora_alpha"], "keys": keys}


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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def converter_version() -> str:
    """`tools/convert_checkpoint.py@<commit>`, marked dirty if the script has uncommitted changes."""
    script = Path(__file__).resolve()

    def git(*args: str) -> str:
        result = subprocess.run(["git", "-C", str(script.parent), *args], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else ""

    commit = git("rev-parse", "--short=12", "HEAD") or "unknown"
    if git("status", "--porcelain", "--", script.name):
        commit += "-dirty"
    return f"tools/{script.name}@{commit}"


def model_card(config: dict) -> str:
    base, encoder, lora, source = config["baseModel"], config["encoder"], config["lora"], config["source"]
    origin = f"`{source['file']}` (SHA-256 `{source['sha256']}`)"
    if "repo" in source:
        origin += f" in [`{source['repo']}`](https://huggingface.co/{source['repo']})"
    if "revision" in source:
        origin += f" at revision `{source['revision']}`"
    lines = [
        "---",
        "library_name: mlx",
        f"base_model: {base['id']}",
        "tags:",
        "- pockettslm",
        "- opentslm",
        "- time-series",
        "---",
        "",
        f"# PocketTSLM checkpoint: {config['task']}",
        "",
        f"OpenTSLM-SP weights for `{base['id']}`, converted to MLX for "
        "[PocketTSLM](https://github.com/SchmiedmayerLab/PocketTSLM), which runs them on iPhone.",
        "",
        "| File | Contents |",
        "|---|---|",
        f"| `encoder.safetensors` | time-series encoder: patch size {encoder['patchSize']}, "
        f"dimension {encoder['embedDim']}, up to {encoder['maxPatches']} patches |",
        f"| `projector.safetensors` | projector into the {base['hiddenSize']}-dimensional token embeddings |",
        f"| `lora.safetensors` | LoRA adapters, rank {lora['rank']} and alpha {lora['alpha']}, "
        f"on {', '.join(lora['keys'])} |",
        "| `config.json` | what PocketTSLM needs to run the checkpoint |",
        "",
        f"The base model is not included: PocketTSLM loads `{base['id']}`"
        + (f" at revision `{base['revision']}`" if "revision" in base else "") + " separately.",
        "",
        "## Source",
        "",
        f"Converted from {origin} with `{source['converter']}`.",
        "",
        "Built with Llama.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="OpenTSLM-SP model_checkpoint.pt")
    parser.add_argument("--out", type=Path, required=True, help="output folder")
    parser.add_argument("--task", required=True, help='task name for config.json, for example "ecg-qa"')
    parser.add_argument("--source-repo", help="Hugging Face repo of the input, e.g. OpenTSLM/llama-3.2-1b-ecg-sp")
    parser.add_argument("--source-revision", help="commit of --source-repo that the input comes from")
    args = parser.parse_args()

    checkpoint = load_checkpoint(args.input)
    if not checkpoint.get("lora_state") or "lora_config" not in checkpoint:
        parser.error("the checkpoint has no LoRA adapters or no PEFT config")
    peft_config = vars(checkpoint["lora_config"]["default"])
    encoder = convert_encoder(checkpoint["encoder_state"])
    projector = convert_projector(checkpoint["projector_state"])
    lora = convert_lora(checkpoint["lora_state"])

    base_id = peft_config["base_model_name_or_path"]
    source = {"repo": args.source_repo, "revision": args.source_revision, "file": args.input.name,
              "sha256": sha256(args.input), "converter": converter_version()}
    config = {
        "format": FORMAT,
        "task": args.task,
        "baseModel": {"id": base_id, **BASE_MODELS.get(base_id, {}),
                      "hiddenSize": int(projector["linear.weight"].shape[0])},
        "encoder": {"patchSize": int(encoder["patch_embed.weight"].shape[1]),
                    "embedDim": int(encoder["patch_embed.weight"].shape[0]),
                    "maxPatches": int(encoder["pos_embed"].shape[1])},
        "lora": lora_settings(peft_config, lora),
        "source": {key: value for key, value in source.items() if value is not None},
    }

    args.out.mkdir(parents=True, exist_ok=True)
    save(encoder, args.out / "encoder.safetensors")
    save(projector, args.out / "projector.safetensors")
    save(lora, args.out / "lora.safetensors")
    (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (args.out / "README.md").write_text(model_card(config))
    print(f"Wrote {args.out}: {len(encoder)} encoder, {len(projector)} projector and {len(lora)} LoRA tensors")


if __name__ == "__main__":
    main()
