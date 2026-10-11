#!/usr/bin/env bash
#
# This source file is part of the PocketTSLM open-source project
#
# SPDX-FileCopyrightText: 2026 Schmiedmayer Lab and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#

# Restores the files the demo app bundles but this repository does not track: the base model's
# config and tokenizer (Tests/UITests/TestApp/Supporting Files/LocalLLM), the converted
# checkpoints (Tests/UITests/Checkpoints) and the evaluation data (Tests/UITests/EvalData).
#
# Usage: scripts/fetch_assets.sh [--with-base-weights] --checkpoints <folder> <source>
#
#   <source>             a folder laid out like "PocketTSLM/Supporting Files" of the paper
#                        artifact, with LocalLLM/ (base model) and OpenTSLM/ (evaluation data)
#   --checkpoints        a folder with llama-3.2-1b-{sleep,ecg}-sp-mlx/, as tools/convert_checkpoint.py
#                        writes them
#   --with-base-weights  also copy LocalLLM/model.safetensors (2.5 GB), which Xcode then bundles.
#                        Without it, the app downloads meta-llama/Llama-3.2-1B on first launch,
#                        which needs HF_TOKEN in the Xcode scheme.
#
# Every file is checked against scripts/assets.sha256. Xcode's synced folder copies the LocalLLM
# files to the root of the app bundle, so their names must not clash with other app resources.

set -euo pipefail

usage() {
    echo "usage: $0 [--with-base-weights] --checkpoints <folder> <source>" >&2
    exit 64
}

with_base_weights=0
checkpoints=""
while [[ $# -gt 1 ]]; do
    case "$1" in
        --with-base-weights) with_base_weights=1; shift ;;
        --checkpoints) checkpoints="$2"; shift 2 ;;
        *) usage ;;
    esac
done
[[ $# -eq 1 && -n "$checkpoints" ]] || usage
source_dir="$1"

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
destination="$repo_root/Tests/UITests"
manifest="$(mktemp)"
trap 'rm -f "$manifest"' EXIT

if [[ $with_base_weights -eq 1 ]]; then
    cp "$repo_root/scripts/assets.sha256" "$manifest"
else
    grep -v ' TestApp/Supporting Files/LocalLLM/model\.safetensors$' "$repo_root/scripts/assets.sha256" > "$manifest"
fi

# Where a manifest path comes from.
source_of() {
    case "$1" in
        "TestApp/Supporting Files/LocalLLM/"*) echo "$source_dir/LocalLLM/${1#TestApp/Supporting Files/LocalLLM/}" ;;
        Checkpoints/*) echo "$checkpoints/${1#Checkpoints/}" ;;
        EvalData/*) echo "$source_dir/OpenTSLM/${1#EvalData/}" ;;
    esac
}

while read -r _ path; do
    source="$(source_of "$path")"
    if [[ ! -f "$source" ]]; then
        echo "error: $source not found" >&2
        exit 1
    fi
    mkdir -p "$destination/$(dirname "$path")"
    # -c clones the file on APFS, so even the base weights take no extra disk space.
    cp -c "$source" "$destination/$path" 2>/dev/null || cp "$source" "$destination/$path"
done < "$manifest"

(cd "$destination" && shasum -a 256 --quiet -c "$manifest")
echo "Restored $(wc -l < "$manifest" | tr -d ' ') files into $destination"
