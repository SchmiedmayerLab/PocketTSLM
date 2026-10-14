#!/usr/bin/env bash
#
# This source file is part of the PocketTSLM open-source project
#
# SPDX-FileCopyrightText: 2026 Schmiedmayer Lab and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#

# Restores the files the demo app bundles but this repository does not track: the converted
# checkpoints (Tests/UITests/Checkpoints) and the evaluation data (Tests/UITests/EvalData). The app
# downloads the base model itself.
#
# Usage: scripts/fetch_assets.sh --checkpoints <folder> <source>
#
#   <source>       a folder laid out like "PocketTSLM/Supporting Files" of the paper artifact, with
#                  the evaluation data in OpenTSLM/
#   --checkpoints  a folder with llama-3.2-1b-{sleep,ecg}-sp-mlx/, as tools/convert_checkpoint.py
#                  writes them
#
# Every file is checked against scripts/assets.sha256.

set -euo pipefail

usage() {
    echo "usage: $0 --checkpoints <folder> <source>" >&2
    exit 64
}

checkpoints=""
while [[ $# -gt 1 ]]; do
    case "$1" in
        --checkpoints) checkpoints="$2"; shift 2 ;;
        *) usage ;;
    esac
done
[[ $# -eq 1 && -n "$checkpoints" ]] || usage
source_dir="$1"

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
destination="$repo_root/Tests/UITests"
manifest="$repo_root/scripts/assets.sha256"

# Where a manifest path comes from.
source_of() {
    case "$1" in
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
    # -c clones the file on APFS, so the copies take no extra disk space.
    cp -c "$source" "$destination/$path" 2>/dev/null || cp "$source" "$destination/$path"
done < "$manifest"

(cd "$destination" && shasum -a 256 --quiet -c "$manifest")
echo "Restored $(wc -l < "$manifest" | tr -d ' ') files into $destination"
