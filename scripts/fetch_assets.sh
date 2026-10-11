#!/usr/bin/env bash
#
# This source file is part of the PocketTSLM open-source project
#
# SPDX-FileCopyrightText: 2026 Schmiedmayer Lab and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#

# Restores the model files and evaluation data that the app reads from
# "Tests/UITests/TestApp/Supporting Files/" and that this repository does not track.
#
# Usage: scripts/fetch_assets.sh [--with-base-weights] <source>
#
#   <source>             a folder laid out like "PocketTSLM/Supporting Files" of the paper
#                        artifact, with LocalLLM/ and OpenTSLM/ subfolders
#   --with-base-weights  also copy LocalLLM/model.safetensors (2.5 GB), which Xcode then bundles.
#                        Without it, the app downloads meta-llama/Llama-3.2-1B on first launch,
#                        which needs HF_TOKEN in the Xcode scheme.
#
# Every file is checked against scripts/assets.sha256. Xcode's synced folder copies all of them
# to the root of the app bundle, so file names must stay unique across the subfolders.

set -euo pipefail

usage() {
    echo "usage: $0 [--with-base-weights] <source>" >&2
    exit 64
}

with_base_weights=0
if [[ "${1:-}" == "--with-base-weights" ]]; then
    with_base_weights=1
    shift
fi
[[ $# -eq 1 ]] || usage
source_dir="$1"

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
destination="$repo_root/Tests/UITests/TestApp/Supporting Files"
manifest="$(mktemp)"
trap 'rm -f "$manifest"' EXIT

if [[ $with_base_weights -eq 1 ]]; then
    cp "$repo_root/scripts/assets.sha256" "$manifest"
else
    grep -v ' LocalLLM/model\.safetensors$' "$repo_root/scripts/assets.sha256" > "$manifest"
fi

while read -r _ path; do
    if [[ ! -f "$source_dir/$path" ]]; then
        echo "error: $source_dir/$path not found" >&2
        exit 1
    fi
    mkdir -p "$destination/$(dirname "$path")"
    # -c clones the file on APFS, so even the base weights take no extra disk space.
    cp -c "$source_dir/$path" "$destination/$path" 2>/dev/null || cp "$source_dir/$path" "$destination/$path"
done < "$manifest"

(cd "$destination" && shasum -a 256 --quiet -c "$manifest")
echo "Restored $(wc -l < "$manifest" | tr -d ' ') files into $destination"
