#!/usr/bin/env bash
# Export iOS ECG-QA CoT CSV prefix + PTB-XL waveform sidecar from OpenTSLM loader.
set -euo pipefail

OPENTSLMMLX="${OPENTSLMMLX:-$HOME/Documents/GitHub/OpenTSLMMLX}"
OPENTSLM_SRC="${OPENTSLM_SRC:-$HOME/OpenTSLM/src}"
SCRIPTS="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="${OUT_DIR:-$SCRIPTS/../Tests/UITests/EvalData}"
MAX_ROWS="${MAX_ROWS:-500}"

cd "$OPENTSLMMLX"
source .venv/bin/activate
pip install -q wfdb datasets

python "$SCRIPTS/export_ecg_qa_ios_assets.py" \
  --split test \
  --max-rows "$MAX_ROWS" \
  --opentslm-src "$OPENTSLM_SRC" \
  --out-dir "$OUT_DIR"

echo "iOS CSV + waveforms: $OUT_DIR"
