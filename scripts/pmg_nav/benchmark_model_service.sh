#!/usr/bin/env bash
set -euo pipefail
: "${MODEL_BASE_URL:?set MODEL_BASE_URL, e.g. https://proxy.example.com}"
NAME="${MODEL_SERVICE_NAME:-default}"
REQUESTS="${MODEL_SERVICE_REQUESTS:-5}"
RECORDS="${EXPERIMENT_RECORDS:-records/experiment_runs.jsonl}"
args=(http --name "$NAME" --url "${MODEL_BASE_URL%/}/v1/models" --requests "$REQUESTS" --records "$RECORDS")
[[ -n "${MODEL_INPUT_PRICE:-}" ]] && args+=(--input-price "$MODEL_INPUT_PRICE")
[[ -n "${MODEL_OUTPUT_PRICE:-}" ]] && args+=(--output-price "$MODEL_OUTPUT_PRICE")
python3 "$(dirname "$0")/experiment_runner.py" "${args[@]}"
