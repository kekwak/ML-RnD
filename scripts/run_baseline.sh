#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="${1:-configs/baseline.yaml}"

uv sync
[ -f data/train.csv ] || uv run python -m src.misc.fetch_dataset

uv run python -m src.train.train --config "$CONFIG"

RUN_DIR=$(ls -td runs/*/ | head -1)
uv run python -m src.evaluation.evaluate --submission "${RUN_DIR}submission.csv" | tee "${RUN_DIR}test_metrics.json"
