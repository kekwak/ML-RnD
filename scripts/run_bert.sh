#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

uv sync
[ -f data/train.csv ] || uv run python -m src.misc.fetch_dataset

for CONFIG in configs/bert_frozen.yaml configs/bert_len128.yaml configs/bert_no_pos_weight.yaml configs/bert.yaml; do
    uv run python -m src.train.train_bert --config "$CONFIG"
done

RUN_DIR=$(ls -td runs/bert-*/ | head -1)
uv run python -m src.evaluation.evaluate --submission "${RUN_DIR}submission.csv" | tee "${RUN_DIR}test_metrics.json"
