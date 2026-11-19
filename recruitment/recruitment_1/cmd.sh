#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
PYTHON="${PYTHON:-python}"
MODE="${1:-evaluate}"
ARTIFACT_DIR="${2:-artifacts}"

case "$MODE" in
  evaluate)
    "$PYTHON" evaluate.py --artifact-dir "$ARTIFACT_DIR" \
      --reference results/metrics.json --check-targets \
      --output "$ARTIFACT_DIR/evaluation_metrics.json"
    ;;
  train)
    "$PYTHON" run_experiment.py --config configs/experiment.json \
      --out-dir "$ARTIFACT_DIR" --stage train
    ;;
  all)
    if [[ $# -lt 3 ]]; then
      printf 'Usage: bash cmd.sh all ARTIFACT_DIR PRIVATE_XLSX_DIR\n' >&2
      exit 2
    fi
    "$PYTHON" run_experiment.py --config configs/experiment.json \
      --out-dir "$ARTIFACT_DIR" --data-dir "$3" --stage all
    ;;
  *)
    printf 'Usage: bash cmd.sh {evaluate|train|all} ARTIFACT_DIR [PRIVATE_XLSX_DIR]\n' >&2
    exit 2
    ;;
esac
