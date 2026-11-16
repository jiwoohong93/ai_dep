#!/usr/bin/env bash
# Adult 프롬프트 분류기 최종 평가 (test.csv 1회, GPU 3 고정)
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-python}
GPU=${GPU:-3}

CUDA_VISIBLE_DEVICES=$GPU $PY src/models/test.py "$@"
