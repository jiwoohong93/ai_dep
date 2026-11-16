#!/usr/bin/env bash
# Adult 프롬프트 분류기 학습 (GPU 3 고정)
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-python}
GPU=${GPU:-3}

$PY src/dataset/generate_text_csv.py
CUDA_VISIBLE_DEVICES=$GPU $PY src/trainer/train.py "$@"
