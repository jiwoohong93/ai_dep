#!/usr/bin/env bash
# 편집 베이스라인 본 실행: 9 케이스 × 4 방법, 확정 단일 설정
# (random_pass 문맥, rule별 피크층, R=64, λ=4) → result/edit_main/
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-python}
OUT=${OUT:-result/edit_main}

run_cases () {
    local gpu=$1; shift
    for c in "$@"; do
        for m in rome memit emmet alphaedit; do
            echo "=== GPU ${gpu} | case ${c} | ${m} ==="
            CUDA_VISIBLE_DEVICES=$gpu $PY src/editing/run_edit.py \
                --case "$c" --method "$m" \
                --context_mode random_pass --lam 4 --R 64 \
                --output_root "$OUT"
        done
    done
}

run_cases 0 1 2 3 &
run_cases 1 4 5 6 &
run_cases 2 7 8 9 &
wait
echo "본 실행 완료: ${OUT}/{method}/case{1..9}_*/"
