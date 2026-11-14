#!/usr/bin/env bash
# 9개 rule locate를 GPU 0/1/2에 3개씩 분산 실행
# 사용법: bash shell/run_locate_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-python}

run_rules () {
    local gpu=$1; shift
    for r in "$@"; do
        echo "=== GPU ${gpu} | rule ${r} ==="
        CUDA_VISIBLE_DEVICES=$gpu $PY src/editing/locate.py --rule "$r"
    done
}

run_rules 0 1 2 3 &
run_rules 1 4 5 6 &
run_rules 2 7 8 9 &
wait
echo "locate 완료: result/locate/rule{1..9}/n{1000..10000}/"
