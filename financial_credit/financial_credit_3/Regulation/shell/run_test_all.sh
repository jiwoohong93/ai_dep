#!/usr/bin/env bash
# 기존 test.csv + 편집 rule 9개 케이스 전체 평가
# 사용법: bash shell/run_test_all.sh [모델명(기본: yongukpark/xlm-roberta-large-rule)]
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${1:-yongukpark/xlm-roberta-large-rule}"

echo "=== base: data/test/text/test.csv ==="
python src/models/test.py \
    --test_path data/test/text/test.csv \
    --output_dir result/base \
    --model_name "$MODEL"

for i in 1 2 3 4 5 6 7 8 9; do
    echo "=== case ${i}: data/test/text/edited_rule_25pct/test_case${i}.csv ==="
    python src/models/test.py \
        --test_path "data/test/text/edited_rule_25pct/test_case${i}.csv" \
        --output_dir "result/edited_rule_25pct/case${i}" \
        --model_name "$MODEL"
done
