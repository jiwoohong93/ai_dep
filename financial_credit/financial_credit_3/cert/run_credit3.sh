#!/bin/bash
# 공인시험(KGcT 2026) [금융/신용3] 정확도 / 재현율 / 응답시간 실행 스크립트
# 실행: bash cert/run_credit3.sh   (인자 없음)
# 규격서 시험방법 2~4번을 순서대로 수행한다.
set -o pipefail

echo "========================================================================"
echo "[금융/신용3] 정확도(≥0.95) / 재현율(≥0.95) / 응답시간(≤0.5초/건)"
echo "========================================================================"

echo "[단계 1] conda 환경 활성화 (rule_edit)"
source /data1/jinsil/miniconda3/etc/profile.d/conda.sh
conda activate rule_edit
echo "  python : $(which python)"

echo "[단계 2] 작업 디렉토리 이동 (이 스크립트 기준 상대경로 = financial_credit_3 루트)"
cd "$(dirname "$0")/.."
echo "  pwd    : $(pwd)"

echo "[단계 3] 시험 데이터 SHA-256 확인 (규격서 기댓값과 비교)"
TEST_CSV=Regulation/data/test/text/test.csv
EXPECTED=5e630e6a8dbb13a9299693b74d4aacf7295368e30d160fb73f8208621141c297
ACTUAL=$(sha256sum "$TEST_CSV" | cut -d' ' -f1)
echo "  파일   : $TEST_CSV"
echo "  기댓값 : $EXPECTED"
echo "  실측값 : $ACTUAL"
if [ "$ACTUAL" = "$EXPECTED" ]; then
    echo "  => SHA-256 일치 (OK)"
else
    echo "  => SHA-256 불일치 (MISMATCH) — 중단"
    exit 1
fi

echo "[단계 4] 공인시험 최종 명령어 실행 — [금융/신용3] 정확도 / 재현율 / 응답시간"
CUDA_VISIBLE_DEVICES=0 python cert/cert_credit3_regulation.py --model_dir Regulation/model/xlm-roberta-large-rule --shift_config Regulation/centroid/result/centroid_matched/edit_matched_v3b2/rule1_matched_shift_config.pt --latency_n 1000 --save_json cert/results/kgct_credit3.json
