#!/bin/bash
# 공인시험(KGcT 2026) [금융/신용1] 예측정확도 / 공정성(DEO) 실행 스크립트
# 실행: bash cert/run_credit1.sh   (인자 없음)
# 규격서 시험방법 2~4번을 순서대로 수행한다.
set -o pipefail

echo "========================================================================"
echo "[금융/신용1] 예측정확도(≥0.82) / 공정성 DEO=|TPR_남-TPR_여|(≤0.05)"
echo "========================================================================"

echo "[단계 1] conda 환경 활성화 (rule_edit)"
source /data1/jinsil/miniconda3/etc/profile.d/conda.sh
conda activate rule_edit
echo "  python : $(which python)"

echo "[단계 2] 작업 디렉토리 이동 (이 스크립트 기준 상대경로 = financial_credit_1 루트)"
cd "$(dirname "$0")/.."
echo "  pwd    : $(pwd)"

echo "[단계 3] 시험 데이터 SHA-256 확인 (규격서 기댓값과 비교)"
TEST_CSV=Bias/data/test/text/test.csv
EXPECTED=cfdb68c36ee5237eec7e369f6a22db72a70e05d5c55dd6b41045dfd5c5efdbad
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

echo "[단계 4] 공인시험 최종 명령어 실행 — [금융/신용1] 정확도 / DEO"
CUDA_VISIBLE_DEVICES=0 python cert/cert_credit1_bias.py --shift_config Bias/result/bias_centroid/edit/layer9_seed0/shift_config.pt --alpha 0.9 --save_json cert/results/kgct_credit1.json
