#!/usr/bin/env bash
# ALINJob V2.0 — 최종학력 분류 + 정책 반영(소수 클래스 선택 가중) 전체 재현 스크립트
#   (데이터 생성 → 학습 → 학습 모델 Hash 확인 → 평가, 공인시험 규격서 시험방법 6~9와 동일)
#   conda env: ai_dep (Python 3.10.20, requirements.txt)
#   run:       bash cmd.sh
set -e
cd "$(dirname "$0")"

# 0. 데이터 생성 (1회): scout_data/스카우트_이력사항DB_20221222.xlsx -> data_stage1.csv
#    data_stage1.csv가 이미 있으면 생략 (비공개 데이터)
[ -f data_stage1.csv ] || python preprocess.py

# 1. 시험 데이터·코드 Hash 확인
ls -l data_stage1.csv
sha256sum -c SHA256SUMS

# 2. 학습: 정책 미반영(none) / 정책 반영(minority, floor=250) 모델 -> models/
#    단일 스레드 결정적 학습이므로 학습 모델 파일이 매번 byte 단위로 동일함
rm -f models/trained_edu_none.pkl models/trained_edu_minority.pkl
python train_edu_gbm.py --data data_stage1.csv --fairness none
python train_edu_gbm.py --data data_stage1.csv

# 3. 학습 모델 Hash 확인
sha256sum -c MODEL_SHA256SUMS

# 4. 평가: 5-fold OOF 정확도·DEO·정책 반영 성능 평균 하락률
#    예상 결과: fair acc 0.8598 / unfair acc 0.8604 / DEO 0.0136 / 하락률 0.060 %
python test_edu_gbm.py --data data_stage1.csv --skip-point

# (선택) 저장 모델의 고정 10 % 분할 평가까지 출력
# python test_edu_gbm.py --data data_stage1.csv
