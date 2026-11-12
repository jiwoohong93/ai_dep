# FinQA-RoBERTa V2.0 (금융/신용3)

Regulation 대출심사 모델(`xlm-roberta-large-rule`)에 **정책 반영 편집**(rule1 부채비율 기준
200% → 85% 강화, centroid v3b2 matched, gated shift, layer 10)을 적용해 정확도·재현율·응답시간
목표를 달성한다.

| 지표 | V2.0 (본 제출) | 목표 |
|---|---|---|
| 정확도 (Accuracy) | **0.985** | ≥ 0.95 |
| 재현율 (통과=1 / 탈락=0) | **0.997 / 0.978** | ≥ 0.95 |
| 응답시간 (초/건) | **≈ 0.039** | ≤ 0.5 |

- 기존 V1.0(BERT_GBSQA 기반 finqa-roberta_v1.0)은 `2024_FinQA-RoBERTa_V1.0/` 에 보관.
- 상세·근거: `cert/results/env.md`, `cert/results/REVIEW_GUIDE.md`.

## 환경 설정

```
conda create -n rule_edit python=3.10
conda activate rule_edit
pip install -r requirements.txt
```
(실측 환경·패키지·SHA-256: `cert/results/env.md`)

## 다운로드 (git 미포함 대용량 — Drive에서 받아 아래 위치에 배치)

| 항목 | 배치 위치 | 링크 |
|---|---|---|
| 대출심사 모델 xlm-roberta-large-rule (2.2GB) | `Regulation/model/xlm-roberta-large-rule/` | 【링크】 (HF: `yongukpark/xlm-roberta-large-rule` 로 대체 가능) |
| 시험 데이터 CSV (은행×기업 test.csv 등) | `Regulation/data/` | 【링크】 |
| (선택) locate 캐시 cache_A/B.pt | `Regulation/centroid/result/centroid_matched/locate_synth_v3b2/` | 【링크】 — shift_config **재생성 시에만** 필요(시험 실행엔 불필요) |

편집 설정 `rule1_matched_shift_config.pt`는 레포에 포함되어 있다.

## 공인시험 실행 (financial_credit_3 루트에서, 인자 없음)

```
bash cert/run_credit3.sh
```
스크립트는 ① conda 활성화 → ② 폴더 루트로 이동(상대경로) → ③ 시험 데이터 SHA-256 비교 →
④ 아래 최종 명령을 실행한다:

```
CUDA_VISIBLE_DEVICES=0 python cert/cert_credit3_regulation.py \
    --model_dir Regulation/model/xlm-roberta-large-rule \
    --shift_config Regulation/centroid/result/centroid_matched/edit_matched_v3b2/rule1_matched_shift_config.pt \
    --latency_n 1000 --save_json cert/results/kgct_credit3.json
```

- `--model_dir` 은 **상대경로**이다. Drive에서 받은 모델을 `Regulation/model/xlm-roberta-large-rule/`
  에 두거나, HuggingFace `yongukpark/xlm-roberta-large-rule` 로 바꿔도 된다.
- **기대 결과**: Accuracy ≈ **0.985** (≥0.95), 재현율 통과 ≈0.997 / 탈락 ≈0.978,
  응답시간 평균 ≈ **0.039초/건** (≤0.5).

## 심사 가이드
- 심사 3기준(데이터 입출력 / 모델 로딩·편집 / 메트릭 수식)을 `파일:줄번호`로 추적:
  `cert/results/REVIEW_GUIDE.md`.
