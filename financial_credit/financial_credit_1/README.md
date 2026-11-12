# Credit Scoring with FairLogitNet V2.0 (금융/신용1)

UCI Adult 데이터셋 기반 신용평가 공정성 모델. XLM-RoBERTa-large를 규칙 학습한 뒤
**비혼 페널티 완화 편집(gated centroid shift, layer 9, alpha=0.9)** 으로 예측정확도와
공정성(DEO)을 동시에 달성한다.

| 지표 | V2.0 (본 제출) | V1.0 (FairLogitNet) | 목표 |
|---|---|---|---|
| 예측정확도 (Accuracy) | **0.871** | 0.810 | ≥ 0.82 |
| 공정성 DEO = \|TPR_남 − TPR_여\| | **0.042** | 0.069 | ≤ 0.05 |

- DEO는 Hardt et al. 2016 Equal Opportunity 정의(TPR 격차, y=1 조건). 상세·근거 수치는
  `cert/results/env.md`, `cert/results/REVIEW_GUIDE.md`, `cert/results/alpha_selection.md`.

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
| 편집 대상 모델 cgmix_rel (XLM-R-large, 2.2GB) | `Bias/model/finetuned_xlm-roberta-large_cgmix_rel/best/` | 【링크】 |
| 시험·학습 데이터 CSV (UCI Adult: test.csv/train.csv 등) | `Bias/data/` | 【링크】 |

압축을 풀어 위 경로에 그대로 두면 된다. (편집 설정 `shift_config.pt`는 레포에 포함되어 있음.)

## 공인시험 실행 (financial_credit_1 루트에서, 인자 없음)

```
bash cert/run_credit1.sh
```
스크립트는 ① conda 활성화 → ② 폴더 루트로 이동(상대경로) → ③ 시험 데이터 SHA-256 비교 →
④ 아래 최종 명령을 실행한다:

```
CUDA_VISIBLE_DEVICES=0 python cert/cert_credit1_bias.py \
    --shift_config Bias/result/bias_centroid/edit/layer9_seed0/shift_config.pt \
    --alpha 0.9 --save_json cert/results/kgct_credit1.json
```

**기대 결과**: Accuracy ≈ **0.871** (≥0.82), DEO ≈ **0.042** (≤0.05).

### 편집 설정(seed) 안내
- 공인시험은 **`layer9_seed0`** (`Bias/result/bias_centroid/edit/layer9_seed0/shift_config.pt`) 만 사용한다.
- `seed1`~`seed3`은 **블라인드 alpha 산출 근거(재현성·분산 확인)용**이며 시험 판정에는 쓰지 않는다.
  산출 과정은 `cert/results/alpha_selection.md` 참조.

## 심사 가이드
- 심사 3기준(데이터 입출력 / 모델 로딩·편집 / 메트릭 수식)을 `파일:줄번호`로 추적:
  `cert/results/REVIEW_GUIDE.md`.
