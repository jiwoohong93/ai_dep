# DEO — 깃허브(ai_dep) 구현과 문헌 정의의 차이

Bias 셋업의 목표 지표인 DEO(Difference of Equal Opportunity)에 대해,
비교 대상인 KAIST ai_dep 리포의 실제 구현과 문헌 정의가 **다른 것을 재고 있음**을 정리한다.

- 비교 대상: [jiwoohong93/ai_dep](https://github.com/jiwoohong93/ai_dep) → `financial_credit/financial_credit_1` (FairLogitNet, UCI Adult)
- 보고 성능: **Accuracy 81.00% / DEO 0.069** (baseline 81.76% / 0.138)
- 로컬 클론: `Bias/external/ai_dep/` (git 미추적)

## 기호 정의

Adult의 한 사람(한 행)마다:

| 기호 | 뜻 | 값 |
|---|---|---|
| **y** | 정답 라벨 = 실제 연소득이 5만 달러 초과인가 (`income`) | 1(초과) / 0(이하) |
| **ŷ** | 모델의 예측 | 1 / 0 |
| **s** | 성별 (sex, 보호 속성) — 라벨이 아니라 **사람들을 두 그룹으로 나누는 기준** | 남 / 여 |

## 깃허브 구현이 실제로 재는 것 — 그룹별 "전체 정확도"의 격차

`fairness/metrics_fairness.py`의 `compute_deo`
(경로: `Bias/external/ai_dep/financial_credit/financial_credit_1/fairness/metrics_fairness.py:29`)

- 남자 그룹 정확도 = 남자들 중 ŷ = y 인 비율
- 여자 그룹 정확도 = 여자들 중 ŷ = y 인 비율

$$\text{DEO}_{\text{repo}} = \Bigl| \tfrac{1}{N_{남}}\textstyle\sum_{i:\,s_i=남} \mathbb{1}[\hat{y}_i = y_i] \;-\; \tfrac{1}{N_{여}}\textstyle\sum_{i:\,s_i=여} \mathbb{1}[\hat{y}_i = y_i] \Bigr|$$

코드의 주석("DEO is the absolute difference in TPRs")과 변수명(`tpr_male`)은 TPR을
의도했지만, **`labels == 1` 조건 필터가 없어** 분자가 TP+TN(그룹 내 정답 전부),
분모가 그룹 전체 인원이다 — 즉 TPR이 아니라 그룹 정확도.

같은 리포의 German credit 구현(`financial_credit_2/eval.py:133-135`)은 `actual==1`
조건을 제대로 걸고 표준형을 계산한다 → Adult 쪽만 조건이 누락된 것으로 보인다.

## 문헌 정의 (Hardt et al. 2016 / Donini et al. 2018) — TPR의 격차

**y=1인 사람들만 골라서** 1로 예측받은 비율(TPR)을 그룹 간 비교:

$$\text{DEO}_{\text{문헌}} = \bigl|\, P(\hat{Y}{=}1 \mid Y{=}1,\, s{=}남) \;-\; P(\hat{Y}{=}1 \mid Y{=}1,\, s{=}여) \,\bigr|$$

차이는 딱 하나 — **"y=1인 사람만 보느냐(문헌), 전원을 보느냐(리포)"**.

## 뭐가 문제인가 — 숫자 예시

가상의 test셋 (Adult의 실제 비율과 비슷: 양성 비율 남 ~31% vs 여 ~11%):

|  | 인원 | 그중 y=1 (소득 높음) |
|---|---|---|
| 남 | 100명 | 30명 |
| 여 | 100명 | 10명 |

모델이 **여자는 전원 ŷ=0**(무조건 거절), 남자는 y=1인 30명 중 24명을 1로,
y=0인 70명 중 64명을 0으로 예측했다고 하자.

| 계산 | 여 | 남 | 격차 | 판정 |
|---|---|---|---|---|
| 리포 방식 (그룹 정확도) | 90/100 = 0.90 | (24+64)/100 = 0.88 | **0.02** | "거의 완전 공정" |
| 문헌 방식 (TPR) | 0/10 = 0.00 | 24/30 = 0.80 | **0.80** | "극심한 차별" |

같은 모델을 두고 정반대 판정이 나온다. 이유: 여자 그룹은 y=0이 90%라서
**"여자는 다 거절"이라는 차별적 전략이 정확도 지표에서는 오히려 고득점**이기 때문.
그룹 정확도 = TPR·P(y=1|s) + TNR·P(y=0|s) 라서, 그룹별 양성 비율이 다르면
TPR 격차와 TNR 격차가 상쇄되어 "자격 있는 사람이 성별 때문에 거절당하는가"가
희석된다. 문헌 DEO는 처음부터 y=1인 사람만 보므로 이것이 그대로 드러난다.

추가 문제 — **비교 가능성**: Donini et al. 등 문헌이 Adult에서 보고하는 DEO는
TPR 격차이므로, 리포의 0.069를 문헌 SOTA 표에 그대로 넣으면 다른 수를 재는 셈이 된다.

## 참고: 81.00% / 0.069를 만든 예측 규칙

리포 모델(FairLogitNet)의 입력은 **sex(유일한 범주형) + 수치형 6개**
(나머지 범주형 7개는 입력에서 제외). 예측은 사실/반사실(성별 뒤집기) 로짓의
고정가중 앙상블:

$$\hat{y} = \text{round}\Big(\sigma\big(\, w(s)\, f(s, x) + w(1{-}s)\, f(1{-}s, x) \,\big)\Big), \quad w(\text{남})=1.52,\; w(\text{여})=2.94$$

- Accuracy = test 16,281행 전체의 1[ŷ=y] 평균 = 0.8100
- DEO_repo(위 수식) = 0.069
- sex는 라벨이 아님 — 라벨은 income(`dataset_stats.json`의 `target_column_name`)

## 재측정 결과 (2026-07-16, `Bias/src/eval_ai_dep_deo.py`)

원 리포의 사전학습 체크포인트(Google Drive)는 **404(삭제/비공개)** 라 공개 모델
자체는 잴 수 없었고, `train_fairness.py`와 동일 설정(seed 210, Adagrad lr 0.01,
batch 256, 50 epoch, 반사실 앙상블)으로 재학습해 매 에폭 세 지표를 측정했다
(전체 로그: `Bias/result/ai_dep_deo/history.json`).

| 지표 | 재학습 결과 (epoch 2~50 안정 구간) | 리포 보고값 |
|---|---|---|
| Accuracy | 0.821 ~ 0.827 | 0.8100 |
| DEO_repo (그룹 정확도 격차) | 0.086 ~ 0.122 | **0.069 — 미재현** |
| DEO_문헌 (TPR 격차) | **0.015 ~ 0.040** | (보고 없음) |
| TPR (남 / 여) | ~0.38–0.52 / 남보다 약간 낮음 | — |

관찰:

1. **리포의 저장 기준(acc>0.81 & DEO_repo<0.07)은 50 에폭 동안 한 번도 달성되지
   않았다** — 보고된 0.069는 이 코드·시드 그대로는 재현 안 됨 (GPU/셔플
   비결정성 또는 코드 변경 이력 추정. `train_fairness.py:39`에 `breakpoint()`가
   남아있는 것도 공개 코드가 실제 실험 코드와 어긋난 방증).
2. 이 모델에서는 **문헌 DEO(0.02~0.04)가 리포 DEO(0.09~0.12)보다 오히려 훨씬 작다**
   — TPR 자체는 남녀 비슷하게 낮고(~0.4대, 진짜 고소득자의 절반 이상을 놓침),
   그룹 정확도 격차는 주로 양성 비율 차이에서 나온다. 두 지표가 다른 것을 잰다는
   본문 논지의 실측 확인.
3. 체크포인트 선택이 **test 지표 기준**(acc>0.81 & DEO<0.07을 만족하는 에폭 저장)
   이라는 점도 주의 — test-set selection이므로 공정한 비교에서는 재현 프로토콜을
   명시해야 한다.

## 우리 셋업의 실무 결론

1. **리포 수치와 직접 비교**할 때는 리포 구현 그대로(그룹 정확도 격차)로 잰다.
2. **보고/논문**에는 표준형(TPR 격차) DEO를 병기한다 — predictions만 있으면 둘 다 한 줄 차이.
3. 전처리 재현 자료는 `Bias/external/ai_dep/financial_credit/financial_credit_1/data/datasets/uci_adult/`
   (train/test.csv, vocabulary.json, mean_std.json, IPS 가중치 json)를 그대로 쓸 수 있다.
