# Baseline(ai_dep FairLogitNet) 학습 방식 — 데이터·입력·구조·출력

비교 대상인 KAIST ai_dep `financial_credit_1`(UCI Adult, Acc 81.00% / DEO 0.069)이
어떻게 학습됐는지 정리. 코드: `Bias/external/ai_dep/financial_credit/financial_credit_1/`
(DEO 지표 문제는 [deo_github_vs_literature.md](deo_github_vs_literature.md) 참고)

## 1. 학습/평가 데이터 구분 — UCI 공식 고정 분할 그대로

리포의 `data/datasets/uci_adult/`를 원본과 대조한 결과:

| 파일 | 행 수 | 정체 |
|---|---|---|
| train.csv | 32,561 | **원본 `adult.data`와 행 단위 완전 동일** (공백 정리만 다름) |
| test.csv | 16,281 | **원본 `adult.test`** (라벨 뒤 `.` 제거, 수치 float 표기) |

- 직접 나눈 게 아니라 **UCI가 배포하는 공식 train/test 분할(2:1)** 을 그대로 사용.
  같은 분할을 쓰면 수치 직접 비교 가능.
- **validation set 없음**: train 전체로 학습하고 에폭마다 **test에서 acc/DEO를 재서
  기준(acc>0.81 & DEO<0.07)을 만족하는 에폭을 체크포인트로 저장** — test가 모델
  선택에 개입하는 test-set selection. 공정 비교 시 프로토콜 명시 필요.

## 2. 입력 전처리 (`fairness/dataloader.py`)

한 사람(행)에서 모델에 들어가는 값은 **7개뿐**:

- **범주형 1개: sex** — pandas category 코드 (알파벳순: Female=0, Male=1)
- **수치형 6개: age, fnlwgt, education-num, capital-gain, capital-loss,
  hours-per-week** — `mean_std.json`(train 통계)으로 z-정규화
- 나머지 범주형 7개(workclass, education, marital-status, occupation,
  relationship, race, native-country)는 `sensitive_column_names` 목록에 넣어
  **입력에서 통째로 제외**
- 라벨 y = (income == ">50K") → 1/0

## 3. 네트워크 구조 (`fairness/model.py`, LearnerNN)

```
sex 코드 ──> Embedding(3, 32) ──┐
                                ├─ concat → 38차원
수치 6개 ───────────────────────┘
   → Linear(38→64) → ReLU → Linear(64→32) → ReLU → Linear(32→1) → 로짓 1개
```

- forward는 (로짓, 시그모이드 확률, 0.5 반올림 예측) 반환. 파라미터 수천 개의 얕은 MLP.
- **Embedding이 (3, 32)인 이유**: `calculate_embedding`이 `len(vocab)+1`을 쓰기
  때문 — sex 어휘 2(Male/Female) + **결측 대비 'unk' 슬롯 1**. 3은 입력 차원이
  아니라 **32차원 벡터 3개짜리 룩업 테이블의 행 수**다. 실제로 sex는 train/test
  모두 결측 0건이라 코드 0/1만 조회되고 **3번째 행(코드 2)은 학습 내내 한 번도
  쓰이지 않는 죽은 파라미터**. (`inference_fairness.py`의 `[(3, 32)]` 하드코딩은
  이 계산 결과를 박은 것.)

## 4. 학습 방식 — 반사실(성별 뒤집기) 앙상블

손실을 단일 forward 로짓에 걸지 않고, **모든 샘플을 두 번 통과**시킨다
(한 번은 실제 성별, 한 번은 뒤집은 성별):

$$\text{final logit} = 1.52 \cdot f(\text{남으로 넣은 출력}) + 2.94 \cdot f(\text{여로 넣은 출력})$$

- 가중치가 "어느 성별을 입력했느냐"에만 붙으므로(남 입력=1.52, 여 입력=2.94),
  원래 샘플의 성별과 무관하게 **최종 로짓 식이 모든 샘플에서 동일** → 예측 함수가
  절차적으로 성별-불변(사실상 sex를 지운 것과 같음). 그런데도 DEO가 0이 아닌 이유는
  수치 피처(근로시간, 자본이득 등)가 성별과 상관되기 때문.
- **1.52/2.94의 출처 = 역빈도(IPS) 가중치 추정**: train에서 남 66.9%/여 33.1% →
  역수 1.49/3.02 ≈ 1.52/2.94 (리포 `IPS_example_weights_*.json`과 부합).
  소수 그룹(여성) 브랜치를 증폭하는 설계.
- **손실**: 앙상블 로짓에 `BCEWithLogitsLoss` (평범한 이진 분류). 공정성 항은
  손실에 없고, 공정성은 위 구조(대칭화 + IPS)로만 유도.
- **최적화**: Adagrad lr 0.01, batch 256, 50 epoch, seed 210.

## 5. 추론(출력)

학습과 동일한 반사실 앙상블 로짓 → 시그모이드 → 0.5 반올림 →
**1(고소득 >50K / 승인) 또는 0(저소득 / 거절)**.
리포의 accuracy/DEO 계산(`metrics_fairness.py`) 모두 이 앙상블 예측 기준.

## 6. 요약 및 우리 셋업과의 대비

**"sex + 수치 6개 → income 이진 분류"를 배우되, 성별을 항상 양쪽 다 넣어
역빈도 가중으로 섞음으로써 예측 함수 자체를 성별-불변으로 만든 얕은 MLP.**

- 공정성 개입 시점: 학습·추론의 **구조** (우리: 학습 후 weight-editing — 좋은 대비)
- 입력 형태: 표(tabular) 임베딩 (우리: 자연어 프롬프트 → XLM-R)
- 재현 주의: 사전학습 체크포인트 404, 동일 코드·시드 재학습으로 DEO 0.069 미재현
  (재측정: `Bias/result/ai_dep_deo/history.json`, 상세는 deo_github_vs_literature.md)
