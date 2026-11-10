# ALINCredit

**ALINCredit**은 **공개 신용 데이터셋 8종**을 하나의 **합집합 피처 공간**(106개 피처)으로 통합해 함께 학습하는
공정성 인지(fairness-aware) 신용평가 알고리즘입니다. MAF 알고리즘 형식
(`algorithms/inprocessing/` 클래스, `run()` → `(metrics_orig, metrics_transf)`,
`metrics/common_utils.compute_metrics`, AIF360 `aifData`)을 따릅니다.

- **기준 모델 (`metrics_orig`)**: MAF 예시 코드와 같은 `StandardScaler` + `LogisticRegression`.
  임계값은 검증 분할에서 balanced accuracy가 가장 높은 값입니다.
- **제안 모델 (`metrics_transf`)**: 성별을 입력에서 제외한 HistGradientBoosting + LightGBM 앙상블을
  reweighing 가중치로 학습하고, 검증 분할에서 DEO 제약 아래 정확도가 가장 높은 단일 임계값을 적용합니다.

## 결과 (credit_union, 시드 1–8 평균)

| 지표 | 목표 | 결과 |
|:--|:--:|:--:|
| 정확도 (데이터셋별 테스트 정확도의 평균) | ≥ 82 % | **83.72 ± 0.62 %** |
| DEO ×100 = \|`equal_opportunity_difference`\| × 100 | < 5 | **0.71 ± 0.30** |
| 정책 반영 하락률 | < 3 % | **0.04 %** |

`metrics/common_utils.compute_metrics` 출력 (전체 테스트 행 기준, 시드 평균):

| 지표 | LR 기준 모델 | 제안 모델 |
|:--|--:|--:|
| accuracy | 0.7190 | **0.8176** |
| Balanced accuracy | 0.7055 | 0.6329 |
| equal_opportunity_difference | 0.0131 | **0.0071** |
| average_odds_difference | 0.0088 | 0.0054 |
| statistical_parity_difference | 0.0261 | 0.0172 |
| disparate_impact | 1.0323 | 1.0184 |

데이터셋별 결과 (제안 모델):

| 데이터셋 | 테스트 정확도 (%) | DEO ×100 |
|:--|--:|--:|
| German | 77.00 | 3.57 |
| LendingClub | 83.00 | – |
| Taiwan | 82.47 | 0.91 |
| Prosper | 72.61 | – |
| HomeCredit | 92.16 | 0.06 |
| HMEQ | 90.98 | – |
| CreditApproval | 86.18 | – |
| Australian | 85.34 | – |

<sub>분할: 데이터셋마다 train/valid/test 70/15/15 (MAF 예시와 같은 비율, 시드별 무작위 분할).
시드별 전체 결과, 패키지 버전, 데이터 파일 SHA-256 해시는 <code>results/credit_union.json</code>에 있습니다.
기본 실행(<code>__main__</code>, seed 1)은 정확도 83.74 %, DEO 0.41, 하락률 0.00 %입니다.</sub>

### 지표 정의

- **정확도**: 데이터셋별 테스트 정확도의 평균(macro). `compute_metrics`의 `accuracy`는 전체 테스트 행 기준(micro)입니다.
- **DEO** (Difference of Equal Opportunity): `|TPR_남성 − TPR_여성| × 100`
  = `|equal_opportunity_difference| × 100`. 성별 정보가 있는 데이터셋(German, Taiwan, HomeCredit)의 테스트 행으로 계산합니다.
- **정책 반영 하락률**: `(acc_ref − acc_fair) / acc_ref × 100`. `acc_ref`는 같은 앙상블을 공정성 보정
  (reweighing, DEO 제약) 없이 학습했을 때의 정확도입니다.

## 방법

1. **데이터 통합 → aifData** (`load_credit_union`)
   각 데이터셋을 공통 컬럼(`amount`, `duration`, `age`)과 접두어가 붙은 고유 컬럼
   (`g_`, `lc_`, `tw_`, `pr_`, `hc_`, `hmeq_`, `ca_`, `aus_`)으로 변환합니다. z-score, 범주 인코딩,
   결측 대체용 중앙값은 데이터셋별 **학습 분할에서만 추정**합니다. 세 분할을 각각 하나의 `aifData`
   (`label_name='credit'`, `protected_attribute_names=['sex']`, 1 = 남성 = privileged)로 만듭니다.
   - 데이터셋에 없는 피처는 `-9999`(`MISSING`)로 저장합니다. `aifData`는 NaN을 허용하지 않기 때문이며,
     제안 모델은 이 값을 다시 NaN으로 바꿔 트리가 결측으로 처리합니다. LR 기준 모델은 0(학습 평균)으로 대체합니다.
   - 성별 속성이 없는 데이터셋의 행은 `sex = -1`로 두어, 정확도에는 포함하되 어느 그룹에도 속하지 않게 합니다.
2. **입력 피처**: 성별과 성별을 담고 있는 속성(German A9 "혼인 상태 및 성별")은 입력에서 제외합니다.
3. **모델**: scikit-learn `HistGradientBoostingClassifier`와 LightGBM `LGBMClassifier`의 소프트 보팅 앙상블.
4. **공정성 보정**
   - 학습: 데이터셋별로 (성별, 레이블) 조합마다 `w = P(성별)·P(레이블) / P(성별, 레이블)`을 주는
     reweighing(Kamiran & Calders, 2012).
   - 판정: 모든 신청자에게 하나의 임계값. 검증 분할에서 DEO ≤ 4(`deo_cap`)를 만족하는 임계값 중 정확도가 가장 높은 값을 고릅니다.
     성별은 판정 시점에 쓰지 않습니다.

## 실행 환경

- Python 3.13에서 검증(3.11 이상). Windows 11, CPU만 사용(GPU 불필요).
- 주요 패키지(`requirements.txt`에 고정): numpy 2.4.6, pandas 2.3.3, scikit-learn 1.9.0, lightgbm 4.6.0, aif360 0.6.1.
  aif360 0.6.1은 pandas 3.x에서 Adult 로더가 동작하지 않으므로 pandas 2.3.3을 사용합니다.

## 설치

```bash
git clone https://github.com/jhoonjwa/ALINCredit.git && cd ALINCredit
# 또는 배포 zip 사용 시: unzip ALINCredit.zip && cd ALINCredit

conda create -n alincredit python=3.13 -y
conda activate alincredit
# 또는: python -m venv .venv && source .venv/bin/activate   (Windows: .venv\Scripts\activate)

pip install -r requirements.txt
python download_data.py
```

`download_data.py`는 `./data`에 없는 파일을 내려받고, AIF360 벤치마크(German/Adult/COMPAS)
원본 파일을 설치된 aif360 패키지의 `aif360/data/raw/` 아래로 복사합니다. 수동으로 받아야 하는 파일은 없습니다.

## 데이터

- **zip 배포본**: 모든 데이터가 `./data`에 포함되어 있습니다.
- **GitHub 저장소**: 용량이 큰 Prosper(86 MB), Home Credit(57 MB)를 제외한 데이터가 포함되어 있으며,
  이 두 파일은 `python download_data.py`가 자동으로 내려받습니다.

각 데이터셋은 원 출처의 라이선스·이용 약관을 따릅니다.

| 데이터셋 | 사용 행 수 | 성별 | `data/` 내 파일 | 출처 |
|:--|--:|:--:|:--|:--|
| German Credit (Statlog) | 1,000 | ✓ | `german.data` | [UCI #144](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data) |
| LendingClub 2007–2011 | 9,004 | – | `lendingclub.csv` | LendingClub 공개 승인 대출 데이터(`LoanStats3a`)에서 상환 완료·부실 대출 9,004건을 추출, [이 저장소에 포함](https://github.com/jhoonjwa/ALINCredit/blob/main/data/lendingclub.csv) |
| Taiwan Credit Card Default | 30,000 | ✓ | `default of credit card clients.xls` | [UCI #350](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) |
| Prosper Loans | 약 55,000 | – | `prosperLoanData.csv` | [Udacity 미러](https://s3.amazonaws.com/udacity-hosted-downloads/ud651/prosperLoanData.csv) |
| Home Credit Default Risk | 40,000 (샘플링) | ✓ | `homecredit_application_train.parquet` | [Hugging Face 미러](https://huggingface.co/datasets/mohameddhameem/home-credit-default-risk) |
| HMEQ | 5,960 | – | `hmeq.csv` | [Credit Risk Analytics](http://www.creditriskanalytics.net/uploads/1/9/5/1/19511601/hmeq.csv) |
| Credit Approval | 690 | – | `creditapproval.csv` | [UCI #27](https://archive.ics.uci.edu/dataset/27/credit+approval) |
| Australian Credit (Statlog) | 690 | – | `australian.csv` | [UCI #143](https://archive.ics.uci.edu/dataset/143/statlog+australian+credit+approval) |
| (AIF360) Adult / COMPAS / German | – | – | `data/aif360/` | UCI Adult, ProPublica COMPAS, UCI German |

## 실행 명령어

```bash
# 공인시험 실행 (데이터 SHA-256 확인 -> 학습된 모델로 테스트 데이터 추론 -> 공정성 DEO / 정책 반영 성능 평균 하락률 판정)
# 모델: checkpoints/alincredit_seed1.joblib, 결과: results/test_result_item{1,2}.json
python run_test.py --item 1    # 시험항목 1. 공정성 DEO x100
python run_test.py --item 2    # 시험항목 2. 정책 반영 성능 평균 하락률
python run_test.py             # 두 항목 모두 (results/test_result.json)

# 모델을 다시 학습하여 checkpoints/ 에 저장한 뒤 평가
python run_test.py --train

# MAF 형식 기본 실행 (credit_union, seed 1): metrics_orig / metrics_transf / 요약 출력, 약 1–2분
python algorithms/inprocessing/ALINCredit.py

# 위 결과표 재현 (시드 1–8, 4개 병렬, 8코어 CPU에서 약 5분) -> results/credit_union.json
python evaluate.py --jobs 4

# AIF360 표준 벤치마크
python evaluate.py --dataset german --jobs 4
python evaluate.py --dataset adult  --jobs 4
python evaluate.py --dataset compas --jobs 4
```

코드에서 사용:

```python
from algorithms.inprocessing.ALINCredit import ALINCredit

model = ALINCredit(dataset_name='credit_union', protected='sex')  # 'german' | 'adult' | 'compas'
metrics_orig, metrics_transf = model.run()   # common_utils.compute_metrics 형식의 OrderedDict 두 개
print(model.summary)                         # 데이터셋별 정확도, macro 정확도, DEO x100, 정책 반영 하락률
```

| 인자 | 기본값 | 설명 |
|:--|:--|:--|
| `dataset_name` | `'credit_union'` | `'credit_union'`, `'german'`, `'adult'`, `'compas'` |
| `protected` | `'sex'` | 보호 속성 (adult/compas는 `'race'`도 가능) |
| `deo_cap` | `4.0` | 임계값 선택 시 검증 분할 DEO ×100 상한 |
| `seed` | `1` | train/valid/test 분할 시드 |
| `n_jobs` | `None` | LightGBM 스레드 수 (`None` = 전체 코어) |

aif360을 불러올 때 tensorflow, fairlearn 등 선택 의존성이 없다는 `WARNING:root:` 메시지가 출력되지만 실행에는 영향이 없습니다.

### AIF360 표준 벤치마크 결과 (시드 1–8 평균)

MAF 예시와 같은 `load_preproc_data_*` 데이터(보호 속성 `sex`)로 실행한 결과입니다.

| 데이터셋 | 모델 | accuracy | Balanced acc. | equal_opportunity_difference | disparate_impact |
|:--|:--|--:|--:|--:|--:|
| German | LR 기준 | 0.5867 | 0.6311 | −0.2029 | 0.6154 |
|  | 제안 | 0.6883 | 0.5037 | −0.0013 | 0.9806 |
| Adult | LR 기준 | 0.7365 | 0.7420 | −0.3599 | 0.2823 |
|  | 제안 | 0.7870 | 0.6600 | 0.0029 | 0.6908 |
| COMPAS | LR 기준 | 0.6569 | 0.6562 | −0.2196 | 0.6099 |
|  | 제안 | 0.6198 | 0.6079 | −0.0701 | 0.8408 |

- 이 알고리즘은 여러 데이터셋을 통합 학습하는 상황을 위해 설계되었습니다. 단일 데이터셋 벤치마크는
  MAF 형식 호환성 확인용입니다.
- German 전처리 데이터는 피처가 10개뿐이고 검증 분할이 150행이라, DEO 제약을 만족하는 임계값이 대부분의
  시드에서 거의 전원을 승인하는 값이 됩니다(Balanced accuracy ≈ 0.50).
- COMPAS는 검증 분할에서 DEO ≤ 4를 맞추려면 정확도를 크게 포기해야 하고, 검증 분할의 DEO가 테스트 분할에서
  그대로 유지되지 않아(테스트 DEO ×100 평균 7.01), 정확도가 LR보다 낮습니다.

## 프로젝트 구조

```
ALINCredit/
├── algorithms/
│   └── inprocessing/
│       ├── ALINCredit.py                 # MAF 알고리즘 클래스 (run() -> metrics_orig, metrics_transf)
│       └── alincredit_function/
│           ├── loaders.py                # 데이터셋별 로더 (원본 피처 + 컬럼 종류, 성별은 분리 반환)
│           ├── harmonize.py              # 학습 분할 기반 전처리, 합집합 피처 공간
│           └── model.py                  # HistGB + LightGBM 앙상블, reweighing, 임계값 선택
├── metrics/common_utils.py               # MAF 가이드의 compute_metrics (원본 그대로)
├── sample.py                             # aifData (MAF 가이드의 AIF360 StandardDataset 래퍼)
├── data/                                 # 신용 데이터 8종 + data/aif360/ (Adult, COMPAS, German 원본)
├── checkpoints/alincredit_seed1.joblib   # 학습된 모델 (seed 1, run_test.py가 사용)
├── results/                              # evaluate.py 결과 (credit_union / german / adult / compas .json)
├── download_data.py
├── evaluate.py
├── run_test.py                           # 공인시험용 실행 스크립트
├── requirements.txt
└── LICENSE
```

## 라이선스

코드는 MIT 라이선스입니다. 데이터는 각 출처의 라이선스를 따릅니다.
