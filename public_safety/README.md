# Regional IPM + Bayesian PO Intervention Pipeline

전체 흐름은 다음 네 단계입니다.

0. `data_preparation.py`: 위도/경도로 가까운 이웃을 찾고 own covariate interaction feature를 생성
1. `get_IPM_weight.py`: prepared data로 IPM/SIPM balancing weight 추정
2. `po_estimate.py`: IPM weight를 사용해서 Bayesian PO 모델 학습
3. `intervention_region.py`: Gurobi aux variable로 지역 간섭이 있는 intervention 배정 최적화

## Part 1. Generic Pipeline

이 파트는 특정 데이터셋에 묶이지 않는 범용 사용법입니다. 입력 데이터가 아래 column naming rule을 따르면 같은 코드 흐름을 그대로 사용할 수 있습니다.

### Raw Data Format

기본 입력 파일은 `data.csv`입니다. 다른 파일을 쓰려면 `--data`로 경로를 넘기면 됩니다.

필수 column 이름 규칙은 다음과 같습니다.

| Column pattern | 의미 |
|---|---|
| `SA01`, `SA02`, ... | sensitive attribute columns |
| `X01`, `X02`, ... | treatment 이전 설명변수 columns |
| `treat` | binary observed treatment |
| `objective` | observed outcome/objective value |
| `long` | longitude |
| `lat` | latitude |

예시:

```csv
SA01,SA02,SA03,X01,X02,treat,objective,long,lat
0.2,0.7,0.1,1.0,5.2,1,0.34,-73.95,40.65
0.8,0.1,0.1,0.0,3.0,0,0.12,-73.88,40.85
```

`SA` columns는 fraction 또는 percent 둘 다 가능합니다. 값이 `1`보다 크면 `data_preparation.py`에서 자동으로 `100`으로 나눕니다.

`treat`는 `0/1` 또는 `Yes/No`, `True/False` 스타일 값을 지원합니다.

`objective`는 기존 school 예시의 `Total_SAT_ACT / Total_students`처럼 이미 계산된 하나의 numeric outcome이어야 합니다.

### Step 0: Prepare Regional Data

```bash
python data_preparation.py --data data.csv --sim 5
```

기본 산출물:

| File | 의미 |
|---|---|
| `data_new.csv` | 이웃, similarity, own covariate interaction feature가 추가된 데이터 |
| `data_new_meta.json` | 생성된 column 목록과 설정 |

`--sim 5`는 자기 자신을 제외하고 가까운 이웃 5개를 사용한다는 뜻입니다. 최종 local neighborhood는 자기 자신 포함 총 `sim + 1`개입니다.

`data_new.csv` 주요 columns:

| Column | 의미 |
|---|---|
| `unit_id` | 원래 row index |
| `objective` | outcome |
| `treat` | observed treatment |
| `long`, `lat` | 좌표 |
| `SA01...SAk` | sensitive attribute fractions |
| `X01...Xm` | own non-treatment covariates |
| `TEXP` | observed own treatment, training feature용 |
| `XF_*` | `X * SA` interaction features |
| `TF_*` | `treat * SA` observed treatment interaction features |
| `neigh0...neigh{sim}` | 자기 자신과 이웃 row indices |
| `sim0...sim{sim}` | 해당 이웃들의 similarity |

현재 similarity는 위도/경도 haversine distance를 사용해서 다음처럼 계산합니다.

```text
similarity = 1 / (1 + distance_km)
```

자기 자신은 항상 `neigh0`, `sim0 = 1`입니다.

### Step 1: Estimate IPM Weights

```bash
python get_IPM_weight.py --data data_new.csv
```

기본 산출물:

| File | 의미 |
|---|---|
| `w_IPM_region.npy` | 각 row의 regional IPM/SIPM balancing weight |
| `w_IPM_region_meta.json` | treated/control 수, 사용 covariate, weight 요약 |

weight 학습에 쓰는 covariates:

```text
[X * SA interaction columns] + [SA columns]
```

즉 treatment 관련 `TF_*`는 balancing covariate에 넣지 않습니다.

주요 옵션:

```bash
python get_IPM_weight.py \
  --data data_new.csv \
  --out w_IPM_region.npy \
  --epochs 1000 \
  --n-critics 10 \
  --device auto
```

### Step 2: Estimate Bayesian PO Model

```bash
python po_estimate.py \
  --data data_new.csv \
  --weights w_IPM_region.npy
```

기본 산출물:

| File | 의미 |
|---|---|
| `po_bnn_region.npz` | Bayesian PO coefficient posterior |
| `po_bnn_region_meta.json` | 데이터/모델 shape와 column 정보 |

PO 모델에 쓰는 features:

```text
[XF_* columns] + [TF_* columns] + [SA columns]
```

즉 모델은 대략 다음 형태입니다.

```text
objective_i
= f(own non-treatment covariates_i, sensitive_i)
+ f(observed own treatment_i, sensitive_i)
+ f(sensitive_i)
```

`po_bnn_region.npz` 주요 keys:

| Key | Shape | 의미 |
|---|---:|---|
| `w` | `(1, d)` | posterior mean coefficient |
| `w_samples` | `(n_samples, 1, d)` | posterior coefficient samples |
| `feature_columns` | `(d,)` | 실제 학습에 사용된 feature 순서 |
| `sensitive_columns` | `(k,)` | `SA` columns |
| `x_columns` | `(m,)` | `X01...Xm` columns |
| `xf_columns` | | `XF_*` columns |
| `tf_columns` | | `TF_*` columns |


```bash
python po_estimate.py --data data_new.csv --weights w_IPM_region.npy --burn-in 1000 --n-samples 100 --thin 50
```

### Step 3: Optimize Regional Intervention

```bash
python intervention_region.py \
  --data data_new.csv \
  --model po_bnn_region.npz \
  --k 25 \
  --taus 0.04,0.06,0.08
```

이 단계는 Gurobi가 필요합니다.

기본 산출물:

| File | 의미 |
|---|---|
| `results/obj_val_region.csv` | tau별 최적화 요약표 |
| `results/obj_val_region_p05.csv` | `--constraint-mode quantile --q 0.05`일 때의 tau별 최적화 요약표 |
| `results/region_ipm_*.npz` | tau별 선택 벡터와 posterior 평가값 |
| `results/region_ipm_*.lp` | `--write-lp`를 켰을 때 저장되는 Gurobi LP 파일 |

최적화 decision variable:

```text
z_i = 1 if unit i receives intervention, else 0
```

budget constraint:

```text
sum_i z_i <= K
```

학습 단계에서는 일반 설명변수와 observed treatment를 own value로만 사용합니다. 하지만 intervention 최적화 단계에서는 treatment만 주변 intervention pattern을 봅니다. 각 row `i`는 `neigh0...neigh{sim}`에 해당하는 주변 intervention pattern을 보고:

```text
treatment_exposure_i = max_j similarity(i, neighbor_j) * z_neighbor_j
```

를 사용합니다.

그래서 `intervention_region.py`는 기존 school 코드처럼 `aux` variable을 만듭니다.

```text
aux_i[f] = row i 주변 local intervention pattern f가 선택되었는지
```

`sim=5`이면 자기 자신 포함 6개 위치가 있으므로 row마다 가능한 local pattern은:

```text
2^(sim + 1) = 64
```

개입니다.

### Fairness Constraint

각 row `i`와 counterfactual sensitive group `r`에 대해 제약을 겁니다.

`mean` mode:

```text
E[PO_i(actual SA_i, local intervention pattern)]
- E[PO_i(counterfactual SA = one-hot r, same local intervention pattern)]
<= tau
```

`quantile` mode:

```text
Q_{1-q}[
  PO_i(actual SA_i, local intervention pattern)
  - PO_i(counterfactual SA = one-hot r, same local intervention pattern)
]
<= tau
```

예를 들어:

```bash
python intervention_region.py \
  --data data_new.csv \
  --model po_bnn_region.npz \
  --k 25 \
  --taus 0.1,0.2,0.5 \
  --constraint-mode quantile \
  --q 0.05
```

이면:

```text
95% quantile(actual PO - counterfactual PO) <= tau
```

를 사용합니다.

Counterfactual sensitive group `r`은 `SA01`, `SA02`, ... 각각의 one-hot 상태입니다.

### Command Examples

전체 실행:

```bash
python data_preparation.py --data data.csv --sim 5
python get_IPM_weight.py --data data_new.csv
python po_estimate.py
python intervention_region.py --taus 0.0398
```

기존 `TAUS_frac.npz`처럼 `TAUS` key가 있는 파일을 쓰는 경우:

```bash
python intervention_region.py \
  --data data_new.csv \
  --model po_bnn_region.npz \
  --k 25 \
  --tau-file bnns/data/TAUS_frac.npz
```

Gurobi LP 파일까지 저장:

```bash
python intervention_region.py \
  --data data_new.csv \
  --model po_bnn_region.npz \
  --k 25 \
  --taus 0.5 \
  --write-lp
```

### School Data Mapping

`bnns/data/school_data.csv`를 이용해 검증할 때는 다음처럼 매핑했습니다.

| Regional column | Source column |
|---|---|
| `SA01` | `Black` |
| `SA02` | `Hispanic` |
| `SA03` | `White` |
| `X01` | `AP and/or IB`, Yes/No -> 1/0 |
| `X02` | `FTE School Counselors` |
| `treat` | `Calculus`, Yes/No -> 1/0 |
| `objective` | `Total_SAT_ACT / Total_students` |
| `long` | `long` |
| `lat` | `lat` |

`Other`는 sensitive attribute에서 제외했습니다.

### `obj_val_region.csv` Columns

`results/obj_val_region.csv`는 `intervention_region.py`가 tau별 최적화 결과를 한 줄씩 요약한 파일입니다. 한 row가 하나의 `tau` 실행 결과입니다.

| Column | 의미 |
|---|---|
| `tau` | fairness constraint의 허용 상한값. 코드상 `actual PO - counterfactual PO <= tau`에 들어가는 값입니다. |
| `status` | Gurobi 최적화 상태. `OPTIMAL`이면 제약을 만족하는 최적해를 찾았다는 뜻입니다. |
| `selected_count` | intervention을 배정받은 row 수. 즉 `sol_i = 1`인 개수입니다. |
| `objective_mean_model` | 최적화할 때 사용한 objective 값. posterior mean coefficient `w` 기준 전체 predicted PO 합입니다. |
| `obj_mean_posterior` | 선택된 해를 posterior samples `w_samples`로 다시 평가했을 때 objective의 평균입니다. |
| `obj_std_posterior` | posterior sample별 objective 값의 표준편차입니다. |
| `max_gap_mean_posterior` | posterior sample마다 `actual PO - counterfactual PO`의 최대값을 계산한 뒤, 그 최대 gap들의 평균입니다. |
| `max_gap_std_posterior` | 위 `max_gap` posterior 평가값의 표준편차입니다. |
| `result_file` | 해당 tau 결과의 상세 `.npz` 파일 경로입니다. |
| `selected_SA01` | intervention으로 선택된 row들의 `SA01` fraction 합입니다. school 예시에서는 Black 비율 합입니다. |
| `selected_SA02` | 선택된 row들의 `SA02` fraction 합입니다. school 예시에서는 Hispanic 비율 합입니다. |
| `selected_SA03` | 선택된 row들의 `SA03` fraction 합입니다. school 예시에서는 White 비율 합입니다. |

`constraint-mode mean`으로 돌린 경우 최적화 제약은 posterior mean model 기준으로 걸립니다. 따라서 `max_gap_mean_posterior`는 사후 샘플로 다시 평가한 diagnostic 값이고, 반드시 `tau`보다 작아야 하는 값은 아닙니다.

또한 `selected_SA01 + selected_SA02 + selected_SA03`이 꼭 `selected_count`와 같지는 않습니다. 이 예시에서는 `Other`를 sensitive attribute에서 제외했기 때문에 각 학교의 `SA01 + SA02 + SA03` 합이 1보다 작을 수 있습니다.

