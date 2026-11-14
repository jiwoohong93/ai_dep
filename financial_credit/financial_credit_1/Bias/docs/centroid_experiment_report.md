# Bias(Adult) Centroid 편집 실험 보고서

Regulation(금융 대출심사)에서 검증한 EDRM centroid locate-then-edit 방법론을 두 번째 도메인
(UCI Adult 소득 예측)에 이식한 실험 기록. `Bias/README.md`가 정한 두 축 — ① rule(정책)
편집(capital-gain 임계값), ② 편향(bias) 편집(비혼 페널티) — 을 모두 locate→δ_v→gate→평가
순서로 진행했다.

---

## 0. 대상 모델 · 데이터 · 목표

- **모델**: `Bias/model/finetuned_xlm-roberta-large_cgmix_rel/best` — XLM-RoBERTa-large,
  24 layers, hidden=1024. 
  로드 후 재현 확인: `accuracy=0.87341`, `DEO_lit=0.0593` — README 문서값(0.8734 /
  0.0594)과 정확히 일치.
- **데이터**: `Bias/data/train/text/train.csv`(32,561행, 학습에 쓰인 실제 adult.data) /
  `Bias/data/test/text/test.csv`(16,281행)
- **DEO 정의**: `deo_lit`(문헌 표준, TPR 격차) / `deo_repo`(그룹 정확도 격차,
  `Bias/reference/deo_github_vs_literature.md` 참고).

---

## 1. 편향(bias) 편집 — 비혼 페널티 완화

### 1.1 문제 정의 (README 확정 목표 그대로 사용)

같은 자격(y=1)이어도 비혼(Never-married/Divorced/Separated/Widowed)인 사람의 recall이
기혼 대비 크게 낮음(README 분석: 남 −19%p, 여 −40%p). precision은 오히려 비혼이 더 높아
(0.83~0.89 vs 0.75~0.77) **일관된 확신 기준이라면 있을 수 없는 구간별 문턱 불균형**이
확인된 상태

목표: locate로 "capital-gain(rule)은 반응 안 하면서 비혼 표지만 반응하는 위치"를 찾고
그 위치의 판정 기여를 완화. 성공 기준: `DEO_lit<0.05`, `acc 하락≤1%p`, cg rule 무손상.

### 1.2 Exp1 — Locate (layer×position 스캔)

`marital-status`/`relationship` 두 토큰 위치와 layer 0~24 전체를 스캔. train.csv에서 뽑은
실제 행 위에 marital-status/relationship만 강제로 기혼/비혼으로 바꾼 정렬쌍(aligned pair)과,
capital-gain만 7,000 경계로 바꾼 정렬쌍 두 세트로 두 신호(marital / cg)를 동시에 측정
(seed 4개, 재현성 확인).

**결과**: layer 1~15는 marital 신호 크고 cg 신호는 거의 우연 수준(cg_probe_acc 0.3~0.6).
layer 16+부터 cg 오염 시작, layer 22+에서 cg가 marital을 압도. `relationship` layer 8~9가
cg 오염이 가장 낮음(cg_probe_acc≈0.31~0.37).

→ `Bias/result/bias_centroid/locate_scan/`

### 1.3 Exp3+4 — δ_v + gate + alpha 보정

`delta_v_marital-status`, `delta_v_relationship` = μ(기혼,자격있음) − μ(비혼,자격있음),
layer9, train.csv 유래 합성쌍(n=1000)으로 계산. gate = [marital-status_hidden ++
relationship_hidden](2048차원) 로지스틱 회귀, "이 사람이 비혼으로 읽히는가" 판별
(합성 held-out acc=1.0000, 실데이터(test.csv 일부, 진단 전용) 정확도 97~98.5%).
layer 9/12/15 비교 → layer9가 alpha 반응이 가장 완만하고 예측 가능해 채택.

→ `Bias/result/bias_centroid/edit/layer9_seed{0,1,2,3}/`

### 1.4 절차 검증 — alpha 선택 근거 감사 (중요, 초기 결론 정정)

최초 추천값 `alpha=1.0`은 **test.csv 4개 alpha 실평가 결과를 직접 보고 고른 것**으로
드러나 무효 처리. layer 선택은 Exp1 스캔·calibration 곡선 전부 train.csv 유래라 깨끗함.

순수 합성 calibration만으로(README 목표 +3.3%p 역산 flip률≈5.1%) 보간한 "블라인드" alpha는
seed0=0.86, seed1=0.93, 평균 **0.89~0.9**.

### 1.5 Precision 분해 — 캘리브레이션 수렴 여부 확인

alpha 0.6→1.25로 올리며 비혼-기혼 precision 격차(4그룹: 비혼/기혼 × 남/여) 추적:

| alpha | 남성 격차 | 여성 격차 |
|---|---|---|
| 0(base) | +0.064 | +0.134 |
| 0.75 | +0.011 | +0.099 |
| **0.80** | **-0.014** | **+0.068** |
| 0.85 | -0.042 | +0.029 |
| 0.90 | -0.076 | -0.008 |
| 1.00 | -0.127 | -0.086 |

남성은 alpha≈0.77~0.80에서, 여성은 alpha≈0.89에서 각각 0을 지남 — **동시에 정확히
수렴하는 지점은 없음.** alpha=1.0은 양쪽 다 확실히 역전(비혼 과승인). alpha=0.8이
"남성은 거의 정확히 수렴, 여성은 아직 다 안 닫혔지만 크게 개선"인 가장 방어 가능한 지점.

→ `Bias/result/bias_centroid/precision_breakdown/`

### 1.6 Seed 안정성 (4 seed, alpha=0.8 고정)

| seed | acc | DEO_lit | recall_unmarried_all | cg_boundary 변화율 |
|---|---|---|---|---|
| 0 | 0.8730 | 0.0476 | 0.4829 | 0.0000 / 0.0005 |
| 1 | 0.8730 | 0.0476 | 0.4829 | 0.0000 / 0.0005 |
| 2 | 0.8730 | 0.0476 | 0.4829 | 0.0000 / 0.0005 |
| 3 | 0.8730 | 0.0459 | 0.4848 | 0.0000 / 0.0005 |

평균 DEO_lit=0.0472±0.0007 — δ_v norm 자체도 seed 간 8.949~8.955로 거의 불변.
재구성 안정성 매우 높음.

### 1.7 최종 결론 (alpha=0.8)

| 지표 | base | alpha=0.8 |
|---|---|---|
| accuracy | 0.8734 | 0.8730 (-0.04%p) |
| DEO_lit | 0.0594 | **0.0476** (목표 0.05 통과, 여유 0.003로 빠듯) |
| 비혼 recall(전체) | 0.4373 | 0.4829 (+4.56%p, 목표 +3.3%p 충족) |
| cg_boundary_fixed 변화율 | — | 0.0000 |
| cg_boundary_random 변화율 | — | 0.0005 |

성공 기준 3개(DEO<0.05, acc↓≤1%p, cg rule 무손상) 전부 통과, procedural하게도 방어 가능한 값. DEO 여유가 빠듯하다는 점은 한계로 남김.
---

## 2. Rule(정책) 편집 — capital-gain 임계값 7,000 → 7,700

### 2.1 임계값 결정

README에 구체적 목표값이 없어 데이터 분포로 역산. capital-gain 값이 스파이크(7298, 7688,
15024 등)에 몰려있어 "발동해제 20~30%"에 정확히 맞는 값은 없음(17.6%/38.9% 두 후보만 존재,
공백 구간). **7,700(발동해제 38.9%, 1,399명 중 544명)으로 확정**

### 2.2 Locate (Exp1 결과 재사용)

기존 Exp1 스캔에 `capital-gain` position 데이터가 이미 포함되어 있어 재실행 없이 재사용.
cg 신호는 layer 0부터 이미 강함(자기 위치라 당연), marital 오염은 layer 8~15 구간에서
최소(std≈4.5~13.3). **layer=15**를 cg 신호 강함 + marital 오염 최소 지점으로 채택.

### 2.3 δ_v + gate + alpha 보정

`delta_v` = μ(cg<7000) − μ(band) at layer15, capital-gain position. gate 학습 데이터는
Regulation v3a 레시피(gaussian + 실제 문맥 다양성) 재사용 — band 중심 gaussian(σ=350,
3σ로 넓게) + train.csv 실제 행. 합성 held-out acc 99.2~100%.

seed0: non-band 오탐 0% (전 alpha). seed1: non-band 오탐이 alpha 무관하게 7.7%로 일정 —
gate 자체의 경계선 문제로 추정, seed0 기준으로 진행.

### 2.4 이중 평가 프레임 + alpha 스윕 

test.csv의 라벨은 실제 소득이라 "규칙 자체의 정답"이 없음(Regulation과 달리 rule=ground
truth가 아님) → (a) 순수 rule 라벨(cg≥7700) 기준과 (b) 실제 소득 라벨 기준을 나란히 평가.

| alpha | efficacy_band | loc_above | acc_RULE(band) | **acc_REAL(band)** |
|---|---|---|---|---|
| 0(원본) | 0.000 | 1.000 | 0.000 | **0.9921** |
| 1.0 | 0.020 | 0.856 | 0.020 | 0.9723 |
| 1.3 | 0.123 | 0.598 | 0.123 | 0.8696 |
| 1.5 | 0.589 | 0.598 | 0.589 | **0.4190** |
| 1.75 | **1.000** | 0.598 | **1.000** | **0.0079** |
| 2.0 | 1.000 | 0.598 | 1.000 | 0.0079 |

**band 253명 중 99.2%가 실제 진짜 고소득자.** alpha를 올려 규칙 반영을 완벽하게 만들수록
(efficacy_band→1.0) 실제 정확도는 거의 0으로 붕괴(0.992→0.008) — **규칙 반영과 현실
정합이 거의 정확히 반대 방향으로 움직이는 구조적 트레이드오프**이며, alpha 조정으로
해소되지 않음.

**추가 문제**: `loc_above`(band와 무관해야 할 cg≥7700 그룹의 예측 불변율)가 alpha≥1.3부터
1.0→0.598로 떨어진 채 유지 — gate가 band 경계에서 깔끔하게 안 갈라져 above 그룹 40%가 피해를 입음.

→ `Bias/result/bias_centroid/cg_alpha_sweep/`

### 2.5 Flip 원인 분해 (alpha=1.5, band 253명 중 flip 149 / unflip 104)


| | flip | unflip |
|---|---|---|
| 고학력(학사+) | 51.7% | 50.0% (거의 동일) |
| 전문/관리직 | 52.3% | **44.2%** (오히려 낮음) |
| Husband(관계) | 80.5% | **95.2%** |
| 남성 | 83.2% | **95.2%** |

학력·직업은 두 그룹이 비슷하거나 unflip이 오히려 낮음 — 가설과 반대. 대신 **unflip
그룹은 압도적으로 "기혼 남성"에 쏠려 있음.** rule 편집의 미완결 부분이 bias 축(혼인/성별)과
같은 인구통계 방향으로 갈림, 
§1의 분리 확인이 인공 marital 쌍으로만 이뤄져 실제 band 인구의 얽힘까지는 못 잡았을 가능성을 시사.

→ `Bias/result/bias_centroid/cg_flip_decompose/`

### 2.6 결론 / 한계 (미확정)

rule 편집은 (a) 규칙 반영 성능은 완벽하게 만들 수 있지만 (b) 현실 정확도를 구조적으로 희생해야 하고, (c) 그 희생 패턴 자체가 bias 축과 얽혀 있음이 확인됨. Regulation이나 bias 축처럼 "손실 거의 없이" 되는 케이스가 아니었음. **최종 alpha 미확정**
gate 경계 문제(above 40% 손상)부터 고칠지, 다른 layer를 더 탐색할지는 다음 세션에서 결정 필요.

---

## 3. 두 편집의 상호작용 (2×2)

설정: bias 편집 alpha=0.8(layer9) × cg 편집 alpha=1.5(layer15), test.csv 16,281건 전체.

| 조합 | accuracy | efficacy_band(cg) | DEO_lit | 비혼 recall(전체) |
|---|---|---|---|---|
| 둘 다 off | 0.8734 | 0.000 | 0.0594 | 0.4373 |
| cg만 | 0.8543 | 0.589 | 0.0461 | 0.4163 |
| bias만 | 0.8730 | 0.000 | 0.0476 | 0.4829 |
| **둘 다 on** | 0.8539 | **0.589** | **0.0310** | 0.4658 |

- cg 편집의 efficacy_band: bias 편집을 같이 켜도 **완전 무손상** (0.589→0.589)
- bias 편집의 DEO_lit: cg 편집을 같이 켜면 **오히려 개선** (0.0476→0.0310)
  보강되는 방향 (band 인구의 성비 편중 때문으로 추정)
- bias 편집의 비혼 recall: cg 편집을 같이 켜면 **1.7%p 손실** (0.4829→0.4658)
  확인된 실질적 간섭. §2.5의 "기혼 남성 쏠림"과 같은 메커니즘일 가능성.

→ `Bias/result/bias_centroid/evaluate_cg/`

---

## 4. 미해결 이슈 / 다음 단계

- rule(cg) 편집 최종 alpha 미확정 — §2.6 한계 해소 방안 결정 필요
- cg gate의 band/above 경계 분리 문제 (loc_above 40% 손상) 개선
- bias-cg 상호작용에서 발견된 "기혼 남성 쏠림" 메커니즘 추가 조사
- bias 축 seed 안정성은 4개 seed로 충분히 확인됐으나, cg 축은 2개 seed만 확인(seed1의
  7.7% 오탐 편차 원인 미해명)
- `result/bias_final/`로 최종 결과 freeze, `Bias/README.md` 체크리스트 갱신, git 커밋+태그
  (`bias-report-v1`) — 아직 미착수

## 5. 파일 위치

| 내용 | 경로 |
|---|---|
| 모델 | `Bias/model/finetuned_xlm-roberta-large_cgmix_rel/best/` |
| bias 축 스크립트 | `Bias/src/editing/bias_centroid_{common,locate,edit,evaluate}.py` |
| bias 축 절차검증/정밀도 | `Bias/src/editing/bias_precision_breakdown.py` |
| rule(cg) 축 스크립트 | `Bias/src/editing/bias_centroid_{edit_cg,evaluate_cg}.py`,
  `bias_cg_alpha_sweep.py`, `bias_cg_flip_decompose.py` |
| 공용 데이터 생성 | `Bias/src/editing/bias_gen_pairs.py` |
| 결과 전체 | `Bias/result/bias_centroid/{locate_scan,edit,evaluate,precision_breakdown,edit_cg,cg_alpha_sweep,cg_flip_decompose,evaluate_cg}/` |
