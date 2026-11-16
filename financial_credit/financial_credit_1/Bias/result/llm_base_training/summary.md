# LLM base 학습 결과 (Adult 프롬프트 × XLM-R-large)

- 계획·설정: [docs/prompt_training_plan.md](../../docs/prompt_training_plan.md)
- 학습: `src/trainer/train.py` — adult.data 32,561 전체(val 없음, KAIST 프로토콜 대칭),
  영어 템플릿(변수 14개 전부), lr 1e-5, batch 32, 3 epoch, seed 42, GPU 3
- 평가: `src/models/test.py` — 에폭별 체크포인트 × {실제 test 16,281, cg 경계셋 2,000}
- 폴더: `epoch{1,2,3}/`(실제 test), `epoch{1,2,3}_cg_boundary/`(경계셋)
  — 각각 metrics.json / confusion_matrix.csv / predictions.csv

## 1. 실제 test 성능 — KAIST 비교표

지표 정의는 [reference/deo_github_vs_literature.md](../../reference/deo_github_vs_literature.md)
(deo_repo = 성별 그룹 정확도 격차, deo_lit = y=1 조건 TPR 격차, 동일 test 16,281행).

| 모델 | Accuracy | DEO_repo | DEO_문헌 |
|---|---|---|---|
| KAIST fair (리포 보고값) | 0.8100 | 0.069 | (보고 없음) |
| KAIST fair (재학습, epoch 2~50 범위) | 0.821~0.827 | 0.086~0.122 | 0.015~0.040 |
| KAIST fair (재학습, 최종 epoch 50) | 0.8256 | 0.0931 | 0.0208 |
| **우리 (epoch 1)** | 0.8581 | 0.1096 | 0.0996 |
| **우리 (epoch 2)** | 0.8707 | 0.0990 | 0.0891 |
| **우리 (epoch 3, 최종)** | **0.8714** | 0.0976 | 0.0961 |

- **정확도는 KAIST 대비 +4.6%p** (0.8714 vs 0.8256) — 텍스트 렌더링 경유로도
  표형 상한(~0.87)에 도달.
- **DEO_문헌이 0.09~0.10으로 KAIST(0.02~0.04)보다 큼** — 당연한 결과: KAIST 모델은
  구조적으로 성별-불변(반사실 앙상블)이고, 우리는 sex를 자유롭게 쓰는 base 모델.
  세부: TPR 남 0.684 vs 여 0.588 (자격 있는 여성이 더 많이 거절됨), 그룹 정확도는
  여 0.937 > 남 0.839. **이 격차가 이후 정책(locate/edit) 개입의 편집 대상**이며,
  "편집으로 DEO를 KAIST 수준 이하로 낮추면서 acc 0.87을 유지"가 목표 구도.

## 2. capital-gain ≥ 7000 soft rule 이해도 진단

경계셋 2종 (생성: `src/dataset/generate_cg_boundary_test.py`, cg ~ N(7000, 525)
양측 각 1,000행):

- **fixed** (`cg_boundary_fixed.csv`, 본 진단): 나머지 13개 feature를 **train label-0
  집단의 대표 프로필**(범주=최빈값, 수치=중앙값: Private·HS-grad·Never-married·
  Adm-clerical·Not-in-family·White·Male·US·34세·40h·loss 0·fnlwgt 179,465)로 전 행
  고정. Regulation의 "다른 rule=통과값 고정"의 대응 — 이 프로필은 cg=0일 때
  실데이터 P(y=1)=0.009라 **cg만이 판정을 뒤집을 수 있음**.
- **random** (`cg_boundary_random.csv`, 참고): 각 feature를 train 주변분포에서 독립
  샘플 (문맥 랜덤화의 영향 측정용).

| 발동률 P(pred=1) | epoch 1 | epoch 2 | epoch 3 |
|---|---|---|---|
| **fixed 문맥**: 경계 위 (기대 1) | 0.059 | 0.393 | **0.398** |
| **fixed 문맥**: 경계 아래 오발동 (기대 0) | 0.008 | 0.109 | 0.231 |
| random 문맥: 경계 위 | 0.260 | 0.551 | 0.581 |
| random 문맥: 경계 아래 오발동 | 0.180 | 0.226 | 0.332 |
| **실제 test**: cg≥7000(656행) | 0.941 | 0.992 | **0.991** |
| 실제 test: 0<cg<7000(667행) | 0.274 | 0.259 | 0.261 |

(실데이터 정답 기준: cg≥7000 라벨1 비율 0.988 / 0<cg<7000은 0.249)

**판정: rule은 실제 문맥에서는 완벽(99%)하지만, 문맥-독립 override로는 학습되지
않았다 — 저소득 고정 문맥에서는 40%밖에 발동하지 않는다.**

- fixed 문맥(저소득 대표 프로필)에서 cg를 아무리 올려도 열에 여섯은 거절 유지 —
  모델은 "cg 크고 문맥도 고소득스러우면 1"을 배웠지, 실데이터가 보여주는
  "cg≥7000이면 문맥 불문 1(P=0.99)"의 override 구조를 배우지 못함.
- random 문맥의 58%는 고소득 쪽 문맥이 섞여 부풀려진 수치였음 (fixed가 더 엄밀).
- → **rule 편집 실험을 위해서는 cg 경계 gaussian 생성데이터(랜덤/저소득 문맥 +
  rule 라벨)를 학습에 혼합(Regulation의 interpolation:gaussian ≈ 2:1 레시피)해
  또렷한 rule을 심는 재학습이 필요.**

## 3. sex 직접 경로 진단 (sex-flip probe) — DEO<0.05 목표의 실현 조건

`src/models/probe_sex_flip.py`: test 전 행의 sex 토큰만 Male↔Female로 뒤집어
epoch 3 모델 예측 비교 (`sex_flip_probe/`).

| 지표 | 값 |
|---|---|
| 예측이 바뀌는 행 (전체 / y=1) | 1.3% / 2.5% |
| DEO_문헌 (원본) | 0.0961 |
| **sex 토큰을 완전 중립화해도 남는 DEO_문헌 (근사)** | **0.0589** |
| TPR_여성: 원본 → 남성 취급 시 | 0.588 → 0.624 (여전히 남성 0.684 미달) |

**함의: 현 DEO 0.096 중 sex 토큰 직접 경로의 지분은 ~0.037뿐이고, 나머지
~0.059는 대리변수(relationship의 Husband/Wife, marital-status 등 성별 상관
feature)를 타고 있다.** sex 토큰 경로만 완벽히 편집해도 0.059 > 목표 0.05 —
**목표 달성을 위해서는 반사실 쌍/편집 범위를 sex 단독이 아니라 성별 대리
토큰(최소 Husband↔Wife 스왑)까지 묶어서 정의해야 한다.** (다음 진단 후보:
sex+relationship 동시 스왑 프로브로 남는 격차 재측정)

## 4. 편집 시나리오 성공 기준 (확정, Bias/README.md와 동일)

| 기준 | 목표 | base (epoch 3) |
|---|---|---|
| 편집 후 DEO (repo·문헌 모두) | **< 0.05** | 0.0976 / 0.0961 |
| 예측 정확도 | **≥ 0.82** | 0.8714 |
| 편집에 의한 정확도 하락 | **≤ 1%p** (→ ≥ 0.8614) | — |
