# Bias — UCI Adult 기반 두 번째 도메인

UCI Adult(Census Income) 데이터를 쓰는 서브프로젝트.
Regulation(한국어 재무규칙 대출심사)과 폴더 계층은 맞추되, 라벨은 실제 소득
(연소득 >50K)을 그대로 쓴다. 두 축을 편집으로 다룬다:
① **rule(=정책) 편집** — "세법 개정" 시나리오: capital-gain ≥ 7,000 soft rule의
   임계값 변경. Regulation의 threshold 편집과 같은 유형 (rule과 정책은 같은 대상).
② **편향(bias) 편집** — 성별 공정성 (DEO < 0.05). rule 변경이 아니라 모델이
   실데이터에서 흡수한 편향을 교정하는 **추가적인 편집** 유형.

## 계층 (Regulation과 동일 구조)

```
Bias/
├── data/
│   ├── raw/adult/       # UCI 원본 (adult.data / adult.test / adult.names)
│   ├── train/text/      # train.csv (adult.data 32,561 전체)
│   ├── train_cg_mix/    # cg rule 주입용 gaussian 생성데이터 (v1/v2)
│   └── test/text/       # test.csv 16,281 + cg 경계 진단셋 (fixed/random)
├── src/                 # common/prompt.py, dataset/, trainer/, models/(test·프로브)
├── shell/               # run_train.sh / run_test.sh (GPU 3 고정)
├── result/              # llm_base_training / llm_cgmix_training / llm_cgmix_rel_training / analysis
├── docs/                # prompt_training_plan.md
├── reference/           # deo_github_vs_literature.md / baseline_training.md
└── external/            # 외부 리포 클론 (ai_dep — git 미추적)
```

## 목표 지표

**DEO** (보호 속성 = sex). 비교 대상: KAIST ai_dep `financial_credit_1`
— Accuracy 81.00% / DEO 0.069. 단, 리포 구현은 문헌 정의(TPR 격차)와 달리
그룹 정확도 격차를 계산함 → 상세: `reference/deo_github_vs_literature.md`

**편집 시나리오 성공 기준 (2026-07-16 확정)**:

| 기준 | 목표 | 현 최선(cgmix_rel epoch 3) |
|---|---|---|
| 편집 후 DEO_문헌 | **< 0.05** (1차 목표) | 0.0594 |
| 편집 후 DEO_repo | < 0.05 (구조적 재검토 권고 — 아래 (B) 참고) | 0.0980 |
| 예측 정확도 | **≥ 0.82** | 0.8734 |
| 편집에 의한 정확도 하락 | **≤ 1%p** | (편집 단계에서 측정) |

## 모델 계보 — base → cgmix → cgmix_rel 경위

XLM-R-large + 영어 프롬프트(변수 14개 전부), adult.data 32,561 전체 학습(val 없음,
KAIST 프로토콜 대칭), 3 epoch. 셋 다 같은 구조이고 **학습 데이터 구성만 다르다.**

| 모델 | 학습 데이터 | acc | DEO_문헌 | cg rule (random 문맥 발동률) | 결과 폴더 |
|---|---|---|---|---|---|
| llm_base | 실데이터만 | 0.8714 | 0.0961 | **0.58 (실패)** | `result/llm_base_training/` |
| llm_cgmix (v1) | + gaussian 16,280 (sex만 라벨별 50:50) | 0.8722 | 0.0705 | 0.999 | `result/llm_cgmix_training/` |
| **llm_cgmix_rel (v2, 현 기준)** | + gaussian 16,280 (sex + Husband/Wife 4셀 균형) | **0.8734** | **0.0594** | 1.000 | `result/llm_cgmix_rel_training/` |

**경위**:

1. **base**: 실데이터만으로는 cg≥7000 rule이 "실제 문맥에 얽힌 형태"로만 학습됨
   (실데이터 발동 99% vs 저소득 고정 문맥 40%) — rule 편집 실험의 전제인
   문맥-독립 rule이 없어서 탈락.
2. **cgmix v1**: Regulation 레시피(실:생성 배치 2:1)로 gaussian 경계 데이터를 혼합해
   rule 확립(OOD 99.9%), acc 무손실. gaussian의 sex를 남성 고정에서 라벨별 50:50으로
   바꾼 버전. DEO도 부수적으로 0.096→0.070 개선됐으나 목표 미달.
3. **cgmix v2**: gaussian 문맥의 기혼 프록시(Husband/Wife)까지 라벨과 독립이 되게
   4셀 균형 — acc 최고 + DEO_문헌 0.0594로 최저. 부작용으로 **여성 표지 우대 토큰
   경로**가 생김(스왑 프로브: 같은 사람을 여성으로 쓰면 승인↑) — 이는 편집 단계에서
   증폭 가능한 손잡이이기도 함. 상세: `result/llm_cgmix_rel_training/summary.md`.

## 세 모델 공통의 핵심 문제 — (B) 구성 차이

DEO 격차의 몸통은 세 모델 모두 **(A) "같은 상황의 여성을 더 거절"이 아니라
(B) "승인율 낮은 상황(비혼 가구주)에 자격 있는 여성이 몰려 있음"** 이다:

- 자격(y=1) 여성의 39%가 비혼 (남성은 10%). 비혼 상황의 TPR은 ~0.4~0.5로 낮음.
- 사고실험(cgmix_rel 실측): 여성의 상황별 승인율은 그대로 두고 혼인 구성만
  남성 것(90:10)으로 바꾸면 여성 TPR 0.598→0.695 — **남성(0.667)을 추월**.
  즉 구성만으로 격차가 전부 설명되고도 남음.
- (A)쪽 검증: v1은 성별 표기 스왑에 예측 불변(토큰 경로 없음), v2는 오히려 여성
  우대 방향. base는 묶음 스왑 미측정이나 같은 실데이터로 학습되어 같은 구성
  문제를 공유하며 DEO가 가장 컸음.
- 따라서 "성별 경로 차단"형 편집은 무효(끊을 것이 없거나 역효과), **비혼 상황의
  승인율을 끌어올리는 (B)형 개입**이 편집의 실질 목표.
  DEO_repo가 세 모델 모두 ~0.10에서 움직이지 않는 것도 (B)와 남녀 양성비율
  차이가 만드는 구조적 하한 때문 (성별-불변 KAIST 재학습 모델도 0.086~0.122).
- 근거 분석: `result/llm_cgmix_training/bias_mechanism_analysis.md`
  (스왑 프로브 / FN 여성 분해 / 혼인×성별 TPR 교차 / 중립화 프로브 / 캘리브레이션)

## 편향 편집의 확정 목표 — "비혼 페널티 완화" (2026-07-17)

**요약 근거** (상세: bias_mechanism_analysis.md §6~8, cgmix_rel 기준):

1. **비혼 디메리트는 남녀 공통이고 여성이 더 심하다**: 비혼 자격자의 recall이
   기혼 대비 남성 −19%p(0.495), 여성 −40%p(0.369)로 급락하는데, precision은
   비혼(0.854)이 기혼(0.763)보다 높다 — 일관된 확신 기준이라면 있을 수 없는
   **구간별 문턱 불균형(과소승인)**이 실측됨. 편집의 근거가 성별이 아니라
   데이터(캘리브레이션 교정)에서 나옴.
2. 전면 개입은 실패 실측: relationship 신호 제거(전원 비혼 취급)는 격차를 거의
   못 줄이고(−0.005) acc만 −3%p; 친여성 토큰 경로 증폭은 데이터 근거가 없는
   정책-보정이라 예비안으로 강등.
3. 산술: 비혼 recall을 남녀 공통 ~3.3%p 올리면 (자격 여성의 39%가 비혼 vs
   남성 10%) 구성 레버리지로 DEO_문헌 <0.05 도달.

**따라서 목표**: locate로 **내부에서 capital-gain(rule)이 반응하지 않으면서
비혼 표지(Not-in-family / Never-married / Unmarried)가 반응하는 위치**를 찾고,
그 위치의 판정 기여를 완화하는 편집을 한다. 성공 조건: DEO_문헌 <0.05 +
acc 하락 ≤1%p + cg rule 무손상(두 위치의 분리로 보장 — rule(정책) 편집과
편향 편집이 서로를 건드리지 않는 구조).

## 상태

- [x] 원본 데이터 확보, DEO 정의 검증 (2026-07-16)
- [x] 영어 프롬프트 학습 파이프라인 + base/cgmix/cgmix_rel 3종 학습 (2026-07-16)
- [x] cg rule 확립 (OOD 발동 99%+) / DEO 기제 분석 (구성 차이 판정) (2026-07-16~17)
- [x] relationship 중립화 프로브 + 캘리브레이션 분석 → 편향 편집 방향 확정:
      **비혼 페널티 완화** (2026-07-17, 위 "확정 목표" 섹션)
- [ ] locate (cg rule 위치 / 비혼 표지 위치 — 분리 확인이 핵심)
- [ ] edit (① rule(정책): cg 임계값 변경, ② 편향: 비혼 recall +3.3%p → DEO<0.05) — 성공 기준 위 표
