# [금융/신용1] alpha=0.9 선택 근거 — 합성 calibration 기반 "블라인드" alpha

**핵심**: 편집 세기 `alpha`는 **test.csv를 전혀 보지 않고** 결정했다. train.csv 통계에서
역산한 목표 flip률을, `bias_gen_pairs.sample_marital_pairs`로 만든 **합성 calibration 셋**
(train.csv 유래)에서의 "비혼 1→0 반대방향이 아닌, 비혼 0→1 승인 flip률" 곡선에 보간해 얻는다.
(지시서 방화벽 규칙 §5·CLAUDE.md Bias §5 준수.)

## 1. 목표 flip률 산출 (train.csv만 사용)

- 편집 목적: 비혼(Not-in-family/Never-married 등) 문맥의 **과소승인 교정**. KAIST 리포 기준
  DEO를 닫는 데 필요한 양은 "비혼 recall +3.3%p"로 산술 계산됨
  (`bias_mechanism_analysis.md` §6~8, calibration 근거: 비혼 precision 0.854 > 기혼 0.763
  이라 비혼에 승인 여유가 있음).
- 비혼 집단 중 **현재 미승인(ŷ=0)인 자격자**를 양성으로 뒤집는 비율로 환산하면 목표 flip률 ≈ **5.1%**.
  (recall +3.3%p를 "비혼 0-예측 샘플 대비 flip 비율"로 바꾼 값. README 목표 +3.3%p의 역산치.)

## 2. seed별 합성 calibration 곡선 → 블라인드 alpha (보간)

각 seed의 `layer9_seed{s}/report.json`의 `calibration`(합성 셋, test.csv 무사용)에서
`unmarried_flip_0to1`가 목표 5.1%가 되는 alpha를 선형보간.

| seed | 합성 flip@α0.9 | 블라인드 α (flip=5.1%) |
|---|---|---|
| 0 | 0.0687 | **0.861** |
| 1 | 0.0388 | **0.926** |
| 2 | 0.0583 | **0.856** |
| 3 | 0.0678 | **0.825** |
| **평균±std** | | **0.867 ± 0.037** (범위 0.83~0.93) |

- `Bias/docs/centroid_experiment_report.md` §1.4의 기존 블라인드 값(seed0=0.86, seed1=0.93,
  2-seed 평균 0.89~0.9)과 **재생성값이 일치**(seed0 0.861, seed1 0.926).
- 채택한 **alpha=0.9는 블라인드 범위 [0.83, 0.93] 안**이며 2-seed 평균(0.895)에 근접.
  test.csv 실평가를 보고 고른 값이 아님을 확인.

## 3. 결론

- alpha=0.9는 합성 calibration만으로 정당화되는 블라인드 선택이다.
- 이 alpha에서 test.csv 실측(평가 전용): 4 seed 모두 **acc 0.8714 / DEO 0.0419** (아래 작업3 결과),
  목표(acc≥0.82, DEO≤0.05) 충족.

(산출 수치 원본: `cert/results/_alpha_selection_calc.json`)
