# Adult → 자연어 프롬프트 학습 계획 (Regulation 방식 이식)

목표: Adult의 **14개 변수 전부**를 Regulation과 같은 템플릿 기반 한국어 프롬프트로
렌더링하고, **동일 모델 구조**(`FacebookAI/xlm-roberta-large`, sequence classification,
num_labels=2)로 실제 라벨(연소득 >50K)을 학습 → 이후 정책(보호 속성) locate/edit의
대상 모델을 확보한다.

## 1. 프롬프트 템플릿 (변수 14개 전부 포함, **전체 영어**)

Regulation의 "속성이 값, 속성이 값입니다. ~입니까?" 기계적 구조를 영어로 유지하고,
의미 단위로 3문장 그룹핑한다:

```
A person's age is {age}, sex is {sex}, race is {race},
native country is {native-country}, marital status is {marital-status},
and household relationship is {relationship}.
Their employment type is {workclass}, occupation is {occupation},
education level is {education}, and years of education is {education-num}.
They work {hours-per-week} hours per week, with a capital gain of
{capital-gain} dollars, a capital loss of {capital-loss} dollars,
and a census sampling weight of {fnlwgt}.
Does this person's annual income exceed 50,000 dollars?
```

렌더링 예 (train 1행):

> A person's age is 39, sex is Male, race is White, native country is
> United-States, marital status is Never-married, and household relationship is
> Not-in-family. Their employment type is State-gov, occupation is Adm-clerical,
> education level is Bachelors, and years of education is 13. They work 40 hours
> per week, with a capital gain of 2174 dollars, a capital loss of 0 dollars, and
> a census sampling weight of 77516. Does this person's annual income exceed
> 50,000 dollars?

### 템플릿 설계 결정과 이유

| 결정 | 선택 | 이유 |
|---|---|---|
| 언어 | **전체 영어** (2026-07-16 확정 — 한국어 틀+영어 범주값 안에서 변경) | 데이터의 범주값·도메인이 원래 영어라 언어 일관성이 자연스럽고, "속성명 is 값" 구조라 locate 시 span 규칙도 단순. XLM-R은 다국어라 Regulation(한국어)과 같은 모델 구조 유지에 문제 없음 |
| 변수 순서 | 인적 속성 → 직업·교육 → 경제 수치 | 의미 그룹핑. 보호 속성 후보(sex, race, age, …)가 앞 문장에 모여 있어 편집 실험 시 span 관리 용이 |
| 결측(`?`) | `unknown`으로 렌더링, **행 삭제 안 함** | workclass 1,836 / occupation 1,843 / native-country 583건. dropna 하면 test가 16,281행에서 줄어 baseline(전체 16,281로 평가)과 비교 불가 |
| 수치 포맷 | 정수는 정수로 (`format_integer_like` 재사용 — test.csv의 `226802.0` 같은 float 표기 정리) | Regulation과 동일 관행 |
| education / education-num | **둘 다 포함** (중복이지만) | "같은 정보의 두 표현"이 표현 공간에서 어떻게 얽히는지가 이후 오염 분석의 프로브가 됨 |
| fnlwgt | 포함 (인구 표본 가중치, 예측과 무관) | Regulation의 distractor(ROA 등)와 같은 역할 — 모델이 무시해야 할 변수 |
| 단위 | hours per week, dollars 등 영어 단위 명시 | 수치의 의미를 명확히 — Regulation의 `%` 역할 |

토큰 길이: 위 예시가 XLM-R 토크나이저로 ~90 토큰 → `max_length=256` 그대로 안전.

## 2. 데이터 구성

| 파일 | 출처 | 행 수 | 용도 |
|---|---|---|---|
| `Bias/data/train/text/train.csv` | `adult.data` **전체** | 32,561 | 학습 (val 분할 없음) |
| `Bias/data/test/text/test.csv` | `adult.test` 전체 | 16,281 | 에폭별 측정 |

- CSV는 Regulation과 동일하게 **원 변수 컬럼 + label** 로 저장하고, 프롬프트는
  train/test 코드의 `format_prompt`가 렌더링 (텍스트를 CSV에 굽지 않음).
- label = (income == ">50K") → 1. 클래스 불균형 24:76이므로 accuracy 외에
  macro-F1·recall도 같이 기록 (Regulation test.py가 이미 지원).
- **프로토콜 (2026-07-16 변경): KAIST baseline과 대칭으로 맞춤** — baseline이
  val 없이 adult.data 전체로 학습하고 test를 에폭마다 쟀으므로, 우리도 전체 학습
  + 매 에폭 체크포인트 저장 + 에폭별 test 측정으로 **범위와 최종 에폭 값**을
  보고한다 (KAIST 재측정 표와 같은 형식). 특정 에폭을 test 성적으로 골라
  단일 수치만 보고하는 것은 금지.

## 3. 학습 설정 (Regulation 동일 구조)

| 항목 | 값 | 비고 |
|---|---|---|
| 모델 | `FacebookAI/xlm-roberta-large`, `AutoModelForSequenceClassification`, num_labels=2 | Regulation 동일 |
| max_length / lr / epochs | 256 / 1e-5 / 3 (필요시 5까지) | Regulation 기본값 |
| batch | GPU 3 단독 기준 16~32 (메모리 보고 조정) | 현재 GPU 3만 사용 가능 |
| 샘플러 | 일반 셔플 | Regulation의 RatioSampler(gaussian:interpolated 1:2)는 합성 2원천 전용이라 불필요 |
| 저장 | `Bias/outputs/finetuned_xlm-roberta-large/best/` | val macro-F1 기준 best |

예상 소요: 29k행 × 3에폭, XLM-R-large, 단일 GPU → 1시간 이내.

## 4. 평가 계획

`Bias/src/models/test.py`(Regulation test.py 이식)로:

1. **Accuracy / macro-F1 / recall / confusion matrix** — test 16,281 전체
2. **predictions.csv** (index, pred) — 이후 GPU 없이 공정성 분석 재계산 가능
3. **공정성**: sex 기준 DEO 두 방식 병기
   - DEO_repo (그룹 정확도 격차) — KAIST 수치(0.069 보고 / 재학습 0.09~0.12)와 비교용
   - DEO_문헌 (y=1 조건 TPR 격차) — 보고용
4. 비교표 — 기준은 `Bias/reference/deo_github_vs_literature.md`의 재측정 결과
   (동일 test 16,281행, 동일 지표 정의):

   | 모델 | Accuracy | DEO_repo | DEO_문헌 |
   |---|---|---|---|
   | KAIST fair (리포 보고값) | 0.8100 | 0.069 | (보고 없음) |
   | KAIST fair (재학습, epoch 2~50 범위) | 0.821~0.827 | 0.086~0.122 | 0.015~0.040 |
   | KAIST fair (재학습, 최종 epoch 50 대표값) | 0.8256 | 0.0931 | 0.0208 |
   | **우리 텍스트 모델 (epoch 1~3 범위)** | 0.858~0.871 | 0.098~0.110 | 0.089~0.100 |
   | **우리 텍스트 모델 (최종 epoch 3)** | 0.8714 | 0.0976 | 0.0961 |

   참고: KAIST nonfair baseline은 리포 보고값 0.8176 / 0.138 (재학습 안 함).
   상세 결과·해석: `result/llm_base_training/summary.md` (2026-07-16 완료)

이 "정책 시행 전" 텍스트 모델의 acc와 DEO가 이후 **locate/edit의 출발점**이 된다
(편집으로 DEO를 낮추면서 acc 유지 → KAIST의 재학습식 개입과 트레이드오프 대결).

## 5. 파일 계획 (Regulation 계층 미러링)

```
Bias/src/dataset/generate_text_csv.py   # raw adult → train/val/test csv (+라벨, 결측→'미상')
Bias/src/models/test.py                 # format_prompt(Adult판) + 지표/예측 저장
Bias/src/trainer/train.py               # Regulation train.py 이식(샘플러 단순화)
Bias/shell/run_train.sh, run_test.sh    # GPU 3 고정
Bias/result/evaluation/base/            # 평가 산출물
```

`format_prompt`는 Regulation과 같은 이유(모델은 학습 문장과 동일한 텍스트에서만
성능이 남)로 train.py/test.py 간 **한 글자까지 동일**해야 함 — 이번엔 중복 대신
`Bias/src/common/prompt.py` 한 곳에 두고 양쪽에서 import.

## 6. 리스크

- **성능 상한**: Adult 표형 SOTA는 ~87%(GBM). 텍스트 렌더링 경유로 84~86%대면
  성공적 — baseline(81~82.5%)보다 높으면 비교 구도에 문제 없음.
- **자본이득 롱테일**(0~99,999, 92%가 0): 모델이 자릿수 단서로만 학습할 수 있음 —
  이후 locate에서 수치 토큰 처리 시 Regulation 금액 rule에서 얻은 노하우 재사용.
- **실제 라벨의 노이즈**: 규칙 유도가 아니므로 100% 정답 경계가 없음 —
  Regulation과 달리 retain셋 정확도가 1.0에서 시작하지 않는다는 점을 지표
  해석에 반영해야 함.
