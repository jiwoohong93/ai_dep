# Rule Editing 베이스라인 실험 계획

**목표**: 9개 편집 케이스([data/rules/edited_rule_25pct.md](../data/rules/edited_rule_25pct.md)) 각각에 대해,
모델 내부에서 해당 rule이 계산되는 위치를 탐지하고, **ROME / MEMIT / EMMET / AlphaEdit**으로
그 위치의 가중치만 편집해 새 threshold를 주입한다. 나머지 8개 rule의 판정은 보존되어야 한다.

**대상 모델**: `yongukpark/xlm-roberta-large-rule` — XLM-RoBERTa-large (encoder, 24층, d=1024,
MLP intermediate 4096) + `<s>` 표현을 읽는 분류 헤드.

---

## 데이터 사용 제약 (과제 가정 — 위반 금지)

**시나리오**: rule이 바뀌었을 때, 바뀐 rule의 내용(θ → θ′)만 알고 있으며
**실제 은행·기업들에 대한 새 라벨은 모른다**. 라벨링 없이 모델 내부 수정만으로
효율적으로 모델을 갱신하는 것이 과제다.

| 구분 | 편집(방법의 입력) | 평가 |
|---|---|---|
| train set (interpolated / gaussian csv) | ❌ 사용 금지 | — |
| test set (실제 은행×기업 쌍, test.csv 및 test_case{i}.csv) | ❌ 사용 금지 | ✅ 평가 전용 |
| **rule 주변 랜덤샘플링 합성 데이터** (기존 θ·새 θ′ 기준, 라벨은 rule에서 유도) | ✅ 유일한 허용 데이터 | ✅ generalization 평가에도 사용 |

- 편집에 필요한 모든 통계량(k\*, v\*, C, K₀, tracing용 최소쌍)은 **합성 데이터에서만** 계산한다.
- **효율성 축 (2개)**: ① locate — 쌍 수 1k~10k에 따른 위치 안정성 (Phase A),
  ② edit — 합성 샘플 예산 N에 따른 편집 성능 곡선 (Phase B/D). 방법별 비교.

---

## Phase A. 위치 탐지 — Causal Tracing의 encoder 적응

### A1. 최소쌍(minimal pairs) 생성 설계

ROME의 causal tracing(주어 토큰에 노이즈 → 복원)을 그대로 쓰지 않고, 우리 셋업의 장점을 활용해
**정확한 counterfactual 쌍**으로 대체한다. train/test 사용 금지 제약 하에서 전부 합성 생성
(라벨은 rule에서 유도).

**쌍의 구조** — rule i의 최소쌍 = 동일 문장 2개, **오직 rule i 값만 현 rule(θ) 경계 반대편**:

```
base   : rule i = v_pass (통과)  → 라벨 1
source : rule i = v_fail (탈락)  → 라벨 0
컨텍스트(나머지 12개 지표): 쌍마다 새로 샘플링, 두 문장에 공유
```

- tracing 신호는 두 문장의 **출력 차이**에서 나오므로 경계 양쪽 대비가 필수
  (플립존끼리는 편집 전 모델에게 둘 다 "통과"라 신호 없음 — θ′는 아직 모델에 없다)
- 문장 렌더링은 `format_prompt` 단일 소스 사용 (형식 불일치 사고 방지)

**쌍의 값 샘플링 — 학습 분포(gaussian) 재현 방식**: 넓은 구간 쌍(120↔320)은 verdict 차이와
크기 차이가 섞여 IE 해석이 오염된다. 학습 때 경계를 학습시킨 바로 그 분포 —
`generate_train_gaussian.py`의 **N(θ, σ)** — 에서 양쪽 절단 샘플링해, 모델이 경계를
가장 선명하게 학습한 in-distribution 영역에서 locate한다:

```
v_pass ~ N(θ, σ) | 통과측 절단,  v_fail ~ N(θ, σ) | 탈락측 절단  (+ 신뢰 마진 δ)
σ: 퍼센트 지표 = max(0.15θ, 1)/2 ≈ θ의 7.5%  /  금액 지표 = 500,000 (절대값)
```

- **학습 시 경계 노출량 (참고)**: gaussian train set 500,000행, 행마다 지표 2개만 흔듦
  (78쌍 순환) → rule당 약 77,000행이 그 경계 주변 학습에 사용됨
- **신뢰 마진 δ**: 경계에 너무 붙으면 모델 출력이 불확실 → IE 노이즈. 띠에서 뽑되
  **pre-edit 모델이 base=1, source=0을 확신 예측하는 쌍만 통과**시키는 필터 적용
- **⚠ rule 4·9 (금액, >0) 스케일 주의**: 모델이 부호 경계를 학습한 범위는 σ=50만
  → **±150만 이내**. 실데이터(100억~400조)와 4~6자릿수 차이. locate 쌍은 ±수십만~150만
  스케일로 생성해야 하며, 편집 목표 θ′(620억·3조)는 gaussian 학습 범위 밖(interpolated
  데이터만 커버, 전부 통과 라벨) — 이 두 케이스가 최난이도일 것으로 사전 예측됨

**값 구간 정의 (rule별)** — 아래 3구간은 쌍 샘플링이 아니라 **편집(B)·평가(D)용
비페어 샘플의 범위** 정의로 사용. 안전존(새 θ′로도 통과) / 플립존(현재 통과, 편집 후 탈락) /
탈락존(현재도 탈락):

| # | rule (θ → θ′) | 안전존 | 플립존 | 탈락존 | 비고 |
|---|---|---|---|---|---|
| 1 | 부채비율 ≤200 → ≤85 | [10, 85) | (85, 200] | (200, 400] | |
| 2 | 유동비율 ≥100 → ≥140 | [140, 400] | [100, 140) | [40, 100) | 정수 자릿수 2↔3 불일치 상존 |
| 3 | 자기자본비율 ≥20 → ≥54 | [54, 95] | [20, 54) | [1, 20) | |
| 4 | 자본총계 >0 → >620억 | (620억, 10조] | (0, 620억] | [-5000억, 0) | **부호 경계** |
| 5 | 보통주자본비율 ≥4.5 → ≥12 | [12, 24] | [4.5, 12) | [0.5, 4.5) | |
| 6 | 기본자본비율 ≥6 → ≥12.5 | [12.5, 24] | [6, 12.5) | [1, 6) | |
| 7 | 총자본비율 ≥8 → ≥12.5 | [12.5, 24] | [8, 12.5) | [2, 8) | |
| 8 | 단순기본자본비율 ≥3 → ≥6.3 | [6.3, 12] | [3, 6.3) | [0.5, 3) | |
| 9 | 공제후자본양수 >0 → >3조 | (3조, 37조] | (0, 3조] | [-5조, 0) | **부호 경계** |

- 구간 상·하한은 실데이터 관측 범위(decisive 분포)를 벗어나지 않게 설정
- 표면형은 학습 데이터 관례 준수: 퍼센트 지표 소수 2자리, 금액 지표 정수
  (`format_integer_like` 규칙 동일 적용)

**컨텍스트(나머지 12개 지표) 샘플링:**

| 그룹 | 분포 |
|---|---|
| 다른 8개 rule 지표 | 각자의 **안전존**에서 uniform — 옛 rule뿐 아니라 그 rule의 θ′로도 통과 |
| distractor 4개 | gaussian 생성기 중심값 주변: ROA~N(0.3, 0.15), 영업흑자·이익잉여금양수 양수, 예대율~N(100, 8) — `"산정불가"` 배제 |

컨텍스트를 안전존으로 제한하는 이유: 편집 후 re-tracing 시 컨텍스트 지표가 자기 rule의
플립존에 있으면 라벨 해석이 오염됨. 안전존 고정이면 어느 케이스의 편집 모델에서도 쌍의 의미가 불변.

**수량과 locate 효율성 스윕 — rule당 10,000쌍 (θ 경계 근처, 문장 20,000개):**

- **rule당 10,000쌍을 한 번에 생성**해두고, tracing은 쌍당 1회만 수행하며 **쌍별 IE를
  개별 저장** → 누적 1k, 2k, ..., 10k 시점의 locate 결과(히트맵·피크 층)를 각각 집계·저장.
  재계산 없이 집계만 누적하면 되므로 스윕의 추가 비용은 0.
- 산출: **"몇 쌍이면 피크 층이 안정되는가"** 곡선 (locate 데이터 효율성) — 이후 실험에서
  tracing 예산을 정하는 근거가 됨
- **방향(1→0, 0→1)은 생성이 아니라 실행 시 처리**: 같은 쌍을 base/source 역할만 바꿔
  두 번 trace → 쌍 수는 늘지 않고 실행이 2배
- **총량: 9 rule × 10,000쌍 = 90,000쌍 (문장 180,000개)** — 생성은 CPU로 수 분,
  모델 정답 필터는 GPU 배치 추론 1 pass (문장 18만 개, 수 분)

**토큰 정렬 (위치별 patching 성립 조건):**

1. **자릿수 매칭 샘플링**: v_pass/v_fail을 같은 정수 자릿수·같은 소수 형식으로 우선 샘플링
2. **rejection sampling**: 두 문장을 실제 토크나이저로 인코딩해 **토큰 수가 다르면 기각·재샘플**
   (sentencepiece의 숫자 분할은 예측 불가하므로 실측만 신뢰)
3. **span-정렬 fallback**: 경계 자체가 표면형 경계인 rule — rule 2(θ=100, 자릿수 2↔3),
   rule 4·9(θ=0, 부호 토큰)처럼 기각 샘플링이 원리적으로 실패하는 경우 —
   patching을 절대 인덱스가 아닌
   **offset mapping 기반 named span**(prefix / value_span / suffix)으로 수행.
   suffix는 길이차만큼 오프셋 정렬, value_span 내부는 마지막 토큰 기준 뒤에서부터 정렬
4. 생성 리포트에 rule별 토큰 일치율 기록 (fallback 경로 사용 빈도 추적)

**라벨 무결성 검증**: 생성된 전 쌍에 대해 `check_rule_labels.compute_rule_labels`로
**base=1, source=0** 기계 검증. 하나라도 실패 시 생성 중단 (컨텍스트 샘플링 버그 차단).

**산출물:**

```
data/rules/edited_rule_25pct/locate/rule{1..9}.csv   # rule당 10,000쌍 (git 추적됨)
컬럼: pair_id, 13개 지표(base 기준), value_fail, base_label, source_label, token_aligned(bool)
생성기: src/editing/generate_locate_pairs.py  # seed 고정
```

문장 텍스트는 저장하지 않음 — trace 시점에 `format_prompt`로 렌더 (단일 소스 원칙).

### A2. Activation patching (interchange intervention)

- **개입**: base run(v_pass)의 (layer, position) hidden state를 source run(v_fail)의 것으로 교체
  → 분류 로짓이 얼마나 뒤집히는지(indirect effect) 측정.
- **스윕 범위**: 24층 전체 × {rule i 값 토큰들(숫자가 여러 토큰으로 쪼개지므로 마지막 토큰 중점),
  해당 지표명 토큰, `<s>`, 문장 끝 질문 토큰} × {residual stream, MLP out, attention out}.

### A3. 산출물: rule별 반응 지도

- rule × layer × position **IE 히트맵** 9장 (locate 예산별: `result/locate/rule{i}/n{1000..10000}/`,
  9 rule × 10 예산 = 90개 결과 + rule별 쌍 단위 원본 `pair_ie.npz`).
- **편집 대상 층 선정**: IE 피크 층 l\* (MEMIT용 구간은 [l\*−2, l\*+2]).
- 예상 패턴: 값 비교는 초중반 층 값-토큰 위치, 결정 집계는 후반 층 `<s>`.
  후반 층 `<s>` 편집은 전부 뒤집는 지름길이 되기 쉬우므로(locality 붕괴) 층별 trade-off 곡선을 남긴다.

### A4. Sanity check (illusion 논문 반영)

- **random-layer 대조군**: tracing 효과가 ~0인 층을 편집해도 efficacy가 동일하게 나오면
  "위치가 무의미하다"는 신호 → Makelov et al. 2024 §6 (rank-1 편집은 fact가 없는 곳에서도
  작동)의 재현 여부 자체가 리포트 항목.

## Phase B. Key–Value 설계 — encoder 분류기 적응 (핵심 기여 지점)

ROME 계열은 MLP fc2(W_out)를 linear associative memory로 보고 `k* → v*` 연상을 삽입한다.
decoder의 (주어 → 목적어) 프레임을 다음과 같이 옮긴다:

| 요소 | 원래 (decoder) | 우리 (encoder 분류기) |
|---|---|---|
| key k\* | 주어 마지막 토큰의 fc2 입력 (GELU 후) | **rule i 값의 마지막 토큰** 위치, 층 l\*의 fc2 입력. 플립존 (θ′, θ] **합성 샘플**들에 대한 평균 |
| value v\* | 목적어 토큰 로짓을 높이는 표현 (경사하강으로 탐색) | **분류 로짓을 0(거절)으로 뒤집는 표현**: v\* = argmin CE(분류출력, 새라벨) + KL(**합성** retain 프롬프트 보존) — 새 라벨은 바뀐 rule에서 유도 (합성값이므로 가능) |
| 공분산 C = E[kkᵀ] | Wikipedia 샘플 | **rule threshold 주변 합성 프롬프트**에서 계산 (train/test 사용 금지) |
| AlphaEdit의 K₀ (보존 키) | 기존 지식 프롬프트 | **합성 retain 키**: 플립존 밖 합성값 + 다른 rule threshold 주변 합성 문맥 |

모든 항목이 합성 데이터만 사용하므로, 합성 샘플 수 N이 각 통계량의 품질을 결정한다
→ N 스윕이 곧 방법별 데이터 효율성 비교가 된다.

**케이스 = 편집 요청 배치(batch)로 프레이밍**: threshold는 연속 구간이므로 단일 요청이 아니라
**플립존을 커버하는 R개 값 샘플 = R개 편집 요청**으로 구성한다. 이 프레이밍에서 4개 방법의
차이가 그대로 드러난다:

- ROME: 요청 1개(또는 순차 R개 — 순차 열화가 알려진 약점, 그대로 리포트)
- MEMIT: R개 배치, 다층 분산
- EMMET: R개 배치, 등식 제약, 단층
- AlphaEdit: R개 배치 + K₀ null-space projection (retain 보존 기대치 최고)

R은 기본 64 고정, 효율성 곡선은 case 1 한정 R{1,16,256} 후속 실행으로 확보
(실행 매트릭스 축소 방침은 Phase B-2 참고).

**열린 질문(=연구 질문)**: 연상 삽입이 연속 threshold 이동으로 일반화되는가?
플립존 안 미학습 값, 경계 바로 밖 값에서의 거동이 관건 → Phase D의 generalization set이 판정.

## Phase B-2. 실행 계획 구체화 (locate 실측 반영)

**locate 확정 결과** (`result/locate/rule{i}/n10000/meta.json`, 유효쌍 전부 ~100%):

| rule | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|
| p2f 피크 | L21 | L18 | L22 | L22 | L22 | L22 | **L12** | L22 | L22 |
| name 전이 시작 | L15 | L15 | L22 | L22 | L22 | L13 | L14 | L22 | L22 |

**단일 기본 설정** (ablation 축소 — 평가 횟수가 병목이므로 설정 하나로 고정):

| 축 | 선택 | 근거 |
|---|---|---|
| 편집 층 | rule별 **p2f 피크** (MEMIT/AlphaEdit은 memit_window 피크±2) | locate의 답 그대로 |
| key 위치 | **value_last** | ROME 주어-마지막-토큰의 직접 대응 |
| R | **64 고정** | 4방법 모두 동일한 64개 요청 (ROME은 순차 적용, 나머지 배치) |

**→ 본 실행 = 9 케이스 × 4 방법 = 36런.**

**유보된 ablation (조건부·후속)**:
- 층/key위치 변형(전이 구간 층 L13~15, name_last, random층 illusion 체크): 본 실행에서
  이상 신호(피크층 locality 붕괴 등)가 나올 때만 case 1 한정으로 시험 (+4~8런)
- **R 효율성 곡선** (합성 데이터 사용량 요구사항): 본 실행 후 case 1 × 4방법 ×
  R{1,16,256} (+12런)으로 대표 곡선 확보

**편집 입력 데이터** (locate의 θ-고정+파트너 컨텍스트 로직 재사용):
편집 요청 = 플립존 (θ′, θ] 값 R개 / C용 일반 프롬프트 ~2k / K₀용 보존 프롬프트
(target rule 안전존 + 다른 rule 문맥) ~2k. 전부 합성 (방화벽 준수).

**실행 단계**: case 1 × 4방법 파일럿(파이프라인 검증) → 나머지 8케이스 (같은 설정이므로
파일럿이 곧 본 실행의 일부).

**저장 전략**: 편집 모델(2.2GB)은 저장하지 않음 — run_edit.py가 메모리에서
편집→평가 후 **수정 층 W₂ 델타(수 MB) + 지표만** `result/edit/`에 저장.

## Phase C. 구현

- **자체 경량 구현** 권장 (EasyEdit은 decoder 전제가 코드 전반에 박혀 있어 포팅 비용 > 재구현 비용).
  4개 방법은 동일한 수학 코어(C, k, v 계산)를 공유하므로 공통 모듈 + 방법별 update rule로 구성.
- 대상 모듈 명칭 (HF XLM-R): `roberta.encoder.layer[l].intermediate.dense` (fc1),
  `roberta.encoder.layer[l].output.dense` (fc2 = 편집 대상 W).
- 파일 구조:
  ```
  src/editing/
  ├── generate_locate_pairs.py  # A: rule별 counterfactual 쌍 생성 (10k/rule) [완료]
  ├── locate.py                 # A: activation patching + 예산별 히트맵 [완료]
  ├── kv.py                     # B: 편집 요청 샘플 생성, k*, v*, C, K0 계산 (공통)
  ├── editors.py                # C: rome/memit/emmet/alphaedit update rules
  ├── run_edit.py               # C→D: 편집 → 즉시 평가 → W₂ 델타·지표 저장 (단일 CLI)
  shell/
  ├── run_locate_all.sh         # 9 rule locate (GPU 0,1,2 분산) [완료]
  └── run_edit_all.sh           # 9 케이스 × 4 방법 × 층/R 스윕
  result/
  ├── locate/rule{1..9}/n{1000..10000}/   # 90개 locate 결과 [완료]
  └── edit/{method}/case{i}_L{l}_{kpos}_R{r}/   # w2_delta.pt + metrics.json + 분해 지표
  ```
- GPU: 0/1/2 사용 (3번은 타 작업 점유). 모델 ~2.2GB라 층 스윕·배치 편집 모두 단일 GPU로 충분.

## Phase D. 평가 프로토콜

케이스 × 방법 × R 조합마다 편집된 모델을 `src/models/test.py`로 평가:

1. **Efficacy** — 플립셋(~4.5천): 새 라벨 정확도. `result/edited_rule_25pct/case{i}` 베이스라인
   (pre-edit, 예상 ~0%)과 대비.
2. **Locality** — retain셋(~33.7천): 편집 전 예측 유지율. 특히 **다른 8개 rule의 decisive 샘플**을
   rule별로 쪼개 리포트 (rule j 경계가 다쳤는지 각각 확인).
3. **Generalization** — 새 θ′ 중심으로 신규 합성한 경계 근처 샘플
   (`generate_train_gaussian.py`를 θ′ 중심으로 재활용): 편집이 "4.5천 쌍 암기"가 아니라
   실제 경계 이동인지 판정.
4. **데이터 효율성**: 방법별 efficacy/locality vs 합성 샘플 예산 N 곡선 (핵심 비교 축).
5. 종합: 9×4 테이블 (efficacy / locality / generalization) + N별 곡선.

**주의 — 평가/편집 방화벽**: 1·2번의 test.csv 기반 셋은 결과 측정에만 쓰며,
편집 과정(층 선택·k\*·v\*·C·K₀·하이퍼파라미터 튜닝 포함)에 절대 노출하지 않는다.
하이퍼파라미터 선택이 필요하면 합성 held-out 셋으로 한다.

> **이력**: case 1 파일럿 진단(H1 문맥·H2 층·λ)은 test.csv 지표를 보며 진행됨 —
> 파이프라인 불능 원인 규명 단계라 불가피했으나, 이 과정은 "설정 탐색"으로 격리한다.
> 본 실행 절차: ① 확정 후보 설정의 하이퍼파라미터를 **합성 검증셋**(random_pass 문맥
> 플립존/안전존 held-out)으로 고정 → ② 9케이스 test.csv 수치는 고정 설정으로 1회 측정.
> case 1은 탐색에 노출되었음을 결과 보고에 명시.

**성공 기준(초안)**: efficacy ≥ 90%, retain 하락 ≤ 1%p, generalization이 efficacy와 동급.

## Phase E. 사후 분석

- 편집 성공 케이스에 대해 **re-tracing**: θ′ 근처 쌍을 새로 생성해 **새 경계 등장** 확인
  + 기존 θ 쌍에서 **옛 경계 소멸** 확인 (편집 전엔 θ′ 쌍이 둘 다 통과라 무신호 — 대조 성립).
- 업데이트 노름·층 분포 비교 (방법 간).
- 은행 rule(케이스 5–8, 동일 은행 플립) vs 기업 rule의 난이도 차이.
- random-layer 대조군 결과와 tracing 지도의 정합성.

## 일정 (초안)

| 주차 | 내용 |
|---|---|
| 1주 | minimal_pairs + trace.py 구현, 9개 rule tracing 완료 → 반응 지도 |
| 2주 | kv.py + ROME 첫 동작 (케이스 1개 end-to-end) → 나머지 3개 방법 확장 |
| 3주 | 전체 9×4×R 실행 + 평가 테이블 + 사후 분석 |

## 리스크

- **연상 편집이 연속 경계를 못 옮길 가능성** — 그 자체가 유의미한 발견 (batch R 스윕이 완화책이자 측정 도구)
- **숫자 토큰화 파편화** — 값 토큰 전체 span을 patching/key 후보로 취급해 완화
- **후반 층 `<s>` 편집의 locality 붕괴** — 층별 trade-off를 명시적으로 리포트.
  verdict 붕괴가 끝난 층은 플립존/안전존이 동일 표현이라 편집 시 안전존까지 뒤집힐 수 있음
  — 의심되면 해당 층에서 "값>θ′?" linear probe로 분리 가능 여부 점검 (보조 진단)
- **illusion 재현** (위치 무관하게 편집 성공) — random-layer 대조군으로 검출, 발견이면 그대로 보고

## 참고 문헌

- ROME: Meng et al., Locating and Editing Factual Associations in GPT (2022)
- MEMIT: Meng et al., Mass-Editing Memory in a Transformer (2023)
- EMMET: Gupta et al., A Unified Framework for Model Editing (2024)
- AlphaEdit: Fang et al., Null-Space Constrained Knowledge Editing (2025)
- 주의점: Makelov et al., An Interpretability Illusion for Subspace Activation Patching (ICLR 2024);
  Grant et al., Addressing Divergent Representations from Causal Interventions (ICLR 2026)
