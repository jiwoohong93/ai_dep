# result/edit_main — 편집 베이스라인 본 실행 결과

생성: `src/editing/run_edit.py` / 일괄 실행: `shell/run_edit_all.sh`
계획·설정 근거: [docs/edit_baseline_plan.md](../../docs/edit_baseline_plan.md)

## 본 실행 확정 설정 (9 케이스 × 4 방법 = 36런, 방법 간 동일 조건)

| 항목 | 값 | 비고 |
|---|---|---|
| context_mode | `random_pass` | 요청/C/K₀ 문맥을 통과값 랜덤으로 — 파일럿에서 합성→실제 전이의 결정 변수로 판명 |
| 편집 층 | rule별 locate p2f 피크 | MEMIT/AlphaEdit은 [피크−2, min(피크+2, 22)] 구간 |
| R (편집 요청 수) | 64 | 플립존 값 샘플 (금액 rule은 log-uniform) |
| λ (보존 가중) | 4 | MEMIT/AlphaEdit의 ridge 가중 |
| ⚠ case 1 | 설정 탐색(H1/H2/λ)에 노출됨 | 나머지 8케이스는 이 설정의 첫 측정 |

폴더: `{method}/case{i}_L{층}_R{64}_random_pass/` — `metrics.json` + `w2_delta.pt`(수정 층 W₂ 델타, git 미추적)

## 용어 정의 (편집 파이프라인의 재료들)

ROME 계열은 층의 MLP 두 번째 선형층(W₂ = `output.dense`)을 **key→value 사전**으로
본다: 입력 패턴 k가 오면 내용 v = W₂k 를 출력에 써넣는다. 편집 = 이 사전의 항목 고쳐쓰기.

| 용어 | 정의 | 우리 케이스의 예 (case 1, 부채비율 200→85) |
|---|---|---|
| **편집 요청** | "이 입력 패턴은 이제 이 답" 지시 1건 | "부채비율 120.5인 문장 → 탈락(0)" |
| **k\* (key)** | 요청 문장을 모델에 넣고 편집 층에서 **측정**한 값-마지막-토큰의 MLP 내부 활성 (GELU(fc1x), 4096차원) | "부채비율이 ~120쯤"이라는 신경 패턴 |
| **v\* (value)** | 그 위치의 MLP 출력을 뭘로 바꿔치면 분류가 0이 되는지 **경사하강으로 탐색**한 목표 벡터 (1024차원) | "그 값이면 탈락"이라는 내용 |
| **C (공분산)** | 일반 프롬프트 수백 개의 전 토큰에서 잰 key 통계 E[kkᵀ]. "평소 자주 오는 key 방향" — 편집 수식이 이 방향들을 **피해서** 고치게 하는 보존 장치 | 9개 rule 주변값 문장 512개에서 측정 |
| **K₀ (보존 키)** | 절대 안 바뀌어야 할 입력들의 key 목록 (AlphaEdit 전용). 편집을 이들의 **null-space**(K₀와 수직인 방향)로 사영해 ΔK₀=0을 수학적으로 보장 | 부채비율 50(안전존) 문장들 + 다른 8개 rule 문장들의 key ~1천 개 |
| **피크 층** | locate(causal tracing)에서 p2f 방향 간접효과(IE)가 최대인 층 — "이 rule의 판정을 가장 강하게 운반하는 층". `result/locate/rule{i}/n10000/meta.json`의 `peak_layer_p2f` | case 1은 L21, case 7은 L12 |
| **memit_window** | MEMIT/AlphaEdit이 갱신을 분산하는 층 구간 [피크−2, min(피크+2, 22)] | case 1은 L19~22 |

## 설정 선택 이유 (case 1 파일럿 탐색의 결론)

**왜 case 1로 탐색했나**: 부채비율은 locate가 가장 깨끗했던 rule(유효쌍 100%, 피크 1k부터
안정, 전형적 퍼센트 rule)이라 "여기서 안 되는 설정은 다른 데서도 안 된다"는 관문 역할.
대신 case 1 수치는 탐색에 노출됐으므로 최종 보고에서 별도 표기.

**context_mode = random_pass — 왜**: 편집 요청 문장의 **나머지 12개 지표를 뭘로 채우나**의
선택. 파일럿에서 이것이 편집 성패를 갈랐다:

```
fixed_partner (locate와 동일, θ-고정):
  "...보통주자본비율이 4.5%, 기본자본비율이 6.0%, ... 부채비율이 120.5%..."
  → 요청은 100% 충족되지만 실제 test.csv 플립셋에선 efficacy 0~29%.
    key에 문맥 서명이 함께 박혀서, 실제 은행-기업 문장의 key에는 편집이 발화 안 함.

random_pass (통과값 랜덤):
  "...보통주자본비율이 17.42%, 기본자본비율이 19.88%, ... 부채비율이 120.5%..."
  → 요청 문맥이 실데이터 다양성에 가까워져 efficacy 0→77% (MEMIT 기준) 급등.
```

locate에서는 랜덤 문맥이 부적합했지만(모델 정답 필터 때문), 편집 요청은 모델이
문장을 맞힐 필요가 없어(목표 행동을 덮어쓰는 것) 랜덤 문맥의 부작용이 없다.

**R = 64 — 왜**: threshold 편집은 사실 하나가 아니라 **연속 구간**(86~200 전체)의 변경이라,
플립존을 훑는 값 64개를 요청 배치로 넣는다. 예: {87.3, 95.1, 110.4, ..., 198.2}. 너무
적으면 구간 커버가 성기고(사이 값이 안 뒤집힘), 너무 많으면 합성 데이터 예산 낭비 —
64는 중간값이며, 데이터 효율성 곡선(R∈{1,16,256})은 후속 실행으로 확보 예정.

**λ = 4 — 왜**: MEMIT/AlphaEdit 수식 `Δ = Res·Kᵀ(λC + KKᵀ)⁻¹`에서 **보존(λC) ↔ 충족(KKᵀ)의
저울추**. λ↑ = "평소 key 방향을 더 조심해서 고쳐라". case 1 스윕 결과 λ 0.25→4에서
locality 96.5→97.9% 개선, efficacy 손실 ~1%p 미만이라 4로 고정. (ROME/EMMET은 λ를
쓰지 않음 — C⁻¹만 사용하는 수식이라 λ가 소거됨.)

**층 = rule별 p2f 피크 — 왜**: "locate가 가리킨 곳을 고친다"는 기본 가설의 직접 구현.
단, case 7(피크 L12)에서 이른 층 편집이 불발하는 문제가 관측됨 — locate 피크 ≠ 편집
최적층일 수 있다는 증거로, 사후 분석 대상.

## metrics.json 지표 설명

**편집 자체 검증 (합성, 평가셋 아님):**

| 키 | 의미 |
|---|---|
| `edit_request_success` | 편집 요청 문장 R개가 실제로 탈락(0) 예측으로 바뀐 비율 |
| `delta_norm` | 층별 W₂ 변경량 노름 — 편집이 얼마나 크게 가중치를 고쳤나 |

**실제 평가 (test.csv 38,160행, dropna 후 — 편집 과정에 미노출):**

| 키 | 의미 |
|---|---|
| `overall_acc` | test_case{i}.csv 전체 정확도 (새 라벨 기준) — test.py가 주는 그 숫자 |
| `efficacy_flip_acc` | **플립셋**(라벨 1→0으로 바뀐 ~4.5천 행)에서 새 라벨 정확도. 편집 전 0% — "바뀐 rule에 대한 결정을 바꿨는가" |
| `locality_retain_acc` | **retain셋**(라벨 불변 ~33.7천 행) 정확도. 편집 전 100% — "다른 예측은 그대로인가" |
| `locality_pred_unchanged` | retain셋에서 편집 전 예측과 동일한 비율 (행 단위 대조) |
| `n_flip` / `n_retain` | 플립/retain 행 수 |

**편집 전 대비 변화 (pre-edit predictions.csv와 대조):**

| 키 | 의미 |
|---|---|
| `preedit_overall_acc` | 편집 전 모델의 test_case{i} 전체 정확도 (= 1 − 플립 비율) |
| `overall_acc_change` | 전체 정확도 개선폭 (+면 편집 이득) |
| `preedit_retain_acc` | 편집 전 retain 정확도 (≈1.0) |
| `retain_acc_drop` | **원래 accuracy 대비 retain 하락률** — 편집의 부수 피해 크기 |

**경계 generalization (합성 2k, 편집에 쓰지 않은 신규 문장 — θ-고정 문맥):**

| 키 | 의미 |
|---|---|
| `gen_flip_acc` | 새 합성 플립존 1k가 0으로 예측되는 비율 — "요청 암기가 아니라 경계가 이동했는가" |
| `gen_safe_acc` | 새 합성 안전존 1k가 1로 유지되는 비율 — "경계가 θ′를 지나쳐 과도하게 밀리지 않았는가" |

주의: gen 셋 문맥은 θ-고정+파트너라서 random_pass로 편집한 모델에는 보수적(낮게 나오는)
측정임 — 실제 전이는 위의 test.csv 지표가 기준.

## 읽는 요령

- 좋은 편집 = `efficacy_flip_acc` 높음 **그리고** `retain_acc_drop` 작음 (동시에)
- `overall_acc`만 보면 속을 수 있음: "전부 탈락" 엉터리 편집도 ~62%가 나옴
- 방법 간 비교는 같은 케이스 안에서 (케이스마다 플립존 난이도가 다름)

## 이력 (파일럿 — result/edit_pilot/, 설정 탐색 기록)

case 1 한정 설정 탐색 13런. 폴더명 규칙: `case1_L{층}_R64[_random_pass][_lam{λ}]`
(접미사 없으면 θ-고정 문맥·λ=1)

- θ-고정 문맥 (L21/L15): 합성→실제 전이 실패 진단 자료 (efficacy 0~29%)
- `*_random_pass`: 문맥 다양화 H1 검증 (efficacy 77~99.9%로 급등)
- `*_lam025`/`*_lam4`: MEMIT λ 스윕
- 결론: 문맥 다양성(H1)이 결정 변수, 층(H2)은 부차적, λ는 완만한 locality 손잡이

## w2_delta.pt 재적용 방법

델타는 "편집된 W₂ − 원본 W₂" (편집 층만, fp16). 원본 모델에 더하면 편집 모델이 복원됨:

```python
import torch
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained("yongukpark/xlm-roberta-large-rule")
deltas = torch.load("result/edit_main/memit/case4_L22_R64_random_pass/w2_delta.pt")
for layer, delta in deltas.items():
    fc2 = model.roberta.encoder.layer[int(layer)].output.dense
    fc2.weight.data += delta.float().to(fc2.weight.device)
# 이후 이 모델로 추론하면 편집된 모델과 동일 (fp16 반올림 오차 무시 수준)
```
