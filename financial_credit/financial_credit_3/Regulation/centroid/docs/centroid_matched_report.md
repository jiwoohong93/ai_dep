# Centroid matched 실험 보고서 — locate까지 100% 합성 + locate↔edit 레시피 일치

2026-07-18 실행. 결과: `Regulation/centroid/result/centroid_matched/`
(기존 `result/centroid/`(v1/v3/v4), `result/centroid_v2/`와 계보를 분리한 신규 폴더)

## 1. 왜 다시 했나 — 기존 계보의 두 가지 문제

**문제 1: locate에 test.csv가 들어갔다.** 기존 편집 지점(layer 21 × 부채비율 토큰)은
v1 locate가 정했는데, 그 A/B 그룹 500/500을 최종 채점용 실데이터 test.csv에서 뽑았다.
v2가 delta_v·gate를 합성으로 다시 만들면서도 **layer 21이라는 위치 자체는 v1 결론을
그대로 상속**했기 때문에, "편집 재료 제작에 실데이터 금지" 원칙이 위치 선정 단계에서
깨진 채로 남아 있었다.

**문제 2: locate와 edit의 데이터 레시피가 달랐다.** v3a/v3b/v3b2는 gate 학습 데이터의
문맥 구성(좁게/넓게/노이즈 혼합)이 결과를 좌우함을 보여줬는데, 정작 위치(L21)와
delta_v는 전 버전이 다른 레시피(v1/v2 유래)의 것을 공유했다.

**이번 원칙 (2026-07-18 확정)**: ① locate도 100% 합성 데이터로, ② 한 버전 안에서는
locate 데이터·delta_v·gate·alpha 보정이 전부 **같은 레시피**의 데이터를 쓴다
(locate를 v3a로 했으면 edit도 v3a로). test.csv는 최종 채점에서 딱 한 번만 연다.

## 2. 절차 — 예시로 따라가기

규칙 시나리오는 동일: 부채비율 **"200% 이하 통과" → "85% 이하만 통과"** 강화.
편집 대상은 85~200% 구간("밴드", 예: 부채비율 150%인 회사 A)이고, 60%인 회사 B와
250%인 회사 D는 건드리면 안 된다.

### 2-1. Locate — 합성 쌍으로 50개 지점 스캔 (`centroid_locate_synth.py`)

후보는 layer 0~24 × 자리 2곳(CLS / 부채비율 숫자 토큰) = 50개 지점.

**데이터**: "같은 문맥에서 부채비율만 통과값/탈락값으로 바꾼" 합성 쌍 500개.
예를 들어 한 쌍은 이렇게 생겼다 (v3b 문맥 기준):

```
A행: (부채비율 183.4 ←통과측, 유동비율 327, 자기자본비율 61, ...)
B행: (부채비율 216.9 ←탈락측, 유동비율 327, 자기자본비율 61, ...)  ← 문맥 완전 동일
```

문맥(나머지 12개 지표)을 만드는 레시피가 버전마다 다르고, **해당 버전의 gate 학습
데이터와 정확히 같은 레시피**를 쓴다:

| locate 문맥 | 레시피 | 대응 edit 버전 |
|---|---|---|
| pairs | 기존 최소쌍 `data/rules/edited_rule_25pct/locate/rule1.csv` (partner-cycle 좁은 문맥, v2 delta_v의 출처) | (기준선 — 대응 edit 없음) |
| v3a | `gate_v3a/data/`(gaussian+interpolation) 파일에서 문맥 샘플링. 단 **decisive(타 rule 전부 통과) 문맥만** — 타 rule이 탈락인 문맥에선 부채비율이 판정을 못 바꿔 locate 전제가 붕괴 (실제로 필터 없이 돌린 1차 실행은 A그룹 정답 예측이 31/100뿐이라 무효 처리 후 재실행) | v3a |
| v3b | 다른 8개 rule 지표를 각자의 안전존 전체에서 균일 샘플링 + 정보성 4개는 넓은 gaussian | v3b |
| v3b2 | v3b + 25% 행은 다른 rule 1~2개를 일부러 탈락값으로 (probe가 "노이즈 속에서도 부채비율을 읽는지" 검증; 단 그런 행은 flip 실험 전제가 안 되므로 patching은 clean 행만) | v3b2 |

**1단계 probe**: 지점마다 독립적으로, A/B 벡터 1,000개(500+500)로 로지스틱 회귀를
학습해 70/30 held-out 정확도 측정 → "이 지점에 통과/탈락 정보가 담겼나".

**2단계 activation patching**: 회사 A₁(부채비율 150, 판정 통과)이 layer L을 지날 때
부채비율 토큰 벡터를 (a) 탈락 그룹 평균 μ_탈락으로(centroid), (b) 같은 문맥의
counterfactual 상대 벡터로(paired) 교체하고 최종 판정이 뒤집히는 비율을 측정.
양방향(통과→탈락 p2f / 탈락→통과 f2p) × 100건.

**3단계 선별**: probe ≥ 0.95 그리고 양방향 centroid flip ≥ 0.70 → 후보.
probe ≥ 0.95인데 양방향 flip < 0.30 → 해리(정보만 있고 조종 안 함, 배제 목록).

### 2-2. Locate 결과 — 네 문맥 모두에서 L21 탈락, 진짜 후보는 L6~15

| 문맥 | debt_span 후보 layer | 해리 지점 수 | L21 실측 (probe / flip p2f / flip f2p) |
|---|---|---|---|
| pairs | 6~13 | 24 | 1.00 / **1.00** / **0.00** → 탈락 |
| v3a | 6~13 | 21 | 1.00 / 1.00 / 0.00 → 탈락 |
| v3b | 7~15 | 18 | 1.00 / 1.00 / 0.00 → 탈락 |
| v3b2 | 7~15 | 17 | 1.00 / 1.00 / 0.00 → 탈락 |

**핵심 발견**: 기존 편집 지점 L21은 **통과→탈락 방향으로만 100% 뒤집히고 탈락→통과
방향은 0%** — "판정을 양방향으로 들고 있는 지점"이 아니라 한쪽으로만 밀리는
지점이었다 (paired 주입도 동일: 1.00/0.00). 이 결론이 문맥 레시피 4종 전부에서
동일하게 재현된다. 양방향이 모두 작동하는 진짜 후보는 **layer 6~15의 부채비율 토큰
구간**이다 (예: v3b의 L13은 probe 1.00, p2f/f2p 모두 1.00/1.00, paired도 동일).

### 2-3. Edit — 버전별 전 구성요소를 같은 레시피로 (`centroid_edit_matched.py`)

버전마다:

1. **layer 선정** (자의성 제거를 위해 규칙 고정): 그 버전 locate의 debt_span 후보 중
   min(양방향 flip) 최대, 동률이면 층 번호 중앙값.
   → v3a: **L10**, v3b: **L13**, v3b2: **L10**
2. **delta_v**: 같은 locate 실행의 캐시 hidden에서 그 layer의 μ(탈락) − μ(통과).
   예: v3a는 L10에서 norm 8.1 (기존 v2의 L21 delta_v norm 23.2와 다른 벡터).
3. **gate**: 그 버전의 gate 데이터(v3a는 `gate_v3a/data` 파일 재사용, v3b/v3b2는 동일
   생성기·동일 라벨 규칙)를 그 layer에서 학습. 합성 held-out 0.988~0.998.
4. **alpha 보정**: 합성만, 스윕 {0.5, 1.0, 2.0}, 안전 오탐 ≤ 2% 제약 내 밴드 flip
   최대 → 세 버전 모두 2.0 선정.
   예: v3b 보정 곡선 — alpha 0.5는 밴드 flip 0.000, 1.0은 0.893, 2.0은 1.000
   (안전 오탐 0.007).
5. **평가**: test.csv 38,160건 전체를 hook 켠 상태로 예측 (여기서만 실데이터).

예시로 회사 A(부채비율 150, v3b 버전) 흐름: L13 도달 → gate(w·h+b) 양성 →
h += 2.0 × delta_v(L13) → 최종 판정 통과→탈락. 회사 B(60%)는 gate 음성 → 무개입.

## 3. 최종 성능 (test.csv 38,160건, 새 라벨 = 밴드→탈락)

### 버전 일치(matched) 결과 — recall 포함

| 버전 | layer | alpha | acc | macro-F1 | recall(통과) | recall(탈락) | macro recall | 반영률 | 보존율 | cross 보존 |
|---|---|---|---|---|---|---|---|---|---|---|
| v3a matched | 10 | 2.0 | 0.9537 | 0.9498 | 0.8797 | 0.9999 | 0.9398 | **0.9994** | 0.8797 | 1.0000 |
| v3b matched | 13 | 2.0 | 0.9849 | 0.9840 | 0.9829 | 0.9860 | 0.9845 | 0.9324 | 0.9829 | 1.0000 |
| **v3b2 matched** | 10 | 2.0 | **0.9854** | **0.9846** | 0.9969 | 0.9782 | **0.9875** | 0.8944 | **0.9969** | 1.0000 |

(반영률 efficacy = 밴드 회사가 탈락으로 바뀐 비율, 보존율 locality_safe = 85% 이하
회사가 통과를 유지한 비율, cross = 타 rule 탈락 회사의 판정 보존율.
recall(통과)는 새 라벨 기준 통과(1) 클래스 재현율 — 보존율과 수치가 사실상 일치하고,
recall(탈락)은 밴드+기존탈락을 합친 탈락(0) 클래스 재현율.)

### 기존(L21 + v2 delta_v) 결과와 대조

| 버전 | 계보 | acc | 반영률 | 보존율 |
|---|---|---|---|---|
| v3a 기존 | L21 상속 | 0.954 | 0.701 | 0.980 |
| v3a matched | **L10, 전체 일치** | 0.954 | **0.999** | 0.880 |
| v3b 기존 | L21 상속 | 0.910 | 0.891 | 0.802 |
| v3b matched | **L13, 전체 일치** | **0.985** | **0.932** | **0.983** |
| v3b2 기존 | L21 상속 | 0.951 | 0.871 | 0.915 |
| v3b2 matched | **L10, 전체 일치** | **0.985** | **0.894** | **0.997** |
| (기존 최고: v4 앙상블) | L21 상속 | 0.970 | 0.836 | 0.977 |

**v3b·v3b2 matched(acc 0.985)가 기존 최고인 v4 앙상블(0.970)을 단일 gate로 넘었다.**
계보를 깨끗하게 만들기 위한 재실험이 성능까지 올린 셈인데, 이유는 중간층(L10~13)이
판정을 양방향으로 실제로 들고 있는 지점이라 같은 delta_v·gate가 더 정확하게 작동하기
때문으로 해석된다. v3a는 여전히 "반영률↑ vs 보존율↓" 트레이드오프의 반대쪽 끝을
보여준다 (기존 v3a와 방향이 뒤집힘 — L10에선 문맥 좁은 gate가 과발동 쪽으로 틀림).

## 4. 한계 / 다음 단계

- **alpha=2.0이 스윕 그리드의 끝값** — {0.5, 1, 2}에서 2.0이 선정됐으므로 3~4까지
  확장하면 v3b/v3b2의 반영률(0.93/0.89)이 더 오를 여지가 있다.
- v3a matched의 보존율 0.880 — 좁은 문맥 gate의 도메인 갭은 여전히 남은 문제.
- matched gate들의 v4식 확률 평균 앙상블은 아직 안 돌림 (기존 v4가 앙상블로 이득을
  본 것처럼 v3a(과발동)+v3b2(보수) 조합 여지).
- rule1(부채비율)만 실행 — 다른 8개 rule로의 확장은 스크립트 인자(`--rule`)로 가능.
- v3a locate 1차 실행은 decisive 필터 누락으로 무효였고 재실행본만 유효 (§2-1 표 참고).

## 5. 파일 위치 / 재현

```
Regulation/centroid/
├── src/editing/centroid_locate_synth.py     # locate (문맥 4종: pairs/v3a/v3b/v3b2)
├── src/editing/centroid_edit_matched.py     # 버전 일치 edit (locate→delta_v→gate→alpha→평가)
└── result/centroid_matched/
    ├── locate_synth_{pairs,v3a,v3b,v3b2}/   # 지점별 probe/flip CSV, locate_report.json,
    │                                        #  곡선 PNG, 생성 데이터(pairs_A/B.csv), manifest
    └── edit_matched_{v3a,v3b,v3b2}/         # matched_edit_report.json(위 표의 원본 수치),
                                             #  predictions.csv, shift_config.pt, manifest
```

재현 (conda `rule_edit`, `Regulation/centroid/src/editing/`에서):

```bash
CUDA_VISIBLE_DEVICES=<gpu> python centroid_locate_synth.py --rule 1 --context_mode v3b   # 문맥별 locate (~7분)
CUDA_VISIBLE_DEVICES=<gpu> python centroid_edit_matched.py --version v3b                 # 버전 일치 edit (~15분)
```

각 결과 폴더의 `manifest.json`에 실행 시각·입력 인자·데이터 md5·git 커밋이 기록된다.
locate 데이터가 바뀌면 md5가 달라지므로 리포트의 `locate_data_md5`로 어느 데이터에서
나온 위치인지 추적 가능.
