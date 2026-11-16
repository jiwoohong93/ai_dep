# result/locate — Locate(causal tracing) 결과

생성: `src/editing/locate.py` (데이터셋·방법은 `data/rules/edited_rule_25pct/locate/README.md` 참고)

## 폴더 구조

```
result/locate/rule{1..9}/
├── pair_ie.npz                 # 쌍별 원본 IE [10,000쌍 × 24층 × 6그룹], p2f/f2p 배열 + valid 마스크
└── n{1000,2000,...,10000}/     # locate 예산별 집계 (앞에서부터 n쌍 중 유효쌍 평균)
    ├── heatmap_p2f.csv         # 24행(layer_0~23) × 6열(위치 그룹) — 통과→탈락 방향 IE
    ├── heatmap_f2p.csv         # 〃 탈락→통과 방향
    └── meta.json               # budget, n_valid_used, peak_layer_*, memit_window_*
```

## 히트맵 읽는 법

- **행 = 층** (layer_0~layer_23): 해당 encoder 층의 출력(residual stream)을 patch
- **열 = 위치 그룹**: 문장에서 어느 토큰(들)을 patch했는가
- **값 = IE(indirect effect) 평균**: patch가 최종 예측 확률을 얼마나 뒤집었는가.
  1.0 ≈ 그 (층, 위치)가 판정을 전부 운반 / 0 ≈ 무관

**p2f (pass→fail)**: **라벨 1(통과) 문장에 라벨 0(탈락) 문장의 내부 상태를 주입**하고,
예측이 탈락 쪽으로 넘어가는지를 본다. IE의 뺄셈은 라벨끼리 비교가 아니라
**같은 문장의 patch 전 vs 후** 비교다:

```
target rule = 부채비율 ≤ 200 인 경우:
  base   문장: 부채비율 190 → 라벨 1 (통과), 모델 예측도 통과: P(탈락) = 0.01
  source 문장: 부채비율 210 → 라벨 0 (탈락)

p2f patch: base(통과) 문장을 돌리다가 (층 l, 위치 p) 상태만 source 것으로 교체
  → "통과 문장인데 예측이 탈락으로 바뀌었나?"

IE_p2f(l, p) = P(탈락 | patch 후의 base 문장) − P(탈락 | patch 전의 base 문장)
                └── 같은 base 문장, patch 후 ──┘   └── patch 전 (≈0.01) ──┘

  layer 21, 값 토큰 교체 → P(탈락) 0.01 → 0.99, IE = 0.98 ≈ 1  (판정 운반)
  layer 3, <s> 교체     → P(탈락) 0.01 → 0.01, IE = 0        (무관)
```

**f2p (fail→pass)**: 반대 — 라벨 0(탈락) 문장에 라벨 1(통과) 상태를 주입,
예측이 통과로 살아나는지:

```
IE_f2p(l, p) = P(통과 | patch 후의 source 문장) − P(통과 | patch 전의 source 문장)

  원래 source(210) 문장: P(통과) = 0.02
  layer 14, 값 토큰을 base(190) 것으로 교체 → P(통과) = 0.95, IE = 0.93
```

**rule 유형별 base/source 방향** — 통과측이 rule마다 다르므로 쌍 구성도 반대가 됨:

| rule 유형 | 예 | base (통과, 라벨 1) | source (탈락, 라벨 0) |
|---|---|---|---|
| ≤ (이하) | 부채비율 ≤ 200 | 190 (경계 아래) | 210 (경계 위) |
| ≥ (이상) | 유동비율 ≥ 100 | 106 (경계 위) | 94 (경계 아래) |
| > (양수) | 자본총계 > 0 | +420,000 | −380,000 |

**rule 변화(θ→θ′)와 patching의 관계** — 이 결과(`result/locate`)는 전부 **편집 전 모델,
현행 θ 기준** 쌍이다. 편집 후에는 같은 patching을 두 위치의 쌍으로 다시 돌려 경계 이동을
내부에서 확인한다 (re-tracing, 예: 부채비율 200→85 편집):

| 쌍 위치 | 편집 전 | 편집 후 (성공 시) |
|---|---|---|
| θ=200 근처 (190↔210) | IE 신호 ✓ (현행 경계) | 190도 210도 탈락 → 신호 소멸해야 |
| θ′=85 근처 (80↔90) | 둘 다 통과 → 신호 없음 | 새 경계 생김 → IE 신호 등장해야 |

## 열(그룹) → 토큰 인덱스 매핑 실례

모든 그룹은 **target rule**(그 히트맵이 추적하는 rule)을 기준으로 정의된다.
아래는 target rule = rule 6(기본자본비율)인 경우, pair_id 0의 base 문장을 실제
토크나이저에 통과시킨 결과. **인덱스는 문장마다 달라지며**, 매 문장에서
문자 span → offset mapping으로 새로 계산됨:

```
문장: "어떤 은행의 보통주자본비율이 4.5%, 기본자본비율이 6.5%, 총자본비율이 8.09%, ..."
총 139 토큰

value_last: [22]                 → ['5%']               # target rule 값(6.5)의 마지막 토큰
value_all : [21, 22]             → ['▁6.', '5%']        # target rule 값의 전체 토큰
name      : [15, 16, 17, 18, 19] → ['▁기본','자','본','비','율']  # target rule의 지표명 라벨
cls       : [0]                  → ['<s>']              # 분류 헤드가 읽는 위치
q_last    : [137]                → ['?']                # 문장 끝 질문의 마지막 토큰
ctrl_roa  : [62]                 → ['▁0.3']             # ROA 값 (distractor 대조군)
```

숫자 토큰화 참고: "6.5%"가 `▁6.` + `5%`로 쪼개지는 등 sentencepiece 분할은 값에 따라
달라짐 — 절대 인덱스가 아니라 그룹 단위로 다뤄야 하는 이유.

## meta.json 키

| 키 | 의미 |
|---|---|
| rule / column | rule 번호 / 지표 컬럼명 |
| budget | 사용한 locate 예산 (쌍 파일의 앞에서부터 n개) |
| n_valid_used | 그중 실제 집계된 유효쌍 수 (모델이 base→1, source→0 정답 예측한 쌍) |
| peak_layer_p2f | p2f 방향, value_last 그룹 기준 \|IE\| 최대 층 — **편집 층 l\* 후보 (주 기준)** |
| memit_window_p2f | [피크−2, 피크+2] — MEMIT/AlphaEdit용 층 구간 |
| peak_layer_f2p / memit_window_f2p | 역방향 기준 (분석용 — 편집은 조이는 방향이라 p2f 사용) |

## 이력

- `result/locate_random_context/`: 초기 실행 (컨텍스트 랜덤 통과값). 상관 지표 모순으로
  rule 3·6·7 유효쌍 붕괴 → 현 버전(θ-고정+파트너)으로 대체. 비교용으로 보존.
- rule 6은 name 그룹 span 버그(지표명 검색이 `단순기본자본비율` 내부를 잡던 문제) 수정 후
  재실행된 결과.
