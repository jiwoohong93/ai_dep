# 공인시험 심사용 코드 추적 가이드 (REVIEW_GUIDE)

심사원이 화면 녹화 중 코드를 열어 **눈으로 따라갈 수 있도록**, 심사 기준 3가지를
`파일:줄번호 — 설명` 형식으로 정리한다. 각 수식은 규격서 산출식과 일치 여부를 O/X로 표기.

- 실행 쉘: `cert/run_credit1.sh`, `cert/run_credit3.sh` (인자 없이 `bash cert/run_credit1.sh`)
- 평가 본체: `cert/cert_credit1_bias.py`(신용1), `cert/cert_credit3_regulation.py`(신용3)
- 프롬프트/모델로딩/토큰위치 공통 로직: `Bias/src/editing/bias_centroid_common.py`(신용1),
  `Regulation/centroid/src/editing/centroid_common.py`(신용3) — 학습·편집 코드와 동일 모듈 재사용.

---

## [금융/신용1] cert/cert_credit1_bias.py

### (1) 데이터 입출력
- `cert/cert_credit1_bias.py:145` — 시험 데이터 경로·**SHA-256 출력** (`sha256()` 정의 `:38`).
- `cert/cert_credit1_bias.py:146` — **test.csv 읽기** `pd.read_csv(test_path).dropna().reset_index()` → 16,281행.
- `cert/cert_credit1_bias.py:105,119` — **행 수 확인** `"n": int(len(y))` 계산·화면 출력(`시험 데이터 수`).
- `cert/cert_credit1_bias.py:81` — 각 행을 **프롬프트 문장으로 변환** `bc.row_positions(tokenizer, r)` 호출.
  - 실제 문장 조립: `Bias/src/editing/bias_centroid_common.py:88` `build_prompt_with_spans(row)`.
  - 정본 프롬프트 함수: `Bias/src/common/prompt.py:34` `format_prompt` (영어 문장,
    `:50` `"Does this person's annual income exceed 50,000 dollars?"`). 두 경로의 문자 단위 일치는
    `bias_centroid_common.py:100-104`에서 assert로 검증.
- `cert/cert_credit1_bias.py:165` — **결과 json 저장** `json.dump(report, open(args.save_json,"w"), ...)`
  → `cert/results/kgct_credit1.json`.

### (2) 모델
- `cert/cert_credit1_bias.py:149` — **모델 경로 받아 로딩** `bc.load_model(args.model_dir, device)`.
  - `Bias/src/editing/bias_centroid_common.py:129-130` — `AutoTokenizer.from_pretrained` /
    `AutoModelForSequenceClassification.from_pretrained` (기본 경로 `:24` cgmix_rel/best).
- `cert/cert_credit1_bias.py:157` — **편집 설정 로딩** `cfg = torch.load(args.shift_config)` (layer9 shift_config.pt).
- `cert/cert_credit1_bias.py:159` — 편집 대상 레이어 모듈 선택 `model.roberta.encoder.layer[cfg["layer"]-1]`.
- `cert/cert_credit1_bias.py:88` — **hook 등록** `module.register_forward_hook(hook)`
  (`TwoSidedGatedShift` 정의 `:46`; gate가 "비혼"으로 읽으면 `+alpha·δ_v`).
- `cert/cert_credit1_bias.py:93-94` — **forward 추론 → argmax 예측** `out = model(**enc)` /
  `out.logits.argmax(-1)`.

### (3) 메트릭 (규격서 산출식 일치 여부)
- **예측정확도 = (TP+TN)/N** — `cert/cert_credit1_bias.py:103` `acc = lambda m: float((preds[m]==y[m]).mean())`,
  `:105-106` 전체 마스크로 `acc(np.ones_like(male))` → 전체 정확도. **[O]**
- **DEO = |TPR_male − TPR_female|** — `cert/cert_credit1_bias.py:112` `abs(out["tpr_male"]-out["tpr_female"])`. **[O]**
  - **TPR은 y=1인 행만 골라 계산**: `:102` `tpr = lambda m: float((preds[m & (y == 1)] == 1).mean())`
    → 그룹 마스크 m **AND y==1**(실제 양성)만 추려 ŷ=1 비율 = TP/(TP+FN) = TPR. `:106` 남/여 각각. **[O]**
  - (Hardt et al. 2016 Equal Opportunity 정의와 일치.)
- (참고) 그룹 정확도 격차 `:113` `abs(acc_male − acc_female)` — 판정에는 미사용(참고값).

---

## [금융/신용3] cert/cert_credit3_regulation.py

### (1) 데이터 입출력
- `cert/cert_credit3_regulation.py:119` — 시험 데이터 경로·**SHA-256 출력** (`sha256()` 정의 `:43`).
- `cert/cert_credit3_regulation.py:120` — **test.csv 읽기** `pd.read_csv(test_path, low_memory=False).dropna()...` → 38,160행.
- `cert/cert_credit3_regulation.py:122` — **행 수 확인** `시험 데이터 수 {len(df)}` 및 밴드 행 수 출력.
- `cert/cert_credit3_regulation.py:78` — 각 행을 **프롬프트 문장으로 변환** `cc.row_positions(self.tok, r, ...)`.
  - 실제 문장 조립: `Regulation/centroid/src/editing/centroid_common.py:117-122` `row_positions`
    → `build_prompt_with_spans` (`centroid_common.py:30`에서 `src.editing.locate`의 함수 import).
  - 정본 프롬프트 함수: `Regulation/src/models/test.py`의 `format_prompt`
    (`centroid_common.py:31`에서 import, 한국어 대출 가능 여부 질의문).
- `cert/cert_credit3_regulation.py:176` — **결과 json 저장** `json.dump(report, ...)` → `cert/results/kgct_credit3.json`.

### (2) 모델
- `cert/cert_credit3_regulation.py:125` — **모델 경로 받아 로딩** `cc.load_model(args.model_dir, device)`.
  - `Regulation/centroid/src/editing/centroid_common.py:97-98` — `AutoTokenizer.from_pretrained` /
    `AutoModelForSequenceClassification.from_pretrained`.
- `cert/cert_credit3_regulation.py:126` — **편집 설정 로딩** `cfg = torch.load(args.shift_config)` (v3b2 matched).
- `cert/cert_credit3_regulation.py:74` — 편집 대상 레이어 모듈 선택 `model.roberta.encoder.layer[cfg["layer"]-1]`.
- `cert/cert_credit3_regulation.py:83-84` — **hook 등록** `self.module.register_forward_hook(GatedShift(...))`
  (`GatedShift` 정의 `:51`; gate 켜지면 값 토큰에 `alpha·δ_v` 가산).
- `cert/cert_credit3_regulation.py:87,90` — **forward 추론 → argmax 예측** `logits = self.model(**enc).logits` /
  `logits.argmax(-1)`.

### (3) 메트릭 (규격서 산출식 일치 여부)
- **새 정책 라벨 산출(부채비율 85%)** — `cert/cert_credit3_regulation.py:93-99` `new_policy_labels`,
  `:40` `NEW_THRESHOLD = 85.0`, `:96` `band = decisive & (v > 85) & (v <= theta)`(θ=기존 200%),
  `:98` `y.loc[band] = 0` → 85%<부채비율≤200% 밴드 기업을 탈락(0)으로 재라벨. **[O]**
  - `decisive`(`:94`): 부채비율 외 8개 rule을 통과한 기업만 대상(`cc.other_rules_pass_mask`,
    `centroid_common.py:55`).
- **정확도 = (TP+TN)/N** — `cert/cert_credit3_regulation.py:136` `accuracy_score(y_new, preds)`. **[O]**
- **재현율 = TP/(TP+FN)** — `:137` `recall_score(y_new, preds, pos_label=1)`(통과),
  `:138` `pos_label=0`(탈락), `:139` macro. **[O]**
- **응답시간(1건씩, perf_counter)** — `cert/cert_credit3_regulation.py:158-162` 루프에서
  `:159` `t = time.perf_counter()` → `:160` `judge(sample.iloc[[i]])`(**정확히 1건** 입력:
  문장생성→토큰화→편집 hook 포함 추론→argmax 판정) → `:161` `torch.cuda.synchronize()` →
  `:162` `times.append(time.perf_counter() - t)`. **[O]**
  - **평균 계산**: `:164` `"mean_s": statistics.mean(times)` (중앙값·p95·최대도 `:164-165`). **[O]**
  - 워밍업 제외: `:154-156`(측정 전 warmup 반복 후 `sync()`).

---

## 요약 (규격서 산출식 대조)

| 항목 | 규격서 산출식 | 코드 위치 | 일치 |
|---|---|---|---|
| 신용1 정확도 | (TP+TN)/N | cert_credit1_bias.py:103,106 | O |
| 신용1 DEO | \|TPR_남−TPR_여\|, TPR=P(ŷ=1\|y=1) | cert_credit1_bias.py:102,112 | O |
| 신용3 정확도 | (TP+TN)/N | cert_credit3_regulation.py:136 | O |
| 신용3 재현율 | TP/(TP+FN) | cert_credit3_regulation.py:137-138 | O |
| 신용3 새 정책 라벨 | 85%<부채비율≤200% 밴드→탈락 | cert_credit3_regulation.py:40,96,98 | O |
| 신용3 응답시간 | 1건 perf_counter, 평균 | cert_credit3_regulation.py:159-162,164 | O |

모든 항목 규격서 산출식과 일치(O). 측정 결과값은 `SUMMARY.md` 참조.
