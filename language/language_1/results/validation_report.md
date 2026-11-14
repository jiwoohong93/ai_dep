# KOLD 데이터 검증 결과

- 데이터: `/home/seogyeong/ai_dep_lang/KOLD/data/kold_v1.json`
- 결과: **PASS** (PASS 99, FAIL 0, INFO 16)

INFO 항목은 논문 자체의 불일치 또는 공개 데이터로 재현 불가능한 수치로, 판정에 포함하지 않습니다.


## 00. 유효 샘플 (우선 검증)

| 상태 | 항목 | 논문/기대값 | 데이터 | 비고 |
|---|---|---|---|---|
| PASS | [규칙] R1a 댓글 텍스트 비어 있음 | 0 | 0 |  |
| INFO | [규칙] R1b 댓글 텍스트 중복 (첫 등장만 유효) | 0 | 221 | 같은 댓글이 다른 기사/영상에 달린 경우. (title, comment) 쌍 기준 중복은 별도 검사 |
| PASS | [규칙] R1c Context(title) 비어 있음 | 0 | 0 |  |
| PASS | [규칙] R2 OFF(혐오 여부) 라벨 없음/비정상 | 0 | 0 |  |
| PASS | [규칙] R3a OFF=True 인데 TGT(대상 유형) 라벨 없음 | 0 | 0 |  |
| PASS | [규칙] R3b TGT=group 인데 GRP(대상 집단) 라벨 없음 | 0 | 0 |  |
| PASS | [규칙] R3c 알 수 없는 GRP 라벨 | 0 | 0 |  |
| PASS | [규칙] R3d TGT!=group 인데 GRP 라벨 존재 | 0 | 0 |  |
| PASS | [규칙] R3e OFF=False 인데 TGT/GRP 라벨 존재 | 0 | 0 |  |
| PASS | [규칙] R4a 어노테이터 수 != 3 | 0 | 0 |  |
| PASS | [규칙] R4b OFF 가 3인 다수결과 불일치 | 0 | 0 |  |
| PASS | [규칙] (Context, 댓글) 쌍 중복 | 0 | 0 |  |
| INFO | 중복 댓글 중 라벨이 서로 다른 그룹 수 | - | 24 | 동일 텍스트라도 Context 에 따라 라벨이 달라질 수 있음 |
| PASS | 유효 샘플 수 >= 40,000 | True | True | 유효 40,208 / 전체 40,429 (99.45%), 제외 221 |
| PASS | 전체 샘플 수·라벨 분포 == 보고 수치 (전체 데이터) | {'total': 40429, 'NOT': 20119, 'OFF': 20310, 'UNT': 2596, 'IND': 3899, 'OTH': 1402, 'GRP': 12413} | {'total': 40429, 'NOT': 20119, 'OFF': 20310, 'UNT': 2596, 'IND': 3899, 'OTH': 1402, 'GRP': 12413} |  |
| INFO | 유효 샘플 기준 라벨 분포 | - | {'total': 40208, 'NOT': 19930, 'OFF': 20278, 'UNT': 2593, 'IND': 3898, 'OTH': 1395, 'GRP': 12392} | 중복 제거 후 분포. 논문 수치는 중복 포함 전체 기준 |

## 0. 스키마/무결성

| 상태 | 항목 | 논문/기대값 | 데이터 | 비고 |
|---|---|---|---|---|
| PASS | 모든 샘플의 필드 구성 | True | True |  |
| PASS | guid 중복 없음 | 40,429 | 40,429 |  |
| PASS | 빈 comment 없음 | 0 | 0 |  |
| PASS | 빈 title(Context) 없음 | 0 | 0 |  |
| PASS | OFF 는 bool | True | True |  |
| PASS | OFF=False 이면 TGT/GRP/span 없음 | 0 | 0 |  |
| PASS | OFF=True 이면 TGT 값이 유효 | 0 | 0 |  |
| PASS | GRP 라벨은 TGT=group 에만 존재 | 0 | 0 |  |
| PASS | 알 수 없는 GRP 라벨 없음 | [] | [] |  |

## 1. 기본 정보

| 상태 | 항목 | 논문/기대값 | 데이터 | 비고 |
|---|---|---|---|---|
| PASS | 전체 샘플 수 | 40,429 | 40,429 |  |
| PASS | 데이터 출처 (NAVER 뉴스, YouTube) | ['naver_news', 'youtube'] | ['naver_news', 'youtube'] |  |
| INFO | 플랫폼별 샘플 수 | 논문 미제시 | {'naver_news': 22028, 'youtube': 18401} |  |
| PASS | 날짜 누락 없음 | 0 | 0 |  |
| PASS | 콘텐츠 기간 (YYYY-MM) | ['2020-03', '2022-03'] | ['2020-03', '2022-03'] | 실제 2020-03-31 ~ 2022-03-31 |
| PASS | 한국어(한글 포함) 댓글 비율 >= 95% | True | True | 한글 포함 40,077/40,429 (99.13%) |

## 2. Level A (Offensiveness)

| 상태 | 항목 | 논문/기대값 | 데이터 | 비고 |
|---|---|---|---|---|
| PASS | NOT 샘플 수 | 20,119 | 20,119 | 49.8% |
| PASS | OFF 샘플 수 | 20,310 | 20,310 | 50.2% |
| PASS | NOT 비율(%) | 49.8 | 49.8 |  |
| PASS | OFF 비율(%) | 50.2 | 50.2 |  |
| INFO | Section 3.3 본문 OFF 수치(20,130) | 20,130 | 20,310 | 논문 오기. Table 2(20,310)가 데이터와 일치 |

## 3. Level B (Target Type)

| 상태 | 항목 | 논문/기대값 | 데이터 | 비고 |
|---|---|---|---|---|
| PASS | UNT 샘플 수 | 2,596 | 2,596 | 전체 대비 6.4% |
| PASS | IND 샘플 수 | 3,899 | 3,899 | 전체 대비 9.6% |
| PASS | OTH 샘플 수 | 1,402 | 1,402 | 전체 대비 3.5% |
| PASS | GRP 샘플 수 | 12,413 | 12,413 | 전체 대비 30.7% |
| PASS | UNT 전체 대비 비율(%) | 6.4 | 6.4 |  |
| PASS | IND 전체 대비 비율(%) | 9.6 | 9.6 |  |
| PASS | OTH 전체 대비 비율(%) | 3.5 | 3.5 |  |
| PASS | GRP 전체 대비 비율(%) | 30.7 | 30.7 |  |
| PASS | Level B 합계 == OFF | 20,310 | 20,310 |  |
| PASS | GRP (UNT 제외 기준, %) | 70.1 | 70.1 |  |
| PASS | IND (UNT 제외 기준, %) | 22.0 | 22.0 |  |
| PASS | OTH (UNT 제외 기준, %) | 7.9 | 7.9 |  |

## 4. Level C (Target Group Attribute)

| 상태 | 항목 | 논문/기대값 | 데이터 | 비고 |
|---|---|---|---|---|
| PASS | Level C 속성 합계 == GRP | 12,413 | 12,413 |  |
| INFO | Gender & Sexual Orientation 댓글 수 | 2,986 | 3,049 | 공개 v1 데이터 기준(첫 라벨 속성). 논문 7.4% / 데이터 7.5%. 논문 Table 2 의 속성별 수치는 v1 최종 라벨로 정확히 재현되지 않음 (Table 3 세부 수치는 재현됨) |
| INFO | Race, Ethnicity & Nationality 댓글 수 | 2,970 | 3,030 | 공개 v1 데이터 기준(첫 라벨 속성). 논문 7.3% / 데이터 7.5%. 논문 Table 2 의 속성별 수치는 v1 최종 라벨로 정확히 재현되지 않음 (Table 3 세부 수치는 재현됨) |
| INFO | Political Affiliation 댓글 수 | 1,939 | 1,985 | 공개 v1 데이터 기준(첫 라벨 속성). 논문 4.8% / 데이터 4.9%. 논문 Table 2 의 속성별 수치는 v1 최종 라벨로 정확히 재현되지 않음 (Table 3 세부 수치는 재현됨) |
| INFO | Religion 댓글 수 | 1,699 | 1,725 | 공개 v1 데이터 기준(첫 라벨 속성). 논문 4.2% / 데이터 4.3%. 논문 Table 2 의 속성별 수치는 v1 최종 라벨로 정확히 재현되지 않음 (Table 3 세부 수치는 재현됨) |
| INFO | Miscellaneous 댓글 수 | 2,819 | 2,624 | 공개 v1 데이터 기준(첫 라벨 속성). 논문 7.0% / 데이터 6.5%. 논문 Table 2 의 속성별 수치는 v1 최종 라벨로 정확히 재현되지 않음 (Table 3 세부 수치는 재현됨) |

## 5. Target Group (Table 3)

| 상태 | 항목 | 논문/기대값 | 데이터 | 비고 |
|---|---|---|---|---|
| PASS | [GEN] LGBTQ+ | 1,369 | 1,369 |  |
| PASS | [GEN] Women | 1,129 | 1,129 |  |
| PASS | [GEN] Men | 591 | 591 |  |
| PASS | [GEN] Others | 108 | 108 |  |
| PASS | [RACE] Chinese | 475 | 475 |  |
| PASS | [RACE] Korean-Chinese | 460 | 460 |  |
| PASS | [RACE] Black | 175 | 175 |  |
| PASS | [RACE] Others | 1,605 | 1,605 |  |
| PASS | [POL] Progressive | 1,052 | 1,052 |  |
| PASS | [POL] Conservative | 316 | 316 |  |
| PASS | [POL] Others | 651 | 651 |  |
| PASS | [REL] Muslim | 1,059 | 1,059 |  |
| PASS | [REL] Christian | 518 | 518 |  |
| PASS | [REL] Catholic | 57 | 57 |  |
| PASS | [REL] Others | 91 | 91 |  |
| PASS | [MISC] Feminist | 1,483 | 1,483 |  |
| PASS | [MISC] Socio-economic Status | 88 | 88 |  |
| PASS | [MISC] Agism | 62 | 62 |  |
| PASS | [MISC] Others | 971 | 971 |  |
| INFO | Table 3 집단 라벨 총수 | 12,981 | 12,703 | 논문 표기 12,981 은 Table 3 행 합(12,260) 및 데이터 라벨 총수와 불일치 |
| PASS | 집단 라벨 수 >= GRP 댓글 수 (복수 라벨) | True | True | 라벨 12,703 / 댓글 12,413 |

## 6. Target Group Top 10 (Table 4)

| 상태 | 항목 | 논문/기대값 | 데이터 | 비고 |
|---|---|---|---|---|
| PASS | Top 10 순위 | ['Feminist', 'LGBTQ+', 'Women', 'Muslim', 'Progressive', 'Men', 'Christian', 'Chinese', 'Korean-Chinese', 'Conservative'] | ['Feminist', 'LGBTQ+', 'Women', 'Muslim', 'Progressive', 'Men', 'Christian', 'Chinese', 'Korean-Chinese', 'Conservative'] |  |
| PASS | Feminist 샘플 수 | 1,483 | 1,483 |  |
| PASS | Feminist 비율(% of 12,981) | 11.42 | 11.42 |  |
| PASS | LGBTQ+ 샘플 수 | 1,369 | 1,369 |  |
| PASS | LGBTQ+ 비율(% of 12,981) | 10.55 | 10.55 |  |
| PASS | Women 샘플 수 | 1,129 | 1,129 |  |
| PASS | Women 비율(% of 12,981) | 8.7 | 8.7 |  |
| PASS | Muslim 샘플 수 | 1,059 | 1,059 |  |
| PASS | Muslim 비율(% of 12,981) | 8.16 | 8.16 |  |
| PASS | Progressive 샘플 수 | 1,052 | 1,052 |  |
| PASS | Progressive 비율(% of 12,981) | 8.1 | 8.1 |  |
| PASS | Men 샘플 수 | 591 | 591 |  |
| PASS | Men 비율(% of 12,981) | 4.55 | 4.55 |  |
| PASS | Christian 샘플 수 | 518 | 518 |  |
| PASS | Christian 비율(% of 12,981) | 3.99 | 3.99 |  |
| PASS | Chinese 샘플 수 | 475 | 475 |  |
| PASS | Chinese 비율(% of 12,981) | 3.66 | 3.66 |  |
| PASS | Korean-Chinese 샘플 수 | 460 | 460 |  |
| PASS | Korean-Chinese 비율(% of 12,981) | 3.54 | 3.54 |  |
| PASS | Conservative 샘플 수 | 316 | 316 |  |
| PASS | Conservative 비율(% of 12,981) | 2.43 | 2.43 |  |

## 7. Target Group 라벨 체계 (Table 10)

| 상태 | 항목 | 논문/기대값 | 데이터 | 비고 |
|---|---|---|---|---|
| PASS | Gender & Sexual Orientation (3개) | ['LGBTQ+', 'Men', 'Women'] | ['LGBTQ+', 'Men', 'Women'] |  |
| PASS | Race, Ethnicity & Nationality (7개) | ['Asian', 'Black', 'Chinese', 'Indian', 'Korean-Chinese', 'Southeast Asian', 'White'] | ['Asian', 'Black', 'Chinese', 'Indian', 'Korean-Chinese', 'Southeast Asian', 'White'] |  |
| PASS | Political Affiliation (2개) | ['Conservative', 'Progressive'] | ['Conservative', 'Progressive'] |  |
| PASS | Religion (4개) | ['Buddhism', 'Catholic', 'Christian', 'Islam'] | ['Buddhism', 'Catholic', 'Christian', 'Islam'] |  |
| PASS | Miscellaneous (6개) | ['Agism', 'Disabled', 'Diseased', 'Feminist', 'Physical Appearance', 'Socio-economic Status'] | ['Agism', 'Disabled', 'Diseased', 'Feminist', 'Physical Appearance', 'Socio-economic Status'] |  |
| PASS | 세부 집단 라벨 수 (Appendix 열거 기준) | 22 | 22 |  |
| INFO | 본문 표기 Target Group 수 | 21 | 22 | 논문 본문 21개 vs Appendix 22개 불일치 |
| PASS | 대분류(Attribute) 수 | 5 | 5 |  |

## 8. Span Annotation

| 상태 | 항목 | 논문/기대값 | 데이터 | 비고 |
|---|---|---|---|---|
| PASS | GRP 중 Target Span 없는 비율(%) | 55.1 | 55.1 | 6,837/12,413 |
| INFO | OFF_span 있는 댓글 수 | 논문 미제시 | 18,422 (OFF 대비 90.7%, 평균 길이 25.3자) |  |
| INFO | TGT_span 있는 댓글 수 | 논문 미제시 | 7,811 (OFF 대비 38.5%, 평균 길이 3.6자) |  |

## 9. 어노테이션

| 상태 | 항목 | 논문/기대값 | 데이터 | 비고 |
|---|---|---|---|---|
| PASS | 전체 어노테이터 수 | 3,124 | 3,124 |  |
| PASS | 샘플당 어노테이터 수 | {3: 40429} | {3: 40429} |  |
| PASS | Level A 최종 라벨 == 3인 다수결 | 0 | 0 |  |
| PASS | Level A Krippendorff's alpha | 0.55 | 0.5516 |  |
| INFO | Level B alpha / Level C alpha / 전체 평균 | 0.45 / 0.65 / 0.55 | 미계산 | 다중 선택 라벨 구조로 논문과 동일한 계산 방식 재현 불가 |

## 10. 비식별화

| 상태 | 항목 | 논문/기대값 | 데이터 | 비고 |
|---|---|---|---|---|
| PASS | 원문 URL 노출 없음 | 0 | 0 |  |
| PASS | 원문 이메일 노출 없음 | 0 | 0 |  |
| INFO | 치환 토큰 사용 수 | - | {'<user>': 3, '<url>': 60, '<email>': 0} |  |

## 계산된 원시 통계

```json
{
  "valid_samples": {
    "valid": 40208,
    "excluded": 221,
    "rule_violations": {
      "R1b 댓글 텍스트 중복 (첫 등장만 유효)": 221
    },
    "valid_level_a": {
      "NOT": 19930,
      "OFF": 20278
    },
    "valid_level_b": {
      "GRP": 12392,
      "UNT": 2593,
      "IND": 3898,
      "OTH": 1395
    },
    "duplicate_label_conflict_groups": 24
  },
  "total": 40429,
  "sources": {
    "naver_news": 22028,
    "youtube": 18401
  },
  "date_min": "2020-03-31",
  "date_max": "2022-03-31",
  "level_a": {
    "NOT": 20119,
    "OFF": 20310
  },
  "level_b": {
    "GRP": 12413,
    "UNT": 2596,
    "IND": 3899,
    "OTH": 1402
  },
  "level_c_first_label": {
    "MISC": 2624,
    "GEN": 3049,
    "POL": 1985,
    "REL": 1725,
    "RACE": 3030
  },
  "target_group_label_counts": {
    "GEN/LGBTQ+": 1369,
    "GEN/Men": 591,
    "GEN/Others": 108,
    "GEN/Women": 1129,
    "MISC/Agism": 62,
    "MISC/Disabled": 24,
    "MISC/Diseased": 8,
    "MISC/Feminist": 1483,
    "MISC/Others": 971,
    "MISC/Physical Appearance": 2,
    "MISC/Socio-economic Status": 88,
    "POL/Conservative": 316,
    "POL/Others": 651,
    "POL/Progressive": 1052,
    "RACE/Asian": 43,
    "RACE/Black": 175,
    "RACE/Chinese": 475,
    "RACE/Indian": 173,
    "RACE/Korean-Chinese": 460,
    "RACE/Others": 1605,
    "RACE/Southeast Asian": 81,
    "RACE/White": 95,
    "REL/Buddhism": 17,
    "REL/Catholic": 57,
    "REL/Christian": 518,
    "REL/Islam": 1059,
    "REL/Others": 91
  },
  "target_group_label_total": 12703,
  "grp_no_tgt_span": 6837,
  "annotators": 3124,
  "alpha_level_a": 0.5516
}
```
