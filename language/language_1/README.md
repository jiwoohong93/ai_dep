# KOLD 데이터 검증
KOLD(Korean Offensive Language Dataset, EMNLP 2022) 공개 데이터 `KOLD/data/kold_v1.json`이 유효한 샘플 4만 개 이상을 갖추고 있는지, 라벨이 빠짐없이 붙어 있는지, 논문이 보고한 샘플 수·라벨 분포와 일치하는지 검사합니다.

**Result** (PASS 99 / FAIL 0 / INFO 16):

| 검사 | 결과 |
|---|---|
| 유효 샘플 수 ≥ 40,000 | 유효 40,208개 / 전체 40,429개 (중복 댓글 221개 제외) |
| (1) 텍스트가 비어 있지 않고 중복이 아닌지 | 빈 댓글·빈 제목 0개, (제목, 댓글) 쌍 중복 0개. 댓글 텍스트만 같은 경우는 221개이며, 처음 나온 것만 유효 샘플로 셈 |
| (2) 혐오 여부(OFF) 라벨 | 누락·비정상 0개, 어노테이터 3명의 다수결과 불일치 0개 |
| (3) 혐오 샘플의 혐오 종류 라벨 | OFF=True인데 대상 유형(TGT)이 없는 샘플 0개, TGT=group인데 대상 집단(GRP)이 없는 샘플 0개, 알 수 없는 GRP 라벨 0개 |
| (4) 샘플 수·라벨 분포 == 논문 | 전체 40,429개, NOT 20,119 / OFF 20,310, UNT 2,596 / IND 3,899 / OTH 1,402 / GRP 12,413, 세부 대상 집단(Table 3)과 Top 10(Table 4) 모두 일치 |

INFO는 판정에서 제외한 참고 항목입니다. 논문 자체에 수치 불일치가 있거나 공개 v1 데이터만으로는 재현할 수 없는 경우입니다. 예를 들어 Table 2의 Level C 속성별 개수는 v1 데이터와 최대 약 7% 차이가 납니다. 반면 Table 3의 세부 수치는 정확히 일치합니다.

## Setup

표준 라이브러리만 사용합니다(Python ≥ 3.8). 데이터는 저장소에 포함된 `KOLD/data/kold_v1.json`을 씁니다.

## Run

```bash
python language_1/validate_kold.py      # FAIL이 하나라도 있으면 exit code 1
cat language_1/results/validation_report.md
```

- 항목별 결과는 `results/validation_report.{md,json}`에 저장되며, 위 (1)–(4)는 `00. 유효 샘플` 섹션에 있습니다.
- 다른 경로의 데이터를 검사하려면 `--data PATH`, 결과를 다른 위치에 저장하려면 `--out DIR` 옵션을 씁니다.

## Reference
- Jeong et al., *KOLD: Korean Offensive Language Dataset*, EMNLP 2022
