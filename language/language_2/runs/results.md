# KOLD replication results (test set)

Level C 는 논문과 같이 세부 target group 27-class. span 점수는 OFF 댓글 대상, 문자 단위 F1 의 샘플 평균 (gold/pred 모두 비면 1, 한쪽만 비면 0). 마지막 열은 gold span 이 있는 샘플만으로 계산한 참고값.

## Table 5 (category prediction)

| Task | Model | lr / ep | Ours P | Ours R | Ours F1 (mean ± std) | Paper F1 | Δ F1 | seeds |
|---|---|---|---|---|---|---|---|---|
| Level A: Offensive | bert-base | 5e-05 / 1 | 83.8 | 82.4 | 83.1 ± 0.6 | 77.2 | +5.9 | 5 |
| Level A: Offensive | roberta-base | 3e-05 / 2 | 83.8 | 83.0 | 83.4 ± 0.4 | 77.2 | +6.2 | 5 |
| Level A: Offensive | roberta-large | 1e-05 / 2 | 85.0 | 84.2 | 84.6 ± 0.2 | 78.8 | +5.8 | 5 |
| Level B: Target Type | bert-base | 2e-05 / 3 | 63.6 | 61.4 | 62.3 ± 0.5 | 57.6 | +4.7 | 5 |
| Level B: Target Type | roberta-base | 1e-05 / 4 | 64.1 | 63.1 | 63.5 ± 0.5 | 59.4 | +4.1 | 5 |
| Level B: Target Type | roberta-large | 1e-05 / 4 | 65.4 | 64.9 | 65.1 ± 0.4 | 62.7 | +2.4 | 5 |
| Level C: Target Group | bert-base | 3e-05 / 5 | 64.5 | 57.8 | 59.0 ± 2.6 | 55.9 | +3.1 | 5 |
| Level C: Target Group | roberta-base | 5e-05 / 4 | 66.8 | 60.9 | 61.9 ± 1.7 | 57.3 | +4.6 | 5 |
| Level C: Target Group | roberta-large | 5e-05 / 4 | 65.6 | 62.3 | 63.0 ± 1.5 | 57.5 | +5.5 | 5 |

## Table 6 (span prediction)

| Task | Model | lr / ep | Ours P | Ours R | Ours F1 (mean ± std) | Paper F1 | Δ F1 | seeds | F1, gold span 있는 샘플만 |
|---|---|---|---|---|---|---|---|---|---|
| Offensive Span | bert-base | 2e-05 / 3 | 71.8 | 72.2 | 66.8 ± 0.5 | 41.9 | +24.9 | 5 | 72.1 |
| Offensive Span | roberta-base | 1e-05 / 3 | 72.3 | 73.0 | 67.4 ± 0.4 | 40.6 | +26.8 | 5 | 72.6 |
| Offensive Span | roberta-large | 1e-05 / 3 | 73.2 | 74.5 | 68.9 ± 0.3 | 45.4 | +23.5 | 5 | 74.1 |
| Target Span | bert-base | 5e-05 / 4 | 75.2 | 74.7 | 74.5 ± 0.2 | 61.9 | +12.6 | 5 | 65.1 |
| Target Span | roberta-base | 3e-05 / 2 | 75.6 | 75.3 | 75.0 ± 0.2 | 62.4 | +12.6 | 5 | 67.2 |
| Target Span | roberta-large | 3e-05 / 3 | 76.6 | 76.3 | 76.0 ± 0.3 | 62.5 | +13.5 | 5 | 66.5 |
