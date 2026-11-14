# KOLD 혐오 표현 탐지
KOLD 데이터(`KOLD/data/kold_v1.json`)로 KLUE BERT/RoBERTa를 fine-tuning해, 댓글(기사·영상 제목 포함)의 혐오 여부를 판별합니다(Level A, OFF vs NOT).
**목표: test 정확도 70% 이상.**

**Result** (Level A, test 4,043개, seed 5개 평균 ± 표준편차):

| Model | lr / epoch | Accuracy | F1 (OFF) |
|---|---|---|---|
| klue/bert-base | 5e-5 / 1 | 83.2 ± 0.3% | 83.1 |
| klue/roberta-base | 3e-5 / 2 | 83.4 ± 0.3% | 83.4 |
| klue/roberta-large | 1e-5 / 2 | **84.6 ± 0.2%** | 84.6 |

모든 run(3 모델 × 5 seed)이 70%를 넘었고, 가장 낮은 값은 82.6%입니다.
데이터는 80/10/10(train 32,343 / dev 4,043 / test 4,043)으로 나눴습니다. lr과 epoch은 dev F1 기준으로 골랐으며, test는 최종 평가에만 사용했습니다.

## Setup

[language_4](../language_4/README.md#setup)와 같은 `ai_dep_lang` conda 환경을 씁니다(torch 2.6.0, transformers 4.57.6, scikit-learn 1.7.2).
학습 1회(roberta-large 기준)는 RTX A6000 1장에서 약 6분 걸립니다. 다른 작업과 GPU를 같이 쓰면 최대 12분 정도 걸립니다.
같은 seed로 다시 학습하면 test 정확도가 똑같이 나옵니다(seed 1: 84.57%).

## Run

저장소 루트에서 실행합니다.

```bash
python language_2/prepare_data.py   # data/{train,dev,test}.jsonl 생성 (저장소에 이미 포함, seed 42로 동일하게 재생성됨)

CUDA_VISIBLE_DEVICES=0 python language_2/train.py --task A --model klue/roberta-large \
    --lr 1e-5 --epochs 2 --seed 1 --out language_2/runs/final/A/roberta-large/s1
```

학습이 끝나면 dev/test 점수가 출력되고 `--out` 디렉터리의 `metrics.json`에 저장됩니다. `test.accuracy` 값이 정확도입니다.
저장된 5개 seed의 정확도를 한 번에 확인하려면 다음을 실행합니다.

```bash
python -c "import json,glob; [print(f, round(json.load(open(f))['test']['accuracy'],2)) for f in sorted(glob.glob('language_2/runs/final/A/*/s*/metrics.json'))]"
```

- 논문 Table 5·6 전체(Level A/B/C, span)를 재현하려면 `python language_2/sweep.py --gpus 0,1,2,3`(lr·epoch 탐색 후 seed 5개 학습)과 `python language_2/aggregate.py`를 차례로 실행합니다. 결과는 `runs/results.md`에 저장됩니다.
- 학습한 모델은 `--save_model`로 저장할 수 있으며, git에는 올리지 않습니다(`.gitignore`).
