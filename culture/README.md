# 한국어 문화 데이터셋 내 텍스트 편향성 검출

## 프로젝트 개요

본 프로젝트는 한국어 문화 데이터셋 내 텍스트 편향성 검출에 활용할 개체명 정보를
추출하기 위해 **Named Entity Recognition(NER)** 태스크를 수행합니다.
사전 학습된 언어 모델 **KoELECTRA v3** 및 **KoBERT**를 지원하며, 모델의 예측에
**고정 BIO 제약 디코딩**을 적용할 수 있습니다.
KoBERT 기학습 가중치에 고정 BIO 제약 디코딩을 적용하여 **Char F1 94.02%,
Entity F1 93.20%**를 확인했습니다.

## 개발 환경

저장소를 내려받고 실행 환경을 구성합니다. 아래 명령은 NVIDIA GPU와 CUDA를
사용하는 환경을 기준으로 합니다. 기존 `train.py`의 학습에는 CUDA GPU가 필요합니다.

```bash
git clone https://github.com/jiwoohong93/ai_dep.git
cd ai_dep

conda create -n culture_ner python=3.10 -y
conda activate culture_ner
python -m pip install torch==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r culture/requirements-eval.txt
python -m pip install wandb==0.28.0

python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

`CUDA available: True`가 출력되는지 확인합니다. 학습 코드에서 `wandb`를 불러오므로
패키지를 설치하며, 아래 예시에서는 `--no_wandb`를 사용하여 로그인 없이 실행합니다.
이후 명령은 모두 저장소 루트(`ai_dep/`)에서 실행합니다.

## 데이터 및 가중치

공식 [KLUE 데이터 저장소](https://github.com/KLUE-benchmark/KLUE)의
KLUE-NER v1.1 데이터를 사용합니다.

| 구분 | 파일명 | 문장 수 |
|---|---|---:|
| 학습 | `klue-ner-v1.1_train.tsv` | 21,008 |
| 검증 및 평가 | `klue-ner-v1.1_dev.tsv` | 5,000 |

```bash
mkdir -p data
git clone --depth 1 https://github.com/KLUE-benchmark/KLUE.git data/KLUE
ls -l data/KLUE/klue_benchmark/klue-ner-v1.1/klue-ner-v1.1_train.tsv
ls -l data/KLUE/klue_benchmark/klue-ner-v1.1/klue-ner-v1.1_dev.tsv
```

학습은 train 분할로 진행하고 dev 분할로 성능을 확인합니다.
학습 중 dev의 Char F1과 Entity F1이 각각 가장 높은 가중치를
`charf1_best.pth`, `entityf1_best.pth`로 저장합니다.
기학습 가중치가 있다면 학습 단계를 건너뛰고 평가 명령의 `--model_path`에 지정할 수 있습니다.

## 학습 실행

다음 명령은 KoBERT 학습 예시입니다.

```bash
python culture/train.py \
  --dataset_path data/KLUE/klue_benchmark/klue-ner-v1.1 \
  --model_name kobert \
  --save_path culture/runs \
  --epoch 200 \
  --bs 64 \
  --lr 5e-4 \
  --wd 0.01 \
  --max_seq_len 50 \
  --num_workers 4 \
  --no_wandb
```

KoELECTRA v3를 학습하려면 `--model_name`을 `koelectra-v3`로 변경합니다.

| 인자 | 예시 값 | 설명 |
|---|---|---|
| `--dataset_path` | `data/KLUE/klue_benchmark/klue-ner-v1.1` | train/dev TSV가 있는 디렉터리 |
| `--model_name` | `kobert` | 사용할 모델 |
| `--save_path` | `culture/runs` | 가중치를 저장할 상위 디렉터리 |
| `--epoch` | `200` | 학습 에포크 수 |
| `--bs` | `64` | 배치 크기 |
| `--lr` | `5e-4` | 분류기 학습률; 인코더에는 이 값의 1/100 적용 |
| `--wd` | `0.01` | Weight decay |
| `--max_seq_len` | `50` | 특수 토큰을 포함한 최대 입력 길이 |
| `--num_workers` | `4` | 데이터 로더 작업자 수 |
| `--no_wandb` | 지정 | W&B 기록 비활성화 |

위 명령으로 학습한 가중치는 다음 경로에 저장됩니다.

```text
culture/runs/KLUE-kobert-lr0.0005-wd0.01-bs64-ep200/
├── charf1_best.pth
└── entityf1_best.pth
```

## 평가 실행

학습한 가중치에 고정 BIO 제약 디코딩을 적용하여 dev 데이터에서 평가합니다.

```bash
python culture/test.py \
  --dataset_path data/KLUE/klue_benchmark/klue-ner-v1.1 \
  --model_name kobert \
  --model_path culture/runs/KLUE-kobert-lr0.0005-wd0.01-bs64-ep200/charf1_best.pth \
  --decoder legal_bio \
  --max_seq_len 50 \
  --bs 32 \
  --device cuda \
  --output_json culture/results/kobert_bio_eval.json
```

`--model_name`은 학습 시 사용한 모델과 동일하게 지정하고, `--model_path`에는
평가할 가중치 경로를 입력합니다. CPU에서 평가하려면 `--device cpu`로 변경합니다.

동일한 가중치로 기존 Argmax 방식과 비교하려면 다음 명령을 실행합니다.

```bash
python culture/test.py \
  --dataset_path data/KLUE/klue_benchmark/klue-ner-v1.1 \
  --model_name kobert \
  --model_path culture/runs/KLUE-kobert-lr0.0005-wd0.01-bs64-ep200/charf1_best.pth \
  --decoder argmax \
  --max_seq_len 50 \
  --bs 32 \
  --device cuda \
  --output_json culture/results/kobert_argmax_eval.json
```

이미 존재하는 결과 파일은 덮어쓰지 않습니다. 다시 평가할 때는 `--output_json`에
새 파일명을 지정합니다.

## 평가 결과 확인

평가가 끝나면 터미널에 Char/Entity Precision, Recall, F1 및 Char F1 목표 달성 여부가
출력됩니다. 저장한 JSON 파일에서도 결과를 확인할 수 있습니다.

```bash
python -m json.tool culture/results/kobert_bio_eval.json
```

| 항목 | 의미 |
|---|---|
| `metrics.char.precision` | Char Precision |
| `metrics.char.recall` | Char Recall |
| `metrics.char.f1` | Char F1 |
| `metrics.entity.f1` | Entity F1 |
| `dev_nonempty_sentences` | 평가한 dev 문장 수 |
| `target_met` | Char F1 0.94 이상이면 `true` |

JSON의 성능 지표는 0–1 범위이며, 백분율로 표시할 때는 100을 곱합니다.

## 평가 성능 및 기존 버전 비교

동일한 KoBERT 가중치(`kobert_culture.pth`)와 공식 KLUE-NER v1.1 dev 5,000문장을
사용하여, 기존 버전과 고정 BIO 제약 디코딩을 적용한 현재 버전을 비교했습니다.

| 지표 | 기존 버전: Argmax (%) | 현재 버전: 고정 BIO (%) | 개선폭 (%p) |
|---|---:|---:|---:|
| Char Precision | 93.4937 | 93.7883 | +0.2946 |
| Char Recall | 94.0447 | 94.2558 | +0.2111 |
| **Char F1** | **93.7684** | **94.0215** | **+0.2531** |
| **Entity F1** | **92.7027** | **93.1987** | **+0.4960** |

현재 버전은 **Char F1 94.0% 이상**의 목표를 달성했습니다.

| 구분 | 기존 버전 | 현재 버전 |
|---|---|---|
| 태그 선택 | 토큰별로 가장 높은 점수의 태그 선택 | BIO 규칙을 만족하는 전체 시퀀스 중 점수 합이 가장 높은 경로 선택 |
| BIO 제약 | 별도 제약 없음 | 문장 시작 및 태그 간 전이에 고정 BIO 제약 적용 |
| 모델 가중치 | 기존 KoBERT 가중치 | 동일한 가중치 |
| 전처리 및 평가 지표 | 기존 코드 사용 | 기존 방식 유지 |

성능 개선은 동일한 모델 출력에 고정 BIO 제약 디코딩을 적용한 결과이며,
이를 위한 추가 학습은 수행하지 않았습니다.

## 고정 BIO 제약 디코딩

BIO 태그에서 `B-X`는 개체의 시작, `I-X`는 같은 유형의 개체가 이어지는 부분,
`O`는 개체에 속하지 않는 부분을 뜻합니다.

- 문장의 첫 태그로 `I-X`가 나올 수 없습니다.
- `I-X`는 같은 유형의 `B-X` 또는 `I-X` 뒤에만 올 수 있습니다.
- Viterbi 알고리즘으로 위 규칙을 만족하면서 모델 점수 합이 가장 높은 경로를 선택합니다.

전이 규칙은 고정되어 있으며, 별도로 학습하는 파라미터는 없습니다.

## 검증

```bash
python -m unittest discover -s culture/tests -v
```

BIO 제약, 디코딩 결과, 가중치 및 입력 검증, 결과 파일 보존 등을 확인합니다.
