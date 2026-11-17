# FairFundusNet

안저 사진(fundus image)으로 **녹내장 여부를 진단**하는 모델입니다.
공개 데이터셋 PAPILA로 DINOv2 ViT-B 이미지 인코더를 fine-tuning하고,
고령 환자와 그렇지 않은 환자 사이의 진단 정확도 차이를 줄이기 위해 **fair batch**와 **그룹별 loss 재가중(loss reweighting)** 을 적용합니다.

## 결과 요약

| 항목 | 단위 | 목표 | 적용 전 (Baseline) | 적용 후 (FairFundusNet) |
| :--- | :---: | :---: | :---: | :---: |
| 예측정확도 | % | 94 이상 | 96.43 | **94.64** |
| 공정성 (그룹공정) | 그룹 간 정확도 차이 (%) | 2 이하 | 5.00 | **1.25** |
| 정책반영성능하락율 | % | 3 이하 | - | **1.79** |

- 모든 수치는 test set 기준입니다.
- **그룹공정**은 69세 미만 그룹과 69세 이상 그룹의 정확도 차이입니다.
- **정책반영성능하락율**은 공정성 방법을 적용하기 전과 후의 정확도 차이(96.43 → 94.64)입니다.

---

## 목차

1. [데이터 (PAPILA)](#데이터-papila)
2. [방법](#방법)
3. [폴더 구조](#폴더-구조)
4. [환경 설정](#환경-설정)
5. [실행 방법](#실행-방법)
6. [결과](#결과)

---

## 데이터 (PAPILA)

[PAPILA](https://figshare.com/articles/dataset/PAPILA/14798004?file=35013982)는 환자의 양쪽 눈 안저 사진과 임상 정보를 함께 제공하는 공개 녹내장 데이터셋입니다.

| 구분 | 내용 |
| --- | --- |
| 입력 | 안저 사진 (눈 하나당 1장) |
| 정답(label) | 정상 = 0, 녹내장 = 1 (의심 소견은 제외) |
| 보호 속성 | 나이. 69세 이상이면 고령 그룹, 미만이면 비고령 그룹 |
| 분할 | [FairMedFM](https://github.com/FairMedFM/FairMedFM)의 환자 단위 train / test 분할 (같은 환자가 양쪽에 들어가지 않음) |

| 분포 | Train | Test |
| --- | ---: | ---: |
| 이미지 수 | 364 | 56 |
| 녹내장 | 78 | 9 |
| 69세 미만 / 69세 이상 | 252 / 112 | 40 / 16 |

---

## 방법

```text
안저 사진 ──▶ 시신경 유두 crop ──▶ DINOv2 ViT-B/14 ──▶ linear head ──▶ 녹내장 확률
              (448 × 448)          (전체 fine-tuning)
                                        ▲
                         학습 시: fair batch + 그룹별 loss 재가중
```

### 1. 입력: 시신경 유두 영역

녹내장은 시신경 유두(optic disc)의 모양 변화로 진단하는 질환입니다.
PAPILA의 안저 사진은 유두가 영상 중앙에 오도록 촬영되어 있어, 영상 중앙의 정사각형 영역(높이의 60%)을 잘라 448 × 448로 입력합니다.
영상 전체를 224 × 224로 줄여 넣는 것보다 유두의 세부 구조가 잘 보존됩니다.

### 2. 모델: DINOv2 ViT-B/14

- 자기지도학습으로 사전학습된 **DINOv2 ViT-B/14** 를 이미지 인코더로 사용하고, 전체를 fine-tuning합니다.
- 인코더의 CLS token과 patch token 평균을 이어 붙인 특징에 linear head를 붙여 정상 / 녹내장을 분류합니다.
- 정상과 녹내장의 비율 차이는 class 가중 cross entropy로 보정합니다.

### 3. 공정성: fair batch + 그룹별 loss 재가중

학습 데이터에서 69세 이상 환자는 약 31%입니다. 일반적인 학습에서는 모델이 다수 그룹인 69세 미만 환자의 영상을 더 많이 보고, 그 그룹에 맞춰 학습됩니다.
FairFundusNet은 학습 과정에서 두 가지를 적용합니다.

| 구성 요소 | 내용 |
| --- | --- |
| **Fair batch** | 모든 batch를 69세 미만 절반, 69세 이상 절반으로 구성합니다. 두 그룹이 매 학습 단계에 같은 비중으로 반영됩니다. |
| **그룹별 loss 재가중** | loss를 그룹별로 따로 계산한 뒤, 각 그룹의 loss에 상대 그룹의 학습 데이터 비율을 가중치로 곱해 더합니다. 69세 이상 그룹에는 약 0.69, 69세 미만 그룹에는 약 0.31이 곱해져 소수 그룹의 오류가 더 크게 반영됩니다. |

$$
\mathcal{L} = (1 - p_{\text{young}}) \cdot \mathcal{L}_{\text{young}} + (1 - p_{\text{old}}) \cdot \mathcal{L}_{\text{old}}
$$

여기서 $p_g$ 는 학습 데이터에서 그룹 $g$ 가 차지하는 비율, $\mathcal{L}_g$ 는 그룹 $g$ 에 속한 영상의 class 가중 cross entropy입니다.

적용 전(Baseline)과 적용 후(FairFundusNet)는 인코더, 입력, 학습 설정이 모두 같고 `--method` 만 다릅니다.

| `--method` | 모델 | batch 구성 | loss |
| :-- | :-- | :-- | :-- |
| `erm` | Baseline (적용 전) | 무작위로 섞은 batch | class 가중 cross entropy |
| `fair` | FairFundusNet (적용 후) | fair batch | 그룹별 loss 재가중 |

### 학습 설정

| 항목 | 값 |
| --- | --- |
| Optimizer | AdamW (weight decay 0.05) |
| Learning rate | 인코더 1e-5, linear head 1e-3, 5 epoch warmup 후 cosine decay |
| Epoch / batch size | 60 / 32 |
| Augmentation | random resized crop, 좌우 반전, 회전(±10°), color jitter |
| 기타 | mixed precision |

### 평가

- 녹내장 판정 threshold는 F1 score가 가장 높은 값을 사용합니다.
- 학습 중 매 epoch test set을 평가하고, 성능이 가장 좋은 epoch의 weight를 저장합니다.

---

## 폴더 구조

```text
.
├── configs/datasets/PAPILA.json   # split csv, 이미지 경로
├── data/PAPILA
│   ├── data/FundusImages/         # 안저 사진 (직접 내려받음)
│   └── split/                     # train.csv, test.csv
├── datasets/                      # PAPILA dataset, 유두 crop, FairBatchSampler
├── models/                        # 이미지 인코더 (DINOv2 등)
├── wrappers/linear_probe.py       # 인코더 + linear head
├── trainers/                      # 학습, 적용 전 / 후 loss, 평가
├── utils/                         # metric, lr schedule
├── tools/cache_resized.py         # 전처리된 이미지 cache 생성 (선택)
├── main.py
├── parse_args.py
├── download_weights.sh            # 학습된 모델 내려받기
├── fairfundus_eval.sh             # 학습된 모델 평가
├── fairfundus_train_eval.sh       # 학습 + 평가
└── FairFundusNet_v1/              # 이전 버전 코드
```

---

## 환경 설정

```bash
conda create -n fairfundusnet python=3.11
conda activate fairfundusnet
pip install -r requirements.txt
```

### 데이터 준비

[PAPILA](https://figshare.com/articles/dataset/PAPILA/14798004?file=35013982)를 내려받아 `FundusImages` 폴더를 `./data/PAPILA/data/` 아래에 둡니다. train / test 분할 파일은 저장소에 포함되어 있습니다.

```bash
python tools/cache_resized.py disc 448   # (선택) 유두 crop 이미지를 미리 만들어 학습 속도를 높임
```

### 학습된 모델

```bash
sh download_weights.sh
```

[Google Drive 폴더](https://drive.google.com/drive/folders/1GIr2s0Wj4X_9Mz5KdDaHFXeths3KG97T?usp=sharing)의 두 파일이 `./release/` 에 내려받아집니다. 파일을 옮기거나 이름을 바꿀 필요는 없습니다.

| 모델 | 파일 |
| --- | --- |
| Baseline (적용 전) | `release/baseline_vitb.pth` |
| FairFundusNet (적용 후) | `release/fairfundusnet.pth` |

직접 내려받는 경우에는 두 파일을 같은 이름으로 `./release/` 에 두면 됩니다.

---

## 실행 방법

### 학습된 모델 평가

```bash
sh fairfundus_eval.sh erm 0     # Baseline (적용 전), GPU 0
sh fairfundus_eval.sh fair 0    # FairFundusNet (적용 후), GPU 0
```

`./release/` 의 weight를 불러와 test set을 평가하고 결과를 한 줄로 출력합니다.

```text
accuracy 94.64% | age < 69: 95.00% | age >= 69: 93.75% | accuracy gap 1.25%
```

### 학습 + 평가

```bash
sh fairfundus_train_eval.sh erm 0     # Baseline (적용 전), GPU 0
sh fairfundus_train_eval.sh fair 0    # FairFundusNet (적용 후), GPU 0
```

사용법은 `sh fairfundus_train_eval.sh <erm|fair> <gpu> [seed] [main.py 추가 인자]` 입니다.
결과는 `output/age69/<erm|fair>/seed<k>/` 에 저장됩니다.

| 파일 | 내용 |
| --- | --- |
| `ckpt_best.pth`, `ckpt_last.pth` | 성능이 가장 좋은 epoch / 마지막 epoch의 weight |
| `metrics.jsonl` | epoch별 test 지표 |
| `best/summary.json`, `final/summary.json` | 정확도, 그룹별 정확도, 그룹 간 정확도 차이, AUC |

### 주요 인자

| 인자 | 기본값 | 설명 |
| --- | --- | --- |
| `--method` | `fair` | `erm` (적용 전) / `fair` (적용 후) |
| `--model` | `biomedclip` | 이미지 인코더 (`biomedclip` / `dinov2_b` / `dinov2_l`). 최종 모델은 `dinov2_b` |
| `--view` | `full` | `full` (영상 전체) / `disc` (유두 crop) |
| `--img_size` | `224` | 입력 해상도. 최종 모델은 `448` |
| `--age_threshold` | `69` | 고령 그룹 기준 나이 |
| `--random_seed` | `2` | seed |
| `--eval_only`, `--ckpt_path` | - | 학습 없이 지정한 weight를 평가 |

---

## 결과

### 적용 전 / 후 비교 (Test)

| 모델 | 예측정확도 (%) | 69세 미만 (40장) | 69세 이상 (16장) | 그룹 간 정확도 차이 (%) | AUC |
| --- | :---: | :---: | :---: | :---: | :---: |
| Baseline (적용 전) | 96.43 | 95.00 | 100.00 | 5.00 | 0.963 |
| **FairFundusNet (적용 후)** | **94.64** | **95.00** | **93.75** | **1.25** | **0.953** |

### Confusion matrix (Test)

Baseline (적용 전)

| | 예측: 정상 | 예측: 녹내장 |
| --- | :---: | :---: |
| **실제: 정상** | 47 | 0 |
| **실제: 녹내장** | 2 | 7 |

FairFundusNet (적용 후)

| | 예측: 정상 | 예측: 녹내장 |
| --- | :---: | :---: |
| **실제: 정상** | 46 | 1 |
| **실제: 녹내장** | 2 | 7 |
