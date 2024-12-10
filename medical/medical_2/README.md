# FairFundusNet

This study developed an AI model for glaucoma diagnosis using the PAPILA dataset of fundus images. FairFundusNet was fine-tuned with a focus on fairness, employing fair batch and loss reweighting techniques to mitigate age-related biases. The model achieved high diagnostic accuracy (94.64%) and fairness, demonstrating its potential for fair and reliable medical AI.



## Project Structure

```
.
├── configs
│   ├── datasets
│   │   ├── PAPILA.json
│   └── models
│       ├── BiomedCLIP.json
│       ├── BLIP2.json
│       ├── BLIP.json
│       ├── MedCLIP.json
│       └── PubMedCLIP.json
├── data
│   └── PAPILA
│       ├── data
│       │   ├── ClinicalData
│       │   │   ├── patient_data_od.csv
│       │   │   └── patient_data_os.csv
│       │   ├── FundusImages
│       │   ├── README.md
│       │   ├── test_age.csv
│       │   └── all.csv
│       └── split
│           ├── test.csv
│           └── train.csv
├── datasets
│   ├── PAPILA.py
│   └── utils.py
├── main.py
├── models
│   ├── biomed_clip.py
│   ├── blip2.py
│   ├── blip.py
│   ├── clip.py
│   └── utils.py
├── parse_args.py
├── pre-processing
│   └── classification
│       └── PAPILA.ipynb
├── README.md
├── requirements.txt
├── trainers
│   ├── base.py
│   ├── cls.py
│   └── utils.py
├── train_eval_lyh.sh
├── utils
│   ├── basics.py
│   ├── lr_sched.py
│   ├── metrics.py
│   ├── static.py
│   └── tokenizer.py
└── wrappers
    ├── base.py
    ├── clip.py
    ├── linear_probe.py
    └── utils.py
```

## Installation

```
conda create -n fairfundusnet python=3.11.10
conda activate fairfundusnet
pip install -r requirements.txt
```

## Dataset

Download PAPILA dataset and put it at './data/PAPILA/data' folder.

- [PAPILA](https://figshare.com/articles/dataset/PAPILA/14798004?file=35013982)



## Model Load

You can download and use the pre-trained baseline model and FairFundusNet from the link. 

- [Baseline Model (ViT-B) pth file](https://drive.google.com/drive/folders/1fJKEJE-6VjdrXebK-KF4_py91pEVgIUE?usp=sharing)
- [FairFundusNet](https://drive.google.com/drive/folders/1fJKEJE-6VjdrXebK-KF4_py91pEVgIUE?usp=sharing)



## Usage

```
sh fairfundus_train_eval.sh
```

## Results

|                   | 그룹 간 정확도 차이 (%) | 예측 정확도 (%) |
| :---------------: | :---------------------: | :-------------: |
| Baseline (ViT-B)  |          5.00           |      96.42      |
| **FairFundusNet** |        **1.25**         |    **94.64**    |

