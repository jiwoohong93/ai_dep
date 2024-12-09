# KLUE-NER SNRoberta

This project implements Named Entity Recognition for the KLUE (Korean Language Understanding Evaluation) benchmark using an ensemble of RoBERTa models. The project includes multi-GPU training automation and ensemble evaluation with Stochastic Weight Averaging (SWA).

## Project Structure
```
.
├── README.md
├── environment.yaml
├── main_SWA_NER.py
├── eval_across_architecture.py
├── run_parallel_experiments-epochs=10_SWA_ensemble.sh
├── eval.sh
├── src/
│   ├── datasets.py
│   └── utils.py
└── data/
    └── klue-ner/
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/jhpark-kaist/KLUE-NER-SNRoberta
cd KLUE-NER-SNRoberta
```

2. Install required packages:
```bash
conda env create -f environment.yaml
conda activate kllm-fairness
```

## Usage

### Training
```bash
chmod +x run_parallel_experiments-epochs=10_SWA_ensemble.sh
./run_parallel_experiments-epochs=10_SWA_ensemble.sh
```

### Evaluation
```bash
chmod +x eval.sh
./eval.sh
```

## Results


### Ensemble Performance Across Seeds

| Seed | Entity F1  | Char F1    |
|------|------------|------------|
| 1    | 89.34      | 93.91      |
| 2    | 89.80      | 94.06      |
| 3    | 89.41      | 93.95      |
| 4    | 89.55      | 93.93      |
| 5    | 89.79      | 94.10      |
| 6    | 89.23      | 94.02      |
| 7    | 89.53      | 94.03      |
| 8    | 89.43      | 94.03      |

### Average Performance Metrics

| Metric    | Score ± Variance |
|-----------|-----------------|
| Entity F1 | 89.51 ± 0.19    |
| Char F1   | 94.00 ± 0.06    |


# Contact

Name : Joonhyeong Park <br/>
E-mail : jhpark.kaist@gmail.com <br/>
Affiliation : Statistical Inference and Machine Learning Lab @ KAIST <br/>







