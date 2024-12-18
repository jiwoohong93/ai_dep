# Credit Scoring

This study developed an AI model for credit scoring using the UCI Adult dataset. The model achieved high accuracy (81.00%) and DEO (0.069), demonstrating its potential for fair and reliable credit scoring AI.

## Project Structure

```
.
├── fairness
│   ├── dataloader.py
│   ├── metrics_fairness.py
│   ├── metrics.py
│   └── model.py
├── data
│   └── datasets
│       ├── uci_adult
│       │   ├── adult.data
│       │   ├── adult.test
│       │   ├── dataset_stats.json
│       │   ├── IPS_example_weights_with_label.json
│       │   ├── IPS_example_weights_without_label.json
│       │   ├── mean_std.json
│       │   ├── test.csv
│       │   ├── train.csv
│       │   └── vocabulary.json
├── main.py
├── train_fairness.py
├── train_nonfairness.py
├── inference_fairness.py
├── README.md
├── requirements.txt
```

## Installation

```
conda create -n faircreditscoring python=3.8
conda activate faircreditscoring
pip install -r requirements.txt
```

## Usage

```
python train_fairness.py
python inference_fairness.py
```

## Results

|                   | Accuracy (%)            | DEO             |
| :---------------: | :---------------------: | :-------------: |
| Baseline          |          81.76          |      0.138      |
| **Ours**          |        **81.00**        |    **0.069**    |

