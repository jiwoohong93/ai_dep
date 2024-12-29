# Credit Scoring with ALINGermanCredit V1.0

This study developed an AI model for credit scoring. The model achieved high accuracy (79.286%) and DEO (5.72), demonstrating its potential for fair and reliable credit scoring AI. The policy drop ratio was 3.90%.

## Project Structure

```
.
├── data
│   └── german.csv
├── train.py
├── eval.py
├── README.md
```

## Installation

Create a new environment (or use an existing one) and install the required libraries:
```
conda create -n alingermanscoring python=3.8
conda activate alingermanscoring

pip install pandas
pip install scikit-learn
pip install imbalanced-learn
pip install joblib
```

## Model Load

A pre-trained version of **ALINGermanCredit V1.0** is available at [this link](https://drive.google.com/drive/folders/12UtyUqOVihpsnvIvq3v5OLzPpqXv-Nup?usp=sharing).

## Usage

```
python train.py
python eval.py
```

- **train.py**: Trains two models (with fairness enhancement and with baseline).
- **eval.py**: Loads the saved models and evaluates them on test data, printing detailed metrics.

## Results

|                            | Accuracy (%)  | DEO    | Policy Drop (%) |
|:--------------------------:|:-------------:|:------:|:---------------------:|
| **With Fairness (SMOTE)**  | **79.286**    | **5.72** | **3.90**             |