# ALINJob V1.0

This study developed an AI model for recruitment prediction using a sampled dataset. Our RandomForest-based model achieved high accuracy (82.80%) and great fairness metric (DEO = 1.23). The policy performance drop was 1.10%, indicating a great trade-off between performance and fairness.

## Project Structure

```
.
├── train_fair.py
├── train_unfair.py
├── test.py
└── readme.md
```

- **train_fair.py**: Trains a classifier with fairness.
- **train_unfair.py**: Trains a classifier without fairness.
- **test.py**: Loads the trained models, evaluates performance metrics, and compares the fair vs. unfair approaches.

## Installation

Below are the key libraries required (tested on Python 3.8+):

```
pip install pandas numpy scikit-learn imblearn
```

## Model Load

You can download and use the pre-trained model from [this link](https://drive.google.com/drive/folders/1g5vdK21UvIUskZu88JpJNkEE3llAOvhu?usp=sharing).

## Usage

1. **Train with fairness:**
   ```
   python train_fair.py
   ```
2. **Train without fairness:**
   ```
   python train_unfair.py
   ```
3. **Evaluate both models:**
   ```
   python test.py
   ```

## Results

| Accuracy (%) | DEO  | Policy Drop (%) |
| :----------: | :--: | :---------------------: |
|    82.80     | 1.23 |          1.10%          |