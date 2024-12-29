# ALINPneumothorax V1.0

This study developed an AI model for pneumothorax detection in chest X-ray images. The model achieved high accuracy (93.38%), DEO (2.03), and maintained a low policy performance average drop (0.25%), demonstrating its potential for fair and reliable pneumothorax detection.

## Project Structure

```
.
├── data
│   ├── four_findings_expert_labels_test_labels.csv
│   └── four_findings_expert_labels_validation_labels.csv
├── train.py
└── README.md
```

## Installation

```bash
conda create -n alinpneumothorax python=3.8
conda activate alinpneumothorax

# Required libraries
pip install torch torchvision scikit-learn numpy tqdm pillow
```

## Model Load

You can download and use the pre-trained model from [the link](https://drive.google.com/drive/folders/1klrp0y24R6383bshHNCmHkLOd5tdfp1O?usp=sharing).

## Usage

```bash
python train.py
```

## Results

|             | Accuracy (%) | DEO   | Policy Drop (%) |
| :---------: | :----------: | :---: | :---------------------------------------: |
| **Ours**    |   **93.38**  | 2.03  | **0.25**                                  |