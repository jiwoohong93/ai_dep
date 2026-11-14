import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL_NAME = "yongukpark/xlm-roberta-large-rule"
BATCH_SIZE = 128


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the fine-tuned classifier.")
    parser.add_argument("--test_path", default="data/test/text/test.csv")
    parser.add_argument("--output_dir", default="result/base")
    parser.add_argument("--model_name", default=MODEL_NAME)
    return parser.parse_args()


def format_integer_like(value):
    if isinstance(value, (float, np.floating)) and value.is_integer():
        return str(int(value))
    return str(value)


def format_prompt(row):
    return (
        f"어떤 은행의 보통주자본비율이 {row['보통주자본비율(%)']}%, "
        f"기본자본비율이 {row['기본자본비율(%)']}%, "
        f"총자본비율이 {row['총자본비율(%)']}%, "
        f"단순기본자본비율이 {row['단순기본자본비율(%)']}%, "
        f"공제후자본양수가 {format_integer_like(row['공제후자본양수'])}, "
        f"총자산순이익률ROA가 {row['총자산순이익률ROA(%)']}%, "
        f"영업흑자가 {format_integer_like(row['영업흑자'])}, "
        f"이익잉여금양수가 {format_integer_like(row['이익잉여금양수'])}, "
        f"예대율이 {row['예대율(%)']}%입니다. "
        f"어떤 기업의 부채비율이 {row['부채비율(%)']}%, "
        f"유동비율이 {row['유동비율(%)']}%, "
        f"자기자본비율이 {row['자기자본비율(%)']}%이고, "
        f"자본총계가 {row['자본총계']}입니다. "
        "이 은행은 이 기업에게 대출할 수 있나요?"
    )


def main():
    args = parse_args()

    df = pd.read_csv(args.test_path, low_memory=False)
    df = df.dropna()  # 원본 csv 행 인덱스 유지 (predictions.csv의 index 컬럼)
    tqdm.pandas(desc="Formatting prompts")
    prompts = df.progress_apply(format_prompt, axis=1).tolist()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name).to(device)
    model.eval()

    predictions = []
    for start in tqdm(range(0, len(prompts), BATCH_SIZE), desc="Predicting"):
        inputs = tokenizer(
            prompts[start : start + BATCH_SIZE],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        ).to(device)

        with torch.no_grad():
            predictions.extend(model(**inputs).logits.argmax(dim=-1).cpu().tolist())

    labels = df["label"].astype(int).tolist()
    metrics = {
        "model_name": args.model_name,
        "test_path": args.test_path,
        "rows": len(labels),
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro"),
        "recall": recall_score(labels, predictions),
        "macro_recall": recall_score(labels, predictions, average="macro"),
    }
    matrix = pd.DataFrame(
        confusion_matrix(labels, predictions, labels=[0, 1]),
        index=["true_0", "true_1"],
        columns=["pred_0", "pred_1"],
    )

    print(f"device: {device}")
    print(f"rows: {len(labels):,}")
    print(f"accuracy: {metrics['accuracy']:.4f}")
    print(f"macro_f1: {metrics['macro_f1']:.4f}")
    print(f"recall: {metrics['recall']:.4f} (label=1) / macro {metrics['macro_recall']:.4f}")
    print("confusion matrix (rows=true, columns=predicted)")
    print(matrix)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    matrix.to_csv(output_dir / "confusion_matrix.csv")
    pd.DataFrame({"index": df.index, "pred": predictions}).to_csv(
        output_dir / "predictions.csv", index=False
    )
    print(f"saved: {output_dir}")


if __name__ == "__main__":
    main()
