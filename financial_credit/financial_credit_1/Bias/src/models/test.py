"""Adult 프롬프트 분류기 평가 — Regulation test.py 이식 + 공정성(DEO) 지표.

저장물 (--output_dir):
- metrics.json        : accuracy/macro_f1/recall/macro_recall + DEO 2종(sex 기준)
- confusion_matrix.csv
- predictions.csv     : index, pred (GPU 없이 행 단위 재분석용)

DEO 2종 (Bias/reference/deo_github_vs_literature.md 참고):
- deo_repo : |Acc_male − Acc_female| — ai_dep 리포 구현과 동일 (비교용)
- deo_lit  : |TPR_male − TPR_female| (y=1 조건) — 문헌 표준 (보고용)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
)
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

BIAS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BIAS_ROOT))
from src.common.prompt import format_prompt


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the Adult prompt classifier.")
    parser.add_argument("--test_path", default=str(BIAS_ROOT / "data/test/text/test.csv"))
    parser.add_argument("--output_dir", default=str(BIAS_ROOT / "result/evaluation/base"))
    parser.add_argument("--model_name",
                        default=str(BIAS_ROOT / "outputs/finetuned_xlm-roberta-large/best"))
    parser.add_argument("--batch_size", type=int, default=128)
    return parser.parse_args()


def fairness_metrics(df, preds):
    y = df["label"].to_numpy()
    male = (df["sex"] == "Male").to_numpy()
    # 한쪽 성별이 없는 셋(고정 문맥 경계셋 등)에서는 공정성 지표 생략
    if male.all() or (~male).all() or (y[male] == 1).sum() == 0 or (y[~male] == 1).sum() == 0:
        return {}
    acc_m = float((preds[male] == y[male]).mean())
    acc_f = float((preds[~male] == y[~male]).mean())
    tpr_m = float((preds[male & (y == 1)] == 1).mean())
    tpr_f = float((preds[~male & (y == 1)] == 1).mean())
    return {
        "acc_male": acc_m, "acc_female": acc_f, "deo_repo": abs(acc_m - acc_f),
        "tpr_male": tpr_m, "tpr_female": tpr_f, "deo_lit": abs(tpr_m - tpr_f),
    }


def main():
    args = parse_args()
    df = pd.read_csv(args.test_path)
    tqdm.pandas(desc="Formatting prompts")
    prompts = df.progress_apply(format_prompt, axis=1).tolist()
    labels = df["label"].astype(int).to_numpy()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name).to(device)
    model.eval()

    preds = []
    with torch.no_grad():
        for start in tqdm(range(0, len(prompts), args.batch_size), desc="Predict"):
            enc = tokenizer(
                prompts[start : start + args.batch_size],
                return_tensors="pt", padding=True, truncation=True, max_length=256,
            ).to(device)
            preds.extend(model(**enc).logits.argmax(-1).cpu().tolist())
    preds = np.array(preds)

    metrics = {
        "n": int(len(df)),
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
        "recall": recall_score(labels, preds),
        "macro_recall": recall_score(labels, preds, average="macro"),
        **fairness_metrics(df, preds),
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)
    pd.DataFrame(
        confusion_matrix(labels, preds),
        index=["true_0", "true_1"], columns=["pred_0", "pred_1"],
    ).to_csv(out_dir / "confusion_matrix.csv")
    pd.DataFrame({"index": df.index, "pred": preds}).to_csv(
        out_dir / "predictions.csv", index=False
    )
    print(json.dumps(metrics, indent=2))
    print(f"saved: {out_dir}")


if __name__ == "__main__":
    main()
